#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""优先级一基础设施接入测试

覆盖：
- UndoManager + 4 个 Command 类的 execute/undo 行为
- 撤销/重做接入 _family.py / _social.py / pipeline.py
- 会话恢复 _check_autosave_recovery
- 最近文件列表 _add_recent_file / _update_recent_files_menu
- 进度取消 _is_cancelled / _open_progress_popup cancel_event
- UI 布局修复（grid 按钮、minsize、状态栏文本精简）
"""

import copy
import os
import sys
import threading
import unittest
from unittest import mock

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from tb_risk.gui.undo import (
    UndoManager, AddContactCommand, EditContactCommand,
    DeleteContactCommand, ImportDataCommand,
)
from tb_risk.assessment import TB_Risk_Assessment


# ===========================================================================
# 辅助 fixture
# ===========================================================================

class _FakeApp:
    """最小可用 GUI app 桩对象"""

    def __init__(self):
        self.family_entries = []
        self.social_entries = []
        self.family_members = []
        self.social_contacts = []
        self.contact_labels = {}
        self._contact_id_counter = 0
        self.ui_mode = 'classic'
        self.adapter = None
        self.auto_save = mock.MagicMock()
        self.undo_manager = UndoManager(max_size=50)
        self.config = mock.MagicMock()
        self.config.recent_files = []

    def _update_family_treeview(self):
        pass

    def _update_social_treeview(self):
        pass

    def _update_fill_progress(self):
        pass

    def _mark_results_stale(self):
        pass

    def _update_gui_from_import(self):
        pass

    def _add_recent_file(self, file_path):
        pass


# ===========================================================================
# 1. UndoManager + Command 类单元测试
# ===========================================================================

class TestUndoManager:
    """UndoManager 核心行为测试"""

    def test_undo_manager_initial_state(self):
        mgr = UndoManager(max_size=50)
        assert mgr.can_undo is False
        assert mgr.can_redo is False

    def test_execute_and_undo(self):
        mgr = UndoManager(max_size=10)
        app = _FakeApp()
        entry = {'name': '张三', 'age': 30}
        cmd = AddContactCommand(app, 'family', entry)
        mgr.execute(cmd)
        assert len(app.family_entries) == 1
        assert mgr.can_undo is True
        assert mgr.undo() is True
        assert len(app.family_entries) == 0
        assert mgr.can_redo is True

    def test_redo_after_undo(self):
        mgr = UndoManager(max_size=10)
        app = _FakeApp()
        entry = {'name': '李四', 'age': 25}
        cmd = AddContactCommand(app, 'family', entry)
        mgr.execute(cmd)
        mgr.undo()
        assert len(app.family_entries) == 0
        mgr.redo()
        assert len(app.family_entries) == 1
        assert app.family_entries[0]['name'] == '李四'

    def test_max_size_limit(self):
        mgr = UndoManager(max_size=3)
        app = _FakeApp()
        for i in range(5):
            mgr.execute(AddContactCommand(app, 'family', {'name': f'p{i}'}))
        # 超出 max_size 后，最旧命令被丢弃
        assert len(mgr._undo_stack) <= 3

    def test_clear(self):
        mgr = UndoManager(max_size=10)
        app = _FakeApp()
        mgr.execute(AddContactCommand(app, 'family', {'name': 'x'}))
        mgr.clear()
        assert mgr.can_undo is False
        assert mgr.can_redo is False


class TestAddContactCommand:
    """AddContactCommand 测试"""

    def test_execute_append_family(self):
        app = _FakeApp()
        entry = {'name': '王五', 'age': 40}
        cmd = AddContactCommand(app, 'family', entry)
        cmd.execute()
        assert len(app.family_entries) == 1
        assert app.family_entries[0]['name'] == '王五'

    def test_execute_append_social(self):
        app = _FakeApp()
        entry = {'name': '赵六', 'age': 35}
        cmd = AddContactCommand(app, 'social', entry)
        cmd.execute()
        assert len(app.social_entries) == 1
        assert app.social_entries[0]['name'] == '赵六'

    def test_undo_removes_last(self):
        app = _FakeApp()
        app.family_entries.append({'name': 'existing'})
        cmd = AddContactCommand(app, 'family', {'name': 'new'})
        cmd.execute()
        assert len(app.family_entries) == 2
        cmd.undo()
        assert len(app.family_entries) == 1
        assert app.family_entries[0]['name'] == 'existing'

    def test_deep_copy_isolation(self):
        """确保 undo 不受外部 entry 变更影响"""
        app = _FakeApp()
        entry = {'name': '原值', 'age': 30}
        cmd = AddContactCommand(app, 'family', entry)
        cmd.execute()
        # 外部修改原 dict 不影响已压栈的数据
        entry['name'] = '被篡改'
        cmd.undo()
        cmd.redo()
        # redo 后的数据应为原始值
        assert app.family_entries[-1]['name'] == '原值'


class TestEditContactCommand:
    """EditContactCommand 测试"""

    def test_execute_replace(self):
        app = _FakeApp()
        app.family_entries.append({'name': '旧名', 'age': 20})
        new_entry = {'name': '新名', 'age': 25}
        cmd = EditContactCommand(app, 'family', 0, new_entry)
        cmd.execute()
        assert app.family_entries[0]['name'] == '新名'
        assert app.family_entries[0]['age'] == 25

    def test_undo_restores_old(self):
        app = _FakeApp()
        app.family_entries.append({'name': '旧名', 'age': 20})
        cmd = EditContactCommand(app, 'family', 0, {'name': '新名', 'age': 25})
        cmd.execute()
        cmd.undo()
        assert app.family_entries[0]['name'] == '旧名'
        assert app.family_entries[0]['age'] == 20


class TestDeleteContactCommand:
    """DeleteContactCommand 测试"""

    def test_execute_pop(self):
        app = _FakeApp()
        app.family_entries.extend([{'name': 'A'}, {'name': 'B'}, {'name': 'C'}])
        cmd = DeleteContactCommand(app, 'family', 1)
        cmd.execute()
        assert len(app.family_entries) == 2
        assert app.family_entries[1]['name'] == 'C'

    def test_undo_reinsert(self):
        app = _FakeApp()
        app.family_entries.extend([{'name': 'A'}, {'name': 'B'}, {'name': 'C'}])
        cmd = DeleteContactCommand(app, 'family', 1)
        cmd.execute()
        cmd.undo()
        assert len(app.family_entries) == 3
        assert app.family_entries[1]['name'] == 'B'


class TestImportDataCommand:
    """ImportDataCommand 测试"""

    def test_execute_replace_all(self):
        app = _FakeApp()
        app.family_entries = [{'name': 'old_family'}]
        app.social_entries = [{'name': 'old_social'}]
        new_family = [{'name': 'new_fam1'}, {'name': 'new_fam2'}]
        new_social = [{'name': 'new_soc1'}]
        cmd = ImportDataCommand(app, new_family, new_social)
        cmd.execute()
        assert len(app.family_entries) == 2
        assert app.family_entries[0]['name'] == 'new_fam1'
        assert len(app.social_entries) == 1
        assert app.social_entries[0]['name'] == 'new_soc1'

    def test_undo_restores_old_state(self):
        app = _FakeApp()
        app.family_entries = [{'name': 'old_family'}]
        app.social_entries = [{'name': 'old_social'}]
        cmd = ImportDataCommand(app, [{'name': 'new'}], [])
        cmd.execute()
        cmd.undo()
        assert app.family_entries[0]['name'] == 'old_family'
        assert app.social_entries[0]['name'] == 'old_social'

    def test_deep_copy_isolation_on_undo(self):
        """undo 恢复的数据与原始对象独立"""
        app = _FakeApp()
        original = [{'name': 'original'}]
        app.family_entries = original
        cmd = ImportDataCommand(app, [{'name': 'new'}], [])
        cmd.execute()
        # 修改原始 list 不影响 undo 恢复
        original.clear()
        cmd.undo()
        assert app.family_entries[0]['name'] == 'original'


# ===========================================================================
# 2. Pipeline _load_data 接入 ImportDataCommand 测试
# ===========================================================================

class TestPipelineUndoIntegration:
    """导入管线 _load_data 接入 UndoManager 测试"""

    def test_load_data_replace_pushes_undo(self):
        """replace 模式应通过 ImportDataCommand 压入撤销栈"""
        from tb_risk.gui.import_panel.pipeline import CSVImportPipeline
        app = _FakeApp()
        app.family_entries = [{'name': 'old'}]
        app.social_entries = []
        pipeline = CSVImportPipeline(app)
        pipeline._source_file_path = '/fake/path.csv'
        new_family = [{'name': 'imported1'}]
        new_social = [{'name': 'imported_s'}]
        with mock.patch.object(app, '_update_gui_from_import'), \
             mock.patch.object(app, '_update_family_treeview'), \
             mock.patch.object(app, '_update_social_treeview'), \
             mock.patch.object(app, '_update_fill_progress'), \
             mock.patch.object(app, '_mark_results_stale'), \
             mock.patch.object(app, '_add_recent_file'), \
             mock.patch('tb_risk.gui.import_panel.pipeline.messagebox'):
            result = pipeline._load_data(new_family, new_social, 'replace')
        assert result is True
        assert app.undo_manager.can_undo is True
        # 撤销应恢复旧数据
        app.undo_manager.undo()
        assert app.family_entries[0]['name'] == 'old'

    def test_load_data_append_mode(self):
        """append 模式应追加并支持撤销"""
        from tb_risk.gui.import_panel.pipeline import CSVImportPipeline
        app = _FakeApp()
        app.family_entries = [{'name': 'existing'}]
        app.social_entries = []
        pipeline = CSVImportPipeline(app)
        pipeline._source_file_path = '/fake/path.csv'
        with mock.patch.object(app, '_update_gui_from_import'), \
             mock.patch.object(app, '_update_family_treeview'), \
             mock.patch.object(app, '_update_social_treeview'), \
             mock.patch.object(app, '_update_fill_progress'), \
             mock.patch.object(app, '_mark_results_stale'), \
             mock.patch.object(app, '_add_recent_file'), \
             mock.patch('tb_risk.gui.import_panel.pipeline.messagebox'):
            pipeline._load_data([{'name': 'new'}], [], 'append')
        assert len(app.family_entries) == 2
        app.undo_manager.undo()
        assert len(app.family_entries) == 1
        assert app.family_entries[0]['name'] == 'existing'

    def test_load_data_merge_mode(self):
        """merge 模式应去重合并并支持撤销"""
        from tb_risk.gui.import_panel.pipeline import CSVImportPipeline
        app = _FakeApp()
        app.family_entries = [{'name': '张三', 'age': 30}]
        app.social_entries = []
        pipeline = CSVImportPipeline(app)
        pipeline._source_file_path = '/fake/path.csv'
        # 同名同年龄应去重
        with mock.patch.object(app, '_update_gui_from_import'), \
             mock.patch.object(app, '_update_family_treeview'), \
             mock.patch.object(app, '_update_social_treeview'), \
             mock.patch.object(app, '_update_fill_progress'), \
             mock.patch.object(app, '_mark_results_stale'), \
             mock.patch.object(app, '_add_recent_file'), \
             mock.patch('tb_risk.gui.import_panel.pipeline.messagebox'):
            pipeline._load_data(
                [{'name': '张三', 'age': 30}, {'name': '李四', 'age': 25}], [], 'merge')
        assert len(app.family_entries) == 2  # 张三去重 + 李四新增
        app.undo_manager.undo()
        assert len(app.family_entries) == 1

    def test_source_file_path_recorded(self):
        """CSV 管线 select_source 应记录 _source_file_path"""
        from tb_risk.gui.import_panel.pipeline import CSVImportPipeline
        app = _FakeApp()
        pipeline = CSVImportPipeline(app)
        with mock.patch('tb_risk.gui.import_panel.pipeline.filedialog') as fd:
            fd.askopenfilename.return_value = '/fake/test.csv'
            source = pipeline.select_source()
        assert source == {'file_path': '/fake/test.csv'}
        assert pipeline._source_file_path == '/fake/test.csv'


# ===========================================================================
# 3. _add_recent_file + _update_recent_files_menu 测试
# ===========================================================================

class TestRecentFiles:
    """最近文件列表功能测试（调用真实 TB_Risk_Assessment._add_recent_file）"""

    def test_add_recent_file_inserts_at_head(self):
        app = _FakeApp()
        TB_Risk_Assessment._add_recent_file(app, '/path/a.csv')
        TB_Risk_Assessment._add_recent_file(app, '/path/b.csv')
        assert app.config.recent_files[0] == '/path/b.csv'
        assert app.config.recent_files[1] == '/path/a.csv'

    def test_add_recent_file_dedupes(self):
        app = _FakeApp()
        TB_Risk_Assessment._add_recent_file(app, '/path/a.csv')
        TB_Risk_Assessment._add_recent_file(app, '/path/b.csv')
        TB_Risk_Assessment._add_recent_file(app, '/path/a.csv')  # 重复
        assert len(app.config.recent_files) == 2
        assert app.config.recent_files[0] == '/path/a.csv'  # 移到头部

    def test_add_recent_file_limits_to_10(self):
        app = _FakeApp()
        for i in range(15):
            TB_Risk_Assessment._add_recent_file(app, f'/path/file_{i}.csv')
        assert len(app.config.recent_files) == 10
        assert app.config.recent_files[0] == '/path/file_14.csv'

    def test_add_recent_file_empty_path_noop(self):
        app = _FakeApp()
        TB_Risk_Assessment._add_recent_file(app, '')
        assert len(app.config.recent_files) == 0

    def test_add_recent_file_none_noop(self):
        app = _FakeApp()
        TB_Risk_Assessment._add_recent_file(app, None)
        assert len(app.config.recent_files) == 0


# ===========================================================================
# 4. _is_cancelled 进度取消测试
# ===========================================================================

class TestProgressCancel:
    """进度对话框取消功能测试（调用真实 TB_Risk_Assessment._is_cancelled）"""

    def test_is_cancelled_no_event(self):
        """无 cancel_event 时返回 False"""
        app = _FakeApp()
        app._progress_cancel_event = None
        assert TB_Risk_Assessment._is_cancelled(app) is False

    def test_is_cancelled_event_not_set(self):
        """cancel_event 未触发时返回 False"""
        app = _FakeApp()
        event = threading.Event()
        app._progress_cancel_event = event
        assert TB_Risk_Assessment._is_cancelled(app) is False

    def test_is_cancelled_event_set(self):
        """cancel_event 触发后返回 True"""
        app = _FakeApp()
        event = threading.Event()
        event.set()
        app._progress_cancel_event = event
        assert TB_Risk_Assessment._is_cancelled(app) is True

    def test_is_cancelled_no_attribute(self):
        """无 _progress_cancel_event 属性时不崩溃"""
        app = _FakeApp()
        # _FakeApp 默认不设置 _progress_cancel_event，getattr 应返回 None
        cancel_event = getattr(app, '_progress_cancel_event', None)
        assert cancel_event is None
        assert TB_Risk_Assessment._is_cancelled(app) is False


# ===========================================================================
# 5. UI 布局修复验证测试
# ===========================================================================

class TestUILayoutFixes:
    """UI 布局修复验证（静态检查，无需 tkinter 显示）"""

    def test_base_minsize_is_1024x680(self):
        """验证最小窗口尺寸不低于 1024x680"""
        import inspect
        from tb_risk.gui import base
        source = inspect.getsource(base)
        assert 'minsize(1100, 720)' in source
        assert 'minsize(900, 600)' not in source

    def test_status_bar_text_simplified(self):
        """验证底部状态栏文本已精简"""
        import inspect
        from tb_risk.gui import base
        source = inspect.getsource(base)
        # 精简后不应包含完整快捷键列表
        assert 'Ctrl+N 添加接触者' not in source
        assert 'Ctrl+Q 退出' not in source
        # 应包含精简版本
        assert 'F5 评估' in source

    def test_ml_button_bar_uses_grid(self):
        """验证 ML 按钮栏使用 grid 而非 pack"""
        import inspect
        from tb_risk.gui.results_panel import init_ui
        source = inspect.getsource(init_ui)
        # 检查 LabelFrame 包裹
        assert 'LabelFrame(ml_tab, text="ML 操作"' in source
        # 检查 grid_columnconfigure（均匀分布）
        assert 'grid_columnconfigure(i, weight=1)' in source

    def test_ml_button_text_shortened(self):
        """验证 ML 按钮文本已缩短（检查按钮 text= 而非警告文本）"""
        import inspect
        from tb_risk.gui.results_panel import init_ui
        source = inspect.getsource(init_ui)
        # 按钮文本不应使用未缩短的长文本（检查 text="..." 形式）
        assert 'text="重新训练模型"' not in source
        assert 'text="真实vs合成数据对比"' not in source
        # 应使用缩短后的按钮文本
        assert 'text="重新训练"' in source
        assert 'text="真实vs合成对比"' in source

    def test_wizard_buttons_no_emoji(self):
        """验证向导按钮已移除 emoji 前缀"""
        import inspect
        from tb_risk.gui.wizard import _steps
        source = inspect.getsource(_steps)
        # 不应有 emoji 前缀
        assert '➕ 添加家庭成员' not in source
        assert '✏️ 编辑选中' not in source
        assert '🗑️ 删除选中' not in source
        # 应有缩短后的文本
        assert '"添加成员"' in source
        assert '"添加接触者"' in source

    def test_classic_family_buttons_grid(self):
        """验证 Classic 家庭按钮栏使用 pack(side='left') 自然排列"""
        import inspect
        from tb_risk.gui.tabs import _family
        source = inspect.getsource(_family)
        assert '添加家庭成员' not in source or 'text="添加成员"' in source
        # 改进3：按钮使用 pack(side='left') 自然排列而非grid
        assert "pack(side='left'" in source
        assert 'grid_columnconfigure(i, weight=1)' not in source

    def test_wizard_nav_padx_reduced(self):
        """验证向导导航栏 padx 从 20 减小为 8"""
        import inspect
        from tb_risk.gui.wizard import _layout
        source = inspect.getsource(_layout)
        assert 'padx=8' in source
        assert 'padx=20' not in source


# ===========================================================================
# 6. 撤销栈接入 family/social 操作的集成测试
# ===========================================================================

class TestUndoIntegrationWithTabs:
    """验证 _family.py / _social.py 中 undo_manager.execute 调用"""

    def test_family_py_uses_undo_manager(self):
        """_family.py 的 add/edit/delete 方法调用 undo_manager.execute"""
        import inspect
        from tb_risk.gui.tabs import _family
        source = inspect.getsource(_family)
        assert 'undo_manager.execute(AddContactCommand' in source
        assert 'undo_manager.execute(EditContactCommand' in source
        assert 'undo_manager.execute(DeleteContactCommand' in source

    def test_social_py_uses_undo_manager(self):
        """_social.py 的 add/edit/delete 方法调用 undo_manager.execute"""
        import inspect
        from tb_risk.gui.tabs import _social
        source = inspect.getsource(_social)
        assert 'undo_manager.execute(AddContactCommand' in source
        assert 'undo_manager.execute(EditContactCommand' in source
        assert 'undo_manager.execute(DeleteContactCommand' in source

    def test_family_py_marks_dirty(self):
        """_family.py 操作后标记 auto_save.mark_dirty()"""
        import inspect
        from tb_risk.gui.tabs import _family
        source = inspect.getsource(_family)
        assert 'auto_save.mark_dirty()' in source

    def test_social_py_marks_dirty(self):
        """_social.py 操作后标记 auto_save.mark_dirty()"""
        import inspect
        from tb_risk.gui.tabs import _social
        source = inspect.getsource(_social)
        assert 'auto_save.mark_dirty()' in source

    def test_pipeline_uses_import_data_command(self):
        """pipeline._load_data 使用 ImportDataCommand"""
        import inspect
        from tb_risk.gui.import_panel import pipeline
        source = inspect.getsource(pipeline)
        assert 'ImportDataCommand' in source
        assert 'undo_manager.execute' in source


# ===========================================================================
# 7. 会话恢复测试
# ===========================================================================

class TestSessionRecovery:
    """_check_autosave_recovery 功能测试"""

    def test_check_autosave_recovery_method_exists(self):
        """验证 _check_autosave_recovery 方法已定义"""
        assert hasattr(TB_Risk_Assessment, '_check_autosave_recovery')

    def test_check_autosave_recovery_no_autosave(self):
        """无 autosave 文件时应安全返回"""
        app = _FakeApp()
        with mock.patch('tb_risk.gui.app_config.AutoSaveManager.has_autosave',
                        return_value=False):
            # 直接调用方法（不经过 __init__）
            TB_Risk_Assessment._check_autosave_recovery(app)
        # 无异常即通过

    def test_check_autosave_recovery_user_declines(self):
        """用户拒绝恢复时应清除 autosave 文件"""
        app = _FakeApp()
        app.family_entries = []
        app.social_entries = []
        app.basic_info_vars = {}
        app.contact_labels = {}
        with mock.patch('tb_risk.gui.app_config.AutoSaveManager.has_autosave',
                        return_value=True), \
             mock.patch('tb_risk.gui.app_config.AutoSaveManager.load_autosave',
                        return_value={'family_entries': [{'name': 'x'}],
                                      'social_entries': [],
                                      'contact_labels': {},
                                      'basic_info_vars': {}}), \
             mock.patch('tb_risk.gui.app_config.AutoSaveManager.clear_autosave') as clear_mock, \
             mock.patch('tb_risk.assessment.messagebox.askyesno',
                        return_value=False):
            TB_Risk_Assessment._check_autosave_recovery(app)
        clear_mock.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
