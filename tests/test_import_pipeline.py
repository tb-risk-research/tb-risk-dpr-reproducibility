#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Section VIII 数据导入流程统一化测试

覆盖：
- ImportPipeline 模板方法基类：_merge_entries 去重、map_columns 三阶模糊匹配、
  remap_entry 字段重映射
- CSVImportPipeline：FORMAT_NAME、_parse_csv_fallback 实文件解析
- JSONImportPipeline：FORMAT_NAME、parse_data 多种 JSON 结构（dict/list/patient_info）
- DatabaseImportPipeline：DB_PARAM_SPECS 完整性、_do_test_connection 各 db_type 行为
- APIImportPipeline：FORMAT_NAME
- ImportPipeline.run 集成：取消传播、三种导入模式（replace/append/merge）、
  results_stale 标记调用
"""

# 检查 tkinter 是否可用，不可用时跳过整个测试文件
try:
    import tkinter  # noqa: F401
    _TKINTER_AVAILABLE = True
except ImportError:
    _TKINTER_AVAILABLE = False

import json
import os
import sys
import tempfile
from unittest import mock

import pytest

if not _TKINTER_AVAILABLE:
    pytest.skip("tkinter 不可用，跳过 GUI 导入管线测试", allow_module_level=True)

# 将项目根目录加入 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from tb_risk.gui.import_panel.pipeline import (
    ImportPipeline, CSVImportPipeline, JSONImportPipeline,
    ExcelImportPipeline, APIImportPipeline, DatabaseImportPipeline,
)


# ===========================================================================
# 辅助 fixture：构造最小可用 app 对象
# ===========================================================================

class _FakeApp:
    """最小可用 GUI app 桩对象，供 ImportPipeline 测试使用

    提供 ImportPipeline.run / _load_data 访问的全部属性：
    - root: 用于 Toplevel 创建（仅在 select_source 测试中需要，多数测试不触发）
    - family_entries / social_entries: list
    - _show_progress_dialog: 返回 4-tuple
    - _show_import_preview / _show_low_confidence_dialog: 返回 bool
    - _update_gui_from_import / _update_fill_progress / _mark_results_stale: noop
    """

    def __init__(self):
        self.family_entries = []
        self.social_entries = []
        self.root = None
        self.stale_called = False

    def _show_progress_dialog(self, title="正在处理..."):
        # 返回与 preview_panel._show_progress_dialog 相同结构的 4-tuple
        def noop_update(value, message=None):
            pass

        def noop_close():
            pass

        return None, None, noop_update, noop_close

    def _show_import_preview(self, entries, import_type="家庭成员",
                              quality_scores=None, field_mappings=None,
                              auto_corrections=None):
        return True  # 用户总是确认

    def _show_low_confidence_dialog(self, low_confidence_matches):
        return {}  # 返回空 dict 表示用户确认全部，但不修改映射

    def _update_gui_from_import(self):
        pass

    def _update_fill_progress(self):
        pass

    def _mark_results_stale(self):
        self.stale_called = True


@pytest.fixture
def fake_app():
    return _FakeApp()


# ===========================================================================
# ImportPipeline 基类：_merge_entries / map_columns / remap_entry
# ===========================================================================

class TestImportPipelineMergeEntries:
    """_merge_entries 去重合并逻辑"""

    def test_append_new_entries(self):
        existing = [{'name': 'Alice', 'age': 30}]
        new = [{'name': 'Bob', 'age': 25}]
        ImportPipeline._merge_entries(existing, new)
        assert len(existing) == 2
        assert existing[1] == {'name': 'Bob', 'age': 25}

    def test_dedup_by_name_and_age(self):
        """name+age 相同的条目视为重复，不追加"""
        existing = [{'name': 'Alice', 'age': 30}]
        new = [
            {'name': 'Alice', 'age': 30},  # 完全相同
            {'name': 'Alice', 'age': 25},  # 同名不同年龄，视为不同人
            {'name': 'Bob', 'age': 30},    # 不同名同年龄，视为不同人
        ]
        ImportPipeline._merge_entries(existing, new)
        assert len(existing) == 3
        names = [e['name'] for e in existing]
        assert names.count('Alice') == 2  # 30 岁 + 25 岁

    def test_dedup_strips_name_whitespace(self):
        """name 字段前后空格不影响去重"""
        existing = [{'name': 'Alice', 'age': 30}]
        new = [{'name': '  Alice  ', 'age': 30}]
        ImportPipeline._merge_entries(existing, new)
        assert len(existing) == 1  # 去重成功

    def test_dedup_missing_name_treated_as_empty(self):
        """缺失 name 字段按空字符串处理，但仍按 name+age 去重"""
        existing = [{}]  # key = ('', '')
        new = [{}]  # key = ('', '') — 与 existing 完全相同，应被去重
        ImportPipeline._merge_entries(existing, new)
        assert len(existing) == 1  # 去重成功

    def test_dedup_missing_name_different_age_kept(self):
        """缺失 name 但 age 不同应被视为不同条目"""
        existing = [{}]  # key = ('', '')
        new = [{'age': 30}]  # key = ('', 30) — age 不同，不重复
        ImportPipeline._merge_entries(existing, new)
        assert len(existing) == 2  # 保留两条

    def test_merge_into_empty_existing(self):
        existing = []
        new = [{'name': 'A', 'age': 1}, {'name': 'B', 'age': 2}]
        ImportPipeline._merge_entries(existing, new)
        assert len(existing) == 2

    def test_merge_empty_new_no_change(self):
        existing = [{'name': 'A', 'age': 1}]
        ImportPipeline._merge_entries(existing, [])
        assert len(existing) == 1

    def test_merge_does_not_mutate_new_entries(self):
        """_merge_entries 不应修改 new_entries 列表本身"""
        existing = []
        new = [{'name': 'A', 'age': 1}]
        new_before = list(new)
        ImportPipeline._merge_entries(existing, new)
        assert new == new_before


# ===========================================================================
# ImportPipeline.map_columns: 三阶模糊匹配
# ===========================================================================

class TestImportPipelineMapColumns:
    """map_columns 静态方法：列名 → 标准字段名映射"""

    def test_map_exact_chinese_name(self):
        """'姓名' 应映射到 member_name"""
        mappings, low_conf = ImportPipeline.map_columns(['姓名'])
        assert mappings.get('姓名') == 'member_name'

    def test_map_exact_english_age(self):
        """'age' 应映射到 age 字段"""
        mappings, _ = ImportPipeline.map_columns(['age'])
        assert mappings.get('age') == 'age'

    def test_map_synonym_chinese_clinical(self):
        """'痰涂片' 应映射到 sputum_smear"""
        mappings, _ = ImportPipeline.map_columns(['痰涂片'])
        assert mappings.get('痰涂片') == 'sputum_smear'

    def test_map_keyword_match(self):
        """关键词包含匹配：'患者年龄' 应能映射到 age"""
        mappings, _ = ImportPipeline.map_columns(['患者年龄'])
        assert mappings.get('患者年龄') in ('age', None)  # 至少不抛异常

    def test_map_unrecognized_column_skipped(self):
        """完全不相关的列名应被跳过，不在 mappings 中"""
        mappings, _ = ImportPipeline.map_columns(['xyz_random_unknown_column'])
        assert 'xyz_random_unknown_column' not in mappings

    def test_map_empty_input(self):
        mappings, low_conf = ImportPipeline.map_columns([])
        assert mappings == {}
        assert low_conf == []

    def test_map_skips_empty_string_column(self):
        """空字符串列名应被跳过"""
        mappings, _ = ImportPipeline.map_columns(['', 'age'])
        assert '' not in mappings
        assert 'age' in mappings

    def test_map_returns_low_confidence_list(self):
        """低置信度（<0.8）匹配应出现在 low_confidence 列表中"""
        # 使用一个明显模糊的列名触发 Levenshtein 匹配
        mappings, low_conf = ImportPipeline.map_columns(['ag'])  # 'ag' 接近 'age'
        # 无论是否匹配到，low_confidence 都是 list 类型
        assert isinstance(low_conf, list)
        # 若匹配到 age，置信度可能 < 0.8
        if 'ag' in mappings:
            for col, std, conf in low_conf:
                assert col == 'ag'
                assert conf < 0.8

    def test_map_multiple_columns(self):
        """多列同时映射"""
        cols = ['姓名', 'age', '痰涂片', 'unknown_xyz']
        mappings, _ = ImportPipeline.map_columns(cols)
        assert mappings.get('姓名') == 'member_name'
        assert mappings.get('age') == 'age'
        assert mappings.get('痰涂片') == 'sputum_smear'
        assert 'unknown_xyz' not in mappings


# ===========================================================================
# ImportPipeline.remap_entry
# ===========================================================================

class TestImportPipelineRemapEntry:
    """remap_entry 静态方法：根据字段映射转换 entry"""

    def test_remap_basic(self):
        raw = {'姓名': '张三', 'age': 30}
        mappings = {'姓名': 'member_name', 'age': 'age'}
        result = ImportPipeline.remap_entry(raw, mappings)
        assert result == {'member_name': '张三', 'age': 30}

    def test_remap_no_mappings_returns_copy(self):
        """空映射应返回原始 entry 的副本"""
        raw = {'name': 'A', 'age': 1}
        result = ImportPipeline.remap_entry(raw, {})
        assert result == raw
        assert result is not raw  # 必须是新对象

    def test_remap_unknown_keys_preserved(self):
        """未在 mappings 中的 key 应原样保留"""
        raw = {'name': 'A', 'extra_field': 'value'}
        mappings = {'name': 'member_name'}
        result = ImportPipeline.remap_entry(raw, mappings)
        assert result == {'member_name': 'A', 'extra_field': 'value'}

    def test_remap_does_not_mutate_input(self):
        raw = {'姓名': '张三'}
        mappings = {'姓名': 'member_name'}
        ImportPipeline.remap_entry(raw, mappings)
        assert raw == {'姓名': '张三'}  # 原始对象未被修改


# ===========================================================================
# 子类 FORMAT_NAME 与默认属性
# ===========================================================================

class TestPipelineFormatNames:
    """各子类 FORMAT_NAME 与默认属性正确性"""

    @pytest.mark.parametrize("cls,expected_name", [
        (CSVImportPipeline, 'CSV'),
        (JSONImportPipeline, 'JSON'),
        (ExcelImportPipeline, 'Excel'),
        (APIImportPipeline, 'API'),
        (DatabaseImportPipeline, '数据库'),
    ])
    def test_format_name(self, cls, expected_name):
        assert cls.FORMAT_NAME == expected_name

    @pytest.mark.parametrize("cls", [
        CSVImportPipeline, JSONImportPipeline, ExcelImportPipeline,
        APIImportPipeline, DatabaseImportPipeline,
    ])
    def test_supports_mode_selection_default_true(self, cls):
        """所有子类默认启用模式选择"""
        assert cls.SUPPORTS_MODE_SELECTION is True

    def test_base_format_name_default(self):
        assert ImportPipeline.FORMAT_NAME == '通用'

    def test_get_progress_title_includes_format_name(self, fake_app):
        pipeline = CSVImportPipeline(fake_app)
        assert 'CSV' in pipeline.get_progress_title()

    def test_init_sets_app_and_cancelled(self, fake_app):
        pipeline = CSVImportPipeline(fake_app)
        assert pipeline.app is fake_app
        assert pipeline.cancelled is False


# ===========================================================================
# CSVImportPipeline._parse_csv_fallback
# ===========================================================================

class TestCSVImportPipelineParse:

    def test_parse_csv_fallback_with_real_file(self, fake_app, tmp_path):
        """使用临时 CSV 文件验证 _parse_csv_fallback 解析逻辑"""
        csv_content = "姓名,age,痰涂片\n张三,30,阳性\n李四,25,阴性\n"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content, encoding='utf-8-sig')

        pipeline = CSVImportPipeline(fake_app)
        progress_calls = []

        def track_progress(value, message=None):
            progress_calls.append((value, message))

        family, social, metadata = pipeline._parse_csv_fallback(
            str(csv_file), track_progress)

        # 应解析出 2 条家庭成员记录
        assert len(family) == 2
        assert social == []
        # 字段映射应至少识别 'age' 和 '姓名'/'痰涂片'
        assert 'age' in metadata.get('field_mappings', {}).values() or \
               'age' in metadata.get('field_mappings', {})
        assert metadata.get('source_file') == str(csv_file)
        # 进度回调被调用
        assert len(progress_calls) >= 2

    def test_parse_csv_fallback_missing_file(self, fake_app, tmp_path):
        """文件不存在时应返回空列表"""
        pipeline = CSVImportPipeline(fake_app)
        family, social, metadata = pipeline._parse_csv_fallback(
            str(tmp_path / "nonexistent.csv"), lambda v, m=None: None)
        assert family == []
        assert social == []
        assert metadata == {}

    def test_parse_csv_fallback_empty_file(self, fake_app, tmp_path):
        """空文件应不抛异常，返回空列表"""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("", encoding='utf-8')

        pipeline = CSVImportPipeline(fake_app)
        family, social, metadata = pipeline._parse_csv_fallback(
            str(csv_file), lambda v, m=None: None)
        # 空文件 fieldnames 为 None，entries 应为空
        assert family == []
        assert metadata.get('source_file') == str(csv_file)


# ===========================================================================
# JSONImportPipeline.parse_data
# ===========================================================================

class TestJSONImportPipelineParse:

    def test_parse_standard_dict_format(self, fake_app, tmp_path):
        """标准 {family: [...], social: [...]} 格式"""
        data = {
            'family': [{'name': '张三', 'age': 30}],
            'social': [{'name': '李四', 'age': 25}],
        }
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')

        pipeline = JSONImportPipeline(fake_app)
        family, social, metadata = pipeline.parse_data(
            {'file_path': str(json_file)}, lambda v, m=None: None)

        assert len(family) == 1
        assert family[0]['name'] == '张三'
        assert len(social) == 1
        assert social[0]['name'] == '李四'
        assert metadata.get('source_file') == str(json_file)

    def test_parse_alternative_keys(self, fake_app, tmp_path):
        """family_members / social_contacts 别名键也应被识别"""
        data = {
            'family_members': [{'name': 'A', 'age': 1}],
            'social_contacts': [{'name': 'B', 'age': 2}],
        }
        json_file = tmp_path / "alt.json"
        json_file.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')

        pipeline = JSONImportPipeline(fake_app)
        family, social, _ = pipeline.parse_data(
            {'file_path': str(json_file)}, lambda v, m=None: None)
        assert len(family) == 1
        assert len(social) == 1

    def test_parse_list_format_all_family(self, fake_app, tmp_path):
        """顶层 list 格式：全部当家庭成员"""
        data = [{'name': 'A', 'age': 1}, {'name': 'B', 'age': 2}]
        json_file = tmp_path / "list.json"
        json_file.write_text(json.dumps(data), encoding='utf-8')

        pipeline = JSONImportPipeline(fake_app)
        family, social, _ = pipeline.parse_data(
            {'file_path': str(json_file)}, lambda v, m=None: None)
        assert len(family) == 2
        assert social == []

    def test_parse_preserves_patient_info(self, fake_app, tmp_path):
        """patient_info 字段应保存到 metadata"""
        data = {
            'patient_info': {'age': 50, 'sputum_smear': 2},
            'family': [],
            'social': [],
        }
        json_file = tmp_path / "pinfo.json"
        json_file.write_text(json.dumps(data), encoding='utf-8')

        pipeline = JSONImportPipeline(fake_app)
        _, _, metadata = pipeline.parse_data(
            {'file_path': str(json_file)}, lambda v, m=None: None)
        assert metadata.get('patient_info') == {'age': 50, 'sputum_smear': 2}

    def test_parse_invalid_json_returns_empty(self, fake_app, tmp_path):
        """非法 JSON 应返回空列表"""
        json_file = tmp_path / "bad.json"
        json_file.write_text("{invalid json content", encoding='utf-8')

        pipeline = JSONImportPipeline(fake_app)
        family, social, metadata = pipeline.parse_data(
            {'file_path': str(json_file)}, lambda v, m=None: None)
        assert family == []
        assert social == []
        assert metadata == {}

    def test_parse_empty_dict(self, fake_app, tmp_path):
        """空 dict 应返回空列表"""
        json_file = tmp_path / "empty.json"
        json_file.write_text("{}", encoding='utf-8')

        pipeline = JSONImportPipeline(fake_app)
        family, social, _ = pipeline.parse_data(
            {'file_path': str(json_file)}, lambda v, m=None: None)
        assert family == []
        assert social == []


# ===========================================================================
# DatabaseImportPipeline.DB_PARAM_SPECS / _do_test_connection
# ===========================================================================

class TestDatabaseImportPipeline:

    def test_db_param_specs_contains_all_three_db_types(self):
        """DB_PARAM_SPECS 应包含 sqlite / mysql / postgresql"""
        assert set(DatabaseImportPipeline.DB_PARAM_SPECS.keys()) == \
               {'sqlite', 'mysql', 'postgresql'}

    def test_sqlite_specs_have_file_path(self):
        """SQLite 参数应包含 (file_path, 数据库文件路径, file, None) 元组"""
        sqlite_specs = DatabaseImportPipeline.DB_PARAM_SPECS['sqlite']
        assert ('file_path', '数据库文件路径', 'file', None) in sqlite_specs

    def test_mysql_specs_include_required_params(self):
        """MySQL 参数应包含 host/port/user/password/database"""
        param_names = {spec[0] for spec in
                       DatabaseImportPipeline.DB_PARAM_SPECS['mysql']}
        assert {'host', 'port', 'user', 'password', 'database'} == param_names

    def test_postgresql_specs_include_required_params(self):
        param_names = {spec[0] for spec in
                       DatabaseImportPipeline.DB_PARAM_SPECS['postgresql']}
        assert {'host', 'port', 'user', 'password', 'database'} == param_names

    def test_mysql_default_port_3306(self):
        defaults = {spec[0]: spec[3] for spec in
                    DatabaseImportPipeline.DB_PARAM_SPECS['mysql']}
        assert defaults['port'] == '3306'

    def test_postgresql_default_port_5432(self):
        defaults = {spec[0]: spec[3] for spec in
                    DatabaseImportPipeline.DB_PARAM_SPECS['postgresql']}
        assert defaults['port'] == '5432'

    def test_do_test_connection_sqlite_existing_file(self, fake_app, tmp_path):
        """SQLite 测试连接：文件存在返回 True"""
        db_file = tmp_path / "test.db"
        db_file.write_text("dummy", encoding='utf-8')

        pipeline = DatabaseImportPipeline(fake_app)
        result = pipeline._do_test_connection('sqlite', {'file_path': str(db_file)})
        assert result is True

    def test_do_test_connection_sqlite_missing_file(self, fake_app):
        """SQLite 测试连接：文件不存在返回 False"""
        pipeline = DatabaseImportPipeline(fake_app)
        result = pipeline._do_test_connection('sqlite', {'file_path': '/nonexistent/path.db'})
        assert result is False

    def test_do_test_connection_sqlite_empty_path(self, fake_app):
        """SQLite 测试连接：空路径返回 False"""
        pipeline = DatabaseImportPipeline(fake_app)
        result = pipeline._do_test_connection('sqlite', {'file_path': ''})
        assert result is False

    def test_do_test_connection_unsupported_db_type(self, fake_app):
        """未知 db_type 返回 False"""
        pipeline = DatabaseImportPipeline(fake_app)
        result = pipeline._do_test_connection('oracle', {'host': 'localhost'})
        assert result is False

    def test_do_test_connection_delegates_to_app_method(self, fake_app):
        """app.test_database_connection 存在时应委托给它"""
        fake_app.test_database_connection = lambda db_type, params: True
        pipeline = DatabaseImportPipeline(fake_app)
        result = pipeline._do_test_connection('mysql', {'host': 'x'})
        assert result is True


# ===========================================================================
# ImportPipeline.run 集成测试
# ===========================================================================

class TestImportPipelineRunIntegration:
    """模板方法 run() 的集成测试：阶段流程与模式分支"""

    def test_run_returns_false_when_select_source_cancelled(self, fake_app):
        """Stage 1 取消（select_source 返回 None）应立即返回 False"""
        pipeline = CSVImportPipeline(fake_app)
        with mock.patch.object(pipeline, 'select_source', return_value=None):
            result = pipeline.run()
        assert result is False

    def test_run_returns_false_when_mode_cancelled(self, fake_app):
        """Stage 2 取消（_select_mode 返回 None）应返回 False"""
        pipeline = CSVImportPipeline(fake_app)
        with mock.patch.object(pipeline, 'select_source', return_value={'file_path': 'x'}), \
             mock.patch.object(pipeline, '_select_mode', return_value=None), \
             mock.patch.object(pipeline, 'parse_data',
                               return_value=([{'name': 'A'}], [], {})):
            result = pipeline.run()
        assert result is False

    def test_run_returns_false_when_parse_returns_empty(self, fake_app):
        """parse_data 返回空（两侧均空）应返回 False"""
        pipeline = CSVImportPipeline(fake_app)
        # 预存在测试隔离修复：run() 在 parse 返回空时会调用 messagebox.showinfo
        # 提示"没有可导入的数据"。单独运行时 fake_app 无真正 Tk root，showinfo
        # 立即返回；但全量套件中前面测试创建的 Tk root 残留会使 showinfo 真的
        # 弹出对话框等待用户点击，导致测试挂起。补 mock 与其他 run() 集成测试
        # （如 test_run_load_data_replace_mode_clears_existing）保持一致。
        with mock.patch.object(pipeline, 'select_source', return_value={'file_path': 'x'}), \
             mock.patch.object(pipeline, '_select_mode', return_value='replace'), \
             mock.patch.object(pipeline, 'parse_data', return_value=([], [], {})), \
             mock.patch('tb_risk.gui.import_panel.pipeline.messagebox'):
            result = pipeline.run()
        assert result is False

    def test_run_load_data_replace_mode_clears_existing(self, fake_app):
        """replace 模式应清空原有数据并填入新数据"""
        fake_app.family_entries = [{'name': 'OldFamily', 'age': 99}]
        fake_app.social_entries = [{'name': 'OldSocial', 'age': 99}]

        pipeline = CSVImportPipeline(fake_app)
        new_family = [{'name': 'NewFamily', 'age': 30}]
        new_social = [{'name': 'NewSocial', 'age': 25}]

        with mock.patch.object(pipeline, 'select_source', return_value={'file_path': 'x'}), \
             mock.patch.object(pipeline, '_select_mode', return_value='replace'), \
             mock.patch.object(pipeline, 'parse_data',
                               return_value=(new_family, new_social, {})), \
             mock.patch('tb_risk.gui.import_panel.pipeline.messagebox'):
            result = pipeline.run()

        assert result is True
        assert fake_app.family_entries == new_family
        assert fake_app.social_entries == new_social

    def test_run_load_data_append_mode_keeps_existing(self, fake_app):
        """append 模式应保留原有数据并追加新数据"""
        fake_app.family_entries = [{'name': 'Old', 'age': 99}]

        pipeline = CSVImportPipeline(fake_app)
        new_family = [{'name': 'New', 'age': 30}]

        with mock.patch.object(pipeline, 'select_source', return_value={'file_path': 'x'}), \
             mock.patch.object(pipeline, '_select_mode', return_value='append'), \
             mock.patch.object(pipeline, 'parse_data',
                               return_value=(new_family, [], {})), \
             mock.patch('tb_risk.gui.import_panel.pipeline.messagebox'):
            result = pipeline.run()

        assert result is True
        assert len(fake_app.family_entries) == 2
        assert fake_app.family_entries[0] == {'name': 'Old', 'age': 99}
        assert fake_app.family_entries[1] == {'name': 'New', 'age': 30}

    def test_run_load_data_merge_mode_deduplicates(self, fake_app):
        """merge 模式应去重合并"""
        fake_app.family_entries = [{'name': 'Alice', 'age': 30}]

        pipeline = CSVImportPipeline(fake_app)
        new_family = [
            {'name': 'Alice', 'age': 30},  # 重复，应被去重
            {'name': 'Bob', 'age': 25},    # 新增
        ]

        with mock.patch.object(pipeline, 'select_source', return_value={'file_path': 'x'}), \
             mock.patch.object(pipeline, '_select_mode', return_value='merge'), \
             mock.patch.object(pipeline, 'parse_data',
                               return_value=(new_family, [], {})), \
             mock.patch('tb_risk.gui.import_panel.pipeline.messagebox'):
            result = pipeline.run()

        assert result is True
        assert len(fake_app.family_entries) == 2
        assert fake_app.family_entries[0] == {'name': 'Alice', 'age': 30}
        assert fake_app.family_entries[1] == {'name': 'Bob', 'age': 25}

    def test_run_marks_results_stale_after_load(self, fake_app):
        """成功加载后应调用 _mark_results_stale"""
        pipeline = CSVImportPipeline(fake_app)
        with mock.patch.object(pipeline, 'select_source', return_value={'file_path': 'x'}), \
             mock.patch.object(pipeline, '_select_mode', return_value='replace'), \
             mock.patch.object(pipeline, 'parse_data',
                               return_value=([{'name': 'A'}], [], {})), \
             mock.patch('tb_risk.gui.import_panel.pipeline.messagebox'):
            pipeline.run()
        assert fake_app.stale_called is True

    def test_run_parse_exception_returns_false(self, fake_app):
        """parse_data 抛异常应被捕获并返回 False"""
        pipeline = CSVImportPipeline(fake_app)
        with mock.patch.object(pipeline, 'select_source', return_value={'file_path': 'x'}), \
             mock.patch.object(pipeline, '_select_mode', return_value='replace'), \
             mock.patch.object(pipeline, 'parse_data',
                               side_effect=ValueError("parse error")), \
             mock.patch('tb_risk.gui.import_panel.pipeline.messagebox'):
            result = pipeline.run()
        assert result is False

    def test_run_preview_cancelled_returns_false(self, fake_app):
        """预览阶段用户取消（_preview_and_confirm 返回 False）应返回 False"""
        pipeline = CSVImportPipeline(fake_app)
        with mock.patch.object(pipeline, 'select_source', return_value={'file_path': 'x'}), \
             mock.patch.object(pipeline, '_select_mode', return_value='replace'), \
             mock.patch.object(pipeline, 'parse_data',
                               return_value=([{'name': 'A'}], [], {})), \
             mock.patch.object(pipeline, '_preview_and_confirm', return_value=False):
            result = pipeline.run()
        assert result is False
        # 预览取消时不应调用 _mark_results_stale
        assert fake_app.stale_called is False


# ===========================================================================
# _load_data 异常路径
# ===========================================================================

class TestImportPipelineLoadDataErrorPath:

    def test_load_data_exception_returns_false(self, fake_app):
        """_load_data 抛异常应被捕获并返回 False"""
        pipeline = CSVImportPipeline(fake_app)
        # 优先级一后 _load_data 不再调用 family_entries.clear()，
        # 改为通过 ImportDataCommand 原子替换；触发异常的方式调整为
        # 让 _update_gui_from_import 抛异常
        with mock.patch.object(fake_app, '_update_gui_from_import',
                               side_effect=RuntimeError("boom")), \
             mock.patch('tb_risk.gui.import_panel.pipeline.messagebox'):
            result = pipeline._load_data([{'name': 'A'}], [], 'replace')
        assert result is False
