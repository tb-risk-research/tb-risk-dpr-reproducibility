#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Section VII 训练面板增强测试

覆盖：
- TrainingConfig 数据类：默认值、范围验证、序列化往返、未知字段忽略
- TrainingConfigDialog 对话框：字段生成、取消/确认逻辑（需 tkinter）
- ModelRepository 模型仓库：版本号生成、保存/加载/删除、列表排序、
  类型筛选、原子写入、索引损坏回退、回滚
- TrainingLogger 训练日志：追加写入、倒序读取、limit/类型筛选、
  损坏单行跳过、清空、统计摘要
- _RepoUIMixin 集成：MLTrainingMixin / TrainingMixin 方法可用性
"""

import json
import os
import sys
import tempfile
import time
from unittest import mock

import pytest

# 将项目根目录加入 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from tb_risk.gui.training_panel.config import (
    TrainingConfig, TrainingConfigDialog, FIELD_METADATA, FIELD_GROUPS,
)
from tb_risk.gui.training_panel.model_repo import ModelRecord, ModelRepository
from tb_risk.gui.training_panel.training_log import (
    TrainingLogEntry, TrainingLogger, make_run_id,
)


# ===========================================================================
# TrainingConfig 数据类测试
# ===========================================================================

class TestTrainingConfigDefaults:
    """TrainingConfig 默认值与基本行为"""

    def test_default_values(self):
        cfg = TrainingConfig()
        assert cfg.n_samples == 2000
        assert cfg.random_state == 42
        assert cfg.ml_enable_hyperopt is False
        assert cfg.ml_cv_folds == 5
        assert cfg.gnn_n_epochs == 50
        assert cfg.gnn_learning_rate == pytest.approx(0.001)
        assert cfg.gnn_enable_hyperopt is False
        assert cfg.gnn_use_focal_loss is False
        assert cfg.gnn_two_phase is True
        assert cfg.early_stopping_enabled is True
        assert cfg.early_stopping_patience == 10

    def test_all_fields_have_metadata(self):
        """FIELD_METADATA 覆盖所有公开字段"""
        cfg = TrainingConfig()
        public_fields = {f for f in cfg.__dataclass_fields__ if not f.startswith('_')}
        meta_fields = set(FIELD_METADATA.keys())
        assert public_fields == meta_fields, (
            f"字段不一致: dataclass={public_fields}, metadata={meta_fields}")

    def test_all_fields_in_groups(self):
        """FIELD_GROUPS 包含所有公开字段（每个字段恰好出现一次）"""
        cfg = TrainingConfig()
        public_fields = {f for f in cfg.__dataclass_fields__ if not f.startswith('_')}
        grouped = []
        for _, fields in FIELD_GROUPS:
            grouped.extend(fields)
        assert set(grouped) == public_fields
        assert len(grouped) == len(public_fields)  # 无重复


class TestTrainingConfigValidation:
    """TrainingConfig.validate() 范围验证"""

    def test_default_config_is_valid(self):
        cfg = TrainingConfig()
        ok, msg = cfg.validate()
        assert ok is True
        assert msg == ''

    @pytest.mark.parametrize("field_name,value,lo,hi", [
        ('n_samples', 50, 100, 100000),
        ('n_samples', 100001, 100, 100000),
        ('random_state', -1, 0, 2**31 - 1),
        ('ml_cv_folds', 1, 2, 20),
        ('gnn_n_epochs', 0, 1, 1000),
        ('gnn_learning_rate', 1e-7, 1e-6, 1.0),
        ('early_stopping_patience', 0, 1, 200),
    ])
    def test_out_of_range_invalid(self, field_name, value, lo, hi):
        cfg = TrainingConfig()
        setattr(cfg, field_name, value)
        ok, msg = cfg.validate()
        assert ok is False
        assert field_name in msg
        assert str(lo) in msg or str(hi) in msg

    @pytest.mark.parametrize("field_name,value", [
        ('n_samples', 100),
        ('n_samples', 100000),
        ('random_state', 0),
        ('ml_cv_folds', 2),
        ('gnn_n_epochs', 1),
        ('gnn_learning_rate', 1e-6),
        ('gnn_learning_rate', 1.0),
        ('early_stopping_patience', 1),
    ])
    def test_boundary_values_valid(self, field_name, value):
        cfg = TrainingConfig()
        setattr(cfg, field_name, value)
        ok, msg = cfg.validate()
        assert ok is True, f"边界值 {field_name}={value} 应合法: {msg}"


class TestTrainingConfigSerialization:
    """TrainingConfig to_dict / from_dict 往返"""

    def test_to_dict_excludes_private(self):
        cfg = TrainingConfig(n_samples=5000, gnn_n_epochs=100)
        d = cfg.to_dict()
        assert '_RANGES' not in d
        assert 'n_samples' in d and d['n_samples'] == 5000
        assert 'gnn_n_epochs' in d and d['gnn_n_epochs'] == 100

    def test_from_dict_round_trip(self):
        original = TrainingConfig(
            n_samples=3000, random_state=7, ml_cv_folds=10,
            gnn_n_epochs=80, gnn_learning_rate=0.005,
            gnn_use_focal_loss=True, early_stopping_patience=15,
        )
        d = original.to_dict()
        restored = TrainingConfig.from_dict(d)
        assert restored.n_samples == 3000
        assert restored.random_state == 7
        assert restored.ml_cv_folds == 10
        assert restored.gnn_n_epochs == 80
        assert restored.gnn_learning_rate == pytest.approx(0.005)
        assert restored.gnn_use_focal_loss is True
        assert restored.early_stopping_patience == 15

    def test_from_dict_ignores_unknown_fields(self):
        d = {'n_samples': 1000, 'unknown_field': 'spam', 'random_state': 99}
        cfg = TrainingConfig.from_dict(d)
        assert cfg.n_samples == 1000
        assert cfg.random_state == 99
        assert not hasattr(cfg, 'unknown_field')

    def test_from_dict_non_dict_returns_default(self):
        cfg = TrainingConfig.from_dict(None)
        assert cfg.n_samples == 2000
        cfg2 = TrainingConfig.from_dict("not a dict")
        assert cfg2.random_state == 42

    def test_from_dict_wrong_type_stored_as_is(self):
        """dataclass 不在构造时强制类型转换，错误类型原样存储

        调用方应通过 validate() 检测类型异常。
        """
        d = {'n_samples': 'not_a_number'}
        cfg = TrainingConfig.from_dict(d)
        # dataclass 不强制类型，字符串原样存储
        assert cfg.n_samples == 'not_a_number'
        # validate() 能检测出类型错误
        ok, msg = cfg.validate()
        assert ok is False
        assert 'n_samples' in msg

    def test_from_dict_empty_dict(self):
        """空字典返回默认配置"""
        cfg = TrainingConfig.from_dict({})
        assert cfg.n_samples == 2000
        assert cfg.gnn_n_epochs == 50


# ===========================================================================
# TrainingConfigDialog 测试（需要 tkinter）
# ===========================================================================

try:
    import tkinter as tk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


@pytest.mark.skipif(not TKINTER_AVAILABLE, reason="tkinter 不可用")
class TestTrainingConfigDialog:
    """TrainingConfigDialog 对话框逻辑

    注意：TrainingConfigDialog.__init__ 调用 wait_window() 会阻塞事件循环，
    测试中通过 mock wait_window 使其立即返回，再单独测试各回调方法。
    """

    @pytest.fixture
    def root(self):
        try:
            r = tk.Tk()
        except tk.TclError as e:
            pytest.skip(f"tkinter/Tcl 初始化失败: {e}")
        r.withdraw()  # 不显示主窗口
        yield r
        try:
            r.destroy()
        except tk.TclError:
            pass

    @pytest.fixture
    def dialog(self, root):
        """创建对话框实例（mock wait_window 避免阻塞）"""
        with mock.patch('tkinter.Toplevel.wait_window'):
            d = TrainingConfigDialog(root, TrainingConfig(), title='测试')
        yield d
        try:
            d._top.destroy()
        except tk.TclError:
            pass

    def test_cancel_returns_none(self, dialog):
        """取消按钮返回 None"""
        dialog._on_cancel()
        assert dialog.result is None

    def test_ok_returns_valid_config(self, dialog):
        """确定按钮返回验证通过的配置"""
        dialog._on_ok()
        assert dialog.result is not None
        assert isinstance(dialog.result, TrainingConfig)

    def test_reset_restores_defaults(self, root):
        """恢复默认按钮重置字段"""
        with mock.patch('tkinter.Toplevel.wait_window'):
            dialog = TrainingConfigDialog(
                root, TrainingConfig(n_samples=9999, gnn_n_epochs=200),
                title='测试')
        dialog._on_reset()
        assert dialog._vars['n_samples'].get() == 2000
        assert dialog._vars['gnn_n_epochs'].get() == 50
        dialog._top.destroy()

    def test_collect_config_invalid_value(self, dialog):
        """无效输入时 _collect_config 返回 None"""
        dialog._vars['n_samples'].set('not_a_number')
        with mock.patch('tkinter.messagebox.showerror'):
            result = dialog._collect_config()
        assert result is None

    def test_collect_config_out_of_range(self, dialog):
        """超出范围的值返回 None"""
        dialog._vars['n_samples'].set(10)  # 低于下限 100
        with mock.patch('tkinter.messagebox.showerror'):
            result = dialog._collect_config()
        assert result is None

    def test_dialog_with_none_config(self, root):
        """config=None 时使用默认配置"""
        with mock.patch('tkinter.Toplevel.wait_window'):
            dialog = TrainingConfigDialog(root, None, title='测试')
        assert isinstance(dialog.config, TrainingConfig)
        dialog._top.destroy()

    def test_dialog_collect_valid_config(self, dialog):
        """合法输入时 _collect_config 返回有效配置"""
        dialog._vars['n_samples'].set(5000)
        dialog._vars['gnn_n_epochs'].set(100)
        result = dialog._collect_config()
        assert result is not None
        assert result.n_samples == 5000
        assert result.gnn_n_epochs == 100

    def test_all_fields_have_ui_vars(self, dialog):
        """所有 FIELD_METADATA 字段都生成了对应的 UI 变量"""
        for fname in FIELD_METADATA:
            assert fname in dialog._vars, f"字段 {fname} 未生成 UI 变量"

    def test_all_groups_have_fields(self):
        """每个分组都包含至少一个字段"""
        for group_name, fields in FIELD_GROUPS:
            assert len(fields) > 0, f"分组 {group_name} 无字段"
            for f in fields:
                assert f in FIELD_METADATA, f"字段 {f} 缺少元数据"


# ===========================================================================
# ModelRepository 测试
# ===========================================================================

class TestModelRepository:
    """ModelRepository 模型仓库"""

    @pytest.fixture
    def repo(self, tmp_path):
        """使用临时目录的仓库实例"""
        return ModelRepository(base_dir=str(tmp_path))

    @pytest.fixture
    def model_file(self, tmp_path):
        """创建临时模型文件"""
        p = tmp_path / "source_model.joblib"
        p.write_bytes(b"fake model content")
        return str(p)

    def test_empty_repository(self, repo):
        assert repo.list_models() == []
        assert repo.count() == 0
        assert repo.get_latest() is None
        assert repo.get_model('v001') is None

    def test_save_model_creates_versioned_record(self, repo, model_file):
        record = repo.save_model(
            model_file, model_type='ml',
            params={'n_samples': 2000},
            metrics={'accuracy': 0.95},
            training_duration=12.5,
        )
        assert record.version == 'v001'
        assert record.model_type == 'ml'
        assert record.timestamp != ''
        assert os.path.exists(record.file_path)
        assert record.params == {'n_samples': 2000}
        assert record.metrics == {'accuracy': 0.95}
        assert record.training_duration == 12.5

    def test_save_model_increments_version(self, repo, model_file):
        r1 = repo.save_model(model_file, model_type='ml')
        r2 = repo.save_model(model_file, model_type='ml')
        r3 = repo.save_model(model_file, model_type='gnn')
        assert r1.version == 'v001'
        assert r2.version == 'v002'
        assert r3.version == 'v003'

    def test_save_model_copies_file(self, repo, model_file):
        record = repo.save_model(model_file, model_type='ml')
        # 源文件和目标文件都存在
        assert os.path.exists(model_file)
        assert os.path.exists(record.file_path)
        # 内容一致
        with open(model_file, 'rb') as f:
            src = f.read()
        with open(record.file_path, 'rb') as f:
            dst = f.read()
        assert src == dst

    def test_save_model_nonexistent_source_raises(self, repo):
        with pytest.raises(FileNotFoundError):
            repo.save_model('/nonexistent/path.joblib', model_type='ml')

    def test_list_models_sorted_desc(self, repo, model_file):
        """list_models 按时间戳降序（最新在前）"""
        r1 = repo.save_model(model_file, model_type='ml')
        time.sleep(1.1)  # 确保时间戳不同
        r2 = repo.save_model(model_file, model_type='ml')
        records = repo.list_models()
        assert len(records) == 2
        # r2 更新，应在前
        assert records[0].version == r2.version
        assert records[1].version == r1.version

    def test_list_models_filter_by_type(self, repo, model_file):
        repo.save_model(model_file, model_type='ml')
        repo.save_model(model_file, model_type='gnn')
        repo.save_model(model_file, model_type='ml')
        ml_records = repo.list_models(model_type='ml')
        gnn_records = repo.list_models(model_type='gnn')
        assert len(ml_records) == 2
        assert len(gnn_records) == 1
        assert all(r.model_type == 'ml' for r in ml_records)
        assert all(r.model_type == 'gnn' for r in gnn_records)

    def test_get_model_by_version(self, repo, model_file):
        r1 = repo.save_model(model_file, model_type='ml')
        r2 = repo.save_model(model_file, model_type='gnn')
        found = repo.get_model('v002')
        assert found is not None
        assert found.version == 'v002'
        assert found.model_type == 'gnn'
        assert repo.get_model('v999') is None

    def test_load_model_path(self, repo, model_file):
        record = repo.save_model(model_file, model_type='ml')
        path = repo.load_model_path('v001')
        assert path == record.file_path
        assert os.path.exists(path)

    def test_load_model_path_missing_file(self, repo, model_file):
        record = repo.save_model(model_file, model_type='ml')
        # 删除模型文件但保留索引
        os.remove(record.file_path)
        assert repo.load_model_path('v001') is None

    def test_load_model_path_nonexistent_version(self, repo):
        assert repo.load_model_path('v999') is None

    def test_delete_model(self, repo, model_file):
        r1 = repo.save_model(model_file, model_type='ml')
        assert repo.count() == 1
        assert repo.delete_model('v001') is True
        assert repo.count() == 0
        assert not os.path.exists(r1.file_path)
        # 再删一次返回 False
        assert repo.delete_model('v001') is False

    def test_delete_model_preserves_others(self, repo, model_file):
        repo.save_model(model_file, model_type='ml')
        repo.save_model(model_file, model_type='gnn')
        repo.save_model(model_file, model_type='ml')
        repo.delete_model('v002')
        assert repo.count() == 2
        remaining = {r.version for r in repo.list_models()}
        assert remaining == {'v001', 'v003'}

    def test_get_latest(self, repo, model_file):
        repo.save_model(model_file, model_type='ml')
        time.sleep(1.1)
        r2 = repo.save_model(model_file, model_type='ml')
        latest = repo.get_latest()
        assert latest is not None
        assert latest.version == r2.version

    def test_get_latest_by_type(self, repo, model_file):
        repo.save_model(model_file, model_type='ml')
        time.sleep(1.1)
        repo.save_model(model_file, model_type='gnn')
        time.sleep(1.1)
        r3 = repo.save_model(model_file, model_type='ml')
        latest_ml = repo.get_latest(model_type='ml')
        assert latest_ml.version == r3.version
        assert latest_ml.model_type == 'ml'

    def test_rollback_returns_record(self, repo, model_file):
        repo.save_model(model_file, model_type='ml')
        repo.save_model(model_file, model_type='ml')
        rolled = repo.rollback('v001')
        assert rolled is not None
        assert rolled.version == 'v001'
        # 回滚不删除较新版本
        assert repo.count() == 2

    def test_rollback_nonexistent(self, repo):
        assert repo.rollback('v999') is None

    def test_compare_models_all(self, repo, model_file):
        repo.save_model(model_file, model_type='ml',
                         metrics={'accuracy': 0.9})
        repo.save_model(model_file, model_type='ml',
                         metrics={'accuracy': 0.95})
        records = repo.compare_models()
        assert len(records) == 2
        # 按版本号升序
        assert records[0].version == 'v001'
        assert records[1].version == 'v002'

    def test_compare_models_subset(self, repo, model_file):
        repo.save_model(model_file, model_type='ml')
        repo.save_model(model_file, model_type='ml')
        repo.save_model(model_file, model_type='gnn')
        records = repo.compare_models(versions=['v001', 'v003'])
        assert len(records) == 2
        versions = {r.version for r in records}
        assert versions == {'v001', 'v003'}

    def test_corrupted_index_falls_back_to_empty(self, repo, model_file):
        """索引文件损坏时回退空列表"""
        repo.save_model(model_file, model_type='ml')
        # 写入损坏的 JSON
        with open(repo.index_path, 'w', encoding='utf-8') as f:
            f.write('{invalid json content')
        assert repo.list_models() == []
        assert repo.count() == 0

    def test_index_not_list_falls_back(self, repo, model_file):
        """索引文件是合法 JSON 但不是列表时回退空列表"""
        repo.save_model(model_file, model_type='ml')
        with open(repo.index_path, 'w', encoding='utf-8') as f:
            json.dump({'not': 'a list'}, f)
        assert repo.list_models() == []

    def test_atomic_write_preserves_index(self, repo, model_file):
        """多次保存后索引文件保持完整"""
        for i in range(5):
            repo.save_model(model_file, model_type='ml')
        assert repo.count() == 5
        # 索引文件是合法 JSON
        with open(repo.index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 5

    def test_next_version_handles_non_standard(self, repo, model_file):
        """版本号非标准格式时也能正确递增"""
        repo.save_model(model_file, model_type='ml')  # v001
        repo.save_model(model_file, model_type='ml')  # v002
        # 手动插入一个非标准版本
        records = repo._load_index()
        records.append(ModelRecord(
            version='custom', timestamp='', model_type='ml', file_path=''))
        repo._save_index(records)
        # 下一个版本应是 v003（max numeric + 1）
        r_new = repo.save_model(model_file, model_type='ml')
        assert r_new.version == 'v003'


class TestModelRecord:
    """ModelRecord 数据类"""

    def test_to_dict_round_trip(self):
        r = ModelRecord(
            version='v001', timestamp='2026-07-15 12:00:00',
            model_type='ml', file_path='/tmp/model.joblib',
            params={'a': 1}, metrics={'acc': 0.9},
            training_duration=10.0, notes='test',
        )
        d = r.to_dict()
        r2 = ModelRecord.from_dict(d)
        assert r2.version == 'v001'
        assert r2.params == {'a': 1}
        assert r2.metrics == {'acc': 0.9}
        assert r2.notes == 'test'

    def test_from_dict_ignores_unknown(self):
        d = {'version': 'v001', 'timestamp': '', 'model_type': 'ml',
             'file_path': '/x', 'unknown': 'field'}
        r = ModelRecord.from_dict(d)
        assert r.version == 'v001'
        assert not hasattr(r, 'unknown')

    def test_from_dict_invalid_falls_back(self):
        d = {'version': 'v001'}  # 缺少必填字段
        r = ModelRecord.from_dict(d)
        # 回退为占位记录
        assert r.version == '?'
        assert r.model_type == '?'


# ===========================================================================
# TrainingLogger 测试
# ===========================================================================

class TestTrainingLogger:
    """TrainingLogger 训练日志"""

    @pytest.fixture
    def logger(self, tmp_path):
        return TrainingLogger(base_dir=str(tmp_path))

    def test_log_creates_entry(self, logger):
        entry = logger.log(
            model_type='ml', params={'n_samples': 1000},
            metrics={'accuracy': 0.9}, training_duration=5.5,
        )
        assert entry.model_type == 'ml'
        assert entry.params == {'n_samples': 1000}
        assert entry.metrics == {'accuracy': 0.9}
        assert entry.training_duration == 5.5
        assert entry.status == 'success'
        assert entry.timestamp != ''

    def test_log_appends_to_file(self, logger):
        logger.log(model_type='ml')
        logger.log(model_type='gnn')
        entries = logger.read_history()
        assert len(entries) == 2

    def test_log_with_decision_reference_archives_snapshot(self, logger):
        """缺陷3（2026-09-17）：决策参考表随 run 归档，审计免加载模型。"""
        ref = {
            'decision_form': 'ranking_cutoff',
            'operating_point': {'sensitivity': 0.5, 'specificity': 0.9},
            'ppv': {'cohort_local': {'prevalence': 0.0053, 'ppv': 0.0457}},
        }
        entry = logger.log(
            model_type='ml', params={'feature_set': 'v4'},
            metrics={'AUROC': 0.95}, decision_reference=ref)
        assert entry.decision_reference == ref
        # 落盘 round-trip：jsonl 总表与 results/{run_id}.json 均携带
        entries = logger.read_history()
        assert entries[0].decision_reference['ppv']['cohort_local'][
            'prevalence'] == 0.0053
        results_path = os.path.join(
            logger.results_dir, f'{entry.run_id}.json')
        assert os.path.exists(results_path)
        with open(results_path, encoding='utf-8') as f:
            archived = json.load(f)
        assert archived['decision_reference'] == ref

    def test_log_without_decision_reference_defaults_empty(self, logger):
        entry = logger.log(model_type='ml')
        assert entry.decision_reference == {}
        # 旧记录向后兼容：from_dict 过滤未知字段，无该字段默认 {}
        legacy = entry.to_dict()
        legacy.pop('decision_reference')
        restored = type(entry).from_dict(legacy)
        assert restored.decision_reference == {}

    def test_read_history_reverse_order(self, logger):
        """read_history 返回倒序（最新在前）"""
        logger.log(model_type='ml')
        time.sleep(1.1)
        logger.log(model_type='gnn')
        entries = logger.read_history()
        assert entries[0].model_type == 'gnn'
        assert entries[1].model_type == 'ml'

    def test_read_history_limit(self, logger):
        for _ in range(5):
            logger.log(model_type='ml')
        entries = logger.read_history(limit=3)
        assert len(entries) == 3

    def test_read_history_filter_by_type(self, logger):
        logger.log(model_type='ml')
        logger.log(model_type='gnn')
        logger.log(model_type='ml')
        ml_entries = logger.read_history(model_type='ml')
        assert len(ml_entries) == 2
        assert all(e.model_type == 'ml' for e in ml_entries)

    def test_log_with_status_and_error(self, logger):
        logger.log(model_type='ml', status='failed',
                    error_message='OOM error')
        entries = logger.read_history()
        assert len(entries) == 1
        assert entries[0].status == 'failed'
        assert entries[0].error_message == 'OOM error'

    def test_clear_history(self, logger):
        logger.log(model_type='ml')
        logger.log(model_type='gnn')
        count = logger.clear_history()
        assert count == 2
        assert logger.read_history() == []

    def test_clear_empty_history(self, logger):
        count = logger.clear_history()
        assert count == 0

    def test_get_stats_empty(self, logger):
        stats = logger.get_stats()
        assert stats['total_count'] == 0
        assert stats['success_count'] == 0
        assert stats['failure_count'] == 0
        assert stats['avg_duration'] == 0.0
        assert stats['last_training_time'] == ''

    def test_get_stats_with_entries(self, logger):
        logger.log(model_type='ml', training_duration=10.0, status='success')
        logger.log(model_type='gnn', training_duration=20.0, status='failed',
                    error_message='err')
        logger.log(model_type='ml', training_duration=30.0, status='success')
        stats = logger.get_stats()
        assert stats['total_count'] == 3
        assert stats['success_count'] == 2
        assert stats['failure_count'] == 1
        assert stats['avg_duration'] == pytest.approx(20.0)
        assert stats['last_training_time'] != ''

    def test_get_stats_zero_duration_excluded(self, logger):
        """avg_duration 排除耗时为 0 的记录"""
        logger.log(model_type='ml', training_duration=0.0)
        logger.log(model_type='ml', training_duration=10.0)
        stats = logger.get_stats()
        assert stats['avg_duration'] == pytest.approx(10.0)

    def test_corrupted_line_skipped(self, logger, tmp_path):
        """损坏的单行不影响其他记录读取"""
        logger.log(model_type='ml')
        logger.log(model_type='gnn')
        # 在文件中间插入一行损坏数据
        with open(logger.log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines.insert(1, 'this is not json\n')
        with open(logger.log_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        entries = logger.read_history()
        # 2 条合法记录被读取，1 条损坏被跳过
        assert len(entries) == 2

    def test_empty_lines_skipped(self, logger, tmp_path):
        """空行被跳过"""
        logger.log(model_type='ml')
        with open(logger.log_path, 'a', encoding='utf-8') as f:
            f.write('\n\n\n')
        entries = logger.read_history()
        assert len(entries) == 1

    def test_read_nonexistent_file(self, tmp_path):
        """日志文件不存在时返回空列表"""
        logger = TrainingLogger(base_dir=str(tmp_path / 'nonexistent'))
        # __init__ 会创建目录，但日志文件不存在
        assert logger.read_history() == []
        assert logger.get_stats()['total_count'] == 0


class TestTrainingLogEntry:
    """TrainingLogEntry 数据类"""

    def test_to_jsonl_single_line(self):
        entry = TrainingLogEntry(
            timestamp='2026-07-15 12:00:00', model_type='ml',
            params={'a': 1}, metrics={'b': 0.5},
        )
        line = entry.to_jsonl()
        assert '\n' not in line
        data = json.loads(line)
        assert data['model_type'] == 'ml'

    def test_from_dict_round_trip(self):
        entry = TrainingLogEntry(
            timestamp='2026-07-15 12:00:00', model_type='gnn',
            params={'lr': 0.001}, status='success',
        )
        d = entry.to_dict()
        restored = TrainingLogEntry.from_dict(d)
        assert restored.model_type == 'gnn'
        assert restored.params == {'lr': 0.001}

    def test_from_dict_invalid_falls_back(self):
        d = {'timestamp': 'x'}  # 缺少 model_type
        entry = TrainingLogEntry.from_dict(d)
        assert entry.timestamp == ''
        assert entry.model_type == '?'


# ===========================================================================
# TrainingLogger 基准对比测试（训练档案系统）
# ===========================================================================

class TestTrainingLoggerComparison:
    """TrainingLogger 基准对比与训练档案"""

    @pytest.fixture
    def logger(self, tmp_path):
        return TrainingLogger(base_dir=str(tmp_path))

    def test_make_run_id_ml_format(self):
        run_id = make_run_id({'n_samples': 2000, 'ml_cv_folds': 5}, 'ml')
        assert run_id.endswith('_ns2000_cv5')
        # 时间戳前缀 20260101_000000 格式
        assert len(run_id.split('_')[0]) == 8

    def test_make_run_id_gnn_format(self):
        run_id = make_run_id(
            {'gnn_learning_rate': 0.001, 'gnn_n_epochs': 50}, 'gnn')
        assert 'gnn' in run_id
        assert 'lr0.001' in run_id
        assert 'ep50' in run_id

    def test_first_success_is_new_best(self, logger):
        entry = logger.log(model_type='ml', metrics={'AUROC': 0.80})
        assert entry.comparison['is_new_best'] is True
        assert entry.comparison['vs_last'] is None
        assert entry.comparison['vs_best'] is None
        best = logger.get_best('ml')
        assert best is not None
        assert best['run_id'] == entry.run_id

    def test_improvement_updates_best_and_deltas(self, logger):
        first = logger.log(model_type='ml', metrics={'AUROC': 0.80})
        second = logger.log(model_type='ml', metrics={'AUROC': 0.85})
        assert second.comparison['vs_last'] == pytest.approx(0.05)
        assert second.comparison['vs_best'] == pytest.approx(0.05)
        assert second.comparison['is_new_best'] is True
        assert second.comparison['last_run_id'] == first.run_id
        assert logger.get_best('ml')['run_id'] == second.run_id

    def test_regression_keeps_old_best(self, logger):
        logger.log(model_type='ml', metrics={'AUROC': 0.90})
        worse = logger.log(model_type='ml', metrics={'AUROC': 0.70})
        assert worse.comparison['is_new_best'] is False
        assert worse.comparison['vs_best'] == pytest.approx(-0.20)
        # best.json 仍指向 0.90 那次
        assert logger.get_best('ml')['metrics']['AUROC'] == pytest.approx(0.90)

    def test_model_types_tracked_separately(self, logger):
        logger.log(model_type='ml', metrics={'AUROC': 0.90})
        gnn = logger.log(model_type='gnn', metrics={'AUROC': 0.70})
        # gnn 首次记录即 gnn 的最佳，不与 ml 比较
        assert gnn.comparison['is_new_best'] is True
        assert logger.get_best('ml')['metrics']['AUROC'] == pytest.approx(0.90)
        assert logger.get_best('gnn')['metrics']['AUROC'] == pytest.approx(0.70)

    def test_failure_does_not_join_comparison(self, logger):
        logger.log(model_type='ml', metrics={'AUROC': 0.90})
        failed = logger.log(model_type='ml', status='failed',
                            error_message='boom')
        assert failed.comparison['vs_last'] is None
        assert failed.comparison['is_new_best'] is False
        assert logger.get_best('ml')['metrics']['AUROC'] == pytest.approx(0.90)

    def test_new_best_saves_model_snapshot(self, logger):
        saved_paths = []

        def saver(path):
            saved_paths.append(path)
            with open(path, 'w', encoding='utf-8') as f:
                f.write('fake model')
            return True

        entry = logger.log(model_type='ml', metrics={'AUROC': 0.88},
                           model_saver=saver)
        assert len(saved_paths) == 1
        assert entry.model_path == saved_paths[0]
        assert os.path.exists(entry.model_path)
        assert logger.get_best('ml')['model_path'] == entry.model_path

    def test_no_saver_snapshot_path_empty(self, logger):
        entry = logger.log(model_type='ml', metrics={'AUROC': 0.88})
        assert entry.model_path == ''
        assert logger.get_best('ml')['model_path'] == ''

    # ---- P1-b（归档链修复，2026-09-20）：追平 + 注册 best 无快照 ----

    @staticmethod
    def _saver_factory(saved):
        def saver(path):
            saved.append(path)
            with open(path, 'w', encoding='utf-8') as f:
                f.write('fake model')
            return True
        return saver

    def test_tie_repair_takes_over_artifactless_best(self, logger):
        """注册 best 无快照（t7 型）+ 本次严格追平 → 新 run 携带
        快照接管桶注册（is_new_best 保持 False，追平不是新纪录）。"""
        first = logger.log(model_type='ml', metrics={'AUROC': 0.88})
        assert first.model_path == ''  # 无 saver：注册有结论无归档
        saved = []
        tie = logger.log(model_type='ml', metrics={'AUROC': 0.88},
                         model_saver=self._saver_factory(saved))
        assert tie.comparison['is_new_best'] is False
        assert tie.comparison['vs_best'] == pytest.approx(0.0)
        assert tie.comparison['best_artifact_repair'] is True
        assert len(saved) == 1
        assert tie.model_path == saved[0]
        # best.json 接管到带快照的追平 run（同分新工件）
        best = logger.get_best('ml')
        assert best['run_id'] == tie.run_id
        assert best['model_path'] == saved[0]
        assert best['metrics']['AUROC'] == pytest.approx(0.88)

    def test_tie_with_artifacted_best_no_repair(self, logger):
        """best 已有快照时的追平：不重复快照、不接管（无缺口可修）。"""
        saved = []
        first = logger.log(model_type='ml', metrics={'AUROC': 0.88},
                           model_saver=self._saver_factory(saved))
        tie = logger.log(model_type='ml', metrics={'AUROC': 0.88},
                         model_saver=self._saver_factory(saved))
        assert tie.comparison['is_new_best'] is False
        assert tie.comparison.get('best_artifact_repair') is False
        assert len(saved) == 1  # 仅首次的快照
        assert tie.model_path == ''
        assert logger.get_best('ml')['run_id'] == first.run_id

    def test_tie_repair_without_saver_noop(self, logger):
        """无 saver 的追平：无从产生快照，不触发接管（维持原注册）。"""
        first = logger.log(model_type='ml', metrics={'AUROC': 0.88})
        tie = logger.log(model_type='ml', metrics={'AUROC': 0.88})
        assert tie.comparison.get('best_artifact_repair') is True
        assert tie.model_path == ''
        assert logger.get_best('ml')['run_id'] == first.run_id

    def test_worse_run_never_repairs(self, logger):
        """退步 run 即使 best 无快照也不触发修复（不能以更差工件
        接管更优注册）。"""
        logger.log(model_type='ml', metrics={'AUROC': 0.88})
        saved = []
        worse = logger.log(model_type='ml', metrics={'AUROC': 0.80},
                           model_saver=self._saver_factory(saved))
        assert worse.comparison['is_new_best'] is False
        assert worse.comparison.get('best_artifact_repair') is False
        assert saved == []
        assert logger.get_best('ml')['metrics']['AUROC'] == \
            pytest.approx(0.88)

    def test_result_archived_in_results_dir(self, logger):
        entry = logger.log(model_type='ml', params={'n_samples': 1000},
                           metrics={'AUROC': 0.80})
        archive = os.path.join(logger.results_dir, f'{entry.run_id}.json')
        assert os.path.exists(archive)
        with open(archive, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data['run_id'] == entry.run_id
        assert data['metrics']['AUROC'] == pytest.approx(0.80)

    def test_entry_has_dataset_and_code_version(self, logger):
        entry = logger.log(
            model_type='ml', metrics={'AUROC': 0.80},
            dataset={'source': 'synthetic', 'n_samples': 2000,
                     'real_data': False})
        assert entry.dataset['source'] == 'synthetic'
        assert entry.code_version != ''  # 至少为 'unknown' 或 git 哈希

    def test_legacy_records_without_new_fields(self, logger):
        """旧格式记录（无 run_id 等新字段）仍可读取且不参与对比基准"""
        legacy = {'timestamp': '2026-01-01 00:00:00', 'model_type': 'ml',
                  'params': {}, 'metrics': {'AUROC': 0.60},
                  'training_duration': 1.0, 'status': 'success'}
        with open(logger.log_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(legacy, ensure_ascii=False) + '\n')
        new = logger.log(model_type='ml', metrics={'AUROC': 0.75})
        # 旧记录作为上次基准参与对比
        assert new.comparison['vs_last'] == pytest.approx(0.15)


# ===========================================================================
# _RepoUIMixin 集成测试
# ===========================================================================

class TestRepoUIMixinIntegration:
    """_RepoUIMixin 已正确集成到 Mixin 聚合"""

    def test_ml_training_mixin_has_methods(self):
        from tb_risk.gui.training_panel.ml_tab import MLTrainingMixin
        assert hasattr(MLTrainingMixin, '_show_model_repository')
        assert hasattr(MLTrainingMixin, '_show_training_history')

    def test_training_mixin_has_methods(self):
        from tb_risk.gui.training_panel import TrainingMixin
        assert hasattr(TrainingMixin, '_show_model_repository')
        assert hasattr(TrainingMixin, '_show_training_history')

    def test_repo_ui_mixin_importable(self):
        from tb_risk.gui.training_panel.ml_tab._repo_ui import _RepoUIMixin
        assert callable(_RepoUIMixin._show_model_repository)
        assert callable(_RepoUIMixin._show_training_history)

    def test_ml_tab_module_exports_all_mixins(self):
        """ml_tab/__init__.py 导出所有 5 个 Mixin"""
        from tb_risk.gui.training_panel import ml_tab
        # 验证 MLTrainingMixin 继承所有子 Mixin
        from tb_risk.gui.training_panel.ml_tab._training import _TrainingMixin
        from tb_risk.gui.training_panel.ml_tab._uncertainty import _UncertaintyMixin
        from tb_risk.gui.training_panel.ml_tab._survival import _SurvivalMixin
        from tb_risk.gui.training_panel.ml_tab._rl import _RLMixin
        from tb_risk.gui.training_panel.ml_tab._repo_ui import _RepoUIMixin
        assert issubclass(ml_tab.MLTrainingMixin, _TrainingMixin)
        assert issubclass(ml_tab.MLTrainingMixin, _UncertaintyMixin)
        assert issubclass(ml_tab.MLTrainingMixin, _SurvivalMixin)
        assert issubclass(ml_tab.MLTrainingMixin, _RLMixin)
        assert issubclass(ml_tab.MLTrainingMixin, _RepoUIMixin)
