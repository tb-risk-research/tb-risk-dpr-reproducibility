#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4：checkpoint 版本管理加"标签代际"（label_generation）字段。

背景（用户 2026-08-24 指令）：宿主通路修复换代标签机制后，best.json
跨代比较失效；checkpoint 元数据若不携带"标签代际"字段，未来可能
静默加载错代模型（旧标签 checkpoint 冒充新代主模型）。

机制：
1. ``_train_all_registered_models`` / ``train_from_real_data`` 接受
   ``label_generation`` 参数 → 写入每模型 entry 并挂到 predictor；
2. ``save_model`` 把 predictor.label_generation 写入 joblib payload；
3. ``load_model`` 恢复该属性；若调用方设置
   ``predictor.expected_label_generation``（非 None）且 checkpoint
   已声明代际且不匹配 → 拒绝加载（返回 False）；checkpoint 未声明
   （旧代文件）→ 放行 + warning（向后兼容）。
"""

import os
import sys

import joblib
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tb_risk.scoring.predictor import MLRiskPredictor  # noqa: E402
from tb_risk.scoring.ml.training import _train_all_registered_models  # noqa: E402


def _toy_matrix(n=300, seed=42):
    """22 列带列名 DataFrame + 弱信号标签（快速训练用）。"""
    import pandas as pd
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 22)
    y = (X[:, 0] + 0.5 * X[:, 3] + rng.normal(0, 0.3, n) > 1.1).astype(int)
    names = list(MLRiskPredictor.ALL_FEATURE_NAMES)
    return pd.DataFrame(X, columns=names), y


class TestLabelGeneration:

    @pytest.fixture(scope='class')
    def trained(self):
        X, y = _toy_matrix()
        pred = MLRiskPredictor()
        ok = _train_all_registered_models(
            pred, X, y, random_state=42, data_source_label='synthetic',
            enable_calibration=False, model_keys=['random_forest'],
            label_generation='v3-host-pathway')
        assert ok
        return pred

    def test_entry_and_predictor_attribute(self, trained):
        """训练 entry 携带 label_generation；predictor 挂同一值。"""
        for entry in trained.model_performance.values():
            assert entry['label_generation'] == 'v3-host-pathway'
        assert trained.label_generation == 'v3-host-pathway'

    def test_default_none_keeps_backcompat(self):
        """不传 label_generation（旧调用路径）→ entry 值为 None。"""
        X, y = _toy_matrix(n=200, seed=1)
        pred = MLRiskPredictor()
        ok = _train_all_registered_models(
            pred, X, y, random_state=42, data_source_label='synthetic',
            enable_calibration=False, model_keys=['random_forest'])
        assert ok
        for entry in pred.model_performance.values():
            assert entry['label_generation'] is None

    def test_save_load_roundtrip(self, trained, tmp_path):
        """checkpoint payload 携带 label_generation，load 恢复。"""
        path = str(tmp_path / 'gen.joblib')
        assert trained.save_model(path)
        payload = joblib.load(path)
        assert payload['label_generation'] == 'v3-host-pathway'

        pred2 = MLRiskPredictor()
        assert pred2.load_model(path)
        assert pred2.label_generation == 'v3-host-pathway'

    def test_load_rejects_generation_mismatch(self, trained, tmp_path):
        """expected 与 checkpoint 声明的代际不匹配 → 拒绝加载。"""
        path = str(tmp_path / 'gen_mismatch.joblib')
        assert trained.save_model(path)
        pred = MLRiskPredictor()
        pred.expected_label_generation = 'v2-exposure-only'
        assert pred.load_model(path) is False

    def test_load_accepts_matching_generation(self, trained, tmp_path):
        """expected 匹配 → 正常加载。"""
        path = str(tmp_path / 'gen_match.joblib')
        assert trained.save_model(path)
        pred = MLRiskPredictor()
        pred.expected_label_generation = 'v3-host-pathway'
        assert pred.load_model(path) is True

    def test_legacy_checkpoint_without_generation_passes(self, tmp_path):
        """未声明代际的旧 checkpoint → 放行（向后兼容）。"""
        X, y = _toy_matrix(n=200, seed=3)
        pred = MLRiskPredictor()
        assert _train_all_registered_models(
            pred, X, y, random_state=42, data_source_label='synthetic',
            enable_calibration=False, model_keys=['random_forest'])
        path = str(tmp_path / 'legacy.joblib')
        assert pred.save_model(path)
        payload = joblib.load(path)
        del payload['label_generation']
        joblib.dump(payload, path)

        pred2 = MLRiskPredictor()
        pred2.expected_label_generation = 'v3-host-pathway'
        assert pred2.load_model(path) is True  # 未声明 → 放行

    def test_train_from_real_data_passes_through(self, tmp_path):
        """train_from_real_data 透传 label_generation（CSV 入口）。"""
        import pandas as pd
        X, y = _toy_matrix(n=200, seed=5)
        csv = tmp_path / 'toy.csv'
        df = X.copy()
        df['tb_outcome'] = y
        df.to_csv(csv, index=False)

        pred = MLRiskPredictor()
        res = pred.train_from_real_data(
            str(csv), target_column='tb_outcome',
            label_generation='clinical-rule-v1')
        assert res.get('success')
        assert pred.label_generation == 'clinical-rule-v1'
