#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""迁移学习与预训练 测试套件

覆盖：
1. WHO/国家公开数据目录（WHO_TB_BURDEN_CATALOG / list_pretrain_regions）
2. 预训练数据合成（generate_pretraining_dataset：形状/区域差异/确定性/患病率校准）
3. 预训练（pretrain_models：阶段标记/性能条目/持久化往返）
4. 本地小样本微调（finetune_models：warm_start/init_model/回退/错误分支）
5. 迁移 vs 从零 对比（compare_finetune_vs_from_scratch）
6. 迁移状态摘要（summarize_transfer_status）与包导出完整性
"""
import os
import sys
import tempfile
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

from tb_risk.scoring.predictor import MLRiskPredictor  # noqa: E402

# 依赖可用性（与 conftest 命名一致，避免额外探测）
try:
    from tb_risk.scoring.ml.training import (
        SKLEARN_AVAILABLE, LIGHTGBM_AVAILABLE, CATBOOST_AVAILABLE, XGBOOST_AVAILABLE,
    )
except Exception:  # pragma: no cover
    SKLEARN_AVAILABLE = False
    LIGHTGBM_AVAILABLE = False
    CATBOOST_AVAILABLE = False
    XGBOOST_AVAILABLE = False

try:
    import joblib  # noqa: F401
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from tb_risk.scoring.ml.transfer_learning import (  # noqa: E402
    WHO_TB_BURDEN_CATALOG,
    DEFAULT_PRETRAIN_REGION,
    TRANSFER_STAGE_UNTRAINED,
    TRANSFER_STAGE_PRETRAINED,
    TRANSFER_STAGE_FINETUNED,
    RETIRED_DEPLOYMENT_RECIPES,
    list_pretrain_regions,
    generate_pretraining_dataset,
    pretrain_models,
    finetune_models,
    compare_finetune_vs_from_scratch,
    summarize_transfer_status,
    _finetune_single_model,
)

try:
    from tb_risk.scoring.ml.training import MODEL_REGISTRY
except Exception:  # pragma: no cover
    MODEL_REGISTRY = {}

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

_NEEDS_ML = unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn 不可用，跳过 ML 训练测试")


def _make_predictor(random_state=42):
    return MLRiskPredictor(random_state=random_state)


def _local_samples(predictor, n=80, random_state=7):
    """生成本地细粒度样本（22 维特征 + 0/1 标签），确保两类齐全。"""
    X, y = predictor._generate_synthetic_training_data(n, random_state)
    # 保证至少包含正负两类
    if len(np.unique(y)) < 2:
        y[0] = 1 - y[0]
    return X, y


# ============================================================================
# 1. WHO / 国家公开数据目录
# ============================================================================

class TestWhoCatalog(unittest.TestCase):

    def test_catalog_has_expected_regions(self):
        for key in ('china_national', 'china_karamay', 'india', 'south_africa',
                    'global_high', 'global_low', 'default'):
            self.assertIn(key, WHO_TB_BURDEN_CATALOG)

    def test_catalog_positive_incidence_and_source(self):
        for key, info in WHO_TB_BURDEN_CATALOG.items():
            self.assertGreater(info['incidence_per_100k'], 0)
            self.assertTrue(info['source'])
            self.assertTrue(info['name'])

    def test_catalog_keys_unique(self):
        self.assertEqual(len(WHO_TB_BURDEN_CATALOG),
                         len({k for k in WHO_TB_BURDEN_CATALOG}))

    def test_list_pretrain_regions(self):
        regions = list_pretrain_regions()
        self.assertGreaterEqual(len(regions), 5)
        first = regions[0]
        for field in ('key', 'name', 'incidence_per_100k', 'source'):
            self.assertIn(field, first)

    def test_default_region(self):
        self.assertIn(DEFAULT_PRETRAIN_REGION, WHO_TB_BURDEN_CATALOG)

    def test_transfer_stage_constants(self):
        self.assertEqual(TRANSFER_STAGE_PRETRAINED, 'pretrained')
        self.assertEqual(TRANSFER_STAGE_FINETUNED, 'finetuned')
        self.assertEqual(TRANSFER_STAGE_UNTRAINED, 'untrained')


# ============================================================================
# 2. 预训练数据合成
# ============================================================================

class TestGeneratePretrainingDataset(unittest.TestCase):

    def setUp(self):
        self.predictor = _make_predictor()

    def test_shape_and_labels(self):
        X, y, meta = generate_pretraining_dataset(
            self.predictor, region='china_national', n_samples=500, random_state=1)
        self.assertEqual(X.shape, (500, 22))
        self.assertEqual(y.shape, (500,))
        self.assertEqual(set(np.unique(y)), {0, 1})

    def test_meta_fields(self):
        _, _, meta = generate_pretraining_dataset(
            self.predictor, region='china_karamay', n_samples=300, random_state=2)
        for field in ('region', 'region_name', 'incidence_per_100k', 'source',
                      'target_prevalence', 'actual_positive_rate', 'n_samples'):
            self.assertIn(field, meta)
        self.assertEqual(meta['region'], 'china_karamay')
        self.assertEqual(meta['n_samples'], 300)

    def test_region_incidence_affects_prevalence(self):
        # 高负担区域的目标患病率应高于低负担区域
        _, _, meta_high = generate_pretraining_dataset(
            self.predictor, region='south_africa', n_samples=800, random_state=3)
        _, _, meta_low = generate_pretraining_dataset(
            self.predictor, region='global_low', n_samples=800, random_state=3)
        self.assertGreater(meta_high['target_prevalence'], meta_low['target_prevalence'])

    def test_deterministic_same_seed(self):
        X1, y1, meta1 = generate_pretraining_dataset(
            self.predictor, region='india', n_samples=400, random_state=5)
        X2, y2, meta2 = generate_pretraining_dataset(
            self.predictor, region='india', n_samples=400, random_state=5)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        self.assertEqual(meta1['actual_positive_rate'], meta2['actual_positive_rate'])

    def test_different_regions_give_different_data(self):
        X1, y1, _ = generate_pretraining_dataset(
            self.predictor, region='china_national', n_samples=800, random_state=6)
        X2, y2, _ = generate_pretraining_dataset(
            self.predictor, region='south_africa', n_samples=800, random_state=6)
        # 两个区域分布参数不同（BCG/HIV/检出率），数据不应完全相同
        self.assertGreater(np.abs(X1 - X2).sum(), 1e-6)

    def test_unknown_region_falls_back_to_default(self):
        X, y, meta = generate_pretraining_dataset(
            self.predictor, region='no_such_region', n_samples=200, random_state=7)
        self.assertEqual(X.shape, (200, 22))
        self.assertEqual(meta['region'], 'default')

    def test_small_n_raises(self):
        with self.assertRaises(ValueError):
            generate_pretraining_dataset(
                self.predictor, region='default', n_samples=50, random_state=8)

    def test_returns_feature_names(self):
        _, _, meta = generate_pretraining_dataset(
            self.predictor, region='default', n_samples=300, random_state=9)
        self.assertEqual(len(meta['feature_names']), 22)


# ============================================================================
# 3. 预训练
# ============================================================================

@_NEEDS_ML
class TestPretrainModels(unittest.TestCase):

    def setUp(self):
        self.predictor = _make_predictor()

    def test_pretrain_success_and_stage(self):
        result = pretrain_models(self.predictor, region='china_national',
                                 n_samples=300, random_state=1,
                                 model_keys=['random_forest', 'gradient_boosting'])
        self.assertTrue(result['success'])
        self.assertEqual(result['region'], 'china_national')
        self.assertEqual(self.predictor.transfer_stage, TRANSFER_STAGE_PRETRAINED)
        self.assertTrue(self.predictor.is_trained)
        self.assertTrue(self.predictor.models)

    def test_transfer_info_metadata(self):
        pretrain_models(self.predictor, region='south_africa', n_samples=300,
                        random_state=2, model_keys=['random_forest'])
        info = self.predictor.transfer_info
        self.assertEqual(info['stage'], TRANSFER_STAGE_PRETRAINED)
        self.assertEqual(info['region'], 'south_africa')
        self.assertGreater(info['n_pretrain_samples'], 0)
        self.assertTrue(info['source'])

    def test_model_performance_tagged_pretrain(self):
        pretrain_models(self.predictor, region='default', n_samples=300,
                        random_state=3, model_keys=['random_forest'])
        for entry in self.predictor.model_performance.values():
            self.assertEqual(entry.get('data_source'), 'pretrain')
            self.assertEqual(entry.get('transfer_stage'), TRANSFER_STAGE_PRETRAINED)

    def test_pretrain_save_and_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            path = f.name
        try:
            result = pretrain_models(self.predictor, region='china_karamay',
                                     n_samples=300, random_state=4,
                                     model_keys=['random_forest'], save_path=path)
            self.assertTrue(result['success'])
            self.assertEqual(result['saved_to'], path)
            self.assertTrue(os.path.exists(path))

            # 加载到新 predictor，验证迁移状态往返
            loaded = _make_predictor()
            self.assertTrue(loaded.load_model(path))
            self.assertEqual(loaded.transfer_stage, TRANSFER_STAGE_PRETRAINED)
            self.assertEqual(loaded.transfer_info.get('region'), 'china_karamay')
            self.assertTrue(loaded.models)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_pretrain_sklearn_unavailable_guard(self):
        # 即便 sklearn 不可用也应返回结构化失败而非异常（仅当确实不可用时触发）
        if not SKLEARN_AVAILABLE:
            result = pretrain_models(self.predictor, region='default', n_samples=200)
            self.assertFalse(result['success'])


# ============================================================================
# 4. 本地小样本微调
# ============================================================================

@_NEEDS_ML
class TestFinetuneModels(unittest.TestCase):

    def setUp(self):
        self.predictor = _make_predictor()

    def test_finetune_after_pretrain(self):
        pretrain_models(self.predictor, region='china_national', n_samples=300,
                        random_state=1, model_keys=['random_forest', 'gradient_boosting'])
        X, y = _local_samples(self.predictor, n=60, random_state=2)
        result = finetune_models(self.predictor, X, y, random_state=3)
        self.assertTrue(result['success'])
        self.assertEqual(result['n_finetune_samples'], 60)
        self.assertEqual(self.predictor.transfer_stage, TRANSFER_STAGE_FINETUNED)
        self.assertIn('random_forest', result['models'])

    def test_finetune_warm_start_modes(self):
        pretrain_models(self.predictor, region='default', n_samples=300,
                        random_state=4, model_keys=['random_forest', 'gradient_boosting'])
        X, y = _local_samples(self.predictor, n=60, random_state=5)
        result = finetune_models(self.predictor, X, y, random_state=6)
        for key, detail in result['models'].items():
            self.assertIn(detail['mode'], ('warm_start', 'fallback_from_scratch', 'init_model'))

    def test_finetune_predict_proba_still_works(self):
        pretrain_models(self.predictor, region='default', n_samples=300,
                        random_state=7, model_keys=['random_forest', 'gradient_boosting'])
        X, y = _local_samples(self.predictor, n=60, random_state=8)
        finetune_models(self.predictor, X, y, random_state=9)
        probe = X[:1]
        for model in self.predictor.models.values():
            proba = model.predict_proba(probe)
            self.assertEqual(proba.shape[1], 2)

    def test_finetune_with_pretrained_path(self):
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            path = f.name
        try:
            pretrain_models(self.predictor, region='china_karamay', n_samples=300,
                            random_state=10, model_keys=['random_forest'], save_path=path)
            fresh = _make_predictor()
            X, y = _local_samples(fresh, n=60, random_state=11)
            result = finetune_models(fresh, X, y, random_state=12, pretrained_path=path)
            self.assertTrue(result['success'])
            self.assertEqual(fresh.transfer_info.get('region'), 'china_karamay')
            self.assertEqual(fresh.transfer_stage, TRANSFER_STAGE_FINETUNED)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_finetune_without_pretrain_errors(self):
        X, y = _local_samples(self.predictor, n=60, random_state=13)
        result = finetune_models(self.predictor, X, y, random_state=14)
        self.assertFalse(result['success'])
        self.assertIn('error', result)

    def test_finetune_tiny_data_errors(self):
        pretrain_models(self.predictor, region='default', n_samples=300,
                        random_state=15, model_keys=['random_forest'])
        X, y = _local_samples(self.predictor, n=80, random_state=16)
        result = finetune_models(self.predictor, X[:5], y[:5], random_state=17)
        self.assertFalse(result['success'])

    def test_finetune_single_class_errors(self):
        pretrain_models(self.predictor, region='default', n_samples=300,
                        random_state=18, model_keys=['random_forest'])
        X, y = _local_samples(self.predictor, n=60, random_state=19)
        y_all_zero = np.zeros_like(y)
        result = finetune_models(self.predictor, X, y_all_zero, random_state=20)
        self.assertFalse(result['success'])

    def test_finetune_with_warm_start_disabled(self):
        pretrain_models(self.predictor, region='default', n_samples=300,
                        random_state=21, model_keys=['random_forest', 'gradient_boosting'])
        X, y = _local_samples(self.predictor, n=60, random_state=22)
        result = finetune_models(self.predictor, X, y, random_state=23, warm_start=False)
        self.assertTrue(result['success'])
        for detail in result['models'].values():
            self.assertEqual(detail['mode'], 'fallback_from_scratch')

    def test_finetune_lightgbm_catboost_init_model(self):
        keys = []
        if LIGHTGBM_AVAILABLE:
            keys.append('lightgbm')
        if CATBOOST_AVAILABLE:
            keys.append('catboost')
        if XGBOOST_AVAILABLE:
            keys.append('xgboost')
        if not keys:
            self.skipTest('无 XGBoost/LightGBM/CatBoost 可用')
        pretrain_models(self.predictor, region='default', n_samples=300,
                        random_state=24, model_keys=keys)
        X, y = _local_samples(self.predictor, n=60, random_state=25)
        result = finetune_models(self.predictor, X, y, random_state=26)
        self.assertTrue(result['success'])
        for key in keys:
            detail = result['models'].get(key)
            if detail is None:
                continue
            self.assertIn(detail['mode'], ('init_model', 'fallback_from_scratch', 'warm_start'))


@_NEEDS_ML
class TestFinetuneSingleModelMechanics(unittest.TestCase):
    """微调机制回归（2026-09-14 修复）：真续训 / 特征名对齐 / 确定性。

    修复前三个缺陷：
      1. XGBoost 分支 set_params(warm_start=True) 对 sklearn API 无效，
         fit 实为从零重训（预训练树丢失）却报告 mode='warm_start'；
      2. 命名 DataFrame 预训练 + numpy 微调输入：CatBoost init_model 直接
         "Feature name mismatch" 回退、LightGBM fit(verbose=) TypeError；
      3. XGBoost 以活 booster 续训非确定（内存 sketch/predictor 缓存
         复用），同种子两次调用新增树不同。
    """

    N_TREES = 12

    @classmethod
    def _make_pretrained(cls, key, X, y, seed=1):
        spec = MODEL_REGISTRY.get(key)
        if spec is None or spec.model_cls is None:
            return None
        params = {**spec.default_params, spec.random_state_param: seed}
        if 'n_estimators' in params:
            params['n_estimators'] = cls.N_TREES
        if 'iterations' in params:
            params['iterations'] = cls.N_TREES
        model = spec.model_cls(**params)
        model.fit(X, y)
        return model

    @staticmethod
    def _toy(n, seed, signal_col=0):
        rng = np.random.RandomState(seed)
        X = rng.rand(n, 5)
        y = (X[:, signal_col] > 0.5).astype(int)
        return X, y

    def _n_finetune_trees(self, n_local):
        # _default_finetune_trees: max(10, min(80, n_local // 5))
        return max(10, min(80, n_local // 5))

    def test_xgboost_true_continuation(self):
        if not XGBOOST_AVAILABLE:
            self.skipTest('XGBoost 不可用')
        Xa, ya = self._toy(200, 0, signal_col=0)
        Xb, yb = self._toy(60, 1, signal_col=1)
        m = self._make_pretrained('xgboost', Xa, ya)
        n0 = m.get_booster().num_boosted_rounds()
        m_ft, mode, _detail = _finetune_single_model(
            'xgboost', m, Xb, yb, 42, True, None, 0.05)
        self.assertEqual(mode, 'init_model')
        # 预训练树保留 + 新增（假 warm_start 从零重训只剩新增树数）
        self.assertEqual(m_ft.get_booster().num_boosted_rounds(),
                         n0 + self._n_finetune_trees(len(yb)))
        # 预训练 booster 不被污染
        self.assertEqual(m.get_booster().num_boosted_rounds(), n0)

    def test_numpy_input_aligned_to_pretrained_names(self):
        # 命名 DataFrame 预训练 + numpy 微调输入：修复前 CatBoost 报
        # Feature name mismatch 回退 fallback，LightGBM 报 verbose TypeError
        if pd is None:
            self.skipTest('pandas 不可用')
        keys = [k for k, avail in (('catboost', CATBOOST_AVAILABLE),
                                   ('lightgbm', LIGHTGBM_AVAILABLE))
                if avail]
        if not keys:
            self.skipTest('无 LightGBM/CatBoost 可用')
        for key in keys:
            with self.subTest(key=key):
                rng = np.random.RandomState(2)
                Xa = pd.DataFrame(rng.rand(200, 5),
                                  columns=[f'feat_{i}' for i in range(5)])
                ya = (Xa.iloc[:, 0].values > 0.5).astype(int)
                Xb_np, yb = self._toy(60, 3, signal_col=1)
                m = self._make_pretrained(key, Xa, ya)
                names = getattr(m, 'feature_names_', None)
                if names is None:
                    names = getattr(m, 'feature_names_in_', None)
                self.assertIsNotNone(names, f'{key}: 预训练应带特征名')
                m_ft, mode, _detail = _finetune_single_model(
                    key, m, Xb_np, yb, 42, True, None, 0.05)
                self.assertEqual(mode, 'init_model',
                                 f'{key}: numpy 输入应经特征名对齐后真续训')

    def test_finetune_deterministic_same_seed(self):
        keys = [k for k, avail in (('xgboost', XGBOOST_AVAILABLE),
                                   ('lightgbm', LIGHTGBM_AVAILABLE),
                                   ('catboost', CATBOOST_AVAILABLE))
                if avail]
        if not keys:
            self.skipTest('无 XGBoost/LightGBM/CatBoost 可用')
        for key in keys:
            with self.subTest(key=key):
                Xa, ya = self._toy(200, 4, signal_col=0)
                Xb, yb = self._toy(60, 5, signal_col=1)
                m = self._make_pretrained(key, Xa, ya)
                m1, _, _ = _finetune_single_model(key, m, Xb, yb, 42, True, None, 0.05)
                m2, _, _ = _finetune_single_model(key, m, Xb, yb, 42, True, None, 0.05)
                p1 = m1.predict_proba(Xb)[:, 1]
                p2 = m2.predict_proba(Xb)[:, 1]
                np.testing.assert_allclose(p1, p2, err_msg=f'{key} 同种子微调应确定')

    def test_random_forest_warm_start_retired(self):
        # RF warm_start 分支下线（2026-09-15，S8 证据：taiwan −0.098 /
        # crp −0.055 负迁移显著且机制清楚）——RF 微调须走
        # fallback_from_scratch：从零拟合注册表默认参（300 树），
        # 不再保留预训练树续加（旧路径为 12 预训练树 + 12 新增树）
        Xa, ya = self._toy(200, 6, signal_col=0)
        Xb, yb = self._toy(60, 7, signal_col=1)
        m = self._make_pretrained('random_forest', Xa, ya)
        m_ft, mode, detail = _finetune_single_model(
            'random_forest', m, Xb, yb, 42, True, None, 0.05)
        self.assertEqual(mode, 'fallback_from_scratch')
        n_default = MODEL_REGISTRY['random_forest'].default_params['n_estimators']
        self.assertEqual(len(m_ft.estimators_), n_default)
        # 预训练模型不被污染
        self.assertEqual(len(m.estimators_), self.N_TREES)

    def test_xgboost_init_model_recipe_retired(self):
        # Kenya→Taiwan XGBoost init_model 微调配方退役（P4，2026-09-16）：
        # S8 22 维口径 +0.067（0.481→0.547）被 taiwan 本地 v4 LR 0.874
        # （主管线）/0.856（F2 口径）支配——confirmed_tb 族一律本地 v4
        # 部署。退役是「部署配方」身份的退役：init_model 续训机制保留为
        # 库能力（下方机制冒烟），重启须在 v4 特征空间下重新评估。
        entry = RETIRED_DEPLOYMENT_RECIPES['kenya_to_taiwan_xgboost_init_model']
        self.assertEqual(entry['retired'], '2026-09-16')
        self.assertIn('0.874', entry['evidence'])
        self.assertIn('本地 v4', entry['successor'])
        self.assertIn('v4', entry['restart_condition'])
        # 机制保留：XGBoost init_model 续训仍可用（库能力未下线）
        if XGBOOST_AVAILABLE:
            Xa, ya = self._toy(200, 6, signal_col=0)
            Xb, yb = self._toy(60, 7, signal_col=1)
            m = self._make_pretrained('xgboost', Xa, ya)
            m_ft, mode, _ = _finetune_single_model(
                'xgboost', m, Xb, yb, 42, True, None, 0.05)
            self.assertEqual(mode, 'init_model')
            self.assertGreater(len(m_ft.get_booster().get_dump()), 0)


# ============================================================================
# 5. 迁移 vs 从零 对比
# ============================================================================

@_NEEDS_ML
class TestCompareFinetuneVsFromScratch(unittest.TestCase):

    def test_compare_returns_structure(self):
        predictor = _make_predictor()
        X, y = _local_samples(predictor, n=80, random_state=30)
        result = compare_finetune_vs_from_scratch(
            predictor, X, y, region='china_national', n_pretrain=300,
            random_state=31, holdout_frac=0.3)
        self.assertTrue(result['success'])
        self.assertEqual(result['n_train'] + result['n_holdout'], 80)
        for branch in ('from_scratch', 'finetuned'):
            self.assertIn(branch, result)
            self.assertIn('AUROC', result[branch])
        self.assertIn('recommendation', result)
        # 对比后 predictor 应处于"预训练+微调"状态
        self.assertEqual(predictor.transfer_stage, TRANSFER_STAGE_FINETUNED)

    def test_compare_too_small_errors(self):
        predictor = _make_predictor()
        X, y = _local_samples(predictor, n=20, random_state=32)
        result = compare_finetune_vs_from_scratch(
            predictor, X, y, n_pretrain=200, holdout_frac=0.9)
        # 0.9 留出 → train 只有 2 个，应失败
        self.assertFalse(result['success'])


# ============================================================================
# 6. 迁移状态摘要与导出完整性
# ============================================================================

class TestTransferStatusAndExports(unittest.TestCase):

    def test_untrained_status(self):
        predictor = _make_predictor()
        status = summarize_transfer_status(predictor)
        self.assertEqual(status['stage'], TRANSFER_STAGE_UNTRAINED)
        self.assertTrue(status['message'])

    @_NEEDS_ML
    def test_pretrained_status(self):
        predictor = _make_predictor()
        pretrain_models(predictor, region='china_national', n_samples=200,
                        random_state=1, model_keys=['random_forest'])
        status = summarize_transfer_status(predictor)
        self.assertEqual(status['stage'], TRANSFER_STAGE_PRETRAINED)
        self.assertIn('中国', status['message'])

    @_NEEDS_ML
    def test_finetuned_status(self):
        predictor = _make_predictor()
        pretrain_models(predictor, region='default', n_samples=200,
                        random_state=2, model_keys=['random_forest'])
        X, y = _local_samples(predictor, n=60, random_state=3)
        finetune_models(predictor, X, y, random_state=4)
        status = summarize_transfer_status(predictor)
        self.assertEqual(status['stage'], TRANSFER_STAGE_FINETUNED)
        self.assertIn('本地微调样本', status['message'])

    def test_module_exports(self):
        import tb_risk.scoring.ml as ml
        for name in ('WHO_TB_BURDEN_CATALOG', 'DEFAULT_PRETRAIN_REGION',
                     'list_pretrain_regions', 'generate_pretraining_dataset',
                     'pretrain_models', 'finetune_models',
                     'compare_finetune_vs_from_scratch', 'summarize_transfer_status'):
            self.assertTrue(hasattr(ml, name), f"scoring.ml 缺少导出 {name}")

    def test_predictor_delegates(self):
        predictor = _make_predictor()
        for name in ('pretrain_models', 'finetune_models',
                     'compare_finetune_vs_from_scratch',
                     'summarize_transfer_status', 'generate_pretraining_dataset'):
            self.assertTrue(hasattr(predictor, name), f"MLRiskPredictor 缺少委托方法 {name}")
        # 未训练时 summarize 应正常返回（不抛异常）
        self.assertEqual(predictor.summarize_transfer_status()['stage'], TRANSFER_STAGE_UNTRAINED)


if __name__ == '__main__':
    unittest.main()
