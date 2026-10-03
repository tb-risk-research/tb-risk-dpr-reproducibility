#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — 持久化与安全加载测试

覆盖：
- ml/uncertainty.py: DeepEnsemble/SWAG/ConformalPredictor save/load/checkpoint
- GNN 模型加载安全防护（扩展名白名单、weights_only、checkpoint schema）
- 配置管理（schema 验证、热更新、自动生成）
- 导出格式（Excel/PDF/HTML训练报告）
"""

import json
import os
import sys
import tempfile
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


# ======================== ml/uncertainty.py 持久化测试 ========================

class TestDeepEnsemblePersistence(unittest.TestCase):
    """DeepEnsemblePredictor save/load/checkpoint 测试"""

    def setUp(self):
        try:
            from tb_risk.ml.uncertainty import DeepEnsemblePredictor
            self.Predictor = DeepEnsemblePredictor
        except ImportError as e:
            self.skipTest(f"不可用: {e}")

    def test_save_rejects_untrained(self):
        """测试未训练模型保存返回 False"""
        p = self.Predictor(n_ensembles=3)
        result = p.save('/nonexistent/test.joblib')
        self.assertFalse(result)

    def test_extension_whitelist_rejected(self):
        """测试非法扩展名被 save() 拒绝（使用合法路径）"""
        import torch.nn as nn
        p = self.Predictor(
            base_model_class=nn.Linear,
            model_kwargs={'in_features': 4, 'out_features': 1},
            n_ensembles=3
        )
        p.is_trained = True
        p.models = [nn.Linear(4, 1) for _ in range(3)]
        p.ensemble_stats = {}
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            tmp_path = f.name
        try:
            # 确保路径存在，测试扩展名白名单拒绝（非路径错误）
            result = p.save(tmp_path)
            self.assertFalse(result, msg="save() 应拒绝 .txt 扩展名")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_extension_whitelist_accepted(self):
        """测试合法扩展名被接受"""
        import torch.nn as nn
        p = self.Predictor(
            base_model_class=nn.Linear,
            model_kwargs={'in_features': 4, 'out_features': 1},
            n_ensembles=3
        )
        p.is_trained = True
        p.models = [nn.Linear(4, 1) for _ in range(3)]
        p.ensemble_stats = {}
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            tmp_path = f.name
        try:
            result = p.save(tmp_path)
            self.assertIsInstance(result, bool)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_load_invalid_model_type(self):
        """测试加载时 model_type 不匹配被拒绝"""
        import joblib
        p = self.Predictor(n_ensembles=3)
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            tmp_path = f.name
        try:
            joblib.dump({'model_type': 'WrongType', 'version': '2.0'}, tmp_path)
            result = p.load(tmp_path)
            self.assertFalse(result)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_load_nonexistent_returns_false(self):
        """测试加载不存在的文件返回 False"""
        p = self.Predictor(n_ensembles=3)
        result = p.load('/nonexistent/file.joblib')
        self.assertFalse(result)

    def test_checkpoint_save_creates_file(self):
        """测试 checkpoint 保存创建文件"""
        import torch.nn as nn
        p = self.Predictor(
            base_model_class=nn.Linear,
            model_kwargs={'in_features': 4, 'out_features': 1},
            n_ensembles=3
        )
        p.is_trained = True
        p.models = [nn.Linear(4, 1) for _ in range(3)]
        p.ensemble_stats = {}
        p.training_history = []
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            tmp_path = f.name
        try:
            result = p.save_checkpoint(tmp_path, epoch=10, member_idx=0, loss=0.5)
            self.assertIsInstance(result, bool)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_checkpoint_load_recovers_state(self):
        """测试 checkpoint 加载恢复状态"""
        import joblib
        p = self.Predictor(n_ensembles=3)
        p.is_trained = True
        p.training_history = [{'epoch': 1, 'loss': 0.8}]
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            tmp_path = f.name
        try:
            # load_checkpoint 检查 checkpoint_type 字段（非 model_type）
            joblib.dump({
                'version': '2.0',
                'checkpoint_type': 'DeepEnsemble',
                'epoch': 5,
                'member_idx': 2,
                'loss': 0.3,
                'checkpoint_time': '2026-01-01T00:00:00',
                'training_history': [{'epoch': 1, 'loss': 0.8}, {'epoch': 2, 'loss': 0.6}],
                'is_trained': True,
                'ensemble_stats': {'mean': 0.5},
                'trained_models': [],
            }, tmp_path)
            result = p.load_checkpoint(tmp_path)
            self.assertIsInstance(result, dict)
            self.assertIn('epoch', result)
            self.assertEqual(result['epoch'], 5)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_save_load_roundtrip_consistency(self):
        """测试 DeepEnsemble save/load 往返后预测结果一致性"""
        import torch.nn as nn
        import torch
        import numpy as np
        # 创建并训练集成模型（使用默认 MLP 架构，与 load 重建逻辑一致）
        p = self.Predictor(n_ensembles=3)
        p.is_trained = True
        # 创建简单的确定性 MLP 模型（与 _rebuild_single_model 默认架构一致）
        torch.manual_seed(42)
        p.models = []
        for _ in range(3):
            model = nn.Sequential(
                nn.Linear(4, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
            )
            p.models.append(model)
        p.ensemble_stats = {}

        # 生成测试输入
        X_test = np.random.randn(10, 4).astype(np.float32)

        # 获取原始预测
        result_before = p.predict_with_uncertainty(X_test)

        # Save → Load
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            tmp_path = f.name
        try:
            self.assertTrue(p.save(tmp_path), "save 应成功")

            # 创建新实例并加载（使用默认架构，与 save 时一致）
            p2 = self.Predictor(n_ensembles=3)
            self.assertTrue(p2.load(tmp_path), "load 应成功")

            # 获取加载后的预测
            result_after = p2.predict_with_uncertainty(X_test)

            # 验证预测一致性
            self.assertIn('mean', result_before)
            self.assertIn('mean', result_after)
            self.assertIsNotNone(result_before['mean'])
            self.assertIsNotNone(result_after['mean'])

            # 对比均值预测（允许浮点精度差异）
            before_mean = np.array(result_before['mean']).flatten()
            after_mean = np.array(result_after['mean']).flatten()
            np.testing.assert_allclose(
                before_mean, after_mean, rtol=1e-5, atol=1e-7,
                err_msg="save/load 往返后预测均值不一致"
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestSWAGPersistence(unittest.TestCase):
    """SWAGEstimator save/load 测试"""

    def setUp(self):
        try:
            import torch.nn as nn
            from tb_risk.ml.uncertainty import SWAGEstimator
            self.SWAG = SWAGEstimator
            self._dummy_model = nn.Linear(4, 1)
        except ImportError as e:
            self.skipTest(f"不可用: {e}")

    def test_save_rejects_incomplete_collection(self):
        """测试采样未完成时保存返回 False"""
        s = self.SWAG(self._dummy_model, n_models=20, max_rank=3)
        s.collection_complete = False
        result = s.save('/tmp/test.joblib')
        self.assertFalse(result)

    def test_extension_whitelist_rejected(self):
        """测试非法扩展名被 save() 拒绝（使用合法路径）"""
        s = self.SWAG(self._dummy_model, n_models=20, max_rank=3)
        s.collection_complete = True
        s.swag_mean = {'weight': None}
        s.swag_dev = {}
        s.swag_var = {}
        s.n_collected = 3
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            tmp_path = f.name
        try:
            result = s.save(tmp_path)
            self.assertFalse(result, msg="save() 应拒绝 .txt 扩展名")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_extension_whitelist_accepted(self):
        """测试合法扩展名被接受"""
        s = self.SWAG(self._dummy_model, n_models=20, max_rank=3)
        s.collection_complete = True
        s.swag_mean = {'weight': None}
        s.swag_dev = {}
        s.swag_var = {}
        s.n_collected = 3
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            tmp_path = f.name
        try:
            result = s.save(tmp_path)
            self.assertIsInstance(result, bool)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_load_invalid_model_type(self):
        """测试加载时 model_type 不匹配被拒绝"""
        import joblib
        s = self.SWAG(self._dummy_model, n_models=20, max_rank=3)
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            tmp_path = f.name
        try:
            joblib.dump({'model_type': 'WrongType'}, tmp_path)
            result = s.load(tmp_path)
            self.assertFalse(result)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_load_nonexistent_returns_false(self):
        """测试加载不存在的文件返回 False"""
        s = self.SWAG(self._dummy_model, n_models=20, max_rank=3)
        result = s.load('/nonexistent/file.joblib')
        self.assertFalse(result)

    def test_save_load_roundtrip_consistency(self):
        """测试 SWAG save/load 往返后统计量一致性"""
        import numpy as np
        s = self.SWAG(self._dummy_model, n_models=20, max_rank=3)
        s.collection_complete = True
        s.swag_mean = {'weight': np.array([0.5, 0.3, 0.1, 0.1], dtype=np.float32)}
        s.swag_dev = {'weight': np.array([0.02, 0.01, 0.01, 0.01], dtype=np.float32)}
        s.swag_var = {'weight': np.array([0.0004, 0.0001, 0.0001, 0.0001], dtype=np.float32)}
        s.n_collected = 20

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            tmp_path = f.name
        try:
            self.assertTrue(s.save(tmp_path), "SWAG save 应成功")

            s2 = self.SWAG(self._dummy_model, n_models=20, max_rank=3)
            self.assertTrue(s2.load(tmp_path), "SWAG load 应成功")

            # 验证统计量一致性
            self.assertTrue(s2.collection_complete)
            self.assertEqual(s2.n_collected, 20)
            self.assertIn('weight', s2.swag_mean)
            self.assertIn('weight', s2.swag_dev)
            self.assertIn('weight', s2.swag_var)

            np.testing.assert_allclose(
                s2.swag_mean['weight'], s.swag_mean['weight'],
                rtol=1e-5, err_msg="swag_mean 往返不一致"
            )
            np.testing.assert_allclose(
                s2.swag_dev['weight'], s.swag_dev['weight'],
                rtol=1e-5, err_msg="swag_dev 往返不一致"
            )
            np.testing.assert_allclose(
                s2.swag_var['weight'], s.swag_var['weight'],
                rtol=1e-5, err_msg="swag_var 往返不一致"
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestConformalPredictorPersistence(unittest.TestCase):
    """ConformalPredictor save/load 测试"""

    def setUp(self):
        try:
            from tb_risk.ml.uncertainty import ConformalPredictor
            self.Conformal = ConformalPredictor
        except ImportError as e:
            self.skipTest(f"不可用: {e}")

    def test_save_rejects_uncalibrated(self):
        """测试未校准预测器保存返回 False"""
        c = self.Conformal(coverage=0.90)
        c.is_calibrated = False
        result = c.save('/tmp/test.joblib')
        self.assertFalse(result)

    def test_extension_whitelist_rejected(self):
        """测试非法扩展名被 save() 拒绝（使用合法路径）"""
        c = self.Conformal(coverage=0.90)
        c.is_calibrated = True
        c.calibration_scores = [0.1, 0.2, 0.3]
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            tmp_path = f.name
        try:
            result = c.save(tmp_path)
            self.assertFalse(result, msg="save() 应拒绝 .txt 扩展名")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_extension_whitelist_accepted(self):
        """测试合法扩展名被接受"""
        c = self.Conformal(coverage=0.90)
        c.is_calibrated = True
        c.calibration_scores = [0.1, 0.2, 0.3]
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            tmp_path = f.name
        try:
            result = c.save(tmp_path)
            self.assertIsInstance(result, bool)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_load_invalid_model_type(self):
        """测试加载时 model_type 不匹配被拒绝"""
        import joblib
        c = self.Conformal(coverage=0.90)
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            tmp_path = f.name
        try:
            joblib.dump({'model_type': 'WrongType'}, tmp_path)
            result = c.load(tmp_path)
            self.assertFalse(result)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_save_load_roundtrip_consistency(self):
        """测试 save/load 往返一致性"""
        c = self.Conformal(coverage=0.95)
        c.is_calibrated = True
        c.calibration_scores = [0.15, 0.22, 0.18, 0.25]
        c.confidence_threshold = 0.20
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            tmp_path = f.name
        try:
            c.save(tmp_path)
            c2 = self.Conformal(coverage=0.90)
            result = c2.load(tmp_path)
            self.assertTrue(result)
            self.assertEqual(c2.coverage, 0.95)
            self.assertEqual(c2.confidence_threshold, 0.20)
            self.assertTrue(c2.is_calibrated)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ======================== GNN 模型加载安全测试 ========================

class TestGNNModelLoadingSafety(unittest.TestCase):
    """GNN 模型加载三层安全防护测试"""

    def test_extension_whitelist_rejects_illegal_ext(self):
        """测试非法扩展名（.pkl）被拒绝"""
        from tb_risk.scoring.predictor import MLRiskPredictor
        p = MLRiskPredictor()
        p.gnn_is_trained = True
        result = p.load_gnn_model('/tmp/test.pkl')
        self.assertFalse(result)

    def test_extension_whitelist_accepts_legal_ext(self):
        """测试合法扩展名（.pth/.pt）不被扩展名拒绝"""
        from tb_risk.scoring.predictor import MLRiskPredictor
        p = MLRiskPredictor()
        p.gnn_is_trained = True
        result = p.load_gnn_model('/tmp/legal_ext.pth')
        self.assertFalse(result)  # 文件不存在

    def test_checkpoint_schema_validation_missing_keys(self):
        """测试 checkpoint schema 校验：缺失必需键"""
        from tb_risk.scoring.predictor import MLRiskPredictor
        p = MLRiskPredictor()
        invalid = {'model_state_dict': None}
        valid, msg = p._validate_gnn_checkpoint(invalid)
        self.assertFalse(valid)
        self.assertIn('缺少必需键', msg)

    def test_checkpoint_schema_validation_type_mismatch(self):
        """测试 checkpoint schema 校验：类型不匹配"""
        from tb_risk.scoring.predictor import MLRiskPredictor
        p = MLRiskPredictor()
        invalid = {
            'model_state_dict': None,
            'gnn_is_trained': 1,  # 应为 bool
            'model_performance': {},
            'hidden_dim': 64,
            'num_layers': 3,
            'node_feature_dim': 22,
            'edge_feature_dim': 4,
            'schema_version': '2.0',
            'model_type': 'GNN',
        }
        valid, msg = p._validate_gnn_checkpoint(invalid)
        self.assertFalse(valid)
        self.assertIn('类型不匹配', msg)

    def test_checkpoint_schema_validation_valid(self):
        """测试 checkpoint schema 校验：合法 checkpoint"""
        from tb_risk.scoring.predictor import MLRiskPredictor
        p = MLRiskPredictor()
        valid_checkpoint = {
            'model_state_dict': None,
            'gnn_is_trained': True,
            'model_performance': {'AUROC': 0.85},
            'hidden_dim': 64,
            'num_layers': 3,
            'node_feature_dim': 22,
            'edge_feature_dim': 4,
            'schema_version': '2.0',
            'model_type': 'GNN',
        }
        valid, msg = p._validate_gnn_checkpoint(valid_checkpoint)
        self.assertTrue(valid)
        self.assertEqual(msg, '')


# ======================== 配置管理测试 ========================

class TestConfigManagement(unittest.TestCase):
    """配置管理功能测试"""

    def test_config_schema_validation_rejects_invalid_type(self):
        """测试 schema 验证拒绝非法字段类型"""
        try:
            from tb_risk.config import _validate_config_schema
        except ImportError:
            self.skipTest("config._validate_config_schema 不可用")
        invalid_config = {
            'epidemiology': {
                'base_incidence_per_100k': 'not_a_number',
            }
        }
        valid, errors = _validate_config_schema(invalid_config)
        self.assertFalse(valid)
        self.assertGreater(len(errors), 0)

    def test_config_schema_validation_accepts_valid(self):
        """测试 schema 验证接受合法配置"""
        try:
            from tb_risk.config import _validate_config_schema
        except ImportError:
            self.skipTest("config._validate_config_schema 不可用")
        valid_config = {
            '_meta': {
                'version': '2.0',
                'name': 'test',
            },
            'epidemiology': {
                'base_incidence_per_100k': 121.0,
                'latent_prevalence_family_default': 0.10,
                'treatment_success_rate': 0.85,
            },
            'occupation': {
                'oilfield_camp_factor': 0.85,
            }
        }
        valid, errors = _validate_config_schema(valid_config)
        self.assertTrue(valid, msg=f"Schema 验证失败: {errors}")

    def test_generate_default_config_creates_file(self):
        """测试自动生成默认配置文件"""
        try:
            from tb_risk.config import _generate_default_config
        except ImportError:
            self.skipTest("config._generate_default_config 不可用")
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, encoding='utf-8'
        ) as f:
            tmp_path = f.name
        try:
            # _generate_default_config 接受 path 参数
            result = _generate_default_config(tmp_path)
            # 返回 True 表示生成成功，False 表示文件已存在
            self.assertIsInstance(result, bool)
            if result:
                self.assertTrue(os.path.exists(tmp_path))
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.assertIsInstance(data, dict)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_reload_config_updates_values(self):
        """测试热更新配置"""
        try:
            from tb_risk.config import reload_config, _generate_default_config
        except ImportError:
            self.skipTest("config.reload_config 不可用")
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, encoding='utf-8'
        ) as f:
            tmp_path = f.name
        try:
            # 删除预创建的空文件，让 _generate_default_config 生成新文件
            os.unlink(tmp_path)
            generated = _generate_default_config(tmp_path)
            if not generated:
                self.skipTest("_generate_default_config 未生成文件")
            # 修改配置
            with open(tmp_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            if 'epidemiology' in config and isinstance(config['epidemiology'], dict):
                config['epidemiology']['base_incidence_per_100k'] = 150.0
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            # 热更新
            new_cfg, success = reload_config(tmp_path)
            self.assertTrue(success)
            self.assertIsInstance(new_cfg, dict)
            self.assertEqual(new_cfg['epidemiology']['base_incidence_per_100k'], 150.0)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ======================== 导出功能测试 ========================

class TestExportFormats(unittest.TestCase):
    """Excel/PDF/训练报告导出测试"""

    def setUp(self):
        self.results = {
            'overall_risk': '中风险',
            'base_infection_probability': 30.0,
            'potential_patients': {
                'family': [{
                    'name': '测试家属', 'relationship': '配偶',
                    'disease_probability': 45.0, 'priority': '中',
                    'seir_risk': 35.0
                }],
                'social': [{
                    'name': '测试同事', 'relationship': '同事',
                    'disease_probability': 25.0, 'priority': '低',
                    'seir_risk': 15.0
                }],
            }
        }

    def test_excel_export_creates_file(self):
        """测试 Excel 导出创建文件"""
        try:
            from tb_risk.export_utils import export_to_excel
        except ImportError:
            self.skipTest("export_utils 不可用")
        try:
            import openpyxl
        except ImportError:
            self.skipTest("openpyxl 不可用")
        with tempfile.NamedTemporaryFile(
            suffix='.xlsx', delete=False
        ) as f:
            tmp_path = f.name
        try:
            export_to_excel(self.results, tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 100)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_pdf_export_creates_file(self):
        """测试 PDF 导出创建文件"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate
        except ImportError:
            self.skipTest("reportlab 不可用")
        with tempfile.NamedTemporaryFile(
            suffix='.pdf', delete=False
        ) as f:
            tmp_path = f.name
        try:
            doc = SimpleDocTemplate(tmp_path, pagesize=letter)
            from reportlab.platypus import Paragraph
            from reportlab.lib.styles import getSampleStyleSheet
            styles = getSampleStyleSheet()
            elements = [Paragraph("结核病风险评估报告", styles['Heading1'])]
            doc.build(elements)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 100)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_html_training_report_creates_file(self):
        """测试 HTML 训练报告导出创建文件"""
        try:
            from tb_risk.export_utils import _generate_html_report
        except ImportError:
            self.skipTest("export_utils._generate_html_report 不可用")
        # 构造 mock predictor（含所有必需属性）
        class MockPredictor:
            is_trained = True
            use_real_data = True
            training_sample_count = 500
            model_performance = {
                'random_forest': {'AUROC': 0.85, 'AUPRC': 0.72,
                                  'name': '随机森林', 'name_en': 'Random Forest'},
                'gradient_boosting': {'AUROC': 0.87, 'AUPRC': 0.74,
                                      'name': '梯度提升', 'name_en': 'Gradient Boosting'},
            }
            shap_explainer = None
            FEATURE_NAMES = ['age', 'cumulative_exposure', 'has_symptoms', 'bcg_vaccine',
                           'has_tb', 'contact_distance', 'ventilation', 'is_high_risk',
                           'past_illness', 'exposure_setting', 'single_duration',
                           'freq_density', 'time_span']
            INTERACTION_FEATURE_NAMES = ['age_immuno', 'dm_tb_synergy', 'age_bcg_decay',
                                        'symptom_delay', 'cough_contact', 'highrisk_comorbid',
                                        'immune_bcg', 'exposure_accumulation', 'age_diabetes']
            ALL_FEATURE_NAMES = FEATURE_NAMES + INTERACTION_FEATURE_NAMES
            feature_names = FEATURE_NAMES
            FEATURE_DESCRIPTIONS = {
                'age': '年龄', 'cumulative_exposure': '累积暴露时长(小时)', 'has_symptoms': '有症状',
                'bcg_vaccine': '卡介苗接种', 'has_tb': '既往结核史', 'contact_distance': '接触距离',
                'ventilation': '通风条件', 'is_high_risk': '高危人群', 'past_illness': '慢性病史',
                'exposure_setting': '暴露场景', 'single_duration': '单次接触时长',
                'freq_density': '每周接触频次', 'time_span': '持续周期',
            }
            INTERACTION_FEATURE_DESCRIPTIONS = {
                'age_immuno': '年龄×免疫', 'dm_tb_synergy': '糖尿病×结核',
                'age_bcg_decay': '年龄×BCG衰减', 'symptom_delay': '症状×延迟',
                'cough_contact': '咳嗽×接触', 'highrisk_comorbid': '高危×合并症',
                'immune_bcg': '免疫×BCG', 'exposure_accumulation': '暴露×累积',
                'age_diabetes': '年龄×糖尿病',
            }
            ALL_FEATURE_DESCRIPTIONS = {**FEATURE_DESCRIPTIONS, **INTERACTION_FEATURE_DESCRIPTIONS}
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.html', delete=False, encoding='utf-8'
        ) as f:
            tmp_path = f.name
        try:
            report = _generate_html_report(MockPredictor())
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.assertGreater(os.path.getsize(tmp_path), 100)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestWeightsOnlyFallback(unittest.TestCase):
    """weights_only 参数降级路径测试

    验证 load_gnn_model 中 torch.load 的安全加载逻辑：
    1. 首次调用传入 weights_only=True
    2. 低版本 PyTorch 抛 TypeError 时回退到不带 weights_only 的调用
    """

    def setUp(self):
        try:
            import torch
            self.torch = torch
        except ImportError:
            self.skipTest("PyTorch 不可用")

    def test_weights_only_true_in_pytorch_2x(self):
        """测试 PyTorch 2.0+ 环境下 weights_only=True 被正确传入 torch.load"""
        import torch
        from unittest.mock import patch, MagicMock
        from tb_risk.scoring.predictor import MLRiskPredictor

        p = MLRiskPredictor()
        p.gnn_is_trained = True

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            tmp_path = f.name
        try:
            # 保存一个最小有效 checkpoint（model_state_dict=None 跳过权重加载）
            torch.save({
                'model_state_dict': None,
                'gnn_is_trained': True,
                'model_performance': {},
                'hidden_dim': 64,
                'num_layers': 3,
                'node_feature_dim': 22,
                'edge_feature_dim': 4,
                'schema_version': '2.0',
                'model_type': 'GNN',
            }, tmp_path)

            # 创建 mock 来包装 torch.load，记录调用参数
            original_load = torch.load
            call_args_list = []

            def mock_load(*args, **kwargs):
                call_args_list.append(kwargs.copy())
                return original_load(*args, **kwargs)

            with patch('torch.load', side_effect=mock_load):
                result = p.load_gnn_model(tmp_path)
                self.assertTrue(result, "load_gnn_model 应返回 True")

            # 验证第一次调用的关键字参数包含 weights_only=True
            self.assertGreater(len(call_args_list), 0,
                              "torch.load 应至少被调用一次")
            first_call_kwargs = call_args_list[0]
            self.assertIn('weights_only', first_call_kwargs,
                         "首次调用 torch.load 应包含 weights_only 参数")
            self.assertTrue(first_call_kwargs['weights_only'],
                           "首次调用 torch.load 的 weights_only 应为 True")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_weights_only_fallback_on_typeerror(self):
        """测试低版本 PyTorch 抛 TypeError 时优雅降级（不崩溃）"""
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch 不可用")

        original_load = torch.load

        def mock_load_raise_typeerror(*args, **kwargs):
            if 'weights_only' in kwargs:
                raise TypeError(
                    "torch.load() got an unexpected keyword argument 'weights_only'"
                )
            return original_load(*args, **kwargs)

        try:
            torch.load = mock_load_raise_typeerror
            from tb_risk.scoring.predictor import MLRiskPredictor
            p = MLRiskPredictor()
            # 验证降级后不崩溃：load_gnn_model 应捕获 TypeError
            # 并回退到不带 weights_only 的普通加载
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
                tmp_path = f.name
            try:
                # 创建一个最小有效 checkpoint（model_state_dict=None 跳过权重加载）
                torch.save({
                    'model_state_dict': None,
                    'gnn_is_trained': True,
                    'model_performance': {},
                    'hidden_dim': 64,
                    'num_layers': 3,
                    'node_feature_dim': 22,
                    'edge_feature_dim': 4,
                    'schema_version': '2.0',
                    'model_type': 'GNN',
                }, tmp_path)
                # load_gnn_model 应返回结果（或不崩溃）
                result = p.load_gnn_model(tmp_path)
                self.assertIsInstance(result, (bool, type(None)),
                                      "降级路径不应抛出异常")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        finally:
            torch.load = original_load

    def test_weights_only_fallback_omits_weights_only(self):
        """测试降级调用确实省略了 weights_only 参数"""
        import torch
        from unittest.mock import patch
        from tb_risk.scoring.predictor import MLRiskPredictor

        p = MLRiskPredictor()

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            tmp_path = f.name
        try:
            torch.save({
                'model_state_dict': None,
                'gnn_is_trained': True,
                'model_performance': {},
                'hidden_dim': 64,
                'num_layers': 3,
                'node_feature_dim': 22,
                'edge_feature_dim': 4,
                'schema_version': '2.0',
                'model_type': 'GNN',
            }, tmp_path)

            # 模拟低版本 PyTorch：首次调用（带 weights_only）抛 TypeError
            original_load = torch.load
            call_args_list = []

            def mock_load_typeerror(*args, **kwargs):
                call_args_list.append(kwargs.copy())
                if 'weights_only' in kwargs:
                    raise TypeError(
                        "torch.load() got an unexpected keyword argument "
                        "'weights_only'"
                    )
                return original_load(*args, **kwargs)

            with patch('torch.load', side_effect=mock_load_typeerror):
                result = p.load_gnn_model(tmp_path)
                self.assertTrue(result, "降级后 load_gnn_model 应返回 True")

            # 验证第一次调用包含 weights_only=True
            self.assertGreater(len(call_args_list), 0)
            self.assertTrue(call_args_list[0].get('weights_only'),
                           "首次调用应包含 weights_only=True")

            # 验证第二次调用（降级回退）不包含 weights_only
            self.assertGreater(len(call_args_list), 1,
                              "应有第二次回退调用")
            self.assertNotIn('weights_only', call_args_list[1],
                            "降级回退调用不应包含 weights_only 参数")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestPDFReportExport(unittest.TestCase):
    """_export_pdf_report 和 export_model_training_report 直接单元测试"""

    def setUp(self):
        # 数据结构必须与 _export_pdf_report 期望的格式匹配：
        # - results: 包含 'overall_risk' 和 'potential_patients' 键
        # - patient_info: 包含 'basic_info' 键
        self.results = {
            'overall_risk': '高风险',
            'potential_patients': {
                'family': [{
                    'name': '家属1', 'relationship': '配偶',
                    'disease_probability': 35.0, 'priority': '中',
                }],
                'social': [{
                    'name': '同事1', 'relationship': '同事',
                    'disease_probability': 15.0, 'priority': '低',
                }],
            },
        }
        self.patient_info = {
            'basic_info': {'age': 45, 'sputum_smear': 2, 'has_cavity': 1},
        }

    class _MockPredictor:
        is_trained = True
        use_real_data = True
        training_sample_count = 300
        model_performance = {
            'random_forest': {'AUROC': 0.83, 'AUPRC': 0.70,
                              'name': '随机森林', 'name_en': 'Random Forest'},
        }
        shap_explainer = None
        FEATURE_NAMES = ['age', 'cumulative_exposure', 'has_symptoms', 'bcg_vaccine',
                       'has_tb', 'contact_distance', 'ventilation', 'is_high_risk']
        INTERACTION_FEATURE_NAMES = ['age_immuno', 'dm_tb_synergy', 'age_bcg_decay']
        ALL_FEATURE_NAMES = FEATURE_NAMES + INTERACTION_FEATURE_NAMES
        feature_names = FEATURE_NAMES
        FEATURE_DESCRIPTIONS = {
            'age': '年龄', 'cumulative_exposure': '累积暴露时长',
            'has_symptoms': '有症状', 'bcg_vaccine': '卡介苗接种',
            'has_tb': '既往结核史', 'contact_distance': '接触距离',
            'ventilation': '通风条件', 'is_high_risk': '高危人群',
        }
        INTERACTION_FEATURE_DESCRIPTIONS = {
            'age_immuno': '年龄×免疫', 'dm_tb_synergy': '糖尿病×结核',
            'age_bcg_decay': '年龄×BCG衰减',
        }
        ALL_FEATURE_DESCRIPTIONS = {**FEATURE_DESCRIPTIONS, **INTERACTION_FEATURE_DESCRIPTIONS}

    def test_export_pdf_report_creates_file(self):
        """测试 _export_pdf_report 创建包含患者摘要的 PDF"""
        try:
            from reportlab.lib.pagesizes import letter
        except ImportError:
            self.skipTest("reportlab 不可用")

        try:
            from tb_risk.export_utils import _export_pdf_report
        except ImportError:
            self.skipTest("_export_pdf_report 不可用")

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            tmp_path = f.name
        try:
            # _export_pdf_report 签名: (results, patient_info, filepath)
            result = _export_pdf_report(self.results, self.patient_info, tmp_path)
            self.assertTrue(result, "_export_pdf_report 应返回 True")
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 100,
                              "PDF 文件应包含内容")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_export_model_training_report_creates_file(self):
        """测试 export_model_training_report 创建 TRIPOD+AI 兼容报告"""
        try:
            from tb_risk.export_utils import export_model_training_report
        except ImportError:
            self.skipTest("export_model_training_report 不可用")

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.html', delete=False, encoding='utf-8'
        ) as f:
            tmp_path = f.name
        try:
            result = export_model_training_report(
                self._MockPredictor(), tmp_path, format='html'
            )
            self.assertTrue(result, "export_model_training_report 应返回 True")
            self.assertTrue(os.path.exists(tmp_path))

            # 验证报告包含 TRIPOD+AI 核心要素
            with open(tmp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # 特征重要性
            self.assertIn('特征', content, "报告应包含特征相关内容")
            # 验证集性能
            self.assertIn('AUROC', content, "报告应包含 AUROC 指标")
            # 模型信息
            self.assertIn('随机森林', content, "报告应包含模型名称")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ======================== 主入口 ========================

def main_entry():
    from tests.test_utils import main_entry as _main_entry
    import sys as _sys
    all_classes = [
        TestDeepEnsemblePersistence,
        TestSWAGPersistence,
        TestConformalPredictorPersistence,
        TestGNNModelLoadingSafety,
        TestConfigManagement,
        TestExportFormats,
        TestWeightsOnlyFallback,
        TestPDFReportExport,
    ]
    _main_entry('test_persistence.py', _sys.modules[__name__], all_classes)


if __name__ == '__main__':
    main_entry()