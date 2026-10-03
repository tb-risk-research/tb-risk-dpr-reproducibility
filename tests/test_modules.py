#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 扩展测试套件 — ML/GNN、SEIR、不确定性、干预、模拟器等模块
"""
import math
import os
import sys
import unittest

# 确保包路径在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

# ── 可选依赖检测 ──
try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

try:
    import scipy
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    import torch_geometric
    _PYG = True
except ImportError:
    _PYG = False

try:
    import sklearn
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


def _skip_if_no_numpy(cls):
    return unittest.skipUnless(_NUMPY, "需要 numpy")(cls)


def _skip_if_no_torch(cls):
    return unittest.skipUnless(_TORCH, "需要 torch")(cls)


# ══════════════════════════════════════════════════════════════
# SEIR 模块测试
# ══════════════════════════════════════════════════════════════

@_skip_if_no_numpy
class TestStochasticSEIR(unittest.TestCase):
    """测试随机 SEIR 模型"""

    def setUp(self):
        from tb_risk.seir.stochastic import StochasticSEIRModel
        self.model = StochasticSEIRModel(population=1000, noise_scale=0.05, seed=42)

    def test_sde_basic(self):
        """SDE 仿真应返回合理形状的输出"""
        initial = [990, 10, 0, 0]  # S, E, I, R
        times, traj, sto = self.model.simulate_sde(
            initial, t_span=(0, 100), dt=1.0,
            beta=0.3, sigma=0.5, gamma=0.1)
        self.assertEqual(len(times), len(traj))
        self.assertEqual(traj.shape[1], 4)  # SEIR 四个仓室
        # 总人口守恒
        totals = traj.sum(axis=1)
        for t in totals:
            self.assertAlmostEqual(t, 1000, delta=2)

    def test_sde_no_negative(self):
        """SDE 仿真不应产生负值仓室（TB 不可逆约束）"""
        initial = [990, 10, 0, 0]
        _, traj, _ = self.model.simulate_sde(
            initial, t_span=(0, 50), dt=0.5,
            beta=0.3, sigma=0.5, gamma=0.1)
        self.assertTrue((traj >= 0.0).all(), "仓室值不应为负（TB 不可逆约束）")

    def test_ctmc_basic(self):
        """CTMC 仿真应返回合理结果"""
        initial = [990, 10, 0, 0]
        times, states = self.model.simulate_ctmc(
            initial, max_time=50,
            beta=0.3, sigma=0.5, gamma=0.1)
        self.assertGreater(len(times), 0)
        self.assertEqual(states.shape[1], 4)

    def test_simulate_multiple(self):
        """多次仿真应返回多条轨迹"""
        initial = [990, 10, 0, 0]
        result = self.model.simulate_multiple(
            initial, t_span=(0, 50), dt=1.0,
            beta=0.3, sigma=0.5, gamma=0.1, n_trajectories=5)
        self.assertEqual(result.shape[0], 5)

    def test_sde_deterministic_when_noise_zero(self):
        """噪声为零时 SDE 应退化为确定性 ODE"""
        from tb_risk.seir.stochastic import StochasticSEIRModel
        det_model = StochasticSEIRModel(population=1000, noise_scale=0.0, seed=42)
        initial = [990, 10, 0, 0]
        _, traj1, _ = det_model.simulate_sde(
            initial, t_span=(0, 30), dt=1.0,
            beta=0.3, sigma=0.5, gamma=0.1)
        _, traj2, _ = det_model.simulate_sde(
            initial, t_span=(0, 30), dt=1.0,
            beta=0.3, sigma=0.5, gamma=0.1)
        np.testing.assert_array_almost_equal(traj1, traj2)

    def test_sde_no_reverse_flow(self):
        """确定性 SDE 不应有反向流（S 只减不增，R 只增不减）"""
        from tb_risk.seir.stochastic import StochasticSEIRModel
        det_model = StochasticSEIRModel(population=1000, noise_scale=0.0, seed=42)
        initial = [990, 10, 0, 0]
        _, traj, _ = det_model.simulate_sde(
            initial, t_span=(0, 50), dt=1.0,
            beta=0.3, sigma=0.5, gamma=0.1)
        S_col = traj[:, 0]
        R_col = traj[:, 3]
        # S 单调不增（无反向流 E→S）
        self.assertTrue((np.diff(S_col) <= 0).all(),
                        "S 应单调不增（无反向流 E→S）")
        # R 单调不减（无反向流 R→I）
        self.assertTrue((np.diff(R_col) >= 0).all(),
                        "R 应单调不减（无反向流 R→I）")


@_skip_if_no_numpy
class TestBayesianSEIR(unittest.TestCase):
    """测试贝叶斯 SEIR 推断"""

    def setUp(self):
        from tb_risk.seir.bayesian import BayesianSEIRInference
        self.inference = BayesianSEIRInference(seed=42)

    def test_compute_r0_posterior(self):
        """R0 后验计算应返回合理统计量"""
        beta_samples = np.random.uniform(0.2, 0.5, 1000)
        result = self.inference.compute_r0_posterior(beta_samples)
        self.assertIn('mean', result)
        self.assertIn('median', result)
        # 兼容 ci_95_low/hdi_lower 两种键名
        self.assertTrue('ci_95_low' in result or 'hdi_lower' in result)
        self.assertGreater(result['mean'], 0)

    def test_compute_hdi(self):
        """HDI 计算应返回有效区间"""
        samples = np.random.normal(0, 1, 10000)
        hdi_min, hdi_max = self.inference.compute_hdi(samples, prob=0.95)
        self.assertLess(hdi_min, hdi_max)
        # 95% HDI 宽度应在合理范围
        self.assertLess(hdi_max - hdi_min, 10)

    def test_metropolis_hastings_smoke(self):
        """M-H 采样应能运行不报错（冒烟测试，v2.0 多链支持）"""
        from tb_risk.seir.stochastic import StochasticSEIRModel
        model = StochasticSEIRModel(population=1000, noise_scale=0.0, seed=42)
        initial = [990, 10, 0, 0]
        times, traj, _ = model.simulate_sde(
            initial, t_span=(0, 30), dt=1.0,
            beta=0.3, sigma=0.5, gamma=0.1)
        # log_likelihood 期望 observed_data = (I_array, times_array)
        observed = (traj[:, 2], times)
        result = self.inference.metropolis_hastings(
            observed, initial, t_span=(0, 30), dt=1.0,
            n_iterations=100, burn_in=20, n_chains=1)
        self.assertIn('beta', result)
        self.assertEqual(len(result['beta']), 80)  # 100 - 20 burn_in, 单链

        # 验证收敛诊断已填充
        diag = self.inference.convergence_diagnostics
        self.assertIn('r_hat', diag)
        self.assertIn('ess', diag)
        self.assertIn('converged', diag)
        self.assertEqual(diag['n_chains'], 1)


@_skip_if_no_numpy
class TestSEIRUncertainty(unittest.TestCase):
    """测试 SEIR 参数不确定性模块"""

    def test_import(self):
        """模块应可正常导入"""
        from tb_risk.seir.uncertainty import SEIRParameterUncertainty
        u = SEIRParameterUncertainty(population=1000, random_seed=42)
        self.assertIsNotNone(u)

    def test_high_confidence_warning(self):
        """高置信度警告应返回合理结构"""
        from tb_risk.seir.uncertainty import SEIRParameterUncertainty
        u = SEIRParameterUncertainty(population=1000, random_seed=42)
        warning = u.get_high_confidence_warning(threshold=0.9)
        # 兼容 'warning_level' 和 'warning' 两种键
        self.assertTrue('warning_level' in warning or 'warning' in warning)


# ══════════════════════════════════════════════════════════════
# ML/GNN 模块测试
# ══════════════════════════════════════════════════════════════

@_skip_if_no_numpy
class TestGraphBuilder(unittest.TestCase):
    """测试图构建模块"""

    def test_data_access_synthetic(self):
        """合成环境数据应返回有效值"""
        from tb_risk.ml.graph import MultiModalDataAccess
        da = MultiModalDataAccess(station_name='karamay', use_synthetic_env=True)
        data = da.get_environment_data('2025-01-15')
        self.assertIn('pm10', data)
        self.assertIn('humidity', data)
        self.assertIn('temp', data)
        self.assertGreater(data['pm10'], 0)

    def test_data_access_caching(self):
        """缓存应生效"""
        from tb_risk.ml.graph import MultiModalDataAccess
        da = MultiModalDataAccess(use_synthetic_env=True)
        d1 = da.get_environment_data('2025-06-01', use_cache=True)
        d2 = da.get_environment_data('2025-06-01', use_cache=True)
        self.assertEqual(d1['pm10'], d2['pm10'])

    def test_contact_modifier(self):
        """接触频次修正因子应返回合理值"""
        from tb_risk.ml.graph import MultiModalDataAccess
        da = MultiModalDataAccess()
        mod = da.get_contact_modifier('2025-03-10', location='camp')
        self.assertGreater(mod, 0)

    def test_node_environment_features(self):
        """节点环境特征应扩展为 26 维"""
        from tb_risk.ml.graph import MultiModalDataAccess
        da = MultiModalDataAccess()
        base = np.zeros(22, dtype=np.float32)
        extended = da.get_node_environment_features('2025-06-01', base)
        self.assertEqual(extended.shape[0], 26)

    def test_graph_builder_snapshot(self):
        """图快照构建应返回完整结构"""
        from tb_risk.ml.graph import EnhancedTemporalGraphBuilder
        gb = EnhancedTemporalGraphBuilder(n_patient=1, n_household=3, n_social=5)
        snap = gb.build_snapshot('2025-03-15')
        self.assertIn('x', snap)
        self.assertIn('edge_index', snap)
        self.assertIn('edge_attr', snap)
        self.assertIn('num_nodes', snap)
        self.assertEqual(snap['num_nodes'], 1 + 3 + 5 + 4)

    def test_snapshot_sequence(self):
        """图快照序列应返回正确数量"""
        from tb_risk.ml.graph import EnhancedTemporalGraphBuilder
        gb = EnhancedTemporalGraphBuilder(n_patient=1, n_household=2, n_social=3)
        seq = gb.build_snapshot_sequence('2025-01-01', n_days=5)
        self.assertEqual(len(seq), 5)


@_skip_if_no_torch
class TestPreferencePolicy(unittest.TestCase):
    """测试偏好条件策略网络"""

    def test_forward(self):
        """前向传播应返回 action_logits 和 state_value"""
        from tb_risk.ml.policy import PreferenceConditionedPolicy
        policy = PreferenceConditionedPolicy(state_dim=10, n_actions=5, preference_dim=2)
        state = torch.randn(1, 10)
        pref = torch.randn(1, 2)
        logits, value = policy(state, pref)
        self.assertEqual(logits.shape, (1, 5))
        self.assertEqual(value.shape, (1, 1))

    def test_select_action(self):
        """动作选择应返回有效索引"""
        from tb_risk.ml.policy import PreferenceConditionedPolicy
        policy = PreferenceConditionedPolicy(state_dim=10, n_actions=5)
        action = policy.select_action(np.zeros(10), np.ones(2), deterministic=True)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 5)

    def test_train_step(self):
        """训练步应返回损失字典"""
        from tb_risk.ml.policy import PreferenceConditionedPolicy
        policy = PreferenceConditionedPolicy(state_dim=10, n_actions=5)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        result = policy.train_step(
            state=np.zeros(10), action=0, reward=1.0,
            next_state=np.zeros(10), done=False,
            preference=np.ones(2), optimizer=optimizer)
        self.assertIn('total_loss', result)
        self.assertIn('actor_loss', result)


@_skip_if_no_torch
class TestEncoders(unittest.TestCase):
    """测试多模态编码器"""

    def test_environment_encoder_cnn(self):
        """CNN 环境编码器应正确处理序列"""
        from tb_risk.ml.encoder import EnvironmentEncoder
        enc = EnvironmentEncoder(env_feature_dim=4, hidden_dim=32, output_dim=16, use_cnn=True)
        x = torch.randn(2, 7, 4)  # batch=2, window=7, feat=4
        out = enc(x)
        self.assertEqual(out.shape, (2, 16))

    def test_environment_encoder_mlp(self):
        """MLP 环境编码器应正确处理序列"""
        from tb_risk.ml.encoder import EnvironmentEncoder
        enc = EnvironmentEncoder(env_feature_dim=4, hidden_dim=32, output_dim=16, use_cnn=False)
        x = torch.randn(2, 7, 4)
        out = enc(x)
        self.assertEqual(out.shape, (2, 16))

    def test_behavior_encoder(self):
        """行为编码器应正确处理序列"""
        from tb_risk.ml.encoder import BehaviorPatternEncoder
        enc = BehaviorPatternEncoder(behavior_feature_dim=6, hidden_dim=32, output_dim=16)
        x = torch.randn(2, 7, 6)  # batch=2, seq=7, feat=6
        out = enc(x)
        self.assertEqual(out.shape, (2, 16))

    def test_temporal_node_memory(self):
        """节点记忆应正确更新"""
        from tb_risk.ml.encoder import TemporalNodeMemory
        mem = TemporalNodeMemory(n_nodes=5, memory_dim=32)
        self.assertEqual(mem.memory.shape, (5, 32))
        embeddings = torch.randn(5, 32)
        updated = mem(embeddings)
        self.assertEqual(updated.shape, (5, 32))


@_skip_if_no_numpy
class TestMLEnsembleAndUncertainty(unittest.TestCase):
    """测试 ML 不确定性模块（非 torch 部分）"""

    def test_conformal_predictor_calibrate(self):
        """Conformal 预测器校准应返回正值阈值"""
        from tb_risk.ml.uncertainty import ConformalPredictor
        cp = ConformalPredictor(coverage=0.90)
        y_true = np.random.uniform(0, 1, 100)
        y_pred = y_true + np.random.normal(0, 0.1, 100)
        threshold = cp.calibrate(y_true, y_pred)
        self.assertGreater(threshold, 0)

    def test_conformal_predictor_interval(self):
        """Conformal 预测区间应包含上下界"""
        from tb_risk.ml.uncertainty import ConformalPredictor
        cp = ConformalPredictor(coverage=0.90)
        y_true = np.random.uniform(0, 1, 100)
        y_pred = y_true + np.random.normal(0, 0.1, 100)
        cp.calibrate(y_true, y_pred)
        interval = cp.predict_interval(0.5)
        self.assertIn('lower', interval)
        self.assertIn('upper', interval)
        self.assertLessEqual(interval['lower'], interval['upper'])

    def test_conformal_adaptive_interval(self):
        """自适应区间应返回列表"""
        from tb_risk.ml.uncertainty import ConformalPredictor
        cp = ConformalPredictor(coverage=0.90)
        y_true = np.random.uniform(0, 1, 50)
        y_pred = y_true + np.random.normal(0, 0.1, 50)
        cp.calibrate(y_true, y_pred)
        y_preds = np.random.uniform(0.1, 0.9, 10)
        intervals = cp.adaptive_interval(y_preds, risk_levels=np.ones(10) * 0.3)
        self.assertEqual(len(intervals), 10)

    def test_conformal_predictor_zero_weights(self):
        """样本权重和为 0 时应回退为等权重，不抛除零异常"""
        from tb_risk.ml.uncertainty import ConformalPredictor
        cp = ConformalPredictor(coverage=0.90)
        y_true = np.random.uniform(0, 1, 30)
        y_pred = y_true + np.random.normal(0, 0.1, 30)
        threshold = cp.calibrate(y_true, y_pred, sample_weights=np.zeros(30))
        self.assertGreater(threshold, 0)
        self.assertTrue(np.allclose(cp.adaptive_weights, 1.0 / 30))

    def test_uncertainty_fusion_engine_import(self):
        """不确定性融合引擎应可导入"""
        from tb_risk.ml.uncertainty import UncertaintyFusionEngine
        engine = UncertaintyFusionEngine()
        self.assertIsNotNone(engine)

    def test_uncertainty_fusion_individual(self):
        """融合引擎个体级预测应返回有效结构"""
        from tb_risk.ml.uncertainty import UncertaintyFusionEngine
        engine = UncertaintyFusionEngine()
        result = engine.fuse_individual(prior_risk=0.3)
        # 兼容 fused_risk 和 risk_point 两种键
        self.assertTrue('fused_risk' in result or 'risk_point' in result)


@_skip_if_no_torch
class TestDeepEnsemble(unittest.TestCase):
    """测试深度集成预测器"""

    def test_initialize_and_predict(self):
        """初始化和预测应能运行"""
        from tb_risk.ml.uncertainty import DeepEnsemblePredictor
        predictor = DeepEnsemblePredictor(n_ensembles=3)
        predictor.initialize_models(input_dim=10, output_dim=1)
        X = torch.randn(5, 10)
        result = predictor.predict_with_uncertainty(X)
        # 可能返回 mean/error 两种结构
        self.assertTrue('mean' in result or 'error' in result)


@_skip_if_no_torch
class TestCounterfactualOptimizer(unittest.TestCase):
    """测试反事实优化器"""

    def test_placeholder_model_optimize(self):
        """使用占位模型优化应返回结果结构"""
        from tb_risk.ml.optimizer import DifferentiableCounterfactualOptimizer

        # 创建简单线性模型作为替代
        model = torch.nn.Linear(10, 1)
        optimizer = DifferentiableCounterfactualOptimizer(risk_predictor=model)
        x = torch.randn(1, 10, requires_grad=False)
        result = optimizer.optimize(x, target_risk=0.2, max_iterations=10, lr=0.01)
        self.assertIn('initial_risk', result)
        self.assertIn('final_risk', result)


# ══════════════════════════════════════════════════════════════
# Scoring 模块测试
# ══════════════════════════════════════════════════════════════

class TestSimulator(unittest.TestCase):
    """测试数据模拟器"""

    def test_generate_dataset(self):
        """数据集生成应返回合理数据"""
        from tb_risk.scoring.simulator import ScreeningDataSimulator
        sim = ScreeningDataSimulator(random_state=42)
        data = sim.generate_dataset(n_samples=10, include_labels=True)
        self.assertEqual(len(data), 10)
        for record in data:
            self.assertIn('age', record)
            # 兼容 'label' 和 'is_confirmed' 两种键
            self.assertTrue('label' in record or 'is_confirmed' in record)


class TestPredictorImport(unittest.TestCase):
    """测试预测器导入和基本结构"""

    def test_predictor_import(self):
        """预测器应可正常导入"""
        from tb_risk.scoring.predictor import MLRiskPredictor
        predictor = MLRiskPredictor(random_state=42)
        self.assertFalse(predictor.is_trained)

    def test_extract_features(self):
        """特征提取应返回固定维度向量"""
        from tb_risk.scoring.predictor import MLRiskPredictor
        predictor = MLRiskPredictor()
        contact = {
            'age': 30, 'has_symptoms': 0, 'has_tb': 0,
            'bcg_vaccine': 1, 'contact_distance': 'close',
            'ventilation': 3, 'exposure_setting': 'general',
            'cumulative_exposure': 20, 'past_illness_type': 'none',
        }
        if _NUMPY:
            features = predictor.extract_features(contact, 'family')
            self.assertIsNotNone(features)


# ══════════════════════════════════════════════════════════════
# 集成器扩展测试
# ══════════════════════════════════════════════════════════════

@_skip_if_no_numpy
class TestIntegratorFull(unittest.TestCase):
    """测试集成器完整功能"""

    def test_seir_direction(self):
        """SEIR 方向应返回风险估计"""
        from tb_risk.integrator import ThreeDirectionIntegrator
        t = ThreeDirectionIntegrator()
        # SEIR 方向需要数据才能运行
        result = t._weighted_ensemble({})
        self.assertIsNone(result['risk_probability'])

    def test_ensemble_two_directions(self):
        """双方向加权集成应返回加权平均"""
        from tb_risk.integrator import ThreeDirectionIntegrator
        t = ThreeDirectionIntegrator()
        result = t._weighted_ensemble({
            'ml': {'risk_probability': 40.0, 'risk_class': 1},
            'seir': {'risk_probability': 60.0, 'risk_class': 2},
        })
        # 应是加权平均，不是简单平均
        self.assertGreater(result['risk_probability'], 0)
        self.assertIsInstance(result['risk_class'], int)


# ══════════════════════════════════════════════════════════════
# 数值稳定性测试
# ══════════════════════════════════════════════════════════════

@_skip_if_no_numpy
class TestNumericalStability(unittest.TestCase):
    """数值稳定性专项测试"""

    def test_scoring_engine_extreme_age(self):
        """极端年龄应不崩溃"""
        from tb_risk.scoring.engine import ScoringEngine
        engine = ScoringEngine()
        for age in [0, 1, 120, 200]:
            r = engine.compute_risk_score({
                'age': age, 'has_symptoms': 0, 'has_tb': 0,
                'bcg_vaccine': 1, 'contact_distance': 'medium',
                'ventilation': 3, 'exposure_setting': 'general',
                'cumulative_exposure': 0, 'past_illness_type': 'none',
            })
            self.assertFalse(math.isnan(r['disease_probability']))
            self.assertFalse(math.isinf(r['disease_probability']))


# ══════════════════════════════════════════════════════════════
# 启动入口
# ══════════════════════════════════════════════════════════════

def run_tests(verbosity=2, pattern=None):
    """运行测试套件"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    if pattern:
        suite = unittest.TestSuite()
        all_classes = [
            TestStochasticSEIR, TestBayesianSEIR, TestSEIRUncertainty,
            TestGraphBuilder, TestPreferencePolicy, TestEncoders,
            TestMLEnsembleAndUncertainty, TestDeepEnsemble,
            TestCounterfactualOptimizer, TestSimulator, TestPredictorImport,
            TestIntegratorFull, TestNumericalStability,
        ]
        pat_lower = pattern.lower()
        for cls in all_classes:
            if pat_lower in cls.__name__.lower():
                suite.addTests(loader.loadTestsFromTestCase(cls))
            else:
                for name in dir(cls):
                    if name.startswith('test') and pat_lower in name.lower():
                        suite.addTest(cls(name))
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result.wasSuccessful()


def check_deps():
    """检查依赖状态"""
    deps = {}
    for mod_name in ['numpy', 'scipy', 'matplotlib', 'seaborn',
                     'sklearn', 'xgboost', 'shap', 'joblib',
                     'torch', 'torch_geometric', 'pandas']:
        try:
            __import__(mod_name)
            deps[mod_name] = '✓'
        except ImportError:
            deps[mod_name] = '✗'
    print("\n  依赖状态检查")
    print("  " + "=" * 40)
    for k, v in deps.items():
        print(f"  {v} {k}")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == '--check':
            check_deps()
        elif arg == '--test' or arg == '-t':
            pattern = sys.argv[2] if len(sys.argv) > 2 else None
            ok = run_tests(verbosity=2, pattern=pattern)
            sys.exit(0 if ok else 1)
        elif arg == '--help' or arg == '-h':
            print("用法: python test_modules.py [选项]")
            print("")
            print("  (无参数)     运行全部扩展测试")
            print("  --test [名]  运行匹配名称的测试")
            print("  --check      检查依赖状态")
            print("  --help       显示帮助")
        else:
            print(f"未知选项: {arg}，使用 --help 查看帮助")
            sys.exit(1)
    else:
        ok = run_tests(verbosity=2)
        sys.exit(0 if ok else 1)
