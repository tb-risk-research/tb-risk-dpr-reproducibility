#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因果推断框架测试套件

覆盖 DAG、TB 领域 DAG、PSM、反事实分析、do-calculus 五大组件。
遵循项目 unittest + pytest 约定，对可选依赖（numpy/sklearn）做 skip 保护。
"""
import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

try:
    import sklearn
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False


def _skip_if_no_numpy(cls):
    return unittest.skipUnless(_NUMPY, "需要 numpy")(cls)


def _skip_if_no_sklearn(cls):
    return unittest.skipUnless(_SKLEARN, "需要 scikit-learn")(cls)


# ══════════════════════════════════════════════════════════════
# DAG 核心测试
# ══════════════════════════════════════════════════════════════

class TestCausalDAG(unittest.TestCase):
    """DAG 数据结构与图算法测试。"""

    def _build_classic_dag(self):
        """经典混杂结构：Z→X, Z→Y, X→Y（Z 为混杂）"""
        from tb_risk.validation.causal.dag import CausalDAG, NodeType
        dag = CausalDAG()
        dag.add_node('Z', NodeType.CONFOUNDER, '混杂因素')
        dag.add_node('X', NodeType.EXPOSURE, '暴露')
        dag.add_node('Y', NodeType.OUTCOME, '结局')
        dag.add_edge('Z', 'X')
        dag.add_edge('Z', 'Y')
        dag.add_edge('X', 'Y')
        return dag

    def test_add_node_and_edge(self):
        dag = self._build_classic_dag()
        self.assertEqual(len(dag.nodes), 3)
        self.assertEqual(len(dag.edges()), 3)
        self.assertEqual(dag.parents('Y'), {'X', 'Z'})
        self.assertEqual(dag.children('Z'), {'X', 'Y'})

    def test_cycle_detection_rejects_loop(self):
        from tb_risk.validation.causal.dag import CausalDAG
        dag = CausalDAG()
        dag.add_edge('A', 'B')
        dag.add_edge('B', 'C')
        # C→A 会形成环，应抛出 ValueError
        with self.assertRaises(ValueError):
            dag.add_edge('C', 'A')

    def test_self_loop_rejected(self):
        from tb_risk.validation.causal.dag import CausalDAG
        dag = CausalDAG()
        with self.assertRaises(ValueError):
            dag.add_edge('A', 'A')

    def test_acyclic_check(self):
        dag = self._build_classic_dag()
        self.assertTrue(dag.is_acyclic())

    def test_ancestors_descendants(self):
        dag = self._build_classic_dag()
        self.assertEqual(dag.ancestors('Y'), {'X', 'Z'})
        self.assertEqual(dag.descendants('Z'), {'X', 'Y'})
        self.assertEqual(dag.ancestors('Z'), set())

    def test_d_separation_chain(self):
        """链 Z→X→Y：给定 X 时 Z 与 Y d-分离"""
        from tb_risk.validation.causal.dag import CausalDAG
        dag = CausalDAG()
        dag.add_edge('Z', 'X')
        dag.add_edge('X', 'Y')
        # 不条件化：Z 与 Y d-连通
        self.assertFalse(dag.is_d_separated('Z', 'Y', set()))
        # 条件化 X：d-分离
        self.assertTrue(dag.is_d_separated('Z', 'Y', {'X'}))

    def test_d_separation_fork(self):
        """分叉 Z←X→Y：给定 X 时 Z 与 Y d-分离"""
        from tb_risk.validation.causal.dag import CausalDAG
        dag = CausalDAG()
        dag.add_edge('X', 'Z')
        dag.add_edge('X', 'Y')
        self.assertFalse(dag.is_d_separated('Z', 'Y', set()))
        self.assertTrue(dag.is_d_separated('Z', 'Y', {'X'}))

    def test_d_separation_collider(self):
        """碰撞 Z→X←Y：默认 d-分离，条件化 X 后 d-连通"""
        from tb_risk.validation.causal.dag import CausalDAG
        dag = CausalDAG()
        dag.add_edge('Z', 'X')
        dag.add_edge('Y', 'X')
        self.assertTrue(dag.is_d_separated('Z', 'Y', set()))
        self.assertFalse(dag.is_d_separated('Z', 'Y', {'X'}))

    def test_backdoor_paths(self):
        dag = self._build_classic_dag()
        # X←Z→Y 是后门路径
        paths = dag.find_backdoor_paths('X', 'Y')
        self.assertTrue(any(p == ['X', 'Z', 'Y'] for p in paths),
                        f"应包含后门路径 X←Z→Y，实际: {paths}")

    def test_valid_adjustment_set(self):
        dag = self._build_classic_dag()
        # Z 满足后门准则
        self.assertTrue(dag.is_valid_adjustment_set('X', 'Y', {'Z'}))
        # 空集不满足（后门路径未阻断）
        self.assertFalse(dag.is_valid_adjustment_set('X', 'Y', set()))
        # X 的后代（Y 本身）不能进调整集
        self.assertFalse(dag.is_valid_adjustment_set('X', 'Y', {'Y'}))

    def test_find_adjustment_set(self):
        dag = self._build_classic_dag()
        adj = dag.find_adjustment_set('X', 'Y')
        self.assertIsNotNone(adj)
        self.assertIn('Z', adj)

    def test_adjustment_excludes_mediator(self):
        """调整集不得包含中介变量（避免过度调整）"""
        from tb_risk.validation.causal.dag import CausalDAG, NodeType
        dag = CausalDAG()
        dag.add_node('X', NodeType.EXPOSURE)
        dag.add_node('M', NodeType.MEDIATOR)
        dag.add_node('Y', NodeType.OUTCOME)
        dag.add_node('Z', NodeType.CONFOUNDER)
        dag.add_edge('Z', 'X')
        dag.add_edge('Z', 'Y')
        dag.add_edge('X', 'M')
        dag.add_edge('M', 'Y')
        adj = dag.find_adjustment_set('X', 'Y')
        self.assertIsNotNone(adj)
        self.assertNotIn('M', adj)  # 中介不得入调整集
        self.assertIn('Z', adj)

    def test_serialization_roundtrip(self):
        dag = self._build_classic_dag()
        d = dag.to_dict()
        from tb_risk.validation.causal.dag import CausalDAG
        dag2 = CausalDAG.from_dict(d)
        self.assertEqual(sorted(dag2.nodes), sorted(dag.nodes))
        self.assertEqual(sorted(dag2.edges()), sorted(dag.edges()))

    def test_nodes_by_type(self):
        from tb_risk.validation.causal.dag import NodeType
        dag = self._build_classic_dag()
        self.assertEqual(dag.nodes_by_type(NodeType.CONFOUNDER), ['Z'])
        self.assertEqual(dag.nodes_by_type(NodeType.EXPOSURE), ['X'])


# ══════════════════════════════════════════════════════════════
# TB 领域 DAG 测试
# ══════════════════════════════════════════════════════════════

class TestTBCausalDAG(unittest.TestCase):
    """TB 领域默认因果 DAG 测试。"""

    def test_build_tb_dag_acyclic(self):
        from tb_risk.validation.causal.tb_dag import build_tb_dag
        dag = build_tb_dag()
        self.assertTrue(dag.is_acyclic(),
                        "TB DAG 必须无环\n" + dag.summary())

    def test_tb_dag_has_exposure_and_outcome(self):
        from tb_risk.validation.causal.tb_dag import (
            build_tb_dag, DEFAULT_EXPOSURE, DEFAULT_OUTCOME)
        dag = build_tb_dag()
        self.assertIn(DEFAULT_EXPOSURE, dag.nodes)
        self.assertIn(DEFAULT_OUTCOME, dag.nodes)

    def test_tb_dag_node_roles(self):
        from tb_risk.validation.causal.tb_dag import build_tb_dag
        from tb_risk.validation.causal.dag import NodeType
        dag = build_tb_dag()
        self.assertEqual(dag.node_type('cumulative_exposure'),
                         NodeType.EXPOSURE)
        self.assertEqual(dag.node_type('disease_probability'),
                         NodeType.OUTCOME)
        self.assertEqual(dag.node_type('has_symptoms'), NodeType.MEDIATOR)
        # 年龄应为混杂
        self.assertEqual(dag.node_type('age'), NodeType.CONFOUNDER)

    def test_tb_dag_default_adjustment_set(self):
        """默认暴露-结局对应能求出调整集"""
        from tb_risk.validation.causal.tb_dag import (
            get_default_adjustment_set, DEFAULT_EXPOSURE, DEFAULT_OUTCOME)
        adj, dag = get_default_adjustment_set(DEFAULT_EXPOSURE, DEFAULT_OUTCOME)
        self.assertIsNotNone(adj,
                             "TB DAG 应能为默认暴露-结局对求解调整集\n"
                             + dag.summary())
        # 调整集应包含产生后门路径的混杂因素：
        # contact_distance_score 与 ventilation_score 同时影响
        # cumulative_exposure（暴露）与 disease_probability（结局）
        self.assertIn('contact_distance_score', adj)
        self.assertIn('ventilation_score', adj)
        # 不得包含中介 has_symptoms（避免过度调整）
        self.assertNotIn('has_symptoms', adj)
        # 不得包含暴露本身或其后代
        self.assertNotIn('cumulative_exposure', adj)
        self.assertNotIn('disease_probability', adj)
        # age 仅影响结局不影响暴露，故非该对的混杂因素，正确地被排除
        # （这验证了框架能区分"结局预测因子"与"混杂因素"）

    def test_tb_dag_adjustment_set_valid(self):
        """求出的调整集应满足后门准则"""
        from tb_risk.validation.causal.tb_dag import (
            get_default_adjustment_set, DEFAULT_EXPOSURE, DEFAULT_OUTCOME)
        adj, dag = get_default_adjustment_set()
        self.assertTrue(
            dag.is_valid_adjustment_set(
                DEFAULT_EXPOSURE, DEFAULT_OUTCOME, adj))


# ══════════════════════════════════════════════════════════════
# PSM 测试
# ══════════════════════════════════════════════════════════════

@_skip_if_no_numpy
@_skip_if_no_sklearn
class TestPropensityScoreMatcher(unittest.TestCase):
    """倾向评分匹配测试。"""

    def _generate_confounded_data(self, n=400, seed=42):
        """生成含混杂的合成数据：Z 同时影响 T 与 Y。"""
        rng = np.random.RandomState(seed)
        # 混杂 Z
        z = rng.normal(0, 1, n)
        # 倾向评分 e(z) = sigmoid(z)：Z 越大越可能接受干预
        propensity = 1 / (1 + np.exp(-z))
        t = (rng.random(n) < propensity).astype(int)
        # 结局 Y = 2*T + 1.5*Z + noise（真实 ATE=2）
        y = 2.0 * t + 1.5 * z + rng.normal(0, 0.5, n)
        data = []
        for i in range(n):
            data.append({'z': float(z[i]), 'treatment': int(t[i]),
                         'outcome': float(y[i])})
        return data

    def test_propensity_estimation(self):
        from tb_risk.validation.causal.psm import PropensityScoreMatcher
        data = self._generate_confounded_data()
        matcher = PropensityScoreMatcher('treatment', ['z'])
        prop = matcher.estimate_propensity(data)
        self.assertEqual(len(prop), len(data))
        self.assertTrue(np.all((prop >= 0) & (prop <= 1)))
        # Z 与倾向评分应正相关
        corr = np.corrcoef([d['z'] for d in data], prop)[0, 1]
        self.assertGreater(corr, 0.5)

    def test_matching_returns_pairs(self):
        from tb_risk.validation.causal.psm import PropensityScoreMatcher
        data = self._generate_confounded_data()
        matcher = PropensityScoreMatcher('treatment', ['z'], caliper=0.3)
        matches = matcher.match(data)
        self.assertGreater(len(matches), 0)
        for t_idx, c_idx in matches:
            self.assertEqual(data[t_idx]['treatment'], 1)
            self.assertEqual(data[c_idx]['treatment'], 0)

    def test_att_recovers_true_effect(self):
        """ATT 应接近真实处理效应 2.0（容差内）"""
        from tb_risk.validation.causal.psm import PropensityScoreMatcher
        data = self._generate_confounded_data(n=800, seed=42)
        matcher = PropensityScoreMatcher('treatment', ['z'], caliper=0.25)
        result = matcher.estimate_att(data, 'outcome')
        self.assertIn('att', result)
        self.assertFalse(np.isnan(result['att']))
        # 真实 ATE=2，ATT 应在 [1.3, 2.7] 区间内（允许采样波动）
        self.assertAlmostEqual(result['att'], 2.0, delta=0.7,
                               msg=f"ATT={result['att']} 偏离真实效应 2.0")
        self.assertGreater(result['n_matched'], 50)

    def test_balance_diagnostics_improves(self):
        """匹配后 SMD 应小于匹配前"""
        from tb_risk.validation.causal.psm import PropensityScoreMatcher
        data = self._generate_confounded_data(n=600)
        matcher = PropensityScoreMatcher('treatment', ['z'], caliper=0.2)
        result = matcher.estimate_att(data, 'outcome')
        smd_before = abs(result['smd_before'].get('z', 0))
        smd_after = abs(result['smd_after'].get('z', 0))
        self.assertLess(smd_after, smd_before,
                        "匹配后协变量平衡应改善（SMD 下降）")

    def test_pandas_dataframe_input(self):
        if not _PANDAS:
            self.skipTest("需要 pandas")
        from tb_risk.validation.causal.psm import PropensityScoreMatcher
        data = self._generate_confounded_data(n=300)
        df = pd.DataFrame(data)
        matcher = PropensityScoreMatcher('treatment', ['z'])
        prop = matcher.estimate_propensity(df)
        self.assertEqual(len(prop), len(df))

    def test_empty_treatment_group(self):
        from tb_risk.validation.causal.psm import PropensityScoreMatcher
        data = [{'z': 0.0, 'treatment': 0, 'outcome': 1.0}] * 10
        matcher = PropensityScoreMatcher('treatment', ['z'])
        result = matcher.estimate_att(data, 'outcome')
        self.assertTrue(np.isnan(result['att']))


# ══════════════════════════════════════════════════════════════
# 反事实分析测试
# ══════════════════════════════════════════════════════════════

@_skip_if_no_numpy
@_skip_if_no_sklearn
class TestCounterfactualAnalyzer(unittest.TestCase):
    """反事实分析测试（需训练 ML 预测器）。"""

    @classmethod
    def setUpClass(cls):
        """训练一个小型 ML 预测器供反事实分析使用。"""
        from tb_risk.scoring.predictor import MLRiskPredictor
        cls.predictor = MLRiskPredictor(random_state=42)
        try:
            cls.predictor.train_models()
        except Exception:
            # 训练失败则跳过反事实相关测试
            cls.predictor = None

    def _make_contact(self, cumulative_exposure=40, age=35):
        return {
            'age': age, 'has_symptoms': 0, 'has_tb': 0,
            'bcg_vaccine': 1, 'contact_distance': 'close',
            'ventilation': 3, 'exposure_setting': 'general',
            'cumulative_exposure': cumulative_exposure,
            'past_illness_type': 'none',
        }

    def test_individual_counterfactual(self):
        if self.predictor is None or not self.predictor.is_trained:
            self.skipTest("预测器训练失败")
        from tb_risk.validation.causal.counterfactual import (
            CounterfactualAnalyzer)
        analyzer = CounterfactualAnalyzer(self.predictor)
        cd = self._make_contact(cumulative_exposure=40)
        result = analyzer.individual_counterfactual(
            cd, feature_overrides={'cumulative_exposure': 20})
        self.assertIn('original_risk', result)
        self.assertIn('counterfactual_risk', result)
        self.assertIn('delta', result)
        self.assertFalse(np.isnan(result['original_risk']))
        self.assertFalse(np.isnan(result['counterfactual_risk']))

    def test_exposure_reduction_lowers_risk(self):
        """减少累积暴露应降低或持平风险（剂量-反应单调性）"""
        if self.predictor is None or not self.predictor.is_trained:
            self.skipTest("预测器训练失败")
        from tb_risk.validation.causal.counterfactual import (
            CounterfactualAnalyzer)
        analyzer = CounterfactualAnalyzer(self.predictor)
        cd = self._make_contact(cumulative_exposure=80)
        result = analyzer.individual_counterfactual(
            cd, feature_overrides={'cumulative_exposure': 10})
        self.assertLessEqual(result['counterfactual_risk'],
                             result['original_risk'] + 1e-6,
                             "减少暴露应降低或持平风险")

    def test_population_counterfactual(self):
        if self.predictor is None or not self.predictor.is_trained:
            self.skipTest("预测器训练失败")
        from tb_risk.validation.causal.counterfactual import (
            CounterfactualAnalyzer)
        analyzer = CounterfactualAnalyzer(self.predictor)
        contacts = [self._make_contact(cumulative_exposure=ex)
                    for ex in [10, 30, 50, 70, 90]]
        result = analyzer.population_counterfactual(
            contacts, feature_overrides={'cumulative_exposure': 20})
        self.assertIn('mean_delta', result)
        self.assertEqual(result['n_samples'], len(contacts))

    def test_scenario_analysis(self):
        if self.predictor is None or not self.predictor.is_trained:
            self.skipTest("预测器训练失败")
        from tb_risk.validation.causal.counterfactual import (
            CounterfactualAnalyzer)
        analyzer = CounterfactualAnalyzer(self.predictor)
        contacts = [self._make_contact(cumulative_exposure=ex)
                    for ex in [40, 60, 80]]
        result = analyzer.scenario_analysis(contacts, 'exposure_halved')
        self.assertEqual(result['scenario'], 'exposure_halved')
        self.assertIn('mean_delta', result)
        self.assertGreater(result['n_samples'], 0)

    def test_compare_scenarios(self):
        if self.predictor is None or not self.predictor.is_trained:
            self.skipTest("预测器训练失败")
        from tb_risk.validation.causal.counterfactual import (
            CounterfactualAnalyzer)
        analyzer = CounterfactualAnalyzer(self.predictor)
        contacts = [self._make_contact(cumulative_exposure=ex)
                    for ex in [40, 60, 80]]
        results = analyzer.compare_scenarios(
            contacts, scenarios=['exposure_halved', 'duration_halved'])
        self.assertIn('exposure_halved', results)
        self.assertIn('duration_halved', results)

    def test_unknown_scenario_raises(self):
        from tb_risk.validation.causal.counterfactual import (
            CounterfactualAnalyzer)
        if self.predictor is None:
            self.skipTest("预测器不可用")
        analyzer = CounterfactualAnalyzer(self.predictor)
        with self.assertRaises(ValueError):
            analyzer.scenario_analysis([], 'nonexistent_scenario')

    def test_untrained_predictor_returns_nan(self):
        from tb_risk.scoring.predictor import MLRiskPredictor
        from tb_risk.validation.causal.counterfactual import (
            CounterfactualAnalyzer)
        untrained = MLRiskPredictor()
        analyzer = CounterfactualAnalyzer(untrained)
        result = analyzer.individual_counterfactual(
            self._make_contact(), {'cumulative_exposure': 20})
        self.assertTrue(np.isnan(result['delta']))

    def test_population_counterfactual_resilient_to_bad_sample(self):
        """群体反事实分析中单样本异常不得中断整体计算。"""
        from tb_risk.validation.causal.counterfactual import (
            CounterfactualAnalyzer)

        class _RaisingPredictor:
            """前两次预测正常，第三次抛异常的 mock 预测器。"""
            is_trained = True
            def __init__(self, real_predictor):
                self._real = real_predictor
                self._call_count = 0
            def predict_risk(self, data, contact_type='family',
                             patient_context=None):
                self._call_count += 1
                if self._call_count == 3:  # 第二个样本的原始预测
                    raise RuntimeError("模拟预测失败")
                return self._real.predict_risk(
                    data, contact_type=contact_type,
                    patient_context=patient_context)

        if self.predictor is None or not self.predictor.is_trained:
            self.skipTest("预测器训练失败")
        analyzer = CounterfactualAnalyzer(_RaisingPredictor(self.predictor))
        contacts = [self._make_contact(cumulative_exposure=ex)
                    for ex in [40, 60, 80]]
        # 不应抛异常；坏样本计入 errors，好样本仍返回结果
        result = analyzer.population_counterfactual(
            contacts, feature_overrides={'cumulative_exposure': 20})
        self.assertIn('n_samples', result)
        self.assertEqual(result['n_errors'], 1)
        self.assertGreater(result['n_samples'], 0)

    def test_scenario_analysis_resilient_to_bad_sample(self):
        """场景分析中单样本预测异常不得中断整体计算。"""
        from tb_risk.validation.causal.counterfactual import (
            CounterfactualAnalyzer)

        class _RaisingPredictor:
            is_trained = True
            def __init__(self, real_predictor):
                self._real = real_predictor
                self._call_count = 0
            def predict_risk(self, data, contact_type='family',
                             patient_context=None):
                self._call_count += 1
                if self._call_count == 2:
                    raise RuntimeError("模拟预测失败")
                return self._real.predict_risk(
                    data, contact_type=contact_type,
                    patient_context=patient_context)

        if self.predictor is None or not self.predictor.is_trained:
            self.skipTest("预测器训练失败")
        analyzer = CounterfactualAnalyzer(_RaisingPredictor(self.predictor))
        contacts = [self._make_contact(cumulative_exposure=ex)
                    for ex in [40, 60, 80]]
        result = analyzer.scenario_analysis(contacts, 'exposure_halved')
        self.assertIn('n_skipped', result)
        self.assertGreaterEqual(result['n_skipped'], 1)
        self.assertGreater(result['n_samples'], 0)


# ══════════════════════════════════════════════════════════════
# do-calculus 测试
# ══════════════════════════════════════════════════════════════

@_skip_if_no_numpy
@_skip_if_no_sklearn
class TestDoCalculusEstimator(unittest.TestCase):
    """do-calculus 后门调整测试。"""

    def _build_simple_dag(self):
        """Z→X, Z→Y, X→Y（Z 为混杂）"""
        from tb_risk.validation.causal.dag import CausalDAG, NodeType
        dag = CausalDAG()
        dag.add_node('Z', NodeType.CONFOUNDER)
        dag.add_node('X', NodeType.EXPOSURE)
        dag.add_node('Y', NodeType.OUTCOME)
        dag.add_edge('Z', 'X')
        dag.add_edge('Z', 'Y')
        dag.add_edge('X', 'Y')
        return dag

    def _generate_confounded_data(self, n=500, seed=42):
        """Y = 3*X - 2*Z + noise（真实 ATE=3）。Z 同时影响 X 与 Y。"""
        rng = np.random.RandomState(seed)
        z = rng.normal(0, 1, n)
        # X 受 Z 影响
        x_prob = 1 / (1 + np.exp(-(0.8 * z)))
        x = (rng.random(n) < x_prob).astype(float)
        # Y = 3*X - 2*Z + noise
        y = 3.0 * x - 2.0 * z + rng.normal(0, 0.3, n)
        data = []
        for i in range(n):
            data.append({'X': float(x[i]), 'Z': float(z[i]),
                         'Y': float(y[i])})
        return data

    def test_identify_adjustment_set(self):
        from tb_risk.validation.causal.do_calculus import DoCalculusEstimator
        dag = self._build_simple_dag()
        est = DoCalculusEstimator(dag)
        adj = est.identify_adjustment_set('X', 'Y')
        self.assertIn('Z', adj)

    def test_ate_recovers_true_effect(self):
        """后门调整后 ATE 应接近真实效应 3.0"""
        from tb_risk.validation.causal.do_calculus import DoCalculusEstimator
        dag = self._build_simple_dag()
        est = DoCalculusEstimator(dag, model_type='linear')
        data = self._generate_confounded_data(n=800, seed=42)
        result = est.estimate_ate(data, 'X', 'Y', treatment_value=1,
                                  control_value=0, n_bootstrap=50)
        self.assertIn('ate', result)
        self.assertFalse(np.isnan(result['ate']))
        # 真实 ATE=3，调整后应在 [2.4, 3.6] 内
        self.assertAlmostEqual(result['ate'], 3.0, delta=0.6,
                               msg=f"ATE={result['ate']} 偏离真实效应 3.0")
        self.assertEqual(set(result['adjustment_set']), {'Z'})

    def test_naive_estimate_is_biased(self):
        """未调整的朴素估计应有偏（验证混杂存在）"""
        data = self._generate_confounded_data(n=800, seed=42)
        # 朴素估计：E[Y|X=1] - E[Y|X=0]
        y1 = np.mean([d['Y'] for d in data if d['X'] == 1])
        y0 = np.mean([d['Y'] for d in data if d['X'] == 0])
        naive = y1 - y0
        # 真实 ATE=3，朴素估计偏离（受 Z 混杂）
        self.assertNotAlmostEqual(naive, 3.0, delta=0.3,
                                  msg="朴素估计应有偏")

    def test_ci_reasonable(self):
        from tb_risk.validation.causal.do_calculus import DoCalculusEstimator
        dag = self._build_simple_dag()
        est = DoCalculusEstimator(dag, model_type='linear')
        data = self._generate_confounded_data(n=500, seed=42)
        result = est.estimate_ate(data, 'X', 'Y', n_bootstrap=50)
        ci = result['ci']
        self.assertEqual(len(ci), 2)
        self.assertLessEqual(ci[0], result['ate'])
        self.assertGreaterEqual(ci[1], result['ate'])

    def test_dose_response(self):
        from tb_risk.validation.causal.do_calculus import DoCalculusEstimator
        dag = self._build_simple_dag()
        est = DoCalculusEstimator(dag, model_type='linear')
        # 连续干预数据：Y = 2*X + Z + noise
        rng = np.random.RandomState(42)
        n = 400
        z = rng.normal(0, 1, n)
        x = rng.uniform(0, 10, n) + z  # X 受 Z 影响
        y = 2.0 * x + z + rng.normal(0, 0.3, n)
        data = [{'X': float(x[i]), 'Z': float(z[i]), 'Y': float(y[i])}
                for i in range(n)]
        result = est.intervention_effect(
            data, 'X', 'Y', intervention_values=[0, 2, 4, 6, 8])
        dr = result['dose_response']
        self.assertEqual(len(dr), 5)
        # 线性关系：Y 随 X 单调上升
        ys = [v for _, v in dr]
        self.assertTrue(all(ys[i] <= ys[i + 1] + 1e-6
                            for i in range(len(ys) - 1)),
                        f"剂量-反应应单调上升: {ys}")

    def test_manual_adjustment_set(self):
        from tb_risk.validation.causal.do_calculus import DoCalculusEstimator
        dag = self._build_simple_dag()
        est = DoCalculusEstimator(dag)
        data = self._generate_confounded_data(n=400, seed=42)
        result = est.estimate_ate(data, 'X', 'Y',
                                  adjustment_set=['Z'],
                                  n_bootstrap=20)
        self.assertEqual(set(result['adjustment_set']), {'Z'})

    def test_conditional_effect(self):
        from tb_risk.validation.causal.do_calculus import DoCalculusEstimator
        dag = self._build_simple_dag()
        # 添加效应修饰因子 G
        from tb_risk.validation.causal.dag import NodeType
        dag.add_node('G', NodeType.EFFECT_MODIFIER)
        dag.add_edge('G', 'Y')
        est = DoCalculusEstimator(dag, model_type='linear')
        rng = np.random.RandomState(42)
        n = 400
        z = rng.normal(0, 1, n)
        g = rng.choice(['A', 'B'], n)
        x = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
        # A 组效应 3，B 组效应 1
        effect = np.where(g == 'A', 3.0, 1.0)
        y = effect * x + z + rng.normal(0, 0.3, n)
        data = [{'X': float(x[i]), 'Z': float(z[i]),
                 'G': str(g[i]), 'Y': float(y[i])} for i in range(n)]
        result = est.conditional_effect(
            data, 'X', 'Y', subgroup_col='G',
            adjustment_set=['Z'])
        effects = result['subgroup_effects']
        self.assertIn('A', effects)
        self.assertIn('B', effects)
        # A 组效应应高于 B 组
        self.assertGreater(effects['A']['ate'], effects['B']['ate'])


# ══════════════════════════════════════════════════════════════
# 包导出测试
# ══════════════════════════════════════════════════════════════

class TestPackageExports(unittest.TestCase):
    """验证 validation 包正确导出因果推断组件。"""

    def test_import_from_validation(self):
        from tb_risk.validation import (
            CausalDAG, NodeType, build_tb_dag,
            PropensityScoreMatcher, CounterfactualAnalyzer,
            DoCalculusEstimator)
        self.assertIsNotNone(CausalDAG)
        self.assertIsNotNone(NodeType)
        self.assertIsNotNone(build_tb_dag)
        self.assertIsNotNone(PropensityScoreMatcher)
        self.assertIsNotNone(CounterfactualAnalyzer)
        self.assertIsNotNone(DoCalculusEstimator)

    def test_import_from_causal_subpackage(self):
        from tb_risk.validation.causal import (
            CausalDAG, NodeType, build_tb_dag, get_default_adjustment_set,
            PropensityScoreMatcher, CounterfactualAnalyzer,
            DoCalculusEstimator)
        self.assertIsNotNone(CausalDAG)


if __name__ == '__main__':
    unittest.main(verbosity=2)
