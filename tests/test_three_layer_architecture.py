#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三层递进架构 + 任务分解 + 网络消融 测试套件（改进一 + 改进二）

覆盖：
1. 第 1 层：个体基础层 P_base（compute_individual_base）
2. 第 2 层：网络增强层残差增量（compute_network_increment，NumPy 兜底）
3. 第 3 层：社区干预反事实模拟（simulate_community_intervention）
4. 决策层：堆叠+门控（gated_integration，无网络门控=0）
5. 高风险集合选择（select_high_risk_set）
6. 端到端流水线（ThreeLayerArchitecture.run_pipeline）
7. 集成器接入（ThreeDirectionIntegrator.integrate_three_layer）
8. 网络消融（compute_c_index / network_contribution_ablation，ΔAUROC/ΔC-index）
9. 任务分解（TaskType / assign_task_labels / evaluate_task_a/b/c / evaluate_task）

遵循项目 unittest + pytest 约定，对可选依赖做 skip 保护。
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

from tb_risk.scoring.architecture import (  # noqa: E402
    DEFAULT_HIGH_RISK_THRESHOLD,
    DEFAULT_GATE_MIN,
    DEFAULT_GATE_MAX,
    DEFAULT_DECISION_WEIGHTS,
    compute_individual_base,
    compute_network_increment,
    select_high_risk_set,
    simulate_community_intervention,
    gated_integration,
    ThreeLayerArchitecture,
    run_three_layer_pipeline,
)
from tb_risk.validation.layer_ablation import (  # noqa: E402
    compute_c_index,
    build_synthetic_network,
    LayerAblationAnalyzer,
    network_contribution_ablation,
)
from tb_risk.validation.task_decomposition import (  # noqa: E402
    TaskType,
    EndPoint,
    TASK_ENDPOINT_MAP,
    evaluate_task_a,
    evaluate_task_b,
    evaluate_task_c,
    assign_task_labels,
    evaluate_task,
)


def _skip_if_no_numpy(cls):
    return unittest.skipUnless(_NUMPY, "需要 numpy")(cls)


def _make_contact(name="接触者A", cumulative_exposure=40, age=35,
                  ventilation=3, has_symptoms=0, record_id=None):
    """构造一个最小合法的接触者数据字典。"""
    return {
        'name': name,
        'record_id': record_id,
        'age': age, 'has_symptoms': has_symptoms, 'has_tb': 0,
        'bcg_vaccine': 1, 'contact_distance': 'close',
        'ventilation': ventilation, 'exposure_setting': 'general',
        'cumulative_exposure': cumulative_exposure,
        'single_duration': 60, 'freq_density': 10, 'time_span': 8,
    }


def _make_assessment(family_entries, social_entries=None, patient_info=None):
    """构造最小合法的评估对象（供 inference/integrator/architecture 消费）。"""
    class _Assessment:
        pass
    a = _Assessment()
    a.family_entries = family_entries or []
    a.social_entries = social_entries or []
    a.patient_info = patient_info or {'basic_info': {'sputum_smear': 1, 'has_cavity': 1}}
    a.results = {
        'base_infection_probability': 30,
        'potential_patients': {
            'family': a.family_entries,
            'social': a.social_entries,
        },
    }
    return a


class _FakeMLPredictor:
    """最小假 ML 预测器（第 1 层 P_base 来源）。"""
    def __init__(self, prob=30.0):
        self.prob = prob
        self.is_trained = True

    def predict_risk(self, contact_data, contact_type='family'):
        return {'ensemble': {'risk_probability': self.prob}}

    def predict_gnn_risk(self, tb_assessment, contact_data, contact_type='family'):
        return None


# ---------------------------------------------------------------------------
# 1. 第 1 层：个体基础层 P_base
# ---------------------------------------------------------------------------

class TestLayer1IndividualBase(unittest.TestCase):

    def test_compute_individual_base_returns_p_base(self):
        base = compute_individual_base(_FakeMLPredictor(40.0), _make_contact())
        self.assertAlmostEqual(base['p_base'], 0.4, places=2)
        self.assertAlmostEqual(base['p_base_percent'], 40.0, places=2)
        self.assertEqual(base['layer'], 1)
        self.assertTrue(base['is_trained'])

    def test_compute_individual_base_none_predictor(self):
        base = compute_individual_base(None, _make_contact())
        self.assertAlmostEqual(base['p_base_percent'], 30.0, places=2)
        self.assertFalse(base['is_trained'])


# ---------------------------------------------------------------------------
# 2. 第 2 层：网络增强层残差增量
# ---------------------------------------------------------------------------

class TestLayer2NetworkIncrement(unittest.TestCase):

    def test_network_increment_semantics(self):
        contact = _make_contact(record_id='c1')
        family = [_make_contact('家人1', cumulative_exposure=80),
                  _make_contact('家人2', cumulative_exposure=120)]
        assessment = _make_assessment(family)
        net = compute_network_increment(
            _FakeMLPredictor(30.0), assessment, contact, 'family', p_base=30.0)
        self.assertTrue(net['residual_learning'])
        self.assertIn('network_increment', net)
        self.assertIn('baseline_probability', net)
        self.assertEqual(net['baseline_probability'], 30.0)
        # 网络增量 = 网络风险 - 个体基线
        self.assertAlmostEqual(
            net['network_increment'],
            net['network_risk'] - 30.0, places=4)

    def test_network_increment_without_contacts_is_zero(self):
        contact = _make_contact(record_id='c1')
        assessment = _make_assessment([])
        net = compute_network_increment(
            _FakeMLPredictor(30.0), assessment, contact, 'family', p_base=30.0)
        self.assertEqual(net['n_contacts'], 0)

    def test_network_increment_p_base_auto(self):
        contact = _make_contact(record_id='c1')
        assessment = _make_assessment([])
        net = compute_network_increment(
            _FakeMLPredictor(40.0), assessment, contact, 'family')
        self.assertAlmostEqual(net['baseline_probability'], 40.0, places=2)


# ---------------------------------------------------------------------------
# 3. 第 3 层：社区干预反事实模拟（小规模快速）
# ---------------------------------------------------------------------------

class TestLayer3Intervention(unittest.TestCase):

    def test_simulate_intervention_returns_report(self):
        rep = simulate_community_intervention(
            high_risk_contacts=[{'record_id': 'c1'}],
            population=1000, t_horizon_days=182, dt=14, random_state=42)
        self.assertEqual(rep.get('level'), 3)
        self.assertIn('baseline', rep)
        self.assertIn('strategies', rep)
        self.assertIn('summary', rep)
        self.assertGreaterEqual(len(rep['strategies']), 1)

    def test_intervention_report_has_metrics(self):
        rep = simulate_community_intervention(
            high_risk_contacts=[], population=500,
            t_horizon_days=182, dt=14, random_state=1)
        self.assertIn('best_avoidable_case_days', rep.get('summary', {}))
        self.assertIn('best_averted_percent', rep.get('summary', {}))


# ---------------------------------------------------------------------------
# 4. 决策层：堆叠 + 门控
# ---------------------------------------------------------------------------

class TestDecisionGating(unittest.TestCase):

    def test_gating_zero_without_network(self):
        d = gated_integration(p_base_percent=30.0, network_increment=20.0,
                              n_contacts=0, network_aware=False)
        self.assertEqual(d['gating'], DEFAULT_GATE_MIN)
        self.assertAlmostEqual(d['gated_network_increment'], 0.0, places=4)
        self.assertAlmostEqual(d['combined_probability'], 30.0, places=4)

    def test_gating_full_with_network(self):
        d = gated_integration(p_base_percent=30.0, network_increment=20.0,
                              n_contacts=50, network_aware=True)
        self.assertAlmostEqual(d['gating'], DEFAULT_GATE_MAX, places=4)
        self.assertAlmostEqual(d['gated_network_increment'], 20.0, places=4)
        self.assertAlmostEqual(d['combined_probability'], 50.0, places=4)

    def test_combined_clipped(self):
        d = gated_integration(p_base_percent=95.0, network_increment=30.0,
                              n_contacts=10, network_aware=True)
        self.assertLessEqual(d['combined_probability'], 100.0)

    def test_integration_method(self):
        d = gated_integration(p_base_percent=30.0, network_increment=0.0,
                              n_contacts=0, network_aware=False)
        self.assertEqual(d['integration_method'], 'stacking_gating')


# ---------------------------------------------------------------------------
# 5. 高风险集合选择
# ---------------------------------------------------------------------------

class TestHighRiskSelection(unittest.TestCase):

    def test_select_high_risk_set(self):
        results = [
            {'record_id': 'a', 'p_base_percent': 30, 'network_increment': 10},
            {'record_id': 'b', 'p_base_percent': 5, 'network_increment': 2},
        ]
        high = select_high_risk_set(results, threshold=DEFAULT_HIGH_RISK_THRESHOLD)
        self.assertEqual(len(high), 1)
        self.assertEqual(high[0]['record_id'], 'a')
        self.assertGreaterEqual(high[0]['decision_risk'], 15.0)


# ---------------------------------------------------------------------------
# 6. 端到端流水线
# ---------------------------------------------------------------------------

class TestPipeline(unittest.TestCase):

    def test_run_pipeline_end_to_end(self):
        contact = _make_contact(record_id='c1')
        assessment = _make_assessment(
            [_make_contact('家人1', cumulative_exposure=80)])
        report = run_three_layer_pipeline(
            _FakeMLPredictor(40.0), assessment, contact, 'family',
            run_intervention=True, population=500,
            t_horizon_days=182, random_state=42)
        self.assertEqual(report['architecture'], 'three_layer')
        self.assertIn('layer1_individual', report['layers'])
        self.assertIn('layer2_network', report['layers'])
        self.assertIn('layer3_intervention', report['layers'])
        self.assertEqual(report['decision']['integration_method'], 'stacking_gating')
        self.assertIn('decision_risk', report['summary'])
        self.assertIn('baseline_probability', report['summary'])

    def test_run_pipeline_without_intervention(self):
        contact = _make_contact(record_id='c1')
        assessment = _make_assessment([])
        report = ThreeLayerArchitecture().run_pipeline(
            _FakeMLPredictor(30.0), assessment, contact, 'family',
            run_intervention=False)
        self.assertIsNone(report['layers']['layer3_intervention'])
        self.assertIn('combined_probability', report['decision'])


# ---------------------------------------------------------------------------
# 7. 集成器接入
# ---------------------------------------------------------------------------

class TestIntegrator(unittest.TestCase):

    def test_integrator_three_layer_available(self):
        from tb_risk.core.integrator import ThreeDirectionIntegrator
        self.assertTrue(ThreeDirectionIntegrator().three_layer_available)

    def test_integrator_integrate_three_layer(self):
        from tb_risk.core.integrator import ThreeDirectionIntegrator
        contact = _make_contact(record_id='c1')
        assessment = _make_assessment([])
        result = ThreeDirectionIntegrator().integrate_three_layer(
            _FakeMLPredictor(30.0), assessment, contact, 'family',
            run_intervention=False)
        self.assertEqual(result['architecture'], 'three_layer')
        self.assertIn('summary', result)


# ---------------------------------------------------------------------------
# 8. 网络消融（ΔAUROC / ΔC-index）
# ---------------------------------------------------------------------------

class TestAblation(unittest.TestCase):

    def test_compute_c_index_perfect(self):
        self.assertEqual(compute_c_index([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_compute_c_index_random(self):
        c = compute_c_index([0, 1] * 10, [0.5] * 20)
        self.assertAlmostEqual(c, 0.5, places=4)

    def test_compute_c_index_constant_score(self):
        c = compute_c_index([0, 1, 1, 0], [0.7, 0.7, 0.7, 0.7])
        self.assertAlmostEqual(c, 0.5, places=4)

    def test_network_contribution_ablation_synthetic(self):
        rep = network_contribution_ablation(
            n_contacts=60, n_cases=15, random_state=7)
        self.assertIn('baseline', rep)
        self.assertIn('full', rep)
        self.assertIn('delta', rep)
        self.assertIn('network_contributes', rep)
        self.assertGreaterEqual(rep['full']['auroc'], 0.0)
        self.assertLessEqual(rep['full']['auroc'], 1.0)

    def test_ablation_returns_delta(self):
        rep = network_contribution_ablation(
            n_contacts=40, n_cases=8, random_state=3)
        self.assertIn('delta_auroc', rep['delta'])
        self.assertIn('delta_cindex', rep['delta'])

    def test_build_synthetic_network(self):
        recs, labels = build_synthetic_network(n_contacts=20, n_cases=5, random_state=1)
        self.assertEqual(len(recs), 20)
        self.assertEqual(sum(labels), 5)
        self.assertTrue(all('record_id' in r for r in recs))

    def test_network_contributes_positively_default(self):
        """v3 回归（2026-08-25 结论反转后重述）：标签机制统一为
        暴露×宿主（v3-host-pathway）后，网络层在合成基准上无独立判别
        增益（20 种子 ΔAUROC=-0.0264，CI [-0.0359,-0.0177]）——旧
        +0.1406 是"病例纯由簇隐藏风险决定"标签的循环论证产物。
        本测试不再断言方向，改为验证机制性质：网络层确实改变评分、
        结论字段与 Δ 符号一致、数值在合理范围。"""
        rep = network_contribution_ablation(
            n_contacts=60, n_cases=15, random_state=42)
        delta = rep['delta']['delta_auroc']
        self.assertGreaterEqual(delta, -0.5)
        self.assertLessEqual(delta, 0.5)
        # 结论字段与实现的显著性定义一致（Δ>0.005 才算贡献，
        # 不撒谎；边界区间 0<Δ≤0.005 两者可不同——2026-08-25 环境
        # 迁移实测 Δ=0.0044 落入该区间暴露本断言与阈值不一致）
        expected = (rep['delta']['delta_auroc'] > 0.005
                    or rep['delta']['delta_cindex'] > 0.005)
        self.assertEqual(rep['network_contributes'], expected)
        # 网络层非恒等变换：完整层评分与基线评分确有差异
        self.assertNotEqual(rep['full']['auroc'], rep['baseline']['auroc'])

    def test_network_contributes_positively_user_config(self):
        """v3 回归（结论反转后重述）：(40/12, seed 42) 配置下不再断言
        正向 Δ（v3 标签下单 seed 实测 -0.0149，方向由多种子 CI 裁决）；
        改为验证该配置消融可运行且报告自洽。"""
        rep = network_contribution_ablation(
            n_contacts=40, n_cases=12, random_state=42)
        delta = rep['delta']['delta_auroc']
        self.assertGreaterEqual(delta, -0.5)
        self.assertLessEqual(delta, 0.5)
        expected = (rep['delta']['delta_auroc'] > 0.005
                    or rep['delta']['delta_cindex'] > 0.005)
        self.assertEqual(rep['network_contributes'], expected)

    def test_network_signal_label_free(self):
        """v3 回归（加强）：网络层使用同簇特征聚合，不读取邻居标签。

        合成网络内已确诊记录全部伪装为未确诊后，消融结果必须与
        原始完全一致（逐字段）——这是"无标签泄漏"的确定性证据，
        比 v2.0 时代的"隐藏标签后 Δ>0"断言更强（v3 标签下 Δ 方向
        已由多种子 CI 裁决为无独立增益，方向断言不再适用）。
        """
        from tb_risk.validation import LayerAblationAnalyzer
        recs, labels = build_synthetic_network(
            n_contacts=60, n_cases=15, random_state=42)
        an = LayerAblationAnalyzer(random_state=42)
        rep_with_labels = an.run_ablation(recs, labels=labels)
        # 清空标签字段（特征聚合不读标签），结果应完全不变
        recs_hidden = []
        for r in recs:
            rr = dict(r)
            rr['is_confirmed'] = 0
            recs_hidden.append(rr)
        rep_no_labels = an.run_ablation(recs_hidden, labels=labels)
        self.assertEqual(
            rep_no_labels['delta']['delta_auroc'],
            rep_with_labels['delta']['delta_auroc'])
        self.assertEqual(
            rep_no_labels['full']['auroc'], rep_with_labels['full']['auroc'])
        self.assertEqual(
            rep_no_labels['baseline']['auroc'],
            rep_with_labels['baseline']['auroc'])

    def test_synthetic_network_deterministic(self):
        """v2.0 回归：合成网络在固定种子上逐字段可复现。"""
        a, la = build_synthetic_network(n_contacts=40, n_cases=12, random_state=5)
        b, lb = build_synthetic_network(n_contacts=40, n_cases=12, random_state=5)
        self.assertEqual(a, b)
        self.assertTrue((la == lb).all())


# ---------------------------------------------------------------------------
# 9. 任务分解与临床终点
# ---------------------------------------------------------------------------

class TestTaskDecomposition(unittest.TestCase):

    def test_task_endpoint_map(self):
        self.assertEqual(TASK_ENDPOINT_MAP[TaskType.SCREENING], EndPoint.ACTIVE_TB)
        self.assertEqual(TASK_ENDPOINT_MAP[TaskType.PROGNOSIS], EndPoint.PROGRESSION)
        self.assertEqual(TASK_ENDPOINT_MAP[TaskType.NETWORK], EndPoint.CONTACT_INFECTION)

    def test_assign_task_labels_distinct_endpoints(self):
        recs = [
            {'is_confirmed': 1, 'progression': 0, 'infection_status': 1},
            {'is_confirmed': 0, 'progression': 1, 'infection_status': 0},
        ]
        labels_a, ep_a = assign_task_labels(recs, TaskType.SCREENING)
        labels_b, ep_b = assign_task_labels(recs, TaskType.PROGNOSIS)
        labels_c, ep_c = assign_task_labels(recs, TaskType.NETWORK)
        self.assertEqual(labels_a, [1, 0])
        self.assertEqual(labels_b, [0, 1])
        self.assertEqual(labels_c, [1, 0])
        # 终点不可混用：同一批数据三个任务标签不同
        self.assertNotEqual(labels_a, labels_b)

    def test_evaluate_task_a_metrics(self):
        probs = [5, 60, 70, 30, 10, 80, 20, 55, 15, 90]
        out = [0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
        rep = evaluate_task_a(probs, out)
        self.assertEqual(rep['task'], 'A')
        self.assertIn('sensitivity', rep['metrics'])
        self.assertIn('specificity', rep['metrics'])
        self.assertIn('ppv', rep['metrics'])
        self.assertIn('npv', rep['metrics'])
        self.assertIn('youden_index', rep['metrics'])
        self.assertIn('expected_calibration_error', rep['metrics'])
        self.assertIn('confusion_matrix', rep)

    def test_evaluate_task_b_metrics(self):
        probs = [5, 60, 70, 30, 10, 80, 20, 55, 15, 90]
        out = [0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
        rep = evaluate_task_b(probs, out, horizon_days=365)
        self.assertEqual(rep['task'], 'B')
        self.assertIn('c_index', rep['metrics'])
        self.assertIn('horizon_auc', rep['metrics'])
        self.assertIn('decision_curve_analysis', rep)
        self.assertIn('best_threshold', rep['decision_curve_analysis'])

    def test_evaluate_task_c_metrics(self):
        rep = evaluate_task_c(
            predicted_rankings=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            true_positive_indices={1, 3, 5, 7, 9},
            secondary_cases=8, n_infectious=4,
            traced_contacts={0, 1, 2, 3, 4}, infected_contacts={1, 3, 5})
        self.assertEqual(rep['task'], 'C')
        self.assertIn('top_5', rep['metrics']['hit_rate'])
        self.assertIn('mean_position', rep['metrics']['ranking_quality'])
        self.assertEqual(rep['metrics']['r_estimate'], 2.0)
        self.assertAlmostEqual(rep['metrics']['contact_tracing_efficiency'], 2 / 3, places=4)

    def test_evaluate_task_dispatch(self):
        probs = [5, 60, 70, 30]
        out = [0, 1, 1, 0]
        for t in ('A', 'B'):
            rep = evaluate_task(probs, out, t)
            self.assertEqual(rep['task'], t)


# ---------------------------------------------------------------------------
# 10. 导出完整性
# ---------------------------------------------------------------------------

class TestExports(unittest.TestCase):

    def test_architecture_exports(self):
        from tb_risk.scoring import (
            ThreeLayerArchitecture, gated_integration, select_high_risk_set,
            compute_individual_base, compute_network_increment,
            simulate_community_intervention, run_three_layer_pipeline,
        )
        self.assertTrue(callable(ThreeLayerArchitecture))
        self.assertTrue(callable(gated_integration))

    def test_validation_exports(self):
        from tb_risk.validation import (
            compute_c_index, network_contribution_ablation,
            LayerAblationAnalyzer, TaskType, evaluate_task,
        )
        self.assertTrue(callable(compute_c_index))
        self.assertTrue(callable(evaluate_task))

    def test_seir_exports(self):
        from tb_risk.seir import (
            InterventionCounterfactualSimulator, simulate_intervention_effects,
        )
        self.assertTrue(callable(InterventionCounterfactualSimulator))


if __name__ == '__main__':
    unittest.main()
