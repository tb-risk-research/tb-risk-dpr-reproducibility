#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验室证据整合（三层证据）测试套件

覆盖：
- 杆菌载量因子 bacillary_load（Ct 分档 / 涂片分级 / 原二值涂阳回退）
- 影像学解析 parse_imaging_findings 与范围因子 imaging_extent_factor
- 传染性加分 compute_infectivity_bonus（Ct / 影像替代 / 旧逻辑回退）
- 证据等级裁决 derive_lab_evidence_grade（GeneXpert > 涂片/IGRA > 影像 > 推断）
- MDR 治疗传染性衰减曲线 mdr_treatment_infectivity（标准 vs 拉长）
- IGRA 类型权重与进展修正因子 igra_progression_factor
- 低证据告警与不确定性自适应因子
- 端到端集成：patient_scorer 传染性加分 + scoring.engine IGRA 进展修正

遵循项目 unittest + pytest 约定（无强依赖，纯 Python 实现）。
"""
import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


# ══════════════════════════════════════════════════════════════
# 杆菌载量因子（传染性层）
# ══════════════════════════════════════════════════════════════

class TestBacillaryLoad(unittest.TestCase):
    def test_ct_high_load(self):
        """Ct < 22 → 高载量（≈涂阳 +++，因子 1.0）。"""
        from tb_risk.core.lab_evidence import bacillary_load
        self.assertEqual(bacillary_load(ct_value=18.0), 1.0)
        self.assertEqual(bacillary_load(ct_value=10.5), 1.0)

    def test_ct_medium_load(self):
        """22 ≤ Ct ≤ 28 → 中载量（因子 0.6）。"""
        from tb_risk.core.lab_evidence import bacillary_load
        self.assertEqual(bacillary_load(ct_value=22.0), 0.6)
        self.assertEqual(bacillary_load(ct_value=25.0), 0.6)
        self.assertEqual(bacillary_load(ct_value=28.0), 0.6)

    def test_ct_low_load(self):
        """Ct > 28 → 低载量（≈涂阴，因子 0.3）。"""
        from tb_risk.core.lab_evidence import bacillary_load
        self.assertEqual(bacillary_load(ct_value=28.1), 0.3)
        self.assertEqual(bacillary_load(ct_value=30.0), 0.3)

    def test_ct_priority_over_smear(self):
        """Ct 值优先于涂片分级与二值涂阳。"""
        from tb_risk.core.lab_evidence import bacillary_load
        self.assertEqual(
            bacillary_load(ct_value=18.0, smear_grade=0,
                           sputum_smear='涂阳'), 1.0)
        self.assertEqual(
            bacillary_load(ct_value=30.0, smear_grade=3,
                           sputum_smear='涂阳'), 0.3)

    def test_smear_grade_fallback(self):
        """无 Ct 时回退到涂片分级（0–3+）。"""
        from tb_risk.core.lab_evidence import bacillary_load
        self.assertEqual(bacillary_load(smear_grade=3), 1.0)
        self.assertEqual(bacillary_load(smear_grade=2), 0.6)
        self.assertEqual(bacillary_load(smear_grade=1), 0.45)
        self.assertEqual(bacillary_load(smear_grade=0), 0.3)

    def test_sputum_smear_fallback(self):
        """无 Ct/涂片分级时回退到原二值涂阳（向后兼容）。"""
        from tb_risk.core.lab_evidence import bacillary_load
        self.assertEqual(bacillary_load(sputum_smear='涂阳'), 1.0)
        self.assertEqual(bacillary_load(sputum_smear='是'), 1.0)
        self.assertEqual(bacillary_load(sputum_smear=2), 1.0)

    def test_invalid_ct_falls_through(self):
        """非法 Ct 值（无法解析）视为缺失，回退到下级证据。"""
        from tb_risk.core.lab_evidence import bacillary_load
        self.assertEqual(bacillary_load(ct_value='abc', smear_grade=3), 1.0)
        self.assertIsNone(bacillary_load(ct_value='abc'))

    def test_no_evidence_returns_none(self):
        """无任何传染性证据时返回 None（不破坏现有流程）。"""
        from tb_risk.core.lab_evidence import bacillary_load
        self.assertIsNone(bacillary_load())
        self.assertIsNone(bacillary_load(ct_value=None))
        self.assertIsNone(bacillary_load(ct_value='not_done'))


# ══════════════════════════════════════════════════════════════
# 影像学解析与范围因子（疾病状态层）
# ══════════════════════════════════════════════════════════════

class TestImagingFindings(unittest.TestCase):
    def test_parse_bitmask(self):
        """整数位掩码解析（bit 0=cavity, 1=tree_in_bud, ...）。"""
        from tb_risk.core.lab_evidence import parse_imaging_findings
        self.assertEqual(parse_imaging_findings(1), {'cavity'})
        self.assertEqual(parse_imaging_findings(2), {'tree_in_bud'})
        self.assertEqual(parse_imaging_findings(3), {'cavity', 'tree_in_bud'})
        self.assertEqual(parse_imaging_findings(0), set())

    def test_parse_string_list(self):
        """字符串列表 / 逗号分隔 / 中文输入归一化。"""
        from tb_risk.core.lab_evidence import parse_imaging_findings
        self.assertEqual(parse_imaging_findings(['cavity', 'miliary']),
                         {'cavity', 'miliary'})
        self.assertEqual(parse_imaging_findings('空洞,树芽征'),
                         {'cavity', 'tree_in_bud'})
        self.assertEqual(parse_imaging_findings('实变、胸膜积液'),
                         {'consolidation', 'pleural_effusion'})
        self.assertEqual(parse_imaging_findings('结节'), {'nodule'})

    def test_parse_missing(self):
        """缺失影像字段 → 空集（不报错）。"""
        from tb_risk.core.lab_evidence import parse_imaging_findings
        self.assertEqual(parse_imaging_findings(None), set())
        self.assertEqual(parse_imaging_findings(''), set())
        self.assertEqual(parse_imaging_findings('未检测'), set())

    def test_imaging_extent_factor_max(self):
        """范围因子取最大值：cavity/tree_in_bud → 0.8。"""
        from tb_risk.core.lab_evidence import imaging_extent_factor
        self.assertEqual(imaging_extent_factor(3), 0.8)   # cavity + tree_in_bud
        self.assertEqual(imaging_extent_factor('空洞'), 0.8)
        self.assertEqual(imaging_extent_factor(['consolidation', 'miliary']),
                         0.6)
        self.assertEqual(imaging_extent_factor('结节'), 0.4)

    def test_imaging_extent_factor_missing(self):
        """无影像证据 → None。"""
        from tb_risk.core.lab_evidence import imaging_extent_factor
        self.assertIsNone(imaging_extent_factor(None))
        self.assertIsNone(imaging_extent_factor(''))


# ══════════════════════════════════════════════════════════════
# 传染性加分（源患者）
# ══════════════════════════════════════════════════════════════

class TestInfectivityBonus(unittest.TestCase):
    def test_ct_driven_bonus(self):
        """传染性加分 = 20 × bacillary_load(Ct)。"""
        from tb_risk.core.lab_evidence import compute_infectivity_bonus
        self.assertEqual(compute_infectivity_bonus({'genexpert_ct': 18.0}),
                         20.0)
        self.assertEqual(compute_infectivity_bonus({'genexpert_ct': 25.0}),
                         12.0)
        self.assertEqual(compute_infectivity_bonus({'genexpert_ct': 30.0}),
                         6.0)

    def test_imaging_fallback_bonus(self):
        """Ct/涂片缺失但影像可用时用影像范围因子替代。"""
        from tb_risk.core.lab_evidence import compute_infectivity_bonus
        self.assertEqual(
            compute_infectivity_bonus({'imaging_findings': '空洞'}), 16.0)
        self.assertEqual(
            compute_infectivity_bonus({'imaging_findings': '结节'}), 8.0)

    def test_sputum_smear_only_bonus(self):
        """无实验室证据但涂阳 → 保持旧逻辑 +20。"""
        from tb_risk.core.lab_evidence import compute_infectivity_bonus
        self.assertEqual(
            compute_infectivity_bonus({'sputum_smear': '涂阳'}), 20.0)

    def test_no_evidence_zero(self):
        """无任何证据 → 0 加分（不改变历史结果）。"""
        from tb_risk.core.lab_evidence import compute_infectivity_bonus
        self.assertEqual(compute_infectivity_bonus({}), 0.0)
        self.assertEqual(compute_infectivity_bonus(None), 0.0)


# ══════════════════════════════════════════════════════════════
# 证据等级裁决
# ══════════════════════════════════════════════════════════════

class TestEvidenceGrade(unittest.TestCase):
    def test_genexpert_grade3(self):
        """GeneXpert（Ct 或 RIF）→ 等级 3。"""
        from tb_risk.core.lab_evidence import derive_lab_evidence_grade
        self.assertEqual(derive_lab_evidence_grade({'genexpert_ct': 25.0}), 3)
        self.assertEqual(
            derive_lab_evidence_grade({'genexpert_rif': 'resistant'}), 3)

    def test_smear_and_igra_grade2(self):
        """涂片分级 / IGRA → 等级 2。"""
        from tb_risk.core.lab_evidence import derive_lab_evidence_grade
        self.assertEqual(derive_lab_evidence_grade({'smear_grade': 2}), 2)
        self.assertEqual(
            derive_lab_evidence_grade({'igra_result': 'positive'}), 2)

    def test_imaging_grade1(self):
        """影像学 → 等级 1。"""
        from tb_risk.core.lab_evidence import derive_lab_evidence_grade
        self.assertEqual(derive_lab_evidence_grade({'imaging_findings': '空洞'}),
                         1)

    def test_priority_conflict(self):
        """多证据冲突时取最高等级（Ct 压过涂片/影像）。"""
        from tb_risk.core.lab_evidence import derive_lab_evidence_grade
        rec = {'genexpert_ct': 25.0, 'smear_grade': 0,
               'imaging_findings': '空洞'}
        self.assertEqual(derive_lab_evidence_grade(rec), 3)

    def test_no_evidence_grade0(self):
        """无证据 → 等级 0（不报错）。"""
        from tb_risk.core.lab_evidence import derive_lab_evidence_grade
        self.assertEqual(derive_lab_evidence_grade({}), 0)
        self.assertEqual(derive_lab_evidence_grade(None), 0)
        self.assertEqual(derive_lab_evidence_grade(
            {'genexpert_ct': 'not_done', 'igra_result': 'not_done'}), 0)


# ══════════════════════════════════════════════════════════════
# MDR 治疗传染性衰减曲线
# ══════════════════════════════════════════════════════════════

class TestMdrInfectivity(unittest.TestCase):
    def test_standard_curve(self):
        """非 MDR：沿用既有 TREATMENT_INFECTIVITY_FACTORS 曲线。"""
        from tb_risk.core.lab_evidence import mdr_treatment_infectivity
        self.assertEqual(mdr_treatment_infectivity(0, is_mdr=False), 1.0)
        self.assertEqual(mdr_treatment_infectivity(5, is_mdr=False), 0.85)
        self.assertEqual(mdr_treatment_infectivity(30, is_mdr=False), 0.60)
        self.assertEqual(mdr_treatment_infectivity(100, is_mdr=False), 0.30)
        self.assertEqual(mdr_treatment_infectivity(200, is_mdr=False), 0.10)

    def test_mdr_extended_curve(self):
        """MDR：后期/完成档位阈值拉长，传染性更高。"""
        from tb_risk.core.lab_evidence import mdr_treatment_infectivity
        # 同一治疗时长下 MDR 传染性高于敏感株（窗口更长）
        self.assertEqual(mdr_treatment_infectivity(100, is_mdr=True), 0.60)
        self.assertGreater(
            mdr_treatment_infectivity(100, is_mdr=True),
            mdr_treatment_infectivity(100, is_mdr=False))
        # 完成治疗（≥2 年）后进一步下降
        self.assertEqual(mdr_treatment_infectivity(400, is_mdr=True), 0.30)
        self.assertLess(
            mdr_treatment_infectivity(800, is_mdr=True),
            mdr_treatment_infectivity(400, is_mdr=True))

    def test_mdr_beyond_completion(self):
        """超过 2 年：MDR 治愈后传染性显著降低但仍高于敏感株治愈。"""
        from tb_risk.core.lab_evidence import mdr_treatment_infectivity
        self.assertLess(mdr_treatment_infectivity(800, is_mdr=True), 0.30)
        self.assertGreater(
            mdr_treatment_infectivity(800, is_mdr=True),
            mdr_treatment_infectivity(800, is_mdr=False))

    def test_is_mdr_record(self):
        """RIF 耐药判定（支持中英文）。"""
        from tb_risk.core.lab_evidence import is_mdr_record
        self.assertTrue(is_mdr_record({'genexpert_rif': 'resistant'}))
        self.assertTrue(is_mdr_record({'genexpert_rif': '耐药'}))
        self.assertFalse(is_mdr_record({'genexpert_rif': 'susceptible'}))
        self.assertFalse(is_mdr_record({}))
        self.assertFalse(is_mdr_record(None))


# ══════════════════════════════════════════════════════════════
# IGRA / TST（进展层，接触者）
# ══════════════════════════════════════════════════════════════

class TestIgra(unittest.TestCase):
    def test_type_weight(self):
        """IGRA 类型置信度权重。"""
        from tb_risk.core.lab_evidence import igra_type_weight
        self.assertEqual(igra_type_weight('T-SPOT.TB'), 1.0)
        self.assertEqual(igra_type_weight('QuantiFERON'), 0.9)
        self.assertEqual(igra_type_weight('QFT-GIT'), 0.9)
        self.assertEqual(igra_type_weight('TST'), 0.7)
        self.assertEqual(igra_type_weight('PPD'), 0.7)
        self.assertEqual(igra_type_weight(None), 0.9)

    def test_positive_enables_baseline(self):
        """阳性：已感染，启用完整基线（×1.0）。"""
        from tb_risk.core.lab_evidence import igra_progression_factor
        self.assertEqual(igra_progression_factor({'igra_result': 'positive'}),
                         1.0)
        self.assertEqual(igra_progression_factor({'igra_result': '阳性'}), 1.0)

    def test_negative_suppresses(self):
        """阴性：进展概率压到接近 0（未感染基本不进展）。"""
        from tb_risk.core.lab_evidence import igra_progression_factor
        # T-SPOT（权重 1.0）：完全采纳 0.05
        self.assertAlmostEqual(igra_progression_factor(
            {'igra_result': 'negative', 'igra_type': 'T-SPOT.TB'}), 0.05)
        # TST（权重 0.7）：阴性结果可信度低，向 1.0 回摆
        tst = igra_progression_factor(
            {'igra_result': 'negative', 'igra_type': 'TST'})
        self.assertGreater(tst, 0.05)
        self.assertAlmostEqual(tst, 1.0 + (0.05 - 1.0) * 0.7)

    def test_indeterminate_partial(self):
        """不确定：部分保留（×0.8，按类型权重回摆）。"""
        from tb_risk.core.lab_evidence import igra_progression_factor
        indet = igra_progression_factor(
            {'igra_result': 'indeterminate', 'igra_type': 'T-SPOT.TB'})
        self.assertAlmostEqual(indet, 0.8)
        self.assertAlmostEqual(igra_progression_factor(
            {'igra_result': 'indeterminate'}), 1.0 + (0.8 - 1.0) * 0.9)

    def test_not_done_keeps_current(self):
        """未检测 / 缺失：保持现状（×1.0）。"""
        from tb_risk.core.lab_evidence import igra_progression_factor
        self.assertEqual(igra_progression_factor({}), 1.0)
        self.assertEqual(igra_progression_factor(None), 1.0)
        self.assertEqual(igra_progression_factor({'igra_result': 'not_done'}),
                         1.0)


# ══════════════════════════════════════════════════════════════
# 低证据告警与不确定性
# ══════════════════════════════════════════════════════════════

class TestUncertainty(unittest.TestCase):
    def test_adaptive_factor(self):
        """证据等级越低，共形预测区间放大越多。"""
        from tb_risk.core.lab_evidence import uncertainty_adaptive_factor
        self.assertEqual(uncertainty_adaptive_factor({}), 1.6)
        self.assertEqual(uncertainty_adaptive_factor({'imaging_findings': '空洞'}),
                         1.3)
        self.assertEqual(uncertainty_adaptive_factor({'smear_grade': 2}), 1.0)
        self.assertEqual(uncertainty_adaptive_factor({'genexpert_ct': 25.0}),
                         1.0)

    def test_low_evidence_warning(self):
        """无实验室证据时返回告警文本（GUI 标注）。"""
        from tb_risk.core.lab_evidence import low_evidence_warning
        self.assertEqual(low_evidence_warning({}), '基于症状推断，建议实验室确诊')
        self.assertIsNone(low_evidence_warning({'genexpert_ct': 25.0}))


# ══════════════════════════════════════════════════════════════
# 端到端集成
# ══════════════════════════════════════════════════════════════

class TestIntegration(unittest.TestCase):
    def test_patient_scorer_evidence_fields(self):
        """patient_scorer 输出含传染性加分与证据等级。"""
        from tb_risk.core.patient_scorer import PatientScorer
        scorer = PatientScorer()
        patient = {
            'basic_info': {
                'genexpert_ct': 25.0, 'sputum_smear': 1,
                'has_cavity': 1, 'active_tb': 1, 'treatment': 2,
            },
            'FCI': 0.0, 'SNC': 0.0,
        }
        result = scorer.compute_patient_score(patient)
        self.assertEqual(result['infectivity_bonus'], 12.0)  # 20 × 0.6
        self.assertEqual(result['lab_evidence_grade'], 3)
        self.assertFalse(result['is_mdr'])

    def test_patient_scorer_mdr_flag(self):
        """RIF 耐药患者的 is_mdr 标记透传。"""
        from tb_risk.core.patient_scorer import PatientScorer
        scorer = PatientScorer()
        patient = {
            'basic_info': {'genexpert_rif': 'resistant'},
            'FCI': 0.0, 'SNC': 0.0,
        }
        result = scorer.compute_patient_score(patient)
        self.assertTrue(result['is_mdr'])
        self.assertEqual(result['lab_evidence_grade'], 3)

    def test_engine_igra_gate_orders_risk(self):
        """IGRA 感染门槛：阳性 > 未检测 > 阴性（disease_prob 由 P_infected 驱动）。"""
        from tb_risk.scoring.engine import ScoringEngine
        engine = ScoringEngine(use_localization=False)
        # 高暴露 base：未检测的暴露先验 > IGRA 阴性残留(0.05)，保证排序成立
        base = {
            'has_symptoms': 1, 'has_tb': 1, 'bcg_vaccine': 0,
            'exposure_setting': 'crowded', 'contact_distance': 'close',
            'ventilation': 1, 'cumulative_exposure': 120, 'age': 30,
        }
        no_igra = engine.compute_risk_score(dict(base))
        neg = engine.compute_risk_score(dict(base, igra_result='negative',
                                             igra_type='T-SPOT.TB'))
        pos = engine.compute_risk_score(dict(base, igra_result='positive'))
        # 未感染者（IGRA 阴性）自动逼近 0，感染者按分层
        self.assertLess(neg['disease_probability'],
                        no_igra['disease_probability'])
        self.assertGreater(pos['disease_probability'],
                           no_igra['disease_probability'])
        self.assertLess(neg['latent_infection_prob'],
                        no_igra['latent_infection_prob'])
        self.assertGreater(pos['latent_infection_prob'],
                           no_igra['latent_infection_prob'])
        self.assertEqual(neg['lab_evidence_grade'], 2)

    def test_engine_dual_outputs(self):
        """engine 返回双输出：latent_infection_prob 与 short_term_active_risk。"""
        from tb_risk.scoring.engine import ScoringEngine
        engine = ScoringEngine(use_localization=False)
        rec = {'igra_result': 'positive', 'time_since_exposure_months': 3,
               'recent_conversion': True, 'age': 4}
        res = engine.compute_risk_score(rec)
        self.assertIn('latent_infection_prob', res)
        self.assertIn('short_term_active_risk', res)
        # short_term_active_risk = P_infected × P_progress（×100 → disease_prob）
        self.assertAlmostEqual(
            res['short_term_active_risk'] * 100.0,
            res['disease_probability'], places=6)
        self.assertAlmostEqual(res['latent_infection_prob'], 0.92)

    def test_engine_evidence_grade(self):
        """engine 输出含证据等级（IGRA→2，无证据→0）。"""
        from tb_risk.scoring.engine import ScoringEngine
        engine = ScoringEngine(use_localization=False)
        rec = {'igra_result': 'positive'}
        self.assertEqual(engine.compute_risk_score(rec)['lab_evidence_grade'],
                         2)
        self.assertEqual(engine.compute_risk_score({})['lab_evidence_grade'],
                         0)


# ══════════════════════════════════════════════════════════════
# 双输出进展建模（时间衰减 / 感染门槛 / 生活方式）
# ══════════════════════════════════════════════════════════════

class TestDualOutputModeling(unittest.TestCase):
    def test_time_decay_factor_bounds(self):
        """时间衰减分档：近期高峰 → 远期再激活率低。"""
        from tb_risk.core.lab_evidence import time_decay_factor
        self.assertEqual(time_decay_factor(0), 1.0)
        self.assertEqual(time_decay_factor(6), 1.0)      # t ≤ 6
        self.assertEqual(time_decay_factor(7), 0.5)      # 6 < t ≤ 24
        self.assertEqual(time_decay_factor(24), 0.5)
        self.assertEqual(time_decay_factor(30), 0.2)     # 24 < t ≤ 60
        self.assertEqual(time_decay_factor(60), 0.2)
        self.assertEqual(time_decay_factor(120), 0.1)    # t > 60
        # 缺失/非法输入默认落入 0.5 档（保守居中）
        self.assertEqual(time_decay_factor(None), 0.5)
        self.assertEqual(time_decay_factor('abc'), 0.5)

    def test_latent_infection_probability(self):
        """P_infected：IGRA 判定感染状态，未检测回退暴露先验。"""
        from tb_risk.core.lab_evidence import latent_infection_probability
        self.assertEqual(latent_infection_probability({'igra_result': 'positive'}),
                         0.92)
        self.assertEqual(latent_infection_probability({'igra_result': 'negative'}),
                         0.05)
        # 不确定：暴露先验缺省 0.3（低于缺省时取 0.3 地板）
        self.assertEqual(latent_infection_probability(
            {'igra_result': 'indeterminate'}), 0.30)
        self.assertEqual(latent_infection_probability(
            {'igra_result': 'indeterminate'}, exposure_prior=0.1), 0.30)
        self.assertEqual(latent_infection_probability(
            {'igra_result': 'indeterminate'}, exposure_prior=0.5), 0.50)
        # 未检测：沿用暴露先验（上限不超过阳性确认值 0.92）
        self.assertEqual(latent_infection_probability(
            {'igra_result': 'not_done'}, exposure_prior=0.4), 0.40)
        self.assertEqual(latent_infection_probability(
            {}, exposure_prior=0.95), 0.92)

    def test_lifestyle_progression_factor(self):
        """生活方式乘数：吸烟/BMI/维生素 D 分档，缺失按默认。"""
        from tb_risk.core.lab_evidence import lifestyle_progression_factor
        # 缺失全部 → ×1.0
        self.assertEqual(lifestyle_progression_factor({}), 1.0)
        # 吸烟
        self.assertEqual(lifestyle_progression_factor({'smoking': True}), 1.2)
        # BMI 分档
        self.assertEqual(lifestyle_progression_factor({'bmi': 17.0}), 2.0)
        self.assertEqual(lifestyle_progression_factor({'bmi': 24.0}), 1.0)
        self.assertEqual(lifestyle_progression_factor({'bmi': 27.0}), 0.95)
        self.assertEqual(lifestyle_progression_factor({'bmi': 32.0}), 0.9)
        # 维生素 D 分档
        self.assertEqual(lifestyle_progression_factor({'vitamin_d': 10.0}), 1.3)
        self.assertEqual(lifestyle_progression_factor({'vitamin_d': 15.0}), 1.15)
        self.assertEqual(lifestyle_progression_factor({'vitamin_d': 30.0}), 1.0)
        self.assertEqual(lifestyle_progression_factor({'vitamin_d': 45.0}), 0.95)
        # 组合
        self.assertAlmostEqual(
            lifestyle_progression_factor(
                {'smoking': True, 'bmi': 17.0, 'vitamin_d': 10.0}),
            1.2 * 2.0 * 1.3)

    def test_short_term_progression_probability(self):
        """P_progress：基线 × 时间衰减 × 年龄/免疫 × (1−BCG) × 生活方式 × 阳转放大。"""
        from tb_risk.core.lab_evidence import short_term_progression_probability
        # 基线成人：0.03 × decay(3月=1.0) = 0.03
        self.assertAlmostEqual(
            short_term_progression_probability(
                {'time_since_exposure_months': 3}), 0.03)
        # 时间衰减：6 月内 1.0 vs 远期 0.1
        self.assertAlmostEqual(
            short_term_progression_probability(
                {'time_since_exposure_months': 3}),
            0.03)
        self.assertAlmostEqual(
            short_term_progression_probability(
                {'time_since_exposure_months': 120}),
            0.003)
        # 年龄/免疫乘数放大
        self.assertAlmostEqual(
            short_term_progression_probability(
                {'time_since_exposure_months': 3}, age_multiplier=4.0),
            0.12)
        # BCG 保护：保护 50% → 进展减半
        self.assertAlmostEqual(
            short_term_progression_probability(
                {'time_since_exposure_months': 3}, bcg_protective=0.5),
            0.015)
        # 近期阳转放大：×(1 + 1.5)
        self.assertAlmostEqual(
            short_term_progression_probability(
                {'time_since_exposure_months': 3, 'recent_conversion': True}),
            0.03 * 2.5)
        # 概率上限（极端组合不超 1）
        p = short_term_progression_probability(
            {'time_since_exposure_months': 3, 'recent_conversion': True,
             'smoking': True, 'bmi': 17.0, 'vitamin_d': 10.0},
            age_multiplier=4.0, immuno_multiplier=8.0, cap=0.30)
        self.assertLessEqual(p, 0.30)

    def test_batch_matches_single_dual_outputs(self):
        """批量路径与单次路径的双输出一致（含 IGRA/时间/生活方式分支）。"""
        from tb_risk.scoring.engine import ScoringEngine
        engine = ScoringEngine(use_localization=False)
        records = [
            {'age': 4, 'igra_result': 'positive',
             'time_since_exposure_months': 3, 'recent_conversion': True},
            {'age': 30, 'igra_result': 'negative', 'igra_type': 'T-SPOT.TB',
             'time_since_exposure_months': 40, 'smoking': True},
            {'age': 70, 'igra_result': 'not_done', 'bmi': 17.0,
             'vitamin_d': 10.0, 'time_since_exposure_months': 120},
        ]
        single = [engine.compute_risk_score(r) for r in records]
        batch = engine.compute_batch(records)
        for i, r in enumerate(records):
            for key in ('disease_probability', 'infection_probability',
                        'latent_infection_prob', 'short_term_active_risk'):
                self.assertAlmostEqual(single[i][key], batch[i][key],
                                       places=6,
                                       msg=f"{key} 记录{i} 不一致")
            # 未感染者自动逼近 0
            if r.get('igra_result') == 'negative':
                self.assertLess(single[i]['disease_probability'], 0.5)

    def test_contact_risk_calculator_exposes_dual_outputs(self):
        """ContactRiskCalculator 单接触者路径输出双输出量（IGRA 门槛生效）。"""
        from tb_risk.core.contact_risk_calculator import ContactRiskCalculator
        from tb_risk.core.seir_integration import SEIRIntegration
        calc = ContactRiskCalculator(seir_integration=SEIRIntegration())
        base = {'age': 30, 'past_illness_type': 'none',
                'cumulative_exposure': 60}
        pos = dict(base, igra_result='positive',
                   time_since_exposure_months=3, recent_conversion=True)
        neg = dict(base, igra_result='negative')
        r_pos = calc.compute_single_contact_risk(pos, patient_treatment_days=0)
        r_neg = calc.compute_single_contact_risk(neg, patient_treatment_days=0)
        for key in ('latent_infection_prob', 'short_term_active_risk'):
            self.assertIn(key, r_pos)
            self.assertIn(key, r_neg)
        # 感染门槛生效：阳性潜伏感染概率显著高于阴性
        self.assertGreater(r_pos['latent_infection_prob'],
                           r_neg['latent_infection_prob'])
        self.assertGreater(r_pos['short_term_active_risk'],
                           r_neg['short_term_active_risk'])
        # 双输出在 0-1 范围
        for key in ('latent_infection_prob', 'short_term_active_risk'):
            for r in (r_pos, r_neg):
                self.assertGreaterEqual(r[key], 0.0)
                self.assertLessEqual(r[key], 1.0)


if __name__ == '__main__':
    unittest.main()
