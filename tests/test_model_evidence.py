#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型证据面板测试：core/model_evidence.py 与纯函数工具。

覆盖：
1. core.model_evidence 各证据包的可用性与结构契约
2. extract_three_layer_series 的防御性行为（None / 空 / 正常结构）
3. task_a_metrics_at_threshold 纯函数（混淆矩阵与诊断指标）
4. kaplan_meier 纯函数（生存曲线单调性与边界）
"""

import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from tb_risk.core import model_evidence as me
from tb_risk.gui.results_panel.model_evidence import (
    task_a_metrics_at_threshold, kaplan_meier, build_screening_state)


class TestEvidenceCollectors(unittest.TestCase):
    """各证据包可用性（后端 API 契约）。"""

    def test_task_evaluations_available(self):
        result = me.collect_task_evaluations(n_contacts=30, n_cases=8)
        self.assertTrue(result.get('available'))
        for tid in ('A', 'B', 'C'):
            self.assertIn(tid, result)
            self.assertIn('task_name', result[tid])
            self.assertIn('decision_meaning', result[tid])
        # 任务 A 指标键
        self.assertIn('sensitivity', result['A']['metrics'])
        self.assertIn('specificity', result['A']['metrics'])
        self.assertIn('ppv', result['A']['metrics'])
        self.assertIn('npv', result['A']['metrics'])
        self.assertIn('expected_calibration_error', result['A']['metrics'])
        # 任务 B 指标键
        self.assertIn('c_index', result['B']['metrics'])
        self.assertIn('horizon_auc', result['B']['metrics'])
        self.assertIn('decision_curve_analysis', result['B'])
        # 任务 C 指标键
        self.assertIn('hit_rate', result['C']['metrics'])
        self.assertIn('ranking_quality', result['C']['metrics'])
        self.assertIn('r_estimate', result['C']['metrics'])
        self.assertIn('contact_tracing_efficiency', result['C']['metrics'])

    def test_task_evaluations_deterministic(self):
        a = me.collect_task_evaluations(n_contacts=30, n_cases=8, random_state=7)
        b = me.collect_task_evaluations(n_contacts=30, n_cases=8, random_state=7)
        self.assertEqual(a['A']['metrics'], b['A']['metrics'])
        self.assertEqual(a['C']['metrics']['hit_rate'], b['C']['metrics']['hit_rate'])

    def test_network_ablation_available(self):
        result = me.collect_network_ablation(n_contacts=30, n_cases=8)
        self.assertTrue(result.get('available'))
        self.assertIn('baseline', result)
        self.assertIn('full', result)
        self.assertIn('delta', result)
        self.assertIn('delta_auroc', result['delta'])
        self.assertIn('delta_cindex', result['delta'])
        self.assertIn('network_contributes', result)
        self.assertIn('conclusion', result)

    def test_public_data_evidence_available(self):
        result = me.collect_public_data_evidence()
        self.assertTrue(result.get('available'))
        self.assertIn('summary', result)
        self.assertIn('strategy', result)
        coverage = result['summary']['coverage_by_task']
        self.assertIn('A', coverage)
        self.assertIn('B', coverage)
        self.assertIn('C', coverage)
        # 任务 A/B 可外部验证，任务 C 不可（方法学过渡）
        self.assertTrue(coverage['A']['external_validatable'])
        self.assertTrue(coverage['B']['external_validatable'])
        self.assertFalse(coverage['C']['external_validatable'])

    def test_dgp_evidence_available(self):
        result = me.collect_dgp_evidence(n_samples=1000)
        self.assertTrue(result.get('available'))
        self.assertIn('field_specs', result)
        self.assertIn('validation', result)
        self.assertIn('distributions', result['validation'])
        self.assertIn('prevalence', result['validation'])
        self.assertIn('reproducibility', result)
        self.assertTrue(result['reproducibility']['identical'])
        self.assertIn('overall_pass', result)

    def test_seir_intervention_available(self):
        result = me.collect_seir_intervention(
            high_risk_contacts=list(range(5)), t_horizon_days=365)
        self.assertTrue(result.get('available'))
        self.assertIn('baseline', result)
        self.assertIn('cases_curve', result['baseline'])
        self.assertIn('times', result['baseline'])
        self.assertIn('strategies', result)
        self.assertGreater(len(result['strategies']), 0)
        self.assertIn('summary', result)
        self.assertIn('best_averted_percent', result['summary'])

    def test_sinan_scale_evidence_available(self):
        """SINAN 百万级证据包：归档存在时返回头条叙事与稳健性曲线数据。"""
        result = me.collect_sinan_scale_evidence()
        archive = os.path.join(_PROJECT_ROOT, 'data', 'processed',
                               me.SINAN_SCALE_ARCHIVE)
        if not os.path.exists(archive):
            self.assertFalse(result.get('available'))
            self.assertIn('reason', result)
            return
        self.assertTrue(result.get('available'), result)
        for key in ('headline', 'narrative', 't1_series', 't2_series',
                    'best_model', 'decay_val_to_far', 't2_best_model',
                    'subgroups', 'top_features'):
            self.assertIn(key, result)
        # T1 稳健性曲线数据契约：每模型 [val, near, far] 三点
        for series in result['t1_series'].values():
            self.assertEqual(len(series), 3)
        self.assertIn(result['best_model'], result['t1_series'])
        # 头条叙事含百万级规模表述
        self.assertIn('SINAN', result['narrative'])
        self.assertIn('万', result['narrative'])
        # round-8 P1：终点语义从裁定字典读取（防旧数字误引）
        ep = result.get('endpoint_correction') or {}
        self.assertIn('code4_semantics', ep)
        self.assertIn('corrected_endpoint', ep)
        self.assertIn('非 TB 死亡', ep.get('code4_semantics', ''))
        self.assertIn('全因死亡', ep.get('corrected_endpoint', ''))
        self.assertIn('非 TB 死亡', result['narrative'])

    def test_sinan_scale_evidence_missing_archive(self):
        """归档缺失/不可读时优雅降级（不抛异常）。"""
        orig = me.SINAN_SCALE_ARCHIVE
        try:
            me.SINAN_SCALE_ARCHIVE = 'no_such_archive.json'
            result = me.collect_sinan_scale_evidence()
            self.assertFalse(result.get('available'))
            self.assertIn('reason', result)
        finally:
            me.SINAN_SCALE_ARCHIVE = orig


class TestNegativeFindings(unittest.TestCase):
    """阴性发现证据包（第 11 个子标签页的数据源）。"""

    def test_negative_findings_available(self):
        """归档存在时返回结构化 findings（含老年双阴性与字典修正）。"""
        result = me.collect_negative_findings()
        ids = {me._NEG_ARCHIVE_ELDERLY, me._NEG_ARCHIVE_ENDPOINT,
               me._NEG_ARCHIVE_DICT, me._NEG_ARCHIVE_PACTS}
        archives = [os.path.join(_PROJECT_ROOT, 'data', 'processed', a)
                    for a in ids]
        if not all(os.path.exists(p) for p in archives):
            # 归档缺失时 collect 仍会返回唯一的无条件 finding
            # （sinan_household_infeasible 是数据字段边界事实，不依赖归档）；
            # 归档派生 finding 则必须全部缺席
            archive_derived = {'elderly_specialization',
                               'elderly_endpoint_switch',
                               'label_dictionary_correction',
                               'pacts_endpoint_ceiling'}
            present = {f['id'] for f in result.get('findings', [])}
            self.assertFalse(present & archive_derived,
                             '归档缺失时不应出现归档派生 finding: %s' % present)
            return
        self.assertTrue(result.get('available'), result)
        for key in ('headline', 'narrative', 'findings',
                    'tradition_note'):
            self.assertIn(key, result)
        found = {f['id'] for f in result['findings']}
        self.assertIn('elderly_specialization', found)
        self.assertIn('elderly_endpoint_switch', found)
        self.assertIn('elderly_line_closure', found)
        self.assertIn('label_dictionary_correction', found)
        self.assertIn('pacts_endpoint_ceiling', found)
        self.assertIn('sinan_household_infeasible', found)
        # 每条 finding 的键契约
        for f in result['findings']:
            for key in ('id', 'verdict', 'title', 'detail',
                        'implication', 'archive'):
                self.assertIn(key, f, f"finding {f.get('id')} 缺 {key}")
        # verdict 枚举契约
        for f in result['findings']:
            self.assertIn(f['verdict'],
                          ('NEGATIVE', 'CORRECTED', 'INCONCLUSIVE'))
        # 老年三条为判别层阴性（专项训练/换终点/线关闭）
        n_neg = sum(1 for f in result['findings']
                   if f['verdict'] == 'NEGATIVE')
        self.assertGreaterEqual(n_neg, 3)
        # round-8 P3：老年线关闭 finding 的定稿结论与 E6 立项决策
        closure = next(f for f in result['findings']
                       if f['id'] == 'elderly_line_closure')
        self.assertIn('临床判断为主', closure['implication'])
        self.assertIn('不再投入调参', closure['detail'])
        self.assertIn('E6', closure['implication'])
        # round-9 P2-5：SINAN 户级下沉不可行（数据字段边界）
        hh = next(f for f in result['findings']
                  if f['id'] == 'sinan_household_infeasible')
        self.assertEqual(hh['verdict'], 'NEGATIVE')
        self.assertIn('无街道', hh['detail'])
        self.assertEqual(hh['metrics']['sinan_geo_resolution'],
                         'municipality')

    def test_negative_findings_elderly_numbers(self):
        """换终点阴性：E5（TB 特异终点）数值低于 E1（旧终点）。"""
        result = me.collect_negative_findings()
        # available 可为 True（含无条件 finding），归档派生 finding 缺席时跳过
        e5 = next((f for f in result.get('findings', [])
                   if f['id'] == 'elderly_endpoint_switch'), None)
        if e5 is None:
            self.skipTest('elderly_endpoint_switch 归档不存在')
        self.assertLess(e5['metrics']['elderly_E5'],
                        e5['metrics']['elderly_E1'])
        # 老年多类别头同样塌方（macro 显著低于 0.7）
        self.assertLess(e5['metrics']['elderly_multiclass_macro'], 0.7)

    def test_negative_findings_missing_archives(self):
        """归档全部缺失时优雅降级（不抛异常）。

        round-9 起 finding #6（SINAN 户级不可行）不读归档（字段清单
        事实），故归档全缺失时仅剩该条，其余归档依赖 findings 消失。
        """
        orig = me._neg_load
        try:
            me._neg_load = lambda archive: None
            result = me.collect_negative_findings()
            ids = {f['id'] for f in result.get('findings', [])}
            self.assertEqual(ids, {'sinan_household_infeasible'})
        finally:
            me._neg_load = orig


class TestGranularityGuide(unittest.TestCase):
    """组粒度部署指南证据包（第 12 个子标签页的数据源）。"""

    def test_granularity_guide_available(self):
        """归档存在时返回决策规则 + 三实例点 + 单调性。"""
        result = me.collect_granularity_guide()
        archive = os.path.join(_PROJECT_ROOT, 'data', 'processed',
                               me.GRANULARITY_ARCHIVE)
        if not os.path.exists(archive):
            self.assertFalse(result.get('available'))
            self.assertIn('reason', result)
            return
        self.assertTrue(result.get('available'), result)
        for key in ('headline', 'narrative', 'decision_rule',
                    'examples', 'points', 'monotonicity', 'plot_path'):
            self.assertIn(key, result)
        # 决策规则四要素（部署前可测量的判据）
        rule = result['decision_rule']
        for key in ('measure', 'worthwhile', 'not_worthwhile', 'grouping'):
            self.assertIn(key, rule)
        self.assertIn('ICC > 0.1', rule['worthwhile'])
        self.assertIn('ICC < 0.05', rule['not_worthwhile'])
        # 三实例点：HomeACF（值得）/SINAN 市（不值得）/PACTS（天花板）
        examples = {p['id']: p for p in result['examples']}
        self.assertEqual(len(result['examples']), 3)
        for pid in ('homeacf_household', 'sinan_munip_small',
                    'pacts_household'):
            self.assertIn(pid, examples)
        self.assertGreater(examples['homeacf_household']['icc'], 0.1)
        self.assertGreater(examples['homeacf_household']['delta'], 0.03)
        self.assertLess(examples['sinan_munip_small']['icc'], 0.05)
        self.assertLess(examples['sinan_munip_small']['delta'], 0.01)
        # ICC 与 ΔAUROC 的量级关系：户级增益约为市级的 10 倍以上
        ratio = (examples['homeacf_household']['delta']
                 / max(abs(examples['sinan_munip_small']['delta']), 1e-6))
        self.assertGreater(ratio, 10)

    def test_granularity_guide_missing_archive(self):
        """归档缺失时优雅降级（不抛异常）。"""
        orig = me.GRANULARITY_ARCHIVE
        try:
            me.GRANULARITY_ARCHIVE = 'no_such_granularity.json'
            result = me.collect_granularity_guide()
            self.assertFalse(result.get('available'))
            self.assertIn('reason', result)
        finally:
            me.GRANULARITY_ARCHIVE = orig


class TestDeploymentMetrics(unittest.TestCase):
    """部署度量证据包（第 13 个子标签页的数据源，round-9 P0-2）。"""

    def test_deployment_metrics_available(self):
        """归档存在时返回 yield 表 + 三子证据包结构。"""
        result = me.collect_deployment_metrics()
        archive = os.path.join(_PROJECT_ROOT, 'data', 'processed',
                               me.DEPLOY_ARCHIVE)
        if not os.path.exists(archive):
            self.assertFalse(result.get('available'))
            self.assertIn('reason', result)
            return
        self.assertTrue(result.get('available'), result)
        for key in ('headline', 'narrative', 'sinan',
                    'kenya_case_population', 'bacteriology_head',
                    'competing_risk', 'planner_api'):
            self.assertIn(key, result)
        # yield 表契约：预算行 + 关键字段
        rows = result['sinan']['yield_rows']
        self.assertGreaterEqual(len(rows), 5)
        budgets = {r['budget_pct'] for r in rows}
        self.assertIn('10', budgets)
        self.assertIn('100', budgets)
        for r in rows:
            for key in ('yield_ind', 'yield_ind_all', 'nns_ind_all',
                        'ppv_ind_all', 'lift_ind_all'):
                self.assertIn(key, r)
            # yield 单调性 + 满预算 = 1.0
            self.assertGreaterEqual(r['yield_ind_all'],
                                     r['yield_ind'] - 1e-9)
        self.assertEqual(rows[-1]['yield_ind_all'], 1.0)
        # 10% 预算的部署头条数字（far test 全因死亡）
        r10 = next(r for r in rows if r['budget_pct'] == '10')
        self.assertGreater(r10['yield_ind'], 0.30)
        self.assertGreater(r10['lift_ind_all'], 3.0)
        self.assertLess(r10['nns_ind_all'], 10.0)
        # 先证增益：yield@10% delta 为正（时序先证提升捕获）
        delta10 = (result['sinan']['prior_gain']['yield_at']['10']
                  ['delta'])
        self.assertGreater(delta10, 0)
        # 老年塌方的部署语言：yield@10% 低于成人
        self.assertIn('17', result['narrative'])
        # Kenya case-population 子包
        ken = result['kenya_case_population']
        if ken:
            self.assertGreater(ken['miss_rate'], 0.5)
            self.assertGreater(ken['top10_never_treated_pos'], 0)
        # 细菌学头子包：正交性
        bac = result['bacteriology_head']
        if bac:
            self.assertGreater(bac['death_auroc'], 0.75)
            self.assertLess(bac['top10_overlap'], 0.30)
        # 跨人群迁移子包（round-9 P3-6）
        trf = result['cross_population_transfer']
        if trf:
            self.assertGreater(trf['br_reference_auroc'], 0.6)
            cohorts = {c['cohort']: c for c in trf['cohorts']}
            self.assertEqual(len(cohorts), 3)
            ken = cohorts['Kenya 患病率调查（确诊终点）']
            self.assertGreater(ken['population_decay'], 0)
            vnm = cohorts['Vietnam PROVE_TB（Xpert 确诊终点）']
            self.assertLess(vnm['transfer_auroc'], 0.5)  # 负迁移实证
            self.assertGreaterEqual(len(trf['excluded']), 2)
        # 双头架构子包（round-10 P2）
        dh = result['dual_head']
        if dh:
            self.assertIn('A_prime', dh['task_mapping'])
            self.assertIn('B', dh['task_mapping'])
            self.assertIn('bacteriology_pos', dh['task_mapping']['A_prime'])
            self.assertIn('death_allcause', dh['task_mapping']['B'])
            self.assertEqual(len(dh['yield_rows']), 7)
            r10 = next(r for r in dh['yield_rows']
                       if r['budget_pct'] == '10')
            self.assertGreater(r10['yield_death'], 0.30)
            self.assertLess(r10['yield_bact'], r10['yield_death'])
            self.assertLess(dh['top10_overlap'], 0.30)
            self.assertIn('DualHeadPredictor', dh['predictor_api'])
            self.assertTrue(dh['model_artifacts'].get('saved'))
        # 分龄阈值策略表（round-10 P3）
        at = result['age_thresholds']
        if at:
            self.assertEqual(len(at['rows']), 7)
            budgets = {r['budget_pct'] for r in at['rows']}
            self.assertIn('10', budgets)
            self.assertIn('100', budgets)
            t10 = next(r for r in at['rows'] if r['budget_pct'] == '10')
            # 老年基线风险高 → 同预算阈值更高
            self.assertGreater(t10['thr_elderly'], t10['thr_adult'])
            # 检出 + 漏检 = 组内事件总数（守恒）
            self.assertEqual(t10['caught_elderly'] + t10['missed_elderly'],
                             at['n_events_elderly'])
            self.assertEqual(t10['caught_adult'] + t10['missed_adult'],
                             at['n_events_adult'])
            for r in at['rows']:
                self.assertGreaterEqual(r['thr_elderly'], 0.0)
                self.assertLessEqual(r['thr_elderly'], 1.0)
        # 老年双死因分解叙事（round-10 P4）
        ec = result['elderly_dual_cause']
        if ec:
            self.assertIn('标签稀释', ec['narrative'])
            self.assertIn('双死因分解', ec['narrative'])
            self.assertIn('HIV', ec['narrative'])
            self.assertTrue(len(ec['age_dilution']) >= 3)
            self.assertIn('论文', ec['paper_note'])

    def test_deployment_round11_subpackages(self):
        """round-11 五个子包：老年双终点 / 重校准 / 细菌学 v2 / 红线 / 封存。"""
        result = me.collect_deployment_metrics()
        archive = os.path.join(_PROJECT_ROOT, 'data', 'processed',
                               me.DEPLOY_ARCHIVE)
        if not os.path.exists(archive) or not result.get('available'):
            self.skipTest('部署归档不存在')
        # P1-1 老年双终点（全因 vs E3 并列）
        de = result['elderly_dual_endpoint']
        if de:
            rows = de['rows']
            self.assertEqual(len(rows), 4)   # 2 终点 × 2 组
            avail = [r for r in rows
                     if r.get('available') is not False]
            endpoints = {r['endpoint'] for r in avail}
            self.assertIn('allcause', endpoints)
            self.assertIn('e3_tb_death', endpoints)
            ac = next(r for r in avail if r['endpoint'] == 'allcause'
                      and r['group'] == 'elderly_65p')
            e3 = next(r for r in avail if r['endpoint'] == 'e3_tb_death'
                      and r['group'] == 'elderly_65p')
            # 全因老年事件率 > E3（含非 TB 死亡的语义滑动）
            self.assertGreater(ac['event_rate'], e3['event_rate'])
            self.assertAlmostEqual(de['elderly_non_tb_share'], 0.63)
            self.assertIn('非 TB', de['paper_note'])
            self.assertIn('E3', de['decision_rule'])
        # P1-2 本地重校准最小样本量（round-13 v4：四臂+裁定成本+网格外推）
        rc = result['transfer_recalibration']
        if rc:
            vnm = next(c for c in rc['cohorts']
                       if 'Vietnam' in c['cohort'])
            self.assertLess(vnm['zero_shot_auroc'], 0.5)
            self.assertEqual(vnm['direction_truth'], 'reversed')
            # v3 stack / v4 先验 / v1 翻序：反向站 N=100 恢复
            self.assertEqual(vnm['n_to_recover_stack'], 100)
            self.assertEqual(vnm['n_to_recover_stack_prior'], 100)
            self.assertEqual(vnm['n_to_recover_score_v1'], 100)
            r100 = vnm['curve'][0]
            self.assertGreater(r100['stack_median'],
                               1 - vnm['zero_shot_auroc'] - 0.02)
            # 强翻转站方向裁定：N≥200 零错误
            self.assertLessEqual(vnm['curve'][-1]['ruling_error_rate'],
                                 0.05)
            ken = next(c for c in rc['cohorts'] if 'Kenya' in c['cohort'])
            self.assertEqual(ken['direction_truth'], 'aligned')
            # round-13 网格外推：对齐站恢复 N=2000（≈11 阳性），
            # v1 单调变换恒不恢复（设计产物）
            self.assertEqual(ken['n_to_recover_stack'], 2000)
            self.assertEqual(ken['n_to_recover_features'], 2000)
            self.assertIsNone(ken['n_to_recover_score_v1'])
            # 事件充足（≈53 阳性）：v3 超全量本地
            last = ken['curve'][-1]
            self.assertEqual(last['n_local'], 10000)
            self.assertGreater(last['stack_median'],
                               ken['full_local_cv_auroc'])
            # 低基率站方向裁定：小 N≈抛硬币、大 N 归零（事件数规则）
            self.assertGreaterEqual(ken['curve'][0]['ruling_error_rate'],
                                    0.2)
            self.assertEqual(ken['curve'][-1]['ruling_error_rate'], 0.0)
            ns = [p['n_local'] for p in vnm['curve']]
            self.assertEqual(ns, sorted(ns))       # N 档位递增
            self.assertIn('阳性事件数', rc['sample_size_rule'])
        # P1-3 细菌学头 v2（症状列裁定 + HIV/职业扩展）
        bv = result['bact_head_v2']
        if bv:
            self.assertEqual(bv['symptom_feasibility']['verdict'],
                             'INFEASIBLE')
            self.assertGreater(bv['v2_death_auroc'], bv['v1_death_auroc'])
            self.assertGreater(bv['v2_gain']['death_auroc'], 0)
            self.assertGreater(bv['v2_bact_auprc'], 0.7)
            self.assertIn('分诊工具', bv['positioning'])
        # P2-5 迁移红线
        trf = result['cross_population_transfer']
        if trf:
            rl = trf['deployment_red_line']
            self.assertIn('禁止零样本部署', rl)
            self.assertIn('Vietnam', rl)
            self.assertIn('本地重校准', rl)
        # P3-6 时序先证封存（无条件子包）
        tp = result['temporal_prior_sealed']
        self.assertIn('分组粒度受限', tp['conclusion'])
        self.assertIn('ICC', tp['evidence'])
        self.assertIn('ERASE-TB', tp['implication'])

    def test_deployment_metrics_missing_archive(self):
        """主归档缺失时优雅降级（不抛异常）。"""
        orig = me.DEPLOY_ARCHIVE
        try:
            me.DEPLOY_ARCHIVE = 'no_such_deployment.json'
            result = me.collect_deployment_metrics()
            self.assertFalse(result.get('available'))
            self.assertIn('reason', result)
        finally:
            me.DEPLOY_ARCHIVE = orig


class TestExtractThreeLayerSeries(unittest.TestCase):
    """三层分解提取的防御性行为。"""

    def test_none_input(self):
        result = me.extract_three_layer_series(None)
        self.assertFalse(result.get('available'))
        self.assertIn('reason', result)

    def test_empty_ml_results(self):
        result = me.extract_three_layer_series({})
        self.assertFalse(result.get('available'))

    def test_valid_structure(self):
        ml_results = {
            'ensemble': {
                'family': [{
                    'name': '张三',
                    'traditional_prob': 20.0,
                    'ensemble_predictions': {
                        'three_layer': {
                            'architecture': 'three_layer',
                            'layers': {
                                'layer1_individual': {'p_base_percent': 18.5},
                                'layer2_network': {
                                    'network_risk': 25.0,
                                    'network_increment': 6.5,
                                    'network_aware': True,
                                },
                                'layer3_intervention': None,
                            },
                            'decision': {
                                'gating': 0.8,
                                'gated_network_increment': 5.2,
                                'combined_probability': 23.7,
                                'intervention_benefit': 0.0,
                                'decision_risk': 20.4,
                                'decision_class': 0,
                            },
                            'summary': {
                                'best_strategy': '预防性治疗',
                                'best_averted_percent': 12.0,
                            },
                        },
                    },
                }],
                'social': [],
            },
        }
        result = me.extract_three_layer_series(ml_results)
        self.assertTrue(result.get('available'))
        self.assertEqual(result['n'], 1)
        c = result['contacts'][0]
        self.assertEqual(c['name'], '张三')
        self.assertAlmostEqual(c['p_base'], 18.5)
        self.assertAlmostEqual(c['network_increment'], 6.5)
        self.assertAlmostEqual(c['gating'], 0.8)
        self.assertAlmostEqual(c['combined'], 23.7)
        self.assertTrue(c['network_aware'])
        self.assertIn('raw', c)

    def test_non_three_layer_architecture_skipped(self):
        ml_results = {
            'ensemble': {
                'family': [{
                    'name': '李四',
                    'ensemble_predictions': {
                        'three_layer': {'architecture': 'unavailable'},
                    },
                }],
                'social': [],
            },
        }
        result = me.extract_three_layer_series(ml_results)
        self.assertFalse(result.get('available'))


class TestTaskAMetricsAtThreshold(unittest.TestCase):
    """阈值指标纯函数。"""

    def test_perfect_separation(self):
        scores = [90.0, 80.0, 10.0, 5.0]
        labels = [1, 1, 0, 0]
        m = task_a_metrics_at_threshold(scores, labels, 50.0)
        self.assertEqual(m['tp'], 2)
        self.assertEqual(m['tn'], 2)
        self.assertEqual(m['fp'], 0)
        self.assertEqual(m['fn'], 0)
        self.assertAlmostEqual(m['sensitivity'], 1.0)
        self.assertAlmostEqual(m['specificity'], 1.0)
        self.assertAlmostEqual(m['youden'], 1.0)

    def test_all_positive_threshold(self):
        scores = [90.0, 80.0]
        labels = [1, 0]
        m = task_a_metrics_at_threshold(scores, labels, 1.0)
        self.assertEqual(m['tp'], 1)
        self.assertEqual(m['fp'], 1)
        self.assertEqual(m['tn'], 0)
        self.assertEqual(m['fn'], 0)
        self.assertAlmostEqual(m['ppv'], 0.5)
        self.assertAlmostEqual(m['npv'], 0.0)

    def test_all_negative_threshold(self):
        scores = [90.0, 80.0]
        labels = [1, 0]
        m = task_a_metrics_at_threshold(scores, labels, 99.0)
        self.assertEqual(m['tp'], 0)
        self.assertEqual(m['fn'], 1)
        self.assertAlmostEqual(m['sensitivity'], 0.0)
        self.assertAlmostEqual(m['specificity'], 1.0)

    def test_empty_input(self):
        m = task_a_metrics_at_threshold([], [], 50.0)
        self.assertEqual(m['tp'], 0)
        self.assertAlmostEqual(m['sensitivity'], 0.0)


class TestKaplanMeier(unittest.TestCase):
    """Kaplan-Meier 纯函数。"""

    def test_empty(self):
        t, s = kaplan_meier([], [])
        self.assertEqual(t, [0.0])
        self.assertEqual(s, [1.0])

    def test_no_events(self):
        t, s = kaplan_meier([10.0, 20.0, 30.0], [0, 0, 0])
        self.assertAlmostEqual(s[-1], 1.0)

    def test_all_events(self):
        t, s = kaplan_meier([10.0, 20.0], [1, 1])
        # 两个事件依次发生：1/2 × 1/1 → 生存率 0
        self.assertAlmostEqual(s[-1], 0.0)

    def test_monotone_non_increasing(self):
        times = [5.0, 10.0, 15.0, 20.0, 25.0]
        events = [1, 0, 1, 1, 0]
        _, s = kaplan_meier(times, events)
        for i in range(1, len(s)):
            self.assertLessEqual(s[i], s[i - 1] + 1e-12)

    def test_starts_at_one(self):
        t, s = kaplan_meier([1.0], [1])
        self.assertEqual(t[0], 0.0)
        self.assertEqual(s[0], 1.0)


# ==============================================================================
# 时序家庭筛查演示输入（第四轮 P1c：GUI 第 9 个子标签页的数据源）
# ==============================================================================

class TestHouseholdScreeningDemo(unittest.TestCase):
    """build_household_screening_demo / household_screening_signature 纯函数。"""

    @staticmethod
    def _entries():
        return [
            {'name': '父亲', 'age': 68, 'diagnosed': True},
            {'name': '母亲', 'age': 45, 'diagnosed': False},
            {'name': '女儿', 'age': 8, 'diagnosed': False},
        ]

    def test_unavailable_when_no_entries(self):
        demo = me.build_household_screening_demo([])
        self.assertFalse(demo['available'])
        self.assertIn('家庭成员', demo['reason'])

    def test_names_scores_bounds_and_preseed(self):
        demo = me.build_household_screening_demo(self._entries())
        self.assertTrue(demo['available'])
        self.assertEqual(demo['names'], ['父亲', '母亲', '女儿'])
        self.assertEqual(len(demo['base_scores']), 3)
        for s in demo['base_scores']:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 100.0)
        self.assertEqual(demo['preseed_positive'], [0])
        self.assertEqual(demo['source'], 'entry_risk')
        self.assertAlmostEqual(demo['base_rate'], me.DEFAULT_DEMO_BASE_RATE)

    def test_p_base_map_priority(self):
        demo = me.build_household_screening_demo(
            self._entries(), p_base_map={'母亲': 55.5})
        self.assertEqual(demo['source'], 'mixed')
        idx = demo['names'].index('母亲')
        self.assertAlmostEqual(demo['base_scores'][idx], 55.5)
        full = me.build_household_screening_demo(
            self._entries(),
            p_base_map={'父亲': 30.0, '母亲': 40.0, '女儿': 10.0})
        self.assertEqual(full['source'], 'ml')
        self.assertEqual(full['base_scores'], [30.0, 40.0, 10.0])

    def test_fallback_risk_score_mirrors_family_tab(self):
        """回退评分与 gui/tabs/_family.py _compute_risk_score 同口径。"""
        entry = {'name': '伯伯', 'age': 3, 'ventilation': 1,
                 'contact_distance': '极近', 'exposure_setting': '拥挤',
                 'is_high_risk': '是', 'has_symptoms': '是', 'has_tb': '是',
                 'cumulative_exposure': 2000}
        demo = me.build_household_screening_demo([entry])
        # 15(年龄) + 15(通风) + 15(距离) + 15(环境) + 20(高危) + 10(症状)
        # + 10(既往结核) + 10(累积暴露) = 110 → 截断为 100
        self.assertAlmostEqual(demo['base_scores'][0], 100.0)

    def test_signature_detects_changes(self):
        entries = self._entries()
        s1 = me.household_screening_signature(entries)
        self.assertEqual(s1, me.household_screening_signature(list(entries)))
        entries[1]['diagnosed'] = True
        self.assertNotEqual(s1, me.household_screening_signature(entries))
        s3 = me.household_screening_signature(self._entries(),
                                              {'父亲': 30.0})
        self.assertNotEqual(s1, s3)
        # 无 ML 基线时签名的 ML 分量为空元组（不含 None）
        self.assertEqual(s1[2], ())


class TestBuildScreeningState(unittest.TestCase):
    """build_screening_state：预置确诊成员 + 其余成员上调重排。"""

    def test_none_when_unavailable(self):
        self.assertIsNone(build_screening_state({'available': False}))
        self.assertIsNone(build_screening_state(None))

    def test_preseed_updates_remaining_members(self):
        demo = me.build_household_screening_demo([
            {'name': '甲', 'diagnosed': True},
            {'name': '乙', 'diagnosed': False},
            {'name': '丙', 'diagnosed': False},
        ], p_base_map={'甲': 40.0, '乙': 30.0, '丙': 20.0})
        state = build_screening_state(demo)
        self.assertIsNotNone(state)
        # 甲已筛（冻结为基线），结果阳性
        self.assertTrue(state.screened_flags[0])
        self.assertEqual(state.results[0], 1)
        # 乙/丙未筛但风险上调（阳性先证 → 收缩户级率 > π）
        scores = state.current_scores()
        self.assertGreater(scores[1], 30.0)
        self.assertGreater(scores[2], 20.0)
        # 下一步建议：更新后风险最高的未筛成员（乙）
        idx = state.next_to_screen()
        self.assertEqual(state.names[idx], '乙')

    def test_no_preseed_first_screener_falls_back_to_base(self):
        demo = me.build_household_screening_demo([
            {'name': '甲', 'diagnosed': False},
            {'name': '乙', 'diagnosed': False},
        ], p_base_map={'甲': 25.0, '乙': 60.0})
        state = build_screening_state(demo)
        # 无先证 → 首筛查者 = 个体基线最高者（highest_risk 策略）
        self.assertEqual(state.names[state.next_to_screen()], '乙')
        self.assertEqual(state.current_scores(), [25.0, 60.0])


class TestTemporalPanelStateLogic(unittest.TestCase):
    """Mixin 状态管理逻辑（不依赖真实 Tk 控件）。"""

    @staticmethod
    def _make_mixin(entries):
        from tb_risk.gui.results_panel.model_evidence import ModelEvidenceMixin
        m = ModelEvidenceMixin.__new__(ModelEvidenceMixin)
        m._me_cache = {}
        m.family_entries = entries
        return m

    def test_state_reused_until_signature_changes(self):
        m = self._make_mixin([{'name': '甲', 'diagnosed': False},
                              {'name': '乙'}])
        state1, demo1 = m._me_temporal_state()
        self.assertIsNotNone(state1)
        self.assertTrue(demo1['available'])
        state_again, _ = m._me_temporal_state()
        self.assertIs(state1, state_again)
        # 确诊标记切换 → 签名变化 → 重建并预置阳性
        m.family_entries[0]['diagnosed'] = True
        state2, _ = m._me_temporal_state()
        self.assertIsNot(state1, state2)
        self.assertTrue(state2.screened_flags[0])
        self.assertEqual(state2.results[0], 1)

    def test_ml_p_base_map_used_when_cache_ready(self):
        m = self._make_mixin([{'name': '甲', 'diagnosed': False}])
        m._me_cache['three_layer'] = {
            'available': True,
            'contacts': [
                {'name': '甲', 'contact_type': 'family', 'p_base': 33.3},
                {'name': '乙(社会)', 'contact_type': 'social', 'p_base': 99.0},
            ],
        }
        state, demo = m._me_temporal_state()
        self.assertEqual(demo['source'], 'ml')
        self.assertAlmostEqual(demo['base_scores'][0], 33.3)
        # 社会接触者不进入家庭筛查面板
        self.assertEqual(len(demo['base_scores']), 1)

    def test_empty_entries_returns_none_with_reason(self):
        m = self._make_mixin([])
        state, demo = m._me_temporal_state()
        self.assertIsNone(state)
        self.assertFalse(demo['available'])
        self.assertIn('家庭成员', demo['reason'])


class TestBuildTemporalTabGui(unittest.TestCase):
    """_build_temporal_household_tab UI 创建（需要真实 Tk root，无 display 时跳过）。"""

    @classmethod
    def setUpClass(cls):
        try:
            import tkinter as tk
            cls._root = tk.Tk()
            cls._root.withdraw()
        except Exception:
            cls._root = None

    @classmethod
    def tearDownClass(cls):
        if cls._root is not None:
            try:
                cls._root.destroy()
            except Exception:
                pass

    def setUp(self):
        if self._root is None:
            self.skipTest("Tkinter display not available")

    def _make_mixin(self):
        from tb_risk.gui.results_panel.model_evidence import ModelEvidenceMixin
        m = ModelEvidenceMixin.__new__(ModelEvidenceMixin)
        m._me_cache = {}
        m.family_entries = [{'name': '甲', 'diagnosed': True},
                            {'name': '乙', 'diagnosed': False}]
        m._get_chart_figsize = lambda w=11, h=6: (w, h)
        m._bind_configure_redraw = lambda *a, **k: None
        m._register_text_widget = lambda w: None
        return m

    def test_build_creates_tab_and_controls(self):
        import tkinter.ttk as ttk
        from tb_risk.gui.results_panel.model_evidence import MATPLOTLIB_AVAILABLE
        m = self._make_mixin()
        nb = ttk.Notebook(self._root)
        m._build_temporal_household_tab(nb)
        self.assertTrue(hasattr(m, 'me_temporal_info'))
        tab_names = [nb.tab(t, 'text') for t in nb.tabs()]
        self.assertIn('时序家庭筛查', tab_names)
        if MATPLOTLIB_AVAILABLE:
            self.assertTrue(hasattr(m, 'me_temporal_figure'))
            self.assertTrue(hasattr(m, 'me_temporal_canvas'))


class TestBuildScaleValidationTabGui(unittest.TestCase):
    """_build_scale_validation_tab UI 创建（第 10 个子标签页）。"""

    @classmethod
    def setUpClass(cls):
        try:
            import tkinter as tk
            cls._root = tk.Tk()
            cls._root.withdraw()
        except Exception:
            cls._root = None

    @classmethod
    def tearDownClass(cls):
        if cls._root is not None:
            try:
                cls._root.destroy()
            except Exception:
                pass

    def setUp(self):
        if self._root is None:
            self.skipTest("Tkinter display not available")

    def _make_mixin(self):
        from tb_risk.gui.results_panel.model_evidence import ModelEvidenceMixin
        m = ModelEvidenceMixin.__new__(ModelEvidenceMixin)
        m._me_cache = {}
        m._get_chart_figsize = lambda w=11, h=6: (w, h)
        m._bind_configure_redraw = lambda *a, **k: None
        m._register_text_widget = lambda w: None
        return m

    def test_build_creates_tab_and_controls(self):
        import tkinter.ttk as ttk
        from tb_risk.gui.results_panel.model_evidence import MATPLOTLIB_AVAILABLE
        m = self._make_mixin()
        nb = ttk.Notebook(self._root)
        m._build_scale_validation_tab(nb)
        self.assertTrue(hasattr(m, 'me_scale_summary'))
        self.assertTrue(hasattr(m, 'me_scale_detail'))
        tab_names = [nb.tab(t, 'text') for t in nb.tabs()]
        self.assertIn('百万级验证', tab_names)
        if MATPLOTLIB_AVAILABLE:
            self.assertTrue(hasattr(m, 'me_scale_figure'))
            self.assertTrue(hasattr(m, 'me_scale_canvas'))

    def test_update_ui_with_unavailable_evidence(self):
        """归档不可用时摘要显示原因（不抛异常）。"""
        m = self._make_mixin()
        m._me_cache['sinan'] = {'available': False, 'reason': '归档不存在'}
        import tkinter.ttk as ttk
        nb = ttk.Notebook(self._root)
        m._build_scale_validation_tab(nb)
        m._update_scale_validation_ui()
        self.assertIn('归档不存在', m.me_scale_summary.cget('text'))

    def test_update_ui_with_available_evidence(self):
        """归档可用时渲染叙事与详情文本。"""
        m = self._make_mixin()
        m._me_cache['sinan'] = {
            'available': True,
            'narrative': 'SINAN 124.6 万个体验证叙事',
            't1_series': {'LR': [0.83, 0.83, 0.83]},
            't2_series': {'LR': 0.845},
            'best_model': 'LR', 'decay_val_to_far': 0.012,
            't2_best_model': 'LR',
            'subgroups': {'age_65p': {'auroc': 0.67, 'n': 100, 'events': 20}},
            'top_features': ['AGRAVAIDS_1'],
            'pakistan_interpretation': 'directional_reference_only',
        }
        import tkinter.ttk as ttk
        nb = ttk.Notebook(self._root)
        m._build_scale_validation_tab(nb)
        m._update_scale_validation_ui()
        self.assertIn('124.6 万', m.me_scale_summary.cget('text'))
        detail = m.me_scale_detail.get('1.0', 'end')
        self.assertIn('age_65p', detail)
        self.assertIn('AGRAVAIDS_1', detail)


if __name__ == '__main__':
    unittest.main()
