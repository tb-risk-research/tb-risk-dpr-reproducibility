#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第一优先：打通宿主因素通路（死通路 → 文献校准的多通路 DGP）。

背景（2026-08-24 特征审计的延伸发现，两个独立成因）：
1. 结构性零值：场景训练 CSV 六列全零（has_tb / is_high_risk /
   dm_tb_synergy / symptom_delay / cough_contact / highrisk_comorbid）
   ——_sample_contact 硬编码 has_tb=0、缺 is_high_risk 键；
   generate_scenario_ml_csv 丢弃 patient 上下文（delay_days / cough_freq /
   contact_count），交互特征全部退化为 0。
2. 标签机制只走暴露通路：_contact_susceptibility 各因子变异幅度过小
   （总范围 0.9~3.5，多数样本集中 0.9~1.4），在乘积-指数结构
   p=1-exp(-I*E*S) 中被跨数量级的暴露强度 E 完全淹没，且
   has_symptoms 根本不参与——宿主特征对标签的单变量 AUROC≈0.50，
   模型只能学成"暴露单通路模型"。

修复完成标准（用户定义，可验证）：
- has_symptoms / age / past_illness 的单变量 AUROC 从 0.50 升到 0.55+；
- 六个结构性零值列全部复活（唯一值数 ≥ 2 且存在非零观测）；
- 阳性率校准（~25%）与同种子可复现性保持。

文献（易感性各因子校准依据）：
- Martinez L et al. Lancet 2020;395:973-984（137,647 名儿童接触者，
  暴露后 2 年进展风险随年龄 U 型：<5 岁 7.6%、5-9 岁 5.2%、10-14 岁 5.6%）
- Seddon JA & Shingadia D. Infect Drug Resist 2014;7:153-165
  （感染后进展：婴儿 ~50%、1-2 岁 20-30%、5-10 岁仅 ~2%、成人 ~5%）
- Jeon CY & Murray MB. PLoS Med 2008;5(7):e152（糖尿病 TB RR=3.11，
  95%CI 2.27-4.26）；Cochrane 2024 CD016013（RR 1.5-2.4）
- Colditz GA et al. JAMA 1995（BCG 总体效力 ~50%）；
  Abubakar I et al. Health Technol Assess 2013;17.37（保护持续 10-15 年后衰减）
- WHO Global TB Report（PLHIV TB 风险约 16-27 倍）
- Fox GJ et al. PLoS Med 2013（接触者调查与症状筛查）
"""

import json
import os
import sys
import tempfile
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False


def _load_ee():
    import evaluate_ensemble as ee
    return ee


def _auc(y, x):
    """单变量 AUROC（Mann-Whitney U，并列取平均秩）。"""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    pos, neg = x[y == 1], x[y == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float('nan')
    all_x = np.concatenate([pos, neg])
    order = np.argsort(all_x, kind='mergesort')
    sorted_x = all_x[order]
    ranks_sorted = np.empty(len(all_x), dtype=float)
    i = 0
    while i < len(all_x):
        j = i
        while j + 1 < len(all_x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks_sorted[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks = np.empty(len(all_x), dtype=float)
    ranks[order] = ranks_sorted
    r1 = ranks[:n1].sum()
    return float((r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _skip_if_no_numpy(cls):
    return unittest.skipUnless(_NUMPY, "需要 numpy")(cls)


@_skip_if_no_numpy
class TestSusceptibilityHostPathway(unittest.TestCase):
    """易感性项的文献校准（机制标签宿主通路）。"""

    def _contact(self, **over):
        c = {'age': 30, 'bcg_vaccine': 1, 'past_illness': 0,
             'past_illness_type': 'none', 'has_symptoms': 0,
             'has_tb': 0, 'is_high_risk': 0}
        c.update(over)
        return c

    def test_age_u_shape_curve(self):
        """年龄 U 型进展曲线：两端高、5-14 岁最低（Martinez 2020）。"""
        ee = _load_ee()
        s = ee._contact_susceptibility
        infant = s(self._contact(age=3))
        golden = s(self._contact(age=10))
        adult = s(self._contact(age=25))
        middle = s(self._contact(age=50))
        elder = s(self._contact(age=70))
        self.assertGreater(infant, adult, "婴幼儿进展风险应高于青壮年")
        self.assertGreater(elder, adult, "老年免疫衰退风险应高于青壮年")
        self.assertLess(golden, adult, "5-14 岁（黄金年龄段）应为最低")
        self.assertGreater(elder, middle, "老年风险应高于中年")

    def test_symptoms_multiplier(self):
        """接触者已有症状 → 进展风险放大 3~5 倍（Fox 2013：有症状
        接触者活动性 TB 检出率 ~10-30% vs 无症状 1-4%，RR 5-10 取保守）。"""
        ee = _load_ee()
        s = ee._contact_susceptibility
        ratio = (s(self._contact(has_symptoms=1))
                 / s(self._contact(has_symptoms=0)))
        self.assertGreaterEqual(ratio, 3.0)
        self.assertLessEqual(ratio, 5.0)

    def test_comorbidity_ranking_and_dm_calibration(self):
        """共病进展放大排序 + 糖尿病 RR 落在文献区间。"""
        ee = _load_ee()
        s = ee._contact_susceptibility
        base = s(self._contact())
        ratios = {t: s(self._contact(past_illness=1,
                                     past_illness_type=t)) / base
                  for t in ('other', 'diabetes', 'immunosuppressants', 'hiv')}
        # 排序：hiv > immunosuppressants > diabetes > other > 1
        self.assertGreater(ratios['hiv'], ratios['immunosuppressants'])
        self.assertGreater(ratios['immunosuppressants'], ratios['diabetes'])
        self.assertGreater(ratios['diabetes'], ratios['other'])
        self.assertGreater(ratios['other'], 1.0)
        # 糖尿病：Jeon 2008 RR=3.11（95%CI 2.27-4.26）；Cochrane 2024 1.5-2.4
        self.assertGreaterEqual(ratios['diabetes'], 2.0)
        self.assertLessEqual(ratios['diabetes'], 3.2)
        # HIV：WHO 16-27 倍，取保守下限（≥8）
        self.assertGreaterEqual(ratios['hiv'], 8.0)

    def test_bcg_waning_with_age(self):
        """BCG 保护随接种后年限衰减（Colditz 1995 + Abubakar 2013）。"""
        ee = _load_ee()
        s = ee._contact_susceptibility
        rr_young = (s(self._contact(age=5, bcg_vaccine=0))
                    / s(self._contact(age=5, bcg_vaccine=1)))
        rr_old = (s(self._contact(age=70, bcg_vaccine=0))
                  / s(self._contact(age=70, bcg_vaccine=1)))
        self.assertGreater(rr_young, rr_old,
                           "未接种相对风险应随年龄（接种后年限）衰减")
        self.assertGreaterEqual(rr_young, 1.6,
                                "儿童期未接种相对风险应显著（保护 ~50%）")
        self.assertLessEqual(rr_old, 1.3,
                             "老年期 BCG 保护基本衰减殆尽")

    def test_prior_tb_and_high_risk_multipliers(self):
        """既往结核史（再激活）与高危人群标记的乘数。"""
        ee = _load_ee()
        s = ee._contact_susceptibility
        base = s(self._contact())
        tb_rr = s(self._contact(has_tb=1)) / base
        hr_rr = s(self._contact(is_high_risk=1)) / base
        # 既往 TB 史再激活（DGP TRANSMISSION_SPEC prior_tb_multiplier=2.0 同源）
        self.assertAlmostEqual(tb_rr, 2.0, places=5)
        # 高危人群（矽肺/透析等，1.5~3 倍取保守下限）
        self.assertAlmostEqual(hr_rr, 1.5, places=5)


@_skip_if_no_numpy
class TestContactSamplingHostFields(unittest.TestCase):
    """场景生成器接触者字典携带症状/基础病字段，修结构性零值。"""

    def test_host_fields_present_and_vary(self):
        ee = _load_ee()
        rng = np.random.default_rng(7)
        contacts = [ee._sample_contact(rng, i, 'family')
                    for i in range(400)]
        for key in ('has_tb', 'is_high_risk', 'has_symptoms',
                    'past_illness'):
            vals = {c[key] for c in contacts}
            self.assertEqual(len(vals), 2, f"{key} 应有 0/1 变异")
        # 既往结核史 ~2%（DGP FIELD_SPECS Bernoulli(0.02)）
        tb_rate = float(np.mean([c['has_tb'] for c in contacts]))
        self.assertGreater(tb_rate, 0.0)
        self.assertLess(tb_rate, 0.06)
        # 高危人群标记 ~10%
        hr_rate = float(np.mean([c['is_high_risk'] for c in contacts]))
        self.assertGreater(hr_rate, 0.04)
        self.assertLess(hr_rate, 0.18)


@_skip_if_no_numpy
class TestMechanisticLabelHostSignal(unittest.TestCase):
    """完成标准：宿主特征单变量 AUROC 从 0.50 升到 0.55+。"""

    def test_host_univariate_auroc(self):
        ee = _load_ee()
        scenarios = ee.generate_scenarios(1500, seed=2026)
        rows = []
        for _pat, _fam, _soc, labels in scenarios:
            for _ctype, c, y in labels:
                rows.append((c, int(y)))
        y = np.array([r[1] for r in rows])
        # 阳性率校准保持（~25%）
        self.assertGreaterEqual(y.mean(), 0.15)
        self.assertLessEqual(y.mean(), 0.35)

        feat_vals = {
            'has_symptoms': [c['has_symptoms'] for c, _ in rows],
            'age': [c['age'] for c, _ in rows],
            'past_illness': [c['past_illness'] for c, _ in rows],
            'bcg_vaccine': [c['bcg_vaccine'] for c, _ in rows],
        }
        aurocs = {name: _auc(y, vals)
                  for name, vals in feat_vals.items()}
        # 用户完成标准（0.55+）：has_symptoms / age / past_illness
        self.assertGreaterEqual(
            aurocs['has_symptoms'], 0.55,
            f"has_symptoms AUROC={aurocs['has_symptoms']:.4f} < 0.55")
        self.assertGreaterEqual(
            aurocs['age'], 0.55,
            f"age AUROC={aurocs['age']:.4f} < 0.55")
        self.assertGreaterEqual(
            aurocs['past_illness'], 0.55,
            f"past_illness AUROC={aurocs['past_illness']:.4f} < 0.55")
        # bcg_vaccine：保护随年龄衰减后整体信号弱于前三者，
        # 但方向应为保护性（AUROC<0.5 表示接种疫苗降风险）
        self.assertLess(aurocs['bcg_vaccine'], 0.50)


@_skip_if_no_numpy
class TestScenarioCSVDeadFeatureRevival(unittest.TestCase):
    """六个结构性零值列复活 + patient 上下文接入 + 可复现。"""

    _DEAD_COLS = ['has_tb', 'is_high_risk', 'dm_tb_synergy',
                  'symptom_delay', 'cough_contact', 'highrisk_comorbid']

    def test_dead_columns_revived_and_reproducible(self):
        ee = _load_ee()
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            csv1 = os.path.join(tmp, 'a.csv')
            meta1 = os.path.join(tmp, 'a.json')
            ee.generate_scenario_ml_csv(200, 2026, csv1, meta1)
            df1 = pd.read_csv(csv1)
            self.assertGreater(len(df1), 800)
            for col in self._DEAD_COLS:
                self.assertGreaterEqual(
                    df1[col].nunique(), 2, f"{col} 仍为死列")
                self.assertGreater(
                    float((df1[col] > 0).mean()), 0.0,
                    f"{col} 应存在非零观测")
            # 阳性率校准保持
            pr = float(df1['tb_outcome'].mean())
            self.assertGreaterEqual(pr, 0.15)
            self.assertLessEqual(pr, 0.35)
            # meta 文档化宿主通路（含文献引用）
            with open(meta1, encoding='utf-8') as fh:
                meta = json.load(fh)
            self.assertIn('host_pathway', meta)
            self.assertIn('references', meta['host_pathway'])
            # 同种子可复现
            csv2 = os.path.join(tmp, 'b.csv')
            meta2 = os.path.join(tmp, 'b.json')
            ee.generate_scenario_ml_csv(200, 2026, csv2, meta2)
            df2 = pd.read_csv(csv2)
            pd.testing.assert_frame_equal(df1, df2)

    def test_clinical_label_mode_also_revived(self):
        """clinical 标签模式共享同一接触者字典 → 死列同样复活。"""
        ee = _load_ee()
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            csv = os.path.join(tmp, 'c.csv')
            meta = os.path.join(tmp, 'c.json')
            ee.generate_scenario_ml_csv(150, 2026, csv, meta,
                                        label_mode='clinical')
            df = pd.read_csv(csv)
            for col in self._DEAD_COLS:
                self.assertGreaterEqual(df[col].nunique(), 2,
                                        f"{col}（clinical 模式）仍为死列")


if __name__ == '__main__':
    unittest.main()
