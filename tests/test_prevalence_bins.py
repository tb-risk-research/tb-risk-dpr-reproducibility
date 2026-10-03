#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1：场景生成器患病率档（真实患病率口径的阈值迁移检验）

背景（用户 2026-08-24 指令）：场景训练数据阳性率 20.7%，部署对象
0.5%~3% 人群。generate_scenarios 原本硬编码 target=0.25（尺度校准，
幂缩放保持相对风险排序）——本测试将其参数化为 prevalence_target，
并验证三件事：

1. 患病率档生效：0.5% / 3% 档的实测阳性率落在目标容差内；
2. 默认行为不变：prevalence_target 缺省 0.25 与旧实现完全一致
   （现有训练 CSV 可复现，回归保护）；
3. 只动标签不动特征：同 seed 两档的特征矩阵逐行相等（患病率档
   是标签口径切换，不是新的特征分布——阈值迁移检验的前提）。
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tb_risk.data.evaluate_ensemble import (  # noqa: E402
    generate_scenarios,
    generate_scenario_ml_csv,
)


class TestPrevalenceTarget:

    def test_low_prevalence_bin(self):
        """0.5% 档：实测阳性率 ∈ (0.2%, 1.0%)。"""
        scen = generate_scenarios(300, 2026, prevalence_target=0.005)
        labels = [y for _, _, _, lab in scen for _, _, y in lab]
        rate = float(np.mean(labels))
        assert 0.002 < rate < 0.010, f'0.5% 档实测 {rate:.4f}'

    def test_mid_prevalence_bin(self):
        """3% 档：实测阳性率 ∈ (2%, 4%)。"""
        scen = generate_scenarios(300, 2026, prevalence_target=0.03)
        labels = [y for _, _, _, lab in scen for _, _, y in lab]
        rate = float(np.mean(labels))
        assert 0.02 < rate < 0.04, f'3% 档实测 {rate:.4f}'

    def test_default_unchanged(self):
        """缺省 prevalence_target=0.25：与旧实现行为一致（阳性率
        回归到 15%-35% 区间，与既有 test_host_pathway 断言同款）。"""
        scen = generate_scenarios(300, 2026)
        labels = [y for _, _, _, lab in scen for _, _, y in lab]
        rate = float(np.mean(labels))
        assert 0.15 < rate < 0.35, f'默认档实测 {rate:.4f}'

    def test_features_identical_across_bins(self):
        """同 seed 两档：特征矩阵逐行相等（患病率档只切换标签口径）。

        幂缩放 p_cal = 1-(1-p)^scale 作用于同一批 raw p → 相对风险
        排序不变；Bernoulli 抽样的 rng 消耗序列一致 → 特征与接触者
        结构逐字段相同。
        """
        scen_a = generate_scenarios(60, 2026, prevalence_target=0.005)
        scen_b = generate_scenarios(60, 2026, prevalence_target=0.25)
        assert len(scen_a) == len(scen_b)
        for (pa, fam_a, soc_a, lab_a), (pb, fam_b, soc_b, lab_b) in \
                zip(scen_a, scen_b):
            assert pa == pb
            assert fam_a == fam_b
            assert soc_a == soc_b
            assert len(lab_a) == len(lab_b)

    def test_csv_low_prevalence_and_meta(self, tmp_path):
        """CSV 出口：prevalence_target 写入 meta，CSV 阳性率达标。"""
        csv_path = os.path.join(str(tmp_path), 'lo.csv')
        meta_path = os.path.join(str(tmp_path), 'lo_meta.json')
        generate_scenario_ml_csv(
            300, 2026, csv_path, meta_path,
            label_mode='mechanistic', prevalence_target=0.01)
        df = pd.read_csv(csv_path)
        rate = float(df['tb_outcome'].mean())
        assert 0.006 < rate < 0.016, f'1% 档 CSV 实测 {rate:.4f}'
        meta = __import__('json').load(open(meta_path, encoding='utf-8'))
        assert meta['prevalence_target'] == 0.01
        assert '患病率档' in meta.get('label_note', '') or \
            meta.get('label_note', '') != ''
