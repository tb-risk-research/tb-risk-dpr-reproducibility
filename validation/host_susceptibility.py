#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""宿主易感性参数的单一真值源（P3，DGP v3 统一，2026-08-24）。

背景：宿主通路修复（2026-08-24）把 mechanistic 标签的易感性项改写为
文献校准的复合宿主因子，但实现只存在于
``data/evaluate_ensemble.py::_contact_susceptibility``——
``validation/layer_ablation.py`` 的合成网络与 ``validation/dgp.py`` 的
DGP 文档仍是旧机制，代码库至少两套标签机制并存。

本模块把宿主易感性参数提取为**唯一权威定义**：
- ``HOST_SUSCEPTIBILITY_SPEC``：结构化参数规范（版本号 + 文献校准
  参数 + 引用），dgp.py 的标签机制登记表引用它（不复制）；
- ``host_susceptibility_multiplier``：数值实现——
  ``evaluate_ensemble._contact_susceptibility`` 委托本函数（行为完全
  一致，由 tests/test_host_susceptibility_source.py 随机一致性测试
  守护），layer_ablation 的合成网络标签同样引用。

标签机制版本：``v3-host-pathway``（暴露+宿主双通路；v2 为暴露单通路
——宿主特征对标签 AUROC≈0.50 的旧代）。
"""

import numpy as np

LABEL_MECHANISM_VERSION = 'v3-host-pathway'

# 年龄 U 型进展相对风险（×参考段 15-35 岁 = 1.0）
AGE_PROGRESSION_RR = {
    '<5': 2.0,     # 婴幼儿感染后进展 ~50% vs 成人 ~5%
    '5-14': 0.4,   # "黄金年龄段"最低（5-10 岁 ~2%）
    '15-35': 1.0,  # 参考段
    '35-65': 1.2,  # 中年缓升
    '>65': 2.2,    # 免疫衰退
}

SYMPTOMS_RR = 3.5          # 接触者已有症状（亚临床进展期）
PRIOR_TB_RR = 2.0          # 既往结核史（再激活）
HIGH_RISK_RR = 1.5         # 高危人群标记（矽肺/透析等，保守下限）
COMORBIDITY_RR = {
    'hiv': 12.0,           # WHO：PLHIV 16-27 倍，取保守下限
    'immunosuppressants': 5.0,   # 抗 TNF/移植 HR 5-25，取下限
    'diabetes': 2.5,       # Jeon 2008 RR=3.11；Cochrane 2024 1.5-2.4
    'other': 1.3,
}
BCG_WANING = {'ve0': 0.5, 'tau_years': 20.0}   # VE(age)=0.5*exp(-age/20)
HAS_TB_PREVALENCE = {'base': 0.02, 'diabetes': 0.05}

HOST_SUSCEPTIBILITY_SPEC = {
    'label_mechanism_version': LABEL_MECHANISM_VERSION,
    'change': ('susceptibility = infection susceptibility x progression '
               'amplification (composite host factor), replacing the '
               'exposure-only v2 label where host features had '
               'univariate AUROC ~= 0.50'),
    'age_progression_rr': dict(AGE_PROGRESSION_RR),
    'symptoms_rr': SYMPTOMS_RR,
    'prior_tb_rr': PRIOR_TB_RR,
    'high_risk_rr': HIGH_RISK_RR,
    'comorbidity_rr': dict(COMORBIDITY_RR),
    'bcg_waning': 'VE(age) = 0.5 * exp(-age/20)',
    'bcg_waning_params': dict(BCG_WANING),
    'has_tb_prevalence': dict(HAS_TB_PREVALENCE),
    'references': [
        'Martinez L et al. Lancet 2020;395:973-984 (137,647 child '
        'contacts; 2y progression U-curve 7.6%/5.2%/5.6%)',
        'Seddon JA & Shingadia D. Infect Drug Resist 2014;7:153-165 '
        '(progression: infants ~50%, 5-10y ~2%, adults ~5%)',
        'Fox GJ et al. PLoS Med 2013 (symptomatic contacts yield 10-30% '
        'vs 1-4% asymptomatic)',
        'Jeon CY & Murray MB. PLoS Med 2008;5(7):e152 (diabetes TB '
        'RR=3.11, 95%CI 2.27-4.26)',
        'Cochrane 2024;CD016013 (diabetes TB RR 1.5-2.4)',
        'Colditz GA et al. JAMA 1995 (BCG ~50% overall efficacy)',
        'Abubakar I et al. Health Technol Assess 2013;17.37 (BCG wanes '
        'over 10-15y)',
        'WHO Global TB Report (PLHIV 16-27x TB risk)',
    ],
}


def host_susceptibility_multiplier(age, has_symptoms=0, has_tb=0,
                                   is_high_risk=0, bcg_vaccine=0,
                                   past_illness_type='none'):
    """文献校准的复合宿主易感性乘数（单一真值源实现）。

    结构：年龄 U 型基础 × 症状/既往 TB/高危标记 × BCG 衰减保护 ×
    共病进展放大。与 evaluate_ensemble._contact_susceptibility 输出
    完全一致（随机一致性测试守护，见 tests/test_host_susceptibility_
    source.py）。

    参数：
        age (int): 接触者年龄
        has_symptoms / has_tb / is_high_risk / bcg_vaccine (0/1)
        past_illness_type (str): 'none'/'hiv'/'diabetes'/
            'immunosuppressants'/'other'

    返回：
        float: 宿主易感性乘数（与暴露强度相乘进入标签公式）
    """
    if age < 5:
        s = AGE_PROGRESSION_RR['<5']
    elif age < 15:
        s = AGE_PROGRESSION_RR['5-14']
    elif age <= 35:
        s = AGE_PROGRESSION_RR['15-35']
    elif age <= 65:
        s = AGE_PROGRESSION_RR['35-65']
    else:
        s = AGE_PROGRESSION_RR['>65']
    if has_symptoms:
        s *= SYMPTOMS_RR
    if has_tb:
        s *= PRIOR_TB_RR
    if is_high_risk:
        s *= HIGH_RISK_RR
    if bcg_vaccine:
        # 接种者：新生儿接种假设，保护随年限指数衰减
        s *= (1.0 - BCG_WANING['ve0']
              * np.exp(-age / BCG_WANING['tau_years']))
    return s * COMORBIDITY_RR.get(past_illness_type, 1.0)


__all__ = [
    'LABEL_MECHANISM_VERSION',
    'HOST_SUSCEPTIBILITY_SPEC',
    'host_susceptibility_multiplier',
    'AGE_PROGRESSION_RR',
    'SYMPTOMS_RR',
    'PRIOR_TB_RR',
    'HIGH_RISK_RR',
    'COMORBIDITY_RR',
    'BCG_WANING',
    'HAS_TB_PREVALENCE',
]
