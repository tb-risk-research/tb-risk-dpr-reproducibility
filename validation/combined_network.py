#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组合机制接触网络 DGP v1：类型 × 时序 × PI 三机制合一（用户 P1/P2）。

背景（用户 2026-08-25 "下一步优化"）：
  P1 "channel_agg > gat_hetero 的发现指明了一条低复杂度路径：把分类型
  邻居聚合特征（家庭/社会/同事各类型的邻居暴露均值）和分时间窗聚合
  特征做成特征工程，直接加进冻结的主模型（随机森林/LightGBM）特征集。
  零深度学习依赖、部署友好、可解释"；
  P2 "类型通道 + 时序窗口 + PI 物理特征能否叠加？跑一轮组合 vs 单机制
  消融，确认收益是否可加"。

三层实验已各自验证的信息不对称载体，在本 DGP 中**同时**存在：
  1. 类型不对称（第一层）：簇类型 β_t（家庭 1.0 … 偶遇 0.05）——
     边类型标注只有图可读，边特征全类型同分布（v3 教训）；
  2. 时序衰减（第二层）：每成员关系的接触窗 w 从接触者时序画像
     （recent/spread/old）抽取，decay(w)=0.5^(中点/8w)——窗口指派
     独立于一切个体可观测量（v1/v2 教训）；
  3. 源异质性（第三层）：指示病例相对传染性 s_c 对数正态
     （σ=0.7，均值 1）——只存在于指示病例节点。

真值（三机制乘性合一，全部机制因子进入每份成员关系的贡献）：

    ν_i = host_i · corr_i · Σ_c Λ(t_{w_c}) · β_{t_c} · decay(w_c)
                      · intens_{c,i} · s_c

  其中 Λ(t_w) 为**窗口时点社区感染力**（SEIR 轨迹在窗中点时刻的
  β·I(t)——早期窗与远期窗的社区流行强度不同），corr_i 为 Wells-Riley
  式暴露修正（个体可观测边特征聚合推导，沿用 pi_network）。

物理基线（部署口径，刻意构造**可学习的物理失配**，服务 P4a）：

    λ_i = host_i · Λ_ref(default β) · corr_i

  - 用默认参数轨迹（非抖动真值）——部署中 SEIR 参数只有文献值；
  - 用平坦 Λ_ref（8 周时点单值）而非窗口化 Λ(t_w)——"均匀暴露
    假设"。P4a 的联合学习臂用可学习 β 重构 Λ(t_w) 形状回收该失配。

特征化（P1 交付物：可部署的网络特征列，零深度学习依赖）：
  - type_exposure_*（5 列）：Σ_{c∈类型t} intens·s——分类型邻居暴露；
  - window_exposure_*（5 列）：Σ_{c∈窗w} intens·s——分时间窗邻居暴露；
  - physics_lambda_calibrated（1 列）：k̂·λ——校准物理感染力。
  树模型可直接从 type/window 分解读出 β·decay·Λ_w 的有效加权
  （channel_agg > gat_hetero 的机制收益大头路径）。

个体可观测（信息不对称上界，对齐 pi_network 14 维基础特征）：
  宿主字段 + 自身边特征聚合（freq/dur/dist/vent 均值）+ 边数。
  不含：类型构成 / 窗口指派 / 源传染性 / 成员关系强度。

诚实边界：三重信息不对称为构造性设定（对齐部署系统 22 列特征无
场景/时间分解暴露、无源传染性的真实约束）；本模块回答"三机制
共存时特征化能否兑现收益大头 + 收益是否可加"（机制可行性），
真实裁决需 ERASE-TB / tier-1 IPD。

文献（继承三层模块锚点）：
  Fox 2013 / Vynnycky & White 2010 —— 类型分层 β；
  Andrews 2012 / CDC 接触者筛查共识 —— 8 周衰减半衰期与窗口；
  Wells 1955 —— corr 的暴露剂量机制形式；
  Raissi 2019 —— PI 物理先验（λ 基线与 P4a 联合学习）。

==============================================================================
v3.1 参数修订依据（2026-08-25 第三轮 P3）：先验-数据对照表
==============================================================================

HomeACF 真实数据乘子审计（lambda_calibration.py，2725 接触者/877 户/
359 TST 阳性；torch 可微联合校准，文献先验为起点）给出逐乘子的
"先验 vs 数据"对照。v1 的文献先验刻度存在系统性偏差——合成实验
若沿用文献刻度，其结论对真实部署的参考价值被削弱。修订结论
（AUDIT_CALIBRATED_PRIORS，供后续 DGP 版本对齐；v1 已归档实验
不回溯改动）：

  方向一致（5/8，刻度需修订）：
    cough_days   先验 0.010/天 → 数据 0.0036（×0.36，高估 2.8 倍）
    host_hiv     先验 log1.3   → 数据 log1.6 量级（×1.78，低估）
    share_bedroom 先验 log1.2   → 数据 ×2.72（被低估最严重——共用
                 卧室的真实效应远强于文献值）
    age_ge45     先验 log2.6   → 数据 ×0.19（弱但方向对）
    log_ts       接触分级线性 → 数据 ×0.15（弱但方向对）
  方向翻转（3/8，v3.1 起自主移除）：
    age_lt5      先验 ×2.8 → 数据 −0.08。终成语义冲突：儿童 TST 硬结
                 反应弱/潜伏感染检出率天然偏低，"儿童高进展"先验在
                 LTBI 检测终点上错配——不是数据错误，是先验语义
                 （发病进展）与终点语义（感染检出）不同域；
    index_hiv    先验 ×1.3 → 数据 −0.18。代理机制：HIV+ 指示病例
                 更早进入临床关注 → 治疗启动更早 → 感染窗口更短；
    sleep_same_bed 先验 ×1.5 → 数据 −0.05。配偶/儿童同床混杂
                 （同床者年龄结构两端化）。
  不可检验（1/9）：
    smear        83% not done → 自主移除（known 子集 n=490 事件 37，
                 阳性率 7.6% vs 7.1%，AUROC 0.408——数据支持移除）。

对齐原则（"结构来自物理、参数来自数据"）：
  - DGP 机制**形式**保留（乘性分解 / 类型分层 / 时序衰减 / 剂量-
    反应），机制**刻度**向 AUDIT_CALIBRATED_PRIORS 对齐；
  - 翻转乘子在合成标签中不再注入文献方向（若注入，等于在合成世界
    复制真实数据已证伪的先验偏差）；
  - 终点语义标注：v1 隐含"感染+进展"混合语义；v3.1 起标注各机制
    因子的终点域（感染检出 vs 发病进展），与 HomeACF（LTBI）/PACTS
    （症状分诊）的真实终点对齐。
==============================================================================
"""

# v3.1：真实数据校准刻度（HomeACF 乘子审计，供后续 DGP 版本引用）
AUDIT_CALIBRATED_PRIORS = {
    'cough_slope_per_day': 0.0036,   # 先验 0.010（Verver）→ 数据 ×0.36
    'host_hiv_log': 0.467,          # 先验 log(1.3)=0.262 → 数据 0.467
    'share_bedroom_log': 0.495,     # 先验 log(1.2)=0.182 → 数据 ×2.72
    'age_ge45_log': 0.183,          # 先验 log(2.6)=0.956 → 数据 ×0.19
    'log_ts_weight': 0.148,         # 先验 1.0 → 数据 ×0.15
    'audit_meta': {
        'source': 'lambda_calibration.py multiplier_audit（HomeACF）',
        'n': 2725, 'n_events': 359, 'n_households': 877,
        'endpoint': 'tst_pos10（TST≥10mm LTBI）',
        'direction_agreement': '5/8',
        'removed_flipped': ['age_lt5', 'index_hiv', 'sleep_same_bed'],
        'removed_untestable': ['smear'],
    },
}

import numpy as np

from .host_susceptibility import host_susceptibility_multiplier
from .typed_network import EDGE_TYPE_IDS, BETA_BY_TYPE
from .temporal_network import NUM_WINDOWS, DECAY_WEIGHTS, PROFILE_SPEC
from .pi_network import (
    _seir_rk4_numpy, exposure_correction, SOURCE_INFECTIVITY_SIGMA,
    SEIR_COMMUNITY_SPEC,
)

from .pi_network import node_features as _pi_node_features   # 14 维基础


# ==============================================================================
# 宏观 SEIR：窗口时点社区感染力（40 周轨迹，覆盖最远窗中点 36 周）
# ==============================================================================

COMBINED_SEIR_SPEC = dict(SEIR_COMMUNITY_SPEC)
COMBINED_SEIR_SPEC.update({
    'n_steps': 80,                 # 40 周（dt=0.5）——覆盖 6 月+ 窗中点
    'window_mid_weeks': [1.0, 3.0, 8.0, 18.0, 36.0],   # 各窗中点
    't_ref_weeks': 8.0,            # 物理基线 Λ_ref 时点
    'note': '真值用抖动参数轨迹 + 窗口化 Λ(t_w)；物理基线用默认参数'
            ' + 平坦 Λ_ref——构造可学习的物理失配（P4a）',
})

_WINDOW_MID_STEPS = [int(round(w / COMBINED_SEIR_SPEC['dt']))
                     for w in COMBINED_SEIR_SPEC['window_mid_weeks']]
_T_REF_STEP = int(round(COMBINED_SEIR_SPEC['t_ref_weeks']
                        / COMBINED_SEIR_SPEC['dt']))


def community_seir_window_forces(rng):
    """跑一次社区 SEIR，返回真值（抖动）与物理基线（默认）的感染力组。

    Returns:
        dict: lam_w_truth（5 窗时点 β·I(t_w)，抖动参数）、
        lam_ref_default（8 周时点默认参数 Λ_ref，平坦物理基线）、
        lam_w_default（默认参数窗口化 Λ(t_w)，P4a frozen_window 臂）、
        beta_truth / I0_truth（抖动真值参数）、lam_trajectory_*（记录）
    """
    spec = COMBINED_SEIR_SPEC
    beta0, sigma, gamma = spec['beta'], spec['sigma'], spec['gamma']
    S0, E0, I0, R0 = spec['S0'], spec['E0'], spec['I0'], spec['R0']

    def _run(beta, i0):
        return _seir_rk4_numpy(beta, sigma, gamma, spec['n_steps'],
                               spec['dt'], S0, E0, i0, R0)

    # 真值：抖动参数（社区异质，对齐 pi_network 抖动幅度）
    beta_t = beta0 * (1.0 + rng.uniform(-spec['beta_jitter'],
                                        spec['beta_jitter']))
    i0_t = I0 * (1.0 + rng.uniform(-spec['i0_jitter'], spec['i0_jitter']))
    traj_t = _run(beta_t, i0_t)
    # 物理基线：默认参数（部署口径——文献值，不知当地抖动）
    traj_d = _run(beta0, I0)

    return {
        'lam_w_truth': (beta_t * traj_t[_WINDOW_MID_STEPS, 2]).tolist(),
        'lam_w_default': (beta0 * traj_d[_WINDOW_MID_STEPS, 2]).tolist(),
        'lam_ref_default': float(beta0 * traj_d[_T_REF_STEP, 2]),
        'beta_truth': float(beta_t), 'I0_truth': float(i0_t),
        'lam_trajectory_truth': (beta_t * traj_t[:, 2]).tolist(),
        'lam_trajectory_default': (beta0 * traj_d[:, 2]).tolist(),
    }


# ==============================================================================
# 网络结构参数（骨架沿用 pi_network v1 验证值）
# ==============================================================================

_HOUSEHOLD_CLUSTER_SHARE = 0.25
_CLUSTER_CAP = 7
_CONTACTS_PER_CLUSTER = 5          # pi_network v1 扫描值（平均簇规模 ~2.6）
_JOIN_PROB = {
    'household': 0.75, 'workplace': 0.30, 'school': 0.1875,
    'social': 0.375, 'casual': 0.225,
}
CLUSTER_TYPES = list(EDGE_TYPE_IDS)
_PROFILE_DRAW = [                     # 时序画像抽取（对齐 temporal v1 定稿）
    ('recent', PROFILE_SPEC['recent']['prob']),
    ('spread', PROFILE_SPEC['spread']['prob']),
    ('old', 1.0),
]

COMBINED_NETWORK_SPEC = {
    'name': 'combined_mechanism_network_v1',
    'label_mechanism':
        'p = 1 − exp(−k·host·corr·Σ_c Λ(t_wc)·β_tc·decay(wc)·intens·s_c)，'
        'k 二分校准，Bernoulli 抽签（仅接触者有标签）',
    'mechanisms': {
        'typed': '簇类型 β_t（typed_network EDGE_TYPE_SPEC）',
        'temporal': '窗指派 w ~ 画像 + decay(w)（temporal_network v1）',
        'pi_source': '源相对传染性 s_c ~ lognormal(σ=0.7)（pi_network）',
        'pi_windowed_lambda': 'Λ(t_w) 窗口时点社区感染力（本模块新增，'
                              'P4a 可学习失配来源）',
    },
    'physics_baseline':
        'λ = host·Λ_ref(默认β, 8周)·corr——部署口径（默认参数 + 平坦Λ）',
    'host_multiplier_source': 'validation/host_susceptibility.py（单一真值源）',
    'info_asymmetry': (
        '类型构成 / 窗口指派 / 源传染性 / 成员强度均只有图可读；'
        '个体仅观测宿主字段 + 自身边特征聚合（pi_network 14 维口径）'),
    'caveat': (
        '三重信息不对称为构造性设定；结论性质 = 机制可行性验证'
        '（特征化收益大头 + 可加性），非自然发现'),
}


def _node_host_fields(rng):
    """宿主字段（分布口径对齐三层 DGP）。"""
    past = 'none'
    if rng.random() < 0.12:
        past = str(rng.choice(
            ['hiv', 'diabetes', 'immunosuppressants', 'other']))
    return {
        'age': int(rng.randint(1, 80)),
        'has_symptoms': int(rng.random() < 0.30),
        'is_high_risk': int(rng.random() < 0.15),
        'has_tb': int(rng.random() < 0.08),
        'bcg_vaccine': int(rng.random() < 0.85),
        'past_illness_type': past,
    }


def _sample_edge_features(rng):
    """6 维边特征（全类型/全窗口同分布——v3+v1 教训沿用）。"""
    return [
        float(rng.randint(3, 20)),       # 0 频率（次/周）
        float(rng.uniform(10, 100)),     # 1 单次时长（分钟）
        float(rng.randint(1, 12)),       # 2 持续周数
        int(rng.randint(1, 5)),          # 3 通风（1 好 - 5 差）
        int(rng.choice(4, p=[0.3, 0.3, 0.25, 0.15])),   # 4 距离秩
        int(rng.randint(0, 4)),          # 5 场景 ordinal
    ]


def _draw_profile(rng):
    r = rng.random()
    acc = 0.0
    for name, prob in _PROFILE_DRAW[:-1]:
        acc += prob
        if r < acc:
            return name
    return 'old'


def _draw_window(rng, profile):
    wins = PROFILE_SPEC[profile]['windows']
    return int(wins[rng.randint(0, len(wins))])


def build_combined_network(n_contacts=400, target_rate=0.25,
                           random_state=42):
    """构建组合机制接触网络 v1（确定性）。

    流程：
      1. 宏观 SEIR：真值窗口化 Λ(t_w)（抖动参数）+ 物理基线平坦
         Λ_ref（默认参数）；
      2. 宿主字段 + 时序画像（recent/spread/old）逐接触者抽取；
      3. 簇（类型 β_t + 源相对传染性 s_c 对数正态）+ 成员关系
         （窗 w 从画像抽取、强度 intens = clip(0.5·s_c+0.5·U)）；
      4. 边（接触者-源 + 同簇互连，特征全网络同分布，扁平 ef 供
         PI 图转换）+ corr_i（Wells-Riley 暴露修正，个体可观测）；
      5. 真值 ν = host·corr·Σ_c Λ_w·β·decay·intens·s；
         物理 λ = host·Λ_ref_default·corr（部署口径）；
      6. k 二分校准 + Bernoulli 标签（仅接触者）；
      7. 特征化列：type/window 邻居暴露分解（P1）。

    Returns:
        dict: nodes（PI 兼容记录结构）、labels、M、ef（扁平）、
        lam_community（=Λ_ref_default，PI 图物理特征用）、
        seir_info、corr/lam/nu/s_bar（接触者长度数组）、
        exposure_by_type [n,5]、exposure_by_window [n,5]、
        memberships（接触者 → [(src, type, window, intens, s)]）、
        profiles、k_calibration、n_clusters、contact_ids、spec
    """
    rng = np.random.RandomState(random_state)

    # ---- 1. 宏观 SEIR（真值窗口化 + 物理基线平坦）----
    seir = community_seir_window_forces(rng)
    lam_w = np.asarray(seir['lam_w_truth'], dtype=float)
    decay = np.asarray(DECAY_WEIGHTS, dtype=float)
    lam_ref_default = seir['lam_ref_default']

    # ---- 2. 宿主 + 时序画像 ----
    contacts = [_node_host_fields(rng) for _ in range(n_contacts)]
    profiles = [_draw_profile(rng) for _ in range(n_contacts)]

    # ---- 3. 簇 + 成员关系 ----
    n_clusters = max(2, n_contacts // _CONTACTS_PER_CLUSTER)
    clusters = []
    for _ in range(n_clusters):
        s_c = float(np.exp(rng.normal(
            -SOURCE_INFECTIVITY_SIGMA ** 2 / 2.0,
            SOURCE_INFECTIVITY_SIGMA)))
        if rng.random() < _HOUSEHOLD_CLUSTER_SHARE:
            t = 'household'
        else:
            t = str(rng.choice(
                ['workplace', 'school', 'social', 'casual']))
        clusters.append({'type': t, 'src': len(clusters), 's': s_c,
                         'members': [], 'intens': {}, 'win': {}})
    for i in range(n_contacts):
        for t in CLUSTER_TYPES:
            if rng.random() >= _JOIN_PROB[t]:
                continue
            open_cs = [c for c in clusters
                       if c['type'] == t and len(c['members']) < _CLUSTER_CAP]
            if not open_cs:
                continue
            c = open_cs[int(rng.randint(0, len(open_cs)))]
            c['members'].append(i)
            c['intens'][i] = float(np.clip(0.5 * c['s'] + 0.5 * rng.random(),
                                           0.05, 1.0))
            c['win'][i] = _draw_window(rng, profiles[i])

    # ---- 4. 边（扁平）+ corr ----
    M = len(clusters)
    cid = {i: M + i for i in range(n_contacts)}
    ef = {}
    for c in clusters:
        for i in c['members']:
            u, v = cid[i], c['src']
            ef[(min(u, v), max(u, v))] = _sample_edge_features(rng)
        mem = [cid[i] for i in c['members']]
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                key = (min(mem[a], mem[b]), max(mem[a], mem[b]))
                ef[key] = _sample_edge_features(rng)

    edge_agg = [[] for _ in range(n_contacts)]
    for (u, v) in ef:
        feat = ef[(u, v)]
        if u >= M:
            edge_agg[u - M].append(feat)
        if v >= M:
            edge_agg[v - M].append(feat)
    corr = np.zeros(n_contacts)
    agg_means = np.zeros((n_contacts, 4))
    n_edges_arr = np.zeros(n_contacts, dtype=int)
    for i in range(n_contacts):
        n_edges_arr[i] = len(edge_agg[i])
        if not edge_agg[i]:
            continue
        arr = np.asarray(edge_agg[i], dtype=float)
        m = arr[:, [0, 1, 4, 3]].mean(axis=0)   # freq, dur, dist, vent
        agg_means[i] = m
        corr[i] = exposure_correction(m[0], m[1], m[2], m[3])

    # ---- 5. 真值 ν / 物理 λ / 特征化分解 ----
    mults = np.array([
        host_susceptibility_multiplier(
            age=h['age'], has_symptoms=h['has_symptoms'],
            has_tb=h['has_tb'], is_high_risk=h['is_high_risk'],
            bcg_vaccine=h['bcg_vaccine'],
            past_illness_type=h['past_illness_type'])
        for h in contacts])

    t_idx = {t: k for k, t in enumerate(EDGE_TYPE_IDS)}
    exposure_by_type = np.zeros((n_contacts, len(EDGE_TYPE_IDS)))
    exposure_by_window = np.zeros((n_contacts, NUM_WINDOWS))
    mech = np.zeros(n_contacts)          # Σ Λ_w·β·decay·intens·s（真值核）
    s_bar = np.ones(n_contacts)
    memberships = [[] for _ in range(n_contacts)]
    for c in clusters:
        for i in c['members']:
            w, inten = c['win'][i], c['intens'][i]
            contrib = lam_w[w] * BETA_BY_TYPE[c['type']] * decay[w] \
                * inten * c['s']
            mech[i] += contrib
            exposure_by_type[i, t_idx[c['type']]] += inten * c['s']
            exposure_by_window[i, w] += inten * c['s']
            memberships[i].append((c['src'], c['type'], w, inten, c['s']))
    # s̄：源相对传染性均值（孤立接触者为 1，对齐 pi_network）
    for i in range(n_contacts):
        if memberships[i]:
            s_bar[i] = float(np.mean([m[4] for m in memberships[i]]))

    lam = mults * lam_ref_default * corr          # 物理基线（部署口径）
    nu = mults * corr * mech                      # 真值（三机制乘性合一）

    # ---- 6. k 二分校准 + Bernoulli 标签 ----
    lo, hi = 1e-6, 1e9
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if (1.0 - np.exp(-mid * nu)).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    k_cal = 0.5 * (lo + hi)
    p_label = 1.0 - np.exp(-k_cal * nu)
    labels = (rng.random(n_contacts) < p_label).astype(int)
    if labels.sum() == 0:
        labels[int(np.argmax(p_label))] = 1
    elif labels.sum() == n_contacts:
        labels[int(np.argmin(p_label))] = 0

    # ---- 7. 节点记录（pi_network 兼容结构 + 组合扩展字段）----
    nodes = []
    for c in clusters:
        nodes.append({
            'is_index_case': 1, 'infectivity': c['s'],
            'age': 40, 'has_symptoms': 1, 'is_high_risk': 0,
            'has_tb': 1, 'bcg_vaccine': 1, 'past_illness_type': 'none',
            'mean_freq': 0.0, 'mean_duration': 0.0, 'mean_span': 0.0,
            'mean_vent': 0.0, 'mean_dist': 0.0, 'n_edges': 0,
            'host_multiplier': 1.0, 'lam_seir': 0.0, 'nu_true': 0.0,
            's_bar': 1.0, 'n_memberships': 0, 'profile': 'index',
        })
    for i in range(n_contacts):
        rec = dict(contacts[i])
        rec.update({
            'is_index_case': 0, 'infectivity': 0.0,
            'mean_freq': float(agg_means[i][0]),
            'mean_duration': float(agg_means[i][1]),
            'mean_span': 0.0,
            'mean_vent': float(agg_means[i][3]),
            'mean_dist': float(agg_means[i][2]),
            'n_edges': int(n_edges_arr[i]),
            'host_multiplier': float(mults[i]),
            'lam_seir': float(lam[i]),       # 物理基线（部署口径）
            'nu_true': float(nu[i]),         # 真值（DGP 内部/分析）
            's_bar': float(s_bar[i]),
            'n_memberships': int(len(memberships[i])),
            'profile': profiles[i],          # 画像（仅 DGP 内部/分析）
        })
        nodes.append(rec)

    return {
        'nodes': nodes,
        'labels': labels,
        'M': M,
        'ef': ef,
        'lam_community': float(lam_ref_default),   # PI 图物理特征口径
        'seir_info': seir,
        'corr': corr,
        'lam': lam,
        'nu': nu,
        's_bar': s_bar,
        'mech': mech,
        'exposure_by_type': exposure_by_type,
        'exposure_by_window': exposure_by_window,
        'memberships': memberships,
        'profiles': profiles,
        'k_calibration': float(k_cal),
        'n_clusters': n_clusters,
        'contact_ids': cid,
        'spec': dict(COMBINED_NETWORK_SPEC),
    }


# ==============================================================================
# P1 特征化：网络特征列（零深度学习依赖，直接进主模型特征集）
# ==============================================================================

TYPE_EXPOSURE_NAMES = [
    'type_exposure_' + t for t in EDGE_TYPE_IDS]
WINDOW_EXPOSURE_NAMES = [
    'window_exposure_0_2w', 'window_exposure_2_4w', 'window_exposure_1_3m',
    'window_exposure_3_6m', 'window_exposure_6m_plus']
PHYSICS_LAMBDA_NAME = 'physics_lambda_calibrated'
NET_FEATURE_NAMES = (TYPE_EXPOSURE_NAMES + WINDOW_EXPOSURE_NAMES
                     + [PHYSICS_LAMBDA_NAME])          # 11 列

FEATURE_SETS = ('ind', 'typed', 'window', 'pi', 'all')
BASE_FEATURE_DIM = 14   # pi_network 基础口径（宿主 + 边聚合，无网络信息）


def feature_matrix(net, feature_set='all', k_hat=None):
    """接触者特征矩阵（P1 特征化口径）。

    Args:
        net: build_combined_network 输出
        feature_set: 'ind'（14 维基础）/ 'typed'（+5 分类型邻居暴露）/
            'window'（+5 分窗邻居暴露）/ 'pi'（+k̂·λ 校准物理感染力）/
            'all'（+11 全部网络特征列）
        k_hat: 物理校准参数（calibrate_physics_k 输出；
            feature_set 含 'pi' 时必需——只用训练半区校准，防泄漏）

    Returns:
        (X [n_contacts, d], names)：特征矩阵与列名（v3 注册口径）
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set 必须是 {FEATURE_SETS} 之一")
    M = net['M']
    cols = [_pi_node_features(net)[M:]]
    names = list(_BASE_NAMES)
    if feature_set in ('typed', 'all'):
        cols.append(net['exposure_by_type'])
        names += TYPE_EXPOSURE_NAMES
    if feature_set in ('window', 'all'):
        cols.append(net['exposure_by_window'])
        names += WINDOW_EXPOSURE_NAMES
    if feature_set in ('pi', 'all'):
        if k_hat is None:
            raise ValueError("feature_set 含 'pi' 需要 k_hat"
                             '（calibrate_physics_k，训练半区校准）')
        cols.append((k_hat * net['lam']).reshape(-1, 1))
        names += [PHYSICS_LAMBDA_NAME]
    return np.concatenate(cols, axis=1), names


def _base_names():
    """14 维基础特征列名（pi_network.node_features 逐维口径）。"""
    return [
        'is_index_case', 'log_infectivity', 'age', 'has_symptoms',
        'is_high_risk', 'has_tb', 'bcg_vaccine', 'comorbidity',
        'mean_freq', 'mean_duration', 'mean_dist', 'mean_vent',
        'span_placeholder', 'n_edges',
    ]


_BASE_NAMES = _base_names()


def feature_dim(feature_set):
    """各特征集维度（测试/注册用）。"""
    return {
        'ind': BASE_FEATURE_DIM,
        'typed': BASE_FEATURE_DIM + 5,
        'window': BASE_FEATURE_DIM + 5,
        'pi': BASE_FEATURE_DIM + 1,
        'all': BASE_FEATURE_DIM + 11,
    }[feature_set]
