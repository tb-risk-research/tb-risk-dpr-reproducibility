#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""物理信息接触网络 DGP v1：SEIR 感染力主预测 + 网络残差的信息分解。

背景（用户"PI-GNN：把 SEIR 先验注入网络层"，2026-08-25）：
  PI-GNN 的哲学来自 PINN："已知用方程，未知用学习"——不让人工智能
  从零学全部规律，而是把物理定律作为先验约束注入，让网络只学物理
  模型没覆盖的部分：

      最终风险 = SEIR 理论值（物理已知）+ GNN 残差（数据驱动）

  在结核病接触者筛查场景，这意味着把风险分解为：

      ν_i（真实感染力） = λ_i（SEIR 理论，个体可观测推导）
                        + r_i（网络残差，只有图可读）

设计（三层桥接，对应用户"宏观-微观桥接"规格）：

  1. **SEIR 模块（宏观）**：冻结 4D SEIR（RK4 积分，numpy 镜像
     ml/gnn/_seir_ode.py 的方程与格式；文献参数 R0=2.5、潜伏期
     8 周=IGRA 窗口锚点、传染期 10 周），跑出社区感染力
     Λ(t) = β·I(t)，取 t_ref = 8 周参考值——每种子 β/I0 小幅
     抖动（网络间社区流行强度差异）；
  2. **感染力映射层（中观）**：λ_i = host_i × Λ × corr_i。
     暴露修正因子 corr_i 是 Wells-Riley 式机制乘积：
     corr = (频率/20)·(时长/100)·近距离衰减·通风因子——全部由
     个体可观测的边特征聚合推导（接触调查中已知：接触频率、单次
     时长、距离、通风），不含任何隐藏信息；
  3. **GNN 残差（微观）**：真实感染力 ν_i = host_i × Λ·corr_i·s̄_i，
     其中 s̄_i 是接触者所属各簇指示病例的**相对传染性均值**
     （infectivity/E[inf]）——SEIR 的"均匀传染源假设"（Λ 为社区
     均值）无法刻画的源异质性，只有图可读（GAT 从指示病例节点
     消息传递）。残差 r_i = ν_i − λ_i = λ_i·(s̄_i − 1)：乘性结构，
     物理方向保持（Spearman ρ(λ,ν) ≈ 0.94）。

  （v1 扫描曾设簇伴热度 h 项——同簇成员 corr 均值，作为第二残差
   通道；实证其与 λ 秩正相关、只放大不改序，且稀释 ν 的相对展开
   度使 oracle 反降（hs 0→0.5→1.5：gap +0.028→+0.012 单调降），
   故删除。单一残差通道恰好对齐用户规格——GNN 增量只来自
   "修正 SEIR 模型无法刻画的传染源异质性"。）

  标签：p_i = 1 − exp(−k·ν_i)，k 二分校准至 target_rate，
  Bernoulli 抽签（仅接触者有标签）。

信息集（消融臂的公平口径）：
  - 纯 SEIR 臂：λ_i 精确已知（公式 + 个体特征 + Λ）——物理部分
    的 oracle 上界；残差 r_i = ν_i − λ_i 完全不可见；
  - 纯 GNN 臂：节点基础特征（宿主 + 边特征聚合）+ 图——不给 λ，
    端到端从零学（含 exp 结构）；
  - PI-GNN 臂：基础特征 + 物理特征（λ̃、Λ、传染源密度）+ 残差
    融合结构（p = 1−exp(−(λ̃+δ))，δ 有界）。

诚实边界（必须随产物声明）：
  1. 物理公式精确已知是**构造性**设定（部署中 SEIR 参数有文献
     不确定性——这使本实验对"残差可学性"的检验偏保守：物理越
     不完美，残差空间越大）；
  2. 残差（s̄）只有图可读是构造性设定（对齐"个体特征已含
     全部个体信息时，网络层增量只能来自修正物理模型均匀暴露
     假设"的路径）；
  3. 本模块回答"物理锚定 + 有界残差能否兑现为判别/追踪增量"
     （机制可行性），不回答"真实世界残差规模是否如此"。

文献：
  Raissi, Perdikaris & Karniadakis (2019) J. Comput. Phys. —— PINN
      （物理方程残差作为损失约束，PI-GNN 理论起点）；
  Wells WF (1955) Am. J. Epidemiol. —— Wells-Riley 空气传播感染
      模型（感染概率 ∝ 暴露剂量：传染源×时长×通风，corr 的机制
      形式来源）；
  Kermack & McKendrick (1927) —— SEIR 房室模型；
  Andrews JR et al. (2012) —— 接触者感染风险的时间-强度依赖；
  HeatGNN (Zheng et al., arXiv 2024) —— 流行病学知情 GNN
      （机制损失对齐 SIR 行为）；
  CSTGNN (Han et al., arXiv 2025) —— Spatio-Contact SIR 作为
      因果先验集成到时空 GNN（特征级注入路线）。
"""

import numpy as np

from .host_susceptibility import host_susceptibility_multiplier


# ==============================================================================
# 宏观：社区 SEIR（冻结文献参数，单一真值源）
# ==============================================================================

SEIR_COMMUNITY_SPEC = {
    'beta': 0.25,        # 传播率/周（R0 = β/γ = 2.5，TB 家庭簇内传播区间）
    'sigma': 0.125,      # 潜伏→感染转化率（1/8 周——IGRA 窗口期锚点）
    'gamma': 0.1,        # 恢复率/周（传染期 ~10 周：症状前+诊断前阶段）
    'S0': 0.98, 'E0': 0.015, 'I0': 0.002, 'R0': 0.003,
    'n_steps': 16, 'dt': 0.5,          # 总模拟 8 周（t_ref 参考时点）
    'beta_jitter': 0.2,                # 种子间 β 抖动 ±20%（社区异质）
    'i0_jitter': 0.5,                  # 种子间 I0 抖动 ±50%
    'solver': 'rk4',                   # 四阶 RK4（镜像 _seir_ode.py 生产路径）
    'mirror_of': 'ml/gnn/_seir_ode.py::seir_rk4_integrate（torch 可微版本）',
}


def _seir_rk4_numpy(beta, sigma, gamma, n_steps, dt, S0, E0, I0, R0):
    """4D SEIR RK4 积分（numpy 镜像，与 _seir_ode.py torch 版逐步一致）。

    方程（归一化比例，S+E+I+R=1）：
        dS/dt = −βSI,  dE/dt = βSI−σE,  dI/dt = σE−γI,  dR/dt = γI
    每步 RK4 推进后 clamp 非负 + 重归一化（与 torch 版格式一致，
    供交叉验证测试逐位比对）。
    """
    state = np.array([S0, E0, I0, R0], dtype=np.float64)

    def _deriv(s):
        S, E, I = s[0], s[1], s[2]
        infection = beta * S * I
        return np.array([-infection,
                         infection - sigma * E,
                         sigma * E - gamma * I,
                         gamma * I])

    trajectory = [state.copy()]
    for _ in range(n_steps):
        k1 = _deriv(state)
        k2 = _deriv(state + 0.5 * dt * k1)
        k3 = _deriv(state + 0.5 * dt * k2)
        k4 = _deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        state = np.clip(state, 0.0, None)
        state = state / (state.sum() + 1e-8)
        trajectory.append(state.copy())
    return np.array(trajectory)          # [n_steps+1, 4]


def community_seir_force(rng):
    """跑一次冻结社区 SEIR，返回 8 周时点感染力 Λ_ref。

    Λ(t) = β·I(t)（S≈1 近似下的 per-capita 感染力）。种子间
    β/I0 抖动使各网络的社区流行强度有差异。

    Returns:
        (lam_ref, info)：Λ_ref 标量 + 轨迹信息 dict（Λ(t) 序列等）
    """
    spec = SEIR_COMMUNITY_SPEC
    beta = spec['beta'] * (1.0 + rng.uniform(-spec['beta_jitter'],
                                             spec['beta_jitter']))
    I0 = spec['I0'] * (1.0 + rng.uniform(-spec['i0_jitter'],
                                         spec['i0_jitter']))
    traj = _seir_rk4_numpy(beta, spec['sigma'], spec['gamma'],
                           spec['n_steps'], spec['dt'],
                           spec['S0'], spec['E0'], I0, spec['R0'])
    lam_t = beta * traj[:, 2]            # Λ(t) = β·I(t)
    lam_ref = float(lam_t[-1])           # t_ref = 8 周
    return lam_ref, {
        'beta': float(beta), 'I0': float(I0),
        'lam_trajectory': lam_t.tolist(),
        'I_trajectory': traj[:, 2].tolist(),
        't_ref_weeks': spec['n_steps'] * spec['dt'],
    }


# ==============================================================================
# 中观：感染力映射层（社区 Λ → 个体 λ，Wells-Riley 式暴露修正）
# ==============================================================================

# 距离秩 → 近距离衰减因子（接触距离越远，单位时长暴露剂量越低）
PROX_BY_DIST = [1.0, 0.6, 0.35, 0.15]
FEAT_SCALE = np.array([20.0, 100.0, 12.0, 5.0, 3.0, 3.0])
EDGE_FEATURE_DIM = 6


def exposure_correction(mean_freq, mean_duration, mean_dist, mean_vent):
    """暴露修正因子 corr（感染力映射层的核心，Wells-Riley 式乘积）。

    corr = (频率/20)·(时长/100)·近距离衰减(距离)·通风因子(通风)
    ——全部来自个体可观测的边特征聚合；接触调查场景中个体对
    自己的接触频率/时长/距离/通风均有报告（不含隐藏信息）。
    """
    f = float(mean_freq) / 20.0
    d = float(mean_duration) / 100.0
    d_rank = int(np.clip(np.floor(float(mean_dist) + 0.5), 0, 3))
    prox = PROX_BY_DIST[d_rank]
    v = (6.0 - float(mean_vent)) / 5.0   # 通风 1(好)→1.0, 5(差)→0.2
    return f * d * prox * v


# ==============================================================================
# 网络结构参数（typed_network v3 骨架沿用）
# ==============================================================================

_HOUSEHOLD_CLUSTER_SHARE = 0.25
_CLUSTER_CAP = 7
# v1 扫描调优（2026-08-25）：12 → 5。容量 7×(n/12)=0.58n < 加入
# 期望 1.84n，~58% 接触者零簇成员（孤立、λ=0 且必阴性）——隔离
# 结构主导 AUROC、残差空间被压缩到 +0.003。5 对应平均簇规模
# ~2.6（ERASE-TB 786 户/2,109 接触者 = 2.7/户，BMJ Open 2022）
_CONTACTS_PER_CLUSTER = 5
_JOIN_PROB = {
    'household': 0.75, 'workplace': 0.30, 'school': 0.1875,
    'social': 0.375, 'casual': 0.225,
}
CLUSTER_TYPES = ['household', 'workplace', 'school', 'social', 'casual']

# 传染源相对传染性 s = exp(N(−σ²/2, σ²))：均值 1、sd ≈ 0.78
# （涂阳/空洞 vs 涂阴传染性跨越约一个数量级——v1 扫描调优：
#   Beta(2,2)/0.5 的 sd 0.45 使残差信号不足）
SOURCE_INFECTIVITY_SIGMA = 0.7

_EDGE_PROFILE = {
    'freq': (3, 20),                    # 接触频率（次/周）
    'duration': (10, 100),              # 单次时长（分钟）
    'dist_weights': [0.3, 0.3, 0.25, 0.15],   # 距离秩 0(近)-3(远)
    'vent': (1, 5),                     # 通风 1(好)-5(差)
}

PI_NETWORK_SPEC = {
    'name': 'pi_contact_network_v1',
    'decomposition': 'ν_i = host_i·Λ·corr_i·s̄_i; '
                     'λ_i = host_i·Λ·corr_i（SEIR 理论）；'
                     '残差 r_i = λ_i·(s̄_i − 1)（只有图可读）',
    'label_mechanism': 'p = 1 − exp(−k·ν)（k 二分校准），Bernoulli 抽签',
    'seir': SEIR_COMMUNITY_SPEC,
    'exposure_correction': 'Wells-Riley 式乘积（频率×时长×距离衰减×通风）',
    'host_multiplier_source': 'validation/host_susceptibility.py（单一真值源）',
    'source_infectivity': '对数正态 exp(N(−σ²/2, σ))，σ=%.1f（均值 1）'
                          % SOURCE_INFECTIVITY_SIGMA,
    'info_asymmetry': (
        '物理公式（corr 映射）精确且只用个体可观测特征；残差由'
        '传染源相对传染性 s̄（均匀传染源假设修正）构成——只有图'
        '可读'),
    'caveat': (
        '物理精确已知 + 残差图可读均为构造性设定；结论性质 = 机制'
        '可行性验证（物理越不完美残差空间越大，本检验偏保守），'
        '非自然发现。'),
}


def _node_host_fields(rng):
    """宿主字段（分布口径对齐 typed/temporal DGP）。"""
    past = 'none'
    if rng.random() < 0.12:
        past = str(rng.choice(['hiv', 'diabetes', 'immunosuppressants', 'other']))
    return {
        'age': int(rng.randint(1, 80)),
        'has_symptoms': int(rng.random() < 0.30),
        'is_high_risk': int(rng.random() < 0.15),
        'has_tb': int(rng.random() < 0.08),
        'bcg_vaccine': int(rng.random() < 0.85),
        'past_illness_type': past,
    }


def _sample_edge_features(rng):
    """采样一条边的 6 维特征（全网络同分布——v3 教训沿用）。"""
    fr = _EDGE_PROFILE['freq']
    du = _EDGE_PROFILE['duration']
    dp = np.asarray(_EDGE_PROFILE['dist_weights'], dtype=float)
    ve = _EDGE_PROFILE['vent']
    return [
        float(rng.randint(*fr)),          # 0 频率
        float(rng.uniform(*du)),          # 1 单次时长
        float(rng.randint(1, 12)),        # 2 持续周数
        int(rng.randint(*ve)),            # 3 通风（1 好 - 5 差）
        int(rng.choice(4, p=dp)),         # 4 距离秩（0 近 - 3 远）
        int(rng.randint(0, 4)),           # 5 场景 ordinal
    ]


def build_pi_network(n_contacts=400, target_rate=0.25, random_state=42):
    """构建物理信息接触网络 v1（确定性）。

    流程：
      1. 宏观：冻结 SEIR → Λ_ref（8 周时点社区感染力）；
      2. 宿主字段 + 簇结构（v3 骨架，类型只作结构多样性）；
      3. 边（接触者-指示病例 + 同簇互连，特征全网络同分布）；
      4. 中观：corr_i = 暴露修正因子（个体边特征聚合 → 机制乘积）；
      5. 微观：s̄_i（源相对传染性均值）——只有图可读；
      6. λ_i = host·Λ·corr（物理）；ν_i = host·Λ·corr·s̄（真值）；
      7. p = 1−exp(−k·ν)，k 二分校准，Bernoulli 标签（仅接触者）。

    Returns:
        dict: nodes（前 M 指示病例 + n_contacts 接触者）、labels、M、
        ef（{(u,v): 6 维特征}，扁平无窗）、lam_community、seir_info、
        corr/s_bar/lam/nu（接触者长度数组）、k_calibration、
        n_clusters、contact_ids、spec
    """
    rng = np.random.RandomState(random_state)

    # ---- 1. 宏观 SEIR ----
    lam_community, seir_info = community_seir_force(rng)

    # ---- 2. 宿主 + 簇 ----
    contacts = [_node_host_fields(rng) for _ in range(n_contacts)]
    n_clusters = max(2, n_contacts // _CONTACTS_PER_CLUSTER)
    index_cases = []
    clusters = []
    for _ in range(n_clusters):
        # 相对传染性采样（v1 扫描调优）：对数正态
        # s = exp(N(−σ²/2, σ))——均值 1、sd≈0.78，跨越约一个
        # 数量级（涂阳/空洞 vs 涂阴传染性差异的文献锚点）。
        # 此前 Beta(2,2)（sd 0.45）残差信号不足。
        infectivity = float(np.exp(rng.normal(
            -SOURCE_INFECTIVITY_SIGMA ** 2 / 2.0, SOURCE_INFECTIVITY_SIGMA)))
        if rng.random() < _HOUSEHOLD_CLUSTER_SHARE:
            t = 'household'
        else:
            t = str(rng.choice(['workplace', 'school', 'social', 'casual']))
        index_cases.append({'infectivity': infectivity})
        clusters.append({'type': t, 'src': len(index_cases) - 1,
                         'infectivity': infectivity, 'members': []})

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

    # ---- 3. 边（扁平：源边 + 同簇互连）----
    M = len(index_cases)
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

    # ---- 4. 中观：暴露修正因子 corr ----
    edge_agg = [[] for _ in range(n_contacts)]
    for (u, v) in ef:
        feat = ef[(u, v)]
        if u >= M:
            edge_agg[u - M].append(feat)
        if v >= M:
            edge_agg[v - M].append(feat)
    corr = np.zeros(n_contacts)
    n_edges_arr = np.zeros(n_contacts)
    agg_means = np.zeros((n_contacts, 4))   # freq/dur/dist/vent 均值
    for i in range(n_contacts):
        n_edges_arr[i] = len(edge_agg[i])
        if not edge_agg[i]:
            continue
        arr = np.asarray(edge_agg[i], dtype=float)
        # 列序修正（v1 扫描发现）：边特征序为
        # [freq, dur, weeks, vent, dist, scene]——距离在第 4 列。
        # 此前 arr[:, :4] 把周数(1-12)当距离传入，clip 成秩 3、
        # prox 恒 0.15，距离信息从未进入 corr。
        m = arr[:, [0, 1, 4, 3]].mean(axis=0)   # freq, dur, dist, vent
        agg_means[i] = m
        corr[i] = exposure_correction(m[0], m[1], m[2], m[3])

    # ---- 5. 微观：s̄（源相对传染性）----
    # 相对传染性：s_c = infectivity / E[s] = infectivity / 1.0
    # （对数正态均值 1——v1 扫描调优，见 SOURCE_INFECTIVITY_SIGMA）
    src_rel = {c['src']: c['infectivity'] / 1.0 for c in clusters}
    s_bar = np.ones(n_contacts)
    memberships = [[] for _ in range(n_contacts)]
    for c in clusters:
        for i in c['members']:
            memberships[i].append(c['src'])
    for i in range(n_contacts):
        if memberships[i]:
            s_bar[i] = float(np.mean([src_rel[s] for s in memberships[i]]))

    # ---- 6. 物理 λ / 真值 ν ----
    mults = np.array([
        host_susceptibility_multiplier(
            age=h['age'], has_symptoms=h['has_symptoms'],
            has_tb=h['has_tb'], is_high_risk=h['is_high_risk'],
            bcg_vaccine=h['bcg_vaccine'],
            past_illness_type=h['past_illness_type'])
        for h in contacts])
    lam = mults * lam_community * corr                  # SEIR 理论
    nu = mults * lam_community * corr * s_bar           # 真值（残差 λ·(s̄−1)）

    # ---- 7. k 二分校准 + Bernoulli 标签 ----
    lo, hi = 1e-6, 1e7
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

    # ---- 8. 节点记录 ----
    nodes = []
    for s in index_cases:
        nodes.append({
            'is_index_case': 1, 'infectivity': s['infectivity'],
            'age': 40, 'has_symptoms': 1, 'is_high_risk': 0,
            'has_tb': 1, 'bcg_vaccine': 1, 'past_illness_type': 'none',
            'mean_freq': 0.0, 'mean_duration': 0.0, 'mean_span': 0.0,
            'mean_vent': 0.0, 'mean_dist': 0.0, 'n_edges': 0,
            'host_multiplier': 1.0, 'lam_seir': 0.0, 'nu_true': 0.0,
            's_bar': 1.0, 'n_memberships': 0,
        })
    for i in range(n_contacts):
        h = contacts[i]
        rec = dict(h)
        rec.update({
            'is_index_case': 0, 'infectivity': 0.0,
            'mean_freq': float(agg_means[i][0]),
            'mean_duration': float(agg_means[i][1]),
            'mean_span': 0.0,
            'mean_vent': float(agg_means[i][3]),
            'mean_dist': float(agg_means[i][2]),
            'n_edges': int(n_edges_arr[i]),
            'host_multiplier': float(mults[i]),
            'lam_seir': float(lam[i]),        # 物理（个体可观测推导）
            'nu_true': float(nu[i]),          # 真值（DGP 内部/分析）
            's_bar': float(s_bar[i]),         # 残差项（只有图可读）
            'n_memberships': int(len(memberships[i])),
        })
        nodes.append(rec)

    return {
        'nodes': nodes,
        'labels': labels,
        'M': M,
        'ef': ef,
        'lam_community': float(lam_community),
        'seir_info': seir_info,
        'corr': corr,
        's_bar': s_bar,
        'lam': lam,
        'nu': nu,
        'k_calibration': float(k_cal),
        'n_clusters': n_clusters,
        'contact_ids': cid,
        'spec': dict(PI_NETWORK_SPEC),
    }


# ==============================================================================
# 特征构造（消融臂的公平特征预算）
# ==============================================================================

BASE_FEATURE_DIM = 14   # 基础节点特征（宿主 + 边特征聚合；无物理 λ）


def node_features(net):
    """全图节点基础特征（M 指示病例 + n 接触者，14 维，无物理信息）。

    维度：是否指示病例 / log 传染性（仅指示病例，接触者 0）/ 年龄 /
    症状 / 高危 / 既往 TB / BCG / 共病 / 频率 / 时长 / 距离 / 通风 /
    接触周数占位（0）/ 边数。

    传染性取 log（v11 扫描定稿）：残差在 log 空间线性
    （log ν = log λ + log s̄），线性读出可直接兑现（原始尺度下
    乘性跨越一个数量级、线性读出只能读单侧）。

    不含 λ（物理特征由 to_pi_graph(include_physics=True) 追加）——
    纯 GNN 臂只见本特征，物理注入臂在此基础上加物理特征。
    """
    rows = []
    for n in net['nodes']:
        rows.append([
            float(n['is_index_case']),
            (float(np.log(max(n['infectivity'], 1e-3)))
             if n['is_index_case'] else 0.0),
            n['age'] / 80.0,
            float(n['has_symptoms']),
            float(n['is_high_risk']),
            float(n['has_tb']),
            float(n['bcg_vaccine']),
            1.0 if n['past_illness_type'] != 'none' else 0.0,
            n['mean_freq'] / 20.0,
            n['mean_duration'] / 100.0,
            n['mean_dist'] / 3.0,
            n['mean_vent'] / 5.0,
            0.0,                            # 周数占位（对齐 6 维边特征语义）
            min(n['n_edges'], 20) / 20.0,
        ])
    return np.asarray(rows, dtype=np.float64)


def individual_features(net):
    """接触者个体特征矩阵（纯 GNN 臂输入的个体部分）。"""
    return node_features(net)[net['M']:]


def physics_lambda(net):
    """物理感染力 λ_i（接触者数组；SEIR 理论，个体可观测推导）。"""
    return np.asarray([n['lam_seir'] for n in net['nodes'][net['M']:]],
                      dtype=float)


def calibrate_physics_k(net, labels, train_contacts):
    """物理检出率单参数校准：k̂ 使 mean(1−exp(−k̂·λ_train)) = 训练阳性率。

    部署语义：SEIR 冻结（文献参数）+ 单参数校准到当地阳性率——
    "SEIR 冻结"最简方案（用户方案 1）的校准件。λ̃ = k̂·λ 即校准
    后物理感染力（AUROC 与 λ 等价——单调变换；Brier 用
    p = 1−exp(−λ̃)）。
    """
    lam = physics_lambda(net)
    labels = np.asarray(labels, dtype=float)
    tr = np.asarray(train_contacts, dtype=np.int64)
    target = float(labels[tr].mean())
    lam_tr = lam[tr]
    if lam_tr.max() <= 0:                    # 退化保护（全孤立节点）
        return 1.0
    lo, hi = 1e-9, 1e9
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if (1.0 - np.exp(-mid * lam_tr)).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _source_density(net):
    """传染源密度：每节点邻居中指示病例占比（物理特征之一）。"""
    M = net['M']
    n = len(net['nodes'])
    deg = np.zeros(n)
    src_nbr = np.zeros(n)
    for (u, v) in net['ef']:
        deg[u] += 1
        deg[v] += 1
        if u < M:
            src_nbr[v] += 1
        if v < M:
            src_nbr[u] += 1
    return np.where(deg > 0, src_nbr / np.maximum(deg, 1), 0.0)


def to_pi_graph(net, include_physics=False, k_hat=None):
    """把 DGP 网络转成 PIGNNNet 输入（torch 张量，全图节点）。

    Args:
        net: build_pi_network 输出
        include_physics: True → 节点特征追加物理 3 维
            [λ̃（=k̂·λ，校准物理感染力）、Λ·1e3（社区感染力）、
            传染源密度]（用户"物理特征提取器"规格）
        k_hat: 物理校准参数（calibrate_physics_k 输出；
            include_physics=True 时必需）

    Returns:
        dict: x [N, 14 或 17]、edge_index [2, 2E]、edge_attr [2E, 6]
        （FEAT_SCALE 归一化）、lam_tilde [N]（校准物理感染力，
        指示病例为 0；include_physics=False 时为 None）
    """
    import torch

    X = node_features(net)
    lam = physics_lambda(net)
    lam_tilde = None
    if include_physics:
        if k_hat is None:
            raise ValueError('include_physics=True 需要 k_hat'
                             '（calibrate_physics_k 输出）')
        lam_full = np.zeros(len(net['nodes']), dtype=float)
        lam_full[net['M']:] = k_hat * lam
        lam_tilde = lam_full
        phys = np.column_stack([
            lam_full,
            np.full(len(net['nodes']), net['lam_community'] * 1e3),
            _source_density(net),
        ])
        X = np.concatenate([X, phys], axis=1)

    x = torch.tensor(X, dtype=torch.float32)
    if not net['ef']:
        ei = torch.zeros((2, 0), dtype=torch.long)
        ea = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32)
    else:
        pairs = sorted(net['ef'].keys())
        arr = np.array(pairs, dtype=np.int64)
        f = np.array([net['ef'][k] for k in pairs],
                     dtype=np.float64) / FEAT_SCALE
        ei = torch.tensor(
            np.concatenate([arr, arr[:, ::-1]], axis=0).T, dtype=torch.long)
        ea = torch.tensor(
            np.concatenate([f, f], axis=0), dtype=torch.float32)
    lam_t = (torch.tensor(lam_tilde, dtype=torch.float32)
             if lam_tilde is not None else None)
    return {'x': x, 'edge_index': ei, 'edge_attr': ea, 'lam_tilde': lam_t}
