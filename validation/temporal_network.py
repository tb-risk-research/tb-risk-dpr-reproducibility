#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""时序接触网络 DGP v1：时间切片 + 衰减权重的信息不对称构造。

背景（用户"第二层：时序 GNN（接触时间维度）"，2026-08-25）：
  接触不是静态的——上周接触的和上个月接触的，风险不一样。个体特征
  里的 time_span（接触持续时间）是粗粒度总量：同一个人，100 小时的
  接触集中在最近两周还是分散在半年里，传播风险完全不同。这一时序
  分布模式只存在于按时间切片的图里（哪条边落在哪个窗），个体标量
  特征捕捉不到。

设计（继承 typed_network v3 的三条教训）：
  - v1 教训（边特征不分层 → 无信号）：窗口指派必须进入标签机制
    （衰减权重），否则时序模式与标签无关；
  - v2 教训（个体特征成为隐藏维度的完美代理）：窗口指派独立于
    边特征 / 宿主特征 / 接触强度 / 传染性——个体从任何可观测聚合
    都无法推断"接触落在哪些窗"；
  - v3 成功要素（同分布 + 差异只在标注）：边特征全窗口同分布，
    时间差异只存在于衰减权重与窗口标注（只有切片图能读）。

本 DGP 构造（v1，时序层）：
  1. **5 个时间窗**（用户规格，结核文献锚定）：0-2 周 / 2-4 周 /
     1-3 月 / 3-6 月 / 6 月+；
  2. **时间衰减权重**（标签机制，半衰期 8 周）：
     decay(w) = 0.5^(窗中点周数/8) = [0.917, 0.771, 0.500, 0.207,
     0.042]——锚点：IGRA 窗口期 ~8 周（复测 10 周）、指示病例传染期
     ~12 周（3 月窗衰减 ≈ 0.35）、6 月+ 接触近乎不计（进展风险虽
     集中在前 1-2 年，但传染源传染性随回溯时间衰减更快）；
  3. **每接触者时序画像**（recency profile，信息不对称载体）：
     recent 30%（边集中落在 {0-2w, 2-4w}）/ spread 40%（均匀
     落在 5 窗）/ old 30%（集中落在 {3-6m, 6m+}）——直接构造
     "同等总接触量、不同时间分布"的对比：recent 画像的有效暴露
     比值 temporal/cum ≈ 0.84，spread ≈ 0.49，old ≈ 0.12；
  4. **簇结构沿用 v3 骨架**（家庭 25% 等 + 指示病例 infectivity），
     但类型不对称**关闭**（所有 β = 1）——第二层只隔离"时序机制"
     单一变量，避免与第一层类型信号混淆；
  5. **个体可观测特征不含时序**：cumulative_exposure =
     Σ intensity·infectivity（不加衰减）、time_span 等边特征混合
     均值——个体只能看到总量；
  6. **真实风险（信息不对称载体）**：temporal_exposure = Σ decay(
     window) × intensity × infectivity(src)，标签 p = 1 - exp(-k ×
     temporal × 宿主乘数)，Bernoulli 抽签；k 二分校准至
     target_rate（宿主乘数单一真值源不变）。

诚实边界（必须随产物声明）：时序画像不可观测是**构造性**设定，
对应部署系统 22 列特征只有 time_span 总量、无按接触时间分解的
暴露特征这一真实约束。本模块回答"时间信息不对称存在时，时序
GNN 能否兑现为判别增量"（机制可行性），不是"真实世界必然存在
该不对称"（需 ERASE-TB 等带接触时间戳的数据裁决）。

文献：
  Andrews JR et al. (2012) PLoS ONE —— 接触者感染风险的时间-强度
      依赖（风险集中于暴露后早期）；
  CDC / 中国防痨协会 (2025) 接触者筛查共识 —— IGRA 窗口期 8-12 周
      （近期暴露后免疫反应需 ~8 周可检）；
  NYC DOHMH TB manual —— 传染期 = 症状出现或隔离/治疗前 ~12 周
      （接触调查标准回溯窗口 3 个月）；
  Fox GJ et al. (2013) PLoS ONE —— 家庭密接感染率 30-40%；
  Xu D et al. (2024) TGN/TGAT 综述 + UTG 统一框架 —— snapshot
      序列 + 窗间 RNN 是时序 GNN 的标准形态。
"""

import numpy as np

from .host_susceptibility import host_susceptibility_multiplier


# ==============================================================================
# 时间窗规格（文献锚定，单一真值源）
# ==============================================================================

WINDOW_SPEC = [
    {'id': 0, 'label': '0-2周', 'range_weeks': (0, 2), 'mid_week': 1.0,
     'reference': 'IGRA 窗口期起点（近期暴露，进展/检出风险最高）'},
    {'id': 1, 'label': '2-4周', 'range_weeks': (2, 4), 'mid_week': 3.0,
     'reference': 'IGRA 窗口期内（免疫反应阳转期）'},
    {'id': 2, 'label': '1-3月', 'range_weeks': (4, 12), 'mid_week': 8.0,
     'reference': 'IGRA 窗口期 8 周 + 接触调查标准回溯 3 个月'},
    {'id': 3, 'label': '3-6月', 'range_weeks': (12, 24), 'mid_week': 18.0,
     'reference': '传染期 12 周边界之外，风险显著衰减'},
    {'id': 4, 'label': '6月+', 'range_weeks': (24, None), 'mid_week': 36.0,
     'reference': '远期接触（进展风险集中于前 1-2 年，但源传染性衰减）'},
]
NUM_WINDOWS = len(WINDOW_SPEC)

# 时间衰减半衰期（周）：IGRA 窗口期 8 周锚点
DECAY_HALF_LIFE_WEEKS = 8.0
# 衰减权重 decay(w) = 0.5^(窗中点/半衰期)，索引 = 窗 id（0 最近）
DECAY_WEIGHTS = [0.5 ** (s['mid_week'] / DECAY_HALF_LIFE_WEEKS)
                 for s in WINDOW_SPEC]

# 时序画像（recency profile）：概率与窗口分布
# （v1 扫描调优 2026-08-25：old 画像极化至 {6月+} 单窗——衰减对比
#   0.92 vs 0.04 ≈ 21x，oracle−个体间隙从 +0.005 提升到可学量级）
PROFILE_SPEC = {
    'recent': {'prob': 0.35, 'windows': (0, 1),
               'label': '近期密集接触（0-4 周内）'},
    'spread': {'prob': 0.30, 'windows': (0, 1, 2, 3, 4),
               'label': '分散接触（全窗均匀）'},
    'old': {'prob': 0.35, 'windows': (4,),
            'label': '远期接触（6 月前）'},
}

TEMPORAL_NETWORK_SPEC = {
    'name': 'temporal_contact_network_v1',
    'label_mechanism':
        'p = 1 - exp(-k * Σ decay(window)·intensity·infectivity(src) × '
        '宿主乘数), Bernoulli 抽签（仅接触者有标签）',
    'host_multiplier_source': 'validation/host_susceptibility.py（单一真值源）',
    'windows': WINDOW_SPEC,
    'decay_weights': DECAY_WEIGHTS,
    'decay_half_life_weeks': DECAY_HALF_LIFE_WEEKS,
    'profiles': PROFILE_SPEC,
    'info_asymmetry': (
        '边特征全窗口同分布 + 窗口指派独立于一切个体可观测量；'
        '时间差异只存在于衰减权重（标签机制）与窗口标注（只有切片'
        '图可读）——个体特征只能观测不分时间的累积暴露总量'),
    'layer_isolation': (
        '簇结构沿用 typed_network v3 骨架但所有 β = 1（类型不对称'
        '关闭）——本层只隔离"时序机制"单一变量'),
    'caveat': (
        '时序画像不可观测为构造性设定（对齐部署系统 22 列特征只有 '
        'time_span 总量、无接触时间分解的真实约束）；结论性质 = '
        '机制可行性验证，非自然发现。'),
}

# ==============================================================================
# 网络结构参数（沿用 typed_network v3 验证值）
# ==============================================================================

_HOUSEHOLD_CLUSTER_SHARE = 0.25   # 家庭簇占比（结构多样化）
_CLUSTER_CAP = 7                  # 每簇接触者容量上限
_CONTACTS_PER_CLUSTER = 12        # 簇数 = max(2, n_contacts // 12)
_JOIN_PROB = {
    'household': 0.75, 'workplace': 0.30, 'school': 0.1875,
    'social': 0.375, 'casual': 0.225,
}
CLUSTER_TYPES = ['household', 'workplace', 'school', 'social', 'casual']

# 边特征分布（v3 教训：全窗口同分布——窗口指派不改变边特征分布）
_EDGE_PROFILE = {
    'freq': (3, 20),                    # 接触频率（次/周）
    'duration': (10, 100),              # 单次时长（分钟）
    'dist_weights': [0.3, 0.3, 0.25, 0.15],   # 距离秩 0(近)-3(远)
    'vent': (1, 5),                     # 通风 1(好)-5(差)
}
EDGE_FEATURE_DIM = 6   # 频率/时长/周数/通风/距离/场景（对齐 HeteroGATLayer）
FEAT_SCALE = np.array([20.0, 100.0, 12.0, 5.0, 3.0, 3.0])


def _node_host_fields(rng):
    """宿主字段（分布口径对齐 typed_network v3 / DGP FIELD_SPECS）。"""
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
    """采样一条边的 6 维特征（频率/时长/周数/通风/距离/场景 ordinal）。

    v3 教训沿用：**全窗口同分布**——无论近期窗还是远期窗，频率/
    时长/距离/通风的分布完全一致。个体从边特征聚合无法推断
    "接触落在哪些时间窗"；时间差异只存在于衰减权重与窗口标注
    （只有切片图能读的信息）。
    """
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


def _draw_profile(rng):
    """抽取接触者时序画像：recent 30% / spread 40% / old 30%。"""
    r = rng.random()
    if r < PROFILE_SPEC['recent']['prob']:
        return 'recent'
    if r < PROFILE_SPEC['recent']['prob'] + PROFILE_SPEC['spread']['prob']:
        return 'spread'
    return 'old'


def _draw_window(rng, profile):
    """从时序画像的窗口分布中抽取一个窗 id（0 最近 - 4 最远）。"""
    wins = PROFILE_SPEC[profile]['windows']
    return int(wins[rng.randint(0, len(wins))])


def build_temporal_network(n_contacts=400, target_rate=0.25, random_state=42):
    """构建时序接触网络 v1（确定性）。

    结构：n_clusters 个簇（沿用 v3 骨架，β 全 1），每簇 1 指示病例
    （节点 id 0..M-1，携带 infectivity）+ 若干接触者（节点 id M..M+n-1）。
    标签只对接触者。

    流程：
      1. 宿主字段 + **时序画像**（recent/spread/old）逐接触者抽取；
      2. 接触者按 _JOIN_PROB 加入各类型簇（类型只作结构多样性，
         不进入标签机制）；
      3. 每个簇成员关系的**接触窗**从该接触者画像的窗口分布抽取；
         边（接触者-指示病例 + 同簇互连）落入对应窗的子图；同簇
         互连边的窗从两端画像之一（等概率）抽取；
      4. 暴露：temporal(i) = Σ decay(w)·intensity·infectivity（真值
         驱动）；cum(i) = Σ intensity·infectivity（个体可观测，无衰减）；
      5. 标签：p = 1 - exp(-k·temporal·host_mult)，k 二分校准至
         target_rate，Bernoulli 抽签。

    Args:
        n_contacts: 接触者数（另有 M 个指示病例节点）
        target_rate: 目标阳性率（k 校准，接触者口径）
        random_state: 随机种子

    Returns:
        dict: nodes（前 M 指示病例 + n_contacts 接触者）、labels
        （仅接触者）、M（指示病例数）、ef（{窗 id: {(u,v): 6 维特征}}）、
        profiles（接触者时序画像列表）、node_exposure_by_window
        （接触者 → {窗 id: 贡献量}）、k_calibration、n_clusters、
        contact_ids、spec
    """
    rng = np.random.RandomState(random_state)

    # ---- 1. 宿主字段 + 时序画像 ----
    contacts = [_node_host_fields(rng) for _ in range(n_contacts)]
    profiles = [_draw_profile(rng) for _ in range(n_contacts)]

    # ---- 2. 簇（沿用 v3 骨架；类型只作结构多样性，β 关闭）----
    n_clusters = max(2, n_contacts // _CONTACTS_PER_CLUSTER)
    index_cases = []
    clusters = []
    for _ in range(n_clusters):
        infectivity = float(rng.beta(2.0, 2.0))   # 涂阳/空洞综合传染性
        if rng.random() < _HOUSEHOLD_CLUSTER_SHARE:
            t = 'household'
        else:
            t = str(rng.choice(['workplace', 'school', 'social', 'casual']))
        index_cases.append({'infectivity': infectivity})
        clusters.append({'type': t, 'src': len(index_cases) - 1,
                         'infectivity': infectivity,
                         'members': [], 'intens': {}, 'win': {}})

    # ---- 3. 接触者加入簇（家庭 + 0-4 个其他类型）----
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
            # 簇内强度相关（共享接触条件 0.5 + 个体差异 0.5）
            c['intens'][i] = float(np.clip(
                0.5 * c['infectivity'] + 0.5 * rng.random(), 0.05, 1.0))
            # 成员关系接触窗：从该接触者的时序画像抽取
            c['win'][i] = _draw_window(rng, profiles[i])

    # ---- 4. 边（按窗分桶：接触者-指示病例 + 同簇互连）----
    M = len(index_cases)
    cid = {i: M + i for i in range(n_contacts)}   # 接触者节点 id
    ef = {w: {} for w in range(NUM_WINDOWS)}      # (u,v) -> 6 维特征
    for c in clusters:
        for i in c['members']:
            u, v = cid[i], c['src']
            key = (min(u, v), max(u, v))
            ef[c['win'][i]][key] = _sample_edge_features(rng)
        mem = [cid[i] for i in c['members']]
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                i_orig, j_orig = c['members'][a], c['members'][b]
                # 互连边窗：从两端画像之一（等概率）抽取
                prof = (profiles[i_orig] if rng.random() < 0.5
                        else profiles[j_orig])
                w = _draw_window(rng, prof)
                key = (min(mem[a], mem[b]), max(mem[a], mem[b]))
                ef[w][key] = _sample_edge_features(rng)

    # ---- 5. 暴露量（时间信息不对称核心）----
    decay = np.asarray(DECAY_WEIGHTS, dtype=float)
    temporal = np.zeros(n_contacts)   # 真值驱动（含衰减）
    cum = np.zeros(n_contacts)        # 个体可观测（无衰减）
    node_exposure_by_window = [dict() for _ in range(n_contacts)]
    for c in clusters:
        for i in c['members']:
            contrib = c['intens'][i] * c['infectivity']
            w = c['win'][i]
            cum[i] += contrib
            temporal[i] += decay[w] * contrib
            d = node_exposure_by_window[i]
            d[w] = d.get(w, 0.0) + contrib

    # ---- 6. 宿主乘数 + k 二分校准 + Bernoulli 标签 ----
    mults = np.array([
        host_susceptibility_multiplier(
            age=h['age'], has_symptoms=h['has_symptoms'],
            has_tb=h['has_tb'], is_high_risk=h['is_high_risk'],
            bcg_vaccine=h['bcg_vaccine'],
            past_illness_type=h['past_illness_type'])
        for h in contacts])
    lo, hi = 1e-6, 80.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (1.0 - np.exp(-mid * temporal * mults)).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    k_cal = 0.5 * (lo + hi)
    p_label = 1.0 - np.exp(-k_cal * temporal * mults)
    labels = (rng.random(n_contacts) < p_label).astype(int)
    # 保证两类结局同时存在（DeLong 前提）
    if labels.sum() == 0:
        labels[int(np.argmax(p_label))] = 1
    elif labels.sum() == n_contacts:
        labels[int(np.argmin(p_label))] = 0

    # ---- 7. 节点记录 ----
    # 指示病例：传染性只存在于图（个体特征层面不可见）
    nodes = []
    for s in index_cases:
        nodes.append({
            'is_index_case': 1, 'infectivity': s['infectivity'],
            'age': 40, 'has_symptoms': 1, 'is_high_risk': 0,
            'has_tb': 1, 'bcg_vaccine': 1, 'past_illness_type': 'none',
            'cumulative_exposure': 0.0, 'temporal_exposure': 0.0,
            'mean_freq': 0.0, 'mean_duration': 0.0, 'mean_span': 0.0,
            'mean_vent': 0.0, 'mean_dist': 0.0, 'n_edges': 0,
            'host_multiplier': 1.0, 'profile': 'index',
        })
    # 接触者关联边特征聚合（不分窗——个体只能看到混合总量/均值）
    edge_agg = [[] for _ in range(n_contacts)]
    for w in range(NUM_WINDOWS):
        for (u, v) in ef[w]:
            feat = ef[w][(u, v)]
            if u >= M:
                edge_agg[u - M].append(feat)
            if v >= M:
                edge_agg[v - M].append(feat)
    for i in range(n_contacts):
        h = contacts[i]
        arr = np.asarray(edge_agg[i] or [[0.0] * EDGE_FEATURE_DIM],
                         dtype=float)
        rec = dict(h)
        rec.update({
            'is_index_case': 0, 'infectivity': 0.0,
            'cumulative_exposure': float(cum[i]),
            'temporal_exposure': float(temporal[i]),   # 真值（仅 DGP 内部/分析）
            'mean_freq': float(arr[:, 0].mean()),
            'mean_duration': float(arr[:, 1].mean()),
            'mean_span': float(arr[:, 2].mean()),
            'mean_vent': float(arr[:, 3].mean()),
            'mean_dist': float(arr[:, 4].mean()),
            'n_edges': int(len(edge_agg[i])),
            'host_multiplier': float(mults[i]),
            'profile': profiles[i],   # 时序画像（仅 DGP 内部/分析，不入特征）
        })
        nodes.append(rec)

    return {
        'nodes': nodes,
        'labels': labels,
        'M': M,
        'ef': ef,
        'profiles': profiles,
        'node_exposure_by_window': node_exposure_by_window,
        'k_calibration': float(k_cal),
        'n_clusters': n_clusters,
        'contact_ids': cid,
        'spec': TEMPORAL_NETWORK_SPEC,
    }


# ==============================================================================
# 特征构造（消融臂的公平特征预算）
# ==============================================================================

NODE_FEATURE_DIM = 12   # 节点特征维度（指示病例与接触者同构）


def node_features(net):
    """全图节点特征（M 指示病例 + n 接触者，12 维）。

    维度：是否指示病例 / 传染性（仅指示病例非零）/ 年龄 / 症状 /
    高危 / 既往 TB / BCG / 共病 / **累积暴露（无衰减总量）** /
    频率 / 时长 / 距离。

    接触者行的前两维恒为 0，且累积暴露为不分窗总量——个体特征不含
    时序信息（时间信息不对称上界）；指示病例行携带传染性（只有图
    可读）。时序画像（profile）不进入特征。
    """
    rows = []
    for n in net['nodes']:
        rows.append([
            float(n['is_index_case']),
            float(n['infectivity']) if n['is_index_case'] else 0.0,
            n['age'] / 80.0,
            float(n['has_symptoms']),
            float(n['is_high_risk']),
            float(n['has_tb']),
            float(n['bcg_vaccine']),
            1.0 if n['past_illness_type'] != 'none' else 0.0,
            min(n['cumulative_exposure'] / 3.0, 1.0),
            n['mean_freq'] / 20.0,
            n['mean_duration'] / 100.0,
            n['mean_dist'] / 3.0,
        ])
    return np.asarray(rows, dtype=np.float64)


def individual_features(net):
    """接触者个体特征矩阵（信息上界基线：无时序 / 无传染性）。"""
    return node_features(net)[net['M']:]


def _all_edges(net):
    """全窗边列表 [(u, v), ...]（不分窗——静态聚合语义）。"""
    edges = []
    for w in range(NUM_WINDOWS):
        edges.extend(net['ef'][w].keys())
    return edges


def _neighbor_agg(net, X, by_window=False):
    """邻居节点特征聚合（在 12 维节点特征上计算）。

    by_window=False → 跨窗等权邻居均值（**现行网络层语义**：所有
    接触不分时间一样权重——时序盲）；
    by_window=True  → 5 个窗各自独立邻居均值拼接（时序区分聚合，
    "window 臂"——时间信息的可学上界参考）。
    """
    n = X.shape[0]
    if by_window:
        chans = []
        for w in range(NUM_WINDOWS):
            nbrs = [[] for _ in range(n)]
            for (i, j) in net['ef'][w]:
                nbrs[i].append(j)
                nbrs[j].append(i)
            ch = np.zeros_like(X)
            for i in range(n):
                if nbrs[i]:
                    ch[i] = X[nbrs[i]].mean(axis=0)
            chans.append(ch)
        return np.concatenate(chans, axis=1)
    nbrs = [[] for _ in range(n)]
    for (i, j) in _all_edges(net):
        nbrs[i].append(j)
        nbrs[j].append(i)
    agg = np.zeros_like(X)
    for i in range(n):
        if nbrs[i]:
            agg[i] = X[nbrs[i]].mean(axis=0)
    return agg


def static_agg_features(net):
    """静态聚合特征：个体 + 跨窗等权邻居均值（无时序信息，现行语义）。"""
    M = net['M']
    ind = node_features(net)[M:]
    agg = _neighbor_agg(net, node_features(net))[M:]
    return np.concatenate([ind, agg], axis=1)


def window_agg_features(net):
    """分窗聚合特征：个体 + 5 窗独立邻居均值拼接（时序区分上界）。

    逻辑回归可直接读取"按窗分解的邻居暴露"（近期窗的传染源邻居
    vs 远期窗的）——这是不引入时序 GNN 时时间信息的最大可兑现
    口径，作为端到端 STGNN 的对照上界。
    """
    M = net['M']
    ind = node_features(net)[M:]
    ch = _neighbor_agg(net, node_features(net), by_window=True)[M:]
    return np.concatenate([ind, ch], axis=1)


def to_stgnn_graph(net, windows_merged=False):
    """把 DGP 网络转成 TemporalGNN 输入（torch 张量，全图节点）。

    窗序列按**最旧 → 最新**排序（窗 id 4 → 0）：GRU 逐步处理，
    最后一步的隐状态对应"当前风险"（时序风险曲线终点）。

    Args:
        net: build_temporal_network 输出
        windows_merged: True → 时序盲对照（全部窗的边并入单一静态
            子图，窗标注抹除 + 跨窗重复边去重），隔离"时间切片 +
            窗间状态传递"贡献

    Returns:
        dict: x [M+n, 12]、windows（list，最旧→最新，每项
        {'edge_index': LongTensor [2, E_w], 'edge_attr': FloatTensor
        [E_w, 6]}，FEAT_SCALE 归一化）、window_ids（对应窗 id 序列）
    """
    import torch

    x = torch.tensor(node_features(net), dtype=torch.float32)

    def _pack(pairs_feats):
        """[(u, v, feat)] → 双向边张量。"""
        if not pairs_feats:
            return (torch.zeros((2, 0), dtype=torch.long),
                    torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32))
        arr = np.array([p[0] for p in pairs_feats], dtype=np.int64)
        f = np.array([p[1] for p in pairs_feats],
                     dtype=np.float64) / FEAT_SCALE
        ei = torch.tensor(
            np.concatenate([arr, arr[:, ::-1]], axis=0).T, dtype=torch.long)
        ea = torch.tensor(
            np.concatenate([f, f], axis=0), dtype=torch.float32)
        return ei, ea

    windows = []
    window_ids = []
    if windows_merged:
        # 时序盲对照：跨窗去重（保留首个出现的窗），边数守恒于并集
        seen = set()
        merged = []
        for w in range(NUM_WINDOWS - 1, -1, -1):   # 最旧窗优先
            for key in sorted(net['ef'][w].keys()):
                if key in seen:
                    continue
                seen.add(key)
                merged.append((key, net['ef'][w][key]))
        ei, ea = _pack(merged)
        windows.append({'edge_index': ei, 'edge_attr': ea})
        window_ids.append(-1)   # 合并窗标记
    else:
        for w in range(NUM_WINDOWS - 1, -1, -1):   # id 4 (最旧) → 0 (最新)
            pairs = [(k, net['ef'][w][k])
                     for k in sorted(net['ef'][w].keys())]
            ei, ea = _pack(pairs)
            windows.append({'edge_index': ei, 'edge_attr': ea})
            window_ids.append(w)
    return {'x': x, 'windows': windows, 'window_ids': window_ids}
