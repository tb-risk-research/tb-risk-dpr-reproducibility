#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""类型化接触网络 DGP v3：传染源节点 + 同分布边特征的信息不对称构造。

背景（用户"第一层：异质边 + 注意力"，2026-08-25）：
  现行网络层是"所有邻居一样权重"的均值聚合。用户论点：网络增量的真实
  来源是**信息不对称**——个体特征只能观测"累积暴露总量"
  （cumulative_exposure），看不到"接触类型构成"；家庭密接与偶遇接触的
  传播风险差一个数量级，这一构成信息只存在于图的边类型里，可被类型
  感知注意力（GAT）利用，均值聚合不可。

DGP 迭代史（三轮设计失败的教训，2026-08-25）：
  - v1（群组全连接 + 边特征挂群组源强度）：typed−mean ≈ +0.0007，
    边特征不分层 → 无类型信号可学；
  - v2（边特征按类型分层）：个体特征成为"类型构成"的完美代理
    （AUROC 天花板 0.95+）→ 网络层无增量空间；
  - **v3（本模块，已验证）**：
    (a) **边特征全类型同分布**——个体从边特征聚合完全无法推断接触
        类型构成；
    (b) 类型差异只存在于 β（边类型标注，只有图能读的信息）；
    (c) 引入**传染源（指示病例）节点**：携带 infectivity（涂阳/空洞
        综合，0-1），只存在于图——接触者个体特征不可见；
    (d) 每簇 1 指示病例 + 若干接触者；家庭簇占比压至 25%（接触者
        类型构成多样化，压缩 cum/typed 冗余）。

本 DGP 构造（v3）：
  1. **5 种边类型**（文献校准基础传播率 β）：家庭 1.0 / 同事 0.35 /
     同学 0.30 / 社会 0.15 / 偶遇 0.05；
  2. **簇结构**：每簇 1 指示病例（infectivity ~ Beta(2,2)）+ 接触者；
     接触者-指示病例边 + 同簇接触者互连边，类型 = 簇类型；边特征
     6 维（频率/时长/周数/通风/距离/场景）全类型同分布；
  3. **个体可观测特征不含类型/传染性**：cumulative_exposure =
     Σ intensity·infectivity（不加 β）、边特征等权聚合；
  4. **真实风险（信息不对称载体）**：typed_exposure = Σ β_type ×
     intensity × infectivity(src)，标签 p = 1 - exp(-k × typed ×
     宿主乘数)，Bernoulli 抽签（概率抽签而非 top-k 排序选病，避免
     排序标签假象）；宿主乘数来自单一真值源
     validation/host_susceptibility.py；
  5. k 二分校准使接触者阳性率 = target_rate。

诚实边界（必须随产物声明）：类型构成不可观测是**构造性**设定，对应部署
系统 22 列特征的真实约束（无按场景分解的暴露特征）。本模块回答的是
"信息不对称存在时，类型注意力能否兑现为判别增量"（机制可行性），
不是"真实世界必然存在该不对称"（需 ERASE-TB 等真实数据裁决）。

文献：
  Fox GJ et al. (2013) PLoS ONE 8:e73212 —— 家庭密接感染率(约30-40%) vs
      社区接触(<5%)，相对梯度 6-10 倍；
  Andrews JR et al. (2012) PLoS ONE —— 接触者感染风险的时间-强度依赖；
  Vynnycky E, White R (2010) An Introduction to Infectious Disease
      Modelling —— 接触类型分层传播率设定。
"""

import numpy as np

from .host_susceptibility import host_susceptibility_multiplier


# ==============================================================================
# 边类型规格（文献校准，单一真值源）
# ==============================================================================

EDGE_TYPE_SPEC = [
    {'id': 'household', 'label': '家庭', 'base_transmission': 1.00,
     'reference': 'Fox 2013 PLoS ONE: 家庭密接感染率 30-40%'},
    {'id': 'workplace', 'label': '同事', 'base_transmission': 0.35,
     'reference': 'Vynnycky & White 2010: 职场接触分层传播率'},
    {'id': 'school', 'label': '同学', 'base_transmission': 0.30,
     'reference': 'Vynnycky & White 2010: 学校接触分层传播率'},
    {'id': 'social', 'label': '社会', 'base_transmission': 0.15,
     'reference': 'Fox 2013 PLoS ONE: 社区接触感染率 <5%'},
    {'id': 'casual', 'label': '偶遇', 'base_transmission': 0.05,
     'reference': 'Fox 2013 PLoS ONE: 偶遇传播可忽略'},
]
EDGE_TYPE_IDS = [e['id'] for e in EDGE_TYPE_SPEC]
BETA_BY_TYPE = {e['id']: e['base_transmission'] for e in EDGE_TYPE_SPEC}
EDGE_FEATURE_DIM = 6   # 频率/时长/周数/通风/距离/场景（对齐 HeteroGATLayer）

# 边特征归一化尺度（频率/时长/周数/通风/距离/场景）
FEAT_SCALE = np.array([20.0, 100.0, 12.0, 5.0, 3.0, 3.0])

TYPED_NETWORK_SPEC = {
    'name': 'typed_contact_network_v3',
    'label_mechanism':
        'p = 1 - exp(-k * Σ β_type·intensity·infectivity(src) × 宿主乘数), '
        'Bernoulli 抽签（仅接触者有标签）',
    'host_multiplier_source': 'validation/host_susceptibility.py（单一真值源）',
    'edge_types': EDGE_TYPE_SPEC,
    'info_asymmetry': (
        '边特征全类型同分布 + 类型差异只存在于 β 与边类型标注；'
        '传染源 infectivity 只存在于指示病例节点——三者均只有图可读，'
        '个体特征只能观测不分类型的累积暴露量'),
    'caveat': (
        '信息不对称为构造性设定（对齐部署系统 22 列特征无场景分解'
        '暴露特征的真实约束）；结论性质 = 机制可行性验证，非自然发现。'
        'v1 边特征不分层无信号、v2 边特征分层致个体代理天花板——'
        'v3 同分布边特征为唯一通过验证的构造'),
}

# ==============================================================================
# 网络结构参数（v3 验证值）
# ==============================================================================

_HOUSEHOLD_CLUSTER_SHARE = 0.25   # 家庭簇占比（压低以多样化类型构成）
_CLUSTER_CAP = 7                  # 每簇接触者容量上限
_CONTACTS_PER_CLUSTER = 12        # 簇数 = max(2, n_contacts // 12)
# 接触者加入各类型簇的概率（家庭 0.75；其余为 join 基率 × 0.75）
_JOIN_PROB = {
    'household': 0.75, 'workplace': 0.30, 'school': 0.1875,
    'social': 0.375, 'casual': 0.225,
}
# 边特征分布（v3 核心：全类型同分布，类型差异只在 β 与边类型标注）
_EDGE_PROFILE = {
    'freq': (3, 20),                    # 接触频率（次/周）
    'duration': (10, 100),              # 单次时长（分钟）
    'dist_weights': [0.3, 0.3, 0.25, 0.15],   # 距离秩 0(近)-3(远)
    'vent': (1, 5),                     # 通风 1(好)-5(差)
}


def _node_host_fields(rng):
    """宿主字段（分布口径对齐 layer_ablation v3 / DGP FIELD_SPECS）。"""
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

    v3 核心：**全类型同分布**——无论家庭边还是偶遇边，频率/时长/距离/
    通风的分布完全一致。个体从边特征聚合无法推断"接触类型构成"；
    类型差异只存在于 β 与边类型标注（只有图能读的信息）。
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


def build_typed_network(n_contacts=400, target_rate=0.25, random_state=42):
    """构建类型化接触网络 v3（确定性）。

    结构：n_clusters 个簇，每簇 1 指示病例（节点 id 0..M-1，携带
    infectivity）+ 若干接触者（节点 id M..M+n-1）。标签只对接触者。

    流程：
      1. 簇：类型（家庭 25% / 其余均分）、指示病例 infectivity ~ Beta(2,2)；
      2. 接触者按 _JOIN_PROB 加入各类型簇（每簇容量 _CLUSTER_CAP）；
      3. 边：接触者-指示病例 + 同簇接触者互连，类型 = 簇类型，
         边特征全类型同分布；
      4. 暴露：typed(i) = Σ β_t·intensity·infectivity（真值驱动）；
         cum(i) = Σ intensity·infectivity（个体可观测，无 β）；
      5. 标签：p = 1 - exp(-k·typed·host_mult)，k 二分校准至
         target_rate，Bernoulli 抽签。

    Args:
        n_contacts: 接触者数（另有 M 个指示病例节点）
        target_rate: 目标阳性率（k 校准，接触者口径）
        random_state: 随机种子

    Returns:
        dict: nodes（前 M 指示病例 + n_contacts 接触者）、labels
        （仅接触者，长度 n_contacts）、M（指示病例数）、ef
        （{类型: {(u,v): 6 维特征}}，节点 id 空间）、
        node_exposure_by_type（接触者 → {类型: 加权暴露}）、
        k_calibration、n_clusters、contact_ids、spec
    """
    rng = np.random.RandomState(random_state)

    # ---- 1. 宿主字段与簇（每簇 1 指示病例）----
    contacts = [_node_host_fields(rng) for _ in range(n_contacts)]
    n_clusters = max(2, n_contacts // _CONTACTS_PER_CLUSTER)
    index_cases = []      # 指示病例（传染源）：节点 id 0..M-1
    clusters = []
    for _ in range(n_clusters):
        infectivity = float(rng.beta(2.0, 2.0))   # 涂阳/空洞综合传染性
        if rng.random() < _HOUSEHOLD_CLUSTER_SHARE:
            t = 'household'
        else:
            t = str(rng.choice(['workplace', 'school', 'social', 'casual']))
        index_cases.append({'infectivity': infectivity,
                            'symptom_weeks': float(rng.uniform(1, 12))})
        clusters.append({'type': t, 'src': len(index_cases) - 1,
                         'infectivity': infectivity,
                         'members': [], 'intens': {}})

    # ---- 2. 接触者加入簇（家庭 + 0-4 个其他类型）----
    for i in range(n_contacts):
        for t in EDGE_TYPE_IDS:
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

    # ---- 3. 边（接触者-指示病例 + 同簇互连；特征全类型同分布）----
    M = len(index_cases)
    cid = {i: M + i for i in range(n_contacts)}   # 接触者节点 id
    ef = {t: {} for t in EDGE_TYPE_IDS}           # (u,v) -> 6 维特征
    for c in clusters:
        t = c['type']
        for i in c['members']:
            u, v = cid[i], c['src']
            key = (min(u, v), max(u, v))
            ef[t][key] = _sample_edge_features(rng)
        mem = [cid[i] for i in c['members']]
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                key = (min(mem[a], mem[b]), max(mem[a], mem[b]))
                ef[t][key] = _sample_edge_features(rng)

    # ---- 4. 暴露量（信息不对称核心）----
    typed = np.zeros(n_contacts)      # 真值驱动（含 β）
    cum = np.zeros(n_contacts)        # 个体可观测（无 β）
    node_exposure_by_type = [dict() for _ in range(n_contacts)]
    for c in clusters:
        for i in c['members']:
            contrib = c['intens'][i] * c['infectivity']
            cum[i] += contrib
            typed[i] += BETA_BY_TYPE[c['type']] * contrib
            d = node_exposure_by_type[i]
            d[c['type']] = d.get(c['type'], 0.0) + contrib

    # ---- 5. 宿主乘数 + k 二分校准 + Bernoulli 标签 ----
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
        if (1.0 - np.exp(-mid * typed * mults)).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    k_cal = 0.5 * (lo + hi)
    p_label = 1.0 - np.exp(-k_cal * typed * mults)
    labels = (rng.random(n_contacts) < p_label).astype(int)
    # 保证两类结局同时存在（DeLong 前提）
    if labels.sum() == 0:
        labels[int(np.argmax(p_label))] = 1
    elif labels.sum() == n_contacts:
        labels[int(np.argmin(p_label))] = 0

    # ---- 6. 节点记录 ----
    # 指示病例：传染性只存在于图（个体特征层面不可见）
    nodes = []
    for s in index_cases:
        nodes.append({
            'is_index_case': 1, 'infectivity': s['infectivity'],
            'symptom_weeks': s['symptom_weeks'], 'age': 40,
            'has_symptoms': 1, 'is_high_risk': 0, 'has_tb': 1,
            'bcg_vaccine': 1, 'past_illness_type': 'none',
            'cumulative_exposure': 0.0, 'typed_exposure': 0.0,
            'mean_freq': 0.0, 'mean_duration': 0.0, 'mean_span': 0.0,
            'mean_vent': 0.0, 'mean_dist': 0.0, 'n_edges': 0,
            'host_multiplier': 1.0,
        })
    # 接触者关联边特征聚合（不分类型——个体只能看到混合均值）
    edge_agg = [[] for _ in range(n_contacts)]
    for t in EDGE_TYPE_IDS:
        for (u, v) in ef[t]:
            feat = ef[t][(u, v)]
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
            'typed_exposure': float(typed[i]),   # 真值（仅 DGP 内部/分析）
            'mean_freq': float(arr[:, 0].mean()),
            'mean_duration': float(arr[:, 1].mean()),
            'mean_span': float(arr[:, 2].mean()),
            'mean_vent': float(arr[:, 3].mean()),
            'mean_dist': float(arr[:, 4].mean()),
            'n_edges': int(len(edge_agg[i])),
            'host_multiplier': float(mults[i]),
        })
        nodes.append(rec)

    return {
        'nodes': nodes,
        'labels': labels,
        'M': M,
        'ef': ef,
        'node_exposure_by_type': node_exposure_by_type,
        'k_calibration': float(k_cal),
        'n_clusters': n_clusters,
        'contact_ids': cid,
        'spec': TYPED_NETWORK_SPEC,
    }


# ==============================================================================
# 特征构造（消融臂的公平特征预算）
# ==============================================================================

NODE_FEATURE_DIM = 12   # 节点特征维度（指示病例与接触者同构）


def node_features(net):
    """全图节点特征（M 指示病例 + n 接触者，12 维）。

    维度：是否指示病例 / 传染性（仅指示病例非零）/ 年龄 / 症状 /
    高危 / 既往 TB / BCG / 共病 / 累积暴露 / 频率 / 时长 / 距离。

    接触者行的前两维恒为 0——个体特征不含类型与传染性信息
    （信息不对称上界）；指示病例行携带传染性（只有图可读）。
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
    """接触者个体特征矩阵（信息上界基线：无类型 / 无传染性）。"""
    return node_features(net)[net['M']:]


def _neighbor_agg(net, X, by_type=False):
    """邻居节点特征聚合（在 12 维节点特征上计算）。

    by_type=False → 跨类型等权邻居均值（现行网络层语义：所有邻居
    一样权重；多重关系邻居按关系数计权）；
    by_type=True  → 5 类型各自独立邻居均值拼接（类型区分聚合，
    "channel 臂"——类型信息的可学上界参考）。
    """
    n = X.shape[0]
    if by_type:
        chans = []
        for t in EDGE_TYPE_IDS:
            nbrs = [[] for _ in range(n)]
            for (i, j) in net['ef'][t]:
                nbrs[i].append(j)
                nbrs[j].append(i)
            ch = np.zeros_like(X)
            for i in range(n):
                if nbrs[i]:
                    ch[i] = X[nbrs[i]].mean(axis=0)
            chans.append(ch)
        return np.concatenate(chans, axis=1)
    nbrs = [[] for _ in range(n)]
    for t in EDGE_TYPE_IDS:
        for (i, j) in net['ef'][t]:
            nbrs[i].append(j)
            nbrs[j].append(i)
    agg = np.zeros_like(X)
    for i in range(n):
        if nbrs[i]:
            agg[i] = X[nbrs[i]].mean(axis=0)
    return agg


def mean_agg_features(net):
    """均值聚合特征：个体 + 等权邻居均值（无类型信息，现行语义）。"""
    M = net['M']
    ind = node_features(net)[M:]
    agg = _neighbor_agg(net, node_features(net))[M:]
    return np.concatenate([ind, agg], axis=1)


def channel_agg_features(net):
    """分通道聚合特征：个体 + 5 类型独立邻居均值拼接（类型区分上界）。

    逻辑回归可直接读取"按类型分解的邻居暴露"——这是不引入 GAT 时
    类型信息的最大可兑现口径，作为端到端 GAT 的对照上界。
    """
    M = net['M']
    ind = node_features(net)[M:]
    ch = _neighbor_agg(net, node_features(net), by_type=True)[M:]
    return np.concatenate([ind, ch], axis=1)


def to_gat_graph(net, edge_types_merged=False):
    """把 DGP 网络转成 HeteroGATLayer 输入（torch 张量，全图节点）。

    Args:
        net: build_typed_network 输出
        edge_types_merged: True → 同质对照（5 类边并入单一通道，
            注意力参数无类型区分），隔离"类型感知"贡献

    Returns:
        dict: x [M+n, 12]、edge_indices（5 或 1 个 LongTensor 2×E_k，
        双向）、edge_attrs（同数 FloatTensor E_k×6，FEAT_SCALE 归一化）
    """
    import torch

    x = torch.tensor(node_features(net), dtype=torch.float32)
    if edge_types_merged:
        idx_rows, feat_rows = [], []
        for t in EDGE_TYPE_IDS:
            keys = sorted(net['ef'][t].keys())
            if not keys:
                continue
            arr = np.array(keys, dtype=np.int64)              # (E, 2)
            f = np.array([net['ef'][t][k] for k in keys],
                         dtype=np.float64) / FEAT_SCALE
            idx_rows.append(np.concatenate([arr, arr[:, ::-1]], axis=0))
            feat_rows.append(np.concatenate([f, f], axis=0))
        ei = [torch.tensor(np.concatenate(idx_rows, axis=0).T,
                           dtype=torch.long)]
        ea = [torch.tensor(np.concatenate(feat_rows, axis=0),
                           dtype=torch.float32)]
    else:
        ei, ea = [], []
        for t in EDGE_TYPE_IDS:
            keys = sorted(net['ef'][t].keys())
            if not keys:
                ei.append(torch.zeros((2, 0), dtype=torch.long))
                ea.append(torch.zeros((0, EDGE_FEATURE_DIM),
                                      dtype=torch.float32))
                continue
            arr = np.array(keys, dtype=np.int64).T            # (2, E)
            f = np.array([net['ef'][t][k] for k in keys],
                         dtype=np.float64) / FEAT_SCALE
            ei.append(torch.tensor(
                np.concatenate([arr, arr[::-1]], axis=1), dtype=torch.long))
            ea.append(torch.tensor(
                np.concatenate([f, f], axis=0), dtype=torch.float32))
    return {'x': x, 'edge_indices': ei, 'edge_attrs': ea}
