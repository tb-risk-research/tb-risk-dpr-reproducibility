#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯 NumPy 轻量图卷积回退（问题三：无深度学习环境时的 GNN 分支兜底）。

当 torch / torch_geometric 缺失（GNN 环境 tier 2）或模型未训练时，
``NumPyGraphRiskPredictor`` 提供确定性的轻量图卷积推理：

  - 特征提取：与 ``ml/framework/_contact_features`` 同口径（累计暴露、频率、
    时长、通风、距离、场景、症状等），保证与完整 GNN 分支的输入语义一致。
  - 图传播：GCN 式对称归一化邻接 + PageRank 式迭代扩散（保留一跳邻居
    均值聚合，多跳传播），使接触网络结构信息参与风险估计。
  - 风险映射：sigmoid 概率化 → 0-100 风险分，与 ML/SEIR 输出尺度一致。

文献：
  Kipf & Welling (2017) ICLR — GCN 一阶邻居均值聚合与对称归一化
  Page et al. (1999) — PageRank 图上迭代传播
  WHO (2024) — 接触者追踪与暴露风险评估
"""

import numpy as np

# 特征权重（领域经验值，保持确定性；同 ML 分支的轻量启发式）
# 维度与 _feature_vector 一一对应
_FEATURE_WEIGHTS = np.array([
    1.6,   # 0  cumulative_exposure  累积暴露
    0.9,   # 1  freq_density         接触频率
    0.7,   # 2  single_duration      单次时长
    0.6,   # 3  time_span            持续周数
    0.8,   # 4  ventilation          通风（越小风险越高，权重取负）
    -0.8,  # 5  contact_distance     接触距离（越近风险越高）
    0.9,   # 6  exposure_setting     暴露场景
    0.6,   # 7  has_symptoms         有症状
    0.7,   # 8  is_high_risk         高危人群
    1.0,   # 9  has_tb               既往 TB
    0.4,   # 10 age/65               年龄
    0.3,   # 11 family_baseline      家庭接触基线（family=1 更高）
], dtype=np.float64)


def _safe_float(value, default=0.0):
    """安全转 float，失败返回 default。"""
    if value is None:
        return float(default)
    try:
        return float(value)
    except (ValueError, TypeError):
        return float(default)


def _is_yes(value):
    """宽松的"是"判断（与 utils._is_yes 同语义，保持模块自包含）。"""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() in ('1', '是', '有', '涂阳', 'yes', 'true')
    return False


def _setting_score(setting):
    """暴露场景评分（与 _contact_features 同口径）。"""
    setting_map = {
        '高危': 1.0, '一般': 0.5, '低危': 0.1,
        'crowded': 1.0, 'closed': 0.9, 'general': 0.6, 'outdoor': 0.3,
        'oilfield_camp': 0.85,
        '拥挤': 1.0, '密闭': 0.9, '户外': 0.3,
        '油田营地': 0.85, '油田': 0.85,
    }
    return setting_map.get(setting, 0.5)


def _distance_score(contact_distance):
    """接触距离评分（与 _contact_features 同口径）。"""
    dist_map = {
        '近': 1.0, '中等': 0.5, '远': 0.25, '很远': 0.1,
        'very_close': 1.0, 'close': 0.8, 'medium': 0.5,
        'far': 0.25, 'distant': 0.1,
        '极近': 1.0, '极远': 0.1,
    }
    return dist_map.get(contact_distance, 0.5)


def feature_vector(entry, contact_type='family'):
    """从接触者数据构建轻量特征向量（12 维，确定性与 _contact_features 同口径）。

    参数：
        entry: 接触者数据 dict
        contact_type: 'family' / 'social'

    返回：
        np.ndarray[12]
    """
    cumulative = _safe_float(entry.get('cumulative_exposure', 0))
    freq_density = _safe_float(entry.get(
        'freq_density', 14 if contact_type == 'family' else 2))
    single_duration = _safe_float(entry.get('single_duration', 30))
    time_span = _safe_float(entry.get('time_span', 4))
    ventilation = _safe_float(entry.get('ventilation', 3))
    setting = entry.get('exposure_setting', '一般')
    # 通风评分与接触距离、场景取"风险值"（数值越大风险越高）
    vent_risk = (5.0 - max(0.0, min(5.0, ventilation))) / 5.0

    age = _safe_float(entry.get('age', 30), default=30.0)
    age = max(0.0, age)

    return np.array([
        min(cumulative / 1000.0, 1.0),
        min(freq_density / 30.0, 1.0),
        min(single_duration / 120.0, 1.0),
        min(time_span / 24.0, 1.0),
        vent_risk,                                    # 通风差 → 风险高
        _distance_score(entry.get('contact_distance', '中等')),
        _setting_score(setting),
        1.0 if _is_yes(entry.get('has_symptoms', '否')) else 0.0,
        1.0 if _is_yes(entry.get('is_high_risk', '否')) else 0.0,
        1.0 if _is_yes(entry.get('has_tb', '否')) else 0.0,
        min(age / 65.0, 1.0),
        1.0 if contact_type == 'family' else 0.0,
    ], dtype=np.float64)


def _logistic(x):
    """数值稳定的 sigmoid。"""
    z = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _sym_normalized_adjacency(n_nodes, edges):
    """GCN 式对称归一化邻接矩阵 D^-1/2 A D^-1/2。

    参数：
        n_nodes: 节点数
        edges: list[(i, j)] 无向边（重复边自动去重，自环忽略）

    返回：
        np.ndarray[n_nodes, n_nodes]
    """
    A = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i, j in edges:
        if i == j or i < 0 or j < 0 or i >= n_nodes or j >= n_nodes:
            continue
        A[i, j] += 1.0
        A[j, i] += 1.0
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    deg_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    return (deg_inv_sqrt[:, None] * A) * deg_inv_sqrt[None, :]


class NumPyGraphRiskPredictor:
    """纯 NumPy 轻量图卷积风险预测器（GNN 分支兜底，确定性）。

    参数：
        random_state: 保留以兼容接口（本实现确定性，无随机采样）
        propagation_steps: PageRank 式传播迭代次数（默认 3）
        damping: 传播阻尼系数 (0, 1)，1 时退化为纯邻居聚合
    """

    def __init__(self, random_state=42, propagation_steps=3, damping=0.85):
        self.random_state = random_state
        self.propagation_steps = max(1, int(propagation_steps))
        self.damping = float(np.clip(damping, 0.0, 0.999))

    # ------------------------------------------------------------------
    # 单接触者风险（无网络传播，快速兜底）
    # ------------------------------------------------------------------

    def _node_risk(self, feat):
        """由特征向量计算风险分（0-100）。

        若特征向量含第 13 维 P_base 注入列，仅用前 12 维做权重打分
        （权重维度固定为 12，避免特征维度变化破坏确定性）。
        """
        core = np.asarray(feat, dtype=np.float64)[: len(_FEATURE_WEIGHTS)]
        logit = float(np.dot(_FEATURE_WEIGHTS, core))
        return float(_logistic(logit) * 100.0)

    def predict_contact_risk(self, entry, contact_type='family', p_base=None):
        """对单个接触者预测风险（不使用网络传播）。

        参数：
            entry: 接触者数据 dict
            contact_type: 'family' / 'social'
            p_base (float|None): 0-100 个体基线概率（第 1 层输出）。
                传入时按"残差增量"语义输出：网络增量 = 网络风险 − 个体基线。

        返回：
            dict: 含 risk_probability (0-100) / risk_class / backend / network_aware
        """
        feat = feature_vector(entry, contact_type)
        risk = self._node_risk(feat)
        result = {
            'risk_probability': float(np.clip(risk, 0.0, 100.0)),
            'risk_class': 1 if risk > 50 else 0,
            'model_name': 'NumPy 轻量图卷积(回退)',
            'model_name_en': 'NumPy Lightweight GraphConv (fallback)',
            'network_aware': False,
            'backend': 'numpy',
            'gnn_used': True,
        }
        if p_base is not None:
            p_base = float(p_base)
            result['baseline_probability'] = p_base
            result['network_increment'] = float(
                np.clip(risk, 0.0, 100.0)) - p_base
            result['residual_learning'] = True
        return result

    # ------------------------------------------------------------------
    # 网络感知风险（构建接触网络 + PageRank 式传播）
    # ------------------------------------------------------------------

    def _build_feature_matrix(self, patient_info, family_entries, social_entries,
                              p_base=None):
        """构建节点特征矩阵与图结构。

        节点顺序: [患者, family_0..n, social_0..m, region]
        边: 患者↔所有接触者; 家庭成员间全连接（共居）; 区域↔所有个体（全局上下文）。

        参数：
            patient_info: 患者信息 dict
            family_entries / social_entries: 接触者列表
            p_base (float|None): 0-100 个体基线概率。传入时把第 1 层输出的
                P_base 作为**第 13 维节点特征**注入到每个接触者节点（堆叠：
                第 2 层以第 1 层输出为特征），供残差学习/未来训练使用；
                打分时仍用前 12 维（见 ``_node_risk``）。

        返回：
            (X: np.ndarray[N, 12 或 13], edges: list[(i,j)], offsets: dict)
        """
        nodes = []
        edges = []

        # 患者节点（特征：基础风险 + 涂阳/空洞，作为源节点）
        basic = (patient_info or {}).get('basic_info', {})
        sputum_positive = 1.0 if basic.get('sputum_smear', 1) == 2 else 0.0
        has_cavity = 1.0 if basic.get('has_cavity', 1) == 2 else 0.0
        patient_feat = np.zeros(12, dtype=np.float64)
        patient_feat[6] = 0.5 + 0.3 * sputum_positive + 0.2 * has_cavity
        patient_feat[7] = 1.0  # 患者默认有症状（传染源）
        nodes.append(patient_feat)

        family_entries = family_entries or []
        social_entries = social_entries or []
        n_family = len(family_entries)
        n_social = len(social_entries)

        for e in family_entries:
            nodes.append(feature_vector(e, 'family'))
        for e in social_entries:
            nodes.append(feature_vector(e, 'social'))

        # 区域级节点（多尺度特征融合 v5.0：个体—家庭—社区—区域）
        # 12 维特征承载区域流行病学/政策强度，作为全局上下文参与传播。
        n_contacts = n_family + n_social
        region_feat = np.zeros(12, dtype=np.float64)
        region_feat[6] = 0.5 + 0.3 * float(n_contacts > 0)   # 暴露场景/规模
        region_feat[4] = 0.5                                  # 区域内通风基线
        nodes.append(region_feat)
        region_idx = len(nodes) - 1

        # 第 1 层 P_base 注入：作为第 13 维节点特征（接触者节点 = p_base/100，
        # 患者/区域节点取 0）。仅在显式传入 p_base 时注入，保持向后兼容。
        if p_base is not None:
            p_norm = float(np.clip(p_base, 0.0, 100.0)) / 100.0
            nodes = [np.append(n, p_norm if i != 0 and i != region_idx else 0.0)
                     for i, n in enumerate(nodes)]

        n_nodes = len(nodes)
        # 患者↔所有接触者
        for k in range(1, n_nodes - 1):
            edges.append((0, k))
        # 家庭成员间共居全连接
        for i in range(n_family):
            for j in range(i + 1, n_family):
                edges.append((1 + i, 1 + j))
        # 区域 ↔ 所有个体节点（全局上下文，模拟"社区→区域"聚合）
        for k in range(0, n_nodes - 1):
            edges.append((region_idx, k))

        offsets = {
            'patient': 0,
            'family': slice(1, 1 + n_family),
            'social': slice(1 + n_family, n_nodes - 1),
            'region': slice(region_idx, region_idx + 1),
        }
        return np.stack(nodes), edges, offsets

    def predict_network_risk(self, entry, contact_type='family',
                             patient_info=None, family_entries=None,
                             social_entries=None, p_base=None):
        """基于完整接触网络预测单个接触者风险（PageRank 式传播）。

        参数：
            entry: 目标接触者数据 dict
            contact_type: 'family' / 'social'
            patient_info: 患者信息 dict（含 basic_info）
            family_entries / social_entries: 全部家庭/社会接触者列表
            p_base (float|None): 0-100 个体基线概率（第 1 层输出）。
                传入时注入为第 13 维节点特征，并输出"网络增量"（残差语义）。

        返回：
            dict: 含 risk_probability (0-100) / risk_class / backend / network_aware
        """
        X, edges, offsets = self._build_feature_matrix(
            patient_info, family_entries, social_entries, p_base=p_base)
        n_nodes = X.shape[0]

        # 目标节点索引
        target_idx = None
        if contact_type == 'family' and offsets['family'].stop > offsets['family'].start:
            candidates = range(offsets['family'].start, offsets['family'].stop)
        elif contact_type == 'social' and offsets['social'].stop > offsets['social'].start:
            candidates = range(offsets['social'].start, offsets['social'].stop)
        else:
            candidates = []
        # 用 record_id/_id 优先精确匹配，否则匹配特征最近的节点
        if entry is not None:
            rid = entry.get('record_id') if isinstance(entry, dict) else None
            mid = entry.get('_id') if isinstance(entry, dict) else None
            entries = family_entries if contact_type == 'family' else social_entries
            for offset, lst in zip((offsets['family'].start, offsets['social'].start),
                                   (family_entries, social_entries)):
                for i, e in enumerate(lst or []):
                    if rid is not None and e.get('record_id') == rid:
                        target_idx = offset + i
                        break
                    if mid is not None and e.get('_id') == mid:
                        target_idx = offset + i
                        break
                if target_idx is not None:
                    break
        if target_idx is None and len(candidates) > 0:
            # 特征匹配：与目标特征最接近的候选节点（P_base 注入列仅作用于
            # 接触者节点，比较时按接触者列取前 12 维，避免维度不一致）
            feat = feature_vector(entry, contact_type)
            target_idx = min(candidates,
                             key=lambda k: np.abs(X[k][: len(feat)] - feat).sum())

        if target_idx is None:
            # 网络无该类型节点时退化为单节点估计
            return self.predict_contact_risk(entry, contact_type, p_base=p_base)

        # GCN 式传播（PageRank 风格迭代扩散）
        A_norm = _sym_normalized_adjacency(n_nodes, edges)
        H = X.copy()
        damp = self.damping
        for _ in range(self.propagation_steps):
            H = (1.0 - damp) * X + damp * (A_norm @ H)

        risk = self._node_risk(H[target_idx])
        result = {
            'risk_probability': float(np.clip(risk, 0.0, 100.0)),
            'risk_class': 1 if risk > 50 else 0,
            'model_name': 'NumPy 轻量图卷积(回退)',
            'model_name_en': 'NumPy Lightweight GraphConv (fallback)',
            'network_aware': True,
            'node_idx': int(target_idx),
            'backend': 'numpy',
            'gnn_used': True,
        }
        if p_base is not None:
            p_base = float(p_base)
            result['baseline_probability'] = p_base
            result['network_increment'] = float(
                np.clip(risk, 0.0, 100.0)) - p_base
            result['residual_learning'] = True
        return result

    # ------------------------------------------------------------------
    # 统一入口（兼容 predict_gnn_risk 调用方）
    # ------------------------------------------------------------------

    def predict_cluster_risk(self, cluster_entries, target_index,
                             contact_type='family', p_base=None):
        """对簇内目标节点做真实 GCN 聚合传播（网络增强层残差来源）。

        与 ``predict_network_risk`` 的差异：
          - 不引入患者/区域节点，节点 = 簇内全部接触者（共享传染源）；
          - 簇内全连接（同簇成员彼此为邻居，构成"共享传染源"网络）；
          - 传播后目标节点特征 = 个体特征 + 邻居特征聚合（GCN 语义），
            使网络层能从**特征层面**恢复簇级隐藏风险，而非读取标签。

        参数：
            cluster_entries (list[dict]): 同簇接触者记录（含目标节点）
            target_index (int): 目标节点在 cluster_entries 中的下标
            contact_type (str): 接触类型（family/social）
            p_base (float|None): 0-100 个体基线概率（第 1 层输出）。
                传入时注入为第 13 维节点特征（堆叠），并输出"网络增量"。

        返回：
            dict: 含 risk_probability (0-100) / network_aware=True /
                  network_increment（残差语义）
        """
        n = len(cluster_entries)
        if n <= 1:
            return self.predict_contact_risk(
                cluster_entries[0] if cluster_entries else {},
                contact_type, p_base=p_base)

        X = np.stack([feature_vector(e, contact_type) for e in cluster_entries])
        # 簇内全连接（同簇共享传染源 → 成员互为邻居）
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

        # 第 1 层 P_base 堆叠（残差学习）：作为第 13 维节点特征
        if p_base is not None:
            p_norm = float(np.clip(p_base, 0.0, 100.0)) / 100.0
            p_col = np.full((n, 1), p_norm, dtype=np.float64)
            X = np.hstack([X, p_col])

        # GCN 式传播（PageRank 风格迭代扩散）
        A_norm = _sym_normalized_adjacency(n, edges)
        H = X.copy()
        damp = self.damping
        for _ in range(self.propagation_steps):
            H = (1.0 - damp) * X + damp * (A_norm @ H)

        risk = self._node_risk(H[target_index])
        result = {
            'risk_probability': float(np.clip(risk, 0.0, 100.0)),
            'risk_class': 1 if risk > 50 else 0,
            'model_name': 'NumPy 轻量图卷积(回退)',
            'model_name_en': 'NumPy Lightweight GraphConv (fallback)',
            'network_aware': True,
            'node_idx': int(target_index),
            'backend': 'numpy',
            'gnn_used': True,
        }
        if p_base is not None:
            p_base = float(p_base)
            result['baseline_probability'] = p_base
            result['network_increment'] = float(
                np.clip(risk, 0.0, 100.0)) - p_base
            result['residual_learning'] = True
        return result

    def predict(self, entry, contact_type='family', patient_info=None,
                family_entries=None, social_entries=None, p_base=None):
        """统一预测入口：有网络信息时用网络传播，否则退化为单节点估计。"""
        has_network = (family_entries or social_entries)
        if has_network:
            return self.predict_network_risk(
                entry, contact_type, patient_info, family_entries,
                social_entries, p_base=p_base)
        return self.predict_contact_risk(entry, contact_type, p_base=p_base)


# 模块级便捷实例（确定性默认配置）
_default_predictor = NumPyGraphRiskPredictor()


def predict_risk_numpy(entry, contact_type='family', patient_info=None,
                       family_entries=None, social_entries=None):
    """模块级便捷函数（幂等、确定性）。"""
    return _default_predictor.predict(
        entry, contact_type, patient_info, family_entries, social_entries)
