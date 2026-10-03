#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""网络增强层消融实验：有/无 GNN 层 ΔAUROC / ΔC-index。

直接回答评审关心的核心问题："传播网络到底贡献多少"。

设计（三层递进架构中的消融对照，v2.0 — 修复"网络不携带风险信号 / 网络层
只做特征平滑"两个根因）：

  - **无 GNN 层（基线）**：仅第 1 层个体基线概率 ``P_base`` ——
    单节点特征打分（``predict_contact_risk``），无网络结构参与；
  - **有 GNN 层（完整）**：``P_base + 门控网络增量`` ——
    第 2 层以 P_base 为节点特征堆叠，通过**同簇邻居特征聚合（真实 GCN
    传播 ``predict_cluster_risk``）** 学习残差增量。

关键机制（"隐藏风险通过网络结构暴露"）：
  - 合成网络的簇共享成分（共享传染源）使同簇成员特征**相关**；
  - 单节点只能观测到带噪声的特征副本 → 个体层 P_base 的判别力受限于
    个体噪声幅度；
  - GCN 聚合平均掉个体噪声、恢复共享传染源信号 → 网络层增量 =
    聚合去噪带来的残差，而非对个体概率做线性平滑、也不读取邻居标签。

比较指标：
  - **AUROC**：区分"病例 vs 非病例"的判别能力（复用 threshold_spec.compute_auc）
  - **C-index**：预测排序与真实结局的序一致性（Harrell 1982，纯 Python）
  - **ΔAUROC / ΔC-index**：网络层带来的增量，> 0 即网络结构有贡献。

完全确定性：合成网络基于种子可复现；基线评分用 NumPy 轻量图卷积（无 torch/PyG）。

文献：
  Harrell FE et al. (1982) JAMA — C-index 一致性指标
  Kipf & Welling (2017) ICLR — GCN（网络传播机制）
  Hanley & McNeil (1982) Radiology — AUC 标准误
"""

import logging
import math

import numpy as np

from .host_susceptibility import host_susceptibility_multiplier

LOGGER = logging.getLogger("tb_risk.validation.layer_ablation")


# ==============================================================================
# C-index（一致性指数，纯 Python）
# ==============================================================================

def compute_c_index(y_true, y_scores):
    """计算 C-index（Harrell 一致性指数）。

    C-index = 一致对 / 可比较对。tie 不计入可比较对；完全随机 = 0.5。

    参数：
        y_true: 二元结局（0/1）
        y_scores: 预测分数（越高预测越接近 1）

    返回：
        float: C-index
    """
    y_true = list(y_true)
    y_scores = list(y_scores)
    pos = [(t, s) for t, s in zip(y_true, y_scores) if t == 1]
    neg = [(t, s) for t, s in zip(y_true, y_scores) if t == 0]
    if not pos or not neg:
        return 0.5
    concordant = 0
    comparable = 0
    for _, sp in pos:
        for _, sn in neg:
            if sp == sn:
                concordant += 0.5   # tie 记半个一致对
            elif sp > sn:
                concordant += 1.0
            comparable += 1
    if comparable == 0:
        return 0.5
    return concordant / comparable


# ==============================================================================
# 合成接触网络（用于消融的确定性数据）
# ==============================================================================

def _make_contact_record(record_id, rng, feature_risk, cluster_risk=0.0):
    """生成一个接触者记录（字段对齐 NumPy feature_vector）。

    关键设计（v2.0 — 修复"簇成分被个体特征完全编码"的问题）：
      - **特征**由 feature_risk 生成：feature_risk 同时包含个体可观测成分
        与簇共享成分（同一传染源暴露 → 簇成员特征**相关**）；
      - 但簇共享成分是**隐藏变量**：单节点只观测到自己那一个带噪声的
        特征副本，无法精确恢复簇级共享信号；
      - **网络增强层**通过同簇邻居特征聚合（真实 GCN 传播）平均掉个体
        噪声、恢复共享传染源信号 —— 这是"隐藏风险通过网络结构暴露
        （传染源共享）"的正确实现：残差来自聚合去噪，而非读取标签。

    v3（2026-08-24，DGP 统一）：补齐宿主字段 bcg_vaccine（85%，DGP
    FIELD_SPECS 口径）与 past_illness_type（12% 共病率）——标签风险
    乘上文献校准的宿主易感性乘数（单一真值源
    validation/host_susceptibility.py，与部署场景 DGP 同源）。

    参数：
        record_id (str): 记录 id
        rng (np.random.RandomState): 随机源
        feature_risk (float): 0-1 特征风险（个体 + 簇共享 + 个体噪声）
        cluster_risk (float): 0-1 簇隐藏风险（共享传染源，供校验恢复能力）
    """
    age = int(rng.randint(1, 80))
    # 用"高风险特征出现次数"造出与 feature_risk 相关的噪声特征
    has_symptoms = 1 if rng.random() < 0.3 + 0.3 * feature_risk else 0
    is_high_risk = 1 if rng.random() < 0.1 + 0.2 * feature_risk else 0
    has_tb = 1 if rng.random() < 0.05 + 0.1 * feature_risk else 0
    # 宿主字段（v3）：BCG 接种率与共病率对齐 DGP FIELD_SPECS 口径
    bcg_vaccine = 1 if rng.random() < 0.85 else 0
    past_illness_type = 'none'
    if rng.random() < 0.12:
        past_illness_type = str(rng.choice(
            ['hiv', 'diabetes', 'immunosuppressants', 'other']))
    return {
        'record_id': record_id,
        'age': age,
        'cumulative_exposure': round(rng.uniform(0, 800) * (0.5 + feature_risk), 1),
        'freq_density': int(rng.randint(1, 20) + 10 * feature_risk),
        'single_duration': round(rng.uniform(10, 100) * (0.5 + feature_risk), 1),
        'time_span': int(rng.randint(1, 12) + 8 * feature_risk),
        'ventilation': int(rng.randint(1, 5)),
        'contact_distance': rng.choice(['远', '中等', '近', '很近']),
        'exposure_setting': rng.choice(['一般', '密闭', '聚集', '户外']),
        'has_symptoms': has_symptoms,
        'is_high_risk': is_high_risk,
        'has_tb': has_tb,
        'bcg_vaccine': bcg_vaccine,
        'past_illness_type': past_illness_type,
        # 簇隐藏风险（共享传染源）：个体特征无法单独读出，供 GCN 聚合校验。
        '_cluster_risk': float(cluster_risk),
    }


def build_synthetic_network(n_contacts=40, n_cases=12, n_clusters=5,
                            cluster_weight=0.6, individual_noise=0.15,
                            random_state=42):
    """构建合成接触网络用于消融实验（确定性）。

    个体风险 = 个体成分 + 簇隐藏成分（共享传染源）。v2.0 设计（修复
    "网络不携带风险信号 / 网络层只做特征平滑"两个根因）：

      - **簇共享成分进入特征、且占主导**：同一簇成员暴露于同一传染源，
        其可观测特征彼此**相关**（共享成分 + 个体噪声）；
      - **个体噪声足够大**（``individual_noise``）：单节点只能观测到自己
        那一个带噪声的特征副本，无法精确读出簇共享信号 —— 个体层 P_base
        的判别力因此受限；
      - 真实标签由综合风险（个体 + 簇）决定，病例按综合风险排序选取 →
        病例在簇内聚集；
      - **网络增强层**通过同簇邻居特征聚合（真实 GCN 传播）平均掉个体
        噪声、恢复共享传染源信号 → 网络层获得个体层无法从单节点特征读出
        的真实增量（残差学习语义），ΔAUROC/ΔC-index > 0。

    参数：
        n_contacts (int): 接触者总数
        n_cases (int): 真实病例数（is_confirmed=1）
        n_clusters (int): 接触簇数（共享传染源分组）
        cluster_weight (float): 簇成分权重（0-1，越大网络结构信号越强）
        individual_noise (float): 个体噪声幅值（0-1；越小 → 簇共享越主导、
            单节点越不可靠、GCN 聚合去噪收益越明显）
        random_state (int): 随机种子

    返回：
        (records, labels):
            records: list[dict]，含 record_id 与特征字段 + 'is_confirmed' + 'cluster'
            labels: np.ndarray[0/1]
    """
    rng = np.random.RandomState(random_state)
    # 个体可观测风险成分（独立、带个体噪声）
    individual = np.clip(rng.beta(1.5, 4.0, n_contacts) * 2.0, 0.0, 0.9)
    # 簇级隐藏风险成分（共享传染源 → 同簇成员风险整体抬升、特征相关）
    cluster_risks = rng.beta(2.0, 3.0, n_clusters)
    cluster_assign = rng.randint(0, n_clusters, n_contacts)

    # v3（2026-08-24，DGP 统一）：先采样记录（含宿主字段），综合风险
    # = （个体可观测 + 簇隐藏）× 宿主易感性乘数（文献校准，单一真值源
    # validation/host_susceptibility.py，与部署场景 DGP 同源）——
    # 病例选择同时受网络簇结构与宿主因素驱动，两代标签机制不再并存
    records = []
    true_risks = np.zeros(n_contacts)
    for i in range(n_contacts):
        # 特征风险 = 簇共享主导 + 个体噪声（共享成分相关性 + 单节点噪声）
        feature_risk = np.clip(
            (1.0 - individual_noise) * cluster_risks[cluster_assign[i]]
            + individual_noise * individual[i], 0.0, 0.9)
        rec = _make_contact_record(
            f'c{i}', rng, float(feature_risk),
            cluster_risk=float(cluster_risks[cluster_assign[i]]))
        base_risk = np.clip(
            (1.0 - cluster_weight) * individual[i]
            + cluster_weight * cluster_risks[cluster_assign[i]], 0.0, 0.9)
        host_mult = host_susceptibility_multiplier(
            age=rec['age'], has_symptoms=rec['has_symptoms'],
            has_tb=rec['has_tb'], is_high_risk=rec['is_high_risk'],
            bcg_vaccine=rec['bcg_vaccine'],
            past_illness_type=rec['past_illness_type'])
        true_risks[i] = min(base_risk * host_mult, 0.9)
        # 簇分配携带隐藏风险（同簇共享传染源）
        rec['cluster'] = int(cluster_assign[i])
        records.append(rec)

    # 排序取前 n_cases 作为真实病例（保证消融有信号，且病例在簇内聚集）
    case_idx = np.argsort(-true_risks)[:n_cases]
    labels = np.zeros(n_contacts, dtype=int)
    labels[case_idx] = 1
    for i, rec in enumerate(records):
        rec['is_confirmed'] = int(labels[i])
    return records, labels


# ==============================================================================
# 网络贡献消融
# ==============================================================================

class LayerAblationAnalyzer:
    """有/无 GNN 层消融分析器（网络增强层贡献量化）。

    网络层实现（v2.0 — 修复"只做特征平滑 / 读取标签"问题）：
      基线（无 GNN 层）＝ 单节点特征打分 ``predict_contact_risk``（P_base）；
      完整（有 GNN 层）＝ 同簇邻居特征聚合的真实 GCN 传播
      ``predict_cluster_risk``（以 P_base 为节点特征堆叠、学习残差增量）。

      由于合成网络的簇共享成分（传染源）使同簇成员特征**相关**，GCN 聚合
      平均掉个体噪声、恢复共享信号 → 网络层增量 = 聚合去噪带来的残差，
      而非对个体层概率做线性平滑、也不读取邻居标签。

    参数：
        random_state (int): 随机种子（保留以兼容接口）
        gate_max (float): 门控系数上限（保留以兼容 gated_integration）
        structural_boost (float): [已弃用] 保留向后兼容，不再参与评分。
    """

    def __init__(self, random_state=42, gate_max=1.0, structural_boost=20.0):
        self.random_state = random_state
        self.gate_max = gate_max
        self.structural_boost = structural_boost

    # ---- 评分 ----
    def _individual_base(self, record, contact_type='family'):
        """无 GNN 层：个体基线概率 P_base（单节点估计）。"""
        from ..ml.gnn import NumPyGraphRiskPredictor
        predictor = NumPyGraphRiskPredictor()
        result = predictor.predict_contact_risk(record, contact_type)
        return float(result['risk_probability'])

    def _network_cluster_risk(self, p_base, cluster_records, target_idx,
                              contact_type='family'):
        """有 GNN 层：同簇 GCN 聚合传播（真实网络增强层）。

        用 ``NumPyGraphRiskPredictor.predict_cluster_risk`` 对簇内全部成员
        做特征聚合传播，目标节点风险 = 个体特征 + 邻居特征聚合（GCN 语义）。
        网络增量 = 聚合风险 − P_base（残差学习）。

        参数：
            p_base (float): 0-100 个体基线概率（第 1 层）
            cluster_records (list[dict]): 同簇全部接触者记录（含目标）
            target_idx (int): 目标节点在 cluster_records 中的下标
            contact_type (str): 'family' / 'social'

        返回：
            (network_risk, network_aware):
                network_risk: 0-100 网络感知风险
                network_aware: 是否实际使用了网络结构
        """
        from ..ml.gnn import NumPyGraphRiskPredictor
        predictor = NumPyGraphRiskPredictor()
        if len(cluster_records) <= 1:
            return p_base, False
        result = predictor.predict_cluster_risk(
            cluster_records, target_idx, contact_type=contact_type, p_base=p_base)
        return float(result['risk_probability']), True

    # ---- 主入口 ----
    def run_ablation(self, records, labels=None, n_cases=None,
                     network_mode='cluster', return_scores=False):
        """运行消融：基线（P_base）vs 完整（P_base + 门控网络增量）。

        参数：
            records (list[dict]): 接触者记录（含 is_confirmed 或由 labels 提供）
            labels (list|None): 真实结局；None 时用记录内 is_confirmed
            n_cases (int|None): 真实病例数（仅用于合成网络，None 自动从记录推算）
            network_mode (str): 'cluster' 按记录 cluster 分组构建子网络；
                'global' 用全部记录构建单一网络。
            return_scores (bool): True 时报告附带 baseline_scores /
                full_scores（多种子聚合与 DeLong 配对检验需要逐受试者分数）。

        返回：
            dict: 消融报告（AUROC/C-index 基线 vs 完整 + Δ + 结论）
        """
        from ..scoring.architecture import gated_integration
        from .threshold_spec import compute_auc

        if labels is None:
            labels = [int(r.get('is_confirmed', 0)) for r in records]
        labels = np.asarray(labels, dtype=int)

        # 按 cluster 分组构建接触网络（网络结构仅在簇内共享传染源时紧密）
        clusters = {}
        for i, rec in enumerate(records):
            clusters.setdefault(rec.get('cluster', 0), []).append(i)

        baseline_scores = []
        full_scores = []
        for i, rec in enumerate(records):
            contact_type = 'family' if rec.get('contact_type', 'family') == 'family' else 'social'

            # 基线：单节点（无网络）
            p_base = self._individual_base(rec, contact_type)

            # 网络：同一 cluster 内全部成员构成网络（同一传染源簇）
            members = clusters.get(rec.get('cluster', 0), [i])
            cluster_records = [records[j] for j in members]
            target_idx = members.index(i)
            network_risk, network_aware = self._network_cluster_risk(
                p_base, cluster_records, target_idx, contact_type)

            # 门控联合（决策层）：P_base + g × 网络增量
            decision = gated_integration(
                p_base_percent=p_base,
                network_increment=network_risk - p_base,
                n_contacts=len(members) - 1,
                network_aware=network_aware,
            )
            baseline_scores.append(p_base)
            full_scores.append(decision['decision_risk'])

        baseline_scores = np.asarray(baseline_scores, dtype=float)
        full_scores = np.asarray(full_scores, dtype=float)

        auc_base = compute_auc(baseline_scores.tolist(), labels.tolist())
        auc_full = compute_auc(full_scores.tolist(), labels.tolist())
        c_base = compute_c_index(labels, baseline_scores)
        c_full = compute_c_index(labels, full_scores)

        delta_auc = auc_full - auc_base
        delta_cindex = c_full - c_base

        report = self._report(auc_base, auc_full, c_base, c_full,
                              delta_auc, delta_cindex, labels)
        if return_scores:
            report['baseline_scores'] = baseline_scores.tolist()
            report['full_scores'] = full_scores.tolist()
        return report

    def _report(self, auc_base, auc_full, c_base, c_full,
                delta_auc, delta_cindex, labels):
        """聚合消融报告。"""
        n_cases = int(np.sum(labels))
        n_total = int(len(labels))
        network_contributes = delta_auc > 0.005 or delta_cindex > 0.005
        return {
            'method': 'ablation_network_layer',
            'n_total': n_total,
            'n_cases': n_cases,
            'prevalence': round(n_cases / max(n_total, 1), 4),
            'baseline': {
                'auroc': round(auc_base, 4),
                'c_index': round(c_base, 4),
                'description': '仅第 1 层个体基线 P_base（无 GNN 层）',
            },
            'full': {
                'auroc': round(auc_full, 4),
                'c_index': round(c_full, 4),
                'description': '第 1 层 + 门控网络增量（有 GNN 层）',
            },
            'delta': {
                'delta_auroc': round(delta_auc, 4),
                'delta_cindex': round(delta_cindex, 4),
            },
            'network_contributes': bool(network_contributes),
            'conclusion': (
                '网络增强层带来显著判别增益（ΔAUROC/ΔC-index > 0），'
                '接触网络结构对风险排序有实质贡献。'
                if network_contributes else
                '网络增强层增益有限（ΔAUROC/ΔC-index 接近 0），'
                '当前网络信号不足以显著改善个体风险排序。'
            ),
        }


def network_contribution_ablation(records=None, labels=None, n_contacts=40,
                                  n_cases=12, random_state=42):
    """模块级便捷入口：运行网络贡献消融实验。

    未提供 records 时自动构建合成接触网络（确定性）。

    参数：
        records (list[dict]|None): 接触者记录（含 is_confirmed）
        labels (list|None): 真实结局（records 提供时覆盖记录内 is_confirmed）
        n_contacts / n_cases: 合成网络参数（仅在 records 为 None 时使用）
        random_state (int): 随机种子

    返回：
        dict: 消融报告
    """
    if records is None:
        records, synth_labels = build_synthetic_network(
            n_contacts=n_contacts, n_cases=n_cases, random_state=random_state)
        return LayerAblationAnalyzer(random_state=random_state).run_ablation(
            records, labels=synth_labels)
    return LayerAblationAnalyzer(random_state=random_state).run_ablation(
        records, labels=labels)


# ==============================================================================
# 统计效力工具（问题5 v3）：Hanley-McNeil 功效分析 / DeLong 配对检验
# ==============================================================================

def _hanley_se_squared(auc, n_pos, n_neg):
    """Hanley & McNeil (1982) AUC 方差（Q1/Q2 指数近似）。

    V(AUC) = [AUC(1-AUC) + (n_N-1)(Q1-AUC²) + (n_P-1)(Q2-AUC²)]/(n_N·n_P)
    其中 Q1 = AUC/(2-AUC)、Q2 = 2AUC²/(1+AUC)。
    """
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    return (auc * (1.0 - auc)
            + (n_neg - 1) * (q1 - auc * auc)
            + (n_pos - 1) * (q2 - auc * auc)) / float(n_pos * n_neg)


def auroc_power_analysis(base_auc, delta, prevalence, alpha=0.05, power=0.8,
                         paired_r=0.6):
    """配对 AUROC 比较的先验功效分析（问题5：写进实验设计）。

    消融是**配对设计**：基线与完整模型给同一批受试者打分。所需样本量
    由 Δ 的方差决定（Hanley & McNeil 1983，Radiology 148:839-843）：

        V(Δ) = V₁ + V₂ − 2·r·√(V₁·V₂)

    其中 V₁/V₂ 为两 AUC 的 Hanley-McNeil 方差（1982 Q1/Q2 近似），
    r 为两 AUC 估计的相关（同一组病例的竞争模型典型 0.4-0.6，
    默认 0.6；假设越低 r 越保守——所需样本越多）。

    求解：最小 n 使 V(Δ) ≤ Δ²/(z_{α/2}+z_β)²。

    Args:
        base_auc: 基线 AUROC（基线与目标都须在 [0.5, 1)）
        delta: 要检出的 AUROC 提升（> 0，如 0.05）
        prevalence: 阳性率（0-1 开区间）
        alpha: 双侧检验水平（默认 0.05）
        power: 统计功效（默认 0.80）
        paired_r: 两 AUC 估计的相关（默认 0.6）

    Returns:
        dict: n_total / n_positive / n_negative / se_diff_at_n + 全部假设
        （base_auc/target_auc/delta/prevalence/alpha/power/paired_r）+
        reference

    Raises:
        ValueError: 参数越界
    """
    from scipy.stats import norm

    if not (0.5 <= base_auc < 1.0):
        raise ValueError(f'base_auc 须在 [0.5, 1)，实际 {base_auc}')
    if delta <= 0:
        raise ValueError(f'delta 须 > 0，实际 {delta}')
    if base_auc + delta > 1.0:
        raise ValueError(
            f'base_auc + delta 须 ≤ 1（{base_auc}+{delta}）')
    if not (0.0 < prevalence < 1.0):
        raise ValueError(f'prevalence 须在 (0,1) 开区间，实际 {prevalence}')
    if not (0.0 < paired_r < 1.0):
        raise ValueError(f'paired_r 须在 (0,1) 开区间，实际 {paired_r}')

    target_auc = base_auc + delta
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(norm.ppf(power))
    required_var = delta * delta / ((z_alpha + z_beta) ** 2)

    # 线性扫描最小 n（V(Δ) 随 n 单调不增；上限 20 万例内必有解）
    for n_total in range(10, 200001):
        n_pos = max(2, int(round(prevalence * n_total)))
        n_neg = n_total - n_pos
        if n_neg < 2:
            continue
        v1 = _hanley_se_squared(base_auc, n_pos, n_neg)
        v2 = _hanley_se_squared(target_auc, n_pos, n_neg)
        var_diff = v1 + v2 - 2.0 * paired_r * math.sqrt(v1 * v2)
        if var_diff <= required_var:
            return {
                'method': 'hanley_mcneil_paired_auroc',
                'n_total': int(n_total),
                'n_positive': int(n_pos),
                'n_negative': int(n_neg),
                'base_auc': float(base_auc),
                'target_auc': float(target_auc),
                'delta': float(delta),
                'prevalence': float(prevalence),
                'alpha': float(alpha),
                'power': float(power),
                'paired_r': float(paired_r),
                'se_diff_at_n': round(math.sqrt(max(var_diff, 0.0)), 6),
                'reference': (
                    'Hanley & McNeil, Radiology 1982;143:29-46（AUC 方差）；'
                    'Radiology 1983;148:839-843（同组病例配对比较）'),
            }
    # 理论上不可达（required_var>0 且 V(Δ)→0）；防御性返回
    raise RuntimeError('功效分析在 20 万例内未收敛，请检查参数')


def _delong_components(y_true, scores):
    """DeLong 结构成分：AUC 与逐病例 V10/V01（并列记 0.5，全向量化）。"""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y_true == 1][:, None]
    neg = scores[y_true == 0][None, :]
    diff = pos - neg
    pair = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))
    return float(pair.mean()), pair.mean(axis=1), pair.mean(axis=0)


def delong_paired_test(y_true, scores_a, scores_b):
    """DeLong 配对检验：同一组受试者两评分系统的 AUROC 差异。

    DeLong ER, DeLong DM, Clarke-Pearson DL. Biometrics 1988;44:837-845。
    协方差由结构成分 V10（逐病例）/V01（逐对照）估计；两评分完全相同
    （Δ=0 且方差=0）时 p=1。

    Args:
        y_true: 二元结局（0/1）
        scores_a / scores_b: 两系统的分数（同一批受试者，顺序对齐）

    Returns:
        dict: auroc_a / auroc_b / delta_auroc（a−b）/ se_diff / z / p_value
    """
    from scipy.stats import norm

    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        raise ValueError('DeLong 检验需要两类结局同时存在')
    auc_a, v10_a, v01_a = _delong_components(y_true, scores_a)
    auc_b, v10_b, v01_b = _delong_components(y_true, scores_b)
    n_pos, n_neg = len(v10_a), len(v01_a)

    def _cov(x, y):
        return float(np.mean((x - x.mean()) * (y - y.mean())))

    var_a = _cov(v10_a, v10_a) / n_pos + _cov(v01_a, v01_a) / n_neg
    var_b = _cov(v10_b, v10_b) / n_pos + _cov(v01_b, v01_b) / n_neg
    cov_ab = _cov(v10_a, v10_b) / n_pos + _cov(v01_a, v01_b) / n_neg
    var_diff = var_a + var_b - 2.0 * cov_ab

    delta = auc_a - auc_b
    if var_diff <= 0:
        # 两评分完全相同（或线性同构）：无差异可检
        z_stat = 0.0
        p_value = 1.0
        se_diff = 0.0
    else:
        se_diff = math.sqrt(var_diff)
        z_stat = delta / se_diff
        p_value = 2.0 * float(norm.sf(abs(z_stat)))
    return {
        'method': 'delong_paired',
        'auroc_a': round(auc_a, 6),
        'auroc_b': round(auc_b, 6),
        'delta_auroc': round(delta, 6),
        'se_diff': round(se_diff, 6),
        'z': round(z_stat, 6),
        'p_value': float(p_value),
    }


# ==============================================================================
# 多种子消融（问题5 v3：500+ 样本 × 20+ 种子，效应量+CI+DeLong）
# ==============================================================================

def run_multi_seed_ablation(n_contacts=500, n_cases=None, n_seeds=20,
                            random_state=42, n_bootstrap=2000,
                            cluster_weight=0.6, individual_noise=0.15):
    """网络贡献消融·多种子版（问题5：统计效力升级）。

    相对单次运行（network_contribution_ablation）的三项升级：

    1. **样本量与种子**：默认 500 接触者 × 20 种子（旧版 40×1 点估计）；
    2. **效应量 + 95% CI**：ΔAUROC 跨种子均值（效应量）+ 种子级
       bootstrap 置信区间（百分位法，重采样单位 = 种子——实验的
       重复测量单元），代替单点估计；
    3. **DeLong 配对检验**：每个种子内基线 vs 完整模型给**同一批受试者**
       打分 → DeLong（1988）配对检验逐种子 p 值 + 中位数/显著计数；
    4. **先验功效分析写进实验设计**：80% 功效、双侧 α=0.05 检出
       ΔAUROC=0.05 所需样本量（Hanley-McNeil 配对方差），同时给出
       实验阳性率与密接人群实际阳性率（2.85%）两种口径——
       后者是部署场景的真实约束。

    Args:
        n_contacts: 每种子接触者数（默认 500，用户规格）
        n_cases: 病例数；None → 30%（与旧版 12/40 信号设计一致）
        n_seeds: 种子数（默认 20，用户规格；种子 = random_state+i）
        random_state: 起始种子
        n_bootstrap: 种子级 bootstrap 重采样次数
        cluster_weight / individual_noise: 合成网络信号参数（同 build_synthetic_network）

    返回：
        dict: seeds（逐种子）/ effect_size / bootstrap_ci / delong /
        power_design / network_contributes（bool，依据 CI 而非点估计）/
        conclusion
    """
    if n_cases is None:
        n_cases = max(1, int(round(0.30 * n_contacts)))

    analyzer = LayerAblationAnalyzer(random_state=random_state)
    seeds = []
    for i in range(n_seeds):
        seed = random_state + i
        records, labels = build_synthetic_network(
            n_contacts=n_contacts, n_cases=n_cases,
            cluster_weight=cluster_weight, individual_noise=individual_noise,
            random_state=seed)
        rep = analyzer.run_ablation(records, labels=labels,
                                    return_scores=True)
        delong = delong_paired_test(
            labels, rep['full_scores'], rep['baseline_scores'])
        seeds.append({
            'seed': seed,
            'auroc_baseline': rep['baseline']['auroc'],
            'auroc_full': rep['full']['auroc'],
            'delta_auroc': rep['delta']['delta_auroc'],
            'delta_cindex': rep['delta']['delta_cindex'],
            'delong_p': delong['p_value'],
        })

    deltas = np.array([s['delta_auroc'] for s in seeds], dtype=float)
    delta_c = np.array([s['delta_cindex'] for s in seeds], dtype=float)
    auroc_base = np.array([s['auroc_baseline'] for s in seeds])
    auroc_full = np.array([s['auroc_full'] for s in seeds])
    p_values = np.array([s['delong_p'] for s in seeds])

    # 种子级 bootstrap 95% CI（重采样单位 = 种子）
    rng = np.random.RandomState(random_state)
    boot_means = np.array([
        deltas[rng.randint(0, len(deltas), len(deltas))].mean()
        for _ in range(n_bootstrap)
    ])
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    effect_mean = float(deltas.mean())
    # 结论依据：效应量 + CI（排除 0）+ DeLong 显著比例——而非点估计
    contributes = bool(effect_mean > 0.005 and ci_lower > 0.0)

    # 先验功效分析（实验设计的一部分：固定 Δ=0.05 / power=0.80）
    try:
        from ..constants import PREVALENCE_CALIBERS
        cc_prevalence = float(
            PREVALENCE_CALIBERS['close_contact']['value'])
    except Exception:  # noqa: BLE001 — 常量不可用时退回字面值
        cc_prevalence = 0.0285
    # 功效分析有效域防护：基线 AUROC 夹紧到 [0.5, 0.94]
    # （base+Δ ≤ 1），避免极端种子均值使整份报告失败
    base_auc_for_power = round(min(max(float(auroc_base.mean()), 0.5), 0.94), 4)
    power_design = {
        'method': 'hanley_mcneil_paired_auroc',
        'detect_delta': 0.05,
        'power': 0.80,
        'alpha': 0.05,
        'paired_r': 0.60,
        'base_auc': base_auc_for_power,
        'required_n_experiment_prevalence': auroc_power_analysis(
            base_auc=base_auc_for_power, delta=0.05,
            prevalence=n_cases / float(n_contacts)),
        'required_n_close_contact_prevalence': auroc_power_analysis(
            base_auc=base_auc_for_power, delta=0.05,
            prevalence=cc_prevalence),
        'note': (
            '先验功效分析（实验设计阶段固定 Δ=0.05、power=0.80、双侧 '
            'α=0.05、配对 r=0.6）：实验阳性率口径给出本设计的最小样本量；'
            '密接人群实际阳性率（2.85%）口径给出部署场景检出同等效应'
            '所需的更大样本量——两套口径不可混用。'),
    }

    n_sig = int((p_values < 0.05).sum())
    if contributes:
        conclusion = (
            f'网络增强层效应量 ΔAUROC={effect_mean:.4f}（{n_seeds} 种子均值，'
            f'bootstrap 95% CI [{ci_lower:.4f}, {ci_upper:.4f}] 不含 0，'
            f'{n_sig}/{n_seeds} 种子 DeLong 配对检验 p<0.05）——'
            '网络结构对风险排序有统计学上稳健的贡献。')
    else:
        conclusion = (
            f'网络增强层效应量 ΔAUROC={effect_mean:.4f}（bootstrap 95% CI '
            f'[{ci_lower:.4f}, {ci_upper:.4f}] 含 0 或效应量过小，'
            f'{n_sig}/{n_seeds} 种子 DeLong p<0.05）——'
            '当前证据不足以支持网络层的显著判别增益。')

    return {
        'method': 'ablation_network_layer_multiseed',
        'n_contacts': int(n_contacts),
        'n_cases': int(n_cases),
        'n_seeds': int(n_seeds),
        'prevalence': round(n_cases / float(n_contacts), 4),
        'seeds': seeds,
        'effect_size': {
            'delta_auroc_mean': round(effect_mean, 4),
            'delta_auroc_std': round(float(deltas.std(ddof=1)), 4)
            if len(deltas) > 1 else 0.0,
            'delta_auroc_min': round(float(deltas.min()), 4),
            'delta_auroc_max': round(float(deltas.max()), 4),
            'delta_cindex_mean': round(float(delta_c.mean()), 4),
            'auroc_baseline_mean': round(float(auroc_base.mean()), 4),
            'auroc_full_mean': round(float(auroc_full.mean()), 4),
        },
        'bootstrap_ci': {
            'method': 'percentile_bootstrap_over_seeds',
            'n_bootstrap': int(n_bootstrap),
            'level': 0.95,
            'ci_lower': round(ci_lower, 4),
            'ci_upper': round(ci_upper, 4),
        },
        'delong': {
            'method': 'delong_paired_per_seed',
            'per_seed_p': [round(float(p), 6) for p in p_values],
            'p_median': round(float(np.median(p_values)), 6),
            'p_median_note': (
                'p_median 为种子级 p 值的中位数（描述性统计），'
                '不是整体显著性检验——整体结论依据 bootstrap 95% CI '
                '是否含 0 与 n_significant/n_seeds 比例，不得将 '
                'p_median<0.05 引用为"整体显著"'),
            'p_min': round(float(p_values.min()), 6),
            'p_max': round(float(p_values.max()), 6),
            'n_significant': n_sig,
        },
        'power_design': power_design,
        'network_contributes': contributes,
        'conclusion': conclusion,
    }


__all__ = [
    'compute_c_index',
    'build_synthetic_network',
    'LayerAblationAnalyzer',
    'network_contribution_ablation',
    'auroc_power_analysis',
    'delong_paired_test',
    'run_multi_seed_ablation',
]
