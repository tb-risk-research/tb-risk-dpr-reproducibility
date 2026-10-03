#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""集成层评估：同一批输入上端到端测量 双ML + GNN + SEIR 四成员加权融合。

背景：训练档案中 ML / GNN / SEIR 各方向已分别有健康指标，但从未有任何一次
"多方向集成"的端到端测量——而这才是系统的最终产出。本脚本补上这一测量空白。
2026-08-22 起升级为四成员：双 ML 成员（机制/临床规则标签错开训练的多样性
工程）+ GNN + SEIR（同时检验 SEIR 作门控信号而非加权票的融合方式）。

评估协议（部署形态对齐）：
- 生成 N 个"患者场景"（1 名传染源 + 若干家庭/社会接触者），与
  RiskAssessmentService 的输入形态一致（patient_info + entries + results）；
- 标签由机制传播模型生成（与 GNN 训练的 M7 同族但独立实现，
  作用于评估特征而非图节点）：
      p = 1 - exp(-infectivity(patient) * intensity(contact) * susceptibility(contact))
- 每个接触者由四个已训练方向独立打分：
      ML(mech): predictor.predict_risk(contact, type)['ensemble']  (0-100)
      ML(clin): 同上，临床规则标签训练的第二 ML 成员              (0-100)
      SEIR: ContactRiskCalculator.generate_potential_patients
            -> seir_infection_probability                       (0-100)
      GNN : M7 对齐图前向传播（_gnn_score）                      (0-100)
- 集成：DEFAULT_ENSEMBLE_WEIGHTS 三成员固定权重作对照基线（0-100 尺度），
  四成员融合权重由 learn_weights 在训练半区学习、测试半区报告。

指标：各方向 AUROC、各集成组合 AUROC/AUPRC/Brier；集成 vs 最优单方向差值
      （回答"集成 > 单方向"这一集成存在的前提）。

归档：TrainingLogger(model_type='ensemble')，dataset bucket 独立，追加不覆盖。

用法：
    python data/evaluate_ensemble.py --scenarios 300 --seed 42
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DESKTOP = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, DESKTOP)

MODELS_DIR = os.path.join(HERE, "training_archive", "models")


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def load_models(ml_path=None, ml_clin_path=None, gnn_path=None):
    """加载双 ML（机制/临床标签）与 GNN（场景对齐）模型。

    双 ML 成员：同一 22 维部署特征管线，标签机制错开训练（mechanistic vs
    clinical rule）——多样性工程，降低成员判错重叠。GNN 挂在 mech
    predictor 上；clin predictor 仅承载第二个 ML 模型。
    """
    from tb_risk.scoring.predictor import MLRiskPredictor

    # 2026-08-24 主模型冻结（宿主通路修复 v3 标签代）：mechanistic 标签
    # 加入文献校准的宿主因素（年龄U型/症状/共病RR/BCG衰减）后标签分布
    # 改变，旧代 checkpoint（185119/185411，暴露单通路标签）由
    # data/freeze_primary_models.py 显式替代——决策档案见
    # data/processed/primary_model_freeze_20260824.json。
    if ml_path is None:
        ml_path = os.path.join(
            MODELS_DIR, "best_ml_20260824_235605_ns12509_cv5.joblib")
    if ml_clin_path is None:
        ml_clin_path = os.path.join(
            MODELS_DIR, "best_ml_20260824_235828_ns12509_cv5.joblib")
    if gnn_path is None:
        gnn_path = os.path.join(
            MODELS_DIR, "best_gnn_20260822_171114_gnn_lr0.001_ep200.pt")

    predictor = MLRiskPredictor()
    # P4（标签代际校验）：mech 成员必须是宿主通路 v3 代标签训练
    predictor.expected_label_generation = 'v3-host-pathway'
    assert predictor.load_model(ml_path), f"ML(mech) 模型加载失败: {ml_path}"
    assert predictor.is_trained, "ML(mech) is_trained=False"
    print(f"[加载] ML(mech): {os.path.basename(ml_path)}")

    predictor_clin = MLRiskPredictor()
    predictor_clin.expected_label_generation = 'clinical-rule-v1'
    assert predictor_clin.load_model(ml_clin_path), \
        f"ML(clin) 模型加载失败: {ml_clin_path}"
    assert predictor_clin.is_trained, "ML(clin) is_trained=False"
    print(f"[加载] ML(clin): {os.path.basename(ml_clin_path)}")

    ok = predictor.load_gnn_model(gnn_path)
    print(f"[加载] GNN: {os.path.basename(gnn_path)} "
          f"({'成功' if ok else '失败'})")
    if not ok:
        print("[警告] GNN 加载失败，评估将退化为 双ML+SEIR 三方向")
    return predictor, predictor_clin, ml_path, ml_clin_path, gnn_path


# ---------------------------------------------------------------------------
# 场景生成（部署形态 + 机制传播标签）
# ---------------------------------------------------------------------------

def _sample_patient(rng):
    """患者（传染源）特征：决定传染力。"""
    smear = rng.choice([1, 2], p=[0.4, 0.6])       # 2=涂阳
    cavity = rng.choice([1, 2], p=[0.55, 0.45])    # 2=有空洞
    return {
        'basic_info': {
            'age': int(rng.integers(20, 75)),
            'sputum_smear': smear,
            'has_cavity': cavity,
            'treatment': rng.choice([1, 2], p=[0.3, 0.7]),  # 1=治疗中
            'cough_freq': int(rng.integers(0, 25)),
            'delay_days': int(rng.integers(0, 90)),
            'symptoms': int(rng.integers(1, 4)),
        }
    }


def _patient_infectivity(patient_info):
    """传染源传染力（连续量，供标签生成）。"""
    b = patient_info['basic_info']
    smear = 1.0 if b['sputum_smear'] == 2 else 0.4
    cavity = 1.0 if b['has_cavity'] == 2 else 0.6
    cough = min(b['cough_freq'] / 20.0, 1.5)
    delay = min(b['delay_days'] / 60.0, 1.5)       # 延迟就诊 → 更长传染窗口
    return 0.6 * smear * cavity * (0.5 + 0.5 * cough) * (0.6 + 0.4 * delay)


def _sample_contact(rng, idx, ctype):
    """接触者特征：ML/GNN/SEIR 三方输入字段的并集。"""
    age = int(rng.integers(3, 80))
    past = rng.choice(['none', 'diabetes', 'hiv', 'immunosuppressants', 'other'],
                      p=[0.75, 0.12, 0.03, 0.04, 0.06])
    if ctype == 'family':
        freq = int(rng.integers(8, 30))            # 次/月
        dur = int(rng.integers(60, 480))           # 分钟/次
        dist = rng.choice(['close', 'medium'], p=[0.7, 0.3])
        vent = int(rng.integers(2, 5))
        setting = rng.choice(['closed', 'general'], p=[0.6, 0.4])
    else:
        freq = int(rng.integers(1, 8))
        dur = int(rng.integers(10, 120))
        dist = rng.choice(['medium', 'far'], p=[0.5, 0.5])
        vent = int(rng.integers(2, 6))
        setting = rng.choice(['general', 'outdoor'], p=[0.6, 0.4])
    time_span = int(rng.integers(1, 26))
    # v2（2026-08-22，死特征修复）：补 cumulative_exposure。部署路径三层
    # 兜底（GUI 录入/导入/assessment）均按 duration/60*freq*span（小时）
    # 填充该字段（core.probability.calculate_cumulative_exposure），而
    # 本评估直调 predict_risk 绕过兜底、_sample_contact 又缺该键 →
    # extract_features 取默认 0，该特征在 Kenya/NHANES/场景训练 CSV
    # 与评估打分中全部恒为 0（死特征）——接触者追踪的核心暴露维度
    # 从未参与训练，越南验证因此暴露该结构缺陷（45 例家庭暴露无处映射）。
    cumulative_exposure = dur / 60.0 * freq * time_span   # 总接触小时数
    # v3（2026-08-24，宿主通路修复）：补既往结核史与高危人群标记。
    # 此前 has_tb 硬编码 0、is_high_risk 缺键 → 特征提取默认 0 →
    # 场景 CSV 六列全零（has_tb/is_high_risk/dm_tb_synergy/
    # highrisk_comorbid），宿主因素维度在训练数据中根本不存在。
    # 糖尿病患者既往 TB 概率更高（Jeon & Murray 2008 RR≈3.11）——
    # dm_tb_synergy 交互列因此在临床人群中并不稀有。
    has_tb_p = 0.05 if past == 'diabetes' else 0.02
    has_tb = int(rng.random() < has_tb_p)  # 既往结核史（基线 2%，DGP FIELD_SPECS Bernoulli(0.02)）
    return {
        'name': f'{ctype[0]}_{idx}',
        'record_id': f'{ctype[0]}_{idx}',   # 与 SEIR 查找键对齐
        'age': age,
        'bcg_vaccine': int(rng.random() < 0.85),
        'past_illness': 0 if past == 'none' else 1,
        'past_illness_type': past,
        'has_symptoms': int(rng.random() < 0.25),
        'has_tb': has_tb,
        'is_high_risk': int(rng.random() < 0.10),   # 高危人群标记（矽肺/透析等）
        'cumulative_exposure': cumulative_exposure,
        'freq_density': freq,
        'single_duration': dur,
        'time_span': time_span,
        'contact_distance': dist,
        'ventilation': vent,
        'exposure_setting': setting,
    }


def _contact_intensity(contact):
    """接触强度（连续量，供标签生成）。"""
    dist_map = {'close': 1.0, 'medium': 0.5, 'far': 0.25}
    freq = min(contact['freq_density'] / 30.0, 1.2)
    dur = min(contact['single_duration'] / 480.0, 1.2)
    span = min(contact['time_span'] / 26.0, 1.0)
    dist = dist_map.get(contact['contact_distance'], 0.5)
    vent = 1.0 - 0.12 * contact['ventilation']     # 通风越好强度越低
    return freq * dur * span * dist * vent


def _contact_susceptibility(contact):
    """易感性（连续量，供标签生成）——宿主因素通路（文献校准版）。

    v3（2026-08-24，DGP 统一）：实现委托
    ``tb_risk.validation.host_susceptibility.host_susceptibility_multiplier``
    （宿主易感性参数的单一真值源——layer_ablation 的合成网络标签与
    dgp.py 的标签机制登记表引用同一模块，防止多处实现漂移；
    随机一致性测试见 tests/test_host_susceptibility_source.py）。

    v2（2026-08-24，宿主通路修复）：原实现各因子变异幅度过小（总范围
    0.9~3.5，多数样本集中 0.9~1.4），在乘积-指数结构
    p=1-exp(-infectivity*intensity*susceptibility) 中被跨多个数量级的
    暴露强度完全淹没，且 has_symptoms 根本不参与——宿主特征（年龄/
    症状/共病/BCG）对标签的单变量 AUROC 全部 ≈0.50，模型只能学成
    "暴露单通路模型"。本版把易感性明确为"感染易感 × 进展放大"的
    复合宿主因子，全部按文献校准（完整参数表与文献引用见
    ``HOST_SUSCEPTIBILITY_SPEC``）：

    - 年龄 U 型进展曲线：<5 岁 ×2.0 / 5-14 岁 ×0.4 / 15-35 ×1.0 /
      35-65 ×1.2 / >65 ×2.2（Martinez 2020 Lancet；Seddon 2014）
    - 接触者已有症状 ×3.5（Fox 2013：有症状接触者检出 10-30% vs
      无症状 1-4%，RR 5-10 取保守下限）
    - 既往结核史 ×2.0（再激活）；高危人群标记 ×1.5（保守下限）
    - BCG 保护随年限指数衰减 VE(age)=0.5*exp(-age/20)
      （Colditz 1995；Abubakar 2013）
    - 共病进展放大：HIV ×12 / 免疫抑制 ×5 / 糖尿病 ×2.5 / 其他 ×1.3
      （WHO；Jeon 2008 RR=3.11；Cochrane 2024）
    """
    from tb_risk.validation.host_susceptibility import (
        host_susceptibility_multiplier)
    return host_susceptibility_multiplier(
        age=contact['age'],
        has_symptoms=contact.get('has_symptoms', 0),
        has_tb=contact.get('has_tb', 0),
        is_high_risk=contact.get('is_high_risk', 0),
        bcg_vaccine=contact.get('bcg_vaccine', 0),
        past_illness_type=contact.get('past_illness_type', 'none'))


def _clinical_rule_prob(patient_info, contact, ctype):
    """经验临床规则标签概率（去循环化第二基准）。

    与机制标签的 1-exp(-beta*I*S) 完全解耦：线性加权打分 + sigmoid
    （非乘积-指数形式），模拟临床筛查判读逻辑（指示病例特征 +
    密接程度 + 接触者高危因素）。SEIR 方向对该标签无公式同构
    优势——用于检验机制基准下"SEIR 既当出题人又当答题人"的
    循环论证嫌疑。
    """
    b = patient_info['basic_info']
    score = 0.0
    if b['sputum_smear'] == 2:
        score += 1.5                          # 涂阳指示病例
    if b['delay_days'] > 30:
        score += 0.8                          # 长延迟就诊
    if ctype == 'family':
        score += 1.0                          # 家庭接触
    if contact['contact_distance'] == 'close':
        score += 1.2                          # 密切距离
    if contact['freq_density'] >= 15:
        score += 0.6                          # 高频接触
    if contact['age'] < 5 or contact['age'] > 65:
        score += 0.9                          # 年龄极端
    score += {'hiv': 1.8, 'immunosuppressants': 1.2,
              'diabetes': 0.7, 'other': 0.3}.get(
                  contact['past_illness_type'], 0.0)
    if not contact['bcg_vaccine']:
        score += 0.5
    if contact['has_symptoms']:
        score += 1.0                          # 接触者已有症状
    return 1.0 / (1.0 + np.exp(-(score - 4.5)))


def generate_scenarios(n_scenarios, seed, label_mode='mechanistic',
                       prevalence_target=0.25):
    """生成场景批次：[(patient_info, family, social, labels)]。

    label_mode:
        mechanistic — 1-exp(-infectivity*intensity*susceptibility)
                      （与 SEIR 感染力公式同构，SEIR 有循环论证优势）
        clinical    — 经验临床规则（线性打分+sigmoid，独立第二基准）
    prevalence_target (float, 2026-08-24 P1)：标签尺度校准的目标阳性率
        档——默认 0.25（与旧行为完全一致）；设 0.005-0.03 生成真实
        患病率档（部署阈值迁移检验）。幂缩放保持个体间相对风险排序
        不变，且同一 seed 下接触者/患者特征逐字段相同（只有标签口径
        切换）——低患病率档不是新的特征分布。
    """
    rng = np.random.default_rng(seed)
    scenarios = []
    for s in range(n_scenarios):
        patient = _sample_patient(rng)
        infectivity = _patient_infectivity(patient)
        family = [_sample_contact(rng, f's{s}f{i}', 'family')
                  for i in range(int(rng.integers(2, 5)))]
        social = [_sample_contact(rng, f's{s}c{i}', 'social')
                  for i in range(int(rng.integers(1, 4)))]

        raw = []
        for ctype, group in (('family', family), ('social', social)):
            for c in group:
                if label_mode == 'clinical':
                    p = _clinical_rule_prob(patient, c, ctype)
                else:
                    p = 1.0 - np.exp(
                        -infectivity * _contact_intensity(c)
                        * _contact_susceptibility(c))
                raw.append((ctype, c, p))
        # 尺度校准：目标阳性率档（保持个体间相对风险排序不变）
        target = float(prevalence_target)
        cur = float(np.mean([p for _, _, p in raw]))
        scale = np.log(1 - target) / np.log(1 - cur) if 0 < cur < 1 else 1.0
        labels = []
        for ctype, c, p in raw:
            p_cal = 1.0 - (1.0 - p) ** scale
            labels.append((ctype, c, int(rng.random() < p_cal)))
        scenarios.append((patient, family, social, labels))
    return scenarios


# ---------------------------------------------------------------------------
# 三方向打分
# ---------------------------------------------------------------------------

def _build_m7_graph(patient, family, social):
    """从评估场景构建 M7 对齐同构图（23 维特征，与 GNN 训练分布同构）。

    背景：GNN 训练数据（_generate_synthetic_graphs）为 22 维随机特征 +
    第 23 维入度；而部署推理路径 build_from_assessment 产出 30 维语义
    特征——存在训练/推理特征断层（30≠23）。本评估按 M7 的特征语义
    （前 5 维=传染性、5-10 维=易感性、第 23 维=入度）从评估场景派生
    特征，使 GNN 在其学过的特征空间内被评估。部署语义桥接（30 维→
    23 维）是独立待办，见评估报告局限说明。

    节点：0=患者，1..n_fam=家庭，随后=社会，最后=区域（与训练同构）。
    """
    import torch
    from torch_geometric.data import Data

    n_fam, n_soc = len(family), len(social)
    num_nodes = 1 + n_fam + n_soc + 1
    region_idx = num_nodes - 1

    # 特征：前5=传染性，5-10=易感性，10-22=0（去噪声位），22=入度（后填）
    x = np.zeros((num_nodes, 23), dtype=np.float32)
    inf_p = float(np.clip(_patient_infectivity(patient), 0.05, 0.95))
    x[0, :5] = inf_p
    contacts = ([('family', c) for c in family]
                + [('social', c) for c in social])
    for k, (_, c) in enumerate(contacts):
        sus = float(np.clip(
            _contact_susceptibility(c) / 2.0, 0.05, 0.95))
        x[1 + k, 5:10] = sus

    edge_index, edge_attr_list = [], []
    intens = [_contact_intensity(c) for _, c in contacts]

    def _add(src, dst, w):
        ef = np.zeros(6, dtype=np.float32)
        ef[0] = float(np.clip(w, 0.0, 1.0))
        edge_index.extend([[src, dst], [dst, src]])
        edge_attr_list.extend([ef, ef.copy()])

    for k, inten in enumerate(intens):
        _add(0, 1 + k, 0.05 + 0.5 * inten)     # 患者→接触者
    for i in range(n_fam):
        for j in range(i + 1, n_fam):
            _add(1 + i, 1 + j, 0.5 + 0.5 * intens[i])  # 家庭内部
    for n in range(num_nodes - 1):
        ef = np.zeros(6, dtype=np.float32)      # 区域↔个体，强度0
        edge_index.extend([[region_idx, n], [n, region_idx]])
        edge_attr_list.extend([ef, ef.copy()])

    edge_index_t = torch.tensor(np.array(edge_index).T, dtype=torch.long)
    edge_attr_t = torch.tensor(np.array(edge_attr_list), dtype=torch.float)

    # 入度（第 23 维，与训练同款归一化）
    in_deg = np.zeros(num_nodes)
    for e in range(edge_index_t.shape[1]):
        in_deg[int(edge_index_t[1, e])] += 1.0
    x[:, 22] = in_deg / max(in_deg.max(), 1.0)

    return Data(x=torch.tensor(x), edge_index=edge_index_t,
                edge_attr=edge_attr_t)


def generate_scenario_graphs(n_graphs, random_state=42):
    """场景对齐 GNN 训练图：复用本评估的场景构造 + M7 图 + 机制标签。

    背景：GNN 在 _generate_synthetic_graphs（22 维噪声特征、噪声边特征）
    上训练后，在部署场景分布上出现域迁移断崖（训练内 AUROC 0.657 →
    部署场景 0.533）。本生成器让训练图直接采用评估侧同一特征编码 /
    图结构分布 / 标签机制，从源头消除域间隙。

    节点：0=患者（y=0, mask=False）、1..k=接触者（y=场景标签）、
    末位=区域（y=0, mask=False）——与 _generate_synthetic_graphs 字段同构。

    注意：训练与评估必须用不同 seed（数据不重叠、分布一致）。
    """
    import torch
    from torch_geometric.data import Data

    scenarios = generate_scenarios(n_graphs, random_state)
    graphs = []
    for patient, family, social, labels in scenarios:
        g = _build_m7_graph(patient, family, social)
        num_nodes = int(g.x.shape[0])
        label_map = {(ctype, c['name']): int(yv)
                     for ctype, c, yv in labels}
        contacts = ([('family', c) for c in family]
                    + [('social', c) for c in social])
        y = np.zeros(num_nodes, dtype=np.float64)
        for k, (ctype, c) in enumerate(contacts):
            y[1 + k] = label_map.get((ctype, c['name']), 0)
        mask = np.ones(num_nodes, dtype=bool)
        mask[0] = False            # 患者节点不参与损失
        mask[num_nodes - 1] = False  # 区域节点无个体标签
        graphs.append(Data(
            x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr,
            y=torch.tensor(y, dtype=torch.float),
            mask=torch.tensor(mask, dtype=torch.bool)))
    # 70/20/10 划分（与 _generate_synthetic_graphs 一致）
    train_size = int(0.7 * n_graphs)
    val_size = int(0.2 * n_graphs)
    return (graphs[:train_size],
            graphs[train_size:train_size + val_size],
            graphs[train_size + val_size:])


def generate_scenario_ml_csv(n_scenarios, seed, csv_path, meta_path,
                             label_mode='mechanistic', prevalence_target=0.25):
    """ML 场景对齐训练数据：部署特征管线 + 场景标签 → 训练 CSV。

    背景：Kenya（训练内 AUROC 0.75）与半合成 ML 模型在部署场景评估中
    ml_only 分别仅 0.47/0.62——训练特征分布与部署形态断崖（Kenya 为
    症状学调查特征，半合成 CSV 的 cumulative_exposure 等列在部署接触
    者字典中缺失→置 0）。本生成器让 ML 训练数据直接采用评估侧同一
    场景构造 + 同一特征提取（extract_features_with_interactions，
    22 维部署管线），从源头消除域间隙——GNN 场景对齐的同一配方。

    注意：训练与评估必须用不同 seed（数据不重叠、分布一致）。

    v2（2026-08-24，宿主通路修复）：
    - 接触者字典携带 has_tb（既往结核史 2%）与 is_high_risk（10%），
      修复六列结构性零值中的四列（has_tb/is_high_risk/dm_tb_synergy/
      highrisk_comorbid）；
    - patient 上下文（delay_days/cough_freq/contact_count）逐场景传入
      交互特征计算——此前实例默认 0，symptom_delay 与 cough_contact
      两列恒为 0（另两列死特征）；
    - 易感性项按文献校准（见 _contact_susceptibility docstring），
      has_symptoms/age/past_illness 单变量 AUROC 从 ≈0.50 升至 0.55+。
    """
    import json

    import pandas as pd
    from tb_risk.scoring.predictor import MLRiskPredictor

    # 仅用其部署特征管线；patient 上下文逐场景显式传入（v2 修复：
    # 此前丢弃 patient → 交互特征 symptom_delay/cough_contact 恒 0）
    p = MLRiskPredictor()
    scenarios = generate_scenarios(n_scenarios, seed, label_mode=label_mode,
                                   prevalence_target=prevalence_target)
    rows = []
    for patient, family, social, labels in scenarios:
        b = patient['basic_info']
        n_contacts = len(family) + len(social)
        for ctype, contact, y in labels:
            feats = p.extract_features_with_interactions(
                contact, ctype,
                patient_ftd=b['delay_days'],
                patient_cough_freq=b['cough_freq'],
                contact_count=n_contacts).ravel()
            rows.append([float(v) for v in feats] + [int(y)])
    df = pd.DataFrame(rows, columns=list(p.ALL_FEATURE_NAMES) + ['tb_outcome'])
    df.to_csv(csv_path, index=False)
    meta = {
        "source": ("deployment-form scenario contacts (ensemble evaluation "
                   f"generator, seed={seed}, label_mode={label_mode})"),
        "n_samples": int(len(df)),
        "n_positive": int(df['tb_outcome'].sum()),
        "positive_rate": float(df['tb_outcome'].mean()),
        "features": "22-dim deployment pipeline "
                    "(extract_features_with_interactions)",
        "note": ("scenario-aligned ML training data; distribution matches "
                 "ensemble evaluation scenarios (different seed, no "
                 "overlap); features identical to deployment scoring path"),
        "prevalence_target": float(prevalence_target),
        "label_note": (
            f"标签尺度校准目标阳性率档 = {prevalence_target}（幂缩放，"
            "保持相对风险排序；默认 0.25 与宿主通路修复前行为一致）"),
        "host_pathway": {
            "version": "v2 (2026-08-24)",
            "change": ("susceptibility redefined as composite host factor "
                       "(infection susceptibility x progression "
                       "amplification), literature-calibrated; contact "
                       "dicts carry has_tb/is_high_risk; patient context "
                       "(delay_days/cough_freq/contact_count) passed to "
                       "interaction features — fixes 6 structurally "
                       "zero columns and revives the host pathway"),
            "age_curve_rr": {"<5": 2.0, "5-14": 0.4, "15-35": 1.0,
                             "35-65": 1.2, ">65": 2.2},
            "symptoms_rr": 3.5,
            "prior_tb_rr": 2.0,
            "high_risk_rr": 1.5,
            "has_tb_prevalence": {"base": 0.02,
                                  "diabetes": 0.05},
            "comorbidity_rr": {"hiv": 12.0, "immunosuppressants": 5.0,
                               "diabetes": 2.5, "other": 1.3},
            "bcg_waning": "VE(age) = 0.5 * exp(-age/20)",
            "references": [
                "Martinez L et al. Lancet 2020;395:973-984 (child contact "
                "U-shaped progression risk: 7.6%/5.2%/5.6% by age)",
                "Seddon JA & Shingadia D. Infect Drug Resist 2014;7:153-165 "
                "(progression: infants ~50%, 5-10y ~2%, adults ~5%)",
                "Jeon CY & Murray MB. PLoS Med 2008;5(7):e152 (diabetes TB "
                "RR=3.11, 95%CI 2.27-4.26)",
                "Cochrane 2024;CD016013 (diabetes TB RR 1.5-2.4)",
                "Colditz GA et al. JAMA 1995 (BCG ~50% overall efficacy)",
                "Abubakar I et al. Health Technol Assess 2013;17.37 (BCG "
                "protection wanes over 10-15y)",
                "WHO Global TB Report (PLHIV 16-27x TB risk)",
                "Fox GJ et al. PLoS Med 2013 (contact investigation)",
            ],
        },
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return csv_path, len(df)


def _gnn_score(predictor, patient, family, social, contact, contact_type):
    """GNN 方向：M7 对齐图上前向传播，取目标节点概率（0-100）。

    节点布局（与 _build_m7_graph 一致）：0=患者，1..n_fam=family，
    随后=social，末位=region。目标索引必须按完整 contacts 顺序计算——
    曾用 `1 + (family|social).index(c)`，social 接触者因此错指向
    family 节点（约 1/3 接触者分数错位，GNN 方向被系统性低估）。
    """
    import torch
    if predictor.gnn_model is None:
        return None
    graph = _build_m7_graph(patient, family, social)
    target_idx = None
    for k, c in enumerate(family + social):
        if c is contact:
            target_idx = 1 + k
            break
    if target_idx is None:
        return None
    predictor.gnn_model.eval()
    with torch.no_grad():
        scores, _ = predictor.gnn_model(graph.x, graph.edge_index,
                                         edge_attr=graph.edge_attr)
    prob = float(scores[target_idx, 0]) * 100.0
    return max(0.0, min(100.0, prob))


def score_scenarios(predictor, scenarios, predictor_clin=None):
    """对每个接触者收集四成员概率 + 标签。

    predictor      承载 ML(mech) + GNN；predictor_clin 承载 ML(clin)。
    predictor_clin 为 None 时 ml_clin 列全 None（退化为三成员）。
    """
    from tb_risk.core.contact_risk_calculator import ContactRiskCalculator
    from tb_risk.core.seir_integration import SEIRIntegration

    calc = ContactRiskCalculator(seir_integration=SEIRIntegration())

    rows = []
    n_gnn_ok = 0
    for s_idx, (patient, family, social, labels) in enumerate(scenarios):
        treatment_days = float(
            patient['basic_info']['delay_days']) * 0.5 + 10.0

        # SEIR 方向：规则+SEIR 融合的纯 SEIR 分（0-100）
        try:
            pp = calc.generate_potential_patients(
                patient, family, social, treatment_days)
        except Exception as e:
            print(f"[警告] SEIR 计算失败: {e}")
            pp = {'family': [], 'social': []}
        seir_map = {}
        for group in ('family', 'social'):
            for entry in pp.get(group, []):
                seir_map[entry.get('record_id') or entry.get('name')] = \
                    float(entry.get('seir_infection_probability', 0.0))

        for ctype, contact, y in labels:
            # ML(mech) 方向
            ml_prob = None
            try:
                pred = predictor.predict_risk(contact, ctype)
                if pred and 'ensemble' in pred:
                    ml_prob = float(pred['ensemble']['risk_probability'])
            except Exception:
                ml_prob = None

            # ML(clin) 方向（临床规则标签训练的第二 ML 成员）
            ml_clin_prob = None
            if predictor_clin is not None:
                try:
                    pred_c = predictor_clin.predict_risk(contact, ctype)
                    if pred_c and 'ensemble' in pred_c:
                        ml_clin_prob = float(
                            pred_c['ensemble']['risk_probability'])
                except Exception:
                    ml_clin_prob = None

            # SEIR 方向（0-100）
            seir_prob = seir_map.get(contact['name'])

            # GNN 方向（M7 对齐图前向，0-100；见 _build_m7_graph 局限说明）
            gnn_prob = None
            try:
                gnn_prob = _gnn_score(
                    predictor, patient, family, social, contact, ctype)
            except Exception:
                gnn_prob = None
            if gnn_prob is not None:
                n_gnn_ok += 1

            rows.append({
                'contact_type': ctype, 'y': y,
                'ml': ml_prob, 'ml_clin': ml_clin_prob,
                'seir': seir_prob, 'gnn': gnn_prob,
                'scenario_idx': s_idx,
            })
    print(f"[打分] 接触者 {len(rows)} 个，GNN 有效输出 {n_gnn_ok} 个")
    return rows


# ---------------------------------------------------------------------------
# 集成与指标
# ---------------------------------------------------------------------------

def _auroc(y, p):
    from sklearn.metrics import roc_auc_score
    m = [i for i, v in enumerate(p) if v is not None]
    if len(m) < 2 or len(set(y[i] for i in m)) < 2:
        return None
    return float(roc_auc_score([y[i] for i in m], [p[i] for i in m]))


def _metrics(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score, \
        brier_score_loss
    m = [i for i, v in enumerate(p) if v is not None]
    if not m:
        return None
    yy = np.array([y[i] for i in m], dtype=float)
    pp_ = np.array([p[i] for i in m], dtype=float) / 100.0  # 0-100 -> 0-1
    if len(set(yy)) < 2:
        return None
    return {
        'AUROC': float(roc_auc_score(yy, pp_)),
        'AUPRC': float(average_precision_score(yy, pp_)),
        'Brier': float(brier_score_loss(yy, pp_)),
        'n': len(m),
    }


def _weighted(rows, key, w):
    """缺方向时按可用方向重归一化（与 integrator 语义一致）。"""
    out = []
    for r in rows:
        num = den = 0.0
        for k, wk in w.items():
            v = r.get(k)
            if v is not None:
                num += v * wk
                den += wk
        out.append(num / den if den > 0 else None)
    return out


def evaluate(rows):
    """各方向 + 集成组合指标，及集成 vs 最优单方向结论。"""
    y = [r['y'] for r in rows]
    from tb_risk.constants import DEFAULT_ENSEMBLE_WEIGHTS as W
    w3 = W['three']
    w_mg = W['two_ml_gnn']
    w_ms = W['two_ml_seir']

    combos = {
        'ml_only': _weighted(rows, None, {'ml': 1.0}),
        'ml_clin_only': _weighted(rows, None, {'ml_clin': 1.0}),
        'seir_only': _weighted(rows, None, {'seir': 1.0}),
        'gnn_only': _weighted(rows, None, {'gnn': 1.0}),
        'ens_ml_gnn': _weighted(rows, None, w_mg),
        'ens_ml_seir': _weighted(rows, None, w_ms),
        'ens_three': _weighted(rows, None, w3),
        'ens_four': _weighted(rows, None, W['four']),  # 冻结四成员默认
    }
    results = {}
    for name, probs in combos.items():
        m = _metrics(y, probs)
        if m:
            results[name] = m
    return results, w3


# ---------------------------------------------------------------------------
# 数据支撑的权重学习（替代拍定的 0.4/0.3/0.3）
# ---------------------------------------------------------------------------

def correlation_diagnostics(rows, label=''):
    """成员多样性诊断：两两 Spearman 秩相关 + 判错重叠率。

    把"成员相关性高"从推断变成测量值；作为多样性改造（标签机制
    错开训练）前后对照。判错重叠率：两方向都判错（相对 y 的
    0.5 阈值）的样本占两者错误并集的比例——1.0 表示错误完全
    重合（零互补），0.0 表示错误完全错开（最大互补）。
    """
    from scipy.stats import spearmanr
    from itertools import combinations
    keys = ('ml', 'ml_clin', 'seir', 'gnn')
    cols = {}
    for k in keys:
        cols[k] = np.array([np.nan if r.get(k) is None else float(r[k])
                            for r in rows])
    valid = ~np.isnan(np.column_stack([cols[k] for k in keys])).any(axis=1)
    y = np.array([r['y'] for r in rows])[valid]
    out = {'n_valid': int(valid.sum()), 'pairs': {}, 'error_overlap': {}}
    for a, b in combinations(keys, 2):
        rho = float(spearmanr(cols[a][valid], cols[b][valid]).statistic)
        out['pairs'][f'{a}_{b}'] = round(rho, 4)
        # 判错重叠（0.5 阈值；分数 0-100 → 0-1）
        ea = (cols[a][valid] / 100.0 > 0.5).astype(int) != y
        eb = (cols[b][valid] / 100.0 > 0.5).astype(int) != y
        union = (ea | eb).sum()
        overlap = (ea & eb).sum()
        out['error_overlap'][f'{a}_{b}'] = (
            round(float(overlap) / float(union), 4) if union else None)
    tag = f'[{label}] ' if label else ''
    print(f"\n  [多样性诊断]{tag}n={out['n_valid']} "
          f"Spearman: "
          + ' '.join(f"{p}={v}" for p, v in out['pairs'].items())
          + f" | 判错重叠率: "
          + ' '.join(f"{p}={v}" for p, v in out['error_overlap'].items()))
    return out


def _fusion_cols(rows, keys=('ml', 'ml_clin', 'seir', 'gnn')):
    """成员分数 → (y, cols[0-1])：缺失中位数填补 + 0-100 → 0-1。"""
    y = np.array([r['y'] for r in rows], dtype=float)
    cols = {}
    for k in keys:
        v = np.array([np.nan if r.get(k) is None else float(r[k])
                      for r in rows])
        v[np.isnan(v)] = np.nanmedian(v) if np.any(~np.isnan(v)) else 0.0
        cols[k] = v / 100.0
    return y, cols


def _rank_transform(cols):
    """各方向分数 → 秩百分位（对尺度/单调变换完全免疫）。

    注意：秩相对当前批次计算——批量评估有效；单接触者部署需存储
    参考分布或改用 Platt 校准（cal_* 系列，可独立部署）。
    """
    from scipy.stats import rankdata
    return {k: rankdata(v, method='average') / (len(v) + 1.0)
            for k, v in cols.items()}


def _platt_cols(y, cols, tr, te):
    """各方向 Platt（sigmoid）校准：训练半区拟合，全量变换。

    每方向 score → 校准概率，三方向输出到同一 [0,1] 概率尺度
    （修"用坏尺子量身高"：原始尺度 ML ~0.005 量级 vs GNN ~0.5）。
    校准器可持久化 → 部署可用（与秩变换不同）。
    """
    from sklearn.linear_model import LogisticRegression
    cal = {}
    for k, v in cols.items():
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(v[tr].reshape(-1, 1), y[tr])
        cal[k] = lr
    out = {k: cal[k].predict_proba(v.reshape(-1, 1))[:, 1]
           for k, v in cols.items()}
    return out, cal


def _grid_search(y_tr, cols_tr, keys=('ml', 'ml_clin', 'seir', 'gnn'),
                 step=0.05):
    """单纯形网格搜索（选择数据上最优权重）。只搜 keys 中列出的方向。

    前 len(keys)-1 个权重在 step 网格上遍历，末位取 1-前缀和（<0 跳过）。
    """
    from itertools import product

    from sklearn.metrics import roc_auc_score
    steps = np.arange(0.0, 1.0001, step)
    best = (None, -1.0)
    for ws in product(steps, repeat=len(keys) - 1):
        w_last = round(1.0 - sum(ws), 2)
        if w_last < -1e-9:
            continue
        w = [float(v) for v in ws] + [max(0.0, float(w_last))]
        p = sum(wi * cols_tr[k] for wi, k in zip(w, keys) if wi > 0)
        auc = roc_auc_score(y_tr, p)
        if auc > best[1]:
            best = (tuple(w), float(auc))
    return best


def _combo_prob(cols, weights, keys=('ml', 'ml_clin', 'seir', 'gnn')):
    return sum(w * cols[k] for k, w in zip(keys, weights) if w > 0)


def _delong_test(y, p1, p2):
    """配对 DeLong 检验（DeLong et al. 1988）：两 AUROC 之差的显著性。

    返回 {'auc1','auc2','z','p_value'}；无法计算时返回 None。
    """
    from scipy import stats as sps
    y = np.asarray(y, dtype=int)
    pos, neg = y == 1, y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 == 0 or n0 == 0:
        return None

    def _placements(p):
        pp, pn = p[pos], p[neg]
        m = ((pp[:, None] > pn[None, :]).astype(float)
             + 0.5 * (pp[:, None] == pn[None, :]))
        return m.mean(axis=1), m.mean(axis=0)   # 每个 pos/neg 的 placement

    x1, y1 = _placements(p1)
    x2, y2 = _placements(p2)
    auc1, auc2 = float(x1.mean()), float(x2.mean())
    v1 = np.var(x1, ddof=1) / n1 + np.var(y1, ddof=1) / n0
    v2 = np.var(x2, ddof=1) / n1 + np.var(y2, ddof=1) / n0
    c12 = (np.cov(x1, x2, ddof=1)[0, 1] / n1
           + np.cov(y1, y2, ddof=1)[0, 1] / n0)
    var_d = v1 + v2 - 2.0 * c12
    if var_d <= 0:
        return None
    z = (auc1 - auc2) / np.sqrt(var_d)
    return {'auc1': auc1, 'auc2': auc2, 'z': float(z),
            'p_value': float(2 * (1 - sps.norm.cdf(abs(z))))}


def learn_weights(rows):
    """四成员融合权重学习 + SEIR 门控 + 统计检验套件。

    成员：ml（机制标签）、ml_clin（临床规则标签，多样性工程）、gnn、seir。
    防泄露协议：接触者行按场景顺序整块排列，前 50% 场景用于学习
    （网格/校准器/LR 全部只看训练半区），后 50% 场景统一报告——
    学与报分离，所有融合方法在未见数据上检验。

    融合方法（按尺度处理递进）：
        grid_*      原始分数四成员网格加权（基线）
        rank_*      秩变换后网格加权（尺度免疫；批量评估用）
        cal_*       Platt 逐成员校准后网格加权（可部署）
        rank2_*     两方向 seir+gnn 秩变换网格（双 ML 摘除的对照基线）
        lr_*        OOF L2 正则 LR stacking（四成员；可部署）
        lr_gated_*  LR stacking + SEIR 门控交互项（seir 场景中心化值
                    及其与 ml/ml_clin/gnn 的乘积——SEIR 不作加权票，
                    只作人群分层调制信号）
        seir_prior_* 三票（ml+ml_clin+gnn）+ SEIR 场景级人群校正
                    （logit 空间加性调制；与 SEIR 人群动力学本职对齐）

    统计检验：各融合 vs 最优单方向的配对 DeLong 检验（测试半区）。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from tb_risk.constants import DEFAULT_ENSEMBLE_WEIGHTS as W

    keys = ('ml', 'ml_clin', 'gnn', 'seir')   # seir 末位=网格余项（历史权重~0）
    y, cols = _fusion_cols(rows, keys)
    half = len(rows) // 2                       # 场景整块切分
    tr, te = slice(0, half), slice(half, None)

    # --- 1) 原始分数：四成员网格（选择在训练半区，报告在测试半区）---
    gw, _gtr = _grid_search(y[tr], {k: v[tr] for k, v in cols.items()}, keys)
    grid_auc = float(roc_auc_score(y[te], _combo_prob(cols, gw, keys)[te]))
    w3 = W['three']   # 部署现行固定权重（三成员，ml=mech）作对照基线
    fixed_auc = float(roc_auc_score(
        y[te], (w3['ml'] * cols['ml'] + w3['seir'] * cols['seir']
                + w3['gnn'] * cols['gnn'])[te]))

    # --- 2) 秩变换：四成员网格（尺度免疫）---
    rcols = _rank_transform(cols)
    rw, _rtr = _grid_search(y[tr], {k: v[tr] for k, v in rcols.items()}, keys)
    rank_auc = float(roc_auc_score(y[te], _combo_prob(rcols, rw, keys)[te]))

    # --- 3) Platt 逐成员校准：网格（可部署融合）---
    ccols, _cal = _platt_cols(y, cols, tr, te)
    cw, _ctr = _grid_search(y[tr], {k: v[tr] for k, v in ccols.items()}, keys)
    cal_auc = float(roc_auc_score(y[te], _combo_prob(ccols, cw, keys)[te]))

    # --- 4) 两方向 seir+gnn（秩变换；双 ML 摘除对照）---
    r2 = {k: rcols[k] for k in ('seir', 'gnn')}
    r2w, _ = _grid_search(y[tr], {k: v[tr] for k, v in r2.items()},
                          keys=('seir', 'gnn'))
    r2p = r2w[0] * rcols['seir'] + r2w[1] * rcols['gnn']
    rank2_auc = float(roc_auc_score(y[te], r2p[te]))

    # --- 4b) SEIR 场景级人群信号（中心化；先验注入与门控共用）---
    s_idx = np.array([r['scenario_idx'] for r in rows])
    seir_scn = np.zeros(len(rows))
    for si in np.unique(s_idx):
        m = s_idx == si
        seir_scn[m] = cols['seir'][m].mean()
    seir_centered = seir_scn - cols['seir'].mean()
    eps = 1e-6

    def _logit(p):
        pc = np.clip(p, eps, 1 - eps)
        return np.log(pc / (1 - pc))

    # --- 4c) SEIR 先验注入：三票 + 场景级人群校正 ---
    # 架构：SEIR 从"第四票"改为"人群校正层"——其场景级均值（中心化）
    # 在 logit 空间对三票融合做加性调制。三票权重与校正强度在训练
    # 半区网格搜索（w_ml, w_clin, w_gnn=余项, beta 四参数）。
    # SEIR 只影响场景内整体偏移（人群风险分层），不参与个体间排序。
    best_sp = (None, -1.0)
    steps2 = np.arange(0.0, 1.0001, 0.1)
    betas = np.arange(-2.0, 2.01, 0.5)
    for w_ml in steps2:
        for w_clin in steps2:
            w_gnn = 1.0 - w_ml - w_clin
            if w_gnn < -1e-9:
                continue
            lg = _logit(w_ml * cols['ml'] + w_clin * cols['ml_clin']
                        + max(0.0, w_gnn) * cols['gnn'])
            for beta in betas:
                p = 1.0 / (1.0 + np.exp(-(lg + beta * seir_centered)))
                auc = roc_auc_score(y[tr], p[tr])
                if auc > best_sp[1]:
                    best_sp = ((float(w_ml), float(w_clin),
                                float(max(0.0, w_gnn)), float(beta)),
                               float(auc))
    w_ml_sp, w_clin_sp, w_gnn_sp, beta_sp = best_sp[0]
    lg = _logit(w_ml_sp * cols['ml'] + w_clin_sp * cols['ml_clin']
                + w_gnn_sp * cols['gnn'])
    sp_prob = 1.0 / (1.0 + np.exp(-(lg + beta_sp * seir_centered)))
    seir_prior_auc = float(roc_auc_score(y[te], sp_prob[te]))
    sp_weights = {'ml': w_ml_sp, 'ml_clin': w_clin_sp,
                  'gnn': w_gnn_sp, 'beta': beta_sp}

    # --- 5) OOF L2 正则 LR stacking（四成员）---
    # 元特征 = 四成员分数（基"模型"为固定打分器，无需逐折重训）；
    # 训练半区内 5 折 OOF 验证元学习器稳定性（oof_auc），最终
    # L2(C=1) LR 在整个训练半区拟合、测试半区报告
    X = np.column_stack([cols[k] for k in keys])
    oof = np.zeros(half)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for ftr, fva in skf.split(X[tr], y[tr]):
        m = LogisticRegression(C=1.0, max_iter=1000)
        m.fit(X[ftr], y[ftr])
        oof[fva] = m.predict_proba(X[fva])[:, 1]
    oof_auc = float(roc_auc_score(y[tr], oof))
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(X[tr], y[tr])
    lr_test_prob = lr.predict_proba(X[te])[:, 1]
    lr_auc = float(roc_auc_score(y[te], lr_test_prob))
    coefs = {k: float(c) for k, c in zip(keys, lr.coef_[0])}
    raw = {k: max(0.0, v) for k, v in coefs.items()}
    s = sum(raw.values())
    lr_weights = {k: (v / s if s > 0 else 1.0 / len(keys))
                  for k, v in raw.items()}

    # --- 5b) LR stacking + SEIR 门控交互项 ---
    # 元特征 = 三成员票 + seir_centered + seir_centered×{ml,ml_clin,gnn}
    # ——SEIR 不作加权票，只作人群分层调制（门控）信号进入 stacking。
    gate_keys = ('ml', 'ml_clin', 'gnn')
    Xg = np.column_stack(
        [cols[k] for k in gate_keys]
        + [seir_centered]
        + [seir_centered * cols[k] for k in gate_keys])
    g_names = (list(gate_keys) + ['seir_ctx']
               + [f'seir_x_{k}' for k in gate_keys])
    oof_g = np.zeros(half)
    for ftr, fva in skf.split(Xg[tr], y[tr]):
        m = LogisticRegression(C=1.0, max_iter=1000)
        m.fit(Xg[ftr], y[ftr])
        oof_g[fva] = m.predict_proba(Xg[fva])[:, 1]
    lr_gated_oof = float(roc_auc_score(y[tr], oof_g))
    lrg = LogisticRegression(C=1.0, max_iter=1000)
    lrg.fit(Xg[tr], y[tr])
    lr_gated_prob = lrg.predict_proba(Xg[te])[:, 1]
    lr_gated_auc = float(roc_auc_score(y[te], lr_gated_prob))
    lr_gated_coefs = {n: float(c) for n, c in zip(g_names, lrg.coef_[0])}

    # --- 6) DeLong 检验：各融合 vs 最优单方向（测试半区）---
    singles = {k: cols[k] for k in keys}
    best_k = max(singles, key=lambda k: roc_auc_score(y[te], singles[k][te]))
    fusion_te = {
        'grid': _combo_prob(cols, gw, keys)[te],
        'rank': _combo_prob(rcols, rw, keys)[te],
        'cal': _combo_prob(ccols, cw, keys)[te],
        'seir_prior': sp_prob[te],
        'lr_stack': lr_test_prob,
        'lr_gated': lr_gated_prob,
    }
    best_te = singles[best_k][te]
    delong = {}
    for fname, fprob in fusion_te.items():
        d = _delong_test(y[te], fprob, best_te)
        if d:
            delong[fname] = {'p_value': d['p_value'],
                             'delta_auc': d['auc1'] - d['auc2']}

    return {
        'grid4_best_weights': dict(zip(keys, gw)),
        'grid4_best_auroc': grid_auc,
        'grid_fixed_auroc': fixed_auc,
        'rank4_grid_best_weights': dict(zip(keys, rw)),
        'rank4_grid_best_auroc': rank_auc,
        'cal4_grid_best_weights': dict(zip(keys, cw)),
        'cal4_grid_best_auroc': cal_auc,
        'rank2_weights': {'seir': r2w[0], 'gnn': r2w[1]},
        'rank2_auroc': rank2_auc,
        'seir_prior_weights': sp_weights,
        'seir_prior_auroc': seir_prior_auc,
        'lr_coefs': coefs,
        'lr_weights': lr_weights,
        'lr_stack_auroc': lr_auc,
        'lr_oof_auroc': oof_auc,
        'lr_gated_coefs': lr_gated_coefs,
        'lr_gated_auroc': lr_gated_auc,
        'lr_gated_oof_auroc': lr_gated_oof,
        'delong_vs_best_single': delong,
        'best_single_name': best_k,
        'n_train_rows': int(half), 'n_test_rows': int(len(rows) - half),
    }


# ---------------------------------------------------------------------------
# 归档
# ---------------------------------------------------------------------------

def archive(rows, results, w3, n_scenarios, seed, duration,
            ml_path=None, ml_clin_path=None, gnn_path=None,
            extra_params=None,
            label_mode='mechanistic', weight_learning=None):
    """TrainingLogger 归档（model_type='ensemble'，追加不覆盖）。"""
    from train_with_public_data import get_logger  # 复用环境变量开关（同目录）

    y = [r['y'] for r in rows]
    pos_rate = float(np.mean(y))
    best_single = max(
        (v['AUROC'], k) for k, v in results.items()
        if k in ('ml_only', 'ml_clin_only', 'seir_only', 'gnn_only'))
    ens = results.get('ens_three', {})
    metrics = {
        'AUROC': ens.get('AUROC'),
        'AUPRC': ens.get('AUPRC'),
        'Brier': ens.get('Brier'),
        'positive_rate': pos_rate,
        'ml_only_AUROC': results.get('ml_only', {}).get('AUROC'),
        'ml_clin_only_AUROC': results.get('ml_clin_only', {}).get('AUROC'),
        'seir_only_AUROC': results.get('seir_only', {}).get('AUROC'),
        'gnn_only_AUROC': results.get('gnn_only', {}).get('AUROC'),
        'ens_ml_gnn_AUROC': results.get('ens_ml_gnn', {}).get('AUROC'),
        'ens_ml_seir_AUROC': results.get('ens_ml_seir', {}).get('AUROC'),
        'ens_four_AUROC': results.get('ens_four', {}).get('AUROC'),
        'ens_four_AUPRC': results.get('ens_four', {}).get('AUPRC'),
        'ens_four_Brier': results.get('ens_four', {}).get('Brier'),
        'best_single_AUROC': best_single[0],
        'best_single_name': best_single[1],
        'ensemble_vs_best_single': (
            ens.get('AUROC', 0.0) - best_single[0]) if ens else None,
        'ens_four_vs_best_single': (
            results['ens_four']['AUROC'] - best_single[0])
        if 'ens_four' in results else None,
    }
    # 权重学习结果入 metrics（vs 固定权重的直接对比）
    if weight_learning:
        for k in ('grid4_best_auroc', 'grid_fixed_auroc',
                  'rank4_grid_best_auroc', 'cal4_grid_best_auroc',
                  'rank2_auroc', 'seir_prior_auroc',
                  'lr_stack_auroc', 'lr_oof_auroc',
                  'lr_gated_auroc', 'lr_gated_oof_auroc'):
            if weight_learning.get(k) is not None:
                metrics[k] = weight_learning[k]
        metrics['grid_vs_fixed_auroc'] = (
            weight_learning['grid4_best_auroc']
            - weight_learning['grid_fixed_auroc'])
        dl = weight_learning.get('delong_vs_best_single') or {}
        if dl:
            # 带符号归档（2026-08-23）：delong_min_p 无方向信息，曾被误读为
            # "显著更好"；改为最强证据条目的 delta（正=融合更强）+ p + 方法名
            fname, d = min(dl.items(), key=lambda kv: kv[1]['p_value'])
            metrics['delong_best_delta'] = float(d['delta_auc'])
            metrics['delong_best_delta_p'] = float(d['p_value'])
            metrics['delong_best_delta_method'] = fname
        else:
            metrics['delong_best_delta'] = None
            metrics['delong_best_delta_p'] = None
            metrics['delong_best_delta_method'] = None
        metrics['best_single_name_learned'] = weight_learning.get(
            'best_single_name')
    # 模型标签从实际加载路径推断（历史教训：曾硬编码导致归档无法
    # 分辨对照实验与主实验，引发"同参数结果漂移"误判）
    ml_tag = os.path.basename(ml_path) if ml_path else 'default(mech)'
    ml_clin_tag = (os.path.basename(ml_clin_path)
                   if ml_clin_path else 'default(clin)')
    gnn_tag = os.path.basename(gnn_path) if gnn_path else 'default'
    params = {
        'n_scenarios': n_scenarios, 'seed': seed,
        'n_contacts': len(rows),
        'weights_three': w3,
        'label_mechanism': (
            'clinical rule (linear score + sigmoid, decoupled from SEIR '
            'force-of-infection form)'
            if label_mode == 'clinical' else
            'mechanistic transmission (1-exp(-beta*I*S))'),
        'ml_model': ml_tag,
        'ml_clin_model': ml_clin_tag,
        'gnn_model': gnn_tag,
    }
    if weight_learning:
        params['learned_weights'] = {
            'grid4_best': weight_learning['grid4_best_weights'],
            'rank4_grid_best': weight_learning['rank4_grid_best_weights'],
            'cal4_grid_best': weight_learning['cal4_grid_best_weights'],
            'rank2_seir_gnn': weight_learning['rank2_weights'],
            'seir_prior': weight_learning.get('seir_prior_weights'),
            'lr_coefs': weight_learning['lr_coefs'],
            'lr_weights': weight_learning['lr_weights'],
            'lr_gated_coefs': weight_learning.get('lr_gated_coefs'),
        }
    if extra_params:
        # metrics_override：多 seed 聚合的均值±std 覆盖单次值
        override = extra_params.pop('metrics_override', None)
        params.update(extra_params)
        if override:
            metrics.update(override)
    # 标签机制 × 成员构成双维度分桶：四成员 run 与历史三成员 run 不互相比较
    bucket = ('ensemble_eval_clinical_4m_v1' if label_mode == 'clinical'
              else 'ensemble_eval_mechanistic_4m_v1')
    dataset_info = {
        'source': bucket,
        'n_samples': len(rows),
        'real_data': False,
        'provenance': (
            'clinical-rule labels (linear score + sigmoid) on deployment-'
            'form scenarios; decoupled second benchmark to test SEIR '
            'circularity on mechanistic labels; four members: dual-ML '
            '(mech/clin labels) + GNN + SEIR'
            if label_mode == 'clinical' else
            'mechanistic transmission labels on deployment-form '
            'scenarios (patient + contacts); four members: dual-ML '
            '(mech/clin scenario-aligned) + GNN (scenario-aligned) + SEIR'),
    }
    logger = get_logger()
    entry = logger.log(
        model_type='ensemble', params=params, metrics=metrics,
        training_duration=duration, status='success', dataset=dataset_info)
    print(f"[归档] run_id={entry.run_id}（bucket={bucket}）")
    return metrics


# ---------------------------------------------------------------------------

def _print_report(rows, results, wl, label_mode):
    """单次评估的完整报告输出。"""
    print("\n=== 集成层评估结果 ===")
    print(f"接触者总数 {len(rows)}，阳性率 {np.mean([r['y'] for r in rows]):.3f}")
    print(f"{'组合':<14}{'AUROC':>8}{'AUPRC':>8}{'Brier':>8}{'n':>7}")
    for name, m in results.items():
        print(f"{name:<14}{m['AUROC']:>8.4f}{m['AUPRC']:>8.4f}"
              f"{m['Brier']:>8.4f}{m['n']:>7d}")

    singles = {k: v['AUROC'] for k, v in results.items()
               if k.endswith('_only')}
    if 'ens_three' in results:
        best_k = max(singles, key=singles.get)
        delta = results['ens_three']['AUROC'] - singles[best_k]
        print(f"\n[结论] 三方向集成 AUROC {results['ens_three']['AUROC']:.4f} "
              f"vs 最优单方向 {best_k} {singles[best_k]:.4f} "
              f"(差 {delta:+.4f})")
        print(f"       集成{'优于' if delta > 0 else '不优于'}最优单方向"
              f"{'，集成前提成立' if delta > 0 else '，权重需重新标定'}")

    if wl:
        gw = wl['grid4_best_weights']
        rw = wl['rank4_grid_best_weights']
        cw = wl['cal4_grid_best_weights']
        r2w = wl['rank2_weights']

        def _f4(w):
            return (f"ml={w['ml']:.2f} clin={w['ml_clin']:.2f} "
                    f"gnn={w['gnn']:.2f} seir={w['seir']:.2f}")

        print(f"\n[融合套件] 标签模式={label_mode} "
              f"(训练半区 {wl['n_train_rows']} 行 / 测试半区 "
              f"{wl['n_test_rows']} 行，学与报分离)")
        print(f"  四成员原始网格 : {_f4(gw)} -> {wl['grid4_best_auroc']:.4f} "
              f"(现行固定三成员为 {wl['grid_fixed_auroc']:.4f})")
        print(f"  四成员秩变换网格: {_f4(rw)} "
              f"-> {wl['rank4_grid_best_auroc']:.4f}")
        print(f"  四成员Platt网格: {_f4(cw)} "
              f"-> {wl['cal4_grid_best_auroc']:.4f}")
        print(f"  两方向 seir+gnn(秩): seir={r2w['seir']:.2f} "
              f"gnn={r2w['gnn']:.2f} -> {wl['rank2_auroc']:.4f}")
        spw = wl['seir_prior_weights']
        print(f"  SEIR 先验注入(三票+人群校正): ml={spw['ml']:.1f} "
              f"clin={spw['ml_clin']:.1f} gnn={spw['gnn']:.1f} "
              f"beta={spw['beta']:+.1f} -> {wl['seir_prior_auroc']:.4f}")
        print(f"  OOF L2 LR stacking: 系数 {wl['lr_coefs']} "
              f"-> 测试半区 {wl['lr_stack_auroc']:.4f} "
              f"(OOF {wl['lr_oof_auroc']:.4f})")
        print(f"  LR+SEIR 门控交互: -> 测试半区 {wl['lr_gated_auroc']:.4f} "
              f"(OOF {wl['lr_gated_oof_auroc']:.4f}) "
              f"系数 {wl['lr_gated_coefs']}")
        dl = wl.get('delong_vs_best_single') or {}
        for fn, d in dl.items():
            print(f"  DeLong {fn} vs {wl['best_single_name']}: "
                  f"delta={d['delta_auc']:+.4f} p={d['p_value']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=int, default=1,
                        help='多 seed 重复次数（>1 时聚合 mean±std 入档；'
                             '声明"集成优于单方向"前需 ≥5 seed 过噪声检验）')
    parser.add_argument('--label-mode', choices=['mechanistic', 'clinical'],
                        default='mechanistic',
                        help='标签机制：mechanistic=1-exp(-b*I*S)（与 SEIR '
                             '同构）；clinical=经验规则（去循环化第二基准）')
    parser.add_argument('--ml-model', default=None,
                        help='机制标签 ML 模型路径（默认场景对齐 185119）')
    parser.add_argument('--ml-model-clinical', default=None,
                        help='临床规则标签 ML 模型路径（默认场景对齐 185411）')
    parser.add_argument('--gnn-model', default=None,
                        help='GNN 模型路径（默认场景对齐 171114）')
    args = parser.parse_args()

    t0 = time.time()
    predictor, predictor_clin, ml_path, ml_clin_path, gnn_path = \
        load_models(args.ml_model, args.ml_model_clinical, args.gnn_model)

    n_seeds = max(1, args.seeds)
    per_seed = []
    diag_list = []
    for s in range(n_seeds):
        seed = args.seed + s
        scenarios = generate_scenarios(args.scenarios, seed,
                                       label_mode=args.label_mode)
        rows = score_scenarios(predictor, scenarios, predictor_clin)
        results, w3 = evaluate(rows)
        wl = learn_weights(rows)
        diag_list.append(correlation_diagnostics(rows, label=f'seed {seed}'))
        if n_seeds > 1:
            print(f"\n──── seed {seed} ────")
        _print_report(rows, results, wl, args.label_mode)
        per_seed.append({'seed': seed, 'rows': rows, 'results': results,
                         'wl': wl, 'w3': w3})
    duration = time.time() - t0

    if n_seeds == 1:
        d = per_seed[0]
        archive(d['rows'], d['results'], d['w3'], args.scenarios,
                args.seed, duration, ml_path=ml_path,
                ml_clin_path=ml_clin_path, gnn_path=gnn_path,
                label_mode=args.label_mode, weight_learning=d['wl'])
        print(f"\n总耗时 {duration:.1f}s")
        return

    # ---- 多 seed 聚合（均值±标准差 + 胜负计数 + DeLong 汇总）----
    def _agg(fn):
        vals = [fn(p) for p in per_seed]
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) \
            if len(vals) > 1 else 0.0

    print(f"\n=== 多 seed 聚合（{n_seeds} seeds: "
          f"{args.seed}..{args.seed + n_seeds - 1}）===")
    keys = ['ml_only_AUROC', 'ml_clin_only_AUROC', 'seir_only_AUROC',
            'gnn_only_AUROC', 'ens_three_AUROC', 'ens_four_AUROC']
    for k in keys:
        m, sd = _agg(lambda p, k=k: p['results'][k.replace(
            '_AUROC', '')]['AUROC'] if k.replace('_AUROC', '')
            in p['results'] else float('nan'))
        print(f"  {k:<20} {m:.4f} ± {sd:.4f}")
    for k in ('grid4_best_auroc', 'grid_fixed_auroc',
              'rank4_grid_best_auroc', 'cal4_grid_best_auroc',
              'rank2_auroc', 'seir_prior_auroc',
              'lr_stack_auroc', 'lr_gated_auroc'):
        vals = [p['wl'][k] for p in per_seed if p['wl'].get(k) is not None]
        if vals:
            m = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            print(f"  {k:<20} {m:.4f} ± {sd:.4f}")

    # 多样性诊断聚合（Spearman / 判错重叠率跨 seed 均值）
    if diag_list:
        pairs = diag_list[0]['pairs'].keys()
        print("  [多样性诊断] Spearman 均值: "
              + ' '.join(
                  f"{p}={np.mean([d['pairs'][p] for d in diag_list]):.3f}"
                  for p in pairs)
              + " | 判错重叠率均值: "
              + ' '.join(
                  f"{p}={np.mean([d['error_overlap'][p] for d in diag_list]):.3f}"
                  for p in pairs))

    # 融合 vs 最优单方向的胜负计数与 DeLong 汇总
    best_name_cnt = {}
    wins = {f: 0 for f in ('grid', 'rank', 'cal', 'seir_prior',
                           'lr_stack', 'lr_gated')}
    pvals = {f: [] for f in wins}
    deltas = {f: [] for f in wins}   # 带符号 delta（正=融合更强）
    for p in per_seed:
        singles = {k: v['AUROC'] for k, v in p['results'].items()
                   if k.endswith('_only')}
        best_auc = max(singles.values())
        best_name_cnt[max(singles, key=singles.get)] = \
            best_name_cnt.get(max(singles, key=singles.get), 0) + 1
        fmap = {'grid': p['wl']['grid4_best_auroc'],
                'rank': p['wl']['rank4_grid_best_auroc'],
                'cal': p['wl']['cal4_grid_best_auroc'],
                'seir_prior': p['wl'].get('seir_prior_auroc'),
                'lr_stack': p['wl']['lr_stack_auroc'],
                'lr_gated': p['wl'].get('lr_gated_auroc')}
        for f, auc in fmap.items():
            if auc is not None and auc > best_auc:
                wins[f] += 1
            dl = (p['wl'].get('delong_vs_best_single') or {}).get(f)
            if dl:
                pvals[f].append(dl['p_value'])
                deltas[f].append(dl['delta_auc'])
    print(f"  最优单方向分布: {best_name_cnt}")
    for f in wins:
        ps = pvals[f]
        pmsg = f" DeLong p: mean={np.mean(ps):.4f}" if ps else ""
        dmsg = (f" delta: mean={np.mean(deltas[f]):+.4f}"
                if deltas[f] else "")
        print(f"  {f} 胜最优单方向 {wins[f]}/{n_seeds} seeds{pmsg}{dmsg}")

    # 聚合归档：metrics 取均值，params 附 per_seed 明细
    agg_metrics = {}
    for k in ('ml_only', 'ml_clin_only', 'seir_only', 'gnn_only',
              'ens_three', 'ens_four'):
        m, sd = _agg(lambda p, k=k: p['results'][k]['AUROC']
                     if k in p['results'] else float('nan'))
        agg_metrics[f'{k}_AUROC'] = m
        agg_metrics[f'{k}_AUROC_std'] = sd
    for k in ('grid4_best_auroc', 'grid_fixed_auroc',
              'rank4_grid_best_auroc', 'cal4_grid_best_auroc',
              'rank2_auroc', 'seir_prior_auroc',
              'lr_stack_auroc', 'lr_gated_auroc'):
        vals = [p['wl'][k] for p in per_seed if p['wl'].get(k) is not None]
        if vals:
            agg_metrics[k] = float(np.mean(vals))
            agg_metrics[k + '_std'] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
    d0 = per_seed[0]
    archive(d0['rows'], d0['results'], d0['w3'], args.scenarios,
            args.seed, duration, ml_path=ml_path,
            ml_clin_path=ml_clin_path, gnn_path=gnn_path,
            label_mode=args.label_mode, weight_learning=d0['wl'],
            extra_params={
                'multi_seed': {
                    'n_seeds': n_seeds, 'seed_start': args.seed,
                    'wins_vs_best_single': wins,
                    'best_single_distribution': best_name_cnt,
                    'delong_p_mean': {f: float(np.mean(v))
                                      for f, v in pvals.items() if v},
                    'delong_delta_mean': {f: float(np.mean(v))
                                          for f, v in deltas.items() if v},
                    'diversity_diagnostics_mean': {
                        'spearman': {p: float(np.mean(
                            [d['pairs'][p] for d in diag_list]))
                            for p in diag_list[0]['pairs']},
                        'error_overlap': {p: float(np.mean(
                            [d['error_overlap'][p] for d in diag_list]))
                            for p in diag_list[0]['error_overlap']},
                    } if diag_list else None,
                },
                'metrics_override': agg_metrics,
            })
    print(f"\n总耗时 {duration:.1f}s（{n_seeds} seeds）")


if __name__ == '__main__':
    main()
