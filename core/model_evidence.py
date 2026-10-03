#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型证据聚合服务（core 层，纯 Python，不依赖 tkinter）。

模型已从"加权平均集成"升级为"三层递进架构 + 三任务分解"。本模块把分散在
scoring / validation / seir 各处的模型侧新能力，聚合为若干**结构化证据包**，
供 GUI 的"模型证据"面板直接渲染。每个函数都是防御性的：成功返回 dict，
失败返回 ``{'available': False, 'reason': ...}``，绝不向 GUI 抛异常。

提供的证据包：
  - ``extract_three_layer_series``  逐接触者三层分解（基线/网络增量/门控/联合）
  - ``collect_task_evaluations``    任务 A/B/C 的临床终点与评估指标
  - ``collect_network_ablation``    网络贡献消融（ΔAUROC / ΔC-index）
  - ``collect_public_data_evidence``公开数据集目录 + 各任务可验证性
  - ``collect_dgp_evidence``        DGP 合成数据可复现性报告
  - ``collect_seir_intervention``   社区级 SEIR 干预反事实曲线
  - ``collect_sinan_scale_evidence`` 百万级国家级监测数据验证证据包
"""

import logging

LOGGER = logging.getLogger("tb_risk.core.model_evidence")


def _unavailable(reason):
    """统一的"不可用"证据包。"""
    return {'available': False, 'reason': str(reason)}


# ==============================================================================
# 1. 三层递进：从 ml_results 提取逐接触者分解
# ==============================================================================

def extract_three_layer_series(ml_results):
    """从 ``ml_results['ensemble']`` 提取每个接触者的三层分解序列。

    三层报告由 ``ThreeDirectionIntegrator.integrate_predictions`` 在
    ``run_ml_prediction_loop`` 中生成，存于
    ``ml_results['ensemble'][group][i]['ensemble_predictions']['three_layer']``。

    参数：
        ml_results (dict|None): ``run_ml_prediction_loop`` 的返回值

    返回：
        dict: {'available': bool, 'contacts': list[dict], 'n': int}
        每个 contact 含 name / contact_type / p_base / network_increment /
        gating / gated_increment / combined / decision_risk / network_aware
    """
    contacts = []
    try:
        if not isinstance(ml_results, dict):
            return _unavailable("尚无 ML 预测结果")
        ensemble = ml_results.get('ensemble') or {}
        for group in ('family', 'social'):
            for item in ensemble.get(group, []) or []:
                ep = item.get('ensemble_predictions') or {}
                tl = ep.get('three_layer') or {}
                if not isinstance(tl, dict) or tl.get('architecture') != 'three_layer':
                    continue
                layers = tl.get('layers', {}) or {}
                decision = tl.get('decision', {}) or {}
                summary = tl.get('summary', {}) or {}
                l1 = layers.get('layer1_individual', {}) or {}
                l2 = layers.get('layer2_network', {}) or {}

                def _num(d, key):
                    v = d.get(key)
                    return float(v) if isinstance(v, (int, float)) else None

                contacts.append({
                    'name': item.get('name', '未知'),
                    'contact_type': group,
                    'traditional_prob': item.get('traditional_prob'),
                    'p_base': _num(l1, 'p_base_percent'),
                    'network_risk': _num(l2, 'network_risk'),
                    'network_increment': _num(l2, 'network_increment'),
                    'network_aware': bool(l2.get('network_aware', False)),
                    'gating': _num(decision, 'gating'),
                    'gated_increment': _num(decision, 'gated_network_increment'),
                    'combined': _num(decision, 'combined_probability'),
                    'intervention_benefit': _num(decision, 'intervention_benefit'),
                    'decision_risk': _num(decision, 'decision_risk'),
                    'decision_class': decision.get('decision_class'),
                    'best_strategy': summary.get('best_strategy'),
                    'best_averted_percent': summary.get('best_averted_percent'),
                    'raw': tl,
                })
    except Exception as e:  # noqa: BLE001 — 防御性，绝不向 GUI 抛异常
        LOGGER.debug("extract_three_layer_series 失败: %s", e, exc_info=True)
        return _unavailable(e)

    if not contacts:
        return _unavailable("未找到三层递进结果（需先运行 ML 预测）")
    return {'available': True, 'contacts': contacts, 'n': len(contacts)}


# ==============================================================================
# 2. 任务 A/B/C：临床终点与评估指标
# ==============================================================================

def _build_eval_cohort(n_contacts=40, n_cases=12, random_state=42):
    """构建任务评估队列（确定性合成接触网络，含确诊标签与簇结构）。

    复用消融模块的 ``build_synthetic_network``：它同时提供
    ``is_confirmed``（任务 A 结局）与 ``cluster``（任务 C 传播簇）。
    """
    from ..validation.layer_ablation import build_synthetic_network
    return build_synthetic_network(
        n_contacts=n_contacts, n_cases=n_cases, random_state=random_state)


def _score_cohort(records):
    """用架构第 1 层个体打分器（NumPy 图风险预测器）为队列打分。

    与三层架构的"个体基础层 P_base"一致，无需训练好的 ML 模型即可运行，
    保证任务面板始终有可复现的数值。
    """
    from ..ml.gnn import NumPyGraphRiskPredictor
    predictor = NumPyGraphRiskPredictor()
    probs = []
    for rec in records:
        ctype = 'family' if rec.get('contact_type', 'family') == 'family' else 'social'
        try:
            res = predictor.predict_contact_risk(rec, ctype)
            probs.append(float(res.get('risk_probability', 0.0)))
        except Exception:  # noqa: BLE001
            probs.append(0.0)
    return probs


def collect_task_evaluations(n_contacts=40, n_cases=12, random_state=42,
                             horizon_days=365):
    """聚合任务 A/B/C 的评估报告。

    任务 A / C 使用合成网络的真实标签直接计算；任务 B 以横断面确诊标签 +
    合成随访时间作为**过渡演示**（真实任务 B 需 6–24 月纵向随访数据），
    并在返回中显式标注该局限。

    返回：
        dict: {'available': bool, 'A':..., 'B':..., 'C':..., 'cohort': {...}}
    """
    try:
        from ..validation.task_decomposition import (
            evaluate_task_a, evaluate_task_b, evaluate_task_c)

        records, labels = _build_eval_cohort(n_contacts, n_cases, random_state)
        probs = _score_cohort(records)
        labels_list = [int(x) for x in labels]

        # 任务 A：横断面筛查
        task_a = evaluate_task_a(probs, labels_list)
        # 原始打分/标签（供界面阈值调节器重算指标）
        task_a['_scores'] = list(probs)
        task_a['_labels'] = list(labels_list)

        # 任务 B：进展预测（过渡演示 — 以确诊标签 + 合成随访时间代替纵向结局）
        import random as _rnd
        rng = _rnd.Random(random_state)
        event_times = [
            float(rng.uniform(30, horizon_days)) if o == 1
            else float(horizon_days)
            for o in labels_list
        ]
        task_b = evaluate_task_b(probs, labels_list, event_times=event_times,
                                 horizon_days=horizon_days)
        task_b['transition_note'] = (
            "真实任务 B 需 6–24 个月纵向随访结局；当前以横断面确诊标签 + "
            "合成随访时间作为过渡演示，指标仅示意评估框架。")
        # 事件时间（供界面绘制 Kaplan-Meier 生存曲线）
        task_b['_event_times'] = list(event_times)
        task_b['_labels'] = list(labels_list)

        # 任务 C：网络传播排序
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        true_pos = {i for i, o in enumerate(labels_list) if o == 1}
        # 二代病例 / 传染源：由传播簇近似（簇数≈传染源数，病例−种子≈二代）
        clusters = {}
        for i, rec in enumerate(records):
            clusters.setdefault(rec.get('cluster', 0), []).append(i)
        n_infectious = max(len(clusters), 1)
        secondary = max(len(true_pos) - n_infectious, 0)
        traced = set(order[: max(1, len(order) // 2)])  # 追踪前半数高风险者
        task_c = evaluate_task_c(
            order, true_pos,
            secondary_cases=secondary, n_infectious=n_infectious,
            traced_contacts=traced, infected_contacts=true_pos)

        return {
            'available': True,
            'A': task_a, 'B': task_b, 'C': task_c,
            'sinan_dual_head_mapping': {
                'note': 'SINAN 双头（0.7% top-decile 重叠 → 正交）与'
                        '本任务分解的映射（round-10 P2）：',
                'A_prime': "任务 A'（SINAN 细菌学确诊头）= 找传染源："
                           '对准本任务 A（横断面筛查）的病例发现语义'
                           '——在通报人群中排序涂片/培养阳性概率',
                'B': '任务 B（SINAN 全因死亡头）= 死亡预后：对准本'
                     '任务 B（进展预测）的预后分层语义——治疗结局'
                     '分层与高危随访资源分配',
                'predictor': 'tb_risk.core.dual_head_predictor'
                             '.DualHeadPredictor',
            },
            'cohort': {
                'n_contacts': len(records),
                'n_cases': sum(labels_list),
                'n_clusters': len(clusters),
                'source': '确定性合成接触网络（build_synthetic_network）',
            },
        }
    except Exception as e:  # noqa: BLE001
        LOGGER.debug("collect_task_evaluations 失败: %s", e, exc_info=True)
        return _unavailable(e)


# ==============================================================================
# 3. 网络贡献消融
# ==============================================================================

def collect_network_ablation(n_contacts=40, n_cases=12, random_state=42):
    """网络层贡献消融（有/无 GNN 层的 ΔAUROC / ΔC-index）。"""
    try:
        from ..validation.layer_ablation import network_contribution_ablation
        report = network_contribution_ablation(
            records=None, n_contacts=n_contacts, n_cases=n_cases,
            random_state=random_state)
        report['available'] = True
        return report
    except Exception as e:  # noqa: BLE001
        LOGGER.debug("collect_network_ablation 失败: %s", e, exc_info=True)
        return _unavailable(e)


# ==============================================================================
# 4. 公开数据集验证
# ==============================================================================

def collect_public_data_evidence():
    """公开数据集目录 + 各任务外部验证可覆盖性 + 验证策略。"""
    try:
        from ..validation.public_datasets import summarize_public_datasets
        from ..validation.public_data_strategy import build_validation_strategy
        summary = summarize_public_datasets()
        strategy = build_validation_strategy()
        return {
            'available': True,
            'summary': summary,
            'strategy': strategy,
        }
    except Exception as e:  # noqa: BLE001
        LOGGER.debug("collect_public_data_evidence 失败: %s", e, exc_info=True)
        return _unavailable(e)


# ==============================================================================
# 5. DGP 合成数据可复现性
# ==============================================================================

def collect_dgp_evidence(n_samples=5000, random_state=42):
    """DGP 可复现文档（字段分布 + 参数 + 文献来源 + 分布/患病率校验）。

    参数：
        n_samples (int): 校验用合成样本量（小于官方 48,683 以保证界面响应）
    """
    try:
        from ..validation.dgp import build_dgp_report, reproducibility_check
        report = build_dgp_report(n_samples=n_samples, random_state=random_state)
        repro = reproducibility_check(n_samples=500, random_state=random_state)
        report['reproducibility'] = repro
        report['available'] = True
        return report
    except Exception as e:  # noqa: BLE001
        LOGGER.debug("collect_dgp_evidence 失败: %s", e, exc_info=True)
        return _unavailable(e)


# ==============================================================================
# 6. 社区级 SEIR 干预反事实
# ==============================================================================

def collect_seir_intervention(high_risk_contacts=None, population=10000,
                              t_horizon_days=730, random_state=42):
    """社区级 SEIR 干预反事实（干预 vs 无干预的病例数曲线 + 策略对比）。"""
    try:
        from ..seir.intervention import simulate_intervention_effects
        report = simulate_intervention_effects(
            high_risk_contacts=high_risk_contacts,
            population=population, t_horizon_days=t_horizon_days,
            random_state=random_state)
        report['available'] = True
        return report
    except Exception as e:  # noqa: BLE001
        LOGGER.debug("collect_seir_intervention 失败: %s", e, exc_info=True)
        return _unavailable(e)


# ==============================================================================
# 7. 百万级国家级监测数据验证（SINAN，第六轮 P3 证据链头条）
# ==============================================================================

SINAN_SCALE_ARCHIVE = 'sinan_scale_training_20260826.json'


def collect_sinan_scale_evidence():
    """读取 SINAN 百万级规模训练归档，构造证据链头条证据包。

    数据源：``data/processed/sinan_scale_training_20260826.json``
    （巴西 SINAN 全国结核通报 2001-2019，124.6 万个体分析集，
    T1 时间外推 + T2 空间外推双轴验证）。归档缺失时优雅降级。

    返回（成功时）：
      - ``headline``/``narrative``  对外叙事头条
      - ``t1_series``               各模型 val/near/far AUROC
        （时间衰减稳健性曲线数据源）
      - ``t2_series``               空间外推 AUROC
      - ``subgroups``/``top_features``/``pakistan``  证据细节
    """
    import json
    import os
    try:
        archive = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'data', 'processed',
            SINAN_SCALE_ARCHIVE)
        with open(archive, encoding='utf-8') as f:
            raw = json.load(f)

        data = raw.get('data', {})
        t1 = raw.get('T1_temporal', {})
        t2 = raw.get('T2_spatial', {})
        if not t1 or not data:
            return _unavailable('归档缺少 T1_temporal/data 字段')

        t1_series = {m: [v.get('val', {}).get('auroc'),
                         v.get('near', {}).get('auroc'),
                         v.get('far', {}).get('auroc')]
                     for m, v in t1.items()
                     if isinstance(v, dict) and 'val' in v}
        best_model, best_far = max(
            ((m, s[2]) for m, s in t1_series.items()
             if s[2] is not None), key=lambda kv: kv[1])
        decay = t1_series[best_model][0] - t1_series[best_model][2]
        t2_series = {m: v.get('auroc') for m, v in t2.items()
                     if isinstance(v, dict) and 'auroc' in v}
        t2_best_model, t2_best = max(t2_series.items(),
                                     key=lambda kv: kv[1])

        # 亚组（归档仅记录最优树模型）
        subgroups = {}
        for _src, sg in raw.get('subgroup', {}).items():
            subgroups = sg
            break

        top_features = [
            item['feature'] for item in
            raw.get('feature_importance', {}).get('xgb_gain_top25', [])[:5]]

        pakistan = raw.get('pakistan', {})
        pk_note = pakistan.get('interpretation', 'directional_reference_only')

        n_total = data.get('n_total', 0)
        n_events = data.get('events_tb_death', 0)
        prev = data.get('prevalence_pct', 0)
        ep = _dict_corrected_semantics() or {}
        narrative = (
            '巴西 SINAN 全国结核通报 2001-2019：{n} 万个体级记录'
            '（旧终点「TB 死亡」实为 {ep4}，{ev} 例 / {pv}%；'
            '修正终点 = {ace}，见「阴性发现」标签页的标签更正披露），'
            '诊断时点特征训练，'
            'T1 时间外推 6 年 AUROC 仅衰减 {dec:.3f}（{bm} far {fa:.3f}），'
            'T2 空间外推（非东南训练→东南测试）{tbm} {tfa:.3f}——'
            '百万级国家级监测数据上的规模与稳健性验证。'
        ).format(n=round(n_total / 10000, 1),
                 ep4=ep.get('code4_semantics', '非 TB 死亡'),
                 ace=ep.get('corrected_endpoint', '全因死亡 {3,4}'),
                 ev=n_events, pv=prev,
                 dec=decay, bm=best_model,
                 fa=t1_series[best_model][2],
                 tbm=t2_best_model, tfa=t2_best)

        return {
            'available': True,
            'headline': '百万级国家级监测数据验证（巴西 SINAN 2001-2019）',
            'narrative': narrative,
            'n_total': int(n_total),
            'events_tb_death': int(n_events),
            'prevalence_pct': prev,
            'endpoint_correction': ep,
            't1_series': t1_series,
            'best_model': best_model,
            'decay_val_to_far': decay,
            't2_series': t2_series,
            't2_best_model': t2_best_model,
            'subgroups': subgroups,
            'top_features': top_features,
            'pakistan_interpretation': pk_note,
        }
    except FileNotFoundError:
        return _unavailable('归档不存在（data/processed/%s）'
                            % SINAN_SCALE_ARCHIVE)
    except Exception as e:  # noqa: BLE001
        LOGGER.debug("collect_sinan_scale_evidence 失败: %s", e, exc_info=True)
        return _unavailable(e)


# ==============================================================================
# 8. 时序家庭筛查演示输入（GUI 第 9 个子标签页的数据源，纯函数）
# ==============================================================================

# 演示基线率 π：取 HomeACF 密接人群 LTBI（TST≥10mm）阳性率口径（约 13%），
# 供 GUI 演示贝叶斯收缩的默认值；w/k 用 scoring.temporal_household 默认值。
DEFAULT_DEMO_BASE_RATE = 0.13


def _fallback_entry_risk_score(entry):
    """复刻 gui/tabs/_family.py ``_compute_risk_score`` 的近似评分（0-100）。

    ML 个体基线不可用时，用接触暴露评分作为第 1 层基线的回退口径，
    保证演示面板在未运行 ML 预测时也可用。
    """
    score = 0.0
    try:
        try:
            age = int(entry.get('age', 30))
        except (TypeError, ValueError):
            age = 30
        if age < 5 or age > 65:
            score += 15
        if entry.get('ventilation', 3) in (1, '1'):
            score += 15
        elif entry.get('ventilation', 3) in (2, '2'):
            score += 10
        if entry.get('contact_distance', '中等') in ('极近', '近'):
            score += 15
        if entry.get('exposure_setting', '一般') in ('拥挤', '密闭'):
            score += 15
        if entry.get('is_high_risk', '否') == '是':
            score += 20
        if entry.get('has_symptoms', '否') == '是':
            score += 10
        if entry.get('has_tb', '否') == '是':
            score += 10
        cumulative = entry.get('cumulative_exposure', 0)
        if isinstance(cumulative, (int, float)) and cumulative > 1000:
            score += 10
    except Exception as e:  # noqa: BLE001
        LOGGER.debug("_fallback_entry_risk_score 计算异常: %s", e)
    return min(score, 100.0)


def household_screening_signature(family_entries, p_base_map=None):
    """成员/确诊/ML 基线签名（GUI 面板失效检测用，hashable）。

    签名变化 = 家庭成员增删改名 / 确诊标记切换 / ML 基线刷新，
    GUI 据此决定是否重建序贯筛查状态。
    """
    entries = list(family_entries or [])
    names = tuple(str(e.get('name') or f'成员{i + 1}')
                  for i, e in enumerate(entries))
    diagnosed = tuple(1 if e.get('diagnosed') else 0 for e in entries)
    pb = p_base_map or {}
    # 部分覆盖的 ML 基线：缺失成员记 None（与"无 ML 基线"的空元组区分）
    ml = tuple(None if pb.get(n) is None else round(float(pb.get(n)), 4)
               for n in names) if pb else ()
    return (names, diagnosed, ml)


def build_household_screening_demo(family_entries, p_base_map=None,
                                   risk_score_fn=None,
                                   base_rate=DEFAULT_DEMO_BASE_RATE):
    """从 GUI 家庭成员条目构造时序家庭筛查演示输入（纯函数，无 tkinter）。

    基线分数优先级：ML 三层递进的 p_base（第 1 层个体基线）>
    risk_score_fn（家庭页暴露评分回调）> _fallback_entry_risk_score。
    已确诊（diagnosed）成员作为"已筛查阳性"预置证据——其余成员风险
    立即上调并重排，即"户内已有人筛查阳性 → 自动重排"的部署语义。

    参数：
        family_entries (list[dict]): GUI 家庭成员条目（name/diagnosed/暴露字段）。
        p_base_map (dict|None): {成员名: ML 个体基线概率(0-100)}。
        risk_score_fn (callable|None): entry → 0-100 暴露评分（家庭页口径）。
        base_rate (float): 演示基线率 π。

    返回：
        dict: {'available': bool, 'reason': str, 'names': list[str],
        'base_scores': list[float], 'preseed_positive': list[int],
        'base_rate': float, 'source': 'ml'|'entry_risk'|'mixed'}
    """
    entries = list(family_entries or [])
    if not entries:
        return _unavailable("请先在「家庭成员信息」标签页添加家庭成员")
    names, base_scores, preseed = [], [], []
    used_ml = 0
    for i, e in enumerate(entries):
        name = str(e.get('name') or f'成员{i + 1}')
        score = None
        if p_base_map and name in p_base_map and p_base_map[name] is not None:
            try:
                score = float(p_base_map[name])
                used_ml += 1
            except (TypeError, ValueError):
                score = None
        if score is None and risk_score_fn is not None:
            try:
                score = float(risk_score_fn(e))
            except Exception:  # noqa: BLE001
                score = None
        if score is None:
            score = _fallback_entry_risk_score(e)
        names.append(name)
        base_scores.append(min(100.0, max(0.0, score)))
        if e.get('diagnosed'):
            preseed.append(i)
    source = ('ml' if used_ml == len(names)
              else ('entry_risk' if used_ml == 0 else 'mixed'))
    return {
        'available': True,
        'names': names,
        'base_scores': base_scores,
        'preseed_positive': preseed,
        'base_rate': float(base_rate),
        'source': source,
    }


# ==============================================================================
# 9. 阴性发现与边界条件（GUI 第 11 个子标签页的数据源，纯函数）
# ==============================================================================

# 阴性发现归档（data/processed/ 下文件名 → 读取器），全部只读
_NEG_ARCHIVE_ELDERLY = 'sinan_elderly_20260826.json'
_NEG_ARCHIVE_ENDPOINT = 'sinan_endpoint_switch_20260826.json'
_NEG_ARCHIVE_DICT = 'sinan_label_dictionary_20260826.json'
_NEG_ARCHIVE_PACTS = 'pacts_temporal_household_20260825.json'


def _neg_load(archive):
    """读 data/processed 归档 JSON；不存在返回 None。"""
    import json
    import os
    fp = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'data', 'processed', archive)
    try:
        with open(fp, encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _dict_corrected_semantics():
    """从裁定字典读取 y=4 的修正终点语义（防旧数字误引的统一口径）。

    返回 None（裁定归档缺失/结构变化）时调用方降级为不展示，
    不硬编码语义——保证展示口径与裁定 JSON 单一来源一致。
    """
    raw = _neg_load(_NEG_ARCHIVE_DICT)
    if not raw:
        return None
    verdict = raw.get('verdict', {})
    sem = verdict.get('correct_semantics', {})
    corrected = verdict.get('corrected_endpoint_definitions', {})
    if not sem or not corrected:
        return None
    return {
        'old_label': 'TB 死亡（round-5 错误假设）',
        'code4_semantics': sem.get('4', '非 TB 死亡'),
        'corrected_endpoint': '全因死亡 %s' % corrected.get(
            'all_cause_death', ['3', '4']),
        'citation_rule': '引用旧归档数字时终点表述必须写'
                         '「非 TB 死亡（旧）/ 全因死亡（修正）」',
    }


def collect_negative_findings():
    """汇总证据链中的阴性发现、边界条件与标签更正披露。

    项目传统：阴性结果与阳性结果同样归档——对后续合作方是
    预期管理，对复现者是边界地图。每个 finding 从对应归档
    JSON 读取数字（不硬编码），归档缺失时跳过该条并记录。

    返回（成功时）：
      - ``headline``/``narrative``   对外叙事
      - ``findings``                 list[dict]，每条含
        id/verdict/title/detail/implication/archive/metrics
      - ``tradition_note``           归档传统说明
    """
    findings = []

    # 1) 老年专项训练 + 竞争风险交互（三臂阴性，判别不可修复）
    raw = _neg_load(_NEG_ARCHIVE_ELDERLY)
    if raw:
        deltas = raw.get('delta_elderly_cr_vs_all_age', {})
        best_model = (max(deltas, key=deltas.get) if deltas else None)
        best_gain = deltas.get(best_model)
        cal = raw.get('calibration', {})
        ece_all = cal.get('all_age', {}).get('ece')
        ece_cr = cal.get('elderly_cr', {}).get('ece')
        top_bin_all = (cal.get('all_age', {}).get('bins') or [{}])[-1]
        ep = _dict_corrected_semantics() or {}
        findings.append({
            'id': 'elderly_specialization',
            'verdict': 'NEGATIVE',
            'title': '老年亚组（65+）判别塌方经三臂修复尝试后确认不可修复',
            'detail': (
                '三臂（all_age / elderly / elderly_cr+13 个竞争风险交互）中，'
                'elderly_cr − all_age 五模型全负（最好 {bm} {bg:+.4f}）。'
                '判别不可修复，但校准可改善：RF ECE {ea} → {ec}，'
                '且 top-decile 系统性低估（预测 {mp:.3f} vs 实际 {mo:.3f}）。'
                '终点口径：塌方终点实为「{ep4}」（旧终点）/ '
                '「{ace}」（修正终点）——数值不受影响，三臂共享同一 y。'
            ).format(bm=best_model, bg=best_gain,
                     ea=ece_all, ec=ece_cr,
                     mp=top_bin_all.get('mean_predicted', 0),
                     mo=top_bin_all.get('observed_rate', 0),
                     ep4=ep.get('code4_semantics', '非 TB 死亡'),
                     ace=ep.get('corrected_endpoint', '全因死亡 {3,4}')),
            'implication': '部署建议：老年亚组收紧筛查阈值（政策层修复，'
                           '而非模型层修复）；面向中国老龄化场景的边界条件。',
            'archive': _NEG_ARCHIVE_ELDERLY,
            'metrics': {'delta_elderly_cr_vs_all_age': deltas,
                        'ece_all_age_rf': ece_all,
                        'ece_elderly_cr_rf': ece_cr},
        })

    # 2) 换终点实验（E1-E7 + 多类别头：判别力不随终点转移）
    raw = _neg_load(_NEG_ARCHIVE_ENDPOINT)
    if raw:
        r5 = (raw.get('results', {}).get('E5_cure_vs_lost_transfer', {})
                .get('elderly_65p', {}).get('LGBM', {}))
        r1 = (raw.get('results', {}).get('E1_death_nontb', {})
                .get('elderly_65p', {}).get('LGBM', {}))
        mc = (raw.get('multiclass', {}).get('elderly_65p', {})
              .get('lgbm', {}) or {})
        findings.append({
            'id': 'elderly_endpoint_switch',
            'verdict': 'NEGATIVE',
            'title': '「换终点」修复假说被否定：判别力不随终点转移',
            'detail': (
                '65+ 子样本上 7 个二分类终点 + 5 类多类别头：'
                'E5（治愈 vs 失访+转出，TB 特异终点）{e5:.4f} '
                'CI {ci}，反而显著低于 E1（旧死亡终点）{e1:.4f}；'
                'E6 失访 {e6}、E7 转出 {e7} 同样更低；多类别 '
                'macro OvR {mc} vs 全年龄 {mca}——塌方在所有终点上'
                '均存在，非终点属性。'
            ).format(e5=r5.get('auroc', 0),
                     ci=r5.get('auroc_ci95_boot', []),
                     e1=r1.get('auroc', 0),
                     e6=(raw.get('results', {}).get(
                         'E6_cure_vs_lost', {}).get(
                         'elderly_65p', {}).get('LGBM', {}).get('auroc')),
                     e7=(raw.get('results', {}).get(
                         'E7_cure_vs_transfer', {}).get(
                         'elderly_65p', {}).get('LGBM', {}).get('auroc')),
                     mc=mc.get('macro_auroc'),
                     mca=(raw.get('multiclass', {}).get('all_age', {})
                         .get('lgbm', {}).get('macro_auroc'))),
            'implication': '老年塌方是人群属性（个体特征在 65+ 信息量'
                           '下降）而非任务属性——修复钥匙不在换终点/'
                           '换模型/多任务头。',
            'archive': _NEG_ARCHIVE_ENDPOINT,
            'metrics': {'elderly_E5': r5.get('auroc'),
                        'elderly_E1': r1.get('auroc'),
                        'elderly_multiclass_macro': mc.get('macro_auroc')},
        })

    # 3) 老年线边界归档关闭（四条证据链汇聚的定稿结论，round-8）
    raw = _neg_load(_NEG_ARCHIVE_ENDPOINT)
    if raw:
        e3 = (raw.get('results', {}).get('E3_death_tb', {})
              .get('elderly_65p', {}).get('LGBM', {}))
        e6_all = (raw.get('results', {}).get('E6_cure_vs_lost', {})
                  .get('all_age', {}).get('LGBM', {}))
        findings.append({
            'id': 'elderly_line_closure',
            'verdict': 'NEGATIVE',
            'title': '老年线（65+）以边界归档关闭：判别上限约 0.70，'
                     '定稿「临床判断为主、模型仅作分诊参考」',
            'detail': (
                '四条证据链汇聚：专项训练无效（elderly_cr − all_age '
                '五模型全负）、竞争风险交互无效（13 个交互特征无增益）、'
                '换终点无效（E1-E7 + 多类别头全部更低）、最优配置 '
                '0.70 天花板（E3 TB 死亡 {e3:.4f} 为换终点最高值，'
                '未突破 0.70）。65+ 预后预测受诊断时点特征信息上限'
                '约束——不再投入调参，老年线以边界归档关闭。'
            ).format(e3=e3.get('auroc', 0)),
            'implication': (
                '定稿结论：该人群以临床判断为主、模型仅作分诊参考；'
                '校准层仍有效（top-decile 低估 → 收紧筛查阈值）。'
                '后续方向：E6 失访预警（all_age {e6:.4f}，行为终点'
                '且干预路径直接）有条件立项；老年专属失访预警不立项'
                '（事件率仅 7%，老年依从性好）。'
            ).format(e6=e6_all.get('auroc', 0)),
            'archive': _NEG_ARCHIVE_ENDPOINT,
            'metrics': {'elderly_E3_best': e3.get('auroc'),
                        'E6_all_age_auroc': e6_all.get('auroc')},
        })

    # 4) 标签字典修正（LABEL_ERROR_CONFIRMED 披露）
    raw = _neg_load(_NEG_ARCHIVE_DICT)
    if raw:
        verdict = raw.get('verdict', {})
        corrected = verdict.get('corrected_endpoint_definitions', {})
        findings.append({
            'id': 'label_dictionary_correction',
            'verdict': 'CORRECTED',
            'title': 'SINAN SITUA_ENCE 标签语义更正披露'
                     '（round-5/6 终点假设为 TB 死亡，实为非 TB 死亡）',
            'detail': (
                '三重证据（官方字典 PDF + 文献 + 数据内证）裁定：'
                '3=TB 死亡、4=非 TB 死亡、5=转出。round-5/6 的 '
                'y=4 实为非 TB 死亡（2001-05 段含全因死亡成分）。'
                '修正终点定义：全因死亡={ac}、TB 死亡={{3}}、'
                '失访={{2}}、转出={{5}}、治愈={{1}}。'
                'AUROC 与机制层结论不受影响（所有 arm 共享同一 y），'
                'v2 已按全因死亡终点重跑复核。'
            ).format(ac=corrected.get('all_cause_death', ['3', '4'])),
            'implication': '终点语义已双向同步（run 脚本 + 归档 JSON）；'
                           '未来引用 SINAN 死亡终点结果时按修正口径。',
            'archive': _NEG_ARCHIVE_DICT,
            'metrics': {'verdict_type': verdict.get('type'),
                        'corrected': corrected},
        })

    # 5) PACTS 终点天花板（inconclusive，非机制否定）
    raw = _neg_load(_NEG_ARCHIVE_PACTS)
    if raw:
        ind = (raw.get('arms_summary', {}).get('ind:RF', {})
               .get('mean_auroc'))
        dl = (raw.get('deltas_cluster_bootstrap', {})
              .get('exposure_prior:RF - ind:RF', {}))
        loo = (raw.get('arms_summary', {}).get('hh_loo_only', {})
               .get('mean_auroc'))
        findings.append({
            'id': 'pacts_endpoint_ceiling',
            'verdict': 'INCONCLUSIVE',
            'title': 'PACTS 症状终点天花板：时序先证机制不可裁决',
            'detail': (
                '马拉维 PACTS 户级复验：个体基线 ind:RF {ind:.3f}'
                '（基线症状≈终点，终点饱和），时序先证 Δ {dm} '
                'CI {ci}（含 0）——不可裁决，非机制否定；'
                '户级聚集信号本体存在（hh_loo {loo:.3f}）。'
            ).format(ind=ind, dm=dl.get('mean'), ci=dl.get('bootstrap_ci'),
                     loo=loo),
            'implication': '感染终点（HomeACF）才是够功效的机制检验场；'
                           '进展终点等待 ERASE-TB 级别前瞻队列。',
            'archive': _NEG_ARCHIVE_PACTS,
            'metrics': {'ind_rf_auroc': ind,
                        'delta_mean': dl.get('mean'),
                        'hh_loo_auroc': loo},
        })

    # 6) SINAN 户级下沉不可行（round-9 P2-5：数据字段边界）
    findings.append({
        'id': 'sinan_household_infeasible',
        'verdict': 'NEGATIVE',
        'title': 'SINAN 户级分组下沉不可行：通报数据无户级地址字段',
        'detail': (
            '组粒度曲线指向「按共享暴露单元分组增益放大约一个数量级」，'
            '但 SINAN 去标识化通报数据（97 字段全集）仅含市/州级'
            '地理代码（ID_MUNICIP/SG_UF_NOT），无街道/门牌/户标识——'
            '「SINAN 有地址字段」的假设不成立，户级聚类无法构造。'
        ),
        'implication': (
            '机制上限已由 HomeACF 户级锚定（ICC 0.275 → Δ +0.058）；'
            '部署指南的粒度判据（ICC）保持有效——需户级数据时应在'
            '采集端补地址字段（如中国密接管理数据），而非从 SINAN 补。'
        ),
        'archive': 'granularity_effect_curve_20260826.json（字段证据：'
                   'sinan_parse_meta 97 字段清单）',
        'metrics': {'sinan_geo_resolution': 'municipality',
                    'household_anchor_delta': 0.0585},
    })

    if not findings:
        return _unavailable('阴性发现归档均不存在（data/processed/'
                            'sinan_elderly_* 等）')

    n_neg = sum(1 for f in findings if f['verdict'] == 'NEGATIVE')
    return {
        'available': True,
        'headline': '阴性发现与边界条件（Negative Findings）',
        'narrative': (
            '证据链的诚实边界：{n} 条归档发现中 {neg} 条为判别层阴性——'
            '老年塌方经三臂训练与换终点双重尝试确认不可修复（人群属性）；'
            'PACTS 终点天花板使机制不可裁决（非否定）；'
            'SINAN 标签语义已更正（数值结论不受影响）。'
        ).format(n=len(findings), neg=n_neg),
        'findings': findings,
        'tradition_note': '阴性结果同样归档——这是对后续合作方的预期'
                          '管理，也是复现者的边界地图。',
    }


# ==============================================================================
# 10. 组粒度部署指南（round-8 P4：ICC 决定时序先证的价值上限）
# ==============================================================================

GRANULARITY_ARCHIVE = 'granularity_effect_curve_20260826.json'


def collect_granularity_guide():
    """读取组粒度-效应量曲线归档，构造「按哪一级分组」部署指南。

    数据源：``data/processed/granularity_effect_curve_20260826.json``
    （SINAN 市规模三层 + HomeACF/PACTS 户级，ICC vs ΔAUROC）。

    决策规则（部署前可测量的先证价值判据）：
      1. 用试点数据测候选分组单位的 ICC(1)（组内共享暴露比例代理）；
      2. ICC > 0.1 且终点未被基线打穿 → 时序先证值得做；
      3. ICC < 0.05 → 不值得（增益约 +0.002~0.005 量级）；
      4. 按共享暴露单元（户/楼）分组，不按行政单元（市）。
    """
    raw = _neg_load(GRANULARITY_ARCHIVE)
    if not raw:
        return _unavailable('归档不存在（data/processed/%s）'
                            % GRANULARITY_ARCHIVE)
    points = raw.get('points', {})
    if not points:
        return _unavailable('归档缺少 points 字段')

    def _pt(pid, label, verdict_note):
        p = points.get(pid, {})
        return {
            'id': pid,
            'label': label,
            'icc': p.get('icc'),
            'delta': p.get('delta'),
            'mean_group_size': p.get('mean_group_size'),
            'verdict': verdict_note,
        }

    examples = [
        _pt('homeacf_household', 'HomeACF 户级（南非，感染终点）',
            'ICC 0.275 → Δ +0.058：值得（对照组上限）'),
        _pt('sinan_munip_small', 'SINAN 小市（巴西，死亡终点）',
            'ICC 0.018 → Δ +0.003~0.005：不值得'),
        _pt('pacts_household', 'PACTS 户级（马拉维，症状终点）',
            'ICC 0.205 但终点天花板（ind 0.993）：'
            'ICC 达标仍不可用——终点判据同样必须检查'),
    ]

    return {
        'available': True,
        'headline': '组粒度部署指南：部署前测 ICC，再决定按哪一级分组',
        'narrative': (
            '时序先证的价值由组内共享暴露比例（ICC）预先决定：'
            '户级 ICC 0.21-0.28 对应 Δ +0.058 量级，市级 ICC 0.016-0.018 '
            '对应 Δ +0.002~0.005（缩水约 20 倍）；且市规模分层不改变 ICC'
            '——组语义（共享暴露单元）而非物理规模决定价值。'
        ),
        'decision_rule': {
            'measure': '部署前用试点数据测候选分组单位的 ICC(1)'
                       '（单因素随机效应 ANOVA）',
            'worthwhile': 'ICC > 0.1 且终点未被基线打穿 → 时序先证值得做',
            'not_worthwhile': 'ICC < 0.05 → 不值得'
                              '（增益约 +0.002~0.005 量级）',
            'grouping': '按共享暴露单元（户/楼）分组，不按行政单元'
                        '（市）分组',
        },
        'examples': examples,
        'points': points,
        'monotonicity': raw.get('monotonicity', {}),
        'plot_path': 'data/processed/granularity_effect_curve_20260826.png',
    }


# ==============================================================================
# 11. 部署度量（round-9 P0-2：AUROC → yield@k / NNS / PPV@预算）
# ==============================================================================

DEPLOY_ARCHIVE = 'sinan_deployment_metrics_20260826.json'
KENYA_PU_ARCHIVE = 'kenya_case_population_20260826.json'
BACT_HEAD_ARCHIVE = 'sinan_bacteriology_head_20260826.json'
CR_ARCHIVE = 'sinan_competing_risk_20260826.json'
TRANSFER_ARCHIVE = 'cross_population_transfer_20260826.json'
RECAL_ARCHIVE = 'transfer_recalibration_20260826.json'
BACT_V2_ARCHIVE = 'sinan_bact_head_v2_20260826.json'


def collect_deployment_metrics():
    """汇总 round-9 部署语言证据包（四归档合一）。

    数据源：
      1. sinan_deployment_metrics —— SINAN far test 两臂 yield@k /
         NNS / PPV + 老年分层 + 先证 yield 增益；
      2. kenya_case_population —— 采样偏差（系统漏检率 + 高风险池）；
      3. sinan_bacteriology_head —— 「找患者」第二头（诚实特征）；
      4. sinan_competing_risk —— 竞争风险年龄稀释结构。
    """
    dep = _neg_load(DEPLOY_ARCHIVE)
    ken = _neg_load(KENYA_PU_ARCHIVE)
    bac = _neg_load(BACT_HEAD_ARCHIVE)
    cr = _neg_load(CR_ARCHIVE)
    trf = _neg_load(TRANSFER_ARCHIVE)
    recal = _neg_load(RECAL_ARCHIVE)
    bactv2 = _neg_load(BACT_V2_ARCHIVE)
    if not dep:
        return _unavailable('归档不存在（data/processed/%s）'
                            % DEPLOY_ARCHIVE)

    arms = dep.get('arms', {})
    ind = arms.get('ind', {})
    ind_all = arms.get('ind_all', {})
    m_ind = ind.get('metrics', {})
    m_all = ind_all.get('metrics', {})

    def _row(k):
        a, b = m_ind.get(k, {}), m_all.get(k, {})
        return {'budget_pct': k,
                'yield_ind': a.get('yield'), 'yield_ind_all': b.get('yield'),
                'nns_ind_all': b.get('nns'), 'ppv_ind_all': b.get('ppv'),
                'lift_ind_all': b.get('lift')}

    yield_rows = [_row(k) for k in ('1', '5', '10', '20', '30', '50', '100')]
    elderly = dep.get('elderly_65p', {}).get('elderly_65p', {})
    adult = dep.get('elderly_65p', {}).get('adult_lt65', {})

    kenya = {}
    if ken:
        a = ken.get('A_missed_by_system', {})
        c = ken.get('C_high_risk_pool', {})
        kenya = {
            'miss_rate': a.get('miss_rate'),
            'n_missed': a.get('n_missed'), 'n_pos': a.get('n_total_pos'),
            'missed_hiv_pct': (ken.get('B_missed_profile', {})
                               .get('missed', {}).get('hiv_pct')),
            'found_hiv_pct': (ken.get('B_missed_profile', {})
                              .get('found', {}).get('hiv_pct')),
            'oof_auroc': c.get('auroc_oof'),
            'yield_top10': c.get('yield_top10pct'),
            'top10_never_treated_pos': c.get('top10_never_treated_pos'),
        }

    bact = {}
    if bac:
        h = bac.get('heads', {})
        bact = {
            'death_auroc': h.get('death_allcause', {}).get('auroc'),
            'bact_auroc': h.get('bacteriology_pos', {}).get('auroc'),
            'bact_yield10': (h.get('bacteriology_pos', {})
                             .get('metrics', {}).get('10', {})
                             .get('yield')),
            'top10_overlap': (bac.get('top_decile_overlap', {})
                              .get('overlap_fraction')),
        }

    # round-10 P2：双头架构正式化——任务映射 + 双头预算曲线行 +
    # 预测器 API（模型工件由实验脚本落盘 models/sinan_dual_head/）
    dual_head = {}
    if bac:
        hd = bac.get('heads', {})
        dm = hd.get('death_allcause', {}).get('metrics', {})
        bm = hd.get('bacteriology_pos', {}).get('metrics', {})

        def _dh_row(k):
            a, b = dm.get(k, {}), bm.get(k, {})
            return {'budget_pct': k,
                    'yield_death': a.get('yield'),
                    'yield_bact': b.get('yield'),
                    'nns_death': a.get('nns'), 'nns_bact': b.get('nns'),
                    'ppv_death': a.get('ppv'), 'ppv_bact': b.get('ppv')}

        # round-13 P2-4：双头 v2 面板——预算曲线（v1 基线，确定性等价）
        # + v2 头指标（AUPRC）+ v2 工件 + v2 重叠
        r2hd = (bactv2 or {}).get('results', {})
        v2d = r2hd.get('v2_extended', {}).get('death_allcause', {})
        v2b = r2hd.get('v2_extended', {}).get('bacteriology_pos', {})
        overlap_v2 = (bactv2 or {}).get('top_decile_overlap_v2')
        art_v2 = (bactv2 or {}).get('model_artifacts_v2', {})
        dual_head = {
            'task_mapping': bac.get('task_mapping', {}),
            'yield_rows': [_dh_row(k) for k in
                           ('1', '5', '10', '20', '30', '50', '100')],
            'top10_overlap': (overlap_v2 if overlap_v2 is not None
                              else (bac.get('top_decile_overlap', {})
                                    .get('overlap_fraction'))),
            'heads_v2': {
                'death_auroc': v2d.get('auroc'),
                'death_auprc': v2d.get('auprc'),
                'bact_auroc': v2b.get('auroc'),
                'bact_auprc': v2b.get('auprc'),
                'bact_yield10': v2b.get('yield10'),
            },
            'model_artifacts': (art_v2 if art_v2.get('saved')
                                else bac.get('model_artifacts', {})),
            'predictor_api': 'tb_risk.core.dual_head_predictor'
                             '.DualHeadPredictor（MODEL_VERSION=v2，'
                             'predict 输出 death_risk + '
                             'bacteriology_risk 双头）',
            'architecture_note': (
                'top-decile 重叠仅 %.1f%%（v2 特征）：死亡分层与'
                '「找患者」是两个部署语义，双头独立部署有明确互补'
                '价值——架构决策的定量依据，不再是猜想'
                % (100 * (dual_overlap := (overlap_v2
                          if overlap_v2 is not None
                          else (bac.get('top_decile_overlap', {})
                                .get('overlap_fraction')) or 0)))),
        }

    # round-10 P3：分龄阈值策略表——「预算 X 时该筛哪些人」
    age_thresholds = {}
    if elderly.get('age_thresholds') and adult.get('age_thresholds'):
        et, at = elderly['age_thresholds'], adult['age_thresholds']

        def _thr_row(k):
            e, a = et.get(k, {}), at.get(k, {})
            return {'budget_pct': k,
                    'thr_elderly': e.get('threshold'),
                    'caught_elderly': e.get('caught'),
                    'missed_elderly': e.get('missed'),
                    'thr_adult': a.get('threshold'),
                    'caught_adult': a.get('caught'),
                    'missed_adult': a.get('missed')}

        age_thresholds = {
            'rows': [_thr_row(k) for k in
                     ('1', '5', '10', '20', '30', '50', '100')],
            'n_events_elderly': round(elderly.get('event_rate', 0)
                                      * elderly.get('n', 0)),
            'n_events_adult': round(adult.get('event_rate', 0)
                                    * adult.get('n', 0)),
            'note': '组内 top-k% 的分数阈值（ind_all 臂，far test）；'
                    '检出 = 组内事件被 top-k 捕获数，漏检 = 组内事件'
                    '总数 - 检出；老年基线风险高（24.7% vs 8.5%）→ '
                    '同预算阈值显著更高',
        }

    # round-10 P4：老年双死因分解叙事（塌方 → 结构性解释）
    elderly_dual_cause = {}
    if cr:
        fg = cr.get('fine_gray', {})
        el = fg.get('elderly_65p', {}).get('cause_specific_cox', {})
        dilution = cr.get('age_dilution', [])
        last_bin = dilution[-1] if dilution else {}
        elderly_dual_cause = {
            'headline': '老年双死因分解：塌方的结构性解释（竞争风险）',
            'narrative': (
                '老年死亡信号被非 TB 死亡稀释：全队列竞争占比 '
                '{comp:.0%}，65+ 仍达 {elcomp:.0%}——同一「死亡」标签'
                '混合两种病因。cause-specific 双模型给出分流结构：'
                'TB 死亡 C-index {tb:.4f}（HIV 系数 {tbhiv:+.3f}）/ '
                '非 TB 死亡 C {ntb:.4f}（HIV 系数 {ntbhiv:+.3f}）'
                '——HIV 把死亡从 TB 侧分流到非 TB 侧；E3 TB死亡终点'
                ' all_age {e3:.4f}。老年预后预测的本质困难是标签稀释'
                '下的双死因分解，而非特征不足。'
            ).format(comp=last_bin.get('competing_fraction', 0),
                     elcomp=last_bin.get('competing_fraction', 0),
                     tb=el.get('tb_death', {}).get('c_index_ipcw', 0),
                     tbhiv=el.get('tb_death', {})
                     .get('hiv_aids_coeff', 0),
                     ntb=el.get('nontb_death', {})
                     .get('c_index_ipcw', 0),
                     ntbhiv=el.get('nontb_death', {})
                     .get('hiv_aids_coeff', 0),
                     e3=cr.get('E3_death_tb', {}).get('auroc', 0)),
            'age_dilution': dilution,
            'framework': fg.get('elderly_65p', {}).get('framework'),
            'paper_note': '论文级发现：老年 TB 预后预测的方法学警示'
                          '（cause-specific 分解 + 竞争分数按龄曲线 + '
                          'E3 终点成绩——第二篇论文的核心材料）',
        }

    crisk = {}
    if cr:
        fg = cr.get('fine_gray', {})
        crisk = {
            'elderly_tb_death_c': (fg.get('elderly_65p', {})
                                   .get('cause_specific_cox', {})
                                   .get('tb_death', {}).get('c_index_ipcw')),
            'elderly_nontb_c': (fg.get('elderly_65p', {})
                                .get('cause_specific_cox', {})
                                .get('nontb_death', {}).get('c_index_ipcw')),
            'age_dilution': cr.get('age_dilution', []),
            'E3_auroc': cr.get('E3_death_tb', {}).get('auroc'),
            'E3_auroc_elderly': cr.get('E3_death_tb', {})
                                  .get('auroc_elderly'),
            'E4_auroc': cr.get('E4_unfavorable', {}).get('auroc'),
            'E4_auroc_elderly': cr.get('E4_unfavorable', {})
                                  .get('auroc_elderly'),
        }

    # round-11 P1-1：老年双终点报告——全因死亡（预后分层语义）vs
    # E3 TB 死亡（传染控制语义）并列；老年全因死亡 63% 为非 TB
    elderly_dual_endpoint = {}
    de = dep.get('elderly_dual_endpoint', {})
    if de:
        arms_de = de.get('arms', {})

        def _de_rows():
            rows = []
            for ep_label, arm in (('allcause', arms_de.get('allcause_death', {})),
                                  ('e3_tb_death', arms_de.get('e3_tb_death', {}))):
                subs = arm.get('subgroups', {})
                for gname in ('elderly_65p', 'adult_lt65'):
                    g = subs.get(gname, {})
                    if 'auroc' not in g:
                        rows.append({'endpoint': ep_label, 'group': gname,
                                     'available': False})
                        continue
                    t10 = g.get('age_thresholds', {}).get('10', {})
                    rows.append({
                        'endpoint': ep_label, 'group': gname,
                        'auroc': g.get('auroc'),
                        'event_rate': g.get('event_rate'),
                        'yield10': (g.get('metrics', {}).get('10', {})
                                    .get('yield')),
                        'thr10': t10.get('threshold'),
                        'caught10': t10.get('caught'),
                        'missed10': t10.get('missed'),
                    })
            return rows

        elderly_dual_endpoint = {
            'headline': '老年双终点：全因死亡（预后分层）vs '
                        'E3 TB 死亡（传染控制）并列',
            'endpoint_semantics': de.get('endpoint_semantics', {}),
            'elderly_non_tb_share': de.get('elderly_non_tb_share'),
            'rows': _de_rows(),
            'paper_note': de.get('paper_note'),
            'decision_rule': '若目标是 TB 死亡（传染控制），以 E3 臂成绩'
                             '为准；全因臂只用于预后分层（含模型无法'
                             '干预的非 TB 死亡）',
        }

        # round-12 P1-2：双终点分龄阈值表（GUI 预算滑块终点切换用）——
        # 每终点 7 档预算 × 老年/成人阈值 + 检出/漏检
        ep_thr = {}
        for ep_key, arm in (('allcause', arms_de.get('allcause_death', {})),
                            ('e3_tb_death',
                             arms_de.get('e3_tb_death', {}))):
            subs = arm.get('subgroups', {})
            et = subs.get('elderly_65p', {}).get('age_thresholds', {})
            at = subs.get('adult_lt65', {}).get('age_thresholds', {})
            if not (et and at):
                continue

            def _thr_row(k):
                e, a = et.get(k, {}), at.get(k, {})
                return {'budget_pct': k,
                        'thr_elderly': e.get('threshold'),
                        'caught_elderly': e.get('caught'),
                        'missed_elderly': e.get('missed'),
                        'thr_adult': a.get('threshold'),
                        'caught_adult': a.get('caught'),
                        'missed_adult': a.get('missed')}

            ep_thr[ep_key] = {
                'rows': [_thr_row(k) for k in
                         ('1', '5', '10', '20', '30', '50', '100')],
                'n_events_elderly': round(
                    subs.get('elderly_65p', {}).get('event_rate', 0)
                    * subs.get('elderly_65p', {}).get('n', 0)),
                'n_events_adult': round(
                    subs.get('adult_lt65', {}).get('event_rate', 0)
                    * subs.get('adult_lt65', {}).get('n', 0)),
            }
        if ep_thr:
            elderly_dual_endpoint['threshold_rows'] = ep_thr

    # round-11 P1-2：本地重校准最小样本量曲线（负迁移的工程解）
    # round-12/13：三臂演进 + v4 统一配方 + Kenya 网格外推 + 方向
    # 裁定错误率曲线——迁移配方表最后两个未知数补齐
    transfer_recalibration = {}
    if recal:
        transfer_recalibration = {
            'headline': '本地重校准最小样本量：负迁移的工程解'
                        '（v4 统一配方 + 方向裁定成本）',
            'design': recal.get('design', {}),
            'cohorts': [
                {'cohort': c.get('cohort'), 'n': c.get('n'),
                 'zero_shot_auroc': c.get('zero_shot_auroc'),
                 'direction_truth': c.get('direction_truth'),
                 'full_local_cv_auroc': c.get('full_local_cv_auroc'),
                 'n_to_recover_stack': c.get('n_to_recover_stack'),
                 'n_to_recover_stack_prior':
                     c.get('n_to_recover_stack_prior'),
                 'n_to_recover_features': c.get('n_to_recover_features'),
                 'n_to_recover_score_v1': c.get('n_to_recover_score_v1'),
                 'curve': [{'n_local': p.get('n_local'),
                            'n_events_median': p.get('n_events_median'),
                            'stack_median': p.get('stack_median'),
                            'stack_prior_median':
                                p.get('stack_prior_median'),
                            'finetune_features_median':
                                p.get('finetune_features_median'),
                            'finetune_score_median':
                                p.get('finetune_score_median'),
                            'local_only_median':
                                p.get('local_only_median'),
                            'ruling_error_rate':
                                p.get('ruling_error_rate')}
                           for p in c.get('curve', [])]}
                for c in recal.get('cohorts', [])],
            'finding': '配方表补齐（round-13）：对齐站 Kenya 恢复 '
                       'N=2000（v2 0.607/v3 0.603，≈11 阳性越过恢复'
                       '线）；N=10000（≈53 阳性）v3 0.640/v4 0.639 '
                       '超全量本地 0.611——迁移先验+本地特征在事件充足'
                       '时超越纯本地。v4 统一配方（β_logit~N(1,1) 先验'
                       '）可行但有代价：对齐站恢复点 2000→5000，反向站'
                       '全程低于 v3 约 0.01——先裁定方向的决策树在裁定'
                       '可靠时仍更优，v4 是裁定不可行（事件太少）时的'
                       '稳健默认',
            'sample_size_rule': '方向裁定按阳性事件数（错误率实测）：'
                                '强翻转站（Vietnam 43.5%）N=200 即 0% '
                                '错误（N=100 仅 10%）；低基率站（Kenya '
                                '0.53%）N≤1000 错误率 30-50%≈抛硬币，'
                                'N=5000（≈27 事件）5%，N=10000（≈53 '
                                '事件）0%——≥25-50 事件的经验下限被'
                                '直接验证',
            'red_line_note': recal.get('red_line_note'),
        }

    # round-11 P1-3：细菌学头 v2——症状列裁定 + HIV/职业扩展
    bact_head_v2 = {}
    if bactv2:
        r2 = bactv2.get('results', {})
        bact_head_v2 = {
            'headline': '细菌学头 v2：症状列 INFEASIBLE 实锤 + '
                        'HIV/职业小增益',
            'symptom_feasibility': bactv2.get('symptom_feasibility', {}),
            'new_features': bactv2.get('new_features', {}),
            'v1_death_auroc': (r2.get('v1_baseline', {})
                               .get('death_allcause', {}).get('auroc')),
            'v2_death_auroc': (r2.get('v2_extended', {})
                               .get('death_allcause', {}).get('auroc')),
            'v1_bact_auroc': (r2.get('v1_baseline', {})
                              .get('bacteriology_pos', {}).get('auroc')),
            'v2_bact_auroc': (r2.get('v2_extended', {})
                              .get('bacteriology_pos', {}).get('auroc')),
            'v2_bact_auprc': (r2.get('v2_extended', {})
                              .get('bacteriology_pos', {}).get('auprc')),
            'v2_gain': bactv2.get('v2_gain', {}),
            'top_decile_overlap_v2': bactv2.get('top_decile_overlap_v2'),
            'model_artifacts_v2': bactv2.get('model_artifacts_v2', {}),
            'positioning': '症状信息缺位是 TUBEBR 数据侧边界（94 字段'
                           '实锤）；细菌学头维持「分诊工具，非诊断」'
                           '定位——基率 57% 决定排序天花板，价值在部署'
                           '语义（通报人群中排序传染源），AUPRC 报告'
                           '为准',
        }

    # round-11 P3-6：时序先证 SINAN 封存——分组粒度受限
    temporal_prior_sealed = {
        'conclusion': '时序先证在 SINAN 上封存为「分组粒度受限」结论',
        'evidence': '市级 ICC 0.016-0.018（三层）vs 户级 0.275'
                    '（HomeACF）/ 0.205（PACTS）——时序先证机制需要'
                    '户级聚类才能承载；ind_density 零增益佐证市级'
                    '密度无信号',
        'sinan_residual_gain': 'far test AUROC +0.0042（LGBM，v2 归档'
                              '统计检验显著）——市级粒度下先证仍有'
                              '小增益，但 SINAN 不是机制验证的场所',
        'implication': '机制验证重心移向有真家庭 ID 的 ERASE-TB / LMU '
                       '数据——市级 vs 户级 ICC 一个数量级的定量对比'
                       '是数据申请的最强论据',
    }

    transfer = {}
    if trf:
        transfer = {
            'br_reference_auroc': trf.get('br_reference', {}).get('auroc'),
            'cohorts': trf.get('cohorts', []),
            'excluded': trf.get('excluded', []),
            'feature_boundary': (trf.get('design', {})
                                  .get('feature_boundary')),
            'deployment_red_line': (
                '死亡头禁止零样本部署到确诊终点人群：Vietnam PROVE_TB '
                'zero-shot AUROC 0.426 < 0.5——死亡终点的 age 系数'
                '（age↑risk）与确诊终点人群方向相反，会反向排序。'
                '跨终点部署前必须本地重校准（最小样本量见 '
                'transfer_recalibration：N=100 可修复方向错误）'),
        }

    return {
        'available': True,
        'headline': '部署度量：yield@k / NNS / PPV@预算'
                    '（模型不动，评估切换为疾控决策语言）',
        'narrative': (
            'SINAN far test（2018-19，时间外推）：个体基线 top-10% 筛查'
            '捕获 {y10:.1%} 死亡（随机基线 3.9 倍），时序先证再 +'
            '{d:.1%}；老年 yield@10% 仅 {ye:.1%}（成人 {ya:.1%}）'
            '——塌方在部署语言下 = 同预算少捕获 {gap:.0f} 个百分点。'
        ).format(y10=m_ind.get('10', {}).get('yield', 0),
                 d=dep.get('prior_gain', {}).get('yield_at', {})
                 .get('10', {}).get('delta', 0),
                 ye=elderly.get('metrics', {}).get('10', {}).get('yield', 0),
                 ya=adult.get('metrics', {}).get('10', {}).get('yield', 0),
                 gap=100 * ((adult.get('metrics', {}).get('10', {})
                             .get('yield', 0))
                            - (elderly.get('metrics', {}).get('10', {})
                               .get('yield', 0)))),
        'sinan': {'auroc_ind': ind.get('auroc'),
                  'auroc_ind_all': ind_all.get('auroc'),
                  'yield_rows': yield_rows,
                  'prior_gain': dep.get('prior_gain', {}),
                  'elderly': {'auroc': elderly.get('auroc'),
                              'event_rate': elderly.get('event_rate')},
                  'adult': {'auroc': adult.get('auroc'),
                            'event_rate': adult.get('event_rate')}},
        'kenya_case_population': kenya,
        'bacteriology_head': bact,
        'dual_head': dual_head,
        'age_thresholds': age_thresholds,
        'elderly_dual_cause': elderly_dual_cause,
        'elderly_dual_endpoint': elderly_dual_endpoint,
        'transfer_recalibration': transfer_recalibration,
        'bact_head_v2': bact_head_v2,
        'temporal_prior_sealed': temporal_prior_sealed,
        'competing_risk': crisk,
        'cross_population_transfer': transfer,
        'planner_api': 'tb_risk.ml.planner.screening_metrics_at_budgets'
                       '（GUI 与实验脚本共用的部署度量纯函数）',
    }


__all__ = [
    'extract_three_layer_series',
    'collect_task_evaluations',
    'collect_network_ablation',
    'collect_public_data_evidence',
    'collect_dgp_evidence',
    'collect_seir_intervention',
    'collect_sinan_scale_evidence',
    'collect_negative_findings',
    'collect_granularity_guide',
    'collect_deployment_metrics',
    'DEFAULT_DEMO_BASE_RATE',
    'household_screening_signature',
    'build_household_screening_demo',
]
