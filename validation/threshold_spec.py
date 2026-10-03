#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型边界与临床校验模块（阈值-规范对照 / 约登指数重切 / 临床分级对照）

将 tb_risk 的风险切分阈值与权威临床依据显式关联起来，并允许通过真实
转归标签重新切分阈值、以及对照既有临床分级输出一致性/差异分析。

三大能力：
1. 阈值与规范对照表（Threshold-Spec Table）
   把"极高/高/中/低"风险的概率切分阈值与权威依据关联：
   - 涂阳/涂阴肺结核分级（涂阳传染性更高，WHO 接触者调查首要优先）
   - WHO 接触者筛查指南（smear-positive PTB / MDR-XDR / PLHIV / <5岁儿童）
   - 中国 WS 288-2017 肺结核诊断标准
2. 约登指数（Youden Index）重切阈值
   依据真实转归标签（病例/非病例），在使 J = 敏感度 + 特异度 - 1 最大处
   重新切分"疾病风险"阈值，得到数据驱动的最优切点。
3. 临床分级对照（Model Grade vs Clinical Standard）
   将模型风险等级与涂阳/涂阴驱动的临床标准等级逐例对照，输出一致性
   （一致率、Cohen's Kappa、混淆矩阵）与差异（高估/低估率）分析。

文献依据：
- WHO (2012). Recommendations for Investigating Contacts of Persons with
  Infectious Tuberculosis in Low- and Middle-Income Countries.
  - 接触者调查首要优先：痰涂片阳性肺结核、MDR/XDR-TB、PLHIV、<5 岁儿童。
- WHO Stop TB Strategy Handbook (2008)：涂阳肺结核均为索引病例。
- WHO (2022). Consolidated guidelines on TB, Module 2/5：密切接触者系统性筛查。
- Youden WJ (1950). Index for rating diagnostic tests. Cancer 3(1):32-35.
- Theron G et al. (2012) Clin Infect Dis 54(3):384-388：
  以约登指数确定预测痰涂片状态的 Ct 切点（≤20.2 特异性高，>31.8 排除价值好）。
- Nouira M et al. (2024) F1000Research 12:1297：
  TST 以约登指数最大点（≥11mm）作为判别切点。
- Cohen J (1960). A coefficient of agreement for nominal scales.
  Educ Psychol Meas 20(1):37-46.
"""

import logging
import math

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - numpy 不可用时降级
    np = None
    NUMPY_AVAILABLE = False

from ..constants import (
    LATENT_BASELINE, PROGRESSION_SCALE_FACTOR, MAX_PROGRESSION_RATE,
    MAX_DISEASE_PROBABILITY, AGE_PROGRESSION, IMMUNO_FACTORS_BASE,
)

LOGGER = logging.getLogger("tb_risk.validation.threshold_spec")

# ==============================================================================
# 风险等级定义（与 constants.DISEASE_PROB_* 对齐）
# ==============================================================================
RISK_GRADES = ['低风险', '中风险', '高风险', '极高风险']

# 默认概率切分阈值（与 constants.py 单一真值源对齐）
DEFAULT_VERY_HIGH = 15.0   # >= 15  → 极高风险
DEFAULT_HIGH = 8.0         # >= 8   → 高风险
DEFAULT_MEDIUM = 3.0       # >= 3   → 中风险；<3 → 低风险

# ==============================================================================
# 阈值与规范对照表
# ==============================================================================
THRESHOLD_SPEC_TABLE = [
    {
        'grade': '低风险',
        'code': 'low',
        'min_prob': 0.0,
        'range': '[0, 3.0)',
        'clinical_basis': '低危暴露 / 一般社会接触',
        'who_priority': '常规监测，无需优先排查',
        'rationale': '接触强度与传染源传染性均低，进展风险可忽略',
        'reference': 'WHO 2022 Consolidated guidelines Module 2',
    },
    {
        'grade': '中风险',
        'code': 'medium',
        'min_prob': DEFAULT_MEDIUM,
        'range': '[3.0, 8.0)',
        'clinical_basis': '一般家庭/密切接触者（无明确高传染因素）',
        'who_priority': 'WHO 建议对家庭及密切接触者进行系统性筛查',
        'rationale': '属常规接触者筛查范畴，存在中等感染概率',
        'reference': 'WHO 2022 Consolidated guidelines Module 2',
    },
    {
        'grade': '高风险',
        'code': 'high',
        'min_prob': DEFAULT_HIGH,
        'range': '[8.0, 15.0)',
        'clinical_basis': '涂阴但有空洞 / 治疗早期传染源 / MDR-XDR / PLHIV / <5岁儿童',
        'who_priority': ('WHO 接触者调查高优先：MDR/XDR-TB、PLHIV、'
                         '<5 岁儿童接触者'),
        'rationale': ('虽有空洞或免疫抑制等高传染/高进展因素，但涂片阴性，'
                      '传染性低于涂阳，仍需优先排查'),
        'reference': ('WHO 2012 Recommendations for Investigating Contacts; '
                      'WHO 2022 Consolidated guidelines Module 5'),
    },
    {
        'grade': '极高风险',
        'code': 'very_high',
        'min_prob': DEFAULT_VERY_HIGH,
        'range': '[15.0, ∞)',
        'clinical_basis': '涂阳肺结核接触者 / 空洞 / 未治疗传染源',
        'who_priority': ('WHO 接触者调查首要优先：痰涂片阳性肺结核 '
                         '(smear-positive PTB)'),
        'rationale': ('涂阳患者传染性显著高于涂阴（显微镜涂阳为传染性金标准），'
                      '其接触者应优先排查'),
        'reference': ('WHO 2012 Recommendations for Investigating Contacts; '
                      'WHO Stop TB Strategy Handbook 2008; WS 288-2017'),
    },
]

# 涂阳/涂阴临床分级到模型分级的映射依据
SMEAR_CLINICAL_GRADE = {
    'sputum_smear_positive': '极高风险',   # 涂阳：传染性最高，首要优先
    'sputum_smear_negative': '中风险',     # 涂阴：传染性低，常规筛查
}


def grade_risk(probability, very_high=DEFAULT_VERY_HIGH,
               high=DEFAULT_HIGH, medium=DEFAULT_MEDIUM):
    """将疾病概率映射为风险等级（极高/高/中/低）。

    参数：
        probability (float): 疾病概率(%)，来自模型或传统评估。
        very_high/high/medium (float): 切分阈值（默认取常量定义值）。

    返回：
        str: 风险等级
    """
    if probability >= very_high:
        return '极高风险'
    if probability >= high:
        return '高风险'
    if probability >= medium:
        return '中风险'
    return '低风险'


def get_threshold_spec(grade):
    """按等级名查找阈值-规范对照表条目。"""
    for row in THRESHOLD_SPEC_TABLE:
        if row['grade'] == grade:
            return dict(row)
    return None


def build_threshold_spec_report(very_high=DEFAULT_VERY_HIGH,
                                high=DEFAULT_HIGH, medium=DEFAULT_MEDIUM):
    """生成完整"阈值与规范对照表"报告。

    返回：
        dict: {'grades': [...], 'thresholds': {...}, 'n_grades': int}
    """
    thresholds = {
        'very_high': float(very_high),
        'high': float(high),
        'medium': float(medium),
    }
    rows = []
    for row in THRESHOLD_SPEC_TABLE:
        r = dict(row)
        r['min_prob'] = thresholds.get(r['code'], 0.0)
        r['range'] = _format_range(r['code'], thresholds)
        rows.append(r)
    return {
        'grades': rows,
        'thresholds': thresholds,
        'source': ('WHO 2012 contact investigation; WHO 2022 consolidated '
                   'guidelines; WS 288-2017; 涂阳/涂阴分级'),
        'n_grades': len(rows),
    }


def _format_range(code, thresholds):
    t = thresholds
    if code == 'very_high':
        return f'[{t["very_high"]:.1f}, ∞)'
    if code == 'high':
        return f'[{t["high"]:.1f}, {t["very_high"]:.1f})'
    if code == 'medium':
        return f'[{t["medium"]:.1f}, {t["high"]:.1f})'
    return f'[0, {t["medium"]:.1f})'


# ==============================================================================
# 约登指数（Youden Index）重切阈值
# ==============================================================================
def youden_optimal_threshold(probabilities, outcomes):
    """在使约登指数 J = 敏感度 + 特异度 - 1 最大处求最优二元切点。

    基于真实转归标签（outcomes: 0=非病例, 1=病例）在候选切点中筛选，
    使诊断效能（敏感度+特异度）最大化。

    文献：
    - Youden WJ (1950). Index for rating diagnostic tests. Cancer 3(1):32-35.
    - Theron G et al. (2012) Clin Infect Dis: 以约登指数确定预测涂阳的切点。
    - Nouira M et al. (2024) F1000Research: 以约登指数最大点作判别切点。

    参数：
        probabilities (list[float]): 每个样本的疾病概率(%)。
        outcomes (list[int]): 真实转归标签（0/1）。

    返回：
        dict: {threshold, youden_index, sensitivity, specificity,
               n_pos, n_neg, n_total}
        数据不足或标签无变异时返回 None。
    """
    probs = list(probabilities)
    labs = [int(o) for o in outcomes]
    if len(probs) != len(labs) or len(probs) < 2:
        return None
    pos = sum(1 for o in labs if o == 1)
    neg = sum(1 for o in labs if o == 0)
    if pos == 0 or neg == 0:
        # 无正/负样本，无法计算敏感度/特异度
        return None

    # 候选切点：去重后的概率值（含最小-ε保证全覆盖）
    candidates = sorted(set(probs))
    if not candidates:
        return None

    best = None
    for t in candidates:
        tp = sum(1 for p, o in zip(probs, labs) if p >= t and o == 1)
        fn = pos - tp
        tn = sum(1 for p, o in zip(probs, labs) if p < t and o == 0)
        fp = neg - tn
        sens = tp / pos if pos else 0.0
        spec = tn / neg if neg else 0.0
        j = sens + spec - 1.0
        if best is None or j > best['youden_index'] or (
                j == best['youden_index'] and t < best['threshold']):
            best = {
                'threshold': float(t),
                'youden_index': float(j),
                'sensitivity': float(sens),
                'specificity': float(spec),
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            }

    if best is None:
        return None
    best['n_pos'] = pos
    best['n_neg'] = neg
    best['n_total'] = len(probs)
    return best


def refit_risk_thresholds(probabilities, outcomes,
                          very_high=DEFAULT_VERY_HIGH,
                          high=DEFAULT_HIGH, medium=DEFAULT_MEDIUM):
    """依据真实转归标签，用约登指数重新切分风险阈值。

    说明：
    - 以约登指数最大点作为"疾病/高危"二元切点。
    - 保留原阈值作为参照，输出对比与建议（是否应采用数据驱动切点）。
    - 极高/高/中三级在二元切点基础上按比例插值给出建议，供临床权衡。

    参数：
        probabilities (list[float]): 模型疾病概率(%)。
        outcomes (list[int]): 真实转归（0/1）。
        very_high/high/medium (float): 原始阈值。

    返回：
        dict: 重切阈值报告。
    """
    opt = youden_optimal_threshold(probabilities, outcomes)
    report = {
        'method': 'Youden Index (J = sensitivity + specificity - 1)',
        'original_thresholds': {
            'very_high': float(very_high),
            'high': float(high),
            'medium': float(medium),
        },
        'youden_optimal': opt,
        'recommended_thresholds': None,
        'note': '数据不足或标签无变异，无法重切' if opt is None else None,
    }
    if opt is None:
        return report

    t_bin = opt['threshold']
    # 二元切点作为"中风险"起点（>= t_bin 视为需要关注/高危）
    # 在 [t_bin, very_high] 区间按 2/3、1/3 插值给出高/极高建议（临床权衡用）
    span = max(float(very_high) - t_bin, 1e-9)
    rec_high = t_bin + span * 2.0 / 3.0
    rec_very_high = t_bin + span * 1.0 / 3.0
    report['recommended_thresholds'] = {
        'very_high': round(rec_very_high, 2),
        'high': round(rec_high, 2),
        'medium': round(t_bin, 2),
    }
    report['note'] = (
        'recommended_thresholds 以约登指数最优二元切点为中风险下限，'
        '高/极高在 [中风险下限, 原极高阈值] 内按 1/3 与 2/3 插值给出，'
        '供临床按资源与漏诊代价权衡调整。'
    )
    return report


# ==============================================================================
# 期望校准误差（ECE, Expected Calibration Error）对比
# ==============================================================================
def expected_calibration_error(probabilities, outcomes, n_bins=10):
    """计算期望校准误差（ECE）。

    将预测概率按 [0, 1] 等分为 n_bins 个箱，逐箱比较预测均值与真实
    阳性率（观测频率），以样本占比加权平均绝对偏差：
        ECE = Σ_b (n_b/N) · |mean(pred_b) - freq(pos_b)|

    用于定量证明"实验室整合版"较"纯症状版"校准更优：
    越低越好，ECE=0 表示完美校准。

    文献：
    - Naeini MP et al. (2015). Obtaining Well Calibrated Probabilities
      Using Bayesian Binning into Quantiles. ICML.
    - Guo C et al. (2017). On Calibration of Modern Neural Networks. ICML.
    - 本模块补充：约登指数重切 + 临床分级对照之外，用 ECE 做校准定量对比。

    参数：
        probabilities (list[float]): 预测概率（0-1，或百分比将自动归一化）。
        outcomes (list[int]): 真实转归标签（0/1）。
        n_bins (int): 校准箱数（默认 10）。

    返回：
        dict: {'ece': float, 'n_bins': int, 'bin_details': [...], 'n': int}
        numpy 不可用或数据不足时返回带 'error' 的 dict。
    """
    probs = [float(p) for p in probabilities]
    labs = [int(o) for o in outcomes]
    if len(probs) != len(labs) or len(probs) < 2:
        return {'n': len(probs), 'error': '样本不足'}

    # 自动归一化：若概率以百分比表示（>1），归一到 [0,1]
    max_p = max(probs)
    if max_p > 1.0:
        probs = [p / 100.0 if p > 1.0 else p for p in probs]
    probs = [min(max(p, 0.0), 1.0) for p in probs]

    bin_details = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        bin_probs = [p for p in probs if lo <= p < hi]
        if b == n_bins - 1:
            bin_probs = [p for p in probs if lo <= p <= hi]
        if not bin_probs:
            continue
        n_b = len(bin_probs)
        pred_mean = sum(bin_probs) / n_b
        # 观测频率：按箱内样本对应的真实标签统计
        idx = [i for i, p in enumerate(probs)
               if (lo <= p < hi) or (b == n_bins - 1 and p == hi)]
        obs_freq = sum(labs[i] for i in idx) / n_b
        bin_details.append({
            'bin': b, 'range': f'[{lo:.2f}, {hi:.2f})',
            'n': n_b, 'predicted_mean': round(pred_mean, 4),
            'observed_freq': round(obs_freq, 4),
            'abs_error': round(abs(pred_mean - obs_freq), 4),
        })

    n_total = len(probs)
    ece = sum(
        d['n'] / n_total * d['abs_error'] for d in bin_details
    )
    return {
        'ece': round(ece, 4),
        'n_bins': n_bins,
        'bin_details': bin_details,
        'n': n_total,
    }


def compare_calibration(pure_symptom_probs, lab_integrated_probs, outcomes,
                        n_bins=10):
    """对比"纯症状版"与"实验室整合版"的校准质量（ECE）。

    设计为方案有效的定量证据：
    - 两版本在同一批真实转归标签上分别计算 ECE。
    - 若实验室整合版 ECE 更低，则证明引入实验室证据后概率校准更优。
    - 同时报告 NLL 近似（对数损失）与 Brier 分数作为辅助。

    参数：
        pure_symptom_probs (list[float]): 纯症状推断版预测概率。
        lab_integrated_probs (list[float]): 实验室证据整合版预测概率。
        outcomes (list[int]): 真实转归标签（0/1）。
        n_bins (int): 校准箱数。

    返回：
        dict: {'pure_symptom': {...}, 'lab_integrated': {...},
               'ece_reduction': float, 'improved': bool, 'conclusion': str}
    """
    a = expected_calibration_error(pure_symptom_probs, outcomes, n_bins)
    b = expected_calibration_error(lab_integrated_probs, outcomes, n_bins)

    report = {
        'pure_symptom': a,
        'lab_integrated': b,
        'ece_reduction': None,
        'improved': None,
        'conclusion': '数据不足，无法对比',
    }
    if 'error' in a or 'error' in b:
        return report

    a_ece, b_ece = a['ece'], b['ece']
    reduction = a_ece - b_ece
    improved = b_ece < a_ece
    report['ece_reduction'] = round(reduction, 4)
    report['improved'] = bool(improved)

    # 辅助指标：Brier 分数与对数损失（numpy 可用时计算）
    try:
        import math
        def _log_loss(probs, labs):
            total = 0.0
            eps = 1e-12
            for p, o in zip(probs, labs):
                p = min(max(float(p), eps), 1.0 - eps)
                total += -(o * math.log(p) + (1 - o) * math.log(1 - p))
            return total / max(len(probs), 1)

        def _brier(probs, labs):
            return sum((float(p) - int(o)) ** 2
                       for p, o in zip(probs, labs)) / max(len(probs), 1)

        def _norm(probs):
            max_p = max(probs) if probs else 0.0
            return [min(max(float(p) / 100.0 if p > 1.0 else p, 0.0), 1.0)
                    for p in probs]

        report['brier_score'] = {
            'pure_symptom': round(_brier(_norm(pure_symptom_probs), outcomes), 4),
            'lab_integrated': round(_brier(_norm(lab_integrated_probs), outcomes), 4),
        }
        report['log_loss'] = {
            'pure_symptom': round(_log_loss(_norm(pure_symptom_probs), outcomes), 4),
            'lab_integrated': round(_log_loss(_norm(lab_integrated_probs), outcomes), 4),
        }
    except Exception:
        pass

    report['conclusion'] = (
        f'实验室整合版 ECE={b_ece:.4f} < 纯症状版 ECE={a_ece:.4f}，'
        f'校准误差降低 {reduction:.4f}，实验室证据整合提升了概率校准质量'
        if improved else
        f'实验室整合版 ECE={b_ece:.4f} ≥ 纯症状版 ECE={a_ece:.4f}，'
        f'校准未见提升（差值 {reduction:.4f}），需核查证据映射是否过强'
    )
    return report


# ==============================================================================
# 近期进展分层：AUC 区分度验证（个体化双输出 vs 固定基线）
# ==============================================================================
# 设计动机：证明"双输出建模"（IGRA 感染门槛 + 时间衰减 + 个体修正）较旧的
# "固定基线 × 静态乘数"结构对真实转归有更好的分层能力。以疾病概率（%）为判分，
# 用 ROC AUC（曼-惠特尼 U 等价形式）衡量对真实转归（0/1）的判别力，并按
# igra_result 分层展示"近期进展"在不同感染状态下的区分度。
#
# 文献：
# - Hanley JA, McNeil BJ (1982). The meaning and use of the area under a
#   receiver operating characteristic (ROC) curve. Radiology 143(1):29-36.
# - Vynnycky E, Fine PEM (1997). The natural history of tuberculosis. Epidemiol
#   Infect 119(2):183-201. —— 终生约 10%、前 2 年约一半，支撑基线标定。
# - Andrews JR et al. (2012) Clin Infect Dis 54(6):784-791. —— 再感染进展差异。

IGRA_STRATA_KEYS = ['positive', 'negative', 'indeterminate', 'not_done', 'unknown']


def _normalize_igra_result(value):
    """归一化 igra_result 到分层键（positive/negative/indeterminate/not_done/unknown）。"""
    if value is None or value == '':
        return 'unknown'
    s = str(value).strip().lower()
    if s in ('positive', 'pos', '阳性', '阳', '是'):
        return 'positive'
    if s in ('negative', 'neg', '阴性', '阴', '否'):
        return 'negative'
    if s in ('indeterminate', 'indet', '不确定', '可疑', 'borderline'):
        return 'indeterminate'
    if s in ('not_done', '未检测', '无', 'none', 'nan', 'n/a'):
        return 'not_done'
    return 'unknown'


def compute_auc(probabilities, outcomes):
    """计算 ROC AUC（曼-惠特尼 U 等价形式），纯 Python 无强依赖。

    AUC = U / (n_pos × n_neg)，其中 U 为所有"病例-非病例"对中病例得分
    高于非病例得分的比例（并列计 0.5）。AUC=0.5 表示无区分度（随机），
    AUC=1.0 表示完全区分。

    参数：
        probabilities (list[float]): 判分（疾病概率%，或 0-1 概率）。
        outcomes (list[int]): 真实转归标签（0=非病例, 1=病例）。

    返回：
        float | None — 0-1 的 AUC；样本不足或无标签变异时返回 None。
    """
    probs = [float(p) for p in probabilities]
    labs = [int(o) for o in outcomes]
    if len(probs) != len(labs) or len(probs) < 2:
        return None
    pos = sum(1 for o in labs if o == 1)
    neg = len(labs) - pos
    if pos == 0 or neg == 0:
        return None
    u = 0.0
    for i, p in enumerate(probs):
        if labs[i] != 1:
            continue
        for j, q in enumerate(probs):
            if labs[j] != 0:
                continue
            if p > q:
                u += 1.0
            elif p == q:
                u += 0.5
    return u / (pos * neg)


def auc_standard_error(auc, n_pos, n_neg):
    """AUC 标准误（Hanley & McNeil 1982 近似），纯 Python。

    SE = sqrt( [AUC(1-AUC) + (n_pos-1)(Q1-AUC²) + (n_neg-1)(Q2-AUC²)]
               / (n_pos × n_neg) )
    其中 Q1 = AUC/(2-AUC)，Q2 = 2AUC²/(1+AUC)。

    参数：
        auc (float): 0-1 的 AUC。
        n_pos / n_neg (int): 病例 / 非病例样本数。

    返回：
        float | None — 标准误；n_pos 或 n_neg 为 0 时返回 None。
    """
    if n_pos == 0 or n_neg == 0:
        return None
    a = float(auc)
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1.0 - a)
           + (n_pos - 1.0) * (q1 - a * a)
           + (n_neg - 1.0) * (q2 - a * a)) / (n_pos * n_neg)
    return math.sqrt(max(var, 1e-12))


def igra_stratified_discrimination(records, individualized_probs, outcomes,
                                   baseline_probs=None, outcome_key='is_confirmed'):
    """按 IGRA 状态分层的近期进展区分度验证（AUC）。

    证明"个体化（双输出）版本"较"固定基线版本"对真实转归的分层能力：
    - 以疾病概率（%）为判分，AUC 衡量对真实转归（outcome_key）的判别力。
    - 按 igra_result 分层（positive/negative/indeterminate/not_done/unknown）。
    - 每层 + 总体输出 AUC、样本量、病例数；numpy 可用时附 AUC 95% CI（±1.96 SE）。

    参数：
        records (list[dict]): 含 igra_result / outcome_key 字段的筛查记录。
        individualized_probs (list[float]): 个体化（双输出）版疾病概率(%)。
        outcomes (list[int]): 真实转归标签（0/1）。
        baseline_probs (list[float] | None): 固定基线版疾病概率(%)，用于对比。
        outcome_key (str): 记录中真实转归字段名（默认 is_confirmed）。

    返回：
        dict: 分层区分度报告。
    """
    n = len(records)
    if n != len(individualized_probs):
        raise ValueError('records 与 individualized_probs 长度不一致')
    outs = [int(o) for o in outcomes]
    if n != len(outs):
        raise ValueError('records 与 outcomes 长度不一致')
    if baseline_probs is not None and len(baseline_probs) != n:
        raise ValueError('baseline_probs 长度不一致')

    # 按 igra_result 分层
    strata_idx = {}
    for i, rec in enumerate(records):
        key = _normalize_igra_result(rec.get('igra_result'))
        strata_idx.setdefault(key, []).append(i)

    overall_ind = compute_auc(individualized_probs, outs)
    overall_base = (compute_auc(list(baseline_probs), outs)
                    if baseline_probs is not None else None)
    n_pos = sum(1 for o in outs if o == 1)
    n_neg = n - n_pos

    def _pack(auc, pos, neg):
        entry = {'auc': auc}
        if auc is not None and pos > 0 and neg > 0:
            se = auc_standard_error(auc, pos, neg)
            entry['se'] = round(se, 4)
            entry['ci95'] = (
                round(max(auc - 1.96 * se, 0.0), 4),
                round(min(auc + 1.96 * se, 1.0), 4),
            )
        else:
            entry['se'] = None
            entry['ci95'] = None
        return entry

    per_stratum = {}
    for key in IGRA_STRATA_KEYS:
        idxs = strata_idx.get(key, [])
        if not idxs:
            per_stratum[key] = {
                'n': 0, 'n_pos': 0, 'n_neg': 0,
                'auc_individualized': None, 'auc_fixed_baseline': None,
            }
            continue
        sub_probs = [individualized_probs[i] for i in idxs]
        sub_outs = [outs[i] for i in idxs]
        sub_pos = sum(sub_outs)
        sub_neg = len(sub_outs) - sub_pos
        ind_auc = compute_auc(sub_probs, sub_outs)
        base_auc = (compute_auc([baseline_probs[i] for i in idxs], sub_outs)
                    if baseline_probs is not None else None)
        per_stratum[key] = {
            'n': len(idxs), 'n_pos': sub_pos, 'n_neg': sub_neg,
            'auc_individualized': _pack(ind_auc, sub_pos, sub_neg),
            'auc_fixed_baseline': _pack(base_auc, sub_pos, sub_neg),
        }

    report = {
        'method': 'ROC AUC（曼-惠特尼 U 等价），按 igra_result 分层',
        'n_records': n,
        'n_pos': n_pos,
        'n_neg': n_neg,
        'overall_auc_individualized': _pack(overall_ind, n_pos, n_neg),
        'overall_auc_fixed_baseline': (_pack(overall_base, n_pos, n_neg)
                                       if baseline_probs is not None else None),
        'strata': per_stratum,
        'conclusion': _conclude_auc(overall_ind, overall_base, per_stratum),
    }
    return report


def _conclude_auc(overall_ind, overall_base, strata):
    """汇总分层 AUC 结论文本。"""
    if overall_ind is None:
        return '数据不足或标签无变异，无法计算 AUC'
    parts = ['个体化（双输出）版总体 AUC=%.3f' % overall_ind]
    if overall_base is not None:
        delta = overall_ind - overall_base
        verdict = '优于' if delta > 0 else ('劣于' if delta < 0 else '持平')
        parts.append('固定基线版 AUC=%.3f，个体化版%s（Δ=%+.3f）'
                     % (overall_base, verdict, delta))
    aucs = [v['auc_individualized']['auc'] for v in strata.values()
            if isinstance(v['auc_individualized'], dict)
            and v['auc_individualized'].get('auc') is not None]
    if aucs:
        worst = min(aucs)
        if worst >= 0.5:
            parts.append('各 IGRA 分层最小 AUC=%.3f，最不利分层仍具区分度' % worst)
        else:
            parts.append('存在 IGRA 分层 AUC<0.5（最不利分层 %.3f）' % worst)
    return '；'.join(parts)


def _fixed_baseline_disease_prob(record, infection_probability):
    """复算旧"固定基线 × 静态乘数"结构的疾病概率（对照基准）。

    镜像 ScoringEngine 升级前的公式：
        adjusted_progress = min(LATENT_BASELINE × (1 + (combined_risk-1) × SF),
                                MAX_PROGRESSION_RATE)
        disease_prob      = min(infection_probability × adjusted_progress,
                                MAX_DISEASE_PROBABILITY)
    其中 combined_risk = prog_mult（免疫）× age_factor（年龄）。

    参数：
        record (dict): 含 past_illness_type / age 等字段的接触者记录。
        infection_probability (float): 感染概率（%），来自引擎。

    返回：
        float: 固定基线版疾病概率（%）。
    """
    prog_mult = 1.0
    illness = str(record.get('past_illness_type', 'none')).strip().lower()
    if illness == 'hiv':
        prog_mult = IMMUNO_FACTORS_BASE.get('hiv', 8.0)
    elif illness == 'immunosuppressants':
        prog_mult = IMMUNO_FACTORS_BASE.get('immunosuppressants', 4.0)
    elif illness == 'diabetes':
        prog_mult = IMMUNO_FACTORS_BASE.get('diabetes', 2.5)
    elif illness == 'other':
        prog_mult = IMMUNO_FACTORS_BASE.get('other', 1.8)

    try:
        age = float(record.get('age', 30))
    except (TypeError, ValueError):
        age = 30.0
    age_factor = 1.0
    if age < 5:
        age_factor = AGE_PROGRESSION['child_under_5']
    elif 5 <= age < 15:
        age_factor = AGE_PROGRESSION['child_5_14']
    elif 15 <= age <= 35:
        age_factor = AGE_PROGRESSION['young']
    elif 35 < age <= 65:
        age_factor = AGE_PROGRESSION['adult']
    else:
        age_factor = AGE_PROGRESSION['elderly']

    combined = prog_mult * age_factor
    adjusted = min(
        LATENT_BASELINE * (1.0 + (combined - 1.0) * PROGRESSION_SCALE_FACTOR),
        MAX_PROGRESSION_RATE)
    return min(infection_probability * adjusted, MAX_DISEASE_PROBABILITY)


def compare_individualized_vs_fixed_baseline(records, engine=None,
                                             outcome_key='is_confirmed'):
    """端到端：同一批记录对比"双输出（个体化）"vs"固定基线"的分层 AUC。

    用 ScoringEngine（非本土化模式）对每条记录计算：
    - individualized_probs = 双输出 disease_probability（IGRA 门槛 + 时间衰减）
    - baseline_probs = 复算固定基线 disease_probability（对照基准）
    随后调用 igra_stratified_discrimination 输出分层区分度报告。

    参数：
        records (list[dict]): 含 igra_result / past_illness_type / age /
                              is_confirmed 等字段的筛查记录。
        engine (ScoringEngine | None): 默认构造 ScoringEngine(use_localization=False)。
        outcome_key (str): 记录中真实转归字段名。

    返回：
        dict: igra_stratified_discrimination 报告。
    """
    if engine is None:
        from ..scoring.engine import ScoringEngine
        engine = ScoringEngine(use_localization=False)
    indiv = []
    base = []
    outs = []
    for rec in records:
        res = engine.compute_risk_score(rec)
        indiv.append(res['disease_probability'])
        base.append(_fixed_baseline_disease_prob(rec, res['infection_probability']))
        outs.append(int(rec.get(outcome_key, 0) or 0))
    return igra_stratified_discrimination(
        records, indiv, outs, baseline_probs=base, outcome_key=outcome_key)



def clinical_grade_from_record(record):
    """根据记录中的临床字段推导"临床标准等级"（涂阳/涂阴驱动）。

    规则（基于 WHO 接触者筛查优先序）：
    - 涂阳（sputum_smear in {2,'阳性','涂阳'}）        → 极高风险
    - 涂阴但有空洞 / MDR / HIV / <5 岁儿童              → 高风险
    - 涂阴一般家庭/密切接触者                           → 中风险
    - 其余 / 信息缺失                                    → 低风险

    参数：
        record (dict): 含 sputum_smear / has_cavity / illness_type / age 等字段。

    返回：
        str: 临床标准等级
    """
    smear = record.get('sputum_smear')
    positive = False
    if isinstance(smear, (int, float)):
        positive = smear >= 2
    elif isinstance(smear, str):
        positive = smear in ('阳性', '涂阳', '2')
    if positive:
        return '极高风险'

    has_cavity = record.get('has_cavity')
    if has_cavity in (2, '有空洞', True):
        return '高风险'

    illness = record.get('illness_type')
    if illness == 'HIV' or (isinstance(illness, str) and 'hiv' in illness.lower()):
        return '高风险'

    age = record.get('age')
    if isinstance(age, (int, float)) and 0 <= age < 5:
        return '高风险'

    if smear is not None:
        return '中风险'
    return '低风险'


class ClinicalGradeComparator:
    """模型风险等级 vs 临床标准等级一致性/差异分析器。

    对逐例的 [模型等级, 临床标准等级] 进行对照，输出：
    - 总体一致率 (accuracy/agreement)
    - Cohen's Kappa（随机校正后一致性）
    - 混淆矩阵（模型等级 × 临床等级）
    - 高估 / 低估率（模型相对临床）
    - 各等级分布
    - 结论与建议

    文献：
    - Cohen J (1960). A coefficient of agreement for nominal scales.
    """

    def __init__(self, grades=RISK_GRADES):
        self.grades = list(grades)
        self._order = {g: i for i, g in enumerate(self.grades)}

    # -- 等级有序比较辅助 --
    def _index(self, grade):
        return self._order.get(grade, len(self.grades))

    def compare(self, model_grades, clinical_grades):
        """运行模型等级 vs 临床标准等级对照。

        参数：
            model_grades (list[str]): 模型分配的风险等级。
            clinical_grades (list[str]): 临床标准等级（可来自
                clinical_grade_from_record）。

        返回：
            dict: 一致性/差异分析报告。
        """
        model_grades = list(model_grades)
        clinical_grades = list(clinical_grades)
        if len(model_grades) != len(clinical_grades):
            raise ValueError('model_grades 与 clinical_grades 长度不一致')

        n = len(model_grades)
        if n == 0:
            return {'n': 0, 'error': '空样本'}

        # 混淆矩阵
        conf = {g: {c: 0 for c in self.grades} for g in self.grades}
        agree = 0
        over = 0
        under = 0
        over_pairs = []
        under_pairs = []
        for mg, cg in zip(model_grades, clinical_grades):
            conf.setdefault(mg, {c: 0 for c in self.grades})
            conf[mg][cg] = conf[mg].get(cg, 0) + 1
            if mg == cg:
                agree += 1
            else:
                mi = self._index(mg)
                ci = self._index(cg)
                if mi > ci:
                    over += 1
                    over_pairs.append((mg, cg))
                else:
                    under += 1
                    under_pairs.append((mg, cg))

        accuracy = agree / n if n else 0.0
        kappa = self._cohen_kappa(model_grades, clinical_grades)

        # 各等级分布
        model_dist = {g: model_grades.count(g) for g in self.grades}
        clinical_dist = {g: clinical_grades.count(g) for g in self.grades}

        return {
            'n': n,
            'accuracy': float(accuracy),
            'agreement_count': agree,
            'cohens_kappa': float(kappa) if kappa is not None else None,
            'confusion_matrix': conf,
            'overestimation_rate': float(over / n) if n else 0.0,
            'underestimation_rate': float(under / n) if n else 0.0,
            'overestimation_count': over,
            'underestimation_count': under,
            'overestimation_examples': over_pairs[:10],
            'underestimation_examples': under_pairs[:10],
            'model_distribution': model_dist,
            'clinical_distribution': clinical_dist,
            'conclusion': self._conclude(accuracy, kappa, over, under, n),
        }

    def _cohen_kappa(self, model_grades, clinical_grades):
        """计算 Cohen's Kappa（观测一致率 vs 机会一致率）。"""
        n = len(model_grades)
        if n == 0:
            return None
        p_o = sum(1 for a, b in zip(model_grades, clinical_grades)
                  if a == b) / n
        p_e = 0.0
        for g in self.grades:
            p_model = model_grades.count(g) / n
            p_clin = clinical_grades.count(g) / n
            p_e += p_model * p_clin
        if p_e == 1.0:
            return 1.0 if p_o == 1.0 else 0.0
        return (p_o - p_e) / (1.0 - p_e)

    @staticmethod
    def _conclude(accuracy, kappa, over, under, n):
        if n == 0:
            return '样本为空'
        kappa_txt = ('kappa=%.2f' % kappa) if kappa is not None else 'kappa=N/A'
        parts = [f'总体一致率 {accuracy*100:.1f}%，{kappa_txt}']
        if over > under:
            parts.append(f'模型较临床标准整体高估（高估 {over} 例 > 低估 {under} 例）')
        elif under > over:
            parts.append(f'模型较临床标准整体低估（低估 {under} 例 > 高估 {over} 例）')
        else:
            parts.append('高估/低估例数持平')
        return '；'.join(parts)


def compare_model_vs_clinical(records, probabilities,
                              grade_fn=None, outcome_key='is_confirmed'):
    """端到端：给定记录与模型概率，执行临床分级对照。

    参数：
        records (list[dict]): 筛查记录（含临床字段与真实转归）。
        probabilities (list[float]): 每例模型疾病概率。
        grade_fn (callable|None): 模型分级函数，默认 grade_risk。
        outcome_key (str): 记录中真实转归字段名。

    返回：
        dict: 综合报告（重切阈值 + 临床对照）。
    """
    if grade_fn is None:
        grade_fn = grade_risk
    model_grades = [grade_fn(p) for p in probabilities]
    clinical_grades = [clinical_grade_from_record(r) for r in records]

    comparator = ClinicalGradeComparator()
    comp = comparator.compare(model_grades, clinical_grades)

    outcomes = [int(r.get(outcome_key, 0) or 0) for r in records]
    refit = refit_risk_thresholds(probabilities, outcomes)

    return {
        'model_vs_clinical': comp,
        'threshold_refit': refit,
        'n_records': len(records),
    }
