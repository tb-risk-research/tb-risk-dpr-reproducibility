#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验室证据整合模块（三层证据：传染性 / 疾病状态 / 进展）

将患者与接触者记录中的可选实验室字段（GeneXpert Ct、RIF 耐药、涂片分级、
IGRA/TST、影像学）统一换算为可乘/可加的传染性因子、证据等级与进展修正，
供 patient_scorer / scoring.engine / contact_risk_calculator 复用。

三层证据（对传染性/进展的直接性）：
- 传染性层（源病例）：杆菌载量（GeneXpert Ct 值 / 涂片分级）
- 疾病状态层（源病例）：影像学范围（空洞/树芽征/实变）
- 进展层（接触者）：IGRA/TST 感染状态 → 潜伏→活动性进展起点

证据优先级（避免冲突，取最高等级）：
    GeneXpert(Ct+RIF)  >  涂片分级  >  影像学  >  症状/接触推断

缺失处理：任一层缺失都回退到现有逻辑（症状/接触史/场景），不报错、
不改变历史结果，仅 lab_evidence_grade 记为 0。

文献（详见 constants.py 对应区块）：
- Ct 值与涂片等级显著负相关：Xpert r≈-0.77、Ultra r≈-0.69，全球中位 Ct≈20.1
- Ct=25 预测涂阳约 95% 敏感/65% 特异；Ct≈27.7 约 98%/48%；截断 28 对涂阳有较好预测价值
- 树芽征存在于约 72% 活动性结核；空洞/树芽征/实变提示排菌与传染性高
- IGRA 在 BCG 接种人群中比 TST 更特异
- IGRA 敏感度（免疫健全成人）：T-SPOT 约 68%、QFT-GIT 约 52%，特异度约 97%；
  HIV 感染者 T-SPOT 约 72%、QFT-GIT 约 61%
"""

import math
import re
from typing import Any, Dict, Optional, Set

from ..constants import (
    SPUTUM_SMEAR_POSITIVE_INCREASE,
    CT_BACILLARY_LOAD, CT_HIGH_THRESHOLD, CT_MEDIUM_THRESHOLD,
    SMEAR_GRADE_LOAD,
    IMAGING_EXTENT_FACTORS,
    IGRA_TYPE_WEIGHT, IGRA_TYPE_DEFAULT_WEIGHT,
    IGRA_NEGATIVE_SUPPRESS, IGRA_INDETERMINATE_FACTOR,
    MDR_INFECTIVITY_LATE, MDR_INFECTIVITY_COMPLETED,
    MDR_TREATMENT_LATE_THRESHOLD_DAYS, MDR_TREATMENT_COMPLETED_THRESHOLD_DAYS,
    TREATMENT_INFECTIVITY_FACTORS,
    TREATMENT_EARLY_THRESHOLD_DAYS, TREATMENT_MID_THRESHOLD_DAYS,
    TREATMENT_LATE_THRESHOLD_DAYS,
    LOW_EVIDENCE_UNCERTAINTY_FACTORS,
    LAB_EVIDENCE_GRADES,
    # 双输出进展建模
    LATENT_SHORT_TERM_BASELINE,
    TIME_DECAY_FACTORS, TIME_DECAY_LONG_TERM_FACTOR,
    TIME_DECAY_DEFAULT_MONTHS,
    IGRA_POSITIVE_PROB, IGRA_NEGATIVE_REMAIN,
    IGRA_INDETERMINATE_PRIOR, IGRA_NOT_DONE_PRIOR_CAP,
    RECENT_CONVERSION_BOOST,
    SMOKING_PROGRESSION_FACTOR,
    BMI_PROGRESSION_FACTOR, BMI_DEFAULT,
    VITAMIN_D_PROGRESSION_FACTOR, VITAMIN_D_DEFAULT,
)

# ------------------------------------------------------------------------------
# 影像学位掩码 → finding key 映射（bit 位置）
# ------------------------------------------------------------------------------
IMAGING_BIT_INDEX = {
    'cavity': 0,
    'tree_in_bud': 1,
    'consolidation': 2,
    'miliary': 3,
    'caseous_pneumonia': 4,
    'pleural_effusion': 5,
    'nodule': 6,
}

# 中文 → 英文 finding key 映射
IMAGING_CN_MAP = {
    '空洞': 'cavity',
    '树芽征': 'tree_in_bud', '树芽': 'tree_in_bud',
    '实变': 'consolidation', '大叶实变': 'consolidation',
    '粟粒': 'miliary', '粟粒样': 'miliary',
    '干酪样肺炎': 'caseous_pneumonia', '干酪肺炎': 'caseous_pneumonia',
    '胸膜积液': 'pleural_effusion', '胸腔积液': 'pleural_effusion',
    '结节': 'nodule',
}


def _is_missing(value: Any) -> bool:
    """判断实验室字段是否缺失（缺失视为未检测）。"""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == '' or value.strip().lower() in (
            'none', 'nan', 'null', 'not_done', '未检测', '无', 'n/a')
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _parse_bool_yes(value: Any) -> bool:
    """宽松判断二值字段是否为"是/阳"。"""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) in (1, 2)
    return str(value).strip() in ('是', '有', '阳性', '涂阳', '阳', 'yes', 'true')


def _parse_int(value: Any) -> Optional[int]:
    """宽松解析整数，失败返回 None。"""
    if _is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------
# 影像学解析
# ------------------------------------------------------------------------------

def parse_imaging_findings(imaging_findings: Any) -> Set[str]:
    """将 imaging_findings 归一化为 finding key 集合。

    支持三种输入形式：
    - 整数位掩码（bit 0=cavity, 1=tree_in_bud, ...）
    - 字符串列表（['cavity', 'tree_in_bud']）
    - 字符串（逗号/顿号/斜杠/空白分隔，支持中文）
    """
    if _is_missing(imaging_findings):
        return set()
    if isinstance(imaging_findings, bool):
        return set()
    if isinstance(imaging_findings, (int, float)):
        mask = int(imaging_findings)
        if mask <= 0:
            return set()
        return {key for key, bit in IMAGING_BIT_INDEX.items()
                if mask & (1 << bit)}
    if isinstance(imaging_findings, str):
        parts = re.split(r'[,，、/;；\s]+', imaging_findings)
    elif isinstance(imaging_findings, (list, tuple, set)):
        parts = list(imaging_findings)
    else:
        return set()

    out: Set[str] = set()
    for p in parts:
        key = str(p).strip().lower()
        if not key:
            continue
        key = IMAGING_CN_MAP.get(key, key)
        if key in IMAGING_EXTENT_FACTORS:
            out.add(key)
    return out


# ------------------------------------------------------------------------------
# 杆菌载量（传染性层）
# ------------------------------------------------------------------------------

def bacillary_load(ct_value: Any = None, smear_grade: Any = None,
                   sputum_smear: Any = None) -> Optional[float]:
    """杆菌载量因子（0.3–1.0），优先级：Ct 值 > 涂片分级 > 原二值涂阳。

    参数：
        ct_value: GeneXpert/Xpert Ultra Ct 值（连续、可量化）
        smear_grade: 涂片分级 0–3+
        sputum_smear: 原二值涂阳（True/2/涂阳 表示阳性）

    返回：
        float | None — 杆菌载量因子；无任何传染性证据时返回 None
    """
    # 1) Ct 值优先（连续、可量化，替代指标）
    if not _is_missing(ct_value):
        try:
            ct = float(ct_value)
        except (TypeError, ValueError):
            ct = None
        if ct is not None:
            if ct < CT_HIGH_THRESHOLD:
                return float(CT_BACILLARY_LOAD['high'])
            if ct <= CT_MEDIUM_THRESHOLD:
                return float(CT_BACILLARY_LOAD['medium'])
            return float(CT_BACILLARY_LOAD['low'])

    # 2) 涂片分级（0–3+）
    grade = _parse_int(smear_grade)
    if grade is not None and grade >= 0:
        return float(SMEAR_GRADE_LOAD.get(min(grade, 3), 0.3))

    # 3) 原二值涂阳（向后兼容）
    if _parse_bool_yes(sputum_smear):
        return float(CT_BACILLARY_LOAD['high'])

    return None


def imaging_extent_factor(imaging_findings: Any) -> Optional[float]:
    """影像学范围因子（疾病状态层，替代/修正杆菌载量，0.4–0.8）。

    cavity / tree_in_bud → 0.8；consolidation / miliary → 0.6；仅 nodule → 0.4。
    无影像证据时返回 None。
    """
    findings = parse_imaging_findings(imaging_findings)
    if not findings:
        return None
    return max(IMAGING_EXTENT_FACTORS.get(f, 0.4) for f in findings)


def compute_infectivity_bonus(record: Dict[str, Any]) -> float:
    """传染性加分 = SPUTUM_SMEAR_POSITIVE_INCREASE × 杆菌载量因子。

    用 Ct 分档连续因子平滑替换原二值 SPUTUM_SMEAR_POSITIVE_INCREASE=20 加法。
    Ct/涂片缺失但影像可用时，用影像学范围因子替代/修正杆菌载量。
    无任何实验室证据时回退到旧逻辑（涂阳才加分）。
    """
    if not record:
        return 0.0
    load = bacillary_load(
        ct_value=record.get('genexpert_ct'),
        smear_grade=record.get('smear_grade'),
        sputum_smear=record.get('sputum_smear'),
    )
    if load is None:
        # Ct/涂片缺失 → 用影像学替代/修正杆菌载量
        load = imaging_extent_factor(record.get('imaging_findings'))
    if load is None:
        # 无实验室证据 → 保持旧逻辑：仅涂阳加分
        load = 1.0 if _parse_bool_yes(record.get('sputum_smear')) else 0.0
    return SPUTUM_SMEAR_POSITIVE_INCREASE * load


# ------------------------------------------------------------------------------
# 证据等级（冲突裁决）
# ------------------------------------------------------------------------------

def derive_lab_evidence_grade(record: Dict[str, Any]) -> int:
    """按证据强度裁决实验室证据等级（0–3）。

    证据等级：GeneXpert(Ct+RIF)[3] > 涂片分级/IGRA[2] > 影像学[1] > 症状推断[0]。
    缺失视为未检测，不报错，仅返回 0。
    """
    if not record:
        return 0

    # 等级 3：GeneXpert 分子证据（Ct 或 RIF）
    ct = record.get('genexpert_ct')
    rif = record.get('genexpert_rif')
    if not _is_missing(ct):
        try:
            float(ct)
            return 3
        except (TypeError, ValueError):
            pass
    if not _is_missing(rif):
        return 3

    # 等级 2：涂片分级 / IGRA（确认感染状态的实验室检测）
    if _parse_int(record.get('smear_grade')) is not None:
        return 2
    if not _is_missing(record.get('igra_result')):
        return 2

    # 等级 1：影像学证据
    if parse_imaging_findings(record.get('imaging_findings')):
        return 1

    return 0


def lab_evidence_grade_label(grade: int) -> str:
    """返回证据等级的中文标签。"""
    return LAB_EVIDENCE_GRADES.get(grade, LAB_EVIDENCE_GRADES[0])


# ------------------------------------------------------------------------------
# MDR（RIF 耐药）治疗传染性衰减曲线
# ------------------------------------------------------------------------------

def _standard_treatment_infectivity(days: int) -> float:
    """标准（非 MDR）治疗阶段传染性因子（保持现有逻辑）。"""
    if days <= 0:
        return TREATMENT_INFECTIVITY_FACTORS['pre_treatment']
    if days < TREATMENT_EARLY_THRESHOLD_DAYS:
        return TREATMENT_INFECTIVITY_FACTORS['early_treatment']
    if days < TREATMENT_MID_THRESHOLD_DAYS:
        return TREATMENT_INFECTIVITY_FACTORS['mid_treatment']
    if days < TREATMENT_LATE_THRESHOLD_DAYS:
        return TREATMENT_INFECTIVITY_FACTORS['late_treatment']
    return TREATMENT_INFECTIVITY_FACTORS['completed_treatment']


def mdr_treatment_infectivity(patient_treatment_days: Any,
                              is_mdr: bool = False) -> float:
    """治疗阶段传染性因子（MDR 耐药时衰减曲线拉长）。

    非 MDR：沿用 TREATMENT_INFECTIVITY_FACTORS 与既有阈值。
    MDR：late_treatment / completed_treatment 档位与阈值拉长——
        后期治疗传染性 0.60（原 0.30）、完成治疗 0.30（原 0.10），
        后期阈值 365 天（原 180）、完成阈值 730 天（WHO 2020 长程 24 月）。

    参数：
        patient_treatment_days: 患者已治疗天数
        is_mdr: genexpert_rif == resistant
    """
    try:
        days = int(patient_treatment_days) if patient_treatment_days is not None else 0
    except (TypeError, ValueError):
        days = 0

    if not is_mdr:
        return _standard_treatment_infectivity(days)

    # MDR：衰减曲线拉长
    if days <= 0:
        return TREATMENT_INFECTIVITY_FACTORS['pre_treatment']
    if days < TREATMENT_EARLY_THRESHOLD_DAYS:
        return TREATMENT_INFECTIVITY_FACTORS['early_treatment']
    if days < TREATMENT_MID_THRESHOLD_DAYS:
        return TREATMENT_INFECTIVITY_FACTORS['mid_treatment']
    if days < MDR_TREATMENT_LATE_THRESHOLD_DAYS:
        return MDR_INFECTIVITY_LATE
    if days < MDR_TREATMENT_COMPLETED_THRESHOLD_DAYS:
        return MDR_INFECTIVITY_COMPLETED
    # 超过 2 年：进一步下降（MDR 治愈后传染性显著降低但仍高于敏感株治愈）
    return MDR_INFECTIVITY_COMPLETED * 0.5


def is_mdr_record(record: Dict[str, Any]) -> bool:
    """判断记录是否为 MDR（genexpert_rif == resistant，支持中文）。"""
    if not record:
        return False
    rif = record.get('genexpert_rif')
    if _is_missing(rif):
        return False
    return str(rif).strip().lower() in (
        'resistant', '耐药', 'r', 'res')

# ------------------------------------------------------------------------------
# IGRA / TST（进展层，接触者）
# ------------------------------------------------------------------------------

def igra_type_weight(igra_type: Any) -> float:
    """IGRA 类型置信度权重（T-SPOT 1.0 / QuantiFERON 0.9 / TST 0.7）。"""
    if _is_missing(igra_type):
        return IGRA_TYPE_DEFAULT_WEIGHT
    key = str(igra_type).strip().lower()
    aliases = {
        't_spot': ('t_spot', 't-spot', 'tspot', 'tspot.tb', 't-spot.tb'),
        'quantiferon': ('quantiferon', 'qft', 'qft-git', 'qft_git', 'qftgit',
                        'quantiferon-tb', 'quanti feron'),
        'tst': ('tst', 'ppd', '结核菌素', '皮试'),
    }
    for canonical, syns in aliases.items():
        if key in syns or key == canonical:
            return float(IGRA_TYPE_WEIGHT[canonical])
    return IGRA_TYPE_DEFAULT_WEIGHT


def igra_progression_factor(record: Dict[str, Any]) -> float:
    """基于 IGRA 感染状态的进展概率修正因子。

    igra_result:
      positive  → 已感染，启用完整 LATENT_BASELINE（×1.0，年龄/免疫因子外层叠加）
      negative  → 未感染基本不进展（×IGRA_NEGATIVE_SUPPRESS≈0.05）
      indeterminate → 部分保留（×IGRA_INDETERMINATE_FACTOR≈0.8）
      not_done / 缺失 → 保持现状（×1.0）

    igra_type 作小的置信度权重：对 negative/indeterminate 的修正按测试可信度
    向"未检测"（1.0）回摆——低置信度测试（TST）的阴性结果可信度更低。
    """
    if not record:
        return 1.0
    result = str(record.get('igra_result', 'not_done')).strip().lower()
    if result in ('', 'none', 'nan', 'not_done', '未检测', '无'):
        return 1.0

    weight = igra_type_weight(record.get('igra_type'))

    if result in ('positive', 'pos', '阳性', '阳', '是'):
        return 1.0  # 已感染：启用完整基线
    if result in ('negative', 'neg', '阴性', '阴', '否'):
        factor = IGRA_NEGATIVE_SUPPRESS
    elif result in ('indeterminate', 'indet', '不确定', '可疑', 'borderline'):
        factor = IGRA_INDETERMINATE_FACTOR
    else:
        return 1.0
    # 低置信度测试：结果影响向"未检测"（1.0）回摆
    return 1.0 + (factor - 1.0) * weight


# ------------------------------------------------------------------------------
# 双输出进展建模：是否已感染 × 近期(1–2年)进展概率
# ------------------------------------------------------------------------------

def _parse_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """宽松解析浮点数，失败返回 default。"""
    if _is_missing(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_bool(value: Any, default: bool = False) -> bool:
    """宽松解析布尔（支持中文/数字/字符串）。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    return str(value).strip() in ('是', '有', 'true', '1', 'yes')


def time_decay_factor(time_since_exposure_months: Any) -> float:
    """时间衰减函数：距最近高危暴露的月数 → 近期(1–2年)进展衰减。

    文献：进展高度集中在前 1–2 年，远期再激活率显著更低——
      t ≤ 6        → 1.0（近期感染/刚阳转，进展率高峰）
      6 < t ≤ 24   → 0.5（前 2 年，约一半病例集中于此）
      24 < t ≤ 60  → 0.2（2–5 年，显著下降）
      t > 60       → 0.1（远期再激活，约 200/10 万人年量级）

    缺失/非法输入 → TIME_DECAY_DEFAULT_MONTHS（0.5 档，保守居中）。
    """
    t = _parse_float(time_since_exposure_months)
    if t is None or t < 0:
        t = TIME_DECAY_DEFAULT_MONTHS
    factor = TIME_DECAY_LONG_TERM_FACTOR
    for cutoff in sorted(TIME_DECAY_FACTORS):
        if t <= cutoff:
            factor = TIME_DECAY_FACTORS[cutoff]
            break
    return float(factor)


def latent_infection_probability(record: Dict[str, Any],
                                 exposure_prior: Optional[float] = None) -> float:
    """P_infected：当前是否已感染（LTBI 患病概率），用 IGRA 判定而非直接给基线。

      igra_result:
        positive      → IGRA_POSITIVE_PROB（0.92，宣布已感染，特异度高可信）
        negative      → IGRA_NEGATIVE_REMAIN（0.05，残留假阴性，不能完全排除）
        indeterminate → 暴露先验（缺省 IGRA_INDETERMINATE_PRIOR=0.30）
        not_done/缺失 → 暴露先验（沿用现有症状/接触推断）

    参数：
        record: 接触者记录
        exposure_prior: 暴露先验（0–1，由现有 exposure 推断给出）；
            未提供时用 IGRA_INDETERMINATE_PRIOR 缺省。

    返回：
        0–1 的感染概率。
    """
    if not record and exposure_prior is None:
        return float(IGRA_INDETERMINATE_PRIOR)

    result = str(record.get('igra_result', 'not_done')).strip().lower()
    if result in ('positive', 'pos', '阳性', '阳', '是'):
        return float(IGRA_POSITIVE_PROB)
    if result in ('negative', 'neg', '阴性', '阴', '否'):
        return float(IGRA_NEGATIVE_REMAIN)
    if result in ('indeterminate', 'indet', '不确定', '可疑', 'borderline'):
        prior = exposure_prior if exposure_prior is not None else IGRA_INDETERMINATE_PRIOR
        return min(max(float(prior), float(IGRA_INDETERMINATE_PRIOR)),
                   float(IGRA_POSITIVE_PROB))

    # not_done / 缺失：沿用现有症状/接触推断（暴露先验），上限不超过阳性确认值
    if exposure_prior is not None:
        return min(max(float(exposure_prior), 0.0), float(IGRA_NOT_DONE_PRIOR_CAP))
    return float(IGRA_INDETERMINATE_PRIOR)


def lifestyle_progression_factor(record: Dict[str, Any]) -> float:
    """生活方式与强化危险信号修正乘数（吸烟 / BMI / 维生素 D）。

    smoking × (1 + RECENT_CONVERSION_BOOST × recent_conversion)
    BMI 分档（低体重 2.0 / 正常 1.0 / 超重 0.95 / 肥胖 0.9）
    维生素 D 分档（缺乏 1.3 / 不足 1.15 / 正常 1.0 / 充足 0.95）
    缺失字段按默认档处理（×1.0），不改变现有流程。
    """
    factor = 1.0
    if _parse_bool(record.get('smoking')):
        factor *= SMOKING_PROGRESSION_FACTOR

    bmi = _parse_float(record.get('bmi'))
    if bmi is None:
        bmi = BMI_DEFAULT
    if bmi < 18.5:
        factor *= BMI_PROGRESSION_FACTOR['low_underweight']
    elif bmi < 22:
        factor *= BMI_PROGRESSION_FACTOR['low_normal']
    elif bmi < 25:
        factor *= BMI_PROGRESSION_FACTOR['normal']
    elif bmi < 30:
        factor *= BMI_PROGRESSION_FACTOR['overweight']
    else:
        factor *= BMI_PROGRESSION_FACTOR['obese']

    vd = _parse_float(record.get('vitamin_d'))
    if vd is None:
        vd = VITAMIN_D_DEFAULT
    if vd < 12:
        factor *= VITAMIN_D_PROGRESSION_FACTOR['deficient']
    elif vd < 20:
        factor *= VITAMIN_D_PROGRESSION_FACTOR['insufficient']
    elif vd < 40:
        factor *= VITAMIN_D_PROGRESSION_FACTOR['normal']
    else:
        factor *= VITAMIN_D_PROGRESSION_FACTOR['optimal']

    return float(factor)


def short_term_progression_probability(
        record: Dict[str, Any],
        age_multiplier: float = 1.0,
        immuno_multiplier: float = 1.0,
        bcg_protective: float = 0.0,
        cap: Optional[float] = None) -> float:
    """P_progress(1–2y | infected)：给定已感染后的近期(1–2年)进展概率。

    P_progress = LATENT_SHORT_TERM_BASELINE × time_decay(t)
                 × age_mult × immuno_mult
                 × (1 − bcg_protective) × lifestyle_mult
                 × (1 + RECENT_CONVERSION_BOOST × recent_conversion)

    参数：
        record: 接触者记录（time_since_exposure_months / recent_conversion /
                smoking / bmi / vitamin_d）
        age_multiplier: 年龄进展乘数（复用 AGE_PROGRESSION）
        immuno_multiplier: 免疫进展乘数（复用 IMMUNO_FACTORS_BASE）
        bcg_protective: BCG 保护效力（0–1，未接种为 0）
        cap: 概率上限（建议传 MAX_PROGRESSION_RATE=0.30，防止极端组合超 1）

    返回：
        0–1 的 2 年短程进展概率。
    """
    if not record:
        record = {}
    decay = time_decay_factor(record.get('time_since_exposure_months'))
    lifestyle = lifestyle_progression_factor(record)
    conversion_boost = (1.0 + RECENT_CONVERSION_BOOST
                        if _parse_bool(record.get('recent_conversion'))
                        else 1.0)

    prob = (LATENT_SHORT_TERM_BASELINE * decay
            * max(float(age_multiplier), 0.0)
            * max(float(immuno_multiplier), 0.0)
            * max(1.0 - float(bcg_protective), 0.0)
            * lifestyle * conversion_boost)
    if cap is not None:
        prob = min(prob, float(cap))
    return float(max(prob, 0.0))


# ------------------------------------------------------------------------------
# 低证据告警（与不确定性模块联动）
# ------------------------------------------------------------------------------

def uncertainty_adaptive_factor(record: Dict[str, Any]) -> float:
    """低证据告警：lab_evidence_grade 越低，共形预测区间越宽。

    grade 0（无实验室证据）→ 1.6（区间显著放大）
    grade 1（影像学）→ 1.3
    grade ≥ 2 → 1.0
    """
    grade = derive_lab_evidence_grade(record)
    return float(LOW_EVIDENCE_UNCERTAINTY_FACTORS.get(grade, 1.0))


def low_evidence_warning(record: Dict[str, Any]) -> Optional[str]:
    """无实验室证据时的告警文本（GUI 标注用）。

    lab_evidence_grade=0 时返回"基于症状推断，建议实验室确诊"；
    否则返回 None。
    """
    if derive_lab_evidence_grade(record) == 0:
        return '基于症状推断，建议实验室确诊'
    return None


__all__ = [
    'IMAGING_BIT_INDEX',
    'parse_imaging_findings',
    'bacillary_load',
    'imaging_extent_factor',
    'compute_infectivity_bonus',
    'derive_lab_evidence_grade',
    'lab_evidence_grade_label',
    'mdr_treatment_infectivity',
    'is_mdr_record',
    'igra_type_weight',
    'igra_progression_factor',
    'uncertainty_adaptive_factor',
    'low_evidence_warning',
    # 双输出进展建模
    'time_decay_factor',
    'latent_infection_probability',
    'lifestyle_progression_factor',
    'short_term_progression_probability',
]
