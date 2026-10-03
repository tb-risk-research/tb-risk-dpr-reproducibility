#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""层间数据契约 schema（dataclass）。

问题六：定义显式 dataclass 作为 data_io → core → integrator 各层间的数据契约，
替代之前三套异构 dict 形状（list[dict] 扁平记录 / patient_info 嵌套 / potential_patients dict）。
按 record_id 而非 name 字符串匹配，字段错配在 schema 层即可被捕获。

设计原则：
1. 仅用 stdlib dataclasses，不引入 Pydantic 依赖（项目以科研工具为主，保持轻量）。
2. 提供 from_dict / to_dict 以兼容既有 dict 消费者（渐进迁移，不破坏现有代码）。
3. 字段集保持最小 — 仅纳入跨层契约实际使用的字段，避免过度设计。
4. record_id 是 ContactRecord / ContactRiskResult 的主键，由 data_io 或
   ContactRiskCalculator 生成（优先 _id，回退稳定合成 id）。

契约流向：
    data_io.import_pipeline → list[PatientRecord | ContactRecord]
    RiskAssessmentService.assess(PatientRecord, ...) → AssessmentResult
    ContactRiskCalculator.generate_potential_patients → list[ContactRiskResult]
    ThreeDirectionIntegrator._get_seir_results 按 record_id 匹配 ContactRiskResult
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ==================== 患者记录 ====================

@dataclass
class PatientRecord:
    """患者记录（data_io 产出 → 评估服务消费）。

    封装 patient_info 的核心字段：basic_info（基础临床信息）+
    FCI/SNC（家庭/社会接触强度综合分）。
    basic_info 中的 flp_percentage / hrsp_percentage / delay_days 由 GUI 层
    或 data_io 文本抽取填入；本 dataclass 仅声明契约，不做字段抽取。

    可选实验室字段（缺失视为"未检测"，不破坏现有流程）：
      - genexpert_ct (float)      GeneXpert/Xpert Ultra Ct 值（连续杆菌载量）
      - genexpert_rif (str)       RIF 耐药：susceptible / resistant / not_done
      - smear_grade (int)         涂片分级 0–3+（替代二值 sputum_smear）
      - imaging_findings (int/str) 影像学多选位掩码（cavity/tree_in_bud/...）
      - lab_evidence_grade (int)  自动推导的证据等级 0–3
    上述字段存于 basic_info 中，随 basic_info 自然透传。
    """
    basic_info: Dict[str, Any] = field(default_factory=dict)
    FCI: float = 0.0
    SNC: float = 0.0
    # 可选扩展字段（如 data_io 质量评分、审计元数据）
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> 'PatientRecord':
        """从 patient_info dict 宽松构造（兼容既有 patient_info 形状）。"""
        if not isinstance(data, dict):
            return cls()
        return cls(
            basic_info=copy.deepcopy(data.get('basic_info', {})) or {},
            FCI=float(data.get('FCI', 0.0) or 0.0),
            SNC=float(data.get('SNC', 0.0) or 0.0),
            meta={k: v for k, v in data.items()
                  if k not in ('basic_info', 'FCI', 'SNC')},
        )

    def to_dict(self) -> Dict[str, Any]:
        """序列化回 patient_info dict 形状（兼容既有消费者）。"""
        d = copy.deepcopy(self.meta)
        d['basic_info'] = copy.deepcopy(self.basic_info)
        d['FCI'] = self.FCI
        d['SNC'] = self.SNC
        return d


# ==================== 接触者记录 ====================

@dataclass
class ContactRecord:
    """接触者记录（家庭/社会通用）。

    data_io 产出 → ContactRiskCalculator 消费 → 产出 ContactRiskResult。
    record_id 作为跨层匹配主键：优先使用数据库 _id，缺失时由 data_io 或
    ContactRiskCalculator 生成稳定合成 id（如 "family_0"、"social_2"）。
    """
    record_id: str
    name: str = '未知'
    contact_type: str = 'family'  # 'family' | 'social'
    _id: Optional[str] = None  # 数据库主键（可能为 None）
    age: Optional[int] = None
    relationship: str = '家庭成员'
    freq_density: float = 0.0
    time_span: float = 0.0
    # 可选实验室字段（进展层证据，缺失视为"未检测"）
    igra_result: Optional[str] = None       # positive / negative / indeterminate / not_done
    igra_type: Optional[str] = None         # T-SPOT.TB / QuantiFERON / TST
    genexpert_ct: Optional[float] = None    # GeneXpert Ct 值（若接触者已确诊/检测）
    genexpert_rif: Optional[str] = None     # susceptible / resistant / not_done
    smear_grade: Optional[int] = None       # 涂片分级 0–3+
    imaging_findings: Optional[Any] = None  # 影像学多选位掩码
    lab_evidence_grade: int = 0             # 自动推导的证据等级 0–3
    # 双输出进展建模：时间维度 / 感染门槛 / 生活方式
    recent_conversion: Optional[bool] = None  # 近期 IGRA 由阴转阳（新鲜感染）
    time_since_exposure_months: Optional[float] = None  # 距最近高危暴露的月数
    smoking: Optional[bool] = None            # 吸烟
    bmi: Optional[float] = None               # 体质指数
    vitamin_d: Optional[float] = None         # 维生素 D（ng/mL）
    # 其他原始字段透传（ventilation/workplace_type/district 等）
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict],
                  contact_type: str = 'family',
                  fallback_idx: Optional[int] = None) -> 'ContactRecord':
        """从接触者 dict 宽松构造。

        fallback_idx：当 _id 与 record_id 均缺失时，用于生成合成 id 的索引
        （如 f"family_{fallback_idx}"）。由调用方（data_io 或 Calculator）传入。

        字段同义词回退：管线同义词映射将 CSV ``name`` 列映射为
        ``member_name``（family）或 ``contact_name``（social），``age`` 映射为
        ``member_age``/``contact_age``。from_dict 按优先级回退查找，确保
        管线产出记录的 name/age 被正确提取。同义字段保留在 extra 中，
        to_dict 原样输出，兼容期望 member_name/contact_name 的下游消费者。
        """
        if not isinstance(data, dict):
            data = {}
        _id = data.get('_id')
        rid = data.get('record_id') or _id
        if rid is None:
            if fallback_idx is not None:
                rid = f"{contact_type}_{fallback_idx}"
            else:
                rid = ''
        known_keys = {
            '_id', 'record_id', 'name', 'age', 'relationship',
            'freq_density', 'time_span',
            'igra_result', 'igra_type', 'genexpert_ct', 'genexpert_rif',
            'smear_grade', 'imaging_findings', 'lab_evidence_grade',
            'recent_conversion', 'time_since_exposure_months',
            'smoking', 'bmi', 'vitamin_d',
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}
        # 字段同义词回退：name → member_name → contact_name
        name = (data.get('name') or data.get('member_name')
                or data.get('contact_name') or '未知')
        # age → member_age → contact_age
        age = data.get('age') or data.get('member_age') or data.get('contact_age')
        return cls(
            record_id=rid,
            name=name,
            contact_type=contact_type,
            _id=_id,
            age=age,
            relationship=data.get('relationship', '家庭成员'),
            freq_density=float(data.get('freq_density', 0.0) or 0.0),
            time_span=float(data.get('time_span', 0.0) or 0.0),
            igra_result=data.get('igra_result'),
            igra_type=data.get('igra_type'),
            genexpert_ct=data.get('genexpert_ct'),
            genexpert_rif=data.get('genexpert_rif'),
            smear_grade=data.get('smear_grade'),
            imaging_findings=data.get('imaging_findings'),
            lab_evidence_grade=int(data.get('lab_evidence_grade', 0) or 0),
            recent_conversion=data.get('recent_conversion'),
            time_since_exposure_months=data.get('time_since_exposure_months'),
            smoking=data.get('smoking'),
            bmi=data.get('bmi'),
            vitamin_d=data.get('vitamin_d'),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        """序列化回接触者 dict 形状（兼容既有 potential_patients 结构）。"""
        d = {
            '_id': self._id,
            'record_id': self.record_id,
            'name': self.name,
            'age': self.age,
            'relationship': self.relationship,
            'freq_density': self.freq_density,
            'time_span': self.time_span,
            'igra_result': self.igra_result,
            'igra_type': self.igra_type,
            'genexpert_ct': self.genexpert_ct,
            'genexpert_rif': self.genexpert_rif,
            'smear_grade': self.smear_grade,
            'imaging_findings': self.imaging_findings,
            'lab_evidence_grade': self.lab_evidence_grade,
            'recent_conversion': self.recent_conversion,
            'time_since_exposure_months': self.time_since_exposure_months,
            'smoking': self.smoking,
            'bmi': self.bmi,
            'vitamin_d': self.vitamin_d,
        }
        d.update(copy.deepcopy(self.extra))
        return d


# ==================== 接触者风险结果 ====================

@dataclass
class ContactRiskResult:
    """单接触者风险计算结果（ContactRiskCalculator 产出 → integrator 消费）。

    问题二：保留 rule/seir/融合 三分段，避免 SEIR 贡献被双重计入。
    问题六：record_id 作为 integrator 匹配主键（替代 name 字符串匹配）。
    """
    record_id: str
    name: str = '未知'
    contact_type: str = 'family'
    risk_score: float = 0.0
    # 三分段概率（0-100 尺度）
    rule_infection_probability: float = 0.0
    seir_infection_probability: float = 0.0
    infection_probability: float = 0.0  # 融合后
    disease_probability: float = 0.0
    latent_infection_prob: float = 0.0     # 潜伏感染概率（P_infected）
    short_term_active_risk: float = 0.0  # 近期(1–2年)发病风险（P_infected × P_progress）
    priority: str = '低'
    is_diagnosed: bool = False
    recommendation: str = ''
    # 实验室证据（接触者侧）
    lab_evidence_grade: int = 0       # 自动推导的证据等级 0–3
    evidence_note: str = ''           # 证据标注（如低证据告警文本）
    # 透传原始字段（描述/年龄/民族/地区等）
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict],
                  contact_type: str = 'family',
                  fallback_idx: Optional[int] = None) -> 'ContactRiskResult':
        """从 potential_patients 列表项 dict 宽松构造。"""
        if not isinstance(data, dict):
            data = {}
        _id = data.get('_id')
        rid = data.get('record_id') or _id
        if rid is None:
            if fallback_idx is not None:
                rid = f"{contact_type}_{fallback_idx}"
            else:
                rid = ''
        known_keys = {
            '_id', 'record_id', 'name', 'risk_score',
            'rule_infection_probability', 'seir_infection_probability',
            'infection_probability', 'disease_probability',
            'latent_infection_prob', 'short_term_active_risk',
            'priority', 'is_diagnosed', 'recommendation',
            'lab_evidence_grade', 'evidence_note',
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            record_id=rid,
            name=data.get('name', '未知') or '未知',
            contact_type=contact_type,
            risk_score=float(data.get('risk_score', 0.0) or 0.0),
            rule_infection_probability=float(
                data.get('rule_infection_probability', 0.0) or 0.0),
            seir_infection_probability=float(
                data.get('seir_infection_probability', 0.0) or 0.0),
            infection_probability=float(
                data.get('infection_probability', 0.0) or 0.0),
            disease_probability=float(
                data.get('disease_probability', 0.0) or 0.0),
            latent_infection_prob=float(
                data.get('latent_infection_prob', 0.0) or 0.0),
            short_term_active_risk=float(
                data.get('short_term_active_risk', 0.0) or 0.0),
            priority=str(data.get('priority', '低') or '低'),
            is_diagnosed=bool(data.get('is_diagnosed', False)),
            recommendation=str(data.get('recommendation', '') or ''),
            lab_evidence_grade=int(data.get('lab_evidence_grade', 0) or 0),
            evidence_note=str(data.get('evidence_note', '') or ''),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        """序列化回 potential_patients 列表项 dict 形状。"""
        d = {
            '_id': self.extra.get('_id'),
            'record_id': self.record_id,
            'name': self.name,
            'risk_score': self.risk_score,
            'rule_infection_probability': self.rule_infection_probability,
            'seir_infection_probability': self.seir_infection_probability,
            'infection_probability': self.infection_probability,
            'disease_probability': self.disease_probability,
            'latent_infection_prob': self.latent_infection_prob,
            'short_term_active_risk': self.short_term_active_risk,
            'priority': self.priority,
            'is_diagnosed': self.is_diagnosed,
            'recommendation': self.recommendation,
            'lab_evidence_grade': self.lab_evidence_grade,
            'evidence_note': self.evidence_note,
        }
        # 透传 extra（排除已显式列出的 _id）
        for k, v in self.extra.items():
            if k != '_id':
                d[k] = v
        return d


# ==================== 评估结果 ====================

@dataclass
class AssessmentResult:
    """完整风险评估结果（RiskAssessmentService.assess 产出）。

    封装 assess() 返回 dict 的核心字段，供 GUI/CLI/REST 消费者统一使用。
    potential_patients 保持 {'family': [...], 'social': [...]} 形状以兼容
    既有消费者；消费者可按需将内层 dict 转为 ContactRiskResult。
    """
    patient_score: float = 0.0
    base_infection_probability: float = 0.0
    overall_risk: str = '低'
    overall_suggestion: str = ''
    potential_patients: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=lambda: {'family': [], 'social': []})
    summary: Dict[str, Any] = field(default_factory=dict)
    individual_risks: Dict[str, Any] = field(default_factory=dict)
    ml_results: Optional[Dict[str, Any]] = None
    seir_results: Optional[Dict[str, Any]] = None
    # 透传原始字段（total_score/weights/patient_risk_multiplier 等）
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> 'AssessmentResult':
        """从 assess() 返回 dict 宽松构造。"""
        if not isinstance(data, dict):
            return cls()
        known_keys = {
            'patient_score', 'base_infection_probability', 'overall_risk',
            'overall_suggestion', 'potential_patients', 'summary',
            'individual_risks', 'ml_results', 'seir_results',
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}
        pp = data.get('potential_patients', {})
        if not isinstance(pp, dict):
            pp = {'family': [], 'social': []}
        pp = {
            'family': list(pp.get('family', [])),
            'social': list(pp.get('social', [])),
        }
        return cls(
            patient_score=float(data.get('patient_score', 0.0) or 0.0),
            base_infection_probability=float(
                data.get('base_infection_probability', 0.0) or 0.0),
            overall_risk=str(data.get('overall_risk', '低') or '低'),
            overall_suggestion=str(
                data.get('overall_suggestion', '') or ''),
            potential_patients=pp,
            summary=copy.deepcopy(data.get('summary', {}) or {}),
            individual_risks=copy.deepcopy(
                data.get('individual_risks', {}) or {}),
            ml_results=copy.deepcopy(data.get('ml_results')) or None,
            seir_results=copy.deepcopy(data.get('seir_results')) or None,
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        """序列化回 assess() 返回 dict 形状（兼容既有消费者）。"""
        d = {
            'patient_score': self.patient_score,
            'base_infection_probability': self.base_infection_probability,
            'overall_risk': self.overall_risk,
            'overall_suggestion': self.overall_suggestion,
            'potential_patients': copy.deepcopy(self.potential_patients),
            'summary': copy.deepcopy(self.summary),
            'individual_risks': copy.deepcopy(self.individual_risks),
            'ml_results': copy.deepcopy(self.ml_results),
            'seir_results': copy.deepcopy(self.seir_results),
        }
        d.update(copy.deepcopy(self.extra))
        return d

    def iter_contact_risk_results(self) -> List[ContactRiskResult]:
        """将 potential_patients 展平为 ContactRiskResult 列表。

        供 integrator / validator 等消费者按 record_id 统一访问。
        """
        out: List[ContactRiskResult] = []
        for ct in ('family', 'social'):
            for idx, item in enumerate(self.potential_patients.get(ct, [])):
                out.append(ContactRiskResult.from_dict(
                    item, contact_type=ct, fallback_idx=idx))
        return out


__all__ = [
    'PatientRecord',
    'ContactRecord',
    'ContactRiskResult',
    'AssessmentResult',
]
