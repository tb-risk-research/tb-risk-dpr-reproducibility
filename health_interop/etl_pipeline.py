#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETL 集成管线 — 统一编排数据映射、术语编码、特征计算与增量更新

将模块二~模块五的所有组件整合为一条端到端管线：

    ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────────┐
    │ 原始数据  │ → │ 字段映射  │ → │ 术语编码   │ → │ 特征计算      │
    │ (HIS/CSV) │   │ (Mapping) │   │ (ICD10/…) │   │ (BCG/BMI/…) │
    └──────────┘   └──────────┘   └───────────┘   └──────┬───────┘
                                                          │
    ┌──────────┐   ┌───────────┐   ┌──────────────┐       │
    │ tb_risk  │ ← │ 增量引擎   │ ← │ 缺失值填补    │ ←──────┘
    │ 评分      │   │ (Event)   │   │ (Imputer)   │
    └──────────┘   └───────────┘   └──────────────┘

用法示例：
    from health_interop.etl_pipeline import ETLPipeline

    pipeline = ETLPipeline()
    pipeline.load_mapping_template("his_hospital_a.json")
    pipeline.initialize_terminology_mappers()

    result = pipeline.process_patient({
        "患者姓名": "张三",
        "年龄": "360",   # 月龄
        "性别": "男",
        "诊断": "肺结核",
    }, source="his")
    print(result.features, result.fill_records, result.warnings)
"""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .mapping import (
    FieldMappingEngine,
    UnitConverter,
    CodeMapper,
    MappingTemplateManager,
    ValueRangeNormalizer,
)
from .terminology import (
    ICD10Mapper,
    ICD9CMMapper,
    LOINCMapper,
    SNOMEDMapper,
    GBCodeMapper,
    DiagnosisMatcher,
)
from .feature_calculator import (
    FeatureCalculator,
    CalculationResult,
    MissingValueImputer,
    FillRecord,
)
from .incremental_engine import (
    IncrementalEngine,
    EventType,
    PatientEvent,
    PatientSnapshot,
    Notification,
    create_default_engine,
)

LOGGER = logging.getLogger("tb_risk.health_interop.etl_pipeline")


# ============================================================================
# 模板结构校验
# ============================================================================

_TEMPLATE_SCHEMA_REQUIRED = {"source_field", "target_field"}


def _validate_template_mappings(mappings: List[Dict[str, Any]]) -> List[str]:
    """校验模板映射规则的结构

    验证条件：
    - mappings 必须为列表
    - 每项包含非空的 source_field 和 target_field
    - transforms 如果存在必须为列表，且每个元素包含 type 字段

    返回：
        list[str]: 错误信息列表，为空表示校验通过
    """
    if not isinstance(mappings, list):
        return ["mappings 字段必须为列表"]

    errors = []
    for i, m in enumerate(mappings):
        if not isinstance(m, dict):
            errors.append(f"第 {i} 条映射规则不是字典")
            continue
        for required in _TEMPLATE_SCHEMA_REQUIRED:
            val = m.get(required)
            if not val or not isinstance(val, str) or not val.strip():
                errors.append(f"第 {i} 条映射规则缺少或无效的 '{required}' 字段")
        transforms = m.get("transforms")
        if transforms is not None:
            if not isinstance(transforms, list):
                errors.append(f"第 {i} 条映射规则的 transforms 必须为列表")
            else:
                for j, t in enumerate(transforms):
                    if not isinstance(t, dict):
                        errors.append(f"第 {i} 条映射规则的第 {j} 个 transform 不是字典")
                    elif not t.get("type"):
                        errors.append(f"第 {i} 条映射规则的第 {j} 个 transform 缺少 type 字段")
    return errors


# ============================================================================
# 管线配置
# ============================================================================

@dataclass
class ETLConfig:
    """ETL 管线配置"""
    validate_mapping: bool = True
    fill_strategy: str = "auto"
    auto_establish_baseline: bool = True
    auto_recalculate_risk: bool = True
    history_years: int = 5
    terminology_mapping_enabled: bool = True
    feature_calculation_enabled: bool = True
    incremental_enabled: bool = True
    record_provenance: bool = True
    output_format: str = "tb_risk"  # tb_risk | raw


# ============================================================================
# 数据来源记录
# ============================================================================

@dataclass
class DataSourceRecord:
    """数据来源记录（用于审计追踪）"""
    source: str                    # his / csv / fhir / hl7 / lis / pacs
    timestamp: str
    patient_id: str
    raw_fields: List[str]
    mapped_fields: List[str]
    terminology_matches: List[Dict[str, Any]] = field(default_factory=list)
    fill_records: List[FillRecord] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# 管线结果
# ============================================================================

@dataclass
class ETLResult:
    """ETL 管线处理结果"""
    success: bool
    patient_id: str
    features: Dict[str, Any] = field(default_factory=dict)
    mapped_data: Dict[str, Any] = field(default_factory=dict)
    calculation_result: Optional[CalculationResult] = None
    snapshot: Optional[PatientSnapshot] = None
    notification: Optional[Notification] = None
    provenance: Optional[DataSourceRecord] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


# ============================================================================
# ETL 管线主类
# ============================================================================

class ETLPipeline:
    """ETL 集成管线 — 统一编排数据映射、术语编码、特征计算与增量更新

    支持全量和增量两种处理模式：
    - 全量模式：一次性处理患者所有数据，适合首次接入
    - 增量模式：基于事件驱动，只处理变更数据，适合持续运行
    """

    def __init__(self, config: Optional[ETLConfig] = None):
        self.config = config or ETLConfig()

        # ---- 模块二：字段映射组件 ----
        self.mapping_engine = FieldMappingEngine()
        self.unit_converter = UnitConverter()
        self.code_mapper = CodeMapper()
        self.template_manager = MappingTemplateManager()
        self.value_normalizer = ValueRangeNormalizer()

        # ---- 模块三：术语编码组件 ----
        self.icd10_mapper: Optional[ICD10Mapper] = None
        self.icd9cm_mapper: Optional[ICD9CMMapper] = None
        self.loinc_mapper: Optional[LOINCMapper] = None
        self.snomed_mapper: Optional[SNOMEDMapper] = None
        self.gb_mapper: Optional[GBCodeMapper] = None
        self.diagnosis_matcher: Optional[DiagnosisMatcher] = None

        # ---- 模块三：特征计算 ----
        self.feature_calculator = FeatureCalculator()
        self.imputer = MissingValueImputer()

        # ---- 模块四：增量引擎 ----
        self.incremental_engine = create_default_engine()

        # ---- 内部状态 ----
        self._initialized = False
        self._patient_count = 0
        self._provenance_log: List[DataSourceRecord] = []

    # ====================================================================
    # 初始化
    # ====================================================================

    def initialize_terminology_mappers(self):
        """初始化所有术语编码映射器"""
        if not self.config.terminology_mapping_enabled:
            return
        self.icd10_mapper = ICD10Mapper()
        self.icd9cm_mapper = ICD9CMMapper()
        self.loinc_mapper = LOINCMapper()
        self.snomed_mapper = SNOMEDMapper()
        self.gb_mapper = GBCodeMapper()
        self.diagnosis_matcher = DiagnosisMatcher()
        LOGGER.info("术语编码映射器初始化完成")

    def load_mapping_template(self, template_id_or_path: str) -> int:
        """加载映射模板到引擎

        参数：
            template_id_or_path: 模板ID（已注册）或 JSON 文件路径

        返回：
            int: 加载的映射数量

        抛出：
            ValueError: 模板结构校验失败
        """
        # 尝试作为模板ID加载
        count = self.template_manager.apply_template(
            template_id_or_path, self.mapping_engine
        )
        if count > 0:
            LOGGER.info("从模板 '%s' 加载 %d 个映射", template_id_or_path, count)
            return count

        # 尝试作为文件路径加载
        try:
            with open(template_id_or_path, "r", encoding="utf-8") as f:
                template_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            LOGGER.warning("无法加载映射模板 '%s': %s", template_id_or_path, e)
            return 0

        # 结构校验
        mappings = template_data.get("mappings", [])
        validation_errors = _validate_template_mappings(mappings)
        if validation_errors:
            error_msg = f"模板 '{template_id_or_path}' 结构校验失败:\n" + "\n".join(validation_errors)
            LOGGER.error(error_msg)
            raise ValueError(error_msg)

        template_id = template_data.get("template_id", "imported")
        self.template_manager.save_template(template_id, template_data)
        count = self.template_manager.apply_template(template_id, self.mapping_engine)
        LOGGER.info("从文件 '%s' 加载 %d 个映射", template_id_or_path, count)
        return count

    def add_mappings(self, mappings: List[Dict[str, Any]]):
        """直接添加映射规则"""
        self.mapping_engine.add_mappings(mappings)

    def set_risk_calculator(self, calculator: Callable):
        """设置风险评分计算函数（注入 tb_risk 评分引擎）"""
        self.incremental_engine.set_risk_calculator(calculator)

    def set_notification_callback(self, callback: Callable):
        """设置通知回调函数"""
        self.incremental_engine.set_notification_callback(callback)

    def set_data_fetcher(self, fetcher: Callable):
        """设置历史数据拉取函数"""
        self.incremental_engine.set_data_fetcher(fetcher)

    # ====================================================================
    # 核心处理
    # ====================================================================

    def process_patient(
        self,
        raw_data: Dict[str, Any],
        *,
        patient_id: str = "",
        source: str = "his",
        incremental: bool = False,
        vaccination_records: Optional[List[Dict[str, Any]]] = None,
        scar_check_result: Optional[str] = None,
        clinical_notes: Optional[List[str]] = None,
        contact_records: Optional[List[Dict[str, Any]]] = None,
        report_card: Optional[Dict[str, Any]] = None,
        lab_results: Optional[List[Dict[str, Any]]] = None,
        imaging_reports: Optional[List[Dict[str, Any]]] = None,
        diagnoses: Optional[List[Dict[str, Any]]] = None,
    ) -> ETLResult:
        """处理单条患者数据（全量管线）

        参数：
            raw_data: 原始数据字典（医院系统字段名 → 值）
            patient_id: 患者ID（自动从 raw_data 提取时留空）
            source: 数据来源标识
            incremental: 是否启用增量模式
            vaccination_records: BCG接种记录
            scar_check_result: 卡痕检查结果
            clinical_notes: 病历文本列表
            contact_records: 接触者记录列表
            report_card: 传染病报告卡
            lab_results: 检验结果列表（用于触发增量事件）
            imaging_reports: 影像报告列表（用于触发增量事件）
            diagnoses: 诊断列表（用于触发增量事件）

        返回：
            ETLResult
        """
        start_time = datetime.datetime.now()
        result = ETLResult(success=True, patient_id="", errors=[], warnings=[])

        # ---- Step 0: 提取患者ID ----
        pid = patient_id or raw_data.get("patient_id") or raw_data.get("患者ID") or ""
        result.patient_id = pid

        # ---- Step 1: 字段映射 ----
        try:
            mapped = self.mapping_engine.apply(
                raw_data, validate=self.config.validate_mapping
            )
            result.mapped_data = mapped
            unmapped = self.mapping_engine._unmapped_source
            if unmapped:
                result.warnings.append(f"未映射字段: {', '.join(unmapped[:10])}")
        except Exception as e:
            result.errors.append(f"字段映射失败: {e}")
            result.success = False
            return result

        # ---- Step 2: 术语编码映射 ----
        terminology_matches = []
        if self.config.terminology_mapping_enabled:
            try:
                # 2a. 诊断文本映射（ICD-10）— 独立检查 diagnosis_matcher
                if self.diagnosis_matcher:
                    raw_diagnosis = (
                        raw_data.get("诊断") or raw_data.get("diagnosis") or
                        raw_data.get("diagnoses") or raw_data.get("疾病诊断") or ""
                    )
                    diagnosis_text = raw_diagnosis or mapped.get("symptom_description", "")
                    if diagnosis_text:
                        matches = self.diagnosis_matcher.match(diagnosis_text)
                        terminology_matches.extend(matches)
                        for m in matches:
                            code = m.get("code", "")
                            if code.startswith("A1") or code.startswith("B90"):
                                mapped["active_tb"] = "1"
                                mapped["diagnosis_code"] = code
                            elif code.startswith("E1"):
                                mapped["comorbidity_diabetes"] = "1"
                            elif code.startswith("B2"):
                                mapped["comorbidity_hiv"] = "1"
                            elif code.startswith("J6"):
                                mapped["comorbidity_copd"] = "1"

                        # 补充 SNOMED CT 编码输出
                        if self.snomed_mapper:
                            for m in matches:
                                code = m.get("code", "")
                                display = m.get("display", "")
                                snomed = self.snomed_mapper.match(display)
                                if snomed:
                                    m["snomed_code"] = snomed.get("code", "")
                                    mapped["diagnosis_snomed_ct"] = snomed.get("code", "")

                # 2b. ICD-9-CM-3 手术操作编码映射 — 独立检查 icd9cm_mapper
                if self.icd9cm_mapper:
                    surgery_text = (
                        raw_data.get("手术") or raw_data.get("手术操作") or
                        raw_data.get("surgery") or raw_data.get("procedure") or ""
                    )
                    if surgery_text:
                        procedure_match = self.icd9cm_mapper.match(surgery_text)
                        if procedure_match:
                            mapped["procedure_code"] = procedure_match.get("code", "")
                            mapped["procedure_type"] = procedure_match.get("type", "")
                            terminology_matches.append({
                                "type": "icd9cm",
                                "original": surgery_text,
                                "standard": procedure_match.get("code", ""),
                                "display": procedure_match.get("display", ""),
                                "confidence": procedure_match.get("confidence", 0.5),
                            })

                # 2c. 国标编码映射 — 独立检查 gb_mapper
                if self.gb_mapper:
                    ethnicity = mapped.get("ethnicity") or raw_data.get("民族") or ""
                    if ethnicity:
                        match = self.gb_mapper.match_ethnicity(ethnicity)
                        if match:
                            mapped["ethnicity"] = match["code"]

                    occupation = mapped.get("occupation") or raw_data.get("职业") or ""
                    if occupation:
                        match = self.gb_mapper.match_occupation(occupation)
                        if match:
                            mapped["occupation"] = match["code"]
                            if self.gb_mapper.is_high_risk_occupation(occupation):
                                mapped["occupation_high_risk"] = "1"

                # 2d. LOINC 检验映射 — 独立检查 loinc_mapper
                if self.loinc_mapper and lab_results:
                    for lab in lab_results:
                        test_name = lab.get("test_name", "")
                        match = self.loinc_mapper.match_test_name(test_name)
                        if match:
                            lab["loinc_code"] = match.get("loinc", "")
                            terminology_matches.append({
                                "type": "loinc",
                                "original": test_name,
                                "standard": match.get("loinc", ""),
                                "display": match.get("name", ""),
                                "confidence": match.get("confidence", 0.5),
                            })

            except Exception as e:
                result.warnings.append(f"术语编码映射异常: {e}")

        # ---- Step 3: 特征计算 ----
        calculation_result = None
        if self.config.feature_calculation_enabled:
            try:
                calculation_result = self.feature_calculator.calculate_all(
                    patient_data=mapped,
                    vaccination_records=vaccination_records,
                    scar_check_result=scar_check_result,
                    clinical_notes=clinical_notes,
                    contact_records=contact_records,
                    report_card=report_card,
                    fill_strategy=self.config.fill_strategy,
                )
                result.features = calculation_result.features
                result.calculation_result = calculation_result
                result.warnings.extend(calculation_result.warnings)
            except Exception as e:
                result.errors.append(f"特征计算失败: {e}")
                result.features = mapped
        else:
            result.features = mapped

        # ---- Step 4: 增量更新（可选） ----
        snapshot = None
        notification = None
        if incremental and pid:
            try:
                # 首次就诊建立基线；后续调用走增量更新，避免 establish_baseline
                # 在基线已存在时提前返回而丢失新特征与计数。
                existing = self.incremental_engine.get_snapshot(pid)
                baseline_done = existing is not None and existing.baseline_established
                if self.config.auto_establish_baseline and not baseline_done:
                    snapshot = self.incremental_engine.establish_baseline(
                        pid, result.features
                    )
                else:
                    snapshot = self.incremental_engine.incremental_update(
                        pid, result.features, source=source
                    )

                # 触发增量事件
                if lab_results:
                    for lab in lab_results:
                        self.incremental_engine.on_lab_result(pid, lab)
                if imaging_reports:
                    for img in imaging_reports:
                        self.incremental_engine.on_imaging_report(pid, img)
                if diagnoses:
                    for dx in diagnoses:
                        self.incremental_engine.on_diagnosis_update(pid, dx)

                # 获取待处理通知
                pending = self.incremental_engine.get_pending_notifications(pid)
                if pending:
                    notification = pending[-1]

                result.snapshot = snapshot
                result.notification = notification
            except Exception as e:
                result.warnings.append(f"增量更新异常: {e}")

        # ---- Step 5: 审计追踪 ----
        if self.config.record_provenance:
            prov = DataSourceRecord(
                source=source,
                timestamp=datetime.datetime.now().isoformat(),
                patient_id=pid,
                raw_fields=list(raw_data.keys()),
                mapped_fields=list(result.features.keys()),
                terminology_matches=terminology_matches,
                fill_records=(
                    calculation_result.fill_records
                    if calculation_result else []
                ),
                errors=list(result.errors),
                warnings=list(result.warnings),
            )
            result.provenance = prov
            self._provenance_log.append(prov)

        # ---- 统计 ----
        self._patient_count += 1
        elapsed = (datetime.datetime.now() - start_time).total_seconds() * 1000
        result.processing_time_ms = round(elapsed, 2)

        LOGGER.info(
            "ETL管线处理完成: patient=%s source=%s features=%d errors=%d time=%.0fms",
            pid or "(unknown)", source, len(result.features),
            len(result.errors), result.processing_time_ms,
        )
        return result

    # ====================================================================
    # 批量处理
    # ====================================================================

    def process_batch(
        self,
        records: List[Dict[str, Any]],
        *,
        source: str = "csv",
        patient_id_field: str = "patient_id",
        batch_size: int = 100,
    ) -> List[ETLResult]:
        """批量处理患者数据

        参数：
            records: 原始数据记录列表
            source: 数据来源
            patient_id_field: 患者ID字段名
            batch_size: 每批处理数量

        返回：
            list[ETLResult]
        """
        results = []
        total = len(records)

        for i in range(0, total, batch_size):
            batch = records[i: i + batch_size]
            for j, record in enumerate(batch):
                pid = str(record.get(patient_id_field, f"batch_{i + j}"))
                try:
                    etl_result = self.process_patient(
                        record, patient_id=pid, source=source
                    )
                    results.append(etl_result)
                except Exception as e:
                    results.append(ETLResult(
                        success=False,
                        patient_id=pid,
                        errors=[f"批量处理异常: {e}"],
                    ))

            LOGGER.info(
                "ETL批量处理: %d/%d (%.1f%%)",
                min(i + batch_size, total), total,
                min(i + batch_size, total) / total * 100,
            )

        LOGGER.info(
            "ETL批量处理完成: total=%d success=%d fail=%d",
            total, sum(1 for r in results if r.success),
            sum(1 for r in results if not r.success),
        )
        return results

    # ====================================================================
    # 事件驱动接口（LIS/PACS/HIS 实时触发）
    # ====================================================================

    def on_lab_result_received(
        self,
        patient_id: str,
        lab_result: Dict[str, Any],
        patient_data: Optional[Dict[str, Any]] = None,
    ) -> ETLResult:
        """LIS 检验结果到达时处理

        自动触发字段映射、术语编码、增量更新。
        """
        # 如果有患者数据，先做全量映射
        mapped = {}
        if patient_data:
            try:
                mapped = self.mapping_engine.apply(patient_data)
            except Exception as e:
                LOGGER.warning("检验事件映射异常: %s", e)

        # LOINC 编码
        if self.loinc_mapper:
            test_name = lab_result.get("test_name", "")
            match = self.loinc_mapper.match_test_name(test_name)
            if match:
                mapped["loinc_code"] = match.get("loinc", "")
                mapped["loinc_display"] = match.get("name", "")
                lab_result["loinc_code"] = match.get("loinc", "")

        # 触发增量引擎
        event = self.incremental_engine.on_lab_result(patient_id, lab_result)

        # 等待异步事件处理完成（最多 0.5 秒）
        if event:
            import time as _time
            for _ in range(50):
                if event.processed:
                    break
                _time.sleep(0.01)

        # 获取通知
        pending = self.incremental_engine.get_pending_notifications(patient_id)

        return ETLResult(
            success=True,
            patient_id=patient_id,
            features=mapped,
            notification=pending[-1] if pending else None,
            processing_time_ms=0.0,
        )

    def on_imaging_report_received(
        self,
        patient_id: str,
        imaging_report: Dict[str, Any],
        patient_data: Optional[Dict[str, Any]] = None,
    ) -> ETLResult:
        """PACS 影像报告到达时处理"""
        mapped = {}
        if patient_data:
            try:
                mapped = self.mapping_engine.apply(patient_data)
            except Exception as e:
                LOGGER.warning("影像事件映射异常: %s", e)

        event = self.incremental_engine.on_imaging_report(patient_id, imaging_report)
        if event:
            import time as _time
            for _ in range(50):
                if event.processed:
                    break
                _time.sleep(0.01)
        pending = self.incremental_engine.get_pending_notifications(patient_id)

        return ETLResult(
            success=True,
            patient_id=patient_id,
            features=mapped,
            notification=pending[-1] if pending else None,
            processing_time_ms=0.0,
        )

    def on_diagnosis_updated(
        self,
        patient_id: str,
        diagnosis: Dict[str, Any],
        patient_data: Optional[Dict[str, Any]] = None,
    ) -> ETLResult:
        """诊断更新时处理"""
        mapped = {}
        if patient_data:
            try:
                mapped = self.mapping_engine.apply(patient_data)
            except Exception as e:
                LOGGER.warning("诊断事件映射异常: %s", e)

        # ICD-10 编码
        if self.icd10_mapper:
            diagnosis_name = diagnosis.get("name", diagnosis.get("diagnosis", ""))
            match = self.icd10_mapper.match_diagnosis(diagnosis_name)
            if match:
                diagnosis["icd10_code"] = match.get("code", "")
                diagnosis["icd10_display"] = match.get("display", "")

        events = self.incremental_engine.on_diagnosis_update(patient_id, diagnosis)
        if events:
            import time as _time
            for _ in range(50):
                if all(getattr(e, 'processed', True) for e in events if e):
                    break
                _time.sleep(0.01)
        pending = self.incremental_engine.get_pending_notifications(patient_id)

        return ETLResult(
            success=True,
            patient_id=patient_id,
            features=mapped,
            notification=pending[-1] if pending else None,
            processing_time_ms=0.0,
        )

    # ====================================================================
    # 查询与统计
    # ====================================================================

    def get_patient_features(self, patient_id: str) -> Dict[str, Any]:
        """获取患者当前特征"""
        return self.incremental_engine.get_all_features(patient_id)

    def get_patient_snapshot(self, patient_id: str) -> Optional[PatientSnapshot]:
        """获取患者快照"""
        return self.incremental_engine.get_snapshot(patient_id)

    def get_pending_notifications(self, patient_id: str = "") -> List[Dict[str, Any]]:
        """获取待处理通知"""
        return self.incremental_engine.get_pending_notifications(patient_id)

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """获取管线统计"""
        eng_stats = self.incremental_engine.get_statistics()
        return {
            "total_patients_processed": self._patient_count,
            "total_provenance_records": len(self._provenance_log),
            "mapping_rules": len(self.mapping_engine.get_all_mappings()),
            "templates_available": len(self.template_manager.list_templates()),
            "incremental_engine": eng_stats,
            "terminology_initialized": self.diagnosis_matcher is not None,
        }

    def get_fill_report(self) -> Dict[str, Any]:
        """获取缺失值填补报告（来自特征计算器的填补记录）"""
        return self.feature_calculator.imputer.get_fill_report()

    def get_provenance_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取审计追踪日志"""
        return [
            {
                "source": r.source,
                "timestamp": r.timestamp,
                "patient_id": r.patient_id,
                "raw_fields": len(r.raw_fields),
                "mapped_fields": len(r.mapped_fields),
                "terminology_matches": len(r.terminology_matches),
                "fill_count": len(r.fill_records),
                "errors": len(r.errors),
                "warnings": len(r.warnings),
            }
            for r in self._provenance_log[-limit:]
        ]

    # ====================================================================
    # 持久化
    # ====================================================================

    def export_features_to_json(
        self, patient_id: str, output_path: str
    ) -> bool:
        """导出患者特征到 JSON 文件

        tb_risk 评分引擎可直接读取此格式。
        """
        features = self.incremental_engine.get_all_features(patient_id)
        if not features:
            LOGGER.warning("患者 '%s' 无特征数据，跳过导出", patient_id)
            return False
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(features, f, ensure_ascii=False, indent=2)
            LOGGER.info("特征导出成功: %s -> %s", patient_id, output_path)
            return True
        except Exception as e:
            LOGGER.error("特征导出失败: %s", e)
            return False

    def export_all_features(self, output_dir: str) -> int:
        """导出所有患者特征到目录"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        count = 0
        for patient_id in self.incremental_engine._snapshots:
            path = os.path.join(output_dir, f"{patient_id}_features.json")
            if self.export_features_to_json(patient_id, path):
                count += 1
        return count

    def save_state(self, filepath: str) -> bool:
        """保存管线状态到 JSON 文件"""
        state = {
            "saved_at": datetime.datetime.now().isoformat(),
            "patient_count": self._patient_count,
            "mappings": self.mapping_engine.to_dict(),
            "templates": self.template_manager.list_templates(),
            "snapshots": {
                pid: {
                    "baseline_established": s.baseline_established,
                    "baseline_time": s.baseline_time.isoformat() if s.baseline_time else None,
                    "last_update_time": s.last_update_time.isoformat() if s.last_update_time else None,
                    "last_risk_score": s.last_risk_score,
                    "last_risk_level": s.last_risk_level,
                    "update_count": s.update_count,
                    "feature_count": len(s.features),
                }
                for pid, s in self.incremental_engine._snapshots.items()
            },
        }
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            LOGGER.info("管线状态已保存: %s", filepath)
            return True
        except Exception as e:
            LOGGER.error("保存管线状态失败: %s", e)
            return False

    def reset(self):
        """重置管线状态"""
        self.mapping_engine.clear()
        self.incremental_engine.reset()
        self.feature_calculator = FeatureCalculator()
        self.imputer = MissingValueImputer()
        self._patient_count = 0
        self._provenance_log.clear()
        LOGGER.info("ETL管线已重置")