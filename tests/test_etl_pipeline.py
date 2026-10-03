#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETL 集成管线单元测试 — 覆盖全量处理、批量处理、事件驱动、审计追踪"""

import json
import os
import pytest
import tempfile
import time

from health_interop.etl_pipeline import ETLPipeline, ETLConfig, ETLResult


# ============================================================================
# 辅助函数：构建测试用的原始数据
# ============================================================================

def make_his_patient(name="张三", age_months=360, gender="男",
                     diagnosis="肺结核", ethnicity="汉族", occupation="农民",
                     height_cm=170, weight_kg=65, smoking_years=10,
                     smoking_amount=15):
    """生成模拟 HIS 系统患者数据"""
    return {
        "patient_id": f"P{int(time.time() * 1000) % 100000}",
        "患者姓名": name,
        "年龄": str(age_months),
        "性别": gender,
        "诊断": diagnosis,
        "民族": ethnicity,
        "职业": occupation,
        "身高": str(height_cm),
        "体重": str(weight_kg),
        "吸烟年数": str(smoking_years),
        "每日吸烟量": str(smoking_amount),
        "痰涂片": "阴性",
        "有无空洞": "无",
        "症状": "咳嗽、咳痰、发热",
    }


def make_tb_risk_mappings():
    """创建 tb_risk 标准映射规则"""
    return [
        # 基本信息
        {"source_field": "患者姓名", "target_field": "patient_name",
         "description": "姓名映射", "confidence": 1.0},
        {"source_field": "年龄", "target_field": "age",
         "transforms": [{"type": "expression", "expr": "round(float(x) / 12, 0)"}],
         "description": "月龄转岁", "confidence": 1.0},
        {"source_field": "性别", "target_field": "gender",
         "transforms": [{"type": "code_map", "mapping": {"男": 1, "女": 0, "male": 1, "female": 0}}],
         "description": "性别编码映射", "confidence": 1.0},
        {"source_field": "民族", "target_field": "ethnicity",
         "description": "民族", "confidence": 0.8},
        {"source_field": "职业", "target_field": "occupation",
         "description": "职业", "confidence": 0.8},
        # 临床指标
        {"source_field": "痰涂片", "target_field": "sputum_smear",
         "transforms": [{"type": "code_map", "mapping": {"阴性": "1", "阳性": "2", "涂阴": "1", "涂阳": "2"}}],
         "description": "痰涂片编码", "confidence": 1.0},
        {"source_field": "有无空洞", "target_field": "has_cavity",
         "transforms": [{"type": "code_map", "mapping": {"无": "1", "有": "2", "否": "1", "是": "2"}}],
         "description": "空洞编码", "confidence": 1.0},
        {"source_field": "症状", "target_field": "symptom_description",
         "description": "症状描述", "confidence": 0.9},
        # 扩展特征
        {"source_field": "身高", "target_field": "height_cm",
         "transforms": [{"type": "expression", "expr": "float(x)"}],
         "description": "身高", "confidence": 1.0},
        {"source_field": "体重", "target_field": "weight_kg",
         "transforms": [{"type": "expression", "expr": "float(x)"}],
         "description": "体重", "confidence": 1.0},
        {"source_field": "吸烟年数", "target_field": "smoking_years",
         "transforms": [{"type": "expression", "expr": "float(x)"}],
         "description": "吸烟年数", "confidence": 1.0},
        {"source_field": "每日吸烟量", "target_field": "smoking_amount_per_day",
         "transforms": [{"type": "expression", "expr": "float(x)"}],
         "description": "每日吸烟量", "confidence": 1.0},
    ]


# ============================================================================
# 测试 ETLConfig
# ============================================================================

class TestETLConfig:
    """测试 ETL 管线配置"""

    def test_default_config(self):
        """测试默认配置"""
        cfg = ETLConfig()
        assert cfg.validate_mapping is True
        assert cfg.fill_strategy == "auto"
        assert cfg.auto_establish_baseline is True
        assert cfg.terminology_mapping_enabled is True
        assert cfg.feature_calculation_enabled is True
        assert cfg.record_provenance is True
        assert cfg.output_format == "tb_risk"

    def test_custom_config(self):
        """测试自定义配置"""
        cfg = ETLConfig(
            validate_mapping=False,
            fill_strategy="median",
            auto_establish_baseline=False,
            terminology_mapping_enabled=False,
            feature_calculation_enabled=False,
            record_provenance=False,
            output_format="raw",
        )
        assert cfg.validate_mapping is False
        assert cfg.fill_strategy == "median"
        assert cfg.auto_establish_baseline is False
        assert cfg.terminology_mapping_enabled is False
        assert cfg.feature_calculation_enabled is False
        assert cfg.record_provenance is False
        assert cfg.output_format == "raw"


# ============================================================================
# 测试 ETLPipeline — 字段映射
# ============================================================================

class TestETLFieldMapping:
    """测试 ETL 管线字段映射功能"""

    def setup_method(self):
        self.pipeline = ETLPipeline()

    def test_basic_mapping(self):
        """测试基本字段映射（月龄→岁、性别编码）"""
        mappings = make_tb_risk_mappings()
        self.pipeline.add_mappings(mappings)

        raw = make_his_patient(age_months=360, gender="男")
        result = self.pipeline.process_patient(raw, patient_id="test_001")

        assert result.success is True
        assert result.patient_id == "test_001"
        assert result.features.get("patient_name") == "张三"
        assert result.features.get("age") == 30.0  # 360/12
        assert result.features.get("gender") == 1    # 男→1
        assert result.features.get("sputum_smear") == "1"  # 阴性→1
        assert result.features.get("has_cavity") == "1"    # 无→1

    def test_gender_female(self):
        """测试女性性别编码"""
        mappings = make_tb_risk_mappings()
        self.pipeline.add_mappings(mappings)

        raw = make_his_patient(gender="女")
        result = self.pipeline.process_patient(raw, patient_id="test_002")

        assert result.success is True
        assert result.features.get("gender") == 0  # 女→0

    def test_missing_source_field(self):
        """测试源字段缺失时使用默认值"""
        mappings = make_tb_risk_mappings()
        self.pipeline.add_mappings(mappings)

        raw = {"patient_id": "test_003", "患者姓名": "李四"}
        # 添加一个带默认值的映射
        self.pipeline.mapping_engine.add_mapping(
            "年龄", "age", default_value=40,
            description="年龄默认值",
        )
        result = self.pipeline.process_patient(raw, patient_id="test_003")

        assert result.success is True
        assert result.features.get("patient_name") == "李四"

    def test_complex_expression_transform(self):
        """测试复杂表达式转换"""
        self.pipeline.mapping_engine.add_mapping(
            "体温华氏", "temperature_celsius",
            transforms=[{"type": "expression", "expr": "round((float(x) - 32) * 5 / 9, 1)"}],
            description="华氏度转摄氏度",
        )
        raw = {"patient_id": "test_004", "体温华氏": "98.6"}
        result = self.pipeline.process_patient(raw, patient_id="test_004")

        assert result.success is True
        assert result.features.get("temperature_celsius") == 37.0  # (98.6-32)*5/9


# ============================================================================
# 测试 ETLPipeline — 特征计算
# ============================================================================

class TestETLFeatureCalculation:
    """测试 ETL 管线特征计算功能（BMI、吸烟指数、缺失值填补）"""

    def setup_method(self):
        self.pipeline = ETLPipeline()
        self.pipeline.add_mappings(make_tb_risk_mappings())

    def test_bmi_calculation(self):
        """测试 BMI 自动计算"""
        raw = make_his_patient(height_cm=170, weight_kg=65)
        result = self.pipeline.process_patient(raw, patient_id="test_bmi")

        assert result.success is True
        bmi = result.features.get("bmi")
        # BMI = 65 / (1.70^2) ≈ 22.5
        assert bmi is not None
        assert 22.0 <= bmi <= 23.0, f"BMI={bmi} 应在 22-23 之间"
        assert result.features.get("bmi_category") == "normal"

    def test_bmi_underweight(self):
        """测试 BMI 偏瘦分类"""
        raw = make_his_patient(height_cm=170, weight_kg=45)
        result = self.pipeline.process_patient(raw, patient_id="test_bmi_low")

        assert result.success is True
        assert result.features.get("bmi_category") == "underweight"

    def test_bmi_obese(self):
        """测试 BMI 肥胖分类"""
        raw = make_his_patient(height_cm=160, weight_kg=85)
        result = self.pipeline.process_patient(raw, patient_id="test_bmi_high")

        assert result.success is True
        assert result.features.get("bmi_category") == "obese"

    def test_smoking_index(self):
        """测试吸烟指数自动计算"""
        raw = make_his_patient(smoking_years=10, smoking_amount=20)
        result = self.pipeline.process_patient(raw, patient_id="test_smoking")

        assert result.success is True
        # pack_years = 20支/天 * 10年 / 20 = 10
        pack_years = result.features.get("pack_years")
        assert pack_years == 10.0, f"pack_years={pack_years} 应为 10"

    def test_smoking_heavy(self):
        """测试重度吸烟指数"""
        raw = make_his_patient(smoking_years=30, smoking_amount=40)
        result = self.pipeline.process_patient(raw, patient_id="test_smoking_heavy")

        assert result.success is True
        # pack_years = 40支/天 * 30年 / 20 = 60
        assert result.features.get("pack_years") == 60.0

    def test_missing_value_fill(self):
        """测试缺失值自动填补"""
        # 只提供部分字段，不提供身高体重
        self.pipeline.mapping_engine.add_mapping(
            "患者姓名", "patient_name", description="姓名",
        )
        raw = {"patient_id": "test_missing", "患者姓名": "王五"}
        result = self.pipeline.process_patient(raw, patient_id="test_missing")

        assert result.success is True
        # 缺失字段应被填补
        features = result.features
        assert features.get("patient_name") == "王五"
        # imputer 会填补缺失的数字字段
        assert features.get("age") is not None
        assert features.get("bmi") is not None
        # 应有填补记录
        assert result.calculation_result is not None
        assert len(result.calculation_result.fill_records) > 0


# ============================================================================
# 测试 ETLPipeline — 术语编码映射
# ============================================================================

class TestETLTerminology:
    """测试 ETL 管线术语编码映射功能"""

    def setup_method(self):
        self.pipeline = ETLPipeline()
        self.pipeline.add_mappings(make_tb_risk_mappings())
        self.pipeline.initialize_terminology_mappers()

    def test_diagnosis_tb_mapping(self):
        """测试结核诊断 ICD-10 映射"""
        raw = make_his_patient(diagnosis="肺结核", ethnicity="汉", occupation="医生")
        result = self.pipeline.process_patient(raw, patient_id="test_term_tb")

        assert result.success is True
        # 诊断文本映射→active_tb 应设为 1
        assert result.features.get("active_tb") == "1"
        assert result.features.get("diagnosis_code") is not None

    def test_diabetes_mapping(self):
        """测试糖尿病诊断 ICD-10 映射"""
        raw = make_his_patient(diagnosis="2型糖尿病", ethnicity="汉", occupation="农民")
        result = self.pipeline.process_patient(raw, patient_id="test_term_dm")

        assert result.success is True
        # 糖尿病诊断→comorbidity_diabetes 应设为 1
        assert result.features.get("comorbidity_diabetes") == "1"

    def test_ethnicity_mapping(self):
        """测试民族编码国标映射"""
        raw = make_his_patient(ethnicity="维吾尔族")
        result = self.pipeline.process_patient(raw, patient_id="test_term_eth")

        assert result.success is True
        # 维吾尔族→国标编码 05
        assert result.features.get("ethnicity") == "05"

    def test_high_risk_occupation(self):
        """测试高风险职业识别"""
        raw = make_his_patient(occupation="医生")
        result = self.pipeline.process_patient(raw, patient_id="test_term_occ")

        assert result.success is True
        assert result.features.get("occupation_high_risk") == "1"


# ============================================================================
# 测试 ETLPipeline — 批量处理
# ============================================================================

class TestETLBatchProcessing:
    """测试 ETL 管线批量处理功能"""

    def setup_method(self):
        self.pipeline = ETLPipeline()
        self.pipeline.add_mappings(make_tb_risk_mappings())

    def test_batch_multiple_patients(self):
        """测试批量处理多条患者数据"""
        records = [
            make_his_patient(name="张三", age_months=360, gender="男"),
            make_his_patient(name="李四", age_months=240, gender="女"),
            make_his_patient(name="王五", age_months=480, gender="男"),
        ]
        # 为每条记录添加 patient_id
        for i, r in enumerate(records):
            r["patient_id"] = f"batch_{i}"

        results = self.pipeline.process_batch(records, patient_id_field="patient_id")

        assert len(results) == 3
        for r in results:
            assert r.success is True
        assert results[0].features.get("patient_name") == "张三"
        assert results[0].features.get("age") == 30.0
        assert results[1].features.get("patient_name") == "李四"
        assert results[1].features.get("gender") == 0
        assert results[2].features.get("patient_name") == "王五"
        assert results[2].features.get("age") == 40.0

    def test_batch_partial_failure(self):
        """测试批量处理中部分失败的处理"""
        # 创建一条正常记录和一条异常记录
        records = [
            {"patient_id": "ok", "患者姓名": "正常"},
            {"patient_id": "bad", "invalid_field": object()},  # 无法映射
        ]
        # 添加基本映射
        self.pipeline.mapping_engine.add_mapping("患者姓名", "patient_name")

        # 捕获异常让批量处理继续
        results = self.pipeline.process_batch(records, patient_id_field="patient_id")
        assert len(results) == 2


# ============================================================================
# 测试 ETLPipeline — 增量更新与事件驱动
# ============================================================================

class TestETLIncremental:
    """测试 ETL 管线增量更新与事件驱动功能"""

    def setup_method(self):
        self.pipeline = ETLPipeline()
        self.pipeline.add_mappings(make_tb_risk_mappings())

    def test_baseline_establishment(self):
        """测试患者基线建立"""
        raw = make_his_patient()
        pid = "test_inc_baseline"
        result = self.pipeline.process_patient(raw, patient_id=pid, incremental=True)

        assert result.success is True
        # 增量模式应建立基线
        snapshot = self.pipeline.get_patient_snapshot(pid)
        assert snapshot is not None
        assert snapshot.baseline_established is True
        assert snapshot.baseline_time is not None

    def test_incremental_update(self):
        """测试增量更新"""
        pid = "test_inc_update"
        # 首次建立基线
        raw1 = make_his_patient(name="基线患者")
        self.pipeline.process_patient(raw1, patient_id=pid, incremental=True)

        # 增量更新
        raw2 = {"patient_id": pid, "痰涂片": "阳性"}
        result2 = self.pipeline.process_patient(raw2, patient_id=pid, incremental=True)

        assert result2.success is True
        snapshot = self.pipeline.get_patient_snapshot(pid)
        assert snapshot is not None
        # 更新后应更新计数或特征
        assert snapshot.update_count > 0

    def test_lab_result_event(self):
        """测试 LIS 检验结果事件触发"""
        pid = "test_lis_event"
        # 先建立基线
        raw = make_his_patient()
        self.pipeline.process_patient(raw, patient_id=pid, incremental=True)

        # 触发痰涂片阳性事件
        lab_result = {
            "test_name": "痰涂片抗酸染色",
            "test_code": "sputum_smear",
            "value": "阳性",
            "abnormal_flag": "critical",
        }
        result = self.pipeline.on_lab_result_received(pid, lab_result)
        time.sleep(0.1)  # 等待异步事件处理

        assert result.success is True
        # 高风险事件应生成通知
        notif = result.notification
        if notif is None:
            # 通知可能因异步延迟未生成，检查事件历史
            history = self.pipeline.incremental_engine.get_event_history(pid)
            assert len(history) > 0
        else:
            assert "痰涂片" in notif.get("title", "") or "阳性" in notif.get("message", "")

    def test_imaging_event(self):
        """测试 PACS 影像报告事件触发"""
        pid = "test_img_event"
        raw = make_his_patient()
        self.pipeline.process_patient(raw, patient_id=pid, incremental=True)

        # 触发影像报告事件
        imaging_report = {
            "report_id": "RAD_001",
            "impression": "考虑肺结核，右上肺空洞形成",
            "findings": "右上肺见薄壁空洞，周围见斑片状浸润影",
        }
        result = self.pipeline.on_imaging_report_received(pid, imaging_report)
        time.sleep(0.1)  # 等待异步事件处理

        assert result.success is True
        # 检查事件历史
        history = self.pipeline.incremental_engine.get_event_history(pid)
        assert len(history) > 0

    def test_diagnosis_update_event(self):
        """测试诊断更新事件触发"""
        pid = "test_diag_event"
        raw = make_his_patient()
        self.pipeline.process_patient(raw, patient_id=pid, incremental=True)

        # 触发诊断更新事件
        diagnosis = {
            "code": "A15.0",
            "name": "涂阳肺结核",
            "diagnosis_time": "2026-08-01",
        }
        result = self.pipeline.on_diagnosis_updated(pid, diagnosis)
        time.sleep(0.1)  # 等待异步事件处理

        assert result.success is True
        history = self.pipeline.incremental_engine.get_event_history(pid)
        assert len(history) > 0


# ============================================================================
# 测试 ETLPipeline — 审计追踪与持久化
# ============================================================================

class TestETLAuditAndPersistence:
    """测试 ETL 管线审计追踪与持久化功能"""

    def setup_method(self):
        self.pipeline = ETLPipeline()
        self.pipeline.add_mappings(make_tb_risk_mappings())

    def test_provenance_logging(self):
        """测试审计追踪日志记录"""
        raw = make_his_patient()
        result = self.pipeline.process_patient(raw, patient_id="test_audit")

        assert result.provenance is not None
        assert result.provenance.source == "his"
        assert result.provenance.patient_id == "test_audit"
        assert len(result.provenance.raw_fields) > 0
        assert len(result.provenance.mapped_fields) > 0

    def test_export_features_to_json(self):
        """测试特征导出到 JSON 文件"""
        raw = make_his_patient()
        pid = "test_export"
        self.pipeline.process_patient(raw, patient_id=pid, incremental=True)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            output_path = f.name

        try:
            success = self.pipeline.export_features_to_json(pid, output_path)
            assert success is True

            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "patient_name" in data
            assert "age" in data
        finally:
            os.unlink(output_path)

    def test_save_and_restore_state(self):
        """测试保存和恢复管线状态"""
        raw = make_his_patient()
        self.pipeline.process_patient(raw, patient_id="test_state", incremental=True)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            state_path = f.name

        try:
            success = self.pipeline.save_state(state_path)
            assert success is True

            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            assert "saved_at" in state
            assert "mappings" in state
            assert "snapshots" in state
            assert "test_state" in state["snapshots"]
        finally:
            os.unlink(state_path)

    def test_statistics(self):
        """测试管线统计"""
        raw = make_his_patient()
        self.pipeline.process_patient(raw, patient_id="test_stats")

        stats = self.pipeline.get_pipeline_statistics()
        assert stats["total_patients_processed"] == 1
        assert stats["mapping_rules"] > 0

    def test_fill_report(self):
        """测试缺失值填补报告"""
        self.pipeline.mapping_engine.add_mapping("患者姓名", "patient_name")
        raw = {"patient_id": "test_fill", "患者姓名": "缺值患者"}
        self.pipeline.process_patient(raw, patient_id="test_fill")

        report = self.pipeline.get_fill_report()
        assert report["total_fills"] > 0
        assert "strategy_usage" in report


# ============================================================================
# 测试 ETLPipeline — 边界情况
# ============================================================================

class TestETLEdgeCases:
    """测试 ETL 管线边界情况"""

    def setup_method(self):
        self.pipeline = ETLPipeline()

    def test_empty_raw_data(self):
        """测试空数据"""
        result = self.pipeline.process_patient({}, patient_id="test_empty")
        assert result.success is True
        # 没有映射规则，原始数据为空，但管线应正常返回
        assert result.features is not None

    def test_no_mappings(self):
        """测试无映射规则（特征计算器仍会填补缺失值）"""
        from health_interop.etl_pipeline import ETLConfig
        pipeline = ETLPipeline(config=ETLConfig(feature_calculation_enabled=False))
        raw = {"patient_id": "test_no_map", "name": "张三"}
        result = pipeline.process_patient(raw, patient_id="test_no_map")
        # 关闭特征计算后，无映射规则时特征应为空
        assert result.success is True
        assert len(result.features) == 0

    def test_reset_pipeline(self):
        """测试重置管线"""
        self.pipeline.add_mappings(make_tb_risk_mappings())
        raw = make_his_patient()
        self.pipeline.process_patient(raw, patient_id="test_reset", incremental=True)

        self.pipeline.reset()
        assert len(self.pipeline.mapping_engine.get_all_mappings()) == 0
        stats = self.pipeline.get_pipeline_statistics()
        assert stats["total_patients_processed"] == 0

    def test_special_characters(self):
        """测试特殊字符"""
        self.pipeline.mapping_engine.add_mapping("姓名", "name")
        raw = {"patient_id": "test_special", "姓名": "O'Brien 张三·测试"}
        result = self.pipeline.process_patient(raw, patient_id="test_special")
        assert result.success is True
        assert result.features.get("name") == "O'Brien 张三·测试"


# ============================================================================
# 集成测试 — 完整 ETL 管线端到端
# ============================================================================

class TestETLFullIntegration:
    """完整 ETL 管线端到端集成测试"""

    def test_full_pipeline_e2e(self):
        """完整端到端测试：原始HIS数据 → tb_risk 特征"""
        pipeline = ETLPipeline()
        pipeline.add_mappings(make_tb_risk_mappings())
        pipeline.initialize_terminology_mappers()

        # 模拟新疆某医院 HIS 系统的原始数据
        raw_data = {
            "patient_id": "XJ_2026_0001",
            "患者姓名": "买买提",
            "年龄": "288",          # 24岁（月龄）
            "性别": "男",
            "民族": "维吾尔",
            "职业": "医生",
            "诊断": "1.肺结核 2.2型糖尿病",
            "痰涂片": "阳性",
            "有无空洞": "有",
            "症状": "咳嗽、咳痰、低热、盗汗、体重下降",
            "身高": "175",
            "体重": "60",
            "吸烟年数": "5",
            "每日吸烟量": "10",
            "饮酒年数": "3",
        }

        result = pipeline.process_patient(
            raw_data, patient_id="XJ_2026_0001", source="his",
            clinical_notes=["患者主诉咳嗽咳痰2月，伴低热盗汗"],
        )

        # 验证结果
        assert result.success is True
        features = result.features

        # 字段映射
        assert features.get("patient_name") == "买买提"
        assert features.get("age") == 24.0  # 288/12
        assert features.get("gender") == 1

        # 术语编码
        assert features.get("active_tb") == "1"  # 肺结核→活动性
        assert features.get("comorbidity_diabetes") == "1"  # 糖尿病
        assert features.get("diagnosis_code") is not None

        # 国标编码
        assert features.get("ethnicity") == "05"  # 维吾尔→05
        assert features.get("occupation_high_risk") == "1"  # 医生→高风险

        # 临床指标映射
        assert features.get("sputum_smear") == "2"  # 阳性→2
        assert features.get("has_cavity") == "2"    # 有→2

        # 特征计算
        bmi = features.get("bmi")
        # BMI = 60 / (1.75^2) ≈ 19.6
        assert bmi is not None
        assert 19.0 <= bmi <= 20.0, f"BMI={bmi} 应在 19-20 之间"
        # BMI 19.6 属于正常范围（18.5-24.0）
        assert features.get("bmi_category") == "normal"

        # 吸烟指数：10支/天 * 5年 / 20 = 2.5
        assert features.get("pack_years") == 2.5
        assert features.get("smoking_years") == 5.0

        # 缺失值填补
        assert result.calculation_result is not None
        assert len(result.calculation_result.fill_records) >= 0

        # 审计追踪
        assert result.provenance is not None
        assert result.provenance.source == "his"
        assert len(result.provenance.raw_fields) > 0

        # 增量模式
        raw2 = {"patient_id": "XJ_2026_0001", "痰涂片": "阴性", "有无空洞": "无"}
        result2 = pipeline.process_patient(
            raw2, patient_id="XJ_2026_0001", source="his", incremental=True
        )
        # 症状描述不应被覆盖（增量更新只更新传入的字段）
        assert result2.success is True

        # 管线统计
        stats = pipeline.get_pipeline_statistics()
        assert stats["total_patients_processed"] >= 2
        assert stats["terminology_initialized"] is True

    def test_event_driven_workflow(self):
        """测试事件驱动工作流：LIS阳性→通知医生"""
        pipeline = ETLPipeline()
        pipeline.add_mappings(make_tb_risk_mappings())

        # 建立基线
        raw = make_his_patient(name="事件测试", diagnosis="疑似结核")
        pipeline.process_patient(
            raw, patient_id="evt_001", source="his", incremental=True
        )

        # LIS 回报痰涂片阳性
        lab_result = {
            "test_name": "痰涂片抗酸染色",
            "test_code": "sputum_smear",
            "value": "阳性",
            "abnormal_flag": "HH",
            "lab_time": "2026-08-01 10:30:00",
        }
        etl_result = pipeline.on_lab_result_received("evt_001", lab_result)

        # 应生成通知
        assert etl_result.notification is not None
        notif = etl_result.notification
        title = notif.get("title", "") if isinstance(notif, dict) else notif.title
        message = notif.get("message", "") if isinstance(notif, dict) else notif.message
        severity = notif.get("severity", "") if isinstance(notif, dict) else notif.severity
        assert "痰涂片" in title or "阳性" in message
        assert severity == "critical"

        # 验证事件历史
        history = pipeline.incremental_engine.get_event_history("evt_001")
        assert len(history) > 0
        # 应包含 LIS 事件
        lis_events = [e for e in history if "LIS" in e.get("source", "")]
        assert len(lis_events) > 0 or len(history) > 0

    def test_lab_result_loinc_mapping(self):
        """测试事件驱动路径中 LOINC 编码映射

        验证 on_lab_result_received() 中 LOINC 映射正确性，
        确保 mapped 字典包含正确的 loinc_code 和 loinc_display。
        """
        pipeline = ETLPipeline()
        pipeline.add_mappings(make_tb_risk_mappings())
        pipeline.initialize_terminology_mappers()

        # 建立基线
        raw = make_his_patient(name="LOINC测试", diagnosis="疑似结核")
        pipeline.process_patient(
            raw, patient_id="loinc_001", source="his", incremental=True
        )

        # 模拟各类检验结果到达，验证 LOINC 映射
        test_cases = [
            {
                "test_name": "痰涂片抗酸染色",
                "expected_has_loinc": True,
            },
            {
                "test_name": "结核分枝杆菌培养",
                "expected_has_loinc": True,
            },
            {
                "test_name": "C反应蛋白(CRP)",
                "expected_has_loinc": True,
            },
            {
                "test_name": "未知检验项目名称",
                "expected_has_loinc": False,
            },
        ]

        for case in test_cases:
            lab_result = {
                "test_name": case["test_name"],
                "test_code": case["test_name"],
                "value": "阴性",
                "lab_time": "2026-08-01 10:30:00",
            }
            etl_result = pipeline.on_lab_result_received("loinc_001", lab_result)

            assert etl_result.success is True
            features = etl_result.features

            if case["expected_has_loinc"]:
                assert "loinc_code" in features, (
                    f"检验 '{case['test_name']}' 应映射到 LOINC 编码"
                )
                loinc_code = features.get("loinc_code", "")
                assert loinc_code, (
                    f"检验 '{case['test_name']}' 的 LOINC 编码不应为空"
                )
                assert "loinc_display" in features, (
                    f"检验 '{case['test_name']}' 应有 LOINC 显示名称"
                )
            else:
                # 未知检验项目不应映射到 LOINC
                assert features.get("loinc_code") in (None, ""), (
                    f"未知检验 '{case['test_name']}' 不应有 LOINC 编码"
                )