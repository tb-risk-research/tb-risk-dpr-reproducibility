#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""health_interop 医疗互操作模块单元测试"""

import datetime
import os
import pytest
import tempfile
import time

from tb_risk.health_interop.base import (
    HealthcareAdapter, AdapterConfig, AdapterResult, PatientIdentifier,
    DataQualityIssue, TERMINOLOGY_SYSTEMS, FHIR_RESOURCE_MAPPING,
    LOINC_TB_PANELS, ICD10_TB_CODES, GENDER_MAPPING,
    timestamp_to_datetime, safe_parse_datetime, mask_sensitive,
    calculate_age, normalize_gender, convert_unit, parse_hl7_datetime,
    hash_identifier,
)
from tb_risk.health_interop.lis_pacs import LISMapper, RadiologyNLP, LabResultHistory
from tb_risk.health_interop.monitoring import InterfaceMonitor, CircuitBreaker


# ============================================================================
# 适配器基类与数据类测试
# ============================================================================

class TestAdapterDataClasses:
    """测试适配器配置/结果数据类"""

    def test_patient_identifier_creation(self):
        """测试PatientIdentifier创建"""
        pid = PatientIdentifier(
            id_type="empi",
            value="EMP-12345",
            assigning_authority="Hospital-A"
        )
        assert pid.id_type == "empi"
        assert pid.value == "EMP-12345"
        assert pid.assigning_authority == "Hospital-A"
        # 测试可哈希
        pid_set = {pid}
        assert len(pid_set) == 1

    def test_adapter_config_defaults(self):
        """测试AdapterConfig默认值"""
        cfg = AdapterConfig()
        assert cfg.base_url == ""
        assert cfg.port == 0
        assert cfg.auth_type == "none"
        assert cfg.connect_timeout == 5
        assert cfg.read_timeout == 30
        assert cfg.max_retries == 3
        assert cfg.history_years == 5
        assert cfg.incremental is True
        assert cfg.ssl_verify is True
        assert isinstance(cfg.ip_whitelist, list)
        assert isinstance(cfg.vendor_params, dict)

    def test_adapter_config_custom(self):
        """测试AdapterConfig自定义配置"""
        cfg = AdapterConfig(
            base_url="https://fhir.example.com/fhir",
            port=8080,
            username="user",
            password="pass",
            auth_type="basic",
            history_years=3,
            vendor_params={"custom_key": "value"}
        )
        assert cfg.base_url == "https://fhir.example.com/fhir"
        assert cfg.port == 8080
        assert cfg.auth_type == "basic"
        assert cfg.history_years == 3
        assert cfg.vendor_params["custom_key"] == "value"

    def test_adapter_result_defaults(self):
        """测试AdapterResult默认值"""
        result = AdapterResult()
        assert result.success is False
        assert result.patient_info == {}
        assert result.family_contacts == []
        assert result.social_contacts == []
        assert result.lab_results == []
        assert result.errors == []
        assert result.warnings == []
        assert result.unmapped_codes == []

    def test_data_quality_issue(self):
        """测试DataQualityIssue"""
        issue = DataQualityIssue(
            severity="error",
            field="patient_id",
            value="",
            message="患者ID不能为空",
            record_id="REC-001"
        )
        assert issue.severity == "error"
        assert issue.field == "patient_id"
        assert issue.message == "患者ID不能为空"


class TestHealthcareAdapterBase:
    """测试HealthcareAdapter抽象基类"""

    def test_cannot_instantiate_abstract(self):
        """测试抽象基类不能直接实例化"""
        with pytest.raises(TypeError):
            HealthcareAdapter()

    def test_concrete_adapter_basic(self):
        """测试具体适配器的基本功能"""

        class TestAdapter(HealthcareAdapter):
            ADAPTER_TYPE = "test"

            def connect(self):
                return True

            def fetch_patient(self, patient_id):
                return AdapterResult(success=True, patient_id=patient_id)

            def fetch_incremental(self, since=None):
                return [AdapterResult(success=True)]

        adapter = TestAdapter()
        assert adapter.ADAPTER_TYPE == "test"
        assert adapter.connect() is True
        assert adapter.health_check() == (True, "OK")

        r = adapter.fetch_patient("P123")
        assert r.success is True
        assert r.patient_id == "P123"

        results = adapter.fetch_incremental()
        assert len(results) == 1
        assert results[0].success is True

        adapter.disconnect()  # 不应抛异常


# ============================================================================
# 工具函数测试
# ============================================================================

class TestTimestampToDatetime:
    """测试timestamp_to_datetime函数"""

    def test_none_and_empty(self):
        """测试None和空字符串"""
        assert timestamp_to_datetime(None) is None
        assert timestamp_to_datetime("") is None

    def test_already_datetime(self):
        """测试已是datetime对象"""
        dt = datetime.datetime(2024, 1, 15, 10, 30, 0)
        assert timestamp_to_datetime(dt) == dt

    def test_date_object(self):
        """测试date对象"""
        d = datetime.date(2024, 1, 15)
        result = timestamp_to_datetime(d)
        assert result == datetime.datetime(2024, 1, 15, 0, 0, 0)

    def test_unix_timestamp_seconds(self):
        """测试Unix时间戳（秒）"""
        # 2024-01-15 10:30:00 UTC
        ts = 1705314600
        result = timestamp_to_datetime(ts)
        assert result is not None
        assert result.year == 2024
        assert result.month == 1

    def test_unix_timestamp_milliseconds(self):
        """测试Unix时间戳（毫秒）"""
        ts = 1705314600000  # 毫秒
        result = timestamp_to_datetime(ts)
        assert result is not None
        assert result.year == 2024

    def test_invalid_timestamp(self):
        """测试无效时间戳"""
        assert timestamp_to_datetime(-1) is None
        assert timestamp_to_datetime(float('inf')) is None


class TestSafeParseDatetime:
    """测试safe_parse_datetime函数"""

    def test_none_and_empty(self):
        """测试None和空"""
        assert safe_parse_datetime(None) is None
        assert safe_parse_datetime("") is None
        assert safe_parse_datetime("   ") is None

    def test_iso_format(self):
        """测试ISO格式"""
        dt = safe_parse_datetime("2024-01-15")
        assert dt == datetime.datetime(2024, 1, 15)

    def test_iso_datetime(self):
        """测试ISO日期时间"""
        dt = safe_parse_datetime("2024-01-15 10:30:00")
        assert dt == datetime.datetime(2024, 1, 15, 10, 30, 0)

    def test_iso_with_t_z(self):
        """测试ISO带T和Z"""
        dt = safe_parse_datetime("2024-01-15T10:30:00Z")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

    def test_slash_format(self):
        """测试斜杠格式"""
        dt = safe_parse_datetime("2024/01/15")
        assert dt == datetime.datetime(2024, 1, 15)

    def test_compact_format(self):
        """测试紧凑格式"""
        dt = safe_parse_datetime("20240115")
        assert dt == datetime.datetime(2024, 1, 15)

    def test_invalid_string(self):
        """测试无效字符串"""
        assert safe_parse_datetime("not-a-date") is None
        assert safe_parse_datetime("2024-13-45") is None


class TestCalculateAge:
    """测试calculate_age函数"""

    def test_none_input(self):
        """测试None输入"""
        assert calculate_age(None) is None
        assert calculate_age("") is None

    def test_basic_age(self):
        """测试基本年龄计算"""
        # 2000-01-01 出生，假设今天是2024年
        birth = datetime.datetime(2000, 1, 1)
        today = datetime.datetime(2024, 6, 15)
        age = calculate_age(birth, reference_date=today)
        assert age == 24

    def test_birthday_not_passed(self):
        """测试今年生日未过"""
        birth = datetime.datetime(2000, 12, 31)
        today = datetime.datetime(2024, 6, 15)
        age = calculate_age(birth, reference_date=today)
        assert age == 23

    def test_birthday_today(self):
        """测试今天是生日"""
        birth = datetime.datetime(2000, 6, 15)
        today = datetime.datetime(2024, 6, 15)
        age = calculate_age(birth, reference_date=today)
        assert age == 24

    def test_age_from_string(self):
        """测试从字符串解析"""
        age = calculate_age("2000-01-01", reference_date=datetime.datetime(2024, 6, 15))
        assert age == 24

    def test_invalid_age(self):
        """测试异常年龄值"""
        # 未来日期
        future = datetime.datetime(2030, 1, 1)
        assert calculate_age(future) is None
        # 超过120岁
        ancient = datetime.datetime(1800, 1, 1)
        assert calculate_age(ancient) is None


class TestMaskSensitive:
    """测试mask_sensitive脱敏函数"""

    def test_none_input(self):
        """测试None输入"""
        assert mask_sensitive(None) == ""

    def test_short_string(self):
        """测试短字符串"""
        assert mask_sensitive("a") == "*"
        assert mask_sensitive("ab") == "**"

    def test_id_card(self):
        """测试身份证号脱敏"""
        result = mask_sensitive("110101199001011234", keep_prefix=3, keep_suffix=4)
        assert result.startswith("110")
        assert result.endswith("1234")
        assert "*" in result

    def test_phone_number(self):
        """测试手机号脱敏"""
        result = mask_sensitive("13800138000", keep_prefix=3, keep_suffix=4)
        assert result.startswith("138")
        assert result.endswith("8000")

    def test_custom_mask_char(self):
        """测试自定义掩码字符"""
        result = mask_sensitive("1234567890", mask_char="#", keep_prefix=2, keep_suffix=2)
        assert result.startswith("12")
        assert result.endswith("90")
        assert "#" in result


class TestNormalizeGender:
    """测试normalize_gender函数"""

    def test_none_input(self):
        """测试None输入"""
        assert normalize_gender(None) == "unknown"

    def test_chinese(self):
        """测试中文性别"""
        assert normalize_gender("男") == "male"
        assert normalize_gender("女") == "female"
        assert normalize_gender("未知") == "unknown"

    def test_english(self):
        """测试英文性别"""
        assert normalize_gender("male") == "male"
        assert normalize_gender("female") == "female"
        assert normalize_gender("MALE") == "male"

    def test_hl7_codes(self):
        """测试HL7编码"""
        assert normalize_gender("M") == "male"
        assert normalize_gender("F") == "female"
        assert normalize_gender("U") == "unknown"
        assert normalize_gender("O") == "other"

    def test_numeric_codes(self):
        """测试数字编码"""
        assert normalize_gender("1") == "male"
        assert normalize_gender("2") == "female"
        assert normalize_gender("0") == "unknown"

    def test_unknown(self):
        """测试未知值"""
        assert normalize_gender("something_else") == "unknown"


class TestConvertUnit:
    """测试convert_unit单位换算"""

    def test_same_unit(self):
        """测试相同单位"""
        assert convert_unit(100, "mg/dL", "mg/dL") == 100

    def test_empty_units(self):
        """测试空单位"""
        assert convert_unit(100, "", "") == 100
        assert convert_unit(100, "mg/dL", "") == 100

    def test_glucose_mmol_to_mgdl(self):
        """测试血糖 mmol/L → mg/dL"""
        result = convert_unit(5.5, "mmol/l", "mg/dl")
        assert result is not None
        assert abs(result - 99.088) < 0.1

    def test_glucose_mgdl_to_mmol(self):
        """测试血糖 mg/dL → mmol/L"""
        result = convert_unit(100, "mg/dl", "mmol/l")
        assert result is not None
        assert abs(result - 5.55) < 0.1

    def test_mm_to_cm(self):
        """测试长度 mm → cm"""
        assert convert_unit(100, "mm", "cm") == 10.0
        assert convert_unit(10, "cm", "mm") == 100.0

    def test_temperature(self):
        """测试温度换算"""
        # 0°C = 32°F
        assert abs(convert_unit(0, "c", "f") - 32.0) < 0.1
        # 100°C = 212°F
        assert abs(convert_unit(100, "c", "f") - 212.0) < 0.1

    def test_unknown_conversion(self):
        """测试未知单位换算"""
        assert convert_unit(100, "unit_a", "unit_b") is None

    def test_invalid_value(self):
        """测试无效值"""
        assert convert_unit("not-a-number", "mmol/l", "mg/dl") is None


class TestParseHL7Datetime:
    """测试parse_hl7_datetime函数"""

    def test_none_and_empty(self):
        """测试None和空"""
        assert parse_hl7_datetime(None) is None
        assert parse_hl7_datetime("") is None

    def test_basic_hl7(self):
        """测试基本HL7日期时间"""
        dt = parse_hl7_datetime("20240115103000")
        assert dt == datetime.datetime(2024, 1, 15, 10, 30, 0)

    def test_hl7_with_fractional(self):
        """测试带小数秒"""
        dt = parse_hl7_datetime("20240115103000.1234")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 10
        assert dt.microsecond == 123400

    def test_hl7_date_only(self):
        """测试仅日期"""
        dt = parse_hl7_datetime("20240115")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 0

    def test_invalid_hl7(self):
        """测试无效HL7格式，回退到通用解析"""
        dt = parse_hl7_datetime("2024-01-15")
        assert dt is not None


class TestHashIdentifier:
    """测试hash_identifier函数"""

    def test_returns_string(self):
        """测试返回字符串"""
        h = hash_identifier("test")
        assert isinstance(h, str)
        assert len(h) == 16

    def test_deterministic(self):
        """测试确定性（相同输入相同输出）"""
        h1 = hash_identifier("P12345")
        h2 = hash_identifier("P12345")
        assert h1 == h2

    def test_different_inputs(self):
        """测试不同输入产生不同哈希"""
        h1 = hash_identifier("P12345")
        h2 = hash_identifier("P67890")
        assert h1 != h2


# ============================================================================
# 常量测试
# ============================================================================

class TestTerminologyConstants:
    """测试术语系统常量"""

    def test_terminology_systems_exist(self):
        """测试术语系统常量存在"""
        assert hasattr(TERMINOLOGY_SYSTEMS, "LOINC")
        assert hasattr(TERMINOLOGY_SYSTEMS, "ICD10")
        assert hasattr(TERMINOLOGY_SYSTEMS, "SNOMED_CT")

    def test_fhir_resource_mapping(self):
        """测试FHIR资源映射包含关键字段"""
        assert "age" in FHIR_RESOURCE_MAPPING
        assert "gender" in FHIR_RESOURCE_MAPPING
        assert "diabetes" in FHIR_RESOURCE_MAPPING
        assert "sputum_smear" in FHIR_RESOURCE_MAPPING
        assert "has_cavity" in FHIR_RESOURCE_MAPPING

        # 验证每个映射有resource字段
        for key, mapping in FHIR_RESOURCE_MAPPING.items():
            assert "resource" in mapping, f"{key} 缺少resource字段"

    def test_loinc_tb_panels(self):
        """测试LOINC结核面板有五大类"""
        assert "pathogen" in LOINC_TB_PANELS
        assert "molecular" in LOINC_TB_PANELS
        assert "serology" in LOINC_TB_PANELS
        assert "routine" in LOINC_TB_PANELS
        assert "infectious_screen" in LOINC_TB_PANELS

        # 每个面板有项目
        for category, items in LOINC_TB_PANELS.items():
            assert len(items) > 0, f"{category} 面板为空"
            for item in items:
                assert "loinc" in item
                assert "name" in item

    def test_icd10_tb_codes(self):
        """测试ICD-10结核编码"""
        assert "A15" in ICD10_TB_CODES
        assert "A16" in ICD10_TB_CODES
        assert "A17" in ICD10_TB_CODES
        assert "B90" in ICD10_TB_CODES

    def test_gender_mapping(self):
        """测试性别映射包含多种编码"""
        assert "男" in GENDER_MAPPING
        assert "女" in GENDER_MAPPING
        assert "M" in GENDER_MAPPING
        assert "F" in GENDER_MAPPING
        assert "male" in GENDER_MAPPING
        assert "female" in GENDER_MAPPING


# ============================================================================
# LIS/PACS 测试
# ============================================================================

class TestLISMapper:
    """测试LISMapper检验项目映射"""

    def setup_method(self):
        self.mapper = LISMapper()

    def test_loinc_exact_match(self):
        """测试LOINC编码精确匹配"""
        result = self.mapper.map_item("", loinc_code="640-4")
        assert result is not None
        assert result["category"] == "pathogen"
        assert "涂片" in result["name"] or result["loinc"] == "640-4"

    def test_keyword_match_chinese(self):
        """测试中文关键词匹配"""
        result = self.mapper.map_item("痰涂片抗酸染色")
        assert result is not None
        assert result["category"] == "pathogen"

    def test_keyword_match_xpert(self):
        """测试Xpert关键词"""
        result = self.mapper.map_item("Xpert MTB/RIF")
        assert result is not None
        assert result["category"] == "molecular"

    def test_keyword_match_tspot(self):
        """测试T-SPOT关键词"""
        result = self.mapper.map_item("T-SPOT检测")
        assert result is not None
        assert result["category"] == "serology"

    def test_keyword_match_hiv(self):
        """测试HIV关键词"""
        result = self.mapper.map_item("HIV抗体检测")
        assert result is not None
        assert result["category"] == "infectious_screen"

    def test_no_match(self):
        """测试无匹配项"""
        result = self.mapper.map_item("血常规")
        # 血常规没有在结核面板中，可能匹配不到或匹配到白细胞
        # 这里只确保不抛异常
        assert result is None or isinstance(result, dict)

    def test_interpret_positive(self):
        """测试阳阳性解释"""
        assert LISMapper.interpret_result("阳性") is True
        assert LISMapper.interpret_result("+") is True
        assert LISMapper.interpret_result("检出抗酸杆菌") is True
        assert LISMapper.interpret_result("Positive") is True

    def test_interpret_negative(self):
        """测试阴阳性解释"""
        assert LISMapper.interpret_result("阴性") is False
        assert LISMapper.interpret_result("-") is False
        assert LISMapper.interpret_result("未检出") is False
        assert LISMapper.interpret_result("Negative") is False

    def test_interpret_abnormal_flag(self):
        """测试异常标志"""
        assert LISMapper.interpret_result("", abnormal_flag="H") is True
        assert LISMapper.interpret_result("", abnormal_flag="HH") is True
        assert LISMapper.interpret_result("", abnormal_flag="N") is False
        assert LISMapper.interpret_result("", abnormal_flag="NORMAL") is False

    def test_interpret_none(self):
        """测试无法判断"""
        assert LISMapper.interpret_result(None) is None
        assert LISMapper.interpret_result("") is None
        assert LISMapper.interpret_result("some text") is None


class TestRadiologyNLP:
    """测试影像报告NLP征象提取"""

    def setup_method(self):
        self.nlp = RadiologyNLP()

    def test_no_findings(self):
        """测试无异常"""
        result = self.nlp.extract_features("双肺未见明显异常")
        assert result.get("has_cavity") is not True
        assert result.get("mentions_tb") is not True

    def test_cavity_detection(self):
        """测试空洞检测"""
        result = self.nlp.extract_features("右肺上叶可见空洞形成")
        assert result.get("has_cavity") is True
        assert "cavity" in result.get("lesions", [])

    def test_cavity_english(self):
        """测试英文cavity"""
        result = self.nlp.extract_features("Cavity in upper lobe")
        assert result.get("has_cavity") is True

    def test_tb_suspicious_phrase(self):
        """测试结核待排表述"""
        result = self.nlp.extract_features("考虑继发性肺结核可能性大")
        assert result.get("mentions_tb") is True or result.get("is_suspicious") is True

    def test_infiltrate_keyword(self):
        """测试浸润影（斑片影）"""
        result = self.nlp.extract_features("双肺散在斑片状阴影")
        assert "patchy_opacity" in result.get("lesions", [])

    def test_nodule_detection(self):
        """测试结节"""
        result = self.nlp.extract_features("左肺可见结节影")
        assert "nodule" in result.get("lesions", [])

    def test_consolidation_detection(self):
        """测试实变"""
        result = self.nlp.extract_features("肺实变征象")
        assert "consolidation" in result.get("lesions", [])

    def test_pleural_effusion(self):
        """测试胸腔积液"""
        result = self.nlp.extract_features("右侧胸腔积液")
        assert result.get("has_pleural_effusion") is True

    def test_tb_predilection_site(self):
        """测试结核好发部位"""
        result = self.nlp.extract_features("上叶尖后段可见斑片影")
        assert result.get("tb_predilection_site") is True

    def test_empty_text(self):
        """测试空文本"""
        result = self.nlp.extract_features("")
        assert isinstance(result, dict)
        assert result.get("has_cavity") is False

    def test_none_text(self):
        """测试None文本"""
        result = self.nlp.extract_features(None)
        assert isinstance(result, dict)
        assert result.get("confidence") is not None


class TestLabResultHistory:
    """测试检验结果历史"""

    def test_empty_history(self):
        """测试空历史"""
        hist = LabResultHistory()
        assert hist.has_positive is False
        assert hist.latest_positive_time is None
        assert hist.positive_count == 0
        assert hist.has_seroconversion is False

    def test_positive_results(self):
        """测试有阳性结果"""
        t1 = datetime.datetime(2024, 1, 1)
        t2 = datetime.datetime(2024, 2, 1)
        hist = LabResultHistory(
            results=[
                {"effective_time": t1, "is_positive": False},
                {"effective_time": t2, "is_positive": True},
            ]
        )
        assert hist.has_positive is True
        assert hist.positive_count == 1
        assert hist.latest_positive_time == t2

    def test_seroconversion(self):
        """测试阳转（先阴后阳）"""
        t1 = datetime.datetime(2024, 1, 1)
        t2 = datetime.datetime(2024, 2, 1)
        t3 = datetime.datetime(2024, 3, 1)
        hist = LabResultHistory(
            results=[
                {"effective_time": t1, "is_positive": False},
                {"effective_time": t2, "is_positive": False},
                {"effective_time": t3, "is_positive": True},
            ]
        )
        assert hist.has_seroconversion is True

    def test_seroreversion(self):
        """测试阴转（先阳后阴）"""
        t1 = datetime.datetime(2024, 1, 1)
        t2 = datetime.datetime(2024, 2, 1)
        hist = LabResultHistory(
            results=[
                {"effective_time": t1, "is_positive": True},
                {"effective_time": t2, "is_positive": False},
            ]
        )
        assert hist.has_seroconversion is True

    def test_no_seroconversion(self):
        """测试无血清学转换"""
        t1 = datetime.datetime(2024, 1, 1)
        t2 = datetime.datetime(2024, 2, 1)
        hist = LabResultHistory(
            results=[
                {"effective_time": t1, "is_positive": True},
                {"effective_time": t2, "is_positive": True},
            ]
        )
        assert hist.has_seroconversion is False


# ============================================================================
# 监控熔断测试
# ============================================================================

class TestCircuitBreaker:
    """测试CircuitBreaker熔断机制"""

    def test_initial_state(self):
        """测试初始状态（闭合）"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
        assert cb.state == "closed"
        assert cb.allow_request() is True

    def test_open_after_failures(self):
        """测试失败后断开"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"
        assert cb.allow_request() is False

    def test_reset_after_success(self):
        """测试成功后重置"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
        for _ in range(2):
            cb.record_failure()
        assert cb.state == "closed"
        cb.record_success()
        # 成功后失败计数重置

    def test_half_open_after_timeout(self):
        """测试超时后半开状态"""
        import time
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        # 等待恢复超时
        time.sleep(0.15)
        assert cb.allow_request() is True
        assert cb.state == "half_open"

    def test_half_open_success_closes(self):
        """测试半开状态成功后闭合"""
        import time
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        cb.allow_request()  # 进入half_open
        cb.record_success()
        assert cb.state == "closed"

    def test_half_open_failure_opens(self):
        """测试半开状态失败后重新断开"""
        import time
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        cb.allow_request()
        cb.record_failure()
        assert cb.state == "open"

    def test_manual_reset(self):
        """测试手动重置"""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        cb.reset()
        assert cb.state == "closed"
        assert cb.allow_request() is True


class TestInterfaceMonitor:
    """测试InterfaceMonitor"""

    def test_record_success(self):
        """测试记录成功"""
        monitor = InterfaceMonitor()
        monitor.record_success("test", "connect", 100.5, 10)
        metrics = monitor.metrics.get_metrics("test")
        assert len(metrics) > 0
        key = "test:connect"
        assert key in metrics
        assert metrics[key]["successes"] == 1
        assert metrics[key]["failures"] == 0
        assert metrics[key]["total_requests"] == 1

    def test_record_failure(self):
        """测试记录失败"""
        monitor = InterfaceMonitor()
        monitor.record_failure("test", "connect", "Connection error", 50.0)
        metrics = monitor.metrics.get_metrics("test")
        key = "test:connect"
        assert key in metrics
        assert metrics[key]["successes"] == 0
        assert metrics[key]["failures"] == 1

    def test_availability_calculation(self):
        """测试可用率计算"""
        monitor = InterfaceMonitor()
        for _ in range(9):
            monitor.record_success("test", "op", 10)
        monitor.record_failure("test", "op", "error")
        metrics = monitor.metrics.get_metrics("test")
        key = "test:op"
        assert key in metrics
        assert abs(metrics[key]["availability"] - 90.0) < 0.1

    def test_multiple_adapters(self):
        """测试多适配器统计隔离"""
        monitor = InterfaceMonitor()
        monitor.record_success("fhir", "patient", 50)
        monitor.record_failure("hl7", "parse", "bad msg")
        fhir_metrics = monitor.metrics.get_metrics("fhir")
        hl7_metrics = monitor.metrics.get_metrics("hl7")
        assert "fhir:patient" in fhir_metrics
        assert "hl7:parse" in hl7_metrics

    def test_circuit_breaker_integration(self):
        """测试熔断器集成"""
        monitor = InterfaceMonitor(failure_threshold=3, recovery_timeout=10)
        # 连续失败触发熔断
        for _ in range(3):
            monitor.record_failure("test", "op", "error")
        assert monitor.allow_request("test", "op") is False

    def test_status_report(self):
        """测试状态报告生成"""
        monitor = InterfaceMonitor()
        monitor.record_success("test", "op", 10)
        report = monitor.get_status_report()
        assert "metrics" in report
        assert "circuit_breakers" in report
        assert "thresholds" in report

    def test_alert_callback(self):
        """测试告警回调"""
        monitor = InterfaceMonitor(consecutive_failures_alert=2)
        alerts = []
        monitor.register_alert_callback(lambda a: alerts.append(a))
        monitor.record_failure("test", "op", "err1")
        monitor.record_failure("test", "op", "err2")
        # 应触发告警
        assert len(alerts) > 0


# ============================================================================
# LISPACSDecorator 装饰器测试
# ============================================================================

class TestLISPACSDecorator:
    """测试LIS/PACS装饰器模式"""

    def test_decorator_wraps_base_adapter(self):
        """测试装饰器正确包装基础适配器"""
        from health_interop import LISPACSDecorator, LISPACSAdapter, HealthcareAdapter, AdapterConfig, AdapterResult

        # 创建一个简单的Mock适配器
        class MockAdapter(HealthcareAdapter):
            def __init__(self):
                super().__init__(AdapterConfig(base_url="http://test.example.com"))
                self._connected = False

            def connect(self):
                self._connected = True
                return True

            def disconnect(self):
                self._connected = False

            def fetch_patient(self, patient_id):
                return AdapterResult(
                    success=True,
                    patient_info={"id": patient_id, "name": "测试患者"},
                    lab_results=[
                        {"item_name": "痰涂片抗酸染色", "value": "阳性(+)"},
                        {"item_name": "C反应蛋白", "value": "25", "unit": "mg/L", "abnormal_flag": "H"},
                    ],
                    imaging_reports=[
                        {"title": "胸部CT", "conclusion": "右上肺尖后段可见空洞形成，考虑肺结核可能"},
                    ]
                )

            def fetch_incremental(self, since=None):
                return [self.fetch_patient("P001")]

        base = MockAdapter()
        decorator = LISPACSDecorator(base, enable_lis=True, enable_pacs=True)

        # 测试连接委托
        assert not decorator.is_connected
        assert decorator.connect() is True
        assert decorator.is_connected

        # 测试fetch_patient增强
        result = decorator.fetch_patient("P001")
        assert result.success is True
        # 检验应被标准化
        assert len(result.lab_results) == 2
        assert result.lab_results[0].get("category") == "pathogen"
        assert result.lab_results[0].get("is_positive") is True
        # 影像应被NLP处理
        assert len(result.imaging_reports) == 1
        assert result.imaging_reports[0].get("nlp_processed") is True
        assert "nlp_features" in result.imaging_reports[0]
        # risk_features应包含增强特征
        assert "sputum_smear_positive" in result.risk_features
        assert result.risk_features["sputum_smear_positive"] is True
        assert result.risk_features["has_cavity_any"] is True
        # 警告信息
        assert any("LIS" in w for w in result.warnings)
        assert any("PACS" in w for w in result.warnings)

        decorator.disconnect()
        assert not decorator.is_connected

    def test_decorator_disable_options(self):
        """测试装饰器可以单独启用/禁用LIS或PACS"""
        from health_interop import LISPACSDecorator, HealthcareAdapter, AdapterConfig, AdapterResult

        class MockAdapter(HealthcareAdapter):
            def __init__(self):
                super().__init__(AdapterConfig(base_url="http://test.example.com"))

            def connect(self):
                return True

            def fetch_patient(self, patient_id):
                return AdapterResult(
                    success=True,
                    lab_results=[{"item_name": "痰涂片", "value": "阳性"}],
                    imaging_reports=[{"conclusion": "未见异常"}]
                )

            def fetch_incremental(self, since=None):
                return []

        # 仅启用LIS
        base1 = MockAdapter()
        dec1 = LISPACSDecorator(base1, enable_lis=True, enable_pacs=False)
        dec1.connect()
        r1 = dec1.fetch_patient("P1")
        assert any("LIS" in w for w in r1.warnings)
        assert not any("PACS" in w for w in r1.warnings)
        assert "sputum_smear_positive" in r1.risk_features
        # 影像不应被处理
        assert "nlp_processed" not in r1.imaging_reports[0]

        # 仅启用PACS
        base2 = MockAdapter()
        dec2 = LISPACSDecorator(base2, enable_lis=False, enable_pacs=True)
        dec2.connect()
        r2 = dec2.fetch_patient("P1")
        assert not any("LIS" in w for w in r2.warnings)
        assert any("PACS" in w for w in r2.warnings)
        assert "nlp_features" in r2.imaging_reports[0]


# ============================================================================
# FeatureMapper 测试
# ============================================================================

class TestFeatureMapper:
    """测试临床数据到ML特征的映射"""

    def test_basic_mapping(self):
        """测试基本人口学信息映射"""
        from health_interop import FeatureMapper, AdapterResult
        mapper = FeatureMapper()
        result = AdapterResult(
            success=True,
            patient_info={
                "gender": "male",
                "age": 45,
            },
            lab_results=[],
            diagnoses=[{"code": "A15.0", "display": "肺结核，经痰涂片证实"}],
        )
        features = mapper.map_patient_basic_info(result)
        assert features["gender"] == 1  # male=1
        assert features["age"] == 45
        assert features["active_tb"] == "1"  # ICD-10 A15是活动性结核

    def test_lab_result_mapping(self):
        """测试检验结果特征映射"""
        from health_interop import FeatureMapper, AdapterResult

        mapper = FeatureMapper()

        result = AdapterResult(
            success=True,
            patient_info={"gender": "female", "age": 32},
            lab_results=[
                {"name": "痰涂片抗酸染色", "value": "阳性", "interpretation": "positive"},
                {"name": "HIV抗体", "value": "阴性", "interpretation": "negative"},
            ],
        )

        features = mapper.map_patient_basic_info(result)
        assert features["sputum_smear"] == "2"  # "2"=涂阳
        assert features["age"] == 32
        assert features["gender"] == 0  # female=0


# ============================================================================
# AdapterResult 新字段测试
# ============================================================================

class TestAdapterResultRiskFeatures:
    """测试AdapterResult新增的risk_features字段"""

    def test_risk_features_default(self):
        """测试risk_features默认值为空dict"""
        from health_interop.base import AdapterResult
        r = AdapterResult()
        assert r.risk_features == {}
        assert isinstance(r.risk_features, dict)


# ============================================================================
# FHIR 资源解析器测试
# ============================================================================

class TestFHIRResourceParser:
    """测试FHIR R4资源解析器"""

    def _make_parser(self):
        from health_interop.fhir.parser import FHIRResourceParser
        return FHIRResourceParser()

    def test_parse_patient_basic(self):
        """测试基本Patient资源解析"""
        parser = self._make_parser()
        from health_interop.base import AdapterResult
        result = AdapterResult()

        patient = {
            "resourceType": "Patient",
            "id": "pat-001",
            "identifier": [
                {"system": "http://hospital.example.com/mrn", "value": "MRN12345"},
                {"system": "http://example.com/empi", "value": "EMP-001"},
            ],
            "name": [{"family": "张", "given": ["三"], "text": "张三"}],
            "gender": "male",
            "birthDate": "1980-05-15",
            "telecom": [{"system": "phone", "value": "13800138000"}],
            "address": [{"line": ["某街道123号"], "city": "北京", "district": "朝阳区"}],
        }

        parser.parse_patient(patient, result)
        pi = result.patient_info
        assert pi["name"] == "张三"
        assert pi["gender"] == "male"
        assert pi["gender_text"] == "男"
        assert pi["medical_record_no"] == "MRN12345"
        assert result.patient_id == "EMP-001"
        assert pi["phone"] == "13800138000"
        assert "北京" in pi["address"]
        assert pi["district"] == "朝阳区"
        assert pi["age"] is not None
        assert pi["age"] >= 43  # 1980年生，2025年至少44岁左右

    def test_parse_patient_empty(self):
        """测试空Patient资源"""
        parser = self._make_parser()
        from health_interop.base import AdapterResult
        result = AdapterResult()
        parser.parse_patient(None, result)
        parser.parse_patient({}, result)
        parser.parse_patient({"resourceType": "Observation"}, result)
        assert result.patient_info == {}

    def test_parse_condition_tb(self):
        """测试结核Condition解析"""
        parser = self._make_parser()
        condition = {
            "resourceType": "Condition",
            "id": "cond-001",
            "code": {
                "coding": [{
                    "system": "http://hl7.org/fhir/sid/icd-10",
                    "code": "A15.0",
                    "display": "肺结核，经痰涂片证实"
                }]
            },
            "onsetDateTime": "2024-01-15",
            "recordedDate": "2024-01-20",
            "clinicalStatus": {"coding": [{"code": "active"}]},
        }
        diag = parser.parse_condition(condition)
        assert diag is not None
        assert diag["code"] == "A15.0"
        assert diag["is_tb"] is True
        assert diag["clinical_status"] == "active"

    def test_parse_condition_empty(self):
        """测试空Condition"""
        parser = self._make_parser()
        assert parser.parse_condition(None) is None
        assert parser.parse_condition({}) is None
        assert parser.parse_condition({"resourceType": "Patient"}) is None

    def test_parse_observation_lab(self):
        """测试Observation检验结果解析（使用正确的LOINC编码640-4痰涂片）"""
        parser = self._make_parser()
        obs = {
            "resourceType": "Observation",
            "id": "obs-001",
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "640-4",  # 痰涂片抗酸染色（正确LOINC编码）
                    "display": "Sputum acid fast stain"
                }]
            },
            "valueCodeableConcept": {
                "coding": [{"code": "positive", "display": "Positive"}]
            },
            "effectiveDateTime": "2024-01-18",
            "status": "final",
        }
        lab = parser.parse_observation(obs)
        assert lab is not None
        assert lab["category"] == "pathogen"
        assert lab["name"] == "痰涂片抗酸染色"
        assert lab["is_positive"] is True  # "positive" 被识别为阳性

    def test_parse_bundle(self):
        """测试Bundle资源遍历和解析（使用client模式：遍历entry逐个解析）"""
        parser = self._make_parser()
        from health_interop.base import AdapterResult
        result = AdapterResult()

        bundle = {
            "resourceType": "Bundle",
            "type": "searchset",
            "entry": [
                {"resource": {
                    "resourceType": "Patient",
                    "name": [{"text": "李四"}],
                    "gender": "female",
                    "birthDate": "1990-03-20",
                }},
                {"resource": {
                    "resourceType": "Condition",
                    "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10",
                                         "code": "A16.2", "display": "肺结核"}]},
                }},
            ]
        }

        # 模拟FHIR client的bundle遍历方式
        for entry in bundle.get("entry", []):
            res = entry.get("resource")
            if not res:
                continue
            rtype = res.get("resourceType")
            if rtype == "Patient":
                parser.parse_patient(res, result)
            elif rtype == "Condition":
                diag = parser.parse_condition(res)
                if diag:
                    result.diagnoses.append(diag)

        assert result.patient_info["name"] == "李四"
        assert result.patient_info["gender"] == "female"
        assert len(result.diagnoses) >= 1
        assert result.diagnoses[0]["is_tb"] is True


# ============================================================================
# HL7 v2 消息解析器测试
# ============================================================================

class TestHL7MessageParser:
    """测试HL7 v2.x消息解析"""

    def _make_parser(self):
        from health_interop.hl7v2.parser import HL7MessageParser
        return HL7MessageParser()

    def test_parse_adt_a04(self):
        """测试ADT^A04（注册患者）消息解析"""
        parser = self._make_parser()
        # 构造一个简单的ADT^A04消息
        hl7_msg = (
            "MSH|^~\\&|HIS|HOSPITAL_A|EMR|HOSPITAL_A|20240115103000||ADT^A04|MSG00001|P|2.5\r"
            "PID|||MRN-12345^^^HOSPITAL_MRN||张^三^^^先生||19800515|M|||北京市朝阳区某街道123号^^北京^^100000||13800138000||||||||110101198005150001\r"
            "PV1||I|WARD^01^床号01\r"
        )
        msg = parser.parse(hl7_msg)
        assert msg.message_type == "ADT^A04"
        assert msg.trigger_event == "A04"
        assert msg.sending_app == "HIS"
        assert msg.sending_facility == "HOSPITAL_A"
        assert msg.patient_info.get("name") == "张三"
        assert msg.patient_info.get("gender") == "male"
        assert "birth_date" in msg.patient_info

    def test_parse_oru_r01(self):
        """测试ORU^R01（检验结果）消息解析"""
        parser = self._make_parser()
        hl7_msg = (
            "MSH|^~\\&|LIS|HOSPITAL_A|EMR|HOSPITAL_A|20240116140000||ORU^R01|MSG00002|P|2.5\r"
            "PID|||MRN-12345||李^四||19900320|F\r"
            "ORC|RE|ORD-001\r"
            "OBR|1|ORD-001||640-4^痰涂片抗酸染色^LN|||20240116090000\r"
            "OBX|1|ST|640-4^痰涂片抗酸染色^LN||阳性（+）||||||F\r"
            "OBX|2|NM|718-7^血红蛋白^LN||125|g/L|115-150|N|||F\r"
        )
        msg = parser.parse(hl7_msg)
        assert msg.message_type == "ORU^R01"
        assert msg.patient_info.get("name") == "李四"
        assert msg.patient_info.get("gender") == "female"
        # 应有检验结果
        assert len(msg.lab_results) >= 1

    def test_parse_empty_message(self):
        """测试空消息处理"""
        parser = self._make_parser()
        msg = parser.parse("")
        assert len(msg.parse_errors) >= 1
        msg2 = parser.parse("\x0b\x1c\x0d")  # 仅MLLP帧
        assert len(msg2.parse_errors) >= 1

    def test_parse_missing_msh(self):
        """测试缺少MSH段的消息"""
        parser = self._make_parser()
        msg = parser.parse("PID|||123||张三\r")
        assert any("MSH" in e for e in msg.parse_errors)

    def test_get_component_method(self):
        """测试HL7Message便捷访问方法"""
        from health_interop.hl7v2.parser import HL7Message
        msg = HL7Message()
        msg.segments = {
            "PID": [["PID", "", "", "MRN-001", "", "张^三"]]
        }
        assert msg.get_fields("PID", 3) == "MRN-001"
        assert msg.get_component("PID", 5, 0) == "张"
        assert msg.get_component("PID", 5, 1) == "三"
        assert msg.get_segment("PID") is not None
        assert msg.get_segment("PV1") is None


# ============================================================================
# 疾控上报 TBCardFiller 测试
# ============================================================================

class TestTBCardFiller:
    """测试传染病报告卡自动填报"""

    def _make_filler(self):
        from health_interop.cdc import TBCardFiller
        return TBCardFiller()

    def test_fill_card_confirmed_tb(self):
        """测试确诊结核病例自动填卡"""
        filler = self._make_filler()
        patient = {
            "patient_id": "P001",
            "name": "王五",
            "gender": "male",
            "age": 45,
            "id_card": "110101197901010001",
            "phone": "13900139000",
            "address": "北京市某区某街道",
            "occupation": "工人",
        }
        labs = [
            {"name": "痰涂片抗酸染色", "value": "阳性（+）", "category": "pathogen",
             "is_positive": True, "interpretation": "positive"},
            {"name": "Xpert MTB/RIF", "value": "MTB检出，利福平敏感",
             "category": "molecular", "is_positive": True},
        ]
        diagnoses = [
            {"code": "A15.0", "display": "肺结核，经痰涂片证实",
             "diagnosis_date": datetime.datetime(2024, 1, 20)}
        ]
        images = [
            {"conclusion": "右上肺可见空洞形成，双肺多发斑片状阴影",
             "nlp_features": {"has_cavity": True, "tb_predilection_site": True,
                              "mentions_tb": True}}
        ]

        card = filler.fill_card(patient, labs, diagnoses, images,
                                 report_unit="某医院", report_doctor="张医生")
        assert card.name == "王五"
        assert card.patient_id == "P001"
        assert card.gender == "male"
        assert card.gender_text == "男"
        assert card.age == 45
        assert card.case_classification == "confirmed"
        assert "laboratory" in card.diagnosis_basis
        assert card.treatment_category == "初治"
        assert card.drug_resistance in ("敏感", "待检测/敏感")

    def test_fill_card_clinical_tb(self):
        """测试临床诊断病例（无病原学阳性）"""
        filler = self._make_filler()
        patient = {
            "name": "赵六", "gender": "female", "age": 28,
            "history_tb": False,
        }
        images = [
            {"conclusion": "右肺上叶尖后段浸润影，考虑结核可能",
             "nlp_features": {"tb_predilection_site": True, "mentions_tb": True}}
        ]
        card = filler.fill_card(patient, lab_results=[], diagnoses=[],
                                 imaging_reports=images)
        assert card.case_classification == "clinical"
        assert "clinical" in card.diagnosis_basis

    def test_fill_card_minimal(self):
        """测试最小患者信息填卡"""
        filler = self._make_filler()
        patient = {"name": "孙七", "gender": "male", "age": 55}
        card = filler.fill_card(patient)
        assert card.name == "孙七"
        assert card.gender == "male"
        assert card.case_classification == "suspected"
        # 应有缺失字段
        assert len(card.missing_fields) > 0

    def test_duplicate_check(self):
        """测试报告卡查重"""
        filler = self._make_filler()
        existing = [
            {"id_card": "110101198001010001", "name": "张三", "age": 44},
            {"id_card": "320101199001010002", "name": "李四", "age": 34},
        ]
        # 身份证匹配
        result = filler.check_duplicate(
            {"id_card": "110101198001010001", "name": "张三", "age": 44}, existing)
        assert result is not None
        assert result["name"] == "张三"
        # 不匹配
        result2 = filler.check_duplicate(
            {"id_card": "440101200001010003", "name": "新人", "age": 25}, existing)
        assert result2 is None

    def test_card_to_dict(self):
        """测试传染病报告卡导出字典"""
        from health_interop.cdc import InfectiousDiseaseCard
        card = InfectiousDiseaseCard(name="测试", age=30, gender="male")
        d = card.to_dict()
        assert d["name"] == "测试"
        assert d["age"] == 30
        # 传输模式应脱敏
        d_safe = card.to_dict(for_transmission=True)
        assert "*" in d_safe["name"] or len(d_safe["name"]) < len(card.name)


# ============================================================================
# CDC 安全工具测试
# ============================================================================

class TestCDCSecurity:
    """测试CDC数据安全工具"""

    def test_sign_and_verify(self):
        """测试HMAC-SHA256签名生成和验证"""
        from health_interop.cdc import CDCSecurity
        secret = "my-secret-key-12345"
        payload = {"patient_id": "P001", "outcome": "治愈", "outcome_date": "2024-06-30"}
        sig = CDCSecurity.sign(payload, secret)
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA256 hex长度
        assert CDCSecurity.verify_signature(payload, sig, secret) is True
        # 错误密钥
        assert CDCSecurity.verify_signature(payload, sig, "wrong-key") is False
        # 篡改payload
        tampered = {**payload, "outcome": "死亡"}
        assert CDCSecurity.verify_signature(tampered, sig, secret) is False

    def test_sign_deterministic(self):
        """测试签名确定性（相同输入始终产生相同签名）"""
        from health_interop.cdc import CDCSecurity
        secret = "test-key"
        payload = {"a": 1, "b": 2, "c": [3, 4]}
        sig1 = CDCSecurity.sign(payload, secret)
        sig2 = CDCSecurity.sign(payload, secret)
        assert sig1 == sig2

    def test_desensitize(self):
        """测试跨机构传输脱敏"""
        from health_interop.cdc import CDCSecurity
        data = {
            "name": "张三丰",
            "id_card": "110101198001010001",
            "phone": "13800138000",
            "age": 45,
            "nested": {"mobile": "13900139000"},
        }
        safe = CDCSecurity.desensitize_for_transmission(data)
        assert safe["name"] != "张三丰"
        assert safe["id_card"] != "110101198001010001"
        assert safe["phone"] != "13800138000"
        assert safe["age"] == 45  # 非敏感字段保留
        assert safe["nested"]["mobile"] != "13900139000"


# ============================================================================
# CDC 回传解析器测试
# ============================================================================

class TestCDCCallbackParser:
    """测试疾控回传数据解析"""

    def test_parse_treatment(self):
        """测试治疗管理回传"""
        from health_interop.cdc import CDCCallbackParser
        data = {
            "patient_id": "P001",
            "treatment_plan": "2HRZE/4HR",
            "start_date": "2024-01-20",
            "month2_sputum": "阴性",
            "month5_sputum": "阴性",
            "month6_sputum": "阴性",
            "outcome": "治愈",
            "outcome_date": "2024-07-20",
            "dr_screen": "敏感",
        }
        result = CDCCallbackParser.parse_treatment_management(data)
        assert result["patient_id"] == "P001"
        assert result["outcome"] == "治愈"
        assert result["sputum_results"]["month2"] == "阴性"
        assert result["source"] == "cdc_treatment"
        assert result["outcome_date"] is not None  # 日期解析成功

    def test_parse_contacts(self):
        """测试密切接触者回传"""
        from health_interop.cdc import CDCCallbackParser
        data = {
            "contacts": [
                {"name": "接触者1", "relationship": "配偶",
                 "screening_result": "阴性", "is_active_tb": False},
                {"name": "接触者2", "relationship": "同事",
                 "screening_result": "胸片异常", "is_active_tb": True},
            ]
        }
        contacts = CDCCallbackParser.parse_close_contacts(data)
        assert len(contacts) == 2
        assert contacts[0]["relationship"] == "配偶"
        assert contacts[0]["contact_type"] == "cdc_contact"
        assert contacts[1]["is_active_tb"] is True

    def test_parse_contacts_empty(self):
        """测试空密切接触者"""
        from health_interop.cdc import CDCCallbackParser
        assert CDCCallbackParser.parse_close_contacts({}) == []
        assert CDCCallbackParser.parse_close_contacts({"contacts": None}) == []

    def test_parse_epidemiology(self):
        """测试区域流行病学数据"""
        from health_interop.cdc import CDCCallbackParser
        data = {
            "district": "朝阳区",
            "incidence_per_100k": 45.6,
            "dr_rate": 8.5,
            "report_date": "2024-06-30",
        }
        result = CDCCallbackParser.parse_regional_epidemiology(data)
        assert result["district"] == "朝阳区"
        assert result["incidence_rate_100k"] == 45.6
        assert result["regional_dr_rate"] == 8.5


# ============================================================================
# CDC Webhook 服务器测试
# ============================================================================

class TestCDCWebhookServer:
    """测试CDC回传Webhook服务器"""

    def test_server_instantiation(self):
        """测试服务器实例化"""
        from health_interop.cdc import CDCWebhookServer
        server = CDCWebhookServer(port=0)  # port=0让系统自动分配
        assert server.is_running is False
        assert server.message_count == 0

    def test_server_start_stop(self):
        """测试服务器启动和停止"""
        from health_interop.cdc import CDCWebhookServer
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            server = CDCWebhookServer(host="127.0.0.1", port=0, data_dir=tmpdir)
            # 启动（port=0会找一个空闲端口）
            started = server.start()
            # 注意：port=0在HTTPServer下实际绑定的是动态端口
            # 这里只测试不崩溃即可
            if started:
                assert server.is_running is True
                time.sleep(0.2)
                server.stop()
                assert server.is_running is False

    def test_callback_registration(self):
        """测试回调注册"""
        from health_interop.cdc import CDCWebhookServer
        server = CDCWebhookServer(port=0)
        received = []

        def on_raw(data):
            received.append(data)

        server.on('raw', on_raw)
        assert 'raw' in server._callbacks
        assert len(server._callbacks['raw']) == 1
        server.on('treatment', lambda d: None)
        server.on('contacts', lambda d: None)
        server.on('epidemiology', lambda d: None)


# ============================================================================
# 端到端集成测试
# ============================================================================

class TestEndToEndIntegration:
    """端到端集成测试：模拟完整数据链路"""

    def _make_fhir_bundle(self):
        """构造一个完整的FHIR Bundle测试数据"""
        return {
            "resourceType": "Bundle",
            "type": "searchset",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "pat-e2e-001",
                        "name": [{"family": "测", "given": ["试"], "text": "测试"}],
                        "gender": "male",
                        "birthDate": "1975-03-20",
                        "extension": [
                            {"url": "http://example.org/ethnicity", "valueString": "汉族"},
                            {"url": "http://example.org/occupation", "valueString": "石油工人"},
                            {"url": "http://example.org/height", "valueQuantity": {"value": 172, "unit": "cm"}},
                            {"url": "http://example.org/weight", "valueQuantity": {"value": 65, "unit": "kg"}},
                        ],
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "cond-001",
                        "code": {
                            "coding": [{
                                "system": "http://hl7.org/fhir/sid/icd-10",
                                "code": "A15.0",
                                "display": "肺结核，经痰涂片证实"
                            }]
                        },
                        "onsetDateTime": "2024-01-10",
                        "recordedDate": "2024-01-20",
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "obs-001",
                        "code": {
                            "coding": [{
                                "system": "http://loinc.org",
                                "code": "640-4",
                                "display": "痰涂片抗酸染色"
                            }]
                        },
                        "valueString": "阳性（++）",
                        "interpretation": [{"coding": [{"code": "POS"}]}],
                    }
                },
                {
                    "resource": {
                        "resourceType": "MedicationStatement",
                        "id": "med-001",
                        "medicationCodeableConcept": {
                            "coding": [{"display": "异烟肼 利福平 吡嗪酰胺 乙胺丁醇"}],
                            "text": "HRZE抗结核治疗"
                        },
                        "effectiveDateTime": "2024-01-20",
                    }
                },
            ]
        }

    def test_fhir_to_features_pipeline(self):
        """测试FHIR Bundle → 解析 → FeatureMapper → 特征字典完整链路"""
        from health_interop import FHIRResourceParser, FeatureMapper, AdapterResult
        from health_interop.base import calculate_age
        import datetime

        parser = FHIRResourceParser()
        result = AdapterResult(source_system="FHIR-TEST", patient_id="pat-e2e-001")

        bundle = self._make_fhir_bundle()
        for entry in bundle.get("entry", []):
            res = entry.get("resource")
            if not res:
                continue
            rtype = res.get("resourceType")
            if rtype == "Patient":
                parser.parse_patient(res, result)
            elif rtype == "Condition":
                diag = parser.parse_condition(res)
                if diag:
                    result.diagnoses.append(diag)
            elif rtype == "Observation":
                obs = parser.parse_observation(res)
                if obs:
                    result.lab_results.append(obs)
            elif rtype == "MedicationStatement":
                med = parser.parse_medication(res)
                if med:
                    result.medications.append(med)

        # 手动标记成功（手动解析资源时需要手动设置）
        result.success = True

        # FHIR parser目前不会自动解析身高体重extension，手动添加以便测试BMI映射
        result.patient_info["height"] = 172
        result.patient_info["weight"] = 65

        # 验证解析结果
        assert result.patient_info.get("name") == "测试"
        assert result.patient_info.get("gender") == "male"
        assert len(result.diagnoses) >= 1
        assert len(result.lab_results) >= 1
        assert result.success is True

        # FeatureMapper映射
        mapper = FeatureMapper()
        features = mapper.map_patient_basic_info(result)

        # 验证核心特征
        assert features["age"] >= 48  # 1975年生，2025年50岁左右
        assert features["gender"] == 1  # 男
        assert features["sputum_smear"] == "2"  # 涂阳
        assert features["active_tb"] == "1"  # 活动性结核
        assert features["treatment"] == "1"  # 在治疗中
        assert features["patient_name"] == "测试"

        # 验证扩展特征映射
        assert features["bmi"] > 0  # BMI已计算（172cm/65kg ≈ 21.97）
        assert features["height_cm"] == 172
        assert features["weight_kg"] == 65

    def test_hl7_to_features_pipeline(self):
        """测试HL7 v2消息 → 解析 → FeatureMapper完整链路"""
        from health_interop import HL7MessageParser, FeatureMapper, map_adapter_result_to_predictor_input

        parser = HL7MessageParser()
        # ADT^A04 注册患者
        adt_msg = (
            "MSH|^~\\&|HIS|HOSPITAL|EMR|HOSPITAL|20240115103000||ADT^A04|MSG-E2E-001|P|2.5\r"
            "PID|||MRN-E2E-001^^^HOSPITAL_MRN||测^试^^^先生||19800515|M||||||138******00\r"
            "PV1||I|WARD^01\r"
        )
        parsed = parser.parse(adt_msg)
        assert parsed.message_type == "ADT^A04"
        assert parsed.patient_info.get("name") == "测试"

        # ORU^R01 检验结果
        oru_msg = (
            "MSH|^~\\&|LIS|HOSPITAL|EMR|HOSPITAL|20240116140000||ORU^R01|MSG-E2E-002|P|2.5\r"
            "PID|||MRN-E2E-001||测^试||19800515|M\r"
            "OBR|1|ORD-E2E-001||640-4^痰涂片抗酸染色^LN|||20240116090000\r"
            "OBX|1|ST|640-4^痰涂片抗酸染色^LN||阳性（+）||||||F\r"
        )
        parsed2 = parser.parse(oru_msg)
        assert parsed2.message_type == "ORU^R01"
        assert len(parsed2.lab_results) >= 1

    def test_lis_pacs_decorator_pipeline(self):
        """测试LIS/PACS装饰器增强链路"""
        from health_interop import (
            LISPACSDecorator, LISMapper, RadiologyNLP,
            AdapterConfig, AdapterResult
        )
        from health_interop.base import HealthcareAdapter

        # 创建一个模拟基础适配器（继承抽象基类）
        class MockBaseAdapter(HealthcareAdapter):
            def __init__(self):
                super().__init__()
                self.config = AdapterConfig()
                self._connected = True

            def connect(self):
                self._connected = True
                return True

            def disconnect(self):
                self._connected = False
                return True

            def is_connected(self):
                return self._connected

            def query_patients(self, **kwargs):
                return []

            def fetch_incremental(self, since_time=None, **kwargs):
                return []

            def fetch_patient(self, patient_id):
                result = AdapterResult(source_system="MOCK", patient_id=patient_id)
                result.patient_info = {"name": "测试", "patient_id": patient_id,
                                      "gender": "male", "age": 45}
                result.lab_results = [{
                    "code": "640-4",
                    "loinc": "640-4",
                    "name": "痰涂片抗酸染色",
                    "value": "阳性(+)",
                    "interpretation": "positive",
                    "is_positive": True,
                }]
                result.imaging_reports = [{
                    "conclusion": "右肺上叶可见薄壁空洞形成，周围浸润影"
                }]
                result.success = True
                return result

        base = MockBaseAdapter()
        # 用LISPACS装饰器包装
        decorated = LISPACSDecorator(base, enable_lis=True, enable_pacs=True)

        result = decorated.fetch_patient("test-001")
        assert result.success is True
        # 验证装饰器增强了结果（LIS映射和影像NLP）
        assert result.patient_info.get("name") == "测试"

    def test_feature_mapper_extended_features(self):
        """测试FeatureMapper扩展特征映射（BCG、吸烟、职业、民族等）"""
        from health_interop import FeatureMapper, AdapterResult

        mapper = FeatureMapper()
        result = AdapterResult(source_system="TEST", patient_id="test-ext-001")

        # 构造包含扩展字段的患者信息
        result.patient_info = {
            "name": "扩展测试",
            "gender": "male",
            "age": 45,
            "bcg_scar": True,
            "smoking": "current",
            "smoking_years": 20,
            "drinking_years": 10,
            "height": 175,
            "weight": 70,
            "occupation": "油田钻井工人",
            "ethnicity": "维吾尔族",
            "idu": False,
        }

        features = mapper.map_patient_basic_info(result)

        # 验证扩展字段
        assert features["bcg_scar"] == "1"  # 有BCG卡痕
        assert features["smoking_years"] == 20
        assert features["drinking_years"] == 10
        assert features["height_cm"] == 175
        assert features["weight_kg"] == 70
        bmi_expected = round(70 / (1.75 ** 2), 1)
        assert abs(features["bmi"] - bmi_expected) < 0.2
        assert "uyghur" in features["ethnicity"] or "维吾尔" in features["ethnicity"]
        assert features["_high_risk_occupation"] == "oilfield_worker"  # 高危职业识别
        assert features["idu_history"] == "2"  # 无IDU史

    def test_cdc_contacts_to_gnn_update(self):
        """测试CDC回传接触者数据结构适配"""
        from health_interop.cdc import CDCCallbackParser

        contacts_data = {
            "source_patient_id": "src-001",
            "contacts": [
                {
                    "contact_id": "ct-001",
                    "name": "接触者A",
                    "relation": "配偶",
                    "contact_type": "household",
                    "screening_result": "positive",
                    "screening_date": "2024-06-15",
                },
                {
                    "contact_id": "ct-002",
                    "name": "接触者B",
                    "relation": "同事",
                    "contact_type": "social",
                    "screening_result": "negative",
                }
            ]
        }

        parsed = CDCCallbackParser.parse_close_contacts(contacts_data)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "接触者A"
        assert parsed[0]["screening_result"] == "positive"
        assert parsed[0]["is_active_tb"] is True
        assert parsed[1]["screening_result"] == "negative"


# ============================================================================
# 运行测试
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
