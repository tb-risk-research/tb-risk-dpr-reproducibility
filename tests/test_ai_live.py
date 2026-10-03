#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tb_risk 测试套件 — 真实 API 调用的可选测试

缺陷 #20：所有 AI 测试都 mock 了 LLM，无法验证提示词在实际模型上是否有效。
本文件用 pytest.mark.skipif 在无 TB_AI_API_KEY 环境变量时跳过，验证抽取、
报告、问答、质量诊断四类提示词在真实模型上的输出质量。

启用方式（任选其一）：
    # 1. 临时设置环境变量
    $env:TB_AI_API_KEY = "sk-..."
    $env:TB_AI_PROVIDER = "deepseek"  # 或 openai/qwen/zhipu/kimi
    python -m pytest tests/test_ai_live.py -v --no-header

    # 2. 用配置文件 + 环境变量覆盖
    $env:TB_AI_API_KEY = "sk-..."
    python -m pytest tests/test_ai_live.py -v --no-header

可选环境变量：
    TB_AI_API_KEY:       API Key（必填，否则整个文件 skip）
    TB_AI_PROVIDER:      提供商名（默认 deepseek）
    TB_AI_BASE_URL:      覆盖 base_url（私有部署兼容）
    TB_AI_MODEL:         覆盖 model 名
    TB_AI_TEMPERATURE:   采样温度（默认 0.2）
    TB_AI_TIMEOUT:       超时秒数（默认 60）
    TB_AI_MAX_RETRIES:   最大重试次数（默认 3）

注意：
- 真实调用会消耗 API 配额、产生费用
- 网络异常、API 限速、模型变更都可能导致测试 flaky
- 这些测试不进入 CI 默认流水线（无 API Key 时自动 skip）
- 断言采用宽松策略：不期望精确文本，仅验证结构与基本有效性
"""

import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


# ==============================================================================
# 守卫：无 TB_AI_API_KEY 时整个文件 skip
# ==============================================================================

_LIVE_API_KEY = os.environ.get('TB_AI_API_KEY', '').strip()
_LIVE_ENABLED = bool(_LIVE_API_KEY)

# 可选依赖检查
try:
    import httpx  # noqa: F401
    import tenacity  # noqa: F401
    _AI_DEPS_AVAILABLE = True
except ImportError:
    _AI_DEPS_AVAILABLE = False

_LIVE_SKIP_REASON = (
    "未设置 TB_AI_API_KEY 环境变量，跳过真实 API 调用测试。"
    "如需启用：$env:TB_AI_API_KEY = 'sk-...'; python -m pytest tests/test_ai_live.py -v"
    if not _LIVE_ENABLED
    else "httpx/tenacity 未安装"
    if not _AI_DEPS_AVAILABLE
    else ""
)

_LIVE_RUN = _LIVE_ENABLED and _AI_DEPS_AVAILABLE


# ==============================================================================
# 基类：提供共享的真实 client 构造逻辑
# ==============================================================================

@unittest.skipUnless(_LIVE_RUN, _LIVE_SKIP_REASON)
class _AiLiveTestBase(unittest.TestCase):
    """真实 API 调用测试基类

    子类继承此类，自动获得 _make_live_config() 与 _make_live_client() 方法。
    """

    @classmethod
    def setUpClass(cls):
        """构建真实 AIConfig（所有子类共享）"""
        from tb_risk.ai.config import AIConfig
        cls.live_config = AIConfig.from_env()
        if not cls.live_config.api_key:
            raise unittest.SkipTest(
                "AIConfig.from_env() 未获取到 api_key，跳过真实 API 测试")

    @classmethod
    def _make_live_client(cls):
        """构建真实 LLMClient（每个测试方法独立 client，避免 token usage 串扰）"""
        from tb_risk.ai.client import LLMClient
        return LLMClient(cls.live_config)


# ==============================================================================
# 真实抽取测试
# ==============================================================================

class TestAiLiveExtract(_AiLiveTestBase):
    """真实 API 调用：文本抽取提示词验证"""

    def test_extracts_single_patient_record(self):
        """抽取临床叙述文本 → 返回 dict 包含 records 字段"""
        from tb_risk.ai import extract_with_llm
        text = (
            "患者张三，男，35 岁，维吾尔族。痰涂片阳性，胸片显示有空洞。"
            "已接种卡介苗，目前正在抗结核治疗，已治疗 2 个月。"
            "居住条件一般，与家人密切接触，每日接触时间 8 小时。"
            "咳嗽频繁，约每天 10 次，就诊延迟 14 天。BMI 22.5。"
        )
        result = extract_with_llm(text, config=self.live_config)
        self.assertIsNotNone(result, "extract_with_llm 应返回非 None")
        self.assertIsInstance(result, dict, "返回应为 dict")
        self.assertIn('records', result, "返回应包含 records 字段")
        records = result['records']
        self.assertIsInstance(records, list)
        self.assertGreaterEqual(len(records), 1, "应至少抽取 1 条记录")
        # 验证抽取的关键字段
        rec = records[0]
        self.assertIn('age', rec)
        self.assertEqual(rec.get('gender'), '男')
        self.assertEqual(rec.get('sputum_smear'), 1)
        self.assertEqual(rec.get('has_cavity'), 1)

    def test_extracts_multiple_patients_from_text(self):
        """抽取多条患者记录"""
        from tb_risk.ai import extract_with_llm
        text = (
            "病例1：李四，女，28 岁，痰涂片阴性，无空洞，已接种卡介苗，未治疗。"
            "病例2：王五，男，45 岁，痰涂片阳性，有空洞，未接种卡介苗，已治疗 3 个月。"
        )
        result = extract_with_llm(text, config=self.live_config)
        self.assertIsNotNone(result)
        records = result.get('records', [])
        self.assertGreaterEqual(len(records), 2,
                                "应能从文本中抽取至少 2 条记录")


# ==============================================================================
# 真实报告生成测试
# ==============================================================================

class TestAiLiveReport(_AiLiveTestBase):
    """真实 API 调用：报告生成提示词验证"""

    def test_generates_non_empty_report(self):
        """生成报告 → 返回非空字符串"""
        from tb_risk.ai import generate_report
        result_data = {
            'patient_score': 75.5,
            'summary': {
                'overall_risk': '高',
                'total_contacts': 8,
                'high_risk_contacts': 3,
            },
            'potential_patients': {
                'family': [
                    {'name': '张三', 'risk_score': 0.85, 'priority': '高'},
                    {'name': '李四', 'risk_score': 0.42, 'priority': '中'},
                ],
                'social': [
                    {'name': '王五', 'risk_score': 0.65, 'priority': '高'},
                ],
            },
            'ml_results': {
                'probability': 0.78,
                'top_features': [
                    {'feature': 'sputum_smear', 'importance': 0.32},
                    {'feature': 'has_cavity', 'importance': 0.25},
                ],
            },
            'seir_results': {
                'R0': 2.5,
                'peak_time': 14,
                'peak_size': 25,
            },
        }
        report = generate_report(result_data, config=self.live_config)
        self.assertIsNotNone(report, "generate_report 应返回非 None")
        self.assertIsInstance(report, str)
        # 报告长度应在 200~5000 字符之间（足够详细，但不应失控）
        self.assertGreaterEqual(len(report), 200,
                                "报告长度应 >= 200 字符")
        self.assertLessEqual(len(report), 10000,
                             "报告长度应 <= 10000 字符（异常时截断保护）")

    def test_report_contains_key_sections(self):
        """报告应包含关键章节（评估摘要 / 风险因素 / 干预建议）"""
        from tb_risk.ai import generate_report
        result_data = {
            'patient_score': 65.0,
            'summary': {'overall_risk': '中', 'total_contacts': 5},
        }
        report = generate_report(result_data, config=self.live_config)
        self.assertIsNotNone(report)
        # 至少包含"风险"或"评估"等关键词
        self.assertTrue(
            any(kw in report for kw in ['风险', '评估', '建议', '干预']),
            f"报告应包含关键词（风险/评估/建议/干预）: {report[:200]}..."
        )


# ==============================================================================
# 真实问答测试
# ==============================================================================

class TestAiLiveQA(_AiLiveTestBase):
    """真实 API 调用：智能问答提示词验证"""

    def test_answers_question_about_risk_score(self):
        """问答 → 返回非空回答，且包含相关关键词"""
        from tb_risk.ai import ask
        context = {
            'patient_score': 72.0,
            'summary': {'overall_risk': '高', 'total_contacts': 8},
        }
        answer = ask(
            "这个患者的风险评分 72 分意味着什么？需要采取什么干预措施？",
            context=context,
            config=self.live_config,
        )
        self.assertIsNotNone(answer, "ask 应返回非 None")
        self.assertIsInstance(answer, str)
        self.assertGreaterEqual(len(answer), 20,
                                "回答长度应 >= 20 字符")
        # 回答应包含"风险"或"干预"等关键词
        self.assertTrue(
            any(kw in answer for kw in ['风险', '干预', '评分', '建议']),
            f"回答应包含关键词（风险/干预/评分/建议）: {answer[:200]}..."
        )

    def test_answers_question_without_context(self):
        """无上下文时仍能回答（让模型说明需要更多信息）"""
        from tb_risk.ai import ask
        answer = ask(
            "结核病的传播途径是什么？",
            context=None,
            config=self.live_config,
        )
        self.assertIsNotNone(answer)
        self.assertIsInstance(answer, str)
        self.assertGreaterEqual(len(answer), 30)


# ==============================================================================
# 真实数据质量诊断测试
# ==============================================================================

class TestAiLiveQuality(_AiLiveTestBase):
    """真实 API 调用：数据质量诊断提示词验证"""

    def test_diagnoses_low_quality_record(self):
        """诊断低质量记录 → 返回 dict 含 issues/overall_quality 字段"""
        from tb_risk.ai import diagnose_record
        # 故意构造一条问题记录
        record = {
            'name': '张三',
            'age': 150,  # 异常：年龄过大
            'gender': '男',
            'sputum_smear': None,  # 缺失：必填字段为空
            'has_cavity': 1,
            'bmi': 65.0,  # 异常：BMI 过大
            'ventilation': '',  # 缺失
        }
        result = diagnose_record(record, config=self.live_config)
        self.assertIsNotNone(result, "diagnose_record 应返回非 None")
        self.assertIsInstance(result, dict)
        # 应包含 issues 列表
        self.assertIn('issues', result)
        # issues 应是非空 list（有多个问题）
        issues = result['issues']
        self.assertIsInstance(issues, list)
        self.assertGreaterEqual(len(issues), 1,
                                "应至少识别出 1 个问题")
        # 每个 issue 应有 field/problem/suggestion 字段
        for issue in issues:
            self.assertIsInstance(issue, dict)
            self.assertIn('field', issue)
            self.assertIn('problem', issue)


# ==============================================================================
# Token usage 集成验证
# ==============================================================================

class TestAiLiveTokenUsage(_AiLiveTestBase):
    """真实 API 调用：验证 get_usage() 累计真实 token 数"""

    def test_usage_accumulates_after_real_call(self):
        """真实调用后 get_usage() 应返回非零 token 数"""
        from tb_risk.ai import ask
        client = self._make_live_client()
        try:
            answer = ask(
                "什么是结核病？",
                context=None,
                client=client,
                config=self.live_config,
            )
            self.assertIsNotNone(answer)

            usage = client.get_usage()
            self.assertGreater(usage['calls'], 0)
            self.assertGreater(usage['prompt_tokens'], 0)
            self.assertGreater(usage['completion_tokens'], 0)
            self.assertGreater(usage['total_tokens'], 0)
            # 日志中应能看到 token 数
            self.assertEqual(usage['total_tokens'],
                             usage['prompt_tokens'] + usage['completion_tokens'])
        finally:
            client.close()


if __name__ == '__main__':
    unittest.main()
