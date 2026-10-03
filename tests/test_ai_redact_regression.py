#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
tb_risk 测试套件 — AI 脱敏回归与改进 (缺陷 #10/#11/#21/#12)

覆盖：
- #10: 脱敏正则在中文环境下 \b 边界不可靠 → 改用零宽断言 (?<!\d)/(?!\d)
       身份证/手机号正则在中文上下文（无空格分隔）应仍能正确匹配
- #11: _PII_FIELD_KEYWORDS 短词（mail/phone/name/tel）匹配过宽
       mailbox/phone_model/name_alias 等非 PII 字段不应被误脱敏
- #21: 脱敏回归测试（邮箱/手机号边界用例）
- #12: 本地模式跳过脱敏（LocalLLMClient 数据不出本机）
"""

import os
import sys
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


# ==============================================================================
# #10: 中文环境下身份证/手机号正则边界
# ==============================================================================

class TestRedactRegexChineseBoundary(unittest.TestCase):
    """脱敏正则在中文环境下的边界处理"""

    def test_redacts_id_card_adjacent_to_chinese_chars(self):
        """身份证号紧贴中文字符（无空格）也应被脱敏

        原 \\b 在中文字符处不构成 word boundary，可能漏匹配。
        """
        from tb_risk.ai.redact import redact_text
        # 中文-身份证-中文（无空格）
        text = "身份证号650102199001011234已登记"
        result = redact_text(text)
        self.assertIn("[REDACTED_ID]", result)
        self.assertNotIn("650102199001011234", result)

    def test_redacts_id_card_with_punctuation_boundary(self):
        """身份证号紧贴标点符号也应被脱敏"""
        from tb_risk.ai.redact import redact_text
        text = "证件号:650102199001011234,请核验"
        result = redact_text(text)
        self.assertIn("[REDACTED_ID]", result)
        self.assertNotIn("650102199001011234", result)

    def test_redacts_id_card_with_space_boundaries(self):
        """身份证号两端有空格（标准 \\b 场景）也应被脱敏"""
        from tb_risk.ai.redact import redact_text
        text = "身份证号 650102199001011234 已登记"
        result = redact_text(text)
        self.assertIn("[REDACTED_ID]", result)
        self.assertNotIn("650102199001011234", result)

    def test_does_not_redact_17_digit_number(self):
        """17 位数字（非 18 位身份证）不应被脱敏为身份证"""
        from tb_risk.ai.redact import redact_text
        text = "订单号 12345678901234567 已生成"
        result = redact_text(text)
        self.assertNotIn("[REDACTED_ID]", result)
        self.assertIn("12345678901234567", result)

    def test_does_not_redact_19_digit_number(self):
        """19 位数字（超长）不应被脱敏"""
        from tb_risk.ai.redact import redact_text
        text = "银行卡号 6501021990010112345 太长"
        result = redact_text(text)
        # 19 位数字中包含 18 位身份证子串，但若用零宽断言 (?!\d) 应不匹配
        # 期望：银行卡号原样保留（不被部分脱敏）
        self.assertNotIn("[REDACTED_ID]", result)

    def test_redacts_phone_adjacent_to_chinese_chars(self):
        """手机号紧贴中文字符也应被脱敏"""
        from tb_risk.ai.redact import redact_text
        text = "电话13812345678已记录"
        result = redact_text(text)
        self.assertIn("[REDACTED_PHONE]", result)
        self.assertNotIn("13812345678", result)

    def test_redacts_phone_with_punctuation(self):
        """手机号紧贴标点也应被脱敏"""
        from tb_risk.ai.redact import redact_text
        text = "联系方式:13812345678,请回复"
        result = redact_text(text)
        self.assertIn("[REDACTED_PHONE]", result)
        self.assertNotIn("13812345678", result)

    def test_does_not_redact_10_digit_number(self):
        """10 位数字（非 11 位手机号）不应被脱敏"""
        from tb_risk.ai.redact import redact_text
        text = "编号 1381234567 不应脱敏"
        result = redact_text(text)
        self.assertNotIn("[REDACTED_PHONE]", result)

    def test_does_not_redact_12_digit_number(self):
        """12 位数字（超长）不应被脱敏"""
        from tb_risk.ai.redact import redact_text
        text = "订单 138123456789 太长"
        result = redact_text(text)
        self.assertNotIn("[REDACTED_PHONE]", result)

    def test_redacts_multiple_id_cards_in_mixed_context(self):
        """混合上下文中多个身份证号全部脱敏"""
        from tb_risk.ai.redact import redact_text
        text = ("张三身份证650102199001011234，"
                "李四身份证 11010519881234567X 已核验")
        result = redact_text(text)
        self.assertIn("[REDACTED_ID]", result)
        self.assertNotIn("650102199001011234", result)
        self.assertNotIn("11010519881234567X", result)


# ==============================================================================
# #11: 字段名歧义字段不应被误脱敏
# ==============================================================================

class TestRedactAmbiguousFields(unittest.TestCase):
    """歧义字段名（mailbox/phone_model/name_alias 等）不应被误脱敏"""

    def test_mailbox_field_not_redacted(self):
        """mailbox 字段（含 mail 子串）不应被脱敏"""
        from tb_risk.ai.redact import redact_record
        record = {'mailbox': 'inbox_001', 'age': 35}
        result = redact_record(record)
        # mailbox 不应被替换为 [REDACTED_EMAIL]
        self.assertEqual(result['mailbox'], 'inbox_001')

    def test_phone_model_field_not_redacted(self):
        """phone_model 字段（含 phone 子串）不应被脱敏"""
        from tb_risk.ai.redact import redact_record
        record = {'phone_model': 'iPhone 15', 'age': 35}
        result = redact_record(record)
        self.assertEqual(result['phone_model'], 'iPhone 15')

    def test_name_alias_field_not_redacted(self):
        """name_alias 字段（含 name 子串）不应被脱敏"""
        from tb_risk.ai.redact import redact_record
        record = {'name_alias': '张三的别名', 'age': 35}
        result = redact_record(record)
        self.assertEqual(result['name_alias'], '张三的别名')

    def test_tel_field_not_redacted_when_ambiguous(self):
        """tel_index 等歧义字段不应被脱敏

        tel 作为短词歧义大（如 telescope, telephone_box 等），
        应仅完全相等匹配，不带前缀/后缀。
        """
        from tb_risk.ai.redact import redact_record
        record = {'tel_index': 'T001', 'age': 35}
        result = redact_record(record)
        self.assertEqual(result['tel_index'], 'T001')

    def test_mail_field_redacted_when_exact_match(self):
        """mail 字段完全相等时应被脱敏"""
        from tb_risk.ai.redact import redact_record
        record = {'mail': 'patient@example.com'}
        result = redact_record(record)
        self.assertEqual(result['mail'], '[REDACTED_EMAIL]')

    def test_tel_field_redacted_when_exact_match(self):
        """tel 字段完全相等时应被脱敏"""
        from tb_risk.ai.redact import redact_record
        record = {'tel': '010-12345678'}
        result = redact_record(record)
        self.assertEqual(result['tel'], '[REDACTED_PHONE]')

    def test_phone_field_redacted_when_exact_match(self):
        """phone 字段完全相等时应被脱敏"""
        from tb_risk.ai.redact import redact_record
        record = {'phone': '13812345678'}
        result = redact_record(record)
        self.assertEqual(result['phone'], '[REDACTED_PHONE]')

    def test_mobile_field_redacted_when_exact_match(self):
        """mobile 字段完全相等时应被脱敏"""
        from tb_risk.ai.redact import redact_record
        record = {'mobile': '13812345678'}
        result = redact_record(record)
        self.assertEqual(result['mobile'], '[REDACTED_PHONE]')

    def test_name_field_redacted_when_exact_match(self):
        """name 字段完全相等时应被脱敏"""
        from tb_risk.ai.redact import redact_record
        record = {'name': '张三'}
        result = redact_record(record)
        self.assertEqual(result['name'], '[REDACTED_NAME]')

    def test_patient_name_field_redacted(self):
        """patient_name 字段完全相等时应被脱敏"""
        from tb_risk.ai.redact import redact_record
        record = {'patient_name': '李四'}
        result = redact_record(record)
        self.assertEqual(result['patient_name'], '[REDACTED_NAME]')

    def test_id_card_field_redacted(self):
        """id_card 字段完全相等时应被脱敏"""
        from tb_risk.ai.redact import redact_record
        record = {'id_card': '650102199001011234'}
        result = redact_record(record)
        self.assertEqual(result['id_card'], '[REDACTED_ID]')

    def test_email_field_redacted_when_exact_match(self):
        """email 字段完全相等时应被脱敏"""
        from tb_risk.ai.redact import redact_record
        record = {'email': 'patient@example.com'}
        result = redact_record(record)
        self.assertEqual(result['email'], '[REDACTED_EMAIL]')

    def test_contact_phone_with_underscore_redacted(self):
        """contact_phone（下划线前缀 + phone）应被脱敏"""
        from tb_risk.ai.redact import redact_record
        record = {'contact_phone': '13812345678'}
        result = redact_record(record)
        self.assertEqual(result['contact_phone'], '[REDACTED_PHONE]')

    def test_phone_contact_with_underscore_suffix_not_redacted(self):
        """phone_contact（phone + 下划线后缀，keyword 在前）不应被脱敏

        按新规则：keyword 在字段名末尾（带下划线前缀）才匹配，
        keyword 在字段名开头（带下划线后缀）不再匹配，
        避免 phone_model / phone_contact / name_alias 等误脱敏。
        """
        from tb_risk.ai.redact import redact_record
        record = {'phone_contact': 'some_contact_info'}
        result = redact_record(record)
        # phone_contact 不应被脱敏为 [REDACTED_PHONE]
        self.assertEqual(result['phone_contact'], 'some_contact_info')

    def test_contact_mail_not_redacted_for_ambiguous_keyword(self):
        """contact_mail（mail 歧义词 + 下划线前缀）不应被脱敏

        mail/tel 作为歧义词仅做完全相等匹配，contact_mail 不应被脱敏。
        """
        from tb_risk.ai.redact import redact_record
        record = {'contact_mail': 'inbox_001'}
        result = redact_record(record)
        self.assertEqual(result['contact_mail'], 'inbox_001')


# ==============================================================================
# #21: 邮箱/手机号回归边界用例
# ==============================================================================

class TestRedactRegression(unittest.TestCase):
    """脱敏回归测试"""

    def test_redacts_email_with_plus_sign(self):
        """邮箱含 + 号（如 user+tag@domain）应被脱敏"""
        from tb_risk.ai.redact import redact_text
        text = "邮箱：user+tag@example.com 已发送"
        result = redact_text(text)
        self.assertIn("[REDACTED_EMAIL]", result)
        self.assertNotIn("user+tag@example.com", result)

    def test_redacts_email_with_dots_in_local_part(self):
        """邮箱本地部分含点号应被脱敏"""
        from tb_risk.ai.redact import redact_text
        text = "邮箱：first.last@example.com 已发送"
        result = redact_text(text)
        self.assertIn("[REDACTED_EMAIL]", result)
        self.assertNotIn("first.last@example.com", result)

    def test_redacts_multiple_emails_in_text(self):
        """文本中多个邮箱全部脱敏"""
        from tb_risk.ai.redact import redact_text
        text = "发件人 a@x.com 收件人 b@y.com 抄送 c@z.com"
        result = redact_text(text)
        self.assertEqual(result.count("[REDACTED_EMAIL]"), 3)

    def test_redacts_multiple_phones_in_text(self):
        """文本中多个手机号全部脱敏"""
        from tb_risk.ai.redact import redact_text
        text = "联系电话 13812345678 和 13987654321 都已登记"
        result = redact_text(text)
        self.assertEqual(result.count("[REDACTED_PHONE]"), 2)

    def test_redacts_mixed_pii_in_complex_text(self):
        """复杂文本中身份证/手机/邮箱/姓名混合全部脱敏"""
        from tb_risk.ai.redact import redact_text
        text = ("姓名：张三，身份证 650102199001011234，"
                "电话 13812345678，邮箱 patient@example.com")
        result = redact_text(text)
        self.assertIn("[REDACTED_NAME]", result)
        self.assertIn("[REDACTED_ID]", result)
        self.assertIn("[REDACTED_PHONE]", result)
        self.assertIn("[REDACTED_EMAIL]", result)
        # 原始 PII 不应残留
        self.assertNotIn("张三", result)
        self.assertNotIn("650102199001011234", result)
        self.assertNotIn("13812345678", result)
        self.assertNotIn("patient@example.com", result)

    def test_preserves_non_pii_numbers(self):
        """非 PII 数字（如年龄、体温）不应被误脱敏"""
        from tb_risk.ai.redact import redact_text
        text = "患者男，35岁，体温36.5℃，血压120/80"
        result = redact_text(text)
        self.assertEqual(result, text)  # 无 PII，原样返回

    def test_redacts_id_card_with_x_suffix(self):
        """身份证末位为 X（大写）应被脱敏"""
        from tb_risk.ai.redact import redact_text
        text = "身份证 11010519881234567X 已核验"
        result = redact_text(text)
        self.assertIn("[REDACTED_ID]", result)
        self.assertNotIn("11010519881234567X", result)

    def test_redacts_id_card_with_lowercase_x_suffix(self):
        """身份证末位为 x（小写）应被脱敏"""
        from tb_risk.ai.redact import redact_text
        text = "身份证 11010519881234567x 已核验"
        result = redact_text(text)
        self.assertIn("[REDACTED_ID]", result)
        self.assertNotIn("11010519881234567x", result)


# ==============================================================================
# #12: 本地模式跳过脱敏
# ==============================================================================

class TestLocalModeSkipRedact(unittest.TestCase):
    """本地模式（LocalLLMClient）数据不出本机，可跳过脱敏"""

    def test_reporter_skips_redact_in_local_mode(self):
        """reporter.generate_report 在本地模式下应跳过脱敏

        验证：传给 LLM 的上下文中包含原始姓名（而非 [REDACTED_NAME]）
        """
        from tb_risk.ai.reporter import generate_report
        from tb_risk.ai.config import AIConfig

        captured = {}

        class _Client:
            def chat(self, messages, **kwargs):
                captured['content'] = ''.join(
                    m['content'] for m in messages if m['role'] == 'user')
                return "报告内容"

        # 本地模式 config
        local_cfg = AIConfig(local_model_path='/fake/path')
        result = {
            'patient_score': 75.0,
            'potential_patients': {
                'family': [
                    {'name': '张三', 'risk_score': 80.0, 'priority': '高'},
                ],
            },
        }
        generate_report(result, client=_Client(), config=local_cfg)
        # 本地模式下应保留原始姓名
        self.assertIn('张三', captured['content'],
                      "本地模式下应跳过脱敏，保留原始姓名")
        self.assertNotIn('[REDACTED_NAME]', captured['content'])

    def test_reporter_redacts_in_cloud_mode(self):
        """reporter.generate_report 在云端模式下应执行脱敏"""
        from tb_risk.ai.reporter import generate_report
        from tb_risk.ai.config import AIConfig

        captured = {}

        class _Client:
            def chat(self, messages, **kwargs):
                captured['content'] = ''.join(
                    m['content'] for m in messages if m['role'] == 'user')
                return "报告内容"

        # 云端模式 config
        cloud_cfg = AIConfig(api_key='sk-test')
        result = {
            'patient_score': 75.0,
            'potential_patients': {
                'family': [
                    {'name': '张三', 'risk_score': 80.0, 'priority': '高'},
                ],
            },
        }
        generate_report(result, client=_Client(), config=cloud_cfg)
        # 云端模式下应脱敏姓名
        self.assertIn('[REDACTED_NAME]', captured['content'])
        self.assertNotIn('张三', captured['content'])

    def test_qa_skips_redact_in_local_mode(self):
        """qa.ask 在本地模式下应跳过脱敏"""
        from tb_risk.ai.qa import ask
        from tb_risk.ai.config import AIConfig

        captured = {}

        class _Client:
            def chat(self, messages, **kwargs):
                captured['content'] = ''.join(
                    m['content'] for m in messages if m['role'] == 'user')
                return "回答"

        local_cfg = AIConfig(local_model_path='/fake/path')
        context = {
            'patient_score': 75.0,
            'potential_patients': {
                'family': [
                    {'name': '李四', 'risk_score': 80.0, 'priority': '高'},
                ],
            },
        }
        ask("患者风险高吗？", context=context, client=_Client(), config=local_cfg)
        self.assertIn('李四', captured['content'])
        self.assertNotIn('[REDACTED_NAME]', captured['content'])

    def test_quality_skips_redact_in_local_mode(self):
        """quality.diagnose_record 在本地模式下应跳过脱敏"""
        from tb_risk.ai.quality import diagnose_record
        from tb_risk.ai.config import AIConfig

        captured = {}

        class _Client:
            def chat_json(self, messages, schema=None, **kwargs):
                captured['content'] = ''.join(
                    m['content'] for m in messages if m['role'] == 'user')
                return {'issues': [], 'overall_quality': '高', 'summary': ''}

        local_cfg = AIConfig(local_model_path='/fake/path')
        record = {'name': '王五', 'age': 35, 'phone': '13812345678'}
        diagnose_record(record, client=_Client(), config=local_cfg)
        # 本地模式下应保留原始 PII
        self.assertIn('王五', captured['content'])
        self.assertIn('13812345678', captured['content'])
        self.assertNotIn('[REDACTED_NAME]', captured['content'])
        self.assertNotIn('[REDACTED_PHONE]', captured['content'])


if __name__ == '__main__':
    unittest.main()
