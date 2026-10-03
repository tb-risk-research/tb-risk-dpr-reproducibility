#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — AI 脱敏模块 (tb_risk.ai.redact)

测试 Layer 7: 发送给云端 LLM 前对文本/记录做脱敏，避免泄露患者身份信息。

脱敏规则：
- 身份证号（18 位）→ [REDACTED_ID]
- 手机号（11 位）→ [REDACTED_PHONE]
- 中文姓名（"患者：张三"/"姓名：李四" 等模式）→ [REDACTED_NAME]
- 邮箱 → [REDACTED_EMAIL]
"""

import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


class TestRedactText(unittest.TestCase):
    """redact_text() 文本脱敏测试"""

    def test_redacts_chinese_id_card(self):
        """脱敏 18 位身份证号"""
        from tb_risk.ai.redact import redact_text
        text = "身份证号：650102199001011234，患者男"
        result = redact_text(text)
        self.assertIn("[REDACTED_ID]", result)
        self.assertNotIn("650102199001011234", result)

    def test_redacts_phone_number(self):
        """脱敏 11 位手机号"""
        from tb_risk.ai.redact import redact_text
        text = "联系电话 13812345678，请回电"
        result = redact_text(text)
        self.assertIn("[REDACTED_PHONE]", result)
        self.assertNotIn("13812345678", result)

    def test_redacts_email(self):
        """脱敏邮箱"""
        from tb_risk.ai.redact import redact_text
        text = "邮箱：patient@example.com 请联系"
        result = redact_text(text)
        self.assertIn("[REDACTED_EMAIL]", result)
        self.assertNotIn("patient@example.com", result)

    def test_redacts_name_after_label(self):
        """脱敏'姓名'/'患者'标签后的中文姓名"""
        from tb_risk.ai.redact import redact_text
        text = "姓名：张三，年龄 35 岁"
        result = redact_text(text)
        self.assertIn("[REDACTED_NAME]", result)
        self.assertNotIn("张三", result)

    def test_redacts_patient_label_name(self):
        """脱敏'患者'标签后的姓名"""
        from tb_risk.ai.redact import redact_text
        text = "患者李四，男，35岁"
        result = redact_text(text)
        self.assertIn("[REDACTED_NAME]", result)
        self.assertNotIn("李四", result)

    def test_preserves_non_pii_text(self):
        """非 PII 文本保持原样"""
        from tb_risk.ai.redact import redact_text
        text = "患者男，35岁，痰涂片阳性，有空洞"
        result = redact_text(text)
        self.assertEqual(result, text)

    def test_handles_empty_text(self):
        """空文本 → 空字符串"""
        from tb_risk.ai.redact import redact_text
        self.assertEqual(redact_text(""), "")
        self.assertEqual(redact_text(None), "")

    def test_handles_multiple_pii(self):
        """同时存在多种 PII → 全部脱敏"""
        from tb_risk.ai.redact import redact_text
        text = "姓名：张三，电话 13812345678，身份证 650102199001011234"
        result = redact_text(text)
        self.assertIn("[REDACTED_NAME]", result)
        self.assertIn("[REDACTED_PHONE]", result)
        self.assertIn("[REDACTED_ID]", result)


class TestRedactRecord(unittest.TestCase):
    """redact_record() 记录脱敏测试"""

    def test_returns_new_dict_not_mutating_input(self):
        """返回新 dict，不修改输入"""
        from tb_risk.ai.redact import redact_record
        record = {'name': '张三', 'age': 35, 'phone': '13812345678'}
        original = dict(record)
        result = redact_record(record)
        # 输入不变
        self.assertEqual(record, original)
        # 返回新 dict
        self.assertIsNot(result, record)

    def test_redacts_name_field(self):
        """name 字段被脱敏"""
        from tb_risk.ai.redact import redact_record
        record = {'name': '张三', 'age': 35}
        result = redact_record(record)
        self.assertEqual(result['name'], '[REDACTED_NAME]')

    def test_redacts_phone_field(self):
        """phone 字段被脱敏"""
        from tb_risk.ai.redact import redact_record
        record = {'phone': '13812345678'}
        result = redact_record(record)
        self.assertEqual(result['phone'], '[REDACTED_PHONE]')

    def test_redacts_id_card_field(self):
        """id_card / id_number 字段被脱敏"""
        from tb_risk.ai.redact import redact_record
        record = {'id_card': '650102199001011234'}
        result = redact_record(record)
        self.assertEqual(result['id_card'], '[REDACTED_ID]')

    def test_preserves_other_fields(self):
        """非 PII 字段保持原样"""
        from tb_risk.ai.redact import redact_record
        record = {'age': 35, 'gender': '男', 'sputum_smear': 1}
        result = redact_record(record)
        self.assertEqual(result['age'], 35)
        self.assertEqual(result['gender'], '男')
        self.assertEqual(result['sputum_smear'], 1)

    def test_handles_empty_record(self):
        """空记录 → 空字典"""
        from tb_risk.ai.redact import redact_record
        self.assertEqual(redact_record({}), {})
        self.assertIsNone(redact_record(None))

    def test_redacts_pii_in_string_value(self):
        """字符串值中嵌入的 PII 也被脱敏"""
        from tb_risk.ai.redact import redact_record
        record = {'notes': '联系电话 13812345678 已确认'}
        result = redact_record(record)
        self.assertIn("[REDACTED_PHONE]", result['notes'])
        self.assertNotIn("13812345678", result['notes'])


if __name__ == '__main__':
    unittest.main()
