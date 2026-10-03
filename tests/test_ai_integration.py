#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tb_risk 测试套件 — AI 主类集成测试

覆盖缺陷改进清单 #1/#2/#4/#5/#19：
- #1: AiAssistantMixin 混入 TB_Risk_Assessment 主类 + __init__ 初始化
- #2: assess_risk() 后 last_assessment_result 被赋值
- #4: main.py _OPTIONAL_DEP_MAP 同步 AI 依赖
- #5: ai/__init__.py re-export 主要入口函数
- #19: 端到端集成测试（mock ai.qa.ask 验证主类调用 ask_ai）

设计：
- 类级别检查（hasattr/issubclass）避免实例化 Tkinter 主类
- main.py 与 ai/__init__.py 模块级别检查
- 端到端测试用 mock.patch 替换 ai.qa.ask，验证 queue 消息流
"""

import json
import os
import sys
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestAiAssistantMixinIntegrated(unittest.TestCase):
    """#1: AiAssistantMixin 被混入 TB_Risk_Assessment 主类"""

    def test_main_class_includes_ai_mixin(self):
        """TB_Risk_Assessment 继承 AiAssistantMixin"""
        from tb_risk.assessment import TB_Risk_Assessment
        from tb_risk.gui.ai_panel import AiAssistantMixin
        self.assertTrue(issubclass(TB_Risk_Assessment, AiAssistantMixin),
                        "TB_Risk_Assessment 必须继承 AiAssistantMixin")

    def test_main_class_has_ask_ai_method(self):
        """TB_Risk_Assessment 类有 ask_ai 方法"""
        from tb_risk.assessment import TB_Risk_Assessment
        self.assertTrue(hasattr(TB_Risk_Assessment, 'ask_ai'))
        self.assertTrue(hasattr(TB_Risk_Assessment, 'generate_ai_report'))
        self.assertTrue(hasattr(TB_Risk_Assessment, 'is_ai_available'))

    def test_gui_init_exports_ai_assistant_mixin(self):
        """gui/__init__.py 导出 AiAssistantMixin"""
        from tb_risk.gui import __all__ as gui_all
        self.assertIn('AiAssistantMixin', gui_all)
        # 也能从 tb_risk.gui 直接导入
        from tb_risk.gui import AiAssistantMixin as _M
        self.assertIsNotNone(_M)


class TestLastAssessmentResultAttribute(unittest.TestCase):
    """#2: TB_Risk_Assessment 有 last_assessment_result 属性"""

    def test_class_declares_last_assessment_result(self):
        """类或 __init__ 中声明 last_assessment_result 属性

        由于 Tkinter 实例化需要 display，这里只做类级别检查。
        实例化测试在 TestEndToEndAskAiFlow 中用 mock 完成。
        """
        from tb_risk.assessment import TB_Risk_Assessment
        # __init__ 应调用 _init_ai_assistant（该方法负责初始化 last_assessment_result）
        import inspect
        init_src = inspect.getsource(TB_Risk_Assessment.__init__)
        self.assertIn('_init_ai_assistant', init_src,
                      "__init__ 必须调用 _init_ai_assistant 初始化 AI 设施")
        # _init_ai_assistant 方法源码应包含 last_assessment_result 赋值
        init_ai_src = inspect.getsource(TB_Risk_Assessment._init_ai_assistant)
        self.assertIn('last_assessment_result', init_ai_src,
                      "_init_ai_assistant 必须初始化 last_assessment_result 属性")

    def test_assess_risk_assigns_last_assessment_result(self):
        """assess_risk 方法在评估完成后赋值 last_assessment_result"""
        from tb_risk.assessment import TB_Risk_Assessment
        import inspect
        src = inspect.getsource(TB_Risk_Assessment.assess_risk)
        self.assertIn('last_assessment_result', src,
                      "assess_risk 必须把结果赋给 self.last_assessment_result")


class TestAiSubpackageExports(unittest.TestCase):
    """#5: ai/__init__.py re-export 主要入口函数"""

    def test_exports_make_client(self):
        from tb_risk.ai import make_client
        self.assertTrue(callable(make_client))

    def test_exports_extract_with_llm(self):
        from tb_risk.ai import extract_with_llm
        self.assertTrue(callable(extract_with_llm))

    def test_exports_generate_report(self):
        from tb_risk.ai import generate_report
        self.assertTrue(callable(generate_report))

    def test_exports_ask(self):
        from tb_risk.ai import ask
        self.assertTrue(callable(ask))

    def test_exports_diagnose_record(self):
        from tb_risk.ai import diagnose_record
        self.assertTrue(callable(diagnose_record))

    def test_exports_batch_diagnose(self):
        from tb_risk.ai import batch_diagnose
        self.assertTrue(callable(batch_diagnose))

    def test_exports_llm_client_classes(self):
        from tb_risk.ai import LLMClient, LocalLLMClient
        self.assertTrue(isinstance(LLMClient, type))
        self.assertTrue(isinstance(LocalLLMClient, type))

    def test_all_list_includes_main_symbols(self):
        from tb_risk.ai import __all__ as ai_all
        for sym in ('AIConfig', 'PROVIDERS', 'DEFAULT_AI_CONFIG_FILE',
                    'make_client', 'extract_with_llm', 'generate_report',
                    'ask', 'diagnose_record', 'batch_diagnose',
                    'LLMClient', 'LocalLLMClient'):
            self.assertIn(sym, ai_all, f"ai.__all__ 必须包含 {sym}")


class TestMainOptionalDepsSync(unittest.TestCase):
    """#4: main.py _OPTIONAL_DEP_MAP 同步 AI 依赖"""

    def test_optional_dep_map_includes_httpx(self):
        from tb_risk.main import _OPTIONAL_DEP_MAP
        self.assertIn('httpx', _OPTIONAL_DEP_MAP)
        info = _OPTIONAL_DEP_MAP['httpx']
        self.assertEqual(info['group'], 'AI')

    def test_optional_dep_map_includes_tenacity(self):
        from tb_risk.main import _OPTIONAL_DEP_MAP
        self.assertIn('tenacity', _OPTIONAL_DEP_MAP)
        info = _OPTIONAL_DEP_MAP['tenacity']
        self.assertEqual(info['group'], 'AI')

    def test_optional_dep_map_includes_transformers(self):
        from tb_risk.main import _OPTIONAL_DEP_MAP
        self.assertIn('transformers', _OPTIONAL_DEP_MAP)
        info = _OPTIONAL_DEP_MAP['transformers']
        self.assertEqual(info['group'], 'AI 本地')

    def test_check_dependencies_includes_ai_modules(self):
        """_check_dependencies() 返回的 dict 包含 httpx/tenacity/transformers 键"""
        from tb_risk.main import _check_dependencies
        deps = _check_dependencies()
        self.assertIn('httpx', deps)
        self.assertIn('tenacity', deps)
        self.assertIn('transformers', deps)

    def test_print_dependency_status_includes_ai_category(self):
        """print_dependency_status 的 categories 含 AI 类别"""
        from tb_risk.main import print_dependency_status
        import inspect
        src = inspect.getsource(print_dependency_status)
        # 至少应包含 "AI" 字符串作为类别名
        self.assertIn('AI', src)

    def test_function_availability_summary_includes_ai(self):
        """功能可用性总结包含 AI 助手项"""
        from tb_risk.main import print_dependency_status
        import inspect
        src = inspect.getsource(print_dependency_status)
        self.assertIn('AI', src)


class TestEndToEndAskAiFlow(unittest.TestCase):
    """#19: 端到端集成测试 — TB_Risk_Assessment 实例 + mock ai.qa.ask

    验证：
    1. 主类实例化后 is_ai_available() 返回 False（默认 ai_enabled=False）
    2. ask_ai() 在 AI 不可用时返回 False
    3. ask_ai() 在 AI 可用且 mock ask 返回非空时，queue 收到 'answer' 消息
    """

    def _make_app_without_gui(self):
        """构造 TB_Risk_Assessment 实例但跳过 GUI 初始化

        TB_Risk_Assessment.__init__ 不依赖 Tkinter display（Tk 仅在 init_gui 时创建）。
        """
        from tb_risk.assessment import TB_Risk_Assessment
        app = TB_Risk_Assessment()
        return app

    def test_app_has_last_assessment_result_attribute(self):
        """实例化后 last_assessment_result 属性存在（初始为 falsy）"""
        app = self._make_app_without_gui()
        self.assertTrue(hasattr(app, 'last_assessment_result'))
        self.assertFalse(app.last_assessment_result)

    def test_app_has_ai_queue_and_callbacks(self):
        """实例化后 _ai_queue/_ai_active_jobs/_callbacks/_poll_timer_id 已初始化"""
        app = self._make_app_without_gui()
        import queue as _q
        self.assertIsInstance(app._ai_queue, _q.Queue)
        self.assertIsInstance(app._ai_active_jobs, set)
        self.assertIsInstance(app._callbacks, dict)
        self.assertIsNone(app._poll_timer_id)

    def test_is_ai_available_returns_false_by_default(self):
        """默认 AppConfig.ai_enabled=False → is_ai_available 返回 False"""
        app = self._make_app_without_gui()
        self.assertFalse(app.is_ai_available())

    def test_ask_ai_returns_false_when_disabled(self):
        """AI 禁用时 ask_ai 返回 False，不启动后台线程"""
        app = self._make_app_without_gui()
        result = app.ask_ai("为什么该接触者风险高？")
        self.assertFalse(result)

    def test_ask_ai_starts_thread_when_enabled(self):
        """启用 AI + mock ai.qa.ask → 后台线程把 answer 放入 queue"""
        app = self._make_app_without_gui()
        # 启用 AI
        app.app_config.ai_enabled = True

        received = []

        def fake_ask(question, context=None, knowledge=None, client=None, config=None):
            received.append((question, context))
            return "这是 mock 回答"

        with mock.patch('tb_risk.ai.qa.ask', side_effect=fake_ask):
            ok = app.ask_ai("为什么该接触者风险高？")
            self.assertTrue(ok)
            # 等待后台线程完成（最多 2 秒）
            import time
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if not app._ai_active_jobs:
                    break
                time.sleep(0.05)
            self.assertFalse(app._ai_active_jobs, "后台线程应在 2s 内完成")

        # 验证 mock ask 被调用
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][0], "为什么该接触者风险高？")

        # 验证 queue 中有 answer 消息
        msg = app._ai_queue.get_nowait()
        self.assertEqual(msg['type'], 'answer')
        self.assertIn('mock 回答', msg['content'])

    def test_ask_ai_with_last_assessment_result_context(self):
        """ask_ai 把 self.last_assessment_result 作为 context 传给 ai.qa.ask"""
        app = self._make_app_without_gui()
        app.app_config.ai_enabled = True
        app.last_assessment_result = {'patient_score': 75.5}

        captured = []

        def fake_ask(question, context=None, knowledge=None, client=None, config=None):
            captured.append(context)
            return "answer"

        with mock.patch('tb_risk.ai.qa.ask', side_effect=fake_ask):
            app.ask_ai("test")
            import time
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if not app._ai_active_jobs:
                    break
                time.sleep(0.05)

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0], {'patient_score': 75.5})

    def test_generate_ai_report_uses_last_assessment_result(self):
        """generate_ai_report 把 self.last_assessment_result 传给 ai.reporter.generate_report"""
        app = self._make_app_without_gui()
        app.app_config.ai_enabled = True
        app.last_assessment_result = {'patient_score': 80.0}

        captured = []

        def fake_report(result, client=None, config=None):
            captured.append(result)
            return "生成的报告内容"

        with mock.patch('tb_risk.ai.reporter.generate_report',
                        side_effect=fake_report):
            ok = app.generate_ai_report()
            self.assertTrue(ok)
            import time
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if not app._ai_active_jobs:
                    break
                time.sleep(0.05)

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0], {'patient_score': 80.0})
        # queue 中有 report 消息
        msg = app._ai_queue.get_nowait()
        self.assertEqual(msg['type'], 'report')
        self.assertIn('报告', msg['content'])


if __name__ == '__main__':
    unittest.main()
