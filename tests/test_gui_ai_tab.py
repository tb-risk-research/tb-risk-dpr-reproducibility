#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — GUI AI 助手子标签页 (tb_risk.gui.results_panel.ai_tab)

缺陷 #3 测试：验证 AiTabMixin 提供的 UI 入口控件与回调绑定。

测试策略：
- 不依赖真实 Tk 渲染（避免 CI 无 display 时失败）
- 用 mock Text / StringVar 模拟 Tkinter 控件
- 验证回调方法 _on_ask_ai_clicked / _on_generate_ai_report_clicked /
  _on_ai_answer / _on_ai_report / _on_ai_error 的业务逻辑
- 验证 _init_ai_tab 正确覆盖 _callbacks（用真实 Tk root，无 display 时跳过）
"""

import os
import sys
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


# ==============================================================================
# Mock Tkinter 控件
# ==============================================================================

class _MockText:
    """模拟 tkinter.Text 控件"""

    def __init__(self, content=''):
        self._content = content

    def get(self, start, end):
        return self._content

    def insert(self, index, text):
        self._content += text

    def delete(self, start, end):
        self._content = ''

    def see(self, index):
        pass

    def configure(self, **kwargs):
        pass

    def yview(self, *args):
        pass


class _MockStringVar:
    """模拟 tkinter.StringVar"""

    def __init__(self, value=''):
        self._value = value

    def set(self, value):
        self._value = value

    def get(self):
        return self._value


# ==============================================================================
# 加载 AiTabMixin 模块
# ==============================================================================

def _load_ai_tab_module():
    """直接加载 gui/results_panel/ai_tab.py 模块

    绕过 gui/__init__.py 和 results_panel/__init__.py 的 Tkinter 依赖链，
    仅加载 AiTabMixin 类定义。
    """
    import importlib.util
    ai_tab_path = os.path.join(
        _PROJECT_ROOT, 'gui', 'results_panel', 'ai_tab.py')
    spec = importlib.util.spec_from_file_location(
        'gui.results_panel.ai_tab', ai_tab_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _make_ai_tab_mixin(ai_available=True, last_result=None):
    """构造 AiTabMixin 实例（不继承 Tk，不调用 _init_ai_tab）

    手动设置回调方法所需的属性，便于直接测试业务逻辑。
    """
    m = _load_ai_tab_module()
    AiTabMixin = m.AiTabMixin
    instance = AiTabMixin.__new__(AiTabMixin)
    # Mock 控件
    instance.ai_question_text = _MockText()
    instance.ai_output_text = _MockText()
    instance.ai_status_var = _MockStringVar(value="AI 状态：就绪")
    # 评估结果
    instance.last_assessment_result = last_result if last_result is not None \
        else {'patient_score': 75.0}
    # _callbacks dict
    instance._callbacks = {}
    # Mock AiAssistantMixin 方法
    instance.ask_ai = mock.MagicMock(return_value=True)
    instance.generate_ai_report = mock.MagicMock(return_value=True)
    instance.is_ai_available = mock.MagicMock(return_value=ai_available)
    return instance


# ==============================================================================
# 回调方法测试（不依赖真实 Tk）
# ==============================================================================

class TestOnAskAiClicked(unittest.TestCase):
    """_on_ask_ai_clicked() 按钮回调测试"""

    def test_reads_question_and_calls_ask_ai(self):
        """读取问题并调用 ask_ai()"""
        m = _make_ai_tab_mixin(ai_available=True)
        m.ai_question_text._content = "为什么风险高?"
        m._on_ask_ai_clicked()
        m.ask_ai.assert_called_once_with("为什么风险高?")

    def test_sets_status_to_thinking_when_started(self):
        """启动成功 → 状态变为 '正在思考...'"""
        m = _make_ai_tab_mixin(ai_available=True)
        m.ai_question_text._content = "为什么?"
        m._on_ask_ai_clicked()
        self.assertIn("正在思考", m.ai_status_var.get())

    def test_appends_question_to_output(self):
        """问题被追加到输出区"""
        m = _make_ai_tab_mixin(ai_available=True)
        m.ai_question_text._content = "为什么风险高?"
        m._on_ask_ai_clicked()
        self.assertIn("为什么风险高?", m.ai_output_text._content)

    def test_warns_when_question_empty(self):
        """空问题 → 输出警告，不调用 ask_ai()"""
        m = _make_ai_tab_mixin(ai_available=True)
        m.ai_question_text._content = ""
        m._on_ask_ai_clicked()
        m.ask_ai.assert_not_called()
        self.assertIn("请输入问题", m.ai_output_text._content)

    def test_warns_when_ai_not_available(self):
        """AI 不可用 → 输出警告，不调用 ask_ai()"""
        m = _make_ai_tab_mixin(ai_available=False)
        m.ai_question_text._content = "为什么?"
        m._on_ask_ai_clicked()
        m.ask_ai.assert_not_called()
        self.assertIn("未启用", m.ai_output_text._content)

    def test_warns_when_ask_ai_returns_false(self):
        """ask_ai() 返回 False → 输出 '无法启动' 警告"""
        m = _make_ai_tab_mixin(ai_available=True)
        m.ask_ai = mock.MagicMock(return_value=False)
        m.ai_question_text._content = "为什么?"
        m._on_ask_ai_clicked()
        self.assertIn("无法启动", m.ai_output_text._content)


class TestOnGenerateAiReportClicked(unittest.TestCase):
    """_on_generate_ai_report_clicked() 按钮回调测试"""

    def test_calls_generate_ai_report_when_available(self):
        """AI 可用且有评估结果 → 调用 generate_ai_report()"""
        m = _make_ai_tab_mixin(ai_available=True,
                               last_result={'patient_score': 75.0})
        m._on_generate_ai_report_clicked()
        m.generate_ai_report.assert_called_once()

    def test_sets_status_to_generating_when_started(self):
        """启动成功 → 状态变为 '正在生成报告...'"""
        m = _make_ai_tab_mixin(ai_available=True)
        m._on_generate_ai_report_clicked()
        self.assertIn("正在生成报告", m.ai_status_var.get())

    def test_warns_when_ai_not_available(self):
        """AI 不可用 → 输出警告"""
        m = _make_ai_tab_mixin(ai_available=False)
        m._on_generate_ai_report_clicked()
        m.generate_ai_report.assert_not_called()
        self.assertIn("未启用", m.ai_output_text._content)

    def test_warns_when_no_assessment_result(self):
        """无评估结果 → 输出警告"""
        m = _make_ai_tab_mixin(ai_available=True, last_result={})
        m._on_generate_ai_report_clicked()
        m.generate_ai_report.assert_not_called()
        self.assertIn("先执行风险评估", m.ai_output_text._content)

    def test_warns_when_generate_returns_false(self):
        """generate_ai_report() 返回 False → 输出 '无法启动' 警告"""
        m = _make_ai_tab_mixin(ai_available=True)
        m.generate_ai_report = mock.MagicMock(return_value=False)
        m._on_generate_ai_report_clicked()
        self.assertIn("无法启动", m.ai_output_text._content)


class TestAiMessageCallbacks(unittest.TestCase):
    """_on_ai_answer / _on_ai_report / _on_ai_error 消息回调测试"""

    def test_on_ai_answer_appends_answer(self):
        """_on_ai_answer 把回答追加到输出区"""
        m = _make_ai_tab_mixin()
        m._on_ai_answer({'type': 'answer', 'content': '因为暴露时间长'})
        self.assertIn("因为暴露时间长", m.ai_output_text._content)
        self.assertIn("回答", m.ai_output_text._content)

    def test_on_ai_answer_resets_status(self):
        """_on_ai_answer 后状态恢复 '就绪'"""
        m = _make_ai_tab_mixin()
        m.ai_status_var.set("AI 状态：正在思考...")
        m._on_ai_answer({'type': 'answer', 'content': '回答'})
        self.assertIn("就绪", m.ai_status_var.get())

    def test_on_ai_report_appends_report(self):
        """_on_ai_report 把报告追加到输出区"""
        m = _make_ai_tab_mixin()
        m._on_ai_report({'type': 'report', 'content': '## 风险评估报告\n高风险'})
        self.assertIn("风险评估报告", m.ai_output_text._content)
        self.assertIn("AI 报告", m.ai_output_text._content)

    def test_on_ai_report_resets_status(self):
        """_on_ai_report 后状态恢复 '就绪'"""
        m = _make_ai_tab_mixin()
        m.ai_status_var.set("AI 状态：正在生成报告...")
        m._on_ai_report({'type': 'report', 'content': '报告'})
        self.assertIn("就绪", m.ai_status_var.get())

    def test_on_ai_error_appends_error(self):
        """_on_ai_error 把错误信息追加到输出区"""
        m = _make_ai_tab_mixin()
        m._on_ai_error({'type': 'error', 'content': '网络超时'})
        self.assertIn("网络超时", m.ai_output_text._content)
        self.assertIn("错误", m.ai_output_text._content)

    def test_on_ai_error_sets_status_to_error(self):
        """_on_ai_error 后状态变为 '错误'"""
        m = _make_ai_tab_mixin()
        m._on_ai_error({'type': 'error', 'content': '失败'})
        self.assertIn("错误", m.ai_status_var.get())

    def test_on_ai_answer_handles_empty_content(self):
        """空 content 不应抛异常"""
        m = _make_ai_tab_mixin()
        m._on_ai_answer({'type': 'answer', 'content': ''})
        # 应正常执行，输出区非空（含 "回答" 前缀）
        self.assertIn("回答", m.ai_output_text._content)


class TestClearAiOutput(unittest.TestCase):
    """_clear_ai_output() 清空输出测试"""

    def test_clears_output_text(self):
        """清空输出区内容"""
        m = _make_ai_tab_mixin()
        m.ai_output_text._content = "旧内容\n更多内容"
        m._clear_ai_output()
        self.assertEqual(m.ai_output_text._content, '')

    def test_resets_status_to_ready(self):
        """清空后状态恢复 '就绪'"""
        m = _make_ai_tab_mixin()
        m.ai_status_var.set("AI 状态：错误")
        m._clear_ai_output()
        self.assertIn("就绪", m.ai_status_var.get())


class TestAppendAiOutput(unittest.TestCase):
    """_append_ai_output() 辅助方法测试"""

    def test_appends_text_to_output(self):
        """文本被追加到输出区末尾"""
        m = _make_ai_tab_mixin()
        m.ai_output_text._content = "第一行\n"
        m._append_ai_output("第二行\n")
        self.assertEqual(m.ai_output_text._content, "第一行\n第二行\n")

    def test_no_error_when_ai_output_text_missing(self):
        """ai_output_text 不存在时安全返回（不抛异常）"""
        m = _make_ai_tab_mixin()
        delattr(m, 'ai_output_text')
        # 不应抛异常
        m._append_ai_output("text")


# ==============================================================================
# _init_ai_tab 测试（需要真实 Tk root，无 display 时跳过）
# ==============================================================================

class TestInitAiTab(unittest.TestCase):
    """_init_ai_tab() UI 创建测试

    需要真实 Tk root 来创建 ttk.Notebook 与子控件。
    无 display 环境（如 CI）跳过。
    """

    @classmethod
    def setUpClass(cls):
        """尝试创建 Tk root，失败则跳过整个测试类"""
        try:
            import tkinter as tk
            cls._tk = tk
            cls._root = tk.Tk()
            cls._root.withdraw()  # 不显示窗口
        except Exception:
            cls._tk = None
            cls._root = None

    @classmethod
    def tearDownClass(cls):
        if cls._root is not None:
            try:
                cls._root.destroy()
            except Exception:
                pass

    def setUp(self):
        if self._tk is None or self._root is None:
            self.skipTest("Tkinter display not available")

    def test_creates_ai_question_text_widget(self):
        """_init_ai_tab 创建 ai_question_text 控件"""
        m = _make_ai_tab_mixin()
        # 临时挂载真实 Tk root 供 _init_ai_tab 使用
        import tkinter.ttk as ttk
        notebook = ttk.Notebook(self._root)
        m._init_ai_tab(notebook)
        self.assertTrue(hasattr(m, 'ai_question_text'))

    def test_creates_ai_output_text_widget(self):
        """_init_ai_tab 创建 ai_output_text 控件"""
        m = _make_ai_tab_mixin()
        import tkinter.ttk as ttk
        notebook = ttk.Notebook(self._root)
        m._init_ai_tab(notebook)
        self.assertTrue(hasattr(m, 'ai_output_text'))

    def test_creates_ai_status_var(self):
        """_init_ai_tab 创建 ai_status_var 变量"""
        m = _make_ai_tab_mixin()
        import tkinter.ttk as ttk
        notebook = ttk.Notebook(self._root)
        m._init_ai_tab(notebook)
        self.assertTrue(hasattr(m, 'ai_status_var'))
        self.assertIn("就绪", m.ai_status_var.get())

    def test_overrides_callbacks_with_ui_methods(self):
        """_init_ai_tab 覆盖 _callbacks 为 UI 回调方法"""
        m = _make_ai_tab_mixin()
        import tkinter.ttk as ttk
        notebook = ttk.Notebook(self._root)
        m._init_ai_tab(notebook)
        self.assertEqual(m._callbacks['answer'], m._on_ai_answer)
        self.assertEqual(m._callbacks['report'], m._on_ai_report)
        self.assertEqual(m._callbacks['error'], m._on_ai_error)

    def test_adds_ai_tab_to_notebook(self):
        """_init_ai_tab 向 notebook 添加 'AI 助手' 标签页"""
        m = _make_ai_tab_mixin()
        import tkinter.ttk as ttk
        notebook = ttk.Notebook(self._root)
        m._init_ai_tab(notebook)
        # notebook.tabs() 返回所有标签页的 tab id
        # 通过 tab(text=...) 获取标签页文本验证
        tab_names = []
        for tab_id in notebook.tabs():
            tab_names.append(notebook.tab(tab_id, 'text'))
        self.assertIn('AI 助手', tab_names)


# ==============================================================================
# AiTabMixin 类结构测试（不依赖 Tk）
# ==============================================================================

class TestAiTabMixinClassStructure(unittest.TestCase):
    """AiTabMixin 类结构验证（确保方法存在）"""

    def test_class_has_required_methods(self):
        """AiTabMixin 定义了所有必需方法"""
        m = _load_ai_tab_module()
        AiTabMixin = m.AiTabMixin
        for method_name in [
            '_init_ai_tab', '_on_ask_ai_clicked',
            '_on_generate_ai_report_clicked', '_clear_ai_output',
            '_on_ai_answer', '_on_ai_report', '_on_ai_error',
            '_append_ai_output',
        ]:
            self.assertTrue(hasattr(AiTabMixin, method_name),
                            f"AiTabMixin 缺少方法: {method_name}")


if __name__ == '__main__':
    unittest.main()
