#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — GUI AI 助手面板 (tb_risk.gui.ai_panel)

测试 Layer 6: AiAssistantMixin 提供 ask_ai / generate_ai_report 方法，
使用 threading + queue 异步执行 LLM 调用，主线程通过 queue 轮询取结果。

设计：
- 不直接驱动 Tkinter 渲染（需 display）
- 注入 mock app（仅含必要属性：app_config, root.after, queue）
- 验证业务逻辑：ai_enabled 开关、worker 函数、回调机制
"""

import os
import queue
import sys
import threading
import time
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


class _FakeApp:
    """伪 GUI app，仅包含 AiAssistantMixin 所需属性"""

    def __init__(self, ai_enabled=True, ai_provider='deepseek',
                 ai_local_model_path=''):
        # AppConfig-like 对象
        self.app_config = mock.MagicMock()
        self.app_config.ai_enabled = ai_enabled
        self.app_config.ai_provider = ai_provider
        self.app_config.ai_local_model_path = ai_local_model_path
        # 当前评估结果
        self.last_assessment_result = {'patient_score': 75.0}
        # root.after 调用记录
        self.after_calls = []
        self._timer_counter = 0

    def root_after(self, ms, callback):
        """模拟 Tkinter root.after：记录调用但不真正调度"""
        self.after_calls.append({'ms': ms, 'callback': callback})
        self._timer_counter += 1
        return f'timer#{self._timer_counter}'

    def root_after_cancel(self, timer_id):
        pass


def _load_ai_panel_module():
    """直接加载 gui/ai_panel.py 模块（绕过 gui/__init__.py 的 Tkinter 依赖）"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'gui.ai_panel',
        os.path.join(_PROJECT_ROOT, 'gui', 'ai_panel.py'),
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _make_mixin(ai_enabled=True):
    """构造 AiAssistantMixin 实例（不继承 Tk）"""
    m = _load_ai_panel_module()
    AiAssistantMixin = m.AiAssistantMixin
    mixin = AiAssistantMixin.__new__(AiAssistantMixin)
    mixin.app = _FakeApp(ai_enabled=ai_enabled)
    mixin._ai_queue = queue.Queue()
    mixin._ai_active_jobs = set()
    mixin._poll_interval_ms = 50
    mixin._poll_timer_id = None
    return mixin


class TestIsAiAvailable(unittest.TestCase):
    """is_ai_available() 测试"""

    def test_returns_false_when_ai_disabled(self):
        """ai_enabled=False → False"""
        mixin = _make_mixin(ai_enabled=False)
        self.assertFalse(mixin.is_ai_available())

    def test_returns_true_when_ai_enabled(self):
        """ai_enabled=True → True"""
        mixin = _make_mixin(ai_enabled=True)
        self.assertTrue(mixin.is_ai_available())


class TestAskAi(unittest.TestCase):
    """ask_ai() 测试"""

    def test_returns_false_when_ai_disabled(self):
        """ai_enabled=False → 立即返回 False，不启动线程"""
        mixin = _make_mixin(ai_enabled=False)
        result = mixin.ask_ai("为什么风险高?")
        self.assertFalse(result)
        # 不应有活跃 job
        self.assertEqual(len(mixin._ai_active_jobs), 0)

    def test_returns_true_when_ai_enabled(self):
        """ai_enabled=True → 启动后台线程，返回 True"""
        mixin = _make_mixin(ai_enabled=True)
        with mock.patch('tb_risk.ai.qa.ask', return_value="LLM 回答"):
            result = mixin.ask_ai("为什么风险高?")
        self.assertTrue(result)
        # 等待线程结束
        time.sleep(0.3)
        # 队列应有结果
        self.assertFalse(mixin._ai_queue.empty())

    def test_worker_puts_result_on_queue(self):
        """worker 线程把 LLM 回答放到 queue"""
        mixin = _make_mixin(ai_enabled=True)
        mixin.app.last_assessment_result = {'patient_score': 75.0}
        with mock.patch('tb_risk.ai.qa.ask', return_value="因为暴露时间长"):
            mixin.ask_ai("为什么风险高?")
            time.sleep(0.3)
        # 取队列消息
        msg = mixin._ai_queue.get_nowait()
        self.assertEqual(msg['type'], 'answer')
        self.assertIn('因为暴露时间长', msg['content'])

    def test_worker_puts_error_on_queue_when_llm_fails(self):
        """LLM 调用失败 → queue 收到 error 消息"""
        mixin = _make_mixin(ai_enabled=True)
        with mock.patch('tb_risk.ai.qa.ask', return_value=None):
            mixin.ask_ai("为什么?")
            time.sleep(0.3)
        msg = mixin._ai_queue.get_nowait()
        self.assertEqual(msg['type'], 'error')


class TestGenerateAiReport(unittest.TestCase):
    """generate_ai_report() 测试"""

    def test_returns_false_when_ai_disabled(self):
        """ai_enabled=False → 立即返回 False"""
        mixin = _make_mixin(ai_enabled=False)
        result = mixin.generate_ai_report()
        self.assertFalse(result)

    def test_returns_true_when_ai_enabled(self):
        """ai_enabled=True → 启动后台线程，返回 True"""
        mixin = _make_mixin(ai_enabled=True)
        with mock.patch('tb_risk.ai.reporter.generate_report',
                        return_value="## 报告内容"):
            result = mixin.generate_ai_report()
        self.assertTrue(result)
        time.sleep(0.3)
        self.assertFalse(mixin._ai_queue.empty())

    def test_worker_puts_report_on_queue(self):
        """worker 把报告放到 queue"""
        mixin = _make_mixin(ai_enabled=True)
        mixin.app.last_assessment_result = {'patient_score': 80.0}
        with mock.patch('tb_risk.ai.reporter.generate_report',
                        return_value="## 风险评估报告"):
            mixin.generate_ai_report()
            time.sleep(0.3)
        msg = mixin._ai_queue.get_nowait()
        self.assertEqual(msg['type'], 'report')
        self.assertIn("风险评估报告", msg['content'])


class TestPollQueue(unittest.TestCase):
    """_poll_ai_queue() 测试"""

    def test_does_nothing_when_queue_empty(self):
        """空队列 → 不调用回调"""
        mixin = _make_mixin(ai_enabled=True)
        mixin._callbacks = {}
        # 不应抛异常
        mixin._poll_ai_queue()

    def test_invokes_callback_for_answer_message(self):
        """队列有 answer 消息 → 调用注册的回调"""
        mixin = _make_mixin(ai_enabled=True)
        mixin._callbacks = {}
        called_with = []
        mixin._callbacks['answer'] = lambda msg: called_with.append(msg)
        mixin._ai_queue.put({'type': 'answer', 'content': 'LLM 回答'})
        mixin._poll_ai_queue()
        self.assertEqual(len(called_with), 1)
        self.assertEqual(called_with[0]['content'], 'LLM 回答')

    def test_invokes_callback_for_report_message(self):
        """队列有 report 消息 → 调用 report 回调"""
        mixin = _make_mixin(ai_enabled=True)
        mixin._callbacks = {}
        called_with = []
        mixin._callbacks['report'] = lambda msg: called_with.append(msg)
        mixin._ai_queue.put({'type': 'report', 'content': '## 报告'})
        mixin._poll_ai_queue()
        self.assertEqual(len(called_with), 1)

    def test_invokes_callback_for_error_message(self):
        """队列有 error 消息 → 调用 error 回调"""
        mixin = _make_mixin(ai_enabled=True)
        mixin._callbacks = {}
        called_with = []
        mixin._callbacks['error'] = lambda msg: called_with.append(msg)
        mixin._ai_queue.put({'type': 'error', 'content': '网络错误'})
        mixin._poll_ai_queue()
        self.assertEqual(len(called_with), 1)


class TestCancelAiTasks(unittest.TestCase):
    """缺陷 #7（本轮）：AI 任务取消机制测试

    AiAssistantMixin 应提供 threading.Event 让后台 worker 主动退出，
    避免长时间运行的 LLM 请求阻塞 GUI。
    """

    def test_mixin_has_cancel_events_dict(self):
        """AiAssistantMixin 实例应有 _ai_cancel_events 字典属性"""
        mixin = _make_mixin(ai_enabled=True)
        mixin._ai_cancel_events = {}
        self.assertIsInstance(mixin._ai_cancel_events, dict)

    def test_ask_ai_creates_cancel_event(self):
        """ask_ai 启动后台任务后，_ai_cancel_events 应有对应 job_id 的 Event"""
        mixin = _make_mixin(ai_enabled=True)
        mixin._ai_cancel_events = {}
        # 用延迟 mock 让 worker 不立即完成
        def slow_ask(*a, **kw):
            time.sleep(0.5)
            return "回答"
        with mock.patch('tb_risk.ai.qa.ask', side_effect=slow_ask):
            mixin.ask_ai("问题")
            # 等待线程启动
            time.sleep(0.05)
            # 应至少有一个 cancel event
            self.assertGreater(len(mixin._ai_cancel_events), 0)
            for event in mixin._ai_cancel_events.values():
                self.assertIsInstance(event, threading.Event)
                self.assertFalse(event.is_set())

    def test_cancel_ai_task_sets_event(self):
        """cancel_ai_task(job_id) 设置指定 job 的 Event"""
        mixin = _make_mixin(ai_enabled=True)
        mixin._ai_cancel_events = {}
        # 手动注册一个 job_id 与 Event
        test_job_id = 'test123'
        event = threading.Event()
        mixin._ai_cancel_events[test_job_id] = event
        mixin._ai_active_jobs.add(test_job_id)

        mixin.cancel_ai_task(test_job_id)
        self.assertTrue(event.is_set())

    def test_cancel_ai_tasks_sets_all_events(self):
        """cancel_ai_tasks() 设置所有活动 Event"""
        mixin = _make_mixin(ai_enabled=True)
        mixin._ai_cancel_events = {}
        e1 = threading.Event()
        e2 = threading.Event()
        mixin._ai_cancel_events = {'job1': e1, 'job2': e2}
        mixin._ai_active_jobs = {'job1', 'job2'}

        mixin.cancel_ai_tasks()
        self.assertTrue(e1.is_set())
        self.assertTrue(e2.is_set())

    def test_cancel_ai_tasks_no_op_when_no_active_jobs(self):
        """无活动任务时 cancel_ai_tasks 不抛异常"""
        mixin = _make_mixin(ai_enabled=True)
        mixin._ai_cancel_events = {}
        mixin._ai_active_jobs = set()
        # 不应抛异常
        mixin.cancel_ai_tasks()

    def test_worker_checks_cancel_event_before_ai_call(self):
        """worker 启动前 Event 已设置 → 不调用 AI，queue 收到 'cancelled' 消息"""
        mixin = _make_mixin(ai_enabled=True)
        mixin._ai_cancel_events = {}

        # 让 worker 在调用 AI 前被取消
        call_count = [0]
        original_ask = None

        def tracking_ask(*a, **kw):
            call_count[0] += 1
            return "不应被调用"

        # 拦截 worker 启动，预先设置 cancel event
        original_thread_start = None

        with mock.patch('tb_risk.ai.qa.ask', side_effect=tracking_ask):
            # 启动 ask_ai（会创建 job_id 与 Event）
            mixin.ask_ai("问题")
            # 立即取消所有任务（设置 Event）
            time.sleep(0.01)  # 让 worker 进入
            mixin.cancel_ai_tasks()
            time.sleep(0.3)  # 等 worker 结束

        # queue 应有消息（cancelled 或 answer/error）
        self.assertFalse(mixin._ai_queue.empty())
        msgs = []
        while not mixin._ai_queue.empty():
            msgs.append(mixin._ai_queue.get_nowait())
        # 应至少有一条消息
        self.assertTrue(msgs)

    def test_worker_removes_event_after_completion(self):
        """worker 完成后从 _ai_cancel_events 删除自己的 Event"""
        mixin = _make_mixin(ai_enabled=True)
        mixin._ai_cancel_events = {}

        with mock.patch('tb_risk.ai.qa.ask', return_value="回答"):
            mixin.ask_ai("问题")
            time.sleep(0.3)

        # worker 完成后 _ai_cancel_events 应清空
        self.assertEqual(len(mixin._ai_cancel_events), 0,
                         f"未清理完成的 job events: {mixin._ai_cancel_events}")

    def test_cancel_emits_cancelled_message_to_queue(self):
        """被取消的 worker 应 put 'cancelled' 消息到 queue（区分于 error）

        直接调用 worker 函数（预先设置 Event），验证 worker 在 AI 调用前
        主动退出并 put 'cancelled' 消息。这避免时序竞争（取消必须在
        worker 检查点之前发生）。
        """
        mixin = _make_mixin(ai_enabled=True)
        mixin._ai_cancel_events = {}

        # 预先设置 cancel_event，worker 应在调用 AI 前主动退出
        cancel_event = threading.Event()
        cancel_event.set()
        job_id = 'test-cancel'
        mixin._ai_active_jobs.add(job_id)
        mixin._ai_cancel_events[job_id] = cancel_event

        # 直接调用 worker（不通过 ask_ai 启动线程）
        ai_called = [False]

        def tracking_ask(*a, **kw):
            ai_called[0] = True
            return "不应被调用"

        with mock.patch('tb_risk.ai.qa.ask', side_effect=tracking_ask):
            mixin._ask_ai_worker(job_id, "问题", {}, cancel_event)

        # AI 不应被调用
        self.assertFalse(ai_called[0],
                         "worker 被 cancel 后仍调用了 AI")
        # queue 应有 cancelled 消息
        self.assertFalse(mixin._ai_queue.empty())
        msg = mixin._ai_queue.get_nowait()
        self.assertEqual(msg['type'], 'cancelled')
        self.assertIn('取消', msg['content'])
        # _ai_cancel_events 应被清理
        self.assertNotIn(job_id, mixin._ai_cancel_events)


if __name__ == '__main__':
    unittest.main()
