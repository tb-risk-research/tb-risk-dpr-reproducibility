#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI AI 助手面板 — AiAssistantMixin

Layer 6: 把 ai/ 子包能力接入 GUI 主界面。

提供：
- is_ai_available(): 检查 AI 是否启用（AppConfig.ai_enabled）
- ask_ai(question, callback=None): 异步问答，结果通过 queue + 轮询传回主线程
- generate_ai_report(callback=None): 异步生成风险评估报告
- _poll_ai_queue(): 主线程定时轮询队列，分发消息到回调

线程模型：
- LLM 调用在后台 threading.Thread 中执行（避免阻塞 Tk 主线程）
- 结果通过 queue.Queue 传回（线程安全）
- 主线程通过 root.after() 定时调用 _poll_ai_queue() 取消息并调用回调
- 回调类型: 'answer' / 'report' / 'error'，对应 mixin._callbacks 字典

AppConfig.ai_enabled=False 时所有方法立即返回 False，不启动线程。

使用方式（主界面）：
    class MainApp(DataImportMixin, AiAssistantMixin, ...):
        def __init__(self):
            ...
            self._ai_queue = queue.Queue()
            self._ai_active_jobs = set()
            self._callbacks = {'answer': self._on_ai_answer,
                               'report': self._on_ai_report,
                               'error': self._on_ai_error}
            self._start_ai_polling()

        def on_ask_ai_clicked(self):
            question = self.ai_question_entry.get()
            self.ask_ai(question)
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from typing import Any, Callable, Dict, Optional

LOGGER = logging.getLogger("tb_risk.gui.ai_panel")


class AiAssistantMixin:
    """GUI AI 助手 Mixin — 提供 ask_ai / generate_ai_report 异步接口

    依赖宿主 app 拥有以下属性（mixin 被混入主类时 self 即 app，故直接 getattr(self, ...)）：
    - self.app_config.ai_enabled: bool
    - self.last_assessment_result: dict (最近一次评估结果)
    - self.root.after(ms, callback): Tkinter 定时器
    - self.root.after_cancel(timer_id)

    Mixin 自身维护：
    - _ai_queue: queue.Queue 跨线程消息
    - _ai_active_jobs: set 后台线程 job_id 集合
    - _callbacks: dict {type: callable(msg)}
    - _poll_interval_ms: int 轮询间隔
    - _poll_timer_id: str|None 当前轮询定时器
    - _ai_cancel_events: dict {job_id: threading.Event} 取消信号（缺陷 #7）
    """

    # 轮询间隔（ms）— 默认 100ms
    _poll_interval_ms: int = 100

    def _get_app(self):
        """获取宿主 app

        - 当 mixin 被混入主类时，self 即 app，返回 self
        - 当 mixin 被独立构造（测试场景）时，self.app 是注入的 fake app
        """
        return getattr(self, 'app', self)

    def is_ai_available(self) -> bool:
        """检查 AI 是否可用（开关 + 配置）"""
        app = self._get_app()
        cfg = getattr(app, 'app_config', None)
        if cfg is None:
            return False
        return bool(getattr(cfg, 'ai_enabled', False))

    def ask_ai(self, question: str,
               callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> bool:
        """异步问答：在后台线程调用 ai.qa.ask，结果通过 queue 传回。

        Args:
            question: 用户问题
            callback: 可选回调（覆盖 _callbacks['answer']）

        Returns:
            bool: True 表示已启动后台线程；False 表示 AI 不可用或参数无效。
        """
        if not self.is_ai_available():
            LOGGER.debug("AI 不可用，跳过 ask_ai")
            return False
        if not question or not isinstance(question, str):
            return False

        if callback is not None:
            self._callbacks['answer'] = callback

        app = self._get_app()
        context = getattr(app, 'last_assessment_result', None) or {}
        job_id = str(uuid.uuid4())[:8]
        self._ai_active_jobs.add(job_id)
        # 缺陷 #7：为该 job 创建取消 Event，worker 在调用 AI 前检查
        if not hasattr(self, '_ai_cancel_events'):
            self._ai_cancel_events = {}
        cancel_event = threading.Event()
        self._ai_cancel_events[job_id] = cancel_event

        thread = threading.Thread(
            target=self._ask_ai_worker,
            args=(job_id, question, context, cancel_event),
            daemon=True,
            name=f"ai-ask-{job_id}",
        )
        thread.start()
        return True

    def generate_ai_report(self,
                          callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> bool:
        """异步生成风险评估报告。

        Args:
            callback: 可选回调（覆盖 _callbacks['report']）

        Returns:
            bool: True 表示已启动后台线程；False 表示 AI 不可用。
        """
        if not self.is_ai_available():
            LOGGER.debug("AI 不可用，跳过 generate_ai_report")
            return False

        if callback is not None:
            self._callbacks['report'] = callback

        app = self._get_app()
        result = getattr(app, 'last_assessment_result', None) or {}
        job_id = str(uuid.uuid4())[:8]
        self._ai_active_jobs.add(job_id)
        # 缺陷 #7：为该 job 创建取消 Event
        if not hasattr(self, '_ai_cancel_events'):
            self._ai_cancel_events = {}
        cancel_event = threading.Event()
        self._ai_cancel_events[job_id] = cancel_event

        thread = threading.Thread(
            target=self._generate_report_worker,
            args=(job_id, result, cancel_event),
            daemon=True,
            name=f"ai-report-{job_id}",
        )
        thread.start()
        return True

    # ======================================================================
    # 取消机制（缺陷 #7）
    # ======================================================================

    def cancel_ai_task(self, job_id: str) -> bool:
        """取消指定的 AI 任务（设置其 Event 让 worker 主动退出）

        Args:
            job_id: 任务 ID（ask_ai / generate_ai_report 启动时分配）

        Returns:
            bool: True 表示找到并取消了任务；False 表示任务不存在或已完成
        """
        if not hasattr(self, '_ai_cancel_events'):
            self._ai_cancel_events = {}
            return False
        event = self._ai_cancel_events.get(job_id)
        if event is None:
            return False
        event.set()
        LOGGER.info("已请求取消 AI 任务 %s", job_id)
        return True

    def cancel_ai_tasks(self) -> int:
        """取消所有活动 AI 任务（设置所有 Event）

        适用于 GUI "取消" 按钮：用户不想再等待长时间运行的 LLM 请求。

        Returns:
            int: 被取消的任务数
        """
        if not hasattr(self, '_ai_cancel_events'):
            self._ai_cancel_events = {}
            return 0
        count = 0
        for job_id, event in list(self._ai_cancel_events.items()):
            if not event.is_set():
                event.set()
                count += 1
                LOGGER.info("已请求取消 AI 任务 %s", job_id)
        return count

    # ======================================================================
    # Worker（后台线程）
    # ======================================================================

    def _ask_ai_worker(self, job_id: str, question: str,
                       context: Dict[str, Any],
                       cancel_event: threading.Event) -> None:
        """问答 worker：调用 ai.qa.ask 并把结果放到 queue

        缺陷 #7：在调用 AI 前检查 cancel_event，被取消则 put 'cancelled' 消息
        """
        try:
            # 缺陷 #7：取消检查（在调用 AI 前主动退出）
            if cancel_event.is_set():
                self._ai_queue.put({'type': 'cancelled',
                                    'content': 'AI 问答已取消'})
                LOGGER.info("ask_ai worker %s 被取消，跳过 AI 调用", job_id)
                return
            from tb_risk.ai.qa import ask
            answer = ask(question, context=context)
        except Exception as e:
            LOGGER.warning("ask_ai worker 异常: %s", e)
            answer = None
        finally:
            self._ai_active_jobs.discard(job_id)
            # 缺陷 #7：清理取消 Event（任务已结束，不再需要取消信号）
            if hasattr(self, '_ai_cancel_events'):
                self._ai_cancel_events.pop(job_id, None)

        if answer:
            self._ai_queue.put({'type': 'answer', 'content': answer})
        else:
            self._ai_queue.put({'type': 'error',
                                'content': 'AI 问答失败（未配置或调用异常）'})

    def _generate_report_worker(self, job_id: str,
                                result: Dict[str, Any],
                                cancel_event: threading.Event) -> None:
        """报告生成 worker：调用 ai.reporter.generate_report

        缺陷 #7：在调用 AI 前检查 cancel_event，被取消则 put 'cancelled' 消息
        """
        try:
            # 缺陷 #7：取消检查
            if cancel_event.is_set():
                self._ai_queue.put({'type': 'cancelled',
                                    'content': 'AI 报告生成已取消'})
                LOGGER.info("generate_report worker %s 被取消，跳过 AI 调用",
                            job_id)
                return
            from tb_risk.ai.reporter import generate_report
            report = generate_report(result)
        except Exception as e:
            LOGGER.warning("generate_ai_report worker 异常: %s", e)
            report = None
        finally:
            self._ai_active_jobs.discard(job_id)
            # 缺陷 #7：清理取消 Event
            if hasattr(self, '_ai_cancel_events'):
                self._ai_cancel_events.pop(job_id, None)

        if report:
            self._ai_queue.put({'type': 'report', 'content': report})
        else:
            self._ai_queue.put({'type': 'error',
                                'content': 'AI 报告生成失败（未配置或调用异常）'})

    # ======================================================================
    # 主线程轮询
    # ======================================================================

    def _poll_ai_queue(self) -> None:
        """主线程定时回调：取 queue 中所有消息，分发到 _callbacks"""
        while True:
            try:
                msg = self._ai_queue.get_nowait()
            except queue.Empty:
                break
            msg_type = msg.get('type', 'unknown')
            cb = self._callbacks.get(msg_type)
            if cb is not None:
                try:
                    cb(msg)
                except Exception as e:
                    LOGGER.warning("AI 回调 (%s) 异常: %s", msg_type, e)
        # 重新调度下一次轮询
        self._schedule_next_poll()

    def _schedule_next_poll(self) -> None:
        """调度下一次 queue 轮询"""
        app = self._get_app()
        root = getattr(app, 'root', None)
        if root is None:
            return
        # 幂等性：先取消旧定时器，防止重复调度
        if self._poll_timer_id is not None:
            try:
                root.after_cancel(self._poll_timer_id)
            except Exception as e:
                LOGGER.debug("取消旧 AI 轮询定时器失败: %s", e)
        try:
            self._poll_timer_id = root.after(
                self._poll_interval_ms, self._poll_ai_queue
            )
        except Exception as e:
            LOGGER.debug("调度 AI 轮询失败: %s", e)

    def _start_ai_polling(self) -> None:
        """启动 AI queue 轮询（GUI 初始化时调用）"""
        self._schedule_next_poll()

    def _stop_ai_polling(self) -> None:
        """停止 AI queue 轮询（GUI 退出时调用）"""
        if self._poll_timer_id is not None:
            app = self._get_app()
            root = getattr(app, 'root', None)
            if root is not None:
                try:
                    root.after_cancel(self._poll_timer_id)
                except Exception as e:
                    LOGGER.debug("停止 AI 轮询定时器失败: %s", e)
            self._poll_timer_id = None


__all__ = ['AiAssistantMixin']
