#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可靠消息队列子包

提供企业级消息可靠性保障：
- ReliableMessageQueue: 可靠消息队列主类（ACK/重试/死信/幂等）
- ReliableMessage: 可靠消息体
- MessageStatus/QueueType: 状态枚举
- MessageHandlerError: 消息处理异常
"""

from .reliable_queue import (
    ReliableMessageQueue,
    ReliableMessage,
    MessageStatus,
    QueueType,
    MessageHandlerError,
)

__all__ = [
    'ReliableMessageQueue',
    'ReliableMessage',
    'MessageStatus',
    'QueueType',
    'MessageHandlerError',
]
