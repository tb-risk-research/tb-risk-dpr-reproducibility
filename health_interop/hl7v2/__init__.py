#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HL7 v2.x MLLP 消息接口模块

提供：
- MLLPServer: 最小下层协议（MLLP）TCP服务端，常驻后台监听医院集成平台推送
- HL7MessageParser: HL7 v2.x消息解析器（MSH/PID/ORC/OBR/OBX/TXA段）
- HL7AckBuilder: ACK消息构造（AA/AR/AE）
- HL7MessageQueue: 消息持久化队列+死信队列+人工重试
支持消息类型：ADT^A01/A03/A04（患者建档）、ORM^O01（检验申请）、
ORU^R01（检验结果回报）、MDM^T02（病历文档传输）。
"""

from .server import MLLPServer
from .parser import HL7MessageParser, HL7Message
from .queue import HL7MessageQueue
from .ack import HL7AckBuilder

__all__ = [
    'MLLPServer',
    'HL7MessageParser',
    'HL7Message',
    'HL7MessageQueue',
    'HL7AckBuilder',
]
