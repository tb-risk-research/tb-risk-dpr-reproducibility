#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tb_risk CLI 命令行接口

基于 click 的命令行评估管线。
零依赖 tkinter，可在服务器/Docker 中运行。
"""

from .commands import cli

__all__ = ['cli']