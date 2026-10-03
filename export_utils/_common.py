#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_utils 子模块共享的模块级状态与条件导入。

子模块通过 ``from ._common import ...`` 复用 LOGGER、pandas/numpy 的可用性标志
及实例，避免重复定义和状态不一致。
"""

import logging

LOGGER = logging.getLogger("tb_risk.export")

# 条件导入
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
