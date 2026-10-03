#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 训练面板 - ML 训练与高级分析（MLTrainingMixin）

包含：ML模型训练、超参数调优、不确定性量化、生存分析、RL干预规划

本包由原 ml_tab.py 单文件拆分为多个 Mixin 子模块，组合后保持
MLTrainingMixin 单一类，所有 self.* 调用无需修改。

子模块：
- _training.py:    训练控制相关 12 个方法（_TrainingMixin）
- _uncertainty.py: 不确定性量化 6 个方法（_UncertaintyMixin）
- _survival.py:    生存分析 3 个方法（_SurvivalMixin）
- _rl.py:          强化学习 2 个方法（_RLMixin）
- _repo_ui.py:     模型仓库与训练历史 UI（_RepoUIMixin，Section VII）
"""

from .._shared import *  # noqa: F401,F403

from ._training import _TrainingMixin
from ._uncertainty import _UncertaintyMixin
from ._survival import _SurvivalMixin
from ._rl import _RLMixin
from ._repo_ui import _RepoUIMixin


class MLTrainingMixin(_TrainingMixin, _UncertaintyMixin, _SurvivalMixin,
                       _RLMixin, _RepoUIMixin):
    """ML 模型训练、不确定性量化与高级分析方法"""
    pass
