#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 部署与监控面板

顶层「部署与监控」标签页，承载三个部署期子功能：

  1. 批量接触者评估 — 导入 → 批量打分（三级回退）→ 风险排序 → 导出
  2. 批量模型对比   — 归档 ML 快照在同一确定性队列上的指标对比
  3. 漂移监控       — PSI 分布漂移（评估日志）+ 性能趋势（训练档案）

计算逻辑全部委托 core 层（deploy_services / assessment_log），
本包只负责界面与交互。
"""

from ._shared import *  # noqa: F401,F403
from .batch_tab import BatchAssessTabMixin
from .compare_tab import ModelCompareTabMixin
from .drift_tab import DriftMonitorTabMixin


class DeployPanelMixin(BatchAssessTabMixin, ModelCompareTabMixin,
                       DriftMonitorTabMixin):
    """部署与监控面板（批量评估 / 模型对比 / 漂移监控）"""

    def _init_deploy_tab(self, parent):
        """初始化「部署与监控」顶层标签页。"""
        # 顶部说明条
        header = ttk.Frame(parent)
        header.pack(fill='x', padx=12, pady=(10, 0))
        ttk.Label(header, text='部署与监控',
                  style='Heading.TLabel').pack(side='left')
        ttk.Label(header,
                  text='批量接触者评估 ｜ 归档模型同队列对比 ｜ '
                       '分布漂移与性能趋势监控',
                  style='Tip.TLabel').pack(side='left', padx=10)

        sub_notebook = ttk.Notebook(parent)
        sub_notebook.pack(fill='both', expand=True, padx=8, pady=8)

        self._build_batch_assess_tab(sub_notebook)
        self._build_model_compare_tab(sub_notebook)
        self._build_drift_monitor_tab(sub_notebook)
