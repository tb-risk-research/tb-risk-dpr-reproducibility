#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 界面模式适配器（优先级六）

将 Classic / Wizard 双模式的条件分支统一为策略模式。
GUIMixin 持有 self.adapter 引用，所有需要区分模式的代码改为
调用 self.adapter.refresh_family_tree() 等统一接口。
模式切换时只需替换 self.adapter 实例。

设计要点：
  - UIModeAdapter 为抽象基类，每个方法在基类级别抛出 NotImplementedError，
    确保子类遗漏方法在调用期即被发现。
  - ClassicAdapter 调用 _update_family_treeview / _update_social_treeview。
  - WizardAdapter 调用 _refresh_family_tree / _refresh_social_tree。
  - 适配器持有 GUI 宿主弱引用（避免循环引用），所有方法委托宿主实现。
"""

from ._imports import LOGGER


class UIModeAdapter:
    """界面模式适配器抽象基类

    统一 Classic / Wizard 双模式的接口契约。
    子类必须实现所有抽象方法，否则在调用时抛出 NotImplementedError。
    """

    def __init__(self, host):
        """初始化适配器

        Args:
            host: GUIMixin 实例（即 TB_Risk_Assessment），提供具体控件方法
        """
        self.host = host

    def refresh_family_tree(self):
        """刷新家庭接触者树视图"""
        raise NotImplementedError("子类必须实现 refresh_family_tree")

    def refresh_social_tree(self):
        """刷新社会接触者树视图"""
        raise NotImplementedError("子类必须实现 refresh_social_tree")

    def refresh_after_undo(self):
        """撤销/重做后刷新所有受影响的 Treeview"""
        raise NotImplementedError("子类必须实现 refresh_after_undo")

    def display_results(self):
        """展示评估结果（委托宿主的具体模式方法）"""
        raise NotImplementedError("子类必须实现 display_results")

    def draw_charts(self):
        """绘制评估图表（委托宿主的具体模式方法）"""
        raise NotImplementedError("子类必须实现 draw_charts")


class ClassicAdapter(UIModeAdapter):
    """传统标签页界面适配器

    委托宿主的 _update_family_treeview / _update_social_treeview 方法。
    """

    def refresh_family_tree(self):
        if hasattr(self.host, '_update_family_treeview'):
            self.host._update_family_treeview()
        else:
            LOGGER.debug("ClassicAdapter: 宿主缺少 _update_family_treeview 方法")
        self._notify_button_state()

    def refresh_social_tree(self):
        if hasattr(self.host, '_update_social_treeview'):
            self.host._update_social_treeview()
        else:
            LOGGER.debug("ClassicAdapter: 宿主缺少 _update_social_treeview 方法")
        self._notify_button_state()

    def refresh_after_undo(self):
        """撤销/重做后刷新家庭/社会树视图 + 概览面板"""
        self.refresh_family_tree()
        self.refresh_social_tree()
        if hasattr(self.host, '_update_overview_panel'):
            self.host._update_overview_panel()

    def _notify_button_state(self):
        """P6-20：数据变更后通知宿主更新评估按钮状态"""
        if hasattr(self.host, '_update_assess_button_state'):
            try:
                self.host._update_assess_button_state()
            except Exception as e:
                LOGGER.debug("ClassicAdapter._notify_button_state 失败: %s", e)

    def display_results(self):
        if hasattr(self.host, '_display_results_classic'):
            self.host._display_results_classic()
        else:
            LOGGER.debug("ClassicAdapter: 宿主缺少 _display_results_classic 方法")

    def draw_charts(self):
        if hasattr(self.host, '_draw_classic_charts'):
            self.host._draw_classic_charts()
        else:
            LOGGER.debug("ClassicAdapter: 宿主缺少 _draw_classic_charts 方法")


class WizardAdapter(UIModeAdapter):
    """向导模式界面适配器

    委托宿主的 _refresh_family_tree / _refresh_social_tree 方法。
    """

    def refresh_family_tree(self):
        if hasattr(self.host, '_refresh_family_tree'):
            self.host._refresh_family_tree()
        else:
            LOGGER.debug("WizardAdapter: 宿主缺少 _refresh_family_tree 方法")
        self._notify_button_state()

    def refresh_social_tree(self):
        if hasattr(self.host, '_refresh_social_tree'):
            self.host._refresh_social_tree()
        else:
            LOGGER.debug("WizardAdapter: 宿主缺少 _refresh_social_tree 方法")
        self._notify_button_state()

    def refresh_after_undo(self):
        """撤销/重做后刷新家庭/社会树视图 + 概览面板"""
        self.refresh_family_tree()
        self.refresh_social_tree()
        if hasattr(self.host, '_update_overview_panel'):
            self.host._update_overview_panel()

    def _notify_button_state(self):
        """P6-20：数据变更后通知宿主更新评估按钮状态"""
        if hasattr(self.host, '_update_assess_button_state'):
            try:
                self.host._update_assess_button_state()
            except Exception as e:
                LOGGER.debug("WizardAdapter._notify_button_state 失败: %s", e)

    def display_results(self):
        if hasattr(self.host, '_display_results_wizard'):
            self.host._display_results_wizard()
        else:
            LOGGER.debug("WizardAdapter: 宿主缺少 _display_results_wizard 方法")

    def draw_charts(self):
        if hasattr(self.host, '_draw_wizard_charts'):
            self.host._draw_wizard_charts()
        # 向导模式额外调用 _finish_wizard_assessment 完成评估收尾
        if hasattr(self.host, '_finish_wizard_assessment'):
            self.host._finish_wizard_assessment()


def create_adapter(ui_mode: str, host) -> UIModeAdapter:
    """工厂方法：根据 ui_mode 创建对应的适配器实例

    Args:
        ui_mode: 'classic' 或 'wizard'（对齐 UI_MODE_CLASSIC / UI_MODE_WIZARD 常量）
        host: GUIMixin 实例

    Returns:
        UIModeAdapter 子类实例；未知模式时回退到 ClassicAdapter
    """
    if ui_mode == 'wizard':
        return WizardAdapter(host)
    # 默认回退到 Classic（包括 None / 'classic' / 未知值）
    return ClassicAdapter(host)


__all__ = ['UIModeAdapter', 'ClassicAdapter', 'WizardAdapter', 'create_adapter']
