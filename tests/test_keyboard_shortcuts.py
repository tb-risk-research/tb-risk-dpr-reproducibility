#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Section X 全局体验层测试 — 快捷键与主题

覆盖 BaseMixin 的快捷键回调方法（无需真实 Tk，使用 mock notebook/style）：
- _shortcut_add_contact: 根据标签页名称判断家庭/社会
- _switch_tab: Ctrl+Tab/Ctrl+Shift+Tab 标签页切换、wrap-around
- _zoom_chart: Ctrl+加号/减号图表缩放、范围限制
- _delete_selected_contact / _edit_selected_contact: Delete/Enter 键
- _switch_theme: 浅色/深色主题切换、未知主题回退
- _setup_keyboard_shortcuts: 快捷键字典映射、绑定失败容错
"""

import os
import sys
from unittest import mock

import pytest
import tkinter as tk

# 将项目根目录加入 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from tb_risk.gui.base import BaseMixin


# ===========================================================================
# 辅助 fixture：BaseMixin 不含 __init__，可直接实例化并设置属性
# ===========================================================================

class _FakeNotebook:
    """最小可用 ttk.Notebook 桩对象

    模拟 tabs() / select() / tab(tab_id, 'text') / select(tab_id) 接口
    """

    def __init__(self, tabs=None, current=None, tab_texts=None):
        # tabs: list of str tab ids
        self._tabs = tabs or []
        self._current = current
        self._tab_texts = tab_texts or {}
        self.select_calls = []

    def tabs(self):
        return list(self._tabs)

    def select(self, tab_id=None):
        if tab_id is None:
            return self._current
        self.select_calls.append(tab_id)
        self._current = tab_id

    def tab(self, tab_id, option=None):
        if option == 'text':
            return self._tab_texts.get(tab_id, '')
        return None


@pytest.fixture
def mixin():
    """构造 BaseMixin 实例（无 root，无 style）"""
    m = BaseMixin()
    # 模拟 _add_family_member / _add_social_contact 等方法
    m._add_family_member = mock.MagicMock()
    m._add_social_contact = mock.MagicMock()
    m._delete_family_member = mock.MagicMock()
    m._delete_social_contact = mock.MagicMock()
    m._edit_family_member = mock.MagicMock()
    m._edit_social_contact = mock.MagicMock()
    return m


# ===========================================================================
# _shortcut_add_contact (Ctrl+N)
# ===========================================================================

class TestShortcutAddContact:
    """Ctrl+N: 根据当前标签页名称添加接触者"""

    def test_add_family_when_family_tab_selected(self, mixin):
        """标签页文本含'家庭'时调用 _add_family_member"""
        nb = _FakeNotebook(tabs=['t1', 't2'], current='t1',
                            tab_texts={'t1': '家庭成员信息'})
        mixin.notebook = nb
        mixin._shortcut_add_contact()
        mixin._add_family_member.assert_called_once()
        mixin._add_social_contact.assert_not_called()

    def test_add_social_when_social_tab_selected(self, mixin):
        """标签页文本含'社会'时调用 _add_social_contact"""
        nb = _FakeNotebook(tabs=['t1', 't2'], current='t2',
                            tab_texts={'t2': '社会接触者信息'})
        mixin.notebook = nb
        mixin._shortcut_add_contact()
        mixin._add_social_contact.assert_called_once()
        mixin._add_family_member.assert_not_called()

    def test_no_action_when_no_tab_selected(self, mixin):
        """select() 返回 None（无选中标签页）时不触发任何方法"""
        nb = _FakeNotebook(tabs=['t1'], current=None, tab_texts={})
        mixin.notebook = nb
        mixin._shortcut_add_contact()
        mixin._add_family_member.assert_not_called()
        mixin._add_social_contact.assert_not_called()

    def test_no_action_when_unknown_tab_text(self, mixin):
        """标签页文本既不含'家庭'也不含'社会'时不触发"""
        nb = _FakeNotebook(tabs=['t1'], current='t1',
                            tab_texts={'t1': '患者基本信息'})
        mixin.notebook = nb
        mixin._shortcut_add_contact()
        mixin._add_family_member.assert_not_called()
        mixin._add_social_contact.assert_not_called()

    def test_no_exception_when_notebook_missing(self, mixin):
        """无 notebook 属性时不应抛异常（被 try/except 捕获）"""
        # 不设置 notebook 属性
        mixin._shortcut_add_contact()  # 不应抛异常
        mixin._add_family_member.assert_not_called()

    def test_no_exception_when_tab_method_fails(self, mixin):
        """notebook.tab() 抛异常时应被捕获"""
        class _BadNotebook:
            def select(self):
                return 't1'
            def tab(self, tab_id, option=None):
                raise RuntimeError("simulated failure")

        mixin.notebook = _BadNotebook()
        mixin._shortcut_add_contact()  # 不应抛异常
        mixin._add_family_member.assert_not_called()


# ===========================================================================
# _switch_tab (Ctrl+Tab / Ctrl+Shift+Tab)
# ===========================================================================

class TestSwitchTab:
    """Ctrl+Tab / Ctrl+Shift+Tab: 标签页切换"""

    def test_switch_to_next_tab(self, mixin):
        """direction=1 应切换到下一个标签页"""
        nb = _FakeNotebook(tabs=['t1', 't2', 't3'], current='t1')
        mixin.notebook = nb
        mixin._switch_tab(1)
        assert nb.select_calls == ['t2']

    def test_switch_to_prev_tab(self, mixin):
        """direction=-1 应切换到上一个标签页"""
        nb = _FakeNotebook(tabs=['t1', 't2', 't3'], current='t2')
        mixin.notebook = nb
        mixin._switch_tab(-1)
        assert nb.select_calls == ['t1']

    def test_switch_wrap_around_from_last_to_first(self, mixin):
        """最后一个标签页 + 1 应 wrap 到第一个"""
        nb = _FakeNotebook(tabs=['t1', 't2', 't3'], current='t3')
        mixin.notebook = nb
        mixin._switch_tab(1)
        assert nb.select_calls == ['t1']

    def test_switch_wrap_around_from_first_to_last(self, mixin):
        """第一个标签页 - 1 应 wrap 到最后一个"""
        nb = _FakeNotebook(tabs=['t1', 't2', 't3'], current='t1')
        mixin.notebook = nb
        mixin._switch_tab(-1)
        assert nb.select_calls == ['t3']

    def test_no_action_when_no_tabs(self, mixin):
        """空 tabs 列表不应触发选择"""
        nb = _FakeNotebook(tabs=[], current=None)
        mixin.notebook = nb
        mixin._switch_tab(1)
        assert nb.select_calls == []

    def test_no_action_when_no_current_selection(self, mixin):
        """select() 返回 None 时不应触发"""
        nb = _FakeNotebook(tabs=['t1', 't2'], current=None)
        mixin.notebook = nb
        mixin._switch_tab(1)
        assert nb.select_calls == []

    def test_no_action_when_current_not_in_tabs(self, mixin):
        """当前选中标签页不在 tabs 列表中时不应触发"""
        nb = _FakeNotebook(tabs=['t1', 't2'], current='unknown')
        mixin.notebook = nb
        mixin._switch_tab(1)
        # tabs.index('unknown') 抛 ValueError，应被捕获
        assert nb.select_calls == []


# ===========================================================================
# _zoom_chart (Ctrl+加号 / Ctrl+减号)
# ===========================================================================

class TestZoomChart:
    """Ctrl+加号/减号: 图表缩放"""

    def _setup_zoomable_mixin(self, mixin):
        """配置 mixin 的图表缩放属性"""
        mixin._chart_zoom_factors = {
            'risk_canvas': 1.0,
            'seir_canvas': 1.0,
            'heatmap_canvas': 1.0,
            'ml_compare_canvas': 1.0,
        }
        # 设置 canvas 属性（非 None 即可触发缩放逻辑）
        mixin.risk_canvas = mock.MagicMock()
        mixin.seir_canvas = mock.MagicMock()
        mixin.heatmap_canvas = mock.MagicMock()
        mixin.ml_compare_canvas = mock.MagicMock()
        # 重绘方法（应被调用）
        mixin._draw_risk_chart = mock.MagicMock()
        mixin._draw_seir_curve = mock.MagicMock()
        mixin._draw_risk_heatmap = mock.MagicMock()
        mixin._update_ml_charts = mock.MagicMock()

    def test_zoom_in_increases_factor(self, mixin):
        """放大（factor=1.2）应增加缩放因子"""
        self._setup_zoomable_mixin(mixin)
        mixin._zoom_chart(1.2)
        assert mixin._chart_zoom_factors['risk_canvas'] == pytest.approx(1.2)

    def test_zoom_out_decreases_factor(self, mixin):
        """缩小（factor=1/1.2）应减少缩放因子"""
        self._setup_zoomable_mixin(mixin)
        mixin._zoom_chart(1 / 1.2)
        assert mixin._chart_zoom_factors['risk_canvas'] < 1.0

    def test_zoom_triggers_redraw(self, mixin):
        """缩放后应触发对应重绘方法"""
        self._setup_zoomable_mixin(mixin)
        mixin._zoom_chart(1.2)
        mixin._draw_risk_chart.assert_called_once()
        mixin._draw_seir_curve.assert_called_once()
        mixin._draw_risk_heatmap.assert_called_once()
        mixin._update_ml_charts.assert_called_once()

    def test_zoom_clamped_to_min(self, mixin):
        """缩放因子下限 0.2"""
        self._setup_zoomable_mixin(mixin)
        # 连续缩小多次，应被限制在 0.2
        for _ in range(20):
            mixin._zoom_chart(1 / 1.5)
        assert mixin._chart_zoom_factors['risk_canvas'] == pytest.approx(0.2)

    def test_zoom_clamped_to_max(self, mixin):
        """缩放因子上限 5.0"""
        self._setup_zoomable_mixin(mixin)
        # 连续放大多次，应被限制在 5.0
        for _ in range(20):
            mixin._zoom_chart(1.5)
        assert mixin._chart_zoom_factors['risk_canvas'] == pytest.approx(5.0)

    def test_zoom_skips_none_canvas(self, mixin):
        """canvas 为 None 的项应被跳过，不更新其缩放因子"""
        self._setup_zoomable_mixin(mixin)
        mixin.risk_canvas = None  # 设为 None
        original_factor = mixin._chart_zoom_factors['risk_canvas']
        mixin._zoom_chart(1.2)
        # risk_canvas 被跳过，缩放因子应保持不变
        assert mixin._chart_zoom_factors['risk_canvas'] == original_factor
        # 其他 canvas 应正常缩放
        assert mixin._chart_zoom_factors['seir_canvas'] == pytest.approx(1.2)

    def test_zoom_no_factors_attr_does_nothing(self, mixin):
        """无 _chart_zoom_factors 属性时不应抛异常"""
        # 不设置 _chart_zoom_factors
        mixin.risk_canvas = mock.MagicMock()
        mixin._draw_risk_chart = mock.MagicMock()
        mixin._zoom_chart(1.2)  # 不应抛异常
        mixin._draw_risk_chart.assert_not_called()

    def test_zoom_empty_factors_dict_does_nothing(self, mixin):
        """空 _chart_zoom_factors 字典时不应抛异常"""
        mixin._chart_zoom_factors = {}
        mixin.risk_canvas = mock.MagicMock()
        mixin._draw_risk_chart = mock.MagicMock()
        mixin._zoom_chart(1.2)
        mixin._draw_risk_chart.assert_not_called()

    def test_zoom_redraw_exception_does_not_propagate(self, mixin):
        """单个图表重绘失败不应影响其他图表"""
        self._setup_zoomable_mixin(mixin)
        mixin._draw_risk_chart.side_effect = RuntimeError("draw failed")
        # 不应抛异常
        mixin._zoom_chart(1.2)
        # 其他图表仍应被调用
        mixin._draw_seir_curve.assert_called_once()


# ===========================================================================
# _delete_selected_contact / _edit_selected_contact (Delete / Enter)
# ===========================================================================

class TestDeleteEditSelectedContact:
    """Delete 键删除 / Enter 键编辑选中行"""

    def test_delete_family_when_family_tab(self, mixin):
        """家庭标签页时调用 _delete_family_member"""
        nb = _FakeNotebook(tabs=['t1'], current='t1',
                            tab_texts={'t1': '家庭成员信息'})
        mixin.notebook = nb
        mixin._delete_selected_contact()
        mixin._delete_family_member.assert_called_once()
        mixin._delete_social_contact.assert_not_called()

    def test_delete_social_when_social_tab(self, mixin):
        """社会标签页时调用 _delete_social_contact"""
        nb = _FakeNotebook(tabs=['t1'], current='t1',
                            tab_texts={'t1': '社会接触者信息'})
        mixin.notebook = nb
        mixin._delete_selected_contact()
        mixin._delete_social_contact.assert_called_once()
        mixin._delete_family_member.assert_not_called()

    def test_delete_no_action_when_no_tab_selected(self, mixin):
        """无选中标签页时不触发"""
        nb = _FakeNotebook(tabs=['t1'], current=None)
        mixin.notebook = nb
        mixin._delete_selected_contact()
        mixin._delete_family_member.assert_not_called()
        mixin._delete_social_contact.assert_not_called()

    def test_delete_no_action_when_method_missing(self, mixin):
        """方法不存在（hasattr False）时不触发"""
        nb = _FakeNotebook(tabs=['t1'], current='t1',
                            tab_texts={'t1': '家庭成员信息'})
        mixin.notebook = nb
        # 删除方法
        del mixin._delete_family_member
        mixin._delete_selected_contact()  # 不应抛异常

    def test_edit_family_when_family_tab(self, mixin):
        """家庭标签页时调用 _edit_family_member"""
        nb = _FakeNotebook(tabs=['t1'], current='t1',
                            tab_texts={'t1': '家庭成员信息'})
        mixin.notebook = nb
        mixin._edit_selected_contact()
        mixin._edit_family_member.assert_called_once()
        mixin._edit_social_contact.assert_not_called()

    def test_edit_social_when_social_tab(self, mixin):
        """社会标签页时调用 _edit_social_contact"""
        nb = _FakeNotebook(tabs=['t1'], current='t1',
                            tab_texts={'t1': '社会接触者信息'})
        mixin.notebook = nb
        mixin._edit_selected_contact()
        mixin._edit_social_contact.assert_called_once()
        mixin._edit_family_member.assert_not_called()

    def test_edit_no_action_when_no_tab_selected(self, mixin):
        """无选中标签页时不触发"""
        nb = _FakeNotebook(tabs=['t1'], current=None)
        mixin.notebook = nb
        mixin._edit_selected_contact()
        mixin._edit_family_member.assert_not_called()
        mixin._edit_social_contact.assert_not_called()


# ===========================================================================
# _switch_theme (主题切换)
# ===========================================================================

class TestSwitchTheme:
    """主题切换：浅色/深色/未知主题回退"""

    def _setup_themable_mixin(self, mixin):
        """配置 mixin 的样式属性"""
        mixin.COLORS = {
            'bg_light': '#f8f9fa', 'bg_card': '#ffffff',
            'text': '#2c3e50', 'text_light': '#5a6c7d',
            'border': '#dee2e6', 'primary': '#2c3e50',
        }
        mixin.style = mock.MagicMock()
        return mixin

    def test_switch_to_dark_theme_updates_colors(self, mixin):
        """切换到深色主题应更新 COLORS"""
        self._setup_themable_mixin(mixin)
        mixin._switch_theme('dark')
        assert mixin.COLORS['bg_light'] == '#2d2d2d'
        assert mixin.COLORS['bg_card'] == '#3d3d3d'
        assert mixin.COLORS['text'] == '#e0e0e0'
        assert mixin._current_theme == 'dark'

    def test_switch_to_light_theme_updates_colors(self, mixin):
        """切换到浅色主题应更新 COLORS"""
        self._setup_themable_mixin(mixin)
        # 先切到 dark，再切回 light
        mixin._switch_theme('dark')
        mixin._switch_theme('light')
        assert mixin.COLORS['bg_light'] == '#f8f9fa'
        assert mixin.COLORS['text'] == '#2c3e50'
        assert mixin._current_theme == 'light'

    def test_switch_to_unknown_theme_falls_back_to_light(self, mixin):
        """未知主题名应回退到浅色"""
        self._setup_themable_mixin(mixin)
        mixin._switch_theme('unknown_theme')
        assert mixin.COLORS['bg_light'] == '#f8f9fa'  # light 默认值
        assert mixin._current_theme == 'unknown_theme'  # 仍记录主题名

    def test_switch_theme_calls_style_configure(self, mixin):
        """切换主题应调用 style.configure 更新 ttk 样式"""
        self._setup_themable_mixin(mixin)
        mixin._switch_theme('dark')
        # style.configure 应被多次调用（TFrame / TLabel / TLabelframe 等）
        assert mixin.style.configure.call_count >= 5

    def test_switch_theme_no_style_attr_does_nothing(self, mixin):
        """无 style 属性时不应抛异常"""
        mixin.COLORS = {'bg_light': '#fff'}
        # 不设置 style 属性
        mixin._switch_theme('dark')  # 不应抛异常

    def test_switch_theme_no_colors_attr_does_nothing(self, mixin):
        """无 COLORS 属性时不应抛异常（虽然实际中 _setup_styles 总会创建）"""
        mixin.style = mock.MagicMock()
        # 不设置 COLORS 属性，模拟异常路径
        if hasattr(mixin, 'COLORS'):
            del mixin.COLORS
        # 不应抛异常（_switch_theme 有 try/except 防护）
        mixin._switch_theme('dark')


# ===========================================================================
# _setup_keyboard_shortcuts (快捷键字典映射)
# ===========================================================================

class TestSetupKeyboardShortcuts:
    """快捷键绑定集中管理"""

    def test_setup_keyboard_shortcuts_binds_all_shortcuts(self):
        """所有快捷键应被绑定到 root"""
        m = BaseMixin()
        m.root = mock.MagicMock()

        # 提供所有快捷键回调依赖的方法
        m._load_from_csv = mock.MagicMock()
        m._save_to_csv = mock.MagicMock()
        m._export_results = mock.MagicMock()
        m._run_assessment = mock.MagicMock()
        m._on_closing = mock.MagicMock()
        m._undo = mock.MagicMock()
        m._redo = mock.MagicMock()
        m._close_top_dialog = mock.MagicMock()
        m._add_family_member = mock.MagicMock()
        m._add_social_contact = mock.MagicMock()
        m.notebook = _FakeNotebook()

        m._setup_keyboard_shortcuts()

        # root.bind 应被多次调用（每个快捷键一次）
        assert m.root.bind.call_count >= 15

        # 验证关键快捷键被绑定
        bound_keys = [call.args[0] for call in m.root.bind.call_args_list]
        assert '<Control-N>' in bound_keys  # Ctrl+N
        assert '<F1>' in bound_keys          # F1 帮助
        assert '<F5>' in bound_keys          # F5 评估
        assert '<Control-Tab>' in bound_keys
        assert '<Control-Shift-Tab>' in bound_keys
        assert '<Control-plus>' in bound_keys
        assert '<Control-minus>' in bound_keys
        assert '<Delete>' in bound_keys
        assert '<Return>' in bound_keys

    def test_setup_keyboard_shortcuts_handles_bind_failure(self):
        """单个 bind 失败不应影响其他快捷键"""
        m = BaseMixin()
        m.root = mock.MagicMock()
        # 第一个 bind 抛 TclError，后续返回 None
        m.root.bind.side_effect = [tk.TclError(), None, None, None, None,
                                    None, None, None, None, None,
                                    None, None, None, None, None,
                                    None, None, None, None, None]
        # 提供依赖方法
        m._load_from_csv = mock.MagicMock()
        m._save_to_csv = mock.MagicMock()
        m._export_results = mock.MagicMock()
        m._run_assessment = mock.MagicMock()
        m._on_closing = mock.MagicMock()
        m._undo = mock.MagicMock()
        m._redo = mock.MagicMock()
        m._close_top_dialog = mock.MagicMock()
        m._add_family_member = mock.MagicMock()
        m._add_social_contact = mock.MagicMock()
        m.notebook = _FakeNotebook()

        # 不应抛异常
        m._setup_keyboard_shortcuts()
        # 即使第一个失败，后续仍应继续
        assert m.root.bind.call_count >= 15
