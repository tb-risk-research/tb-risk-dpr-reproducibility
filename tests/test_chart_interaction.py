#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Section VI 图表交互层测试

覆盖：
- InteractiveChartMixin 状态管理（选中/注释/高亮）
- 安全事件注册与断开
- 命中测试辅助
- SEIR 悬停回调（_on_seir_hover）
- 热力图点击回调（_on_heatmap_click）
- 风险柱状图点击/悬停回调
- Configure 自动重绘绑定
- 模块级辅助函数（_contact_age_index 等）
- _update_all_charts 调用入口补齐
"""

import os
import sys
from unittest import mock

import pytest

# 将项目根目录加入 sys.path（与 test_field_validator.py 一致）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

# matplotlib 可用性
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import numpy as np

from tb_risk.gui.charts.interactive import InteractiveChartMixin
from tb_risk.gui.charts.risk_charts import (
    RiskChartMixin,
    _contact_age_index,
    _contact_risk_index,
    _group_contacts_by_cell,
)
from tb_risk.gui.charts.seir_charts import SEIRChartMixin


# ---------------------------------------------------------------------------
# 测试用桩 Mixin，提供 InteractiveChartMixin 所需的最小属性
# ---------------------------------------------------------------------------

class _StubMixin(InteractiveChartMixin):
    """最小化的桩对象，仅初始化交互状态"""

    def __init__(self):
        self._chart_selection = {}
        self._chart_annotations = {}
        self._chart_bars = {}
        self._chart_cids = {}
        self._chart_last_size = {}


class _SEIRStub(InteractiveChartMixin, SEIRChartMixin):
    """SEIR 图表测试桩，提供 _on_seir_hover 等方法"""

    def __init__(self):
        self._chart_selection = {}
        self._chart_annotations = {}
        self._chart_bars = {}
        self._chart_cids = {}
        self._chart_last_size = {}


class _RiskChartStub(InteractiveChartMixin, RiskChartMixin):
    """风险图表测试桩，提供 _on_heatmap_click / _on_risk_bar_click 等方法"""

    def __init__(self):
        self._chart_selection = {}
        self._chart_annotations = {}
        self._chart_bars = {}
        self._chart_cids = {}
        self._chart_last_size = {}


# ---------------------------------------------------------------------------
# 模块级辅助函数
# ---------------------------------------------------------------------------

class TestContactIndexHelpers:
    """测试 _contact_age_index / _contact_risk_index / _group_contacts_by_cell"""

    def test_age_index_boundaries(self):
        assert _contact_age_index(0) == 0     # <5
        assert _contact_age_index(4) == 0
        assert _contact_age_index(5) == 1     # 5-14
        assert _contact_age_index(14) == 1
        assert _contact_age_index(15) == 2    # 15-34
        assert _contact_age_index(34) == 2
        assert _contact_age_index(35) == 3    # 35-64
        assert _contact_age_index(64) == 3
        assert _contact_age_index(65) == 4    # 65+
        assert _contact_age_index(120) == 4

    def test_risk_index_boundaries(self):
        assert _contact_risk_index(0) == 0     # <10
        assert _contact_risk_index(9.9) == 0
        assert _contact_risk_index(10) == 1    # 10-24
        assert _contact_risk_index(24) == 1
        assert _contact_risk_index(25) == 2    # 25-39
        assert _contact_risk_index(39) == 2
        assert _contact_risk_index(40) == 3    # 40-59
        assert _contact_risk_index(59) == 3
        assert _contact_risk_index(60) == 4    # 60+
        assert _contact_risk_index(100) == 4

    def test_group_contacts_by_cell_empty(self):
        assert _group_contacts_by_cell([]) == {}

    def test_group_contacts_by_cell_groups(self):
        contacts = [
            {'name': 'A', 'age': 3, 'disease_probability': 5},    # (0, 0)
            {'name': 'B', 'age': 10, 'disease_probability': 15},  # (1, 1)
            {'name': 'C', 'age': 3, 'disease_probability': 5},    # (0, 0)
            {'name': 'D', 'age': 70, 'disease_probability': 80},  # (4, 4)
        ]
        groups = _group_contacts_by_cell(contacts)
        assert (0, 0) in groups and len(groups[(0, 0)]) == 2
        assert (1, 1) in groups and len(groups[(1, 1)]) == 1
        assert (4, 4) in groups and len(groups[(4, 4)]) == 1
        assert groups[(0, 0)][0]['name'] == 'A'

    def test_group_contacts_default_age(self):
        """age 缺失时使用默认值 30（落 15-34 组）"""
        contacts = [{'name': 'X', 'disease_probability': 50}]
        groups = _group_contacts_by_cell(contacts)
        assert (2, 3) in groups  # age=30→idx 2, prob=50→idx 3


# ---------------------------------------------------------------------------
# InteractiveChartMixin 状态管理
# ---------------------------------------------------------------------------

class TestInteractiveChartMixinState:
    """测试选中状态、注释、高亮管理"""

    def setup_method(self):
        self.stub = _StubMixin()

    def test_init_chart_interaction_idempotent(self):
        """多次调用 _init_chart_interaction 不覆盖已有状态"""
        self.stub._chart_selection['existing'] = [1, 2]
        self.stub._init_chart_interaction()
        assert self.stub._chart_selection['existing'] == [1, 2]

    def test_set_get_selection(self):
        self.stub._set_selection('chart1', [0, 2])
        assert self.stub._get_selection('chart1') == [0, 2]

    def test_get_selection_missing(self):
        assert self.stub._get_selection('nonexistent') == []

    def test_clear_selection_specific(self):
        self.stub._set_selection('a', [1])
        self.stub._set_selection('b', [2])
        self.stub._clear_selection('a')
        assert not self.stub._has_selection('a')
        assert self.stub._has_selection('b')

    def test_clear_selection_all(self):
        self.stub._set_selection('a', [1])
        self.stub._set_selection('b', [2])
        self.stub._clear_selection()
        assert not self.stub._has_selection()

    def test_has_selection_any(self):
        assert not self.stub._has_selection()
        self.stub._set_selection('x', [0])
        assert self.stub._has_selection()

    def test_has_selection_specific_empty(self):
        self.stub._set_selection('x', [])
        assert not self.stub._has_selection('x')

    def test_hide_all_annotations(self):
        """_hide_all_annotations 不抛错即使无注释"""
        self.stub._hide_all_annotations()
        assert self.stub._chart_annotations == {}


# ---------------------------------------------------------------------------
# 安全事件注册
# ---------------------------------------------------------------------------

class TestSafeMplConnect:
    """测试 _safe_mpl_connect / _disconnect_chart_events"""

    def setup_method(self):
        self.stub = _StubMixin()

    def test_connect_none_canvas(self):
        """canvas 为 None 时安全返回 None"""
        cid = self.stub._safe_mpl_connect(None, 'test', 'button_press_event', lambda e: None)
        assert cid is None

    def test_disconnect_unknown_chart(self):
        """断开未注册的图表不抛错"""
        self.stub._disconnect_chart_events(None, 'unknown')

    def test_disconnect_clears_cids(self):
        """断开后 cid 列表被清空"""
        self.stub._chart_cids['chart'] = [1, 2, 3]
        self.stub._disconnect_chart_events(None, 'chart')
        assert 'chart' not in self.stub._chart_cids


# ---------------------------------------------------------------------------
# 命中测试
# ---------------------------------------------------------------------------

class TestHitTest:
    """测试 _hit_test / _hit_test_multiple"""

    def setup_method(self):
        self.stub = _StubMixin()

    def test_hit_test_none_artist(self):
        hit, indices = self.stub._hit_test(None, None)
        assert hit is False
        assert indices == []

    def test_hit_test_multiple_empty(self):
        idx, indices = self.stub._hit_test_multiple([], None)
        assert idx is None
        assert indices == []

    def test_hit_test_multiple_skips_none(self):
        """None artist 被跳过"""
        idx, indices = self.stub._hit_test_multiple([None, None], None)
        assert idx is None


# ---------------------------------------------------------------------------
# SEIR 悬停回调
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib 不可用")
class TestSEIRHoverCallback:
    """测试 _on_seir_hover 回调"""

    def setup_method(self):
        self.stub = _SEIRStub()
        # 模拟 SEIR 数据
        self.weeks = np.linspace(0, 52, 100)
        self.S = np.full(100, 10.0)
        self.E = np.full(100, 5.0)
        self.I = np.full(100, 2.0)
        self.R = np.full(100, 3.0)
        self.stub._chart_bars['seir_curve'] = {
            'weeks': self.weeks, 'S': self.S, 'E': self.E,
            'I': self.I, 'R': self.R, 'total': 20,
        }

    def _make_event(self, xdata, inaxes=None):
        """构造 mock MouseEvent"""
        event = mock.Mock()
        event.inaxes = inaxes
        event.xdata = xdata
        return event

    def test_hover_none_event(self):
        """event 为 None 时不抛错"""
        self.stub._on_seir_hover(None)

    def test_hover_no_inaxes(self):
        """inaxes 为 None 时隐藏注释"""
        event = self._make_event(10.0, inaxes=None)
        self.stub._on_seir_hover(event)
        assert 'seir_curve' not in self.stub._chart_annotations

    def test_hover_no_data(self):
        """无 seir_curve 数据时安全返回"""
        stub = _SEIRStub()
        event = self._make_event(10.0, inaxes=mock.Mock())
        stub._on_seir_hover(event)

    def test_hover_xdata_none(self):
        """xdata 为 None 时隐藏注释"""
        event = mock.Mock()
        event.inaxes = mock.Mock()
        event.xdata = None
        self.stub._on_seir_hover(event)
        assert 'seir_curve' not in self.stub._chart_annotations

    def test_hover_finds_nearest_timepoint(self):
        """悬停时找到最近时间点并显示注释"""
        fig = Figure()
        ax = fig.add_subplot(111)
        event = self._make_event(10.0, inaxes=ax)
        self.stub._on_seir_hover(event)
        assert 'seir_curve' in self.stub._chart_annotations

    def test_hover_clips_out_of_range(self):
        """xdata 超出范围时被 clip 到有效索引"""
        fig = Figure()
        ax = fig.add_subplot(111)
        event = self._make_event(1000.0, inaxes=ax)
        self.stub._on_seir_hover(event)
        assert 'seir_curve' in self.stub._chart_annotations

    def test_hover_negative_xdata(self):
        """xdata 为负数时被 clip 到索引 0"""
        fig = Figure()
        ax = fig.add_subplot(111)
        event = self._make_event(-10.0, inaxes=ax)
        self.stub._on_seir_hover(event)
        assert 'seir_curve' in self.stub._chart_annotations


# ---------------------------------------------------------------------------
# 热力图点击回调
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib 不可用")
class TestHeatmapClickCallback:
    """测试 _on_heatmap_click 回调"""

    def setup_method(self):
        self.stub = _RiskChartStub()
        age_groups = ['<5岁', '5-14岁', '15-35岁', '36-65岁', '>65岁']
        risk_groups = ['低风险', '中低风险', '中风险', '中高风险', '高风险']
        self.stub._chart_bars['heatmap'] = {
            'count_matrix': np.array([[0, 1, 0, 0, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 2, 0, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 1]]),
            'avg_risk_matrix': np.zeros((5, 5)),
            'contacts_by_cell': {
                (0, 1): [{'name': 'A', 'type': '家庭', 'age': 3, 'disease_probability': 12.0}],
                (2, 2): [{'name': 'B', 'type': '社会', 'age': 20, 'disease_probability': 30.0},
                          {'name': 'C', 'type': '家庭', 'age': 25, 'disease_probability': 35.0}],
                (4, 4): [{'name': 'D', 'type': '家庭', 'age': 70, 'disease_probability': 85.0}],
            },
            'age_groups': age_groups,
            'risk_groups': risk_groups,
            'im': None,
        }

    def _make_event(self, xdata, ydata, inaxes=None):
        event = mock.Mock()
        event.inaxes = inaxes
        event.xdata = xdata
        event.ydata = ydata
        return event

    def test_click_none_event(self):
        self.stub._on_heatmap_click(None)

    def test_click_no_inaxes(self):
        event = self._make_event(1.0, 0.0, inaxes=None)
        self.stub._on_heatmap_click(event)

    def test_click_no_data(self):
        stub = _RiskChartStub()
        event = self._make_event(1.0, 0.0, inaxes=mock.Mock())
        stub._on_heatmap_click(event)

    def test_click_empty_cell(self):
        """点击空单元格显示无接触者"""
        fig = Figure()
        ax = fig.add_subplot(111)
        event = self._make_event(0.5, 0.5, inaxes=ax)  # (row=0, col=0) 空
        self.stub._on_heatmap_click(event)
        assert 'heatmap' in self.stub._chart_annotations

    def test_click_populated_cell(self):
        """点击有接触者的单元格展示列表"""
        fig = Figure()
        ax = fig.add_subplot(111)
        event = self._make_event(1.5, 0.5, inaxes=ax)  # (row=0, col=1) 有 1 人
        self.stub._on_heatmap_click(event)
        assert 'heatmap' in self.stub._chart_annotations
        assert self.stub._has_selection('heatmap')

    def test_click_clips_out_of_range(self):
        """点击超出范围时被 clip"""
        fig = Figure()
        ax = fig.add_subplot(111)
        event = self._make_event(100.0, 100.0, inaxes=ax)
        self.stub._on_heatmap_click(event)
        assert 'heatmap' in self.stub._chart_annotations

    def test_click_xdata_none(self):
        """xdata 为 None 时安全返回"""
        fig = Figure()
        ax = fig.add_subplot(111)
        event = self._make_event(None, 0.5, inaxes=ax)
        self.stub._on_heatmap_click(event)


# ---------------------------------------------------------------------------
# 风险柱状图点击/悬停回调
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib 不可用")
class TestRiskBarCallbacks:
    """测试 _on_risk_bar_click / _on_risk_bar_hover"""

    def setup_method(self):
        self.stub = _RiskChartStub()
        # 构造 5 个 bar 对象
        fig = Figure()
        ax = fig.add_subplot(111)
        self.bars = list(ax.bar(['A', 'B', 'C', 'D', 'E'], [1, 2, 3, 4, 5]))
        self.stub._chart_bars['risk_bar'] = self.bars
        self.ax = ax

    def _make_event(self, xdata, ydata, inaxes=None):
        event = mock.Mock()
        event.inaxes = inaxes
        event.xdata = xdata
        event.ydata = ydata
        return event

    def test_click_none_event(self):
        self.stub._on_risk_bar_click(None)

    def test_click_no_inaxes(self):
        event = self._make_event(0.5, 0.5, inaxes=None)
        self.stub._on_risk_bar_click(event)

    def test_click_no_bars(self):
        stub = _RiskChartStub()
        event = self._make_event(0.5, 0.5, inaxes=self.ax)
        stub._on_risk_bar_click(event)

    def test_hover_none_event(self):
        self.stub._on_risk_bar_hover(None)

    def test_hover_no_inaxes(self):
        event = self._make_event(0.5, 0.5, inaxes=None)
        self.stub._on_risk_bar_hover(event)


# ---------------------------------------------------------------------------
# Configure 自动重绘
# ---------------------------------------------------------------------------

class TestConfigureRedraw:
    """测试 _bind_configure_redraw"""

    def setup_method(self):
        self.stub = _StubMixin()

    def test_bind_none_widget(self):
        """widget 为 None 时安全返回"""
        self.stub._bind_configure_redraw(None, lambda: None, 'test')

    def test_bind_records_last_size(self):
        """绑定后记录初始尺寸"""
        # 使用 mock 控件
        widget = mock.Mock()
        widget.bind = mock.Mock()
        self.stub._bind_configure_redraw(widget, lambda: None, 'test_chart')
        assert 'test_chart' in self.stub._chart_last_size
        widget.bind.assert_called_once()

    def test_bind_multiple_charts(self):
        """多个图表各自独立记录尺寸"""
        w1 = mock.Mock()
        w2 = mock.Mock()
        self.stub._bind_configure_redraw(w1, lambda: None, 'chart1')
        self.stub._bind_configure_redraw(w2, lambda: None, 'chart2')
        assert 'chart1' in self.stub._chart_last_size
        assert 'chart2' in self.stub._chart_last_size


# ---------------------------------------------------------------------------
# Esc 键清除选中
# ---------------------------------------------------------------------------

class TestEscapeClearSelection:
    """测试 _on_escape_pressed"""

    def setup_method(self):
        self.stub = _StubMixin()

    def test_escape_clears_all_selections(self):
        self.stub._set_selection('a', [1])
        self.stub._set_selection('b', [2])
        self.stub._on_escape_pressed()
        assert not self.stub._has_selection()

    def test_escape_no_selection_safe(self):
        """无选中时 Esc 不抛错"""
        self.stub._on_escape_pressed()

    def test_bind_esc_none_widget(self):
        """widget 为 None 时安全返回"""
        self.stub._bind_esc_to_clear_selection(None)


# ---------------------------------------------------------------------------
# _update_all_charts 调用入口补齐
# ---------------------------------------------------------------------------

class TestUpdateAllChartsHooks:
    """测试 _update_all_charts 中的 hasattr 守卫调用入口"""

    def setup_method(self):
        """构造最小桩对象，组合 RiskChartMixin + 必要属性"""
        class _ChartStub(RiskChartMixin):
            def __init__(self):
                self.results = {'potential_patients': {'family': [], 'social': []}}
                self.patient_info = {'basic_info': {}}
                self.root = mock.Mock()
                self._chart_selection = {}
                self._chart_annotations = {}
                self._chart_bars = {}
                self._chart_cids = {}
                # 各 _draw_* 方法替换为 mock，避免依赖 matplotlib canvas
                self._draw_seir_curve = mock.Mock()
                self._draw_risk_heatmap = mock.Mock()
                self._draw_probability_histogram = mock.Mock()
                self._draw_delay_impact_curve = mock.Mock()
                self._init_chart_interaction = mock.Mock()
        self.stub = _ChartStub()

    def test_update_all_charts_calls_basic_four(self):
        """_update_all_charts 调用基础 4 个图表"""
        self.stub._update_all_charts()
        self.stub._draw_seir_curve.assert_called_once()
        self.stub._draw_risk_heatmap.assert_called_once()
        self.stub._draw_probability_histogram.assert_called_once()
        self.stub._draw_delay_impact_curve.assert_called_once()

    def test_update_all_charts_no_results(self):
        """results 为空时不绘制"""
        self.stub.results = None
        self.stub._update_all_charts()
        self.stub._draw_seir_curve.assert_not_called()

    def test_update_all_charts_skips_missing_ml_charts(self):
        """无 _update_ml_charts 时不抛错"""
        self.stub._update_all_charts()  # 不应抛 AttributeError

    def test_update_all_charts_calls_ml_charts_if_present(self):
        """有 _update_ml_charts 时调用"""
        self.stub._update_ml_charts = mock.Mock()
        self.stub._update_all_charts()
        self.stub._update_ml_charts.assert_called_once()

    def test_update_all_charts_calls_causal_dag_if_present(self):
        """有 _draw_causal_dag 时调用"""
        self.stub._draw_causal_dag = mock.Mock()
        self.stub._update_all_charts()
        self.stub._draw_causal_dag.assert_called_once()

    def test_update_all_charts_calls_mcmc_diagnostics_if_present(self):
        """有 _draw_mcmc_diagnostics 时调用"""
        self.stub._draw_mcmc_diagnostics = mock.Mock()
        self.stub._update_all_charts()
        self.stub._draw_mcmc_diagnostics.assert_called_once()

    def test_update_all_charts_calls_gnn_topology_if_present(self):
        """有 _draw_gnn_topology 时调用"""
        self.stub._draw_gnn_topology = mock.Mock()
        self.stub._update_all_charts()
        self.stub._draw_gnn_topology.assert_called_once()

    def test_update_all_charts_stochastic_seir_disabled_by_default(self):
        """默认不调用随机 SEIR（use_stochastic_seir 未设置）"""
        self.stub._draw_stochastic_seir_curve = mock.Mock()
        self.stub._update_all_charts()
        self.stub._draw_stochastic_seir_curve.assert_not_called()

    def test_update_all_charts_stochastic_seir_enabled(self):
        """启用 use_stochastic_seir 时调用随机 SEIR"""
        self.stub._draw_stochastic_seir_curve = mock.Mock()
        self.stub.use_stochastic_seir = True
        self.stub._update_all_charts()
        self.stub._draw_stochastic_seir_curve.assert_called_once()

    def test_update_all_charts_swallows_exceptions(self):
        """单个图表抛异常时不影响其他图表"""
        self.stub._draw_seir_curve.side_effect = ValueError("test error")
        self.stub._update_all_charts()  # 不应抛错
        self.stub._draw_risk_heatmap.assert_called_once()
