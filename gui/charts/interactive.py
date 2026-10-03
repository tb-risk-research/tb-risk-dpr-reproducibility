#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 图表交互基类（Section VI）

InteractiveChartMixin 封装通用的交互事件注册、命中测试、高亮管理逻辑。
每个具体图表类继承并实现自己的回调语义，新增图表类型时自动获得基础交互能力。

设计要点：
  - 所有 mpl_connect 回调中 try-except 包裹，防止回调异常导致 matplotlib
    内部状态损坏。
  - 点击事件的命中测试使用 matplotlib 的 contains 方法而非手算坐标变换。
  - 交互状态可清除（Esc 键取消所有选中高亮）。
  - matplotlib 不可用时所有方法安全降级为空操作。
"""

from ._shared import *


class SelectionBroker:
    """优先级七：图表间联动选中中介者

    各图表点击回调在更新自身选中状态后，通知 SelectionBroker 广播选中条件。
    其他图表订阅 SelectionBroker 的事件，收到通知后根据条件过滤自身数据并更新高亮。

    联动关系（叠加过滤，非排他）：
      - 热力图选中 → 概率直方图过滤为对应人群分布
      - 风险柱状图选中 → SEIR 曲线标注该接触者的风险贡献
      - SHAP 图选中 → 风险柱状图按该特征值重新排序
      - Esc 键清除所有联动选中

    选中条件格式：
      - 热力图: {'age_group': int, 'risk_level': int}
      - 风险柱状图: {'indicator': str}
      - SHAP 图: {'feature': str}
    """

    def __init__(self):
        # chart_name -> callback(source_chart, selection)
        self._subscribers = {}
        # source_chart -> selection_dict
        self._active_filters = {}

    def subscribe(self, chart_name, callback):
        """订阅选中事件

        Args:
            chart_name: 订阅方图表标识（如 'histogram'）
            callback: 回调函数，签名为 callback(source_chart, selection)
        """
        self._subscribers[chart_name] = callback

    def unsubscribe(self, chart_name):
        """取消订阅"""
        self._subscribers.pop(chart_name, None)
        self._active_filters.pop(chart_name, None)

    def notify(self, source_chart, selection):
        """通知所有订阅者（除了源图表）

        Args:
            source_chart: 触发选中的源图表标识
            selection: 选中条件字典（如 {'age_group': 2, 'risk_level': 3}）
        """
        if selection is None:
            self._active_filters.pop(source_chart, None)
        else:
            self._active_filters[source_chart] = selection
        # 广播给所有其他订阅者
        for chart_name, callback in self._subscribers.items():
            if chart_name != source_chart:
                try:
                    callback(source_chart, selection)
                except Exception as e:
                    LOGGER.warning("SelectionBroker 通知 %s 失败: %s", chart_name, e)

    def clear_all(self):
        """清除所有联动选中（Esc 键触发）"""
        self._active_filters.clear()
        # 通知所有订阅者清除过滤
        for chart_name, callback in self._subscribers.items():
            try:
                callback('__clear_all__', None)
            except Exception as e:
                LOGGER.debug("SelectionBroker 清除 %s 失败: %s", chart_name, e)

    def get_filters(self, exclude_chart=None):
        """获取所有活跃的过滤条件

        Args:
            exclude_chart: 排除的图表名（通常为查询方自身）

        Returns:
            dict: {source_chart: selection_dict}
        """
        if exclude_chart:
            return {k: v for k, v in self._active_filters.items() if k != exclude_chart}
        return dict(self._active_filters)

    def has_filters(self):
        """是否有活跃的过滤条件"""
        return len(self._active_filters) > 0


class InteractiveChartMixin:
    """图表交互基础设施 Mixin

    提供通用的交互事件注册、命中测试、高亮管理。
    子类（如 RiskChartMixin / SEIRChartMixin）继承后可实现具体回调语义。

    交互状态：
      - self._chart_selection: dict, {chart_name: selected_indices}
      - self._chart_annotations: dict, {chart_name: annotation_artist}
      - self._chart_bars: dict, {chart_name: bar_artists} 用于高亮
      - self._selection_broker: SelectionBroker, 图表间联动中介者（优先级七）
    """

    def _init_chart_interaction(self):
        """初始化图表交互状态（在 init_gui 或 _init_result_tab 中调用一次）"""
        if not hasattr(self, '_chart_selection'):
            self._chart_selection = {}
        if not hasattr(self, '_chart_annotations'):
            self._chart_annotations = {}
        if not hasattr(self, '_chart_bars'):
            self._chart_bars = {}
        if not hasattr(self, '_chart_cids'):
            self._chart_cids = {}  # chart_name -> [connection ids]
        # 优先级七：初始化 SelectionBroker 中介者
        if not hasattr(self, '_selection_broker'):
            self._selection_broker = SelectionBroker()

    # ==================================================================
    # 安全事件注册
    # ==================================================================

    def _safe_mpl_connect(self, canvas, chart_name, event, callback):
        """安全地注册 mpl_connect 事件回调

        Args:
            canvas: FigureCanvasTkAgg 实例
            chart_name: 图表标识（如 'risk_bar', 'heatmap', 'seir'）
            event: matplotlib 事件名（如 'button_press_event', 'motion_notify_event'）
            callback: 回调函数，签名为 callback(event)

        Returns:
            connection id（成功时）或 None（失败时）
        """
        if not MATPLOTLIB_AVAILABLE or canvas is None:
            return None
        try:
            cid = canvas.mpl_connect(event, callback)
            if chart_name not in self._chart_cids:
                self._chart_cids[chart_name] = []
            self._chart_cids[chart_name].append(cid)
            return cid
        except Exception as e:
            LOGGER.warning("注册 %s 的 %s 事件失败: %s", chart_name, event, e)
            return None

    def _disconnect_chart_events(self, canvas, chart_name):
        """断开指定图表的所有事件连接"""
        # 先弹出 cid 列表（即使 canvas 为 None 也清理本地状态）
        cids = self._chart_cids.pop(chart_name, [])
        if not MATPLOTLIB_AVAILABLE or canvas is None:
            return
        for cid in cids:
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass

    # ==================================================================
    # 命中测试
    # ==================================================================

    def _hit_test(self, artist, mouseevent):
        """使用 matplotlib 的 contains 方法进行命中测试

        Args:
            artist: matplotlib Artist 对象（如 BarContainer、Line2D、Patch）
            mouseevent: matplotlib MouseEvent

        Returns:
            (hit, indices) — hit 为 bool，indices 为命中的索引列表
        """
        if artist is None or mouseevent is None:
            return False, []
        try:
            contains, attrd = artist.contains(mouseevent)
            if not contains:
                return False, []
            # contains 返回的 attrd 通常含 'ind' 键（索引数组）
            indices = attrd.get('ind', []) if isinstance(attrd, dict) else []
            # numpy 数组转列表
            if hasattr(indices, 'tolist'):
                indices = indices.tolist()
            return True, list(indices)
        except Exception:
            return False, []

    def _hit_test_multiple(self, artists, mouseevent):
        """对多个 artist 进行命中测试，返回首个命中

        Args:
            artists: list of Artist
            mouseevent: matplotlib MouseEvent

        Returns:
            (artist_index, hit_indices) 或 (None, [])
        """
        for i, artist in enumerate(artists):
            if artist is None:
                continue
            hit, indices = self._hit_test(artist, mouseevent)
            if hit:
                return i, indices
        return None, []

    # ==================================================================
    # 选中状态管理
    # ==================================================================

    def _set_selection(self, chart_name, indices):
        """设置图表选中状态

        Args:
            chart_name: 图表标识
            indices: 选中的索引列表
        """
        self._chart_selection[chart_name] = list(indices)

    def _get_selection(self, chart_name):
        """获取图表选中状态"""
        return self._chart_selection.get(chart_name, [])

    def _clear_selection(self, chart_name=None):
        """清除选中状态

        Args:
            chart_name: 指定图表名；None 时清除所有图表选中
        """
        if chart_name is None:
            self._chart_selection.clear()
        else:
            self._chart_selection.pop(chart_name, None)

    def _has_selection(self, chart_name=None):
        """是否有选中状态"""
        if chart_name is None:
            return len(self._chart_selection) > 0
        return chart_name in self._chart_selection and len(self._chart_selection[chart_name]) > 0

    # ==================================================================
    # 悬停注释（Tooltip）
    # ==================================================================

    def _show_hover_annotation(self, ax, x, y, text, chart_name='default'):
        """在图表上显示悬停注释

        Args:
            ax: matplotlib Axes
            x, y: 注释位置（data 坐标）
            text: 注释文本
            chart_name: 图表标识（用于管理注释生命周期）
        """
        if not MATPLOTLIB_AVAILABLE or ax is None:
            return
        # 先隐藏已有注释
        self._hide_hover_annotation(chart_name)
        try:
            annot = ax.annotate(
                text, xy=(x, y), xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8,
                          edgecolor='gray'),
                fontsize=9, zorder=100,
            )
            self._chart_annotations[chart_name] = annot
        except Exception as e:
            LOGGER.warning("显示悬停注释失败: %s", e)

    def _hide_hover_annotation(self, chart_name='default'):
        """隐藏悬停注释"""
        annot = self._chart_annotations.pop(chart_name, None)
        if annot is not None:
            try:
                annot.set_visible(False)
                # 从 axes 中移除
                if hasattr(annot, 'remove'):
                    annot.remove()
            except Exception:
                pass

    def _hide_all_annotations(self):
        """隐藏所有悬停注释"""
        for chart_name in list(self._chart_annotations.keys()):
            self._hide_hover_annotation(chart_name)

    # ==================================================================
    # 高亮管理
    # ==================================================================

    def _highlight_bars(self, bars, selected_indices, chart_name='default',
                        highlight_color='#FFD700', normal_color=None):
        """高亮选中的柱子

        Args:
            bars: list of matplotlib.patches.Rectangle（bar 对象）
            selected_indices: 选中的索引列表
            chart_name: 图表标识
            highlight_color: 高亮颜色
            normal_color: 恢复颜色（None 时不恢复）
        """
        try:
            for i, bar in enumerate(bars):
                if i in selected_indices:
                    bar.set_facecolor(highlight_color)
                    bar.set_edgecolor('red')
                    bar.set_linewidth(2)
                elif normal_color is not None:
                    bar.set_facecolor(normal_color)
                    bar.set_edgecolor('black')
                    bar.set_linewidth(1)
        except Exception as e:
            LOGGER.warning("高亮柱子失败: %s", e)

    def _redraw_chart_canvas(self, canvas):
        """安全重绘图表 canvas"""
        if canvas is None:
            return
        try:
            canvas.draw_idle()
        except Exception:
            pass

    # ==================================================================
    # Esc 键清除选中
    # ==================================================================

    def _bind_esc_to_clear_selection(self, root_widget):
        """绑定 Esc 键清除所有图表选中

        Args:
            root_widget: tkinter 根窗口或控件
        """
        if not TKINTER_AVAILABLE or root_widget is None:
            return
        try:
            root_widget.bind('<Escape>', lambda e: self._on_escape_pressed())
        except Exception:
            pass

    def _on_escape_pressed(self):
        """Esc 键回调：清除所有选中 + 隐藏注释 + 清除联动过滤"""
        self._clear_selection()
        self._hide_all_annotations()
        # 优先级七：清除所有联动过滤
        if hasattr(self, '_selection_broker') and self._selection_broker is not None:
            self._selection_broker.clear_all()
        # 重绘所有已注册的 canvas 以移除高亮
        for chart_name, bars in getattr(self, '_chart_bars', {}).items():
            try:
                for bar in bars:
                    bar.set_facecolor(bar.get_facecolor())  # 保持原色
                    bar.set_linewidth(1)
                    bar.set_edgecolor('black')
            except Exception:
                pass
        # 触发各 canvas 重绘
        for attr_name in ['risk_canvas', 'heatmap_canvas', 'seir_canvas',
                          'histogram_canvas', 'delay_canvas']:
            canvas = getattr(self, attr_name, None)
            self._redraw_chart_canvas(canvas)

    # ==================================================================
    # 优先级七：SelectionBroker 联动辅助方法
    # ==================================================================

    def _notify_broker(self, source_chart, selection):
        """通知 SelectionBroker 广播选中事件

        Args:
            source_chart: 触发选中的源图表标识
            selection: 选中条件字典（None 表示清除该图表的过滤）
        """
        if hasattr(self, '_selection_broker') and self._selection_broker is not None:
            self._selection_broker.notify(source_chart, selection)

    def _register_chart_with_broker(self, chart_name, callback):
        """注册图表到 SelectionBroker 订阅列表

        Args:
            chart_name: 订阅方图表标识
            callback: 回调函数，签名为 callback(source_chart, selection)
        """
        if hasattr(self, '_selection_broker') and self._selection_broker is not None:
            self._selection_broker.subscribe(chart_name, callback)

    def _get_broker_filters(self, exclude_chart=None):
        """获取所有活跃的联动过滤条件"""
        if hasattr(self, '_selection_broker') and self._selection_broker is not None:
            return self._selection_broker.get_filters(exclude_chart)
        return {}

    # ==================================================================
    # <Configure> 自动重绘
    # ==================================================================

    def _bind_configure_redraw(self, widget, redraw_func=None, chart_name='default',
                               figure=None, base_height=None):
        """绑定 <Configure> 事件实现窗口缩放时自动重绘

        防抖：仅在尺寸实际变化时触发重绘，避免频繁调用。
        若传入 figure，会在容器尺寸变化时按容器实际【宽高】更新 figsize，
        使图表大小与容器完全匹配（铺满容器），从根本上解决：
          - 右侧/底部裁切（边框不完整）
          - 上下留白（图表不居中）
        redraw_func 可为 None：此时仅同步 figure 尺寸，不触发重绘函数。

        Args:
            widget: tkinter 控件
            redraw_func: 可选重绘函数（无参数）
            chart_name: 图表标识
            figure: 可选 matplotlib Figure，用于随容器尺寸自适应
            base_height: 保留参数（不再使用固定高度，高度由容器实时决定）
        """
        if not TKINTER_AVAILABLE or widget is None:
            return
        # 记录上次尺寸用于防抖
        if not hasattr(self, '_chart_last_size'):
            self._chart_last_size = {}
        if chart_name not in self._chart_last_size:
            self._chart_last_size[chart_name] = (0, 0)

        # 改进10：包装 figure.canvas.draw()，使每次渲染前都强制将 figure 尺寸
        # 对齐到 canvas 控件实际像素尺寸。这是最可靠的铺满手段——无论 matplotlib
        # 内部是否随控件自动缩放，每次出图都会重新对齐，彻底消除偏左留白/裁切。
        if figure is not None:
            try:
                canvas = figure.canvas
                if canvas is not None and not getattr(canvas, '_tb_risk_fitted', False):
                    orig_draw = canvas.draw

                    def _fitted_draw(*args, **kwargs):
                        try:
                            self._fit_figure_to_canvas(canvas, figure)
                        except Exception:
                            pass
                        return orig_draw(*args, **kwargs)

                    canvas.draw = _fitted_draw
                    canvas._tb_risk_fitted = True
            except Exception:
                pass

        def _on_configure(_event):
            try:
                w = widget.winfo_width()
                h = widget.winfo_height()
                last = self._chart_last_size.get(chart_name, (0, 0))
                # 仅在尺寸变化超过 5 像素时重绘（防抖）
                if abs(w - last[0]) > 5 or abs(h - last[1]) > 5:
                    self._chart_last_size[chart_name] = (w, h)
                    # 按容器实际宽高更新 figsize，使图表铺满容器（居中、边框完整）
                    if figure is not None and w > 50 and h > 50:
                        try:
                            dpi = figure.get_dpi() or 100
                            fig_width = min(20.0, max(2.0, w / dpi))
                            fig_height = min(20.0, max(2.0, h / dpi))
                            figure.set_size_inches(fig_width, fig_height)
                            try:
                                figure.canvas.draw_idle()
                            except Exception:
                                pass
                        except Exception as e:
                            LOGGER.debug("%s 更新 figsize 失败: %s", chart_name, e)
                    if redraw_func is not None:
                        redraw_func()
            except Exception as e:
                LOGGER.debug("%s Configure 重绘失败: %s", chart_name, e)

        try:
            widget.bind('<Configure>', _on_configure, add=True)
        except Exception:
            pass

        def _initial_sync():
            """控件完成布局后立即同步一次 figure 尺寸并重绘。

            Configure 事件在控件首次布局时可能携带中间尺寸，导致图表
            保持初始 figsize（偏大/偏小）而出现裁切或不居中。此回调在
            布局稳定后强制同步一次，保证首次显示即铺满容器。
            """
            try:
                widget.update_idletasks()
                w = widget.winfo_width()
                h = widget.winfo_height()
                if w > 50 and h > 50:
                    self._chart_last_size[chart_name] = (w, h)
                    if figure is not None:
                        dpi = figure.get_dpi() or 100
                        figure.set_size_inches(min(20.0, max(2.0, w / dpi)),
                                               min(20.0, max(2.0, h / dpi)))
                        try:
                            figure.canvas.draw_idle()
                        except Exception:
                            pass
                    if redraw_func is not None:
                        redraw_func()
            except Exception as e:
                LOGGER.debug("%s 初始尺寸同步失败: %s", chart_name, e)

        try:
            widget.after(80, _initial_sync)
        except Exception:
            pass

    def _fit_figure_to_canvas(self, canvas, figure):
        """将 figure 尺寸强制对齐到 canvas 控件实际像素尺寸（改进10）。

        每次渲染前调用，保证图表铺满控件、边框完整且居中。
        figure 为 None 时自动通过属性命名约定（*_canvas <-> *_figure）反查。
        """
        if canvas is None:
            return
        if figure is None:
            figure = self._figure_for_canvas(canvas)
        if figure is None:
            return
        try:
            widget = canvas.get_tk_widget()
            w = widget.winfo_width()
            h = widget.winfo_height()
            if w > 50 and h > 50:
                dpi = figure.get_dpi() or 100
                figure.set_size_inches(min(20.0, max(1.0, w / dpi)),
                                       min(20.0, max(1.0, h / dpi)))
        except Exception:
            pass

    def _figure_for_canvas(self, canvas):
        """根据 canvas 对象反查其对应的 figure（按属性命名约定 *_canvas <-> *_figure）"""
        if canvas is None:
            return None
        try:
            for attr, value in self.__dict__.items():
                if value is canvas and attr.endswith('_canvas'):
                    fig_attr = attr[:-len('_canvas')] + '_figure'
                    fig = self.__dict__.get(fig_attr)
                    if fig is not None:
                        return fig
            for attr, value in self.__dict__.items():
                if value is canvas:
                    for cand in ('_figure',):
                        if False:
                            pass
        except Exception:
            pass
        return None


__all__ = ['InteractiveChartMixin', 'SelectionBroker']
