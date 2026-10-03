#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""部署与监控面板 - 漂移监控子标签

双视图：
  1. 分布漂移 — 基于评估日志（assessment_log.jsonl）计算数值特征/评分
     分布的 PSI 与布尔特征阳性率偏移，附告警汇总与噪声底提示。
  2. 性能趋势 — 从训练档案读取各 run 的 AUROC/AUPRC，按 模型类型@数据集
     分桶绘制时间序列，best.json 当前最佳作为参考线。
"""

from ._shared import *  # noqa: F401,F403
# 星号导入不携带下划线名，显式补充
from ._shared import _deploy, _alog  # noqa: F401

_LEVEL_COLORS_KEYS = {
    '稳定': 'risk_low',
    '轻度漂移': 'risk_medium',
    '显著漂移': 'risk_high',
    '不可计算': 'disabled',
}


class DriftMonitorTabMixin:
    """漂移监控：PSI 分布漂移 + 性能趋势"""

    # ==================== 界面构建 ====================

    def _build_drift_monitor_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text='漂移监控')

        self._trend_data = None

        # ── 顶部：刷新 + 数据源 ──
        top = ttk.Frame(tab)
        top.pack(fill='x', padx=10, pady=(10, 4))
        ttk.Button(top, text='⟳ 刷新监控数据', style='Accent.TButton',
                   command=self._drift_refresh).pack(side='left', padx=(0, 8))
        self._drift_dir_label = ttk.Label(top, text='', style='Tip.TLabel')
        self._drift_dir_label.pack(side='left')

        # ── 汇总卡片 ──
        cards = ttk.Frame(tab)
        cards.pack(fill='x', padx=10, pady=(0, 6))
        self._drift_cards = {}
        for i, (key, label) in enumerate([
                ('n_records', '评估日志记录'),
                ('level', '总体漂移等级'),
                ('alerts', '告警条数'),
                ('noise', 'PSI 噪声底')]):
            cards.columnconfigure(i, weight=1, uniform='drift_card')
            card = ttk.Frame(cards, style='Card.TFrame', padding=8)
            card.grid(row=0, column=i, sticky='nsew', padx=2)
            ttk.Label(card, text=label, font=self._FONT_SMALL,
                      style='CardTitle.TLabel').pack(anchor='w')
            val = ttk.Label(card, text='--', style='CardValue.TLabel')
            val.pack(anchor='w')
            self._drift_cards[key] = val

        # ── 双视图子 Notebook ──
        sub = ttk.Notebook(tab)
        sub.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        self._build_psi_view(sub)
        self._build_trend_view(sub)

        # 初次进入自动刷新一次
        self._drift_refresh()

    # -------------------- 视图 1：分布漂移 --------------------

    def _build_psi_view(self, sub):
        view = ttk.Frame(sub)
        sub.add(view, text='分布漂移 (PSI)')

        self._drift_hint_label = ttk.Label(
            view, text='PSI 判读：<0.10 稳定 / 0.10~0.25 轻度漂移 / ≥0.25 显著漂移',
            style='Tip.TLabel')
        self._drift_hint_label.pack(anchor='w', padx=4, pady=(6, 2))

        paned = ttk.PanedWindow(view, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=4, pady=4)

        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        feat_frame = ttk.LabelFrame(left, text='数值特征与评分分布', padding=4)
        feat_frame.pack(fill='both', expand=True)
        fcols = ('label', 'psi', 'level', 'base', 'cur')
        self._psi_tree = ttk.Treeview(feat_frame, columns=fcols,
                                      show='headings', height=8)
        self._register_tree_widget(self._psi_tree)
        for col, text, width in (
                ('label', '特征', 110), ('psi', 'PSI', 70),
                ('level', '等级', 75), ('base', '基线均值', 75),
                ('cur', '当前均值', 75)):
            self._psi_tree.heading(col, text=text)
            self._psi_tree.column(col, width=width,
                                  anchor='w' if col == 'label' else 'center')
        for level, color_key in _LEVEL_COLORS_KEYS.items():
            self._psi_tree.tag_configure(
                f'lv_{level}',
                foreground=self.COLORS.get(color_key, '#999999'))
        fscroll = ttk.Scrollbar(feat_frame, orient='vertical',
                                command=self._psi_tree.yview)
        self._psi_tree.configure(yscrollcommand=fscroll.set)
        self._psi_tree.pack(side='left', fill='both', expand=True)
        fscroll.pack(side='right', fill='y')

        cat_frame = ttk.LabelFrame(left, text='布尔特征阳性率偏移', padding=4)
        cat_frame.pack(fill='both', expand=True, pady=(6, 0))
        ccols = ('label', 'base', 'cur', 'delta', 'level')
        self._cat_tree = ttk.Treeview(cat_frame, columns=ccols,
                                      show='headings', height=4)
        self._register_tree_widget(self._cat_tree)
        for col, text, width in (
                ('label', '特征', 110), ('base', '基线阳性率', 80),
                ('cur', '当前阳性率', 80), ('delta', '偏移', 70),
                ('level', '等级', 75)):
            self._cat_tree.heading(col, text=text)
            self._cat_tree.column(col, width=width,
                                  anchor='w' if col == 'label' else 'center')
        for level, color_key in _LEVEL_COLORS_KEYS.items():
            self._cat_tree.tag_configure(
                f'lv_{level}',
                foreground=self.COLORS.get(color_key, '#999999'))
        cscroll = ttk.Scrollbar(cat_frame, orient='vertical',
                                command=self._cat_tree.yview)
        self._cat_tree.configure(yscrollcommand=cscroll.set)
        self._cat_tree.pack(side='left', fill='both', expand=True)
        cscroll.pack(side='right', fill='y')

        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        chart_frame = ttk.LabelFrame(right, text='PSI 分布图', padding=4)
        chart_frame.pack(fill='both', expand=True)
        if MATPLOTLIB_AVAILABLE:
            self._psi_figure = Figure(figsize=(6, 5), dpi=100)
            self._psi_canvas = FigureCanvasTkAgg(self._psi_figure,
                                                 master=chart_frame)
            self._psi_canvas.get_tk_widget().pack(fill='both', expand=True)
        else:
            self._psi_figure = None
            self._psi_canvas = None
            ttk.Label(chart_frame, text='matplotlib 不可用，无法显示图表',
                      style='Tip.TLabel').pack(pady=20)

    # -------------------- 视图 2：性能趋势 --------------------

    def _build_trend_view(self, sub):
        view = ttk.Frame(sub)
        sub.add(view, text='性能趋势')

        top = ttk.Frame(view)
        top.pack(fill='x', padx=4, pady=(6, 2))
        ttk.Label(top, text='数据桶:').pack(side='left')
        self._trend_bucket_var = tk.StringVar(value='全部')
        self._trend_bucket_combo = ttk.Combobox(
            top, textvariable=self._trend_bucket_var, width=36,
            state='readonly', values=['全部'])
        self._trend_bucket_combo.pack(side='left', padx=6)
        self._trend_bucket_combo.bind('<<ComboboxSelected>>',
                                      lambda _e: self._trend_draw())
        self._trend_info_label = ttk.Label(top, text='', style='Tip.TLabel')
        self._trend_info_label.pack(side='left', padx=10)

        chart_frame = ttk.LabelFrame(view, text='AUROC 训练趋势', padding=4)
        chart_frame.pack(fill='both', expand=True, padx=4, pady=4)
        if MATPLOTLIB_AVAILABLE:
            self._trend_figure = Figure(figsize=(9, 5), dpi=100)
            self._trend_canvas = FigureCanvasTkAgg(self._trend_figure,
                                                   master=chart_frame)
            self._trend_canvas.get_tk_widget().pack(fill='both', expand=True)
        else:
            self._trend_figure = None
            self._trend_canvas = None
            ttk.Label(chart_frame, text='matplotlib 不可用，无法显示图表',
                      style='Tip.TLabel').pack(pady=20)

    # ==================== 数据刷新 ====================

    @staticmethod
    def _chart_placeholder(figure, canvas, text, colors):
        if figure is None:
            return
        figure.clear()
        ax = figure.add_subplot(111)
        ax.text(0.5, 0.5, text, ha='center', va='center',
                transform=ax.transAxes, fontsize=11,
                color=colors.get('text_light', '#5a6c7d'), wrap=True)
        ax.axis('off')
        canvas.draw_idle()

    def _drift_refresh(self):
        """刷新分布漂移报告与性能趋势（轻量计算，主线程执行）。"""
        archive_dir = _deploy.resolve_archive_dir()
        self._drift_dir_label.configure(
            text=f'数据源：{_alog.assessment_log_path()} ｜ 档案：{archive_dir}')

        # ---- 分布漂移 ----
        try:
            report = _alog.compute_drift_report()
        except Exception as e:  # noqa: BLE001
            LOGGER.debug('漂移报告计算失败: %s', e)
            report = {'available': False, 'reason': str(e)}
        self._drift_render(report)

        # ---- 性能趋势 ----
        try:
            self._trend_data = _deploy.collect_performance_trend(archive_dir)
        except Exception as e:  # noqa: BLE001
            LOGGER.debug('性能趋势读取失败: %s', e)
            self._trend_data = {'available': False, 'reason': str(e)}
        self._trend_refresh_bucket_combo()
        self._trend_draw()

    # ==================== 渲染：分布漂移 ====================

    def _drift_render(self, report):
        for item in self._psi_tree.get_children():
            self._psi_tree.delete(item)
        for item in self._cat_tree.get_children():
            self._cat_tree.delete(item)

        if not report.get('available'):
            n = report.get('n_records', 0)
            self._drift_cards['n_records'].configure(text=str(n))
            self._drift_cards['level'].configure(text='--')
            self._drift_cards['alerts'].configure(text='--')
            self._drift_cards['noise'].configure(text='--')
            self._chart_placeholder(
                self._psi_figure, self._psi_canvas,
                report.get('reason', '暂无评估日志数据'), self.COLORS)
            return

        level = report.get('overall_level', '稳定')
        alerts = report.get('alerts') or []
        self._drift_cards['n_records'].configure(
            text=str(report.get('n_records', 0)))
        self._drift_cards['level'].configure(
            text=level,
            foreground=self.COLORS.get(
                _LEVEL_COLORS_KEYS.get(level, 'disabled'), '#999999'))
        self._drift_cards['alerts'].configure(text=str(len(alerts)))
        self._drift_cards['noise'].configure(
            text=f"{report.get('noise_floor', 0):.3f}")

        base_win, cur_win = (report.get('baseline_window') or ('', ''),
                             report.get('current_window') or ('', ''))
        hint = (f"基线窗口 {report.get('n_baseline', 0)} 条"
                f"（{base_win[0]} ~ {base_win[1]}）｜"
                f"当前窗口 {report.get('n_current', 0)} 条"
                f"（{cur_win[0]} ~ {cur_win[1]}）｜"
                'PSI 判读：<0.10 稳定 / 0.10~0.25 轻度 / ≥0.25 显著')
        self._drift_hint_label.configure(text=hint)

        for f in report.get('features') or []:
            psi = f.get('psi')
            lv = f.get('level', '不可计算')
            self._psi_tree.insert('', 'end', tags=(f'lv_{lv}',), values=(
                f.get('label', f.get('name', '')),
                f'{psi:.3f}' if isinstance(psi, (int, float)) else '--',
                lv,
                f"{f['base_mean']:.1f}" if isinstance(
                    f.get('base_mean'), (int, float)) else '--',
                f"{f['cur_mean']:.1f}" if isinstance(
                    f.get('cur_mean'), (int, float)) else '--',
            ))
        for c in report.get('categorical') or []:
            lv = c.get('level', '稳定')
            self._cat_tree.insert('', 'end', tags=(f'lv_{lv}',), values=(
                c.get('label', c.get('name', '')),
                f"{c.get('base_rate', 0):.1%}",
                f"{c.get('cur_rate', 0):.1%}",
                f"{c.get('delta', 0):+.1%}",
                lv,
            ))

        self._drift_draw_psi_chart(report)

    def _drift_draw_psi_chart(self, report):
        if not self._psi_figure:
            return
        features = [f for f in (report.get('features') or [])
                    if isinstance(f.get('psi'), (int, float))]
        if not features:
            self._chart_placeholder(
                self._psi_figure, self._psi_canvas,
                '没有可计算的 PSI 特征', self.COLORS)
            return
        features.sort(key=lambda f: f.get('psi') or 0.0)

        fig = self._psi_figure
        fig.clear()
        colors = self.COLORS
        ax = fig.add_subplot(111)

        labels = [f.get('label', f.get('name', '')) for f in features]
        psis = [f.get('psi') or 0.0 for f in features]
        bar_colors = [
            colors.get(_LEVEL_COLORS_KEYS.get(f.get('level'), 'disabled'),
                       '#999999')
            for f in features
        ]
        y_pos = list(range(len(features)))
        ax.barh(y_pos, psis, color=bar_colors, height=0.6)
        for y, v in zip(y_pos, psis):
            ax.text(v + 0.005, y, f'{v:.3f}', va='center', fontsize=8)

        ax.axvline(_alog.PSI_MODERATE, ls='--', lw=1,
                   color=colors.get('risk_medium', '#f39c12'))
        ax.axvline(_alog.PSI_SIGNIFICANT, ls='--', lw=1,
                   color=colors.get('risk_high', '#e74c3c'))
        xmax = max(max(psis) * 1.15, _alog.PSI_SIGNIFICANT * 1.3, 0.1)
        ax.text(_alog.PSI_MODERATE, len(features) - 0.3, ' 0.10 关注',
                fontsize=8, color=colors.get('risk_medium', '#f39c12'))
        ax.text(_alog.PSI_SIGNIFICANT, len(features) - 0.3, ' 0.25 告警',
                fontsize=8, color=colors.get('risk_high', '#e74c3c'))

        noise = report.get('noise_floor')
        if isinstance(noise, (int, float)) and noise > 0:
            ax.axvspan(0, noise, alpha=0.12,
                       color=colors.get('text_light', '#5a6c7d'))
            ax.text(noise / 2, -0.45, f'噪声底 {noise:.3f}', fontsize=7,
                    ha='center', color=colors.get('text_light', '#5a6c7d'))

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(0, xmax)
        ax.set_xlabel('PSI')
        ax.set_title('特征与评分分布漂移（基线 vs 当前窗口）',
                     fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        fig.tight_layout()
        self._psi_canvas.draw_idle()

    # ==================== 渲染：性能趋势 ====================

    def _trend_refresh_bucket_combo(self):
        data = self._trend_data or {}
        buckets = sorted((data.get('series') or {}).keys()) \
            if data.get('available') else []
        self._trend_bucket_combo.configure(values=['全部'] + buckets)
        if self._trend_bucket_var.get() not in ['全部'] + buckets:
            self._trend_bucket_var.set('全部')

    def _trend_draw(self):
        if not self._trend_figure:
            return
        data = self._trend_data or {}
        if not data.get('available'):
            self._trend_info_label.configure(text='')
            self._chart_placeholder(
                self._trend_figure, self._trend_canvas,
                data.get('reason', '暂无训练档案数据'), self.COLORS)
            return

        series = data.get('series') or {}
        best = data.get('best') or {}
        bucket_sel = self._trend_bucket_var.get() or '全部'
        buckets = sorted(series.keys()) if bucket_sel == '全部' \
            else [b for b in [bucket_sel] if b in series]
        if not buckets:
            self._chart_placeholder(
                self._trend_figure, self._trend_canvas,
                '所选数据桶暂无记录', self.COLORS)
            return

        fig = self._trend_figure
        fig.clear()
        colors = self.COLORS
        ax = fig.add_subplot(111)
        palette = [colors.get('accent', '#0f766e'),
                   colors.get('chart_series_3', '#45B7D1'),
                   colors.get('chart_series_1', '#FF6B6B'),
                   colors.get('chart_series_4', '#FFA07A'),
                   colors.get('chart_series_2', '#4ECDC4'),
                   colors.get('chart_series_5', '#98D8C8')]

        info_parts = []
        for i, bucket in enumerate(buckets[:6]):
            points = series[bucket]
            xs = list(range(1, len(points) + 1))
            ys = [p['AUROC'] for p in points]
            color = palette[i % len(palette)]
            ax.plot(xs, ys, marker='o', ms=4, lw=1.5, color=color,
                    label=bucket if bucket_sel == '全部' else None)
            # 新高 Best 点加星标
            best_xs = [x for x, p in zip(xs, points) if p.get('is_best')]
            best_ys = [y for y, p in zip(ys, points) if p.get('is_best')]
            if best_xs:
                ax.scatter(best_xs, best_ys, marker='*', s=90,
                           color=color, zorder=5)
            # 单桶时叠加 best.json 参考线
            if bucket_sel != '全部' and bucket in best:
                ax.axhline(best[bucket], ls='--', lw=1,
                           color=colors.get('risk_medium', '#f39c12'))
                ax.text(xs[-1], best[bucket], f" 存档最佳 {best[bucket]:.3f}",
                        fontsize=8, va='bottom', ha='right',
                        color=colors.get('risk_medium', '#f39c12'))
            latest = ys[-1] if ys else None
            if isinstance(latest, (int, float)):
                info_parts.append(f'{bucket}: 最新 {latest:.4f}')

        ax.set_xlabel('训练轮次（按时间正序）')
        ax.set_ylabel('AUROC')
        ax.set_ylim(0.4, 1.0)
        ax.set_title('模型性能趋势（按 模型类型@数据集 分桶）',
                     fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        if bucket_sel == '全部' and len(buckets) > 1:
            ax.legend(fontsize=8, loc='lower right')
        fig.tight_layout()
        self._trend_canvas.draw_idle()

        self._trend_info_label.configure(
            text=' ｜ '.join(info_parts[:3]) +
            (' ...' if len(info_parts) > 3 else ''))
