#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""部署与监控面板 - 批量模型对比子标签

多个归档 ML 模型快照在同一确定性评估队列上对比 AUROC / AUPRC / Brier，
队列由 core.deploy_services.compare_models_on_cohort 生成（固定随机种子，
跨次可复现）。GNN .pt 快照需图构建上下文，不参与逐条打分对比。
"""

from ._shared import *  # noqa: F401,F403
# 星号导入不携带下划线名，显式补充
from ._shared import _deploy  # noqa: F401


class ModelCompareTabMixin:
    """批量模型/归档对比：快照多选 → 同队列评估 → 指标表 + 对比图"""

    # ==================== 界面构建 ====================

    def _build_model_compare_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text='批量模型对比')

        self._mc_snapshots = []
        self._mc_running = False

        paned = ttk.PanedWindow(tab, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=10, pady=10)

        # ── 左：快照选择 ──
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        snap_frame = ttk.LabelFrame(left, text='模型快照（可多选）', padding=6)
        snap_frame.pack(fill='both', expand=True)
        cols = ('label', 'bucket', 'auroc')
        self._mc_tree = ttk.Treeview(snap_frame, columns=cols, show='headings',
                                     height=12, selectmode='extended')
        self._register_tree_widget(self._mc_tree)
        self._mc_tree.heading('label', text='模型 / run_id')
        self._mc_tree.heading('bucket', text='数据桶')
        self._mc_tree.heading('auroc', text='存档AUROC')
        self._mc_tree.column('label', width=200, anchor='w')
        self._mc_tree.column('bucket', width=110, anchor='center')
        self._mc_tree.column('auroc', width=80, anchor='center')
        snap_scroll = ttk.Scrollbar(snap_frame, orient='vertical',
                                    command=self._mc_tree.yview)
        self._mc_tree.configure(yscrollcommand=snap_scroll.set)
        self._mc_tree.pack(side='left', fill='both', expand=True)
        snap_scroll.pack(side='right', fill='y')

        ttk.Button(left, text='刷新快照列表',
                   command=self._mc_refresh_snapshots).pack(
                       fill='x', pady=(6, 4))

        # 队列参数
        param_frame = ttk.LabelFrame(left, text='评估队列参数', padding=8)
        param_frame.pack(fill='x')
        self._mc_n_contacts = tk.StringVar(value='200')
        self._mc_n_cases = tk.StringVar(value='40')
        self._mc_seed = tk.StringVar(value='42')
        for row, (label, var) in enumerate([
                ('接触者数', self._mc_n_contacts),
                ('阳性病例数', self._mc_n_cases),
                ('随机种子', self._mc_seed)]):
            ttk.Label(param_frame, text=f'{label}:').grid(
                row=row, column=0, sticky='w', pady=2)
            ttk.Spinbox(param_frame, from_=1, to=100000, width=10,
                        textvariable=var).grid(row=row, column=1,
                                               sticky='w', padx=6, pady=2)
        ttk.Label(param_frame,
                  text='队列为确定性合成接触网络（固定种子可复现）',
                  style='Tip.TLabel', wraplength=220).grid(
                      row=3, column=0, columnspan=2, sticky='w', pady=(4, 0))

        self._mc_start_btn = ttk.Button(
            left, text='▶ 开始对比', style='Accent.TButton',
            command=self._mc_start_compare)
        self._mc_start_btn.pack(fill='x', pady=(8, 0))

        # ── 右：结果 ──
        right = ttk.Frame(paned)
        paned.add(right, weight=3)

        self._mc_status_label = ttk.Label(
            right, text='请选择至少两个模型快照后点击「开始对比」',
            style='Tip.TLabel')
        self._mc_status_label.pack(anchor='w', pady=(0, 4))

        result_frame = ttk.LabelFrame(right, text='对比指标', padding=6)
        result_frame.pack(fill='x')
        rcols = ('label', 'auroc', 'auprc', 'brier', 'mean', 'note')
        self._mc_result_tree = ttk.Treeview(result_frame, columns=rcols,
                                            show='headings', height=6)
        self._register_tree_widget(self._mc_result_tree)
        for col, text, width in (
                ('label', '模型', 220), ('auroc', 'AUROC', 80),
                ('auprc', 'AUPRC', 80), ('brier', 'Brier↓', 80),
                ('mean', '平均评分', 80), ('note', '备注', 140)):
            self._mc_result_tree.heading(col, text=text)
            self._mc_result_tree.column(
                col, width=width, anchor='w' if col in ('label', 'note')
                else 'center')
        self._mc_result_tree.tag_configure(
            'best', foreground=self.COLORS['accent'])
        result_scroll = ttk.Scrollbar(result_frame, orient='vertical',
                                      command=self._mc_result_tree.yview)
        self._mc_result_tree.configure(yscrollcommand=result_scroll.set)
        self._mc_result_tree.pack(side='left', fill='both', expand=True)
        result_scroll.pack(side='right', fill='y')

        chart_frame = ttk.LabelFrame(right, text='指标对比图', padding=6)
        chart_frame.pack(fill='both', expand=True, pady=(6, 0))
        if MATPLOTLIB_AVAILABLE:
            self._mc_figure = Figure(figsize=(8, 4), dpi=100)
            self._mc_canvas = FigureCanvasTkAgg(self._mc_figure,
                                                master=chart_frame)
            self._mc_canvas.get_tk_widget().pack(fill='both', expand=True)
            self._mc_placeholder('对比结果将在此显示')
        else:
            self._mc_figure = None
            self._mc_canvas = None
            ttk.Label(chart_frame, text='matplotlib 不可用，无法显示图表',
                      style='Tip.TLabel').pack(pady=20)

        # 初始化加载快照列表
        self._mc_refresh_snapshots()

    def _mc_placeholder(self, text):
        if not self._mc_figure:
            return
        fig = self._mc_figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, text, ha='center', va='center',
                transform=ax.transAxes, fontsize=11,
                color=self.COLORS.get('text_light', '#5a6c7d'))
        ax.axis('off')
        self._mc_canvas.draw_idle()

    # ==================== 快照列表 ====================

    def _mc_refresh_snapshots(self):
        """扫描训练档案目录，刷新可用 ML 快照列表。"""
        try:
            self._mc_snapshots = _deploy.list_model_snapshots()
        except Exception as e:  # noqa: BLE001
            LOGGER.debug('模型快照扫描失败: %s', e)
            self._mc_snapshots = []
        for item in self._mc_tree.get_children():
            self._mc_tree.delete(item)
        for idx, s in enumerate(self._mc_snapshots):
            auroc = s.get('auroc')
            self._mc_tree.insert('', 'end', iid=str(idx), values=(
                s.get('label', ''),
                s.get('bucket', '') or '--',
                f"{auroc:.4f}" if isinstance(auroc, (int, float)) else '--',
            ))
        # 默认全选，便于一键对比
        all_ids = [str(i) for i in range(len(self._mc_snapshots))]
        if all_ids:
            self._mc_tree.selection_set(all_ids)
        n = len(self._mc_snapshots)
        self._mc_status_label.configure(
            text=f'发现 {n} 个可用 ML 快照' +
            ('' if n else '（训练档案中暂无模型快照，请先训练模型）'))

    # ==================== 对比执行 ====================

    def _mc_start_compare(self):
        if self._mc_running:
            messagebox.showinfo('提示', '模型对比正在进行中，请稍候')
            return
        selected = self._mc_tree.selection()
        if len(selected) < 2:
            messagebox.showinfo('提示', '请至少选择两个模型快照进行对比')
            return
        specs = [self._mc_snapshots[int(iid)] for iid in selected
                 if int(iid) < len(self._mc_snapshots)]
        try:
            n_contacts = max(20, int(self._mc_n_contacts.get()))
            n_cases = max(1, int(self._mc_n_cases.get()))
            seed = int(self._mc_seed.get())
        except (ValueError, tk.TclError):
            messagebox.showwarning('参数错误', '队列参数必须为整数')
            return
        if n_cases >= n_contacts:
            messagebox.showwarning('参数错误', '阳性病例数必须小于接触者数')
            return

        self._mc_running = True
        self._mc_start_btn.configure(state='disabled')
        self._mc_status_label.configure(text='对比评估中...')
        self._start_training_thread(
            lambda: self._mc_worker(specs, n_contacts, n_cases, seed))

    def _mc_worker(self, specs, n_contacts, n_cases, seed):
        """后台线程：同队列对比评估，完成后回主线程渲染。"""
        try:
            report = _deploy.compare_models_on_cohort(
                specs, n_contacts=n_contacts, n_cases=n_cases,
                random_state=seed)
        except Exception as e:  # noqa: BLE001
            LOGGER.error('模型对比失败: %s', e, exc_info=True)
            report = {'available': False, 'reason': str(e), 'models': []}
        try:
            self.root.after(0, lambda r=report: self._mc_finish(r))
        except Exception:  # noqa: BLE001
            self._mc_running = False

    def _mc_finish(self, report):
        self._mc_running = False
        self._mc_start_btn.configure(state='normal')

        for item in self._mc_result_tree.get_children():
            self._mc_result_tree.delete(item)

        if not report.get('available'):
            self._mc_status_label.configure(
                text='对比失败：' + (report.get('reason') or '未知原因'))
            self._mc_placeholder('对比失败 — ' +
                                 (report.get('reason') or '未知原因'))
            return

        cohort = report.get('cohort') or {}
        models = report.get('models') or []
        self._mc_status_label.configure(
            text=f"队列：{cohort.get('n_contacts', 0)} 接触者 / "
                 f"{cohort.get('n_cases', 0)} 阳性（种子 {cohort.get('random_state', '?')}）"
                 f" — {sum(1 for m in models if m.get('available'))}/{len(models)} 个模型评估成功")

        # 表：按 AUROC 降序，最优行高亮
        ok_models = [m for m in models if m.get('available')]
        best_label = None
        scored = [(m.get('auroc'), m) for m in ok_models]
        scored = [p for p in scored if isinstance(p[0], (int, float))]
        if scored:
            best_label = max(scored, key=lambda p: p[0])[1].get('label')

        def _sort_key(m):
            auroc = m.get('auroc')
            return (auroc is None, -(auroc or 0.0))

        for m in sorted(models, key=_sort_key):
            if m.get('available'):
                def _fmt(v):
                    return f'{v:.4f}' if isinstance(v, (int, float)) else '--'
                tags = ('best',) if m.get('label') == best_label else ()
                self._mc_result_tree.insert('', 'end', tags=tags, values=(
                    m.get('label', ''),
                    _fmt(m.get('auroc')),
                    _fmt(m.get('auprc')),
                    _fmt(m.get('brier')),
                    f"{m['mean_score']:.1f}" if isinstance(
                        m.get('mean_score'), (int, float)) else '--',
                    '★ 本队列最优' if m.get('label') == best_label else '',
                ))
            else:
                self._mc_result_tree.insert('', 'end', values=(
                    m.get('label', ''), '--', '--', '--', '--',
                    m.get('reason') or '不可用',
                ))

        self._mc_draw_chart(models, cohort)

    # ==================== 图表 ====================

    def _mc_draw_chart(self, models, cohort):
        if not self._mc_figure:
            return
        ok = [m for m in models
              if m.get('available') and isinstance(m.get('auroc'), (int, float))]
        if not ok:
            self._mc_placeholder('没有可绘制的模型指标')
            return
        ok.sort(key=lambda m: m.get('auroc') or 0.0)

        fig = self._mc_figure
        fig.clear()
        colors = self.COLORS
        ax = fig.add_subplot(111)

        labels = [m.get('label', '')[:28] for m in ok]
        y_pos = list(range(len(ok)))
        aurocs = [m.get('auroc') or 0.0 for m in ok]
        auprcs = [m.get('auprc') for m in ok]
        auprcs = [v if isinstance(v, (int, float)) else 0.0 for v in auprcs]

        height = 0.38
        ax.barh([y + height / 2 for y in y_pos], aurocs, height=height,
                color=colors.get('accent', '#0f766e'), label='AUROC')
        ax.barh([y - height / 2 for y in y_pos], auprcs, height=height,
                color=colors.get('chart_series_3', '#45B7D1'), label='AUPRC')
        for y, v in zip([y + height / 2 for y in y_pos], aurocs):
            ax.text(v + 0.01, y, f'{v:.3f}', va='center', fontsize=8)
        ax.axvline(0.5, ls=':', lw=1,
                   color=colors.get('text_light', '#5a6c7d'))
        ax.text(0.5, len(ok) - 0.2, '随机基线 0.5', fontsize=8,
                color=colors.get('text_light', '#5a6c7d'), ha='center')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel('指标值')
        ax.set_title(
            f"同队列模型对比（{cohort.get('n_contacts', '?')} 接触者 / "
            f"{cohort.get('n_cases', '?')} 阳性 / 种子 {cohort.get('random_state', '?')}）",
            fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
        fig.tight_layout()
        self._mc_canvas.draw_idle()
