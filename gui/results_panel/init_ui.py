#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 结果面板 - 界面初始化 Mixin（_init_result_tab / _clear_form）"""

from ._shared import *


class InitUIMixin:
    """结果面板 UI 初始化方法"""

    def _get_chart_figsize(self, base_width=8, base_height=3, min_width=6, max_width=16):
        """改进8：根据窗口宽度动态计算图表 figsize

        参数：
            base_width/base_height: 1280px 窗口下的基准尺寸
            min_width/max_width: 宽度缩放范围
        返回：
            (width, height) 元组
        """
        try:
            if hasattr(self, 'root'):
                win_width = self.root.winfo_width()
                if win_width > 100:
                    ratio = win_width / 1280.0
                    w = max(min_width, min(max_width, base_width * ratio))
                    h = base_height * ratio
                    return (w, h)
        except Exception:
            pass
        return (base_width, base_height)

    def _init_result_tab(self, parent):
        """初始化风险评估结果标签页
        改进5：顶部仪表盘卡片 + Accent按钮 + COLORS化stale横幅 + PanedWindow
        """
        # 顶部按钮
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill='x', padx=15, pady=(12, 8))

        self._assess_button = ttk.Button(top_frame, text="▶  开始风险评估",
            command=self._run_assessment, style='Accent.TButton')
        self._assess_button.pack(side='left', padx=(0, 8))
        if MATPLOTLIB_AVAILABLE:
            ttk.Button(top_frame, text="导出所有图表",
                       command=self._export_all_charts).pack(side='left', padx=3)

        # ── 总体风险等级大横幅卡片（评估前显示引导文字，评估后显示大字号等级） ──
        self._risk_banner = tk.Frame(
            parent,
            bg=self.COLORS.get('bg_card', '#ffffff'),
            highlightthickness=1,
            highlightbackground=self.COLORS.get('border', '#dee2e6'),
            padx=16, pady=12,
        )
        self._risk_banner.pack(fill='x', padx=12, pady=(0, 6))
        # 左侧图标/徽标
        self._risk_banner_icon = tk.Label(
            self._risk_banner, text="◉",
            font=('Microsoft YaHei', 18),
            bg=self.COLORS.get('bg_card', '#ffffff'),
            fg=self.COLORS.get('accent', '#0f766e'),
        )
        self._risk_banner_icon.pack(side='left', padx=(0, 12))
        # 中部文字
        banner_text_frame = tk.Frame(self._risk_banner, bg=self.COLORS.get('bg_card', '#ffffff'))
        banner_text_frame.pack(side='left', fill='both', expand=True)
        self._risk_banner_title = tk.Label(
            banner_text_frame, text="尚未进行风险评估",
            font=('Microsoft YaHei', 13, 'bold'),
            bg=self.COLORS.get('bg_card', '#ffffff'),
            fg=self.COLORS.get('text', '#2c3e50'),
            anchor='w',
        )
        self._risk_banner_title.pack(fill='x', anchor='w')
        self._risk_banner_subtitle = tk.Label(
            banner_text_frame,
            text="请填写患者基本信息并添加至少一名接触者，然后点击上方「开始风险评估」按钮",
            font=('Microsoft YaHei', 9),
            bg=self.COLORS.get('bg_card', '#ffffff'),
            fg=self.COLORS.get('text_light', '#5a6c7d'),
            anchor='w', justify='left',
        )
        self._risk_banner_subtitle.pack(fill='x', anchor='w', pady=(2, 0))
        # 右侧百分比大字（评估后显示）
        self._risk_banner_pct = tk.Label(
            self._risk_banner, text="",
            font=('Microsoft YaHei', 18, 'bold'),
            bg=self.COLORS.get('bg_card', '#ffffff'),
            fg=self.COLORS.get('accent', '#0f766e'),
        )
        self._risk_banner_pct.pack(side='right')

        # 改进5：仪表盘概览卡片区域
        dashboard_frame = ttk.Frame(parent)
        dashboard_frame.pack(fill='x', padx=12, pady=(0, 4))
        for i in range(4):
            dashboard_frame.columnconfigure(i, weight=1, uniform='dash_col')

        # 高风险卡片
        self._dash_high_card = ttk.Frame(dashboard_frame, style='Card.TFrame', padding=8)
        self._dash_high_card.grid(row=0, column=0, sticky='nsew', padx=2)
        ttk.Label(self._dash_high_card, text="高风险",
                 font=self._FONT_SMALL, style='CardTitle.TLabel').pack(anchor='w')
        self._dash_high_value = ttk.Label(self._dash_high_card, text="--",
                                          style='CardValue.TLabel',
                                          foreground=self.COLORS['risk_high'])
        self._dash_high_value.pack(anchor='w')

        # 中风险卡片
        self._dash_med_card = ttk.Frame(dashboard_frame, style='Card.TFrame', padding=8)
        self._dash_med_card.grid(row=0, column=1, sticky='nsew', padx=2)
        ttk.Label(self._dash_med_card, text="中风险",
                 font=self._FONT_SMALL, style='CardTitle.TLabel').pack(anchor='w')
        self._dash_med_value = ttk.Label(self._dash_med_card, text="--",
                                        style='CardValue.TLabel',
                                        foreground=self.COLORS['risk_medium'])
        self._dash_med_value.pack(anchor='w')

        # 低风险卡片
        self._dash_low_card = ttk.Frame(dashboard_frame, style='Card.TFrame', padding=8)
        self._dash_low_card.grid(row=0, column=2, sticky='nsew', padx=2)
        ttk.Label(self._dash_low_card, text="低风险",
                 font=self._FONT_SMALL, style='CardTitle.TLabel').pack(anchor='w')
        self._dash_low_value = ttk.Label(self._dash_low_card, text="--",
                                        style='CardValue.TLabel',
                                        foreground=self.COLORS['risk_low'])
        self._dash_low_value.pack(anchor='w')

        # 建议措施卡片
        self._dash_action_card = ttk.Frame(dashboard_frame, style='Card.TFrame', padding=8)
        self._dash_action_card.grid(row=0, column=3, sticky='nsew', padx=2)
        ttk.Label(self._dash_action_card, text="建议措施",
                 font=self._FONT_SMALL, style='CardTitle.TLabel').pack(anchor='w')
        self._dash_action_value = ttk.Label(self._dash_action_card, text="请先评估",
                                           style='CardValue.TLabel',
                                           foreground=self.COLORS['accent'])
        self._dash_action_value.pack(anchor='w')

        # 改进5：stale 横幅 — 使用 COLORS 语义色（自适应高度，避免字体放大后裁剪）
        self._stale_banner_frame = tk.Frame(
            parent, bg=self.COLORS['warning_bg']
        )
        self._stale_banner_label = tk.Label(
            self._stale_banner_frame,
            text="⚠ 数据已修改，当前结果可能不准确，请重新执行风险评估",
            bg=self.COLORS['warning_bg'], fg=self.COLORS['warning_fg'],
            font=self._FONT_DEFAULT,
            padx=10, pady=5
        )
        self._stale_banner_label.pack(side='left', padx=10, pady=3)
        ttk.Button(self._stale_banner_frame, text="×",
                   width=3,
                   command=self._hide_stale_banner).pack(side='right', padx=5, pady=3)

        self.results_stale = False

        # 创建子标签页
        result_notebook = ttk.Notebook(parent)
        result_notebook.pack(fill='both', expand=True, padx=10, pady=5)

        # 1. 评估结果子标签页
        text_tab = ttk.Frame(result_notebook)
        result_notebook.add(text_tab, text="评估结果")
        # Section IX: 缓存 text_tab 引用，供 display.py 中的 _add_results_treeview 使用
        self._results_text_tab = text_tab

        text_frame = ttk.LabelFrame(text_tab, text="详细评估结果", padding=10)
        text_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # 改进5：PanedWindow 允许用户拖拽调整文本和图表高度
        result_paned = ttk.PanedWindow(text_frame, orient='vertical')
        result_paned.pack(fill='both', expand=True)

        # 上部分：结果文本
        text_subframe = ttk.Frame(result_paned)
        self.result_text = self._register_text_widget(tk.Text(text_subframe, wrap='word', height=10,
                                   font=self._FONT_DEFAULT))
        result_scrollbar = ttk.Scrollbar(text_subframe, orient='vertical',
                                         command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scrollbar.set)
        self.result_text.pack(side='left', fill='both', expand=True)
        result_scrollbar.pack(side='right', fill='y')
        result_paned.add(text_subframe, weight=1)

        # 下部分：风险排名
        rank_subframe = ttk.LabelFrame(result_paned, text="风险排名", padding=8)
        self._rank_tree = ttk.Treeview(rank_subframe,
                                       columns=('rank_name', 'rank_score', 'rank_level'),
                                       show='headings', height=6)
        self._register_tree_widget(self._rank_tree)
        self._rank_tree.heading('rank_name', text='姓名')
        self._rank_tree.heading('rank_score', text='风险评分')
        self._rank_tree.heading('rank_level', text='风险等级')
        self._rank_tree.column('rank_name', width=120)
        self._rank_tree.column('rank_score', width=80)
        self._rank_tree.column('rank_level', width=80)
        self._rank_tree.tag_configure('rank_high', foreground=self.COLORS['risk_high'])
        self._rank_tree.tag_configure('rank_medium', foreground=self.COLORS['risk_medium'])
        self._rank_tree.tag_configure('rank_low', foreground=self.COLORS['risk_low'])
        rank_scrollbar = ttk.Scrollbar(rank_subframe, orient='vertical',
                                       command=self._rank_tree.yview)
        self._rank_tree.configure(yscrollcommand=rank_scrollbar.set)
        self._rank_tree.pack(side='left', fill='both', expand=True)
        rank_scrollbar.pack(side='right', fill='y')
        result_paned.add(rank_subframe, weight=1)

        # 简单图表区域（Section VI: 迁移到 matplotlib，统一工具栏交互体验）
        simple_chart_frame = ttk.LabelFrame(text_tab, text="风险评分分布", padding=10)
        simple_chart_frame.pack(fill='both', expand=False, padx=10, pady=5)

        if MATPLOTLIB_AVAILABLE:
            # Section VI: matplotlib 版本（支持缩放/平移/保存 + 点击/悬停交互）
            self.risk_figure = Figure(figsize=self._get_chart_figsize(8, 3), dpi=100)
            self.risk_canvas = FigureCanvasTkAgg(self.risk_figure, master=simple_chart_frame)
            self.risk_canvas.get_tk_widget().pack(fill='both', expand=True)
            risk_toolbar = NavigationToolbar2Tk(self.risk_canvas, simple_chart_frame)
            risk_toolbar.update()
            # 保留 self.canvas 引用为 None（向后兼容检查）
            self.canvas = None
        else:
            # 回退：matplotlib 不可用时使用 tkinter Canvas
            self.canvas = tk.Canvas(simple_chart_frame, height=200,
                                     bg=self.COLORS.get('bg_card', '#ffffff'))
            self.canvas.pack(fill='x', pady=5)

        # Section VI: 初始化图表交互基础设施 + Esc 键清除选中
        self._init_chart_interaction()
        self._bind_esc_to_clear_selection(self.root)
        # Section VI: <Configure> 自动重绘绑定（窗口缩放时重绘风险柱状图）
        if MATPLOTLIB_AVAILABLE and hasattr(self, 'risk_canvas'):
            self._bind_configure_redraw(
                self.risk_canvas.get_tk_widget(), self._draw_risk_chart, 'risk_bar',
                figure=self.risk_figure, base_height=3
            )

        # 2. SEIR动力学曲线子标签页
        if MATPLOTLIB_AVAILABLE:
            seir_tab = ttk.Frame(result_notebook)
            result_notebook.add(seir_tab, text="SEIR动力学曲线")

            # 优先级八：可折叠的 SEIR 参数控制面板（beta/sigma/gamma 滑块）
            seir_param_frame = ttk.Frame(seir_tab)
            seir_param_frame.pack(fill='x', padx=10, pady=(10, 2))
            try:
                self._create_seir_param_panel(seir_param_frame)
            except Exception as e:
                LOGGER.debug("SEIR 参数面板创建失败（非致命）: %s", e)

            seir_chart_frame = ttk.Frame(seir_tab)
            seir_chart_frame.pack(fill='both', expand=True, padx=10, pady=10)

            self.seir_figure = Figure(figsize=self._get_chart_figsize(10, 6), dpi=100)
            self.seir_canvas = FigureCanvasTkAgg(self.seir_figure, master=seir_chart_frame)
            self.seir_canvas.get_tk_widget().pack(fill='both', expand=True)

            seir_toolbar = NavigationToolbar2Tk(self.seir_canvas, seir_chart_frame)
            seir_toolbar.update()
            # Section VI: <Configure> 自动重绘绑定
            self._bind_configure_redraw(
                self.seir_canvas.get_tk_widget(), self._draw_seir_curve, 'seir_curve',
                figure=self.seir_figure, base_height=6
            )

        # 3. 风险热力图子标签页
        if MATPLOTLIB_AVAILABLE and SEABORN_AVAILABLE:
            heatmap_tab = ttk.Frame(result_notebook)
            result_notebook.add(heatmap_tab, text="风险热力图")

            heatmap_chart_frame = ttk.Frame(heatmap_tab)
            heatmap_chart_frame.pack(fill='both', expand=True, padx=10, pady=10)

            self.heatmap_figure = Figure(figsize=self._get_chart_figsize(10, 8), dpi=100)
            self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_figure, master=heatmap_chart_frame)
            self.heatmap_canvas.get_tk_widget().pack(fill='both', expand=True)

            heatmap_toolbar = NavigationToolbar2Tk(self.heatmap_canvas, heatmap_chart_frame)
            heatmap_toolbar.update()
            # Section VI: <Configure> 自动重绘绑定
            self._bind_configure_redraw(
                self.heatmap_canvas.get_tk_widget(), self._draw_risk_heatmap, 'heatmap',
                figure=self.heatmap_figure, base_height=8
            )

        # 4. 概率分布直方图子标签页
        if MATPLOTLIB_AVAILABLE:
            histogram_tab = ttk.Frame(result_notebook)
            result_notebook.add(histogram_tab, text="概率分布")

            histogram_chart_frame = ttk.Frame(histogram_tab)
            histogram_chart_frame.pack(fill='both', expand=True, padx=10, pady=10)

            self.histogram_figure = Figure(figsize=self._get_chart_figsize(10, 6), dpi=100)
            self.histogram_canvas = FigureCanvasTkAgg(self.histogram_figure, master=histogram_chart_frame)
            self.histogram_canvas.get_tk_widget().pack(fill='both', expand=True)

            histogram_toolbar = NavigationToolbar2Tk(self.histogram_canvas, histogram_chart_frame)
            histogram_toolbar.update()
            # Section VI: <Configure> 自动重绘绑定
            self._bind_configure_redraw(
                self.histogram_canvas.get_tk_widget(),
                self._draw_probability_histogram, 'histogram',
                figure=self.histogram_figure, base_height=6
            )

        # 5. 延迟就诊影响曲线子标签页
        if MATPLOTLIB_AVAILABLE:
            delay_tab = ttk.Frame(result_notebook)
            result_notebook.add(delay_tab, text="延迟就诊影响")

            delay_chart_frame = ttk.Frame(delay_tab)
            delay_chart_frame.pack(fill='both', expand=True, padx=10, pady=10)

            self.delay_figure = Figure(figsize=(10, 6), dpi=100)
            self.delay_canvas = FigureCanvasTkAgg(self.delay_figure, master=delay_chart_frame)
            self.delay_canvas.get_tk_widget().pack(fill='both', expand=True)

            delay_toolbar = NavigationToolbar2Tk(self.delay_canvas, delay_chart_frame)
            delay_toolbar.update()
            # Section VI: <Configure> 自动重绘绑定
            self._bind_configure_redraw(
                self.delay_canvas.get_tk_widget(),
                self._draw_delay_impact_curve, 'delay',
                figure=self.delay_figure, base_height=6
            )

        # 6. 机器学习风险评估子标签页
        if SKLEARN_AVAILABLE and MATPLOTLIB_AVAILABLE:
            ml_tab = ttk.Frame(result_notebook)
            result_notebook.add(ml_tab, text="ML风险评估")

            # 警示标签
            warning_frame = ttk.LabelFrame(ml_tab, text="⚠️ 模型状态提示", padding=5)
            warning_frame.pack(fill='x', padx=5, pady=3)

            self.ml_warning_label = tk.Label(warning_frame,
                text="当前使用合成数据训练模型，预测结果仅供参考，不能替代临床诊断！\n"
                "如需实际应用，请使用本地流行病学数据重新训练模型",
                font=self._FONT_SMALL,
                fg=self.COLORS['danger_fg'], bg=self.COLORS['danger_bg'],
                justify='center', padx=10, pady=5)
            self.ml_warning_label.pack(fill='x')

            # 模型依赖可用性提示
            model_status_parts = []
            model_status_parts.append("✓ sklearn" if SKLEARN_AVAILABLE else "✗ sklearn未安装")
            if XGBOOST_AVAILABLE:
                model_status_parts.append("✓ XGBoost")
            else:
                model_status_parts.append("✗ XGBoost未安装(pip install xgboost)")
            if LIGHTGBM_AVAILABLE:
                model_status_parts.append("✓ LightGBM")
            else:
                model_status_parts.append("✗ LightGBM未安装(pip install lightgbm)")
            if CATBOOST_AVAILABLE:
                model_status_parts.append("✓ CatBoost")
            else:
                model_status_parts.append("✗ CatBoost未安装(pip install catboost)")
            self.ml_model_status_label = tk.Label(warning_frame,
                text="  ".join(model_status_parts),
                font=('Consolas', 8),
                fg='#6c757d', bg=self.COLORS['bg_card'],
                anchor='w', padx=10, pady=2)
            self.ml_model_status_label.pack(fill='x', pady=(2, 0))

            # ML操作按钮栏 — 两行 grid 布局 + LabelFrame 视觉分组（P0 修复溢出）
            self.ml_btn_frame = ttk.LabelFrame(ml_tab, text="ML 操作", padding=5)
            self.ml_btn_frame.pack(fill='x', padx=5, pady=3)

            # 第一行：训练相关按钮（6-8 个，均匀分布）
            row1 = ttk.Frame(self.ml_btn_frame)
            row1.grid(row=0, column=0, sticky='ew', pady=(0, 3))
            _col = 0
            ttk.Button(row1, text="重新训练", command=self._ml_retrain).grid(row=0, column=_col, sticky='ew', padx=2)
            _col += 1
            if JOBLIB_AVAILABLE:
                ttk.Button(row1, text="保存模型", command=self._ml_save_model).grid(row=0, column=_col, sticky='ew', padx=2)
                _col += 1
                ttk.Button(row1, text="加载模型", command=self._ml_load_model).grid(row=0, column=_col, sticky='ew', padx=2)
                _col += 1
            ttk.Button(row1, text="导入真实数据", command=self._ml_train_from_real_data).grid(row=0, column=_col, sticky='ew', padx=2)
            _col += 1
            ttk.Button(row1, text="真实vs合成对比", command=self._ml_compare_real_vs_synthetic).grid(row=0, column=_col, sticky='ew', padx=2)
            _col += 1
            ttk.Button(row1, text="自动训练", command=self._auto_train_ml_from_ui).grid(row=0, column=_col, sticky='ew', padx=2)
            _col += 1
            ttk.Button(row1, text="反事实分析", command=self._show_counterfactual_analysis).grid(row=0, column=_col, sticky='ew', padx=2)
            _col += 1
            ttk.Button(row1, text="三方向演示", command=self._demo_three_directions).grid(row=0, column=_col, sticky='ew', padx=2)
            _col += 1
            # 配置列权重使按钮均匀分布并随窗口缩放
            for i in range(_col):
                row1.grid_columnconfigure(i, weight=1)

            # 第二行：模型管理 + GNN（2-5 个，均匀分布）
            row2 = ttk.Frame(self.ml_btn_frame)
            row2.grid(row=1, column=0, sticky='ew', pady=(3, 0))
            _col2 = 0
            # Section VII: 模型仓库 + 训练历史
            ttk.Button(row2, text="模型仓库", command=self._show_model_repository).grid(row=0, column=_col2, sticky='ew', padx=2)
            _col2 += 1
            ttk.Button(row2, text="训练历史", command=self._show_training_history).grid(row=0, column=_col2, sticky='ew', padx=2)
            _col2 += 1
            if PYTORCH_AVAILABLE and PYG_AVAILABLE:
                ttk.Button(row2, text="训练GNN", command=self._train_gnn_gui).grid(row=0, column=_col2, sticky='ew', padx=2)
                _col2 += 1
                ttk.Button(row2, text="保存GNN", command=self._save_gnn_gui).grid(row=0, column=_col2, sticky='ew', padx=2)
                _col2 += 1
                ttk.Button(row2, text="加载GNN", command=self._load_gnn_gui).grid(row=0, column=_col2, sticky='ew', padx=2)
                _col2 += 1
                # Section VII: PIGNN/RL/Survival 模块未就绪，完全隐藏 UI 入口
                # （而非灰色禁用），避免用户困惑。模块就绪后在此处添加按钮。
            for i in range(_col2):
                row2.grid_columnconfigure(i, weight=1)

            # GNN训练日志区
            gnn_log_frame = ttk.LabelFrame(ml_tab, text="GNN训练日志", padding=5)
            gnn_log_frame.pack(fill='x', padx=5, pady=5)
            self.gnn_log_text = self._register_text_widget(tk.Text(gnn_log_frame, wrap='word', height=8,
                                   font=self._FONT_SMALL))
            gnn_scrollbar = ttk.Scrollbar(gnn_log_frame, orient='vertical', command=self.gnn_log_text.yview)
            self.gnn_log_text.configure(yscrollcommand=gnn_scrollbar.set)
            self.gnn_log_text.pack(side='left', fill='both', expand=True)
            gnn_scrollbar.pack(side='right', fill='y')

            # ML结果文本区
            ml_text_frame = ttk.LabelFrame(ml_tab, text="机器学习风险预测结果", padding=5)
            ml_text_frame.pack(fill='both', expand=True, padx=5, pady=5)

            self.ml_result_text = self._register_text_widget(tk.Text(ml_text_frame, wrap='word', height=10,
                                   font=self._FONT_SMALL))
            ml_scrollbar = ttk.Scrollbar(ml_text_frame, orient='vertical', command=self.ml_result_text.yview)
            self.ml_result_text.configure(yscrollcommand=ml_scrollbar.set)
            self.ml_result_text.pack(side='left', fill='both', expand=True)
            ml_scrollbar.pack(side='right', fill='y')

            # ML图表区
            ml_chart_notebook = ttk.Notebook(ml_tab)
            ml_chart_notebook.pack(fill='both', expand=True, padx=5, pady=5)

            # 对比图
            ml_compare_tab = ttk.Frame(ml_chart_notebook)
            ml_chart_notebook.add(ml_compare_tab, text="传统 vs ML对比")
            ml_compare_frame = ttk.Frame(ml_compare_tab)
            ml_compare_frame.pack(fill='both', expand=True, padx=5, pady=5)
            self.ml_compare_figure = Figure(figsize=self._get_chart_figsize(10, 5), dpi=100)
            self.ml_compare_canvas = FigureCanvasTkAgg(self.ml_compare_figure, master=ml_compare_frame)
            self.ml_compare_canvas.get_tk_widget().pack(fill='both', expand=True)
            ml_compare_toolbar = NavigationToolbar2Tk(self.ml_compare_canvas, ml_compare_frame)
            ml_compare_toolbar.update()
            # 改进9：<Configure> 自动重绘绑定（按容器宽高铺满，避免裁切/不居中）
            self._bind_configure_redraw(
                self.ml_compare_canvas.get_tk_widget(), self._draw_ml_comparison_chart,
                'ml_compare', figure=self.ml_compare_figure
            )

            # SHAP特征重要性图
            if SHAP_AVAILABLE:
                shap_tab = ttk.Frame(ml_chart_notebook)
                ml_chart_notebook.add(shap_tab, text="SHAP特征重要性")
                shap_frame = ttk.Frame(shap_tab)
                shap_frame.pack(fill='both', expand=True, padx=5, pady=5)
                self.shap_figure = Figure(figsize=self._get_chart_figsize(10, 6), dpi=100)
                self.shap_canvas = FigureCanvasTkAgg(self.shap_figure, master=shap_frame)
                self.shap_canvas.get_tk_widget().pack(fill='both', expand=True)
                shap_toolbar = NavigationToolbar2Tk(self.shap_canvas, shap_frame)
                shap_toolbar.update()
                self._bind_configure_redraw(
                    self.shap_canvas.get_tk_widget(), self._draw_shap_chart,
                    'shap', figure=self.shap_figure
                )

            # 模型性能图
            ml_perf_tab = ttk.Frame(ml_chart_notebook)
            ml_chart_notebook.add(ml_perf_tab, text="模型性能")
            ml_perf_frame = ttk.Frame(ml_perf_tab)
            ml_perf_frame.pack(fill='both', expand=True, padx=5, pady=5)
            self.ml_perf_figure = Figure(figsize=self._get_chart_figsize(10, 5), dpi=100)
            self.ml_perf_canvas = FigureCanvasTkAgg(self.ml_perf_figure, master=ml_perf_frame)
            self.ml_perf_canvas.get_tk_widget().pack(fill='both', expand=True)
            ml_perf_toolbar = NavigationToolbar2Tk(self.ml_perf_canvas, ml_perf_frame)
            ml_perf_toolbar.update()
            self._bind_configure_redraw(
                self.ml_perf_canvas.get_tk_widget(), self._draw_ml_performance_chart,
                'ml_perf', figure=self.ml_perf_figure
            )

            # 校准曲线图
            ml_calib_tab = ttk.Frame(ml_chart_notebook)
            ml_chart_notebook.add(ml_calib_tab, text="校准曲线")
            ml_calib_frame = ttk.Frame(ml_calib_tab)
            ml_calib_frame.pack(fill='both', expand=True, padx=5, pady=5)
            self.ml_calib_figure = Figure(figsize=self._get_chart_figsize(10, 6), dpi=100)
            self.ml_calib_canvas = FigureCanvasTkAgg(self.ml_calib_figure, master=ml_calib_frame)
            self.ml_calib_canvas.get_tk_widget().pack(fill='both', expand=True)
            ml_calib_toolbar = NavigationToolbar2Tk(self.ml_calib_canvas, ml_calib_frame)
            ml_calib_toolbar.update()
            self._bind_configure_redraw(
                self.ml_calib_canvas.get_tk_widget(), self._draw_calibration_chart,
                'ml_calib', figure=self.ml_calib_figure
            )

            # 个体SHAP瀑布图
            if SHAP_AVAILABLE:
                ml_shap_ind_tab = ttk.Frame(ml_chart_notebook)
                ml_chart_notebook.add(ml_shap_ind_tab, text="个体SHAP解释")
                ml_shap_ind_top = ttk.Frame(ml_shap_ind_tab)
                ml_shap_ind_top.pack(fill='x', padx=5, pady=3)
                ttk.Label(ml_shap_ind_top, text="选择接触者:").pack(side='left', padx=3)
                self.shap_ind_contact_var = tk.StringVar()
                self.shap_ind_contact_combo = ttk.Combobox(ml_shap_ind_top, textvariable=self.shap_ind_contact_var, width=20, state='readonly')
                self.shap_ind_contact_combo.pack(side='left', padx=3)
                ttk.Button(ml_shap_ind_top, text="生成瀑布图", command=self._draw_individual_shap).pack(side='left', padx=3)
                ml_shap_ind_frame = ttk.Frame(ml_shap_ind_tab)
                ml_shap_ind_frame.pack(fill='both', expand=True, padx=5, pady=5)
                self.shap_ind_figure = Figure(figsize=self._get_chart_figsize(10, 6), dpi=100)
                self.shap_ind_canvas = FigureCanvasTkAgg(self.shap_ind_figure, master=ml_shap_ind_frame)
                self.shap_ind_canvas.get_tk_widget().pack(fill='both', expand=True)
                shap_ind_toolbar = NavigationToolbar2Tk(self.shap_ind_canvas, ml_shap_ind_frame)
                shap_ind_toolbar.update()
                self._bind_configure_redraw(
                    self.shap_ind_canvas.get_tk_widget(), self._draw_individual_shap,
                    'shap_ind', figure=self.shap_ind_figure
                )

            # 风险归因条形图（SHAP因素 + GNN节点/边）
            ml_attr_tab = ttk.Frame(ml_chart_notebook)
            ml_chart_notebook.add(ml_attr_tab, text="风险归因")
            ml_attr_frame = ttk.Frame(ml_attr_tab)
            ml_attr_frame.pack(fill='both', expand=True, padx=5, pady=5)
            self.attribution_figure = Figure(figsize=self._get_chart_figsize(10, 6), dpi=100)
            self.attribution_canvas = FigureCanvasTkAgg(self.attribution_figure, master=ml_attr_frame)
            self.attribution_canvas.get_tk_widget().pack(fill='both', expand=True)
            ml_attr_toolbar = NavigationToolbar2Tk(self.attribution_canvas, ml_attr_frame)
            ml_attr_toolbar.update()
            self._bind_configure_redraw(
                self.attribution_canvas.get_tk_widget(), self._draw_attribution_bar_chart,
                'attribution', figure=self.attribution_figure
            )

            # 因果DAG可视化
            ml_dag_tab = ttk.Frame(ml_chart_notebook)
            ml_chart_notebook.add(ml_dag_tab, text="因果DAG")
            ml_dag_frame = ttk.Frame(ml_dag_tab)
            ml_dag_frame.pack(fill='both', expand=True, padx=5, pady=5)
            self.dag_figure = Figure(figsize=self._get_chart_figsize(12, 7), dpi=100)
            self.dag_canvas = FigureCanvasTkAgg(self.dag_figure, master=ml_dag_frame)
            self.dag_canvas.get_tk_widget().pack(fill='both', expand=True)
            dag_toolbar = NavigationToolbar2Tk(self.dag_canvas, ml_dag_frame)
            dag_toolbar.update()
            self._bind_configure_redraw(
                self.dag_canvas.get_tk_widget(), self._draw_causal_dag,
                'dag', figure=self.dag_figure
            )

            # 模型验证报告（判别/校准/重校准/交叉验证汇总）
            ml_val_tab = ttk.Frame(ml_chart_notebook)
            ml_chart_notebook.add(ml_val_tab, text="模型验证报告")
            ml_val_pane = ttk.PanedWindow(ml_val_tab, orient='vertical')
            ml_val_pane.pack(fill='both', expand=True, padx=5, pady=5)

            ml_val_text_wrap = ttk.Frame(ml_val_pane)
            ml_val_pane.add(ml_val_text_wrap, weight=1)
            self.ml_validation_text = tk.Text(
                ml_val_text_wrap, height=14, wrap='none',
                font=('Consolas', 9), background='#fafafa')
            ml_val_scroll = ttk.Scrollbar(ml_val_text_wrap, orient='vertical',
                                          command=self.ml_validation_text.yview)
            self.ml_validation_text.configure(
                yscrollcommand=ml_val_scroll.set)
            self.ml_validation_text.pack(side='left', fill='both', expand=True)
            ml_val_scroll.pack(side='right', fill='y')

            ml_val_fig_frame = ttk.Frame(ml_val_pane)
            ml_val_pane.add(ml_val_fig_frame, weight=2)
            self.validation_figure = Figure(
                figsize=self._get_chart_figsize(10, 6), dpi=100)
            self.validation_canvas = FigureCanvasTkAgg(
                self.validation_figure, master=ml_val_fig_frame)
            self.validation_canvas.get_tk_widget().pack(fill='both', expand=True)
            ml_val_toolbar = NavigationToolbar2Tk(
                self.validation_canvas, ml_val_fig_frame)
            ml_val_toolbar.update()
            self._bind_configure_redraw(
                self.validation_canvas.get_tk_widget(), self._update_validation_report,
                'validation', figure=self.validation_figure
            )

        # 7. 优先级九：MCMC 诊断图 + GNN 拓扑图子标签页
        if MATPLOTLIB_AVAILABLE:
            # MCMC 诊断图
            mcmc_diag_tab = ttk.Frame(result_notebook)
            result_notebook.add(mcmc_diag_tab, text="MCMC诊断")

            mcmc_diag_frame = ttk.Frame(mcmc_diag_tab)
            mcmc_diag_frame.pack(fill='both', expand=True, padx=10, pady=10)

            self.mcmc_diag_figure = Figure(figsize=self._get_chart_figsize(10, 8), dpi=100)
            self.mcmc_diag_canvas = FigureCanvasTkAgg(self.mcmc_diag_figure, master=mcmc_diag_frame)
            self.mcmc_diag_canvas.get_tk_widget().pack(fill='both', expand=True)

            mcmc_diag_toolbar = NavigationToolbar2Tk(self.mcmc_diag_canvas, mcmc_diag_frame)
            mcmc_diag_toolbar.update()
            self._bind_configure_redraw(
                self.mcmc_diag_canvas.get_tk_widget(), self._draw_mcmc_diagnostics,
                'mcmc_diag', figure=self.mcmc_diag_figure
            )

            # GNN 拓扑图
            gnn_topo_tab = ttk.Frame(result_notebook)
            result_notebook.add(gnn_topo_tab, text="接触网络拓扑")

            gnn_topo_frame = ttk.Frame(gnn_topo_tab)
            gnn_topo_frame.pack(fill='both', expand=True, padx=10, pady=10)

            self.gnn_topology_figure = Figure(figsize=self._get_chart_figsize(10, 8), dpi=100)
            self.gnn_topology_canvas = FigureCanvasTkAgg(self.gnn_topology_figure, master=gnn_topo_frame)
            self.gnn_topology_canvas.get_tk_widget().pack(fill='both', expand=True)

            gnn_topo_toolbar = NavigationToolbar2Tk(self.gnn_topology_canvas, gnn_topo_frame)
            gnn_topo_toolbar.update()
            self._bind_configure_redraw(
                self.gnn_topology_canvas.get_tk_widget(), self._draw_gnn_topology,
                'gnn_topo', figure=self.gnn_topology_figure
            )

            # 初始绘制占位提示（等待风险评估完成后显示实际数据）
            if hasattr(self, '_draw_mcmc_placeholder'):
                try:
                    self._draw_mcmc_placeholder("请先运行风险评估\n评估完成后将显示MCMC收敛诊断图")
                except Exception:
                    pass
            if hasattr(self, '_draw_gnn_placeholder'):
                try:
                    self._draw_gnn_placeholder("请先运行风险评估\n评估完成后将显示接触网络拓扑图")
                except Exception:
                    pass

        # 8. AI 助手子标签页（缺陷 #3 — UI 入口控件）
        # 由 AiTabMixin 提供，若 Mixin 未接入则安全跳过
        if hasattr(self, '_init_ai_tab'):
            try:
                self._init_ai_tab(result_notebook)
            except Exception as e:
                LOGGER.debug("AI 助手标签页初始化失败（非致命）: %s", e)

        # 9. 模型证据子标签页组（三层递进 / 任务A/B/C / 网络贡献 / 验证与证据 /
        #    数据与可复现性 / 社区干预反事实）——与模型三层递进架构升级对齐。
        # 由 ModelEvidenceMixin 提供，若 Mixin 未接入则安全跳过
        if hasattr(self, '_init_model_evidence_tabs'):
            try:
                self._init_model_evidence_tabs(result_notebook)
            except Exception as e:
                LOGGER.debug("模型证据标签页初始化失败（非致命）: %s", e)

    def _update_assess_button_state(self):
        """P6-20：根据必填字段是否填写完成，更新评估按钮状态"""
        if not hasattr(self, '_assess_button'):
            return
        try:
            # 检查是否有至少一个家庭成员或社会接触者
            has_family = bool(self.family_entries)
            has_social = bool(self.social_entries)
            # 检查患者基本信息必填字段
            has_age = bool(self.basic_info_vars.get('age', tk.StringVar(value='')).get())
            can_assess = has_age and (has_family or has_social)
            if can_assess:
                self._assess_button.configure(state='normal')
            else:
                self._assess_button.configure(state='disabled')
        except Exception:
            pass

    def _update_risk_banner(self, results):
        """根据评估结果更新顶部风险等级大横幅。"""
        if not hasattr(self, '_risk_banner'):
            return
        try:
            overall_risk = str(results.get('overall_risk', '未知')).strip()
            prob = results.get('base_infection_probability', 0) or 0
            suggestion = str(results.get('overall_suggestion', '')).strip()
        except Exception:
            return

        # 根据风险等级选择颜色
        if '高' in overall_risk or overall_risk in ('high', 'High', 'HIGH'):
            color = self.COLORS.get('risk_high', '#e74c3c')
            icon = '⚠'
            level_cn = '高风险'
        elif '中' in overall_risk or overall_risk in ('medium', 'Medium'):
            color = self.COLORS.get('risk_medium', '#f39c12')
            icon = '◉'
            level_cn = '中风险'
        elif '低' in overall_risk or overall_risk in ('low', 'Low'):
            color = self.COLORS.get('risk_low', '#2ecc71')
            icon = '✓'
            level_cn = '低风险'
        else:
            color = self.COLORS.get('accent', '#0f766e')
            icon = '◉'
            level_cn = overall_risk

        # 更新横幅背景色（浅色风险底色，主色文字）
        bg_map = {
            self.COLORS.get('risk_high', '#e74c3c'): self.COLORS.get('risk_high_bg', '#ffe0e0'),
            self.COLORS.get('risk_medium', '#f39c12'): self.COLORS.get('risk_medium_bg', '#fff3cd'),
            self.COLORS.get('risk_low', '#2ecc71'): self.COLORS.get('risk_low_bg', '#e0ffe0'),
        }
        banner_bg = bg_map.get(color, self.COLORS.get('bg_card', '#ffffff'))

        try:
            self._risk_banner.configure(bg=banner_bg, highlightbackground=color)
            self._risk_banner_icon.configure(text=icon, bg=banner_bg, fg=color)
            self._risk_banner_title.configure(
                text=f"综合风险等级：{level_cn}",
                bg=banner_bg, fg=color,
                font=('Microsoft YaHei', 13, 'bold'))
            self._risk_banner_subtitle.configure(
                text=suggestion or "评估已完成，请查看下方详细结果与图表",
                bg=banner_bg,
                fg=self.COLORS.get('text', '#2c3e50'))
            self._risk_banner_pct.configure(
                text=f"{prob:.1f}%",
                bg=banner_bg, fg=color,
                font=('Microsoft YaHei', 18, 'bold'))
        except Exception as e:
            LOGGER.debug("_update_risk_banner 更新横幅失败: %s", e)

    def _clear_form(self):
        """清空表单，重置为默认值"""
        if self.assessing.is_set():
            messagebox.showwarning("警告", "正在评估中，请稍后再清空")
            return
        if not messagebox.askyesno("确认", "确定要清空所有表单数据吗？"):
            return

        # 清空患者基本信息
        for key in self.basic_info_vars:
            var = self.basic_info_vars[key]
            if isinstance(var, tk.IntVar):
                var.set(30 if key == 'age' else 0)
            elif isinstance(var, tk.StringVar):
                if key in ['sputum_smear', 'has_cavity', 'active_tb', 'treatment']:
                    var.set("1" if key in ['sputum_smear', 'has_cavity', 'active_tb'] else "2")
                elif key in ['symptoms', 'family_living_conditions', 'ventilation']:
                    var.set("3")
                else:
                    var.set("")
            elif isinstance(var, tk.Text):
                var.delete('1.0', tk.END)
                var.insert('1.0', '咳嗽、低热、盗汗')

        # 清空场景选择
        self.scenario_var.set("custom")
        self.scenario_desc_var.set("自定义模式：手动输入所有参数")

        # 清空家庭成员
        self.family_entries.clear()
        self.contact_labels.clear()
        self._contact_id_counter = 0
        self._update_family_treeview()

        # 清空社会接触者
        self.social_entries.clear()
        self._update_social_treeview()

        # 清空结果显示
        self.result_text.delete(1.0, tk.END)

        # 重置数据
        self.patient_info = {}
        self.results = {}
        if hasattr(self, 'ml_results_lock'):
            with self.ml_results_lock:
                self.ml_results = {}
        else:
            self.ml_results = {}
        self.family_members = []
        self.social_contacts = []
        self.contact_count = 0
        self.family_count = 0
        self.current_scenario = None

        messagebox.showinfo("成功", "表单已清空")