#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 结果面板 - 模型证据 Mixin（三层递进 / 任务 A/B/C / 消融 / 验证 / DGP / SEIR 干预 / 时序家庭筛查）

模型已从"加权平均集成"升级为"三层递进架构 + 三任务分解"。本 Mixin 在
"风险评估结果"下新增 9 个子标签页，为模型侧新能力提供可视化承载：

  1. 三层递进   —— 逐接触者瀑布图（基线 → 网络增量 → 联合）+ 门控/决策分解
  2. 任务A筛查  —— 敏感度/特异度/PPV/NPV/校准误差 + 筛查决策阈值调节器
  3. 任务B进展  —— C-index/固定时点AUC + 决策曲线分析(DCA) + 随访生存曲线
  4. 任务C网络  —— 命中率/排序质量/繁殖数估计/接触者追踪效率
  5. 网络贡献   —— 有/无网络层消融对比柱状图（ΔAUROC / ΔC-index）
  6. 验证与证据 —— 公开数据集目录 + 各任务外部验证可覆盖性
  7. 数据与可复现性 —— DGP 参数/分布对照校验/复现性校验
  8. 社区干预   —— SEIR 反事实曲线（干预 vs 无干预 + 策略对比）
  9. 时序家庭筛查 —— 户内第一人 → 风险更新 → 重排的序贯动态演示
     （第 2 层网络增强层的部署实现：已筛查阳性 → 其余成员风险上调并
     自动重排；首筛查者回退个体基线。交互式逐步筛查 + 检出曲线）

设计要点：
  - 静态证据（任务/消融/公开数据/DGP/SEIR）在标签页初始化后于后台线程一次性
    计算并缓存，不阻塞启动；三层递进依赖 ML 预测结果，评估完成后单独刷新。
  - 时序家庭筛查以"签名"检测数据源变化（成员增删/确诊切换/ML 基线刷新），
    变化时自动重建序贯状态；「重置」按钮可随时强制重读最新成员。
  - 全部渲染防御性编程：数据未就绪时显示占位提示，绝不抛异常。
  - 文本/Treeview 控件注册到主题系统，深色主题自动换色。
"""

from ._shared import *

# 证据计算统一委托 core 层（纯 Python，可独立测试）
from ...core import model_evidence as _evidence
# 时序家庭先验评分服务（第 2 层部署实现，纯 numpy）
from ...scoring.temporal_household import HouseholdScreeningState


# ==============================================================================
# 纯函数工具（阈值指标 / Kaplan-Meier），便于单元测试
# ==============================================================================

def task_a_metrics_at_threshold(scores, labels, threshold):
    """在给定阈值下重算任务 A 的混淆矩阵与诊断指标。"""
    tp = sum(1 for p, o in zip(scores, labels) if p >= threshold and o == 1)
    fp = sum(1 for p, o in zip(scores, labels) if p >= threshold and o == 0)
    tn = sum(1 for p, o in zip(scores, labels) if p < threshold and o == 0)
    fn = sum(1 for p, o in zip(scores, labels) if p < threshold and o == 1)

    def _ratio(num, den):
        return num / den if den else 0.0

    sens = _ratio(tp, tp + fn)
    spec = _ratio(tn, tn + fp)
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'sensitivity': sens,
        'specificity': spec,
        'ppv': _ratio(tp, tp + fp),
        'npv': _ratio(tn, tn + fn),
        'youden': sens + spec - 1.0,
    }


def kaplan_meier(times, events):
    """Kaplan-Meier 生存（无事件）曲线。

    参数：
        times (list[float]): 随访时间
        events (list[int]): 事件指示（1 = 事件发生，0 = 截尾）

    返回：
        (t_points, s_points): 阶梯曲线坐标（含起点 (0, 1)）
    """
    n = len(times)
    if n == 0:
        return [0.0], [1.0]
    order = sorted(range(n), key=lambda i: (times[i], -events[i]))
    at_risk = n
    surv = 1.0
    t_points = [0.0]
    s_points = [1.0]
    i = 0
    while i < n:
        t = times[order[i]]
        j = i
        d = 0
        c = 0
        while j < n and times[order[j]] == t:
            if events[order[j]]:
                d += 1
            else:
                c += 1
            j += 1
        if d > 0 and at_risk > 0:
            surv *= (1.0 - d / at_risk)
        t_points.append(t)
        s_points.append(surv)
        at_risk -= (d + c)
        i = j
    return t_points, s_points


def build_screening_state(demo):
    """从演示输入构造序贯筛查状态（纯函数，可独立测试）。

    已确诊成员按基线风险降序预置为"已筛查阳性"（部署口径：他们的
    结果先于本面板演示发生），其余成员风险立即上调并重排。

    参数：
        demo (dict): ``build_household_screening_demo`` 的返回值。

    返回：
        HouseholdScreeningState | None: demo 不可用时返回 None。
    """
    if not isinstance(demo, dict) or not demo.get('available'):
        return None
    state = HouseholdScreeningState(
        demo['base_scores'], demo['names'], demo.get('base_rate', 0.13))
    for idx in sorted(demo.get('preseed_positive', []),
                      key=lambda i: -demo['base_scores'][i]):
        state.record_result(idx, 1)
    return state


class ModelEvidenceMixin:
    """模型证据面板（13 个子标签页）"""

    # ==================== 初始化 ====================

    def _init_model_evidence_tabs(self, result_notebook):
        """在结果 Notebook 中创建 13 个模型证据子标签页并启动后台计算。"""
        if not hasattr(self, '_me_cache'):
            self._me_cache = {}

        self._build_three_layer_tab(result_notebook)
        self._build_task_a_tab(result_notebook)
        self._build_task_b_tab(result_notebook)
        self._build_task_c_tab(result_notebook)
        self._build_ablation_tab(result_notebook)
        self._build_public_data_tab(result_notebook)
        self._build_dgp_tab(result_notebook)
        self._build_seir_intervention_tab(result_notebook)
        self._build_temporal_household_tab(result_notebook)
        self._build_scale_validation_tab(result_notebook)
        self._build_negative_findings_tab(result_notebook)
        self._build_granularity_guide_tab(result_notebook)
        self._build_deployment_metrics_tab(result_notebook)

        # 静态证据后台计算（不依赖评估结果，启动后即可运行）
        if not self._me_cache.get('_static_started'):
            self._me_cache['_static_started'] = True
            try:
                self._start_training_thread(self._compute_static_evidence_background)
            except Exception as e:
                LOGGER.debug("模型证据后台计算启动失败: %s", e)

    # ---------- Tab 1: 三层递进 ----------

    def _build_three_layer_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="三层递进")

        # 顶部：接触者选择 + 信息行
        top = ttk.Frame(tab)
        top.pack(fill='x', padx=10, pady=(8, 4))
        ttk.Label(top, text="接触者：").pack(side='left')
        self.me_contact_var = tk.StringVar()
        self.me_contact_combo = ttk.Combobox(
            top, textvariable=self.me_contact_var, state='readonly', width=28)
        self.me_contact_combo.pack(side='left', padx=5)
        self.me_contact_combo.bind('<<ComboboxSelected>>',
                                   lambda e: self._draw_three_layer_waterfall())
        self.me_three_layer_info = ttk.Label(top, text="请先运行风险评估（含 ML 预测）",
                                             foreground='gray')
        self.me_three_layer_info.pack(side='left', padx=15)

        if MATPLOTLIB_AVAILABLE:
            chart_frame = ttk.Frame(tab)
            chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
            self.me_three_layer_figure = Figure(
                figsize=self._get_chart_figsize(11, 6), dpi=100)
            self.me_three_layer_canvas = FigureCanvasTkAgg(
                self.me_three_layer_figure, master=chart_frame)
            self.me_three_layer_canvas.get_tk_widget().pack(fill='both', expand=True)
            toolbar = NavigationToolbar2Tk(self.me_three_layer_canvas, chart_frame)
            toolbar.update()
            self._bind_configure_redraw(
                self.me_three_layer_canvas.get_tk_widget(),
                self._draw_three_layer_waterfall, 'me_three_layer',
                figure=self.me_three_layer_figure)
            self._me_placeholder(self.me_three_layer_figure,
                                 self.me_three_layer_canvas,
                                 "请先运行风险评估\nML 预测完成后显示三层递进分解")
        else:
            self.me_three_layer_text = self._me_make_text(tab)

    # ---------- Tab 2: 任务 A ----------

    def _build_task_a_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="任务A筛查")

        header = ttk.LabelFrame(tab, text="任务 A：横断面活动性结核筛查（谁现在做病原学/影像学检查）")
        header.pack(fill='x', padx=10, pady=(8, 4))
        self.me_task_a_labels = {}
        grid = ttk.Frame(header)
        grid.pack(fill='x', padx=8, pady=4)
        for i, key in enumerate(('sensitivity', 'specificity', 'ppv', 'npv',
                                 'youden', 'ece', 'threshold', 'n')):
            lbl = ttk.Label(grid, text=f"{key}: --", font=('Microsoft YaHei', 9))
            lbl.grid(row=i // 4, column=i % 4, sticky='w', padx=8, pady=2)
            self.me_task_a_labels[key] = lbl
        # round-10 P2：SINAN 双头与任务分解的映射（A'=找传染源 / B=死亡预后）
        self.me_task_a_dual_head_note = ttk.Label(
            header, text='', wraplength=880, justify='left',
            foreground='#555555')
        self.me_task_a_dual_head_note.pack(fill='x', padx=8, pady=(0, 4))

        # 阈值调节器
        slider_frame = ttk.LabelFrame(tab, text="筛查决策阈值调节器")
        slider_frame.pack(fill='x', padx=10, pady=4)
        self.me_task_a_threshold_var = tk.DoubleVar(value=15.0)
        self.me_task_a_scale = ttk.Scale(
            slider_frame, from_=1.0, to=99.0, orient='horizontal',
            variable=self.me_task_a_threshold_var,
            command=self._on_task_a_threshold_changed)
        self.me_task_a_scale.pack(fill='x', padx=10, pady=4)
        self.me_task_a_threshold_label = ttk.Label(slider_frame, text="当前阈值: 15.0%")
        self.me_task_a_threshold_label.pack(anchor='w', padx=10, pady=(0, 4))

        if MATPLOTLIB_AVAILABLE:
            chart_frame = ttk.Frame(tab)
            chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
            self.me_task_a_figure = Figure(
                figsize=self._get_chart_figsize(11, 5), dpi=100)
            self.me_task_a_canvas = FigureCanvasTkAgg(
                self.me_task_a_figure, master=chart_frame)
            self.me_task_a_canvas.get_tk_widget().pack(fill='both', expand=True)
            toolbar = NavigationToolbar2Tk(self.me_task_a_canvas, chart_frame)
            toolbar.update()
            self._bind_configure_redraw(
                self.me_task_a_canvas.get_tk_widget(),
                self._draw_task_a_chart, 'me_task_a',
                figure=self.me_task_a_figure)
            self._me_placeholder(self.me_task_a_figure, self.me_task_a_canvas,
                                 "正在计算任务 A 评估指标...")

    # ---------- Tab 3: 任务 B ----------

    def _build_task_b_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="任务B进展")

        header = ttk.LabelFrame(tab, text="任务 B：感染后进展预测（谁接受预防性治疗 TPT）")
        header.pack(fill='x', padx=10, pady=(8, 4))
        self.me_task_b_labels = {}
        grid = ttk.Frame(header)
        grid.pack(fill='x', padx=8, pady=4)
        for i, key in enumerate(('c_index', 'horizon_auc', 'ece', 'n_events')):
            lbl = ttk.Label(grid, text=f"{key}: --", font=('Microsoft YaHei', 9))
            lbl.grid(row=0, column=i, sticky='w', padx=10, pady=2)
            self.me_task_b_labels[key] = lbl
        self.me_task_b_note = ttk.Label(header, text="", foreground='gray',
                                        wraplength=900, justify='left')
        self.me_task_b_note.pack(fill='x', padx=8, pady=(0, 4))

        if MATPLOTLIB_AVAILABLE:
            chart_frame = ttk.Frame(tab)
            chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
            self.me_task_b_figure = Figure(
                figsize=self._get_chart_figsize(11, 5), dpi=100)
            self.me_task_b_canvas = FigureCanvasTkAgg(
                self.me_task_b_figure, master=chart_frame)
            self.me_task_b_canvas.get_tk_widget().pack(fill='both', expand=True)
            toolbar = NavigationToolbar2Tk(self.me_task_b_canvas, chart_frame)
            toolbar.update()
            self._bind_configure_redraw(
                self.me_task_b_canvas.get_tk_widget(),
                self._draw_task_b_charts, 'me_task_b',
                figure=self.me_task_b_figure)
            self._me_placeholder(self.me_task_b_figure, self.me_task_b_canvas,
                                 "正在计算任务 B 评估指标...")

    # ---------- Tab 4: 任务 C ----------

    def _build_task_c_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="任务C网络")

        header = ttk.LabelFrame(tab, text="任务 C：接触网络传播预测（追踪谁、在哪里干预）")
        header.pack(fill='x', padx=10, pady=(8, 4))
        self.me_task_c_labels = {}
        grid = ttk.Frame(header)
        grid.pack(fill='x', padx=8, pady=4)
        for i, key in enumerate(('r_estimate', 'tracing', 'mean_pos', 'n_contacts')):
            lbl = ttk.Label(grid, text=f"{key}: --", font=('Microsoft YaHei', 9))
            lbl.grid(row=0, column=i, sticky='w', padx=10, pady=2)
            self.me_task_c_labels[key] = lbl
        hint = ttk.Label(header, foreground='gray',
                         text="网络拓扑可视化见「接触网络拓扑」子标签页（与本任务联动）")
        hint.pack(anchor='w', padx=8, pady=(0, 4))

        if MATPLOTLIB_AVAILABLE:
            chart_frame = ttk.Frame(tab)
            chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
            self.me_task_c_figure = Figure(
                figsize=self._get_chart_figsize(11, 5), dpi=100)
            self.me_task_c_canvas = FigureCanvasTkAgg(
                self.me_task_c_figure, master=chart_frame)
            self.me_task_c_canvas.get_tk_widget().pack(fill='both', expand=True)
            toolbar = NavigationToolbar2Tk(self.me_task_c_canvas, chart_frame)
            toolbar.update()
            self._bind_configure_redraw(
                self.me_task_c_canvas.get_tk_widget(),
                self._draw_task_c_chart, 'me_task_c',
                figure=self.me_task_c_figure)
            self._me_placeholder(self.me_task_c_figure, self.me_task_c_canvas,
                                 "正在计算任务 C 评估指标...")

    # ---------- Tab 5: 网络贡献 ----------

    def _build_ablation_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="网络贡献")

        header = ttk.LabelFrame(tab, text="网络层贡献消融：传播网络到底贡献多少？")
        header.pack(fill='x', padx=10, pady=(8, 4))
        self.me_ablation_conclusion = ttk.Label(header, text="正在计算消融实验...",
                                                wraplength=900, justify='left')
        self.me_ablation_conclusion.pack(fill='x', padx=8, pady=4)

        if MATPLOTLIB_AVAILABLE:
            chart_frame = ttk.Frame(tab)
            chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
            self.me_ablation_figure = Figure(
                figsize=self._get_chart_figsize(10, 5), dpi=100)
            self.me_ablation_canvas = FigureCanvasTkAgg(
                self.me_ablation_figure, master=chart_frame)
            self.me_ablation_canvas.get_tk_widget().pack(fill='both', expand=True)
            toolbar = NavigationToolbar2Tk(self.me_ablation_canvas, chart_frame)
            toolbar.update()
            self._bind_configure_redraw(
                self.me_ablation_canvas.get_tk_widget(),
                self._draw_ablation_chart, 'me_ablation',
                figure=self.me_ablation_figure)
            self._me_placeholder(self.me_ablation_figure, self.me_ablation_canvas,
                                 "正在运行网络贡献消融实验...")

    # ---------- Tab 6: 验证与证据 ----------

    def _build_public_data_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="验证与证据")

        top = ttk.LabelFrame(tab, text="公开数据集外部验证（证据强度透明可见）")
        top.pack(fill='x', padx=10, pady=(8, 4))
        self.me_public_summary = ttk.Label(top, text="正在加载公开数据集目录...",
                                           wraplength=900, justify='left')
        self.me_public_summary.pack(fill='x', padx=8, pady=4)

        cols = ('id', 'name', 'layer', 'tasks', 'access', 'size', 'priority')
        tree_frame = ttk.Frame(tab)
        tree_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.me_public_tree = ttk.Treeview(tree_frame, columns=cols,
                                           show='headings', height=8)
        for col, text, width in (
                ('id', '数据集', 130), ('name', '名称', 220),
                ('layer', '层级', 80), ('tasks', '适用任务', 80),
                ('access', '获取方式', 100), ('size', '规模', 120),
                ('priority', '优先级', 60)):
            self.me_public_tree.heading(col, text=text)
            self.me_public_tree.column(col, width=width, anchor='w')
        sb = ttk.Scrollbar(tree_frame, orient='vertical',
                           command=self.me_public_tree.yview)
        self.me_public_tree.configure(yscrollcommand=sb.set)
        self.me_public_tree.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        self._register_tree_widget(self.me_public_tree)

        self.me_public_text = self._me_make_text(tab, height=8)

    # ---------- Tab 7: 数据与可复现性 ----------

    def _build_dgp_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="数据与可复现性")

        top = ttk.LabelFrame(tab, text="DGP 合成数据：怎么来的、靠不靠谱")
        top.pack(fill='x', padx=10, pady=(8, 4))
        self.me_dgp_summary = ttk.Label(top, text="正在生成 DGP 可复现报告...",
                                        wraplength=900, justify='left')
        self.me_dgp_summary.pack(fill='x', padx=8, pady=4)

        paned = ttk.PanedWindow(tab, orient='vertical')
        paned.pack(fill='both', expand=True, padx=10, pady=5)

        # 上：字段分布对照校验表
        tree_frame = ttk.Frame(paned)
        cols = ('field', 'kind', 'distribution', 'delta', 'tolerance', 'pass')
        self.me_dgp_tree = ttk.Treeview(tree_frame, columns=cols,
                                        show='headings', height=9)
        for col, text, width in (
                ('field', '字段', 160), ('kind', '类型', 80),
                ('distribution', '分布', 140), ('delta', '最大偏差', 90),
                ('tolerance', '容差', 80), ('pass', '校验', 60)):
            self.me_dgp_tree.heading(col, text=text)
            self.me_dgp_tree.column(col, width=width, anchor='w')
        sb = ttk.Scrollbar(tree_frame, orient='vertical',
                           command=self.me_dgp_tree.yview)
        self.me_dgp_tree.configure(yscrollcommand=sb.set)
        self.me_dgp_tree.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        self._register_tree_widget(self.me_dgp_tree)
        paned.add(tree_frame, weight=3)

        # 下：文献对照与详情文本
        text_frame = ttk.Frame(paned)
        self.me_dgp_text = self._me_make_text(text_frame, height=8)
        paned.add(text_frame, weight=2)

    # ---------- Tab 8: 社区干预反事实 ----------

    def _build_seir_intervention_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="社区干预")

        top = ttk.LabelFrame(tab, text="社区级 SEIR 反事实模拟（干预 vs 无干预）")
        top.pack(fill='x', padx=10, pady=(8, 4))
        self.me_seir_summary = ttk.Label(top, text="正在运行 SEIR 干预反事实模拟...",
                                         wraplength=900, justify='left')
        self.me_seir_summary.pack(fill='x', padx=8, pady=4)

        if MATPLOTLIB_AVAILABLE:
            chart_frame = ttk.Frame(tab)
            chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
            self.me_seir_figure = Figure(
                figsize=self._get_chart_figsize(11, 5), dpi=100)
            self.me_seir_canvas = FigureCanvasTkAgg(
                self.me_seir_figure, master=chart_frame)
            self.me_seir_canvas.get_tk_widget().pack(fill='both', expand=True)
            toolbar = NavigationToolbar2Tk(self.me_seir_canvas, chart_frame)
            toolbar.update()
            self._bind_configure_redraw(
                self.me_seir_canvas.get_tk_widget(),
                self._draw_seir_intervention, 'me_seir',
                figure=self.me_seir_figure)
            self._me_placeholder(self.me_seir_figure, self.me_seir_canvas,
                                 "正在运行 SEIR 干预反事实模拟...")

    # ==================== 后台计算 ====================

    # ---------- Tab 10: 百万级验证（SINAN 证据链头条） ----------

    def _build_scale_validation_tab(self, nb):
        """百万级国家级监测数据验证：头条叙事 + 时间衰减稳健性曲线。"""
        tab = ttk.Frame(nb)
        nb.add(tab, text="百万级验证")

        top = ttk.LabelFrame(tab, text="百万级国家级监测数据验证（巴西 SINAN 2001-2019）")
        top.pack(fill='x', padx=10, pady=(8, 4))
        self.me_scale_summary = ttk.Label(
            top, text="正在加载 SINAN 规模训练归档...",
            wraplength=900, justify='left')
        self.me_scale_summary.pack(fill='x', padx=8, pady=4)

        if MATPLOTLIB_AVAILABLE:
            chart_frame = ttk.Frame(tab)
            chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
            self.me_scale_figure = Figure(
                figsize=self._get_chart_figsize(11, 5), dpi=100)
            self.me_scale_canvas = FigureCanvasTkAgg(
                self.me_scale_figure, master=chart_frame)
            self.me_scale_canvas.get_tk_widget().pack(fill='both', expand=True)
            toolbar = NavigationToolbar2Tk(self.me_scale_canvas, chart_frame)
            toolbar.update()
            self._bind_configure_redraw(
                self.me_scale_canvas.get_tk_widget(),
                self._draw_scale_robustness, 'me_scale',
                figure=self.me_scale_figure)
            self._me_placeholder(self.me_scale_figure, self.me_scale_canvas,
                                 "正在加载 SINAN 规模训练归档...")
        else:
            self.me_scale_text = self._me_make_text(tab, height=8)

        self.me_scale_detail = self._me_make_text(tab, height=7)

    # ---------- Tab 11: 阴性发现与边界条件 ----------

    def _build_negative_findings_tab(self, nb):
        """阴性发现与边界条件：与阳性结果同样归档的诚实边界。"""
        tab = ttk.Frame(nb)
        nb.add(tab, text="阴性发现")

        top = ttk.LabelFrame(tab, text="阴性发现与边界条件（Negative Findings）")
        top.pack(fill='x', padx=10, pady=(8, 4))
        self.me_negative_summary = ttk.Label(
            top, text="正在加载阴性发现归档...",
            wraplength=900, justify='left')
        self.me_negative_summary.pack(fill='x', padx=8, pady=4)

        self.me_negative_detail = self._me_make_text(tab, height=22)

    def _update_negative_findings_ui(self):
        neg = self._me_cache.get('negative') or {}
        if not neg.get('available'):
            self.me_negative_summary.configure(
                text=f"阴性发现不可用：{neg.get('reason', '未知原因')}")
            return
        self.me_negative_summary.configure(
            text=neg.get('narrative', '') + '  ' + neg.get('tradition_note', ''))

        text = self.me_negative_detail
        text.delete('1.0', 'end')
        for f in neg.get('findings', []):
            text.insert('end', f"[{f.get('verdict', '?')}] {f.get('title', '')}\n")
            text.insert('end', f"  {f.get('detail', '')}\n")
            text.insert('end', f"  部署含义：{f.get('implication', '')}\n")
            text.insert('end', f"  归档：{f.get('archive', '')}\n\n")

    # ---------- Tab 12: 组粒度部署指南 ----------

    def _build_granularity_guide_tab(self, nb):
        """组粒度部署指南：ICC 决定时序先证价值上限，回答「按哪一级分组」。"""
        tab = ttk.Frame(nb)
        nb.add(tab, text="组粒度指南")

        top = ttk.LabelFrame(
            tab, text="组粒度部署指南：部署前测 ICC，再决定按哪一级分组")
        top.pack(fill='x', padx=10, pady=(8, 4))
        self.me_granularity_summary = ttk.Label(
            top, text="正在加载组粒度曲线归档...",
            wraplength=900, justify='left')
        self.me_granularity_summary.pack(fill='x', padx=8, pady=4)

        self.me_granularity_detail = self._me_make_text(tab, height=18)

    def _update_granularity_guide_ui(self):
        g = self._me_cache.get('granularity') or {}
        if not g.get('available'):
            self.me_granularity_summary.configure(
                text=f"组粒度部署指南不可用：{g.get('reason', '未知原因')}")
            return
        self.me_granularity_summary.configure(
            text=g.get('narrative', ''))

        text = self.me_granularity_detail
        text.delete('1.0', 'end')
        rule = g.get('decision_rule', {})
        text.insert('end', "部署前判据（可预先测量）——\n")
        text.insert('end', f"  第 1 步：{rule.get('measure', '')}\n")
        text.insert('end', f"  值得做：{rule.get('worthwhile', '')}\n")
        text.insert('end', f"  不值得：{rule.get('not_worthwhile', '')}\n")
        text.insert('end', f"  分组原则：{rule.get('grouping', '')}\n")
        text.insert('end', "\n实例点（ICC → ΔAUROC）——\n")
        for p in g.get('examples', []):
            text.insert(
                'end', f"  {p.get('label', '')}：ICC {p.get('icc', 0):.3f}"
                       f" → Δ {p.get('delta', 0):+.4f}"
                       f"（组均规模 {p.get('mean_group_size', 0):.1f}）\n")
            text.insert('end', f"    判定：{p.get('verdict', '')}\n")
        mono = g.get('monotonicity', {})
        if mono:
            text.insert(
                'end', "\n单调性检验：Spearman(ICC, Δ) = "
                       f"{mono.get('spearman_icc_vs_delta', 0):.3f}"
                       f"（{mono.get('note', '')}）\n")
        text.insert('end', f"\n曲线图：{g.get('plot_path', '')}\n")

    # ---------- Tab 13: 部署度量（round-9 P0-2） ----------

    def _build_deployment_metrics_tab(self, nb):
        """部署度量：yield@k / NNS / PPV@预算——评估切换为疾控决策语言。"""
        tab = ttk.Frame(nb)
        nb.add(tab, text="部署度量")

        top = ttk.LabelFrame(
            tab, text="部署度量：筛查前 k% 捕获多少死亡、每例需筛多少人"
                      "（模型不动，只换评估轴）")
        top.pack(fill='x', padx=10, pady=(8, 4))
        self.me_deploy_summary = ttk.Label(
            top, text="正在加载 SINAN far test 部署度量归档...",
            wraplength=900, justify='left')
        self.me_deploy_summary.pack(fill='x', padx=8, pady=4)

        # yield@k 预算曲线表（ind 个体基线 vs ind_all +时序先证）
        cols = ('budget', 'yield_ind', 'yield_ind_all', 'nns', 'ppv', 'lift')
        tree_frame = ttk.Frame(tab)
        tree_frame.pack(fill='both', expand=False, padx=10, pady=5)
        self.me_deploy_tree = ttk.Treeview(tree_frame, columns=cols,
                                           show='headings', height=8)
        for col, text, width in (
                ('budget', '筛查预算 k%', 90),
                ('yield_ind', 'yield@k 基线', 110),
                ('yield_ind_all', 'yield@k +先证', 115),
                ('nns', 'NNS 每例需筛', 110),
                ('ppv', 'PPV 阳性率', 100),
                ('lift', 'lift 倍数', 80)):
            self.me_deploy_tree.heading(col, text=text)
            self.me_deploy_tree.column(col, width=width, anchor='w')
        sb = ttk.Scrollbar(tree_frame, orient='vertical',
                           command=self.me_deploy_tree.yview)
        self.me_deploy_tree.configure(yscrollcommand=sb.set)
        self.me_deploy_tree.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        self._register_tree_widget(self.me_deploy_tree)

        # round-10 P2：双头预算曲线（死亡头 vs 细菌学头——任务 A'/B）
        # round-13：v2 特征（重叠 0.5%）
        dh_frame = ttk.LabelFrame(
            tab, text="双头架构：任务 A'=细菌学确诊（找传染源）vs "
                      "任务 B=死亡预后（v2 top-decile 重叠 0.5%，正交）")
        dh_frame.pack(fill='x', padx=10, pady=5)
        dh_cols = ('budget', 'yield_death', 'yield_bact',
                   'nns_death', 'nns_bact')
        self.me_dual_head_tree = ttk.Treeview(dh_frame, columns=dh_cols,
                                              show='headings', height=7)
        for col, text, width in (
                ('budget', '筛查预算 k%', 90),
                ('yield_death', 'yield@k 死亡头', 110),
                ('yield_bact', 'yield@k 细菌学头', 115),
                ('nns_death', 'NNS 死亡头', 100),
                ('nns_bact', 'NNS 细菌学头', 105)):
            self.me_dual_head_tree.heading(col, text=text)
            self.me_dual_head_tree.column(col, width=width, anchor='w')
        self.me_dual_head_tree.pack(fill='x', padx=8, pady=4)
        self._register_tree_widget(self.me_dual_head_tree)

        # round-10 P3：分龄阈值策略表（预算 → 分龄阈值 → 检出/漏检）
        thr_frame = ttk.LabelFrame(
            tab, text="分龄阈值策略表：预算 X 时该筛哪些人"
                      "（组内 top-k% 分数阈值，老年/成人两套）")
        thr_frame.pack(fill='x', padx=10, pady=5)
        thr_cols = ('budget', 'thr_eld', 'eld_caught', 'eld_missed',
                    'thr_adu', 'adu_caught', 'adu_missed')
        self.me_threshold_tree = ttk.Treeview(thr_frame, columns=thr_cols,
                                               show='headings', height=7)
        for col, text, width in (
                ('budget', '预算 k%', 70),
                ('thr_eld', '老年阈值', 85),
                ('eld_caught', '老年检出', 90),
                ('eld_missed', '老年漏检', 90),
                ('thr_adu', '成人阈值', 85),
                ('adu_caught', '成人检出', 90),
                ('adu_missed', '成人漏检', 90)):
            self.me_threshold_tree.heading(col, text=text)
            self.me_threshold_tree.column(col, width=width, anchor='w')
        self.me_threshold_tree.pack(fill='x', padx=8, pady=4)
        self._register_tree_widget(self.me_threshold_tree)

        # round-11 P2-4：预算滑块——预算 → 各年龄阈值 → 检出/漏检直读
        # round-12 P1-2：终点切换（全因=预后分层 / E3=传染控制）
        self._me_thr_rows = []
        self._me_thr_by_ep = {}
        self._me_current_ep = 'allcause'
        self._me_ep_map = {'全因死亡（预后分层）': 'allcause',
                           'E3 TB 死亡（传染控制）': 'e3_tb_death'}
        slide_frame = ttk.LabelFrame(
            tab, text="预算滑块：拖动预算 → 各年龄组阈值与预期检出/漏检"
                      "（终点可切换，档位与策略表一致，不插值）")
        slide_frame.pack(fill='x', padx=10, pady=5)
        ep_row = ttk.Frame(slide_frame)
        ep_row.pack(fill='x', padx=8, pady=(4, 0))
        ttk.Label(ep_row, text="终点").pack(side='left')
        self.me_endpoint_combo = ttk.Combobox(
            ep_row, state='readonly', width=24,
            values=list(self._me_ep_map.keys()))
        self.me_endpoint_combo.current(0)
        self.me_endpoint_combo.bind('<<ComboboxSelected>>',
                                    self._on_endpoint_change)
        self.me_endpoint_combo.pack(side='left', padx=(4, 0))
        self.me_budget_slider = ttk.Scale(
            slide_frame, from_=0, to=6, length=560,
            command=self._on_budget_slider_change)
        self.me_budget_slider.set(2)
        self.me_budget_slider.pack(fill='x', padx=8, pady=(4, 0))
        self.me_budget_readout = ttk.Label(
            slide_frame, text="拖动选择预算（1/5/10/20/30/50/100%）",
            justify='left', wraplength=880)
        self.me_budget_readout.pack(fill='x', padx=8, pady=(2, 4))

        self.me_deploy_detail = self._me_make_text(tab, height=18)

    def _update_deployment_metrics_ui(self):
        d = self._me_cache.get('deployment') or {}
        if not d.get('available'):
            self.me_deploy_summary.configure(
                text=f"部署度量不可用：{d.get('reason', '未知原因')}")
            return
        self.me_deploy_summary.configure(text=d.get('narrative', ''))

        # yield 表
        tree = self.me_deploy_tree
        for iid in tree.get_children():
            tree.delete(iid)
        random_note = '（随机基线 yield@k = k%）'
        for row in d.get('sinan', {}).get('yield_rows', []):
            yi = row.get('yield_ind')
            ya = row.get('yield_ind_all')
            tree.insert('', 'end', values=(
                f"{row.get('budget_pct', '?')}%",
                f"{yi:.1%}" if yi is not None else '--',
                f"{ya:.1%}" if ya is not None else '--',
                f"{row.get('nns_ind_all', 0):.2f}",
                f"{row.get('ppv_ind_all', 0):.1%}",
                f"{row.get('lift_ind_all', 0):.1f}x"))
        if not tree.get_children():
            tree.insert('', 'end', values=(random_note, '--', '--', '--',
                                           '--', '--'))

        # round-10 P2：双头预算曲线表
        dh_tree = self.me_dual_head_tree
        for iid in dh_tree.get_children():
            dh_tree.delete(iid)
        dh = d.get('dual_head') or {}
        for row in dh.get('yield_rows', []):
            yd, yb = row.get('yield_death'), row.get('yield_bact')
            nd, nb = row.get('nns_death'), row.get('nns_bact')
            dh_tree.insert('', 'end', values=(
                f"{row.get('budget_pct', '?')}%",
                f"{yd:.1%}" if yd is not None else '--',
                f"{yb:.1%}" if yb is not None else '--',
                f"{nd:.2f}" if nd is not None else '--',
                f"{nb:.2f}" if nb is not None else '--'))

        # round-10 P3：分龄阈值策略表
        thr_tree = self.me_threshold_tree
        for iid in thr_tree.get_children():
            thr_tree.delete(iid)
        at = d.get('age_thresholds') or {}
        for row in at.get('rows', []):
            te_, ta_ = row.get('thr_elderly'), row.get('thr_adult')
            thr_tree.insert('', 'end', values=(
                f"{row.get('budget_pct', '?')}%",
                f"{te_:.3f}" if te_ is not None else '--',
                row.get('caught_elderly', '--'),
                row.get('missed_elderly', '--'),
                f"{ta_:.3f}" if ta_ is not None else '--',
                row.get('caught_adult', '--'),
                row.get('missed_adult', '--')))
        self._me_thr_by_ep = {}
        de_thr = ((d.get('elderly_dual_endpoint') or {})
                  .get('threshold_rows') or {})
        if de_thr:
            self._me_thr_by_ep = {k: [r for r in v.get('rows', [])
                                      if r.get('thr_elderly') is not None]
                                  for k, v in de_thr.items()}
        elif at.get('rows'):
            self._me_thr_by_ep = {
                'allcause': [r for r in at.get('rows', [])
                             if r.get('thr_elderly') is not None]}
        eps = [k for k in ('allcause', 'e3_tb_death')
               if self._me_thr_by_ep.get(k)]
        if len(eps) > 1:
            self.me_endpoint_combo.configure(state='readonly')
        else:
            self.me_endpoint_combo.configure(state='disabled')
        self._me_current_ep = eps[0] if eps else 'allcause'
        disp = {v: k for k, v in self._me_ep_map.items()}
        self.me_endpoint_combo.set(
            disp.get(self._me_current_ep, '全因死亡（预后分层）'))
        self._me_thr_rows = self._me_thr_by_ep.get(self._me_current_ep, [])
        if self._me_thr_rows:
            self.me_budget_slider.configure(
                to=len(self._me_thr_rows) - 1)
            self._refresh_budget_readout()

        # 详情文本
        text = self.me_deploy_detail
        text.delete('1.0', 'end')
        s = d.get('sinan', {})
        pg = s.get('prior_gain', {})
        y10 = pg.get('yield_at', {}).get('10', {})
        text.insert('end', "SINAN far test（2018-19 时间外推，全因死亡终点）——\n")
        text.insert(
            'end', f"  个体基线 AUROC {s.get('auroc_ind', 0):.4f}"
                   f" → +时序先证 {s.get('auroc_ind_all', 0):.4f}"
                   f"（Δ {pg.get('auroc_delta', 0):+.4f}）\n")
        text.insert(
            'end', f"  先证部署增益：yield@10% {y10.get('ind', 0):.1%}"
                   f" → {y10.get('ind_all', 0):.1%}"
                   f"（Δ {y10.get('delta', 0):+.2%}）\n")
        eld = s.get('elderly', {})
        adu = s.get('adult', {})
        text.insert(
            'end', f"  老年 65+：AUROC {eld.get('auroc', 0):.4f}"
                   f"（事件率 {eld.get('event_rate', 0):.1%}）vs 成人 "
                   f"{adu.get('auroc', 0):.4f}——塌方在部署语言下"
                   " = 同预算少捕获 17 个百分点\n")

        ken = d.get('kenya_case_population') or {}
        if ken:
            text.insert('end', "\nKenya 患病率调查（case-population 采样偏差）——\n")
            text.insert(
                'end', f"  系统从未接触的确诊比例：{ken.get('miss_rate', 0):.1%}"
                       f"（{ken.get('n_missed', 0)}/{ken.get('n_pos', 0)}）\n")
            text.insert(
                'end', f"  漏检 HIV 率 {ken.get('missed_hiv_pct', 0):.1%} vs "
                       f"已发现 {ken.get('found_hiv_pct', 0):.1%}"
                       "（HIV 阴性者被系统性漏检）\n")
            text.insert(
                'end', f"  高风险池模型 OOF AUROC {ken.get('oof_auroc', 0):.4f}："
                       f"top-10% 捕获 {ken.get('yield_top10', 0):.1%} 确诊，"
                       f"其中 {ken.get('top10_never_treated_pos', 0)} 名"
                       "从未接触系统（被动就诊永远不可及的增益池）\n")

        bac = d.get('bacteriology_head') or {}
        if bac:
            text.insert('end', "\n「找患者」第二头（细菌学确诊，诚实特征）——\n")
            text.insert(
                'end', f"  死亡头 AUROC {bac.get('death_auroc', 0):.4f} / "
                       f"细菌学头 {bac.get('bact_auroc', 0):.4f}"
                       f"（yield@10% {bac.get('bact_yield10', 0):.1%}）\n")
            text.insert(
                'end', f"  top-decile 重叠 {bac.get('top10_overlap', 0):.1%}"
                       "：两语义近乎正交——部署需双头，非单模型可覆盖\n")

        # round-10 P2：双头架构正式化（任务映射 + 预测器）
        dh = d.get('dual_head') or {}
        if dh:
            text.insert('end', "\n双头架构正式化（0.7% 重叠的定量依据）——\n")
            tm = dh.get('task_mapping') or {}
            for key in ('A_prime', 'B'):
                if tm.get(key):
                    text.insert('end', f"  {tm[key]}\n")
            text.insert('end', f"  {dh.get('architecture_note', '')}\n")
            hv2 = dh.get('heads_v2') or {}
            if hv2:
                text.insert(
                    'end', f"  v2 头指标：死亡 AUROC "
                           f"{hv2.get('death_auroc', 0):.4f} / AUPRC "
                           f"{hv2.get('death_auprc', 0):.4f} | 细菌学 "
                           f"AUROC {hv2.get('bact_auroc', 0):.4f} / "
                           f"AUPRC {hv2.get('bact_auprc', 0):.4f}"
                           "（基率 57% 下 AUPRC 为准）\n")
            text.insert('end', f"  预测器：{dh.get('predictor_api', '')}\n")
            ma = dh.get('model_artifacts') or {}
            if ma.get('saved'):
                text.insert(
                    'end', f"  模型工件：{ma.get('model_dir', '')}"
                           f"（{ma.get('version', '')}，"
                           f"{len(ma.get('files', []))} 个文件）\n")

        # round-10 P3：分龄阈值策略（操作答案）
        at = d.get('age_thresholds') or {}
        if at:
            rows = at.get('rows', [])
            r10 = next((r for r in rows if r.get('budget_pct') == '10'),
                       None)
            if r10:
                text.insert('end', "\n分龄阈值策略（预算 10% 示例）——\n")
                text.insert(
                    'end', f"  老年：风险分 ≥ {r10['thr_elderly']:.3f} → "
                           f"检出 {r10['caught_elderly']}/"
                           f"{at.get('n_events_elderly', 0)}"
                           f"（漏检 {r10['missed_elderly']}）\n")
                text.insert(
                    'end', f"  成人：风险分 ≥ {r10['thr_adult']:.3f} → "
                           f"检出 {r10['caught_adult']}/"
                           f"{at.get('n_events_adult', 0)}"
                           f"（漏检 {r10['missed_adult']}）\n")
                text.insert(
                    'end', "  注：老年基线风险高（24.7% vs 8.5%）→ "
                           "同预算阈值显著更高；预算→阈值→检出/漏检"
                           "全表见上方策略表\n")

        # round-10 P4：老年双死因分解（塌方的结构性解释）
        ec = d.get('elderly_dual_cause') or {}
        if ec:
            text.insert('end', f"\n{ec.get('headline', '')}——\n")
            text.insert('end', f"  {ec.get('narrative', '')}\n")
            text.insert('end', f"  {ec.get('paper_note', '')}\n")

        cr = d.get('competing_risk') or {}
        if cr:
            text.insert('end', "\n竞争风险（cause-specific Cox，老年）——\n")
            if cr.get('elderly_tb_death_c') is not None:
                text.insert(
                    'end', f"  TB 死亡 IPCW C {cr.get('elderly_tb_death_c', 0):.4f}"
                           f" / 非 TB 死亡 {cr.get('elderly_nontb_c', 0):.4f}"
                           "（HIV 把死亡分流到竞争事件）\n")
            for e3, e4 in ((cr.get('E3_auroc'), cr.get('E4_auroc')),):
                if e3 is not None:
                    text.insert(
                        'end', f"  E3 TB死亡 {e3:.4f} / E4 复合不良结局 {e4:.4f}"
                               "（老年 far test 对照）\n")

        trf = d.get('cross_population_transfer') or {}
        if trf:
            text.insert('end', "\n跨人群迁移衰减（巴西训练 → 各国测试，三特征交集）——\n")
            text.insert(
                'end', f"  巴西 far test 参照 AUROC "
                       f"{trf.get('br_reference_auroc', 0):.4f}"
                       "（内部时间外推）\n")
            for c in trf.get('cohorts', []):
                text.insert(
                    'end', f"  {c.get('cohort', '')}：迁移 "
                           f"{c.get('transfer_auroc', 0):.4f} / 本地 CV "
                           f"{c.get('local_cv_auroc', 0):.4f}"
                           f"±{c.get('local_cv_std', 0):.3f}"
                           f"（人群衰减 {c.get('population_decay', 0):+.4f}）\n")
            for e in trf.get('excluded', []):
                text.insert('end', f"  排除——{e.get('cohort', '')}："
                                   f"{e.get('reason', '')}\n")
            text.insert(
                'end', "  caveat：终点异质（死亡/确诊/症状），负迁移"
                       "（<0.5）是跨终点部署风险的经验证据\n")
            rl = trf.get('deployment_red_line')
            if rl:
                text.insert('end', f"  部署红线：{rl}\n")

        # round-11 P1-1：老年双终点报告（全因 vs E3 并列）
        de2 = d.get('elderly_dual_endpoint') or {}
        if de2:
            text.insert('end', f"\n{de2.get('headline', '')}——\n")
            text.insert('end', f"  {de2.get('decision_rule', '')}\n")
            for r in de2.get('rows', []):
                if r.get('available') is False:
                    continue
                text.insert(
                    'end', f"  {r.get('endpoint')} / {r.get('group')}: "
                           f"AUROC {r.get('auroc', 0):.4f} | "
                           f"yield@10% {r.get('yield10', 0):.1%} | "
                           f"thr@10% {r.get('thr10', 0):.3f}"
                           f"（检出 {r.get('caught10')} / "
                           f"漏检 {r.get('missed10')}）\n")
            text.insert(
                'end', f"  老年全因死亡 {de2.get('elderly_non_tb_share', 0):.0%}"
                       " 为非 TB——全因臂的 NNS 效率含模型无法干预的"
                       "死亡；双终点并列报告是纠偏最直接的方式\n")

        # round-11 P1-2：本地重校准最小样本量曲线（v4 四臂 + 裁定成本）
        rc = d.get('transfer_recalibration') or {}
        if rc:
            text.insert('end', f"\n{rc.get('headline', '')}——\n")
            for c in rc.get('cohorts', []):
                n_st = c.get('n_to_recover_stack')
                n_p4 = c.get('n_to_recover_stack_prior')
                n_v2 = c.get('n_to_recover_features')
                curve = c.get('curve', [])
                err_first = (curve[0].get('ruling_error_rate')
                             if curve else None)
                err_last = (curve[-1].get('ruling_error_rate')
                            if curve else None)
                text.insert(
                    'end', f"  {c.get('cohort', '')}（{c.get('direction_truth', '?')}）："
                           f"zero-shot {c.get('zero_shot_auroc', 0):.4f} / "
                           f"全量本地 {c.get('full_local_cv_auroc', 0):.4f}"
                           f" | 恢复 N：v4先验="
                           f"{n_p4 if n_p4 is not None else '未达'} / "
                           f"v3stack={n_st if n_st is not None else '未达'}"
                           f" / v2加性={n_v2 if n_v2 is not None else '未达'}\n")
                if err_first is not None and err_last is not None:
                    text.insert(
                        'end', f"    方向裁定错误率 vs N：首档 "
                               f"{err_first:.0%} → 末档 "
                               f"{err_last:.0%}（事件数不足时≈抛硬币）\n")
            text.insert('end', f"  {rc.get('finding', '')}\n")
            text.insert('end', f"  样本量规则：{rc.get('sample_size_rule', '')}\n")

        # round-11 P1-3：细菌学头 v2（症状列裁定 + HIV/职业扩展）
        bv = d.get('bact_head_v2') or {}
        if bv:
            text.insert('end', f"\n{bv.get('headline', '')}——\n")
            sf = bv.get('symptom_feasibility') or {}
            text.insert('end', f"  症状列裁定：{sf.get('verdict', '')}——"
                               f"{sf.get('evidence', '')}\n")
            g = bv.get('v2_gain') or {}
            text.insert(
                'end', f"  HIV/职业扩展：死亡 AUROC "
                       f"{bv.get('v1_death_auroc', 0):.4f}→"
                       f"{bv.get('v2_death_auroc', 0):.4f}"
                       f"（+{g.get('death_auroc', 0):.4f}）/ 细菌学 "
                       f"{bv.get('v1_bact_auroc', 0):.4f}→"
                       f"{bv.get('v2_bact_auroc', 0):.4f}"
                       f"（AUPRC {bv.get('v2_bact_auprc', 0):.4f}）\n")
            text.insert('end', f"  {bv.get('positioning', '')}\n")
            ma2 = bv.get('model_artifacts_v2') or {}
            if ma2.get('saved'):
                text.insert(
                    'end', "  预测器 v2：DualHeadPredictor "
                           "MODEL_VERSION='v2' 工件已落盘"
                           "（+HIV 检测结果 + 职业编码 top-30）\n")

        # round-11 P3-6：时序先证 SINAN 封存（分组粒度受限）
        tp = d.get('temporal_prior_sealed') or {}
        if tp:
            text.insert('end', f"\n{tp.get('conclusion', '')}——\n")
            text.insert('end', f"  {tp.get('evidence', '')}\n")
            text.insert('end', f"  {tp.get('sinan_residual_gain', '')}\n")
            text.insert('end', f"  {tp.get('implication', '')}\n")

        text.insert('end', f"\n共用 API：{d.get('planner_api', '')}\n")

    def _on_budget_slider_change(self, _value):
        self._refresh_budget_readout()

    def _on_endpoint_change(self, _event=None):
        """终点切换（全因 / E3 TB 死亡）→ 重载阈值行并刷新读数。"""
        key = self._me_ep_map.get(self.me_endpoint_combo.get())
        if not key or key not in self._me_thr_by_ep:
            return
        self._me_current_ep = key
        self._me_thr_rows = self._me_thr_by_ep.get(key, [])
        if self._me_thr_rows:
            self.me_budget_slider.configure(
                to=len(self._me_thr_rows) - 1)
            self._refresh_budget_readout()

    def _refresh_budget_readout(self):
        """预算滑块 → 当前预算下各年龄组阈值/检出/漏检直读。"""
        rows = getattr(self, '_me_thr_rows', None) or []
        if not rows:
            return
        try:
            idx = int(round(float(self.me_budget_slider.get())))
        except (TypeError, ValueError):
            return
        idx = max(0, min(idx, len(rows) - 1))
        r = rows[idx]
        te_, ta_ = r.get('thr_elderly'), r.get('thr_adult')
        txt = (f"预算 {r.get('budget_pct', '?')}%："
               f"老年 风险分 ≥ {te_:.3f} → 检出 "
               f"{r.get('caught_elderly', '--')} / 漏检 "
               f"{r.get('missed_elderly', '--')}"
               f"  |  成人 风险分 ≥ {ta_:.3f} → 检出 "
               f"{r.get('caught_adult', '--')} / 漏检 "
               f"{r.get('missed_adult', '--')}"
               "（组内 top-k% 阈值，far test 预期值）")
        self.me_budget_readout.configure(text=txt)

    def _compute_static_evidence_background(self):
        """后台线程：一次性计算全部静态证据，完成后回主线程渲染。"""
        cache = {}
        try:
            cache['tasks'] = _evidence.collect_task_evaluations()
        except Exception as e:
            cache['tasks'] = {'available': False, 'reason': str(e)}
        try:
            cache['ablation'] = _evidence.collect_network_ablation()
        except Exception as e:
            cache['ablation'] = {'available': False, 'reason': str(e)}
        try:
            cache['public'] = _evidence.collect_public_data_evidence()
        except Exception as e:
            cache['public'] = {'available': False, 'reason': str(e)}
        try:
            cache['dgp'] = _evidence.collect_dgp_evidence(n_samples=5000)
        except Exception as e:
            cache['dgp'] = {'available': False, 'reason': str(e)}
        try:
            cache['seir'] = _evidence.collect_seir_intervention()
        except Exception as e:
            cache['seir'] = {'available': False, 'reason': str(e)}
        try:
            cache['sinan'] = _evidence.collect_sinan_scale_evidence()
        except Exception as e:
            cache['sinan'] = {'available': False, 'reason': str(e)}
        try:
            cache['negative'] = _evidence.collect_negative_findings()
        except Exception as e:
            cache['negative'] = {'available': False, 'reason': str(e)}

        try:
            cache['granularity'] = _evidence.collect_granularity_guide()
        except Exception as e:
            cache['granularity'] = {'available': False, 'reason': str(e)}

        try:
            cache['deployment'] = _evidence.collect_deployment_metrics()
        except Exception as e:
            cache['deployment'] = {'available': False, 'reason': str(e)}

        if not getattr(self, '_closing', False) and self.root is not None:
            try:
                self.root.after(0, lambda: self._on_static_evidence_ready(cache))
            except (RuntimeError, tk.TclError):
                pass

    def _on_static_evidence_ready(self, cache):
        """主线程：静态证据就绪，渲染各标签页。"""
        if getattr(self, '_closing', False):
            return
        self._me_cache.update(cache)
        try:
            self._update_task_a_ui()
            self._update_task_b_ui()
            self._update_task_c_ui()
            self._update_ablation_ui()
            self._update_public_data_ui()
            self._update_dgp_ui()
            self._update_seir_ui()
            self._update_scale_validation_ui()
            self._update_negative_findings_ui()
        except Exception as e:
            LOGGER.debug("模型证据渲染失败（非致命）: %s", e, exc_info=True)

        try:
            self._update_granularity_guide_ui()
        except Exception as e:
            LOGGER.debug("组粒度指南渲染失败（非致命）: %s", e, exc_info=True)

        try:
            self._update_deployment_metrics_ui()
        except Exception as e:
            LOGGER.debug("部署度量渲染失败（非致命）: %s", e, exc_info=True)

    def refresh_model_evidence_three_layer(self):
        """ML 预测完成后刷新三层递进标签页（主线程调用）。"""
        if getattr(self, '_closing', False):
            return
        try:
            with getattr(self, 'ml_results_lock', threading.Lock()):
                ml_results = getattr(self, 'ml_results', None)
            series = _evidence.extract_three_layer_series(ml_results)
            self._me_cache['three_layer'] = series
            if series.get('available'):
                names = [c['name'] for c in series['contacts']]
                self.me_contact_combo['values'] = names
                if names and not self.me_contact_var.get():
                    self.me_contact_var.set(names[0])
                # 回填 self.results，使结果文本区的三层/任务摘要也能显示
                self._backfill_results_three_layer(series)
                # 重新渲染结果文本区（三层摘要回填后需要刷新才能可见）
                if getattr(self, 'adapter', None) is not None:
                    try:
                        self.adapter.display_results()
                    except Exception as e:
                        LOGGER.debug("结果文本区刷新失败（非致命）: %s", e)
            self._draw_three_layer_waterfall()
            # ML 基线刷新后，时序家庭筛查面板按签名自动重建并重绘
            self._draw_temporal_household_chart()
        except Exception as e:
            LOGGER.debug("三层递进刷新失败（非致命）: %s", e, exc_info=True)

    def _backfill_results_three_layer(self, series):
        """把最高决策风险的接触者三层报告回填到 self.results（供文本摘要）。"""
        try:
            contacts = [c for c in series['contacts'] if c.get('raw')]
            if not contacts or not isinstance(self.results, dict):
                return
            top = max(contacts,
                      key=lambda c: c.get('decision_risk') or 0.0)
            self.results['three_layer'] = top['raw']
            tasks = self._me_cache.get('tasks') or {}
            if tasks.get('available'):
                self.results['task_reports'] = {
                    tid: {
                        'task_name': tasks[tid].get('task_name', ''),
                        'decision_meaning': tasks[tid].get('decision_meaning', ''),
                    }
                    for tid in ('A', 'B', 'C') if tid in tasks
                }
        except Exception:
            LOGGER.debug("三层结果回填失败（非致命）", exc_info=True)

    # ==================== 渲染：三层递进 ====================

    def _selected_three_layer_contact(self):
        series = self._me_cache.get('three_layer') or {}
        if not series.get('available'):
            return None, series
        name = self.me_contact_var.get()
        for c in series['contacts']:
            if c['name'] == name:
                return c, series
        return series['contacts'][0], series

    def _draw_three_layer_waterfall(self):
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'me_three_layer_figure'):
            return
        fig, canvas = self.me_three_layer_figure, self.me_three_layer_canvas
        contact, series = self._selected_three_layer_contact()
        fig.clear()
        if contact is None:
            reason = series.get('reason', '请先运行风险评估（含 ML 预测）')
            self._me_placeholder(fig, canvas, reason)
            return

        p_base = contact.get('p_base') or 0.0
        gated = contact.get('gated_increment')
        if gated is None:
            gated = 0.0
        combined = contact.get('combined')
        if combined is None:
            combined = max(0.0, min(100.0, p_base + gated))
        gating = contact.get('gating')
        benefit = contact.get('intervention_benefit') or 0.0
        decision = contact.get('decision_risk')

        colors = getattr(self, 'COLORS', {})
        c1 = colors.get('chart_series_3', '#45B7D1')
        c2 = '#2ecc71' if gated >= 0 else '#e74c3c'
        c3 = colors.get('accent', '#0f766e')

        # 左：瀑布图（第 1 层 → 第 2 层 → 决策层联合）
        ax = fig.add_subplot(1, 2, 1)
        xs = [0, 1, 2]
        ax.bar(xs[0], p_base, color=c1, width=0.6)
        ax.bar(xs[1], abs(gated), bottom=min(p_base, p_base + gated),
               color=c2, width=0.6)
        ax.bar(xs[2], combined, color=c3, width=0.6)
        ax.plot([0.3, 0.7], [p_base, p_base], ls='--', lw=0.8, color='gray')
        ax.plot([1.3, 1.7], [combined, combined], ls='--', lw=0.8, color='gray')
        ax.set_xticks(xs)
        ax.set_xticklabels(['第1层\n个体基线\nP_base',
                            '第2层\n门控网络增量\n(g×Δ网络)',
                            '决策层\n联合概率'], fontsize=9)
        ax.set_ylabel('概率 (%)')
        ax.set_ylim(0, max(100.0, combined + 10))
        ax.set_title(f"三层递进瀑布图 — {contact['name']}"
                     f"（{'家庭' if contact.get('contact_type') == 'family' else '社会'}）",
                     fontsize=11, fontweight='bold')
        for x, v, label in ((xs[0], p_base, f"{p_base:.1f}"),
                            (xs[1], p_base + gated, f"{gated:+.1f}"),
                            (xs[2], combined, f"{combined:.1f}")):
            ax.annotate(label, (x, v), textcoords='offset points',
                        xytext=(0, 4), ha='center', fontsize=9, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # 右：门控与决策分解
        ax_gate = fig.add_subplot(2, 2, 2)
        g = gating if isinstance(gating, (int, float)) else 0.0
        ax_gate.barh([0], [g], color=c2, height=0.5)
        ax_gate.set_xlim(0, 1)
        ax_gate.set_yticks([0])
        ax_gate.set_yticklabels(['门控 g'])
        ax_gate.set_title(f"网络门控系数 g = {g:.2f}"
                          + ('' if contact.get('network_aware') else '（未用网络结构）'),
                          fontsize=10)
        ax_gate.grid(axis='x', alpha=0.3)

        ax_dec = fig.add_subplot(2, 2, 4)
        w_ind, w_int = 0.6, 0.1
        denom = w_ind + w_int
        part_ind = w_ind * combined / denom
        part_int = w_int * benefit / denom
        ax_dec.barh([0], [part_ind], color=c1, height=0.5,
                    label=f'个体×联合 ({w_ind:.1f}/{denom:.1f})')
        ax_dec.barh([0], [part_int], left=[part_ind], color=c3, height=0.5,
                    label=f'干预收益 ({w_int:.1f}/{denom:.1f})')
        ax_dec.set_xlim(0, max(100.0, part_ind + part_int + 10))
        ax_dec.set_yticks([0])
        ax_dec.set_yticklabels(['决策风险'])
        d_str = f"{decision:.1f}" if isinstance(decision, (int, float)) else 'N/A'
        ax_dec.set_title(f"决策层加权 = {d_str}（堆叠+门控，非概率平均）", fontsize=10)
        ax_dec.legend(fontsize=8, loc='lower right')
        ax_dec.grid(axis='x', alpha=0.3)

        fig.tight_layout()
        canvas.draw_idle()

        # 信息行
        aware = '使用网络结构' if contact.get('network_aware') else '未使用网络结构'
        cls = contact.get('decision_class')
        cls_str = {0: '低风险', 1: '高风险'}.get(cls, 'N/A')
        strat = contact.get('best_strategy') or '—'
        self.me_three_layer_info.configure(
            text=f"网络增量 {contact.get('network_increment') or 0:+.1f}% | {aware} | "
                 f"决策分级 {cls_str} | 最优干预 {strat}",
            foreground='')

    # ==================== 渲染：任务 A ====================

    def _on_task_a_threshold_changed(self, _value=None):
        try:
            thr = float(self.me_task_a_threshold_var.get())
        except (TypeError, ValueError):
            return
        self.me_task_a_threshold_label.configure(text=f"当前阈值: {thr:.1f}%")
        self._update_task_a_metrics(thr)
        self._draw_task_a_chart()

    def _update_task_a_ui(self):
        tasks = self._me_cache.get('tasks') or {}
        task_a = tasks.get('A') or {}
        # round-10 P2：双头映射说明（SINAN 侧的部署语义）
        mapping = tasks.get('sinan_dual_head_mapping') or {}
        if mapping:
            self.me_task_a_dual_head_note.configure(
                text=f"映射：{mapping.get('A_prime', '')}；"
                     f"{mapping.get('B', '')}")
        if not tasks.get('available') or not task_a:
            return
        thr = task_a.get('threshold', 15.0)
        self.me_task_a_threshold_var.set(float(thr))
        self._update_task_a_metrics(float(thr), base_report=task_a)
        self._draw_task_a_chart()

    def _update_task_a_metrics(self, threshold, base_report=None):
        tasks = self._me_cache.get('tasks') or {}
        task_a = base_report or tasks.get('A') or {}
        scores = task_a.get('_scores') or []
        labels = task_a.get('_labels') or []
        if not scores:
            return
        m = task_a_metrics_at_threshold(scores, labels, threshold)
        ece = task_a.get('metrics', {}).get('expected_calibration_error')
        lbl = self.me_task_a_labels
        lbl['sensitivity'].configure(text=f"敏感度: {m['sensitivity']:.3f}")
        lbl['specificity'].configure(text=f"特异度: {m['specificity']:.3f}")
        lbl['ppv'].configure(text=f"阳性预测值: {m['ppv']:.3f}")
        lbl['npv'].configure(text=f"阴性预测值: {m['npv']:.3f}")
        lbl['youden'].configure(text=f"约登指数: {m['youden']:.3f}")
        ece_str = f"{ece:.3f}" if isinstance(ece, (int, float)) else '--'
        lbl['ece'].configure(text=f"校准误差ECE: {ece_str}")
        lbl['threshold'].configure(text=f"阈值: {threshold:.1f}%")
        lbl['n'].configure(
            text=f"样本: {task_a.get('n_samples', len(labels))} | "
                 f"混淆矩阵 TP/FP/TN/FN = {m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}")

    def _draw_task_a_chart(self):
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'me_task_a_figure'):
            return
        fig, canvas = self.me_task_a_figure, self.me_task_a_canvas
        tasks = self._me_cache.get('tasks') or {}
        task_a = tasks.get('A') or {}
        scores = task_a.get('_scores') or []
        labels = task_a.get('_labels') or []
        fig.clear()
        if not scores:
            self._me_placeholder(fig, canvas,
                                 tasks.get('reason', '任务 A 评估不可用'))
            return

        try:
            thr = float(self.me_task_a_threshold_var.get())
        except (TypeError, ValueError, tk.TclError):
            thr = task_a.get('threshold', 15.0)

        grid = [float(t) for t in range(1, 100, 2)]
        sens, spec, ppv, npv = [], [], [], []
        for t in grid:
            m = task_a_metrics_at_threshold(scores, labels, t)
            sens.append(m['sensitivity'])
            spec.append(m['specificity'])
            ppv.append(m['ppv'])
            npv.append(m['npv'])

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(grid, sens, label='敏感度', color='#e74c3c')
        ax1.plot(grid, spec, label='特异度', color='#3498db')
        ax1.plot(grid, ppv, label='阳性预测值', color='#f39c12', ls='--')
        ax1.plot(grid, npv, label='阴性预测值', color='#2ecc71', ls='--')
        ax1.axvline(thr, color='gray', ls=':', lw=1.2)
        ax1.set_xlabel('筛查决策阈值 (%)')
        ax1.set_ylabel('指标值')
        ax1.set_title('指标随阈值变化', fontsize=10)
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 1.05)
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        ax2 = fig.add_subplot(1, 2, 2)
        m = task_a_metrics_at_threshold(scores, labels, thr)
        keys = ['sensitivity', 'specificity', 'ppv', 'npv']
        zh = ['敏感度', '特异度', 'PPV', 'NPV']
        vals = [m[k] for k in keys]
        bar_colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']
        ax2.bar(zh, vals, color=bar_colors)
        ax2.set_ylim(0, 1.05)
        ax2.set_title(f"阈值 = {thr:.1f}% 时的诊断指标", fontsize=10)
        for i, v in enumerate(vals):
            ax2.annotate(f"{v:.2f}", (i, v), textcoords='offset points',
                         xytext=(0, 3), ha='center', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)

        fig.tight_layout()
        canvas.draw_idle()

    # ==================== 渲染：任务 B ====================

    def _update_task_b_ui(self):
        tasks = self._me_cache.get('tasks') or {}
        task_b = tasks.get('B') or {}
        if not tasks.get('available') or not task_b:
            return
        metrics = task_b.get('metrics', {})
        lbl = self.me_task_b_labels
        lbl['c_index'].configure(
            text=f"C-index: {metrics.get('c_index', 0):.3f}")
        lbl['horizon_auc'].configure(
            text=f"固定时点AUC: {metrics.get('horizon_auc', 0):.3f}")
        lbl['ece'].configure(
            text=f"校准误差ECE: {metrics.get('expected_calibration_error', 0):.3f}")
        lbl['n_events'].configure(
            text=f"事件数: {task_b.get('n_events', 0)}/{task_b.get('n_samples', 0)}")
        self.me_task_b_note.configure(text=task_b.get('transition_note', ''))
        self._draw_task_b_charts()

    def _draw_task_b_charts(self):
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'me_task_b_figure'):
            return
        fig, canvas = self.me_task_b_figure, self.me_task_b_canvas
        tasks = self._me_cache.get('tasks') or {}
        task_b = tasks.get('B') or {}
        fig.clear()
        if not tasks.get('available') or not task_b:
            self._me_placeholder(fig, canvas,
                                 tasks.get('reason', '任务 B 评估不可用'))
            return

        # 左：DCA 决策曲线
        ax1 = fig.add_subplot(1, 2, 1)
        dca = task_b.get('decision_curve_analysis') or {}
        curves = dca.get('curves') or []
        if curves:
            ths = [c['threshold'] for c in curves]
            ax1.plot(ths, [c['net_benefit'] for c in curves],
                     label='模型', color='#0f766e', lw=2)
            ax1.plot(ths, [c['treat_all'] for c in curves],
                     label='全部治疗', color='#e74c3c', ls='--')
            ax1.axhline(0, color='gray', ls=':', lw=1, label='均不治疗')
            best = dca.get('best_threshold')
            if isinstance(best, (int, float)):
                ax1.axvline(best, color='#f39c12', ls=':', lw=1.2)
                ax1.annotate(f"最优阈值 {best:.2f}", (best, ax1.get_ylim()[1]),
                             textcoords='offset points', xytext=(4, -10),
                             fontsize=8, color='#f39c12')
            ax1.set_xlabel('阈值概率')
            ax1.set_ylabel('净收益')
            ax1.set_title('决策曲线分析 (DCA)', fontsize=10)
            ax1.legend(fontsize=8)
            ax1.grid(alpha=0.3)

        # 右：Kaplan-Meier 随访生存曲线（按预测风险中位数分层）
        ax2 = fig.add_subplot(1, 2, 2)
        times = task_b.get('_event_times') or []
        labels = task_b.get('_labels') or []
        task_a = tasks.get('A') or {}
        scores = task_a.get('_scores') or []
        if times and labels and len(scores) == len(times):
            median = sorted(scores)[len(scores) // 2]
            hi_t = [times[i] for i in range(len(times)) if scores[i] >= median]
            hi_e = [labels[i] for i in range(len(times)) if scores[i] >= median]
            lo_t = [times[i] for i in range(len(times)) if scores[i] < median]
            lo_e = [labels[i] for i in range(len(times)) if scores[i] < median]
            t1, s1 = kaplan_meier(hi_t, hi_e)
            t2, s2 = kaplan_meier(lo_t, lo_e)
            ax2.step(t1, s1, where='post', label=f'高风险组 (n={len(hi_t)})',
                     color='#e74c3c', lw=2)
            ax2.step(t2, s2, where='post', label=f'低风险组 (n={len(lo_t)})',
                     color='#2ecc71', lw=2)
            ax2.set_xlabel('随访时间 (天)')
            ax2.set_ylabel('无进展生存概率')
            ax2.set_title('随访生存曲线（按预测风险分层）', fontsize=10)
            ax2.set_ylim(0, 1.05)
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)

        fig.tight_layout()
        canvas.draw_idle()

    # ==================== 渲染：任务 C ====================

    def _update_task_c_ui(self):
        tasks = self._me_cache.get('tasks') or {}
        task_c = tasks.get('C') or {}
        if not tasks.get('available') or not task_c:
            return
        metrics = task_c.get('metrics', {})
        quality = metrics.get('ranking_quality', {})
        tracing = metrics.get('contact_tracing_efficiency')
        lbl = self.me_task_c_labels
        lbl['r_estimate'].configure(
            text=f"繁殖数估计 R: {metrics.get('r_estimate', 0):.2f}")
        tracing_str = f"{tracing:.2%}" if isinstance(tracing, (int, float)) else '--'
        lbl['tracing'].configure(text=f"追踪效率: {tracing_str}")
        lbl['mean_pos'].configure(
            text=f"阳性平均排位: {quality.get('mean_position', '--')}")
        lbl['n_contacts'].configure(
            text=f"接触者: {task_c.get('n_contacts', 0)} | "
                 f"真阳性: {task_c.get('n_true_positives', 0)}")
        self._draw_task_c_chart()

    def _draw_task_c_chart(self):
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'me_task_c_figure'):
            return
        fig, canvas = self.me_task_c_figure, self.me_task_c_canvas
        tasks = self._me_cache.get('tasks') or {}
        task_c = tasks.get('C') or {}
        fig.clear()
        if not tasks.get('available') or not task_c:
            self._me_placeholder(fig, canvas,
                                 tasks.get('reason', '任务 C 评估不可用'))
            return

        metrics = task_c.get('metrics', {})
        hit = metrics.get('hit_rate', {}) or {}
        ax1 = fig.add_subplot(1, 2, 1)
        ks = sorted(hit.keys(), key=lambda k: int(k.split('_')[1]))
        vals = [hit[k] for k in ks]
        xlabels = [f"Top-{k.split('_')[1]}" for k in ks]
        bars = ax1.bar(xlabels, vals, color='#45B7D1')
        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel('命中率')
        ax1.set_title('高风险排序命中率（前 K 名中真阳性占比）', fontsize=10)
        for bar, v in zip(bars, vals):
            ax1.annotate(f"{v:.0%}", (bar.get_x() + bar.get_width() / 2, v),
                         textcoords='offset points', xytext=(0, 3),
                         ha='center', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.axis('off')
        quality = metrics.get('ranking_quality', {})
        tracing = metrics.get('contact_tracing_efficiency')
        lines = [
            "排序质量（真阳性在预测排序中的排位）",
            f"  平均排位：{quality.get('mean_position', '--')}",
            f"  中位排位：{quality.get('median_position', '--')}",
            f"  最好排位：{quality.get('min_position', '--')}",
            f"  最差排位：{quality.get('max_position', '--')}",
            "",
            f"繁殖数估计 R = {metrics.get('r_estimate', 0):.2f}"
            "（二代病例 / 传染源）",
            "",
            "接触者追踪效率：" +
            (f"{tracing:.1%}（被追踪到的感染者占比）"
             if isinstance(tracing, (int, float)) else '--'),
            "",
            "决策含义：决定追踪谁、在哪里干预",
        ]
        ax2.text(0.05, 0.95, "\n".join(lines), transform=ax2.transAxes,
                 va='top', ha='left', fontsize=10, family='sans-serif')
        ax2.set_title('网络传播指标详情', fontsize=10)

        fig.tight_layout()
        canvas.draw_idle()

    # ==================== 渲染：网络贡献消融 ====================

    def _update_ablation_ui(self):
        ab = self._me_cache.get('ablation') or {}
        if not ab.get('available'):
            self.me_ablation_conclusion.configure(
                text=f"消融实验不可用：{ab.get('reason', '未知原因')}")
            return
        delta = ab.get('delta', {})
        contributes = ab.get('network_contributes', False)
        verdict = "✓ 网络层贡献显著" if contributes else "△ 网络层贡献有限"
        self.me_ablation_conclusion.configure(
            text=(f"{verdict} | ΔAUROC = {delta.get('delta_auroc', 0):+.4f}，"
                  f"ΔC-index = {delta.get('delta_cindex', 0):+.4f} | "
                  f"{ab.get('conclusion', '')}"))
        self._draw_ablation_chart()

    def _draw_ablation_chart(self):
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'me_ablation_figure'):
            return
        fig, canvas = self.me_ablation_figure, self.me_ablation_canvas
        ab = self._me_cache.get('ablation') or {}
        fig.clear()
        if not ab.get('available'):
            self._me_placeholder(fig, canvas,
                                 ab.get('reason', '消融实验不可用'))
            return

        base = ab.get('baseline', {})
        full = ab.get('full', {})
        delta = ab.get('delta', {})

        ax = fig.add_subplot(1, 1, 1)
        x = [0, 1]
        width = 0.35
        auroc = [base.get('auroc', 0), full.get('auroc', 0)]
        cindex = [base.get('c_index', 0), full.get('c_index', 0)]
        b1 = ax.bar([i - width / 2 for i in x], auroc, width,
                    label='AUROC', color='#45B7D1')
        b2 = ax.bar([i + width / 2 for i in x], cindex, width,
                    label='C-index', color='#FF6B6B')
        ax.set_xticks(x)
        ax.set_xticklabels(['基线：仅个体层 P_base\n（无 GNN 网络层）',
                            '完整：个体层 + 门控网络增量\n（有 GNN 网络层）'])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('判别性能')
        ax.set_title(
            f"网络贡献消融：ΔAUROC = {delta.get('delta_auroc', 0):+.4f}，"
            f"ΔC-index = {delta.get('delta_cindex', 0):+.4f}",
            fontsize=11, fontweight='bold')
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h),
                            textcoords='offset points', xytext=(0, 3),
                            ha='center', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        n_total = ab.get('n_total', 0)
        n_cases = ab.get('n_cases', 0)
        ax.text(0.99, 0.02,
                f"评估队列：{n_total} 名接触者 / {n_cases} 例确诊"
                f"（患病率 {ab.get('prevalence', 0):.1%}，确定性合成网络）",
                transform=ax.transAxes, ha='right', fontsize=8, color='gray')

        fig.tight_layout()
        canvas.draw_idle()

    # ==================== 渲染：验证与证据 ====================

    def _update_public_data_ui(self):
        pub = self._me_cache.get('public') or {}
        if not pub.get('available'):
            self.me_public_summary.configure(
                text=f"公开数据目录不可用：{pub.get('reason', '未知原因')}")
            return

        summary = pub.get('summary', {})
        strategy = pub.get('strategy', {})
        coverage = summary.get('coverage_by_task', {})

        # 任务可验证性摘要
        lines = []
        for tid in ('A', 'B', 'C'):
            cov = coverage.get(tid, {})
            status = ('可外部验证' if cov.get('external_validatable')
                      else ('仅方法学公开（过渡验证）' if cov.get('methodology_only')
                            else '无公开数据'))
            lines.append(f"任务{tid}：{status}（{cov.get('n', 0)} 个匹配数据集）")
        strat_summary = strategy.get('summary', {})
        for tid in ('A', 'B', 'C'):
            if tid in strat_summary:
                lines.append(f"  · {strat_summary[tid]}")
        self.me_public_summary.configure(text="\n".join(lines))

        # 数据集表格
        tree = self.me_public_tree
        tree.delete(*tree.get_children())
        access_zh = {
            'open': '开放下载', 'application': '申请获取',
            'data_use_agreement': '数据使用协议', 'contact_author': '联系作者',
            'methodology_open': '仅方法学公开',
        }
        layer_zh = {'individual': '个体层', 'contact': '接触者层',
                    'transmission': '传播层'}
        from ...validation.public_datasets import list_public_datasets
        for ds in list_public_datasets():
            tree.insert('', 'end', values=(
                ds.get('id', ''),
                ds.get('name', ''),
                layer_zh.get(ds.get('layer', ''), ds.get('layer', '')),
                '/'.join(ds.get('tasks', [])),
                access_zh.get(ds.get('access', ''), ds.get('access', '')),
                ds.get('size', ''),
                f"P{ds.get('priority', '')}",
            ))

        # 详情文本：合作数据需求 + 过渡验证步骤
        text = self.me_public_text
        text.delete('1.0', 'end')
        text.insert('end', "=== 合作数据需求（按优先级） ===\n", 'heading')
        for need in strategy.get('cooperation_data_needs', []):
            text.insert('end',
                        f"P{need.get('priority')} {need.get('data', '')}\n"
                        f"    为什么：{need.get('why', '')}\n")
        plan_c = strategy.get('plan', {}).get('C', {})
        if not plan_c.get('can_external_validate', True):
            text.insert('end', "\n=== 任务 C 过渡验证步骤 ===\n", 'heading')
            for i, step in enumerate(plan_c.get('transition', {}).get('steps', []), 1):
                text.insert('end', f"{i}. {step}\n")

    # ==================== 渲染：数据与可复现性 ====================

    def _update_dgp_ui(self):
        dgp = self._me_cache.get('dgp') or {}
        if not dgp.get('available'):
            self.me_dgp_summary.configure(
                text=f"DGP 报告不可用：{dgp.get('reason', '未知原因')}")
            return

        gen = dgp.get('generation', {})
        params = dgp.get('dgp_parameters', {})
        validation = dgp.get('validation', {})
        prevalence = validation.get('prevalence', {})
        repro = dgp.get('reproducibility', {})
        overall = dgp.get('overall_pass', False)

        dist_pass = all(v.get('pass', False)
                        for v in validation.get('distributions', []))
        source = params.get('source', '')
        source_str = f" | 数据来源 {source}" if source else ''
        self.me_dgp_summary.configure(text=(
            f"总体校验：{'✓ 通过' if overall else '✗ 未通过'} | "
            f"样本量 {gen.get('n_samples', '--')} | 随机种子 {gen.get('random_seed', '--')} | "
            f"生成器 {gen.get('generator', '--')}\n"
            f"分布对照：{'✓ 全部通过' if dist_pass else '✗ 存在偏差'} | "
            f"检出率 {prevalence.get('observed_per_100k', '--')}/10万 "
            f"（目标 {prevalence.get('target_per_100k', '--')} ± "
            f"{prevalence.get('tolerance_per_100k', '--')}） | "
            f"可复现性：{'✓ 同种子两次生成完全一致' if repro.get('identical') else '✗ 不一致'}"
            f"{source_str}"))

        # 字段分布校验表
        tree = self.me_dgp_tree
        tree.delete(*tree.get_children())
        for v in validation.get('distributions', []):
            delta = v.get('max_abs_delta')
            if delta is None:
                checks = v.get('checks', [])
                mean_check = next((c for c in checks if c.get('check') == 'mean'), None)
                delta = mean_check.get('relative_error') if mean_check else '--'
            tree.insert('', 'end', values=(
                v.get('field', ''),
                '分类' if v.get('kind') == 'categorical' else '连续',
                v.get('distribution', ''),
                f"{delta}" if isinstance(delta, str) else f"{delta:.4f}",
                f"{v.get('tolerance', '--')}",
                '✓' if v.get('pass') else '✗',
            ))

        # 文献对照文本
        text = self.me_dgp_text
        text.delete('1.0', 'end')
        text.insert('end', "=== 文献参考量对照 ===\n", 'heading')
        lit = validation.get('literature', {})
        for entry in lit.get('entries', []):
            status = {'pass': '✓ 通过', 'fail': '✗ 未通过',
                      'not_applicable': '— 不适用'}.get(
                          entry.get('status'), entry.get('status', ''))
            observed = entry.get('observed')
            obs_str = f" | 观测 {observed}" if observed else ''
            text.insert('end',
                        f"[{status}] {entry.get('quantity', '')}："
                        f"目标 {entry.get('target', '')}{obs_str}\n"
                        f"    来源：{entry.get('source', '')}\n")
        text.insert('end', "\n=== 传播链标签生成机制（TRANSMISSION_SPEC） ===\n",
                    'heading')
        tspec = dgp.get('transmission_spec', {})
        for k, v in tspec.items():
            text.insert('end', f"{k}: {v}\n")

    # ==================== 渲染：社区干预反事实 ====================

    def _update_seir_ui(self):
        seir = self._me_cache.get('seir') or {}
        if not seir.get('available'):
            self.me_seir_summary.configure(
                text=f"SEIR 干预模拟不可用：{seir.get('reason', '未知原因')}")
            return
        summary = seir.get('summary', {})
        self.me_seir_summary.configure(text=(
            f"{summary.get('recommendation', '')}\n"
            f"模拟设置：人口 {seir.get('population', '--')} | 水平 "
            f"{seir.get('t_horizon_days', '--')} 天 | 覆盖率 "
            f"{seir.get('coverage', 0):.0%} | 高风险个体 "
            f"{seir.get('n_high_risk', 0)} 人"))
        self._draw_seir_intervention()

    def _draw_seir_intervention(self):
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'me_seir_figure'):
            return
        fig, canvas = self.me_seir_figure, self.me_seir_canvas
        seir = self._me_cache.get('seir') or {}
        fig.clear()
        if not seir.get('available'):
            self._me_placeholder(fig, canvas,
                                 seir.get('reason', 'SEIR 干预模拟不可用'))
            return

        baseline = seir.get('baseline', {})
        times = baseline.get('times', [])
        t_months = [t / 30.44 for t in times]

        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t_months, baseline.get('cases_curve', []),
                label='无干预（基线）', color='#2c3e50', lw=2.5, ls='--')
        palette = ['#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']
        for i, (key, strat) in enumerate(seir.get('strategies', {}).items()):
            color = palette[i % len(palette)]
            averted = strat.get('averted_percent', 0)
            ax.plot(t_months, strat.get('cases_curve', []),
                    label=f"{strat.get('label', key)}（可避免 {averted:.1f}%）",
                    color=color, lw=1.8)
        ax.set_xlabel('时间（月）')
        ax.set_ylabel('活动性病例数')
        ax.set_title('社区级 SEIR 反事实模拟：干预 vs 无干预',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)

        best = seir.get('summary', {}).get('best_strategy_label')
        if best:
            ax.annotate(f"推荐策略：{best}", (0.02, 0.96),
                        xycoords='axes fraction', fontsize=9,
                        color='#0f766e', fontweight='bold', va='top')

        fig.tight_layout()
        canvas.draw_idle()

    # ==================== 渲染：百万级验证（SINAN） ====================

    def _update_scale_validation_ui(self):
        sinan = self._me_cache.get('sinan') or {}
        if not sinan.get('available'):
            self.me_scale_summary.configure(
                text=f"SINAN 规模验证不可用：{sinan.get('reason', '未知原因')}")
            return
        self.me_scale_summary.configure(text=sinan.get('narrative', ''))
        self._draw_scale_robustness()

        # 详情文本：亚组塌方 + 特征重要性 + 对照集警示
        text = self.me_scale_detail
        text.delete('1.0', 'end')
        text.insert('end', "亚组（T1 far 测试集，最优树模型）——残余误差集中区：\n")
        for key in ('age_15_49', 'age_50_64', 'age_65p', 'aids', 'no_aids'):
            sg = sinan.get('subgroups', {}).get(key)
            if sg:
                text.insert(
                    'end', f"  {key}: AUROC {sg.get('auroc', 0):.3f}"
                           f"（n={sg.get('n', 0)}，事件 {sg.get('events', 0)}）\n")
        text.insert('end', "\nXGB gain 特征重要性 Top5：\n  ")
        text.insert('end', '、'.join(sinan.get('top_features', [])) + '\n')
        ep = sinan.get('endpoint_correction')
        if ep:
            text.insert(
                'end', "\n终点语义更正（round-7 裁定）：本归档旧终点「{old}」"
                       "实为「{ep4}」；修正终点 = {ace}。\n  {rule}\n".format(
                           old=ep.get('old_label', ''),
                           ep4=ep.get('code4_semantics', ''),
                           ace=ep.get('corrected_endpoint', ''),
                           rule=ep.get('citation_rule', '')))
        text.insert(
            'end', "\n对照集口径：巴基斯坦儿科 AUROC 0.96-0.98 属"
                   "诊断流程饱和（终点泄漏，与 PACTS 同构），"
                   "仅作方向对照（{0}）。\n".format(
                       sinan.get('pakistan_interpretation',
                                 'directional_reference_only')))

    def _draw_scale_robustness(self):
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'me_scale_figure'):
            return
        fig, canvas = self.me_scale_figure, self.me_scale_canvas
        sinan = self._me_cache.get('sinan') or {}
        fig.clear()
        if not sinan.get('available'):
            self._me_placeholder(fig, canvas,
                                 sinan.get('reason', 'SINAN 规模验证不可用'))
            return

        ax = fig.add_subplot(1, 1, 1)
        x = [0, 1, 2]
        xlabels = ['验证 13-15', '近端 16-17', '远端 18-19']
        palette = ['#2c3e50', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']
        best = sinan.get('best_model')
        for i, (model, series) in enumerate(
                sorted(sinan.get('t1_series', {}).items())):
            if any(v is None for v in series):
                continue
            lw = 3.0 if model == best else 1.6
            color = palette[0] if model == best else palette[1 + i % 4]
            ax.plot(x, series, marker='o', lw=lw, color=color,
                    label=f"{model}（far {series[2]:.4f}）")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_ylabel('AUROC')
        ax.set_ylim(0.80, 0.87)
        ax.set_title('T1 时间外推稳健性：训练 2001-2012，6 年外推'
                     ' AUROC 仅衰减 %.3f（%s）'
                     % (sinan.get('decay_val_to_far', 0), best),
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(alpha=0.3)

        t2b = sinan.get('t2_series', {}).get(sinan.get('t2_best_model'))
        if t2b:
            ax.annotate('T2 空间外推：%s %.4f（非东南训练→东南测试）'
                        % (sinan.get('t2_best_model'), t2b),
                        (0.02, 0.96), xycoords='axes fraction', fontsize=9,
                        color='#0f766e', fontweight='bold', va='top')

        fig.tight_layout()
        canvas.draw_idle()

    # ==================== Tab 9: 时序家庭筛查（第 2 层部署演示） ====================

    def _build_temporal_household_tab(self, nb):
        """第 2 层网络增强层的部署语义动态演示：户内第一人 → 风险更新 → 重排。"""
        tab = ttk.Frame(nb)
        nb.add(tab, text="时序家庭筛查")

        header = ttk.LabelFrame(
            tab, text="时序家庭先验（第 2 层网络增强层的部署实现，2026-08-25 第四轮 P1）")
        header.pack(fill='x', padx=10, pady=(8, 4))
        ttk.Label(
            header, justify='left', wraplength=980, foreground='gray',
            text="部署语义：户内已筛查成员的结果作为输入——同户有人筛查阳性时，"
                 "其余成员风险按贝叶斯收缩户级率上调（logit 空间）并自动重排；"
                 "首筛查者无先证信息，回退个体基线。已确诊成员（家庭成员页双击标记）"
                 "作为「已筛查阳性」预置证据。"
                 "验证依据：HomeACF 20 种子 exposure+prior:RF 0.714 vs ind:RF 0.643"
                 "（+0.058 CI 显著）——网络层构造首次在真实数据上取得显著正增益。"
        ).pack(fill='x', padx=8, pady=4)

        # 控制栏：筛查下一人 / 重置 + 信息行
        ctrl = ttk.Frame(tab)
        ctrl.pack(fill='x', padx=10, pady=4)
        ttk.Button(ctrl, text="筛查下一人",
                   command=self._me_temporal_screen_next).pack(side='left', padx=2)
        ttk.Button(ctrl, text="重置（读取最新成员）",
                   command=self._me_temporal_reset).pack(side='left', padx=2)
        self.me_temporal_info = ttk.Label(
            ctrl, text="请先在「家庭成员信息」标签页添加成员，或运行风险评估",
            foreground='gray')
        self.me_temporal_info.pack(side='left', padx=15)

        if MATPLOTLIB_AVAILABLE:
            chart_frame = ttk.Frame(tab)
            chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
            self.me_temporal_figure = Figure(
                figsize=self._get_chart_figsize(11, 6), dpi=100)
            self.me_temporal_canvas = FigureCanvasTkAgg(
                self.me_temporal_figure, master=chart_frame)
            self.me_temporal_canvas.get_tk_widget().pack(fill='both', expand=True)
            toolbar = NavigationToolbar2Tk(self.me_temporal_canvas, chart_frame)
            toolbar.update()
            self._bind_configure_redraw(
                self.me_temporal_canvas.get_tk_widget(),
                self._draw_temporal_household_chart, 'me_temporal',
                figure=self.me_temporal_figure)
            self._me_placeholder(self.me_temporal_figure,
                                 self.me_temporal_canvas,
                                 "请先在「家庭成员信息」标签页添加成员\n"
                                 "（或运行风险评估以使用 ML 个体基线）")
        else:
            self.me_temporal_text = self._me_make_text(tab)

    # ---------- 状态管理 ----------

    def _me_temporal_p_base_map(self):
        """从三层递进缓存提取 {家庭成员名: ML 个体基线概率}（0-100）。"""
        series = self._me_cache.get('three_layer') or {}
        if not series.get('available'):
            return None
        pb = {}
        for c in series.get('contacts', []):
            if c.get('contact_type') == 'family' and c.get('p_base') is not None:
                pb[c['name']] = c['p_base']
        return pb or None

    def _me_temporal_state(self, force=False):
        """获取（并在数据源变化时重建）序贯筛查状态。"""
        if not hasattr(self, '_me_cache'):
            self._me_cache = {}
        entries = list(getattr(self, 'family_entries', None) or [])
        pb = self._me_temporal_p_base_map()
        signature = _evidence.household_screening_signature(entries, pb)
        if (not force and getattr(self, '_me_temporal_signature', None) == signature
                and getattr(self, '_me_temporal_state_obj', None) is not None):
            return self._me_temporal_state_obj, self._me_temporal_demo
        demo = _evidence.build_household_screening_demo(
            entries, p_base_map=pb,
            risk_score_fn=getattr(self, '_compute_risk_score', None))
        self._me_temporal_signature = signature
        self._me_temporal_demo = demo
        self._me_temporal_state_obj = build_screening_state(demo)
        return self._me_temporal_state_obj, demo

    def _me_temporal_reset(self):
        """强制重读最新家庭成员并重建序贯状态。"""
        try:
            self._me_temporal_state(force=True)
            self._draw_temporal_household_chart()
        except Exception as e:
            LOGGER.debug("时序家庭筛查重置失败（非致命）: %s", e, exc_info=True)

    def _me_temporal_screen_next(self):
        """筛查当前更新后风险最高的未筛成员（弹窗询问结果，揭示后重排）。"""
        try:
            state, demo = self._me_temporal_state()
            if state is None:
                if messagebox is not None:
                    messagebox.showinfo(
                        "时序家庭筛查",
                        (demo or {}).get('reason', '请先添加家庭成员'))
                return
            idx = state.next_to_screen()
            if idx is None:
                self._me_temporal_update_info(state, demo)
                if messagebox is not None:
                    messagebox.showinfo(
                        "时序家庭筛查",
                        f"全部 {len(state.base_scores)} 名成员已筛查完成，"
                        f"累计阳性 {sum(r or 0 for r in state.results)} 例")
                return
            name = state.names[idx]
            if messagebox is None:
                return
            positive = messagebox.askyesno(
                "筛查结果", f"成员「{name}」的筛查结果是否为阳性？\n\n"
                           f"（选择「是」= 阳性，其余成员风险将上调并重排）")
            state.record_result(idx, 1 if positive else 0)
            self._draw_temporal_household_chart()
        except Exception as e:
            LOGGER.debug("时序筛查下一步失败（非致命）: %s", e, exc_info=True)

    # ---------- 渲染 ----------

    def _me_temporal_update_info(self, state, demo):
        """更新信息行（下一步建议 / 已筛进度 / 累计阳性）。"""
        if not hasattr(self, 'me_temporal_info'):
            return
        if state is None:
            reason = (demo or {}).get('reason', '请先添加家庭成员')
            self.me_temporal_info.configure(text=reason, foreground='gray')
            return
        n = len(state.base_scores)
        k = sum(1 for f in state.screened_flags if f)
        found = sum(1 for r in state.results if r)
        idx = state.next_to_screen()
        src = {'ml': 'ML 个体基线', 'entry_risk': '接触暴露评分',
               'mixed': 'ML+暴露混合'}.get((demo or {}).get('source'), '')
        if idx is None:
            nxt = f"筛查完成：累计阳性 {found} 例"
        else:
            scores = state.current_scores()
            nxt = (f"下一步建议筛查：{state.names[idx]}"
                   f"（更新后风险 {scores[idx]:.1f}%）")
        self.me_temporal_info.configure(
            text=f"{nxt} | 已筛 {k}/{n} | 累计阳性 {found} | "
                 f"基线率 π={state.base_rate:.0%} | 基线来源：{src}",
            foreground='')

    def _draw_temporal_household_chart(self):
        """左：按更新后风险重排的成员条形图；右：序贯检出曲线。"""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'me_temporal_figure'):
            return
        fig, canvas = self.me_temporal_figure, self.me_temporal_canvas
        try:
            state, demo = self._me_temporal_state()
        except Exception as e:
            LOGGER.debug("时序筛查状态获取失败（非致命）: %s", e)
            state, demo = None, None
        fig.clear()
        if state is None:
            reason = (demo or {}).get(
                'reason', '请先在「家庭成员信息」标签页添加成员')
            self._me_placeholder(fig, canvas, reason)
            self._me_temporal_update_info(None, demo)
            return

        scores = state.current_scores()
        n = len(scores)
        colors = getattr(self, 'COLORS', {})
        c_unscreened = colors.get('chart_series_3', '#45B7D1')
        c_pos = '#e74c3c'
        c_neg = '#2ecc71'

        # 左：重排视图（风险最高在顶部）
        ax = fig.add_subplot(1, 2, 1)
        order = sorted(range(n), key=lambda i: scores[i])   # 升序 → 顶部最高
        labels, vals, bar_colors = [], [], []
        for i in order:
            if state.screened_flags[i]:
                mark = '●阳' if state.results[i] else '○阴'
            else:
                mark = '·未筛'
            labels.append(f"{state.names[i]} {mark}")
            vals.append(scores[i])
            bar_colors.append(
                c_pos if (state.screened_flags[i] and state.results[i])
                else (c_neg if state.screened_flags[i] else c_unscreened))
        y = range(n)
        ax.barh(y, vals, color=bar_colors, height=0.62)
        ax.scatter([state.base_scores[i] for i in order], y,
                   marker='D', s=26, color='#333333', zorder=3,
                   label='个体基线（第1层）')
        for yi, v in zip(y, vals):
            ax.text(v + 0.8, yi, f"{v:.1f}", va='center', fontsize=8)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(0, 105)
        ax.set_xlabel('风险（%）')
        src = {'ml': 'ML 个体基线', 'entry_risk': '接触暴露评分',
               'mixed': 'ML+暴露混合'}.get(demo.get('source'), '')
        ax.set_title(f"更新后风险排序（户间重排）— 基线：{src}",
                     fontsize=10, fontweight='bold')
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(color=c_unscreened, label='未筛查'),
            Patch(color=c_pos, label='已筛·阳性'),
            Patch(color=c_neg, label='已筛·阴性'),
        ]
        if ax.collections:
            legend_handles.append(ax.collections[0])   # 个体基线散点
        ax.legend(handles=legend_handles, fontsize=8, loc='lower right')
        ax.grid(axis='x', alpha=0.3)

        # 右：序贯检出曲线（含预置确诊）
        ax2 = fig.add_subplot(1, 2, 2)
        steps = [rec['step'] for rec in state.history]
        cum = []
        running = 0
        for rec in state.history:
            running += int(rec['result'])
            cum.append(running)
        if steps:
            ax2.step([0] + steps, [0] + cum, where='post',
                     color=colors.get('accent', '#0f766e'), lw=2)
            ax2.plot(steps, cum, 'o', color=colors.get('accent', '#0f766e'),
                     ms=4)
            nns = len(steps) / max(running, 1)
            ax2.annotate(
                f"已筛 {len(steps)} 人 | 检出 {running} 例 | NNS≈{nns:.1f}",
                (0.03, 0.95), xycoords='axes fraction', fontsize=9,
                va='top', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, "尚未开始筛查\n（无已确诊成员 → 首筛查者回退个体基线）",
                     transform=ax2.transAxes, ha='center', va='center',
                     fontsize=11, color='gray')
        ax2.set_xlabel('筛查序号')
        ax2.set_ylabel('累计阳性检出')
        ax2.set_xlim(0, max(n, 1))
        ax2.set_ylim(0, max(running, 1) + 0.8)
        ax2.set_title("序贯筛查检出曲线（先证信息 → 提前检出）",
                      fontsize=10, fontweight='bold')
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        canvas.draw_idle()
        self._me_temporal_update_info(state, demo)

    # ==================== 通用辅助 ====================

    def _redraw_model_evidence_charts(self):
        """重绘全部模型证据图表（主题切换 / 评估完成后调用）。"""
        for draw_func in (
                getattr(self, '_draw_three_layer_waterfall', None),
                getattr(self, '_draw_task_a_chart', None),
                getattr(self, '_draw_task_b_charts', None),
                getattr(self, '_draw_task_c_chart', None),
                getattr(self, '_draw_ablation_chart', None),
                getattr(self, '_draw_seir_intervention', None),
                getattr(self, '_draw_temporal_household_chart', None)):
            if draw_func is None:
                continue
            try:
                draw_func()
            except Exception as e:
                LOGGER.debug("模型证据图表重绘失败（非致命）: %s", e)

    def _me_make_text(self, parent, height=8):
        """创建注册到主题系统的 ScrolledText。parent 为 None 时延迟 pack。"""
        from tkinter.scrolledtext import ScrolledText
        if parent is None:
            parent = ttk.Frame()
        text = ScrolledText(parent, wrap='word', height=height,
                            font=('Microsoft YaHei', 9))
        text.pack(fill='both', expand=True, padx=2, pady=2)
        self._register_text_widget(text)
        return text

    def _me_placeholder(self, figure, canvas, message):
        """在图表上绘制占位提示。"""
        try:
            figure.clear()
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, message, transform=ax.transAxes,
                    ha='center', va='center', fontsize=13, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw_idle()
        except Exception:
            LOGGER.debug("模型证据占位绘制失败（非致命）", exc_info=True)


__all__ = ['ModelEvidenceMixin', 'task_a_metrics_at_threshold', 'kaplan_meier',
           'build_screening_state']
