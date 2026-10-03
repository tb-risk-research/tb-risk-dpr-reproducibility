#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 向导布局子 Mixin

包含原 WizardMixin 中的界面布局方法：
  - 向导界面初始化
  - 主布局创建
  - 步骤进度条创建与绘制
  - 向导步骤创建
  - 导航按钮创建
  - 概览面板创建
"""

from .._gui_common import tk, ttk


class WizardLayoutMixin:
    """GUI 向导布局方法混合类"""

    def init_gui_wizard(self):
        """初始化改进后的向导式界面"""
        self.ui_mode = self.UI_MODE_WIZARD
        # 优先级六：创建 WizardAdapter 实例
        from ..ui_mode_adapter import create_adapter
        self.adapter = create_adapter(self.ui_mode, self)
        self.root = tk.Tk()
        self.root.title("结核病传播风险评估系统 (优化版)")
        self.root.geometry("1200x800")
        self.root.minsize(1024, 768)
        self.root.resizable(True, True)

        self._start_progress_polling()

        self._setup_styles()
        self._setup_keyboard_shortcuts()

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.basic_info_vars = {}

        self._create_menu()

        self._create_wizard_layout()

        self.family_entries = []
        self.social_entries = []

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)


    def _create_wizard_layout(self):
        """创建向导式界面布局"""
        # 主框架：左侧向导区 + 右侧概览面板
        main_paned = ttk.PanedWindow(self.root, orient='horizontal')
        main_paned.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        # === 左侧：向导区域 ===
        wizard_frame = ttk.Frame(main_paned)
        wizard_frame.grid_rowconfigure(0, weight=0)  # 顶部进度条
        wizard_frame.grid_rowconfigure(1, weight=1)  # 内容区域
        wizard_frame.grid_rowconfigure(2, weight=0)  # 底部导航按钮
        wizard_frame.grid_columnconfigure(0, weight=1)
        main_paned.add(wizard_frame, weight=3)

        # 1. 顶部：步骤进度条
        self.current_wizard_step = 0
        self._create_step_progress(wizard_frame, row=0)

        # 2. 中间：内容区域（使用Notebook模拟步骤切换）
        self.wizard_notebook = ttk.Notebook(wizard_frame)
        self.wizard_notebook.grid(row=1, column=0, sticky='nsew', pady=10)
        self.wizard_notebook.bind('<<NotebookTabChanged>>', self._on_wizard_step_change)

        # 创建4个步骤标签页
        self._create_wizard_steps()

        # 3. 底部：导航按钮
        self._create_wizard_navigation(wizard_frame, row=2)

        # === 右侧：概览面板 ===
        self._create_overview_panel(main_paned)

        # 初始步骤为0
        self.current_wizard_step = 0
        self._update_wizard_navigation()


    def _create_step_progress(self, parent, row):
        """创建Canvas步骤进度条（圆形数字 + 连接线）"""
        self.step_titles = ["患者临床特征", "核心接触者",
                            "扩展接触者", "评估结果"]
        self.step_descriptions = [
            "填写患者年龄、痰检、\n临床症状等基本信息",
            "添加家庭成员及\n核心密切接触者",
            "添加社会接触者及\n扩展关系网络",
            "查看风险评估结果\n与临床建议",
        ]

        progress_frame = ttk.Frame(parent)
        progress_frame.grid(row=row, column=0, sticky='ew', pady=(5, 0))

        canvas_height = 58
        self.step_canvas = tk.Canvas(progress_frame, height=canvas_height,
                                      bg=self.COLORS['bg_light'],
                                      highlightthickness=0)
        self.step_canvas.pack(fill='x')

        self.step_canvas_items = []

        self._draw_step_progress()

        self.step_desc_label = ttk.Label(progress_frame, text="",
                                          font=self._FONT_SMALL,
                                          foreground=self.COLORS['text_light'],
                                          justify='center')
        self.step_desc_label.pack(fill='x', pady=(2, 4))


    def _draw_step_progress(self):
        """绘制步骤进度条"""
        canvas = self.step_canvas
        canvas.delete('all')
        self.step_canvas_items = []

        w = canvas.winfo_width()
        if w < 100:
            w = 800

        n_steps = len(self.step_titles)
        padding = 50
        usable = w - 2 * padding
        spacing = usable / max(n_steps - 1, 1)

        circle_r = 13
        y_center = 28

        for i in range(n_steps):
            cx = int(padding + i * spacing)
            if i <= self.current_wizard_step:
                fill_color = self.COLORS['accent'] if i == self.current_wizard_step else self.COLORS['success']
                text_color = 'white'
                text_str = '✓' if i < self.current_wizard_step else str(i + 1)
            else:
                fill_color = self.COLORS['border']
                text_color = self.COLORS['text_light']
                text_str = str(i + 1)

            circle = canvas.create_oval(cx - circle_r, y_center - circle_r,
                                         cx + circle_r, y_center + circle_r,
                                         fill=fill_color, outline='',
                                         tags=(f'step_{i}',))
            text_item = canvas.create_text(cx, y_center, text=text_str,
                                            fill=text_color,
                                            font=('Microsoft YaHei', 9, 'bold'),
                                            tags=(f'step_{i}',))

            title_y = y_center + circle_r + 10
            title_color = self.COLORS['primary'] if i == self.current_wizard_step else self.COLORS['text_light']
            title_item = canvas.create_text(cx, title_y,
                                             text=self.step_titles[i],
                                             fill=title_color,
                                             font=('Microsoft YaHei', 9,
                                                   'bold' if i == self.current_wizard_step else 'normal'),
                                             tags=(f'step_{i}',))

            self.step_canvas_items.append((circle, text_item, title_item))

            if i < n_steps - 1:
                next_cx = int(padding + (i + 1) * spacing)
                line_color = self.COLORS['success'] if i < self.current_wizard_step else self.COLORS['border']
                line_thickness = 3 if i < self.current_wizard_step else 1
                canvas.create_line(cx + circle_r + 4, y_center,
                                    next_cx - circle_r - 4, y_center,
                                    fill=line_color, width=line_thickness,
                                    tags=('line',))

        canvas.bind('<Configure>', lambda e: self._draw_step_progress())


    def _create_wizard_steps(self):
        """创建向导的4个步骤"""
        self.tab1_patient = ttk.Frame(self.wizard_notebook)
        self.wizard_notebook.add(self.tab1_patient, text='')
        self._init_improved_basic_info_tab(self.tab1_patient)

        self.tab2_family = ttk.Frame(self.wizard_notebook)
        self.wizard_notebook.add(self.tab2_family, text='')
        self._init_improved_family_tab(self.tab2_family)

        self.tab3_social = ttk.Frame(self.wizard_notebook)
        self.wizard_notebook.add(self.tab3_social, text='')
        self._init_improved_social_tab(self.tab3_social)

        self.tab4_result = ttk.Frame(self.wizard_notebook)
        self.wizard_notebook.add(self.tab4_result, text='')
        self._init_dashboard_result_tab(self.tab4_result)


    def _create_wizard_navigation(self, parent, row):
        """创建向导导航按钮"""
        nav_frame = ttk.Frame(parent)
        nav_frame.grid(row=row, column=0, sticky='ew', pady=10)

        # 左对齐按钮
        self.btn_prev = ttk.Button(nav_frame, text="<< 上一步",
                                  command=self._wizard_prev, state='disabled')
        self.btn_prev.pack(side='left', padx=5)

        # P2 修复溢出：弹性占位 Frame 分离左组与中间按钮，避免窄窗口碰撞
        left_spacer = ttk.Frame(nav_frame)
        left_spacer.pack(side='left', fill='x', expand=True)

        # 中间执行评估按钮（padx 从 20 减小为 8）
        self.btn_assess = ttk.Button(nav_frame, text="▶ 执行风险评估",
                                    command=self._run_wizard_assessment)
        self.btn_assess.pack(side='left', padx=8)

        # 右侧弹性占位
        right_spacer = ttk.Frame(nav_frame)
        right_spacer.pack(side='left', fill='x', expand=True)

        # 右对齐按钮
        self.btn_next = ttk.Button(nav_frame, text="下一步 >>",
                                  command=self._wizard_next)
        self.btn_next.pack(side='right', padx=5)

        # 跳过按钮
        self.btn_skip = ttk.Button(nav_frame, text="跳过此步",
                                  command=self._wizard_next)
        self.btn_skip.pack(side='right', padx=5)


    def _create_overview_panel(self, parent_paned):
        """创建嵌入式概览面板"""
        overview_frame = ttk.LabelFrame(parent_paned, text="📊 数据概览", padding=10)
        overview_frame.grid_rowconfigure(0, weight=0)
        overview_frame.grid_rowconfigure(1, weight=1)
        overview_frame.grid_columnconfigure(0, weight=1)
        parent_paned.add(overview_frame, weight=1)

        # 统计数据卡片
        stats_frame = ttk.Frame(overview_frame)
        stats_frame.grid(row=0, column=0, sticky='ew', pady=5)

        self.stats_family_label = ttk.Label(stats_frame, text="👨‍👩‍👧‍👦 家庭成员: 0人",
                                           font=('Microsoft YaHei', 10))
        self.stats_family_label.grid(row=0, column=0, sticky='w', pady=2)

        self.stats_social_label = ttk.Label(stats_frame, text="👥 社会接触者: 0人",
                                           font=('Microsoft YaHei', 10))
        self.stats_social_label.grid(row=1, column=0, sticky='w', pady=2)

        # 分隔符
        ttk.Separator(overview_frame, orient='horizontal').grid(row=1, column=0,
                                                               sticky='ew', pady=10)

        # 快速操作按钮
        actions_frame = ttk.LabelFrame(overview_frame, text="快速操作", padding=5)
        actions_frame.grid(row=2, column=0, sticky='ew', pady=5)

        ttk.Button(actions_frame, text="添加家庭成员",
                  command=lambda: self._go_to_wizard_step(1)).pack(fill='x', pady=2)
        ttk.Button(actions_frame, text="添加社会接触者",
                  command=lambda: self._go_to_wizard_step(2)).pack(fill='x', pady=2)

        ttk.Separator(overview_frame, orient='horizontal').grid(row=3, column=0,
                                                               sticky='ew', pady=10)

        # 完成度指示器
        ttk.Label(overview_frame, text="进度：", font=('Microsoft YaHei', 10, 'bold')).grid(row=4,
                                                                                 column=0, sticky='w')
        self.completion_progress = ttk.Progressbar(overview_frame, orient='horizontal',
                                                  mode='determinate', length=200)
        self.completion_progress.grid(row=5, column=0, sticky='ew', pady=5)

        self._update_overview_panel()


    # ==================== 向导导航 ====================

    def _wizard_next(self):
        """跳转到下一步"""
        if self.current_wizard_step < len(self.step_titles) - 1:
            self._go_to_wizard_step(self.current_wizard_step + 1)

    def _wizard_prev(self):
        """返回上一步"""
        if self.current_wizard_step > 0:
            self._go_to_wizard_step(self.current_wizard_step - 1)

    def _go_to_wizard_step(self, step):
        """跳转到指定步骤"""
        if 0 <= step < len(self.step_titles):
            self.wizard_notebook.select(step)
            self.current_wizard_step = step
            self._draw_step_progress()
            self._update_wizard_navigation()

    def _on_wizard_step_change(self, event=None):
        """Notebook 标签页切换事件处理"""
        try:
            current = self.wizard_notebook.index(self.wizard_notebook.select())
        except tk.TclError:
            return
        if current != self.current_wizard_step:
            self.current_wizard_step = current
            self._draw_step_progress()
            self._update_wizard_navigation()

    def _update_wizard_navigation(self):
        """更新导航按钮的启用/禁用状态"""
        if not hasattr(self, 'btn_prev'):
            return
        n_steps = len(self.step_titles)
        # 上一步按钮：第一步时禁用
        self.btn_prev.configure(state='normal' if self.current_wizard_step > 0 else 'disabled')
        # 下一步按钮：最后一步时禁用
        self.btn_next.configure(state='normal' if self.current_wizard_step < n_steps - 1 else 'disabled')
        # 跳过按钮：最后一步时隐藏（btn_skip 由 pack 管理，故用 pack 保持一致）
        if hasattr(self, 'btn_skip'):
            if self.current_wizard_step < n_steps - 1:
                self.btn_skip.pack(side='right', padx=5)
            else:
                self.btn_skip.pack_forget()
        # 评估按钮：仅在最后一步前可见
        if hasattr(self, 'btn_assess'):
            if self.current_wizard_step == n_steps - 1:
                self.btn_assess.configure(text="🔄 重新评估")
            else:
                self.btn_assess.configure(text="▶ 执行风险评估")

    # ==================== 概览面板更新 ====================

    def _update_overview_panel(self):
        """更新概览面板的统计数据和进度条"""
        if not hasattr(self, 'stats_family_label'):
            return

        family_count = len(getattr(self, 'family_entries', []))
        social_count = len(getattr(self, 'social_entries', []))

        self.stats_family_label.configure(text=f"👨‍👩‍👧‍👦 家庭成员: {family_count}人")
        self.stats_social_label.configure(text=f"👥 社会接触者: {social_count}人")

        # 完成度：基于已填写的数据量估算
        completion = 0
        if hasattr(self, 'basic_info_vars') and self.basic_info_vars:
            # 基本信息已初始化，计 30%
            filled = sum(1 for v in self.basic_info_vars.values()
                        if hasattr(v, 'get') and str(v.get()).strip())
            total = max(len(self.basic_info_vars), 1)
            completion += int(30 * filled / total)
        if family_count > 0:
            completion += min(35, family_count * 7)
        if social_count > 0:
            completion += min(35, social_count * 7)
        completion = min(completion, 100)
        self.completion_progress['value'] = completion

    # ==================== 评估流程 ====================

    def _run_wizard_assessment(self):
        """执行向导模式风险评估

        委托给通用 _run_assessment（OrchestratorMixin），后者已实现
        后台线程 + 进度回传 + 模式分派（_display_results_wizard / _draw_wizard_charts）。
        """
        # 跳转到结果步骤
        self._go_to_wizard_step(len(self.step_titles) - 1)
        # 调用通用评估流程
        self._run_assessment()

    def _finish_wizard_assessment(self):
        """评估完成后的向导界面收尾工作

        由 _update_gui_after_assessment 在 wizard 模式下调用，
        确保结果步骤处于激活状态并刷新所有 Treeview。
        """
        # 确保停留在结果步骤
        if hasattr(self, 'wizard_notebook'):
            self._go_to_wizard_step(len(self.step_titles) - 1)
        # 刷新 Treeview 显示最新评估结果
        if hasattr(self, '_refresh_wizard_trees'):
            self._refresh_wizard_trees()
        # 更新概览面板
        if hasattr(self, '_update_overview_panel'):
            self._update_overview_panel()
