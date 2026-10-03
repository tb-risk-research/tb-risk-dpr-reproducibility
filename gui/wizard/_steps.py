#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 向导步骤内容子 Mixin

包含原 WizardMixin 中的各步骤标签页初始化方法：
  - 患者基本信息标签页
  - 带标签输入字段辅助函数
  - 家庭成员标签页
  - 社会接触者标签页
  - 仪表盘结果标签页
  - 信息卡片创建辅助函数
"""

from .._gui_common import tk, ttk, KARAMAY_LOCALIZER_AVAILABLE


class WizardStepsMixin:
    """GUI 向导步骤内容方法混合类"""

    def _init_improved_basic_info_tab(self, parent):
        """初始化改进后的患者基本信息标签页（视觉分组）"""
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        main_frame = ttk.Frame(parent, padding=10)
        main_frame.grid(row=0, column=0, sticky='nsew')
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        # === 说明文字区域 ===
        info_frame = ttk.LabelFrame(main_frame, text="📖 填写说明", padding=10)
        info_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=5)
        info_text = "• 年龄 <5岁 或 >65岁 风险较高\n• 涂阳患者传染性显著高于涂阴\n• 延迟就诊时间越长，传播风险越高\n• 家庭通风条件影响所有接触者的默认值\n• 更多帮助请将鼠标悬停在输入框上"
        ttk.Label(info_frame, text=info_text, justify='left', font=('Microsoft YaHei', 9)).pack(anchor='w')

        # === 场景选择区域 ===
        scenario_frame = ttk.LabelFrame(main_frame, text="🔧 智能场景选择", padding=10)
        scenario_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=5)

        ttk.Label(scenario_frame, text="选择预设场景：", font=('Microsoft YaHei', 10)).grid(row=0, column=0, sticky='w', padx=5)
        self.scenario_var = tk.StringVar(value="custom")
        scenario_options = ["custom", "rural_family", "urban_workplace", "school"]
        if KARAMAY_LOCALIZER_AVAILABLE:
            scenario_options.append("karamay_oilfield")
        scenario_combo = ttk.Combobox(scenario_frame, textvariable=self.scenario_var,
                                     values=scenario_options,
                                     width=18, state="readonly")
        scenario_combo.grid(row=0, column=1, padx=5)
        scenario_combo.bind('<<ComboboxSelected>>', self._on_scenario_change)
        self._add_tooltip(scenario_combo, "选择预设场景可以自动填充相关参数，简化数据输入。")

        self.scenario_desc_var = tk.StringVar(value="自定义模式：手动输入所有参数")
        ttk.Label(scenario_frame, textvariable=self.scenario_desc_var,
                 font=('Microsoft YaHei', 9), foreground=self.COLORS.get('accent', 'blue')).grid(row=0, column=2, sticky='w', padx=10)

        ttk.Button(scenario_frame, text="应用到现有条目",
                  command=self._apply_scenario_to_existing).grid(row=1, column=0, columnspan=3, pady=5)

        # === 左侧：临床特征组 ===
        clinical_frame = ttk.LabelFrame(main_frame, text="📋 临床特征", padding=15)
        clinical_frame.grid(row=2, column=0, sticky='nsew', padx=5, pady=5)

        # 使用grid布局
        row = 0
        self._add_labeled_field(clinical_frame, "年龄:", 'age', row, 0, 'Int', 0, 120, 30,
                               tooltip="年龄（岁），0-120。年龄 <5岁 或 >65岁 风险较高。")
        row += 1
        self._add_labeled_field(clinical_frame, "痰涂片结果:", 'sputum_smear', row, 0,
                               'Combo', ["1", "2"], ["涂阴", "涂阳"], "涂阴",
                               tooltip="痰涂片结果：涂阳表示传染性强，涂阴传染性较弱。")
        row += 1
        self._add_labeled_field(clinical_frame, "是否有空洞:", 'has_cavity', row, 0,
                               'Combo', ["1", "2"], ["无", "有"], "无",
                               tooltip="是否有空洞：有空洞表示病情较重，传染性更高。")
        row += 1
        self._add_labeled_field(clinical_frame, "是否活动性结核:", 'active_tb', row, 0,
                               'Combo', ["1", "2"], ["是", "否"], "是",
                               tooltip="是否活动性结核：活动性结核具有传染性。")

        # === 左侧2：治疗与症状组 ===
        treatment_frame = ttk.LabelFrame(main_frame, text="💊 治疗与症状", padding=15)
        treatment_frame.grid(row=2, column=1, sticky='nsew', padx=5, pady=5)

        row = 0
        self._add_labeled_field(treatment_frame, "是否接受治疗:", 'treatment', row, 0,
                               'Combo', ["1", "2"], ["是", "否"], "否",
                               tooltip="是否接受治疗：接受治疗可显著降低传染性。")
        row += 1
        self._add_labeled_field(treatment_frame, "治疗时长 (月):", 'treatment_duration', row, 0,
                               'Int', 0, 24, 0,
                               tooltip="治疗时长（月）：治疗时间越长，传染性越低。")
        row += 1
        self._add_labeled_field(treatment_frame, "每小时咳嗽次数:", 'cough_freq', row, 0,
                               'Int', 0, 100, 0,
                               tooltip="每小时咳嗽次数：频繁咳嗽会增加飞沫传播风险。")
        row += 1
        self._add_labeled_field(treatment_frame, "症状严重程度:", 'symptoms', row, 0,
                               'Combo', ["1", "2", "3", "4"], ["无症状", "轻度", "中度", "重度"], "轻度",
                               tooltip="症状严重程度：症状越重，传染性越高。")
        row += 1
        self._add_labeled_field(treatment_frame, "延迟就诊天数:", 'delay_days', row, 0,
                               'Int', 0, 365, 0,
                               tooltip="从出现咳嗽、发热等症状到确诊的天数。延迟越久，传播风险越高。")

        # === 右侧：环境与社会背景组 ===
        env_frame = ttk.LabelFrame(main_frame, text="🏠 环境与社会背景", padding=15)
        env_frame.grid(row=3, column=0, columnspan=2, sticky='ew', padx=5, pady=5)

        row = 0
        env_frame.grid_columnconfigure(0, weight=1)
        env_frame.grid_columnconfigure(1, weight=1)

        self._add_labeled_field(env_frame, "家庭居住条件:", 'family_living_conditions', row, 0,
                               'Combo', ["1", "2", "3", "4", "5"],
                               ["非常拥挤", "拥挤", "一般", "宽敞", "非常宽敞"], "一般",
                               col_width=25,
                               tooltip="家庭居住条件：居住越拥挤，感染风险越高。")

        self._add_labeled_field(env_frame, "家庭潜伏感染比例 (%):", 'flp_percentage', row, 1,
                               'Int', 0, 100, 5, col_width=25,
                               tooltip="家庭潜伏感染比例（%）：家庭中结核感染的比例。")
        row += 1

        self._add_labeled_field(env_frame, "社会高风险人群比例 (%):", 'hrsp_percentage', row, 0,
                               'Int', 0, 100, 10, col_width=25,
                               tooltip="社会高风险人群比例（%）：接触人群中HIV、糖尿病等高风险人群的比例。")

        self._add_labeled_field(env_frame, "家庭通风条件(默认):", 'ventilation', row, 1,
                               'Combo', ["1", "2", "3", "4", "5"],
                               ["极差", "差", "一般", "好", "极好"], "一般",
                               col_width=25,
                               tooltip="家庭通风条件：1=几乎不通风（无窗），2=很少开窗，3=偶尔开窗，4=经常开窗，5=通风极好。通风越好，感染风险越低。")


    def _add_labeled_field(self, parent, label_text, var_name, row, col, field_type,
                          param1=None, param2=None, default_val=None, col_width=15, tooltip=None):
        """添加带标签的输入字段（辅助函数）

        优先级四：对 Int 类型（Spinbox）自动调用 FieldValidator.attach()，
        绑定 validatecommand + <FocusOut> 验证，与 Classic 模式行为一致。
        Combo 类型因 state='readonly' 已约束可选值，无需额外验证。
        """
        ttk.Label(parent, text=label_text, font=('Microsoft YaHei', 10)).grid(row=row, column=col*2,
                                                                     sticky='w', pady=5, padx=5)

        if field_type == 'Int':
            self.basic_info_vars[var_name] = tk.IntVar(value=default_val or 0)
            widget = ttk.Spinbox(parent, from_=param1, to=param2,
                                textvariable=self.basic_info_vars[var_name], width=col_width)
        elif field_type == 'Combo':
            # param1 = 实际数值（如 ["1", "2"]），param2 = 显示标签（如 ["无", "有"]）
            actual_values = param1
            display_values = param2 if param2 else param1

            # Combobox 显示标签（良好 UX），StringVar 也存储标签
            # 数据采集时通过 _combo_maps 将标签翻译为实际数值
            self.basic_info_vars[var_name] = tk.StringVar(value=default_val or '')
            widget = ttk.Combobox(parent, textvariable=self.basic_info_vars[var_name],
                                 values=display_values, state='readonly', width=col_width-2)
            # 建立 label→value 和 value→label 双向映射
            if param2:
                if not hasattr(self, '_combo_maps'):
                    self._combo_maps = {}
                self._combo_maps[var_name] = dict(zip(param2, actual_values))
                if not hasattr(self, '_combo_maps_reverse'):
                    self._combo_maps_reverse = {}
                self._combo_maps_reverse[var_name] = dict(zip(actual_values, param2))

            # 设置默认值
            if default_val in display_values:
                widget.set(default_val)
            else:
                widget.set(display_values[0])

        widget.grid(row=row, column=col*2+1, sticky='w', pady=5, padx=5)

        # 添加tooltip
        if tooltip:
            self._add_tooltip(widget, tooltip)

        # 优先级四：对 Int 类型（Spinbox）附加 FieldValidator 验证
        # Combo 类型因 state='readonly' 已约束可选值，无需额外验证
        if field_type == 'Int' and hasattr(self, 'field_validator'):
            # required=True 仅对 age 字段生效（其他字段允许 0）
            required = (var_name == 'age')
            self.field_validator.attach(
                widget, var_name,
                min_val=param1, max_val=param2,
                is_int=True, required=required,
            )

        return widget


    def _init_improved_family_tab(self, parent):
        """初始化改进后的家庭成员标签页（精简核心字段）"""
        parent.grid_rowconfigure(0, weight=0)
        parent.grid_rowconfigure(1, weight=0)
        parent.grid_rowconfigure(2, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # 说明文字区域
        info_frame = ttk.LabelFrame(parent, text="📖 填写说明", padding=10)
        info_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        info_text = "• 填写与患者共同生活的家庭成员\n• 接触距离越近、时间越长，风险越高\n• 双击表格行可以编辑或标记确诊状态\n• 添加成员时会弹出详细编辑窗口\n• 更多帮助请在添加/编辑窗口中查看"
        ttk.Label(info_frame, text=info_text, justify='left', font=('Microsoft YaHei', 9)).pack(anchor='w')

        # 顶部按钮栏 — grid 两行布局，移除 emoji，缩短文本（P0 修复溢出）
        top_frame = ttk.Frame(parent, padding=5)
        top_frame.grid(row=1, column=0, sticky='ew')

        # 第一行：5 个操作按钮，均匀分布
        row1 = ttk.Frame(top_frame)
        row1.grid(row=0, column=0, sticky='ew', pady=(0, 3))
        for i in range(5):
            row1.grid_columnconfigure(i, weight=1)
        ttk.Button(row1, text="添加成员", command=self._add_family_member).grid(row=0, column=0, sticky='ew', padx=2)
        ttk.Button(row1, text="编辑", command=self._edit_family_member).grid(row=0, column=1, sticky='ew', padx=2)
        ttk.Button(row1, text="删除", command=self._delete_selected_family_wizard).grid(row=0, column=2, sticky='ew', padx=2)
        # 优先级一：Wizard 模式批量删除按钮（与 Classic 模式功能对等）
        ttk.Button(row1, text="批量删除", command=self._batch_delete_family_wizard).grid(row=0, column=3, sticky='ew', padx=2)
        ttk.Button(row1, text="粘贴导入", command=lambda: self._show_paste_dialog('family')).grid(row=0, column=4, sticky='ew', padx=2)

        # 第二行：标记高风险为确诊（独占一行）
        row2 = ttk.Frame(top_frame)
        row2.grid(row=1, column=0, sticky='ew', pady=(3, 0))
        row2.grid_columnconfigure(0, weight=1)
        ttk.Button(row2, text="标记高风险确诊", command=self._mark_high_risk_diagnosed).grid(row=0, column=0, sticky='ew', padx=2)

        # 精简后的表格列（核心字段）
        columns = ('name', 'age', 'relationship', 'freq_density', 'time_span',
                  'risk_prob', 'diagnosed')
        col_widths = {'name': 80, 'age': 50, 'relationship': 70, 'freq_density': 80,
                     'time_span': 70, 'risk_prob': 80, 'diagnosed': 60}
        col_headers = {'name': '姓名', 'age': '年龄', 'relationship': '关系',
                      'freq_density': '每周频次', 'time_span': '持续周期(周)',
                      'risk_prob': '风险概率', 'diagnosed': '确诊'}

        self.family_tree = self._create_treeview_wizard(parent, columns, col_widths, col_headers,
                                                        row=2)
        # 优先级一：启用 extended 选择模式（Ctrl+多选 / Shift+范围选择）
        self.family_tree.configure(selectmode='extended')
        self.family_tree.bind('<Double-1>', self._on_family_tree_double_click)


    def _init_improved_social_tab(self, parent):
        """初始化改进后的社会接触者标签页"""
        parent.grid_rowconfigure(0, weight=0)
        parent.grid_rowconfigure(1, weight=0)
        parent.grid_rowconfigure(2, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # 说明文字区域
        info_frame = ttk.LabelFrame(parent, text="📖 填写说明", padding=10)
        info_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        info_text = "• 填写与患者有密切接触的社会关系（同事、朋友等）\n• 注意标注高危人群（HIV、糖尿病、免疫抑制等）\n• 拥挤和密闭场所风险显著高于一般环境\n• 双击表格行可以编辑或标记确诊状态\n• 更多帮助请在添加/编辑窗口中查看"
        ttk.Label(info_frame, text=info_text, justify='left', font=('Microsoft YaHei', 9)).pack(anchor='w')

        # 顶部按钮栏 — grid 布局，移除 emoji，缩短文本（P0 修复溢出）
        top_frame = ttk.Frame(parent, padding=5)
        top_frame.grid(row=1, column=0, sticky='ew')

        # 单行：5 个操作按钮，均匀分布
        for i in range(5):
            top_frame.grid_columnconfigure(i, weight=1)
        ttk.Button(top_frame, text="添加接触者", command=self._add_social_contact).grid(row=0, column=0, sticky='ew', padx=2)
        ttk.Button(top_frame, text="编辑", command=self._edit_social_contact).grid(row=0, column=1, sticky='ew', padx=2)
        ttk.Button(top_frame, text="删除", command=self._delete_selected_social_wizard).grid(row=0, column=2, sticky='ew', padx=2)
        # 优先级一：Wizard 模式批量删除按钮（与 Classic 模式功能对等）
        ttk.Button(top_frame, text="批量删除", command=self._batch_delete_social_wizard).grid(row=0, column=3, sticky='ew', padx=2)
        ttk.Button(top_frame, text="粘贴导入", command=lambda: self._show_paste_dialog('social')).grid(row=0, column=4, sticky='ew', padx=2)

        # 精简后的表格列
        columns = ('name', 'age', 'freq_density', 'time_span', 'risk_prob', 'diagnosed')
        col_widths = {'name': 80, 'age': 50, 'freq_density': 80,
                     'time_span': 70, 'risk_prob': 80, 'diagnosed': 60}
        col_headers = {'name': '姓名', 'age': '年龄', 'freq_density': '每周频次',
                      'time_span': '持续周期(周)', 'risk_prob': '风险概率',
                      'diagnosed': '确诊'}

        self.social_tree = self._create_treeview_wizard(parent, columns, col_widths, col_headers,
                                                        row=2)
        # 优先级一：启用 extended 选择模式（Ctrl+多选 / Shift+范围选择）
        self.social_tree.configure(selectmode='extended')
        self.social_tree.bind('<Double-1>', self._on_social_tree_double_click)


    def _init_dashboard_result_tab(self, parent):
        """初始化仪表盘结果标签页"""
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        main_frame = ttk.Frame(parent, padding=10)
        main_frame.grid(row=0, column=0, sticky='nsew')
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=2)
        main_frame.grid_columnconfigure(0, weight=1)

        # === 顶部：核心卡片区域 ===
        cards_frame = ttk.Frame(main_frame)
        cards_frame.grid(row=0, column=0, sticky='nsew', pady=10)
        cards_frame.grid_columnconfigure(0, weight=1)
        cards_frame.grid_columnconfigure(1, weight=1)
        cards_frame.grid_columnconfigure(2, weight=1)

        # 根据主题选择卡片配色（深色主题用深色背景+浅色文字）
        is_dark = getattr(self, '_current_theme', 'light') == 'dark'
        if is_dark:
            risk_bg, risk_border = '#1a3a4a', '#5dade2'
            warn_bg, warn_border = '#4a3a00', '#f4d03f'
            ok_bg, ok_border = '#1a3a1a', '#58d68d'
        else:
            risk_bg, risk_border = '#e6f7ff', '#91d5ff'
            warn_bg, warn_border = '#fff7e6', '#ffd591'
            ok_bg, ok_border = '#f6ffed', '#b7eb8f'

        # 卡片1：综合风险等级
        self.risk_card = self._create_info_card(cards_frame, "综合风险等级",
                                               "低风险", "🟢", row=0, col=0,
                                               bg_color=risk_bg, border_color=risk_border)

        # 卡片2：高风险接触者
        self.high_risk_card = self._create_info_card(cards_frame, "高风险接触者",
                                                    "0人", "⚠️", row=0, col=1,
                                                    bg_color=warn_bg, border_color=warn_border)

        # 卡片3：ML/GNN预测
        self.ml_card = self._create_info_card(cards_frame, "ML集成预测概率",
                                             "等待评估...", "🤖", row=0, col=2,
                                             bg_color=ok_bg, border_color=ok_border)

        # === 底部：图表与详情区域（使用折叠面板）===
        self._create_result_detail_area(main_frame, row=1)


    def _create_info_card(self, parent, title, value, icon, row, col,
                         bg_color='white', border_color='gray'):
        """创建信息卡片（辅助函数，带悬停高亮效果）

        Returns:
            dict: {'frame': card_frame, 'title_label': ..., 'value_label': ..., 'icon_label': ...}
            消费方通过 card['value_label'].configure(text=...) 更新卡片数值。
        """
        card_frame = tk.Frame(parent, bg=bg_color, highlightbackground=border_color,
                               highlightthickness=1, padx=16, pady=12)
        card_frame.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
        card_frame.grid_propagate(False)

        # 图标 + 标题行
        header_frame = tk.Frame(card_frame, bg=bg_color)
        header_frame.pack(fill='x', pady=(0, 4))
        # 获取主题颜色（向导作为Mixin可访问self.COLORS）
        text_light = getattr(self, 'COLORS', {}).get('text_light', '#555555')
        text_main = getattr(self, 'COLORS', {}).get('text', '#333333')
        icon_label = tk.Label(header_frame, text=icon, font=('Microsoft YaHei', 13),
                              bg=bg_color, fg=border_color)
        icon_label.pack(side='left', padx=(0, 5))
        title_label = tk.Label(header_frame, text=title, font=('Microsoft YaHei', 9),
                               bg=bg_color, fg=text_light)
        title_label.pack(side='left')

        # 数值标签（大字号居中）
        value_label = tk.Label(card_frame, text=value, font=('Microsoft YaHei', 14, 'bold'),
                               bg=bg_color, fg=text_main)
        value_label.pack(fill='x', pady=(2, 0))

        # 悬停高亮：进入时加深背景，离开时恢复
        hover_color = border_color
        original_bg = bg_color

        def _on_enter(event):
            card_frame.configure(bg=hover_color)
            for sub in (header_frame, icon_label, title_label, value_label):
                try:
                    sub.configure(bg=hover_color)
                except tk.TclError:
                    pass

        def _on_leave(event):
            card_frame.configure(bg=original_bg)
            for sub in (header_frame, icon_label, title_label, value_label):
                try:
                    sub.configure(bg=original_bg)
                except tk.TclError:
                    pass

        card_frame.bind('<Enter>', _on_enter)
        card_frame.bind('<Leave>', _on_leave)
        for sub in (icon_label, title_label, value_label):
            sub.bind('<Enter>', _on_enter)
            sub.bind('<Leave>', _on_leave)

        return {
            'frame': card_frame,
            'title_label': title_label,
            'value_label': value_label,
            'icon_label': icon_label,
        }

    def _create_treeview_wizard(self, parent, columns, col_widths, col_headers, row=0):
        """创建带滚动的 Treeview（向导布局，使用 grid）

        与 assessment.py 中 _create_treeview 功能一致，但使用 grid 布局
        以适配向导步骤页的 row/column 定位。

        Returns:
            ttk.Treeview: 创建好的 Treeview 对象
        """
        container = ttk.Frame(parent)
        container.grid(row=row, column=0, sticky='nsew', padx=10, pady=5)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        tree = ttk.Treeview(container, columns=columns, show='headings', height=8)

        for col in columns:
            tree.heading(col, text=col_headers.get(col, col))
            tree.column(col, width=col_widths.get(col, 60), minwidth=40)

        v_scroll = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        h_scroll = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        tree.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')

        # 注册到主题同步列表（斑马纹 + 风险色随主题切换）
        if hasattr(self, '_register_tree_widget'):
            self._register_tree_widget(tree)

        return tree

    def _create_result_detail_area(self, parent, row=0):
        """创建结果详情区域（图表 + 文本详情，使用可折叠 Notebook）"""
        detail_frame = ttk.LabelFrame(parent, text="📋 评估详情", padding=5)
        detail_frame.grid(row=row, column=0, sticky='nsew', pady=5)
        detail_frame.grid_rowconfigure(0, weight=1)
        detail_frame.grid_columnconfigure(0, weight=1)

        # 使用 Notebook 容纳图表和文本
        self.detail_notebook = ttk.Notebook(detail_frame)
        self.detail_notebook.grid(row=0, column=0, sticky='nsew')

        # 图表标签页
        self.chart_tab = ttk.Frame(self.detail_notebook)
        self.detail_notebook.add(self.chart_tab, text="📊 图表")
        self.chart_tab.grid_rowconfigure(0, weight=1)
        self.chart_tab.grid_columnconfigure(0, weight=1)

        # 图表容器（matplotlib Figure 由 charts 模块后续填充）
        self.wizard_chart_frame = ttk.Frame(self.chart_tab)
        self.wizard_chart_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        self.wizard_chart_frame.grid_rowconfigure(0, weight=1)
        self.wizard_chart_frame.grid_columnconfigure(0, weight=1)

        # 文本详情标签页
        self.text_tab = ttk.Frame(self.detail_notebook)
        self.detail_notebook.add(self.text_tab, text="📝 文本详情")
        self.text_tab.grid_rowconfigure(0, weight=1)
        self.text_tab.grid_columnconfigure(0, weight=1)

        from tkinter.scrolledtext import ScrolledText
        self.result_text = ScrolledText(self.text_tab, wrap='word', font=('Microsoft YaHei', 10),
                                        height=8, state='disabled')
        self.result_text.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
