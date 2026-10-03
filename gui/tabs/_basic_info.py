#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""患者基本信息标签页 Mixin

从原 gui/tabs/__init__.py 拆分而来，仅包含 _init_basic_info_tab 方法。
对应原文件第 208-359 行。

改进1：重构为三栏卡片分组布局，统一字体和间距，自适应宽度。
"""

from .._gui_common import tk, ttk, KARAMAY_LOCALIZER_AVAILABLE
# Section V: 验证常量从 constants.py 单一真相源导入（消除硬编码上下限）
from ...constants import (
    MIN_AGE, MAX_AGE,
    MAX_TREATMENT_DURATION_MONTHS, MIN_TREATMENT_DURATION_MONTHS,
    MAX_COUGH_FREQ, MIN_COUGH_FREQ,
    MAX_DELAY_DAYS, MIN_DELAY_DAYS,
    PERCENTAGE_MIN, PERCENTAGE_MAX,
)


class BasicInfoTabMixin:
    """GUI 选项卡方法混合类 — 患者基本信息标签页"""

    def _init_basic_info_tab(self, parent):
        """初始化患者基本信息标签页"""
        # 场景选择 — 顶部独立区域
        scenario_frame = ttk.LabelFrame(parent, text="智能场景选择", padding=12)
        scenario_frame.pack(fill='x', padx=20, pady=(12, 6))

        scenario_inner = ttk.Frame(scenario_frame)
        scenario_inner.pack(fill='x')
        ttk.Label(scenario_inner, text="选择预设场景：",
                 font=self._FONT_DEFAULT).pack(side='left', padx=(0, 8))
        self.scenario_var = tk.StringVar(value="custom")
        scenario_options_basic = ["custom", "rural_family", "urban_workplace", "school"]
        if KARAMAY_LOCALIZER_AVAILABLE:
            scenario_options_basic.append("karamay_oilfield")
        scenario_combo = ttk.Combobox(scenario_inner, textvariable=self.scenario_var,
                                     values=scenario_options_basic,
                                     state="readonly")
        scenario_combo.pack(side='left', padx=(0, 8))
        scenario_combo.bind('<<ComboboxSelected>>', self._on_scenario_change)

        self.scenario_desc_var = tk.StringVar(value="自定义模式：手动输入所有参数")
        scenario_desc_label = ttk.Label(scenario_inner, textvariable=self.scenario_desc_var,
                                       font=self._FONT_SMALL,
                                       foreground=self.COLORS['info'])
        scenario_desc_label.pack(side='left', padx=8)

        ttk.Button(scenario_inner, text="应用到现有条目",
                  command=self._apply_scenario_to_existing).pack(side='right', padx=8)
        # 改进7：智能填写按钮
        ttk.Button(scenario_inner, text="智能填写",
                  command=self._smart_fill_basic_info).pack(side='right', padx=8)

        # 汇总容器 — 三列网格
        cards_frame = ttk.Frame(parent)
        cards_frame.pack(fill='both', expand=True, padx=20, pady=(6, 12))
        for i in range(3):
            cards_frame.columnconfigure(i, weight=1, uniform='card_col')
        cards_frame.rowconfigure(0, weight=1)

        # ── 卡片 1：患者核心信息 ──
        card1 = ttk.Frame(cards_frame, style='Card.TFrame', padding=18)
        card1.grid(row=0, column=0, sticky='nsew', padx=(0, 10), pady=5)
        card1.columnconfigure(1, weight=1, minsize=60)

        ttk.Label(card1, text="患者核心信息",
                 font=self._FONT_HEADING, style='CardTitle.TLabel').grid(
            row=0, column=0, columnspan=3, sticky='w', pady=(0, 8))

        r = 1
        self._card_row(card1, r, "年龄:", self.basic_info_vars, 'age', tk.IntVar(value=30),
                       widget_type='spinbox', from_=MIN_AGE, to=MAX_AGE,
                       tooltip="年龄（岁），0-120。年龄 <5岁 或 >65岁 风险较高。",
                       field_name='age', required=True)
        r += 1
        self._card_row(card1, r, "痰涂片结果:", self.basic_info_vars, 'sputum_smear',
                       tk.StringVar(value="1"),
                       widget_type='combo', values=["1", "2"],
                       hint="(1=涂阴，2=涂阳)",
                       tooltip="痰涂片结果：涂阳（2）表示传染性强；涂阴（1）传染性较弱。")
        r += 1
        self._card_row(card1, r, "是否有空洞:", self.basic_info_vars, 'has_cavity',
                       tk.StringVar(value="1"),
                       widget_type='combo', values=["1", "2"],
                       hint="(1=无，2=有)",
                       tooltip="是否有空洞：有空洞表示病情较重，传染性更高。")
        r += 1
        self._card_row(card1, r, "是否活动性结核:", self.basic_info_vars, 'active_tb',
                       tk.StringVar(value="1"),
                       widget_type='combo', values=["1", "2"],
                       hint="(1=是，2=否)",
                       tooltip="是否活动性结核：活动性结核具有传染性。")
        r += 1
        self._card_row(card1, r, "是否接受治疗:", self.basic_info_vars, 'treatment',
                       tk.StringVar(value="2"),
                       widget_type='combo', values=["1", "2"],
                       hint="(1=是，2=否)",
                       tooltip="是否接受治疗：接受治疗可显著降低传染性。")
        r += 1
        self._card_row(card1, r, "治疗时长 (月):", self.basic_info_vars, 'treatment_duration',
                       tk.IntVar(value=0),
                       widget_type='spinbox', from_=MIN_TREATMENT_DURATION_MONTHS,
                       to=MAX_TREATMENT_DURATION_MONTHS,
                       tooltip="治疗时长（月）：治疗时间越长，传染性越低。",
                       field_name='treatment_duration')

        # ── 卡片 2：传播相关因素 ──
        card2 = ttk.Frame(cards_frame, style='Card.TFrame', padding=18)
        card2.grid(row=0, column=1, sticky='nsew', padx=10, pady=5)
        card2.columnconfigure(1, weight=1, minsize=60)

        ttk.Label(card2, text="传播相关因素",
                 font=self._FONT_HEADING, style='CardTitle.TLabel').grid(
            row=0, column=0, columnspan=3, sticky='w', pady=(0, 8))

        r = 1
        self._card_row(card2, r, "每小时咳嗽次数:", self.basic_info_vars, 'cough_freq',
                       tk.IntVar(value=0),
                       widget_type='spinbox', from_=MIN_COUGH_FREQ, to=MAX_COUGH_FREQ,
                       tooltip="每小时咳嗽次数：频繁咳嗽会增加飞沫传播风险。",
                       field_name='cough_freq')
        r += 1
        self._card_row(card2, r, "症状严重程度:", self.basic_info_vars, 'symptoms',
                       tk.StringVar(value="2"),
                       widget_type='combo', values=["1", "2", "3", "4"],
                       hint="(1无2轻3中4重)",
                       tooltip="症状严重程度：1=无症状，2=轻度，3=中度，4=重度。症状越重，传染性越高。")
        r += 1
        self._card_row(card2, r, "延迟就诊天数:", self.basic_info_vars, 'delay_days',
                       tk.IntVar(value=0),
                       widget_type='spinbox', from_=MIN_DELAY_DAYS, to=MAX_DELAY_DAYS,
                       tooltip="从出现症状到确诊的天数。延迟越久，传播风险越高。",
                       field_name='delay_days')
        r += 1
        self._card_row(card2, r, "家庭通风条件(默认):", self.basic_info_vars, 'ventilation',
                       tk.StringVar(value="3"),
                       widget_type='combo', values=["1", "2", "3", "4", "5"],
                       hint="(1差5好)",
                       tooltip="家庭通风条件：1=极差，5=极好。通风越好传播风险越低。新成员默认值。")
        r += 1
        # 症状描述 — 标签在列0，Text横跨列1和列2（该行无hint标签）
        ttk.Label(card2, text="症状描述:", font=self._FONT_DEFAULT,
                  style='Card.TLabel').grid(
            row=r, column=0, sticky='ne', padx=10, pady=7)
        # 使用tk.Text默认边框设置(relief='sunken', bd=2, highlightthickness=1)
        # 注意：sunken relief至少需要bd>=2才能渲染可见的3D斜角边框
        symptom_text = self._register_text_widget(
            tk.Text(card2, height=5, font=self._FONT_DEFAULT, wrap='word'))
        symptom_text.grid(row=r, column=1, columnspan=2, sticky='ew', padx=10, pady=7)
        symptom_text.insert('1.0', '咳嗽、低热、盗汗')
        self.basic_info_vars['symptom_description'] = symptom_text

        # ── 卡片 3：环境与其他 ──
        card3 = ttk.Frame(cards_frame, style='Card.TFrame', padding=18)
        card3.grid(row=0, column=2, sticky='nsew', padx=(10, 0), pady=5)
        card3.columnconfigure(1, weight=1, minsize=60)

        ttk.Label(card3, text="环境与其他",
                 font=self._FONT_HEADING, style='CardTitle.TLabel').grid(
            row=0, column=0, columnspan=3, sticky='w', pady=(0, 8))

        r = 1
        self._card_row(card3, r, "家庭居住条件:", self.basic_info_vars, 'family_living_conditions',
                       tk.StringVar(value="3"),
                       widget_type='combo', values=["1", "2", "3", "4", "5"],
                       hint="(1挤5宽)",
                       tooltip="家庭居住条件：1=非常拥挤，5=非常宽敞。居住越拥挤，感染风险越高。")
        r += 1
        self._card_row(card3, r, "家庭潜伏感染比例 (%):", self.basic_info_vars, 'flp_percentage',
                       tk.IntVar(value=5),
                       widget_type='spinbox', from_=PERCENTAGE_MIN, to=PERCENTAGE_MAX,
                       tooltip="家庭中结核感染的比例。",
                       field_name='flp_percentage')
        r += 1
        self._card_row(card3, r, "社会高风险人群比例 (%):", self.basic_info_vars, 'hrsp_percentage',
                       tk.IntVar(value=10),
                       widget_type='spinbox', from_=PERCENTAGE_MIN, to=PERCENTAGE_MAX,
                       tooltip="接触人群中HIV、糖尿病等高风险人群的比例。",
                       field_name='hrsp_percentage')

        # 填写说明
        info_frame = ttk.LabelFrame(parent, text="填写说明", padding=12)
        info_frame.pack(fill='x', padx=20, pady=(6, 12))
        info_text = ("年龄 <5岁 或 >65岁 风险较高 | "
                     "涂阳患者传染性显著高于涂阴 | "
                     "延迟就诊时间越长，传播风险越高 | "
                     "家庭通风条件影响所有接触者的默认值")
        ttk.Label(info_frame, text=info_text, justify='left',
                 font=self._FONT_SMALL,
                 style='Tip.TLabel').pack(anchor='w')

    def _card_row(self, parent, row, label_text, var_dict, var_key, default_var,
                  widget_type='entry', values=None, hint=None, tooltip=None,
                  from_=None, to=None, field_name=None, required=False):
        """在卡片中创建一行标签+控件

        参数：
            widget_type: 'entry' | 'spinbox' | 'combo'
        """
        # 必填字段：使用Frame容器放文本标签+红色*号（与dialog.py保持一致）
        if required:
            lbl_frame = ttk.Frame(parent)
            lbl_frame.grid(row=row, column=0, sticky='e', padx=10, pady=7)
            asterisk = tk.Label(lbl_frame, text="*", fg=self.COLORS.get('danger', '#e74c3c'),
                               font=self._FONT_DEFAULT,
                               bg=self.COLORS['bg_card'])
            asterisk.pack(side='right', padx=(0, 1))
            lbl = ttk.Label(lbl_frame, text=label_text, font=self._FONT_DEFAULT,
                            style='Card.TLabel')
            lbl.pack(side='right')
        else:
            lbl = ttk.Label(parent, text=label_text, font=self._FONT_DEFAULT,
                            style='Card.TLabel')
            lbl.grid(row=row, column=0, sticky='e', padx=10, pady=7)

        widget = None
        if widget_type == 'combo':
            var_dict[var_key] = default_var
            widget = ttk.Combobox(parent, textvariable=var_dict[var_key],
                                 values=values, state="readonly", width=5)
            widget.grid(row=row, column=1, sticky='ew', padx=10, pady=7)
        elif widget_type == 'spinbox':
            var_dict[var_key] = default_var
            widget = ttk.Spinbox(parent, from_=from_, to=to,
                                textvariable=var_dict[var_key], width=5)
            widget.grid(row=row, column=1, sticky='ew', padx=10, pady=7)
        elif widget_type == 'entry':
            var_dict[var_key] = default_var
            widget = ttk.Entry(parent, textvariable=var_dict[var_key], width=8)
            widget.grid(row=row, column=1, sticky='ew', padx=10, pady=7)

        if hint:
            ttk.Label(parent, text=hint, font=self._FONT_SMALL,
                     style='CardTip.TLabel').grid(row=row, column=2, sticky='w', padx=(4, 0))

        # 改进7：绑定焦点事件，在底部状态栏显示帮助文本
        if tooltip and widget:
            self._add_tooltip(widget, tooltip)
            widget.bind('<FocusIn>', lambda e, t=tooltip: self._set_help_status(t))
            widget.bind('<FocusOut>', lambda e: self._set_help_status(''))

        if field_name and hasattr(self, 'field_validator') and widget:
            self.field_validator.attach(widget, field_name, required=required)

    def _smart_fill_basic_info(self):
        """改进7：根据场景自动填充合理默认值"""
        scenario = self.scenario_var.get()
        if scenario == 'custom':
            # 通用智能填写：典型结核患者特征
            defaults = {
                'age': 45,
                'sputum_smear': '2',       # 涂阳
                'has_cavity': '1',          # 无空洞
                'active_tb': '1',           # 活动性
                'treatment': '2',           # 未治疗
                'treatment_duration': 0,
                'cough_freq': 3,
                'symptoms': '3',            # 中度
                'delay_days': 14,
                'ventilation': '3',
                'family_living_conditions': '3',
                'flp_percentage': 5,
                'hrsp_percentage': 10,
            }
        elif scenario == 'rural_family':
            defaults = {
                'age': 55,
                'sputum_smear': '2',
                'has_cavity': '2',          # 有空洞
                'active_tb': '1',
                'treatment': '2',
                'treatment_duration': 0,
                'cough_freq': 5,
                'symptoms': '4',            # 重度
                'delay_days': 30,
                'ventilation': '2',
                'family_living_conditions': '2',
                'flp_percentage': 15,
                'hrsp_percentage': 5,
            }
        elif scenario == 'urban_workplace':
            defaults = {
                'age': 35,
                'sputum_smear': '1',
                'has_cavity': '1',
                'active_tb': '1',
                'treatment': '1',           # 已治疗
                'treatment_duration': 2,
                'cough_freq': 1,
                'symptoms': '2',            # 轻度
                'delay_days': 7,
                'ventilation': '4',
                'family_living_conditions': '4',
                'flp_percentage': 3,
                'hrsp_percentage': 5,
            }
        elif scenario == 'school':
            defaults = {
                'age': 16,
                'sputum_smear': '2',
                'has_cavity': '1',
                'active_tb': '1',
                'treatment': '2',
                'treatment_duration': 0,
                'cough_freq': 3,
                'symptoms': '3',
                'delay_days': 5,
                'ventilation': '3',
                'family_living_conditions': '3',
                'flp_percentage': 8,
                'hrsp_percentage': 3,
            }
        elif scenario == 'karamay_oilfield':
            defaults = {
                'age': 42,
                'sputum_smear': '2',
                'has_cavity': '1',
                'active_tb': '1',
                'treatment': '2',
                'treatment_duration': 0,
                'cough_freq': 2,
                'symptoms': '3',
                'delay_days': 21,
                'ventilation': '2',
                'family_living_conditions': '2',
                'flp_percentage': 10,
                'hrsp_percentage': 8,
            }
        else:
            return

        for key, value in defaults.items():
            if key in self.basic_info_vars:
                var = self.basic_info_vars[key]
                if isinstance(var, tk.IntVar):
                    var.set(int(value))
                elif isinstance(var, tk.StringVar):
                    var.set(str(value))

        # 更新症状描述
        symptom_map = {
            'custom': '咳嗽、低热、盗汗',
            'rural_family': '持续咳嗽、咳血、胸痛、夜间盗汗、体重下降',
            'urban_workplace': '偶有干咳、轻微乏力',
            'school': '咳嗽、发热、食欲减退',
            'karamay_oilfield': '慢性咳嗽、胸闷、气短、盗汗、乏力',
        }
        if 'symptom_description' in self.basic_info_vars:
            text_widget = self.basic_info_vars['symptom_description']
            text_widget.delete('1.0', tk.END)
            text_widget.insert('1.0', symptom_map.get(scenario, symptom_map['custom']))

        self._set_help_status(f"已应用「{scenario}」场景的智能填写值")