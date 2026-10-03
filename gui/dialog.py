#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ContactEditDialog - 接触者编辑对话框

改进2：重新组织为4个LabelFrame分组，两列网格布局，添加滚动条支持，
必填字段红色*标记，按钮右对齐。
"""
# tkinter 在无图形环境（服务器/容器）中可能不可用
from ..utils import TKINTER_AVAILABLE  # tkinter 可用性标志（单一真相源）
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    tk = None
    ttk = None
    messagebox = None

from ..utils import LOGGER
from .tooltip import ToolTip
# Section V: 验证常量从 constants.py 单一真相源导入
from ..constants import (
    MIN_AGE, MAX_AGE,
    MAX_SINGLE_DURATION_MINUTES, MIN_SINGLE_DURATION_MINUTES,
    MAX_FREQ_DENSITY, MIN_FREQ_DENSITY,
    MAX_TIME_SPAN_WEEKS, MIN_TIME_SPAN_WEEKS,
    MIN_VENTILATION, MAX_VENTILATION,
)

# 改进6：模块级语义化颜色常量，替换硬编码 'gray'/'black'/'red'
_DIALOG_COLORS = {
    'text': '#2c3e50',
    'text_light': '#5a6c7d',
    'required': '#e74c3c',
    'disabled': '#999999',
    'bg_light': '#f8f9fa',
}

# 改进6：模块级字体常量，保持与主窗口一致
_DIALOG_FONT_DEFAULT = ('Microsoft YaHei', 10)
_DIALOG_FONT_SMALL = ('Microsoft YaHei', 8)
_DIALOG_FONT_BOLD = ('Microsoft YaHei', 10, 'bold')


if TKINTER_AVAILABLE:
    class ContactEditDialog(tk.Toplevel):
        """接触者编辑对话框，用于添加/编辑家庭成员或社会接触者"""

        def __init__(self, parent, contact_type='family', initial_data=None, scenario_defaults=None):
            """初始化编辑对话框

            参数：
                parent: 父窗口
                contact_type (str): 'family' 或 'social'
                initial_data (dict|None): 初始数据，为None时使用默认值
                scenario_defaults (dict|None): 场景默认值
            """
            super().__init__(parent)
            self.result = None
            self.contact_type = contact_type
            self.transient(parent)
            self.grab_set()

            title = "编辑家庭成员" if contact_type == 'family' else "编辑社会接触者"
            self.title(title)
            self.resizable(True, True)
            self.minsize(520, 580)

            defaults = scenario_defaults or {}

            # 可滚动区域
            canvas = tk.Canvas(self, highlightthickness=0, bg=_DIALOG_COLORS['bg_light'])
            scrollbar = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side='left', fill='both', expand=True)
            scrollbar.pack(side='right', fill='y')

            # 鼠标滚轮支持
            self._bind_mousewheel(canvas)

            # 主内容
            main_frame = ttk.Frame(scrollable_frame, padding=10)
            main_frame.pack(fill='both', expand=True)
            main_frame.columnconfigure(0, weight=1)
            main_frame.columnconfigure(1, weight=1)
            main_frame.rowconfigure(2, weight=1)  # 让并排分组可垂直扩展

            # ── 分组 1：基本信息 ──
            group1 = ttk.LabelFrame(main_frame, text="基本信息", padding=10)
            group1.grid(row=0, column=0, columnspan=2, sticky='ew', padx=5, pady=(5, 3))
            group1.columnconfigure(1, weight=1)
            group1.columnconfigure(3, weight=1)
            self._create_required_label(group1, "姓名:", 0, 0)
            self.name_var = tk.StringVar(value=initial_data.get('name', '') if initial_data else '')
            name_entry = ttk.Entry(group1, textvariable=self.name_var)
            name_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=3)
            ToolTip(name_entry, "接触者姓名。")

            self._create_required_label(group1, "年龄:", 0, 2)
            self.age_var = tk.IntVar(value=initial_data.get('age', 30) if initial_data else 30)
            age_spin = ttk.Spinbox(group1, from_=MIN_AGE, to=MAX_AGE, textvariable=self.age_var)
            age_spin.grid(row=0, column=3, sticky='ew', padx=5, pady=3)
            ToolTip(age_spin, "接触者年龄（0-120岁）。年龄 <5岁 或 >65岁 风险较高。")

            if contact_type == 'family':
                ttk.Label(group1, text="关系:", font=_DIALOG_FONT_DEFAULT).grid(
                    row=1, column=0, sticky='e', padx=5, pady=3)
                self.relationship_var = tk.StringVar(
                    value=initial_data.get('relationship', '配偶') if initial_data else '配偶')
                relationship_combo = ttk.Combobox(group1, textvariable=self.relationship_var,
                                                  values=["配偶", "子女", "父母", "兄弟姐妹", "其他"],
                                                  state="readonly")
                relationship_combo.grid(row=1, column=1, sticky='ew', padx=5, pady=3)
                ToolTip(relationship_combo, "与患者的关系。")

            # ── 分组 2：接触特征 ──
            group2 = ttk.LabelFrame(main_frame, text="接触特征", padding=10)
            group2.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5, pady=3)
            group2.columnconfigure(1, weight=1)
            group2.columnconfigure(3, weight=1)

            # 接触三要素同一行
            ttk.Label(group2, text="单次时长(分钟):", font=_DIALOG_FONT_DEFAULT).grid(
                row=0, column=0, sticky='e', padx=5, pady=3)
            self.single_duration_var = tk.IntVar(
                value=initial_data.get('single_duration', defaults.get('single_duration', 30))
                if initial_data else defaults.get('single_duration', 30))
            sd_spin = ttk.Spinbox(group2, from_=MIN_SINGLE_DURATION_MINUTES,
                                  to=MAX_SINGLE_DURATION_MINUTES,
                                  textvariable=self.single_duration_var)
            sd_spin.grid(row=0, column=1, sticky='ew', padx=5, pady=3)
            ToolTip(sd_spin, "每次接触的持续时间（分钟）。接触时长越长，风险越高。")

            ttk.Label(group2, text="每周频次:", font=_DIALOG_FONT_DEFAULT).grid(
                row=0, column=2, sticky='e', padx=5, pady=3)
            default_freq = defaults.get('freq_density', 14 if contact_type == 'family' else 2)
            self.freq_density_var = tk.IntVar(
                value=initial_data.get('freq_density', default_freq) if initial_data else default_freq)
            freq_spin = ttk.Spinbox(group2, from_=MIN_FREQ_DENSITY, to=MAX_FREQ_DENSITY,
                                    textvariable=self.freq_density_var)
            freq_spin.grid(row=0, column=3, sticky='ew', padx=5, pady=3)
            ToolTip(freq_spin, "每周接触次数。接触越频繁，风险越高。")

            ttk.Label(group2, text="持续周期(周):", font=_DIALOG_FONT_DEFAULT).grid(
                row=1, column=0, sticky='e', padx=5, pady=3)
            self.time_span_var = tk.IntVar(
                value=initial_data.get('time_span', defaults.get('time_span', 4))
                if initial_data else defaults.get('time_span', 4))
            ts_spin = ttk.Spinbox(group2, from_=MIN_TIME_SPAN_WEEKS, to=MAX_TIME_SPAN_WEEKS,
                                  textvariable=self.time_span_var)
            ts_spin.grid(row=1, column=1, sticky='ew', padx=5, pady=3)
            ToolTip(ts_spin, "接触持续的周期（周）。接触时间越长，风险越高。")

            # 通风条件 + 接触距离 同一行
            ttk.Label(group2, text="通风条件(1-5):", font=_DIALOG_FONT_DEFAULT).grid(
                row=2, column=0, sticky='e', padx=5, pady=3)
            vent_default = str(defaults.get('ventilation', 3))
            self.ventilation_var = tk.StringVar(
                value=initial_data.get('ventilation', vent_default) if initial_data else vent_default)
            vent_combo = ttk.Combobox(group2, textvariable=self.ventilation_var,
                                      values=["1", "2", "3", "4", "5"], state="readonly")
            vent_combo.grid(row=2, column=1, sticky='ew', padx=5, pady=3)
            ToolTip(vent_combo, "接触环境的通风条件：1=几乎不通风，5=通风极好。")

            dist_default = defaults.get('contact_distance', '近' if contact_type == 'family' else '中等')
            ttk.Label(group2, text="接触距离:", font=_DIALOG_FONT_DEFAULT).grid(
                row=2, column=2, sticky='e', padx=5, pady=3)
            self.contact_distance_var = tk.StringVar(
                value=initial_data.get('contact_distance', dist_default) if initial_data else dist_default)
            dist_combo = ttk.Combobox(group2, textvariable=self.contact_distance_var,
                                      values=["极近", "近", "中等", "远", "极远"], state="readonly")
            dist_combo.grid(row=2, column=3, sticky='ew', padx=5, pady=3)
            ToolTip(dist_combo, "与患者的接触距离。距离越近，风险越高。")

            setting_default = defaults.get('exposure_setting', '一般')
            ttk.Label(group2, text="暴露场景:", font=_DIALOG_FONT_DEFAULT).grid(
                row=3, column=0, sticky='e', padx=5, pady=3)
            self.exposure_setting_var = tk.StringVar(
                value=initial_data.get('exposure_setting', setting_default)
                if initial_data else setting_default)
            setting_combo = ttk.Combobox(group2, textvariable=self.exposure_setting_var,
                                         values=["拥挤", "密闭", "一般", "户外", "油田营地"],
                                         state="readonly")
            setting_combo.grid(row=3, column=1, sticky='ew', padx=5, pady=3)
            ToolTip(setting_combo, "接触的环境场景。拥挤和密闭场所风险最高。")

            # ── 分组 3：健康状况 ──
            group3 = ttk.LabelFrame(main_frame, text="健康状况", padding=10)
            group3.grid(row=2, column=0, sticky='nsew', padx=5, pady=3)
            group3.columnconfigure(1, weight=1)
            group3.columnconfigure(3, weight=1)

            ttk.Label(group3, text="是否高危:", font=_DIALOG_FONT_DEFAULT).grid(
                row=0, column=0, sticky='e', padx=5, pady=3)
            self.is_high_risk_var = tk.StringVar(
                value=initial_data.get('is_high_risk', '否') if initial_data else '否')
            high_risk_combo = ttk.Combobox(group3, textvariable=self.is_high_risk_var,
                                           values=["否", "是"], state="readonly")
            high_risk_combo.grid(row=0, column=1, sticky='ew', padx=5, pady=3)
            ToolTip(high_risk_combo, "高危人群包括：HIV感染者、糖尿病患者、长期使用免疫抑制剂者等。")

            ttk.Label(group3, text="有无症状:", font=_DIALOG_FONT_DEFAULT).grid(
                row=0, column=2, sticky='e', padx=5, pady=3)
            self.has_symptoms_var = tk.StringVar(
                value=initial_data.get('has_symptoms', '否') if initial_data else '否')
            has_symptoms_combo = ttk.Combobox(group3, textvariable=self.has_symptoms_var,
                                              values=["否", "是"], state="readonly")
            has_symptoms_combo.grid(row=0, column=3, sticky='ew', padx=5, pady=3)
            ToolTip(has_symptoms_combo, "接触者是否有结核相关症状。有症状时风险更高。")

            ttk.Label(group3, text="卡介苗接种:", font=_DIALOG_FONT_DEFAULT).grid(
                row=1, column=0, sticky='e', padx=5, pady=3)
            self.bcg_vaccine_var = tk.StringVar(
                value=initial_data.get('bcg_vaccine', '是') if initial_data else '是')
            bcg_vaccine_combo = ttk.Combobox(group3, textvariable=self.bcg_vaccine_var,
                                             values=["否", "是"], state="readonly")
            bcg_vaccine_combo.grid(row=1, column=1, sticky='ew', padx=5, pady=3)
            ToolTip(bcg_vaccine_combo, "是否接种过卡介苗（BCG）。未接种者风险更高。")

            ttk.Label(group3, text="既往结核病史:", font=_DIALOG_FONT_DEFAULT).grid(
                row=1, column=2, sticky='e', padx=5, pady=3)
            self.has_tb_var = tk.StringVar(
                value=initial_data.get('has_tb', '否') if initial_data else '否')
            has_tb_combo = ttk.Combobox(group3, textvariable=self.has_tb_var,
                                        values=["否", "是"], state="readonly")
            has_tb_combo.grid(row=1, column=3, sticky='ew', padx=5, pady=3)
            ToolTip(has_tb_combo, "接触者是否有结核病史。有结核病史者风险更高。")

            # 既往病史 + 疾病类型（紧邻放置）
            ttk.Label(group3, text="既往病史:", font=_DIALOG_FONT_DEFAULT).grid(
                row=2, column=0, sticky='e', padx=5, pady=3)
            self.past_illness_var = tk.StringVar(
                value=initial_data.get('past_illness', '否') if initial_data else '否')
            self.illness_combo = ttk.Combobox(group3, textvariable=self.past_illness_var,
                                              values=["否", "是"], state="readonly")
            self.illness_combo.grid(row=2, column=1, sticky='ew', padx=5, pady=3)
            ToolTip(self.illness_combo, "接触者是否有其他慢性疾病史。有慢性疾病者风险更高。")

            self.illness_type_label = ttk.Label(group3, text="疾病类型:",
                                               font=_DIALOG_FONT_DEFAULT)
            self.illness_type_label.grid(row=2, column=2, sticky='e', padx=5, pady=3)
            illness_type_val = initial_data.get('past_illness_type', 'none') if initial_data else 'none'
            self.past_illness_type_var = tk.StringVar(value=illness_type_val)
            self.illness_type_combo = ttk.Combobox(group3, textvariable=self.past_illness_type_var,
                                                   values=["HIV", "糖尿病", "免疫抑制", "其他"],
                                                   state="readonly")
            self.illness_type_combo.grid(row=2, column=3, sticky='ew', padx=5, pady=3)
            self.illness_type_hint = ttk.Label(group3, text="请先选择「既往病史」为「是」",
                                              font=_DIALOG_FONT_SMALL,
                                              foreground=_DIALOG_COLORS['text_light'])
            self.illness_type_hint.grid(row=3, column=3, sticky='w', padx=5, pady=0)
            ToolTip(self.illness_type_combo, "HIV、糖尿病、免疫抑制、其他慢性病会增加感染风险。")

            # 初始状态设置
            if self.past_illness_var.get() != "是":
                self.illness_type_combo.configure(state="disabled")
                self.illness_type_label.configure(foreground=_DIALOG_COLORS['text_light'])
            else:
                self.illness_type_hint.grid_forget()

            def on_illness_change(event=None):
                if self.past_illness_var.get() == "是":
                    self.illness_type_combo.configure(state="readonly")
                    self.illness_type_label.configure(foreground=_DIALOG_COLORS['text'])
                    self.illness_type_hint.grid_forget()
                else:
                    self.illness_type_combo.configure(state="disabled")
                    self.illness_type_label.configure(foreground=_DIALOG_COLORS['text_light'])
                    self.past_illness_type_var.set("none")
                    self.illness_type_hint.grid(row=3, column=3, sticky='w', padx=5, pady=0)

            self.illness_combo.bind('<<ComboboxSelected>>', on_illness_change)

            # ── 分组 4：地域与其他 ──
            group4 = ttk.LabelFrame(main_frame, text="地域与其他", padding=10)
            group4.grid(row=2, column=1, sticky='nsew', padx=5, pady=3)
            group4.columnconfigure(1, weight=1)
            group4.columnconfigure(3, weight=1)

            ttk.Label(group4, text="民族:", font=_DIALOG_FONT_DEFAULT).grid(
                row=0, column=0, sticky='e', padx=5, pady=3)
            eth_default = initial_data.get('ethnicity', '汉族') if initial_data else '汉族'
            self.ethnicity_var = tk.StringVar(value=eth_default)
            ethnicity_combo = ttk.Combobox(group4, textvariable=self.ethnicity_var,
                                           values=["汉族", "维吾尔族", "哈萨克族", "回族", "蒙古族", "其他"],
                                           state="readonly")
            ethnicity_combo.grid(row=0, column=1, sticky='ew', padx=5, pady=3)
            ToolTip(ethnicity_combo, "接触者的民族（描述性统计变量，不参与风险计算）。")

            ttk.Label(group4, text="原籍海拔(米):", font=_DIALOG_FONT_DEFAULT).grid(
                row=0, column=2, sticky='e', padx=5, pady=3)
            alt_default = initial_data.get('origin_altitude', 300) if initial_data else 300
            self.origin_altitude_var = tk.IntVar(value=alt_default)
            altitude_spin = ttk.Spinbox(group4, from_=0, to=5000,
                                        textvariable=self.origin_altitude_var)
            altitude_spin.grid(row=0, column=3, sticky='ew', padx=5, pady=3)
            ToolTip(altitude_spin, "接触者原籍海拔高度（米）。克拉玛依约300米。")

            ttk.Label(group4, text="工作场景:", font=_DIALOG_FONT_DEFAULT).grid(
                row=1, column=0, sticky='e', padx=5, pady=3)
            wp_default = initial_data.get('workplace_type', '非油田') if initial_data else '非油田'
            self.workplace_type_var = tk.StringVar(value=wp_default)
            workplace_combo = ttk.Combobox(group4, textvariable=self.workplace_type_var,
                                           values=["非油田", "油田"], state="readonly")
            workplace_combo.grid(row=1, column=1, sticky='ew', padx=5, pady=3)
            ToolTip(workplace_combo, "接触者的工作场景。油田作业者面临特异性风险因素。")

            ttk.Label(group4, text="吸毒史:", font=_DIALOG_FONT_DEFAULT).grid(
                row=1, column=2, sticky='e', padx=5, pady=3)
            idu_default = initial_data.get('idu_status', '否') if initial_data else '否'
            self.idu_status_var = tk.StringVar(value=idu_default)
            idu_combo = ttk.Combobox(group4, textvariable=self.idu_status_var,
                                     values=["否", "是"], state="readonly")
            idu_combo.grid(row=1, column=3, sticky='ew', padx=5, pady=3)
            ToolTip(idu_combo, "接触者是否有静脉吸毒史。IDU人群结核免疫抑制风险更高。")

            ttk.Label(group4, text="所在城区:", font=_DIALOG_FONT_DEFAULT).grid(
                row=2, column=0, sticky='e', padx=5, pady=3)
            dist_default = initial_data.get('district', '克拉玛依区') if initial_data else '克拉玛依区'
            self.district_var = tk.StringVar(value=dist_default)
            district_combo = ttk.Combobox(group4, textvariable=self.district_var,
                                          values=["克拉玛依区", "独山子区", "白碱滩区", "乌尔禾区"],
                                          state="readonly")
            district_combo.grid(row=2, column=1, sticky='ew', padx=5, pady=3)
            ToolTip(district_combo, "接触者所在行政区划。")

            ttk.Label(group4, text="迁入月数:", font=_DIALOG_FONT_DEFAULT).grid(
                row=2, column=2, sticky='e', padx=5, pady=3)
            mig_default = initial_data.get('months_since_migration', 0) if initial_data else 0
            self.years_in_karamay_var = tk.IntVar(value=mig_default)
            months_spin = ttk.Spinbox(group4, from_=0, to=960,
                                      textvariable=self.years_in_karamay_var)
            months_spin.grid(row=2, column=3, sticky='ew', padx=5, pady=3)
            ToolTip(months_spin, "接触者迁入克拉玛依的月数。0表示本地出生。")

            # ── 底部按钮栏（右对齐） ──
            btn_frame = ttk.Frame(main_frame)
            btn_frame.grid(row=3, column=0, columnspan=2, sticky='e', pady=15, padx=15)
            ttk.Button(btn_frame, text="确定", command=self._on_ok, width=10).pack(side='right', padx=5)
            ttk.Button(btn_frame, text="取消", command=self._on_cancel, width=10).pack(side='right', padx=5)

            self.protocol("WM_DELETE_WINDOW", self._on_cancel)
            self.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
            self.wait_window()

        def _create_required_label(self, parent, text, row, col):
            """创建带红色*号的必填标签（*号与文本在同一容器中，不占用额外列）"""
            lbl_frame = ttk.Frame(parent)
            lbl_frame.grid(row=row, column=col, sticky='e', padx=5, pady=3)
            # 文本标签右对齐
            lbl = ttk.Label(lbl_frame, text=text, font=_DIALOG_FONT_DEFAULT)
            lbl.pack(side='right')
            # 红色星号紧贴文本左侧（视觉上在文字前）
            asterisk = tk.Label(lbl_frame, text="*", fg=_DIALOG_COLORS['required'],
                               font=_DIALOG_FONT_BOLD)
            asterisk.pack(side='right', padx=(0, 1))

        def _bind_mousewheel(self, canvas):
            """绑定鼠标滚轮事件 — 仅鼠标悬停画布区域时滚动，离开时解绑"""
            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

            def _on_linux_scroll_up(event):
                canvas.yview_scroll(-1, "units")

            def _on_linux_scroll_down(event):
                canvas.yview_scroll(1, "units")

            def _on_enter(event):
                # Windows 滚轮
                canvas.bind_all("<MouseWheel>", _on_mousewheel)
                # Linux 滚轮
                canvas.bind_all("<Button-4>", _on_linux_scroll_up)
                canvas.bind_all("<Button-5>", _on_linux_scroll_down)

            def _on_leave(event):
                canvas.unbind_all("<MouseWheel>")
                canvas.unbind_all("<Button-4>")
                canvas.unbind_all("<Button-5>")

            canvas.bind("<Enter>", _on_enter, add="+")
            canvas.bind("<Leave>", _on_leave, add="+")

            # 保存回调引用，防止被 GC 回收
            self._mousewheel_callbacks = (_on_mousewheel, _on_linux_scroll_up, _on_linux_scroll_down,
                                          _on_enter, _on_leave)

            # 对话框销毁时解绑全局事件
            def _on_destroy():
                try:
                    canvas.unbind_all("<MouseWheel>")
                    canvas.unbind_all("<Button-4>")
                    canvas.unbind_all("<Button-5>")
                except Exception as e:
                    LOGGER.debug("ContactEditDialog 销毁时解绑鼠标滚轮事件失败: %s", e)
            self.bind("<Destroy>", lambda e: _on_destroy(), add="+")

        def _on_ok(self):
            """确定按钮回调"""
            name = self.name_var.get().strip()
            if not name:
                messagebox.showwarning("输入校验", "姓名不能为空，请输入姓名。", parent=self)
                return

            try:
                age = int(self.age_var.get())
                if age < MIN_AGE or age > MAX_AGE:
                    raise ValueError
            except (ValueError, tk.TclError):
                messagebox.showwarning("输入校验", f"年龄必须在{MIN_AGE}-{MAX_AGE}之间。", parent=self)
                return

            def _safe_int(var, default=1, min_val=0, max_val=9999, field_name="数值"):
                """安全获取 Spinbox/IntVar 的整数值，失败时弹提示返回 None"""
                try:
                    val = int(var.get())
                    if val < min_val or val > max_val:
                        raise ValueError
                    return val
                except (ValueError, tk.TclError):
                    messagebox.showwarning("输入校验", f"{field_name}必须是{min_val}-{max_val}之间的整数。", parent=self)
                    return None

            ventilation = _safe_int(self.ventilation_var, default=3, min_val=MIN_VENTILATION, max_val=MAX_VENTILATION, field_name="通风条件")
            if ventilation is None:
                return
            single_duration = _safe_int(self.single_duration_var, default=60, min_val=MIN_SINGLE_DURATION_MINUTES, max_val=MAX_SINGLE_DURATION_MINUTES, field_name="单次接触时长")
            if single_duration is None:
                return
            freq_density = _safe_int(self.freq_density_var, default=3, min_val=MIN_FREQ_DENSITY, max_val=MAX_FREQ_DENSITY, field_name="每周接触频次")
            if freq_density is None:
                return
            time_span = _safe_int(self.time_span_var, default=12, min_val=MIN_TIME_SPAN_WEEKS, max_val=MAX_TIME_SPAN_WEEKS, field_name="持续周期")
            if time_span is None:
                return
            months_mig = _safe_int(self.years_in_karamay_var, default=0, min_val=0, max_val=960, field_name="迁入月数")
            if months_mig is None:
                return

            self.result = {
                'name': name,
                'age': age,
                'single_duration': single_duration,
                'freq_density': freq_density,
                'time_span': time_span,
                'has_symptoms': 1 if self.has_symptoms_var.get() == '是' else 0,
                'bcg_vaccine': 1 if self.bcg_vaccine_var.get() == '是' else 0,
                'has_tb': 1 if self.has_tb_var.get() == '是' else 0,
                'ventilation': ventilation,
                'contact_distance': self.contact_distance_var.get(),
                'exposure_setting': self.exposure_setting_var.get(),
                'past_illness': 1 if self.past_illness_var.get() == '是' else 0,
                'past_illness_type': self.past_illness_type_var.get() if self.past_illness_var.get() == '是' else 'none',
                'ethnicity': self.ethnicity_var.get(),
                'origin_altitude': self.origin_altitude_var.get(),
                'workplace_type': self.workplace_type_var.get(),
                'idu_status': self.idu_status_var.get(),
                'district': self.district_var.get(),
                'months_since_migration': months_mig
            }
            if self.contact_type == 'family':
                self.result['relationship'] = self.relationship_var.get()
            self.result['is_high_risk'] = 1 if self.is_high_risk_var.get() == '是' else 0
            self.destroy()

        def _on_cancel(self):
            """取消按钮回调"""
            self.result = None
            self.destroy()

else:
    # tkinter 不可用时，ContactEditDialog 置为 None
    ContactEditDialog = None


# =============================================================================
# 方向一：随机SEIR模型 + 贝叶斯MCMC参数推断
# 参考文献：
#   Rosato C et al. (2022) Particle-MCMC + NUTS for stochastic SEIR
#   Chakraborty T et al. (2025) EGDL: Bayesian MCMC for TB transmission
#   Dye C et al. (2005) Population dynamics of TB epidemics
# =============================================================================