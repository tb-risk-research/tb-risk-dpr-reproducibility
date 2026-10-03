#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 可视化字段映射配置面板

提供系统实施人员使用的映射配置界面：
1. 字段映射配置（源字段 → 标准字段）
2. 单位转换配置
3. 编码映射配置
4. 值范围标准化配置
5. 映射模板管理（保存/加载/复用）
"""

import datetime
import json
import logging
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any, Dict, List, Optional, Tuple

from ..health_interop.mapping import (
    FieldMappingEngine,
    UnitConverter,
    CodeMapper,
    MappingTemplateManager,
    ValueRangeNormalizer,
)

LOGGER = logging.getLogger("tb_risk.gui.mapping_config")

# 标准字段列表（供下拉选择）
STANDARD_FIELDS = [
    ("patient_name", "患者姓名", "string"),
    ("age", "年龄（岁）", "int"),
    ("gender", "性别（0=女, 1=男）", "int"),
    ("ethnicity", "民族", "string"),
    ("occupation", "职业", "string"),
    ("district", "行政区划", "string"),
    ("sputum_smear", "痰涂片结果", "string"),
    ("has_cavity", "有无空洞", "string"),
    ("active_tb", "活动性结核", "string"),
    ("treatment", "治疗状态", "string"),
    ("treatment_duration", "治疗时长（月）", "int"),
    ("cough_freq", "咳嗽频率", "int"),
    ("symptoms", "症状严重度", "string"),
    ("delay_days", "延迟就诊天数", "int"),
    ("ventilation", "通风条件", "string"),
    ("family_living_conditions", "家庭居住条件", "string"),
    ("bcg_scar", "卡痕/BCG接种", "string"),
    ("smoking_years", "吸烟年数", "int"),
    ("bmi", "BMI指数", "float"),
    ("height_cm", "身高（cm）", "float"),
    ("weight_kg", "体重（kg）", "float"),
    ("comorbidity_diabetes", "糖尿病", "string"),
    ("comorbidity_hiv", "HIV", "string"),
    ("past_tb_history", "既往结核史", "string"),
    ("symptom_cough", "咳嗽", "string"),
    ("symptom_fever", "发热", "string"),
    ("symptom_night_sweat", "盗汗", "string"),
    ("symptom_weight_loss", "体重下降", "string"),
]

# 转换类型
TRANSFORM_TYPES = [
    ("unit_convert", "单位转换"),
    ("code_map", "编码映射"),
    ("value_range", "值范围标准化"),
    ("expression", "Python表达式"),
    ("text_extract", "文本提取"),
    ("trim", "去空格"),
    ("lower", "转小写"),
    ("upper", "转大写"),
    ("default", "默认值"),
    ("round", "四舍五入"),
]


class MappingConfigDialog(tk.Toplevel):
    """字段映射配置对话框

    可视化配置医院系统字段到 tb_risk 标准特征的映射关系。
    """

    def __init__(self, parent, engine: Optional[FieldMappingEngine] = None):
        super().__init__(parent)
        self.title("字段映射配置工具")
        self.geometry("1000x700")
        self.transient(parent)
        self.grab_set()

        self.engine = engine or FieldMappingEngine()
        self.unit_converter = UnitConverter()
        self.code_mapper = CodeMapper()
        self.template_manager = MappingTemplateManager()
        self.value_normalizer = ValueRangeNormalizer()

        self._mappings: List[Dict[str, Any]] = []
        self._source_field_var = tk.StringVar()
        self._target_field_var = tk.StringVar()
        self._transform_listbox: Optional[tk.Listbox] = None
        self._mapping_tree: Optional[ttk.Treeview] = None

        self._build_ui()
        self._refresh_mapping_list()

    def _build_ui(self):
        """构建界面"""
        # 主布局：左侧映射列表，右侧配置区
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill='both', expand=True, padx=5, pady=5)

        # ========== 左侧：映射列表 ==========
        left_frame = ttk.LabelFrame(main_paned, text="已配置映射", padding=5)
        main_paned.add(left_frame, weight=1)

        # 工具栏
        tool_frame = ttk.Frame(left_frame)
        tool_frame.pack(fill='x', pady=(0, 5))

        ttk.Button(tool_frame, text="添加映射", width=12,
                   command=self._add_mapping_dialog).pack(side='left', padx=2)
        ttk.Button(tool_frame, text="删除映射", width=12,
                   command=self._delete_mapping).pack(side='left', padx=2)
        ttk.Button(tool_frame, text="清空全部", width=10,
                   command=self._clear_all).pack(side='left', padx=2)

        # 映射列表
        columns = ("source", "target", "confidence")
        self._mapping_tree = ttk.Treeview(left_frame, columns=columns,
                                          show='headings', height=20)
        self._mapping_tree.heading("source", text="源字段")
        self._mapping_tree.heading("target", text="标准字段")
        self._mapping_tree.heading("confidence", text="置信度")
        self._mapping_tree.column("source", width=150)
        self._mapping_tree.column("target", width=150)
        self._mapping_tree.column("confidence", width=60)
        self._mapping_tree.pack(fill='both', expand=True)

        scrollbar = ttk.Scrollbar(left_frame, orient='vertical',
                                  command=self._mapping_tree.yview)
        scrollbar.pack(side='right', fill='y')
        self._mapping_tree.configure(yscrollcommand=scrollbar.set)

        self._mapping_tree.bind('<Double-Button-1>', self._edit_mapping)

        # ========== 右侧：配置区 ==========
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)

        # 使用 Notebook 分页
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill='both', expand=True)

        # 页1：自动映射
        self._build_auto_map_page(notebook)

        # 页2：模板管理
        self._build_template_page(notebook)

        # 页3：编码映射配置
        self._build_code_map_page(notebook)

        # 页4：单位转换配置
        self._build_unit_convert_page(notebook)

        # 页5：值范围标准化
        self._build_value_range_page(notebook)

        # 页6：数据源拖拽映射
        self._build_drag_map_page(notebook)

        # 底部按钮
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill='x', padx=10, pady=5)
        ttk.Button(btn_frame, text="测试映射", command=self._test_mapping).pack(side='left', padx=2)
        ttk.Label(btn_frame, text="输入源数据后点击测试", font=('', 8)).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="保存配置", command=self._save_config).pack(side='right', padx=2)
        ttk.Button(btn_frame, text="取消", command=self.destroy).pack(side='right', padx=2)

    def _build_auto_map_page(self, notebook):
        """自动映射页"""
        page = ttk.Frame(notebook, padding=10)
        notebook.add(page, text="自动映射")

        ttk.Label(page, text="输入源字段名（每行一个）:",
                  font=('Microsoft YaHei', 9)).pack(anchor='w')

        text_frame = ttk.Frame(page)
        text_frame.pack(fill='both', expand=True)

        self._source_text = tk.Text(text_frame, height=8, width=40)
        self._source_text.pack(side='left', fill='both', expand=True)
        scroll = ttk.Scrollbar(text_frame, orient='vertical',
                               command=self._source_text.yview)
        scroll.pack(side='right', fill='y')
        self._source_text.configure(yscrollcommand=scroll.set)

        # 示例按钮
        example_frame = ttk.Frame(page)
        example_frame.pack(fill='x', pady=5)
        ttk.Button(example_frame, text="加载示例字段",
                   command=self._load_example_fields).pack(side='left', padx=2)
        ttk.Button(example_frame, text="自动推断映射",
                   command=self._auto_map).pack(side='left', padx=2)
        ttk.Label(example_frame, text="相似度阈值:").pack(side='left', padx=(10, 2))
        self._threshold_var = tk.DoubleVar(value=0.3)
        ttk.Spinbox(example_frame, from_=0.0, to=1.0, increment=0.05,
                    textvariable=self._threshold_var, width=5).pack(side='left')

        # 建议结果
        ttk.Label(page, text="建议映射:", font=('Microsoft YaHei', 9)).pack(anchor='w', pady=(5, 0))

        self._suggestions_tree = ttk.Treeview(
            page, columns=("source", "target", "confidence", "action"),
            show='headings', height=6)
        self._suggestions_tree.heading("source", text="源字段")
        self._suggestions_tree.heading("target", text="目标字段")
        self._suggestions_tree.heading("confidence", text="置信度")
        self._suggestions_tree.heading("action", text="操作")
        self._suggestions_tree.column("source", width=150)
        self._suggestions_tree.column("target", width=150)
        self._suggestions_tree.column("confidence", width=80)
        self._suggestions_tree.column("action", width=100)
        self._suggestions_tree.pack(fill='x', pady=5)

        btn_frame = ttk.Frame(page)
        btn_frame.pack(fill='x')
        ttk.Button(btn_frame, text="接受选中", command=self._accept_suggestion).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="接受全部", command=self._accept_all_suggestions).pack(side='left', padx=2)

    def _build_template_page(self, notebook):
        """模板管理页"""
        page = ttk.Frame(notebook, padding=10)
        notebook.add(page, text="模板管理")

        # 模板列表
        list_frame = ttk.LabelFrame(page, text="可用模板", padding=5)
        list_frame.pack(fill='both', expand=True)

        columns = ("name", "vendor", "type", "version", "mappings")
        self._template_tree = ttk.Treeview(list_frame, columns=columns,
                                           show='headings', height=8)
        self._template_tree.heading("name", text="模板名称")
        self._template_tree.heading("vendor", text="厂商")
        self._template_tree.heading("type", text="系统类型")
        self._template_tree.heading("version", text="版本")
        self._template_tree.heading("mappings", text="映射数")
        self._template_tree.column("name", width=180)
        self._template_tree.column("vendor", width=100)
        self._template_tree.column("type", width=80)
        self._template_tree.column("version", width=60)
        self._template_tree.column("mappings", width=60)
        self._template_tree.pack(fill='both', expand=True)

        btn_frame = ttk.Frame(list_frame)
        btn_frame.pack(fill='x', pady=5)
        ttk.Button(btn_frame, text="应用模板", command=self._apply_template).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="保存为模板", command=self._save_as_template).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="导入模板", command=self._import_template).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="导出模板", command=self._export_template).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="刷新", command=self._refresh_template_list).pack(side='left', padx=2)

        self._refresh_template_list()

    def _build_code_map_page(self, notebook):
        """编码映射配置页"""
        page = ttk.Frame(notebook, padding=10)
        notebook.add(page, text="编码映射")

        # 映射表选择
        select_frame = ttk.Frame(page)
        select_frame.pack(fill='x', pady=5)

        ttk.Label(select_frame, text="选择映射表:").pack(side='left')
        self._code_map_var = tk.StringVar()
        map_choices = [m["id"] + " - " + m["name"]
                       for m in self.code_mapper.get_available_maps()]
        self._code_map_combo = ttk.Combobox(select_frame, textvariable=self._code_map_var,
                                            values=map_choices, width=30, state='readonly')
        self._code_map_combo.pack(side='left', padx=5)
        self._code_map_combo.bind('<<ComboboxSelected>>', self._on_code_map_selected)

        # 映射条目列表
        self._code_entries_tree = ttk.Treeview(
            page, columns=("source", "target"), show='headings', height=10)
        self._code_entries_tree.heading("source", text="源编码")
        self._code_entries_tree.heading("target", text="目标编码")
        self._code_entries_tree.pack(fill='both', expand=True, pady=5)

        # 添加条目
        add_frame = ttk.LabelFrame(page, text="添加映射条目", padding=5)
        add_frame.pack(fill='x', pady=5)

        ttk.Label(add_frame, text="源编码:").grid(row=0, column=0, padx=5, pady=2)
        self._code_source_var = tk.StringVar()
        ttk.Entry(add_frame, textvariable=self._code_source_var, width=20).grid(row=0, column=1, padx=5)

        ttk.Label(add_frame, text="目标编码:").grid(row=0, column=2, padx=5, pady=2)
        self._code_target_var = tk.StringVar()
        ttk.Entry(add_frame, textvariable=self._code_target_var, width=20).grid(row=0, column=3, padx=5)

        ttk.Button(add_frame, text="添加", command=self._add_code_entry).grid(row=0, column=4, padx=5)
        ttk.Button(add_frame, text="删除选中", command=self._delete_code_entry).grid(row=0, column=5, padx=5)

    def _build_unit_convert_page(self, notebook):
        """单位转换配置页"""
        page = ttk.Frame(notebook, padding=10)
        notebook.add(page, text="单位转换")

        # 可用规则列表
        list_frame = ttk.LabelFrame(page, text="可用转换规则", padding=5)
        list_frame.pack(fill='both', expand=True)

        columns = ("name", "from", "to", "formula", "precision")
        self._unit_tree = ttk.Treeview(list_frame, columns=columns,
                                       show='headings', height=10)
        self._unit_tree.heading("name", text="规则名称")
        self._unit_tree.heading("from", text="源单位")
        self._unit_tree.heading("to", text="目标单位")
        self._unit_tree.heading("formula", text="公式")
        self._unit_tree.heading("precision", text="精度")
        self._unit_tree.column("name", width=180)
        self._unit_tree.column("from", width=80)
        self._unit_tree.column("to", width=80)
        self._unit_tree.column("formula", width=150)
        self._unit_tree.column("precision", width=60)
        self._unit_tree.pack(fill='both', expand=True)

        # 填充规则
        for rule in self.unit_converter.get_available_rules():
            self._unit_tree.insert("", "end", values=(
                rule.get("name", ""),
                rule.get("from_unit", ""),
                rule.get("to_unit", ""),
                rule.get("formula", ""),
                rule.get("precision", 2),
            ))

        # 测试转换
        test_frame = ttk.LabelFrame(page, text="测试转换", padding=5)
        test_frame.pack(fill='x', pady=5)

        ttk.Label(test_frame, text="值:").pack(side='left')
        self._convert_value_var = tk.StringVar()
        ttk.Entry(test_frame, textvariable=self._convert_value_var, width=10).pack(side='left', padx=2)

        ttk.Label(test_frame, text="规则ID:").pack(side='left', padx=(10, 0))
        self._convert_rule_var = tk.StringVar()
        rule_ids = [r["id"] for r in self.unit_converter.get_available_rules()]
        ttk.Combobox(test_frame, textvariable=self._convert_rule_var,
                     values=rule_ids, width=25).pack(side='left', padx=2)

        ttk.Button(test_frame, text="转换", command=self._test_unit_convert).pack(side='left', padx=5)
        self._convert_result_label = ttk.Label(test_frame, text="", font=('', 9, 'bold'))
        self._convert_result_label.pack(side='left', padx=5)

    def _build_value_range_page(self, notebook):
        """值范围标准化页"""
        page = ttk.Frame(notebook, padding=10)
        notebook.add(page, text="值范围标准化")

        # 标准范围列表
        list_frame = ttk.LabelFrame(page, text="标准参考范围", padding=5)
        list_frame.pack(fill='both', expand=True)

        columns = ("code", "name", "unit", "normal_range", "interpretation")
        self._range_tree = ttk.Treeview(list_frame, columns=columns,
                                        show='headings', height=10)
        self._range_tree.heading("code", text="检验项目")
        self._range_tree.heading("name", text="名称")
        self._range_tree.heading("unit", text="单位")
        self._range_tree.heading("normal_range", text="正常范围")
        self._range_tree.heading("interpretation", text="解释方式")
        self._range_tree.column("code", width=120)
        self._range_tree.column("name", width=150)
        self._range_tree.column("unit", width=80)
        self._range_tree.column("normal_range", width=150)
        self._range_tree.column("interpretation", width=80)
        self._range_tree.pack(fill='both', expand=True)

        # 填充数据
        for code, info in self.value_normalizer.get_all_standard_ranges().items():
            normal = info.get("normal", "")
            if isinstance(normal, dict):
                normal = f"{normal.get('low', '?')} - {normal.get('high', '?')}"
            self._range_tree.insert("", "end", values=(
                code,
                info.get("name", ""),
                info.get("standard_unit", ""),
                str(normal),
                info.get("interpretation", ""),
            ))

        # 测试标准化
        test_frame = ttk.LabelFrame(page, text="测试标准化", padding=5)
        test_frame.pack(fill='x', pady=5)

        ttk.Label(test_frame, text="检验项目:").pack(side='left')
        test_codes = list(self.value_normalizer.get_all_standard_ranges().keys())
        self._range_test_code_var = tk.StringVar()
        ttk.Combobox(test_frame, textvariable=self._range_test_code_var,
                     values=test_codes, width=20).pack(side='left', padx=2)

        ttk.Label(test_frame, text="值:").pack(side='left', padx=(10, 0))
        self._range_test_value_var = tk.StringVar()
        ttk.Entry(test_frame, textvariable=self._range_test_value_var, width=10).pack(side='left', padx=2)

        ttk.Button(test_frame, text="测试", command=self._test_range_normalize).pack(side='left', padx=5)
        self._range_result_label = ttk.Label(test_frame, text="", font=('', 9, 'bold'))
        self._range_result_label.pack(side='left', padx=5)

    # ==================== 数据源拖拽映射 ====================

    def _build_drag_map_page(self, notebook):
        """数据源字段拖拽映射页

        实施人员将左侧"医院源字段"拖拽到右侧"tb_risk 标准字段"，即可
        生成字段映射，无需修改代码。支持：
        - 添加/删除源字段
        - 从源字段拖拽到标准字段生成映射
        - 自动匹配辅助
        """
        page = ttk.Frame(notebook, padding=10)
        notebook.add(page, text="数据源映射（拖拽）")

        hint = ttk.Label(
            page,
            text="在左侧医院源字段上按下鼠标，拖拽到右侧标准字段上释放，即生成一条映射。",
            font=('Microsoft YaHei', 9), foreground='#555555')
        hint.pack(anchor='w', pady=(0, 6))

        paned = ttk.PanedWindow(page, orient=tk.HORIZONTAL)
        paned.pack(fill='both', expand=True)

        # ---------- 左侧：医院源字段 ----------
        left = ttk.LabelFrame(paned, text="医院系统源字段", padding=5)
        paned.add(left, weight=1)

        add_frame = ttk.Frame(left)
        add_frame.pack(fill='x', pady=(0, 4))
        self._drag_source_var = tk.StringVar()
        ttk.Entry(add_frame, textvariable=self._drag_source_var,
                  width=22).pack(side='left', padx=(0, 3))
        ttk.Button(add_frame, text="添加", width=6,
                   command=self._drag_add_source).pack(side='left', padx=2)
        ttk.Button(add_frame, text="删除", width=6,
                   command=self._drag_del_source).pack(side='left', padx=2)

        self._source_list = tk.Listbox(left, height=18)
        source_scroll = ttk.Scrollbar(left, orient='vertical',
                                      command=self._source_list.yview)
        self._source_list.configure(yscrollcommand=source_scroll.set)
        self._source_list.pack(side='left', fill='both', expand=True)
        source_scroll.pack(side='right', fill='y')

        # 拖拽起始
        self._source_list.bind('<Button-1>', self._on_drag_start)
        self._source_list.bind('<B1-Motion>', self._on_drag_motion)

        # ---------- 右侧：标准字段 ----------
        right = ttk.LabelFrame(paned, text="tb_risk 标准字段", padding=5)
        paned.add(right, weight=1)

        # 自动匹配源字段到标准字段
        auto_frame = ttk.Frame(right)
        auto_frame.pack(fill='x', pady=(0, 4))
        ttk.Button(auto_frame, text="自动匹配全部源字段",
                   command=self._drag_auto_map).pack(side='left', padx=2)

        self._target_list = tk.Listbox(right, height=18)
        target_scroll = ttk.Scrollbar(right, orient='vertical',
                                      command=self._target_list.yview)
        self._target_list.configure(yscrollcommand=target_scroll.set)
        self._target_list.pack(side='left', fill='both', expand=True)
        target_scroll.pack(side='right', fill='y')

        # 目标列表：仅允许拖放（释放时生成映射）
        self._target_list.bind('<ButtonRelease-1>', self._on_drag_drop)

        # 填充标准字段
        for f in STANDARD_FIELDS:
            label = f"{f[0]}  ({f[1]})"
            self._target_list.insert("end", label)

        # 状态标签
        self._drag_status_var = tk.StringVar(value="就绪")
        status = ttk.Label(page, textvariable=self._drag_status_var,
                           font=('', 9, 'bold'), foreground='#006600')
        status.pack(anchor='w', pady=(6, 0))

    def _drag_add_source(self):
        """添加医院源字段到左侧列表"""
        val = self._drag_source_var.get().strip()
        if not val:
            return
        self._source_list.insert("end", val)
        self._drag_source_var.set("")
        self._drag_status_var.set(f"已添加源字段: {val}")

    def _drag_del_source(self):
        """删除选中的医院源字段"""
        sel = self._source_list.curselection()
        for idx in reversed(sel):
            self._source_list.delete(idx)
        if sel:
            self._drag_status_var.set("已删除源字段")

    def _on_drag_start(self, event):
        """记录拖拽起始位置（源字段）"""
        index = self._source_list.nearest(event.y)
        if index < 0:
            return
        self._drag_index = index
        self._drag_value = self._source_list.get(index)

    def _on_drag_motion(self, event):
        """拖拽中——选中目标行并高亮"""
        index = self._target_list.nearest(event.y)
        if 0 <= index < self._target_list.size():
            self._target_list.selection_clear(0, "end")
            self._target_list.selection_set(index)

    def _on_drag_drop(self, event):
        """在目标列表释放——生成 源字段→标准字段 映射"""
        src = getattr(self, '_drag_value', None)
        if not src:
            return
        index = self._target_list.nearest(event.y)
        if not (0 <= index < self._target_list.size()):
            return
        target_raw = self._target_list.get(index)
        target = target_raw.split("  ")[0].strip()

        self.engine.add_mapping(source_field=src, target_field=target,
                                description="拖拽映射")
        self._drag_status_var.set(f"已生成映射: {src} → {target}")
        self._refresh_mapping_list()
        self._drag_value = None

    def _drag_auto_map(self):
        """对左侧所有源字段执行自动匹配，并添加到引擎"""
        source_fields = list(self._source_list.get(0, "end"))
        source_fields = [f for f in source_fields if f.strip()]
        if not source_fields:
            self._drag_status_var.set("左侧无源字段可匹配")
            return
        suggestions = self.engine.auto_map(source_fields)
        count = 0
        for s in suggestions:
            self.engine.add_mapping(
                source_field=s["source_field"],
                target_field=s["target_field"],
                confidence=s["confidence"],
                description="自动匹配",
            )
            count += 1
        self._refresh_mapping_list()
        self._drag_status_var.set(f"自动匹配完成，生成 {count} 条映射")

    # ==================== 映射管理 ====================

    def _add_mapping_dialog(self):
        """添加映射对话框"""
        dialog = tk.Toplevel(self)
        dialog.title("添加字段映射")
        dialog.geometry("500x400")
        dialog.transient(self)
        dialog.grab_set()

        # 源字段
        ttk.Label(dialog, text="源字段名（医院系统字段）:",
                  font=('Microsoft YaHei', 9)).pack(anchor='w', padx=10, pady=(10, 2))
        source_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=source_var, width=40).pack(fill='x', padx=10)

        # 目标字段
        ttk.Label(dialog, text="目标字段（tb_risk标准字段）:",
                  font=('Microsoft YaHei', 9)).pack(anchor='w', padx=10, pady=(10, 2))
        target_var = tk.StringVar()
        target_combo = ttk.Combobox(dialog, textvariable=target_var,
                                    values=[f"{f[0]} - {f[1]}" for f in STANDARD_FIELDS],
                                    width=40, state='normal')
        target_combo.pack(fill='x', padx=10)

        # 描述
        ttk.Label(dialog, text="描述:",
                  font=('Microsoft YaHei', 9)).pack(anchor='w', padx=10, pady=(10, 2))
        desc_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=desc_var, width=40).pack(fill='x', padx=10)

        # 默认值
        ttk.Label(dialog, text="默认值（源字段缺失时使用）:",
                  font=('Microsoft YaHei', 9)).pack(anchor='w', padx=10, pady=(10, 2))
        default_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=default_var, width=40).pack(fill='x', padx=10)

        # 转换配置
        ttk.Label(dialog, text="转换规则（可选，一行一个）:",
                  font=('Microsoft YaHei', 9)).pack(anchor='w', padx=10, pady=(10, 2))
        transform_text = tk.Text(dialog, height=5)
        transform_text.pack(fill='both', expand=True, padx=10, pady=5)
        transform_text.insert("1.0", '{"type": "trim"}\n{"type": "code_map", "mapping": {"源值": "目标值"}}')

        # 按钮
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill='x', padx=10, pady=10)
        ttk.Button(btn_frame, text="确定", command=lambda: self._confirm_mapping(
            dialog, source_var, target_var, desc_var, default_var, transform_text
        )).pack(side='right', padx=2)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side='right', padx=2)

    def _confirm_mapping(self, dialog, source_var, target_var,
                          desc_var, default_var, transform_text):
        """确认添加映射"""
        source = source_var.get().strip()
        target_raw = target_var.get().strip()

        if not source:
            messagebox.showwarning("警告", "请输入源字段名")
            return

        # 解析目标字段
        target = target_raw.split(" - ")[0] if " - " in target_raw else target_raw

        # 解析转换规则
        transforms = []
        raw_text = transform_text.get("1.0", "end-1c").strip()
        if raw_text:
            for line in raw_text.split("\n"):
                line = line.strip()
                if line:
                    try:
                        t = json.loads(line)
                        transforms.append(t)
                    except json.JSONDecodeError:
                        pass

        # 添加映射
        self.engine.add_mapping(
            source_field=source,
            target_field=target,
            transforms=transforms,
            default_value=default_var.get().strip() or None,
            description=desc_var.get().strip(),
        )

        self._refresh_mapping_list()
        dialog.destroy()
        messagebox.showinfo("成功", f"映射 '{source} → {target}' 已添加")

    def _delete_mapping(self):
        """删除选中映射"""
        selection = self._mapping_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要删除的映射")
            return

        for item in selection:
            values = self._mapping_tree.item(item, "values")
            source = values[0]
            self.engine.remove_mapping(source)

        self._refresh_mapping_list()

    def _clear_all(self):
        """清空所有映射"""
        if messagebox.askyesno("确认", "确定要清空所有映射配置吗？"):
            self.engine.clear()
            self._mappings.clear()
            self._refresh_mapping_list()

    def _edit_mapping(self, event):
        """双击编辑映射"""
        selection = self._mapping_tree.selection()
        if not selection:
            return
        item = selection[0]
        values = self._mapping_tree.item(item, "values")
        source = values[0]
        messagebox.showinfo("映射详情", f"源字段: {values[0]}\n目标字段: {values[1]}\n置信度: {values[2]}")

    def _refresh_mapping_list(self):
        """刷新映射列表"""
        if not self._mapping_tree:
            return
        for item in self._mapping_tree.get_children():
            self._mapping_tree.delete(item)

        for mapping in self.engine.get_all_mappings():
            self._mapping_tree.insert("", "end", values=(
                mapping.source_field,
                mapping.target_field,
                mapping.confidence,
            ))

    # ==================== 自动映射 ====================

    def _load_example_fields(self):
        """加载示例字段"""
        examples = """患者姓名
年龄
性别
民族
职业
联系电话
现住址
吸烟史
身高
体重
诊断编码
检验结果
影像所见"""
        self._source_text.delete("1.0", "end")
        self._source_text.insert("1.0", examples)

    def _auto_map(self):
        """自动推断映射"""
        raw = self._source_text.get("1.0", "end-1c").strip()
        if not raw:
            messagebox.showwarning("警告", "请先输入源字段名")
            return

        source_fields = [f.strip() for f in raw.split("\n") if f.strip()]
        threshold = self._threshold_var.get()

        suggestions = self.engine.auto_map(source_fields, threshold)

        # 清空建议列表
        for item in self._suggestions_tree.get_children():
            self._suggestions_tree.delete(item)

        for s in suggestions:
            self._suggestions_tree.insert("", "end", values=(
                s["source_field"],
                s["target_field"],
                s["confidence"],
                "接受" if s["confidence"] > 0.5 else "确认",
            ))

        messagebox.showinfo("完成", f"自动匹配完成，找到 {len(suggestions)} 个建议映射")

    def _accept_suggestion(self):
        """接受选中的建议映射"""
        selection = self._suggestions_tree.selection()
        count = 0
        for item in selection:
            values = self._suggestions_tree.item(item, "values")
            source, target = values[0], values[1]
            self.engine.add_mapping(source_field=source, target_field=target)
            count += 1

        self._suggestions_tree.delete(*selection)
        self._refresh_mapping_list()
        messagebox.showinfo("成功", f"已接受 {count} 个映射")

    def _accept_all_suggestions(self):
        """接受所有建议映射"""
        count = 0
        for item in self._suggestions_tree.get_children():
            values = self._suggestions_tree.item(item, "values")
            source, target = values[0], values[1]
            self.engine.add_mapping(source_field=source, target_field=target)
            count += 1

        for item in self._suggestions_tree.get_children():
            self._suggestions_tree.delete(item)
        self._refresh_mapping_list()
        messagebox.showinfo("成功", f"已接受全部 {count} 个映射")

    # ==================== 模板管理 ====================

    def _refresh_template_list(self):
        """刷新模板列表"""
        if not hasattr(self, '_template_tree'):
            return
        for item in self._template_tree.get_children():
            self._template_tree.delete(item)

        for t in self.template_manager.list_templates():
            self._template_tree.insert("", "end", values=(
                t.get("name", ""),
                t.get("vendor", ""),
                t.get("system_type", ""),
                t.get("version", "1.0"),
                t.get("mapping_count", 0),
            ))

    def _apply_template(self):
        """应用模板"""
        selection = self._template_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请先选择模板")
            return

        values = self._template_tree.item(selection[0], "values")
        template_name = values[0]

        # 查找模板ID
        template_id = ""
        for t in self.template_manager.list_templates():
            if t.get("name") == template_name:
                template_id = t["id"]
                break

        if not template_id:
            messagebox.showerror("错误", "模板未找到")
            return

        count = self.template_manager.apply_template(template_id, self.engine)
        self._refresh_mapping_list()
        messagebox.showinfo("成功", f"已应用模板 '{template_name}'，添加 {count} 个映射")

    def _save_as_template(self):
        """保存为模板"""
        if not self.engine.get_all_mappings():
            messagebox.showwarning("警告", "当前没有映射配置可保存")
            return

        dialog = tk.Toplevel(self)
        dialog.title("保存映射模板")
        dialog.geometry("400x300")
        dialog.transient(self)
        dialog.grab_set()

        ttk.Label(dialog, text="模板名称:", font=('Microsoft YaHei', 9)).pack(anchor='w', padx=10, pady=(10, 2))
        name_var = tk.StringVar(value="医院HIS映射模板")
        ttk.Entry(dialog, textvariable=name_var, width=35).pack(fill='x', padx=10)

        ttk.Label(dialog, text="厂商名称:", font=('Microsoft YaHei', 9)).pack(anchor='w', padx=10, pady=(10, 2))
        vendor_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=vendor_var, width=35).pack(fill='x', padx=10)

        ttk.Label(dialog, text="系统类型:", font=('Microsoft YaHei', 9)).pack(anchor='w', padx=10, pady=(10, 2))
        type_var = tk.StringVar(value="HIS")
        ttk.Combobox(dialog, textvariable=type_var,
                     values=["HIS", "LIS", "PACS", "EMR", "其他"],
                     width=33).pack(fill='x', padx=10)

        ttk.Label(dialog, text="版本:", font=('Microsoft YaHei', 9)).pack(anchor='w', padx=10, pady=(10, 2))
        version_var = tk.StringVar(value="1.0")
        ttk.Entry(dialog, textvariable=version_var, width=35).pack(fill='x', padx=10)

        def do_save():
            name = name_var.get().strip()
            vendor = vendor_var.get().strip()
            sys_type = type_var.get().strip()
            version = version_var.get().strip()
            template_id = name.lower().replace(" ", "_").replace("-", "_")

            self.template_manager.save_from_engine(
                template_id=template_id,
                name=name,
                engine=self.engine,
                vendor=vendor,
                system_type=sys_type,
                version=version,
                overwrite=True,
            )
            self._refresh_template_list()
            dialog.destroy()
            messagebox.showinfo("成功", f"模板 '{name}' 已保存")

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill='x', padx=10, pady=15)
        ttk.Button(btn_frame, text="保存", command=do_save).pack(side='right', padx=2)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side='right', padx=2)

    def _import_template(self):
        """导入模板"""
        file_path = filedialog.askopenfilename(
            title="导入映射模板",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
        )
        if file_path:
            template_id = self.template_manager.import_template_from_file(file_path)
            if template_id:
                self._refresh_template_list()
                messagebox.showinfo("成功", f"模板已导入: {template_id}")
            else:
                messagebox.showerror("错误", "导入失败，请检查文件格式")

    def _export_template(self):
        """导出模板"""
        selection = self._template_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要导出的模板")
            return

        values = self._template_tree.item(selection[0], "values")
        template_name = values[0]

        template_id = ""
        for t in self.template_manager.list_templates():
            if t.get("name") == template_name:
                template_id = t["id"]
                break

        if not template_id:
            return

        file_path = filedialog.asksaveasfilename(
            title="导出映射模板",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json")],
            initialfile=f"{template_id}.json",
        )
        if file_path:
            success = self.template_manager.export_template_to_file(template_id, file_path)
            if success:
                messagebox.showinfo("成功", "模板已导出")
            else:
                messagebox.showerror("错误", "导出失败")

    # ==================== 编码映射 ====================

    def _on_code_map_selected(self, event=None):
        """编码映射表选中"""
        selected = self._code_map_combo.get()
        map_id = selected.split(" - ")[0]

        for item in self._code_entries_tree.get_children():
            self._code_entries_tree.delete(item)

        map_data = self.code_mapper.get_map(map_id)
        if map_data:
            mapping = map_data.get("mapping", {})
            for source, target in mapping.items():
                self._code_entries_tree.insert("", "end", values=(source, target))

    def _add_code_entry(self):
        """添加编码映射条目"""
        selected = self._code_map_combo.get()
        if not selected:
            messagebox.showwarning("警告", "请先选择映射表")
            return
        map_id = selected.split(" - ")[0]

        source = self._code_source_var.get().strip()
        target = self._code_target_var.get().strip()
        if not source or not target:
            messagebox.showwarning("警告", "请输入源编码和目标编码")
            return

        self.code_mapper.add_entry(map_id, source, target)
        self._on_code_map_selected()
        self._code_source_var.set("")
        self._code_target_var.set("")
        messagebox.showinfo("成功", f"添加映射: {source} → {target}")

    def _delete_code_entry(self):
        """删除编码映射条目"""
        selection = self._code_entries_tree.selection()
        if not selection:
            return

        selected = self._code_map_combo.get()
        map_id = selected.split(" - ")[0]

        for item in selection:
            values = self._code_entries_tree.item(item, "values")
            source = values[0]
            self.code_mapper.remove_entry(map_id, source)

        self._on_code_map_selected()

    # ==================== 测试功能 ====================

    def _test_unit_convert(self):
        """测试单位转换"""
        value = self._convert_value_var.get().strip()
        rule_id = self._convert_rule_var.get().strip()

        if not value or not rule_id:
            self._convert_result_label.config(text="请输入值和规则")
            return

        result = self.unit_converter.convert(value, rule_id)
        if result:
            text = f"{value} → {result['value']} {result['unit']}"
            self._convert_result_label.config(text=text, foreground='green')
        else:
            self._convert_result_label.config(text="转换失败", foreground='red')

    def _test_range_normalize(self):
        """测试值范围标准化"""
        test_code = self._range_test_code_var.get().strip()
        value = self._range_test_value_var.get().strip()

        if not test_code or not value:
            self._range_result_label.config(text="请输入检验项目和值")
            return

        result = self.value_normalizer.normalize(value, test_code)
        if result:
            text = f"{value} → {result['interpretation']} (严重度: {result['severity']})"
            self._range_result_label.config(text=text, foreground='green')
        else:
            self._range_result_label.config(text="标准化失败", foreground='red')

    def _test_mapping(self):
        """测试映射功能"""
        dialog = tk.Toplevel(self)
        dialog.title("测试映射")
        dialog.geometry("450x350")
        dialog.transient(self)
        dialog.grab_set()

        ttk.Label(dialog, text="输入测试数据（JSON格式）:",
                  font=('Microsoft YaHei', 9)).pack(anchor='w', padx=10, pady=(10, 2))

        text_widget = tk.Text(dialog, height=8)
        text_widget.pack(fill='both', expand=True, padx=10, pady=5)
        text_widget.insert("1.0", '{"患者姓名": "张三", "年龄": "35", "性别": "男"}')

        result_text = tk.Text(dialog, height=6, foreground='#006600')
        result_text.pack(fill='both', expand=True, padx=10, pady=5)

        def do_test():
            raw = text_widget.get("1.0", "end-1c").strip()
            try:
                test_data = json.loads(raw)
                result = self.engine.apply(test_data)
                result_text.delete("1.0", "end")
                result_text.insert("1.0", json.dumps(result, ensure_ascii=False, indent=2))

                # 显示警告和错误
                errors = self.engine.get_errors()
                warnings = self.engine.get_warnings()
                if errors:
                    result_text.insert("end", "\n\n错误:\n" + json.dumps(errors, ensure_ascii=False, indent=2))
                if warnings:
                    result_text.insert("end", "\n\n警告:\n" + json.dumps(warnings, ensure_ascii=False, indent=2))
            except Exception as e:
                result_text.delete("1.0", "end")
                result_text.insert("1.0", f"测试失败: {e}")

        ttk.Button(dialog, text="执行测试", command=do_test).pack(pady=5)

    def _save_config(self):
        """保存配置并关闭"""
        # 返回当前引擎供调用方使用
        self.result = self.engine
        self.destroy()

    def get_engine(self) -> FieldMappingEngine:
        """获取配置好的映射引擎"""
        return getattr(self, 'result', self.engine)


def show_mapping_config(parent=None) -> Optional[FieldMappingEngine]:
    """显示映射配置对话框的便捷函数

    参数：
        parent: 父窗口

    返回：
        FieldMappingEngine | None: 配置好的映射引擎
    """
    import tkinter as tk
    root = parent or tk.Tk()
    dialog = MappingConfigDialog(root)
    root.wait_window(dialog)
    return dialog.get_engine()