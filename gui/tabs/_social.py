#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""社会接触者信息标签页 Mixin

从原 gui/tabs/__init__.py 拆分而来，包含社会接触者管理相关方法。
对应原文件第 586-808 行。

改进3：新增风险评分列、列显示/隐藏菜单、增强搜索筛选、按钮自然排列。
"""

from .._gui_common import ttk, messagebox, ContactEditDialog, tk, LOGGER
from ...core import calculate_cumulative_exposure
from ..undo import AddContactCommand, EditContactCommand, DeleteContactCommand


class SocialTabMixin:
    """GUI 选项卡方法混合类 — 社会接触者信息标签页"""

    SOCIAL_DEFAULT_VISIBLE_COLS = (
        'risk', 'name', 'age', 'single_duration', 'freq_density',
        'time_span', 'is_high_risk', 'has_symptoms', 'ventilation',
        'contact_distance', 'exposure_setting', 'diagnosed'
    )

    def _init_social_tab(self, parent):
        """初始化社会接触者信息标签页（使用Treeview替代grid布局）"""
        info_frame = ttk.LabelFrame(parent, text="填写说明", padding=10)
        info_frame.pack(fill='x', padx=20, pady=(6, 6))
        info_text = ("填写与患者有密切接触的社会关系（同事、朋友等） | "
                     "注意标注高危人群（HIV、糖尿病、免疫抑制等） | "
                     "拥挤和密闭场所风险显著高于一般环境 | "
                     "双击表格行可切换「是否确诊」状态")
        ttk.Label(info_frame, text=info_text, justify='left',
                 font=self._FONT_SMALL, style='Tip.TLabel').pack(anchor='w')

        # 按钮栏 — pack(side='left') 自然排列
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill='x', padx=20, pady=(0, 6))
        ttk.Button(top_frame, text="添加接触者",
                   command=self._add_social_contact).pack(side='left', padx=(0, 6))
        ttk.Button(top_frame, text="编辑",
                   command=self._edit_social_contact).pack(side='left', padx=6)
        ttk.Button(top_frame, text="删除",
                   command=self._delete_selected_social).pack(side='left', padx=6)
        ttk.Button(top_frame, text="批量删除",
                   command=self._batch_delete_social).pack(side='left', padx=6)
        ttk.Button(top_frame, text="粘贴导入",
                   command=lambda: self._show_paste_dialog('social')).pack(side='left', padx=6)

        # 增强搜索筛选栏
        filter_frame = ttk.Frame(parent)
        filter_frame.pack(fill='x', padx=20, pady=(0, 6))

        ttk.Label(filter_frame, text="搜索:",
                 font=self._FONT_DEFAULT).pack(side='left', padx=(0, 5))
        self.social_search_var = tk.StringVar()
        self.social_search_var.trace_add('write', lambda *_: self._filter_social_tree())
        search_entry = ttk.Entry(filter_frame, textvariable=self.social_search_var)
        search_entry.pack(side='left', fill='x', expand=True, padx=(0, 8))

        ttk.Label(filter_frame, text="筛选:",
                 font=self._FONT_DEFAULT).pack(side='left', padx=(0, 5))
        self.social_filter_var = tk.StringVar(value="全部")
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.social_filter_var,
                                    values=["全部", "高危", "确诊", "有症状"],
                                    state="readonly", width=10)
        filter_combo.pack(side='left', padx=(0, 5))
        filter_combo.bind('<<ComboboxSelected>>', lambda e: self._filter_social_tree())

        columns = ('risk', 'name', 'age', 'single_duration', 'freq_density',
                   'time_span', 'is_high_risk', 'has_symptoms', 'bcg_vaccine',
                   'ventilation', 'contact_distance', 'exposure_setting',
                   'has_tb', 'past_illness', 'past_illness_type',
                   'ethnicity', 'workplace_type', 'diagnosed')
        col_widths = {
            'risk': 70, 'name': 70, 'age': 45, 'single_duration': 65,
            'freq_density': 65, 'time_span': 60, 'is_high_risk': 50,
            'has_symptoms': 45, 'bcg_vaccine': 50, 'ventilation': 45,
            'contact_distance': 65, 'exposure_setting': 65, 'has_tb': 55,
            'past_illness': 45, 'past_illness_type': 65,
            'ethnicity': 50, 'workplace_type': 55, 'diagnosed': 60
        }
        col_headers = {
            'risk': '风险评分', 'name': '姓名', 'age': '年龄',
            'single_duration': '单次时长', 'freq_density': '每周频次',
            'time_span': '持续周期', 'is_high_risk': '高危',
            'has_symptoms': '症状', 'bcg_vaccine': '卡介苗',
            'ventilation': '通风', 'contact_distance': '接触距离',
            'exposure_setting': '暴露场景', 'has_tb': '既往结核',
            'past_illness': '病史', 'past_illness_type': '疾病类型',
            'ethnicity': '民族', 'workplace_type': '工作场景',
            'diagnosed': '是否确诊'
        }
        self.social_tree = self._create_treeview(parent, columns, col_widths, col_headers)
        self.social_tree.configure(selectmode='extended')
        self._register_tree_widget(self.social_tree)

        # 风险评分列颜色标签
        self.social_tree.tag_configure('risk_high', background=self.COLORS['risk_high_bg'], foreground=self.COLORS['risk_high_fg'])
        self.social_tree.tag_configure('risk_medium', background=self.COLORS['risk_medium_bg'], foreground=self.COLORS['risk_medium_fg'])
        self.social_tree.tag_configure('risk_low', background=self.COLORS['risk_low_bg'], foreground=self.COLORS['risk_low_fg'])

        self.social_tree.bind('<Double-1>', lambda e: self._toggle_social_diagnosed())
        self.social_tree.bind('<Button-3>', self._social_tree_context_menu)

        # 初始隐藏非默认列
        self._apply_social_column_visibility()

    def _social_tree_context_menu(self, event):
        """改进3：右键表头菜单 — 显示/隐藏列"""
        # 仅在点击表头区域时弹出列控制菜单
        region = self.social_tree.identify_region(event.x, event.y)
        if region != 'heading':
            return
        menu = tk.Menu(self.social_tree, tearoff=0)
        all_columns = self.social_tree['columns']
        self._social_col_vars = {}
        for col in all_columns:
            col_header = self.social_tree.heading(col, 'text')
            visible = self.social_tree.column(col, 'width') > 0
            var = tk.BooleanVar(value=visible)
            self._social_col_vars[col] = var
            menu.add_checkbutton(
                label=col_header,
                command=lambda c=col: self._toggle_social_column(c),
                variable=var
            )
        menu.add_separator()
        menu.add_command(label="显示全部列", command=self._show_all_social_columns)
        menu.add_command(label="恢复默认列", command=self._apply_social_column_visibility)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _toggle_social_column(self, col):
        """切换单列的显示/隐藏"""
        current_width = self.social_tree.column(col, 'width')
        if current_width > 0:
            self.social_tree.column(col, width=0, stretch=False)
        else:
            col_widths = {
                'risk': 70, 'name': 70, 'age': 45, 'single_duration': 65,
                'freq_density': 65, 'time_span': 60, 'is_high_risk': 50,
                'has_symptoms': 45, 'bcg_vaccine': 50, 'ventilation': 45,
                'contact_distance': 65, 'exposure_setting': 65, 'has_tb': 55,
                'past_illness': 45, 'past_illness_type': 65,
                'ethnicity': 50, 'workplace_type': 55, 'diagnosed': 60
            }
            self.social_tree.column(col, width=col_widths.get(col, 60), stretch=True)

    def _show_all_social_columns(self):
        """显示所有列"""
        for col in self.social_tree['columns']:
            col_widths = {
                'risk': 70, 'name': 70, 'age': 45, 'single_duration': 65,
                'freq_density': 65, 'time_span': 60, 'is_high_risk': 50,
                'has_symptoms': 45, 'bcg_vaccine': 50, 'ventilation': 45,
                'contact_distance': 65, 'exposure_setting': 65, 'has_tb': 55,
                'past_illness': 45, 'past_illness_type': 65,
                'ethnicity': 50, 'workplace_type': 55, 'diagnosed': 60
            }
            self.social_tree.column(col, width=col_widths.get(col, 60), stretch=True)

    def _apply_social_column_visibility(self):
        """应用默认列可见性设置"""
        for col in self.social_tree['columns']:
            if col in self.SOCIAL_DEFAULT_VISIBLE_COLS:
                continue
            self.social_tree.column(col, width=0, stretch=False)

    def _compute_social_risk_score(self, entry):
        """计算社会接触者的近似风险评分（0-100）"""
        score = 0
        try:
            age = int(entry.get('age', 30))
            if age < 5 or age > 65:
                score += 15
            if entry.get('ventilation', 3) in (1, '1'):
                score += 15
            elif entry.get('ventilation', 3) in (2, '2'):
                score += 10
            if entry.get('contact_distance', '中等') in ('极近', '近'):
                score += 15
            if entry.get('exposure_setting', '一般') in ('拥挤', '密闭'):
                score += 15
            if entry.get('is_high_risk', '否') == '是':
                score += 20
            if entry.get('has_symptoms', '否') == '是':
                score += 10
            if entry.get('has_tb', '否') == '是':
                score += 10
            cumulative = entry.get('cumulative_exposure', 0)
            if isinstance(cumulative, (int, float)) and cumulative > 1000:
                score += 10
        except Exception as e:
            LOGGER.debug("_compute_social_risk_score 计算评分异常: %s", e)
        return min(score, 100)

    def _get_social_risk_category(self, score):
        """获取风险等级"""
        if score >= 60:
            return 'risk_high', '高风险'
        elif score >= 30:
            return 'risk_medium', '中风险'
        else:
            return 'risk_low', '低风险'

    def _toggle_social_diagnosed(self):
        """切换社会接触者的是否确诊标签"""
        selected = self.social_tree.selection()
        if not selected:
            self._edit_social_contact()
            return

        idx = self.social_tree.index(selected[0])
        if idx < 0 or idx >= len(self.social_entries):
            return

        entry = self.social_entries[idx]
        current = entry.get('diagnosed', False)
        entry['diagnosed'] = not current
        contact_id = entry.get('_id', None)
        if contact_id:
            self.contact_labels[contact_id] = 1 if entry['diagnosed'] else 0
        if self.adapter is not None:
            self.adapter.refresh_social_tree()

    def _add_social_contact(self):
        """添加社会接触者条目（通过弹窗编辑对话框）"""
        if len(self.social_entries) >= self.MAX_SOCIAL_CONTACTS:
            messagebox.showwarning("警告", f"社会接触者数量已达上限（{self.MAX_SOCIAL_CONTACTS}人）")
            return

        scenario_defaults = {}
        if self.current_scenario and self.current_scenario in self.scenario_defaults:
            scenario_defaults = self.scenario_defaults[self.current_scenario].get('social_contact', {})
            if 'contact_distance' in scenario_defaults:
                scenario_defaults['contact_distance'] = self.DISTANCE_MAPPING_REVERSE.get(
                    scenario_defaults['contact_distance'], "中等")
            if 'exposure_setting' in scenario_defaults:
                scenario_defaults['exposure_setting'] = self.SETTING_MAPPING_REVERSE.get(
                    scenario_defaults['exposure_setting'], "一般")
            if 'ventilation' in scenario_defaults:
                scenario_defaults['ventilation'] = str(scenario_defaults['ventilation'])

        row_num = len(self.social_entries) + 1
        initial = {'name': f'接触者{row_num}'}

        dialog = ContactEditDialog(self.root, contact_type='social',
                                    initial_data=initial,
                                    scenario_defaults=scenario_defaults)
        if dialog.result:
            self._contact_id_counter += 1
            contact_id = f'social_{self._contact_id_counter}'
            dialog.result['_id'] = contact_id
            self.contact_labels[contact_id] = 0
            single_duration = dialog.result.get('single_duration', 1)
            freq_density = dialog.result.get('freq_density', 1)
            time_span = dialog.result.get('time_span', 1)
            cumulative_exposure = calculate_cumulative_exposure(single_duration, freq_density, time_span)
            dialog.result['cumulative_exposure'] = cumulative_exposure
            if hasattr(self, 'undo_manager'):
                self.undo_manager.execute(AddContactCommand(self, 'social', dialog.result))
            else:
                self.social_entries.append(dialog.result)
            if self.adapter is not None:
                self.adapter.refresh_social_tree()
            if hasattr(self, 'auto_save'):
                self.auto_save.mark_dirty()

    def _edit_social_contact(self):
        """编辑选中的社会接触者"""
        selected = self.social_tree.selection()
        if not selected:
            messagebox.showinfo("提示", "请先选择要编辑的社会接触者")
            return

        idx = self.social_tree.index(selected[0])
        if idx < 0 or idx >= len(self.social_entries):
            return

        dialog = ContactEditDialog(self.root, contact_type='social',
                                    initial_data=self.social_entries[idx])
        if dialog.result:
            old_entry = self.social_entries[idx]
            if '_id' in old_entry:
                dialog.result['_id'] = old_entry['_id']
            single_duration = dialog.result.get('single_duration', 1)
            freq_density = dialog.result.get('freq_density', 1)
            time_span = dialog.result.get('time_span', 1)
            cumulative_exposure = calculate_cumulative_exposure(single_duration, freq_density, time_span)
            dialog.result['cumulative_exposure'] = cumulative_exposure
            if hasattr(self, 'undo_manager'):
                self.undo_manager.execute(EditContactCommand(self, 'social', idx, dialog.result))
            else:
                self.social_entries[idx] = dialog.result
            if self.adapter is not None:
                self.adapter.refresh_social_tree()
            if hasattr(self, 'auto_save'):
                self.auto_save.mark_dirty()

    def _delete_selected_social(self):
        """删除选中的社会接触者"""
        selected = self.social_tree.selection()
        if not selected:
            messagebox.showinfo("提示", "请先选择要删除的社会接触者")
            return

        idx = self.social_tree.index(selected[0])
        if idx < 0 or idx >= len(self.social_entries):
            return

        contact_name = self.social_entries[idx].get('name', '未知')
        if not messagebox.askyesno("确认删除", f"确定要删除社会接触者\"{contact_name}\"吗？"):
            return

        contact_id = self.social_entries[idx].get('_id', None)
        if hasattr(self, 'undo_manager'):
            self.undo_manager.execute(DeleteContactCommand(self, 'social', idx))
        else:
            self.social_entries.pop(idx)
        if contact_id and contact_id in self.contact_labels:
            del self.contact_labels[contact_id]
        if self.adapter is not None:
            self.adapter.refresh_social_tree()
        if hasattr(self, 'auto_save'):
            self.auto_save.mark_dirty()

    def _update_social_treeview(self):
        """刷新社会接触者Treeview显示（含斑马纹）"""
        self.social_tree.delete(*self.social_tree.get_children())
        for idx, entry in enumerate(self.social_entries):
            illness_display = entry.get('past_illness_type', '') if entry.get('past_illness') == "是" else ''
            diagnosed = "是" if entry.get('diagnosed', False) else (
                "是" if (entry.get('_id', None) and self.contact_labels.get(entry.get('_id'), 0) == 1) else "否"
            )
            risk_score = self._compute_social_risk_score(entry)
            risk_tag, risk_text = self._get_social_risk_category(risk_score)

            values = (
                f"{risk_score}",
                entry.get('name', ''), entry.get('age', ''),
                entry.get('single_duration', ''), entry.get('freq_density', ''),
                entry.get('time_span', ''), entry.get('is_high_risk', ''),
                entry.get('has_symptoms', ''), entry.get('bcg_vaccine', ''),
                entry.get('ventilation', ''), entry.get('contact_distance', ''),
                entry.get('exposure_setting', ''), entry.get('has_tb', ''),
                entry.get('past_illness', ''), illness_display,
                entry.get('ethnicity', '汉族'), entry.get('workplace_type', '非油田'),
                diagnosed
            )
            zebra_tag = 'even' if idx % 2 == 0 else 'odd'
            self.social_tree.insert('', 'end', values=values, tags=(zebra_tag, risk_tag))

    def _delete_social_contact_by_index(self, index):
        """删除指定索引的社会接触者条目"""
        if index < 0 or index >= len(self.social_entries):
            return

        contact_name = self.social_entries[index].get('name', '未知')
        if not messagebox.askyesno("确认删除", f"确定要删除社会接触者\"{contact_name}\"吗？"):
            return

        contact_id = self.social_entries[index].get('_id', None)
        if hasattr(self, 'undo_manager'):
            self.undo_manager.execute(DeleteContactCommand(self, 'social', index))
        else:
            self.social_entries.pop(index)
        if contact_id and contact_id in self.contact_labels:
            del self.contact_labels[contact_id]
        if self.adapter is not None:
            self.adapter.refresh_social_tree()
        if hasattr(self, 'auto_save'):
            self.auto_save.mark_dirty()

    def _delete_social_contact(self):
        """删除最后一个社会接触者条目（保留为备用方法）"""
        if not self.social_entries:
            messagebox.showwarning("警告", "没有可删除的条目")
            return

        self._delete_social_contact_by_index(len(self.social_entries) - 1)

    def _filter_social_tree(self):
        """改进3：根据搜索框和筛选下拉实时过滤社会接触者 Treeview"""
        if not hasattr(self, 'social_tree'):
            return
        query = self.social_search_var.get().strip().lower() if hasattr(self, 'social_search_var') else ''
        filter_val = self.social_filter_var.get() if hasattr(self, 'social_filter_var') else '全部'

        self.social_tree.delete(*self.social_tree.get_children())
        for entry in self.social_entries:
            name = str(entry.get('name', ''))
            if query and query not in name.lower():
                continue
            if filter_val == '高危' and entry.get('is_high_risk', '否') != '是':
                continue
            if filter_val == '确诊' and not entry.get('diagnosed', False):
                diagnosed = entry.get('_id', None) and self.contact_labels.get(entry.get('_id'), 0) == 1
                if not diagnosed:
                    continue
            if filter_val == '有症状' and entry.get('has_symptoms', '否') != '是':
                continue

            illness_display = entry.get('past_illness_type', '') if entry.get('past_illness') == "是" else ''
            diagnosed = "是" if entry.get('diagnosed', False) else (
                "是" if (entry.get('_id', None) and self.contact_labels.get(entry.get('_id'), 0) == 1) else "否"
            )
            risk_score = self._compute_social_risk_score(entry)
            risk_tag, risk_text = self._get_social_risk_category(risk_score)

            values = (
                f"{risk_score}",
                entry.get('name', ''), entry.get('age', ''),
                entry.get('single_duration', ''), entry.get('freq_density', ''),
                entry.get('time_span', ''), entry.get('is_high_risk', ''),
                entry.get('has_symptoms', ''), entry.get('bcg_vaccine', ''),
                entry.get('ventilation', ''), entry.get('contact_distance', ''),
                entry.get('exposure_setting', ''), entry.get('has_tb', ''),
                entry.get('past_illness', ''), illness_display,
                entry.get('ethnicity', '汉族'), entry.get('workplace_type', '非油田'),
                diagnosed
            )
            self.social_tree.insert('', 'end', values=values, tags=(risk_tag,))

    def _batch_delete_social(self):
        """优先级十二：批量删除选中的社会接触者"""
        selected = self.social_tree.selection()
        if not selected:
            messagebox.showwarning("提示", "请先选中要删除的行（支持 Ctrl+多选）")
            return

        indices = sorted([self.social_tree.index(item) for item in selected], reverse=True)
        count = len(indices)
        if not messagebox.askyesno("确认批量删除",
                                    f"确定要删除 {count} 个社会接触者吗？\n（可通过撤销恢复）"):
            return

        for idx in indices:
            if idx < len(self.social_entries):
                contact_id = self.social_entries[idx].get('_id', None)
                if hasattr(self, 'undo_manager'):
                    self.undo_manager.execute(DeleteContactCommand(self, 'social', idx))
                else:
                    self.social_entries.pop(idx)
                if contact_id and contact_id in self.contact_labels:
                    del self.contact_labels[contact_id]

        if self.adapter is not None:
            self.adapter.refresh_social_tree()
        if hasattr(self, 'auto_save'):
            self.auto_save.mark_dirty()
        messagebox.showinfo("完成", f"已删除 {count} 个社会接触者")