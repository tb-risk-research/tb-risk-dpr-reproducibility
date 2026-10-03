#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""杂项功能 Mixin（场景切换、批量应用、克拉玛依验证、帮助）

从原 gui/tabs/__init__.py 拆分而来，包含场景与帮助相关 5 个方法。
对应原文件第 811 行至文件末尾。
"""

from .._gui_common import tk, ttk, messagebox, KARAMAY_VALIDATOR_AVAILABLE, KaramayValidator, LOGGER


class MiscTabMixin:
    """GUI 选项卡方法混合类 — 场景应用、克拉玛依验证与帮助"""

    def _on_scenario_change(self, event=None):
        """场景选择改变时的处理"""
        scenario = self.scenario_var.get()

        scenario_names = {
            'custom': '自定义模式：手动输入所有参数',
            'rural_family': '农村家庭场景：通风条件 3，接触距离近，每周接触 7 次',
            'urban_workplace': '城市职场场景：通风条件 4，接触距离中等，每周接触 5 次',
            'school': '学校场景：通风条件 3，接触距离极近，暴露场景拥挤',
            'karamay_oilfield': '克拉玛依油田场景：油田营地板房，轮班制暴露，戈壁扬尘修正'
        }

        self.scenario_desc_var.set(scenario_names.get(scenario, ''))

        # 更新当前场景标志
        if scenario == 'custom':
            self.current_scenario = None

        # 应用场景默认值到 GUI 字段
        if scenario != 'custom' and scenario in self.scenario_defaults:
            self.current_scenario = scenario
            defaults = self.scenario_defaults[scenario]['patient']

            # 更新患者基本信息的默认值
            if 'ventilation' in defaults:
                self.basic_info_vars['ventilation'].set(str(defaults['ventilation']))
            if 'family_living_conditions' in defaults:
                self.basic_info_vars['family_living_conditions'].set(str(defaults['family_living_conditions']))
            if 'flp_percentage' in defaults:
                self.basic_info_vars['flp_percentage'].set(defaults['flp_percentage'])
            if 'hrsp_percentage' in defaults:
                self.basic_info_vars['hrsp_percentage'].set(defaults['hrsp_percentage'])

            messagebox.showinfo("场景已应用",
                              f"已应用{scenario_names.get(scenario, '')}场景的默认值。\n\n"
                              f"家庭成员和社会接触者的默认值将在添加新条目时自动应用。\n"
                              "如需应用到已有条目，请点击\"应用到现有条目\"按钮。")


    def _apply_scenario_to_existing(self):
        """将当前场景的默认值应用到所有现有条目"""
        scenario = self.scenario_var.get()
        if scenario == 'custom':
            messagebox.showinfo("提示", "当前为自定义模式，无预设默认值可应用。")
            return

        if not self.family_entries and not self.social_entries:
            messagebox.showinfo("提示", "当前没有家庭成员或社会接触者条目。")
            return

        if not messagebox.askyesno("确认", f"确定要将\"{scenario}\"场景的默认值应用到所有现有条目吗？\n这将覆盖通风、接触距离、暴露场景等环境参数的当前值。\n（不会覆盖单次时长、每周频次、持续周期等接触频率参数）"):
            return

        # 获取场景默认值
        family_defaults = self.scenario_defaults.get(scenario, {}).get('family_member', {})
        social_defaults = self.scenario_defaults.get(scenario, {}).get('social_contact', {})

        # 仅覆盖环境参数，不覆盖用户已调整的接触频率参数
        environment_keys = {'ventilation', 'contact_distance', 'exposure_setting'}

        applied_count = 0

        # 应用到家庭成员
        if family_defaults:
            for entry in self.family_entries:
                for key in environment_keys:
                    if key in family_defaults:
                        if key == 'contact_distance':
                            entry['contact_distance'] = self.DISTANCE_MAPPING_REVERSE.get(family_defaults['contact_distance'], family_defaults['contact_distance'])
                        elif key == 'exposure_setting':
                            entry['exposure_setting'] = self.SETTING_MAPPING_REVERSE.get(family_defaults['exposure_setting'], family_defaults['exposure_setting'])
                        else:
                            entry[key] = str(family_defaults[key])
                applied_count += 1

        # 应用到社会接触者
        if social_defaults:
            for entry in self.social_entries:
                for key in environment_keys:
                    if key in social_defaults:
                        if key == 'contact_distance':
                            entry['contact_distance'] = self.DISTANCE_MAPPING_REVERSE.get(social_defaults['contact_distance'], social_defaults['contact_distance'])
                        elif key == 'exposure_setting':
                            entry['exposure_setting'] = self.SETTING_MAPPING_REVERSE.get(social_defaults['exposure_setting'], social_defaults['exposure_setting'])
                        else:
                            entry[key] = str(social_defaults[key])
                applied_count += 1

        # 刷新表格显示
        # 优先级六：通过 adapter 统一刷新（消除 ui_mode 分支）
        if self.adapter is not None:
            self.adapter.refresh_after_undo()

        # 清除旧的评估结果，确保数据一致性
        self.results = {}
        if hasattr(self, 'ml_results_lock'):
            with self.ml_results_lock:
                self.ml_results = {}
        else:
            self.ml_results = {}

        messagebox.showinfo("成功", f"已将场景默认值应用到 {applied_count} 个条目。\n\n旧的评估结果已清除，请重新执行风险评估。")


    def _run_karamay_validation(self):
        if not KARAMAY_VALIDATOR_AVAILABLE:
            messagebox.showwarning("不可用", "克拉玛依验证模块未安装")
            return
        try:
            self.karamay_validator = KaramayValidator(self.karamay_localizer)
            results = self.karamay_validator.run_full_validation(n_samples=5000)
            summary = self.karamay_validator.generate_summary_text()
            dlg = tk.Toplevel(self.root)
            dlg.title("克拉玛依本土化验证报告")
            dlg.geometry("700x600")
            dlg.transient(self.root)
            dlg.grab_set()
            text_frame = ttk.Frame(dlg)
            text_frame.pack(fill='both', expand=True, padx=10, pady=10)
            text = tk.Text(text_frame, wrap='word', font=('Consolas', 10))
            scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=text.yview)
            text.configure(yscrollcommand=scrollbar.set)
            text.pack(side='left', fill='both', expand=True)
            scrollbar.pack(side='right', fill='y')
            text.insert('1.0', summary)
            text.config(state='disabled')
        except Exception as e:
            messagebox.showerror("错误", f"验证分析失败: {e}")


    def _show_karamay_info(self):
        if not self.use_karamay or self.karamay_localizer is None:
            messagebox.showinfo("克拉玛依", "本土化模块未启用")
            return
        info = self.karamay_localizer.get_module_info()
        refs = self.karamay_localizer.get_all_references()
        text = f"模块: {info['name']} v{info['version']}\n状态: {'已启用' if info['enabled'] else '已禁用'}\n文献引用: {info['references_count']}篇\n\n七维度子模块:\n"
        for k, v in info['dimensions'].items():
            text += f"  {k}. {v}\n"
        text += "\n核心文献:\n" + '\n'.join(f"  - {r}" for r in refs)
        messagebox.showinfo("克拉玛依本土化模块", text)


    def _show_help(self):
        """Section X: 显示帮助窗口（独立 Toplevel，非 messagebox）

        优先委托给 BaseMixin._show_help_window，弹出可导航、可搜索的
        独立帮助窗口；若该方法不可用则回退到原始 messagebox 简版说明。
        """
        try:
            if hasattr(self, '_show_help_window'):
                self._show_help_window()
                return
        except Exception as e:
            LOGGER.debug("_show_help 委托 _show_help_window 失败，回退 messagebox: %s", e)

        # 回退：原始 messagebox 简版说明
        help_text = """结核病传播风险评估系统 - 使用说明

1. 智能场景选择（新增功能）：
   - 农村家庭：适合农村地区家庭场景，默认通风条件 3、近距离接触
   - 城市职场：适合城市办公室场景，默认通风条件 4、中等距离接触
   - 学校：适合学校场景，默认通风条件 3、极近距离接触
   - 自定义：手动输入所有参数

2. 患者基本信息：
   - 填写患者年龄、痰涂片结果、是否有空洞等基本信息
   - 选择治疗情况、症状严重程度等
   - 设置家庭居住条件和潜伏感染比例

3. 家庭成员信息：
   - 点击"添加家庭成员"按钮添加家庭成员
   - 填写姓名、年龄、关系等信息
   - 设置接触频率、症状、卡介苗接种情况等

4. 社会接触者信息：
   - 点击"添加社会接触者"按钮添加社会接触者
   - 填写相关信息，包括接触距离、通风条件等

5. CSV 批量导入（新增功能）：
   - 使用"文件"菜单中的"导出 CSV 模板"功能获取标准模板
   - 按模板填写数据后，使用"从 CSV 导入数据"功能批量导入
   - 导入时自动校验数据有效性，支持错误提示

6. 风险评估结果：
   - 点击"开始风险评估"按钮进行评估
   - 查看综合风险等级、各项指标评估结果
   - 查看潜在患者识别结果和建议

7. 数据保存/加载：
   - 使用"文件"菜单保存数据到 CSV 或从 CSV 加载

注意：所有数据仅保存在本地，不会上传到网络。
        """
        messagebox.showinfo("使用说明", help_text)
