#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反事实干预分析 Mixin（_CounterfactualMixin）

基于She et al. (2024)的反事实分析框架，提供个体化干预建议。
"""

from .._shared import *  # noqa: F401,F403


class _CounterfactualMixin:
    """反事实干预分析方法"""

    def _show_counterfactual_analysis(self):
        """显示反事实干预分析对话框

        基于She et al. (2024)的反事实分析框架，提供个体化干预建议。
        """
        if not self.results or self.ml_predictor is None or not self.ml_predictor.is_trained:
            messagebox.showwarning("警告", "请先执行风险评估")
            return

        try:
            dialog = tk.Toplevel(self.root)
            dialog.title("反事实干预分析")
            # Section IX: 扩大窗口以容纳并排的柱状图
            dialog.geometry("1100x620")
            dialog.transient(self.root)
            dialog.grab_set()

            ttk.Label(dialog, text="选择接触者进行干预分析:",
                     font=('Arial', 11, 'bold')).pack(padx=10, pady=(10, 5), anchor='w')

            contact_var = tk.StringVar()
            contact_combo = ttk.Combobox(dialog, textvariable=contact_var, width=40, state='readonly')

            contact_options = []
            potential_patients = self.results.get('potential_patients', {})
            family_list = potential_patients.get('family', [])
            social_list = potential_patients.get('social', [])

            if family_list:
                for item in family_list:
                    contact_options.append(f"[家庭] {item.get('name', '未知')}")
            if social_list:
                for item in social_list:
                    contact_options.append(f"[社会] {item.get('name', '未知')}")

            contact_combo['values'] = contact_options
            contact_combo.pack(padx=10, pady=5, fill='x')

            # Section IX: 中间容器，左侧文本结果 + 右侧柱状图并排
            content_frame = ttk.Frame(dialog)
            content_frame.pack(fill='both', expand=True, padx=10, pady=5)

            # 左侧：文本结果区（保持原有逻辑）
            text_side = ttk.Frame(content_frame)
            text_side.pack(side='left', fill='both', expand=True, padx=(0, 5))
            result_text = tk.Text(text_side, wrap='word', height=20, font=('Arial', 9))
            result_scroll = ttk.Scrollbar(text_side, orient='vertical', command=result_text.yview)
            result_text.configure(yscrollcommand=result_scroll.set)
            result_text.pack(side='left', fill='both', expand=True)
            result_scroll.pack(side='right', fill='y')

            # 右侧：matplotlib 柱状图区
            chart_side = ttk.LabelFrame(content_frame, text="干预方案风险降低对比", padding=5)
            chart_side.pack(side='right', fill='both', expand=True, padx=(5, 0))
            cf_figure = None
            cf_canvas = None
            if MATPLOTLIB_AVAILABLE:
                cf_figure = Figure(figsize=(5, 5), dpi=100)
                cf_canvas = FigureCanvasTkAgg(cf_figure, master=chart_side)
                cf_canvas.get_tk_widget().pack(fill='both', expand=True)
            else:
                ttk.Label(chart_side, text="matplotlib 未安装，无法显示柱状图",
                          foreground='gray').pack(padx=10, pady=20)

            def analyze_counterfactual():
                selected = contact_var.get()
                if not selected:
                    messagebox.showwarning("提示", "请选择一个接触者")
                    return

                is_family = selected.startswith('[家庭]')
                name = selected.replace('[家庭] ', '').replace('[社会] ', '')

                contact_type = 'family' if is_family else 'social'
                contact_data = None
                entry_data = None

                # 重新获取数据（确保闭包访问正确）
                potential_patients_inner = self.results.get('potential_patients', {})
                family_list_inner = potential_patients_inner.get('family', [])
                social_list_inner = potential_patients_inner.get('social', [])

                if is_family:
                    for entry in self.family_entries:
                        if entry.get('name') == name:
                            entry_data = entry
                            break
                    for item in family_list_inner:
                        if item.get('name') == name:
                            contact_data = item
                            break
                else:
                    for entry in self.social_entries:
                        if entry.get('name') == name:
                            entry_data = entry
                            break
                    for item in social_list_inner:
                        if item.get('name') == name:
                            contact_data = item
                            break

                if not contact_data or not entry_data:
                    messagebox.showerror("错误", "无法获取接触者数据")
                    return

                original_prob = contact_data.get('disease_probability', 0)

                # 构建干预场景
                interventions = []

                # 场景1：改善通风（从当前提升到最好）
                vent_current = self._safe_int_convert(entry_data.get('ventilation', 3), '通风条件', 1, 5, 3)
                if vent_current < 5:
                    new_entry = dict(entry_data)
                    new_entry['ventilation'] = 5
                    pred = self.ml_predictor.predict_risk(new_entry, contact_type)
                    new_prob = pred['ensemble']['risk_probability'] if pred else original_prob
                    risk_reduction = original_prob - new_prob
                    if risk_reduction > 0:
                        interventions.append({
                            'name': '改善通风条件',
                            'from': f"{vent_current}分",
                            'to': "5分（最好）",
                            'reduction': risk_reduction,
                            'cost': '低（仅需开窗）'
                        })

                # 场景2：增加接触距离（从当前提升一级）
                dist_map = {
                    'very_close': '近', '极近': '近', '近': '中等',
                    'close': '中等', '中等': '远', 'medium': '远',
                    '远': '很远', 'far': '很远', '很远': '很远',
                    'very_far': '很远',
                }
                dist_current = entry_data.get('contact_distance', '中等')
                if dist_current in dist_map:
                    new_entry = dict(entry_data)
                    new_entry['contact_distance'] = dist_map[dist_current]
                    pred = self.ml_predictor.predict_risk(new_entry, contact_type)
                    new_prob = pred['ensemble']['risk_probability'] if pred else original_prob
                    risk_reduction = original_prob - new_prob
                    if risk_reduction > 0:
                        interventions.append({
                            'name': '增加接触距离',
                            'from': dist_current,
                            'to': dist_map[dist_current],
                            'reduction': risk_reduction,
                            'cost': '低（保持距离）'
                        })

                # 场景3：接种BCG（如果未接种）
                bcg_current = 1 if entry_data.get('bcg_vaccine', '是') == '是' else 0
                if bcg_current == 0:
                    new_entry = dict(entry_data)
                    new_entry['bcg_vaccine'] = 1
                    pred = self.ml_predictor.predict_risk(new_entry, contact_type)
                    new_prob = pred['ensemble']['risk_probability'] if pred else original_prob
                    risk_reduction = original_prob - new_prob
                    if risk_reduction > 0:
                        interventions.append({
                            'name': '接种BCG疫苗',
                            'from': '未接种',
                            'to': '已接种',
                            'reduction': risk_reduction,
                            'cost': '中（需接种）'
                        })

                # 场景4：减少接触频次（降为一半）
                freq_current = entry_data.get('freq_density', 14) if is_family else entry_data.get('freq_density', 2)
                if isinstance(freq_current, str):
                    try:
                        freq_current = int(freq_current)
                    except ValueError:
                        freq_current = 14
                if freq_current > 1:
                    new_entry = dict(entry_data)
                    new_freq = max(1, freq_current // 2)
                    new_entry['freq_density'] = new_freq
                    # 需要重新计算累积暴露
                    new_entry['cumulative_exposure'] = calculate_cumulative_exposure(
                        new_entry.get('single_duration', 30),
                        new_freq,
                        new_entry.get('time_span', 4)
                    )
                    pred = self.ml_predictor.predict_risk(new_entry, contact_type)
                    new_prob = pred['ensemble']['risk_probability'] if pred else original_prob
                    risk_reduction = original_prob - new_prob
                    if risk_reduction > 0:
                        interventions.append({
                            'name': '减少接触频次',
                            'from': f"{freq_current}次/周",
                            'to': f"{new_freq}次/周",
                            'reduction': risk_reduction,
                            'cost': '中（减少接触）'
                        })

                # 场景5：缩短接触时间（降为一半）
                duration_current = entry_data.get('single_duration', 30)
                if isinstance(duration_current, str):
                    try:
                        duration_current = int(duration_current)
                    except ValueError:
                        duration_current = 30
                if duration_current > 15:
                    new_entry = dict(entry_data)
                    new_duration = max(15, duration_current // 2)
                    new_entry['single_duration'] = new_duration
                    # 需要重新计算累积暴露
                    new_entry['cumulative_exposure'] = calculate_cumulative_exposure(
                        new_duration,
                        new_entry.get('freq_density', 14 if is_family else 2),
                        new_entry.get('time_span', 4)
                    )
                    pred = self.ml_predictor.predict_risk(new_entry, contact_type)
                    new_prob = pred['ensemble']['risk_probability'] if pred else original_prob
                    risk_reduction = original_prob - new_prob
                    if risk_reduction > 0:
                        interventions.append({
                            'name': '缩短单次接触时间',
                            'from': f"{duration_current}分钟",
                            'to': f"{new_duration}分钟",
                            'reduction': risk_reduction,
                            'cost': '中（缩短会面）'
                        })

                # 场景6：改善暴露场景
                setting_map = {'高危': '一般', '一般': '低危'}
                setting_current = entry_data.get('exposure_setting', '一般')
                setting_current_en = self._convert_chinese_to_value(setting_current, self.SETTING_MAPPING)
                if setting_current_en in setting_map:
                    new_entry = dict(entry_data)
                    new_setting_en = setting_map[setting_current_en]
                    new_setting = self.SETTING_MAPPING_REVERSE.get(new_setting_en, new_setting_en)
                    new_entry['exposure_setting'] = new_setting
                    pred = self.ml_predictor.predict_risk(new_entry, contact_type)
                    new_prob = pred['ensemble']['risk_probability'] if pred else original_prob
                    risk_reduction = original_prob - new_prob
                    if risk_reduction > 0:
                        interventions.append({
                            'name': '改善暴露场景',
                            'from': setting_current,
                            'to': new_setting,
                            'reduction': risk_reduction,
                            'cost': '中（更换场所）'
                        })

                # 场景7：缩短暴露时长
                timespan_current = entry_data.get('time_span', 4)
                if isinstance(timespan_current, str):
                    try:
                        timespan_current = int(timespan_current)
                    except ValueError:
                        timespan_current = 4
                if timespan_current > 2:
                    new_entry = dict(entry_data)
                    new_timespan = max(1, timespan_current // 2)
                    new_entry['time_span'] = new_timespan
                    # 需要重新计算累积暴露
                    new_entry['cumulative_exposure'] = calculate_cumulative_exposure(
                        new_entry.get('single_duration', 30),
                        new_entry.get('freq_density', 14 if is_family else 2),
                        new_timespan
                    )
                    pred = self.ml_predictor.predict_risk(new_entry, contact_type)
                    new_prob = pred['ensemble']['risk_probability'] if pred else original_prob
                    risk_reduction = original_prob - new_prob
                    if risk_reduction > 0:
                        interventions.append({
                            'name': '缩短暴露时长',
                            'from': f"{timespan_current}周",
                            'to': f"{new_timespan}周",
                            'reduction': risk_reduction,
                            'cost': '高（减少接触周期）'
                        })

                # 克拉玛依本土化干预场景
                if self.use_karamay and self.karamay_localizer:
                    for ki in self.karamay_localizer.get_counterfactual_interventions():
                        interventions.append({
                            'name': ki['name'], 'from': ki.get('from', 'N/A'),
                            'to': ki.get('to', 'N/A'),
                            'reduction': original_prob * 0.25,
                            'cost': ki.get('cost', '中'),
                            'literature': ki.get('literature', '')})

                # 按风险降低排序
                interventions.sort(key=lambda x: x['reduction'], reverse=True)

                # 计算效果/成本比并按此排序（贪心算法）
                cost_weights = {'低': 1, '中': 2, '高': 3}
                for interv in interventions:
                    cost_str = interv['cost']
                    cost_weight = cost_weights.get(cost_str.split('（')[0], 2)
                    interv['cost_weight'] = cost_weight
                    interv['efficiency_ratio'] = interv['reduction'] / cost_weight if cost_weight > 0 else 0

                # 按效果/成本比排序
                interventions_by_efficiency = sorted(interventions, key=lambda x: x['efficiency_ratio'], reverse=True)

                # 显示结果
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, f"接触者: {name}\n")
                result_text.insert(tk.END, f"原始风险: {original_prob:.1f}%\n\n")
                result_text.insert(tk.END, "="*50 + "\n")
                result_text.insert(tk.END, "【推荐方案】最小干预集（贪心算法）:\n")
                result_text.insert(tk.END, "="*50 + "\n\n")

                if interventions_by_efficiency:
                    # 贪心选择：优先效果/成本比高的方案
                    total_reduction = 0
                    recommended = []
                    target_reduction = max(original_prob * 0.5, 5.0)  # 目标降低50%或至少5%

                    for interv in interventions_by_efficiency:
                        if total_reduction < target_reduction and interv['reduction'] > 0:
                            recommended.append(interv)
                            total_reduction += interv['reduction']

                    if recommended:
                        result_text.insert(tk.END, "目标: 风险降低 ≥ {:.1f}%\n\n".format(target_reduction))
                        for i, interv in enumerate(recommended, 1):
                            result_text.insert(tk.END, f"【推荐{i}】{interv['name']}\n")
                            result_text.insert(tk.END, f"  从: {interv['from']} → 到: {interv['to']}\n")
                            result_text.insert(tk.END, f"  风险降低: {interv['reduction']:.1f}%\n")
                            result_text.insert(tk.END, f"  实施成本: {interv['cost']}\n")
                            result_text.insert(tk.END, f"  效果/成本比: {interv['efficiency_ratio']:.1f}\n\n")
                        result_text.insert(tk.END, f"【预计总效果】风险降低 {total_reduction:.1f}%\n\n")
                    else:
                        result_text.insert(tk.END, "风险已较低，无需强烈干预\n\n")

                result_text.insert(tk.END, "="*50 + "\n")
                result_text.insert(tk.END, "【全部方案】按效果排序:\n")
                result_text.insert(tk.END, "="*50 + "\n\n")

                if interventions:
                    for i, interv in enumerate(interventions, 1):
                        result_text.insert(tk.END, f"【方案{i}】{interv['name']}\n")
                        result_text.insert(tk.END, f"  从: {interv['from']} → 到: {interv['to']}\n")
                        result_text.insert(tk.END, f"  风险降低: {interv['reduction']:.1f}%\n")
                        result_text.insert(tk.END, f"  实施成本: {interv['cost']}\n")
                        result_text.insert(tk.END, f"  效果/成本比: {interv['efficiency_ratio']:.1f}\n\n")
                else:
                    result_text.insert(tk.END, "未找到有效干预措施（风险已较低）\n")

                result_text.insert(tk.END, "\n" + "="*50 + "\n")
                result_text.insert(tk.END, "文献支撑: She et al. (2024) 反事实分析框架\n")

                # Section IX: 绘制反事实干预柱状图
                # X 轴：干预方案名称；Y 轴：风险降低百分比
                # 颜色编码：效果/成本比高的方案用绿色，低的用橙色
                try:
                    _draw_counterfactual_bar_chart(cf_figure, cf_canvas, interventions, name)
                except Exception:
                    LOGGER.error("绘制反事实柱状图失败", exc_info=True)

            ttk.Button(dialog, text="分析干预效果", command=analyze_counterfactual).pack(pady=10)

        except Exception as e:
            messagebox.showerror("错误", f"反事实分析失败: {str(e)}")


# Section IX: 反事实干预柱状图绘制（模块级函数，便于复用与单测）
def _draw_counterfactual_bar_chart(figure, canvas, interventions, contact_name=''):
    """绘制反事实干预风险降低对比柱状图

    在 ``_show_counterfactual_analysis`` 中被调用，将各干预方案的风险降低效果
    以柱状图形式可视化，便于用户直观对比不同方案的效果。

    Args:
        figure: matplotlib.figure.Figure 对象（若为 None 则跳过）
        canvas: FigureCanvasTkAgg 对象（用于 draw 刷新）
        interventions (list): 干预方案列表，每项需含 ``name`` / ``reduction`` /
            ``efficiency_ratio`` 字段
        contact_name (str): 接触者姓名（用于图表标题）
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    if figure is None or canvas is None:
        return

    # 清空旧图
    figure.clear()

    # 无干预数据时显示提示
    if not interventions:
        ax = figure.add_subplot(111)
        ax.text(0.5, 0.5, "无有效干预方案\n（风险已较低）",
                ha='center', va='center', fontsize=12, color='gray',
                transform=ax.transAxes)
        ax.set_axis_off()
        figure.tight_layout()
        canvas.draw()
        return

    ax = figure.add_subplot(111)

    # 按效果/成本比降序排列（柱状图从左到右：高效 → 低效）
    sorted_interv = sorted(interventions,
                           key=lambda x: x.get('efficiency_ratio', 0),
                           reverse=True)
    names = [i['name'] for i in sorted_interv]
    reductions = [i.get('reduction', 0) for i in sorted_interv]
    ratios = [i.get('efficiency_ratio', 0) for i in sorted_interv]

    # 颜色编码：效果/成本比高的方案用绿色，低的用橙色
    # 以中位数为阈值划分
    if ratios:
        median_ratio = sorted(ratios)[len(ratios) // 2]
    else:
        median_ratio = 0
    colors = ['#2ca02c' if r >= median_ratio else '#ff7f0e' for r in ratios]

    # 绘制柱状图
    bars = ax.bar(range(len(names)), reductions, color=colors,
                  edgecolor='black', linewidth=0.5)

    # X 轴标签：方案名称（旋转避免重叠）
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)

    # Y 轴标签
    ax.set_ylabel('风险降低百分比 (%)', fontsize=10)
    title = '干预方案风险降低对比'
    if contact_name:
        title += f'\n接触者: {contact_name}'
    ax.set_title(title, fontsize=11)

    # 在每个柱顶标注数值
    for bar, reduction in zip(bars, reductions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{reduction:.1f}%', ha='center', va='bottom', fontsize=9)

    # 图例：绿色=效果/成本比高，橙色=效果/成本比低
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', label='效果/成本比高（推荐）'),
        Patch(facecolor='#ff7f0e', label='效果/成本比低'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # 留出顶部空间给数值标签
    if reductions:
        ax.set_ylim(0, max(reductions) * 1.2)

    ax.grid(axis='y', linestyle='--', alpha=0.4)
    figure.tight_layout()
    canvas.draw()
