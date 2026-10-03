#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 图表 - 风险分布图（RiskChartMixin）

包含：风险评分柱状图、风险热力图、概率直方图、延迟就诊影响曲线
"""

from ._shared import *


def _contact_age_index(age):
    """根据年龄返回年龄组索引（0-4）"""
    if age < 5:
        return 0
    if age < 15:
        return 1
    if age < 35:
        return 2
    if age < 65:
        return 3
    return 4


def _contact_risk_index(prob):
    """根据发病概率返回风险等级索引（0-4）"""
    if prob < 10:
        return 0
    if prob < 25:
        return 1
    if prob < 40:
        return 2
    if prob < 60:
        return 3
    return 4


def _group_contacts_by_cell(contacts):
    """将接触者列表按 (age_idx, risk_idx) 分组

    Returns:
        dict: {(age_idx, risk_idx): [contact_dict, ...]}
    """
    groups = {}
    for c in contacts:
        age_idx = _contact_age_index(c.get('age', 30))
        risk_idx = _contact_risk_index(c.get('disease_probability', 0))
        groups.setdefault((age_idx, risk_idx), []).append(c)
    return groups


class RiskChartMixin:
    """风险分布图表方法"""

    def _draw_risk_chart(self):
        """绘制风险评分柱状图（Section VI: 迁移到 matplotlib，支持工具栏缩放/平移/保存 + 点击交互）"""
        # Section VI: 优先使用 matplotlib 版本
        if MATPLOTLIB_AVAILABLE and hasattr(self, 'risk_figure'):
            self._draw_risk_chart_mpl()
            return
        # 向后兼容：matplotlib 不可用时回退到 tkinter Canvas
        self._draw_risk_chart_canvas()

    def _draw_risk_chart_mpl(self):
        """matplotlib 版本的风险评分柱状图（支持交互）"""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'risk_figure'):
            return
        if not self.results or not self.patient_info:
            return

        self._init_chart_interaction()

        indicators = ['FCI', 'SNC', 'FLP', 'HRSP', 'FTD']
        scores = []
        # 优先级十：从 COLORS 字典读取语义化颜色（主题切换时同步）
        colors = [self.COLORS.get(f'chart_series_{i+1}', c) for i, c in enumerate(['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])]

        basic_info = self.patient_info.get('basic_info', {}) if isinstance(self.patient_info, dict) else {}
        for indicator in indicators:
            if indicator == 'FCI':
                score = self.patient_info.get('FCI', 0) if isinstance(self.patient_info, dict) else 0
            elif indicator == 'SNC':
                score = self.patient_info.get('SNC', 0) if isinstance(self.patient_info, dict) else 0
            elif indicator == 'FLP':
                flp_pct = basic_info.get('flp_percentage', 0)
                try:
                    flp_pct = float(flp_pct)
                except (TypeError, ValueError):
                    flp_pct = 0.0
                score = flp_pct / 10
            elif indicator == 'HRSP':
                hrsp_pct = basic_info.get('hrsp_percentage', 0)
                try:
                    hrsp_pct = float(hrsp_pct)
                except (TypeError, ValueError):
                    hrsp_pct = 0.0
                score = hrsp_pct / 10
            elif indicator == 'FTD':
                delay_days = basic_info.get('delay_days', 0)
                try:
                    delay_days = float(delay_days)
                except (TypeError, ValueError):
                    delay_days = 0.0
                score = min(delay_days / 365, 1) * 10
            scores.append(min(score, 10))

        try:
            self.risk_figure.clear()
            ax = self.risk_figure.add_subplot(111)
            bars = ax.bar(indicators, scores, color=colors, edgecolor='black', linewidth=1)

            # 在柱顶显示数值
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        f'{score:.1f}', ha='center', va='bottom', fontsize=10)

            ax.set_ylabel('风险评分 (0-10)')
            ax.set_title('风险评分分布', fontsize=13, fontweight='bold')
            ax.set_ylim(0, 11)
            ax.grid(axis='y', alpha=0.3)
            self.risk_figure.tight_layout()

            # 保存 bar 对象用于交互高亮
            self._chart_bars['risk_bar'] = list(bars)

            # Section VI: 绑定点击交互（柱子点击高亮 + 显示详情）
            self._safe_mpl_connect(
                getattr(self, 'risk_canvas', None), 'risk_bar',
                'button_press_event', self._on_risk_bar_click
            )
            # Section VI: 绑定悬停交互（显示具体数值）
            self._safe_mpl_connect(
                getattr(self, 'risk_canvas', None), 'risk_bar',
                'motion_notify_event', self._on_risk_bar_hover
            )

            self._redraw_chart_canvas(getattr(self, 'risk_canvas', None))
        except Exception as e:
            LOGGER.warning("matplotlib 风险柱状图绘制失败: %s", e, exc_info=True)

    def _on_risk_bar_click(self, event):
        """风险柱状图点击回调：高亮选中柱子并显示指标详情"""
        try:
            if event is None or event.inaxes is None:
                return
            bars = self._chart_bars.get('risk_bar', [])
            if not bars:
                return
            # 命中测试
            hit_idx, indices = self._hit_test_multiple(bars, event)
            if hit_idx is None:
                # 点击空白：清除选中
                self._clear_selection('risk_bar')
                self._hide_hover_annotation('risk_bar')
                self._highlight_bars(bars, [], 'risk_bar',
                                     normal_color=[self.COLORS.get(f'chart_series_{i+1}', c) for i, c in enumerate(['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])])
                self._redraw_chart_canvas(getattr(self, 'risk_canvas', None))
                return
            # 高亮选中柱子
            self._set_selection('risk_bar', [hit_idx])
            # 优先级十：从 COLORS 字典读取语义化颜色（主题切换时同步）
            colors = [self.COLORS.get(f'chart_series_{i+1}', c) for i, c in enumerate(['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])]
            self._highlight_bars(bars, [hit_idx], 'risk_bar',
                                 normal_color=[colors[i] if i != hit_idx else '#FFD700' for i in range(len(bars))])
            # 优先级七：通知 SelectionBroker 广播选中条件，触发 SEIR 曲线联动标注
            indicators = ['FCI', 'SNC', 'FLP', 'HRSP', 'FTD']
            self._notify_broker('risk_bar', {'indicator': indicators[hit_idx]})
            self._redraw_chart_canvas(getattr(self, 'risk_canvas', None))
        except Exception as e:
            LOGGER.debug("风险柱状图点击回调失败: %s", e)

    def _on_risk_bar_hover(self, event):
        """风险柱状图悬停回调：显示指标名称和具体数值"""
        try:
            if event is None or event.inaxes is None:
                self._hide_hover_annotation('risk_bar')
                self._redraw_chart_canvas(getattr(self, 'risk_canvas', None))
                return
            bars = self._chart_bars.get('risk_bar', [])
            if not bars:
                return
            hit_idx, _ = self._hit_test_multiple(bars, event)
            if hit_idx is None:
                self._hide_hover_annotation('risk_bar')
                self._redraw_chart_canvas(getattr(self, 'risk_canvas', None))
                return
            indicators = ['FCI', 'SNC', 'FLP', 'HRSP', 'FTD']
            labels = ['家庭接触指数', '社会网络复杂度', '家庭潜伏感染比例',
                      '高危人群比例', '延迟就诊天数']
            bar = bars[hit_idx]
            tooltip_text = f'{indicators[hit_idx]} ({labels[hit_idx]})\n评分: {bar.get_height():.1f}'
            self._show_hover_annotation(
                event.inaxes, bar.get_x() + bar.get_width() / 2,
                bar.get_height(), tooltip_text, 'risk_bar'
            )
            self._redraw_chart_canvas(getattr(self, 'risk_canvas', None))
        except Exception as e:
            LOGGER.debug("风险柱状图悬停回调失败: %s", e)

    def _draw_risk_chart_canvas(self):
        """绘制风险评分柱状图（tkinter Canvas 回退版本，matplotlib 不可用时使用）"""
        if not getattr(self, 'canvas', None) or not self.results or not self.patient_info:
            return

        self.canvas.delete("all")

        indicators = ['FCI', 'SNC', 'FLP', 'HRSP', 'FTD']
        scores = []
        # 优先级十：从 COLORS 字典读取语义化颜色（主题切换时同步）
        colors = [self.COLORS.get(f'chart_series_{i+1}', c) for i, c in enumerate(['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])]

        basic_info = self.patient_info.get('basic_info', {}) if isinstance(self.patient_info, dict) else {}
        for indicator in indicators:
            if indicator == 'FCI':
                score = self.patient_info.get('FCI', 0) if isinstance(self.patient_info, dict) else 0
            elif indicator == 'SNC':
                score = self.patient_info.get('SNC', 0) if isinstance(self.patient_info, dict) else 0
            elif indicator == 'FLP':
                flp_pct = basic_info.get('flp_percentage', 0)
                try:
                    flp_pct = float(flp_pct)
                except (TypeError, ValueError):
                    flp_pct = 0.0
                score = flp_pct / 10
            elif indicator == 'HRSP':
                hrsp_pct = basic_info.get('hrsp_percentage', 0)
                try:
                    hrsp_pct = float(hrsp_pct)
                except (TypeError, ValueError):
                    hrsp_pct = 0.0
                score = hrsp_pct / 10
            elif indicator == 'FTD':
                delay_days = basic_info.get('delay_days', 0)
                try:
                    delay_days = float(delay_days)
                except (TypeError, ValueError):
                    delay_days = 0.0
                score = min(delay_days / 365, 1) * 10
            scores.append(min(score, 10))

        self.canvas.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width < 10:
            canvas_width = 600
        if canvas_height < 50:
            canvas_height = 200

        bar_width = max((canvas_width - 100) / len(indicators) - 10, 20)
        max_height = max(canvas_height - 60, 10)

        for i, (indicator, score) in enumerate(zip(indicators, scores)):
            x1 = 50 + i * (bar_width + 10)
            x2 = x1 + bar_width
            bar_height = (score / 10) * max_height
            y1 = canvas_height - 30 - bar_height
            y2 = canvas_height - 30

            self.canvas.create_rectangle(x1, y1, x2, y2, fill=colors[i], outline='black')
            self.canvas.create_text(x1 + bar_width/2, y1 - 10, text=f"{score:.1f}", font=('Arial', 10))
            self.canvas.create_text(x1 + bar_width/2, y2 + 15, text=indicator, font=('Arial', 10, 'bold'))

        self.canvas.create_line(40, 10, 40, canvas_height - 30, width=2)
        self.canvas.create_text(20, 20, text="10", font=('Arial', 8))
        self.canvas.create_text(20, canvas_height - 30, text="0", font=('Arial', 8))
        self.canvas.create_text(20, canvas_height/2, text="5", font=('Arial', 8))

    def _draw_classic_charts(self):
        """绘制传统界面的图表（防御性编程）"""
        # Section VI: matplotlib 版本不需要 update_idletasks
        if hasattr(self, 'canvas') and getattr(self, 'canvas', None) is not None:
            try:
                self.canvas.update_idletasks()
            except Exception:
                pass
        self._draw_risk_chart()
        if hasattr(self, '_update_all_charts'):
            self._update_all_charts()

    def _draw_wizard_charts(self):
        """绘制向导界面的图表（防御性编程）"""
        if hasattr(self, '_update_all_charts'):
            try:
                self._update_all_charts()
            except Exception as e:
                LOGGER.warning("向导界面图表绘制失败: %s", e, exc_info=True)

    def _update_all_charts(self):
        """更新所有图表（评估完成后调用，带界面刷新）

        Section VI: 补齐随机 SEIR 曲线、MCMC 诊断图、GNN 网络拓扑图等
        已实现但未接入的方法调用入口。未实现的方法通过 hasattr 守卫安全跳过。
        """
        # 模型证据面板（三层递进 / 任务A/B/C / 消融 / SEIR 干预）图表重绘。
        # 静态证据（消融/DGP/公开数据/SEIR）不依赖评估结果，置于早退守卫之前，
        # 保证评估前切换主题也能刷新这些图表。
        if MATPLOTLIB_AVAILABLE and hasattr(self, '_redraw_model_evidence_charts'):
            try:
                self._redraw_model_evidence_charts()
                self.root.update_idletasks()
            except Exception as e:
                LOGGER.warning("模型证据图表更新失败: %s", e, exc_info=True)

        if not MATPLOTLIB_AVAILABLE or not self.results:
            return

        charts = [
            ("SEIR传播动力学曲线", self._draw_seir_curve),
            ("风险热力图", self._draw_risk_heatmap),
            ("概率直方图", self._draw_probability_histogram),
            ("延迟就诊影响曲线", self._draw_delay_impact_curve)
        ]

        for chart_name, chart_func in charts:
            try:
                chart_func()
                self.root.update_idletasks()
            except Exception as e:
                LOGGER.warning("%s更新失败: %s", chart_name, e, exc_info=True)

        # Section VI: 补齐已实现但未接入的图表调用入口
        # 随机 SEIR 曲线（与确定性 SEIR 共用 seir_figure，仅在启用标志时绘制）
        if getattr(self, 'use_stochastic_seir', False) and hasattr(self, '_draw_stochastic_seir_curve'):
            try:
                self._draw_stochastic_seir_curve()
                self.root.update_idletasks()
            except Exception as e:
                LOGGER.warning("随机SEIR曲线更新失败: %s", e, exc_info=True)

        # ML 图表统一刷新（包含对比图/性能图/校准图/SHAP/DAG 等）
        if hasattr(self, '_update_ml_charts'):
            try:
                self._update_ml_charts()
                self.root.update_idletasks()
            except Exception as e:
                LOGGER.warning("ML图表更新失败: %s", e, exc_info=True)

        # 因果 DAG 可视化（独立 canvas）
        if hasattr(self, '_draw_causal_dag'):
            try:
                self._draw_causal_dag()
                self.root.update_idletasks()
            except Exception as e:
                LOGGER.warning("因果DAG更新失败: %s", e, exc_info=True)

        # 预留入口：MCMC 诊断图（后验分布直方图、踪迹图）
        if hasattr(self, '_draw_mcmc_diagnostics'):
            try:
                self._draw_mcmc_diagnostics()
                self.root.update_idletasks()
            except Exception as e:
                LOGGER.warning("MCMC诊断图更新失败: %s", e, exc_info=True)

        # 预留入口：GNN 网络拓扑图
        if hasattr(self, '_draw_gnn_topology'):
            try:
                self._draw_gnn_topology()
                self.root.update_idletasks()
            except Exception as e:
                LOGGER.warning("GNN网络拓扑图更新失败: %s", e, exc_info=True)

    def _draw_risk_heatmap(self):
        """绘制接触者风险热力图（基于实际输入的接触者数据）"""
        if not (MATPLOTLIB_AVAILABLE and NUMPY_AVAILABLE and SEABORN_AVAILABLE) or not hasattr(self, 'heatmap_figure'):
            return

        try:
            self.heatmap_figure.clear()
            ax = self.heatmap_figure.add_subplot(111)

            potential_patients = self.results.get('potential_patients', {})
            all_contacts = []

            for patient in potential_patients.get('family', []):
                all_contacts.append({
                    'type': '家庭',
                    'age': patient.get('age', 30),
                    'disease_probability': patient.get('disease_probability', 0),
                    'name': patient.get('name', '未知')
                })

            for patient in potential_patients.get('social', []):
                all_contacts.append({
                    'type': '社会',
                    'age': patient.get('age', 30),
                    'disease_probability': patient.get('disease_probability', 0),
                    'name': patient.get('name', '未知')
                })

            if not all_contacts:
                ax.text(0.5, 0.5, '无接触者数据\n请添加家庭成员或社会接触者',
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title('接触者风险热力图', fontsize=14, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                self.heatmap_canvas.draw()
                return

            age_groups = ['<5岁', '5-14岁', '15-35岁', '36-65岁', '>65岁']
            risk_groups = ['低风险(0-10%)', '中低风险(10-25%)', '中风险(25-40%)', '中高风险(40-60%)', '高风险(>60%)']

            risk_matrix = np.zeros((5, 5))
            count_matrix = np.zeros((5, 5), dtype=int)

            for contact in all_contacts:
                age = contact['age']
                prob = contact['disease_probability']

                if age < 5:
                    age_idx = 0
                elif age < 15:
                    age_idx = 1
                elif age < 35:
                    age_idx = 2
                elif age < 65:
                    age_idx = 3
                else:
                    age_idx = 4

                if prob < 10:
                    risk_idx = 0
                elif prob < 25:
                    risk_idx = 1
                elif prob < 40:
                    risk_idx = 2
                elif prob < 60:
                    risk_idx = 3
                else:
                    risk_idx = 4

                risk_matrix[age_idx, risk_idx] += prob
                count_matrix[age_idx, risk_idx] += 1

            avg_risk_matrix = np.divide(risk_matrix, count_matrix,
                                        where=count_matrix > 0,
                                        out=np.zeros_like(risk_matrix))

            annot_matrix = []
            for i in range(5):
                row = []
                for j in range(5):
                    if count_matrix[i, j] > 0:
                        row.append(f'{avg_risk_matrix[i, j]:.1f}% (n={count_matrix[i, j]})')
                    else:
                        row.append('')
                annot_matrix.append(row)

            im = sns.heatmap(avg_risk_matrix, annot=annot_matrix, fmt='',
                            cmap='YlOrRd', ax=ax,
                            xticklabels=risk_groups,
                            yticklabels=age_groups,
                            cbar_kws={'label': '平均发病概率 (%)'},
                            vmin=0, vmax=80)

            total_count = len(all_contacts)
            avg_prob = np.mean([c['disease_probability'] for c in all_contacts])
            stat_text = f'总接触者: {total_count}人\n平均风险: {avg_prob:.1f}%'
            ax.text(0.02, 0.98, stat_text, transform=ax.transAxes,
                    ha='left', va='top', fontsize=10, bbox=dict(boxstyle='round', alpha=0.1))

            ax.set_xlabel('风险等级', fontsize=12)
            ax.set_ylabel('年龄组', fontsize=12)
            ax.set_title('接触者风险热力图 [基于实际输入数据]', fontsize=14, fontweight='bold')

            # Section VI: 保存数据用于点击交互（单元格点击展示对应人群列表）
            self._chart_bars['heatmap'] = {
                'count_matrix': count_matrix,
                'avg_risk_matrix': avg_risk_matrix,
                'contacts_by_cell': _group_contacts_by_cell(all_contacts),
                'age_groups': age_groups,
                'risk_groups': risk_groups,
                'im': im,
            }
            self._safe_mpl_connect(
                getattr(self, 'heatmap_canvas', None), 'heatmap',
                'button_press_event', self._on_heatmap_click
            )

            self.heatmap_canvas.draw()
        except Exception as e:
            LOGGER.warning("绘制风险热力图时出错: %s", e, exc_info=True)

    def _on_heatmap_click(self, event):
        """热力图点击回调：展示对应年龄-风险单元格的人群列表"""
        try:
            if event is None or event.inaxes is None:
                return
            data = self._chart_bars.get('heatmap')
            if not data:
                return
            # 通过 data 坐标计算单元格索引（热力图 5x5 网格）
            x_data = event.xdata
            y_data = event.ydata
            if x_data is None or y_data is None:
                return
            # 热力图 x 轴是风险等级（0-4），y 轴是年龄组（0-4，自上而下）
            col = int(np.clip(np.floor(x_data), 0, 4))
            row = int(np.clip(np.floor(y_data), 0, 4))
            count = int(data['count_matrix'][row, col])
            age_label = data['age_groups'][row]
            risk_label = data['risk_groups'][col]
            contacts = data['contacts_by_cell'].get((row, col), [])
            if not contacts:
                tooltip_text = (f'{age_label} × {risk_label}\n'
                                f'该单元格无接触者')
            else:
                lines = [f'{age_label} × {risk_label} (共 {count} 人)']
                # 最多展示 8 人，避免注释过长
                for c in contacts[:8]:
                    lines.append(f"  · {c['name']} ({c['type']}, "
                                 f"{c['age']}岁, {c['disease_probability']:.1f}%)")
                if len(contacts) > 8:
                    lines.append(f"  ... 还有 {len(contacts) - 8} 人")
                tooltip_text = '\n'.join(lines)
            # 在单元格中心显示注释
            self._show_hover_annotation(
                event.inaxes, col + 0.5, row + 0.5, tooltip_text, 'heatmap'
            )
            self._set_selection('heatmap', [(row, col)])
            # 优先级七：通知 SelectionBroker 广播选中条件，触发直方图联动过滤
            self._notify_broker('heatmap', {'age_group': row, 'risk_level': col})
            self._redraw_chart_canvas(getattr(self, 'heatmap_canvas', None))
        except Exception as e:
            LOGGER.debug("热力图点击回调失败: %s", e)

    def _draw_probability_histogram(self):
        """绘制发病概率分布直方图与累积概率曲线（增强版）

        优先级七：支持 SelectionBroker 联动过滤。当热力图选中某年龄-风险单元格时，
        直方图自动过滤为对应人群分布。
        """
        if not (MATPLOTLIB_AVAILABLE and NUMPY_AVAILABLE) or not hasattr(self, 'histogram_figure'):
            return

        try:
            self._init_chart_interaction()

            # 收集接触者（保留 age/type 用于联动过滤）
            all_contacts = []
            potential_patients = self.results.get('potential_patients', {})

            for patient in potential_patients.get('family', []):
                if 'disease_probability' in patient:
                    all_contacts.append({
                        'probability': patient['disease_probability'],
                        'age': patient.get('age', 30),
                        'type': 'family',
                    })

            for patient in potential_patients.get('social', []):
                if 'disease_probability' in patient:
                    all_contacts.append({
                        'probability': patient['disease_probability'],
                        'age': patient.get('age', 30),
                        'type': 'social',
                    })

            # 优先级七：应用 broker 联动过滤
            filters = self._get_broker_filters(exclude_chart='histogram')
            filtered_contacts = self._apply_histogram_filters(all_contacts, filters)

            all_probabilities = [c['probability'] for c in filtered_contacts]

            self.histogram_figure.clear()
            ax = self.histogram_figure.add_subplot(111)

            if not all_probabilities:
                ax.text(0.5, 0.5, '无接触者数据\n请添加家庭成员或社会接触者',
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title('发病概率分布直方图与累积概率曲线', fontsize=14, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                self.histogram_canvas.draw()
                return

            n, bins, patches = ax.hist(all_probabilities, bins=10,
                                        alpha=0.7, color='#3498db',
                                        edgecolor='black', density=True,
                                        label='概率分布')

            sorted_probs = np.sort(all_probabilities)
            cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
            ax2 = ax.twinx()
            ax2.plot(sorted_probs, cumulative, color='#e74c3c', linewidth=2,
                    label='累积概率')

            median = np.median(all_probabilities)
            q1 = np.percentile(all_probabilities, 25)
            q3 = np.percentile(all_probabilities, 75)
            mean = np.mean(all_probabilities)

            ax.axvline(x=median, color='#2c3e50', linestyle='-', linewidth=2,
                      label=f'中位数={median:.1f}%')
            ax.axvline(x=q1, color='#95a5a6', linestyle='--', linewidth=1.5,
                      label=f'Q1={q1:.1f}%')
            ax.axvline(x=q3, color='#95a5a6', linestyle='--', linewidth=1.5,
                      label=f'Q3={q3:.1f}%')

            ax.fill_betweenx([0, ax.get_ylim()[1]], q1, q3,
                            color='#95a5a6', alpha=0.2, label='四分位间距')

            stat_text = '统计摘要:\n'
            stat_text += f'样本数: {len(all_probabilities)}\n'
            stat_text += f'最小值: {np.min(all_probabilities):.1f}%\n'
            stat_text += f'最大值: {np.max(all_probabilities):.1f}%\n'
            stat_text += f'均值: {mean:.1f}%\n'
            stat_text += f'中位数: {median:.1f}%\n'
            stat_text += f'四分位间距: [{q1:.1f}%, {q3:.1f}%]'
            # 优先级七：显示过滤状态
            if filters:
                filter_descs = []
                for src, sel in filters.items():
                    if src == 'heatmap':
                        age_groups = ['0-4', '5-14', '15-34', '35-64', '65+']
                        risk_groups = ['<10', '10-25', '25-40', '40-60', '>=60']
                        ag = age_groups[sel['age_group']] if 'age_group' in sel else '*'
                        rg = risk_groups[sel['risk_level']] if 'risk_level' in sel else '*'
                        filter_descs.append(f'年龄={ag}, 风险={rg}')
                if filter_descs:
                    stat_text += f'\n过滤: {"; ".join(filter_descs)}'

            ax.text(0.98, 0.98, stat_text, transform=ax.transAxes,
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            ax.set_xlabel('发病概率 (%)', fontsize=12)
            ax.set_ylabel('密度', fontsize=12, color='#3498db')
            ax2.set_ylabel('累积概率', fontsize=12, color='#e74c3c')
            title = '发病概率分布直方图与累积概率曲线'
            if filters:
                title += ' [已联动过滤]'
            ax.set_title(title, fontsize=14, fontweight='bold')

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3, axis='x')

            # 优先级七：保存原始接触者数据用于联动过滤重绘
            self._chart_bars['histogram'] = {'contacts': all_contacts}
            # 注册到 broker 订阅（首次绘制时注册，后续重绘不重复注册）
            self._register_chart_with_broker('histogram', self._on_broker_filter_histogram)

            self.histogram_canvas.draw()
        except Exception as e:
            LOGGER.warning("绘制概率分布直方图时出错: %s", e, exc_info=True)

    def _apply_histogram_filters(self, contacts, filters):
        """优先级七：根据 broker 过滤条件筛选接触者

        Args:
            contacts: 原始接触者列表（含 probability/age/type）
            filters: broker 活跃过滤条件字典

        Returns:
            过滤后的接触者列表
        """
        if not filters:
            return list(contacts)

        filtered = list(contacts)
        for source_chart, selection in filters.items():
            if source_chart == 'heatmap' and selection:
                age_group = selection.get('age_group')
                risk_level = selection.get('risk_level')
                # 年龄组边界: 0-4, 5-14, 15-34, 35-64, 65+
                age_bounds = [(0, 5), (5, 15), (15, 35), (35, 65), (65, 200)]
                # 风险等级边界: <10, 10-25, 25-40, 40-60, >=60
                risk_bounds = [(0, 10), (10, 25), (25, 40), (40, 60), (60, 200)]
                if age_group is not None and 0 <= age_group < len(age_bounds):
                    lo, hi = age_bounds[age_group]
                    filtered = [c for c in filtered if lo <= c['age'] < hi]
                if risk_level is not None and 0 <= risk_level < len(risk_bounds):
                    lo, hi = risk_bounds[risk_level]
                    filtered = [c for c in filtered if lo <= c['probability'] < hi]
        return filtered

    def _on_broker_filter_histogram(self, source_chart, selection):
        """优先级七：SelectionBroker 回调 - 联动过滤概率直方图

        当热力图选中某年龄-风险单元格时，重绘直方图为对应人群分布。

        Args:
            source_chart: 触发选中的源图表标识
            selection: 选中条件字典（None 表示清除过滤）
        """
        try:
            # 直接重绘直方图（_draw_probability_histogram 内部会读取 broker 过滤条件）
            self._draw_probability_histogram()
        except Exception as e:
            LOGGER.debug("直方图联动过滤失败: %s", e)

    def _draw_delay_impact_curve(self):
        """绘制延迟就诊天数对家庭传播风险的影响曲线（改进版）"""
        if not (MATPLOTLIB_AVAILABLE and NUMPY_AVAILABLE) or not hasattr(self, 'delay_figure'):
            return

        try:
            self.delay_figure.clear()
            ax = self.delay_figure.add_subplot(111)

            delay_days = np.arange(0, 91, 1)

            potential_patients = self.results.get('potential_patients', {})
            contact_probs = []

            for patient in potential_patients.get('family', []):
                if 'disease_probability' in patient:
                    contact_probs.append(patient['disease_probability'])

            for patient in potential_patients.get('social', []):
                if 'disease_probability' in patient:
                    contact_probs.append(patient['disease_probability'])

            if not contact_probs:
                contact_probs = [20]

            q10 = np.percentile(contact_probs, 10)
            q25 = np.percentile(contact_probs, 25)
            median = np.median(contact_probs)
            q75 = np.percentile(contact_probs, 75)
            q90 = np.percentile(contact_probs, 90)

            base_risk = self.results.get('base_infection_probability', 20)

            colors = ['#3498db', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']
            labels = ['10%分位', '25%分位', '中位数', '75%分位', '90%分位']
            quantiles = [q10, q25, median, q75, q90]

            low_risk_25 = q25 * (1 - np.exp(-delay_days / 30)) * (base_risk / 20)
            high_risk_75 = q75 * (1 - np.exp(-delay_days / 30)) * (base_risk / 20)
            ax.fill_between(delay_days, low_risk_25, high_risk_75,
                           color='#f39c12', alpha=0.2, label='25%-75%分位区间')

            low_risk_10 = q10 * (1 - np.exp(-delay_days / 30)) * (base_risk / 20)
            high_risk_90 = q90 * (1 - np.exp(-delay_days / 30)) * (base_risk / 20)
            ax.fill_between(delay_days, low_risk_10, high_risk_90,
                           color='#3498db', alpha=0.1, label='10%-90%分位区间')

            for i, quantile in enumerate(quantiles):
                delay_risk = quantile * (1 - np.exp(-delay_days / 30)) * (base_risk / 20)
                delay_risk = np.clip(delay_risk, 0, 95)
                linewidth = 3 if i == 2 else 2
                ax.plot(delay_days, delay_risk, label=labels[i],
                       color=colors[i], linewidth=linewidth)

            ax.axvline(x=14, color='gray', linestyle='--', alpha=0.6, label='2周阈值')
            ax.axvline(x=30, color='gray', linestyle=':', alpha=0.6, label='1个月阈值')

            stat_text = '接触者统计:\n'
            stat_text += f'样本数: {len(contact_probs)}\n'
            stat_text += f'基准风险: {base_risk:.1f}%\n'
            stat_text += f'中位数: {median:.1f}%\n'
            stat_text += f'[Q10, Q90]: [{q10:.1f}%, {q90:.1f}%]'

            ax.text(0.02, 0.98, stat_text, transform=ax.transAxes,
                    ha='left', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            ax.set_xlabel('延迟就诊天数', fontsize=12)
            ax.set_ylabel('接触者感染概率 (%)', fontsize=12)
            ax.set_title('延迟就诊天数对接触者传播风险的影响 [基于实际风险分布]', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=9, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

            self.delay_canvas.draw()
        except Exception as e:
            LOGGER.warning("绘制延迟就诊影响曲线时出错: %s", e, exc_info=True)