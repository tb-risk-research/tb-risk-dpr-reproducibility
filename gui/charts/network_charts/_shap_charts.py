#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SHAP 特征重要性图 Mixin（_SHAPChartsMixin）

包含：SHAP特征重要性图、个体SHAP解释下拉框更新、个体SHAP瀑布图
"""

from .._shared import *  # noqa: F401,F403


class _SHAPChartsMixin:
    """SHAP 特征重要性可视化方法"""

    def _draw_shap_chart(self):
        """绘制SHAP特征重要性图"""
        if not SHAP_AVAILABLE or not hasattr(self, 'shap_figure'):
            return

        if hasattr(self, 'ml_results_lock'):
            with self.ml_results_lock:
                ml_results = self.ml_results
        else:
            ml_results = self.ml_results

        shap_info = ml_results.get('shap_analysis', {})
        if not shap_info or not shap_info.get('feature_importance'):
            return

        try:
            self.shap_figure.clear()

            feature_importance = shap_info.get('feature_importance', [])
            feature_descs = shap_info.get('feature_descriptions', {})

            top_n = min(10, len(feature_importance))
            top_features = feature_importance[:top_n]

            names = [feature_descs.get(f[0], f[0]) for f in reversed(top_features)]
            values = [f[1] for f in reversed(top_features)]

            ax1 = self.shap_figure.add_subplot(121)
            ax1.barh(range(top_n), values, color='#45B7D1', alpha=0.85)
            ax1.set_yticks(range(top_n))
            ax1.set_yticklabels(names, fontsize=9)
            ax1.set_xlabel('平均|SHAP值|')
            ax1.set_title('特征重要性排名')

            shap_values = shap_info.get('shap_values')
            feature_matrix = shap_info.get('feature_matrix')
            feature_names = shap_info.get('feature_names', [])

            ax2 = self.shap_figure.add_subplot(122)
            if shap_values is not None and feature_matrix is not None and len(shap_values) > 0:
                display_names = [feature_descs.get(n, n) for n in feature_names]

                feature_name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
                top_indices = [feature_name_to_idx[f[0]] for f in top_features if f[0] in feature_name_to_idx]

                if top_indices:
                    shap_subset = shap_values[:, top_indices]
                    name_subset = [display_names[i] for i in top_indices]

                    for i in range(min(shap_subset.shape[0], 50)):
                        for j in range(len(top_indices)):
                            color = '#FF6B6B' if shap_subset[i, j] > 0 else '#4ECDC4'
                            size = min(abs(shap_subset[i, j]) * 15 + 3, 50)
                            ax2.scatter(shap_subset[i, j], j, c=color, s=size, alpha=0.5, edgecolors='none')

                    ax2.set_yticks(range(len(name_subset)))
                    ax2.set_yticklabels(name_subset, fontsize=8)
                    ax2.set_xlabel('SHAP值')
                    ax2.set_title('SHAP值分布（红=高风险，蓝=低风险）')
                    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                else:
                    ax2.text(0.5, 0.5, '无法计算SHAP值分布\n特征不匹配',
                           ha='center', va='center', fontsize=12, color='#666666')
            else:
                ax2.text(0.5, 0.5, '无法计算SHAP值\n数据不完整',
                       ha='center', va='center', fontsize=12, color='#666666')

            self.shap_figure.tight_layout()
            self.shap_canvas.draw()
        except Exception as e:
            LOGGER.warning("SHAP图绘制失败: %s", e, exc_info=True)

    def _update_shap_individual_combo(self):
        """更新个体SHAP解释的接触者选择下拉框"""
        if not SHAP_AVAILABLE or not hasattr(self, 'shap_ind_contact_combo'):
            return

        if hasattr(self, 'ml_results_lock'):
            with self.ml_results_lock:
                ml_results = self.ml_results
        else:
            ml_results = self.ml_results

        names = []
        if ml_results:
            for item in ml_results.get('family', []):
                names.append(f"[家庭] {item.get('name', '未知')}")
            for item in ml_results.get('social', []):
                names.append(f"[社会] {item.get('name', '未知')}")

        self.shap_ind_contact_combo['values'] = names
        if names:
            self.shap_ind_contact_combo.current(0)

    def _draw_individual_shap(self):
        """绘制单个接触者的SHAP瀑布图（直接使用当前界面数据）"""
        if not SHAP_AVAILABLE or not hasattr(self, 'shap_ind_figure'):
            return

        selected = self.shap_ind_contact_var.get() if hasattr(self, 'shap_ind_contact_var') else ''
        if not selected:
            messagebox.showwarning("提示", "请先选择一个接触者")
            return

        try:
            is_family = selected.startswith('[家庭]')
            name = selected.replace('[家庭] ', '').replace('[社会] ', '')

            contact_data = None
            contact_type = 'family' if is_family else 'social'

            if is_family:
                # 直接从 family_entries 构建数据
                for entry in self.family_entries:
                    if entry.get('name') == name:
                        contact_data = {
                            'name': entry.get('name', ''),
                            'age': entry.get('age', 30),
                            'single_duration': entry.get('single_duration', 30),
                            'freq_density': entry.get('freq_density', 14),
                            'time_span': entry.get('time_span', 4),
                            'cumulative_exposure': calculate_cumulative_exposure(
                                entry.get('single_duration', 30),
                                entry.get('freq_density', 14),
                                entry.get('time_span', 4)
                            ),
                            'has_tb': self._to_bool(entry.get('has_tb', '否')),
                            'has_symptoms': self._to_bool(entry.get('has_symptoms', '否')),
                            'bcg_vaccine': self._to_bool(entry.get('bcg_vaccine', '是')),
                            'past_illness': self._to_bool(entry.get('past_illness', '否')),
                            'past_illness_type': self._convert_chinese_to_value(entry.get('past_illness_type', ''), self.ILLNESS_TYPE_MAPPING) if _is_yes(entry.get('past_illness', '否')) else "none",
                            'contact_distance': self._convert_chinese_to_value(entry.get('contact_distance', '近'), self.DISTANCE_MAPPING),
                            'ventilation': int(entry.get('ventilation', 3)),
                            'exposure_setting': self._convert_chinese_to_value(entry.get('exposure_setting', '一般'), self.SETTING_MAPPING)
                        }
                        break
            else:
                # 直接从 social_entries 构建数据
                for entry in self.social_entries:
                    if entry.get('name') == name:
                        contact_data = {
                            'name': entry.get('name', ''),
                            'age': entry.get('age', 30),
                            'single_duration': entry.get('single_duration', 30),
                            'freq_density': entry.get('freq_density', 2),
                            'time_span': entry.get('time_span', 4),
                            'cumulative_exposure': calculate_cumulative_exposure(
                                entry.get('single_duration', 30),
                                entry.get('freq_density', 2),
                                entry.get('time_span', 4)
                            ),
                            'is_high_risk': self._to_bool(entry.get('is_high_risk', '否')),
                            'has_tb': self._to_bool(entry.get('has_tb', '否')),
                            'past_illness': self._to_bool(entry.get('past_illness', '否')),
                            'past_illness_type': self._convert_chinese_to_value(entry.get('past_illness_type', ''), self.ILLNESS_TYPE_MAPPING) if _is_yes(entry.get('past_illness', '否')) else "none",
                            'bcg_vaccine': self._to_bool(entry.get('bcg_vaccine', '是')),
                            'ventilation': int(entry.get('ventilation', 3)),
                            'has_symptoms': self._to_bool(entry.get('has_symptoms', '否')),
                            'contact_distance': self._convert_chinese_to_value(entry.get('contact_distance', '中等'), self.DISTANCE_MAPPING),
                            'exposure_setting': self._convert_chinese_to_value(entry.get('exposure_setting', '一般'), self.SETTING_MAPPING)
                        }
                        break

            if contact_data is None or self.ml_predictor is None or not self.ml_predictor.is_trained:
                messagebox.showwarning("提示", "无法获取该接触者的数据")
                return

            if self.ml_predictor.shap_explainer is None:
                messagebox.showwarning("提示", "SHAP解释器不可用")
                return

            # 获取患者上下文信息
            patient_ftd = None
            patient_cough_freq = None
            contact_count = None
            if self.patient_info:
                basic_info = self.patient_info.get('basic_info', {})
                patient_ftd = basic_info.get('delay_days', 0)
                patient_cough_freq = basic_info.get('cough_freq', 0)
                contact_count = len(self.family_entries) + len(self.social_entries)

            features = self.ml_predictor.extract_features_with_interactions(
                contact_data, contact_type,
                patient_ftd=patient_ftd,
                patient_cough_freq=patient_cough_freq,
                contact_count=contact_count
            )

            shap_values = self.ml_predictor.shap_explainer.shap_values(features)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]
            elif isinstance(shap_values, list) and len(shap_values) == 1:
                shap_values = shap_values[0]
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                shap_values = shap_values[0]

            base_value = self.ml_predictor.shap_explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[-1] if len(base_value) > 1 else base_value[0]

            self.shap_ind_figure.clear()
            ax = self.shap_ind_figure.add_subplot(111)

            feature_descs = self.ml_predictor.ALL_FEATURE_DESCRIPTIONS
            feature_names = self.ml_predictor.ALL_FEATURE_NAMES

            contributions = list(zip(feature_names, features[0], shap_values))
            contributions.sort(key=lambda x: abs(x[2]))
            contributions = contributions[-15:]

            display_names = [feature_descs.get(c[0], c[0]) for c in contributions]
            shap_vals = [c[2] for c in contributions]
            feat_vals = [c[1] for c in contributions]

            y_pos = range(len(contributions))
            colors = ['#FF6B6B' if v > 0 else '#4ECDC4' for v in shap_vals]

            ax.barh(y_pos, shap_vals, color=colors, alpha=0.85, height=0.7)

            for i, (sv, fv) in enumerate(zip(shap_vals, feat_vals)):
                if isinstance(fv, float):
                    val_str = f"{fv:.2f}"
                else:
                    val_str = str(fv)
                offset = 0.01 * (1 if sv >= 0 else -1)
                ax.text(sv + offset, i, f" {val_str}", va='center', fontsize=8,
                       ha='left' if sv >= 0 else 'right', color='#333333')

            ax.set_yticks(y_pos)
            ax.set_yticklabels(display_names, fontsize=9)
            ax.set_xlabel('SHAP值（对预测风险的贡献）', fontsize=11)
            ax.set_title(f'{name} 的个体风险解释（SHAP瀑布图）', fontsize=12, fontweight='bold')
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

            pred = self.ml_predictor.predict_risk(contact_data, contact_type)
            if pred and 'ensemble' in pred:
                prob = pred['ensemble']['risk_probability']
                ax.text(0.98, 0.02, f'ML集成预测概率: {prob:.1f}%', transform=ax.transAxes,
                       ha='right', va='bottom', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

            self.shap_ind_figure.tight_layout()
            self.shap_ind_canvas.draw()
        except Exception as e:
            messagebox.showerror("错误", f"个体SHAP图生成失败: {str(e)}")

    # ==================== 统一风险归因条形图 ====================

    def _show_attribution_empty(self):
        """在风险归因图区域显示占位提示。"""
        if not (MATPLOTLIB_AVAILABLE and hasattr(self, 'attribution_figure')):
            return
        try:
            self.attribution_figure.clear()
            ax = self.attribution_figure.add_subplot(111)
            ax.text(0.5, 0.5, '暂无风险归因数据\n请先训练 ML/GNN 模型并重新评估',
                    ha='center', va='center', fontsize=12, color='#666666')
            ax.set_xticks([])
            ax.set_yticks([])
            self.attribution_canvas.draw()
        except Exception:
            LOGGER.debug("风险归因占位图绘制失败（非致命）", exc_info=True)

    def _draw_attribution_bar_chart(self):
        """绘制统一风险归因条形图（SHAP 因素贡献 + GNN 关键节点/边）。

        选取首个含 SHAP 归因的接触者，左图展示其因素贡献条形图，
        右图展示 GNN 关键节点重要性（若可用）。
        """
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'attribution_figure'):
            return

        if hasattr(self, 'ml_results_lock'):
            with self.ml_results_lock:
                ml_results = self.ml_results
        else:
            ml_results = self.ml_results

        if not ml_results:
            return

        attribution = ml_results.get('attribution', {})
        contacts = list(attribution.get('family', []) or []) + \
            list(attribution.get('social', []) or [])
        if not contacts:
            self._show_attribution_empty()
            return

        target = next((c for c in contacts if c.get('shap_contributions')), None)
        if target is None:
            self._show_attribution_empty()
            return

        try:
            self.attribution_figure.clear()

            contribs = target['shap_contributions'][:8][::-1]
            names = [c.get('description', c.get('feature', '')) for c in contribs]
            vals = [c.get('contribution', 0) for c in contribs]
            colors = ['#FF6B6B' if v > 0 else '#4ECDC4' for v in vals]

            gnn = target.get('gnn_explanation') or {}
            nodes = gnn.get('node_importance', []) or []

            if nodes:
                ax = self.attribution_figure.add_subplot(121)
            else:
                ax = self.attribution_figure.add_subplot(111)

            ax.barh(range(len(vals)), vals, color=colors, alpha=0.85)
            ax.set_yticks(range(len(vals)))
            ax.set_yticklabels(names, fontsize=9)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('SHAP贡献值')
            ax.set_title(f"{target.get('name', '接触者')} 的风险因素归因（SHAP）")

            if nodes:
                ax2 = self.attribution_figure.add_subplot(122)
                node_names = [n.get('label', f'节点{n.get("node_idx", "")}') for n in nodes[:6][::-1]]
                node_vals = [n.get('importance', 0) for n in nodes[:6][::-1]]
                ax2.barh(range(len(node_vals)), node_vals, color='#45B7D1', alpha=0.85)
                ax2.set_yticks(range(len(node_vals)))
                ax2.set_yticklabels(node_names, fontsize=8)
                ax2.set_xlabel('节点重要性')
                ax2.set_title('GNN关键节点')

            self.attribution_figure.tight_layout()
            self.attribution_canvas.draw()
        except Exception as e:
            LOGGER.warning("风险归因图绘制失败: %s", e, exc_info=True)
