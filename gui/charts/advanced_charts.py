#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 高级图表 Mixin（优先级九）

实现 MCMC 诊断图表和 GNN 网络拓扑图，补齐 _update_all_charts 中预留的调用入口。

图表说明：
  - _draw_mcmc_diagnostics: 2x2 子图布局
      * 左上：后验分布直方图（各参数边缘后验分布，含先验曲线叠加）
      * 右上：踪迹图（MCMC 采样链的迭代-参数值轨迹）
      * 左下：R-hat 收敛诊断条形图
      * 右下：ESS 有效样本量条形图
  - _draw_gnn_topology: 使用 networkx 构建接触网络图
      * 患者为中心节点，接触者为叶节点
      * 边权重 = 接触频率 × 时长
      * 节点颜色按风险等级着色，节点大小按累积暴露剂量缩放
"""

from ._shared import *


class AdvancedChartMixin:
    """高级图表方法混合类 — MCMC 诊断 + GNN 拓扑"""

    # ==================================================================
    # MCMC 诊断图表
    # ==================================================================

    def _draw_mcmc_diagnostics(self):
        """优先级九：绘制 MCMC 诊断图（2x2 子图布局）

        数据来源为 self.seir_inference（BayesianSEIRInference 实例）的
        posterior_samples 和 convergence_diagnostics 属性。

        子图布局：
          - 左上：后验分布直方图（各参数边缘后验分布 + 先验曲线叠加）
          - 右上：踪迹图（MCMC 采样链的迭代-参数值轨迹）
          - 左下：R-hat 收敛诊断条形图（<1.1 收敛良好）
          - 右下：ESS 有效样本量条形图（>400 为良好）
        """
        if not (MATPLOTLIB_AVAILABLE and NUMPY_AVAILABLE):
            return
        if not hasattr(self, 'mcmc_diag_figure'):
            return
        # 需要 seir_inference 且有后验采样
        seir_inf = getattr(self, 'seir_inference', None)
        if seir_inf is None:
            self._draw_mcmc_placeholder("未运行贝叶斯 MCMC 推断\n请先运行 SEIR 贝叶斯推断")
            return
        posterior = getattr(seir_inf, 'posterior_samples', None)
        if not posterior:
            self._draw_mcmc_placeholder("后验采样为空\n请先运行 SEIR 贝叶斯推断")
            return

        try:
            self.mcmc_diag_figure.clear()

            # 2x2 子图布局
            ax_post = self.mcmc_diag_figure.add_subplot(2, 2, 1)
            ax_trace = self.mcmc_diag_figure.add_subplot(2, 2, 2)
            ax_rhat = self.mcmc_diag_figure.add_subplot(2, 2, 3)
            ax_ess = self.mcmc_diag_figure.add_subplot(2, 2, 4)

            # 收集参数名（过滤掉非数值或空采样的参数）
            param_names = [k for k, v in posterior.items()
                          if isinstance(v, (list, np.ndarray)) and len(v) > 0]
            if not param_names:
                self._draw_mcmc_placeholder("后验采样无可视化参数")
                return

            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

            # --- 左上：后验分布直方图 ---
            for i, name in enumerate(param_names[:6]):  # 最多显示 6 个参数
                samples = np.asarray(posterior[name], dtype=float)
                if samples.size == 0:
                    continue
                color = colors[i % len(colors)]
                ax_post.hist(samples, bins=30, alpha=0.5, color=color, label=name, density=True)

            ax_post.set_xlabel('参数值', fontsize=9)
            ax_post.set_ylabel('密度', fontsize=9)
            ax_post.set_title('后验分布直方图', fontsize=11, fontweight='bold')
            ax_post.legend(fontsize=7, loc='upper right')
            ax_post.grid(True, alpha=0.3)

            # --- 右上：踪迹图 ---
            for i, name in enumerate(param_names[:6]):
                samples = np.asarray(posterior[name], dtype=float)
                if samples.size == 0:
                    continue
                color = colors[i % len(colors)]
                iterations = np.arange(samples.size)
                ax_trace.plot(iterations, samples, linewidth=0.5, alpha=0.7,
                            color=color, label=name)

            ax_trace.set_xlabel('迭代次数', fontsize=9)
            ax_trace.set_ylabel('参数值', fontsize=9)
            ax_trace.set_title('MCMC 采样踪迹图', fontsize=11, fontweight='bold')
            ax_trace.legend(fontsize=7, loc='upper right')
            ax_trace.grid(True, alpha=0.3)

            # --- 左下：R-hat 收敛诊断条形图 ---
            conv_diag = getattr(seir_inf, 'convergence_diagnostics', {}) or {}
            rhat_dict = conv_diag.get('r_hat', {}) if isinstance(conv_diag, dict) else {}

            if rhat_dict:
                names = list(rhat_dict.keys())
                rhats = [float(rhat_dict[n]) if rhat_dict[n] is not None else np.nan
                        for n in names]
                bar_colors = ['#2ecc71' if (r is not np.nan and r < 1.1) else '#e74c3c'
                             for r in rhats]
                bars = ax_rhat.bar(range(len(names)), rhats, color=bar_colors, alpha=0.8)
                ax_rhat.axhline(y=1.1, color='#e74c3c', linestyle='--', linewidth=1.5,
                               label='阈值 1.1')
                ax_rhat.set_xticks(range(len(names)))
                ax_rhat.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
                ax_rhat.set_ylabel('R-hat', fontsize=9)
                ax_rhat.set_title('R-hat 收敛诊断（<1.1 良好）', fontsize=11, fontweight='bold')
                ax_rhat.legend(fontsize=8)
                ax_rhat.grid(True, alpha=0.3, axis='y')
            else:
                ax_rhat.text(0.5, 0.5, 'R-hat 数据不可用',
                           transform=ax_rhat.transAxes, ha='center', va='center',
                           fontsize=10, color='gray')
                ax_rhat.set_title('R-hat 收敛诊断', fontsize=11, fontweight='bold')

            # --- 右下：ESS 有效样本量条形图 ---
            ess_dict = conv_diag.get('ess', {}) if isinstance(conv_diag, dict) else {}

            if ess_dict:
                names = list(ess_dict.keys())
                esss = [float(ess_dict[n]) if ess_dict[n] is not None else 0.0
                       for n in names]
                bar_colors = ['#2ecc71' if e > 400 else ('#f39c12' if e > 100 else '#e74c3c')
                             for e in esss]
                ax_ess.bar(range(len(names)), esss, color=bar_colors, alpha=0.8)
                ax_ess.axhline(y=400, color='#2ecc71', linestyle='--', linewidth=1.5,
                              label='阈值 400')
                ax_ess.set_xticks(range(len(names)))
                ax_ess.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
                ax_ess.set_ylabel('ESS', fontsize=9)
                ax_ess.set_title('有效样本量 ESS（>400 良好）', fontsize=11, fontweight='bold')
                ax_ess.legend(fontsize=8)
                ax_ess.grid(True, alpha=0.3, axis='y')
            else:
                ax_ess.text(0.5, 0.5, 'ESS 数据不可用',
                          transform=ax_ess.transAxes, ha='center', va='center',
                          fontsize=10, color='gray')
                ax_ess.set_title('有效样本量 ESS', fontsize=11, fontweight='bold')

            self.mcmc_diag_figure.tight_layout(pad=1.5)
            canvas = getattr(self, 'mcmc_diag_canvas', None)
            if canvas is not None:
                canvas.draw()
        except Exception as e:
            LOGGER.warning("绘制 MCMC 诊断图失败: %s", e, exc_info=True)
            self._draw_mcmc_placeholder(f"绘制失败: {e}")

    def _draw_mcmc_placeholder(self, message):
        """绘制 MCMC 诊断图占位提示"""
        if not hasattr(self, 'mcmc_diag_figure'):
            return
        try:
            self.mcmc_diag_figure.clear()
            ax = self.mcmc_diag_figure.add_subplot(111)
            ax.text(0.5, 0.5, message, transform=ax.transAxes,
                   ha='center', va='center', fontsize=14, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('MCMC 诊断图', fontsize=12, fontweight='bold')
            canvas = getattr(self, 'mcmc_diag_canvas', None)
            if canvas is not None:
                canvas.draw()
        except Exception:
            pass

    # ==================================================================
    # GNN 网络拓扑图
    # ==================================================================

    def _draw_gnn_topology(self):
        """优先级九：绘制 GNN 网络拓扑图

        使用 networkx 从 family_entries 和 social_entries 构建图：
          - 患者为中心节点
          - 接触者为叶节点
          - 边权重 = 接触频率 × 时长
          - 节点颜色按风险等级着色
          - 节点大小按累积暴露剂量缩放
        """
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            return
        if not hasattr(self, 'gnn_topology_figure'):
            return

        # networkx 可选依赖
        try:
            import networkx as nx
        except ImportError:
            self._draw_gnn_placeholder("networkx 未安装\n无法绘制网络拓扑图")
            return

        try:
            family = getattr(self, 'family_entries', []) or []
            social = getattr(self, 'social_entries', []) or []

            if not family and not social:
                self._draw_gnn_placeholder("无接触者数据\n请添加家庭成员或社会接触者")
                return

            # 构建图
            G = nx.Graph()

            # 添加患者中心节点
            patient_name = "患者"
            G.add_node(patient_name, node_type='patient', risk=1.0, exposure=1.0)

            # 潜在患者数据（用于风险着色）
            potential = (self.results or {}).get('potential_patients', {})
            family_pp = {p.get('name', f'家庭{i}'): p
                        for i, p in enumerate(potential.get('family', []))}
            social_pp = {p.get('name', f'社会{i}'): p
                        for i, p in enumerate(potential.get('social', []))}

            # 添加家庭成员节点和边
            for i, entry in enumerate(family):
                name = entry.get('name', f'家庭{i}')
                # 风险等级：从潜在患者数据获取，否则按是否有症状估算
                pp = family_pp.get(name, {})
                risk_prob = pp.get('disease_probability', 0) / 100.0
                # 累积暴露剂量 = freq_density * time_span * single_duration
                freq = float(entry.get('freq_density', 1) or 1)
                span = float(entry.get('time_span', 1) or 1)
                duration = float(entry.get('single_duration', 30) or 30)
                exposure = freq * span * duration

                G.add_node(name, node_type='family', risk=risk_prob, exposure=exposure)
                # 边权重 = 接触频率 × 时长
                edge_weight = freq * span
                G.add_edge(patient_name, name, weight=edge_weight, contact_type='family')

            # 添加社会接触者节点和边
            for i, entry in enumerate(social):
                name = entry.get('name', f'社会{i}')
                pp = social_pp.get(name, {})
                risk_prob = pp.get('disease_probability', 0) / 100.0
                freq = float(entry.get('freq_density', 1) or 1)
                span = float(entry.get('time_span', 1) or 1)
                duration = float(entry.get('single_duration', 30) or 30)
                exposure = freq * span * duration

                G.add_node(name, node_type='social', risk=risk_prob, exposure=exposure)
                edge_weight = freq * span
                G.add_edge(patient_name, name, weight=edge_weight, contact_type='social')

            # 绘制
            self.gnn_topology_figure.clear()
            ax = self.gnn_topology_figure.add_subplot(111)

            # 力导向布局
            pos = nx.spring_layout(G, k=2.0/np.sqrt(max(G.number_of_nodes(), 1)),
                                   iterations=50, seed=42)

            # 节点颜色：按风险等级着色
            node_colors = []
            for node in G.nodes():
                if G.nodes[node].get('node_type') == 'patient':
                    node_colors.append('#2c3e50')  # 患者中心：深蓝灰
                else:
                    risk = G.nodes[node].get('risk', 0)
                    if risk >= 0.4:
                        node_colors.append('#e74c3c')  # 高风险：红
                    elif risk >= 0.2:
                        node_colors.append('#f39c12')  # 中风险：橙
                    else:
                        node_colors.append('#2ecc71')  # 低风险：绿

            # 节点大小：按累积暴露剂量缩放
            node_sizes = []
            for node in G.nodes():
                if G.nodes[node].get('node_type') == 'patient':
                    node_sizes.append(1500)  # 患者中心节点最大
                else:
                    exposure = G.nodes[node].get('exposure', 100)
                    # 对数缩放 + 基线，避免太小
                    size = 200 + np.log1p(exposure) * 50
                    node_sizes.append(min(size, 1200))

            # 边样式：家庭实线，社会虚线
            family_edges = [(u, v) for u, v, d in G.edges(data=True)
                           if d.get('contact_type') == 'family']
            social_edges = [(u, v) for u, v, d in G.edges(data=True)
                           if d.get('contact_type') == 'social']

            # 绘制边
            nx.draw_networkx_edges(G, pos, edgelist=family_edges, ax=ax,
                                  edge_color='#3498db', width=1.5, style='solid',
                                  alpha=0.6)
            nx.draw_networkx_edges(G, pos, edgelist=social_edges, ax=ax,
                                  edge_color='#95a5a6', width=1.0, style='dashed',
                                  alpha=0.5)

            # 绘制节点
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                                  node_size=node_sizes, alpha=0.85,
                                  edgecolors='white', linewidths=1.5)

            # 绘制标签
            labels = {node: node for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8,
                                   font_family='Microsoft YaHei', font_color='white')

            ax.set_title('接触网络拓扑图（GNN 输入可视化）', fontsize=12, fontweight='bold')
            ax.axis('off')

            # 图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2c3e50', label='患者（中心）'),
                Patch(facecolor='#e74c3c', label='高风险 (≥40%)'),
                Patch(facecolor='#f39c12', label='中风险 (20-40%)'),
                Patch(facecolor='#2ecc71', label='低风险 (<20%)'),
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
                     framealpha=0.9)

            self.gnn_topology_figure.tight_layout(pad=1.0)
            canvas = getattr(self, 'gnn_topology_canvas', None)
            if canvas is not None:
                canvas.draw()
        except Exception as e:
            LOGGER.warning("绘制 GNN 拓扑图失败: %s", e, exc_info=True)
            self._draw_gnn_placeholder(f"绘制失败: {e}")

    def _draw_gnn_placeholder(self, message):
        """绘制 GNN 拓扑图占位提示"""
        if not hasattr(self, 'gnn_topology_figure'):
            return
        try:
            self.gnn_topology_figure.clear()
            ax = self.gnn_topology_figure.add_subplot(111)
            ax.text(0.5, 0.5, message, transform=ax.transAxes,
                   ha='center', va='center', fontsize=14, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('接触网络拓扑图', fontsize=12, fontweight='bold')
            canvas = getattr(self, 'gnn_topology_canvas', None)
            if canvas is not None:
                canvas.draw()
        except Exception:
            pass


__all__ = ['AdvancedChartMixin']
