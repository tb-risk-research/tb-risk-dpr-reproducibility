#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因果 DAG 可视化 Mixin（_CausalChartsMixin）"""

from .._shared import *  # noqa: F401,F403


class _CausalChartsMixin:
    """因果 DAG 可视化方法"""

    def _draw_causal_dag(self):
        """绘制因果DAG（有向无环图）可视化

        基于Du et al. (2023) 结核病因果DAG设计，展示变量间的因果关系。
        """
        if not hasattr(self, 'dag_figure'):
            return

        try:
            self.dag_figure.clear()
            ax = self.dag_figure.add_subplot(111)
            ax.axis('off')

            # 定义节点位置和样式
            nodes = {
                'age': {'pos': (0.1, 0.8), 'label': '年龄', 'color': '#FFE4B5'},
                'bcg': {'pos': (0.1, 0.6), 'label': 'BCG接种', 'color': '#90EE90'},
                'past_illness': {'pos': (0.1, 0.4), 'label': '基础疾病', 'color': '#FFB6C1'},
                'hiv': {'pos': (0.05, 0.25), 'label': 'HIV', 'color': '#DDA0DD'},
                'diabetes': {'pos': (0.15, 0.25), 'label': '糖尿病', 'color': '#DDA0DD'},

                'exposure': {'pos': (0.5, 0.7), 'label': '暴露强度', 'color': '#87CEEB'},
                'ventilation': {'pos': (0.5, 0.5), 'label': '通风条件', 'color': '#F0E68C'},
                'distance': {'pos': (0.5, 0.3), 'label': '接触距离', 'color': '#F0E68C'},
                'symptoms': {'pos': (0.5, 0.1), 'label': '患者症状', 'color': '#FFDAB9'},

                'risk': {'pos': (0.9, 0.5), 'label': 'TB感染风险', 'color': '#FF6B6B'}
            }

            # 定义边（因果关系）
            edges = [
                ('age', 'risk'),
                ('age', 'past_illness'),
                ('bcg', 'risk'),
                ('past_illness', 'risk'),
                ('hiv', 'past_illness'),
                ('diabetes', 'past_illness'),
                ('exposure', 'risk'),
                ('ventilation', 'risk'),
                ('distance', 'risk'),
                ('symptoms', 'exposure'),
                ('symptoms', 'risk')
            ]

            # 绘制边
            for from_node, to_node in edges:
                from_pos = nodes[from_node]['pos']
                to_pos = nodes[to_node]['pos']

                # 绘制箭头
                ax.annotate('',
                           xy=to_pos, xycoords='data',
                           xytext=from_pos, textcoords='data',
                           arrowprops=dict(arrowstyle="->", color='#666666', lw=2, alpha=0.7))

            # 绘制节点
            for node_name, node_info in nodes.items():
                x, y = node_info['pos']
                ax.scatter(x, y, s=800, c=node_info['color'], edgecolors='#333333',
                          linewidth=2, zorder=10)
                ax.text(x, y, node_info['label'], ha='center', va='center',
                       fontsize=10, fontweight='bold', zorder=11)

            # 添加标题和图例
            ax.set_title('结核病感染风险因果DAG\n(Du et al., 2023改编)',
                        fontsize=14, fontweight='bold', pad=20)

            # 添加图例说明
            legend_text = (
                '节点类型:\n'
                '● 人口学特征\n'
                '● 保护因素\n'
                '● 风险因素\n'
                '● 暴露条件\n'
                '● 结局变量'
            )
            ax.text(0.02, 0.02, legend_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            self.dag_figure.tight_layout()
            self.dag_canvas.draw()
        except Exception as e:
            LOGGER.warning("因果DAG绘制失败: %s", e, exc_info=True)
