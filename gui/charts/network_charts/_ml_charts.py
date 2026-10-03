#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML 图表绘制 Mixin（_MLChartsMixin）

包含：传统评分与ML评分对比图、模型性能对比图、校准曲线
"""

from .._shared import *  # noqa: F401,F403


class _MLChartsMixin:
    """ML 图表绘制方法"""

    def _draw_ml_comparison_chart(self):
        """绘制传统评分与ML评分对比图"""
        # 原子性读取：在锁保护下获取快照
        if hasattr(self, 'ml_results_lock'):
            with self.ml_results_lock:
                ml_results = self.ml_results
        else:
            ml_results = self.ml_results

        if not hasattr(self, 'ml_compare_figure') or not ml_results:
            return

        try:
            self.ml_compare_figure.clear()

            all_items = []
            for item in ml_results.get('family', []):
                name = item.get('name', '未知')
                trad_prob = item.get('traditional_prob', 0)
                ml_preds = item.get('ml_predictions', {})
                all_items.append((name, trad_prob, ml_preds, '家庭'))
            for item in ml_results.get('social', []):
                name = item.get('name', '未知')
                trad_prob = item.get('traditional_prob', 0)
                ml_preds = item.get('ml_predictions', {})
                all_items.append((name, trad_prob, ml_preds, '社会'))

            if not all_items:
                ax = self.ml_compare_figure.add_subplot(111)
                ax.text(0.5, 0.5, '无接触者数据', ha='center', va='center', fontsize=14, color='gray')
                self.ml_compare_canvas.draw()
                return

            names = [item[0] for item in all_items]
            trad_probs = [item[1] for item in all_items]
            ml_ensemble_probs = [item[2].get('ensemble', {}).get('risk_probability', 0) for item in all_items]

            # 改进9：不再手动 set_size_inches（原先强制最小 10 英寸宽导致右侧裁切、边框不完整）。
            # figure 尺寸由 _bind_configure_redraw 按容器实际宽高自动同步铺满。

            x = np.arange(len(names))
            width = 0.35

            ax = self.ml_compare_figure.add_subplot(111)
            bars1 = ax.bar(x - width/2, trad_probs, width, label='传统Sigmoid评分', color='#4ECDC4', alpha=0.85)
            bars2 = ax.bar(x + width/2, ml_ensemble_probs, width, label='ML集成评分', color='#FF6B6B', alpha=0.85)

            ax.set_ylabel('发病概率 (%)')
            ax.set_title('传统评分 vs 机器学习评分对比')
            ax.set_xticks(x)
            max_name_len = max(len(n) for n in names) if names else 4
            if max_name_len > 6:
                ax.set_xticklabels(names, rotation=60, ha='right', fontsize=7)
            else:
                ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax.legend(loc='upper right')
            ax.set_ylim(0, max(max(trad_probs, default=0), max(ml_ensemble_probs, default=0)) * 1.3 + 5)

            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)

            self.ml_compare_figure.tight_layout()
            self.ml_compare_canvas.draw()
        except Exception as e:
            LOGGER.warning("ML对比图绘制失败: %s", e, exc_info=True)

    def _draw_ml_performance_chart(self):
        """绘制模型性能对比图"""
        if hasattr(self, 'ml_results_lock'):
            with self.ml_results_lock:
                ml_results = self.ml_results
        else:
            ml_results = self.ml_results

        if not hasattr(self, 'ml_perf_figure') or not ml_results:
            return

        perf = ml_results.get('model_performance', {})
        if not perf:
            return

        try:
            self.ml_perf_figure.clear()

            model_names = []
            auroc_values = []
            auprc_values = []

            for model_name, metrics in perf.items():
                model_names.append(metrics.get('name', model_name))
                auroc_values.append(metrics.get('AUROC', 0))
                auprc_values.append(metrics.get('AUPRC', 0))

            x = np.arange(len(model_names))
            width = 0.35

            ax = self.ml_perf_figure.add_subplot(111)
            bars1 = ax.bar(x - width/2, auroc_values, width, label='AUROC', color='#45B7D1', alpha=0.85)
            bars2 = ax.bar(x + width/2, auprc_values, width, label='AUPRC', color='#FFA07A', alpha=0.85)

            ax.set_ylabel('性能指标值')
            ax.set_title('机器学习模型性能对比（5折交叉验证）')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names)
            ax.legend(loc='lower right')
            ax.set_ylim(0, 1.05)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='随机基线')

            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

            self.ml_perf_figure.tight_layout()
            self.ml_perf_canvas.draw()
        except Exception as e:
            LOGGER.warning("ML性能图绘制失败: %s", e, exc_info=True)

    def _draw_calibration_chart(self):
        """绘制模型校准曲线和Brier评分"""
        if not hasattr(self, 'ml_calib_figure') or self.ml_predictor is None:
            return

        try:
            calib_results = self.ml_predictor.compute_calibration()
            if not calib_results:
                ax = self.ml_calib_figure.add_subplot(111)
                ax.text(0.5, 0.5, '校准数据不可用', ha='center', va='center', fontsize=14, color='gray')
                self.ml_calib_canvas.draw()
                return

            self.ml_calib_figure.clear()

            ax1 = self.ml_calib_figure.add_subplot(121)

            ax1.plot([0, 1], [0, 1], "k:", label="完美校准", linewidth=2)

            colors = ['#4ECDC4', '#FF6B6B', '#45B7D1']
            for i, (model_name, calib) in enumerate(calib_results.items()):
                color = colors[i % len(colors)]
                ax1.plot(calib.get('mean_predicted_value', []), calib.get('fraction_of_positives', []),
                        "s-", color=color, label=f"{calib.get('name', model_name)} (Brier={calib.get('brier_score', 0):.4f})",
                        linewidth=2, markersize=6)

            ax1.set_xlabel("平均预测概率", fontsize=11)
            ax1.set_ylabel("实际正例比例", fontsize=11)
            ax1.set_title("校准曲线（Reliability Diagram）", fontsize=12, fontweight='bold')
            ax1.legend(loc="lower right", fontsize=9)
            ax1.set_xlim([-0.05, 1.05])
            ax1.set_ylim([-0.05, 1.05])
            ax1.grid(True, alpha=0.3)

            ax2 = self.ml_calib_figure.add_subplot(122)
            model_names = [calib.get('name', str(i)) for i, calib in enumerate(calib_results.values())]
            brier_scores = [calib.get('brier_score', 0) for calib in calib_results.values()]

            bars = ax2.bar(range(len(model_names)), brier_scores,
                          color=colors[:len(model_names)], alpha=0.85)
            ax2.set_xticks(range(len(model_names)))
            ax2.set_xticklabels(model_names, fontsize=9)
            ax2.set_ylabel("Brier评分（越低越好）", fontsize=11)
            ax2.set_title("Brier评分对比", fontsize=12, fontweight='bold')
            ax2.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='随机基线(0.25)')
            ax2.legend(loc='upper right', fontsize=9)

            for bar, score in zip(bars, brier_scores):
                ax2.annotate(f'{score:.4f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

            self.ml_calib_figure.tight_layout()
            self.ml_calib_canvas.draw()
        except Exception as e:
            LOGGER.warning("校准曲线绘制失败: %s", e, exc_info=True)
