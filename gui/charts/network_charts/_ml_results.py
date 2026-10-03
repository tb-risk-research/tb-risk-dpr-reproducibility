#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML 结果展示与图表更新 Mixin（_MLResultsMixin）

包含：ML结果文本展示、ML图表批量更新编排
"""

from .._shared import *  # noqa: F401,F403


class _MLResultsMixin:
    """ML 结果展示与图表更新方法"""

    def _display_ml_results(self):
        """在ML结果文本区显示机器学习预测结果"""
        # 原子性读取：在锁保护下获取快照，避免与后台写入竞态
        if hasattr(self, 'ml_results_lock'):
            with self.ml_results_lock:
                ml_results = self.ml_results
        else:
            ml_results = self.ml_results

        if not hasattr(self, 'ml_result_text') or not ml_results:
            return

        self.ml_result_text.delete('1.0', 'end')

        self.ml_result_text.insert('end', "=== 机器学习风险分层结果 ===\n\n", 'title')

        perf = ml_results.get('model_performance', {})
        if perf:
            self.ml_result_text.insert('end', "模型性能（5折交叉验证）:\n", 'heading')
            self.ml_result_text.insert('end', f"  训练样本量: {ml_results.get('training_sample_count', 0)}\n")
            for model_name, metrics in perf.items():
                self.ml_result_text.insert('end',
                    f"  {metrics.get('name', model_name)}: AUROC={metrics.get('AUROC', 0):.3f}, AUPRC={metrics.get('AUPRC', 0):.3f}\n")

            if self.ml_predictor and self.ml_predictor.is_trained:
                ci_results = self.ml_predictor.compute_bootstrap_ci(n_bootstrap=50)
                if ci_results:
                    self.ml_result_text.insert('end', "\nBootstrap 95%置信区间:\n", 'heading')
                    for model_name, ci in ci_results.items():
                        ci_name = ci.get('name', model_name)
                        auroc_mean = ci.get('AUROC_mean', 0)
                        auroc_ci_lower = ci.get('AUROC_ci_lower', 0)
                        auroc_ci_upper = ci.get('AUROC_ci_upper', 0)
                        auprc_mean = ci.get('AUPRC_mean', 0)
                        auprc_ci_lower = ci.get('AUPRC_ci_lower', 0)
                        auprc_ci_upper = ci.get('AUPRC_ci_upper', 0)
                        self.ml_result_text.insert('end',
                            f"  {ci_name}: AUROC={auroc_mean:.3f} "
                            f"[{auroc_ci_lower:.3f}-{auroc_ci_upper:.3f}], "
                            f"AUPRC={auprc_mean:.3f} "
                            f"[{auprc_ci_lower:.3f}-{auprc_ci_upper:.3f}]\n")

            self.ml_result_text.insert('end', "\n")
            self.ml_result_text.insert('end', "注意: 当前模型基于合成数据训练，实际使用需用本地数据重新训练。\n", 'warning')

        if ml_results.get('family'):
            self.ml_result_text.insert('end', "家庭接触者ML预测:\n", 'heading')
            for item in ml_results['family']:
                name = item.get('name', '未知')
                trad_prob = item.get('traditional_prob', 0)
                ml_preds = item.get('ml_predictions', {})
                self.ml_result_text.insert('end', f"  {name}:\n")
                self.ml_result_text.insert('end', f"    传统评分: {trad_prob:.1f}%\n")
                ensemble = ml_preds.get('ensemble', {})
                if ensemble:
                    self.ml_result_text.insert('end',
                        f"    ML集成: {ensemble.get('risk_probability', 0):.1f}% "
                        f"(最佳模型: {ensemble.get('best_model', 'N/A')}, {ensemble.get('best_model_probability', 0):.1f}%)\n")
                for mk, mp in ml_preds.items():
                    if mk not in ['ensemble']:
                        self.ml_result_text.insert('end',
                            f"    {mp.get('model_name', mk)}: {mp.get('risk_probability', 0):.1f}%\n")

        if ml_results.get('social'):
            self.ml_result_text.insert('end', "\n社会接触者ML预测:\n", 'heading')
            for item in ml_results['social']:
                name = item.get('name', '未知')
                trad_prob = item.get('traditional_prob', 0)
                ml_preds = item.get('ml_predictions', {})
                self.ml_result_text.insert('end', f"  {name}:\n")
                self.ml_result_text.insert('end', f"    传统评分: {trad_prob:.1f}%\n")
                ensemble = ml_preds.get('ensemble', {})
                if ensemble:
                    self.ml_result_text.insert('end',
                        f"    ML集成: {ensemble.get('risk_probability', 0):.1f}%\n")

        # 显示三方向集成结果
        if ml_results.get('ensemble', {}).get('family'):
            self.ml_result_text.insert('end', "\n三方向集成（家庭接触者）:\n", 'heading')
            for item in ml_results['ensemble']['family']:
                name = item.get('name', '未知')
                trad_prob = item.get('traditional_prob', 0)
                ensemble_result = item.get('ensemble_predictions', {}).get('ensemble', {})
                self.ml_result_text.insert('end', f"  {name}:\n")
                self.ml_result_text.insert('end', f"    传统评分: {trad_prob:.1f}%\n")
                if ensemble_result:
                    self.ml_result_text.insert('end',
                        f"    三方向集成: {ensemble_result.get('risk_probability', 0):.1f}%\n")
                    # 显示各方向贡献
                    weights = ensemble_result.get('weights', {})
                    weight_strs = []
                    if 'ml' in weights:
                        weight_strs.append(f"ML: {weights.get('ml', 0):.1f}")
                    if 'seir' in weights:
                        weight_strs.append(f"SEIR: {weights.get('seir', 0):.1f}")
                    if 'gnn' in weights:
                        weight_strs.append(f"GNN: {weights.get('gnn', 0):.1f}")
                    if weight_strs:
                        self.ml_result_text.insert('end', f"    权重: {', '.join(weight_strs)}\n")

        if ml_results.get('ensemble', {}).get('social'):
            self.ml_result_text.insert('end', "\n三方向集成（社会接触者）:\n", 'heading')
            for item in ml_results['ensemble']['social']:
                name = item.get('name', '未知')
                trad_prob = item.get('traditional_prob', 0)
                ensemble_result = item.get('ensemble_predictions', {}).get('ensemble', {})
                self.ml_result_text.insert('end', f"  {name}:\n")
                self.ml_result_text.insert('end', f"    传统评分: {trad_prob:.1f}%\n")
                if ensemble_result:
                    self.ml_result_text.insert('end',
                        f"    三方向集成: {ensemble_result.get('risk_probability', 0):.1f}%\n")

        # 显示GNN预测结果（如果有）
        if ml_results.get('gnn', {}).get('family') or ml_results.get('gnn', {}).get('social'):
            self.ml_result_text.insert('end', "\nGNN预测结果:\n", 'heading')
            if ml_results.get('gnn', {}).get('family'):
                for item in ml_results['gnn']['family']:
                    name = item.get('name', '未知')
                    gnn_pred = item.get('gnn_prediction', {})
                    if gnn_pred:
                        self.ml_result_text.insert('end', f"  {name}: "
                            f"GNN: {gnn_pred.get('risk_probability', 0):.1f}%\n")
            if ml_results.get('gnn', {}).get('social'):
                for item in ml_results['gnn']['social']:
                    name = item.get('name', '未知')
                    gnn_pred = item.get('gnn_prediction', {})
                    if gnn_pred:
                        self.ml_result_text.insert('end', f"  {name}: "
                            f"GNN: {gnn_pred.get('risk_probability', 0):.1f}%\n")

        shap_info = ml_results.get('shap_analysis', {})
        if shap_info and shap_info.get('feature_importance'):
            self.ml_result_text.insert('end', "\nSHAP特征重要性排名:\n", 'heading')
            for fname, importance in shap_info.get('feature_importance', [])[:5]:
                desc = shap_info.get('feature_descriptions', {}).get(fname, fname)
                self.ml_result_text.insert('end', f"  {desc}: {importance:.4f}\n")

        self.ml_result_text.tag_configure('title', font=('Arial', 11, 'bold'))
        self.ml_result_text.tag_configure('heading', font=('Arial', 10, 'bold'))
        self.ml_result_text.tag_configure('warning', font=('Arial', 9, 'italic'), foreground='#CC6600')

        if ml_results.get('multistate'):
            multistate_data = ml_results['multistate']
            self.ml_result_text.insert('end', "\n竞争风险多结局预测 (DeepHit + Fine-Gray):\n", 'heading')

            for item in multistate_data.get('family', []):
                name = item.get('name', '未知')
                self.ml_result_text.insert('end', f"  {name} (家庭):\n")
                pred = item.get('multistate_prediction', {})
                cause_probs = pred.get('cause_probs', {})
                for label, info in cause_probs.items():
                    prob = info.get('final_prob', 0)
                    self.ml_result_text.insert('end',
                        f"    {label}: {prob*100:.1f}%\n")
                dominant = pred.get('dominant_label', '未知')
                high_conf = '是' if pred.get('high_confidence') else '否'
                self.ml_result_text.insert('end',
                    f"    主导结局: {dominant} | 高置信度: {high_conf}\n")

                rec = item.get('clinical_recommendation', {})
                if rec:
                    self.ml_result_text.insert('end',
                        f"    建议: {rec.get('recommendation', '')}\n")

            for item in multistate_data.get('social', []):
                name = item.get('name', '未知')
                self.ml_result_text.insert('end', f"  {name} (社会):\n")
                pred = item.get('multistate_prediction', {})
                cause_probs = pred.get('cause_probs', {})
                for label, info in cause_probs.items():
                    prob = info.get('final_prob', 0)
                    self.ml_result_text.insert('end',
                        f"    {label}: {prob*100:.1f}%\n")
                dominant = pred.get('dominant_label', '未知')
                self.ml_result_text.insert('end',
                    f"    主导结局: {dominant}\n")

                rec = item.get('clinical_recommendation', {})
                if rec:
                    self.ml_result_text.insert('end',
                        f"    建议: {rec.get('recommendation', '')}\n")

            eval_metrics = multistate_data.get('eval_metrics', {})
            if eval_metrics:
                self.ml_result_text.insert('end', "\n模型性能 (原因别C-index):\n", 'heading')
                for label, m in eval_metrics.items():
                    self.ml_result_text.insert('end',
                        f"  {label}: C-index={m.get('c_index', 0):.4f}, "
                        f"AUC(52w)={m.get('auc_52w', 0):.4f}\n")

    def _update_ml_charts(self):
        """更新所有ML相关图表"""
        # 原子性读取：在锁保护下获取快照
        if hasattr(self, 'ml_results_lock'):
            with self.ml_results_lock:
                ml_results = self.ml_results
        else:
            ml_results = self.ml_results

        if not MATPLOTLIB_AVAILABLE or not ml_results:
            return

        # 使用独立异常处理，一个图表失败不影响其他图表
        charts = [
            ("传统评分与ML评分对比图", self._draw_ml_comparison_chart),
            ("SHAP特征重要性图", self._draw_shap_chart),
            ("模型性能对比图", self._draw_ml_performance_chart),
            ("校准曲线图", self._draw_calibration_chart),
            ("个体SHAP解释下拉框", self._update_shap_individual_combo),
            ("风险归因条形图", self._draw_attribution_bar_chart),
            ("因果DAG可视化", self._draw_causal_dag),
            ("模型验证报告", self._update_validation_report)
        ]

        for chart_name, chart_func in charts:
            try:
                chart_func()
            except Exception as e:
                LOGGER.warning("%s更新失败: %s", chart_name, e, exc_info=True)

    # ================================================================
    # 模型验证报告（训练后验证阶段）
    # ================================================================

    def _get_validation_report(self):
        """生成（并缓存）模型验证报告。"""
        if not hasattr(self, '_validation_report_cache'):
            self._validation_report_cache = None
        if self._validation_report_cache is not None:
            return self._validation_report_cache
        if not (hasattr(self, 'ml_predictor') and self.ml_predictor
                and self.ml_predictor.is_trained):
            return None
        try:
            report = self.ml_predictor.build_validation_report(
                n_samples=2000, cv_folds=5, n_bins=10)
            self._validation_report_cache = report
            return report
        except Exception as e:
            LOGGER.warning("模型验证报告生成失败: %s", e, exc_info=True)
            return None

    def _update_validation_report(self):
        """渲染模型验证报告（文本 + 汇总图）。"""
        report = self._get_validation_report()
        if report is None:
            return
        self._display_validation_report_text(report)
        self._draw_validation_report_chart(report)

    def _display_validation_report_text(self, report):
        """将验证报告渲染到文本区。"""
        if not hasattr(self, 'ml_validation_text'):
            return
        self.ml_validation_text.delete('1.0', 'end')

        # 标签配色
        self.ml_validation_text.tag_configure('title',
                                              font=('Microsoft YaHei', 11, 'bold'))
        self.ml_validation_text.tag_configure('heading',
                                              font=('Microsoft YaHei', 10, 'bold'))
        self.ml_validation_text.tag_configure('warn',
                                              foreground='#CC6600')
        self.ml_validation_text.tag_configure('ok',
                                              foreground='#2E7D32')

        status = report.get('status', 'error')
        self.ml_validation_text.insert('end', "=== 模型验证报告 ===\n\n", 'title')

        if status != 'ok':
            self.ml_validation_text.insert(
                'end', f"验证未完成: {report.get('conclusion', {}).get('summary', '')}\n",
                'warn')
            return

        data = report.get('data', {})
        self.ml_validation_text.insert('end', "【数据划分】\n", 'heading')
        split_method = data.get('split_method', '')
        method_cn = '时间外验证' if split_method == 'temporal' else '随机三层划分'
        self.ml_validation_text.insert(
            'end',
            f"  划分方式: {method_cn}\n"
            f"  样本量: 训练={data.get('n_train', 0)}, "
            f"验证={data.get('n_val', 0)}, 测试={data.get('n_test', 0)} "
            f"(总={data.get('n_total', 0)})\n"
            f"  阳性率: {data.get('positive_rate', 0) * 100:.1f}%\n")

        models = report.get('models', {})
        self.ml_validation_text.insert('end', "\n【判别与校准指标】\n", 'heading')
        for key, entry in models.items():
            name = entry.get('name', key)
            test = entry.get('test', {})
            calib = entry.get('calibration', {})
            assess = entry.get('calibration_assessment', {})
            rec = entry.get('recalibration', {})
            cv = entry.get('cv', {}).get('auc', {})
            self.ml_validation_text.insert(
                'end',
                f"  {name}: 测试AUROC={test.get('roc_auc', 0):.3f}, "
                f"PR-AUC={test.get('pr_auc', 0):.3f}, "
                f"Brier={calib.get('brier', float('nan')):.3f}\n")
            hl = calib.get('hl', {})
            if not isinstance(hl.get('p_value'), float) or \
                    not np.isnan(hl.get('p_value', float('nan'))):
                self.ml_validation_text.insert(
                    'end',
                    f"    Hosmer-Lemeshow: χ²={hl.get('chi2', 0):.2f}, "
                    f"df={hl.get('df', 0)}, p={hl.get('p_value', float('nan')):.4f}\n")
            if rec.get('was_applied'):
                self.ml_validation_text.insert(
                    'end',
                    f"    [重校准] {rec.get('method', '')} 已应用，"
                    f"测试Brier {rec.get('test_before_brier', 0):.3f}→"
                    f"{rec.get('test_after_brier', 0):.3f}\n", 'warn')
            if cv:
                self.ml_validation_text.insert(
                    'end',
                    f"    CV AUROC(训练集): {cv.get('mean', 0):.3f}±"
                    f"{cv.get('std', 0):.3f}\n")

        ensemble = report.get('ensemble', {})
        if ensemble:
            et = ensemble.get('test', {})
            self.ml_validation_text.insert(
                'end',
                f"\n  集成(简单平均): 测试AUROC={et.get('roc_auc', 0):.3f}\n")

        conclusion = report.get('conclusion', {})
        self.ml_validation_text.insert('end', "\n【结论】\n", 'heading')
        self.ml_validation_text.insert('end', f"  {conclusion.get('summary', '')}\n")
        self.ml_validation_text.insert(
            'end', f"  总体评估: {conclusion.get('overall_assessment', '')}\n")

    def _draw_validation_report_chart(self, report):
        """绘制验证报告汇总图：各模型测试AUROC条形图 + 集成校准曲线。"""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'validation_figure'):
            return
        if report.get('status') != 'ok':
            return
        try:
            self.validation_figure.clear()

            models = report.get('models', {})
            names = [v.get('name', k) for k, v in models.items()]
            test_aucs = [v.get('test', {}).get('roc_auc', 0)
                         for v in models.values()]

            ax1 = self.validation_figure.add_subplot(121)
            ax1.barh(range(len(names)), test_aucs, color='#45B7D1', alpha=0.85)
            ax1.set_yticks(range(len(names)))
            ax1.set_yticklabels(names, fontsize=9)
            ax1.set_xlabel('测试集 AUROC')
            ax1.set_title('模型判别性能（验证阶段）')
            ax1.set_xlim(0, 1)

            # 集成分数的校准曲线
            ax2 = self.validation_figure.add_subplot(122)
            ens_calib = report.get('ensemble', {}).get('calibration', {}).get(
                'calibration_curve', {})
            mp = ens_calib.get('mean_predicted', [])
            fp = ens_calib.get('fraction_positives', [])
            if mp and fp:
                ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='理想校准')
                ax2.plot(mp, fp, 'o-', color='#4ECDC4',
                         label='集成校准曲线', alpha=0.85)
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.set_xlabel('预测概率')
                ax2.set_ylabel('观测阳性率')
                ax2.set_title('集成校准曲线（验证阶段）')
                ax2.legend(fontsize=8)
            else:
                ax2.text(0.5, 0.5, '校准数据不足', ha='center', va='center')
                ax2.set_title('校准曲线')

            self.validation_figure.tight_layout()
            self.validation_canvas.draw()
        except Exception as e:
            LOGGER.warning("验证报告图绘制失败: %s", e, exc_info=True)
