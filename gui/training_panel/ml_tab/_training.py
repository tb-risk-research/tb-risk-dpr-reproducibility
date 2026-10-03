#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 训练面板 - ML 训练控制（_TrainingMixin）

包含：ML模型训练、超参数调优、模型保存/加载、真实数据训练、对比实验
"""

from .._shared import *


class _TrainingMixin:
    """ML 训练控制相关方法"""

    # ==================== ML 训练核心 ====================

    def _ml_model_status_lines(self):
        """生成当前模型状态提示行（含迁移学习阶段），供警示标签/加载提示复用"""
        if not self.ml_predictor:
            return []
        lines = []
        stage = getattr(self.ml_predictor, 'transfer_stage', 'untrained')
        transfer_info = getattr(self.ml_predictor, 'transfer_info', {}) or {}

        if stage == 'finetuned':
            source = transfer_info.get('source') or '-'
            region_name = transfer_info.get('region_name') or '-'
            n_finetune = transfer_info.get('n_finetune_samples', 0)
            lines.append(
                f"迁移学习模型（预训练+本地微调）：预训练源 {source}；区域 {region_name}；"
                f"本地微调样本 {n_finetune} 例")
        elif stage == 'pretrained':
            source = transfer_info.get('source') or '-'
            region_name = transfer_info.get('region_name') or '-'
            incidence = transfer_info.get('incidence_per_100k', '-')
            lines.append(
                f"预训练模型（待本地微调）：预训练源 {source}；区域 {region_name}；"
                f"发病率 {incidence}/10万")

        if not getattr(self.ml_predictor, 'is_trained', False):
            lines.append("模型尚未训练！")
        elif self.ml_predictor.use_real_data:
            lines.append("当前使用真实数据训练模型，预测结果基于本地流行病学数据！")
        else:
            lines.append("当前使用合成数据训练模型，预测结果仅供参考，不能替代临床诊断！")
            lines.append("如需实际应用，请使用本地流行病学数据重新训练模型")
        return lines

    def _update_ml_warning_label(self):
        """更新ML警示标签以反映当前模型状态（含迁移学习阶段与版本）"""
        if SKLEARN_AVAILABLE and MATPLOTLIB_AVAILABLE and hasattr(self, 'ml_warning_label'):
            if self.ml_predictor:
                current_type = 'real' if self.ml_predictor.use_real_data else 'synthetic'

                if self._last_used_model_type is not None and current_type != self._last_used_model_type:
                    self._synthetic_warning_shown = False

                self._last_used_model_type = current_type

                lines = self._ml_model_status_lines()
                stage = getattr(self.ml_predictor, 'transfer_stage', 'untrained')
                if not lines:
                    return

                if stage in ('pretrained', 'finetuned'):
                    # 迁移学习模型：使用信息色（蓝青色）突出"预训练+微调"状态
                    self.ml_warning_label.config(
                        text='\n'.join(lines),
                        fg=self.COLORS['info'], bg=self.COLORS['info_bg']
                    )
                elif self.ml_predictor.use_real_data:
                    self.ml_warning_label.config(
                        text='\n'.join(lines),
                        fg=self.COLORS['success_fg'], bg=self.COLORS['success_bg']
                    )
                else:
                    self.ml_warning_label.config(
                        text='\n'.join(lines),
                        fg=self.COLORS['danger_fg'], bg=self.COLORS['danger_bg']
                    )

    def _ml_retrain(self):
        """重新训练ML模型（支持后台线程和进度提示）

        Section VII: 使用统一训练参数配置对话框替代 askyesno 弹窗。
        """
        if self._training.is_set():
            messagebox.showwarning("警告", "模型正在训练中，请稍后再试！")
            return

        if self.ml_predictor is None:
            messagebox.showwarning("警告", "ML模块不可用")
            return

        # Section VII: 弹出统一训练参数配置对话框
        from ..config import TrainingConfig, TrainingConfigDialog
        current_config = getattr(self, '_training_config', None) or TrainingConfig()
        dialog = TrainingConfigDialog(self.root, current_config,
                                       title='ML 训练参数配置')
        if dialog.result is None:
            return  # 用户取消
        config = dialog.result
        self._training_config = config  # 缓存供下次使用

        self._start_background_training(
            enable_hyperopt=config.ml_enable_hyperopt,
            cv_folds=config.ml_cv_folds,
            config=config,
        )

    def _start_background_training(self, enable_hyperopt=True, cv_folds=5,
                                     config=None):
        """在后台线程中执行模型训练，显示进度条

        Section VII: 接受 cv_folds 和 config 参数，记录训练日志。
        """
        self._training.set()
        import time as _time
        _train_start_time = _time.time()

        progress_win = tk.Toplevel(self.root)
        progress_win.title("模型训练中")
        progress_win.transient(self.root)
        progress_win.grab_set()
        progress_win.resizable(False, False)

        ttk.Label(progress_win, text="正在训练模型，请稍候...",
                  font=('Arial', 11)).pack(padx=20, pady=(15, 5))

        detail_label = ttk.Label(progress_win, text="初始化训练数据...",
                                 font=('Arial', 9))
        detail_label.pack(padx=20, pady=2)

        progress_bar = ttk.Progressbar(progress_win, mode='indeterminate', length=300)
        progress_bar.pack(padx=20, pady=10)
        progress_bar.start(15)

        def stop_training():
            if self.ml_predictor:
                self.ml_predictor.stop_training()
            stop_button.config(state='disabled', text="正在停止...")
            detail_label.config(text="正在停止训练，请稍候...")

        stop_button = ttk.Button(progress_win, text="停止", command=stop_training)
        stop_button.pack(pady=(5, 15))

        training_result = {'success': False, 'error': None, 'stopped': False}

        # Section VII: 从 config 获取 n_samples（向后兼容 None）
        n_samples = 2000
        if config is not None:
            n_samples = getattr(config, 'n_samples', 2000)

        def training_thread():
            try:
                detail_msg = "超参数调优中（可中途停止）..." if enable_hyperopt else "训练模型中..."
                self.root.after(0, lambda: detail_label.configure(text=detail_msg))
                data_source = getattr(self, '_validation_data_path', None)
                result = self.ml_predictor.train_models(
                    n_samples=n_samples, enable_hyperopt=enable_hyperopt,
                    data_source=data_source
                )
                training_result['success'] = result
                training_result['stopped'] = self.ml_predictor._stop_training if hasattr(self.ml_predictor, '_stop_training') else False
            except Exception as e:
                training_result['error'] = str(e)
            finally:
                self.root.after(0, lambda: self._finish_background_training(
                    progress_win, training_result, enable_hyperopt,
                    config=config, start_time=_train_start_time))

        self._start_training_thread(training_thread)

        progress_win.geometry("+%d+%d" % (self.root.winfo_rootx() + 150,
                                           self.root.winfo_rooty() + 200))
        progress_win.protocol("WM_DELETE_WINDOW", lambda: None)

    def _extract_ml_training_metrics(self):
        """从 ml_predictor.model_performance 提取最佳模型的验证指标

        Returns:
            dict: {AUROC, AUPRC, Brier, best_model, ...}；无有效指标时返回 {}
        """
        if not self.ml_predictor:
            return {}
        perf = getattr(self.ml_predictor, 'model_performance', {}) or {}
        candidates = []
        for key, info in perf.items():
            if not isinstance(info, dict):
                continue
            auroc = info.get('AUROC')
            if isinstance(auroc, (int, float)):
                candidates.append((float(auroc), key, info))
        if not candidates:
            return {}
        _, best_key, best_info = max(candidates, key=lambda x: x[0])
        metrics = {}
        for metric_name in ('AUROC', 'AUPRC', 'Brier', 'Accuracy', 'F1'):
            value = best_info.get(metric_name)
            if isinstance(value, (int, float)):
                metrics[metric_name] = float(value)
        metrics['best_model'] = best_info.get('name', best_key)
        return metrics

    def _training_dataset_info(self):
        """构建训练数据集版本信息（用于训练档案归因）"""
        source = getattr(self, '_validation_data_path', None)
        n_samples = 0
        if self.ml_predictor:
            n_samples = getattr(self.ml_predictor, 'training_sample_count', 0) or 0
        if source:
            import os as _os
            return {
                'source': _os.path.basename(str(source)),
                'source_path': str(source),
                'n_samples': int(n_samples),
                'real_data': True,
            }
        return {
            'source': 'synthetic',
            'n_samples': int(n_samples),
            'real_data': False,
        }

    def _finish_background_training(self, progress_win, training_result,
                                      enable_hyperopt, config=None,
                                      start_time=None):
        """后台训练完成后的回调处理

        Section VII: 添加训练日志持久化（含指标、数据集版本与基准对比）。
        """
        self._training.clear()

        try:
            progress_win.destroy()
        except (tk.TclError, RuntimeError) as e:
            LOGGER.warning("销毁训练进度窗口失败: %s", e, exc_info=True)

        # Section VII: 训练日志持久化（成功/失败/取消均写入，避免断档）
        _duration = 0.0
        if start_time is not None:
            import time as _time
            _duration = _time.time() - start_time
        try:
            from ..training_log import TrainingLogger
            logger = TrainingLogger()
            params_dict = config.to_dict() if config is not None else {}
            dataset_info = self._training_dataset_info()
            if training_result.get('error'):
                logger.log(
                    model_type='ml', params=params_dict,
                    training_duration=_duration, status='failed',
                    error_message=str(training_result['error']),
                    dataset=dataset_info,
                )
            elif training_result.get('stopped'):
                logger.log(
                    model_type='ml', params=params_dict,
                    training_duration=_duration, status='cancelled',
                    dataset=dataset_info,
                )
            else:
                # 刷新历史最佳时自动保存模型权重快照
                model_saver = None
                if self.ml_predictor is not None:
                    model_saver = lambda path: bool(
                        self.ml_predictor.save_model(path))
                logger.log(
                    model_type='ml', params=params_dict,
                    metrics=self._extract_ml_training_metrics(),
                    training_duration=_duration, status='success',
                    dataset=dataset_info,
                    model_saver=model_saver,
                )
        except Exception:
            pass  # 日志失败不影响主流程

        if training_result.get('stopped'):
            messagebox.showinfo("提示", "训练已被用户手动停止。")
            return

        if training_result['error']:
            messagebox.showerror("错误", f"训练失败: {training_result['error']}")
            return

        if training_result['success']:
            msg = "模型重新训练完成"
            if enable_hyperopt:
                msg += "（已启用超参数调优）"
            messagebox.showinfo("成功", msg)
            self._update_ml_warning_label()
            if self.results:
                self._run_ml_predictions()
                self._update_ml_charts()
                self._display_ml_results()
        else:
            messagebox.showerror("错误", "模型训练失败")

    def _ml_save_model(self):
        """保存ML模型到文件"""
        if self.ml_predictor is None or not self.ml_predictor.is_trained:
            messagebox.showwarning("警告", "模型未训练，无法保存")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".joblib",
            filetypes=[("Joblib文件", "*.joblib"), ("Pickle文件", "*.pkl"), ("所有文件", "*.*")],
            title="保存ML模型",
            initialfile="tb_ml_model.joblib"
        )

        if file_path:
            if self.ml_predictor.save_model(file_path):
                messagebox.showinfo("成功", f"模型已保存到：{file_path}")
            else:
                messagebox.showerror("错误", "模型保存失败")

    def _ml_load_model(self):
        """从文件加载ML模型"""
        if self.ml_predictor is None:
            messagebox.showwarning("警告", "ML模块不可用")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("Joblib文件", "*.joblib"), ("Pickle文件", "*.pkl"), ("所有文件", "*.*")],
            title="加载ML模型"
        )

        if file_path:
            if self.ml_predictor.load_model(file_path):
                # 展示迁移阶段/预训练源/微调样本等最新版本信息
                try:
                    status = self.ml_predictor.summarize_transfer_status()
                    transfer_line = f"\n{status['message']}" if status.get('message') else ''
                except Exception:
                    transfer_line = ''
                messagebox.showinfo(
                    "成功",
                    f"模型已加载：{file_path}\n"
                    f"训练样本量: {self.ml_predictor.training_sample_count}"
                    f"{transfer_line}")
                self._update_ml_warning_label()
                if self.results:
                    self._run_ml_predictions()
                    self._update_ml_charts()
                    self._display_ml_results()
            else:
                messagebox.showerror("错误", "模型加载失败")

    # ==================== 从 UI 数据训练 ====================

    def _auto_train_ml_from_ui(self):
        """从当前UI数据自动训练ML模型"""
        if self.ml_predictor is None:
            messagebox.showwarning("警告", "ML模块不可用")
            return

        X, y = self.build_ml_dataset_from_ui()

        if X is None or y is None:
            messagebox.showwarning("警告", "无法构建数据集，请添加接触者数据")
            return

        import numpy as np
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            messagebox.showwarning("警告", "标签只有一种类型，请标记一些接触者为'是'，一些为'否'")
            return

        enable_hyperopt = messagebox.askyesno("超参数调优",
            "是否启用超参数调优？\n\n"
            "启用：更精确，但训练较慢\n"
            "不启用：使用默认参数，训练较快")

        self._start_ml_training_thread(X, y, enable_hyperopt)

    def _start_ml_training_thread(self, X, y, enable_hyperopt):
        """在后台线程中训练ML模型"""
        if self._training.is_set():
            messagebox.showwarning("警告", "训练正在进行中，请稍后再试")
            return

        self._training.set()

        progress_window, progress_var, update_progress, close_progress = self._show_progress_dialog("正在训练ML模型...")
        update_progress(10, "正在初始化...")

        def train_task():
            try:
                update_progress(20, "正在训练随机森林...")

                result = self.ml_predictor.train_from_arrays(
                    X, y,
                    random_state=42,
                    enable_hyperopt=enable_hyperopt
                )

                if result:
                    update_progress(90, "训练完成，正在更新界面...")

                    def update_ui():
                        try:
                            messagebox.showinfo("成功",
                                f"ML模型训练完成！\n\n"
                                f"训练样本：{self.ml_predictor.training_sample_count}\n"
                                f"模型状态：已使用真实数据训练")

                            self._update_ml_warning_label()
                            if self.results:
                                self._run_ml_predictions()
                                self._update_ml_charts()
                                self._display_ml_results()
                        finally:
                            self._training.clear()
                            try:
                                close_progress()
                            except (tk.TclError, RuntimeError):
                                pass

                    self.root.after(0, update_ui)
                else:
                    def show_error():
                        try:
                            messagebox.showerror("错误", "ML模型训练失败")
                        finally:
                            self._training.clear()
                            try:
                                close_progress()
                            except (tk.TclError, RuntimeError):
                                pass
                    self.root.after(0, show_error)

            except Exception as e:
                error_msg = str(e)
                import traceback
                tb_str = traceback.format_exc()

                def show_exception():
                    try:
                        messagebox.showerror("错误", f"训练过程中发生错误：{error_msg}")
                        LOGGER.error("训练异常堆栈:\n%s", tb_str)
                    finally:
                        self._training.clear()
                        try:
                            close_progress()
                        except (tk.TclError, RuntimeError):
                            pass
                self.root.after(0, show_exception)

        self._start_training_thread(train_task)

    def _ml_train_from_real_data(self):
        """从真实数据CSV文件训练ML模型"""
        if self.ml_predictor is None:
            messagebox.showwarning("警告", "ML模块不可用")
            return

        if not PANDAS_AVAILABLE:
            messagebox.showwarning("依赖缺失",
                "pandas未安装，无法从CSV文件读取数据。\n请安装pandas: pip install pandas")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
            title="选择真实数据CSV文件"
        )

        if not file_path:
            return

        info_message = (
            "CSV文件格式要求：\n\n"
            "1. 必须包含以下特征列：\n"
            "   age, cumulative_exposure, has_symptoms, bcg_vaccine,\n"
            "   has_tb, contact_distance_score, ventilation_score,\n"
            "   is_high_risk, past_illness, exposure_setting_score,\n"
            "   single_duration, freq_density, time_span\n\n"
            "2. 必须包含目标列：tb_outcome（0=未发病，1=发病）\n\n"
            "3. 数值型特征应为整数或浮点数\n\n"
            "是否继续？"
        )

        if not messagebox.askyesno("真实数据训练", info_message):
            return

        try:
            result = self.ml_predictor.train_from_real_data(file_path)
            if result.get('success'):
                perf = result.get('model_performance', {})
                perf_str = ''
                if perf:
                    perf_str = '\n'.join(
                        f"  {v.get('name', k)}: AUROC={v.get('AUROC', 0):.4f}"
                        for k, v in perf.items()
                    )
                messagebox.showinfo("成功",
                    f"从真实数据训练完成\n"
                    f"样本量: {result.get('n_samples', 0)}\n"
                    f"确诊: {result.get('n_positive', 0)} ({result.get('positive_rate', 0)*100:.1f}%)\n"
                    f"模型性能:\n{perf_str}")
                self._update_ml_warning_label()
                self._validation_data_path = file_path
                if self.results:
                    self._run_ml_predictions()
                    self._update_ml_charts()
                    self._display_ml_results()
            else:
                error_type = result.get('error_type', 'unknown')
                diagnostics = result.get('diagnostics', '未知错误')
                suggestions = '\n'.join(f"  • {s}" for s in result.get('suggestions', []))
                msg = f"训练失败 [{error_type}]\n\n{diagnostics}"
                if suggestions:
                    msg += f"\n\n建议:\n{suggestions}"
                messagebox.showerror("错误", msg)
        except Exception as e:
            messagebox.showerror("错误", f"训练失败: {str(e)}")

    def _ml_compare_real_vs_synthetic(self):
        """真实数据 vs 合成数据性能对比实验"""
        if self.ml_predictor is None:
            messagebox.showwarning("警告", "ML模块不可用")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
            title="选择真实数据CSV文件进行对比实验"
        )

        if not file_path:
            return

        info_message = (
            "真实 vs 合成数据对比实验\n\n"
            "将执行以下操作：\n"
            "1. 加载真实数据并划分训练/测试集\n"
            "2. 分别训练三种模式：\n"
            "   • 纯合成数据\n"
            "   • 纯真实数据\n"
            "   • 真实+合成增强\n"
            "3. Bootstrap BCa 95% 置信区间\n"
            "4. Wilcoxon 符号秩检验（显著性判断）\n"
            "5. 基于统计证据生成建议\n\n"
            "此过程可能需要数分钟，是否继续？"
        )

        if not messagebox.askyesno("对比实验确认", info_message):
            return

        progress_win = tk.Toplevel(self.root)
        progress_win.title("对比实验进行中")
        progress_win.geometry("400x120")
        progress_win.transient(self.root)
        progress_win.grab_set()

        progress_label = ttk.Label(
            progress_win, text="正在运行对比实验...",
            font=('Arial', 11)
        )
        progress_label.pack(pady=15)

        detail_label = ttk.Label(
            progress_win, text="加载数据中...",
            font=('Arial', 9)
        )
        detail_label.pack(pady=5)

        progress_bar = ttk.Progressbar(
            progress_win, mode='indeterminate', length=300
        )
        progress_bar.pack(pady=10)
        progress_bar.start()

        comparison_result = {'status': 'error', 'error': None}

        def run_comparison():
            try:
                self.root.after(0, lambda: detail_label.configure(
                    text="训练三种模式中..."
                ))
                result = self.ml_predictor.compare_real_vs_synthetic(
                    real_data_path=file_path
                )
                comparison_result.update(result)
            except Exception as e:
                comparison_result['error'] = str(e)
            finally:
                self.root.after(0, lambda: self._finish_comparison_display(
                    comparison_result, progress_win, file_path
                ))

        import threading
        thread = threading.Thread(target=run_comparison, daemon=True)
        thread.start()

    def _finish_comparison_display(self, result, progress_win, file_path):
        """完成对比实验后展示结果"""
        progress_win.destroy()

        if result.get('error'):
            messagebox.showerror("错误", f"对比实验失败: {result['error']}")
            return

        if result.get('status') == 'skipped':
            messagebox.showwarning("跳过", result.get('recommendation', '对比实验未执行'))
            return

        if result.get('status') == 'error':
            messagebox.showerror("错误", result.get('recommendation', '对比实验失败'))
            return

        modes = result.get('modes', {})
        wilcoxon = result.get('wilcoxon', {})
        bootstrap_ci = result.get('bootstrap_ci', {})
        recommendation = result.get('recommendation', '')

        lines = []
        lines.append("=" * 65)
        lines.append("  真实数据 vs 合成数据 性能对比实验报告")
        lines.append("=" * 65)
        lines.append(f"时间: {result.get('timestamp', '')}")
        lines.append(f"测试集样本量: {result.get('test_size', 0)}")
        lines.append("")

        lines.append("-" * 65)
        lines.append("  一、三种训练模式性能对比")
        lines.append("-" * 65)
        mode_order = ['synthetic', 'real', 'mixed']
        headers = ['模式', 'AUROC', 'AUPRC', 'Brier', '训练样本']
        lines.append(f"  {headers[0]:<12} {headers[1]:>8} {headers[2]:>8} {headers[3]:>8} {headers[4]:>8}")
        lines.append("  " + "-" * 50)
        for mode_key in mode_order:
            m = modes.get(mode_key, {})
            label = m.get('label', mode_key)
            lines.append(
                f"  {label:<12} {m.get('auroc', 0):>8.4f} {m.get('auprc', 0):>8.4f} "
                f"{m.get('brier', 0):>8.4f} {m.get('train_size', 0):>8}"
            )
        lines.append("")

        lines.append("-" * 65)
        lines.append("  二、Bootstrap 95% 置信区间（AUROC）")
        lines.append("-" * 65)
        for mode_key in mode_order:
            if mode_key not in bootstrap_ci:
                continue
            ci = bootstrap_ci[mode_key].get('auroc', {})
            label = modes.get(mode_key, {}).get('label', mode_key)
            lines.append(
                f"  {label:<12}: {ci.get('mean', 0):.4f} "
                f"[{ci.get('ci_95_lower', 0):.4f} - {ci.get('ci_95_upper', 0):.4f}]"
            )
        lines.append("")

        lines.append("-" * 65)
        lines.append("  三、Wilcoxon 符号秩检验（成对比较）")
        lines.append("-" * 65)
        for pair_key, pair_info in wilcoxon.items():
            if 'error' in pair_info:
                lines.append(f"  {pair_info.get('label', pair_key)}: 检验失败")
                continue
            label = pair_info.get('label', pair_key)
            p_val = pair_info.get('p_value', 1.0)
            sig = "显著 (p<0.05)" if pair_info.get('significant_at_0_05') else "不显著 (p≥0.05)"
            lines.append(f"  {label}:")
            lines.append(f"    统计量 W={pair_info.get('statistic', 0):.1f}, p={p_val:.4f} → {sig}")
        lines.append("")

        lines.append("-" * 65)
        lines.append("  四、统计建议")
        lines.append("-" * 65)
        lines.append(f"  {recommendation}")
        lines.append("")
        lines.append("=" * 65)

        report_text = '\n'.join(lines)

        if hasattr(self, 'ml_result_text'):
            self.ml_result_text.delete('1.0', 'end')
            self.ml_result_text.insert('end', report_text)

        result_dialog = tk.Toplevel(self.root)
        result_dialog.title("对比实验结果")
        result_dialog.geometry("700x550")
        result_dialog.transient(self.root)

        text_frame = ttk.Frame(result_dialog)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)

        text_widget = self._register_text_widget(tk.Text(
            text_frame, wrap='word', font=('Consolas', 10),
            width=80, height=25
        ))
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        text_widget.insert('1.0', report_text)
        text_widget.configure(state='disabled')

        btn_frame = ttk.Frame(result_dialog)
        btn_frame.pack(fill='x', padx=10, pady=(0, 10))

        def export_comparison():
            export_path = filedialog.asksaveasfilename(
                defaultextension=".html",
                filetypes=[("HTML文件", "*.html"), ("所有文件", "*.*")],
                title="导出对比报告"
            )
            if export_path:
                try:
                    from ...export_utils import export_comparison_report
                    success = export_comparison_report(result, export_path)
                    if success:
                        messagebox.showinfo("成功", f"对比报告已导出: {export_path}")
                    else:
                        messagebox.showerror("错误", "报告导出失败")
                except ImportError as e:
                    messagebox.showerror("错误", f"导出模块不可用: {e}")

        ttk.Button(btn_frame, text="导出HTML报告", command=export_comparison).pack(side='right', padx=5)
        ttk.Button(btn_frame, text="关闭", command=result_dialog.destroy).pack(side='right', padx=5)

        self._validation_data_path = file_path

    def build_ml_dataset_from_ui(self):
        """从当前UI数据构建ML训练数据集

        Returns:
            tuple: (X, y) X为特征矩阵，y为标签数组；失败返回(None, None)
        """
        if not self.ml_predictor:
            return None, None

        patient_ftd = int(self.basic_info_vars['delay_days'].get()) if 'delay_days' in self.basic_info_vars else 0
        patient_cough_freq = int(self.basic_info_vars.get('cough_freq', tk.StringVar(value='0')).get())
        contact_count = len(self.family_entries) + len(self.social_entries)

        self.ml_predictor.set_patient_context(patient_ftd, patient_cough_freq, contact_count)

        features = []
        labels = []

        for entry in self.family_entries:
            try:
                contact_id = entry.get('_id')
                if contact_id:
                    label = self.contact_labels.get(contact_id, 0)
                    if 'cumulative_exposure' not in entry:
                        entry['cumulative_exposure'] = calculate_cumulative_exposure(
                            entry.get('single_duration', 30),
                            entry.get('freq_density', self.DEFAULT_FAMILY_FREQ),
                            entry.get('time_span', 4)
                        )
                    feature_vec = self.ml_predictor.extract_features_with_interactions(
                        entry, 'family', patient_ftd, patient_cough_freq, contact_count
                    )
                    if feature_vec is not None:
                        features.append(feature_vec.flatten())
                        labels.append(label)
            except (ValueError, RuntimeError, TypeError, KeyError) as e:
                LOGGER.warning(
                    "准备家庭成员 ML 训练数据失败，entry=%s: %s", entry, e,
                    exc_info=True)
                continue

        for entry in self.social_entries:
            try:
                contact_id = entry.get('_id')
                if contact_id:
                    label = self.contact_labels.get(contact_id, 0)
                    if 'cumulative_exposure' not in entry:
                        entry['cumulative_exposure'] = calculate_cumulative_exposure(
                            entry.get('single_duration', 30),
                            entry.get('freq_density', self.DEFAULT_SOCIAL_FREQ),
                            entry.get('time_span', 4)
                        )
                    feature_vec = self.ml_predictor.extract_features_with_interactions(
                        entry, 'social', patient_ftd, patient_cough_freq, contact_count
                    )
                    if feature_vec is not None:
                        features.append(feature_vec.flatten())
                        labels.append(label)
            except (ValueError, RuntimeError, TypeError, KeyError) as e:
                LOGGER.warning(
                    "准备社会接触者 ML 训练数据失败，entry=%s: %s", entry, e,
                    exc_info=True)
                continue

        if not features:
            return None, None

        import numpy as np
        X = np.array(features)
        y = np.array(labels)

        return X, y
