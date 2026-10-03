#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 结果面板 - 导出 Mixin（CSV / TXT / 图表 / Excel / JSON / ML / PDF）"""

from ._shared import *


class ExportMixin:
    """结果导出方法（CSV 保存 / TXT 导出 / CSV 导出 / 图表导出 / 各类报告）"""

    def _save_to_csv(self):
        """保存数据到 CSV 文件（标准格式，与import_csv兼容）"""
        if self.assessing.is_set():
            messagebox.showwarning("警告", "正在评估中，请稍后再保存")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="保存数据到 CSV"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)

                # 中文到英文的映射字典
                distance_map_save = {
                    "极近": "very_close",
                    "近": "close",
                    "中等": "medium",
                    "远": "far",
                    "极远": "distant"
                }
                setting_map_save = {
                    "拥挤": "crowded",
                    "密闭": "closed",
                    "一般": "general",
                    "户外": "outdoor"
                }
                # 直接使用ILLNESS_TYPE_MAPPING将中文转换为英文

                # 写入患者基本信息
                writer.writerow(['=== 患者基本信息 ==='])
                patient_headers = []
                patient_values = []
                for key, var in self.basic_info_vars.items():
                    patient_headers.append(key)
                    if isinstance(var, (tk.IntVar, tk.StringVar)):
                        value = var.get()
                        if isinstance(value, tk.Variable):
                            value = value.get()
                        patient_values.append(str(value))
                    elif isinstance(var, tk.Text):
                        value = var.get('1.0', tk.END).strip()
                        # 移除会影响CSV格式的字符
                        value = value.replace('\n', ' ').replace('\r', ' ')
                        patient_values.append(str(value))
                writer.writerow(patient_headers)
                writer.writerow(patient_values)

                writer.writerow([])

                # 写入家庭成员（与export_csv_template列顺序一致）
                writer.writerow(['=== 家庭成员信息 ==='])
                if self.family_entries:
                    family_headers = list(self.FAMILY_CSV_COLUMNS)
                    writer.writerow(family_headers)
                    for entry in self.family_entries:
                        contact_distance = entry.get('contact_distance', '近')
                        contact_distance_en = distance_map_save.get(contact_distance, 'close')
                        exposure_setting = entry.get('exposure_setting', '一般')
                        exposure_setting_en = setting_map_save.get(exposure_setting, 'general')
                        illness_type = entry.get('past_illness_type', '') if _is_yes(entry.get('past_illness', '否')) else ""
                        illness_type_en = self.ILLNESS_TYPE_MAPPING.get(illness_type, illness_type)
                        row = [
                            entry.get('name', ''),
                            entry.get('age', 30),
                            entry.get('relationship', '其他'),
                            entry.get('single_duration', 30),
                            entry.get('freq_density', 14),
                            entry.get('time_span', 4),
                            self._to_bool(entry.get('has_tb', '否')),
                            self._to_bool(entry.get('has_symptoms', '否')),
                            self._to_bool(entry.get('bcg_vaccine', '是')),
                            self._to_bool(entry.get('past_illness', '否')),
                            illness_type_en,
                            contact_distance_en,
                            entry.get('ventilation', '3'),
                            exposure_setting_en
                        ]
                        writer.writerow(row)

                writer.writerow([])

                # 写入社会接触者（与export_csv_template列顺序一致）
                writer.writerow(['=== 社会接触者信息 ==='])
                if self.social_entries:
                    social_headers = list(self.SOCIAL_CSV_COLUMNS)
                    writer.writerow(social_headers)
                    for entry in self.social_entries:
                        contact_distance = entry.get('contact_distance', '中等')
                        contact_distance_en = distance_map_save.get(contact_distance, 'medium')
                        exposure_setting = entry.get('exposure_setting', '一般')
                        exposure_setting_en = setting_map_save.get(exposure_setting, 'general')
                        illness_type = entry.get('past_illness_type', '') if _is_yes(entry.get('past_illness', '否')) else ""
                        illness_type_en = self.ILLNESS_TYPE_MAPPING.get(illness_type, illness_type)
                        has_tb_value = self._to_bool(entry.get('has_tb', '否'))

                        row = [
                            entry.get('name', ''),
                            entry.get('age', 30),
                            entry.get('single_duration', 30),
                            entry.get('freq_density', 2),
                            entry.get('time_span', 4),
                            self._to_bool(entry.get('is_high_risk', '否')),
                            self._to_bool(entry.get('has_symptoms', '否')),
                            self._to_bool(entry.get('bcg_vaccine', '是')),
                            has_tb_value,
                            entry.get('ventilation', '3'),
                            contact_distance_en,
                            exposure_setting_en,
                            self._to_bool(entry.get('past_illness', '否')),
                            illness_type_en
                        ]
                        writer.writerow(row)

            messagebox.showinfo("成功", f"数据已保存到：{file_path}")

        except Exception as e:
            messagebox.showerror("错误", f"保存数据时发生错误：{str(e)}")

    def _export_results(self):
        """导出评估结果到文件"""
        if self.assessing.is_set():
            messagebox.showwarning("警告", "正在评估中，请稍后再导出")
            return

        # 使用线程锁复制数据，避免读到不完整数据
        with self._data_lock:
            if not self.results:
                messagebox.showwarning("警告", "请先执行风险评估")
                return
            # 数据安全：统一强化脱敏后再导出
            results_copy = self._mask_for_export(self.results)

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")],
            title="导出评估结果",
            initialfile="结核病风险评估结果.txt"
        )

        if not file_path:
            return

        try:
            if file_path.endswith('.csv'):
                self._export_results_csv(file_path, results_copy)
            else:
                self._export_results_text(file_path, results_copy)
            self._log_security_operation("EXPORT", details={"format": "txt/csv",
                                                             "filepath": file_path})
            messagebox.showinfo("成功", f"评估结果已导出到：{file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出结果时发生错误：{str(e)}")

    def _export_results_text(self, filename, results):
        """导出评估结果为文本文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("结核病风险评估结果报告\n")
            f.write("=" * 60 + "\n\n")

            # 整体风险
            f.write("一、整体风险评估\n")
            f.write("-" * 40 + "\n")
            f.write(f"风险等级：{results.get('overall_risk', 'N/A')}\n")
            f.write(f"总评分：{results.get('total_score', 0):.1f}\n")
            f.write(f"感染概率：{results.get('base_infection_probability', 0):.1f}%\n")
            f.write(f"建议：{results.get('overall_suggestion', '')}\n\n")

            # 各维度风险
            f.write("二、各维度风险评估\n")
            f.write("-" * 40 + "\n")
            for dim, data in results.get('individual_risks', {}).items():
                f.write(f"{dim}：{data.get('risk', 'N/A')}\n")
                f.write(f"  建议：{data.get('suggestion', '')}\n")
            f.write("\n")

            # 潜在患者
            if 'potential_patients' in results:
                f.write("三、潜在患者识别\n")
                f.write("-" * 40 + "\n")

                family_patients = results['potential_patients'].get('family', [])
                if family_patients:
                    f.write("\n家庭成员潜在患者：\n")
                    for patient in family_patients:
                        name = patient.get('name', '未知')
                        prob = patient.get('disease_probability', 0)
                        priority = patient.get('priority', 'N/A')
                        f.write(f"  - {name}：发病概率 {prob:.1f}%，优先级：{priority}\n")

                social_patients = results['potential_patients'].get('social', [])
                if social_patients:
                    f.write("\n社会接触者潜在患者：\n")
                    for patient in social_patients:
                        name = patient.get('name', '未知')
                        prob = patient.get('disease_probability', 0)
                        priority = patient.get('priority', 'N/A')
                        f.write(f"  - {name}：发病概率 {prob:.1f}%，优先级：{priority}\n")

            # 机器学习评估结果
            # 原子性读取：在锁保护下获取快照
            if hasattr(self, 'ml_results_lock'):
                with self.ml_results_lock:
                    ml_results = self.ml_results
            else:
                ml_results = self.ml_results

            if ml_results:
                f.write("\n四、机器学习风险评估\n")
                f.write("-" * 40 + "\n")

                perf = ml_results.get('model_performance', {})
                if perf:
                    f.write("\n模型性能（5折交叉验证）:\n")
                    f.write(f"  训练样本量: {ml_results.get('training_sample_count', 0)}\n")
                    for model_name, metrics in perf.items():
                        f.write(f"  {metrics.get('name', model_name)}: AUROC={metrics.get('AUROC', 0):.3f}, AUPRC={metrics.get('AUPRC', 0):.3f}\n")

                if self.ml_predictor and self.ml_predictor.is_trained:
                    try:
                        calib_results = self.ml_predictor.compute_calibration()
                        if calib_results:
                            f.write("\n模型校准（Brier评分）:\n")
                            for model_name, calib in calib_results.items():
                                f.write(f"  {calib.get('name', model_name)}: Brier={calib.get('brier_score', 0):.4f}\n")
                    except Exception:
                        import traceback as _tb
                        _tb.print_exc()

                if ml_results.get('family'):
                    f.write("\n家庭接触者ML预测:\n")
                    for item in ml_results['family']:
                        name = item.get('name', '未知')
                        trad_prob = item.get('traditional_prob', 0)
                        ml_preds = item.get('ml_predictions', {})
                        f.write(f"  {name}: 传统={trad_prob:.1f}%")
                        ensemble = ml_preds.get('ensemble', {})
                        if ensemble:
                            f.write(f", ML集成={ensemble.get('risk_probability', 0):.1f}%")
                        f.write("\n")

                if ml_results.get('social'):
                    f.write("\n社会接触者ML预测:\n")
                    for item in ml_results['social']:
                        name = item.get('name', '未知')
                        trad_prob = item.get('traditional_prob', 0)
                        ml_preds = item.get('ml_predictions', {})
                        f.write(f"  {name}: 传统={trad_prob:.1f}%")
                        ensemble = ml_preds.get('ensemble', {})
                        if ensemble:
                            f.write(f", ML集成={ensemble.get('risk_probability', 0):.1f}%")
                        f.write("\n")

                shap_info = ml_results.get('shap_analysis', {})
                if shap_info and shap_info.get('feature_importance'):
                    f.write("\nSHAP特征重要性排名（Top 5）:\n")
                    for fname, importance in shap_info.get('feature_importance', [])[:5]:
                        desc = shap_info.get('feature_descriptions', {}).get(fname, fname)
                        f.write(f"  {desc}: {importance:.4f}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("报告生成时间：" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

    def _export_results_csv(self, filename, results):
        """导出评估结果为CSV文件"""
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)

            # 整体风险
            writer.writerow(['整体风险评估'])
            writer.writerow(['风险等级', results.get('overall_risk', 'N/A')])
            writer.writerow(['总评分', results.get('total_score', 0)])
            writer.writerow(['感染概率', f"{results.get('base_infection_probability', 0):.1f}%"])
            writer.writerow(['建议', results.get('overall_suggestion', '')])
            writer.writerow([])

            # 各维度风险
            writer.writerow(['各维度风险评估'])
            writer.writerow(['维度', '风险等级', '建议'])
            for dim, data in results.get('individual_risks', {}).items():
                writer.writerow([dim, data.get('risk', 'N/A'), data.get('suggestion', '')])
            writer.writerow([])

            # 潜在患者
            if 'potential_patients' in results:
                family_patients = results['potential_patients'].get('family', [])
                if family_patients:
                    writer.writerow(['家庭成员潜在患者'])
                    writer.writerow(['姓名', '发病概率', '优先级'])
                    for patient in family_patients:
                        writer.writerow([patient.get('name', '未知'), f"{patient.get('disease_probability', 0):.1f}%", patient.get('priority', 'N/A')])
                    writer.writerow([])

                social_patients = results['potential_patients'].get('social', [])
                if social_patients:
                    writer.writerow(['社会接触者潜在患者'])
                    writer.writerow(['姓名', '发病概率', '优先级'])
                    for patient in social_patients:
                        writer.writerow([patient.get('name', '未知'), f"{patient.get('disease_probability', 0):.1f}%", patient.get('priority', 'N/A')])

    def _export_all_charts(self):
        """导出所有图表为图片文件（Section IX: 委托 ExportManager 统一处理）

        保持原签名（无参数）以向后兼容。内部委托给 ExportManager.export_all_charts，
        新增格式选择对话框（PNG/SVG/PDF），并将校准曲线(ml_calib_figure)和
        个体SHAP图(shap_ind_figure)纳入批量导出列表。
        """
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showwarning("警告", "matplotlib未安装，无法导出图表")
            return

        # 使用线程锁检查数据是否就绪（本函数仅保存图表对象，不读取结果数据）
        with self._data_lock:
            if not self.results:
                messagebox.showwarning("警告", "请先执行风险评估")
                return

        # Section IX: 检查 results_stale 标记
        if getattr(self, 'results_stale', False):
            if not messagebox.askyesno(
                "结果已过期",
                "数据已修改，当前结果可能不准确。\n是否仍要继续导出？"
            ):
                return

        # Section IX: 格式选择对话框（PNG/SVG/PDF）
        format_choice = self._ask_chart_export_format()
        if format_choice is None:
            return  # 用户取消

        try:
            folder_path = filedialog.askdirectory(title="选择图表保存目录")
            if not folder_path:
                return

            # Section IX: 委托 ExportManager 统一导出（含校准曲线和个体SHAP图）
            export_manager = self._get_export_manager()
            exported = export_manager.export_all_charts(self, folder_path, format=format_choice)

            if exported:
                messagebox.showinfo(
                    "成功",
                    f"已导出 {len(exported)} 张图表（{format_choice.upper()}）到：\n{folder_path}"
                )
            else:
                messagebox.showwarning(
                    "提示",
                    f"未导出任何图表，请确认已生成图表后重试。\n目标目录：{folder_path}"
                )
        except Exception as e:
            messagebox.showerror("错误", f"导出图表时发生错误：{str(e)}")

    def _ask_chart_export_format(self):
        """弹出格式选择对话框（PNG/SVG/PDF）

        Returns:
            str | None: 'png' / 'svg' / 'pdf'；用户取消时返回 None
        """
        try:
            choice_win = tk.Toplevel(self.root)
            choice_win.title("选择导出格式")
            choice_win.geometry("320x180")
            choice_win.transient(self.root)
            choice_win.grab_set()

            # 优先级十一：允许放大但不建议缩小（高 DPI 友好）
            choice_win.resizable(True, True)
            choice_win.minsize(320, 180)

            ttk.Label(choice_win, text="选择图表导出格式：",
                      font=('Microsoft YaHei', 10, 'bold')).pack(padx=10, pady=(10, 5), anchor='w')
            ttk.Label(choice_win, text="PNG：日常归档（位图）\n"
                                       "SVG：学术发表/再编辑（矢量）\n"
                                       "PDF：学术发表/打印（矢量）",
                      font=('Microsoft YaHei', 9)).pack(padx=10, pady=5, anchor='w')

            chosen = {'value': None}

            def _pick(fmt):
                chosen['value'] = fmt
                choice_win.destroy()

            btn_frame = ttk.Frame(choice_win)
            btn_frame.pack(pady=10)
            ttk.Button(btn_frame, text="PNG", command=lambda: _pick('png')).pack(side='left', padx=5)
            ttk.Button(btn_frame, text="SVG", command=lambda: _pick('svg')).pack(side='left', padx=5)
            ttk.Button(btn_frame, text="PDF", command=lambda: _pick('pdf')).pack(side='left', padx=5)
            ttk.Button(btn_frame, text="取消",
                       command=choice_win.destroy).pack(side='left', padx=5)

            # 模态等待
            self.root.wait_window(choice_win)
            return chosen['value']
        except Exception:
            LOGGER.error("格式选择对话框失败，默认使用 PNG", exc_info=True)
            return 'png'

    def _get_export_manager(self):
        """获取 ExportManager 实例（懒初始化，缓存复用）

        Returns:
            ExportManager: 统一导出管理器实例
        """
        try:
            if not hasattr(self, '_export_manager') or self._export_manager is None:
                from .export_manager import ExportManager
                self._export_manager = ExportManager(self)
            return self._export_manager
        except Exception:
            LOGGER.error("创建 ExportManager 失败，回退到内联实现", exc_info=True)
            from .export_manager import ExportManager
            return ExportManager(self)

    def _export_excel_report(self):
        """导出 Excel 报告（含多工作表：摘要、接触者、风险评估详情）"""
        if not self.results:
            messagebox.showwarning("警告", "请先执行风险评估")
            return
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                title="导出 Excel 报告",
                initialfile="结核病风险评估报告.xlsx"
            )
            if not file_path:
                return
            self.export_to_excel(file_path)
            messagebox.showinfo("成功", f"Excel 报告已导出到：{file_path}")
        except ImportError:
            messagebox.showerror("错误", "Excel 导出需要 openpyxl。请安装: pip install openpyxl")
        except Exception as e:
            messagebox.showerror("错误", f"导出 Excel 报告失败: {str(e)}")

    def _export_json_report(self):
        """导出 JSON 报告（结构化数据，便于程序化处理和 API 集成）"""
        if not self.results:
            messagebox.showwarning("警告", "请先执行风险评估")
            return
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="导出 JSON 报告",
                initialfile="结核病风险评估报告.json"
            )
            if not file_path:
                return
            self.export_to_json(file_path)
            messagebox.showinfo("成功", f"JSON 报告已导出到：{file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出 JSON 报告失败: {str(e)}")

    def _export_ml_training_report(self):
        """导出模型训练报告（HTML 格式，含特征重要性、模型性能、校准曲线）"""
        if not hasattr(self, 'ml_predictor') or self.ml_predictor is None:
            messagebox.showwarning("警告", "ML 模块未初始化")
            return
        if not self.ml_predictor.use_real_data:
            messagebox.showwarning("警告", "请先使用真实数据训练 ML 模型")
            return
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".html",
                filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
                title="导出模型训练报告",
                initialfile="模型训练报告.html"
            )
            if not file_path:
                return
            self.export_model_training_report(self.ml_predictor, file_path)
            messagebox.showinfo("成功", f"模型训练报告已导出到：{file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出训练报告失败: {str(e)}")

    def _export_pdf_report(self):
        """导出完整的 PDF 报告（委托 export_utils._export_pdf_report 纯函数）

        GUI 层仅负责：文件对话框选择、消息框提示、reportlab 可用性检查。
        核心 PDF 生成逻辑委托给 tb_risk.export_utils._export_pdf_report，
        确保 GUI 和非 GUI 场景使用同一套导出逻辑，避免分叉风险。
        """
        if not self.results:
            messagebox.showwarning("警告", "请先执行风险评估")
            return

        # 检查 reportlab 可用性
        try:
            from reportlab.lib.pagesizes import letter
        except ImportError:
            messagebox.showinfo("提示", "PDF 导出功能需要安装 reportlab 库。\n\n"
                                       "当前暂使用 TXT 导出替代，请安装后重试。\n"
                                       "命令: pip install reportlab")
            self._export_results()
            return

        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                title="导出 PDF 报告"
            )
            if not file_path:
                return

            # 委托纯函数生成 PDF（数据安全：统一强化脱敏后再导出）
            from tb_risk.export_utils import _export_pdf_report
            masked_results = self._mask_for_export(self.results)
            masked_patient = self._mask_for_export(self.patient_info or {})
            success = _export_pdf_report(masked_results, masked_patient, file_path)

            if success:
                self._log_security_operation("EXPORT", details={"format": "pdf",
                                                                 "filepath": file_path})
                messagebox.showinfo("成功", f"PDF 报告已导出到：{file_path}")
            else:
                messagebox.showwarning("警告", "PDF 导出失败，请检查日志。")
                self._export_results()

        except Exception as e:
            messagebox.showerror("错误", f"导出 PDF 报告时发生错误：{str(e)}")
            self._export_results()