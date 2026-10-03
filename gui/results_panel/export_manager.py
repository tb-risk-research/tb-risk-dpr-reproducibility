#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 结果面板 - 统一导出管理器（Section IX）

将各格式导出逻辑（图表 / 文本 / CSV / PDF）统一收敛到 ``ExportManager`` 类，
各格式实现为独立方法，便于复用与单测。``ExportManager`` 为独立类，不依赖 Mixin，
通过 ``app`` 参数访问 GUI 状态（``self.results`` / ``self.ml_results`` / 图表 figure 对象等）。

关键要点：
- SVG/PDF 适合学术发表场景；PNG 适合日常归档
- 导出前检查 ``results_stale`` 标记，若为 True 则提示用户先重新评估
- 使用 matplotlib 的 ``savefig`` 原生支持 SVG（无需额外依赖）和 PDF
  （matplotlib 的 pdf 后端），reportlab 用于带版式的 PDF 报告
"""

from ._shared import *


# reportlab 可用性检测（仅用于 PDF 报告，不影响 SVG/PNG/PDF 图表导出）
try:
    from reportlab.lib.pagesizes import A4
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class ExportManager:
    """统一导出管理器（独立类，不依赖 Mixin）

    通过 ``app`` 参数访问 GUI 状态：``app.results`` / ``app.ml_results`` /
    各类 ``app.<chart>_figure`` 对象。所有方法均做 try/except 防护，避免异常
    影响主流程。
    """

    # 支持的图表导出格式
    SUPPORTED_CHART_FORMATS = ('png', 'svg', 'pdf')

    def __init__(self, app):
        """初始化导出管理器

        Args:
            app: GUI 主应用对象（ResultsMixin 实例），需提供：
                - ``results``: 评估结果字典
                - ``results_stale``: bool 标记结果是否过期
                - ``<chart>_figure``: matplotlib Figure 对象
        """
        self.app = app

    # ============ 内部工具 ============

    def _check_stale(self):
        """检查结果是否过期，过期则提示用户重新评估

        Returns:
            bool: True 表示可继续导出；False 表示已过期且应中止
        """
        try:
            if getattr(self.app, 'results_stale', False):
                # 通过 messagebox 提示用户先重新评估
                if hasattr(self.app, 'root') and self.app.root is not None:
                    messagebox.showwarning(
                        "结果已过期",
                        "数据已修改，当前结果可能不准确。\n"
                        "建议先重新执行风险评估，再导出结果。"
                    )
                return False
        except Exception:
            LOGGER.error("检查 results_stale 失败", exc_info=True)
        return True

    def _get_results_copy(self):
        """线程安全地获取 results 副本

        Returns:
            tuple: (results_copy, error_message)；成功时 error_message 为 None
        """
        try:
            data_lock = getattr(self.app, '_data_lock', None)
            if data_lock is not None:
                with data_lock:
                    if not self.app.results:
                        return None, "请先执行风险评估"
                    results_copy = copy.deepcopy(self.app.results)
            else:
                if not self.app.results:
                    return None, "请先执行风险评估"
                results_copy = copy.deepcopy(self.app.results)
            return results_copy, None
        except Exception as e:
            return None, f"获取结果数据失败: {e}"

    def _get_patient_info(self):
        """获取 patient_info（线程安全）"""
        try:
            data_lock = getattr(self.app, '_data_lock', None)
            if data_lock is not None:
                with data_lock:
                    return copy.deepcopy(getattr(self.app, 'patient_info', {}) or {})
            return copy.deepcopy(getattr(self.app, 'patient_info', {}) or {})
        except Exception:
            return {}

    # ============ 图表导出 ============

    def export_chart(self, figure, path, format='png'):
        """导出单个 matplotlib Figure 到指定路径

        支持 PNG/SVG/PDF 三种格式，均通过 matplotlib 的 ``savefig`` 实现：
        - PNG: 位图，dpi=150，日常归档使用
        - SVG: 矢量图，适合学术发表/再编辑
        - PDF: 矢量图，适合学术发表/直接打印

        Args:
            figure: matplotlib.figure.Figure 对象
            path (str): 输出文件路径（不含扩展名，扩展名会根据 format 自动添加）
            format (str): 导出格式，'png' / 'svg' / 'pdf'

        Returns:
            str: 实际写入的文件路径；失败时返回 None
        """
        if not MATPLOTLIB_AVAILABLE:
            LOGGER.warning("matplotlib 未安装，无法导出图表")
            return None
        if figure is None:
            LOGGER.warning("figure 为 None，跳过导出")
            return None

        format_lower = (format or 'png').lower()
        if format_lower not in self.SUPPORTED_CHART_FORMATS:
            LOGGER.warning("不支持的图表格式: %s，回退到 png", format)
            format_lower = 'png'

        # 自动添加扩展名
        if not path.lower().endswith(f'.{format_lower}'):
            full_path = f"{path}.{format_lower}"
        else:
            full_path = path

        try:
            # PNG 使用 dpi=150；SVG/PDF 为矢量图，dpi 不影响最终质量但保留以兼容
            kwargs = {'bbox_inches': 'tight'}
            if format_lower == 'png':
                kwargs['dpi'] = 150
            figure.savefig(full_path, format=format_lower, **kwargs)
            return full_path
        except Exception as e:
            LOGGER.error("导出图表失败 (%s -> %s): %s", format_lower, full_path, e,
                         exc_info=True)
            return None

    def export_all_charts(self, app, folder, format='png'):
        """批量导出所有图表到指定目录

        导出列表包含：SEIR 曲线、风险热力图、概率分布直方图、延迟影响曲线、
        ML 对比图、SHAP 特征重要性图、ML 性能图、校准曲线、个体 SHAP 图、因果 DAG 图。

        Args:
            app: GUI 主应用对象（保留参数以符合任务约定的签名）
            folder (str): 输出目录
            format (str): 导出格式，'png' / 'svg' / 'pdf'

        Returns:
            list: 成功导出的文件路径列表
        """
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showwarning("警告", "matplotlib 未安装，无法导出图表")
            return []

        format_lower = (format or 'png').lower()
        if format_lower not in self.SUPPORTED_CHART_FORMATS:
            LOGGER.warning("不支持的图表格式: %s，回退到 png", format)
            format_lower = 'png'

        # 导出前检查 stale 标记
        if not self._check_stale():
            return []

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 图表对象 -> 文件名前缀映射（按 init_ui.py 中的属性名）
        chart_specs = [
            ('seir_figure', 'seir_curve'),
            ('heatmap_figure', 'risk_heatmap'),
            ('histogram_figure', 'probability_distribution'),
            ('delay_figure', 'delay_impact'),
            ('ml_compare_figure', 'ml_comparison'),
            ('shap_figure', 'shap_importance'),
            ('ml_perf_figure', 'ml_performance'),
            ('ml_calib_figure', 'ml_calibration'),  # Section IX: 校准曲线
            ('shap_ind_figure', 'shap_individual'),  # Section IX: 个体 SHAP 图
            ('dag_figure', 'causal_dag'),
            ('risk_figure', 'risk_distribution'),
        ]

        exported = []
        for attr_name, file_prefix in chart_specs:
            figure = getattr(app, attr_name, None)
            if figure is None:
                continue
            file_path = os.path.join(folder, f"{file_prefix}_{timestamp}")
            actual_path = self.export_chart(figure, file_path, format=format_lower)
            if actual_path:
                exported.append(actual_path)

        return exported

    # ============ 文本导出 ============

    def export_results_text(self, results, path):
        """导出评估结果为文本文件

        Args:
            results (dict): 评估结果字典
            path (str): 输出文件路径

        Returns:
            bool: 是否成功
        """
        if not results:
            LOGGER.warning("结果为空，无法导出文本")
            return False
        try:
            with open(path, 'w', encoding='utf-8') as f:
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

                f.write("\n" + "=" * 60 + "\n")
                f.write("报告生成时间：" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            return True
        except Exception as e:
            LOGGER.error("导出文本结果失败: %s", e, exc_info=True)
            return False

    # ============ CSV 导出 ============

    def export_results_csv(self, results, path):
        """导出评估结果为 CSV 文件

        Args:
            results (dict): 评估结果字典
            path (str): 输出文件路径

        Returns:
            bool: 是否成功
        """
        if not results:
            LOGGER.warning("结果为空，无法导出 CSV")
            return False
        try:
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
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
                            writer.writerow([
                                patient.get('name', '未知'),
                                f"{patient.get('disease_probability', 0):.1f}%",
                                patient.get('priority', 'N/A'),
                            ])
                        writer.writerow([])

                    social_patients = results['potential_patients'].get('social', [])
                    if social_patients:
                        writer.writerow(['社会接触者潜在患者'])
                        writer.writerow(['姓名', '发病概率', '优先级'])
                        for patient in social_patients:
                            writer.writerow([
                                patient.get('name', '未知'),
                                f"{patient.get('disease_probability', 0):.1f}%",
                                patient.get('priority', 'N/A'),
                            ])
            return True
        except Exception as e:
            LOGGER.error("导出 CSV 结果失败: %s", e, exc_info=True)
            return False

    # ============ PDF 报告导出 ============

    def export_pdf_report(self, results, patient_info, path):
        """导出 PDF 报告（基于 reportlab）

        Section IX 调整：PDF 导出不再静默降级为 TXT。
        reportlab 不可用时返回 False 并由调用方显示明确提示。

        Args:
            results (dict): 评估结果字典
            patient_info (dict): 患者基本信息
            path (str): 输出 PDF 文件路径

        Returns:
            bool: 是否成功；reportlab 不可用时返回 False
        """
        if not REPORTLAB_AVAILABLE:
            # 明确提示用户 reportlab 未安装，不静默降级
            LOGGER.warning("reportlab 未安装，PDF 报告无法生成")
            return False

        if not results:
            LOGGER.warning("结果为空，无法导出 PDF")
            return False

        try:
            # 数据安全：统一强化脱敏后再导出
            if hasattr(self.app, '_mask_for_export'):
                results = self.app._mask_for_export(results)
                patient_info = self.app._mask_for_export(patient_info or {}) or {}
            # 复用 export_utils 中的纯函数实现，避免逻辑分叉
            from tb_risk.export_utils import _export_pdf_report as _do_export
            ok = _do_export(results, patient_info or {}, path)
            if ok and hasattr(self.app, '_log_security_operation'):
                self.app._log_security_operation(
                    "EXPORT", details={"format": "pdf", "filepath": path})
            return ok
        except Exception as e:
            LOGGER.error("导出 PDF 报告失败: %s", e, exc_info=True)
            return False
