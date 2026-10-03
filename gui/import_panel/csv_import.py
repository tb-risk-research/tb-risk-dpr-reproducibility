#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 导入面板 - CSV 导入（CSVImportMixin）"""

from ._shared import *  # noqa: F401,F403


class CSVImportMixin:
    """CSV 导入相关方法"""

    def _export_csv_template(self):
        """导出 CSV 模板文件"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="导出 CSV 模板",
            initialfile="结核病风险评估模板.csv"
        )

        if file_path:
            if self.export_csv_template(file_path):
                messagebox.showinfo("成功", f"CSV 模板已导出到：{file_path}")
            else:
                messagebox.showerror("错误", "导出 CSV 模板失败")


    def _load_from_csv(self):
        """从 CSV 文件加载数据（Section VIII: 走统一导入管线）"""
        from .pipeline import CSVImportPipeline
        pipeline = CSVImportPipeline(self)
        pipeline.run()