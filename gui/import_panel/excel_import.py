#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 导入面板 - Excel 导入（ExcelImportMixin）"""

from ._shared import *  # noqa: F401,F403


class ExcelImportMixin:
    """Excel 导入相关方法"""

    def _load_from_excel(self):
        """从 Excel 文件加载数据（Section VIII: 走统一导入管线）"""
        if not PANDAS_AVAILABLE:
            messagebox.showerror("依赖缺失",
                               "Excel导入功能需要安装pandas库。\n\n请运行以下命令安装：\n\npip install pandas openpyxl")
            return
        from .pipeline import ExcelImportPipeline
        pipeline = ExcelImportPipeline(self)
        pipeline.run()

    def _do_load_excel(self, file_path, mode='replace'):
        """执行 Excel 导入（保留向后兼容，内部委托给管线）"""
        if mode == 'replace':
            self.family_entries.clear()
            self.social_entries.clear()
        if self.import_excel(file_path, mode=mode):
            self._update_gui_from_import()
            summary = self.get_import_summary()
            messagebox.showinfo("成功",
                f"数据已从 Excel 文件加载（模式：{mode}）：\n"
                f"- 家庭成员：{summary['family_members']}人\n"
                f"- 社会接触者：{summary['social_contacts']}人")