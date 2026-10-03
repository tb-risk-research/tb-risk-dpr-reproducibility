#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 导入面板 - 数据库导入/导出 Mixin（_load_from_database / _export_to_database）"""

from ._shared import *


class DatabaseMixin:
    """数据库导入/导出方法"""

    def _load_from_database(self):
        """从数据库导入数据（Section VIII: 走统一导入管线，动态参数 + 测试连接）"""
        from .pipeline import DatabaseImportPipeline
        pipeline = DatabaseImportPipeline(self)
        pipeline.run()

    def _export_to_database(self):
        """导出数据到数据库"""
        if self.assessing.is_set():
            messagebox.showwarning("警告", "正在评估中，请稍后再导出")
            return

        has_data = bool(self.patient_info or self.family_members or self.social_contacts)
        has_results = bool(self.results)

        if not has_data and not has_results:
            messagebox.showwarning("警告", "没有可导出的数据，请先输入数据或执行评估")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("导出到数据库")
        dialog.geometry("550x480")

        ttk.Label(dialog, text="数据库类型：").pack(pady=5, padx=10, anchor='w')
        db_type_var = tk.StringVar(value='sqlite')
        type_frame = ttk.Frame(dialog)
        type_frame.pack(pady=5, padx=10, fill='x')
        ttk.Radiobutton(type_frame, text="SQLite", variable=db_type_var, value='sqlite').pack(side='left', padx=5)
        ttk.Radiobutton(type_frame, text="MySQL", variable=db_type_var, value='mysql').pack(side='left', padx=5)
        ttk.Radiobutton(type_frame, text="PostgreSQL", variable=db_type_var, value='postgresql').pack(side='left', padx=5)

        ttk.Label(dialog, text="连接信息：").pack(pady=5, padx=10, anchor='w')
        conn_frame = ttk.Frame(dialog)
        conn_frame.pack(pady=5, padx=10, fill='x')

        ttk.Label(conn_frame, text="SQLite文件：").grid(row=0, column=0, sticky='w', pady=2)
        sqlite_path_var = tk.StringVar()
        sqlite_entry = ttk.Entry(conn_frame, textvariable=sqlite_path_var)
        sqlite_entry.grid(row=0, column=1, sticky='ew', pady=2)
        ttk.Button(conn_frame, text="浏览...", command=lambda: sqlite_path_var.set(filedialog.asksaveasfilename(
            defaultextension=".db",
            filetypes=[("SQLite files", "*.db *.sqlite"), ("All files", "*.*")],
            title="选择SQLite数据库文件"
        ))).grid(row=0, column=2, padx=5, pady=2)

        ttk.Label(conn_frame, text="MySQL/PostgreSQL连接字符串：").grid(row=1, column=0, columnspan=3, sticky='w', pady=2)
        ttk.Label(conn_frame, text="格式: host:port:user:password:database").grid(row=2, column=0, columnspan=3, sticky='w', pady=0)
        conn_str_var = tk.StringVar(value="localhost:3306:root::database")
        conn_entry = ttk.Entry(conn_frame, textvariable=conn_str_var)
        conn_entry.grid(row=3, column=0, columnspan=3, sticky='ew', pady=2)
        conn_frame.columnconfigure(1, weight=1)

        ttk.Label(dialog, text="导出模式：").pack(pady=5, padx=10, anchor='w')
        mode_frame = ttk.Frame(dialog)
        mode_frame.pack(pady=5, padx=10, fill='x')
        mode_var = tk.StringVar(value='data_and_results')
        ttk.Radiobutton(mode_frame, text="数据和评估结果",
                       variable=mode_var, value='data_and_results').pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="仅数据（患者信息/家庭成员/社会接触者）",
                       variable=mode_var, value='data_only').pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="仅评估结果",
                       variable=mode_var, value='results_only').pack(anchor='w')

        if not has_results:
            for child in mode_frame.winfo_children():
                if hasattr(child, 'cget') and child.cget('value') in ('results_only', 'data_and_results'):
                    child.configure(state='disabled')

        status_label = ttk.Label(dialog, text="", foreground='#5a6c7d')
        status_label.pack(pady=5)

        def do_export():
            db_type = db_type_var.get()
            mode = mode_var.get()

            if mode == 'results_only' and not self.results:
                messagebox.showwarning("警告", "没有评估结果可导出，请先执行评估")
                return

            if db_type == 'sqlite':
                db_path = sqlite_path_var.get()
                if not db_path:
                    messagebox.showwarning("警告", "请选择SQLite文件路径")
                    return
            else:
                db_path = conn_str_var.get()
                if not db_path:
                    messagebox.showwarning("警告", "请输入连接字符串")
                    return

            status_label.configure(text="正在导出...", foreground='#f39c12')
            dialog.update()

            try:
                if self.export_to_database(db_type=db_type, db_path=db_path, mode=mode):
                    dialog.destroy()
                    errors = self.get_import_errors()
                    success_msg = f"数据已成功导出到数据库（模式：{mode}）\n"
                    if self.patient_info and mode != 'results_only':
                        success_msg += "- 患者信息：已导出\n"
                    if self.family_members and mode != 'results_only':
                        success_msg += f"- 家庭成员：{len(self.family_members)}条\n"
                    if self.social_contacts and mode != 'results_only':
                        success_msg += f"- 社会接触者：{len(self.social_contacts)}条\n"
                    if self.results and mode != 'data_only':
                        success_msg += f"- 评估结果：已导出\n"

                    messagebox.showinfo("成功", success_msg)
                else:
                    status_label.configure(text="导出失败", foreground='#e74c3c')
                    errors = self.get_import_errors()
                    self._show_error_details("数据库导出错误", "导出到数据库失败，以下是详情：", errors)
            except Exception as e:
                status_label.configure(text=f"导出失败：{str(e)}", foreground='#e74c3c')
                messagebox.showerror("错误", f"导出过程中发生错误：{str(e)}")

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=20)
        ttk.Button(btn_frame, text="导出", command=do_export).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side='left', padx=5)