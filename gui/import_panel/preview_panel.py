#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 导入面板 - 预览与进度显示（PreviewMixin）"""

from ._shared import *  # noqa: F401,F403


class PreviewMixin:
    """预览与进度显示相关方法"""

    def _show_import_preview(self, entries, import_type="家庭成员", quality_scores=None,
                           field_mappings=None, auto_corrections=None):
        """显示导入数据预览表格（增强版：含质量评分和自动修正高亮）

        参数：
            entries: 导入数据列表
            import_type: 导入类型
            quality_scores: DataQualityScorer.score_batch() 返回的结果（可选）
            field_mappings: 列名映射结果（可选）
            auto_corrections: 自动修正记录 {record_index: {field: (original, corrected)}}
        """
        if not entries:
            messagebox.showinfo("提示", "没有可预览的数据")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title(f"导入数据预览 - {import_type}")
        dlg.geometry("800x550")
        dlg.transient(self.root)
        dlg.grab_set()

        header = ttk.Frame(dlg, padding=10)
        header.pack(fill='x')

        # 质量评分摘要
        if quality_scores and 'summary' in quality_scores:
            summary = quality_scores['summary']
            ttk.Label(header,
                      text=f"预览 {import_type} ({len(entries)}条) | "
                           f"质量分: {summary.get('mean_score', 0):.2f} | "
                           f"标记复核: {summary.get('flagged_count', 0)}条",
                      font=self._FONT_HEADING).pack(anchor='w')
            if summary.get('flagged_count', 0) > 0:
                ttk.Label(header,
                          text=f"⚠ {summary['flagged_count']} 条记录需人工复核，请检查后确认导入",
                          font=self._FONT_SMALL, foreground=self.COLORS['warning']).pack(anchor='w')
        else:
            ttk.Label(header, text=f"预览 {import_type} ({len(entries)}条)",
                      font=self._FONT_HEADING).pack(anchor='w')
            ttk.Label(header,
                      text="请确认数据无误后点击导入。红色标记的值为异常/缺失值。",
                      font=self._FONT_SMALL, foreground=self.COLORS['warning']).pack(anchor='w')

        # 自动修正提示
        if auto_corrections:
            correction_count = sum(len(corr) for corr in auto_corrections.values())
            if correction_count > 0:
                ttk.Label(header,
                          text=f"🔧 已自动修正 {correction_count} 个字段值（如'涂阳'→2、'缺失'→中位数）",
                          font=self._FONT_SMALL, foreground='#2196F3').pack(anchor='w')

        tree_frame = ttk.Frame(dlg, padding=5)
        tree_frame.pack(fill='both', expand=True)

        display_keys = ['name', 'age', 'relationship', 'has_tb', 'cumulative_exposure',
                        'contact_distance', 'freq_density']
        columns = [k for k in display_keys if any(k in e for e in entries)]

        tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=min(15, len(entries)))
        col_labels = {'name': '姓名', 'age': '年龄', 'relationship': '关系',
                      'has_tb': 'TB史', 'cumulative_exposure': '累积暴露',
                      'contact_distance': '距离', 'freq_density': '频次'}
        for col in columns:
            tree.heading(col, text=col_labels.get(col, col), anchor='center')
            tree.column(col, width=max(80, 100 if col in ('name', 'relationship') else 70),
                       anchor='center')

        tree.tag_configure('missing', foreground=self.COLORS['danger'])
        tree.tag_configure('warning', foreground=self.COLORS['warning'])
        tree.tag_configure('corrected', foreground='#2196F3')

        for i, entry in enumerate(entries[:20]):
            row_values = []
            tags = []
            corrections = (auto_corrections or {}).get(i, {})
            for col in columns:
                val = entry.get(col, '')
                # 显示自动修正后的值，并在前面加标记
                if col in corrections:
                    row_values.append(f"✓{val}" if val is not None else '')
                    if 'corrected' not in tags:
                        tags.append('corrected')
                else:
                    row_values.append(str(val) if val is not None else '')
                if col in ('name', 'age') and (val is None or val == ''):
                    if 'missing' not in tags:
                        tags = ['missing']
                elif col in ('cumulative_exposure', 'freq_density') and (
                    val is None or val == '' or (isinstance(val, (int, float)) and val == 0)):
                    if 'missing' not in tags and 'corrected' not in tags:
                        tags.append('warning')
            tree.insert('', 'end', values=row_values, tags=tuple(tags) if tags else ())

        scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        btn_frame = ttk.Frame(dlg, padding=10)
        btn_frame.pack(fill='x')

        _import_confirmed = [False]

        def on_confirm():
            _import_confirmed[0] = True
            dlg.destroy()

        def on_cancel():
            dlg.destroy()

        ttk.Button(btn_frame, text="取消",
                  command=on_cancel).pack(side='right', padx=5)
        ttk.Button(btn_frame, text="✓ 确认导入", style='Accent.TButton',
                  command=on_confirm).pack(side='right', padx=5)

        dlg.wait_window()
        return _import_confirmed[0]



    def _show_low_confidence_dialog(self, low_confidence_matches):
        """显示低置信度匹配置确认对话框。

        参数：
            low_confidence_matches: [(原始列名, 匹配字段, 置信度), ...]

        返回：
            dict: {原始列名: 用户确认的标准字段名} 或 None（用户取消）
        """
        if not low_confidence_matches:
            return {}

        dlg = tk.Toplevel(self.root)
        dlg.title("低置信度匹配置确认")
        dlg.geometry("550x400")
        dlg.transient(self.root)
        dlg.grab_set()

        ttk.Label(dlg, text="以下列名的匹配置信度较低，请确认或手动选择正确的字段：",
                  font=self._FONT_HEADING, padding=10).pack(anchor='w')

        # 获取所有可用标准字段
        try:
            from ..data_io.synonyms import FIELD_SYNONYM_LIBRARY
            available_fields = list(FIELD_SYNONYM_LIBRARY.keys())
        except ImportError:
            available_fields = ['age', 'sputum_smear', 'has_cavity', 'bcg_vaccine',
                               'has_tb', 'treatment', 'ventilation', 'contact_distance',
                               'exposure_setting', 'cough_freq', 'symptoms', 'delay_days']

        field_choices = []  # [(orig_name, match_name, confidence, StringVar)]

        for orig_name, matched_field, confidence in low_confidence_matches:
            frame = ttk.Frame(dlg, padding=5)
            frame.pack(fill='x', padx=10, pady=2)

            ttk.Label(frame, text=f"'{orig_name}'", font=('Microsoft YaHei', 10, 'bold'),
                      width=20).pack(side='left')
            ttk.Label(frame, text=f"→ 置信度: {confidence:.2f}",
                      foreground='orange').pack(side='left', padx=5)

            var = tk.StringVar(value=matched_field if matched_field else '')
            combo = ttk.Combobox(frame, textvariable=var, values=available_fields,
                                width=25, state='readonly')
            combo.pack(side='right', padx=5)
            field_choices.append((orig_name, var))

        _confirmed = [None]

        def on_confirm():
            result = {}
            for orig_name, var in field_choices:
                val = var.get().strip()
                if val:
                    result[orig_name] = val
            _confirmed[0] = result
            dlg.destroy()

        def on_cancel():
            dlg.destroy()

        btn_frame = ttk.Frame(dlg, padding=10)
        btn_frame.pack(fill='x', side='bottom')
        ttk.Button(btn_frame, text="取消",
                  command=on_cancel).pack(side='right', padx=5)
        ttk.Button(btn_frame, text="✓ 确认并保存",
                  command=on_confirm).pack(side='right', padx=5)

        ttk.Label(dlg, text="用户的选择将被写入同义词库，下次自动匹配。",
                  font=self._FONT_SMALL, foreground='#5a6c7d',
                  padding=5).pack(side='bottom')

        dlg.wait_window()
        return _confirmed[0]


    def _show_progress_dialog(self, title="正在处理..."):
        """显示进度条对话框（Section VIII: 取消按钮真正生效）

        参数：
            title: 对话框标题

        返回：
            (进度对话框, 进度变量, 更新回调函数, 关闭回调函数, 取消标志)
            取消标志为 threading.Event，用户点击取消时 set()
        """
        import threading

        progress_window = tk.Toplevel(self.root)
        progress_window.title(title)
        progress_window.geometry("400x170")
        progress_window.resizable(False, False)
        progress_window.transient(self.root)
        progress_window.grab_set()

        # 居中显示
        progress_window.update_idletasks()
        width = progress_window.winfo_width()
        height = progress_window.winfo_height()
        x = (progress_window.winfo_screenwidth() // 2) - (width // 2)
        y = (progress_window.winfo_screenheight() // 2) - (height // 2)
        progress_window.geometry('{}x{}+{}+{}'.format(width, height, x, y))

        # 进度条
        progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100, length=350)
        progress_bar.pack(pady=15, padx=20)

        # 状态标签
        status_label = ttk.Label(progress_window, text="准备中...")
        status_label.pack(pady=5)

        # Section VIII: 取消按钮真正生效（使用 threading.Event 信号）
        cancel_event = threading.Event()

        def on_cancel():
            cancel_event.set()
            cancel_btn.config(state='disabled', text="正在取消...")
            status_label.config(text="正在取消，请稍候...")

        cancel_btn = ttk.Button(progress_window, text="取消", command=on_cancel)
        cancel_btn.pack(pady=10)

        def update_progress(value, message=None):
            """更新进度条（线程安全）"""
            try:
                self.root.after(0, lambda: _do_update_progress(value, message))
            except (tk.TclError, RuntimeError, AttributeError):
                pass

        def _do_update_progress(value, message=None):
            try:
                progress_var.set(value)
                if message:
                    status_label.config(text=message)
                progress_window.update_idletasks()
            except (tk.TclError, RuntimeError, AttributeError):
                pass

        def close_progress():
            """关闭进度对话框"""
            try:
                progress_window.destroy()
            except (tk.TclError, RuntimeError, AttributeError):
                pass

        # 向后兼容：返回 4 个值（原有调用方期望 4-tuple）
        # 第 5 个元素 cancel_event 通过 progress_window.cancel_event 属性访问
        progress_window.cancel_event = cancel_event
        return progress_window, progress_var, update_progress, close_progress