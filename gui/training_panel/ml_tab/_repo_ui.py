#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 训练面板 - 模型仓库与训练历史 UI（Section VII）

提供模型仓库浏览对话框和训练历史查看对话框。
"""

from .._shared import *


class _RepoUIMixin:
    """模型仓库与训练历史 UI 方法"""

    def _show_model_repository(self):
        """显示模型仓库浏览对话框

        展示仓库中所有模型的版本列表，支持加载、删除、对比。
        """
        try:
            from ..model_repo import ModelRepository
        except ImportError:
            messagebox.showerror("错误", "模型仓库模块不可用")
            return

        repo = ModelRepository()
        records = repo.list_models()

        top = tk.Toplevel(self.root)
        top.title("模型仓库")
        top.transient(self.root)
        top.grab_set()
        top.geometry("800x500")

        # 顶部统计
        stats_frame = ttk.Frame(top, padding=10)
        stats_frame.pack(fill='x')
        ttk.Label(stats_frame, text=f"仓库路径: {repo.base_dir}",
                  font=('Arial', 9), foreground='gray').pack(anchor='w')
        ttk.Label(stats_frame, text=f"模型总数: {len(records)} 个",
                  font=('Arial', 10, 'bold')).pack(anchor='w')

        # 模型列表 Treeview
        list_frame = ttk.Frame(top, padding=10)
        list_frame.pack(fill='both', expand=True)

        columns = ('version', 'type', 'timestamp', 'duration', 'metrics', 'notes')
        tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15,
                            selectmode='extended')
        tree.heading('version', text='版本')
        tree.heading('type', text='类型')
        tree.heading('timestamp', text='时间戳')
        tree.heading('duration', text='耗时(秒)')
        tree.heading('metrics', text='性能指标')
        tree.heading('notes', text='备注')

        tree.column('version', width=60)
        tree.column('type', width=50)
        tree.column('timestamp', width=150)
        tree.column('duration', width=80)
        tree.column('metrics', width=250)
        tree.column('notes', width=150)

        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        for r in records:
            metrics_str = ', '.join(f'{k}={v:.3f}' for k, v in r.metrics.items())
            tree.insert('', 'end', values=(
                r.version, r.model_type, r.timestamp,
                f'{r.training_duration:.1f}',
                metrics_str or '-', r.notes or '-'
            ))

        # 底部按钮
        btn_frame = ttk.Frame(top, padding=10)
        btn_frame.pack(fill='x')

        def _load_selected():
            sel = tree.selection()
            if not sel:
                messagebox.showwarning("提示", "请先选择一个模型", parent=top)
                return
            vals = tree.item(sel[0])['values']
            version = str(vals[0])
            path = repo.load_model_path(version)
            if path is None:
                messagebox.showerror("错误", f"模型 {version} 的文件不存在", parent=top)
                return
            # 加载模型
            if vals[1] == 'ml' and self.ml_predictor is not None:
                if self.ml_predictor.load_model(path):
                    # 展示迁移阶段/预训练源/微调样本等最新版本信息
                    try:
                        status = self.ml_predictor.summarize_transfer_status()
                        transfer_line = f"\n{status['message']}" if status.get('message') else ''
                    except Exception:
                        transfer_line = ''
                    messagebox.showinfo(
                        "成功",
                        f"已加载模型 {version}\n训练样本量: {self.ml_predictor.training_sample_count}"
                        f"{transfer_line}",
                        parent=top)
                    self._update_ml_warning_label()
                    if self.results:
                        self._run_ml_predictions()
                        self._update_ml_charts()
                        self._display_ml_results()
                else:
                    messagebox.showerror("错误", "模型加载失败", parent=top)
            elif vals[1] == 'gnn':
                messagebox.showinfo("提示",
                    f"GNN 模型 {version} 路径已复制，请使用'加载GNN'按钮加载:\n{path}",
                    parent=top)
            top.destroy()

        def _delete_selected():
            sel = tree.selection()
            if not sel:
                messagebox.showwarning("提示", "请先选择一个模型", parent=top)
                return
            vals = tree.item(sel[0])['values']
            version = str(vals[0])
            if messagebox.askyesno("确认删除",
                                    f"确定要删除模型 {version} 吗？此操作不可撤销。",
                                    parent=top):
                if repo.delete_model(version):
                    tree.delete(sel[0])
                    messagebox.showinfo("成功", f"已删除模型 {version}", parent=top)
                else:
                    messagebox.showerror("错误", "删除失败", parent=top)

        def _compare_selected():
            """版本差异对比：选中 ≥2 个版本时对比验证集指标。"""
            sel = tree.selection()
            if len(sel) < 2:
                messagebox.showwarning(
                    "提示", "请按住 Ctrl 选择至少两个版本进行对比", parent=top)
                return
            versions = [str(tree.item(s)['values'][0]) for s in sel]
            self._show_version_comparison(repo, versions, parent=top)

        def _rollback_selected():
            """回滚到选中版本（记录回滚原因）。"""
            sel = tree.selection()
            if not sel:
                messagebox.showwarning("提示", "请先选择要回滚到的版本", parent=top)
                return
            vals = tree.item(sel[0])['values']
            version = str(vals[0])
            self._rollback_model_version(repo, version, tree, parent=top)

        ttk.Button(btn_frame, text="加载选中模型", command=_load_selected).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="版本对比", command=_compare_selected).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="回滚到此版本", command=_rollback_selected).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="删除选中模型", command=_delete_selected).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="关闭", command=top.destroy).pack(side='right', padx=5)

    def _show_version_comparison(self, repo, versions, parent=None):
        """版本差异对比对话框：各版本在验证集上的指标并排展示 + 差异高亮。"""
        records = repo.compare_models(versions)
        if not records:
            messagebox.showerror("错误", "未找到所选版本的记录", parent=parent)
            return

        top = tk.Toplevel(parent or self.root)
        top.title("模型版本对比")
        top.transient(parent or self.root)
        top.geometry("760x460")

        ttk.Label(top, text=f"对比版本：{', '.join(r.version for r in records)}",
                  font=('Microsoft YaHei', 10, 'bold')).pack(anchor='w', padx=10, pady=8)

        # 指标矩阵：行=指标，列=版本
        cols = ('metric',) + tuple(r.version for r in records)
        frame = ttk.Frame(top, padding=10)
        frame.pack(fill='both', expand=True)
        tree = ttk.Treeview(frame, columns=cols, show='headings', height=14)
        tree.heading('metric', text='指标')
        tree.column('metric', width=160, anchor='w')
        for r in records:
            tree.heading(r.version, text=f"{r.version}\n({r.model_type})")
            tree.column(r.version, width=120, anchor='center')
        sb = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

        # 汇总全部指标键
        metric_keys = []
        for r in records:
            for k in r.metrics:
                if k not in metric_keys:
                    metric_keys.append(k)

        # 越大越好的指标（其余视为越小越好）
        higher_better = {'accuracy', 'auc', 'AUROC', 'AUPRC', 'f1', 'precision',
                         'recall', 'c_index', 'sensitivity', 'specificity'}

        def _is_higher_better(key):
            return any(h.lower() in key.lower() for h in higher_better)

        for key in metric_keys:
            values = []
            for r in records:
                v = r.metrics.get(key)
                values.append(v)
            nums = [v for v in values if isinstance(v, (int, float))]
            best = None
            if len(nums) >= 2:
                best = max(nums) if _is_higher_better(key) else min(nums)
            row = [key]
            for v in values:
                if v is None:
                    row.append('—')
                elif isinstance(v, float):
                    mark = ' ★' if best is not None and abs(v - best) < 1e-9 else ''
                    row.append(f"{v:.4f}{mark}")
                else:
                    row.append(str(v))
            tree.insert('', 'end', values=row)

        # 训练参数差异
        param_keys = []
        for r in records:
            for k in r.params:
                if k not in param_keys:
                    param_keys.append(k)
        for key in param_keys:
            values = [r.params.get(key) for r in records]
            if len({str(v) for v in values}) <= 1:
                continue  # 参数相同则不展示
            tree.insert('', 'end', values=(
                f"[参数] {key}",) + tuple(
                    '—' if v is None else str(v) for v in values))

        ttk.Label(top, text="★ = 该指标的最优版本（越大越好/越小越好按指标语义自动判断）",
                  foreground='gray').pack(anchor='w', padx=10, pady=(0, 8))
        ttk.Button(top, text="关闭", command=top.destroy).pack(pady=8)

    def _rollback_model_version(self, repo, version, tree=None, parent=None):
        """回滚流程：确认 + 填写原因 + 加载旧版本模型。"""
        record = repo.rollback(version)
        if record is None:
            messagebox.showerror("错误", f"版本 {version} 不存在", parent=parent)
            return

        # 回滚原因输入对话框
        dlg = tk.Toplevel(parent or self.root)
        dlg.title("回滚原因")
        dlg.transient(parent or self.root)
        dlg.grab_set()
        dlg.geometry("420x180")
        ttk.Label(dlg, text=f"回滚到版本 {version}（{record.model_type}，"
                            f"{record.timestamp}）\n请填写回滚原因（记录到版本备注）：",
                  wraplength=380, justify='left').pack(anchor='w', padx=10, pady=8)
        reason_text = tk.Text(dlg, height=3, wrap='word')
        reason_text.pack(fill='x', padx=10)
        reason_text.focus_set()

        def _confirm():
            reason = reason_text.get('1.0', 'end').strip()
            if not messagebox.askyesno(
                    "确认回滚",
                    f"确定回滚到 {version} 吗？\n较新版本将保留在仓库中（不删除）。\n\n"
                    f"原因：{reason or '（未填写）'}", parent=dlg):
                return
            repo.record_rollback(version, reason)
            path = repo.load_model_path(version)
            if path is None:
                messagebox.showerror("错误", f"版本 {version} 的模型文件不存在",
                                     parent=dlg)
                dlg.destroy()
                return
            loaded = False
            if record.model_type == 'ml' and self.ml_predictor is not None:
                loaded = bool(self.ml_predictor.load_model(path))
            if loaded:
                messagebox.showinfo(
                    "回滚成功",
                    f"已回滚并加载模型 {version}，回滚原因已记录。", parent=dlg)
                # 刷新 ML 结果与图表
                try:
                    self._update_ml_warning_label()
                    if self.results:
                        self._run_ml_predictions()
                        self._update_ml_charts()
                        self._display_ml_results()
                except Exception as e:
                    LOGGER.debug("回滚后刷新 ML 结果失败（非致命）: %s", e)
            elif record.model_type == 'gnn':
                messagebox.showinfo(
                    "回滚成功",
                    f"已记录回滚原因。GNN 模型请使用「加载GNN」按钮加载:\n{path}",
                    parent=dlg)
            else:
                messagebox.showwarning(
                    "提示", "已记录回滚原因，但当前无可用的模型加载器。", parent=dlg)
            dlg.destroy()

        btns = ttk.Frame(dlg)
        btns.pack(pady=8)
        ttk.Button(btns, text="确认回滚", command=_confirm).pack(side='left', padx=5)
        ttk.Button(btns, text="取消", command=dlg.destroy).pack(side='left', padx=5)

    def _show_training_history(self):
        """显示训练历史对话框

        展示历史训练记录，包含时间戳、模型类型、参数、状态、耗时。
        """
        try:
            from ..training_log import TrainingLogger
        except ImportError:
            messagebox.showerror("错误", "训练日志模块不可用")
            return

        logger = TrainingLogger()
        entries = logger.read_history(limit=100)
        stats = logger.get_stats()

        top = tk.Toplevel(self.root)
        top.title("训练历史")
        top.transient(self.root)
        top.grab_set()
        top.geometry("800x500")

        # 顶部统计
        stats_frame = ttk.Frame(top, padding=10)
        stats_frame.pack(fill='x')
        ttk.Label(stats_frame,
                  text=f"总训练次数: {stats['total_count']} | "
                       f"成功: {stats['success_count']} | "
                       f"失败: {stats['failure_count']} | "
                       f"平均耗时: {stats['avg_duration']:.1f}s",
                  font=('Arial', 10, 'bold')).pack(anchor='w')
        if stats['last_training_time']:
            ttk.Label(stats_frame,
                      text=f"最近训练: {stats['last_training_time']}",
                      font=('Arial', 9), foreground='gray').pack(anchor='w')

        # 历史列表 Treeview
        list_frame = ttk.Frame(top, padding=10)
        list_frame.pack(fill='both', expand=True)

        columns = ('timestamp', 'type', 'status', 'duration', 'params', 'error')
        tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        tree.heading('timestamp', text='时间戳')
        tree.heading('type', text='类型')
        tree.heading('status', text='状态')
        tree.heading('duration', text='耗时(秒)')
        tree.heading('params', text='参数摘要')
        tree.heading('error', text='错误信息')

        tree.column('timestamp', width=150)
        tree.column('type', width=50)
        tree.column('status', width=80)
        tree.column('duration', width=80)
        tree.column('params', width=250)
        tree.column('error', width=200)

        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        for e in entries:
            params_str = ', '.join(f'{k}={v}' for k, v in
                                    list(e.params.items())[:4])
            if len(e.params) > 4:
                params_str += '...'
            tree.insert('', 'end', values=(
                e.timestamp, e.model_type, e.status,
                f'{e.training_duration:.1f}',
                params_str or '-',
                (e.error_message[:50] + '...' if len(e.error_message) > 50
                 else e.error_message) or '-'
            ))

        # 底部按钮
        btn_frame = ttk.Frame(top, padding=10)
        btn_frame.pack(fill='x')

        def _clear_history():
            if messagebox.askyesno("确认清空",
                                    "确定要清空所有训练历史吗？此操作不可撤销。",
                                    parent=top):
                count = logger.clear_history()
                messagebox.showinfo("成功", f"已清空 {count} 条记录", parent=top)
                top.destroy()

        ttk.Button(btn_frame, text="清空历史", command=_clear_history).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="关闭", command=top.destroy).pack(side='right', padx=5)


__all__ = ['_RepoUIMixin']
