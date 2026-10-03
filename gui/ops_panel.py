#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 运维中心面板（部署与运维友好）

提供四个核心能力入口：
1. 运行环境/离线状态：联网检测、本地模型与规则缓存状态。
2. 模型/规则热更新：指定目录监控开关、历史记录。
3. 部署打包：一键生成可分发的离线安装包。
4. 可读错误：训练/运行异常的可读错误与恢复建议（由 ops.errors 提供）。
"""

import os

import tkinter as tk
from tkinter import ttk, messagebox

from ..ops import (
    offline_status, ensure_cached, check_network,
    HotUpdateWatcher, DEFAULT_HOTUPDATE_DIR, pack_distribution,
)


class OpsPanel(tk.Toplevel):
    """运维中心面板（Toplevel）。"""

    def __init__(self, root, app=None):
        super().__init__(root)
        self.app = app
        self.title("运维中心 - 部署与运维")
        self.geometry("720x620")
        self.minsize(620, 520)
        self.transient(root)

        self._watcher = None
        self._build_ui()
        self._refresh_env()
        self._refresh_watcher()

    # ==================== UI 构建 ====================

    def _build_ui(self):
        main = ttk.Frame(self, padding=12)
        main.pack(fill='both', expand=True)

        # ---- 1. 运行环境 / 离线状态 ----
        env_frame = ttk.LabelFrame(main, text="运行环境与离线状态")
        env_frame.pack(fill='x', pady=(0, 8))
        self.env_text = tk.Text(env_frame, height=6, font=('Consolas', 9),
                                state='disabled')
        self.env_text.pack(fill='x', padx=8, pady=8)
        ttk.Button(env_frame, text="刷新状态", command=self._refresh_env,
                   style='Accent.TButton').pack(anchor='e', padx=8, pady=(0, 8))

        # ---- 2. 模型/规则热更新 ----
        hu_frame = ttk.LabelFrame(main, text="模型/规则热更新")
        hu_frame.pack(fill='x', pady=(0, 8))
        path_row = ttk.Frame(hu_frame)
        path_row.pack(fill='x', padx=8, pady=(8, 2))
        ttk.Label(path_row, text="热更新目录:").pack(side='left')
        self.hu_path_var = tk.StringVar(value=DEFAULT_HOTUPDATE_DIR)
        ttk.Label(path_row, textvariable=self.hu_path_var,
                  font=('Consolas', 9), foreground='#5a6c7d').pack(
            side='left', padx=6)
        self.watch_status_label = ttk.Label(hu_frame, text="未启动",
                                            foreground='#e74c3c')
        self.watch_status_label.pack(anchor='w', padx=8)
        self._watcher_controls = ttk.Frame(hu_frame)
        self._watcher_controls.pack(fill='x', padx=8, pady=4)
        self.btn_start = ttk.Button(self._watcher_controls, text="启动监控",
                                    command=self._start_watcher)
        self.btn_start.pack(side='left', padx=3)
        self.btn_stop = ttk.Button(self._watcher_controls, text="停止监控",
                                   command=self._stop_watcher,
                                   state='disabled')
        self.btn_stop.pack(side='left', padx=3)
        ttk.Label(hu_frame,
                  text="提示：将新的 .joblib/.pkl(ML)、.pth/.pt(GNN)、.json(规则) "
                       "放入该目录即可热加载，无需重装。",
                  font=('Microsoft YaHei', 8), foreground='#5a6c7d').pack(
            anchor='w', padx=8, pady=(2, 6))

        # 热更新历史
        hist_frame = ttk.Frame(hu_frame)
        hist_frame.pack(fill='x', padx=8, pady=(0, 8))
        ttk.Label(hist_frame, text="热更新记录:").pack(anchor='w')
        self.hist_tree = ttk.Treeview(
            hist_frame, columns=('time', 'name', 'kind', 'result'),
            show='headings', height=4)
        for c, w, t in (('time', 150, '时间'), ('name', 130, '文件'),
                        ('kind', 60, '类型'), ('result', 260, '结果')):
            self.hist_tree.heading(c, text=t)
            self.hist_tree.column(c, width=w)
        self.hist_tree.pack(fill='x')

        # ---- 3. 部署打包 ----
        dep_frame = ttk.LabelFrame(main, text="部署打包")
        dep_frame.pack(fill='x')
        ttk.Label(dep_frame,
                  text="一键打包可分发的离线安装包（含源码、依赖锁文件、"
                       "Dockerfile、本地模型与配置）。",
                  font=('Microsoft YaHei', 9)).pack(anchor='w', padx=8, pady=(8, 4))
        row = ttk.Frame(dep_frame)
        row.pack(fill='x', padx=8, pady=(0, 8))
        ttk.Button(row, text="打包离线分发", command=self._do_pack,
                   style='Accent.TButton').pack(side='left', padx=3)
        self.pack_result_var = tk.StringVar(value="")
        ttk.Label(row, textvariable=self.pack_result_var,
                  font=('Consolas', 8), foreground='#2ecc71').pack(
            side='left', padx=8)

        # ---- 4. 关闭 ----
        ttk.Button(main, text="关闭", command=self.destroy).pack(
            anchor='e', pady=(8, 0))

    # ==================== 环境状态 ====================

    def _refresh_env(self):
        s = offline_status()
        lines = [
            f"联网状态: {'可联网' if s['online'] else '离线'} "
            f"(探测 {s['probe_host']})",
            f"本地模型缓存: {'就绪（%d 个模型）' % s['model_count'] if s['model_ready'] else '未就绪（可训练后保存）'}",
            f"规则配置: {'就绪' if s['rule_ready'] else '缺失'}",
            f"缓存目录: {s['data_dir']}",
        ]
        if not s['online']:
            lines.append("说明: 离线模式下规则始终可用，核心评估可正常运行。")
        self.env_text.config(state='normal')
        self.env_text.delete('1.0', 'end')
        self.env_text.insert('1.0', "\n".join(lines))
        self.env_text.config(state='disabled')

    # ==================== 热更新 ====================

    def _get_ml_loaders(self):
        """构建绑定到 MLRiskPredictor 的加载回调。"""
        predictor = getattr(self.app, 'ml_predictor', None) if self.app else None

        def ml_loader(path):
            if predictor is not None and hasattr(predictor, 'load_model'):
                return bool(predictor.load_model(path))
            return False

        def gnn_loader(path):
            if predictor is not None and hasattr(predictor, 'load_gnn_model'):
                return bool(predictor.load_gnn_model(path))
            return False

        return ml_loader, gnn_loader

    def _start_watcher(self):
        try:
            if self._watcher is not None and self._watcher.is_running():
                return
            ml_loader, gnn_loader = self._get_ml_loaders()
            self._watcher = HotUpdateWatcher(
                watch_dir=self.hu_path_var.get() or DEFAULT_HOTUPDATE_DIR,
                ml_loader=ml_loader,
                gnn_loader=gnn_loader,
                on_update=self._on_hot_update,
            )
            self._watcher.start()
            self._set_watcher_ui(True)
            self._refresh_watcher()
        except Exception as e:
            messagebox.showerror("错误", f"启动热更新监控失败: {e}")

    def _stop_watcher(self):
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None
        self._set_watcher_ui(False)
        self._refresh_watcher()

    def _set_watcher_ui(self, running: bool):
        self.watch_status_label.config(
            text="监控中" if running else "未启动",
            foreground='#2ecc71' if running else '#e74c3c')
        self.btn_start.config(state='disabled' if running else 'normal')
        self.btn_stop.config(state='normal' if running else 'disabled')

    def _refresh_watcher(self):
        running = self._watcher is not None and self._watcher.is_running()
        self._set_watcher_ui(running)
        self._render_history()

    def _on_hot_update(self, result):
        """热更新回调（后台线程），通过主线程刷新 UI。"""
        try:
            self.after(0, self._render_history)
        except (tk.TclError, RuntimeError):
            pass

    def _render_history(self):
        records = self._watcher.history() if self._watcher else []
        for item in self.hist_tree.get_children():
            self.hist_tree.delete(item)
        for r in reversed(records):
            self.hist_tree.insert('', 'end', values=(
                r.get('time', '-'), r.get('name', ''),
                r.get('kind') or '-',
                r.get('message', ''), ))

    # ==================== 打包 ====================

    def _do_pack(self):
        try:
            self.pack_result_var.set("打包中...")
            self.update_idletasks()
            out = pack_distribution()
            self.pack_result_var.set(f"完成: {out}")
            messagebox.showinfo("打包完成", f"离线分发包已生成:\n{out}")
        except Exception as e:
            self.pack_result_var.set("打包失败")
            messagebox.showerror("错误", f"打包失败: {e}")


__all__ = ['OpsPanel']
