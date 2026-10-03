#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 训练面板 - 强化学习干预规划（_RLMixin）

包含：RL干预策略规划对话框、BCQ离线预训练对话框
"""

from .._shared import *


class _RLMixin:
    """强化学习干预规划相关方法"""

    # ==================== RL 干预规划 ====================

    def _plan_rl_intervention_gui(self):
        """RL干预规划对话框"""
        if not CAUSAL_RL_AVAILABLE or self.integrator is None or not self.integrator.rl_ready:
            messagebox.showwarning("不可用", "RL干预规划模块不可用（缺少torch库）")
            return

        engine = self.integrator.rl_engine
        if not engine.is_offline_trained and not engine.is_online_finetuned:
            resp = messagebox.askyesno("未训练", "RL模型尚未训练，是否现在进行离线预训练？")
            if resp:
                self._pretrain_rl_gui()
                return
            else:
                return

        dlg = tk.Toplevel(self.root)
        dlg.title("RL干预策略规划")
        dlg.geometry("650x550")
        dlg.transient(self.root)
        dlg.grab_set()

        frm = ttk.Frame(dlg, padding=15)
        frm.pack(fill='both', expand=True)

        ttk.Label(frm, text="因果强化学习动态干预序列规划",
                  font=('Arial', 11, 'bold')).pack(pady=(0, 10))

        ttk.Label(frm, text="策略偏好:", font=('Arial', 9)).pack(anchor='w')
        pref_var = tk.StringVar(value='balanced')
        ttk.Radiobutton(frm, text="平衡成本-效果", variable=pref_var,
                       value='balanced').pack(anchor='w')
        ttk.Radiobutton(frm, text="成本优先", variable=pref_var,
                       value='cost_sensitive').pack(anchor='w')
        ttk.Radiobutton(frm, text="效果优先", variable=pref_var,
                       value='effect_sensitive').pack(anchor='w')

        ttk.Label(frm, text="规划步数:", font=('Arial', 9)).pack(anchor='w', pady=(8, 0))
        step_var = tk.IntVar(value=8)
        ttk.Scale(frm, variable=step_var, from_=3, to=20).pack(fill='x', pady=2)
        ttk.Label(frm, textvariable=step_var, font=('Arial', 8)).pack()

        result_text = self._register_text_widget(tk.Text(frm, font=('Arial', 9), wrap='word'))
        result_text.pack(fill='both', expand=True, pady=10)

        def _plan():
            result_text.delete('1.0', 'end')
            pref = pref_var.get()
            n_steps = step_var.get()

            result_text.insert('end', f"策略偏好: {pref}\n")
            result_text.insert('end', f"规划步数: {n_steps}\n\n")
            result_text.insert('end', "=" * 55 + "\n")
            result_text.insert('end', "推荐的动态干预序列:\n")
            result_text.insert('end', "=" * 55 + "\n\n")

            seq, metrics = engine.recommend_intervention_sequence(
                n_steps=n_steps, preference=pref)
            if not seq:
                result_text.insert('end', "无法生成干预序列（模型未训练？）\n")
                return

            for item in seq:
                causal_label = "✓因果路径" if item['causal_path_valid'] else "⚠路径异常"
                result_text.insert('end',
                    f"第{item['step']}周 | 动作: {item['name']}\n"
                    f"  说明: {item['description']}\n"
                    f"  因果效应: {item['causal_effect']:.3f} {causal_label}\n\n")

            result_text.insert('end', "-" * 55 + "\n")
            result_text.insert('end', "预期指标:\n")
            result_text.insert('end',
                f"  总成本: {metrics.get('total_cost', 0):.0f}元\n"
                f"  避免感染: {metrics.get('infections_averted', 0):.1f}例\n"
                f"  每例成本: {metrics.get('cost_per_infection_averted', 0):.0f}元\n")

        ttk.Button(frm, text="生成干预序列", command=_plan).pack(pady=10)
        ttk.Label(frm, text="文献: Gottesman et al. (2018), Puli et al. (NeurIPS 2022)",
                  font=('Arial', 7)).pack(side='bottom')

    def _pretrain_rl_gui(self):
        """RL离线预训练对话框"""
        if not CAUSAL_RL_AVAILABLE or self.integrator is None or not self.integrator.rl_ready:
            return

        engine = self.integrator.rl_engine

        dlg = tk.Toplevel(self.root)
        dlg.title("RL离线预训练")
        dlg.geometry("450x380")
        dlg.transient(self.root)
        dlg.grab_set()

        frm = ttk.Frame(dlg, padding=15)
        frm.pack(fill='both', expand=True)

        ttk.Label(frm, text="BCQ离线策略预训练",
                  font=('Arial', 11, 'bold')).pack(pady=(0, 10))

        ttk.Label(frm, text="离线样本数:", font=('Arial', 9)).pack(anchor='w')
        n_var = tk.IntVar(value=500)
        ttk.Scale(frm, variable=n_var, from_=100, to=2000).pack(fill='x', pady=2)
        ttk.Label(frm, textvariable=n_var, font=('Arial', 8)).pack()

        ttk.Label(frm, text="训练轮数:", font=('Arial', 9)).pack(anchor='w', pady=(8, 0))
        ep_var = tk.IntVar(value=15)
        ttk.Scale(frm, variable=ep_var, from_=5, to=50).pack(fill='x', pady=2)
        ttk.Label(frm, textvariable=ep_var, font=('Arial', 8)).pack()

        log_text = self._register_text_widget(tk.Text(frm, height=8, font=('Arial', 8)))
        log_text.pack(fill='both', expand=True, pady=10)

        def _safe_tk_append(msg):
            try:
                log_text.insert('end', msg + '\n')
                log_text.see('end')
            except (tk.TclError, RuntimeError):
                pass

        def _append_log(msg):
            if self.root is not None:
                self.root.after(0, lambda m=msg: _safe_tk_append(m))
            else:
                _safe_tk_append(msg)

        def _start():
            for w in frm.winfo_children():
                if isinstance(w, ttk.Button):
                    w.config(state='disabled')

            n_samples = n_var.get()
            n_epochs = ep_var.get()

            def _train_thread(n=n_samples, e=n_epochs):
                try:
                    result = engine.pretrain_offline(
                        n_samples=n, n_epochs=e,
                        batch_size=32, log_callback=_append_log)
                    _append_log(f"完成! status={result.get('status')}, "
                               f"final_loss={result.get('final_loss', 'N/A')}")
                    self.ml_predictor.rl_engine = engine
                except Exception as e:
                    _append_log(f"预训练失败: {e}")
                finally:
                    self.root.after(0, lambda: [
                        w.config(state='normal') for w in frm.winfo_children()
                        if isinstance(w, ttk.Button)
                    ])

            self._start_training_thread(_train_thread)

        ttk.Button(frm, text="开始预训练", command=_start).pack(pady=10)
        ttk.Label(frm, text="文献: Fujimoto et al. (BCQ, ICML 2019)",
                  font=('Arial', 7)).pack(side='bottom')
