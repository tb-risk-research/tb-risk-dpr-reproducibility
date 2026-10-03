#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 训练面板 - 生存分析（_SurvivalMixin）

包含：贝叶斯生存分析、竞争风险多结局生存分析、多结局预测展示
"""

from .._shared import *


class _SurvivalMixin:
    """生存分析相关方法"""

    # ==================== 生存分析 ====================

    def _train_survival_gui(self):
        """训练贝叶斯生存分析模型（GUI对话框）"""
        if not SURVIVAL_BAYESIAN_AVAILABLE or self.ml_predictor is None or not self.ml_predictor._survival_ready:
            messagebox.showwarning("不可用", "生存分析模块不可用（缺少torch库）")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("生存分析模型训练")
        dlg.geometry("450x400")
        dlg.transient(self.root)
        dlg.grab_set()

        frm = ttk.Frame(dlg, padding=15)
        frm.pack(fill='both', expand=True)

        ttk.Label(frm, text="贝叶斯生存分析 + DeepSurv 训练",
                  font=('Arial', 11, 'bold')).pack(pady=(0, 10))

        ttk.Label(frm, text="合成样本数:", font=('Arial', 9)).pack(anchor='w')
        n_var = tk.IntVar(value=500)
        ttk.Scale(frm, variable=n_var, from_=100, to=2000).pack(fill='x', pady=2)
        ttk.Label(frm, textvariable=n_var, font=('Arial', 8)).pack()

        ttk.Label(frm, text="训练轮数:", font=('Arial', 9)).pack(anchor='w', pady=(8, 0))
        ep_var = tk.IntVar(value=30)
        ttk.Scale(frm, variable=ep_var, from_=5, to=100).pack(fill='x', pady=2)
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
            epochs = ep_var.get()

            def _train_thread():
                try:
                    _append_log("生成合成生存数据...")
                    sp = self.ml_predictor.survival_predictor
                    X, times, events, causes = sp.generate_synthetic_survival_data(
                        n_samples=n_samples, random_state=42)

                    _append_log(f"训练DeepSurv: {epochs}轮...")
                    result = sp.fit(X, times, events, epochs=epochs, batch_size=64,
                                  learning_rate=0.001, log_callback=_append_log)

                    _append_log(f"训练完成! loss={result.get('final_loss', 'N/A')}")
                    self._survival_trained = True
                except Exception as e:
                    _append_log(f"训练失败: {e}")
                finally:
                    self.root.after(0, lambda: [
                        w.config(state='normal') for w in frm.winfo_children()
                        if isinstance(w, ttk.Button)
                    ])

            self._start_training_thread(_train_thread)

        ttk.Button(frm, text="开始训练", command=_start).pack(pady=10)
        ttk.Label(frm, text="文献: Lee et al. (AAAI 2018), Gal & Ghahramani (ICML 2016)",
                  font=('Arial', 7)).pack(side='bottom')

    def _competing_risk_training_gui(self):
        """竞争风险多结局生存分析训练对话框"""
        if not MULTISTATE_AVAILABLE or self.ml_predictor is None or not self.ml_predictor._multistate_ready:
            messagebox.showwarning("不可用", "竞争风险多结局模块不可用（缺少torch库）")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("竞争风险多结局生存分析训练")
        dlg.geometry("520x520")
        dlg.transient(self.root)
        dlg.grab_set()

        frm = ttk.Frame(dlg, padding=15)
        frm.pack(fill='both', expand=True)

        ttk.Label(frm, text="竞争风险多结局 DeepHit 训练 (Fine-Gray 损失)",
                  font=('Arial', 11, 'bold')).pack(pady=(0, 10))

        ttk.Label(frm, text="四种结局: 潜伏持续 | 活动性发病 | 免疫清除 | 失访",
                  font=('Arial', 8, 'italic')).pack()

        ttk.Label(frm, text="合成样本数:", font=('Arial', 9)).pack(anchor='w', pady=(8, 0))
        n_var = tk.IntVar(value=500)
        ttk.Scale(frm, variable=n_var, from_=100, to=2000).pack(fill='x', pady=2)
        ttk.Label(frm, textvariable=n_var, font=('Arial', 8)).pack()

        ttk.Label(frm, text="训练轮数:", font=('Arial', 9)).pack(anchor='w', pady=(8, 0))
        ep_var = tk.IntVar(value=50)
        ttk.Scale(frm, variable=ep_var, from_=10, to=200).pack(fill='x', pady=2)
        ttk.Label(frm, textvariable=ep_var, font=('Arial', 8)).pack()

        log_text = self._register_text_widget(tk.Text(frm, height=12, font=('Arial', 8), wrap='word'))
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
            epochs = ep_var.get()

            def _train_thread():
                try:
                    _append_log("开始竞争风险多结局训练...")
                    _append_log(f"参数: n={n_samples}, epochs={epochs}")

                    result = self.ml_predictor.train_survival_multistate(
                        n_samples=n_samples, epochs=epochs,
                        batch_size=64, learning_rate=0.001,
                        log_callback=_append_log, random_state=42)

                    if result.get('status') == 'trained':
                        _append_log(f"\n训练完成！最终损失: {result['final_loss']:.4f}")
                        _append_log(f"事件数: {result['n_events']}")
                        self._multistate_trained = True

                        eval_metrics = result.get('eval_metrics', {})
                        _append_log("\n=== 各结局区分度 (C-index) ===")
                        for label, m in eval_metrics.items():
                            _append_log(f"  {label}: {m.get('c_index', 0):.4f}")

                        _append_log("\n使用 predict_multistate() 进行个体预测。")
                    else:
                        _append_log(f"训练失败: {result.get('status', 'unknown')}")
                except Exception as e:
                    _append_log(f"训练异常: {e}")
                finally:
                    self.root.after(0, lambda: [
                        w.config(state='normal') for w in frm.winfo_children()
                        if isinstance(w, ttk.Button)
                    ])

            self._start_training_thread(_train_thread)

        ttk.Button(frm, text="开始训练", command=_start).pack(pady=10)
        ttk.Label(frm, text="文献: Lee et al. (AAAI 2018), Fine & Gray (JASA 1999), "
                  "Houben & Dodd (PLoS Med 2016)",
                  font=('Arial', 7)).pack(side='bottom')

    def _show_multistate_prediction_gui(self):
        """多结局预测结果显示对话框"""
        if not MULTISTATE_AVAILABLE or self.ml_predictor is None:
            messagebox.showwarning("不可用", "竞争风险模块不可用")
            return

        msp = self.ml_predictor.multi_state_predictor
        if msp is None or not msp.is_trained:
            messagebox.showwarning("未训练", "请先在ML标签页训练竞争风险模型")
            return

        contacts = []
        for member in self.family_members:
            contacts.append((f"家庭-{member.get('name', '未知')}",
                           self.ml_predictor._contact_features(member, 'family')))
        for contact in self.social_contacts:
            contacts.append((f"社会-{contact.get('name', '未知')}",
                           self.ml_predictor._contact_features(contact, 'social')))

        if not contacts:
            messagebox.showinfo("无数据", "请先添加接触者")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("竞争风险多结局预测")
        dlg.geometry("700x600")
        dlg.transient(self.root)
        dlg.grab_set()

        frm = ttk.Frame(dlg, padding=15)
        frm.pack(fill='both', expand=True)

        ttk.Label(frm, text="竞争风险多结局累积发生率预测",
                  font=('Arial', 11, 'bold')).pack(pady=(0, 5))

        ttk.Label(frm, text="选择接触者查看详细预测:",
                  font=('Arial', 9)).pack(anchor='w')

        sel_var = tk.StringVar(value=contacts[0][0] if contacts else "")
        sel_combo = ttk.Combobox(frm, textvariable=sel_var,
                                  values=[c[0] for c in contacts],
                                  state='readonly', width=30)
        sel_combo.pack(pady=5)

        result_text = self._register_text_widget(tk.Text(frm, height=20, font=('Arial', 9), wrap='word'))
        result_text.pack(fill='both', expand=True, pady=10)

        def _update_display(_=None):
            result_text.delete('1.0', 'end')
            selected = sel_var.get()
            features = None
            for name, feats in contacts:
                if name == selected:
                    features = feats
                    break

            if features is None:
                result_text.insert('end', "未找到选中接触者的特征数据\n")
                return

            pred = msp.predict_multistate(features, include_uncertainty=True)

            result_text.insert('end', f"接触者: {selected}\n\n", 'title')

            result_text.insert('end', "=== 各结局52周累积发生率 ===\n\n", 'heading')

            cause_probs = pred.get('cause_probs', {})
            for label, info in cause_probs.items():
                prob = info.get('final_prob', 0)
                bar = '█' * int(prob * 50)
                result_text.insert('end',
                    f"  {label}: {prob*100:.1f}% {bar}\n")

                if info.get('ci_lower') is not None and info.get('ci_upper') is not None:
                    ci_low = info['ci_lower'][-1]
                    ci_high = info['ci_upper'][-1]
                    result_text.insert('end',
                        f"    90%CI: [{ci_low*100:.1f}% - {ci_high*100:.1f}%]\n")

            result_text.insert('end', "\n=== 临床建议 ===\n", 'heading')

            rec = msp.get_clinical_recommendation(pred)
            result_text.insert('end', f"  风险等级: {rec['risk_level']}\n")
            result_text.insert('end', f"  建议: {rec['recommendation']}\n")
            result_text.insert('end', f"  置信度: {rec['confidence']}\n")
            result_text.insert('end', f"  活动性结核1年概率: {rec['active_tb_1y_prob']*100:.1f}%\n")
            result_text.insert('end', "\n  具体行动:\n")
            for action in rec['actions']:
                result_text.insert('end', f"    • {action}\n")

            if pred.get('is_fallback'):
                result_text.insert('end', "\n[注意: 当前为经验估计，非模型预测]\n", 'warning')

            result_text.insert('end',
                f"\n主导结局: {pred.get('dominant_label', '未知')} | "
                f"高置信度: {'是' if pred.get('high_confidence') else '否'}",
                'info')

            result_text.tag_configure('title', font=('Arial', 10, 'bold'))
            result_text.tag_configure('heading', font=('Arial', 9, 'bold'))
            result_text.tag_configure('warning', font=('Arial', 8, 'italic'),
                                       foreground='#CC6600')
            result_text.tag_configure('info', font=('Arial', 8),
                                       foreground='#2980B9')

        sel_combo.bind('<<ComboboxSelected>>', _update_display)
        if contacts:
            _update_display()
