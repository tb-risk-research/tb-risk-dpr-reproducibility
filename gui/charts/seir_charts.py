#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 图表 - SEIR 传播动力学图（SEIRChartMixin）

包含：确定性SEIR曲线、随机SEIR曲线（含后验预测区间）、贝叶斯MCMC参数推断

文献支撑：
- Dye et al. (2005) 结核病SEIR模型
- Rosato et al. (2022); Chakraborty et al. (2025) 贝叶斯MCMC推断
"""

from ._shared import *


class SEIRChartMixin:
    """SEIR 传播动力学图表与贝叶斯推断方法"""

    # 优先级八：SEIR 参数滑块范围与默认值
    _SEIR_PARAM_RANGES = {
        'beta': (0.05, 2.0, 0.01),   # 有效接触率 (min, max, step)
        'sigma': (0.05, 1.0, 0.01),  # 潜伏期转化率
        'gamma': (0.01, 0.5, 0.005), # 恢复率
    }

    def _create_seir_param_panel(self, parent):
        """优先级八：创建可折叠的 SEIR 参数控制面板

        包含 beta（有效接触率）、sigma（潜伏期转化率）、gamma（恢复率）三个滑块，
        以及"重置为默认值"和"从后验采样"按钮。滑块拖动时通过防抖机制（200ms 延迟）
        触发 SEIR 曲线重绘。

        Args:
            parent: 父容器（tkinter Frame）
        """
        if not TKINTER_AVAILABLE or parent is None:
            return

        # 初始化用户参数存储（None 表示使用代码计算的默认值）
        if not hasattr(self, '_seir_user_params'):
            self._seir_user_params = {'beta': None, 'sigma': None, 'gamma': None}
        if not hasattr(self, '_seir_param_after_id'):
            self._seir_param_after_id = None  # 防抖 after_id

        # 可折叠标题栏
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill='x')

        self._seir_param_collapsed = tk.BooleanVar(value=False)
        toggle_btn = ttk.Checkbutton(
            header_frame, text="SEIR 参数调整（点击展开/折叠）",
            variable=self._seir_param_collapsed,
            command=self._toggle_seir_param_panel
        )
        toggle_btn.pack(side='left', padx=5)

        # 参数控制区域（可折叠）
        self._seir_param_body = ttk.Frame(parent)
        self._seir_param_body.pack(fill='x', padx=5, pady=2)

        # 三个滑块：beta / sigma / gamma
        self._seir_param_vars = {}
        self._seir_param_labels = {}
        self._seir_param_sliders = {}

        param_info = [
            ('beta', 'β 有效接触率', 0.3, '%.3f'),
            ('sigma', 'σ 潜伏期转化率', 0.25, '%.3f'),
            ('gamma', 'γ 恢复率', 0.038, '%.3f'),
        ]

        for param_name, label_text, default_val, fmt in param_info:
            row_frame = ttk.Frame(self._seir_param_body)
            row_frame.pack(fill='x', pady=1)

            ttk.Label(row_frame, text=f"{label_text}:", width=18).pack(side='left')

            lo, hi, step = self._SEIR_PARAM_RANGES[param_name]
            var = tk.DoubleVar(value=default_val)
            self._seir_param_vars[param_name] = var

            # 数值标签
            val_label = ttk.Label(row_frame, text=fmt % default_val, width=8)
            val_label.pack(side='right', padx=5)
            self._seir_param_labels[param_name] = val_label

            # 滑块
            # ttk.Scale 不支持 step，用 from_/to/resolution；改用 tk.Scale
            slider = tk.Scale(
                row_frame, from_=lo, to=hi, resolution=step,
                orient='horizontal', variable=var,
                command=lambda v, n=param_name, f=fmt: self._on_seir_param_changed(n, f, v),
                bg=self.COLORS.get('bg_card', '#ffffff'),
                fg=self.COLORS.get('text', '#2c3e50'),
                troughcolor=self.COLORS.get('bg_light', '#f0f4f8'),
                activebackground=self.COLORS.get('accent', '#3498db'),
                highlightthickness=0,
            )
            slider.pack(side='left', fill='x', expand=True, padx=5)
            self._seir_param_sliders[param_name] = slider

        # 按钮区域
        btn_frame = ttk.Frame(self._seir_param_body)
        btn_frame.pack(fill='x', pady=3)

        ttk.Button(btn_frame, text="重置为默认值",
                   command=self._reset_seir_params).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="从后验采样",
                   command=self._sample_seir_from_posterior).pack(side='left', padx=5)

        # R0 显示标签（使用主题色而非硬编码蓝色）
        self._seir_r0_label = ttk.Label(btn_frame, text="R0 = -",
                                         foreground=self.COLORS.get('accent', '#3498db'))
        self._seir_r0_label.pack(side='right', padx=10)

        # 初始折叠状态
        self._toggle_seir_param_panel()

    def _toggle_seir_param_panel(self):
        """优先级八：展开/折叠 SEIR 参数面板"""
        if not hasattr(self, '_seir_param_body'):
            return
        if self._seir_param_collapsed.get():
            self._seir_param_body.pack_forget()
        else:
            self._seir_param_body.pack(fill='x', padx=5, pady=2)

    def _on_seir_param_changed(self, param_name, fmt, value):
        """优先级八：滑块值变化回调（带 200ms 防抖）

        Args:
            param_name: 参数名（'beta' / 'sigma' / 'gamma'）
            fmt: 数值格式化字符串
            value: 滑块当前值（字符串）
        """
        try:
            val = float(value)
        except (ValueError, TypeError):
            return

        # 更新数值标签
        if param_name in self._seir_param_labels:
            self._seir_param_labels[param_name].config(text=fmt % val)

        # 记录用户参数（None 表示使用代码计算值的逻辑被覆盖）
        self._seir_user_params[param_name] = val

        # 更新 R0 显示
        self._update_seir_r0_label()

        # 防抖：200ms 后触发重绘
        if self._seir_param_after_id is not None:
            try:
                self.root.after_cancel(self._seir_param_after_id)
            except (tk.TclError, RuntimeError, AttributeError):
                pass
        try:
            self._seir_param_after_id = self.root.after(200, self._redraw_seir_with_user_params)
        except (tk.TclError, RuntimeError, AttributeError):
            pass

    def _update_seir_r0_label(self):
        """优先级八：更新 R0 显示标签"""
        if not hasattr(self, '_seir_r0_label'):
            return
        try:
            beta = self._seir_user_params.get('beta')
            gamma = self._seir_user_params.get('gamma')
            if beta is not None and gamma is not None and gamma > 0:
                r0 = beta / gamma
                self._seir_r0_label.config(text=f"R0 = {r0:.2f}")
            else:
                self._seir_r0_label.config(text="R0 = -（使用默认参数）")
        except Exception:
            pass

    def _redraw_seir_with_user_params(self):
        """优先级八：使用用户调整的参数重绘 SEIR 曲线"""
        try:
            self._draw_seir_curve()
        except Exception as e:
            LOGGER.debug("SEIR 参数重绘失败: %s", e)
        finally:
            self._seir_param_after_id = None

    def _reset_seir_params(self):
        """优先级八：重置参数为默认值（清除用户覆盖）"""
        # 清除用户覆盖（None 表示使用代码计算值）
        self._seir_user_params = {'beta': None, 'sigma': None, 'gamma': None}

        # 重置滑块到默认值
        defaults = {'beta': 0.3, 'sigma': 0.25, 'gamma': 0.038}
        for param_name, default_val in defaults.items():
            if param_name in self._seir_param_vars:
                self._seir_param_vars[param_name].set(default_val)
            if param_name in self._seir_param_labels:
                self._seir_param_labels[param_name].config(text='%.3f' % default_val)

        # 更新 R0 标签
        if hasattr(self, '_seir_r0_label'):
            self._seir_r0_label.config(text="R0 = -（使用默认参数）")

        # 重绘
        self._redraw_seir_with_user_params()

    def _sample_seir_from_posterior(self):
        """优先级八：从 MCMC 后验分布随机采样参数组合"""
        try:
            if not hasattr(self, 'seir_inference') or self.seir_inference is None:
                messagebox.showinfo("提示", "贝叶斯后验分布不可用，请先运行贝叶斯 MCMC 推断。")
                return

            posterior = getattr(self.seir_inference, 'posterior_samples', None)
            if not posterior:
                messagebox.showinfo("提示", "后验采样为空，请先运行贝叶斯 MCMC 推断。")
                return

            # 从后验采样随机抽取一组参数
            import random as _random
            beta_samples = posterior.get('beta', [0.3])
            sigma_samples = posterior.get('rho_fast') or posterior.get('sigma', [0.25])
            gamma_samples = posterior.get('gamma', [0.038])

            beta_val = float(_random.choice(beta_samples))
            sigma_val = float(_random.choice(sigma_samples))
            gamma_val = float(_random.choice(gamma_samples))

            # 裁剪到滑块范围
            for param_name, val in [('beta', beta_val), ('sigma', sigma_val), ('gamma', gamma_val)]:
                lo, hi, _ = self._SEIR_PARAM_RANGES[param_name]
                val = max(lo, min(hi, val))
                self._seir_user_params[param_name] = val
                if param_name in self._seir_param_vars:
                    self._seir_param_vars[param_name].set(val)
                if param_name in self._seir_param_labels:
                    self._seir_param_labels[param_name].config(text='%.3f' % val)

            # 更新 R0 标签
            self._update_seir_r0_label()

            # 重绘
            self._redraw_seir_with_user_params()
        except Exception as e:
            LOGGER.warning("从后验采样失败: %s", e, exc_info=True)
            messagebox.showerror("错误", f"从后验采样失败: {e}")

    def _draw_seir_curve(self):
        """绘制SEIR传播动力学曲线（使用scipy.integrate.odeint精确求解）

        文献支撑：
        - Dye et al. (2005) 结核病SEIR模型
        - 异质性参数化：基于患者特征和接触者风险

        优先级八：支持用户通过滑块面板调整 beta/sigma/gamma 参数。
        """
        if not (MATPLOTLIB_AVAILABLE and NUMPY_AVAILABLE) or not hasattr(self, 'seir_figure'):
            return

        try:
            self.seir_figure.clear()
            ax = self.seir_figure.add_subplot(111)

            # 模拟52周
            weeks = np.linspace(0, 52, 100)  # 使用更精细的时间步长

            # 收集所有接触者的发病概率作为权重
            potential_patients = self.results.get('potential_patients', {})
            contact_weights = []

            for patient in potential_patients.get('family', []):
                if 'disease_probability' in patient:
                    contact_weights.append(patient['disease_probability'] / 100)

            for patient in potential_patients.get('social', []):
                if 'disease_probability' in patient:
                    contact_weights.append(patient['disease_probability'] / 100)

            if len(contact_weights) == 0:
                ax.text(0.5, 0.5, '无接触者数据\n无法绘制SEIR曲线',
                       ha='center', va='center', fontsize=14, color='gray',
                       transform=ax.transAxes)
                ax.set_title('SEIR传播动力学曲线', fontsize=12)
                self.seir_canvas.draw()
                return

            total_contacts = max(len(contact_weights), 1)
            avg_weight = np.mean(contact_weights) if contact_weights else 0.3

            # 获取患者基本信息（提前获取，避免作用域问题）
            patient_basic_info = (self.patient_info or {}).get('basic_info', {})
            treatment_status = patient_basic_info.get('treatment', 2)
            sputum_smear = patient_basic_info.get('sputum_smear', 1)
            has_cavity = patient_basic_info.get('has_cavity', 1)

            # 初始状态：根据接触者风险分配初始状态
            s_frac = 0.5 * (1 - avg_weight) + 0.1  # 易感比例
            e_frac = avg_weight * 0.3  # 暴露比例
            i_frac = avg_weight * 0.15  # 感染比例（基线）
            r_frac = 1.0 - s_frac - e_frac - i_frac  # 康复/免疫比例

            # 考虑患者治疗阶段对初始I的影响（WHO 2024指南）
            # 治疗阶段衰减因子：治疗中患者初始感染人数减少
            treatment_infectivity_factor = 1.0
            if treatment_status == 1:  # 正在治疗
                # 假设已治疗2周，传染性下降40%
                treatment_infectivity_factor = 0.6
            elif treatment_status == 2:  # 未治疗
                treatment_infectivity_factor = 1.0

            S0 = total_contacts * s_frac
            E0 = total_contacts * e_frac
            I0 = total_contacts * i_frac * treatment_infectivity_factor
            R0 = total_contacts * r_frac

            initial_state = [S0, E0, I0, R0]

            # SEIR模型参数：根据患者类型动态调整
            infection_val = self.results.get('base_infection_probability', 30)
            base_infection_prob = (30 if infection_val is None else infection_val) / 100

            smear_positive = 1 if sputum_smear == 2 else 0
            has_cavity_flag = 1 if has_cavity == 2 else 0

            beta_multiplier = 1.0
            if smear_positive == 1:
                beta_multiplier *= 2.0
            if has_cavity_flag == 1:
                beta_multiplier *= 1.5
            if treatment_status == 2:  # 未治疗
                beta_multiplier *= 1.2
            elif treatment_status == 1:  # 正在治疗
                beta_multiplier *= 0.5

            beta = 0.3 * base_infection_prob * beta_multiplier
            sigma = 1.0 / 4  # 暴露→感染率（潜伏期4周）
            gamma = 1.0 / 26  # 感染→康复率（恢复周期26周）

            # 优先级八：用户通过滑块调整的参数覆盖代码计算值
            user_params = getattr(self, '_seir_user_params', {})
            if user_params.get('beta') is not None:
                beta = float(user_params['beta'])
            if user_params.get('sigma') is not None:
                sigma = float(user_params['sigma'])
            if user_params.get('gamma') is not None:
                gamma = float(user_params['gamma'])

            if SCIPY_AVAILABLE:
                # 使用scipy.integrate.odeint精确求解SEIR微分方程
                def seir_derivatives(y, t, beta, sigma, gamma, N):
                    S, E, I, R = y
                    dS = -beta * S * I / N
                    dE = beta * S * I / N - sigma * E
                    dI = sigma * E - gamma * I
                    dR = gamma * I
                    return [dS, dE, dI, dR]

                solution = odeint(seir_derivatives, initial_state, weeks,
                                args=(beta, sigma, gamma, total_contacts))
                S_sol = solution[:, 0]
                E_sol = solution[:, 1]
                I_sol = solution[:, 2]
                R_sol = solution[:, 3]

                solver_label = '（odeint精确求解）'
            else:
                # 降级：使用欧拉法近似（当scipy不可用时）
                S_sol = np.zeros_like(weeks)
                E_sol = np.zeros_like(weeks)
                I_sol = np.zeros_like(weeks)
                R_sol = np.zeros_like(weeks)
                S_sol[0], E_sol[0], I_sol[0], R_sol[0] = initial_state

                dt = weeks[1] - weeks[0]
                for i in range(1, len(weeks)):
                    dS = -beta * S_sol[i-1] * I_sol[i-1] / total_contacts
                    dE = beta * S_sol[i-1] * I_sol[i-1] / total_contacts - sigma * E_sol[i-1]
                    dI = sigma * E_sol[i-1] - gamma * I_sol[i-1]
                    dR = gamma * I_sol[i-1]

                    S_sol[i] = max(S_sol[i-1] + dS * dt, 0)
                    E_sol[i] = max(E_sol[i-1] + dE * dt, 0)
                    I_sol[i] = max(I_sol[i-1] + dI * dt, 0)
                    R_sol[i] = max(R_sol[i-1] + dR * dt, 0)

                solver_label = '（欧拉法近似）'

            # 绘制曲线
            ax.plot(weeks, S_sol, label='易感者(S)', color='#3498db', linewidth=2)
            ax.plot(weeks, E_sol, label='暴露者(E)', color='#f39c12', linewidth=2)
            ax.plot(weeks, I_sol, label='感染者(I)', color='#e74c3c', linewidth=2)
            ax.plot(weeks, R_sol, label='康复者(R)', color='#27ae60', linewidth=2)

            # 添加患者特征标注
            features = []
            if smear_positive == 1:
                features.append('涂阳')
            if has_cavity_flag == 1:
                features.append('有空洞')
            if features:
                feature_text = f'特征: {",".join(features)}, β系数={beta:.2f}'
                ax.text(0.02, 0.98, feature_text, transform=ax.transAxes,
                        ha='left', va='top', fontsize=9, bbox=dict(boxstyle='round', alpha=0.1))

            # 添加求解器信息
            ax.text(0.98, 0.02, solver_label, transform=ax.transAxes,
                   ha='right', va='bottom', fontsize=8, style='italic', color='gray')

            # 优先级八：同步更新 R0 标注（参数变化时实时反映）
            r0_val = beta / max(gamma, 0.001)
            user_params = getattr(self, '_seir_user_params', {})
            param_source = '[用户调整]' if any(v is not None for v in user_params.values()) else '[默认]'
            r0_text = f'$R_0$={r0_val:.2f} {param_source}'
            ax.text(0.98, 0.98, r0_text, transform=ax.transAxes, ha='right', va='top',
                   fontsize=10, bbox=dict(boxstyle='round', alpha=0.1))
            # 同步更新参数面板的 R0 标签
            if hasattr(self, '_seir_r0_label'):
                try:
                    self._seir_r0_label.config(text=f"R0 = {r0_val:.2f} {param_source}")
                except Exception:
                    pass

            ax.set_xlabel('时间（周）', fontsize=12)
            ax.set_ylabel('人数', fontsize=12)
            ax.set_title(f'SEIR传播动力学曲线 [基于接触者风险权重]{solver_label}',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

            # Section VI: 保存数据用于悬停交互（显示具体时间点各仓室数值）
            self._chart_bars['seir_curve'] = {
                'weeks': weeks, 'S': S_sol, 'E': E_sol, 'I': I_sol, 'R': R_sol,
                'total': total_contacts,
            }
            self._safe_mpl_connect(
                getattr(self, 'seir_canvas', None), 'seir_curve',
                'motion_notify_event', self._on_seir_hover
            )

            self.seir_canvas.draw()
        except Exception as e:
            LOGGER.warning("绘制SEIR曲线时出错: %s", e, exc_info=True)

    def _on_seir_hover(self, event):
        """SEIR 曲线悬停回调：显示具体时间点的 S/E/I/R 各仓室数值

        通过鼠标 x 坐标找到最近的时间点，展示该时间点各仓室的人数。
        """
        try:
            if event is None or event.inaxes is None:
                self._hide_hover_annotation('seir_curve')
                self._redraw_chart_canvas(getattr(self, 'seir_canvas', None))
                return
            data = self._chart_bars.get('seir_curve')
            if not data:
                return
            weeks = data['weeks']
            # 找到最近的索引（data 坐标）
            x_data = event.xdata
            if x_data is None:
                self._hide_hover_annotation('seir_curve')
                self._redraw_chart_canvas(getattr(self, 'seir_canvas', None))
                return
            idx = int(np.clip(np.searchsorted(weeks, x_data), 0, len(weeks) - 1))
            t = float(weeks[idx])
            s = float(data['S'][idx])
            e = float(data['E'][idx])
            i = float(data['I'][idx])
            r = float(data['R'][idx])
            total = int(data.get('total', s + e + i + r))
            tooltip_text = (f'时间: {t:.1f} 周\n'
                            f'易感者(S): {s:.1f}\n'
                            f'暴露者(E): {e:.1f}\n'
                            f'感染者(I): {i:.1f}\n'
                            f'康复者(R): {r:.1f}\n'
                            f'总计: {total}')
            # 找一个合适的 y 位置：取该时间点四条曲线的中间值
            y_pos = (s + e + i + r) / 4.0
            self._show_hover_annotation(
                event.inaxes, t, y_pos, tooltip_text, 'seir_curve'
            )
            self._redraw_chart_canvas(getattr(self, 'seir_canvas', None))
        except Exception as e:
            LOGGER.debug("SEIR 悬停回调失败: %s", e)

    def _draw_stochastic_seir_curve(self, n_trajectories=50, ci_level=0.95):
        """绘制随机SEIR曲线（含后验预测区间置信带）

        文献：Rosato et al. (2022); Chakraborty et al. (2025)
        """
        if not (MATPLOTLIB_AVAILABLE and NUMPY_AVAILABLE) or not hasattr(self, 'seir_figure'):
            return

        try:
            self.seir_figure.clear()
            ax = self.seir_figure.add_subplot(111)

            results = self.results or {}
            potential_patients = results.get('potential_patients', {})
            contact_weights = []
            for patient in potential_patients.get('family', []):
                if 'disease_probability' in patient:
                    contact_weights.append(patient['disease_probability'] / 100)
            for patient in potential_patients.get('social', []):
                if 'disease_probability' in patient:
                    contact_weights.append(patient['disease_probability'] / 100)

            if not contact_weights:
                ax.text(0.5, 0.5, '无接触者数据', ha='center', va='center', fontsize=14, color='gray',
                       transform=ax.transAxes)
                ax.set_title('随机SEIR传播动力学曲线', fontsize=12)
                self.seir_canvas.draw()
                return

            total_contacts = max(len(contact_weights), 1)
            avg_weight = np.mean(contact_weights) if contact_weights else 0.3

            patient_bi = (self.patient_info or {}).get('basic_info', {})
            treatment_status = patient_bi.get('treatment', 2)
            sputum_smear = patient_bi.get('sputum_smear', 1)
            has_cavity = patient_bi.get('has_cavity', 1)

            s_frac = 0.5 * (1 - avg_weight) + 0.1
            e_frac = avg_weight * 0.3
            i_frac = avg_weight * 0.15
            r_frac = 1.0 - s_frac - e_frac - i_frac

            treatment_factor = 0.6 if treatment_status == 1 else 1.0

            initial_state = [
                total_contacts * s_frac,
                total_contacts * e_frac,
                total_contacts * i_frac * treatment_factor,
                total_contacts * r_frac,
            ]

            base_inf_prob = results.get('base_infection_probability', 30) / 100
            beta_mult = 1.0
            if sputum_smear == 2: beta_mult *= 2.0
            if has_cavity == 2: beta_mult *= 1.5
            if treatment_status == 2: beta_mult *= 1.2
            elif treatment_status == 1: beta_mult *= 0.5

            # 使用贝叶斯后验均值（如果可用），否则使用确定性值
            # v3.0: sigma → rho_fast（7-房室模型中快进展率替代旧潜伏率）
            if self.seir_inference.posterior_samples:
                beta = float(np.mean(self.seir_inference.posterior_samples.get('beta', [0.3])))
                # 优先使用新参数 rho_fast，回退到旧参数 sigma
                rho_fast_samples = self.seir_inference.posterior_samples.get('rho_fast')
                if rho_fast_samples is not None:
                    sigma = float(np.mean(rho_fast_samples))
                else:
                    sigma = float(np.mean(self.seir_inference.posterior_samples.get('sigma', [0.25])))
                gamma = float(np.mean(self.seir_inference.posterior_samples.get('gamma', [0.038])))
            else:
                beta = 0.3 * base_inf_prob * beta_mult
                sigma = 1.0 / 4
                gamma = 1.0 / 26

            t_span = (0, 52)
            dt = 0.5
            times = np.linspace(0, 52, int(52 / dt) + 1)

            all_I = self.stochastic_seir.simulate_multiple(
                initial_state, t_span, dt, beta, sigma, gamma, n_trajectories)

            I_mean = np.mean(all_I, axis=0)
            ci_low = np.percentile(all_I, (1 - ci_level) / 2 * 100, axis=0)
            ci_high = np.percentile(all_I, (1 + ci_level) / 2 * 100, axis=0)

            ax.plot(times, I_mean, color='#e74c3c', linewidth=2, label='感染者(I) 后验均值')
            ax.fill_between(times, ci_low, ci_high, alpha=0.25, color='#e74c3c',
                           label=f'{ci_level*100:.0f}% 后验预测区间')

            if SCIPY_AVAILABLE:
                def seir_ode(y, t, b, s, g, N):
                    S, E, I, R = y
                    dS = -b * S * I / N; dE = b * S * I / N - s * E
                    dI = s * E - g * I; dR = g * I
                    return [dS, dE, dI, dR]
                det_sol = odeint(seir_ode, initial_state, times, args=(beta, sigma, gamma, total_contacts))
                ax.plot(times, det_sol[:, 2], '--', color='#2c3e50', linewidth=1.5, alpha=0.7,
                       label='确定性SEIR (对比)')

            # R0标注
            r0_val = beta / max(gamma, 0.001)
            r0_text = f'$R_0$={r0_val:.2f}'
            if self.seir_inference.posterior_samples:
                r0_post = self.seir_inference.compute_r0_posterior()
                r0_text += f' [95% CI: {r0_post["ci_95_low"]:.2f}-{r0_post["ci_95_high"]:.2f}]'

            ax.text(0.98, 0.98, r0_text, transform=ax.transAxes, ha='right', va='top',
                   fontsize=10, bbox=dict(boxstyle='round', alpha=0.1))

            ax.set_xlabel('时间（周）', fontsize=12)
            ax.set_ylabel('感染者人数', fontsize=12)
            ax.set_title(f'随机SEIR传播动力学曲线（{n_trajectories}条随机轨迹）', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            self.seir_canvas.draw()
        except Exception as e:
            LOGGER.warning("绘制随机SEIR曲线时出错: %s", e, exc_info=True)

    def _run_bayesian_seir_inference(self, n_iterations=2000, burn_in=500, progress_callback=None):
        """从接触者追踪数据运行贝叶斯MCMC参数推断

        文献：Rosato et al. (2022); Chakraborty et al. (2025)
        """
        try:
            results = self.results or {}
            potential_patients = results.get('potential_patients', {})

            observed_I = []
            obs_times_list = []

            for i, patient in enumerate(potential_patients.get('family', [])):
                prob = patient.get('disease_probability', 0) / 100
                if prob > 0:
                    observed_I.append(prob * 5)
                    obs_times_list.append(float(i + 1))

            for i, patient in enumerate(potential_patients.get('social', [])):
                prob = patient.get('disease_probability', 0) / 100
                if prob > 0:
                    observed_I.append(prob * 3)
                    obs_times_list.append(float(i + 1))

            if not observed_I:
                return {'error': '无有效接触者数据'}

            observed_data = (np.array(observed_I), np.array(obs_times_list))

            total_contacts = max(
                len(potential_patients.get('family', [])) + len(potential_patients.get('social', [])),
                1
            )
            initial_state = [total_contacts * 0.7, total_contacts * 0.15,
                            total_contacts * 0.1, total_contacts * 0.05]

            t_span = (0, float(max(obs_times_list) + 5))
            dt = 0.5

            posterior = self.seir_inference.metropolis_hastings(
                observed_data, initial_state, t_span, dt,
                n_iterations=n_iterations, burn_in=burn_in
            )

            self.posterior_infectivity.update_from_mcmc(self.seir_inference)

            r0_result = self.seir_inference.compute_r0_posterior()

            result = {
                'convergence': self.seir_inference.convergence_diagnostics,
                'posterior_beta_mean': float(np.mean(posterior['beta'])),
                # v3.0: 优先 rho_fast，回退 sigma
                'posterior_sigma_mean': float(np.mean(
                    posterior.get('rho_fast', posterior.get('sigma', [0.0])))),
                'posterior_gamma_mean': float(np.mean(posterior['gamma'])),
                'r0': {k: v for k, v in r0_result.items() if k != 'samples'},
                'infectivity_factors': self.posterior_infectivity.factors,
            }

            return result
        except Exception as e:
            return {'error': str(e)}