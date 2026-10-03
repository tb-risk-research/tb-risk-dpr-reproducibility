#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 训练面板 - GNN 训练与干预优化（GNNMixin）

包含：异质GNN、多模态GNN、GNN训练GUI、反事实干预优化
"""

from ._shared import *


class GNNMixin:
    """GNN 模型训练与干预优化方法"""

    def _extract_gnn_training_metrics(self):
        """从 ml_predictor.model_performance['gnn'] 提取 GNN 验证指标

        Returns:
            dict: {AUROC, AUPRC, F1, Brier, ...}；无有效指标时返回 {}
        """
        if not self.ml_predictor:
            return {}
        perf = getattr(self.ml_predictor, 'model_performance', {}) or {}
        info = perf.get('gnn')
        if not isinstance(info, dict):
            return {}
        metrics = {}
        for metric_name in ('AUROC', 'AUPRC', 'Brier', 'Accuracy', 'F1'):
            value = info.get(metric_name)
            if isinstance(value, (int, float)):
                metrics[metric_name] = float(value)
        return metrics

    def _gnn_dataset_info(self, config):
        """构建 GNN 训练数据集信息（用于训练档案归因）"""
        source = getattr(self, '_validation_data_path', None)
        if source:
            import os as _os
            return {
                'source': _os.path.basename(str(source)),
                'source_path': str(source),
                'n_samples': int(getattr(config, 'n_samples', 0) or 0),
                'real_data': True,
            }
        return {
            'source': 'synthetic_graphs',
            'n_samples': int(getattr(config, 'n_samples', 0) or 0),
            'real_data': False,
        }


    # ==================== 异质时变GNN ====================

    def _init_hetero_gnn(self):
        """初始化异质时变GNN（使用因果图实例）"""
        try:
            if not _TORCH_GEOMETRIC_PIGNN_OK:
                return False

            causal_graph = None
            if hasattr(self, 'ml_predictor') and self.ml_predictor is not None:
                if hasattr(self.ml_predictor, 'causal_model') and self.ml_predictor.causal_model is not None:
                    causal_graph = self.ml_predictor.causal_model
                elif hasattr(self.ml_predictor, '_causal_constraint') and self.ml_predictor._causal_constraint is not None:
                    causal_graph = self.ml_predictor._causal_constraint

            # CAUSAL_RL_AVAILABLE 为 False 时跳过 CausalGraphConstraint 构造（类未实现）

            self.hetero_gnn = HeteroTimeVaryingGNN(
                node_feature_dim=30, edge_feature_dim=6, hidden_dim=64,
                num_edge_types=5, num_layers=1, num_time_steps=4,
                causal_graph=causal_graph, dropout=0.3
            )

            if hasattr(self.ml_predictor, 'seir_gnn') and self.ml_predictor.seir_gnn is not None:
                base_gnn = self.ml_predictor.seir_gnn
            else:
                base_gnn = None

            if base_gnn is not None and self.hetero_gnn is not None:
                self.three_phase_trainer = ThreePhaseTrainer(
                    base_gnn, self.hetero_gnn, causal_graph
                )

            return True
        except Exception as e:
            LOGGER.error("初始化异质GNN时出错: %s", e, exc_info=True)
            return False

    def _predict_with_hetero_gnn(self, contact_data, contact_type='family'):
        """使用异质时变GNN进行风险预测

        文献：HHAN (He et al., 2025); HeatGNN (Zheng et al., 2024)
        """
        if self.hetero_gnn is None:
            return None

        try:
            contact_data = contact_data or {}
            node_features = self._extract_node_features(contact_data)
            edge_indices = self._build_edge_indices(contact_data, contact_type)
            edge_attrs = self._build_edge_attrs(contact_data)

            prediction = self.hetero_gnn.forward_single(
                node_features, edge_indices, edge_attrs
            )

            return float(prediction.squeeze().item())
        except Exception as e:
            LOGGER.error("异质GNN预测时出错: %s", e, exc_info=True)
            return None

    def _build_edge_indices(self, contact_data, contact_type):
        """构建异质图边索引（简化实现）"""
        try:
            if not _TORCH_GEOMETRIC_PIGNN_OK:
                return [torch.zeros(2, 0, dtype=torch.long) for _ in range(5)]

            num_nodes = max(8, min(50, contact_data.get('num_contacts', 15)))
            edge_types_list = []

            for k in range(5):
                if k == 0:
                    num_edges = max(2, num_nodes // 3)
                    src = torch.randint(0, max(1, num_nodes // 2), (num_edges,))
                    dst = torch.randint(0, num_nodes, (num_edges,))
                else:
                    num_edges = max(1, num_nodes // 2)
                    src = torch.randint(0, num_nodes, (num_edges,))
                    dst = torch.randint(0, num_nodes, (num_edges,))

                edge_types_list.append(torch.stack([src, dst], dim=0))

            return edge_types_list
        except Exception:
            return [torch.zeros(2, 0, dtype=torch.long) for _ in range(5)]

    def _build_edge_attrs(self, contact_data):
        """构建异质图边特征（简化实现）"""
        try:
            if not _TORCH_GEOMETRIC_PIGNN_OK:
                return [None for _ in range(5)]

            contact_data = contact_data or {}
            edge_attrs_list = []
            num_nodes = max(8, min(50, contact_data.get('num_contacts', 15)))

            for k in range(5):
                if k == 0:
                    num_edges = max(2, num_nodes // 3)
                else:
                    num_edges = max(1, num_nodes // 2)
                attrs = torch.rand(num_edges, 6)

                if k == 0:
                    attrs[:, 0] = contact_data.get('freq_density', 15) / 30.0
                    attrs[:, 1] = contact_data.get('single_duration', 60) / 480.0
                elif k == 1:
                    attrs[:, 3] = contact_data.get('ventilation', 3) / 5.0
                    attrs[:, 2] = contact_data.get('contact_distance', 2) / 4.0

                edge_attrs_list.append(attrs)

            return edge_attrs_list
        except Exception:
            return [None for _ in range(5)]

    def _extract_node_features(self, contact_data):
        """提取节点特征（30维，v3.0）"""
        try:
            if not _TORCH_GEOMETRIC_PIGNN_OK:
                return torch.zeros(1, 30)

            num_nodes = max(8, min(50, contact_data.get('num_contacts', 15)))
            features = torch.zeros(num_nodes, 30)

            age = contact_data.get('age', 45)
            features[:, 0] = (age - 18) / 72.0

            features[:, 1] = 1.0 if contact_data.get('is_high_risk', 0) == 1 else 0.0
            features[:, 2] = 1.0 if contact_data.get('has_symptoms', 0) == 1 else 0.0
            features[:, 3] = 1.0 if contact_data.get('has_tb', 0) == 1 else 0.0
            features[:, 4] = 0.0 if contact_data.get('bcg_vaccine', 1) == 0 else 1.0
            features[:, 5] = contact_data.get('freq_density', 15) / 30.0
            features[:, 6] = contact_data.get('single_duration', 60) / 480.0
            features[:, 7] = contact_data.get('time_span', 12) / 52.0
            features[:, 8] = contact_data.get('ventilation', 3) / 5.0
            features[:, 9] = 1.0 if contact_data.get('past_illness', 0) == 1 else 0.0
            features[:, 10] = contact_data.get('contact_distance', 2) / 4.0
            features[:, 11] = 1.0 if contact_data.get('idu_status', '否') == '是' else 0.0

            return features
        except Exception:
            return torch.zeros(1, 30)

    # ==================== 反事实干预优化 ====================

    def _init_intervention_optimizer(self):
        """初始化方向三：个体化动态干预序列优化模块"""
        try:
            if not PYTORCH_AVAILABLE:
                self._intervention_optimizer_ready = False
                return False

            risk_predictor = None
            if hasattr(self, 'ml_predictor') and self.ml_predictor is not None:
                if hasattr(self.ml_predictor, 'seir_gnn') and \
                   self.ml_predictor.seir_gnn is not None:
                    risk_predictor = self.ml_predictor.seir_gnn

            if risk_predictor is not None and PYTORCH_AVAILABLE:
                self.counterfactual_optimizer = \
                    DifferentiableCounterfactualOptimizer(risk_predictor)
            else:
                self.counterfactual_optimizer = None

            causal_graph = None
            if CAUSAL_RL_AVAILABLE:
                try:
                    if hasattr(self, 'ml_predictor') and self.ml_predictor is not None:
                        if hasattr(self.ml_predictor, 'causal_model') and \
                           self.ml_predictor.causal_model is not None:
                            causal_graph = self.ml_predictor.causal_model
                        elif hasattr(self.ml_predictor, '_causal_constraint') and \
                             self.ml_predictor._causal_constraint is not None:
                            causal_graph = self.ml_predictor._causal_constraint
                    # CausalGraphConstraint 类未实现，此处不再构造
                except Exception:
                    causal_graph = None

            if PYTORCH_AVAILABLE and CAUSAL_RL_AVAILABLE:
                self.preference_policy = PreferenceConditionedPolicy(
                    state_dim=TBInterventionEnv.STATE_DIM,
                    n_actions=TBInterventionEnv.N_ACTIONS)
                if causal_graph is not None:
                    self.preference_policy.set_causal_constraint(causal_graph)
            else:
                self.preference_policy = None

            env_instance = None
            rl_engine_instance = None
            if CAUSAL_RL_AVAILABLE:
                try:
                    env_instance = TBInterventionEnv(n_nodes=30, horizon=26)
                    rl_engine_instance = RLInterventionEngine(
                        n_nodes=30, horizon=26)
                except Exception:
                    env_instance = None
                    rl_engine_instance = None

            self.dynamic_planner = DynamicInterventionPlanner(
                env=env_instance,
                bcq_agent=None,
                causal_graph=causal_graph,
                preference_policy=self.preference_policy,
                rl_engine=rl_engine_instance)

            self.intervention_validator = InterventionValidator(
                causal_graph=causal_graph,
                env_class=TBInterventionEnv if CAUSAL_RL_AVAILABLE else None)

            self._intervention_optimizer_ready = True
            return True
        except Exception:
            self._intervention_optimizer_ready = False
            return False

    def _optimize_counterfactual(self, features, target_risk=0.20,
                                  lambda_cost=0.1):
        """执行基于梯度的反事实优化"""
        if self.counterfactual_optimizer is None:
            return {'error': '反事实优化器未初始化'}
        try:
            result = self.counterfactual_optimizer.optimize(
                features, target_risk=target_risk, lambda_cost=lambda_cost)
            return result
        except Exception as e:
            return {'error': str(e)}

    def _generate_pareto_frontier(self, features, target_risk=0.20):
        """生成反事实干预的帕累托前沿"""
        if self.counterfactual_optimizer is None:
            return [{'error': '反事实优化器未初始化'}]
        try:
            return self.counterfactual_optimizer.generate_pareto_frontier(
                features, target_risk=target_risk)
        except Exception as e:
            return [{'error': str(e)}]

    def _plan_intervention_sequence(self, initial_state=None, n_steps=8,
                                     preference='balanced',
                                     progress_callback=None):
        """规划个体化动态干预序列"""
        if self.dynamic_planner is None:
            return [], {'error': '动态规划器未初始化'}

        if initial_state is None:
            initial_state = np.zeros(TBInterventionEnv.STATE_DIM if
                                     CAUSAL_RL_AVAILABLE else 31)
            initial_state[2] = 2.0
            initial_state[4] = 5.0

        try:
            return self.dynamic_planner.plan_sequence(
                initial_state, n_steps=n_steps, preference=preference,
                progress_callback=progress_callback)
        except Exception as e:
            return [], {'error': str(e)}

    def _validate_intervention_plan(self, intervention_plan,
                                     patient_context=None):
        """验证干预计划"""
        if self.intervention_validator is None:
            return {'error': '验证器未初始化'}
        try:
            return self.intervention_validator.generate_validation_report(
                intervention_plan, patient_context=patient_context)
        except Exception as e:
            return {'error': str(e)}

    def _extract_contact_features(self, contact_data):
        """从接触者数据中提取30维特征向量（v3.0）"""
        features = np.zeros(30)
        age = contact_data.get('age', 45)
        features[0] = (age - 18) / 72.0
        features[1] = 1.0 if contact_data.get('is_high_risk', 0) == 1 else 0.0
        features[2] = 1.0 if contact_data.get('has_symptoms', 0) == 1 else 0.0
        features[3] = 1.0 if contact_data.get('has_tb', 0) == 1 else 0.0
        features[4] = 0.0 if contact_data.get('bcg_vaccine', 1) == 0 else 1.0
        features[5] = contact_data.get('freq_density', 15) / 30.0
        features[6] = contact_data.get('single_duration', 60) / 480.0
        features[7] = contact_data.get('time_span', 12) / 52.0
        features[8] = contact_data.get('ventilation', 3) / 5.0
        features[9] = 1.0 if contact_data.get('past_illness', 0) == 1 else 0.0
        features[10] = contact_data.get('contact_distance', 2) / 4.0
        features[11] = 1.0 if contact_data.get('idu_status', 0) == 1 else 0.0
        features[12] = 0.5
        features[13] = contact_data.get('family_living_conditions', 3) / 5.0
        features[14] = contact_data.get('cough_freq', 0) / 100.0
        features[15] = 1.0 if contact_data.get('sputum_smear', 1) == 2 else 0.0
        features[16] = 1.0 if contact_data.get('has_cavity', 1) == 2 else 0.0
        features[17] = 1.0 if contact_data.get('treatment', 2) == 2 else 0.0
        features[18] = (contact_data.get('bmi', 22) - 15) / 25.0
        features[19] = 1.0 if contact_data.get('smoking', 0) == 1 else 0.0
        features[20] = 1.0 if contact_data.get('alcohol', 0) == 1 else 0.0
        features[21] = contact_data.get('delay_days', 7) / 52.0
        return features

    # ==================== 多模态GNN ====================

    def _init_multimodal_gnn(self, start_date=None, behavior_data=None,
                              fusion_mode='late'):
        """初始化方向四：多模态GNN环境感知系统"""
        try:
            import datetime as dt_lib
            if start_date is None:
                start_date = dt_lib.date.today()

            self.multimodal_data_access = MultiModalDataAccess(
                climate_interaction=self.climate_interaction if hasattr(
                    self, 'climate_interaction') else None,
                use_synthetic_env=True)

            if behavior_data:
                if 'shifts' in behavior_data:
                    self.multimodal_data_access.set_shift_data(
                        behavior_data['shifts'])
                if 'mobility' in behavior_data:
                    self.multimodal_data_access.set_mobility_data(
                        behavior_data['mobility'])

            self.temporal_graph_builder = EnhancedTemporalGraphBuilder(
                data_access=self.multimodal_data_access,
                climate_interaction=self.climate_interaction if hasattr(
                    self, 'climate_interaction') else None)

            self.gnn_snapshot_cache = self.temporal_graph_builder.\
                build_snapshot_sequence(start_date=start_date, n_days=28,
                                        stride='day')

            if PYTORCH_AVAILABLE:
                self.multimodal_gnn = MultimodalGNN(
                    node_feature_dim=30, edge_feature_dim=6,
                    hidden_dim=64, output_dim=1,
                    n_nodes=self.temporal_graph_builder.total_nodes,
                    fusion_mode=fusion_mode)

                self.temporal_trainer = TemporalTrainer(
                    multimodal_gnn=self.multimodal_gnn,
                    climate_interaction=self.climate_interaction if hasattr(
                        self, 'climate_interaction') else None,
                    graph_builder=self.temporal_graph_builder,
                    data_access=self.multimodal_data_access)

            self._multimodal_ready = True
            return True
        except Exception:
            self._multimodal_ready = False
            return False

    def _predict_multimodal_risk(self, contact_data=None,
                                  use_cache=True, return_curve=False):
        """使用多模态GNN预测感染风险"""
        if not self._multimodal_ready or self.multimodal_gnn is None:
            return None

        try:
            self.multimodal_gnn.eval()
            with torch.no_grad():
                if use_cache and self.gnn_snapshot_cache:
                    snapshots = self.gnn_snapshot_cache
                else:
                    import datetime as dt_lib
                    snapshots = self.temporal_graph_builder.\
                        build_snapshot_sequence(start_date=dt_lib.date.today(),
                                                n_days=28, stride='day')
                    self.gnn_snapshot_cache = snapshots

                env_seq = np.array([
                    self.multimodal_data_access.get_environment_data(
                        s['date']) if self.multimodal_data_access else
                    {'pm10': 80, 'humidity': 50, 'temp': 10, 'wind_speed': 3.5}
                    for s in snapshots
                ])
                env_array = np.array([
                    [min(1.0, e.get('pm10', 80) / 300.0),
                     e.get('humidity', 50) / 100.0,
                     (e.get('temp', 10) + 20.0) / 60.0,
                     min(1.0, e.get('wind_speed', 3.5) / 15.0)]
                    for e in env_seq
                ], dtype=np.float32)
                env_tensor = torch.tensor(env_array)

                risk = self.multimodal_gnn(snapshots,
                                            env_sequence=env_tensor)
                risk_value = float(risk.item())

                result = {'risk': round(risk_value, 4)}

                if return_curve:
                    risk_curve = self.multimodal_gnn.predict_risk_curve(
                        snapshots, env_sequence=env_array)
                    result['risk_curve'] = [round(r, 4) for r in risk_curve]

                    high_dates = []
                    for i, r in enumerate(risk_curve):
                        if r > 0.3:
                            high_dates.append({
                                'date': snapshots[i]['date'],
                                'risk': round(r, 4),
                                'climate_mult': round(
                                    snapshots[i].get('climate_multiplier', 1.0), 3),
                            })
                    result['high_impact_dates'] = high_dates
                    result['climate_impact'] = round(
                        float(np.mean([s.get('climate_multiplier', 1.0)
                                       for s in snapshots])), 3)

                return result
        except Exception:
            return None

    def _get_climate_risk_forecast(self, start_date=None, n_days=14):
        """获取气候风险预报"""
        import datetime as dt_lib
        if start_date is None:
            start_date = dt_lib.date.today()
        elif isinstance(start_date, str):
            start_date = dt_lib.datetime.strptime(
                start_date, '%Y-%m-%d').date()

        if not hasattr(self, 'climate_interaction') or \
           self.climate_interaction is None:
            return []

        forecast = []
        for i in range(n_days):
            d = start_date + dt_lib.timedelta(days=i)
            daily = self.climate_interaction.get_daily_climate_multiplier(
                date=d)
            forecast.append({
                'date': d.isoformat(),
                'climate_multiplier': daily.get('climate_multiplier', 1.0),
                'daily_risk_score': daily.get('daily_risk_score', 0.1),
                'pm10': daily.get('pm10_value', 80.0),
                'humidity': daily.get('humidity_value', 50.0),
                'temperature': daily.get('temperature_value', 10.0),
                'season': daily.get('season', 'unknown'),
                'is_heating': daily.get('is_heating', False),
            })
        return forecast

    def _get_contact_behavior_modifiers(self, dates, locations=None):
        """获取接触行为修正因子序列"""
        if self.multimodal_data_access is None:
            return [1.0] * len(dates)

        if locations is None:
            locations = ['home'] * len(dates)
        elif isinstance(locations, str):
            locations = [locations] * len(dates)

        return [
            self.multimodal_data_access.get_contact_modifier(date, location=loc)
            for date, loc in zip(dates, locations)
        ]

    # ==================== GNN 训练 GUI ====================

    def _append_gnn_log(self, text):
        """向GNN日志文本框追加信息（线程安全）"""
        if hasattr(self, 'gnn_log_text') and hasattr(self, 'root') and self.root:
            try:
                self.root.after(0, lambda t=text: self._do_append_gnn_log(t))
            except (tk.TclError, RuntimeError):
                pass

    def _do_append_gnn_log(self, text):
        if not hasattr(self, 'gnn_log_text'):
            return
        try:
            self.gnn_log_text.insert('end', text + '\n')
            line_count = int(self.gnn_log_text.index('end-1c').split('.')[0])
            if line_count > 500:
                self.gnn_log_text.delete('1.0', f'{line_count - 500}.0')
            self.gnn_log_text.see('end')
        except (tk.TclError, RuntimeError):
            pass

    def _train_gnn_gui(self):
        """训练GNN模型（GUI界面触发）

        Section VII: 使用统一训练参数配置对话框替代两个 askyesno 弹窗。
        """
        if not (PYTORCH_AVAILABLE and PYG_AVAILABLE):
            messagebox.showwarning("依赖缺失",
                "PyTorch或PyTorch Geometric未安装，无法训练GNN模型。\n\n"
                "请安装：\n"
                "pip install torch torch-geometric")
            return

        if self.ml_predictor is None:
            messagebox.showwarning("警告", "ML模块不可用")
            return

        # Section VII: 弹出统一训练参数配置对话框（替代两个 askyesno）
        from .config import TrainingConfig, TrainingConfigDialog
        current_config = getattr(self, '_training_config', None) or TrainingConfig()
        dialog = TrainingConfigDialog(self.root, current_config,
                                       title='GNN 训练参数配置')
        if dialog.result is None:
            return  # 用户取消
        config = dialog.result
        self._training_config = config  # 缓存供下次使用

        try:
            if hasattr(self, 'gnn_log_text'):
                self.gnn_log_text.delete('1.0', 'end')

            self._append_gnn_log("=" * 50)
            self._append_gnn_log("开始GNN训练")
            self._append_gnn_log(f"参数: samples={config.n_samples}, "
                                 f"epochs={config.gnn_n_epochs}, "
                                 f"lr={config.gnn_learning_rate}, "
                                 f"hyperopt={config.gnn_enable_hyperopt}, "
                                 f"focal={config.gnn_use_focal_loss}")
            self._append_gnn_log("=" * 50)

            for widget in self.ml_btn_frame.winfo_children():
                if isinstance(widget, ttk.Button):
                    widget.config(state='disabled')

            # 记录训练开始时间（用于日志持久化）
            import time as _time
            _train_start_time = _time.time()

            # 进度窗口（确定进度条 + 说明文字）
            progress_win = tk.Toplevel(self.root)
            progress_win.title("GNN 训练中")
            progress_win.transient(self.root)
            progress_win.resizable(False, False)
            progress_var = tk.DoubleVar(value=0)
            ttk.Progressbar(progress_win, variable=progress_var,
                            maximum=100, length=320).pack(padx=20, pady=(15, 5))
            prog_label = ttk.Label(progress_win, text="初始化...",
                                   font=('Microsoft YaHei', 9))
            prog_label.pack(padx=20, pady=(0, 15))
            progress_win.protocol("WM_DELETE_WINDOW", lambda: None)

            def _progress_cb(pct, msg):
                """进度回调（后台线程 -> 主线程 UI 更新）"""
                try:
                    self.root.after(
                        0, lambda p=pct, m=msg: (
                            progress_var.set(p),
                            prog_label.config(text=m),
                        ))
                except (tk.TclError, RuntimeError):
                    pass

            def _close_progress():
                try:
                    progress_win.destroy()
                except (tk.TclError, RuntimeError):
                    pass

            def _train_background():
                try:
                    result = self.ml_predictor.train_gnn(
                        n_samples=config.n_samples,
                        random_state=config.random_state,
                        enable_hyperopt=config.gnn_enable_hyperopt,
                        n_epochs=config.gnn_n_epochs,
                        learning_rate=config.gnn_learning_rate,
                        use_focal_loss=config.gnn_use_focal_loss,
                        two_phase_training=config.gnn_two_phase,
                        log_callback=self._append_gnn_log,
                        progress_callback=_progress_cb
                    )

                    def _update_ui_after_train():
                        _duration = _time.time() - _train_start_time
                        _close_progress()
                        if result:
                            messagebox.showinfo("成功", "GNN模型训练完成")
                            self._append_gnn_log("✓ GNN模型训练成功！")
                            # Section VII: 训练日志持久化（含指标与基准对比）
                            try:
                                from .training_log import TrainingLogger
                                logger = TrainingLogger()
                                model_saver = None
                                if self.ml_predictor is not None:
                                    model_saver = lambda path: bool(
                                        self.ml_predictor.save_gnn_model(path))
                                logger.log(
                                    model_type='gnn',
                                    params=config.to_dict(),
                                    metrics=self._extract_gnn_training_metrics(),
                                    training_duration=_duration,
                                    status='success',
                                    dataset=self._gnn_dataset_info(config),
                                    model_saver=model_saver,
                                )
                            except Exception:
                                pass  # 日志失败不影响主流程
                            if self.results:
                                self._run_ml_predictions()
                                self._update_ml_charts()
                                self._display_ml_results()
                        else:
                            self._append_gnn_log("✗ GNN模型训练失败")
                            from ...ops.errors import render_error_dialog_text
                            err = getattr(self.ml_predictor,
                                          '_last_gnn_train_error', None)
                            if err is not None:
                                msg = ("GNN模型训练失败\n\n"
                                       + render_error_dialog_text(err))
                            else:
                                msg = ("GNN模型训练失败\n\n"
                                       "请检查依赖是否完整、训练参数是否合理，"
                                       "并查看训练日志。")
                            messagebox.showerror("错误", msg)
                            try:
                                from .training_log import TrainingLogger
                                logger = TrainingLogger()
                                logger.log(
                                    model_type='gnn',
                                    params=config.to_dict(),
                                    training_duration=_duration,
                                    status='failed',
                                    error_message=str(err) if err is not None else 'unknown',
                                    dataset=self._gnn_dataset_info(config),
                                )
                            except Exception:
                                pass

                        for widget in self.ml_btn_frame.winfo_children():
                            if isinstance(widget, ttk.Button):
                                widget.config(state='normal')

                    self.root.after(0, _update_ui_after_train)

                except Exception as e:
                    def _update_ui_after_error():
                        _duration = _time.time() - _train_start_time
                        _close_progress()
                        from ...ops.errors import render_error_dialog_text
                        messagebox.showerror("错误",
                                             render_error_dialog_text(e))
                        self._append_gnn_log(f"✗ 训练异常: {str(e)}")
                        import traceback
                        self._append_gnn_log(traceback.format_exc())
                        try:
                            from .training_log import TrainingLogger
                            logger = TrainingLogger()
                            logger.log(
                                model_type='gnn',
                                params=config.to_dict(),
                                training_duration=_duration,
                                status='failed',
                                error_message=str(e),
                                dataset=self._gnn_dataset_info(config),
                            )
                        except Exception:
                            pass

                        for widget in self.ml_btn_frame.winfo_children():
                            if isinstance(widget, ttk.Button):
                                widget.config(state='normal')

                    self.root.after(0, _update_ui_after_error)

            training_thread = threading.Thread(target=_train_background)
            training_thread.daemon = True
            training_thread.start()

        except Exception as e:
            messagebox.showerror("错误", f"启动训练失败: {str(e)}")
            import traceback
            self._append_gnn_log(traceback.format_exc())

    def _train_pignn_gui(self):
        """训练PIGNN物理信息图神经网络（GUI界面触发）

        文献支撑：
        - Raissi et al., Physics-informed neural networks. J. Comp. Phys., 2019.
        - Chen et al., Neural ordinary differential equations. NeurIPS, 2018.
        """
        if not (PYTORCH_AVAILABLE and PYG_AVAILABLE):
            messagebox.showwarning("依赖缺失",
                "PyTorch或PyTorch Geometric未安装，无法训练PIGNN模型。")
            return
        if not PIGNN_AVAILABLE:
            messagebox.showwarning("模块缺失", "pignn_model.py未找到，无法使用PIGNN。")
            return
        if self.ml_predictor is None:
            messagebox.showwarning("警告", "ML模块不可用")
            return

        lambda_physics = 0.1
        lambda_consv = 0.05
        n_samples = 1000
        n_epochs = 30

        dlg = tk.Toplevel(self.root)
        dlg.title("PIGNN训练配置")
        dlg.geometry("450x320")
        dlg.resizable(False, False)

        frm = ttk.Frame(dlg, padding=15)
        frm.pack(fill='both', expand=True)

        ttk.Label(frm, text="Physics-Informed Graph Neural Network",
                  font=('Arial', 11, 'bold')).pack(anchor='w', pady=(0, 10))
        ttk.Label(frm, text="L_total = L_data + λ₁·L_physics + λ₂·L_conservation",
                  font=('Consolas', 9)).pack(anchor='w', pady=(0, 10))

        row = ttk.Frame(frm); row.pack(fill='x', pady=3)
        ttk.Label(row, text="物理残差系数 λ_physics:", width=20).pack(side='left')
        lp_var = tk.StringVar(value=str(lambda_physics))
        ttk.Entry(row, textvariable=lp_var, width=10).pack(side='left')

        row = ttk.Frame(frm); row.pack(fill='x', pady=3)
        ttk.Label(row, text="守恒约束系数 λ_consv:", width=20).pack(side='left')
        lc_var = tk.StringVar(value=str(lambda_consv))
        ttk.Entry(row, textvariable=lc_var, width=10).pack(side='left')

        row = ttk.Frame(frm); row.pack(fill='x', pady=3)
        ttk.Label(row, text="合成样本数:", width=20).pack(side='left')
        ns_var = tk.StringVar(value=str(n_samples))
        ttk.Entry(row, textvariable=ns_var, width=10).pack(side='left')

        row = ttk.Frame(frm); row.pack(fill='x', pady=3)
        ttk.Label(row, text="训练轮次:", width=20).pack(side='left')
        ep_var = tk.StringVar(value=str(n_epochs))
        ttk.Entry(row, textvariable=ep_var, width=10).pack(side='left')

        def _start():
            try:
                lp = float(lp_var.get())
                lc = float(lc_var.get())
                ns = int(ns_var.get())
                ep = int(ep_var.get())
            except ValueError:
                messagebox.showerror("错误", "请输入有效数值")
                return
            dlg.destroy()
            use_focal = messagebox.askyesno("损失函数选择", "是否使用Focal Loss处理类别不平衡？")

            try:
                if hasattr(self, 'gnn_log_text'):
                    self.gnn_log_text.delete('1.0', 'end')
                self._append_gnn_log("=" * 50)
                self._append_gnn_log("PIGNN 物理信息图神经网络训练")
                self._append_gnn_log("=" * 50)
                self._append_gnn_log(f"λ_physics={lp}, λ_consv={lc}, samples={ns}, epochs={ep}")
                for w in self.ml_btn_frame.winfo_children():
                    if isinstance(w, ttk.Button):
                        w.config(state='disabled')

                def _train():
                    try:
                        result = self.ml_predictor.train_pignn(
                            n_samples=ns, n_epochs=ep, random_state=42,
                            lambda_physics=lp, lambda_consv=lc,
                            use_focal_loss=use_focal, log_callback=self._append_gnn_log)

                        def _update():
                            if result:
                                msg = f"PIGNN训练完成\nAUROC={result.get('auroc',0):.4f}\n"
                                if 'physics_convergence' in result:
                                    msg += f"物理残差收敛: {result['physics_convergence']:.2e}"
                                messagebox.showinfo("成功", msg)
                                self._append_gnn_log("OK PIGNN训练成功！")
                                if self.results:
                                    self._run_ml_predictions()
                                    self._update_ml_charts()
                                    self._display_ml_results()
                            else:
                                messagebox.showerror("错误", "PIGNN训练失败")
                                self._append_gnn_log("FAIL PIGNN训练失败")
                            for w in self.ml_btn_frame.winfo_children():
                                if isinstance(w, ttk.Button):
                                    w.config(state='normal')
                        self.root.after(0, _update)
                    except Exception as e:
                        def _err():
                            messagebox.showerror("错误", f"训练失败: {e}")
                            self._append_gnn_log(f"FAIL 异常: {e}")
                            import traceback
                            self._append_gnn_log(traceback.format_exc())
                            for w in self.ml_btn_frame.winfo_children():
                                if isinstance(w, ttk.Button):
                                    w.config(state='normal')
                        self.root.after(0, _err)

                self._start_training_thread(_train)
            except Exception as e:
                messagebox.showerror("错误", f"启动训练失败: {e}")
                import traceback as _tb
                self._append_gnn_log(_tb.format_exc())

        ttk.Button(frm, text="开始训练", command=_start).pack(pady=15)
        ttk.Label(frm, text="文献: Raissi et al. (2019), Chen et al. (2018)",
                  font=('Arial', 7)).pack(side='bottom')

    def _save_gnn_gui(self):
        """保存GNN模型（GUI界面触发）"""
        if not (PYTORCH_AVAILABLE and PYG_AVAILABLE):
            messagebox.showwarning("警告", "PyTorch/PyG不可用")
            return

        if self.ml_predictor is None or not self.ml_predictor.gnn_is_trained:
            messagebox.showwarning("警告", "GNN模型未训练，无法保存")
            return

        file_path = filedialog.asksaveasfilename(
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")],
            title="保存GNN模型",
            defaultextension=".pth"
        )

        if not file_path:
            return

        if self.ml_predictor.save_gnn_model(file_path):
            messagebox.showinfo("成功", f"GNN模型已保存：{file_path}")
            self._append_gnn_log(f"✓ GNN模型已保存到 {file_path}")
        else:
            messagebox.showerror("错误", "GNN模型保存失败")

    def _load_gnn_gui(self):
        """加载GNN模型（GUI界面触发）"""
        if not (PYTORCH_AVAILABLE and PYG_AVAILABLE):
            messagebox.showwarning("警告", "PyTorch/PyG不可用")
            return

        if self.ml_predictor is None:
            messagebox.showwarning("警告", "ML模块不可用")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")],
            title="加载GNN模型"
        )

        if not file_path:
            return

        if self.ml_predictor.load_gnn_model(file_path):
            messagebox.showinfo("成功", f"GNN模型已加载：{file_path}")
            self._append_gnn_log(f"✓ GNN模型已从 {file_path} 加载")
            if self.results:
                self._run_ml_predictions()
                self._update_ml_charts()
                self._display_ml_results()
        else:
            messagebox.showerror("错误", "GNN模型加载失败")