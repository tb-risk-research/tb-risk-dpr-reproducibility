#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 训练面板 - 不确定性量化（_UncertaintyMixin）

包含：SEIR参数不确定性、DeepEnsemble、Conformal预测、MC Dropout、SWAG
"""

from .._shared import *

from tb_risk.core.lab_evidence import (
    uncertainty_adaptive_factor,
    derive_lab_evidence_grade,
)


class _UncertaintyMixin:
    """不确定性量化相关方法"""

    # ==================== 不确定性量化 ====================

    def _init_uncertainty_system(self, observed_data=None,
                                  ensemble_data=None, calibrate_data=None):
        """初始化方向五：不确定性量化体系"""
        try:
            self.seir_param_uncertainty = SEIRParameterUncertainty(
                population=10000, random_seed=42)

            if observed_data:
                self.seir_param_uncertainty.run_mcmc(
                    observed_data, n_iterations=3000, n_burnin=500,
                    n_chains=2)

            self.deep_ensemble = DeepEnsemblePredictor(
                n_ensembles=5, base_model_class=QuantileRegressionForest)
            self.deep_ensemble.initialize_models(input_dim=22)

            if ensemble_data and PYTORCH_AVAILABLE:
                X, y = ensemble_data
                self.deep_ensemble.train_ensemble(
                    X, y, n_epochs=20, bootstrap_ratio=0.8)

            self.conformal_predictor = ConformalPredictor(coverage=0.90)

            if calibrate_data:
                y_true, y_pred = calibrate_data
                self.conformal_predictor.calibrate(y_true, y_pred)

            gnn_model = None
            if hasattr(self, 'ml_predictor') and self.ml_predictor is not None:
                if hasattr(self.ml_predictor, 'seir_gnn') and \
                   self.ml_predictor.seir_gnn is not None:
                    gnn_model = self.ml_predictor.seir_gnn

            if gnn_model is not None and PYTORCH_AVAILABLE:
                self.mc_dropout_gnn = MCDropoutGNNUncertainty(
                    gnn_model, n_samples=30)
                self.swag_estimator = SWAGEstimator(
                    gnn_model, n_models=20, max_rank=5)
            else:
                self.mc_dropout_gnn = None
                self.swag_estimator = None

            self.uncertainty_fusion = UncertaintyFusionEngine(
                seir_uncertainty=self.seir_param_uncertainty,
                ml_ensemble=self.deep_ensemble,
                conformal=self.conformal_predictor,
                mc_dropout=self.mc_dropout_gnn,
                swag=self.swag_estimator)

            self.uncertainty_visualizer = UncertaintyVisualizer(
                fusion_engine=self.uncertainty_fusion)

            if self.seir_param_uncertainty.inference_completed:
                self.seir_posterior_predictive = \
                    self.seir_param_uncertainty.generate_posterior_predictive(
                        n_samples=50, horizon=365)

            self._uncertainty_ready = True
            return True
        except Exception:
            self._uncertainty_ready = False
            return False

    def _predict_with_uncertainty(self, features, edge_index=None,
                                   return_details=False, evidence_factor=1.0):
        """带不确定性量化的风险预测

        参数：
            features: 模型特征
            edge_index: GNN 边索引（可选）
            return_details: 是否返回各层原始结果
            evidence_factor: 实验室证据自适应因子（≥1 时放大共形预测区间）。
                由 uncertainty_adaptive_factor(record) 计算：
                lab_evidence_grade=0（无实验室证据）→ 1.6，仅影像 → 1.3，
                高等级证据 → 1.0。默认 1.0 表示无记录/证据时保持原区间。
        """
        if not self._uncertainty_ready:
            return {'error': '不确定性系统未初始化'}

        ml_result = None
        if self.deep_ensemble and self.deep_ensemble.is_trained:
            try:
                ml_result = self.deep_ensemble.predict_with_uncertainty(
                    features)
            except Exception:
                ml_result = None

        gnn_result = None
        if self.mc_dropout_gnn is not None and edge_index is not None:
            try:
                x_t = torch.tensor(np.array(features, dtype=np.float32))
                ei_t = torch.tensor(np.array(edge_index, dtype=torch.long))
                gnn_result = self.mc_dropout_gnn.predict_mc(x_t, ei_t)
            except Exception:
                gnn_result = None

        conformal_result = None
        if ml_result and 'error' not in ml_result and \
           self.conformal_predictor and self.conformal_predictor.is_calibrated:
            try:
                conformal_result = self.conformal_predictor.predict_interval(
                    ml_result.get('mean', 0.3),
                    adaptive_factor=float(evidence_factor))
            except Exception:
                conformal_result = None

        prior_risk = 0.3
        if isinstance(features, (list, np.ndarray)):
            feat_arr = np.array(features).flatten()
            if len(feat_arr) >= 22:
                prior_risk = float(0.15 + 0.2 *
                    (feat_arr[5] + feat_arr[8] + feat_arr[13]) / 3.0)

        result = self.uncertainty_fusion.fuse_individual(
            ml_result=ml_result,
            gnn_result=gnn_result,
            conformal_result=conformal_result,
            prior_risk=prior_risk,
            evidence_factor=evidence_factor)

        if return_details:
            result['ml_raw'] = ml_result
            result['gnn_raw'] = gnn_result
            result['conformal_raw'] = conformal_result

        return result

    def _get_uncertainty_report(self):
        """获取不确定性量化综合报告"""
        if self.uncertainty_fusion is None:
            return {'error': '融合引擎未初始化'}
        return self.uncertainty_fusion.generate_report()

    def _get_formatted_risk_table(self, contacts_data):
        """获取带不确定性信息的格式化风险列表"""
        if self.uncertainty_fusion is None or \
           self.uncertainty_visualizer is None:
            return []

        fusion_results = self.uncertainty_fusion.fuse_batch(
            contacts_data,
            feature_extractor=self._extract_contact_features)

        return self.uncertainty_visualizer.prepare_table_data(
            fusion_results)

    def _run_swag_collection(self, dataloader, n_collect=15,
                              collect_epoch=0):
        """运行SWAG权重收集"""
        if self.swag_estimator is None or not PYTORCH_AVAILABLE:
            return False
        self.swag_estimator.collection_complete = False
        self.swag_estimator.n_collected = 0
        for i in range(n_collect):
            self.swag_estimator.collect_model(
                self.swag_estimator.base_model)
        return self.swag_estimator.collection_complete

    def _get_seir_uncertainty_plot(self):
        """获取SEIR不确定性可视化数据"""
        if self.seir_posterior_predictive is None or \
           self.uncertainty_visualizer is None:
            return None
        return self.uncertainty_visualizer.prepare_seir_plot_data(
            self.seir_posterior_predictive)
