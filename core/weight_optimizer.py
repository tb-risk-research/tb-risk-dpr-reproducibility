#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""集成权重自动优化器

纯业务逻辑层，零依赖 tkinter 或任何 GUI 库。
从 assessment.py 迁移而来，独立于 TB_Risk_Assessment 类。

职责：
- 自动编排消融实验（确保 SEIR/GNN 方向 AUROC 数据可用）
- 基于 softmax(log-AUROC) 自动优化集成权重
"""

import logging
import os
from typing import Any, Dict, Optional

LOGGER = logging.getLogger("tb_risk.core.weight_optimizer")


class EnsembleWeightOptimizer:
    """集成权重自动优化器

    建立"消融实验 → 边际贡献分析 → 权重优化 → 集成预测"的闭环。

    用法：
        optimizer = EnsembleWeightOptimizer(
            ml_predictor=app.ml_predictor,
            integrator=app.integrator,
            karamay_validator_ref=app,  # 用于回写 karamay_validator
            karamay_localizer=app.karamay_localizer,
        )
        optimizer.ensure_validation()
        optimizer.optimize_weights()
    """

    def __init__(
        self,
        ml_predictor: Any = None,
        integrator: Any = None,
        karamay_validator_ref: Any = None,
        karamay_localizer: Any = None,
        validation_data_path: Optional[str] = None,
    ):
        """初始化优化器

        参数：
            ml_predictor: MLRiskPredictor 实例
            integrator: ThreeDirectionIntegrator 实例
            karamay_validator_ref: 持有 karamay_validator 属性的对象（用于回写）
            karamay_localizer: KaramayLocalizer 实例（可选）
            validation_data_path: 验证数据路径（可选）
        """
        self._ml_predictor = ml_predictor
        self._integrator = integrator
        self._karamay_validator_ref = karamay_validator_ref
        self._karamay_localizer = karamay_localizer
        self._validation_data_path = validation_data_path
        self._validation_auto_triggered = False

    @property
    def karamay_validator(self):
        """从引用对象获取 karamay_validator"""
        if self._karamay_validator_ref is None:
            return None
        return getattr(self._karamay_validator_ref, 'karamay_validator', None)

    @karamay_validator.setter
    def karamay_validator(self, value):
        """回写 karamay_validator 到引用对象"""
        if self._karamay_validator_ref is not None:
            self._karamay_validator_ref.karamay_validator = value

    def ensure_validation(self) -> None:
        """自动确保消融实验已运行（用于权重优化时序编排）。

        若 karamay_validator 尚未实例化（用户未手动点击"克拉玛依验证"菜单），
        自动实例化并运行轻量消融实验，确保 SEIR 方向的 AUROC 和方向边际贡献
        数据可用，不依赖于用户手动操作的先后顺序。

        设计约束：
        1. 使用少量样本（500 条合成数据）以控制耗时
        2. 仅运行消融实验（不运行完整验证套件）
        3. 每个会话只运行一次（_validation_auto_triggered 标志）
        4. 失败时记录警告并回退到默认权重，不阻塞主流程
        """
        # 已运行过，跳过
        if self._validation_auto_triggered:
            return

        # 已有 karamay_validator（用户手动运行过），跳过
        if self.karamay_validator is not None:
            self._validation_auto_triggered = True
            return

        # 无本土化模块，无法运行验证
        if self._karamay_localizer is None:
            LOGGER.info("无本土化模块，跳过自动验证编排")
            self._validation_auto_triggered = True
            return

        try:
            from ..validation.validator import KaramayValidator
            from ..scoring.simulator import ScreeningDataSimulator

            LOGGER.info("自动编排：触发消融实验以获取 SEIR/GNN 方向权重数据...")
            self.karamay_validator = KaramayValidator(
                self._karamay_localizer, random_state=42
            )

            # 生成数据用于消融实验（优先真实数据，无真实数据时回退合成）
            simulator = ScreeningDataSimulator(random_state=42)
            if self._validation_data_path and os.path.exists(self._validation_data_path):
                records, _ = simulator.load_or_generate(
                    filepath=self._validation_data_path, n_samples=500, include_labels=True
                )
                LOGGER.info("消融实验使用真实数据: %s (%d 条)", self._validation_data_path, len(records))
            else:
                records = simulator.generate_dataset(n_samples=500)
                LOGGER.info("消融实验使用合成数据 (%d 条)", len(records))

            # 运行消融实验（含 AUROC 和边际贡献分析）
            from ..validation.ablation import AblationAnalyzer
            ablation = AblationAnalyzer(self._karamay_localizer)
            ablation_report = ablation.run_ablation_with_metrics(records)

            # 存入 _last_results，与手动验证路径保持一致的数据结构
            kv = self.karamay_validator
            if kv is not None:
                kv._last_results = {
                    'ablation': ablation_report,
                }

            LOGGER.info(
                "自动验证编排完成：SEIR AUROC=%.4f, 边际贡献方向=%s",
                ablation_report.get('full_model', {}).get('report', {}).get('auroc', 0),
                list(ablation_report.get('direction_contributions', {}).keys())
            )
        except Exception as e:
            LOGGER.warning(
                "自动验证编排失败: %s，权重优化将仅使用 ML 方向数据，"
                "SEIR/GNN 方向使用默认权重 0.3/0.3",
                e
            )
        finally:
            self._validation_auto_triggered = True

    def optimize_weights(self) -> None:
        """自动优化集成权重：基于各模型验证集 AUROC 计算性能加权。

        建立"消融实验 → 边际贡献分析 → 权重优化 → 集成预测"的闭环。
        当 ML 模型训练完成且 model_performance 可用时自动调用。

        优化路径（按优先级）：
        1. 若消融实验结果可用（karamay_validator 已运行），
           结合 direction_contributions 边际贡献进行交叉验证，
           用 softmax(log-AUROC) 计算性能加权。
        2. 若仅 ML 模型有 AUROC 数据，用 optimize_weights_from_validation
           进行单方向基准权重优化，SEIR/GNN 保持默认权重。
        3. 若所有方向均无 AUROC 数据，保持默认权重 0.4/0.3/0.3。

        文献：
        - Perrone & Cooper (1993) "When Networks Disagree"
        - Jacobs et al. (1991) "Adaptive Mixtures of Local Experts"
        """
        try:
            # 收集各方向的 AUROC
            validation_auroc: Dict[str, float] = {}
            perf = self._ml_predictor.model_performance if self._ml_predictor else {}

            # ML 方向：取最佳模型的 AUROC
            if isinstance(perf, dict):
                auroc_values = []
                for model_name, metrics in perf.items():
                    if isinstance(metrics, dict) and 'AUROC' in metrics:
                        auroc_values.append(metrics['AUROC'])
                if auroc_values:
                    validation_auroc['ml'] = max(auroc_values)

            # SEIR 方向：从消融实验结果获取（若 karamay_validator 已运行）
            kv = self.karamay_validator
            if kv is not None:
                try:
                    last_results = getattr(kv, '_last_results', {})
                    ablation = last_results.get('ablation', {})
                    seir_auroc = ablation.get('full_model', {}).get('report', {}).get('auroc')
                    if seir_auroc is not None:
                        validation_auroc['seir'] = seir_auroc
                except Exception as e:
                    LOGGER.debug("获取SEIR方向AUROC失败: %s", e)

            # GNN 方向：从 model_performance 获取
            gnn_perf = perf.get('gnn', {}) if isinstance(perf, dict) else {}
            if isinstance(gnn_perf, dict) and 'AUROC' in gnn_perf:
                validation_auroc['gnn'] = gnn_perf['AUROC']

            if len(validation_auroc) < 2:
                # 路径 2: 单方向或零方向 AUROC → 单方向基准优化
                if len(validation_auroc) == 1:
                    LOGGER.info(
                        "集成权重优化：仅 %s 方向可用，使用单方向基准优化",
                        list(validation_auroc.keys())[0]
                    )
                    self._integrator.optimize_weights_from_validation(validation_auroc)
                else:
                    LOGGER.info(
                        "集成权重优化跳过：无可用模型方向 AUROC（需要至少 2 个方向），"
                        "保持默认权重 0.4/0.3/0.3"
                    )
                return

            # 路径 1: 结合消融实验的边际贡献进行交叉验证
            direction_contribs = None
            kv = self.karamay_validator
            if kv is not None:
                try:
                    last_results = getattr(kv, '_last_results', {})
                    ablation = last_results.get('ablation', {})
                    direction_contribs = ablation.get('direction_contributions')
                except Exception as e:
                    LOGGER.debug("获取方向边际贡献失败: %s", e)

            # 若某方向边际贡献不显著，降低其权重
            if direction_contribs:
                for direction, contrib in direction_contribs.items():
                    if direction in validation_auroc and contrib.get('marginal_auroc_sum', 0) < 0.01:
                        LOGGER.warning(
                            "方向 '%s' 消融边际贡献不显著 (ΔAUROC=%.4f)，建议简化集成结构",
                            direction, contrib.get('marginal_auroc_sum', 0)
                        )
                        scale = max(0.1, contrib.get('marginal_auroc_sum', 0) / 0.05)
                        validation_auroc[direction] *= scale

            # 执行 softmax(log-AUROC) 加权优化
            self._integrator.optimize_weights_from_validation(validation_auroc)
            LOGGER.info(
                "集成权重已自动优化 (来源: %s): %s",
                self._integrator.weights_source,
                self._integrator.ensemble_weights.get('three', {})
            )

        except Exception as e:
            LOGGER.warning("自动权重优化失败: %s，保持默认权重", e)