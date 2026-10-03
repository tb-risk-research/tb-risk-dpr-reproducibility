"""共形预测

在独立校准集上计算非一致性分数，生成分布无关的预测区间。
支持自适应加权（高风险人群区间自动收紧）。

文献：[4] Barber et al. 2021, [5] Adaptive CP via Bayesian UW 2026
"""

import os
import numpy as np

from ._common import LOGGER, JOBLIB_AVAILABLE, joblib


class ConformalPredictor:
    """共形预测

    在独立校准集上计算非一致性分数，生成分布无关的预测区间。
    支持自适应加权（高风险人群区间自动收紧）。

    文献：[4] Barber et al. 2021, [5] Adaptive CP via Bayesian UW 2026
    """

    def __init__(self, coverage=0.90):
        self.coverage = coverage
        self.calibration_scores = None
        self.confidence_threshold = None
        self.is_calibrated = False
        self.adaptive_weights = None

    def calibrate(self, y_true, y_pred, sample_weights=None):
        """在校准集上计算非一致性阈值

        参数：
            y_true: 真实标签
            y_pred: 模型预测
            sample_weights: 样本权重（用于自适应）

        返回：
            float: 共形预测阈值
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        scores = np.abs(y_true - y_pred)

        if sample_weights is not None:
            weights = np.array(sample_weights).flatten()
            weights_sum = weights.sum()
            if weights_sum <= 0:
                LOGGER.warning("样本权重和为 %.3f，回退为等权重", weights_sum)
                weights_sum = len(weights)
                weights = np.ones_like(weights)
            self.adaptive_weights = weights / weights_sum
            sorted_idx = np.argsort(scores)
            cum_weights = np.cumsum(
                self.adaptive_weights[sorted_idx])
            threshold_idx = np.searchsorted(
                cum_weights, self.coverage)
            threshold_idx = min(threshold_idx, len(scores) - 1)
            self.confidence_threshold = scores[sorted_idx[threshold_idx]]
        else:
            n_cal = len(scores) + 1
            k = int(np.ceil(self.coverage * n_cal))
            k = min(k, len(scores))
            self.confidence_threshold = float(np.partition(scores, k-1)[k-1])

        self.calibration_scores = scores
        self.is_calibrated = True

        return self.confidence_threshold

    def predict_interval(self, y_pred, adaptive_factor=None):
        """生成预测区间

        参数：
            y_pred: 模型预测
            adaptive_factor: 自适应因子（0-1之间，小值收紧区间）

        返回：
            dict: {'lower', 'upper', 'width', 'coverage_target'}
        """
        if not self.is_calibrated:
            return {'error': '未校准'}

        threshold = self.confidence_threshold
        if adaptive_factor is not None:
            factor = np.clip(float(adaptive_factor), 0.25, 2.0)
            threshold *= factor

        y_pred = np.array(y_pred).flatten()
        lower = np.maximum(0, y_pred - threshold)
        upper = np.minimum(1, y_pred + threshold)

        return {
            'lower': lower.tolist(),
            'upper': upper.tolist(),
            'width': (upper - lower).tolist(),
            'coverage_target': self.coverage,
            'threshold': float(threshold),
        }

    def adaptive_interval(self, y_pred, risk_levels):
        """自适应区间：高风险收紧，低风险放宽

        文献：[5] Bayesian UW Adaptive CP
        """
        if not self.is_calibrated:
            return {'error': '未校准'}

        risk_levels = np.array(risk_levels).flatten()
        adaptive_factors = 1.0 - risk_levels * 0.5

        intervals = []
        for pred, factor in zip(np.array(y_pred).flatten(), adaptive_factors):
            threshold = self.confidence_threshold * factor
            intervals.append({
                'lower': max(0, pred - threshold),
                'upper': min(1, pred + threshold),
                'adaptive_factor': round(float(factor), 3),
            })

        return intervals

    # ========== 持久化方法 ==========

    def save(self, filepath):
        """保存共形预测器到文件

        保存校准集残差、非一致性分数分布和置信度阈值，
        使模型重启后无需重新校准。

        参数：
            filepath (str): 保存路径，建议使用 .joblib 扩展名

        返回：
            bool: 保存是否成功
        """
        if not JOBLIB_AVAILABLE:
            LOGGER.warning("joblib 未安装，无法保存 ConformalPredictor")
            return False
        if not self.is_calibrated:
            LOGGER.warning("ConformalPredictor 未校准，无法保存")
            return False

        allowed_exts = ('.joblib', '.pkl', '.pickle')
        if not filepath.lower().endswith(allowed_exts):
            LOGGER.warning("拒绝保存到非模型文件扩展名: %s (仅支持 %s)", filepath, allowed_exts)
            return False

        try:
            save_data = {
                'version': '2.0',
                'model_type': 'ConformalPredictor',
                'coverage': self.coverage,
                'calibration_scores': self.calibration_scores,
                'confidence_threshold': self.confidence_threshold,
                'is_calibrated': self.is_calibrated,
                'adaptive_weights': self.adaptive_weights,
            }
            joblib.dump(save_data, filepath)
            LOGGER.info("ConformalPredictor 已保存到: %s (coverage=%.2f)", filepath, self.coverage)
            return True
        except Exception as e:
            LOGGER.warning("ConformalPredictor 保存失败: %s", e, exc_info=True)
            return False

    def load(self, filepath):
        """从文件加载共形预测器

        恢复校准状态，无需重新校准即可直接使用。

        参数：
            filepath (str): 模型文件路径

        返回：
            bool: 加载是否成功
        """
        if not JOBLIB_AVAILABLE:
            LOGGER.warning("joblib 未安装，无法加载 ConformalPredictor")
            return False

        allowed_exts = ('.joblib', '.pkl', '.pickle')
        if not filepath.lower().endswith(allowed_exts):
            LOGGER.warning("拒绝加载非模型文件: %s (仅支持 %s)", filepath, allowed_exts)
            return False

        if not os.path.exists(filepath):
            LOGGER.warning("ConformalPredictor 文件不存在: %s", filepath)
            return False

        try:
            LOGGER.info("加载 ConformalPredictor: %s", filepath)
            save_data = joblib.load(filepath)

            if not isinstance(save_data, dict) or save_data.get('model_type') != 'ConformalPredictor':
                LOGGER.warning("无效的 ConformalPredictor 文件格式")
                return False

            self.coverage = save_data.get('coverage', 0.90)
            self.calibration_scores = save_data.get('calibration_scores')
            self.confidence_threshold = save_data.get('confidence_threshold')
            self.is_calibrated = save_data.get('is_calibrated', True)
            self.adaptive_weights = save_data.get('adaptive_weights')

            LOGGER.info("ConformalPredictor 加载成功: coverage=%.2f, threshold=%.4f",
                      self.coverage, self.confidence_threshold or 0)
            return True
        except Exception as e:
            LOGGER.warning("ConformalPredictor 加载失败: %s", e, exc_info=True)
            return False
