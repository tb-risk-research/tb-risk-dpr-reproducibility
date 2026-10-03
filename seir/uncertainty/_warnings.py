"""SEIR 参数不确定性量化 — 高置信度警告模块

基于 R0 后验概率生成传播风险警告。
拆分自原 seir/uncertainty.py（业务逻辑不变）。
"""


class _WarningsMixin:
    """高置信度传播风险警告"""

    def get_high_confidence_warning(self, threshold=0.9):
        """获取高置信度传播风险警告"""
        if not self.inference_completed:
            return {'warning': False, 'message': 'MCMC未完成'}
        p_r0_gt_1 = self.r0_posterior.get('p_r0_gt_1', 0.0)
        return {
            'warning': p_r0_gt_1 > threshold,
            'level': 'high' if p_r0_gt_1 > 0.95 else (
                'medium' if p_r0_gt_1 > 0.8 else 'low'),
            'message': (f'R0>1后验概率={p_r0_gt_1:.1%}, '
                         f'均值={self.r0_posterior["mean"]:.2f}'),
            'p_r0_gt_1': float(p_r0_gt_1),
            'r0_mean': float(self.r0_posterior.get('mean', 0)),
        }
