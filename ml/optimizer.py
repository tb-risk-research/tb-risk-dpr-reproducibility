"""
基于梯度的组合反事实优化器

通过梯度下降搜索最小成本的特征修改方案，使风险预测降至阈值以下。

文献：
  Wachter S et al. (2018) Harvard Data Sci Review
  Mothilal RK et al. (2020) AAAI
  Verma S et al. (2020) IJCAI
"""

import numpy as np

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None


if PYTORCH_AVAILABLE:
    class _StraightThroughClamp(torch.autograd.Function):
        """Straight-Through Estimator for clamping.

        前向：与 torch.clamp 相同，硬截断到 [min, max]（保证可行域精确）。
        反向：梯度直通（identity），边界处梯度不为零，
        允许优化器在达到边界后仍接收梯度信号持续探索。

        相比 torch.clamp（边界处梯度为零，会"卡住"优化器），
        STE 使 loss.backward() 能自动回传梯度到 delta，
        无需手工 detach+requires_grad 拼补计算图。

        文献：Bengio et al. (2013) "Estimating or Propagating Gradients
        Through Stochastic Neurons for Conditional Computation"
        """
        @staticmethod
        def forward(ctx, x, lo, hi):
            return torch.clamp(x, lo, hi)

        @staticmethod
        def backward(ctx, grad_output):
            # 梯度直通：返回对 x 的梯度，lo/hi 无梯度（None）
            return grad_output, None, None

    def _straight_through_clamp(x, lo, hi):
        """_StraightThroughClamp 的函数式包装。"""
        return _StraightThroughClamp.apply(x, lo, hi)
else:
    def _straight_through_clamp(x, lo, hi):
        # torch 不可用时不会被调用（_clip_to_feasible 的 tensor 分支不可达）
        raise RuntimeError("torch 不可用")


class DifferentiableCounterfactualOptimizer:
    """基于梯度的组合反事实优化器

    通过梯度下降搜索最小成本的特征修改方案，使风险预测降至阈值以下。

    文献：
      Wachter S et al. (2018) Harvard Data Sci Review
      Mothilal RK et al. (2020) AAAI
      Verma S et al. (2020) IJCAI
    """

    FEATURE_NAMES_CN = [
        '年龄', '高危人群', '症状', '结核史', 'BCG接种', '接触频次',
        '单次时长', '持续周期', '通风条件', '慢性病史', '接触距离',
        'IDU状态', '暴露场景', '居住条件', '咳嗽频率', '菌阳状态',
        '空洞', '治疗阶段', 'BMI', '吸烟', '饮酒', '延迟就诊'
    ]

    MUTABLE_FEATURES = {
        5: ('接触频次', 0, 30),
        6: ('单次时长(分钟)', 0, 480),
        7: ('持续周期(周)', 1, 52),
        8: ('通风条件', 1, 5),
        10: ('接触距离', 0.0, 4.0),
        13: ('居住条件', 1, 5),
        14: ('咳嗽频率', 0, 100),
        17: ('治疗阶段', 1, 3),
        21: ('延迟就诊(周)', 0, 52),
    }

    IMMUTABLE_FEATURES = {0, 1, 2, 3, 4, 9, 11, 12, 15, 16, 18, 19, 20}

    INTERVENTION_COSTS = {
        '接触频次': 0.5,
        '单次时长(分钟)': 0.3,
        '持续周期(周)': 0.8,
        '通风条件': 2.0,
        '接触距离': 0.2,
        '居住条件': 3.0,
        '咳嗽频率': 1.0,
        '治疗阶段': 5.0,
        '延迟就诊(周)': 4.0,
    }

    def __init__(self, risk_predictor, device='cpu', baseline_risk=None):
        """初始化反事实优化器。

        参数：
            risk_predictor: 已训练的风险预测模型（GNN 或 MLP）
            device: 计算设备
            baseline_risk: 可选的训练集均值风险（float，0~1）。
                          当 risk_predictor 两次调用均失败时作为常量回退。
                          若为 None，则回退时抛出 RuntimeError，由上层调用方处理。
                          常量回退的梯度为零，会正确信号梯度下降该方向无收益。
        """
        self.risk_predictor = risk_predictor
        self.device = device
        self.risk_predictor.eval()
        # 训练集均值风险，作为 risk_predictor 调用失败时的常量回退
        # （梯度为零，避免 sigmoid(x).sum() 引入虚假梯度信号）
        if baseline_risk is not None:
            try:
                self.baseline_risk = float(baseline_risk)
                if not (0.0 <= self.baseline_risk <= 1.0):
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError(
                    f"baseline_risk 必须为 [0, 1] 范围内的数值，得到 {baseline_risk!r}")
        else:
            self.baseline_risk = None

    def _clip_to_feasible(self, x, delta):
        """将x+delta投影回可行域（支持numpy和torch）。

        张量路径使用 Straight-Through Estimator (STE)：
        前向用 torch.clamp 硬截断（保证可行域精确），
        反向梯度直通（边界处梯度不为零，允许优化器持续探索）。
        这样 loss.backward() 能自动回传梯度到 delta，
        无需手工 detach+requires_grad 拼补。

        关键实现细节：避免对 x_new（非叶子张量、处于计算图中）做原地切片赋值
        x_new[idx] = ...，这会触发 "one of the variables needed for gradient
        computation has been modified by an inplace operation"。改为构造完整的
        上下界张量 lo_vec/hi_vec，对整个张量做 element-wise STE clamp。

        文献：Bengio et al. (2013) "Estimating or Propagating Gradients
        Through Stochastic Neurons for Conditional Computation"
        """
        x_new = x + delta
        n = x_new.shape[0] if hasattr(x_new, 'shape') else len(x_new)
        if isinstance(x_new, torch.Tensor):
            # 构造 element-wise 上下界；不可变特征或超出范围的索引保持原值
            lo_vec = torch.full_like(x_new, -float('inf'))
            hi_vec = torch.full_like(x_new, float('inf'))
            for idx, (name, lo, hi) in self.MUTABLE_FEATURES.items():
                if idx < n:
                    lo_vec[idx] = float(lo)
                    hi_vec[idx] = float(hi)
            # STE clamp：前向硬截断，反向梯度直通；无原地赋值
            return _straight_through_clamp(x_new, lo_vec, hi_vec)
        else:
            x_new = np.asarray(x_new, dtype=np.float64)
            for idx, (name, lo, hi) in self.MUTABLE_FEATURES.items():
                if idx < n:
                    x_new[idx] = float(np.clip(x_new[idx], float(lo), float(hi)))
            return x_new

    def _intervention_cost(self, delta):
        """计算干预向量的L1加权成本（numpy版本，用于非梯度路径）"""
        cost = 0.0
        for idx, (name, lo, hi) in self.MUTABLE_FEATURES.items():
            if idx < len(delta):
                unit_cost = self.INTERVENTION_COSTS.get(name, 1.0)
                cost += abs(delta[idx]) * unit_cost
        return float(cost)
    
    def _intervention_cost_tensor(self, delta):
        """计算干预向量的L1加权成本（张量版本，保持可微，用于梯度优化路径）"""
        if not PYTORCH_AVAILABLE:
            return self._intervention_cost(delta.detach().cpu().numpy())
        cost = torch.tensor(0.0, device=delta.device, dtype=delta.dtype)
        for idx, (name, lo, hi) in self.MUTABLE_FEATURES.items():
            if idx < delta.shape[0]:
                unit_cost = self.INTERVENTION_COSTS.get(name, 1.0)
                cost = cost + torch.abs(delta[idx]) * unit_cost
        return cost

    def optimize(self, x, target_risk=0.20, lambda_cost=0.1,
                 max_iterations=200, lr=0.05, tolerance=1e-4,
                 verbose=False):
        """通过梯度下降求解最小成本反事实

        参数：
            x: 特征向量 numpy [22], 不可变维置0
            target_risk: 目标风险阈值
            lambda_cost: 成本正则化权重
            max_iterations: 最大迭代次数
            lr: 学习率
            tolerance: 收敛容差
            verbose: 是否打印迭代信息

        返回：
            dict: {
                'delta': 最优delta向量,
                'x_counterfactual': 反事实特征向量,
                'initial_risk': 初始风险,
                'final_risk': 最终风险,
                'total_cost': 总干预成本,
                'iterations': 迭代次数,
                'converged': 是否收敛,
                'intervention_details': 各特征干预详情,
                'pareto_points': 不同lambda下的帕累托前沿
            }
        """
        if not PYTORCH_AVAILABLE:
            return {'error': 'PyTorch不可用，无法进行梯度优化'}

        # 防御性处理：若 x 已是 tensor 则 detach+clone，否则转换
        # x_tensor 是常量叶子（requires_grad=False），不参与梯度计算
        if isinstance(x, torch.Tensor):
            x_tensor = x.detach().clone().to(device=self.device, dtype=torch.float32)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        delta = torch.zeros_like(x_tensor, requires_grad=True)

        mutable_mask = torch.ones_like(x_tensor)
        for idx in self.IMMUTABLE_FEATURES:
            if idx < len(mutable_mask):
                mutable_mask[idx] = 0.0

        initial_risk = self._predict_risk(x_tensor)
        best_delta = None
        best_final_risk = float('inf')
        best_cost = float('inf')

        # 预初始化 iteration，避免 max_iterations=0 时循环不执行导致下方
        # 'iterations': iteration + 1 抛 NameError
        iteration = -1
        for iteration in range(max_iterations):
            if delta.grad is not None:
                delta.grad.zero_()

            # delta 始终留在计算图中，无需 detach+requires_grad 拼补：
            #   delta (leaf, requires_grad)
            #     -> delta_masked = delta * mutable_mask      (in graph)
            #     -> x_modified = _clip_to_feasible(x_tensor, delta_masked)
            #          内部用 STE clamp（前向硬截断，反向梯度直通）(in graph)
            #     -> x_combined = x_modified * mask + x_tensor * (1-mask)  (in graph)
            #     -> risk = risk_predictor(x_combined)         (in graph)
            #     -> loss = risk_penalty + lambda * cost       (in graph)
            #   loss.backward() 自动回传梯度到 delta，无需手工拼补。
            delta_masked = delta * mutable_mask
            x_modified = self._clip_to_feasible(x_tensor, delta_masked)

            if not isinstance(x_modified, torch.Tensor):
                # _clip_to_feasible 在 tensor 输入下必返回 tensor；
                # 此防御仅在意外类型时触发
                x_modified = torch.tensor(x_modified, dtype=torch.float32,
                                           device=self.device)

            x_combined = x_modified * mutable_mask + \
                         x_tensor * (1 - mutable_mask)

            risk = self._predict_risk(x_combined, differentiable=True)

            risk_penalty = torch.relu(risk - target_risk)

            cost = self._intervention_cost_tensor(delta_masked)

            loss = risk_penalty + lambda_cost * cost

            if float(risk.item()) <= target_risk + tolerance and float(cost.item()) < best_cost:
                best_delta = delta_masked.detach().clone()
                best_final_risk = risk.item()
                best_cost = cost.item()

            # loss.backward() 自动回传梯度到 delta（通过 STE clamp 直通）
            loss.backward()

            # 标准 SGD 更新：delta.grad 已由 autograd 正确填充
            with torch.no_grad():
                if delta.grad is not None:
                    delta -= lr * delta.grad
                delta *= mutable_mask  # 确保不可变特征的 delta 始终为零

            if iteration % 50 == 0 and verbose:
                pass

            if float(risk.item()) <= target_risk:
                x_try = self._clip_to_feasible(
                    x_tensor.detach().cpu().numpy(),
                    delta.detach().cpu().numpy())
                risk_check = self._predict_risk(
                    torch.tensor(x_try, dtype=torch.float32, device=self.device))
                if float(risk_check.item()) <= target_risk + tolerance and \
                   best_cost < float('inf') and \
                   mutable_mask.sum().item() > 0:
                    break

        if best_delta is None:
            best_delta = delta.detach().clone() * mutable_mask
            x_final_np = self._clip_to_feasible(
                x_tensor.detach().cpu().numpy(),
                best_delta.detach().cpu().numpy())
            best_final_risk = self._predict_risk(
                torch.tensor(x_final_np, dtype=torch.float32, device=self.device)).item()
            best_cost = self._intervention_cost(best_delta.detach().cpu().numpy())

        x_counterfactual = self._clip_to_feasible(
            x_tensor.detach().cpu().numpy(),
            best_delta.detach().cpu().numpy())

        intervention_details = []
        for idx, (name, lo, hi) in self.MUTABLE_FEATURES.items():
            if idx < len(best_delta):
                d = float(best_delta[idx].item())
                if abs(d) > 1e-4:
                    old_val = float(x[idx])
                    new_val = float(x_counterfactual[idx])
                    unit_cost = self.INTERVENTION_COSTS.get(name, 1.0)
                    intervention_details.append({
                        'feature_idx': idx,
                        'feature_name': name,
                        'original_value': round(old_val, 2),
                        'recommended_value': round(new_val, 2),
                        'delta': round(d, 2),
                        'unit_cost': unit_cost,
                        'total_feature_cost': round(abs(d) * unit_cost, 2),
                        'direction': '降低' if d < 0 else '提高'
                    })

        intervention_details.sort(key=lambda d: d['total_feature_cost'], reverse=True)

        converged = best_final_risk <= target_risk + 0.01

        return {
            'delta': best_delta.detach().cpu().numpy(),
            'x_counterfactual': x_counterfactual,
            'initial_risk': float(initial_risk),
            'final_risk': float(best_final_risk),
            'risk_reduction': float(initial_risk - best_final_risk),
            'total_cost': float(best_cost),
            'iterations': iteration + 1,
            'converged': converged,
            'intervention_details': intervention_details,
        }

    def _predict_risk(self, x_tensor, differentiable=False, graph_data=None):
        """使用风险预测器预测风险。

        risk_predictor 两次调用均失败时的回退策略：
          - 若 self.baseline_risk 已设置（训练集均值风险）：返回常量张量，
            梯度为零，正确信号梯度下降该方向无收益。
          - 若 self.baseline_risk 为 None：抛出 RuntimeError，
            由上层调用方决定跳过该样本或使用全局基线风险。
        不再使用 torch.sigmoid(x.sum()) * 0.3 的简化回退，
        因为它会引入依赖输入特征的虚假梯度信号，误导优化方向。

        参数：
            x_tensor: 输入特征张量
            differentiable: 是否需要梯度
            graph_data: 可选图结构数据，包含 edge_index 和 edge_attr。
                        传入时使用真实图结构，否则回退到虚拟自环边
                        （GNN 退化为 MLP，图结构信息丢失）。
        """
        import logging
        logger = logging.getLogger('tb_risk.optimizer')

        # Use eval mode for deterministic gradients during counterfactual optimization.
        # Dropout noise would interfere with gradient descent convergence.
        self.risk_predictor.eval()

        # 构建边索引：优先使用真实图结构，否则回退到虚拟自环
        if graph_data is not None:
            edge_index = graph_data.get('edge_index',
                torch.zeros(2, 1, dtype=torch.long, device=self.device))
        else:
            edge_index = torch.zeros(2, 1, dtype=torch.long, device=self.device)
            logger.debug(
                "未提供 graph_data，GNN 退化为 MLP（自环边），"
                "反事实优化无法正确评估接触网络特征变化对风险的影响")

        def _call_predictor(x_input):
            """尝试多种调用约定，返回 (risk, success, error_chain)

            优先使用真实图结构；若无 graph_data，回退到虚拟自环边
            （GNN 退化为 MLP，图结构信息丢失）。
            """
            errors = []
            try:
                risk, _ = self.risk_predictor(x_input, edge_index)
                return risk, True, errors
            except Exception as e:
                errors.append(f"(x, edge_index): {type(e).__name__}: {e}")
            try:
                risk = self.risk_predictor(x_input)
                return risk, True, errors
            except Exception as e:
                errors.append(f"(x): {type(e).__name__}: {e}")
            return None, False, errors

        if differentiable:
            x_input = x_tensor.unsqueeze(0)
            risk, ok, errors = _call_predictor(x_input)
            if ok:
                return risk.squeeze()
            # 两次调用均失败：记录错误链
            logger.warning(
                "risk_predictor 两次调用均失败: %s", " | ".join(errors))
            if self.baseline_risk is not None:
                # 常量回退：梯度为零（detach 切断计算图）
                logger.warning(
                    "回退到 baseline_risk=%.4f（常量，梯度为零）",
                    self.baseline_risk)
                return torch.full(
                    (1,), self.baseline_risk,
                    dtype=torch.float32, device=self.device).squeeze()
            raise RuntimeError(
                "risk_predictor 两次调用均失败且未设置 baseline_risk: "
                + " | ".join(errors))
        else:
            with torch.no_grad():
                x_input = x_tensor.unsqueeze(0)
                risk, ok, errors = _call_predictor(x_input)
                if ok:
                    return risk.squeeze()
                logger.warning(
                    "risk_predictor 两次调用均失败: %s", " | ".join(errors))
                if self.baseline_risk is not None:
                    logger.warning(
                        "回退到 baseline_risk=%.4f（常量）", self.baseline_risk)
                    return torch.full(
                        (1,), self.baseline_risk,
                        dtype=torch.float32, device=self.device).squeeze()
                raise RuntimeError(
                    "risk_predictor 两次调用均失败且未设置 baseline_risk: "
                    + " | ".join(errors))

    def generate_pareto_frontier(self, x, target_risk=0.20,
                                  lambda_values=None, verbose=False):
        """生成帕累托前沿：遍历不同成本权重求解

        参数：
            x: 特征向量
            target_risk: 目标风险阈值
            lambda_values: 成本权重列表

        返回：
            list: 帕累托前沿点 [{'lambda_cost', 'final_risk', 'total_cost', ...}]
        """
        if lambda_values is None:
            lambda_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

        pareto_points = []
        for lam in lambda_values:
            result = self.optimize(x, target_risk=target_risk,
                                    lambda_cost=lam, verbose=verbose)
            if 'error' not in result:
                pareto_points.append({
                    'lambda_cost': lam,
                    'final_risk': result['final_risk'],
                    'total_cost': result['total_cost'],
                    'converged': result['converged'],
                    'intervention_count': len(result['intervention_details']),
                    'intervention_details': result['intervention_details'],
                })

        pareto_points.sort(key=lambda p: p['total_cost'])

        non_dominated = []
        for p in pareto_points:
            dominated = False
            for nd in non_dominated:
                if nd['final_risk'] <= p['final_risk'] and nd['total_cost'] <= p['total_cost']:
                    if nd['final_risk'] < p['final_risk'] or nd['total_cost'] < p['total_cost']:
                        dominated = True
                        break
            if not dominated:
                non_dominated.append(p)

        return non_dominated