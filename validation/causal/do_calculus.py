#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""do-calculus 后门调整 — 将观察关联转化为干预效应估计

do-calculus（Pearl 1995）是因果推断的理论核心。它定义了 do(X=x)（干预操作）
与条件概率 P(Y|X=x)（观察）的本质区别：
- P(Y|X=x)：观察到 X=x 时 Y 的分布（受混杂影响）
- P(Y|do(X=x))：强制设定 X=x 后 Y 的分布（去除混杂，反映因果效应）

后门调整公式（Backdoor Adjustment, Pearl 1995, Thm 3.2.2）：
    P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) · P(Z=z)
其中 Z 是满足后门准则的充分调整集（由 DAG 自动识别）。

本模块通过 g-computation（回归调整）实现后门调整：
1. 基于 DAG 求解调整集 Z
2. 拟合结局模型 μ(x,z) = E[Y | X=x, Z=z]
3. 对每个样本，分别预测 X=x₁ 与 X=x₀ 下的 Y，取平均差值即 ATE

这是"预测概率→干预效应估计"升级的关键实现。
SHAP 只能说"特征重要性"，do-calculus 能说"若实施干预 X=x，结局将改变多少"。

文献支撑：
- Pearl J. Causal diagrams for empirical research. JRSS-B 57(3):669-688, 1995.
- Pearl J. Causality, 2009. §3.3 后门准则, §3.4 do-calculus
- Robins JM. A new approach to causal inference in mortality studies.
  Math Model 7(9-12):1393-1512, 1986. (g-computation)
- Hernán MA, Robins JM. Causal Inference: What If, 2020. §13-15

依赖：scikit-learn（结局模型）。numpy。DAG 模块。
"""
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    LinearRegression = None
    RandomForestRegressor = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from .dag import CausalDAG

LOGGER = logging.getLogger("tb_risk.validation.causal.do_calculus")


class DoCalculusEstimator:
    """do-calculus 后门调整估计器。

    基于 DAG 自动识别调整集，通过 g-computation 估计干预效应。

    用法：
        from .tb_dag import build_tb_dag
        dag = build_tb_dag()
        estimator = DoCalculusEstimator(dag)
        result = estimator.estimate_ate(
            data, treatment='cumulative_exposure_binary',
            outcome='disease_probability',
            treatment_value=1, control_value=0)
        # result = {'ate': 0.15, 'ci': [...], 'adjustment_set': {...}, ...}

    参数：
        dag: CausalDAG 实例（用于识别调整集）
        model_type: str, 结局模型类型 ('linear' 或 'forest')
        random_state: int
    """

    def __init__(self, dag: CausalDAG, model_type: str = 'linear',
                 random_state: int = 42):
        self.dag = dag
        self.model_type = model_type
        self.random_state = random_state
        self._outcome_model = None
        self._scaler = None
        self._adjustment_set: Optional[set] = None

    # ------------------------------------------------------------------
    # 调整集识别
    # ------------------------------------------------------------------
    def identify_adjustment_set(self, treatment: str,
                                outcome: str) -> set:
        """基于 DAG 后门准则识别充分调整集。

        若 DAG 中存在 treatment→outcome 的有向路径，则因果效应可识别。
        调整集自动求解；若失败则抛出 RuntimeError。
        """
        adj = self.dag.find_adjustment_set(treatment, outcome)
        if adj is None:
            raise RuntimeError(
                f"无法从 DAG 自动求解调整集 (treatment={treatment}, "
                f"outcome={outcome})。请检查图结构或手动指定 adjustment_set。")
        self._adjustment_set = adj
        LOGGER.info("调整集识别: %s → %s, Z = %s",
                    treatment, outcome, sorted(adj))
        return adj

    # ------------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------------
    def _prepare_data(self, data, treatment: str, outcome: str,
                      adjustment_set: Sequence[str]
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """提取 (X=treatment, Z=adjustment, Y=outcome) 数组。"""
        if not NUMPY_AVAILABLE:
            raise RuntimeError("do-calculus 需要 numpy")
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            T = data[treatment].values.astype(float)
            Y = data[outcome].values.astype(float)
            Z = data[list(adjustment_set)].values.astype(float) \
                if adjustment_set else np.empty((len(T), 0))
            return T, Z, Y

        if isinstance(data, (list, tuple)):
            T = np.array([float(r.get(treatment, 0.0)) for r in data])
            Y = np.array([float(r.get(outcome, 0.0)) for r in data])
            if adjustment_set:
                Z = np.array(
                    [[float(r.get(c, 0.0)) for c in adjustment_set]
                     for r in data], dtype=float)
            else:
                Z = np.empty((len(T), 0))
            return T, Z, Y

        raise TypeError(f"不支持的数据类型: {type(data)}")

    # ------------------------------------------------------------------
    # 结局模型拟合
    # ------------------------------------------------------------------
    def _fit_outcome_model(self, T: np.ndarray, Z: np.ndarray,
                           Y: np.ndarray, binary_outcome: bool = False):
        """拟合结局模型 μ(t, z) = E[Y | T=t, Z=z]。"""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("do-calculus 结局模型需要 scikit-learn")
        # 特征矩阵 [T, Z]
        X = np.column_stack([T, Z]) if Z.shape[1] > 0 else T.reshape(-1, 1)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        if binary_outcome:
            model = LogisticRegression(
                random_state=self.random_state, max_iter=1000)
        elif self.model_type == 'forest':
            model = RandomForestRegressor(
                n_estimators=100, random_state=self.random_state,
                max_depth=8)
        else:
            model = LinearRegression()

        model.fit(X_scaled, Y)
        self._outcome_model = model
        return model

    def _predict_outcome(self, T_values: np.ndarray,
                         Z: np.ndarray) -> np.ndarray:
        """在给定 T 值与 Z 矩阵下预测结局。"""
        if self._outcome_model is None or self._scaler is None:
            raise RuntimeError("结局模型未拟合")
        # 广播 T_values 与 Z
        if Z.shape[1] > 0:
            X = np.column_stack([T_values, Z])
        else:
            X = T_values.reshape(-1, 1)
        X_scaled = self._scaler.transform(X)
        if hasattr(self._outcome_model, 'predict_proba'):
            # 二分类结局：返回 P(Y=1)
            return self._outcome_model.predict_proba(X_scaled)[:, 1]
        return self._outcome_model.predict(X_scaled)

    # ------------------------------------------------------------------
    # ATE 估计（g-computation）
    # ------------------------------------------------------------------
    def estimate_ate(self, data, treatment: str, outcome: str,
                     treatment_value: float = 1.0,
                     control_value: float = 0.0,
                     adjustment_set: Optional[Sequence[str]] = None,
                     binary_outcome: bool = False,
                     n_bootstrap: int = 200) -> Dict:
        """估计平均处理效应（ATE）via 后门调整 + g-computation。

        ATE = E[Y(do(T=treatment_value))] - E[Y(do(T=control_value))]
            = (1/n) Σ_i [μ(treatment_value, z_i) - μ(control_value, z_i)]

        参数：
            data: DataFrame 或 list[dict]
            treatment: str, 干预变量列名
            outcome: str, 结局变量列名
            treatment_value: 干预值（默认 1）
            control_value: 对照值（默认 0）
            adjustment_set: 手动指定调整集；None 则从 DAG 自动识别
            binary_outcome: bool, 结局是否二分类
            n_bootstrap: int, Bootstrap 置信区间次数

        返回：
            dict: {ate, ci, adjustment_set, risk_difference, ...}
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("do-calculus 需要 numpy")
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("do-calculus 需要 scikit-learn")

        # 调整集识别
        if adjustment_set is None:
            adjustment_set = self.identify_adjustment_set(treatment, outcome)
        else:
            adjustment_set = set(adjustment_set)
            self._adjustment_set = adjustment_set

        T, Z, Y = self._prepare_data(data, treatment, outcome, adjustment_set)

        # 拟合结局模型
        self._fit_outcome_model(T, Z, Y, binary_outcome=binary_outcome)

        # g-computation: 对每个样本，分别预测 treatment 与 control 下的 Y
        ate, mu_treated, mu_control = self._g_compute(
            Z, treatment_value, control_value)

        # Bootstrap CI（_bootstrap_ate 会在重采样上重拟合模型，覆写 self 上的模型状态；
        # 保存全数据模型并在 bootstrap 后恢复，避免估计器残留最后一次重采样模型）
        saved_model = self._outcome_model
        saved_scaler = self._scaler
        try:
            ci = self._bootstrap_ate(
                data, treatment, outcome, adjustment_set,
                treatment_value, control_value, binary_outcome, n_bootstrap)
        finally:
            self._outcome_model = saved_model
            self._scaler = saved_scaler

        return {
            'ate': float(ate),
            'ci': ci,
            'adjustment_set': sorted(adjustment_set),
            'treatment': treatment,
            'outcome': outcome,
            'treatment_value': float(treatment_value),
            'control_value': float(control_value),
            'mean_potential_outcome_treated': float(np.mean(mu_treated)),
            'mean_potential_outcome_control': float(np.mean(mu_control)),
            'n_samples': int(len(T)),
            'method': 'backdoor_adjustment_g_computation',
            'note': ('ATE via 后门调整 (Pearl 1995) + g-computation (Robins 1986)；'
                     '调整集由 DAG 后门准则识别；'
                     '此为因果效应估计，区别于 SHAP 的相关性分析'),
        }

    def _g_compute(self, Z: np.ndarray, treatment_value: float,
                   control_value: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """g-computation：估计潜在结局均值与 ATE。"""
        n = Z.shape[0]
        t_vec = np.full(n, treatment_value)
        c_vec = np.full(n, control_value)
        mu_treated = self._predict_outcome(t_vec, Z)
        mu_control = self._predict_outcome(c_vec, Z)
        ate = float(np.mean(mu_treated - mu_control))
        return ate, mu_treated, mu_control

    def _bootstrap_ate(self, data, treatment: str, outcome: str,
                       adjustment_set: set, treatment_value: float,
                       control_value: float, binary_outcome: bool,
                       n_bootstrap: int) -> List[float]:
        """Bootstrap 估计 ATE 置信区间。"""
        if n_bootstrap <= 0:
            return [float('nan'), float('nan')]
        T, Z, Y = self._prepare_data(data, treatment, outcome, adjustment_set)
        n = len(T)
        rng = np.random.RandomState(self.random_state)
        boot_ates = np.empty(n_bootstrap)

        for b in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            try:
                self._fit_outcome_model(
                    T[idx], Z[idx], Y[idx], binary_outcome=binary_outcome)
                ate_b, _, _ = self._g_compute(
                    Z[idx], treatment_value, control_value)
                boot_ates[b] = ate_b
            except (ValueError, RuntimeError):
                boot_ates[b] = np.nan

        valid = boot_ates[~np.isnan(boot_ates)]
        if len(valid) == 0:
            return [float('nan'), float('nan')]
        return [float(np.percentile(valid, 2.5)),
                float(np.percentile(valid, 97.5))]

    # ------------------------------------------------------------------
    # 干预效应对比（连续干预）
    # ------------------------------------------------------------------
    def intervention_effect(self, data, treatment: str, outcome: str,
                            intervention_values: Sequence[float],
                            adjustment_set: Optional[Sequence[str]] = None,
                            binary_outcome: bool = False) -> Dict:
        """对比不同干预水平下的潜在结局（剂量-反应）。

        回答："若将暴露分别设为 v1, v2, ..., 结局会如何变化？"

        参数：
            intervention_values: list[float], 一组干预值（如 [0, 20, 40, 80]）

        返回：
            dict: {dose_response: [(v, mean_y), ...], marginal_effects: [...]}
        """
        if not NUMPY_AVAILABLE or not SKLEARN_AVAILABLE:
            raise RuntimeError("do-calculus 需要 numpy 和 scikit-learn")

        if adjustment_set is None:
            adjustment_set = self.identify_adjustment_set(treatment, outcome)
        else:
            adjustment_set = set(adjustment_set)

        T, Z, Y = self._prepare_data(data, treatment, outcome, adjustment_set)
        self._fit_outcome_model(T, Z, Y, binary_outcome=binary_outcome)

        n = Z.shape[0]
        dose_response: List[Tuple[float, float]] = []
        for v in intervention_values:
            mu = self._predict_outcome(np.full(n, v), Z)
            dose_response.append((float(v), float(np.mean(mu))))

        # 边际效应：相邻干预水平的结局差
        marginal: List[Tuple[float, float]] = []
        for i in range(1, len(dose_response)):
            dv = dose_response[i][0] - dose_response[i - 1][0]
            dy = dose_response[i][1] - dose_response[i - 1][1]
            marginal.append((float(dv), float(dy)))

        return {
            'dose_response': dose_response,
            'marginal_effects': marginal,
            'adjustment_set': sorted(adjustment_set),
            'treatment': treatment,
            'outcome': outcome,
            'note': ('剂量-反应曲线：P(Y|do(T=v)) 在不同 v 下的潜在结局均值；'
                     'marginal_effects 为相邻水平的边际变化'),
        }

    # ------------------------------------------------------------------
    # 条件干预效应（CATE）
    # ------------------------------------------------------------------
    def conditional_effect(self, data, treatment: str, outcome: str,
                           subgroup_col: str,
                           treatment_value: float = 1.0,
                           control_value: float = 0.0,
                           adjustment_set: Optional[Sequence[str]] = None,
                           binary_outcome: bool = False) -> Dict:
        """估计亚组条件平均处理效应（CATE）。

        用于效应修饰因子分析：不同亚组（如年龄组、糖尿病组）的干预效应差异。
        """
        if adjustment_set is None:
            adjustment_set = self.identify_adjustment_set(treatment, outcome)
        else:
            adjustment_set = set(adjustment_set)

        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            subgroups = data[subgroup_col].unique()
            results = {}
            for sg in subgroups:
                sub_data = data[data[subgroup_col] == sg]
                if len(sub_data) < 5:
                    results[str(sg)] = {'ate': float('nan'),
                                         'n': int(len(sub_data))}
                    continue
                try:
                    r = self.estimate_ate(
                        sub_data, treatment, outcome, treatment_value,
                        control_value, adjustment_set, binary_outcome,
                        n_bootstrap=0)
                    results[str(sg)] = {'ate': r['ate'],
                                         'n': int(r['n_samples'])}
                except (ValueError, RuntimeError) as e:
                    results[str(sg)] = {'error': str(e)}
            return {
                'subgroup_col': subgroup_col,
                'subgroup_effects': results,
                'note': ('CATE: 条件平均处理效应；'
                         '亚组间差异提示效应修饰因子存在'),
            }

        # list[dict]
        sub_data = {}
        for row in data:
            sg = row.get(subgroup_col)
            sub_data.setdefault(sg, []).append(row)
        results = {}
        for sg, rows in sub_data.items():
            if len(rows) < 5:
                results[str(sg)] = {'ate': float('nan'),
                                     'n': len(rows)}
                continue
            try:
                r = self.estimate_ate(
                    rows, treatment, outcome, treatment_value,
                    control_value, adjustment_set, binary_outcome,
                    n_bootstrap=0)
                results[str(sg)] = {'ate': r['ate'],
                                     'n': int(r['n_samples'])}
            except (ValueError, RuntimeError) as e:
                results[str(sg)] = {'error': str(e)}
        return {
            'subgroup_col': subgroup_col,
            'subgroup_effects': results,
            'note': 'CATE: 条件平均处理效应；亚组间差异提示效应修饰因子存在',
        }
