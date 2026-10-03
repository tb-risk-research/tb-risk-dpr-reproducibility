#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""倾向评分匹配（PSM）— 评估干预的因果效应

PSM 通过匹配干预组与对照组在"接受干预倾向"上的相似性，构造类实验场景，
从而估计干预的平均处理效应（ATE/ATT）。它解决的是"观察数据中干预非随机
分配"导致的混杂偏倚问题。

与 SHAP 的区别：SHAP 解释的是模型特征贡献（相关性），PSM 估计的是
"若让相似的人接受/不接受干预，结局差异是多少"（因果效应）。

核心步骤：
1. 估计倾向评分 e(x) = P(treatment=1 | X) via 逻辑回归
2. 基于倾向评分匹配（最近邻 + 卡钳）
3. 估计 ATT = E[Y(1) - Y(0) | T=1]
4. 平衡诊断：标准化均值差（SMD）匹配前后对比

文献支撑：
- Rosenbaum PR, Rubin DB. The central role of the propensity score
  in observational studies for causal effects. Biometrika 70(1):41-55, 1983.
- Austin PC. An introduction to propensity score methods for reducing
  the effects of confounding. Multivar Behav Res 46(3):399-424, 2011.
- Stuart EA. Matching methods for causal inference. Stat Sci 25(1):1-21, 2010.

依赖：scikit-learn（项目已有）。不可用时降级为简单实现。
"""
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    LogisticRegression = None
    StandardScaler = None
    NearestNeighbors = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

LOGGER = logging.getLogger("tb_risk.validation.causal.psm")

# SMD 平衡阈值（Austin 2011 推荐 |SMD| < 0.1 视为平衡）
SMD_BALANCE_THRESHOLD = 0.1


class PropensityScoreMatcher:
    """倾向评分匹配器。

    用法：
        matcher = PropensityScoreMatcher(
            treatment_col='high_exposure',
            covariate_cols=['age', 'bcg_vaccine', 'past_illness'],
            caliper=0.2)
        result = matcher.estimate_att(data, outcome_col='disease_probability')
        # result = {'att': 0.12, 'ci': [0.05, 0.19], 'smd_before': {...}, 'smd_after': {...}}

    参数：
        treatment_col: str, 干预变量列名（二值 0/1）
        covariate_cols: list[str], 协变量列名（用于估计倾向评分）
        caliper: float, 卡钳宽度（倾向评分 logitscale 上的标准差倍数，
                 Rosenbaum 1985 建议 0.2）
        method: str, 匹配方法 ('nearest' 最近邻)
        random_state: int, 随机种子
    """

    def __init__(self, treatment_col: str, covariate_cols: Sequence[str],
                 caliper: float = 0.2, method: str = 'nearest',
                 random_state: int = 42):
        if not treatment_col:
            raise ValueError("treatment_col 不能为空")
        if not covariate_cols:
            raise ValueError("covariate_cols 不能为空")
        self.treatment_col = treatment_col
        self.covariate_cols = list(covariate_cols)
        self.caliper = caliper
        self.method = method
        self.random_state = random_state

    # ------------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------------
    def _to_arrays(self, data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """将 DataFrame 或 list[dict] 转换为 (X, T, indices) 数组。"""
        if not NUMPY_AVAILABLE:
            raise RuntimeError("PSM 需要 numpy，但当前不可用")
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            X = data[self.covariate_cols].values.astype(float)
            T = data[self.treatment_col].values.astype(int)
            return X, T, np.arange(len(data))

        # list[dict]
        if isinstance(data, (list, tuple)):
            X = np.array(
                [[float(row.get(c, 0.0)) for c in self.covariate_cols]
                 for row in data], dtype=float)
            T = np.array(
                [int(row.get(self.treatment_col, 0)) for row in data],
                dtype=int)
            return X, T, np.arange(len(data))

        raise TypeError(f"不支持的数据类型: {type(data)}；期望 DataFrame 或 list[dict]")

    # ------------------------------------------------------------------
    # 倾向评分估计
    # ------------------------------------------------------------------
    def estimate_propensity(self, data) -> np.ndarray:
        """估计倾向评分 e(x) = P(T=1 | X)。

        返回：
            np.ndarray: 每个样本的倾向评分（0~1）
        """
        if not SKLEARN_AVAILABLE:
            LOGGER.warning("sklearn 不可用，降级为协变量均值近似倾向评分")
            return self._fallback_propensity(data)

        X, T, _ = self._to_arrays(data)

        # 标准化协变量
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 逻辑回归估计倾向评分
        model = LogisticRegression(
            random_state=self.random_state, max_iter=1000,
            solver='lbfgs')
        # 处理单一类别情形
        if len(np.unique(T)) < 2:
            LOGGER.warning("干预变量只有一个类别，倾向评分恒为 %.2f",
                           float(T.mean()))
            return np.full(len(T), float(T.mean()))

        model.fit(X_scaled, T)
        propensity = model.predict_proba(X_scaled)[:, 1]
        # 裁剪到 [eps, 1-eps] 避免极端值
        eps = 1e-6
        return np.clip(propensity, eps, 1 - eps)

    def _fallback_propensity(self, data) -> np.ndarray:
        """无 sklearn 时的降级：用协变量标准化求和的 sigmoid 近似。"""
        X, T, _ = self._to_arrays(data)
        if X.shape[0] == 0:
            return np.array([])
        # 标准化后线性组合
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        z = (X - mean) / std
        # 简单等权线性组合
        score = z.sum(axis=1) / max(z.shape[1], 1)
        prop = 1.0 / (1.0 + np.exp(-score))
        eps = 1e-6
        return np.clip(prop, eps, 1 - eps)

    # ------------------------------------------------------------------
    # 匹配
    # ------------------------------------------------------------------
    def match(self, data) -> List[Tuple[int, int]]:
        """执行倾向评分匹配。

        返回：
            list[(treated_idx, control_idx)]: 匹配对索引列表
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("PSM 需要 numpy")
        propensity = self.estimate_propensity(data)
        _, T, indices = self._to_arrays(data)

        treated_mask = T == 1
        control_mask = T == 0
        treated_idx = np.where(treated_mask)[0]
        control_idx = np.where(control_mask)[0]

        if len(treated_idx) == 0 or len(control_idx) == 0:
            LOGGER.warning("干预组或对照组为空，无法匹配 (treated=%d, control=%d)",
                           len(treated_idx), len(control_idx))
            return []

        # 转换到 logit 尺度做匹配（Rosenbaum 1985 推荐）
        prop_treated = propensity[treated_idx]
        prop_control = propensity[control_idx]
        logit_treated = np.log(prop_treated / (1 - prop_treated))
        logit_control = np.log(prop_control / (1 - prop_control))

        # 卡钳阈值（logit 尺度标准差 × caliper）
        pooled_std = math.sqrt(
            (logit_treated.var() * len(logit_treated) +
             logit_control.var() * len(logit_control))
            / max(len(logit_treated) + len(logit_control), 1))
        caliper_abs = self.caliper * pooled_std if pooled_std > 0 else float('inf')

        matches: List[Tuple[int, int]] = []

        if SKLEARN_AVAILABLE and self.method == 'nearest':
            # 用 sklearn NearestNeighbors 做 1:1 最近邻匹配
            nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
            nn.fit(logit_control.reshape(-1, 1))
            distances, neighbors = nn.kneighbors(logit_treated.reshape(-1, 1))
            used_control: set = set()
            for i, (dist, nbr) in enumerate(zip(distances.ravel(),
                                                neighbors.ravel())):
                if dist > caliper_abs:
                    continue
                if nbr in used_control:
                    continue  # 无放回匹配
                used_control.add(nbr)
                matches.append((int(treated_idx[i]), int(control_idx[nbr])))
        else:
            # 纯 numpy 贪心最近邻匹配
            available = list(range(len(control_idx)))
            for i, lt in enumerate(logit_treated):
                if not available:
                    break
                diffs = np.abs(logit_control[available] - lt)
                best = int(np.argmin(diffs))
                if diffs[best] > caliper_abs:
                    continue
                c_idx = available.pop(best)
                matches.append((int(treated_idx[i]), int(control_idx[c_idx])))

        LOGGER.info("PSM 匹配完成: %d/%d 干预组样本获得匹配",
                    len(matches), len(treated_idx))
        return matches

    # ------------------------------------------------------------------
    # 效应估计
    # ------------------------------------------------------------------
    def estimate_att(self, data, outcome_col: str) -> Dict:
        """估计 ATT（受干预者平均处理效应）。

        ATT = E[Y(1) - Y(0) | T=1] = mean(Y_treated_matched) - mean(Y_control_matched)

        参数：
            data: DataFrame 或 list[dict]
            outcome_col: str, 结局变量列名

        返回：
            dict: {att, ci, n_matched, n_treated, smd_before, smd_after, balance_achieved}
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("PSM 需要 numpy")
        X, T, _ = self._to_arrays(data)
        # 结局向量
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            Y = data[outcome_col].values.astype(float)
        else:
            Y = np.array(
                [float(row.get(outcome_col, 0.0)) for row in data],
                dtype=float)

        matches = self.match(data)
        if not matches:
            return self._empty_result(len(np.where(T == 1)[0]), "无有效匹配对")

        treated_idx = np.array([m[0] for m in matches])
        control_idx = np.array([m[1] for m in matches])

        y_treated = Y[treated_idx]
        y_control = Y[control_idx]
        att = float(np.mean(y_treated - y_control))

        # 配对 Bootstrap 置信区间
        ci = self._paired_bootstrap_ci(y_treated, y_control)

        # 平衡诊断
        smd_before = self._compute_smd(X, T)
        # 匹配后样本的协变量矩阵与干预向量
        X_matched = np.vstack([X[treated_idx], X[control_idx]])
        T_matched = np.concatenate([np.ones(len(treated_idx)),
                                    np.zeros(len(control_idx))])
        smd_after = self._compute_smd(X_matched, T_matched)

        max_smd_after = max((abs(v) for v in smd_after.values()),
                            default=0.0)
        return {
            'att': att,
            'ci': ci,
            'n_matched': len(matches),
            'n_treated': int(np.sum(T == 1)),
            'n_control': int(np.sum(T == 0)),
            'smd_before': smd_before,
            'smd_after': smd_after,
            'balance_achieved': max_smd_after < SMD_BALANCE_THRESHOLD,
            'method': self.method,
            'caliper': self.caliper,
            'note': ('ATT: 受干预者平均处理效应; '
                     f'CI via paired bootstrap (n=1000); '
                     f'平衡判定阈值 |SMD|<{SMD_BALANCE_THRESHOLD}'),
        }

    def _paired_bootstrap_ci(self, y_treated: np.ndarray,
                             y_control: np.ndarray,
                             n_bootstrap: int = 1000,
                             alpha: float = 0.05) -> List[float]:
        """配对 Bootstrap 置信区间。"""
        if not NUMPY_AVAILABLE:
            return [float('nan'), float('nan')]
        n = len(y_treated)
        if n == 0:
            return [float('nan'), float('nan')]
        rng = np.random.RandomState(self.random_state)
        diffs = y_treated - y_control
        boot_atts = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            sample = rng.randint(0, n, size=n)
            boot_atts[b] = np.mean(diffs[sample])
        return [float(np.percentile(boot_atts, 100 * alpha / 2)),
                float(np.percentile(boot_atts, 100 * (1 - alpha / 2)))]

    def _compute_smd(self, X: np.ndarray, T: np.ndarray) -> Dict[str, float]:
        """计算每个协变量的标准化均值差（SMD）。

        SMD = (mean_treated - mean_control) / sqrt((var_treated + var_control)/2)
        """
        smd: Dict[str, float] = {}
        treated = X[T == 1]
        control = X[T == 0]
        for j, name in enumerate(self.covariate_cols):
            mt = treated[:, j].mean() if len(treated) > 0 else 0.0
            mc = control[:, j].mean() if len(control) > 0 else 0.0
            vt = treated[:, j].var() if len(treated) > 1 else 0.0
            vc = control[:, j].var() if len(control) > 1 else 0.0
            denom = math.sqrt((vt + vc) / 2.0)
            smd[name] = float((mt - mc) / denom) if denom > 1e-12 else 0.0
        return smd

    def _empty_result(self, n_treated: int, reason: str) -> Dict:
        return {
            'att': float('nan'),
            'ci': [float('nan'), float('nan')],
            'n_matched': 0,
            'n_treated': n_treated,
            'n_control': 0,
            'smd_before': {},
            'smd_after': {},
            'balance_achieved': False,
            'note': f'PSM 未执行: {reason}',
        }
