#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型验证与可靠性指标 — 训练后"验证阶段"模块

在 ML 训练流程之后新增的验证阶段，产出可评审的"模型验证报告"。核心解决三件事：

1. **数据划分防泄漏**：按 训练/验证/测试 三层随机划分；若提供时间字段则改用
   **时间外验证**（按时间先后切分，训练用历史、测试用未来），避免未来信息泄漏。
2. **判别 + 校准指标**：判别指标用 ROC-AUC / PR-AUC；校准指标用校准曲线、
   Brier 分数、Hosmer-Lemeshow 检验。将模型输出的概率分数与实际转归标签对比，
   若概率系统性偏高/偏低（方向性偏差）则触发概率重校准（Platt 缩放 / 保序回归）。
3. **交叉验证报告**：在训练集上跑分层 K 折交叉验证，报告平均性能与标准差。

所有指标汇总为结构化"模型验证报告"，供评审页展示。

文献支撑：
- Hosmer DW, Lemeshow S. Applied Logistic Regression (2nd ed). Wiley, 2000.
- Platt J. Probabilistic Outputs for Support Vector Machines. 1999.
- Zadrozny B, Elkan C. Transforming Classifier Scores into Accurate
  Multiclass Probability Estimates. KDD 2002.
"""
import datetime
import logging

import numpy as np

LOGGER = logging.getLogger("tb_risk.ml")

# 条件导入（与主类保持一致）
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 brier_score_loss)
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from sklearn.calibration import CalibratedClassifierCV
    _CALIBRATOR_AVAILABLE = True
except ImportError:
    _CALIBRATOR_AVAILABLE = False

try:
    import scipy.stats as _stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

NUMPY_AVAILABLE = True

# 预测输入特征名对齐（LightGBM numpy 拟合也会记录自动特征名，numpy 预测
# 触发 sklearn UserWarning；详见 evaluation._aligned_feature_frame docstring）
from .evaluation import _aligned_feature_frame  # noqa: E402


# ============================================================================
# 1. 数据划分：训练/验证/测试 三层 + 时间外验证
# ============================================================================

def split_train_val_test(X, y, test_size=0.2, val_size=0.2,
                         random_state=42, stratify=True, groups=None):
    """将数据划分为 训练/验证/测试 三层，避免数据泄漏。

    参数：
        groups (array-like|None): 样本所属组（如 household id）。提供时
            启用**组感知划分**——同组样本不跨集合（家庭接触者特征/标签
            强相关，HomeACF/PACTS 户级 cluster CV 实证；2026-09-05 地基
            审计：随机划分下"训练时见过同户样本"构成泄漏）。组级近似
            分层：按"组内是否含阳性"分池等比例切，保持各集合阳性率
            接近总体；None（默认）保持原随机分层行为完全不变。

    返回：
        dict: {'X_train','y_train','X_val','y_val','X_test','y_test',
               'n_train','n_val','n_test',
               'method':'random_three_way'|'group_three_way',
               'n_groups'（组感知时给出组总数）}
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(X)

    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != n:
            raise ValueError('groups 长度 (%d) 与样本数 (%d) 不一致'
                             % (len(groups), n))
        rng = np.random.RandomState(random_state)
        uniq = np.unique(groups)
        # 组级分层键：组内是否含阳性（近似保持各集合阳性率）
        pos_pool = [g for g in uniq if y[groups == g].any()]
        neg_pool = [g for g in uniq if not y[groups == g].any()]

        def _take(pool, frac):
            pool = list(pool)
            rng.shuffle(pool)
            k = int(round(len(pool) * frac))
            return set(pool[:k])

        # 先切测试集（占总组数 test_size），再从剩余切验证集（占剩余 val_size）
        test_g = _take(pos_pool, test_size) | _take(neg_pool, test_size)
        rest_pos = [g for g in pos_pool if g not in test_g]
        rest_neg = [g for g in neg_pool if g not in test_g]
        val_g = _take(rest_pos, val_size) | _take(rest_neg, val_size)
        held_out = test_g | val_g

        idx_all = np.arange(n)
        idx_test = idx_all[np.isin(groups, list(test_g))]
        idx_val = idx_all[np.isin(groups, list(val_g))]
        idx_train = idx_all[~np.isin(groups, list(held_out))]
        return {
            'X_train': X[idx_train], 'y_train': y[idx_train],
            'X_val': X[idx_val], 'y_val': y[idx_val],
            'X_test': X[idx_test], 'y_test': y[idx_test],
            'n_train': int(len(idx_train)), 'n_val': int(len(idx_val)),
            'n_test': int(len(idx_test)), 'method': 'group_three_way',
            'n_groups': int(len(uniq)),
        }

    first_test = int(n * test_size)
    remaining = n - first_test
    first_val = int(remaining * val_size)

    # 先切测试集，再从剩余中切验证集，剩余为训练集
    idx = np.arange(n)
    rng = np.random.RandomState(random_state)
    if stratify and len(np.unique(y)) > 1:
        from sklearn.model_selection import train_test_split
        X_rest, X_test, y_rest, y_test, idx_rest, idx_test = train_test_split(
            X, y, idx, test_size=first_test, random_state=random_state,
            stratify=y)
        X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
            X_rest, y_rest, idx_rest, test_size=first_val,
            random_state=random_state, stratify=y_rest)
    else:
        rng.shuffle(idx)
        idx_test = idx[:first_test]
        idx_val = idx[first_test:first_test + first_val]
        idx_train = idx[first_test + first_val:]
        X_test, y_test = X[idx_test], y[idx_test]
        X_val, y_val = X[idx_val], y[idx_val]
        X_train, y_train = X[idx_train], y[idx_train]

    return {
        'X_train': np.asarray(X_train), 'y_train': np.asarray(y_train),
        'X_val': np.asarray(X_val), 'y_val': np.asarray(y_val),
        'X_test': np.asarray(X_test), 'y_test': np.asarray(y_test),
        'n_train': int(len(y_train)), 'n_val': int(len(y_val)),
        'n_test': int(len(y_test)), 'method': 'random_three_way',
    }


def cluster_bootstrap_metric_ci(oof_scores, y, groups, metric='auroc',
                                n_bootstrap=1000, seed=0):
    """户级 cluster bootstrap：重采样唯一组 → 指标分布 → 95% CI。

    协议对齐 validation/real_data_pi.py::_cluster_bootstrap_delta（实验脚本
    判决性协议的户级重采样模式）：以组（household/community）为重采样单元
    有放回抽取，保持户内相关结构；全阴/全阳的退化重采样跳过。
    本模块只依赖 .evaluation（与 training.py 无循环导入），作为主训练
    路径组感知评估的 bootstrap CI 单一真值源。

    参数：
        oof_scores (array-like): 池化 OOF 预测概率（与 y/groups 等长对齐）
        y (array-like): 0/1 标签
        groups (array-like): 样本所属组（household id / community id）
        metric (str): 'auroc' 或 'auprc'
        n_bootstrap (int): 重采样次数
        seed (int): 随机种子

    返回：
        dict: {'point': float, 'ci95': [lo, hi], 'n_effective': int}
        （n_effective = 非退化重采样次数；组数 < 2 或无法计算时
        ci95 为 None，point 仍尽量给出）
    """
    oof_scores = np.asarray(oof_scores, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    n = len(y)
    if len(oof_scores) != n or len(groups) != n:
        raise ValueError('oof_scores/y/groups 长度不一致（%d/%d/%d）'
                         % (len(oof_scores), len(y), len(groups)))

    def _metric(ya, sa):
        if ya.sum() == 0 or ya.sum() == len(ya):
            return None
        if metric == 'auprc':
            return float(average_precision_score(ya, sa))
        return float(roc_auc_score(ya, sa))

    point = _metric(y, oof_scores)

    uniq = np.unique(groups)
    if len(uniq) < 2 or n_bootstrap < 1 or point is None:
        return {'point': point, 'ci95': None, 'n_effective': 0}

    rng = np.random.RandomState(seed)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    values = []
    for _ in range(int(n_bootstrap)):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_g[g] for g in pick])
        v = _metric(y[rows], oof_scores[rows])
        if v is not None:
            values.append(v)
    values = np.array(values)
    if len(values) == 0:
        return {'point': point, 'ci95': None, 'n_effective': 0}
    lo, hi = float(np.percentile(values, 2.5)), \
        float(np.percentile(values, 97.5))
    return {'point': point, 'ci95': [lo, hi],
            'n_effective': int(len(values))}


def temporal_split(X, y, temporal_col, test_ratio=0.2, val_ratio=0.2,
                   random_state=42):
    """时间外验证划分：按时间先后排序后切分，训练用历史、测试用未来。

    参数：
        X: 特征矩阵 (n, d)
        y: 标签 (n,)
        temporal_col (array-like): 与样本等长的时间戳/序数（越大越近期）
        test_ratio/val_ratio: 测试/验证集占整体比例（按最新时间取）

    返回：
        dict: 同 split_train_val_test，'method' 为 'temporal'
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    t = np.asarray(temporal_col)

    order = np.argsort(t, kind='mergesort')  # 时间升序
    X_sorted = X[order]
    y_sorted = y[order]
    n = len(X_sorted)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    X_test, y_test = X_sorted[n - n_test:], y_sorted[n - n_test:]
    X_val, y_val = X_sorted[n - n_test - n_val:n - n_test], \
        y_sorted[n - n_test - n_val:n - n_test]
    X_train, y_train = X_sorted[:n - n_test - n_val], \
        y_sorted[:n - n_test - n_val]

    return {
        'X_train': np.asarray(X_train), 'y_train': np.asarray(y_train),
        'X_val': np.asarray(X_val), 'y_val': np.asarray(y_val),
        'X_test': np.asarray(X_test), 'y_test': np.asarray(y_test),
        'n_train': int(len(y_train)), 'n_val': int(len(y_val)),
        'n_test': int(len(y_test)), 'method': 'temporal',
    }


# ============================================================================
# 2. 判别指标
# ============================================================================

def compute_discrimination_metrics(y_true, y_prob):
    """计算判别指标：ROC-AUC、PR-AUC。单类时安全回退。"""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {'roc_auc': 0.5, 'pr_auc': float(np.mean(y_true))}
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except (ValueError, RuntimeError):
        roc_auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except (ValueError, RuntimeError):
        pr_auc = float(np.mean(y_true))
    return {'roc_auc': roc_auc, 'pr_auc': pr_auc}


# ============================================================================
# 3. 校准指标：校准曲线、Brier、Hosmer-Lemeshow
# ============================================================================

def _hosmer_lemeshow(y_true, y_prob, g=10):
    """Hosmer-Lemeshow 拟合优度检验。

    将样本按预测概率分为 g 组，比较每组观测阳性数与期望阳性数，
    构造 χ² 统计量并计算 p 值。

    返回：
        dict: {'chi2': float, 'df': int, 'p_value': float,
               'n_groups': int, 'groups': [{'bin':..,'n':..,'observed':..,'expected':..}]}
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    n = len(y)
    if n < 2 * g:
        g = max(2, n // 2)
    if g < 2:
        return {'chi2': float('nan'), 'df': 0, 'p_value': float('nan'),
                'n_groups': 0, 'groups': []}

    # 按预测概率升序排序后等分
    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]
    bin_edges = np.round(np.linspace(0, n, g + 1)).astype(int)
    bin_edges[0] = 0
    bin_edges[-1] = n

    groups = []
    chi2 = 0.0
    df = 0
    for i in range(g):
        lo, hi = int(bin_edges[i]), int(bin_edges[i + 1])
        if hi - lo <= 0:
            continue
        y_bin = y_sorted[lo:hi]
        p_bin = p_sorted[lo:hi]
        observed = float(np.sum(y_bin))
        expected = float(np.sum(p_bin))
        n_bin = float(len(y_bin))
        groups.append({
            'bin': i + 1, 'n': int(n_bin),
            'observed': observed, 'expected': expected,
        })
        denom = n_bin * (expected / n_bin) * (1 - expected / n_bin)
        if denom <= 1e-12:
            continue
        chi2 += (observed - expected) ** 2 / denom
        df += 1

    if df > 0 and _SCIPY_AVAILABLE:
        p_value = float(_stats.chi2.sf(chi2, df))
    elif df > 0:
        # 无 scipy：用正态近似（χ² 自由度为 df 时，√(2χ²)≈N(√(2df-1),1)）
        z = (np.sqrt(2 * chi2) - np.sqrt(2 * df - 1))
        p_value = float(2.0 * (1.0 - 0.5 * (1.0 + np.erf(np.abs(z) / np.sqrt(2)))))
    else:
        p_value = float('nan')

    return {'chi2': float(chi2), 'df': int(df), 'p_value': p_value,
            'n_groups': int(len(groups)), 'groups': groups}


def compute_calibration_metrics(y_true, y_prob, n_bins=10):
    """计算校准指标：Brier 分数、校准曲线、Hosmer-Lemeshow 检验。"""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)

    try:
        brier = float(brier_score_loss(y, p))
    except (ValueError, RuntimeError):
        brier = float('nan')

    # 校准曲线（等分位分箱）
    curve = {'mean_predicted': [], 'fraction_positives': [], 'n': []}
    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]
    n = len(p)
    g = min(n_bins, n)
    if g > 0:
        edges = np.round(np.linspace(0, n, g + 1)).astype(int)
        edges[0] = 0
        edges[-1] = n
        for i in range(g):
            lo, hi = int(edges[i]), int(edges[i + 1])
            if hi - lo <= 0:
                continue
            curve['mean_predicted'].append(float(np.mean(p_sorted[lo:hi])))
            curve['fraction_positives'].append(float(np.mean(y_sorted[lo:hi])))
            curve['n'].append(int(hi - lo))

    hl = _hosmer_lemeshow(y, p, g=n_bins)

    return {
        'brier': brier,
        'calibration_curve': curve,
        'hl': hl,
        'mean_predicted': float(np.mean(p)),
        'mean_observed': float(np.mean(y)),
    }


def assess_calibration(calib, brier_threshold=0.25, mean_tol=0.05,
                       p_threshold=0.05):
    """评估是否存在系统性概率偏差，决定是否触发重校准。

    系统低/高风险判定逻辑：
    - 若预测均值与观测均值偏差超过 mean_tol，记为方向性偏差；
    - 若 Hosmer-Lemeshow p<阈值 且校准曲线整体同侧偏移，记为系统性偏差。

    返回：
        dict: {'systematic_bias': bool, 'direction': 'over'|'under'|None,
               'recalibrate': bool, 'reason': str}
    """
    mean_pred = calib.get('mean_predicted', 0)
    mean_obs = calib.get('mean_observed', 0)
    brier = calib.get('brier', float('nan'))
    hl = calib.get('hl', {})
    p_value = hl.get('p_value', float('nan'))

    reasons = []
    direction = None
    bias = False

    diff = mean_pred - mean_obs
    if abs(diff) > mean_tol:
        bias = True
        direction = 'over' if diff > 0 else 'under'
        reasons.append(
            f"预测均值({mean_pred:.3f})与观测均值({mean_obs:.3f})偏差"
            f"{diff:+.3f}（阈值±{mean_tol}）")

    if not np.isnan(p_value) and p_value < p_threshold:
        bias = True
        reasons.append(f"Hosmer-Lemeshow p={p_value:.4f}<{p_threshold}，拟合不佳")
        if direction is None:
            # 用校准曲线中点判断方向
            mp = calib.get('calibration_curve', {}).get('mean_predicted', [])
            fp = calib.get('calibration_curve', {}).get('fraction_positives', [])
            if mp and fp:
                mid = len(mp) // 2
                d = mp[mid] - fp[mid]
                direction = 'over' if d > 0 else 'under'

    if not np.isnan(brier) and brier > brier_threshold:
        reasons.append(f"Brier分数({brier:.3f})偏高（阈值{brier_threshold}）")

    recalibrate = bias
    return {
        'systematic_bias': bias,
        'direction': direction,
        'recalibrate': recalibrate,
        'reason': '；'.join(reasons) if reasons else '未检测到系统性偏差',
    }


# ============================================================================
# 4. 概率重校准：Platt 缩放 / 保序回归
# ============================================================================

def _recalibrate_model(model, X_train, y_train, method='isotonic',
                       random_state=42):
    """对模型概率做重校准（Platt 缩放或保序回归）。

    返回：
        校准后的模型 | None（不可用时返回原模型）
    """
    if not _CALIBRATOR_AVAILABLE or model is None:
        return model
    try:
        # 特征名对齐：CalibratedClassifierCV 内部逐折 fit/predict 克隆模型，
        # numpy 输入下 LightGBM 克隆会记录自动特征名并触发警告（且
        # n_jobs=-1 并行 worker 的警告绕过 catch_warnings 直接打印 stderr，
        # 实测 3 折 = 3 条）。按基模型自身记录的名字包装后传入，折内名称
        # 一致 → 警告消除，数值不变。
        X_fit = X_train
        names = getattr(model, 'feature_names_in_', None)
        if PANDAS_AVAILABLE and not isinstance(X_train, pd.DataFrame) \
                and names is not None:
            _X_arr = np.asarray(X_train)
            if _X_arr.ndim == 2 and _X_arr.shape[1] == len(names):
                X_fit = pd.DataFrame(_X_arr, columns=list(names))
        calibrated = CalibratedClassifierCV(
            model, method=method, cv=3, n_jobs=-1)
        calibrated.fit(X_fit, y_train)
        return calibrated
    except (ValueError, RuntimeError, TypeError, OSError):
        return model


# ============================================================================
# 5. 交叉验证报告（平均 ± 标准差）
# ============================================================================

def _cross_validation_scores(model, X, y, n_splits=5, n_repeats=1,
                             random_state=42):
    """在给定数据上跑（重复）分层 K 折 CV，返回各折判别/校准指标。"""
    if X is None or len(X) == 0 or len(np.unique(y)) < 2:
        return None

    if n_repeats > 1:
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=random_state)

    fold_auc, fold_prauc, fold_brier = [], [], []
    for train_idx, test_idx in cv.split(X, y):
        try:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            # 每折重训克隆模型，避免数据泄漏（与主报告一致）
            m = _clone_model(model)
            m.fit(X_tr, y_tr)
            proba = m.predict_proba(_aligned_feature_frame(m, X_te))[:, 1]
            disc = compute_discrimination_metrics(y_te, proba)
            brier = float(brier_score_loss(y_te, proba))
            fold_auc.append(disc['roc_auc'])
            fold_prauc.append(disc['pr_auc'])
            fold_brier.append(brier)
        except Exception as e:
            LOGGER.debug("CV折评估失败，跳过: %s", e)
            continue

    if not fold_auc:
        return None
    return {
        'auc': {'mean': float(np.mean(fold_auc)), 'std': float(np.std(fold_auc)),
                'scores': fold_auc},
        'pr_auc': {'mean': float(np.mean(fold_prauc)),
                   'std': float(np.std(fold_prauc)), 'scores': fold_prauc},
        'brier': {'mean': float(np.mean(fold_brier)),
                  'std': float(np.std(fold_brier)), 'scores': fold_brier},
        'n_folds': len(fold_auc),
    }


def _clone_model(model):
    """克隆未拟合模型（用于 CV 每折重训，避免数据泄漏）。"""
    try:
        import copy
        return copy.deepcopy(model)
    except Exception:
        cls = model.__class__
        try:
            return cls(**model.get_params())
        except (TypeError, ValueError):
            return model


# ============================================================================
# 6. 主入口：生成聚合"模型验证报告"
# ============================================================================

def build_validation_report(predictor, X=None, y=None, temporal_col=None,
                            n_samples=2000, test_size=0.2, val_size=0.2,
                            n_bins=10, cv_folds=5, cv_repeats=1,
                            recalibrate_method='isotonic', random_state=42,
                            groups=None):
    """生成聚合的模型验证报告（训练后验证阶段）。

    参数：
        predictor: MLRiskPredictor（需已训练，用于提供模型与特征配置）
        X, y: 显式特征矩阵与标签（None 时用合成数据）
        temporal_col: 若提供，改用时间外验证；否则随机三层划分
        n_samples: 合成数据样本量（X/y 为空时使用）
        test_size/val_size: 测试/验证集比例
        n_bins: 校准曲线分箱数
        cv_folds/cv_repeats: 交叉验证折数与重复次数
        recalibrate_method: 重校准方法 'isotonic' / 'sigmoid'
        random_state: 随机种子
        groups: 样本所属组（如 household id，与 X/y 等长）。真实家庭
            接触者数据必传——户内相关下随机划分泄漏（见
            split_train_val_test 的 groups 分支）；合成数据无户结构
            保持 None。temporal_col 与 groups 同时给出时以时间外为准。

    返回：
        dict: 结构化验证报告（见下方字段说明）
    """
    report = {
        'status': 'skipped',
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'data': {}, 'models': {}, 'ensemble': {}, 'conclusion': {},
    }

    if not SKLEARN_AVAILABLE or not predictor.is_trained:
        report['status'] = 'skipped'
        report['conclusion'] = {'summary': '预测器未训练或 sklearn 不可用，无法验证。'}
        return report

    try:
        # ---- 准备数据 ----
        if X is None or y is None:
            X, y = predictor._generate_synthetic_training_data(
                n_samples, random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if len(np.unique(y)) < 2:
            report['status'] = 'error'
            report['conclusion'] = {'summary': '标签仅单类，无法验证。'}
            return report

        # ---- 1. 数据划分（防泄漏：训练/验证/测试 或 时间外）----
        if temporal_col is not None:
            split = temporal_split(X, y, temporal_col, test_size, val_size,
                                   random_state)
        else:
            split = split_train_val_test(X, y, test_size, val_size,
                                         random_state, groups=groups)
        report['data'] = {
            'n_total': int(len(y)),
            'n_positive': int(np.sum(y == 1)),
            'n_negative': int(np.sum(y == 0)),
            'positive_rate': float(np.mean(y)),
            'n_train': split['n_train'], 'n_val': split['n_val'],
            'n_test': split['n_test'],
            'split_method': split['method'],
        }
        if 'n_groups' in split:
            report['data']['n_groups'] = split['n_groups']

        X_train, y_train = split['X_train'], split['y_train']
        X_val, y_val = split['X_val'], split['y_val']
        X_test, y_test = split['X_test'], split['y_test']

        model_keys = list(predictor.models.keys())
        if not model_keys:
            report['status'] = 'error'
            report['conclusion'] = {'summary': '预测器无已训练模型。'}
            return report

        # ---- 2. 在训练集上重训各模型（避免使用全量拟合模型泄漏）----
        from .training import _train_models_quick
        models = _train_models_quick(model_keys, X_train, y_train, random_state)

        # ---- 3. 逐模型评估：验证集 + 测试集 + 校准 + 重校准 + CV ----
        ensemble_val = np.zeros(len(X_val), dtype=float)
        ensemble_test = np.zeros(len(X_test), dtype=float)
        n_trained = 0

        for key in model_keys:
            model = models.get(key)
            if model is None:
                continue
            n_trained += 1
            name = predictor.model_performance.get(key, {}).get('name', key)
            entry = {'name': name, 'name_en': key}

            val_proba = model.predict_proba(
                _aligned_feature_frame(model, X_val))[:, 1]
            test_proba = model.predict_proba(
                _aligned_feature_frame(model, X_test))[:, 1]
            ensemble_val += val_proba
            ensemble_test += test_proba

            # 判别指标
            entry['validation'] = compute_discrimination_metrics(y_val, val_proba)
            entry['test'] = compute_discrimination_metrics(y_test, test_proba)

            # 校准指标（验证集上评估）
            calib = compute_calibration_metrics(y_val, val_proba, n_bins)
            entry['calibration'] = calib
            entry['calibration_assessment'] = assess_calibration(calib)

            # ---- 4. 若系统性偏差则重校准，并在测试集上复评 ----
            assessment = entry['calibration_assessment']
            rec_entry = {
                'was_applied': False, 'method': recalibrate_method,
                'val_before': float(calib['brier']),
                'test_before_brier': float(
                    brier_score_loss(y_test, test_proba)),
            }
            if assessment['recalibrate']:
                recal = _recalibrate_model(model, X_train, y_train,
                                           recalibrate_method, random_state)
                recal_val = recal.predict_proba(
                    _aligned_feature_frame(recal, X_val))[:, 1]
                recal_test = recal.predict_proba(
                    _aligned_feature_frame(recal, X_test))[:, 1]
                rec_entry['was_applied'] = True
                rec_entry['val_after_brier'] = float(
                    brier_score_loss(y_val, recal_val))
                rec_entry['test_after_brier'] = float(
                    brier_score_loss(y_test, recal_test))
                rec_entry['brier_improvement'] = float(
                    rec_entry['test_before_brier']
                    - rec_entry['test_after_brier'])
                # 若重校准提升，用重校准后的概率更新测试集判别
                if rec_entry['brier_improvement'] > 0:
                    entry['test'] = compute_discrimination_metrics(
                        y_test, recal_test)
            entry['recalibration'] = rec_entry

            # ---- 5. 交叉验证报告（训练集上，平均 ± 标准差）----
            entry['cv'] = _cross_validation_scores(
                model, X_train, y_train, n_splits=cv_folds,
                n_repeats=cv_repeats, random_state=random_state)

            report['models'][key] = entry

        # ---- 6. 集成分数验证（简单平均）----
        if n_trained > 0:
            ensemble_val /= n_trained
            ensemble_test /= n_trained
            ens_calib = compute_calibration_metrics(y_val, ensemble_val, n_bins)
            report['ensemble'] = {
                'name': '集成（简单平均）',
                'validation': compute_discrimination_metrics(y_val, ensemble_val),
                'test': compute_discrimination_metrics(y_test, ensemble_test),
                'calibration': ens_calib,
                'calibration_assessment': assess_calibration(ens_calib),
            }

        # ---- 7. 结论汇总 ----
        report['conclusion'] = _build_conclusion(report)

        report['status'] = 'ok'
        LOGGER.info("模型验证报告生成完成: %d 个模型, 三层划分=%s, 测试AUROC≈%.3f",
                    n_trained, report['data']['split_method'],
                    report['ensemble'].get('test', {}).get('roc_auc', 0))
        return report

    except Exception as e:
        LOGGER.warning("模型验证报告生成失败: %s", e, exc_info=True)
        report['status'] = 'error'
        report['conclusion'] = {'summary': f'验证失败: {e}'}
        return report


def _build_conclusion(report):
    """基于各模型指标生成结论。"""
    models = report.get('models', {})
    if not models:
        return {'summary': '无可用模型结论。'}

    test_aurocs = {k: v.get('test', {}).get('roc_auc', 0)
                   for k, v in models.items()}
    test_briers = {k: v.get('recalibration', {}).get(
        'test_after_brier',
        v.get('calibration', {}).get('brier', float('nan')))
        for k, v in models.items()}

    best_model = max(test_aurocs, key=lambda k: test_aurocs[k])
    best_auc = test_aurocs[best_model]
    best_name = models.get(best_model, {}).get('name', best_model)

    recal_count = sum(1 for v in models.values()
                      if v.get('recalibration', {}).get('was_applied'))
    cv_aucs = [v.get('cv', {}).get('auc', {}).get('mean', float('nan'))
               for v in models.values()]
    cv_auds_valid = [a for a in cv_aucs if not np.isnan(a)]
    cv_mean = float(np.mean(cv_auds_valid)) if cv_auds_valid else float('nan')
    cv_std = float(np.std(cv_auds_valid)) if cv_auds_valid else float('nan')

    mean_test_auc = float(np.mean(list(test_aurocs.values())))
    mean_test_brier = float(np.mean([b for b in test_briers.values()
                                     if not np.isnan(b)]))

    summary_parts = [
        f"最佳模型：《{best_name}》测试集 AUROC={best_auc:.3f}",
        f"各模型测试集 AUROC 均值={mean_test_auc:.3f}",
        f"训练集交叉验证 AUROC 均值±标准差={cv_mean:.3f}±{cv_std:.3f}",
        f"测试集 Brier 均值={mean_test_brier:.3f}",
    ]
    if recal_count:
        summary_parts.append(f"{recal_count} 个模型检测到系统性概率偏差并已重校准")
    else:
        summary_parts.append("未检测到系统性概率偏差，概率输出可信")

    if best_auc >= 0.8:
        overall = '判别性能良好'
    elif best_auc >= 0.7:
        overall = '判别性能中等，建议结合临床评估使用'
    else:
        overall = '判别性能偏弱，建议补充真实数据重训'

    return {
        'best_model': best_model,
        'best_model_name': best_name,
        'best_test_auc': best_auc,
        'mean_test_auc': mean_test_auc,
        'mean_test_brier': mean_test_brier,
        'cv_mean_auc': cv_mean,
        'cv_std_auc': cv_std,
        'recalibrated_count': recal_count,
        'overall_assessment': overall,
        'summary': '；'.join(summary_parts),
    }