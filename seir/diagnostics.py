"""贝叶斯 MCMC 收敛诊断函数

提供 rank-normalized R-hat (Vehtari 2021) 与有效样本量 ESS 的独立实现。
从 seir/bayesian.py 中 BayesianSEIRInference 与 HMCSampler 的诊断方法提取而来，
供 bayesian.py 与 hmc_sampler.py 共享，避免循环导入。

文献：
  Vehtari A et al. (2021) Rank-normalization, folding, and localization:
  An improved R-hat for assessing convergence of MCMC. Bayesian Anal 16(2).
"""

import numpy as np


def _rank_normalized_r_hat(chain_vals):
    """核心 R-hat 计算：rank-normalized (Vehtari 2021)，输入各链样本列表。

    chain_vals: list[np.ndarray]，每条链为该参数的 1-D 样本序列。
    返回 float R-hat（上限截断 10.0）。
    """
    n_total = sum(len(c) for c in chain_vals)
    if n_total == 0 or len(chain_vals) < 2:
        return 1.0

    try:
        from scipy import stats as sps
    except ImportError:
        sps = None

    if sps is not None:
        # Rank-normalized R-hat (Vehtari 2021)
        all_vals = np.concatenate(chain_vals)
        ranks = np.argsort(np.argsort(all_vals)) + 1.0
        blom_offset = 3.0 / 8.0
        blom_denom = n_total + 0.25
        z_all = sps.norm.ppf((ranks - blom_offset) / blom_denom)

        z_chains = []
        start = 0
        for c in chain_vals:
            n = len(c)
            z_chains.append(z_all[start:start + n])
            start += n
    else:
        # 无 scipy 时用原始值（简化 R-hat）
        z_chains = list(chain_vals)

    chain_means = np.array([np.mean(z) for z in z_chains])
    chain_vars = np.array([np.var(z, ddof=1) for z in z_chains])

    m = len(z_chains)
    n_samples = len(z_chains[0])
    if n_samples < 2:
        return 1.0
    B = n_samples / (m - 1) * np.sum((chain_means - np.mean(chain_means)) ** 2)
    W = np.mean(chain_vars)
    var_plus = ((n_samples - 1) / n_samples) * W + B / n_samples
    r_hat = np.sqrt(var_plus / max(W, 1e-10))
    return float(min(r_hat, 10.0))


def compute_split_r_hat(chains, burn_in, param_names=None):
    """Split R-hat (Vehtari 2021 推荐)：每条链拆前后两半再计算。

    经典 R-hat 只检测链间不一致，单链内部分布漂移（前半与后半
    采样自不同分布）时仍可能报 1.0。split 版本把每条链视为两条
    子链（2m 条链），链内不平稳同样被放大为 R-hat > 1。

    参数与返回值同 compute_r_hat。
    """
    if param_names is None:
        param_names = ['beta', 'sigma', 'gamma']

    if len(chains) < 1:
        return {name: 1.0 for name in param_names}

    r_hats = {}
    for idx, name in enumerate(param_names):
        halves = []
        for c in chains:
            seg = c[burn_in:, idx]
            n = len(seg)
            if n < 4:  # 不足以拆两半（每半至少 2 样本）
                r_hats[name] = 1.0
                break
            mid = n // 2
            halves.append(seg[:mid])
            halves.append(seg[mid:])
        else:
            r_hats[name] = _rank_normalized_r_hat(halves)

    return r_hats


def compute_r_hat(chains, burn_in, param_names=None):
    """计算 rank-normalized R-hat (Vehtari 2021)。

    注意：本函数为链间经典 R-hat；链内不平稳检测请用
    compute_split_r_hat（Vehtari 2021 推荐的 split 版本，
    每链拆前后两半 → 2m 条子链，能暴露单链内部分布漂移）。

    参数：
        chains: list[np.ndarray], 各链的完整轨迹，每条 shape (n_iterations, n_params)
        burn_in: int, burn-in 迭代数
        param_names: list[str], 参数名列表；默认 ['beta', 'sigma', 'gamma']

    返回：
        dict: 参数名到 R-hat 值的映射
    """
    if param_names is None:
        param_names = ['beta', 'sigma', 'gamma']

    if len(chains) < 2:
        return {name: 1.0 for name in param_names}

    r_hats = {}
    for idx, name in enumerate(param_names):
        chain_vals = [c[burn_in:, idx] for c in chains]
        r_hats[name] = _rank_normalized_r_hat(chain_vals)

    return r_hats


def compute_ess(merged_chain, param_names=None):
    """计算有效样本量 ESS（自相关方法）。

    参数：
        merged_chain: np.ndarray, shape (n_samples, n_params)
        param_names: list[str], 参数名列表；默认 ['beta', 'sigma', 'gamma']

    返回：
        dict: 参数名到 ESS 值的映射
    """
    if param_names is None:
        param_names = ['beta', 'sigma', 'gamma']

    ess_dict = {}
    n = merged_chain.shape[0]

    for idx, name in enumerate(param_names):
        samples = merged_chain[:, idx]
        if n < 2:
            ess_dict[name] = n
            continue

        mean = np.mean(samples)
        var = np.var(samples)
        if var < 1e-15:
            ess_dict[name] = n
            continue

        max_lag = min(n // 2, 100)
        rho = np.zeros(max_lag)
        for lag in range(1, max_lag + 1):
            rho[lag - 1] = np.mean(
                (samples[:n - lag] - mean) * (samples[lag:] - mean)
            ) / var
            if rho[lag - 1] < 0:
                break

        tau = 1.0 + 2.0 * np.sum(rho[rho > 0])
        ess_dict[name] = int(n / max(tau, 1.0))

    return ess_dict


def compute_ess_single(samples):
    """计算单参数的有效样本量。

    参数：
        samples: np.ndarray, shape (n_samples,)

    返回：
        int: 有效样本量
    """
    n = len(samples)
    if n < 10:
        return n
    mean = np.mean(samples)
    var = np.var(samples)
    if var < 1e-10:
        return n
    max_lag = min(n // 2, 100)
    rho = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        rho[lag - 1] = np.mean(
            (samples[:n - lag] - mean) * (samples[lag:] - mean)
        ) / var
        if rho[lag - 1] < 0:
            break
    tau = 1.0 + 2.0 * np.sum(rho[rho > 0])
    return int(n / max(tau, 1.0))
