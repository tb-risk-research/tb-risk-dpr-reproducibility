"""SEIR 参数不确定性量化 — 收敛诊断模块

R-hat (rank-normalized Gelman-Rubin) 与 ESS (有效样本量) 收敛诊断。
拆分自原 seir/uncertainty.py（业务逻辑不变）。
"""

import numpy as np

try:
    import scipy.stats as sps
    SCIPY_STATS_AVAILABLE = True
except ImportError:
    SCIPY_STATS_AVAILABLE = False


class _DiagnosticsMixin:
    """收敛诊断: R-hat 与 ESS"""

    def _compute_r_hat(self, chains, burnin):
        """计算增强版 Gelman-Rubin R-hat 收敛诊断

        使用 rank-normalized 方法（Vehtari 2021），
        对厚尾后验分布比经典 R-hat 更稳健。

        参考：Vehtari et al. (2021) "Rank-Normalization, Folding, and
        Localization: An Improved R-hat for Assessing Convergence of MCMC"
        Bayesian Analysis 16(2): 667-718.
        """
        if not chains or len(chains) < 2:
            n_params = chains[0].shape[1] if chains and hasattr(chains[0], 'shape') else 0
            return [float('nan')] * max(n_params, 1)
        n_params = chains[0].shape[1]
        r_hats = []
        for p in range(n_params):
            chain_vals = [c[burnin:, p] for c in chains]
            n_samples = len(chain_vals[0])

            if n_samples < 10:
                r_hats.append(1.0)
                continue

            # Rank-normalized R-hat:
            # 1. Pool all samples and rank them
            all_vals = np.concatenate(chain_vals)

            # 2. Normalize ranks via inverse normal CDF
            if SCIPY_STATS_AVAILABLE:
                ranks = np.argsort(np.argsort(all_vals)) + 1  # 1-based ranks
                blom_offset = 3.0 / 8.0
                blom_denom = len(all_vals) + 0.25
                z_all = sps.norm.ppf((ranks - blom_offset) / blom_denom)
                # Split back per chain
                z_chains = []
                offset = 0
                for cv in chain_vals:
                    z_chains.append(z_all[offset:offset + len(cv)])
                    offset += len(cv)
            else:
                # Fallback: use original chain values
                z_chains = chain_vals

            # 3. Compute R-hat on z-scores
            chain_means = [np.mean(z) for z in z_chains]
            chain_vars = [np.var(z) for z in z_chains]
            overall_mean = np.mean(chain_means)
            m = len(z_chains)
            B = n_samples / (m - 1) * sum((cm - overall_mean)**2 for cm in chain_means) if m > 1 else 0.0
            W = np.mean(chain_vars)
            var_plus = ((n_samples - 1) / max(n_samples, 1)) * W + B / max(n_samples, 1)
            r_hat = np.sqrt(var_plus / max(W, 1e-10))
            r_hats.append(float(min(r_hat, 10.0)))

        return r_hats

    def _compute_ess(self, chains, burnin):
        """计算有效样本量 (Effective Sample Size)

        使用多链批量均值估计，对每个参数返回 ESS。
        ESS < 400 表示后验估计可能不可靠。
        参考：Vehtari et al. (2021) "Rank-Normalization, Folding, and
        Localization: An Improved R-hat for Assessing Convergence of MCMC"
        Bayesian Analysis 16(2): 667-718.
        """
        if not chains or len(chains) < 2:
            n_params = chains[0].shape[1] if chains and hasattr(chains[0], 'shape') else 0
            return [float('nan')] * max(n_params, 1)
        n_params = chains[0].shape[1]
        ess_values = []
        for p in range(n_params):
            chain_vals = [c[burnin:, p] for c in chains]
            n_samples = len(chain_vals[0])
            m = len(chain_vals)

            if n_samples < 10:
                ess_values.append(float(n_samples * m))
                continue

            all_vals = np.concatenate(chain_vals)
            mean_all = np.mean(all_vals)
            var_all = np.var(all_vals)

            if var_all < 1e-15:
                ess_values.append(float(n_samples * m))
                continue

            # 自相关截断（Geyer 1992 初始单调正序列估计）
            max_lag = min(n_samples // 2, 100)
            autocorr = []
            for lag in range(1, max_lag + 1):
                ac = 0.0
                for chain in chain_vals:
                    x = chain - mean_all
                    ac += np.sum(x[:-lag] * x[lag:]) / (n_samples - lag)
                ac /= (m * var_all)
                autocorr.append(ac)

            # 截断自相关：当两个连续滞后和为负时停止
            tau = 1.0
            for lag_idx in range(0, len(autocorr) - 1, 2):
                rho_sum = autocorr[lag_idx]
                if lag_idx + 1 < len(autocorr):
                    rho_sum += autocorr[lag_idx + 1]
                if rho_sum <= 0:
                    break
                tau += 2 * rho_sum

            ess = n_samples * m / max(tau, 1e-10)
            ess_values.append(float(min(ess, n_samples * m)))

        return ess_values
