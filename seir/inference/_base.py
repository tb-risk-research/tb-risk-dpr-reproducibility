"""贝叶斯推断基类：共享工具方法与默认迭代参数。

各版本（v1/v2/v3/v4）推断类继承本基类，复用：
  - _neg_binom_loglik     负二项对数似然
  - compute_r0_posterior  R0 后验统计量
  - compute_hdi           最高密度区间
  - DEFAULT_N_ITERATIONS / DEFAULT_N_BURNIN / DEFAULT_N_CHAINS
"""

import math

import numpy as np

from ..stochastic import StochasticSEIRModel


class BaseBayesianInference:
    """贝叶斯 MCMC 参数推断引擎基类（共享工具方法）。

    子类（v1/v2/v3/v4）按模型代际实现各自的先验/似然/后验与采样器。
    """

    # 默认迭代参数（与 uncertainty.py 一致）
    DEFAULT_N_ITERATIONS = 10000
    DEFAULT_N_BURNIN = 2000
    DEFAULT_N_CHAINS = 4

    # 默认负二项过度离散参数 k (Gamma(2, 10) 先验均值 ≈ 0.2, Lloyd-Smith 2005)
    DEFAULT_NEG_BINOM_K = 0.2

    # 多链MCMC种子派生用大素数（确保各链种子间距足够大，不重叠）
    _CHAIN_SEED_PRIME = 1000003
    _CHAIN_SEED_MOD = 2**31 - 1

    def __init__(self, seir_model=None, seed=42):
        import numpy as np
        self._np = np
        # 显式传递 seed 确保 SEIR 模型与 MCMC 使用一致的随机种子
        self.seir_model = seir_model or StochasticSEIRModel(seed=int(seed))
        self.base_seed = int(seed) % self._CHAIN_SEED_MOD  # 保存基础种子用于派生链种子
        self.rng = np.random.RandomState(seed)
        self.traces = {}
        self.posterior_samples = {}
        self.convergence_diagnostics = {}

    def _derive_chain_seed(self, chain_idx: int) -> int:
        """从base_seed确定性派生各链的种子，不依赖numpy随机状态内部表示。
        
        使用 base_seed + chain_idx * large_prime 方式，确保：
        1. 可复现：相同base_seed和chain_idx永远得到相同种子
        2. 独立：不同链的RNG序列互不重叠（大素数间距足够）
        3. 鲁棒：不依赖numpy RandomState内部get_state()结构
        
        参数：
            chain_idx: 链索引（0-based）
            
        返回：
            该链的随机种子（int，[0, 2^31-1]范围）
        """
        return (self.base_seed + chain_idx * self._CHAIN_SEED_PRIME) % self._CHAIN_SEED_MOD

    # ==================== uncertainty/ 兼容适配属性 ====================
    # 问题三：统一贝叶斯框架。保留 seir/inference/ 作为唯一入口，废弃
    # seir/uncertainty/。消费者（SEIRIntegration、PosteriorDrivenInfectivity）
    # 原依赖 uncertainty/ 的 inference_completed / posterior_summary /
    # mcmc_diagnostics 三个属性名，此处提供等价适配，由 posterior_samples +
    # traces + convergence_diagnostics 派生计算，无需子类显式赋值。

    @property
    def inference_completed(self):
        """后验采样是否完成（派生自 posterior_samples 与 convergence_diagnostics）"""
        return bool(self.posterior_samples) and bool(self.convergence_diagnostics)

    @property
    def posterior_summary(self):
        """后验摘要 dict[name → {mean, std, median, ci_2_5, ci_97_5, prior_type, unit}]

        无样本时返回 None（与 uncertainty/ 行为一致）。
        """
        np = self._np
        if not self.posterior_samples:
            return None
        priors = getattr(self, 'SEIR_PARAM_PRIORS', {}) or {}
        summary = {}
        for name, samples in self.posterior_samples.items():
            arr = np.asarray(samples, dtype=float)
            if arr.size == 0:
                continue
            prior_info = priors.get(name, {})
            summary[name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'median': float(np.median(arr)),
                'ci_2_5': float(np.percentile(arr, 2.5)),
                'ci_97_5': float(np.percentile(arr, 97.5)),
                'prior_type': prior_info.get('dist', 'unknown'),
                'unit': prior_info.get('unit', 'unknown'),
            }
        return summary

    @property
    def mcmc_diagnostics(self):
        """MCMC 诊断 dict（与 uncertainty/ 形状对齐）

        返回 {n_iterations, n_burnin, n_chains, accept_rates, mean_accept_rate,
              r_hat(list), ess(list)}。HMCSampler 的 r_hat/ess 可能是 dict，
        此处统一转为 list 以匹配消费者期望（SEIRIntegration.compute_seir_confidence_weight
        用 max(r_hat)/min(ess/400) 计算）。
        """
        np = self._np
        if not self.convergence_diagnostics:
            return {}
        diag = self.convergence_diagnostics
        traces = self.traces or {}

        r_hat = diag.get('r_hat', [])
        ess = diag.get('ess', [])
        # HMCSampler 诊断返回 dict(name→float)，消费者期望 list[float]
        if isinstance(r_hat, dict):
            r_hat = list(r_hat.values())
        if isinstance(ess, dict):
            ess = list(ess.values())

        accept_rates = traces.get('acceptance_rates', [])
        if isinstance(accept_rates, dict):
            accept_rates = list(accept_rates.values())
        accept_rates = [float(r) for r in accept_rates]

        return {
            'n_iterations': int(diag.get('n_samples', diag.get('n_iterations', 0))),
            'n_burnin': int(diag.get('n_warmup', diag.get('n_burnin', 0))),
            'n_chains': int(diag.get('n_chains', 0)),
            'accept_rates': accept_rates,
            'mean_accept_rate': float(np.mean(accept_rates)) if accept_rates else 0.0,
            'r_hat': [float(r) for r in r_hat],
            'ess': [float(e) for e in ess],
        }

    def _neg_binom_loglik(self, obs, sim, k=None):
        """负二项分布对数似然（改进5, Lloyd-Smith et al. 2005, Nature）

        参数化: 均值 = sim, 方差 = sim + sim^2 / k
        较小 k 表示高度过度离散（聚集性 outbreak）；k → ∞ 退化为 Poisson。

        对数形式:
          ll = lgamma(obs+k) - lgamma(k) - lgamma(obs+1)
               + k·log(k/(k+sim)) + obs·log(sim/(k+sim))

        数值保护: sim=0 时加 1e-6 避免 log(0)。
        """
        np = self._np
        if k is None:
            k = self.DEFAULT_NEG_BINOM_K
        k = max(float(k), 1e-6)
        sim = np.maximum(np.asarray(sim, dtype=float), 1e-6)
        obs = np.asarray(obs, dtype=float)
        try:
            from scipy.special import gammaln
            lg_obs_k = gammaln(obs + k)
            lg_k = float(gammaln(k))
            lg_obs_1 = gammaln(obs + 1.0)
        except ImportError:
            _vlgamma = np.vectorize(math.lgamma)
            lg_obs_k = _vlgamma(obs + k)
            lg_k = math.lgamma(k)
            lg_obs_1 = _vlgamma(obs + 1.0)
        ll = (lg_obs_k - lg_k - lg_obs_1
              + k * np.log(k / (k + sim))
              + obs * np.log(sim / (k + sim)))
        return float(np.sum(ll))

    def compute_r0_posterior(self, beta_samples=None):
        if beta_samples is None:
            beta_samples = self.posterior_samples.get('beta', np.array([0.2]))
        gamma_samples = self.posterior_samples.get('gamma', np.array([0.05]))
        if beta_samples is None or len(beta_samples) == 0:
            beta_samples = np.array([0.2])
        if gamma_samples is None or len(gamma_samples) == 0:
            gamma_samples = np.array([0.05])
        r0_samples = beta_samples / np.maximum(gamma_samples, 1e-10)
        return {
            'mean': float(np.mean(r0_samples)),
            'median': float(np.median(r0_samples)),
            'ci_95_low': float(np.percentile(r0_samples, 2.5)),
            'ci_95_high': float(np.percentile(r0_samples, 97.5)),
            'samples': r0_samples,
        }

    def compute_hdi(self, samples, prob=0.95):
        np = self._np
        if len(samples) == 0:
            raise ValueError("samples 不能为空")
        sorted_samples = np.sort(samples)
        n = len(sorted_samples)
        # interval_idx 至少为 1，否则 n=1 时循环不执行，返回单点区间语义不正确
        interval_idx = max(1, int(np.floor(prob * n)))
        if interval_idx >= n:
            interval_idx = n - 1
        min_interval_width = float('inf')
        hdi_min = hdi_max = sorted_samples[0]
        for i in range(n - interval_idx):
            width = sorted_samples[i + interval_idx] - sorted_samples[i]
            if width < min_interval_width:
                min_interval_width = width
                hdi_min, hdi_max = sorted_samples[i], sorted_samples[i + interval_idx]
        return float(hdi_min), float(hdi_max)
