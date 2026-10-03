"""哈密顿蒙特卡洛 (HMC) 采样器 + 对偶平均法步长自适应

从 seir/bayesian.py 中拆分而来。使用 PyTorch autodiff 计算梯度，支持：
  - Leapfrog 辛积分器 (symplectic integrator)
  - 对偶平均法 (Nesterov 2009 / Hoffman & Gelman 2014 Algorithm 5)
  - 对角质量矩阵 (从 warmup 样本估计)
  - 多链 + rank-normalized R-hat (Vehtari 2021)

在 15 维 v4 参数空间中，HMC 比随机游走 MH 效率高 10-100x。

文献：
  Hoffman MD, Gelman A. (2014) JMLR 15(1):1593-1623 — NUTS/HMC + 对偶平均
  Nesterov Y. (2009) Ecological Modelling 12:127-145 — 对偶平均法
  Betancourt M. (2017) arXiv:1701.02434 — HMC 概念入门
  Neal RM. (2011) MCMC Handbook — HMC 综述
"""

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .diagnostics import compute_r_hat, compute_split_r_hat, compute_ess_single


def _safe_rhat_converged(r_hat_dict):
    """安全检查 R-hat 收敛性，跳过 None/NaN/非标量值。

    防御性类型处理：r_hat_dict.values() 中的值在某些随机状态下可能为
    None、numpy 数组或 torch 张量而非标量，此时 float() 转换会抛出
    TypeError。本函数捕获异常并将异常值视为未收敛（跳过）。

    文献: Vehtari et al. (2021) Bayesian Anal 16(2) — R-hat < 1.1 收敛判据
    """
    for r in r_hat_dict.values():
        if r is None:
            continue
        try:
            r_float = float(r)
        except (TypeError, ValueError):
            continue
        if not np.isnan(r_float) and r_float >= 1.1:
            return False
    return True


class HMCSampler:
    """哈密顿蒙特卡洛采样器 + 对偶平均法步长自适应

    使用 PyTorch autodiff 计算梯度，支持:
      - Leapfrog 辛积分器 (symplectic integrator)
      - 对偶平均法 (Nesterov 2009 / Hoffman & Gelman 2014 Algorithm 5)
      - 对角质量矩阵 (从 warmup 样本估计)
      - 多链 + rank-normalized R-hat (Vehtari 2021)

    在 15 维 v4 参数空间中，HMC 比随机游走 MH 效率高 10-100x。

    文献:
      Hoffman MD, Gelman A. (2014) JMLR 15(1):1593-1623 — NUTS/HMC + 对偶平均
      Nesterov Y. (2009) Ecological Modelling 12:127-145 — 对偶平均法
      Betancourt M. (2017) arXiv:1701.02434 — HMC 概念入门
      Neal RM. (2011) MCMC Handbook — HMC 综述
    """

    def __init__(self, log_posterior_fn, grad_fn, n_params,
                 target_accept=0.8, n_leapfrog=20,
                 n_warmup=1000, n_samples=2000, n_chains=4,
                 device=None, seed=42,
                 adaptive_leapfrog=True,
                 leapfrog_min=5, leapfrog_max=50,
                 leapfrog_adjust_interval=50,
                 precision='float64',
                 use_hash_cache=False, cache_maxsize=1024):
        """初始化 HMC 采样器。

        要点B: 自适应 n_leapfrog (路径二: 基于接受率的动态调整)。
        当 ``adaptive_leapfrog=True`` 时，在 warmup 阶段每隔
        ``leapfrog_adjust_interval`` 步检查滑动窗口平均接受率：
          - 接受率 > 0.95: 步长过小/轨迹过短 → n_leapfrog += 2
          - 接受率 < 0.65: 可能出现 U-turn → n_leapfrog = max(min, n-2)
        warmup 结束时锁定 n_leapfrog，用于采样阶段。
        这种简化方案虽不如完整 NUTS (Hoffman & Gelman 2014 Alg. 3) 优雅，
        但实现成本低且能缓解固定 leapfrog 步数的主要问题。

        要点3a (第三轮): float32 混合精度。``precision`` 参数控制 dtype:
          - 'float64' (默认): 全程 float64, 最高精度
          - 'float32': 全程 float32, GPU 提速 ~2x
          - 'mixed': warmup float64 (保证自适应精度), 采样 float32 (提速)
        mixed 模式在 warmup 结束后将 theta/inv_mass 转为 float32。
        CPU 环境下 float32 提速有限 (~10%), GPU 环境下提速显著 (~2x)。
        数值稳定性: autodiff.constrain 已对 float32 使用更大 epsilon (1e-7)
        保护 log(s) 和 log(1-s), 避免 sigmoid 饱和时梯度溢出。

        改进15 (六): MD5 hash 似然缓存。``use_hash_cache=True`` 时启用
        多条目 LRU 缓存 (LikelihoodCache), 作为现有 identity 缓存的补充。
        identity 缓存 (``theta is last_theta``) 在拒绝时命中 (零开销),
        但在 theta 被克隆/batch 拆分/warmup→采样切换时失效。hash 缓存
        通过 MD5(theta.tobytes()) 键控, 能在这些场景命中, 代价是每查询
        一次 ~µs 哈希开销。默认关闭 (identity 缓存已覆盖主要场景)。

        文献:
          Hoffman & Gelman (2014) JMLR 15(1):1593-1623 — NUTS/对偶平均
          Neal (2011) MCMC Handbook — 固定 leapfrog 步数的局限性
          Micikevicius et al. (2018) arXiv:1710.03740 — float32 混合精度训练
        """
        if not TORCH_AVAILABLE:
            raise ImportError("HMC 需要 PyTorch。请安装: pip install torch")
        if precision not in ('float64', 'float32', 'mixed'):
            raise ValueError(
                f"precision 必须为 'float64'/'float32'/'mixed', got {precision!r}")
        self.log_posterior_fn = log_posterior_fn
        self.grad_fn = grad_fn
        self.n_params = n_params
        self.target_accept = target_accept
        self.n_leapfrog = n_leapfrog
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.device = device or torch.device('cpu')
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # 要点B: 自适应 leapfrog 参数
        self.adaptive_leapfrog = bool(adaptive_leapfrog)
        self.leapfrog_min = max(int(leapfrog_min), 1)
        self.leapfrog_max = max(int(leapfrog_max), self.leapfrog_min)
        self.leapfrog_adjust_interval = max(int(leapfrog_adjust_interval), 1)
        # 要点3a: 精度模式
        self.precision = precision
        self._warmup_dtype = torch.float64 if precision in ('float64', 'mixed') else torch.float32
        self._sample_dtype = torch.float32 if precision in ('float32', 'mixed') else torch.float64

        # 改进15 (六): 可选 MD5 hash 缓存 (补充 identity 缓存)
        # 要点2 (第五轮): GPU 环境自动禁用。hash_theta 的 .cpu().numpy()
        # 在 GPU 上触发同步, 抵消 batch 模式的 GPU 并行优化。检测 device
        # 为 CUDA 时强制 use_hash_cache=False (LikelihoodCache 内部也有
        # 防御性 auto_disable_on_gpu 检查作为备份)。
        self.use_hash_cache = bool(use_hash_cache)
        is_cuda_device = (
            isinstance(self.device, torch.device) and self.device.type == 'cuda'
        ) or (isinstance(self.device, str) and 'cuda' in self.device.lower())
        if self.use_hash_cache and is_cuda_device:
            import warnings as _warnings
            _warnings.warn(
                "use_hash_cache=True 在 GPU 设备上会触发 GPU-CPU 同步 "
                "(hash_theta 内部 .cpu().numpy()), 抵消 batch 模式的 GPU "
                "并行优化。已自动禁用 use_hash_cache。建议在 CPU 环境使用 "
                "hash cache, GPU 环境依赖 identity cache 即可。",
                UserWarning, stacklevel=2)
            self.use_hash_cache = False
        self.cache_maxsize = int(cache_maxsize)
        self._hash_cache = None
        if self.use_hash_cache:
            from .likelihood_cache import LikelihoodCache
            self._hash_cache = LikelihoodCache(maxsize=self.cache_maxsize)

        self.chains = []
        self.log_probs = []
        self.acceptance_rates = []
        self.diagnostics = {}
        self.step_sizes = []
        self.mass_diag = None
        # 每条链 warmup 结束时锁定的 n_leapfrog (要点B)
        self.final_n_leapfrog = []

    def _cached_grad_log_post(self, theta, identity_cache_hit):
        """查询 grad_fn 结果 (identity cache → hash cache → grad_fn)

        改进15 (六): 三级缓存查询。identity cache (调用方维护) 命中时
        直接返回, 零开销; 未命中时查 hash cache (MD5 键); 仍未命中则
        调用 grad_fn 并存入 hash cache。

        Args:
            theta: 当前参数向量 (torch.Tensor)
            identity_cache_hit: tuple (grad, log_post) 或 None
                调用方先查 identity cache, 命中则传入, 未命中传 None
        Returns:
            (grad, log_post, from_cache) — from_cache 标识是否命中 hash cache
        """
        if identity_cache_hit is not None:
            grad, log_post = identity_cache_hit
            return grad, log_post, False
        if self._hash_cache is not None:
            cached = self._hash_cache.get(theta)
            if cached is not None:
                return cached[1], cached[0], True
        grad, log_post = self.grad_fn(theta)
        if self._hash_cache is not None:
            self._hash_cache.put(theta, log_post, grad)
        return grad, log_post, False

    def _leapfrog(self, theta, p, epsilon, inv_mass, n_steps,
                  grad_logpost=None, inv_mass_t=None):
        """Leapfrog 辛积分器

        要点3b: ``grad_logpost`` 可选参数接受预计算的 (grad, log_post),
        避免 _leapfrog 内部重复调用 grad_fn(theta)。调用方 (_run_single_chain
        / _run_warmup) 在 leapfrog 前已计算 grad_fn(theta) 用于 H0,
        传入后 _leapfrog 跳过首次 grad_fn 调用。每次 leapfrog 节省
        1 次 grad_fn 调用 (n_leapfrog+1 → n_leapfrog), 提速 ~5%
        (n_leapfrog=20 时)。

        要点3 (第四轮): 返回值增加 final grad。原实现返回
        (theta, p, log_post) 后丢弃最终梯度, 但接受提议后下一轮发现
        ``theta is not last_theta`` 会重新调用 grad_fn 计算完全相同的
        梯度 (因 leapfrog 最后一步已对 theta_new 计算过 grad_fn)。
        返回 (theta, p, log_post, grad) 后, 调用方在接受时缓存该 grad,
        接受率 80% 场景下额外减少 80% 的 grad_fn 调用, 与现有拒绝缓存
        互补后接近零冗余。注意: 返回的 grad 与 theta 对应 (leapfrog
        最后一步的 grad_fn(theta) 结果), 可直接作为下一轮 H0 计算的
        预计算 grad 使用。

        要点6 (第五轮): ``inv_mass_t`` 可选参数接受预计算的 torch 张量,
        避免每次 leapfrog 调用从 numpy 数组重建。单链采样中 inv_mass
        固定不变, _run_single_chain 开始时预计算一次, n_samples × 1 次
        重建降为 0。warmup 中 inv_mass 偶尔更新 (质量矩阵估计), 更新时
        重算 inv_mass_t 即可。batch 模式 inv_mass 已是 (B, 15) 张量,
        无需此优化。
        """
        # 要点3 (第十三轮): theta/p 初始 .detach() 切断 autograd 历史。
        # HMC 的梯度计算在 compute_grad_log_posterior 中通过 torch.autograd.grad
        # 独立完成 (内部 y = theta.clone().detach().requires_grad_(True)),
        # 不依赖外部 theta 的梯度连接。leapfrog 中间步的 theta 更新链
        # (theta = theta + epsilon * inv_mass_t * p) 不需要跨步梯度——
        # HMC 的梯度仅在最终 theta 处计算一次, 中间步通过数值积分隐式传播。
        # 若不 detach, n_leapfrog 步的 theta 更新会累积 autograd 图
        # (每步一个 grad_fn 子图), 内存随 n_leapfrog 线性增长。
        # .clone().detach() 创建独立存储的新张量, 切断历史且不影响后续计算。
        # 不需 requires_grad_(True) — grad_fn 内部自行处理梯度启用。
        theta = theta.clone().detach()
        p = p.clone().detach()
        # 要点6: 优先使用预计算的 inv_mass_t, 否则从 inv_mass 重建 (向后兼容)
        if inv_mass_t is None:
            inv_mass_t = torch.tensor(
                inv_mass, dtype=theta.dtype, device=theta.device)

        if grad_logpost is not None:
            grad, log_post = grad_logpost
        else:
            grad, log_post = self.grad_fn(theta)
        p = p + 0.5 * epsilon * grad

        for i in range(n_steps - 1):
            theta = theta + epsilon * inv_mass_t * p
            grad, log_post = self.grad_fn(theta)
            p = p + epsilon * grad

        theta = theta + epsilon * inv_mass_t * p
        grad, log_post = self.grad_fn(theta)
        p = p + 0.5 * epsilon * grad

        return theta, p, log_post, grad

    def _hamiltonian(self, log_post, p, inv_mass, inv_mass_t=None):
        """H = -log_post + 0.5 * p^T M^{-1} p

        优先级2: 纯 torch 计算, 避免 p.detach().cpu().numpy() 的
        CPU-GPU 传输开销 (GPU 场景下每次 leapfrog 步都调用,
        传输 15D 动量向量开销显著)。

        要点1 (第六轮): 返回 tensor 而非 python float。原实现
        ``return float(-log_post + kinetic)`` 在 GPU 环境下从 CUDA
        张量提取 Python float 触发 GPU-CPU 同步。warmup 和
        single_chain 每步调用两次 _hamiltonian (H0 和 H_new),
        n_samples=5000 时累积 10000 次同步。返回 tensor 后调用方
        用 ``torch.exp(H0 - H_new)`` 在设备上完成 alpha 计算, 与
        batch 模式 (``_hamiltonian_batch`` 返回 tensor) 完全对称。

        要点1 (第六轮): ``inv_mass_t`` 可选参数接收预计算的 torch
        张量, 避免每次调用 ``torch.as_tensor(inv_mass, ...)`` 从
        numpy 重建。与 ``_leapfrog`` 的要点6 修复对称。单链中
        inv_mass 固定 (采样) 或偶尔更新 (warmup), 调用方预计算
        inv_mass_t 一次传入, n_samples × 2 次重建降为 0。
        """
        if inv_mass_t is None:
            inv_mass_t = torch.as_tensor(
                inv_mass, dtype=p.dtype, device=p.device)
        kinetic = 0.5 * torch.sum(inv_mass_t * p * p)
        # 要点1: 返回 tensor (不转 float), 避免 GPU-CPU 同步
        # 要点2 (第十三轮): 防御性 .detach()。H 值用于接受/拒绝决策和标量比较,
        # 不应参与 autograd 图。即使输入 log_post 或 p 意外携带梯度 (当前
        # compute_grad_log_posterior 已返回 .detach(), 但防御性措施确保未来
        # 修改 grad_fn 时不引入梯度泄漏), H 值也不携带梯度。
        return (-log_post + kinetic).detach()

    def _find_reasonable_epsilon(self, theta, inv_mass, gen=None,
                                  inv_mass_t=None):
        """Hoffman & Gelman 2014 Algorithm 4: 寻找合理初始步长

        要点5 (第四轮): 与全局缓存策略一致。原实现直接调用
        ``self.grad_fn(theta)`` 后调用 ``_leapfrog`` (未传入 grad_logpost),
        _leapfrog 内部重复计算首次梯度。该方法每链仅调用一次、影响有限,
        但与采样/warmup 路径的缓存模式不统一。修改为: 将首次 grad_fn
        结果传入 _leapfrog 的 grad_logpost 参数, 与 _run_single_chain
        / _run_warmup 保持统一模式, 每次 _leapfrog 调用省 1 次 grad_fn。

        要点1/2 (第六轮): 用 torch.Generator + torch.randn 生成动量
        (与 batch 模式统一); H0/H_new 为 tensor (不转 float, 避免
        GPU 同步); log_ratio 用 .item() 提取用于标量比较 (每链仅
        1-2 次 .item(), 远少于原实现的 4 次/步)。

        要点2 (第十五轮): 动量乘 sqrt_mass = sqrt(1/inv_mass_t) 保证从
        N(0, M) 采样 (Hoffman & Gelman 2014 Algorithm 4 标准实现)。
        当前 inv_mass_init 通常为全 1 (sqrt_mass=1.0 无影响), 但防御性
        正确性确保 future inv_mass_init 非全 1 时与 batch 版本一致。
        """
        epsilon = 1.0
        # 要点2: 用 torch.randn + generator (与 batch 模式统一)
        if gen is None:
            gen = torch.Generator(device=theta.device)
            gen.manual_seed(int(self.seed))
        p = torch.randn(self.n_params, generator=gen,
                        device=theta.device, dtype=theta.dtype)
        # 要点2 (第十五轮): 乘 sqrt_mass 保证 N(0, M) 采样 (与 batch 版本一致)
        if inv_mass_t is None:
            inv_mass_t = torch.as_tensor(
                inv_mass, dtype=p.dtype, device=p.device)
        p = p * torch.sqrt(1.0 / inv_mass_t)
        # 改进15 (六): 查 hash cache (此函数每链仅调用一次, 收益有限,
        # 但与采样/warmup 路径统一)
        grad, log_post_0, _ = self._cached_grad_log_post(theta, None)
        # 要点1: H0 为 tensor (传入预计算 inv_mass_t)
        H0 = self._hamiltonian(log_post_0, p, inv_mass, inv_mass_t=inv_mass_t)

        # 要点5: 传入预计算 grad, 避免 _leapfrog 首次 grad_fn 重复调用
        # 要点3: _leapfrog 现返回 4-tuple (theta, p, log_post, grad)
        # 要点6: 传入预计算 inv_mass_t
        theta_new, p_new, log_post_new, _ = self._leapfrog(
            theta, p, epsilon, inv_mass, 1,
            grad_logpost=(grad, log_post_0), inv_mass_t=inv_mass_t)
        H_new = self._hamiltonian(
            log_post_new, p_new, inv_mass, inv_mass_t=inv_mass_t)

        # 要点1: log_ratio 为 tensor, .item() 提取用于标量比较
        log_ratio = (H0 - H_new).item()
        a = 1.0 if log_ratio > np.log(0.5) else -1.0

        for _ in range(50):
            epsilon *= (2.0 ** a)
            if epsilon < 1e-10 or epsilon > 1e5:
                break
            # 要点5: 循环内也传入 grad_logpost (grad 未变, 复用首次结果)
            theta_new, p_new, log_post_new, _ = self._leapfrog(
                theta, p, epsilon, inv_mass, 1,
                grad_logpost=(grad, log_post_0), inv_mass_t=inv_mass_t)
            H_new = self._hamiltonian(
                log_post_new, p_new, inv_mass, inv_mass_t=inv_mass_t)
            log_ratio = (H0 - H_new).item()
            if a * log_ratio > -a * np.log(2):
                break

        return max(epsilon, 1e-6)

    def _find_reasonable_epsilon_batch(self, theta, inv_mass_t, gens):
        """Hoffman & Gelman 2014 Algorithm 4 的 batch 版本: 同时为 B 条链搜索合理初始步长

        要点1 (第十四轮): 消除 _run_warmup_batch 中最后一处串行瓶颈。
        原实现 _run_warmup_batch 第664-669 行用 Python for 循环逐链调用
        _find_reasonable_epsilon, B=16 链时 16 次串行调用, 每次含最多 50 次
        leapfrog+hamiltonian 计算。本方法同时对 B 条链搜索步长。

        核心挑战: 每条链的 a 方向 (log_ratio > log(0.5) 判断) 和循环终止
        条件独立——某些链 3 步收敛, 另一些需 50 步。用 per-chain active mask
        控制哪些链仍在搜索, inactive 链 epsilon 保持不变 (torch.where 仅
        更新 active 链)。

        实现策略:
          - per-chain active mask (B,) bool 张量, 初始全 True
          - 每步: epsilon = where(active, epsilon * 2^a, epsilon)
          - 越界检查: epsilon < 1e-10 或 > 1e5 的链置 inactive
          - 收敛检查: a*log_ratio > -a*log(2) 的链置 inactive
          - 要点1 (第十五轮): device 自适应循环策略
            * GPU: 跑满 50 步不提前退出, 避免 active.any() 触发 GPU-CPU
              同步 (50 步 batch leapfrog 计算量远小于 1 次 GPU 同步开销)
            * CPU: 用 active.any().item() 提前退出 (CPU 上 .item() 无同步
              开销, inactive 链的 leapfrog 计算是纯冗余, B=4 链时若 3 链
              在第 5 步收敛、1 链在第 40 步收敛, 则 35 步中有 3 链做无用计算)

        Args:
            theta: (B, 15) 张量, 每链初始参数
            inv_mass_t: (B, 15) 张量, 每链质量矩阵倒数
            gens: list of B 个 torch.Generator (per-chain 随机数)
        Returns:
            epsilon: (B,) 张量, 每链合理初始步长 (>= 1e-6)
        """
        B = theta.shape[0]
        dtype = theta.dtype
        device = theta.device

        # 动量 (per-chain generator, 列表推导 — 要点3 文档说明)
        # 要点2 (第十五轮): 乘 sqrt_mass = sqrt(1/inv_mass_t) 保证从 N(0, M) 采样。
        # 当前 inv_mass_init 通常为全 1 (sqrt_mass=1.0 无影响), 但防御性正确性
        # 确保 future inv_mass_init 非全 1 时 batch/串行均正确 (Hoffman &
        # Gelman 2014 Algorithm 4 标准实现应从 N(0, M) 采样动量)。
        p_list = [
            torch.randn(self.n_params, generator=g,
                        device=device, dtype=dtype)
            for g in gens
        ]
        p = torch.stack(p_list)  # (B, 15)
        p = p * torch.sqrt(1.0 / inv_mass_t)  # (B, 15) 广播

        # H0 (batch)
        grad, log_post_0 = self.grad_fn(theta)  # (B, 15), (B,)
        H0 = self._hamiltonian_batch(
            log_post_0, p, inv_mass_t, inv_mass_t=inv_mass_t)  # (B,)

        # 初始 epsilon = 1.0
        epsilon = torch.ones(B, dtype=dtype, device=device)  # (B,)

        # 第一次 leapfrog 1步 (所有链)
        nspc_one = torch.ones(B, device=device, dtype=dtype)
        theta_new, p_new, log_post_new, _ = self._leapfrog_batch(
            theta, p, epsilon, inv_mass_t, 1,
            n_steps_per_chain=nspc_one,
            grad_logpost=(grad, log_post_0), inv_mass_t=inv_mass_t)
        H_new = self._hamiltonian_batch(
            log_post_new, p_new, inv_mass_t, inv_mass_t=inv_mass_t)  # (B,)

        # a 方向: log_ratio > log(0.5) → a=1, 否则 a=-1
        log_ratio = H0 - H_new  # (B,)
        log_half = torch.tensor(
            float(np.log(0.5)), dtype=dtype, device=device)
        a_t = torch.where(
            log_ratio > log_half,
            torch.ones(B, dtype=dtype, device=device),
            -torch.ones(B, dtype=dtype, device=device))  # (B,)

        # per-chain active mask
        active = torch.ones(B, dtype=torch.bool, device=device)
        log_two = torch.tensor(
            float(np.log(2.0)), dtype=dtype, device=device)
        eps_min = torch.tensor(1e-10, dtype=dtype, device=device)
        eps_max = torch.tensor(1e5, dtype=dtype, device=device)

        # 要点1 (第十五轮): device 自适应循环策略
        # CPU: .item() 无同步开销, 提前退出省 inactive 链冗余计算
        # GPU: .item() 触发同步, 跑满 50 步避免 (50 步计算 < 1 次同步)
        # 要点1 (第十六轮): Python `and` 短路求值保证 GPU 环境下
        # `active.any().item()` 不执行 — early_exit=False 时 and 左操作数
        # 已为 False, 右操作数 `not active.any().item()` 根本不会被求值,
        # 因此 GPU 环境 active.any() 张量计算也不会发生, 无同步开销。
        # 此注释防止未来维护者误认为 GPU 环境也有同步风险。
        early_exit = (device.type == 'cpu')

        # 要点5 (第十五轮): 50 步循环中 grad_fn 调用次数 = 50 × 1 = 50 次
        # (每步 _leapfrog_batch 内部对新的 theta 调用 grad_fn, theta 每步不同
        # 因 epsilon 变化, 无法缓存)。首次 grad_fn 结果 (grad, log_post_0) 通过
        # grad_logpost 参数复用, 仅省每步首次 grad_fn 调用。
        for _ in range(50):
            # 仅更新 active 链的 epsilon
            epsilon = torch.where(
                active, epsilon * (2.0 ** a_t), epsilon)

            # 越界检查: 越界的链置 inactive (epsilon 保持越界值)
            out_of_bounds = (epsilon < eps_min) | (epsilon > eps_max)
            active = active & ~out_of_bounds

            # CPU 提前退出: 所有链 inactive 则结束
            # (Python and 短路: early_exit=False 时 active.any().item() 不执行)
            if early_exit and not active.any().item():
                break

            # 重新 leapfrog 1步 (batch, inactive 链结果被丢弃)
            theta_new, p_new, log_post_new, _ = self._leapfrog_batch(
                theta, p, epsilon, inv_mass_t, 1,
                n_steps_per_chain=nspc_one,
                grad_logpost=(grad, log_post_0), inv_mass_t=inv_mass_t)
            H_new = self._hamiltonian_batch(
                log_post_new, p_new, inv_mass_t, inv_mass_t=inv_mass_t)

            # 收敛检查: a * log_ratio > -a * log(2)
            log_ratio = H0 - H_new  # (B,)
            converged = (a_t * log_ratio) > (-a_t * log_two)
            active = active & ~converged

            # CPU 提前退出: 所有链收敛则结束
            # (Python and 短路: early_exit=False 时 active.any().item() 不执行)
            if early_exit and not active.any().item():
                break

        # 最终 epsilon 至少 1e-6 (与串行版本一致)
        eps_floor = torch.tensor(1e-6, dtype=dtype, device=device)
        epsilon = torch.maximum(epsilon, eps_floor)
        return epsilon  # (B,)

    def _run_warmup(self, theta_init, inv_mass_init, chain_seed=None,
                    epsilon_scale=1.0):
        """预热阶段: 对偶平均法自适应步长 + 质量矩阵估计 + 自适应 n_leapfrog

        要点B: 在对偶平均法之外，额外维护一个滑动窗口的接受率序列，
        每隔 ``self.leapfrog_adjust_interval`` 步检查窗口平均：
          - > 0.95 → n_leapfrog += 2 (轨迹过短)
          - < 0.65 → n_leapfrog = max(min, n-2) (可能 U-turn)
        warmup 结束时锁定当前 n_leapfrog，供采样阶段使用。

        要点2 (第三轮): GPU 预分配 warmup 样本张量。原实现每接受一步
        都 ``warmup_samples.append(theta.detach().cpu().numpy())``,
        在 GPU 环境下产生 O(n_warmup) 次 GPU-CPU 传输 (1000-2000 次)。
        改为在 GPU 上预分配 ``warmup_gpu`` 张量, 接受时原地写入,
        质量矩阵更新时用 ``torch.var`` 在 GPU 上计算方差, 仅传输
        15 维结果到 CPU。传输次数从 O(n_warmup) 降到 0 (质量矩阵
        完全在 GPU 计算), 仅结束时传输一次完整 warmup 样本。

        要点3b: 似然缓存 (与 _run_single_chain 对称)。拒绝时 theta 不变,
        复用缓存的 grad/log_post 避免重复 grad_fn 调用。

        要点3 (第四轮): 接受时缓存 leapfrog 的 final grad。原实现接受
        后下一轮发现 theta is not last_theta 重新调用 grad_fn, 但
        leapfrog 最后一步已对 theta_new 计算过 grad_fn。现在 _leapfrog
        返回 (theta, p, log_post, grad), 接受时缓存该 grad, 下一轮
        命中缓存, 与拒绝缓存互补后接近零冗余。

        要点1/2 (第六轮): 串行模式与 batch 模式完全对称 — 用
        ``torch.Generator`` (基于 chain_seed) + ``torch.randn`` 生成
        动量 (消除 numpy rng 的 CPU 生成→GPU 传输); H0/H_new 为
        tensor (不转 float, 避免 GPU 同步); alpha 用 ``torch.exp`` +
        ``torch.clamp`` 在设备上计算; 接受/拒绝用 ``torch.rand``
        (各链独立 generator)。原实现每步 4 次 GPU-CPU 同步
        (H0 float + H_new float + isnan + isinf), 优化后仅 1 次
        (alpha.item() 用于对偶平均法标量计算)。

        要点1/3/6 (第十一轮): 串行模式全 tensor 化, 消除每步所有 GPU-CPU 同步。
        原实现 (第六轮后) 每步仍 3 次同步: (1) bool(isfinite(log_post_new))
        (2) alpha_t.item() (对偶平均法标量) (3) bool(rand < alpha)。
        tensor 化: alpha 为 0-dim tensor (torch.where 处理 is_valid);
        接受/拒绝用 torch.where (theta_new * mask + theta * (1-mask));
        对偶平均法状态量 (H_bar/log_epsilon/log_epsilon_bar/mu/epsilon) 全部
        tensor 化, 用 torch 运算更新; warmup_gpu_full 每步写入 theta +
        accept_mask 记录接受/拒绝 (替代仅写接受样本, 避免 n_accepted 同步);
        质量矩阵用 masked variance (滑动窗口 tensor 运算, 无 .cpu().numpy());
        自适应 leapfrog 用 alpha_history tensor, 仅检查点 (每 interval 步)
        1 次 .mean().item() 同步。循环内零同步 (除自适应 leapfrog 检查点),
        结束时 3 次同步提取标量 (epsilon/inv_mass/warmup_samples)。

        要点6 (第十轮): warmup alpha.item() GPU 优化评估。
        当前每步 1 次 alpha.item() 同步 (用于对偶平均法标量计算),
        n_warmup=1000 时累积 1000 次同步。完全消除需:
          1. 对偶平均法状态量 (H_bar/log_epsilon_bar/mu) 张量化
          2. alpha 保持 tensor, 用 torch 运算更新状态量
          3. 接受/拒绝用 torch.where 替代 bool() (零同步)
          4. warmup_gpu 每步写入 (替代接受时写入, 避免 n_accepted 标量)
          5. 缓存逻辑用 cache_valid 替代 theta is last_theta
             (torch.where 使 theta 总是新对象)
          6. 质量矩阵估计用所有步骤 (含重复), 而非仅接受样本
        评估后决定暂不实现完整 GPU 零同步路径, 理由:
          - 改动复杂度高 (约 80 行新代码, 两条路径分叉)
          - GPU 路径无法在 CPU 环境充分测试
          - warmup 仅占总时间 10-20% (n_warmup=1000 vs n_samples=5000)
          - 采样阶段已零同步 (batch 模式), warmup 的 1 次/步同步
            对总时间影响有限
          - 对偶平均法标量运算用 torch 张量有 kernel 启动开销,
            15 维小张量 CPU 可能更快 (用户已指出此权衡)
        当前实现 (每步 1 次 .item()) 已从原 4 次同步大幅降低,
        与采样阶段的零同步基本对称。若未来 GPU profiling 显示
        warmup 同步成为瓶颈, 可参考上述方案实现完整 GPU 路径。

        要点6 (第十一轮): 已实施上述 1-4 项完整 GPU 零同步路径。
        对偶平均法状态量张量化 (H_bar_t/log_epsilon_t/log_epsilon_bar_t/mu_t);
        alpha 保持 0-dim tensor; 接受/拒绝用 torch.where; warmup_gpu_full
        每步写入 + accept_mask 过滤。第 5 项: 缓存逻辑保留 identity 检查
        (theta is last_theta), 因 torch.where 后 last_theta=theta 使下一轮
        identity 命中 (见下文注释)。第 6 项: 质量矩阵用 masked variance
        (滑动窗口 accept_mask 过滤, 仅用接受样本, 保持原行为)。
        自适应 leapfrog 保留每 interval 1 次 .mean().item() 同步 (需 Python
        int 传给 _leapfrog 的 n_steps)。循环内零同步, 结束时 3 次标量提取。
        CPU 环境 tensor 运算有轻微 kernel 启动开销但可忽略 (15 维小张量),
        GPU 环境消除 n_warmup × 3 次同步, 收益显著。
        """
        theta = theta_init.clone()
        inv_mass = inv_mass_init.copy()

        # 要点6 (第五轮): 预计算 inv_mass_t, warmup 中 inv_mass 偶尔更新时重算
        inv_mass_t = torch.tensor(
            inv_mass, dtype=theta.dtype, device=theta.device)

        # 要点2 (第六轮): torch.Generator (基于 chain_seed, 与 batch 模式统一)
        if chain_seed is None:
            chain_seed = int(self.seed)
        gen = torch.Generator(device=theta.device)
        gen.manual_seed(int(chain_seed))

        # 要点6 (第十一轮): epsilon 及对偶平均法状态量全部 tensor 化, 消除每步 .item() 同步。
        # 原实现每步 3 次 GPU-CPU 同步: (1) bool(isfinite(log_post_new))
        # (2) alpha_t.item() (对偶平均法标量) (3) bool(rand < alpha)。
        # tensor 化后: alpha 为 0-dim tensor, torch.where 处理 is_valid 与接受/拒绝,
        # 对偶平均法用 tensor 运算更新 H_bar/log_epsilon/log_epsilon_bar,
        # 仅自适应 leapfrog 检查点 (每 leapfrog_adjust_interval 步) 1 次 .item() 同步。
        epsilon_init = self._find_reasonable_epsilon(
            theta, inv_mass, gen=gen, inv_mass_t=inv_mass_t)
        # 低接受率重试路径的初始步长缩放（与 _run_warmup_batch 对称：
        # epsilon_scale != 1 时缩放初始 epsilon，重试用更小步长）
        if epsilon_scale != 1.0:
            epsilon_init = epsilon_init * float(epsilon_scale)
        epsilon_t = torch.tensor(epsilon_init, dtype=theta.dtype, device=theta.device)
        mu_t = torch.log(10.0 * epsilon_t)  # tensor
        log_epsilon_bar_t = torch.zeros((), dtype=theta.dtype, device=theta.device)
        H_bar_t = torch.zeros((), dtype=theta.dtype, device=theta.device)
        gamma = 0.05  # Python float (常量, 不需 tensor)
        t0 = 10.0
        kappa = 0.75
        target_accept = float(self.target_accept)

        # 要点6 (第十一轮): warmup_gpu_full 每步写入 theta (无论接受/拒绝),
        # accept_mask 记录每步是否接受。结束时用 accept_mask 过滤出接受样本。
        # 这避免了 "仅写接受样本" 需要的 n_accepted 标量同步。
        warmup_gpu_full = torch.zeros(
            self.n_warmup, self.n_params,
            device=theta.device, dtype=theta.dtype)
        accept_mask = torch.zeros(
            self.n_warmup, dtype=torch.bool, device=theta.device)
        n_accept_t = torch.zeros((), dtype=theta.dtype, device=theta.device)
        # alpha_history: 自适应 leapfrog 用, 避免每步 float(alpha) 同步
        alpha_history = torch.zeros(
            self.n_warmup, dtype=theta.dtype, device=theta.device)

        n_mass_start = max(self.n_warmup // 4, 50)

        # 要点B: 自适应 leapfrog 状态
        current_n_leapfrog = int(self.n_leapfrog)

        # 要点3b: 似然缓存
        last_theta = None
        last_grad = None
        last_log_post = None

        # 要点2 (第六轮): 动量缩放 sqrt(mass) = sqrt(1/inv_mass)
        sqrt_mass = torch.sqrt(1.0 / inv_mass_t)

        # 要点1/3/6 (第十一轮): 零同步常量张量 (避免循环内重复创建)
        zeros_alpha = torch.zeros((), dtype=theta.dtype, device=theta.device)
        var_floor = torch.tensor(1e-8, dtype=theta.dtype, device=theta.device)

        for m in range(1, self.n_warmup + 1):
            # 要点4 (第十三轮): 防御性 theta.detach() 切断 autograd 历史。
            # n_warmup 步的 theta 通过 torch.where 更新会累积 autograd 图
            # (若 theta_new 携带梯度)。当前 theta_new 来自 _leapfrog (要点3
            # 已内部 detach), theta_init 来自 as_tensor().detach().clone()
            # (第十二轮), 正常情况下 theta 不 requires_grad。但防御性 detach
            # 提供显式保证。关键: detach() 在 requires_grad=False 时返回同一
            # 对象 (不创建新张量), 不影响 identity 缓存 (theta is last_theta
            # 仍为 True)。仅在 theta 意外 requires_grad 时创建新对象 (此时
            # identity 缓存应失效, 需重新计算梯度)。不需 requires_grad_(True)
            # — grad_fn 内部 (compute_grad_log_posterior) 自行处理梯度启用。
            theta = theta.detach()
            # 要点2 (第六轮): torch.randn + generator (无 CPU→GPU 传输)
            p = torch.randn(self.n_params, generator=gen,
                            device=theta.device, dtype=theta.dtype) * sqrt_mass

            # 要点3b: theta 未变时复用缓存的 grad/log_post
            # 改进15 (六): identity 未命中时查 hash cache (opt-in)
            if last_theta is not None and theta is last_theta:
                grad, log_post_0 = last_grad, last_log_post
            else:
                grad, log_post_0, _ = self._cached_grad_log_post(theta, None)
                last_theta = theta
                last_grad = grad
                last_log_post = log_post_0
            # 要点1 (第六轮): H0 为 tensor (传入预计算 inv_mass_t)
            H0 = self._hamiltonian(
                log_post_0, p, inv_mass, inv_mass_t=inv_mass_t)

            # 要点3b: 传入预计算的 grad, 避免 _leapfrog 首次 grad_fn 重复调用
            # 要点3 (第四轮): _leapfrog 返回 4-tuple, final_grad 为最终梯度
            # 要点6 (第五轮): 传入预计算的 inv_mass_t, 避免每次 leapfrog 重建
            # 要点6 (第十一轮): epsilon_t 为 0-dim tensor, _leapfrog 支持 tensor epsilon
            theta_new, p_new, log_post_new, final_grad = self._leapfrog(
                theta, p, epsilon_t, inv_mass, current_n_leapfrog,
                grad_logpost=(grad, log_post_0), inv_mass_t=inv_mass_t)

            # 要点1 (第十一轮): alpha 为 0-dim tensor (无 .item() 同步)
            # torch.where 选择: 有效时 clamp(exp(H0-H_new), max=1), 无效时 0
            # H_new 即使 log_post_new 无效也会计算 (NaN), 但 torch.where 选择 0 分支
            is_valid = torch.isfinite(log_post_new)
            H_new = self._hamiltonian(
                log_post_new, p_new, inv_mass, inv_mass_t=inv_mass_t)
            alpha_t = torch.where(
                is_valid,
                torch.clamp(torch.exp(H0 - H_new), max=1.0),
                zeros_alpha
            )  # 0-dim tensor

            # 要点3 (第十一轮): 接受/拒绝用 torch.where (无 bool() 同步)
            # 与 batch 模式 (theta_new * mask + theta * (1-mask)) 完全对称
            rand_val = torch.rand((), generator=gen,
                                  device=theta.device, dtype=theta.dtype)
            accept_t = rand_val < alpha_t  # 0-dim bool tensor
            mask_t = accept_t.to(theta.dtype)  # 0-dim float
            theta = theta_new * mask_t + theta * (1.0 - mask_t)

            # 要点3 (第四轮) + 要点3 (第十一轮): grad 缓存用 tensor merge
            # 接受: 缓存 final_grad/log_post_new (theta_new 的梯度/对数后验)
            # 拒绝: 缓存 grad/log_post_0 (旧 theta 的梯度/对数后验, 仍有效)
            # torch.where 后 theta 总是新对象, 设 last_theta=theta 使下一轮
            # identity 缓存命中 (theta is last_theta 为 True)
            last_grad = final_grad * mask_t + grad * (1.0 - mask_t)
            last_log_post = log_post_new * mask_t + log_post_0 * (1.0 - mask_t)
            last_theta = theta

            # 要点6 (第十一轮): 每步写入 warmup_gpu_full + accept_mask (无 n_accepted 同步)
            warmup_gpu_full[m - 1] = theta
            accept_mask[m - 1] = accept_t
            n_accept_t += mask_t
            alpha_history[m - 1] = alpha_t

            # 要点6 (第十一轮): 对偶平均法全部 tensor 运算 (无 .item() 同步)
            eta = 1.0 / (m + t0)  # Python float
            H_bar_t = (1.0 - eta) * H_bar_t + eta * (target_accept - alpha_t)
            log_epsilon_t = mu_t - (np.sqrt(m) / gamma) * H_bar_t
            epsilon_t = torch.exp(log_epsilon_t)
            x_eta = m ** (-kappa)  # Python float
            log_epsilon_bar_t = (x_eta * log_epsilon_t
                                 + (1.0 - x_eta) * log_epsilon_bar_t)

            # 要点B: 自适应 n_leapfrog (路径二)
            # 要点6 (第十一轮): alpha_history 累积 tensor, 仅检查点 1 次 .mean().item() 同步
            # 原 recent_alphas.append(float(alpha)) 每步 1 次同步, 降至每 interval 1 次
            if (self.adaptive_leapfrog
                    and m % self.leapfrog_adjust_interval == 0):
                start_idx = max(0, m - self.leapfrog_adjust_interval)
                mean_alpha = float(alpha_history[start_idx:m].mean())
                if mean_alpha > 0.95:
                    # 接受率过高: 轨迹过短, 增加 leapfrog 步数
                    current_n_leapfrog = min(
                        current_n_leapfrog + 2, self.leapfrog_max)
                elif mean_alpha < 0.65:
                    # 接受率过低: 可能 U-turn, 减少 leapfrog 步数
                    current_n_leapfrog = max(
                        current_n_leapfrog - 2, self.leapfrog_min)

            # 要点6 (第十一轮): 质量矩阵估计全部 tensor 运算 (无 .cpu().numpy() 同步)
            # 原 var.cpu().numpy() 每步 1 次同步, 改为 tensor 更新 inv_mass_t
            # 用 accept_mask 过滤滑动窗口内的接受样本, masked variance 公式:
            #   mean = sum(theta * mask) / sum(mask)
            #   var = sum((theta - mean)^2 * mask) / sum(mask)
            # 条件更新 (n_eff > 5) 用 torch.where 避免 Python if 同步
            if m >= n_mass_start:
                win_start = max(0, m - 200)
                mask_slice = accept_mask[win_start:m].to(theta.dtype).unsqueeze(1)
                theta_slice = warmup_gpu_full[win_start:m]
                n_eff = mask_slice.sum()  # (1,) tensor
                n_eff_safe = n_eff.clamp(min=1.0)
                mean = (theta_slice * mask_slice).sum(dim=0) / n_eff_safe
                diff_sq = (theta_slice - mean.unsqueeze(0)) ** 2
                var = (diff_sq * mask_slice).sum(dim=0) / n_eff_safe
                var = torch.maximum(var, var_floor)
                # 条件更新: n_eff > 5 时更新 inv_mass_t
                update = (n_eff.squeeze() > 5).to(theta.dtype)  # 0-dim
                new_inv_mass = 0.5 * inv_mass_t + 0.5 * var
                inv_mass_t = inv_mass_t * (1.0 - update) + new_inv_mass * update
                sqrt_mass = torch.sqrt(1.0 / inv_mass_t)

        # 要点6 (第十一轮): 结束时一次性提取标量 (仅 3 次同步, vs 原 3*n_warmup 次)
        epsilon_final = float(torch.exp(log_epsilon_bar_t).item())
        inv_mass = inv_mass_t.cpu().numpy()
        n_accepted_final = int(n_accept_t.item())
        warmup_samples = (warmup_gpu_full[accept_mask].cpu().numpy()
                          if n_accepted_final > 0 else None)
        return epsilon_final, inv_mass, \
               warmup_samples, current_n_leapfrog

    def _run_warmup_batch(self, theta_inits, inv_mass_inits, chain_seeds,
                          epsilon_scale=1.0):
        """Batch warmup: B 条链同时对偶平均法自适应步长 + 质量矩阵 + 自适应 n_leapfrog

        要点1 (第十三轮): 实现 warmup batch 化, 消除上一轮要点3 遗留的性能瓶颈。
        原实现 run() 用 Python for 循环串行执行每条链的 warmup, 4 链时 warmup
        总时间是单链的 4 倍。本方法同时对 B 条链执行 warmup, 复用现有 batch
        基础设施 (_leapfrog_batch / _hamiltonian_batch / per-chain Generator)。

        核心挑战: 每条链状态独立 — 步长 epsilon(B,)、H_bar(B,)、
        log_epsilon_bar(B,)、inv_mass(B,15)、current_n_leapfrog(B,) 均为
        per-chain 向量。对偶平均法标量更新全部改为 (B,) 张量运算。

        实现策略:
          - 初始化阶段 batch (要点1 第十四轮): _find_reasonable_epsilon_batch
            同时为 B 链搜索步长, per-chain active mask 控制独立收敛
          - warmup 循环 batch: _leapfrog_batch 同时积分 B 链 (per-chain
            n_steps_per_chain mask 控制各链不同步数)
          - per-chain 接受/拒绝: torch.where 对 (B,15) theta 操作
          - 对偶平均法: (B,) 张量运算 (H_bar_t/epsilon_t/log_epsilon_bar_t)
          - 质量矩阵: per-chain 滑动窗口方差 (B,win,15) → (B,15)
          - 自适应 leapfrog (要点2 第十四轮): torch.where + torch.clamp 向量化
            调整, current_n_leapfrog_t 为 (B,) 张量, n_steps_max 同步跟踪

        收益: warmup 阶段加速 n_chains 倍 (CPU 4×, GPU 更高), warmup 占
        HMC 总时间 30-50%, 4 链时总时间可降 20-40%。

        Args:
            theta_inits: list of B 个 torch tensor (15,)
            inv_mass_inits: list of B 个 np.array (15,)
            chain_seeds: list of B 个 int
        Returns:
            epsilons (B,) numpy, inv_masses (B,15) numpy,
            warmup_samples list (每链), n_leapfrogs (B,) numpy

        要点4 (第十五轮): warmup_samples 返回格式文档统一。
            - _run_warmup (串行): 返回单个 np.array (n_accepted, 15) 或 None
            - _run_warmup_batch (本方法): 返回 list of B 个 np.array 或 None
            - sample() 当前用 _ 丢弃返回值, 诊断用途未启用
            - 要点3 (第十五轮) 环形缓冲区使 batch 版仅返回最近 buffer_size 步
              (非完整 warmup), 串行版仍返回完整 warmup; 若未来诊断用途启用
              且需完整 warmup, 需取消环形缓冲区或额外维护完整副本
        """
        B = len(theta_inits)
        theta = torch.stack(theta_inits)  # (B, 15)
        # 要点4 (第十三轮): 防御性 detach (与 _run_single_chain/_run_warmup 对称)
        theta = theta.detach()
        inv_mass_t = torch.tensor(
            np.array(inv_mass_inits), dtype=theta.dtype,
            device=theta.device)  # (B, 15)

        # 要点1: 每条链独立 torch.Generator (基于 chain_seed, 可复现)
        gens = [torch.Generator(device=theta.device) for _ in range(B)]
        for g, s in zip(gens, chain_seeds):
            g.manual_seed(int(s))

        # 初始化阶段 (batch): 同时为 B 条链搜索合理初始步长
        # 要点1 (第十四轮): 用 _find_reasonable_epsilon_batch 替代逐链串行调用,
        # 消除 batch warmup 的最后一处串行瓶颈 (B=16 链时加速 16 倍)
        epsilon_t = self._find_reasonable_epsilon_batch(
            theta, inv_mass_t, gens)  # (B,)
        # 低接受率重试路径的初始步长缩放（见 _run_warmup 同名参数）
        if epsilon_scale != 1.0:
            epsilon_t = epsilon_t * float(epsilon_scale)

        # 要点6 (第十一轮): 对偶平均法状态量全部 (B,) 张量化
        mu_t = torch.log(10.0 * epsilon_t)  # (B,)
        log_epsilon_bar_t = torch.zeros(
            B, dtype=theta.dtype, device=theta.device)  # (B,)
        H_bar_t = torch.zeros(
            B, dtype=theta.dtype, device=theta.device)  # (B,)
        gamma = 0.05
        t0 = 10.0
        kappa = 0.75
        target_accept = float(self.target_accept)

        # warmup_gpu_full: (buffer_size, B, 15) 环形缓冲区, 每步写入 theta
        # 要点3 (第十五轮): 改为环形缓冲区, 内存从 O(n_warmup) 降至 O(buffer_size)。
        # 质量矩阵仅用最近 200 步 (win_start = max(0, m-200)), 环形缓冲区
        # buffer_size = min(n_warmup, 200) 恰好覆盖。accept_mask 同步环形化
        # (与 warmup_gpu_full 配对用于 masked variance)。
        # 要点4 (第十六轮): alpha_history 保持完整 (n_warmup, B) 未环形化 —
        # alpha_history 用于自适应 leapfrog 的滑动窗口均值计算
        # (alpha_history[start_idx:m].mean(dim=0)), start_idx =
        # max(0, m - leapfrog_adjust_interval) 通常 leapfrog_adjust_interval=50
        # 远小于 n_warmup。未环形化原因: (1) 仅用最近 leapfrog_adjust_interval
        # 步, 但 start_idx 随 m 增长需访问早期历史; (2) 内存占用小
        # (n_warmup=5000 B=32 float64 仅 1.6MB, 远小于 warmup_gpu_full 的
        # n_warmup × B × 15 = 19MB); (3) 环形化需复杂索引管理, 收益有限。
        # warmup_samples_list 返回最近 buffer_size 步的接受样本 (乱序但顺序
        # 对诊断无影响); sample() 当前丢弃返回值, 未来诊断用途启用时需注意
        # 仅最近 buffer_size 步 (非完整 warmup)。
        buffer_size = min(self.n_warmup, 200)
        warmup_gpu_full = torch.zeros(
            buffer_size, B, self.n_params,
            device=theta.device, dtype=theta.dtype)  # (buffer_size, B, 15)
        accept_mask = torch.zeros(
            buffer_size, B, dtype=torch.bool,
            device=theta.device)  # (buffer_size, B)
        # 要点2 (第十六轮): 防御性断言保证 accept_mask 与 warmup_gpu_full
        # 长度同步。两者必须同为 buffer_size, 否则质量矩阵计算的切片会
        # 不一致。未来若修改其中一个的长度, 此断言会立即触发, 避免静默错误。
        assert accept_mask.shape[0] == warmup_gpu_full.shape[0] == buffer_size, \
            f"环形缓冲区长度不一致: accept_mask={accept_mask.shape[0]}, " \
            f"warmup_gpu_full={warmup_gpu_full.shape[0]}, buffer_size={buffer_size}"
        # 要点3 (第十六轮): n_accept_t 必须用浮点累加, 不能改为
        # torch.any(accept_mask, dim=0) — accept_mask 是 (buffer_size, B)
        # 环形缓冲区仅含最近 buffer_size 步, 而 n_accept_t 记录整个 warmup
        # 的接受总数。当 n_warmup > buffer_size 时, 某链早期接受但最近
        # buffer_size 步未接受, torch.any 会误报 False 而 n_accept_t > 0。
        # 浮点精度漂移可忽略 (n_warmup=5000 步 float64 累加 0/1 误差 < 1e-10)。
        n_accept_t = torch.zeros(
            B, dtype=theta.dtype, device=theta.device)  # (B,)
        alpha_history = torch.zeros(
            self.n_warmup, B, dtype=theta.dtype,
            device=theta.device)  # (n_warmup, B) 保持完整

        n_mass_start = max(self.n_warmup // 4, 50)

        # 要点2 (第十四轮): current_n_leapfrog 改为 (B,) 张量, 向量化调整逻辑
        # (原 Python list + for 循环改为 torch.where + torch.clamp)
        # 同步维护 n_steps_max (Python int) 避免 max() 的 GPU 同步
        current_n_leapfrog_t = torch.full(
            (B,), int(self.n_leapfrog),
            dtype=theta.dtype, device=theta.device)  # (B,)
        n_steps_max = int(self.n_leapfrog)  # Python int, 无同步

        # per-chain 缓存 (与 _run_sampling_batch 一致, Python bool)
        # 要点5 (第十四轮): cache_valid 语义统一文档。
        # cache_valid 表示 cached_grad/cached_log_post 是否与当前 theta 匹配。
        # warmup 和 sampling 中均为: 首步 False (强制调用 grad_fn 初始化缓存),
        # 之后恒为 True (接受时缓存 final_grad/log_post_new, 拒绝时缓存
        # pre-leapfrog grad/log_post_0, 两种情况缓存均有效)。
        # Python bool 替代 torch.bool 张量消除 bool(cache_valid.all()) 同步 (要点2 第十一轮)。
        cache_valid = False
        cached_grad = None
        cached_log_post = None

        # 动量缩放: sqrt(1/inv_mass) = sqrt(mass)
        sqrt_mass = torch.sqrt(1.0 / inv_mass_t)  # (B, 15)

        # 零同步常量张量
        zeros_alpha = torch.zeros(
            B, dtype=theta.dtype, device=theta.device)
        var_floor = torch.tensor(1e-8, dtype=theta.dtype, device=theta.device)

        for m in range(1, self.n_warmup + 1):
            # 要点4 (第十三轮): 防御性 theta.detach()
            theta = theta.detach()
            # 动量 (各链独立 generator)
            # 要点3 (第十四轮): 列表推导由 torch.randn 的 generator API 限制所致
            # (torch.randn 的 generator 参数只接受单个 Generator, 无法一次为 B 条链
            # 生成不同随机数)。torch.vmap 对 generator 支持有限; 单 generator + seed
            # 偏移方案会改变可复现性语义。当前实现是合理权衡, B=16 时列表推导开销
            # ~µs 级, 远小于 grad_fn 调用 (~ms 级)。
            p_list = [
                torch.randn(self.n_params, generator=g,
                            device=theta.device, dtype=theta.dtype)
                * sqrt_mass[b]
                for b, g in enumerate(gens)
            ]
            p = torch.stack(p_list)  # (B, 15)

            # per-chain 缓存 (与 _run_sampling_batch 一致)
            if cache_valid:
                grad = cached_grad
                log_post_0 = cached_log_post
            else:
                grad, log_post_0 = self.grad_fn(theta)  # (B, 15), (B,)

            H0 = self._hamiltonian_batch(
                log_post_0, p, inv_mass_t, inv_mass_t=inv_mass_t)  # (B,)

            # batch leapfrog (per-chain n_steps_per_chain)
            # 要点2 (第十四轮): n_steps_max 为 Python int (无 GPU 同步),
            # nspc 直接用 current_n_leapfrog_t 张量 (无需 torch.tensor 重建)
            n_steps = n_steps_max
            nspc = current_n_leapfrog_t
            theta_new, p_new, log_post_new, final_grad = self._leapfrog_batch(
                theta, p, epsilon_t, inv_mass_t, n_steps,
                n_steps_per_chain=nspc,
                grad_logpost=(grad, log_post_0),
                inv_mass_t=inv_mass_t)

            # per-chain 接受/拒绝
            is_valid = torch.isfinite(log_post_new)  # (B,)
            H_new = self._hamiltonian_batch(
                log_post_new, p_new, inv_mass_t, inv_mass_t=inv_mass_t)
            alpha_t = torch.where(
                is_valid,
                torch.clamp(torch.exp(H0 - H_new), max=1.0),
                zeros_alpha
            )  # (B,)

            rand_vals = torch.stack([
                torch.rand((), generator=g,
                           device=theta.device, dtype=theta.dtype)
                for g in gens
            ])  # (B,)
            accept_t = rand_vals < alpha_t  # (B,) bool
            mask_t = accept_t.unsqueeze(1).to(theta.dtype)  # (B, 1)
            theta = theta_new * mask_t + theta * (1.0 - mask_t)  # (B, 15)
            n_accept_t += accept_t.to(theta.dtype)

            # 更新 per-chain 缓存 (与 _run_sampling_batch 一致)
            cached_grad = torch.where(
                accept_t.unsqueeze(1), final_grad, grad)
            cached_log_post = torch.where(accept_t, log_post_new, log_post_0)
            cache_valid = True

            # 每步写入 warmup_gpu_full + accept_mask (环形缓冲区)
            # 要点3 (第十五轮): 用 (m-1) % buffer_size 索引写入
            buf_idx = (m - 1) % buffer_size
            warmup_gpu_full[buf_idx] = theta
            accept_mask[buf_idx] = accept_t
            alpha_history[m - 1] = alpha_t

            # 对偶平均法 (per-chain (B,) 张量运算)
            eta = 1.0 / (m + t0)
            H_bar_t = (1.0 - eta) * H_bar_t + eta * (target_accept - alpha_t)
            log_epsilon_t = mu_t - (np.sqrt(m) / gamma) * H_bar_t
            epsilon_t = torch.exp(log_epsilon_t)
            x_eta = m ** (-kappa)
            log_epsilon_bar_t = (x_eta * log_epsilon_t
                                 + (1.0 - x_eta) * log_epsilon_bar_t)

            # 自适应 leapfrog (per-chain, 每 interval 1 次同步)
            # 要点2 (第十四轮): 向量化调整逻辑 — torch.where + torch.clamp
            # 替代 Python for 循环 + mean_alpha.cpu().numpy()
            if (self.adaptive_leapfrog
                    and m % self.leapfrog_adjust_interval == 0):
                start_idx = max(0, m - self.leapfrog_adjust_interval)
                # alpha_history: (n_warmup, B), 切片 [start_idx:m] → (win, B)
                mean_alpha = alpha_history[start_idx:m].mean(dim=0)  # (B,)
                # 向量化: 高接受率 → +2 (clamp 上界), 低接受率 → -2 (clamp 下界)
                lf_max = float(self.leapfrog_max)
                lf_min = float(self.leapfrog_min)
                current_n_leapfrog_t = torch.where(
                    mean_alpha > 0.95,
                    torch.clamp(current_n_leapfrog_t + 2, max=lf_max),
                    torch.where(
                        mean_alpha < 0.65,
                        torch.clamp(current_n_leapfrog_t - 2, min=lf_min),
                        current_n_leapfrog_t))
                # 更新 n_steps_max (1 次同步/interval, 与原 mean_alpha.cpu().numpy() 同步次数相同)
                n_steps_max = int(current_n_leapfrog_t.max().item())

            # 质量矩阵 (per-chain 滑动窗口方差)
            # 要点3 (第十五轮): 环形缓冲区读取
            # - m < buffer_size: 缓冲区未满, 用 [0:m]
            # - m >= buffer_size: 缓冲区已满, 用整个缓冲区 (最近 buffer_size 步)
            #   注意: 环形写入使样本乱序, 但 mean/var 是顺序无关统计量, 无影响
            if m >= n_mass_start:
                if m < buffer_size:
                    mask_slice = accept_mask[:m].unsqueeze(2).to(
                        theta.dtype)  # (m, B, 1)
                    theta_slice = warmup_gpu_full[:m]  # (m, B, 15)
                else:
                    mask_slice = accept_mask.unsqueeze(2).to(
                        theta.dtype)  # (buffer_size, B, 1)
                    theta_slice = warmup_gpu_full  # (buffer_size, B, 15)
                n_eff = mask_slice.sum(dim=0)  # (B, 1)
                n_eff_safe = n_eff.clamp(min=1.0)
                mean = (theta_slice * mask_slice).sum(dim=0) / n_eff_safe  # (B, 15)
                diff_sq = (theta_slice - mean.unsqueeze(0)) ** 2  # (win, B, 15)
                var = (diff_sq * mask_slice).sum(dim=0) / n_eff_safe  # (B, 15)
                var = torch.maximum(var, var_floor)
                update = (n_eff.squeeze(1) > 5).to(theta.dtype).unsqueeze(1)  # (B, 1)
                new_inv_mass = 0.5 * inv_mass_t + 0.5 * var
                inv_mass_t = inv_mass_t * (1.0 - update) + new_inv_mass * update
                sqrt_mass = torch.sqrt(1.0 / inv_mass_t)

        # 结束时一次性提取标量 (每链独立)
        epsilon_final = torch.exp(log_epsilon_bar_t).cpu().numpy()  # (B,)
        inv_mass_final = inv_mass_t.cpu().numpy()  # (B, 15)
        n_accepted_final = n_accept_t.cpu().numpy().astype(int)  # (B,)
        # per-chain warmup_samples (要点3 第十五轮: 环形缓冲区仅含最近 buffer_size 步)
        # 注意: 样本顺序可能乱序 (环形写入), 但对诊断 (R-hat/ESS) 无影响。
        # 若需完整 warmup 样本, 需取消环形缓冲区改造或额外维护完整副本。
        warmup_samples_list = []
        accept_mask_np = accept_mask.cpu().numpy()  # (buffer_size, B)
        warmup_gpu_full_np = warmup_gpu_full.cpu().numpy()  # (buffer_size, B, 15)
        for b in range(B):
            if n_accepted_final[b] > 0:
                warmup_samples_list.append(
                    warmup_gpu_full_np[accept_mask_np[:, b], b])
            else:
                warmup_samples_list.append(None)
        # 要点2 (第十四轮): current_n_leapfrog_t 已是张量, 直接 .cpu().numpy()
        n_leapfrog_final = current_n_leapfrog_t.cpu().numpy().astype(np.int64)

        return epsilon_final, inv_mass_final, \
               warmup_samples_list, n_leapfrog_final

    def _run_single_chain(self, theta_init, epsilon, inv_mass, chain_seed,
                          n_leapfrog=None):
        """单链 HMC 采样 (固定步长)

        要点B: ``n_leapfrog`` 参数允许传入 warmup 阶段锁定的最优值。
        未传入时回退到 ``self.n_leapfrog`` (向后兼容)。

        要点1 (第三轮): GPU 预分配样本张量。原实现每步
        ``samples[i] = theta.detach().cpu().numpy()`` 在 GPU 环境下
        产生 ``n_samples`` 次 GPU-CPU 同步 (5000 步 × 1 次 = 5000 次传输)。
        改为在 GPU 上预分配 ``samples_gpu``/``log_probs_gpu`` 张量,
        每步原地写入, 结束后一次性 ``.cpu().numpy()`` 传输。与 batch
        模式的优化完全对称。CPU 环境下 detach().cpu().numpy() 实际不
        涉及 GPU 传输, 开销很小, 但预分配仍消除了 ``n_samples`` 次
        张量到数组的转换开销。

        要点3b: 似然缓存。HMC 接受/拒绝机制保证: 当样本被拒绝时,
        ``theta`` 不变, 下一轮的 ``grad_fn(theta)`` 计算的是同一
        ``theta`` 的梯度和 log_post, 与上一轮完全相同, 重复计算
        浪费 ODE 积分。维护 ``last_theta/last_grad/last_log_post``
        缓存, 拒绝时直接复用, 接受时更新。CPU 环境下减少 grad_fn
        调用次数 (高拒绝率场景可减 30-50%); GPU 环境下额外减少
        GPU kernel 启动开销。

        要点3 (第四轮): 接受时缓存 leapfrog 的 final grad。原实现接受
        后下一轮 ``theta is not last_theta`` 重新调用 grad_fn, 但
        leapfrog 最后一步已对 theta_new 计算过 grad_fn。现在 _leapfrog
        返回 (theta, p, log_post, grad), 接受时缓存该 grad, 下一轮
        命中缓存, 接受率 80% 场景下额外减少 80% 的 grad_fn 调用,
        与拒绝缓存互补后接近零冗余。

        要点1/2 (第六轮): 串行模式与 batch 模式完全对称 — torch.Generator
        + torch.randn (动量), H0/H_new 为 tensor, torch.exp + torch.clamp
        (alpha), torch.rand (接受/拒绝)。原实现每步 4 次 GPU-CPU 同步,
        优化后仅 1 次 (alpha.item())。同时解决可复现性问题 — 串行与
        batch 用相同种子产生相同随机数序列。

        要点1/3 (第十一轮): 串行模式全 tensor 化, 消除每步所有 GPU-CPU 同步。
        原实现 (第六轮后) 每步 3 次同步: (1) bool(isfinite(log_post_new))
        (2) alpha.item() (3) bool(rand < alpha)。tensor 化: alpha 为 0-dim
        tensor (torch.where 处理 is_valid); 接受/拒绝用 torch.where
        (theta_new * mask + theta * (1-mask)); n_accept 为 tensor 累积。
        grad 缓存用 tensor merge (final_grad * mask + grad * (1-mask)),
        last_theta=theta 使下一轮 identity 缓存命中。循环内零同步,
        结束时 2 次标量提取 (samples/log_probs + accept_rate)。
        """
        theta = theta_init.clone()
        if n_leapfrog is None:
            n_leapfrog = self.n_leapfrog

        # 要点6 (第五轮): 预计算 inv_mass_t (采样阶段 inv_mass 固定不变)
        inv_mass_t = torch.tensor(
            inv_mass, dtype=theta.dtype, device=theta.device)

        # 要点2 (第六轮): torch.Generator (基于 chain_seed, 与 batch 模式统一)
        gen = torch.Generator(device=theta.device)
        gen.manual_seed(int(chain_seed))

        # 要点2 (第六轮): 动量缩放 sqrt(mass) = sqrt(1/inv_mass)
        sqrt_mass = torch.sqrt(1.0 / inv_mass_t)

        # 要点1: GPU 预分配样本张量 (消除逐步 CPU 传输)
        samples_gpu = torch.zeros(
            self.n_samples, self.n_params,
            device=theta.device, dtype=theta.dtype)
        log_probs_gpu = torch.zeros(
            self.n_samples,
            device=theta.device, dtype=theta.dtype)
        # 要点1/3 (第十一轮): n_accept 改为 tensor 累积 (零同步)
        n_accept_t = torch.zeros((), dtype=theta.dtype, device=theta.device)
        # 要点1/3 (第十一轮): 零同步常量张量 (避免循环内重复创建)
        zeros_alpha = torch.zeros((), dtype=theta.dtype, device=theta.device)
        neg_inf_t = torch.tensor(float('-inf'), dtype=theta.dtype, device=theta.device)

        # 要点3b: 似然缓存 (拒绝时复用 grad/log_post, 避免重复 grad_fn 调用)
        last_theta = None  # 缓存的 theta (用于命中判断)
        last_grad = None
        last_log_post = None

        for i in range(self.n_samples):
            # 要点4 (第十三轮): 防御性 theta.detach() 切断 autograd 历史 (与 _run_warmup 对称)
            theta = theta.detach()
            # 要点2 (第六轮): torch.randn + generator (无 CPU→GPU 传输)
            p = torch.randn(self.n_params, generator=gen,
                            device=theta.device, dtype=theta.dtype) * sqrt_mass

            # 要点3b: theta 未变时复用缓存的 grad/log_post
            # 改进15 (六): identity 未命中时查 hash cache (opt-in)
            if last_theta is not None and theta is last_theta:
                grad, log_post_0 = last_grad, last_log_post
            else:
                # identity 未命中: 走 hash cache → grad_fn 路径
                grad, log_post_0, _ = self._cached_grad_log_post(theta, None)
                last_theta = theta
                last_grad = grad
                last_log_post = log_post_0
            # 要点1 (第六轮): H0 为 tensor (传入预计算 inv_mass_t)
            H0 = self._hamiltonian(
                log_post_0, p, inv_mass, inv_mass_t=inv_mass_t)

            # 要点3b: 传入预计算的 grad, 避免 _leapfrog 首次 grad_fn 重复调用
            # 要点3 (第四轮): _leapfrog 返回 4-tuple, final_grad 为最终梯度
            # 要点6 (第五轮): 传入预计算的 inv_mass_t, 避免每次 leapfrog 重建
            theta_new, p_new, log_post_new, final_grad = self._leapfrog(
                theta, p, epsilon, inv_mass, n_leapfrog,
                grad_logpost=(grad, log_post_0), inv_mass_t=inv_mass_t)

            # 要点1 (第十一轮): alpha 为 0-dim tensor (无 .item() 同步)
            # torch.where 选择: 有效时 clamp(exp(H0-H_new), max=1), 无效时 0
            is_valid = torch.isfinite(log_post_new)
            H_new = self._hamiltonian(
                log_post_new, p_new, inv_mass, inv_mass_t=inv_mass_t)
            alpha_t = torch.where(
                is_valid,
                torch.clamp(torch.exp(H0 - H_new), max=1.0),
                zeros_alpha
            )  # 0-dim tensor

            # 要点3 (第十一轮): 接受/拒绝用 torch.where (无 bool() 同步)
            # 与 batch 模式完全对称, 与 _run_warmup (要点3 第十一轮) 一致
            rand_val = torch.rand((), generator=gen,
                                  device=theta.device, dtype=theta.dtype)
            accept_t = rand_val < alpha_t  # 0-dim bool tensor
            mask_t = accept_t.to(theta.dtype)  # 0-dim float
            theta = theta_new * mask_t + theta * (1.0 - mask_t)
            n_accept_t += mask_t

            # 要点3 (第四轮) + 要点3 (第十一轮): grad 缓存用 tensor merge
            # 接受: 缓存 final_grad/log_post_new; 拒绝: 缓存 grad/log_post_0
            # torch.where 后 theta 总是新对象, 设 last_theta=theta 使下一轮
            # identity 缓存命中 (theta is last_theta 为 True)
            last_grad = final_grad * mask_t + grad * (1.0 - mask_t)
            last_log_post = log_post_new * mask_t + log_post_0 * (1.0 - mask_t)
            last_theta = theta

            # 要点1: 原地写入 GPU 张量 (无 CPU 传输)
            samples_gpu[i] = theta
            # 要点1/3 (第十一轮): log_probs 用 torch.where (无 Python if 同步)
            # 无效样本写 -inf, 有效样本写 log_post_new (保持原行为)
            # 要点4 (第十三轮): 防御性 .detach()。log_post_new 可能携带梯度
            # (若 grad_fn 错误返回 requires_grad=True 的结果), 累积到 log_probs_gpu
            # 后 .cpu().numpy() 会失败。与循环体 theta.detach() 配合, 确保
            # 整个采样链的 autograd 图不累积。
            log_probs_gpu[i] = torch.where(is_valid, log_post_new, neg_inf_t).detach()

        # 要点1: 一次性传输到 CPU + 提取接受率 (2 次同步, vs 原 3*n_samples 次)
        samples = samples_gpu.cpu().numpy()
        log_probs = log_probs_gpu.cpu().numpy()
        accept_rate = float(n_accept_t.item()) / self.n_samples
        return samples, log_probs, accept_rate

    # ==================== 优先级3: batch 模式 (4 链并行) ====================

    def _hamiltonian_batch(self, log_post, p, inv_mass, inv_mass_t=None):
        """Batch Hamiltonian: B 条链的 H = -log_post + 0.5 * p^T M^{-1} p

        log_post: (B,), p: (B, 15), inv_mass: (B, 15) 或 (15,)
        返回: (B,) 张量 (每条链的 H 值, 不转为 float 以支持批量接受/拒绝)

        要点5 (第六轮): ``inv_mass_t`` 可选参数接收预计算的 torch 张量,
        避免每步调用 ``torch.as_tensor(inv_mass, ...)``。与 ``_leapfrog_batch``
        和 ``_hamiltonian`` (串行版要点1) 对称。``_run_sampling_batch``
        开始时预计算一次传入, n_samples × 2 次重建降为 0。

        要点5 (第十一轮): ``inv_mass`` 在 ``inv_mass_t`` 提供时完全未使用
        (仅用于 ``if inv_mass_t is None`` 的 fallback 分支)。保留为位置参数
        以向后兼容外部调用方。预计算路径下调用方可传 ``None`` 或忽略,
        但因 Python 位置参数语义, 实际调用仍需传值 (建议传 ``inv_mass_t``
        的 numpy 源数组或 None)。未来可重构为关键字参数。
        """
        if inv_mass_t is None:
            inv_mass_t = torch.as_tensor(
                inv_mass, dtype=p.dtype, device=p.device)
            if inv_mass_t.dim() == 1:
                inv_mass_t = inv_mass_t.unsqueeze(0)  # (1, 15) → broadcast
        kinetic = 0.5 * torch.sum(inv_mass_t * p * p, dim=1)  # (B,)
        # 要点2 (第十三轮): 防御性 .detach() (与 _hamiltonian 对称)
        return (-log_post + kinetic).detach()  # (B,)

    def _leapfrog_batch(self, theta, p, epsilon, inv_mass, n_steps,
                        n_steps_per_chain=None, grad_logpost=None,
                        inv_mass_t=None):
        """Batch Leapfrog: B 条链同时积分, 支持各链不同步数 (要点3)

        theta: (B, 15), p: (B, 15)
        epsilon: (B, 1) 或标量 (每条链可独立步长)
        inv_mass: (B, 15) 或 (15,) (每条链可独立质量矩阵)
        n_steps: 标量 (总步数, 通常取 max(n_steps_per_chain))
        n_steps_per_chain: (B,) tensor 或 None。若提供, 每条链 b 执行
            n_steps_per_chain[b] 步, 多余步数跳过 (theta/p 保持不变,
            通过 mask 实现)。避免 n_leapfrog 取 max 导致的 10-20% 冗余计算。
        grad_logpost: 可选 (grad, log_post) 元组, 调用方预计算的结果
            (要点3b: 避免 _leapfrog_batch 首次 grad_fn 重复调用)。
        inv_mass: (B, 15) 或 (15,) — 当 inv_mass_t 提供时可传 None (要点5 第十一轮)。
        inv_mass_t: 可选预计算的 torch 张量 (B, 15) (要点1 第十轮)。
        返回: theta (B, 15), p (B, 15), log_post (B,), grad (B, 15)

        优先级3: 4 条链的梯度计算 batch 化, 一次 grad_fn 调用同时计算
        (B, 15) 梯度, 消除 4× 串行 grad_fn 调用。GPU 场景下 4 链并行
        可将采样时间从 4× 降到 1×。

        要点3: 当各链 warmup 锁定的 n_leapfrog 不同时 (如 [18,20,20,22]),
        原实现统一用 max=22 步, 浪费 10-20% 计算。本实现用 mask 让
        n_leapfrog 较少的链在完成指定步数后跳过更新 (theta/p 保持),
        其他链继续 leapfrog。Python for 循环保持 (因不同链步数不同),
        但每步的 grad_fn 调用仍 batch 化, GPU 利用率不受影响。

        要点3 (第四轮): 返回值增加 final grad (B, 15)。与 _leapfrog 对称,
        最后一步的 grad_fn(theta) 结果一并返回, 调用方 (_run_sampling_batch)
        在接受时用 ``torch.where(accept_t, grad_new, grad)`` 构建 per-chain
        缓存梯度 (要点4), 接受链用 leapfrog 最终 grad, 拒绝链用 pre-leapfrog
        grad (上一轮缓存的 grad)。

        要点1 (第十轮): ``inv_mass_t`` 可选参数接收预计算的 torch 张量,
        与 ``_hamiltonian_batch`` (要点5 第六轮) 和串行版 ``_leapfrog``
        (要点6 第五轮) 完全对称。原实现每步调用
        ``torch.as_tensor(inv_mass, ...)`` 从输入重建 + dim() 检查 +
        unsqueeze + expand_as, batch 模式每步调用一次 _leapfrog_batch,
        n_samples=5000 时累积 5000 次冗余操作。``_run_sampling_batch``
        开始时预计算 inv_mass_t_batch 一次, 同时传给 _hamiltonian_batch
        和 _leapfrog_batch, 两条路径共享同一批预计算。
        """
        # 要点3 (第十三轮): theta/p 初始 .detach() 切断 autograd 历史 (与 _leapfrog 对称)
        theta = theta.clone().detach()
        p = p.clone().detach()
        # 要点1 (第十轮): 优先使用预计算 inv_mass_t, 否则从 inv_mass 重建
        if inv_mass_t is None:
            inv_mass_t = torch.as_tensor(inv_mass, dtype=theta.dtype,
                                         device=theta.device)
            if inv_mass_t.dim() == 1:
                inv_mass_t = inv_mass_t.unsqueeze(0).expand_as(theta)  # (B, 15)

        # epsilon: (B, 1) 用于广播
        if isinstance(epsilon, torch.Tensor):
            eps = epsilon.unsqueeze(1) if epsilon.dim() == 1 else epsilon
        else:
            eps = float(epsilon)

        # 要点3: 各链不同步数 mask
        if n_steps_per_chain is not None:
            nspc = torch.as_tensor(n_steps_per_chain, device=theta.device,
                                   dtype=theta.dtype)
        else:
            nspc = None

        # 要点3b: 使用调用方预计算的 grad, 避免首次 grad_fn 重复调用
        if grad_logpost is not None:
            grad, log_post = grad_logpost
        else:
            grad, log_post = self.grad_fn(theta)  # (B, 15), (B,)
        p = p + 0.5 * eps * grad  # 初始 half kick (所有链执行)

        # 统一循环结构: n_steps 次 (drift + kick), kick 在最后一步为 half
        for step in range(n_steps):
            if nspc is not None:
                # active[b] = (step < nspc[b]): 链 b 是否仍在执行
                # is_last[b] = (step == nspc[b]-1): 链 b 是否为最后一步 (half kick)
                active = (nspc > step).unsqueeze(1).to(theta.dtype)  # (B, 1)
                is_last = (nspc == step + 1).unsqueeze(1).to(theta.dtype)
                # kick 系数: 0.5 if is_last, 1.0 if active and not last, 0 if not active
                # (is_last 蕴含 active, 所以 active - is_last = active & not is_last)
                kick = 0.5 * is_last + 1.0 * (active - is_last)
            else:
                active = 1.0
                kick = 0.5 if step == n_steps - 1 else 1.0

            # drift (仅 active 链更新)
            theta_new = theta + eps * inv_mass_t * p
            theta = theta * (1.0 - active) + theta_new * active
            grad, log_post = self.grad_fn(theta)
            # kick (仅 active 链更新)
            p_new = p + kick * eps * grad
            p = p * (1.0 - active) + p_new * active

        # 要点3 (第四轮): 返回最终 grad (B, 15), 与 theta 对应。
        # 注意: 当各链步数不同时, 步数较少的链提前结束, 后续循环中
        # grad_fn(theta) 对这些链计算的是不更新 theta 的梯度, 仍等于
        # 这些链最后一步的 grad (theta 未变)。因此统一返回最后一步
        # 的 grad 对所有链都正确。
        return theta, p, log_post, grad

    def _run_sampling_batch(self, theta_inits, epsilons, inv_masses,
                            chain_seeds, n_leapfrog):
        """同时运行 B 条链的采样阶段 (batch 化 grad_fn 调用)

        theta_inits: list of B 个 torch tensor (15,)
        epsilons: list of B 个 float (每条链独立步长)
        inv_masses: list of B 个 np.array (15,) (每条链独立质量矩阵)
        chain_seeds: list of B 个 int
        n_leapfrog: list of B 个 int (各链 warmup 锁定的步数)
        返回: samples (B, n_samples, n_params), log_probs (B, n_samples),
              accept_rates (B,)

        优先级3: 4 条链同步运行, 每个采样步的 grad_fn 调用 batch 化。

        要点1: 接受/拒绝改用 torch 随机数 (每条链独立 generator),
               消除 alpha.detach().cpu().numpy() 的 GPU-CPU 同步。
               原实现每步 2 次传输 (alpha→CPU, mask→GPU), 5000 步累积
               10000 次传输。改用 torch.rand 在设备上直接比较, 无传输。
        要点2: 样本在 GPU 上预分配张量累积 (samples_gpu, log_probs_gpu),
               结束后一次性 .cpu().numpy() 传输。原实现每步 2 次传输
               (theta→CPU, log_post→CPU), 5000 步累积 10000 次传输。
               预分配张量内存: 5000×4×15×8B≈2.4MB, 远小于 GPU 内存。
        要点3: 传递各链 n_leapfrog (不再取 max), _leapfrog_batch 用 mask
               让步数少的链跳过冗余 drift/kick, 节省 10-20% 计算。

        要点4 (第四轮): per-chain 似然缓存。原实现每步无条件调用
        ``self.grad_fn(theta)`` 计算 H0, 但拒绝链的 theta 未变, 缓存仍有效。
        batch 模式的复杂之处在于接受/拒绝是逐链的——同一步中部分链接受、
        部分链拒绝。引入 (B,) 布尔缓存命中向量 ``cache_valid``:
          - 全 True: 所有链 theta 未变 (上一步全拒绝或刚接受后缓存已更新),
            完全跳过 grad_fn, 用 cached_grad/cached_log_post 计算 H0
          - 全 False 或混合: 调用 grad_fn (batch, 所有链一起算), 用 fresh 结果
        每步结束后用 ``torch.where(accept_t, final_grad, grad)`` 更新缓存:
        接受链用 leapfrog 最终 grad (theta_new 的梯度), 拒绝链用 H0 时的
        grad (theta 未变, 缓存仍有效)。与 _run_single_chain 的要点3 对称,
        接受率 80% 场景下额外减少 80% 的 H0 grad_fn 调用。
        """
        B = len(theta_inits)
        theta = torch.stack(theta_inits)  # (B, 15)
        eps = torch.tensor(epsilons, dtype=theta.dtype,
                           device=theta.device)  # (B,)
        inv_mass = torch.tensor(np.array(inv_masses), dtype=theta.dtype,
                                device=theta.device)  # (B, 15)
        # 要点3: 各链 n_leapfrog (不再取 max)
        nspc = torch.tensor(n_leapfrog, device=theta.device,
                            dtype=theta.dtype)  # (B,)
        n_steps = int(max(n_leapfrog))

        # 要点2: GPU 上预分配样本张量 (结束时一次性传输)
        samples_gpu = torch.zeros(self.n_samples, B, self.n_params,
                                  device=theta.device, dtype=theta.dtype)
        log_probs_gpu = torch.zeros(self.n_samples, B,
                                    device=theta.device, dtype=theta.dtype)
        n_accept = torch.zeros(B, device=theta.device, dtype=theta.dtype)

        # 要点1: 每条链独立 torch.Generator (基于 chain_seed, 可复现)
        gens = [torch.Generator(device=theta.device) for _ in range(B)]
        for g, s in zip(gens, chain_seeds):
            g.manual_seed(int(s))

        # 动量缩放: sqrt(1/inv_mass) = sqrt(mass)
        sqrt_mass = torch.sqrt(1.0 / inv_mass)  # (B, 15)

        # 要点5 (第六轮): 预计算 inv_mass_t (已是 (B, 15), 无需 unsqueeze)
        # 传给 _hamiltonian_batch 避免 n_samples × 2 次 as_tensor 重建
        inv_mass_t_batch = inv_mass  # 已是正确形状 (B, 15)

        # 要点4: per-chain 似然缓存 (首次必须计算, cache_valid=False)
        cached_grad = None  # (B, 15), 仅在 cache_valid=True 时有效
        cached_log_post = None  # (B,)
        # 要点2 (第十一轮): cache_valid 用 Python bool 替代 torch.bool 张量。
        # 分析发现: 每步结束后 cache_valid 总被设为全 True (接受链缓存 final_grad,
        # 拒绝链缓存 H0 时的 grad, 两种情况缓存均有效)。因此除首步外 cache_valid
        # 恒为 True, 用 Python bool 检查 (零同步) 替代 bool(cache_valid.all())
        # (每步 1 次同步)。首步 cache_valid=False 强制调用 grad_fn 初始化缓存。
        # 正确性: 采样阶段 inv_mass 固定, theta 经 torch.where 更新后缓存对应
        # 的 grad/log_post 已同步更新 (torch.where 选择 final_grad 或 grad),
        # 故缓存恒有效。mixed 精度模式下采样阶段以 float32 重新初始化缓存。
        cache_valid = False  # Python bool

        for i in range(self.n_samples):
            # 要点5 (第十三轮): 防御性 theta.detach() 切断 autograd 历史 (与串行对称)。
            # batch 模式 B 条链的 theta 若携带历史梯度, 内存占用是串行的 B 倍。
            # detach() 在 requires_grad=False 时返回同一对象, 不影响后续操作。
            theta = theta.detach()
            # 要点1: 动量用 torch.randn (各链独立 generator, 无 GPU-CPU 传输)
            p_list = [
                torch.randn(self.n_params, generator=g,
                            device=theta.device, dtype=theta.dtype)
                * sqrt_mass[b]
                for b, g in enumerate(gens)
            ]
            p = torch.stack(p_list)  # (B, 15)

            # 要点4: per-chain 缓存判断
            # 要点2 (第十一轮): Python bool 检查 (零同步), 替代 bool(cache_valid.all())
            # cache_valid 仅首步为 False, 之后恒为 True (缓存总有效)
            if cache_valid:
                grad = cached_grad
                log_post_0 = cached_log_post
            else:
                grad, log_post_0 = self.grad_fn(theta)  # (B, 15), (B,)
            # 要点5 (第六轮): 传入预计算 inv_mass_t_batch
            H0 = self._hamiltonian_batch(
                log_post_0, p, inv_mass, inv_mass_t=inv_mass_t_batch)  # (B,)

            # batch leapfrog (要点3: 传递各链 n_leapfrog;
            #   要点3b: 传入预计算 grad 避免 _leapfrog_batch 首次 grad_fn 重复调用;
            #   要点3: 返回 4-tuple, final_grad 为 leapfrog 最终梯度;
            #   要点1 第十轮: 传入预计算 inv_mass_t_batch, 与 _hamiltonian_batch
            #   共享同一批预计算, 消除 _leapfrog_batch 内部 as_tensor 重建)
            theta_new, p_new, log_post_new, final_grad = self._leapfrog_batch(
                theta, p, eps, inv_mass, n_steps, n_steps_per_chain=nspc,
                grad_logpost=(grad, log_post_0),
                inv_mass_t=inv_mass_t_batch)

            # batch 接受/拒绝 (要点1: 全在设备上, 无 CPU 传输)
            # 要点5 (第六轮): 传入预计算 inv_mass_t_batch
            H_new = self._hamiltonian_batch(
                log_post_new, p_new, inv_mass, inv_mass_t=inv_mass_t_batch)
            # 要点2 (第十轮): torch.isfinite 一次性检查, 与串行模式统一
            # (原 ~(isnan | isinf) 两次计算, isfinite 一次计算替代)
            valid = torch.isfinite(log_post_new)
            # alpha = min(1, exp(H0 - H_new)), 无效链 alpha=0
            alpha = torch.where(
                valid,
                torch.clamp(torch.exp(H0 - H_new), max=1.0),
                torch.zeros(B, dtype=theta.dtype, device=theta.device)
            )  # (B,)

            # 要点1: torch 随机数接受/拒绝 (各链独立 generator, 无 CPU 传输)
            rand_vals = torch.stack([
                torch.rand((), generator=g,
                           device=theta.device, dtype=theta.dtype)
                for g in gens
            ])  # (B,)
            accept_t = rand_vals < alpha  # (B,) bool, 全在设备上
            mask_t = accept_t.unsqueeze(1).to(theta.dtype)  # (B, 1)
            theta = theta_new * mask_t + theta * (1.0 - mask_t)
            n_accept += accept_t.to(theta.dtype)

            # 要点4: 更新 per-chain 缓存
            # 接受链: theta=theta_new, 缓存 leapfrog 的 final_grad/log_post_new
            # 拒绝链: theta 不变, 缓存 H0 时的 grad/log_post_0 (仍有效)
            # 两种情况都有有效缓存, cache_valid 置 True
            # 要点2 (第十一轮): Python bool 替代 torch.ones (零同步)
            cached_grad = torch.where(
                accept_t.unsqueeze(1), final_grad, grad)
            cached_log_post = torch.where(accept_t, log_post_new, log_post_0)
            cache_valid = True

            # 要点2: 样本记录到 GPU 张量 (无逐步 CPU 传输)
            samples_gpu[i] = theta
            # 要点5 (第十三轮): 防御性 .detach() (与 _run_single_chain 对称)
            log_probs_gpu[i] = torch.where(
                valid, log_post_new,
                torch.full_like(log_post_new, -float('inf'))).detach()

        # 要点2: 结束后一次性传输 (3 次传输 vs 原 10000+ 次)
        samples = samples_gpu.cpu().numpy().transpose(1, 0, 2)  # (B, n_samples, n_params)
        log_probs = log_probs_gpu.cpu().numpy().transpose(1, 0)  # (B, n_samples)
        accept_rates = (n_accept / self.n_samples).cpu().numpy()  # (B,)

        return samples, log_probs, accept_rates

    def sample(self, init_params_fn=None, batch_chains=False):
        """运行多链 HMC 采样

        要点B: 每条链使用 warmup 阶段锁定的 n_leapfrog 进行采样。
        优先级3: ``batch_chains=True`` 时, 采样阶段 4 链 batch 化
        (一次 grad_fn 调用同时计算 4 链梯度, GPU 场景提速 4×)。

        要点1 (第十三轮): ``batch_chains=True`` 时, warmup 阶段也 batch 化
        (``_run_warmup_batch``), 消除原串行 warmup 的 n_chains 倍开销。
        warmup 占 HMC 总时间 30-50%, 4 链 batch 化后总时间可降 20-40%。

        要点6 (第十二轮): ``batch_chains='auto'`` 根据 device.type 自适应
        选择。CUDA 设备用 batch 模式 (利用 GPU 并行), CPU 设备用串行模式
        (避免 batch 的 tensor reshape/mask overhead)。用户可显式传
        True/False 覆盖 auto 行为。

        Args:
            init_params_fn: 初始化函数 (chain_idx → np.array 或 torch.Tensor)
                要点1 (第十二轮): 支持返回 torch 张量, 内部用
                as_tensor().detach().clone() 安全转换, 不触发 UserWarning。
            batch_chains: bool 或 'auto' — 是否 batch 化 warmup + 采样阶段。
                False (默认, 向后兼容): 串行 warmup + 串行采样。
                True: batch warmup + batch 采样 (GPU 场景提速)。
                'auto': 根据 device.type 自适应 (CUDA→batch, CPU→串行)。
        """
        if init_params_fn is None:
            init_params_fn = lambda idx: np.zeros(self.n_params)

        # 要点6 (第十二轮): batch_chains='auto' 解析提前到 warmup 之前
        # (要点1 第十三轮: warmup 也需根据 batch_chains 决定是否 batch 化)
        if batch_chains == 'auto':
            batch_chains = (isinstance(self.device, torch.device)
                            and self.device.type == 'cuda')

        inv_mass_init = np.ones(self.n_params)
        theta_inits = []
        chain_seeds = []
        all_n_leapfrog = []  # 要点B: 每条链的锁定 n_leapfrog

        # 要点3a: warmup 用 _warmup_dtype (mixed 模式下为 float64)
        # 要点1 (第十二轮): 用 torch.as_tensor().detach().clone() 替代 torch.tensor()。
        for chain_idx in range(self.n_chains):
            chain_seed = self.rng.randint(0, 2**30) + chain_idx
            theta_init_np = init_params_fn(chain_idx)
            theta_init = torch.as_tensor(
                theta_init_np, dtype=self._warmup_dtype,
                device=self.device).detach().clone()
            theta_inits.append(theta_init)
            chain_seeds.append(chain_seed)

        # HMC 低接受率自适应（监控缺口修复）：采样结束后检查平均接受率，
        # 显著低于 target_accept（< 0.5）说明步长/质量矩阵自适应不足，
        # 直接用于推断会得到低 ESS/高自相关后验。策略：降 initial
        # epsilon（×0.1）+ 延长 warmup（×2）重跑一次，仍低则保留较好
        # 结果并标记 low_efficiency_sampling（下游不得作为有效采样引用）。
        def _warmup_and_sample(epsilon_scale, theta_inits_orig):
            theta_inits_local = [t.detach().clone()
                                 for t in theta_inits_orig]
            epsilons_local = []
            inv_masses_local = []
            n_leapfrogs_local = []
            if batch_chains:
                eps_b, im_b, _, nlf_b = self._run_warmup_batch(
                    theta_inits_local, [inv_mass_init] * self.n_chains,
                    chain_seeds, epsilon_scale=epsilon_scale)
                epsilons_local = list(eps_b)
                inv_masses_local = [im_b[b] for b in range(self.n_chains)]
                n_leapfrogs_local = [int(nlf_b[b]) for b in range(self.n_chains)]
            else:
                for chain_idx in range(self.n_chains):
                    epsilon, inv_mass, warmup, locked_n_leapfrog = \
                        self._run_warmup(
                            theta_inits_local[chain_idx], inv_mass_init,
                            chain_seed=chain_seeds[chain_idx],
                            epsilon_scale=epsilon_scale)
                    epsilons_local.append(epsilon)
                    inv_masses_local.append(inv_mass)
                    n_leapfrogs_local.append(int(locked_n_leapfrog))

            # 要点3a: mixed 模式下, warmup 后转 float32 加速采样
            if self.precision == 'mixed':
                theta_inits_local = [
                    t.detach().to(self._sample_dtype) for t in theta_inits_local]
                inv_masses_local = [
                    im.astype(np.float32) for im in inv_masses_local]

            if batch_chains:
                samples_b, log_probs_b, acc_rates_b = \
                    self._run_sampling_batch(
                        theta_inits_local, epsilons_local, inv_masses_local,
                        chain_seeds, n_leapfrogs_local)
                chains_local = [samples_b[b] for b in range(self.n_chains)]
                log_probs_local = [log_probs_b[b] for b in range(self.n_chains)]
                acc_local = list(acc_rates_b)
            else:
                chains_local = []
                log_probs_local = []
                acc_local = []
                for chain_idx in range(self.n_chains):
                    samples, log_probs, acc_rate = self._run_single_chain(
                        theta_inits_local[chain_idx],
                        epsilons_local[chain_idx],
                        inv_masses_local[chain_idx],
                        chain_seeds[chain_idx],
                        n_leapfrog=n_leapfrogs_local[chain_idx])
                    chains_local.append(samples)
                    log_probs_local.append(log_probs)
                    acc_local.append(acc_rate)
            return {
                'chains': chains_local, 'log_probs': log_probs_local,
                'accept_rates': acc_local, 'epsilons': epsilons_local,
                'inv_masses': inv_masses_local,
                'n_leapfrogs': n_leapfrogs_local,
            }

        # 第一次采样（标准自适应）
        run_result = _warmup_and_sample(1.0, theta_inits)
        mean_acc = float(np.mean(run_result['accept_rates']))

        # 低接受率判定与一次重试（降 initial epsilon ×0.1 + warmup ×2）
        LOW_ACCEPT_THRESHOLD = 0.5
        retry_attempted = False
        retry_improved = False
        if mean_acc < LOW_ACCEPT_THRESHOLD:
            retry_attempted = True
            orig_n_warmup = self.n_warmup
            try:
                self.n_warmup = int(2 * orig_n_warmup)
                retry_result = _warmup_and_sample(0.1, theta_inits)
                retry_mean_acc = float(np.mean(retry_result['accept_rates']))
                if retry_mean_acc > mean_acc:
                    run_result = retry_result
                    mean_acc = retry_mean_acc
                    retry_improved = True
            finally:
                self.n_warmup = orig_n_warmup
        # 低效采样标记：重试后接受率仍低于阈值（重试成功时 mean_acc
        # 已更新，达标则不标记）
        low_efficiency_sampling = mean_acc < LOW_ACCEPT_THRESHOLD

        all_chains = run_result['chains']
        all_log_probs = run_result['log_probs']
        all_accept_rates = run_result['accept_rates']
        epsilons = run_result['epsilons']
        inv_masses = run_result['inv_masses']
        all_n_leapfrog = run_result['n_leapfrogs']

        all_step_sizes = epsilons
        self.chains = all_chains
        self.log_probs = all_log_probs
        self.acceptance_rates = all_accept_rates
        self.step_sizes = all_step_sizes
        self.final_n_leapfrog = all_n_leapfrog
        self.mass_diag = 1.0 / inv_masses[-1]  # 最后一个链的 inv_mass (向后兼容)

        # 收敛诊断：使用 diagnostics.py 中的独立函数
        r_hat = compute_r_hat(all_chains, 0)
        # L 级修复：split R-hat（Vehtari 2021 推荐）——链内不平稳
        # （前半/后半分布漂移）经典 R-hat 检测不到，split 版本可暴露
        split_r_hat = compute_split_r_hat(all_chains, 0)
        merged = np.concatenate(all_chains, axis=0)

        ess = {}
        for p_idx in range(self.n_params):
            ess[f'param_{p_idx}'] = compute_ess_single(merged[:, p_idx])

        self.diagnostics = {
            'n_chains': self.n_chains,
            'n_warmup': self.n_warmup,
            'n_samples': self.n_samples,
            'n_leapfrog': self.n_leapfrog,
            'final_n_leapfrog': all_n_leapfrog,  # 要点B
            'mean_final_n_leapfrog': float(np.mean(all_n_leapfrog)),  # 要点B
            'adaptive_leapfrog': self.adaptive_leapfrog,  # 要点B
            'target_accept': self.target_accept,
            'acceptance_rates': all_accept_rates,
            'mean_acceptance_rate': float(np.mean(all_accept_rates)),
            # 低效采样监控（低接受率自适应重试结果）：
            # - low_accept_retry_attempted: 是否触发过重试
            # - low_accept_retry_improved: 重试是否改善接受率
            # - low_efficiency_sampling: 最终平均接受率仍 < 0.5，
            #   该 run 不得作为有效采样证据引用
            'low_accept_retry_attempted': retry_attempted,
            'low_accept_retry_improved': retry_improved,
            'low_efficiency_sampling': low_efficiency_sampling,
            'step_sizes': all_step_sizes,
            'mean_step_size': float(np.mean(all_step_sizes)),
            'mass_matrix_diag': self.mass_diag.tolist(),
            'r_hat': r_hat,
            # split R-hat（每链拆前后两半 → 2m 条子链）：单链内分布漂移
            # 会使 split_r_hat > 1 而经典 r_hat 仍接近 1，引用诊断时
            # 两者都应报告（Vehtari 2021）
            'split_r_hat': split_r_hat,
            'ess': ess,
            'converged': _safe_rhat_converged(r_hat),
            'algorithm': 'HMC + Dual Averaging (Nesterov 2009) + Adaptive Leapfrog',
            'reference': 'Hoffman & Gelman (2014) JMLR 15(1):1593-1623',
        }

        # 改进15 (六): hash cache 统计 (仅 use_hash_cache=True 时有意义)
        if self._hash_cache is not None:
            self.diagnostics['hash_cache_stats'] = self._hash_cache.stats()

        return {
            'chains': all_chains,
            'log_probs': all_log_probs,
            'acceptance_rates': all_accept_rates,
            'diagnostics': self.diagnostics,
        }

    def get_posterior_mean(self):
        """返回后验均值 (约束空间)"""
        from .autodiff import constrain_params, PARAM_NAMES
        merged = np.concatenate(self.chains, axis=0)
        y_tensor = torch.tensor(merged, dtype=torch.float64)
        params_dict, _ = constrain_params(y_tensor)
        return {name: float(params_dict[name].mean()) for name in PARAM_NAMES}

    def get_constrained_samples(self):
        """返回约束空间中的后验样本"""
        from .autodiff import constrain_params, PARAM_NAMES
        merged = np.concatenate(self.chains, axis=0)
        y_tensor = torch.tensor(merged, dtype=torch.float64)
        params_dict, _ = constrain_params(y_tensor)
        return {name: params_dict[name].detach().cpu().numpy()
                for name in PARAM_NAMES}

    def identifiability_analysis(self):
        """参数可识别性分析: 后验相关矩阵 + 方差膨胀因子 (VIF)

        文献: Dankwa et al. (2025) PLoS Comput Biol 21(11):e1013647
        """
        from .autodiff import PARAM_NAMES
        merged = np.concatenate(self.chains, axis=0)
        corr = np.corrcoef(merged.T)

        high_pairs = []
        for i in range(self.n_params):
            for j in range(i + 1, self.n_params):
                r = corr[i, j]
                if abs(r) > 0.7:
                    high_pairs.append({
                        'param1': PARAM_NAMES[i],
                        'param2': PARAM_NAMES[j],
                        'correlation': float(r),
                    })

        vif = {}
        for i in range(self.n_params):
            y_i = merged[:, i]
            X = np.delete(merged, i, axis=1)
            try:
                X_aug = np.column_stack([np.ones(len(X)), X])
                coeffs, _, _, _ = np.linalg.lstsq(X_aug, y_i, rcond=None)
                y_pred = X_aug @ coeffs
                ss_res = np.sum((y_i - y_pred) ** 2)
                ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)
                r_squared = 1 - ss_res / max(ss_tot, 1e-10)
                r_squared = max(0, min(r_squared, 0.9999))
                vif[PARAM_NAMES[i]] = float(1.0 / (1.0 - r_squared))
            except Exception:
                vif[PARAM_NAMES[i]] = float('inf')

        identifiable = [name for name, v in vif.items() if v < 10.0]

        return {
            'correlation_matrix': corr.tolist(),
            'param_names': PARAM_NAMES,
            'high_correlation_pairs': high_pairs,
            'vif': vif,
            'identifiable': identifiable,
            'n_identifiable': len(identifiable),
            'reference': 'Dankwa et al. (2025) PLoS Comput Biol 21(11):e1013647',
        }
