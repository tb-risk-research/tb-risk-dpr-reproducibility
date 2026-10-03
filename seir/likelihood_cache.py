#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
似然缓存 (Likelihood Cache) — MD5 哈希键 + LRU 淘汰

在 MCMC/HMC 中, 当提议被拒绝时, 当前 theta 不变, 下一轮需要重新计算
相同的 log_posterior 与梯度。缓存可避免这些重复的 ODE 积分。

设计:
  - 键: MD5(theta.tobytes()) — 值无关, 适合 torch.Tensor / numpy.ndarray
  - 值: (log_posterior, gradient) 元组
  - LRU 淘汰: 超过 maxsize 时淘汰最久未访问的条目
  - 线程安全: threading.Lock 保护 (HMC batch 模式可能多线程访问)
  - 统计: hits/misses 计数, 用于诊断缓存命中率

与现有 identity cache (``theta is last_theta``) 的关系:
  - identity cache: 单条目, 速度极快 (无哈希计算), 但仅在 theta 是同一
    Python 对象时命中 (拒绝时 theta 不重新赋值)
  - hash cache: 多条目 LRU, 哈希开销小 (~µs), 但能在 theta 被克隆/
    batch 拆分时命中 (identity 失效场景)
  - 两者可共存: 先查 identity cache (零开销), 未命中再查 hash cache

文献:
  Andrieu & Doucet (2010) JRSS-B 72(3):269-342 — detailed balance 保护
  Hoffman & Gelman (2014) JMLR 15(1):1593-1623 — HMC/NUTS
"""

import hashlib
import threading
import warnings
from collections import OrderedDict


__all__ = ['LikelihoodCache', 'hash_theta', 'is_gpu_tensor']


def is_gpu_tensor(theta):
    """检测 theta 是否为 GPU 上的 torch.Tensor

    要点2 (第五轮): GPU 环境下 hash_theta 的 ``.cpu().numpy()`` 会触发
    GPU-CPU 同步, 抵消 batch 模式的 GPU 并行优化。本函数用于
    LikelihoodCache 自动禁用机制 — 检测到 GPU 张量时跳过缓存查询。

    Args:
        theta: 任意对象
    Returns:
        bool: True 若 theta 是位于 CUDA 设备的 torch.Tensor
    """
    try:
        import torch
        if isinstance(theta, torch.Tensor):
            return theta.is_cuda or 'cuda' in str(theta.device).lower()
    except ImportError:
        pass
    return False


def hash_theta(theta):
    """计算 theta 的 MD5 哈希 (缓存键)

    支持 numpy.ndarray 和 torch.Tensor。对于 torch.Tensor, 先 detach
    转 CPU 再转 numpy (避免 GPU-CPU 同步开销仅在必要时发生)。

    Args:
        theta: numpy.ndarray (n,) 或 torch.Tensor (n,) / (B, n)
    Returns:
        str: 32 字符 MD5 十六进制字符串

    要点5 (第五轮): 跳过 ``ascontiguousarray`` 当数组已 C 连续。
    原实现无条件调用 ``np.ascontiguousarray(arr, dtype=np.float64)``,
    若 arr 已是 C 连续的 float64 数组 (常见情况: numpy 用户传入的
    theta 通常是连续的), ascontiguousarray 会触发一次不必要的拷贝。
    检查 ``arr.flags['C_CONTIGUOUS']`` 与 dtype, 已连续且 dtype 匹配
    时直接使用原数组, 节省 15 维 float64 约 120 字节的拷贝开销。
    在 _leapfrog 内部每步查询时 (n_leapfrog=20 即 20 次/步), 累积可观。
    """
    try:
        import torch
        if isinstance(theta, torch.Tensor):
            arr = theta.detach().cpu().numpy()
        else:
            arr = theta
    except ImportError:
        arr = theta

    # 要点5: 仅在非 C 连续或 dtype 不匹配时才拷贝
    import numpy as np
    if arr.dtype != np.float64 or not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
    return hashlib.md5(arr.tobytes()).hexdigest()


class LikelihoodCache:
    """似然缓存 (MD5 哈希键 + LRU 淘汰)

    用法:
        ::

            cache = LikelihoodCache(maxsize=1024)
            # 查询
            cached = cache.get(theta)
            if cached is not None:
                log_post, grad = cached
            else:
                log_post, grad = grad_fn(theta)
                cache.put(theta, log_post, grad)
            # 统计
            print(cache.stats())

    线程安全: 所有读写操作受 ``threading.Lock`` 保护, 适合 HMC batch
    模式下多线程并发访问 (虽然当前 HMC 是单线程, 但为未来扩展预留)。

    Args:
        maxsize: 最大条目数 (默认 1024, 超过后 LRU 淘汰)
        auto_disable_on_gpu: 要点2 (第五轮)。True (默认) 时, 若检测到
            theta 是 CUDA 张量, 自动禁用缓存 (get 返回 None, put 无操作),
            避免 ``hash_theta`` 的 ``.cpu().numpy()`` 触发 GPU-CPU 同步
            抵消 batch 模式的 GPU 并行优化。首次检测到 GPU 张量时发出
            UserWarning。HMCSampler 在 device='cuda' 时应直接 force
            use_hash_cache=False, 此参数作为防御性备份。

    属性:
        disabled: 缓存是否已被自动禁用 (GPU 检测触发)
    """

    def __init__(self, maxsize=1024, auto_disable_on_gpu=True):
        self.maxsize = int(maxsize)
        if self.maxsize < 1:
            raise ValueError(f"maxsize 必须 >= 1, got {maxsize}")
        self.auto_disable_on_gpu = bool(auto_disable_on_gpu)
        self.disabled = False  # GPU 检测触发后置 True
        self._gpu_warned = False  # 仅警告一次
        self._store = OrderedDict()  # key -> (log_post, grad)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _check_gpu_disable(self, theta):
        """检测 GPU 张量并禁用缓存 (内部, 调用方持锁)

        返回 True 表示已禁用 (调用方应短路返回)。
        """
        if not self.auto_disable_on_gpu or self.disabled:
            return self.disabled
        if is_gpu_tensor(theta):
            self.disabled = True
            if not self._gpu_warned:
                self._gpu_warned = True
                warnings.warn(
                    "LikelihoodCache 检测到 GPU 张量, 自动禁用 hash cache "
                    "(避免 .cpu().numpy() 触发 GPU-CPU 同步)。建议在 "
                    "HMCSampler 中 device='cuda' 时设置 use_hash_cache=False。",
                    UserWarning, stacklevel=3)
        return self.disabled

    def get(self, theta):
        """查询缓存

        Args:
            theta: 参数向量 (numpy 或 torch)
        Returns:
            (log_post, grad) 元组, 未命中返回 None
        """
        with self._lock:
            # 要点2: GPU 张量自动禁用 (短路, 不调用 hash_theta 避免同步)
            if self._check_gpu_disable(theta):
                self._misses += 1
                return None
            key = hash_theta(theta)
            if key in self._store:
                # LRU: 移到末尾 (最近访问)
                value = self._store.pop(key)
                self._store[key] = value
                self._hits += 1
                return value
            self._misses += 1
            return None

    def put(self, theta, log_post, grad=None):
        """存入缓存

        Args:
            theta: 参数向量
            log_post: log_posterior 值 (标量)
            grad: 梯度向量 (与 theta 同形状, 可选 — MH 不需要梯度)
        """
        with self._lock:
            # 要点2: GPU 张量自动禁用 (短路)
            if self._check_gpu_disable(theta):
                return
            key = hash_theta(theta)
            # 若已存在, 先删除 (LRU 更新)
            if key in self._store:
                self._store.pop(key)
            # 淘汰最旧条目
            while len(self._store) >= self.maxsize:
                self._store.popitem(last=False)
            self._store[key] = (log_post, grad)

    def __contains__(self, theta):
        """支持 ``theta in cache`` 语法"""
        with self._lock:
            if self._check_gpu_disable(theta):
                return False
            key = hash_theta(theta)
            return key in self._store

    def __len__(self):
        with self._lock:
            return len(self._store)

    def clear(self):
        """清空缓存"""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
            # 不重置 disabled — 一旦检测到 GPU, 保持禁用状态

    def stats(self):
        """返回缓存统计

        Returns:
            dict: {
                'size': 当前条目数,
                'maxsize': 最大条目数,
                'hits': 命中次数,
                'misses': 未命中次数,
                'hit_rate': 命中率 (0-1),
                'disabled': 是否被自动禁用 (GPU 检测触发),
            }
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                'size': len(self._store),
                'maxsize': self.maxsize,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': self._hits / total if total > 0 else 0.0,
                'disabled': self.disabled,
            }
