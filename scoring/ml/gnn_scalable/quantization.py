#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型量化（quantization）— FP32 → FP16 / INT8，降低推理延迟与显存占用。

实时临床决策要求推理从"秒级"降到"毫秒级"。量化是主要手段：
- FP16：将模型权重/激活转为半精度，显存减半，GPU 上吞吐提升；
- INT8 动态量化：将 Linear 权重转为 int8，推理时动态反量化，适合 CPU 推理。

本模块提供量化副本构造、量化推理与延迟测量（含统计百分位），
量化失败时自动回退到原始 FP32 模型，保证功能不中断。

文献：
- Micikevicius et al. Mixed Precision Training. ICLR 2018. — FP16 训练/推理。
- Jacob et al. Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference. CVPR 2018. — INT8 量化。
"""
import copy
import logging
import time
from typing import Dict, Optional

from ._common import (
    LOGGER, PYTORCH_AVAILABLE, PYG_AVAILABLE, DYNAMIC_QUANT_AVAILABLE,
    torch, quantize_dynamic,
)
from .sampling import sample_k_hop_subgraph
from .inference import predict_subgraph

__all__ = [
    "quantize_model_fp16", "quantize_model_int8",
    "predict_quantized", "measure_inference_latency",
    "DYNAMIC_QUANT_AVAILABLE",
]


def quantize_model_fp16(model):
    """返回 FP16 副本（权重半精度）。失败时返回原始模型。

    注意：CPU 上半精度可能更慢（无 fp16 内核），主要用于 GPU 推理与显存削减。
    """
    if not PYTORCH_AVAILABLE:
        return model
    import torch.nn as nn
    try:
        model_fp16 = copy.deepcopy(model).half()
        return model_fp16
    except Exception as e:
        LOGGER.warning("FP16 量化失败，回退 FP32: %s", e)
        return model


def quantize_model_int8(model, module_types=None):
    """返回 INT8 动态量化副本（量化 Linear 层）。失败时回退 FP16/FP32。

    参数：
        model: 待量化模型
        module_types: 待量化模块类型集合；None 默认 {nn.Linear}

    返回：
        (quantized_model, backend) — backend ∈ {'int8','fp16','fp32'}
    """
    if not PYTORCH_AVAILABLE:
        return model, 'fp32'
    import torch.nn as nn
    if module_types is None:
        module_types = {nn.Linear}
    try:
        if quantize_dynamic is not None:
            q = quantize_dynamic(
                copy.deepcopy(model), qconfig_spec=module_types, dtype=torch.qint8)
            return q, 'int8'
    except Exception as e:
        LOGGER.warning("INT8 量化失败，回退: %s", e)
    # 回退 FP16
    return quantize_model_fp16(model), 'fp16'


def predict_quantized(model, sub_data, target_mapping=None, quant_type='int8',
                      device=None, forward_fn=None):
    """对子图子做量化推理，返回风险概率与原精度推理的对比。

    参数：
        model: 原始 FP32 模型
        sub_data: k-hop 子图
        target_mapping: 目标节点在子图内索引
        quant_type: 'int8' 或 'fp16'
        forward_fn: 自定义前向

    返回：
        dict: {backend, risk_probability, risk_class, subgraph_size}
        依赖缺失返回 None。
    """
    if not (PYTORCH_AVAILABLE and PYG_AVAILABLE) or sub_data is None:
        return None
    if quant_type == 'int8':
        qmodel, backend = quantize_model_int8(model)
    else:
        qmodel = quantize_model_fp16(model)
        backend = 'fp16'
        # FP16 模型需 FP16 输入，避免 dtype 不匹配。
        # 注意：PyG 的 Data.to(dtype) 不接受纯 dtype 关键字/位置参数（会把 torch.float16 当 device），
        # 故仅手动转换浮点特征张量 x/edge_attr，edge_index 保持整型。
        try:
            sub_data = copy.deepcopy(sub_data)
            sub_data.x = sub_data.x.half()
            if sub_data.edge_attr is not None:
                sub_data.edge_attr = sub_data.edge_attr.half()
        except Exception as e:
            LOGGER.debug("子图转 FP16 失败，保持原精度: %s", e)
    return predict_subgraph(qmodel, sub_data, target_mapping, device=device,
                            forward_fn=forward_fn)


def measure_inference_latency(model, large_data, target_idx=0, num_hops=2,
                              n_runs=10, device=None, quant_type=None,
                              warmup=2):
    """测量目标节点 k-hop 子图推理延迟（含统计百分位）。

    参数：
        model: GNN 模块
        large_data: 大图
        target_idx: 目标节点
        num_hops: 邻域跳数
        n_runs: 计时次数
        quant_type: None（FP32）/ 'fp16' / 'int8'
        warmup: 预热次数（不计入统计）

    返回：
        dict: {backend, mean_ms, std_ms, p50_ms, p95_ms, per_run_ms}
        依赖缺失返回 None。
    """
    if not (PYTORCH_AVAILABLE and PYG_AVAILABLE):
        return None
    sub, mapping = sample_k_hop_subgraph(large_data, target_idx, num_hops)
    if sub is None:
        return None

    if quant_type == 'int8':
        run_model, backend = quantize_model_int8(model)
    elif quant_type == 'fp16':
        run_model, backend = quantize_model_fp16(model), 'fp16'
    else:
        run_model, backend = model, 'fp32'

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_model.eval()
    run_model = run_model.to(device)
    if backend == 'fp16':
        try:
            sub.x = sub.x.half()
            if sub.edge_attr is not None:
                sub.edge_attr = sub.edge_attr.half()
        except Exception:
            pass
    sub = sub.to(device)

    def _run():
        edge_attr = sub.edge_attr if (hasattr(sub, 'edge_attr') and sub.edge_attr is not None) else None
        with torch.no_grad():
            if edge_attr is not None:
                run_model(sub.x, sub.edge_index, edge_attr=edge_attr)
            else:
                run_model(sub.x, sub.edge_index)

    for _ in range(warmup):
        _run()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _run()
        times.append((time.perf_counter() - t0) * 1e3)  # ms

    times = sorted(times)
    import statistics
    mean_ms = float(statistics.mean(times))
    std_ms = float(statistics.pstdev(times)) if len(times) > 1 else 0.0

    def _perc(p):
        if not times:
            return 0.0
        idx = int(round((len(times) - 1) * p))
        return times[idx]

    return {
        "backend": backend,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "p50_ms": _perc(0.5),
        "p95_ms": _perc(0.95),
        "per_run_ms": times,
        "subgraph_size": int(sub.num_nodes),
        "n_edges": int(sub.edge_index.shape[1]),
    }