#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大规模图网络处理子包 — mini-batch 训练 / 子图推理 / 量化 / 蒸馏 / 增量推理。

当接触网络达到十万级节点时，整图全量前向在显存/内存与延迟上不可行。本包为
GNN 模块补齐五类规模化能力（纯 Python + torch，PyG 可选 + 渐进降级）：

1. 采样（sampling）      — k-hop 邻域子图抽取（含边特征保留）+ 节点级 mini-batch
2. mini-batch 训练        — NeighborLoader / k-hop 采样训练，整图不加载进显存
3. 子图推理（inference）  — 目标患者仅算其 k-hop 邻域子图，而非全图
4. 量化（quantization）   — FP16 / INT8 动态量化 + 推理延迟测量
5. 蒸馏（distillation）   — 大 GNN 蒸馏出轻量 MLP 学生（毫秒级单样本推理）
6. 增量（incremental）    — 新增接触者仅局部重算子图，而非全图重算

入口统一从 ``tb_risk.scoring.ml.gnn_scalable`` 导入，无需改动既有 GNN 训练路径。
"""
from ._common import (
    LOGGER,
    PYTORCH_AVAILABLE,
    PYG_AVAILABLE,
    NEIGHBOR_LOADER_AVAILABLE,
    DYNAMIC_QUANT_AVAILABLE,
)

from .sampling import (
    sample_k_hop_subgraph,
    manual_k_hop_subgraph,
    build_neighbor_sampler,
    NeighborBatch,
)

from .mini_batch import MinibatchConfig, train_gnn_minibatch

from .inference import predict_target_k_hop, predict_subgraph

from .quantization import (
    quantize_model_fp16,
    quantize_model_int8,
    predict_quantized,
    measure_inference_latency,
)

from .distillation import GNNStudentMLP, distill_gnn_to_mlp, predict_with_mlp

from .incremental import add_new_node, incremental_predict_new_node

__all__ = [
    # 可用性标志
    'LOGGER', 'PYTORCH_AVAILABLE', 'PYG_AVAILABLE',
    'NEIGHBOR_LOADER_AVAILABLE', 'DYNAMIC_QUANT_AVAILABLE',
    # sampling
    'sample_k_hop_subgraph', 'manual_k_hop_subgraph',
    'build_neighbor_sampler', 'NeighborBatch',
    # mini-batch
    'MinibatchConfig', 'train_gnn_minibatch',
    # inference
    'predict_target_k_hop', 'predict_subgraph',
    # quantization
    'quantize_model_fp16', 'quantize_model_int8',
    'predict_quantized', 'measure_inference_latency',
    # distillation
    'GNNStudentMLP', 'distill_gnn_to_mlp', 'predict_with_mlp',
    # incremental
    'add_new_node', 'incremental_predict_new_node',
]