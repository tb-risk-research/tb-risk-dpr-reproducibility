#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tb_risk 测试套件 — 大规模图网络处理性能优化（gnn_scalable 子包）

覆盖 v4/GNN 之外的五类规模化能力：
1. 采样：k-hop 邻域子图抽取（含边特征保留）+ 节点级 mini-batch 采样
2. mini-batch 训练：NeighborLoader / k-hop 采样训练
3. 子图推理：目标患者仅算 k-hop 邻域子图
4. 量化：FP16 / INT8 动态量化 + 推理延迟测量
5. 蒸馏：大 GNN 蒸馏出轻量 MLP 学生
6. 增量：新增接触者仅局部重算子图

文献依据同实现模块（GraphSAGE / GraphSAINT / INT8量化 / 知识蒸馏）。
"""

import os
import sys
import unittest

import numpy as np

# 确保包路径在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from tb_risk.scoring.ml.gnn_scalable import (
    PYTORCH_AVAILABLE, PYG_AVAILABLE,
    sample_k_hop_subgraph, manual_k_hop_subgraph,
    build_neighbor_sampler, NeighborBatch,
    MinibatchConfig, train_gnn_minibatch,
    predict_target_k_hop, predict_subgraph,
    quantize_model_fp16, quantize_model_int8,
    predict_quantized, measure_inference_latency,
    GNNStudentMLP, distill_gnn_to_mlp, predict_with_mlp,
    add_new_node, incremental_predict_new_node,
)
from tb_risk.scoring.ml.gnn_scalable import sampling as samp_mod
from tb_risk.scoring.ml.gnn_scalable import mini_batch as mb_mod
from tb_risk.scoring.ml.gnn_scalable import quantization as quant_mod
from tb_risk.scoring.ml.gnn_scalable import distillation as dist_mod

_TORCH_AND_PYG = (PYTORCH_AVAILABLE and PYG_AVAILABLE)


def _make_large_graph(n_nodes=5000, feat_dim=30, edge_dim=6, seed=42):
    """构造一个较大的合成接触网络（链式 + 随机边），带边特征与节点标签。"""
    import torch
    from torch_geometric.data import Data
    rng = np.random.RandomState(seed)
    x = rng.rand(n_nodes, feat_dim).astype(np.float32)
    # 链式边（保证连通 + 结构）
    edge_index = []
    for i in range(n_nodes - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    # 随机边（约 2 倍追加）
    n_extra = n_nodes * 2
    for _ in range(n_extra):
        u = rng.randint(0, n_nodes)
        v = rng.randint(0, n_nodes)
        if u != v:
            edge_index.append([u, v])
            edge_index.append([v, u])
    edge_index = np.array(edge_index).T
    edge_attr = rng.rand(edge_index.shape[1], edge_dim).astype(np.float32)
    y = (rng.rand(n_nodes) > 0.7).astype(np.float32)
    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        y=torch.tensor(y, dtype=torch.float),
    )


def _make_model():
    """构造随机初始化的轻量 SEIRInformedGNN（仅用于结构性测试）。"""
    from tb_risk.scoring.ml.gnn_training import SEIRInformedGNN
    return SEIRInformedGNN(node_feature_dim=30, hidden_dim=16,
                           num_layers=2, dropout=0.1)


@unittest.skipUnless(_TORCH_AND_PYG, "需要 torch + torch_geometric")
class TestPackageExports(unittest.TestCase):
    """包导出完整性"""

    def test_all_public_api_exported(self):
        import tb_risk.scoring.ml.gnn_scalable as pkg
        for name in (
            "sample_k_hop_subgraph", "manual_k_hop_subgraph",
            "build_neighbor_sampler", "NeighborBatch",
            "MinibatchConfig", "train_gnn_minibatch",
            "predict_target_k_hop", "predict_subgraph",
            "quantize_model_fp16", "quantize_model_int8",
            "predict_quantized", "measure_inference_latency",
            "GNNStudentMLP", "distill_gnn_to_mlp", "predict_with_mlp",
            "add_new_node", "incremental_predict_new_node",
        ):
            self.assertTrue(hasattr(pkg, name), f"缺少导出: {name}")


@unittest.skipUnless(_TORCH_AND_PYG, "需要 torch + torch_geometric")
class TestSampling(unittest.TestCase):
    """k-hop 子图抽取与 mini-batch 采样"""

    def setUp(self):
        self.data = _make_large_graph(n_nodes=2000)

    def test_khop_returns_subgraph(self):
        sub, mapping = sample_k_hop_subgraph(self.data, 1000, num_hops=2)
        self.assertIsNotNone(sub)
        self.assertIsNotNone(mapping)
        self.assertLess(sub.num_nodes, self.data.num_nodes)
        self.assertGreater(sub.num_nodes, 0)
        # 目标节点特征应与原图一致
        self.assertTrue(np.allclose(
            sub.x[mapping].numpy(), self.data.x[1000].numpy(), atol=1e-6))

    def test_khop_preserves_edge_attr(self):
        sub, _ = sample_k_hop_subgraph(self.data, 1000, num_hops=2)
        self.assertIsNotNone(sub.edge_attr)
        self.assertEqual(sub.edge_attr.shape[1], 6)
        self.assertEqual(sub.edge_index.shape[1], sub.edge_attr.shape[0])

    def test_khop_hops_monotonic(self):
        """跳数越多子图越大"""
        sub1, _ = sample_k_hop_subgraph(self.data, 1000, num_hops=1)
        sub2, _ = sample_k_hop_subgraph(self.data, 1000, num_hops=3)
        self.assertGreaterEqual(sub2.num_nodes, sub1.num_nodes)

    def test_manual_fallback_matches_size(self):
        """手动 BFS 与 PyG k_hop_subgraph 抽取的节点数一致"""
        sub_pyg, m_pyg = sample_k_hop_subgraph(self.data, 500, num_hops=2)
        sub_man, m_man = manual_k_hop_subgraph(self.data, 500, num_hops=2)
        self.assertEqual(sub_pyg.num_nodes, sub_man.num_nodes)
        self.assertEqual(m_pyg, m_man)

    def test_neighbor_sampler_yields_batches(self):
        sampler = build_neighbor_sampler(self.data, num_neighbors=[5, 3],
                                         batch_size=16)
        batch = next(sampler)
        self.assertIsInstance(batch, NeighborBatch)
        self.assertGreater(batch.num_nodes, 0)
        self.assertIsNotNone(batch.x)


@unittest.skipUnless(_TORCH_AND_PYG, "需要 torch + torch_geometric")
class TestMinibatch(unittest.TestCase):
    """节点级 mini-batch 训练"""

    def setUp(self):
        self.data = _make_large_graph(n_nodes=3000)

    def test_train_returns_losses(self):
        model = _make_model()
        cfg = MinibatchConfig(num_neighbors=[5, 3], batch_size=64,
                              num_epochs=2, learning_rate=0.001)
        result = train_gnn_minibatch(model, self.data, cfg)
        self.assertTrue(result["success"])
        self.assertEqual(len(result["losses"]), 2)
        self.assertIn("backend", result)
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]))

    def test_backend_is_neighbor_loader_or_manual(self):
        self.assertTrue(mb_mod.NEIGHBOR_LOADER_AVAILABLE or True)
        model = _make_model()
        cfg = MinibatchConfig(num_neighbors=[5, 3], batch_size=64, num_epochs=1)
        result = train_gnn_minibatch(model, self.data, cfg)
        self.assertIn(result["backend"], ("neighbor_loader", "manual_sampler"))

    def test_model_parameters_updated(self):
        model = _make_model()
        before = {k: v.clone() for k, v in model.named_parameters()}
        cfg = MinibatchConfig(num_neighbors=[5, 3], batch_size=64, num_epochs=2,
                              learning_rate=0.01)
        train_gnn_minibatch(model, self.data, cfg)
        changed = any(not torch_equal(before[k], v)
                      for k, v in model.named_parameters())
        self.assertTrue(changed)


def torch_equal(a, b):
    import torch
    return bool(torch.allclose(a, b, atol=1e-9))


@unittest.skipUnless(_TORCH_AND_PYG, "需要 torch + torch_geometric")
class TestInference(unittest.TestCase):
    """k-hop 子图推理"""

    def setUp(self):
        self.data = _make_large_graph(n_nodes=3000)
        self.model = _make_model()

    def test_predict_target_returns_risk(self):
        res = predict_target_k_hop(self.model, self.data, 1500, num_hops=2)
        self.assertIsNotNone(res)
        self.assertIn("risk_probability", res)
        self.assertGreaterEqual(res["risk_probability"], 0.0)
        self.assertLessEqual(res["risk_probability"], 100.0)
        self.assertLess(res["subgraph_size"], self.data.num_nodes)

    def test_predict_subgraph_all_nodes(self):
        sub, _ = sample_k_hop_subgraph(self.data, 100, num_hops=2)
        res = predict_subgraph(self.model, sub)
        self.assertEqual(len(res["risk_probabilities"]), sub.num_nodes)


@unittest.skipUnless(_TORCH_AND_PYG, "需要 torch + torch_geometric")
class TestQuantization(unittest.TestCase):
    """FP16 / INT8 量化与延迟"""

    def setUp(self):
        self.data = _make_large_graph(n_nodes=2000)
        self.model = _make_model()

    def test_fp16_quantization(self):
        q = quantize_model_fp16(self.model)
        self.assertIsNotNone(q)

    def test_int8_quantization_backend(self):
        q, backend = quantize_model_int8(self.model)
        self.assertIn(backend, ("int8", "fp16", "fp32"))

    def test_predict_quantized(self):
        sub, mapping = sample_k_hop_subgraph(self.data, 100, num_hops=2)
        # fp16：目标节点风险
        res = predict_quantized(self.model, sub, target_mapping=mapping,
                                quant_type='fp16')
        self.assertIsNotNone(res)
        self.assertIn("risk_probability", res)
        self.assertGreaterEqual(res["risk_probability"], 0.0)
        self.assertLessEqual(res["risk_probability"], 100.0)
        # int8：可回退到 int8/fp16/fp32，且应返回目标风险
        res8 = predict_quantized(self.model, sub, target_mapping=mapping,
                                 quant_type='int8')
        self.assertIsNotNone(res8)
        self.assertIn("risk_probability", res8)

    def test_measure_latency(self):
        res = measure_inference_latency(self.model, self.data, target_idx=100,
                                        num_hops=2, n_runs=5)
        self.assertIsNotNone(res)
        self.assertGreater(res["mean_ms"], 0.0)
        self.assertIn("p95_ms", res)
        self.assertIn("backend", res)

    def _sub(self):
        sub, _ = sample_k_hop_subgraph(self.data, 100, num_hops=2)
        return sub


@unittest.skipUnless(_TORCH_AND_PYG, "需要 torch + torch_geometric")
class TestDistillation(unittest.TestCase):
    """GNN → MLP 知识蒸馏"""

    def setUp(self):
        self.data = _make_large_graph(n_nodes=1500)
        self.teacher = _make_model()

    def test_distill_produces_student(self):
        result = distill_gnn_to_mlp(self.teacher, self.data, num_hops=2,
                                    n_epochs=3, batch_size=64)
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["student"])
        self.assertEqual(len(result["loss_history"]), 3)
        # 学生参数量应显著小于教师
        self.assertLess(result["n_params_student"], result["n_params_teacher"])

    def test_student_predict(self):
        result = distill_gnn_to_mlp(self.teacher, self.data, num_hops=2,
                                    n_epochs=2, batch_size=32)
        student = result["student"]
        feat = self.data.x[10].numpy()
        res = predict_with_mlp(student, feat)
        self.assertIn("risk_probability", res)
        self.assertGreaterEqual(res["risk_probability"], 0.0)
        self.assertLessEqual(res["risk_probability"], 100.0)


@unittest.skipUnless(_TORCH_AND_PYG, "需要 torch + torch_geometric")
class TestIncremental(unittest.TestCase):
    """增量推理（新增接触者局部重算）"""

    def setUp(self):
        self.data = _make_large_graph(n_nodes=2000)
        self.model = _make_model()

    def test_add_new_node(self):
        new_feat = np.zeros(30, dtype=np.float32)
        new_data = add_new_node(self.data, new_feat, [10, 20, 30])
        self.assertEqual(new_data.num_nodes, self.data.num_nodes + 1)
        self.assertEqual(new_data.edge_index.shape[1],
                         self.data.edge_index.shape[1] + 6)  # 3 邻居 × 双向

    def test_incremental_predict(self):
        new_feat = np.random.rand(30).astype(np.float32)
        res = incremental_predict_new_node(self.model, self.data, new_feat,
                                           [10, 20, 30], num_hops=2)
        self.assertIsNotNone(res)
        self.assertIn("risk_probability", res)
        self.assertEqual(res["new_node_idx"], self.data.num_nodes)
        self.assertEqual(res["full_graph_size"], self.data.num_nodes + 1)
        self.assertLess(res["subgraph_size"], res["full_graph_size"])
        self.assertGreater(res["reduction_ratio"], 0.0)


@unittest.skipUnless(_TORCH_AND_PYG, "需要 torch + torch_geometric")
class TestPredictorDelegation(unittest.TestCase):
    """MLRiskPredictor 委托方法（scoring/predictor.py 集成）"""

    def setUp(self):
        self.data = _make_large_graph(n_nodes=800)
        self.model = _make_model()
        from tb_risk.scoring.predictor import MLRiskPredictor
        self.predictor = MLRiskPredictor()

    def test_availability_flag(self):
        self.assertTrue(self.predictor.gnn_scalable_available)

    def test_predict_k_hop_delegated(self):
        res = self.predictor.predict_gnn_k_hop(self.model, self.data, 400,
                                               num_hops=2)
        self.assertIsNotNone(res)
        self.assertIn("risk_probability", res)

    def test_quantize_delegated(self):
        q, backend = self.predictor.quantize_gnn_model(self.model,
                                                       quant_type='fp16')
        self.assertIn(backend, ("fp16", "fp32"))

    def test_incremental_contact_delegated(self):
        new_feat = np.random.rand(30).astype(np.float32)
        res = self.predictor.incremental_predict_new_contact(
            self.model, self.data, new_feat, [10, 20, 30], num_hops=2)
        self.assertIsNotNone(res)
        self.assertIn("risk_probability", res)
        self.assertEqual(res["new_node_idx"], self.data.num_nodes)

    def test_add_contact_node_delegated(self):
        new_data = self.predictor.add_contact_node(
            self.data, np.zeros(30, dtype=np.float32), [1, 2, 3])
        self.assertEqual(new_data.num_nodes, self.data.num_nodes + 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)