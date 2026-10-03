#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GNN训练 — 数据准备子模块。

包含合成图数据集生成（特征工程、图数据生成、标签传播）。
从原 scoring/ml/gnn_training.py 拆分而来，不修改业务逻辑。
"""
import numpy as np

from ._common import LOGGER, PYTORCH_AVAILABLE, PYG_AVAILABLE


def _generate_synthetic_graphs(predictor, n_graphs=100, random_state=42, device=None):
        """生成合成图数据集（M7：感染沿边传播，标签= f(自身易感性, 邻居感染×边强度)）

        M6→M7 诊断结论（2026-08-22 基线实验 data/_diagnose_gnn_baseline.py）：
        - 旧版节点特征为纯随机噪声（rng.rand(22)）且种子随机选，
          "谁是种子"在特征上不可见 → 邻居特征对预测邻居风险无信息量
        - 患者节点（确诊传染源）不参与传播（risk[0] 恒 0）
        - 实测：仅特征基线 AUROC=0.517、特征+邻居聚合线性基线=0.658、
          GNN=0.526 → 图上仅有度数/拓扑位置等微弱间接信号

        M7 生成模型（网络传播标准范式）：
        - Keeling & Rohani, Modeling Infectious Diseases, 2008（边加权传播）
        - Newman, SIAM Review 2003（网络流行病学）
        - Zhu et al., GAST 2024（边特征使用）

        生成流程：
        1. 节点特征携带真实信息：
           x[:, :5]   传染性因子（决定节点作为感染源的强度）
           x[:, 5:10] 易感性因子（决定暴露后感染概率）
           x[:, 10:]  无关噪声特征（模拟现实测量冗余）
        2. 感染源 = 确诊患者（满传染性）+ 1-2 个潜伏种子
           （按传染性加权抽样 → 高传染性节点更可能成为种子，
            使"邻居特征"对邻居是否感染有预测力）
        3. 两轮沿边传播（一跳 + 二跳）：
           exposure[j] = Σ_n infectious[n] × edge_weight(n→j)
           p[j] = 1 - exp(-exposure[j] × susceptibility[j])
           y[j] ~ Bernoulli(p[j])，随机性来自传播过程本身
        4. 边传播强度写入 edge_attr[:, 0]（家庭边强 U(0.5,1)，
           患者边中 U(0.2,0.6)，区域边 0 —— 区域节点仅作上下文中转）

        验证标准：修复后 GNN 显著优于仅特征 MLP（AUROC 差 > 0.05），
        且特征+邻居聚合基线显著优于仅特征基线。
        """
        if not PYTORCH_AVAILABLE or not PYG_AVAILABLE:
            return [], [], []

        import torch
        from torch_geometric.data import Data

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        MAX_GRAPHS = 5000
        if n_graphs > MAX_GRAPHS:
            LOGGER.warning(f"_generate_synthetic_graphs: n_graphs={n_graphs} exceeds MAX_GRAPHS={MAX_GRAPHS}, limiting to {MAX_GRAPHS}")
            n_graphs = MAX_GRAPHS

        rng = np.random.RandomState(random_state)
        graphs = []

        # 传染性读出权重（前5维）；中心 -2.5 使 sigmoid 输出约 0.2-0.6
        w_infect = np.array([0.6, 0.8, 1.0, 0.8, 0.6], dtype=np.float64)

        for i in range(n_graphs):
            num_family = rng.randint(2, 6)
            num_social = rng.randint(3, 10)

            # 生成节点特征
            # 多尺度特征融合 v5.0：末尾追加 1 个区域级节点（全局上下文），
            # 与真实 heterogeneous_network 的"个体—家庭—社区—区域"层次对齐。
            num_region = 1
            num_nodes = 1 + num_family + num_social + num_region
            x = rng.rand(num_nodes, 22).astype(np.float32)
            region_idx = num_nodes - 1  # 区域节点索引

            # 生成边索引和边特征（6 维；第 0 维 = 传播强度，供 GAT 边注意力使用）
            edge_index = []
            edge_attr_list = []
            # 患者到所有接触者：中等传播强度 U(0.2, 0.6)
            for j in range(1, num_nodes - num_region):
                w = 0.05 + rng.rand() * 0.2
                edge_feature = rng.rand(6).astype(np.float32)
                edge_feature[0] = w
                edge_index.append([0, j])
                edge_index.append([j, 0])
                edge_attr_list.append(edge_feature)
                edge_attr_list.append(edge_feature.copy())
            # 家庭内部连接：强传播 U(0.5, 1.0)
            for j in range(1, 1 + num_family):
                for k in range(j + 1, 1 + num_family):
                    w = 0.5 + rng.rand() * 0.5
                    edge_feature = rng.rand(6).astype(np.float32)
                    edge_feature[0] = w
                    edge_index.append([j, k])
                    edge_index.append([k, j])
                    edge_attr_list.append(edge_feature)
                    edge_attr_list.append(edge_feature.copy())
            # 区域节点 ↔ 所有个体节点：上下文中转，传播强度 0
            # （避免"患者→区域→社交"路径抹平家庭/社交的结构差异）
            for j in range(0, num_nodes - num_region):
                edge_feature = rng.rand(6).astype(np.float32)
                edge_feature[0] = 0.0
                edge_index.append([region_idx, j])
                edge_index.append([j, region_idx])
                edge_attr_list.append(edge_feature)
                edge_attr_list.append(edge_feature.copy())

            edge_index = np.array(edge_index).T
            edge_attr = np.array(edge_attr_list, dtype=np.float32)

            # 接触广度（入度）作为第 23 维特征：暴露风险的直接量度。
            # 流行病学依据：接触者追踪中"接触人数"是核心风险因子
            # （家庭规模大/社交广 → 暴露机会多）；同时修复 GAT 均值型
            # 聚合对度数编码弱的问题（线性基线 B 实验显示度数单独
            # 贡献 AUROC +0.07）
            in_deg = np.zeros(num_nodes)
            for e in range(edge_index.shape[1]):
                in_deg[int(edge_index[1, e])] += 1.0
            x = np.concatenate(
                [x, (in_deg / max(in_deg.max(), 1.0)).reshape(-1, 1)],
                axis=1).astype(np.float32)

            # === M7 标签：感染沿边传播（Bernoulli 抽样，非阈值切分） ===
            # 传染性：前5维加权和 → sigmoid（约 0.2-0.6）
            infectious = 1.0 / (1.0 + np.exp(
                -(x[:, :5].astype(np.float64) @ w_infect - 2.0)))
            # 易感性：5-9 维均值 → [0.5, 1.0]
            susceptibility = 0.5 + 0.5 * x[:, 5:10].mean(axis=1)

            # 感染源初始化：患者 = 确诊（满传染性）；种子按传染性加权抽样
            infected = np.zeros(num_nodes, dtype=bool)
            infected[0] = True
            contact_indices = np.arange(1, num_nodes - 1)  # 排除区域节点
            n_seeds = 1 + int(rng.rand() * 3)
            if len(contact_indices) > n_seeds:
                p_seed = infectious[contact_indices]
                seeds = rng.choice(contact_indices, size=n_seeds, replace=False,
                                   p=p_seed / p_seed.sum())
                infected[seeds] = True

            # 入边表（dst 的暴露来自 src 的传染性 × 边强度）
            in_edges = {j: [] for j in range(num_nodes)}
            for e in range(edge_index.shape[1]):
                src, dst = int(edge_index[0, e]), int(edge_index[1, e])
                in_edges[dst].append((src, float(edge_attr[e, 0])))

            # === 邻居确认密度（第 24 维，M 级审计修复——约束要求的结构信号） ===
            # 定义：节点的一跳邻居中"已确认"节点占比。确认集取**传播前的
            # 初始确认集**（确诊患者 + 抽样种子）——这是传播过程开始前
            # 已观察到的状态，最终感染态不参与 → 无标签泄漏。
            # 现实对应：接触者追踪级联中"该接触者的邻居里已确诊的比例"
            # （观察史，部署时可得）；与入度（接触广度）互补——入度度量
            # 暴露机会总量，确认密度度量暴露源的"已确认强度"。
            # 显式构造为节点特征后，GNN 无需仅靠聚合隐式恢复该信号。
            initial_confirmed = infected.copy()   # 此时尚未传播：患者+种子
            initial_confirmed[region_idx] = False
            conf_density = np.zeros(num_nodes, dtype=np.float32)
            for j in range(num_nodes):
                nbrs = in_edges[j]
                if nbrs:
                    conf_density[j] = float(sum(
                        1 for s, _ in nbrs if initial_confirmed[s])) / len(nbrs)
            x = np.concatenate(
                [x, conf_density.reshape(-1, 1)], axis=1).astype(np.float32)

            # 两轮传播（一跳 + 二跳）；区域节点既不传染也不被感染
            current = infectious * infected
            current[0] = 0.35      # 确诊患者部分隔离（治疗中），保留中等传染性
            current[region_idx] = 0.0
            for _hop in range(2):
                newly_infected = []
                for j in range(1, num_nodes - 1):
                    if infected[j]:
                        continue
                    exposure = sum(current[s] * w for s, w in in_edges[j])
                    p_inf = 1.0 - np.exp(-exposure * susceptibility[j])
                    if rng.rand() < p_inf:
                        newly_infected.append(j)
                for j in newly_infected:
                    infected[j] = True
                    current[j] = infectious[j]

            y = infected.astype(np.float64)
            y[region_idx] = 0.0  # 区域节点无个体标签

            # 生成节点mask，训练时忽略患者节点（仅预测接触者）
            mask = np.ones(num_nodes, dtype=bool)
            mask[0] = False  # 患者节点不参与损失计算
            mask[region_idx] = False  # 区域级节点（抽象上下文，不产出个体标签）

            data = Data(
                x=torch.tensor(x, dtype=torch.float, device=device),
                edge_index=torch.tensor(edge_index, dtype=torch.long, device=device),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float, device=device),
                y=torch.tensor(y, dtype=torch.float, device=device),
                mask=torch.tensor(mask, dtype=torch.bool, device=device)
            )
            graphs.append(data)

        # 划分数据集
        train_size = int(0.7 * n_graphs)
        val_size = int(0.2 * n_graphs)

        return graphs[:train_size], graphs[train_size:train_size+val_size], graphs[train_size+val_size:]
