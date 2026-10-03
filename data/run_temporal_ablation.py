#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""时序接触网络五臂消融正式运行（"第二层：时序 GNN（接触时间维度）"，
2026-08-25）。

用户规格：时间离散化（4~6 个时间窗：0-2 周 / 2-4 周 / 1-3 月 /
3-6 月 / 6 月+）；每窗一张子图，子图内 GAT 聚合；子图之间用
GRU 传递时序状态（文献调研：SE-HTGNN NeurIPS 2025 显示 GRU 与
LSTM 相当且更高效，Transformer 变体欠佳）；输出每个节点的
时序风险曲线。

DGP v1（validation/temporal_network.py，2026-08-25 扫描调优）：
  时间切片 + 衰减权重（半衰期 8 周，IGRA 窗口期锚点）+ 每接触者
  时序画像（recent 35% / spread 30% / old 35%）——个体只见
  time_span 总量，时序分布只在切片图里（时间信息不对称的构造性
  保证）。oracle−个体间隙 +0.031（5 种子）。

五臂阶梯（validation/temporal_ablation.py）：
  individual（个体基线，含 time_span 总量）→ static_agg（跨窗
  等权聚合，现行网络层语义）→ window_agg（5 窗独立聚合，时序
  区分可学上界）→ stgnn（每窗 GAT + 窗间 GRU 端到端）→
  stgnn_merged（时序盲对照：全窗并入单一静态子图）。

统计口径：分层 50/50 半区（接触者空间）+ 逐种子 DeLong 配对检验 +
种子级 bootstrap 95% CI（重采样单位 = 种子）。

诚实边界：时间信息不对称为构造性设定（对齐部署系统 22 列特征
只有 time_span 总量、无接触时间分解的真实约束）；本实验回答
"机制能否兑现"，不回答"真实世界必然存在"（需带接触时间戳的
真实数据裁决）。

产物：data/processed/temporal_ablation_multiseed_20260825.json
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DESKTOP = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, DESKTOP)

PROC = os.path.join(HERE, 'processed')

from tb_risk.validation.temporal_ablation import (  # noqa: E402
    run_multi_seed_temporal_ablation)


def main():
    n_contacts, n_seeds, epochs, target_rate = 400, 20, 500, 0.25

    print('=== 时序接触网络五臂消融（DGP v1，%d 种子）===' % n_seeds)
    report = run_multi_seed_temporal_ablation(
        n_contacts=n_contacts, n_seeds=n_seeds, seed_start=101,
        epochs=epochs, target_rate=target_rate)

    # ---- 汇总表 ----
    print('\n--- 各臂 AUROC（%d 种子均值 [bootstrap 95%% CI]）---' % n_seeds)
    for k, v in report['arm_summary'].items():
        print('%-14s %.4f [%.4f, %.4f]' % (
            k, v['mean_auroc'],
            v['bootstrap_ci'][0], v['bootstrap_ci'][1]))

    print('\n--- 消融阶梯（ΔAUROC，正值 = 上位臂更优）---')
    for k, v in report['ladder_summary'].items():
        print('%-28s %+0.4f [%+0.4f, %+0.4f]  正种子 %2d/%d' % (
            k, v['mean'], v['bootstrap_ci'][0], v['bootstrap_ci'][1],
            v['positive_seeds'], n_seeds))

    print('\n网络层增量存在:', report['network_increment_exists'])
    print('时序机制净贡献显著:', report['temporal_mechanism_contributes'])
    print('\n结论:', report['conclusion'])

    # ---- 归档 ----
    out = {
        'experiment': 'temporal_network_five_arm_ablation',
        'user_directive': '第二层：时序 GNN（接触时间维度）'
                          '（复杂网络机制实现路径）',
        'date': '2026-08-25',
        'dgp': 'temporal_contact_network_v1',
        'five_arms': [
            'individual: 个体特征 logistic（宿主 + time_span 总量等'
            '无时序聚合）',
            'static_agg: + 跨窗等权邻居均值聚合（现行网络层语义，'
            '时序盲）',
            'window_agg: + 5 窗独立邻居均值（时序区分可学上界）',
            'stgnn: 每窗 GAT + 窗间 GRU 端到端（时序感知，输出时序'
            '风险曲线）',
            'stgnn_merged: 同结构时序盲对照（全窗并入单一静态子图，'
            '窗标注抹除）',
        ],
        'report': report,
    }
    path = os.path.join(PROC, 'temporal_ablation_multiseed_20260825.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('\n归档:', path)


if __name__ == '__main__':
    main()
