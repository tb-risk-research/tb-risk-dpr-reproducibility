#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""类型化边网络五臂消融正式运行（"第一层：异质边 + 注意力"，2026-08-25）。

用户规格：边类型（家庭/社会/同事/同学/偶遇，5 种基础传播率）、
边特征（频率/时长/距离/通风等 6 维）、GAT 注意力（节点+边特征
共同计算注意力系数）、2 层 + 残差连接。

DGP v3（validation/typed_network.py，经三轮设计迭代验证）：
  传染源节点（infectivity 只在图）+ 边特征全类型同分布 +
  类型差异只存在于 β 与边类型标注——信息不对称的构造性保证。

五臂阶梯（validation/typed_ablation.py）：
  individual（个体基线）→ mean_agg（等权邻居均值，现行语义）→
  channel_agg（5 类型独立聚合，类型区分可学上界）→
  gat_hetero（2 层 HeteroGAT 端到端）→ gat_homo（边类型抹除对照）。

统计口径：分层 50/50 半区（接触者空间）+ 逐种子 DeLong 配对检验 +
种子级 bootstrap 95% CI（重采样单位 = 种子）。

诚实边界：信息不对称为构造性设定（对齐部署系统 22 列特征无场景
分解暴露特征的真实约束）；本实验回答"机制能否兑现"，不回答
"真实世界必然存在"。

产物：data/processed/typed_ablation_multiseed_20260825.json
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

from tb_risk.validation.typed_ablation import (  # noqa: E402
    run_multi_seed_typed_ablation, run_typed_ablation_once)


def main():
    n_contacts, n_seeds, epochs, target_rate = 400, 20, 500, 0.25

    print('=== 类型化边网络五臂消融（DGP v3，%d 种子）===' % n_seeds)
    report = run_multi_seed_typed_ablation(
        n_contacts=n_contacts, n_seeds=n_seeds, seed_start=101,
        epochs=epochs, target_rate=target_rate)

    # ---- 汇总表 ----
    print('\n--- 各臂 AUROC（%d 种子均值 [bootstrap 95%% CI]）---' % n_seeds)
    for k, v in report['arm_summary'].items():
        print('%-12s %.4f [%.4f, %.4f]' % (
            k, v['mean_auroc'],
            v['bootstrap_ci'][0], v['bootstrap_ci'][1]))

    print('\n--- 消融阶梯（ΔAUROC，正值 = 上位臂更优）---')
    for k, v in report['ladder_summary'].items():
        print('%-28s %+0.4f [%+0.4f, %+0.4f]  正种子 %2d/%d' % (
            k, v['mean'], v['bootstrap_ci'][0], v['bootstrap_ci'][1],
            v['positive_seeds'], n_seeds))

    print('\n网络层增量存在:', report['network_increment_exists'])
    print('类型注意力净贡献显著:', report['type_attention_contributes'])
    print('\n结论:', report['conclusion'])

    # ---- 归档 ----
    out = {
        'experiment': 'typed_network_five_arm_ablation',
        'user_directive': '第一层：异质边 + 注意力（复杂网络机制实现路径）',
        'date': '2026-08-25',
        'dgp': 'typed_contact_network_v3',
        'five_arms': [
            'individual: 个体特征 logistic（宿主 + 无类型暴露聚合）',
            'mean_agg: + 等权邻居均值聚合（现行网络层语义）',
            'channel_agg: + 5 类型独立邻居均值（类型区分可学上界）',
            'gat_hetero: 2 层 HeteroGAT 端到端（类型感知注意力）',
            'gat_homo: 同结构对照（5 类边并入单通道，无类型区分）',
        ],
        'report': report,
    }
    path = os.path.join(PROC, 'typed_ablation_multiseed_20260825.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('\n归档:', path)


if __name__ == '__main__':
    main()
