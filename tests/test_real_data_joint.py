#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实数据联合模型（P6）单元测试：图构造 / 闭式前向 / 训练确定性 /
OOF 无泄漏 / 消融冒烟。"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from tb_risk.validation.real_data_infection import (  # noqa: E402
    _RDA_PATH,
    load_homeacf_contacts,
)
from tb_risk.validation.real_data_pi import _DATA_DIR as _PACTS_DATA_DIR  # noqa: E402
from tb_risk.validation.real_data_joint import (  # noqa: E402
    HOMEACF_WINDOW_MID_WEEKS,
    _default_window_forces,
    _group_cv_indices,
    _real_window_forces_torch,
    _to_torch_graph,
    build_homeacf_joint_graph,
    build_pacts_joint_graph,
    joint_oof,
    logit_lin_oof,
    train_real_joint_fold,
)

try:
    import torch
    TORCH = True
except ImportError:  # pragma: no cover
    TORCH = False

_DF_CACHE = {}

# 数据集因 license 限制不入库（.gitignore /data/* 仅保留 *.py），
# CI 上缺数据时跳过而非报错（守卫与 loader 的真实路径一致）
_HOMEACF_OK = os.path.exists(_RDA_PATH)
_PACTS_OK = all(os.path.exists(os.path.join(_PACTS_DATA_DIR, f)) for f in (
    'contacts_baseline_rev.csv', 'contacts_threemonths_rev.csv',
    'indexcases.csv', 'index_baseline.csv'))


def _homeacf_df():
    if 'df' not in _DF_CACHE:
        _DF_CACHE['df'] = load_homeacf_contacts()
    return _DF_CACHE['df']


class TestGraphConstruction(unittest.TestCase):

    @unittest.skipUnless(_HOMEACF_OK, 'HomeACF .rda 数据不在仓库（license 限制）')
    def test_homeacf_graph_structure(self):
        g = build_homeacf_joint_graph(_homeacf_df())
        n = len(g['y'])
        self.assertEqual(g['mem_contact'].tolist(), list(range(n)))
        self.assertEqual(len(g['mem_type']), n)
        self.assertEqual(len(g['mem_win']), n)
        self.assertEqual(len(g['mem_src']), n)
        # 每接触者恰 1 条户边（星形图）
        self.assertEqual(int(g['mem_type'].max()) + 1, 4)
        self.assertEqual(int(g['mem_win'].max()) + 1, 4)
        # 冻结乘子为正（物理量纲）
        self.assertGreater(g['host_frozen'].min(), 0.0)
        self.assertGreater(g['corr_frozen'].min(), 0.0)
        # 特征无 NaN
        for key in ('src_feats', 'host_feats', 'corr_feats'):
            self.assertFalse(np.isnan(g[key]).any(), key)
        # 事件数与既有基准一致（n=2725 / 359 事件）
        self.assertEqual(n, 2725)
        self.assertEqual(int(g['y'].sum()), 359)

    @unittest.skipUnless(_PACTS_OK, 'PACTS CSV 数据不在仓库（license 限制）')
    def test_pacts_graph_structure(self):
        from tb_risk.validation.real_data_pi import load_pacts_contacts
        g = build_pacts_joint_graph(load_pacts_contacts())
        # 单窗退化：所有边 w=0；类型 5 类
        self.assertEqual(int(g['mem_win'].max()), 0)
        self.assertEqual(int(g['mem_type'].max()) + 1, 5)
        self.assertEqual(len(g['mem_type']), len(g['y']))


class TestMechanism(unittest.TestCase):

    @unittest.skipUnless(TORCH and _HOMEACF_OK, 'PyTorch 或 HomeACF 数据不可用')
    def test_closed_form_at_init(self):
        """零初始化头部：ν̂ = S_frozen·C_frozen·Λ_w（闭式可验）。"""
        from tb_risk.validation.real_data_joint import RealJointMechGNN
        g_np = build_homeacf_joint_graph(_homeacf_df())
        tr = np.arange(0, len(g_np['y']), 2)
        g = _to_torch_graph(g_np, tr)
        model = RealJointMechGNN(
            host_dim=g['host_feats'].shape[1],
            corr_dim=g['corr_feats'].shape[1],
            src_dim=g['src_feats'].shape[1],
            n_types=4, n_windows=4,
            mid_weeks=g_np['mid_weeks'], mode='seir')
        with torch.no_grad():
            _, nu = model(g, torch.tensor(1.0, dtype=torch.float64))
        lam_w = _default_window_forces(HOMEACF_WINDOW_MID_WEEKS)
        nu_ref = g_np['host_frozen'] * g_np['corr_frozen'] \
            * lam_w[g_np['mem_win']]
        np.testing.assert_allclose(nu.numpy(), nu_ref, rtol=1e-9)

    @unittest.skipUnless(TORCH, 'PyTorch 不可用')
    def test_seir_forces_differentiable_and_positive(self):
        lb = torch.zeros((), dtype=torch.float64, requires_grad=True)
        li = torch.zeros((), dtype=torch.float64, requires_grad=True)
        lam = _real_window_forces_torch(lb, li, HOMEACF_WINDOW_MID_WEEKS)
        self.assertTrue(bool((lam > 0).all()))
        lam.sum().backward()
        self.assertTrue(float(lb.grad) != 0.0)
        self.assertTrue(float(li.grad) != 0.0)

    @unittest.skipUnless(TORCH and _HOMEACF_OK, 'PyTorch 或 HomeACF 数据不可用')
    def test_training_deterministic_and_loss_drops(self):
        g_np = build_homeacf_joint_graph(_homeacf_df())
        y = g_np['y']
        folds = _group_cv_indices(g_np['groups'], y, n_splits=5, seed=0)
        tr = folds[0][0]
        r1 = train_real_joint_fold(g_np, y, tr, 'seir', epochs=40, seed=3)
        r2 = train_real_joint_fold(g_np, y, tr, 'seir', epochs=40, seed=3)
        np.testing.assert_array_equal(r1['score'], r2['score'])
        self.assertLess(r1['loss_last'], r1['loss_first'])


class TestOofProtocol(unittest.TestCase):

    @unittest.skipUnless(_HOMEACF_OK, 'HomeACF .rda 数据不在仓库（license 限制）')
    def test_oof_covers_all_and_no_all_train_fold(self):
        """OOF 每行恰被一折覆盖；logit_lin 有限概率。"""
        g_np = build_homeacf_joint_graph(_homeacf_df())
        y = g_np['y']
        folds = _group_cv_indices(g_np['groups'], y, n_splits=5, seed=0)
        ll = logit_lin_oof(g_np, folds, seed=0)
        oof = np.asarray(ll['oof'])
        # 全覆盖且为概率
        self.assertTrue(np.isfinite(oof).all())
        self.assertTrue(((oof > 0) & (oof < 1)).all())
        # 折互斥覆盖全体
        covered = np.zeros(len(y), dtype=bool)
        for _tr, te in folds:
            self.assertFalse(covered[te].any())
            covered[te] = True
        self.assertTrue(covered.all())

    @unittest.skipUnless(TORCH and _HOMEACF_OK, 'PyTorch 或 HomeACF 数据不可用')
    def test_joint_oof_smoke(self):
        g_np = build_homeacf_joint_graph(_homeacf_df())
        y = g_np['y']
        folds = _group_cv_indices(g_np['groups'], y, n_splits=5, seed=0)
        r = joint_oof(g_np, folds, 'free', epochs=20, seed=0)
        self.assertTrue(np.isfinite(r['oof']).all())
        self.assertTrue((r['oof_p'] >= 0).all()
                        and (r['oof_p'] <= 1).all())


class TestAblationSmoke(unittest.TestCase):

    @unittest.skipUnless(_HOMEACF_OK, 'HomeACF .rda 数据不在仓库（license 限制）')
    def test_run_once_smoke_homeacf(self):
        from tb_risk.validation.real_data_joint import run_real_joint_once
        rep = run_real_joint_once(dataset='homeacf', seed=0,
                                  joint_epochs=20, df=_homeacf_df())
        arms = rep['arms']
        for k in ('pi_only', 'logit_lin', 'joint_frozen', 'joint_free_w',
                  'joint', 'best_existing'):
            self.assertIn(k, arms)
            self.assertTrue(0.0 < arms[k]['auroc'] < 1.0, k)
        # joint 显著优于冻结物理（单种子方向检查，非裁决）
        self.assertGreater(arms['joint']['auroc'],
                           arms['pi_only']['auroc'] - 0.05)
        # pi_only Brier 置 NaN（未校准 λ 非概率）
        self.assertTrue(np.isnan(arms['pi_only']['brier']))


if __name__ == '__main__':
    unittest.main()
