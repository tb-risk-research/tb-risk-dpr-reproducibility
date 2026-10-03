#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4a SEIR 参数联合学习测试：可微 RK4 镜像 / 窗口化 λ / 学习器 / 五臂消融。"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DESKTOP = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, DESKTOP)

from tb_risk.validation.combined_network import (  # noqa: E402
    COMBINED_SEIR_SPEC, community_seir_window_forces,
    build_combined_network,
)
from tb_risk.validation.pi_network import _seir_rk4_numpy  # noqa: E402
from tb_risk.validation.temporal_network import DECAY_WEIGHTS  # noqa: E402


# ==============================================================================
# 可微 RK4 内核
# ==============================================================================

class TestTorchRK4Mirror:
    """torch RK4 与 numpy 镜像逐位一致 + 梯度流 + 默认参数重构。"""

    def test_mirror_numpy_bitwise(self):
        """默认与抖动参数下 torch/numpy 轨迹 < 1e-10。"""
        torch = pytest.importorskip('torch')
        from tb_risk.validation.seir_joint import _seir_rk4_torch
        spec = COMBINED_SEIR_SPEC
        for beta, i0 in [(spec['beta'], spec['I0']),
                         (spec['beta'] * 1.18, spec['I0'] * 0.85)]:
            np_traj = _seir_rk4_numpy(
                beta, spec['sigma'], spec['gamma'], spec['n_steps'],
                spec['dt'], spec['S0'], spec['E0'], i0, spec['R0'])
            t_beta = torch.tensor(beta, dtype=torch.float64)
            t_sigma = torch.tensor(spec['sigma'], dtype=torch.float64)
            t_gamma = torch.tensor(spec['gamma'], dtype=torch.float64)
            t_traj = _seir_rk4_torch(
                t_beta, t_sigma, t_gamma, spec['n_steps'], spec['dt'],
                spec['S0'], spec['E0'], i0, spec['R0'])
            assert np.abs(np_traj - t_traj.numpy()).max() < 1e-10

    def test_gradient_flows_through_integration(self):
        """∂(末端状态)/∂β 和 ∂/∂I0 有限且非零——RK4 计算图完整反传。

        I0 经初始状态进入（torch.tensor 列表构造会静默 detach——
        历史缺陷的回归锚点）。
        """
        torch = pytest.importorskip('torch')
        from tb_risk.validation.seir_joint import _seir_rk4_torch
        spec = COMBINED_SEIR_SPEC
        sigma = torch.tensor(spec['sigma'], dtype=torch.float64)
        gamma = torch.tensor(spec['gamma'], dtype=torch.float64)
        beta = torch.tensor(spec['beta'], dtype=torch.float64,
                            requires_grad=True)
        i0 = torch.tensor(spec['I0'], dtype=torch.float64,
                          requires_grad=True)
        traj = _seir_rk4_torch(beta, sigma, gamma, spec['n_steps'],
                               spec['dt'], spec['S0'], spec['E0'],
                               i0, spec['R0'])
        traj[-1, 2].backward()
        assert beta.grad is not None and torch.isfinite(beta.grad)
        assert float(beta.grad.abs()) > 0.0
        assert i0.grad is not None and torch.isfinite(i0.grad)
        assert float(i0.grad.abs()) > 0.0

    def test_default_params_reconstruct_default_windows(self):
        """δ=0 重构的 Λ̂(t_w) = 默认参数窗值（DGP lam_w_default）。"""
        torch = pytest.importorskip('torch')
        from tb_risk.validation.seir_joint import (
            _window_forces_from_params)
        forces = community_seir_window_forces(np.random.RandomState(7))
        lam_w = _window_forces_from_params(
            torch.zeros((), dtype=torch.float64),
            torch.zeros((), dtype=torch.float64))
        assert np.allclose(lam_w.detach().numpy(),
                           forces['lam_w_default'], atol=1e-10)


# ==============================================================================
# 窗口化物理感染力
# ==============================================================================

class TestWindowedLambda:
    """windowed_lambda 手算核对 + oracle 排序优势 + k 校准。"""

    def test_matches_hand_computation(self):
        from tb_risk.validation.seir_joint import windowed_lambda
        net = build_combined_network(n_contacts=60, random_state=11)
        lam_w = [0.30, 0.28, 0.22, 0.14, 0.06]
        M = net['M']
        host = np.array([n['host_multiplier'] for n in net['nodes'][M:]])
        expect = host * net['corr'] * (
            net['exposure_by_window'] @ (np.array(lam_w) * DECAY_WEIGHTS))
        assert np.allclose(windowed_lambda(net, lam_w), expect, atol=1e-12)

    def test_oracle_window_ranks_better_than_default(self):
        """真值窗力的 λ 与 ν 的相关高于平坦 λ_default（失配可回收）。"""
        from tb_risk.validation.seir_joint import windowed_lambda
        net = build_combined_network(n_contacts=200, random_state=23)
        lam_w_truth = net['seir_info']['lam_w_truth']
        r_ow = np.corrcoef(windowed_lambda(net, lam_w_truth),
                           net['nu'])[0, 1]
        r_def = np.corrcoef(net['lam'], net['nu'])[0, 1]
        assert r_ow > r_def

    def test_calibrate_k_generic(self):
        """二分恢复训练阳性率；全零 λ 退化保护返回 1。"""
        from tb_risk.validation.seir_joint import _calibrate_k_generic
        rng = np.random.RandomState(5)
        lam = rng.gamma(2.0, 0.5, size=100)
        y = (rng.random(100) < 0.25).astype(float)
        tr = np.arange(0, 100, 2)
        k = _calibrate_k_generic(lam, y, tr)
        assert abs((1.0 - np.exp(-k * lam[tr])).mean()
                   - y[tr].mean()) < 1e-3
        assert _calibrate_k_generic(np.zeros(100), y, tr) == 1.0


# ==============================================================================
# 学习器
# ==============================================================================

class TestLearner:
    """确定性 / 收敛 / 输出契约 / 尺度校准。"""

    @pytest.fixture()
    def small_net(self):
        pytest.importorskip('torch')
        net = build_combined_network(n_contacts=120, random_state=31)
        labels = np.asarray(net['labels'], dtype=float)
        tr = np.arange(0, 120, 2)
        return net, labels, tr

    def test_deterministic(self, small_net):
        from tb_risk.validation.seir_joint import learn_windowed_seir
        net, y, tr = small_net
        a = learn_windowed_seir(net, y, tr, mode='seir', epochs=80, seed=3)
        b = learn_windowed_seir(net, y, tr, mode='seir', epochs=80, seed=3)
        assert np.array_equal(a['score'], b['score'])
        assert a['k_hat'] == b['k_hat']

    def test_loss_decreases_both_modes(self, small_net):
        """seir / free 两模式 BCE 均下降（学习发生）。"""
        from tb_risk.validation.seir_joint import learn_windowed_seir
        net, y, tr = small_net
        for mode in ('seir', 'free'):
            r = learn_windowed_seir(net, y, tr, mode=mode, epochs=150,
                                    seed=3)
            assert r['loss_last'] < r['loss_first'], mode

    def test_output_contract(self, small_net):
        from tb_risk.validation.seir_joint import learn_windowed_seir
        net, y, tr = small_net
        r_seir = learn_windowed_seir(net, y, tr, mode='seir', epochs=80,
                                     seed=3)
        assert r_seir['mode'] == 'seir'
        assert 'beta_hat' in r_seir and 'I0_hat' in r_seir
        assert r_seir['beta_hat'] > 0.0 and r_seir['I0_hat'] > 0.0
        assert len(r_seir['lam_w_hat']) == 5
        assert np.isfinite(r_seir['score']).all()
        assert ((r_seir['p'] >= 0.0) & (r_seir['p'] <= 1.0)).all()
        r_free = learn_windowed_seir(net, y, tr, mode='free', epochs=80,
                                     seed=3)
        assert 'beta_hat' not in r_free
        assert np.isfinite(r_free['score']).all()
        with pytest.raises(ValueError):
            learn_windowed_seir(net, y, tr, mode='bogus', epochs=5)

    def test_k_calibration_holds(self, small_net):
        """k 交替校准：训练半区 p 均值 ≈ 训练阳性率（口径成立）。"""
        from tb_risk.validation.seir_joint import learn_windowed_seir
        net, y, tr = small_net
        r = learn_windowed_seir(net, y, tr, mode='seir', epochs=120,
                                seed=3)
        assert abs(float(r['p'][tr].mean()) - float(y[tr].mean())) < 0.02


# ==============================================================================
# 五臂消融协议
# ==============================================================================

class TestSEIRJointAblation:
    """协议结构：臂齐全 / 阶梯恒等式 / 诊断与 DeLong 键。"""

    def test_run_once_structure(self):
        pytest.importorskip('torch')
        from tb_risk.validation.seir_joint import (
            run_seir_joint_ablation_once, ARMS)
        rep = run_seir_joint_ablation_once(n_contacts=120, seed=41,
                                           joint_epochs=100)
        assert set(rep['arms']) == set(ARMS)
        for name, arm in rep['arms'].items():
            assert 0.0 <= arm['auroc'] <= 1.0
            assert len(arm['test_scores']) == rep['n_test']
            assert arm['brier'] >= 0.0
        # 阶梯恒等式
        lad, a = rep['ladder'], {k: v['auroc']
                                 for k, v in rep['arms'].items()}
        assert abs(lad['joint_minus_default']
                   - (a['seir_joint'] - a['seir_default'])) < 1e-12
        assert abs(lad['free_minus_joint']
                   - (a['free_windows'] - a['seir_joint'])) < 1e-12
        # oracle_window 是 oracle_nu 的下界路径（同测半区可比）
        assert 0.0 <= lad['recovery_ratio'] <= 3.0 or \
            np.isnan(lad['recovery_ratio'])
        # 参数恢复诊断键齐全
        assert set(rep['param_recovery']) >= {
            'beta_rel_err', 'i0_rel_err', 'lam_w_shape_err',
            'free_shape_err', 'joint_loss_drop', 'free_loss_drop'}
        # DeLong 键齐全
        assert set(rep['delong']) == {
            'joint_vs_default', 'free_vs_default', 'free_vs_joint',
            'oracle_window_vs_default'}

    def test_multi_seed_smoke(self):
        pytest.importorskip('torch')
        from tb_risk.validation.seir_joint import (
            run_multi_seed_seir_joint_ablation)
        rep = run_multi_seed_seir_joint_ablation(
            n_contacts=100, n_seeds=2, seed_start=51,
            joint_epochs=80, n_bootstrap=200)
        assert rep['design']['n_seeds'] == 2
        assert len(rep['seeds']) == 2
        for k in ('seir_default', 'seir_joint', 'free_windows',
                  'oracle_window', 'oracle_nu'):
            entry = rep['arm_summary'][k]
            ci = entry['bootstrap_ci_auroc']
            assert ci[0] <= entry['mean_auroc'] <= ci[1] + 1e-9
        assert set(rep['conclusion']) == {
            'joint_learning_helps', 'structure_prior_verdict',
            'recovery_ratio'}
