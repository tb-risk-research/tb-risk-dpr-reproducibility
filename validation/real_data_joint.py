#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实数据端到端联合复验（P6，2026-09-05）：HomeACF / PACTS。

用户问题（P0.1，2026-09-05）："合成 DGP 内 joint +0.074 超越集成——
把 joint 上已有的真实基准跑一遍，看 +0.05~0.07 量级的增益在真实
数据上留存多少。这是它能否作为论文创新点的唯一判据。"

模型（P5 JointMechGNN 的真实数据版，机制结构不变）：

    ν̂_i = Ŝ_i · Ĉ_i · Σ_c Λ̂(w_c)·β̂_{t_c}·d̂_{w_c}·ŝ_{src(c)}

    Ŝ_i  = S_frozen(age,hiv) · exp(a·tanh(host_head(z_i)))   有界修正
    Ĉ_i  = C_frozen(分级暴露) · exp(a·tanh(corr_head(z_i)))  有界修正
    Λ̂_w  = β̂·Î(t_w)（可微 RK4，咳嗽时长窗中点求值；可学 δβ/δI0）
    β̂_t  关系类型权重（child/spouse/sibling/other，初值 1）
    d̂_w  窗权重（初值 1，先验收缩）
    ŝ    指示病例特征线性头 → log 传染性（替换冻结 I_index 公式）

真实数据的机制映射（诚实声明）：
  - 合成世界 Λ(t_w) = 社区疫情轨迹；真实 HomeACF 无社区轨迹观测，
    窗口 = 指示病例咳嗽时长分层（[0,14)/[14,60)/[60,180)/[180,365] 天，
    事件率 0.098→0.140→0.148→0.173 单调升），SEIR 参数化的对象是
    "传染力随病程的剂量-反应曲线"——结构角色（2 参数 ODE 形状约束
    4 窗轨迹）不变，社区疫情解读不直接迁移；
  - 每接触者恰好 1 条 membership 边（户→指示病例星形图），消息和
    退化为单项——joint 在此 = 机制重参数化的可学 λ，图卷积意义
    弱于合成世界（无多源聚合）；PACTS 无咳嗽时长 → 单窗退化
    （SEIR 尺度被 k 吸收，joint 即自由单窗，frozen/free 臂别名）；
  - ŝ 用特征头而非逐源自由参数：户分组 CV 测试折的源未在训练
    出现，自由参数不可迁移，特征头是唯一合法读出。

归因阶梯（P0.2 真实数据部分，同一信息集分解）：
    pi_only      冻结物理 λ（文献先验，不学习）
    logit_lin    同信息加性 logistic（无物理形式对照）
    joint_frozen SEIR 冻结（δβ/δI0=0），类型/衰减/三头可学
    joint_free_w 自由 Λ_w（无 ODE 约束），其余同 joint
    joint        全可学（SEIR 2 + 类型 4 + 衰减 4 + 三头）
    best_existing  逐种子最优既有特征臂（RF/LGBM，表征学习）
  阶梯：
    joint − pi_only         机制学习总增益（vs 冻结物理）
    joint − joint_frozen    学 ODE 参数的价值
    joint − joint_free_w    ODE 形状约束的价值（负 = 自由更好）
    joint − logit_lin       乘性物理形式的价值
    joint − best_existing   vs 最强既有臂（保守口径）

信息对齐（防"joint 输少"伪影）：host/corr/src 特征块覆盖 ind/
index/exposure 臂的列（host 含 site 等混杂修正、corr 含户规模稀释
效应 hh_n——暴露预算在户内接触者间分配的机制近似）。

协议（与 real_data_infection 完全同口径）：
    StratifiedGroupKFold(5) by household × 20 种子（seed 0-19）；
    池化 OOF；户级 cluster bootstrap CI（末种子）；逐种子 DeLong；
    头输入标准化只用训练折统计（源特征按训练折出现的源拟合——
    无泄漏）；k 交替校准（每 50 epoch，训练折二分，不进梯度）；
    机制 log 参数先验收缩到初值。

诚实边界：
  - HomeACF 终点 TST≥10mm = LTBI 感染非发病，BCG/环境分枝杆菌
    致假阳——方向性外部验证口径（同 real_data_infection）；
  - 冻结 λ 基线 AUROC 0.529（近随机）：文献先验在该队列弱对齐，
    joint 的可学成分（三头 + 类型 + 窗）承载几乎全部信号——
    "物理结构 vs 可学参数"的归因在本数据上偏保守（先验越差，
    结构起点越低）；
  - PACTS 主终点被基线症状饱和（ind:RF 0.9928）且无时间维——
    joint 在该数据不可判别检验，仅作结构退化记录；
  - 结论口径 = 单队列外部复验，非荟萃。
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover —— 环境守卫
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from .layer_ablation import delong_paired_test
from .real_data_pi import (
    _cluster_bootstrap_delta,
    _group_cv_indices,
    _pr_auc,
    _recall_at_budget,
)
from .seir_joint import _calibrate_k_generic, _seir_rk4_torch
from .threshold_spec import compute_auc

# ==============================================================================
# 真实窗口/类型规范
# ==============================================================================

# HomeACF 咳嗽时长窗（天，左闭右开；事件率梯度见模块 docstring）
HOMEACF_WINDOW_EDGES_DAYS = (0, 14, 60, 180, 366)
# 窗中点（周）：剂量-反应曲线的 SEIR 求值时点
HOMEACF_WINDOW_MID_WEEKS = (1.0, 5.3, 17.1, 39.1)
HOMEACF_TYPES = ('child', 'spouse', 'sibling', 'other')

# SEIR 文献参数（TB：潜伏 8 周、传染期 10 周；β/I0 与合成世界同
# 中性起点——绝对尺度被 k 吸收，只有形状进入判别。dt=1.0：窗时点
# 间隔 4-22 周 ≫ dt，RK4 精度充足且比 dt=0.25 快 4 倍）
REAL_SEIR_SPEC = {
    'beta': 0.25, 'sigma': 0.125, 'gamma': 0.1,
    'S0': 0.98, 'E0': 0.015, 'I0': 0.002, 'R0': 0.003,
    'dt': 1.0,
}

BOUND_SCALE = 1.0   # S/C 修正界：exp(±1) ∈ [0.37, 2.72]


def _real_window_forces_torch(log_d_beta, log_d_i0, mid_weeks):
    """SEIR RK4 → 窗中点感染力 Λ̂(t_w) = β̂·Î(t_w)（可微，任意时点）。"""
    spec = REAL_SEIR_SPEC
    beta = spec['beta'] * torch.exp(log_d_beta)
    i0 = spec['I0'] * torch.exp(log_d_i0)
    t_max = float(max(mid_weeks))
    n_steps = int(np.ceil(t_max / spec['dt']))
    traj = _seir_rk4_torch(beta, spec['sigma'], spec['gamma'],
                           n_steps, spec['dt'],
                           spec['S0'], spec['E0'], i0, spec['R0'])
    idx = torch.as_tensor(
        [int(round(t / spec['dt'])) for t in mid_weeks], dtype=torch.long)
    return beta * traj[idx, 2]


def _default_window_forces(mid_weeks):
    """默认参数窗力（frozen 臂 / free 臂初值，numpy）。"""
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError('PyTorch 不可用')
    with torch.no_grad():
        return _real_window_forces_torch(
            torch.zeros((), dtype=torch.float64),
            torch.zeros((), dtype=torch.float64),
            mid_weeks).numpy().astype(float)


# ==============================================================================
# 真实图构造（HomeACF / PACTS → joint 输入）
# ==============================================================================

def _first_row_per_source(groups):
    """每源（户）首行下标（源特征取首行——同户同值）。"""
    _, first_idx, _ = np.unique(groups, return_index=True,
                                return_inverse=True)
    return first_idx


def _build_graph_common(df, y, groups, mem_type, mem_win, src_cols,
                        host_cols, corr_cols, host_frozen, corr_frozen,
                        dataset, mid_weeks, type_names):
    uniq, src_id = np.unique(groups, return_inverse=True)
    first_idx = _first_row_per_source(groups)
    src_feats = np.nan_to_num(
        df[list(src_cols)].to_numpy(dtype=float)[first_idx])
    return {
        'dataset': dataset,
        'y': np.asarray(y, dtype=int),
        'groups': np.asarray(groups),
        'mem_contact': np.arange(len(df), dtype=np.int64),
        'mem_src': src_id.astype(np.int64),
        'mem_type': np.asarray(mem_type, dtype=np.int64),
        'mem_win': np.asarray(mem_win, dtype=np.int64),
        'src_feats': src_feats,
        'host_feats': df[list(host_cols)].to_numpy(dtype=float),
        'host_frozen': np.asarray(host_frozen, dtype=float),
        'corr_feats': df[list(corr_cols)].to_numpy(dtype=float),
        'corr_frozen': np.asarray(corr_frozen, dtype=float),
        'mid_weeks': tuple(mid_weeks),
        'type_names': list(type_names),
    }


def build_homeacf_joint_graph(df):
    """HomeACF → joint 图输入（每接触者 1 条户边，4 咳嗽时长窗）。"""
    cd = df['idx_coughdays'].to_numpy(dtype=float)
    mem_win = np.digitize(cd, HOMEACF_WINDOW_EDGES_DAYS[1:-1])

    rel = df[['rel_child', 'rel_spouse', 'rel_sibling']].to_numpy(dtype=float)
    # other = 非 child/spouse/sibling（parent 在本队列=0）
    mem_type = np.where(rel[:, 0] == 1, 0,
                np.where(rel[:, 1] == 1, 1,
                np.where(rel[:, 2] == 1, 2, 3)))

    # 冻结物理乘子（real_data_infection.build_lambda_infection 同式）
    host_frozen = (1.0 + 1.8 * df['age_lt5'] + 1.6 * df['age_ge45']) \
        * (1.3 ** df['hiv_pos_h'])
    corr_frozen = (0.5 * df['ts_low'] + 1.0 * df['ts_mid']
                   + 1.5 * df['ts_high']) \
        * (1.2 ** df['share_bedroom']) * (1.5 ** df['sleep_same_bed'])

    return _build_graph_common(
        df, df['tst_pos10'], df['record_id'], mem_type, mem_win,
        src_cols=('idx_smear_pos', 'idx_smear_known', 'idx_hiv_pos',
                  'idx_age', 'idx_sex_m', 'idx_dead'),
        host_cols=('age_lt5', 'age_ge45', 'hiv_pos_h', 'hiv_unknown_h',
                   'bmi_h', 'smoke_ever_h', 'diabetes_h_f', 'contact_sex_m',
                   'contact_age', 'site_capricorn'),
        corr_cols=('ts_low', 'ts_mid', 'ts_high', 'share_bedroom',
                   'sleep_same_bed', 'airspace_shared', 'hh_n_contacts'),
        host_frozen=host_frozen, corr_frozen=corr_frozen,
        dataset='homeacf', mid_weeks=HOMEACF_WINDOW_MID_WEEKS,
        type_names=HOMEACF_TYPES)


def build_pacts_joint_graph(df):
    """PACTS → joint 图输入（单窗退化：无咳嗽时长维度）。"""
    rel = df[['rel_1', 'rel_2', 'rel_3', 'rel_4', 'rel_5']].to_numpy(
        dtype=float)
    has_rel = rel.sum(axis=1) > 0
    mem_type = np.where(has_rel, np.argmax(rel, axis=1), 4)

    host_frozen = 1.0 + 1.8 * df['age_lt5'] + 1.6 * df['age_ge45']
    corr_frozen = 1.0 * df['lives_in_household'] \
        + 0.5 * (1.0 - df['lives_in_household'])

    return _build_graph_common(
        df, df['sx_3m'], df['id'], mem_type, np.zeros(len(df)),
        src_cols=('idx_smear_pos', 'idx_hiv_pos', 'idx_hiv_unknown',
                  'idx_age', 'idx_sex_m', 'hh_size_f'),
        host_cols=('age_lt5', 'age_ge45', 'contact_sex_m',
                   'educ_no_school', 'educ_primary', 'prior_tb_suspected',
                   'arm_intervention'),
        corr_cols=('lives_in_household', 'nights_shared', 'hh_n_contacts'),
        host_frozen=host_frozen, corr_frozen=corr_frozen,
        dataset='pacts', mid_weeks=(1.0,),
        type_names=('rel_1', 'rel_2', 'rel_3', 'rel_4', 'other'))


# ==============================================================================
# 模型
# ==============================================================================

if TORCH_AVAILABLE:

    class RealJointMechGNN(nn.Module):
        """真实数据端到端联合模型（P5 JointMechGNN 的星形图版）。

        Args:
            host_dim/corr_dim/src_dim: 三头输入维度
            n_types/n_windows: 类型数 / 窗数
            mid_weeks: 窗中点（周，SEIR 求值时点）
            mode: 'seir'（可学 δβ/δI0）/ 'frozen'（默认参数 Λ）/
                  'free'（自由 log Λ_w，无 ODE 约束，初值=默认窗力）
            bound_scale: S/C 有界修正尺度 a（修正 ∈ [e^−a, e^a]）
        """

        def __init__(self, host_dim, corr_dim, src_dim, n_types,
                     n_windows, mid_weeks, mode='seir',
                     bound_scale=BOUND_SCALE):
            super().__init__()
            if mode not in ('seir', 'frozen', 'free'):
                raise ValueError("mode ∈ {'seir','frozen','free'}")
            self.mode = mode
            self.n_windows = int(n_windows)
            self.mid_weeks = tuple(float(t) for t in mid_weeks)
            self.bound_scale = float(bound_scale)
            self.log_beta_t = nn.Parameter(
                torch.zeros(int(n_types), dtype=torch.float64))
            self.log_decay = nn.Parameter(
                torch.zeros(int(n_windows), dtype=torch.float64))
            lam0 = torch.tensor(
                _default_window_forces(self.mid_weeks),
                dtype=torch.float64)
            if mode == 'seir':
                self.log_d_beta = nn.Parameter(
                    torch.zeros((), dtype=torch.float64))
                self.log_d_i0 = nn.Parameter(
                    torch.zeros((), dtype=torch.float64))
                self.register_buffer('lam_w_anchor', lam0)
            elif mode == 'free':
                self.log_lam_w = nn.Parameter(torch.log(lam0).clone())
                self.register_buffer('lam_w_anchor', lam0)
            else:
                self.register_buffer('lam_w_fixed', lam0)
                self.register_buffer('lam_w_anchor', lam0)
            self.host_head = nn.Linear(host_dim, 1).double()
            self.corr_head = nn.Linear(corr_dim, 1).double()
            self.src_head = nn.Linear(src_dim, 1).double()
            # 零初始化：初始 Ŝ=S_frozen、Ĉ=C_frozen、ŝ=1（闭式可验）
            for head in (self.host_head, self.corr_head, self.src_head):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

        def lam_w(self):
            """窗感染力 Λ̂_w [W]（mode 决定参数化）。"""
            if self.mode == 'seir':
                return _real_window_forces_torch(
                    self.log_d_beta, self.log_d_i0, self.mid_weeks)
            if self.mode == 'free':
                return torch.exp(self.log_lam_w)
            return self.lam_w_fixed

        def forward(self, g, k):
            """g: _to_torch_graph 输出（头输入已训练折标准化）。"""
            lam_w = self.lam_w()
            beta_t = torch.exp(self.log_beta_t)
            decay = torch.exp(self.log_decay)
            s_src = torch.exp(self.src_head(g['src_feats']).squeeze(-1))
            msg = (lam_w[g['mem_win']] * beta_t[g['mem_type']]
                   * decay[g['mem_win']] * s_src[g['mem_src']])
            mech = torch.zeros(g['host'].shape[0], dtype=torch.float64)
            mech.index_add_(0, g['mem_contact'], msg)
            s_hat = g['host'] * torch.exp(self.bound_scale * torch.tanh(
                self.host_head(g['host_feats']).squeeze(-1)))
            c_hat = g['corr'] * torch.exp(self.bound_scale * torch.tanh(
                self.corr_head(g['corr_feats']).squeeze(-1)))
            nu = s_hat * c_hat * mech
            z = torch.clamp(k * nu, min=1e-9)
            logit = z + torch.log1p(-torch.exp(-z))
            return logit, nu


# ==============================================================================
# 训练器（OOF 协议：训练折标准化 + k 交替 + 先验收缩）
# ==============================================================================

def _fit_scaler(x, fit_idx):
    """fit_idx 行上估计均值/方差并标准化全体（std=0 列防除零）。"""
    mu = x[fit_idx].mean(axis=0)
    sd = x[fit_idx].std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (x - mu) / sd


def _to_torch_graph(g_np, train_idx):
    """numpy 图 → torch 张量（标准化统计只用训练折信息）。

    源特征：在训练折接触者所属源上拟合（测试折源特征只做变换）；
    host/corr：在训练折接触者行上拟合。
    """
    train_edges = np.isin(g_np['mem_contact'], train_idx)
    train_srcs = np.unique(g_np['mem_src'][train_edges])
    src_std = _fit_scaler(g_np['src_feats'], train_srcs)
    return {
        'mem_contact': torch.as_tensor(g_np['mem_contact'],
                                       dtype=torch.long),
        'mem_src': torch.as_tensor(g_np['mem_src'], dtype=torch.long),
        'mem_type': torch.as_tensor(g_np['mem_type'], dtype=torch.long),
        'mem_win': torch.as_tensor(g_np['mem_win'], dtype=torch.long),
        'src_feats': torch.as_tensor(
            src_std[g_np['mem_src']], dtype=torch.float64),
        'host_feats': torch.as_tensor(
            _fit_scaler(g_np['host_feats'], train_idx),
            dtype=torch.float64),
        'corr_feats': torch.as_tensor(
            _fit_scaler(g_np['corr_feats'], train_idx),
            dtype=torch.float64),
        'host': torch.as_tensor(g_np['host_frozen'], dtype=torch.float64),
        'corr': torch.as_tensor(g_np['corr_frozen'], dtype=torch.float64),
    }


def train_real_joint_fold(g_np, y, train_idx, mode, epochs=400, lr=0.03,
                          k_refit_every=50, prior_lambda=1e-3, seed=0):
    """单折训练 → 全体接触者分数/概率（OOF 由调用方按折拼装）。"""
    if not TORCH_AVAILABLE:
        raise RuntimeError('PyTorch 不可用')
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    g = _to_torch_graph(g_np, train_idx)
    y_t = torch.as_tensor(np.asarray(y, dtype=float), dtype=torch.float64)
    tr = torch.as_tensor(np.asarray(train_idx, dtype=np.int64))

    model = RealJointMechGNN(
        host_dim=g['host_feats'].shape[1],
        corr_dim=g['corr_feats'].shape[1],
        src_dim=g['src_feats'].shape[1],
        n_types=int(g_np['mem_type'].max()) + 1,
        n_windows=int(g_np['mem_win'].max()) + 1,
        mid_weeks=g_np['mid_weeks'], mode=mode)

    mech_params = [model.log_beta_t, model.log_decay]
    if mode == 'seir':
        mech_params += [model.log_d_beta, model.log_d_i0]
    elif mode == 'free':
        mech_params += [model.log_lam_w]
    heads = list(model.host_head.parameters()) \
        + list(model.corr_head.parameters()) \
        + list(model.src_head.parameters())
    opt = torch.optim.Adam(
        [{'params': mech_params},
         {'params': heads, 'weight_decay': 3e-3}], lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()

    def _prior():
        """机制 log 参数先验收缩到初值（free 臂锚定默认窗力）。"""
        if prior_lambda <= 0:
            return torch.zeros((), dtype=torch.float64)
        terms = [model.log_beta_t.pow(2).sum(),
                 model.log_decay.pow(2).sum()]
        if mode == 'seir':
            terms += [model.log_d_beta.pow(2), model.log_d_i0.pow(2)]
        elif mode == 'free':
            anchor = torch.log(model.lam_w_anchor)
            terms.append((model.log_lam_w - anchor).pow(2).sum())
        return prior_lambda * sum(terms)

    # 初始 k（默认参数 ν̂ 上训练折校准）
    with torch.no_grad():
        _, nu0 = model(g, torch.tensor(1.0, dtype=torch.float64))
    k = _calibrate_k_generic(nu0.numpy(), y, train_idx)
    k_t = torch.tensor(float(k), dtype=torch.float64)

    loss_first = loss_last = None
    for epoch in range(epochs):
        opt.zero_grad()
        logit, _ = model(g, k_t)
        loss = bce(logit[tr], y_t[tr]) + _prior()
        if loss_first is None:
            loss_first = float(loss.detach())
        loss.backward()
        opt.step()
        loss_last = float(loss.detach())
        if (epoch + 1) % k_refit_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                _, nu_cur = model(g, torch.tensor(1.0, dtype=torch.float64))
            k = _calibrate_k_generic(nu_cur.numpy(), y, train_idx)
            k_t = torch.tensor(float(k), dtype=torch.float64)

    with torch.no_grad():
        _, nu_hat = model(g, k_t)
        k_hat = float(_calibrate_k_generic(nu_hat.numpy(), y, train_idx))
        z = np.clip(k_hat * nu_hat.numpy(), 1e-9, None)
    return {
        'score': k_hat * nu_hat.numpy(),
        'p': 1.0 - np.exp(-z),
        'k_hat': k_hat,
        'loss_first': loss_first,
        'loss_last': loss_last,
        'lam_w_hat': model.lam_w().detach().numpy().tolist(),
    }


def joint_oof(g_np, folds, mode, epochs=400, lr=0.03, seed=0):
    """逐折训练 → 池化 OOF 分数/概率（与特征臂同折同种子）。"""
    y = g_np['y']
    oof_score = np.zeros(len(y))
    oof_p = np.zeros(len(y))
    for tr, te in folds:
        r = train_real_joint_fold(g_np, y, tr, mode, epochs=epochs,
                                  lr=lr, seed=seed)
        oof_score[te] = r['score'][te]
        oof_p[te] = r['p'][te]
    return {'oof': oof_score, 'oof_p': oof_p}


def logit_lin_oof(g_np, folds, seed=0):
    """同信息加性 logistic 对照（sklearn，无物理形式）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y = g_np['y']
    n_types = int(g_np['mem_type'].max()) + 1
    n_wins = int(g_np['mem_win'].max()) + 1
    X = np.hstack([
        g_np['host_feats'],
        g_np['corr_feats'],
        g_np['src_feats'][g_np['mem_src']],
        np.eye(n_types)[g_np['mem_type']],
        np.eye(n_wins)[g_np['mem_win']],
    ])
    oof = np.zeros(len(y))
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000, C=1.0,
                               random_state=seed).fit(
            sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return {'oof': oof, 'oof_p': oof}


# ==============================================================================
# 消融协议
# ==============================================================================

JOINT_ARMS = ('pi_only', 'logit_lin', 'joint_frozen', 'joint_free_w',
              'joint')

_CONTRASTS = (
    ('joint', 'pi_only'),
    ('joint', 'joint_frozen'),
    ('joint', 'joint_free_w'),
    ('joint', 'logit_lin'),
    ('joint_frozen', 'pi_only'),
    ('logit_lin', 'pi_only'),
)


def run_real_joint_once(dataset='homeacf', seed=0, n_splits=5,
                        joint_epochs=400, lr=0.03, df=None):
    """单种子：既有特征臂（复用真实基准）+ joint 四臂，同折配对。"""
    if dataset == 'homeacf':
        from .real_data_infection import (
            run_infection_ablation_once as _run_feat)
        if df is None:
            from .real_data_infection import load_homeacf_contacts
            df = load_homeacf_contacts()
        g_np = build_homeacf_joint_graph(df)
        modes = {'joint': 'seir', 'joint_frozen': 'frozen',
                 'joint_free_w': 'free'}
    elif dataset == 'pacts':
        from .real_data_pi import run_real_data_ablation_once as _run_feat
        if df is None:
            from .real_data_pi import load_pacts_contacts
            df = load_pacts_contacts()
        g_np = build_pacts_joint_graph(df)
        # 单窗退化：Λ 尺度被 k 吸收，frozen/free 与 joint 同模型
        modes = {'joint': 'free'}
    else:
        raise ValueError("dataset ∈ {'homeacf','pacts'}")

    rep = _run_feat(seed=seed, n_splits=n_splits, df=df)
    y = rep['y']
    groups = rep['groups']
    folds = _group_cv_indices(groups, y, n_splits=n_splits, seed=seed)

    arms = {'pi_only': {'oof': rep['oof']['pi_only'],
                        'oof_p': rep['oof']['pi_only']}}
    # pi_only 的 oof 是未校准 λ（非概率）——Brier 无意义，置 NaN
    arms['pi_only']['brier_invalid'] = True
    arms['logit_lin'] = logit_lin_oof(g_np, folds, seed=seed)
    for arm, mode in modes.items():
        arms[arm] = joint_oof(g_np, folds, mode, epochs=joint_epochs,
                              lr=lr, seed=seed)
    if dataset == 'pacts':
        arms['joint_frozen'] = arms['joint']
        arms['joint_free_w'] = arms['joint']

    existing = {k: v for k, v in rep['arms'].items() if k != 'pi_only'}
    best_existing = max(existing, key=lambda k: existing[k]['auroc'])

    out = {'seed': seed, 'y': y, 'groups': groups, 'arms': {},
           'best_existing_arm': best_existing,
           'existing_arms': existing}
    for name, a in arms.items():
        oof, oof_p = np.asarray(a['oof'], dtype=float), \
            np.asarray(a['oof_p'], dtype=float)
        brier = (float('nan') if a.get('brier_invalid')
                 else float(np.mean((oof_p - y) ** 2)))
        out['arms'][name] = {
            'auroc': float(compute_auc(oof, y)),
            'pr_auc': _pr_auc(y, oof),
            'recall_at_budget': _recall_at_budget(oof, y),
            'brier': brier,
        }
    # best_existing 概率口径 Brier（RF/LGBM oof 即概率）
    be_oof = np.asarray(rep['oof'][best_existing], dtype=float)
    out['arms']['best_existing'] = {
        'auroc': float(existing[best_existing]['auroc']),
        'pr_auc': existing[best_existing]['pr_auc'],
        'recall_at_budget': existing[best_existing]['recall_at_budget'],
        'brier': float(np.mean((be_oof - y) ** 2)),
        'arm_name': best_existing,
    }
    out['oof'] = {name: np.asarray(a['oof'], dtype=float).tolist()
                  for name, a in arms.items()}
    out['oof']['best_existing'] = be_oof.tolist()
    out['delong'] = {
        f'{a}_vs_{b}': delong_paired_test(
            y, np.asarray(arms[a]['oof']), np.asarray(arms[b]['oof']))
        for a, b in _CONTRASTS}
    out['delong']['joint_vs_best_existing'] = delong_paired_test(
        y, np.asarray(arms['joint']['oof']), be_oof)
    out['design'] = {
        'dataset': dataset, 'n': int(len(y)),
        'n_events': int(np.sum(y)),
        'n_households': int(len(np.unique(groups))),
        'n_splits': n_splits,
        'joint_epochs': joint_epochs,
        'modes': {k: v for k, v in modes.items()},
    }
    return out


def run_real_joint_multi_seed(dataset='homeacf', n_seeds=20, seed_start=0,
                              n_splits=5, joint_epochs=400, lr=0.03,
                              n_bootstrap=2000, df=None):
    """多种子 + 户级 cluster bootstrap + DeLong 汇总。"""
    per_seed = []
    for s in range(seed_start, seed_start + n_seeds):
        rep = run_real_joint_once(dataset=dataset, seed=s,
                                  n_splits=n_splits, df=df,
                                  joint_epochs=joint_epochs, lr=lr)
        per_seed.append(rep)
        print('  seed %d: joint AUROC=%.4f | best_existing %s %.4f'
              % (s, rep['arms']['joint']['auroc'],
                 rep['best_existing_arm'],
                 rep['arms']['best_existing']['auroc']), flush=True)

    arm_names = list(per_seed[0]['arms'].keys())
    arm_summary = {}
    for name in arm_names:
        vals = {m: [r['arms'][name][m] for r in per_seed]
                for m in ('auroc', 'pr_auc', 'recall_at_budget', 'brier')}
        arm_summary[name] = {
            # pi_only 的 brier 合法为 NaN（未校准 λ 非概率）——全 NaN
            # 切片 nanmean 会告警，显式落 NaN 保持归档口径
            'mean_' + m: (float('nan') if not np.isfinite(v).any()
                          else float(np.nanmean(v)))
            for m, v in vals.items()}
        arm_summary[name]['sd_auroc'] = float(np.std(vals['auroc']))

    last = per_seed[-1]
    ladder_summary = {}
    for a, b in _CONTRASTS + (('joint', 'best_existing'),):
        oof_a = np.asarray(last['oof'][a], dtype=float)
        oof_b = np.asarray(last['oof'][b], dtype=float)
        ladder_summary[f'{a}_minus_{b}'] = _cluster_bootstrap_delta(
            oof_a, oof_b, last['y'], last['groups'],
            n_bootstrap=n_bootstrap, seed=seed_start)

    delong_summary = {}
    for key in last['delong']:
        ps = [float(r['delong'][key]['p_value']) for r in per_seed]
        delong_summary[key] = {
            'mean_p': float(np.mean(ps)),
            'median_p': float(np.median(ps)),
            'frac_p_lt_0.05': float(np.mean(np.array(ps) < 0.05)),
        }

    best_counts = {}
    for r in per_seed:
        best_counts[r['best_existing_arm']] = \
            best_counts.get(r['best_existing_arm'], 0) + 1

    j = ladder_summary['joint_minus_best_existing']
    return {
        'design': dict(per_seed[0]['design'],
                       n_seeds=n_seeds, seed_start=seed_start,
                       name=f'real_data_joint_{dataset}_v1',
                       ci=f'户级 cluster bootstrap ×{n_bootstrap}（末种子 OOF）',
                       arms=arm_names),
        'arm_summary': arm_summary,
        'ladder_summary': ladder_summary,
        'delong_summary': delong_summary,
        'best_existing_arm_counts': best_counts,
        'seeds': [{'seed': r['seed'], 'arms': r['arms'],
                   'best_existing_arm': r['best_existing_arm']}
                  for r in per_seed],
        'conclusion': {
            'joint_minus_best_existing': {
                k: j[k] for k in ('mean', 'bootstrap_ci',
                                  'ci_excludes_zero')},
            'joint_minus_pi_only': {
                k: ladder_summary['joint_minus_pi_only'][k]
                for k in ('mean', 'bootstrap_ci', 'ci_excludes_zero')},
            'note': '合成 P5 joint−ensemble=+0.0745；本队列留存见 '
                    'ladder_summary（唯一判据口径）',
        },
    }
