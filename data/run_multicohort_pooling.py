#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多队列部分池化评估：cohort 随机截距 + 共享斜率（任务2，S7）

背景：7 个真实队列（nhanes 7080 / treats 4529 / crp 1777 / kenya 63050 /
taiwan 129 / brazil 202 / peru_mdr 3406，合计 80,173 行 3,511 事件）此前
各自独立训练，小队列（Taiwan 129 / Brazil 202）斜率估计方差大。本脚本
用 statsmodels BinomialBayesMixedGLM 部分池化（scoring/ml/partial_pooling.py）
回答：共享斜率跨队列迁移到底值多少？

三臂设计（LOCO = Leave-One-Cohort-Out，防斜率层泄漏——每留出队列用
其余 6 队列独立重训，共 7 次拟合）：
  (a) 逐队列独立 LR 基线：现状对照（各队列本地 CV OOF）。
      treats（14 社区）/ peru_mdr（688 户）用 StratifiedGroupKFold 组感知
      CV + 户级 cluster bootstrap CI（任务1 协议）；无组列队列 StratifiedKFold
      + 个体 bootstrap（每样本一簇，CI 机制统一）。
  (b) LOCO 零样本臂：留出队列以总体截距预测（cohort=None，无 BLUP）。
      **deployable=false**（项目硬约束：零样本禁止部署到确诊终点人群，
      仅作斜率迁移的方向对照）。
  (c) LOCO + 本地截距重校准：留出队列对半切（一半重校准截距、一半评估），
      多种子。斜率冻结、只重估截距（跨端点部署的许可路径）。

方法论披露（预注册于输出 JSON）：
  - 截距-only 重校准不改变队列内排序 → 臂 (c) 的 AUROC 与臂 (b) 相同
    （秩不变性，构造使然）；重校准的收益体现在**校准**指标
    （Brier / ECE），臂 (c) 同时报告两者与零样本臂的逐种子配对差。
  - 端点异质性（用户决策：全 7 队列混合池化，仅披露不分组）：
    LTBI 族（nhanes/treats/brazil）vs 确诊/活动性 TB 族（kenya/crp/taiwan）
    vs incident TB（peru_mdr）。随机截距吸收基线率差异，但共享斜率可能
    被终点差异污染——LOCO 是唯一诚实的泛化证据。

族内分池（--pool-scope family，2026-09-15 用户指令：成本最低的直接归因）：
  LTBI 族（nhanes+treats+brazil，11.8k）/ confirmed_tb 族（kenya+crp+taiwan，
  64.9k）/ peru_mdr 单独（单成员族，无 LOCO——生存分析是其路径）。
  归因判据：若 nhanes 族内 LOCO 零样本从全池化 0.38 恢复，证明全池化灾难
  是终点污染而非特征不可迁移。输出含与全池化结果的逐队列对照
  （attribution_vs_all_pool，自动定位最新 multicohort_pooling_*.json）。

用法：
    python data/run_multicohort_pooling.py [--n-bootstrap 1000] [--n-seeds 5]
        [--pool-scope all|family]
输出：
    data/processed/multicohort_pooling_<date>.json（--pool-scope all，默认）
    data/processed/family_pooling_<date>.json（--pool-scope family）
"""
import argparse
import copy
import datetime
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

from tb_risk.scoring.predictor import MLRiskPredictor  # noqa: E402
from tb_risk.scoring.ml.training import ensure_interaction_features  # noqa: E402
from tb_risk.scoring.ml.validation import cluster_bootstrap_metric_ci  # noqa: E402
from tb_risk.scoring.ml.partial_pooling import (  # noqa: E402
    CohortMixedModel, STATSMODELS_AVAILABLE,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    StratifiedKFold, StratifiedGroupKFold, cross_val_predict,
    train_test_split,
)
from sklearn.metrics import roc_auc_score, brier_score_loss  # noqa: E402

# 队列注册表：文件、组列（任务1 探测结论）、终点家族（披露用）
COHORTS = {
    'nhanes':   {'file': 'ml_training_nhanes_real.csv',  'group_col': None,
                 'endpoint_family': 'ltbi',       'endpoint': 'LTBI (IGRA)'},
    'treats':   {'file': 'ml_training_treats_real.csv',  'group_col': 'group_id',
                 'endpoint_family': 'ltbi',       'endpoint': 'LTBI (TST/IGRA)'},
    'crp':      {'file': 'ml_training_crp_real.csv',     'group_col': None,
                 'endpoint_family': 'confirmed_tb', 'endpoint': '确诊 TB'},
    'kenya':    {'file': 'kenya_ml_training.csv',        'group_col': None,
                 'endpoint_family': 'confirmed_tb', 'endpoint': '细菌学确诊 TB'},
    'taiwan':   {'file': 'ml_training_taiwan_real.csv',  'group_col': None,
                 'endpoint_family': 'confirmed_tb', 'endpoint': '活动性 TB'},
    'brazil':   {'file': 'ml_training_brazil_real.csv',  'group_col': None,
                 'endpoint_family': 'ltbi',       'endpoint': 'LTBI (TST)'},
    'peru_mdr': {'file': 'ml_training_peru_mdr_real.csv', 'group_col': 'family_id',
                 'endpoint_family': 'incident_tb', 'endpoint': 'incident TB（MDR 密切接触者）'},
}

# 臂 (a) 基线 LR 超参（冻结，与主注册表 LR 同档：无正则搜索）
BASELINE_LR_KWARGS = dict(max_iter=2000, random_state=42)

# 终点族注册表（--pool-scope family 用）
FAMILIES = {}
for _name, _meta in COHORTS.items():
    FAMILIES.setdefault(_meta['endpoint_family'], []).append(_name)


def _loco_train_names(held_out, pool_scope='all'):
    """LOCO 训练集队列名单；族内单成员族返回 []（无族内 LOCO）。"""
    if pool_scope == 'family':
        fam = COHORTS[held_out]['endpoint_family']
        return [n for n in FAMILIES[fam] if n != held_out]
    return [n for n in COHORTS if n != held_out]


def load_all_cohorts():
    """加载 7 队列 → 22 维特征对齐（13 基础列自动补 9 交互列）。

    返回：
        {cohort: {'X','y','groups'(array|None),'n','n_pos','n_groups'}}
    """
    pred = MLRiskPredictor()
    names = pred.ALL_FEATURE_NAMES
    out = {}
    for name, meta in COHORTS.items():
        path = os.path.join(HERE, 'processed', meta['file'])
        df = pd.read_csv(path, encoding='utf-8-sig')
        df, missing = ensure_interaction_features(df, pred.INTERACTION_FEATURE_NAMES)
        if missing:
            print(f'  [{name}] 自动补交互列: {missing}')
        X = df[names].astype(float).to_numpy()
        y = df['tb_outcome'].astype(int).to_numpy()
        groups = (df[meta['group_col']].to_numpy()
                  if meta['group_col'] and meta['group_col'] in df.columns else None)
        out[name] = {'X': X, 'y': y, 'groups': groups,
                     'n': len(y), 'n_pos': int(y.sum()),
                     'n_groups': (int(len(np.unique(groups)))
                                  if groups is not None else None)}
        print(f'  [{name}] n={len(y)}, pos={int(y.sum())} '
              f'({y.mean():.3f}), groups={out[name]["n_groups"]}')
    return out


def _auroc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def _brier(y, p):
    return float(brier_score_loss(y, p))


def _ece(y, p, n_bins=10):
    """10 桶等宽 ECE（与项目"ECE 第一护栏"口径一致）。"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = ((p > lo) & (p <= hi)) if i > 0 else ((p >= lo) & (p <= hi))
        if mask.sum() == 0:
            continue
        ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return ece


def _ci(oof, y, groups, n_bootstrap, seed):
    """cluster bootstrap CI；无组列 → 每样本一簇（个体 bootstrap，机制统一）。"""
    g = np.arange(len(y)) if groups is None else np.asarray(groups)
    r = cluster_bootstrap_metric_ci(oof, y, g, metric='auroc',
                                    n_bootstrap=n_bootstrap, seed=seed)
    return {'point': r['point'], 'ci95': r['ci95'], 'n_effective': r['n_effective']}


def arm_a_baseline(cohort_data, n_bootstrap, seed=0):
    """臂 (a)：逐队列独立 LR 基线（组感知 CV / 随机 CV，池化 OOF）。"""
    results = {}
    for name, d in cohort_data.items():
        X, y, groups = d['X'], d['y'], d['groups']
        lr = LogisticRegression(**BASELINE_LR_KWARGS)
        if groups is not None:
            n_groups = d['n_groups']
            n_splits = max(2, min(5, n_groups))
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                      random_state=seed)
            protocol = f'stratified_group_kfold({n_splits})'
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            protocol = 'stratified_kfold(5)'
        oof = cross_val_predict(lr, X, y, cv=cv, method='predict_proba',
                               groups=groups)[:, 1]
        ci = _ci(oof, y, groups, n_bootstrap, seed)
        results[name] = {'AUROC': ci['point'], 'AUROC_ci95': ci['ci95'],
                         'n_bootstrap_effective': ci['n_effective'],
                         'cv_protocol': protocol}
        print(f'  [{name}] baseline AUROC={ci["point"]:.4f} '
              f'CI95=[{ci["ci95"][0]:.4f}, {ci["ci95"][1]:.4f}] ({protocol})')
    return results


def fit_loco_models(cohort_data, n_starts=3, random_state=42, pool_scope='all'):
    """一次性拟合全部 LOCO 模型（臂 b/c 共用，保证同一斜率口径）。

    每留出队列用其余队列独立重拟合（防斜率层泄漏）；多种子起点 +
    ELBO 选优（statsmodels fit_vb 起点未播种的非确定性修复）。
    pool_scope='family' 时训练集限定同族队列；单成员族跳过（无族内 LOCO）。
    """
    pred = MLRiskPredictor()
    names = pred.ALL_FEATURE_NAMES
    models = {}
    for held_out in COHORTS:
        train_names = _loco_train_names(held_out, pool_scope)
        if not train_names:
            print(f'  [{held_out}] 单成员族（{COHORTS[held_out]["endpoint_family"]}）'
                  f'——无族内 LOCO，跳过拟合')
            continue
        t0 = time.time()
        train = [(n, cohort_data[n]) for n in train_names]
        X = np.vstack([d['X'] for _, d in train])
        y = np.concatenate([d['y'] for _, d in train])
        c = np.concatenate([np.full(d['n'], n) for n, d in train])
        model = CohortMixedModel(feature_names=names).fit(
            X, y, c, fit_method='vb', random_state=random_state,
            n_starts=n_starts)
        models[held_out] = model
        print(f'  [{held_out}] LOCO 拟合（train={train_names}, '
              f'n={len(y)}）: conv={model.convergence_note} '
              f'elbo={model.elbo_selected} [{time.time()-t0:.0f}s]')
    return models


def arm_b_loco_zero_shot(cohort_data, models, baseline, n_bootstrap, seed=0,
                         pool_scope='all'):
    """臂 (b)：LOCO 零样本（总体截距，无 BLUP）——deployable=false。"""
    results = {}
    for held_out in COHORTS:
        model = models.get(held_out)
        if model is None:
            results[held_out] = {
                'skipped': ('单成员族（%s）无族内 LOCO——生存分析是其路径'
                            % COHORTS[held_out]['endpoint_family'])}
            continue
        d = cohort_data[held_out]
        p_zero = model.predict_proba(d['X'], cohort=None)
        ci = _ci(p_zero, d['y'], d['groups'], n_bootstrap, seed)
        base_auc = baseline[held_out]['AUROC']
        results[held_out] = {
            'AUROC': ci['point'], 'AUROC_ci95': ci['ci95'],
            'n_bootstrap_effective': ci['n_effective'],
            'delta_vs_baseline': (ci['point'] - base_auc
                                  if ci['point'] is not None and base_auc is not None
                                  else None),
            'baseline_AUROC': base_auc,
            'fit_method': model.fit_method,
            'convergence_note': model.convergence_note,
            'elbo_selected': model.elbo_selected,
            'elbo_all_starts': model.elbo_all_starts,
            'starts_agreement_fe_max_diff': model.starts_agreement,
            'loco_train': {'n_train': model.n_samples,
                           'train_cohorts': _loco_train_names(held_out, pool_scope)},
            'deployable': False,  # 项目硬约束：零样本臂禁止部署
            'deployment_note': ('零样本（总体截距）仅作斜率迁移方向对照；'
                                '跨端点部署须本地重校准（臂 c 路径）'),
        }
        print(f'  [{held_out}] LOCO zero-shot AUROC={ci["point"]:.4f} '
              f'(baseline {base_auc:.4f}, delta {ci["point"]-base_auc:+.4f})')
    return results


def arm_c_loco_local_recalibration(cohort_data, models, n_seeds, seed0=1000):
    """臂 (c)：LOCO + 本地截距重校准（对半切，多种子）。

    秩不变性披露：截距-only 重校准不改队列内排序 → 判别与臂 (b) 同分布；
    收益在校准（Brier/ECE）。逐种子配对差（同一评估半区）是主证据。
    单成员族（族内分池时 peru_mdr）无 LOCO 模型 → 跳过。
    """
    results = {}
    for held_out in COHORTS:
        model = models.get(held_out)
        if model is None:
            results[held_out] = {
                'skipped': ('单成员族（%s）无族内 LOCO——生存分析是其路径'
                            % COHORTS[held_out]['endpoint_family'])}
            continue
        d = cohort_data[held_out]
        X, y, groups = d['X'], d['y'], d['groups']
        per_seed = []
        for s in range(n_seeds):
            seed = seed0 + s
            XA, XB, yA, yB = train_test_split(
                X, y, test_size=0.5, stratify=y, random_state=seed)
            gB = None
            if groups is not None:
                idx = np.arange(len(y))
                _, idxB = train_test_split(
                    idx, test_size=0.5, stratify=y, random_state=seed)
                gB = groups[np.sort(idxB)]
            # 零样本概率（重校准前，同一评估半区）→ 配对对照
            p_zero = model.predict_proba(XB, cohort=None)
            # 本地重校准（斜率冻结）→ 评估半区
            model_recal = copy.deepcopy(model)
            model_recal.recalibrate_intercept_locally(XA, yA)
            p_recal = model_recal.predict_proba(XB, cohort=None)
            ci = _ci(p_recal, yB, gB, n_bootstrap=200, seed=seed)
            per_seed.append({
                'seed': seed,
                'AUROC': ci['point'], 'AUROC_ci95': ci['ci95'],
                'brier_zero_shot': _brier(yB, p_zero),
                'brier_recal': _brier(yB, p_recal),
                'brier_delta_paired': _brier(yB, p_recal) - _brier(yB, p_zero),
                'ece_zero_shot': _ece(yB, p_zero),
                'ece_recal': _ece(yB, p_recal),
                'ece_delta_paired': _ece(yB, p_recal) - _ece(yB, p_zero),
                'recal_intercept': model_recal.local_intercept,
                'recal_method': model_recal.local_recal_note,
            })
        agg = lambda k: float(np.mean([r[k] for r in per_seed]))
        results[held_out] = {
            'n_seeds': n_seeds,
            'mean_AUROC': agg('AUROC'),
            'AUROC_note': ('与臂 (b) 同一斜率/同一留出队列，评估在对半子样本'
                           '上（截距不改排序 → 判别与臂 (b) 同分布，均值差异'
                           '仅来自子样本抽样）'),
            'mean_brier_zero_shot': agg('brier_zero_shot'),
            'mean_brier_recal': agg('brier_recal'),
            'mean_brier_delta_paired': agg('brier_delta_paired'),
            'mean_ece_zero_shot': agg('ece_zero_shot'),
            'mean_ece_recal': agg('ece_recal'),
            'mean_ece_delta_paired': agg('ece_delta_paired'),
            'per_seed': per_seed,
            'deployable': True,
            'deployment_note': ('斜率冻结 + 本地截距重校准 = 跨端点部署'
                                '许可路径（项目硬约束满足）'),
        }
        print(f'  [{held_out}] LOCO+recal: AUROC={agg("AUROC"):.4f} '
              f'Brier {agg("brier_zero_shot"):.4f}→{agg("brier_recal"):.4f} '
              f'({agg("brier_delta_paired"):+.4f}) '
              f'ECE {agg("ece_zero_shot"):.4f}→{agg("ece_recal"):.4f} '
              f'({agg("ece_delta_paired"):+.4f})')
    return results


def _pool_descriptive_fit(cohort_data, members, n_starts, random_state):
    """对给定队列名单做一次混合模型描述性拟合。"""
    pred = MLRiskPredictor()
    X = np.vstack([cohort_data[n]['X'] for n in members])
    y = np.concatenate([cohort_data[n]['y'] for n in members])
    c = np.concatenate([np.full(cohort_data[n]['n'], n) for n in members])
    model = CohortMixedModel(feature_names=pred.ALL_FEATURE_NAMES)
    model.fit(X, y, c, fit_method='vb', random_state=random_state,
              n_starts=n_starts)
    summary = model.summary_dict()
    summary['implied_intercepts'] = {
        n: {'population_plus_blup': float(model.fe_mean[0] + model.vc_mean[n]),
            'observed_logit': float(np.log(
                cohort_data[n]['n_pos'] /
                max(cohort_data[n]['n'] - cohort_data[n]['n_pos'], 1)))}
        for n in members}
    return summary


def full_pool_descriptive(cohort_data, n_starts=3, random_state=42,
                          pool_scope='all'):
    """池化描述性拟合（BLUP / 截距方差 / 共享斜率）。

    pool_scope='all'：全 7 队列一次拟合。
    pool_scope='family'：逐族拟合（族内斜率/截距方差的描述性对照）；
    单成员族跳过（随机截距方差不可识别——生存分析是其路径）。
    """
    if pool_scope == 'family':
        out = {}
        for fam, members in FAMILIES.items():
            if len(members) < 2:
                out[fam] = {'skipped': ('单成员族——随机截距方差不可识别，'
                                        '生存分析是其路径')}
                continue
            t0 = time.time()
            out[fam] = _pool_descriptive_fit(cohort_data, members,
                                             n_starts, random_state)
            print(f'  [{fam}] 族内池化描述性拟合（n={sum(cohort_data[n]["n"] for n in members)}）'
                  f'[{time.time()-t0:.0f}s]')
        return {'pool_scope': 'family', 'per_family': out}
    return _pool_descriptive_fit(cohort_data, list(COHORTS),
                                 n_starts, random_state)


def attribution_vs_all_pool(arm_b, pool_scope):
    """族内 vs 全池化 LOCO 零样本逐队列对照（族内分池归因，2026-09-15）。

    自动定位最新的全池化结果（multicohort_pooling_<8位日期>.json，排除冒烟）。
    归因判据（用户指令）：若 nhanes 族内 LOCO 从全池化 0.38 恢复，
    证明全池化灾难是终点污染而非特征不可迁移。
    """
    if pool_scope != 'family':
        return None
    import glob
    pattern = os.path.join(HERE, 'processed', 'multicohort_pooling_????????.json')
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not candidates:
        return {'error': ('未找到全池化结果（multicohort_pooling_<date>.json）'
                          '——先跑 --pool-scope all')}
    latest = candidates[-1]
    with open(latest, encoding='utf-8') as f:
        all_pool = json.load(f)
    all_arm_b = all_pool.get('arm_b_loco_zero_shot', {})
    out = {
        'all_pool_file': os.path.basename(latest),
        'all_pool_date': all_pool.get('date'),
        'criterion': ('族内 LOCO 零样本 − 全池化 LOCO 零样本；'
                      'nhanes 全池 0.38 若族内恢复 → 终点污染归因成立'),
        'per_cohort': {},
    }
    for name in COHORTS:
        fam_res = arm_b.get(name, {})
        if 'skipped' in fam_res:
            out['per_cohort'][name] = {'skipped': fam_res['skipped']}
            continue
        all_res = all_arm_b.get(name, {})
        fam_auc = fam_res.get('AUROC')
        all_auc = all_res.get('AUROC')
        base_auc = fam_res.get('baseline_AUROC')
        entry = {
            'family_loco_zero_shot_AUROC': fam_auc,
            'all_pool_loco_zero_shot_AUROC': all_auc,
            'recovery_delta_family_minus_all': (
                fam_auc - all_auc
                if fam_auc is not None and all_auc is not None else None),
            'baseline_AUROC': base_auc,
        }
        if fam_auc is not None and all_auc is not None:
            delta = fam_auc - all_auc
            if all_auc < 0.5 <= fam_auc:
                entry['attribution'] = ('全池化低于随机而族内恢复 ≥0.5：'
                                        '终点污染归因成立')
            elif delta >= 0.05:
                entry['attribution'] = ('族内显著优于全池化（Δ≥0.05）：'
                                        '终点污染证据')
            elif delta <= -0.05:
                entry['attribution'] = ('族内显著差于全池化（Δ≤-0.05）：'
                                        '跨族样本对判别有正贡献')
            else:
                entry['attribution'] = '族内与全池化相当（|Δ|<0.05）'
        out['per_cohort'][name] = entry
    # 归因主判据：nhanes（全池 0.38 灾难）
    nh = out['per_cohort'].get('nhanes', {})
    if 'recovery_delta_family_minus_all' in nh:
        out['nhanes_criterion'] = {
            'all_pool': nh.get('all_pool_loco_zero_shot_AUROC'),
            'family': nh.get('family_loco_zero_shot_AUROC'),
            'recovered_to_nonrandom': (
                nh.get('family_loco_zero_shot_AUROC') is not None
                and nh['family_loco_zero_shot_AUROC'] >= 0.5),
            'conclusion': ('终点污染归因成立（族内恢复 ≥0.5）'
                           if nh.get('family_loco_zero_shot_AUROC', 0) >= 0.5
                           else '族内未恢复——特征不可迁移假说仍存活'),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description='多队列部分池化评估（LOCO）')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='cluster bootstrap 次数（默认 1000）')
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='臂 (c) 重校准种子数（默认 5）')
    parser.add_argument('--n-starts', type=int, default=3,
                        help='VB 多起点数（ELBO 选优，默认 3）')
    parser.add_argument('--pool-scope', choices=['all', 'family'], default='all',
                        help='池化范围：all=全 7 队列（默认）；'
                             'family=族内分池（终点族归因，2026-09-15）')
    parser.add_argument('--out', type=str, default=None,
                        help='输出 JSON 路径（默认 data/processed/multicohort_pooling_<date>.json'
                             ' 或 family_pooling_<date>.json）')
    args = parser.parse_args()

    if not STATSMODELS_AVAILABLE:
        print('错误：statsmodels 不可用（pip install statsmodels）')
        sys.exit(1)

    date_tag = datetime.date.today().strftime('%Y%m%d')
    prefix = ('family_pooling' if args.pool_scope == 'family'
              else 'multicohort_pooling')
    out_path = args.out or os.path.join(
        HERE, 'processed', f'{prefix}_{date_tag}.json')

    t_start = time.time()
    scope_note = ('族内分池（LTBI: nhanes+treats+brazil / confirmed_tb: '
                  'kenya+crp+taiwan / peru_mdr 单成员族跳过）'
                  if args.pool_scope == 'family' else '全 7 队列混合池化')
    print('== 多队列部分池化评估（cohort 随机截距 + 共享斜率）==')
    print(f'--n-bootstrap={args.n_bootstrap}, --n-seeds={args.n_seeds}, '
          f'--pool-scope={args.pool_scope}')
    print(f'池化范围：{scope_note}')

    print('\n[1/5] 加载 7 队列（22 维对齐）')
    cohort_data = load_all_cohorts()

    print('\n[2/5] 臂 (a)：逐队列独立 LR 基线')
    baseline = arm_a_baseline(cohort_data, args.n_bootstrap)

    print('\n[3/5] LOCO 模型拟合（× %d 起点，臂 b/c 共用）' % args.n_starts)
    models = fit_loco_models(cohort_data, n_starts=args.n_starts,
                             pool_scope=args.pool_scope)

    print('\n[4/5] 臂 (b)：LOCO 零样本 + 臂 (c)：本地截距重校准')
    arm_b = arm_b_loco_zero_shot(cohort_data, models, baseline, args.n_bootstrap,
                                 pool_scope=args.pool_scope)
    arm_c = arm_c_loco_local_recalibration(cohort_data, models, args.n_seeds)

    print('\n[5/5] 池化描述性拟合（BLUP / 共享斜率）')
    full_pool = full_pool_descriptive(cohort_data, n_starts=args.n_starts,
                                      pool_scope=args.pool_scope)

    # ---- 族内 vs 全池化归因对照（--pool-scope family）----
    attribution = attribution_vs_all_pool(arm_b, args.pool_scope)

    # ---- 汇总输出 ----
    endpoint_families = {}
    for name, meta in COHORTS.items():
        endpoint_families.setdefault(meta['endpoint_family'], []).append(
            {'cohort': name, 'endpoint': meta['endpoint'],
             'n': cohort_data[name]['n'],
             'n_pos': cohort_data[name]['n_pos'],
             'prevalence': round(cohort_data[name]['n_pos'] /
                                 cohort_data[name]['n'], 4)})

    output = {
        'experiment': 'multicohort_partial_pooling_loco',
        'date': date_tag,
        'pool_scope': args.pool_scope,
        'pool_scope_note': ('族内分池（用户指令 2026-09-15）：LTBI 族'
                            '（nhanes+treats+brazil）/ confirmed_tb 族'
                            '（kenya+crp+taiwan）/ peru_mdr 单成员族无 LOCO'
                            if args.pool_scope == 'family' else
                            '全 7 队列混合池化（用户决策 2026-09-14）'),
        'runtime_seconds': round(time.time() - t_start, 1),
        'protocol': {
            'model': ('BinomialBayesMixedGLM: logit P(y=1) = α + βᵀx + '
                      'u_cohort, u ~ N(0, σ²_α)；内部 StandardScaler；'
                      'fit_vb 多起点（%d）× ELBO 选优 / fit_map 回退'
                      % args.n_starts),
            'determinism_note': ('statsmodels fit_vb/fit_map 起点用未播种'
                                 '全局 RNG（bayes_mixed_glm.py L402/L759）'
                                 '——已封装为种子隔离 + 多起点 ELBO 选优 + '
                                 '起点间 fe_mean 最大差稳定性披露'),
            'feature_space': '22 维（13 基础 + 9 交互，ensure_interaction_features 对齐）',
            'arm_a': '逐队列独立 LR（组感知 CV for treats/peru_mdr）',
            'arm_b': ('LOCO 零样本：留出队列总体截距预测，每留出队列'
                      '独立重拟合（防斜率层泄漏）'),
            'arm_c': ('LOCO + 本地截距重校准：留出队列对半切'
                      '（一半重校准、一半评估）× %d 种子' % args.n_seeds),
            'n_bootstrap': args.n_bootstrap,
            'rank_invariance_note': ('臂 (c) 截距-only 重校准不改队列内排序 → '
                                     '判别与臂 (b) 同分布（构造使然）；重校准收益'
                                     '在校准指标（Brier/ECE 逐种子配对差）'),
            'low_prevalence_caveat': ('低患病率队列（kenya 0.5%）的 Brier/ECE '
                                      '改善主要由截距重校准的基线率匹配驱动'
                                      '（判别不变）——校准指标达标不能单独作为'
                                      '迁移成功证据，AUROC 仍是判别审计口径'),
        },
        'cohorts': {
            name: {'n': d['n'], 'n_pos': d['n_pos'],
                   'prevalence': round(d['n_pos'] / d['n'], 4),
                   'endpoint': COHORTS[name]['endpoint'],
                   'endpoint_family': COHORTS[name]['endpoint_family'],
                   'group_column': COHORTS[name]['group_col'],
                   'n_groups': d['n_groups']}
            for name, d in cohort_data.items()},
        'endpoint_heterogeneity_disclosure': {
            'families': endpoint_families,
            'decision': ('族内分池（用户指令 2026-09-15）：按 endpoint_family '
                         '分组池化，单成员族跳过 LOCO'
                         if args.pool_scope == 'family' else
                         '全 7 队列混合池化（用户决策 2026-09-14）：不分组'),
            'risk': ('LTBI 族与确诊/活动性 TB 族共用斜率——随机截距吸收'
                     '基线率差异，但斜率可能被终点差异污染；LOCO 臂 (b) '
                     '的 delta_vs_baseline 是逐队列迁移损失的诚实证据'),
            'hard_constraint': ('零样本臂 deployable=false；跨端点部署'
                                '许可路径 = 斜率冻结 + 本地截距重校准（臂 c）'),
        },
        'arm_a_per_cohort_baseline': baseline,
        'arm_b_loco_zero_shot': arm_b,
        'arm_c_loco_local_recalibration': arm_c,
        'full_pool_descriptive': full_pool,
    }
    if attribution is not None:
        output['attribution_vs_all_pool'] = attribution

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\n输出 → {out_path}（{time.time()-t_start:.0f}s）')

    # 控制台速览：逐队列 delta 表（单成员族显示跳过）
    print('\n== 逐队列速览（LOCO 零样本 vs 独立基线）==')
    print(f'{"cohort":<10}{"family":<14}{"n":>7}{"baseline":>10}'
          f'{"zero-shot":>11}{"delta":>9}{"Brier Δ(recal)":>15}')
    for name in COHORTS:
        b = baseline[name]['AUROC']
        if 'skipped' in arm_b[name]:
            print(f'{name:<10}{COHORTS[name]["endpoint_family"]:<14}'
                  f'{cohort_data[name]["n"]:>7}{b:>10.4f}'
                  f'{"（单成员族，跳过）":>11}')
            continue
        z = arm_b[name]['AUROC']
        dl = arm_b[name]['delta_vs_baseline']
        bd = arm_c[name]['mean_brier_delta_paired']
        print(f'{name:<10}{COHORTS[name]["endpoint_family"]:<14}'
              f'{cohort_data[name]["n"]:>7}{b:>10.4f}{z:>11.4f}'
              f'{dl:>+9.4f}{bd:>+15.4f}')

    # 族内 vs 全池化归因速览
    if attribution is not None and 'per_cohort' in attribution:
        print(f'\n== 族内 vs 全池化归因（参照 {attribution["all_pool_file"]}）==')
        print(f'{"cohort":<10}{"all-pool":>10}{"family":>10}{"recovery":>10}'
              f'  attribution')
        for name, e in attribution['per_cohort'].items():
            if 'skipped' in e:
                print(f'{name:<10}{"—":>10}{"—":>10}{"—":>10}  {e["skipped"]}')
                continue
            a = e['all_pool_loco_zero_shot_AUROC']
            f_ = e['family_loco_zero_shot_AUROC']
            d = e['recovery_delta_family_minus_all']
            print(f'{name:<10}{a:>10.4f}{f_:>10.4f}{d:>+10.4f}'
                  f'  {e.get("attribution", "")}')
        if 'nhanes_criterion' in attribution:
            nc = attribution['nhanes_criterion']
            print(f'\nnhanes 归因判据：全池 {nc["all_pool"]:.4f} → 族内 '
                  f'{nc["family"]:.4f}——{nc["conclusion"]}')


if __name__ == '__main__':
    main()
