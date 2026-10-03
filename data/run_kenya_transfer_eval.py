#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kenya 预训练 → 小队列微调：迁移 vs 从零的多种子判决实验（任务2，S8）

背景：Kenya 患病率调查（63,050 行 / 336 事件 / 13 基础特征）是最大的
真实队列，适合做预训练底座；Taiwan（129）/ Brazil（202）/ CRP（1777）/
TREATS（4529）小队列本地训练方差大。本脚本回答：Kenya 底座 + 续训式
微调（warm_start / init_model，transfer_learning.py 已实现）相对本地
从零训练，判别到底增益多少？

双臂设计（同一折分配 → 逐种子配对）：
  (a) 从零臂：_train_single_model（主训练管线同款构造：注册表默认参 +
      类不平衡加权）在每折训练集上从零拟合；
  (b) 迁移臂：Kenya checkpoint 模型 + _finetune_single_model
      （GB warm_start / XGBoost xgb_model 续训 / LightGBM·CatBoost
      init_model，学习率减半防灾难性遗忘；RF warm_start 分支已下线
      2026-09-15——S8 实测 taiwan −0.098 / crp −0.055 负迁移显著，
      RF 走 fallback_from_scratch）。

部署配方收窄（2026-09-15，S8 判决 → P4 退役 2026-09-16）：
  - XGBoost init_model 微调配方已退役（P4）：S8 曾以「同族终点 +
    本地 n<300」收窄入配方（taiwan 0.481→0.547，+0.067，22 维口径）；
    v4 特征空间定版后 taiwan 本地 v4 LR 0.874（主管线）/0.856（F2
    口径）全面支配微调上限，confirmed_tb 族（kenya 0.952 / taiwan
    0.874 / crp 0.781，kenya 为 P6 原始数据再审后 v4 口径）一律
    本地 v4 部署——退役登记见
    transfer_learning.RETIRED_DEPLOYMENT_RECIPES；
  - RF warm_start 分支下线（负迁移显著且机制清楚）。

  每种子：同 CV splitter（TREATS 组感知 StratifiedGroupKFold by
  group_id（14 社区）；其余 StratifiedKFold 5 折）→ 池化 OOF AUROC +
  配对 delta；N 种子汇总（mean±sd + Wilcoxon 符号秩）。

端点披露（项目硬约束）：
  - Kenya = 细菌学确诊 TB → taiwan（活动性）/ crp（确诊）同族：
    同族微调 = 样本效率增益的诚实证据；
  - → brazil / treats（LTBI 族）跨终点：标注为**重校准型迁移**，
    判别结论仅作方向对照；
  - 微调模式 fallback_from_scratch 的模型键 = 该键迁移臂并未真正迁移，
    结果按"从零（减半学习率）"解读。

用法：
    python data/run_kenya_transfer_eval.py [--seeds 10] [--folds 5]
        [--force-pretrain] [--models rf,xgboost,...]
输出：
    data/processed/kenya_transfer_eval_<date>.json
    （预训练 checkpoint：data/processed/kenya_pretrain_checkpoint.joblib）
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
from tb_risk.scoring.ml.training import (  # noqa: E402
    ensure_interaction_features, train_from_real_data,
    MODEL_REGISTRY, _train_single_model,
)
from tb_risk.scoring.ml.transfer_learning import (  # noqa: E402
    _finetune_single_model,
)
from tb_risk.scoring.ml.validation import cluster_bootstrap_metric_ci  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    StratifiedKFold, StratifiedGroupKFold,
)
from sklearn.metrics import roc_auc_score  # noqa: E402

KENYA_CSV = os.path.join(HERE, 'processed', 'kenya_ml_training.csv')
CKPT = os.path.join(HERE, 'processed', 'kenya_pretrain_checkpoint.joblib')

# 目标队列（计划 S8）：小队列 + treats（组感知协议代表）
TARGETS = {
    'taiwan':   {'file': 'ml_training_taiwan_real.csv', 'group_col': None,
                 'endpoint': '活动性 TB', 'endpoint_family': 'confirmed_tb',
                 'transfer_type': 'same_family_finetune'},
    'brazil':   {'file': 'ml_training_brazil_real.csv', 'group_col': None,
                 'endpoint': 'LTBI (TST)', 'endpoint_family': 'ltbi',
                 'transfer_type': 'cross_endpoint_recalibration'},
    'crp':      {'file': 'ml_training_crp_real.csv', 'group_col': None,
                 'endpoint': '确诊 TB', 'endpoint_family': 'confirmed_tb',
                 'transfer_type': 'same_family_finetune'},
    'treats':   {'file': 'ml_training_treats_real.csv', 'group_col': 'group_id',
                 'endpoint': 'LTBI (TST/IGRA)', 'endpoint_family': 'ltbi',
                 'transfer_type': 'cross_endpoint_recalibration'},
}

MIN_LEARNING_RATE = 0.05


def pretrain_kenya(force=False):
    """Kenya 预训练（一次，checkpoint 缓存）。"""
    if os.path.exists(CKPT) and not force:
        pred = MLRiskPredictor()
        if pred.load_model(CKPT) and getattr(pred, 'models', None):
            print(f'  加载已有 checkpoint: {CKPT}'
                  f'（{len(pred.models)} 模型: {sorted(pred.models)}）')
            return pred, {'cached': True}
    print(f'  预训练 Kenya（{KENYA_CSV}）...')
    t0 = time.time()
    pred = MLRiskPredictor()
    result = train_from_real_data(pred, KENYA_CSV, target_column='tb_outcome')
    if not result['success']:
        print(f'  预训练失败: {result.get("error_type")} / {result.get("diagnostics")}')
        sys.exit(1)
    assert pred.save_model(CKPT), 'checkpoint 保存失败'
    info = {'cached': False,
            'n': result['n_samples'], 'n_pos': result['n_positive'],
            'runtime_s': round(time.time() - t0, 1),
            'model_auroc': {k: v.get('AUROC')
                            for k, v in result['model_performance'].items()},
            'cv_protocol': {k: v.get('cv_protocol')
                            for k, v in result['model_performance'].items()}}
    print(f'  预训练完成: n={info["n"]}, pos={info["n_pos"]}, '
          f'{len(pred.models)} 模型 [{info["runtime_s"]}s] → {CKPT}')
    return pred, info


def load_targets():
    """目标队列 → 22 维对齐 + 组列。"""
    ref = MLRiskPredictor()
    names = ref.ALL_FEATURE_NAMES
    out = {}
    for name, meta in TARGETS.items():
        df = pd.read_csv(os.path.join(HERE, 'processed', meta['file']),
                         encoding='utf-8-sig')
        df, _ = ensure_interaction_features(df, ref.INTERACTION_FEATURE_NAMES)
        # 保持命名 DataFrame：与主训练管线同款输入契约（命名特征），
        # 且与 Kenya 预训练模型的 feature_names_in_ 同名同序对齐
        X = df[names].apply(pd.to_numeric, errors='coerce').fillna(0) \
                     .astype(float)
        y = df['tb_outcome'].astype(int).to_numpy()
        groups = (df[meta['group_col']].to_numpy()
                  if meta['group_col'] else None)
        out[name] = {'X': X, 'y': y, 'groups': groups, 'n': len(y),
                     'n_pos': int(y.sum())}
        print(f'  [{name}] n={len(y)}, pos={int(y.sum())} '
              f'({y.mean():.3f}), groups='
              f'{len(np.unique(groups)) if groups is not None else None}')
    return out


def _make_splitter(groups, n_folds, seed):
    if groups is not None:
        n_groups = len(np.unique(groups))
        n_splits = max(2, min(n_folds, n_groups))
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed)
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)


def _predict_pos(model, X):
    """predict_proba 正类列（异常 → NaN 行，汇总时跳过）。"""
    try:
        p = model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else np.asarray(p, dtype=float)
    except Exception:
        return np.full(len(X), np.nan)


def eval_cohort(name, d, template, model_keys, n_seeds, n_folds,
                n_bootstrap, seed0=2000):
    """单目标队列：双臂多种子配对 CV。

    返回：
        {model_key: {seeds: [...], mean_*, wilcoxon_p, pooled_*, ...}}
    """
    X, y, groups = d['X'], d['y'], d['groups']
    n = len(y)
    results = {}
    for key in model_keys:
        spec = MODEL_REGISTRY.get(key)
        if spec is None or template.models.get(key) is None:
            continue
        per_seed, modes = [], []
        oof_acc = {'scratch': np.zeros(n), 'transfer': np.zeros(n)}
        for s in range(n_seeds):
            seed = seed0 + s
            cv = _make_splitter(groups, n_folds, seed)
            oof_s = np.full(n, np.nan)
            oof_t = np.full(n, np.nan)
            seed_modes = []
            for tr, te in cv.split(X, y, groups):
                # 从零臂：主训练管线同款构造
                m_scratch = _train_single_model(
                    spec, X.iloc[tr], y[tr], enable_hyperopt=False, random_state=seed)
                if m_scratch is not None:
                    oof_s[te] = _predict_pos(m_scratch, X.iloc[te])
                # 迁移臂：Kenya 底座 + 续训微调
                # （_finetune_single_model 内部 deepcopy，模板不被污染）
                m_ft, mode, _detail = _finetune_single_model(
                    key, template.models[key], X.iloc[tr], y[tr], seed,
                    warm_start=True, n_finetune_trees=None,
                    min_learning_rate=MIN_LEARNING_RATE)
                if m_ft is not None:
                    oof_t[te] = _predict_pos(m_ft, X.iloc[te])
                    seed_modes.append(mode)
            ok = (~np.isnan(oof_s)) & (~np.isnan(oof_t))
            a_s = roc_auc_score(y[ok], oof_s[ok]) if ok.sum() > 10 else None
            a_t = roc_auc_score(y[ok], oof_t[ok]) if ok.sum() > 10 else None
            if a_s is not None and a_t is not None:
                per_seed.append({'seed': seed, 'auroc_scratch': a_s,
                                 'auroc_transfer': a_t, 'delta': a_t - a_s})
                oof_acc['scratch'] += np.nan_to_num(oof_s)
                oof_acc['transfer'] += np.nan_to_num(oof_t)
            if seed_modes:
                modes.extend(seed_modes)
        if not per_seed:
            continue
        deltas = np.array([r['delta'] for r in per_seed])
        # 池化 OOF（逐种子平均，Peru/HomeACF 协议同款）+ cluster bootstrap CI
        oof_pool = {arm: oof_acc[arm] / len(per_seed)
                    for arm in ('scratch', 'transfer')}
        g = np.arange(n) if groups is None else np.asarray(groups)
        pooled = {}
        for arm in ('scratch', 'transfer'):
            ci = cluster_bootstrap_metric_ci(
                oof_pool[arm], y, g, metric='auroc',
                n_bootstrap=n_bootstrap, seed=seed0)
            pooled[arm] = {'AUROC': ci['point'], 'ci95': ci['ci95']}
        # Wilcoxon 符号秩（逐种子配对 delta 的推断汇总）
        try:
            from scipy.stats import wilcoxon
            wp = float(wilcoxon(deltas).pvalue) if np.any(deltas != 0) else 1.0
        except Exception:
            wp = None
        mode_counts = {m: modes.count(m) for m in set(modes)}
        r = {
            'n_seeds': len(per_seed),
            'seeds': per_seed,
            'mean_auroc_scratch': float(np.mean([r['auroc_scratch'] for r in per_seed])),
            'mean_auroc_transfer': float(np.mean([r['auroc_transfer'] for r in per_seed])),
            'mean_delta': float(deltas.mean()),
            'sd_delta': float(deltas.std(ddof=1)) if len(deltas) > 1 else None,
            'n_positive_delta_seeds': int((deltas > 0).sum()),
            'wilcoxon_p': wp,
            'pooled_oof': pooled,
            'finetune_mode_counts': mode_counts,
            'transfer_effective': ('fallback_from_scratch' not in mode_counts),
        }
        results[key] = r
        sd_txt = f'{r["sd_delta"]:.4f}' if r['sd_delta'] is not None else 'n/a'
        p_txt = f'{wp:.3f}' if wp is not None else 'n/a'
        print(f'    [{key}] scratch {r["mean_auroc_scratch"]:.4f} '
              f'→ transfer {r["mean_auroc_transfer"]:.4f} '
              f'(Δ {r["mean_delta"]:+.4f}±{sd_txt}, p={p_txt}) '
              f'modes={mode_counts}')
    return results


def main():
    parser = argparse.ArgumentParser(description='Kenya 预训练→小队列微调评估')
    parser.add_argument('--seeds', type=int, default=10,
                        help='CV 种子数（默认 10）')
    parser.add_argument('--folds', type=int, default=5,
                        help='CV 折数（默认 5）')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='池化 OOF cluster bootstrap 次数（默认 1000）')
    parser.add_argument('--models', type=str, default=None,
                        help='逗号分隔模型键（默认 checkpoint 全部模型）')
    parser.add_argument('--force-pretrain', action='store_true',
                        help='忽略缓存重新预训练 Kenya')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    date_tag = datetime.date.today().strftime('%Y%m%d')
    out_path = args.out or os.path.join(
        HERE, 'processed', f'kenya_transfer_eval_{date_tag}.json')

    t_start = time.time()
    print('== Kenya 预训练 → 小队列微调（迁移 vs 从零，多种子配对）==')
    print(f'--seeds={args.seeds}, --folds={args.folds}, '
          f'--n-bootstrap={args.n_bootstrap}')

    print('\n[1/4] Kenya 预训练 / checkpoint')
    template, pretrain_info = pretrain_kenya(force=args.force_pretrain)

    model_keys = (args.models.split(',') if args.models
                  else list(template.models.keys()))
    model_keys = [k for k in model_keys if k in template.models]
    print(f'  评估模型键: {model_keys}')

    print('\n[2/4] 加载目标队列（22 维对齐）')
    targets = load_targets()

    print('\n[3/4] 双臂 CV（同折配对 × %d 种子）' % args.seeds)
    all_results = {}
    for name, d in targets.items():
        t0 = time.time()
        print(f'  [{name}]（{TARGETS[name]["endpoint"]}, '
              f'{TARGETS[name]["transfer_type"]}）')
        all_results[name] = eval_cohort(
            name, d, template, model_keys, args.seeds, args.folds,
            args.n_bootstrap)
        print(f'  [{name}] 完成 [{time.time()-t0:.0f}s]')

    print('\n[4/4] 汇总输出')
    output = {
        'experiment': 'kenya_pretrain_finetune_vs_scratch',
        'date': date_tag,
        'runtime_seconds': round(time.time() - t_start, 1),
        'pretrain': {
            'source': 'kenya_ml_training.csv (Kenya 患病率调查)',
            'endpoint': '细菌学确诊 TB（涂片/Xpert/培养任一阳性）',
            'checkpoint': CKPT,
            'cached': pretrain_info['cached'],
            'n': pretrain_info.get('n'),
            'n_pos': pretrain_info.get('n_pos'),
            'model_auroc': pretrain_info.get('model_auroc'),
        },
        'protocol': {
            'scratch_arm': ('_train_single_model（主训练管线同款：注册表'
                            '默认参 + 类不平衡加权），每折从零拟合'),
            'transfer_arm': ('Kenya checkpoint + _finetune_single_model：'
                             'GB warm_start / XGBoost xgb_model 续训 / '
                             'LightGBM·CatBoost init_model；学习率减半'
                             '（下限 %.2f）防灾难性遗忘。RF warm_start 分支'
                             '已下线（2026-09-15：taiwan −0.098 / crp −0.055 '
                             '负迁移显著），RF 走 fallback_from_scratch'
                             % MIN_LEARNING_RATE),
            'cv': ('TREATS StratifiedGroupKFold(5) by group_id（14 社区，'
                   '组感知防泄漏）；其余 StratifiedKFold(%d)；同折双臂配对'
                   % args.folds),
            'seeds': args.seeds,
            'pooled_oof': '逐种子 OOF 平均 → 单一 AUROC + cluster bootstrap CI',
            'inference': '逐种子配对 delta：mean±sd + Wilcoxon 符号秩',
            'fairness_note': ('从零臂含类不平衡加权（_train_single_model '
                              '标准路径）；微调臂为 transfer_learning.py '
                              '既有实现（构造参数级加权随 deepcopy 保留，'
                              'fit 级 sample_weight 不重传）——两臂各自是'
                              '其生产配方，差异即配方差异'),
        },
        'deployment_recipe': {
            'updated': '2026-09-15（S8 判决收窄）',
            'xgboost_init_model': {
                'eligible': '同族终点 + 本地 n<300',
                'evidence': {
                    'taiwan': 'n=129 同族 confirmed_tb，mean_delta +0.067（入配方）',
                    'brazil': ('n=202 跨族 ltbi，mean_delta +0.037，但跨终点'
                               '部署须本地重校准（项目硬约束）——不入配方'),
                    'crp': 'n=1777 同族但样本充足，mean_delta −0.018——不入配方',
                    'treats': 'n=4529 跨族且样本充足，mean_delta −0.062——不入配方',
                },
            },
            'rf_warm_start': {
                'status': '下线（2026-09-15）',
                'evidence': ('taiwan −0.098 / crp −0.055 负迁移显著且机制清楚'
                             '（续训 fit 上 class_weight=balanced 按本地标签'
                             '重算 + 预训练树与本地树类先验冲突）；'
                             'RF 微调一律 fallback_from_scratch'),
            },
        },
        'endpoint_disclosure': {
            'kenya_endpoint': '细菌学确诊 TB',
            'same_family': {
                'taiwan': '活动性 TB（同族：微调增益=样本效率证据）',
                'crp': '确诊 TB（同族：微调增益=样本效率证据）'},
            'cross_endpoint': {
                'brazil': ('LTBI (TST)——跨终点：重校准型迁移，判别结论仅作'
                           '方向对照（项目硬约束：跨终点部署须本地重校准）'),
                'treats': ('LTBI (TST/IGRA)——跨终点：同 brazil，方向对照'
                           '级证据')},
            'fallback_note': ('finetune_mode_counts 含 fallback_from_scratch '
                              '的模型键：该键迁移臂实为"从零（减半学习率）"，'
                              'transfer_effective=false'),
        },
        'cohorts': {
            name: {'n': d['n'], 'n_pos': d['n_pos'],
                   'prevalence': round(d['n_pos'] / d['n'], 4),
                   'endpoint': TARGETS[name]['endpoint'],
                   'endpoint_family': TARGETS[name]['endpoint_family'],
                   'transfer_type': TARGETS[name]['transfer_type'],
                   'cv_protocol': ('stratified_group_kfold'
                                   if d['groups'] is not None
                                   else 'stratified_kfold')}
            for name, d in targets.items()},
        'results': all_results,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\n输出 → {out_path}（{time.time()-t_start:.0f}s）')

    # 控制台速览
    print('\n== 逐队列速览（迁移 − 从零，逐模型键）==')
    print(f'{"cohort":<9}{"model":<18}{"scratch":>9}{"transfer":>10}'
          f'{"Δ":>9}{"p":>8}{"transfer?"}')
    for name in TARGETS:
        for key, r in all_results[name].items():
            p_txt = (f'{r["wilcoxon_p"]:>8.3f}' if r['wilcoxon_p'] is not None
                     else f'{"n/a":>8}')
            print(f'{name:<9}{key:<18}{r["mean_auroc_scratch"]:>9.4f}'
                  f'{r["mean_auroc_transfer"]:>10.4f}'
                  f'{r["mean_delta"]:>+9.4f}{p_txt}'
                  f'{"yes" if r["transfer_effective"] else "FALLBACK"}')


if __name__ == '__main__':
    main()
