#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LTBI 族 v4 联合池化：列并集 + 结构缺失披露（P3，2026-09-16 用户指令）

背景：F2 定量确证特征空间是第一瓶颈（v4 extras 增益 LR 配对差全部
CI 排除 0：taiwan +0.279 / crp +0.186 / brazil +0.109 / nhanes +0.089 /
treats +0.079 / peru +0.064）；F1 族内分池坐实终点污染归因（LTBI 族
随机截距 SD 0.217，族内池化安全；treats 族内零样本 0.5537 超本地
22 维基线 0.4376）。本实验检验：v4 特征下 LTBI 族（nhanes+treats+
brazil，11.8k）联合池化能否让 treats 从本地 v4 0.615 再上一阶。

机制披露（预注册，列并集的先验上限分析）：
  - 三队列 v4 extras 完全不相交（treats: smoking_status/roomshare_n/
    hhdens_quartile/hh_tb_contact/alcohol_use/country_sa；nhanes: bmi/
    smoking/born_us/income_pir/race_eth；brazil: ifng_genotype/ethnic），
    唯一共享 extras 列是 gender_female；
  - 结构缺失列（某队列无此源列）在列并集中按 0 填充——该列在缺源
    队列内为常数，对队列内判别无贡献（截距吸收）；结构缺失指示器与
    队列成员身份完全共线（随机截距 / 队列哑变量已吸收），按设计不加
    （用户处方的"缺失指示器"在并集语境下退化为队列身份，无增量信息）；
  - 因此臂 (b) 零样本的机制上限 = 共享列（22 维 + gender_female）的
    斜率迁移；臂 (c) 增广的增益来源 = 共享列斜率的族内信息增量 +
    队列专属 extras 的本地斜率（与本地 v4 相同）。

三臂设计：
  arm_a_local_v4：逐队列本地 v4 LR（F2 协议精确复现：同折组感知/
      分层 CV、折内中位填补、z 标准化、LR max_iter=2000）——基线
      （预期复现 treats 0.615 / nhanes 0.819 / brazil 0.690）。
  arm_b_family_loco_zero_shot：v4 列并集，其余两族成员训练
      CohortMixedModel（随机截距，VB 3 起点 ELBO 选优），留出队列
      总体截距零样本预测——方向对照（deployable=false，与 F1 的
      22 维族内零样本可比：差值 ≈ gender_female 单列增量）。
  arm_c_family_augmented（决定臂）：留出队列内同折 CV，每折训练集 =
      该队列训练折 + 其余两队列全量（v4 列并集 + 队列哑变量，参照 =
      留出队列），预测留出折——外源增广检验。与 arm_a 同折 → OOF
      逐样本配对。
主判据：arm_c − arm_a 同折配对 AUROC 差（组感知 cluster bootstrap
CI；treats 按 group_id 社区，其余按个体）。

用法：
    python data/run_ltbi_v4_pooling.py [--n-bootstrap 1000] [--folds 5]
        [--seed 42] [--n-starts 3]
输出：
    data/processed/ltbi_v4_pooling_<date>.json
"""
import argparse
import datetime
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

from tb_risk.scoring.predictor import MLRiskPredictor  # noqa: E402
from tb_risk.scoring.ml.training import ensure_interaction_features  # noqa: E402
from tb_risk.scoring.ml.validation import cluster_bootstrap_metric_ci  # noqa: E402
from tb_risk.scoring.ml.partial_pooling import (  # noqa: E402
    CohortMixedModel, STATSMODELS_AVAILABLE,
)
from tb_risk.scoring.ml.cohort_features import build_v4_extras  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

# LTBI 族注册表（F1 口径：文件 / 组列 / 终点）
COHORTS_LTBI = {
    'nhanes': {'file': 'ml_training_nhanes_real.csv', 'group_col': None,
               'endpoint': 'LTBI (IGRA)'},
    'treats': {'file': 'ml_training_treats_real.csv', 'group_col': 'group_id',
               'endpoint': 'LTBI (TST/IGRA)'},
    'brazil': {'file': 'ml_training_brazil_real.csv', 'group_col': None,
               'endpoint': 'LTBI (TST)'},
}

# 与 F2（run_cohort_v4_features.py）冻结同档：无正则搜索
LR_KWARGS = dict(max_iter=2000, random_state=42)


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def _fold_fill(X_tr, X_te):
    """折内中位数填补（训练折拟合；全 NaN 列填 0）——F2 同款。"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    return (np.where(np.isnan(X_tr), med, X_tr),
            np.where(np.isnan(X_te), med, X_te))


def _full_fill(X):
    """全数据中位数填补（臂 b/c 的常驻训练队列用）。"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    return np.where(np.isnan(X), med, X)


def _zscore(ref, *arrays):
    """以 ref 拟合 z 标准化（0 方差列除 1 保护）——F2 同款。"""
    mu, sd = ref.mean(0), ref.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return tuple((a - mu) / sd for a in arrays)


def _paired_bootstrap_delta(y, s_a, s_c, groups, n_bootstrap, seed):
    """AUROC(c) − AUROC(a) 的重采样 CI（组感知队列按组，否则按个体）。"""
    n = len(y)
    rng = np.random.RandomState(seed)
    if groups is not None:
        uniq = np.unique(groups)
        by_g = {g: np.where(groups == g)[0] for g in uniq}
    deltas = []
    for _ in range(n_bootstrap):
        idx = (np.concatenate([by_g[g] for g in
                               rng.choice(uniq, size=len(uniq), replace=True)])
               if groups is not None else
               rng.randint(0, n, size=n))
        if len(np.unique(y[idx])) < 2:
            continue
        a, c = _auc(y[idx], s_a[idx]), _auc(y[idx], s_c[idx])
        if a is None or c is None:
            continue
        deltas.append(c - a)
    if not deltas:
        return None, 0
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return [float(lo), float(hi)], len(deltas)


def load_ltbi_cohorts():
    """加载 LTBI 三队列：22 维 + per-cohort extras（NaN 保留）+ 并集对齐。

    返回 {cohort: {'X22','y','groups','own_cols','X_own','X_union',
    'n','n_pos'}}；X_union 列序 = 22 维 + extras 并集（注册表顺序去重），
    结构缺失列填 0（常数，截距吸收——见模块 docstring 机制披露）。
    """
    pred = MLRiskPredictor()
    names22 = list(pred.ALL_FEATURE_NAMES)
    data = {}
    union_cols = []
    for name, meta in COHORTS_LTBI.items():
        path = os.path.join(HERE, 'processed', meta['file'])
        df = pd.read_csv(path, encoding='utf-8-sig')
        df, missing = ensure_interaction_features(
            df, pred.INTERACTION_FEATURE_NAMES)
        X22 = df[names22].astype(float).to_numpy()
        y = df['tb_outcome'].astype(int).to_numpy()
        groups = (df[meta['group_col']].to_numpy()
                  if meta['group_col'] and meta['group_col'] in df.columns
                  else None)
        own_cols, X_own, _audit = build_v4_extras(df, name, fill='none')
        for c in own_cols:
            if c not in union_cols:
                union_cols.append(c)
        data[name] = {
            'X22': X22, 'y': y, 'groups': groups,
            'own_cols': list(own_cols), 'X_own': X_own,
            'n': len(y), 'n_pos': int(y.sum()),
        }
        print(f'  [{name}] n={len(y)}, pos={int(y.sum())} ({y.mean():.3f}), '
              f'extras={len(own_cols)} 列, groups='
              f'{(len(np.unique(groups)) if groups is not None else None)}')
    # 并集对齐：own extras 保留 NaN（折内/全量填补由各臂自行处理），
    # 结构缺失列填 0
    for name, d in data.items():
        own = dict(zip(d['own_cols'], range(d['X_own'].shape[1])))
        X_union = np.zeros((d['n'], len(union_cols)))
        for j, c in enumerate(union_cols):
            if c in own:
                X_union[:, j] = d['X_own'][:, own[c]]
        d['X_union'] = X_union
    return data, list(names22), union_cols


def _make_folds(y, groups, folds, seed):
    """与 F2 同款折生成（y/groups 决定折分配，与 X 无关 → 各臂同折）。"""
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True,
                                  random_state=seed)
        return list(cv.split(np.zeros(len(y)), y, groups)), \
            f'StratifiedGroupKFold({folds})'
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y)), f'StratifiedKFold({folds})'


def arm_a_local_v4(data, names22, folds, seed):
    """臂 (a)：逐队列本地 v4 LR（F2 协议精确复现，非并集矩阵）。

    返回 (results, oof_dict)：OOF 逐样本分数保留，供臂 (c) 同折配对。
    """
    results, oof_dict = {}, {}
    for name, d in data.items():
        X_v4 = np.hstack([d['X22'], d['X_own']])
        fold_list, cv_desc = _make_folds(d['y'], d['groups'], folds, seed)
        oof = np.full(d['n'], np.nan)
        for tr, te in fold_list:
            Xe_tr, Xe_te = _fold_fill(d['X_own'][tr], d['X_own'][te])
            Xtr, Xte = np.hstack([d['X22'][tr], Xe_tr]), \
                np.hstack([d['X22'][te], Xe_te])
            Ztr, Zte = _zscore(Xtr, Xtr, Xte)
            oof[te] = LogisticRegression(**LR_KWARGS).fit(
                Ztr, d['y'][tr]).predict_proba(Zte)[:, 1]
        assert not np.isnan(oof).any()
        results[name] = {'AUROC': _auc(d['y'], oof), 'cv': cv_desc,
                         'n_features': X_v4.shape[1]}
        oof_dict[name] = oof
        print(f'  [{name}] arm_a local v4 AUROC={results[name]["AUROC"]:.4f}')
    return results, oof_dict


def arm_b_family_loco_zero_shot(data, names22, union_cols, n_bootstrap,
                                seed, n_starts):
    """臂 (b)：v4 列并集族内 LOCO 零样本（随机截距，总体截距预测）。

    训练 = 其余两族成员（own extras 全量中位填补；结构缺失 0）；
    拟合前剔除训练池零方差列（peru 数值教训：常数列 → 奇异/秩亏），
    留出队列在同一列子集上预测（其 own extras 的真实值因训练池中
    零方差（结构缺失）而被剔除——机制上即共享列迁移）。
    """
    results = {}
    for held_out in COHORTS_LTBI:
        others = [n for n in COHORTS_LTBI if n != held_out]
        Xs, ys, cs = [], [], []
        for n in others:
            d = data[n]
            Xs.append(np.hstack([d['X22'], _full_fill(d['X_union'])]))
            ys.append(d['y'])
            cs.append(np.full(d['n'], n))
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        c = np.concatenate(cs)
        keep = np.where(X.std(axis=0) > 1e-12)[0]
        t0 = time.time()
        model = CohortMixedModel(
            feature_names=[names22[i] if i < len(names22)
                           else union_cols[i - len(names22)]
                           for i in keep]).fit(
            X[:, keep], y, c, fit_method='vb', random_state=seed,
            n_starts=n_starts)
        d = data[held_out]
        # 留出队列预测矩阵：own extras NaN 以本队列中位填补（保留列中
        # 仅共享列可能有 NaN；结构缺失列本就为 0 常数）
        Xh = np.hstack([d['X22'], _full_fill(d['X_union'])])[:, keep]
        p_zero = model.predict_proba(Xh, cohort=None)
        g = np.arange(d['n']) if d['groups'] is None else d['groups']
        r = cluster_bootstrap_metric_ci(p_zero, d['y'], g, metric='auroc',
                                        n_bootstrap=n_bootstrap, seed=seed)
        results[held_out] = {
            'AUROC': r['point'], 'AUROC_ci95': r['ci95'],
            'n_bootstrap_effective': r['n_effective'],
            'loco_train': others,
            'n_train': int(len(y)),
            'n_kept_columns': int(len(keep)),
            'dropped_zero_var_cols': [
                (names22[i] if i < len(names22)
                 else union_cols[i - len(names22)])
                for i in range(len(names22) + len(union_cols))
                if i not in set(keep.tolist())],
            'fit_method': model.fit_method,
            'elbo_selected': model.elbo_selected,
            'deployable': False,
            'deployment_note': '零样本臂仅作方向对照，禁止部署（项目硬约束）',
        }
        print(f'  [{held_out}] arm_b LOCO zero-shot AUROC={r["point"]:.4f} '
              f'CI95=[{r["ci95"][0]:.4f}, {r["ci95"][1]:.4f}] '
              f'(train={others}, kept {len(keep)}/{len(names22)+len(union_cols)} '
              f'cols) [{time.time()-t0:.0f}s]')
    return results


def arm_c_family_augmented(data, names22, union_cols, folds, seed,
                           n_bootstrap, oof_local):
    """臂 (c)（决定臂）：外源增广 LR——留出队列同折 CV + 其余队列全量。

    每折：训练 = 留出队列训练折（own extras 折内中位填补，与臂 (a)
    同口径）+ 其余两队列全量（own extras 全量中位填补）；v4 列并集 +
    队列哑变量（参照 = 留出队列，每常驻队列一列）；z 标准化以池化
    训练集拟合。与臂 (a) 同折（同 y/groups/seed）→ OOF 逐样本配对差。
    """
    results = {}
    n22 = len(names22)
    for held_out in COHORTS_LTBI:
        d = data[held_out]
        others = [n for n in COHORTS_LTBI if n != held_out]
        # 常驻队列全量并集矩阵 + 队列哑变量（第 k 个常驻队列 → 第 k 列 1）
        Xo = np.vstack([np.hstack([data[n]['X22'],
                                   _full_fill(data[n]['X_union'])])
                        for n in others])
        yo = np.concatenate([data[n]['y'] for n in others])
        co = np.zeros((Xo.shape[0], len(others)))
        _row = 0
        for k, n in enumerate(others):
            _nn = data[n]['n']
            co[_row:_row + _nn, k] = 1.0
            _row += _nn
        Xo_full = np.hstack([Xo, co])
        fold_list, cv_desc = _make_folds(d['y'], d['groups'], folds, seed)
        oof = np.full(d['n'], np.nan)
        for tr, te in fold_list:
            # 留出队列训练折/测试折：并集矩阵折内填补（臂 a 同款口径；
            # 结构缺失列本就 0 常数，折内填补不影响）+ 哑变量全 0（参照）
            Xe_tr, Xe_te = _fold_fill(d['X_union'][tr], d['X_union'][te])
            Xh_tr = np.hstack([d['X22'][tr], Xe_tr,
                               np.zeros((len(tr), len(others)))])
            Xh_te = np.hstack([d['X22'][te], Xe_te,
                               np.zeros((len(te), len(others)))])
            X_pool = np.vstack([Xh_tr, Xo_full])
            y_pool = np.concatenate([d['y'][tr], yo])
            Z_pool, Z_te = _zscore(X_pool, X_pool, Xh_te)
            oof[te] = LogisticRegression(**LR_KWARGS).fit(
                Z_pool, y_pool).predict_proba(Z_te)[:, 1]
        assert not np.isnan(oof).any()
        oof_a = oof_local[held_out]  # 臂 (a) 同折 OOF（配对）
        delta_ci, n_eff = _paired_bootstrap_delta(
            d['y'], oof_a, oof, d['groups'], n_bootstrap, seed)
        results[held_out] = {
            'AUROC': _auc(d['y'], oof),
            'cv': cv_desc,
            'paired_local_AUROC': _auc(d['y'], oof_a),
            'delta_augmented_minus_local': _auc(d['y'], oof) - _auc(d['y'], oof_a),
            'delta_ci95_bootstrap': delta_ci,
            'n_bootstrap_effective': n_eff,
            'n_held_out': d['n'],
            'n_augmenting': int(sum(data[n]['n'] for n in others)),
            'augmenting_cohorts': others,
            'n_features': n22 + len(union_cols) + len(others),
        }
        dd = results[held_out]['delta_augmented_minus_local']
        print(f'  [{held_out}] arm_c augmented AUROC={results[held_out]["AUROC"]:.4f} '
              f'(local {results[held_out]["paired_local_AUROC"]:.4f}, '
              f'delta {dd:+.4f}, CI95={delta_ci})')
    return results


def main():
    parser = argparse.ArgumentParser(description='LTBI 族 v4 联合池化（P3）')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-starts', type=int, default=3)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    if not STATSMODELS_AVAILABLE:
        print('错误：statsmodels 不可用（pip install statsmodels）')
        sys.exit(1)

    date_tag = datetime.date.today().strftime('%Y%m%d')
    out_path = args.out or os.path.join(
        HERE, 'processed', f'ltbi_v4_pooling_{date_tag}.json')

    t_start = time.time()
    print('== LTBI 族 v4 联合池化（列并集 + 结构缺失披露）==')
    print(f'--n-bootstrap={args.n_bootstrap}, --folds={args.folds}, '
          f'--seed={args.seed}, --n-starts={args.n_starts}')

    print('\n[1/4] 加载 LTBI 三队列（22 维 + per-cohort extras + 并集对齐）')
    data, names22, union_cols = load_ltbi_cohorts()
    shared = [c for c in union_cols
              if all(c in data[n]['own_cols'] for n in COHORTS_LTBI)]
    print(f'  并集 extras {len(union_cols)} 列；跨三队列共享：{shared}')
    print(f'  逐队列专属：' + ' | '.join(
        f'{n}: {len(data[n]["own_cols"])} 列' for n in COHORTS_LTBI))

    print('\n[2/4] 臂 (a)：逐队列本地 v4 LR（F2 协议复现）')
    arm_a, oof_local = arm_a_local_v4(data, names22, args.folds, args.seed)

    print('\n[3/4] 臂 (b)：v4 列并集族内 LOCO 零样本（方向对照）')
    arm_b = arm_b_family_loco_zero_shot(
        data, names22, union_cols, args.n_bootstrap, args.seed,
        args.n_starts)

    print('\n[4/4] 臂 (c)：外源增广 LR（决定臂，同折配对）')
    arm_c = arm_c_family_augmented(data, names22, union_cols, args.folds,
                                   args.seed, args.n_bootstrap, oof_local)

    # ---- 主判据汇总 ----
    verdicts = {}
    for name in COHORTS_LTBI:
        c = arm_c[name]
        ci = c['delta_ci95_bootstrap']
        delta = c['delta_augmented_minus_local']
        if ci is None:
            verdict = 'bootstrap 失效，无法判定'
        elif ci[0] > 0:
            verdict = ('池化显著增益（CI 排除 0）——treats 型部署配方应'
                       '切换为族内增广训练' if name == 'treats' else
                       '池化显著增益（CI 排除 0）')
        elif ci[1] < 0:
            verdict = '池化显著有害（CI 排除 0 于负侧）——维持本地 v4'
        else:
            verdict = '池化无显著增量（CI 含 0）——维持本地 v4（最简配方）'
        verdicts[name] = {
            'local_v4_AUROC': c['paired_local_AUROC'],
            'augmented_AUROC': c['AUROC'],
            'delta': delta, 'delta_ci95': ci,
            'loco_zero_shot_AUROC': arm_b[name]['AUROC'],
            'verdict': verdict,
        }

    output = {
        'experiment': 'ltbi_family_v4_union_pooling',
        'date': date_tag,
        'user_question': ('v4 特征下 LTBI 族联合池化能否让 treats 从本地 '
                          'v4 0.615 再上一阶（2026-09-16 用户指令 P3）'),
        'runtime_seconds': round(time.time() - t_start, 1),
        'mechanism_disclosure': {
            'extras_disjoint': ('三队列 v4 extras 完全不相交，唯一共享列 '
                                f'{shared}'),
            'structural_missing_rule': ('列并集中结构缺失列按 0 填充（队列内'
                                        '常数，截距吸收）；结构缺失指示器与'
                                        '队列身份完全共线，按设计不加'),
            'zero_shot_ceiling': ('臂 (b) 机制上限 = 共享列（22 维 + '
                                  'gender_female）斜率迁移；与 F1 22 维族内'
                                  '零样本的差 ≈ gender_female 单列增量'),
            'augmentation_source': ('臂 (c) 增益来源 = 共享列斜率的族内信息'
                                    '增量 + 队列专属 extras 本地斜率（后者与'
                                    '本地 v4 相同）'),
        },
        'protocol': {
            'feature_space': ('22 维基础（fillna(0)，主管线口径）+ v4 extras '
                              '并集（哨兵清洗 + 缺失指示器 + 折内/全量中位'
                              '填补，spec 单一真值源 ml/cohort_features.py）'),
            'arm_a': '逐队列本地 v4 LR（F2 协议精确复现）',
            'arm_b': ('v4 列并集族内 LOCO 零样本（CohortMixedModel 随机'
                      '截距 VB %d 起点；训练池零方差列剔除；总体截距预测；'
                      'deployable=false' % args.n_starts),
            'arm_c': ('外源增广 LR：留出队列同折 CV + 其余两队列全量 + '
                      '队列哑变量（参照=留出队列）；z 标准化以池化训练集'
                      '拟合；与臂 (a) 同折 → 逐样本配对差'),
            'lr_kwargs': LR_KWARGS,
            'folds': args.folds, 'seed': args.seed,
            'n_bootstrap': args.n_bootstrap,
        },
        'union_columns': {
            'base_22': names22,
            'extras_union': union_cols,
            'shared_across_all': shared,
            'per_cohort': {n: data[n]['own_cols'] for n in COHORTS_LTBI},
        },
        'cohorts': {
            n: {'n': data[n]['n'], 'n_pos': data[n]['n_pos'],
                'endpoint': COHORTS_LTBI[n]['endpoint'],
                'group_column': COHORTS_LTBI[n]['group_col']}
            for n in COHORTS_LTBI},
        'arm_a_local_v4': arm_a,
        'arm_b_family_loco_zero_shot': arm_b,
        'arm_c_family_augmented': arm_c,
        'verdicts': verdicts,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\n输出 → {out_path}（{time.time()-t_start:.0f}s）')

    print('\n== 判决速览（arm_c 增广 vs arm_a 本地 v4，同折配对）==')
    print(f'{"cohort":<10}{"local v4":>10}{"augmented":>11}{"delta":>9}'
          f'{"CI95":>22}  verdict')
    for name, v in verdicts.items():
        ci_s = (f'[{v["delta_ci95"][0]:+.4f}, {v["delta_ci95"][1]:+.4f}]'
                if v['delta_ci95'] else '—')
        print(f'{name:<10}{v["local_v4_AUROC"]:>10.4f}{v["augmented_AUROC"]:>11.4f}'
              f'{v["delta"]:>+9.4f}{ci_s:>22}  {v["verdict"]}')


if __name__ == '__main__':
    main()
