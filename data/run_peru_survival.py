#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""peru_mdr 转生存分析：Cox PH + Harrell C + 固定窗 AUROC + HR 表（任务4，S9）

背景：peru_mdr（MDR 密切接触者，3406 行 / 149 事件 / 688 户）是 7 队列中
唯一的时间-事件（time-to-event）结构：follow_up_days 已在手（1–962 天，
事件中位 153d vs 截尾中位 424d），incident TB 天然是 time-to-event 终点。
终点族分池（family pooling）判定其为单成员 incident_tb 族——二分类池化
无族内 LOCO，生存分析是其路径（S7 归因结论）。本脚本也接回此前搁置的
P2（SOP/顺序结局先验在感染→进展两阶段的检验）所需的 time-to-event
基础设施：Cox 线性预测子即"进展风险排序"，后续可叠加先证序贯先验。

双模型设计（组感知 5 折 CV by family_id，OOF 评估）：
  (A) 主线对齐：22 维冻结特征空间（13 基础 + 9 交互）——与二分类管线
      口径对齐（arm_a 基线 AUROC 0.6296），回答"时间维度信息值多少"；
  (B) 族感知 v4 方向：22 维 + peru 专属先证/户级列（gender、
      mdr_household、index_smear_grade、index_smear_missing、
      index_cough_days[中位填补+缺失指示器]、index_hiv）——
      "被丢弃的队列信息列"方向在生存视图下的判别上行空间。

指标：
  - Harrell C（OOF，户级 cluster bootstrap CI）：连续时间判别；
  - 固定窗 AUROC（180d / 365d，OOF 风险分）：窗前截尾无事件者移出
    风险集（删失感知），与二分类 AUROC 口径可比；
  - HR 表（全数据拟合）：exp(β) + 95% CI（渐近）+ 户级 cluster
    bootstrap CI（重拟合）+ p 值，两模型均报，附 EPV 披露。

方法学披露：
  - 连续列训练折内 z 标准化 → HR 为"每 1 SD"口径；
  - PH 假设检验已补齐（P5，2026-09-16）：Schoenfeld 残差的
    Grambsch-Therneau score test（R cox.zph 同族，transform=log），
    逐变量 + 全局；拒绝（p<0.05）的协变量 HR 按时间平均效应解读；
  - 149 事件 / 10（A）或 15（B）拟合参数（22/29 列中 11 列零方差 +
    3 对完全共线剔除：highrisk_comorbid↔past_illness、mdr_household↔
    exposure_setting_score、index_smear_grade↔cumulative_exposure——
    peru 的暴露列本是先证列的派生函数）→ EPV 14.9 / 9.9，B 低于经典
    10 EPV 规则——HR 表按探索性解读，正式推断以 bootstrap CI 为准；
  - 家户聚集（ICC>0）：渐近 SE 低估不确定性，主证据用 cluster
    bootstrap CI。

数值机器（zfit/fit_mask/cox_fit/harrell_c/window_label/
cluster_bootstrap_ci/PH 检验）自 P5 起以 scoring/ml/survival_cox.py
为单一真值源（主管线生存路径与脚本实验层共享）。

用法：
    python data/run_peru_survival.py [--n-bootstrap 1000] [--hr-bootstrap 200]
        [--folds 5] [--seed 42]
输出：
    data/processed/peru_survival_<date>.json
"""
import argparse
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
from tb_risk.scoring.ml.survival_cox import (  # noqa: E402
    zfit, fit_mask, cox_fit, harrell_c, window_label,
    cluster_bootstrap_ci, schoenfeld_ph_test,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

PERU_CSV = os.path.join(HERE, 'processed', 'ml_training_peru_mdr_real.csv')

# 族感知 v4 增量列（构建规则）：peru 专属先证/户级信息——此前 22 维对齐
# 时被丢弃的队列信息列（任务4 与任务2 特征空间 v4 同方向）
PERU_EXTRA_SPEC = {
    'gender_female':   ('gender', 'binary_map', {'pos': 'f'}),
    'mdr_household':   ('mdr_household', 'as_is', None),
    'index_smear_grade': ('index_smear_grade', 'as_is', None),
    'index_smear_missing': ('index_smear_missing', 'as_is', None),
    'index_cough_days': ('index_cough_days', 'median_impute', None),
    'index_cough_missing': ('index_cough_days', 'missing_indicator', None),
    'index_hiv':       ('index_hiv', 'as_is', None),
}

WINDOWS = [180, 365]  # 固定窗（天）：6 个月 / 12 个月


def build_features(df):
    """构建模型 A（22 维）与模型 B（22+7 维）特征矩阵。

    返回 (names_A, X_A, names_B, X_B, impute_note)
    """
    pred = MLRiskPredictor()
    df2, missing = ensure_interaction_features(df, pred.INTERACTION_FEATURE_NAMES)
    names_a = list(pred.ALL_FEATURE_NAMES)
    X_a = df2[names_a].astype(float).to_numpy()

    parts_b = [pd.DataFrame(X_a, columns=names_a, index=df2.index)]
    names_b = list(names_a)
    for out_col, (src, mode, kw) in PERU_EXTRA_SPEC.items():
        s = df2[src]
        if mode == 'binary_map':
            col = (s.astype(str).str.lower() == kw['pos']).astype(float)
        elif mode == 'as_is':
            col = s.astype(float)
        elif mode == 'median_impute':
            col = s.astype(float).fillna(s.median())
        elif mode == 'missing_indicator':
            col = s.isna().astype(float)
        else:
            raise ValueError(mode)
        parts_b.append(pd.DataFrame({out_col: col.astype(float)}))
        names_b.append(out_col)
    X_b = pd.concat(parts_b, axis=1).to_numpy()
    impute_note = {'index_cough_days': {
        'n_missing': int(df2['index_cough_days'].isna().sum()),
        'impute': 'median', 'indicator': 'index_cough_missing'}}
    return names_a, X_a, names_b, X_b, impute_note


def run():
    parser = argparse.ArgumentParser(description='peru_mdr 生存分析（Cox PH）')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='C-index/固定窗 AUROC 的 cluster bootstrap 次数')
    parser.add_argument('--hr-bootstrap', type=int, default=200,
                        help='HR 表 cluster bootstrap 重拟合次数')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    date_tag = datetime.date.today().strftime('%Y%m%d')
    out_path = args.out or os.path.join(
        HERE, 'processed', f'peru_survival_{date_tag}.json')

    t_start = time.time()
    print('== peru_mdr 生存分析（Cox PH + Harrell C + 固定窗 AUROC + HR 表）==')
    print(f'--folds={args.folds}, --n-bootstrap={args.n_bootstrap}, '
          f'--hr-bootstrap={args.hr_bootstrap}, --seed={args.seed}')

    # ---- 1) 数据 ----
    df = pd.read_csv(PERU_CSV, encoding='utf-8-sig')
    names_a, X_a, names_b, X_b, impute_note = build_features(df)
    t = df['follow_up_days'].astype(float).to_numpy()
    e = df['tb_outcome'].astype(int).to_numpy()
    groups = df['family_id'].to_numpy()
    print(f'\n[1/5] 数据：n={len(t)}, 事件={int(e.sum())} ({e.mean():.3f}), '
          f'户={len(np.unique(groups))}')
    print(f'  follow_up_days: 中位 {np.median(t):.0f}d '
          f'[q25 {np.quantile(t,.25):.0f}, q75 {np.quantile(t,.75):.0f}]；'
          f'事件中位 {np.median(t[e==1]):.0f}d vs 截尾中位 {np.median(t[e==0]):.0f}d')
    print(f'  模型 A：{X_a.shape[1]} 维（22 维主线对齐）；'
          f'模型 B：{X_b.shape[1]} 维（+{X_b.shape[1]-X_a.shape[1]} peru 专属列）')

    # ---- 2) 组感知 CV OOF ----
    models = {'A_22dim': (names_a, X_a), 'B_family_aware': (names_b, X_b)}
    oof_risk = {}
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                              random_state=args.seed)
    folds = list(cv.split(X_a, e, groups))
    print(f'\n[2/5] 组感知 {args.folds} 折 CV（by family_id，事件分层）→ OOF 风险分')
    for mname, (mnames, X) in models.items():
        oof = np.full(len(t), np.nan)
        for tr, te in folds:
            mu, sd = zfit(X[tr])
            mask, _ = fit_mask(X[tr])
            Xtr = (X[tr] - mu) / sd
            res = cox_fit(t[tr], Xtr[:, mask], e[tr])
            beta = np.zeros(X.shape[1])          # 剔除列系数 0
            beta[mask] = res.params
            oof[te] = ((X[te] - mu) / sd) @ beta
        assert not np.isnan(oof).any()
        oof_risk[mname] = oof
        c = harrell_c(t, e, oof)
        print(f'  [{mname}] OOF Harrell C = {c:.4f}')

    # ---- 3) Harrell C + 固定窗 AUROC（cluster bootstrap CI）----
    print(f'\n[3/5] 判别指标（户级 cluster bootstrap × {args.n_bootstrap}）')
    disc = {}
    for mname in models:
        risk = oof_risk[mname]
        c_point = harrell_c(t, e, risk)
        ci, n_eff, _ = cluster_bootstrap_ci(
            lambda idx: harrell_c(t[idx], e[idx], risk[idx]),
            groups, args.n_bootstrap, args.seed)
        entry = {'harrell_c_oof': c_point,
                 'harrell_c_ci95_cluster': ci,
                 'n_bootstrap_effective': n_eff}
        for W in WINDOWS:
            keep, y_w = window_label(t, e, W)
            if len(np.unique(y_w)) < 2:
                entry[f'auroc_{W}d'] = None
                continue
            r_w, y_full = risk[keep], y_w
            a_point = float(roc_auc_score(y_full, r_w))
            a_ci, a_eff, _ = cluster_bootstrap_ci(
                lambda idx: (float(roc_auc_score(y_full[idx], r_w[idx]))
                             if len(np.unique(y_full[idx])) == 2 else None),
                groups[keep], args.n_bootstrap, args.seed)
            n_ev_w = int(y_full.sum())
            entry[f'auroc_{W}d'] = {
                'point': a_point, 'ci95_cluster': a_ci,
                'n_at_risk': int(keep.sum()), 'n_events_in_window': n_ev_w,
                'note': ('窗前截尾无事件者移出风险集（删失感知）；'
                         '窗后事件按非事件计（与二分类终点口径的差异披露）')}
            print(f'  [{mname}] AUROC({W}d)={a_point:.4f} '
                  f'CI95={["%.4f" % v for v in a_ci]} '
                  f'(at-risk={int(keep.sum())}, ev={n_ev_w})')
        disc[mname] = entry
        print(f'  [{mname}] Harrell C={c_point:.4f} '
              f'CI95={["%.4f" % v for v in ci]}')

    # A vs B 配对差（同折同 OOF 口径）
    delta_c = (harrell_c(t, e, oof_risk['B_family_aware'])
               - harrell_c(t, e, oof_risk['A_22dim']))
    d_ci, d_eff, d_mean = cluster_bootstrap_ci(
        lambda idx: (harrell_c(t[idx], e[idx], oof_risk['B_family_aware'][idx])
                     - harrell_c(t[idx], e[idx], oof_risk['A_22dim'][idx])),
        groups, args.n_bootstrap, args.seed)

    # ---- 4) HR 表（全数据拟合 + cluster bootstrap 重拟合）----
    print(f'\n[4/5] HR 表（全数据 Cox，Efron ties；户级 bootstrap 重拟合 × {args.hr_bootstrap}）')
    hr_tables = {}
    for mname, (mnames, X) in models.items():
        mu, sd = zfit(X)
        Xs = (X - mu) / sd
        mask, collinear = fit_mask(X)
        mask_idx = np.where(mask)[0]
        n_fit = int(mask.sum())
        res = cox_fit(t, Xs[:, mask], e)
        # 剔除列（死列/共线列）：系数 0 / HR 1 无信息，标注后跳过推断
        beta = np.zeros(X.shape[1])
        beta[mask] = res.params
        hr = np.exp(beta)
        ci_asym_live = np.exp(res.conf_int())
        ci_asym = np.full((X.shape[1], 2), np.nan)
        ci_asym[mask] = ci_asym_live
        pvals = np.full(X.shape[1], np.nan)
        pvals[mask] = res.pvalues
        # 户级 bootstrap：重采户 → 重拟合 → exp(β) 百分位（仅拟合列）
        uniq = np.unique(groups)
        by_g = {g: np.where(groups == g)[0] for g in uniq}
        rng = np.random.RandomState(args.seed + 1)
        boot_hr = np.full((args.hr_bootstrap, len(mnames)), np.nan)
        for b in range(args.hr_bootstrap):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([by_g[g] for g in pick])
            try:
                bmask, _ = fit_mask(X[idx])
                if not np.array_equal(bmask, mask):
                    # 重采样内剔除集变化 → 参数不可对齐，弃用该次
                    continue
                rb = cox_fit(t[idx], Xs[idx][:, bmask], e[idx])
                boot_hr[b, bmask] = np.exp(rb.params)
            except Exception:
                continue  # 重采样内奇异等数值失败 → 该次弃用（计数披露）
        n_ok = int(np.sum(~np.isnan(boot_hr[:, mask_idx[0]])))
        boot_ci = (np.nanpercentile(boot_hr, [2.5, 97.5], axis=0)
                   if n_ok else np.full((2, len(mnames)), np.nan))
        table = []
        for j, name in enumerate(mnames):
            if not mask[j]:
                if j in collinear:
                    table.append({
                        'feature': name, 'status': 'collinear_dropped',
                        'note': (f'与 {mnames[collinear[j]]} |r|>0.999'
                                 '（全数据）：完全共线，剔除保秩'),
                    })
                else:
                    table.append({
                        'feature': name, 'status': 'zero_variance_in_full_data',
                        'note': '全数据 sd=0（合成场景产物）：常数列，无信息',
                    })
                continue
            table.append({
                'feature': name, 'status': 'live',
                'coef_per_sd': float(beta[j]),
                'HR_per_sd': float(hr[j]),
                'HR_ci95_asymptotic': [float(ci_asym[j, 0]), float(ci_asym[j, 1])],
                'HR_ci95_cluster_bootstrap': [float(boot_ci[0, j]), float(boot_ci[1, j])],
                'p_value': float(pvals[j]),
            })
        epv = e.sum() / n_fit
        # PH 检验（P5，2026-09-16 补齐）：全量拟合的 Schoenfeld 残差
        # Grambsch-Therneau score test——F4 自披露缺陷的正式检验
        names_fitted = [mnames[j] for j in mask_idx]
        ph = schoenfeld_ph_test(res, t, e, names_fitted, transform='log')
        ph_viol = [v['feature'] for v in ph.get('per_variable', [])
                   if v.get('p_value') is not None and v['p_value'] < 0.05]
        hr_tables[mname] = {
            'n_params': len(mnames), 'n_params_fitted': n_fit,
            'events': int(e.sum()),
            'events_per_variable': round(epv, 2),
            'epv_note': (f'EPV={int(e.sum())}/{n_fit} 拟合参数（{n_fit}/'
                         f'{len(mnames)} 列进模型，零方差/共线列剔除）：'
                         'EPV<10 时 HR 表按探索性解读；正式推断以 cluster '
                         'bootstrap CI 为准（家户聚集下渐近 SE 低估不确定性）'),
            'n_hr_bootstrap_ok': n_ok,
            'ph_test': ph,
            'table': table,
        }
        sig = [r['feature'] for r in table if r['status'] == 'live'
               and (r['HR_ci95_cluster_bootstrap'][0] > 1.0
                    or r['HR_ci95_cluster_bootstrap'][1] < 1.0)]
        print(f'  [{mname}] 拟合列 {n_fit}/{len(mnames)}，EPV={epv:.1f}，'
              f'cluster CI 排除 1 的特征 ({len(sig)}): {sig}')
        print(f'  [{mname}] PH 检验: global p={ph["global"]["p_value"]:.4f}'
              if ph.get('global') else f'  [{mname}] PH 检验: 退化',
              f'（违反协变量 {ph_viol}）' if ph_viol else '（无 p<0.05 违反）')

    # ---- 5) 汇总输出 ----
    output = {
        'experiment': 'peru_mdr_survival_cox',
        'date': date_tag,
        'runtime_seconds': round(time.time() - t_start, 1),
        'protocol': {
            'model': 'Cox PH（statsmodels PHReg，Efron ties）',
            'feature_sets': {
                'A_22dim': '22 维冻结主线（13 基础 + 9 交互）',
                'B_family_aware': ('22 维 + peru 专属 7 列（gender / '
                                   'mdr_household / index_smear_grade / '
                                   'index_smear_missing / index_cough_days'
                                   '[中位填补+缺失指示器] / index_hiv）'),
            },
            'imputation': impute_note,
            'cv': (f'StratifiedGroupKFold({args.folds}) by family_id '
                   f'（{len(np.unique(groups))} 户，事件分层，seed={args.seed}）'),
            'standardization': '连续列训练折内 z 标准化 → HR 为每 1 SD 口径',
            'zero_variance_handling': (
                '零方差列（peru 上 22 维中 11 列 sd=0，合成场景产物）与完全'
                '共线列（3 对 r=+1.0：highrisk_comorbid↔past_illness、'
                'mdr_household↔exposure_setting_score、index_smear_grade↔'
                'cumulative_exposure——peru 的暴露列本是先证列的派生函数）'
                '在训练折/全数据内剔除后再进 Cox（常数列的 Hessian 零行/零列、'
                '共线列的秩亏均致矩阵奇异/MLE 不收敛）；剔除列系数置 0，'
                'HR 表标注 zero_variance_in_full_data / collinear_dropped；'
                'EPV 按实际拟合参数数计算'),
            'harrell_c': 'OOF 风险分；事件者 vs 更长观察时间者；风险并列 0.5',
            'fixed_window': ('删失感知：窗前截尾无事件移出风险集；窗后事件'
                             '按非事件计（与二分类终点口径差异披露）'),
            'ph_assumption_note': ('PH 假设正式检验已补齐（P5，2026-09-16）：'
                                   'Schoenfeld 残差 Grambsch-Therneau score '
                                   'test（R cox.zph 同族，transform=log），'
                                   '逐变量 + 全局，见 hr_tables.*.ph_test；'
                                   '拒绝（p<0.05）的协变量 HR 按时间平均'
                                   '效应解读，判别以 180d vs 365d 固定窗 '
                                   'AUROC 稳定性交叉佐证'),
            'epv_note': (f'{int(e.sum())} 事件：A 拟合 {int(fit_mask(X_a)[0].sum())}/22 列、'
                         f'B 拟合 {int(fit_mask(X_b)[0].sum())}/29 列（零方差/共线'
                         '剔除），EPV 按拟合参数计'
                         '（<10：HR 探索性，正式推断用 cluster bootstrap CI）'),
            'context_baseline': ('二分类 arm_a 基线 AUROC=0.6296 '
                                 '（multicohort_pooling_20260914，组感知 CV）'
                                 '——固定窗 AUROC 与其口径可比（删失处理差异'
                                 '见 fixed_window）'),
            'p2_connection': ('接回搁置的 P2（SOP/顺序结局先验的感染→进展'
                              '检验）：Cox 线性预测子即进展风险排序，'
                              'time-to-event 基础设施就位后可叠加先证序贯先验'),
        },
        'data': {
            'n': int(len(t)), 'events': int(e.sum()),
            'prevalence': round(float(e.mean()), 4),
            'n_households': int(len(np.unique(groups))),
            'follow_up_days': {
                'median': float(np.median(t)),
                'q25': float(np.quantile(t, .25)),
                'q75': float(np.quantile(t, .75)),
                'max': float(t.max()),
                'median_event': float(np.median(t[e == 1])),
                'median_censored': float(np.median(t[e == 0]))},
            'endpoint': 'incident TB（MDR 密切接触者，随访期内发病）',
        },
        'discrimination': disc,
        'paired_delta_B_minus_A': {
            'harrell_c': delta_c,
            'ci95_cluster': d_ci,
            'n_bootstrap_effective': d_eff,
            'note': '同折同 OOF 口径配对差（B 族感知 − A 主线）',
        },
        'hr_tables': hr_tables,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\n输出 → {out_path}（{time.time()-t_start:.0f}s）')

    # 控制台速览
    print('\n== 速览 ==')
    print(f'{"model":<16}{"Harrell C":>10}{"AUROC(180d)":>13}{"AUROC(365d)":>13}')
    for mname in models:
        d = disc[mname]
        a180 = d.get('auroc_180d') or {}
        a365 = d.get('auroc_365d') or {}
        print(f'{mname:<16}{d["harrell_c_oof"]:>10.4f}'
              f'{(a180.get("point") or float("nan")):>13.4f}'
              f'{(a365.get("point") or float("nan")):>13.4f}')
    print(f'B−A Harrell C: {delta_c:+.4f} CI95={["%.4f" % v for v in d_ci]}')


if __name__ == '__main__':
    run()
