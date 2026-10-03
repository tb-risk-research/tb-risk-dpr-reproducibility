#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""peru_mdr PH 违反处置实验（缺陷2，2026-09-17）

背景：t7 主管线（cohort_v4_pipeline_20260916.json）Schoenfeld PH 检验
拒绝 3 个协变量的比例风险假设——has_tb p=0.0114 / is_high_risk p=0.0245
/ past_illness p=0.0478——静态 Cox 点估计仍按时间平均效应部署。
标准处置：违反变量作分层因子重拟合（时间漂移效应被层特异基线吸收），
看同层对集 OOF Harrell C 是否改善。

判决规则（预注册，survival_cox.stratified_cox_refit）：
  ΔC>0 且户级 cluster bootstrap CI 下界>0 → 分层版转正；
  否则静态版维持（HR 时间平均效应解读 + 固定窗分窗披露，现状）。

用法：
    python data/run_peru_ph_stratified.py [--strata has_tb,is_high_risk,past_illness]
        [--folds 5] [--bootstrap 1000] [--seed 42]
产出：data/processed/peru_ph_stratified_<date>.json
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tb_risk.scoring.predictor import MLRiskPredictor  # noqa: E402
from tb_risk.scoring.ml.training import (  # noqa: E402
    ensure_interaction_features, _get_git_hash)
from tb_risk.scoring.ml.cohort_features import (  # noqa: E402
    apply_v4_feature_space)
from tb_risk.scoring.ml.survival_cox import stratified_cox_refit  # noqa: E402

PERU_CSV = os.path.join(HERE, 'processed', 'ml_training_peru_mdr_real.csv')
TIME_COL = 'follow_up_days'
TARGET_COL = 'tb_outcome'
GROUP_COL = 'family_id'


def main():
    parser = argparse.ArgumentParser(description='peru_mdr PH 违反分层处置实验')
    parser.add_argument('--strata', type=str,
                        default='has_tb,is_high_risk,past_illness',
                        help='逗号分隔的 PH 违反变量（二元，作分层因子）')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--bootstrap', type=int, default=1000,
                        help='户级 cluster bootstrap 次数（0 关闭 CI）')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    strata_vars = [s.strip() for s in args.strata.split(',') if s.strip()]

    # 警告风暴抑制（2026-09-17 实测教训）：bootstrap 重采样内奇异
    # Hessian 的 HessianInversionWarning 逐行走 stderr，经沙箱管道
    # 节流后 8 小时墙钟仅推进 ~13 分钟 CPU（占空比 2.6%，照此需数天）。
    # bootstrap 路径只用点估计不用 cov_params，该警告为纯噪声；
    # 数值 RuntimeWarning 同理（NaN 列已由 fit_mask 结构性剔除）。
    warnings.filterwarnings('ignore', message='Inverting hessian failed')
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    t0 = time.time()
    pred = MLRiskPredictor()
    df = pd.read_csv(PERU_CSV, encoding='utf-8-sig')
    df, _ = ensure_interaction_features(df, pred.INTERACTION_FEATURE_NAMES)
    X, _raw, names, audit = apply_v4_feature_space(
        df, list(pred.ALL_FEATURE_NAMES), return_raw=True)
    if X is None:
        sys.exit(f'v4 队列未识别: {audit.get("cohort")}')
    print(f'[1/3] v4 特征空间: cohort={audit.get("cohort")} '
          f'{X.shape[1]} 列（22 基础 + {X.shape[1] - 22} extras）', flush=True)

    t_s = pd.to_numeric(df[TIME_COL], errors='coerce')
    ok = t_s.notna() & (t_s > 0)
    t = t_s[ok].to_numpy()
    e = pd.to_numeric(df.loc[ok, TARGET_COL],
                      errors='coerce').fillna(0).to_numpy().astype(int)
    X_ok = np.asarray(X)[ok.to_numpy()]
    groups = df.loc[ok, GROUP_COL].to_numpy()
    print(f'[2/3] 生存数据: n={len(t)} events={int(e.sum())} '
          f'户={len(np.unique(groups))}；分层变量: {strata_vars}', flush=True)

    res = stratified_cox_refit(
        t, e, X_ok, names, strata_vars, groups=groups,
        folds=args.folds, seed=args.seed, n_bootstrap=args.bootstrap,
        progress_every=25)
    res['n_samples'] = int(len(t))
    res['n_events'] = int(e.sum())
    res['n_households'] = int(len(np.unique(groups)))

    print(f'[3/3] 结果: C(静态|同层对)={res.get("c_static_within_strata_pairs"):.4f} '
          f'C(分层|同层对)={res.get("c_stratified_within_strata_pairs"):.4f} '
          f'ΔC={res.get("delta_c"):+.4f}', flush=True)
    if res.get('delta_c_ci95'):
        print(f'      ΔC bootstrap CI95: {res["delta_c_ci95"]} '
              f'(valid {res.get("n_bootstrap_valid")})', flush=True)
    ph = res.get('ph_retest') or {}
    glob = ph.get('global') if isinstance(ph, dict) else None
    if glob:
        print(f'      PH 复检（分层后剩余协变量）: global p='
              f'{glob.get("p_value"):.4f} 违反数={ph.get("n_violated_0.05")}',
              flush=True)
    elif isinstance(ph, dict) and ph.get('error'):
        print(f'      PH 复检不可用: {ph.get("error")}', flush=True)

    date_tag = datetime.date.today().strftime('%Y%m%d')
    out_path = args.out or os.path.join(
        HERE, 'processed', f'peru_ph_stratified_{date_tag}.json')
    archive = {
        'run': {
            'date': datetime.datetime.now().isoformat(timespec='seconds'),
            'script': os.path.basename(__file__),
            'git_commit': _get_git_hash(),
            'csv': os.path.basename(PERU_CSV),
            'runtime_s': round(time.time() - t0, 1),
        },
        'context': (
            't7 主管线 Schoenfeld PH 检验拒绝 3 协变量（has_tb 0.0114 / '
            'is_high_risk 0.0245 / past_illness 0.0478）；本实验把违反变量'
            '作分层因子重拟合，同层对集 OOF Harrell C 对比静态版。'
            '全对集静态 C（t7 harrell_c_oof=0.6777）口径不同不可直比。'),
        'result': res,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(archive, f, ensure_ascii=False, indent=2)
    print(f'档案已写入: {out_path}', flush=True)


if __name__ == '__main__':
    main()
