#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TBESC-II Part B（LTBI 级联）+ Carabayllo（接触评估级联）消化
（A2，2026-09-07）。

设计地位：两个级联数据集的落地侧（P1）消化 + Part B 大 n 宿主臂
感染排序（§8.7 结构的个体层锚）。

Part B（CDC TBESC-II 接触者级联，管道分隔 CSV，25,792 注册）：
  - 串行结构不存在（人均皮试 1.09 次）→ 无 SOP/时序先证路径；
  - 价值 1（级联完成率）：注册 → 皮试放置/读数/判读 → QFT →
    临床评估 → 治疗启动，逐级流失量化（部署级联设计参考）；
  - 价值 2（宿主臂锚）：TST 阳性（n≈15.5k 判读）与 QFT 阳性
    （n≈8.5k）的个体层排序——臂：人口学（年龄/性别/种族/出生地）
    vs +风险因子（HCW/惩教/流浪/移民/注射吸毒/吸烟/饮酒/DM/HIV）。
    无户键无暴露梯度——只补宿主臂锚，不构成终点依赖腿；
  - 感染患病率与亚组（出生国分组）描述。

Carabayllo（秘鲁利马接触评估，n=314 联系人 / 100 指示病例）：
  - 户簇结构存在（均值 3.14）但 PPD 完成率致命不足（73/314 =
    23% 读数）→ 簇内先证无功效，SOP 路径死；
  - 价值：评估级联流失表（PPD 放置 25% 是主要断点）+ 按
    指示病例类型（肺/肺外/MDR）的描述性感染率（73 读数，薄）。

预声明（Part B 排序）：
  - B1：人口学臂 TST 阳性 AUROC > 0.5（年龄/出生国携带感染
    患病率梯度——接触转诊人群的标准结构）；
  - B2：风险因子增量（B1 臂 + 风险）> 0——个体层风险代理
    携带增量（无暴露梯度下的上限参考）。

协议：StratifiedKFold(5) × 10 种子（大 n 稳定）；冻结注册表
RF；首测/首抽原则（人均 1.09 测）；个体 bootstrap ×1000
（末种子 OOF）；判读排除不确定（TST 99 / QFT 3,4）。

诚实边界：Part B 无户键/指示病例特征/暴露梯度——宿主臂锚
不可与 §8.7 各臂直接对齐（人群与特征空间不同）；级联完成率
受各站点上报差异影响；Carabayllo 感染读数 73 例为选择性
子集（完成评估者），指示性描述。

用法：
    python data/run_cascade_digests.py
输出：
    data/processed/partb_cascade_ranking_20260907.json
    data/processed/carabayllo_cascade_digest_20260907.json
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

from tb_risk.validation.real_data_pi import _make_model  # noqa: E402

PARTB_DIR = os.path.join(HERE, 'raw', 'cdc', 'tbesc2_partB',
                         'Export Data')
CARA_TAB = os.path.join(HERE, 'raw', 'dataverse',
                        'carabayllo_contact_evaluation.tab')
OUT_PARTB = os.path.join(HERE, 'processed',
                         'partb_cascade_ranking_%s.json'
                         % time.strftime('%Y%m%d'))
OUT_CARA = os.path.join(HERE, 'processed',
                        'carabayllo_cascade_digest_%s.json'
                        % time.strftime('%Y%m%d'))

N_SEEDS = 10
N_SPLITS = 5
N_BOOTSTRAP = 1000
SEED_START = 0


def _auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def rf_oof(X, y, seed):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        m = _make_model('random_forest', seed)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def bootstrap_delta(a, b, y, n_boot, seed):
    rng = np.random.RandomState(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yr = y[idx]
        if yr.sum() == 0 or yr.sum() == len(yr):
            continue
        deltas.append(_auroc(yr, a[idx]) - _auroc(yr, b[idx]))
    d = np.asarray(deltas)
    return {
        'mean': float(np.mean(d)),
        'bootstrap_ci': [float(np.percentile(d, 2.5)),
                         float(np.percentile(d, 97.5))],
        'ci_excludes_zero': bool(np.percentile(d, 2.5) > 0
                                 or np.percentile(d, 97.5) < 0),
    }


# ---------------- Part B ----------------

def load_partb():
    reg = pd.read_csv(os.path.join(PARTB_DIR, 'PatientRegistration.csv'),
                      sep='|', low_memory=False)
    st = pd.read_csv(os.path.join(PARTB_DIR, 'DiagnosticSkinTest.csv'),
                     sep='|', low_memory=False)
    qft = pd.read_csv(os.path.join(PARTB_DIR, 'DiagnosticQft.csv'),
                      sep='|', low_memory=False)
    rf = pd.read_csv(os.path.join(PARTB_DIR, 'PatientRiskFactor.csv'),
                     sep='|', low_memory=False)
    tx = pd.read_csv(os.path.join(PARTB_DIR, 'TreatmentLtbi.csv'),
                      sep='|', low_memory=False)
    ce = pd.read_csv(os.path.join(
        PARTB_DIR, 'PatientClinicalEval.csv'), sep='|', low_memory=False)
    return reg, st, qft, rf, tx, ce


def partb_analysis():
    reg, st, qft, rfk, tx, ce = load_partb()
    n_reg = len(reg)

    # ---- 级联完成率 ----
    st_sorted = st.sort_values(['PatientId', 'TstPlacedDate'])
    first_st = st_sorted.groupby('PatientId').first().reset_index()
    qft_sorted = qft.sort_values(['PatientId', 'BloodDrawDate'])
    first_qft = qft_sorted.groupby('PatientId').first().reset_index()
    cascade = {
        'registration': n_reg,
        'tst_placed': int(first_st['TstPlacedDate'].notna().sum()),
        'tst_read': int(first_st['TstResult'].notna().sum()),
        'tst_interpreted': int(first_st['TstInterpretation']
                               .isin([1, 2]).sum()),
        'qft_drawn': int(first_qft['BloodDrawDate'].notna().sum()),
        'qft_result': int(first_qft['QftResult'].isin([1, 2]).sum()),
        'clinical_eval': int(ce['PatientId'].nunique()),
        'riskfactor_form': int(rfk['PatientId'].nunique()),
        'ltbi_treatment_started': int(tx['PatientId'].nunique()),
    }

    # ---- 分析队列：首测 TST 判读（排除 99 不确定）----
    tst = first_st[first_st['TstInterpretation'].isin([1, 2])].copy()
    tst['y'] = (tst['TstInterpretation'] == 1).astype(int)
    # 合并注册表人口学 + 风险表（每患者首表）
    reg_first = reg.groupby('PatientId').first().reset_index()
    rf_first = rfk.groupby('PatientId').first().reset_index()
    a = tst[['PatientId', 'y']].merge(
        reg_first[['PatientId', 'd_age', 'd_sex', 'RaceWhite',
                   'RaceBlack', 'RaceAsian', 'IsLatino',
                   'BirthCountry']],
        on='PatientId', how='left').merge(
        rf_first[['PatientId', 'HealthcareWorker',
                  'CorrectionalFacilityResident', 'Homeless',
                  'MigrantWorker', 'InjDrugUser', 'Smoker',
                  'AlcoholUser', 'DiabetesMellitus']],
        on='PatientId', how='left')

    demo_cols = ['d_age', 'd_sex', 'RaceWhite', 'RaceBlack',
                 'RaceAsian', 'IsLatino']
    risk_cols = ['HealthcareWorker', 'CorrectionalFacilityResident',
                 'Homeless', 'MigrantWorker', 'InjDrugUser', 'Smoker',
                 'AlcoholUser', 'DiabetesMellitus']

    def run_endpoint(df, y_name, label, risk_subset=None):
        """demo 臂全队列；风险增量在 risk_subset（有风险表者，
        防表单完成选择泄漏——诊断：无表者 TST 阳 0.51% vs 有表者
        34.07%，表单完成与结局近乎共线）。88/99 哨兵清洗为 0
        （未采集/未知 → 无该风险）。"""
        def clean_risk(sub):
            sub = sub.copy()
            for c in risk_cols:
                v = pd.to_numeric(sub[c], errors='coerce')
                sub[c] = v.where(v.isin([0, 1]), 0.0)
            return sub

        ok = df[y_name].notna() & df['d_age'].notna()
        sub = df[ok].copy()
        for c in demo_cols:
            sub[c] = pd.to_numeric(sub[c], errors='coerce')
        sub[demo_cols] = sub[demo_cols].fillna(
            sub[demo_cols].median())
        y = sub[y_name].to_numpy(dtype=int)
        res = {'label': label, 'n': int(len(y)),
               'n_events': int(y.sum()),
               'prevalence': round(float(y.mean()), 4),
               'per_seed': [], 'arms_mean': {}}
        arms = {'demo': demo_cols}
        oof_store = {}
        for seed in range(SEED_START, SEED_START + N_SEEDS):
            row = {'seed': seed}
            for arm, cols in arms.items():
                oof = rf_oof(sub[cols].to_numpy(dtype=float), y, seed)
                row[arm] = round(_auroc(y, oof), 4)
                oof_store[(seed, arm)] = oof
            res['per_seed'].append(row)
        res['arms_mean'] = {k: round(float(np.mean(
            [r[k] for r in res['per_seed']])), 4) for k in arms}
        last = SEED_START + N_SEEDS - 1
        # 风险增量：有表子集内（防选择泄漏）
        if risk_subset is not None:
            sub2 = risk_subset[risk_subset[y_name].notna()
                               & risk_subset['d_age'].notna()].copy()
            for c in demo_cols:
                sub2[c] = pd.to_numeric(sub2[c], errors='coerce')
            sub2 = clean_risk(sub2)
            fill2 = sub2[demo_cols + risk_cols].median()
            sub2[demo_cols + risk_cols] = sub2[demo_cols + risk_cols] \
                .fillna(fill2)
            y2 = sub2[y_name].to_numpy(dtype=int)
            res['risk_subset'] = {
                'n': int(len(y2)), 'n_events': int(y2.sum()),
                'prevalence': round(float(y2.mean()), 4),
                'selection_warning': '有风险表者阳性率 34.1% vs 无表者 '
                                     '0.5%——表单完成与结局共线，风险'
                                     '增量仅在此选择性子集内可估',
            }
            arms2 = {'demo': demo_cols,
                     'demo_risk': demo_cols + risk_cols}
            oof2 = {}
            per_seed2 = []
            for seed in range(SEED_START, SEED_START + N_SEEDS):
                row = {'seed': seed}
                for arm, cols in arms2.items():
                    o = rf_oof(sub2[cols].to_numpy(dtype=float),
                               y2, seed)
                    row[arm] = round(_auroc(y2, o), 4)
                    oof2[(seed, arm)] = o
                per_seed2.append(row)
            res['risk_subset']['arms_mean'] = {
                k: round(float(np.mean([r[k] for r in per_seed2])), 4)
                for k in arms2}
            res['risk_subset']['risk_increment'] = bootstrap_delta(
                oof2[(last, 'demo_risk')], oof2[(last, 'demo')],
                y2, N_BOOTSTRAP, last)
        return res

    has_rf = a[risk_cols].notna().any(axis=1)
    tst_rank = run_endpoint(a, 'y', 'TST positivity (primary)',
                            risk_subset=a[has_rf])

    # QFT 次要
    q = first_qft[first_qft['QftResult'].isin([1, 2])].copy()
    q['y'] = (q['QftResult'] == 1).astype(int)
    aq = q[['PatientId', 'y']].merge(
        reg_first[['PatientId'] + demo_cols], on='PatientId',
        how='left').merge(
        rf_first[['PatientId'] + risk_cols], on='PatientId', how='left')
    has_rf_q = aq[risk_cols].notna().any(axis=1)
    qft_rank = run_endpoint(aq, 'y', 'QFT positivity (secondary)',
                            risk_subset=aq[has_rf_q])

    # 出生国描述
    a['us_born'] = a['BirthCountry'].astype(str).str.upper() \
        .isin(['US', 'USA', 'UNITED STATES'])
    birth_desc = {
        'n_with_country': int(a['BirthCountry'].notna().sum()),
        'us_born_n': int(a['us_born'].sum()),
        'us_born_prevalence': round(float(
            a.loc[a['us_born'], 'y'].mean()), 4),
        'non_us_prevalence': round(float(
            a.loc[~a['us_born'], 'y'].mean()), 4),
    }
    return {
        'cascade': cascade,
        'tst_ranking': tst_rank,
        'qft_ranking': qft_rank,
        'birth_country_descriptive': birth_desc,
    }


# ---------------- Carabayllo ----------------

def carabayllo_analysis():
    df = pd.read_csv(CARA_TAB, sep='\t')
    n = len(df)
    ppd_placed = df['PPDAPLACED'].notna().sum()
    ppd_placed_yes = int((df['PPDAPLACED'] == 1).sum())
    ppd_read = int(df['PPDLINDMM'].notna().sum())
    ppd_pos = int((pd.to_numeric(df['PPDLINDMM'], errors='coerce')
                   >= 10).sum())
    rad_done = int(df['RADDONEV0'].notna().sum())
    rad_abn = int((pd.to_numeric(df['RADRESULTV0'], errors='coerce')
                   == 2).sum())
    ptrx = int((df['PTRX'] == 1).sum())
    cluster_sizes = df.groupby('INDEXID').size()
    # 按指示病例类型描述（PPD 读数子集）
    sub = df[df['PPDLINDMM'].notna()].copy()
    sub['ppd_pos'] = (pd.to_numeric(sub['PPDLINDMM'],
                                    errors='coerce') >= 10).astype(int)
    by_index = sub.groupby('INDEXTBTYPE')['ppd_pos'].agg(
        ['size', 'sum', 'mean']).round(3)
    # 评估结果编码
    eval_v0 = df['EVALRESULTV0'].value_counts(dropna=False).to_dict()
    return {
        'n_contacts': n,
        'n_index_cases': int(df['INDEXID'].nunique()),
        'cluster_size': {'mean': round(float(cluster_sizes.mean()), 2),
                         'min': int(cluster_sizes.min()),
                         'max': int(cluster_sizes.max()),
                         'multi_member': int((cluster_sizes >= 2).sum())},
        'cascade': {
            'ppd_placed_any': int(ppd_placed),
            'ppd_placed_yes': ppd_placed_yes,
            'ppd_placed_rate': round(ppd_placed_yes / n, 3),
            'ppd_read': ppd_read,
            'ppd_read_rate': round(ppd_read / n, 3),
            'ppd_positive_ge10mm': ppd_pos,
            'xray_done_v0': rad_done,
            'xray_abnormal_v0': rad_abn,
            'preventive_treatment': ptrx,
            'main_gap': 'PPD 放置率 25% 为主要断点——簇内先证'
                        '无功效（73 读数 / 85 多员簇）',
        },
        'infection_by_index_type': {
            str(k): {'n_read': int(r['size']),
                     'ppd_pos': int(r['sum']),
                     'rate': float(r['mean'])}
            for k, r in by_index.iterrows()},
        'eval_result_v0_coding': {str(k): int(v)
                                  for k, v in eval_v0.items()},
        'disclosures': [
            'PPD 读数 73/314（23%）为完成评估的选择性子集——'
            '按指示病例类型的感染率为指示性描述',
            '年龄只有分类（00-04/05-19/20+），无连续年龄',
            '级联断点分析受站点上报差异影响',
        ],
    }


def main():
    t0 = time.time()
    print('=== Part B ===')
    partb = partb_analysis()
    print('cascade:', partb['cascade'])
    print('TST ranking:', partb['tst_ranking']['arms_mean'],
          'n=%d ev=%d' % (partb['tst_ranking']['n'],
                          partb['tst_ranking']['n_events']))
    rs = partb['tst_ranking']['risk_subset']
    print('  risk subset: n=%d prev=%.3f demo=%.4f demo_risk=%.4f '
          'inc=%+.4f CI %s' % (
              rs['n'], rs['prevalence'], rs['arms_mean']['demo'],
              rs['arms_mean']['demo_risk'],
              rs['risk_increment']['mean'],
              [round(x, 4) for x in rs['risk_increment']['bootstrap_ci']]))
    print('QFT ranking:', partb['qft_ranking']['arms_mean'],
          'n=%d ev=%d' % (partb['qft_ranking']['n'],
                          partb['qft_ranking']['n_events']))
    rsq = partb['qft_ranking']['risk_subset']
    print('  risk subset: n=%d prev=%.3f demo=%.4f demo_risk=%.4f '
          'inc=%+.4f CI %s' % (
              rsq['n'], rsq['prevalence'], rsq['arms_mean']['demo'],
              rsq['arms_mean']['demo_risk'],
              rsq['risk_increment']['mean'],
              [round(x, 4) for x in rsq['risk_increment']['bootstrap_ci']]))
    print('birth country:', partb['birth_country_descriptive'])

    out_pb = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'partb_cascade_ranking_v1',
        'status': '级联消化（P1）+ 宿主臂锚（探索性，预声明 B1/B2）；'
                  '无户键/指示病例/暴露梯度——不构成终点依赖腿',
        'design': {
            'source': 'CDC TBESC-II Part B（管道分隔 CSV，25,792 注册）',
            'serial_structure': '不存在（人均皮测 1.09 次）',
            'protocol': 'StratifiedKFold(%d) × %d 种子，冻结 RF，'
                        '首测原则，个体 bootstrap ×%d' % (
                            N_SPLITS, N_SEEDS, N_BOOTSTRAP),
        },
        'hypotheses': {
            'B1': '人口学臂 TST 阳性 AUROC > 0.5（年龄/出生国携带'
                  '感染患病率梯度）',
            'B2': '风险因子增量 > 0（个体层风险代理）——**仅在'
                  '有风险表子集内估计**（表单完成选择泄漏修正：'
                  '无表者 TST 阳 0.51% vs 有表者 34.07%，'
                  '全队列估计会得到 +0.35 的选择伪影）',
        },
        **partb,
    }
    with open(OUT_PARTB, 'w', encoding='utf-8') as f:
        json.dump(out_pb, f, ensure_ascii=False, indent=1)
    print('归档: %s' % OUT_PARTB)

    print('\n=== Carabayllo ===')
    cara = carabayllo_analysis()
    print('cascade:', cara['cascade'])
    print('by index type:', cara['infection_by_index_type'])
    out_c = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'carabayllo_cascade_digest_v1',
        'status': '级联消化（P1，描述性）——PPD 完成率不足封死'
                  '排序/先证路径',
        'design': {'source': 'Carabayllo 接触评估（dataverse，'
                             'n=314/100 指示病例）'},
        **cara,
    }
    with open(OUT_CARA, 'w', encoding='utf-8') as f:
        json.dump(out_c, f, ensure_ascii=False, indent=1)
    print('归档: %s (%.0fs)' % (OUT_CARA, time.time() - t0))


if __name__ == '__main__':
    main()
