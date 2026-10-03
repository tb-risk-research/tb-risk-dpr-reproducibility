#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""韩国 HCW 串行 TST：SOP 工作场所版先证方向检验（A1，2026-09-07）。

设计地位：除 HomeACF 外第二个有时序结构的感染终点队列——但分组
单元从**户**（~3 人/户，强共享暴露）变为**医院科室**（~100 人/站，
弱共享暴露），终点从**患病率阳性**（横断面 TST）变为**发病率阳转**
（轮间硬结差 ≥6mm）。这是 §8.4-P5 条件 (b)"同分组粒度"的**边界
探针**：P5 预测先证块增益随组内共享暴露依赖度单调——若粒度从户
粗化为站点、暴露共享大幅稀释，站点先证增益应远弱于 HomeACF
的 +0.0585（甚至 null）。

数据：Severance 医院 HCW 年度筛查队列（dataverse，n=458；
H1 2009 / H2 2011 / H3 2012 / H4 2013 四轮 TST + IGRA 子集 80）。
病程终点死亡（Active_TB_develop 仅 2 事件，不分析——PACTS 9
事件教训的同款披露）。

队列（主）：H2CONV 评估者 = H1 阴性且参加 H2 轮（n=89，阳转 27，
窗口 2009→2011 约 2 年）。
队列（次）：H4CONV 评估者（n=69，阳转 12，2012→2013 窗口）——
事件更少，仅方向参考；H3CONV（5 事件）不做判别分析。

臂结构：
  host         年龄、性别、BMI、BCG、DM、吸烟（0/1/2 序）
  serial       + 自身 H1 硬结 mm（阴性范围内早期应答梯度）、
               工龄、职业（医/护/助理等 6 类 one-hot）
  prior        + 站点 H1 阳性率（**他人早期结果**：LOO =
               sum_site/(n_site−1)，H1 阴性者自证 0 贡献——
               站点常量 4 水平）
  prior_only   纯站点先证排序（方向检查）
站点（Work_location 门诊/急诊/住院/ICU）不进宿主臂——其信息
完全由先证臂承载（预声明，防止站点方差被宿主臂吸收）。

预声明假设：
  - K1（串行梯度）：H1 阴性范围内的基线硬结尺寸正向预测阳转
    （早期应答谱：boost 相邻现象的连续版；勘察已见阳转者
    3.7mm vs 非阳转 1.7mm——方向已知，判别层为新增检验）；
  - K2（SOP 站点先证，主）：站点 H1 阳性率对 host+serial 的
    增量。P5 粒度边界预测：**预期弱/null**（站点暴露共享 ≪
    户内共享）——K2 CI 含零与 P5 一致，不含零且正向则站点
    携带跨户传输信号（信息量更大的结果）；
  - K3（描述）：IGRA-TST 一致性（n=80 子集）；新雇员两步法
    boost 现象（Conversion_1）；站点级阳转率表。

协议：StratifiedKFold(5) 个体 × 20 种子（n=89 不支持按站点
分组 CV——站点即特征），冻结注册表 RF 主 + LR 敏感性；折内
中位数填补；末种子 OOF 个体 bootstrap ×2000（无簇键——站点
仅 4 簇不支持 cluster bootstrap，个体 bootstrap 披露）；逐
种子 Δ 四栏（R5a 纪律）。

诚实边界：
  - n=89 / 27 事件：方向级检验（80% 功效仅够 ΔAUROC ≳0.10），
    非效应量锚；不进任何合并/合成表；
  - H2CONV 评估为选择性亚组（89/~348 H1 阴性者参加 H2 轮）——
    回访选择机制未检验（MAR 未验证）；
  - 阳转定义 = 硬结差 ≥6mm（编码本），含 boost/远程感染谱
    混合，非纯新发感染；
  - 站点先证 = 轮间结构（2009 基线轮 → 2011 阳转），与
    HomeACF 户内横断面先证（同轮筛查序）时序结构不同——
    两者同为"他人早期结果"但不可逐位对齐，只做方向对照；
  - 单医院单队列；科室 4 个有效水平。

用法：
    python data/run_korea_hcw_sop.py
输出：
    data/processed/korea_hcw_sop_20260907.json
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

XLSX = os.path.join(HERE, 'raw', 'dataverse', 'korea_hcw_tst_igra.xlsx')
OUT = os.path.join(HERE, 'processed',
                   'korea_hcw_sop_%s.json' % time.strftime('%Y%m%d'))

N_SEEDS = 20
N_SPLITS = 5
N_BOOTSTRAP = 2000
SEED_START = 0

HOST_COLS = ['age', 'male', 'bmi', 'bcg', 'dm', 'smoke']
SERIAL_EXTRA = ['h1_size', 'work_dur_y', 'wg_1', 'wg_2', 'wg_3',
                'wg_4', 'wg_5']
PRIOR_COL = 'site_prior'
FULL_COLS = HOST_COLS + SERIAL_EXTRA + [PRIOR_COL]


def _num(s):
    return pd.to_numeric(s, errors='coerce')


def load_cohort():
    df = pd.read_excel(XLSX, sheet_name='Sheet1')
    d = pd.DataFrame(index=df.index)
    d['sex_raw'] = _num(df['Sex'])          # 1=Male 2=Female
    d['male'] = (d['sex_raw'] == 1).astype(float)
    d['age'] = _num(df['Age_y'])
    d['bmi'] = _num(df['BMI'])
    d['bcg'] = _num(df['BCG_Hx'])
    d['dm'] = _num(df['DM'])
    d['smoke'] = _num(df['Smokestatus'])   # 0/1/2 序
    d['h1_size'] = _num(df['H1_Size'])
    d['work_dur_y'] = _num(df['Work_dur_y'])
    wg = _num(df['Work_g_sub'])
    for k in range(1, 6):
        d['wg_%d' % k] = (wg == k).astype(float)
    d['site'] = _num(df['Work_location'])   # 1-4 有效
    d['h1_r'] = _num(df['H1_R'])
    d['h2_r'] = _num(df['H2_R'])
    d['h3_r'] = _num(df['H3_R'])
    for conv in ('H2CONV', 'H3CONV', 'H4CONV'):
        d[conv.lower()] = _num(df[conv])    # blank -> NaN
    d['h4_size'] = _num(df['H4_Size'])
    d['h3_size'] = _num(df['H3_Size'])
    d['conversion_1'] = _num(df['Conversion_1'])
    d['igra_r'] = _num(df['IGRA_R'])
    d['fu_months'] = _num(df['FU_months'])
    return d


def build_analysis(d, conv_col):
    """某轮阳转分析队列 + 站点先证（H1 基线轮锚，全队列覆盖）。"""
    ok = d[conv_col].notna() & d['site'].notna()
    sub = d[ok].copy()
    y = sub[conv_col].to_numpy(dtype=int)
    # 站点 H1 阳性率（全队列，LOO：eligible 全为 H1 阴性 → sum/(n−1)）
    site_tab = d[d['site'].notna()].groupby('site')['h1_r'].agg(
        ['size', 'sum'])
    rate_map = (site_tab['sum'] / (site_tab['size'] - 1)).to_dict()
    sub['site_prior'] = sub['site'].map(rate_map).astype(float)
    return sub, y, rate_map, site_tab


def _auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def rf_oof(X, y, seed, model_key='random_forest'):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=seed)
    X = np.nan_to_num(X, nan=0.0)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        m = _make_model(model_key, seed)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def impute_fold_median(X):
    """全表中位数填补（n=89 小样本——折内中位数在小折不稳，
    改全表中位数并披露；缺失仅 h1_size/work_dur_y 少量）。"""
    X = X.copy()
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    return X


def bootstrap_delta(a, b, y, n_boot, seed):
    """个体 bootstrap 配对 ΔAUROC CI。"""
    rng = np.random.RandomState(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yr = y[idx]
        if yr.sum() == 0 or yr.sum() == len(yr):
            continue
        deltas.append(_auroc(yr, a[idx]) - _auroc(yr, b[idx]))
    deltas = np.asarray(deltas)
    return {
        'mean': float(np.mean(deltas)),
        'bootstrap_ci': [float(np.percentile(deltas, 2.5)),
                         float(np.percentile(deltas, 97.5))],
        'p_positive': float((deltas > 0).mean()),
        'ci_excludes_zero': bool(np.percentile(deltas, 2.5) > 0
                                 or np.percentile(deltas, 97.5) < 0),
    }


def run_round(d, conv_col, label, model_key):
    sub, y, rate_map, site_tab = build_analysis(d, conv_col)
    if y.sum() < 8:
        return {'label': label, 'skipped': 'events<8'}
    X = impute_fold_median(sub[FULL_COLS])
    arms = {
        'host': HOST_COLS,
        'serial': HOST_COLS + SERIAL_EXTRA,
        'full': FULL_COLS,
        'prior_only': [PRIOR_COL],
    }
    res = {
        'label': label,
        'n': int(len(y)), 'n_events': int(y.sum()),
        'site_prior_levels': {int(k): round(v, 4)
                              for k, v in rate_map.items()},
        'site_table': {str(int(k)): {'n': int(r['size']),
                                     'h1_pos': int(r['sum'])}
                       for k, r in site_tab.iterrows()},
        'per_seed': [], 'arms_mean': {},
    }
    oof_store = {}
    for seed in range(SEED_START, SEED_START + N_SEEDS):
        row = {'seed': seed}
        for arm, cols in arms.items():
            oof = rf_oof(X[cols].to_numpy(dtype=float), y, seed,
                         model_key)
            row[arm] = round(_auroc(y, oof), 4)
            oof_store[(seed, arm)] = oof
        res['per_seed'].append(row)
    for arm in arms:
        res['arms_mean'][arm] = round(float(np.mean(
            [r[arm] for r in res['per_seed']])), 4)
    # 末种子 OOF bootstrap 对比
    last = SEED_START + N_SEEDS - 1
    contrasts = {}
    for a, b, note in [
            ('serial', 'host', 'K1：串行梯度增量'),
            ('full', 'serial', 'K2：站点先证增量（主）'),
            ('full', 'host', 'K1+K2 合并增量'),
            ('prior_only', 'host', 'K2b：纯先证 vs 纯宿主')]:
        cb = bootstrap_delta(oof_store[(last, a)], oof_store[(last, b)],
                             y, N_BOOTSTRAP, last)
        per_seed_d = [r[a] - r[b] for r in res['per_seed']]
        cb['per_seed_delta'] = {
            'mean': round(float(np.mean(per_seed_d)), 4),
            'min': round(float(np.min(per_seed_d)), 4),
            'max': round(float(np.max(per_seed_d)), 4),
            'share_positive': round(float(
                (np.asarray(per_seed_d) > 0).mean()), 3),
        }
        cb['note'] = note
        contrasts['%s_minus_%s' % (a, b)] = cb
    res['contrasts'] = contrasts
    return res


def main():
    t0 = time.time()
    d = load_cohort()
    print('Korea HCW n=%d' % len(d))
    for conv in ('h2conv', 'h3conv', 'h4conv'):
        print('  %s assessed=%d events=%d' % (
            conv, d[conv].notna().sum(), int(d[conv].sum())))

    # H2 轮覆盖率披露
    print('  H1_R/H2_R/H3_R nonnull: %d/%d/%d' % (
        d['h1_r'].notna().sum(), d['h2_r'].notna().sum(),
        d['h3_r'].notna().sum()))

    primary = run_round(d, 'h2conv', 'H2CONV 2009->2011 (primary)',
                        'random_forest')
    print('\n=== H2CONV primary (RF) ===')
    print('  arms:', primary['arms_mean'])
    for k, v in primary['contrasts'].items():
        print('  %-20s %+0.4f CI[%+0.4f,%+0.4f] share+=%.2f' % (
            k, v['mean'], *v['bootstrap_ci'],
            v['per_seed_delta']['share_positive']))

    primary_lr = run_round(d, 'h2conv', 'H2CONV (LR sensitivity)',
                           'logistic')
    print('\n=== H2CONV LR sensitivity ===')
    print('  arms:', primary_lr['arms_mean'])
    for k, v in primary_lr['contrasts'].items():
        print('  %-20s %+0.4f CI[%+0.4f,%+0.4f]' % (
            k, v['mean'], *v['bootstrap_ci']))

    secondary = run_round(d, 'h4conv', 'H4CONV 2012->2013 (secondary)',
                          'random_forest')
    print('\n=== H4CONV secondary (RF) ===')
    if 'contrasts' in secondary:
        print('  arms:', secondary['arms_mean'])
        for k, v in secondary['contrasts'].items():
            print('  %-20s %+0.4f CI[%+0.4f,%+0.4f]' % (
                k, v['mean'], *v['bootstrap_ci']))
    else:
        print('  skipped:', secondary.get('skipped'))

    # K3 描述层
    ig = d[d['igra_r'].notna() & d['h1_r'].notna()]
    agree = {
        'n_both': int(len(ig)),
        'tst_pos_igra_pos': int(((ig['h1_r'] == 1) &
                                 (ig['igra_r'] == 1)).sum()),
        'tst_pos_igra_neg': int(((ig['h1_r'] == 1) &
                                 (ig['igra_r'] == 0)).sum()),
        'tst_neg_igra_pos': int(((ig['h1_r'] == 0) &
                                 (ig['igra_r'] == 1)).sum()),
        'tst_neg_igra_neg': int(((ig['h1_r'] == 0) &
                                 (ig['igra_r'] == 0)).sum()),
    }
    boost = {
        'newemp_2step': int((_num(pd.read_excel(
            XLSX, sheet_name='Sheet1')['NewEmp_2stepTST']) == 1).sum()),
        'conversion_1_boosters': int(
            (d['conversion_1'] == 1).sum()),
        'note': '两步法 boost 现象（Conversion_1）——短窗增强，'
                '非传播信号；主分析用轮间阳转',
    }
    print('\n=== K3 描述 ===')
    print('  IGRA-TST 2x2:', agree)
    print('  boost:', {k: v for k, v in boost.items() if k != 'note'})

    out = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'korea_hcw_sop_v1',
        'status': '探索性方向分析（预声明 K1/K2/K3）；P5 粒度边界'
                  '探针——n=89/27 事件为方向级，不进合并/合成表',
        'design': {
            'source': 'Severance 医院 HCW 年度筛查（dataverse，n=458）',
            'endpoint': 'TST 阳转（轮间硬结差≥6mm）；主=H2CONV '
                        '(2009->2011)，次=H4CONV (2012->2013)',
            'progression_endpoint': '死亡：Active_TB_develop 仅 2 事件',
            'n_seeds': N_SEEDS, 'cv': 'StratifiedKFold(%d) 个体' % N_SPLITS,
            'ci': '个体 bootstrap ×%d（末种子 OOF；站点仅 4 簇不'
                  '支持 cluster bootstrap）' % N_BOOTSTRAP,
            'prior_def': '站点 H1 阳性率 LOO=sum_site/(n_site-1) '
                         '（他人基线轮结果；站点常量）',
            'site_not_in_host': '预声明：站点信息仅由先证臂承载',
            'imputation': '全表中位数（n=89 小样本折内中位数不稳）',
        },
        'hypotheses': {
            'K1': 'H1 阴性范围内基线硬结 mm 正向预测阳转（早期应答梯度）',
            'K2': '站点先证增量（主）——P5 粒度边界预测弱/null：站点'
                  '暴露共享≪户内；若显著正向则站点携带跨户传输信号',
            'K3': '描述：IGRA-TST 一致性 + boost 现象',
        },
        'primary_rf': primary,
        'primary_lr': primary_lr,
        'secondary_h4': secondary,
        'descriptives': {'igra_tst_2x2': agree, 'booster': boost},
        'disclosures': [
            'n=89/27 事件：方向级检验（80% 功效仅够 ΔAUROC≳0.10），'
            '非效应量锚',
            'H2CONV 评估为选择性亚组（89/~348 H1 阴性者回访）——'
            '选择机制未检验',
            '阳转定义含 boost/远程感染谱混合，非纯新发感染',
            '站点先证=轮间结构（2009→2011），与 HomeACF 户内横断面'
            '先证时序结构不同，只做方向对照',
            '单医院、4 个有效科室水平',
        ],
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n归档: %s (%.0fs)' % (OUT, time.time() - t0))
    return out


if __name__ == '__main__':
    main()
