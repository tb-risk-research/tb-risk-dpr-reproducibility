#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SINAN SITUA_ENCE 标签字典判定与归档（第七轮，2026-08-26）。

背景：round-5 起的实验把 SITUA_ENCE=='4' 当作「TB 死亡」。第七轮
构造老年多终点实验时发现该假设与官方字典不符。本脚本用三重证据
裁定编码语义，并把判定过程完整归档（证据链可复现）。

证据一（权威字典，巴西卫生部 SINAN Net 5.0 官方数据字典 PDF）：
  Campo 62 tp_situacao_encerramento（SITUA_ENCE）：
    1  Cura                    治愈
    2  Abandono                放弃治疗（失访）
    3  Óbito por TB            TB 死亡
    4  Óbito por outras causas 非 TB 死亡
    5  Transferência           转出
    6  Mudança de Diagnóstico  更改诊断
    7  TB-DR                   耐药结核
    8  Mudança de Esquema      更换方案
    9  Falência                治疗失败
    10 Abandono Primário       原发放弃
  来源：DICI_DADOS_NET_Tuberculose_23_07_2020.pdf
  （portalsinan.saude.gov.br，已存 data/raw/dici_tb.pdf）。

证据二（文献佐证，独立样本同分布）：
  - 东北五州 2012-2021（184,657 例）：治愈 63.9% / 放弃 9.1% /
    TB 死亡 3.2% / 非 TB 死亡 3.8% / 转出 9.1%（Lima et al. 2024，
    Medicina Ribeirão Preto 57(2)）
  - 巴西监狱系统 2014（Rev Saude Publica 2015;49:66）：
    TB 死亡与非 TB 死亡分开统计
  - 巴西卫生部死亡调查表 2013 版编码表与 SINAN Net 5.0 一致。

证据三（数据内证，本仓库 168.5 万通报记录上可复现）：
  (a) 值 5（n=111,407）年龄结构与治愈组完全一致
      （65+ 占比 8.3% vs 8.1%，年龄中位 36 vs 37）且跨
      2001-05/2006-12/2013-19 三个时期稳定——行政性转出特征，
      排除「死亡类」（死亡随年龄陡增，65+ 占比必 ≥20%）。
  (b) 值 3 在 2001-2005 几乎未用（0.1%），2006/2007 起跳升至
      3.7% 后稳定——与 SINAN Net 推广启用「TB 死亡」细分编码的
      时点吻合；同期值 4 从 6.6% 降至 4.4%（部分死亡分流至 3）。
  (c) 值 3/4 在全部三个时期均呈死亡年龄特征（65+ 22-26%），
      值 2 呈放弃治疗特征（最年轻、酒精/毒品最高）。
  (d) 2007-19 期间：值 3 结案中位 14 天（暴发性早期死亡）、
      值 4 结案中位 34 天、涂阳率 65.6%（远高于治愈 45.7%，
      细菌负荷与死亡的剂量-反应存在，且 4 类含大量 HIV 相关
      死亡：AIDS 占比 40%）。

结论与影响：
  1. round-5/6 的 y=(SITUA_ENCE=='4') 是「非 TB 死亡」（2006+）
     与「全因死亡」（2001-05，值 3 未启用期间死亡几乎全录 4）
     的混合标签，非「TB 死亡」。
  2. 受影响归档：sinan_scale_training / sinan_elderly /
     sinan_temporal_prior（*_20260826.json）的终点标签语义。
     机制层结论（arm 间差值、特征排序）不受影响——所有 arm
     共享同一 y；对外叙事中的「死亡终点」须更正为
     「非 TB 死亡终点（2001-05 段含全因死亡）」。
  3. 修正终点定义：全因死亡 = {3,4}（稳健，不依赖 3/4 细分）；
     TB 死亡 = {3}（仅 2006+ 语义一致）。

归档：data/processed/sinan_label_dictionary_20260826.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

PROC = os.path.join(BASE, 'tb_risk', 'data', 'processed')
OUT = os.path.join(PROC, 'sinan_label_dictionary_20260826.json')

# 官方字典（证据一）
OFFICIAL = {
    '1': 'Cura（治愈）',
    '2': 'Abandono（放弃治疗/失访）',
    '3': 'Óbito por TB（TB 死亡）',
    '4': 'Óbito por outras causas（非 TB 死亡）',
    '5': 'Transferência（转出）',
    '6': 'Mudança de Diagnóstico（更改诊断）',
    '7': 'TB-DR（耐药结核）',
    '8': 'Mudança de Esquema（更换方案）',
    '9': 'Falência（治疗失败）',
    '10': 'Abandono Primário（原发放弃）',
}
SOURCE_PDF = ('DICI_DADOS_NET_Tuberculose_23_07_2020.pdf, '
              'Campo 62 tp_situacao_encerramento; local copy '
              'data/raw/dici_tb.pdf')


def main():
    d = pd.read_parquet(
        os.path.join(PROC, 'sinan', 'tubebra_2001_2019.parquet'),
        columns=['SITUA_ENCE', 'NU_ANO', 'AGE_YEARS', 'AGRAVAIDS',
                 'AGRAVALCOO', 'BACILOSC_E', 'DT_NOTIFIC',
                 'DT_ENCERRA'])
    d['s'] = d['SITUA_ENCE'].astype(str).str.strip()
    d['age'] = pd.to_numeric(d['AGE_YEARS'], errors='coerce')
    d['yr'] = pd.to_numeric(d['NU_ANO'], errors='coerce')
    d['aids'] = d['AGRAVAIDS'].astype(str).str.strip() == '1'
    d['alco'] = d['AGRAVALCOO'].astype(str).str.strip() == '1'
    d['smear'] = d['BACILOSC_E'].astype(str).str.strip().isin(['2', '3', '4'])
    notif = pd.to_datetime(d['DT_NOTIFIC'].astype(str).str.strip(),
                           format='%Y%m%d', errors='coerce')
    enc = pd.to_datetime(d['DT_ENCERRA'].astype(str).str.strip(),
                         format='%Y%m%d', errors='coerce')
    d['dur'] = ((enc - notif).dt.days).clip(0, 1000)

    res = {'date': '2026-08-26',
           'experiment': 'sinan_label_dictionary_adjudication_v1',
           'question': 'SITUA_ENCE 编码语义裁定（round-5 的 4=TB死亡假设复核）',
           'official_dictionary': OFFICIAL,
           'official_source': SOURCE_PDF,
           'literature_corroboration': [
               {'sample': '巴西东北部 2012-2021, 184,657 例',
                'source': 'Lima CCD et al., Medicina (Ribeirão Preto) '
                          '2024;57(2), doi:10.11606/issn.2176-7262.'
                          'rmrp.2024.212984',
                'distribution_pct': {'cura': 63.9, 'abandono': 9.1,
                                     'obito_tb': 3.2, 'obito_outras': 3.8,
                                     'transferencia': 9.1}},
               {'sample': '巴西监狱系统 2014',
                'source': 'Rev Saude Publica 2015;49:66',
                'note': 'TB 死亡与非 TB 死亡分开统计，编码同官方字典'},
           ]}

    # 数据内证 (a)(c)：全期人口学交叉表
    g = d[d['s'].isin(['1', '2', '3', '4', '5'])].groupby('s').agg(
        n=('s', 'size'), age_median=('age', 'median'),
        pct_65p=('age', lambda a: float((a >= 65).mean())),
        aids_pct=('aids', 'mean'), alco_pct=('alco', 'mean'),
        smear_pos_pct=('smear', 'mean'),
        closure_days_median=('dur', 'median'))
    res['evidence_demographics_full_period'] = {
        s: {k: (round(float(v), 4) if isinstance(v, float) else int(v))
            for k, v in row.items()}
        for s, row in g.round(4).to_dict('index').items()}

    # 数据内证 (b)：编码使用率的时间切片（% of {1..5}）
    d5 = d[d['s'].isin(['1', '2', '3', '4', '5'])]
    per = pd.cut(d5['yr'], [2000, 2005, 2012, 2019],
                 labels=['2001-05', '2006-12', '2013-19'])
    ts = (pd.crosstab(per, d5['s'], normalize='index') * 100).round(2)
    res['evidence_time_slice_pct'] = {
        str(idx): {str(c): float(v) for c, v in row.items()}
        for idx, row in ts.to_dict('index').items()}

    # 数据内证 (a)：三时期年龄平坦性（值 5 vs 值 1）
    flat = {}
    for p in ['2001-05', '2006-12', '2013-19']:
        m = per == p
        a5 = d5.loc[m & (d5['s'] == '5'), 'age']
        a1 = d5.loc[m & (d5['s'] == '1'), 'age']
        flat[p] = {'value5_pct65': round(float((a5 >= 65).mean()), 4),
                   'value1_pct65': round(float((a1 >= 65).mean()), 4)}
    res['evidence_value5_age_flatness'] = flat

    res['verdict'] = {
        'type': 'LABEL_ERROR_CONFIRMED',
        'round5_assumption': 'SITUA_ENCE 4 = TB death（错误）',
        'correct_semantics': {
            '4': '非 TB 死亡（2001-05 段因值 3 未启用，实际含全因死亡）',
            '3': 'TB 死亡（2006+ 语义一致）',
            '5': '转出', '2': '放弃治疗（失访）', '1': '治愈'},
        'impact': [
            'sinan_scale_training/sinan_elderly/sinan_temporal_prior '
            '(_20260826.json) 的终点标签语义须更正为「非 TB 死亡'
            '（2001-05 段含全因死亡）」；AUROC 数值与机制层结论'
            '（arm 间差值、特征重要性排序）不受影响——所有 arm 共享'
            '同一 y 定义',
            '「三国双终点（感染+死亡）证据链」的死亡终点表述更正为'
            '「非 TB 死亡为主」；修正终点 = 全因死亡 {3,4}（本轮 '
            'temporal_prior v2 重验）',
            '老年塌方 0.670 的终点实为非 TB 死亡预测；换终点实验'
            '（第七轮 P1）按正确字典构造多类别'],
        'corrected_endpoint_definitions': {
            'all_cause_death': ['3', '4'],
            'tb_death': ['3'],
            'non_tb_death': ['4'],
            'lost': ['2'], 'transfer': ['5'], 'cured': ['1'],
            'note': '多类别分析集 = {1,2,3,4,5}；0/空/6-10 为小类或'
                    '在治状态，不进入终点分析'}}

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('saved', OUT)
    print('verdict:', res['verdict']['type'])
    print(ts.to_string())
    return res


if __name__ == '__main__':
    main()
