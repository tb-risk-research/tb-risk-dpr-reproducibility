#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SINAN 全国结核通报数据解析管线（2001-2019，DBC → parquet）。

数据源：DataSUS SINAN TUBEBR01-19.dbc（ftp.datasus.gov.br），
巴西全国法定结核通报个案（约 150 万个体级记录，97 字段）。

读取：tb_risk.data.dbc_reader.read_dbc（纯 Python blast 解压，
2026-08-26 以 read.dbc 1.2.0 官方样例 sids/storm 黄金标准验证通过，
列子集偏移 bug 已修复并回归）。

年龄编码（NU_IDADE_N，SINAN 字典）：首位 = 单位（1=小时 2=天
3=月 4=年），后三位 = 数值；经验证 99.6% 落在 4xxx（岁）。

输出：data/processed/sinan/tubebra_2001_2019.parquet +
sinan_parse_meta_20260826.json（按年计数/字段核对/数据质量）。
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # Desktop

from tb_risk.data.dbc_reader import read_dbc  # noqa: E402

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw',
                       'sinan')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'processed', 'sinan')

# 建模相关列（诊断时点可得 + 终末结局；97 字段全集见
# data/raw/sinan/download_log.json 与各文件 DBF 头）
COLUMNS = [
    # 时空/分组
    'NU_ANO', 'SG_UF_NOT', 'ID_MUNICIP',
    # 个体
    'NU_IDADE_N', 'CS_SEXO', 'CS_GESTANT', 'CS_RACA', 'CS_ESCOL_N',
    # 临床（诊断时点）
    'FORMA', 'EXTRAPU1_N', 'RAIOX_TORA', 'TESTE_TUBE', 'BACILOSC_E',
    'CULTURA_ES', 'HISTOPATOL', 'TEST_MOLEC',
    # 合并症
    'AGRAVAIDS', 'AGRAVALCOO', 'AGRAVDIABE', 'AGRAVDOENC', 'AGRAVDROGA',
    'AGRAVTABAC', 'AGRAVOUTRA',
    # 人群/社会
    'POP_LIBER', 'POP_RUA', 'POP_SAUDE', 'POP_IMIG', 'BENEF_GOV',
    'INSTITUCIO',
    # 接触者/治疗
    'NU_CONTATO', 'TRATAMENTO', 'TRAT_SUPER', 'DOENCA_TRA',
    # 结局
    'SITUA_ENCE', 'DT_DIAG', 'DT_ENCERRA', 'DT_INIC_TR', 'DT_NOTIFIC',
]


def decode_age(nu_idade):
    """SINAN 年龄编码 → 岁（float）：首位单位 1时/2天/3月/4年。"""
    nu = np.asarray(nu_idade, dtype=np.float64)
    out = np.full(nu.shape, np.nan)
    m = (nu >= 4000) & (nu <= 4130)
    out[m] = nu[m] - 4000
    m = (nu >= 3000) & (nu < 4000)
    out[m] = (nu[m] - 3000) / 12.0
    m = (nu >= 2000) & (nu < 3000)
    out[m] = (nu[m] - 2000) / 365.25
    m = (nu >= 1000) & (nu < 2000)
    out[m] = (nu[m] - 1000) / 8766.0
    m = (nu > 0) & (nu < 1000)   # 个别旧记录直接存岁
    out[m] = nu[m]
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith('.dbc'))
    frames = []
    meta = {'date': '2026-08-26', 'files': []}
    for fn in files:
        t0 = time.time()
        try:
            df, hdr = read_dbc(os.path.join(RAW_DIR, fn),
                               columns=COLUMNS)
        except Exception as e:
            # 下载进行中的不完整文件（blast 输入耗尽）等：跳过并如实记录
            meta['files'].append({'file': fn, 'error': str(e)[:120]})
            print('SKIP', fn, str(e)[:80])
            continue
        yr = int(fn.replace('TUBEBR', '').replace('.dbc', ''))
        df['FILE_YEAR'] = yr
        frames.append(df)
        meta['files'].append({
            'file': fn, 'rows': int(len(df)),
            'n_records_declared': int(hdr['n_records']),
            'fields': len(hdr['fields']),
            'seconds': round(time.time() - t0, 1),
        })
        print('parsed', fn, len(df), 'rows', round(time.time() - t0, 1), 's')
    if not frames:
        raise RuntimeError('无可解析文件')

    data = pd.concat(frames, ignore_index=True)
    data['AGE_YEARS'] = decode_age(data['NU_IDADE_N'].values)
    # 日期字段（D8 yyyymmdd）→ 年份，供时序分割
    for c in ('DT_DIAG', 'DT_ENCERRA', 'DT_INIC_TR', 'DT_NOTIFIC'):
        data[c + '_Y'] = pd.to_numeric(
            data[c].astype(str).str.slice(0, 4), errors='coerce')

    out_parquet = os.path.join(OUT_DIR, 'tubebra_2001_2019.parquet')
    data.to_parquet(out_parquet, index=False)

    meta.update({
        'total_rows': int(len(data)),
        'n_columns': int(data.shape[1]),
        'by_year_notified': data['DT_NOTIFIC_Y'].value_counts()
            .sort_index().astype(int).to_dict(),
        'by_situa_ence': data['SITUA_ENCE'].value_counts(dropna=False)
            .astype(int).to_dict(),
        'age_median': float(data['AGE_YEARS'].median()),
        'male_pct': float((data['CS_SEXO'] == 'M').mean() * 100),
        'age_bounds': [float(data['AGE_YEARS'].min()),
                       float(data['AGE_YEARS'].max())],
        'parquet': out_parquet,
    })
    meta_path = os.path.join(OUT_DIR, 'sinan_parse_meta_20260826.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != 'files'},
                     ensure_ascii=True))


if __name__ == '__main__':
    main()
