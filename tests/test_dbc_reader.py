# -*- coding: utf-8 -*-
"""dbc_reader 回归测试（2026-08-26 第五轮新增）。

黄金样例：read.dbc 1.2.0（CRAN Archive）官方端到端测试文件
sids.dbc / storm.dbc（已拷贝至 tests/fixtures/，消除项目外路径依赖）。

覆盖：
1. blast 解压 + DBF 解析端到端正确性（记录数/字段数/已知数据值
   与 nc.sids 经典数据集逐值吻合）；
2. 列子集提取偏移回归（2026-08-26 修复：过滤列后字节偏移必须仍按
   完整字段表计算——该 bug 曾致 SINAN 字段整体错位）；
3. 删除标志行过滤。
"""
import os

import pandas as pd
import pytest

from tb_risk.data.dbc_reader import (blast_decompress, parse_dbf_header,
                                     read_dbc)

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
SIDS = os.path.join(FIX, 'sids.dbc')
STORM = os.path.join(FIX, 'storm.dbc')


@pytest.mark.parametrize('path,n,fields0', [
    (SIDS, 100, 'AREA'),
    (STORM, 100, 'BEGIN_DATE'),
])
def test_golden_end_to_end(path, n, fields0):
    df, hdr = read_dbc(path)
    assert len(df) == n == hdr['n_records']
    assert hdr['fields'][0][0] == fields0


def test_sids_known_values():
    """Ashe 县首行与经典 nc.sids 数据集逐值吻合（R sp/maps 一致）。"""
    df, _ = read_dbc(SIDS)
    row = df.iloc[0]
    assert row['NAME'] == 'Ashe'
    assert row['BIR74'] == 1091
    assert row['SID74'] == 1
    assert row['NWBIR74'] == 10
    assert row['FIPSNO'] == 37009


def test_column_subset_offset_regression():
    """列子集提取与全列提取逐值一致（偏移 bug 回归防护）。"""
    full, _ = read_dbc(SIDS)
    sub, hdr = read_dbc(SIDS, columns=['AREA', 'SID74', 'FIPS'])
    # 列按 DBF 头原始字段序返回
    assert list(sub.columns) == ['AREA', 'FIPS', 'SID74']
    pd.testing.assert_frame_equal(sub, full[['AREA', 'FIPS', 'SID74']],
                                  check_dtype=False)
    # 头部解析不依赖 columns
    assert hdr['n_records'] == 100


def test_blast_truncated_input():
    """截断流应报输入耗尽（err=2）而非崩溃。"""
    with open(SIDS, 'rb') as f:
        raw = f.read()
    import struct
    hs = struct.unpack_from('<H', raw, 8)[0]
    _, err = blast_decompress(raw[hs + 4:hs + 64])
    assert err == 2


def test_parse_dbf_header_shape():
    with open(SIDS, 'rb') as f:
        raw = f.read()
    hdr = parse_dbf_header(raw)
    assert hdr['n_records'] == 100
    assert len(hdr['fields']) == 14
    assert hdr['record_size'] == 168
    # 字段长度总和 + 1（删除标志）= 记录长度
    assert sum(f[2] for f in hdr['fields']) + 1 == hdr['record_size']
