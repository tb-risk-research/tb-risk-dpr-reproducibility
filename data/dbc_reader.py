#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯 Python DBC 读取器（巴西 DataSUS .dbc → pandas DataFrame）。

实现来源（2026-08-26 移植）：
- blast.c（Mark Adler，zlib contrib/blast，PKWare Compression Library
  "explode" 解压的公开参考实现）——逐位端口到 Python，含 4KB 首窗
  距离检查与 16KB 滑动窗口语义；
- dbc2dbf.c（Daniela Petruzalek，R 包 read.dbc 1.2.0，CRAN 归档）——
  DBC 容器结构：bytes 8-9 (uint16 LE) = DBF 头长度；[0, header) 为
  原样 DBF 头 + 字段描述；[header, header+4) 为 CRC32（跳过）；
  [header+4, EOF) 为单一连续 blast 压缩流。

为何自研：pysus/pyreaddbc 依赖 Rust/C 工具链，本机（Windows Store
Python 3.9）wheel 构建失败；DBF 部分自实现（字段描述符 32B × n +
0x0D 结束，记录 = 删除标志 1B + 定长字段）。

性能：纯 Python 逐位 Huffman + LZ77 慢（~1-4 MB 压缩数据/分钟量级），
单文件分钟级；SINAN 全国结核文件 3-5 MB/个，可接受。

用法：
    from dbc_reader import read_dbc
    df = read_dbc('TUBEBR15.dbc')          # 全部列
    df = read_dbc('TUBEBR15.dbc', columns=['NU_IDADE_N', 'CS_SEXO'])
"""
import struct

import numpy as np
import pandas as pd

__all__ = ['read_dbc', 'blast_decompress', 'parse_dbf']

MAXBITS = 13
MAXWIN = 16384

# ---- blast.c 静态码表（勿改，与 PKWare 压缩库规范一致）----
_LITLEN = bytes([
    11, 124, 8, 7, 28, 7, 188, 13, 76, 4, 10, 8, 12, 10, 12, 10, 8, 23, 8,
    9, 7, 6, 7, 8, 7, 6, 55, 8, 23, 24, 12, 11, 7, 9, 11, 12, 6, 7, 22, 5,
    7, 24, 6, 11, 9, 6, 7, 22, 7, 11, 38, 7, 9, 8, 25, 11, 8, 11, 9, 12,
    8, 12, 5, 38, 5, 38, 5, 11, 7, 5, 6, 21, 6, 10, 53, 8, 7, 24, 10, 27,
    44, 253, 253, 253, 252, 252, 252, 13, 12, 45, 12, 45, 12, 61, 12, 45,
    44, 173])
_LENLEN = bytes([2, 35, 36, 53, 38, 23])
_DISTLEN = bytes([2, 20, 53, 230, 247, 151, 248])
_BASE = (3, 2, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 40, 72, 136, 264)
_EXTRA = (0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8)


def _construct(rep):
    """construct()：紧凑码长表示 → {(len, code): symbol} 解码表。

    返回 (table, complete_flag)。
    """
    lengths = []
    for b in rep:
        count = (b >> 4) + 1
        lengths.extend([b & 15] * count)
    n = len(lengths)
    cnt = [0] * (MAXBITS + 1)
    for l in lengths:
        cnt[l] += 1
    if cnt[0] == n:
        return {}, True
    # 排序：按长度、同长度按符号序
    syms_by_len = [[] for _ in range(MAXBITS + 1)]
    for sym, l in enumerate(lengths):
        if l:
            syms_by_len[l].append(sym)
    table = {}
    first = 0
    left = 1
    for l in range(1, MAXBITS + 1):
        for k, sym in enumerate(syms_by_len[l]):
            table[(l, first + k)] = sym
        first = (first + cnt[l]) << 1
        left = (left << 1) - cnt[l]
        if left < 0:
            return table, False
    return table, True


def blast_decompress(data):
    """blast 解压：data（不含 DBF 头，纯压缩流）→ 原始字节。

    返回 (bytes, err)：err=0 成功；2 输入耗尽；-1/-2/-3 格式错误。
    """
    # 预构建解码表
    lit_table, _ = _construct(_LITLEN)
    len_table, _ = _construct(_LENLEN)
    dist_table, _ = _construct(_DISTLEN)

    n_in = len(data)
    pos = 0
    bitbuf = 0
    bitcnt = 0

    out = bytearray()
    window = bytearray(MAXWIN)
    nxt = 0
    first = True

    # ---- 读取 blast 头 2 字节 ----
    lit_coded = data[0]
    dict_bits = data[1]
    pos = 2
    if lit_coded > 1:
        return b'', -1
    if not (4 <= dict_bits <= 6):
        return b'', -2

    err = 0
    while True:
        # 读 1 bit：0=字面量，1=长度/距离
        if bitcnt == 0:
            if pos >= n_in:
                err = 2
                break
            bitbuf = data[pos]
            pos += 1
            bitcnt = 8
        flag = bitbuf & 1
        bitbuf >>= 1
        bitcnt -= 1

        if not flag:
            # ---- 字面量 ----
            if lit_coded:
                # Huffman 解码（字面量码）
                code = 0
                length = 0
                sym = None
                while length < MAXBITS:
                    length += 1
                    if bitcnt == 0:
                        if pos >= n_in:
                            err = 2
                            break
                        bitbuf = data[pos]
                        pos += 1
                        bitcnt = 8
                    code = (code << 1) | ((bitbuf & 1) ^ 1)
                    bitbuf >>= 1
                    bitcnt -= 1
                    sym = lit_table.get((length, code))
                    if sym is not None:
                        break
                if err:
                    break
                if sym is None:
                    err = -9
                    break
            else:
                # 未编码字面量：8 bits
                while bitcnt < 8:
                    if pos >= n_in:
                        err = 2
                        break
                    bitbuf |= data[pos] << bitcnt
                    pos += 1
                    bitcnt += 8
                if err:
                    break
                sym = bitbuf & 0xFF
                bitbuf >>= 8
                bitcnt -= 8
            window[nxt] = sym
            nxt += 1
            if nxt == MAXWIN:
                out += window
                nxt = 0
                first = False
        else:
            # ---- 长度码 Huffman 解码 ----
            code = 0
            length = 0
            symbol = None
            while length < MAXBITS:
                length += 1
                if bitcnt == 0:
                    if pos >= n_in:
                        err = 2
                        break
                    bitbuf = data[pos]
                    pos += 1
                    bitcnt = 8
                code = (code << 1) | ((bitbuf & 1) ^ 1)
                bitbuf >>= 1
                bitcnt -= 1
                symbol = len_table.get((length, code))
                if symbol is not None:
                    break
            if err:
                break
            if symbol is None:
                err = -9
                break

            # 长度 extra bits
            need = _EXTRA[symbol]
            while bitcnt < need:
                if pos >= n_in:
                    err = 2
                    break
                bitbuf |= data[pos] << bitcnt
                pos += 1
                bitcnt += 8
            if err:
                break
            extra = bitbuf & ((1 << need) - 1)
            bitbuf >>= need
            bitcnt -= need
            ln = _BASE[symbol] + extra
            if ln == 519:
                break  # 结束码

            # ---- 距离码 ----
            dist_sym_bits = dict_bits if ln != 2 else 2
            code = 0
            length = 0
            dsym = None
            while length < MAXBITS:
                length += 1
                if bitcnt == 0:
                    if pos >= n_in:
                        err = 2
                        break
                    bitbuf = data[pos]
                    pos += 1
                    bitcnt = 8
                code = (code << 1) | ((bitbuf & 1) ^ 1)
                bitbuf >>= 1
                bitcnt -= 1
                dsym = dist_table.get((length, code))
                if dsym is not None:
                    break
            if err:
                break
            if dsym is None:
                err = -9
                break
            dist = dsym << dist_sym_bits

            need = dist_sym_bits
            while bitcnt < need:
                if pos >= n_in:
                    err = 2
                    break
                bitbuf |= data[pos] << bitcnt
                pos += 1
                bitcnt += 8
            if err:
                break
            dist += bitbuf & ((1 << need) - 1)
            bitbuf >>= need
            bitcnt -= need
            dist += 1
            if dist > MAXWIN or (first and dist > nxt):
                err = -3
                break

            # ---- 复制 ln 字节（自距离 dist 回，允许重叠）----
            while ln > 0:
                copy = min(ln, MAXWIN - nxt)
                start = (nxt - dist) % MAXWIN
                for i in range(copy):
                    window[nxt + i] = window[(start + i) % MAXWIN]
                nxt += copy
                ln -= copy
                if nxt == MAXWIN:
                    out += window
                    nxt = 0
                    first = False

    if err == 0 and nxt:
        out += window[:nxt]
    return bytes(out), err


def parse_dbf_header(raw):
    """DBF 头解析：返回 dict（n_records/header_size/record_size/fields）。

    fields: [(name, ftype, flen, dec), ...]
    """
    n_records = struct.unpack_from('<I', raw, 4)[0]
    header_size = struct.unpack_from('<H', raw, 8)[0]
    record_size = struct.unpack_from('<H', raw, 10)[0]
    fields = []
    p = 32
    while p < header_size and raw[p] != 0x0D:
        name = raw[p:p + 11].split(b'\x00')[0].decode('latin-1').strip()
        ftype = chr(raw[p + 11])
        flen = raw[p + 16]
        dec = raw[p + 17]
        fields.append((name, ftype, flen, dec))
        p += 32
    return {
        'n_records': n_records, 'header_size': header_size,
        'record_size': record_size, 'fields': fields,
        'version': raw[0],
    }


def parse_dbf(raw, columns=None, encoding='latin-1'):
    """DBF 记录解析 → DataFrame（定长记录切片，numpy 向量化解码）。"""
    hdr = parse_dbf_header(raw)
    fields = hdr['fields']
    n_records = hdr['n_records']
    record_size = hdr['record_size']
    body = raw[hdr['header_size']:]

    # 删除标志列在每记录首字节：* = 删除，空格 = 正常
    if len(body) < n_records * record_size:
        # DBC 解压得到的记录数可能与头声明略有出入，按实际长度截断
        n_records = len(body) // record_size
    flags = np.frombuffer(body[:n_records * record_size], dtype=np.uint8
                          ).reshape(n_records, record_size)[:, 0]
    keep = flags != ord('*')

    if columns is not None:
        wanted = set(columns)
        fields = [f for f in fields if f[0] in wanted]
        if not fields:
            raise ValueError('columns 与 DBF 字段无交集: %s' % columns)

    # 记录体（去掉标志位）整体展开成 uint8 矩阵
    rec = np.frombuffer(body[:n_records * record_size], dtype=np.uint8
                        ).reshape(n_records, record_size)
    # 字节偏移必须按【完整】字段表计算（过滤列后偏移不变）
    offsets = {}
    p = 1
    for name, ftype, flen, dec in hdr['fields']:
        offsets[name] = (p, flen, ftype, dec)
        p += flen
    data = {}
    for name, ftype, flen, dec in fields:
        p, flen = offsets[name][0], offsets[name][1]
        if flen == 0:
            continue
        # 定长字节串数组（零拷贝），C 级解析
        arr = np.frombuffer(rec[:, p:p + flen].tobytes(), dtype='S%d' % flen)
        s = pd.Series(arr, copy=False)
        if ftype == 'N':
            vals = pd.to_numeric(
                s.str.decode(encoding, errors='replace').str.strip(),
                errors='coerce').fillna(0)
            data[name] = vals.astype('int64') if dec == 0 else vals.astype('float64')
        else:
            data[name] = s.str.decode(encoding, errors='replace').str.strip()
    df = pd.DataFrame(data)
    df = df[keep].reset_index(drop=True)
    return df, hdr


def read_dbc(path, columns=None, encoding='latin-1'):
    """读取 .dbc → (DataFrame, header dict)。

    columns=None 全列；指定列子集（其余列不解析，提速）。
    """
    with open(path, 'rb') as f:
        raw = f.read()
    if raw[:1] not in (b'\x03',) and not (0x30 <= raw[0] <= 0x30):
        # DBC 版本字节与 DBF 同族（0x03 常见）；宽限 0x30 视觉表
        pass
    header_size = struct.unpack_from('<H', raw, 8)[0]
    dbf_header = raw[:header_size]
    comp = raw[header_size + 4:]
    decomp, err = blast_decompress(comp)
    if err != 0:
        raise IOError('blast 解压失败 err=%s (输入 %d 字节)' % (err, len(comp)))
    full = dbf_header + decomp
    return parse_dbf(full, columns=columns, encoding=encoding)
