#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI 输出格式化工具

支持 JSON、CSV、Table 等多种输出格式。
"""

import csv
import io
import json
from typing import Any, Dict, List, Optional


class _NumpyEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，处理 numpy 类型"""

    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
        except ImportError:
            pass
        return super().default(obj)


def _json_dumps(data, **kwargs):
    """使用自定义编码器执行 json.dumps"""
    return json.dumps(data, cls=_NumpyEncoder, **kwargs)


def format_output(data: Any, fmt: str = 'json', pretty: bool = True) -> str:
    """格式化数据为指定格式

    参数：
        data: Any — 要格式化的数据
        fmt: str — 输出格式 ('json', 'csv', 'table')
        pretty: bool — 是否美化输出

    返回：
        str: 格式化后的字符串
    """
    if fmt == 'json':
        return _format_json(data, pretty)
    elif fmt == 'csv':
        return _format_csv(data)
    elif fmt == 'table':
        return _format_table(data)
    else:
        return str(data)


def _format_json(data: Any, pretty: bool = True) -> str:
    """JSON 格式输出"""
    if pretty:
        return _json_dumps(data, ensure_ascii=False, indent=2)
    return _json_dumps(data, ensure_ascii=False)


def _format_csv(data: Any) -> str:
    """CSV 格式输出"""
    output = io.StringIO()
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], dict):
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        else:
            writer = csv.writer(output)
            for row in data:
                writer.writerow([row] if not isinstance(row, (list, tuple)) else row)
    elif isinstance(data, dict):
        writer = csv.writer(output)
        for key, value in data.items():
            writer.writerow([key, value])
    return output.getvalue()


def _format_table(data: Any) -> str:
    """表格格式输出（简单 ASCII 表格）"""
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return _format_dict_list_table(data)
    elif isinstance(data, dict):
        return _format_key_value_table(data)
    return str(data)


def _format_dict_list_table(rows: List[Dict]) -> str:
    """格式化字典列表为表格"""
    if not rows:
        return "(empty)"

    keys = list(rows[0].keys())
    # 计算列宽
    col_widths = {k: len(str(k)) for k in keys}
    for row in rows:
        for k in keys:
            val = str(row.get(k, ''))
            col_widths[k] = max(col_widths[k], len(val[:50]))

    # 构建分隔线
    sep = '+' + '+'.join('-' * (col_widths[k] + 2) for k in keys) + '+'

    lines = [sep]
    # 表头
    header = '|' + '|'.join(f' {k:<{col_widths[k]}} ' for k in keys) + '|'
    lines.append(header)
    lines.append(sep)

    # 数据行
    for row in rows:
        line = '|' + '|'.join(
            f' {str(row.get(k, ""))[:50]:<{col_widths[k]}} ' for k in keys
        ) + '|'
        lines.append(line)

    lines.append(sep)
    return '\n'.join(lines)


def _format_key_value_table(data: Dict) -> str:
    """格式化键值对为表格"""
    max_key_len = max(len(str(k)) for k in data.keys())
    lines = []
    for key, value in data.items():
        if isinstance(value, (list, dict)):
            value = f"[{len(value)} items]"
        lines.append(f"  {key:<{max_key_len}} : {value}")
    return '\n'.join(lines)


def output_result(data: Any, fmt: str = 'json', output_file: Optional[str] = None):
    """输出结果到文件或标准输出

    参数：
        data: Any — 要输出的数据
        fmt: str — 输出格式
        output_file: str | None — 输出文件路径，None 则输出到 stdout
    """
    formatted = format_output(data, fmt)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted)
        print(f"结果已保存到: {output_file}")
    else:
        print(formatted)