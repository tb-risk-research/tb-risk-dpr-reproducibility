#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tb_risk 公共 IO 工具模块

提供跨模块共享的编码检测、字段映射和数据验证功能。
提取自 persistence/mixins.py 和 validation/validator.py 的重复逻辑，
统一所有 CSV/Excel/JSON 导入的编码检测和字段匹配。

用法：
    from tb_risk.io_utils import detect_file_encoding, dynamic_column_match, ...
"""

import json
import logging
import os

LOGGER = logging.getLogger("tb_risk.io")

# data_io 统一常量（问题九-2：纯委托 data_io 子包，删除回退实现）
from .data_io.conf import FIELD_RANGES, MISSING_VALUE_MARKER

# data_io 同义词库（问题九-2：纯委托，删除本地 _FIELD_SYNONYMS 影子层）
from .data_io.synonyms import (
    fuzzy_match_column, normalize_name, dynamic_column_match as _new_dynamic_column_match,
    register_synonym_feedback, get_field_synonyms as _new_get_field_synonyms,
    FIELD_SYNONYM_LIBRARY, generate_legacy_synonyms,
)

# 条件导入
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


# ==============================================================================
# 一、编码检测（统一入口）
# ==============================================================================


def detect_file_encoding(filepath, sample_size=100000, use_chardet=True):
    """自动检测文件编码

    编码检测链：
    1. chardet 自动检测（高置信度时优先）
    2. 回退到常见编码列表依次尝试（utf-8-sig, utf-8, gbk, gb18030, gb2312, big5, latin-1）

    参数：
        filepath (str): 文件路径
        sample_size (int): 检测时读取的字节数（默认 100KB）
        use_chardet (bool): 是否优先使用 chardet 检测

    返回：
        str: 编码名称（如 'utf-8-sig'、'gbk' 等）
    """
    if not os.path.exists(filepath):
        LOGGER.warning("文件不存在，无法检测编码: %s", filepath)
        return 'utf-8-sig'

    # 步骤1：chardet 自动检测
    if use_chardet:
        try:
            import chardet
            with open(filepath, 'rb') as f:
                raw = f.read(sample_size)
            result = chardet.detect(raw)
            if result['confidence'] > 0.7:
                encoding = result['encoding']
                LOGGER.debug("chardet 检测到编码: %s (置信度=%.2f)", encoding, result['confidence'])
                if encoding and encoding.lower() in ('gb2312', 'gbk', 'gb18030'):
                    return 'gb18030'  # 统一为 gb18030（GB2312/GBK 的超集）
                return encoding or 'utf-8-sig'
        except ImportError:
            LOGGER.debug("chardet 未安装，使用回退编码检测")
        except Exception as e:
            LOGGER.debug("chardet 检测失败: %s，使用回退编码检测", e)

    # 步骤2：回退到常见编码列表
    common_encodings = ['utf-8-sig', 'utf-8', 'gb18030', 'gbk', 'gb2312', 'big5', 'latin-1']
    for encoding in common_encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                f.read(2000)  # 尝试读取前2000字符
            if encoding == 'latin-1':
                LOGGER.warning("文件编码检测退回到 latin-1，可能存在乱码，请检查文件编码")
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue

    # 都不行，默认返回 utf-8-sig
    LOGGER.warning("无法自动检测编码，回退到 utf-8-sig")
    return 'utf-8-sig'


# ==============================================================================
# 二、字段名映射（中英文同义词自动匹配）
# ==============================================================================


# 字段同义词字典：标准字段名 → 中英文别名列表
# 问题九-2：纯委托 data_io.synonyms.generate_legacy_synonyms 生成，
# 删除本地字面量副本（原 ~50 行 _FIELD_SYNONYMS 字典已移除）。
_FIELD_SYNONYMS = generate_legacy_synonyms()


def _normalize_name(name):
    """标准化名称：去除空格、下划线、连字符，转小写。

    问题九-2：纯委托 data_io.synonyms.normalize_name，删除本地回退实现。
    """
    return normalize_name(name)


def dynamic_column_match(col_name):
    """动态列名匹配：中英文同义词自动映射

    问题九-2：纯委托 data_io.synonyms.dynamic_column_match，
    删除本地三策略回退实现（原 ~50 行 keyword_priority 匹配已移除）。

    参数：
        col_name (str): 原始列名

    返回：
        str|None: 匹配到的标准字段名，或 None
    """
    return _new_dynamic_column_match(col_name)


def map_columns(columns, verbose=True):
    """批量映射列名到标准字段名

    参数：
        columns (list[str]): 原始列名列表
        verbose (bool): 是否输出映射日志

    返回：
        dict: {原始列名: 标准字段名}
    """
    mapping = {}
    for col in columns:
        matched = dynamic_column_match(col)
        if matched:
            mapping[col] = matched
            if verbose:
                LOGGER.debug("列名映射: '%s' → '%s'", col, matched)
        else:
            if verbose:
                LOGGER.debug("列名未匹配: '%s'，保留原始名称", col)

    if verbose and mapping:
        LOGGER.info("字段映射完成: %d/%d 列已映射", len(mapping), len(columns))
        for orig, std in mapping.items():
            LOGGER.info("  %s → %s", orig, std)

    return mapping


def get_field_synonyms():
    """获取字段同义词字典（只读副本）

    问题九-2：纯委托 data_io.synonyms.get_field_synonyms，删除本地回退。

    返回：
        dict: {标准字段名: [别名列表]}
    """
    return _new_get_field_synonyms()


# ==============================================================================
# 三、数据完整性预验证
# ==============================================================================


def validate_data_integrity(records, required_fields=None, verbose=True):
    """数据完整性预验证

    在数据加载后、模型训练前执行前置质量检查，返回结构化报告。

    检查项：
    - 必填字段是否存在
    - 值域是否合理（年龄 0-120、暴露时长非负等）
    - 标签分布是否均衡
    - 缺失值比例是否超阈值

    参数：
        records (list[dict]): 数据记录列表
        required_fields (list[str]|None): 必填字段列表，None 使用默认
        verbose (bool): 是否输出详细日志

    返回：
        dict: {
            'valid': bool,
            'total_records': int,
            'warnings': list[str],
            'errors': list[str],
            'field_completeness': dict,
            'value_range_violations': dict,
            'label_distribution': dict,
            'missing_rate': float,
            'suggestions': list[str],
        }
    """
    if required_fields is None:
        required_fields = ['age', 'cumulative_exposure', 'has_symptoms', 'bcg_vaccine',
                          'has_tb', 'contact_distance', 'ventilation', 'is_high_risk',
                          'past_illness', 'exposure_setting']

    if not records:
        return {
            'valid': False,
            'total_records': 0,
            'warnings': [],
            'errors': ['数据集为空'],
            'field_completeness': {},
            'value_range_violations': {},
            'label_distribution': {},
            'missing_rate': 1.0,
            'suggestions': ['请检查数据文件是否包含有效记录'],
        }

    warnings = []
    errors = []
    suggestions = []
    total = len(records)

    # 1. 字段完整性检查
    field_completeness = {}
    all_keys = set()
    for rec in records:
        all_keys.update(rec.keys())

    for field in required_fields:
        present = sum(1 for r in records if field in r and r[field] is not None
                     and str(r[field]).strip() != '')
        completeness = present / total
        field_completeness[field] = {
            'present': present,
            'total': total,
            'completeness': round(completeness, 4),
        }
        if completeness < 0.80:
            warnings.append(f"字段 '{field}' 完整率仅 {completeness:.1%}，可能影响模型质量")
        if completeness == 0:
            errors.append(f"必填字段 '{field}' 完全缺失")

    # 2. 值域检查
    value_range_violations = {}
    # 从 data_io.conf.FIELD_RANGES 动态生成值域检查（Phase 0 集成）
    value_checks = {}
    if FIELD_RANGES:
        for field_name, range_info in FIELD_RANGES.items():
            lo = range_info.get('min', 0)
            hi = range_info.get('max', float('inf'))
            field_type = range_info.get('type', 'float')
            if field_type == 'int':
                desc = f"{field_name} 应在 {lo}-{hi} 范围"
            else:
                desc = f"{field_name} 应在 {lo}-{hi} 范围"
            value_checks[field_name] = (lambda v, lo=lo, hi=hi: lo <= v <= hi, desc)
    else:
        # 回退硬编码（FIELD_RANGES 不可用时）
        value_checks = {
            'age': (lambda v: 0 <= v <= 120, '年龄应在 0-120 范围'),
            'cumulative_exposure': (lambda v: v >= 0, '累积暴露应为非负数'),
            'single_duration': (lambda v: 0 <= v <= 1440, '单次接触时长应在 0-1440 分钟范围'),
            'freq_density': (lambda v: 0 <= v <= 50, '每周接触频次应在 0-50 范围'),
            'time_span': (lambda v: 1 <= v <= 104, '持续周期应在 1-104 周范围'),
            'ventilation': (lambda v: 1 <= v <= 5, '通风条件应在 1-5 范围'),
        }

    for field, (check_fn, desc) in value_checks.items():
        violations = 0
        for rec in records:
            if field in rec:
                try:
                    val = float(rec[field])
                    if not check_fn(val):
                        violations += 1
                except (ValueError, TypeError):
                    violations += 1
        if violations > 0:
            value_range_violations[field] = {
                'count': violations,
                'rate': round(violations / total, 4),
                'description': desc,
            }
            if violations / total > 0.10:
                warnings.append(f"字段 '{field}' 有 {violations} 条 ({violations/total:.1%}) 值域违规: {desc}")

    # 3. 标签分布检查
    label_distribution = {}
    label_field_candidates = ['is_confirmed', 'tb_outcome', 'label', 'target', 'diagnosed']
    for lf in label_field_candidates:
        if lf in all_keys:
            values = []
            for rec in records:
                try:
                    values.append(int(float(rec.get(lf, 0))))
                except (ValueError, TypeError):
                    values.append(0)
            unique = list(set(values))
            label_distribution = {
                'field': lf,
                'unique_values': unique,
                'counts': {v: values.count(v) for v in unique},
                'positive_count': sum(1 for v in values if v == 1),
                'negative_count': sum(1 for v in values if v == 0),
                'positive_rate': round(sum(1 for v in values if v == 1) / total, 4),
            }
            break

    if label_distribution:
        pos_count = label_distribution.get('positive_count', 0)
        neg_count = label_distribution.get('negative_count', 0)

        if pos_count == 0 or neg_count == 0:
            errors.append(
                f"目标变量 '{label_distribution['field']}' 只有一个类别 "
                f"(正样本={pos_count}, 负样本={neg_count})，无法训练分类模型。"
                f"建议：确保数据包含至少两类标签，或使用异常检测方法。"
            )
        elif pos_count < 5:
            suggestions.append(
                f"正样本数仅 {pos_count} 条，建议使用过采样（SMOTE）或数据增强，"
                f"或合并更多数据源。所需最小正样本量建议 ≥ 10。"
            )
        elif pos_count / total < 0.005:
            suggestions.append(
                f"正样本比例仅 {pos_count/total:.2%}，数据极度不均衡。"
                f"建议使用分层抽样、过采样或调整类别权重。"
            )

    # 4. 缺失值统计
    total_cells = total * len(all_keys) if all_keys else 1
    missing_cells = 0
    for rec in records:
        for key in all_keys:
            val = rec.get(key)
            if val is None or (isinstance(val, str) and val.strip() == ''):
                missing_cells += 1
    missing_rate = missing_cells / total_cells if total_cells > 0 else 0

    if missing_rate > 0.20:
        warnings.append(f"整体缺失率 {missing_rate:.1%}，建议检查数据源或使用缺失值填充策略")
    if missing_rate > 0.50:
        errors.append(f"整体缺失率 {missing_rate:.1%}，数据质量过低，不建议直接用于训练")

    # 5. 汇总
    valid = len(errors) == 0
    if verbose:
        status = "通过" if valid else "未通过"
        LOGGER.info("数据完整性预验证: %s (记录数=%d, 缺失率=%.1%%)", status, total, missing_rate)
        if warnings:
            for w in warnings:
                LOGGER.warning("  ⚠ %s", w)
        if errors:
            for e in errors:
                LOGGER.error("  ✗ %s", e)
        if suggestions:
            for s in suggestions:
                LOGGER.info("  💡 %s", s)

    return {
        'valid': valid,
        'total_records': total,
        'warnings': warnings,
        'errors': errors,
        'field_completeness': field_completeness,
        'value_range_violations': value_range_violations,
        'label_distribution': label_distribution,
        'missing_rate': round(missing_rate, 4),
        'suggestions': suggestions,
    }


# ==============================================================================
# 四、统一数据加载接口
# ==============================================================================


def load_data_from_file(filepath, **kwargs):
    """统一的数据加载接口

    支持 CSV、JSON、JSONL、Excel、Parquet、纯文本格式。
    自动检测编码并映射字段名。
    纯文本文件通过 text_parser 抽取结构化字段。

    参数：
        filepath (str): 数据文件路径
        **kwargs: 传递给各格式加载器的参数

    返回：
        tuple[list[dict], dict]: (记录列表, 加载元信息)
    """
    ext = os.path.splitext(filepath)[1].lower()
    supported = ('.csv', '.json', '.jsonl', '.xlsx', '.xls', '.parquet')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")

    meta = {'filepath': filepath, 'format': ext, 'encoding': None}

    # 新增：纯文本文件路由到 text_parser
    if ext in ('.txt', '.text'):
        return _load_text_file(filepath, meta, **kwargs)

    # 新增：未知扩展名自动检测
    if ext not in supported:
        try:
            from .data_io.text_parser import detect_input_type
            input_type = detect_input_type(filepath)
        except ImportError:
            input_type = 'unknown'

        if input_type == 'text':
            return _load_text_file(filepath, meta, **kwargs)
        elif input_type in ('structured_csv', 'structured_json'):
            if input_type == 'structured_csv':
                records, meta = _load_csv_generic(filepath, meta, **kwargs)
            else:
                records, meta = _load_json_generic(filepath, meta, **kwargs)
            return records, meta
        else:
            raise ValueError(f"不支持的文件格式 '{ext}'，且无法自动检测类型。支持: {supported}")

    if ext == '.csv':
        records, meta = _load_csv_generic(filepath, meta, **kwargs)
    elif ext in ('.json', '.jsonl'):
        records, meta = _load_json_generic(filepath, meta, **kwargs)
    elif ext in ('.xlsx', '.xls'):
        records, meta = _load_excel_generic(filepath, meta, **kwargs)
    elif ext == '.parquet':
        records, meta = _load_parquet_generic(filepath, meta, **kwargs)
    else:
        records = []

    return records, meta


def _load_text_file(filepath, meta, **kwargs):
    """加载纯文本文件，通过 text_parser 抽取结构化字段。

    流程：文本读取 → 字段抽取 → 字段名标准化 → 数据完整性验证

    参数：
        filepath (str): 文本文件路径
        meta (dict): 元信息字典
        **kwargs: 传递给 parse_text_input 的参数（如 use_llm, model_path）

    返回：
        tuple[list[dict], dict]: (记录列表, 元信息)
    """
    try:
        from .data_io.text_parser import parse_text_input
    except ImportError:
        raise ImportError(
            "文本抽取需要 data_io.text_parser 模块。"
            "请确保 data_io/text_parser.py 和 data_io/regex_templates.json 存在。"
        )

    encoding = detect_file_encoding(filepath)
    meta['encoding'] = encoding
    meta['input_type'] = 'text'

    with open(filepath, 'r', encoding=encoding) as f:
        text = f.read()

    use_llm = kwargs.pop('use_llm', False)
    model_path = kwargs.pop('model_path', None)
    records, text_meta = parse_text_input(text, use_llm=use_llm, model_path=model_path)
    meta.update(text_meta)

    if not records:
        return records, meta

    # 字段名标准化：文本抽取的字段名可能不完全匹配标准名称
    records = rename_fields(records, auto_map=True, verbose=False)
    meta['renamed'] = True

    # 数据完整性验证
    validation_report = validate_data_integrity(records, verbose=False)
    meta['validation'] = validation_report

    if validation_report.get('errors') or validation_report.get('warnings'):
        LOGGER.warning("文本数据完整性验证发现 %d 个问题: %s",
                       len(validation_report.get('errors', [])) + len(validation_report.get('warnings', [])),
                       ([e for e in validation_report.get('errors', [])[:5]] +
                        [w for w in validation_report.get('warnings', [])[:5]]))

    return records, meta


def _load_csv_generic(filepath, meta, **kwargs):
    """通用 CSV 加载（自动编码检测）"""
    import csv
    encoding = detect_file_encoding(filepath)
    meta['encoding'] = encoding
    records = []
    with open(filepath, 'r', encoding=encoding, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records, meta


def _load_json_generic(filepath, meta, record_key='records', **kwargs):
    """通用 JSON/JSONL 加载"""
    encoding = detect_file_encoding(filepath)
    meta['encoding'] = encoding
    if filepath.endswith('.jsonl'):
        records = []
        with open(filepath, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        with open(filepath, 'r', encoding=encoding) as f:
            data = json.load(f)
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = data.get(record_key, [])
        else:
            records = []
    return records, meta


def _load_excel_generic(filepath, meta, sheet_name=0, **kwargs):
    """通用 Excel 加载"""
    if not PANDAS_AVAILABLE:
        raise ImportError('加载 Excel 需要 pandas。请安装: pip install pandas openpyxl')
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    records = df.to_dict('records')
    # 处理 numpy 类型转换
    for rec in records:
        for k, v in list(rec.items()):
            if hasattr(v, 'item'):
                rec[k] = v.item()
    return records, meta


def _load_parquet_generic(filepath, meta, **kwargs):
    """通用 Parquet 加载"""
    if not PANDAS_AVAILABLE:
        raise ImportError('加载 Parquet 需要 pandas。请安装: pip install pandas pyarrow')
    df = pd.read_parquet(filepath)
    records = df.to_dict('records')
    return records, meta


# ==============================================================================
# 五、字段重命名工具
# ==============================================================================


def rename_fields(records, field_mapping=None, auto_map=True, verbose=True):
    """将记录中的字段名重命名为标准名称

    参数：
        records (list[dict]): 原始记录列表
        field_mapping (dict|None): 手动指定的字段映射 {原始名: 标准名}
        auto_map (bool): 是否自动映射未指定的字段
        verbose (bool): 是否输出映射日志

    返回：
        list[dict]: 字段名标准化后的记录列表
    """
    if not records:
        return records

    # 收集所有列名
    all_columns = set()
    for rec in records:
        all_columns.update(rec.keys())

    # 构建映射
    mapping = dict(field_mapping) if field_mapping else {}

    if auto_map:
        auto_mapping = map_columns(list(all_columns), verbose=verbose)
        for orig, std in auto_mapping.items():
            if orig not in mapping:
                mapping[orig] = std

    if not mapping:
        return records

    # 应用映射
    renamed = []
    for rec in records:
        new_rec = {}
        for key, value in rec.items():
            new_key = mapping.get(key, key)
            new_rec[new_key] = value
        renamed.append(new_rec)

    return renamed


def rename_fields_with_details(records, field_mapping=None, auto_map=True, verbose=True):
    """将记录中的字段名重命名为标准名称，同时返回映射详情。

    与 rename_fields 功能相同，但额外返回每个字段的映射来源信息，
    供管线在质量评分和审计中使用。

    参数：
        records (list[dict]): 原始记录列表
        field_mapping (dict|None): 手动指定的字段映射 {原始名: 标准名}
        auto_map (bool): 是否自动映射未指定的字段
        verbose (bool): 是否输出映射日志

    返回：
        tuple[list[dict], dict]: (字段名标准化后的记录列表, 映射详情)
        映射详情格式: {原始列名: (标准字段名, 来源, 置信度)}
    """
    if not records:
        return records, {}

    # 收集所有列名
    all_columns = set()
    for rec in records:
        all_columns.update(rec.keys())

    # 构建映射（含来源信息）
    mapping = dict(field_mapping) if field_mapping else {}
    mapping_details = {}

    if auto_map:
        for col_name in all_columns:
            if col_name.startswith('_'):
                continue
            if col_name in mapping:
                continue

            # 问题九-2：直接使用模块级导入的 fuzzy_match_column（纯委托 data_io.synonyms）
            # 删除原三层 try/except 回退导入逻辑
            std_name, confidence, source = fuzzy_match_column(col_name)
            if std_name:
                mapping[col_name] = std_name
                mapping_details[col_name] = (std_name, source, confidence)

    # 手动映射的字段也补充详情
    if field_mapping:
        for orig, std in field_mapping.items():
            if orig not in mapping_details:
                mapping_details[orig] = (std, 'manual', 1.0)

    if not mapping:
        return records, mapping_details

    # 应用映射
    renamed = []
    for rec in records:
        new_rec = {}
        for key, value in rec.items():
            new_key = mapping.get(key, key)
            new_rec[new_key] = value
        renamed.append(new_rec)

    return renamed, mapping_details