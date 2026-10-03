#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自由文本信息抽取模块

从结核病临床叙述文本中提取结构化字段。

支持：
- 正则模板匹配（约 25 个字段模板）
- 段落分割策略
- 可选 LLM 增强抽取
- 多来源融合（结构化 > 正则 > LLM）

参考来源：
- 中国结核病临床数据字典 (2023)
- HL7 FHIR R4 结核病资源模型
"""

import json
import logging
import os
import re

try:
    from .conf import (
        CLINICAL_TEXT_MAP, DISTANCE_TEXT_MAP, SETTING_TEXT_MAP,
        RATING_TEXT_MAP, BOOL_TEXT_MAP,
    )
except ImportError:
    CLINICAL_TEXT_MAP = {}
    DISTANCE_TEXT_MAP = {}
    SETTING_TEXT_MAP = {}
    RATING_TEXT_MAP = {}
    BOOL_TEXT_MAP = {}

LOGGER = logging.getLogger("tb_risk.data_io.text_parser")

# ==============================================================================
# 可选依赖：LLM
# ==============================================================================

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False


# ==============================================================================
# 一、模板加载
# ==============================================================================

_DEFAULT_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), 'regex_templates.json')
_cached_templates = None


def load_templates(template_path=None):
    """加载正则模板。

    参数：
        template_path (str|None): 模板文件路径，None 时使用默认路径

    返回：
        list[dict]: 模板列表，按 priority 降序排列
    """
    global _cached_templates

    if template_path is None:
        template_path = _DEFAULT_TEMPLATE_PATH

    if _cached_templates is not None and template_path == _DEFAULT_TEMPLATE_PATH:
        return _cached_templates

    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        LOGGER.warning("加载正则模板失败: %s，使用内置模板", e)
        data = {"templates": _get_builtin_templates()}

    templates = data.get('templates', [])
    # 按优先级降序排列
    templates.sort(key=lambda t: t.get('priority', 0), reverse=True)

    if template_path == _DEFAULT_TEMPLATE_PATH:
        _cached_templates = templates

    return templates


def _get_builtin_templates():
    """内置模板（当 JSON 文件不可用时的回退）。"""
    return [
        {"pattern": r"年龄[约为]?(\d{1,3})[岁周岁]", "field": "age", "transform": "int", "priority": 10},
        {"pattern": r"痰涂片(阳性|阴性|涂阳|涂阴|\+\+|--)", "field": "sputum_smear", "transform": "clinical", "priority": 10},
        {"pattern": r"(有|无|存在|未见|可见)空洞", "field": "has_cavity", "transform": "bool", "priority": 10},
        {"pattern": r"卡介苗[接种]*(已接种|未接种|有|无|是|否)", "field": "bcg_vaccine", "transform": "bool", "priority": 10},
        {"pattern": r"(已[经曾]?|未[经曾]?|正在|接受|未接受)治疗", "field": "treatment", "transform": "clinical", "priority": 10},
        {"pattern": r"治疗[时长时间]*[约为]?(\d{1,2})[个]?月", "field": "treatment_duration", "transform": "int", "priority": 9},
        {"pattern": r"通风[条件状况情况]*[：:]*\s*(极差|差|一般|好|极好|良好|较差|不好)", "field": "ventilation", "transform": "rating", "priority": 9},
        {"pattern": r"接触[距离]*[：:]*\s*(极近|近|中等|远|极远|近距离|远距离)", "field": "contact_distance", "transform": "distance", "priority": 9},
        {"pattern": r"(场所|环境)[：:]*\s*(拥挤|密闭|一般|户外|油田|营地)", "field": "exposure_setting", "transform": "setting", "priority": 8},
        {"pattern": r"(男性|女性|男|女|male|female)", "field": "gender", "transform": "gender", "priority": 9},
        {"pattern": r"(\d{1,2})\s*[岁岁数]", "field": "age", "transform": "int", "priority": 8},
        {"pattern": r"[曾既]往[有患]?(结核|肺结核|TB)", "field": "has_tb", "transform": "bool_positive", "priority": 9},
        {"pattern": r"咳嗽[频率频次]*[约为]?(\d{1,3})[次下]", "field": "cough_freq", "transform": "int", "priority": 7},
        {"pattern": r"延迟就诊[约为]?(\d{1,3})[天日]", "field": "delay_days", "transform": "int", "priority": 8},
        {"pattern": r"居住[条件环境]*[：:]*\s*(极差|差|一般|好|极好|较差|良好)", "field": "family_living_conditions", "transform": "rating", "priority": 8},
        {"pattern": r"BMI[为是]?[约为]?(\d{1,2}(\.\d)?)", "field": "bmi", "transform": "float", "priority": 6},
        {"pattern": r"体重[约为]?(\d{2,3})\s*(kg|公斤|千克)?", "field": "weight", "transform": "float", "priority": 5},
        {"pattern": r"(汉族|维吾尔族|哈萨克族|回族|蒙古族|藏族|其他)", "field": "ethnicity", "transform": "str", "priority": 8},
        {"pattern": r"[\u4e00-\u9fff]{2,4}族", "field": "ethnicity", "transform": "str", "priority": 6},
    ]


# ==============================================================================
# 二、段落分割
# ==============================================================================

def split_paragraphs(text):
    """按换行符分段，段内保留完整语义。

    策略：
    1. 先按换行符分段（保持语义边界）
    2. 检查每段是否过长（>200字符），过长则再按句末标点拆分
    3. 过滤空段落

    参数：
        text (str): 输入文本

    返回：
        list[str]: 段落列表
    """
    if not text or not isinstance(text, str):
        return []

    # 先按换行符分段
    raw_paragraphs = re.split(r'\n\r?|\r\n?', text)

    paragraphs = []
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        # 段落过长时按句末标点细分
        if len(para) > 200:
            sub_paras = re.split(r'[。！？!?；;]+', para)
            for sp in sub_paras:
                sp = sp.strip()
                if sp and len(sp) >= 2:
                    paragraphs.append(sp)
        else:
            paragraphs.append(para)

    return paragraphs


def _merge_sparse_paragraphs(paragraphs, min_fields_per_record=1):
    """段间合并：连续段落提取到的字段数很少时，合并为一条记录。

    当连续段落每段只提取到 ≤min_fields_per_record 个字段时，
    将它们合并，直到某个段落有更多字段。

    参数：
        paragraphs (list[str]): 段落列表
        min_fields_per_record (int): 最小字段数阈值

    返回：
        list[str]: 合并后的段落列表
    """
    if not paragraphs:
        return []

    merged = []
    buffer = ''
    buffer_count = 0

    # 简单计数：用常见字段模式估算段落中的字段数
    _field_pattern = re.compile(
        r'(年龄|痰涂片|空洞|卡介苗|治疗|通风|咳嗽|延迟|症状|接触|性别|民族|'
        r'BMI|体重|结核|既往|慢性病|高危|居住|场景|距离|频次|周期|确诊)'
    )

    for para in paragraphs:
        field_count = len(_field_pattern.findall(para))

        if field_count <= min_fields_per_record:
            # 稀疏段落，合并到缓冲区
            separator = '；' if buffer else ''
            buffer = buffer + separator + para if buffer else para
            buffer_count += 1
        else:
            # 有足够字段的段落，先输出缓冲区
            if buffer:
                merged.append(buffer)
                buffer = ''
                buffer_count = 0
            merged.append(para)

    # 输出剩余缓冲区
    if buffer:
        merged.append(buffer)

    if len(merged) < len(paragraphs):
        LOGGER.debug("段间合并: %d 段 → %d 段", len(paragraphs), len(merged))

    return merged


def _detect_record_boundaries(paragraphs):
    """检测记录边界：识别分隔符和重复字段。

    规则：
    1. 空行或分隔符（---, ***, ===）视为记录边界
    2. 同一字段（如 age）在短距离内再次出现，视为新记录起点

    参数：
        paragraphs (list[str]): 段落列表

    返回：
        list[int]: 记录边界索引列表（每个边界是新记录的开始）
    """
    if not paragraphs:
        return []

    boundaries = [0]  # 第一条记录从索引0开始
    separator_pattern = re.compile(r'^[-*=]{3,}$')
    _age_pattern = re.compile(r'(年龄|岁\d|age)')

    for i, para in enumerate(paragraphs):
        if i == 0:
            continue

        # 规则1：分隔符行
        if separator_pattern.match(para.strip()):
            boundaries.append(i + 1)  # 下一条记录从 i+1 开始
            continue

        # 规则2：段落开头是"患者编号"等明显的新记录标记
        if re.match(r'^(患者|病例|病人|编号|记录|第\d+|No\.?\d+)', para.strip()):
            boundaries.append(i)
            continue

        # 规则3：在相邻段落中检测到重复的年龄/姓名字段
        if i >= 1 and _age_pattern.search(para):
            prev_para = paragraphs[i - 1]
            if _age_pattern.search(prev_para):
                # 两个连续段落都有年龄，可能是两条记录
                boundaries.append(i)

    return sorted(set(boundaries))


# ==============================================================================
# 三、值标准化
# ==============================================================================

def _normalize_text_value(value, field, transform_type):
    """将正则匹配到的文本值转换为标准格式。

    参数：
        value (str): 正则匹配到的文本值
        field (str): 目标字段名
        transform_type (str): 转换类型

    返回：
        转换后的值
    """
    if value is None:
        return None

    value = str(value).strip()

    if transform_type == 'int':
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    if transform_type == 'float':
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    if transform_type == 'clinical':
        # 委托到 CLINICAL_TEXT_MAP
        if value in CLINICAL_TEXT_MAP:
            return CLINICAL_TEXT_MAP[value]
        # 尝试数字解析
        try:
            return int(float(value))
        except (ValueError, TypeError):
            # 返回 None（而非 0），由缺失值填充模块处理。
            # 0 不是临床字段的有效值（如 sputum_smear 有效值为 1-2），
            # 不可识别的文本不应被静默转换为无效值。
            return None

    if transform_type == 'bool':
        if value in BOOL_TEXT_MAP:
            return BOOL_TEXT_MAP[value]
        # 正向含义 → 1
        if value in ['有', '存在', '可见', '是', '已接种', '阳性', '涂阳', '++']:
            return 1
        if value in ['无', '未见', '否', '未接种', '阴性', '涂阴', '--']:
            return 0
        # 无法识别的文本返回 None（而非 0），由缺失值填充模块处理。
        # 0 可能被误解为"否"，与 clinical 类型问题一致。
        return None

    if transform_type == 'bool_positive':
        # 正向模式：匹配到即视为 1
        return 1

    if transform_type == 'distance':
        if value in DISTANCE_TEXT_MAP:
            return DISTANCE_TEXT_MAP[value]
        return value

    if transform_type == 'setting':
        if value in SETTING_TEXT_MAP:
            return SETTING_TEXT_MAP[value]
        return value

    if transform_type == 'rating':
        if value in RATING_TEXT_MAP:
            return RATING_TEXT_MAP[value]
        try:
            val = int(value)
            return max(1, min(5, val))
        except (ValueError, TypeError):
            return 3

    if transform_type == 'gender':
        if value in ['男性', '男', 'male', 'Male']:
            return '男'
        if value in ['女性', '女', 'female', 'Female']:
            return '女'
        return value

    if transform_type == 'str':
        return value

    return value


# ==============================================================================
# 四、正则文本抽取
# ==============================================================================

def extract_fields_from_text(text, templates=None):
    """对输入文本执行正则匹配，提取结构化字段。

    流程：
    1. 段落分割 → split_paragraphs()
    2. 段间合并 → _merge_sparse_paragraphs()
    3. 记录边界检测 → _detect_record_boundaries()（多患者文本）
    4. 对每段文本独立执行所有字段的正则匹配
    5. 对每个提取结果标注 confidence 和 source

    参数：
        text (str): 输入文本
        templates (list[dict]|None): 正则模板，None 时自动加载

    返回：
        list[dict]: 每条记录是一个字典，包含提取的字段和元数据
    """
    if templates is None:
        templates = load_templates()

    if not text or not isinstance(text, str):
        return []

    paragraphs = split_paragraphs(text)

    # 段间合并：将稀疏段落合并为完整记录
    paragraphs = _merge_sparse_paragraphs(paragraphs)

    if not paragraphs:
        return []

    # 记录边界检测（多患者场景）
    boundaries = _detect_record_boundaries(paragraphs)
    if len(boundaries) <= 1:
        # 单条记录或无法检测边界
        boundaries = [0]

    records = []

    for b_idx, start in enumerate(boundaries):
        end = boundaries[b_idx + 1] if b_idx + 1 < len(boundaries) else len(paragraphs)
        segment_paragraphs = paragraphs[start:end]

        record = {}
        field_sources = {}
        field_confidences = {}

        for para in segment_paragraphs:
            for template in templates:
                pattern = template.get('pattern', '')
                field = template.get('field', '')
                transform = template.get('transform', 'str')
                priority = template.get('priority', 5)

                try:
                    match = re.search(pattern, para)
                except re.error as e:
                    LOGGER.debug("正则模板 '%s' 编译失败: %s", pattern, e)
                    continue

                if match:
                    # 提取匹配到的值（取第一个捕获组，若无则取整个匹配）
                    if match.groups():
                        raw_value = match.group(1)
                    else:
                        raw_value = match.group(0)

                    normalized = _normalize_text_value(raw_value, field, transform)

                    # 改进的置信度公式：基于优先级 + 捕获组匹配质量
                    confidence = _compute_regex_confidence(priority, match, para, pattern)

                    # 冲突处理：保留优先级更高的值
                    if field in record:
                        existing_priority = field_sources.get(field, {}).get('priority', 0)
                        if priority > existing_priority:
                            record[field] = normalized
                            field_sources[field] = {
                                'source': 'regex',
                                'priority': priority,
                                'raw_value': raw_value,
                                'pattern': pattern,
                            }
                            field_confidences[field] = confidence
                    else:
                        record[field] = normalized
                        field_sources[field] = {
                            'source': 'regex',
                            'priority': priority,
                            'raw_value': raw_value,
                            'pattern': pattern,
                        }
                        field_confidences[field] = confidence

        if record:
            record['_confidence'] = field_confidences
            record['_source'] = {'source': 'regex'}
            record['_field_sources'] = field_sources
            records.append(record)

    return records


def _compute_regex_confidence(priority, match, paragraph, pattern):
    """改进的置信度计算：综合优先级、捕获组长度、匹配完整度。

    参数：
        priority (int): 模板优先级
        match (re.Match): 正则匹配对象
        paragraph (str): 段落文本
        pattern (str): 正则表达式

    返回：
        float: 置信度 (0.0-1.0)
    """
    base = 0.60 + priority * 0.03  # 基础分 0.75-0.90（对应内置模板 priority 5-10）

    # 捕获组质量调整
    if match.groups():
        # 捕获组非空且长度合理
        captured = match.group(1) or ''
        if len(captured) >= 1:
            base += 0.05  # 有效捕获组加分
        if len(captured) >= 3:
            base += 0.03  # 长捕获组更可靠
    else:
        base -= 0.05  # 无捕获组，使用整个匹配，可靠性略低

    # 匹配位置调整：仅对段落末尾的匹配施加轻微惩罚
    # 段落开头的匹配不惩罚（临床文本关键信息如"患者年龄35岁"通常出现在开头）
    match_start = match.start()
    if match_start > len(paragraph) * 0.9:
        base -= 0.02

    return max(0.0, min(1.0, base))


# ==============================================================================
# 五、LLM 辅助抽取（可选，委托 tb_risk.ai.parser）
# ==============================================================================

def extract_fields_with_llm(text, model_path=None):
    """可选 LLM 增强抽取。

    委托 `tb_risk.ai.parser.extract_with_llm` 完成实际抽取工作。
    支持两种模式：
    - **云端 LLM**（OpenAI 兼容协议）：当 `model_path=None` 且环境变量
      `TB_AI_API_KEY`/`TB_AI_PROVIDER` 已配置时启用。
    - **本地部署模型**（transformers + torch）：当 `model_path` 提供时启用，
      自动构造 `AIConfig(local_model_path=model_path)`。

    未配置 / 调用失败 / 无有效记录时返回 None（调用方降级到正则抽取）。

    参数：
        text (str): 输入文本
        model_path (str|None): 本地模型路径，None 时走云端 LLM 配置

    返回：
        list[dict]|None: 抽取的记录列表，每条含字段值 + _confidence + _source。
        未配置/失败时返回 None。
    """
    try:
        from tb_risk.ai.parser import extract_with_llm as _ai_extract
        from tb_risk.ai.config import AIConfig
    except ImportError as e:
        LOGGER.debug("tb_risk.ai 子包不可用，跳过 LLM 抽取: %s", e)
        return None

    # 构造 config：model_path 优先 → 本地模式；否则 None（ai.parser 自动从环境加载）
    config = None
    if model_path:
        config = AIConfig(local_model_path=model_path)

    return _ai_extract(text, client=None, config=config)


# ==============================================================================
# 六、多来源融合
# ==============================================================================

def _merge_field_values(regex_records, llm_records):
    """多来源字段值融合。

    优先级：结构化表格数据 > 正则抽取结果 > LLM 抽取结果
    冲突时保留置信度最高的值并记录冲突日志。

    参数：
        regex_records (list[dict]): 正则抽取结果
        llm_records (list[dict]|None): LLM 抽取结果

    返回：
        list[dict]: 融合后的记录列表
    """
    conflicts = []

    if not llm_records:
        return regex_records

    # 简单策略：合并两条记录列表，取非空值
    merged = []
    max_len = max(len(regex_records), len(llm_records))

    for i in range(max_len):
        record = {}
        regex_rec = regex_records[i] if i < len(regex_records) else {}
        llm_rec = llm_records[i] if i < len(llm_records) else {}

        # 收集所有字段
        all_fields = set(list(regex_rec.keys()) + list(llm_rec.keys()))
        # 去除元数据字段
        all_fields = {f for f in all_fields if not f.startswith('_')}

        for field in all_fields:
            regex_val = regex_rec.get(field)
            llm_val = llm_rec.get(field)
            regex_conf = regex_rec.get('_confidence', {}).get(field, 0.75)
            llm_conf = llm_rec.get('_confidence', {}).get(field, 0.65)

            if regex_val is not None and llm_val is not None and regex_val != llm_val:
                conflicts.append({
                    'field': field,
                    'record_index': i,
                    'regex_value': regex_val,
                    'llm_value': llm_val,
                    'regex_confidence': regex_conf,
                    'llm_confidence': llm_conf,
                })

            # 优先正则（置信度更高），回退到 LLM
            if regex_val is not None:
                record[field] = regex_val
            elif llm_val is not None:
                record[field] = llm_val

        if record:
            record['_source'] = 'merged'
            record['_conflicts'] = [c for c in conflicts if c['record_index'] == i]
            merged.append(record)

    return merged


# ==============================================================================
# 七、主入口
# ==============================================================================

def parse_text_input(text, use_llm=False, model_path=None, templates=None):
    """自由文本输入主入口。

    流程：段落分割 → 正则提取 → 可选 LLM 融合 → 多来源融合

    当 `use_llm=True` 时调用 `extract_fields_with_llm` 增强抽取：
    - AI 已配置（环境变量或 model_path）→ 真正调用 LLM，结果与正则融合
    - AI 未配置 → LLM 返回 None，自动降级到纯正则抽取，`meta['llm_used']=False`

    参数：
        text (str): 输入文本
        use_llm (bool): 是否启用 LLM 增强（需配置 AI 才真正生效）
        model_path (str|None): 本地 LLM 模型路径，None 时走云端 LLM 配置
        templates (list[dict]|None): 正则模板

    返回：
        (records, meta): 记录列表和元数据
    """
    if templates is None:
        templates = load_templates()

    meta = {
        'input_type': 'text',
        'paragraph_count': 0,
        'regex_records': 0,
        'llm_records': 0,
        'merged_records': 0,
        'conflicts': [],
        'llm_used': False,
    }

    # 段落分割
    paragraphs = split_paragraphs(text)
    meta['paragraph_count'] = len(paragraphs)

    if not paragraphs:
        return [], meta

    # 正则提取
    regex_records = extract_fields_from_text(text, templates)
    meta['regex_records'] = len(regex_records)

    # 可选 LLM 增强
    llm_records = None
    if use_llm:
        llm_records = extract_fields_with_llm(text, model_path)
        if llm_records:
            meta['llm_records'] = len(llm_records)
            meta['llm_used'] = True

    # 多来源融合
    records = _merge_field_values(regex_records, llm_records)
    meta['merged_records'] = len(records)

    # 收集冲突
    for rec in records:
        conflicts = rec.get('_conflicts', [])
        if conflicts:
            meta['conflicts'].extend(conflicts)

    return records, meta


def detect_input_type(filepath):
    """检测输入类型。

    检测优先级：分隔符结构 > JSON 格式 > 中文文本启发式

    参数：
        filepath (str): 文件路径

    返回：
        str: 'structured_csv' / 'structured_excel' / 'structured_json' / 'text' / 'unknown'
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ('.csv', '.tsv'):
        return 'structured_csv'
    if ext in ('.xlsx', '.xls'):
        return 'structured_excel'
    if ext in ('.json', '.jsonl'):
        return 'structured_json'
    if ext in ('.txt', '.text'):
        return 'text'

    # 尝试读取文件前几行判断
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            first_lines = ''.join(f.readline() for _ in range(5))

        # 优先级1：检测 JSON 格式
        stripped = first_lines.strip()
        if stripped.startswith('[') or stripped.startswith('{'):
            return 'structured_json'

        # 优先级2：检测分隔符结构（逗号/制表符 + 每行列数一致）
        if '\t' in first_lines or ',' in first_lines:
            lines = [l for l in first_lines.split('\n') if l.strip()]
            if len(lines) >= 2:
                # 检查每行的分隔符数量是否一致
                if '\t' in first_lines:
                    col_counts = [len(l.split('\t')) for l in lines]
                else:
                    col_counts = [len(l.split(',')) for l in lines]
                if len(set(col_counts)) <= 1 and col_counts[0] >= 2:
                    # 每行列数一致且 ≥2 → 结构化表格
                    if any(keyword in first_lines.lower() for keyword in
                           ['age', '年龄', 'name', '姓名', 'id', '编号', '性别', 'gender']):
                        return 'structured_csv'
                    # 即使没有关键词，列数一致也倾向于结构化
                    return 'structured_csv'

        # 优先级3：检测中文文本（中文字符 > 30 且无清晰分隔符结构）
        chinese_chars = sum(1 for c in first_lines if '\u4e00' <= c <= '\u9fff')
        if chinese_chars > 30:
            # 确认没有分隔符结构
            if '\t' not in first_lines:
                return 'text'

    except Exception as e:
        LOGGER.debug("文件类型检测失败，默认unknown: %s", e)

    return 'unknown'