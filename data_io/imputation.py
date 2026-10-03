#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""缺失值智能填充模块

提供众数填充、条件中位数填充、场景默认值回退和缺失值标记。

策略选择：
- 二值字段（BOOLEAN_FIELDS、CLINICAL_FIELDS）→ 众数填充
- 数值字段（NUMERIC_FIELDS）且存在 scenario_type → 条件中位数
- 数值字段无 scenario_type → 简单中位数
- 所有策略均失败 → 场景默认值回退
- 最终无法填充 → MISSING_VALUE_MARKER (-999)
"""

import logging
import math

try:
    from .conf import (
        BOOLEAN_FIELDS, CLINICAL_FIELDS, RATING_FIELDS, NUMERIC_FIELDS,
        DEFAULT_VALUES, CONTACT_TYPE_OVERRIDES, SCENE_DEFAULTS,
        MISSING_VALUE_MARKER,
    )
except ImportError:
    BOOLEAN_FIELDS = []
    CLINICAL_FIELDS = []
    RATING_FIELDS = []
    NUMERIC_FIELDS = []
    DEFAULT_VALUES = {}
    CONTACT_TYPE_OVERRIDES = {}
    SCENE_DEFAULTS = {}
    MISSING_VALUE_MARKER = -999

LOGGER = logging.getLogger("tb_risk.data_io.imputation")


# ==============================================================================
# 一、分类字段众数填充
# ==============================================================================

def impute_mode(data_list, column_name, min_valid_count=5):
    """分类字段众数填充。

    适用于：bcg_vaccine、has_tb、has_symptoms、past_illness、is_high_risk 等二值字段。

    参数：
        data_list (list[dict]): 数据记录列表
        column_name (str): 列名
        min_valid_count (int): 最小有效值数量，少于此值时降级为场景默认值

    返回：
        (filled_count, strategy, fill_value): 填充记录数、策略名('mode')、众数值
    """
    values = []
    for data in data_list:
        val = data.get(column_name)
        if val is not None and val != '' and val != MISSING_VALUE_MARKER:
            try:
                values.append(int(float(val)))
            except (ValueError, TypeError):
                # 尝试字符串值
                values.append(str(val).strip())

    if len(values) < min_valid_count:
        LOGGER.debug("列 '%s': 有效值仅 %d 条 (<%d)，众数填充不可靠，跳过",
                     column_name, len(values), min_valid_count)
        return 0, 'mode', None

    # 计算众数
    from collections import Counter
    counter = Counter(values)
    mode_value = counter.most_common(1)[0][0]

    # 填充缺失值
    filled_count = 0
    for data in data_list:
        val = data.get(column_name)
        if val is None or val == '' or val == MISSING_VALUE_MARKER:
            data[column_name] = mode_value
            filled_count += 1

    if filled_count > 0:
        LOGGER.debug("列 '%s': 众数填充 %d 条, 值=%s", column_name, filled_count, mode_value)

    return filled_count, 'mode', mode_value


# ==============================================================================
# 二、条件中位数填充
# ==============================================================================

def impute_conditional_median(data_list, column_name, group_by_field='scenario_type'):
    """条件中位数填充。

    适用于：age 等数值字段。
    先按分组字段（如 scenario_type）分组，再在组内取中位数。
    当主分组字段不存在时，尝试自适应选择备选分组字段。

    参数：
        data_list (list[dict]): 数据记录列表
        column_name (str): 列名
        group_by_field (str): 首选分组字段，默认 'scenario_type'

    返回：
        (filled_count, strategy, fill_value): 填充记录数、策略名('conditional_median')、各组中位数
    """
    # 确定实际可用的分组字段
    effective_group_field = _select_grouping_field(data_list, column_name, group_by_field)

    if effective_group_field is None:
        # 无法分组，回退到简单中位数（标签与 impute_missing 回退路径一致）
        filled_count, _, fill_value = _impute_median(data_list, column_name)
        if filled_count > 0:
            return filled_count, 'median_fallback', fill_value
        return 0, 'median_fallback', None

    # 按分组收集有效值
    group_values = {}
    for data in data_list:
        group = data.get(effective_group_field, 'unknown')
        if group not in group_values:
            group_values[group] = []

        val = data.get(column_name)
        if val is not None and val != '' and val != MISSING_VALUE_MARKER:
            try:
                group_values[group].append(float(val))
            except (ValueError, TypeError):
                pass

    # 计算各组中位数
    group_medians = {}
    for group, vals in group_values.items():
        if len(vals) >= 2:
            sorted_vals = sorted(vals)
            n = len(sorted_vals)
            if n % 2 == 0:
                median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
            else:
                median = sorted_vals[n // 2]
            group_medians[group] = median

    # 如果没有足够的数据分组，回退到简单中位数
    if not group_medians:
        all_values = []
        for vals in group_values.values():
            all_values.extend(vals)
        if len(all_values) >= 2:
            sorted_all = sorted(all_values)
            n = len(sorted_all)
            if n % 2 == 0:
                median = (sorted_all[n // 2 - 1] + sorted_all[n // 2]) / 2
            else:
                median = sorted_all[n // 2]
            group_medians['_fallback'] = median

    # 填充缺失值
    filled_count = 0
    for data in data_list:
        val = data.get(column_name)
        if val is None or val == '' or val == MISSING_VALUE_MARKER:
            group = data.get(effective_group_field, 'unknown')
            median = group_medians.get(group, group_medians.get('_fallback'))
            if median is not None:
                try:
                    if math.isfinite(median) and median == int(median):
                        data[column_name] = round(median)
                    else:
                        data[column_name] = median
                except (ValueError, OverflowError):
                    data[column_name] = median
                filled_count += 1

    if filled_count > 0:
        LOGGER.debug("列 '%s': 条件中位数填充 %d 条 (分组字段=%s), 各组=%s",
                     column_name, filled_count, effective_group_field, group_medians)

    return filled_count, 'conditional_median', group_medians


def _select_grouping_field(data_list, column_name, preferred_field='scenario_type'):
    """自适应选择分组字段。

    优先级：preferred_field → exposure_setting → contact_distance → None

    在选择分组字段时，验证分组是否对目标字段有实际区分度：
    如果各组的目标字段中位数几乎相同，说明该分组字段无区分度，跳过。

    参数：
        data_list (list[dict]): 数据记录列表
        column_name (str): 目标字段名（需要被填充的字段）
        preferred_field (str): 首选分组字段

    返回：
        str|None: 可用的分组字段名，或 None
    """
    # 候选分组字段
    candidates = [
        preferred_field,
        'scenario_type',
        'exposure_setting',
        'contact_distance',
        'workplace_type',
    ]

    # 去重并保持顺序
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    for field in unique_candidates:
        # 检查该字段是否有足够多的不同值（至少2个不同值才有分组意义）
        distinct_values = set()
        valid_count = 0
        for data in data_list:
            val = data.get(field)
            if val is not None and val != '':
                distinct_values.add(val)
                valid_count += 1

        if len(distinct_values) < 2 or valid_count < 3:
            continue

        # 分组有效性验证：检查目标字段在各组中的中位数是否有区分度
        if not _grouping_has_discriminative_power(data_list, field, column_name):
            LOGGER.debug("分组字段 '%s' 对 '%s' 无区分度，跳过", field, column_name)
            continue

        return field

    return None


def _grouping_has_discriminative_power(data_list, group_field, target_column):
    """验证分组字段对目标字段是否有区分度。

    计算各组的目标字段中位数，若组间中位数差异过小，说明该分组字段
    无法有效区分目标字段的分布。

    阈值根据数据总量自适应调整：
    - 总有效值 < 20 条 → 10%（小样本放宽）
    - 20 ≤ 总有效值 < 50 → 12.5%
    - 总有效值 ≥ 50 → 15%

    参数：
        data_list (list[dict]): 数据记录列表
        group_field (str): 分组字段名
        target_column (str): 目标字段名

    返回：
        bool: 是否有区分度
    """
    # 按分组收集目标字段的有效值
    group_values = {}
    for data in data_list:
        group = data.get(group_field, 'unknown')
        if group not in group_values:
            group_values[group] = []
        val = data.get(target_column)
        if val is not None and val != '' and val != MISSING_VALUE_MARKER:
            try:
                group_values[group].append(float(val))
            except (ValueError, TypeError):
                pass

    # 计算各组中位数
    group_medians = {}
    for group, vals in group_values.items():
        if len(vals) >= 2:
            sorted_vals = sorted(vals)
            n = len(sorted_vals)
            if n % 2 == 0:
                median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
            else:
                median = sorted_vals[n // 2]
            group_medians[group] = median

    # 至少需要 2 个组有有效中位数
    if len(group_medians) < 2:
        return False

    medians = list(group_medians.values())
    median_range = max(medians) - min(medians)
    overall_mean = sum(medians) / len(medians)

    # 相对差异：组间中位数范围 / 总均值
    relative_diff = median_range / (abs(overall_mean) + 1e-8)

    # 自适应阈值：小样本放宽，大样本收紧
    total_valid = sum(len(vals) for vals in group_values.values())
    if total_valid < 20:
        threshold = 0.10
    elif total_valid < 50:
        threshold = 0.125
    else:
        threshold = 0.15

    LOGGER.debug("分组 '%s' 对 '%s': 有效值=%d, 相对差异=%.3f, 阈值=%.3f",
                 group_field, target_column, total_valid, relative_diff, threshold)

    return relative_diff >= threshold


# ==============================================================================
# 三、场景默认值回退
# ==============================================================================

def impute_scene_default(data_list, column_name, scene_defaults=None,
                         scenario_field='scenario_type'):
    """场景默认值回退。

    当某字段完全缺失且无法统计推断时，回退到当前场景的预设默认值。

    参数：
        data_list (list[dict]): 数据记录列表
        column_name (str): 列名
        scene_defaults (dict|None): 场景默认值，None 时使用 SCENE_DEFAULTS
        scenario_field (str): 场景字段名

    返回：
        (filled_count, strategy, fill_value): 填充记录数、策略名('scene_default')、使用的默认值
    """
    if scene_defaults is None:
        scene_defaults = SCENE_DEFAULTS

    filled_count = 0
    default_value = None

    for data in data_list:
        val = data.get(column_name)
        if val is None or val == '' or val == MISSING_VALUE_MARKER:
            scenario = data.get(scenario_field, 'rural_family')
            # 在场景默认值中查找该字段
            scene_default = None
            if scenario in scene_defaults:
                # 尝试在 patient / family_member / social_contact 中查找
                for role in ['patient', 'family_member', 'social_contact']:
                    if role in scene_defaults[scenario]:
                        if column_name in scene_defaults[scenario][role]:
                            scene_default = scene_defaults[scenario][role][column_name]
                            break

            if scene_default is not None:
                data[column_name] = scene_default
                default_value = scene_default
                filled_count += 1

    if filled_count > 0:
        LOGGER.debug("列 '%s': 场景默认值填充 %d 条, 值=%s",
                     column_name, filled_count, default_value)

    return filled_count, 'scene_default', default_value


# ==============================================================================
# 四、简单中位数填充
# ==============================================================================

def _impute_median(data_list, column_name):
    """简单中位数填充（回退策略）。

    参数：
        data_list (list[dict]): 数据记录列表
        column_name (str): 列名

    返回：
        (filled_count, strategy, fill_value): 填充记录数、策略名('median')、中位数值
    """
    values = []
    for data in data_list:
        val = data.get(column_name)
        if val is not None and val != '' and val != MISSING_VALUE_MARKER:
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                pass

    if len(values) < 2:
        return 0, 'median', None

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 0:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    else:
        median = sorted_vals[n // 2]

    filled_count = 0
    for data in data_list:
        val = data.get(column_name)
        if val is None or val == '' or val == MISSING_VALUE_MARKER:
            try:
                if math.isfinite(median) and median == int(median):
                    data[column_name] = round(median)
                else:
                    data[column_name] = median
            except (ValueError, OverflowError):
                data[column_name] = median
            filled_count += 1

    if filled_count > 0:
        LOGGER.debug("列 '%s': 简单中位数填充 %d 条, 值=%s", column_name, filled_count, median)

    return filled_count, 'median', median


# ==============================================================================
# 五、统一缺失值填充接口
# ==============================================================================

def _compute_cardinality(data_list, column_name):
    """计算字段的基数（不同值的数量）。

    忽略 None、空字符串和 MISSING_VALUE_MARKER。

    参数：
        data_list (list[dict]): 数据记录列表
        column_name (str): 列名

    返回：
        int|None: 不同值数量，或 None（无有效值时）
    """
    distinct = set()
    for data in data_list:
        val = data.get(column_name)
        if val is not None and val != '' and val != MISSING_VALUE_MARKER:
            try:
                distinct.add(float(val))
            except (ValueError, TypeError):
                distinct.add(str(val))
    return len(distinct) if distinct else None


def _compute_top_value_ratio(data_list, column_name):
    """计算字段中最高频值的占比。

    参数：
        data_list (list[dict]): 数据记录列表
        column_name (str): 列名

    返回：
        float|None: 最高频值占比，或 None（无有效值时）
    """
    from collections import Counter
    values = []
    for data in data_list:
        val = data.get(column_name)
        if val is not None and val != '' and val != MISSING_VALUE_MARKER:
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                values.append(str(val))
    if not values:
        return None
    counter = Counter(values)
    _, top_count = counter.most_common(1)[0]
    return top_count / len(values)


def impute_missing(data_list, column_name, strategy='auto', scene_defaults=None):
    """统一缺失值填充接口。

    自动选择策略：
    1. 二值字段（BOOLEAN_FIELDS、CLINICAL_FIELDS）→ 众数填充
    2. 数值字段（NUMERIC_FIELDS）且存在 scenario_type → 条件中位数
    3. 数值字段无 scenario_type → 简单中位数
    4. 所有策略均失败 → 场景默认值回退
    5. 最终无法填充 → MISSING_VALUE_MARKER (-999)

    参数：
        data_list (list[dict]): 数据记录列表
        column_name (str): 列名
        strategy (str): 填充策略 ('auto' / 'mode' / 'conditional_median' / 'median' / 'scene_default')
        scene_defaults (dict|None): 场景默认值

    返回：
        (filled_count, strategy_used, fill_value): 填充记录数、使用的策略、填充值
    """
    if not data_list:
        return 0, 'none', None

    if scene_defaults is None:
        scene_defaults = SCENE_DEFAULTS

    # 策略自动选择
    if strategy == 'auto':
        if column_name in BOOLEAN_FIELDS or column_name in CLINICAL_FIELDS:
            strategy = 'mode'
        elif column_name in NUMERIC_FIELDS:
            # 数据驱动微调：数值字段基数 ≤ 5 时用众数（行为接近有序分类）
            cardinality = _compute_cardinality(data_list, column_name)
            if cardinality is not None and cardinality <= 5:
                strategy = 'mode'
            else:
                has_scenario = any(
                    data.get('scenario_type') is not None and data.get('scenario_type') != ''
                    for data in data_list
                )
                if has_scenario:
                    strategy = 'conditional_median'
                else:
                    strategy = 'median'
        else:
            # 数据驱动微调：分类字段检查分布倾斜度
            top_ratio = _compute_top_value_ratio(data_list, column_name)
            if top_ratio is not None and top_ratio >= 0.80:
                strategy = 'mode'
            else:
                strategy = 'scene_default'

    # 执行填充
    if strategy == 'mode':
        filled_count, _, fill_value = impute_mode(data_list, column_name)
        if filled_count > 0:
            return filled_count, 'mode', fill_value

    if strategy == 'conditional_median':
        filled_count, _, fill_value = impute_conditional_median(data_list, column_name)
        if filled_count > 0:
            return filled_count, 'conditional_median', fill_value
        # 回退到简单中位数
        filled_count, _, fill_value = _impute_median(data_list, column_name)
        if filled_count > 0:
            return filled_count, 'median_fallback', fill_value

    if strategy == 'median':
        filled_count, _, fill_value = _impute_median(data_list, column_name)
        if filled_count > 0:
            return filled_count, 'median', fill_value

    if strategy in ('mode', 'conditional_median', 'median', 'scene_default'):
        # 场景默认值回退
        filled_count, _, fill_value = impute_scene_default(
            data_list, column_name, scene_defaults
        )
        if filled_count > 0:
            return filled_count, 'scene_default', fill_value

    if strategy in ('mode', 'conditional_median', 'median', 'scene_default', 'auto'):
        # 最终回退：标记为 -999
        filled_count, _, _ = mark_unfillable(data_list, column_name)
        if filled_count > 0:
            return filled_count, 'missing_marker', MISSING_VALUE_MARKER

    return 0, 'none', None


# ==============================================================================
# 六、缺失值标记
# ==============================================================================

def mark_unfillable(data_list, column_name):
    """将无法填充的字段标记为 MISSING_VALUE_MARKER (-999)。

    参数：
        data_list (list[dict]): 数据记录列表
        column_name (str): 列名

    返回：
        (marked_count, strategy, fill_value): 标记记录数、策略名('missing_marker')、标记值(-999)
    """
    marked_count = 0
    for data in data_list:
        val = data.get(column_name)
        if val is None or val == '':
            data[column_name] = MISSING_VALUE_MARKER
            marked_count += 1

    if marked_count > 0:
        LOGGER.warning("列 '%s': %d 条记录标记为 %d (无法填充)",
                       column_name, marked_count, MISSING_VALUE_MARKER)

    return marked_count, 'missing_marker', MISSING_VALUE_MARKER


def filter_missing_marker(record, exclude_fields=None):
    """过滤记录中的 MISSING_VALUE_MARKER 值。

    在下游消费端（评分引擎、ML 预测器）使用前，应将 -999 替换为 None
    或默认值，防止标记值被当作真实特征值参与计算。

    参数：
        record (dict): 数据记录
        exclude_fields (list[str]|None): 不检查的字段列表（如元数据字段）

    返回：
        dict: 过滤后的记录
    """
    if exclude_fields is None:
        exclude_fields = ['_confidence', '_source', '_field_sources', '_conflicts']

    filtered = {}
    for key, value in record.items():
        if key in exclude_fields:
            filtered[key] = value
            continue
        if isinstance(value, (int, float)) and value == MISSING_VALUE_MARKER:
            filtered[key] = None  # 替换为 None 而非保留 -999
            LOGGER.debug("字段 '%s' 的 MISSING_VALUE_MARKER 已过滤为 None", key)
        elif isinstance(value, str) and value.strip() == str(MISSING_VALUE_MARKER):
            filtered[key] = None
            LOGGER.debug("字段 '%s' 的字符串 MISSING_VALUE_MARKER 已过滤为 None", key)
        else:
            filtered[key] = value

    return filtered


def filter_missing_marker_batch(records, exclude_fields=None):
    """批量过滤记录中的 MISSING_VALUE_MARKER 值。

    参数：
        records (list[dict]): 记录列表
        exclude_fields (list[str]|None): 不检查的字段列表

    返回：
        list[dict]: 过滤后的记录列表
    """
    return [filter_missing_marker(r, exclude_fields) for r in records]


# ==============================================================================
# 七、统一默认值填充
# ==============================================================================

def fill_record_defaults(record, default_type='family', scene_defaults=None):
    """统一默认值填充。

    提取 mixins._fill_missing_fields 中的 setdefault 逻辑。

    参数：
        record (dict): 单条数据记录
        default_type (str): 类型 ('family' / 'social' / 'patient')
        scene_defaults (dict|None): 场景默认值

    返回：
        list[str]: 被填充的字段列表
    """
    if scene_defaults is None:
        scene_defaults = SCENE_DEFAULTS

    filled_fields = []

    # 条件默认值处理
    if record.get('past_illness') == 0:
        record['past_illness_type'] = 'none'
        filled_fields.append('past_illness_type')

    # 基础默认值
    for field, default in DEFAULT_VALUES.items():
        if field not in record or record.get(field) is None or record.get(field) == '' or record.get(field) == MISSING_VALUE_MARKER:
            # 根据类型调整默认值（使用 CONTACT_TYPE_OVERRIDES）
            if default_type in CONTACT_TYPE_OVERRIDES and field in CONTACT_TYPE_OVERRIDES[default_type]:
                record[field] = CONTACT_TYPE_OVERRIDES[default_type][field]
            else:
                record[field] = default
            filled_fields.append(field)

    return filled_fields


def fill_missing_fields_batch(data_list, default_type='family', scene_defaults=None):
    """批量默认值填充。

    参数：
        data_list (list[dict]): 数据记录列表
        default_type (str): 类型
        scene_defaults (dict|None): 场景默认值

    返回：
        int: 填充的记录数
    """
    filled_count = 0
    for record in data_list:
        filled = fill_record_defaults(record, default_type, scene_defaults)
        if filled:
            filled_count += 1
    return filled_count