#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据质量评分模块

提供逐字段置信度评分和记录级质量汇总。

评分规则：
- 直接匹配（exact column match）→ 1.0
- 关键词匹配（keyword match）→ 0.85-0.95
- 模糊匹配（fuzzy Levenshtein/difflib）→ 0.7-0.85
- 正则抽取（regex）→ 0.80
- 用户反馈学习（user_feedback）→ 0.95
- 推断填充（imputed）→ 0.45-0.55
- 默认值回退（default fallback）→ 0.30
- 缺失标记（missing marker）→ 0.0
"""

import logging

LOGGER = logging.getLogger("tb_risk.data_io.scoring")

try:
    from .conf import MISSING_VALUE_MARKER
except ImportError:
    MISSING_VALUE_MARKER = -999


# ==============================================================================
# 一、来源 → 置信度映射
# ==============================================================================

SOURCE_CONFIDENCE = {
    # 列名匹配来源
    'exact': 1.0,
    'keyword': 0.90,
    'fuzzy_levenshtein': 0.75,
    'fuzzy_difflib': 0.70,
    'user_feedback': 0.95,
    # 文本抽取来源
    'regex': 0.80,
    'llm': 0.70,
    # 填充来源
    'imputed_mode': 0.50,
    'imputed_median': 0.45,
    'imputed_conditional_median': 0.55,
    'imputed_scene_default': 0.30,
    'imputed_missing_marker': 0.0,
    'imputed_median_fallback': 0.45,
    'missing_marker': 0.0,
    # 通用回退
    'fallback': 0.85,
    'default': 0.30,
    'direct': 0.95,
    'unknown': 0.40,
}

# 质量评级阈值
QUALITY_THRESHOLDS = {
    'excellent': 0.90,
    'good': 0.75,
    'fair': 0.50,
    'poor': 0.30,
    'critical': 0.0,
}

# 字段权重：必填字段权重高，可选字段权重低
# 权重影响记录级评分中该字段的贡献
FIELD_WEIGHTS = {
    # 必填核心字段（权重 3.0）
    'age': 3.0,
    'sputum_smear': 3.0,
    'has_cavity': 3.0,
    'is_confirmed': 3.0,
    'cumulative_exposure': 3.0,
    'treatment': 3.0,
    # 重要字段（权重 2.0）
    'bcg_vaccine': 2.0,
    'has_tb': 2.0,
    'has_symptoms': 2.0,
    'contact_distance': 2.0,
    'ventilation': 2.0,
    'exposure_setting': 2.0,
    'cough_freq': 2.0,
    'symptoms': 2.0,
    'is_high_risk': 2.0,
    'past_illness': 2.0,
    'active_tb': 2.0,
    'treatment_duration': 2.0,
    'delay_days': 2.0,
    # 一般字段（权重 1.5）
    'single_duration': 1.5,
    'freq_density': 1.5,
    'time_span': 1.5,
    'family_living_conditions': 1.5,
    'flp_percentage': 1.5,
    'hrsp_percentage': 1.5,
    # 可选字段（权重 1.0）
    'gender': 1.0,
    'ethnicity': 1.0,
    'occupation': 1.0,
    'district': 1.0,
    'record_id': 1.0,
    'past_illness_type': 1.0,
    'scenario_type': 1.0,
    'member_name': 1.0,
    'member_age': 1.0,
    'contact_name': 1.0,
    'contact_age': 1.0,
    'relationship': 1.0,
    'origin_altitude': 1.0,
    'workplace_type': 1.0,
    'idu_status': 1.0,
    'months_since_migration': 1.0,
    'bmi': 1.0,
    'weight': 1.0,
}
# 默认权重（未在 FIELD_WEIGHTS 中定义的字段）
DEFAULT_FIELD_WEIGHT = 1.0


# ==============================================================================
# 二、数据质量评分器
# ==============================================================================

class DataQualityScorer:
    """数据质量评分器

    每条记录的每个字段在提取后，根据来源赋予 0-1 的置信度分值。
    支持逐字段评分、记录级加权汇总和批量评分。

    用法：
        scorer = DataQualityScorer(flagged_threshold=0.6)
        score = scorer.score_field('age', 35, 'exact')
        record_score = scorer.score_record(record, field_sources)
        results = scorer.score_batch(records, field_sources_map)
    """

    def __init__(self, source_confidence=None, quality_thresholds=None,
                 field_weights=None, flagged_threshold=0.5):
        """初始化评分器。

        参数：
            source_confidence (dict|None): 自定义来源置信度映射
            quality_thresholds (dict|None): 自定义质量评级阈值
            field_weights (dict|None): 自定义字段权重（None 使用默认 FIELD_WEIGHTS）
            flagged_threshold (float): 需人工复核的分数阈值，默认 0.5
        """
        self.source_confidence = source_confidence or SOURCE_CONFIDENCE.copy()
        self.quality_thresholds = quality_thresholds or QUALITY_THRESHOLDS.copy()
        self.field_weights = field_weights or FIELD_WEIGHTS.copy()
        self.flagged_threshold = flagged_threshold

    # ------------------------------------------------------------------
    # 逐字段评分
    # ------------------------------------------------------------------

    def score_field(self, field_name, value, source, match_confidence=None):
        """计算单个字段的置信度分数 (0.0-1.0)。

        参数：
            field_name (str): 字段名
            value: 字段值
            source (str): 数据来源类型（如 'exact', 'keyword', 'regex' 等）
            match_confidence (float|None): 匹配时的额外置信度（如 fuzzy 匹配的相似度）

        返回：
            float: 0.0-1.0 的置信度分数
        """
        # 来源基础置信度
        base = self.source_confidence.get(source, self.source_confidence.get('unknown', 0.40))

        # 如果提供了匹配置信度，取加权
        if match_confidence is not None:
            base = 0.6 * base + 0.4 * match_confidence

        # 缺失值惩罚
        if value is None or value == '' or value == MISSING_VALUE_MARKER:
            base = min(base, 0.1)

        return min(1.0, max(0.0, base))

    # ------------------------------------------------------------------
    # 记录级评分
    # ------------------------------------------------------------------

    def score_record(self, record, field_sources=None):
        """计算单条记录的加权质量分。

        参数：
            record (dict): 字段名 → 值的字典
            field_sources (dict|None): 字段名 → 来源类型 的字典
                                      或 字段名 → (source, confidence) 的元组字典

        返回：
            dict: {
                'score': float,           # 0.0-1.0 加权质量分
                'rating': str,            # 质量评级
                'field_scores': dict,     # 逐字段分数
                'flagged': bool,          # 是否需要人工复核
                'missing_fields': list,   # 缺失字段列表
                'low_confidence_fields': list,  # 低置信度字段列表
            }
        """
        if not record:
            return {
                'score': 0.0,
                'rating': 'critical',
                'field_scores': {},
                'flagged': True,
                'missing_fields': [],
                'low_confidence_fields': [],
            }

        if field_sources is None:
            field_sources = {}

        field_scores = {}
        missing_fields = []
        low_confidence_fields = []

        for field_name, value in record.items():
            # 跳过元数据字段
            if field_name.startswith('_'):
                continue

            source_info = field_sources.get(field_name, 'unknown')
            if isinstance(source_info, tuple):
                source, confidence = source_info
            else:
                source = source_info
                confidence = None

            score = self.score_field(field_name, value, source, confidence)
            field_scores[field_name] = score

            if value is None or value == '' or value == MISSING_VALUE_MARKER:
                missing_fields.append(field_name)
            if score < 0.5:
                low_confidence_fields.append(field_name)

        # 加权平均（使用字段权重）
        if field_scores:
            total_weighted = 0.0
            total_weight = 0.0
            for field_name, score in field_scores.items():
                weight = self.field_weights.get(field_name, DEFAULT_FIELD_WEIGHT)
                total_weighted += score * weight
                total_weight += weight
            avg_score = total_weighted / total_weight if total_weight > 0 else 0.0
        else:
            avg_score = 0.0

        # 质量评级
        rating = self._get_rating(avg_score)

        # 是否需人工复核（使用可配置阈值）
        # 有效字段：排除以 _ 开头的元数据字段（如 _source, _import_time 等），
        # 避免分母偏大致使 30% 阈值难以触发，导致问题记录漏标。
        effective_fields = [k for k in record if not k.startswith('_')]
        effective_count = len(effective_fields) if effective_fields else 1
        flagged = avg_score < self.flagged_threshold or len(missing_fields) > effective_count * 0.3

        return {
            'score': round(avg_score, 4),
            'rating': rating,
            'field_scores': field_scores,
            'flagged': flagged,
            'missing_fields': missing_fields,
            'low_confidence_fields': low_confidence_fields,
        }

    # ------------------------------------------------------------------
    # 批量评分
    # ------------------------------------------------------------------

    def score_batch(self, records, field_sources_map=None):
        """批量评分，返回每条记录的质量分和汇总统计。

        参数：
            records (list[dict]): 记录列表
            field_sources_map (dict|None): 记录索引 → 字段来源映射

        返回：
            dict: {
                'record_scores': list[dict],
                'summary': {
                    'total': int,
                    'mean_score': float,
                    'median_score': float,
                    'min_score': float,
                    'max_score': float,
                    'flagged_count': int,
                    'rating_distribution': dict,
                }
            }
        """
        if field_sources_map is None:
            field_sources_map = {}

        record_scores = []
        scores_list = []

        for i, record in enumerate(records):
            sources = field_sources_map.get(i, {})
            result = self.score_record(record, sources)
            result['record_index'] = i
            record_scores.append(result)
            scores_list.append(result['score'])

        # 汇总统计
        if scores_list:
            sorted_scores = sorted(scores_list)
            n = len(sorted_scores)
            if n % 2 == 0:
                median_score = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
            else:
                median_score = sorted_scores[n // 2]

            mean_score = sum(scores_list) / n
            min_score = min(scores_list)
            max_score = max(scores_list)
            flagged_count = sum(1 for r in record_scores if r['flagged'])
        else:
            mean_score = median_score = min_score = max_score = 0.0
            flagged_count = 0

        # 评级分布
        rating_distribution = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0, 'critical': 0}
        for r in record_scores:
            rating_distribution[r['rating']] = rating_distribution.get(r['rating'], 0) + 1

        return {
            'record_scores': record_scores,
            'summary': {
                'total': len(records),
                'mean_score': round(mean_score, 4),
                'median_score': round(median_score, 4),
                'min_score': round(min_score, 4),
                'max_score': round(max_score, 4),
                'flagged_count': flagged_count,
                'rating_distribution': rating_distribution,
            },
        }

    # ------------------------------------------------------------------
    # 低质量标记
    # ------------------------------------------------------------------

    def flag_low_quality(self, records, threshold=0.5):
        """标记低于阈值的记录为'需人工复核'。

        参数：
            records (list[dict]): 记录列表
            threshold (float): 质量分阈值，默认 0.5

        返回：
            list[int]: 需要复核的记录索引列表
        """
        batch_result = self.score_batch(records)
        flagged = [
            r['record_index']
            for r in batch_result['record_scores']
            if r['score'] < threshold
        ]
        return flagged

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _get_rating(self, score):
        """根据分数返回质量评级。

        参数：
            score (float): 质量分

        返回：
            str: 'excellent' / 'good' / 'fair' / 'poor' / 'critical'
        """
        if score >= self.quality_thresholds.get('excellent', 0.90):
            return 'excellent'
        if score >= self.quality_thresholds.get('good', 0.75):
            return 'good'
        if score >= self.quality_thresholds.get('fair', 0.50):
            return 'fair'
        if score >= self.quality_thresholds.get('poor', 0.30):
            return 'poor'
        return 'critical'

    def get_rating_label(self, rating):
        """获取评级的中文标签。

        参数：
            rating (str): 评级

        返回：
            str: 中文标签
        """
        labels = {
            'excellent': '优秀',
            'good': '良好',
            'fair': '一般',
            'poor': '较差',
            'critical': '极差',
        }
        return labels.get(rating, rating)

    def get_source_label(self, source):
        """获取来源的中文标签。

        参数：
            source (str): 来源类型

        返回：
            str: 中文标签
        """
        labels = {
            'exact': '精确匹配',
            'keyword': '关键词匹配',
            'fuzzy_levenshtein': '模糊匹配(编辑距离)',
            'fuzzy_difflib': '模糊匹配(相似度)',
            'regex': '正则抽取',
            'llm': 'LLM抽取',
            'user_feedback': '用户反馈',
            'imputed_mode': '众数填充',
            'imputed_median': '中位数填充',
            'imputed_conditional_median': '条件中位数填充',
            'imputed_scene_default': '场景默认值',
            'missing_marker': '缺失标记',
            'default': '默认值',
            'direct': '直接提供',
            'unknown': '未知来源',
        }
        return labels.get(source, source)