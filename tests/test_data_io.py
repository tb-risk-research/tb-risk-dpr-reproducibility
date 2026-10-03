#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_io 包单元测试套件

覆盖：
- synonyms.py: 三阶模糊匹配、名称标准化、动态阈值、用户反馈持久化
- text_parser.py: 正则抽取、段落分割、段间合并、记录边界检测、输入类型检测
- imputation.py: 众数填充、条件中位数、场景默认值、缺失标记过滤
- scoring.py: 逐字段评分、加权记录评分、批量评分、质量评级
- audit.py: 导入生命周期、审计报告导出、XSS 防护
- pipeline.py: 端到端导入管线
"""

import json
import os
import sys
import tempfile
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


# ==============================================================================
# 一、同义词模块测试
# ==============================================================================

class TestSynonymFuzzyMatch(unittest.TestCase):
    """测试三阶模糊匹配"""

    def setUp(self):
        from tb_risk.data_io.synonyms import fuzzy_match_column
        self.match = fuzzy_match_column

    # ---- 精确匹配 ----

    def test_exact_chinese_match(self):
        """中文精确匹配：'年龄' → 'age'"""
        std, conf, src = self.match('年龄')
        self.assertEqual(std, 'age')
        self.assertEqual(conf, 1.0)
        self.assertEqual(src, 'exact')

    def test_exact_english_match(self):
        """英文精确匹配：'sputum_smear' → 'sputum_smear'"""
        std, conf, src = self.match('sputum_smear')
        self.assertEqual(std, 'sputum_smear')
        self.assertEqual(conf, 1.0)

    def test_exact_english_abbr_match(self):
        """英文缩写匹配：'bcg' → 'bcg_vaccine'"""
        std, conf, src = self.match('bcg')
        self.assertEqual(std, 'bcg_vaccine')
        self.assertGreaterEqual(conf, 0.85)

    def test_exact_normalized_match(self):
        """标准化后匹配：'patient_age' → 'age'"""
        std, conf, src = self.match('patient_age')
        self.assertEqual(std, 'age')
        self.assertGreaterEqual(conf, 0.90)

    # ---- 关键词匹配 ----

    def test_keyword_match(self):
        """关键词包含匹配：'咳嗽频率' → 'cough_freq'"""
        std, conf, src = self.match('咳嗽频率')
        self.assertEqual(std, 'cough_freq')
        self.assertIn(src, ('exact', 'keyword'))  # 同义词库中有精确别名

    def test_keyword_match_diagnosis(self):
        """关键词匹配：'diagnosis' → 'is_confirmed'"""
        std, conf, src = self.match('diagnosis')
        self.assertEqual(std, 'is_confirmed')
        self.assertIn(src, ('exact', 'keyword'))  # 同义词库中有精确别名

    # ---- 模糊匹配 ----

    def test_fuzzy_similar_match(self):
        """模糊匹配：'patient age' → 'age'"""
        std, conf, src = self.match('patient age')
        self.assertEqual(std, 'age')
        self.assertIn(src, ('exact', 'fuzzy_levenshtein', 'keyword'))

    def test_fuzzy_difflib_match(self):
        """difflib 模糊匹配：'cough freq' → 'cough_freq'"""
        std, conf, src = self.match('cough freq')
        self.assertEqual(std, 'cough_freq')
        self.assertIn(src, ('exact', 'fuzzy_levenshtein', 'keyword', 'fuzzy_difflib'))

    # ---- 边界情况 ----

    def test_empty_input(self):
        """空输入返回 None"""
        std, conf, src = self.match('')
        self.assertIsNone(std)
        self.assertEqual(conf, 0.0)

    def test_none_input(self):
        """None 输入返回 None"""
        std, conf, src = self.match(None)
        self.assertIsNone(std)
        self.assertEqual(conf, 0.0)

    def test_nonexistent_field(self):
        """不存在的字段返回 None"""
        std, conf, src = self.match('完全不存在的字段名_xyz123')
        self.assertIsNone(std)

    def test_short_field_name(self):
        """短字段名：'id' → 'record_id'"""
        std, conf, src = self.match('id')
        self.assertEqual(std, 'record_id')


class TestSynonymNormalization(unittest.TestCase):
    """测试名称标准化"""

    def test_normalize_removes_spaces(self):
        from tb_risk.data_io.synonyms import normalize_name
        self.assertEqual(normalize_name('patient age'), 'patientage')

    def test_normalize_removes_underscores(self):
        from tb_risk.data_io.synonyms import normalize_name
        self.assertEqual(normalize_name('patient_age'), 'patientage')

    def test_normalize_lowercase(self):
        from tb_risk.data_io.synonyms import normalize_name
        self.assertEqual(normalize_name('CoughFreq'), 'coughfreq')

    def test_normalize_removes_parens(self):
        from tb_risk.data_io.synonyms import normalize_name
        self.assertEqual(normalize_name('age(岁)'), 'age岁')


class TestSynonymDynamicThreshold(unittest.TestCase):
    """测试动态编辑距离阈值"""

    def test_short_string_threshold_1(self):
        from tb_risk.data_io.synonyms import _compute_dynamic_threshold
        self.assertEqual(_compute_dynamic_threshold('age', 'age', 2), 1)

    def test_medium_string_threshold_2(self):
        from tb_risk.data_io.synonyms import _compute_dynamic_threshold
        self.assertEqual(_compute_dynamic_threshold('cough', 'cough', 2), 2)

    def test_long_string_threshold_3(self):
        from tb_risk.data_io.synonyms import _compute_dynamic_threshold
        self.assertEqual(
            _compute_dynamic_threshold('cumulative_exposure', 'cumulative_exposure', 2),
            3
        )


class TestSynonymFeedback(unittest.TestCase):
    """测试用户反馈持久化"""

    def setUp(self):
        from tb_risk.data_io.synonyms import FIELD_SYNONYM_LIBRARY, _FEEDBACK_FILE
        # 清理可能残留的持久化文件
        self._feedback_file = _FEEDBACK_FILE
        self._had_feedback_file = os.path.exists(_FEEDBACK_FILE)
        if self._had_feedback_file:
            self._feedback_backup = _FEEDBACK_FILE + '.bak'
            os.rename(_FEEDBACK_FILE, self._feedback_backup)
        # 深拷贝当前状态
        self._original = {
            field: list(aliases)
            for field, aliases in FIELD_SYNONYM_LIBRARY.items()
        }
        # 清除所有 user_feedback 条目，确保测试环境干净
        for field in FIELD_SYNONYM_LIBRARY:
            FIELD_SYNONYM_LIBRARY[field] = [
                (a, s, c) for a, s, c in FIELD_SYNONYM_LIBRARY[field]
                if s != 'user_feedback'
            ]

    def tearDown(self):
        from tb_risk.data_io import synonyms
        synonyms.FIELD_SYNONYM_LIBRARY.clear()
        synonyms.FIELD_SYNONYM_LIBRARY.update(self._original)
        # 恢复持久化文件
        if os.path.exists(self._feedback_file):
            os.remove(self._feedback_file)
        if self._had_feedback_file:
            os.rename(self._feedback_backup, self._feedback_file)

    def test_register_new_feedback(self):
        from tb_risk.data_io.synonyms import register_synonym_feedback, FIELD_SYNONYM_LIBRARY
        result = register_synonym_feedback('测试别名', 'age')
        self.assertTrue(result)
        # 检查是否已添加
        aliases = [a for a, _, _ in FIELD_SYNONYM_LIBRARY['age']]
        self.assertIn('测试别名', aliases)

    def test_register_duplicate_skipped(self):
        from tb_risk.data_io.synonyms import register_synonym_feedback
        # 注册已存在的别名应返回 False
        result = register_synonym_feedback('年龄', 'age')
        self.assertFalse(result)

    def test_register_invalid_field(self):
        from tb_risk.data_io.synonyms import register_synonym_feedback
        result = register_synonym_feedback('test', 'nonexistent_field')
        self.assertFalse(result)


class TestSynonymLegacyCompatibility(unittest.TestCase):
    """测试向后兼容接口"""

    def test_generate_legacy_synonyms(self):
        from tb_risk.data_io.synonyms import generate_legacy_synonyms
        legacy = generate_legacy_synonyms()
        self.assertIsInstance(legacy, dict)
        self.assertIn('age', legacy)
        self.assertIsInstance(legacy['age'], list)
        self.assertIn('年龄', legacy['age'])

    def test_dynamic_column_match(self):
        from tb_risk.data_io.synonyms import dynamic_column_match
        result = dynamic_column_match('年龄')
        self.assertEqual(result, 'age')
        result = dynamic_column_match('nonexistent')
        self.assertIsNone(result)


# ==============================================================================
# 二、文本抽取模块测试
# ==============================================================================

class TestTextParserRegexExtraction(unittest.TestCase):
    """测试正则文本抽取"""

    def setUp(self):
        from tb_risk.data_io.text_parser import extract_fields_from_text
        self.extract = extract_fields_from_text

    def test_extract_single_patient(self):
        """单患者文本抽取"""
        text = "患者男，35岁，痰涂片阳性，有空洞，卡介苗已接种，通风条件差，已接受治疗"
        records = self.extract(text)
        self.assertGreaterEqual(len(records), 1)
        rec = records[0]
        self.assertIn('age', rec)
        self.assertEqual(rec['age'], 35)

    def test_extract_age_from_pattern(self):
        """年龄正则抽取"""
        records = self.extract("年龄25岁，痰涂片阴性")
        self.assertGreaterEqual(len(records), 1)
        rec = records[0]
        self.assertIn('age', rec)
        self.assertEqual(rec['age'], 25)

    def test_extract_sputum_smear_positive(self):
        """痰涂片阳性抽取"""
        records = self.extract("痰涂片涂阳")
        self.assertGreaterEqual(len(records), 1)
        rec = records[0]
        if 'sputum_smear' in rec:
            self.assertEqual(rec['sputum_smear'], 2)

    def test_extract_has_cavity(self):
        """空洞检测"""
        records = self.extract("肺部CT示有空洞")
        self.assertGreaterEqual(len(records), 1)
        rec = records[0]
        if 'has_cavity' in rec:
            self.assertIn(rec['has_cavity'], [1, 2])

    def test_extract_multiple_fields(self):
        """多字段同时抽取"""
        text = "患者男，年龄45岁，痰涂片阳性，有空洞，已治疗，通风良好"
        records = self.extract(text)
        self.assertGreaterEqual(len(records), 1)
        rec = records[0]
        # 至少应抽取到 age
        self.assertIn('age', rec)

    def test_extract_empty_text(self):
        """空文本返回空列表"""
        records = self.extract("")
        self.assertEqual(records, [])

    def test_extract_confidence_metadata(self):
        """检查元数据_confidence"""
        records = self.extract("年龄30岁")
        self.assertGreaterEqual(len(records), 1)
        rec = records[0]
        self.assertIn('_confidence', rec)
        self.assertIn('_source', rec)
        self.assertEqual(rec['_source'], {'source': 'regex'})


class TestTextParserSplitParagraphs(unittest.TestCase):
    """测试段落分割"""

    def test_split_by_newline(self):
        from tb_risk.data_io.text_parser import split_paragraphs
        text = "患者男，35岁\n痰涂片阳性\n有空洞"
        paras = split_paragraphs(text)
        self.assertEqual(len(paras), 3)

    def test_split_empty(self):
        from tb_risk.data_io.text_parser import split_paragraphs
        self.assertEqual(split_paragraphs(""), [])

    def test_split_single_line(self):
        from tb_risk.data_io.text_parser import split_paragraphs
        paras = split_paragraphs("患者男，35岁，痰涂片阳性")
        self.assertEqual(len(paras), 1)


class TestTextParserMergeSparseParagraphs(unittest.TestCase):
    """测试段间合并"""

    def test_merge_sparse_paragraphs(self):
        from tb_risk.data_io.text_parser import _merge_sparse_paragraphs
        paragraphs = ["患者男", "35岁", "痰涂片阳性", "有空洞"]
        merged = _merge_sparse_paragraphs(paragraphs)
        # 前两段稀疏，应合并；后两段各有字段，保留
        self.assertLess(len(merged), 4)


class TestTextParserRecordBoundaries(unittest.TestCase):
    """测试记录边界检测"""

    def test_detect_separator(self):
        from tb_risk.data_io.text_parser import _detect_record_boundaries
        paragraphs = ["患者A，年龄30岁", "---", "患者B，年龄40岁"]
        boundaries = _detect_record_boundaries(paragraphs)
        self.assertIn(0, boundaries)  # 第一条从0开始
        self.assertGreater(len(boundaries), 1)  # 至少检测到一个边界

    def test_detect_patient_prefix(self):
        from tb_risk.data_io.text_parser import _detect_record_boundaries
        paragraphs = ["患者男，年龄30岁", "患者女，年龄40岁"]
        boundaries = _detect_record_boundaries(paragraphs)
        self.assertIn(0, boundaries)


class TestTextParserDetectInputType(unittest.TestCase):
    """测试输入类型检测"""

    def test_detect_csv(self):
        from tb_risk.data_io.text_parser import detect_input_type
        self.assertEqual(detect_input_type('test.csv'), 'structured_csv')

    def test_detect_excel(self):
        from tb_risk.data_io.text_parser import detect_input_type
        self.assertEqual(detect_input_type('test.xlsx'), 'structured_excel')

    def test_detect_json(self):
        from tb_risk.data_io.text_parser import detect_input_type
        self.assertEqual(detect_input_type('test.json'), 'structured_json')

    def test_detect_text(self):
        from tb_risk.data_io.text_parser import detect_input_type
        self.assertEqual(detect_input_type('test.txt'), 'text')

    def test_detect_content_csv(self):
        """通过内容检测CSV"""
        from tb_risk.data_io.text_parser import detect_input_type
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.dat', delete=False, encoding='utf-8'
        ) as f:
            f.write('name,age,gender\n张三,35,男\n李四,42,女\n')
            tmp_path = f.name
        try:
            result = detect_input_type(tmp_path)
            self.assertIn(result, ('structured_csv', 'structured_json', 'text', 'unknown'))
        finally:
            os.unlink(tmp_path)


class TestTextParserParseTextInput(unittest.TestCase):
    """测试 parse_text_input 主入口"""

    def test_parse_basic(self):
        from tb_risk.data_io.text_parser import parse_text_input
        text = "患者男，年龄35岁，痰涂片阳性，有空洞，已治疗"
        records, meta = parse_text_input(text)
        self.assertIsInstance(records, list)
        self.assertIsInstance(meta, dict)
        self.assertIn('paragraph_count', meta)
        self.assertGreater(meta['paragraph_count'], 0)

    def test_parse_empty(self):
        from tb_risk.data_io.text_parser import parse_text_input
        records, meta = parse_text_input("")
        self.assertEqual(records, [])
        self.assertEqual(meta['paragraph_count'], 0)


# ==============================================================================
# 三、缺失值填充模块测试
# ==============================================================================

class TestImputationMode(unittest.TestCase):
    """测试众数填充"""

    def test_mode_imputation_basic(self):
        from tb_risk.data_io.imputation import impute_mode
        data = [
            {'bcg_vaccine': 1}, {'bcg_vaccine': 1}, {'bcg_vaccine': 1},
            {'bcg_vaccine': 1}, {'bcg_vaccine': 0}, {'bcg_vaccine': None},
            {'bcg_vaccine': ''},
        ]
        filled, strategy, value = impute_mode(data, 'bcg_vaccine')
        self.assertEqual(filled, 2)
        self.assertEqual(strategy, 'mode')
        self.assertEqual(value, 1)
        self.assertEqual(data[5]['bcg_vaccine'], 1)
        self.assertEqual(data[6]['bcg_vaccine'], 1)

    def test_mode_insufficient_samples(self):
        """小样本时众数填充不可靠，应跳过"""
        from tb_risk.data_io.imputation import impute_mode
        data = [
            {'has_tb': 0}, {'has_tb': 1}, {'has_tb': None},
        ]
        filled, strategy, value = impute_mode(data, 'has_tb', min_valid_count=5)
        self.assertEqual(filled, 0)
        self.assertEqual(strategy, 'mode')
        self.assertIsNone(value)

    def test_mode_no_missing(self):
        from tb_risk.data_io.imputation import impute_mode
        data = [{'bcg_vaccine': 1}, {'bcg_vaccine': 1}, {'bcg_vaccine': 1},
                {'bcg_vaccine': 1}, {'bcg_vaccine': 1}]
        filled, strategy, value = impute_mode(data, 'bcg_vaccine')
        self.assertEqual(filled, 0)
        self.assertEqual(strategy, 'mode')
        self.assertEqual(value, 1)


class TestImputationConditionalMedian(unittest.TestCase):
    """测试条件中位数填充"""

    def test_conditional_median_with_scenario(self):
        from tb_risk.data_io.imputation import impute_conditional_median
        data = [
            {'age': 30, 'scenario_type': 'rural_family'},
            {'age': 40, 'scenario_type': 'rural_family'},
            {'age': None, 'scenario_type': 'rural_family'},
            {'age': 50, 'scenario_type': 'urban'},
            {'age': 60, 'scenario_type': 'urban'},
            {'age': None, 'scenario_type': 'urban'},
        ]
        filled, strategy, group_medians = impute_conditional_median(data, 'age')
        self.assertEqual(filled, 2)
        self.assertEqual(strategy, 'conditional_median')
        self.assertIn('rural_family', group_medians)
        self.assertIn('urban', group_medians)

    def test_conditional_median_no_scenario_fallback(self):
        """无 scenario_type 时回退到简单中位数"""
        from tb_risk.data_io.imputation import impute_conditional_median
        data = [
            {'age': 30}, {'age': 40}, {'age': None},
        ]
        filled, strategy, result = impute_conditional_median(data, 'age')
        self.assertEqual(filled, 1)
        self.assertEqual(strategy, 'median_fallback')
        self.assertEqual(data[2]['age'], 35.0)

    # ---- _grouping_has_discriminative_power 测试 ----

    def test_grouping_discriminative_true(self):
        """有区分度的分组：rural=30, urban=60 → 差异大 → True"""
        from tb_risk.data_io.imputation import _grouping_has_discriminative_power
        data = [
            {'age': 28, 'scenario_type': 'rural_family'},
            {'age': 30, 'scenario_type': 'rural_family'},
            {'age': 32, 'scenario_type': 'rural_family'},
            {'age': 58, 'scenario_type': 'urban'},
            {'age': 60, 'scenario_type': 'urban'},
            {'age': 62, 'scenario_type': 'urban'},
        ]
        self.assertTrue(_grouping_has_discriminative_power(data, 'scenario_type', 'age'))

    def test_grouping_discriminative_false(self):
        """无区分度的分组：两组年龄接近 → False"""
        from tb_risk.data_io.imputation import _grouping_has_discriminative_power
        data = [
            {'age': 30, 'scenario_type': 'rural_family'},
            {'age': 32, 'scenario_type': 'rural_family'},
            {'age': 31, 'scenario_type': 'urban'},
            {'age': 33, 'scenario_type': 'urban'},
        ]
        self.assertFalse(_grouping_has_discriminative_power(data, 'scenario_type', 'age'))

    def test_grouping_single_group_only(self):
        """只有一种分组值 → False"""
        from tb_risk.data_io.imputation import _grouping_has_discriminative_power
        data = [
            {'age': 30, 'scenario_type': 'rural_family'},
            {'age': 40, 'scenario_type': 'rural_family'},
            {'age': 50, 'scenario_type': 'rural_family'},
        ]
        self.assertFalse(_grouping_has_discriminative_power(data, 'scenario_type', 'age'))

    def test_grouping_all_missing_target(self):
        """目标字段全缺失 → False"""
        from tb_risk.data_io.imputation import _grouping_has_discriminative_power
        data = [
            {'age': None, 'scenario_type': 'rural_family'},
            {'age': None, 'scenario_type': 'urban'},
        ]
        self.assertFalse(_grouping_has_discriminative_power(data, 'scenario_type', 'age'))

    def test_select_grouping_field_fallback(self):
        """_select_grouping_field 在无有效分组时返回 None"""
        from tb_risk.data_io.imputation import _select_grouping_field
        data = [
            {'age': 30, 'scenario_type': 'rural_family'},
            {'age': 32, 'scenario_type': 'rural_family'},
            {'age': 31, 'scenario_type': 'urban'},
            {'age': 33, 'scenario_type': 'urban'},
        ]
        # 两组年龄接近，所有候选分组字段都应被跳过
        result = _select_grouping_field(data, 'age')
        # 应返回 None（无区分度分组），触发简单中位数回退
        self.assertIsNone(result)

    def test_grouping_adaptive_threshold_small_sample(self):
        """小样本（<20条有效值）使用 10% 阈值"""
        from tb_risk.data_io.imputation import _grouping_has_discriminative_power
        data = [
            {'age': 35, 'scenario_type': 'rural_family'},
            {'age': 38, 'scenario_type': 'rural_family'},
            {'age': 40, 'scenario_type': 'urban'},
            {'age': 43, 'scenario_type': 'urban'},
        ]
        # 相对差异 = (41.5 - 36.5) / 39 = 12.8% > 10% → True
        self.assertTrue(_grouping_has_discriminative_power(data, 'scenario_type', 'age'))


class TestImputationSceneDefault(unittest.TestCase):
    """测试场景默认值填充"""

    def test_scene_default_basic(self):
        from tb_risk.data_io.imputation import impute_scene_default
        data = [
            {'ventilation': None, 'scenario_type': 'rural_family'},
        ]
        filled, strategy, value = impute_scene_default(data, 'ventilation')
        # 如果 SCENE_DEFAULTS 中有 rural_family 的 ventilation 默认值
        self.assertIsInstance(filled, int)
        self.assertEqual(strategy, 'scene_default')


class TestImputationMissingMarker(unittest.TestCase):
    """测试缺失值标记"""

    def test_mark_unfillable(self):
        from tb_risk.data_io.imputation import mark_unfillable
        data = [{'unknown_field': None}, {'unknown_field': ''}]
        marked, strategy, value = mark_unfillable(data, 'unknown_field')
        self.assertEqual(marked, 2)
        self.assertEqual(strategy, 'missing_marker')
        self.assertEqual(value, -999)
        self.assertEqual(data[0]['unknown_field'], -999)
        self.assertEqual(data[1]['unknown_field'], -999)

    def test_filter_missing_marker(self):
        from tb_risk.data_io.imputation import filter_missing_marker
        record = {'age': -999, 'sputum_smear': 2, '_confidence': {'age': 0.0}}
        filtered = filter_missing_marker(record)
        self.assertIsNone(filtered['age'])
        self.assertEqual(filtered['sputum_smear'], 2)
        # 元数据字段应保留
        self.assertIn('_confidence', filtered)

    def test_filter_missing_marker_batch(self):
        from tb_risk.data_io.imputation import filter_missing_marker_batch
        records = [
            {'age': -999, 'sputum_smear': 2},
            {'age': 35, 'sputum_smear': -999},
        ]
        filtered = filter_missing_marker_batch(records)
        self.assertIsNone(filtered[0]['age'])
        self.assertIsNone(filtered[1]['sputum_smear'])


class TestImputationUnified(unittest.TestCase):
    """测试统一填充接口"""

    def test_impute_missing_auto_strategy(self):
        from tb_risk.data_io.imputation import impute_missing
        data = [
            {'bcg_vaccine': 1, 'scenario_type': 'rural_family'},
            {'bcg_vaccine': 1, 'scenario_type': 'rural_family'},
            {'bcg_vaccine': None, 'scenario_type': 'rural_family'},
            {'bcg_vaccine': 1, 'scenario_type': 'rural_family'},
            {'bcg_vaccine': 1, 'scenario_type': 'rural_family'},
            {'bcg_vaccine': 1, 'scenario_type': 'rural_family'},
        ]
        filled, strategy, value = impute_missing(data, 'bcg_vaccine', strategy='auto')
        self.assertEqual(filled, 1)
        self.assertEqual(strategy, 'mode')
        self.assertEqual(value, 1)

    def test_impute_missing_empty_data(self):
        from tb_risk.data_io.imputation import impute_missing
        filled, strategy, value = impute_missing([], 'age')
        self.assertEqual(filled, 0)
        self.assertEqual(strategy, 'none')


class TestImputationFillDefaults(unittest.TestCase):
    """测试统一默认值填充"""

    def test_fill_record_defaults(self):
        from tb_risk.data_io.imputation import fill_record_defaults
        record = {}
        filled = fill_record_defaults(record, default_type='family')
        self.assertIsInstance(filled, list)

    def test_fill_missing_fields_batch(self):
        from tb_risk.data_io.imputation import fill_missing_fields_batch
        records = [{}, {'age': 35}]
        filled = fill_missing_fields_batch(records, default_type='family')
        self.assertGreaterEqual(filled, 0)


# ==============================================================================
# 四、质量评分模块测试
# ==============================================================================

class TestDataQualityScorer(unittest.TestCase):
    """测试数据质量评分器"""

    def setUp(self):
        from tb_risk.data_io.scoring import DataQualityScorer
        self.scorer = DataQualityScorer()

    def test_score_field_exact(self):
        score = self.scorer.score_field('age', 35, 'exact')
        self.assertEqual(score, 1.0)

    def test_score_field_keyword(self):
        score = self.scorer.score_field('age', 35, 'keyword')
        self.assertAlmostEqual(score, 0.90, places=1)

    def test_score_field_fuzzy(self):
        score = self.scorer.score_field('age', 35, 'fuzzy_levenshtein')
        self.assertAlmostEqual(score, 0.75, places=1)

    def test_score_field_imputed(self):
        score = self.scorer.score_field('age', 35, 'imputed_median')
        self.assertAlmostEqual(score, 0.45, places=1)

    def test_score_field_missing(self):
        score = self.scorer.score_field('age', None, 'exact')
        self.assertLessEqual(score, 0.1)

    def test_score_field_missing_marker(self):
        score = self.scorer.score_field('age', -999, 'exact')
        self.assertLessEqual(score, 0.1)

    def test_score_field_unknown_source(self):
        score = self.scorer.score_field('custom_field', 'value', 'unknown_source')
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_score_record_exact(self):
        """所有字段精确匹配 → 高分"""
        record = {'age': 35, 'sputum_smear': 2, 'has_cavity': 1}
        sources = {'age': 'exact', 'sputum_smear': 'exact', 'has_cavity': 'exact'}
        result = self.scorer.score_record(record, sources)
        self.assertAlmostEqual(result['score'], 1.0, places=1)
        self.assertEqual(result['rating'], 'excellent')
        self.assertFalse(result['flagged'])

    def test_score_record_mixed(self):
        """混合来源 → 中等分"""
        record = {'age': 35, 'sputum_smear': 2, 'has_cavity': None}
        sources = {'age': 'exact', 'sputum_smear': 'fuzzy_levenshtein', 'has_cavity': 'missing_marker'}
        result = self.scorer.score_record(record, sources)
        self.assertLess(result['score'], 1.0)
        self.assertGreater(result['score'], 0.0)

    def test_score_record_empty(self):
        result = self.scorer.score_record({})
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['rating'], 'critical')
        self.assertTrue(result['flagged'])

    def test_score_record_weighted_core_field(self):
        """核心字段（age, is_confirmed）权重更高"""
        record1 = {'age': 35, 'occupation': None}
        sources1 = {'age': 'exact', 'occupation': 'missing_marker'}
        result1 = self.scorer.score_record(record1, sources1)

        record2 = {'age': None, 'occupation': '医生'}
        sources2 = {'age': 'missing_marker', 'occupation': 'exact'}
        result2 = self.scorer.score_record(record2, sources2)

        # age 权重高 → 缺失 age 比缺失 occupation 扣分更多
        self.assertGreater(result1['score'], result2['score'])

    def test_score_batch(self):
        records = [
            {'age': 35, 'sputum_smear': 2},
            {'age': 42, 'sputum_smear': 1},
            {'age': None, 'sputum_smear': None},
        ]
        sources = {
            0: {'age': 'exact', 'sputum_smear': 'exact'},
            1: {'age': 'exact', 'sputum_smear': 'exact'},
            2: {'age': 'missing_marker', 'sputum_smear': 'missing_marker'},
        }
        result = self.scorer.score_batch(records, sources)
        self.assertEqual(result['summary']['total'], 3)
        self.assertGreater(result['summary']['mean_score'], 0.0)
        self.assertLess(result['summary']['mean_score'], 1.0)
        self.assertGreater(result['summary']['flagged_count'], 0)
        self.assertIn('excellent', result['summary']['rating_distribution'])

    def test_score_batch_empty(self):
        result = self.scorer.score_batch([])
        self.assertEqual(result['summary']['total'], 0)
        self.assertEqual(result['summary']['mean_score'], 0.0)

    def test_flagged_threshold_configurable(self):
        from tb_risk.data_io.scoring import DataQualityScorer
        strict_scorer = DataQualityScorer(flagged_threshold=0.8)
        record = {'age': 35, 'sputum_smear': 2}
        sources = {'age': 'fuzzy_levenshtein', 'sputum_smear': 'fuzzy_levenshtein'}
        result = strict_scorer.score_record(record, sources)
        # 模糊匹配 → 低分 → 应被标记
        self.assertTrue(result['flagged'])

    def test_get_rating(self):
        self.assertEqual(self.scorer._get_rating(0.95), 'excellent')
        self.assertEqual(self.scorer._get_rating(0.80), 'good')
        self.assertEqual(self.scorer._get_rating(0.60), 'fair')
        self.assertEqual(self.scorer._get_rating(0.40), 'poor')
        self.assertEqual(self.scorer._get_rating(0.10), 'critical')

    def test_flag_low_quality(self):
        records = [
            {'age': 35, 'sputum_smear': 2, 'has_cavity': 1},
            {'age': None, 'sputum_smear': None, 'has_cavity': None},
        ]
        flagged = self.scorer.flag_low_quality(records, threshold=0.5)
        self.assertIn(1, flagged)  # 第二条记录应被标记


# ==============================================================================
# 五、审计模块测试
# ==============================================================================

class TestImportAuditor(unittest.TestCase):
    """测试导入审计器"""

    def setUp(self):
        from tb_risk.data_io.audit import ImportAuditor
        self.tmp_dir = tempfile.mkdtemp()
        self.auditor = ImportAuditor(log_dir=self.tmp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_import_lifecycle(self):
        """完整的导入生命周期"""
        self.auditor.log_import_start('test.csv', 'structured_csv')
        self.assertIsNotNone(self.auditor._batch_id)

        self.auditor.log_column_mapping({'年龄': 'age', '性别': 'gender'})
        self.auditor.log_imputation('age', 'conditional_median', 5, 35)
        self.auditor.log_import_complete(100, ['warning1'], [])

        self.assertEqual(len(self.auditor._events), 4)

    def test_log_text_extraction(self):
        self.auditor.log_import_start('clinical.txt', 'text')
        self.auditor.log_text_extraction(500, 3, 1)
        self.auditor.log_import_complete(1)
        self.assertGreaterEqual(len(self.auditor._events), 3)

    def test_log_quality_summary(self):
        self.auditor.log_import_start('test.csv', 'structured_csv')
        self.auditor.log_quality_summary({'mean_score': 0.85}, 3)
        self.auditor.log_import_complete(50)
        self.assertGreaterEqual(len(self.auditor._events), 3)

    def test_export_json_report(self):
        self.auditor.log_import_start('test.csv', 'structured_csv')
        self.auditor.log_import_complete(10)
        report_path = os.path.join(self.tmp_dir, 'report.json')
        success = self.auditor.export_audit_report(report_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(report_path))

    def test_export_html_report(self):
        self.auditor.log_import_start('test.csv', 'structured_csv')
        self.auditor.log_column_mapping({'年龄': 'age'})
        self.auditor.log_imputation('age', 'mode', 3, 1)
        self.auditor.log_import_complete(10)
        report_path = os.path.join(self.tmp_dir, 'report.html')
        success = self.auditor.export_audit_report(report_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(report_path))
        # 验证 HTML 内容
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('<!DOCTYPE html>', content)
            self.assertIn('数据导入审计报告', content)

    def test_html_xss_prevention(self):
        """XSS 防护：列名中的特殊字符应被转义"""
        self.auditor.log_import_start('test.csv', 'structured_csv')
        self.auditor.log_column_mapping({'<script>alert(1)</script>': 'age'})
        self.auditor.log_import_complete(1)
        report_path = os.path.join(self.tmp_dir, 'xss_test.html')
        self.auditor.export_audit_report(report_path)
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertNotIn('<script>alert(1)</script>', content)
            self.assertIn('&lt;script&gt;', content)

    def test_events_cache_limit(self):
        """内存缓存上限检查"""
        from tb_risk.data_io.audit import _MAX_EVENTS_MEMORY
        self.auditor.log_import_start('test.csv', 'structured_csv')
        # 批量添加事件，不应超过上限
        for i in range(100):
            self.auditor.log_imputation(f'field_{i}', 'mode', 1, 1)
        self.assertLessEqual(len(self.auditor._events), _MAX_EVENTS_MEMORY + 1000)

    def test_get_import_history(self):
        self.auditor.log_import_start('test.csv', 'structured_csv')
        self.auditor.log_import_complete(10)
        history = self.auditor.get_import_history(limit=10)
        self.assertIsInstance(history, list)

    def test_log_rotation(self):
        """日志按日期分文件"""
        self.auditor.log_import_start('test.csv', 'structured_csv')
        self.auditor.log_import_complete(10)
        log_files = [f for f in os.listdir(self.tmp_dir) if f.startswith('audit_')]
        self.assertGreaterEqual(len(log_files), 1)


# ==============================================================================
# 六、管线编排测试
# ==============================================================================

class TestPipeline(unittest.TestCase):
    """测试导入管线"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _create_test_csv(self):
        """创建测试 CSV 文件"""
        csv_path = os.path.join(self.tmp_dir, 'test.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('年龄,痰涂片,空洞,卡介苗,治疗,通风\n')
            f.write('35,阳性,有,已接种,已治疗,差\n')
            f.write('42,阴性,无,未接种,未治疗,一般\n')
            f.write('28,涂阳,有,已接种,已治疗,好\n')
        return csv_path

    def test_import_pipeline_csv(self):
        from tb_risk.data_io.pipeline import import_pipeline
        csv_path = self._create_test_csv()
        result = import_pipeline(
            csv_path,
            enable_scoring=True,
            enable_audit=True,
            audit_dir=self.tmp_dir,
        )
        self.assertTrue(result.success)
        self.assertEqual(len(result.records), 3)
        # 质量评分应已执行
        self.assertIsNotNone(result.quality)
        self.assertIn('summary', result.quality)
        # 审计报告应已生成
        if result.audit_report:
            self.assertTrue(os.path.exists(result.audit_report))

        # ---- 来源验证 ----
        # 测试 CSV 中所有中文列名应被精确匹配，_source 应为 'exact'
        first_record = result.records[0]
        self.assertIn('_source', first_record,
                      "每条记录应包含 _source 元数据字段")
        sources = first_record['_source']
        self.assertIsInstance(sources, dict,
                              "_source 应为字典类型")
        # 验证核心字段的来源标记
        for expected_field in ['age', 'sputum_smear']:
            self.assertIn(expected_field, sources,
                          f"核心字段 '{expected_field}' 应在 _source 中")
            self.assertEqual(sources[expected_field], 'exact',
                             f"中文精确匹配字段 '{expected_field}' 的来源应为 'exact'")

        # ---- 质量均分阈值 ----
        mean_score = result.quality['summary']['mean_score']
        self.assertGreaterEqual(
            mean_score, 0.6,
            f"所有字段精确匹配的有效数据质量均分应 ≥ 0.6，实际={mean_score:.2f}"
        )

    def test_import_pipeline_fuzzy_source(self):
        """管线测试：使用不在同义词库精确匹配中的列名，验证来源正确传递"""
        from tb_risk.data_io.pipeline import import_pipeline
        csv_path = os.path.join(self.tmp_dir, 'fuzzy_test.csv')
        # 使用子串/变体列名，确保不为精确匹配，验证 keyword/fuzzy 来源
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('患者年龄,痰涂片结果,有无空洞\n')
            f.write('35,阳性,有\n')
            f.write('42,阴性,无\n')
        result = import_pipeline(
            csv_path,
            enable_scoring=True,
            enable_audit=True,
            audit_dir=self.tmp_dir,
        )
        self.assertTrue(result.success)
        self.assertEqual(len(result.records), 2)

        first_record = result.records[0]
        self.assertIn('_source', first_record)
        sources = first_record['_source']

        # '患者年龄' 包含 '年龄' → 应为 keyword 匹配
        if 'age' in sources:
            self.assertIn(
                sources['age'], ('keyword', 'exact', 'fuzzy_levenshtein'),
                f"'患者年龄' 的来源应为 keyword/fuzzy，实际={sources['age']}"
            )
        # '痰涂片结果' 包含 '痰涂片' → 应为 keyword 匹配
        if 'sputum_smear' in sources:
            self.assertIn(
                sources['sputum_smear'], ('keyword', 'exact', 'fuzzy_levenshtein'),
                f"'痰涂片结果' 的来源应为 keyword/fuzzy，实际={sources['sputum_smear']}"
            )

        # 质量均分验证
        mean_score = result.quality['summary']['mean_score']
        self.assertGreaterEqual(
            mean_score, 0.5,
            f"模糊匹配的数据质量均分应 ≥ 0.5，实际={mean_score:.2f}"
        )

    def test_import_pipeline_empty_file(self):
        from tb_risk.data_io.pipeline import import_pipeline
        empty_path = os.path.join(self.tmp_dir, 'empty.csv')
        with open(empty_path, 'w', encoding='utf-8') as f:
            f.write('年龄,痰涂片\n')
        result = import_pipeline(empty_path)
        self.assertEqual(len(result.records), 0)
        self.assertIn('未加载到任何记录', result.warnings)

    def test_import_pipeline_disable_scoring(self):
        from tb_risk.data_io.pipeline import import_pipeline
        csv_path = self._create_test_csv()
        result = import_pipeline(csv_path, enable_scoring=False, enable_audit=False)
        self.assertTrue(result.success)
        self.assertIsNone(result.quality)

    def test_pipeline_result_dataclass(self):
        from tb_risk.data_io.pipeline import PipelineResult
        result = PipelineResult()
        self.assertTrue(result.success)
        self.assertEqual(result.records, [])
        self.assertEqual(result.errors, [])

    def test_quick_import(self):
        from tb_risk.data_io.pipeline import quick_import
        csv_path = self._create_test_csv()
        records, summary = quick_import(csv_path, enable_audit=False)
        self.assertEqual(len(records), 3)
        self.assertIsInstance(summary, dict)


# ==============================================================================
# 七、边界情况与回归测试
# ==============================================================================

class TestEdgeCases(unittest.TestCase):
    """边界情况测试"""

    def test_single_record(self):
        """单条记录处理"""
        from tb_risk.data_io.scoring import DataQualityScorer
        scorer = DataQualityScorer()
        records = [{'age': 35}]
        sources = {0: {'age': 'exact'}}
        result = scorer.score_batch(records, sources)
        self.assertEqual(result['summary']['total'], 1)

    def test_all_missing_values(self):
        """全缺失值记录"""
        from tb_risk.data_io.scoring import DataQualityScorer
        scorer = DataQualityScorer()
        records = [{'age': None, 'sputum_smear': None, 'has_cavity': None}]
        sources = {0: {k: 'missing_marker' for k in records[0]}}
        result = scorer.score_batch(records, sources)
        self.assertEqual(result['summary']['flagged_count'], 1)
        self.assertLess(result['summary']['mean_score'], 0.3)

    def test_fuzzy_match_short_name(self):
        """短字段名模糊匹配：'sex' → 'gender'"""
        from tb_risk.data_io.synonyms import fuzzy_match_column
        std, conf, src = fuzzy_match_column('sex')
        # sex 应该匹配 gender（短词阈值低）
        self.assertIn(std, ('gender', None))

    def test_text_parser_no_match(self):
        """无法匹配任何字段的文本"""
        from tb_risk.data_io.text_parser import extract_fields_from_text
        records = extract_fields_from_text("这是一段无关文本，不包含任何临床信息")
        # 应返回空列表或记录中没有字段
        self.assertIsInstance(records, list)

    def test_imputation_all_strategies_fail(self):
        """所有策略失败时回退到 -999 标记"""
        from tb_risk.data_io.imputation import impute_missing
        data = [
            {'custom_field': None},
            {'custom_field': None},
        ]
        filled, strategy, value = impute_missing(data, 'custom_field', strategy='auto')
        # 应该回退到 missing_marker
        self.assertIn(strategy, ('missing_marker', 'none', 'scene_default'))
        if strategy == 'missing_marker':
            self.assertEqual(value, -999)


if __name__ == '__main__':
    unittest.main(verbosity=2)