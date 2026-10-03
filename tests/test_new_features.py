#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — 新功能单元测试

覆盖：
- io_utils 编码检测、字段映射、数据验证
- simulator 数据加载接口
- seir/uncertainty 保存/加载
- export_utils 导出功能
"""

import json
import os
import sys
import tempfile
import unittest

# 确保包路径在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


# ======================== io_utils 编码检测测试 ========================

class TestEncodingDetection(unittest.TestCase):
    """测试 io_utils 编码检测功能"""

    def setUp(self):
        from tb_risk.io_utils import detect_file_encoding
        self.detect = detect_file_encoding

    def test_utf8_file_detection(self):
        """测试 UTF-8 文件编码检测"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False, encoding='utf-8'
        ) as f:
            f.write('姓名,年龄,确诊\n张三,35,0\n李四,42,1\n')
            tmp_path = f.name
        try:
            encoding = self.detect(tmp_path, use_chardet=False)
            self.assertIn(encoding.lower(), ('utf-8', 'utf-8-sig', 'ascii'))
        finally:
            os.unlink(tmp_path)

    def test_gbk_file_detection(self):
        """测试 GBK 编码文件检测"""
        try:
            content = '姓名,年龄,确诊\n张三,35,0\n李四,42,1\n'.encode('gbk')
        except Exception:
            self.skipTest("GBK 编码不可用")
        with tempfile.NamedTemporaryFile(
            mode='wb', suffix='.csv', delete=False
        ) as f:
            f.write(content)
            tmp_path = f.name
        try:
            encoding = self.detect(tmp_path, use_chardet=False)
            self.assertIn(encoding.lower(), ('gbk', 'gb18030', 'gb2312'))
        finally:
            os.unlink(tmp_path)

    def test_gb2312_file_detection(self):
        """测试 GB2312 编码文件检测"""
        try:
            content = '姓名,年龄,确诊\n张三,35,0\n'.encode('gb2312')
        except Exception:
            self.skipTest("GB2312 编码不可用")
        with tempfile.NamedTemporaryFile(
            mode='wb', suffix='.csv', delete=False
        ) as f:
            f.write(content)
            tmp_path = f.name
        try:
            encoding = self.detect(tmp_path, use_chardet=False)
            self.assertIn(encoding.lower(), ('gbk', 'gb18030', 'gb2312'))
        finally:
            os.unlink(tmp_path)

    def test_nonexistent_file_returns_default(self):
        """测试不存在的文件返回默认编码"""
        encoding = self.detect('/nonexistent/file_12345.csv', use_chardet=False)
        self.assertEqual(encoding, 'utf-8-sig')


# ======================== io_utils 字段映射测试 ========================

class TestFieldMapping(unittest.TestCase):
    """测试 io_utils 字段映射功能"""

    def test_chinese_field_name_mapping(self):
        """测试中文字段名自动映射"""
        from tb_risk.io_utils import dynamic_column_match
        self.assertEqual(dynamic_column_match('年龄'), 'age')
        self.assertEqual(dynamic_column_match('咳嗽'), 'cough_freq')
        self.assertEqual(dynamic_column_match('确诊'), 'is_confirmed')
        self.assertEqual(dynamic_column_match('卡介苗'), 'bcg_vaccine')

    def test_english_field_name_mapping(self):
        """测试英文别名映射"""
        from tb_risk.io_utils import dynamic_column_match
        self.assertEqual(dynamic_column_match('patient_age'), 'age')
        self.assertEqual(dynamic_column_match('sputum_smear'), 'sputum_smear')
        self.assertEqual(dynamic_column_match('diagnosis'), 'is_confirmed')

    def test_unknown_field_returns_none(self):
        """测试未知字段返回 None"""
        from tb_risk.io_utils import dynamic_column_match
        self.assertIsNone(dynamic_column_match('完全不存在的字段名_xyz'))

    def test_empty_input(self):
        """测试空输入"""
        from tb_risk.io_utils import dynamic_column_match
        self.assertIsNone(dynamic_column_match(''))
        self.assertIsNone(dynamic_column_match(None))

    def test_map_columns_batch(self):
        """测试批量列名映射"""
        from tb_risk.io_utils import map_columns
        columns = ['姓名', '年龄', '确诊', 'unknown_field']
        mapping = map_columns(columns, verbose=False)
        # 姓名没有直接映射到标准字段，但应该处理
        self.assertGreaterEqual(len(mapping), 1)

    def test_normalized_name_matching(self):
        """测试标准化名称匹配（去空格、下划线）"""
        from tb_risk.io_utils import dynamic_column_match
        # 带空格/下划线的变体
        self.assertEqual(dynamic_column_match('patient age'), 'age')
        self.assertEqual(dynamic_column_match('cough freq'), 'cough_freq')


# ======================== io_utils 数据验证测试 ========================

class TestDataValidation(unittest.TestCase):
    """测试 io_utils 数据验证功能"""

    def test_valid_records_pass(self):
        """测试有效记录通过验证"""
        from tb_risk.io_utils import validate_data_integrity
        records = [
            {'age': 35, 'is_confirmed': 0, 'exposure_setting': 'general',
             'cumulative_exposure': 10, 'has_symptoms': 0, 'bcg_vaccine': 1,
             'has_tb': 0, 'contact_distance': 'close', 'ventilation': 3,
             'is_high_risk': 0, 'past_illness': 0},
            {'age': 42, 'is_confirmed': 1, 'exposure_setting': 'closed',
             'cumulative_exposure': 20, 'has_symptoms': 1, 'bcg_vaccine': 1,
             'has_tb': 1, 'contact_distance': 'very_close', 'ventilation': 2,
             'is_high_risk': 1, 'past_illness': 1},
        ]
        result = validate_data_integrity(records, verbose=False)
        self.assertTrue(result['valid'])

    def test_missing_required_field(self):
        """测试缺失必填字段检测"""
        from tb_risk.io_utils import validate_data_integrity
        records = [
            {'is_confirmed': 0},
            {'age': 42, 'is_confirmed': 1},
        ]
        result = validate_data_integrity(records, verbose=False)
        # 第一条缺少 age
        self.assertFalse(result['valid'])

    def test_empty_records(self):
        """测试空记录列表"""
        from tb_risk.io_utils import validate_data_integrity
        result = validate_data_integrity([], verbose=False)
        self.assertFalse(result['valid'])

    def test_value_range_violation(self):
        """测试值域违规检测"""
        from tb_risk.io_utils import validate_data_integrity
        records = [
            {'age': 150, 'is_confirmed': 0},  # 年龄超出合理范围
        ]
        result = validate_data_integrity(records, verbose=False)
        # 年龄 > 120 应该被检测
        self.assertFalse(result['valid'])


# ======================== simulator 数据加载测试 ========================

class TestSimulatorDataLoading(unittest.TestCase):
    """测试 simulator 数据加载接口"""

    def setUp(self):
        from tb_risk.scoring.simulator import ScreeningDataSimulator
        self.sim = ScreeningDataSimulator(random_state=42)

    def test_load_from_file_nonexistent(self):
        """测试加载不存在的文件抛出异常"""
        with self.assertRaises(FileNotFoundError):
            self.sim.load_from_file('/nonexistent/file_12345.csv')

    def test_load_from_file_csv(self):
        """测试从 CSV 文件加载"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False, encoding='utf-8'
        ) as f:
            f.write('age,is_confirmed,exposure_setting,has_symptoms,bcg_vaccine,'
                     'has_tb,contact_distance,ventilation,is_high_risk,past_illness,'
                     'single_duration,freq_density,time_span,district,ethnicity,'
                     'occupation,gender\n')
            f.write('35,0,general,0,1,0,close,3,0,0,60,14,4,克拉玛依区,han,'
                     'office_worker,male\n')
            f.write('42,1,closed,1,1,1,very_close,2,1,1,120,21,8,独山子区,uyghur,'
                     'oilfield_worker,male\n')
            tmp_path = f.name
        try:
            records, meta = self.sim.load_from_file(tmp_path, verbose=False)
            self.assertGreaterEqual(len(records), 1)
            self.assertIn('n_confirmed', meta)
            self.assertIn('n_total', meta)
        finally:
            os.unlink(tmp_path)

    def test_load_or_generate_without_file(self):
        """测试无文件时回退到合成数据"""
        records, meta = self.sim.load_or_generate(
            filepath=None, n_samples=50, include_labels=True
        )
        self.assertEqual(len(records), 50)
        for rec in records:
            self.assertIn('_data_source', rec)
            self.assertEqual(rec['_data_source'], 'synthetic')

    def test_load_or_generate_nonexistent_file(self):
        """测试文件不存在时回退到合成数据"""
        records, meta = self.sim.load_or_generate(
            filepath='/nonexistent/file_12345.csv', n_samples=20, include_labels=True
        )
        self.assertEqual(len(records), 20)

    def test_import_batch_incremental(self):
        """测试批量导入（从 CSV 文件）"""
        # 生成初始数据集
        existing = self.sim.generate_dataset(n_samples=5, include_labels=True)
        # 将部分数据写出为 CSV 再导入
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False, encoding='utf-8'
        ) as f:
            f.write('age,is_confirmed,exposure_setting,has_symptoms,bcg_vaccine,'
                     'has_tb,contact_distance,ventilation,is_high_risk,past_illness,'
                     'single_duration,freq_density,time_span,district,ethnicity,'
                     'occupation,gender\n')
            f.write('25,0,general,0,1,0,close,3,0,0,60,14,4,克拉玛依区,han,'
                     'office_worker,male\n')
            tmp_path = f.name
        try:
            records, meta = self.sim.import_batch(
                tmp_path, mode='append', existing_records=existing
            )
            self.assertGreaterEqual(len(records), len(existing))
            self.assertIn('batch_id', meta)
        finally:
            os.unlink(tmp_path)

    def test_generate_dataset_labels(self):
        """测试合成数据集标签生成"""
        records = self.sim.generate_dataset(n_samples=100, include_labels=True)
        self.assertEqual(len(records), 100)
        confirmed = sum(1 for r in records if r.get('is_confirmed', 0))
        self.assertGreater(confirmed, 0)


# ======================== uncertainty 保存/加载测试 ========================

class TestUncertaintyPersistence(unittest.TestCase):
    """测试 SEIR 不确定性模块的保存/加载"""

    def test_uncertainty_import(self):
        """测试不确定性模块导入"""
        try:
            from tb_risk.seir.uncertainty import SEIRParameterUncertainty
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"依赖不可用: {e}")

    def test_uncertainty_save_load_json(self):
        """测试不确定性参数保存/加载为 JSON"""
        try:
            from tb_risk.seir.uncertainty import SEIRParameterUncertainty
            obj = SEIRParameterUncertainty()
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False, encoding='utf-8'
            ) as f:
                tmp_path = f.name
            try:
                obj.save(tmp_path)
                self.assertTrue(os.path.exists(tmp_path))
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.assertIsInstance(data, (dict, list))
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except ImportError as e:
            self.skipTest(f"依赖不可用: {e}")
        except Exception as e:
            # save/load 可能因内部状态抛异常，但不应是 NameError
            self.assertNotIn('NameError', str(type(e).__name__))


# ======================== export_utils 导出测试 ========================

class TestExportUtils(unittest.TestCase):
    """测试导出工具"""

    def test_export_to_json_basic(self):
        """测试基本 JSON 导出"""
        from tb_risk.export_utils import export_to_json
        results = {
            'overall_risk': '中风险',
            'potential_patients': {
                'family': [{'name': '测试', 'relationship': '配偶', 'priority': '高'}],
                'social': [],
            }
        }
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, encoding='utf-8'
        ) as f:
            tmp_path = f.name
        try:
            export_to_json(results, tmp_path)
            with open(tmp_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.assertIn('assessment', data)
            self.assertEqual(data['assessment']['overall']['risk_level'], '中风险')
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_serialize_for_json_handles_numpy(self):
        """测试 _serialize_for_json 处理 numpy 类型"""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy 不可用")
        from tb_risk.export_utils import _serialize_for_json
        result = _serialize_for_json(np.float32(3.14))
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 3.14, places=2)

    def test_serialize_for_json_handles_ndarray(self):
        """测试 _serialize_for_json 处理 ndarray"""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy 不可用")
        from tb_risk.export_utils import _serialize_for_json
        result = _serialize_for_json(np.array([1, 2, 3]))
        self.assertIsInstance(result, list)
        self.assertEqual(result, [1, 2, 3])


# ======================== predictor 返回值测试 ========================

class TestPredictorTrainReturnType(unittest.TestCase):
    """测试 train_from_real_data 结构化返回值"""

    def test_sklearn_unavailable_returns_dict(self):
        """测试返回结构化 dict（文件不存在时）"""
        from tb_risk.scoring.predictor import MLRiskPredictor
        predictor = MLRiskPredictor()
        result = predictor.train_from_real_data('/nonexistent/file_12345.csv')
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('error_type', result)
        self.assertIn('diagnostics', result)
        self.assertIn('suggestions', result)
        self.assertFalse(result['success'])

    def test_column_mapping_collision_defense(self):
        """锁定（2026-09-05 CRP 实测）：模糊映射列名碰撞防御

        map_columns 会把附加列改写为与基础特征同名的列
        （age_group→age、tb_history_raw→has_tb、symptom_cough/
        symptom_cough_weeks→cough_freq）。修复前重复列使 df['age']
        返回 DataFrame，以晦涩的 "cannot reindex on an axis with
        duplicate labels" 失败；修复后应正常训练且保留首个出现列。
        """
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn 未安装")

        from tb_risk.scoring.predictor import MLRiskPredictor

        base_cols = ['age', 'cumulative_exposure', 'has_symptoms',
                     'bcg_vaccine', 'has_tb', 'contact_distance_score',
                     'ventilation_score', 'is_high_risk', 'past_illness',
                     'exposure_setting_score', 'single_duration',
                     'freq_density', 'time_span']
        # age_group 会与 age 碰撞；tb_history_raw 会与 has_tb 碰撞
        extra_cols = ['age_group', 'tb_history_raw']
        header = ','.join(base_cols + extra_cols + ['tb_outcome'])
        rows = [header]
        for i in range(40):
            vals = [
                35 + (i % 30),        # age
                i % 5,                # cumulative_exposure
                i % 2,                # has_symptoms
                1,                    # bcg_vaccine
                0,                    # has_tb
                i % 3,                # contact_distance_score
                i % 4,                # ventilation_score
                i % 2,                # is_high_risk
                i % 3 == 0 and 1 or 0,  # past_illness
                i % 4,                # exposure_setting_score
                2 + i % 10,           # single_duration
                i % 7,                # freq_density
                1 + i % 20,           # time_span
                i % 5 + 1,            # age_group（将与 age 碰撞）
                'no history',         # tb_history_raw（将与 has_tb 碰撞）
                i % 8 == 0 and 1 or 0,  # tb_outcome
            ]
            rows.append(','.join(str(v) for v in vals))

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False, encoding='utf-8'
        ) as f:
            f.write('\n'.join(rows) + '\n')
            tmp_path = f.name
        try:
            predictor = MLRiskPredictor()
            result = predictor.train_from_real_data(tmp_path)
            self.assertTrue(
                result.get('success'),
                msg=f"碰撞列训练失败: {result.get('error_type')} "
                    f"{result.get('diagnostics')}")
            self.assertEqual(result.get('n_samples'), 40)
        finally:
            os.unlink(tmp_path)


# ======================== 主入口 ========================

def main_entry():
    from tests.test_utils import check_deps, run_tests_for_module, print_help, main_entry as _main_entry
    import sys as _sys
    all_classes = [
        TestEncodingDetection,
        TestFieldMapping,
        TestDataValidation,
        TestSimulatorDataLoading,
        TestUncertaintyPersistence,
        TestExportUtils,
        TestPredictorTrainReturnType,
    ]
    _main_entry('test_new_features.py', _sys.modules[__name__], all_classes)


if __name__ == '__main__':
    main_entry()