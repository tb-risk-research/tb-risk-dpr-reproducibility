#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双头预测器测试（round-10 P2 正式化 / round-12 工件 v2 升级）。

``tb_risk.core.dual_head_predictor.DualHeadPredictor`` 的契约：
  - 模型工件（models/sinan_dual_head/{encoder,death_head,bact_head}_v2）
    缺失时优雅降级（available=False，不抛异常）
  - 工件齐备时 load + predict 输出双头（death_risk +
    bacteriology_risk，两套语义名等价）
  - 特征 schema 与 v2 归档 JSON 双向核对（含 HIV/职业派生列；
    职业词表 occupation_top 只在归档，缺失时加载失败）
  - 任务映射 A'=细菌学（找传染源）/ B=死亡预后
"""
import os
import unittest

import pandas as pd

from tb_risk.core.dual_head_predictor import (
    BINARY_FEATS, CATEGORICAL_FEATS, EXCLUDED_LEAKAGE, NUMERIC_FEATS,
    REQUIRED_COLUMNS, MODEL_VERSION, TASK_MAPPING, DualHeadPredictor,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _make_row(age=50, hiv='1'):
    """合成一行 SINAN 特征（GUI 环境无 pyarrow，不依赖 parquet）。

    v2：类别列含 HIV；OCC_TRUNC 为派生列（不直接提供，由
    ID_OCUPA_N + 归档词表在 build_features 内构造）。
    """
    row = {'AGE_YEARS': age, 'NU_CONTATO': 1, 'ID_OCUPA_N': '5'}
    for c in BINARY_FEATS:
        row[c] = hiv if c == 'AGRAVAIDS' else '0'
    for c in CATEGORICAL_FEATS:
        if c == 'HIV':
            row[c] = hiv
        elif c != 'OCC_TRUNC':
            row[c] = '1'
    return row


class TestTaskMapping(unittest.TestCase):

    def test_mapping_structure(self):
        """任务映射：A'=细菌学（找传染源）/ B=死亡预后。"""
        self.assertIn('A_prime', TASK_MAPPING)
        self.assertIn('B', TASK_MAPPING)
        self.assertEqual(TASK_MAPPING['A_prime']['head'],
                         'bacteriology_pos')
        self.assertEqual(TASK_MAPPING['B']['head'], 'death_allcause')
        self.assertIn('找传染源', TASK_MAPPING['A_prime']['label'])
        self.assertIn('死亡预后', TASK_MAPPING['B']['label'])

    def test_leakage_columns_excluded(self):
        """病原学确诊方式列（标签构成）不进特征空间。"""
        feats = set(NUMERIC_FEATS + BINARY_FEATS + CATEGORICAL_FEATS)
        for c in EXCLUDED_LEAKAGE:
            self.assertNotIn(c, feats)

    def test_v2_schema_contract(self):
        """v2 工件契约：版本号、必需列含 HIV/职业原始列、
        派生列 OCC_TRUNC 不要求输入提供。"""
        self.assertEqual(MODEL_VERSION, 'v2')
        self.assertIn('HIV', CATEGORICAL_FEATS)
        self.assertIn('OCC_TRUNC', CATEGORICAL_FEATS)
        self.assertIn('ID_OCUPA_N', REQUIRED_COLUMNS)
        self.assertNotIn('OCC_TRUNC', REQUIRED_COLUMNS)
        self.assertIn('HIV', REQUIRED_COLUMNS)


class TestDualHeadPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pred = DualHeadPredictor()
        cls.ok = cls.pred.load()

    def test_availability_contract(self):
        """is_available 与工件存在性一致；不可用时 predict 降级不抛。"""
        available = self.pred.is_available()
        self.assertEqual(available, self.ok)
        if not available:
            out = self.pred.predict(pd.DataFrame([_make_row()]))
            self.assertFalse(out.get('available'))
            self.assertIn('reason', out)

    def test_predict_dual_output(self):
        """predict 输出双头（两套语义名等价，概率在 [0,1]）。"""
        if not self.ok:
            self.skipTest('双头模型工件不存在（先跑实验脚本落盘）')
        df = pd.DataFrame([_make_row(age=70), _make_row(age=25)])
        out = self.pred.predict(df)
        self.assertTrue(out['available'])
        for key in ('death_risk', 'bacteriology_risk',
                    'task_a_prime_risk', 'task_b_risk'):
            self.assertIn(key, out)
            self.assertEqual(len(out[key]), 2)
            for v in out[key]:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)
        self.assertEqual(out['death_risk'], out['task_b_risk'])
        self.assertEqual(out['bacteriology_risk'],
                         out['task_a_prime_risk'])

    def test_predict_one(self):
        """单行预测返回标量双头。"""
        if not self.ok:
            self.skipTest('双头模型工件不存在')
        out = self.pred.predict_one(_make_row())
        self.assertTrue(out['available'])
        self.assertIsInstance(out['death_risk'], float)
        self.assertIsInstance(out['bacteriology_risk'], float)

    def test_missing_numeric_column_raises(self):
        """缺数值特征列 → ValueError（系统边界显式报错）。"""
        if not self.ok:
            self.skipTest('双头模型工件不存在')
        with self.assertRaises(ValueError):
            self.pred.predict(pd.DataFrame([{'CS_SEXO': '1'}]))

    def test_nan_age_single_row(self):
        """单行缺失年龄：age_med 回退常数，输出无 NaN（round-13 边角）。"""
        if not self.ok:
            self.skipTest('双头模型工件不存在')
        row = _make_row()
        row['AGE_YEARS'] = float('nan')
        out = self.pred.predict(pd.DataFrame([row]))
        self.assertTrue(out['available'])
        for key in ('death_risk', 'bacteriology_risk'):
            self.assertEqual(len(out[key]), 1)
            self.assertTrue(pd.notna(out[key][0]))
            self.assertGreaterEqual(out[key][0], 0.0)
            self.assertLessEqual(out[key][0], 1.0)

    def test_elderly_high_death_risk(self):
        """方向性：老年行死亡风险高于年轻行（模型语义健全）。"""
        if not self.ok:
            self.skipTest('双头模型工件不存在')
        out = self.pred.predict(pd.DataFrame(
            [_make_row(age=75), _make_row(age=20)]))
        self.assertGreater(out['death_risk'][0], out['death_risk'][1])


if __name__ == '__main__':
    unittest.main()
