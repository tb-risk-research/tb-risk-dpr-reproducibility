#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多尺度特征融合测试套件（v5.0：个体—家庭—社区—区域）

覆盖：
- 区域级节点特征提取（_extract_region_features：30 维、配置驱动、默认回退）
- 层次化异质图构建（heterogeneous_network：区域节点 + 社区→区域边）
- 同构图转换（graph_conversion：区域节点偏移 + 社区→区域边偏移）
- 合成图生成（data_preparation：区域节点 + mask 排除）
- NumPy 回退图（_numpy_gnn：区域节点参与传播）

遵循项目 unittest + pytest 约定，对可选依赖做 skip 保护。
"""
import os
import sys
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    import torch_geometric
    _PYG = True
except ImportError:
    _PYG = False


def _skip_if_no_numpy(cls):
    return unittest.skipUnless(_NUMPY, "需要 numpy")(cls)


def _skip_if_no_torch(cls):
    return unittest.skipUnless(_TORCH, "需要 torch")(cls)


def _skip_if_no_pyg(cls):
    return unittest.skipUnless(_PYG and _TORCH, "需要 torch + torch_geometric")(cls)


def _make_contact(name="接触者A", cumulative_exposure=40, age=35,
                  ventilation=3, has_symptoms=0, record_id=None):
    """构造一个最小合法的接触者数据字典。"""
    return {
        'name': name,
        'record_id': record_id,
        'age': age, 'has_symptoms': has_symptoms, 'has_tb': 0,
        'bcg_vaccine': 1, 'contact_distance': 'close',
        'ventilation': ventilation, 'exposure_setting': 'general',
        'cumulative_exposure': cumulative_exposure,
        'single_duration': 60, 'freq_density': 10, 'time_span': 8,
    }


def _make_assessment(family_entries, social_entries=None, patient_info=None):
    """构造最小合法的评估对象（patient_info 为 dict，供特征提取消费）。"""
    class _Assessment:
        pass
    a = _Assessment()
    a.family_entries = family_entries or []
    a.social_entries = social_entries or []
    a.patient_info = patient_info or {'basic_info': {'sputum_smear': 1, 'has_cavity': 1}}
    return a


# ---------------------------------------------------------------------------
# 1. 区域级节点特征提取
# ---------------------------------------------------------------------------

@_skip_if_no_numpy
class TestRegionFeatureExtraction(unittest.TestCase):
    """_extract_region_features：30 维、配置驱动、默认回退。"""

    def _mixin(self):
        from tb_risk.ml.framework._contact_features import _ContactFeatureMixin
        return _ContactFeatureMixin()

    def test_feature_dim_30(self):
        m = self._mixin()
        feats = m._extract_region_features(_make_assessment([]), num_contacts=0)
        self.assertEqual(len(feats), 30)

    def test_feature_range(self):
        m = self._mixin()
        feats = m._extract_region_features(_make_assessment([]), num_contacts=5)
        for v in feats:
            self.assertTrue(0.0 <= float(v) <= 1.0)

    def test_incidence_high_district_gives_higher_feature0(self):
        m = self._mixin()
        cfg = mock.MagicMock()
        cfg.get.return_value = 121.0
        a = _make_assessment([], patient_info={'basic_info': {'district': '高发区'}})
        cfg.get_dict.return_value = {'高发区': {'incidence_per_100k': 400.0}}
        feats_high = m._extract_region_features(a, num_contacts=0, config=cfg)
        cfg.get_dict.return_value = {'高发区': {'incidence_per_100k': 60.0}}
        feats_low = m._extract_region_features(a, num_contacts=0, config=cfg)
        self.assertGreater(feats_high[0], feats_low[0])

    def test_screening_frequency_shorter_interval_lower_risk(self):
        # 特征1 = 1 - months/12：筛查间隔越短（months 越小）风险暴露窗口越短
        m = self._mixin()
        cfg = mock.MagicMock()
        cfg.get.return_value = 121.0
        cfg.get_dict.return_value = {'克拉玛依区': {'screening_frequency_months': 3.0}}
        feats_short = m._extract_region_features(_make_assessment([]), config=cfg)
        cfg.get_dict.return_value = {'克拉玛依区': {'screening_frequency_months': 12.0}}
        feats_long = m._extract_region_features(_make_assessment([]), config=cfg)
        self.assertGreater(feats_short[1], feats_long[1])

    def test_care_action_active_flag(self):
        m = self._mixin()
        cfg = mock.MagicMock()
        cfg.get.return_value = 121.0
        cfg.get_dict.return_value = {'克拉玛依区': {'care_action_active': True}}
        feats_on = m._extract_region_features(_make_assessment([]), config=cfg)
        cfg.get_dict.return_value = {'克拉玛依区': {'care_action_active': False}}
        feats_off = m._extract_region_features(_make_assessment([]), config=cfg)
        self.assertEqual(feats_on[3], 1.0)
        self.assertEqual(feats_off[3], 0.0)

    def test_contact_scale_feature(self):
        m = self._mixin()
        cfg = mock.MagicMock()
        cfg.get_dict.return_value = {}
        cfg.get.return_value = 121.0
        feats_many = m._extract_region_features(_make_assessment([]), num_contacts=20, config=cfg)
        feats_few = m._extract_region_features(_make_assessment([]), num_contacts=0, config=cfg)
        self.assertGreaterEqual(feats_many[6], feats_few[6])

    def test_seir_onehot_none(self):
        # 区域为抽象节点：潜伏=None（22-25），疾病=None（26-29）
        m = self._mixin()
        feats = m._extract_region_features(_make_assessment([]))
        self.assertEqual(list(feats[22:26]), [0, 0, 0, 1])
        self.assertEqual(list(feats[26:30]), [0, 0, 0, 1])

    def test_no_config_falls_back_to_defaults(self):
        m = self._mixin()
        with mock.patch.object(type(m), '_load_region_config', return_value=None):
            feats = m._extract_region_features(_make_assessment([]), num_contacts=3)
        self.assertEqual(len(feats), 30)
        # 默认克拉玛依区 121/500
        self.assertAlmostEqual(float(feats[0]), 121.0 / 500.0, places=3)

    def test_exported_from_framework(self):
        from tb_risk.ml.framework import HeterogeneousTBNetwork
        self.assertTrue(hasattr(HeterogeneousTBNetwork, '_extract_region_features'))


# ---------------------------------------------------------------------------
# 2. 层次化异质图构建（区域节点 + 社区→区域边）
# ---------------------------------------------------------------------------

@_skip_if_no_pyg
class TestHeterogeneousRegionNode(unittest.TestCase):
    """heterogeneous_network：区域节点与社区→区域边。"""

    def setUp(self):
        from tb_risk.ml.framework import HeterogeneousTBNetwork
        self.builder = HeterogeneousTBNetwork(device='cpu')
        self.assessment = _make_assessment(
            family_entries=[
                _make_contact(name='家1', record_id='f0'),
                _make_contact(name='家2', record_id='f1'),
            ],
            social_entries=[
                _make_contact(name='社1', record_id='s0'),
            ],
            patient_info={'basic_info': {'sputum_smear': 2, 'has_cavity': 1}},
        )
        self.data = self.builder.build_from_assessment(self.assessment)

    def test_region_node_present(self):
        self.assertIn('region', self.data.node_types)

    def test_region_node_shape(self):
        x = self.data['region'].x
        self.assertEqual(tuple(x.shape), (1, 30))

    def test_region_node_dtype_float(self):
        self.assertEqual(self.data['region'].x.dtype, torch.float32)

    def test_community_to_region_edge(self):
        self.assertIn(('community', 'belongs_to', 'region'), self.data.edge_types)

    def test_community_to_region_edge_index(self):
        ei = self.data['community', 'belongs_to', 'region'].edge_index
        self.assertEqual(tuple(ei.shape), (2, 1))
        # 社区节点0 → 区域节点0
        self.assertEqual(int(ei[0, 0]), 0)
        self.assertEqual(int(ei[1, 0]), 0)

    def test_community_to_region_edge_attr_6dim(self):
        ea = self.data['community', 'belongs_to', 'region'].edge_attr
        self.assertEqual(tuple(ea.shape), (1, 6))

    def test_region_features_use_district_config(self):
        # 独山子区（关爱行动激活、筛查间隔3个月）应反映到区域特征
        assessment = _make_assessment(
            family_entries=[_make_contact()],
            patient_info={'basic_info': {'district': '独山子区',
                                         'sputum_smear': 1, 'has_cavity': 1}},
        )
        data = self.builder.build_from_assessment(assessment)
        feats = data['region'].x[0].tolist()
        # 特征3（关爱行动激活）应为 1（独山子区 care_action_active=True）
        self.assertEqual(feats[3], 1.0)


# ---------------------------------------------------------------------------
# 3. 同构图转换（区域节点偏移 + 社区→区域边偏移）
# ---------------------------------------------------------------------------

@_skip_if_no_pyg
class TestHomoConversionRegion(unittest.TestCase):
    """graph_conversion：_convert_hetero_to_homo 正确处理区域节点。"""

    def setUp(self):
        from tb_risk.ml.framework import HeterogeneousTBNetwork
        from tb_risk.scoring.ml.graph_conversion import _convert_hetero_to_homo
        self.convert = _convert_hetero_to_homo
        self.builder = HeterogeneousTBNetwork(device='cpu')
        self.assessment = _make_assessment(
            family_entries=[
                _make_contact(name='家1', record_id='f0'),
                _make_contact(name='家2', record_id='f1'),
            ],
            social_entries=[
                _make_contact(name='社1', record_id='s0'),
            ],
        )
        self.hetero = self.builder.build_from_assessment(self.assessment)
        self.homo = self.convert(None, self.hetero)

    def test_num_region_stored(self):
        self.assertEqual(int(self.homo.num_region), 1)

    def test_num_community_stored(self):
        self.assertEqual(int(self.homo.num_community), 1)

    def test_total_nodes(self):
        expected = (int(self.homo.num_patient) + int(self.homo.num_family)
                    + int(self.homo.num_social) + int(self.homo.num_community)
                    + int(self.homo.num_region))
        self.assertEqual(self.homo.x.shape[0], expected)

    def test_region_node_at_end(self):
        # 区域节点应位于同构图节点列表末尾
        region_offset = (int(self.homo.num_patient) + int(self.homo.num_family)
                         + int(self.homo.num_social) + int(self.homo.num_community))
        # 与 x 的行数-1 一致（区域是最后一个节点）
        self.assertEqual(region_offset, self.homo.x.shape[0] - 1)

    def test_community_to_region_edge_offset(self):
        # 社区→区域边：源=社区偏移，目标=区域偏移
        community_offset = (int(self.homo.num_patient) + int(self.homo.num_family)
                            + int(self.homo.num_social))
        region_offset = community_offset + int(self.homo.num_community)
        # 找到社区→区域边（源节点=community_offset，目标=region_offset）
        ei = self.homo.edge_index
        found = False
        for k in range(ei.shape[1]):
            if int(ei[0, k]) == community_offset and int(ei[1, k]) == region_offset:
                found = True
                break
        self.assertTrue(found, "同构图中应存在 社区→区域 边且偏移正确")

    def test_community_to_region_edge_attr_preserved(self):
        self.assertIsNotNone(self.homo.edge_attr)
        community_offset = (int(self.homo.num_patient) + int(self.homo.num_family)
                            + int(self.homo.num_social))
        region_offset = community_offset + int(self.homo.num_community)
        ei = self.homo.edge_index
        k = None
        for j in range(ei.shape[1]):
            if int(ei[0, j]) == community_offset and int(ei[1, j]) == region_offset:
                k = j
                break
        self.assertIsNotNone(k)
        self.assertEqual(self.homo.edge_attr.shape[1], 6)


# ---------------------------------------------------------------------------
# 4. 合成图生成（区域节点 + mask 排除）
# ---------------------------------------------------------------------------

@_skip_if_no_pyg
class TestSyntheticGraphsRegion(unittest.TestCase):
    """data_preparation：_generate_synthetic_graphs 含区域节点。"""

    def test_graphs_have_region_node(self):
        from tb_risk.scoring.ml.gnn_training.data_preparation import _generate_synthetic_graphs
        train, val, test = _generate_synthetic_graphs(None, n_graphs=3, random_state=42)
        self.assertTrue(train)
        data = train[0]
        num_nodes = data.x.shape[0]
        # 区域节点为最后一个（索引 num_nodes-1），且被 mask 排除
        self.assertFalse(bool(data.mask[num_nodes - 1].item()))
        # 患者节点也被排除
        self.assertFalse(bool(data.mask[0].item()))
        # 其余个体节点参与训练
        n_individuals = num_nodes - 1  # 除去区域节点
        self.assertEqual(int(data.mask[:n_individuals].sum().item()), n_individuals - 1)

    def test_region_node_connected_to_all_individuals(self):
        from tb_risk.scoring.ml.gnn_training.data_preparation import _generate_synthetic_graphs
        train, _, _ = _generate_synthetic_graphs(None, n_graphs=2, random_state=1)
        data = train[0]
        region_idx = data.x.shape[0] - 1
        ei = data.edge_index
        neighbors = set()
        for k in range(ei.shape[1]):
            if int(ei[0, k]) == region_idx:
                neighbors.add(int(ei[1, k]))
        self.assertIn(0, neighbors)  # 区域↔患者
        self.assertGreaterEqual(len(neighbors), 2)  # 区域↔所有个体


# ---------------------------------------------------------------------------
# 5. NumPy 回退图（区域节点参与传播）
# ---------------------------------------------------------------------------

@_skip_if_no_numpy
class TestNumpyRegionNode(unittest.TestCase):
    """_numpy_gnn：区域节点参与图构建与传播。"""

    def setUp(self):
        from tb_risk.ml.gnn._numpy_gnn import NumPyGraphRiskPredictor
        self.predictor = NumPyGraphRiskPredictor(random_state=42)

    def test_feature_matrix_has_region(self):
        family = [_make_contact(name='家1', record_id='f0')]
        social = [_make_contact(name='社1', record_id='s0')]
        X, edges, offsets = self.predictor._build_feature_matrix(
            {'basic_info': {}}, family, social)
        # 节点 = 患者 + 1家 + 1社 + 区域 = 4
        self.assertEqual(X.shape[0], 4)
        self.assertIn('region', offsets)
        region_slice = offsets['region']
        self.assertEqual(region_slice.start, 3)
        self.assertEqual(region_slice.stop, 4)

    def test_region_connected_to_all_individuals(self):
        family = [_make_contact(name='家1', record_id='f0'),
                  _make_contact(name='家2', record_id='f1')]
        social = [_make_contact(name='社1', record_id='s0')]
        X, edges, offsets = self.predictor._build_feature_matrix(
            {'basic_info': {}}, family, social)
        region_idx = offsets['region'].start
        n_nodes = X.shape[0]
        # 区域与所有个体节点（除自身）相连
        connected = {j for i, j in edges if i == region_idx}
        self.assertEqual(connected, set(range(region_idx)))
        # 患者仍与所有接触者相连（排除区域节点）
        patient_connected = {j for i, j in edges if i == 0}
        self.assertEqual(patient_connected, set(range(1, n_nodes - 1)))

    def test_network_risk_still_works_with_region(self):
        family = [_make_contact(name='家1', cumulative_exposure=80, record_id='f0'),
                  _make_contact(name='家2', cumulative_exposure=20, record_id='f1')]
        social = [_make_contact(name='社1', cumulative_exposure=10, record_id='s0')]
        target = family[0]
        r = self.predictor.predict_network_risk(
            target, 'family', {'basic_info': {'sputum_smear': 2, 'has_cavity': 2}},
            family, social)
        self.assertEqual(r['backend'], 'numpy')
        self.assertTrue(r['network_aware'])
        # 节点顺序：患者(0)、family(1..2)、social(3)、region(4) → 目标家1=1
        self.assertEqual(r['node_idx'], 1)
        self.assertTrue(0.0 <= r['risk_probability'] <= 100.0)

    def test_social_offset_shifted_by_region(self):
        family = [_make_contact(name='家1', record_id='f0')]
        social = [_make_contact(name='社1', record_id='s0')]
        X, edges, offsets = self.predictor._build_feature_matrix(
            {'basic_info': {}}, family, social)
        # social 从 2 开始（患者0、family1），region 占最后一位
        self.assertEqual(offsets['social'].start, 2)
        self.assertEqual(offsets['social'].stop, 3)


if __name__ == '__main__':
    unittest.main()
