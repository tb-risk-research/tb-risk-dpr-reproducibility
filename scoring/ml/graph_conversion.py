#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图转换：异构图转同构图、节点索引查找

从 MLRiskPredictor 拆分出的独立函数模块。
所有函数无状态，通过参数传递数据。
"""
import copy
import datetime
import logging
import math
import os
import random

import numpy as np

LOGGER = logging.getLogger("tb_risk.ml")

# 条件导入（与主类保持一致）
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 accuracy_score, precision_score, recall_score,
                                 f1_score, brier_score_loss)
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

NUMPY_AVAILABLE = True



# ============================================================
# 以下函数从 MLRiskPredictor 的实例方法拆分而来
# 每个函数接收 predictor 实例作为第一个参数
# ============================================================

def _convert_hetero_to_homo(predictor, hetero_data):
        """将HeteroData转换为同构Data（修复：保留边特征；边特征缺失时填充零向量）
        
        文献支撑：Liu et al., SIGKDD 2024; Yang et al., HGAT-AMR 2021
        """
        import torch
        from torch_geometric.data import Data

        # 推断设备：优先使用 hetero_data 中已有张量的设备
        device = torch.device('cpu')
        for node_type in getattr(hetero_data, 'x_dict', {}):
            x_t = hetero_data[node_type].x
            if hasattr(x_t, 'device'):
                device = x_t.device
                break

        all_x_list = []

        # 收集所有节点特征
        # 患者节点 (index 0)
        if 'patient' in hetero_data.x_dict and hetero_data['patient'].x is not None:
            all_x_list.append(hetero_data['patient'].x)
            num_patient = hetero_data['patient'].x.shape[0]
        else:
            num_patient = 0
        
        # 家庭节点 (index 1)
        if 'family' in hetero_data.x_dict and hetero_data['family'].x is not None:
            all_x_list.append(hetero_data['family'].x)
            num_family = hetero_data['family'].x.shape[0]
        else:
            num_family = 0
        
        # 社会节点 (index 2)
        if 'social' in hetero_data.x_dict and hetero_data['social'].x is not None:
            all_x_list.append(hetero_data['social'].x)
            num_social = hetero_data['social'].x.shape[0]
        else:
            num_social = 0
        
        # 社区节点 (index 3)
        if 'community' in hetero_data.x_dict and hetero_data['community'].x is not None:
            all_x_list.append(hetero_data['community'].x)
            num_community = hetero_data['community'].x.shape[0]
        else:
            num_community = 0
        
        # 区域节点 (index 4，多尺度特征融合 v5.0)
        if 'region' in hetero_data.x_dict and hetero_data['region'].x is not None:
            all_x_list.append(hetero_data['region'].x)
            num_region = hetero_data['region'].x.shape[0]
        else:
            num_region = 0
        
        # 合并所有节点特征
        x = torch.cat(all_x_list, dim=0) if all_x_list else torch.zeros((0, 22), dtype=torch.float, device=device)
        
        # 收集所有边和边特征（文献支撑：Zhu et al., GAST 2024）
        edge_index_list = []
        edge_attr_list = []
        
        # 患者到家庭的边
        if ('patient', 'infects', 'family') in hetero_data.edge_types:
            edge_pf = hetero_data[('patient', 'infects', 'family')].edge_index
            edge_pf = edge_pf.clone()
            edge_pf[1] += num_patient
            edge_index_list.append(edge_pf)
            if hasattr(hetero_data[('patient', 'infects', 'family')], 'edge_attr') and hetero_data[('patient', 'infects', 'family')].edge_attr is not None:
                edge_attr_list.append(hetero_data[('patient', 'infects', 'family')].edge_attr)
            else:
                # 填充6维零向量
                num_edges = edge_pf.shape[1]
                edge_attr_list.append(torch.zeros((num_edges, 6), dtype=torch.float, device=device))
        
        # 患者到社会的边
        if ('patient', 'infects', 'social') in hetero_data.edge_types:
            edge_ps = hetero_data[('patient', 'infects', 'social')].edge_index
            edge_ps = edge_ps.clone()
            edge_ps[1] += num_patient + num_family
            edge_index_list.append(edge_ps)
            if hasattr(hetero_data[('patient', 'infects', 'social')], 'edge_attr') and hetero_data[('patient', 'infects', 'social')].edge_attr is not None:
                edge_attr_list.append(hetero_data[('patient', 'infects', 'social')].edge_attr)
            else:
                # 填充6维零向量
                num_edges = edge_ps.shape[1]
                edge_attr_list.append(torch.zeros((num_edges, 6), dtype=torch.float, device=device))
        
        # 家庭内部边
        if ('family', 'lives_with', 'family') in hetero_data.edge_types:
            edge_ff = hetero_data[('family', 'lives_with', 'family')].edge_index
            edge_ff = edge_ff.clone()
            edge_ff[0] += num_patient
            edge_ff[1] += num_patient
            edge_index_list.append(edge_ff)
            if hasattr(hetero_data[('family', 'lives_with', 'family')], 'edge_attr') and hetero_data[('family', 'lives_with', 'family')].edge_attr is not None:
                edge_attr_list.append(hetero_data[('family', 'lives_with', 'family')].edge_attr)
            else:
                # 填充6维零向量
                num_edges = edge_ff.shape[1]
                edge_attr_list.append(torch.zeros((num_edges, 6), dtype=torch.float, device=device))
        
        # 家庭到社区的边
        if ('family', 'belongs_to', 'community') in hetero_data.edge_types:
            edge_fc = hetero_data[('family', 'belongs_to', 'community')].edge_index
            edge_fc = edge_fc.clone()
            edge_fc[0] += num_patient
            edge_fc[1] += num_patient + num_family + num_social
            edge_index_list.append(edge_fc)
            if hasattr(hetero_data[('family', 'belongs_to', 'community')], 'edge_attr') and hetero_data[('family', 'belongs_to', 'community')].edge_attr is not None:
                edge_attr_list.append(hetero_data[('family', 'belongs_to', 'community')].edge_attr)
            else:
                # 填充6维零向量
                num_edges = edge_fc.shape[1]
                edge_attr_list.append(torch.zeros((num_edges, 6), dtype=torch.float, device=device))
        
        # 社会到社区的边
        if ('social', 'belongs_to', 'community') in hetero_data.edge_types:
            edge_sc = hetero_data[('social', 'belongs_to', 'community')].edge_index
            edge_sc = edge_sc.clone()
            edge_sc[0] += num_patient + num_family
            edge_sc[1] += num_patient + num_family + num_social
            edge_index_list.append(edge_sc)
            if hasattr(hetero_data[('social', 'belongs_to', 'community')], 'edge_attr') and hetero_data[('social', 'belongs_to', 'community')].edge_attr is not None:
                edge_attr_list.append(hetero_data[('social', 'belongs_to', 'community')].edge_attr)
            else:
                # 填充6维零向量
                num_edges = edge_sc.shape[1]
                edge_attr_list.append(torch.zeros((num_edges, 6), dtype=torch.float, device=device))
        
        # 社区到区域的边（多尺度特征融合 v5.0：个体—家庭—社区—区域 链）
        if ('community', 'belongs_to', 'region') in hetero_data.edge_types:
            edge_cr = hetero_data[('community', 'belongs_to', 'region')].edge_index
            edge_cr = edge_cr.clone()
            edge_cr[0] += num_patient + num_family + num_social
            edge_cr[1] += num_patient + num_family + num_social + num_community
            edge_index_list.append(edge_cr)
            if hasattr(hetero_data[('community', 'belongs_to', 'region')], 'edge_attr') and hetero_data[('community', 'belongs_to', 'region')].edge_attr is not None:
                edge_attr_list.append(hetero_data[('community', 'belongs_to', 'region')].edge_attr)
            else:
                # 填充6维零向量
                num_edges = edge_cr.shape[1]
                edge_attr_list.append(torch.zeros((num_edges, 6), dtype=torch.float, device=device))
        
        # 合并所有边索引
        edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.zeros((2, 0), dtype=torch.long, device=device)
        
        # 合并所有边特征（6维）- 只要有边就有特征，即使全是零向量
        edge_attr = None
        if edge_attr_list:
            edge_attr = torch.cat(edge_attr_list, dim=0)
        
        # 创建同构图数据对象
        homo_data = Data(x=x, edge_index=edge_index)
        if edge_attr is not None:
            homo_data.edge_attr = edge_attr
        
        # 存储节点类型信息
        homo_data.num_patient = num_patient
        homo_data.num_family = num_family
        homo_data.num_social = num_social
        homo_data.num_community = num_community
        homo_data.num_region = num_region
        
        return homo_data
    

def _find_target_node_idx(predictor, contact_data, homo_data, contact_type, tb_assessment):
        """找出目标接触者在同构图中的节点索引
        """
        import torch
        
        name_to_find = contact_data.get('name', '')
        
        # 计算索引偏移
        patient_end = homo_data.num_patient
        family_end = patient_end + homo_data.num_family

        if contact_type == 'family':
            # 在家庭节点中查找
            entries = tb_assessment.family_entries
            for i, entry in enumerate(entries):
                if entry.get('name', '') == name_to_find:
                    return patient_end + i
        elif contact_type == 'social':
            # 在社会节点中查找
            entries = tb_assessment.social_entries
            for i, entry in enumerate(entries):
                if entry.get('name', '') == name_to_find:
                    return family_end + i
        
        # 未找到时返回None
        return None
    

