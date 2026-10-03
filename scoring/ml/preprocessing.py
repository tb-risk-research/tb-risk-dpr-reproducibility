#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据预处理：特征提取、合成数据生成、患者上下文

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

from ...constants import DEFAULT_CONTACT_AGE

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



from ...utils import SyntheticDataCalibrator, expected_feature_means_from_config, _is_yes

# ============================================================
# 以下函数从 MLRiskPredictor 的实例方法拆分而来
# 每个函数接收 predictor 实例作为第一个参数
# ============================================================

def _extract_features_from_records(predictor, records):
        """从记录字典列表提取特征矩阵 (X, y)
        
        将 simulator.load_or_generate() 返回的 raw records 转换为 ML 模型
        所需的 22 维特征矩阵（13 基础 + 9 交互），使真实数据能接入训练主流程。
        
        参数：
            records (list[dict]): 每条记录包含 age, has_symptoms, contact_distance 等字段
        
        返回：
            tuple: (X, y) — X 为 (n, 22) 特征矩阵，y 为 (n,) 标签数组
        """
        import numpy as np
        n = len(records)
        
        # 基础特征提取
        ages = np.array([r.get('age', 30) for r in records], dtype=float)
        cumulative_exposure = np.array([r.get('cumulative_exposure', 0) for r in records], dtype=float)
        has_symptoms = np.array([int(r.get('has_symptoms', 0)) for r in records], dtype=float)
        bcg_vaccine = np.array([int(r.get('bcg_vaccine', 1)) for r in records], dtype=float)
        has_tb = np.array([int(r.get('has_tb', 0)) for r in records], dtype=float)
        
        # 文本字段 → 数值评分（与 ScoringEngine 口径一致）
        contact_distance_score = np.array([
            predictor.DISTANCE_SCORE_MAP.get(r.get('contact_distance', 'medium'), 0.5)
            for r in records
        ], dtype=float)
        ventilation_score = np.array([
            predictor.VENTILATION_SCORE_MAP.get(r.get('ventilation', 3), 0.5)
            for r in records
        ], dtype=float)
        exposure_setting_score = np.array([
            predictor.SETTING_SCORE_MAP.get(r.get('exposure_setting', 'general'), 0.6)
            for r in records
        ], dtype=float)
        
        is_high_risk = np.array([int(r.get('is_high_risk', 0)) for r in records], dtype=float)
        past_illness = np.array([int(r.get('past_illness', 0)) for r in records], dtype=float)
        single_duration = np.array([r.get('single_duration', 60) for r in records], dtype=float)
        freq_density = np.array([r.get('freq_density', 14) for r in records], dtype=float)
        time_span = np.array([r.get('time_span', 4) for r in records], dtype=float)
        
        X_base = np.column_stack([
            ages, cumulative_exposure, has_symptoms, bcg_vaccine,
            has_tb, contact_distance_score, ventilation_score,
            is_high_risk, past_illness, exposure_setting_score,
            single_duration, freq_density, time_span
        ])
        
        # 交互特征计算
        # 免疫抑制评分（从 past_illness_type 推断）
        immunosuppression = np.zeros(n, dtype=float)
        for i, r in enumerate(records):
            pt = r.get('past_illness_type', 'none')
            if pt == 'HIV' or pt == 'hiv':
                immunosuppression[i] = 2.0
            elif pt == 'diabetes':
                immunosuppression[i] = 1.0
            elif pt == 'immunosuppressants':
                immunosuppression[i] = 1.5
            elif pt == 'idu':
                immunosuppression[i] = 2.0
            elif pt not in ('none', '', None):
                immunosuppression[i] = 0.8
        
        diabetes = np.array([
            1.0 if r.get('past_illness_type', '') == 'diabetes' else 0.0
            for r in records
        ], dtype=float)
        
        # 交互特征（部分字段真实数据中可能缺失，使用合理默认值）
        delay_days = np.array([r.get('delay_days', 30) for r in records], dtype=float)
        cough_freq = np.array([r.get('cough_freq', 0) for r in records], dtype=float)
        contact_count = np.array([r.get('contact_count', 1) for r in records], dtype=float)
        
        age_immuno = ages * immunosuppression
        dm_tb_synergy = diabetes * has_tb
        age_bcg_decay = ages * (1 - bcg_vaccine)
        symptom_delay = has_symptoms * np.minimum(delay_days / 30, 1.0)
        cough_contact = np.minimum(cough_freq / 20, 1.0) * (contact_count / 10)
        highrisk_comorbid = is_high_risk * past_illness
        immune_bcg = (1 - bcg_vaccine) * immunosuppression
        exposure_accumulation = (cumulative_exposure / 80) * (time_span / 10)
        age_diabetes = ages * diabetes
        
        X_interactions = np.column_stack([
            age_immuno, dm_tb_synergy, age_bcg_decay, symptom_delay,
            cough_contact, highrisk_comorbid, immune_bcg, exposure_accumulation, age_diabetes
        ])
        
        X = np.hstack([X_base, X_interactions])
        
        # 标签提取
        y = np.array([int(r.get('is_confirmed', 0)) for r in records], dtype=int)
        
        return X, y


def _generate_synthetic_training_data(predictor, n_samples=2000, random_state=42, local_params=None):
        """基于文献发现生成合成训练数据（含交互特征）

        关键文献发现：
        - 年轻年龄是独立预测因子(aOR=2.41, 95%CI:1.48-3.94)（东开普省研究, IJERPH 2025）
        - 较大家庭规模(aOR=1.12 per additional member)
        - 症状、BCG、暴露时长等均为已知风险因素

        ⚠️ 循环论证风险（设计局限，使用前务必阅读）：
        本方法生成的标签 y 通过 log_odds 线性组合特征得到，而其系数（症状+0.88、
        既往TB+1.25、BCG-0.35、暴露累积+0.005/单位等）与 ScoringEngine 的风险评分
        逻辑同源——症状、BCG、暴露、既往史在两套体系中承担相似的角色与方向。
        这意味着：
          1. ML 模型在此合成数据上学到的，本质上是 ScoringEngine 评分逻辑的"影子"，
             训练得到的 AUC（约 0.84）反映的是模型复现评分引擎的能力，
             而非在真实流行病学数据上的泛化能力；
          2. 合成数据中的特征-标签关系是人为指定的，无法验证真实世界中混淆因素、
             交互效应、人群异质性的影响；
          3. 用该合成数据训练的模型去"验证"评分引擎或与之对比，会高估一致性。
        缓解措施：
          - 报告 ML 性能时必须标注"基于合成数据，不可外推到真实人群"；
          - 一旦获得本地真实筛查数据，应立即用真实数据重新训练并报告独立指标；
          - 严格的模型对比应在独立、前瞻采集的真实数据集上进行。

        注意：合成数据仅用于演示与流程验证目的，实际应用需使用本地流行病学数据重新训练。
        特征映射数值为作者基于文献定性描述进行的量化假设，需更多实证研究校准。

        参数：
            n_samples (int): 生成样本数，默认2000
            random_state (int): 随机种子，确保可复现性
            local_params (dict|None): 可选的本土化校准参数（由 KaramayLocalizer提供），
               包含 flp_percentage_min/max/default, base_incidence_per_100k 等

        返回：
            tuple: (X, y) - X为特征矩阵(n_samples, 22)，y为标签数组(n_samples,)
        """
        # 检查是否有缓存数据可用
        current_params = (n_samples, random_state)
        if predictor._cached_training_data is not None and predictor._cached_training_params == current_params:
            return predictor._cached_training_data
        
        import numpy as np
        rng = np.random.RandomState(random_state)
        
        ages = rng.randint(0, 90, n_samples)
        cumulative_exposure = rng.exponential(40, n_samples)
        has_symptoms = rng.binomial(1, 0.3, n_samples)
        bcg_vaccine = rng.binomial(1, 0.7, n_samples)
        has_tb = rng.binomial(1, 0.1, n_samples)
        contact_distance_score = rng.choice([0.1, 0.25, 0.5, 0.8, 1.0], n_samples,
                                            p=[0.1, 0.15, 0.35, 0.3, 0.1])
        ventilation_score = rng.choice([0.15, 0.3, 0.5, 0.75, 1.0], n_samples,
                                       p=[0.15, 0.2, 0.35, 0.2, 0.1])
        is_high_risk = rng.binomial(1, 0.2, n_samples)
        past_illness_type = rng.choice(['none', 'hiv', 'diabetes', 'immunosuppressants', 'other'],
                                       n_samples, p=[0.7, 0.05, 0.1, 0.05, 0.1])
        past_illness = (past_illness_type != 'none').astype(float)
        exposure_setting_score = rng.choice([0.3, 0.6, 0.85, 0.9, 1.0], n_samples,
                                            p=[0.2, 0.35, 0.2, 0.15, 0.1])
        single_duration = rng.randint(5, 481, n_samples)
        freq_density = rng.randint(1, 31, n_samples)
        time_span = rng.randint(1, 53, n_samples)
        
        X_base = np.column_stack([
            ages, cumulative_exposure, has_symptoms, bcg_vaccine,
            has_tb, contact_distance_score, ventilation_score,
            is_high_risk, past_illness, exposure_setting_score,
            single_duration, freq_density, time_span
        ])
        hiv = (past_illness_type == 'hiv').astype(float)
        diabetes = (past_illness_type == 'diabetes').astype(float)  # 仅糖尿病，不包含other
        immunosuppressants = (past_illness_type == 'immunosuppressants').astype(float)
        other_illness = (past_illness_type == 'other').astype(float)
        # 移除重复加权：past_illness已包含在hiv/diabetes/immunosuppressants/other中
        immunosuppression_score = hiv * 2.0 + diabetes * 1.0 + immunosuppressants * 1.5 + other_illness * 0.8
        
        delay_days = rng.exponential(30, n_samples)
        cough_freq = rng.randint(0, 21, n_samples)
        contact_count = rng.randint(1, 15, n_samples)
        
        age_immuno = ages * immunosuppression_score
        dm_tb_synergy = diabetes * has_tb
        age_bcg_decay = ages * (1 - bcg_vaccine)
        symptom_delay = has_symptoms * np.minimum(delay_days / 30, 1.0)
        cough_contact = np.minimum(cough_freq / 20, 1.0) * (contact_count / 10)
        highrisk_comorbid = is_high_risk * past_illness
        immune_bcg = (1 - bcg_vaccine) * immunosuppression_score
        exposure_accumulation = (cumulative_exposure / 80) * (time_span / 10)
        age_diabetes = ages * diabetes
        
        X_interactions = np.column_stack([
            age_immuno, dm_tb_synergy, age_bcg_decay, symptom_delay,
            cough_contact, highrisk_comorbid, immune_bcg, exposure_accumulation, age_diabetes
        ])
        
        X = np.hstack([X_base, X_interactions])
        
        log_odds_base = -3.0
        if local_params and 'base_incidence_per_100k' in local_params:
            local_rate = local_params['base_incidence_per_100k']
            log_odds_base += (local_rate / 62.0 - 1.0) * 0.8

        try:
            calibrator = SyntheticDataCalibrator(
                random_state=random_state, region=local_params.get('region', 'default') if local_params else 'default'
            )
            expected = expected_feature_means_from_config()
            calib = calibrator.run_full_calibration(expected, local_params, n_samples)
            log_odds_base = calib['log_odds_base']
            if 'validation_report' in calib:
                LOGGER.info(f"SyntheticDataCalibrator: prevalence_match={calib['validation_report'].get('moment_matching', {}).get('relative_error', 'N/A')}")
        except Exception as e:
            LOGGER.warning(f"SyntheticDataCalibrator校准失败: {e}")

        log_odds = (
            log_odds_base
            + 0.02 * (ages < 5).astype(float)
            + 0.015 * ((ages >= 5) & (ages < 15)).astype(float)
            + 0.88 * has_symptoms
            - 0.35 * bcg_vaccine
            + 1.25 * has_tb
            + 0.8 * contact_distance_score
            + 0.6 * ventilation_score
            + 0.5 * is_high_risk
            + 0.7 * past_illness
            + 0.5 * exposure_setting_score
            + 0.005 * np.minimum(cumulative_exposure, 200)
            + 0.003 * np.minimum(single_duration, 480)
            + 0.02 * np.minimum(freq_density, 30)
            + 0.01 * np.minimum(time_span, 52)
            + 0.003 * age_immuno
            + 0.8 * dm_tb_synergy
            + 0.005 * age_bcg_decay
            + 0.6 * symptom_delay
            + 0.5 * cough_contact
            + 0.4 * highrisk_comorbid
            + 0.5 * immune_bcg
            + 0.3 * exposure_accumulation
            + 0.004 * age_diabetes
        )
        
        noise = rng.normal(0, 0.5, n_samples)
        prob = 1.0 / (1.0 + np.exp(-(log_odds + noise)))
        y = (prob > 0.5).astype(int)
        
        # 缓存结果
        predictor._cached_training_data = (X, y)
        predictor._cached_training_params = current_params
        
        return X, y
    

def extract_features(predictor, contact_data, contact_type='family'):
        """从接触者数据中提取特征向量
        
        将接触者数据字典转换为13维特征向量，用于ML模型预测。
        特征映射基于文献定性描述进行的量化假设：
        - 接触距离：very_close=1.0, close=0.8, medium=0.5, far=0.25, distant=0.1
          （参考Escombe et al. 2007的相对风险概念）
        - 通风条件：1(极差)=1.0, 2(差)=0.75, 3(中)=0.5, 4(良)=0.3, 5(优)=0.15
          （1=完全密闭风险最高，5=完全户外风险最低；值越大表示通风越差→传播风险越高）
        - 暴露场景：crowded=1.0, closed=0.9, oilfield_camp=0.85, general=0.6, outdoor=0.3
          （基于WHO结核病感染控制指南的定性描述）
        
        参数：
            contact_data (dict): 接触者数据字典，需包含age, has_symptoms, bcg_vaccine等字段
            contact_type (str): 'family'（家庭接触者）或 'social'（社会接触者），
                               影响默认值的选择
        
        返回：
            numpy.ndarray: 特征向量，形状为(1, 13)
        """
        age = contact_data.get('age', DEFAULT_CONTACT_AGE)
        if isinstance(age, str):
            try:
                age = int(age)
            except (ValueError, TypeError):
                age = DEFAULT_CONTACT_AGE
        
        cumulative_exposure = contact_data.get('cumulative_exposure', 0)
        if isinstance(cumulative_exposure, str):
            try:
                cumulative_exposure = float(cumulative_exposure)
            except (ValueError, TypeError):
                cumulative_exposure = 0
        
        has_symptoms = 1 if _is_yes(contact_data.get('has_symptoms', 0)) else 0
        bcg_vaccine = 1 if _is_yes(contact_data.get('bcg_vaccine', 0)) else 0
        has_tb = 1 if _is_yes(contact_data.get('has_tb', 0)) else 0
        
        contact_distance = contact_data.get('contact_distance', 'close' if contact_type == 'family' else 'medium')
        contact_distance_score = predictor.DISTANCE_SCORE_MAP.get(contact_distance, 0.5)
        
        ventilation = contact_data.get('ventilation', 3)
        try:
            ventilation = int(ventilation)
        except (ValueError, TypeError):
            ventilation = 3
        ventilation_score = predictor.VENTILATION_SCORE_MAP.get(ventilation, 0.5)
        
        is_high_risk = 1 if _is_yes(contact_data.get('is_high_risk', 0)) else 0
        past_illness = 1 if _is_yes(contact_data.get('past_illness', 0)) else 0
        
        exposure_setting = contact_data.get('exposure_setting', 'general')
        exposure_setting_score = predictor.SETTING_SCORE_MAP.get(exposure_setting, 0.6)
        
        single_duration = contact_data.get('single_duration', 30)
        try:
            single_duration = int(single_duration)
        except (ValueError, TypeError):
            single_duration = 30
        
        freq_density = contact_data.get('freq_density', 14 if contact_type == 'family' else 2)
        try:
            freq_density = int(freq_density)
        except (ValueError, TypeError):
            freq_density = 14 if contact_type == 'family' else 2
        
        time_span = contact_data.get('time_span', 4)
        try:
            time_span = int(time_span)
        except (ValueError, TypeError):
            time_span = 4
        
        features = np.array([[
            age, cumulative_exposure, has_symptoms, bcg_vaccine,
            has_tb, contact_distance_score, ventilation_score,
            is_high_risk, past_illness, exposure_setting_score,
            single_duration, freq_density, time_span
        ]])
        
        return features
    

def extract_features_with_interactions(predictor, contact_data, contact_type='family',
                                            patient_ftd=None, patient_cough_freq=None,
                                            contact_count=None):
        """提取13维基础特征 + 9维交互特征 = 22维特征向量
        
        基于生物学机制显式创建交互特征，使SHAP分析可单独展示每个交互特征的贡献。
        
        交互特征文献支撑：
        - 糖尿病×结核病史交互：IDCases 2025，糖尿病患者TB风险增加3倍，存在乘法级协同
        - 年龄×未接种BCG交互：Lancet 2021/JAMA 1994，BCG保护效力随年龄衰减
        - 特征交互层架构：Computer Methods and Programs in Biomedicine, 2024
        
        参数：
            contact_data (dict): 接触者数据字典
            contact_type (str): 'family' 或 'social'
            patient_ftd (int|None): 患者延迟就诊天数，None时使用实例属性
            patient_cough_freq (int|None): 患者咳嗽频率，None时使用实例属性
            contact_count (int|None): 总接触者数量，None时使用实例属性
        
        返回：
            numpy.ndarray: 特征向量，形状为(1, 22)
        """
        base_features = predictor.extract_features(contact_data, contact_type)
        
        # 安全转换年龄
        age_raw = contact_data.get('age', DEFAULT_CONTACT_AGE)
        try:
            age = float(age_raw)
        except (ValueError, TypeError):
            age = float(DEFAULT_CONTACT_AGE)
        
        has_symptoms = 1 if _is_yes(contact_data.get('has_symptoms', 0)) else 0
        bcg_vaccine = 1 if _is_yes(contact_data.get('bcg_vaccine', 0)) else 0
        past_illness = 1 if _is_yes(contact_data.get('past_illness', 0)) else 0
        is_high_risk = 1 if _is_yes(contact_data.get('is_high_risk', 0)) else 0
        has_tb = 1 if _is_yes(contact_data.get('has_tb', 0)) else 0
        
        # 安全转换delay_days
        delay_days_raw = patient_ftd if patient_ftd is not None else predictor.patient_ftd
        try:
            delay_days = int(delay_days_raw)
        except (ValueError, TypeError):
            delay_days = 0
        
        # 安全转换cough_freq
        cough_freq_raw = patient_cough_freq if patient_cough_freq is not None else predictor.patient_cough_freq
        try:
            cough_freq = int(cough_freq_raw)
        except (ValueError, TypeError):
            cough_freq = 0
        
        # 安全转换n_contacts
        n_contacts_raw = contact_count if contact_count is not None else predictor.contact_count
        try:
            n_contacts = int(n_contacts_raw)
        except (ValueError, TypeError):
            n_contacts = 0
        
        past_illness_type = predictor._normalize_illness_type(contact_data.get('past_illness_type', 'none'))
        hiv = 1 if past_illness_type == 'hiv' else 0
        diabetes = 1 if past_illness_type == 'diabetes' else 0
        immunosuppressants = 1 if past_illness_type == 'immunosuppressants' else 0
        other_illness = 1 if past_illness_type == 'other' else 0
        # 移除重复加权：past_illness已包含在hiv/diabetes/immunosuppressants/other中
        immunosuppression_score = hiv * 2.0 + diabetes * 1.0 + immunosuppressants * 1.5 + other_illness * 0.8
        
        # 安全转换cumulative_exposure
        cumulative_exposure_raw = contact_data.get('cumulative_exposure', 0)
        try:
            cumulative_exposure = float(cumulative_exposure_raw)
        except (ValueError, TypeError):
            cumulative_exposure = 0
        
        # 安全转换time_span
        time_span_raw = contact_data.get('time_span', 4)
        try:
            time_span = int(time_span_raw)
        except (ValueError, TypeError):
            time_span = 4
        
        interaction_features = np.array([[
            age * immunosuppression_score,
            diabetes * has_tb,
            age * (1 - bcg_vaccine),
            has_symptoms * min(delay_days / 30, 1.0),
            min(cough_freq / 20, 1.0) * (n_contacts / 10),
            is_high_risk * past_illness,
            (1 - bcg_vaccine) * immunosuppression_score,
            (cumulative_exposure / 80) * (time_span / 10),
            age * diabetes
        ]])

        features = np.hstack([base_features, interaction_features])

        # 特征契约运行时强制（M 级修复：22 维契约不再靠约定维持）
        # extract_features（13 维基础）+ 交互层（9 维）必须与 ALL_FEATURE_NAMES
        # 声明的契约一致；任何一处维度漂移在此立即暴露而非静默传播到训练/部署
        expected_dim = len(getattr(predictor, 'ALL_FEATURE_NAMES', ()))
        if expected_dim and features.shape[1] != expected_dim:
            raise ValueError(
                f"特征契约违反: extract_features_with_interactions 输出 "
                f"{features.shape[1]} 维，但 ALL_FEATURE_NAMES 声明 {expected_dim} 维。"
                f"基础特征 {base_features.shape[1]} 维 + 交互特征 "
                f"{interaction_features.shape[1]} 维需与契约定义同步修改。"
            )

        return features
    

def set_patient_context(predictor, patient_ftd=0, patient_cough_freq=0, contact_count=0):
        """设置患者上下文信息，用于交互特征计算
        
        参数：
            patient_ftd (int): 患者延迟就诊天数
            patient_cough_freq (int): 患者咳嗽频率（次/小时）
            contact_count (int): 总接触者数量
        """
        predictor.patient_ftd = patient_ftd
        predictor.patient_cough_freq = patient_cough_freq
        predictor.contact_count = contact_count
    

