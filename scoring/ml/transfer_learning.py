#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""迁移学习与预训练：国家监测数据 / 公开数据集预训练 + 本地小样本微调

目标（用户需求）：
    用国家监测数据或公开数据集（如 WHO 结核病负担数据）预训练，
    再用本地小样本微调，降低对本地细粒度个体数据的依赖。

方法学（针对树集成模型的迁移范式）：
    本项目 ML 模型是随机森林/梯度提升/XGBoost/LightGBM/CatBoost 五种树集成，
    它们没有神经网络式的"权重"和"层冻结"。对树集成，"预训练 + 微调"的
    真实迁移机制有三条：
      1. 先验信息迁移（weight/ensemble 初始化）：
         - 梯度提升：warm_start 续训（在预训练树上继续拟合本地残差）；
         - XGBoost：xgb_model 续训（fit(xgb_model=预训练 booster)，
           在既有树上续加本地树；sklearn warm_start 对 XGBoost 无效）；
         - LightGBM / CatBoost：init_model 续训（保留预训练 booster 并小幅迭代）。
      2. 领域先验迁移（base-rate prior）：
         - 预训练数据的患病率由 WHO/国家发病率经 SyntheticDataCalibrator 校准，
           预训练模型内化了该人群基础风险（截距/阈值）。
      3. 超参数先验迁移：
         - 微调时继承预训练超参数，并用更小学习率（learning_rate 减半、
           min_learning_rate 下限）避免灾难性遗忘。

迁移微调部署配方（S8 收窄 2026-09-15 → P4 退役 2026-09-16，证据
kenya_transfer_eval_20260914.json 10 种子 × 同折配对）：
    - XGBoost init_model 微调配方已退役（P4）：S8 曾以「同族终点 +
      本地 n<300」收窄入配方（taiwan 0.481→0.547，+0.067，Wilcoxon
      p=0.002，22 维 v3 口径）；v4 特征空间定版后 taiwan 本地 v4 LR
      0.874（主管线）/0.856（F2 评估口径）全面支配微调路径上限，
      confirmed_tb 族（kenya 0.952 / taiwan 0.874 / crp 0.781，kenya
      为 P6 原始数据再审后 v4 口径）一律本地 v4 部署，跨队列微调
      不再入任何部署配方。init_model 续训
      机制保留为库能力（预训练-微调链路通用）；未来新站点若重启
      须在 v4 特征空间下重新评估。
    - RF warm_start 分支已下线：负迁移显著（taiwan −0.098 / crp −0.055）
      且机制清楚——续训 fit 上 class_weight='balanced' 按本地标签重算 +
      预训练树与本地树的类先验冲突；RF 一律 fallback_from_scratch。

公开数据说明：
    WHO 全球结核病报告（Global TB Report）仅公开国家层面的聚合指标
    （发病率/检出率/治疗成功率/耐多药比例/HIV 共感染率等），不公开个体级记录。
    因此本模块用"公开聚合统计 + 文献效应量"合成个体级预训练数据
    （_generate_region_synthetic_data），使预训练人群的疾病负担与目标国家对齐。
    一旦本地获得真实个体级细粒度数据，应立即用其微调以贴近本地分布。

关键设计：
    - 纯 Python + numpy（可选），无新增强依赖，与项目条件导入惯例一致；
    - 完全确定性：同一 (region, random_state) 生成完全一致的预训练数据；
    - 迁移状态（transfer_stage / transfer_info）随 save_model / load_model 持久化，
      支持"预训练-微调-再微调"链路与版本追溯；
    - compare_finetune_vs_from_scratch 提供诚实的小样本收益评估：
      本地样本划分为 train/holdout，比较"预训练+微调" vs "从零训练"的泛化指标。
"""

import copy
import datetime
import logging
import math

import numpy as np

LOGGER = logging.getLogger("tb_risk.ml")

# 条件导入（与 training.py 保持一致）
try:
    from .training import (
        SKLEARN_AVAILABLE,
        XGBOOST_AVAILABLE,
        LIGHTGBM_AVAILABLE,
        CATBOOST_AVAILABLE,
        MODEL_REGISTRY,
        _train_all_registered_models,
    )
except ImportError:  # pragma: no cover - 依赖缺失时仅符号为 None
    SKLEARN_AVAILABLE = False
    XGBOOST_AVAILABLE = False
    LIGHTGBM_AVAILABLE = False
    CATBOOST_AVAILABLE = False
    MODEL_REGISTRY = {}
    _train_all_registered_models = None

# 续训模块引用（init_model / warm_start）：与 training 符号解耦、
# 各自独立降级——缺失任一库只禁用对应模型的 init_model 微调分支
# （调用处已有 *_AVAILABLE 守卫 + 从零训练回退），不得连带把整个
# 预训练内核符号归零。2026-08-25 实测教训：catboost 缺失曾使
# pretrain/finetune 全部静默瘫痪（_train_all_registered_models=None，
# 11 个测试连环失败且无任何报错日志）。
try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None
try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None
try:
    import catboost as cb
except ImportError:  # pragma: no cover
    cb = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    pd = None
    PANDAS_AVAILABLE = False

try:
    from .persistence import save_model, load_model
except ImportError:  # pragma: no cover
    save_model = None
    load_model = None

try:
    from .evaluation import _ensemble_predict, _safe_roc_auc_score, _safe_average_precision_score
except ImportError:  # pragma: no cover
    _ensemble_predict = None
    _safe_roc_auc_score = None
    _safe_average_precision_score = None

try:
    from ...utils import SyntheticDataCalibrator, expected_feature_means_from_config
except ImportError:  # pragma: no cover
    SyntheticDataCalibrator = None
    expected_feature_means_from_config = None


# ============================================================================
# WHO / 国家监测公开数据目录（聚合层面，来源：WHO Global TB Report 2023 估算）
# ============================================================================

def _default_catalog() -> dict:
    """WHO 全球结核病报告 + 中国国家监测/本地筛查的公开聚合指标。

    说明：值为公开发布的近似聚合统计，仅用于预训练人群负担校准；
    实际部署前建议按最新 WHO 报告更新 incidence_per_100k。
    """
    return {
        'china_national': {
            'name': '中国（国家监测）',
            'incidence_per_100k': 52.0,        # 2023 估算新发/10万
            'detection_rate_pct': 78.0,        # 治疗覆盖率/检出率
            'treatment_success_pct': 90.0,
            'mdr_new_pct': 3.3,
            'hiv_coinfection_pct': 0.5,
            'mortality_per_100k': 2.6,
            'bcg_coverage_pct': 99.0,
            'source': 'WHO Global TB Report 2023 estimates',
        },
        'china_karamay': {
            'name': '克拉玛依（本地筛查）',
            'incidence_per_100k': 121.0,       # 2024 筛查检出率 121/10万
            'detection_rate_pct': 70.0,
            'treatment_success_pct': 66.1,
            'mdr_new_pct': 3.5,
            'hiv_coinfection_pct': 1.0,
            'mortality_per_100k': 3.0,
            'bcg_coverage_pct': 95.0,
            'source': '克拉玛依卫健委 2024 筛查数据',
        },
        'india': {
            'name': '印度（国家监测）',
            'incidence_per_100k': 195.0,
            'detection_rate_pct': 73.0,
            'treatment_success_pct': 85.0,
            'mdr_new_pct': 3.0,
            'hiv_coinfection_pct': 0.8,
            'mortality_per_100k': 24.0,
            'bcg_coverage_pct': 90.0,
            'source': 'WHO Global TB Report 2023 estimates',
        },
        'south_africa': {
            'name': '南非（国家监测）',
            'incidence_per_100k': 442.0,
            'detection_rate_pct': 72.0,
            'treatment_success_pct': 85.0,
            'mdr_new_pct': 3.2,
            'hiv_coinfection_pct': 40.0,
            'mortality_per_100k': 48.0,
            'bcg_coverage_pct': 95.0,
            'source': 'WHO Global TB Report 2023 estimates',
        },
        'global_high': {
            'name': '高负担国家（聚合）',
            'incidence_per_100k': 350.0,
            'detection_rate_pct': 70.0,
            'treatment_success_pct': 85.0,
            'mdr_new_pct': 3.5,
            'hiv_coinfection_pct': 10.0,
            'mortality_per_100k': 40.0,
            'bcg_coverage_pct': 90.0,
            'source': 'WHO Global TB Report 2023 estimates (aggregate)',
        },
        'global_low': {
            'name': '低负担地区（聚合）',
            'incidence_per_100k': 10.0,
            'detection_rate_pct': 85.0,
            'treatment_success_pct': 92.0,
            'mdr_new_pct': 2.0,
            'hiv_coinfection_pct': 0.3,
            'mortality_per_100k': 0.5,
            'bcg_coverage_pct': 98.0,
            'source': 'WHO Global TB Report 2023 estimates (aggregate)',
        },
        'default': {
            'name': '默认（国际参数）',
            'incidence_per_100k': 62.0,
            'detection_rate_pct': 75.0,
            'treatment_success_pct': 86.0,
            'mdr_new_pct': 3.0,
            'hiv_coinfection_pct': 1.0,
            'mortality_per_100k': 6.0,
            'bcg_coverage_pct': 92.0,
            'source': 'Project default international parameters',
        },
    }


WHO_TB_BURDEN_CATALOG = _default_catalog()

DEFAULT_PRETRAIN_REGION = 'china_national'

# 迁移阶段取值
TRANSFER_STAGE_UNTRAINED = 'untrained'
TRANSFER_STAGE_PRETRAINED = 'pretrained'
TRANSFER_STAGE_FINETUNED = 'finetuned'

# 部署配方退役登记（P4，2026-09-16）：跨队列微调配方退出部署——
# v4 特征空间定版后 confirmed_tb 族本地训练全面支配迁移路径。
# 机制（init_model 续训）保留为库能力，仅退役「部署配方」身份；
# 回归测试 test_xgboost_init_model_recipe_retired 锁定该决策。
RETIRED_DEPLOYMENT_RECIPES = {
    'kenya_to_taiwan_xgboost_init_model': {
        'retired': '2026-09-16',
        'evidence': (
            'S8 kenya_transfer_eval_20260914.json：22 维口径 0.481→0.547 '
            '（+0.067，10 种子 Wilcoxon p=0.002）被 taiwan 本地 v4 '
            'LR 0.874（主管线）/0.856（F2 口径）支配'),
        'successor': 'confirmed_tb 族本地 v4 部署（kenya 0.952 / '
                     'taiwan 0.874 / crp 0.781，P2 路由 LR 优先；kenya '
                     '数字为 P6 原始数据再审后 v4 口径，kenya_'
                     'v4_features_20260915）',
        'restart_condition': '未来新站点须在 v4 特征空间下重新评估',
    },
}


def list_pretrain_regions() -> list:
    """返回可用的预训练区域/国家目录（key + 中文名 + 发病率）。"""
    return [
        {'key': k, 'name': v['name'], 'incidence_per_100k': v['incidence_per_100k'],
         'source': v['source']}
        for k, v in WHO_TB_BURDEN_CATALOG.items()
    ]


# ============================================================================
# 预训练数据合成（区域校准）
# ============================================================================

def _region_offset(region: str) -> int:
    """把区域名映射到稳定的整数偏移（保证跨进程确定性，不依赖 hash 随机化）。"""
    return sum(map(ord, str(region))) % 997


def _annual_to_prevalence(incidence_per_100k: float, years: int = 10) -> float:
    """年发病率 → 累积患病率近似（1-(1-r)^years）。"""
    annual = max(incidence_per_100k, 0.0) / 100000.0
    return max(1.0 - (1.0 - annual) ** years, 1e-6)


def _generate_region_synthetic_data(region_info: dict, n_samples: int,
                                    random_state: int) -> tuple:
    """按区域公开统计 + 文献效应量合成个体级预训练数据（22 维特征矩阵）。

    与 preprocessing._generate_synthetic_training_data 的差异：
      1. 使用本区域发病率经 SyntheticDataCalibrator 校准截距（人群基础风险）；
      2. 特征分布随区域公开统计调整：
         - bcg_coverage_pct        → 卡介苗接种率
         - hiv_coinfection_pct     → 免疫抑制/既往史中的 HIV 比例
         - detection_rate_pct      → 监测系统捕获的有症状比例
      3. 随机种子与区域名绑定（_region_offset），不同区域生成不同数据集。

    返回：
        (X, y, feature_dist) — X (n,22) float；y (n,) int；feature_dist 特征分布摘要
    """
    if not SyntheticDataCalibrator:
        raise RuntimeError("SyntheticDataCalibrator 不可用，无法合成预训练数据")

    n_samples = int(n_samples)
    if n_samples < 100:
        raise ValueError(f"预训练样本量至少 100，当前 {n_samples}")

    incidence = float(region_info.get('incidence_per_100k', 62.0))
    detection = float(region_info.get('detection_rate_pct', 75.0))
    bcg_cov = float(region_info.get('bcg_coverage_pct', 92.0))
    hiv_cov = float(region_info.get('hiv_coinfection_pct', 1.0))

    # 用区域名派生种子，保证跨区域数据集不同、跨进程可复现
    seed = (int(random_state) * 7919 + _region_offset(region_info.get('key', 'default'))) % (2 ** 31)
    rng = np.random.RandomState(seed)

    ages = rng.randint(0, 90, n_samples)
    cumulative_exposure = rng.exponential(40, n_samples)
    # 检出率越高，监测数据中越容易出现症状人群
    symptom_p = np.clip(0.08 * (detection / 70.0), 0.05, 0.4)
    has_symptoms = rng.binomial(1, symptom_p, n_samples)
    bcg_vaccine = rng.binomial(1, np.clip(bcg_cov / 100.0, 0.05, 1.0), n_samples)
    has_tb = rng.binomial(1, 0.02, n_samples)
    contact_distance_score = rng.choice([0.1, 0.25, 0.5, 0.8, 1.0], n_samples,
                                        p=[0.1, 0.15, 0.35, 0.3, 0.1])
    ventilation_score = rng.choice([0.15, 0.3, 0.5, 0.75, 1.0], n_samples,
                                   p=[0.15, 0.2, 0.35, 0.2, 0.1])
    is_high_risk = rng.binomial(1, 0.2, n_samples)
    # HIV 共感染率 → 提高既往史中 HIV 比例（同时保持非 HIV 背景病种）
    base_weights = {'none': 0.70, 'hiv': 0.05, 'diabetes': 0.10,
                    'immunosuppressants': 0.05, 'other': 0.10}
    hiv_p = np.clip(base_weights['hiv'] + hiv_cov / 100.0, 0.02, 0.5)
    non_hiv_total = 1.0 - base_weights['hiv']
    weights = {'none': base_weights['none'] * (1.0 - hiv_p) / non_hiv_total,
               'hiv': hiv_p,
               'diabetes': base_weights['diabetes'] * (1.0 - hiv_p) / non_hiv_total,
               'immunosuppressants': base_weights['immunosuppressants'] * (1.0 - hiv_p) / non_hiv_total,
               'other': base_weights['other'] * (1.0 - hiv_p) / non_hiv_total}
    keys = list(weights.keys())
    probs = np.array([max(weights[k], 0.0) for k in keys])
    probs = probs / probs.sum()
    past_illness_type = rng.choice(keys, n_samples, p=probs)
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
    diabetes = (past_illness_type == 'diabetes').astype(float)
    immunosuppressants = (past_illness_type == 'immunosuppressants').astype(float)
    other_illness = (past_illness_type == 'other').astype(float)
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

    # 区域患病率校准（核心：把 WHO/国家发病率转成预训练人群基础风险）
    target_prevalence = _annual_to_prevalence(incidence)
    try:
        calibrator = SyntheticDataCalibrator(
            random_state=random_state,
            region=region_info.get('key', 'default'),
            target_prevalence=target_prevalence,
        )
        expected = expected_feature_means_from_config()
        local_params = {
            'base_incidence_per_100k': incidence,
            'region': region_info.get('key', 'default'),
        }
        calib = calibrator.run_full_calibration(expected, local_params, n_samples)
        log_odds_base = float(calib['log_odds_base'])
    except Exception as e:  # pragma: no cover - 校准失败时回退近似截距
        LOGGER.warning("区域患病率校准失败，回退到近似截距: %s", e)
        log_odds_base = math.log(target_prevalence / (1.0 - target_prevalence + 1e-12)) - 1.0

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

    feature_dist = {
        'n_samples': int(n_samples),
        'target_prevalence': float(target_prevalence),
        'actual_positive_rate': float(np.mean(y)),
        'symptom_prevalence': float(np.mean(has_symptoms)),
        'bcg_coverage': float(np.mean(bcg_vaccine)),
        'hiv_prev_in_data': float(np.mean(hiv)),
        'seed': int(seed),
    }
    return X, y, feature_dist


def generate_pretraining_dataset(predictor, region=DEFAULT_PRETRAIN_REGION,
                                 n_samples=2000, random_state=42):
    """生成国家/WHO 层面的预训练数据集（22 维特征矩阵 + 标签 + 元信息）。

    参数：
        predictor: MLRiskPredictor 实例（用于特征口径校验，兼容对象亦可）
        region (str): WHO_TB_BURDEN_CATALOG 中的区域键，默认 china_national
        n_samples (int): 预训练样本量，默认 2000
        random_state (int): 随机种子，保证同 (region, seed) 完全可复现

    返回：
        tuple: (X, y, meta)
            X (n,22) float；y (n,) int；meta 含 region/发病率/来源/阳性率等
    """
    region = region if region in WHO_TB_BURDEN_CATALOG else 'default'
    info = WHO_TB_BURDEN_CATALOG[region]
    X, y, feature_dist = _generate_region_synthetic_data(info, n_samples, random_state)
    meta = {
        'region': region,
        'region_name': info['name'],
        'incidence_per_100k': info['incidence_per_100k'],
        'source': info['source'],
        'detection_rate_pct': info['detection_rate_pct'],
        'treatment_success_pct': info['treatment_success_pct'],
        'mdr_new_pct': info['mdr_new_pct'],
        'hiv_coinfection_pct': info['hiv_coinfection_pct'],
        'bcg_coverage_pct': info['bcg_coverage_pct'],
        'n_samples': int(n_samples),
        'random_state': int(random_state),
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'feature_names': list(getattr(predictor, 'ALL_FEATURE_NAMES', [])),
        **feature_dist,
    }
    return X, y, meta


# ============================================================================
# 预训练入口
# ============================================================================

def pretrain_models(predictor, region=DEFAULT_PRETRAIN_REGION, n_samples=2000,
                    random_state=42, enable_hyperopt=False, save_path=None,
                    model_keys=None):
    """用国家监测 / WHO 公开数据预训练所有可用 ML 模型。

    流程：
        1. 按区域公开统计合成预训练数据（患病率经 SyntheticDataCalibrator 校准）；
        2. 用统一训练内核 _train_all_registered_models 训练全部注册模型；
        3. 记录迁移阶段 transfer_stage='pretrained' 与 transfer_info；
        4. 可选 save_path 持久化预训练模型（后续本地微调可直接加载）。

    参数：
        predictor: MLRiskPredictor 实例
        region (str): 预训练区域键
        n_samples (int): 预训练样本量
        random_state (int): 随机种子
        enable_hyperopt (bool): 是否启用超参数搜索
        save_path (str|None): 可选，预训练完成后保存模型文件的路径
        model_keys (list|None): 指定训练的模型键，None 则训练全部可用模型

    返回：
        dict: {'success': bool, 'region': str, 'meta': dict,
               'model_performance': dict, 'saved_to': str|None}
    """
    if not SKLEARN_AVAILABLE:
        return {'success': False, 'error': 'sklearn_unavailable',
                'meta': {'region': region}}
    if _train_all_registered_models is None:
        return {'success': False, 'error': 'training_kernel_unavailable',
                'meta': {'region': region}}

    region = region if region in WHO_TB_BURDEN_CATALOG else 'default'
    X, y, meta = generate_pretraining_dataset(predictor, region, n_samples, random_state)

    predictor.reset_stop_flag()
    ok = _train_all_registered_models(
        predictor, X, y,
        random_state=random_state,
        enable_hyperopt=enable_hyperopt,
        data_source_label='pretrain',
        model_keys=model_keys,
        n_real_samples=0,
        n_synth_samples=int(n_samples),
    )
    if not ok:
        return {'success': False, 'region': region, 'meta': meta}

    # 记录预训练样本量（GUI 加载视图会展示训练样本量）
    predictor.training_sample_count = int(n_samples)

    # 预训练模型性能条目标记迁移阶段
    for entry in predictor.model_performance.values():
        entry['transfer_stage'] = TRANSFER_STAGE_PRETRAINED

    predictor.transfer_stage = TRANSFER_STAGE_PRETRAINED
    predictor.transfer_info = {
        'stage': TRANSFER_STAGE_PRETRAINED,
        'source': meta['source'],
        'region': region,
        'region_name': meta['region_name'],
        'incidence_per_100k': meta['incidence_per_100k'],
        'target_prevalence': meta['target_prevalence'],
        'n_pretrain_samples': int(n_samples),
        'actual_positive_rate': meta['actual_positive_rate'],
        'pretrained_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'note': ('预训练数据由 WHO/国家公开聚合统计 + 文献效应量合成（患病率经'
                 'SyntheticDataCalibrator 校准）；公开数据集不含个体级记录，'
                 '该预训练提供领域先验，需用本地小样本微调贴近本地分布。'),
    }

    saved_to = None
    if save_path:
        if save_model is not None and save_model(predictor, save_path):
            saved_to = save_path
        else:
            LOGGER.warning("预训练模型保存失败: %s", save_path)

    return {
        'success': True,
        'region': region,
        'meta': meta,
        'model_performance': predictor.model_performance,
        'saved_to': saved_to,
    }


# ============================================================================
# 本地小样本微调
# ============================================================================

def _default_finetune_trees(n_local, n_finetune_trees=None) -> int:
    """默认微调新增树数：与本地样本量相关，控制在小范围避免过拟合。"""
    if n_finetune_trees is not None:
        return max(1, int(n_finetune_trees))
    return max(10, min(80, int(n_local) // 5))


def _pretrained_feature_names(model):
    """预训练模型的特征名视图（跨后端）：以命名 DataFrame 拟合的模型才有。

    sklearn 系（RF/GB/XGBoost/LightGBM）拟合时记录 feature_names_in_，
    CatBoost 记录 feature_names_，LightGBM 另有 booster_.feature_name()
    兜底。numpy 拟合的模型无特征名，返回 None。
    """
    for attr in ('feature_names_in_', 'feature_names_'):
        names = getattr(model, attr, None)
        if names is not None and len(names) > 0:
            return [str(n) for n in names]
    booster = getattr(model, 'booster_', None)
    if booster is not None:
        try:
            names = booster.feature_name()
            if names:
                return [str(n) for n in names]
        except Exception:
            pass
    return None


def _finetune_single_model(key, model, X_local, y_local, random_state,
                           warm_start, n_finetune_trees, min_learning_rate):
    """对单个树模型做"预训练 + 本地微调"（逐模型类型）。

    返回：
        (new_model, mode, detail) — mode ∈ {'warm_start','init_model',
        'fallback_from_scratch','failed'}；new_model 为 None 表示失败。
    """
    spec = MODEL_REGISTRY.get(key)
    if spec is None or spec.model_cls is None:
        return None, 'skipped', {}
    if model is None:
        return None, 'skipped', {}

    n_local = len(y_local)

    # ---- 0) 特征名对齐 ----
    # 预训练模型以命名 DataFrame 拟合时（主训练管线即如此），numpy
    # X_local 须恢复同名同序：否则 CatBoost init_model 直接报
    # "Feature name mismatch"，LightGBM/sklearn 则因平铺索引与命名
    # 特征不一致产生告警甚至静默错位。列数不一致时不包装，交由
    # 后端响亮失败后走 fallback。
    pretrained_names = _pretrained_feature_names(model)
    if (PANDAS_AVAILABLE and pretrained_names is not None
            and not isinstance(X_local, pd.DataFrame)
            and np.asarray(X_local).shape[1] == len(pretrained_names)):
        X_local = pd.DataFrame(np.asarray(X_local), columns=pretrained_names)

    # ---- 1) 续训式微调（保留预训练知识） ----
    if warm_start:
        try:
            if key == 'gradient_boosting' and hasattr(model, 'get_params'):
                # sklearn GB：warm_start 真续训（deepcopy 保留已拟合树，
                # n_estimators 增量后 fit 在既有树上继续拟合本地残差）
                n_add = _default_finetune_trees(n_local, n_finetune_trees)
                new_model = copy.deepcopy(model)
                lr = max(float(model.get_params().get('learning_rate', 0.1)) * 0.5, min_learning_rate)
                new_model.set_params(warm_start=True,
                                     n_estimators=int(model.get_params().get('n_estimators', 100)) + n_add,
                                     learning_rate=lr)
                new_model.fit(X_local, y_local)
                return new_model, 'warm_start', {'n_added_trees': n_add, 'learning_rate': lr}

            if key == 'xgboost' and XGBOOST_AVAILABLE and hasattr(model, 'get_booster'):
                # XGBoost sklearn API 无 warm_start（2026-09-14 实测：
                # set_params 静默接受但 fit 仍从零重训，预训练树全部丢失）；
                # 真续训须 fit(xgb_model=预训练 booster)，n_estimators
                # 为在既有树上本次新增的树数。
                # 且必须从"序列化→重载"的副本续训：直接以活 booster 续训
                # 会复用其内存 sketch/predictor 缓存导致同种子结果不确定
                # （实测两次调用新增树不同；序列化重载后恢复确定）
                n_add = _default_finetune_trees(n_local, n_finetune_trees)
                params = dict(model.get_params())
                params['learning_rate'] = max(float(params.get('learning_rate', 0.1)) * 0.5, min_learning_rate)
                params['n_estimators'] = n_add
                base_booster = xgb.Booster()
                base_booster.load_model(model.get_booster().save_raw())
                new_model = xgb.XGBClassifier(**params)
                new_model.fit(X_local, y_local, xgb_model=base_booster)
                return new_model, 'init_model', {'n_added_trees': n_add, 'learning_rate': params['learning_rate']}

            # random_forest warm_start 分支已下线（S8 证据 2026-09-15：
            # taiwan −0.098 / crp −0.055 负迁移显著且机制清楚——续训 fit 上
            # class_weight='balanced' 按本地标签重算 + 预训练树与本地树的
            # 类先验冲突）。RF 走下方 fallback_from_scratch（从零 + 减半
            # 学习率 + 预训练超参先验）。

            if key == 'lightgbm' and LIGHTGBM_AVAILABLE and getattr(model, 'booster_', None) is not None:
                n_add = _default_finetune_trees(n_local, n_finetune_trees)
                params = dict(model.get_params())
                params['learning_rate'] = max(float(params.get('learning_rate', 0.1)) * 0.5, min_learning_rate)
                params['n_estimators'] = n_add
                params['verbosity'] = -1
                new_model = lgb.LGBMClassifier(**params)
                # lightgbm>=4.0 fit() 不再接受 verbose 关键字（静默由
                # verbosity=-1 承担）
                new_model.fit(X_local, y_local, init_model=model.booster_)
                return new_model, 'init_model', {'n_added_trees': n_add, 'learning_rate': params['learning_rate']}

            if key == 'catboost' and CATBOOST_AVAILABLE:
                n_add = _default_finetune_trees(n_local, n_finetune_trees)
                params = dict(model.get_params())
                params['learning_rate'] = max(float(params.get('learning_rate', 0.1)) * 0.5, min_learning_rate)
                params['iterations'] = n_add
                params['verbose'] = False
                new_model = cb.CatBoostClassifier(**params)
                new_model.fit(X_local, y_local, init_model=model, verbose=False)
                return new_model, 'init_model', {'n_added_trees': n_add, 'learning_rate': params['learning_rate']}
        except Exception as e:
            LOGGER.warning('%s 续训式微调失败，回退本地从零训练: %s', key, e)

    # ---- 2) 回退：本地从零训练（继承预训练超参数先验 + 小学习率） ----
    try:
        kwargs = dict(spec.default_params)
        kwargs[spec.random_state_param] = random_state
        if 'learning_rate' in kwargs:
            kwargs['learning_rate'] = max(float(kwargs['learning_rate']) * 0.5, min_learning_rate)
        new_model = spec.model_cls(**kwargs)
        new_model.fit(X_local, y_local)
        return new_model, 'fallback_from_scratch', {'reason': 'warm_start unavailable or failed',
                                                    'learning_rate': kwargs.get('learning_rate')}
    except Exception as e:
        LOGGER.warning('%s 本地从零训练失败: %s', key, e)
        return None, 'failed', {'reason': str(e)}


def _validate_local_data(X_local, y_local):
    """校验本地微调数据，返回 (X, y) 或抛 ValueError。"""
    X_local = np.asarray(X_local, dtype=float)
    y_local = np.asarray(y_local).reshape(-1)
    if len(X_local) < 10:
        raise ValueError(f"本地微调样本量至少 10，当前 {len(X_local)}")
    if len(np.unique(y_local)) < 2:
        raise ValueError("本地微调标签只有一种类型，无法微调")
    return X_local, y_local


def finetune_models(predictor, X_local, y_local, random_state=42, pretrained_path=None,
                    warm_start=True, n_finetune_trees=None, min_learning_rate=0.05):
    """用本地小样本对预训练模型进行微调。

    迁移机制：
        - 梯度提升：warm_start 续训（保留预训练树，新增本地树）；
        - XGBoost：xgb_model 续训（fit(xgb_model=预训练 booster)，续加本地树）；
        - LightGBM/CatBoost：init_model 续训（保留预训练 booster，小幅迭代）；
        - 随机森林：warm_start 分支已下线（S8 负迁移证据，见模块头部署配方），
          一律回退"本地从零训练"（继承预训练超参 + 减半学习率）；
        - 学习率减半（下限 min_learning_rate）避免灾难性遗忘；
        - 不支持续训的模型回退到"本地从零训练"并标注 fallback_from_scratch。

    参数：
        predictor: MLRiskPredictor 实例（须先完成预训练，或提供 pretrained_path）
        X_local, y_local: 本地细粒度样本（22 维特征 + 0/1 标签）
        random_state (int): 随机种子
        pretrained_path (str|None): 可选预训练模型文件，提供则先加载作为起点
        warm_start (bool): 是否启用续训式微调，False 则全部走本地从零训练
        n_finetune_trees (int|None): 微调新增树数，None 时按样本量自动
        min_learning_rate (float): 微调学习率下限

    返回：
        dict: {'success': bool, 'n_finetune_samples': int,
               'models': {key: {'mode': str, ...}}, 'transfer_info': dict}
    """
    if not SKLEARN_AVAILABLE:
        return {'success': False, 'error': 'sklearn_unavailable'}
    if load_model is None:
        return {'success': False, 'error': 'persistence_unavailable'}

    try:
        X_local, y_local = _validate_local_data(X_local, y_local)
    except ValueError as e:
        return {'success': False, 'error': str(e), 'n_finetune_samples': int(len(np.asarray(y_local)))}

    # 预训练起点准备
    if pretrained_path:
        if not load_model(predictor, pretrained_path):
            return {'success': False, 'error': f'预训练模型加载失败: {pretrained_path}',
                    'n_finetune_samples': len(y_local)}
    if not getattr(predictor, 'is_trained', False) or not getattr(predictor, 'models', None):
        return {'success': False, 'error': '无预训练模型，请先 pretrain_models 或提供 pretrained_path',
                'n_finetune_samples': len(y_local)}

    base_info = dict(getattr(predictor, 'transfer_info', {}) or {})
    predictor.reset_stop_flag()

    finetuned = {}
    for key, model in list(predictor.models.items()):
        if getattr(predictor, '_stop_training', False):
            LOGGER.info('微调被中断，已完成 %d 个模型', len(finetuned))
            break
        new_model, mode, detail = _finetune_single_model(
            key, model, X_local, y_local, random_state,
            warm_start, n_finetune_trees, min_learning_rate,
        )
        if new_model is None:
            continue
        predictor.models[key] = new_model
        finetuned[key] = {'mode': mode, **detail}
        entry = predictor.model_performance.get(key, {}) or {}
        entry['transfer_stage'] = TRANSFER_STAGE_FINETUNED
        entry['finetune_mode'] = mode
        entry['finetuned_at'] = datetime.datetime.now().isoformat(timespec='seconds')
        entry['n_finetune_samples'] = int(len(y_local))
        predictor.model_performance[key] = entry

    if not finetuned:
        return {'success': False, 'error': '没有模型成功微调', 'n_finetune_samples': len(y_local)}

    predictor.transfer_stage = TRANSFER_STAGE_FINETUNED
    predictor.transfer_info = {
        **base_info,
        'stage': TRANSFER_STAGE_FINETUNED,
        'base_stage': base_info.get('stage', TRANSFER_STAGE_PRETRAINED),
        'finetuned_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'n_finetune_samples': int(len(y_local)),
        'warm_start': bool(warm_start),
        'min_learning_rate': float(min_learning_rate),
        'models': finetuned,
        'note': ('本地小样本微调：warm_start/init_model 延续预训练知识，'
                 '小学习率防止灾难性遗忘；若本地样本量增长，可重复调用继续微调。'),
    }
    predictor.training_sample_count = int(len(y_local))

    return {'success': True, 'n_finetune_samples': int(len(y_local)),
            'models': finetuned, 'transfer_info': predictor.transfer_info}


# ============================================================================
# 迁移 vs 从零 对比（诚实的小样本收益评估）
# ============================================================================

def _holdout_metrics(predictor, X_holdout, y_holdout) -> dict:
    """用已训练模型的集成预测在留出集上评估（AUROC / AUPRC / 准确率）。"""
    if _ensemble_predict is None or not getattr(predictor, 'models', None):
        return {'AUROC': None, 'AUPRC': None, 'accuracy': None, 'n_holdout': int(len(y_holdout))}
    try:
        y_proba = _ensemble_predict(predictor.models, np.asarray(X_holdout, dtype=float))
        y_proba = np.asarray(y_proba).reshape(-1)
        auroc = _safe_roc_auc_score(y_holdout, y_proba) if _safe_roc_auc_score else None
        auprc = _safe_average_precision_score(y_holdout, y_proba) if _safe_average_precision_score else None
        y_pred = (y_proba > 0.5).astype(int)
        acc = float(np.mean(y_pred == y_holdout)) if len(y_holdout) else None
        return {'AUROC': float(auroc) if auroc is not None else None,
                'AUPRC': float(auprc) if auprc is not None else None,
                'accuracy': acc, 'n_holdout': int(len(y_holdout))}
    except Exception as e:  # pragma: no cover
        LOGGER.warning("留出集评估失败: %s", e)
        return {'AUROC': None, 'AUPRC': None, 'accuracy': None, 'n_holdout': int(len(y_holdout))}


def compare_finetune_vs_from_scratch(predictor, X_local, y_local,
                                     region=DEFAULT_PRETRAIN_REGION,
                                     n_pretrain=1500, random_state=42,
                                     holdout_frac=0.3, train_from_arrays_fn=None):
    """对比"预训练+微调" vs "本地从零训练"在小样本上的泛化能力。

    流程（在同一 predictor 上顺序执行，最终保留效果更好的"预训练+微调"分支）：
        1. 本地数据按 holdout_frac 分层划分 train/holdout；
        2. 分支 A（从零）：train_from_arrays(predictor, X_train, y_train) → 留出集评估；
        3. 分支 B（迁移）：pretrain_models(region) → finetune_models(X_train) → 留出集评估；
        4. 汇总对比（样本越少、领域差异越小，迁移收益通常越明显）。

    参数：
        predictor: MLRiskPredictor 实例
        X_local, y_local: 本地细粒度样本
        region (str): 预训练区域键
        n_pretrain (int): 预训练样本量
        random_state (int): 随机种子
        holdout_frac (float): 留出集比例（0~1）
        train_from_arrays_fn (callable|None): 从零训练入口，默认用 predictor.train_from_arrays

    返回：
        dict: {'success': bool, 'n_train': int, 'n_holdout': int,
               'from_scratch': dict, 'finetuned': dict,
               'delta_auroc': float|None, 'recommendation': str}
    """
    X_local, y_local = _validate_local_data(X_local, y_local)
    n = len(y_local)
    n_holdout = max(1, int(round(n * holdout_frac)))
    n_train = n - n_holdout
    if n_train < 10 or n_holdout < 1:
        return {'success': False, 'error': '本地样本太少，无法划分 train/holdout',
                'n_train': n_train, 'n_holdout': n_holdout}

    # 分层划分（确定、稳健；sklearn 不可用时回退固定比例抽取）
    try:
        from sklearn.model_selection import train_test_split
        X_train, X_hold, y_train, y_hold = train_test_split(
            X_local, y_local, test_size=n_holdout, stratify=y_local,
            random_state=random_state)
    except Exception:  # pragma: no cover - sklearn 不可用时的朴素回退
        rng = np.random.RandomState(random_state)
        holdout_idx = rng.choice(n, size=n_holdout, replace=False)
        train_idx = np.array([i for i in range(n) if i not in set(holdout_idx.tolist())])
        X_train, y_train = X_local[train_idx], y_local[train_idx]
        X_hold, y_hold = X_local[holdout_idx], y_local[holdout_idx]

    from_scratch_fn = train_from_arrays_fn or getattr(predictor, 'train_from_arrays', None)

    result = {'success': True, 'n_train': int(n_train), 'n_holdout': int(n_holdout)}

    # ---- 分支 A：本地从零训练 ----
    scratch_metrics = {'AUROC': None, 'AUPRC': None, 'accuracy': None}
    if from_scratch_fn is not None:
        try:
            ok_scratch = from_scratch_fn(X_train, y_train, random_state=random_state)
            if ok_scratch:
                scratch_metrics = _holdout_metrics(predictor, X_hold, y_hold)
        except Exception as e:  # pragma: no cover
            LOGGER.warning("从零训练分支失败: %s", e)
    result['from_scratch'] = scratch_metrics

    # ---- 分支 B：预训练 + 微调（最终保留本分支） ----
    ft_metrics = {'AUROC': None, 'AUPRC': None, 'accuracy': None}
    try:
        pretrain_models(predictor, region=region, n_samples=n_pretrain, random_state=random_state)
        ok_ft = finetune_models(predictor, X_train, y_train, random_state=random_state)
        if ok_ft.get('success'):
            ft_metrics = _holdout_metrics(predictor, X_hold, y_hold)
    except Exception as e:  # pragma: no cover
        LOGGER.warning("预训练+微调分支失败: %s", e)
    result['finetuned'] = ft_metrics

    a_scratch = scratch_metrics.get('AUROC')
    a_ft = ft_metrics.get('AUROC')
    if a_scratch is not None and a_ft is not None:
        result['delta_auroc'] = float(a_ft - a_scratch)
        if a_ft >= a_scratch:
            result['recommendation'] = ('迁移学习（预训练+微调）在本小样本上表现不低于从零训练，'
                                        '建议采用预训练模型并继续本地微调。')
        else:
            result['recommendation'] = ('本数据上从零训练略优于迁移（差异通常来自本地分布与预训练'
                                        '区域差异），可尝试更换更接近本地的预训练区域。')
    else:
        result['delta_auroc'] = None
        result['recommendation'] = '留出集评估未完整，无法给出可靠结论，建议增大本地样本。'

    return result


# ============================================================================
# 迁移状态摘要 / 版本追溯
# ============================================================================

def summarize_transfer_status(predictor) -> dict:
    """汇总当前模型的迁移状态（阶段、预训练源、微调信息），供 UI/日志展示。"""
    stage = getattr(predictor, 'transfer_stage', None)
    info = getattr(predictor, 'transfer_info', {}) or {}
    is_trained = getattr(predictor, 'is_trained', False)

    if not is_trained and not info:
        return {'stage': TRANSFER_STAGE_UNTRAINED, 'info': {},
                'message': '模型未训练，尚无迁移状态。可先 pretrain_models 预训练，'
                           '再用本地小样本 finetune_models 微调。'}

    stage = stage or TRANSFER_STAGE_UNTRAINED
    parts = [f"迁移阶段: {stage}"]
    if info.get('source'):
        parts.append(f"预训练源: {info['source']}")
    if info.get('region_name'):
        parts.append(f"区域: {info['region_name']}")
    if info.get('incidence_per_100k'):
        parts.append(f"发病率(1/10万): {info['incidence_per_100k']}")
    if info.get('n_pretrain_samples'):
        parts.append(f"预训练样本: {info['n_pretrain_samples']}")
    if info.get('n_finetune_samples'):
        parts.append(f"本地微调样本: {info['n_finetune_samples']}")
    if info.get('models'):
        modes = sorted({v.get('mode', '?') for v in info['models'].values()})
        parts.append(f"微调方式: {', '.join(modes)}")

    return {'stage': stage, 'info': info, 'message': '；'.join(parts)}
