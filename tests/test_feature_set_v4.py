#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1（2026-09-16）：v4 族感知特征空间接入主训练管线。

覆盖：
1. cohort_features 单元：detect_cohort 列签名探测 / build_v4_extras 哨兵
   清洗 + 缺失指示器 + 中位数填补（fill_medians 落 audit）/ 部署端单样本
   构造 build_v4_extras_for_prediction 的列集对齐语义；
2. train_from_real_data feature_set='v4' 全链路（brazil 小队列冒烟）：
   模型拟合于 22+extras 列、三件套挂载（v4_cohort / v4_feature_names /
   v4_fill_medians）、result['feature_space'] 披露、entry 记录 v4_cohort、
   map_columns 不改写 extras 源列（ethnic 等契约列名保护）；
3. 无签名合成数据（BASE13 无 extras 源列——P6 前的 kenya 型）v4 请求
   → 回退 v1 + 披露（不冒认）；真 kenya CSV 现由签名列命中（P6）；
4. 预测端：predict_risk 对 v4 模型从 contact_data 原始字段构造 extras
   （缺失字段 → 训练中位数 + 指示器），持久化 round-trip 后仍可用；
5. 防回归：_model_input_frame 对 v4 extras 不静默补零（v3 网络列语义
   保留）；_check_feature_contract 对 26 维 v4（brazil 22+4）不误触
   v3 契约告警。

依据：cohort_v4_features_20260915（LR 配对增益 brazil +0.109 等，
全部 bootstrap CI 排除 0）；spec 单一真值源 ml/cohort_features.py。
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tb_risk.scoring.predictor import MLRiskPredictor  # noqa: E402
from tb_risk.scoring.ml import cohort_features as cf  # noqa: E402
from tb_risk.scoring.ml.training import train_from_real_data  # noqa: E402

ALL_FEATURE_NAMES = list(MLRiskPredictor.ALL_FEATURE_NAMES)
BASE13 = [
    'age', 'cumulative_exposure', 'has_symptoms', 'bcg_vaccine',
    'has_tb', 'contact_distance_score', 'ventilation_score',
    'is_high_risk', 'past_illness', 'exposure_setting_score',
    'single_duration', 'freq_density', 'time_span',
]


# ============================================================================
# 测试数据构造
# ============================================================================

def _brazil_frame(n=400, seed=42):
    """brazil 型合成数据：13 维基础 + ifng_genotype/ethnic 签名 + 目标。

    信号置于 extras（ifng_genotype TT 风险升高）——v4 的意义即让注册表
    模型吃到 extras 信息。
    """
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        'age': rng.randint(15, 85, n),
        'cumulative_exposure': rng.uniform(0, 800, n),
        'has_symptoms': rng.randint(0, 2, n),
        'bcg_vaccine': rng.randint(0, 2, n),
        'has_tb': rng.randint(0, 2, n),
        'contact_distance_score': rng.uniform(0, 10, n),
        'ventilation_score': rng.uniform(0, 10, n),
        'is_high_risk': rng.randint(0, 2, n),
        'past_illness': rng.randint(0, 2, n),
        'exposure_setting_score': rng.uniform(0, 10, n),
        'single_duration': rng.uniform(10, 300, n),
        'freq_density': rng.uniform(1, 20, n),
        'time_span': rng.uniform(1, 52, n),
        'gender': rng.choice(['m', 'f'], n),
        'ifng_genotype': rng.choice(['AA', 'AT', 'TT'], n, p=[.5, .3, .2]),
        'ethnic': rng.choice(['WHITE', 'BLACK', '4'], n, p=[.5, .3, .2]),
    })
    logit = (0.8 * (df['ifng_genotype'] == 'TT')
             + 0.3 * (df['ethnic'] == '4')
             + 0.02 * (df['age'] - 45)
             + rng.normal(0, 0.8, n) - 0.6)
    df['tb_outcome'] = (logit > 0).astype(int)
    if df['tb_outcome'].sum() < 20:
        df.loc[:19, 'tb_outcome'] = 1
    return df


def _kenya_frame(n=300, seed=7):
    """无签名合成帧（BASE13 + tb_outcome，无任何 extras 源列）。

    P6 之前的 kenya CSV 即此形态；真 kenya CSV 现含签名列
    （cxr_tb_suspect/symptom_bloodcough）会被探测命中——本帧用于
    验证未识别数据的 v1 回退路径仍有效。
    """
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        c: rng.rand(n) for c in BASE13
    })
    df['age'] = rng.randint(15, 85, n)
    df['tb_outcome'] = rng.randint(0, 2, n)
    df.loc[:19, 'tb_outcome'] = 1
    return df


# ============================================================================
# 1. cohort_features 单元
# ============================================================================

class TestDetectCohort:

    @pytest.mark.parametrize('sigs,expected', [
        (['income_pir', 'race_eth'], 'nhanes'),
        (['roomshare_n', 'hhdens_quartile'], 'treats'),
        (['cad_score', 'crp_mgdl'], 'crp'),
        (['qft_tb_ag_nil', 'cxr_score'], 'taiwan'),
        (['ifng_genotype', 'ethnic'], 'brazil'),
        (['index_smear_grade', 'mdr_household'], 'peru_mdr'),
        (['cxr_tb_suspect', 'symptom_bloodcough'], 'kenya'),  # P6
    ])
    def test_signature_hits(self, sigs, expected):
        df = pd.DataFrame({c: [0.0, 1.0] for c in sigs})
        assert cf.detect_cohort(df) == expected

    def test_no_signature_returns_none(self):
        df = _kenya_frame(n=20)
        assert cf.detect_cohort(df) is None


class TestBuildV4Extras:

    def test_sentinel_crp(self):
        """crp_mgdl==999 / cad_score==-1 → 置 NaN + 指示器 + 中位数填补。"""
        n = 100
        rng = np.random.RandomState(0)
        # crp spec 全部源列（spec 是列契约，缺源列即坏输入）
        df = pd.DataFrame({
            'gender': rng.choice(['m', 'f'], n),
            'cad_score': np.where(rng.rand(n) < 0.2, -1.0,
                                  rng.uniform(20, 90, n)),
            'crp_mgdl': np.where(rng.rand(n) < 0.3, 999.0,
                                 rng.uniform(0.1, 10, n)),
            'symptom_cough': rng.randint(0, 2, n),
            'symptom_cough_weeks': rng.randint(0, 12, n),
            'symptom_fever': rng.randint(0, 2, n),
            'symptom_fever_weeks': rng.randint(0, 12, n),
            'symptom_chestpain': rng.randint(0, 2, n),
            'symptom_chestpain_weeks': rng.randint(0, 12, n),
            'symptom_nightsweats': rng.randint(0, 2, n),
            'symptom_nightsweats_weeks': rng.randint(0, 12, n),
            'symptom_weightlost': rng.randint(0, 2, n),
            'symptom_weightlost_weeks': rng.randint(0, 12, n),
            'tb_history_raw': rng.choice(['no', 'yes', 'past'], n),
            'country_code': rng.choice(['SA', 'OTHER'], n),
        })
        cols, X, audit = cf.build_v4_extras(df, 'crp', fill='median_full')
        assert 'cad_score_missing' in cols and 'crp_missing' in cols
        rules = {a['column']: a['rule'] for a in audit['sentinels_applied']}
        assert rules.get('crp_mgdl') == '==999.0'
        assert rules.get('cad_score') == '==-1.0'
        # 哨兵行：指示器=1，主列=非哨兵中位数（非 999/-1）
        i_crp = cols.index('crp_mgdl'); i_ind = cols.index('crp_missing')
        sent_rows = X[:, i_ind] == 1.0
        assert sent_rows.any()
        med = np.median(df.loc[~sent_rows, 'crp_mgdl'])
        assert np.allclose(X[sent_rows, i_crp], med)
        assert not np.isnan(X).any()
        assert 'fill_medians' in audit

    def test_brazil_onehot_and_unknown_ind(self):
        df = _brazil_frame(n=200)
        cols, X, audit = cf.build_v4_extras(df, 'brazil', fill='median_full')
        # onehot：base 'AA' 不生成列；命名下划线连接（LightGBM 拒绝
        # 方括号等特殊 JSON 字符）
        assert 'ifng_genotype_AT' in cols and 'ifng_genotype_TT' in cols
        assert not any(c.startswith('ifng_genotype_AA') for c in cols)
        # unknown_ind：ethnic=='4' → 1
        i_unk = cols.index('ethnic_unknown')
        assert set(np.unique(X[:, i_unk])) <= {0.0, 1.0}
        assert (X[:, i_unk] == 1.0).sum() == (df['ethnic'] == '4').sum()
        # binary_map：gender=='f' → 1
        i_f = cols.index('gender_female')
        assert np.array_equal(X[:, i_f], (df['gender'] == 'f').astype(float))

    def test_median_fill_uses_non_nan_median(self):
        n = 50
        df = pd.DataFrame({
            'gender': ['m'] * n,
            'bmi': np.r_[np.full(25, 22.0), np.full(25, np.nan)],
            'smoking': rng_smoke(n), 'born_us': np.ones(n),
            'income_pir': rng_pir(n), 'race_eth': np.ones(n),
        })
        cols, X, _ = cf.build_v4_extras(df, 'nhanes', fill='median_full')
        i = cols.index('bmi')
        assert np.allclose(X[:, i], 22.0)

    def test_kenya_spec_indicators(self):
        """P6：kenya 16 extras——gender 映射 / 条件缺失指示器 /
        cxr 三列直传（ETL 派生契约列）。"""
        n = 200
        rng = np.random.RandomState(11)
        df = pd.DataFrame({
            'gender': rng.choice(['m', 'f'], n),
            'cxr_tb_suspect': rng.randint(0, 2, n).astype(float),
            'cxr_abnormal_other': rng.randint(0, 2, n).astype(float),
            'cxr_missing': np.zeros(n),
            'symptom_bloodcough': rng.randint(0, 2, n).astype(float),
            'symptom_sputum': rng.randint(0, 2, n).astype(float),
            'symptom_chestpains': rng.randint(0, 2, n).astype(float),
            'symptom_fever': rng.randint(0, 2, n).astype(float),
            'symptom_fatigue': rng.randint(0, 2, n).astype(float),
            'symptom_nightsweats': rng.randint(0, 2, n).astype(float),
            # breathless 问卷流条件缺失 ~30% → 指示器
            'symptom_breathless': np.where(rng.rand(n) < 0.3, np.nan,
                                           rng.randint(0, 2, n)),
            'cough_weeks': rng.randint(0, 40, n).astype(float),
            # treatment_sought 未问缺失 ~60% → 指示器
            'treatment_sought': np.where(rng.rand(n) < 0.6, np.nan,
                                         rng.randint(0, 2, n)),
            'hiv_tested': rng.randint(0, 2, n).astype(float),
        })
        cols, X, audit = cf.build_v4_extras(df, 'kenya', fill='median_full')
        assert len(cols) == 16
        assert 'symptom_breathless_missing' in cols
        assert 'treatment_sought_missing' in cols
        assert cols.count('symptom_breathless_missing') == 1
        i_f = cols.index('gender_female')
        assert np.array_equal(X[:, i_f], (df['gender'] == 'f').astype(float))
        i_b = cols.index('symptom_breathless')
        i_bi = cols.index('symptom_breathless_missing')
        miss_b = df['symptom_breathless'].isna().to_numpy()
        assert np.array_equal(X[:, i_bi], miss_b.astype(float))
        # 缺失行 → 非缺失中位数填补（中位数 0/1 由数据决定，非 NaN 即可）
        assert not np.isnan(X).any()
        assert {a['column'] for a in audit['indicators_added']} == {
            'symptom_breathless_missing', 'treatment_sought_missing'}
        assert audit['n_extra_cols'] == 16

    def test_no_spec_raises(self):
        with pytest.raises(ValueError):
            cf.build_v4_extras(pd.DataFrame(), 'nonexistent_cohort')


def rng_smoke(n):
    return np.random.RandomState(3).randint(0, 2, n).astype(float)


def rng_pir(n):
    return np.random.RandomState(4).uniform(0.1, 5, n)


class TestApplyV4FeatureSpace:

    def test_hstack_and_names(self):
        from tb_risk.scoring.ml.training import ensure_interaction_features
        df = _brazil_frame(n=100)
        ensure_interaction_features(df, MLRiskPredictor.INTERACTION_FEATURE_NAMES)
        X, names, audit = cf.apply_v4_feature_space(df, ALL_FEATURE_NAMES)
        assert X.shape[0] == 100 and X.shape[1] == len(names)
        assert names[:22] == ALL_FEATURE_NAMES
        assert 'ifng_genotype_TT' in names
        assert audit['cohort'] == 'brazil'
        assert audit['n_total'] == len(names)
        assert not np.isnan(X).any()

    def test_unidentified_returns_none(self):
        df = _kenya_frame(n=30)
        X, names, audit = cf.apply_v4_feature_space(df, ALL_FEATURE_NAMES)
        assert X is None and names is None
        assert audit['cohort'] is None and 'note' in audit

    def test_return_raw_dual_matrices(self):
        """缺陷1口径统一：return_raw=True 产出部署填补版 + NaN 保留版。

        X（部署工件）全数据中位填补无 NaN；X_raw（CV 折内填补输入）
        extras NaN 保留；fill_medians 与 X 的填补值逐位一致。
        """
        from tb_risk.scoring.ml.training import ensure_interaction_features
        df = _brazil_frame(n=100)
        df.loc[df.index[:20], 'gender'] = np.nan  # extras 源列注入缺失
        ensure_interaction_features(df, MLRiskPredictor.INTERACTION_FEATURE_NAMES)
        X, X_raw, names, audit = cf.apply_v4_feature_space(
            df, ALL_FEATURE_NAMES, return_raw=True)
        assert X.shape == X_raw.shape == (100, len(names))
        assert not np.isnan(X).any()
        i_gf = names.index('gender_female')
        raw_nan = np.isnan(X_raw[:, i_gf])
        assert raw_nan.sum() == 20  # 注入的缺失在 raw 中保留
        # 部署填补值 = 全数据中位数（非 NaN 行的中位），逐位一致
        med = float(np.nanmedian(X_raw[:, i_gf]))
        assert audit['fill_medians']['gender_female'] == pytest.approx(med)
        assert np.allclose(X[raw_nan, i_gf], med)
        # 未缺失单元格两矩阵逐位一致
        assert np.allclose(X[~raw_nan, i_gf], X_raw[~raw_nan, i_gf])

    def test_return_raw_unidentified_four_tuple(self):
        df = _kenya_frame(n=30)
        out = cf.apply_v4_feature_space(df, ALL_FEATURE_NAMES,
                                        return_raw=True)
        assert len(out) == 4
        X, X_raw, names, audit = out
        assert X is None and X_raw is None and names is None
        assert audit['cohort'] is None and 'note' in audit


class TestFoldInternalImputationCV:
    """缺陷1口径统一：v4 CV 指标折内中位填补（与 F2 判决口径一致）。"""

    def _imputer_pipeline(self, estimator):
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        assert isinstance(estimator, Pipeline)
        imputer = estimator.named_steps.get('impute')
        assert isinstance(imputer, SimpleImputer)
        assert imputer.strategy == 'median'
        assert imputer.keep_empty_features is True  # 全 NaN 列填 0（F2 同语义）
        return estimator

    def test_wraps_pipeline_when_raw_given(self, monkeypatch):
        """X_cv_raw 在场 → cross_val_* 收到 Pipeline(SimpleImputer→model)
        且评估矩阵是 raw（折内拟合填补器），不是填补后的 X。"""
        from tb_risk.scoring.ml import training as tr
        rng = np.random.RandomState(0)
        X = rng.rand(60, 4)
        X_raw = X.copy()
        X_raw[:15, 2] = np.nan
        y = (X[:, 0] > 0.5).astype(int)
        captured = {}

        def fake_cvs(estimator, X_arg, *a, **kw):
            captured['estimator'] = estimator
            captured['X'] = X_arg
            return np.array([0.8, 0.8, 0.8, 0.8, 0.8])

        monkeypatch.setattr(tr, 'cross_val_score', fake_cvs)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression().fit(X, y)
        metrics = tr._evaluate_model_cv(
            model, X, y, tr.StratifiedKFold(5, shuffle=True,
                                            random_state=42),
            X_cv_raw=X_raw)
        pipe = self._imputer_pipeline(captured['estimator'])
        assert pipe.named_steps.get('model') is model
        assert captured['X'] is X_raw  # 评估在未填补矩阵上进行
        assert metrics['AUROC'] == pytest.approx(0.8)

    def test_legacy_path_without_raw(self, monkeypatch):
        """X_cv_raw=None → 存量语义：estimator 是裸模型、矩阵是 X。"""
        from tb_risk.scoring.ml import training as tr
        rng = np.random.RandomState(0)
        X = rng.rand(60, 4)
        y = (X[:, 0] > 0.5).astype(int)
        captured = {}

        def fake_cvs(estimator, X_arg, *a, **kw):
            captured['estimator'] = estimator
            captured['X'] = X_arg
            return np.array([0.7] * 5)

        monkeypatch.setattr(tr, 'cross_val_score', fake_cvs)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression().fit(X, y)
        tr._evaluate_model_cv(
            model, X, y, tr.StratifiedKFold(5, shuffle=True,
                                            random_state=42))
        assert captured['estimator'] is model
        assert captured['X'] is X


class TestCVFeatureNameWarningCleanup:
    """P1-a（2026-09-20）：CV 评估管线的 LGBM 特征名告警清理。

    背景：Pipeline(SimpleImputer→LGBM) 中 imputer 默认输出 ndarray，
    LGBM sklearn 包装器在 ndarray 拟合时写入自动特征名（Column_N），
    此后每折 numpy 打分触发一条 "X does not have valid feature names"
    UserWarning（7 队列 × 6 模型 × 2 指标的 v4 主管线 run 刷屏数百条，
    曾靠 run 脚本 filterwarnings 压制）。修复：imputer
    set_output(transform='pandas') → 模型拟合与预测均带名。
    注：组路径 cross_val_predict(n_jobs=-1) 的告警发生在 worker 进程，
    无法在父进程 catch_warnings 捕获；管线构造与无组路径同一行代码，
    以无组路径 + 直接 cross_val_predict(n_jobs=1) 双测覆盖。
    """

    def _lgbm_warn_count(self, fn):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fn()
            return [x for x in w if 'feature name' in str(x.message)]

    def _data(self, seed=0, n=200):
        rng = np.random.RandomState(seed)
        X = rng.rand(n, 5)
        X[rng.rand(n, 5) < 0.2] = np.nan  # 带缺失 → 走 imputer
        y = (X[:, 0] + rng.randn(n) * 0.3 > 0.5).astype(int)
        if len(np.unique(y)) < 2 or min(np.bincount(y)) < 3:
            y[:10] = 1
            y[-10:] = 0
        return X, y

    def test_pipeline_cv_dataframe_raw_no_warnings(self):
        from tb_risk.scoring.ml import training as tr
        from lightgbm import LGBMClassifier
        X, y = self._data()
        model = LGBMClassifier(n_estimators=20, verbose=-1,
                               class_weight='balanced')
        cv = tr.StratifiedKFold(3, shuffle=True, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
        hits = self._lgbm_warn_count(
            lambda: tr._evaluate_model_cv(model, X, y, cv, X_cv_raw=X_df))
        assert hits == [], [str(h.message) for h in hits]

    def test_pipeline_cv_ndarray_raw_no_warnings(self):
        """ndarray 输入（列名保护未触发的回退路径）同样不告警。"""
        from tb_risk.scoring.ml import training as tr
        from lightgbm import LGBMClassifier
        X, y = self._data(seed=1)
        model = LGBMClassifier(n_estimators=20, verbose=-1,
                               class_weight='balanced')
        cv = tr.StratifiedKFold(3, shuffle=True, random_state=42)
        hits = self._lgbm_warn_count(
            lambda: tr._evaluate_model_cv(model, X, y, cv, X_cv_raw=X))
        assert hits == [], [str(h.message) for h in hits]

    def test_pipeline_values_unchanged_by_set_output(self):
        """set_output 只改输出容器（ndarray→DataFrame），数值逐位不变。"""
        from tb_risk.scoring.ml import training as tr
        from lightgbm import LGBMClassifier
        X, y = self._data(seed=2)
        model = LGBMClassifier(n_estimators=20, verbose=-1,
                               class_weight='balanced')
        cv = tr.StratifiedKFold(3, shuffle=True, random_state=42)
        with_set_output = tr._evaluate_model_cv(
            model, X, y, cv, X_cv_raw=X)['AUROC']
        # 关闭 set_output 的基准：手工构造未配置的管线
        import warnings
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pipe = Pipeline([
                ('impute', SimpleImputer(strategy='median',
                                         keep_empty_features=True)),
                ('model', LGBMClassifier(n_estimators=20, verbose=-1,
                                         class_weight='balanced'))])
            baseline = tr.cross_val_score(
                pipe, X, y, cv=cv, scoring='roc_auc').mean()
        assert with_set_output == pytest.approx(float(baseline), abs=1e-12)

    def test_cross_val_predict_inprocess_no_warnings(self):
        """组路径同构造管线的进程内 cross_val_predict 验证（n_jobs=1）。"""
        from tb_risk.scoring.ml import training as tr
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from lightgbm import LGBMClassifier
        X, y = self._data(seed=3)
        pipe = Pipeline([
            ('impute', SimpleImputer(strategy='median',
                                     keep_empty_features=True)),
            ('model', LGBMClassifier(n_estimators=20, verbose=-1,
                                     class_weight='balanced'))])
        # 复刻库内构造（与 _evaluate_model_cv 相同的 set_output 配置）
        pipe.named_steps['impute'].set_output(transform='pandas')
        cv = tr.StratifiedKFold(3, shuffle=True, random_state=42)
        hits = self._lgbm_warn_count(
            lambda: tr.cross_val_predict(
                pipe, X, y, cv=cv, method='predict_proba', n_jobs=1))
        assert hits == [], [str(h.message) for h in hits]


class TestPredictionExtras:

    def test_deployment_alignment_semantics(self):
        """部署单样本：哨兵/缺失 → 训练中位数；基类 onehot 补 0；
        新类别列丢弃；新出现的 spec 漂移列补 0。"""
        df = _brazil_frame(n=200, seed=1)
        cols, X_train, audit = cf.build_v4_extras(df, 'brazil',
                                                  fill='median_full')
        medians = audit['fill_medians']
        # 部署样本 1：TT + BLACK（全字段在场）
        rec1 = {'gender': 'f', 'ifng_genotype': 'TT', 'ethnic': 'BLACK'}
        Xp1, a1 = cf.build_v4_extras_for_prediction(rec1, 'brazil', cols,
                                                    medians)
        assert Xp1.shape == (1, len(cols))
        d1 = dict(zip(cols, Xp1[0]))
        assert d1['ifng_genotype_TT'] == 1.0
        assert d1['ifng_genotype_AT'] == 0.0  # 基类/未命中补 0
        assert d1['gender_female'] == 1.0
        assert d1['ethnic_unknown'] == 0.0
        # 部署样本 2：extras 全缺失 → 中位数 + 指示器路径（字段级 NaN）
        rec2 = {'gender': None, 'ifng_genotype': None, 'ethnic': None}
        Xp2, _ = cf.build_v4_extras_for_prediction(rec2, 'brazil', cols,
                                                   medians)
        assert not np.isnan(Xp2).any()
        # 部署样本 3：训练未见的类别（新类别 → 全 onehot 0 = 基类语义）
        rec3 = {'gender': 'm', 'ifng_genotype': 'GG', 'ethnic': 'ASIAN'}
        Xp3, _ = cf.build_v4_extras_for_prediction(rec3, 'brazil', cols,
                                                   medians)
        d3 = dict(zip(cols, Xp3[0]))
        assert d3['ifng_genotype_AT'] == 0.0 and d3['ifng_genotype_TT'] == 0.0
        # 部署样本 4：expected 含漂移列（spec 已改）→ 补 0 + audit 披露
        Xp4, a4 = cf.build_v4_extras_for_prediction(
            rec1, 'brazil', cols + ['phantom_col'], medians)
        assert Xp4.shape == (1, len(cols) + 1)
        assert Xp4[0, -1] == 0.0
        assert 'spec_drift_filled_zero' in a4

    def test_legitimate_absence_not_drift(self):
        """onehot 未命中类别列 / 记录未缺失时的指示器列是单样本构造的
        合法缺席（值语义与训练逐值一致），不得误报 spec 漂移。"""
        # brazil：rec 命中 TT → ifng_genotype_AT 缺席 = 基类/其他类语义
        df = _brazil_frame(n=200, seed=3)
        cols, _, audit = cf.build_v4_extras(df, 'brazil', fill='median_full')
        rec = {'gender': 'f', 'ifng_genotype': 'TT', 'ethnic': 'BLACK'}
        Xp, a = cf.build_v4_extras_for_prediction(
            rec, 'brazil', cols, audit['fill_medians'])
        assert 'ifng_genotype_AT' in cols  # 训练列在场（缺席合法）
        assert 'spec_drift_filled_zero' not in a
        # crp：cad_score 在场且非哨兵 → cad_score_missing 不产出（=0）
        n = 50
        rng = np.random.RandomState(0)
        crp = pd.DataFrame({
            'gender': rng.choice(['m', 'f'], n),
            'cad_score': rng.uniform(20, 90, n),
            'crp_mgdl': rng.uniform(0.1, 10, n),
            'symptom_cough': rng.randint(0, 2, n),
            'symptom_cough_weeks': rng.randint(0, 12, n),
            'symptom_fever': rng.randint(0, 2, n),
            'symptom_fever_weeks': rng.randint(0, 12, n),
            'symptom_chestpain': rng.randint(0, 2, n),
            'symptom_chestpain_weeks': rng.randint(0, 12, n),
            'symptom_nightsweats': rng.randint(0, 2, n),
            'symptom_nightsweats_weeks': rng.randint(0, 12, n),
            'symptom_weightlost': rng.randint(0, 2, n),
            'symptom_weightlost_weeks': rng.randint(0, 12, n),
            'tb_history_raw': rng.choice(['no', 'yes', 'past'], n),
            'country_code': rng.choice(['SA', 'OTHER'], n),
        })
        # 制造训练期哨兵缺失，确保 cad_score_missing 进训练列集
        crp.loc[crp.index[:10], 'cad_score'] = -1.0
        cols_c, _, audit_c = cf.build_v4_extras(
            crp, 'crp', fill='median_full')
        assert 'cad_score_missing' in cols_c
        rec_c = {k: v for k, v in crp.iloc[20].to_dict().items()}
        Xp_c, a_c = cf.build_v4_extras_for_prediction(
            rec_c, 'crp', cols_c, audit_c['fill_medians'])
        assert 'spec_drift_filled_zero' not in a_c
        d = dict(zip(cols_c, Xp_c[0]))
        assert d['cad_score_missing'] == 0.0  # 在场非哨兵 → 未缺失


# ============================================================================
# 2. train_from_real_data feature_set='v4' 全链路（brazil 冒烟）
# ============================================================================

@pytest.fixture(scope='module')
def brazil_v4_trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('brazil_v4')
    csv = tmp / 'brazil.csv'
    _brazil_frame(n=400).to_csv(csv, index=False)
    predictor = MLRiskPredictor()
    result = train_from_real_data(predictor, str(csv), feature_set='v4')
    return predictor, result


class TestTrainFromRealDataV4:

    def test_success_and_disclosure(self, brazil_v4_trained):
        predictor, result = brazil_v4_trained
        assert result['success'] is True
        fs = result['feature_space']
        assert fs['requested'] == 'v4' and fs['effective'] == 'v4'
        assert fs['cohort'] == 'brazil'
        assert fs['n_base'] == 22 and fs['n_extra_cols'] >= 3
        assert fs['n_total'] == fs['n_base'] + fs['n_extra_cols']
        assert fs['note'] is None

    def test_models_fitted_on_v4_columns(self, brazil_v4_trained):
        from tb_risk.scoring.ml.evaluation import _model_feature_names
        predictor, _ = brazil_v4_trained
        v4_names = predictor.v4_feature_names
        assert len(v4_names) > 22
        assert 'ifng_genotype_TT' in v4_names
        for key, model in predictor.models.items():
            names = _model_feature_names(model)
            assert set(names) == set(v4_names), (
                f'{key} 应拟合于 v4 列集，实际 {len(names)} 列')

    def test_triplet_mounted(self, brazil_v4_trained):
        predictor, _ = brazil_v4_trained
        assert predictor.v4_cohort == 'brazil'
        assert isinstance(predictor.v4_fill_medians, dict)
        assert len(predictor.v4_fill_medians) == \
            len(predictor.v4_feature_names) - 22

    def test_entry_records_v4(self, brazil_v4_trained):
        predictor, _ = brazil_v4_trained
        for key, entry in predictor.model_performance.items():
            assert entry.get('feature_set') == 'v4', key
            assert entry.get('v4_cohort') == 'brazil', key
            assert entry.get('n_features') == len(predictor.v4_feature_names)

    def test_cv_imputation_caliber_disclosed(self, brazil_v4_trained):
        """缺陷1口径披露：v4 训练档案记录 CV 折内中位填补口径。"""
        predictor, _ = brazil_v4_trained
        assert predictor.model_performance
        for key, entry in predictor.model_performance.items():
            assert entry.get('cv_imputation') == 'fold_internal_median', key

    def test_map_columns_keeps_contract_names(self, brazil_v4_trained):
        """ethnic 等契约列不被模糊映射改写（签名探测依赖）。"""
        predictor, result = brazil_v4_trained
        # 若 ethnic 被改写为 ethnicity，cohort 探测会失败 → effective=v1
        assert result['feature_space']['effective'] == 'v4'


class TestKenyaFallback:

    def test_v4_request_falls_back_to_v1(self, tmp_path):
        csv = tmp_path / 'kenya.csv'
        _kenya_frame(n=200).to_csv(csv, index=False)
        predictor = MLRiskPredictor()
        result = train_from_real_data(predictor, str(csv), feature_set='v4')
        assert result['success'] is True
        fs = result['feature_space']
        assert fs['effective'] == 'v1'
        assert fs['cohort'] is None and fs['note']
        from tb_risk.scoring.ml.evaluation import _model_feature_names
        for key, model in predictor.models.items():
            names = _model_feature_names(model)
            if not names:
                continue  # 无名 numpy 档（如部分包装器），按位置预测
            assert set(names) == set(ALL_FEATURE_NAMES), key
        assert not hasattr(predictor, 'v4_feature_names')

    def test_v1_default_no_feature_space_key(self, tmp_path):
        csv = tmp_path / 'plain.csv'
        _kenya_frame(n=120).to_csv(csv, index=False)
        predictor = MLRiskPredictor()
        result = train_from_real_data(predictor, str(csv))
        assert result['success'] is True
        assert 'feature_space' not in result


class TestLeakReferenceColumnProtection:
    """P6：泄漏参照列不被 map_columns 模糊改写（披露表契约）。

    kenya 实测：lab_smear_pos / lab_sputum_requested 双双被改写为
    sputum_smear（重复列碰撞），tb_current_treatment → treatment——
    保护集（V4_SOURCE_COLUMNS ∪ LEAK_REFERENCE_COLUMNS）过滤后保持原名。
    """

    def test_kenya_leak_columns_kept(self):
        from tb_risk.io_utils import map_columns
        from tb_risk.scoring.ml.cohort_features import LEAK_REFERENCE_COLUMNS
        for col in ('lab_smear_pos', 'lab_sputum_requested',
                    'tb_current_treatment'):
            assert col in LEAK_REFERENCE_COLUMNS, col
        cols = ['lab_smear_pos', 'lab_sputum_requested',
                'tb_current_treatment', 'symptom_sputum', 'gender']
        mapping = map_columns(cols, verbose=False)
        prot = set(cf.V4_SOURCE_COLUMNS) | set(LEAK_REFERENCE_COLUMNS)
        # 与 training.py 同一过滤规则：保护列的改写被丢弃
        kept = {o: s for o, s in mapping.items()
                if o not in prot or s == o}
        assert all(o == s for o, s in kept.items()), (
            f'保护列仍被改写: {kept}')


# ============================================================================
# 3. 预测端 + 持久化 round-trip
# ============================================================================

class TestPredictRiskV4:

    def _contact(self, **extra):
        base = {
            'age': 60, 'cumulative_exposure': 500, 'has_symptoms': 1,
            'bcg_vaccine': 1, 'past_illness': 0, 'is_high_risk': 1,
            'single_duration': 120, 'freq_density': 10, 'time_span': 8,
        }
        base.update(extra)
        return base

    def test_predict_with_extras_fields(self, brazil_v4_trained):
        from tb_risk.scoring.ml.evaluation import predict_risk
        predictor, _ = brazil_v4_trained
        contact = self._contact(gender='f', ifng_genotype='TT',
                                ethnic='WHITE')
        result = predict_risk(predictor, contact, contact_type='family')
        assert result is not None and 'ensemble' in result
        for key in predictor.models:
            assert result[key]['risk_probability'] > 0.0, (
                f'{key} 走了零值回退——v4 预测输入构造失败')

    def test_predict_without_extras_fields_uses_medians(
            self, brazil_v4_trained):
        """extras 字段全缺失 → 训练中位数 + 指示器，与训练语义一致。"""
        from tb_risk.scoring.ml.evaluation import predict_risk
        predictor, _ = brazil_v4_trained
        result = predict_risk(predictor, self._contact(),
                              contact_type='family')
        assert result is not None
        for key in predictor.models:
            assert result[key]['risk_probability'] > 0.0, key

    def test_persistence_round_trip(self, brazil_v4_trained, tmp_path):
        from tb_risk.scoring.ml.persistence import save_model, load_model
        from tb_risk.scoring.ml.evaluation import predict_risk
        predictor, _ = brazil_v4_trained
        ckpt = tmp_path / 'brazil_v4.pkl'
        assert save_model(predictor, str(ckpt)) is True

        loaded = MLRiskPredictor()
        assert load_model(loaded, str(ckpt)) is True
        assert loaded.v4_cohort == 'brazil'
        assert list(loaded.v4_feature_names) == list(predictor.v4_feature_names)
        assert loaded.v4_fill_medians == predictor.v4_fill_medians
        assert loaded.feature_set == 'v4'

        contact = self._contact(gender='f', ifng_genotype='TT',
                                ethnic='WHITE')
        result = predict_risk(loaded, contact, contact_type='family')
        assert result is not None
        for key in loaded.models:
            assert result[key]['risk_probability'] > 0.0, key
        # round-trip 数值一致
        ref = predict_risk(predictor, contact, contact_type='family')
        for key in loaded.models:
            assert abs(result[key]['risk_probability']
                       - ref[key]['risk_probability']) < 1e-9, key


# ============================================================================
# 4. 防回归：补零收紧 + 26 维契约歧义
# ============================================================================

class TestNoSilentZeroFill:

    def _fit_on_names(self, names, seed=0):
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.RandomState(seed)
        Xf = pd.DataFrame(rng.rand(150, len(names)), columns=names)
        y = (rng.rand(150) > 0.5).astype(int)
        model = RandomForestClassifier(n_estimators=10, random_state=0)
        model.fit(Xf, y)
        return model

    def test_v4_extras_not_zero_filled(self):
        """v4 extras（真实部署信号列）不得走 v3 网络列的补零降级路径。"""
        from tb_risk.scoring.ml.evaluation import _model_input_frame
        names = ALL_FEATURE_NAMES + ['ifng_genotype_TT', 'cad_score']
        model = self._fit_on_names(names)
        X22 = np.random.RandomState(1).rand(5, 22)
        out = _model_input_frame(model, X22, ALL_FEATURE_NAMES)
        # 不补零 → 原样返回 22 列 numpy（predict 将维度不匹配响亮失败）
        assert isinstance(out, np.ndarray) and out.shape == (5, 22)

    def test_v3_network_cols_still_zero_filled(self):
        from tb_risk.scoring.ml.evaluation import _model_input_frame
        from tb_risk.scoring.ml.feature_audit import NETWORK_FEATURE_NAMES_V3
        net_col = list(NETWORK_FEATURE_NAMES_V3)[0]
        names = ALL_FEATURE_NAMES + [net_col]
        model = self._fit_on_names(names)
        X22 = np.random.RandomState(1).rand(5, 22)
        out = _model_input_frame(model, X22, ALL_FEATURE_NAMES)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == names
        assert np.allclose(out[net_col].values, 0.0)

    def test_contract_check_skips_26dim_v4(self, caplog):
        """brazil 型 v4（22+4=26 维）不误触 v3 契约告警。"""
        import logging
        from tb_risk.scoring.ml.evaluation import _check_feature_contract
        names = ALL_FEATURE_NAMES + ['gender_female', 'ifng_genotype_AT',
                                     'ifng_genotype_TT', 'ethnic_white']
        assert len(names) == 26
        with caplog.at_level(logging.WARNING, logger='tb_risk.ml'):
            _check_feature_contract(object(), names, ALL_FEATURE_NAMES)
        assert not any('特征契约违规' in r.message for r in caplog.records)

    def test_contract_check_still_flags_26dim_v3_mismatch(self, caplog):
        """真 v3 契约违规（网络列拼写错）仍被捕获。"""
        import logging
        from tb_risk.scoring.ml.evaluation import _check_feature_contract
        from tb_risk.scoring.ml.feature_audit import NETWORK_FEATURE_NAMES_V3
        # 15 个体列 + 10 真网络列 + 1 拼写错列 = 26 维，extras 与网络列
        # 有交集 → v3 家族 → 与 SELECTED_FEATURES_V3 名集不等 → 告警
        names = ALL_FEATURE_NAMES[:15] + list(NETWORK_FEATURE_NAMES_V3)[:10] \
            + ['typo_network_col']
        assert len(names) == 26
        with caplog.at_level(logging.WARNING, logger='tb_risk.ml'):
            _check_feature_contract(object(), names, ALL_FEATURE_NAMES)
        assert any('特征契约违规' in r.message for r in caplog.records)


# ============================================================================
# 5. P2：小队列部署模型路由（LR 优先）
# ============================================================================

class TestSelectDeploymentModel:

    """纯函数单测：触发条件 / 回退 / 披露口径。"""

    @staticmethod
    def _perf(**overrides):
        base = {'logistic': {'AUROC': 0.70},
                'lightgbm': {'AUROC': 0.75},
                'random_forest': {'AUROC': 0.72}}
        base.update(overrides)
        return base

    def test_small_n_triggers_lr(self):
        from tb_risk.scoring.ml.training import select_deployment_model
        sel = select_deployment_model(400, 'v1', 22, self._perf())
        assert sel['triggered'] is True
        assert sel['preferred_key'] == 'logistic'
        assert sel['rule'] == 'lr_first_small_or_wide'
        assert any('n=400' in r for r in sel['reasons'])
        # LR 落后幅度披露（lightgbm 0.75 - logistic 0.70 = 0.05）
        assert abs(sel['lr_vs_best_gap'] - 0.05) < 1e-9

    def test_wide_v4_triggers_lr_even_large_n(self):
        from tb_risk.scoring.ml.training import select_deployment_model
        sel = select_deployment_model(5000, 'v4', 27, self._perf())
        assert sel['triggered'] is True
        assert sel['preferred_key'] == 'logistic'
        assert any('27 维' in r for r in sel['reasons'])

    def test_large_n_v1_routes_best_cv(self):
        from tb_risk.scoring.ml.training import select_deployment_model
        sel = select_deployment_model(5000, 'v1', 22, self._perf())
        assert sel['triggered'] is False
        assert sel['preferred_key'] == 'lightgbm'  # CV 最优
        assert sel['rule'] == 'best_cv_auroc'

    def test_lr_missing_falls_back_to_best(self):
        from tb_risk.scoring.ml.training import select_deployment_model
        perf = {'lightgbm': {'AUROC': 0.75},
                'random_forest': {'AUROC': 0.72}}
        sel = select_deployment_model(400, 'v4', 27, perf)
        assert sel['triggered'] is False  # 条件命中但 LR 未训练
        assert sel['preferred_key'] == 'lightgbm'
        assert 'logistic 未训练' in sel['note']

    def test_no_cv_metrics(self):
        from tb_risk.scoring.ml.training import select_deployment_model
        sel = select_deployment_model(400, 'v4', 27, {})
        assert sel['preferred_key'] is None
        assert sel['rule'] == 'no_cv_metrics'


class TestDeploymentModelRouting:

    """train_from_real_data 集成：挂载 / 披露 / 预测端 / 持久化。"""

    def test_brazil_v4_routes_lr(self, brazil_v4_trained):
        """brazil（n=400<1000 且 27 维 v4）双条件命中 → LR 优先。"""
        predictor, result = brazil_v4_trained
        sel = result['model_selection']
        assert sel['triggered'] is True
        assert sel['preferred_key'] == 'logistic'
        assert 'logistic' in predictor.models
        assert predictor.deployment_model['preferred_key'] == 'logistic'
        assert len(sel['reasons']) == 2  # 小样本 + 宽特征空间
        assert set(sel['cv_auroc']) == set(predictor.models)

    def test_kenya_small_v1_routes_lr_on_n_only(self, tmp_path):
        csv = tmp_path / 'kenya.csv'
        _kenya_frame(n=200).to_csv(csv, index=False)
        predictor = MLRiskPredictor()
        result = train_from_real_data(predictor, str(csv))
        sel = result['model_selection']
        assert sel['triggered'] is True
        assert sel['preferred_key'] == 'logistic'
        assert len(sel['reasons']) == 1  # 仅小样本维度
        assert 'v4' not in sel['reasons'][0]

    def test_predict_risk_reports_preferred(self, brazil_v4_trained):
        from tb_risk.scoring.ml.evaluation import predict_risk
        predictor, _ = brazil_v4_trained
        contact = {'age': 60, 'cumulative_exposure': 500, 'has_symptoms': 1,
                   'bcg_vaccine': 1, 'past_illness': 0, 'is_high_risk': 1,
                   'single_duration': 120, 'freq_density': 10, 'time_span': 8,
                   'gender': 'f', 'ifng_genotype': 'TT', 'ethnic': 'WHITE'}
        out = predict_risk(predictor, contact, contact_type='family')
        ens = out['ensemble']
        assert ens['preferred_model_rule'] == 'lr_first_small_or_wide'
        assert ens['preferred_model_probability'] == out['logistic'][
            'risk_probability']
        # best_model 原始口径保留（供对照，不因路由改写）
        assert 'best_model' in ens and 'best_model_auc' in ens

    def test_persistence_round_trip_selector(self, brazil_v4_trained,
                                             tmp_path):
        from tb_risk.scoring.ml.persistence import save_model, load_model
        predictor, _ = brazil_v4_trained
        ckpt = tmp_path / 'brazil_v4_sel.pkl'
        assert save_model(predictor, str(ckpt)) is True
        loaded = MLRiskPredictor()
        assert load_model(loaded, str(ckpt)) is True
        assert loaded.deployment_model == predictor.deployment_model


# ============================================================================
# 6. P1-c（treats 降级机制护栏，2026-09-20）
# ============================================================================

def _treats_frame(n=400, seed=11):
    """treats 型合成数据：BASE13 + treats 签名列（roomshare_n/
    hhdens_quartile）+ extras 源列 + 目标。

    护栏测试不依赖模型质量，信号置于 roomshare_n/hh_tb_contact
    仅保证两类别充分（v4 链路与 brazil 冒烟同构）。
    """
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        'age': rng.randint(15, 85, n),
        'cumulative_exposure': rng.uniform(0, 800, n),
        'has_symptoms': rng.randint(0, 2, n),
        'bcg_vaccine': rng.randint(0, 2, n),
        'has_tb': rng.randint(0, 2, n),
        'contact_distance_score': rng.uniform(0, 10, n),
        'ventilation_score': rng.uniform(0, 10, n),
        'is_high_risk': rng.randint(0, 2, n),
        'past_illness': rng.randint(0, 2, n),
        'exposure_setting_score': rng.uniform(0, 10, n),
        'single_duration': rng.uniform(10, 300, n),
        'freq_density': rng.uniform(1, 20, n),
        'time_span': rng.uniform(1, 52, n),
        'gender': rng.choice(['m', 'f'], n),
        'smoking_status': rng.randint(0, 3, n),      # onehot base 0 → 2 列
        'roomshare_n': rng.randint(0, 6, n),
        'hhdens_quartile': rng.randint(1, 5, n),
        'hh_tb_contact': rng.randint(0, 2, n),
        'alcohol_use': rng.randint(0, 4, n),         # onehot base 0 → 3 列
        'country_code': rng.choice(['SA', 'TZ', 'ZM'], n),
    })
    logit = (0.5 * (df['roomshare_n'] >= 3)
             + 0.4 * df['hh_tb_contact']
             + 0.02 * (df['age'] - 45)
             + rng.normal(0, 0.8, n) - 0.5)
    df['tb_outcome'] = (logit > 0).astype(int)
    if df['tb_outcome'].sum() < 20:
        df.loc[:19, 'tb_outcome'] = 1
    return df


@pytest.fixture(scope='module')
def treats_v4_trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('treats_v4')
    csv = tmp / 'treats.csv'
    _treats_frame(n=400).to_csv(csv, index=False)
    predictor = MLRiskPredictor()
    result = train_from_real_data(predictor, str(csv), feature_set='v4')
    return predictor, result


class TestTreatsResearchOnlyGuardrail:
    """P1-c 三层护栏：训练标记 → 落盘/恢复/旧档回填 → 部署入口拦截。

    判决依据 docs/model_boundary_clinical_validation.md §5（AUROC
    0.618 + 10% 预算敏感度 14.5%），注册表单一真值源
    constants.RESEARCH_ONLY_COHORTS。此前仅文档约束，任何站点可静默
    加载 treats checkpoint 做筛查决策。
    """

    _CONTACT = {
        'age': 40, 'cumulative_exposure': 300, 'has_symptoms': 1,
        'bcg_vaccine': 1, 'past_illness': 0, 'is_high_risk': 0,
        'single_duration': 120, 'freq_density': 10, 'time_span': 8,
        'gender': 'f', 'smoking_status': 1, 'roomshare_n': 2,
        'hhdens_quartile': 3, 'hh_tb_contact': 1, 'alcohol_use': 0,
        'country_code': 'SA',
    }

    def test_training_marks_research_only(self, treats_v4_trained):
        from tb_risk.constants import RESEARCH_ONLY_COHORTS
        predictor, result = treats_v4_trained
        assert result['success'] is True
        assert result['feature_space']['cohort'] == 'treats'
        assert predictor.deployment_status == 'research_only'
        assert predictor.research_only_reason == \
            RESEARCH_ONLY_COHORTS['treats']['reason']

    def test_result_disclosure(self, treats_v4_trained):
        """降级状态随 result 披露（编排层 run JSON 直接可读）。"""
        predictor, result = treats_v4_trained
        disc = result['deployment_status']
        assert disc['status'] == 'research_only'
        assert disc['cohort'] == 'treats'
        assert disc['reason'] == predictor.research_only_reason

    def test_predict_risk_blocked_by_default(self, treats_v4_trained):
        from tb_risk.scoring.ml.evaluation import predict_risk
        predictor, _ = treats_v4_trained
        with pytest.raises(ValueError, match='研究资产'):
            predict_risk(predictor, dict(self._CONTACT),
                         contact_type='family')

    def test_predict_risk_allow_research_passes(self, treats_v4_trained):
        from tb_risk.scoring.ml.evaluation import predict_risk
        predictor, _ = treats_v4_trained
        out = predict_risk(predictor, dict(self._CONTACT),
                           contact_type='family', allow_research=True)
        assert out is not None and 'ensemble' in out
        for key in predictor.models:
            assert out[key]['risk_probability'] > 0.0, key

    def test_persistence_round_trip_flag(self, treats_v4_trained, tmp_path):
        from tb_risk.scoring.ml.persistence import save_model, load_model
        from tb_risk.scoring.ml.evaluation import predict_risk
        predictor, _ = treats_v4_trained
        ckpt = tmp_path / 'treats_v4.pkl'
        assert save_model(predictor, str(ckpt)) is True
        loaded = MLRiskPredictor()
        assert load_model(loaded, str(ckpt)) is True
        assert loaded.deployment_status == 'research_only'
        assert loaded.research_only_reason == predictor.research_only_reason
        with pytest.raises(ValueError, match='研究资产'):
            predict_risk(loaded, dict(self._CONTACT),
                         contact_type='family')
        out = predict_risk(loaded, dict(self._CONTACT),
                           contact_type='family', allow_research=True)
        assert out is not None and 'ensemble' in out

    def test_old_archive_backfill(self, treats_v4_trained, tmp_path):
        """t7 旧档（保存早于护栏落地：无 flag 键但 v4_cohort='treats'）
        加载时按 checkpoint 自带队列名回填补 flag——旧档同样被拦。"""
        import joblib
        from tb_risk.scoring.ml.persistence import save_model, load_model
        from tb_risk.scoring.ml.evaluation import predict_risk
        from tb_risk.constants import RESEARCH_ONLY_COHORTS
        predictor, _ = treats_v4_trained
        ckpt = tmp_path / 'treats_old.pkl'
        assert save_model(predictor, str(ckpt)) is True
        # 剥离 flag 键模拟 t7 旧档
        data = joblib.load(str(ckpt))
        assert data.pop('deployment_status') == 'research_only'
        data.pop('research_only_reason', None)
        assert data['v4_cohort'] == 'treats'
        joblib.dump(data, str(ckpt))
        loaded = MLRiskPredictor()
        assert load_model(loaded, str(ckpt)) is True
        assert loaded.deployment_status == 'research_only'
        # reason 回填自注册表（旧档无原 reason）
        assert loaded.research_only_reason == \
            RESEARCH_ONLY_COHORTS['treats']['reason']
        with pytest.raises(ValueError, match='研究资产'):
            predict_risk(loaded, dict(self._CONTACT),
                         contact_type='family')

    def test_unrelated_cohort_not_affected(self, brazil_v4_trained):
        """非降级队列（brazil）不落标记、部署入口不受影响。"""
        from tb_risk.scoring.ml.evaluation import predict_risk
        predictor, result = brazil_v4_trained
        assert getattr(predictor, 'deployment_status', None) is None
        assert 'deployment_status' not in result
        out = predict_risk(predictor, {
            'age': 60, 'cumulative_exposure': 500, 'has_symptoms': 1,
            'bcg_vaccine': 1, 'past_illness': 0, 'is_high_risk': 1,
            'single_duration': 120, 'freq_density': 10, 'time_span': 8,
        }, contact_type='family')
        assert out is not None and 'ensemble' in out

    def test_non_v4_training_clears_stale_mark(self, tmp_path):
        """同一 predictor 重复训练非 v4 数据 → 陈旧降级标记清除
        （多队列复用同一实例时不残留）。"""
        csv = tmp_path / 'plain.csv'
        _kenya_frame(n=120).to_csv(csv, index=False)
        predictor = MLRiskPredictor()
        # 模拟此前训练过 treats v4（P1-c 标记在场）
        predictor.deployment_status = 'research_only'
        predictor.research_only_reason = 'stale'
        result = train_from_real_data(predictor, str(csv))
        assert result['success'] is True
        assert getattr(predictor, 'deployment_status', None) is None
        assert getattr(predictor, 'research_only_reason', None) is None
        assert 'deployment_status' not in result
