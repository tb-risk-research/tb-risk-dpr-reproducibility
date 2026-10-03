#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反事实分析（Counterfactual Analysis）— what-if 干预推理

反事实分析回答"如果该患者暴露时间减少 50%，风险会降低多少"这类 what-if 问题。
这是因果推断阶梯的第二层（干预），高于 SHAP 所在的相关性层（观察）。

实现方式：基于已训练的 ML 预测器，修改特征后重新预测，对比原预测与反事实预测。
- 个体反事实：单个接触者在特征改变后的风险变化
- 群体反事实：一群接触者的平均反事实效应
- 场景化反事实：暴露减半、通风改善、延迟缩短等预设干预

注意：反事实分析的有效性依赖预测器已正确建模特征→结局的因果机制。
若预测器仅拟合了相关性（如未调整混杂），反事实结果可能有偏。
建议配合 do-calculus（基于 DAG 的后门调整）使用以获得无偏估计。

文献支撑：
- Pearl J. Causality, 2009. §1.4 反事实定义（结构因果模型）
- Pearl J. From Bayesian networks to causal networks, 1994.
- Hernán MA, Robins JM. Causal Inference: What If. Chapman & Hall, 2020.
- Rubin DB. Estimating causal effects of treatments in randomized and
  nonrandomized studies. J Educ Psychol 66(5):688-701, 1974.

依赖：复用 scoring.predictor.MLRiskPredictor（无新依赖）。
"""
import copy
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

LOGGER = logging.getLogger("tb_risk.validation.causal.counterfactual")


class CounterfactualAnalyzer:
    """反事实分析器：基于 ML 预测器回答 what-if 干预问题。

    用法：
        analyzer = CounterfactualAnalyzer(predictor)
        # 个体反事实：若将累积暴露减半
        result = analyzer.individual_counterfactual(
            contact_data,
            feature_overrides={'cumulative_exposure': contact_data['cumulative_exposure'] / 2})
        # result = {'original_risk': 0.35, 'counterfactual_risk': 0.18, 'delta': -0.17, ...}

    参数：
        predictor: MLRiskPredictor 实例（须已训练）
        risk_key: str, 从预测结果中提取的风险值键名（默认 'ensemble'）
    """

    # 预设干预场景模板
    SCENARIO_TEMPLATES = {
        # 暴露减半（缩短接触时长/频次）
        'exposure_halved': lambda d: _scale_feature(d, 'cumulative_exposure', 0.5),
        # 暴露减少 80%（接近完全隔离）
        'exposure_isolated': lambda d: _scale_feature(d, 'cumulative_exposure', 0.2),
        # 通风改善至最好（5=完全户外）
        'ventilation_improved': lambda d: _override_feature(d, 'ventilation', 5),
        # 距离拉远至最远
        'distance_maximized': lambda d: _override_feature(d, 'contact_distance', 'distant'),
        # 单次时长减半
        'duration_halved': lambda d: _scale_feature(d, 'single_duration', 0.5),
        # 频次减半
        'frequency_halved': lambda d: _scale_feature(d, 'freq_density', 0.5),
    }

    def __init__(self, predictor, risk_key: str = 'ensemble'):
        self.predictor = predictor
        self.risk_key = risk_key

    # ------------------------------------------------------------------
    # 个体反事实
    # ------------------------------------------------------------------
    def individual_counterfactual(
        self,
        contact_data: Dict,
        feature_overrides: Dict[str, Any],
        contact_type: str = 'family',
        patient_context: Optional[Dict] = None,
    ) -> Dict:
        """计算单个接触者的反事实风险变化。

        参数：
            contact_data: dict, 接触者原始数据
            feature_overrides: dict, 反事实特征覆盖（如 {'cumulative_exposure': 20}）
            contact_type: str, 'family' 或 'social'
            patient_context: dict, 可选患者上下文

        返回：
            dict: {
                'original_risk': float,       原始风险概率
                'counterfactual_risk': float, 反事实风险概率
                'delta': float,               风险变化（反事实 - 原始）
                'relative_change': float,     相对变化（%）
                'overrides': dict,            应用的覆盖
                'model_detail': dict,         各模型原始/反事实对比
            }
        """
        if not self.predictor.is_trained:
            return self._untrained_result(feature_overrides)

        # 原始预测
        original = self.predictor.predict_risk(
            contact_data, contact_type=contact_type,
            patient_context=patient_context)
        if original is None:
            return self._error_result("原始预测失败", feature_overrides)

        # 反事实数据（深拷贝避免污染原数据）
        cf_data = copy.deepcopy(contact_data)
        for k, v in feature_overrides.items():
            cf_data[k] = v

        cf_prediction = self.predictor.predict_risk(
            cf_data, contact_type=contact_type,
            patient_context=patient_context)
        if cf_prediction is None:
            return self._error_result("反事实预测失败", feature_overrides)

        orig_risk = self._extract_risk(original)
        cf_risk = self._extract_risk(cf_prediction)
        delta = cf_risk - orig_risk
        rel = (delta / orig_risk * 100.0) if orig_risk > 1e-12 else 0.0

        # 各模型对比
        model_detail = {}
        for model_name in original:
            if model_name == 'ensemble':
                continue
            o = original[model_name].get('risk_probability') \
                if isinstance(original[model_name], dict) else None
            c = cf_prediction.get(model_name, {})
            c = c.get('risk_probability') if isinstance(c, dict) else None
            if o is not None and c is not None:
                model_detail[model_name] = {
                    'original': float(o), 'counterfactual': float(c),
                    'delta': float(c - o),
                }

        return {
            'original_risk': float(orig_risk),
            'counterfactual_risk': float(cf_risk),
            'delta': float(delta),
            'relative_change': float(rel),
            'overrides': dict(feature_overrides),
            'model_detail': model_detail,
            'note': ('delta = 反事实风险 - 原始风险；'
                     '负值表示干预降低风险。基于 ML 预测器的反事实推理，'
                     '其无偏性依赖预测器已正确建模因果机制'),
        }

    # ------------------------------------------------------------------
    # 群体反事实
    # ------------------------------------------------------------------
    def population_counterfactual(
        self,
        contact_data_list: Sequence[Dict],
        feature_overrides: Dict[str, Any],
        contact_types: Optional[Sequence[str]] = None,
        patient_context: Optional[Dict] = None,
    ) -> Dict:
        """计算一群接触者的平均反事实效应。

        返回群体平均原始风险、反事实风险及平均风险变化，
        并给出个体效应的分布（min/max/std）。
        """
        if not self.predictor.is_trained:
            return self._untrained_result(feature_overrides)
        if not contact_data_list:
            return self._error_result("空样本列表", feature_overrides)

        if contact_types is None:
            contact_types = ['family'] * len(contact_data_list)

        deltas: List[float] = []
        orig_risks: List[float] = []
        cf_risks: List[float] = []
        errors = 0

        for cd, ct in zip(contact_data_list, contact_types):
            try:
                res = self.individual_counterfactual(
                    cd, feature_overrides, contact_type=ct,
                    patient_context=patient_context)
            except Exception as e:  # 单样本异常不得中断群体分析
                LOGGER.warning("样本反事实计算异常，跳过: %s", e)
                errors += 1
                continue
            if 'delta' in res and res['delta'] == res['delta']:  # 非 NaN
                deltas.append(res['delta'])
                orig_risks.append(res['original_risk'])
                cf_risks.append(res['counterfactual_risk'])
            else:
                errors += 1

        if not deltas:
            return self._error_result("所有样本反事实计算失败", feature_overrides)

        if NUMPY_AVAILABLE:
            arr = np.array(deltas)
            stats = {
                'mean_delta': float(arr.mean()),
                'std_delta': float(arr.std()),
                'min_delta': float(arr.min()),
                'max_delta': float(arr.max()),
            }
            orig_arr = np.array(orig_risks)
            cf_arr = np.array(cf_risks)
            stats['mean_original'] = float(orig_arr.mean())
            stats['mean_counterfactual'] = float(cf_arr.mean())
            stats['mean_relative_change'] = float(
                (arr / orig_arr * 100.0).mean()) if (orig_arr > 1e-12).all() else 0.0
        else:
            stats = {
                'mean_delta': sum(deltas) / len(deltas),
                'std_delta': 0.0,
                'min_delta': min(deltas),
                'max_delta': max(deltas),
                'mean_original': sum(orig_risks) / len(orig_risks),
                'mean_counterfactual': sum(cf_risks) / len(cf_risks),
                'mean_relative_change': 0.0,
            }

        return {
            **stats,
            'n_samples': len(deltas),
            'n_errors': errors,
            'overrides': dict(feature_overrides),
            'individual_deltas': deltas if len(deltas) <= 100 else deltas[:100],
            'note': '群体平均反事实效应；individual_deltas 截断至前 100 个样本',
        }

    # ------------------------------------------------------------------
    # 场景化反事实
    # ------------------------------------------------------------------
    def scenario_analysis(
        self,
        contact_data_list: Sequence[Dict],
        scenario: str,
        contact_types: Optional[Sequence[str]] = None,
        patient_context: Optional[Dict] = None,
    ) -> Dict:
        """预设干预场景的反事实分析。

        参数：
            scenario: str, 场景名。支持：
                - 'exposure_halved': 累积暴露减半
                - 'exposure_isolated': 暴露减少 80%（接近隔离）
                - 'ventilation_improved': 通风改善至完全户外
                - 'distance_maximized': 距离拉至最远
                - 'duration_halved': 单次时长减半
                - 'frequency_halved': 频次减半
        """
        if scenario not in self.SCENARIO_TEMPLATES:
            raise ValueError(
                f"未知场景: {scenario}；支持: {list(self.SCENARIO_TEMPLATES)}")

        # 对每个样本应用场景模板，提取实际覆盖值
        override_template = self.SCENARIO_TEMPLATES[scenario]
        overridden = [override_template(copy.deepcopy(d)) for d in contact_data_list]

        # 推断实际覆盖（取第一个样本的差异）
        if contact_data_list:
            sample_orig = contact_data_list[0]
            sample_new = overridden[0]
            actual_overrides = {
                k: sample_new[k] for k in sample_new
                if k not in sample_orig or sample_orig[k] != sample_new[k]
            }
        else:
            actual_overrides = {}

        # 直接对已覆盖的数据预测，与原始对比
        if not self.predictor.is_trained:
            return self._untrained_result(actual_overrides)

        if contact_types is None:
            contact_types = ['family'] * len(contact_data_list)

        deltas: List[float] = []
        orig_risks: List[float] = []
        cf_risks: List[float] = []

        skipped = 0
        for orig_data, cf_data, ct in zip(contact_data_list, overridden,
                                          contact_types):
            try:
                o = self.predictor.predict_risk(
                    orig_data, contact_type=ct,
                    patient_context=patient_context)
                c = self.predictor.predict_risk(
                    cf_data, contact_type=ct,
                    patient_context=patient_context)
            except Exception as e:  # 单样本异常不得中断场景分析
                LOGGER.warning("场景 %s 样本预测异常，跳过: %s", scenario, e)
                skipped += 1
                continue
            if o is None or c is None:
                continue
            or_ = self._extract_risk(o)
            cr = self._extract_risk(c)
            orig_risks.append(or_)
            cf_risks.append(cr)
            deltas.append(cr - or_)

        if not deltas:
            return self._error_result(
                f"场景 {scenario} 计算失败（skipped={skipped}）",
                actual_overrides)

        if NUMPY_AVAILABLE:
            arr = np.array(deltas)
            mean_delta = float(arr.mean())
            std_delta = float(arr.std())
        else:
            mean_delta = sum(deltas) / len(deltas)
            std_delta = 0.0

        mean_orig = sum(orig_risks) / len(orig_risks)
        mean_cf = sum(cf_risks) / len(cf_risks)

        return {
            'scenario': scenario,
            'mean_original_risk': float(mean_orig),
            'mean_counterfactual_risk': float(mean_cf),
            'mean_delta': float(mean_delta),
            'std_delta': float(std_delta),
            'mean_relative_change': float(
                (mean_cf - mean_orig) / mean_orig * 100.0)
            if mean_orig > 1e-12 else 0.0,
            'n_samples': len(deltas),
            'n_skipped': skipped,
            'actual_overrides': actual_overrides,
            'note': f'场景化反事实分析: {scenario}',
        }

    def compare_scenarios(
        self,
        contact_data_list: Sequence[Dict],
        scenarios: Optional[Sequence[str]] = None,
        contact_types: Optional[Sequence[str]] = None,
        patient_context: Optional[Dict] = None,
    ) -> Dict[str, Dict]:
        """对比多个干预场景的效应，辅助干预策略优化。"""
        if scenarios is None:
            scenarios = list(self.SCENARIO_TEMPLATES.keys())
        results = {}
        for s in scenarios:
            try:
                results[s] = self.scenario_analysis(
                    contact_data_list, s, contact_types, patient_context)
            except (ValueError, RuntimeError) as e:
                results[s] = {'error': str(e)}
        return results

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def _extract_risk(self, prediction: Dict) -> float:
        """从预测结果中提取风险概率。"""
        if self.risk_key in prediction:
            entry = prediction[self.risk_key]
            if isinstance(entry, dict):
                return float(entry.get('risk_probability', 0.0))
            if entry is None:
                # 集成结果可能为 None（如所有基模型均不可用），回退到基模型
                pass
            elif isinstance(entry, (int, float)):
                return float(entry)
        # 回退：取第一个含 risk_probability 的模型结果
        for k, v in prediction.items():
            if k == 'ensemble':
                continue
            if isinstance(v, dict) and 'risk_probability' in v:
                return float(v['risk_probability'])
        return 0.0

    def _untrained_result(self, overrides) -> Dict:
        return {
            'original_risk': float('nan'),
            'counterfactual_risk': float('nan'),
            'delta': float('nan'),
            'relative_change': float('nan'),
            'overrides': dict(overrides),
            'note': '预测器未训练，反事实分析不可用',
        }

    def _error_result(self, reason: str, overrides) -> Dict:
        return {
            'original_risk': float('nan'),
            'counterfactual_risk': float('nan'),
            'delta': float('nan'),
            'relative_change': float('nan'),
            'overrides': dict(overrides),
            'note': f'反事实分析失败: {reason}',
        }


# ======================================================================
# 模块级辅助函数（场景模板用）
# ======================================================================

def _scale_feature(data: Dict, key: str, factor: float) -> Dict:
    """对数值特征按比例缩放（如暴露减半 factor=0.5）。"""
    if key in data and isinstance(data[key], (int, float)):
        data[key] = data[key] * factor
    return data


def _override_feature(data: Dict, key: str, value) -> Dict:
    """直接覆盖特征值。"""
    data[key] = value
    return data
