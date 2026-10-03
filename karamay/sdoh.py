#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
维度六替代：社会决定因素（SDOH）模块 — 替代民族遗传易感性

设计原则（遵循 Belmont Report 1979 三原则）：
1. 尊重（Respect）：知情同意，民族仅作为描述性统计变量
2. 行善（Beneficence）：风险评估有助于公共卫生干预
3. 公正（Justice）：不基于民族身份歧视任何群体

文献依据：
- WHO. Social determinants of tuberculosis: a framework for action. WHO/HTM/TB/2023.
- Lönnroth K et al. Drivers of tuberculosis epidemics. Soc Sci Med 68(12):2240-2246, 2009.
- Lönnroth K et al. BMI and TB incidence. Int J Epidemiol 39(1):149-155, 2010.
- Mitchell S et al. Prediction-based decisions and fairness. Annu Rev Stat Appl, 2024.

克拉玛依地区背景：
- 高度城市化（城镇化率 > 98%），以单位公寓为主，人均居住面积约 25-30m²
- 克拉玛依市中心医院为市级结核病定点诊疗机构，独山子/白碱滩/乌尔禾有区级诊疗点
- 油田职工年度体检提供 BMI 数据
- 油田职工收入高于新疆平均水平，2024 年城镇居民人均可支配收入约 5.2 万元/年
"""

import math
import logging

logger = logging.getLogger("tb_risk.karamay.sdoh")


class SDOHModule:
    """社会决定因素（SDOH）风险模块。

    替代原有的 EthnicityPathogenModule，用可测量的结构性变量
    （住房拥挤、医疗可及性、营养风险、收入分层）替代遗传易感性假设。

    用法：
        sdoh = SDOHModule(config)
        factor = sdoh.compute_sdoh_multiplier(contact_data)
        # → 返回加法组合后的 SDOH 乘数
    """

    def __init__(self, config):
        """
        参数：
            config: ConfigProxy 实例
        """
        self._cfg = config
        self._sdoh_cfg = config.get_dict('population.sdoh')

    # ========== 单维度计算 ==========

    def compute_housing_crowding(self, living_area_per_person=None):
        """住房拥挤指数。

        定义：人均居住面积（m²/人）。
        克拉玛依以单位公寓为主，平均约 25-30m²/人。

        当人均面积 < 15m² → 乘数 = 1.0 + (15 - area) × 0.02，上限 1.3。
        文献：WHO SDOH Framework; Lönnroth et al. 2009.

        参数：
            living_area_per_person (float|None): 人均居住面积(m²)，None 时返回 1.0

        返回：
            float: 住房拥挤乘数
        """
        if living_area_per_person is None:
            return 1.0

        hc = self._sdoh_cfg.get('housing_crowding', {})
        threshold = hc.get('threshold_m2', 15.0)
        risk_per_m2 = hc.get('risk_per_m2', 0.02)
        max_mult = hc.get('max_multiplier', 1.3)

        if living_area_per_person >= threshold:
            return 1.0

        factor = 1.0 + (threshold - living_area_per_person) * risk_per_m2
        return min(factor, max_mult)

    def compute_healthcare_access(self, distance_km=None):
        """医疗可及性指数。

        定义：距最近结核病定点诊疗机构的距离(km)。
        克拉玛依市中心医院为市级定点，各区有区级诊疗点。

        当距离 > 30km → 乘数 = 1.0 + (distance - 30) × 0.005，上限 1.25。
        延迟就诊延长传播窗口。
        文献：Lönnroth et al. 2009; 克拉玛依市结核病防治规划 2024.

        参数：
            distance_km (float|None): 距最近诊疗机构的距离(km)，None 时返回 1.0

        返回：
            float: 医疗可及性乘数
        """
        if distance_km is None:
            return 1.0

        ha = self._sdoh_cfg.get('healthcare_access', {})
        threshold = ha.get('threshold_km', 30.0)
        risk_per_km = ha.get('risk_per_km', 0.005)
        max_mult = ha.get('max_multiplier', 1.25)

        if distance_km <= threshold:
            return 1.0

        factor = 1.0 + (distance_km - threshold) * risk_per_km
        return min(factor, max_mult)

    def compute_nutrition_risk(self, bmi=None):
        """营养风险评分。

        定义：BMI 分层。
        油田年度体检提供 BMI 数据。

        BMI < 18.5（低体重）→ 乘数 1.3
        BMI 18.5-25（正常）→ 乘数 1.0
        BMI > 25（超重）→ 保守设为 1.0（肥胖保护效应有争议）
        文献：Lönnroth et al. 2010.

        参数：
            bmi (float|None): 体重指数(kg/m²)，None 时返回 1.0

        返回：
            float: 营养风险乘数
        """
        if bmi is None:
            return 1.0

        nut = self._sdoh_cfg.get('nutrition', {})
        underweight_threshold = nut.get('underweight_threshold', 18.5)
        underweight_mult = nut.get('underweight_multiplier', 1.3)
        normal_mult = nut.get('normal_multiplier', 1.0)
        overweight_mult = nut.get('overweight_multiplier', 1.0)

        if bmi < underweight_threshold:
            return underweight_mult
        elif bmi <= nut.get('normal_high', 25.0):
            return normal_mult
        else:
            return overweight_mult

    def compute_income_risk(self, income_ratio=None):
        """收入分层。

        定义：家庭年收入相对于当地中位数的比值。
        克拉玛依油田职工收入高于新疆平均水平。

        当比值 < 0.6（贫困线）→ 乘数 1.2。
        文献：WHO SDOH Framework.

        参数：
            income_ratio (float|None): 收入/中位数比值，None 时返回 1.0

        返回：
            float: 收入风险乘数
        """
        if income_ratio is None:
            return 1.0

        inc = self._sdoh_cfg.get('income', {})
        poverty_threshold = inc.get('poverty_threshold_ratio', 0.6)
        poverty_mult = inc.get('poverty_multiplier', 1.2)

        if income_ratio < poverty_threshold:
            return poverty_mult
        return 1.0

    # ========== 组合计算 ==========

    def compute_sdoh_multiplier(self, contact_data=None):
        """计算组合 SDOH 乘数（加法模型）。

        加法模型公式：
        combined = 1.0 + Σ(factor_i - 1.0)

        上限 2.0，下限 0.8（保护性因素可降低风险）。
        避免乘法模型导致的多重叠加极端值。

        参数：
            contact_data (dict|None): 接触者数据，可包含：
                - living_area_per_person (float): 人均居住面积(m²)
                - distance_to_tb_center_km (float): 距诊疗机构距离(km)
                - bmi (float): 体重指数
                - income_ratio (float): 收入/中位数比值

        返回：
            dict: {
                'multiplier': float,         # 组合 SDOH 乘数
                'components': dict,          # 各维度详情
                'description': str,          # 文本描述
                'fairness_audit': dict,      # 公平性审计元数据
            }
        """
        if contact_data is None:
            contact_data = {}

        comb = self._sdoh_cfg.get('combination', {})
        max_combined = comb.get('max_combined', 2.0)
        min_combined = comb.get('min_combined', 0.8)

        components = {}
        descriptions = []

        # 1. 住房拥挤
        housing_factor = self.compute_housing_crowding(
            contact_data.get('living_area_per_person'))
        components['housing_crowding'] = {
            'factor': housing_factor,
            'input': contact_data.get('living_area_per_person'),
        }
        if not math.isclose(housing_factor, 1.0, rel_tol=1e-9):
            area = contact_data.get('living_area_per_person', '?')
            descriptions.append(f'住房拥挤(人均{area}m²) ×{housing_factor:.2f}')

        # 2. 医疗可及性
        access_factor = self.compute_healthcare_access(
            contact_data.get('distance_to_tb_center_km'))
        components['healthcare_access'] = {
            'factor': access_factor,
            'input': contact_data.get('distance_to_tb_center_km'),
        }
        if not math.isclose(access_factor, 1.0, rel_tol=1e-9):
            dist = contact_data.get('distance_to_tb_center_km', '?')
            descriptions.append(f'医疗可及性(距定点{dist}km) ×{access_factor:.2f}')

        # 3. 营养风险
        nutrition_factor = self.compute_nutrition_risk(
            contact_data.get('bmi'))
        components['nutrition'] = {
            'factor': nutrition_factor,
            'input': contact_data.get('bmi'),
        }
        if not math.isclose(nutrition_factor, 1.0, rel_tol=1e-9):
            bmi = contact_data.get('bmi', '?')
            descriptions.append(f'营养风险(BMI={bmi}) ×{nutrition_factor:.2f}')

        # 4. 收入分层
        income_factor = self.compute_income_risk(
            contact_data.get('income_ratio'))
        components['income'] = {
            'factor': income_factor,
            'input': contact_data.get('income_ratio'),
        }
        if not math.isclose(income_factor, 1.0, rel_tol=1e-9):
            ratio = contact_data.get('income_ratio', '?')
            descriptions.append(f'收入分层(比值={ratio}) ×{income_factor:.2f}')

        # 加法组合
        combined = 1.0
        for comp in components.values():
            combined += (comp['factor'] - 1.0)

        # 裁剪到上下限
        combined = max(min_combined, min(max_combined, combined))

        # 公平性审计元数据
        fairness_audit = {
            'method': 'additive',
            'formula': 'combined = 1.0 + Σ(factor_i - 1.0)',
            'max_combined': max_combined,
            'min_combined': min_combined,
            'n_dimensions_applied': sum(
                1 for c in components.values()
                if not math.isclose(c['factor'], 1.0, rel_tol=1e-9)
            ),
            'dimensions': list(components.keys()),
            'belmont_compliance': {
                'respect': '民族仅作为描述性统计变量，不参与风险计算',
                'beneficence': 'SDOH评估有助于针对性公共卫生干预',
                'justice': '不基于民族身份进行风险区分，公平对待所有群体',
            },
            'irb_note': '使用前需获得新疆第二医学院或克拉玛依市中心医院伦理委员会批准',
        }

        return {
            'multiplier': combined,
            'components': components,
            'description': '; '.join(descriptions) if descriptions else '无SDOH风险因素',
            'fairness_audit': fairness_audit,
        }

    # ========== 公平性审计 ==========

    def audit_fairness(self, scored_records, sdoh_field='income_ratio',
                        threshold=0.6, metric='equalized_odds'):
        """公平性审计：检查模型在不同 SDOH 分层中的性能差异。

        指标：
        - equalized_odds: 各组的 TPR 和 FPR 差异
        - demographic_parity: 各组的阳性预测率差异

        阈值：差异 < 5% 视为通过。

        参数：
            scored_records (list[dict]): 包含 risk_score 和 is_confirmed 的记录
            sdoh_field (str): 用于分层的 SDOH 字段名
            threshold (float): 分层阈值
            metric (str): 审计指标 ('equalized_odds' 或 'demographic_parity')

        返回：
            dict: 公平性审计报告
        """
        # 分组
        group_a = [r for r in scored_records
                    if r.get(sdoh_field, 0) >= threshold]
        group_b = [r for r in scored_records
                    if r.get(sdoh_field, 0) < threshold]

        if len(group_a) < 5 or len(group_b) < 5:
            return {
                'metric': metric,
                'sdoh_field': sdoh_field,
                'threshold': threshold,
                'warning': '样本量不足 (每组 < 5)，无法进行公平性审计',
                'passed': None,
            }

        result = {
            'metric': metric,
            'sdoh_field': sdoh_field,
            'threshold': threshold,
            'group_a_size': len(group_a),
            'group_b_size': len(group_b),
            'group_a_label': f'{sdoh_field} >= {threshold}',
            'group_b_label': f'{sdoh_field} < {threshold}',
        }

        if metric == 'equalized_odds':
            result.update(self._compute_equalized_odds(group_a, group_b))
        elif metric == 'demographic_parity':
            result.update(self._compute_demographic_parity(group_a, group_b))

        result['passed'] = all(
            abs(diff) < 0.05 for diff in result.get('differences', {}).values()
        )

        return result

    @staticmethod
    def _compute_equalized_odds(group_a, group_b):
        """计算 equalized odds（TPR 和 FPR 差异）。"""
        import numpy as np

        def compute_rates(records):
            y_true = np.array([r.get('is_confirmed', 0) for r in records])
            y_pred = np.array([r.get('disease_probability', 0.0) for r in records])
            threshold = np.median(y_pred) if len(y_pred) > 0 else 0.5
            y_pred_bin = (y_pred >= threshold).astype(int)

            tp = np.sum((y_true == 1) & (y_pred_bin == 1))
            fn = np.sum((y_true == 1) & (y_pred_bin == 0))
            fp = np.sum((y_true == 0) & (y_pred_bin == 1))
            tn = np.sum((y_true == 0) & (y_pred_bin == 0))

            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            return tpr, fpr

        tpr_a, fpr_a = compute_rates(group_a)
        tpr_b, fpr_b = compute_rates(group_b)

        return {
            'group_a': {'TPR': tpr_a, 'FPR': fpr_a},
            'group_b': {'TPR': tpr_b, 'FPR': fpr_b},
            'differences': {
                'TPR_difference': abs(tpr_a - tpr_b),
                'FPR_difference': abs(fpr_a - fpr_b),
            },
        }

    @staticmethod
    def _compute_demographic_parity(group_a, group_b):
        """计算 demographic parity（阳性预测率差异）。"""
        import numpy as np

        def positive_rate(records):
            y_pred = np.array([r.get('disease_probability', 0.0) for r in records])
            threshold = np.median(y_pred) if len(y_pred) > 0 else 0.5
            return np.mean(y_pred >= threshold)

        rate_a = positive_rate(group_a)
        rate_b = positive_rate(group_b)

        return {
            'group_a_positive_rate': rate_a,
            'group_b_positive_rate': rate_b,
            'differences': {
                'positive_rate_difference': abs(rate_a - rate_b),
            },
        }


class EthnicityDescriptor:
    """民族描述性统计模块（仅用于人口学描述，不参与风险计算）。

    遵循 Belmont Report 公正原则和《个人信息保护法》敏感信息处理要求：
    - 民族信息仅用于疫情分布描述
    - 不参与任何风险计算公式
    - 数据收集需获得知情同意
    - 遵循数据最小化原则
    """

    def __init__(self, config):
        self._cfg = config

    def get_ethnicity_label(self, ethnicity):
        """获取民族标签（描述性用途）"""
        labels = self._cfg.get_dict('population.ethnicity_labels')
        return labels.get(ethnicity, str(ethnicity))

    @staticmethod
    def _resolve_ethnicity_key(ethnicity):
        """将中文民族名映射为内部英文键"""
        _ETHNICITY_CN_TO_EN = {
            '汉族': 'han', '维吾尔族': 'uyghur', '哈萨克族': 'kazakh',
            '回族': 'hui', '蒙古族': 'mongolian', '其他': 'other',
            '柯尔克孜族': 'kyrgyz', '塔吉克族': 'tajik',
            '锡伯族': 'xibe', '满族': 'manchu', '俄罗斯族': 'russian',
            '乌孜别克族': 'uzbek', '塔塔尔族': 'tatar', '达斡尔族': 'daur',
        }
        return _ETHNICITY_CN_TO_EN.get(
            ethnicity.strip() if isinstance(ethnicity, str) else ethnicity,
            ethnicity.lower() if isinstance(ethnicity, str) else str(ethnicity))

    def get_descriptive_info(self, ethnicity='han'):
        """获取民族描述性信息（不参与风险计算）。

        ⚠️ 注意：此方法仅返回人口学描述信息，
        不返回任何风险乘数。民族身份不应用于风险分层。
        """
        key = self._resolve_ethnicity_key(ethnicity)
        label = self.get_ethnicity_label(key)
        return {
            'ethnicity_key': key,
            'ethnicity_label': label,
            'risk_contribution': 0.0,
            'note': '民族为描述性统计变量，不参与风险计算（Belmont Report 公正原则）',
        }