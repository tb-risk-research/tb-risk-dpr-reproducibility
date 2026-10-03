#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
维度七：地方政策规则引擎（深化版）
"""


class LocalPolicyEngine:
    """维度七：地方政策规则引擎（深化版）

    输入：对象行政区划、是否为关爱行动试点
    输出：
    - 筛查建议升级：独山子关爱行动→高风险者升级至"胸片+分子检测"
    - 集中收治假设：收治率66.1%→已确诊患者默认社区暴露期为收治前周数
    - 传染期衰减：根据政策调整treatment_infectivity_factors中"未治疗"持续时间

    关联点：
    - _generate_potential_patients中动态生成recommendation文本
    - 治疗阶段传染性衰减计算中根据政策调整未治疗期实际持续时间
    """

    def __init__(self, config):
        self._cfg = config

    @property
    def city_centralized_treatment_rate(self):
        return self._cfg.get('policy.city_centralized_treatment_rate', 0.661)

    @property
    def is_free_treatment(self):
        return self._cfg.get('policy.free_treatment_policy', True)

    @property
    def migrant_population_coverage(self):
        return self._cfg.get('policy.migrant_population_coverage', 0.75)

    def get_district_policy(self, district='克拉玛依区'):
        districts = self._cfg.get_dict('policy.districts')
        return districts.get(district, districts.get('克拉玛依区', {}))

    def is_care_action_district(self, district):
        policy = self.get_district_policy(district)
        return policy.get('care_action_active', False)

    def get_centralized_treatment_info(self, district='克拉玛依区'):
        policy = self.get_district_policy(district)
        return {
            'rate': policy.get('centralized_treatment_rate', self.city_centralized_treatment_rate),
            'center': policy.get('treatment_center', '克拉玛依市中心医院'),
            'care_action_active': policy.get('care_action_active', False)
        }

    def get_estimated_community_exposure_weeks(self, district='克拉玛依区'):
        """估算确诊患者在社区的暴露周数

        基于集中收治率：收治率66.1% → 约1/3延迟收治
        假设高效收治：确诊后2周内入院; 非高效：4-8周
        """
        ti = self.get_centralized_treatment_info(district)
        rate = ti.get('rate', self.city_centralized_treatment_rate)
        if rate >= 0.85:
            return 2.0
        elif rate >= 0.70:
            return 3.0
        elif rate >= 0.60:
            return 5.0
        else:
            return 8.0

    def get_screening_recommendation(self, risk_level='medium',
                                        district='克拉玛依区',
                                        is_high_risk_group=False):
        """根据地方政策动态生成筛查建议

        独山子关爱行动试点区域：
          - 高风险 → "立即进行胸部X线+分子生物学检测"
          - 中风险/重点人群 → "升级为胸部X线+IGRA联合筛查"
        """
        templates = self._cfg.get_dict('policy.screening_templates')
        is_care = self.is_care_action_district(district)
        care_name = '独山子关爱行动' if is_care else '市防控要求'

        if risk_level == 'high':
            if is_care:
                return templates.get('care_action_enhanced',
                                     '立即进行胸部X线+分子生物学检测')
            return templates.get('urgent', '立即进行综合筛查（胸片+GeneXpert+痰培养）').format(
                action=care_name
            )
        elif risk_level == 'medium' and is_high_risk_group:
            return templates.get('enhanced',
                                 '根据{action}要求，升级为胸部X线+IGRA联合筛查').format(
                action=care_name
            )
        elif is_care:
            interval_days = self._cfg.get('policy.dushanzi_care_action.followup_interval_days_care', 14)
            tpl = templates.get('followup_care', '纳入{action}社区随访计划，每{interval}天随访一次')
            return tpl.format(action=care_name, interval=interval_days)
        return templates.get('standard', '建议进行PPD/IGRA筛查')

    def get_followup_interval(self, district='克拉玛依区', is_high_risk=False):
        policy = self.get_district_policy(district)
        if policy.get('care_action_active'):
            if is_high_risk:
                return self._cfg.get('policy.dushanzi_care_action.followup_interval_days_care_highrisk', 7)
            return self._cfg.get('policy.dushanzi_care_action.followup_interval_days_care', 14)
        base = policy.get('screening_frequency_months',
                          self._cfg.get('policy.default_screening_interval_months', 6)) * 30
        return max(7, base // 2) if is_high_risk else base

    def get_counterfactual_interventions(self):
        return self._cfg.get_list('counterfactual_interventions',
                                  self._get_default_interventions())

    def enhance_recommendation(self, contact_info):
        """对已生成的基础recommendation进行政策增强（基于行政区划差异化）

        参数：
            contact_info (dict): 包含 priority, district, workplace_type, 
                                 is_high_risk, past_illness_type 等字段

        返回：
            str: 增强后的推荐建议文本
        """
        priority = contact_info.get('priority', '低')
        district = contact_info.get('district', '克拉玛依区')
        is_high_risk = contact_info.get('is_high_risk', False)
        risk_level = {'极高': 'high', '高': 'high', '中': 'medium', '低': 'low'}.get(priority, 'low')
        return self.get_screening_recommendation(risk_level, district, is_high_risk)

    @staticmethod
    def _get_default_interventions():
        return [
            {
                'id': 'centralized_treatment', 'name': '集中收治管理',
                'estimated_effect': '治疗成功率↑10-15%，传播风险↓30-40%',
                'cost': '中', 'literature': '中国CDC结核病中心, 2024'
            },
            {
                'id': 'care_action_enhanced', 'name': '关爱行动强化筛查',
                'estimated_effect': '早期发现率↑25%，潜伏感染干预率↑20%',
                'cost': '中-高', 'literature': '克拉玛依网, 2025'
            },
            {
                'id': 'migrant_coverage', 'name': '流动人口结核病服务覆盖',
                'estimated_effect': '诊断延迟↓40%，治疗完成率↑20%',
                'cost': '中', 'literature': '克拉玛依市流动人口防控政策'
            }
        ]

    def get_policy_summary(self):
        return {
            'city_centralized_treatment_rate': self.city_centralized_treatment_rate,
            'free_treatment': self.is_free_treatment,
            'migrant_coverage': self.migrant_population_coverage,
            'districts': {
                d: self.get_district_policy(d)
                for d in ['克拉玛依区', '独山子区', '白碱滩区', '乌尔禾区']
            }
        }