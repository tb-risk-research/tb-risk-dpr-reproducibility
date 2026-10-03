#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
维度三：静脉吸毒人群（IDU）风险模块（深化版）
"""


class IDURiskModule:
    """维度三：静脉吸毒人群（IDU）风险模块（深化版）

    输入：contact['idu_status'] = True
    输出：
    - 免疫抑制组合因子 idu_combined = 10.0（综合营养不良+HIV共感染+免疫损伤）
    - 治疗依从性惩罚 adherence_penalty = 0.6（源于0%正规疗程完成率）
    - 年龄分层年进展率：青年30%、成人20%、老人15%，替换默认10%基线

    关联点：
    - _generate_potential_patients中新增social子类 'idu'
    - 反事实分析增加"强制戒毒/美沙酮维持治疗"场景
    """

    CONTACT_TYPE_IDU = 'idu'

    def __init__(self, config):
        self._cfg = config
        sp = self._cfg.get_dict('population.special_populations.idu')
        self._idu_params = sp if sp else {}

    @property
    def immunosuppression_combined(self):
        return self._idu_params.get('immunosuppression_combined', 10.0)

    @property
    def adherence_penalty(self):
        return self._idu_params.get('adherence_penalty', 0.6)

    @property
    def progression_rate_youth(self):
        return self._idu_params.get('progression_rate_youth', 0.30)

    @property
    def progression_rate_adult(self):
        return self._idu_params.get('progression_rate_adult', 0.20)

    @property
    def progression_rate_elder(self):
        return self._idu_params.get('progression_rate_elder', 0.15)

    @property
    def age_threshold_youth(self):
        return self._idu_params.get('age_threshold_youth', 35)

    @property
    def age_threshold_adult(self):
        return self._idu_params.get('age_threshold_adult', 50)

    def get_immunosuppression_factor(self):
        return {'idu': self.immunosuppression_combined}

    def get_progression_multiplier(self, past_illness_type='idu'):
        if past_illness_type == self.CONTACT_TYPE_IDU:
            return self.immunosuppression_combined
        return 1.0

    def get_age_stratified_progression_rate(self, age):
        """IDU年龄分层进展率

        青年(≤35岁)：30%；成人(36-50岁)：20%；老人(>50岁)：15%
        """
        if age <= self.age_threshold_youth:
            return self.progression_rate_youth
        elif age <= self.age_threshold_adult:
            return self.progression_rate_adult
        else:
            return self.progression_rate_elder

    def calculate_idu_disease_probability(self, infection_probability, age=30, with_hiv=False):
        """计算IDU接触者的发病概率

        参数：
            infection_probability: 感染概率 (%)
            age: 年龄
            with_hiv: 是否合并HIV（加倍风险）

        返回：发病概率 (%)
        """
        base_progress = self.get_age_stratified_progression_rate(age)
        if with_hiv:
            base_progress *= 1.5

        adjusted_progress = base_progress
        disease_prob = min(infection_probability * adjusted_progress, 60.0)
        return disease_prob

    def get_screening_recommendation(self, risk_level='high'):
        recs = {
            'high': '立即IGRA+HIV联合筛查，纳入美沙酮维持治疗评估，建立DOTS管理档案',
            'medium': 'IGRA筛查+HIV自愿检测，纳入社区随访管理',
            'low': '健康教育+定期症状监测，每6个月随访一次'
        }
        return recs.get(risk_level, recs['low'])