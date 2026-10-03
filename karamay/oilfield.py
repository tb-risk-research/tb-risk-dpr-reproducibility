#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
维度二：油田职业暴露模型（深化版）
"""

import datetime


class OilfieldExposureModel:
    """维度二：油田职业暴露模型（深化版）

    输入：接触者作业类型和轮班周期
    输出：
    - 暴露场景因子 oilfield_camp=0.85（介于closed和crowded之间）
    - 轮班去重算法: effective_hours = total_hours × continuous_ratio
      continuous_ratio根据班次模式动态确定(0.65-0.80)
    - 冬季修正：11-2月野外作业通风评分下降1个等级

    关联点：
    - _generate_potential_patients中识别oilfield_camp，覆盖setting_factor
    - 累积暴露计算前对time_span和freq_density进行轮班去重调整
    """

    def __init__(self, config):
        self._cfg = config

    @property
    def oilfield_camp_factor(self):
        return self._cfg.get('occupation.oilfield_camp_factor', 0.85)

    @property
    def dust_factor_default(self):
        return self._cfg.get('occupation.dust_factor_default', 1.25)

    @property
    def winter_months(self):
        return self._cfg.get_list('occupation.winter_months', [11, 12, 1, 2])

    @property
    def winter_ventilation_penalty(self):
        return self._cfg.get('occupation.winter_ventilation_penalty', 1)

    def _get_shift_pattern(self, crew_type):
        patterns = self._cfg.get_dict('occupation.shift_patterns')
        default = {'work_days': 14, 'rest_days': 7, 'continuous_ratio': 0.75}
        pattern = patterns.get(crew_type, default)
        return self._validate_shift_pattern(pattern)

    @staticmethod
    def _validate_shift_pattern(pattern):
        """校验轮班参数并在越界时记录警告。

        continuous_ratio 的合理范围为 0.65-0.80（参见类文档字符串）。
        若配置值超出此范围，则钳制到最近边界，防止不合理暴露折算。
        """
        import logging
        validated = dict(pattern)
        ratio = float(validated.get('continuous_ratio', 0.75))
        if ratio < 0.65 or ratio > 0.80:
            logging.getLogger('tb_risk.karamay').warning(
                "continuous_ratio %.3f 超出合理范围 [0.65, 0.80]，已钳制", ratio)
            ratio = min(max(ratio, 0.65), 0.80)
            validated['continuous_ratio'] = ratio
        return validated

    def get_exposure_setting_factor(self, setting='oilfield_camp'):
        if setting == 'oilfield_camp':
            return self.oilfield_camp_factor
        return 1.0

    def get_extended_exposure_factors(self):
        """返回扩展暴露场景因子字典，可直接合并"""
        return {'oilfield_camp': self.oilfield_camp_factor}

    def calculate_shift_adjusted_exposure(self, total_hours, crew_type='maintenance'):
        """轮班去重算法：effective_hours = total_hours × continuous_ratio

        将非连续轮班周期的有效接触总时长折算为连续等价时长。
        continuous_ratio根据班次模式动态确定，范围0.65-0.80。
        """
        pattern = self._get_shift_pattern(crew_type)
        ratio = pattern.get('continuous_ratio', 0.75)
        return total_hours * ratio

    def get_continuous_ratio(self, crew_type='maintenance'):
        pattern = self._get_shift_pattern(crew_type)
        return pattern.get('continuous_ratio', 0.75)

    def apply_dust_correction(self, base_factor, pm10_value=None):
        """应用戈壁扬尘修正因子"""
        if pm10_value is None:
            return base_factor * self.dust_factor_default

        levels = self._cfg.get_dict('occupation.dust_exposure_levels')
        if not levels:
            return base_factor * self.dust_factor_default

        level_order = ['low', 'moderate', 'high', 'extreme']
        for lv in level_order:
            if lv not in levels:
                continue
            threshold = levels[lv].get('pm10_threshold', 9999)
            if pm10_value <= threshold:
                return base_factor * levels[lv].get('factor', self.dust_factor_default)
        return base_factor * self.dust_factor_default

    def apply_winter_ventilation_penalty(self, ventilation_score, month=None):
        """冬季修正：11-2月野外作业通风评分下降1个等级

        参数：
            ventilation_score (int|str): 原始通风评分(1-5)
            month (int|None): 当前月份，None则使用当前系统月份

        返回：
            int: 修正后的通风评分
        """
        try:
            ventilation_score = int(ventilation_score)
        except (ValueError, TypeError):
            ventilation_score = 3

        if month is None:
            month = datetime.datetime.now().month

        if month in self.winter_months:
            penalty = self.winter_ventilation_penalty
            return max(1, ventilation_score - penalty)

        return ventilation_score

    def is_winter(self, month=None):
        if month is None:
            month = datetime.datetime.now().month
        return month in self.winter_months

    def get_karamay_oilfield_scenario(self):
        """返回'克拉玛依油田'场景的完整配置"""
        scenario_cfg = self._cfg.get_dict('occupation.karamay_oilfield_scenario')
        if not scenario_cfg:
            scenario_cfg = {
                'patient': {
                    'ventilation': 2, 'family_living_conditions': 1,
                    'flp_percentage': 15.0, 'hrsp_percentage': 25.0
                },
                'family_member': {
                    'contact_distance': 'close', 'ventilation': 2,
                    'exposure_setting': 'oilfield_camp',
                    'freq_density': 21, 'single_duration': 240, 'time_span': 3
                },
                'social_contact': {
                    'contact_distance': 'medium', 'ventilation': 3,
                    'exposure_setting': 'oilfield_camp',
                    'freq_density': 7, 'single_duration': 480, 'time_span': 3
                }
            }
        return {
            'karamay_oilfield': {
                'name': '克拉玛依油田',
                'patient': dict(scenario_cfg.get('patient', {})),
                'family_member': dict(scenario_cfg.get('family_member', {})),
                'social_contact': dict(scenario_cfg.get('social_contact', {}))
            }
        }

    def calculate_oilfield_exposure(self, single_duration_min, freq_per_week,
                                      time_span_weeks, crew_type='maintenance',
                                      with_dust=True, pm10=None):
        base_exposure = (single_duration_min / 60.0) * freq_per_week * time_span_weeks
        adjusted = self.calculate_shift_adjusted_exposure(base_exposure, crew_type)
        cf = self.oilfield_camp_factor
        result = {
            'base_cumulative_exposure': base_exposure,
            'shift_adjusted_exposure': adjusted,
            'exposure_setting': 'oilfield_camp',
            'exposure_factor': cf,
            'crew_type': crew_type,
            'continuous_ratio': self.get_continuous_ratio(crew_type)
        }
        result['dust_corrected_factor'] = (
            self.apply_dust_correction(cf, pm10) if with_dust else cf
        )
        return result