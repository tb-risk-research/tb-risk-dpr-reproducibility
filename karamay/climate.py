#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
维度四：干旱气候-气溶胶交互模型（深化版）
"""

import math
import datetime as dt_lib


class ClimateTBInteraction:
    """维度四：干旱气候-气溶胶交互模型（深化版）

    输入：外部环境API或手动输入的月均PM2.5/PM10/湿度/温度
    输出：
    - climate_multiplier = 1.0 + PM10_factor + humidity_factor（加法模型）
      PM10每增50μg/m³ → +0.15（参考喀什研究）
      湿度<30%每降10% → +0.10（基于飞沫核蒸发模型）
    - 室内气候乘数折半（油田营地粉尘侵入但室内）
    - 季节调整表：月均参数预置

    关联点：
    - assess_risk开始前计算乘数注入到beta系数:
      beta = beta_base × climate_multiplier
    - SEIR曲线绘制"有无气候修正"对比
    """

    def __init__(self, config):
        self._cfg = config

    @staticmethod
    def _safe_float(value, default=None):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _baseline(self, key, default=0):
        return self._cfg.get(f'climate.baseline.{key}', default)

    @property
    def baseline_pm10(self): return self._baseline('pm10', 100.0)
    @property
    def baseline_pm25(self): return self._baseline('pm25', 35.0)
    @property
    def baseline_humidity(self): return self._baseline('humidity', 45.0)
    @property
    def baseline_temp(self): return self._baseline('temperature', 8.0)
    @property
    def baseline_wind(self): return self._baseline('wind_speed', 3.5)

    def calculate_climate_multiplier(self, pm10=None, pm25=None,
                                        humidity=None, temp=None, wind_speed=None,
                                        month=None, is_indoor=False, is_outdoor=False):
        """计算气候-传播综合乘数

        公式: climate_multiplier = 1.0 + PM10_factor + humidity_factor
        附加: 冬季供暖 + PM2.5 + 风稀释 + 室内折半
        """
        pm10_val = pm10 if pm10 is not None else self.baseline_pm10
        pm25_val = pm25 if pm25 is not None else self.baseline_pm25
        humidity_val = humidity if humidity is not None else self.baseline_humidity
        wind_val = wind_speed if wind_speed is not None else self.baseline_wind

        multiplier = 1.0
        component_descriptions = []

        # 1. PM10因子：每增加50μg/m³，传播力 +0.15
        pm10_per_50 = self._cfg.get('climate.pm10_effect_per_50ug', 0.15)
        pm10_above_baseline = pm10_val - self.baseline_pm10
        if pm10_above_baseline > 0:
            steps = pm10_above_baseline / 50.0
            pm10_factor = steps * pm10_per_50
            multiplier += pm10_factor
            component_descriptions.append(f'PM10({pm10_val:.0f}μg/m³)+{pm10_factor:.3f}')

        # 2. PM2.5超额附加
        pm25_threshold = self.baseline_pm25
        if pm25_val > pm25_threshold:
            pm25_extra = self._cfg.get('climate.pm25_additional_factor', 0.10)
            multiplier += pm25_extra
            component_descriptions.append(f'PM2.5+{pm25_extra:.3f}')

        # 3. 湿度因子：低于30%时，每降10%湿度 +0.10
        humidity_threshold = self._cfg.get('climate.humidity_threshold_low', 30.0)
        humidity_per_10 = self._cfg.get('climate.humidity_effect_per_10pct', 0.10)
        if humidity_val < humidity_threshold:
            deficit = humidity_threshold - humidity_val
            steps = math.ceil(deficit / 10.0)
            humidity_factor = steps * humidity_per_10
            multiplier += humidity_factor
            component_descriptions.append(f'干旱(RH={humidity_val:.0f}%)+{humidity_factor:.3f}')

        # 4. 冬季供暖
        is_heating = False
        if month is not None:
            monthly = self._cfg.get_dict('climate.monthly_profile')
            mkey = str(int(month))
            if mkey in monthly and monthly[mkey].get('is_heating', False):
                is_heating = True
        if is_heating:
            heat_factor = self._cfg.get('climate.winter_heating_factor', 1.15)
            extra = heat_factor - 1.0
            multiplier += extra
            component_descriptions.append(f'供暖+{extra:.3f}')

        # 5. 高风速稀释（户外）
        wind_threshold = self._cfg.get('climate.wind_dilution_threshold', 5.0)
        if is_outdoor and wind_val > wind_threshold:
            dil = self._cfg.get('climate.wind_dilution_factor', 0.80)
            reduction = 1.0 - dil
            multiplier -= reduction
            component_descriptions.append(f'风稀释-{reduction:.3f}')

        # 6. 室内折半
        indoor_fraction = self._cfg.get('climate.indoor_climate_fraction', 0.5)
        if is_indoor:
            excess = multiplier - 1.0
            if excess > 0:
                multiplier = 1.0 + excess * indoor_fraction
                component_descriptions.append('室内折半')

        multiplier = max(0.3, min(3.0, multiplier))
        return multiplier

    def get_seasonal_profile(self):
        return self._cfg.get_dict('climate.monthly_profile')

    def get_profile_for_month(self, month):
        monthly = self._cfg.get_dict('climate.monthly_profile')
        mkey = str(int(month))
        return monthly.get(mkey, {})

    def get_daily_climate_multiplier(self, date=None, pm10=None, pm25=None,
                                      humidity=None, temp=None, wind_speed=None,
                                      is_indoor=False, is_outdoor=False,
                                      **kwargs):
        """获取指定日期的气候传播乘数

        优先使用传入的实时环境数据，若无则回退到月均值。
        支持 datetime.date 或 ISO 格式字符串 'YYYY-MM-DD'。

        参数：
            date: datetime.date 或 ISO字符串，None 则使用今天
            pm10/pm25/humidity/temp/wind_speed: 实时观测值（可选）
            is_indoor/is_outdoor: 室内/室外标记
            **kwargs: 其他环境参数

        返回：
            dict: {
                'date': 日期字符串,
                'month': 月份,
                'climate_multiplier': 乘数值,
                'pm10_value': 使用的PM10值,
                'humidity_value': 使用的湿度值,
                'season': 季节,
                'is_heating': 是否供暖季,
                'daily_risk_score': 当日环境风险评分(0-1),
                'components': 乘数组成说明
            }
        """
        if date is None:
            parsed_date = dt_lib.date.today()
        elif isinstance(date, str):
            try:
                parsed_date = dt_lib.datetime.strptime(date, '%Y-%m-%d').date()
            except ValueError:
                try:
                    parsed_date = dt_lib.datetime.strptime(date, '%Y/%m/%d').date()
                except ValueError:
                    parsed_date = dt_lib.date.today()
        elif isinstance(date, dt_lib.date):
            parsed_date = date
        elif isinstance(date, dt_lib.datetime):
            parsed_date = date.date()
        else:
            parsed_date = dt_lib.date.today()

        month = parsed_date.month

        profile = self.get_profile_for_month(month)
        season = profile.get('season', 'unknown')
        is_heating = profile.get('is_heating', False)

        if pm10 is None:
            pm10 = profile.get('pm10_avg', self.baseline_pm10)
        if pm25 is None:
            pm25 = kwargs.get('pm25_real', profile.get('pm25_avg', self.baseline_pm25))
        if humidity is None:
            humidity = profile.get('humidity_avg', self.baseline_humidity)
        if temp is None:
            temp = profile.get('temp_avg', self.baseline_temp)

        multiplier = self.calculate_climate_multiplier(
            pm10=pm10, pm25=pm25, humidity=humidity,
            temp=temp, wind_speed=wind_speed,
            month=month, is_indoor=is_indoor,
            is_outdoor=is_outdoor)

        # 风险评分必须在 [0, 1] 区间内；旧实现未对 multiplier < 0.7 做下界防护，
        # 当 multiplier=0.3 时 risk_score 为负数，与文档“0-1 环境风险评分”矛盾。
        risk_score = max(0.0, min(1.0, (multiplier - 0.7) / 1.3))

        # wind_speed 为 0 时不应因 Python falsy 被误判为 None；改为显式判 None。
        wind_speed_value = self._safe_float(wind_speed) if wind_speed is not None else None

        return {
            'date': parsed_date.isoformat(),
            'month': month,
            'climate_multiplier': round(float(multiplier), 4),
            'pm10_value': self._safe_float(pm10),
            'pm25_value': self._safe_float(pm25),
            'humidity_value': self._safe_float(humidity),
            'temperature_value': self._safe_float(temp),
            'wind_speed_value': wind_speed_value,
            'season': season,
            'is_heating': bool(is_heating),
            'daily_risk_score': round(risk_score, 4),
            'components': f'M={multiplier:.3f}',
        }

    def get_environment_feature_vector(self, date=None, pm10=None,
                                        humidity=None, temp=None,
                                        wind_speed=None):
        """获取标准化的环境特征向量 [4维] 用于多模态GNN

        返回：
            numpy array: [pm10_norm, humidity_norm, temp_norm, wind_norm]
        """
        daily = self.get_daily_climate_multiplier(
            date=date, pm10=pm10, humidity=humidity,
            temp=temp, wind_speed=wind_speed)

        pm10_norm = min(1.0, daily['pm10_value'] / 300.0)
        humidity_norm = daily['humidity_value'] / 100.0
        temp_norm = (daily['temperature_value'] + 20.0) / 60.0
        # 风速 0 与 None 语义不同：0 应保持 0，None 才回退默认值 3.5。
        ws = daily['wind_speed_value']
        wind_norm = min(1.0, (ws if ws is not None else 3.5) / 15.0)

        try:
            import numpy as np
            return np.array([pm10_norm, humidity_norm, temp_norm, wind_norm],
                           dtype=np.float32)
        except ImportError:
            return [pm10_norm, humidity_norm, temp_norm, wind_norm]