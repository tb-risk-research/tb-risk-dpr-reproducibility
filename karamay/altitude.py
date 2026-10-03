#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
维度五：海拔适应性模块（深化版）
"""

import math


class AltitudeAdaptation:
    """维度五：海拔适应性模块（深化版）

    输入：接触者原籍海拔
    输出：
    - multiplier = 1.3 + 0.1 × descent_km（上限2.0）
      仅当原籍海拔 > threshold=1500m 时触发
    - 迁入时间衰减：半衰期5年

    关联点：
    - _generate_potential_patients中读取迁移史，
      计算调整后progression_multiplier（乘在原进展率上）
    """

    def __init__(self, config):
        self._cfg = config

    @property
    def karamay_altitude_m(self):
        return self._cfg.get('altitude.karamay_altitude_m', 350)

    @property
    def high_altitude_threshold_m(self):
        return self._cfg.get('altitude.high_altitude_threshold_m', 1500)

    @property
    def descent_base_additive(self):
        return self._cfg.get('altitude.descent_base_additive', 1.3)

    @property
    def descent_per_km_additive(self):
        return self._cfg.get('altitude.descent_per_km_additive', 0.1)

    @property
    def formula_upper_limit(self):
        return self._cfg.get('altitude.formula_upper_limit', 2.0)

    @property
    def time_decay_half_life_years(self):
        return self._cfg.get('altitude.time_decay_half_life_years', 5.0)

    def is_high_altitude_immigrant(self, origin_altitude):
        if origin_altitude is None:
            return False
        return origin_altitude > self.high_altitude_threshold_m

    def calculate_descent_multiplier(self, origin_altitude):
        """海拔下降效应乘数

        公式: multiplier = 1.3 + 0.1 × descent_km，上限2.0
              descent_km = (origin_altitude - karamay_altitude) / 1000
        """
        if origin_altitude <= self.high_altitude_threshold_m:
            return 1.0

        descent_m = origin_altitude - self.karamay_altitude_m
        if descent_m <= 0:
            return 1.0

        descent_km = descent_m / 1000.0
        multiplier = self.descent_base_additive + self.descent_per_km_additive * descent_km
        return min(multiplier, self.formula_upper_limit)

    def apply_time_decay(self, multiplier, years_since_migration):
        """应用迁入时间衰减

        半衰期公式: decayed = 1.0 + (multiplier-1.0) × (0.5)^(years/half_life)
        years_since_migration < 0 时视为 0（不衰减）
        """
        years_since_migration = max(0, years_since_migration)
        if years_since_migration == 0 or multiplier <= 1.0:
            return multiplier

        half_life = self.time_decay_half_life_years
        decay_factor = math.pow(0.5, years_since_migration / half_life)
        excess = multiplier - 1.0
        decayed = 1.0 + excess * decay_factor
        return max(1.0, decayed)

    def get_progression_adjustment(self, origin_altitude, months_since_migration=0):
        """获取海拔适应的完整进展率调整

        返回：
            dict: {'applicable', 'factor', 'description'}
        """
        if not self.is_high_altitude_immigrant(origin_altitude):
            return {
                'applicable': False,
                'factor': 1.0,
                'description': '原籍海拔未超过阈值，不适用海拔调整'
            }

        raw_multiplier = self.calculate_descent_multiplier(origin_altitude)
        years = months_since_migration / 12.0
        final_multiplier = self.apply_time_decay(raw_multiplier, years)

        descent_m = origin_altitude - self.karamay_altitude_m
        desc = (
            f'原海拔 {origin_altitude:.0f}m → 克拉玛依 {self.karamay_altitude_m:.0f}m，'
            f'海拔差 {descent_m:.0f}m ({descent_m/1000:.1f}km)，'
            f'初始乘数 {raw_multiplier:.2f}，'
            f'迁入 {months_since_migration:.0f} 月({years:.1f}年)，'
            f'衰减后因子 {final_multiplier:.2f}x'
        )
        return {
            'applicable': True,
            'factor': final_multiplier,
            'raw_multiplier': raw_multiplier,
            'description': desc
        }