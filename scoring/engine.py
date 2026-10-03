import math

try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False
    np = None

# 统一常量源（单一真值源）
from tb_risk.constants import (
    SYMPTOMS_BONUS, PAST_TB_BONUS, NO_BCG_BONUS,
    DISTANCE_FACTORS, VENTILATION_FACTORS, SETTING_FACTORS_BASE,
    AGE_PROGRESSION, IMMUNO_FACTORS_BASE,
    LATENT_BASELINE,
    SIGMOID_COEFF, SIGMOID_OFFSET, SIGMOID_CAP, SIGMOID_CLIP_THRESHOLD,
    PROGRESSION_SCALE_FACTOR, MAX_PROGRESSION_RATE,
    EXPOSURE_SATURATION_HOURS, EXPOSURE_HIGH_HOURS, EXPOSURE_MEDIUM_HOURS,
    MAX_DISEASE_PROBABILITY, RISK_THRESHOLD_HIGH,
    DEFAULT_DISTANCE_FACTOR, DEFAULT_VENTILATION_FACTOR, DEFAULT_SETTING_FACTOR,
    FOCAL_LOSS_ALPHA, FOCAL_LOSS_GAMMA,
    # 双输出进展建模
    LATENT_SHORT_TERM_BASELINE,
    TIME_DECAY_FACTORS, TIME_DECAY_LONG_TERM_FACTOR, TIME_DECAY_DEFAULT_MONTHS,
    IGRA_POSITIVE_PROB, IGRA_NEGATIVE_REMAIN,
    IGRA_INDETERMINATE_PRIOR, IGRA_NOT_DONE_PRIOR_CAP,
    RECENT_CONVERSION_BOOST,
    SMOKING_PROGRESSION_FACTOR,
    BMI_PROGRESSION_FACTOR, BMI_DEFAULT,
    VITAMIN_D_PROGRESSION_FACTOR, VITAMIN_D_DEFAULT,
)


class ScoringEngine:
    """评分计算引擎

    将接触者数据转换为统一风险评分，支持本土化/非本土化两种模式。
    模拟 TB_Risk_Assessment._generate_potential_patients 的核心逻辑。

    所有常量定义在 tb_risk.constants 中（单一真值源）。
    """

    # 评分常数（从 tb_risk.constants 引用，保证唯一定义）
    SYMPTOMS_BONUS = SYMPTOMS_BONUS
    PAST_TB_BONUS = PAST_TB_BONUS
    NO_BCG_BONUS = NO_BCG_BONUS

    DISTANCE_FACTORS = DISTANCE_FACTORS
    VENTILATION_FACTORS = VENTILATION_FACTORS
    SETTING_FACTORS_BASE = SETTING_FACTORS_BASE

    AGE_PROGRESSION = AGE_PROGRESSION
    IMMUNO_FACTORS_BASE = IMMUNO_FACTORS_BASE

    LATENT_BASELINE = LATENT_BASELINE

    # 双输出进展建模常量
    LATENT_SHORT_TERM_BASELINE = LATENT_SHORT_TERM_BASELINE
    TIME_DECAY_FACTORS = TIME_DECAY_FACTORS
    TIME_DECAY_LONG_TERM_FACTOR = TIME_DECAY_LONG_TERM_FACTOR
    TIME_DECAY_DEFAULT_MONTHS = TIME_DECAY_DEFAULT_MONTHS
    IGRA_POSITIVE_PROB = IGRA_POSITIVE_PROB
    IGRA_NEGATIVE_REMAIN = IGRA_NEGATIVE_REMAIN
    IGRA_INDETERMINATE_PRIOR = IGRA_INDETERMINATE_PRIOR
    IGRA_NOT_DONE_PRIOR_CAP = IGRA_NOT_DONE_PRIOR_CAP
    RECENT_CONVERSION_BOOST = RECENT_CONVERSION_BOOST
    SMOKING_PROGRESSION_FACTOR = SMOKING_PROGRESSION_FACTOR
    BMI_PROGRESSION_FACTOR = BMI_PROGRESSION_FACTOR
    BMI_DEFAULT = BMI_DEFAULT
    VITAMIN_D_PROGRESSION_FACTOR = VITAMIN_D_PROGRESSION_FACTOR
    VITAMIN_D_DEFAULT = VITAMIN_D_DEFAULT

    SIGMOID_COEFF = SIGMOID_COEFF
    SIGMOID_OFFSET = SIGMOID_OFFSET
    SIGMOID_CAP = SIGMOID_CAP
    SIGMOID_CLIP_THRESHOLD = SIGMOID_CLIP_THRESHOLD
    PROGRESSION_SCALE_FACTOR = PROGRESSION_SCALE_FACTOR
    MAX_PROGRESSION_RATE = MAX_PROGRESSION_RATE
    EXPOSURE_SATURATION_HOURS = EXPOSURE_SATURATION_HOURS
    EXPOSURE_HIGH_HOURS = EXPOSURE_HIGH_HOURS
    EXPOSURE_MEDIUM_HOURS = EXPOSURE_MEDIUM_HOURS
    MAX_DISEASE_PROBABILITY = MAX_DISEASE_PROBABILITY
    RISK_THRESHOLD_HIGH = RISK_THRESHOLD_HIGH

    DEFAULT_DISTANCE_FACTOR = DEFAULT_DISTANCE_FACTOR
    DEFAULT_VENTILATION_FACTOR = DEFAULT_VENTILATION_FACTOR
    DEFAULT_SETTING_FACTOR = DEFAULT_SETTING_FACTOR
    FOCAL_LOSS_ALPHA = FOCAL_LOSS_ALPHA
    FOCAL_LOSS_GAMMA = FOCAL_LOSS_GAMMA

    def __init__(self, localizer=None, use_localization=True):
        self.localizer = localizer
        self.use_localization = use_localization and localizer is not None
        self._param_overrides = {}   # 参数覆盖字典，用于敏感性分析
        self._original_values = {}   # 保存原始值，用于恢复

    # ==================== 参数注入机制 ====================

    def apply_param_overrides(self, overrides):
        """注入参数覆盖，用于敏感性分析和贝叶斯校准。

        支持覆盖的键（与 localizer 子模块参数对应）：
        - 'oilfield_camp_factor': float — 油田营地暴露因子
        - 'idu_combined_factor': float — IDU 组合免疫抑制因子
        - 'pm10_effect_per_50ug': float — PM10 效应
        - 'humidity_effect_per_10': float — 干旱湿度效应
        - 'altitude_base_additive': float — 海拔基线加数
        - 'altitude_per_km_additive': float — 海拔每公里加数
        - 'altitude_threshold_m': float — 高海拔阈值
        - 'sdoh_combined_factor': float — SDOH 组合乘数（住房/医疗/营养/收入）
        - 'centralized_trt_rate': float — 集中收治率
        - 'base_incidence_per_100k': float — 本土发病率基线
        - 'LATENT_BASELINE': float — 潜伏基线
        - 'PROGRESSION_SCALE_FACTOR': float — 进展缩放因子
        - 'SIGMOID_COEFF': float — Sigmoid 系数
        - 'SIGMOID_OFFSET': float — Sigmoid 偏移
        - 'SIGMOID_CAP': float — Sigmoid 上限
        - 'SETTING_FACTORS_BASE': dict — 场景因子完整覆盖
        - 'IMMUNO_FACTORS_BASE': dict — 免疫因子完整覆盖

        参数：
            overrides: dict, 键为参数名，值为覆盖值
        """
        self._param_overrides = dict(overrides)
        self._original_values = {}

        # 为简单类型参数创建实例属性覆盖（影子类属性）
        simple_keys = ['LATENT_BASELINE', 'PROGRESSION_SCALE_FACTOR',
                       'SIGMOID_COEFF', 'SIGMOID_OFFSET', 'SIGMOID_CAP']
        for key in simple_keys:
            if key in overrides:
                self._original_values[key] = getattr(self, key, None)
                setattr(self, key, overrides[key])

        # 为字典类型参数创建浅拷贝覆盖
        dict_keys = ['SETTING_FACTORS_BASE', 'IMMUNO_FACTORS_BASE']
        for key in dict_keys:
            if key in overrides:
                original = getattr(self, key, None)
                if original is not None:
                    self._original_values[key] = dict(original)
                setattr(self, key, dict(overrides[key]))

    def reset_param_overrides(self):
        """重置所有参数覆盖，恢复原始值。"""
        for key, value in self._original_values.items():
            setattr(self, key, value)
        self._original_values.clear()
        self._param_overrides.clear()

    def compute_risk_score(self, record):
        """计算单条记录的综合风险评分 (0-100)

        返回：
            dict: {
                'disease_probability': float,
                'infection_probability': float,
                'risk_components': dict,
                'progression_multiplier': float,
            }
        """
        r = record
        risk_score = 0.0
        components = {}

        # 1. 症状
        if r.get('has_symptoms', 0):
            risk_score += self.SYMPTOMS_BONUS
            components['symptoms'] = self.SYMPTOMS_BONUS

        # 2. 既往TB史
        if r.get('has_tb', 0):
            risk_score += self.PAST_TB_BONUS
            components['past_tb'] = self.PAST_TB_BONUS

        # 3. 未接种BCG（保守默认：未知=未接种=加分）
        if not r.get('bcg_vaccine', 0):
            risk_score += self.NO_BCG_BONUS
            components['no_bcg'] = self.NO_BCG_BONUS

        # 4. 暴露场景因子（参数覆盖优先：oilfield_camp_factor 可覆盖）
        exposure_setting = r.get('exposure_setting', 'general')
        setting_factor = self.SETTING_FACTORS_BASE.get(
            exposure_setting, self.DEFAULT_SETTING_FACTOR
        )
        if self.use_localization and exposure_setting == 'oilfield_camp':
            if 'oilfield_camp_factor' in self._param_overrides:
                setting_factor = self._param_overrides['oilfield_camp_factor']
            else:
                setting_factor = self.localizer.oilfield.oilfield_camp_factor

        # 5. 构建最终感染/发病概率
        dist_factor = self.DISTANCE_FACTORS.get(
            r.get('contact_distance', 'medium'), self.DEFAULT_DISTANCE_FACTOR
        )
        try:
            vent_raw = int(float(r.get('ventilation', 3)))
        except (ValueError, TypeError):
            vent_raw = 3
        vent_factor = self.VENTILATION_FACTORS.get(
            vent_raw, self.DEFAULT_VENTILATION_FACTOR
        )

        cumulative = r.get('cumulative_exposure', 0)
        exposure_risk = self._exposure_to_risk(cumulative)
        risk_score += exposure_risk

        # 数值稳定的 sigmoid：裁剪 z 防止 math.exp() 溢出
        z = self.SIGMOID_COEFF * (risk_score - self.SIGMOID_OFFSET)
        z_clipped = max(-self.SIGMOID_CLIP_THRESHOLD, min(self.SIGMOID_CLIP_THRESHOLD, -z))
        base_infection = min(
            self.SIGMOID_CAP / (1.0 + math.exp(z_clipped)),
            self.SIGMOID_CAP
        )
        infection_prob = (
            base_infection * dist_factor * vent_factor * setting_factor
        )
        infection_prob = min(infection_prob, 100.0)

        # 进展率
        prog_mult = 1.0
        prog_components = {}

        past_illness_type = str(r.get('past_illness_type', 'none')).strip().lower()
        if past_illness_type == 'hiv':
            prog_mult = self.IMMUNO_FACTORS_BASE.get('hiv', 8.0)
            prog_components['immuno_hiv'] = prog_mult
        elif past_illness_type == 'immunosuppressants':
            prog_mult = self.IMMUNO_FACTORS_BASE.get('immunosuppressants', 4.0)
            prog_components['immuno_suppressants'] = prog_mult
        elif past_illness_type == 'diabetes':
            prog_mult = self.IMMUNO_FACTORS_BASE.get('diabetes', 2.5)
            prog_components['immuno_diabetes'] = prog_mult
        elif past_illness_type == 'other':
            prog_mult = self.IMMUNO_FACTORS_BASE.get('other', 1.8)
            prog_components['immuno_other'] = prog_mult

        # 年龄进展因子
        age = r.get('age', 30)
        try:
            age = float(age)
        except (TypeError, ValueError):
            age = 30
        age_factor = 1.0
        if age < 5:
            age_factor = self.AGE_PROGRESSION['child_under_5']
        elif 5 <= age < 15:
            age_factor = self.AGE_PROGRESSION['child_5_14']
        elif 15 <= age <= 35:
            age_factor = self.AGE_PROGRESSION['young']
        elif 35 < age <= 65:
            age_factor = self.AGE_PROGRESSION['adult']
        elif age > 65:
            age_factor = self.AGE_PROGRESSION['elderly']
        prog_components['age'] = age_factor

        combined_risk = prog_mult * age_factor

        # 本土化增强（参数覆盖优先：检查 _param_overrides）
        # 乘数必须在缩放公式内部施加（与 assessment.py 保持一致），
        # 即 combined_risk * adj_mult 后再套入 LATENT_BASELINE * (1 + (cr - 1) * sf)，
        # 而非 adjusted_progress *= adj_mult 的外部乘法。
        # 文献：Anderson & May (1991), Bjørnstad et al. (2002) 参数缩放层次分析。
        adj_mult = 1.0
        if self.use_localization and self.localizer is not None:
            has_overrides = bool(self._param_overrides)

            if has_overrides and any(k in self._param_overrides for k in [
                    'idu_combined_factor', 'altitude_base_additive',
                    'altitude_per_km_additive', 'altitude_threshold_m',
                    'sdoh_combined_factor']):
                # 使用参数覆盖构建本土化乘数（绕过 localizer 子模块）
                adj_mult = self._compute_localization_multiplier_from_overrides(r)
            else:
                # 正常路径：委托给 localizer
                idu_raw = r.get('idu_status', False)
                idu_normalized = idu_raw if isinstance(idu_raw, bool) else (str(idu_raw).strip() == '是')
                combined = self.localizer.get_combined_progression_multiplier(
                    contact_data={
                        'idu_status': idu_normalized,
                        'ethnicity': r.get('ethnicity', 'han'),
                    },
                    origin_altitude=r.get('origin_altitude'),
                    months_since_migration=r.get('months_since_migration', 0),
                    ethnicity=r.get('ethnicity', 'han'),
                    strain_type='beijing',
                )
                extra_mult = combined.get('multiplier', 1.0)
                adj_mult = extra_mult  # 允许双向调整（与 localizer.py 一致）

            prog_components['localization'] = adj_mult

            # 气候β增强（参数覆盖优先）
            if has_overrides and any(k in self._param_overrides for k in [
                    'pm10_effect_per_50ug', 'humidity_effect_per_10']):
                climate_beta = self._compute_climate_multiplier_from_overrides(r)
            else:
                climate_beta = self.localizer.climate.calculate_climate_multiplier(
                    pm10=r.get('pm10', 100), humidity=r.get('humidity', 45),
                    month=r.get('month', 4), is_indoor=True,
                )
            infection_prob *= climate_beta
            # climate_beta 上限 3.0，乘后可能超过 100%。必须在乘法后再次钳制，
            # 否则返回的 infection_probability 可达 300%（与 assessment.py:2154
            # `adj_infection = min(infection_prob * climate_beta, 100.0)` 保持一致）。
            infection_prob = max(0.0, min(infection_prob, 100.0))
            prog_components['climate_beta'] = climate_beta

        # 将本土化乘数并入 combined_risk（与 assessment.py 公式结构一致）。
        # 文献：Anderson & May (1991), Bjørnstad et al. (2002) 参数缩放层次分析。
        combined_risk_total = combined_risk * adj_mult
        # 保留旧"固定基线 × 静态乘数"的 adjusted_progress（向后兼容，供 progression 组件展示）
        adjusted_progress = min(
            self.LATENT_BASELINE * (1.0 + (combined_risk_total - 1.0) * self.PROGRESSION_SCALE_FACTOR),
            self.MAX_PROGRESSION_RATE
        )

        # ─── 双输出进展建模（实验室证据 / 时间维度 / 感染门槛） ───
        # 旧结构是 infection_prob × adjusted_progress，缺三个维度：
        #   1) 无"感染时间"维度（固定基线无法表达"刚阳转 vs 多年前感染"）；
        #   2) 年度基线与终生 10% 混用（高估单年进展，见 constants 标定说明）；
        #   3) 无"是否已感染"门槛（未感染者也拿到 LATENT_BASELINE）。
        # 新结构拆成两个概率：
        #   P_infected  = 用 IGRA/TST 判定感染状态（未检测则回退暴露先验）
        #   P_progress  = 给定已感染后的 2 年短程进展概率（时间衰减 × 年龄/免疫/生活方式）
        # disease_probability 改由 P_infected × P_progress 驱动（未感染者自动逼近 0）。
        # 惰性导入避免 tb_risk.core 包初始化时的循环依赖
        from tb_risk.core.lab_evidence import (
            latent_infection_probability,
            short_term_progression_probability,
            time_decay_factor,
            derive_lab_evidence_grade,
        )

        # P_infected：IGRA 判定感染状态；未检测时用现有暴露推断（infection_prob/100）
        # 作暴露先验，保证"未检测"路径与旧 exposure 推断连续。
        exposure_prior = max(0.0, min(infection_prob / 100.0, 1.0))
        latent_infection_prob = latent_infection_probability(r, exposure_prior)

        # P_progress：短程进展基线 × 时间衰减 × 年龄/免疫乘数 × (1−BCG保护) × 生活方式
        # age_multiplier 用 combined_risk_total（含本土化 adj_mult，与旧结构一致）：
        # 批量路径在 use_localization=False 时 adj_mult=1.0，两路径保持一致。
        prog_components['time_decay'] = None
        short_term_progress = short_term_progression_probability(
            r,
            age_multiplier=combined_risk_total,
            immuno_multiplier=1.0,          # prog_mult 已并入 combined_risk，不再重复
            bcg_protective=0.0,
            cap=self.MAX_PROGRESSION_RATE,
        )

        # 近期(1–2年)发病风险：P_infected × P_progress
        short_term_active_risk = latent_infection_prob * short_term_progress
        disease_prob = max(0.0, min(
            short_term_active_risk * 100.0, self.MAX_DISEASE_PROBABILITY))

        # 诊断组件：记录 IGRA / 时间衰减 / 生活方式中间量
        igra_result = str(r.get('igra_result', 'not_done')).strip().lower()
        prog_components['igra'] = igra_result
        prog_components['latent_infection_prob'] = round(latent_infection_prob, 4)
        prog_components['short_term_progress'] = round(short_term_progress, 4)
        prog_components['time_decay'] = round(
            time_decay_factor(r.get('time_since_exposure_months')), 4)
        lab_evidence_grade = derive_lab_evidence_grade(r)

        # 与 disease_probability 计算保持一致：adj_mult 已并入 combined_risk_total
        #（combined_risk_total = combined_risk * adj_mult），乘数在公式内部而非外部。
        # SDOH 保护性因子 < 1.0 时也能正确反映在 progression_multiplier 中。
        final_prog_multiplier = combined_risk * adj_mult

        return {
            'disease_probability': disease_prob,
            'infection_probability': infection_prob,
            'risk_score': risk_score,
            'setting_factor': setting_factor,
            'progression_multiplier': final_prog_multiplier,
            'latent_infection_prob': latent_infection_prob,
            'short_term_active_risk': short_term_active_risk,
            'risk_components': components,
            'progression_components': prog_components,
            'lab_evidence_grade': lab_evidence_grade,
        }

    @staticmethod
    def _exposure_to_risk(cumulative_hours):
        # L 级修复：阈值引用 constants 单一真值源（与批量路径
        # _compute_risk_scores 的 EXPOSURE_* 类常量同源），消除
        # 标量/批量两处独立维护导致的分裂风险
        if cumulative_hours >= EXPOSURE_SATURATION_HOURS:
            return 4.0
        elif cumulative_hours >= EXPOSURE_HIGH_HOURS:
            return 3.0
        elif cumulative_hours >= EXPOSURE_MEDIUM_HOURS:
            return 2.0
        elif cumulative_hours > 0:
            return 1.0
        return 0.0

    def _compute_localization_multiplier_from_overrides(self, record):
        """使用参数覆盖构建本土化进展乘数（绕过 localizer 子模块）。

        复制 KaramayLocalizer.get_combined_progression_multiplier 的核心逻辑，
        但所有参数从 self._param_overrides 读取，实现完全可控的敏感性分析。

        参数：
            record: dict, 接触者记录

        返回：
            float: 本土化进展乘数 (≥1.0)
        """
        multiplier = 1.0
        ov = self._param_overrides

        # IDU 免疫抑制因子
        idu_val = record.get('idu_status', False)
        idu_bool = idu_val if isinstance(idu_val, bool) else (str(idu_val).strip() == '是')
        if idu_bool and 'idu_combined_factor' in ov:
            multiplier *= ov['idu_combined_factor']

        # 海拔适应因子
        origin_alt = record.get('origin_altitude')
        if origin_alt is not None:
            alt_thr = ov.get('altitude_threshold_m', 1500)
            if origin_alt > alt_thr:
                alt_base = ov.get('altitude_base_additive', 1.3)
                alt_pkm = ov.get('altitude_per_km_additive', 0.1)
                descent_km = (origin_alt - 350) / 1000.0  # KARAMAY_BASE_ALTITUDE
                alt_factor = alt_base + alt_pkm * descent_km
                alt_factor = min(2.0, max(1.0, alt_factor))
                months = record.get('months_since_migration', 0)
                years = months / 12.0
                decay = 0.5 ** (years / 5.0)
                alt_factor = 1.0 + (alt_factor - 1.0) * decay
                multiplier *= max(1.0, alt_factor)

        # SDOH（社会决定因素）组合乘数
        # 替代原民族遗传易感性逻辑（E1 修复）
        # 文献：WHO SDOH Framework (Lönnroth 2009); Belmont Report 1979
        sdoh_factor = ov.get('sdoh_combined_factor', 1.0)
        multiplier *= sdoh_factor

        return multiplier  # 允许双向调整（与 localizer.py 一致）

    def _compute_climate_multiplier_from_overrides(self, record):
        """使用参数覆盖构建气候乘数（绕过 localizer.climate 子模块）。

        参数：
            record: dict, 接触者记录

        返回：
            float: 气候乘数 (0.3 - 3.0)
        """
        ov = self._param_overrides
        pm10 = record.get('pm10', 100)
        humidity = record.get('humidity', 45)

        # 基线 PM10 (默认 100 µg/m³，克拉玛依基准)
        baseline_pm10 = 100.0
        pm10_effect = ov.get('pm10_effect_per_50ug', 0.15)

        climate_m = 1.0
        if pm10 > baseline_pm10:
            climate_m += ((pm10 - baseline_pm10) / 50.0) * pm10_effect

        # 干旱效应：湿度 < 30% 时每降 10% 增加传播风险
        humidity_effect = ov.get('humidity_effect_per_10', 0.10)
        if humidity < 30:
            steps = math.ceil((30 - humidity) / 10.0)
            climate_m += steps * humidity_effect

        return max(0.3, min(3.0, climate_m))

    def compute_batch(self, records):
        """批量计算风险评分（向量化加速，需numpy）

        参数：
            records: list[dict] 或 dict-of-lists，每个元素为一条接触者记录

        返回：
            list[dict]: 各记录的风险评分结果
        """
        if not _NUMPY_OK or not records:
            return [self.compute_risk_score(r) for r in records]

        # 本土化模式守卫：向量化路径尚未覆盖 climate_beta、本土化进展乘数（adj_mult，
        # 含 IDU/海拔/SDOH）与 oilfield_camp localizer 覆盖。若直接向量化会与
        # compute_risk_score 产生系统性分歧（批量路径偏低）。为正确性回退到标量路径。
        # 详见 compute_risk_score 第 243-286 行的本土化增强逻辑。
        if self.use_localization:
            return [self.compute_risk_score(r) for r in records]

        n = len(records)

        # 提取数值数组（age 可能来自 CSV 字符串，需安全转换）
        def _safe_age(r):
            a = r.get('age', 30)
            try:
                return float(a)
            except (TypeError, ValueError):
                return 30.0

        ages = np.array([_safe_age(r) for r in records], dtype=np.float64)
        has_symptoms = np.array([bool(r.get('has_symptoms', 0)) for r in records])
        has_tb = np.array([bool(r.get('has_tb', 0)) for r in records])
        bcg = np.array([bool(r.get('bcg_vaccine', 0)) for r in records])

        # 风险评分
        risk_scores = np.zeros(n, dtype=np.float64)
        risk_scores += np.where(has_symptoms, self.SYMPTOMS_BONUS, 0.0)
        risk_scores += np.where(has_tb, self.PAST_TB_BONUS, 0.0)
        risk_scores += np.where(~bcg, self.NO_BCG_BONUS, 0.0)

        # 距离/通风/场景因子
        # 默认值必须与标量路径 compute_risk_score 一致：
        #   DISTANCE_FACTORS.get(..., DEFAULT_DISTANCE_FACTOR)、
        #   VENTILATION_FACTORS.get(..., DEFAULT_VENTILATION_FACTOR)、
        #   SETTING_FACTORS_BASE.get(..., DEFAULT_SETTING_FACTOR)。旧实现用 np.ones(1.0)
        #   初始化，对未识别值会高估，导致两路径分歧。
        dist_factor = np.full(n, self.DEFAULT_DISTANCE_FACTOR, dtype=np.float64)
        for k, v in self.DISTANCE_FACTORS.items():
            mask = np.array([r.get('contact_distance', 'medium') == k for r in records])
            dist_factor = np.where(mask, v, dist_factor)

        vent_factor = np.full(n, self.DEFAULT_VENTILATION_FACTOR, dtype=np.float64)
        for k, v in self.VENTILATION_FACTORS.items():
            mask = np.array([int(r.get('ventilation', 3)) == k for r in records])
            vent_factor = np.where(mask, v, vent_factor)

        setting_factor = np.full(n, self.DEFAULT_SETTING_FACTOR, dtype=np.float64)
        for k, v in self.SETTING_FACTORS_BASE.items():
            mask = np.array([r.get('exposure_setting', 'general') == k for r in records])
            setting_factor = np.where(mask, v, setting_factor)

        # 累积暴露风险（阈值与 _exposure_to_risk 保持一致）
        cumul = np.array([r.get('cumulative_exposure', 0) for r in records], dtype=np.float64)
        exposure_risk = np.where(cumul >= self.EXPOSURE_SATURATION_HOURS, 4.0,
                        np.where(cumul >= self.EXPOSURE_HIGH_HOURS, 3.0,
                        np.where(cumul >= self.EXPOSURE_MEDIUM_HOURS, 2.0,
                        np.where(cumul > 0, 1.0, 0.0))))
        risk_scores += exposure_risk

        # sigmoid感染概率（使用类常量，与标量路径保持一致；裁剪 z 防止 np.exp 溢出）
        z = -self.SIGMOID_COEFF * (risk_scores - self.SIGMOID_OFFSET)
        z_clipped = np.clip(z, -self.SIGMOID_CLIP_THRESHOLD, self.SIGMOID_CLIP_THRESHOLD)
        base_infection = np.clip(
            self.SIGMOID_CAP / (1.0 + np.exp(z_clipped)),
            0.0, self.SIGMOID_CAP
        )
        infection_probs = np.clip(base_infection * dist_factor * vent_factor * setting_factor, 0.0, 100.0)

        # 进展率（与标量路径 compute_risk_score 保持一致，包含 'other'）
        prog_mult = np.ones(n, dtype=np.float64)
        illness_types = np.array([str(r.get('past_illness_type', 'none')).strip().lower() for r in records])
        for k in ['hiv', 'immunosuppressants', 'diabetes', 'other']:
            if k in self.IMMUNO_FACTORS_BASE:
                prog_mult = np.where(illness_types == k, self.IMMUNO_FACTORS_BASE[k], prog_mult)

        # 年龄进展因子
        age_factor = np.ones(n, dtype=np.float64)
        age_factor = np.where(ages < 5, self.AGE_PROGRESSION['child_under_5'], age_factor)
        age_factor = np.where((ages >= 5) & (ages < 15), self.AGE_PROGRESSION['child_5_14'], age_factor)
        age_factor = np.where((ages >= 15) & (ages <= 35), self.AGE_PROGRESSION['young'], age_factor)
        age_factor = np.where((ages > 35) & (ages <= 65), self.AGE_PROGRESSION['adult'], age_factor)
        age_factor = np.where(ages > 65, self.AGE_PROGRESSION['elderly'], age_factor)

        combined_risk = prog_mult * age_factor
        # 保留旧"固定基线 × 静态乘数"的 adjusted_progress（向后兼容参考量）
        adjusted_progress = np.clip(self.LATENT_BASELINE * (1.0 + (combined_risk - 1.0) * self.PROGRESSION_SCALE_FACTOR), 0.0, self.MAX_PROGRESSION_RATE)

        # ─── 双输出进展建模（与标量路径 compute_risk_score 保持一致） ───
        def _f(v, default):
            try:
                x = float(v)
            except (TypeError, ValueError):
                x = float(default)
            return x

        # P_infected：IGRA 判定感染状态；未检测时用暴露先验（infection_prob/100）
        igra_results = [str(r.get('igra_result', 'not_done')).strip().lower() for r in records]
        exposure_prior = np.clip(infection_probs / 100.0, 0.0, 1.0)
        latent_infected = np.full(n, self.IGRA_INDETERMINATE_PRIOR, dtype=np.float64)
        pos_mask = np.array([x in ('positive', 'pos', '阳性', '阳', '是') for x in igra_results])
        neg_mask = np.array([x in ('negative', 'neg', '阴性', '阴', '否') for x in igra_results])
        indet_mask = np.array([x in ('indeterminate', 'indet', '不确定', '可疑', 'borderline') for x in igra_results])
        not_done_mask = ~(pos_mask | neg_mask | indet_mask)
        latent_infected = np.where(pos_mask, self.IGRA_POSITIVE_PROB, latent_infected)
        latent_infected = np.where(neg_mask, self.IGRA_NEGATIVE_REMAIN, latent_infected)
        latent_infected = np.where(
            indet_mask, np.clip(np.maximum(exposure_prior, self.IGRA_INDETERMINATE_PRIOR), 0.0, self.IGRA_POSITIVE_PROB),
            latent_infected)
        latent_infected = np.where(
            not_done_mask, np.clip(exposure_prior, 0.0, self.IGRA_NOT_DONE_PRIOR_CAP),
            latent_infected)

        # time_decay：距感染时间（缺失默认 24 月 → 0.5 档）
        # 降序遍历分档上界，使 t 取"最接近的小档"（与标量 time_decay_factor 一致）：
        # 例如 t=24 命中 0.5 而非被更大档 60→0.2 覆盖。
        t_months = np.array([_f(r.get('time_since_exposure_months'), self.TIME_DECAY_DEFAULT_MONTHS) for r in records])
        t_months = np.where(t_months < 0, self.TIME_DECAY_DEFAULT_MONTHS, t_months)
        time_decay = np.full(n, self.TIME_DECAY_LONG_TERM_FACTOR, dtype=np.float64)
        for cutoff in sorted(self.TIME_DECAY_FACTORS, reverse=True):
            time_decay = np.where(t_months <= cutoff, self.TIME_DECAY_FACTORS[cutoff], time_decay)

        # 生活方式乘数：smoking × BMI 档 × 维生素 D 档
        smoking = np.array([bool(r.get('smoking', False)) for r in records])
        lifestyle = np.where(smoking, self.SMOKING_PROGRESSION_FACTOR, 1.0)
        bmis = np.array([_f(r.get('bmi'), self.BMI_DEFAULT) for r in records])
        lifestyle = np.where(bmis < 18.5, lifestyle * self.BMI_PROGRESSION_FACTOR['low_underweight'], lifestyle)
        lifestyle = np.where((bmis >= 18.5) & (bmis < 22), lifestyle * self.BMI_PROGRESSION_FACTOR['low_normal'], lifestyle)
        lifestyle = np.where((bmis >= 22) & (bmis < 25), lifestyle * self.BMI_PROGRESSION_FACTOR['normal'], lifestyle)
        lifestyle = np.where((bmis >= 25) & (bmis < 30), lifestyle * self.BMI_PROGRESSION_FACTOR['overweight'], lifestyle)
        lifestyle = np.where(bmis >= 30, lifestyle * self.BMI_PROGRESSION_FACTOR['obese'], lifestyle)
        vds = np.array([_f(r.get('vitamin_d'), self.VITAMIN_D_DEFAULT) for r in records])
        lifestyle = np.where(vds < 12, lifestyle * self.VITAMIN_D_PROGRESSION_FACTOR['deficient'], lifestyle)
        lifestyle = np.where((vds >= 12) & (vds < 20), lifestyle * self.VITAMIN_D_PROGRESSION_FACTOR['insufficient'], lifestyle)
        lifestyle = np.where((vds >= 20) & (vds < 40), lifestyle * self.VITAMIN_D_PROGRESSION_FACTOR['normal'], lifestyle)
        lifestyle = np.where(vds >= 40, lifestyle * self.VITAMIN_D_PROGRESSION_FACTOR['optimal'], lifestyle)

        # recent_conversion 前置放大（近期阳转）
        conversion_boost = np.where(
            np.array([bool(r.get('recent_conversion', False)) for r in records]),
            1.0 + self.RECENT_CONVERSION_BOOST, 1.0)

        # P_progress(1–2y | infected) = 0.03 × decay × combined_risk × lifestyle × boost（cap 0.30）
        # 注：prog_mult/age_factor 已并入 combined_risk；BCG 保护项 (1−protective)=1.0
        short_term_progress = np.clip(
            self.LATENT_SHORT_TERM_BASELINE * time_decay * combined_risk
            * lifestyle * conversion_boost,
            0.0, self.MAX_PROGRESSION_RATE)

        # 近期(1–2年)发病风险 = P_infected × P_progress；disease_prob 由它驱动
        short_term_active_risk = latent_infected * short_term_progress
        disease_probs = np.clip(short_term_active_risk * 100.0, 0.0, self.MAX_DISEASE_PROBABILITY)

        # 组装结果
        results = []
        for i in range(n):
            results.append({
                'disease_probability': float(disease_probs[i]),
                'infection_probability': float(infection_probs[i]),
                'risk_score': float(risk_scores[i]),
                'setting_factor': float(setting_factor[i]),
                'progression_multiplier': float(combined_risk[i]),
                'latent_infection_prob': float(latent_infected[i]),
                'short_term_active_risk': float(short_term_active_risk[i]),
            })
        return results