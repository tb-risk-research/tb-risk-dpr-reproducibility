#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
克拉玛依本土化适配器（主类）— 深化版 v2.0
"""

import math
import threading

from ..config import _load_config, ConfigProxy
from .calibrator import LocalEpiCalibrator
from .oilfield import OilfieldExposureModel
from .idu import IDURiskModule
from .climate import ClimateTBInteraction
from .altitude import AltitudeAdaptation
from .ethnicity import EthnicityPathogenModule
from .sdoh import SDOHModule
from .policy import LocalPolicyEngine


class KaramayLocalizer:
    """克拉玛依本土化适配器（主类）— 深化版 v2.0

    策略注册与参数提供中心架构：
    - 每个子模块作为"插件"，向外部暴露参数提供器和规则增强器
    - 核心系统在初始化时注册适配器，无需针对克拉玛依分支判断
    - 所有差异封装在内部

    配置管理：
    - 初始化时读取 karamay_local_params.json
    - 文件不存在则使用默认国际参数
    - 模块内无硬编码

    用法示例：
        localizer = KaramayLocalizer.get_instance()

        # 策略注册
        registry = localizer.get_strategy_registry()
        # registry = {
        #   'exposure_setting_factors': {...},
        #   'immunosuppression_factors': {...},
        #   'scenario_defaults': {...},
        #   ...
        # }

        # 规则增强
        enhancer = localizer.get_rule_enhancer('oilfield_exposure')
        adjusted = enhancer(contact_data)
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, config_path=None):
        """初始化所有子模块

        参数：
            config_path (str|None): 配置文件路径，None使用默认
        """
        raw_config = _load_config(config_path)
        self._config = ConfigProxy(raw_config)

        # 八个子模块（全部从配置驱动）
        self.epi = LocalEpiCalibrator(self._config)
        self.oilfield = OilfieldExposureModel(self._config)
        self.idu = IDURiskModule(self._config)
        self.climate = ClimateTBInteraction(self._config)
        self.altitude = AltitudeAdaptation(self._config)
        self.ethnicity = EthnicityPathogenModule(self._config)
        self.sdoh = SDOHModule(self._config)
        self.policy = LocalPolicyEngine(self._config)

        self.enabled = True
        self._strategy_registry = None
        self._rule_enhancers = None

        # L2: 初始化时执行贝叶斯校准（默认路径）
        # 贝叶斯校准失败时自动回退到硬编码公式，不影响系统可用性
        self._init_bayesian_calibration(raw_config)

    def _init_bayesian_calibration(self, raw_config):
        """初始化时执行贝叶斯校准（L2：默认路径）。

        从配置中读取观测数据，运行 MCMC 推断本地化参数的后验分布。
        成功后 self.epi._calibration_result 被填充，后续
        get_infection_probability_baseline_offset 将使用后验均值。
        失败时自动回退到硬编码公式，记录警告但不中断初始化。
        """
        try:
            # 从配置加载观测数据
            observed_data = self._load_observed_data_from_config(raw_config)
            if observed_data is None:
                return  # 配置中无观测数据，跳过校准

            # 执行贝叶斯校准
            result = self.epi.run_bayesian_calibration(
                localizer=self,
                observed_data=observed_data,
                n_iterations=2000,  # 初始化时使用较少的迭代
                n_burnin=1000,
                n_chains=2,
                random_seed=42,
            )
            if result is not None:
                import logging
                logging.getLogger("tb_risk.karamay").info(
                    "贝叶斯校准完成: base_incidence=%.1f/10万, "
                    "oilfield_camp=%.3f, 收敛=%s",
                    result.posterior_mean.get('base_incidence_per_100k', 121.0),
                    result.posterior_mean.get('oilfield_camp_factor', 0.85),
                    result.converged,
                )
        except Exception as e:
            import logging
            logging.getLogger("tb_risk.karamay").warning(
                "贝叶斯校准失败，回退到硬编码公式: %s", e)

    @staticmethod
    def _load_observed_data_from_config(raw_config):
        """从配置中加载观测数据（各区县筛查阳性率）。

        配置路径: karamay_local_params.json → bayesian_calibration.observed_data
        若配置中不存在，返回 None 跳过校准。

        返回:
            dict or None: 区县名 → 阳性率(%) 的映射
        """
        bc_config = raw_config.get('bayesian_calibration', {})
        observed = bc_config.get('observed_data', {})
        if not observed or not isinstance(observed, dict):
            return None
        # 验证数据格式
        valid = {}
        for district, rate in observed.items():
            try:
                rate = float(rate)
                # 允许 rate == 0（某区县确实未检出阳性），仅排除负数与 >=100 的非法值
                if 0 <= rate < 100:
                    valid[district] = rate
            except (ValueError, TypeError):
                pass
        return valid if valid else None

    @classmethod
    def get_instance(cls, config_path=None):
        """获取单例实例（线程安全）

        参数：
            config_path (str|None): 首次初始化时使用的配置文件路径
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(config_path=config_path)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        with cls._instance_lock:
            cls._instance = None

    def reload_config(self, config_path=None):
        """热更新配置（支持不重启程序重新加载）

        重新加载外部 JSON 配置并通知所有子模块更新参数。
        临床使用中很重要，例如季节性参数调整后可立即生效。

        参数：
            config_path (str|None): 配置文件路径，None 使用默认路径

        返回：
            bool: 是否重载成功
        """
        from ..config import reload_config
        try:
            new_cfg, success = reload_config(config_path, localizer_instance=self)
            if success:
                # 清除缓存，强制下次访问时重建
                self._strategy_registry = None
                self._rule_enhancers = None
                return True
            return False
        except Exception as e:
            import logging
            logging.getLogger("tb_risk.karamay").warning("配置热更新失败: %s", e)
            return False

    def _reinit_modules(self, raw_config):
        """重新初始化所有子模块（配置热更新时调用）

        参数：
            raw_config (dict): 新的配置字典
        """
        from ..config import ConfigProxy
        self._config = ConfigProxy(raw_config)
        from .calibrator import LocalEpiCalibrator
        from .oilfield import OilfieldExposureModel
        from .idu import IDURiskModule
        from .climate import ClimateTBInteraction
        from .altitude import AltitudeAdaptation
        from .ethnicity import EthnicityPathogenModule
        from .sdoh import SDOHModule
        from .policy import LocalPolicyEngine
        self.epi = LocalEpiCalibrator(self._config)
        self.oilfield = OilfieldExposureModel(self._config)
        self.idu = IDURiskModule(self._config)
        self.climate = ClimateTBInteraction(self._config)
        self.altitude = AltitudeAdaptation(self._config)
        self.ethnicity = EthnicityPathogenModule(self._config)
        self.sdoh = SDOHModule(self._config)
        self.policy = LocalPolicyEngine(self._config)
        # 清除缓存
        self._strategy_registry = None
        self._rule_enhancers = None
        # 重新执行贝叶斯校准
        self._init_bayesian_calibration(raw_config)

    def is_enabled(self): return self.enabled
    def enable(self): self.enabled = True
    def disable(self): self.enabled = False

    # ==================== 策略注册中心接口 ====================

    def get_strategy_registry(self):
        """获取策略注册中心

        返回一个字典，包含所有可注入到核心系统的策略参数。
        核心系统在初始化时调用此方法即可批量注册。

        返回：
            dict: {
                'exposure_setting_factors': dict,
                'immunosuppression_factors': dict,
                'scenario_defaults': dict,
                'localized_params': dict,
                'counterfactual_interventions': list
            }
        """
        if self._strategy_registry is not None:
            return self._strategy_registry

        self._strategy_registry = {
            'exposure_setting_factors': self.oilfield.get_extended_exposure_factors(),
            'immunosuppression_factors': self.idu.get_immunosuppression_factor(),
            'scenario_defaults': self.get_all_scenario_defaults(),
            'localized_params': {
                'synthetic_data': self.epi.calibrate_synthetic_params(),
                'infection_probability_offset': self.epi.get_infection_probability_baseline_offset(),
            },
            'counterfactual_interventions': self.policy.get_counterfactual_interventions(),
            'policy_summary': self.policy.get_policy_summary(),
            'references': self._config.get_list('references', []),
        }
        return self._strategy_registry

    # ==================== 规则增强器接口 ====================

    def _build_rule_enhancers(self):
        """构建规则增强器映射表"""
        return {
            'oilfield_exposure': self._enhance_oilfield_exposure,
            'idu_risk': self._enhance_idu_risk,
            'climate_beta': self._enhance_climate_beta,
            'altitude_adaptation': self._enhance_altitude,
            'sdoh': self._enhance_sdoh,
            'policy_screening': self._enhance_policy_screening,
        }

    def get_rule_enhancer(self, enhancer_name):
        """获取指定规则增强器回调函数

        参数：
            enhancer_name (str): 增强器名称，如 'oilfield_exposure'

        返回：
            callable|None: 增强器回调函数
        """
        if self._rule_enhancers is None:
            self._rule_enhancers = self._build_rule_enhancers()
        return self._rule_enhancers.get(enhancer_name)

    def _enhance_oilfield_exposure(self, contact_data, month=None):
        """油田暴露规则增强器

        在 _generate_potential_patients 中，对油田营地接触者：
          1. 覆盖 setting_factor
          2. 轮班去重调整累积暴露
          3. 冬季修正通风评分
        """
        if not self.enabled:
            return contact_data

        exposure_setting = contact_data.get('exposure_setting', '')
        if exposure_setting not in ('oilfield_camp', 'oilfield_shift'):
            return contact_data

        result = dict(contact_data)
        result['exposure_setting_factor'] = self.oilfield.oilfield_camp_factor

        crew_type = contact_data.get('crew_type', 'maintenance')
        ratio = self.oilfield.get_continuous_ratio(crew_type)
        if 'cumulative_exposure' in result:
            result['cumulative_exposure'] *= ratio

        if 'ventilation' in result:
            original_vent = result['ventilation']
            result['ventilation'] = self.oilfield.apply_winter_ventilation_penalty(
                original_vent, month
            )

        return result

    @staticmethod
    def _normalize_idu_status(value):
        """将idu_status规范化为布尔值（兼容中文'是'/'否'）"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip() == '是'
        if isinstance(value, (int, float)):
            return value == 1
        return bool(value)

    def _enhance_idu_risk(self, contact_data):
        """IDU风险规则增强器

        对 idu_status=True 的接触者：
          1. 替换进展率倍乘为 idu_combined
          2. 年龄分层进展率
          3. 治疗依从性惩罚
        """
        if not self.enabled:
            return contact_data

        if not self._normalize_idu_status(contact_data.get('idu_status', False)):
            return contact_data

        result = dict(contact_data)
        result['progression_multiplier'] = self.idu.immunosuppression_combined
        result['age_stratified_progression'] = self.idu.get_age_stratified_progression_rate(
            contact_data.get('age', 30)
        )
        result['adherence_penalty'] = self.idu.adherence_penalty
        result['contact_subtype'] = 'idu'
        return result

    def _enhance_climate_beta(self, beta_base, pm10=None, humidity=None,
                                 month=None, is_indoor=False):
        """气候β乘数增强器

        在 assess_risk 的 beta 计算中注入：
          beta = beta_base × climate_multiplier
        """
        if not self.enabled:
            return beta_base

        multiplier = self.climate.calculate_climate_multiplier(
            pm10=pm10, humidity=humidity, month=month, is_indoor=is_indoor
        )
        return beta_base * multiplier

    def _enhance_altitude(self, contact_data):
        """海拔适应规则增强器

        对迁入自高海拔的接触者调整 progression_multiplier。
        """
        if not self.enabled:
            return contact_data

        origin_alt = contact_data.get('origin_altitude')
        if origin_alt is None:
            return contact_data

        months = contact_data.get('months_since_migration', 0)
        adj = self.altitude.get_progression_adjustment(origin_alt, months)
        if not adj['applicable']:
            return contact_data

        result = dict(contact_data)
        existing = result.get('progression_multiplier', 1.0)
        result['progression_multiplier'] = existing * adj['factor']
        result['altitude_adjustment'] = adj
        return result

    def _enhance_sdoh(self, contact_data):
        """社会决定因素（SDOH）规则增强器

        替代原民族遗传易感性增强器。
        基于可测量的结构性变量（住房拥挤、医疗可及性、
        营养风险、收入分层）调整 progression_multiplier。

        遵循 Belmont Report 公正原则：不基于民族身份进行风险区分。
        文献：WHO SDOH Framework (Lönnroth 2009); Lönnroth 2010; Belmont Report 1979.
        """
        if not self.enabled:
            return contact_data

        sdoh_result = self.sdoh.compute_sdoh_multiplier(contact_data)
        if math.isclose(sdoh_result['multiplier'], 1.0, rel_tol=1e-9):
            return contact_data

        result = dict(contact_data)
        existing = result.get('progression_multiplier', 1.0)
        result['progression_multiplier'] = existing * sdoh_result['multiplier']
        result['sdoh_adjustment'] = sdoh_result
        return result

    def _enhance_policy_screening(self, risk_level, district, is_high_risk):
        """政策筛查建议增强器"""
        if not self.enabled:
            if risk_level == 'high':
                return '立即进行胸部X线/痰涂片检查'
            elif risk_level == 'medium':
                return '进行PPD/IGRA筛查'
            return '定期观察'
        return self.policy.get_screening_recommendation(risk_level, district, is_high_risk)

    # ==================== 合并注入接口 ====================

    def get_all_scenario_defaults(self):
        scenarios = {}
        scenarios.update(self.oilfield.get_karamay_oilfield_scenario())
        epi_defaults = self.epi.get_scenario_defaults()
        for sk in scenarios:
            scenarios[sk]['patient'].update(epi_defaults)
        return scenarios

    def get_extended_exposure_factors(self):
        return self.oilfield.get_extended_exposure_factors()

    def get_extended_immunosuppression_factors(self):
        return self.idu.get_immunosuppression_factor()

    def get_counterfactual_interventions(self):
        return self.policy.get_counterfactual_interventions()

    def enhance_recommendation(self, contact_info):
        return self.policy.enhance_recommendation(contact_info)

    def get_combined_progression_multiplier(self, contact_data=None,
                                               origin_altitude=None,
                                               months_since_migration=0,
                                               ethnicity='han',
                                               strain_type='beijing'):
        """综合本土化进展率乘数（深度计算）

        按顺序应用: IDU → 海拔 → SDOH（社会决定因素）

        注意：ethnicity 和 strain_type 参数保留仅用于向后兼容，
        不参与风险计算。民族身份仅作为描述性统计变量。
        风险评估使用 SDOH 模块（住房拥挤、医疗可及性、营养风险、收入分层）。
        """
        if not self.enabled:
            return {'multiplier': 1.0, 'components': {}, 'description': '本土化模块未启用'}

        multiplier = 1.0
        components = {}
        descriptions = []

        if contact_data:
            if self._normalize_idu_status(contact_data.get('idu_status', False)):
                idu_f = self.idu.immunosuppression_combined
                multiplier *= idu_f
                components['idu'] = idu_f
                descriptions.append(f'IDU组合因子 ×{idu_f:.1f}')

        if origin_altitude is not None:
            alt_r = self.altitude.get_progression_adjustment(origin_altitude, months_since_migration)
            if alt_r['applicable']:
                multiplier *= alt_r['factor']
                components['altitude'] = alt_r['factor']
                descriptions.append(alt_r['description'])

        # SDOH（社会决定因素）替代原民族遗传易感性
        # 使用可测量的结构性变量：住房拥挤、医疗可及性、营养风险、收入分层
        sdoh_result = self.sdoh.compute_sdoh_multiplier(contact_data)
        if not math.isclose(sdoh_result['multiplier'], 1.0, rel_tol=1e-9):
            multiplier *= sdoh_result['multiplier']
            components['sdoh'] = sdoh_result['multiplier']
            components['sdoh_details'] = sdoh_result['components']
            descriptions.append(sdoh_result['description'])

        return {
            'multiplier': multiplier,  # 允许双向调整（<1.0 降低风险，>1.0 增加风险）
            'components': components,
            'description': '; '.join(descriptions) if descriptions else '无本土化调整'
        }

    def get_climate_defaults(self):
        """获取气候环境默认参数（PM10, humidity, month）"""
        baseline_pm10 = self._config.get('climate.baseline.pm10', 100)
        baseline_humidity = self._config.get('climate.baseline.humidity', 45)
        monthly = self._config.get_dict('climate.monthly_profile')
        if monthly:
            pm10_values = [v.get('pm10_avg', baseline_pm10) for v in monthly.values() if isinstance(v, dict)]
            humidity_values = [v.get('humidity_avg', baseline_humidity) for v in monthly.values() if isinstance(v, dict)]
            annual_mean_pm10 = sum(pm10_values) / len(pm10_values) if pm10_values else baseline_pm10
            annual_mean_humidity = sum(humidity_values) / len(humidity_values) if humidity_values else baseline_humidity
        else:
            annual_mean_pm10 = baseline_pm10
            annual_mean_humidity = baseline_humidity
        return {
            'pm10': baseline_pm10,
            'humidity': baseline_humidity,
            'annual_mean_pm10': annual_mean_pm10,
            'annual_mean_humidity': annual_mean_humidity,
        }

    # ==================== 信息接口 ====================

    def get_module_info(self):
        cfg_meta = self._config.get_dict('_meta')
        return {
            'name': 'KaramayLocalizer',
            'version': cfg_meta.get('version', '2.0'),
            'description': '克拉玛依本土化适配器 - 策略注册中心架构 - 七维度深化版',
            'dimensions': {
                '一': 'LocalEpiCalibrator（发病率121/10万, σ²ω≈1.806）',
                '二': 'OilfieldExposureModel（轮班去重, 冬季通风修正）',
                '三': 'IDURiskModule（组合因子10.0, 年龄分层进展率）',
                '四': 'ClimateTBInteraction（加法模型, PM10+湿度协同）',
                '五': 'AltitudeAdaptation（1.3+0.1×km, 半衰期5年）',
                '六': 'SDOHModule（社会决定因素：住房拥挤、医疗可及性、营养风险、收入分层）',
                '七': 'LocalPolicyEngine（收治率66.1%, 关爱行动试点）'
            },
            'enabled': self.enabled,
            'config_loaded': bool(self._config._data),
            'references_count': len(self._config.get_list('references', []))
        }

    def get_all_references(self):
        return self._config.get_list('references', [])