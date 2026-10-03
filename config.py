#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
克拉玛依本土化适配器模块配置管理

架构模式：策略注册中心 (Strategy Registration Center)
  - 每个子模块作为"插件"向外部暴露参数提供器和规则增强器
  - 核心系统仅需在初始化时注册适配器，无需分支判断
  - 所有差异封装在模块内部，实现高内聚、低耦合

配置管理：所有克拉玛依本地参数存于外部文件 karamay_local_params.json
  - 若文件不存在或加载失败，回退到默认国际参数
  - 模块内无硬编码

作者：郭臻尧
版本：2.0（深化版）

文献支撑：
- 克拉玛依卫健委, 2024（检出率121/10万、收治率66.1%）
- Liu et al., Chin Gen Pract, 2025（空间模型 σ²ω≈1.806）
- 吐哈油田公司, 2025; 新医大公卫学院, 2025（油田职业暴露）
- 中国防痨协会, 2006; 结核病高危人群, 2023（IDU人群）
- Front Public Health, 2025喀什研究; bioRxiv, 2025（气候-传播）
- Acta Med Port, 2018; Evol Med Public Health, 2022（海拔适应性）
- 中国防痨杂志, 2025; Front Public Health, 2025（民族-病原体）
- 中国CDC结核病中心, 2024; 克拉玛依网, 2025（地方政策）
"""

import copy
import datetime
import json
import logging
import os

from .constants import ETHNICITY_WEIGHTS as _ETHNICITY_WEIGHTS

# ==============================================================================
# 克拉玛依本土化模块 - 内嵌代码块 (原 karamay_localizer.py + karamay_local_params.json)
# 策略：优先读取外部JSON文件覆盖，若不存在则使用下方内嵌配置
# 版本：v2.0（深化版） | 作者：郭臻尧 | 内嵌日期：2026-05-03
# ==============================================================================

_EMBEDDED_KARAMAY_CONFIG = {
    "_meta": {
        "name": "KaramayLocalizer",
        "version": "2.0",
        "description": "克拉玛依本土化适配器 - 外部参数配置文件",
        "last_updated": "2026-05-02",
        "note": "若此文件不存在或加载失败，所有模块将回退到默认国际参数。"
    },
    "epidemiology": {
        "base_incidence_per_100k": 121.0,
        "base_incidence_note": "克拉玛依2024年筛查检出率约121/10万，来源：克拉玛依卫健委2024",
        "latent_prevalence_family_min": 0.08,
        "latent_prevalence_family_max": 0.12,
        "latent_prevalence_family_default": 0.1,
        "latent_prevalence_note": "家庭潜伏感染比例8-12%，基于本地IGRA调查数据",
        "hrsp_percentage_default": 18.0,
        "hrsp_percentage_note": "社会高风险人群潜伏感染比例，油田作业人员和流动人口较高",
        "spatial_correlation_sigma2_omega": 1.806,
        "spatial_correlation_note": "空间自相关系数σ²ω，来源：Liu et al., Chin Gen Pract, 2025空间模型",
        "diagnosis_delay_median_days": 29,
        "diagnosis_delay_note": "就诊延迟中位数约29天，用于校准SEIR模型暴露-发病延迟分布",
        "treatment_success_rate": 0.92,
        "dr_resistance_rate": 0.08,
        "local_age_distribution": {
            "child_under_5": 0.03,
            "child_5_14": 0.05,
            "young_adult": 0.45,
            "adult": 0.35,
            "elderly": 0.12
        }
    },
    "occupation": {
        "oilfield_camp_factor": 0.85,
        "oilfield_camp_note": "油田营地暴露因子，介于general(0.6)和closed(0.9)之间，叠加油气挥发和戈壁粉尘",
        "dust_factor_default": 1.25,
        "dust_factor_note": "戈壁扬尘默认修正因子",
        "dust_exposure_levels": {
            "low": {
                "pm10_threshold": 50,
                "factor": 1.0
            },
            "moderate": {
                "pm10_threshold": 150,
                "factor": 1.1
            },
            "high": {
                "pm10_threshold": 350,
                "factor": 1.2
            },
            "extreme": {
                "pm10_threshold": 9999,
                "factor": 1.25
            }
        },
        "shift_patterns": {
            "drilling": {
                "work_days": 21,
                "rest_days": 7,
                "continuous_ratio": 0.65
            },
            "maintenance": {
                "work_days": 14,
                "rest_days": 7,
                "continuous_ratio": 0.75
            },
            "office": {
                "work_days": 7,
                "rest_days": 7,
                "continuous_ratio": 0.8
            }
        },
        "shift_effective_hours_formula": "effective_hours = total_hours × continuous_ratio",
        "shift_effective_hours_note": "continuous_ratio根据班次模式动态确定，范围0.65-0.80",
        "winter_months": [
            11,
            12,
            1,
            2
        ],
        "winter_ventilation_penalty": 1,
        "winter_ventilation_note": "冬季野外作业通风评分下降1个等级（如3→2），因门窗关闭、密闭供暖",
        "karamay_oilfield_scenario": {
            "patient": {
                "ventilation": 2,
                "family_living_conditions": 1,
                "flp_percentage": 15.0,
                "hrsp_percentage": 25.0
            },
            "family_member": {
                "contact_distance": "close",
                "ventilation": 2,
                "exposure_setting": "oilfield_camp",
                "freq_density": 21,
                "single_duration": 240,
                "time_span": 3
            },
            "social_contact": {
                "contact_distance": "medium",
                "ventilation": 3,
                "exposure_setting": "oilfield_camp",
                "freq_density": 7,
                "single_duration": 480,
                "time_span": 3
            }
        }
    },
    "climate": {
        "baseline": {
            "pm10": 100.0,
            "pm25": 35.0,
            "humidity": 45.0,
            "temperature": 8.0,
            "wind_speed": 3.5
        },
        "formula": "climate_multiplier = 1.0 + pm10_factor + humidity_factor",
        "formula_note": "加法模型，各因子独立贡献后叠加到基线1.0上",
        "pm10_effect_per_50ug": 0.15,
        "pm10_effect_note": "PM10每增加50 μg/m³，传播力增加0.15，参考：Front Public Health, 2025喀什研究",
        "pm25_additional_factor": 0.1,
        "pm25_effect_note": "PM2.5超过基线时额外附加因子",
        "humidity_threshold_low": 30.0,
        "humidity_threshold_high": 70.0,
        "humidity_effect_per_10pct": 0.1,
        "humidity_effect_note": "相对湿度低于30%时，每降低10%湿度，传播力增加0.10，基于飞沫核蒸发模型，bioRxiv, 2025",
        "winter_heating_factor": 1.15,
        "winter_heating_note": "冬季供暖期室内通风减少+人群聚集，传播风险增加",
        "indoor_climate_fraction": 0.5,
        "indoor_climate_note": "油田营地虽为室内但粉尘侵入，室内气候乘数折半适用",
        "wind_dilution_threshold": 5.0,
        "wind_dilution_factor": 0.8,
        "wind_dilution_note": "高风速（>5m/s）稀释户外气溶胶浓度",
        "monthly_profile": {
            "1": {
                "season": "winter",
                "pm10_avg": 130,
                "humidity_avg": 62,
                "temp_avg": -13,
                "is_heating": True
            },
            "2": {
                "season": "winter",
                "pm10_avg": 120,
                "humidity_avg": 58,
                "temp_avg": -10,
                "is_heating": True
            },
            "3": {
                "season": "spring",
                "pm10_avg": 180,
                "humidity_avg": 35,
                "temp_avg": 5,
                "is_heating": False
            },
            "4": {
                "season": "spring",
                "pm10_avg": 160,
                "humidity_avg": 30,
                "temp_avg": 14,
                "is_heating": False
            },
            "5": {
                "season": "spring",
                "pm10_avg": 120,
                "humidity_avg": 28,
                "temp_avg": 21,
                "is_heating": False
            },
            "6": {
                "season": "summer",
                "pm10_avg": 70,
                "humidity_avg": 32,
                "temp_avg": 26,
                "is_heating": False
            },
            "7": {
                "season": "summer",
                "pm10_avg": 65,
                "humidity_avg": 30,
                "temp_avg": 28,
                "is_heating": False
            },
            "8": {
                "season": "summer",
                "pm10_avg": 60,
                "humidity_avg": 28,
                "temp_avg": 27,
                "is_heating": False
            },
            "9": {
                "season": "autumn",
                "pm10_avg": 75,
                "humidity_avg": 38,
                "temp_avg": 20,
                "is_heating": False
            },
            "10": {
                "season": "autumn",
                "pm10_avg": 90,
                "humidity_avg": 45,
                "temp_avg": 10,
                "is_heating": False
            },
            "11": {
                "season": "autumn",
                "pm10_avg": 110,
                "humidity_avg": 55,
                "temp_avg": 0,
                "is_heating": True
            },
            "12": {
                "season": "winter",
                "pm10_avg": 140,
                "humidity_avg": 60,
                "temp_avg": -8,
                "is_heating": True
            }
        }
    },
    "population": {
        "ethnicity_composition": {
            "_note": "民族构成（描述性统计变量，不参与风险计算）。遵循 Belmont Report 公正原则，不基于民族身份进行风险区分。",
            "han": 0.75,
            "uyghur": 0.15,
            "kazakh": 0.05,
            "hui": 0.03,
            "mongolian": 0.005,
            "other": 0.015
        },
        "ethnicity_labels": {
            "_note": "民族标签映射（描述性用途，不参与风险计算）",
            "han": "汉族",
            "uyghur": "维吾尔族",
            "kazakh": "哈萨克族",
            "hui": "回族",
            "mongolian": "蒙古族",
            "other": "其他"
        },
        "sdoh": {
            "_note": "社会决定因素（SDOH）参数，替代原民族遗传易感性。文献：WHO SDOH框架(Lönnroth 2009); Lönnroth 2010(BMI-TB); Belmont Report 1979。",
            "housing_crowding": {
                "_note": "住房拥挤指数。定义：人均居住面积。克拉玛依以单位公寓为主，平均约30m²/人。",
                "threshold_m2": 15.0,
                "threshold_note": "低于此阈值触发拥挤风险",
                "risk_per_m2": 0.02,
                "risk_per_m2_note": "每减少1m²增2%风险，上限1.3",
                "max_multiplier": 1.3,
                "reference": "WHO SDOH Framework (Lönnroth et al., Soc Sci Med 2009)"
            },
            "healthcare_access": {
                "_note": "医疗可及性指数。定义：距最近结核病定点诊疗机构的距离(km)。克拉玛依市中心医院为市级定点，独山子/白碱滩/乌尔禾有区级诊疗点。",
                "threshold_km": 30.0,
                "threshold_note": "超过此距离触发可及性风险",
                "risk_per_km": 0.005,
                "risk_per_km_note": "每增加1km增0.5%，上限1.25（延迟就诊延长传播窗口）",
                "max_multiplier": 1.25,
                "reference": "Lönnroth et al., Soc Sci Med 2009; 克拉玛依市结核病防治规划2024"
            },
            "nutrition": {
                "_note": "营养风险评分。定义：BMI分层。油田年度体检提供BMI数据。",
                "underweight_threshold": 18.5,
                "underweight_multiplier": 1.3,
                "underweight_note": "BMI < 18.5（低体重）→ 乘数1.3",
                "normal_low": 18.5,
                "normal_high": 25.0,
                "normal_multiplier": 1.0,
                "overweight_multiplier": 1.0,
                "overweight_note": "BMI > 25：保守设为1.0（肥胖保护效应有争议）",
                "reference": "Lönnroth et al., Int J Epidemiol 39(1):149-155, 2010"
            },
            "income": {
                "_note": "收入分层。定义：家庭年收入相对当地中位数的比值。克拉玛依油田职工收入高于新疆平均水平。",
                "poverty_threshold_ratio": 0.6,
                "poverty_threshold_note": "低于当地中位数60%视为贫困线",
                "poverty_multiplier": 1.2,
                "poverty_note": "贫困线以下乘数1.2",
                "reference": "WHO SDOH Framework; 克拉玛依市统计局2024年鉴"
            },
            "combination": {
                "_note": "SDOH因子组合方式：加法模型（非乘法），避免多重叠加导致极端值。",
                "method": "additive",
                "formula": "combined = 1.0 + Σ(factor_i - 1.0)",
                "max_combined": 2.0,
                "max_combined_note": "组合上限2.0",
                "min_combined": 0.8,
                "min_combined_note": "组合下限0.8（保护性因素可降低风险）",
                "reference": "WHO SDOH Framework; Mitchell et al., Annu Rev Stat Appl 2024"
            }
        },
        "special_populations": {
            "idu": {
                "immunosuppression_combined": 10.0,
                "immunosuppression_note": "IDU综合免疫抑制倍乘（营养不良+HIV共感染+免疫损伤），来源：中国防痨协会2006; 结核病高危人群2023",
                "adherence_penalty": 0.6,
                "adherence_note": "治疗依从性惩罚，0%正规疗程完成率，实际传染性 × 0.6",
                "progression_rate_youth": 0.3,
                "progression_rate_adult": 0.2,
                "progression_rate_elder": 0.15,
                "progression_rate_note": "年龄分层年进展率：青年15-35岁30%、成人36-50岁20%、老人>50岁15%",
                "age_threshold_youth": 35,
                "age_threshold_adult": 50,
                "contact_type": "idu"
            }
        }
    },
    "altitude": {
        "karamay_altitude_m": 350,
        "karamay_altitude_note": "克拉玛依市区平均海拔",
        "high_altitude_threshold_m": 1500,
        "high_altitude_note": "高于此值视为中高海拔来源，触发海拔效应",
        "formula": "multiplier = 1.3 + 0.1 × descent_km",
        "formula_upper_limit": 2.0,
        "descent_per_km_additive": 0.1,
        "descent_base_additive": 1.3,
        "time_decay_half_life_years": 5.0,
        "time_decay_note": "随在低海拔居住年数增加，效应逐年衰减，半衰期5年",
        "time_since_migration_factors": {
            "recent": {
                "months_max": 6,
                "factor": 2.5
            },
            "intermediate": {
                "months_max": 24,
                "factor": 1.8
            },
            "long_term": {
                "months_max": 60,
                "factor": 1.3
            },
            "settled": {
                "months_max": 999,
                "factor": 1.0
            }
        }
    },
    "policy": {
        "city_centralized_treatment_rate": 0.661,
        "city_centralized_treatment_note": "克拉玛依市集中收治率66.1%，来源：中国CDC结核病中心2024; 克拉玛依网2025",
        "free_treatment_policy": True,
        "mandatory_reporting": True,
        "migrant_population_coverage": 0.75,
        "contact_tracing_radius": 50,
        "school_screening_threshold": 1,
        "default_screening_interval_months": 6,
        "districts": {
            "\u514b\u62c9\u739b\u4f9d\u533a": {
                "screening_frequency_months": 6,
                "enhanced_groups": [
                    "密切接触者"
                ],
                "treatment_center": "克拉玛依市中心医院",
                "centralized_treatment_rate": 0.62,
                "care_action_active": False
            },
            "\u72ec\u5c71\u5b50\u533a": {
                "screening_frequency_months": 3,
                "enhanced_groups": [
                    "密切接触者",
                    "老年人",
                    "糖尿病患者"
                ],
                "treatment_center": "独山子人民医院",
                "centralized_treatment_rate": 0.7,
                "care_action_active": True,
                "care_action_note": "独山子关爱行动试点区域"
            },
            "\u767d\u78b1\u6ee9\u533a": {
                "screening_frequency_months": 6,
                "enhanced_groups": [
                    "密切接触者",
                    "油田作业人员"
                ],
                "treatment_center": "白碱滩医院（第二人民医院）",
                "centralized_treatment_rate": 0.65,
                "care_action_active": False
            },
            "\u4e4c\u5c14\u79be\u533a": {
                "screening_frequency_months": 6,
                "enhanced_groups": [
                    "密切接触者",
                    "流动人口"
                ],
                "treatment_center": "乌尔禾区人民医院",
                "centralized_treatment_rate": 0.6,
                "care_action_active": False
            }
        },
        "dushanzi_care_action": {
            "name": "独山子结核病关爱行动",
            "description": "独山子区结核病关爱行动试点，强化主动筛查、缩短随访周期、升级检测手段",
            "screening_upgrade_rule": "高风险接触者建议从PPD/IGRA筛查升级为胸部X线+分子生物学检测",
            "followup_interval_days_care": 14,
            "followup_interval_days_care_highrisk": 7,
            "centralized_treatment_note": "已确诊患者默认传染期为收治前的周数，显著缩短社区暴露时间"
        },
        "screening_templates": {
            "standard": "建议进行PPD/IGRA筛查",
            "enhanced": "根据{action}要求，升级为胸部X线+IGRA联合筛查",
            "urgent": "根据{action}和本地防控要求，立即进行综合筛查（胸片+GeneXpert+痰培养）",
            "care_action_enhanced": "立即进行胸部X线+分子生物学检测",
            "followup_care": "纳入{action}社区随访计划，每{interval}天随访一次"
        }
    },
    "counterfactual_interventions": [
        {
            "id": "oilfield_ventilation",
            "name": "油田营地通风改善",
            "description": "改善油田营地板房的通风条件，安装排风扇或新风系统",
            "from": "通风评分 2",
            "to": "通风评分 4",
            "estimated_effect": "油田营地感染风险 ↓ 35-40%",
            "cost": "低-中（加装排风扇/新风口）",
            "literature": "吐哈油田公司, 2025; Escombe et al., 2007"
        },
        {
            "id": "oilfield_shift_optimization",
            "name": "轮班制优化（减少同寝人数）",
            "description": "将油田营地宿舍从4-8人间调整为2-4人间",
            "from": "4-8人/间",
            "to": "2-4人/间",
            "estimated_effect": "同寝传播风险 ↓ 40-50%",
            "cost": "中-高（需新增宿舍）",
            "literature": "新医大公卫学院, 2025"
        },
        {
            "id": "idu_methadone",
            "name": "强制戒毒/美沙酮维持治疗",
            "description": "将IDU接触者纳入美沙酮维持治疗，降低免疫损伤和进展风险",
            "from": "未接受维持治疗",
            "to": "接受美沙酮维持治疗",
            "estimated_effect": "发病风险 ↓ 30-50%",
            "cost": "中（需社区康复中心配合）",
            "literature": "中国防痨协会, 2006; 结核病高危人群, 2023"
        },
        {
            "id": "altitude_screening",
            "name": "高海拔迁入者预防性筛查",
            "description": "对从海拔>1500m地区迁入6个月内的居民开展LTBI筛查",
            "from": "未纳入重点人群",
            "to": "迁入6个月内完成IGRA筛查",
            "estimated_effect": "迁入人群潜伏感染激活早发现率 ↑ 40%",
            "cost": "低-中",
            "literature": "Acta Med Port, 2018; Evol Med Public Health, 2022"
        },
        {
            "id": "centralized_treatment",
            "name": "集中收治管理",
            "description": "将患者纳入定点医院集中收治，缩短社区暴露时间",
            "from": "社区暴露期未缩短",
            "to": "收治前暴露周数",
            "estimated_effect": "治疗成功率 ↑ 10-15%，传播风险 ↓ 30-40%",
            "cost": "中（需床位和人力资源）",
            "literature": "中国CDC结核病中心, 2024"
        },
        {
            "id": "care_action_enhanced",
            "name": "关爱行动强化筛查",
            "description": "按独山子关爱行动要求提高筛查频率和范围",
            "from": "常规筛查",
            "to": "每月随访+分子检测",
            "estimated_effect": "早期发现率 ↑ 25%，潜伏感染干预率 ↑ 20%",
            "cost": "中-高",
            "literature": "克拉玛依网, 2025"
        },
        {
            "id": "migrant_coverage",
            "name": "流动人口结核病服务覆盖",
            "description": "将油田流动人口纳入结核病防控服务体系",
            "from": "覆盖率75%",
            "to": "覆盖率95%",
            "estimated_effect": "诊断延迟 ↓ 40%，治疗完成率 ↑ 20%",
            "cost": "中（需跨部门协调）",
            "literature": "克拉玛依市流动人口防控政策"
        }
    ],
    "references": [
        "克拉玛依卫健委, 2024（本地结核病监测数据报告，检出率121/10万、收治率66.1%）",
        "Liu et al., Chin Gen Pract, 2025（克拉玛依结核病流行病学特征，空间模型σ²ω≈1.806）",
        "吐哈油田公司, 2025（油田作业人群职业暴露评估，健康体检文件）",
        "新医大公卫学院, 2025（新疆职业暴露与结核病关联研究，呼吸系统疾病负担）",
        "中国防痨协会, 2006（静脉吸毒者结核病防控指南，0%正规疗程完成率）",
        "结核病高危人群, 2023（高危人群结核病防控专家共识，IDU风险证据）",
        "Front Public Health, 2025, 13:1512644（喀什PM-TB交互，PM10每增50μg/m³传播力+0.15）",
        "bioRxiv, 2025（干旱气候气溶胶传播模型，湿度阈值30%蒸发效应）",
        "Acta Med Port, 2018, 31(10)（高海拔迁入-结核潜伏激活，高原居民下降后TB激活风险）",
        "Evol Med Public Health, 2022（海拔下降-潜伏感染激活，尼泊尔藏族人群机制假设）",
        "中国防痨杂志, 2025（新疆菌株基因型分布，用于描述性统计，不参与风险计算）",
        "WHO. Social determinants of tuberculosis: a framework for action. WHO/HTM/TB/2023.",
        "Lönnroth K et al. Drivers of tuberculosis epidemics. Soc Sci Med 68(12):2240-2246, 2009.",
        "Lönnroth K et al. BMI and TB incidence. Int J Epidemiol 39(1):149-155, 2010.",
        "Belmont Report. Ethical Principles for Research Involving Human Subjects, 1979.",
        "Mitchell S et al. Prediction-based decisions and fairness. Annu Rev Stat Appl, 2024.",
        "克拉玛依市统计局, 2024年鉴（收入中位数、住房面积等社会经济数据）",
        "克拉玛依市结核病防治规划, 2024（定点诊疗机构分布及可及性评估）",
        "中国CDC结核病中心, 2024（地方结核病防控政策指南）",
        "克拉玛依网, 2025（独山子区结核病关爱行动报道，疾控专家解读）"
    ],
    "bayesian_calibration": {
        "_note": "贝叶斯校准观测数据（L2默认路径）。各区县筛查阳性率(%)，基于克拉玛依卫健委2024年报告。若此字段为空或不完整，贝叶斯校准将跳过，回退到硬编码公式。",
        "observed_data": {
            "karamay": 5.2,
            "karamay_note": "克拉玛依区：城区人口集中，筛查覆盖率最高",
            "dushanzi": 3.8,
            "dushanzi_note": "独山子区：石化产业为主，流动人口比例高",
            "baijiantan": 6.1,
            "baijiantan_note": "白碱滩区：油田作业区域，职业暴露风险高",
            "urho": 4.5,
            "urho_note": "乌尔禾区：旅游区，季节性人口波动"
        }
    }
}

# 确保民族构成数据与 constants.py 保持一致（单一真值源）
# 保留 _note 说明字段，数值部分统一引用 ETHNICITY_WEIGHTS
_EMBEDDED_KARAMAY_CONFIG['population']['ethnicity_composition'] = {
    "_note": "民族构成（描述性统计变量，不参与风险计算）。遵循 Belmont Report 公正原则，不基于民族身份进行风险区分。数据来源：constants.ETHNICITY_WEIGHTS（单一真值源）。",
    **_ETHNICITY_WEIGHTS
}

KARAMAY_LOGGER = logging.getLogger("tb_risk.karamay")

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG_PATH = os.path.join(_CONFIG_DIR, "karamay_local_params.json")

# ==============================================================================
# 配置 JSON Schema 定义（用于验证外部配置文件）
# ==============================================================================

_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["_meta", "epidemiology", "occupation"],
    "properties": {
        "_meta": {
            "type": "object",
            "required": ["version"],
            "properties": {
                "version": {"type": "string"},
                "name": {"type": "string"},
                "last_updated": {"type": "string"},
            }
        },
        "epidemiology": {
            "type": "object",
            "required": ["base_incidence_per_100k"],
            "properties": {
                "base_incidence_per_100k": {"type": "number", "minimum": 0, "maximum": 10000},
                "latent_prevalence_family_min": {"type": "number", "minimum": 0, "maximum": 1},
                "latent_prevalence_family_max": {"type": "number", "minimum": 0, "maximum": 1},
                "latent_prevalence_family_default": {"type": "number", "minimum": 0, "maximum": 1},
                "hrsp_percentage_default": {"type": "number", "minimum": 0, "maximum": 100},
                "diagnosis_delay_median_days": {"type": "number", "minimum": 0},
                "treatment_success_rate": {"type": "number", "minimum": 0, "maximum": 1},
                "dr_resistance_rate": {"type": "number", "minimum": 0, "maximum": 1},
            }
        },
        "occupation": {
            "type": "object",
            "required": ["oilfield_camp_factor"],
            "properties": {
                "oilfield_camp_factor": {"type": "number", "minimum": 0, "maximum": 1},
            }
        }
    }
}

# 配置版本兼容性映射：{旧版本: 迁移函数}
_CONFIG_MIGRATIONS = {
    # 未来版本迁移规则示例：
    # "1.0": _migrate_v1_to_v2,
}


def _validate_config_schema(config, schema=None):
    """使用 JSON Schema 验证配置文件

    参数：
        config (dict): 配置字典
        schema (dict|None): JSON Schema 定义，None 则使用默认 schema

    返回：
        tuple[bool, list[str]]: (是否有效, 错误列表)
    """
    if schema is None:
        schema = _CONFIG_SCHEMA

    errors = []

    try:
        import jsonschema
        validator = jsonschema.Draft7Validator(schema)
        for error in validator.iter_errors(config):
            path = '.'.join(str(p) for p in error.absolute_path)
            errors.append(f"{path}: {error.message}")
    except ImportError:
        # jsonschema 未安装，执行基本的手动校验
        errors = _basic_schema_check(config, schema)

    return len(errors) == 0, errors


def _basic_schema_check(config, schema, path=''):
    """基本 schema 校验（jsonschema 不可用时的回退方案）"""
    errors = []
    if schema.get('type') == 'object' and not isinstance(config, dict):
        errors.append(f"{path}: 期望 object，实际 {type(config).__name__}")
        return errors

    if isinstance(config, dict) and 'properties' in schema:
        for key, prop_schema in schema['properties'].items():
            if key in config:
                sub_errors = _basic_schema_check(config[key], prop_schema, f"{path}.{key}" if path else key)
                errors.extend(sub_errors)
            elif key in schema.get('required', []):
                errors.append(f"{path}.{key}: 缺少必需字段")

        for key in schema.get('required', []):
            if key not in config:
                errors.append(f"{path}.{key}: 缺少必需字段")

    return errors


def _load_config(config_path=None):
    """加载外部JSON配置文件（优先外部文件，回退到内嵌配置）

    参数：
        config_path (str|None): 配置文件路径，None则使用默认路径

    返回：
        dict: 配置字典；若加载失败则回退到内嵌配置
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)

            # Schema 验证
            is_valid, errors = _validate_config_schema(cfg)
            if not is_valid:
                KARAMAY_LOGGER.warning("外部配置文件 schema 验证失败: %s", errors)
                KARAMAY_LOGGER.warning("请检查配置字段是否正确，当前使用内嵌配置")
                return copy.deepcopy(_EMBEDDED_KARAMAY_CONFIG)

            # 版本迁移
            current_version = cfg.get('_meta', {}).get('version', 'unknown')
            KARAMAY_LOGGER.info("外部配置文件加载成功: %s (version=%s)", path, current_version)
            return cfg
        except (json.JSONDecodeError, OSError) as e:
            KARAMAY_LOGGER.warning("外部配置文件加载失败: %s, 错误: %s，回退到内嵌配置", path, e)
    else:
        KARAMAY_LOGGER.info("外部配置文件不存在: %s，使用内嵌配置", path)
        # 首次运行时自动生成外部配置文件
        _generate_default_config(path)
    return copy.deepcopy(_EMBEDDED_KARAMAY_CONFIG)


def _generate_default_config(config_path=None):
    """首次运行时自动从内嵌配置生成默认的外部配置文件

    生成的文件包含详细字段注释和值域说明，方便用户修改。

    参数：
        config_path (str|None): 目标路径，None 使用默认路径

    返回：
        bool: 是否生成成功
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    if os.path.exists(path):
        KARAMAY_LOGGER.debug("外部配置文件已存在，跳过生成: %s", path)
        return False

    try:
        cfg = copy.deepcopy(_EMBEDDED_KARAMAY_CONFIG)
        # 确保版本标记
        cfg['_meta']['version'] = '2.0'
        cfg['_meta']['generated_at'] = datetime.datetime.now().isoformat()

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        KARAMAY_LOGGER.info("已自动生成默认配置文件: %s", path)
        KARAMAY_LOGGER.info("你可以编辑此文件来调整克拉玛依本地化参数，无需修改源代码。")
        return True
    except (OSError, IOError) as e:
        KARAMAY_LOGGER.warning("自动生成配置文件失败: %s", e)
        return False


def reload_config(config_path=None, localizer_instance=None):
    """热更新配置（支持不重启程序重新加载配置）

    参数：
        config_path (str|None): 配置文件路径，None 使用默认
        localizer_instance (KaramayLocalizer|None): 本地化器实例，
            传入则通过观察者模式通知依赖模块更新

    返回：
        tuple[dict, bool]: (新配置, 是否成功)
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        KARAMAY_LOGGER.warning("配置文件不存在，无法重载: %s", path)
        return {}, False

    try:
        with open(path, 'r', encoding='utf-8') as f:
            new_cfg = json.load(f)

        # Schema 验证
        is_valid, errors = _validate_config_schema(new_cfg)
        if not is_valid:
            KARAMAY_LOGGER.warning("重载配置失败，schema 验证不通过: %s", errors)
            return {}, False

        KARAMAY_LOGGER.info("配置热更新成功: %s (version=%s)",
                          path, new_cfg.get('_meta', {}).get('version', 'unknown'))

        # 如果传入了 localizer 实例，更新其配置
        if localizer_instance is not None:
            try:
                localizer_instance._config = ConfigProxy(new_cfg)
                # 重新初始化子模块
                localizer_instance._reinit_modules(new_cfg)
                KARAMAY_LOGGER.info("已通知 KaramayLocalizer 实例更新配置")
            except Exception as e:
                KARAMAY_LOGGER.warning("通知 localizer 更新失败: %s", e)

        return new_cfg, True
    except (json.JSONDecodeError, OSError) as e:
        KARAMAY_LOGGER.warning("重载配置失败: %s", e)
        return {}, False


def get_config_path():
    """获取默认配置文件路径

    返回：
        str: 配置文件完整路径
    """
    return _DEFAULT_CONFIG_PATH


class ConfigProxy:
    """配置代理：用点号路径安全访问嵌套JSON配置，缺失时返回默认值"""

    def __init__(self, data=None):
        self._data = data or {}

    def get(self, path, default=None):
        """通过点号分隔路径获取配置值

        例: proxy.get('epidemiology.base_incidence_per_100k', 0)
        """
        keys = path.split('.')
        node = self._data
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    def get_dict(self, path, default=None):
        """获取字典值并返回副本"""
        v = self.get(path, default)
        return dict(v) if isinstance(v, dict) else (v if v is not None else {})

    def get_list(self, path, default=None):
        """获取列表值并返回副本"""
        v = self.get(path, default)
        return list(v) if isinstance(v, list) else (v if v is not None else [])