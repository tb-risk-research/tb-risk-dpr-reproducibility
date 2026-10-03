#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
维度六：民族描述性统计模块（描述性用途，不参与风险计算）

⚠️ 重要变更（v3.0）：
原 EthnicityPathogenModule 中的民族遗传易感性参数（SLC11A1 基因多态性、
菌株传播力系数等）已移除。民族身份仅作为描述性统计变量，
不参与任何风险计算公式。

替代方案：使用 karamy.sdoh.SDOHModule 基于可测量的社会决定因素
（住房拥挤、医疗可及性、营养风险、收入分层）进行风险评估。

遵循：
- Belmont Report 1979 三原则（尊重、行善、公正）
- 《个人信息保护法》敏感信息处理要求
- 数据最小化原则

文献：
- WHO. Social determinants of tuberculosis: a framework for action. WHO/HTM/TB/2023.
- Lönnroth K et al. Drivers of tuberculosis epidemics. Soc Sci Med 68(12):2240-2246, 2009.
"""


class EthnicityPathogenModule:
    """民族描述性统计模块（v3.0 — 仅描述性，不参与风险计算）。

    保留此类以维持向后兼容性。所有返回的风险乘数均为 1.0。

    若需基于社会决定因素进行风险评估，请使用 sdoh.SDOHModule。
    """

    def __init__(self, config):
        self._cfg = config

    def get_strain_transmissibility(self, strain_type='beijing'):
        """[已弃用] 菌株传播力系数。始终返回 1.0。

        原实现基于民族-菌株绑定假设，已被移除。
        菌株分布信息仅用于描述性统计。
        """
        return 1.0

    def get_ethnicity_susceptibility(self, ethnicity='han'):
        """[已弃用] 民族遗传易感性。始终返回 1.0。

        原 SLC11A1 基因多态性假设无充分文献支持，
        且将民族间 TB 发病率差异归因于遗传因素
        忽略了社会经济结构性因素。
        """
        return 1.0

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

    def get_ethnicity_label(self, ethnicity):
        """获取民族标签（描述性用途）"""
        labels = self._cfg.get_dict('population.ethnicity_labels')
        return labels.get(ethnicity, str(ethnicity))

    def get_combined_strain_ethnicity_factor(self, strain_type='beijing', ethnicity='han'):
        """[已弃用] 民族-菌株组合因子。始终返回 1.0。

        原方法将民族身份与菌株传播力绑定，已被移除。
        请使用 sdoh.SDOHModule.compute_sdoh_multiplier() 替代。
        """
        return 1.0

    def estimate_local_strain_weighted_factor(self):
        """[已弃用] 本地菌株加权因子。始终返回 1.0。"""
        return 1.0

    def get_screening_priority_adjustment(self, ethnicity='han', strain_type='beijing'):
        """[已弃用] 筛查优先级调整。始终返回无调整。

        民族身份不应用于筛查优先级排序。
        如需基于风险的筛查优先级，请使用 SDOH 模块。
        """
        eth_label = self.get_ethnicity_label(ethnicity)
        return {
            'adjustment': 1.0,
            'ethnicity': ethnicity,
            'strain_type': strain_type,
            'reason': f'{eth_label}：民族为描述性统计变量，不参与风险计算（Belmont Report 公正原则）',
            'note': '民族遗传易感性参数已移除，请使用 SDOH 模块替代',
        }