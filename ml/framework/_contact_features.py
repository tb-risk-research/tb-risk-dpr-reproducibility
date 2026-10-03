#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""接触者/边特征提取方法混入。

将 ``HeterogeneousTBNetwork`` 中纯 Python 的接触者特征解析逻辑
拆分至此 mixin，使主类文件行数控制在 500 行以内。所有方法通过
正常继承在 ``HeterogeneousTBNetwork`` 实例上可用，无业务逻辑变更。
"""

import numpy as np

# 使用 utils 中统一的 _is_yes 函数，避免重复实现
try:
    from ...utils import _is_yes
except ImportError:
    # 回退实现（仅在相对导入不可用时使用）
    def _is_yes(value):
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value > 0
        if isinstance(value, str):
            return value.strip().lower() in ('1', '是', '有', '涂阳', 'yes', 'true')
        return False


class _ContactFeatureMixin:
    """接触者特征与边特征提取方法集合（供 ``HeterogeneousTBNetwork`` 继承）。"""

    def _parse_contact_basic(self, entry):
        """解析接触者基础特征：年龄、BCG、病史、免疫状态（特征1-6）"""
        features = []
        # 年龄
        age_str = entry.get('age', 30)
        try:
            age = float(age_str) if age_str else 30.0
        except (ValueError, TypeError):
            age = 30.0
        features.append(age / 100.0)

        # BCG、既往病史
        bcg = 1 if _is_yes(entry.get('bcg_vaccine', '是')) else 0
        features.append(bcg)
        past_illness = 1 if _is_yes(entry.get('past_illness', '否')) else 0
        features.append(past_illness)

        # 疾病类型（HIV、糖尿病、免疫抑制）
        illness_type = entry.get('past_illness_type', 'none')
        features.append(1.0 if illness_type in ['hiv', 'HIV'] else 0.0)
        features.append(1.0 if illness_type in ['diabetes', '糖尿病'] else 0.0)
        features.append(1.0 if illness_type in ['immunosuppressants', '免疫抑制', '免疫抑制剂'] else 0.0)

        return features, age, bcg, illness_type

    def _parse_contact_environment(self, entry):
        """解析接触者环境特征：通风、接触距离（特征7-8）"""
        features = []
        # 通风条件
        ventilation_str = entry.get('ventilation', 3)
        try:
            ventilation = float(ventilation_str)
        except (ValueError, TypeError):
            ventilation = 3.0
        features.append(ventilation / 5.0)

        # 接触距离
        contact_distance = entry.get('contact_distance', '中等')
        dist_map = {
            '近': 1.0, '中等': 0.5, '远': 0.25, '很远': 0.1,
            'very_close': 1.0, 'close': 0.8, 'medium': 0.5,
            'far': 0.25, 'distant': 0.1,
            '极近': 1.0, '极远': 0.1
        }
        features.append(dist_map.get(contact_distance, 0.5))

        return features

    def _parse_contact_exposure(self, entry, contact_type):
        """解析接触者暴露特征：累计暴露、频率、时长、周期、场景（特征9-13）"""
        features = []
        # 累计暴露
        cumulative = entry.get('cumulative_exposure', 0)
        try:
            cumulative = float(cumulative)
        except (ValueError, TypeError):
            cumulative = 0.0
        features.append(min(cumulative / 1000.0, 1.0))

        # 接触频率
        freq_density = entry.get('freq_density', 14 if contact_type == 'family' else 2)
        try:
            freq_density = float(freq_density)
        except (ValueError, TypeError):
            freq_density = 14 if contact_type == 'family' else 2
        features.append(min(freq_density / 30.0, 1.0))

        # 单次接触时长
        single_duration = entry.get('single_duration', 30)
        try:
            single_duration = float(single_duration)
        except (ValueError, TypeError):
            single_duration = 30.0
        features.append(single_duration / 120.0)

        # 持续周数
        time_span = entry.get('time_span', 4)
        try:
            time_span = float(time_span)
        except (ValueError, TypeError):
            time_span = 4.0
        features.append(time_span / 24.0)

        # 暴露场景
        setting = entry.get('exposure_setting', '一般')
        setting_map = {
            '高危': 1.0, '一般': 0.5, '低危': 0.1,
            'crowded': 1.0, 'closed': 0.9, 'general': 0.6, 'outdoor': 0.3,
            'oilfield_camp': 0.85,
            '拥挤': 1.0, '密闭': 0.9, '户外': 0.3,
            '油田营地': 0.85, '油田': 0.85
        }
        features.append(setting_map.get(setting, 0.5))

        return features, cumulative, time_span

    def _build_contact_interaction_features(self, age, bcg, cumulative, time_span,
                                              illness_type, entry):
        """构建交互特征和附加特征（特征14-22）"""
        features = []
        # 交互特征
        hiv = 1.0 if illness_type in ['hiv', 'HIV'] else 0.0
        diabetes = 1.0 if illness_type in ['diabetes', '糖尿病'] else 0.0
        immunosuppressants = 1.0 if illness_type in ['immunosuppressants', '免疫抑制', '免疫抑制剂'] else 0.0
        immunosuppression_score = hiv * 2.0 + diabetes * 1.0 + immunosuppressants * 1.5
        features.append((age / 100.0) * (immunosuppression_score / 2.0))
        features.append((age / 100.0) * (1 - bcg))
        features.append(min(cumulative / 1000.0, 1.0) * (time_span / 24.0))

        # 附加特征
        has_symptoms = 1 if _is_yes(entry.get('has_symptoms', '否')) else 0
        features.append(has_symptoms)
        is_high_risk = 1 if _is_yes(entry.get('is_high_risk', '否')) else 0
        features.append(is_high_risk)
        has_tb = 1 if _is_yes(entry.get('has_tb', '否')) else 0
        features.append(has_tb)
        features.append(min(age / 65.0, 1.0))
        features.append(min(cumulative / 500.0, 1.0))
        features.append(0.0)  # 保留位 (特征22，确保 22 维基础特征)

        return features

    def _parse_contact_seir_state(self, entry):
        """解析接触者 SEIR 房室状态 one-hot 编码（特征23-30，v3.0 新增）

        潜伏感染状态 (4维 one-hot): L_fast / L_slow / Cleared / None
        疾病状态 (4维 one-hot): Minimal / Subclinical / Clinical / None

        从 entry 中读取 'latent_state' 和 'disease_state' 字段，
        缺失时默认为 'None'（未感染/未分层）。

        文献：
            Houben et al. (2016) BMC Med — 快/慢潜伏分层
            Emery et al. (2023) eLife — 亚临床 TB
            Horton et al. (2023) PNAS — 自清除
        """
        # 潜伏感染状态 one-hot
        latent_state = str(entry.get('latent_state', 'None')).strip()
        latent_map = {
            'L_fast': [1, 0, 0, 0],
            'L_slow': [0, 1, 0, 0],
            'Cleared': [0, 0, 1, 0],
            'None': [0, 0, 0, 1],
            # 中文别名
            '快潜伏': [1, 0, 0, 0],
            '慢潜伏': [0, 1, 0, 0],
            '已清除': [0, 0, 1, 0],
            '无': [0, 0, 0, 1],
        }
        latent_onehot = latent_map.get(latent_state, [0, 0, 0, 1])

        # 疾病状态 one-hot
        disease_state = str(entry.get('disease_state', 'None')).strip()
        disease_map = {
            'Minimal': [1, 0, 0, 0],
            'Subclinical': [0, 1, 0, 0],
            'Clinical': [0, 0, 1, 0],
            'None': [0, 0, 0, 1],
            # 中文别名
            '微小': [1, 0, 0, 0],
            '亚临床': [0, 1, 0, 0],
            '临床': [0, 0, 1, 0],
            '无': [0, 0, 0, 1],
        }
        disease_onehot = disease_map.get(disease_state, [0, 0, 0, 1])

        return latent_onehot + disease_onehot

    def _extract_contact_features(self, entry, contact_type):
        """提取接触者特征（30维，v3.0 扩展）

        特征包括：
        1-6:   基础特征（年龄、BCG、病史、免疫状态）
        7-8:   环境特征（通风、接触距离）
        9-13:  暴露特征（累计暴露、频率、时长、周期、场景）
        14-22: 交互特征和附加特征
        23-26: 潜伏感染状态 one-hot (L_fast/L_slow/Cleared/None)
        27-30: 疾病状态 one-hot (Minimal/Subclinical/Clinical/None)
        """
        basic_features, age, bcg, illness_type = \
            self._parse_contact_basic(entry)
        env_features = self._parse_contact_environment(entry)
        exp_features, cumulative, time_span = \
            self._parse_contact_exposure(entry, contact_type)
        advanced_features = self._build_contact_interaction_features(
            age, bcg, cumulative, time_span, illness_type, entry)
        seir_state_features = self._parse_contact_seir_state(entry)

        features = basic_features + env_features + exp_features + advanced_features + seir_state_features
        return features[:30]

    def _extract_edge_features(self, entry, contact_type):
        """
        提取边特征（6维，文献支撑：Zhu et al., PLOS ONE 2024）

        边特征包括：
        - 接触频率评分
        - 单次接触时长
        - 持续周数
        - 通风条件评分
        - 接触距离评分
        - 暴露场景评分
        """
        edge_features = []

        # 接触频率评分（归一化）
        freq_density = entry.get('freq_density', 14 if contact_type == 'family' else 2)
        try:
            freq_score = float(freq_density) / 30.0
        except (ValueError, TypeError):
            freq_score = 0.47 if contact_type == 'family' else 0.07
        edge_features.append(np.clip(freq_score, 0.0, 1.0))

        # 单次接触时长
        single_duration = entry.get('single_duration', 30)
        try:
            dur_score = float(single_duration) / 480.0
        except (ValueError, TypeError):
            dur_score = 0.06
        edge_features.append(np.clip(dur_score, 0.0, 1.0))

        # 持续周数
        time_span = entry.get('time_span', 4)
        try:
            time_score = float(time_span) / 52.0
        except (ValueError, TypeError):
            time_score = 0.08
        edge_features.append(np.clip(time_score, 0.0, 1.0))

        # 通风条件评分
        ventilation = entry.get('ventilation', 3)
        vent_map = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.3, 5: 0.15}
        try:
            vent_score = vent_map.get(int(ventilation), 0.5)
        except (ValueError, TypeError):
            vent_score = 0.5
        edge_features.append(np.clip(vent_score, 0.0, 1.0))

        # 接触距离评分
        contact_distance = entry.get('contact_distance', '中等')
        dist_map = {'近': 1.0, '中等': 0.5, '远': 0.25, '很远': 0.1,
                   'very_close': 1.0, 'close': 0.8, 'medium': 0.5, 'far': 0.25, 'distant': 0.1}
        dist_score = dist_map.get(contact_distance, 0.5)
        edge_features.append(np.clip(dist_score, 0.0, 1.0))

        # 暴露场景评分
        exposure_setting = entry.get('exposure_setting', '一般')
        setting_map = {'高危': 1.0, '一般': 0.5, '低危': 0.1,
                      'crowded': 1.0, 'closed': 0.9, 'general': 0.6, 'outdoor': 0.3,
                      'oilfield_camp': 0.85,
                      '拥挤': 1.0, '密闭': 0.9, '户外': 0.3,
                      '油田营地': 0.85, '油田': 0.85}
        set_score = setting_map.get(exposure_setting, 0.6)
        edge_features.append(np.clip(set_score, 0.0, 1.0))

        return edge_features

    # ------------------------------------------------------------------
    # 区域级节点特征（多尺度特征融合：个体—家庭—社区—区域）
    # ------------------------------------------------------------------
    # 区域节点特征维度常量（与节点特征 30 维对齐，v5.0 多尺度）
    REGION_FEATURE_DIM = 30

    # 区域发病率归一化除数（每 10 万人口）
    REGION_INCIDENCE_NORM = 500.0

    def _extract_region_features(self, tb_assessment, region_name=None,
                                 num_contacts=0, config=None):
        """提取区域级节点特征（30 维，多尺度特征融合 v5.0）。

        区域节点是"个体—家庭—社区—区域"层次结构的最顶层抽象节点，
        聚合区域层面的流行病学与卫生政策特征，使个体预测与区域传播
        预测共用同一套图结构。

        特征布局（30 维，与接触者节点对齐）：
            0:  区域年发病率归一化（per_100k / 500）
            1:  区域筛查强度（筛查间隔月数越小风险越低）
            2:  集中收治率（≥85% 高收治 → 社区暴露风险低）
            3:  关爱行动激活标志（1=激活，加强筛查）
            4:  增强筛查群体数量归一化（/5）
            5:  区域人口密度代理（默认 0.5，可传 population_density）
            6:  接触网络规模（num_contacts / 20）
            7:  区域病例负荷代理（患者涂阳/空洞严重度，0-1）
            8:  既往区域疫情水平代理（0.5 默认）
            9:  区域干预强度代理（0.5 默认）
            10-21: 保留位（0，供后续扩展）
            22-25: 潜伏状态 one-hot（区域抽象节点 → None）
            26-29: 疾病状态 one-hot（区域抽象节点 → None）

        文献：
            Zheng et al. (2024) HeatGNN — 社区/区域级抽象节点
            WHO (2024) — 区域筛查频率与集中收治政策
        """
        # 区域名称（默认克拉玛依区）
        if region_name is None:
            region_name = '克拉玛依区'
            try:
                basic = tb_assessment.patient_info.get('basic_info', {})
                region_name = basic.get('district', region_name)
            except AttributeError:
                pass

        # 区域配置（优先外部注入，其次本地配置，最后默认值）
        district_cfg = {}
        base_incidence_per_100k = 121.0  # 克拉玛依 2024 检出率（默认）
        if config is None:
            config = self._load_region_config()
        if config is not None:
            district_cfg = config.get_dict('policy.districts', {}).get(region_name, {})
            base_incidence_per_100k = float(config.get(
                'epidemiology.base_incidence_per_100k', 121.0))

        # 特征 0：区域年发病率（归一化）
        incidence = float(district_cfg.get(
            'incidence_per_100k', base_incidence_per_100k))
        features = [min(incidence / self.REGION_INCIDENCE_NORM, 1.0)]

        # 特征 1：区域筛查强度（筛查间隔越短 → 风险暴露窗口越短）
        screening_months = float(district_cfg.get('screening_frequency_months', 6.0))
        features.append(min(max(0.0, 1.0 - screening_months / 12.0), 1.0))

        # 特征 2：集中收治率
        central_rate = float(district_cfg.get('centralized_treatment_rate', 0.661))
        features.append(min(max(central_rate, 0.0), 1.0))

        # 特征 3：关爱行动激活
        features.append(1.0 if district_cfg.get('care_action_active', False) else 0.0)

        # 特征 4：增强筛查群体数量归一化
        enhanced_groups = district_cfg.get('enhanced_groups', []) or []
        features.append(min(len(enhanced_groups) / 5.0, 1.0))

        # 特征 5：区域人口密度代理
        features.append(float(district_cfg.get('population_density', 0.5)))

        # 特征 6：接触网络规模（归一化）
        try:
            num_contacts = int(num_contacts)
        except (ValueError, TypeError):
            num_contacts = 0
        features.append(min(max(num_contacts, 0) / 20.0, 1.0))

        # 特征 7：区域病例负荷代理（患者涂阳/空洞严重度）
        burden = 0.5
        try:
            basic = tb_assessment.patient_info.get('basic_info', {})
            sputum = 1.0 if basic.get('sputum_smear', 1) == 2 else 0.0
            cavity = 1.0 if basic.get('has_cavity', 1) == 2 else 0.0
            burden = min(0.5 + 0.25 * sputum + 0.25 * cavity, 1.0)
        except AttributeError:
            burden = 0.5
        features.append(burden)

        # 特征 8：既往区域疫情水平代理
        features.append(float(district_cfg.get('historical_incidence_level', 0.5)))

        # 特征 9：区域干预强度代理
        features.append(float(district_cfg.get('intervention_intensity', 0.5)))

        # 特征 10-21：保留位
        features.extend([0.0] * (22 - len(features)))

        # 特征 22-29：SEIR 状态 one-hot（区域抽象节点 → 潜伏=None，疾病=None）
        features.extend([0, 0, 0, 1])  # latent: None
        features.extend([0, 0, 0, 1])  # disease: None

        return features[:self.REGION_FEATURE_DIM]

    @staticmethod
    def _load_region_config():
        """加载区域配置（ConfigProxy），失败时回退到空配置。"""
        try:
            from ...config import _load_config, ConfigProxy
            return ConfigProxy(_load_config())
        except Exception:
            return None

