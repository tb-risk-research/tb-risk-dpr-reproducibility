#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四层异质结核病接触网络构建器（方向二，文献驱动实现）。

接触者/边特征解析逻辑见 ``_contact_features._ContactFeatureMixin``，
通过继承注入本类，无业务逻辑变更。
"""

from ._common import (
    _GNNFrameworkPlaceholder, PYTORCH_AVAILABLE, PYG_AVAILABLE,
    torch, HeteroData, np,
)
from ._contact_features import _ContactFeatureMixin
from ...core import calculate_cumulative_exposure

HeterogeneousTBNetwork = _GNNFrameworkPlaceholder

if PYTORCH_AVAILABLE and PYG_AVAILABLE:
    class HeterogeneousTBNetwork(_ContactFeatureMixin):
        """
        四层异质结核病接触网络构建器（方向二，文献驱动实现）

        网络层次（文献支撑：Liu et al., ACM SIGKDD 2024; Zheng et al., HeatGNN 2024）：
        1. 患者节点（源节点）
        2. 家庭接触者节点（第一层网络，家庭内全连接）
        3. 社会接触者节点（第二层网络）
        4. 社区节点（第三层，抽象节点，连接所有接触者）

        边类型：
        - patient→family: 患者感染家庭成员
        - patient→social: 患者感染社会接触者
        - family→family: 家庭成员间共居传播（全连接）
        - family→community: 家庭连接到社区
        - social→community: 社会接触者连接到社区

        边特征（Yang et al., HGAT-AMR 2021）：
        - 接触频率、单次时长、持续周数、通风条件、接触距离、暴露场景
        """

        def __init__(self, device='cpu'):
            self.patient_node = None
            self.family_nodes = []
            self.social_nodes = []
            self.community_node = None
            self.device = torch.device(device)

        def build_from_assessment(self, tb_assessment, device=None,
                                  p_base_map=None):
            """从TB_Risk_Assessment对象构建四层异质图

            参数：
                tb_assessment: 结核病风险评估实例
                device: 目标设备（None 时用构造时设备）
                p_base_map: 第 1 层（ML 个体基础层）输出的 P_base 映射，
                    dict {record_id: p_base(0-100)} 或 {_id: p_base}。
                    传入时把 P_base 作为**第 31 维节点特征**注入到接触者节点
                    （堆叠：第 2 层以第 1 层输出为特征），供残差学习使用；
                    缺失记录回退 0。None 时不注入，保持 30 维向后兼容。
            """
            if device is None:
                device = self.device
            data = HeteroData()

            # 患者节点
            patient_features = self._extract_patient_features(tb_assessment)
            if p_base_map is not None:
                patient_features = list(patient_features) + [0.0]
            data['patient'].x = torch.tensor(patient_features, dtype=torch.float, device=device).unsqueeze(0)

            # 家庭接触者节点
            family_features_list = []
            family_edge_index = [[], []]
            family_edge_attr = []
            family_intra_edge_index = [[], []]
            family_intra_edge_attr = []

            for i, entry in enumerate(tb_assessment.family_entries):
                features = self._extract_contact_features(entry, 'family')
                features = self._maybe_inject_p_base(features, entry, p_base_map)
                family_features_list.append(features)

                # 患者→家庭接触者边 + 边特征
                family_edge_index[0].append(0)  # 患者节点索引
                family_edge_index[1].append(i)  # 家庭接触者节点索引
                family_edge_attr.append(self._extract_edge_features(entry, 'family'))

            # 家庭成员间全连接（共居传播风险，文献支撑：Yang et al. 2021）
            # 边特征从家庭成员 entry 推断（复用 _extract_edge_features），
            # 而非使用硬编码默认值。i→j 边用 entry[i] 的接触特征，
            # j→i 边用 entry[j] 的接触特征（同户成员共享居住环境）。
            num_family = len(family_features_list)
            for i in range(num_family):
                for j in range(i + 1, num_family):
                    family_intra_edge_index[0].append(i)
                    family_intra_edge_index[1].append(j)
                    family_intra_edge_index[0].append(j)
                    family_intra_edge_index[1].append(i)
                    # i→j 边：使用成员 i 的接触特征
                    family_intra_edge_attr.append(
                        self._extract_edge_features(tb_assessment.family_entries[i], 'family'))
                    # j→i 边：使用成员 j 的接触特征
                    family_intra_edge_attr.append(
                        self._extract_edge_features(tb_assessment.family_entries[j], 'family'))

            if family_features_list:
                data['family'].x = torch.tensor(family_features_list, dtype=torch.float, device=device)
                data['patient', 'infects', 'family'].edge_index = torch.tensor(
                    family_edge_index, dtype=torch.long, device=device
                )
                data['patient', 'infects', 'family'].edge_attr = torch.tensor(
                    family_edge_attr, dtype=torch.float, device=device
                )
                if family_intra_edge_index[0]:
                    data['family', 'lives_with', 'family'].edge_index = torch.tensor(
                        family_intra_edge_index, dtype=torch.long, device=device
                    )
                    data['family', 'lives_with', 'family'].edge_attr = torch.tensor(
                        family_intra_edge_attr, dtype=torch.float, device=device
                    )

            # 社会接触者节点
            social_features_list = []
            social_edge_index = [[], []]
            social_edge_attr = []

            for i, entry in enumerate(tb_assessment.social_entries):
                features = self._extract_contact_features(entry, 'social')
                features = self._maybe_inject_p_base(features, entry, p_base_map)
                social_features_list.append(features)

                # 患者→社会接触者边 + 边特征
                social_edge_index[0].append(0)
                social_edge_index[1].append(i)
                social_edge_attr.append(self._extract_edge_features(entry, 'social'))

            if social_features_list:
                data['social'].x = torch.tensor(social_features_list, dtype=torch.float, device=device)
                data['patient', 'infects', 'social'].edge_index = torch.tensor(
                    social_edge_index, dtype=torch.long, device=device
                )
                data['patient', 'infects', 'social'].edge_attr = torch.tensor(
                    social_edge_attr, dtype=torch.float, device=device
                )

            # 确定特征维度：p_base_map 注入时 +1 维（接触者 30→31，患者/社区/区域对齐）
            p_base_active = p_base_map is not None
            feat_dim = 31 if p_base_active else 30

            # 社区节点（抽象，连接所有接触者，文献支撑：Zheng et al. 2024）
            data['community'].x = torch.zeros(1, feat_dim, dtype=torch.float, device=device)

            # 接触者到社区的边
            if num_family > 0:
                family_to_community_edge_index = [[i for i in range(num_family)], [0] * num_family]
                data['family', 'belongs_to', 'community'].edge_index = torch.tensor(
                    family_to_community_edge_index, dtype=torch.long, device=device
                )
                # 家庭到社区的边特征：从家庭成员 entry 推断（复用 _extract_edge_features），
                # 而非硬编码。社区层面接触强度通常低于家庭内，但仍由 entry 的接触
                # 频率/时长/通风等字段决定，保持特征语义一致性。
                family_to_community_attr = []
                for entry in tb_assessment.family_entries:
                    family_to_community_attr.append(
                        self._extract_edge_features(entry, 'family'))
                data['family', 'belongs_to', 'community'].edge_attr = torch.tensor(
                    family_to_community_attr, dtype=torch.float, device=device
                )

            num_social = len(social_features_list)
            if num_social > 0:
                social_to_community_edge_index = [[i for i in range(num_social)], [0] * num_social]
                data['social', 'belongs_to', 'community'].edge_index = torch.tensor(
                    social_to_community_edge_index, dtype=torch.long, device=device
                )
                # 社会到社区的边特征：从社会接触 entry 推断（复用 _extract_edge_features），
                # 而非硬编码。保持与 patient→social 边特征的同源推断逻辑。
                social_to_community_attr = []
                for entry in tb_assessment.social_entries:
                    social_to_community_attr.append(
                        self._extract_edge_features(entry, 'social'))
                data['social', 'belongs_to', 'community'].edge_attr = torch.tensor(
                    social_to_community_attr, dtype=torch.float, device=device
                )

            # 区域级节点（多尺度特征融合 v5.0：个体—家庭—社区—区域）
            # 区域为最高层抽象节点，聚合区域流行病学与卫生政策特征，
            # 使个体预测与区域传播预测共用同一套图结构。
            num_contacts = num_family + num_social
            region_features = self._extract_region_features(
                tb_assessment, num_contacts=num_contacts)
            if p_base_active:
                region_features = list(region_features) + [0.0]
            data['region'].x = torch.tensor(
                region_features, dtype=torch.float, device=device).unsqueeze(0)

            # 社区 → 区域 边（层次化归属，完成"个体—家庭—社区—区域"链）
            community_to_region_edge_index = [[0], [0]]
            data['community', 'belongs_to', 'region'].edge_index = torch.tensor(
                community_to_region_edge_index, dtype=torch.long, device=device
            )
            # 社区→区域边特征：区域层面接触强度代理（6 维，与边特征对齐）
            region_edge_attr = [
                min(1.0, 0.2 + num_contacts / 40.0),  # 频率评分
                0.1,                                    # 单次时长
                0.2,                                    # 持续周数
                0.5,                                    # 通风条件（默认）
                0.5,                                    # 接触距离（默认）
                0.6,                                    # 暴露场景（默认）
            ]
            data['community', 'belongs_to', 'region'].edge_attr = torch.tensor(
                region_edge_attr, dtype=torch.float, device=device).unsqueeze(0)

            return data

        # 患者特征归一化常数（统一管理，避免魔法数字）
        AGE_NORM = 100.0          # 年龄归一化除数
        COUGH_FREQ_NORM = 10.0    # 咳嗽频率归一化除数
        DELAY_DAYS_NORM = 60.0    # 延迟就诊天数归一化除数
        SYMPTOMS_NORM = 4.0       # 症状严重度归一化除数
        AGE_RISK_NORM = 65.0      # 年龄风险归一化除数
        COUGH_RISK_NORM = 20.0    # 咳嗽频率风险归一化除数
        DELAY_RISK_NORM = 30.0    # 延迟就诊风险归一化除数

        @staticmethod
        def _maybe_inject_p_base(features, entry, p_base_map):
            """把第 1 层输出 P_base 注入为接触者节点的第 31 维特征（堆叠）。

            仅当 p_base_map 非 None 时生效；按 record_id/_id 匹配，
            缺失记录回退 0（未评估个体的基线概率视为无信息）。"""
            if p_base_map is None:
                return features
            p_base = 0.0
            if isinstance(entry, dict):
                rid = entry.get('record_id') or entry.get('_id')
                if rid is not None:
                    p_base = float(p_base_map.get(rid, 0.0))
            return list(features[:30]) + [min(max(p_base, 0.0), 1.0)]

        def _extract_patient_features(self, tb_assessment):
            """提取患者特征（30维，v3.0 扩展）

            特征包括：
            1. age
            2. sputum_smear_positive
            3. has_cavity
            4. is_treated
            5. cough_freq
            6. ftd
            7. symptoms_severity
            8. age_immuno (交互特征：年龄 * 免疫状态)
            9. ftd_cough (交互特征：延迟就诊 * 咳嗽频率)
            10. cavity_smear (交互特征：空洞 * 涂阳)
            11. treated_delay (交互特征：治疗 * 延迟就诊)
            12. age_cough (交互特征：年龄 * 咳嗽频率)
            13-22: 附加特征（风险因素归一化）
            23-26: 潜伏感染状态 one-hot (患者默认 None)
            27-30: 疾病状态 one-hot (患者默认 Clinical)
            """
            features = []
            basic_info = tb_assessment.patient_info.get('basic_info', {})

            # 基础特征
            age = basic_info.get('age', 45)
            try:
                age = float(age)
            except (ValueError, TypeError):
                age = 45.0
            features.append(age / self.AGE_NORM)
            sputum_positive = 1.0 if basic_info.get('sputum_smear', 1) == 2 else 0.0
            features.append(sputum_positive)
            has_cavity = 1.0 if basic_info.get('has_cavity', 1) == 2 else 0.0
            features.append(has_cavity)
            is_treated = 1.0 if basic_info.get('treatment', 2) == 1 else 0.0
            features.append(is_treated)
            cough_freq = basic_info.get('cough_freq', 0)
            try:
                cough_freq = float(cough_freq)
            except (ValueError, TypeError):
                cough_freq = 0.0
            features.append(cough_freq / self.COUGH_FREQ_NORM)
            ftd = tb_assessment.patient_info.get('basic_info', {}).get('delay_days', 0) if tb_assessment.patient_info else 0
            try:
                ftd = float(ftd)
            except (ValueError, TypeError):
                ftd = 0.0
            features.append(ftd / self.DELAY_DAYS_NORM)
            symptoms = basic_info.get('symptoms', 2)
            try:
                symptoms = float(symptoms)
            except (ValueError, TypeError):
                symptoms = 2.0
            features.append(symptoms / self.SYMPTOMS_NORM)

            # 交互特征
            # 根据患者信息计算免疫抑制评分：年龄越大、有并发症风险越高
            age_factor = 0.2 + (age / self.AGE_NORM) * 0.5  # 年龄因素
            cavity_factor = has_cavity * 0.2  # 有空洞增加风险
            smear_factor = sputum_positive * 0.15  # 涂阳增加风险
            immunosuppressed = min(0.1 + age_factor + cavity_factor + smear_factor, 1.0)
            features.append((age / self.AGE_NORM) * immunosuppressed)
            features.append((ftd / self.DELAY_DAYS_NORM) * (cough_freq / self.COUGH_FREQ_NORM))
            features.append(has_cavity * sputum_positive)
            features.append(is_treated * (ftd / self.DELAY_DAYS_NORM))
            features.append((age / self.AGE_NORM) * (cough_freq / self.COUGH_FREQ_NORM))

            # 附加特征：风险因素归一化
            features.append(min(age / self.AGE_RISK_NORM, 1.0))  # 年龄风险
            features.append(min(cough_freq / self.COUGH_RISK_NORM, 1.0))  # 咳嗽频率风险
            features.append(sputum_positive * 0.8)  # 涂阳权重
            features.append(has_cavity * 0.7)  # 空洞权重
            features.append(min(ftd / self.DELAY_RISK_NORM, 1.0))  # 延迟就诊风险
            features.append(symptoms / self.SYMPTOMS_NORM * 0.6)  # 症状严重度权重
            features.append(0.0)  # 保留
            features.append(0.0)  # 保留
            features.append(0.0)  # 保留
            features.append(0.0)  # 保留

            # v3.0: SEIR 房室状态 one-hot (特征23-30)
            # 患者默认: 潜伏=None, 疾病=Clinical (活动性 TB)
            features.extend([0, 0, 0, 1])  # latent: None
            features.extend([0, 0, 1, 0])  # disease: Clinical

            return features[:30]

        def build_temporal_features(self, tb_assessment, contact_data, contact_type, max_weeks=52):
            """
            构建时序特征序列

            参数：
                tb_assessment: TB_Risk_Assessment 对象
                contact_data: 单个接触者的数据字典
                contact_type: 'family' 或 'social'
                max_weeks: 最大时间步数（默认 52 周）

            返回：
                seq: 时序特征矩阵 [max_weeks, 30]
                treatment_weeks: 时序治疗周数 [max_weeks, 1]

            文献支撑：
                - WHO 2024 指南中"接触者追踪时间窗"概念
            """
            seq = []
            treatment_weeks = []

            # 获取患者的 FTD (首次治疗日期)，同时兼容 ftd 和 delay_days 两种键名
            ftd = 0
            if hasattr(tb_assessment, 'patient_info') and tb_assessment.patient_info:
                if 'basic_info' in tb_assessment.patient_info:
                    basic_info = tb_assessment.patient_info['basic_info']
                    ftd = basic_info.get('ftd', basic_info.get('delay_days', 0))

            for week in range(max_weeks):
                # 创建该周的特征副本
                modified = contact_data.copy()

                # 根据周数调整累积暴露相关参数
                time_span = min(modified.get('time_span', 4), week + 1)
                modified['time_span'] = time_span

                # 重新计算累积暴露
                cumulative_exposure = calculate_cumulative_exposure(
                    modified.get('single_duration', 30),
                    modified.get('freq_density', 14 if contact_type == 'family' else 2),
                    time_span
                )
                modified['cumulative_exposure'] = cumulative_exposure

                # 计算该周的治疗阶段和传染性衰减
                treatment_week = max(0, week - ftd)

                # 获取该周的传染性因子
                if hasattr(tb_assessment, 'treatment_infectivity_factors'):
                    inf_factors = tb_assessment.treatment_infectivity_factors
                    if treatment_week == 0:
                        infectivity = inf_factors.get('pre_treatment', 0.85)
                    elif treatment_week < 2:
                        infectivity = inf_factors.get('early_treatment', 0.60)
                    elif treatment_week < 8:
                        infectivity = inf_factors.get('mid_treatment', 0.30)
                    elif treatment_week < 24:
                        infectivity = inf_factors.get('late_treatment', 0.10)
                    else:
                        infectivity = inf_factors.get('completed_treatment', 0.05)
                    modified['infectivity_factor'] = infectivity

                # 提取该周的特征
                features = self._extract_contact_features(modified, contact_type)
                seq.append(features)
                treatment_weeks.append([float(treatment_week)])

            return np.array(seq), np.array(treatment_weeks)

        def _calculate_cumulative_exposure(self, single_duration, freq_density, time_span):
            """计算累积暴露时间（小时）"""
            try:
                return float(single_duration) * float(freq_density) * float(time_span)
            except (ValueError, TypeError):
                return 30.0 * 14.0 * 4.0  # 默认值

        def build_temporal_graphs(self, tb_assessment, max_weeks=52, device=None):
            """
            构建时序图快照序列

            参数：
                tb_assessment: TB_Risk_Assessment 对象
                max_weeks: 最大时间步数
                device: 目标设备，默认使用构造时指定的设备

            返回：
                temporal_graphs: 图快照列表 [Data] * max_weeks
                temporal_features: 时序特征字典，包含各节点的特征序列

            文献支撑：
                - ST-GCN (Yan et al. 2018)
                - 时间快照图用于时空建模
            """
            if device is None:
                device = self.device
            temporal_graphs = []
            temporal_features = {
                'patient': [],
                'family': [],
                'social': []
            }

            # 构建基础图
            base_data = self.build_from_assessment(tb_assessment, device=device)

            for week in range(max_weeks):
                # 对于每个时间步，创建图的副本
                data = base_data.clone()

                # 更新节点特征（考虑时间演化）
                patient_features = self._extract_patient_features(tb_assessment)
                # 调整患者特征以反映时间变化（如治疗进展）
                if hasattr(tb_assessment, 'patient_info') and tb_assessment.patient_info:
                    basic_info = tb_assessment.patient_info.get('basic_info', {})
                    ftd = basic_info.get('ftd', basic_info.get('delay_days', 0))
                    treatment_week = week - ftd
                    if treatment_week > 0:
                        # 时间衰减仅对传染性相关的连续特征施加（文献：Wearne et al. 2015 涂阳转阴非线性曲线）。
                        # 二值特征（涂阳、空洞、BCG等）和人口学特征（年龄等）不随时间衰减。
                        # 特征维度索引（30维）：0=年龄, 1=累积暴露, 2=症状, 3=BCG, 4=TB史,
                        #   5=接触距离, 6=通风, 7=高危, 8=慢性病, 9=场景, 10=单次时长,
                        #   11=频次, 12=时间跨度, 13-22=交互特征, 23-26=潜伏状态one-hot,
                        #   27-30=疾病状态one-hot
                        infectivity_dims = {1, 5, 6, 9, 10, 11, 12}  # 暴露、距离、通风、场景、时长、频次、跨度
                        decay_factor = max(0.1, 1 - treatment_week / 52.0)
                        patient_features = [
                            f * decay_factor if i in infectivity_dims else f
                            for i, f in enumerate(patient_features)
                        ]
                data['patient'].x = torch.tensor(patient_features, dtype=torch.float, device=device).unsqueeze(0)

                # 存储时序特征
                temporal_features['patient'].append(patient_features)

                # 对每个时间步的图进行处理
                temporal_graphs.append(data)

            # 收集所有节点的完整时序特征
            for contact_type in ['family', 'social']:
                entries = getattr(tb_assessment, f'{contact_type}_entries', [])
                for entry in entries:
                    feat_seq, _ = self.build_temporal_features(tb_assessment, entry, contact_type, max_weeks)
                    temporal_features[contact_type].append(feat_seq)

            return temporal_graphs, temporal_features
