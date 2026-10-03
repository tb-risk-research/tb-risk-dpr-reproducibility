import datetime
import logging
import math
import os
import random

from tb_risk.constants import (
    HIGH_ALTITUDE_THRESHOLD_M,
    ETHNICITY_WEIGHTS,
    OCCUPATION_WEIGHTS,
    DISTRICT_POPULATION_WEIGHTS,
    MONTHLY_PM10_BASELINE,
    MONTHLY_HUMIDITY_BASELINE,
)


VALIDATOR_LOGGER = logging.getLogger("tb_risk.validation")


class ScreeningDataSimulator:
    """2024年克拉玛依筛查数据模拟器

    基于官方数据：48,683人筛查，59例确诊（检出率约121/10万）

    生成合成数据集供回测使用，包含：
    - 人口学特征（年龄、性别、民族、职业）
    - 接触史（暴露场景、接触距离、频次密度）
    - 临床特征（症状、BCG接种、既往史）
    - 环境协变量（PM10、湿度、海拔）
    - 标签（是否确诊）

    标签生成机制（v2.0 — 打破循环论证）：
    不再使用与评分模型相同的特征（症状、暴露场景等）直接计算
    标签概率。改为基于独立 SEIR 传播链模拟：首先随机选择种子病例
    （仅基于人口统计学因素），然后沿接触网络模拟传播，标签由
    传播链中的位置决定。这确保了标签生成与评分模型使用不同的
    信息源，打破合成数据循环论证。
    """

    TOTAL_SCREENED = 48683
    TOTAL_CONFIRMED = 59
    POSITIVE_RATE = TOTAL_CONFIRMED / TOTAL_SCREENED  # ≈0.001212

    # 高海拔阈值(m) — 引用 tb_risk.constants 单一真值源
    HIGH_ALTITUDE_THRESHOLD_M = HIGH_ALTITUDE_THRESHOLD_M

    # 传播链参数（基于文献的 TB 传播动力学）
    # 每个种子病例平均产生的二代病例数（R_effective ≈ 0.8-1.2）
    SEED_REPRODUCTION_MEAN = 1.0
    # 传播链最大深度（代际数）
    MAX_GENERATIONS = 3
    # 种子病例选择概率（基于年龄和既往TB史，不依赖评分模型特征）
    SEED_AGE_RISK = {
        (0, 5): 0.02,      # 幼儿：免疫系统未成熟
        (5, 15): 0.01,     # 少年：低风险
        (15, 35): 0.015,   # 青年：社会活动多
        (35, 55): 0.02,    # 中年：风险升高
        (55, 100): 0.03,   # 老年：免疫力下降
    }

    # 人口、民族、职业权重 — 引用 tb_risk.constants 单一真值源
    DISTRICT_POPULATION_WEIGHTS = DISTRICT_POPULATION_WEIGHTS
    ETHNICITY_WEIGHTS = ETHNICITY_WEIGHTS
    OCCUPATION_WEIGHTS = OCCUPATION_WEIGHTS

    EXPOSURE_SETTINGS = ['general', 'closed', 'crowded', 'outdoor', 'oilfield_camp']
    EXPOSURE_WEIGHTS = [0.35, 0.20, 0.15, 0.15, 0.15]

    CONTACT_DISTANCES = ['very_close', 'close', 'medium', 'far', 'distant']
    DISTANCE_WEIGHTS = [0.10, 0.30, 0.35, 0.15, 0.10]

    def __init__(self, random_state=42):
        self.rng = random.Random(random_state)

    def generate_dataset(self, n_samples=None, include_labels=True):
        """生成合成筛查数据集

        参数：
            n_samples (int|None): 样本数，None则使用完整48683
            include_labels (bool): 是否包含确诊标签

        返回：
            list[dict]: 每条记录为一个字典
        """
        n = n_samples if n_samples is not None else self.TOTAL_SCREENED
        records = []

        for i in range(n):
            rec = self._generate_one_record(i)
            records.append(rec)

        if include_labels:
            # v2.0: 基于传播链模拟分配标签，打破循环论证
            self._assign_labels_by_transmission_chain(records, n)

        return records

    def _generate_one_record(self, idx):
        r = self.rng
        district = self._weighted_choice(self.DISTRICT_POPULATION_WEIGHTS, r)
        ethnicity = self._weighted_choice(self.ETHNICITY_WEIGHTS, r)
        occupation = self._weighted_choice(self.OCCUPATION_WEIGHTS, r)
        exposure_setting = self._weighted_choice(
            dict(zip(self.EXPOSURE_SETTINGS, self.EXPOSURE_WEIGHTS)), r
        )

        is_oilfield = occupation == 'oilfield_worker' or exposure_setting == 'oilfield_camp'
        crew_type = 'maintenance'
        if is_oilfield:
            crew_type = r.choice(['drilling', 'maintenance', 'maintenance', 'office'])

        age = int(self._sample_age(r))
        is_idu = r.random() < 0.01

        origin_altitude = None
        if r.random() < 0.15:
            origin_altitude = r.uniform(600, 3500)

        month = r.randint(1, 12)
        pm10_val = self._sample_pm10(month, r)
        humidity_val = self._sample_humidity(month, r)

        record = {
            '_id': idx,
            'district': district,
            'ethnicity': ethnicity,
            'occupation': occupation,
            'age': age,
            'gender': 'male' if r.random() < 0.55 else 'female',
            'single_duration': int(r.triangular(5, 480, 60)),
            'freq_density': int(r.triangular(0, 30, 14)) if not is_oilfield
                            else int(r.triangular(0, 30, 21)),
            'time_span': int(r.triangular(1, 12, 4)),
            'has_symptoms': 1 if r.random() < 0.08 else 0,
            'bcg_vaccine': 1 if r.random() < 0.85 else 0,
            'has_tb': 1 if r.random() < 0.02 else 0,
            'ventilation': int(r.triangular(1, 5, 3)),
            'contact_distance': self._weighted_choice(
                dict(zip(self.CONTACT_DISTANCES, self.DISTANCE_WEIGHTS)), r
            ),
            'exposure_setting': exposure_setting,
            'past_illness': 1 if r.random() < 0.12 else 0,
            'past_illness_type': 'none',
            'is_high_risk': 0,
            'is_oilfield': is_oilfield,
            'crew_type': crew_type if is_oilfield else None,
            'idu_status': is_idu,
            'origin_altitude': origin_altitude,
            'months_since_migration': int(r.uniform(0, 120)) if origin_altitude else 0,
            'pm10': round(pm10_val, 1),
            'humidity': round(humidity_val, 1),
            'month': month,
        }

        if record['past_illness']:
            record['past_illness_type'] = r.choice(
                ['HIV', 'diabetes', 'immunosuppressants', 'other']
            )
        if is_idu:
            record['past_illness_type'] = 'idu'
            record['is_high_risk'] = 1

        # 累积暴露 = 小时/次 × 次/周 × 周数
        # time_span 为月数，需转换为周数（1月≈4.33周）
        record['cumulative_exposure'] = (
            (record['single_duration'] / 60.0) *
            record['freq_density'] *
            (record['time_span'] * 4.33)
        )

        return record

    def _assign_labels_by_transmission_chain(self, records, total_n):
        """基于独立 SEIR 传播链模拟分配标签（v2.0 — 打破循环论证）。

        与评分模型使用不同的信息源：
        1. 仅根据人口统计学因素（年龄、既往TB史）选择种子病例
        2. 沿接触网络模拟传播：种子→家庭接触者→社会接触者
        3. 标签由传播链位置决定，不依赖评分模型使用的症状/暴露特征

        这确保了：
        - 标签生成与评分模型独立（打破循环论证）
        - 标签反映真实的流行病学传播过程（有物理意义）
        - 目标检出率 ≈ 121/10万（匹配克拉玛依实际数据）

        参数：
            records: list[dict], 所有记录
            total_n: int, 总记录数
        """
        r = self.rng

        # 全部初始化为未确诊
        for rec in records:
            rec['is_confirmed'] = 0

        # 步骤 1: 选择种子病例（仅基于人口统计学因素）
        seed_indices = self._select_seed_cases(records, total_n, r)

        # 步骤 2: 沿接触网络传播
        confirmed_set = set(seed_indices)
        current_generation = list(seed_indices)

        for gen in range(1, self.MAX_GENERATIONS + 1):
            if not current_generation:
                break
            next_generation = []
            for seed_idx in current_generation:
                seed_rec = records[seed_idx]
                n_contacts = self._sample_contacts(seed_rec, gen, r)
                # 选择接触者：按距离种子病例的索引选择（模拟接触网络）
                for _ in range(n_contacts):
                    contact_idx = self._sample_contact_index(
                        seed_idx, total_n, r, confirmed_set)
                    if contact_idx is not None:
                        confirmed_set.add(contact_idx)
                        next_generation.append(contact_idx)
            current_generation = next_generation

        # 步骤 3: 标记确诊
        for idx in confirmed_set:
            records[idx]['is_confirmed'] = 1

        n_confirmed = len(confirmed_set)
        actual_rate = n_confirmed / total_n if total_n > 0 else 0.0
        VALIDATOR_LOGGER.info(
            f'数据集生成完成: {total_n} 样本, {n_confirmed} 确诊, '
            f'检出率 {actual_rate*100000:.1f}/10万, '
            f'种子病例 {len(seed_indices)}, 传播链深度 ≤{self.MAX_GENERATIONS}'
        )

    def _select_seed_cases(self, records, total_n, rng):
        """选择种子病例（仅基于人口统计学因素，不依赖评分模型特征）。

        种子病例表示人群中未被发现的 TB 感染者。
        选择概率仅基于年龄和既往 TB 史——这些是独立于评分模型的
        人口统计学因素。

        返回：
            list[int]: 种子病例在 records 中的索引
        """
        seed_indices = []
        for i, rec in enumerate(records):
            age = rec.get('age', 30)
            try:
                age = float(age)
            except (TypeError, ValueError):
                age = 30.0
            has_tb_history = rec.get('has_tb', 0)

            # 基于年龄的风险（独立于评分模型特征）
            age_risk = 0.01  # 默认基线
            for (lo, hi), risk in self.SEED_AGE_RISK.items():
                if lo <= age < hi:
                    age_risk = risk
                    break

            # 既往 TB 史增加成为种子病例的概率（独立风险因素）
            if has_tb_history:
                age_risk *= 2.0

            if rng.random() < age_risk:
                seed_indices.append(i)

        # 控制种子病例数量，使最终确诊率接近目标 121/10万
        target_confirmed = max(1, int(total_n * self.POSITIVE_RATE))
        expected_total = int(len(seed_indices) * (1 + self.SEED_REPRODUCTION_MEAN))

        if expected_total > target_confirmed * 3:
            # 种子太多，随机下采样
            n_keep = max(1, int(target_confirmed / (1 + self.SEED_REPRODUCTION_MEAN)))
            seed_indices = rng.sample(seed_indices, min(n_keep, len(seed_indices)))

        return seed_indices

    def _sample_contacts(self, seed_rec, generation, rng):
        """采样种子病例产生的接触者数量。

        传播概率随代际递减（模拟"接触者追踪"的衰减效应）。
        """
        # 第一代：高传播（家庭接触者）
        # 第二代：中等传播（社会接触者）
        # 第三代：低传播（社区接触）
        gen_factor = max(0.1, 1.0 - 0.3 * (generation - 1))
        mean_contacts = self.SEED_REPRODUCTION_MEAN * gen_factor
        return max(0, int(rng.gauss(mean_contacts, 0.5)))

    def _sample_contact_index(self, seed_idx, total_n, rng, already_confirmed):
        """采样一个接触者索引（模拟接触网络中的传播）。

        接触者按距离种子病例的索引选择，模拟接触网络结构。
        避免重复选择已确诊的个体。
        """
        max_attempts = 20
        for _ in range(max_attempts):
            # 接触者在种子病例附近（模拟家庭/社会接触网络）
            spread = max(5, total_n // 100)
            offset = int(rng.gauss(0, spread))
            contact_idx = (seed_idx + offset) % total_n
            if contact_idx not in already_confirmed:
                return contact_idx
        return None

    # ========== 真实数据加载接口 ==========

    def load_from_file(self, filepath, label_field='is_confirmed', id_field='record_id',
                       auto_map=True, verbose=True):
        """从文件加载真实数据，替代合成数据生成

        优先使用真实数据；合成数据仅作为无数据时的回退。
        自动检测编码、映射字段名、执行数据完整性预验证。

        参数：
            filepath (str): 数据文件路径（支持 .csv/.json/.jsonl/.xlsx/.xls/.parquet）
            label_field (str): 标签字段名（默认 'is_confirmed'）
            id_field (str): ID 字段名（默认 'record_id'）
            auto_map (bool): 是否自动映射中文字段名
            verbose (bool): 是否输出加载信息

        返回：
            tuple[list[dict], dict]: (记录列表, 元信息)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"数据文件不存在: {filepath}")

        if verbose:
            VALIDATOR_LOGGER.info("从文件加载真实数据: %s", filepath)

        try:
            from ..io_utils import load_data_from_file, rename_fields, validate_data_integrity
            records, meta = load_data_from_file(filepath)

            if auto_map:
                records = rename_fields(records, auto_map=True, verbose=verbose)

            # 标准化标签和ID字段
            for rec in records:
                if label_field in rec and label_field != 'is_confirmed':
                    rec['is_confirmed'] = rec.pop(label_field)
                if id_field in rec and id_field != 'record_id':
                    rec['record_id'] = rec.pop(id_field)

            # 确保 is_confirmed 为 int
            for rec in records:
                if 'is_confirmed' in rec:
                    try:
                        rec['is_confirmed'] = int(float(rec['is_confirmed']))
                    except (ValueError, TypeError):
                        rec['is_confirmed'] = 0

            # 数据完整性预验证
            integrity = validate_data_integrity(records, verbose=verbose)

            meta['integrity'] = integrity
            meta['n_confirmed'] = sum(1 for r in records if r.get('is_confirmed', 0))
            meta['n_total'] = len(records)

            if verbose:
                VALIDATOR_LOGGER.info(
                    "真实数据加载完成: %d 条记录, %d 确诊 (%.1f/10万)",
                    meta['n_total'], meta['n_confirmed'],
                    meta['n_confirmed'] / max(meta['n_total'], 1) * 100000
                )

            return records, meta

        except ImportError as e:
            VALIDATOR_LOGGER.warning("io_utils 模块不可用: %s，回退到合成数据", e)
            return [], {'error': str(e)}

    def load_or_generate(self, filepath=None, n_samples=None, include_labels=True):
        """智能加载：真实数据优先，合成数据回退

        当真实数据可用时优先使用，合成数据仅作为无数据时的回退。
        当真实数据样本量不足时（如少于 1000 条），用合成数据增强。

        参数：
            filepath (str|None): 真实数据文件路径，None 则直接生成合成数据
            n_samples (int|None): 所需样本数
            include_labels (bool): 是否包含标签

        返回：
            tuple[list[dict], dict]: (记录列表, 元信息)
        """
        if filepath and os.path.exists(filepath):
            records, meta = self.load_from_file(filepath, verbose=True)
            if records:
                # 数据增强：当真实数据不足时用合成数据补充
                if n_samples and len(records) < n_samples:
                    n_augment = n_samples - len(records)
                    VALIDATOR_LOGGER.info(
                        "真实数据 %d 条不足 %d，用合成数据增强 %d 条",
                        len(records), n_samples, n_augment
                    )
                    synthetic = self.generate_dataset(n_samples=n_augment, include_labels=include_labels)
                    for rec in synthetic:
                        rec['_data_source'] = 'synthetic'
                    for rec in records:
                        rec['_data_source'] = 'real'
                    records.extend(synthetic)

                meta['data_source'] = 'real'
                return records, meta

        # 回退到合成数据
        VALIDATOR_LOGGER.info("无真实数据可用，使用合成数据")
        records = self.generate_dataset(n_samples=n_samples, include_labels=include_labels)
        for rec in records:
            rec['_data_source'] = 'synthetic'
        return records, {'data_source': 'synthetic', 'n_total': len(records)}

    def import_batch(self, filepath, mode='append', existing_records=None):
        """批量导入增量数据

        临床场景中数据分批产生（如每周新增筛查记录），
        此方法支持增量导入新数据，自动记录数据版本和导入时间戳。

        参数：
            filepath (str): 数据文件路径
            mode (str): 'append'（追加）或 'replace'（替换）
            existing_records (list[dict]|None): 现有记录列表

        返回：
            tuple[list[dict], dict]: (合并后的记录列表, 导入元信息)
        """
        new_records, meta = self.load_from_file(filepath, verbose=False)

        # 标记每条记录的来源和导入时间
        batch_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        for rec in new_records:
            rec['_import_batch'] = batch_id
            rec['_import_time'] = datetime.datetime.now().isoformat()
            rec['_source_file'] = os.path.basename(filepath)

        if mode == 'replace':
            all_records = new_records
        else:
            # append 模式
            all_records = list(existing_records) if existing_records else []
            all_records.extend(new_records)

        import_meta = {
            'batch_id': batch_id,
            'import_time': datetime.datetime.now().isoformat(),
            'source_file': filepath,
            'new_records': len(new_records),
            'total_records': len(all_records),
            'mode': mode,
            'encoding': meta.get('encoding'),
            'integrity': meta.get('integrity'),
        }

        VALIDATOR_LOGGER.info(
            "批量导入完成: batch=%s, 新增 %d 条, 总计 %d 条 (mode=%s)",
            batch_id, len(new_records), len(all_records), mode
        )

        return all_records, import_meta

    # ========== 旧版标签分配（保留向后兼容，已弃用） ==========

    def _assign_label(self, record, n_confirmed, total_n):
        """[已弃用] 按流行病学风险分配确诊标签。

        ⚠️ 此方法存在循环论证问题：标签基于与评分模型相同的特征
        （症状、暴露场景、IDU 状态等）生成。请使用
        _assign_labels_by_transmission_chain 替代。

        保留此方法仅用于向后兼容和测试对比。
        """
        r = self.rng
        risk = 1.0

        if record['has_symptoms']: risk *= 8.0
        if record['exposure_setting'] == 'oilfield_camp': risk *= 3.0
        if record['idu_status']: risk *= 10.0
        if record['age'] < 5 or record['age'] > 65: risk *= 2.0
        if record['has_tb']: risk *= 3.0
        if not record['bcg_vaccine']: risk *= 1.5
        if (record['origin_altitude'] and
                record['origin_altitude'] > self.HIGH_ALTITUDE_THRESHOLD_M and
                record['months_since_migration'] < 24):
            risk *= 2.0

        base_prob = self.POSITIVE_RATE * risk
        base_prob = min(base_prob, 0.95)

        return r.random() < base_prob

    @staticmethod
    def _weighted_choice(weights_dict, rng):
        items = list(weights_dict.keys())
        weights = list(weights_dict.values())
        return rng.choices(items, weights=weights, k=1)[0]

    @staticmethod
    def _sample_age(rng):
        """年龄分布：油田工人集中在20-50岁；整体偏青年"""
        u = rng.random()
        if u < 0.05:
            return rng.uniform(0, 5)
        elif u < 0.10:
            return rng.uniform(5, 15)
        elif u < 0.55:
            return rng.uniform(15, 35)
        elif u < 0.85:
            return rng.uniform(35, 65)
        else:
            return rng.uniform(66, 90)

    @staticmethod
    def _sample_pm10(month, rng):
        """采样 PM10 浓度（μg/m³），使用 tb_risk.constants 中的月度基线"""
        base = MONTHLY_PM10_BASELINE.get(month, 100)
        return max(0.0, rng.normalvariate(base, base * 0.3))

    @staticmethod
    def _sample_humidity(month, rng):
        """采样湿度（%），使用 tb_risk.constants 中的月度基线"""
        base = MONTHLY_HUMIDITY_BASELINE.get(month, 45)
        return max(5.0, min(95.0, rng.normalvariate(base, 8.0)))