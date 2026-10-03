#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI 交互式患者信息收集器

零依赖 tkinter 或任何 GUI 库，可在纯控制台环境中使用。
从 assessment.py 迁移而来，独立于 TB_Risk_Assessment 类。
"""

# Section V: 验证常量统一从 constants.py 导入（单一真相源，与 assessment.py 一致）
from ..constants import (
    MIN_AGE, MAX_AGE, MIN_VENTILATION, MAX_VENTILATION,
    MAX_SINGLE_DURATION_MINUTES, MAX_TIME_SPAN_WEEKS,
    MAX_FAMILY_MEMBERS, MAX_SOCIAL_CONTACTS,
    MAX_COUGH_FREQ, MAX_TREATMENT_DURATION_MONTHS,
    MAX_DELAY_DAYS, MAX_FREQ_DENSITY,
    DEFAULT_FAMILY_FREQ, DEFAULT_SOCIAL_FREQ,
)


class InteractivePatientCollector:
    """CLI 交互式患者信息收集器

    用法：
        collector = InteractivePatientCollector()
        result = collector.collect_all()
        # result = {'patient_info': {...}, 'family_members': [...], 'social_contacts': [...]}
    """

    # -- 输入验证阈值常量（Section V: 从 constants.py 单一真相源引用） --
    MIN_AGE = MIN_AGE
    MAX_AGE = MAX_AGE
    MIN_VENTILATION = MIN_VENTILATION
    MAX_VENTILATION = MAX_VENTILATION
    MAX_SINGLE_DURATION_MINUTES = MAX_SINGLE_DURATION_MINUTES
    MAX_TIME_SPAN_WEEKS = MAX_TIME_SPAN_WEEKS
    MAX_FAMILY_MEMBERS = MAX_FAMILY_MEMBERS
    MAX_SOCIAL_CONTACTS = MAX_SOCIAL_CONTACTS
    MAX_COUGH_FREQ = MAX_COUGH_FREQ
    MAX_TREATMENT_DURATION_MONTHS = MAX_TREATMENT_DURATION_MONTHS
    MAX_DELAY_DAYS = MAX_DELAY_DAYS
    MAX_FREQ_DENSITY = MAX_FREQ_DENSITY
    DEFAULT_FAMILY_FREQ = DEFAULT_FAMILY_FREQ
    DEFAULT_SOCIAL_FREQ = DEFAULT_SOCIAL_FREQ

    def __init__(self):
        self.family_members = []
        self.social_contacts = []
        self.patient_info = {}

    # ==================== 公共 API ====================

    def collect_all(self) -> dict:
        """收集完整患者信息，返回包含 patient_info、family_members、social_contacts 的字典"""
        print("\n=== 结核病传播风险评估系统 ===")
        print("本系统将帮助评估结核病患者在家庭和社会网络中的潜在传播风险。")
        print("请根据提示输入相关信息，带*的为必填项。\n")

        patient_info = {
            'FCI': 0,
            'SNC': 0,
            'FLP': 0,
            'HRSP': 0,
            'FTD': 0,
            'basic_info': {},
        }

        print("\n--- 家庭生活情况 ---\n")

        family_count, family_members = self._input_family_members()
        self.family_members = family_members

        contact_count, social_contacts = self._input_social_contacts()
        self.social_contacts = social_contacts

        basic_info = self._input_patient_basic_info()
        patient_info['basic_info'] = basic_info

        patient_info['FLP'] = basic_info['flp_percentage']
        patient_info['HRSP'] = basic_info['hrsp_percentage']
        patient_info['FTD'] = basic_info['delay_days']

        self.patient_info = patient_info
        self._family_count = family_count
        self._contact_count = contact_count

        return {
            'patient_info': patient_info,
            'family_members': family_members,
            'social_contacts': social_contacts,
        }

    # ==================== 输入验证方法 ====================

    @staticmethod
    def _validate_input(prompt, input_type=int, min_value=None, max_value=None, default=None):
        """验证用户输入是否符合要求"""
        while True:
            try:
                user_input = input(prompt)
                if not user_input and default is not None:
                    return default
                value = input_type(user_input)
                if min_value is not None and value < min_value:
                    print(f"输入值不能小于{min_value}")
                    continue
                if max_value is not None and value > max_value:
                    print(f"输入值不能大于{max_value}")
                    continue
                return value
            except (EOFError, KeyboardInterrupt):
                print("\n输入已取消")
                return default
            except ValueError:
                print(f"请输入有效的{input_type.__name__}类型数据")

    @staticmethod
    def _validate_text(prompt, min_length=1, max_length=100, allow_empty=False, default=None):
        """验证文本输入，确保不为空且符合长度要求

        参数：
            prompt: 输入提示
            min_length: 最小长度
            max_length: 最大长度
            allow_empty: 是否允许为空
            default: 默认值
        """
        while True:
            try:
                user_input = input(prompt)
                if not user_input:
                    if default is not None:
                        return default
                    if allow_empty:
                        return ""
                    print("输入不能为空，请重新输入")
                    continue
                if len(user_input) < min_length:
                    print(f"输入长度不能小于{min_length}个字符")
                    continue
                if len(user_input) > max_length:
                    print(f"输入长度不能超过{max_length}个字符")
                    continue
                return user_input.strip()
            except (EOFError, KeyboardInterrupt):
                print("\n输入已取消")
                return default

    @staticmethod
    def _validate_choice(prompt, valid_choices, default=None):
        """验证用户选择是否在有效选项中"""
        while True:
            try:
                user_input = input(prompt)
                if not user_input and default is not None:
                    return default
                if user_input in valid_choices:
                    return user_input
                print(f"请输入有效的选项: {', '.join(valid_choices)}")
            except (EOFError, KeyboardInterrupt):
                print("\n输入已取消")
                return default

    # ==================== 信息收集方法 ====================

    def _input_family_members(self):
        """收集家庭成员信息"""
        family_count = self._validate_input(
            "请输入患者共同居住的家庭成员数量（包括患者本人）: ",
            int, 1, self.MAX_FAMILY_MEMBERS)

        family_members = []
        print("\n请依次输入每个家庭成员的详细信息：")

        for i in range(family_count):
            print(f"\n--- 家庭成员 {i + 1} ---\n")

            name = self._validate_text("请输入家庭成员姓名: ", min_length=1, max_length=50)
            age = self._validate_input("请输入家庭成员年龄: ", int, self.MIN_AGE, self.MAX_AGE)
            relationship = self._validate_text(
                "请输入与患者的关系（如：配偶、子女、父母等）: ", min_length=1, max_length=50)

            print("\n请输入与患者的接触频率信息：")
            single_duration = self._validate_input(
                "单次接触时长（分钟）: ", int, 0, self.MAX_SINGLE_DURATION_MINUTES, 30)
            freq_density = self._validate_input(
                "每周接触频次（次数/周，例如：每天2次则填14）: ",
                int, 0, self.MAX_FREQ_DENSITY, self.DEFAULT_FAMILY_FREQ)
            time_span = self._validate_input(
                "接触持续周期（周数）: ", int, 1, self.MAX_TIME_SPAN_WEEKS, 4)

            has_tb = self._validate_input("既往是否患有结核病史？(1=是, 0=否): ", int, 0, 1, 0)
            has_symptoms = self._validate_input(
                "是否有咳嗽、低热、盗汗等结核相关症状？(1=是, 0=否): ", int, 0, 1, 0)
            bcg_vaccine = self._validate_input("是否接种过卡介苗？(1=是, 0=否): ", int, 0, 1, 0)
            past_illness = self._validate_input(
                "是否有其他慢性疾病史？(1=是, 0=否): ", int, 0, 1, 0)

            past_illness_type = "none"
            if past_illness == 1:
                illness_choices = {"1": "hiv", "2": "diabetes", "3": "immunosuppressants", "4": "other"}
                illness_choice = self._validate_choice(
                    "请选择疾病类型：(1=HIV感染, 2=糖尿病, 3=使用免疫抑制剂, 4=其他): ",
                    ["1", "2", "3", "4"])
                past_illness_type = illness_choices[illness_choice]

            ventilation = self._validate_input(
                "家庭居住环境通风情况：(1=极差, 2=较差, 3=一般, 4=较好, 5=极好): ",
                int, 1, 5, 3)

            family_members.append({
                'name': name,
                'age': age,
                'relationship': relationship,
                'single_duration': single_duration,
                'freq_density': freq_density,
                'time_span': time_span,
                'cumulative_exposure': None,  # 由调用方计算，避免 CLI 层依赖 core
                'has_tb': has_tb,
                'has_symptoms': has_symptoms,
                'bcg_vaccine': bcg_vaccine,
                'past_illness': past_illness,
                'past_illness_type': past_illness_type,
                'contact_distance': 'close',
                'ventilation': ventilation,
                'exposure_setting': 'general',
            })

        return family_count, family_members

    def _input_social_contacts(self):
        """收集社会接触者信息"""
        contact_count = self._validate_input(
            "请输入患者的主要社会接触者数量（如同事、朋友、同学等）: ",
            int, 0, self.MAX_SOCIAL_CONTACTS)

        social_contacts = []

        if contact_count > 0:
            print("\n请依次输入每个社会接触者的详细信息：")

            for i in range(contact_count):
                print(f"\n--- 社会接触者 {i + 1} ---\n")

                name = self._validate_text("请输入社会接触者姓名: ", min_length=1, max_length=50)

                age_input = input("请输入社会接触者年龄（可选，直接回车跳过）: ")
                age = int(age_input) if age_input.isdigit() else None

                print("\n请输入与患者的接触频率信息：")
                single_duration = self._validate_input(
                    "单次接触时长（分钟）: ", int, 0, self.MAX_SINGLE_DURATION_MINUTES, 30)
                freq_density = self._validate_input(
                    "每周接触频次（次数/周，例如：每天2次则填14）: ",
                    int, 0, self.MAX_FREQ_DENSITY, self.DEFAULT_SOCIAL_FREQ)
                time_span = self._validate_input(
                    "接触持续周期（周数）: ", int, 1, self.MAX_TIME_SPAN_WEEKS, 4)

                is_high_risk = self._validate_input(
                    "是否为高危人群（如HIV感染者、糖尿病患者、免疫抑制者等）？(1=是, 0=否): ",
                    int, 0, 1, 0)
                past_illness = self._validate_input(
                    "是否有其他慢性疾病史？(1=是, 0=否): ", int, 0, 1, 0)

                past_illness_type = "none"
                if past_illness == 1:
                    illness_choices = {"1": "hiv", "2": "diabetes", "3": "immunosuppressants", "4": "other"}
                    illness_choice = self._validate_choice(
                        "请选择疾病类型：(1=HIV感染, 2=糖尿病, 3=使用免疫抑制剂, 4=其他): ",
                        ["1", "2", "3", "4"])
                    past_illness_type = illness_choices[illness_choice]

                bcg_vaccine = self._validate_input("是否接种过卡介苗？(1=是, 0=否): ", int, 0, 1, 0)
                ventilation = self._validate_input(
                    "接触场所通风情况：(1=极差, 2=较差, 3=一般, 4=较好, 5=极好): ",
                    int, 1, 5, 3)
                has_symptoms = self._validate_input(
                    "是否有咳嗽、低热、盗汗等结核相关症状？(1=是, 0=否): ", int, 0, 1, 0)

                distance_choices = {"1": "very_close", "2": "close", "3": "medium", "4": "far", "5": "distant"}
                distance_choice = self._validate_choice(
                    "接触距离：(1=极近(<0.5米), 2=近(0.5-1米), 3=中等(1-2米), 4=远(2-3米), 5=极远(>3米)): ",
                    ["1", "2", "3", "4", "5"])
                contact_distance = distance_choices[distance_choice]

                setting_choices = {"1": "crowded", "2": "closed", "3": "general", "4": "outdoor", "5": "oilfield_camp"}
                setting_choice = self._validate_choice(
                    "暴露场景：(1=拥挤场所(如公共交通), 2=密闭场所(如电梯), 3=一般场所, 4=户外场所, 5=油田营地): ",
                    ["1", "2", "3", "4", "5"])
                exposure_setting = setting_choices[setting_choice]

                social_contact = {
                    'name': name,
                    'single_duration': single_duration,
                    'freq_density': freq_density,
                    'time_span': time_span,
                    'cumulative_exposure': None,  # 由调用方计算
                    'is_high_risk': is_high_risk,
                    'past_illness': past_illness,
                    'past_illness_type': past_illness_type,
                    'bcg_vaccine': bcg_vaccine,
                    'ventilation': ventilation,
                    'has_symptoms': has_symptoms,
                    'contact_distance': contact_distance,
                    'exposure_setting': exposure_setting,
                }

                if age is not None:
                    social_contact['age'] = age

                social_contacts.append(social_contact)

        return contact_count, social_contacts

    def _input_patient_basic_info(self):
        """收集患者基本信息"""
        print("\n--- 患者基本信息 ---\n")

        age = self._validate_input("患者年龄: ", int, self.MIN_AGE, self.MAX_AGE)
        sputum_smear = self._validate_input("痰涂片结果：(1=涂阴, 2=涂阳): ", int, 1, 2)
        has_cavity = self._validate_input("是否有空洞（胸部X光或CT检查结果）：(1=无, 2=有): ", int, 1, 2)
        active_tb = self._validate_input("是否诊断为活动性结核：(1=是, 2=否): ", int, 1, 2)

        treatment = 2
        treatment_duration = 0
        if active_tb == 1:
            treatment = self._validate_input("是否正在接受抗结核治疗：(1=是, 2=否): ", int, 1, 2)
            if treatment == 1:
                treatment_duration = self._validate_input(
                    "治疗时长（月）: ", int, 0, self.MAX_TREATMENT_DURATION_MONTHS, 0)

        cough_freq = self._validate_input("平均每小时咳嗽次数（估算）: ", int, 0, self.MAX_COUGH_FREQ, 0)
        symptoms = self._validate_input("症状严重程度：(1=无症状, 2=轻度, 3=中度, 4=重度): ", int, 1, 4, 2)

        delay_days = 0
        if active_tb == 1:
            delay_days = self._validate_input(
                "从出现症状到确诊的天数：", int, 0, self.MAX_DELAY_DAYS, 0)

        family_living_conditions = self._validate_input(
            "家庭居住条件拥挤程度：(1=非常拥挤, 2=拥挤, 3=一般, 4=宽敞, 5=非常宽敞): ",
            int, 1, 5, 3)

        flp_percentage = self._validate_input(
            "家庭内潜伏结核感染比例（估算，%）: ", int, 0, 100, 5)

        hrsp_percentage = self._validate_input(
            "社会接触中的高风险人群比例（估算，%）: ", int, 0, 100, 10)

        return {
            'age': age,
            'sputum_smear': sputum_smear,
            'has_cavity': has_cavity,
            'active_tb': active_tb,
            'treatment': treatment,
            'treatment_duration': treatment_duration,
            'cough_freq': cough_freq,
            'symptoms': symptoms,
            'delay_days': delay_days,
            'family_living_conditions': family_living_conditions,
            'flp_percentage': flp_percentage,
            'hrsp_percentage': hrsp_percentage,
        }