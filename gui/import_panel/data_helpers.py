#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 导入面板 - 数据助手 Mixin（_standardize_contact_fields / _add_contact_from_dict / _update_gui_from_import）"""

from ._shared import *


class DataHelpersMixin:
    """数据导入/导出助手方法"""

    def _standardize_contact_fields(self, contact, contact_type):
        """标准化接触者字段名

        Args:
            contact: 原始接触者数据
            contact_type: 'family' 或 'social'

        Returns:
            dict: 标准化后的接触者数据
        """
        standardized = dict(contact)

        # 姓名映射
        if 'member_name' in standardized and 'name' not in standardized:
            standardized['name'] = standardized['member_name']
        elif 'contact_name' in standardized and 'name' not in standardized:
            standardized['name'] = standardized['contact_name']

        # 年龄映射
        if 'member_age' in standardized and 'age' not in standardized:
            standardized['age'] = standardized['member_age']
        elif 'contact_age' in standardized and 'age' not in standardized:
            standardized['age'] = standardized['contact_age']

        return standardized

    def _add_contact_from_dict(self, contact, contact_type):
        """从字典添加接触者

        Args:
            contact: 接触者数据字典
            contact_type: 'family' 或 'social'
        """
        if contact_type == 'family':
            single_duration = contact.get('single_duration', 30)
            freq_density = contact.get('freq_density', 14)
            time_span = contact.get('time_span', 4)
            entry = {
                'name': str(contact.get('name', f'家庭成员{len(self.family_entries)+1}')),
                'age': str(contact.get('age', 30)),
                'relationship': str(contact.get('relationship', '')),
                'single_duration': str(single_duration),
                'freq_density': str(freq_density),
                'time_span': str(time_span),
                'has_symptoms': "是" if _is_yes(contact.get('has_symptoms', 0)) else "否",
                'bcg_vaccine': "是" if _is_yes(contact.get('bcg_vaccine', 1)) else "否",
                'ventilation': str(contact.get('ventilation', 3)),
                'contact_distance': self.DISTANCE_MAPPING_REVERSE.get(
                    contact.get('contact_distance', 'medium'),
                    contact.get('contact_distance', '中等')
                ),
                'exposure_setting': self.SETTING_MAPPING_REVERSE.get(
                    contact.get('exposure_setting', 'general'),
                    contact.get('exposure_setting', '一般')
                ),
                'has_tb': "是" if _is_yes(contact.get('has_tb', 0)) else "否",
                'past_illness': "是" if _is_yes(contact.get('past_illness', 0)) else "否",
                'past_illness_type': self.ILLNESS_TYPE_MAPPING_REVERSE.get(
                    contact.get('past_illness_type', 'none'),
                    contact.get('past_illness_type', 'none')
                ) if _is_yes(contact.get('past_illness', 0)) else "none",
                'ethnicity': str(contact.get('ethnicity', '汉族')),
                'origin_altitude': self._safe_int_convert(
                    contact.get('origin_altitude', 300), 'origin_altitude', default=300),
                'workplace_type': str(contact.get('workplace_type', '非油田')),
                'idu_status': '是' if _is_yes(contact.get('idu_status', False)) else '否',
                'district': str(contact.get('district', '克拉玛依区')),
                'months_since_migration': self._safe_int_convert(
                    contact.get('months_since_migration', 0), 'months_since_migration', default=0)
            }
            # 计算并添加累积暴露时长（安全类型转换）
            try:
                sd = int(single_duration) if single_duration != '' else 30
                fd = int(freq_density) if freq_density != '' else self.DEFAULT_FAMILY_FREQ
                ts = int(time_span) if time_span != '' else 4
            except (ValueError, TypeError):
                sd, fd, ts = 30, self.DEFAULT_FAMILY_FREQ, 4
            cumulative_exposure = calculate_cumulative_exposure(
                sd, fd, ts, entry.get('workplace_type', '非油田'))
            entry['cumulative_exposure'] = cumulative_exposure

            # 补充唯一ID和标签
            self._contact_id_counter += 1
            contact_id = f'family_{self._contact_id_counter}'
            entry['_id'] = contact_id
            if 'diagnosed' in contact:
                self.contact_labels[contact_id] = 1 if _is_yes(contact['diagnosed']) else 0
            else:
                self.contact_labels[contact_id] = 0

            self.family_entries.append(entry)

        else:
            single_duration = contact.get('single_duration', 30)
            freq_density = contact.get('freq_density', 2)
            time_span = contact.get('time_span', 4)
            entry = {
                'name': str(contact.get('name', f'社会接触者{len(self.social_entries)+1}')),
                'age': str(contact.get('age', 30)),
                'single_duration': str(single_duration),
                'freq_density': str(freq_density),
                'time_span': str(time_span),
                'is_high_risk': "是" if _is_yes(contact.get('is_high_risk', 0)) else "否",
                'has_symptoms': "是" if _is_yes(contact.get('has_symptoms', 0)) else "否",
                'bcg_vaccine': "是" if _is_yes(contact.get('bcg_vaccine', 1)) else "否",
                'ventilation': str(contact.get('ventilation', 3)),
                'contact_distance': self.DISTANCE_MAPPING_REVERSE.get(
                    contact.get('contact_distance', 'medium'),
                    contact.get('contact_distance', '中等')
                ),
                'exposure_setting': self.SETTING_MAPPING_REVERSE.get(
                    contact.get('exposure_setting', 'general'),
                    contact.get('exposure_setting', '一般')
                ),
                'has_tb': "是" if _is_yes(contact.get('has_tb', 0)) else "否",
                'past_illness': "是" if _is_yes(contact.get('past_illness', 0)) else "否",
                'past_illness_type': self.ILLNESS_TYPE_MAPPING_REVERSE.get(
                    contact.get('past_illness_type', 'none'),
                    contact.get('past_illness_type', 'none')
                ) if _is_yes(contact.get('past_illness', 0)) else "none",
                'ethnicity': str(contact.get('ethnicity', '汉族')),
                'origin_altitude': self._safe_int_convert(
                    contact.get('origin_altitude', 300), 'origin_altitude', default=300),
                'workplace_type': str(contact.get('workplace_type', '非油田')),
                'idu_status': '是' if _is_yes(contact.get('idu_status', False)) else '否',
                'district': str(contact.get('district', '克拉玛依区')),
                'months_since_migration': self._safe_int_convert(
                    contact.get('months_since_migration', 0), 'months_since_migration', default=0)
            }
            # 计算并添加累积暴露时长（安全类型转换）
            try:
                sd = int(single_duration) if single_duration != '' else 30
                fd = int(freq_density) if freq_density != '' else self.DEFAULT_SOCIAL_FREQ
                ts = int(time_span) if time_span != '' else 4
            except (ValueError, TypeError):
                sd, fd, ts = 30, self.DEFAULT_SOCIAL_FREQ, 4
            cumulative_exposure = calculate_cumulative_exposure(
                sd, fd, ts, entry.get('workplace_type', '非油田'))
            entry['cumulative_exposure'] = cumulative_exposure

            # 补充唯一ID和标签
            self._contact_id_counter += 1
            contact_id = f'social_{self._contact_id_counter}'
            entry['_id'] = contact_id
            if 'diagnosed' in contact:
                self.contact_labels[contact_id] = 1 if _is_yes(contact['diagnosed']) else 0
            else:
                self.contact_labels[contact_id] = 0

            self.social_entries.append(entry)

    def _update_gui_from_import(self):
        """从导入的数据更新 GUI（Treeview方式）"""
        clinical_fields = {'sputum_smear', 'has_cavity', 'active_tb', 'treatment'}
        if self.patient_info and 'basic_info' in self.patient_info:
            basic = self.patient_info['basic_info']
            for key, value in basic.items():
                if key in self.basic_info_vars and value is not None:
                    var = self.basic_info_vars[key]
                    if isinstance(var, tk.IntVar):
                        if key in clinical_fields and isinstance(value, str):
                            value_str = value.strip()
                            if value_str in CLINICAL_TEXT_MAP:
                                value = CLINICAL_TEXT_MAP[value_str]
                        var.set(int(value))
                    elif isinstance(var, tk.StringVar):
                        var.set(str(value))
                    elif isinstance(var, tk.Text):
                        var.delete('1.0', tk.END)
                        var.insert('1.0', str(value))

        self.family_entries = []
        for member in self.family_members:
            contact_distance = member.get('contact_distance', 'close')
            exposure_setting = member.get('exposure_setting', 'general')
            single_duration = member.get('single_duration', 30)
            freq_density = member.get('freq_density', self.DEFAULT_FAMILY_FREQ)
            time_span = member.get('time_span', 4)
            entry = {
                'name': member.get('name', ''),
                'age': member.get('age', 30),
                'relationship': member.get('relationship', '其他'),
                'single_duration': single_duration,
                'freq_density': freq_density,
                'time_span': time_span,
                'ventilation': str(member.get('ventilation', 3)),
                'contact_distance': self.DISTANCE_MAPPING_REVERSE.get(contact_distance, contact_distance),
                'exposure_setting': self.SETTING_MAPPING_REVERSE.get(exposure_setting, exposure_setting),
                'has_symptoms': "是" if _is_yes(member.get('has_symptoms', 0)) else "否",
                'bcg_vaccine': "是" if _is_yes(member.get('bcg_vaccine', 1)) else "否",
                'has_tb': "是" if _is_yes(member.get('has_tb', 0)) else "否",
                'past_illness': "是" if _is_yes(member.get('past_illness', 0)) else "否",
                'past_illness_type': self.ILLNESS_TYPE_MAPPING_REVERSE.get(member.get('past_illness_type', ''), member.get('past_illness_type', 'none')) if _is_yes(member.get('past_illness', 0)) else "none",
                'ethnicity': str(member.get('ethnicity', '汉族')),
                'origin_altitude': int(member.get('origin_altitude', 300)),
                'workplace_type': str(member.get('workplace_type', '非油田')),
                'idu_status': '是' if _is_yes(member.get('idu_status', False)) else '否',
                'district': str(member.get('district', '克拉玛依区')),
                'months_since_migration': int(member.get('months_since_migration', 0))
            }
            # 计算并添加累积暴露时长
            cumulative_exposure = calculate_cumulative_exposure(single_duration, freq_density, time_span)
            entry['cumulative_exposure'] = cumulative_exposure
            # 补充唯一ID和标签
            if '_id' not in member:
                self._contact_id_counter += 1
                contact_id = f'family_{self._contact_id_counter}'
                entry['_id'] = contact_id
                if 'diagnosed' in member:
                    self.contact_labels[contact_id] = 1 if _is_yes(member['diagnosed']) else 0
                else:
                    self.contact_labels[contact_id] = 0
            else:
                entry['_id'] = member['_id']
                if member['_id'] not in self.contact_labels:
                    self.contact_labels[member['_id']] = 0
            self.family_entries.append(entry)
        # 优先级六：通过 adapter 统一刷新（消除 ui_mode 分支）
        if self.adapter is not None:
            self.adapter.refresh_family_tree()

        self.social_entries = []
        for contact in self.social_contacts:
            contact_distance = contact.get('contact_distance', 'medium')
            exposure_setting = contact.get('exposure_setting', 'general')
            single_duration = contact.get('single_duration', 30)
            freq_density = contact.get('freq_density', self.DEFAULT_SOCIAL_FREQ)
            time_span = contact.get('time_span', 4)
            entry = {
                'name': contact.get('name', ''),
                'age': contact.get('age', 30),
                'single_duration': single_duration,
                'freq_density': freq_density,
                'time_span': time_span,
                'is_high_risk': "是" if _is_yes(contact.get('is_high_risk', 0)) else "否",
                'has_symptoms': "是" if _is_yes(contact.get('has_symptoms', 0)) else "否",
                'bcg_vaccine': "是" if _is_yes(contact.get('bcg_vaccine', 1)) else "否",
                'ventilation': str(contact.get('ventilation', 3)),
                'contact_distance': self.DISTANCE_MAPPING_REVERSE.get(contact_distance, contact_distance),
                'exposure_setting': self.SETTING_MAPPING_REVERSE.get(exposure_setting, exposure_setting),
                'has_tb': "是" if _is_yes(contact.get('has_tb', 0)) else "否",
                'past_illness': "是" if _is_yes(contact.get('past_illness', 0)) else "否",
                'past_illness_type': self.ILLNESS_TYPE_MAPPING_REVERSE.get(contact.get('past_illness_type', ''), contact.get('past_illness_type', 'none')) if _is_yes(contact.get('past_illness', 0)) else "none",
                'ethnicity': str(contact.get('ethnicity', '汉族')),
                'origin_altitude': int(contact.get('origin_altitude', 300)),
                'workplace_type': str(contact.get('workplace_type', '非油田')),
                'idu_status': '是' if _is_yes(contact.get('idu_status', False)) else '否',
                'district': str(contact.get('district', '克拉玛依区')),
                'months_since_migration': int(contact.get('months_since_migration', 0))
            }
            # 计算并添加累积暴露时长
            cumulative_exposure = calculate_cumulative_exposure(single_duration, freq_density, time_span)
            entry['cumulative_exposure'] = cumulative_exposure
            # 补充唯一ID和标签
            if '_id' not in contact:
                self._contact_id_counter += 1
                contact_id = f'social_{self._contact_id_counter}'
                entry['_id'] = contact_id
                if 'diagnosed' in contact:
                    self.contact_labels[contact_id] = 1 if _is_yes(contact['diagnosed']) else 0
                else:
                    self.contact_labels[contact_id] = 0
            else:
                entry['_id'] = contact['_id']
                if contact['_id'] not in self.contact_labels:
                    self.contact_labels[contact['_id']] = 0
            self.social_entries.append(entry)
        # 优先级六：通过 adapter 统一刷新（消除 ui_mode 分支）
        if self.adapter is not None:
            self.adapter.refresh_social_tree()
        # 刷新概览面板
        if hasattr(self, '_update_overview_panel'):
            self._update_overview_panel()