#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 导入面板 - API 导入 Mixin（_load_from_api / import_from_api）"""

from ._shared import *


class ApiImportMixin:
    """API 导入方法"""

    def _load_from_api(self):
        """从 REST API 加载数据（Section VIII: 走统一导入管线）"""
        from .pipeline import APIImportPipeline
        pipeline = APIImportPipeline(self)
        pipeline.run()

    def import_from_api(self, api_url, api_key=None):
        """从 REST API 导入数据

        Args:
            api_url: API URL
            api_key: 可选的 API Key

        Returns:
            bool: 导入是否成功
        """
        try:
            import requests
        except ImportError:
            self._add_import_error("requests 库未安装，请先运行: pip install requests")
            return False

        try:
            # 设置请求头
            headers = {}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
            headers['Accept'] = 'application/json'

            # 发送请求
            response = requests.get(api_url, headers=headers, timeout=30)
            response.raise_for_status()

            # 解析 JSON 数据
            data = response.json()

            # 处理患者基本信息（如果有）
            if 'patient' in data:
                patient_data = data['patient']
                clinical_fields = {'sputum_smear', 'has_cavity', 'active_tb', 'treatment'}
                for key, value in patient_data.items():
                    if key in self.basic_info_vars:
                        var = self.basic_info_vars[key]
                        if isinstance(var, tk.IntVar):
                            if key in clinical_fields and isinstance(value, str):
                                value_str = value.strip()
                                if value_str in CLINICAL_TEXT_MAP:
                                    value = CLINICAL_TEXT_MAP[value_str]
                            var.set(int(value))
                        elif isinstance(var, tk.StringVar):
                            var.set(str(value))

            # 处理家庭成员
            if 'family' in data:
                family_list = data['family']
                for contact in family_list:
                    standardized = self._standardize_contact_fields(contact, 'family')
                    self._add_contact_from_dict(standardized, 'family')

            # 处理社会接触者
            if 'social' in data:
                social_list = data['social']
                for contact in social_list:
                    standardized = self._standardize_contact_fields(contact, 'social')
                    self._add_contact_from_dict(standardized, 'social')

            # 也支持直接返回顶层结构
            if 'family_members' in data and 'family' not in data:
                for contact in data['family_members']:
                    standardized = self._standardize_contact_fields(contact, 'family')
                    self._add_contact_from_dict(standardized, 'family')

            if 'social_contacts' in data and 'social' not in data:
                for contact in data['social_contacts']:
                    standardized = self._standardize_contact_fields(contact, 'social')
                    self._add_contact_from_dict(standardized, 'social')

            return True

        except requests.exceptions.RequestException as e:
            self._add_import_error(f"API 请求失败: {str(e)}")
            return False
        except Exception as e:
            self._add_import_error(f"API 导入异常: {str(e)}")
            return False
