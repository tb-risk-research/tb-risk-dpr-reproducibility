#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""映射模板管理

管理字段映射配置的模板，支持：
1. 模板保存：将当前映射配置保存为模板
2. 模板加载：加载已有模板并应用到引擎
3. 模板列表：按医院、厂商、版本查询
4. 模板复用：同一厂商HIS系统可复用映射模板
5. 模板版本管理：支持迭代升级
6. 模板导入/导出：JSON格式交换
"""

import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional

from .field_mapping_engine import FieldMappingEngine

LOGGER = logging.getLogger("tb_risk.health_interop.mapping.template")

# 默认模板存储目录
DEFAULT_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

# ============================================================================
# 内置模板示例
# ============================================================================

BUILTIN_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "default_his": {
        "name": "通用HIS系统映射模板",
        "version": "1.0",
        "vendor": "通用",
        "system_type": "HIS",
        "description": "适用于大多数HIS系统的基础字段映射模板",
        "mappings": [
            {"source_field": "患者姓名", "target_field": "patient_name", "confidence": 0.9},
            {"source_field": "年龄", "target_field": "age", "confidence": 0.9},
            {"source_field": "性别", "target_field": "gender", "confidence": 0.9,
             "transforms": [{"type": "code_map", "mapping": {"男": "1", "女": "0", "未知": "2"}}]},
            {"source_field": "民族", "target_field": "ethnicity", "confidence": 0.8},
            {"source_field": "职业", "target_field": "occupation", "confidence": 0.7},
            {"source_field": "联系电话", "target_field": "phone", "confidence": 0.8},
            {"source_field": "现住址", "target_field": "address", "confidence": 0.8},
            {"source_field": "吸烟史", "target_field": "smoking_years", "confidence": 0.7,
             "transforms": [{"type": "text_extract", "pattern": r"(\d+)\s*年"}]},
            {"source_field": "身高", "target_field": "height_cm", "confidence": 0.8},
            {"source_field": "体重", "target_field": "weight_kg", "confidence": 0.8},
        ],
    },
    "default_lis": {
        "name": "通用LIS检验系统映射模板",
        "version": "1.0",
        "vendor": "通用",
        "system_type": "LIS",
        "description": "适用于大多数LIS系统的检验结果字段映射模板",
        "mappings": [
            {"source_field": "检验项目编码", "target_field": "lab_code", "confidence": 0.9},
            {"source_field": "检验项目名称", "target_field": "lab_name", "confidence": 0.9},
            {"source_field": "检验结果", "target_field": "lab_value", "confidence": 0.9},
            {"source_field": "结果单位", "target_field": "lab_unit", "confidence": 0.8},
            {"source_field": "参考范围下限", "target_field": "ref_low", "confidence": 0.8},
            {"source_field": "参考范围上限", "target_field": "ref_high", "confidence": 0.8},
            {"source_field": "异常标志", "target_field": "abnormal_flag", "confidence": 0.8},
            {"source_field": "检验日期", "target_field": "lab_date", "confidence": 0.9},
        ],
    },
    "default_pacs": {
        "name": "通用PACS影像系统映射模板",
        "version": "1.0",
        "vendor": "通用",
        "system_type": "PACS",
        "description": "适用于大多数PACS系统的影像报告字段映射模板",
        "mappings": [
            {"source_field": "检查部位", "target_field": "exam_body_part", "confidence": 0.9},
            {"source_field": "检查方法", "target_field": "exam_method", "confidence": 0.8},
            {"source_field": "影像所见", "target_field": "imaging_findings", "confidence": 0.8},
            {"source_field": "诊断意见", "target_field": "impression", "confidence": 0.8},
            {"source_field": "报告日期", "target_field": "report_date", "confidence": 0.9},
        ],
    },
    "hisi_default": {
        "name": "HISI系统映射模板（克拉玛依）",
        "version": "1.0",
        "vendor": "HISI",
        "system_type": "HIS",
        "description": "克拉玛依区域HISI系统的字段映射模板",
        "mappings": [
            {"source_field": "XM", "target_field": "patient_name", "description": "姓名"},
            {"source_field": "NL", "target_field": "age", "description": "年龄"},
            {"source_field": "XB", "target_field": "gender", "confidence": 0.9,
             "transforms": [{"type": "code_map", "mapping": {"1": "1", "2": "0", "男": "1", "女": "0"}}]},
            {"source_field": "MZ", "target_field": "ethnicity", "description": "民族"},
            {"source_field": "ZY", "target_field": "occupation", "description": "职业"},
            {"source_field": "SFZH", "target_field": "id_card", "description": "身份证号"},
            {"source_field": "DH", "target_field": "phone", "description": "联系电话"},
            {"source_field": "DZ", "target_field": "address", "description": "地址"},
            {"source_field": "ZDM", "target_field": "diagnosis_code", "description": "诊断编码"},
            {"source_field": "ZDM", "target_field": "active_tb", "confidence": 0.6,
             "transforms": [{"type": "code_map",
                             "mapping": {"A15": "1", "A16": "1", "A17": "1", "A18": "1", "A19": "1"},
                             "fuzzy": True}],
             "description": "从诊断编码推断活动性结核"},
        ],
    },
}


class MappingTemplateManager:
    """映射模板管理器

    管理字段映射模板的增删改查、导入导出、版本控制。
    """

    def __init__(self, template_dir: str = ""):
        self._template_dir = template_dir or DEFAULT_TEMPLATE_DIR
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._load_builtin_templates()

    def _load_builtin_templates(self):
        """加载内置模板"""
        for template_id, template in BUILTIN_TEMPLATES.items():
            self._templates[template_id] = {
                **template,
                "id": template_id,
                "builtin": True,
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
            }

    # ==================== 模板管理 ====================

    def list_templates(self, vendor: str = "", system_type: str = "",
                       keyword: str = "") -> List[Dict[str, Any]]:
        """列出可用模板

        参数：
            vendor: 按厂商筛选
            system_type: 按系统类型筛选（HIS/LIS/PACS/EMR等）
            keyword: 按关键字搜索

        返回：
            list[dict]: 模板列表
        """
        results = []
        for template_id, template in self._templates.items():
            # 筛选
            if vendor and template.get("vendor", "") != vendor:
                continue
            if system_type and template.get("system_type", "") != system_type:
                continue
            if keyword:
                kw = keyword.lower()
                name = template.get("name", "").lower()
                desc = template.get("description", "").lower()
                if kw not in name and kw not in desc:
                    continue

            results.append({
                "id": template_id,
                "name": template.get("name", ""),
                "version": template.get("version", "1.0"),
                "vendor": template.get("vendor", ""),
                "system_type": template.get("system_type", ""),
                "description": template.get("description", ""),
                "builtin": template.get("builtin", False),
                "mapping_count": len(template.get("mappings", [])),
                "created_at": template.get("created_at", ""),
                "updated_at": template.get("updated_at", ""),
            })

        return results

    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """获取模板详情"""
        return self._templates.get(template_id)

    def save_template(self, template_id: str, name: str,
                      mappings: List[Dict[str, Any]],
                      vendor: str = "", system_type: str = "",
                      version: str = "1.0",
                      description: str = "",
                      overwrite: bool = False) -> str:
        """保存模板

        参数：
            template_id: 模板标识符
            name: 模板名称
            mappings: 映射配置列表
            vendor: 厂商名称
            system_type: 系统类型
            version: 版本号
            description: 描述
            overwrite: 是否覆盖已有模板

        返回：
            str: 模板ID
        """
        if template_id in self._templates and not overwrite:
            raise ValueError(f"模板 '{template_id}' 已存在，设置 overwrite=True 覆盖。")

        now = datetime.datetime.now().isoformat()
        template = {
            "id": template_id,
            "name": name,
            "version": version,
            "vendor": vendor,
            "system_type": system_type,
            "description": description,
            "mappings": mappings,
            "builtin": False,
            "created_at": self._templates.get(template_id, {}).get("created_at", now),
            "updated_at": now,
        }
        self._templates[template_id] = template

        # 持久化保存
        self._persist_template(template_id)

        LOGGER.info("保存模板: %s (%s, %d 个映射)", template_id, name, len(mappings))
        return template_id

    def save_from_engine(self, template_id: str, name: str,
                         engine: FieldMappingEngine,
                         vendor: str = "", system_type: str = "",
                         version: str = "1.0",
                         description: str = "",
                         overwrite: bool = False) -> str:
        """从引擎保存模板"""
        mappings = [m.to_dict() for m in engine.get_all_mappings()]
        return self.save_template(template_id, name, mappings,
                                  vendor, system_type, version,
                                  description, overwrite)

    def delete_template(self, template_id: str) -> bool:
        """删除模板"""
        if template_id in self._templates:
            template = self._templates[template_id]
            if template.get("builtin", False):
                LOGGER.warning("内置模板不可删除: %s", template_id)
                return False
            del self._templates[template_id]
            self._remove_persisted_template(template_id)
            return True
        return False

    # ==================== 模板应用 ====================

    def apply_template(self, template_id: str,
                       engine: FieldMappingEngine) -> int:
        """将模板应用到映射引擎

        参数：
            template_id: 模板ID
            engine: FieldMappingEngine 实例

        返回：
            int: 应用的映射数量
        """
        template = self._templates.get(template_id)
        if not template:
            LOGGER.warning("模板不存在: %s", template_id)
            return 0

        mappings = template.get("mappings", [])
        for m in mappings:
            engine.add_mapping(
                source_field=m["source_field"],
                target_field=m["target_field"],
                transforms=m.get("transforms"),
                condition=m.get("condition"),
                default_value=m.get("default_value"),
                description=m.get("description", ""),
                confidence=m.get("confidence", 1.0),
            )

        LOGGER.info("应用模板 '%s' 到引擎，添加 %d 个映射", template_id, len(mappings))
        return len(mappings)

    def apply_template_to_engine(self, template_id: str,
                                 engine: FieldMappingEngine) -> int:
        """将模板应用到引擎（别名，兼容旧接口）"""
        return self.apply_template(template_id, engine)

    # ==================== 模板导入/导出 ====================

    def export_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """导出模板（JSON序列化）"""
        template = self._templates.get(template_id)
        if not template:
            return None
        return {
            "export_version": "1.0",
            "exported_at": datetime.datetime.now().isoformat(),
            "template": {
                "id": template_id,
                "name": template.get("name", ""),
                "version": template.get("version", "1.0"),
                "vendor": template.get("vendor", ""),
                "system_type": template.get("system_type", ""),
                "description": template.get("description", ""),
                "mappings": template.get("mappings", []),
            },
        }

    def export_template_to_file(self, template_id: str,
                                file_path: str) -> bool:
        """导出模板到JSON文件"""
        data = self.export_template(template_id)
        if not data:
            return False
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            LOGGER.info("导出模板到文件: %s", file_path)
            return True
        except OSError as e:
            LOGGER.error("导出模板失败: %s", e)
            return False

    def import_template(self, data: Dict[str, Any],
                        overwrite: bool = False) -> Optional[str]:
        """导入模板（从JSON数据）

        参数：
            data: 模板数据（export_template 的格式）
            overwrite: 是否覆盖已有模板

        返回：
            str | None: 模板ID
        """
        template_data = data.get("template", data)
        template_id = template_data.get("id", "")
        if not template_id:
            # 自动生成ID
            name = template_data.get("name", "imported_template")
            template_id = name.lower().replace(" ", "_").replace("-", "_")

        return self.save_template(
            template_id=template_id,
            name=template_data.get("name", template_id),
            mappings=template_data.get("mappings", []),
            vendor=template_data.get("vendor", ""),
            system_type=template_data.get("system_type", ""),
            version=template_data.get("version", "1.0"),
            description=template_data.get("description", ""),
            overwrite=overwrite,
        )

    def import_template_from_file(self, file_path: str,
                                  overwrite: bool = False) -> Optional[str]:
        """从JSON文件导入模板"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self.import_template(data, overwrite)
        except (OSError, json.JSONDecodeError) as e:
            LOGGER.error("从文件导入模板失败: %s", e)
            return None

    # ==================== 模板持久化 ====================

    def _persist_template(self, template_id: str):
        """将模板持久化到文件"""
        try:
            os.makedirs(self._template_dir, exist_ok=True)
            file_path = os.path.join(self._template_dir, f"{template_id}.json")
            data = self.export_template(template_id)
            if data:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            LOGGER.warning("持久化模板失败: %s", e)

    def _remove_persisted_template(self, template_id: str):
        """删除持久化的模板文件"""
        file_path = os.path.join(self._template_dir, f"{template_id}.json")
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError as e:
            LOGGER.warning("删除模板文件失败: %s", e)

    def load_persisted_templates(self) -> int:
        """从文件系统加载持久化的模板"""
        count = 0
        try:
            os.makedirs(self._template_dir, exist_ok=True)
            for filename in os.listdir(self._template_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(self._template_dir, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        template_data = data.get("template", data)
                        template_id = template_data.get("id",
                                                        filename[:-5])
                        if template_id not in self._templates:
                            self._templates[template_id] = {
                                **template_data,
                                "id": template_id,
                                "builtin": False,
                                "created_at": template_data.get("created_at",
                                                                datetime.datetime.now().isoformat()),
                                "updated_at": template_data.get("updated_at",
                                                                datetime.datetime.now().isoformat()),
                            }
                            count += 1
                    except (json.JSONDecodeError, OSError) as e:
                        LOGGER.warning("加载模板文件失败 %s: %s", filename, e)
        except OSError as e:
            LOGGER.warning("读取模板目录失败: %s", e)
        return count

    def get_statistics(self) -> Dict[str, Any]:
        """获取模板管理统计"""
        builtin = sum(1 for t in self._templates.values() if t.get("builtin"))
        custom = sum(1 for t in self._templates.values() if not t.get("builtin"))
        total_mappings = sum(len(t.get("mappings", [])) for t in self._templates.values())
        vendors = set(t.get("vendor", "") for t in self._templates.values() if t.get("vendor"))
        return {
            "total_templates": len(self._templates),
            "builtin": builtin,
            "custom": custom,
            "total_mappings": total_mappings,
            "vendors": sorted(v for v in vendors if v),
            "template_dir": self._template_dir,
        }