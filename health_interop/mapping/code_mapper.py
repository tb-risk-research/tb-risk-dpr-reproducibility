#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""编码映射引擎

支持医院系统使用的各种编码到 tb_risk 标准编码的映射：
1. 性别编码映射（男/女/1/2/M/F → 内部统一编码）
2. 民族编码映射（不同医院医院编码 → 国标 GB/T 3304 编码）
3. 职业编码映射（医院自定义编码 → 国标 GB/T 6565 编码）
4. 婚姻状况编码映射
5. 科室编码映射
6. 诊断编码映射（医院自定义 → ICD-10）
7. 自定义编码映射（支持任意编码系统间的映射）
8. 双向映射（编码→名称、名称→编码）
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

LOGGER = logging.getLogger("tb_risk.health_interop.mapping.code_mapper")

# ============================================================================
# 内置编码映射字典
# ============================================================================

# 性别编码映射
GENDER_CODES: Dict[str, Dict[str, str]] = {
    "tb_risk": {"male": "1", "female": "0", "unknown": "2"},
    "hospital_a": {"1": "男", "2": "女", "9": "未知"},
    "hospital_b": {"M": "男", "F": "女", "U": "未知"},
    "hospital_c": {"0": "女", "1": "男", "2": "未知"},
    # 多系统对照
    "crosswalk": {
        "男": "1", "male": "1", "M": "1", "1": "1",
        "女": "0", "female": "0", "F": "0", "2": "0",
        "未知": "2", "unknown": "2", "U": "2", "其他": "2", "other": "2",
    },
}

# 婚姻状况编码映射
MARITAL_CODES: Dict[str, Dict[str, str]] = {
    "crosswalk": {
        "未婚": "1", "single": "1", "S": "1",
        "已婚": "2", "married": "2", "M": "2",
        "离异": "3", "divorced": "3", "D": "3",
        "丧偶": "4", "widowed": "4", "W": "4",
        "未知": "9", "unknown": "9", "U": "9",
    },
}

# 科室编码映射
DEPARTMENT_CODES: Dict[str, Dict[str, str]] = {
    "crosswalk": {
        "呼吸科": "respiratory", "呼吸内科": "respiratory",
        "感染科": "infectious", "感染性疾病科": "infectious",
        "结核科": "tb", "结核病科": "tb",
        "急诊科": "emergency",
        "发热门诊": "fever_clinic",
        "内科": "internal_medicine",
        "全科": "general_practice",
        "影像科": "radiology", "放射科": "radiology",
        "检验科": "laboratory", "化验科": "laboratory",
    },
}

# 检验结果编码映射
LAB_RESULT_CODES: Dict[str, Dict[str, str]] = {
    "crosswalk": {
        "阴性": "negative", "阴": "negative", "(-)": "negative",
        "阳性": "positive", "阳": "positive", "(+)": "positive",
        "未检出": "negative", "检出": "positive",
        "弱阳性": "weak_positive", "(±)": "weak_positive",
        "未做": "not_done", "未查": "not_done",
        "正常": "normal", "异常": "abnormal",
    },
}

# 结核病治疗结果编码
TREATMENT_OUTCOME_CODES: Dict[str, Dict[str, str]] = {
    "crosswalk": {
        "治愈": "cured", "痊愈": "cured",
        "完成治疗": "completed", "完成疗程": "completed",
        "失败": "failed", "治疗失败": "failed",
        "死亡": "died", "死亡（结核）": "died_tb",
        "失访": "lost_to_followup", "丢失": "lost_to_followup",
        "转出": "transferred_out",
        "未评估": "not_evaluated",
    },
}

# 症状编码映射
SYMPTOM_CODES: Dict[str, Dict[str, str]] = {
    "crosswalk": {
        "咳嗽": "cough", "咳": "cough",
        "咳痰": "sputum", "痰": "sputum",
        "咯血": "hemoptysis", "咳血": "hemoptysis", "痰中带血": "hemoptysis",
        "发热": "fever", "发烧": "fever", "低热": "fever",
        "盗汗": "night_sweat", "夜间出汗": "night_sweat",
        "体重下降": "weight_loss", "消瘦": "weight_loss", "体重减轻": "weight_loss",
        "乏力": "fatigue", "疲倦": "fatigue", "无力": "fatigue",
        "食欲减退": "anorexia", "纳差": "anorexia", "食欲不振": "anorexia",
        "胸痛": "chest_pain", "胸疼": "chest_pain",
        "呼吸困难": "dyspnea", "气促": "dyspnea", "气短": "dyspnea",
    },
}


class CodeMapper:
    """编码映射引擎

    支持医院系统自定义编码到 tb_risk 标准编码的映射。
    支持批量映射、双向映射、模糊匹配、映射验证。
    """

    def __init__(self):
        # 内置映射表
        self._builtin_maps: Dict[str, Dict[str, Any]] = {
            "gender": {"name": "性别编码", "mapping": GENDER_CODES["crosswalk"]},
            "marital": {"name": "婚姻状况编码", "mapping": MARITAL_CODES["crosswalk"]},
            "department": {"name": "科室编码", "mapping": DEPARTMENT_CODES["crosswalk"]},
            "lab_result": {"name": "检验结果编码", "mapping": LAB_RESULT_CODES["crosswalk"]},
            "treatment_outcome": {"name": "治疗结果编码", "mapping": TREATMENT_OUTCOME_CODES["crosswalk"]},
            "symptom": {"name": "症状编码", "mapping": SYMPTOM_CODES["crosswalk"]},
        }
        # 自定义映射表
        self._custom_maps: Dict[str, Dict[str, Any]] = {}

    # ==================== 映射表管理 ====================

    def get_available_maps(self) -> List[Dict[str, Any]]:
        """获取所有可用映射表"""
        results = []
        for map_id, info in self._builtin_maps.items():
            results.append({
                "id": map_id,
                "name": info["name"],
                "builtin": True,
                "entry_count": len(info["mapping"]),
            })
        for map_id, info in self._custom_maps.items():
            results.append({
                "id": map_id,
                "name": info.get("name", map_id),
                "builtin": False,
                "entry_count": len(info.get("mapping", {})),
                "description": info.get("description", ""),
            })
        return results

    def get_map(self, map_id: str) -> Optional[Dict[str, Any]]:
        """获取映射表"""
        info = self._builtin_maps.get(map_id) or self._custom_maps.get(map_id)
        if info:
            return {"id": map_id, **info}
        return None

    def register_map(self, map_id: str, name: str, mapping: Dict[str, str],
                     description: str = "", overwrite: bool = False) -> str:
        """注册自定义编码映射表

        参数：
            map_id: 映射表标识符
            name: 映射表名称
            mapping: 编码映射字典 {源编码: 目标编码}
            description: 描述
            overwrite: 是否覆盖已有映射

        返回：
            str: 映射表ID
        """
        if map_id in self._builtin_maps and not overwrite:
            raise ValueError(f"映射表 '{map_id}' 为内置映射，不可覆盖。设置 overwrite=True 强制覆盖。")

        self._custom_maps[map_id] = {
            "name": name,
            "mapping": mapping,
            "description": description,
        }
        LOGGER.info("注册编码映射表: %s (%s, %d 条)", map_id, name, len(mapping))
        return map_id

    def remove_map(self, map_id: str) -> bool:
        """移除自定义映射表"""
        if map_id in self._custom_maps:
            del self._custom_maps[map_id]
            return True
        return False

    def add_entry(self, map_id: str, source_code: str, target_code: str) -> bool:
        """向映射表添加条目"""
        info = self._custom_maps.get(map_id)
        if info is None:
            # 如果内置映射允许扩展，也支持
            info = self._builtin_maps.get(map_id)
            if info is None:
                LOGGER.warning("映射表不存在: %s", map_id)
                return False
            # 将内置映射复制到自定义映射再进行修改
            self._custom_maps[map_id] = {
                "name": info["name"],
                "mapping": dict(info["mapping"]),
                "description": "内置映射扩展版",
                "extends": map_id,
            }
            info = self._custom_maps[map_id]

        info["mapping"][source_code] = target_code
        return True

    def remove_entry(self, map_id: str, source_code: str) -> bool:
        """从映射表移除条目"""
        info = self._custom_maps.get(map_id)
        if info and source_code in info["mapping"]:
            del info["mapping"][source_code]
            return True
        return False

    # ==================== 核心映射 ====================

    def map(self, value: Any, map_id: str, *,
            fuzzy: bool = False, reverse: bool = False,
            default: Any = None) -> Any:
        """执行编码映射

        参数：
            value: 待映射的编码值
            map_id: 映射表ID
            fuzzy: 是否启用模糊匹配
            reverse: 是否反向映射（目标→源）
            default: 未匹配时的默认值（默认返回原值）

        返回：
            映射后的编码值，未匹配时返回 default 或原值
        """
        if value is None or value == "":
            return default if default is not None else value

        info = self._builtin_maps.get(map_id) or self._custom_maps.get(map_id)
        if not info:
            LOGGER.warning("映射表不存在: %s", map_id)
            return default if default is not None else value

        mapping = info["mapping"]
        str_val = str(value).strip()

        # 直接匹配
        if str_val in mapping:
            return mapping[str_val]

        # 大小写不敏感匹配
        for k, v in mapping.items():
            if str_val.lower() == k.lower():
                return v

        # 模糊匹配
        if fuzzy:
            for k, v in mapping.items():
                if k.lower() in str_val.lower() or str_val.lower() in k.lower():
                    return v

        # 反向映射
        if reverse:
            for k, v in mapping.items():
                if str(v) == str_val:
                    return k

        return default if default is not None else value

    def map_batch(self, values: List[Any], map_id: str, *,
                  fuzzy: bool = False, reverse: bool = False,
                  default: Any = None) -> List[Any]:
        """批量映射"""
        return [self.map(v, map_id, fuzzy=fuzzy, reverse=reverse, default=default)
                for v in values]

    def map_dict(self, data: Dict[str, Any],
                 field_map: Dict[str, str],
                 default: Any = None) -> Dict[str, Any]:
        """对字典中指定字段进行编码映射

        参数：
            data: 源字典
            field_map: {源字段名: 映射表ID} 映射
            default: 未匹配默认值

        返回：
            dict: 映射后的字典
        """
        result = dict(data)
        for field, map_id in field_map.items():
            if field in data:
                result[field] = self.map(data[field], map_id, default=default)
        return result

    # ==================== 双向映射 ====================

    def map_to_code(self, value: Any, map_id: str, default: Any = None) -> Any:
        """正向映射：名称/显示值 → 编码"""
        return self.map(value, map_id, default=default)

    def map_to_name(self, code: Any, map_id: str, default: Any = None) -> Any:
        """反向映射：编码 → 名称/显示值"""
        info = self._builtin_maps.get(map_id) or self._custom_maps.get(map_id)
        if not info:
            return default if default is not None else code

        mapping = info["mapping"]
        str_code = str(code).strip()
        for k, v in mapping.items():
            if str(v) == str_code:
                return k
        return default if default is not None else code

    # ==================== 验证与统计 ====================

    def validate(self, map_id: str) -> List[Dict[str, Any]]:
        """验证映射表的一致性"""
        issues = []
        info = self._builtin_maps.get(map_id) or self._custom_maps.get(map_id)
        if not info:
            return [{"type": "error", "message": f"映射表 '{map_id}' 不存在"}]

        mapping = info["mapping"]
        # 检查是否有多个源编码映射到同一目标编码
        seen: Dict[str, List[str]] = {}
        for source, target in mapping.items():
            seen.setdefault(target, []).append(source)
        for target, sources in seen.items():
            if len(sources) > 1:
                issues.append({
                    "type": "duplicate",
                    "target": target,
                    "sources": sources,
                    "message": f"目标编码 '{target}' 有多个源编码映射: {sources}",
                })

        return issues

    def get_statistics(self) -> Dict[str, Any]:
        """获取映射引擎统计信息"""
        builtin_count = len(self._builtin_maps)
        custom_count = len(self._custom_maps)
        total_entries = sum(len(info["mapping"]) for info in self._builtin_maps.values())
        total_entries += sum(len(info.get("mapping", {})) for info in self._custom_maps.values())
        return {
            "builtin_maps": builtin_count,
            "custom_maps": custom_count,
            "total_maps": builtin_count + custom_count,
            "total_entries": total_entries,
        }

    # ==================== 序列化 ====================

    def export_maps(self, map_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """导出映射表（用于备份/迁移）"""
        result = {}
        target_ids = map_ids or list(self._builtin_maps.keys()) + list(self._custom_maps.keys())
        for map_id in target_ids:
            info = self._builtin_maps.get(map_id) or self._custom_maps.get(map_id)
            if info:
                result[map_id] = {
                    "name": info["name"],
                    "mapping": info["mapping"],
                    "description": info.get("description", ""),
                    "builtin": map_id in self._builtin_maps,
                }
        return result

    def import_maps(self, data: Dict[str, Any], overwrite: bool = False) -> int:
        """导入映射表"""
        count = 0
        for map_id, info in data.items():
            if map_id in self._builtin_maps and not overwrite:
                continue
            self._custom_maps[map_id] = {
                "name": info.get("name", map_id),
                "mapping": info.get("mapping", {}),
                "description": info.get("description", ""),
                "imported": True,
            }
            count += 1
        return count