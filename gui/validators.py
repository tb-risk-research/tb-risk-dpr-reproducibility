#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 输入验证工具类（Section V）

三层验证体系：
  1. 控件级 validatecommand：输入时即时检查范围和类型
  2. 字段级 <FocusOut>：离开字段时执行更复杂的交叉验证
  3. 表单级预检：评估前执行完整性检查

设计要点：
  - 纯验证函数（validate_int / validate_range 等）不依赖 tkinter，
    可在无显示环境（CI/CD、headless 测试）下直接调用并断言结果。
  - tkinter 相关方法（make_validate_command / bind_focusout_validation）
    在 tkinter 不可用时优雅降级，返回安全默认值。
  - 验证常量统一从 constants.py 导入，避免硬编码上下限。
  - 错误反馈通过 ttk.Style 配置 'Error.TEntry' / 'Error.TSpinbox'，
    标红边框；同时维护错误标签供详细提示。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

# Section V: 验证常量从 constants.py 单一真相源导入
from ..constants import (
    MIN_AGE, MAX_AGE,
    MIN_VENTILATION, MAX_VENTILATION,
    MIN_SINGLE_DURATION_MINUTES, MAX_SINGLE_DURATION_MINUTES,
    MIN_TIME_SPAN_WEEKS, MAX_TIME_SPAN_WEEKS,
    MIN_FREQ_DENSITY, MAX_FREQ_DENSITY,
    MAX_FAMILY_MEMBERS, MAX_SOCIAL_CONTACTS,
    MIN_COUGH_FREQ, MAX_COUGH_FREQ,
    MIN_TREATMENT_DURATION_MONTHS, MAX_TREATMENT_DURATION_MONTHS,
    MIN_DELAY_DAYS, MAX_DELAY_DAYS,
    PERCENTAGE_MIN, PERCENTAGE_MAX,
)

# tkinter 可选依赖（验证函数本身不依赖 tkinter）
from ..utils import TKINTER_AVAILABLE  # tkinter 可用性标志（单一真相源）
try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = None
    ttk = None


# ==============================================================================
# 预定义字段规则（field_name -> (min, max, is_int, label)）
# 控件创建时可通过 FIELD_RULES[field_name] 查询上下限，消除硬编码
# ==============================================================================
FIELD_RULES: Dict[str, Tuple[Optional[int], Optional[int], bool, str]] = {
    'age':              (MIN_AGE, MAX_AGE, True, '年龄'),
    'single_duration':  (MIN_SINGLE_DURATION_MINUTES, MAX_SINGLE_DURATION_MINUTES, True, '单次接触时长(分钟)'),
    'freq_density':     (MIN_FREQ_DENSITY, MAX_FREQ_DENSITY, True, '每周接触频次'),
    'time_span':        (MIN_TIME_SPAN_WEEKS, MAX_TIME_SPAN_WEEKS, True, '持续周期(周)'),
    'ventilation':      (MIN_VENTILATION, MAX_VENTILATION, True, '通风条件'),
    'cough_freq':       (MIN_COUGH_FREQ, MAX_COUGH_FREQ, True, '咳嗽频率'),
    'treatment_duration': (MIN_TREATMENT_DURATION_MONTHS, MAX_TREATMENT_DURATION_MONTHS, True, '治疗时长(月)'),
    'delay_days':       (MIN_DELAY_DAYS, MAX_DELAY_DAYS, True, '延迟就诊天数'),
    'flp_percentage':   (PERCENTAGE_MIN, PERCENTAGE_MAX, True, '家庭潜伏感染比例(%)'),
    'hrsp_percentage':  (PERCENTAGE_MIN, PERCENTAGE_MAX, True, '高危人群比例(%)'),
}

# 接触者数量上限规则（非数值字段，单独处理）
COUNT_RULES: Dict[str, Tuple[int, str]] = {
    'family_count':  (MAX_FAMILY_MEMBERS, '家庭成员数'),
    'social_count':  (MAX_SOCIAL_CONTACTS, '社会接触者数'),
}


class FieldValidator:
    """字段验证工具类

    用法示例（控件级）::

        validator = FieldValidator()
        spin = ttk.Spinbox(parent, from_=MIN_AGE, to=MAX_AGE, textvariable=var)
        validator.attach(spin, 'age')  # 自动绑定 validatecommand + FocusOut

    用法示例（表单级预检）::

        ok, errors = validator.validate_form([
            {'field': 'age', 'value': var.get(), 'rules': ['int', 'range']},
            {'field': 'flp_percentage', 'value': var.get(), 'rules': ['int', 'range']},
        ])
        if not ok:
            messagebox.showerror('输入错误', '\\n'.join(errors.values()))
    """

    # ttk 样式名（_setup_error_styles 注册）
    ERROR_STYLE_ENTRY = 'Error.TEntry'
    ERROR_STYLE_SPINBOX = 'Error.TSpinbox'
    NORMAL_STYLE_ENTRY = 'TEntry'
    NORMAL_STYLE_SPINBOX = 'TSpinbox'

    def __init__(self):
        self._errors: Dict[str, str] = {}      # field_name -> error_message
        self._widgets: Dict[str, Any] = {}     # field_name -> widget
        self._error_labels: Dict[str, Any] = {} # field_name -> error Label
        self._error_styles_registered = False

    # ==================================================================
    # 纯验证函数（无 tkinter 依赖，便于单元测试）
    # ==================================================================

    @staticmethod
    def validate_int(value: Any, field_name: str = '字段',
                     min_val: Optional[int] = None,
                     max_val: Optional[int] = None) -> Tuple[bool, str]:
        """整数类型验证，可附加范围检查

        Returns:
            (is_valid, error_message) — 合法时 error_message 为空串
        """
        if value is None:
            return False, f'{field_name}不能为空'
        # 支持 tkinter 变量
        if hasattr(value, 'get'):
            try:
                value = value.get()
            except Exception:
                return False, f'{field_name}读取失败'
        # 已是数值
        if isinstance(value, bool):
            # bool 是 int 的子类，但这里应视为非法输入
            return False, f'{field_name}必须是整数，不能是布尔值'
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not value.is_integer():
                return False, f'{field_name}必须是整数（当前：{value}）'
            int_val = int(value)
        else:
            val_str = str(value).strip()
            if not val_str:
                return False, f'{field_name}不能为空'
            try:
                int_val = int(val_str)
            except ValueError:
                # 尝试浮点→整数（如 "30.0"）
                try:
                    f = float(val_str)
                    if not f.is_integer():
                        return False, f'{field_name}必须是整数（当前：{val_str}）'
                    int_val = int(f)
                except ValueError:
                    return False, f'{field_name}必须是整数（当前：{val_str}）'
        # 范围检查
        if min_val is not None and int_val < min_val:
            return False, f'{field_name}不能小于{min_val}（当前：{int_val}）'
        if max_val is not None and int_val > max_val:
            return False, f'{field_name}不能大于{max_val}（当前：{int_val}）'
        return True, ''

    @staticmethod
    def validate_range(value: Any, min_val: Optional[float], max_val: Optional[float],
                       field_name: str = '字段') -> Tuple[bool, str]:
        """范围检查（不强制类型）"""
        if value is None:
            return False, f'{field_name}不能为空'
        if hasattr(value, 'get'):
            try:
                value = value.get()
            except Exception:
                return False, f'{field_name}读取失败'
        try:
            num = float(value)
        except (TypeError, ValueError):
            return False, f'{field_name}必须是数值（当前：{value}）'
        if min_val is not None and num < min_val:
            return False, f'{field_name}不能小于{min_val}（当前：{num}）'
        if max_val is not None and num > max_val:
            return False, f'{field_name}不能大于{max_val}（当前：{num}）'
        return True, ''

    @staticmethod
    def validate_non_empty(value: Any, field_name: str = '字段') -> Tuple[bool, str]:
        """非空检查"""
        if value is None:
            return False, f'{field_name}不能为空'
        if hasattr(value, 'get'):
            try:
                value = value.get()
            except Exception:
                return False, f'{field_name}读取失败'
        if isinstance(value, str):
            if not value.strip():
                return False, f'{field_name}不能为空'
        elif isinstance(value, (list, dict, tuple)):
            if len(value) == 0:
                return False, f'{field_name}不能为空'
        return True, ''

    @staticmethod
    def validate_percentage(value: Any, field_name: str = '百分比') -> Tuple[bool, str]:
        """百分比验证（0-100 整数）"""
        return FieldValidator.validate_int(
            value, field_name, min_val=PERCENTAGE_MIN, max_val=PERCENTAGE_MAX
        )

    @staticmethod
    def validate_count(count: int, max_count: int, field_name: str = '数量') -> Tuple[bool, str]:
        """数量上限检查"""
        if count is None:
            return False, f'{field_name}不能为空'
        try:
            c = int(count)
        except (TypeError, ValueError):
            return False, f'{field_name}必须是整数'
        if c < 0:
            return False, f'{field_name}不能为负数（当前：{c}）'
        if c > max_count:
            return False, f'{field_name}不能超过{max_count}（当前：{c}）'
        return True, ''

    # ==================================================================
    # 交叉验证（字段间依赖关系）
    # ==================================================================

    @staticmethod
    def cross_validate_smear_symptoms(entry: Dict[str, Any]) -> Tuple[bool, str]:
        """交叉验证：涂阳患者应有症状提示

        规则：若 has_tb=是 且 sputum_smear=涂阳，建议 has_symptoms=是。
        此为软警告（返回 is_valid=True 但带提示消息），不阻塞评估。
        """
        # 仅在提供相关字段时检查
        has_tb = str(entry.get('has_tb', '否'))
        smear = str(entry.get('sputum_smear', ''))
        symptoms = str(entry.get('has_symptoms', '否'))
        if _is_yes_str(has_tb) and smear in ('涂阳', '阳性', '2'):
            if not _is_yes_str(symptoms):
                return True, '提示：涂阳接触者通常有症状，请确认症状字段'
        return True, ''

    # ==================================================================
    # tkinter 控件级验证（validatecommand）
    # ==================================================================

    def _setup_error_styles(self):
        """注册错误样式（仅 tkinter 可用时）

        注意：base.py 初始化时和主题切换时会统一配置 Error.TEntry/Error.TSpinbox
        的 fieldbackground 为 COLORS['danger_bg']，此处仅提供 fallback 颜色
        （与 base.py 中 danger_bg='#f8d7da' 保持一致），防止 FieldValidator
        在主窗口初始化前被使用时样式缺失。
        """
        if not TKINTER_AVAILABLE or self._error_styles_registered:
            return
        try:
            style = ttk.Style()
            # 使用与主应用 COLORS['danger_bg'] 一致的颜色
            style.configure(self.ERROR_STYLE_ENTRY, fieldbackground='#f8d7da')
            style.configure(self.ERROR_STYLE_SPINBOX, fieldbackground='#f8d7da')
            self._error_styles_registered = True
        except Exception:
            # 样式注册失败不阻塞验证逻辑
            pass

    def make_validate_command(self, widget: Any, field_name: str,
                              min_val: Optional[int], max_val: Optional[int],
                              is_int: bool = True) -> Optional[str]:
        """创建 validatecommand 并注册到控件

        Args:
            widget: ttk.Entry 或 ttk.Spinbox
            field_name: 字段名（用于错误提示）
            min_val, max_val: 范围
            is_int: True=整数验证，False=数值范围验证

        Returns:
            注册后的 validate 函数名（传入 widget.config(validate=...)），
            tkinter 不可用时返回 None
        """
        if not TKINTER_AVAILABLE or widget is None:
            return None

        self._setup_error_styles()

        def _validate(reason: str, new_value: str) -> bool:
            """validatecommand 回调

            Args:
                reason: 'key'（按键）/ 'focus'（焦点变化）/ 'focusin' / 'focusout'
                new_value: 控件将获得的新值

            Returns:
                True 表示接受输入，False 表示拒绝
            """
            try:
                # 空值允许（用户正在删除输入），由 FocusOut 检查必填
                if new_value is None or str(new_value).strip() == '':
                    self._clear_field_error(field_name)
                    return True
                if is_int:
                    ok, msg = self.validate_int(new_value, field_name, min_val, max_val)
                else:
                    ok, msg = self.validate_range(new_value, min_val, max_val, field_name)
                if ok:
                    self._clear_field_error(field_name)
                    return True
                # 按键时仅在值明显非法时拒绝（如字母），范围越界仅提示不拒绝
                if reason == 'key':
                    try:
                        if is_int:
                            int(new_value)
                        else:
                            float(new_value)
                        # 数值合法但范围越界：允许输入，由 FocusOut 标红
                        self._set_field_error(field_name, msg)
                        return True
                    except ValueError:
                        # 非法字符：拒绝
                        return False
                # focusout 时范围越界：允许但标红
                self._set_field_error(field_name, msg)
                return True
            except Exception:
                # 验证回调异常时允许输入，避免控件被锁死
                return True

        try:
            # validatecommand 需要注册 Python 函数
            vcmd = (widget.register(_validate), '%V', '%P')
            return vcmd
        except Exception:
            return None

    def attach(self, widget: Any, field_name: str,
               min_val: Optional[int] = None, max_val: Optional[int] = None,
               is_int: bool = True, required: bool = False) -> None:
        """为控件附加验证：自动配置 validatecommand + FocusOut

        Args:
            widget: ttk.Entry / ttk.Spinbox
            field_name: 字段名（从 FIELD_RULES 查询默认范围）
            min_val, max_val: 显式范围（覆盖 FIELD_RULES）
            is_int: 是否整数验证
            required: 是否必填（FocusOut 时检查非空）
        """
        if not TKINTER_AVAILABLE or widget is None:
            return
        # 从 FIELD_RULES 补全范围
        if field_name in FIELD_RULES and (min_val is None or max_val is None):
            rule_min, rule_max, rule_is_int, _ = FIELD_RULES[field_name]
            if min_val is None:
                min_val = rule_min
            if max_val is None:
                max_val = rule_max
            if is_int:
                is_int = rule_is_int
        self._widgets[field_name] = widget
        vcmd = self.make_validate_command(widget, field_name, min_val, max_val, is_int)
        if vcmd is not None:
            try:
                widget.configure(validate='all', validatecommand=vcmd)
            except Exception as e:
                LOGGER.debug("绑定 validatecommand 到字段 %s 失败: %s", field_name, e)
        def _on_focusout(_event):
            try:
                value = widget.get()
            except Exception as e:
                LOGGER.debug("获取字段 %s 的值失败: %s", field_name, e)
                return
            if required:
                ok, msg = self.validate_non_empty(value, field_name)
                if not ok:
                    self._set_field_error(field_name, msg)
                    return
            if value is not None and str(value).strip():
                if is_int:
                    ok, msg = self.validate_int(value, field_name, min_val, max_val)
                else:
                    ok, msg = self.validate_range(value, min_val, max_val, field_name)
                if not ok:
                    self._set_field_error(field_name, msg)
                else:
                    self._clear_field_error(field_name)
        try:
            widget.bind('<FocusOut>', _on_focusout)
        except Exception as e:
            LOGGER.debug("绑定 FocusOut 事件到字段 %s 失败: %s", field_name, e)

    # ==================================================================
    # 视觉反馈：标红边框 + 错误标签
    # ==================================================================

    def _set_field_error(self, field_name: str, message: str) -> None:
        """标记字段错误：标红边框 + 记录错误消息"""
        self._errors[field_name] = message
        widget = self._widgets.get(field_name)
        if widget is not None and TKINTER_AVAILABLE:
            try:
                wclass = widget.winfo_class()
                if wclass == 'TSpinbox':
                    widget.configure(style=self.ERROR_STYLE_SPINBOX)
                elif wclass == 'TEntry':
                    widget.configure(style=self.ERROR_STYLE_ENTRY)
            except Exception as e:
                LOGGER.debug("设置字段 %s 错误样式失败: %s", field_name, e)
        label = self._error_labels.get(field_name)
        if label is not None and TKINTER_AVAILABLE:
            try:
                label.configure(text=message)
            except Exception as e:
                LOGGER.debug("更新字段 %s 错误标签失败: %s", field_name, e)

    def _clear_field_error(self, field_name: str) -> None:
        """清除字段错误高亮"""
        self._errors.pop(field_name, None)
        widget = self._widgets.get(field_name)
        if widget is not None and TKINTER_AVAILABLE:
            try:
                wclass = widget.winfo_class()
                if wclass == 'TSpinbox':
                    widget.configure(style=self.NORMAL_STYLE_SPINBOX)
                elif wclass == 'TEntry':
                    widget.configure(style=self.NORMAL_STYLE_ENTRY)
            except Exception as e:
                LOGGER.debug("清除字段 %s 错误样式失败: %s", field_name, e)
        label = self._error_labels.get(field_name)
        if label is not None and TKINTER_AVAILABLE:
            try:
                label.configure(text='')
            except Exception as e:
                LOGGER.debug("清除字段 %s 错误标签失败: %s", field_name, e)

    def attach_error_label(self, field_name: str, label: Any) -> None:
        """绑定错误提示 Label 控件，错误消息将显示在 Label 上"""
        self._error_labels[field_name] = label

    # ==================================================================
    # 表单级预检
    # ==================================================================

    def validate_form(self, checks: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, str]]:
        """执行表单级完整性预检

        Args:
            checks: 检查项列表，每项::

                {
                    'field': 'age',
                    'value': var.get() 或直接值,
                    'rules': ['non_empty', 'int', 'range'],  # 任一组合
                    'min': 0, 'max': 120,  # range 规则所需
                    'is_int': True,  # 默认 True
                    'label': '年龄',  # 显示名（覆盖 FIELD_RULES）
                }

        Returns:
            (is_valid, errors_dict) — errors_dict: {field_name: message}
        """
        errors: Dict[str, str] = {}
        for check in checks:
            field = check.get('field', '')
            value = check.get('value')
            rules = check.get('rules', [])
            label = check.get('label') or (FIELD_RULES.get(field, (None, None, True, field))[3] if field in FIELD_RULES else field)
            min_val = check.get('min')
            max_val = check.get('max')
            # 从 FIELD_RULES 补全 min/max
            if field in FIELD_RULES:
                rule_min, rule_max, _, _ = FIELD_RULES[field]
                if min_val is None:
                    min_val = rule_min
                if max_val is None:
                    max_val = rule_max
            for rule in rules:
                if rule == 'non_empty':
                    ok, msg = self.validate_non_empty(value, label)
                elif rule == 'int':
                    ok, msg = self.validate_int(value, label, min_val, max_val)
                elif rule == 'range':
                    ok, msg = self.validate_range(value, min_val, max_val, label)
                elif rule == 'percentage':
                    ok, msg = self.validate_percentage(value, label)
                else:
                    continue
                if not ok:
                    errors[field] = msg
                    break  # 该字段首个错误即可
        # 同步到内部 _errors 状态
        self._errors.update(errors)
        return (len(errors) == 0), errors

    def validate_contact_counts(self, family_count: int, social_count: int) -> Tuple[bool, Dict[str, str]]:
        """接触者数量上限检查（表单级专用）"""
        errors: Dict[str, str] = {}
        ok, msg = self.validate_count(family_count, MAX_FAMILY_MEMBERS, '家庭成员数')
        if not ok:
            errors['family_count'] = msg
        ok, msg = self.validate_count(social_count, MAX_SOCIAL_CONTACTS, '社会接触者数')
        if not ok:
            errors['social_count'] = msg
        self._errors.update(errors)
        return (len(errors) == 0), errors

    # ==================================================================
    # 查询接口
    # ==================================================================

    def get_errors(self) -> Dict[str, str]:
        """返回所有字段错误"""
        return dict(self._errors)

    def has_errors(self) -> bool:
        """是否含有错误"""
        return len(self._errors) > 0

    def clear_all_errors(self) -> None:
        """清除所有错误状态"""
        for field in list(self._errors.keys()):
            self._clear_field_error(field)
        self._errors.clear()

    def get_first_error(self) -> Optional[str]:
        """返回首个错误消息（用于 messagebox 简短提示）"""
        if not self._errors:
            return None
        return next(iter(self._errors.values()))


# ==============================================================================
# 辅助函数
# ==============================================================================

def _is_yes_str(value: Any) -> bool:
    """判断中文字符串是否表示"是" """
    if value is None:
        return False
    s = str(value).strip()
    return s in ('是', '有', '已接种', '曾患', '阳性', '已治疗', '曾治疗', '正治疗')


__all__ = [
    'FieldValidator',
    'FIELD_RULES',
    'COUNT_RULES',
]
