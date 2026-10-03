#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 结果面板 - 评估编排 Mixin（_run_assessment / _update_gui_after_assessment）"""

import threading
import traceback

from ._shared import *
from ..app_config import AutoSaveManager, CONFIG_DIR
from ...utils import _is_yes
# Section V: 验证常量从 constants.py 单一真相源导入（消除硬编码上下限）
from ...constants import (
    MIN_AGE as _MIN_AGE, MAX_AGE as _MAX_AGE,
    MAX_DELAY_DAYS as _MAX_DELAY_DAYS, MIN_DELAY_DAYS as _MIN_DELAY_DAYS,
    MAX_COUGH_FREQ as _MAX_COUGH_FREQ, MIN_COUGH_FREQ as _MIN_COUGH_FREQ,
    MAX_TREATMENT_DURATION_MONTHS as _MAX_TREATMENT_DURATION_MONTHS,
    MIN_TREATMENT_DURATION_MONTHS as _MIN_TREATMENT_DURATION_MONTHS,
    PERCENTAGE_MIN as _PERCENTAGE_MIN, PERCENTAGE_MAX as _PERCENTAGE_MAX,
    MAX_SINGLE_DURATION_MINUTES as _MAX_SINGLE_DURATION_MINUTES,
    MAX_FREQ_DENSITY as _MAX_FREQ_DENSITY,
    MAX_TIME_SPAN_WEEKS as _MAX_TIME_SPAN_WEEKS,
    MIN_VENTILATION as _MIN_VENTILATION, MAX_VENTILATION as _MAX_VENTILATION,
)


# 优先级三：错误日志文件路径
_ERROR_LOG_FILE = os.path.join(CONFIG_DIR, 'error.log')


class OrchestratorMixin:
    """评估流程编排方法"""

    @staticmethod
    def _log_error_to_file(context: str, exc: BaseException) -> str:
        """优先级三：将完整异常堆栈写入日志文件

        Args:
            context: 错误上下文描述（如 "评估过程中" / "结果显示过程中"）
            exc: 捕获的异常对象

        Returns:
            写入日志文件的路径，供"查看详情"按钮使用
        """
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        tb_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        log_entry = (
            f"\n{'='*60}\n"
            f"[{timestamp}] {context}\n"
            f"异常类型: {type(exc).__name__}\n"
            f"异常消息: {exc}\n"
            f"完整堆栈:\n{tb_str}\n"
        )
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            with open(_ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except OSError:
            LOGGER.error("写入错误日志文件失败", exc_info=True)
        LOGGER.error("%s: %s", context, exc, exc_info=True)
        return _ERROR_LOG_FILE

    def _show_simplified_error(self, title: str, context: str, exc: BaseException):
        """优先级三：显示简明错误消息，完整堆栈写入日志文件

        对用户显示异常类型和消息摘要，不暴露完整堆栈跟踪。
        同时提供"查看详情"选项，点击后打开日志查看对话框。

        Args:
            title: messagebox 标题
            context: 错误上下文描述（如 "评估过程中发生错误"）
            exc: 捕获的异常对象
        """
        log_path = self._log_error_to_file(context, exc)
        exc_type_name = type(exc).__name__
        simple_msg = (
            f"{context}：\n\n"
            f"  {exc_type_name}: {exc}\n\n"
            f"详细信息已记录到日志文件：\n  {log_path}\n\n"
            f"是否查看完整错误详情？"
        )
        if messagebox.askyesno(title, simple_msg, default='no'):
            self._show_error_log_dialog(log_path)

    def _show_error_log_dialog(self, log_path: str):
        """优先级三：打开日志查看对话框（ScrolledText 显示完整堆栈）"""
        from tkinter.scrolledtext import ScrolledText
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except OSError:
            content = "无法读取日志文件。"

        win = tk.Toplevel(self.root)
        win.title("错误详情")
        win.geometry("700x500")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text=f"日志文件: {log_path}",
                  font=('Microsoft YaHei', 9)).pack(padx=10, pady=5, anchor='w')
        text = ScrolledText(win, wrap='word', font=('Consolas', 9))
        text.pack(fill='both', expand=True, padx=10, pady=5)
        text.insert('1.0', content)
        text.config(state='disabled')

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="关闭",
                   command=win.destroy).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="复制全部",
                   command=lambda: [
                       win.clipboard_clear(),
                       win.clipboard_append(content),
                       messagebox.showinfo("提示", "已复制到剪贴板", parent=win)
                   ]).pack(side='left', padx=5)

    def _pre_validate_form(self):
        """表单级预检（Section V 第三层验证）

        在打开进度条之前执行完整性预检，避免半途失败需要清理进度弹窗。
        检查项：
          - 至少有一个接触者（家庭或社会）
          - 基本信息必填字段合法
          - 接触者数量未超上限

        Returns:
            (is_valid, error_message) — is_valid=True 时 error_message 为空串
        """
        # 至少有一个接触者
        family_count = len(getattr(self, 'family_entries', []))
        social_count = len(getattr(self, 'social_entries', []))
        if family_count == 0 and social_count == 0:
            return False, "请至少添加一个家庭成员或社会接触者后再执行评估。"

        # 使用 FieldValidator 检查接触者数量上限
        validator = getattr(self, 'field_validator', None)
        if validator is not None:
            ok, errors = validator.validate_contact_counts(family_count, social_count)
            if not ok:
                return False, '；'.join(errors.values())

            # 检查基本信息关键字段（年龄必填且合法）
            try:
                checks = []
                if 'age' in self.basic_info_vars:
                    checks.append({
                        'field': 'age', 'value': self.basic_info_vars['age'],
                        'rules': ['non_empty', 'int'], 'label': '年龄',
                    })
                if 'flp_percentage' in self.basic_info_vars:
                    checks.append({
                        'field': 'flp_percentage', 'value': self.basic_info_vars['flp_percentage'],
                        'rules': ['int'], 'label': '家庭潜伏感染比例',
                    })
                if 'hrsp_percentage' in self.basic_info_vars:
                    checks.append({
                        'field': 'hrsp_percentage', 'value': self.basic_info_vars['hrsp_percentage'],
                        'rules': ['int'], 'label': '高危人群比例',
                    })
                if checks:
                    ok, errors = validator.validate_form(checks)
                    if not ok:
                        return False, '；'.join(errors.values())
            except Exception:
                # 预检失败不阻塞，由后续 _safe_int_convert 兜底
                pass
        return True, ''

    def _run_assessment(self):
        """执行风险评估"""
        # 检查是否已有评估在进行
        if self.assessing.is_set():
            messagebox.showinfo("提示", "风险评估正在进行中，请稍候...")
            return

        # 检查模型是否正在训练
        if self._training.is_set():
            messagebox.showwarning("警告", "模型正在训练中，请稍后再试！")
            return

        # Section V: 表单级预检（在打开进度条前执行，失败快速返回）
        form_ok, form_err = self._pre_validate_form()
        if not form_ok:
            messagebox.showwarning("输入不完整", form_err)
            return

        # 打开弹窗式进度条
        self._open_progress_popup("结核病风险评估")
        self._update_progress(0, "准备开始评估...")

        # 在主线程中预先收集所有GUI数据，避免后台线程访问tkinter变量
        patient_info = None
        family_members = None
        social_contacts = None
        try:
            # 设置评估标志，放在try块最前面确保能正确重置
            self.assessing.set()

            self._update_progress(5, "验证输入数据...")

            scenario = self.scenario_var.get()
            if scenario != 'custom' and scenario in self.scenario_defaults:
                self.current_scenario = scenario
            else:
                self.current_scenario = None

            self._update_progress(10, "收集患者基本信息...")

            # 收集患者基本信息并进行范围验证（使用安全转换）
            # Section V: 范围常量统一从 constants.py 引用
            age = self._safe_int_convert(self.basic_info_vars['age'], "年龄", _MIN_AGE, _MAX_AGE)
            flp_percentage = self._safe_int_convert(self.basic_info_vars['flp_percentage'], "家庭潜伏感染比例", _PERCENTAGE_MIN, _PERCENTAGE_MAX)
            hrsp_percentage = self._safe_int_convert(self.basic_info_vars['hrsp_percentage'], "高危人群比例", _PERCENTAGE_MIN, _PERCENTAGE_MAX)
            delay_days = self._safe_int_convert(self.basic_info_vars['delay_days'], "延迟治疗天数", _MIN_DELAY_DAYS, _MAX_DELAY_DAYS)
            cough_freq = self._safe_int_convert(self.basic_info_vars['cough_freq'], "咳嗽频率", _MIN_COUGH_FREQ, _MAX_COUGH_FREQ)

            # 中文选项到数值的映射
            def _map_chinese_option(value, option_map, default):
                if value in option_map:
                    return option_map[value]
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return default

            # 症状严重程度映射
            symptoms_map = {"无症状": 1, "轻度": 2, "中度": 3, "重度": 4}
            symptoms = _map_chinese_option(self.basic_info_vars['symptoms'].get(), symptoms_map, 2)

            # 家庭居住条件映射
            living_map = {"非常拥挤": 1, "拥挤": 2, "一般": 3, "宽敞": 4, "非常宽敞": 5}
            family_living_conditions = _map_chinese_option(self.basic_info_vars['family_living_conditions'].get(), living_map, 3)

            # 通风条件映射
            vent_map = {"极差": 1, "差": 2, "一般": 3, "好": 4, "极好": 5}
            ventilation = _map_chinese_option(self.basic_info_vars['ventilation'].get(), vent_map, 3)

            treatment_duration = self._safe_int_convert(self.basic_info_vars['treatment_duration'], "治疗持续月数", _MIN_TREATMENT_DURATION_MONTHS, _MAX_TREATMENT_DURATION_MONTHS)

            # GUI Combobox 使用 "1"/"2" 编码，通过 _safe_int_convert 兼容中文值
            # Wizard GUI 的 Combobox 存储中文标签，通过 _combo_maps 翻译为实际数值
            sputum_smear_raw = self.basic_info_vars['sputum_smear'].get()
            has_cavity_raw = self.basic_info_vars['has_cavity'].get()
            active_tb_raw = self.basic_info_vars['active_tb'].get()
            treatment_raw = self.basic_info_vars['treatment'].get()
            if hasattr(self, '_combo_maps'):
                sputum_smear_raw = self._combo_maps.get('sputum_smear', {}).get(sputum_smear_raw, sputum_smear_raw)
                has_cavity_raw = self._combo_maps.get('has_cavity', {}).get(has_cavity_raw, has_cavity_raw)
                active_tb_raw = self._combo_maps.get('active_tb', {}).get(active_tb_raw, active_tb_raw)
                treatment_raw = self._combo_maps.get('treatment', {}).get(treatment_raw, treatment_raw)
            sputum_smear = self._safe_int_convert(sputum_smear_raw, "痰涂片检查", 1, 2)
            has_cavity = self._safe_int_convert(has_cavity_raw, "空洞情况", 1, 2)
            active_tb = self._safe_int_convert(active_tb_raw, "是否结核患者", 1, 2)
            treatment = self._safe_int_convert(treatment_raw, "治疗状态", 1, 2)

            patient_info = {
                'FCI': 0,
                'SNC': 0,
                'FLP': flp_percentage,
                'HRSP': hrsp_percentage,
                'FTD': delay_days,
                'basic_info': {
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
                    'ventilation': ventilation,
                }
            }

            self._update_progress(20, "处理家庭成员数据...")

            family_members = []
            total_family = len(self.family_entries)
            for idx, entry in enumerate(self.family_entries, 1):
                try:
                    # 安全转换家庭成员数据
                    age = self._safe_int_convert(entry.get('age', 30), f"家庭成员{idx}年龄", _MIN_AGE, _MAX_AGE)
                    single_duration = self._safe_int_convert(entry.get('single_duration', 30), f"家庭成员{idx}单次接触时长", 0, _MAX_SINGLE_DURATION_MINUTES)
                    freq_density = self._safe_int_convert(entry.get('freq_density', 14), f"家庭成员{idx}接触频率", 0, _MAX_FREQ_DENSITY)
                    time_span = self._safe_int_convert(entry.get('time_span', 4), f"家庭成员{idx}接触周期", 1, _MAX_TIME_SPAN_WEEKS)
                    ventilation = self._safe_int_convert(entry.get('ventilation', 3), f"家庭成员{idx}通风条件", _MIN_VENTILATION, _MAX_VENTILATION)

                    member = {
                        'name': entry.get('name', ''),
                        'age': age,
                        'relationship': entry.get('relationship', '其他'),
                        'single_duration': single_duration,
                        'freq_density': freq_density,
                        'time_span': time_span,
                        'cumulative_exposure': calculate_cumulative_exposure(
                            single_duration,
                            freq_density,
                            time_span,
                            entry.get('workplace_type', '非油田')
                        ),
                        'has_tb': self._to_bool(entry.get('has_tb', '否')),
                        'has_symptoms': self._to_bool(entry.get('has_symptoms', '否')),
                        'bcg_vaccine': self._to_bool(entry.get('bcg_vaccine', '是')),
                        'past_illness': self._to_bool(entry.get('past_illness', '否')),
                        'past_illness_type': self._convert_chinese_to_value(entry.get('past_illness_type', ''), self.ILLNESS_TYPE_MAPPING) if _is_yes(entry.get('past_illness', '否')) else "none",
                        'contact_distance': self._convert_chinese_to_value(entry.get('contact_distance', '近'), self.DISTANCE_MAPPING),
                        'ventilation': ventilation,
                        'exposure_setting': self._convert_chinese_to_value(entry.get('exposure_setting', '一般'), self.SETTING_MAPPING),
                        'ethnicity': entry.get('ethnicity', '汉族'),
                        'origin_altitude': entry.get('origin_altitude', 300),
                        'workplace_type': entry.get('workplace_type', '非油田'),
                        'idu_status': entry.get('idu_status', '否'),
                        'district': entry.get('district', '克拉玛依区'),
                        'months_since_migration': entry.get('months_since_migration', 0),
                        '_id': entry.get('_id', None)
                    }
                    family_members.append(member)

                    # 更新进度
                    if total_family > 0:
                        progress = 20 + int((idx / total_family) * 15)
                        self._update_progress(progress, f"处理家庭成员 {idx}/{total_family}...")
                except ValueError as e:
                    raise ValueError(f"家庭成员{idx}数据错误：{str(e)}")

            self._update_progress(35, "处理社会接触者数据...")

            social_contacts = []
            total_social = len(self.social_entries)
            for idx, entry in enumerate(self.social_entries, 1):
                try:
                    # 安全转换社会接触者数据
                    age = self._safe_int_convert(entry.get('age', 30), f"社会接触者{idx}年龄", _MIN_AGE, _MAX_AGE)
                    single_duration = self._safe_int_convert(entry.get('single_duration', 30), f"社会接触者{idx}单次接触时长", 0, _MAX_SINGLE_DURATION_MINUTES)
                    freq_density = self._safe_int_convert(entry.get('freq_density', 2), f"社会接触者{idx}接触频率", 0, _MAX_FREQ_DENSITY)
                    time_span = self._safe_int_convert(entry.get('time_span', 4), f"社会接触者{idx}接触周期", 1, _MAX_TIME_SPAN_WEEKS)
                    ventilation = self._safe_int_convert(entry.get('ventilation', 3), f"社会接触者{idx}通风条件", _MIN_VENTILATION, _MAX_VENTILATION)

                    contact = {
                        'name': entry.get('name', ''),
                        'age': age,
                        'single_duration': single_duration,
                        'freq_density': freq_density,
                        'time_span': time_span,
                        'cumulative_exposure': calculate_cumulative_exposure(
                            single_duration,
                            freq_density,
                            time_span,
                            entry.get('workplace_type', '非油田')
                        ),
                        'is_high_risk': self._to_bool(entry.get('is_high_risk', '否')),
                        'has_tb': self._to_bool(entry.get('has_tb', '否')),
                        'past_illness': self._to_bool(entry.get('past_illness', '否')),
                        'past_illness_type': self._convert_chinese_to_value(entry.get('past_illness_type', ''), self.ILLNESS_TYPE_MAPPING) if _is_yes(entry.get('past_illness', '否')) else "none",
                        'bcg_vaccine': self._to_bool(entry.get('bcg_vaccine', '是')),
                        'ventilation': ventilation,
                        'has_symptoms': self._to_bool(entry.get('has_symptoms', '否')),
                        'contact_distance': self._convert_chinese_to_value(entry.get('contact_distance', '中等'), self.DISTANCE_MAPPING),
                        'exposure_setting': self._convert_chinese_to_value(entry.get('exposure_setting', '一般'), self.SETTING_MAPPING),
                        'ethnicity': entry.get('ethnicity', '汉族'),
                        'origin_altitude': entry.get('origin_altitude', 300),
                        'workplace_type': entry.get('workplace_type', '非油田'),
                        'idu_status': entry.get('idu_status', '否'),
                        'district': entry.get('district', '克拉玛依区'),
                        'months_since_migration': entry.get('months_since_migration', 0),
                        '_id': entry.get('_id', None)
                    }
                    social_contacts.append(contact)

                    # 更新进度
                    if total_social > 0:
                        progress = 35 + int((idx / total_social) * 10)
                        self._update_progress(progress, f"处理社会接触者 {idx}/{total_social}...")
                except ValueError as e:
                    raise ValueError(f"社会接触者{idx}数据错误：{str(e)}")
        except (ValueError, tk.TclError) as e:
            self.assessing.clear()
            self._reset_progress()
            messagebox.showerror("输入错误", f"数据验证失败：{str(e)}")
            return
        except Exception as e:
            self.assessing.clear()
            self._reset_progress()
            LOGGER.error("捕获未处理异常", exc_info=True)
            messagebox.showerror("错误", f"数据收集过程中发生未预期的错误：{str(e)}")
            return

        def assessment_task(patient_info, family_members, social_contacts):
            if self._closing:
                return
            try:
                # 优先级一：检查取消信号
                if self._is_cancelled():
                    self._safe_progress_update(0, "评估已取消")
                    self.root.after(0, lambda: self._reset_progress())
                    self.root.after(0, lambda: messagebox.showinfo("已取消", "评估任务已被用户取消。"))
                    return

                if not self._closing and self.root is not None:
                    self._safe_progress_update(45, "初始化数据结构...")

                with self._data_lock:
                    self.family_members = family_members
                    self.family_count = len(family_members)
                    self.social_contacts = social_contacts
                    self.contact_count = len(social_contacts)

                # 优先级一：检查取消信号（数据结构初始化后）
                if self._is_cancelled():
                    self._safe_progress_update(0, "评估已取消")
                    self.root.after(0, lambda: self._reset_progress())
                    self.root.after(0, lambda: messagebox.showinfo("已取消", "评估任务已被用户取消。"))
                    return

                if not self._closing and self.root is not None:
                    self._safe_progress_update(55, "计算 FCI 和 SNC 得分...")

                patient_info['FCI'] = self._assessment_service.calculate_fci_score(
                    family_members,
                    patient_info['basic_info'].get('family_living_conditions', 3)
                )
                patient_info['SNC'] = self._assessment_service.calculate_snc_score(social_contacts)

                with self._data_lock:
                    self.patient_info = patient_info

                # 优先级一：检查取消信号（FCI/SNC 计算后）
                if self._is_cancelled():
                    self._safe_progress_update(0, "评估已取消")
                    self.root.after(0, lambda: self._reset_progress())
                    self.root.after(0, lambda: messagebox.showinfo("已取消", "评估任务已被用户取消。"))
                    return

                if not self._closing and self.root is not None:
                    self._safe_progress_update(70, "执行风险评估...")

                with self._data_lock:
                    self._assessment_error = None
                self.assess_risk()

                # 优先级一：检查取消信号（SEIR 积分后）
                if self._is_cancelled():
                    self._safe_progress_update(0, "评估已取消")
                    self.root.after(0, lambda: self._reset_progress())
                    self.root.after(0, lambda: messagebox.showinfo("已取消", "评估任务已被用户取消。"))
                    return

                if not self._closing and self.root is not None:
                    with self._data_lock:
                        assessment_error = self._assessment_error
                    if assessment_error:
                        self.root.after(0, lambda err=assessment_error: messagebox.showerror("评估错误", err))
                        self.root.after(0, lambda: self._reset_progress())
                    else:
                        self.root.after(0, lambda: self._update_progress(85, "生成评估结果..."))
                        self.root.after(0, self._update_gui_after_assessment)

            except (RuntimeError, tk.TclError):
                if not self._closing:
                    LOGGER.error("捕获未处理异常", exc_info=True)
                    try:
                        self.root.after(0, lambda: self._reset_progress())
                    except (RuntimeError, tk.TclError):
                        pass
            except ValueError as e:
                # 优先级三：输入数据问题，简明提示
                if not self._closing and self.root is not None:
                    exc = e
                    try:
                        self.root.after(0, lambda: self._reset_progress())
                        self.root.after(0, lambda: self._show_simplified_error(
                            "输入数据错误", "评估过程中发生输入数据错误", exc))
                    except (RuntimeError, tk.TclError):
                        pass
            except ImportError as e:
                # 优先级三：依赖缺失，简明提示
                if not self._closing and self.root is not None:
                    exc = e
                    try:
                        self.root.after(0, lambda: self._reset_progress())
                        self.root.after(0, lambda: self._show_simplified_error(
                            "依赖缺失", "评估过程中缺少必要的依赖模块", exc))
                    except (RuntimeError, tk.TclError):
                        pass
            except Exception as e:
                # 优先级三：兜底异常，简明提示 + 完整堆栈写入日志文件
                if not self._closing and self.root is not None:
                    exc = e
                    try:
                        self.root.after(0, lambda: self._reset_progress())
                        self.root.after(0, lambda: self._show_simplified_error(
                            "评估错误", "评估过程中发生未预期的错误", exc))
                    except (RuntimeError, tk.TclError):
                        pass
            finally:
                self.assessing.clear()

        # 启动后台线程执行评估，传递预先收集的数据
        try:
            thread = threading.Thread(target=assessment_task, args=(patient_info, family_members, social_contacts), daemon=True)
            thread.start()
        except Exception as e:
            # 如果线程启动失败，重置标志
            self.assessing.clear()
            self._reset_progress()
            messagebox.showerror("错误", f"无法启动评估线程：{str(e)}")

    def _update_gui_after_assessment(self):
        """评估完成后在主线程更新GUI（策略模式：根据界面模式分流）

        优先级二：ML 预测移至后台线程，避免阻塞主线程。
        先展示非 ML 结果（风险评分、SEIR 曲线等），同时在后台线程启动 ML 预测。
        ML 预测完成后通过 root.after 通知主线程更新 ML 相关图表。
        """
        if self._closing:
            return
        try:
            self._update_progress(90, "显示评估结果...")

            # Section IX: 评估完成，清除 results_stale 标记并隐藏横幅
            # 新结果已生成，旧数据修改导致的结果失效状态不再适用
            try:
                self.results_stale = False
                if hasattr(self, '_hide_stale_banner'):
                    self._hide_stale_banner()
            except Exception:
                LOGGER.debug("清除 results_stale 标记失败（非致命）", exc_info=True)

            # 优先级六：通过 adapter 统一展示结果和绘制图表（消除 ui_mode 分支）
            if self.adapter is not None:
                self.adapter.display_results()
                self.root.update_idletasks()
                self._update_progress(95, "绘制图表...")
                self.adapter.draw_charts()
            else:
                # adapter 未初始化（防御性回退）
                LOGGER.warning("adapter 未初始化，跳过结果展示")

            # 优先级二：ML 预测移至后台线程
            # 先完成非 ML 结果展示，进度条推进到 100%，然后在后台执行 ML 预测
            self._update_progress(100, "评估完成！")

            # 数据安全：记录关键操作（评估）审计
            self._log_security_operation("ASSESS", details={"overall_risk": self.results.get('overall_risk', '')})

            if self.ml_predictor is not None:
                # 在主线程预检合成数据警告（tkinter 非线程安全）
                if self._precheck_ml_synthetic_warning():
                    # 用户确认，启动后台 ML 预测线程
                    self._start_ml_prediction_background()
                # else: 用户拒绝合成数据警告，跳过 ML 预测
            else:
                # 无 ML 预测器，延迟关闭进度条（让用户看到"评估完成"消息）
                self._schedule_delayed_progress_reset(2000)
        except Exception as e:
            # 优先级三：简明错误提示 + 完整堆栈写入日志文件
            self._reset_progress()
            self._show_simplified_error(
                "结果显示错误", "结果显示过程中发生错误", e)

    def _precheck_ml_synthetic_warning(self) -> bool:
        """优先级二：在主线程预检合成数据警告

        tkinter 非线程安全，messagebox 必须在主线程调用。
        若用户确认，设置 _synthetic_warning_shown 标志，
        使后台线程中的 _run_ml_predictions 跳过警告直接执行。

        Returns:
            True 表示用户确认（或无需警告），可启动后台 ML 预测；
            False 表示用户拒绝或 ml_predictor 不可用
        """
        if self.ml_predictor is None:
            return False

        current_type = 'real' if self.ml_predictor.use_real_data else 'synthetic'
        if not self.ml_predictor.use_real_data and (
            not getattr(self, '_synthetic_warning_shown', False)
            or getattr(self, '_last_used_model_type', None) != current_type
        ):
            warning_msg = (
                "⚠️ 重要提示：\n\n"
                "当前机器学习模型基于合成数据训练，\n"
                "预测结果仅供参考，不能替代临床诊断！\n\n"
                "如需实际临床应用，请使用本地流行病学数据\n"
                "重新训练模型（ML标签页 -> 导入真实数据）。\n\n"
                "是否继续使用合成数据模型进行预测？"
            )
            if not messagebox.askyesno("合成数据模型警告", warning_msg):
                return False
            self._synthetic_warning_shown = True
            self._last_used_model_type = current_type
        return True

    def _start_ml_prediction_background(self):
        """优先级二：在后台线程执行 ML 预测，完成后通知主线程更新图表

        复用 _start_training_thread 基础设施跟踪后台线程，
        便于优雅退出时 join。ML 预测期间显示占位提示。
        """
        # 显示 ML 占位提示（若存在 ML 图表区域）
        self._show_ml_placeholder("正在计算 ML 预测...")

        # 关闭进度条（非 ML 部分已完成，延迟1秒让用户看到"评估完成"）
        self._schedule_delayed_progress_reset(1000)

        def _ml_background_task():
            """后台线程：执行 ML 预测，完成后通过 root.after 通知主线程"""
            try:
                self._run_ml_predictions()
                # 成功完成，通知主线程更新 ML 图表
                if not self._closing and self.root is not None:
                    try:
                        self.root.after(0, self._on_ml_prediction_done)
                    except (RuntimeError, tk.TclError):
                        pass
            except (ValueError, RuntimeError, TypeError, KeyError) as e:
                LOGGER.warning("后台 ML 预测失败: %s", e, exc_info=True)
                if not self._closing and self.root is not None:
                    try:
                        self.root.after(0, lambda err=e: self._on_ml_prediction_error(err))
                    except (RuntimeError, tk.TclError):
                        pass
            except Exception as e:
                LOGGER.error("后台 ML 预测发生未预期错误: %s", e, exc_info=True)
                if not self._closing and self.root is not None:
                    try:
                        self.root.after(0, lambda err=e: self._on_ml_prediction_error(err))
                    except (RuntimeError, tk.TclError):
                        pass

        # 启动后台线程（复用 _start_training_thread 进行跟踪）
        self._start_training_thread(_ml_background_task)

    def _show_ml_placeholder(self, message: str):
        """优先级二：在 ML 图表区域显示占位提示

        Args:
            message: 占位提示文本（如 "正在计算 ML 预测..."）
        """
        try:
            # 若存在 ML 卡片，更新其值标签为占位文本
            if hasattr(self, 'ml_card') and self.ml_card:
                self.ml_card['value_label'].configure(text=message)
            # 若存在 ML 图表 canvas，显示占位文本
            if hasattr(self, 'ml_compare_canvas') and self.ml_compare_canvas is not None:
                if MATPLOTLIB_AVAILABLE and hasattr(self, 'ml_compare_figure'):
                    self.ml_compare_figure.clear()
                    ax = self.ml_compare_figure.add_subplot(111)
                    ax.text(0.5, 0.5, message, ha='center', va='center',
                           fontsize=14, color='gray', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    self.ml_compare_canvas.draw_idle()
        except Exception:
            LOGGER.debug("显示 ML 占位提示失败（非致命）", exc_info=True)

    def _on_ml_prediction_done(self):
        """优先级二：后台 ML 预测完成后的主线程回调

        更新 ML 相关图表和结果展示。
        """
        if self._closing:
            return
        try:
            self.root.update_idletasks()
            if hasattr(self, '_update_ml_charts'):
                self._update_ml_charts()
            if hasattr(self, '_display_ml_results'):
                self._display_ml_results()
            # 模型证据面板：ML 预测完成后刷新三层递进分解（含结果文本区回填）
            if hasattr(self, 'refresh_model_evidence_three_layer'):
                try:
                    self.refresh_model_evidence_three_layer()
                except Exception as e:
                    LOGGER.debug("三层递进面板刷新失败（非致命）: %s", e)
        except (ValueError, RuntimeError, TypeError, KeyError) as e:
            LOGGER.warning("ML 图表更新失败: %s", e, exc_info=True)

    def _on_ml_prediction_error(self, exc: BaseException):
        """优先级二：后台 ML 预测失败后的主线程回调

        在 ML 图表区域显示错误提示，不阻塞用户操作。
        """
        if self._closing:
            return
        try:
            error_msg = f"ML 预测失败: {type(exc).__name__}: {exc}"
            # 更新 ML 卡片显示错误
            if hasattr(self, 'ml_card') and self.ml_card:
                self.ml_card['value_label'].configure(text="ML 预测失败")
            # 在 ML 图表区域显示错误
            if hasattr(self, 'ml_compare_canvas') and self.ml_compare_canvas is not None:
                if MATPLOTLIB_AVAILABLE and hasattr(self, 'ml_compare_figure'):
                    self.ml_compare_figure.clear()
                    ax = self.ml_compare_figure.add_subplot(111)
                    ax.text(0.5, 0.5, error_msg, ha='center', va='center',
                           fontsize=10, color='red', transform=ax.transAxes,
                           wrap=True)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    self.ml_compare_canvas.draw_idle()
            LOGGER.warning("ML 预测失败（已显示错误提示）: %s", exc, exc_info=True)
        except Exception:
            LOGGER.debug("显示 ML 错误提示失败（非致命）", exc_info=True)

    def _on_closing(self):
        """窗口关闭事件处理，确保后台线程安全结束"""
        if self.assessing.is_set():
            if not messagebox.askokcancel("关闭", "正在进行风险评估，确定要关闭吗？"):
                return
        self._closing = True
        self._stop_progress_polling()
        self._close_progress_popup()

        # 停止AI消息轮询定时器
        if hasattr(self, '_stop_ai_polling'):
            try:
                self._stop_ai_polling()
            except Exception:
                pass

        # 取消SEIR参数防抖定时器
        seir_after_id = getattr(self, '_seir_param_after_id', None)
        if seir_after_id is not None and self.root is not None:
            try:
                self.root.after_cancel(seir_after_id)
            except Exception:
                pass
            self._seir_param_after_id = None

        # 阻塞等待评估线程结束（Event.wait 替代 while+root.update，消除重入风险）
        self.assessing.wait(timeout=10.0)
        # 优雅等待后台训练线程完成，避免资源泄漏和不完整写入
        self._cleanup_training_threads(timeout=3.0)

        # Section IV：保存配置 + 停止自动保存 + 清理临时文件
        if hasattr(self, 'auto_save'):
            self.auto_save.stop()
            AutoSaveManager.clear_autosave()
        if hasattr(self, 'config'):
            try:
                self.config.window_geometry = self.root.geometry()
                self.config.save()
            except Exception:
                pass  # 配置保存失败不应阻止退出

        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _show_error_details(self, title, header, errors):
        """显示完整错误详情的弹窗，支持滚动和复制"""
        from tkinter.scrolledtext import ScrolledText

        error_window = tk.Toplevel(self.root)
        error_window.title(title)
        error_window.geometry("600x400")
        error_window.transient(self.root)
        error_window.grab_set()

        ttk.Label(error_window, text=header, font=('Microsoft YaHei', 10)).pack(padx=10, pady=5, anchor='w')

        text_frame = ttk.Frame(error_window)
        text_frame.pack(fill='both', expand=True, padx=10, pady=5)

        error_text = ScrolledText(text_frame, wrap='word', font=('Microsoft YaHei', 9))
        error_text.pack(fill='both', expand=True)
        error_text.insert('1.0', "\n".join(errors))
        error_text.config(state='normal')

        btn_frame = ttk.Frame(error_window)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="关闭", command=error_window.destroy).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="复制全部", command=lambda: [error_window.clipboard_clear(), error_window.clipboard_append("\n".join(errors)), messagebox.showinfo("提示", "已复制到剪贴板", parent=error_window)]).pack(side='left', padx=5)

    def _collect_basic_info_from_vars(self):
        """从向导式界面的变量收集患者信息（改为返回数据而非修改 patient_info）
        兼容性：保留方法，返回收集的变量字典
        """
        if not hasattr(self, 'basic_info_vars'):
            return {}

        # 收集数据并返回字典
        collected = {}
        for var_name, var_obj in self.basic_info_vars.items():
            try:
                value = var_obj.get()
                collected[var_name] = value
            except Exception as e:
                LOGGER.warning(f"收集变量 {var_name} 失败: {e}")
        return collected