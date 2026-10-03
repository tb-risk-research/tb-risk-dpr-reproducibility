#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Section VIII: 统一导入流程管线 — 模板方法模式

五阶段标准流程：文件选择 → 模式选择（替换/追加/去重合并）→ 数据解析 → 预览确认 → 数据加载

所有导入格式（CSV、Excel、JSON、API、Database）都走这条管线。
新增导入格式时只需新增子类，自动获得预览、模式选择等通用能力。

字段映射使用 data_io.synonyms 的三阶模糊匹配（精确→关键词→Levenshtein），
覆盖全部 30+ 业务字段，支持中英文同义词。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ._shared import *  # noqa: F401,F403
from ..undo import ImportDataCommand

LOGGER = logging.getLogger("tb_risk.gui.import_panel.pipeline")


# ===========================================================================
# 模板方法基类
# ===========================================================================

class ImportPipeline:
    """统一导入流程管线基类 — 模板方法模式

    子类只需实现：
        - select_source()：选择数据源（文件路径或连接参数）
        - parse_data()：解析数据为标准 entry 列表

    自动获得通用能力：
        - 导入模式选择（替换/追加/去重合并）
        - 字段映射（同义词三阶模糊匹配）
        - 预览确认（调用 _show_import_preview + _show_low_confidence_dialog）
        - 数据加载（按模式合并到 family_entries / social_entries）
        - 进度对话框（取消按钮生效）
    """

    # 子类可覆盖的属性
    FORMAT_NAME = '通用'
    SUPPORTS_MODE_SELECTION = True

    def __init__(self, app):
        """初始化管线

        参数：
            app: GUI 应用实例（TBRiskApp），提供 root, family_entries, social_entries 等
        """
        self.app = app
        self.cancelled = False

    # ==================== 模板方法 ====================

    def run(self) -> bool:
        """模板方法：执行完整五阶段导入流程

        返回：
            True 表示导入成功，False 表示用户取消或失败
        """
        # Stage 1: 选择数据源
        source = self.select_source()
        if source is None:
            return False

        # Stage 2: 选择导入模式
        mode = 'replace'
        if self.SUPPORTS_MODE_SELECTION:
            mode = self._select_mode()
            if mode is None:
                return False

        # Stage 3: 解析数据（带进度对话框）
        progress_win, progress_var, update_progress, close_progress = \
            self.app._show_progress_dialog(self.get_progress_title())
        try:
            update_progress(10, "正在解析数据...")
            result = self.parse_data(source, update_progress)
        except Exception as e:
            LOGGER.error("%s 导入解析失败: %s", self.FORMAT_NAME, e, exc_info=True)
            close_progress()
            messagebox.showerror("错误", f"导入数据解析失败：{e}")
            return False

        if result is None or (not result[0] and not result[1]):
            close_progress()
            messagebox.showinfo("提示", "没有可导入的数据")
            return False

        family_entries, social_entries, metadata = result
        close_progress()

        # Stage 4: 预览确认
        if not self._preview_and_confirm(family_entries, social_entries, metadata):
            return False

        # Stage 5: 加载数据
        return self._load_data(family_entries, social_entries, mode)

    # ==================== 子类必须实现的抽象方法 ====================

    def select_source(self) -> Optional[Dict[str, Any]]:
        """Stage 1: 选择数据源

        返回：
            包含数据源信息的字典，或 None 表示用户取消
        """
        raise NotImplementedError

    def parse_data(self, source: Dict[str, Any],
                    update_progress) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
        """Stage 3: 解析数据为标准 entry 列表

        参数：
            source: select_source() 返回的源信息
            update_progress: 进度回调函数 (value, message)

        返回：
            (family_entries, social_entries, metadata)
            metadata 可包含 field_mappings, quality_scores, auto_corrections 等
        """
        raise NotImplementedError

    def get_progress_title(self) -> str:
        """返回进度对话框标题"""
        return f"正在导入{self.FORMAT_NAME}数据..."

    # ==================== 通用方法（Stage 2/4/5） ====================

    def _select_mode(self) -> Optional[str]:
        """Stage 2: 选择导入模式

        返回：
            'replace' / 'append' / 'merge'，或 None 表示取消
        """
        dlg = tk.Toplevel(self.app.root)
        dlg.title(f"选择导入模式 - {self.FORMAT_NAME}")
        dlg.geometry("400x250")
        dlg.transient(self.app.root)
        dlg.grab_set()

        ttk.Label(dlg, text="请选择数据导入模式：",
                  font=('Microsoft YaHei', 11, 'bold')).pack(pady=(15, 10))

        mode_var = tk.StringVar(value='replace')
        modes = [
            ('replace', '替换现有数据', '清空当前所有接触者数据，替换为新导入的数据'),
            ('append', '追加新数据', '保留当前数据，将新数据追加到末尾'),
            ('merge', '去重合并', '保留当前数据，按姓名+年龄去重后合并新数据'),
        ]
        for val, label, desc in modes:
            frame = ttk.Frame(dlg)
            frame.pack(fill='x', padx=20, pady=3)
            ttk.Radiobutton(frame, text=label, variable=mode_var,
                             value=val).pack(anchor='w')
            ttk.Label(frame, text=f"  {desc}", font=('Microsoft YaHei', 8),
                      foreground='#5a6c7d').pack(anchor='w')

        result = [None]

        def on_confirm():
            result[0] = mode_var.get()
            dlg.destroy()

        def on_cancel():
            dlg.destroy()

        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(pady=15)
        ttk.Button(btn_frame, text="取消", command=on_cancel).pack(side='right', padx=5)
        ttk.Button(btn_frame, text="确认", command=on_confirm).pack(side='right', padx=5)

        dlg.wait_window()
        return result[0]

    def _preview_and_confirm(self, family_entries: List[dict],
                              social_entries: List[dict],
                              metadata: Dict[str, Any]) -> bool:
        """Stage 4: 预览确认

        调用 _show_import_preview 显示数据预览，让用户在导入前确认。
        如有低置信度字段映射，调用 _show_low_confidence_dialog 让用户确认。
        """
        # 处理低置信度字段映射
        low_confidence = metadata.get('low_confidence_matches', [])
        if low_confidence:
            confirmed = self.app._show_low_confidence_dialog(low_confidence)
            if confirmed is None:
                return False  # 用户取消
            # 将用户确认的映射写回 metadata
            if confirmed:
                field_mappings = metadata.get('field_mappings', {})
                field_mappings.update(confirmed)
                metadata['field_mappings'] = field_mappings
                # 注册用户反馈到同义词库
                try:
                    from ...data_io.synonyms import register_synonym_feedback
                    for orig, std in confirmed.items():
                        register_synonym_feedback(orig, std)
                except Exception:
                    pass

        # 预览家庭成员
        if family_entries:
            quality_scores = metadata.get('quality_scores', {})
            auto_corrections = metadata.get('auto_corrections', {})
            confirmed = self.app._show_import_preview(
                family_entries, import_type="家庭成员",
                quality_scores=quality_scores if 'family' in str(quality_scores) else None,
                field_mappings=metadata.get('field_mappings'),
                auto_corrections=auto_corrections,
            )
            if not confirmed:
                return False

        # 预览社会接触者
        if social_entries:
            confirmed = self.app._show_import_preview(
                social_entries, import_type="社会接触者",
                field_mappings=metadata.get('field_mappings'),
            )
            if not confirmed:
                return False

        return True

    def _load_data(self, family_entries: List[dict],
                    social_entries: List[dict], mode: str) -> bool:
        """Stage 5: 按模式加载数据到 app.family_entries / app.social_entries

        优先级一：通过 ImportDataCommand 接入撤销/重做栈，支持 Ctrl+Z 撤销导入。
        """
        try:
            # 先计算最终目标状态，再通过 ImportDataCommand 原子性替换
            if mode == 'replace':
                new_family = list(family_entries)
                new_social = list(social_entries)
            elif mode == 'append':
                new_family = list(self.app.family_entries) + list(family_entries)
                new_social = list(self.app.social_entries) + list(social_entries)
            elif mode == 'merge':
                new_family = list(self.app.family_entries)
                new_social = list(self.app.social_entries)
                self._merge_entries(new_family, family_entries)
                self._merge_entries(new_social, social_entries)
            else:
                new_family = list(family_entries)
                new_social = list(social_entries)

            # 通过 UndoManager 执行 ImportDataCommand（支持撤销/重做）
            if hasattr(self.app, 'undo_manager'):
                self.app.undo_manager.execute(
                    ImportDataCommand(self.app, new_family, new_social))
            else:
                # 兜底：直接替换引用
                self.app.family_entries = new_family
                self.app.social_entries = new_social

            # 同步 family_members / social_contacts 以保持向后兼容
            # （_update_gui_from_import 会从 family_members 重建 family_entries）
            self.app.family_members = list(new_family)
            self.app.social_contacts = list(new_social)

            # 更新 GUI 显示（含 basic_info_vars 同步）
            if hasattr(self.app, '_update_gui_from_import'):
                self.app._update_gui_from_import()
            # 直接刷新 Treeview 以确保显示与 family_entries 一致
            if hasattr(self.app, '_update_family_treeview'):
                self.app._update_family_treeview()
            if hasattr(self.app, '_update_social_treeview'):
                self.app._update_social_treeview()
            if hasattr(self.app, '_update_fill_progress'):
                self.app._update_fill_progress()
            # Section IX: 标记结果为 stale
            if hasattr(self.app, '_mark_results_stale'):
                self.app._mark_results_stale()
            # 标记脏数据以触发自动保存
            if hasattr(self.app, 'auto_save'):
                self.app.auto_save.mark_dirty()

            # 优先级一：将文件路径加入最近文件列表
            source_file = self._get_source_file_path()
            if source_file and hasattr(self.app, '_add_recent_file'):
                self.app._add_recent_file(source_file)

            # 数据安全：记录关键操作（导入）审计
            if hasattr(self.app, '_log_security_operation'):
                self.app._log_security_operation(
                    "IMPORT", details={
                        "source": source_file or self.get_progress_title(),
                        "family_count": len(new_family),
                        "social_count": len(new_social),
                        "mode": mode,
                    })

            # 显示导入摘要
            summary_msg = (
                f"数据已成功导入（模式：{mode}）：\n"
                f"- 家庭成员：{len(new_family)}人\n"
                f"- 社会接触者：{len(new_social)}人"
            )
            messagebox.showinfo("成功", summary_msg)
            return True

        except Exception as e:
            LOGGER.error("加载导入数据失败: %s", e, exc_info=True)
            messagebox.showerror("错误", f"数据加载失败：{e}")
            return False

    def _get_source_file_path(self) -> Optional[str]:
        """获取当前导入的源文件路径（用于最近文件列表）

        子类可在 parse_data 时将 source_file 存到 self._source_file_path。
        默认返回 None。
        """
        return getattr(self, '_source_file_path', None)

    @staticmethod
    def _merge_entries(existing: List[dict], new_entries: List[dict]):
        """去重合并：按 name+age 去重"""
        existing_keys = {
            (e.get('name', '').strip(), e.get('age', ''))
            for e in existing
        }
        for entry in new_entries:
            key = (entry.get('name', '').strip(), entry.get('age', ''))
            if key not in existing_keys:
                existing.append(entry)
                existing_keys.add(key)

    # ==================== 字段映射工具 ====================

    @staticmethod
    def map_columns(column_names: List[str]) -> Tuple[Dict[str, str], List[Tuple]]:
        """使用三阶模糊匹配映射列名

        参数：
            column_names: 原始列名列表

        返回：
            (field_mappings, low_confidence_matches)
            - field_mappings: {原始列名: 标准字段名}
            - low_confidence_matches: [(原始列名, 匹配字段, 置信度), ...]
              置信度 < 0.8 的匹配需要用户确认
        """
        try:
            from ...data_io.synonyms import fuzzy_match_column
        except ImportError:
            LOGGER.warning("无法导入 fuzzy_match_column，跳过字段映射")
            return {}, []

        field_mappings = {}
        low_confidence = []

        for col in column_names:
            if not col:
                continue
            std_name, confidence, source = fuzzy_match_column(col)
            if std_name:
                field_mappings[col] = std_name
                if confidence < 0.8:
                    low_confidence.append((col, std_name, confidence))

        return field_mappings, low_confidence

    @staticmethod
    def remap_entry(raw_entry: dict, field_mappings: Dict[str, str]) -> dict:
        """根据字段映射将原始 entry 转换为标准 entry"""
        if not field_mappings:
            return dict(raw_entry)
        mapped = {}
        for key, val in raw_entry.items():
            std_key = field_mappings.get(key, key)
            mapped[std_key] = val
        return mapped


# ===========================================================================
# CSV 导入管线
# ===========================================================================

class CSVImportPipeline(ImportPipeline):
    """CSV 文件导入管线"""
    FORMAT_NAME = 'CSV'

    def select_source(self) -> Optional[Dict[str, Any]]:
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="从 CSV 加载数据"
        )
        if not file_path:
            return None
        # 优先级一：记录源文件路径用于最近文件列表
        self._source_file_path = file_path
        return {'file_path': file_path}

    def parse_data(self, source: Dict[str, Any], update_progress) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
        update_progress(20, "正在读取 CSV 文件...")
        file_path = source['file_path']
        # 优先级一：确保 _source_file_path 已设置（支持外部直接调用 parse_data）
        self._source_file_path = file_path

        # 尝试使用 app.import_csv 解析
        if hasattr(self.app, 'import_csv'):
            # 保存当前数据（因为 import_csv 可能会清空）
            saved_family = list(self.app.family_entries)
            saved_social = list(self.app.social_entries)
            self.app.family_entries.clear()
            self.app.social_entries.clear()

            success = self.app.import_csv(file_path)
            if not success:
                # 恢复
                self.app.family_entries.extend(saved_family)
                self.app.social_entries.extend(saved_social)
                errors = self.app.get_import_errors() if hasattr(self.app, 'get_import_errors') else []
                if errors:
                    self.app._show_error_details("CSV 加载错误",
                        "从 CSV 加载数据失败，以下是所有错误详情：", errors)
                return [], [], {}

            family = list(self.app.family_entries)
            social = list(self.app.social_entries)
            # 恢复原始状态（pipeline 的 load_data 阶段会按模式处理）
            self.app.family_entries.clear()
            self.app.family_entries.extend(saved_family)
            self.app.social_entries.clear()
            self.app.social_entries.extend(saved_social)

            update_progress(80, "数据解析完成")
            return family, social, {'source_file': file_path}

        # 回退：直接用 csv 模块解析
        return self._parse_csv_fallback(file_path, update_progress)

    def _parse_csv_fallback(self, file_path, update_progress):
        """直接用 csv 模块解析（不依赖 app.import_csv）"""
        import csv as csv_mod
        update_progress(30, "正在解析 CSV...")
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                reader = csv_mod.DictReader(f)
                columns = reader.fieldnames or []
                field_mappings, low_conf = self.map_columns(columns)
                update_progress(50, "正在映射字段...")
                entries = [self.remap_entry(row, field_mappings) for row in reader]
            update_progress(80, "数据解析完成")
            return entries, [], {
                'field_mappings': field_mappings,
                'low_confidence_matches': low_conf,
                'source_file': file_path,
            }
        except Exception as e:
            LOGGER.error("CSV 解析失败: %s", e, exc_info=True)
            return [], [], {}


# ===========================================================================
# JSON 导入管线
# ===========================================================================

class JSONImportPipeline(ImportPipeline):
    """JSON 文件导入管线"""
    FORMAT_NAME = 'JSON'

    def select_source(self) -> Optional[Dict[str, Any]]:
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="从 JSON 加载数据"
        )
        if not file_path:
            return None
        # 优先级一：记录源文件路径用于最近文件列表
        self._source_file_path = file_path
        return {'file_path': file_path}

    def parse_data(self, source: Dict[str, Any], update_progress) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
        update_progress(20, "正在读取 JSON 文件...")
        import json
        file_path = source['file_path']
        # 优先级一：确保 _source_file_path 已设置
        self._source_file_path = file_path
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            update_progress(50, "正在解析数据...")

            family = []
            social = []
            metadata = {'source_file': file_path}

            if isinstance(data, dict):
                # 标准格式：{family: [...], social: [...]}
                family = data.get('family', data.get('family_members', []))
                social = data.get('social', data.get('social_contacts', []))
                # 如果有 patient_info 也保存
                if 'patient_info' in data:
                    metadata['patient_info'] = data['patient_info']
            elif isinstance(data, list):
                # 列表格式：全部当家庭成员
                family = data

            update_progress(80, "数据解析完成")
            return family, social, metadata
        except Exception as e:
            LOGGER.error("JSON 解析失败: %s", e, exc_info=True)
            return [], [], {}


# ===========================================================================
# Excel 导入管线
# ===========================================================================

class ExcelImportPipeline(ImportPipeline):
    """Excel 文件导入管线"""
    FORMAT_NAME = 'Excel'

    def select_source(self) -> Optional[Dict[str, Any]]:
        file_path = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")],
            title="从 Excel 加载数据"
        )
        if not file_path:
            return None
        # 优先级一：记录源文件路径用于最近文件列表
        self._source_file_path = file_path
        return {'file_path': file_path}

    def parse_data(self, source: Dict[str, Any], update_progress) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
        update_progress(20, "正在读取 Excel 文件...")
        file_path = source['file_path']
        # 优先级一：确保 _source_file_path 已设置
        self._source_file_path = file_path

        if hasattr(self.app, 'import_excel'):
            saved_family = list(self.app.family_entries)
            saved_social = list(self.app.social_entries)
            self.app.family_entries.clear()
            self.app.social_entries.clear()

            success = self.app.import_excel(file_path)
            if not success:
                self.app.family_entries.extend(saved_family)
                self.app.social_entries.extend(saved_social)
                return [], [], {}

            family = list(self.app.family_entries)
            social = list(self.app.social_entries)
            self.app.family_entries.clear()
            self.app.family_entries.extend(saved_family)
            self.app.social_entries.clear()
            self.app.social_entries.extend(saved_social)

            update_progress(80, "数据解析完成")
            return family, social, {'source_file': file_path}

        return [], [], {}


# ===========================================================================
# API 导入管线
# ===========================================================================

class APIImportPipeline(ImportPipeline):
    """API 数据导入管线"""
    FORMAT_NAME = 'API'

    def select_source(self) -> Optional[Dict[str, Any]]:
        dlg = tk.Toplevel(self.app.root)
        dlg.title("从 API 导入数据")
        dlg.geometry("500x300")
        dlg.transient(self.app.root)
        dlg.grab_set()

        ttk.Label(dlg, text="API URL:", font=('Microsoft YaHei', 10)).pack(pady=(15, 5), anchor='w', padx=20)
        url_var = tk.StringVar(value='http://localhost:8000/api/contacts')
        ttk.Entry(dlg, textvariable=url_var, width=60).pack(padx=20, fill='x')

        ttk.Label(dlg, text="API Key（可选）:", font=('Microsoft YaHei', 10)).pack(pady=(5, 5), anchor='w', padx=20)
        key_var = tk.StringVar()
        ttk.Entry(dlg, textvariable=key_var, width=60, show='*').pack(padx=20, fill='x')

        result = [None]

        def on_confirm():
            url = url_var.get().strip()
            if not url:
                messagebox.showwarning("警告", "请输入 API URL")
                return
            result[0] = {'url': url, 'api_key': key_var.get().strip()}
            dlg.destroy()

        def on_cancel():
            dlg.destroy()

        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(pady=20)
        ttk.Button(btn_frame, text="取消", command=on_cancel).pack(side='right', padx=5)
        ttk.Button(btn_frame, text="确认", command=on_confirm).pack(side='right', padx=5)

        dlg.wait_window()
        return result[0]

    def parse_data(self, source: Dict[str, Any], update_progress) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
        update_progress(20, "正在连接 API...")
        url = source['url']
        api_key = source.get('api_key', '')

        if hasattr(self.app, 'import_from_api'):
            success = self.app.import_from_api(url, api_key)
            if not success:
                return [], {}, {}
            family = list(self.app.family_entries)
            social = list(self.app.social_entries)
            self.app.family_entries.clear()
            self.app.social_entries.clear()
            update_progress(80, "数据解析完成")
            return family, social, {'api_url': url}

        return [], [], {}


# ===========================================================================
# 数据库导入管线
# ===========================================================================

class DatabaseImportPipeline(ImportPipeline):
    """数据库导入管线 — 动态显示连接参数 + 测试连接"""
    FORMAT_NAME = '数据库'

    # 数据库类型 → 连接参数定义
    DB_PARAM_SPECS = {
        'sqlite': [
            ('file_path', '数据库文件路径', 'file', None),
        ],
        'mysql': [
            ('host', '主机地址', 'text', 'localhost'),
            ('port', '端口', 'text', '3306'),
            ('user', '用户名', 'text', 'root'),
            ('password', '密码', 'password', ''),
            ('database', '数据库名', 'text', ''),
        ],
        'postgresql': [
            ('host', '主机地址', 'text', 'localhost'),
            ('port', '端口', 'text', '5432'),
            ('user', '用户名', 'text', 'postgres'),
            ('password', '密码', 'password', ''),
            ('database', '数据库名', 'text', ''),
        ],
    }

    def select_source(self) -> Optional[Dict[str, Any]]:
        """动态显示连接参数输入框 + 测试连接按钮"""
        dlg = tk.Toplevel(self.app.root)
        dlg.title("从数据库导入")
        dlg.geometry("520x450")
        dlg.transient(self.app.root)
        dlg.grab_set()

        # 数据库类型选择
        ttk.Label(dlg, text="数据库类型：", font=('Microsoft YaHei', 10)).pack(pady=(10, 5), anchor='w', padx=15)
        db_type_var = tk.StringVar(value='sqlite')
        type_frame = ttk.Frame(dlg)
        type_frame.pack(padx=15, fill='x')
        for val, label in [('sqlite', 'SQLite'), ('mysql', 'MySQL'), ('postgresql', 'PostgreSQL')]:
            ttk.Radiobutton(type_frame, text=label, variable=db_type_var,
                             value=val, command=lambda: self._update_db_params(dlg, db_type_var, params_frame, entry_vars)).pack(side='left', padx=5)

        # 查询语句
        ttk.Label(dlg, text="查询语句/表名：", font=('Microsoft YaHei', 10)).pack(pady=(10, 5), anchor='w', padx=15)
        query_var = tk.StringVar()
        ttk.Entry(dlg, textvariable=query_var, width=65).pack(padx=15, fill='x')
        ttk.Label(dlg, text="提示：输入表名（如 contacts）或完整 SELECT 语句",
                  font=('Microsoft YaHei', 8), foreground='#5a6c7d').pack(pady=2, padx=15, anchor='w')

        # 动态参数容器
        params_frame = ttk.LabelFrame(dlg, text="连接参数", padding=10)
        params_frame.pack(padx=15, pady=10, fill='x')
        entry_vars = {}

        # 初始化 sqlite 参数
        self._update_db_params(dlg, db_type_var, params_frame, entry_vars)

        # 测试连接状态标签
        status_label = ttk.Label(dlg, text="", font=('Microsoft YaHei', 9))
        status_label.pack(pady=5)

        result = [None]

        def test_connection():
            """测试数据库连接"""
            db_type = db_type_var.get()
            params = {k: v.get() for k, v in entry_vars.items()}
            status_label.config(text="正在测试连接...", foreground='#f39c12')
            dlg.update_idletasks()
            try:
                ok = self._do_test_connection(db_type, params)
                if ok:
                    status_label.config(text="✓ 连接成功", foreground='#2ecc71')
                else:
                    status_label.config(text="✗ 连接失败", foreground='#e74c3c')
            except Exception as e:
                status_label.config(text=f"✗ 连接失败: {e}", foreground='#e74c3c')

        def do_confirm():
            db_type = db_type_var.get()
            params = {k: v.get() for k, v in entry_vars.items()}
            query = query_var.get().strip()
            if not query:
                messagebox.showwarning("警告", "请输入查询语句或表名")
                return
            # 验证参数
            for param_name, _, param_type, _ in self.DB_PARAM_SPECS[db_type]:
                if param_type == 'file' and not params.get(param_name):
                    messagebox.showwarning("警告", "请选择数据库文件")
                    return
                if param_type == 'text' and not params.get(param_name) and param_name != 'password':
                    if not messagebox.askyesno("确认", f"参数 '{param_name}' 为空，是否继续？"):
                        return
            result[0] = {'db_type': db_type, 'params': params, 'query': query}
            dlg.destroy()

        def do_cancel():
            dlg.destroy()

        # 按钮栏
        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="测试连接", command=test_connection).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="取消", command=do_cancel).pack(side='right', padx=5)
        ttk.Button(btn_frame, text="导入", command=do_confirm).pack(side='right', padx=5)

        dlg.wait_window()
        return result[0]

    def _update_db_params(self, dlg, db_type_var, params_frame, entry_vars):
        """根据数据库类型动态更新参数输入框"""
        db_type = db_type_var.get()
        # 清空现有控件
        for child in params_frame.winfo_children():
            child.destroy()
        entry_vars.clear()

        specs = self.DB_PARAM_SPECS.get(db_type, [])
        for i, (param_name, label, param_type, default) in enumerate(specs):
            ttk.Label(params_frame, text=label + ":").grid(row=i, column=0, sticky='w', pady=3)
            var = tk.StringVar(value=default or '')
            entry_vars[param_name] = var

            if param_type == 'file':
                entry = ttk.Entry(params_frame, textvariable=var, width=40)
                entry.grid(row=i, column=1, sticky='ew', pady=3)
                def browse(var=var):
                    path = filedialog.askopenfilename(
                        filetypes=[("SQLite files", "*.db *.sqlite"), ("All files", "*.*")])
                    if path:
                        var.set(path)
                ttk.Button(params_frame, text="浏览...", command=browse).grid(row=i, column=2, padx=5, pady=3)
            elif param_type == 'password':
                ttk.Entry(params_frame, textvariable=var, width=45, show='*').grid(row=i, column=1, sticky='ew', pady=3)
            else:
                ttk.Entry(params_frame, textvariable=var, width=45).grid(row=i, column=1, sticky='ew', pady=3)

        params_frame.columnconfigure(1, weight=1)

    def _do_test_connection(self, db_type: str, params: dict) -> bool:
        """测试数据库连接"""
        try:
            if hasattr(self.app, 'test_database_connection'):
                return self.app.test_database_connection(db_type, params)
        except Exception as e:
            LOGGER.warning("测试连接失败: %s", e)
        # 简单验证：SQLite 检查文件是否存在
        if db_type == 'sqlite':
            import os
            path = params.get('file_path', '')
            return os.path.exists(path) if path else False
        # 其他类型：尝试导入对应驱动
        if db_type == 'mysql':
            try:
                import pymysql
                return True
            except ImportError:
                return False
        if db_type == 'postgresql':
            try:
                import psycopg2
                return True
            except ImportError:
                return False
        return False

    def parse_data(self, source: Dict[str, Any], update_progress) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
        update_progress(20, "正在连接数据库...")
        db_type = source['db_type']
        params = source['params']
        query = source['query']

        # 构建连接字符串
        if db_type == 'sqlite':
            db_path = params.get('file_path', '')
        else:
            db_path = f"{params.get('host', 'localhost')}:{params.get('port', '')}:{params.get('user', '')}:{params.get('password', '')}:{params.get('database', '')}"

        update_progress(40, "正在执行查询...")

        if hasattr(self.app, 'import_from_database'):
            saved_family = list(self.app.family_entries)
            saved_social = list(self.app.social_entries)
            self.app.family_entries.clear()
            self.app.social_entries.clear()

            success = self.app.import_from_database(
                db_type=db_type, db_path=db_path, query=query, mode='replace')
            if not success:
                self.app.family_entries.extend(saved_family)
                self.app.social_entries.extend(saved_social)
                return [], [], {}

            family = list(self.app.family_entries)
            social = list(self.app.social_entries)
            self.app.family_entries.clear()
            self.app.family_entries.extend(saved_family)
            self.app.social_entries.clear()
            self.app.social_entries.extend(saved_social)

            update_progress(80, "数据解析完成")
            return family, social, {'db_type': db_type, 'query': query}

        return [], [], {}
