#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计日志模块

提供完整的审计追踪功能，包括：
- 导入审计（映射决策、缺失值填充、冲突处理）
- 临床操作审计（患者数据访问、修改、上报、审核）
- 防篡改哈希链（链式校验，确保日志不可篡改）
- 审计报告导出（JSON/HTML）
- 符合《医疗卫生机构网络安全管理办法》要求，保留期限15年
"""

import datetime
import hashlib
import html
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("tb_risk.data_io.audit")

# 审计日志轮转配置
_MAX_LOG_SIZE = 50 * 1024 * 1024  # 50MB
_MAX_EVENTS_MEMORY = 10000  # 内存缓存上限
_RETENTION_YEARS = 15  # 审计日志保留期限（年）

# 临床操作类型常量
OPERATION_TYPES = {
    'CREATE': '创建',
    'READ': '查看',
    'UPDATE': '修改',
    'DELETE': '删除',
    'SUBMIT': '提交审核',
    'APPROVE': '审核通过',
    'REJECT': '审核驳回',
    'REPORT': '上报疾控',
    'EXPORT': '导出数据',
    'ASSESS': '风险评估',
    'IMPORT_DATA': '数据导入',
    'LOGIN': '用户登录',
    'LOGOUT': '用户登出',
    'CONFIG_CHANGE': '配置变更',
}


# ==============================================================================
# 一、防篡改哈希链
# ==============================================================================

class HashChain:
    """哈希链实现，用于确保审计日志不可篡改。
    
    每个事件包含前一个事件的哈希值，形成链式结构。
    篡改任何一个事件都会导致后续所有哈希校验失败。
    """

    def __init__(self, seed: Optional[str] = None):
        """初始化哈希链。
        
        参数：
            seed: 创世块种子，如果为None则使用当前时间戳
        """
        self._previous_hash = seed or self._generate_genesis_hash()
        self._chain_length = 0

    def _generate_genesis_hash(self) -> str:
        """生成创世块哈希。"""
        genesis_data = f"tb_risk_audit_genesis_{datetime.datetime.now().isoformat()}"
        return hashlib.sha256(genesis_data.encode('utf-8')).hexdigest()

    def generate_hash(self, event_data: Dict[str, Any]) -> str:
        """为事件生成哈希，包含前一个哈希值。
        
        参数：
            event_data: 事件数据字典
            
        返回：
            str: 本次事件的哈希值
        """
        # 将事件数据（排除hash字段）序列化为JSON
        event_copy = {k: v for k, v in event_data.items() if k != 'hash'}
        serialized = json.dumps(event_copy, sort_keys=True, ensure_ascii=False, default=str)
        # 组合前一哈希 + 序列化数据
        combined = f"{self._previous_hash}:{serialized}"
        current_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()
        # 更新前一哈希
        self._previous_hash = current_hash
        self._chain_length += 1
        return current_hash

    def verify_chain(self, events: List[Dict[str, Any]]) -> Tuple[bool, List[int]]:
        """验证事件链的完整性。
        
        参数：
            events: 事件列表（按时间顺序）
            
        返回：
            Tuple[bool, List[int]]: (是否全部有效, 损坏的事件索引列表)
        """
        if not events:
            return True, []

        damaged_indices = []
        temp_chain = HashChain()

        for i, event in enumerate(events):
            event_copy = {k: v for k, v in event.items() if k != 'hash'}
            expected_hash = temp_chain.generate_hash(event_copy)
            actual_hash = event.get('hash')
            if actual_hash != expected_hash:
                damaged_indices.append(i)
                # 遇到损坏后重置链，继续检查后续
                temp_chain = HashChain()
                # 用当前事件重新开始链
                temp_chain.generate_hash(event_copy)

        return len(damaged_indices) == 0, damaged_indices

    @property
    def chain_length(self) -> int:
        return self._chain_length

    @property
    def last_hash(self) -> str:
        return self._previous_hash


# ==============================================================================
# 二、导入审计器（原有功能扩展）
# ==============================================================================

class ImportAuditor:
    """导入审计器

    包装 export_utils.AuditLogger，提供面向导入流程的便捷接口。
    当 db_manager 不可用时，自动回退到文件日志模式。

    用法：
        auditor = ImportAuditor(db_manager=db, log_dir='./logs')
        auditor.log_import_start('data.csv', 'structured_csv')
        auditor.log_column_mapping({'年龄': 'age'}, [])
        auditor.log_imputation('age', 'conditional_median', 5, 35)
        auditor.log_import_complete(100, [], [])
        auditor.export_audit_report('audit_report.json')
    """

    def __init__(self, db_manager=None, log_dir=None):
        """初始化审计器。

        参数：
            db_manager (object|None): 数据库管理器（需有 execute 方法）
            log_dir (str|None): 文件日志目录，None 时使用默认 './logs'
        """
        self.db_manager = db_manager
        self.log_dir = log_dir or os.path.join(os.getcwd(), 'logs')
        self._audit_logger = None
        self._batch_id = None
        self._events = []  # 内存中的事件缓存

        # 尝试初始化 AuditLogger
        self._init_audit_logger()

        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)

    def _init_audit_logger(self):
        """初始化底层 AuditLogger（可选依赖）。"""
        try:
            from ..export_utils import AuditLogger
            self._audit_logger = AuditLogger(
                db_manager=self.db_manager,
                log_file=os.path.join(self.log_dir, 'audit.log'),
            )
        except ImportError:
            LOGGER.debug("AuditLogger 不可用，使用文件日志回退")

    # ------------------------------------------------------------------
    # 导入生命周期
    # ------------------------------------------------------------------

    def log_import_start(self, source_file, input_type):
        """记录导入开始。

        参数：
            source_file (str): 数据来源文件路径
            input_type (str): 输入类型 ('structured_csv' / 'text' / 'excel' 等)
        """
        self._batch_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        event = {
            'event': 'import_start',
            'batch_id': self._batch_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'source_file': source_file,
            'input_type': input_type,
        }
        self._append_event(event)
        self._write_event(event)
        LOGGER.info("导入开始: %s (类型=%s, 批次=%s)", source_file, input_type, self._batch_id)

    def log_import_complete(self, total_records, warnings=None, errors=None):
        """记录导入完成。

        参数：
            total_records (int): 导入记录总数
            warnings (list[str]|None): 警告列表
            errors (list[str]|None): 错误列表
        """
        event = {
            'event': 'import_complete',
            'batch_id': self._batch_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'total_records': total_records,
            'warnings': warnings or [],
            'errors': errors or [],
        }
        self._append_event(event)
        self._write_event(event)

        # 记录到 AuditLogger
        if self._audit_logger:
            try:
                self._audit_logger.log_import(
                    source_file=self._get_source_file(),
                    record_count=total_records,
                    batch_id=self._batch_id,
                    notes=f"warnings={len(warnings or [])}, errors={len(errors or [])}",
                )
            except Exception as e:
                LOGGER.debug("AuditLogger 记录失败: %s", e)

        LOGGER.info("导入完成: %d 条记录 (批次=%s)", total_records, self._batch_id)

    # ------------------------------------------------------------------
    # 列名映射
    # ------------------------------------------------------------------

    def log_column_mapping(self, mappings, unmatched=None):
        """记录列名映射决策（原始列名→标准字段名，含置信度）。

        参数：
            mappings (dict): {原始列名: (标准字段名, 置信度, 来源)} 或 {原始列名: 标准字段名}
            unmatched (list[str]|None): 未匹配的列名列表
        """
        # 规范化 mappings
        normalized = {}
        for orig, mapped in (mappings or {}).items():
            if isinstance(mapped, tuple):
                normalized[orig] = {
                    'standard': mapped[0],
                    'confidence': mapped[1] if len(mapped) > 1 else 1.0,
                    'source': mapped[2] if len(mapped) > 2 else 'exact',
                }
            else:
                normalized[orig] = {
                    'standard': mapped,
                    'confidence': 1.0,
                    'source': 'exact',
                }

        event = {
            'event': 'column_mapping',
            'batch_id': self._batch_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'mappings': normalized,
            'unmatched': unmatched or [],
            'total_mapped': len(normalized),
            'total_unmatched': len(unmatched or []),
        }
        self._append_event(event)
        self._write_event(event)

        if unmatched:
            LOGGER.warning("未匹配的列: %s", unmatched)

    # ------------------------------------------------------------------
    # 缺失值填充
    # ------------------------------------------------------------------

    def log_imputation(self, column_name, strategy, filled_count, fill_value):
        """记录缺失值填充操作。

        参数：
            column_name (str): 列名
            strategy (str): 填充策略
            filled_count (int): 填充的记录数
            fill_value: 使用的填充值
        """
        event = {
            'event': 'imputation',
            'batch_id': self._batch_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'column_name': column_name,
            'strategy': strategy,
            'filled_count': filled_count,
            'fill_value': str(fill_value) if fill_value is not None else None,
        }
        self._append_event(event)
        self._write_event(event)

        if filled_count > 0:
            LOGGER.debug("缺失值填充: %s → %s (%d条, 值=%s)",
                         column_name, strategy, filled_count, fill_value)

    # ------------------------------------------------------------------
    # 文本抽取
    # ------------------------------------------------------------------

    def log_text_extraction(self, text_length, paragraphs, records_extracted):
        """记录文本抽取结果。

        参数：
            text_length (int): 输入文本长度
            paragraphs (int): 段落数
            records_extracted (int): 抽取到的记录数
        """
        event = {
            'event': 'text_extraction',
            'batch_id': self._batch_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'text_length': text_length,
            'paragraphs': paragraphs,
            'records_extracted': records_extracted,
        }
        self._append_event(event)
        self._write_event(event)

        LOGGER.info("文本抽取: %d 字符 → %d 段 → %d 条记录",
                    text_length, paragraphs, records_extracted)

    # ------------------------------------------------------------------
    # 质量评分
    # ------------------------------------------------------------------

    def log_quality_summary(self, scores, flagged_count):
        """记录质量评分汇总。

        参数：
            scores (dict): score_batch 返回的 summary 字典
            flagged_count (int): 标记为需人工复核的记录数
        """
        event = {
            'event': 'quality_summary',
            'batch_id': self._batch_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'summary': scores,
            'flagged_count': flagged_count,
        }
        self._append_event(event)
        self._write_event(event)

        mean_score = scores.get('mean_score', 0) if scores else 0
        LOGGER.info("质量评分: 均值=%.2f, 标记=%d 条",
                    mean_score, flagged_count)

    # ------------------------------------------------------------------
    # 导入历史
    # ------------------------------------------------------------------

    def get_import_history(self, limit=50):
        """获取导入历史。

        参数：
            limit (int): 返回的最大记录数

        返回：
            list[dict]: 导入事件列表
        """
        if self._audit_logger and self.db_manager:
            try:
                return self._audit_logger.get_import_history(limit=limit)
            except Exception as e:
                LOGGER.debug("获取导入历史失败: %s", e)

        # 回退：从文件日志读取
        return self._read_history_from_file(limit)

    def _read_history_from_file(self, limit=50):
        """从文件日志读取导入历史（扫描所有 audit_*.log 文件）。
        
        与 _write_event 写入路径（按日期分文件 audit_YYYYMMDD.log）保持一致，
        扫描所有匹配的文件并合并读取。

        参数：
            limit (int): 最大记录数

        返回：
            list[dict]
        """
        import glob as _glob
        history = []
        # 扫描所有 audit_*.log 文件，按修改时间排序
        pattern = os.path.join(self.log_dir, 'audit_*.log')
        log_files = sorted(_glob.glob(pattern), key=os.path.getmtime, reverse=True)
        
        for log_path in log_files:
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                history.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                if len(history) >= limit:
                    break
            except Exception as e:
                LOGGER.debug("读取审计历史失败: %s", e)

        return history[-limit:]

    # ------------------------------------------------------------------
    # 审计报告导出
    # ------------------------------------------------------------------

    def export_audit_report(self, output_path):
        """导出审计报告（JSON 或 HTML）。

        参数：
            output_path (str): 输出文件路径（.json 或 .html）

        返回：
            bool: 是否成功
        """
        ext = os.path.splitext(output_path)[1].lower()

        if ext == '.json':
            return self._export_json(output_path)
        elif ext == '.html':
            return self._export_html(output_path)
        else:
            LOGGER.warning("不支持的导出格式: %s，使用 JSON", ext)
            return self._export_json(output_path)

    def _export_json(self, output_path):
        """导出 JSON 格式审计报告。"""
        report = {
            'generated_at': datetime.datetime.now().isoformat(),
            'batch_id': self._batch_id,
            'total_events': len(self._events),
            'events': self._events,
        }
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            LOGGER.info("审计报告已导出: %s", output_path)
            return True
        except Exception as e:
            LOGGER.error("导出审计报告失败: %s", e)
            return False

    def _export_html(self, output_path):
        """导出 HTML 格式审计报告。"""
        # 统计汇总
        mapping_events = [e for e in self._events if e['event'] == 'column_mapping']
        imputation_events = [e for e in self._events if e['event'] == 'imputation']
        quality_events = [e for e in self._events if e['event'] == 'quality_summary']

        total_mapped = sum(e.get('total_mapped', 0) for e in mapping_events)
        total_unmatched = sum(e.get('total_unmatched', 0) for e in mapping_events)
        total_imputed = sum(e.get('filled_count', 0) for e in imputation_events)

        quality_summary = quality_events[-1].get('summary', {}) if quality_events else {}

        html_parts = ['<!DOCTYPE html>',
            '<html lang="zh-CN">',
            '<head><meta charset="UTF-8"><title>数据导入审计报告</title>',
            '<style>',
            '  body { font-family: "Microsoft YaHei", sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; color: #333; }',
            '  h1 { border-bottom: 2px solid #2196F3; padding-bottom: 10px; }',
            '  table { border-collapse: collapse; width: 100%; margin: 15px 0; }',
            '  th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }',
            '  th { background: #f5f5f5; }',
            '  .warn { color: #e67e22; } .error { color: #e74c3c; } .good { color: #27ae60; }',
            '</style></head><body>',
            '<h1>📋 数据导入审计报告</h1>',
            f'<p>生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>',
            f'<p>批次ID: {self._batch_id or "N/A"}</p>',
            '<h2>📊 总体概览</h2>',
            '<table><tr><th>指标</th><th>值</th></tr>',
            f'<tr><td>总事件数</td><td>{len(self._events)}</td></tr>',
            f'<tr><td>列名映射</td><td>成功 {total_mapped} / 未匹配 {total_unmatched}</td></tr>',
            f'<tr><td>缺失值填充</td><td>{total_imputed} 条记录</td></tr>',
            f'<tr><td>质量分均值</td><td>{quality_summary.get("mean_score", "N/A")}</td></tr>',
        ]
        flagged = quality_summary.get('flagged_count', 0)
        flagged_class = 'warn' if flagged > 0 else 'good'
        html_parts.append(f'<tr><td>需人工复核</td><td class="{flagged_class}">{flagged} 条</td></tr>')
        html_parts.append('</table>')
        html_parts.append('<h2>🔍 列名映射详情</h2>')
        html_parts.append('<table><tr><th>原始列名</th><th>标准字段</th><th>置信度</th><th>来源</th></tr>')

        for e in mapping_events:
            for orig, info in e.get('mappings', {}).items():
                html_parts.append(
                    f'<tr><td>{html.escape(str(orig))}</td>'
                    f'<td>{html.escape(str(info["standard"]))}</td>'
                    f'<td>{info["confidence"]:.2f}</td>'
                    f'<td>{html.escape(str(info["source"]))}</td></tr>'
                )

        html_parts.append('</table>')
        html_parts.append('<h2>📝 缺失值填充详情</h2>')
        html_parts.append('<table><tr><th>列名</th><th>策略</th><th>填充数</th><th>填充值</th></tr>')

        for e in imputation_events:
            html_parts.append(
                f'<tr><td>{html.escape(str(e["column_name"]))}</td>'
                f'<td>{html.escape(str(e["strategy"]))}</td>'
                f'<td>{e["filled_count"]}</td>'
                f'<td>{html.escape(str(e.get("fill_value", "")))}</td></tr>'
            )

        html_parts.append('</table></body></html>')
        html_content = '\n'.join(html_parts)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            LOGGER.info("审计报告已导出: %s", output_path)
            return True
        except Exception as e:
            LOGGER.error("导出审计报告失败: %s", e)
            return False

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _write_event(self, event):
        """将事件写入文件日志（含轮转检查）。

        参数：
            event (dict): 事件字典
        """
        log_path = self._get_log_path()
        try:
            # 检查是否需要轮转
            if os.path.exists(log_path) and os.path.getsize(log_path) > _MAX_LOG_SIZE:
                self._rotate_log(log_path)

            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        except Exception as e:
            LOGGER.debug("写入审计日志失败: %s", e)

    def _append_event(self, event):
        """添加事件到内存缓存（含上限检查）。

        参数：
            event (dict): 事件字典
        """
        self._events.append(event)
        # 超过上限时，将最早的事件写入磁盘并清空，保留最近 1000 条
        if len(self._events) > _MAX_EVENTS_MEMORY:
            self._flush_events_to_disk()
            self._events = self._events[-1000:]

    def _flush_events_to_disk(self):
        """仅清理内存缓存（事件已通过 _write_event 实时写入磁盘）。
        
        不再重复写入，因为每个事件在 _append_event 之前已通过 _write_event 
        写入按日期命名的文件（如 audit_20260712.log）。此处仅截断内存缓存。
        """
        # 事件已实时写入，仅清理内存缓存
        flushed_count = max(0, len(self._events) - 1000)
        self._events = self._events[-1000:]
        if flushed_count > 0:
            LOGGER.debug("已清理内存缓存 %d 条事件（已实时写入磁盘）", flushed_count)

    def _get_log_path(self):
        """获取当前日志文件路径（按日期分文件）。"""
        today = datetime.datetime.now().strftime('%Y%m%d')
        return os.path.join(self.log_dir, f'audit_{today}.log')

    def _rotate_log(self, log_path):
        """轮转日志文件：重命名为带时间戳的备份。"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        backup_path = log_path.replace('.log', f'_{timestamp}.log')
        try:
            os.rename(log_path, backup_path)
            LOGGER.info("审计日志已轮转: %s → %s", log_path, backup_path)
        except Exception as e:
            LOGGER.debug("日志轮转失败: %s", e)

    def _get_source_file(self):
        """从事件中提取源文件路径。"""
        for e in self._events:
            if e['event'] == 'import_start':
                return e.get('source_file', 'unknown')
        return 'unknown'


# ==============================================================================
# 三、临床操作审计器（新增，符合医疗法规要求）
# ==============================================================================

class ClinicalAuditor:
    """临床操作审计器
    
    记录所有对患者数据的访问、修改、上报、审核操作。
    使用哈希链确保日志不可篡改，保留期限15年。
    
    符合：
    - 《电子签名法》
    - 《医疗卫生机构网络安全管理办法》
    - 《医疗机构病历管理规定》
    - 《信息安全技术 健康医疗数据安全指南》
    
    用法：
        auditor = ClinicalAuditor(user_id='doctor_001', user_name='张医生',
                                  db_manager=db, log_dir='./audit_logs')
        auditor.log_operation(
            operation_type='UPDATE',
            patient_id='P20260001',
            patient_name='李某某',
            details={'field': 'diagnosis', 'old_value': '疑似', 'new_value': '确诊'},
            ip_address='192.168.1.100'
        )
    """

    def __init__(self, user_id: str = 'system', user_name: str = '系统',
                 user_role: str = 'system', db_manager=None, log_dir=None):
        """初始化临床审计器。
        
        参数：
            user_id: 当前用户ID
            user_name: 当前用户姓名
            user_role: 当前用户角色（doctor/director/phc_admin/system等）
            db_manager: 数据库管理器（可选）
            log_dir: 审计日志目录（默认 ./audit_logs/clinical）
        """
        self.user_id = user_id
        self.user_name = user_name
        self.user_role = user_role
        self.db_manager = db_manager
        self.log_dir = log_dir or os.path.join(os.getcwd(), 'logs', 'clinical')
        
        # 哈希链
        self._hash_chain = HashChain()
        self._events = []
        
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'archive'), exist_ok=True)
        
        # 确保数据库表存在
        self._ensure_tables()
        
        # 加载最后一个哈希值以续接链
        self._load_last_hash()

    def _ensure_tables(self):
        """确保临床审计表存在。"""
        if self.db_manager:
            try:
                self.db_manager.execute("""
                    CREATE TABLE IF NOT EXISTS audit_clinical (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT NOT NULL UNIQUE,
                        timestamp TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        user_name TEXT NOT NULL,
                        user_role TEXT,
                        operation_type TEXT NOT NULL,
                        operation_name TEXT NOT NULL,
                        patient_id TEXT,
                        patient_name TEXT,
                        record_id TEXT,
                        details TEXT,
                        old_value TEXT,
                        new_value TEXT,
                        ip_address TEXT,
                        user_agent TEXT,
                        reason TEXT,
                        hash TEXT NOT NULL,
                        previous_hash TEXT NOT NULL,
                        created_at TEXT DEFAULT (datetime('now'))
                    )
                """)
                self.db_manager.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audit_clinical_patient 
                    ON audit_clinical(patient_id)
                """)
                self.db_manager.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audit_clinical_user 
                    ON audit_clinical(user_id)
                """)
                self.db_manager.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audit_clinical_time 
                    ON audit_clinical(timestamp)
                """)
                self.db_manager.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audit_clinical_op 
                    ON audit_clinical(operation_type)
                """)
            except Exception as e:
                LOGGER.warning("临床审计表创建失败: %s", e)

    def _load_last_hash(self):
        """从数据库或文件加载最后一个哈希值以续接哈希链。"""
        # 优先从数据库加载
        if self.db_manager:
            try:
                result = self.db_manager.execute(
                    "SELECT hash FROM audit_clinical ORDER BY id DESC LIMIT 1"
                )
                if result:
                    last_hash = result[0][0] if isinstance(result[0], (tuple, list)) else result[0]['hash']
                    self._hash_chain = HashChain(seed=last_hash)
                    return
            except Exception as e:
                LOGGER.debug("从数据库恢复hash链失败: %s", e)
        
        # 从文件加载
        log_path = self._get_log_path()
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    last_line = None
                    for line in f:
                        line = line.strip()
                        if line:
                            last_line = line
                    if last_line:
                        last_event = json.loads(last_line)
                        if 'hash' in last_event:
                            self._hash_chain = HashChain(seed=last_event['hash'])
            except Exception as e:
                LOGGER.debug("从文件恢复hash链失败: %s", e)

    # ------------------------------------------------------------------
    # 核心日志记录方法
    # ------------------------------------------------------------------

    def log_operation(self, operation_type: str, patient_id: Optional[str] = None,
                      patient_name: Optional[str] = None, record_id: Optional[str] = None,
                      details: Optional[Dict[str, Any]] = None,
                      old_value: Optional[Any] = None, new_value: Optional[Any] = None,
                      ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                      reason: Optional[str] = None) -> str:
        """记录一次临床操作。
        
        参数：
            operation_type: 操作类型（见 OPERATION_TYPES 常量）
            patient_id: 患者ID
            patient_name: 患者姓名（脱敏存储）
            record_id: 记录ID（报告卡ID等）
            details: 操作详情字典
            old_value: 修改前的值（UPDATE操作必填）
            new_value: 修改后的值（UPDATE/APPROVE等操作必填）
            ip_address: 操作IP地址
            user_agent: 客户端标识
            reason: 操作原因/审核意见
            
        返回：
            str: 事件ID
        """
        event_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        timestamp = datetime.datetime.now().isoformat()
        
        # 操作名称
        operation_name = OPERATION_TYPES.get(operation_type, operation_type)
        
        # 患者姓名脱敏
        safe_patient_name = self._mask_name(patient_name) if patient_name else None
        
        # 构建事件
        event = {
            'event': 'clinical_operation',
            'event_id': event_id,
            'timestamp': timestamp,
            'user_id': self.user_id,
            'user_name': self.user_name,
            'user_role': self.user_role,
            'operation_type': operation_type,
            'operation_name': operation_name,
            'patient_id': patient_id,
            'patient_name': safe_patient_name,
            'record_id': record_id,
            'details': details,
            'old_value': old_value,
            'new_value': new_value,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'reason': reason,
            'previous_hash': self._hash_chain.last_hash,
        }
        
        # 生成哈希
        event['hash'] = self._hash_chain.generate_hash(event)
        
        # 添加到内存
        self._events.append(event)
        
        # 写入数据库
        self._write_to_db(event)
        
        # 写入文件
        self._write_to_file(event)
        
        LOGGER.info(
            "临床操作: %s - 患者=%s, 操作人=%s(%s), 事件ID=%s",
            operation_name, patient_id or 'N/A', self.user_name, self.user_id, event_id
        )
        
        return event_id

    def log_create(self, patient_id: str, patient_name: str, record_id: str,
                   record_data: Dict[str, Any], ip_address: str = None,
                   reason: str = None) -> str:
        """记录创建操作（快捷方法）。"""
        return self.log_operation(
            operation_type='CREATE',
            patient_id=patient_id,
            patient_name=patient_name,
            record_id=record_id,
            new_value=record_data,
            ip_address=ip_address,
            reason=reason
        )

    def log_read(self, patient_id: str, patient_name: str = None,
                 record_id: str = None, ip_address: str = None,
                 reason: str = None) -> str:
        """记录查看操作（快捷方法）。"""
        return self.log_operation(
            operation_type='READ',
            patient_id=patient_id,
            patient_name=patient_name,
            record_id=record_id,
            ip_address=ip_address,
            reason=reason or '临床诊疗需要'
        )

    def log_update(self, patient_id: str, patient_name: str, record_id: str,
                   updated_fields: Dict[str, Any], old_value: Dict[str, Any],
                   new_value: Dict[str, Any], ip_address: str = None,
                   reason: str = None) -> str:
        """记录修改操作（快捷方法）。"""
        return self.log_operation(
            operation_type='UPDATE',
            patient_id=patient_id,
            patient_name=patient_name,
            record_id=record_id,
            details={'updated_fields': list(updated_fields.keys())},
            old_value=old_value,
            new_value=new_value,
            ip_address=ip_address,
            reason=reason
        )

    def log_submit(self, patient_id: str, patient_name: str, record_id: str,
                   submit_data: Dict[str, Any], ip_address: str = None,
                   reason: str = None) -> str:
        """记录提交审核操作（快捷方法）。"""
        return self.log_operation(
            operation_type='SUBMIT',
            patient_id=patient_id,
            patient_name=patient_name,
            record_id=record_id,
            new_value=submit_data,
            ip_address=ip_address,
            reason=reason or '提交上级审核'
        )

    def log_approve(self, patient_id: str, patient_name: str, record_id: str,
                    approved_data: Dict[str, Any], ip_address: str = None,
                    comments: str = None) -> str:
        """记录审核通过操作（快捷方法）。"""
        return self.log_operation(
            operation_type='APPROVE',
            patient_id=patient_id,
            patient_name=patient_name,
            record_id=record_id,
            new_value=approved_data,
            ip_address=ip_address,
            reason=comments or '审核通过'
        )

    def log_reject(self, patient_id: str, patient_name: str, record_id: str,
                   reject_reason: str, current_data: Dict[str, Any],
                   ip_address: str = None) -> str:
        """记录审核驳回操作（快捷方法）。"""
        return self.log_operation(
            operation_type='REJECT',
            patient_id=patient_id,
            patient_name=patient_name,
            record_id=record_id,
            old_value=current_data,
            ip_address=ip_address,
            reason=reject_reason
        )

    def log_report(self, patient_id: str, patient_name: str, record_id: str,
                   report_data: Dict[str, Any], report_type: str = 'cdc',
                   ip_address: str = None) -> str:
        """记录上报疾控操作（快捷方法）。"""
        return self.log_operation(
            operation_type='REPORT',
            patient_id=patient_id,
            patient_name=patient_name,
            record_id=record_id,
            details={'report_type': report_type},
            new_value=report_data,
            ip_address=ip_address,
            reason='传染病疫情报告'
        )

    def log_assessment(self, patient_id: str, patient_name: str, record_id: str,
                       risk_score: float, risk_level: str,
                       features: Dict[str, Any] = None,
                       ip_address: str = None) -> str:
        """记录风险评估操作（快捷方法）。"""
        return self.log_operation(
            operation_type='ASSESS',
            patient_id=patient_id,
            patient_name=patient_name,
            record_id=record_id,
            details={
                'risk_score': risk_score,
                'risk_level': risk_level,
                'features_used': list(features.keys()) if features else []
            },
            new_value={'risk_score': risk_score, 'risk_level': risk_level},
            ip_address=ip_address,
            reason='结核病风险评估'
        )

    # ------------------------------------------------------------------
    # 查询方法
    # ------------------------------------------------------------------

    def get_patient_audit_trail(self, patient_id: str, 
                                start_time: str = None,
                                end_time: str = None) -> List[Dict[str, Any]]:
        """获取单个患者的审计追踪记录。
        
        参数：
            patient_id: 患者ID
            start_time: 开始时间（ISO格式）
            end_time: 结束时间（ISO格式）
            
        返回：
            List[Dict]: 事件列表（按时间排序）
        """
        events = []
        
        if self.db_manager:
            try:
                query = """
                    SELECT event_id, timestamp, user_id, user_name, user_role,
                           operation_type, operation_name, patient_id, record_id,
                           details, old_value, new_value, ip_address, reason, hash
                    FROM audit_clinical 
                    WHERE patient_id = ?
                """
                params = [patient_id]
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                query += " ORDER BY timestamp ASC"
                
                rows = self.db_manager.execute(query, params)
                for row in rows:
                    if isinstance(row, (tuple, list)):
                        event = {
                            'event_id': row[0],
                            'timestamp': row[1],
                            'user_id': row[2],
                            'user_name': row[3],
                            'user_role': row[4],
                            'operation_type': row[5],
                            'operation_name': row[6],
                            'patient_id': row[7],
                            'record_id': row[8],
                            'details': json.loads(row[9]) if row[9] else None,
                            'old_value': json.loads(row[10]) if row[10] else None,
                            'new_value': json.loads(row[11]) if row[11] else None,
                            'ip_address': row[12],
                            'reason': row[13],
                            'hash': row[14],
                        }
                    else:
                        event = dict(row)
                        if event.get('details'):
                            event['details'] = json.loads(event['details'])
                        if event.get('old_value'):
                            event['old_value'] = json.loads(event['old_value'])
                        if event.get('new_value'):
                            event['new_value'] = json.loads(event['new_value'])
                    events.append(event)
                return events
            except Exception as e:
                LOGGER.warning("从数据库查询审计记录失败: %s", e)
        
        # 回退：从文件读取
        return self._read_events_from_file(patient_id=patient_id, 
                                           start_time=start_time,
                                           end_time=end_time)

    def get_user_operations(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取指定用户的操作记录。"""
        events = []
        if self.db_manager:
            try:
                rows = self.db_manager.execute("""
                    SELECT event_id, timestamp, operation_type, operation_name, 
                           patient_id, record_id, details, ip_address, reason
                    FROM audit_clinical 
                    WHERE user_id = ?
                    ORDER BY timestamp DESC LIMIT ?
                """, [user_id, limit])
                for row in rows:
                    if isinstance(row, (tuple, list)):
                        events.append({
                            'event_id': row[0],
                            'timestamp': row[1],
                            'operation_type': row[2],
                            'operation_name': row[3],
                            'patient_id': row[4],
                            'record_id': row[5],
                            'details': json.loads(row[6]) if row[6] else None,
                            'ip_address': row[7],
                            'reason': row[8],
                        })
                    else:
                        events.append(dict(row))
                return events
            except Exception as e:
                LOGGER.warning("查询用户操作记录失败: %s", e)
        return events

    # ------------------------------------------------------------------
    # 完整性验证
    # ------------------------------------------------------------------

    def verify_integrity(self, patient_id: str = None) -> Dict[str, Any]:
        """验证审计日志完整性。
        
        参数：
            patient_id: 如果指定，只验证该患者的日志；否则验证全部
            
        返回：
            Dict: 验证结果 {
                'valid': bool,
                'total_events': int,
                'damaged_count': int,
                'damaged_indices': List[int],
                'message': str
            }
        """
        if patient_id:
            events = self.get_patient_audit_trail(patient_id)
        else:
            events = self._read_all_events_from_db()
        
        if not events:
            return {
                'valid': True,
                'total_events': 0,
                'damaged_count': 0,
                'damaged_indices': [],
                'message': '无审计记录'
            }
        
        temp_chain = HashChain()
        damaged_indices = []
        
        for i, event in enumerate(events):
            event_copy = {k: v for k, v in event.items() if k != 'hash'}
            expected_hash = temp_chain.generate_hash(event_copy)
            if event.get('hash') != expected_hash:
                damaged_indices.append(i)
                # 重置链继续检查
                temp_chain = HashChain()
                temp_chain.generate_hash(event_copy)
        
        return {
            'valid': len(damaged_indices) == 0,
            'total_events': len(events),
            'damaged_count': len(damaged_indices),
            'damaged_indices': damaged_indices,
            'message': '审计日志完整有效' if len(damaged_indices) == 0 
                       else f'发现 {len(damaged_indices)} 处日志被篡改'
        }

    # ------------------------------------------------------------------
    # 审计报告导出
    # ------------------------------------------------------------------

    def export_audit_report(self, output_path: str, patient_id: str = None,
                            start_time: str = None, end_time: str = None) -> bool:
        """导出临床审计报告。
        
        参数：
            output_path: 输出文件路径（.json 或 .html）
            patient_id: 患者ID（可选，不指定则导出全部）
            start_time: 开始时间
            end_time: 结束时间
            
        返回：
            bool: 是否成功
        """
        if patient_id:
            events = self.get_patient_audit_trail(patient_id, start_time, end_time)
            title = f'患者 {patient_id} 审计追踪报告'
        else:
            events = self._read_all_events_from_db()
            if start_time or end_time:
                events = [e for e in events 
                         if (not start_time or e['timestamp'] >= start_time) and
                            (not end_time or e['timestamp'] <= end_time)]
            title = '临床操作审计报告'
        
        ext = os.path.splitext(output_path)[1].lower()
        
        if ext == '.json':
            return self._export_clinical_json(output_path, events, title)
        elif ext == '.html':
            return self._export_clinical_html(output_path, events, title)
        else:
            return self._export_clinical_json(output_path, events, title)

    def _export_clinical_json(self, output_path, events, title):
        """导出JSON格式临床审计报告。"""
        # 验证完整性
        integrity = self.verify_integrity()
        
        report = {
            'report_title': title,
            'generated_at': datetime.datetime.now().isoformat(),
            'generated_by': f'{self.user_name}({self.user_id})',
            'retention_period': f'{_RETENTION_YEARS}年',
            'integrity_check': integrity,
            'total_events': len(events),
            'events': events,
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            LOGGER.info("临床审计报告已导出: %s", output_path)
            return True
        except Exception as e:
            LOGGER.error("导出临床审计报告失败: %s", e)
            return False

    def _export_clinical_html(self, output_path, events, title):
        """导出HTML格式临床审计报告。"""
        integrity = self.verify_integrity()
        
        # 统计
        op_counts = {}
        for e in events:
            op = e.get('operation_name', '未知')
            op_counts[op] = op_counts.get(op, 0) + 1
        
        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="zh-CN">',
            '<head><meta charset="UTF-8">',
            f'<title>{html.escape(title)}</title>',
            '<style>',
            '  body { font-family: "Microsoft YaHei", sans-serif; max-width: 1200px; margin: 40px auto; padding: 20px; color: #333; }',
            '  h1 { border-bottom: 2px solid #c0392b; padding-bottom: 10px; color: #c0392b; }',
            '  h2 { color: #2c3e50; margin-top: 30px; }',
            '  table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }',
            '  th, td { border: 1px solid #ddd; padding: 8px 10px; text-align: left; }',
            '  th { background: #f8f9fa; font-weight: 600; }',
            '  tr:nth-child(even) { background: #fcfcfc; }',
            '  .integrity-valid { color: #27ae60; font-weight: bold; }',
            '  .integrity-damaged { color: #e74c3c; font-weight: bold; }',
            '  .op-CREATE { background: #d4edda; }',
            '  .op-UPDATE { background: #fff3cd; }',
            '  .op-DELETE { background: #f8d7da; }',
            '  .op-APPROVE { background: #d1ecf1; }',
            '  .op-REJECT { background: #f5c6cb; }',
            '  .op-REPORT { background: #e2e3e5; }',
            '  .disclaimer { margin-top: 40px; padding: 15px; background: #f8f9fa; border-left: 4px solid #6c757d; font-size: 12px; color: #6c757d; }',
            '</style></head><body>',
            f'<h1>🔒 {html.escape(title)}</h1>',
            f'<p>生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>',
            f'<p>生成人: {html.escape(self.user_name)} ({html.escape(self.user_id)})</p>',
            f'<p>保留期限: {_RETENTION_YEARS}年（符合《医疗卫生机构网络安全管理办法》要求）</p>',
            '<h2>📊 完整性校验</h2>',
            '<table><tr><th>项目</th><th>结果</th></tr>',
            f'<tr><td>总事件数</td><td>{len(events)}</td></tr>',
        ]
        
        integrity_class = 'integrity-valid' if integrity['valid'] else 'integrity-damaged'
        html_parts.append(
            f'<tr><td>完整性状态</td><td class="{integrity_class}">{html.escape(integrity["message"])}</td></tr>'
        )
        html_parts.append('</table>')
        
        html_parts.append('<h2>📈 操作类型统计</h2>')
        html_parts.append('<table><tr><th>操作类型</th><th>次数</th></tr>')
        for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
            html_parts.append(f'<tr><td>{html.escape(op)}</td><td>{count}</td></tr>')
        html_parts.append('</table>')
        
        html_parts.append('<h2>📋 详细操作记录</h2>')
        html_parts.append('<table><tr><th>时间</th><th>操作人</th><th>角色</th><th>操作类型</th>'
                          '<th>患者ID</th><th>记录ID</th><th>IP地址</th><th>原因/备注</th></tr>')
        
        for e in events:
            op_type = e.get('operation_type', '')
            op_class = f'op-{op_type}' if op_type in ['CREATE', 'UPDATE', 'DELETE', 'APPROVE', 'REJECT', 'REPORT'] else ''
            html_parts.append(
                f'<tr class="{op_class}">'
                f'<td>{html.escape(str(e.get("timestamp", "")))}</td>'
                f'<td>{html.escape(str(e.get("user_name", "")))}<br><small>{html.escape(str(e.get("user_id", "")))}</small></td>'
                f'<td>{html.escape(str(e.get("user_role", "")))}</td>'
                f'<td>{html.escape(str(e.get("operation_name", "")))}</td>'
                f'<td>{html.escape(str(e.get("patient_id", "")))}</td>'
                f'<td>{html.escape(str(e.get("record_id", "")))}</td>'
                f'<td>{html.escape(str(e.get("ip_address", "")))}</td>'
                f'<td>{html.escape(str(e.get("reason", "")[:100] if e.get("reason") else ""))}</td>'
                f'</tr>'
            )
        
        html_parts.append('</table>')
        
        html_parts.append(
            '<div class="disclaimer">'
            '<strong>免责声明：</strong>本审计报告由tb_risk结核病风险评估系统自动生成，'
            '包含的所有操作记录均通过SHA-256哈希链保护，任何篡改均可被检测。'
            '审计日志保留期限为15年，符合《中华人民共和国电子签名法》、《医疗卫生机构网络安全管理办法》'
            '及《医疗机构病历管理规定》的相关要求。'
            '</div>'
        )
        
        html_parts.append('</body></html>')
        html_content = '\n'.join(html_parts)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            LOGGER.info("临床审计报告已导出: %s", output_path)
            return True
        except Exception as e:
            LOGGER.error("导出临床审计报告失败: %s", e)
            return False

    # ------------------------------------------------------------------
    # 归档与清理
    # ------------------------------------------------------------------

    def archive_old_logs(self, days_to_keep: int = 365):
        """归档旧日志（超过指定天数的移动到archive目录）。
        
        参数：
            days_to_keep: 保留最近多少天的日志在主目录
        """
        import glob
        cutoff = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
        pattern = os.path.join(self.log_dir, 'clinical_*.log')
        
        for log_path in glob.glob(pattern):
            try:
                filename = os.path.basename(log_path)
                # 从文件名提取日期
                date_str = filename.replace('clinical_', '').replace('.log', '')
                log_date = datetime.datetime.strptime(date_str, '%Y%m%d')
                
                if log_date < cutoff:
                    archive_path = os.path.join(self.log_dir, 'archive', filename)
                    shutil.move(log_path, archive_path)
                    LOGGER.info("已归档审计日志: %s", filename)
            except Exception as e:
                LOGGER.debug("归档日志失败 %s: %s", log_path, e)

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    def _mask_name(self, name: str) -> str:
        """姓名脱敏：保留姓，名用*代替。"""
        if not name:
            return name
        if len(name) <= 1:
            return name
        if len(name) == 2:
            return name[0] + '*'
        return name[0] + '*' * (len(name) - 2) + name[-1]

    def _serialize_value(self, value: Any) -> Optional[str]:
        """序列化值为JSON字符串。"""
        if value is None:
            return None
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)

    def _write_to_db(self, event: Dict[str, Any]):
        """写入数据库。"""
        if not self.db_manager:
            return
        try:
            self.db_manager.execute("""
                INSERT INTO audit_clinical
                (event_id, timestamp, user_id, user_name, user_role,
                 operation_type, operation_name, patient_id, patient_name,
                 record_id, details, old_value, new_value, ip_address,
                 user_agent, reason, hash, previous_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                event['event_id'],
                event['timestamp'],
                event['user_id'],
                event['user_name'],
                event['user_role'],
                event['operation_type'],
                event['operation_name'],
                event['patient_id'],
                event['patient_name'],
                event['record_id'],
                self._serialize_value(event.get('details')),
                self._serialize_value(event.get('old_value')),
                self._serialize_value(event.get('new_value')),
                event.get('ip_address'),
                event.get('user_agent'),
                event.get('reason'),
                event['hash'],
                event['previous_hash'],
            ])
        except Exception as e:
            LOGGER.warning("写入临床审计数据库失败: %s", e)

    def _write_to_file(self, event: Dict[str, Any]):
        """写入文件日志。"""
        log_path = self._get_log_path()
        try:
            # 检查轮转
            if os.path.exists(log_path) and os.path.getsize(log_path) > _MAX_LOG_SIZE:
                self._rotate_log(log_path)
            
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + '\n')
        except Exception as e:
            LOGGER.debug("写入临床审计文件失败: %s", e)

    def _get_log_path(self):
        """获取当前临床审计日志文件路径。"""
        today = datetime.datetime.now().strftime('%Y%m%d')
        return os.path.join(self.log_dir, f'clinical_{today}.log')

    def _rotate_log(self, log_path):
        """轮转日志文件。"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        backup_path = log_path.replace('.log', f'_{timestamp}.log')
        try:
            os.rename(log_path, backup_path)
        except Exception as e:
            LOGGER.debug("临床审计日志轮转失败: %s", e)

    def _read_events_from_file(self, patient_id=None, start_time=None, end_time=None):
        """从文件读取事件。"""
        import glob
        events = []
        pattern = os.path.join(self.log_dir, 'clinical_*.log')
        
        for log_path in sorted(glob.glob(pattern)):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                            # 过滤条件
                            if patient_id and event.get('patient_id') != patient_id:
                                continue
                            if start_time and event.get('timestamp', '') < start_time:
                                continue
                            if end_time and event.get('timestamp', '') > end_time:
                                continue
                            events.append(event)
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                LOGGER.debug("从文件读取审计事件失败: %s", e)
        
        return sorted(events, key=lambda x: x.get('timestamp', ''))

    def _read_all_events_from_db(self):
        """从数据库读取所有事件。"""
        events = []
        if self.db_manager:
            try:
                rows = self.db_manager.execute("""
                    SELECT event_id, timestamp, user_id, user_name, user_role,
                           operation_type, operation_name, patient_id, patient_name,
                           record_id, details, old_value, new_value, ip_address,
                           reason, hash, previous_hash
                    FROM audit_clinical ORDER BY timestamp ASC
                """)
                for row in rows:
                    if isinstance(row, (tuple, list)):
                        event = {
                            'event_id': row[0],
                            'timestamp': row[1],
                            'user_id': row[2],
                            'user_name': row[3],
                            'user_role': row[4],
                            'operation_type': row[5],
                            'operation_name': row[6],
                            'patient_id': row[7],
                            'patient_name': row[8],
                            'record_id': row[9],
                            'details': json.loads(row[10]) if row[10] else None,
                            'old_value': json.loads(row[11]) if row[11] else None,
                            'new_value': json.loads(row[12]) if row[12] else None,
                            'ip_address': row[13],
                            'reason': row[14],
                            'hash': row[15],
                            'previous_hash': row[16],
                        }
                    else:
                        event = dict(row)
                        if event.get('details'):
                            event['details'] = json.loads(event['details'])
                        if event.get('old_value'):
                            event['old_value'] = json.loads(event['old_value'])
                        if event.get('new_value'):
                            event['new_value'] = json.loads(event['new_value'])
                    events.append(event)
            except Exception as e:
                LOGGER.warning("读取所有临床审计记录失败: %s", e)
        
        return events


# ==============================================================================
# 四、全局审计器工厂
# ==============================================================================

_audit_instances: Dict[str, Any] = {}


def get_import_auditor(db_manager=None, log_dir=None) -> ImportAuditor:
    """获取导入审计器单例。"""
    key = f'import_{id(db_manager)}_{log_dir}'
    if key not in _audit_instances:
        _audit_instances[key] = ImportAuditor(db_manager, log_dir)
    return _audit_instances[key]


def get_clinical_auditor(user_id: str = 'system', user_name: str = '系统',
                         user_role: str = 'system',
                         db_manager=None, log_dir=None) -> ClinicalAuditor:
    """获取临床审计器实例。"""
    return ClinicalAuditor(user_id, user_name, user_role, db_manager, log_dir)


def reset_audit_instances():
    """重置所有审计器实例（主要用于测试）。"""
    _audit_instances.clear()
