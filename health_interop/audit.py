#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计日志模块

提供医疗系统合规要求的审计追踪功能：
- 操作记录持久化（SQLite，只允许INSERT）
- 哈希链完整性校验（防止篡改）
- 按条件查询和导出
- 兼容workflow和contact_tracing的auditor接口
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("tb_risk.audit")

# 审计操作类型常量
class AuditOperationType:
    CREATE = "CREATE"
    READ = "READ"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    SUBMIT = "SUBMIT"
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    RETURN = "RETURN"
    REPORT = "REPORT"
    CANCEL = "CANCEL"
    ASSESS = "ASSESS"
    EXPORT = "EXPORT"
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"
    IMPORT = "IMPORT"
    SYNC = "SYNC"


class AuditRecord:
    """审计记录数据结构"""
    __slots__ = (
        "record_id", "timestamp", "user_id", "user_name", "user_role",
        "operation_type", "patient_id", "patient_name",
        "record_type", "target_record_id",
        "old_value", "new_value", "details",
        "ip_address", "reason",
        "prev_hash", "record_hash"
    )
    
    def __init__(self, **kwargs):
        for k in self.__slots__:
            setattr(self, k, kwargs.get(k))
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__ if getattr(self, k) is not None}


class AuditLogger:
    """医疗审计日志器
    
    功能：
    - SQLite持久化，审计表只允许INSERT，禁止UPDATE/DELETE
    - 哈希链机制：每条记录包含前一条记录的哈希，防止插入/删除/篡改
    - 线程安全
    """
    
    def __init__(self, db_path: str = "audit.db", auto_flush: bool = True):
        self.db_path = db_path
        self.auto_flush = auto_flush
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._last_hash: Optional[str] = None
        self._buffer: List[AuditRecord] = []
        self._ensure_db()
        self._load_last_hash()
    
    def _ensure_db(self):
        """确保数据库表存在（审计表只允许INSERT）"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                user_id TEXT DEFAULT '',
                user_name TEXT DEFAULT '',
                user_role TEXT DEFAULT '',
                operation_type TEXT NOT NULL,
                patient_id TEXT DEFAULT '',
                patient_name TEXT DEFAULT '',
                record_type TEXT DEFAULT '',
                target_record_id TEXT DEFAULT '',
                old_value TEXT DEFAULT '',
                new_value TEXT DEFAULT '',
                details TEXT DEFAULT '',
                ip_address TEXT DEFAULT '',
                reason TEXT DEFAULT '',
                prev_hash TEXT DEFAULT '',
                record_hash TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)
        
        # 根哈希快照表（用于离线保存，检测整体数据库篡改）
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_root_hashes (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_time REAL NOT NULL,
                snapshot_date TEXT NOT NULL,
                last_record_id INTEGER NOT NULL,
                record_count INTEGER NOT NULL,
                root_hash TEXT NOT NULL,
                signature TEXT DEFAULT '',
                notes TEXT DEFAULT '',
                exported INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)
        
        # 创建索引加速查询
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
            ON audit_log(timestamp)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_patient 
            ON audit_log(patient_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_user 
            ON audit_log(user_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_op 
            ON audit_log(operation_type)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_root_hash_date 
            ON audit_root_hashes(snapshot_date)
        """)
        self._conn.commit()
    
    def _load_last_hash(self):
        """加载最后一条记录的哈希用于链校验"""
        try:
            cur = self._conn.execute(
                "SELECT record_hash FROM audit_log ORDER BY record_id DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                self._last_hash = row[0]
        except Exception as e:
            LOGGER.debug("加载最后哈希失败: %s", e)
            self._last_hash = None
    
    def _compute_hash(self, record_dict: Dict[str, Any], prev_hash: str) -> str:
        """计算记录哈希，包含前一条哈希形成链"""
        payload = json.dumps({
            "timestamp": record_dict.get("timestamp", 0),
            "user_id": record_dict.get("user_id", ""),
            "operation_type": record_dict.get("operation_type", ""),
            "patient_id": record_dict.get("patient_id", ""),
            "target_record_id": record_dict.get("target_record_id", ""),
            "details": record_dict.get("details", ""),
            "prev_hash": prev_hash or "",
        }, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    
    def log_operation(self,
                      operation_type: str,
                      user_id: str = "",
                      user_name: str = "",
                      user_role: str = "",
                      patient_id: str = "",
                      patient_name: str = "",
                      record_id: str = "",
                      record_type: str = "",
                      details: Any = None,
                      old_value: Any = None,
                      new_value: Any = None,
                      ip_address: str = "",
                      reason: str = "") -> bool:
        """记录审计操作（与workflow/contact_tracing接口兼容）
        
        参数：
            operation_type: 操作类型（CREATE/UPDATE/APPROVE等）
            user_id: 操作人ID
            user_name: 操作人姓名
            user_role: 操作人角色
            patient_id: 患者ID
            patient_name: 患者姓名
            record_id: 目标记录ID
            record_type: 目标记录类型
            details: 操作详情
            old_value: 变更前值
            new_value: 变更后值
            ip_address: 操作IP
            reason: 操作原因
        """
        with self._lock:
            now = time.time()
            
            record_dict = {
                "timestamp": now,
                "user_id": user_id or "",
                "user_name": user_name or "",
                "user_role": user_role or "",
                "operation_type": operation_type,
                "patient_id": patient_id or "",
                "patient_name": patient_name or "",
                "record_type": record_type or "",
                "target_record_id": str(record_id) if record_id else "",
                "old_value": json.dumps(old_value, ensure_ascii=False, default=str) if old_value is not None else "",
                "new_value": json.dumps(new_value, ensure_ascii=False, default=str) if new_value is not None else "",
                "details": json.dumps(details, ensure_ascii=False, default=str) if details is not None else "",
                "ip_address": ip_address or "",
                "reason": reason or "",
                "prev_hash": self._last_hash or "",
            }
            
            record_dict["record_hash"] = self._compute_hash(record_dict, self._last_hash)
            
            record = AuditRecord(**record_dict)
            self._buffer.append(record)
            
            if self.auto_flush:
                self.flush()
            
            return True
    
    def flush(self) -> int:
        """将缓冲区的记录写入数据库"""
        with self._lock:
            if not self._buffer:
                return 0
            
            count = 0
            for rec in self._buffer:
                try:
                    self._conn.execute("""
                        INSERT INTO audit_log (
                            timestamp, user_id, user_name, user_role,
                            operation_type, patient_id, patient_name,
                            record_type, target_record_id,
                            old_value, new_value, details,
                            ip_address, reason, prev_hash, record_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        rec.timestamp, rec.user_id, rec.user_name, rec.user_role,
                        rec.operation_type, rec.patient_id, rec.patient_name,
                        rec.record_type, rec.target_record_id,
                        rec.old_value, rec.new_value, rec.details,
                        rec.ip_address, rec.reason, rec.prev_hash, rec.record_hash
                    ))
                    self._last_hash = rec.record_hash
                    count += 1
                except Exception as e:
                    LOGGER.error("审计记录写入失败: %s", e)
            
            self._conn.commit()
            self._buffer.clear()
            return count
    
    def query(self,
              start_time: float = None,
              end_time: float = None,
              user_id: str = None,
              patient_id: str = None,
              operation_type: str = None,
              record_id: str = None,
              limit: int = 500) -> List[AuditRecord]:
        """按条件查询审计日志"""
        with self._lock:
            self.flush()
            
            sql = "SELECT * FROM audit_log WHERE 1=1"
            params = []
            
            if start_time is not None:
                sql += " AND timestamp >= ?"
                params.append(start_time)
            if end_time is not None:
                sql += " AND timestamp <= ?"
                params.append(end_time)
            if user_id:
                sql += " AND user_id = ?"
                params.append(user_id)
            if patient_id:
                sql += " AND patient_id = ?"
                params.append(patient_id)
            if operation_type:
                sql += " AND operation_type = ?"
                params.append(operation_type)
            if record_id:
                sql += " AND target_record_id = ?"
                params.append(str(record_id))
            
            sql += " ORDER BY record_id DESC LIMIT ?"
            params.append(limit)
            
            cur = self._conn.execute(sql, params)
            columns = [desc[0] for desc in cur.description]
            results = []
            for row in cur.fetchall():
                data = dict(zip(columns, row))
                results.append(AuditRecord(**data))
            return results
    
    def verify_integrity(self) -> Dict[str, Any]:
        """验证哈希链完整性，检查是否有篡改
        
        返回：
            {"valid": bool, "errors": [...], "total": int, "checked": int}
        """
        with self._lock:
            self.flush()
            
            cur = self._conn.execute(
                "SELECT * FROM audit_log ORDER BY record_id ASC"
            )
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            
            errors = []
            prev_hash = None
            checked = 0
            
            for i, row in enumerate(rows):
                data = dict(zip(columns, row))
                stored_hash = data["record_hash"]
                stored_prev = data["prev_hash"]
                
                # 验证prev_hash链接
                if stored_prev != (prev_hash or ""):
                    errors.append(f"记录#{data['record_id']} prev_hash不匹配，可能被删除或插入")
                
                # 重新计算当前哈希
                computed = self._compute_hash(data, prev_hash)
                if computed != stored_hash:
                    errors.append(f"记录#{data['record_id']} hash不匹配，内容可能被篡改")
                
                prev_hash = stored_hash
                checked += 1
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "total": len(rows),
                "checked": checked,
            }
    
    def compute_root_hash(self, up_to_record_id: int = None) -> Dict[str, Any]:
        """计算审计链的根哈希（Merkle根风格，对最后N条记录的哈希链求根）
        
        参数：
            up_to_record_id: 计算到指定记录ID为止的根哈希，None表示到最新记录
        返回：
            {"root_hash": str, "last_record_id": int, "record_count": int, "timestamp": float}
        """
        with self._lock:
            self.flush()
            
            if up_to_record_id is not None:
                sql = "SELECT record_id, record_hash FROM audit_log WHERE record_id <= ? ORDER BY record_id ASC"
                cur = self._conn.execute(sql, (up_to_record_id,))
            else:
                cur = self._conn.execute("SELECT record_id, record_hash FROM audit_log ORDER BY record_id ASC")
            
            rows = cur.fetchall()
            if not rows:
                return {
                    "root_hash": hashlib.sha256(b"empty_audit_chain").hexdigest(),
                    "last_record_id": 0,
                    "record_count": 0,
                    "timestamp": time.time(),
                }
            
            # 计算链根哈希：对所有记录哈希按顺序组合求哈希
            hasher = hashlib.sha256()
            for record_id, record_hash in rows:
                hasher.update(f"{record_id}:{record_hash}".encode("utf-8"))
            
            last_id = rows[-1][0]
            return {
                "root_hash": hasher.hexdigest(),
                "last_record_id": last_id,
                "record_count": len(rows),
                "timestamp": time.time(),
            }
    
    def save_root_hash_snapshot(self, notes: str = "", signature: str = "") -> Dict[str, Any]:
        """保存当前根哈希快照（用于离线归档检测数据库整体替换）
        
        参数：
            notes: 快照备注
            signature: 可选的数字签名（如私钥签名根哈希）
        返回：
            快照信息字典
        """
        with self._lock:
            root_info = self.compute_root_hash()
            now = datetime.now()
            snapshot_date = now.strftime("%Y-%m-%d")
            
            self._conn.execute("""
                INSERT INTO audit_root_hashes (
                    snapshot_time, snapshot_date, last_record_id, 
                    record_count, root_hash, signature, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                root_info["timestamp"], snapshot_date, root_info["last_record_id"],
                root_info["record_count"], root_info["root_hash"], signature, notes
            ))
            self._conn.commit()
            
            snapshot_id = self._conn.execute(
                "SELECT last_insert_rowid()"
            ).fetchone()[0]
            
            return {
                "snapshot_id": snapshot_id,
                "snapshot_date": snapshot_date,
                **root_info,
            }
    
    def verify_against_root_hash(self, snapshot_id: int = None) -> Dict[str, Any]:
        """验证当前数据库与保存的根哈希快照是否一致
        
        参数：
            snapshot_id: 要对比的快照ID，None表示最新快照
        返回：
            {"valid": bool, "snapshot": dict, "current": dict, "errors": [...]}
        """
        with self._lock:
            if snapshot_id is not None:
                cur = self._conn.execute(
                    "SELECT * FROM audit_root_hashes WHERE snapshot_id = ?", (snapshot_id,)
                )
            else:
                cur = self._conn.execute(
                    "SELECT * FROM audit_root_hashes ORDER BY snapshot_id DESC LIMIT 1"
                )
            
            row = cur.fetchone()
            if not row:
                return {
                    "valid": False,
                    "errors": ["未找到根哈希快照"],
                    "snapshot": None,
                    "current": None,
                }
            
            columns = [desc[0] for desc in cur.description]
            snapshot = dict(zip(columns, row))
            
            # 计算当前数据库在快照last_record_id处的根哈希
            current = self.compute_root_hash(up_to_record_id=snapshot["last_record_id"])
            
            errors = []
            if current["root_hash"] != snapshot["root_hash"]:
                errors.append(
                    f"根哈希不匹配！快照根哈希={snapshot['root_hash'][:16]}...，"
                    f"当前根哈希={current['root_hash'][:16]}...，"
                    f"数据库可能被整体替换或篡改"
                )
            if current["record_count"] != snapshot["record_count"]:
                errors.append(
                    f"记录数不匹配：快照记录数={snapshot['record_count']}，"
                    f"当前记录数={current['record_count']}"
                )
            
            return {
                "valid": len(errors) == 0,
                "snapshot": snapshot,
                "current": current,
                "errors": errors,
            }
    
    def get_root_hash_snapshots(self, limit: int = 30) -> List[Dict[str, Any]]:
        """获取根哈希快照列表"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM audit_root_hashes ORDER BY snapshot_id DESC LIMIT ?", (limit,)
            )
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
    
    def export_root_hashes(self, filepath: str) -> int:
        """导出所有根哈希快照为文本文件（用于离线打印/归档）
        
        格式为易读的文本，包含每个快照的日期、记录数、根哈希，便于人工核对。
        """
        snapshots = self.get_root_hash_snapshots(limit=9999)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("结核病风险评估系统 - 审计日志根哈希归档\n")
            f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据库路径: {self.db_path}\n")
            f.write("=" * 70 + "\n\n")
            
            for snap in reversed(snapshots):
                f.write(f"快照ID: {snap['snapshot_id']}\n")
                f.write(f"快照日期: {snap['snapshot_date']}\n")
                f.write(f"记录数: {snap['record_count']}\n")
                f.write(f"最后记录ID: {snap['last_record_id']}\n")
                f.write(f"根哈希: {snap['root_hash']}\n")
                if snap['notes']:
                    f.write(f"备注: {snap['notes']}\n")
                if snap['signature']:
                    f.write(f"签名: {snap['signature']}\n")
                f.write("-" * 50 + "\n")
        
        # 标记快照为已导出
        if snapshots:
            ids = [s['snapshot_id'] for s in snapshots if not s['exported']]
            if ids:
                placeholders = ",".join("?" * len(ids))
                self._conn.execute(
                    f"UPDATE audit_root_hashes SET exported = 1 WHERE snapshot_id IN ({placeholders})",
                    ids
                )
                self._conn.commit()
        
        return len(snapshots)
    
    def export_csv(self, filepath: str, **query_kwargs) -> int:
        """导出审计日志为CSV文件（供监管检查）"""
        import csv
        
        records = self.query(**query_kwargs)
        fieldnames = [
            "record_id", "timestamp", "user_id", "user_name", "user_role",
            "operation_type", "patient_id", "patient_name", "record_type",
            "target_record_id", "ip_address", "reason", "created_at",
        ]
        
        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for rec in records:
                row = rec.to_dict()
                # 格式化时间戳
                if "timestamp" in row:
                    row["timestamp"] = datetime.fromtimestamp(row["timestamp"]).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                writer.writerow(row)
        
        return len(records)
    
    def export_json(self, filepath: str, **query_kwargs) -> int:
        """导出审计日志为JSON文件"""
        records = self.query(**query_kwargs)
        export_data = [r.to_dict() for r in records]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        return len(export_data)
    
    def start_periodic_snapshot(self, interval_hours: int = 24) -> threading.Thread:
        """启动后台线程，定期自动保存根哈希快照
        
        参数：
            interval_hours: 快照间隔小时数，默认24小时（每日一次）
        返回：
            后台线程对象
        """
        def _snapshot_worker():
            while not self._snapshot_stop_event.is_set():
                try:
                    self.save_root_hash_snapshot(notes="自动定期快照")
                    LOGGER.info("审计日志根哈希快照已自动保存")
                except Exception as e:
                    LOGGER.error("自动根哈希快照失败: %s", e)
                # 等待指定间隔，可被stop_event中断
                self._snapshot_stop_event.wait(interval_hours * 3600)
        
        self._snapshot_stop_event = threading.Event()
        thread = threading.Thread(target=_snapshot_worker, daemon=True, name="audit-snapshot")
        thread.start()
        return thread
    
    def stop_periodic_snapshot(self):
        """停止定期快照线程"""
        if hasattr(self, "_snapshot_stop_event"):
            self._snapshot_stop_event.set()
    
    def close(self):
        """关闭数据库连接"""
        with self._lock:
            self.stop_periodic_snapshot()
            self.flush()
            if self._conn:
                try:
                    self._conn.close()
                except Exception as e:
                    LOGGER.debug("关闭审计数据库连接失败: %s", e)
                self._conn = None


class _NullAuditor:
    """空审计器（不做任何操作，用于兼容无auditor场景）"""
    
    def log_operation(self, *args, **kwargs):
        return True
    
    def flush(self):
        return 0
    
    def close(self):
        pass


# 全局默认审计器实例（延迟创建）
_default_auditor: Optional[AuditLogger] = None
_auditor_lock = threading.Lock()


def get_default_auditor(db_path: str = "audit.db") -> AuditLogger:
    """获取全局默认审计器实例（单例）"""
    global _default_auditor
    with _auditor_lock:
        if _default_auditor is None:
            _default_auditor = AuditLogger(db_path)
        return _default_auditor
