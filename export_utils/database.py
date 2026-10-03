#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据库连接池管理与审计追踪。"""

import datetime
import json
import threading
import time

from ._common import LOGGER


class DatabaseManager:
    """数据库连接管理器（统一 SQLite/MySQL/PostgreSQL 接口）

    提供连接池管理、自动重连和统一的数据操作接口。

    用法：
        db = DatabaseManager(db_type='sqlite', db_path='tb_risk.db')
        db.connect()
        db.execute("INSERT INTO ...", params)
        db.close()
    """

    SUPPORTED_TYPES = ('sqlite', 'mysql', 'postgresql')

    def __init__(self, db_type='sqlite', db_path=None, pool_size=5,
                 auto_reconnect=True, max_retries=3):
        self.db_type = db_type.lower()
        if self.db_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"不支持的数据库类型: {db_type}。支持: {self.SUPPORTED_TYPES}")

        self.db_path = db_path or 'tb_risk.db'
        self.pool_size = pool_size
        self.auto_reconnect = auto_reconnect
        self.max_retries = max_retries
        self._connection = None
        self._lock = threading.Lock()
        self._connected = False

    def connect(self):
        """建立数据库连接"""
        with self._lock:
            if self._connected:
                return True

            try:
                if self.db_type == 'sqlite':
                    import sqlite3
                    self._connection = sqlite3.connect(
                        self.db_path, check_same_thread=False)
                    self._connection.row_factory = sqlite3.Row
                    self._placeholder = '?'
                elif self.db_type == 'mysql':
                    import pymysql
                    # 支持两种格式：
                    #   1. URI 格式: mysql://user:password@host:port/database
                    #   2. 冒号分隔: host:port:user:password:database (向后兼容)
                    if self.db_path.startswith('mysql://'):
                        from urllib.parse import urlparse, unquote
                        parsed = urlparse(self.db_path)
                        host = parsed.hostname or 'localhost'
                        port = parsed.port or 3306
                        user = unquote(parsed.username) if parsed.username else 'root'
                        password = unquote(parsed.password) if parsed.password else ''
                        database = parsed.path.lstrip('/') if parsed.path else None
                    else:
                        parts = self.db_path.split(':')
                        host = parts[0] if len(parts) > 0 else 'localhost'
                        port = int(parts[1]) if len(parts) > 1 else 3306
                        user = parts[2] if len(parts) > 2 else 'root'
                        password = ':'.join(parts[3:-1]) if len(parts) > 4 else (parts[3] if len(parts) > 3 else '')
                        database = parts[-1] if len(parts) > 4 else (parts[4] if len(parts) > 4 else None)
                    self._connection = pymysql.connect(
                        host=host, port=port, user=user,
                        password=password, database=database,
                        autocommit=False,
                    )
                    self._placeholder = '%s'
                elif self.db_type == 'postgresql':
                    import psycopg2
                    self._connection = psycopg2.connect(self.db_path)
                    self._placeholder = '%s'

                self._connected = True
                LOGGER.info("数据库连接成功: %s", self.db_type)
                return True

            except Exception as e:
                LOGGER.warning("数据库连接失败: %s", e)
                self._connected = False
                return False

    def execute(self, sql, params=None, commit=True):
        """执行 SQL 语句（含自动重连）"""
        for attempt in range(self.max_retries + 1):
            try:
                if not self._connected:
                    self.connect()

                cursor = self._connection.cursor()
                if params:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)

                if commit:
                    self._connection.commit()

                return cursor

            except Exception as e:
                LOGGER.warning("数据库执行失败 (尝试 %d/%d): %s",
                             attempt + 1, self.max_retries + 1, e)
                if attempt < self.max_retries and self.auto_reconnect:
                    self._connected = False
                    time.sleep(0.5 * (attempt + 1))
                    self.connect()
                else:
                    raise

    def execute_many(self, sql, params_list, commit=True):
        """批量执行 SQL 语句"""
        if not self._connected:
            self.connect()
        cursor = self._connection.cursor()
        cursor.executemany(sql, params_list)
        if commit:
            self._connection.commit()
        return cursor

    def fetch_all(self, sql, params=None):
        """执行查询并返回所有结果"""
        cursor = self.execute(sql, params, commit=False)
        return cursor.fetchall()

    def fetch_one(self, sql, params=None):
        """执行查询并返回第一条结果"""
        cursor = self.execute(sql, params, commit=False)
        return cursor.fetchone()

    def commit(self):
        """提交当前事务"""
        if self._connection:
            try:
                self._connection.commit()
            except Exception as e:
                LOGGER.warning("数据库提交失败: %s", e)

    def cursor(self):
        """获取底层连接的游标（用于多步事务，调用方负责 commit）"""
        if not self._connected:
            self.connect()
        return self._connection.cursor()

    def close(self):
        """关闭数据库连接"""
        with self._lock:
            if self._connection:
                try:
                    self._connection.close()
                except Exception as e:
                    LOGGER.warning("关闭数据库连接失败: %s", e)
                finally:
                    self._connection = None
                    self._connected = False

    @property
    def placeholder(self):
        """获取当前数据库类型的参数占位符"""
        return getattr(self, '_placeholder', '?')

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class AuditLogger:
    """数据审计日志器

    记录每次数据导入操作的时间、来源、记录数等信息，
    支持按时间点回溯数据状态。

    用法：
        audit = AuditLogger(db_manager)
        audit.log_import(source_file='data_2024Q1.csv', record_count=5000)
        history = audit.get_import_history()
    """

    def __init__(self, db_manager=None, log_file=None):
        self.db_manager = db_manager
        self.log_file = log_file
        self._ensure_tables()

    def _ensure_tables(self):
        """确保审计表存在"""
        if self.db_manager:
            try:
                self.db_manager.execute("""
                    CREATE TABLE IF NOT EXISTS audit_imports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        import_time TEXT NOT NULL,
                        source_file TEXT NOT NULL,
                        record_count INTEGER NOT NULL,
                        format TEXT,
                        encoding TEXT,
                        batch_id TEXT,
                        operator TEXT,
                        notes TEXT,
                        created_at TEXT DEFAULT (datetime('now'))
                    )
                """)
                self.db_manager.execute("""
                    CREATE TABLE IF NOT EXISTS audit_data_lineage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        record_id TEXT NOT NULL,
                        source_file TEXT,
                        import_batch TEXT,
                        import_time TEXT,
                        data_source TEXT DEFAULT 'real',
                        created_at TEXT DEFAULT (datetime('now'))
                    )
                """)
            except Exception as e:
                LOGGER.warning("审计表创建失败: %s", e)

    def log_import(self, source_file, record_count, format=None,
                   encoding=None, batch_id=None, operator=None, notes=None):
        """记录一次数据导入操作

        参数：
            source_file (str): 数据来源文件
            record_count (int): 导入记录数
            format (str|None): 文件格式
            encoding (str|None): 文件编码
            batch_id (str|None): 批次ID
            operator (str|None): 操作者
            notes (str|None): 备注

        返回：
            bool: 是否成功
        """
        import_time = datetime.datetime.now().isoformat()

        if self.db_manager:
            try:
                self.db_manager.execute(
                    """INSERT INTO audit_imports
                       (import_time, source_file, record_count, format, encoding,
                        batch_id, operator, notes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (import_time, source_file, record_count, format, encoding,
                     batch_id, operator, notes)
                )
                LOGGER.debug("审计日志已记录: %s (%d 条)", source_file, record_count)
                return True
            except Exception as e:
                LOGGER.warning("审计日志写入失败: %s", e)

        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        'import_time': import_time,
                        'source_file': source_file,
                        'record_count': record_count,
                        'format': format,
                        'encoding': encoding,
                        'batch_id': batch_id,
                        'operator': operator,
                        'notes': notes,
                    }, ensure_ascii=False) + '\n')
                return True
            except Exception as e:
                LOGGER.warning("审计日志文件写入失败: %s", e)

        return False

    def log_data_lineage(self, records, source_file, batch_id=None):
        """记录数据血缘关系

        参数：
            records (list[dict]): 记录列表
            source_file (str): 来源文件
            batch_id (str|None): 批次ID
        """
        if not self.db_manager:
            return

        import_time = datetime.datetime.now().isoformat()
        batch_id = batch_id or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        params_list = []
        for rec in records:
            rec_id = str(rec.get('record_id', rec.get('_id', '')))
            data_source = rec.get('_data_source', 'real')
            params_list.append((rec_id, source_file, batch_id, import_time, data_source))

        if params_list:
            try:
                self.db_manager.execute_many(
                    """INSERT INTO audit_data_lineage
                       (record_id, source_file, import_batch, import_time, data_source)
                       VALUES (?, ?, ?, ?, ?)""",
                    params_list
                )
                LOGGER.debug("数据血缘已记录: %d 条记录", len(params_list))
            except Exception as e:
                LOGGER.warning("数据血缘记录失败: %s", e)

    def get_import_history(self, limit=50):
        """获取导入历史记录

        参数：
            limit (int): 返回记录数上限

        返回：
            list[dict]: 导入历史
        """
        if not self.db_manager:
            return []

        try:
            rows = self.db_manager.fetch_all(
                "SELECT * FROM audit_imports ORDER BY import_time DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in rows]
        except Exception as e:
            LOGGER.warning("查询导入历史失败: %s", e)
            return []

    def get_record_lineage(self, record_id):
        """查询单条记录的数据血缘

        参数：
            record_id (str): 记录ID

        返回：
            dict|None: 血缘信息
        """
        if not self.db_manager:
            return None

        try:
            row = self.db_manager.fetch_one(
                "SELECT * FROM audit_data_lineage WHERE record_id = ? ORDER BY import_time DESC LIMIT 1",
                (str(record_id),)
            )
            return dict(row) if row else None
        except Exception as e:
            LOGGER.warning("查询数据血缘失败: %s", e)
            return None


class BusinessTableExporter:
    """业务表导出器（患者/家庭接触者/社会接触者/评估结果）。

    问题七-1：将 IOMixin.export_to_database 中散落的业务表 CREATE TABLE 与
    INSERT 语句提取到此处集中维护，IOMixin 仅负责连接管理委托与调用编排，
    不再持有任何 SQL 语句。表 schema 与字段映射在此处单一来源。

    表前缀通过 table_prefix 参数支持（默认 'tb_risk_'），与原 IOMixin 行为一致。
    原实现中 family/social 的 CREATE 使用 table_prefix 而 INSERT 硬编码
    'tb_risk_' 前缀（非默认前缀下 CREATE 与 INSERT 目标表不一致），
    此处统一为全部使用 table_prefix，消除该潜在不一致（默认前缀下行为不变）。
    """

    def __init__(self, db_manager, table_prefix='tb_risk_'):
        self.db_manager = db_manager
        self.table_prefix = table_prefix

    def _table(self, name):
        """拼接带前缀的表名。"""
        return f"{self.table_prefix}{name}"

    def create_tables(self, cur):
        """创建所有业务表（IF NOT EXISTS）。

        参数：
            cur: 数据库游标（由调用方负责事务边界）
        """
        cur.execute(f'''CREATE TABLE IF NOT EXISTS {self._table('family_members')} (
            id INTEGER PRIMARY KEY AUTOINCREMENT, member_name TEXT, member_age INTEGER,
            relationship TEXT, single_duration INTEGER, freq_density INTEGER,
            time_span INTEGER, has_tb INTEGER, has_symptoms INTEGER,
            bcg_vaccine INTEGER, past_illness INTEGER, past_illness_type TEXT,
            contact_distance TEXT, ventilation INTEGER, exposure_setting TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        cur.execute(f'''CREATE TABLE IF NOT EXISTS {self._table('social_contacts')} (
            id INTEGER PRIMARY KEY AUTOINCREMENT, contact_name TEXT, contact_age INTEGER,
            single_duration INTEGER, freq_density INTEGER, time_span INTEGER,
            is_high_risk INTEGER, has_symptoms INTEGER, bcg_vaccine INTEGER,
            has_tb INTEGER, ventilation INTEGER, contact_distance TEXT,
            exposure_setting TEXT, past_illness INTEGER, past_illness_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        cur.execute(f'''CREATE TABLE IF NOT EXISTS {self._table('patient')} (
            id INTEGER PRIMARY KEY AUTOINCREMENT, age INTEGER, sputum_smear INTEGER,
            has_cavity INTEGER, active_tb INTEGER, treatment INTEGER,
            treatment_duration INTEGER, cough_freq INTEGER, symptoms INTEGER,
            delay_days INTEGER, family_living_conditions INTEGER,
            flp_percentage REAL, hrsp_percentage REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        cur.execute(f'''CREATE TABLE IF NOT EXISTS {self._table('results')} (
            id INTEGER PRIMARY KEY AUTOINCREMENT, overall_risk TEXT,
            base_infection_probability REAL, family_count INTEGER,
            social_count INTEGER, result_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    def export_patient(self, cur, pi):
        """插入患者基础信息。"""
        try:
            cur.execute(f'''INSERT INTO {self._table('patient')}
                (age,sputum_smear,has_cavity,active_tb,treatment,treatment_duration,
                 cough_freq,symptoms,delay_days,family_living_conditions,
                 flp_percentage,hrsp_percentage)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''', (
                pi.get('age', 0), pi.get('sputum_smear', 0), pi.get('has_cavity', 0),
                pi.get('active_tb', 0), pi.get('treatment', 0), pi.get('treatment_duration', 0),
                pi.get('cough_freq', 0), pi.get('symptoms', 0), pi.get('delay_days', 0),
                pi.get('family_living_conditions', 0), pi.get('flp_percentage', 0),
                pi.get('hrsp_percentage', 0)))
        except Exception as e:
            LOGGER.warning("Export patient data failed: %s", e)

    def export_family_member(self, cur, m):
        """插入单个家庭接触者。"""
        try:
            cur.execute(f'''INSERT INTO {self._table('family_members')}
                (member_name,member_age,relationship,single_duration,freq_density,
                 time_span,has_tb,has_symptoms,bcg_vaccine,past_illness,past_illness_type,
                 contact_distance,ventilation,exposure_setting)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
                m.get('member_name', ''), m.get('member_age', 0), m.get('relationship', ''),
                m.get('single_duration', 0), m.get('freq_density', 0), m.get('time_span', 0),
                m.get('has_tb', 0), m.get('has_symptoms', 0), m.get('bcg_vaccine', 0),
                m.get('past_illness', 0), m.get('past_illness_type', ''),
                m.get('contact_distance', ''), m.get('ventilation', 0),
                m.get('exposure_setting', '')))
        except Exception as e:
            LOGGER.warning("Export family member failed: %s", e)

    def export_social_contact(self, cur, c):
        """插入单个社会接触者。"""
        try:
            cur.execute(f'''INSERT INTO {self._table('social_contacts')}
                (contact_name,contact_age,single_duration,freq_density,time_span,
                 is_high_risk,has_symptoms,bcg_vaccine,has_tb,ventilation,
                 contact_distance,exposure_setting,past_illness,past_illness_type)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
                c.get('contact_name', ''), c.get('contact_age', 0),
                c.get('single_duration', 0), c.get('freq_density', 0), c.get('time_span', 0),
                c.get('is_high_risk', 0), c.get('has_symptoms', 0), c.get('bcg_vaccine', 0),
                c.get('has_tb', 0), c.get('ventilation', 0),
                c.get('contact_distance', ''), c.get('exposure_setting', ''),
                c.get('past_illness', 0), c.get('past_illness_type', '')))
        except Exception as e:
            LOGGER.warning("Export social contact failed: %s", e)

    def export_results(self, cur, results, family_count, social_count):
        """插入评估结果汇总。"""
        try:
            cur.execute(f'''INSERT INTO {self._table('results')}
                (overall_risk,base_infection_probability,family_count,social_count,result_json)
                VALUES (?,?,?,?,?)''', (
                results.get('overall_risk', ''), results.get('base_infection_probability', 0),
                family_count, social_count,
                json.dumps(results, ensure_ascii=False, default=str)))
        except Exception as e:
            LOGGER.warning("Export results failed: %s", e)

    def export_all(self, cur, patient_info, family_members, social_contacts, results):
        """一键导出全部业务数据（含建表）。

        参数：
            cur: 数据库游标（由调用方负责事务边界）
            patient_info: dict 或 None，患者基础信息
            family_members: list[dict]，家庭接触者列表
            social_contacts: list[dict]，社会接触者列表
            results: dict 或 None，评估结果汇总
        """
        self.create_tables(cur)
        if patient_info:
            self.export_patient(cur, patient_info)
        for m in family_members:
            self.export_family_member(cur, m)
        for c in social_contacts:
            self.export_social_contact(cur, c)
        if results:
            self.export_results(cur, results, len(family_members), len(social_contacts))
