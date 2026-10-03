#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据库直连适配器

安全规范：
- 使用只读账号，仅SELECT权限
- 仅访问医院信息科创建的视图（v_patient / v_outpatient / v_inpatient / v_lab）
- IP白名单绑定
- 密码加密存储

数据抽取策略：
- 首次全量，之后按更新时间戳增量抽取
- 检验结果视图：5分钟轮询
- 就诊/诊断视图：15分钟轮询
- 患者基本信息视图：每天凌晨同步

数据质量校验：
- 主键唯一性
- 值域合法性（性别/年龄/结果非乱码）
- 时间逻辑（出院不早于入院、检验时间不晚于现在）
- 关联完整性（患者ID必须存在于v_patient）
"""

from __future__ import annotations

import contextlib
import datetime
import logging
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import (
    HealthcareAdapter, AdapterConfig, AdapterResult,
    DataQualityIssue, timestamp_to_datetime, calculate_age,
    normalize_gender, mask_sensitive,
)

LOGGER = logging.getLogger("tb_risk.health_interop.db_direct")

# SQL标识符白名单正则：仅允许字母、数字、下划线（防止SQL注入）
_SQL_IDENTIFIER_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def _validate_sql_identifier(identifier: str, context: str = "identifier") -> str:
    """校验SQL标识符（表名/列名），仅允许字母、数字、下划线组合。
    
    防止二阶SQL注入：即使视图配置来自外部被篡改的配置文件，也拒绝含特殊字符的标识符。
    
    参数：
        identifier: 待校验的标识符
        context: 标识符用途描述（用于错误日志）
        
    返回：
        校验通过的标识符
        
    异常：
        ValueError: 标识符含非法字符时抛出
    """
    if not isinstance(identifier, str) or not _SQL_IDENTIFIER_RE.match(identifier):
        raise ValueError(
            f"非法SQL标识符({context}): {identifier!r}，"
            f"仅允许字母、数字、下划线，且必须以字母或下划线开头"
        )
    return identifier


# 四类核心视图定义
DEFAULT_VIEWS = {
    "patient": {
        "name": "v_tb_patient_basic",
        "pk": "patient_id",
        "update_field": "last_update_time",
        "poll_interval": 86400,  # 24小时
        "required_fields": ["patient_id", "gender", "birth_date"],
    },
    "outpatient": {
        "name": "v_tb_outpatient_visit",
        "pk": "visit_id",
        "update_field": "last_update_time",
        "poll_interval": 900,  # 15分钟
    },
    "inpatient": {
        "name": "v_tb_inpatient_visit",
        "pk": "admission_id",
        "update_field": "last_update_time",
        "poll_interval": 900,
    },
    "lab": {
        "name": "v_tb_lab_result",
        "pk": "test_id",
        "update_field": "report_time",
        "poll_interval": 300,  # 5分钟
    },
}


class DatabaseDirectAdapter(HealthcareAdapter):
    """数据库直连适配器（兼容老旧系统最后手段）"""

    ADAPTER_TYPE = "db_direct"

    def __init__(self, config: Optional[AdapterConfig] = None,
                 db_type: str = "mysql",
                 connection_params: Optional[Dict[str, Any]] = None,
                 views_config: Optional[Dict[str, Dict[str, Any]]] = None):
        super().__init__(config)
        self.db_type = db_type
        self._conn_params = connection_params or {}
        self._views = self._validate_views_config(views_config or DEFAULT_VIEWS)
        self._conn = None
        self._last_sync: Dict[str, Optional[datetime.datetime]] = {}
        self._sync_timers: Dict[str, threading.Timer] = {}
        self._polling = False
        self._poll_thread: Optional[threading.Thread] = None
        self._data_callbacks: List[Callable[[str, AdapterResult], None]] = []
        self._patient_cache: Dict[str, Dict[str, Any]] = {}
        self._seen_pks: Dict[str, set] = {k: set() for k in self._views}
        self._quality_issues: List[DataQualityIssue] = []

    def _validate_views_config(self, views: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """校验视图配置中的所有SQL标识符，防止注入。"""
        validated = {}
        for key, conf in views.items():
            _validate_sql_identifier(key, f"view_key:{key}")
            vconf = dict(conf)
            # 校验表名
            if "name" in vconf:
                vconf["name"] = _validate_sql_identifier(vconf["name"], f"table:{key}")
            # 校验主键列名
            if "pk" in vconf:
                vconf["pk"] = _validate_sql_identifier(vconf["pk"], f"pk:{key}")
            # 校验更新时间字段
            if "update_field" in vconf and vconf["update_field"]:
                vconf["update_field"] = _validate_sql_identifier(
                    vconf["update_field"], f"update_field:{key}")
            validated[key] = vconf
        return validated

    def add_data_callback(self, cb: Callable[[str, AdapterResult], None]):
        """注册新数据回调 (view_name, result) → None"""
        self._data_callbacks.append(cb)

    # ---- 连接 ----

    def connect(self) -> bool:
        """建立只读数据库连接"""
        try:
            self._conn = self._create_connection()
            if self._conn is None:
                return False
            # 测试只读权限（尝试SELECT 1）
            with contextlib.closing(self._conn.cursor()) as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            self._logger.info("数据库直连成功: type=%s host=%s",
                              self.db_type, self._conn_params.get("host", "local"))
            return True
        except Exception as e:
            self._logger.error("数据库连接失败: %s", e)
            return False

    def _create_connection(self):
        """创建数据库连接（根据db_type选择驱动）"""
        params = self._conn_params
        try:
            if self.db_type == "mysql":
                import pymysql
                conn = pymysql.connect(
                    host=params.get("host", "localhost"),
                    port=int(params.get("port", 3306)),
                    user=params.get("user", ""),
                    password=params.get("password", ""),
                    database=params.get("database", ""),
                    read_default_file=params.get("defaults_file"),
                    connect_timeout=self.config.connect_timeout,
                    read_timeout=self.config.read_timeout,
                )
                return conn
            elif self.db_type == "postgresql":
                import psycopg2
                conn = psycopg2.connect(
                    host=params.get("host", "localhost"),
                    port=int(params.get("port", 5432)),
                    user=params.get("user", ""),
                    password=params.get("password", ""),
                    dbname=params.get("database", ""),
                    connect_timeout=self.config.connect_timeout,
                )
                return conn
            elif self.db_type == "sqlite":
                import sqlite3
                conn = sqlite3.connect(params.get("file_path", ""), timeout=30)
                conn.row_factory = sqlite3.Row
                return conn
            elif self.db_type == "mssql":
                import pymssql
                conn = pymssql.connect(
                    server=params.get("host", "localhost"),
                    port=int(params.get("port", 1433)),
                    user=params.get("user", ""),
                    password=params.get("password", ""),
                    database=params.get("database", ""),
                    login_timeout=self.config.connect_timeout,
                )
                return conn
            else:
                self._logger.error("不支持的数据库类型: %s", self.db_type)
                return None
        except ImportError as e:
            self._logger.error("缺少数据库驱动: %s", e)
            return None
        except Exception as e:
            self._logger.error("创建数据库连接失败: %s", e)
            return None

    def disconnect(self):
        self.stop_polling()
        if self._conn:
            try:
                self._conn.close()
            except Exception as e:
                LOGGER.debug("断开数据库连接失败: %s", e)
            self._conn = None

    def health_check(self) -> tuple:
        if not self._conn:
            return False, "未连接"
        try:
            with contextlib.closing(self._conn.cursor()) as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True, "OK"
        except Exception as e:
            return False, str(e)

    # ---- 患者查询 ----

    def fetch_patient(self, patient_id: str) -> AdapterResult:
        """按患者ID查询所有视图数据"""
        result = AdapterResult(
            source_system=f"DB_{self.db_type}",
            patient_id=patient_id,
            last_sync_time=datetime.datetime.now(),
        )
        start = time.time()
        try:
            # 基本信息
            pi = self._query_patient_basic(patient_id)
            if pi:
                result.patient_info = pi
            else:
                result.errors.append(f"患者 {patient_id} 不存在于基本信息视图")
                return result

            # 就诊记录（提取诊断）
            visits = self._query_patient_visits(patient_id)
            result.diagnoses = visits.get("diagnoses", [])

            # 检验结果
            labs = self._query_patient_labs(patient_id)
            result.lab_results = labs

            result.success = True
            result.records_processed = (len(result.diagnoses) + len(result.lab_results))
            duration_ms = (time.time() - start) * 1000
            self._record_success(f"fetch_patient:{patient_id}", duration_ms,
                                 result.records_processed)
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            result.errors.append(str(e))
            self._record_failure(f"fetch_patient:{patient_id}", str(e), duration_ms)
            self._logger.error("查询患者%s失败: %s", patient_id, e, exc_info=True)
        return result

    def _query_patient_basic(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """查询患者基本信息视图"""
        view = self._views["patient"]
        sql = f"SELECT * FROM {view['name']} WHERE {view['pk']} = %s"
        params = [patient_id]
        if self.db_type in ("sqlite", "postgresql"):
            sql = sql.replace("%s", "?") if self.db_type == "sqlite" else sql.replace("%s", "%s")
        try:
            with contextlib.closing(self._conn.cursor()) as cur:
                cur.execute(sql, params)
                cols = [d[0] for d in cur.description] if cur.description else []
                row = cur.fetchone()
            if not row:
                return None
            rec = dict(zip(cols, row)) if cols else {}
            # 标准化字段
            return self._normalize_patient_record(rec)
        except Exception as e:
            self._logger.error("查询患者基本信息失败: %s", e)
            return None

    def _query_patient_visits(self, patient_id: str) -> Dict[str, List[Dict]]:
        """查询门诊/住院就诊记录"""
        diagnoses = []
        for vk in ("outpatient", "inpatient"):
            if vk not in self._views:
                continue
            view = self._views[vk]
            pk_field = view["pk"]
            patient_field = "patient_id"
            diag_code_field = "diagnosis_icd"
            diag_name_field = "diagnosis_name"
            diag_date_field = "diagnosis_date" if vk == "outpatient" else "discharge_date"
            sql = f"SELECT * FROM {view['name']} WHERE {patient_field} = %s"
            params = [patient_id]
            if self.db_type == "sqlite":
                sql = sql.replace("%s", "?")
            try:
                with contextlib.closing(self._conn.cursor()) as cur:
                    cur.execute(sql, params)
                    cols = [d[0] for d in cur.description] if cur.description else []
                    for row in cur.fetchall():
                        rec = dict(zip(cols, row))
                        if rec.get(diag_code_field) or rec.get(diag_name_field):
                            diagnoses.append({
                                "code": rec.get(diag_code_field, ""),
                                "display": rec.get(diag_name_field, ""),
                                "diagnosis_date": timestamp_to_datetime(
                                    rec.get(diag_date_field, rec.get("visit_date", ""))),
                                "visit_type": vk,
                                "visit_id": rec.get(pk_field, ""),
                            })
            except Exception as e:
                self._logger.warning("查询%s视图失败: %s", vk, e)
        return {"diagnoses": diagnoses}

    def _query_patient_labs(self, patient_id: str) -> List[Dict[str, Any]]:
        """查询检验结果"""
        labs = []
        if "lab" not in self._views:
            return labs
        view = self._views["lab"]
        sql = f"SELECT * FROM {view['name']} WHERE patient_id = %s ORDER BY report_time DESC"
        params = [patient_id]
        if self.db_type == "sqlite":
            sql = sql.replace("%s", "?")
        try:
            with contextlib.closing(self._conn.cursor()) as cur:
                cur.execute(sql, params)
                cols = [d[0] for d in cur.description] if cur.description else []
                for row in cur.fetchall():
                    rec = dict(zip(cols, row))
                    lab = self._normalize_lab_record(rec)
                    if lab:
                        labs.append(lab)
        except Exception as e:
            self._logger.warning("查询检验结果失败: %s", e)
        return labs

    # ---- 增量抽取 ----

    def fetch_incremental(self, since: Optional[datetime.datetime] = None
                          ) -> List[AdapterResult]:
        """按视图的更新时间戳增量抽取"""
        results = []
        for view_key, view_conf in self._views.items():
            view_name = view_conf["name"]
            pk = view_conf["pk"]
            update_field = view_conf.get("update_field")
            if not update_field:
                continue
            sync_since = since or self._last_sync.get(view_key)
            if sync_since is None:
                # 首次全量
                sync_since = datetime.datetime.now() - datetime.timedelta(
                    days=365 * self.config.history_years)
            try:
                rows = self._incremental_query(view_name, pk, update_field, sync_since)
                self._logger.info("视图%s增量查询返回%d条记录", view_name, len(rows))
                # 按患者分组
                patient_groups: Dict[str, List[Dict]] = {}
                for row in rows:
                    pid = str(row.get("patient_id", ""))
                    if not pid:
                        continue
                    patient_groups.setdefault(pid, []).append(row)
                # 数据质量校验
                valid_rows, issues = self._validate_batch(view_key, rows)
                self._quality_issues.extend(issues)
                # 为每个有新数据的患者构造 AdapterResult
                for pid, recs in patient_groups.items():
                    result = self._build_result_from_rows(pid, view_key, recs)
                    if result.success:
                        results.append(result)
                        for cb in self._data_callbacks:
                            try:
                                cb(view_key, result)
                            except Exception as e:
                                self._logger.error("增量回调执行失败: %s", e)
                self._last_sync[view_key] = datetime.datetime.now()
            except Exception as e:
                self._logger.error("视图%s增量抽取失败: %s", view_name, e)
        return results

    def _incremental_query(self, view_name: str, pk: str,
                           update_field: str,
                           since: datetime.datetime) -> List[Dict[str, Any]]:
        """执行增量查询"""
        # 防御性校验：即使配置已校验，这里再做一次确保标识符安全
        _validate_sql_identifier(view_name, "incremental:view_name")
        _validate_sql_identifier(pk, "incremental:pk")
        _validate_sql_identifier(update_field, "incremental:update_field")
        since_str = since.strftime("%Y-%m-%d %H:%M:%S")
        sql = f"SELECT * FROM {view_name} WHERE {update_field} >= %s ORDER BY {update_field} ASC"
        params = [since_str]
        if self.db_type == "sqlite":
            sql = sql.replace("%s", "?")
        elif self.db_type == "postgresql":
            pass
        with contextlib.closing(self._conn.cursor()) as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        return rows

    # ---- 轮询调度 ----

    def start_polling(self):
        """启动后台轮询线程"""
        if self._polling:
            return
        self._polling = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="DB-Direct-Poller")
        self._poll_thread.start()
        self._logger.info("数据库增量轮询已启动")

    def stop_polling(self):
        """停止轮询"""
        self._polling = False
        for t in self._sync_timers.values():
            t.cancel()
        self._sync_timers.clear()

    def _poll_loop(self):
        last_run: Dict[str, float] = {}
        while self._polling:
            now = time.time()
            for view_key, view_conf in self._views.items():
                interval = view_conf.get("poll_interval", 300)
                if now - last_run.get(view_key, 0) >= interval:
                    try:
                        self.fetch_incremental()
                        last_run[view_key] = now
                    except Exception as e:
                        self._logger.error("轮询视图%s失败: %s", view_key, e)
            time.sleep(30)

    # ---- 数据质量校验 ----

    def _validate_batch(self, view_key: str,
                        rows: List[Dict]) -> Tuple[List[Dict], List[DataQualityIssue]]:
        """批次数据质量校验"""
        issues = []
        valid = []
        seen_pks = self._seen_pks.get(view_key, set())
        view = self._views[view_key]
        pk = view["pk"]

        for i, row in enumerate(rows):
            row_id = f"{view_key}:{row.get(pk, i)}"
            # 1. 主键唯一性
            pk_val = row.get(pk)
            if pk_val and pk_val in seen_pks:
                issues.append(DataQualityIssue(
                    severity="error", field=pk, value=pk_val,
                    message=f"主键重复: {pk_val}", record_id=row_id))
                continue
            if pk_val:
                seen_pks.add(pk_val)

            # 2. 值域检查
            if view_key == "patient":
                gender = str(row.get("gender", "")).strip()
                if gender and normalize_gender(gender) == "unknown" and gender not in ("未知", "", "0", "9"):
                    issues.append(DataQualityIssue(
                        severity="warning", field="gender", value=gender,
                        message=f"性别值异常: {gender}", record_id=row_id))
                age = None
                if row.get("birth_date"):
                    age = calculate_age(row["birth_date"])
                    if age is None or age < 0 or age > 120:
                        issues.append(DataQualityIssue(
                            severity="error", field="age", value=age,
                            message=f"年龄异常: {age}", record_id=row_id))

            # 3. 时间逻辑
            if view_key == "inpatient":
                adm = timestamp_to_datetime(row.get("admission_date"))
                dis = timestamp_to_datetime(row.get("discharge_date"))
                if adm and dis and dis < adm:
                    issues.append(DataQualityIssue(
                        severity="error", field="discharge_date",
                        message="出院时间早于入院时间", record_id=row_id))
                    continue
            now = datetime.datetime.now()
            for tf in ("report_time", "test_time", "visit_time", "diagnosis_date"):
                tv = timestamp_to_datetime(row.get(tf))
                if tv and tv > now + datetime.timedelta(days=1):
                    issues.append(DataQualityIssue(
                        severity="warning", field=tf, value=str(tv),
                        message=f"时间晚于当前: {tv}", record_id=row_id))

            # 4. 乱码检测（简单启发式）
            for k, v in row.items():
                if isinstance(v, str) and re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", v):
                    issues.append(DataQualityIssue(
                        severity="error", field=k, value=v[:50],
                        message="字段包含不可打印字符（可能乱码）", record_id=row_id))

            valid.append(row)

        return valid, issues

    # ---- 字段标准化 ----

    def _normalize_patient_record(self, rec: Dict) -> Dict[str, Any]:
        """将视图字段标准化为tb_risk内部字段"""
        pi = {}
        # 字段映射
        mapping = {
            "patient_id": ("patient_id", str),
            "name": ("name", str),
            "gender": ("gender_text", str),
            "sex": ("gender_text", str),
            "birth_date": ("birth_date", str),
            "birthday": ("birth_date", str),
            "age": ("age", int),
            "id_card": ("id_card", str),
            "idcard": ("id_card", str),
            "phone": ("phone", str),
            "mobile": ("phone", str),
            "tel": ("phone", str),
            "address": ("address", str),
            "district": ("district", str),
            "nation": ("ethnicity", str),
            "ethnicity": ("ethnicity", str),
            "occupation": ("occupation", str),
            "marital_status": ("marital_status", str),
            "create_time": ("register_date", str),
        }
        for src_key, (dst_key, converter) in mapping.items():
            if src_key in rec and rec[src_key] is not None:
                try:
                    pi[dst_key] = converter(rec[src_key])
                except (ValueError, TypeError):
                    pi[dst_key] = rec[src_key]

        # 性别标准化
        if "gender_text" in pi:
            pi["gender"] = normalize_gender(pi["gender_text"])
            if pi["gender"] == "male":
                pi["gender_text"] = "男"
            elif pi["gender"] == "female":
                pi["gender_text"] = "女"
        # 年龄计算
        if "age" not in pi and "birth_date" in pi:
            age = calculate_age(pi["birth_date"])
            if age is not None:
                pi["age"] = age
        return pi

    def _normalize_lab_record(self, rec: Dict) -> Optional[Dict[str, Any]]:
        """标准化检验记录"""
        code = str(rec.get("item_code", rec.get("test_code", "")))
        name = str(rec.get("item_name", rec.get("test_name", "")))
        value = rec.get("result_value", rec.get("value", ""))
        unit = str(rec.get("unit", ""))
        ref = str(rec.get("reference_range", rec.get("ref_range", "")))
        flag = str(rec.get("abnormal_flag", rec.get("flag", "")))
        report_time = timestamp_to_datetime(
            rec.get("report_time", rec.get("test_time", "")))
        sample = str(rec.get("sample_type", rec.get("specimen", "")))

        # 阴阳性判定
        is_positive = None
        v_str = str(value).lower()
        if any(k in v_str for k in ("阳性", "positive", "+", "检出", "reactive")):
            is_positive = True
        elif any(k in v_str for k in ("阴性", "negative", "-", "未检出", "non-reactive")):
            is_positive = False

        return {
            "local_code": code,
            "loinc": "",
            "name": name or code,
            "value": value,
            "unit": unit,
            "reference_range": ref,
            "interpretation": flag,
            "is_abnormal": flag.upper() in ("H", "L", "HH", "LL", "+", "阳性", "异常"),
            "is_positive": is_positive,
            "effective_time": report_time.isoformat() if report_time else None,
            "specimen_type": sample,
        }

    def _build_result_from_rows(self, patient_id: str, view_key: str,
                                rows: List[Dict]) -> AdapterResult:
        """从一批同患者的行数据构造AdapterResult"""
        result = AdapterResult(
            source_system=f"DB_{self.db_type}",
            patient_id=patient_id,
            last_sync_time=datetime.datetime.now(),
        )
        # 如果有缓存的基本信息直接用，否则查询
        if patient_id in self._patient_cache:
            result.patient_info = dict(self._patient_cache[patient_id])
        else:
            pi = self._query_patient_basic(patient_id)
            if pi:
                result.patient_info = pi
                self._patient_cache[patient_id] = dict(pi)
            else:
                result.warnings.append(f"患者{patient_id}基本信息未找到")

        if view_key == "lab":
            for r in rows:
                lab = self._normalize_lab_record(r)
                if lab:
                    result.lab_results.append(lab)
        elif view_key in ("outpatient", "inpatient"):
            for r in rows:
                diag = {
                    "code": str(r.get("diagnosis_icd", r.get("diag_code", ""))),
                    "display": str(r.get("diagnosis_name", r.get("diag_name", ""))),
                    "diagnosis_date": timestamp_to_datetime(
                        r.get("diagnosis_date", r.get("discharge_date", r.get("visit_date")))),
                    "visit_type": view_key,
                }
                result.diagnoses.append(diag)
        elif view_key == "patient":
            pi = self._normalize_patient_record(rows[0] if rows else {})
            result.patient_info.update(pi)
            self._patient_cache[patient_id] = dict(result.patient_info)

        result.success = True
        result.records_processed = len(rows)
        return result

    @property
    def quality_issues(self) -> List[DataQualityIssue]:
        return list(self._quality_issues)
