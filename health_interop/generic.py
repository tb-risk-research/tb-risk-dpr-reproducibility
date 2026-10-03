#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用WebService/RESTful适配器

提供厂商自定义接口的通用适配能力：
- RESTfulClient: REST API客户端（OAuth2/API Key/签名认证）
- SOAPClient: WebService SOAP客户端
- 超时设置（连接5秒/读取30秒）
- 失败重试（指数退避，最多3次）
- 熔断机制（连续5次失败熔断5分钟）
- 日期时间格式自动识别
- 枚举值映射
- XML/JSON格式适配
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Optional, Tuple
from xml.dom import minidom

from .base import (
    HealthcareAdapter, AdapterConfig, AdapterResult,
    timestamp_to_datetime, safe_parse_datetime, normalize_gender,
    mask_sensitive, calculate_age, GENDER_MAPPING,
)

LOGGER = logging.getLogger("tb_risk.health_interop.generic")


# ============================================================================
# 通用认证机制
# ============================================================================

class AuthProvider:
    """认证提供者基类"""

    def get_headers(self) -> Dict[str, str]:
        raise NotImplementedError

    def refresh_if_needed(self):
        pass


class NoAuth(AuthProvider):
    def get_headers(self) -> Dict[str, str]:
        return {}


class APIKeyAuth(AuthProvider):
    """API Key认证（Header或Query参数）"""

    def __init__(self, api_key: str, header_name: str = "X-API-Key",
                 param_name: Optional[str] = None):
        self.api_key = api_key
        self.header_name = header_name
        self.param_name = param_name

    def get_headers(self) -> Dict[str, str]:
        return {self.header_name: self.api_key} if self.api_key else {}


class OAuth2ClientCredentials(AuthProvider):
    """OAuth2 客户端凭证模式"""

    def __init__(self, token_url: str, client_id: str, client_secret: str,
                 scope: str = ""):
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self._token: Optional[str] = None
        self._expires_at: float = 0

    def get_headers(self) -> Dict[str, str]:
        self.refresh_if_needed()
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def refresh_if_needed(self):
        if self._token and time.time() < self._expires_at - 60:
            return
        try:
            data = urllib.parse.urlencode({
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": self.scope,
            }).encode("utf-8")
            req = urllib.request.Request(
                self.token_url, data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            self._token = result.get("access_token")
            self._expires_at = time.time() + int(result.get("expires_in", 3600))
            LOGGER.info("OAuth2 token刷新成功")
        except Exception as e:
            LOGGER.error("OAuth2 token刷新失败: %s", e)
            self._token = None


class SignatureAuth(AuthProvider):
    """API Key + 签名认证"""

    def __init__(self, app_id: str, app_secret: str,
                 sign_method: str = "HMAC-SHA256"):
        self.app_id = app_id
        self.app_secret = app_secret
        self.sign_method = sign_method

    def sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """对请求参数签名（追加timestamp/app_id/sign）"""
        params = dict(params)
        params["app_id"] = self.app_id
        params["timestamp"] = str(int(time.time()))
        # 按key排序
        sorted_items = sorted((str(k), str(v)) for k, v in params.items())
        sign_str = "&".join(f"{k}={v}" for k, v in sorted_items)
        sign_str += f"&key={self.app_secret}"
        sign = hashlib.md5(sign_str.encode("utf-8")).hexdigest().upper()
        if "HMAC" in self.sign_method.upper():
            sign = hmac.new(
                self.app_secret.encode("utf-8"),
                sign_str.encode("utf-8"),
                hashlib.sha256).hexdigest()
        params["sign"] = sign
        return params

    def get_headers(self) -> Dict[str, str]:
        return {}


# ============================================================================
# REST API 客户端
# ============================================================================

class RESTClient(HealthcareAdapter):
    """通用RESTful API适配器

    可通过vendor_params自定义字段映射、分页参数、认证方式等。
    """

    ADAPTER_TYPE = "rest_generic"

    def __init__(self, config: Optional[AdapterConfig] = None,
                 auth: Optional[AuthProvider] = None,
                 endpoints: Optional[Dict[str, str]] = None,
                 field_mapping: Optional[Dict[str, str]] = None):
        super().__init__(config)
        self._base_url = self.config.base_url.rstrip("/") if self.config.base_url else ""
        self._auth = auth or NoAuth()
        self._endpoints = endpoints or {}
        self._field_mapping = field_mapping or {}
        self._enum_mappings: Dict[str, Dict[str, str]] = {}

    def connect(self) -> bool:
        if not self._base_url:
            return False
        try:
            # 简单健康检查（如果配置了health端点）
            health_ep = self._endpoints.get("health", "/health")
            if health_ep:
                self.request("GET", health_ep, timeout=self.config.connect_timeout,
                            skip_monitor=True)
            self._logger.info("REST客户端连接成功: %s", self._base_url)
            return True
        except Exception as e:
            self._logger.warning("REST健康检查失败: %s（继续运行）", e)
            return True  # 不阻塞，实际调用时再验证

    def add_enum_mapping(self, field: str, mapping: Dict[str, str]):
        """注册枚举值映射（例如 gender: {"1":"male","2":"female"}）"""
        self._enum_mappings[field] = mapping

    def request(self, method: str, path: str,
                params: Optional[Dict] = None,
                json_data: Optional[Dict] = None,
                timeout: Optional[int] = None,
                retries: Optional[int] = None,
                skip_monitor: bool = False) -> Dict[str, Any]:
        """发送HTTP请求，带重试/熔断/监控"""
        if not self._base_url:
            raise RuntimeError("REST客户端未配置base_url，无法发送请求")
        timeout = timeout or self.config.read_timeout
        retries = retries if retries is not None else self.config.max_retries
        endpoint_key = f"{method} {path.split('?')[0]}"

        # 签名认证
        if isinstance(self._auth, SignatureAuth):
            if method == "GET" and params is None:
                params = {}
            if params is not None:
                params = self._auth.sign(params)
            elif json_data is not None:
                json_data = self._auth.sign(json_data)

        url = f"{self._base_url}/{path.lstrip('/')}"
        if params:
            qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
            url = f"{url}?{qs}"

        if not skip_monitor and self._monitor:
            if not self._monitor.allow_request(self.ADAPTER_TYPE, endpoint_key):
                raise RuntimeError(f"REST接口已熔断: {endpoint_key}")

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
        }
        headers.update(self._auth.get_headers())

        body = json.dumps(json_data, ensure_ascii=False).encode("utf-8") if json_data else None

        last_error = None
        for attempt in range(retries + 1):
            self._auth.refresh_if_needed()
            start = time.time()
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method=method)
                ctx = None
                if not self.config.ssl_verify:
                    import ssl
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                    resp_body = resp.read().decode("utf-8")
                    duration_ms = (time.time() - start) * 1000
                    if not skip_monitor:
                        self._record_success(endpoint_key, duration_ms)
                    if not resp_body:
                        return {}
                    return json.loads(resp_body)
            except urllib.error.HTTPError as e:
                duration_ms = (time.time() - start) * 1000
                err_text = ""
                try:
                    err_text = e.read().decode("utf-8", errors="replace")[:500]
                except Exception as ex:
                    LOGGER.debug("读取HTTP错误响应失败: %s", ex)
                last_error = f"HTTP {e.code}: {err_text}"
                if 400 <= e.code < 500 and e.code != 429:
                    if not skip_monitor:
                        self._record_failure(endpoint_key, last_error, duration_ms)
                    raise RuntimeError(last_error)
                LOGGER.warning("REST请求失败(%d/%d): %s", attempt + 1, retries, last_error)
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                last_error = str(e)
                LOGGER.warning("REST请求异常(%d/%d): %s", attempt + 1, retries, e)
            if attempt < retries:
                wait = self.config.retry_backoff ** attempt
                time.sleep(wait)

        if not skip_monitor:
            self._record_failure(endpoint_key, last_error or "未知错误")
        raise RuntimeError(f"REST请求最终失败: {last_error}")

    def fetch_patient(self, patient_id: str) -> AdapterResult:
        """按需查询患者数据（使用endpoints配置）"""
        result = AdapterResult(
            source_system="REST",
            patient_id=patient_id,
            last_sync_time=datetime.datetime.now(),
        )
        start = time.time()
        try:
            patient_ep = self._endpoints.get("patient", "/patient/{id}")
            path = patient_ep.replace("{id}", urllib.parse.quote(str(patient_id)))
            data = self.request("GET", path)
            result.patient_info = self._map_patient_fields(data)

            # 可选：拉取检验、诊断等
            for ep_key in ("labs", "diagnoses", "medications", "imaging"):
                ep = self._endpoints.get(ep_key)
                if not ep:
                    continue
                list_path = ep.replace("{id}", urllib.parse.quote(str(patient_id)))
                try:
                    list_data = self.request("GET", list_path)
                    items = list_data if isinstance(list_data, list) else list_data.get("data", [])
                    if not isinstance(items, list):
                        items = [items]
                    if ep_key == "labs":
                        result.lab_results = [self._map_lab_fields(x) for x in items]
                    elif ep_key == "diagnoses":
                        result.diagnoses = [self._map_diagnosis_fields(x) for x in items]
                    elif ep_key == "medications":
                        result.medications = [self._map_medication_fields(x) for x in items]
                    elif ep_key == "imaging":
                        result.imaging_reports = [self._map_imaging_fields(x) for x in items]
                except Exception as e:
                    LOGGER.warning("拉取%s失败: %s", ep_key, e)
                    result.warnings.append(f"{ep_key}拉取失败: {e}")

            result.success = True
            result.records_processed = sum(
                len(x) for x in (result.lab_results, result.diagnoses,
                                 result.medications, result.imaging_reports))
            duration_ms = (time.time() - start) * 1000
            self._record_success(f"fetch_patient:{patient_id}", duration_ms,
                                 result.records_processed)
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            result.errors.append(str(e))
            self._record_failure(f"fetch_patient:{patient_id}", str(e), duration_ms)
        return result

    def fetch_incremental(self, since=None) -> List[AdapterResult]:
        """增量拉取（使用endpoints中的incremental端点）"""
        inc_ep = self._endpoints.get("incremental")
        if not inc_ep:
            self._logger.info("未配置incremental端点，跳过增量拉取")
            return []
        if since is None:
            since = self._last_sync_time
        since_str = since.isoformat() if since else ""
        results = []
        try:
            params = {"since": since_str} if since_str else {}
            data = self.request("GET", inc_ep, params=params)
            patients = data if isinstance(data, list) else data.get("patients", data.get("data", []))
            if not isinstance(patients, list):
                patients = [patients]
            for p in patients:
                pid = p.get("patient_id") or p.get("id")
                if pid:
                    r = self.fetch_patient(pid)
                    if r.success:
                        results.append(r)
            self._last_sync_time = datetime.datetime.now()
        except Exception as e:
            self._logger.error("增量拉取失败: %s", e)
        return results

    # ---- 字段映射 ----

    def _apply_mapping(self, data: Dict) -> Dict:
        """应用字段名映射配置"""
        if not self._field_mapping:
            return dict(data)
        mapped = {}
        for src, dst in self._field_mapping.items():
            if src in data:
                mapped[dst] = data[src]
        # 保留未映射字段
        for k, v in data.items():
            if k not in self._field_mapping:
                mapped.setdefault(k, v)
        return mapped

    def _apply_enum_mapping(self, field: str, value: Any) -> Any:
        """应用枚举值映射"""
        mapping = self._enum_mappings.get(field)
        if not mapping or value is None:
            return value
        s = str(value).strip()
        return mapping.get(s, mapping.get(s.lower(), value))

    def _map_patient_fields(self, data: Dict) -> Dict[str, Any]:
        mapped = self._apply_mapping(data)
        pi = {}
        # 通用字段识别
        field_patterns = {
            "patient_id": ("patient_id", "id", "pid", "mrn"),
            "name": ("name", "patient_name", "xm", "xingming"),
            "gender": ("gender", "sex", "xb"),
            "birth_date": ("birth_date", "birthday", "birth", "csrq"),
            "age": ("age", "nl"),
            "id_card": ("id_card", "idcard", "sfz", "id_number"),
            "phone": ("phone", "mobile", "tel", "dh"),
            "address": ("address", "current_address", "zz", "xianzz"),
            "ethnicity": ("ethnicity", "nation", "mz"),
            "occupation": ("occupation", "zy"),
            "district": ("district", "county", "xq"),
        }
        for std_key, candidates in field_patterns.items():
            for c in candidates:
                if c in mapped and mapped[c] is not None:
                    val = mapped[c]
                    if std_key == "gender":
                        val = normalize_gender(val)
                    elif std_key in ("birth_date",):
                        dt = safe_parse_datetime(str(val))
                        if dt:
                            val = dt.strftime("%Y-%m-%d")
                    pi[std_key] = val
                    break
        # 年龄推导
        if "age" not in pi and "birth_date" in pi:
            age = calculate_age(pi["birth_date"])
            if age is not None:
                pi["age"] = age
        return pi

    def _map_lab_fields(self, data: Dict) -> Dict[str, Any]:
        mapped = self._apply_mapping(data)
        effective = mapped.get("effective_time") or mapped.get("test_time") or mapped.get("report_time")
        return {
            "local_code": mapped.get("item_code", mapped.get("code", "")),
            "name": mapped.get("item_name", mapped.get("name", "")),
            "value": mapped.get("value", mapped.get("result_value")),
            "unit": mapped.get("unit", ""),
            "reference_range": mapped.get("reference_range", mapped.get("ref_range", "")),
            "interpretation": str(mapped.get("abnormal_flag", mapped.get("flag", ""))),
            "effective_time": timestamp_to_datetime(effective),
            "specimen_type": mapped.get("specimen", mapped.get("sample_type", "")),
        }

    def _map_diagnosis_fields(self, data: Dict) -> Dict[str, Any]:
        mapped = self._apply_mapping(data)
        return {
            "code": mapped.get("icd_code", mapped.get("diagnosis_code", "")),
            "display": mapped.get("diagnosis_name", mapped.get("name", "")),
            "diagnosis_date": timestamp_to_datetime(
                mapped.get("diagnosis_date", mapped.get("date"))),
            "diagnosis_type": mapped.get("type", "diagnosis"),
        }

    def _map_medication_fields(self, data: Dict) -> Dict[str, Any]:
        mapped = self._apply_mapping(data)
        return {
            "name": mapped.get("medication_name", mapped.get("name", "")),
            "code": mapped.get("medication_code", mapped.get("code", "")),
            "start_date": timestamp_to_datetime(mapped.get("start_date")),
            "end_date": timestamp_to_datetime(mapped.get("end_date")),
            "dose": mapped.get("dose", ""),
        }

    def _map_imaging_fields(self, data: Dict) -> Dict[str, Any]:
        mapped = self._apply_mapping(data)
        return {
            "title": mapped.get("exam_name", mapped.get("title", "影像报告")),
            "conclusion": mapped.get("conclusion", mapped.get("impression",
                           mapped.get("report_text", ""))),
            "effective_time": timestamp_to_datetime(
                mapped.get("exam_time", mapped.get("report_time"))),
            "nlp_pending": True,
        }

    def health_check(self) -> tuple:
        if not self._base_url:
            return False, "未配置服务地址 (Base URL)"
        try:
            health_ep = self._endpoints.get("health", "/health")
            self.request("GET", health_ep, timeout=5, retries=0, skip_monitor=True)
            return True, "REST服务可用"
        except Exception as e:
            return False, str(e)


# ============================================================================
# SOAP WebService 客户端（简化版）
# ============================================================================

class SOAPClient:
    """简单SOAP WebService客户端"""

    def __init__(self, wsdl_url: str, service_url: str = "",
                 namespace: str = "http://tempuri.org/",
                 auth: Optional[AuthProvider] = None,
                 timeout: int = 30):
        self.wsdl_url = wsdl_url
        self.service_url = service_url or wsdl_url.split("?")[0]
        self.namespace = namespace
        self.auth = auth or NoAuth()
        self.timeout = timeout

    def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """调用SOAP方法"""
        params = params or {}
        # 构造SOAP Envelope
        envelope = ET.Element("{http://schemas.xmlsoap.org/soap/envelope/}Envelope",
                              attrib={"{http://www.w3.org/2000/xmlns/}xsi":
                                      "http://www.w3.org/2001/XMLSchema-instance",
                                      "{http://www.w3.org/2000/xmlns/}xsd":
                                      "http://www.w3.org/2001/XMLSchema"})
        body = ET.SubElement(envelope, "{http://schemas.xmlsoap.org/soap/envelope/}Body")
        method_el = ET.SubElement(body, f"{{{self.namespace}}}{method}")
        for k, v in params.items():
            el = ET.SubElement(method_el, f"{{{self.namespace}}}{k}")
            el.text = str(v) if v is not None else ""

        xml_str = ET.tostring(envelope, encoding="unicode", xml_declaration=True)
        req = urllib.request.Request(
            self.service_url,
            data=xml_str.encode("utf-8"),
            headers={
                "Content-Type": "text/xml; charset=utf-8",
                "SOAPAction": f'"{self.namespace}{method}"',
                **self.auth.get_headers(),
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            resp_body = resp.read().decode("utf-8")
        return self._parse_soap_response(resp_body, method)

    @staticmethod
    def _parse_soap_response(xml_str: str, method: str) -> Dict[str, Any]:
        """解析SOAP响应为字典（简化解析）"""
        result: Dict[str, Any] = {}
        try:
            root = ET.fromstring(xml_str)
            # 去除命名空间
            for elem in root.iter():
                if "}" in elem.tag:
                    elem.tag = elem.tag.split("}", 1)[1]
            # 查找Body → methodResponse → Result
            body = root.find("Body")
            if body is None:
                return result
            resp_el = body.find(f"{method}Response") or body.find(f"{method}Result") or body
            for child in resp_el:
                if len(child) > 0:
                    # 嵌套对象
                    sub = {}
                    for sub_child in child:
                        sub[sub_child.tag] = sub_child.text
                    result[child.tag] = sub
                else:
                    result[child.tag] = child.text
            # 如果返回的是JSON字符串，解析之
            if "Result" in result and isinstance(result["Result"], str):
                try:
                    return json.loads(result["Result"])
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            LOGGER.error("SOAP响应解析失败: %s", e)
        return result
