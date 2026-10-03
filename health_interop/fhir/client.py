#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FHIR R4 REST API 客户端

支持：
- OAuth2 客户端凭证模式认证
- 按患者ID按需查询 Patient/Condition/Observation/MedicationStatement/DiagnosticReport
- 增量拉取（_since参数，首次拉取近N年历史，之后只拉新增）
- Bundle分页自动遍历
- 超时/重试/熔断（通过InterfaceMonitor）
"""

from __future__ import annotations

import datetime
import json
import logging
import ssl
import time
import urllib.parse
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from ..base import (
    HealthcareAdapter, AdapterConfig, AdapterResult,
    TERMINOLOGY_SYSTEMS, FHIR_RESOURCE_MAPPING, LOINC_TB_PANELS,
    timestamp_to_datetime, safe_parse_datetime,
)
from .parser import FHIRResourceParser
from .terminology import FHIRTermMapper

LOGGER = logging.getLogger("tb_risk.health_interop.fhir.client")


class FHIRClient(HealthcareAdapter):
    """HL7 FHIR R4 客户端适配器"""

    ADAPTER_TYPE = "fhir"

    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config)
        self._base_url = self.config.base_url.rstrip("/") if self.config.base_url else ""
        self._token: Optional[str] = None
        self._token_expires_at: float = 0
        self._parser = FHIRResourceParser()
        self._term_mapper = FHIRTermMapper()
        self._ssl_warned = False

    # ---- 连接与认证 ----

    def connect(self) -> bool:
        """建立连接：OAuth2 客户端凭证模式或API Key"""
        if not self._base_url:
            self._logger.error("FHIR base_url 未配置")
            return False
        try:
            if self.config.auth_type == "oauth2":
                self._fetch_oauth_token()
            # 简单连通性测试：读取CapabilityStatement
            self._request("GET", "metadata", timeout=self.config.connect_timeout)
            self._logger.info("FHIR 服务器连接成功: %s", self._base_url)
            return True
        except Exception as e:
            self._logger.error("FHIR 连接失败: %s", e)
            return False

    def _fetch_oauth_token(self):
        """获取OAuth2 access_token"""
        if self._token and time.time() < self._token_expires_at - 60:
            return
        token_url = self.config.vendor_params.get("token_url", "")
        if not token_url:
            self._logger.warning("未配置 token_url，跳过OAuth2认证")
            return
        try:
            data = urllib.parse.urlencode({
                "grant_type": "client_credentials",
                "client_id": self.config.username,
                "client_secret": self.config.password,
            }).encode("utf-8")
            req = urllib.request.Request(token_url, data=data,
                                         headers={"Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=self.config.connect_timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
            self._token = resp_data.get("access_token")
            expires_in = int(resp_data.get("expires_in", 3600))
            self._token_expires_at = time.time() + expires_in
            self._logger.info("OAuth2 token 获取成功，有效期 %d 秒", expires_in)
        except Exception as e:
            self._logger.error("OAuth2 token 获取失败: %s", e)
            raise

    # ---- HTTP 请求封装 ----

    def _request(self, method: str, path: str,
                 params: Optional[Dict[str, Any]] = None,
                 data: Optional[Dict] = None,
                 timeout: Optional[int] = None,
                 retries: Optional[int] = None) -> Dict[str, Any]:
        """发送FHIR HTTP请求，带重试和熔断检查"""
        if not self._base_url:
            raise RuntimeError("FHIR客户端未配置base_url，无法发送请求")
        timeout = timeout or self.config.read_timeout
        retries = retries if retries is not None else self.config.max_retries

        endpoint = f"{method} {path}"
        url = f"{self._base_url}/{path.lstrip('/')}"
        if params:
            qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
            url = f"{url}?{qs}"

        if self._monitor and not self._monitor.allow_request(self.ADAPTER_TYPE, endpoint):
            raise RuntimeError(f"FHIR 接口已熔断，拒绝请求: {endpoint}")

        headers = {
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        elif self.config.api_key:
            headers["X-API-Key"] = self.config.api_key

        last_error = None
        for attempt in range(retries + 1):
            start = time.time()
            try:
                body = json.dumps(data).encode("utf-8") if data else None
                req = urllib.request.Request(url, data=body, headers=headers, method=method)
                ctx = None
                if not self.config.ssl_verify:
                    if not self._ssl_warned:
                        LOGGER.warning(
                            "FHIR客户端已禁用SSL证书验证（ssl_verify=False），"
                            "存在中间人攻击风险，仅建议在测试环境使用。")
                        self._ssl_warned = True
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                    resp_body = resp.read().decode("utf-8")
                    duration_ms = (time.time() - start) * 1000
                    result = json.loads(resp_body) if resp_body else {}
                    self._record_success(endpoint, duration_ms)
                    return result
            except urllib.error.HTTPError as e:
                duration_ms = (time.time() - start) * 1000
                err_body = e.read().decode("utf-8", errors="replace")[:500] if e.fp else ""
                last_error = f"HTTP {e.code}: {err_body}"
                # 4xx 不重试（除429限流）
                if 400 <= e.code < 500 and e.code != 429:
                    self._record_failure(endpoint, last_error, duration_ms)
                    raise RuntimeError(last_error)
                self._logger.warning("FHIR请求失败(第%d次): %s", attempt + 1, last_error)
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                last_error = str(e)
                self._logger.warning("FHIR请求异常(第%d次): %s", attempt + 1, e)
            if attempt < retries:
                wait = self.config.retry_backoff ** attempt
                time.sleep(wait)

        self._record_failure(endpoint, last_error or "未知错误")
        raise RuntimeError(f"FHIR请求最终失败: {last_error}")

    # ---- 患者数据拉取 ----

    def fetch_patient(self, patient_id: str) -> AdapterResult:
        """按需查询单个患者的完整22维特征所需资源"""
        result = AdapterResult(
            source_system="FHIR",
            patient_id=patient_id,
            last_sync_time=datetime.datetime.now(),
        )
        start = time.time()
        try:
            # 1. Patient 资源（基础信息）
            patient = self._request("GET", f"Patient/{patient_id}")
            self._parser.parse_patient(patient, result)

            # 2. Condition 资源（既往病史、合并症、结核诊断）
            since = self._history_since_param()
            self._fetch_conditions(patient_id, result, since)

            # 3. Observation 资源（检验结果、症状体征）
            self._fetch_observations(patient_id, result, since)

            # 4. MedicationStatement 资源（用药史）
            self._fetch_medications(patient_id, result, since)

            # 5. DiagnosticReport 资源（影像报告）
            self._fetch_diagnostic_reports(patient_id, result, since)

            duration_ms = (time.time() - start) * 1000
            result.success = True
            result.records_processed = (
                len(result.diagnoses) + len(result.lab_results)
                + len(result.imaging_reports) + len(result.medications)
            )
            self._record_success(f"fetch_patient:{patient_id}", duration_ms,
                                 result.records_processed)
            self._last_sync_time = datetime.datetime.now()
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            result.errors.append(str(e))
            self._record_failure(f"fetch_patient:{patient_id}", str(e), duration_ms)
            self._logger.error("拉取患者 %s 数据失败: %s", patient_id, e, exc_info=True)

        return result

    def _history_since_param(self) -> Optional[str]:
        """构造 _since 参数：首次拉取近N年历史，之后用上次同步时间"""
        if self._last_sync_time:
            return self._last_sync_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        if self.config.history_years > 0:
            since = datetime.datetime.now() - datetime.timedelta(
                days=365 * self.config.history_years)
            return since.strftime("%Y-%m-%dT%H:%M:%SZ")
        return None

    def _fetch_conditions(self, patient_id: str, result: AdapterResult,
                          since: Optional[str]):
        """拉取 Condition 资源"""
        try:
            params = {"patient": patient_id, "_count": 100}
            if since:
                params["_lastUpdated"] = f"ge{since}"
            bundle = self._request("GET", "Condition", params=params)
            resources = self._iterate_bundle(bundle)
            for res in resources:
                diag = self._parser.parse_condition(res)
                if diag:
                    result.diagnoses.append(diag)
        except Exception as e:
            self._logger.warning("拉取 Condition 失败: %s", e)
            result.warnings.append(f"Condition拉取失败: {e}")

    def _fetch_observations(self, patient_id: str, result: AdapterResult,
                            since: Optional[str]):
        """拉取 Observation 资源（检验相关LOINC编码）"""
        # 构造结核相关LOINC编码列表
        loinc_codes = set()
        for category in LOINC_TB_PANELS.values():
            for item in category:
                loinc_codes.add(item["loinc"])
        code_param = ",".join(
            f"http://loinc.org|{code}" for code in loinc_codes) if loinc_codes else None
        try:
            params = {"patient": patient_id, "_count": 200}
            if code_param:
                params["code"] = code_param
            if since:
                params["_lastUpdated"] = f"ge{since}"
            bundle = self._request("GET", "Observation", params=params)
            resources = self._iterate_bundle(bundle)
            for res in resources:
                lab = self._parser.parse_observation(res, self._term_mapper)
                if lab:
                    result.lab_results.append(lab)
        except Exception as e:
            self._logger.warning("拉取 Observation 失败: %s", e)
            result.warnings.append(f"Observation拉取失败: {e}")

    def _fetch_medications(self, patient_id: str, result: AdapterResult,
                           since: Optional[str]):
        """拉取 MedicationStatement 资源"""
        try:
            params = {"patient": patient_id, "_count": 100}
            if since:
                params["_lastUpdated"] = f"ge{since}"
            bundle = self._request("GET", "MedicationStatement", params=params)
            resources = self._iterate_bundle(bundle)
            for res in resources:
                med = self._parser.parse_medication(res)
                if med:
                    result.medications.append(med)
        except Exception as e:
            self._logger.warning("拉取 MedicationStatement 失败: %s", e)
            result.warnings.append(f"MedicationStatement拉取失败: {e}")

    def _fetch_diagnostic_reports(self, patient_id: str, result: AdapterResult,
                                  since: Optional[str]):
        """拉取 DiagnosticReport 资源（影像报告，LOINC放射编码）"""
        try:
            params = {
                "patient": patient_id,
                "category": "http://terminology.hl7.org/CodeSystem/v2-0074|RAD",
                "_count": 100,
            }
            if since:
                params["_lastUpdated"] = f"ge{since}"
            bundle = self._request("GET", "DiagnosticReport", params=params)
            resources = self._iterate_bundle(bundle)
            for res in resources:
                report = self._parser.parse_diagnostic_report(res)
                if report:
                    result.imaging_reports.append(report)
        except Exception as e:
            self._logger.warning("拉取 DiagnosticReport 失败: %s", e)
            result.warnings.append(f"DiagnosticReport拉取失败: {e}")

    def _iterate_bundle(self, bundle: Dict) -> List[Dict]:
        """遍历FHIR Bundle分页，收集所有entry.resource"""
        resources: List[Dict] = []
        current = bundle
        visited = set()
        while current:
            if current.get("resourceType") == "Bundle":
                for entry in current.get("entry", []) or []:
                    res = entry.get("resource")
                    if res and res.get("resourceType"):
                        rid = res.get("id", "")
                        if rid not in visited:
                            visited.add(rid)
                            resources.append(res)
                # 下一页
                next_url = None
                for link in current.get("link", []) or []:
                    if link.get("relation") == "next":
                        next_url = link.get("url")
                        break
                if next_url and next_url.startswith(self._base_url):
                    try:
                        path = next_url[len(self._base_url):].lstrip("/")
                        current = self._request("GET", path)
                    except Exception as e:
                        self._logger.warning("Bundle分页遍历失败: %s", e)
                        break
                else:
                    break
            else:
                # 单资源而非Bundle
                resources.append(current)
                break
        return resources

    # ---- 增量拉取 ----

    def fetch_incremental(self, since: Optional[datetime.datetime] = None
                          ) -> List[AdapterResult]:
        """增量拉取新数据（通过FHIR _lastUpdated参数 + Subscription webhook触发）

        这里实现轮询模式：查询自since以来有更新的患者列表，然后逐个fetch_patient。
        实际生产中建议结合FHIR Subscription（见 subscription.py）做推送通知。
        """
        if since is None:
            since = self._last_sync_time
        if since is None:
            since = datetime.datetime.now() - datetime.timedelta(
                days=365 * self.config.history_years)

        results: List[AdapterResult] = []
        try:
            # 查找自since以来有新Condition/Observation/DiagnosticReport的患者
            since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")
            patient_ids = set()
            for resource_type in ("Condition", "Observation", "DiagnosticReport"):
                try:
                    params = {
                        "_lastUpdated": f"ge{since_str}",
                        "_count": 500,
                        "_elements": "subject",
                    }
                    bundle = self._request("GET", resource_type, params=params)
                    for res in self._iterate_bundle(bundle):
                        subj = res.get("subject", {}) or {}
                        ref = subj.get("reference", "")
                        if ref.startswith("Patient/"):
                            pid = ref.split("/", 1)[1]
                            patient_ids.add(pid)
                except Exception as e:
                    self._logger.warning("增量查询 %s 失败: %s", resource_type, e)

            self._logger.info("增量拉取发现 %d 名患者有新数据", len(patient_ids))
            for pid in patient_ids:
                r = self.fetch_patient(pid)
                if r.success:
                    results.append(r)
            self._last_sync_time = datetime.datetime.now()
        except Exception as e:
            self._logger.error("增量拉取失败: %s", e, exc_info=True)

        return results

    def health_check(self) -> tuple:
        try:
            self._request("GET", "metadata", timeout=5, retries=0)
            return True, "FHIR服务可用"
        except Exception as e:
            return False, str(e)
