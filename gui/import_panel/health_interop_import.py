#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 医疗标准接口导入（HL7 FHIR / HL7 v2.x / 数据库视图 / LIS-PACS / 疾控 / REST）

本模块通过 health_interop 包提供的六大适配器对接医院HIS/EMR/LIS/PACS/疾控系统，
实现符合《医疗信息标准接口适配》规范的标准化数据接入。

使用方法（与 ApiImportMixin 相同的入口风格）：
    self._load_from_health_interop()  # 弹出配置对话框，选择接口类型并导入
    self.generate_tb_report_card()    # 基于当前数据自动生成传染病报告卡
"""

from ._shared import *  # noqa: F401,F403
from .pipeline import ImportPipeline
from typing import Dict, Any, List, Optional, Tuple


# ============================================================================
# 接口类型定义
# ============================================================================

ADAPTER_TYPES = {
    "fhir": {
        "label": "HL7 FHIR R4",
        "desc": "支持FHIR标准的HIS/EMR系统（按需查询+订阅通知）",
        "available": True,
        "supports_enhancement": True,
        "params": [
            ("base_url", "FHIR服务地址 (Base URL)", "text", "https://fhir.hospital.example.com/fhir"),
            ("auth_type", "认证方式", "choice", ["none", "basic", "oauth2", "api_key"]),
            ("username", "用户名 (Basic Auth)", "text", ""),
            ("password", "密码 (Basic Auth)", "password", ""),
            ("token", "Token / API Key", "password", ""),
            ("token_url", "OAuth2 Token URL", "text", ""),
            ("client_id", "OAuth2 Client ID", "text", ""),
            ("history_years", "拉取历史年数", "text", "5"),
        ],
    },
    "hl7v2": {
        "label": "HL7 v2.x MLLP",
        "desc": "通过MLLP协议接收医院集成平台推送的消息（常驻监听服务）",
        "available": False,
        "coming_soon": "请在「系统设置 → 接口服务」中配置并启动HL7 MLLP后台监听",
        "params": [
            ("host", "监听地址 (Host)", "text", "0.0.0.0"),
            ("port", "监听端口 (Port)", "text", "2575"),
            ("data_dir", "消息持久化目录", "dir", "hl7_messages"),
        ],
    },
    "db_direct": {
        "label": "数据库直连（只读视图）",
        "desc": "兼容老旧系统的最后手段，通过信息科提供的只读视图增量拉取",
        "available": True,
        "supports_enhancement": True,
        "params": [
            ("db_type", "数据库类型", "choice", ["mysql", "postgresql", "sqlite"]),
            ("host", "主机地址", "text", "localhost"),
            ("port", "端口", "text", "3306"),
            ("database", "数据库名", "text", "hospital_emr"),
            ("username", "只读账号", "text", "tb_risk_readonly"),
            ("password", "密码", "password", ""),
            ("patient_view", "患者基本信息视图", "text", "v_tb_patient"),
            ("visit_view", "就诊视图", "text", "v_tb_visit"),
            ("lab_view", "检验结果视图", "text", "v_tb_lab_result"),
        ],
    },
    "cdc": {
        "label": "疾控中心对接",
        "desc": "传染病报告卡自动填报 + 疾控回传数据接收",
        "available": False,
        "coming_soon": "确诊后请使用「生成传染病报告卡」功能进行上报，回传接收需部署Webhook",
        "params": [
            ("report_unit", "报告单位名称", "text", "XX医院"),
            ("report_doctor", "报告医生", "text", ""),
            ("callback_url", "回传接收URL (Webhook)", "text", ""),
            ("sign_secret", "签名密钥", "password", ""),
            ("region_code", "行政区划代码", "text", ""),
        ],
    },
    "rest_generic": {
        "label": "通用 REST / WebService",
        "desc": "厂商自定义REST/SOAP接口适配（OAuth2/API Key/签名认证）",
        "available": True,
        "supports_enhancement": True,
        "params": [
            ("base_url", "API Base URL", "text", "https://api.hospital.example.com/"),
            ("auth_type", "认证方式", "choice", ["none", "api_key", "oauth2", "signature"]),
            ("api_key", "API Key", "password", ""),
            ("app_id", "签名 AppID", "text", ""),
            ("app_secret", "签名 AppSecret", "password", ""),
            ("token_url", "OAuth2 Token URL", "text", ""),
            ("client_id", "OAuth2 Client ID", "text", ""),
            ("patient_endpoint", "患者查询端点 (含{id})", "text", "/api/patient/{id}"),
            ("lab_endpoint", "检验端点 (含{id})", "text", ""),
        ],
    },
}


# ============================================================================
# 配置对话框
# ============================================================================

class HealthInteropConfigDialog(tk.Toplevel):
    """医疗标准接口配置对话框"""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("医疗标准接口接入配置")
        self.geometry("640x720")
        self.transient(parent)
        self.grab_set()
        self.result = None
        self._param_vars: Dict[str, tk.Variable] = {}
        self._params_frame = None
        self._confirm_btn = None
        self._coming_soon_label = None
        self._enhancement_frame = None
        self._enable_lis_var = None
        self._enable_pacs_var = None
        self._build_ui()

    def _build_ui(self):
        # 顶部说明
        info = ttk.Label(
            self,
            text="选择要对接的医院信息系统类型，填写连接参数。\n"
                 "所有连接均通过医院信息科授权，遵循只读、脱敏、加密安全规范。",
            font=('Microsoft YaHei', 9), foreground='#5a6c7d', wraplength=600, justify='left')
        info.pack(pady=(15, 10), padx=20, anchor='w')

        # 接口类型选择
        type_frame = ttk.LabelFrame(self, text="接口类型", padding=10)
        type_frame.pack(fill='x', padx=20, pady=5)
        self._adapter_var = tk.StringVar(value='fhir')
        self._radio_buttons = {}
        for i, (key, info) in enumerate(ADAPTER_TYPES.items()):
            col = i % 3
            row = i // 3
            available = info.get("available", True)
            label_text = info["label"]
            if not available:
                label_text = info["label"] + "（即将推出）"
            rb = ttk.Radiobutton(
                type_frame, text=label_text, variable=self._adapter_var, value=key,
                command=self._refresh_params,
                state='normal' if available else 'disabled')
            rb.grid(row=row, column=col, sticky='w', padx=10, pady=3)
            self._radio_buttons[key] = rb
        type_frame.columnconfigure(0, weight=1)
        type_frame.columnconfigure(1, weight=1)
        type_frame.columnconfigure(2, weight=1)

        # 即将推出提示区域
        self._coming_soon_frame = ttk.Frame(self)
        self._coming_soon_frame.pack(fill='x', padx=20, pady=2)
        self._coming_soon_label = ttk.Label(
            self._coming_soon_frame, text="",
            font=('Microsoft YaHei', 9), foreground='#856404',
            wraplength=580, justify='left')
        self._coming_soon_label.pack(fill='x')

        # 参数容器
        self._params_container = ttk.LabelFrame(self, text="连接参数", padding=10)
        self._params_container.pack(fill='both', expand=True, padx=20, pady=5)

        # 患者ID输入（按需查询）
        query_frame = ttk.LabelFrame(self, text="患者查询", padding=10)
        query_frame.pack(fill='x', padx=20, pady=5)
        ttk.Label(query_frame, text="患者ID (留空则进行增量拉取):",
                  font=('Microsoft YaHei', 9)).pack(anchor='w')
        self._patient_id_var = tk.StringVar()
        ttk.Entry(query_frame, textvariable=self._patient_id_var, width=40).pack(fill='x', pady=3)
        ttk.Label(query_frame, text="提示：输入EMPI/住院号/门诊号，打开患者页时按需查询",
                  font=('Microsoft YaHei', 8), foreground='#5a6c7d').pack(anchor='w')

        # LIS/PACS增强选项
        self._enhancement_frame = ttk.LabelFrame(self, text="数据增强选项 (LIS/PACS)", padding=10)
        self._enhancement_frame.pack(fill='x', padx=20, pady=5)
        self._enable_lis_var = tk.BooleanVar(value=True)
        self._enable_pacs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self._enhancement_frame, text="启用LIS检验项目标准化映射（五大类结核相关检验结果解释、历史时间线分析）",
                        variable=self._enable_lis_var).pack(anchor='w', pady=2)
        ttk.Checkbutton(self._enhancement_frame, text="启用PACS影像报告NLP征象提取（病变部位/性质/范围/播散自动识别）",
                        variable=self._enable_pacs_var).pack(anchor='w', pady=2)
        ttk.Label(self._enhancement_frame, text="提示：勾选后将自动增强检验和影像数据解析，提升风险评估准确性",
                  font=('Microsoft YaHei', 8), foreground='#5a6c7d').pack(anchor='w', pady=(3,0))

        # 按钮
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill='x', padx=20, pady=15)
        ttk.Button(btn_frame, text="测试连接", command=self._test_connection).pack(side='left')
        self._status_label = ttk.Label(btn_frame, text="", font=('Microsoft YaHei', 9))
        self._status_label.pack(side='left', padx=10)
        ttk.Button(btn_frame, text="取消", command=self.destroy).pack(side='right', padx=5)
        self._confirm_btn = ttk.Button(btn_frame, text="开始导入", command=self._on_confirm)
        self._confirm_btn.pack(side='right', padx=5)

        self._refresh_params()

    def _refresh_params(self):
        """根据选择的接口类型刷新参数输入框"""
        for w in self._params_container.winfo_children():
            w.destroy()
        self._param_vars.clear()

        adapter_key = self._adapter_var.get()
        spec = ADAPTER_TYPES.get(adapter_key, {})
        params = spec.get("params", [])
        available = spec.get("available", True)
        coming_soon = spec.get("coming_soon", "")
        supports_enhancement = spec.get("supports_enhancement", False)

        # 显示/隐藏"即将推出"提示
        if not available and coming_soon:
            self._coming_soon_label.config(text=f"ℹ {coming_soon}")
        else:
            self._coming_soon_label.config(text="")

        # 显示/隐藏增强选项
        if supports_enhancement and available:
            self._enhancement_frame.pack(fill='x', padx=20, pady=5)
        else:
            self._enhancement_frame.pack_forget()

        # 不可用适配器禁用确认和测试按钮
        if self._confirm_btn:
            self._confirm_btn.config(state='normal' if available else 'disabled')

        # 不可用适配器不显示参数输入区
        if not available:
            ttk.Label(self._params_container,
                      text="该功能正在开发中，敬请期待。",
                      font=('Microsoft YaHei', 10), foreground='#856404').pack(pady=30)
            return

        canvas_frame = ttk.Frame(self._params_container)
        canvas_frame.pack(fill='both', expand=True)
        canvas = tk.Canvas(canvas_frame, height=200, highlightthickness=0, bg='#ffffff')
        scrollbar = ttk.Scrollbar(canvas_frame, orient='vertical', command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        for i, (p_key, label, p_type, default) in enumerate(params):
            ttk.Label(inner, text=label + ":", font=('Microsoft YaHei', 9)).grid(
                row=i, column=0, sticky='w', pady=2, padx=(0, 8))
            if p_type == "choice":
                var = tk.StringVar(value=default[0] if isinstance(default, list) else default)
                w = ttk.Combobox(inner, textvariable=var, values=default, state='readonly', width=30)
            elif p_type == "password":
                var = tk.StringVar(value=default or '')
                w = ttk.Entry(inner, textvariable=var, width=35, show='*')
            elif p_type == "check":
                var = tk.BooleanVar(value=bool(default))
                w = ttk.Checkbutton(inner, variable=var, text="启用")
            elif p_type == "dir":
                var = tk.StringVar(value=default or '')
                w_frame = ttk.Frame(inner)
                w_frame.grid(row=i, column=1, sticky='ew', pady=2)
                w = ttk.Entry(w_frame, textvariable=var, width=30)
                w.pack(side='left')
                def browse(v=var):
                    d = filedialog.askdirectory()
                    if d:
                        v.set(d)
                ttk.Button(w_frame, text="浏览...", command=browse, width=6).pack(side='left', padx=3)
                self._param_vars[p_key] = var
                continue
            else:
                var = tk.StringVar(value=default or '')
                w = ttk.Entry(inner, textvariable=var, width=35)
            w.grid(row=i, column=1, sticky='ew', pady=2)
            self._param_vars[p_key] = var
        inner.columnconfigure(1, weight=1)

    def _get_config(self) -> Dict[str, Any]:
        adapter_key = self._adapter_var.get()
        params = {k: v.get() for k, v in self._param_vars.items()}
        params["adapter_type"] = adapter_key
        params["patient_id"] = self._patient_id_var.get().strip()
        params["enable_lis_enhancement"] = self._enable_lis_var.get() if self._enable_lis_var else False
        params["enable_pacs_enhancement"] = self._enable_pacs_var.get() if self._enable_pacs_var else False
        return params

    def _validate_and_convert_params(self, params: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """验证并转换参数类型，返回 (ok, error_msg, converted_params)"""
        converted = dict(params)
        
        # 整数参数验证与转换
        int_fields = {
            "port": (1, 65535, "端口号必须是1-65535之间的有效整数"),
            "history_years": (1, 20, "历史年数必须是1-20之间的有效整数"),
        }
        
        for field, (min_val, max_val, err_msg) in int_fields.items():
            if field in converted:
                val = converted[field]
                if val == "" or val is None:
                    # 空值使用合理默认值
                    defaults = {"port": 0, "history_years": 5}
                    converted[field] = defaults.get(field, 0)
                    continue
                try:
                    ival = int(val)
                    if field == "port" and ival == 0:
                        converted[field] = 0  # 0表示使用协议默认端口，允许
                    elif ival < min_val or ival > max_val:
                        return False, err_msg, params
                    else:
                        converted[field] = ival
                except (TypeError, ValueError):
                    return False, err_msg, params
        
        # 布尔参数转换
        for bool_field in ("enable_nlp",):
            if bool_field in converted and isinstance(converted[bool_field], str):
                converted[bool_field] = converted[bool_field].lower() in ("1", "true", "yes", "on")
        
        return True, "", converted

    def _test_connection(self):
        """测试适配器连接（轻量验证）"""
        cfg = self._get_config()
        ok, err, cfg = self._validate_and_convert_params(cfg)
        if not ok:
            messagebox.showerror("参数错误", err)
            return
        self._status_label.config(text="正在测试...", foreground='#f39c12')
        self.update_idletasks()
        try:
            ok, msg = _test_adapter_connection(cfg)
            if ok:
                self._status_label.config(text=f"✓ {msg}", foreground='#2ecc71')
            else:
                self._status_label.config(text=f"✗ {msg}", foreground='#e74c3c')
        except Exception as e:
            self._status_label.config(text=f"✗ 异常: {e}", foreground='#e74c3c')

    def _on_confirm(self):
        cfg = self._get_config()
        ok, err, cfg = self._validate_and_convert_params(cfg)
        if not ok:
            messagebox.showerror("参数错误", err)
            return
        # 基础校验
        if cfg.get("adapter_type") in ("fhir", "rest_generic") and not cfg.get("base_url"):
            messagebox.showwarning("提示", "请填写服务地址")
            return
        self.result = cfg
        self.destroy()


def _test_adapter_connection(cfg: Dict[str, Any]) -> tuple:
    """测试适配器连接（返回 (ok, message)）"""
    from ... import health_interop
    atype = cfg.get("adapter_type")
    try:
        if atype == "fhir":
            from ...health_interop import FHIRClient, AdapterConfig
            base_url = cfg.get("base_url", "").strip()
            if not base_url:
                return False, "请填写FHIR服务地址"
            config = AdapterConfig(
                base_url=base_url,
                auth_type=cfg.get("auth_type", "none"),
                username=cfg.get("username", ""),
                password=cfg.get("password", ""),
                token=cfg.get("token", ""),
                connect_timeout=5, read_timeout=10, max_retries=1,
            )
            adapter = FHIRClient(config)
            connected = adapter.connect()
            return connected, (
                "FHIR服务连接成功" if connected else "FHIR服务不可达或认证失败"
            )
        elif atype == "hl7v2":
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            try:
                port = int(cfg.get("port", 2575))
                sock.bind((cfg.get("host", "0.0.0.0"), port))
                sock.close()
                return True, f"端口 {port} 可绑定，服务可启动"
            except (OSError, ValueError) as e:
                sock.close()
                return False, f"端口不可用: {e}"
        elif atype == "rest_generic":
            from ...health_interop import RESTClient, AdapterConfig, NoAuth, APIKeyAuth
            base_url = cfg.get("base_url", "").strip()
            if not base_url:
                return False, "请填写API Base URL"
            config = AdapterConfig(
                base_url=base_url,
                auth_type=cfg.get("auth_type", "none"),
                username=cfg.get("username", ""),
                password=cfg.get("password", ""),
                api_key=cfg.get("api_key", ""),
                connect_timeout=5, read_timeout=10, max_retries=1,
            )
            auth = APIKeyAuth(cfg.get("api_key", "")) if cfg.get("api_key") else NoAuth()
            client = RESTClient(config, auth=auth)
            ok, msg = client.health_check()
            return ok, msg
        elif atype in ("db_direct", "cdc"):
            return True, f"{ADAPTER_TYPES[atype]['label']} 参数已填写，" \
                         "导入时将实际建立连接"
        return False, "未知适配器类型"
    except Exception as e:
        return False, str(e)


# ============================================================================
# 导入管线
# ============================================================================

class HealthInteropImportPipeline(ImportPipeline):
    """医疗标准接口统一导入管线"""

    FORMAT_NAME = '医疗标准接口'
    SUPPORTS_MODE_SELECTION = True

    def __init__(self, app):
        super().__init__(app)
        self._adapter_result = None
        self._patient_info: Dict[str, Any] = {}

    def select_source(self) -> Optional[Dict[str, Any]]:
        dlg = HealthInteropConfigDialog(self.app.root)
        self.app.root.wait_window(dlg)
        return dlg.result

    def parse_data(self, source: Dict[str, Any], update_progress
                   ) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
        from ... import health_interop
        from ...health_interop.base import AdapterConfig

        atype = source.get("adapter_type")
        patient_id = source.get("patient_id", "")
        self._adapter_result = None

        update_progress(10, f"正在配置 {ADAPTER_TYPES[atype]['label']} 适配器...")

        # 构建 AdapterConfig
        config = AdapterConfig(
            base_url=source.get("base_url", ""),
            port=source.get("port", 0),
            username=source.get("username", ""),
            password=source.get("password", ""),
            api_key=source.get("api_key", ""),
            token=source.get("token", ""),
            auth_type=source.get("auth_type", "none"),
            history_years=source.get("history_years", 5),
            vendor_params=source,
        )

        adapter = None
        if atype == "fhir":
            adapter = health_interop.FHIRClient(config)
        elif atype == "rest_generic":
            from ...health_interop import RESTClient, NoAuth, APIKeyAuth, OAuth2ClientCredentials
            auth = NoAuth()
            if source.get("auth_type") == "api_key" and source.get("api_key"):
                auth = APIKeyAuth(source["api_key"])
            elif source.get("auth_type") == "oauth2":
                auth = OAuth2ClientCredentials(
                    token_url=source.get("token_url", ""),
                    client_id=source.get("client_id", ""),
                    client_secret=source.get("password", ""),
                )
            endpoints = {}
            if source.get("patient_endpoint"):
                endpoints["patient"] = source["patient_endpoint"]
            if source.get("lab_endpoint"):
                endpoints["labs"] = source["lab_endpoint"]
            adapter = RESTClient(config, auth=auth, endpoints=endpoints)
        elif atype == "cdc":
            messagebox.showinfo(
                "提示",
                "疾控中心对接模式：\n"
                "1) 使用'生成传染病报告卡'功能自动填报\n"
                "2) 回传数据接收需要部署Webhook，请参考实施文档。")
            return [], [], {"source": "cdc"}
        elif atype == "hl7v2":
            messagebox.showinfo(
                "提示",
                "HL7 v2.x MLLP 消息服务为常驻后台监听模式。\n"
                "请在'系统设置 → 接口配置'中启动监听服务。\n"
                "本次导入不通过MLLP拉取数据。")
            return [], [], {"source": "hl7v2"}
        elif atype == "db_direct":
            adapter = health_interop.DatabaseDirectAdapter(config)
            # 补充视图配置到 vendor_params
            config.vendor_params.update({
                "patient_view": source.get("patient_view", ""),
                "visit_view": source.get("visit_view", ""),
                "lab_view": source.get("lab_view", ""),
                "db_type": source.get("db_type", ""),
                "host": source.get("host", ""),
                "database": source.get("database", ""),
            })
        else:
            return [], [], {}

        if adapter is None:
            return [], [], {}

        # 如果启用了LIS/PACS增强，用装饰器包装基础适配器
        enable_lis = source.get("enable_lis_enhancement", False)
        enable_pacs = source.get("enable_pacs_enhancement", False)
        if (enable_lis or enable_pacs) and atype in ("fhir", "db_direct", "rest_generic"):
            from ...health_interop import LISPACSDecorator
            adapter = LISPACSDecorator(adapter, enable_lis=enable_lis, enable_pacs=enable_pacs)
            update_progress(15, f"已启用LIS/PACS增强 (LIS:{'✓' if enable_lis else '✗'}, PACS:{'✓' if enable_pacs else '✗'})...")

        update_progress(20, "正在建立连接...")
        try:
            connected = adapter.connect()
            if not connected:
                messagebox.showerror("错误", "适配器连接失败，请检查配置")
                return [], [], {}
        except Exception as e:
            messagebox.showerror("错误", f"连接失败: {e}")
            return [], [], {}

        update_progress(40, "正在拉取患者数据...")
        results: List[health_interop.AdapterResult] = []
        if patient_id:
            r = adapter.fetch_patient(patient_id)
            if r.success:
                results.append(r)
            else:
                errs = "; ".join(r.errors) or "未找到数据"
                messagebox.showerror("错误", f"拉取患者数据失败: {errs}")
                return [], [], {}
        else:
            update_progress(50, "正在增量拉取新数据...")
            results = adapter.fetch_incremental()
            if not results:
                messagebox.showinfo("提示", "没有新的增量数据（可输入患者ID进行按需查询）")
                return [], [], {}

        update_progress(70, "正在处理数据...")

        # 汇总所有结果（支持单患者或增量多患者，但导入面板只支持单患者，取第一个）
        if not results:
            return [], [], {}
        primary = results[0]
        self._adapter_result = primary
        self._patient_info = primary.patient_info or {}

        # 患者基本信息 → basic_info_vars（稍后在 _load_data 中填充）
        # 接触者：family_contacts / social_contacts
        family_entries = []
        social_entries = []
        for c in (primary.family_contacts or []):
            family_entries.append(_standardize_contact(c, "family"))
        for c in (primary.social_contacts or []):
            social_entries.append(_standardize_contact(c, "social"))

        # 从LIS-PACS回传中提取的密切接触者（如果有配置LIS/PACS增强）
        # 疾控回传的密切接触者合并到社会接触
        # （实际实施中，CDC回传会通过其他通道接收）

        metadata = {
            "source": atype,
            "source_system": primary.source_system,
            "patient_info": self._patient_info,
            "lab_results": primary.lab_results,
            "diagnoses": primary.diagnoses,
            "medications": primary.medications,
            "imaging_reports": primary.imaging_reports,
            "last_sync_time": primary.last_sync_time.isoformat()
                if primary.last_sync_time else None,
            "unmapped_codes": primary.unmapped_codes,
            "warnings": primary.warnings,
        }

        update_progress(90, "数据拉取完成")
        return family_entries, social_entries, metadata

    def _load_data(self, family_entries, social_entries, mode):
        """重写_load_data：先加载接触者，再将patient_info填入basic_info_vars"""
        ok = super()._load_data(family_entries, social_entries, mode)
        if not ok:
            return False

        # 填充患者基本信息到 basic_info_vars
        pi = self._patient_info
        if not pi:
            return True

        basic_vars = getattr(self.app, 'basic_info_vars', {})

        # 性别映射到已有选项
        gender = pi.get("gender", "")
        if gender and "gender" in basic_vars:
            g_map = {"male": 1, "female": 0}  # 根据系统约定调整
            v = g_map.get(gender, 0)
            if hasattr(basic_vars["gender"], 'set'):
                basic_vars["gender"].set(v)

        # 年龄
        age = pi.get("age")
        if age is not None and "age" in basic_vars:
            try:
                basic_vars["age"].set(int(age))
            except (TypeError, ValueError):
                pass

        # 姓名（如果有对应字段）
        name = pi.get("name", "")
        if name and "patient_name" in basic_vars:
            basic_vars["patient_name"].set(name)
        elif name and hasattr(self.app, 'patient_name_var'):
            self.app.patient_name_var.set(name)

        # 从lab_results/diagnoses/imaging自动推导临床布尔字段
        if self._adapter_result:
            self._apply_clinical_features(self._adapter_result)

        # 提示未映射编码
        unmapped = self._adapter_result.unmapped_codes if self._adapter_result else []
        if unmapped:
            msg = f"发现 {len(unmapped)} 个未映射编码，请联系实施人员补充术语映射：\n"
            msg += "\n".join(f"- [{u.get('system','')}] {u.get('code','')}: {u.get('display','')}"
                            for u in unmapped[:10])
            if len(unmapped) > 10:
                msg += f"\n... 还有 {len(unmapped)-10} 个"
            messagebox.showwarning("提示（未映射编码）", msg)

        # 提示警告信息
        warnings = self._adapter_result.warnings if self._adapter_result else []
        if warnings:
            LOGGER.warning("接口导入警告: %s", warnings)

        return True

    def _apply_clinical_features(self, result):
        """根据检验/影像/诊断结果自动设置临床布尔字段"""
        from ...health_interop.lis_pacs import LISPACSAdapter
        basic_vars = getattr(self.app, 'basic_info_vars', {})

        # 使用LISPACSAdapter提取风险特征（已通过装饰器处理的结果可直接用risk_features）
        lis_pacs = LISPACSAdapter()
        features = {}
        if result.lab_results:
            features = lis_pacs.extract_risk_features_from_labs(result.lab_results)

        # 合并risk_features（装饰器已处理过的特征）
        if hasattr(result, 'risk_features') and result.risk_features:
            features.update(result.risk_features)

        # 影像空洞检测
        has_cavity = features.get("has_cavity_any", False)
        for img in (result.imaging_reports or []):
            nlp_feats = img.get("nlp_features", {})
            if nlp_feats.get("has_cavity"):
                has_cavity = True
                break

        # 设置布尔字段（约定：1=阳性/是，0=阴性/否，2=未知，按系统约定）
        bool_map = {
            "sputum_smear": features.get("sputum_smear_positive", False),
            "has_cavity": has_cavity,
            "diabetes": features.get("diabetes_risk", False),
            "hiv": features.get("hiv_positive", False),
            "xpert": features.get("xpert_positive", False),
            "tspot": features.get("tspot_positive", False),
        }
        for field, is_true in bool_map.items():
            if field in basic_vars and hasattr(basic_vars[field], 'set'):
                basic_vars[field].set(1 if is_true else 0)


def _standardize_contact(c: Dict[str, Any], contact_type: str) -> Dict[str, Any]:
    """将适配器返回的接触者记录转换为GUI使用的标准格式"""
    entry = dict(c)
    if "name" not in entry:
        entry["name"] = c.get("name", c.get("contact_name", ""))
    if "age" not in entry:
        age = c.get("age")
        if age is None and c.get("birth_date"):
            from ...health_interop.base import calculate_age
            age = calculate_age(c["birth_date"])
        entry["age"] = age if age is not None else ""
    if "relationship" not in entry and "contact_relation" in c:
        entry["relationship"] = c["contact_relation"]
    entry["contact_type"] = contact_type
    return entry


# ============================================================================
# Mixin（供 DataImportMixin 组合使用）
# ============================================================================

class HealthInteropMixin:
    """医疗标准接口导入 Mixin"""

    def _load_from_health_interop(self):
        """触发医疗标准接口导入（菜单/按钮入口）"""
        pipeline = HealthInteropImportPipeline(self)
        pipeline.run()

    def generate_tb_report_card(self):
        """基于当前患者数据自动生成传染病报告卡"""
        from ...health_interop import TBCardFiller

        # 收集当前患者信息
        pi = self._collect_current_patient_info()
        lab_results = getattr(self, '_imported_lab_results', [])
        diagnoses = getattr(self, '_imported_diagnoses', [])
        imaging = getattr(self, '_imported_imaging_reports', [])

        filler = TBCardFiller()
        report_unit = ""
        report_doctor = ""
        card = filler.fill_card(pi, lab_results, diagnoses, imaging,
                                report_unit=report_unit,
                                report_doctor=report_doctor)

        # 显示结果对话框
        valid, missing = card.validate()
        self._show_report_card_dialog(card, missing)

    def _collect_current_patient_info(self) -> Dict[str, Any]:
        """从当前GUI状态收集患者基本信息"""
        pi = {}
        bv = getattr(self, 'basic_info_vars', {})
        if "patient_name" in bv:
            pi["name"] = bv["patient_name"].get()
        if hasattr(self, 'patient_name_var'):
            pi["name"] = self.patient_name_var.get()
        if "age" in bv:
            try:
                pi["age"] = int(bv["age"].get())
            except (TypeError, ValueError):
                pass
        if "gender" in bv:
            g = bv["gender"].get()
            pi["gender"] = "male" if g == 1 else "female" if g == 0 else ""
        # 其余字段根据系统变量扩展
        return pi

    def _show_report_card_dialog(self, card, missing_fields):
        """显示传染病报告卡预览对话框"""
        from ...health_interop.cdc import CASE_CLASSIFICATIONS
        dlg = tk.Toplevel(self.root)
        dlg.title("传染病报告卡 - 自动填报预览")
        dlg.geometry("600x580")
        dlg.transient(self.root)
        dlg.grab_set()

        # 使用 Text 显示卡信息
        frame = ttk.Frame(dlg, padding=15)
        frame.pack(fill='both', expand=True)

        ttk.Label(frame, text="传染病报告卡（肺结核）",
                  font=('Microsoft YaHei', 13, 'bold')).pack(pady=(0, 10))

        text = tk.Text(frame, wrap='word', font=('Microsoft YaHei', 10), height=22)
        scroll = ttk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')

        text.insert('end', "【患者基本信息】\n")
        text.insert('end', f"  姓名：{card.name or '（待填写）'}\n")
        text.insert('end', f"  性别：{card.gender_text or '（待填写）'}\n")
        text.insert('end', f"  年龄：{card.age if card.age is not None else '（待填写）'}\n")
        text.insert('end', f"  身份证号：{card.id_card or '（待填写）'}\n")
        text.insert('end', f"  联系电话：{card.phone or '（待填写）'}\n")
        text.insert('end', f"  现住址：{card.current_address or '（待填写）'}\n")
        text.insert('end', f"  工作单位：{card.work_unit or '（待填写）'}\n")
        text.insert('end', f"  职业：{card.occupation or '（待填写）'}\n")
        text.insert('end', f"  民族：{card.ethnicity or '（待填写）'}\n")

        text.insert('end', "\n【疾病信息】\n")
        text.insert('end', f"  病例分类：{CASE_CLASSIFICATIONS.get(card.case_classification, '（待判定）')}\n")
        text.insert('end', f"  发病日期：{card.onset_date or '（待填写）'}\n")
        text.insert('end', f"  诊断日期：{card.diagnosis_date or '（待填写）'}\n")
        text.insert('end', f"  诊断依据：{'、'.join(card.diagnosis_basis) if card.diagnosis_basis else '（待填写）'}\n")
        text.insert('end', f"  病原学结果：{card.pathogen_result or '（待填写）'}\n")
        text.insert('end', f"  耐药情况：{card.drug_resistance or '（待检测）'}\n")
        text.insert('end', f"  结核分型：{card.tb_type or '（待填写）'}\n")
        text.insert('end', f"  治疗分类：{card.treatment_category or '初治'}\n")

        text.insert('end', "\n【报告信息】\n")
        text.insert('end', f"  报告单位：{card.report_unit or '（待填写）'}\n")
        text.insert('end', f"  报告医生：{card.report_doctor or '（待填写）'}\n")
        text.insert('end', f"  报告日期：{card.report_date}\n")
        text.insert('end', f"  自动填报来源：{card.fill_source}\n")
        text.insert('end', f"  是否重报：{'是' if card.is_repeat else '否'}\n")

        if missing_fields:
            text.insert('end', f"\n⚠ 以下必填项未自动填充，请医生补充：\n")
            for f in missing_fields:
                text.insert('end', f"  - {f}\n")
            text.tag_add("warn", "end - %d lines" % (len(missing_fields)+2), "end")
            text.tag_config("warn", foreground='#e74c3c')

        text.configure(state='disabled')

        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(fill='x', padx=15, pady=10)

        def export_xml():
            xml_str = card.to_cdc_xml()
            path = filedialog.asksaveasfilename(
                defaultextension=".xml",
                filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
                initialfile=f"结核报卡_{card.name or card.card_id}.xml")
            if path:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(xml_str)
                messagebox.showinfo("成功", f"已导出到: {path}")

        def export_json():
            d = card.to_dict(for_transmission=False)
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile=f"结核报卡_{card.name or card.card_id}.json")
            if path:
                import json as _json
                with open(path, 'w', encoding='utf-8') as f:
                    _json.dump(d, f, ensure_ascii=False, indent=2)
                messagebox.showinfo("成功", f"已导出到: {path}")

        ttk.Button(btn_frame, text="导出XML（疾控格式）", command=export_xml).pack(side='left', padx=3)
        ttk.Button(btn_frame, text="导出JSON", command=export_json).pack(side='left', padx=3)
        ttk.Button(btn_frame, text="关闭", command=dlg.destroy).pack(side='right', padx=3)
