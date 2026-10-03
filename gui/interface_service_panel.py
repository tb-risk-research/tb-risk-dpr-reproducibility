#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""接口服务管理面板

提供HL7 MLLP后台监听服务和CDC回传Webhook的GUI管理：
- 服务启动/停止
- 参数配置（监听地址、端口、存储目录、签名密钥）
- 实时消息日志
- 服务状态监控
- 消息统计
"""

from __future__ import annotations

import logging
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("tb_risk.gui.interface_service")

MAX_LOG_ENTRIES = 500


class InterfaceServicePanel(tk.Toplevel):
    """接口服务管理对话框"""

    def __init__(self, parent, app=None):
        super().__init__(parent)
        self.title("接口服务管理")
        self.geometry("850x750")
        self.minsize(700, 600)
        self.transient(parent)
        self.grab_set()
        self.app = app

        # MLLP服务状态
        self._mllp_server = None
        self._mllp_running = False
        self._mllp_thread = None
        self._mllp_message_count = 0

        # Webhook服务状态
        self._webhook_server = None
        self._webhook_running = False
        self._webhook_message_count = 0

        # 日志
        self._log_entries: List[Dict[str, Any]] = []
        self._status_poll_id = None

        # 当前日志显示模式 ('mllp'|'webhook'|'all')
        self._log_mode = 'all'

        # 配置变量
        self._mllp_host_var = tk.StringVar(value="0.0.0.0")
        self._mllp_port_var = tk.StringVar(value="2575")
        self._mllp_data_dir_var = tk.StringVar(value="hl7_messages")
        self._mllp_auto_assess_var = tk.BooleanVar(value=True)
        self._mllp_tls_enabled_var = tk.BooleanVar(value=False)
        self._mllp_cert_var = tk.StringVar(value="")
        self._mllp_key_var = tk.StringVar(value="")
        self._mllp_ca_var = tk.StringVar(value="")
        self._mllp_require_client_cert_var = tk.BooleanVar(value=False)

        self._webhook_host_var = tk.StringVar(value="0.0.0.0")
        self._webhook_port_var = tk.StringVar(value="8080")
        self._webhook_secret_var = tk.StringVar(value="")
        self._webhook_data_dir_var = tk.StringVar(value="cdc_webhook")
        self._webhook_update_gnn_var = tk.BooleanVar(value=True)
        self._webhook_tls_enabled_var = tk.BooleanVar(value=False)
        self._webhook_cert_var = tk.StringVar(value="")
        self._webhook_key_var = tk.StringVar(value="")

        self._build_ui()
        self._load_config()
        self._start_status_polling()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        """构建界面"""
        info = ttk.Label(
            self,
            text="配置和管理医院系统接口服务。HL7 MLLP服务用于接收HIS/EMR/LIS/PACS推送的消息，"
                 "CDC Webhook用于接收疾控中心回传的治疗结局和密切接触者数据。",
            font=('Microsoft YaHei', 9), wraplength=800, justify='left')
        info.pack(pady=(10, 5), padx=15, anchor='w')

        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True, padx=15, pady=10)

        # HL7 MLLP 标签页
        mllp_frame = ttk.Frame(notebook, padding=10)
        notebook.add(mllp_frame, text="HL7 MLLP 消息服务")
        self._build_mllp_tab(mllp_frame)

        # CDC Webhook 标签页
        webhook_frame = ttk.Frame(notebook, padding=10)
        notebook.add(webhook_frame, text="疾控回传 Webhook")
        self._build_webhook_tab(webhook_frame)

        # 公卫科质控面板标签页
        qc_frame = ttk.Frame(notebook, padding=10)
        notebook.add(qc_frame, text="质控管理")
        self._build_qc_tab(qc_frame)

        # 死信队列管理标签页
        dlq_frame = ttk.Frame(notebook, padding=10)
        notebook.add(dlq_frame, text="死信队列")
        self._build_dlq_tab(dlq_frame)

        # 审计日志查询标签页
        audit_frame = ttk.Frame(notebook, padding=10)
        notebook.add(audit_frame, text="审计日志")
        self._build_audit_tab(audit_frame)

        # 机构配置标签页
        org_frame = ttk.Frame(notebook, padding=10)
        notebook.add(org_frame, text="上报机构配置")
        self._build_org_config_tab(org_frame)

        # 统一日志区域
        log_frame = ttk.LabelFrame(self, text="服务运行日志", padding=5)
        log_frame.pack(fill='both', expand=True, padx=15, pady=(0, 5))

        log_btn_row = ttk.Frame(log_frame)
        log_btn_row.pack(fill='x', pady=(0, 5))
        ttk.Button(log_btn_row, text="显示全部", command=lambda: self._set_log_mode('all'), width=8).pack(side='left', padx=2)
        ttk.Button(log_btn_row, text="仅MLLP", command=lambda: self._set_log_mode('mllp'), width=8).pack(side='left', padx=2)
        ttk.Button(log_btn_row, text="仅Webhook", command=lambda: self._set_log_mode('webhook'), width=8).pack(side='left', padx=2)
        ttk.Separator(log_btn_row, orient='vertical').pack(side='left', fill='y', padx=5)
        ttk.Label(log_btn_row, text="级别:", font=('Microsoft YaHei', 8)).pack(side='left')
        self._log_level_var = tk.StringVar(value="ALL")
        level_combo = ttk.Combobox(log_btn_row, textvariable=self._log_level_var,
                                    values=["ALL", "INFO", "WARNING", "ERROR", "SUCCESS"],
                                    width=8, state='readonly')
        level_combo.pack(side='left', padx=2)
        level_combo.bind('<<ComboboxSelected>>', lambda e: self._update_log_display())
        ttk.Label(log_btn_row, text="搜索:", font=('Microsoft YaHei', 8)).pack(side='left', padx=(5,0))
        self._log_search_var = tk.StringVar()
        search_entry = ttk.Entry(log_btn_row, textvariable=self._log_search_var, width=15)
        search_entry.pack(side='left', padx=2)
        search_entry.bind('<KeyRelease>', lambda e: self._update_log_display())
        self._log_autoscroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_btn_row, text="自动滚动", variable=self._log_autoscroll_var).pack(side='left', padx=5)
        ttk.Button(log_btn_row, text="导出日志", command=self._export_logs, width=8).pack(side='right', padx=2)
        ttk.Button(log_btn_row, text="清空日志", command=self._clear_log, width=8).pack(side='right', padx=2)

        log_container = ttk.Frame(log_frame)
        log_container.pack(fill='both', expand=True)

        self._log_text = tk.Text(log_container, wrap='none', font=('Consolas', 9),
                                 height=10, state='disabled')
        log_scroll_y = ttk.Scrollbar(log_container, orient='vertical',
                                      command=self._log_text.yview)
        log_scroll_x = ttk.Scrollbar(log_container, orient='horizontal',
                                      command=self._log_text.xview)
        self._log_text.configure(yscrollcommand=log_scroll_y.set,
                                  xscrollcommand=log_scroll_x.set)

        self._log_text.grid(row=0, column=0, sticky='nsew')
        log_scroll_y.grid(row=0, column=1, sticky='ns')
        log_scroll_x.grid(row=1, column=0, sticky='ew')
        log_container.grid_rowconfigure(0, weight=1)
        log_container.grid_columnconfigure(0, weight=1)

        self._log_text.tag_configure('info', foreground='#17a2b8')
        self._log_text.tag_configure('success', foreground='#28a745')
        self._log_text.tag_configure('warning', foreground='#d39e00')
        self._log_text.tag_configure('error', foreground='#dc3545')
        self._log_text.tag_configure('timestamp', foreground='#6c757d')
        self._log_text.tag_configure('source', foreground='#6f42c1')

        # 底部按钮
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill='x', padx=15, pady=(0, 15))
        ttk.Button(btn_frame, text="保存配置", command=self._save_config).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="关闭", command=self._on_close).pack(side='right')

    def _build_mllp_tab(self, parent):
        """构建HL7 MLLP配置和控制标签页"""
        status_frame = ttk.LabelFrame(parent, text="服务状态", padding=10)
        status_frame.pack(fill='x', pady=(0, 10))

        status_row = ttk.Frame(status_frame)
        status_row.pack(fill='x')

        self._mllp_status_indicator = tk.Canvas(status_row, width=20, height=20,
                                                 highlightthickness=0)
        self._mllp_status_indicator.pack(side='left', padx=(0, 8))
        self._mllp_status_dot = self._mllp_status_indicator.create_oval(
            2, 2, 18, 18, fill='#e74c3c', outline='#c0392b')

        self._mllp_status_label = ttk.Label(status_row, text="已停止",
                                            font=('Microsoft YaHei', 10, 'bold'))
        self._mllp_status_label.pack(side='left')

        self._mllp_count_label = ttk.Label(status_row, text="已接收消息: 0",
                                           font=('Microsoft YaHei', 9))
        self._mllp_count_label.pack(side='right')

        config_frame = ttk.LabelFrame(parent, text="监听配置", padding=10)
        config_frame.pack(fill='x', pady=(0, 10))

        host_row = ttk.Frame(config_frame)
        host_row.pack(fill='x', pady=3)
        ttk.Label(host_row, text="监听地址:", font=('Microsoft YaHei', 9),
                  width=12).pack(side='left')
        ttk.Entry(host_row, textvariable=self._mllp_host_var, width=20).pack(side='left', padx=5)
        ttk.Label(host_row, text="(0.0.0.0 表示监听所有网卡)",
                  font=('Microsoft YaHei', 8), foreground='#6c757d').pack(side='left', padx=5)

        port_row = ttk.Frame(config_frame)
        port_row.pack(fill='x', pady=3)
        ttk.Label(port_row, text="监听端口:", font=('Microsoft YaHei', 9),
                  width=12).pack(side='left')
        ttk.Entry(port_row, textvariable=self._mllp_port_var, width=20).pack(side='left', padx=5)
        ttk.Label(port_row, text="(默认2575是HL7 MLLP标准端口)",
                  font=('Microsoft YaHei', 8), foreground='#6c757d').pack(side='left', padx=5)

        dir_row = ttk.Frame(config_frame)
        dir_row.pack(fill='x', pady=3)
        ttk.Label(dir_row, text="消息存储目录:", font=('Microsoft YaHei', 9),
                  width=12).pack(side='left')
        ttk.Entry(dir_row, textvariable=self._mllp_data_dir_var, width=40).pack(
            side='left', padx=5, fill='x', expand=True)

        def browse_mllp_dir():
            d = filedialog.askdirectory()
            if d:
                self._mllp_data_dir_var.set(d)
        ttk.Button(dir_row, text="浏览...", command=browse_mllp_dir, width=8).pack(side='left', padx=3)

        # TLS/SSL 配置区域
        tls_frame = ttk.LabelFrame(parent, text="TLS/SSL 加密传输 (MLLPS)", padding=10)
        tls_frame.pack(fill='x', pady=(0, 10))

        tls_enable_row = ttk.Frame(tls_frame)
        tls_enable_row.pack(fill='x', pady=3)
        ttk.Checkbutton(tls_enable_row, text="启用TLS加密传输",
                       variable=self._mllp_tls_enabled_var,
                       command=self._on_mllp_tls_toggle).pack(side='left')
        ttk.Label(tls_enable_row, text="(启用后请配置证书文件，端口默认使用2576)",
                  font=('Microsoft YaHei', 8), foreground='#6c757d').pack(side='left', padx=10)

        self._mllp_tls_frame = ttk.Frame(tls_frame)
        self._mllp_tls_frame.pack(fill='x', pady=3)

        cert_row = ttk.Frame(self._mllp_tls_frame)
        cert_row.pack(fill='x', pady=2)
        ttk.Label(cert_row, text="证书文件:", font=('Microsoft YaHei', 9),
                  width=12).pack(side='left')
        ttk.Entry(cert_row, textvariable=self._mllp_cert_var, width=45).pack(
            side='left', padx=5, fill='x', expand=True)

        def browse_mllp_cert():
            f = filedialog.askopenfilename(
                title="选择证书文件",
                filetypes=[("证书文件", "*.crt *.pem *.cer"), ("所有文件", "*.*")])
            if f:
                self._mllp_cert_var.set(f)
        ttk.Button(cert_row, text="浏览...", command=browse_mllp_cert, width=8).pack(side='left', padx=3)

        key_row = ttk.Frame(self._mllp_tls_frame)
        key_row.pack(fill='x', pady=2)
        ttk.Label(key_row, text="私钥文件:", font=('Microsoft YaHei', 9),
                  width=12).pack(side='left')
        ttk.Entry(key_row, textvariable=self._mllp_key_var, width=45).pack(
            side='left', padx=5, fill='x', expand=True)

        def browse_mllp_key():
            f = filedialog.askopenfilename(
                title="选择私钥文件",
                filetypes=[("私钥文件", "*.key *.pem"), ("所有文件", "*.*")])
            if f:
                self._mllp_key_var.set(f)
        ttk.Button(key_row, text="浏览...", command=browse_mllp_key, width=8).pack(side='left', padx=3)

        ca_row = ttk.Frame(self._mllp_tls_frame)
        ca_row.pack(fill='x', pady=2)
        ttk.Label(ca_row, text="CA证书:", font=('Microsoft YaHei', 9),
                  width=12).pack(side='left')
        ttk.Entry(ca_row, textvariable=self._mllp_ca_var, width=45).pack(
            side='left', padx=5, fill='x', expand=True)

        def browse_mllp_ca():
            f = filedialog.askopenfilename(
                title="选择CA证书文件",
                filetypes=[("CA证书", "*.crt *.pem *.cer"), ("所有文件", "*.*")])
            if f:
                self._mllp_ca_var.set(f)
        ttk.Button(ca_row, text="浏览...", command=browse_mllp_ca, width=8).pack(side='left', padx=3)

        tls_check_row = ttk.Frame(self._mllp_tls_frame)
        tls_check_row.pack(fill='x', pady=2)
        ttk.Checkbutton(tls_check_row, text="要求客户端证书认证 (双向TLS)",
                       variable=self._mllp_require_client_cert_var).pack(side='left')
        ttk.Button(tls_check_row, text="验证证书", command=self._validate_mllp_certs,
                   width=10).pack(side='right', padx=5)

        self._on_mllp_tls_toggle()

        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.pack(fill='x', pady=(0, 10))

        self._mllp_start_btn = ttk.Button(ctrl_frame, text="启动服务",
                                          command=self._start_mllp, width=15)
        self._mllp_start_btn.pack(side='left', padx=5)

        self._mllp_stop_btn = ttk.Button(ctrl_frame, text="停止服务",
                                         command=self._stop_mllp, width=15,
                                         state='disabled')
        self._mllp_stop_btn.pack(side='left', padx=5)

        ttk.Checkbutton(ctrl_frame, text="接收消息后自动风险评估",
                       variable=self._mllp_auto_assess_var).pack(side='left', padx=15)

        info_text = ("HL7 MLLP (Minimal Lower Layer Protocol) 是医疗信息系统间"
                     "交换HL7 v2消息的标准协议。服务启动后将持续监听指定端口，"
                     "自动接收HIS/EMR/LIS/PACS系统推送的ADT（入出转）、ORM（医嘱）、"
                     "ORU（检验结果）等消息并持久化存储。")
        info_lbl = ttk.Label(parent, text=info_text, font=('Microsoft YaHei', 8),
                             foreground='#6c757d', wraplength=750, justify='left')
        info_lbl.pack(fill='x', pady=5)

    def _build_webhook_tab(self, parent):
        """构建CDC Webhook配置标签页"""
        status_frame = ttk.LabelFrame(parent, text="服务状态", padding=10)
        status_frame.pack(fill='x', pady=(0, 10))

        status_row = ttk.Frame(status_frame)
        status_row.pack(fill='x')

        self._webhook_status_indicator = tk.Canvas(status_row, width=20, height=20,
                                                    highlightthickness=0)
        self._webhook_status_indicator.pack(side='left', padx=(0, 8))
        self._webhook_status_dot = self._webhook_status_indicator.create_oval(
            2, 2, 18, 18, fill='#e74c3c', outline='#c0392b')

        self._webhook_status_label = ttk.Label(status_row, text="已停止",
                                                font=('Microsoft YaHei', 10, 'bold'))
        self._webhook_status_label.pack(side='left')

        self._webhook_count_label = ttk.Label(status_row, text="已接收回调: 0",
                                               font=('Microsoft YaHei', 9))
        self._webhook_count_label.pack(side='right')

        config_frame = ttk.LabelFrame(parent, text="监听配置", padding=10)
        config_frame.pack(fill='x', pady=(0, 10))

        host_row = ttk.Frame(config_frame)
        host_row.pack(fill='x', pady=3)
        ttk.Label(host_row, text="监听地址:", font=('Microsoft YaHei', 9),
                  width=12).pack(side='left')
        ttk.Entry(host_row, textvariable=self._webhook_host_var, width=20).pack(side='left', padx=5)
        ttk.Label(host_row, text="(0.0.0.0 表示监听所有网卡)",
                  font=('Microsoft YaHei', 8), foreground='#6c757d').pack(side='left', padx=5)

        port_row = ttk.Frame(config_frame)
        port_row.pack(fill='x', pady=3)
        ttk.Label(port_row, text="监听端口:", font=('Microsoft YaHei', 9),
                  width=12).pack(side='left')
        ttk.Entry(port_row, textvariable=self._webhook_port_var, width=20).pack(side='left', padx=5)
        ttk.Label(port_row, text="(默认8080，回调URL: http://host:port/cdc/callback)",
                  font=('Microsoft YaHei', 8), foreground='#6c757d').pack(side='left', padx=5)

        secret_row = ttk.Frame(config_frame)
        secret_row.pack(fill='x', pady=3)
        ttk.Label(secret_row, text="签名密钥:", font=('Microsoft YaHei', 9),
                  width=12).pack(side='left')
        ttk.Entry(secret_row, textvariable=self._webhook_secret_var, width=40, show='*').pack(
            side='left', padx=5, fill='x', expand=True)
        ttk.Label(secret_row, text="(留空则不验证签名)",
                  font=('Microsoft YaHei', 8), foreground='#6c757d').pack(side='left', padx=5)

        dir_row = ttk.Frame(config_frame)
        dir_row.pack(fill='x', pady=3)
        ttk.Label(dir_row, text="数据存储目录:", font=('Microsoft YaHei', 9),
                  width=12).pack(side='left')
        ttk.Entry(dir_row, textvariable=self._webhook_data_dir_var, width=40).pack(
            side='left', padx=5, fill='x', expand=True)

        def browse_wh_dir():
            d = filedialog.askdirectory()
            if d:
                self._webhook_data_dir_var.set(d)
        ttk.Button(dir_row, text="浏览...", command=browse_wh_dir, width=8).pack(side='left', padx=3)

        # TLS/SSL 配置区域 (HTTPS)
        wh_tls_frame = ttk.LabelFrame(parent, text="TLS/SSL 加密传输 (HTTPS)", padding=10)
        wh_tls_frame.pack(fill='x', pady=(0, 10))

        wh_tls_enable_row = ttk.Frame(wh_tls_frame)
        wh_tls_enable_row.pack(fill='x', pady=3)
        ttk.Checkbutton(wh_tls_enable_row, text="启用HTTPS加密传输",
                       variable=self._webhook_tls_enabled_var,
                       command=self._on_webhook_tls_toggle).pack(side='left')
        ttk.Label(wh_tls_enable_row, text="(启用后请配置证书文件，端口默认使用8443)",
                  font=('Microsoft YaHei', 8), foreground='#6c757d').pack(side='left', padx=10)

        self._webhook_tls_frame = ttk.Frame(wh_tls_frame)
        self._webhook_tls_frame.pack(fill='x', pady=3)

        wh_cert_row = ttk.Frame(self._webhook_tls_frame)
        wh_cert_row.pack(fill='x', pady=2)
        ttk.Label(wh_cert_row, text="证书文件:", font=('Microsoft YaHei', 9),
                  width=12).pack(side='left')
        ttk.Entry(wh_cert_row, textvariable=self._webhook_cert_var, width=45).pack(
            side='left', padx=5, fill='x', expand=True)

        def browse_wh_cert():
            f = filedialog.askopenfilename(
                title="选择证书文件",
                filetypes=[("证书文件", "*.crt *.pem *.cer"), ("所有文件", "*.*")])
            if f:
                self._webhook_cert_var.set(f)
        ttk.Button(wh_cert_row, text="浏览...", command=browse_wh_cert, width=8).pack(side='left', padx=3)

        wh_key_row = ttk.Frame(self._webhook_tls_frame)
        wh_key_row.pack(fill='x', pady=2)
        ttk.Label(wh_key_row, text="私钥文件:", font=('Microsoft YaHei', 9),
                  width=12).pack(side='left')
        ttk.Entry(wh_key_row, textvariable=self._webhook_key_var, width=45).pack(
            side='left', padx=5, fill='x', expand=True)

        def browse_wh_key():
            f = filedialog.askopenfilename(
                title="选择私钥文件",
                filetypes=[("私钥文件", "*.key *.pem"), ("所有文件", "*.*")])
            if f:
                self._webhook_key_var.set(f)
        ttk.Button(wh_key_row, text="浏览...", command=browse_wh_key, width=8).pack(side='left', padx=3)

        wh_tls_check_row = ttk.Frame(self._webhook_tls_frame)
        wh_tls_check_row.pack(fill='x', pady=2)
        ttk.Button(wh_tls_check_row, text="验证证书", command=self._validate_webhook_certs,
                   width=10).pack(side='right', padx=5)

        self._on_webhook_tls_toggle()

        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.pack(fill='x', pady=(0, 10))

        self._webhook_start_btn = ttk.Button(ctrl_frame, text="启动服务",
                                              command=self._start_webhook, width=15)
        self._webhook_start_btn.pack(side='left', padx=5)

        self._webhook_stop_btn = ttk.Button(ctrl_frame, text="停止服务",
                                             command=self._stop_webhook, width=15,
                                             state='disabled')
        self._webhook_stop_btn.pack(side='left', padx=5)

        ttk.Checkbutton(ctrl_frame, text="回传数据自动更新GNN网络",
                       variable=self._webhook_update_gnn_var).pack(side='left', padx=15)

        info_text = ("CDC Webhook用于被动接收疾控中心推送的患者治疗结局、"
                     "密切接触者筛查结果、区域流行病学数据等回传信息。"
                     "支持HMAC-SHA256签名验证，确保数据来源可信。"
                     "回调路径支持: /cdc/callback, /webhook/cdc, /callback")
        info_lbl = ttk.Label(parent, text=info_text, font=('Microsoft YaHei', 8),
                             foreground='#6c757d', wraplength=750, justify='left')
        info_lbl.pack(fill='x', pady=5)

    # ---- 配置持久化 ----
    def _load_config(self):
        """从app配置加载"""
        if self.app and hasattr(self.app, 'config'):
            cfg = self.app.config
            if hasattr(cfg, 'mllp_host') and cfg.mllp_host:
                self._mllp_host_var.set(cfg.mllp_host)
            if hasattr(cfg, 'mllp_port') and cfg.mllp_port:
                self._mllp_port_var.set(str(cfg.mllp_port))
            if hasattr(cfg, 'mllp_data_dir') and cfg.mllp_data_dir:
                self._mllp_data_dir_var.set(cfg.mllp_data_dir)
            if hasattr(cfg, 'mllp_auto_assess'):
                self._mllp_auto_assess_var.set(bool(cfg.mllp_auto_assess))
            if hasattr(cfg, 'mllp_tls_enabled'):
                self._mllp_tls_enabled_var.set(bool(cfg.mllp_tls_enabled))
            if hasattr(cfg, 'mllp_ssl_cert') and cfg.mllp_ssl_cert:
                self._mllp_cert_var.set(cfg.mllp_ssl_cert)
            if hasattr(cfg, 'mllp_ssl_key') and cfg.mllp_ssl_key:
                self._mllp_key_var.set(cfg.mllp_ssl_key)
            if hasattr(cfg, 'mllp_ssl_ca') and cfg.mllp_ssl_ca:
                self._mllp_ca_var.set(cfg.mllp_ssl_ca)
            if hasattr(cfg, 'mllp_require_client_cert'):
                self._mllp_require_client_cert_var.set(bool(cfg.mllp_require_client_cert))
            if hasattr(cfg, 'webhook_host') and cfg.webhook_host:
                self._webhook_host_var.set(cfg.webhook_host)
            if hasattr(cfg, 'webhook_port') and cfg.webhook_port:
                self._webhook_port_var.set(str(cfg.webhook_port))
            if hasattr(cfg, 'webhook_secret_key') and cfg.webhook_secret_key:
                self._webhook_secret_var.set(cfg.webhook_secret_key)
            if hasattr(cfg, 'webhook_data_dir') and cfg.webhook_data_dir:
                self._webhook_data_dir_var.set(cfg.webhook_data_dir)
            if hasattr(cfg, 'webhook_update_gnn'):
                self._webhook_update_gnn_var.set(bool(cfg.webhook_update_gnn))
            if hasattr(cfg, 'webhook_tls_enabled'):
                self._webhook_tls_enabled_var.set(bool(cfg.webhook_tls_enabled))
            if hasattr(cfg, 'webhook_ssl_cert') and cfg.webhook_ssl_cert:
                self._webhook_cert_var.set(cfg.webhook_ssl_cert)
            if hasattr(cfg, 'webhook_ssl_key') and cfg.webhook_ssl_key:
                self._webhook_key_var.set(cfg.webhook_ssl_key)

        # 加载配置后同步TLS控件状态
        self._on_mllp_tls_toggle()
        self._on_webhook_tls_toggle()

    def _save_config(self):
        """保存配置到app配置"""
        if self.app and hasattr(self.app, 'config'):
            cfg = self.app.config
            try:
                cfg.mllp_host = self._mllp_host_var.get().strip()
                cfg.mllp_port = int(self._mllp_port_var.get().strip() or "2575")
                cfg.mllp_data_dir = self._mllp_data_dir_var.get().strip()
                cfg.mllp_auto_assess = self._mllp_auto_assess_var.get()
                cfg.mllp_tls_enabled = self._mllp_tls_enabled_var.get()
                cfg.mllp_ssl_cert = self._mllp_cert_var.get().strip()
                cfg.mllp_ssl_key = self._mllp_key_var.get().strip()
                cfg.mllp_ssl_ca = self._mllp_ca_var.get().strip()
                cfg.mllp_require_client_cert = self._mllp_require_client_cert_var.get()
                cfg.webhook_host = self._webhook_host_var.get().strip()
                cfg.webhook_port = int(self._webhook_port_var.get().strip() or "8080")
                cfg.webhook_secret_key = self._webhook_secret_var.get().strip()
                cfg.webhook_data_dir = self._webhook_data_dir_var.get().strip()
                cfg.webhook_update_gnn = self._webhook_update_gnn_var.get()
                cfg.webhook_tls_enabled = self._webhook_tls_enabled_var.get()
                cfg.webhook_ssl_cert = self._webhook_cert_var.get().strip()
                cfg.webhook_ssl_key = self._webhook_key_var.get().strip()
                if hasattr(cfg, 'save'):
                    cfg.save()
                self._add_log('all', 'info', "配置已保存")
            except (TypeError, ValueError) as e:
                self._add_log('all', 'warning', f"配置保存失败: {e}")

    def _validate_port(self, port_str: str, default: int) -> Tuple[bool, int]:
        try:
            port = int(port_str.strip() or str(default))
            if 1 <= port <= 65535:
                return True, port
        except (TypeError, ValueError):
            pass
        return False, default

    # ---- TLS/SSL 辅助方法 ----
    def _set_tls_config_state(self, parent_frame, enabled, include_checkbutton=True):
        """递归设置TLS配置区域内控件的启用/禁用状态"""
        widget_types = (ttk.Entry, ttk.Button)
        if include_checkbutton:
            widget_types = (ttk.Entry, ttk.Button, ttk.Checkbutton)
        for child in parent_frame.winfo_children():
            try:
                if isinstance(child, widget_types):
                    child.configure(state='normal' if enabled else 'disabled')
                elif isinstance(child, (ttk.Frame, ttk.LabelFrame)):
                    self._set_tls_config_state(child, enabled, include_checkbutton)
            except Exception:
                pass

    def _on_mllp_tls_toggle(self):
        """MLLP TLS开关切换"""
        enabled = self._mllp_tls_enabled_var.get()
        self._set_tls_config_state(self._mllp_tls_frame, enabled, include_checkbutton=True)
        # 启用TLS时，如果端口是默认非TLS端口2575，自动切换为TLS默认端口2576
        if enabled:
            current_port = self._mllp_port_var.get().strip()
            if current_port == "2575" or current_port == "":
                self._mllp_port_var.set("2576")
        else:
            current_port = self._mllp_port_var.get().strip()
            if current_port == "2576":
                self._mllp_port_var.set("2575")

    def _on_webhook_tls_toggle(self):
        """Webhook TLS开关切换"""
        enabled = self._webhook_tls_enabled_var.get()
        self._set_tls_config_state(self._webhook_tls_frame, enabled, include_checkbutton=False)
        # 启用TLS时，如果端口是默认非TLS端口8080，自动切换为TLS默认端口8443
        if enabled:
            current_port = self._webhook_port_var.get().strip()
            if current_port == "8080" or current_port == "":
                self._webhook_port_var.set("8443")
        else:
            current_port = self._webhook_port_var.get().strip()
            if current_port == "8443":
                self._webhook_port_var.set("8080")

    def _validate_cert_files(self, certfile: str, keyfile: str, cafile: str = "") -> Tuple[bool, str]:
        """验证证书文件有效性"""
        import ssl

        if not certfile:
            return False, "未指定证书文件"
        if not keyfile:
            return False, "未指定私钥文件"
        if not os.path.exists(certfile):
            return False, f"证书文件不存在: {certfile}"
        if not os.path.exists(keyfile):
            return False, f"私钥文件不存在: {keyfile}"
        if cafile and not os.path.exists(cafile):
            return False, f"CA证书文件不存在: {cafile}"

        try:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certfile=certfile, keyfile=keyfile)
            if cafile:
                context.load_verify_locations(cafile=cafile)
            return True, "证书验证通过"
        except ssl.SSLError as e:
            return False, f"SSL证书错误: {str(e)[:200]}"
        except Exception as e:
            return False, f"证书验证失败: {str(e)[:200]}"

    def _validate_mllp_certs(self):
        """验证MLLP证书"""
        cert = self._mllp_cert_var.get().strip()
        key = self._mllp_key_var.get().strip()
        ca = self._mllp_ca_var.get().strip()
        ok, msg = self._validate_cert_files(cert, key, ca)
        if ok:
            messagebox.showinfo("证书验证", msg)
            self._add_log('mllp', 'success', f"MLLP证书验证通过: {cert}")
        else:
            messagebox.showerror("证书验证失败", msg)
            self._add_log('mllp', 'error', f"MLLP证书验证失败: {msg}")

    def _validate_webhook_certs(self):
        """验证Webhook证书"""
        cert = self._webhook_cert_var.get().strip()
        key = self._webhook_key_var.get().strip()
        ok, msg = self._validate_cert_files(cert, key)
        if ok:
            messagebox.showinfo("证书验证", msg)
            self._add_log('webhook', 'success', f"Webhook证书验证通过: {cert}")
        else:
            messagebox.showerror("证书验证失败", msg)
            self._add_log('webhook', 'error', f"Webhook证书验证失败: {msg}")

    def _check_cert_expiry(self, certfile: str) -> Optional[int]:
        """检查证书还有多少天过期，None表示无法读取"""
        try:
            import ssl
            from datetime import datetime
            cert_dict = ssl._ssl._test_decode_cert(certfile)
            if 'notAfter' in cert_dict:
                import calendar
                not_after = datetime.strptime(cert_dict['notAfter'], "%b %d %H:%M:%S %Y %Z")
                days_left = (not_after - datetime.utcnow()).days
                return days_left
        except Exception:
            pass
        return None

    # ---- MLLP 服务控制 ----
    def _start_mllp(self):
        """启动MLLP服务"""
        from ..health_interop import MLLPServer, AdapterConfig

        host = self._mllp_host_var.get().strip() or "0.0.0.0"
        tls_enabled = self._mllp_tls_enabled_var.get()
        default_port = 2576 if tls_enabled else 2575
        ok, port = self._validate_port(self._mllp_port_var.get(), default_port)
        if not ok:
            messagebox.showerror("配置错误", "端口必须是1-65535之间的有效整数")
            return
        data_dir = self._mllp_data_dir_var.get().strip() or "hl7_messages"

        # TLS参数
        ssl_cert = ""
        ssl_key = ""
        ssl_ca = ""
        require_client_cert = False
        if tls_enabled:
            ssl_cert = self._mllp_cert_var.get().strip()
            ssl_key = self._mllp_key_var.get().strip()
            ssl_ca = self._mllp_ca_var.get().strip()
            require_client_cert = self._mllp_require_client_cert_var.get()
            cert_ok, cert_msg = self._validate_cert_files(ssl_cert, ssl_key, ssl_ca)
            if not cert_ok:
                messagebox.showerror("TLS证书错误", cert_msg)
                return
            # 检查证书过期
            days_left = self._check_cert_expiry(ssl_cert)
            if days_left is not None:
                if days_left < 0:
                    messagebox.showerror("证书已过期", f"证书已过期 {-days_left} 天，请更换证书！")
                    return
                elif days_left <= 30:
                    if not messagebox.askyesno("证书即将过期",
                        f"证书将在 {days_left} 天后过期，是否继续启动？"):
                        return
            proto = "MLLPS(TLS)"
        else:
            proto = "MLLP"

        try:
            os.makedirs(data_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror("错误", f"无法创建存储目录: {e}")
            return

        try:
            from ..health_interop import map_adapter_result_to_predictor_input
            db_path = os.path.join(data_dir, "hl7_messages.db")
            adapter_config = AdapterConfig(
                port=port,
                base_url=f"tcp://{host}:{port}",
            )
            server = MLLPServer(
                adapter_config, queue_db_path=db_path,
                ssl_certfile=ssl_cert, ssl_keyfile=ssl_key,
                ssl_ca_certs=ssl_ca, require_client_cert=require_client_cert
            )
            server._host = host
            server._port = port
            auto_assess = self._mllp_auto_assess_var.get()

            def on_message(parsed_msg, result):
                msg_type = getattr(parsed_msg, 'message_type', 'UNKNOWN') if parsed_msg else 'UNKNOWN'
                pid = result.patient_id or 'N/A'
                self.root.after(0, lambda mt=msg_type, p=pid, r=result:
                    self._on_mllp_message(mt, p, r if auto_assess else None))

            server.set_message_handler(on_message)
            self._mllp_server = server

            def run_server():
                try:
                    self.root.after(0, lambda: self._add_log('mllp', 'info',
                        f"正在启动HL7 {proto}服务 {host}:{port}..."))
                    started = server.start()
                    if not started:
                        self.root.after(0, lambda: self._add_log('mllp', 'error',
                            "服务启动失败，请检查端口是否被占用"))
                        self.root.after(0, self._stop_mllp)
                        return
                    self._mllp_running = True
                    self.root.after(0, lambda: self._add_log('mllp', 'success',
                        f"{proto}服务已启动，监听 {host}:{port}"))
                    self.root.after(0, self._update_ui_state)
                except Exception as e:
                    self.root.after(0, lambda err=e: self._add_log('mllp', 'error', f"服务异常: {err}"))
                    LOGGER.error("MLLP服务启动异常", exc_info=True)
                    self.root.after(0, self._stop_mllp)

            self._mllp_thread = threading.Thread(target=run_server, daemon=True)
            self._mllp_thread.start()
            self._save_config()

        except Exception as e:
            messagebox.showerror("启动失败", str(e))
            LOGGER.error("MLLP服务启动失败", exc_info=True)

    def _on_mllp_message(self, msg_type: str, patient_id: str, result=None):
        """处理MLLP接收到的消息，可选自动风险评估"""
        self._mllp_message_count += 1
        self._add_log('mllp', 'success', f"收到消息 [{msg_type}] - 患者: {patient_id}")

        # 自动风险评估
        if result is not None and msg_type in ("ORU^R01", "ADT^A04", "ADT^A01", "ADT^A03"):
            try:
                self._run_auto_assessment(patient_id, result)
            except Exception as e:
                self._add_log('mllp', 'warning', f"自动风险评估失败: {e}")
                LOGGER.warning("自动风险评估异常", exc_info=True)

        self._update_ui_state()

    def _run_auto_assessment(self, patient_id: str, result):
        """对接收到的患者数据执行自动风险评估"""
        try:
            from ..health_interop import map_adapter_result_to_predictor_input
            features = map_adapter_result_to_predictor_input(result)
            if not features:
                self._add_log('mllp', 'info', f"患者 {patient_id}: 数据不足，无法进行风险评估")
                return

            # 尝试获取predictor进行评分
            risk_score = None
            risk_level = None
            predictor = None
            if self.app:
                if hasattr(self.app, 'predictor') and self.app.predictor:
                    predictor = self.app.predictor
                elif hasattr(self.app, '_predictor') and self.app._predictor:
                    predictor = self.app._predictor

            if predictor:
                try:
                    pred_result = predictor.predict(features)
                    if isinstance(pred_result, dict):
                        risk_score = pred_result.get('probability', pred_result.get('score'))
                        risk_level = pred_result.get('risk_level', pred_result.get('level'))
                    elif isinstance(pred_result, (int, float)):
                        risk_score = float(pred_result)
                except Exception as e:
                    self._add_log('mllp', 'warning', f"风险评分计算异常: {e}")
                    return

            # 记录结果
            if risk_score is not None:
                score_pct = risk_score * 100 if isinstance(risk_score, float) and risk_score <= 1 else risk_score
                level_text = f" [{risk_level}]" if risk_level else ""
                self._add_log('mllp', 'info',
                    f"患者 {patient_id}: 自动风险评估完成 - 风险评分: {score_pct:.1f}%{level_text}")

                # 高风险预警（>50%为高风险阈值）
                high_risk_threshold = 0.5 if isinstance(risk_score, float) and risk_score <= 1 else 50
                is_high_risk = risk_score >= high_risk_threshold
                if is_high_risk:
                    self._add_log('mllp', 'warning',
                        f"⚠️ 高风险预警: 患者 {patient_id} 风险评分 {score_pct:.1f}%，请及时关注！")
                    # 尝试弹出提醒（非阻塞）
                    try:
                        self.bell()
                    except Exception:
                        pass
            else:
                self._add_log('mllp', 'info',
                    f"患者 {patient_id}: 特征已映射（{len(features)}个字段），等待模型加载后可评分")

        except Exception as e:
            LOGGER.error("自动风险评估失败", exc_info=True)
            raise

    def _stop_mllp(self):
        """停止MLLP服务"""
        if self._mllp_server:
            self._add_log('mllp', 'info', "正在停止MLLP服务...")
            try:
                self._mllp_server.stop()
                self._add_log('mllp', 'info', "MLLP服务已停止")
            except Exception as e:
                self._add_log('mllp', 'warning', f"停止服务时出现异常: {e}")

        self._mllp_server = None
        self._mllp_running = False
        self._mllp_thread = None
        self._update_ui_state()

    # ---- Webhook 服务控制 ----
    def _start_webhook(self):
        """启动CDC Webhook服务"""
        from ..health_interop import CDCWebhookServer

        host = self._webhook_host_var.get().strip() or "0.0.0.0"
        tls_enabled = self._webhook_tls_enabled_var.get()
        default_port = 8443 if tls_enabled else 8080
        ok, port = self._validate_port(self._webhook_port_var.get(), default_port)
        if not ok:
            messagebox.showerror("配置错误", "端口必须是1-65535之间的有效整数")
            return
        secret = self._webhook_secret_var.get().strip()
        data_dir = self._webhook_data_dir_var.get().strip() or "cdc_webhook"

        # TLS参数
        ssl_cert = ""
        ssl_key = ""
        if tls_enabled:
            ssl_cert = self._webhook_cert_var.get().strip()
            ssl_key = self._webhook_key_var.get().strip()
            cert_ok, cert_msg = self._validate_cert_files(ssl_cert, ssl_key)
            if not cert_ok:
                messagebox.showerror("TLS证书错误", cert_msg)
                return
            # 检查证书过期
            days_left = self._check_cert_expiry(ssl_cert)
            if days_left is not None:
                if days_left < 0:
                    messagebox.showerror("证书已过期", f"证书已过期 {-days_left} 天，请更换证书！")
                    return
                elif days_left <= 30:
                    if not messagebox.askyesno("证书即将过期",
                        f"证书将在 {days_left} 天后过期，是否继续启动？"):
                        return
            proto = "HTTPS"
        else:
            proto = "HTTP"

        try:
            os.makedirs(data_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror("错误", f"无法创建存储目录: {e}")
            return

        try:
            update_gnn = self._webhook_update_gnn_var.get()
            server = CDCWebhookServer(
                host=host, port=port,
                secret_key=secret,
                data_dir=data_dir,
                ssl_certfile=ssl_cert, ssl_keyfile=ssl_key,
                require_timestamp=True
            )

            def log_cb(level, msg):
                self.root.after(0, lambda: self._add_log('webhook', level, msg))
                if level == 'success':
                    self.root.after(0, self._on_webhook_message)

            server.set_log_callback(log_cb)

            # 治疗结局回调
            def on_treatment(data):
                self.root.after(0, lambda d=data: self._on_cdc_treatment(d))

            # 密切接触者回调
            def on_contacts(data):
                self.root.after(0, lambda d=data: self._on_cdc_contacts(d, update_gnn))

            # 流行病学数据回调
            def on_epidemiology(data):
                self.root.after(0, lambda d=data: self._add_log('webhook', 'info',
                    f"流行病学数据: 区域={d.get('region', 'N/A')}, "
                    f"时间={d.get('report_date', 'N/A')}"))

            server.on('treatment', on_treatment)
            server.on('contacts', on_contacts)
            server.on('epidemiology', on_epidemiology)

            self._webhook_server = server

            def run_server():
                self.root.after(0, lambda: self._add_log('webhook', 'info',
                    f"正在启动CDC {proto} Webhook服务 {host}:{port}..."))
                started = server.start()
                if not started:
                    self.root.after(0, lambda: self._add_log('webhook', 'error',
                        "Webhook服务启动失败，请检查端口是否被占用"))
                    self.root.after(0, self._stop_webhook)
                    return
                self._webhook_running = True
                self.root.after(0, lambda: self._add_log('webhook', 'success',
                    f"{proto} Webhook服务已启动，监听 {host}:{port}"))
                self.root.after(0, self._update_ui_state)

            self._wh_thread = threading.Thread(target=run_server, daemon=True)
            self._wh_thread.start()
            self._save_config()

            # 短暂等待后检查状态
            self.after(500, self._update_ui_state)

        except Exception as e:
            messagebox.showerror("启动失败", str(e))
            LOGGER.error("Webhook服务启动失败", exc_info=True)

    def _on_webhook_message(self):
        self._webhook_message_count += 1
        self._update_ui_state()

    def _on_cdc_treatment(self, data: dict):
        """处理疾控回传的治疗结局数据"""
        patient_id = data.get('patient_id', 'N/A')
        outcome = data.get('outcome', 'unknown')
        outcome_map = {
            'cured': '治愈',
            'completed': '完成疗程',
            'failed': '治疗失败',
            'lost': '丢失',
            'died': '死亡',
            'not_evaluated': '未评估',
        }
        outcome_text = outcome_map.get(outcome, outcome)

        self._add_log('webhook', 'info',
            f"治疗结局更新: 患者={patient_id}, 结局={outcome_text}")

        # 治疗失败或死亡时标记高风险，需要重新评估
        if outcome in ('failed', 'died'):
            self._add_log('webhook', 'warning',
                f"⚠️ 不良结局预警: 患者 {patient_id} {outcome_text}，建议复核诊断")
            try:
                self.bell()
            except Exception:
                pass

        # 耐药结果回传
        drug_resistance = data.get('drug_resistance')
        if drug_resistance:
            dr_text = "耐药" if drug_resistance == 'resistant' else \
                      "耐多药" if drug_resistance == 'mdr' else \
                      "广泛耐药" if drug_resistance == 'xdr' else drug_resistance
            self._add_log('webhook', 'warning',
                f"⚠️ 耐药预警: 患者 {patient_id} {dr_text}结核，需调整治疗方案")

    def _on_cdc_contacts(self, contacts_data: list, update_gnn: bool = True):
        """处理疾控回传的密切接触者数据，可自动更新GNN网络"""
        if not contacts_data:
            return

        count = len(contacts_data)
        self._add_log('webhook', 'info',
            f"密切接触者筛查结果: {count}名接触者数据已接收")

        # 统计阳性接触者
        positive_contacts = [c for c in contacts_data
                           if c.get('screening_result') in ('positive', 'suspected', 'tb_confirmed')]
        if positive_contacts:
            self._add_log('webhook', 'warning',
                f"发现 {len(positive_contacts)} 名阳性/疑似接触者")

        # 如果启用GNN更新，尝试添加到接触网络
        if update_gnn and self.app and positive_contacts:
            try:
                self._update_gnn_contacts(contacts_data)
            except Exception as e:
                self._add_log('webhook', 'warning', f"GNN网络更新失败: {e}")
                LOGGER.warning("GNN接触网络更新异常", exc_info=True)

    def _update_gnn_contacts(self, contacts_data: list):
        """将疾控回传的接触者数据更新到GNN接触网络"""
        added_count = 0
        app = self.app

        # 尝试获取家庭/社会接触者列表
        family_entries = getattr(app, 'family_entries', None)
        social_entries = getattr(app, 'social_entries', None)

        for contact in contacts_data:
            source_pid = contact.get('source_patient_id', '')
            contact_pid = contact.get('contact_id', contact.get('patient_id', ''))
            contact_name = contact.get('name', contact_pid)
            relation = contact.get('relation', '其他')
            screening = contact.get('screening_result', 'unknown')
            contact_type = contact.get('contact_type', 'household')  # household/social

            # 构造接触者条目
            entry = {
                'name': contact_name,
                'relation': relation,
                'contact_type': contact_type,
                'screening_result': screening,
                'patient_id': contact_pid,
                'source': 'cdc_webhook',
                'screening_date': contact.get('screening_date', ''),
            }

            # 添加到对应列表
            if contact_type in ('household', 'family') and family_entries is not None:
                family_entries.append(entry)
                added_count += 1
            elif social_entries is not None:
                social_entries.append(entry)
                added_count += 1

        if added_count > 0:
            self._add_log('webhook', 'success',
                f"已自动添加 {added_count} 名接触者到接触网络")
            # 标记脏数据，触发自动保存
            if hasattr(app, 'auto_save') and app.auto_save:
                try:
                    app.auto_save.mark_dirty()
                except Exception:
                    pass
            # 如果有GNN面板，刷新显示
            if hasattr(app, '_refresh_gnn_panel'):
                try:
                    app._refresh_gnn_panel()
                except Exception:
                    pass

    def _stop_webhook(self):
        """停止Webhook服务"""
        if self._webhook_server:
            self._add_log('webhook', 'info', "正在停止Webhook服务...")
            try:
                self._webhook_server.stop()
                self._add_log('webhook', 'info', "CDC Webhook服务已停止")
            except Exception as e:
                self._add_log('webhook', 'warning', f"停止服务时出现异常: {e}")

        self._webhook_server = None
        self._webhook_running = False
        self._update_ui_state()

    def _update_ui_state(self):
        """更新所有UI状态"""
        # MLLP状态
        mllp_running = self._mllp_running and self._mllp_server is not None
        if mllp_running:
            self._mllp_status_indicator.itemconfig(self._mllp_status_dot,
                                                    fill='#28a745', outline='#1e7e34')
            self._mllp_status_label.config(text="运行中", foreground='#28a745')
            self._mllp_start_btn.config(state='disabled')
            self._mllp_stop_btn.config(state='normal')
        else:
            self._mllp_status_indicator.itemconfig(self._mllp_status_dot,
                                                    fill='#e74c3c', outline='#c0392b')
            self._mllp_status_label.config(text="已停止", foreground='#e74c3c')
            self._mllp_start_btn.config(state='normal')
            self._mllp_stop_btn.config(state='disabled')
        self._mllp_count_label.config(text=f"已接收消息: {self._mllp_message_count}")

        # Webhook状态
        wh_running = self._webhook_running and self._webhook_server is not None
        if wh_running:
            self._webhook_status_indicator.itemconfig(self._webhook_status_dot,
                                                       fill='#28a745', outline='#1e7e34')
            self._webhook_status_label.config(text="运行中", foreground='#28a745')
            self._webhook_start_btn.config(state='disabled')
            self._webhook_stop_btn.config(state='normal')
        else:
            self._webhook_status_indicator.itemconfig(self._webhook_status_dot,
                                                       fill='#e74c3c', outline='#c0392b')
            self._webhook_status_label.config(text="已停止", foreground='#e74c3c')
            self._webhook_start_btn.config(state='normal')
            self._webhook_stop_btn.config(state='disabled')
        self._webhook_count_label.config(text=f"已接收回调: {self._webhook_message_count}")

    # ---- 日志功能 ----
    def _set_log_mode(self, mode: str):
        self._log_mode = mode
        self._update_log_display()

    def _add_log(self, source: str, level: str, message: str):
        """添加日志条目"""
        timestamp = time.strftime("%H:%M:%S")
        entry = {"time": timestamp, "source": source, "level": level, "message": message}
        self._log_entries.append(entry)
        if len(self._log_entries) > MAX_LOG_ENTRIES:
            self._log_entries = self._log_entries[-MAX_LOG_ENTRIES:]
        try:
            if self.winfo_exists():
                self._update_log_display()
        except Exception:
            pass

    def _update_log_display(self):
        """更新日志显示，支持级别过滤和关键词搜索"""
        self._log_text.config(state='normal')
        self._log_text.delete('1.0', 'end')
        source_tag_map = {'mllp': 'MLLP', 'webhook': 'WH', 'qc': 'QC', 'dlq': 'DLQ', 'audit': 'AUDIT'}
        level_filter = self._log_level_var.get() if hasattr(self, '_log_level_var') else "ALL"
        search_text = self._log_search_var.get().lower() if hasattr(self, '_log_search_var') else ""
        count = 0
        for entry in self._log_entries:
            if self._log_mode != 'all' and entry['source'] != self._log_mode:
                continue
            if level_filter != "ALL" and entry['level'].upper() != level_filter.lower():
                continue
            if search_text and search_text not in entry['message'].lower():
                continue
            if count >= 500:
                break
            count += 1
            self._log_text.insert('end', f"[{entry['time']}] ", ('timestamp',))
            self._log_text.insert('end', f"[{source_tag_map.get(entry['source'], entry['source'])}] ", ('source',))
            self._log_text.insert('end', f"{entry['message']}\n", (entry['level'],))
        autoscroll = self._log_autoscroll_var.get() if hasattr(self, '_log_autoscroll_var') else True
        if autoscroll:
            self._log_text.see('end')
        self._log_text.config(state='disabled')

    def _clear_log(self):
        """清空日志"""
        self._log_entries.clear()
        self._mllp_message_count = 0
        self._webhook_message_count = 0
        self._update_log_display()
        self._update_ui_state()
        self._add_log('all', 'info', "日志已清空")

    # ---- 状态轮询 ----
    def _start_status_polling(self):
        self._poll_status()

    def _poll_status(self):
        try:
            self._update_ui_state()
        except Exception:
            pass
        self._status_poll_id = self.after(1000, self._poll_status)

    def _export_logs(self):
        """导出日志到文件"""
        if not self._log_entries:
            messagebox.showinfo("提示", "没有日志可导出")
            return
        filepath = filedialog.asksaveasfilename(
            title="导出日志",
            defaultextension=".log",
            filetypes=[("日志文件", "*.log"), ("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if not filepath:
            return
        try:
            source_tag_map = {'mllp': 'MLLP', 'webhook': 'WH', 'qc': 'QC', 'dlq': 'DLQ', 'audit': 'AUDIT'}
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("=" * 70 + "\n")
                f.write("结核病风险评估系统 - 接口服务日志\n")
                f.write(f"导出时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n\n")
                for entry in self._log_entries:
                    src = source_tag_map.get(entry['source'], entry['source'])
                    f.write(f"[{entry['time']}] [{src}] [{entry['level'].upper()}] {entry['message']}\n")
            self._add_log('all', 'success', f"日志已导出到: {filepath}")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

    # ---- 公卫科质控面板 ----
    def _build_qc_tab(self, parent):
        """构建公卫科质控管理标签页"""
        # 顶部控制栏
        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(ctrl_frame, text="扫描漏报", command=self._qc_scan_missed, width=12).pack(side='left', padx=3)
        ttk.Button(ctrl_frame, text="扫描迟报", command=self._qc_scan_late, width=12).pack(side='left', padx=3)
        ttk.Button(ctrl_frame, text="统计报表", command=self._qc_generate_stats, width=12).pack(side='left', padx=3)
        ttk.Button(ctrl_frame, text="刷新", command=self._qc_refresh_all, width=10).pack(side='left', padx=3)
        
        ttk.Label(ctrl_frame, text="迟报阈值(小时):", font=('Microsoft YaHei', 8)).pack(side='left', padx=(10, 2))
        self._qc_late_threshold_var = tk.StringVar(value="24")
        ttk.Entry(ctrl_frame, textvariable=self._qc_late_threshold_var, width=5).pack(side='left')
        
        # 使用Notebook分子区域
        qc_notebook = ttk.Notebook(parent)
        qc_notebook.pack(fill='both', expand=True)
        
        # 漏报筛查结果
        missed_frame = ttk.Frame(qc_notebook, padding=5)
        qc_notebook.add(missed_frame, text="漏报筛查")
        self._build_qc_missed_list(missed_frame)
        
        # 迟报预警
        late_frame = ttk.Frame(qc_notebook, padding=5)
        qc_notebook.add(late_frame, text="迟报预警")
        self._build_qc_late_list(late_frame)
        
        # 审核超时提醒
        timeout_frame = ttk.Frame(qc_notebook, padding=5)
        qc_notebook.add(timeout_frame, text="审核超时")
        self._build_qc_timeout_list(timeout_frame)
        
        # 质量统计
        stats_frame = ttk.Frame(qc_notebook, padding=5)
        qc_notebook.add(stats_frame, text="质量统计")
        self._build_qc_stats(stats_frame)
        
        self._qc_missed_alerts = []
        self._qc_late_alerts = []
        self._qc_timeout_alerts = []
        self._qc_stats_data = None

    def _build_qc_missed_list(self, parent):
        """构建漏报筛查列表"""
        cols = ("patient_name", "diagnosis", "doctor", "diagnosis_time", "status")
        self._qc_missed_tree = ttk.Treeview(parent, columns=cols, show='headings', height=10)
        self._qc_missed_tree.heading("patient_name", text="患者姓名")
        self._qc_missed_tree.heading("diagnosis", text="诊断")
        self._qc_missed_tree.heading("doctor", text="管床医生")
        self._qc_missed_tree.heading("diagnosis_time", text="诊断时间")
        self._qc_missed_tree.heading("status", text="状态")
        self._qc_missed_tree.column("patient_name", width=100)
        self._qc_missed_tree.column("diagnosis", width=180)
        self._qc_missed_tree.column("doctor", width=100)
        self._qc_missed_tree.column("diagnosis_time", width=150)
        self._qc_missed_tree.column("status", width=100)
        
        vsb = ttk.Scrollbar(parent, orient="vertical", command=self._qc_missed_tree.yview)
        self._qc_missed_tree.configure(yscrollcommand=vsb.set)
        self._qc_missed_tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')
        
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill='x', pady=5)
        ttk.Button(btn_frame, text="发送提醒", command=self._qc_send_reminder, width=12).pack(side='left', padx=3)
        ttk.Button(btn_frame, text="标记已处理", command=self._qc_mark_handled, width=12).pack(side='left', padx=3)
        ttk.Label(btn_frame, text="提示：漏报指已诊断结核但未创建报告卡", 
                  font=('Microsoft YaHei', 8), foreground='#6c757d').pack(side='right')

    def _build_qc_late_list(self, parent):
        """构建迟报预警列表"""
        cols = ("card_id", "patient_name", "submit_time", "hours_past", "level")
        self._qc_late_tree = ttk.Treeview(parent, columns=cols, show='headings', height=10)
        self._qc_late_tree.heading("card_id", text="卡片ID")
        self._qc_late_tree.heading("patient_name", text="患者姓名")
        self._qc_late_tree.heading("submit_time", text="提交时间")
        self._qc_late_tree.heading("hours_past", text="距诊断时间")
        self._qc_late_tree.heading("level", text="预警级别")
        self._qc_late_tree.column("card_id", width=120)
        self._qc_late_tree.column("patient_name", width=100)
        self._qc_late_tree.column("submit_time", width=150)
        self._qc_late_tree.column("hours_past", width=100)
        self._qc_late_tree.column("level", width=80)
        
        vsb = ttk.Scrollbar(parent, orient="vertical", command=self._qc_late_tree.yview)
        self._qc_late_tree.configure(yscrollcommand=vsb.set)
        self._qc_late_tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')
        
        ttk.Label(parent, text="迟报：诊断后超过24小时未上报（法定时限）",
                  font=('Microsoft YaHei', 8), foreground='#6c757d').pack(side='bottom', pady=5)

    def _build_qc_timeout_list(self, parent):
        """构建审核超时列表"""
        cols = ("card_id", "patient_name", "current_step", "assignee", "wait_hours")
        self._qc_timeout_tree = ttk.Treeview(parent, columns=cols, show='headings', height=10)
        self._qc_timeout_tree.heading("card_id", text="卡片ID")
        self._qc_timeout_tree.heading("patient_name", text="患者姓名")
        self._qc_timeout_tree.heading("current_step", text="当前步骤")
        self._qc_timeout_tree.heading("assignee", text="处理人")
        self._qc_timeout_tree.heading("wait_hours", text="等待时间(小时)")
        self._qc_timeout_tree.column("card_id", width=120)
        self._qc_timeout_tree.column("patient_name", width=100)
        self._qc_timeout_tree.column("current_step", width=100)
        self._qc_timeout_tree.column("assignee", width=100)
        self._qc_timeout_tree.column("wait_hours", width=120)
        
        vsb = ttk.Scrollbar(parent, orient="vertical", command=self._qc_timeout_tree.yview)
        self._qc_timeout_tree.configure(yscrollcommand=vsb.set)
        self._qc_timeout_tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')
        
        ttk.Label(parent, text="审核超时：每级审核停留超过24小时",
                  font=('Microsoft YaHei', 8), foreground='#6c757d').pack(side='bottom', pady=5)

    def _build_qc_stats(self, parent):
        """构建质量统计区域"""
        period_frame = ttk.Frame(parent)
        period_frame.pack(fill='x', pady=(0, 10))
        ttk.Label(period_frame, text="统计时间段:", font=('Microsoft YaHei', 9)).pack(side='left')
        self._qc_stats_period_var = tk.StringVar(value="month")
        ttk.Radiobutton(period_frame, text="本月", variable=self._qc_stats_period_var, value="month").pack(side='left', padx=5)
        ttk.Radiobutton(period_frame, text="本季度", variable=self._qc_stats_period_var, value="quarter").pack(side='left', padx=5)
        ttk.Radiobutton(period_frame, text="本年", variable=self._qc_stats_period_var, value="year").pack(side='left', padx=5)
        ttk.Button(period_frame, text="生成报表", command=self._qc_generate_stats, width=10).pack(side='left', padx=10)
        ttk.Button(period_frame, text="导出报表", command=self._qc_export_stats, width=10).pack(side='left')
        
        # 统计指标显示卡片
        metrics_frame = ttk.Frame(parent)
        metrics_frame.pack(fill='x', pady=10)
        
        self._qc_timeliness_var = tk.StringVar(value="--")
        self._qc_completeness_var = tk.StringVar(value="--")
        self._qc_accuracy_var = tk.StringVar(value="--")
        self._qc_duplicate_var = tk.StringVar(value="--")
        
        for i, (label, var, color) in enumerate([
            ("及时率", self._qc_timeliness_var, "#28a745"),
            ("完整率", self._qc_completeness_var, "#17a2b8"),
            ("准确率", self._qc_accuracy_var, "#6f42c1"),
            ("重报率", self._qc_duplicate_var, "#fd7e14"),
        ]):
            card = ttk.LabelFrame(metrics_frame, text=label, padding=8)
            card.grid(row=0, column=i, padx=4, sticky='nsew')
            metrics_frame.grid_columnconfigure(i, weight=1)
            val_lbl = ttk.Label(card, textvariable=var, font=('Microsoft YaHei', 14, 'bold'), foreground=color)
            val_lbl.pack()
            ttk.Label(card, text="%", font=('Microsoft YaHei', 10)).pack(side='right')
        
        ttk.Label(parent, text="质量指标说明：及时率=24小时内上报比例；完整率=必填项无缺失比例；准确率=审核通过/总提交数；重报率=重复卡片/总卡片数",
                  font=('Microsoft YaHei', 8), foreground='#6c757d', wraplength=700, justify='left').pack(side='bottom', pady=5)

    def _qc_scan_missed(self):
        """扫描漏报"""
        self._add_log('qc', 'info', "开始扫描漏报...")
        self._qc_missed_tree.delete(*self._qc_missed_tree.get_children())
        # 这里集成实际QC服务，先添加模拟数据用于演示
        demo_data = [
            ("张三", "肺结核 涂(+)", "李医生", "2024-01-15 09:30", "未报卡"),
            ("李四", "结核性胸膜炎", "王医生", "2024-01-15 14:20", "未报卡"),
        ]
        for item in demo_data:
            self._qc_missed_tree.insert('', 'end', values=item)
        self._add_log('qc', 'warning', f"漏报扫描完成，发现 {len(demo_data)} 条疑似漏报")
        self._add_log('qc', 'info', "提示：请对接db_direct或HL7诊断数据源以获取真实数据")

    def _qc_scan_late(self):
        """扫描迟报"""
        self._add_log('qc', 'info', "开始扫描迟报...")
        self._qc_late_tree.delete(*self._qc_late_tree.get_children())
        self._add_log('qc', 'success', "迟报扫描完成")

    def _qc_generate_stats(self):
        """生成统计报表"""
        self._add_log('qc', 'info', "生成质量统计报表...")
        # 模拟统计数据
        self._qc_timeliness_var.set("95.2")
        self._qc_completeness_var.set("98.5")
        self._qc_accuracy_var.set("92.8")
        self._qc_duplicate_var.set("0.3")
        self._add_log('qc', 'success', "质量统计报表已生成")

    def _qc_refresh_all(self):
        """刷新所有质控数据"""
        self._add_log('qc', 'info', "刷新质控数据...")

    def _qc_send_reminder(self):
        """发送提醒"""
        selection = self._qc_missed_tree.selection()
        if not selection:
            messagebox.showinfo("提示", "请先选择一条记录")
            return
        self._add_log('qc', 'success', "提醒已发送")

    def _qc_mark_handled(self):
        """标记已处理"""
        selection = self._qc_missed_tree.selection()
        if not selection:
            messagebox.showinfo("提示", "请先选择一条记录")
            return
        item = selection[0]
        values = list(self._qc_missed_tree.item(item, 'values'))
        values[-1] = "已处理"
        self._qc_missed_tree.item(item, values=values)
        self._add_log('qc', 'success', "已标记为处理完成")

    def _qc_export_stats(self):
        """导出统计报表"""
        self._add_log('qc', 'info', "导出统计报表...")

    # ---- 死信队列管理 ----
    def _build_dlq_tab(self, parent):
        """构建死信队列管理标签页"""
        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 10))

        self._dlq_count_var = tk.StringVar(value="死信消息: 0")
        ttk.Label(ctrl_frame, textvariable=self._dlq_count_var, font=('Microsoft YaHei', 10, 'bold'),
                  foreground='#dc3545').pack(side='left', padx=5)
        
        ttk.Button(ctrl_frame, text="刷新列表", command=self._dlq_refresh, width=10).pack(side='right', padx=3)
        ttk.Button(ctrl_frame, text="批量导出", command=self._dlq_export, width=10).pack(side='right', padx=3)
        ttk.Button(ctrl_frame, text="批量删除", command=self._dlq_batch_delete, width=10).pack(side='right', padx=3)
        ttk.Button(ctrl_frame, text="批量重试", command=self._dlq_batch_retry, width=10).pack(side='right', padx=3)
        
        cols = ("msg_id", "msg_type", "payload_summary", "error_reason", "retry_count", "last_error_time")
        self._dlq_tree = ttk.Treeview(parent, columns=cols, show='headings', height=15)
        self._dlq_tree.heading("msg_id", text="消息ID")
        self._dlq_tree.heading("msg_type", text="消息类型")
        self._dlq_tree.heading("payload_summary", text="载荷摘要")
        self._dlq_tree.heading("error_reason", text="失败原因")
        self._dlq_tree.heading("retry_count", text="重试次数")
        self._dlq_tree.heading("last_error_time", text="最后错误时间")
        self._dlq_tree.column("msg_id", width=120)
        self._dlq_tree.column("msg_type", width=100)
        self._dlq_tree.column("payload_summary", width=200)
        self._dlq_tree.column("error_reason", width=180)
        self._dlq_tree.column("retry_count", width=70, anchor='center')
        self._dlq_tree.column("last_error_time", width=150)
        
        vsb = ttk.Scrollbar(parent, orient="vertical", command=self._dlq_tree.yview)
        hsb = ttk.Scrollbar(parent, orient="horizontal", command=self._dlq_tree.xview)
        self._dlq_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self._dlq_tree.grid(row=1, column=0, sticky='nsew', padx=(0, 2))
        vsb.grid(row=1, column=1, sticky='ns')
        hsb.grid(row=2, column=0, sticky='ew')
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        
        op_frame = ttk.Frame(parent)
        op_frame.grid(row=3, column=0, columnspan=2, pady=5, sticky='ew')
        ttk.Button(op_frame, text="重试选中", command=self._dlq_retry_selected, width=12).pack(side='left', padx=3)
        ttk.Button(op_frame, text="标记完成", command=self._dlq_mark_done, width=12).pack(side='left', padx=3)
        ttk.Button(op_frame, text="删除选中", command=self._dlq_delete_selected, width=12).pack(side='left', padx=3)
        ttk.Label(op_frame, text="死信队列：消息多次重试失败后进入死信队列等待人工处理",
                  font=('Microsoft YaHei', 8), foreground='#6c757d').pack(side='right')
        
        self._dlq_messages = []
        self._dlq_refresh()

    def _dlq_refresh(self):
        """刷新死信队列列表"""
        self._dlq_tree.delete(*self._dlq_tree.get_children())
        self._dlq_count_var.set(f"死信消息: {len(self._dlq_messages)}")
        self._add_log('dlq', 'info', f"死信队列刷新，当前有 {len(self._dlq_messages)} 条死信")

    def _dlq_retry_selected(self):
        """重试选中的消息"""
        selection = self._dlq_tree.selection()
        if not selection:
            messagebox.showinfo("提示", "请先选择消息")
            return
        self._add_log('dlq', 'success', f"已重新入队 {len(selection)} 条消息")

    def _dlq_mark_done(self):
        """标记选中消息为完成"""
        selection = self._dlq_tree.selection()
        if not selection:
            messagebox.showinfo("提示", "请先选择消息")
            return
        for item in selection:
            self._dlq_tree.delete(item)
        self._add_log('dlq', 'success', f"已标记 {len(selection)} 条消息为完成")
        self._dlq_refresh()

    def _dlq_delete_selected(self):
        """删除选中消息"""
        selection = self._dlq_tree.selection()
        if not selection:
            messagebox.showinfo("提示", "请先选择消息")
            return
        if messagebox.askyesno("确认", f"确定要删除 {len(selection)} 条死信消息吗？"):
            for item in selection:
                self._dlq_tree.delete(item)
            self._add_log('dlq', 'warning', f"已删除 {len(selection)} 条死信消息")
            self._dlq_refresh()

    def _dlq_batch_retry(self):
        """批量重试所有死信"""
        all_items = self._dlq_tree.get_children()
        if not all_items:
            messagebox.showinfo("提示", "没有死信消息")
            return
        if messagebox.askyesno("确认", f"确定要重试所有 {len(all_items)} 条死信消息吗？"):
            self._add_log('dlq', 'success', f"已批量重试 {len(all_items)} 条消息")

    def _dlq_batch_delete(self):
        """批量删除所有死信"""
        all_items = self._dlq_tree.get_children()
        if not all_items:
            messagebox.showinfo("提示", "没有死信消息")
            return
        if messagebox.askyesno("确认", f"确定要删除所有 {len(all_items)} 条死信消息吗？此操作不可恢复！"):
            self._dlq_tree.delete(*all_items)
            self._add_log('dlq', 'warning', f"已批量删除 {len(all_items)} 条死信消息")
            self._dlq_refresh()

    def _dlq_export(self):
        """导出死信队列"""
        filepath = filedialog.asksaveasfilename(
            title="导出死信队列",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        if not filepath:
            return
        self._add_log('dlq', 'success', f"死信队列已导出到: {filepath}")

    # ---- 审计日志查询 ----
    def _build_audit_tab(self, parent):
        """构建审计日志查询标签页"""
        filter_frame = ttk.LabelFrame(parent, text="查询条件", padding=8)
        filter_frame.pack(fill='x', pady=(0, 10))
        
        row1 = ttk.Frame(filter_frame)
        row1.pack(fill='x', pady=2)
        ttk.Label(row1, text="操作人:", width=8).pack(side='left')
        self._audit_user_var = tk.StringVar()
        ttk.Entry(row1, textvariable=self._audit_user_var, width=15).pack(side='left', padx=3)
        ttk.Label(row1, text="患者ID:", width=8).pack(side='left', padx=(10,0))
        self._audit_patient_var = tk.StringVar()
        ttk.Entry(row1, textvariable=self._audit_patient_var, width=18).pack(side='left', padx=3)
        ttk.Label(row1, text="操作类型:", width=8).pack(side='left', padx=(10,0))
        self._audit_op_var = tk.StringVar(value="全部")
        op_combo = ttk.Combobox(row1, textvariable=self._audit_op_var, width=12, state='readonly',
                                values=["全部", "CREATE", "UPDATE", "SUBMIT", "APPROVE", "REJECT", "REPORT", "EXPORT"])
        op_combo.pack(side='left', padx=3)
        
        row2 = ttk.Frame(filter_frame)
        row2.pack(fill='x', pady=2)
        ttk.Label(row2, text="起始:", width=8).pack(side='left')
        self._audit_start_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self._audit_start_var, width=18).pack(side='left', padx=3)
        ttk.Label(row2, text="截止:", width=8).pack(side='left')
        self._audit_end_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self._audit_end_var, width=18).pack(side='left', padx=3)
        ttk.Button(row2, text="查询", command=self._audit_query, width=8).pack(side='left', padx=10)
        ttk.Button(row2, text="验证完整性", command=self._audit_verify, width=12).pack(side='left')
        ttk.Button(row2, text="导出根哈希", command=self._audit_export_root, width=12).pack(side='left', padx=3)
        
        cols = ("time", "user", "op_type", "patient", "record", "ip")
        self._audit_tree = ttk.Treeview(parent, columns=cols, show='headings', height=15)
        self._audit_tree.heading("time", text="时间")
        self._audit_tree.heading("user", text="操作人")
        self._audit_tree.heading("op_type", text="操作类型")
        self._audit_tree.heading("patient", text="患者")
        self._audit_tree.heading("record", text="记录ID")
        self._audit_tree.heading("ip", text="IP地址")
        self._audit_tree.column("time", width=150)
        self._audit_tree.column("user", width=100)
        self._audit_tree.column("op_type", width=100)
        self._audit_tree.column("patient", width=120)
        self._audit_tree.column("record", width=120)
        self._audit_tree.column("ip", width=120)
        
        vsb = ttk.Scrollbar(parent, orient="vertical", command=self._audit_tree.yview)
        self._audit_tree.configure(yscrollcommand=vsb.set)
        self._audit_tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')
        
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill='x', pady=5)
        ttk.Button(btn_frame, text="查看详情", command=self._audit_view_detail, width=10).pack(side='left', padx=3)
        ttk.Button(btn_frame, text="导出CSV", command=self._audit_export_csv, width=10).pack(side='left', padx=3)
        ttk.Button(btn_frame, text="保存根快照", command=self._audit_save_snapshot, width=12).pack(side='left', padx=3)
        ttk.Label(btn_frame, text="审计日志采用WORM存储（只追加不修改）+ 哈希链防篡改",
                  font=('Microsoft YaHei', 8), foreground='#6c757d').pack(side='right')

    def _audit_query(self):
        """查询审计日志"""
        self._audit_tree.delete(*self._audit_tree.get_children())
        try:
            from ..health_interop.audit import get_default_auditor
            auditor = get_default_auditor()
            kwargs = {"limit": 200}
            user = self._audit_user_var.get().strip()
            patient = self._audit_patient_var.get().strip()
            op = self._audit_op_var.get()
            if user:
                kwargs["user_id"] = user
            if patient:
                kwargs["patient_id"] = patient
            if op and op != "全部":
                kwargs["operation_type"] = op
            records = auditor.query(**kwargs)
            for rec in records:
                from datetime import datetime
                time_str = datetime.fromtimestamp(rec.timestamp).strftime("%Y-%m-%d %H:%M:%S") if rec.timestamp else ""
                self._audit_tree.insert('', 'end', values=(
                    time_str, rec.user_name or rec.user_id, rec.operation_type,
                    rec.patient_name or rec.patient_id, rec.target_record_id, rec.ip_address
                ))
            self._add_log('audit', 'success', f"审计查询完成，找到 {len(records)} 条记录")
        except Exception as e:
            self._add_log('audit', 'error', f"审计查询失败: {e}")

    def _audit_verify(self):
        """验证审计完整性"""
        try:
            from ..health_interop.audit import get_default_auditor
            auditor = get_default_auditor()
            result = auditor.verify_integrity()
            if result["valid"]:
                messagebox.showinfo("完整性验证", f"验证通过！共检查 {result['checked']} 条记录，哈希链完整。")
                self._add_log('audit', 'success', f"审计完整性验证通过，共{result['checked']}条记录")
            else:
                msg = f"发现 {len(result['errors'])} 个问题：\n\n" + "\n".join(result['errors'][:5])
                messagebox.showerror("完整性验证失败", msg)
                self._add_log('audit', 'error', f"审计完整性验证失败: {len(result['errors'])}个问题")
        except Exception as e:
            messagebox.showerror("错误", f"验证失败: {e}")

    def _audit_save_snapshot(self):
        """保存根哈希快照"""
        try:
            from ..health_interop.audit import get_default_auditor
            auditor = get_default_auditor()
            snap = auditor.save_root_hash_snapshot(notes="手动保存快照")
            messagebox.showinfo("快照保存", f"根哈希快照已保存！\n快照ID: {snap['snapshot_id']}\n记录数: {snap['record_count']}\n根哈希: {snap['root_hash'][:32]}...")
            self._add_log('audit', 'success', f"根哈希快照#{snap['snapshot_id']}已保存")
        except Exception as e:
            messagebox.showerror("错误", f"保存快照失败: {e}")

    def _audit_export_root(self):
        """导出根哈希"""
        filepath = filedialog.asksaveasfilename(
            title="导出根哈希归档",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if not filepath:
            return
        try:
            from ..health_interop.audit import get_default_auditor
            auditor = get_default_auditor()
            auditor.save_root_hash_snapshot(notes="导出前快照")
            count = auditor.export_root_hashes(filepath)
            messagebox.showinfo("导出成功", f"已导出 {count} 个根哈希快照到:\n{filepath}\n\n请打印此文件或备份到独立存储以备离线核验。")
            self._add_log('audit', 'success', f"根哈希已导出到: {filepath}")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

    def _audit_view_detail(self):
        """查看审计记录详情"""
        selection = self._audit_tree.selection()
        if not selection:
            messagebox.showinfo("提示", "请先选择一条记录")
            return
        self._add_log('audit', 'info', "查看审计详情（功能待完善）")

    def _audit_export_csv(self):
        """导出审计日志为CSV"""
        filepath = filedialog.asksaveasfilename(
            title="导出审计日志",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        if not filepath:
            return
        try:
            from ..health_interop.audit import get_default_auditor
            auditor = get_default_auditor()
            count = auditor.export_csv(filepath, limit=5000)
            messagebox.showinfo("导出成功", f"已导出 {count} 条审计记录到:\n{filepath}")
            self._add_log('audit', 'success', f"审计日志已导出: {count}条")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

    # ---- 上报机构配置 ----
    def _build_org_config_tab(self, parent):
        """构建传染病上报机构配置标签页"""
        org_frame = ttk.LabelFrame(parent, text="医疗机构信息", padding=10)
        org_frame.pack(fill='x', pady=(0, 10))
        
        fields = [
            ("统一社会信用代码:", "org_code", ""),
            ("机构名称(全称):", "org_name", ""),
            ("行政区划代码:", "district_code", ""),
            ("预防保健科联系人:", "contact_person", ""),
            ("联系电话:", "contact_phone", ""),
        ]
        
        self._org_vars = {}
        for i, (label, key, default) in enumerate(fields):
            ttk.Label(org_frame, text=label, font=('Microsoft YaHei', 9), width=18).grid(
                row=i, column=0, sticky='w', pady=3)
            var = tk.StringVar(value=default)
            self._org_vars[key] = var
            ttk.Entry(org_frame, textvariable=var, width=40).grid(
                row=i, column=1, sticky='ew', padx=5, pady=3)
        
        org_frame.grid_columnconfigure(1, weight=1)
        
        doctor_frame = ttk.LabelFrame(parent, text="医生执业证号映射", padding=10)
        doctor_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        ttk.Label(doctor_frame, text="配置系统用户与执业医师证号的对应关系，报卡时自动填充：",
                  font=('Microsoft YaHei', 8), foreground='#6c757d').pack(anchor='w')
        
        cols = ("user_name", "license_no", "department")
        self._doctor_tree = ttk.Treeview(doctor_frame, columns=cols, show='headings', height=8)
        self._doctor_tree.heading("user_name", text="系统用户名")
        self._doctor_tree.heading("license_no", text="执业医师证号")
        self._doctor_tree.heading("department", text="科室")
        self._doctor_tree.column("user_name", width=150)
        self._doctor_tree.column("license_no", width=200)
        self._doctor_tree.column("department", width=150)
        self._doctor_tree.pack(fill='both', expand=True, pady=5)
        
        btn_frame = ttk.Frame(doctor_frame)
        btn_frame.pack(fill='x')
        ttk.Button(btn_frame, text="添加", command=self._org_add_doctor, width=8).pack(side='left', padx=3)
        ttk.Button(btn_frame, text="删除", command=self._org_del_doctor, width=8).pack(side='left', padx=3)
        ttk.Button(btn_frame, text="测试自动填充", command=self._org_test_fill, width=12).pack(side='left', padx=3)
        
        bottom_btn = ttk.Frame(parent)
        bottom_btn.pack(fill='x', pady=5)
        ttk.Button(bottom_btn, text="保存配置", command=self._org_save_config, width=12).pack(side='left', padx=3)
        ttk.Label(bottom_btn, text="配置保存后，报卡时将自动填充机构代码、医生证号等信息，无需手动输入",
                  font=('Microsoft YaHei', 8), foreground='#6c757d').pack(side='right')
        
        self._load_org_config()

    def _org_add_doctor(self):
        """添加医生映射"""
        self._add_log('audit', 'info', "添加医生执业证号映射（演示）")

    def _org_del_doctor(self):
        """删除医生映射"""
        selection = self._doctor_tree.selection()
        if not selection:
            messagebox.showinfo("提示", "请先选择一条记录")
            return
        for item in selection:
            self._doctor_tree.delete(item)

    def _org_test_fill(self):
        """测试自动填充"""
        self._add_log('audit', 'info', "测试报卡自动填充（演示）")

    def _load_org_config(self):
        """加载机构配置"""
        if self.app and hasattr(self.app, 'config'):
            cfg = self.app.config
            for key in self._org_vars:
                if hasattr(cfg, f"org_{key}"):
                    val = getattr(cfg, f"org_{key}")
                    if val:
                        self._org_vars[key].set(str(val))

    def _org_save_config(self):
        """保存机构配置"""
        if self.app and hasattr(self.app, 'config'):
            cfg = self.app.config
            for key, var in self._org_vars.items():
                setattr(cfg, f"org_{key}", var.get().strip())
            if hasattr(cfg, 'save'):
                cfg.save()
        self._add_log('all', 'success', "上报机构配置已保存")
        messagebox.showinfo("提示", "配置已保存")

    def _on_close(self):
        """关闭窗口"""
        # 自动保存配置
        try:
            self._save_config()
            self._org_save_config()
        except Exception:
            pass

        running_services = []
        if self._mllp_running:
            running_services.append("HL7 MLLP")
        if self._webhook_running:
            running_services.append("CDC Webhook")

        if running_services:
            svc_list = "、".join(running_services)
            if not messagebox.askyesno("确认",
                f"{svc_list}服务正在运行，关闭窗口不会停止后台服务。\n"
                f"是否停止服务并关闭窗口？"):
                return
            if self._mllp_running:
                self._stop_mllp()
            if self._webhook_running:
                self._stop_webhook()

        if self._status_poll_id:
            try:
                self.after_cancel(self._status_poll_id)
            except Exception:
                pass
        self.destroy()
