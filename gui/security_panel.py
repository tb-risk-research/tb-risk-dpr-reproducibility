#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 数据安全与合规配置面板（Toplevel 对话框）。

功能：
- 角色切换：普通用户 / 评估员 / 管理员（不同角色可见字段不同）。
- 策略配置：本地优先存储、联网上传禁止、导出强化脱敏、关键操作审计。
- 合规说明展示（PIPL 及卫生健康行业规范要点）。
- 关键操作审计日志查看与完整性校验。
"""

import tkinter as tk
from tkinter import ttk, scrolledtext

from ..security import (
    Role, role_label, DataSecurityConfig, DataSecurityManager,
    SecurityAuditor, compliance_summary,
)

LOGGER = __import__("logging").getLogger("tb_risk.security.gui")


class SecurityPanel(tk.Toplevel):
    """数据安全与合规配置面板。"""

    def __init__(self, root, app=None):
        super().__init__(root)
        self.app = app
        self._manager = self._load_manager()
        self.title("数据安全与合规")
        self.geometry("640x560")
        self.transient(root)
        self.resizable(True, True)
        self.minsize(560, 480)

        self._build_ui()
        self._refresh_compliance()
        self._refresh_audit()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _load_manager(self):
        """优先复用 app 上的安全管理器，否则新建。"""
        if self.app is not None and hasattr(self.app, '_get_security'):
            mgr = self.app._get_security()
            if mgr is not None:
                return mgr
        return DataSecurityManager()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        main = ttk.Frame(self, padding=12)
        main.pack(fill='both', expand=True)

        # 角色选择
        role_frame = ttk.LabelFrame(main, text="角色权限控制")
        role_frame.pack(fill='x', pady=(0, 8))

        ttk.Label(role_frame, text="当前角色：").grid(
            row=0, column=0, padx=8, pady=6, sticky='w')
        self.role_var = tk.StringVar(value=self._manager.current_role)
        role_combo = ttk.Combobox(
            role_frame, textvariable=self.role_var, state='readonly',
            values=[Role.NORMAL, Role.ASSESSOR, Role.ADMIN],
            width=12)
        role_combo.grid(row=0, column=1, padx=4, pady=6, sticky='w')
        ttk.Button(role_frame, text="应用",
                   command=self._on_role_change).grid(
            row=0, column=2, padx=8, pady=6)

        ttk.Label(
            role_frame,
            text="普通用户：姓名打码，隐藏证件/电话/地址；\n"
                 "评估员：姓名/电话/地址脱敏，隐藏证件号；\n"
                 "管理员：完整可见（可关闭脱敏）。",
            foreground='#5a6c7d').grid(row=1, column=0, columnspan=3,
                                       padx=8, pady=(0, 6), sticky='w')

        # 策略配置
        cfg_frame = ttk.LabelFrame(main, text="安全策略配置")
        cfg_frame.pack(fill='x', pady=(0, 8))

        self.var_local = tk.BooleanVar(value=self._manager.config.local_only)
        self.var_upload = tk.BooleanVar(value=self._manager.config.network_upload_allowed)
        self.var_masking = tk.BooleanVar(value=self._manager.config.export_masking_enabled)
        self.var_audit = tk.BooleanVar(value=self._manager.config.audit_enabled)

        ttk.Checkbutton(cfg_frame, text="本地优先存储（数据仅存本机）",
                        variable=self.var_local).grid(
            row=0, column=0, padx=8, pady=4, sticky='w')
        ttk.Checkbutton(cfg_frame, text="允许联网上传（默认禁止）",
                        variable=self.var_upload).grid(
            row=0, column=1, padx=8, pady=4, sticky='w')
        ttk.Checkbutton(cfg_frame, text="导出统一强化脱敏",
                        variable=self.var_masking).grid(
            row=1, column=0, padx=8, pady=4, sticky='w')
        ttk.Checkbutton(cfg_frame, text="关键操作审计（评估/导出/导入）",
                        variable=self.var_audit).grid(
            row=1, column=1, padx=8, pady=4, sticky='w')

        ttk.Button(cfg_frame, text="保存策略",
                   command=self._on_save_config).grid(
            row=2, column=0, columnspan=2, padx=8, pady=6, sticky='w')

        # 合规说明
        comp_frame = ttk.LabelFrame(main, text="数据合规说明（个人信息保护法 PIPL & 卫生健康行业规范）")
        comp_frame.pack(fill='both', expand=True, pady=(0, 8))

        self.compliance_text = scrolledtext.ScrolledText(
            comp_frame, wrap='word', font=('Microsoft YaHei', 9),
            height=8, state='disabled')
        self.compliance_text.pack(fill='both', expand=True, padx=6, pady=6)

        # 审计日志
        audit_frame = ttk.LabelFrame(main, text="关键操作审计日志")
        audit_frame.pack(fill='both', expand=True)

        toolbar = ttk.Frame(audit_frame)
        toolbar.pack(fill='x', padx=6, pady=(6, 2))
        ttk.Button(toolbar, text="刷新",
                   command=self._refresh_audit).pack(side='left', padx=(0, 6))
        ttk.Button(toolbar, text="校验完整性",
                   command=self._on_verify_integrity).pack(side='left')

        cols = ('time', 'op', 'role', 'details')
        self.tree = ttk.Treeview(audit_frame, columns=cols, show='headings', height=6)
        self.tree.heading('time', text='时间')
        self.tree.heading('op', text='操作')
        self.tree.heading('role', text='角色')
        self.tree.heading('details', text='详情')
        self.tree.column('time', width=120, anchor='w')
        self.tree.column('op', width=70, anchor='center')
        self.tree.column('role', width=70, anchor='center')
        self.tree.column('details', width=300, anchor='w')
        self.tree.pack(fill='both', expand=True, padx=6, pady=(0, 6))

        btn = ttk.Button(self, text="关闭", command=self.destroy)
        btn.pack(pady=6)

    # ------------------------------------------------------------------
    # 交互
    # ------------------------------------------------------------------

    def _on_role_change(self):
        role = self.role_var.get()
        self._manager.set_role(role)
        self._manager.auditor.log_operation(SecurityAuditor.OPERATION_CONFIG,
                                            role=role, details={"action": "role_change"})
        self._refresh_compliance()
        self._refresh_audit()

    def _on_save_config(self):
        self._manager.set_config(
            local_only=self.var_local.get(),
            network_upload_allowed=self.var_upload.get(),
            export_masking_enabled=self.var_masking.get(),
            audit_enabled=self.var_audit.get(),
        )
        self._manager.auditor.log_operation(
            SecurityAuditor.OPERATION_CONFIG,
            role=self._manager.current_role,
            details={"action": "config_save"})
        self._refresh_compliance()

    def _refresh_compliance(self):
        try:
            text = compliance_summary(self._manager.config)
        except Exception:
            text = "合规说明生成失败。"
        self.compliance_text.config(state='normal')
        self.compliance_text.delete('1.0', 'end')
        self.compliance_text.insert('1.0', text)
        self.compliance_text.config(state='disabled')

    def _refresh_audit(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        try:
            records = self._manager.auditor.recent(limit=50)
        except Exception:
            records = []
        import datetime as _dt
        for rec in records:
            ts = rec.get('timestamp')
            try:
                time_str = _dt.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                time_str = ''
            details = rec.get('details', '')
            op = rec.get('operation_type', '')
            role = role_label(rec.get('user_role', ''))
            self.tree.insert('', 'end', values=(time_str, op, role, str(details)))

    def _on_verify_integrity(self):
        result = self._manager.auditor.verify_integrity()
        if result.get('valid'):
            msg = (f"审计链完整，共 {result.get('total', 0)} 条记录，"
                   f"已校验 {result.get('checked', 0)} 条。")
        else:
            msg = "审计链可能存在篡改！\n" + "\n".join(result.get('errors', []))
        from tkinter import messagebox
        messagebox.showinfo("完整性校验", msg)
