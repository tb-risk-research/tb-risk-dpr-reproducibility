#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""传染病报告卡多级审核工作流模块

实现符合医院管理要求的多级审核流程：
- 医生提交 → 科室主任审核 → 公共卫生科复核 → 院感科备案 → 上报疾控
- 支持驳回重填、退回修改、审核意见记录
- 完整状态机管理：草稿→待审核→审核中→审核通过→已驳回→已上报→上报成功/失败
- 每个节点记录操作人、操作时间、审核意见

符合：
- 《医疗机构病历管理规定》
- 《传染病信息报告管理规范》
- 医院感染管理要求
"""

from __future__ import annotations

import datetime
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from .audit import AuditLogger, get_default_auditor

LOGGER = logging.getLogger("tb_risk.health_interop.workflow")


# ============================================================================
# 状态与角色枚举
# ============================================================================

class CardStatus(Enum):
    """报告卡状态枚举"""
    DRAFT = "draft"                     # 草稿
    FILLED = "filled"                   # 已填写完成（待提交）
    PENDING_REVIEW = "pending_review"   # 待科室主任审核
    DEPT_REVIEWING = "dept_reviewing"   # 科室主任审核中
    DEPT_APPROVED = "dept_approved"     # 科室主任审核通过
    DEPT_REJECTED = "dept_rejected"     # 科室主任驳回
    PHC_REVIEWING = "phc_reviewing"     # 公共卫生科复核中
    PHC_APPROVED = "phc_approved"       # 公共卫生科复核通过
    PHC_REJECTED = "phc_rejected"       # 公共卫生科驳回
    INFECTION_REVIEWING = "infection_reviewing"  # 院感科备案中
    APPROVED = "approved"               # 全部审核通过（待上报）
    REPORTING = "reporting"             # 上报中
    REPORTED = "reported"               # 已上报
    REPORT_SUCCESS = "report_success"   # 上报成功
    REPORT_FAILED = "report_failed"     # 上报失败
    CANCELLED = "cancelled"             # 已作废
    ARCHIVED = "archived"               # 已归档
    # 订正报告相关状态
    CORRECTION_DRAFT = "correction_draft"       # 订正草稿
    CORRECTION_SUBMITTED = "correction_submitted"  # 订正已提交审核
    CORRECTION_APPROVED = "correction_approved"    # 订正审核通过
    CORRECTION_REPORTING = "correction_reporting"  # 订正上报中


class UserRole(Enum):
    """用户角色枚举"""
    DOCTOR = "doctor"                   # 临床医生（填报人）
    DEPT_DIRECTOR = "dept_director"     # 科室主任
    PHC_STAFF = "phc_staff"             # 公共卫生科人员
    INFECTION_STAFF = "infection_staff"  # 院感科人员
    ADMIN = "admin"                     # 系统管理员
    SYSTEM = "system"                   # 系统自动


class ReviewAction(Enum):
    """审核动作枚举"""
    SAVE_DRAFT = "save_draft"           # 保存草稿
    FILL_COMPLETE = "fill_complete"     # 填写完成
    SUBMIT = "submit"                   # 提交审核
    APPROVE = "approve"                 # 审核通过
    REJECT = "reject"                   # 驳回
    RETURN = "return"                   # 退回修改
    REPORT = "report"                   # 上报疾控
    REPORT_SUCCESS = "report_success"   # 疾控回调-上报成功
    REPORT_FAILED = "report_failed"     # 疾控回调-上报失败
    CANCEL = "cancel"                   # 作废
    ARCHIVE = "archive"                 # 归档
    # 订正报告相关动作
    INITIATE_CORRECTION = "initiate_correction"  # 发起订正
    SUBMIT_CORRECTION = "submit_correction"      # 提交订正审核
    CORRECTION_REPORT = "correction_report"      # 订正上报


# ============================================================================
# 状态转换表
# ============================================================================

# 定义合法的状态转换：列表格式，每条记录为 (当前状态, 动作, 角色, 目标状态, 下一个角色, 描述)
# 使用列表而不是字典，支持同一(状态,动作)对针对不同角色有不同转换
STATE_TRANSITIONS: List[Tuple[CardStatus, ReviewAction, Optional[UserRole],
                               CardStatus, Optional[UserRole], str]] = [
    # 草稿阶段
    (CardStatus.DRAFT, ReviewAction.SAVE_DRAFT, None,
        CardStatus.DRAFT, None, "保存草稿"),
    (CardStatus.DRAFT, ReviewAction.FILL_COMPLETE, UserRole.DOCTOR,
        CardStatus.FILLED, None, "医生完成填写"),
    
    # 已填写状态 - 医生可继续编辑或提交
    (CardStatus.FILLED, ReviewAction.FILL_COMPLETE, UserRole.DOCTOR,
        CardStatus.FILLED, None, "医生更新填写内容"),
    (CardStatus.FILLED, ReviewAction.SAVE_DRAFT, UserRole.DOCTOR,
        CardStatus.FILLED, None, "保存修改"),
    (CardStatus.FILLED, ReviewAction.SUBMIT, UserRole.DOCTOR,
        CardStatus.PENDING_REVIEW, UserRole.DEPT_DIRECTOR, "提交科室主任审核"),
    
    # 驳回状态 - 医生可以编辑修改后重新提交
    (CardStatus.DEPT_REJECTED, ReviewAction.FILL_COMPLETE, UserRole.DOCTOR,
        CardStatus.FILLED, None, "医生修改驳回内容"),
    (CardStatus.DEPT_REJECTED, ReviewAction.SUBMIT, UserRole.DOCTOR,
        CardStatus.PENDING_REVIEW, UserRole.DEPT_DIRECTOR, "修改后重新提交科室主任审核"),
    (CardStatus.PHC_REJECTED, ReviewAction.FILL_COMPLETE, UserRole.DOCTOR,
        CardStatus.FILLED, None, "医生修改驳回内容"),
    (CardStatus.PHC_REJECTED, ReviewAction.SUBMIT, UserRole.DOCTOR,
        CardStatus.PENDING_REVIEW, UserRole.DEPT_DIRECTOR, "修改后重新提交（从头审核）"),
    
    # 科室主任审核
    (CardStatus.PENDING_REVIEW, ReviewAction.APPROVE, UserRole.DEPT_DIRECTOR,
        CardStatus.DEPT_APPROVED, UserRole.PHC_STAFF, "科室主任审核通过，提交公共卫生科"),
    (CardStatus.PENDING_REVIEW, ReviewAction.REJECT, UserRole.DEPT_DIRECTOR,
        CardStatus.DEPT_REJECTED, UserRole.DOCTOR, "科室主任驳回"),
    (CardStatus.PENDING_REVIEW, ReviewAction.RETURN, UserRole.DEPT_DIRECTOR,
        CardStatus.FILLED, UserRole.DOCTOR, "科室主任退回修改"),
    
    # 公共卫生科复核
    (CardStatus.DEPT_APPROVED, ReviewAction.APPROVE, UserRole.PHC_STAFF,
        CardStatus.PHC_APPROVED, UserRole.INFECTION_STAFF, "公共卫生科复核通过，提交院感科备案"),
    (CardStatus.DEPT_APPROVED, ReviewAction.REJECT, UserRole.PHC_STAFF,
        CardStatus.PHC_REJECTED, UserRole.DOCTOR, "公共卫生科驳回"),
    (CardStatus.DEPT_APPROVED, ReviewAction.RETURN, UserRole.PHC_STAFF,
        CardStatus.FILLED, UserRole.DOCTOR, "公共卫生科退回修改"),
    
    # 院感科备案（人工审核和系统自动备案均支持）
    (CardStatus.PHC_APPROVED, ReviewAction.APPROVE, UserRole.INFECTION_STAFF,
        CardStatus.APPROVED, None, "院感科备案通过，审核完成"),
    (CardStatus.PHC_APPROVED, ReviewAction.APPROVE, UserRole.SYSTEM,
        CardStatus.APPROVED, None, "系统自动备案通过"),
    (CardStatus.PHC_APPROVED, ReviewAction.REJECT, UserRole.INFECTION_STAFF,
        CardStatus.PHC_REJECTED, UserRole.DOCTOR, "院感科驳回"),
    
    # 上报疾控
    (CardStatus.APPROVED, ReviewAction.REPORT, UserRole.PHC_STAFF,
        CardStatus.REPORTING, None, "开始上报疾控"),
    (CardStatus.REPORTING, ReviewAction.REPORT_SUCCESS, UserRole.SYSTEM,
        CardStatus.REPORT_SUCCESS, None, "疾控接收成功"),
    (CardStatus.REPORTING, ReviewAction.REPORT_FAILED, UserRole.SYSTEM,
        CardStatus.REPORT_FAILED, UserRole.PHC_STAFF, "疾控接收失败"),
    (CardStatus.REPORT_FAILED, ReviewAction.REPORT, UserRole.PHC_STAFF,
        CardStatus.REPORTING, None, "重新上报"),
    
    # 上报失败后公卫科可以修改再重新上报
    (CardStatus.REPORT_FAILED, ReviewAction.FILL_COMPLETE, UserRole.PHC_STAFF,
        CardStatus.APPROVED, UserRole.PHC_STAFF, "修正后待重新上报"),
    
    # 作废（主要状态支持作废）
    (CardStatus.DRAFT, ReviewAction.CANCEL, UserRole.DOCTOR,
        CardStatus.CANCELLED, None, "作废草稿"),
    (CardStatus.FILLED, ReviewAction.CANCEL, UserRole.DOCTOR,
        CardStatus.CANCELLED, None, "作废"),
    (CardStatus.PENDING_REVIEW, ReviewAction.CANCEL, UserRole.DEPT_DIRECTOR,
        CardStatus.CANCELLED, None, "作废"),
    (CardStatus.DEPT_APPROVED, ReviewAction.CANCEL, UserRole.PHC_STAFF,
        CardStatus.CANCELLED, None, "作废"),
    (CardStatus.PHC_APPROVED, ReviewAction.CANCEL, UserRole.INFECTION_STAFF,
        CardStatus.CANCELLED, None, "作废"),
    (CardStatus.APPROVED, ReviewAction.CANCEL, UserRole.PHC_STAFF,
        CardStatus.CANCELLED, None, "作废"),
    
    # 归档
    (CardStatus.REPORT_SUCCESS, ReviewAction.ARCHIVE, UserRole.SYSTEM,
        CardStatus.ARCHIVED, None, "自动归档"),
    
    # ========== 订正报告流程 ==========
    # 上报成功后公卫科可以发起订正
    (CardStatus.REPORT_SUCCESS, ReviewAction.INITIATE_CORRECTION, UserRole.PHC_STAFF,
        CardStatus.CORRECTION_DRAFT, UserRole.PHC_STAFF, "公卫科发起订正报告"),
    
    # 订正草稿阶段：公卫科可编辑、保存
    (CardStatus.CORRECTION_DRAFT, ReviewAction.SAVE_DRAFT, UserRole.PHC_STAFF,
        CardStatus.CORRECTION_DRAFT, UserRole.PHC_STAFF, "保存订正草稿"),
    (CardStatus.CORRECTION_DRAFT, ReviewAction.FILL_COMPLETE, UserRole.PHC_STAFF,
        CardStatus.CORRECTION_DRAFT, UserRole.PHC_STAFF, "更新订正内容"),
    
    # 提交订正审核（简化流程：公卫科内部审核后直接上报，或走简化审核）
    (CardStatus.CORRECTION_DRAFT, ReviewAction.SUBMIT_CORRECTION, UserRole.PHC_STAFF,
        CardStatus.CORRECTION_SUBMITTED, UserRole.PHC_STAFF, "提交订正审核"),
    
    # 订正审核通过
    (CardStatus.CORRECTION_SUBMITTED, ReviewAction.APPROVE, UserRole.PHC_STAFF,
        CardStatus.CORRECTION_APPROVED, UserRole.PHC_STAFF, "订正审核通过"),
    (CardStatus.CORRECTION_SUBMITTED, ReviewAction.REJECT, UserRole.PHC_STAFF,
        CardStatus.CORRECTION_DRAFT, UserRole.PHC_STAFF, "订正驳回修改"),
    
    # 订正上报
    (CardStatus.CORRECTION_APPROVED, ReviewAction.CORRECTION_REPORT, UserRole.PHC_STAFF,
        CardStatus.CORRECTION_REPORTING, None, "开始上报订正报告"),
    (CardStatus.CORRECTION_REPORTING, ReviewAction.REPORT_SUCCESS, UserRole.SYSTEM,
        CardStatus.REPORT_SUCCESS, None, "订正报告上报成功"),
    (CardStatus.CORRECTION_REPORTING, ReviewAction.REPORT_FAILED, UserRole.SYSTEM,
        CardStatus.CORRECTION_APPROVED, UserRole.PHC_STAFF, "订正上报失败，可重试"),
]


def _find_transition(from_status: CardStatus, action: ReviewAction, 
                     role: UserRole) -> Optional[Tuple[CardStatus, Optional[UserRole], str]]:
    """查找状态转换规则。
    
    优先精确匹配(状态,动作,角色)三元组，其次匹配角色为None（任意角色）的规则。
    """
    # 先精确匹配角色
    for rule in STATE_TRANSITIONS:
        s, a, r, ts, nr, desc = rule
        if s == from_status and a == action and r == role:
            return (ts, nr, desc)
    # 再匹配任意角色(None)
    for rule in STATE_TRANSITIONS:
        s, a, r, ts, nr, desc = rule
        if s == from_status and a == action and r is None:
            return (ts, nr, desc)
    return None


# ============================================================================
# 审核记录
# ============================================================================

def _utcnow_iso() -> str:
    """生成UTC时间戳（ISO 8601格式，带Z后缀）。医疗系统统一使用UTC存储，显示时转换本地时区。"""
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')


@dataclass
class ReviewRecord:
    """单条审核/操作记录"""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=_utcnow_iso)
    from_status: CardStatus = CardStatus.DRAFT
    to_status: CardStatus = CardStatus.DRAFT
    action: ReviewAction = ReviewAction.SAVE_DRAFT
    user_id: str = ""
    user_name: str = ""
    user_role: UserRole = UserRole.SYSTEM
    comments: str = ""                    # 审核意见/备注
    ip_address: str = ""
    attachments: List[str] = field(default_factory=list)  # 附件ID列表
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据


# ============================================================================
# 审核工作流引擎
# ============================================================================

class ReviewWorkflow:
    """报告卡多级审核工作流引擎
    
    用法：
        workflow = ReviewWorkflow(card_id='CARD001', current_user_id='D001',
                                  current_user_name='张医生',
                                  current_user_role=UserRole.DOCTOR)
        workflow.fill_complete(card_data)
        workflow.submit(comments='请主任审核')
        # ... 切换到主任账号
        workflow.set_current_user('DIR001', '王主任', UserRole.DEPT_DIRECTOR)
        workflow.approve(comments='同意上报')
    """

    def __init__(self, card_id: str, 
                 current_user_id: str = 'system',
                 current_user_name: str = '系统',
                 current_user_role: UserRole = UserRole.SYSTEM,
                 db_manager=None,
                 auditor=None,
                 auto_load: bool = True):
        """初始化工作流。
        
        参数：
            card_id: 报告卡ID
            current_user_id: 当前用户ID
            current_user_name: 当前用户姓名
            current_user_role: 当前用户角色
            db_manager: 数据库管理器（可选）
            auditor: 审计器（可选，用于记录审计日志）
            auto_load: 如果数据库中有记录，是否自动加载状态
        """
        self.card_id = card_id
        self.current_user_id = current_user_id
        self.current_user_name = current_user_name
        self.current_user_role = current_user_role
        self.db_manager = db_manager
        # 如果未传入auditor，使用默认审计日志器（医疗系统合规要求不能无审计）
        self.auditor = auditor if auditor is not None else get_default_auditor()
        
        # 状态
        self.status: CardStatus = CardStatus.DRAFT
        self.current_reviewer: Optional[UserRole] = None
        self.card_data: Dict[str, Any] = {}
        self.history: List[ReviewRecord] = []
        self.corrections: List[Dict[str, Any]] = []
        
        # 回调钩子
        self._on_status_change: Optional[Callable[[CardStatus, CardStatus, ReviewRecord], None]] = None
        self._on_submit: Optional[Callable[[ReviewRecord], None]] = None
        self._on_approve: Optional[Callable[[ReviewRecord], None]] = None
        self._on_reject: Optional[Callable[[ReviewRecord], None]] = None
        self._on_report: Optional[Callable[[ReviewRecord], None]] = None
        
        # 确保数据库表存在
        self._ensure_tables()
        
        # 自动从数据库加载
        if auto_load and db_manager:
            self._load_from_db()
    
    @property
    def current_assignee(self) -> Optional[UserRole]:
        """当前待处理人角色（别名，与current_reviewer保持一致）。"""
        return self.current_reviewer
    
    def get_current_assignee(self) -> Optional[UserRole]:
        """获取当前待处理人角色。"""
        return self.current_reviewer
    
    @classmethod
    def load(cls, card_id: str, db_manager,
             current_user_id: str = 'system',
             current_user_name: str = '系统',
             current_user_role: UserRole = UserRole.SYSTEM,
             auditor=None) -> 'ReviewWorkflow':
        """从数据库加载已存在的工作流。
        
        参数：
            card_id: 报告卡ID
            db_manager: 数据库管理器
            current_user_id: 当前用户ID
            current_user_name: 当前用户姓名
            current_user_role: 当前用户角色
            auditor: 审计器
            
        返回：
            ReviewWorkflow: 加载后的工作流实例
            
        异常：
            WorkflowError: 如果卡片不存在
        """
        wf = cls(card_id, current_user_id, current_user_name, current_user_role,
                 db_manager, auditor, auto_load=False)
        if not wf._load_from_db():
            raise WorkflowError(f"报告卡 {card_id} 不存在或无历史记录")
        return wf

    def set_current_user(self, user_id: str, user_name: str, user_role: UserRole):
        """切换当前操作用户。"""
        self.current_user_id = user_id
        self.current_user_name = user_name
        self.current_user_role = user_role

    def set_callback(self, event: str, callback: Callable):
        """设置事件回调。
        
        参数：
            event: 事件名 (status_change/submit/approve/reject/report)
            callback: 回调函数
        """
        if event == 'status_change':
            self._on_status_change = callback
        elif event == 'submit':
            self._on_submit = callback
        elif event == 'approve':
            self._on_approve = callback
        elif event == 'reject':
            self._on_reject = callback
        elif event == 'report':
            self._on_report = callback

    # ------------------------------------------------------------------
    # 工作流操作
    # ------------------------------------------------------------------

    def save_draft(self, card_data: Dict[str, Any] = None, 
                   comments: str = "", ip_address: str = "") -> ReviewRecord:
        """保存草稿。"""
        if card_data:
            self.card_data.update(card_data)
        return self._transition(ReviewAction.SAVE_DRAFT, comments, ip_address)

    def fill_complete(self, card_data: Dict[str, Any] = None,
                      comments: str = "", ip_address: str = "") -> ReviewRecord:
        """医生完成填写。"""
        if card_data:
            self.card_data.update(card_data)
        return self._transition(ReviewAction.FILL_COMPLETE, comments, ip_address)

    def submit(self, comments: str = "提交审核", ip_address: str = "",
               attachments: List[str] = None) -> ReviewRecord:
        """提交审核（医生操作）。"""
        record = self._transition(ReviewAction.SUBMIT, comments, ip_address, attachments)
        if self._on_submit:
            self._on_submit(record)
        return record

    def approve(self, comments: str = "审核通过", ip_address: str = "",
                attachments: List[str] = None,
                metadata: Dict[str, Any] = None) -> ReviewRecord:
        """审核通过（各级审核人员操作）。"""
        record = self._transition(ReviewAction.APPROVE, comments, ip_address, 
                                  attachments, metadata)
        if self._on_approve:
            self._on_approve(record)
        return record

    def reject(self, comments: str = "审核驳回", ip_address: str = "",
               attachments: List[str] = None,
               metadata: Dict[str, Any] = None) -> ReviewRecord:
        """驳回（各级审核人员操作）。"""
        record = self._transition(ReviewAction.REJECT, comments, ip_address,
                                  attachments, metadata)
        if self._on_reject:
            self._on_reject(record)
        return record

    def return_for_edit(self, comments: str = "退回修改", ip_address: str = "",
                        attachments: List[str] = None) -> ReviewRecord:
        """退回修改（回到医生填写状态）。"""
        return self._transition(ReviewAction.RETURN, comments, ip_address, attachments)

    def report_to_cdc(self, report_data: Dict[str, Any] = None,
                      comments: str = "上报疾控", ip_address: str = "") -> ReviewRecord:
        """上报疾控（公共卫生科操作）。"""
        if report_data:
            self.card_data['report_data'] = report_data
        record = self._transition(ReviewAction.REPORT, comments, ip_address)
        if self._on_report:
            self._on_report(record)
        return record

    def handle_report_callback(self, success: bool, cdc_response: Dict[str, Any] = None,
                               comments: str = "") -> ReviewRecord:
        """处理疾控上报回调（系统操作）。
        
        注意：必须先显式调用report_to_cdc()进入REPORTING状态后才能处理回调。
        """
        # 严格校验状态：必须在REPORTING状态
        if self.status != CardStatus.REPORTING:
            raise WorkflowError(
                f"当前状态 {self.get_status_display()} 不允许上报回调，"
                f"请先通过report_to_cdc()进入上报中状态"
            )
        
        # 临时设置为系统用户
        orig_user = (self.current_user_id, self.current_user_name, self.current_user_role)
        self.set_current_user('system', '系统', UserRole.SYSTEM)
        
        metadata = {'success': success}
        if cdc_response:
            metadata['cdc_response'] = cdc_response
        
        # 根据success选择正确的action
        action = ReviewAction.REPORT_SUCCESS if success else ReviewAction.REPORT_FAILED
        
        try:
            record = self._transition(
                action,
                comments or ("疾控接收成功" if success else "疾控接收失败"),
                "", [], metadata
            )
        finally:
            # 恢复原用户
            self.set_current_user(*orig_user)
        
        return record

    def cancel(self, comments: str = "作废", ip_address: str = "") -> ReviewRecord:
        """作废报告卡。"""
        return self._transition(ReviewAction.CANCEL, comments, ip_address)

    def archive(self, comments: str = "归档") -> ReviewRecord:
        """归档报告卡（系统自动）。"""
        orig_user = (self.current_user_id, self.current_user_name, self.current_user_role)
        self.set_current_user('system', '系统', UserRole.SYSTEM)
        record = self._transition(ReviewAction.ARCHIVE, comments, "")
        self.set_current_user(*orig_user)
        return record

    # ------------------------------------------------------------------
    # 订正报告操作
    # ------------------------------------------------------------------

    def initiate_correction(self, original_card_id: str = None,
                            correction_items: List[str] = None,
                            correction_reason: str = "",
                            card_data_updates: Dict[str, Any] = None,
                            comments: str = "发起订正",
                            ip_address: str = "") -> ReviewRecord:
        """发起订正报告
        
        参数：
            original_card_id: 原始报告卡编号（如CDC返回的编号）
            correction_items: 订正项列表，如 ["姓名", "身份证号", "诊断"]
            correction_reason: 订正原因
            card_data_updates: 需要更新的卡片数据字段
            comments: 备注
            ip_address: 操作IP
        """
        metadata = {
            'correction': True,
            'original_card_id': original_card_id or self.card_id,
            'correction_items': correction_items or [],
            'correction_reason': correction_reason,
        }
        # 保存订正信息
        self.corrections.append({
            'original_card_id': original_card_id or self.card_id,
            'correction_items': correction_items or [],
            'correction_reason': correction_reason,
            'initiated_at': _utcnow_iso(),
            'initiated_by': self.current_user_name,
        })
        self.card_data['is_correction'] = True
        self.card_data['original_card_id'] = original_card_id or self.card_id
        self.card_data['correction_items'] = correction_items or []
        self.card_data['correction_reason'] = correction_reason
        if card_data_updates:
            self.card_data.update(card_data_updates)
        
        record = self._transition(ReviewAction.INITIATE_CORRECTION, comments, ip_address,
                                  metadata=metadata)
        return record

    def submit_correction(self, comments: str = "提交订正审核",
                          ip_address: str = "") -> ReviewRecord:
        """提交订正报告审核"""
        return self._transition(ReviewAction.SUBMIT_CORRECTION, comments, ip_address)

    def report_correction(self, report_data: Dict[str, Any] = None,
                          comments: str = "上报订正报告",
                          ip_address: str = "") -> ReviewRecord:
        """上报订正报告到疾控"""
        if report_data:
            self.card_data['correction_report_data'] = report_data
        # 标记CDC XML应包含订正标识
        self.card_data['report_type'] = 'correction'
        return self._transition(ReviewAction.CORRECTION_REPORT, comments, ip_address)

    def is_correction(self) -> bool:
        """当前是否为订正报告流程"""
        return self.status in (
            CardStatus.CORRECTION_DRAFT,
            CardStatus.CORRECTION_SUBMITTED,
            CardStatus.CORRECTION_APPROVED,
            CardStatus.CORRECTION_REPORTING,
        ) or self.card_data.get('is_correction', False)

    def get_correction_info(self) -> Dict[str, Any]:
        """获取订正信息"""
        return {
            'is_correction': self.is_correction(),
            'original_card_id': self.card_data.get('original_card_id'),
            'correction_items': self.card_data.get('correction_items', []),
            'correction_reason': self.card_data.get('correction_reason', ''),
        }

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_status_display(self) -> str:
        """获取状态中文显示名。"""
        status_names = {
            CardStatus.DRAFT: "草稿",
            CardStatus.FILLED: "已填写",
            CardStatus.PENDING_REVIEW: "待科室主任审核",
            CardStatus.DEPT_REVIEWING: "科室主任审核中",
            CardStatus.DEPT_APPROVED: "科室主任已通过",
            CardStatus.DEPT_REJECTED: "科室主任驳回",
            CardStatus.PHC_REVIEWING: "公共卫生科复核中",
            CardStatus.PHC_APPROVED: "公共卫生科已通过",
            CardStatus.PHC_REJECTED: "公共卫生科驳回",
            CardStatus.INFECTION_REVIEWING: "院感科备案中",
            CardStatus.APPROVED: "审核通过（待上报）",
            CardStatus.REPORTING: "上报中",
            CardStatus.REPORTED: "已上报",
            CardStatus.REPORT_SUCCESS: "上报成功",
            CardStatus.REPORT_FAILED: "上报失败",
            CardStatus.CANCELLED: "已作废",
            CardStatus.ARCHIVED: "已归档",
            CardStatus.CORRECTION_DRAFT: "订正草稿",
            CardStatus.CORRECTION_SUBMITTED: "订正审核中",
            CardStatus.CORRECTION_APPROVED: "订正待上报",
            CardStatus.CORRECTION_REPORTING: "订正上报中",
        }
        return status_names.get(self.status, self.status.value)

    def get_next_reviewer_role_display(self) -> str:
        """获取下一个审核人角色中文名。"""
        if not self.current_reviewer:
            return ""
        role_names = {
            UserRole.DOCTOR: "临床医生",
            UserRole.DEPT_DIRECTOR: "科室主任",
            UserRole.PHC_STAFF: "公共卫生科",
            UserRole.INFECTION_STAFF: "院感科",
            UserRole.ADMIN: "系统管理员",
            UserRole.SYSTEM: "系统",
        }
        return role_names.get(self.current_reviewer, "")

    def can_perform_action(self, action: ReviewAction) -> bool:
        """检查当前用户是否可以执行指定动作。"""
        return _find_transition(self.status, action, self.current_user_role) is not None

    def get_available_actions(self) -> List[Dict[str, Any]]:
        """获取当前用户可执行的操作列表。"""
        actions = []
        action_meta = {
            ReviewAction.SAVE_DRAFT: ("保存草稿", "保存当前内容为草稿"),
            ReviewAction.FILL_COMPLETE: ("完成填写", "填写完成，待提交"),
            ReviewAction.SUBMIT: ("提交审核", "提交给上级审核"),
            ReviewAction.APPROVE: ("审核通过", "同意通过，进入下一环节"),
            ReviewAction.REJECT: ("驳回", "驳回申请，需重新填写"),
            ReviewAction.RETURN: ("退回修改", "退回给医生修改"),
            ReviewAction.REPORT: ("上报疾控", "将报告卡上报疾控中心"),
            ReviewAction.CANCEL: ("作废", "作废该报告卡"),
        }
        
        for action in ReviewAction:
            if self.can_perform_action(action):
                name, desc = action_meta.get(action, (action.value, ""))
                actions.append({
                    'action': action.value,
                    'name': name,
                    'description': desc,
                })
        return actions

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取审核历史记录。"""
        records = self.history[-limit:]
        return [self._record_to_dict(r) for r in records]

    def get_review_summary(self) -> Dict[str, Any]:
        """获取审核摘要。"""
        approved_by = []
        rejected_by = []
        submit_time = None
        approve_time = None
        
        for r in self.history:
            if r.action == ReviewAction.SUBMIT:
                submit_time = r.timestamp
            elif r.action == ReviewAction.APPROVE and r.user_role in [
                UserRole.DEPT_DIRECTOR, UserRole.PHC_STAFF, UserRole.INFECTION_STAFF
            ]:
                approved_by.append({
                    'role': r.user_role.value,
                    'user': r.user_name,
                    'time': r.timestamp,
                    'comments': r.comments,
                })
            elif r.action == ReviewAction.REJECT:
                rejected_by.append({
                    'role': r.user_role.value,
                    'user': r.user_name,
                    'time': r.timestamp,
                    'comments': r.comments,
                })
            if r.to_status == CardStatus.APPROVED:
                approve_time = r.timestamp
        
        return {
            'card_id': self.card_id,
            'current_status': self.status.value,
            'current_status_display': self.get_status_display(),
            'next_reviewer': self.current_reviewer.value if self.current_reviewer else None,
            'next_reviewer_display': self.get_next_reviewer_role_display(),
            'submit_time': submit_time,
            'approve_time': approve_time,
            'approved_by': approved_by,
            'rejected_count': len(rejected_by),
            'rejected_by': rejected_by,
            'total_steps': len(self.history),
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _transition(self, action: ReviewAction, comments: str, ip_address: str,
                    attachments: List[str] = None,
                    metadata: Dict[str, Any] = None) -> ReviewRecord:
        """执行状态转换。"""
        from_status = self.status
        
        # 查找转换规则
        transition = _find_transition(from_status, action, self.current_user_role)
        
        if not transition:
            # 检查是否存在该(状态,动作)的任何角色规则
            has_any_role = any(
                r[0] == from_status and r[1] == action for r in STATE_TRANSITIONS
            )
            if has_any_role:
                raise WorkflowError(
                    f"用户 {self.current_user_name}({self.current_user_role.value}) "
                    f"在状态 {self.get_status_display()} 下无权执行 "
                    f"{action.value} 操作"
                )
            raise WorkflowError(
                f"状态 {self.get_status_display()} 不支持操作 {action.value}"
            )
        
        to_status, next_reviewer, desc = transition
        
        # 创建审核记录
        record = self._create_record(from_status, to_status, action,
                                     comments or desc, ip_address,
                                     attachments or [], metadata or {})
        
        # 执行转换
        self._execute_transition(record, next_reviewer)
        
        return record

    def _create_record(self, from_status: CardStatus, to_status: CardStatus,
                       action: ReviewAction, comments: str, ip_address: str,
                       attachments: List[str], metadata: Dict[str, Any]) -> ReviewRecord:
        """创建审核记录。"""
        return ReviewRecord(
            from_status=from_status,
            to_status=to_status,
            action=action,
            user_id=self.current_user_id,
            user_name=self.current_user_name,
            user_role=self.current_user_role,
            comments=comments,
            ip_address=ip_address,
            attachments=attachments,
            metadata=metadata,
        )

    def _execute_transition(self, record: ReviewRecord, next_reviewer: Optional[UserRole]):
        """执行状态转换并持久化。"""
        old_status = self.status
        
        # 更新状态
        self.status = record.to_status
        self.current_reviewer = next_reviewer
        
        # 添加到历史
        self.history.append(record)
        
        # 记录审计日志
        if self.auditor:
            try:
                self.auditor.log_operation(
                    operation_type=self._action_to_audit_type(record.action),
                    patient_id=self.card_data.get('patient_id'),
                    patient_name=self.card_data.get('name'),
                    record_id=self.card_id,
                    details={
                        'from_status': record.from_status.value,
                        'to_status': record.to_status.value,
                        'action': record.action.value,
                    },
                    old_value={'status': record.from_status.value},
                    new_value={'status': record.to_status.value},
                    ip_address=record.ip_address,
                    reason=record.comments,
                )
            except Exception as e:
                LOGGER.debug("审计日志记录失败: %s", e)
        
        # 持久化到数据库
        self._save_record(record)
        
        # 触发回调
        if self._on_status_change:
            try:
                self._on_status_change(old_status, self.status, record)
            except Exception as e:
                LOGGER.warning("状态变更回调执行失败: %s", e)
        
        LOGGER.info(
            "报告卡 %s 状态变更: %s → %s (%s, 操作人=%s)",
            self.card_id, old_status.value, self.status.value,
            record.action.value, self.current_user_name
        )

    def _action_to_audit_type(self, action: ReviewAction) -> str:
        """转换动作到审计操作类型。"""
        mapping = {
            ReviewAction.SAVE_DRAFT: 'CREATE',
            ReviewAction.FILL_COMPLETE: 'UPDATE',
            ReviewAction.SUBMIT: 'SUBMIT',
            ReviewAction.APPROVE: 'APPROVE',
            ReviewAction.REJECT: 'REJECT',
            ReviewAction.RETURN: 'UPDATE',
            ReviewAction.REPORT: 'REPORT',
            ReviewAction.REPORT_SUCCESS: 'REPORT',
            ReviewAction.REPORT_FAILED: 'REPORT',
            ReviewAction.CANCEL: 'DELETE',
            ReviewAction.ARCHIVE: 'UPDATE',
            ReviewAction.INITIATE_CORRECTION: 'UPDATE',
            ReviewAction.SUBMIT_CORRECTION: 'SUBMIT',
            ReviewAction.CORRECTION_REPORT: 'REPORT',
        }
        return mapping.get(action, 'UPDATE')

    def _record_to_dict(self, record: ReviewRecord) -> Dict[str, Any]:
        """转换审核记录为字典。"""
        return {
            'record_id': record.record_id,
            'timestamp': record.timestamp,
            'from_status': record.from_status.value,
            'to_status': record.to_status.value,
            'action': record.action.value,
            'user_id': record.user_id,
            'user_name': record.user_name,
            'user_role': record.user_role.value,
            'comments': record.comments,
            'ip_address': record.ip_address,
            'attachments': record.attachments,
            'metadata': record.metadata,
        }

    def _ensure_tables(self):
        """确保数据库表存在。"""
        if not self.db_manager:
            return
        try:
            # SQLite UTC时间戳格式：ISO 8601 with Z suffix
            self.db_manager.execute("""
                CREATE TABLE IF NOT EXISTS review_workflow (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    card_id TEXT NOT NULL,
                    from_status TEXT NOT NULL,
                    to_status TEXT NOT NULL,
                    action TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    user_name TEXT NOT NULL,
                    user_role TEXT NOT NULL,
                    comments TEXT,
                    ip_address TEXT,
                    metadata TEXT,
                    attachments TEXT,
                    record_id TEXT,
                    timestamp TEXT,
                    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                )
            """)
            self.db_manager.execute("""
                CREATE INDEX IF NOT EXISTS idx_review_card 
                ON review_workflow(card_id)
            """)
            self.db_manager.execute("""
                CREATE TABLE IF NOT EXISTS card_status (
                    card_id TEXT PRIMARY KEY,
                    current_status TEXT NOT NULL,
                    current_reviewer TEXT,
                    card_data TEXT,
                    updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                )
            """)
        except Exception as e:
            LOGGER.warning("审核工作流表创建失败: %s", e)
    
    def _load_from_db(self) -> bool:
        """从数据库加载工作流状态和历史记录。
        
        返回：
            bool: 是否成功加载（False表示没有历史记录）
        """
        if not self.db_manager:
            return False
        
        try:
            # 加载当前状态
            status_row = self.db_manager.execute(
                "SELECT current_status, current_reviewer, card_data FROM card_status WHERE card_id = ?",
                [self.card_id]
            )
            
            if not status_row or len(status_row) == 0:
                return False
            
            # 解析状态行（兼容不同数据库返回格式）
            row = status_row[0]
            if isinstance(row, dict):
                current_status = row.get('current_status')
                current_reviewer = row.get('current_reviewer')
                card_data_str = row.get('card_data')
            else:
                current_status = row[0]
                current_reviewer = row[1]
                card_data_str = row[2]
            
            self.status = CardStatus(current_status) if current_status else CardStatus.DRAFT
            self.current_reviewer = UserRole(current_reviewer) if current_reviewer else None
            
            if card_data_str:
                try:
                    self.card_data = json.loads(card_data_str)
                except Exception as _e:
                    LOGGER.warning("解析卡片JSON数据失败，回退为空字典: %s", _e)
                    self.card_data = {}
            
            # 加载历史记录
            history_rows = self.db_manager.execute(
                """SELECT record_id, timestamp, from_status, to_status, action,
                          user_id, user_name, user_role, comments, ip_address,
                          metadata, attachments
                   FROM review_workflow 
                   WHERE card_id = ? 
                   ORDER BY id ASC""",
                [self.card_id]
            )
            
            self.history = []
            for hrow in history_rows:
                if isinstance(hrow, dict):
                    record = ReviewRecord(
                        record_id=hrow.get('record_id', str(uuid.uuid4())),
                        timestamp=hrow.get('timestamp', _utcnow_iso()),
                        from_status=CardStatus(hrow.get('from_status')),
                        to_status=CardStatus(hrow.get('to_status')),
                        action=ReviewAction(hrow.get('action')),
                        user_id=hrow.get('user_id', ''),
                        user_name=hrow.get('user_name', ''),
                        user_role=UserRole(hrow.get('user_role')),
                        comments=hrow.get('comments', ''),
                        ip_address=hrow.get('ip_address', ''),
                        attachments=json.loads(hrow.get('attachments')) if hrow.get('attachments') else [],
                        metadata=json.loads(hrow.get('metadata')) if hrow.get('metadata') else {},
                    )
                else:
                    record = ReviewRecord(
                        record_id=hrow[0] if hrow[0] else str(uuid.uuid4()),
                        timestamp=hrow[1] if hrow[1] else _utcnow_iso(),
                        from_status=CardStatus(hrow[2]),
                        to_status=CardStatus(hrow[3]),
                        action=ReviewAction(hrow[4]),
                        user_id=hrow[5] or '',
                        user_name=hrow[6] or '',
                        user_role=UserRole(hrow[7]),
                        comments=hrow[8] or '',
                        ip_address=hrow[9] or '',
                        metadata=json.loads(hrow[10]) if hrow[10] else {},
                        attachments=json.loads(hrow[11]) if hrow[11] else [],
                    )
                self.history.append(record)
            
            LOGGER.debug("从数据库加载工作流 %s: status=%s, history=%d条",
                        self.card_id, self.status.value, len(self.history))
            return True
            
        except Exception as e:
            LOGGER.warning("从数据库加载工作流失败: %s", e)
            return False

    def _save_record(self, record: ReviewRecord):
        """保存审核记录到数据库。"""
        if not self.db_manager:
            return
        try:
            # 保存审核记录
            self.db_manager.execute("""
                INSERT INTO review_workflow
                (card_id, from_status, to_status, action, user_id, user_name,
                 user_role, comments, ip_address, metadata, attachments, record_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                self.card_id,
                record.from_status.value,
                record.to_status.value,
                record.action.value,
                record.user_id,
                record.user_name,
                record.user_role.value,
                record.comments,
                record.ip_address,
                json.dumps(record.metadata, ensure_ascii=False, default=str) if record.metadata else None,
                json.dumps(record.attachments, ensure_ascii=False) if record.attachments else None,
                record.record_id,
                record.timestamp,
            ])
            
            # 更新当前状态（使用UTC时间戳）
            self.db_manager.execute("""
                INSERT OR REPLACE INTO card_status
                (card_id, current_status, current_reviewer, card_data, updated_at)
                VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            """, [
                self.card_id,
                self.status.value,
                self.current_reviewer.value if self.current_reviewer else None,
                json.dumps(self.card_data, ensure_ascii=False, default=str) if self.card_data else None,
            ])
        except Exception as e:
            LOGGER.warning("保存审核记录失败: %s", e)


class WorkflowError(Exception):
    """工作流异常"""
    pass
