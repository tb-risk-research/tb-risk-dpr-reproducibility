#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审核工作流模块单元测试"""

import json
import os
import sqlite3
import tempfile
import time
import pytest

from health_interop.workflow import (
    ReviewWorkflow, CardStatus, ReviewAction, UserRole, ReviewRecord
)


class SimpleDBManager:
    """测试用简单SQLite数据库管理器"""
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
    
    def execute(self, sql: str, params=None):
        cursor = self.conn.cursor()
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)
        self.conn.commit()
        if sql.strip().upper().startswith(("SELECT", "PRAGMA")):
            return [dict(row) for row in cursor.fetchall()]
        return None
    
    def close(self):
        self.conn.close()


class TestStateTransitions:
    """测试状态机转换逻辑（内存模式）"""

    def _make_workflow(self, card_id="test-card-001", user_role=UserRole.DOCTOR, user_id="D001"):
        """创建内存模式工作流实例（db_manager=None）"""
        return ReviewWorkflow(
            card_id=card_id,
            current_user_id=user_id,
            current_user_name="测试用户",
            current_user_role=user_role,
            db_manager=None,
        )

    def _fill_and_submit(self, wf, card_data=None):
        """辅助方法：医生完成填写并提交审核"""
        wf.fill_complete(card_data or {"name": "测试患者"})
        wf.submit(comments="请审核")

    def test_initial_state_draft(self):
        """测试初始状态为DRAFT"""
        wf = self._make_workflow()
        assert wf.status == CardStatus.DRAFT

    def test_doctor_fill_and_submit(self):
        """测试医生填写完成并提交（DRAFT -> FILLED -> PENDING_REVIEW）"""
        wf = self._make_workflow()
        wf.fill_complete({"name": "张三"})
        assert wf.status == CardStatus.FILLED
        
        wf.submit(comments="请主任审核")
        assert wf.status == CardStatus.PENDING_REVIEW
        assert wf.current_assignee == UserRole.DEPT_DIRECTOR

    def test_dept_director_approve(self):
        """测试科室主任审核通过"""
        wf = self._make_workflow(user_id="D001")
        self._fill_and_submit(wf)
        
        wf.set_current_user("DH001", "王主任", UserRole.DEPT_DIRECTOR)
        wf.approve(comments="科室审核通过")
        
        assert wf.status == CardStatus.DEPT_APPROVED
        assert wf.current_assignee == UserRole.PHC_STAFF

    def test_full_review_to_report_success(self):
        """测试完整三级审核流程到上报成功"""
        wf = self._make_workflow(user_id="D001")
        self._fill_and_submit(wf)
        
        # 科主任通过
        wf.set_current_user("DH001", "王主任", UserRole.DEPT_DIRECTOR)
        wf.approve()
        assert wf.status == CardStatus.DEPT_APPROVED
        
        # 公卫科通过
        wf.set_current_user("PH001", "李公卫", UserRole.PHC_STAFF)
        wf.approve()
        assert wf.status == CardStatus.PHC_APPROVED
        
        # 院感科备案
        wf.set_current_user("INF001", "赵院感", UserRole.INFECTION_STAFF)
        wf.approve()
        assert wf.status == CardStatus.APPROVED
        
        # 上报疾控
        wf.set_current_user("PH001", "李公卫", UserRole.PHC_STAFF)
        wf.report_to_cdc(comments="上报至疾控")
        assert wf.status == CardStatus.REPORTING
        
        # 上报成功
        wf.handle_report_callback(success=True, cdc_response={"msg": "接收成功"})
        assert wf.status == CardStatus.REPORT_SUCCESS

    def test_dept_reject_and_resubmit(self):
        """测试科室主任驳回后医生修改重提"""
        wf = self._make_workflow(user_id="D001")
        self._fill_and_submit(wf)
        
        wf.set_current_user("DH001", "王主任", UserRole.DEPT_DIRECTOR)
        wf.reject(comments="诊断依据不足")
        assert wf.status == CardStatus.DEPT_REJECTED
        
        # 医生修改后重新提交
        wf.set_current_user("D001", "张医生", UserRole.DOCTOR)
        wf.fill_complete({"diagnosis": "涂阳肺结核", "sputum_smear": True})
        wf.submit(comments="已补充痰检结果")
        assert wf.status == CardStatus.PENDING_REVIEW

    def test_phc_reject(self):
        """测试公卫科驳回"""
        wf = self._make_workflow(user_id="D001")
        self._fill_and_submit(wf)
        
        wf.set_current_user("DH001", "王主任", UserRole.DEPT_DIRECTOR)
        wf.approve()
        
        wf.set_current_user("PH001", "李公卫", UserRole.PHC_STAFF)
        wf.reject(comments="身份证号格式有误")
        assert wf.status == CardStatus.PHC_REJECTED
        assert wf.current_assignee == UserRole.DOCTOR

    def test_wrong_role_cannot_approve(self):
        """测试错误角色不能越级审核"""
        wf = self._make_workflow()
        wf.fill_complete({"name": "测试"})
        wf.submit()
        
        # 公卫科不能跳过科主任
        wf.set_current_user("PH001", "李公卫", UserRole.PHC_STAFF)
        with pytest.raises(Exception):
            wf.approve()

    def test_callback_requires_reporting_state(self):
        """测试回调必须从REPORTING状态处理"""
        wf = self._make_workflow(user_id="D001")
        self._fill_and_submit(wf)
        wf.set_current_user("DH001", "王主任", UserRole.DEPT_DIRECTOR)
        wf.approve()
        wf.set_current_user("PH001", "李公卫", UserRole.PHC_STAFF)
        wf.approve()
        wf.set_current_user("INF001", "赵院感", UserRole.INFECTION_STAFF)
        wf.approve()
        assert wf.status == CardStatus.APPROVED
        
        # APPROVED状态直接处理回调应失败
        with pytest.raises(Exception):
            wf.handle_report_callback(success=True)

    def test_report_failed_then_retry(self):
        """测试上报失败后重试"""
        wf = self._make_workflow(user_id="D001")
        self._fill_and_submit(wf)
        wf.set_current_user("DH001", "王主任", UserRole.DEPT_DIRECTOR)
        wf.approve()
        wf.set_current_user("PH001", "李公卫", UserRole.PHC_STAFF)
        wf.approve()
        wf.set_current_user("INF001", "赵院感", UserRole.INFECTION_STAFF)
        wf.approve()
        wf.set_current_user("PH001", "李公卫", UserRole.PHC_STAFF)
        wf.report_to_cdc()
        assert wf.status == CardStatus.REPORTING
        
        wf.handle_report_callback(success=False, cdc_response={"error": "网络超时"})
        assert wf.status == CardStatus.REPORT_FAILED
        
        # 重新上报
        wf.report_to_cdc(comments="重试")
        assert wf.status == CardStatus.REPORTING

    def test_audit_history_recorded(self):
        """测试审核历史完整记录"""
        wf = self._make_workflow(user_id="D001")
        wf.fill_complete({"name": "张三"})
        wf.submit(comments="提交")
        
        wf.set_current_user("DH001", "王主任", UserRole.DEPT_DIRECTOR)
        wf.approve(comments="通过")
        
        history = wf.get_history()
        assert len(history) >= 3  # fill_complete + submit + approve
        assert isinstance(history[0], dict)
        actions = {h['action'] for h in history}
        assert ReviewAction.FILL_COMPLETE.value in actions
        assert ReviewAction.SUBMIT.value in actions
        assert ReviewAction.APPROVE.value in actions


class TestWorkflowDatabasePersistence:
    """测试工作流数据库持久化和恢复"""

    def setup_method(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db_path = self._tmp.name
        self.db = SimpleDBManager(self.db_path)

    def teardown_method(self):
        self.db.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_save_and_load_workflow(self):
        """测试工作流保存后可从数据库恢复"""
        wf1 = ReviewWorkflow(
            card_id="persist-001",
            current_user_id="D001",
            current_user_name="张医生",
            current_user_role=UserRole.DOCTOR,
            db_manager=self.db,
        )
        wf1.fill_complete({"name": "测试患者", "diagnosis": "肺结核"})
        wf1.submit(comments="提交审核")
        
        wf1.set_current_user("DH001", "王主任", UserRole.DEPT_DIRECTOR)
        wf1.approve(comments="科室通过")
        
        # 重新加载
        wf2 = ReviewWorkflow.load(
            card_id="persist-001",
            db_manager=self.db,
            current_user_id="PH001",
            current_user_name="李公卫",
            current_user_role=UserRole.PHC_STAFF,
        )
        assert wf2.status == CardStatus.DEPT_APPROVED
        assert wf2.current_assignee == UserRole.PHC_STAFF
        assert wf2.card_data.get("name") == "测试患者"
        
        history = wf2.get_history()
        assert len(history) >= 3

    def test_load_nonexistent_raises(self):
        """测试加载不存在的card抛出异常"""
        with pytest.raises(Exception):
            ReviewWorkflow.load(card_id="nonexistent", db_manager=self.db)

    def test_resume_and_continue(self):
        """测试恢复后可继续后续操作"""
        wf1 = ReviewWorkflow(
            card_id="resume-001",
            current_user_id="D001",
            current_user_name="张医生",
            current_user_role=UserRole.DOCTOR,
            db_manager=self.db,
        )
        wf1.fill_complete({"name": "李四"})
        wf1.submit()
        
        wf2 = ReviewWorkflow.load(
            card_id="resume-001",
            db_manager=self.db,
            current_user_id="DH001",
            current_user_name="王主任",
            current_user_role=UserRole.DEPT_DIRECTOR,
        )
        assert wf2.status == CardStatus.PENDING_REVIEW
        
        wf2.approve(comments="同意")
        assert wf2.status == CardStatus.DEPT_APPROVED
