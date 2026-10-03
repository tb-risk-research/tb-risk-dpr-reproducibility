#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GNN接触网络与疾控数据双向联动模块

实现数据闭环流程：
1. 确诊病例上报 → 自动从GNN网络提取密切接触者 → 生成筛查任务
2. 疾控回传密切接触者筛查结果 → 自动更新GNN传播网络
3. 网络更新 → 触发相关人员风险重评估

闭环流程：
    确诊病例 → 密接追踪 → 筛查任务 → 筛查结果 → 更新网络 → 风险重评估
"""

from __future__ import annotations

import datetime
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple

from .cdc import CDCCallbackParser
from .audit import AuditLogger, get_default_auditor

LOGGER = logging.getLogger("tb_risk.health_interop.contact_tracing")


# ============================================================================
# 接口协议（GNN网络解耦）
# ============================================================================

class GNNNetworkProtocol(Protocol):
    """GNN接触网络适配器协议。
    
    ContactTracingManager通过此协议与GNN网络交互，
    避免硬编码依赖具体GNN实现，GNN不可用时可优雅降级。
    """
    
    def get_contacts(self, patient_id: str) -> List[Dict[str, Any]]:
        """获取与指定患者有接触的人员列表。
        
        返回：
            List[Dict]: 接触者列表，每个字典包含patient_id、relationship、contact_type等字段
        """
        ...
    
    def add_contact(self, source_id: str, target_id: str, metadata: Dict[str, Any] = None) -> bool:
        """添加接触关系。"""
        ...
    
    def update_contact_risk(self, patient_id: str, risk_level: str) -> bool:
        """更新患者接触风险等级。"""
        ...
    
    def update_node(self, patient_id: str, data: Dict[str, Any]) -> bool:
        """更新节点数据（如标记为确诊、治愈等）。"""
        ...


class NullGNNNetwork:
    """空GNN网络实现（优雅降级）。
    
    当GNN模块不可用时使用，所有方法返回空列表/True，不影响核心报卡流程。
    """
    
    def get_contacts(self, patient_id: str) -> List[Dict[str, Any]]:
        return []
    
    def add_contact(self, source_id: str, target_id: str, metadata: Dict[str, Any] = None) -> bool:
        return True
    
    def update_contact_risk(self, patient_id: str, risk_level: str) -> bool:
        return True
    
    def update_node(self, patient_id: str, data: Dict[str, Any]) -> bool:
        return True


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ContactRecord:
    """接触记录"""
    contact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_patient_id: str = ""        # 传染源患者ID（确诊病例）
    contact_patient_id: str = ""       # 接触者患者ID
    contact_name: str = ""             # 接触者姓名（未建档时使用）
    relationship: str = ""             # 与病例关系（家属/同事/同学等）
    contact_type: str = ""             # 接触类型（household/social/medical等）
    contact_date: str = ""             # 最后接触日期
    contact_duration: str = ""         # 接触时长（如>8小时/频繁接触等）
    contact_intensity: str = ""        # 接触强度（高/中/低）—— 根据接触时长、距离、频率评估
    screening_priority: str = ""       # 筛查优先级（high/medium/low）—— 综合接触强度+传染性+易感程度
    is_active_tb: bool = False         # 是否为活动性结核
    screening_result: str = ""         # 筛查结果
    screening_date: str = ""           # 筛查日期
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    tracing_status: str = "pending"    # pending/screened/positive/negative/lost

    def to_dict(self) -> Dict[str, Any]:
        return {
            'contact_id': self.contact_id,
            'source_patient_id': self.source_patient_id,
            'contact_patient_id': self.contact_patient_id,
            'contact_name': self.contact_name,
            'relationship': self.relationship,
            'contact_type': self.contact_type,
            'contact_date': self.contact_date,
            'contact_duration': self.contact_duration,
            'contact_intensity': self.contact_intensity,
            'screening_priority': self.screening_priority,
            'is_active_tb': self.is_active_tb,
            'screening_result': self.screening_result,
            'screening_date': self.screening_date,
            'notes': self.notes,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'tracing_status': self.tracing_status,
        }


@dataclass
class ScreeningTask:
    """筛查任务"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_case_id: str = ""           # 来源确诊病例ID
    contact_id: str = ""               # 关联接触记录ID
    target_patient_id: str = ""        # 筛查对象患者ID（已建档）
    target_name: str = ""              # 筛查对象姓名（未建档）
    task_type: str = "contact_screening"  # contact_screening/active_followup
    priority: str = "medium"           # high/medium/low
    status: str = "pending"            # pending/assigned/in_progress/completed/cancelled
    assigned_to: str = ""              # 分配给谁
    due_date: str = ""                 # 截止日期
    screening_items: List[str] = field(default_factory=list)  # 需要筛查的项目
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    created_by: str = "system"
    completed_at: str = ""
    result_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'source_case_id': self.source_case_id,
            'contact_id': self.contact_id,
            'target_patient_id': self.target_patient_id,
            'target_name': self.target_name,
            'task_type': self.task_type,
            'priority': self.priority,
            'status': self.status,
            'assigned_to': self.assigned_to,
            'due_date': self.due_date,
            'screening_items': self.screening_items,
            'notes': self.notes,
            'created_at': self.created_at,
            'created_by': self.created_by,
            'completed_at': self.completed_at,
            'result_summary': self.result_summary,
        }


# ============================================================================
# 接触追踪管理器
# ============================================================================

class ContactTracingManager:
    """GNN接触网络与疾控数据双向联动管理器
    
    实现确诊病例→密接追踪→筛查→更新网络→重评估的完整闭环。
    
    用法：
        manager = ContactTracingManager(gnn_network=gnn, db_manager=db)
        
        # 注册回调
        manager.on_screening_task_created(send_notification)
        manager.on_network_updated(trigger_risk_reassessment)
        
        # 确诊病例触发密接追踪
        contacts = manager.trace_contacts_from_confirmed_case('P001')
        
        # 处理疾控回传的筛查结果
        manager.handle_screening_results('P001', contacts_data_from_cdc)
    """

    # 筛查项目默认配置
    DEFAULT_SCREENING_ITEMS = [
        '症状筛查（咳嗽/咳痰/咯血/发热/盗汗等）',
        '结核菌素皮肤试验（PPD）',
        'γ-干扰素释放试验（IGRA/T-SPOT）',
        '胸部X线/CT检查',
        '痰涂片抗酸杆菌检查',
        '痰结核分枝杆菌培养',
        '分子生物学检测（Xpert MTB/RIF）',
    ]

    # 接触类型 → 接触强度映射（根据接触时长、距离、频率）
    CONTACT_INTENSITY = {
        'household': 'high',           # 家庭共同居住 → 高强度（长时间近距离）
        'family': 'high',              # 家属
        'spouse': 'high',              # 配偶
        'medical': 'high',             # 诊疗操作（近距离飞沫暴露风险高）
        'colleague': 'medium',         # 同事
        'classmate': 'medium',         # 同学
        'social': 'medium',            # 社会接触
        'casual': 'low',               # 偶然接触
        'cdc_contact': 'medium',       # 疾控反馈的密接
    }

    # 接触强度基础 → 筛查优先级映射
    INTENSITY_TO_PRIORITY = {
        'high': 'high',
        'medium': 'medium',
        'low': 'low',
    }

    def __init__(self, gnn_network=None, db_manager=None, auditor=None,
                 screening_days: int = 14):
        """初始化接触追踪管理器。
        
        参数：
            gnn_network: GNN网络适配器（需符合GNNNetworkProtocol协议），
                        None时使用NullGNNNetwork优雅降级，不影响核心报卡流程
            db_manager: 数据库管理器
            auditor: 审计器（需有log_operation方法）
            screening_days: 筛查任务默认截止天数
        """
        self.gnn_network = gnn_network if gnn_network else NullGNNNetwork()
        self.db_manager = db_manager
        # 如果未传入auditor，使用默认审计日志器
        self.auditor = auditor if auditor is not None else get_default_auditor()
        self.screening_days = screening_days
        
        # 回调钩子
        self._on_task_created: List[Callable[[ScreeningTask], None]] = []
        self._on_network_updated: List[Callable[[str, List[ContactRecord]], None]] = []
        self._on_reassessment_needed: List[Callable[[List[str]], None]] = []
        self._on_positive_found: List[Callable[[ContactRecord], None]] = []
        
        # 内存缓存
        self._contacts_cache: Dict[str, ContactRecord] = {}
        self._tasks_cache: Dict[str, ScreeningTask] = {}
        
        # 确保数据库表存在
        self._ensure_tables()

    def set_gnn_network(self, gnn_network):
        """设置GNN网络对象。"""
        self.gnn_network = gnn_network

    # ------------------------------------------------------------------
    # 回调注册
    # ------------------------------------------------------------------

    def on_screening_task_created(self, callback: Callable[[ScreeningTask], None]):
        """注册筛查任务创建回调。"""
        self._on_task_created.append(callback)

    def on_network_updated(self, callback: Callable[[str, List[ContactRecord]], None]):
        """注册网络更新回调（参数：source_patient_id, updated_contacts）。"""
        self._on_network_updated.append(callback)

    def on_reassessment_needed(self, callback: Callable[[List[str]], None]):
        """注册风险重评估回调（参数：需要重评估的患者ID列表）。"""
        self._on_reassessment_needed.append(callback)

    def on_positive_contact_found(self, callback: Callable[[ContactRecord], None]):
        """注册阳性接触者发现回调（用于触发新一轮密接追踪）。"""
        self._on_positive_found.append(callback)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _calculate_contact_intensity(self, contact_type: str, contact_duration: str = "") -> str:
        """根据接触类型和时长评估接触强度。
        
        参数：
            contact_type: 接触类型
            contact_duration: 接触时长描述
        返回：
            "high"/"medium"/"low"
        """
        # 先根据类型确定基础强度
        base_intensity = self.CONTACT_INTENSITY.get(contact_type, 'medium')
        
        # 根据接触时长调整
        duration_lower = contact_duration.lower()
        if contact_duration:
            # 长时间接触升级强度
            if any(k in duration_lower for k in (">8小时", "频繁", "每日", "daily", ">8h", "长期")):
                return 'high'
            # 短暂接触降级强度
            if any(k in duration_lower for k in ("<1小时", "短暂", "brief", "偶尔", "一次")):
                if base_intensity != 'high':  # 高基础强度不因短暂而降级太多
                    return 'low'
        
        return base_intensity
    
    def _calculate_screening_priority(self, contact_intensity: str,
                                       case_classification: str = None) -> str:
        """根据接触强度和传染源传染性计算筛查优先级。
        
        参数：
            contact_intensity: 接触强度
            case_classification: 传染源病例分类（涂阳/培阳等传染性更强），
                                默认None，不隐式假设为确诊病例
        返回：
            "high"/"medium"/"low"
        """
        base_priority = self.INTENSITY_TO_PRIORITY.get(contact_intensity, 'medium')
        
        # 只有显式传入高传染性病例分类时才提升优先级
        high_risk_cases = {"smear_positive", "culture_positive", "bacteriologically_positive",
                          "confirmed", "xdr", "mdr"}
        if case_classification and case_classification.lower() in high_risk_cases:
            if base_priority == 'low':
                return 'medium'
        
        return base_priority

    # ------------------------------------------------------------------
    # 密接追踪（确诊病例 → GNN提取接触者 → 生成筛查任务）
    # ------------------------------------------------------------------

    def trace_contacts_from_confirmed_case(
        self, 
        patient_id: str,
        patient_name: str = "",
        case_classification: str = "confirmed",
        contact_depth: int = 1,
        custom_screening_items: List[str] = None,
    ) -> List[ContactRecord]:
        """从确诊病例触发密切接触者追踪。
        
        流程：
        1. 从GNN网络提取与该患者有接触的人员
        2. 为每个接触者创建接触记录
        3. 生成筛查任务
        4. 触发回调通知
        
        参数：
            patient_id: 确诊病例患者ID
            patient_name: 患者姓名
            case_classification: 病例分类（confirmed/suspected等）
            contact_depth: 接触追踪深度（1=直接接触，2=间接接触）
            custom_screening_items: 自定义筛查项目
            
        返回：
            List[ContactRecord]: 创建的接触记录列表
        """
        LOGGER.info("开始密接追踪: 病例=%s, 深度=%d", patient_id, contact_depth)
        
        # 从GNN网络提取接触者
        gnn_contacts = self._get_contacts_from_gnn(patient_id, contact_depth)
        
        # 如果GNN网络不可用，返回空列表
        if not gnn_contacts and self.gnn_network is None:
            LOGGER.warning("GNN网络不可用，无法自动提取接触者")
            return []
        
        contacts = []
        screening_items = custom_screening_items or self.DEFAULT_SCREENING_ITEMS
        
        for contact_data in gnn_contacts:
            # 创建接触记录
            contact = ContactRecord(
                source_patient_id=patient_id,
                contact_patient_id=contact_data.get('patient_id', ''),
                contact_name=contact_data.get('name', ''),
                relationship=contact_data.get('relationship', ''),
                contact_type=contact_data.get('contact_type', 'social'),
                contact_date=contact_data.get('last_contact_date', ''),
                contact_duration=contact_data.get('duration', ''),
                contact_intensity=contact_data.get('intensity', 'medium'),
            )
            
            # 确定接触强度和筛查优先级
            contact.contact_intensity = self._calculate_contact_intensity(
                contact.contact_type, contact.contact_duration
            )
            contact.screening_priority = self._calculate_screening_priority(
                contact.contact_intensity, case_classification
            )
            
            # 保存接触记录
            self._save_contact(contact)
            contacts.append(contact)
            
            # 创建筛查任务
            task = self._create_screening_task(
                contact=contact,
                screening_items=screening_items,
                source_case=patient_id,
            )
            
            # 触发回调
            for callback in self._on_task_created:
                try:
                    callback(task)
                except Exception as e:
                    LOGGER.debug("筛查任务回调执行失败: %s", e)
            
            # 记录审计日志
            if self.auditor:
                try:
                    self.auditor.log_operation(
                        operation_type='ASSESS',
                        patient_id=contact.contact_patient_id or None,
                        patient_name=contact.contact_name,
                        record_id=task.task_id,
                        details={
                            'action': 'contact_tracing',
                            'source_case': patient_id,
                            'contact_type': contact.contact_type,
                            'priority': task.priority,
                        },
                        reason=f'确诊病例({patient_name or patient_id})的密切接触者筛查'
                    )
                except Exception as e:
                    LOGGER.debug("写入密接追踪审计日志失败: %s", e)
        
        LOGGER.info("密接追踪完成: 病例=%s, 发现接触者=%d人", patient_id, len(contacts))
        return contacts

    def _get_contacts_from_gnn(self, patient_id: str, depth: int = 1) -> List[Dict[str, Any]]:
        """从GNN网络提取接触者。
        
        参数：
            patient_id: 患者ID
            depth: 追踪深度
            
        返回：
            List[Dict]: 接触者列表
        """
        contacts = []
        
        if self.gnn_network is None:
            return contacts
        
        try:
            # 尝试调用GNN网络的方法
            if hasattr(self.gnn_network, 'get_contacts'):
                raw_contacts = self.gnn_network.get_contacts(patient_id, depth=depth)
                if isinstance(raw_contacts, list):
                    contacts.extend(raw_contacts)
            elif hasattr(self.gnn_network, 'get_neighbors'):
                # 兼容不同的GNN接口
                neighbors = self.gnn_network.get_neighbors(patient_id)
                if isinstance(neighbors, list):
                    for n in neighbors:
                        if isinstance(n, dict):
                            contacts.append(n)
                        else:
                            contacts.append({'patient_id': str(n)})
        except Exception as e:
            LOGGER.warning("从GNN网络提取接触者失败: %s", e)
        
        # 如果数据库中有接触者数据，补充进去
        db_contacts = self._get_contacts_from_db(patient_id)
        existing_ids = {c.get('patient_id') or c.get('contact_patient_id') for c in contacts}
        
        for dc in db_contacts:
            cid = dc.get('contact_patient_id')
            if cid and cid not in existing_ids:
                contacts.append(dc)
                existing_ids.add(cid)
        
        return contacts

    def _get_contacts_from_db(self, patient_id: str) -> List[Dict[str, Any]]:
        """从数据库获取接触者记录。"""
        contacts = []
        if not self.db_manager:
            return contacts
        
        try:
            rows = self.db_manager.execute("""
                SELECT * FROM contact_records 
                WHERE source_patient_id = ? OR contact_patient_id = ?
                ORDER BY created_at DESC
            """, [patient_id, patient_id])
            
            for row in rows:
                if isinstance(row, (tuple, list)):
                    # 简单转换
                    contact = {
                        'patient_id': row[2] if len(row) > 2 else '',
                        'name': row[3] if len(row) > 3 else '',
                        'relationship': row[4] if len(row) > 4 else '',
                        'contact_type': row[5] if len(row) > 5 else 'social',
                    }
                else:
                    contact = dict(row)
                contacts.append(contact)
        except Exception as e:
            LOGGER.debug("从数据库查询接触者失败: %s", e)
        
        return contacts

    def _create_screening_task(self, contact: ContactRecord,
                                screening_items: List[str],
                                source_case: str) -> ScreeningTask:
        """创建筛查任务。"""
        due_date = (
            datetime.datetime.now() + datetime.timedelta(days=self.screening_days)
        ).strftime('%Y-%m-%d')
        
        # 使用screening_priority字段，fallback到通过contact_type计算
        priority = contact.screening_priority or self._calculate_screening_priority(
            contact.contact_intensity or self._calculate_contact_intensity(contact.contact_type)
        )
        
        task = ScreeningTask(
            source_case_id=source_case,
            contact_id=contact.contact_id,
            target_patient_id=contact.contact_patient_id,
            target_name=contact.contact_name,
            priority=priority,
            due_date=due_date,
            screening_items=screening_items,
            notes=f"与确诊病例{source_case}的{contact.relationship or '密切'}接触",
        )
        
        self._save_task(task)
        return task

    # ------------------------------------------------------------------
    # 处理疾控回传筛查结果（疾控数据 → 更新GNN网络 → 触发重评估）
    # ------------------------------------------------------------------

    def handle_screening_results(
        self,
        source_patient_id: str,
        screening_results: List[Dict[str, Any]],
        result_source: str = "cdc_callback",
    ) -> Dict[str, Any]:
        """处理疾控回传的密切接触者筛查结果。
        
        流程：
        1. 更新接触记录的筛查结果
        2. 阳性接触者标记为活动性结核
        3. 更新GNN接触网络
        4. 触发相关人员风险重评估
        5. 阳性病例触发新一轮密接追踪
        
        参数：
            source_patient_id: 来源患者ID
            screening_results: 筛查结果列表
            result_source: 结果来源（cdc_callback/manual_entry等）
            
        返回：
            Dict: 处理结果统计
        """
        LOGGER.info("处理筛查结果: 来源=%s, 结果数=%d", source_patient_id, len(screening_results))
        
        stats = {
            'total_received': len(screening_results),
            'updated_contacts': 0,
            'positive_cases': 0,
            'negative_cases': 0,
            'reassessment_triggered': 0,
            'new_trace_tasks': 0,
        }
        
        patients_to_reassess: Set[str] = set()
        positive_contacts: List[ContactRecord] = []
        updated_contacts: List[ContactRecord] = []
        
        for result in screening_results:
            contact_id = result.get('contact_id') or result.get('id')
            contact_patient_id = result.get('patient_id') or result.get('contact_patient_id', '')
            
            # 查找现有接触记录
            contact = self._find_contact(contact_id, contact_patient_id, source_patient_id)
            
            if not contact:
                # 创建新的接触记录（疾控回传的新接触者）
                contact = ContactRecord(
                    source_patient_id=source_patient_id,
                    contact_patient_id=contact_patient_id,
                    contact_name=result.get('name', ''),
                    relationship=result.get('relationship', ''),
                    contact_type=result.get('contact_type', 'cdc_contact'),
                )
            
            # 更新筛查结果
            contact.screening_result = result.get('screening_result', '')
            contact.screening_date = result.get('screening_date', 
                                                  datetime.datetime.now().strftime('%Y-%m-%d'))
            contact.updated_at = datetime.datetime.now().isoformat()
            
            # 使用精确NLP分类判断筛查结果（避免"无异常"误判）
            screening_text = str(contact.screening_result)
            result_class = result.get('screening_classification')
            if not result_class:
                result_class = CDCCallbackParser.classify_screening_result(screening_text)
            
            is_positive = (result_class == "positive")
            
            # 对于新创建的疾控接触记录，计算强度和优先级
            if not contact.contact_intensity:
                contact.contact_intensity = self._calculate_contact_intensity(contact.contact_type)
                contact.screening_priority = self._calculate_screening_priority(contact.contact_intensity)
            
            contact.is_active_tb = is_positive
            
            if is_positive:
                contact.tracing_status = "positive"
                stats['positive_cases'] += 1
                positive_contacts.append(contact)
            elif result_class == "negative":
                contact.tracing_status = "negative"
                stats['negative_cases'] += 1
            else:
                contact.tracing_status = "pending"  # pending/screened待确认
            
            # 保存更新
            self._save_contact(contact)
            updated_contacts.append(contact)
            stats['updated_contacts'] += 1
            
            # 更新GNN网络
            self._update_gnn_network(contact, is_positive)
            
            # 收集需要重评估的患者
            if contact.contact_patient_id:
                patients_to_reassess.add(contact.contact_patient_id)
            # 传染源也需要重评估（更新传播链信息）
            patients_to_reassess.add(source_patient_id)
        
        # 更新GNN网络边权重/属性
        self._finalize_gnn_update(source_patient_id, updated_contacts)
        
        # 触发网络更新回调
        for callback in self._on_network_updated:
            try:
                callback(source_patient_id, updated_contacts)
            except Exception as e:
                LOGGER.debug("网络更新回调失败: %s", e)
        
        # 触发风险重评估
        if patients_to_reassess:
            patient_list = list(patients_to_reassess)
            stats['reassessment_triggered'] = len(patient_list)
            for callback in self._on_reassessment_needed:
                try:
                    callback(patient_list)
                except Exception as e:
                    LOGGER.debug("重评估回调失败: %s", e)
        
        # 阳性病例触发新一轮密接追踪
        for pos_contact in positive_contacts:
            if pos_contact.contact_patient_id:
                new_contacts = self.trace_contacts_from_confirmed_case(
                    patient_id=pos_contact.contact_patient_id,
                    patient_name=pos_contact.contact_name,
                )
                stats['new_trace_tasks'] += len(new_contacts)
                
                for callback in self._on_positive_found:
                    try:
                        callback(pos_contact)
                    except Exception as e:
                        LOGGER.debug("阳性接触者回调失败: %s", e)
        
        # 记录审计日志
        if self.auditor:
            try:
                self.auditor.log_operation(
                    operation_type='UPDATE',
                    patient_id=source_patient_id,
                    details={
                        'action': 'screening_results_processed',
                        'source': result_source,
                        **stats,
                    },
                    reason=f'处理{stats["total_received"]}条筛查结果，发现{stats["positive_cases"]}例阳性'
                )
            except Exception as e:
                LOGGER.debug("写入筛查结果处理审计日志失败: %s", e)
        
        LOGGER.info(
            "筛查结果处理完成: 更新=%d, 阳性=%d, 触发重评估=%d, 新追踪=%d",
            stats['updated_contacts'], stats['positive_cases'],
            stats['reassessment_triggered'], stats['new_trace_tasks']
        )
        
        return stats

    def _find_contact(self, contact_id: str, contact_patient_id: str,
                       source_patient_id: str) -> Optional[ContactRecord]:
        """查找接触记录。"""
        # 先查缓存
        if contact_id and contact_id in self._contacts_cache:
            return self._contacts_cache[contact_id]
        
        # 查数据库
        if self.db_manager:
            try:
                if contact_id:
                    rows = self.db_manager.execute(
                        "SELECT * FROM contact_records WHERE contact_id = ?",
                        [contact_id]
                    )
                elif contact_patient_id and source_patient_id:
                    rows = self.db_manager.execute("""
                        SELECT * FROM contact_records 
                        WHERE contact_patient_id = ? AND source_patient_id = ?
                        ORDER BY created_at DESC LIMIT 1
                    """, [contact_patient_id, source_patient_id])
                else:
                    rows = None
                
                if rows and len(rows) > 0:
                    row = rows[0]
                    if isinstance(row, (tuple, list)):
                        contact = self._row_to_contact(row)
                    else:
                        contact = self._dict_to_contact(dict(row))
                    self._contacts_cache[contact.contact_id] = contact
                    return contact
            except Exception as e:
                LOGGER.debug("查询接触记录失败: %s", e)
        
        return None

    def _update_gnn_network(self, contact: ContactRecord, is_positive: bool):
        """更新GNN网络中的节点/边属性。"""
        if self.gnn_network is None:
            return
        
        try:
            # 更新接触者节点状态
            if contact.contact_patient_id:
                node_attrs = {
                    'screening_result': contact.screening_result,
                    'is_active_tb': is_positive,
                    'last_screening_date': contact.screening_date,
                    'contact_traced': True,
                }
                
                if hasattr(self.gnn_network, 'update_node'):
                    self.gnn_network.update_node(contact.contact_patient_id, node_attrs)
            
            # 更新接触边属性
            if contact.source_patient_id and contact.contact_patient_id:
                edge_attrs = {
                    'relationship': contact.relationship,
                    'contact_type': contact.contact_type,
                    'screening_completed': bool(contact.screening_result),
                    'screening_result': contact.screening_result if contact.screening_result else None,
                    'transmission_confirmed': is_positive,
                }
                
                if hasattr(self.gnn_network, 'update_edge'):
                    self.gnn_network.update_edge(
                        contact.source_patient_id,
                        contact.contact_patient_id,
                        edge_attrs,
                    )
        except Exception as e:
            LOGGER.warning("更新GNN网络失败: %s", e)

    def _finalize_gnn_update(self, source_patient_id: str, contacts: List[ContactRecord]):
        """完成GNN网络更新（可触发GNN重计算等）。"""
        if self.gnn_network is None:
            return
        
        try:
            # 标记传染源节点为确诊
            if hasattr(self.gnn_network, 'update_node'):
                self.gnn_network.update_node(source_patient_id, {
                    'is_confirmed': True,
                    'status': 'confirmed_tb',
                })
        except Exception as e:
            LOGGER.debug("最终更新GNN失败: %s", e)

    def add_contact_manually(self, source_patient_id: str, contact_patient_id: str,
                              relationship: str, contact_type: str = "social",
                              contact_name: str = "", case_classification: str = None,
                              contact_duration: str = "", **kwargs) -> ContactRecord:
        """手动添加接触记录（医生录入）。
        
        参数：
            source_patient_id: 传染源患者ID
            contact_patient_id: 接触者患者ID
            relationship: 关系
            contact_type: 接触类型
            contact_name: 接触者姓名
            case_classification: 传染源病例分类（用于计算筛查优先级）
            contact_duration: 接触时长描述
            **kwargs: 其他字段
        """
        contact = ContactRecord(
            source_patient_id=source_patient_id,
            contact_patient_id=contact_patient_id,
            contact_name=contact_name,
            relationship=relationship,
            contact_type=contact_type,
            **kwargs,
        )
        
        # 计算接触强度（与自动追踪路径保持一致）
        contact.contact_intensity = self._calculate_contact_intensity(
            contact_type, contact_duration
        )
        # 计算筛查优先级
        contact.screening_priority = self._calculate_screening_priority(
            contact.contact_intensity, case_classification
        )
        
        self._save_contact(contact)
        
        # 更新GNN网络
        if self.gnn_network and hasattr(self.gnn_network, 'add_contact'):
            try:
                self.gnn_network.add_contact(source_patient_id, contact_patient_id, {
                    'relationship': relationship,
                    'contact_type': contact_type,
                    'manual_entry': True,
                })
            except Exception as e:
                LOGGER.debug("手动添加接触到GNN失败: %s", e)
        
        return contact

    # ------------------------------------------------------------------
    # 查询方法
    # ------------------------------------------------------------------

    def get_contacts_for_patient(self, patient_id: str) -> List[ContactRecord]:
        """获取患者的所有接触者记录（同时查询内存缓存和数据库）。"""
        contacts = []
        seen_ids = set()
        
        # 先从内存缓存中查找
        for c in self._contacts_cache.values():
            if c.source_patient_id == patient_id or c.contact_patient_id == patient_id:
                contacts.append(c)
                seen_ids.add(c.contact_id)
        
        # 再从数据库查询
        if self.db_manager:
            try:
                rows = self.db_manager.execute("""
                    SELECT * FROM contact_records
                    WHERE source_patient_id = ? OR contact_patient_id = ?
                    ORDER BY created_at DESC
                """, [patient_id, patient_id])
                
                for row in rows:
                    if isinstance(row, (tuple, list)):
                        c = self._row_to_contact(row)
                    else:
                        c = self._dict_to_contact(dict(row))
                    if c.contact_id not in seen_ids:
                        contacts.append(c)
                        # 同时更新内存缓存
                        self._contacts_cache[c.contact_id] = c
            except Exception as e:
                LOGGER.debug("查询患者接触者失败: %s", e)
        
        return contacts

    def get_pending_screening_tasks(self, assigned_to: str = None,
                                      limit: int = 50) -> List[ScreeningTask]:
        """获取待处理的筛查任务。"""
        tasks = []
        
        if self.db_manager:
            try:
                if assigned_to:
                    rows = self.db_manager.execute("""
                        SELECT * FROM screening_tasks
                        WHERE status = 'pending' AND assigned_to = ?
                        ORDER BY priority DESC, created_at ASC LIMIT ?
                    """, [assigned_to, limit])
                else:
                    rows = self.db_manager.execute("""
                        SELECT * FROM screening_tasks
                        WHERE status = 'pending'
                        ORDER BY priority DESC, created_at ASC LIMIT ?
                    """, [limit])
                
                for row in rows:
                    if isinstance(row, (tuple, list)):
                        tasks.append(self._row_to_task(row))
                    else:
                        tasks.append(self._dict_to_task(dict(row)))
            except Exception as e:
                LOGGER.debug("查询筛查任务失败: %s", e)
        
        return tasks

    def get_tracing_summary(self, patient_id: str = None) -> Dict[str, Any]:
        """获取接触追踪摘要统计。"""
        summary = {
            'total_contacts': 0,
            'screened': 0,
            'positive': 0,
            'negative': 0,
            'pending': 0,
            'tasks_pending': 0,
        }
        
        if self.db_manager:
            try:
                for status_key, status_val in [
                    ('total_contacts', None),
                    ('screened', 'screened'),
                    ('positive', 'positive'),
                    ('negative', 'negative'),
                    ('pending', 'pending'),
                ]:
                    if status_val is None:
                        rows = self.db_manager.execute(
                            "SELECT COUNT(*) FROM contact_records"
                        )
                    else:
                        rows = self.db_manager.execute(
                            "SELECT COUNT(*) FROM contact_records WHERE tracing_status = ?",
                            [status_val]
                        )
                    if rows:
                        summary[status_key] = rows[0][0] if isinstance(rows[0], (tuple, list)) else list(rows[0].values())[0]
                
                task_rows = self.db_manager.execute(
                    "SELECT COUNT(*) FROM screening_tasks WHERE status = 'pending'"
                )
                if task_rows:
                    summary['tasks_pending'] = task_rows[0][0] if isinstance(task_rows[0], (tuple, list)) else list(task_rows[0].values())[0]
            except Exception as e:
                LOGGER.debug("获取追踪摘要失败: %s", e)
        
        return summary

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _ensure_tables(self):
        """确保数据库表存在。"""
        if not self.db_manager:
            return
        
        try:
            self.db_manager.execute("""
                CREATE TABLE IF NOT EXISTS contact_records (
                    contact_id TEXT PRIMARY KEY,
                    source_patient_id TEXT NOT NULL,
                    contact_patient_id TEXT,
                    contact_name TEXT,
                    relationship TEXT,
                    contact_type TEXT,
                    contact_date TEXT,
                    contact_duration TEXT,
                    contact_intensity TEXT,
                    screening_priority TEXT,
                    is_active_tb INTEGER DEFAULT 0,
                    screening_result TEXT,
                    screening_date TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    tracing_status TEXT DEFAULT 'pending'
                )
            """)
            # 尝试添加screening_priority列（兼容旧表）
            try:
                self.db_manager.execute("ALTER TABLE contact_records ADD COLUMN screening_priority TEXT")
            except Exception as e:
                LOGGER.debug("添加screening_priority列失败（列可能已存在）: %s", e)  # 列已存在或不支持ALTER
            self.db_manager.execute("""
                CREATE INDEX IF NOT EXISTS idx_contact_source
                ON contact_records(source_patient_id)
            """)
            self.db_manager.execute("""
                CREATE INDEX IF NOT EXISTS idx_contact_target
                ON contact_records(contact_patient_id)
            """)
            self.db_manager.execute("""
                CREATE INDEX IF NOT EXISTS idx_contact_status
                ON contact_records(tracing_status)
            """)
            self.db_manager.execute("""
                CREATE TABLE IF NOT EXISTS screening_tasks (
                    task_id TEXT PRIMARY KEY,
                    source_case_id TEXT NOT NULL,
                    contact_id TEXT,
                    target_patient_id TEXT,
                    target_name TEXT,
                    task_type TEXT DEFAULT 'contact_screening',
                    priority TEXT DEFAULT 'medium',
                    status TEXT DEFAULT 'pending',
                    assigned_to TEXT,
                    due_date TEXT,
                    screening_items TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    created_by TEXT,
                    completed_at TEXT,
                    result_summary TEXT
                )
            """)
            self.db_manager.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_status
                ON screening_tasks(status, priority)
            """)
            self.db_manager.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_assignee
                ON screening_tasks(assigned_to)
            """)
        except Exception as e:
            LOGGER.warning("创建接触追踪表失败: %s", e)

    def _save_contact(self, contact: ContactRecord):
        """保存接触记录。"""
        contact.updated_at = datetime.datetime.now().isoformat()
        self._contacts_cache[contact.contact_id] = contact
        
        if self.db_manager:
            try:
                self.db_manager.execute("""
                    INSERT OR REPLACE INTO contact_records
                    (contact_id, source_patient_id, contact_patient_id, contact_name,
                     relationship, contact_type, contact_date, contact_duration,
                     contact_intensity, screening_priority, is_active_tb, screening_result, screening_date,
                     notes, created_at, updated_at, tracing_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    contact.contact_id,
                    contact.source_patient_id,
                    contact.contact_patient_id,
                    contact.contact_name,
                    contact.relationship,
                    contact.contact_type,
                    contact.contact_date,
                    contact.contact_duration,
                    contact.contact_intensity,
                    contact.screening_priority,
                    1 if contact.is_active_tb else 0,
                    contact.screening_result,
                    contact.screening_date,
                    contact.notes,
                    contact.created_at,
                    contact.updated_at,
                    contact.tracing_status,
                ])
            except Exception as e:
                LOGGER.debug("保存接触记录失败: %s", e)

    def _save_task(self, task: ScreeningTask):
        """保存筛查任务。"""
        self._tasks_cache[task.task_id] = task
        
        if self.db_manager:
            try:
                import json
                self.db_manager.execute("""
                    INSERT OR REPLACE INTO screening_tasks
                    (task_id, source_case_id, contact_id, target_patient_id,
                     target_name, task_type, priority, status, assigned_to,
                     due_date, screening_items, notes, created_at, created_by,
                     completed_at, result_summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    task.task_id,
                    task.source_case_id,
                    task.contact_id,
                    task.target_patient_id,
                    task.target_name,
                    task.task_type,
                    task.priority,
                    task.status,
                    task.assigned_to,
                    task.due_date,
                    json.dumps(task.screening_items, ensure_ascii=False) if task.screening_items else None,
                    task.notes,
                    task.created_at,
                    task.created_by,
                    task.completed_at,
                    task.result_summary,
                ])
            except Exception as e:
                LOGGER.debug("保存筛查任务失败: %s", e)

    def _row_to_contact(self, row: tuple) -> ContactRecord:
        """将数据库行转换为ContactRecord。"""
        columns = ['contact_id', 'source_patient_id', 'contact_patient_id', 'contact_name',
                   'relationship', 'contact_type', 'contact_date', 'contact_duration',
                   'contact_intensity', 'screening_priority', 'is_active_tb', 'screening_result',
                   'screening_date', 'notes', 'created_at', 'updated_at', 'tracing_status']
        data = {}
        for i, col in enumerate(columns):
            if i < len(row):
                data[col] = row[i]
        contact = ContactRecord()
        for k, v in data.items():
            if k == 'is_active_tb':
                setattr(contact, k, bool(v))
            else:
                setattr(contact, k, v if v is not None else '')
        return contact

    def _dict_to_contact(self, data: Dict[str, Any]) -> ContactRecord:
        """将字典转换为ContactRecord。"""
        contact = ContactRecord()
        for k, v in data.items():
            if k == 'is_active_tb':
                setattr(contact, k, bool(v))
            elif hasattr(contact, k):
                setattr(contact, k, v if v is not None else '')
        return contact

    def _row_to_task(self, row: tuple) -> ScreeningTask:
        """将数据库行转换为ScreeningTask。"""
        columns = ['task_id', 'source_case_id', 'contact_id', 'target_patient_id',
                   'target_name', 'task_type', 'priority', 'status', 'assigned_to',
                   'due_date', 'screening_items', 'notes', 'created_at', 'created_by',
                   'completed_at', 'result_summary']
        data = {}
        for i, col in enumerate(columns):
            if i < len(row):
                data[col] = row[i]
        
        task = ScreeningTask()
        for k, v in data.items():
            if k == 'screening_items' and v:
                try:
                    import json
                    setattr(task, k, json.loads(v))
                except Exception:
                    setattr(task, k, [])
            elif hasattr(task, k):
                setattr(task, k, v if v is not None else '')
        return task

    def _dict_to_task(self, data: Dict[str, Any]) -> ScreeningTask:
        """将字典转换为ScreeningTask。"""
        task = ScreeningTask()
        for k, v in data.items():
            if k == 'screening_items' and isinstance(v, str):
                try:
                    import json
                    setattr(task, k, json.loads(v))
                except Exception:
                    setattr(task, k, [])
            elif hasattr(task, k):
                setattr(task, k, v if v is not None else '')
        return task
