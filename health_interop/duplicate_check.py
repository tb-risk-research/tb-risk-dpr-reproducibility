#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""传染病报告查重模块

实现多维度查重检测，防止重报：
1. 精确匹配：身份证号完全一致
2. 近似匹配：姓名同音不同字、身份证号部分匹配、出生日期+姓名组合
3. 相似度评分：综合评分判定疑似重复
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("tb_risk.health_interop.duplicate_check")

# 常用汉字拼音映射（用于同音匹配，简化版）
# 实际生产环境可使用pypinyin库
_PINYIN_INITIALS = {
    '张': 'zhang', '王': 'wang', '李': 'li', '赵': 'zhao', '刘': 'liu',
    '陈': 'chen', '杨': 'yang', '黄': 'huang', '周': 'zhou', '吴': 'wu',
    '徐': 'xu', '孙': 'sun', '胡': 'hu', '朱': 'zhu', '高': 'gao',
    '林': 'lin', '何': 'he', '郭': 'guo', '马': 'ma', '罗': 'luo',
    '梁': 'liang', '宋': 'song', '郑': 'zheng', '谢': 'xie', '韩': 'han',
    '唐': 'tang', '冯': 'feng', '于': 'yu', '董': 'dong', '萧': 'xiao',
    '程': 'cheng', '曹': 'cao', '袁': 'yuan', '邓': 'deng', '许': 'xu',
    '傅': 'fu', '沈': 'shen', '曾': 'zeng', '彭': 'peng', '吕': 'lv',
    '苏': 'su', '卢': 'lu', '蒋': 'jiang', '蔡': 'cai', '贾': 'jia',
    '丁': 'ding', '魏': 'wei', '薛': 'xue', '叶': 'ye', '阎': 'yan',
    '余': 'yu', '潘': 'pan', '杜': 'du', '戴': 'dai', '夏': 'xia',
    '钟': 'zhong', '汪': 'wang', '田': 'tian', '任': 'ren', '姜': 'jiang',
    '范': 'fan', '方': 'fang', '石': 'shi', '姚': 'yao', '谭': 'tan',
    '廖': 'liao', '邹': 'zou', '熊': 'xiong', '金': 'jin', '陆': 'lu',
    '郝': 'hao', '孔': 'kong', '白': 'bai', '崔': 'cui', '康': 'kang',
    '毛': 'mao', '邱': 'qiu', '秦': 'qin', '江': 'jiang', '史': 'shi',
    '顾': 'gu', '侯': 'hou', '邵': 'shao', '孟': 'meng', '龙': 'long',
    '万': 'wan', '段': 'duan', '章': 'zhang', '钱': 'qian', '汤': 'tang',
    '尹': 'yin', '黎': 'li', '易': 'yi', '常': 'chang', '武': 'wu',
    '乔': 'qiao', '贺': 'he', '赖': 'lai', '龚': 'gong', '文': 'wen',
}


@dataclass
class DuplicateMatch:
    """查重匹配结果"""
    patient_id: str
    patient_name: str
    id_card: str
    match_type: str           # exact/phonetic/partial_id/birthday_name/composite
    similarity_score: float   # 0-100
    existing_card_id: str
    existing_patient_name: str
    existing_id_card: str
    match_details: Dict[str, Any]
    report_date: Optional[str] = None
    
    @property
    def is_confirmed_duplicate(self) -> bool:
        """是否确认重复（高分）"""
        return self.similarity_score >= 90
    
    @property
    def is_suspected_duplicate(self) -> bool:
        """是否疑似重复（中等分数）"""
        return 70 <= self.similarity_score < 90


class DuplicateChecker:
    """报告卡查重器
    
    支持多维度匹配策略：
    - 精确匹配：身份证号完全一致
    - 拼音同音匹配：姓名同音不同字
    - 身份证部分匹配：出生年月日一致+地址码前6位一致
    - 姓名+出生日期组合匹配
    """
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        # 查重阈值
        self.exact_threshold = 100
        self.phonetic_threshold = 85
        self.partial_id_threshold = 80
        self.composite_threshold = 70
    
    def set_db_manager(self, db_manager):
        """设置数据库管理器"""
        self.db_manager = db_manager
    
    def check_duplicate(self, 
                        patient_name: str,
                        id_card: str = "",
                        birthday: str = "",
                        gender: str = "",
                        existing_cards: List[Dict[str, Any]] = None) -> List[DuplicateMatch]:
        """检查是否存在重复报告
        
        参数：
            patient_name: 患者姓名
            id_card: 身份证号
            birthday: 出生日期 (YYYY-MM-DD)
            gender: 性别
            existing_cards: 已有卡片列表（如果不提供则从数据库查询）
            
        返回：
            List[DuplicateMatch]: 匹配结果列表，按相似度降序排列
        """
        matches = []
        
        # 获取已有卡片
        if existing_cards is None and self.db_manager:
            existing_cards = self._load_existing_cards()
        if not existing_cards:
            return matches
        
        # 规范化查询参数
        norm_name = self._normalize_name(patient_name)
        norm_id = self._normalize_id_card(id_card)
        birthday_from_id = self._extract_birthday_from_id(norm_id) if norm_id else ""
        norm_birthday = birthday or birthday_from_id
        
        for card in existing_cards:
            card_name = self._normalize_name(card.get('patient_name', ''))
            card_id = self._normalize_id_card(card.get('id_card', ''))
            card_birthday = card.get('birthday', '') or self._extract_birthday_from_id(card_id)
            
            if not card_name:
                continue
            
            # 1. 身份证精确匹配（最高优先级）
            if norm_id and card_id and norm_id == card_id:
                matches.append(DuplicateMatch(
                    patient_id=card.get('patient_id', ''),
                    patient_name=patient_name,
                    id_card=id_card,
                    match_type='exact',
                    similarity_score=100,
                    existing_card_id=card.get('card_id', ''),
                    existing_patient_name=card.get('patient_name', ''),
                    existing_id_card=card.get('id_card', ''),
                    match_details={'reason': '身份证号完全一致'},
                    report_date=card.get('report_date'),
                ))
                continue
            
            # 2. 姓名+身份证部分匹配（出生日期+地址码前6位）
            score = 0
            details = {}
            
            if norm_id and card_id and len(norm_id) >= 14 and len(card_id) >= 14:
                # 比较出生日期部分（第7-14位）
                id_birthday_match = norm_id[6:14] == card_id[6:14]
                # 比较地址码前6位
                id_region_match = norm_id[:6] == card_id[:6]
                # 比较顺序码（第15-17位，同地区同日出生的顺序号）
                id_seq_match = norm_id[14:17] == card_id[14:17]
                
                if id_birthday_match and norm_name == card_name:
                    score += 70
                    details['id_birthday_match'] = True
                    if id_region_match:
                        score += 15
                        details['id_region_match'] = True
                    if id_seq_match:
                        score += 10
                        details['id_seq_match'] = True
                    score = min(score, 95)
            
            # 3. 同音姓名匹配
            if not score and norm_name and card_name:
                if self._is_phonetic_match(norm_name, card_name):
                    name_score = 80
                    # 如果出生日期也匹配，加分
                    if norm_birthday and card_birthday and norm_birthday == card_birthday:
                        name_score += 15
                    score = max(score, name_score)
                    details['phonetic_name_match'] = True
                    details['pinyin_query'] = self._name_to_pinyin(norm_name)
                    details['pinyin_existing'] = self._name_to_pinyin(card_name)
            
            # 4. 姓名+出生日期组合匹配
            if score < self.composite_threshold and norm_name == card_name:
                if norm_birthday and card_birthday and norm_birthday == card_birthday:
                    composite_score = 75
                    if gender and card.get('gender') and gender == card.get('gender'):
                        composite_score += 10
                    score = max(score, composite_score)
                    details['name_birthday_match'] = True
            
            # 5. 编辑距离相似度（姓名相似）
            if score < self.composite_threshold and norm_name and card_name:
                name_similarity = self._name_similarity(norm_name, card_name)
                if name_similarity >= 0.8:
                    sim_score = int(name_similarity * 80)
                    if norm_birthday and card_birthday and norm_birthday == card_birthday:
                        sim_score += 15
                    score = max(score, sim_score)
                    details['name_edit_distance'] = name_similarity
            
            if score >= self.composite_threshold:
                if score >= self.exact_threshold:
                    match_type = 'exact'
                elif score >= self.phonetic_threshold:
                    match_type = 'phonetic'
                elif score >= self.partial_id_threshold:
                    match_type = 'partial_id'
                else:
                    match_type = 'composite'
                
                matches.append(DuplicateMatch(
                    patient_id=card.get('patient_id', ''),
                    patient_name=patient_name,
                    id_card=id_card,
                    match_type=match_type,
                    similarity_score=score,
                    existing_card_id=card.get('card_id', ''),
                    existing_patient_name=card.get('patient_name', ''),
                    existing_id_card=card.get('id_card', ''),
                    match_details=details,
                    report_date=card.get('report_date'),
                ))
        
        # 按相似度降序排列
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches
    
    def _load_existing_cards(self) -> List[Dict[str, Any]]:
        """从数据库加载已有报告卡"""
        if not self.db_manager:
            return []
        try:
            # 查询最近1年的已上报卡片（避免历史过久数据）
            rows = self.db_manager.execute("""
                SELECT card_id, patient_id, patient_name, id_card, 
                       birthday, gender, report_date
                FROM card_status cs
                LEFT JOIN review_workflow rw ON cs.card_id = rw.card_id
                WHERE cs.current_status IN ('approved', 'report_success', 'reported')
                ORDER BY rw.created_at DESC
                LIMIT 5000
            """)
            cards = []
            for row in rows:
                if isinstance(row, dict):
                    cards.append(row)
                elif isinstance(row, (list, tuple)) and len(row) >= 6:
                    cards.append({
                        'card_id': row[0],
                        'patient_id': row[1],
                        'patient_name': row[2],
                        'id_card': row[3],
                        'birthday': row[4],
                        'gender': row[5],
                        'report_date': row[6] if len(row) > 6 else None,
                    })
            return cards
        except Exception as e:
            LOGGER.warning("加载已有卡片失败: %s", e)
            return []
    
    def _normalize_name(self, name: str) -> str:
        """规范化姓名：去除空格、特殊字符"""
        if not name:
            return ""
        # 去除空格和常见特殊字符
        name = re.sub(r'[\s·•・]', '', name)
        return name.strip()
    
    def _normalize_id_card(self, id_card: str) -> str:
        """规范化身份证号：去除空格、转大写"""
        if not id_card:
            return ""
        return re.sub(r'\s', '', id_card).strip().upper()
    
    def _extract_birthday_from_id(self, id_card: str) -> str:
        """从身份证号提取出生日期 (YYYY-MM-DD)"""
        if not id_card or len(id_card) < 14:
            return ""
        try:
            year = id_card[6:10]
            month = id_card[10:12]
            day = id_card[12:14]
            if year.isdigit() and month.isdigit() and day.isdigit():
                return f"{year}-{month}-{day}"
        except Exception as e:
            LOGGER.debug("从身份证号提取出生日期失败: %s", e)
        return ""
    
    def _name_to_pinyin(self, name: str) -> str:
        """姓名转拼音（简化版，使用常用字映射）"""
        if not name:
            return ""
        pinyin_parts = []
        for char in name:
            py = _PINYIN_INITIALS.get(char, char)
            pinyin_parts.append(py)
        return ''.join(pinyin_parts)
    
    def _is_phonetic_match(self, name1: str, name2: str) -> bool:
        """判断两个姓名是否同音"""
        if not name1 or not name2:
            return False
        # 长度不同则不匹配
        if len(name1) != len(name2):
            return False
        py1 = self._name_to_pinyin(name1)
        py2 = self._name_to_pinyin(name2)
        return py1 == py2 and py1 != name1 and py2 != name2
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """计算两个姓名的相似度（0-1），使用编辑距离"""
        if not name1 or not name2:
            return 0.0
        if name1 == name2:
            return 1.0
        # Levenshtein距离
        m, n = len(name1), len(name2)
        if m == 0:
            return 0.0
        if n == 0:
            return 0.0
        
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if name1[i-1] == name2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # 删除
                    dp[i][j-1] + 1,      # 插入
                    dp[i-1][j-1] + cost  # 替换
                )
        
        max_len = max(m, n)
        return 1.0 - dp[m][n] / max_len


class CDCQuarantineProtocol:
    """疾控查重接口协议（适配器模式基类）
    
    不同省份疾控中心接口规范不同，通过继承此类实现适配器。
    """
    
    def check_duplicate_remote(self, 
                                patient_name: str,
                                id_card: str = "",
                                birthday: str = "",
                                gender: str = "") -> List[Dict[str, Any]]:
        """远程调用疾控查重接口
        
        返回：
            匹配到的在治病例列表，格式同DuplicateMatch
        """
        raise NotImplementedError("子类必须实现check_duplicate_remote方法")


class NullCDCQuarantine(CDCQuarantineProtocol):
    """空疾控查重适配器（不实际调用远程接口）"""
    
    def check_duplicate_remote(self, **kwargs) -> List[Dict[str, Any]]:
        return []


def check_duplicate_before_submit(patient_name: str,
                                  id_card: str = "",
                                  birthday: str = "",
                                  gender: str = "",
                                  db_manager=None,
                                  cdc_adapter: CDCQuarantineProtocol = None) -> Dict[str, Any]:
    """提交前查重便捷函数
    
    参数：
        patient_name: 患者姓名
        id_card: 身份证号
        birthday: 出生日期
        gender: 性别
        db_manager: 数据库管理器（本地查重）
        cdc_adapter: 疾控查重适配器（远程查重）
        
    返回：
        {
            'has_duplicate': bool,
            'has_confirmed': bool,
            'local_matches': List[DuplicateMatch],
            'remote_matches': List[...],
            'suggestion': str  # 'new' / 'correction' / 'block'
        }
    """
    local_checker = DuplicateChecker(db_manager)
    local_matches = local_checker.check_duplicate(patient_name, id_card, birthday, gender)
    
    remote_matches = []
    if cdc_adapter:
        try:
            remote_matches = cdc_adapter.check_duplicate_remote(
                patient_name, id_card, birthday, gender
            )
        except Exception as e:
            LOGGER.warning("疾控远程查重失败: %s", e)
    
    all_matches = list(local_matches) + list(remote_matches)

    def _similarity_score(m):
        """从 DuplicateMatch 对象或同构字典中提取相似度分数"""
        if isinstance(m, DuplicateMatch):
            return m.similarity_score
        if isinstance(m, dict):
            try:
                return float(m.get("similarity_score", 0) or 0)
            except (TypeError, ValueError):
                return 0
        return 0

    # confirmed/suspected 需同时考虑本地与远程匹配，否则与 has_duplicate 矛盾
    confirmed = any(_similarity_score(m) >= 90 for m in all_matches)
    suspected = any(70 <= _similarity_score(m) < 90 for m in all_matches)
    
    suggestion = 'new'
    if confirmed:
        suggestion = 'correction'  # 建议订正而非新增
    elif suspected:
        suggestion = 'review'      # 建议人工复核
    
    return {
        'has_duplicate': len(local_matches) > 0 or len(remote_matches) > 0,
        'has_confirmed': confirmed,
        'has_suspected': suspected,
        'local_matches': local_matches,
        'remote_matches': remote_matches,
        'suggestion': suggestion,
    }
