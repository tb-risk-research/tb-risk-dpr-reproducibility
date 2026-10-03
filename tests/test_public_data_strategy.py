#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 - 公开数据验证策略（改进三-第三步）

针对 validation/public_data_strategy.py：
- 逐任务验证计划（任务 A/B 可外部验证，任务 C 过渡验证）
- 任务 C 过渡验证步骤（合成网络 + 文献校准 + 敏感性分析）
- 合作数据需求优先级（真实网络数据第一优先级）
- 策略 Markdown 文档渲染
"""
import os
import sys
import unittest

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


class TestValidationPlan(unittest.TestCase):
    """逐任务验证计划测试。"""

    def test_plan_covers_all_tasks(self):
        from tb_risk.validation import build_validation_strategy
        strategy = build_validation_strategy()
        for task in ('A', 'B', 'C'):
            self.assertIn(task, strategy['plan'])
            self.assertIn('task_name', strategy['plan'][task])

    def test_ab_can_external_validate(self):
        from tb_risk.validation import EXTERNAL_VALIDATION_PLAN
        self.assertTrue(EXTERNAL_VALIDATION_PLAN['A']['can_external_validate'])
        self.assertTrue(EXTERNAL_VALIDATION_PLAN['B']['can_external_validate'])
        self.assertTrue(EXTERNAL_VALIDATION_PLAN['A']['datasets'])
        self.assertTrue(EXTERNAL_VALIDATION_PLAN['B']['datasets'])

    def test_c_is_transition(self):
        from tb_risk.validation import EXTERNAL_VALIDATION_PLAN
        plan_c = EXTERNAL_VALIDATION_PLAN['C']
        self.assertFalse(plan_c['can_external_validate'])
        self.assertTrue(plan_c['reason'])
        self.assertIn('transition', plan_c)
        self.assertEqual(plan_c['transition']['approach'],
                         '合成网络 + 文献参数校准 + 敏感性分析')


class TestTransitionSteps(unittest.TestCase):
    """任务 C 过渡验证步骤测试。"""

    def test_steps_nonempty_and_ordered(self):
        from tb_risk.validation import transition_validation_steps
        steps = transition_validation_steps()
        self.assertGreaterEqual(len(steps), 4)
        self.assertIn('合成网络', steps[0])
        self.assertIn('敏感性分析', steps[2])


class TestCooperationNeeds(unittest.TestCase):
    """合作数据需求优先级测试。"""

    def test_top_priority_is_network_data(self):
        from tb_risk.validation import top_cooperation_data_need
        top = top_cooperation_data_need()
        self.assertEqual(top['priority'], 1)
        self.assertIn('网络', top['data'])
        self.assertEqual(top['tasks'], ['C'])

    def test_needs_sorted_by_priority(self):
        from tb_risk.validation import cooperation_data_needs
        needs = cooperation_data_needs()
        prios = [n['priority'] for n in needs]
        self.assertEqual(prios, sorted(prios))
        self.assertEqual(prios[0], 1)


class TestRenderStrategy(unittest.TestCase):
    """策略 Markdown 文档渲染测试。"""

    def test_document_contains_key_sections(self):
        from tb_risk.validation import render_strategy_document
        doc = render_strategy_document()
        self.assertIn('# 公开数据验证策略', doc)
        for task in ('A', 'B', 'C'):
            self.assertIn(f'## 任务 {task}', doc)
        self.assertIn('合作数据需求', doc)
        self.assertIn('第一优先级', doc)

    def test_document_states_c_transition(self):
        from tb_risk.validation import render_strategy_document
        doc = render_strategy_document()
        self.assertIn('过渡验证方案', doc)


if __name__ == '__main__':
    unittest.main()
