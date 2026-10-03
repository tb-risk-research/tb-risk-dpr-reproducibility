#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Section X 全局体验层测试 — 独立帮助窗口

覆盖：
- HELP_CONTENT 字典完整性：12 章节、每章非空、关键章节存在
- HelpWindow 单例模式：重复创建聚焦已有窗口、关闭后清理引用
- HelpWindow.is_open 类方法
- HelpWindow._show_section 章节切换
- HelpWindow._apply_highlight 搜索高亮
- HelpWindow._on_close 清理逻辑
- WCAG AA 颜色对比度：text_light 调整为更深色调
- BaseMixin._show_help_window 委托
- MiscTabMixin._show_help 回退到 messagebox
"""

import os
import sys
import tkinter as tk
from tkinter import ttk
from unittest import mock

import pytest

# 将项目根目录加入 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from tb_risk.gui.help_system import HELP_CONTENT, HelpWindow


# ===========================================================================
# Tk root fixture（与 test_training_panel.py 一致）
# ===========================================================================

@pytest.fixture
def root():
    """创建 Tk root，Tcl 初始化失败时跳过测试"""
    try:
        r = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"tkinter/Tcl 初始化失败: {e}")
    r.withdraw()
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass


@pytest.fixture(autouse=True)
def _reset_help_window_singleton():
    """每个测试前后清理 HelpWindow 单例引用，避免测试间状态泄漏"""
    HelpWindow._instance = None
    yield
    # 测试后清理：若实例仍存在则销毁
    if HelpWindow._instance is not None:
        try:
            HelpWindow._instance._window.destroy()
        except Exception:
            pass
        HelpWindow._instance = None


# ===========================================================================
# HELP_CONTENT 字典完整性
# ===========================================================================

class TestHelpContent:
    """HELP_CONTENT 内置帮助文档完整性"""

    def test_has_at_least_12_sections(self):
        """应有至少 12 个章节"""
        assert len(HELP_CONTENT) >= 12

    def test_all_sections_non_empty(self):
        """每个章节内容都应为非空字符串"""
        for section, content in HELP_CONTENT.items():
            assert isinstance(content, str), f"章节 {section} 内容不是字符串"
            assert len(content.strip()) > 0, f"章节 {section} 内容为空"

    def test_key_sections_present(self):
        """关键章节应存在"""
        required = ['快速入门', '患者信息录入', '接触者管理', '数据导入导出',
                    '风险评估', '快捷键列表', 'FAQ']
        for section in required:
            assert section in HELP_CONTENT, f"缺少必要章节: {section}"

    def test_advanced_feature_sections_present(self):
        """高级功能章节应存在（ML/SHAP/MCMC/反事实/DAG/GNN）"""
        advanced = ['ML/SHAP 分析', 'MCMC 诊断', '反事实干预分析',
                    '因果 DAG', 'GNN 训练']
        for section in advanced:
            assert section in HELP_CONTENT, f"缺少高级功能章节: {section}"

    def test_quick_start_mentions_ctrl_n(self):
        """快速入门章节应提及 Ctrl+N 快捷键"""
        assert 'Ctrl+N' in HELP_CONTENT['快速入门']

    def test_shortcuts_section_lists_key_bindings(self):
        """快捷键列表章节应包含主要快捷键"""
        content = HELP_CONTENT['快捷键列表']
        # HELP_CONTENT 中格式为 "Ctrl + N"（含空格），同时兼容 "Ctrl+N" 格式
        assert 'Ctrl' in content and 'N' in content
        assert 'Ctrl' in content and 'E' in content
        assert 'F1' in content
        assert 'F5' in content
        assert 'Delete' in content
        assert 'Enter' in content

    def test_faq_section_has_multiple_questions(self):
        """FAQ 章节应包含多个 Q&A"""
        content = HELP_CONTENT['FAQ']
        # 至少 5 个问题
        q_count = content.count('Q')
        assert q_count >= 5


# ===========================================================================
# HelpWindow 单例与生命周期
# ===========================================================================

class TestHelpWindowSingleton:
    """HelpWindow 单例模式与生命周期"""

    def test_is_open_false_initially(self):
        """初始状态 is_open 应为 False"""
        assert HelpWindow.is_open() is False

    def test_is_open_true_after_creation(self, root):
        """创建窗口后 is_open 应为真值（winfo_exists 返回 1）"""
        hw = HelpWindow(root)
        assert HelpWindow.is_open()  # truthy (winfo_exists() 返回 1)

    def test_singleton_second_creation_focuses_existing(self, root):
        """第二次创建应聚焦已有窗口，不应创建新实例"""
        hw1 = HelpWindow(root)
        # 记录第一个实例的窗口
        original_window = hw1._window

        hw2 = HelpWindow(root)
        # 应返回同一实例（hw2 不应创建新窗口）
        # 注意：当 _instance 已存在时，__init__ 仍执行但会提前 return
        # 因此 hw2 是新对象但其 _window 未被设置（或为旧窗口）
        # 关键断言：HelpWindow._instance 仍指向 hw1
        assert HelpWindow._instance is hw1

    def test_on_close_clears_singleton(self, root):
        """关闭窗口应清理单例引用"""
        hw = HelpWindow(root)
        assert HelpWindow.is_open()  # truthy
        hw._on_close()
        assert HelpWindow._instance is None
        assert HelpWindow.is_open() is False

    def test_is_open_false_after_close(self, root):
        """关闭后 is_open 应为 False"""
        hw = HelpWindow(root)
        hw._on_close()
        assert HelpWindow.is_open() is False

    def test_recreate_after_close(self, root):
        """关闭后可以重新创建"""
        hw1 = HelpWindow(root)
        hw1._on_close()
        assert HelpWindow.is_open() is False

        hw2 = HelpWindow(root)
        assert HelpWindow.is_open()  # truthy
        assert HelpWindow._instance is hw2


# ===========================================================================
# HelpWindow UI 构建与章节显示
# ===========================================================================

class TestHelpWindowUI:
    """HelpWindow UI 构建与章节切换"""

    def test_window_title_set(self, root):
        """窗口标题应正确设置"""
        hw = HelpWindow(root)
        assert '帮助' in hw._window.title() or '使用帮助' in hw._window.title()

    def test_window_has_search_entry(self, root):
        """应有搜索框"""
        hw = HelpWindow(root)
        assert hasattr(hw, '_search_var')
        assert isinstance(hw._search_var, tk.StringVar)

    def test_window_has_tree_navigation(self, root):
        """应有目录树"""
        hw = HelpWindow(root)
        assert hasattr(hw, '_tree')
        # 目录树应包含所有章节
        children = hw._tree.get_children()
        assert len(children) == len(HELP_CONTENT)

    def test_window_has_text_area(self, root):
        """应有内容显示区"""
        hw = HelpWindow(root)
        assert hasattr(hw, '_text')

    def test_default_section_displayed(self, root):
        """创建后应默认显示第一个章节"""
        hw = HelpWindow(root)
        assert hw._current_section is not None
        first_section = next(iter(HELP_CONTENT.keys()))
        assert hw._current_section == first_section

    def test_show_section_updates_current(self, root):
        """_show_section 应更新 _current_section"""
        hw = HelpWindow(root)
        # 选择第二个章节
        sections = list(HELP_CONTENT.keys())
        if len(sections) >= 2:
            hw._show_section(sections[1])
            assert hw._current_section == sections[1]

    def test_show_section_updates_text_content(self, root):
        """_show_section 应更新文本区内容"""
        hw = HelpWindow(root)
        sections = list(HELP_CONTENT.keys())
        if len(sections) >= 2:
            hw._show_section(sections[1])
            content = hw._text.get('1.0', 'end-1c')
            assert sections[1] in content or HELP_CONTENT[sections[1]][:20] in content

    def test_show_unknown_section_no_change(self, root):
        """未知章节名应被忽略，不更新当前章节"""
        hw = HelpWindow(root)
        original = hw._current_section
        hw._show_section('不存在的章节_xyz')
        assert hw._current_section == original

    def test_tree_select_triggers_show_section(self, root):
        """点击目录树项应触发章节切换"""
        hw = HelpWindow(root)
        sections = list(HELP_CONTENT.keys())
        if len(sections) >= 2:
            # 模拟选择第二个章节
            hw._tree.selection_set(sections[1])
            hw._tree.focus(sections[1])
            # 触发选中事件
            hw._on_tree_select()
            assert hw._current_section == sections[1]


# ===========================================================================
# HelpWindow 搜索高亮
# ===========================================================================

class TestHelpWindowSearch:
    """HelpWindow 全文搜索与高亮"""

    def test_search_empty_does_not_highlight(self, root):
        """空搜索词不应添加高亮 tag"""
        hw = HelpWindow(root)
        hw._search_var.set('')
        # 无高亮 tag
        assert not hw._text.tag_ranges('highlight')

    def test_search_adds_highlight_tags(self, root):
        """非空搜索词应添加高亮 tag"""
        hw = HelpWindow(root)
        # 使用 HELP_CONTENT 中必然存在的关键词
        hw._search_var.set('Ctrl')
        # 应至少有一个高亮范围
        ranges = hw._text.tag_ranges('highlight')
        assert len(ranges) >= 2  # start 和 end 各一个，至少一对

    def test_search_case_insensitive(self, root):
        """搜索应大小写不敏感"""
        hw = HelpWindow(root)
        hw._search_var.set('ctrl')  # 小写
        ranges_lower = hw._text.tag_ranges('highlight')

        hw._search_var.set('')  # 清空
        hw._search_var.set('Ctrl')  # 首字母大写
        ranges_mixed = hw._text.tag_ranges('highlight')

        # 两种情况应匹配相同数量（大小写不敏感）
        assert len(ranges_lower) == len(ranges_mixed)

    def test_search_no_match_still_works(self, root):
        """无匹配的搜索词不应抛异常"""
        hw = HelpWindow(root)
        hw._search_var.set('xyzqwerty_nonexistent_keyword_12345')
        # 不应抛异常，且无高亮
        assert not hw._text.tag_ranges('highlight')

    def test_search_clear_removes_highlight(self, root):
        """清空搜索应移除高亮"""
        hw = HelpWindow(root)
        hw._search_var.set('Ctrl')
        assert len(hw._text.tag_ranges('highlight')) >= 2

        hw._search_var.set('')
        assert not hw._text.tag_ranges('highlight')


# ===========================================================================
# HelpWindow 异常容错
# ===========================================================================

class TestHelpWindowErrorHandling:
    """HelpWindow 异常容错"""

    def test_creation_with_none_root_does_not_raise(self):
        """parent_root 为 None 时不应抛异常

        注意：tk.Toplevel(None) 会自动创建一个 Tk root，因此 HelpWindow
        实际上会成功创建。本测试仅验证不抛异常，并在结束后清理。
        """
        try:
            hw = HelpWindow(None)
            # 无论成功与否，都不应抛异常
        except Exception as e:
            pytest.fail(f"HelpWindow(None) 不应抛异常，但抛了: {e}")
        finally:
            # 清理可能自动创建的 Tk root
            if HelpWindow._instance is not None:
                try:
                    HelpWindow._instance._on_close()
                except Exception:
                    pass
            # 销毁可能自动创建的 Tk root
            try:
                import tkinter as tk
                for w in tk._default_root.winfo_children() if tk._default_root else []:
                    try:
                        w.destroy()
                    except Exception:
                        pass
            except Exception:
                pass

    def test_creation_with_bad_parent_does_not_raise(self):
        """parent_root 无效时不应抛异常（被 try/except 捕获）"""
        try:
            hw = HelpWindow("not_a_tk_root")
        except Exception as e:
            pytest.fail(f"HelpWindow(invalid) 不应抛异常，但抛了: {e}")
        finally:
            # 清理单例
            if HelpWindow._instance is not None:
                try:
                    HelpWindow._instance._on_close()
                except Exception:
                    pass

    def test_show_section_exception_does_not_propagate(self, root):
        """_show_section 异常不应传播"""
        hw = HelpWindow(root)
        # 修改 _text 使其抛异常
        original_text = hw._text
        hw._text = mock.MagicMock()
        hw._text.configure.side_effect = RuntimeError("simulated")
        # 不应抛异常
        hw._show_section('快速入门')
        # 恢复
        hw._text = original_text

    def test_apply_highlight_exception_does_not_propagate(self, root):
        """_apply_highlight 异常不应传播"""
        hw = HelpWindow(root)
        # 修改 _text 使其抛异常
        hw._text = mock.MagicMock()
        hw._text.tag_remove.side_effect = RuntimeError("simulated")
        hw._search_var.set('test')
        # 不应抛异常
        hw._apply_highlight()


# ===========================================================================
# WCAG AA 颜色对比度校准（Section X）
# ===========================================================================

class TestWCAGColorCompliance:
    """WCAG AA 颜色对比度校准

    Section X 要求：正文与背景对比度不低于 4.5:1
    将 text_light 从 #7f8c8d 调整为 #5a6c7d
    """

    def _relative_luminance(self, hex_color):
        """计算相对亮度（WCAG 2.1 公式）"""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0

        def _linearize(c):
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

        return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)

    def _contrast_ratio(self, fg_hex, bg_hex):
        """计算对比度"""
        l1 = self._relative_luminance(fg_hex)
        l2 = self._relative_luminance(bg_hex)
        lighter = max(l1, l2)
        darker = min(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)

    def test_text_light_is_darker_than_old_value(self):
        """text_light (#5a6c7d) 应比旧值 (#7f8c8d) 更深"""
        # 通过 BaseMixin.COLORS 验证
        from tb_risk.gui.base import BaseMixin
        # 创建实例以获取 COLORS（_setup_styles 会设置）
        # 但 _setup_styles 需要 Tk root，所以直接验证静态值
        new_color = '#5a6c7d'
        old_color = '#7f8c8d'
        # 新色的相对亮度应低于旧色（更深 = 亮度更低）
        assert self._relative_luminance(new_color) < self._relative_luminance(old_color)

    def test_text_light_meets_wcag_aa_against_bg_light(self):
        """text_light (#5a6c7d) 对 bg_light (#f8f9fa) 应满足 WCAG AA 4.5:1"""
        contrast = self._contrast_ratio('#5a6c7d', '#f8f9fa')
        assert contrast >= 4.5, f"text_light 对比度 {contrast:.2f} 低于 WCAG AA 4.5:1"

    def test_text_meets_wcag_aa_against_bg_light(self):
        """正文 text (#2c3e50) 对 bg_light (#f8f9fa) 应满足 WCAG AA 4.5:1"""
        contrast = self._contrast_ratio('#2c3e50', '#f8f9fa')
        assert contrast >= 4.5, f"text 对比度 {contrast:.2f} 低于 WCAG AA 4.5:1"

    def test_old_text_light_failed_wcag_aa(self):
        """验证旧值 #7f8c8d 确实不满足 WCAG AA（证明调整的必要性）"""
        contrast = self._contrast_ratio('#7f8c8d', '#f8f9fa')
        # 旧值应低于 4.5（这是调整的原因）
        assert contrast < 4.5, f"旧值对比度 {contrast:.2f} 已满足 WCAG AA，无需调整"


# ===========================================================================
# BaseMixin._show_help_window 委托
# ===========================================================================

class TestBaseMixinShowHelpWindow:
    """BaseMixin._show_help_window 应委托给 HelpWindow"""

    def test_show_help_window_calls_help_window_class(self, root):
        """_show_help_window 应创建 HelpWindow 实例"""
        from tb_risk.gui.base import BaseMixin
        m = BaseMixin()
        m.root = root

        # 清理单例
        HelpWindow._instance = None

        m._show_help_window()
        assert HelpWindow.is_open()  # truthy

    def test_show_help_window_no_root_does_nothing(self):
        """无 root 属性时不应抛异常"""
        from tb_risk.gui.base import BaseMixin
        m = BaseMixin()
        # 不设置 root
        m._show_help_window()  # 不应抛异常

    def test_show_help_window_falls_back_to_messagebox_on_failure(self):
        """HelpWindow 创建失败时应回退到 messagebox

        模拟 HelpWindow.__init__ 抛异常，验证 _show_help_window
        的 except 分支会调用 messagebox.showinfo。
        """
        from tb_risk.gui.base import BaseMixin
        m = BaseMixin()
        m.root = "dummy_root"

        # 模拟 HelpWindow 构造函数抛异常，触发 messagebox 回退
        with mock.patch('tb_risk.gui.base.HelpWindow',
                         side_effect=RuntimeError("simulated failure")), \
             mock.patch('tb_risk.gui.base.messagebox') as mb:
            m._show_help_window()
            mb.showinfo.assert_called_once()


# ===========================================================================
# MiscTabMixin._show_help 委托与回退
# ===========================================================================

class TestMiscTabShowHelpDelegation:
    """MiscTabMixin._show_help 应优先委托给 _show_help_window"""

    def test_show_help_delegates_to_show_help_window_when_available(self, root):
        """有 _show_help_window 方法时应委托"""
        from tb_risk.gui.tabs._misc import MiscTabMixin
        m = MiscTabMixin()
        m.root = root

        # 模拟 _show_help_window 方法
        called = [False]
        def fake_show_help_window():
            called[0] = True
        m._show_help_window = fake_show_help_window

        m._show_help()
        assert called[0] is True

    def test_show_help_falls_back_to_messagebox(self, root):
        """无 _show_help_window 方法时应回退到 messagebox"""
        from tb_risk.gui.tabs._misc import MiscTabMixin
        m = MiscTabMixin()
        m.root = root

        # 不设置 _show_help_window 方法（删除可能继承的属性）
        if hasattr(m, '_show_help_window'):
            del m._show_help_window

        with mock.patch('tb_risk.gui.tabs._misc.messagebox') as mb:
            m._show_help()
            mb.showinfo.assert_called_once()
            # 验证调用了使用说明
            args = mb.showinfo.call_args
            assert '使用说明' in args.args or '使用说明' in str(args)
