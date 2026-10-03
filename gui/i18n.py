#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""国际化框架（优先级十）

提供 gettext 翻译框架，支持中文（默认）和英文两种语言。

使用方式：
    from .i18n import _, set_language, get_current_language

    label = ttk.Label(parent, text=_("患者基本信息"))
    set_language('en')  # 切换到英文

分阶段推进策略：
  - 阶段 1（当前）：建立框架 + _() 函数 + 语言切换基础设施
  - 阶段 2（后续）：提取所有 UI 字符串到 messages.po 文件
  - 阶段 3（后续）：完成英文翻译 + 编译 .mo 文件
  - 阶段 4（后续）：CLI 和 GUI 共享同一翻译文件

当前阶段 1：_() 函数默认返回原始中文字符串（identity function），
待 .mo 文件就绪后自动切换为 gettext 翻译。
"""

import gettext as _gettext_module
import os

# 当前语言代码（'zh' 为默认，'en' 为英文）
_current_language = 'zh'

# 翻译函数（默认为 identity，待 .mo 文件就绪后替换为真正的 gettext）
_translation_func = lambda s: s

# locales 目录路径（相对于项目根目录）
_LOCALES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'locales')


def _(message: str) -> str:
    """翻译函数

    包裹硬编码中文文本，待 .mo 翻译文件就绪后自动返回对应语言翻译。
    当前阶段返回原始字符串（identity function）。

    Args:
        message: 待翻译的字符串（通常为中文）

    Returns:
        翻译后的字符串；当前阶段返回原始 message
    """
    return _translation_func(message)


def set_language(lang_code: str) -> bool:
    """切换语言

    Args:
        lang_code: 语言代码（'zh' 中文，'en' 英文）

    Returns:
        True 如果切换成功（或回退到 identity），False 如果语言代码无效
    """
    global _current_language, _translation_func

    if lang_code not in ('zh', 'en'):
        return False

    _current_language = lang_code

    # 中文默认使用 identity（原始字符串即中文）
    if lang_code == 'zh':
        _translation_func = lambda s: s
        return True

    # 英文：尝试加载 .mo 文件
    try:
        translation = _gettext_module.translation(
            'messages', localedir=_LOCALES_DIR, languages=[lang_code]
        )
        _translation_func = translation.gettext
        return True
    except (FileNotFoundError, OSError):
        # .mo 文件不存在，回退到 identity（阶段 2/3 完成后可用）
        _translation_func = lambda s: s
        return True


def get_current_language() -> str:
    """获取当前语言代码"""
    return _current_language


def get_available_languages() -> list:
    """获取可用语言列表

    Returns:
        语言代码列表（始终包含 'zh'，'en' 在 .mo 文件就绪后加入）
    """
    langs = ['zh']
    en_mo = os.path.join(_LOCALES_DIR, 'en', 'LC_MESSAGES', 'messages.mo')
    if os.path.exists(en_mo):
        langs.append('en')
    return langs


__all__ = ['_', 'set_language', 'get_current_language', 'get_available_languages']
