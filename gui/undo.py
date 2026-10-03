#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""撤销/重做基础设施 — 命令模式

将用户操作（添加/删除接触者、编辑字段、导入数据、应用场景）封装为 Command
对象，维护 UndoStack 和 RedoStack。Ctrl+Z 弹出 UndoStack 执行 undo()，
Ctrl+Y 弹出 RedoStack 执行 redo()。

使用方式（在 assessment.py 或 Mixin 中）：
    self.undo_manager = UndoManager(max_size=50)
    # 执行操作前压栈
    cmd = AddContactCommand(self, 'family', new_entry)
    self.undo_manager.execute(cmd)
    # 撤销
    self.undo_manager.undo()
"""

import copy
from abc import ABC, abstractmethod


class Command(ABC):
    """命令基类 — 每个子类必须实现 execute 和 undo。

    健壮性要点：
      - undo 依赖操作前的状态快照（深拷贝），而非引用
      - execute 在首次调用时执行实际操作
    """

    @abstractmethod
    def execute(self):
        """执行操作"""

    @abstractmethod
    def undo(self):
        """撤销操作，恢复到执行前的状态"""

    def redo(self):
        """重做（默认等同于 execute，子类可覆盖以优化）"""
        self.execute()

    @property
    def description(self):
        """人类可读的操作描述（用于菜单提示）"""
        return self.__class__.__name__


class AddContactCommand(Command):
    """添加接触者命令"""

    def __init__(self, target, contact_type, entry):
        """
        Args:
            target: 具有 family_entries / social_entries 列表的对象
            contact_type: 'family' 或 'social'
            entry: 要添加的接触者字典
        """
        self.target = target
        self.contact_type = contact_type
        self.entry = copy.deepcopy(entry)

    def _get_list(self):
        attr = f'{self.contact_type}_entries'
        return getattr(self.target, attr)

    def execute(self):
        self._get_list().append(copy.deepcopy(self.entry))

    def undo(self):
        lst = self._get_list()
        if lst:
            lst.pop()

    @property
    def description(self):
        return f"添加{ '家庭' if self.contact_type == 'family' else '社会'}接触者"


class DeleteContactCommand(Command):
    """删除接触者命令（保存被删除项以供撤销恢复）"""

    def __init__(self, target, contact_type, index):
        self.target = target
        self.contact_type = contact_type
        self.index = index
        self._deleted_entry = None

    def _get_list(self):
        return getattr(self.target, f'{self.contact_type}_entries')

    def execute(self):
        lst = self._get_list()
        if 0 <= self.index < len(lst):
            self._deleted_entry = copy.deepcopy(lst[self.index])
            lst.pop(self.index)

    def undo(self):
        lst = self._get_list()
        if self._deleted_entry is not None:
            lst.insert(self.index, copy.deepcopy(self._deleted_entry))

    @property
    def description(self):
        return f"删除{'家庭' if self.contact_type == 'family' else '社会'}接触者"


class EditContactCommand(Command):
    """编辑接触者命令（保存修改前后的快照）"""

    def __init__(self, target, contact_type, index, new_entry):
        self.target = target
        self.contact_type = contact_type
        self.index = index
        self.new_entry = copy.deepcopy(new_entry)
        self._old_entry = None

    def _get_list(self):
        return getattr(self.target, f'{self.contact_type}_entries')

    def execute(self):
        lst = self._get_list()
        if 0 <= self.index < len(lst):
            self._old_entry = copy.deepcopy(lst[self.index])
            lst[self.index] = copy.deepcopy(self.new_entry)

    def undo(self):
        lst = self._get_list()
        if self._old_entry is not None and 0 <= self.index < len(lst):
            lst[self.index] = copy.deepcopy(self._old_entry)

    @property
    def description(self):
        return f"编辑{'家庭' if self.contact_type == 'family' else '社会'}接触者"


class ImportDataCommand(Command):
    """批量导入数据命令（保存导入前的完整列表快照）"""

    def __init__(self, target, family_entries, social_entries):
        self.target = target
        self._new_family = copy.deepcopy(family_entries)
        self._new_social = copy.deepcopy(social_entries)
        self._old_family = None
        self._old_social = None

    def execute(self):
        self._old_family = copy.deepcopy(self.target.family_entries)
        self._old_social = copy.deepcopy(self.target.social_entries)
        self.target.family_entries = copy.deepcopy(self._new_family)
        self.target.social_entries = copy.deepcopy(self._new_social)

    def undo(self):
        if self._old_family is not None:
            self.target.family_entries = copy.deepcopy(self._old_family)
        if self._old_social is not None:
            self.target.social_entries = copy.deepcopy(self._old_social)

    @property
    def description(self):
        return "导入数据"


class UndoManager:
    """撤销/重做栈管理器

    维护 UndoStack 和 RedoStack。每次 execute 压入 UndoStack 并清空 RedoStack。
    """

    def __init__(self, max_size=50):
        self._undo_stack = []
        self._redo_stack = []
        self.max_size = max_size

    def execute(self, command):
        """执行命令并压入撤销栈"""
        command.execute()
        self._undo_stack.append(command)
        if len(self._undo_stack) > self.max_size:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self):
        """撤销最近一次操作，返回 True 表示成功"""
        if not self._undo_stack:
            return False
        command = self._undo_stack.pop()
        command.undo()
        self._redo_stack.append(command)
        return True

    def redo(self):
        """重做最近一次撤销的操作，返回 True 表示成功"""
        if not self._redo_stack:
            return False
        command = self._redo_stack.pop()
        command.redo()
        self._undo_stack.append(command)
        return True

    @property
    def can_undo(self):
        return len(self._undo_stack) > 0

    @property
    def can_redo(self):
        return len(self._redo_stack) > 0

    @property
    def next_undo_description(self):
        return self._undo_stack[-1].description if self._undo_stack else ""

    @property
    def next_redo_description(self):
        return self._redo_stack[-1].description if self._redo_stack else ""

    def clear(self):
        """清空所有历史（如新建会话时）"""
        self._undo_stack.clear()
        self._redo_stack.clear()
