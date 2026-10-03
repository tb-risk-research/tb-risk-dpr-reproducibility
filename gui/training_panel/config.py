#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 训练面板 - 训练参数配置（Section VII）

提供统一的训练参数配置 dataclass 和对话框，替代分散的 askyesno 弹窗。

设计要点：
  - TrainingConfig 数据类集中定义所有可调参数（ML + GNN 共用）
  - TrainingConfigDialog 从数据类自动生成表单字段
  - 每个参数显示当前值、推荐范围和简短说明
  - 参数验证在对话框确认时执行，超出范围的值立即提示并阻止关闭
  - 新增参数时只需在数据类中添加字段，对话框自动适配
"""

import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# 训练参数数据类（单一真相源）
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """统一的训练参数配置（ML + GNN 共用）

    新增参数时只需在此添加字段和默认值，对话框自动适配。
    """
    # ======== 通用参数 ========
    n_samples: int = 2000
    """合成数据样本数（ML/GNN 训练用）"""

    random_state: int = 42
    """随机种子，确保可复现"""

    # ======== ML 训练参数 ========
    ml_enable_hyperopt: bool = False
    """是否启用 ML 超参数调优（GridSearchCV）"""

    ml_cv_folds: int = 5
    """交叉验证折数（仅启用超参调优时生效）"""

    # ======== GNN 训练参数 ========
    gnn_n_epochs: int = 50
    """GNN 训练轮次"""

    gnn_learning_rate: float = 0.001
    """GNN 学习率"""

    gnn_enable_hyperopt: bool = False
    """是否启用 GNN 超参数调优（搜索 hidden_dim/num_layers/lr）"""

    gnn_use_focal_loss: bool = False
    """是否使用 Focal Loss 处理类别不平衡"""

    gnn_two_phase: bool = True
    """是否启用两阶段训练（先结构后时序）"""

    # ======== 早停机制 ========
    early_stopping_enabled: bool = True
    """是否启用早停（基于验证集损失）"""

    early_stopping_patience: int = 10
    """早停耐心值：连续 N 轮无改善则停止"""

    # ======== 参数范围约束（用于对话框验证） ========
    # 不参与序列化，仅用于验证
    _RANGES = {
        'n_samples': (100, 100000),
        'random_state': (0, 2**31 - 1),
        'ml_cv_folds': (2, 20),
        'gnn_n_epochs': (1, 1000),
        'gnn_learning_rate': (1e-6, 1.0),
        'early_stopping_patience': (1, 200),
    }

    def validate(self):
        """验证参数是否在合法范围内

        Returns:
            (is_valid, error_message)
        """
        ranges = self._RANGES
        for field_name, (lo, hi) in ranges.items():
            val = getattr(self, field_name)
            if not isinstance(val, (int, float)):
                return False, f"参数 {field_name} 不是数值类型: {type(val)}"
            if val < lo or val > hi:
                return False, f"参数 {field_name}={val} 超出范围 [{lo}, {hi}]"
        return True, ''

    def to_dict(self):
        """序列化为字典（排除私有字段）"""
        return {k: v for k, v in asdict(self).items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, data):
        """从字典反序列化（忽略未知字段）"""
        if not isinstance(data, dict):
            return cls()
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()
                        if not f.name.startswith('_')}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        try:
            return cls(**filtered)
        except TypeError:
            return cls()


# ---------------------------------------------------------------------------
# 参数字段元数据（用于对话框自动生成表单）
# ---------------------------------------------------------------------------

# 每个字段的显示标签、说明、控件类型
FIELD_METADATA = {
    'n_samples': {
        'label': '合成数据样本数',
        'desc': 'ML/GNN 训练用的合成数据量（100-100000）',
        'type': 'int',
    },
    'random_state': {
        'label': '随机种子',
        'desc': '确保训练结果可复现（0-2147483647）',
        'type': 'int',
    },
    'ml_enable_hyperopt': {
        'label': 'ML 超参调优',
        'desc': '启用 GridSearchCV 搜索最优超参数（训练时间增加）',
        'type': 'bool',
    },
    'ml_cv_folds': {
        'label': '交叉验证折数',
        'desc': '超参调优时的 K 折交叉验证数（2-20）',
        'type': 'int',
    },
    'gnn_n_epochs': {
        'label': 'GNN 训练轮次',
        'desc': 'GNN 训练的 epoch 数（1-1000）',
        'type': 'int',
    },
    'gnn_learning_rate': {
        'label': 'GNN 学习率',
        'desc': '优化器学习率（0.000001-1.0）',
        'type': 'float',
    },
    'gnn_enable_hyperopt': {
        'label': 'GNN 超参调优',
        'desc': '搜索 hidden_dim/num_layers/lr（训练时间显著增加）',
        'type': 'bool',
    },
    'gnn_use_focal_loss': {
        'label': 'Focal Loss',
        'desc': '使用 Focal Loss 处理类别不平衡',
        'type': 'bool',
    },
    'gnn_two_phase': {
        'label': '两阶段训练',
        'desc': '先训练结构特征再训练时序特征',
        'type': 'bool',
    },
    'early_stopping_enabled': {
        'label': '启用早停',
        'desc': '基于验证集损失，连续耐心值轮次无改善则停止',
        'type': 'bool',
    },
    'early_stopping_patience': {
        'label': '早停耐心值',
        'desc': '连续 N 轮无改善后停止（1-200）',
        'type': 'int',
    },
}

# 参数分组（用于对话框分节显示）
FIELD_GROUPS = [
    ('通用参数', ['n_samples', 'random_state']),
    ('ML 训练', ['ml_enable_hyperopt', 'ml_cv_folds']),
    ('GNN 训练', ['gnn_n_epochs', 'gnn_learning_rate', 'gnn_enable_hyperopt',
                  'gnn_use_focal_loss', 'gnn_two_phase']),
    ('早停机制', ['early_stopping_enabled', 'early_stopping_patience']),
]


# ---------------------------------------------------------------------------
# 训练参数配置对话框
# ---------------------------------------------------------------------------

class TrainingConfigDialog:
    """统一的训练参数配置对话框

    从 TrainingConfig 数据类自动生成表单字段，每个参数显示
    当前值、推荐范围和简短说明。确认时执行参数验证。

    用法：
        config = TrainingConfig()
        dialog = TrainingConfigDialog(parent, config)
        if dialog.result is not None:
            config = dialog.result  # 用户确认后的新配置
    """

    def __init__(self, parent, config: Optional[TrainingConfig] = None,
                 title: str = '训练参数配置'):
        self.parent = parent
        self.config = config if config is not None else TrainingConfig()
        self.result: Optional[TrainingConfig] = None
        self._vars = {}  # field_name -> tk.Variable

        self._top = tk.Toplevel(parent)
        self._top.title(title)
        self._top.transient(parent)
        self._top.grab_set()
        # 优先级十一：允许调整大小 + 设置最小尺寸（高 DPI/小屏幕友好）
        self._top.resizable(True, True)
        self._top.minsize(500, 600)

        self._build_ui()
        self._center_window()

        # 等待窗口关闭
        self._top.wait_window()

    def _build_ui(self):
        """构建对话框 UI"""
        main_frame = ttk.Frame(self._top, padding=15)
        main_frame.pack(fill='both', expand=True)

        row = 0
        for group_name, field_names in FIELD_GROUPS:
            # 分组标题
            ttk.Label(main_frame, text=group_name,
                      font=('Arial', 11, 'bold')).grid(
                row=row, column=0, columnspan=3, sticky='w', pady=(10, 2))
            row += 1

            # 分隔线
            ttk.Separator(main_frame, orient='horizontal').grid(
                row=row, column=0, columnspan=3, sticky='ew', pady=2)
            row += 1

            for fname in field_names:
                meta = FIELD_METADATA.get(fname, {})
                label = meta.get('label', fname)
                desc = meta.get('desc', '')
                ftype = meta.get('type', 'str')

                # 标签
                ttk.Label(main_frame, text=label + ':').grid(
                    row=row, column=0, sticky='w', padx=(0, 10), pady=3)

                # 控件
                current_val = getattr(self.config, fname)
                if ftype == 'bool':
                    var = tk.BooleanVar(value=bool(current_val))
                    widget = ttk.Checkbutton(main_frame, variable=var)
                elif ftype == 'int':
                    var = tk.IntVar(value=int(current_val))
                    widget = ttk.Entry(main_frame, textvariable=var, width=15)
                elif ftype == 'float':
                    var = tk.DoubleVar(value=float(current_val))
                    widget = ttk.Entry(main_frame, textvariable=var, width=15)
                else:
                    var = tk.StringVar(value=str(current_val))
                    widget = ttk.Entry(main_frame, textvariable=var, width=15)

                widget.grid(row=row, column=1, sticky='w', padx=2, pady=3)
                self._vars[fname] = var

                # 说明文字
                ttk.Label(main_frame, text=desc,
                          font=('Arial', 8), foreground='gray').grid(
                    row=row, column=2, sticky='w', padx=(10, 0), pady=3)
                row += 1

        # 按钮区域
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=(15, 0))

        ttk.Button(btn_frame, text='确定', command=self._on_ok).pack(
            side='left', padx=5)
        ttk.Button(btn_frame, text='取消', command=self._on_cancel).pack(
            side='left', padx=5)
        ttk.Button(btn_frame, text='恢复默认', command=self._on_reset).pack(
            side='left', padx=5)

    def _center_window(self):
        """将对话框居中于父窗口"""
        self._top.update_idletasks()
        w = self._top.winfo_width()
        h = self._top.winfo_height()
        parent = self.parent
        try:
            px = parent.winfo_rootx() + parent.winfo_width() // 2 - w // 2
            py = parent.winfo_rooty() + parent.winfo_height() // 2 - h // 2
        except Exception:
            px = py = 100
        self._top.geometry(f'+{max(px, 0)}+{max(py, 0)}')

    def _collect_config(self):
        """从 UI 变量收集配置，返回 TrainingConfig 或 None（验证失败时）"""
        kwargs = {}
        for fname, var in self._vars.items():
            try:
                val = var.get()
            except (tk.TclError, ValueError):
                messagebox.showerror(
                    '参数错误', f'字段 "{FIELD_METADATA.get(fname, {}).get("label", fname)}" '
                              f'的值无效，请输入有效数值。', parent=self._top)
                return None
            kwargs[fname] = val

        config = TrainingConfig(**kwargs)
        is_valid, err_msg = config.validate()
        if not is_valid:
            messagebox.showerror('参数范围错误', err_msg, parent=self._top)
            return None
        return config

    def _on_ok(self):
        """确定按钮：验证并保存配置"""
        config = self._collect_config()
        if config is not None:
            self.result = config
            self._top.destroy()

    def _on_cancel(self):
        """取消按钮：不保存配置"""
        self.result = None
        self._top.destroy()

    def _on_reset(self):
        """恢复默认按钮：重置所有字段为默认值"""
        default = TrainingConfig()
        for fname, var in self._vars.items():
            val = getattr(default, fname)
            try:
                var.set(val)
            except (tk.TclError, ValueError):
                pass


__all__ = ['TrainingConfig', 'TrainingConfigDialog', 'FIELD_METADATA', 'FIELD_GROUPS']
