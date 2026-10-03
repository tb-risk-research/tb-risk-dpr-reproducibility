#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 启动入口脚本（由启动.bat调用，避免 -c 参数的引号转义问题）

改进：启动时强制清理项目自身 __pycache__，确保加载最新源码而非旧缓存。
项目历史教训：Python __pycache__ 必须清除，否则加载的是缓存旧代码，
导致代码修改后界面表现"始终没有变化"。
"""
import sys
import os
import shutil

# 确保项目根目录在 sys.path 中（桌面目录）
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


def _clear_bytecode_cache(root):
    """递归清除项目源码目录下的 __pycache__，跳过第三方 .venv。

    在 import 任何项目模块之前调用，保证加载的是最新 .py 源码。
    """
    if not root or not os.path.isdir(root):
        return
    for dirpath, dirnames, _filenames in os.walk(root):
        # 跳过 .venv（第三方库缓存，避免拖慢启动）
        dirnames[:] = [d for d in dirnames if d != '.venv']
        if os.path.basename(dirpath) == '__pycache__':
            try:
                shutil.rmtree(dirpath, ignore_errors=True)
            except Exception:
                pass


def main():
    # 改进：import 项目模块前先清除自身缓存，确保加载最新代码
    _clear_bytecode_cache(_script_dir)

    from tb_risk.assessment import TB_Risk_Assessment
    app = TB_Risk_Assessment()
    app.init_gui()
    app.root.mainloop()


if __name__ == '__main__':
    main()