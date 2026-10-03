#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python语法检查脚本 - 编译检查所有.py文件

用法:
    python tools/dev/syntax_check.py

从项目根目录或任意位置运行均可，自动计算项目根路径。
自动跳过: .venv, __pycache__, .pytest_cache, .hypothesis, tests, tools 目录。
"""
import os
import py_compile
import sys

# 自动计算tb_risk包根目录（tools/dev/ -> ../../.. 即tb_risk/）
# __file__: tb_risk/tools/dev/syntax_check.py -> 3层dirname到达tb_risk/
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

_ERRORS = []
_SKIP_PATHS = ('.venv', '__pycache__', '.pytest_cache', '.hypothesis', 'tools', '.git', 'catboost_info')


def main():
    for dirpath, dirnames, filenames in os.walk(_ROOT_DIR):
        # Skip excluded directories
        if any(skip in dirpath for skip in _SKIP_PATHS):
            dirnames[:] = []
            continue

        for filename in filenames:
            if filename.endswith('.py'):
                filepath = os.path.join(dirpath, filename)
                try:
                    py_compile.compile(filepath, doraise=True)
                except py_compile.PyCompileError as e:
                    _ERRORS.append(str(e))

    if _ERRORS:
        print(f"发现 {len(_ERRORS)} 个语法错误:")
        for err in _ERRORS:
            print(f"  {err}")
        sys.exit(1)
    else:
        print(f"所有Python文件语法检查通过（根目录: {_ROOT_DIR}）。")


if __name__ == '__main__':
    main()
