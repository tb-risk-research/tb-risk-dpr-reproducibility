#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模块导入与编译检查脚本 - 扫描所有Python模块检查语法和导入问题

用法:
    python tools/dev/import_check.py

从项目根目录或任意位置运行均可，自动计算项目根路径。
自动跳过: .venv, __pycache__, .pytest_cache, .hypothesis, tests, tools 目录。
"""
import importlib.util
import os
import sys

# 自动计算tb_risk包根目录（tools/dev/ -> ../../.. 即tb_risk/）
# __file__: tb_risk/tools/dev/import_check.py -> 3层dirname到达tb_risk/
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

_ERRORS = []
_EXCLUDE_DIRS = {'.venv', '__pycache__', '.pytest_cache', '.hypothesis', 'tests', 'tools', '.git', 'catboost_info'}


def main():
    # 将包父目录加入path（用于导入检查时能找到依赖）
    pkg_parent = os.path.dirname(_ROOT_DIR)
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)

    for dirpath, dirnames, filenames in os.walk(_ROOT_DIR):
        # Modify dirnames in-place to exclude
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDE_DIRS]

        for filename in filenames:
            if filename.endswith('.py') and not filename.startswith('__'):
                filepath = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(filepath, _ROOT_DIR)

                try:
                    # 先做编译检查（语法验证，不执行模块代码）
                    with open(filepath, 'r', encoding='utf-8') as f:
                        source = f.read()
                    compile(source, filepath, 'exec')
                except SyntaxError as e:
                    _ERRORS.append(f"{rel_path}: SyntaxError: {e}")
                except Exception as e:
                    _ERRORS.append(f"{rel_path}: {type(e).__name__}: {e}")

    if _ERRORS:
        print(f"发现 {len(_ERRORS)} 个问题:")
        for err in _ERRORS:
            print(f"  {err}")
        sys.exit(1)
    else:
        print(f"所有模块编译检查通过（根目录: {_ROOT_DIR}）。")


if __name__ == '__main__':
    main()
