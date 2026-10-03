#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试工具函数 — 供 test_core.py 和 test_modules.py 共用，避免代码重复"""

import sys
import unittest


def check_deps():
    """检查依赖状态"""
    deps = {}
    for mod_name in ['numpy', 'scipy', 'matplotlib', 'seaborn',
                     'sklearn', 'xgboost', 'shap', 'joblib',
                     'torch', 'torch_geometric', 'pandas']:
        try:
            __import__(mod_name)
            deps[mod_name] = '✓'
        except ImportError:
            deps[mod_name] = '✗'
    print("\n  依赖状态检查")
    print("  " + "=" * 40)
    for k, v in deps.items():
        print(f"  {v} {k}")
    print()


def run_tests_for_module(module, all_classes, verbosity=2, pattern=None):
    """运行测试套件（通用版本）

    参数：
        module: 测试模块（sys.modules[__name__]）
        all_classes: 测试类列表
        verbosity: 输出详细程度 (0/1/2)
        pattern: 仅运行匹配的测试名（如 "Vectorized"）
    """
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(module)
    if pattern:
        suite = unittest.TestSuite()
        pat_lower = pattern.lower()
        for cls in all_classes:
            # 匹配类名或方法名
            if pat_lower in cls.__name__.lower():
                suite.addTests(loader.loadTestsFromTestCase(cls))
            else:
                for name in dir(cls):
                    if name.startswith('test') and pat_lower in name.lower():
                        suite.addTest(cls(name))
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result.wasSuccessful()


def print_help(script_name, extra_commands=None):
    """打印帮助信息"""
    print(f"用法: python {script_name} [选项]")
    print("")
    print("  (无参数)     运行全部测试")
    print("  --test [名]  运行匹配名称的测试")
    print("  --check      检查依赖状态")
    print("  --help       显示帮助")
    if extra_commands:
        for cmd, desc in extra_commands:
            print(f"  {cmd:<14}{desc}")
    print()


def main_entry(script_name, module, all_classes, extra_commands=None):
    """通用入口：解析命令行参数并执行相应操作

    参数：
        script_name: 脚本名（如 "test_core.py"）
        module: 测试模块
        all_classes: 测试类列表
        extra_commands: [(arg, handler, description), ...] 额外命令
    """
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == '--check':
            check_deps()
        elif arg == '--test' or arg == '-t':
            pattern = sys.argv[2] if len(sys.argv) > 2 else None
            ok = run_tests_for_module(module, all_classes, verbosity=2, pattern=pattern)
            sys.exit(0 if ok else 1)
        elif arg == '--help' or arg == '-h':
            print_help(script_name, extra_commands)
        elif extra_commands:
            for cmd, handler, _ in extra_commands:
                if arg == cmd:
                    handler()
                    return
            print(f"未知选项: {arg}，使用 --help 查看帮助")
        else:
            print(f"未知选项: {arg}，使用 --help 查看帮助")
            sys.exit(1)
    else:
        ok = run_tests_for_module(module, all_classes, verbosity=2)
        sys.exit(0 if ok else 1)