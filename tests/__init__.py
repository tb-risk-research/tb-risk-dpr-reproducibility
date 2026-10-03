#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试运行器
"""
import sys
import os
import unittest

# 确保包路径在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    # 发现并运行所有测试
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回适当的退出码
    sys.exit(0 if result.wasSuccessful() else 1)