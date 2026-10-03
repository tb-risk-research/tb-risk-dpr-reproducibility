#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 tb_risk 启动问题"""
import sys
import os
import traceback

# 模拟启动.bat的环境
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(ROOT))

print('=== 启动诊断 ===')
print(f'Python 版本: {sys.version}')
print(f'Python 可执行文件: {sys.executable}')
print(f'项目根目录: {ROOT}')
print(f'sys.path[0:5]: {sys.path[:5]}')
print()

try:
    print('1. 导入 tb_risk 包...')
    import tb_risk
    print(f'   tb_risk 版本: {tb_risk.__version__}')
except Exception as e:
    print(f'   失败: {e}')
    traceback.print_exc()
    sys.exit(1)

try:
    print('2. 导入 assessment 模块...')
    from tb_risk.assessment import TB_Risk_Assessment
    print('   成功')
except Exception as e:
    print(f'   失败: {e}')
    traceback.print_exc()
    sys.exit(1)

try:
    print('3. 创建应用实例...')
    app = TB_Risk_Assessment()
    print('   成功')
except Exception as e:
    print(f'   失败: {e}')
    traceback.print_exc()
    sys.exit(1)

print()
print('=== 所有导入和实例化成功 ===')
print('正在尝试 GUI 初始化（2秒后自动销毁）...')

try:
    app.init_gui()
    print('GUI 初始化成功！')
    
    # 2秒后自动销毁
    app.root.after(2000, app.root.destroy)
    print('2秒后自动关闭窗口...')
    app.root.mainloop()
    print('窗口已关闭')
except Exception as e:
    print(f'GUI 失败: {e}')
    traceback.print_exc()
    sys.exit(1)

print()
print('=== 诊断完成：程序可以正常启动 ===')
