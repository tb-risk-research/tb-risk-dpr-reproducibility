#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI调试脚本 - 快速验证RiskAssessmentService基本功能

用法:
    python tools/dev/debug_cli.py

从项目根目录或任意位置运行均可，自动计算项目根路径。
"""
import json
import os
import sys
import tempfile
import traceback

# 自动计算项目根目录（tools/dev/ -> ../../../ 即tb_risk包所在的父目录Desktop/）
# __file__: tb_risk/tools/dev/debug_cli.py -> 4层dirname到达Desktop/
_PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)


def main():
    tmpdir = tempfile.mkdtemp()
    config_path = os.path.join(tmpdir, 'test_config.json')
    config = {
        'patient_info': {
            'basic_info': {
                'sputum_smear': 1, 'has_cavity': 1, 'cough_freq': 8, 'symptoms': 2,
                'delay_days': 10, 'family_living_conditions': 3, 'flp_percentage': 10,
                'hrsp_percentage': 15, 'treatment_duration': 2,
            }
        },
        'family_members': [
            {'name': 'member', 'relationship': 'spouse', 'age': 35, 'contact_distance': 'close',
             'ventilation': 3, 'exposure_setting': 'general', 'freq_density': 14,
             'single_duration': 120, 'time_span': 4}
        ],
        'social_contacts': [],
        'use_ml': False, 'use_seir': False,
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False)
    print(f"Config written to {config_path}")

    from tb_risk.core import RiskAssessmentService
    try:
        service = RiskAssessmentService()
        result = service.assess(
            patient_info=config['patient_info'],
            family_members=config['family_members'],
            social_contacts=config['social_contacts'],
            use_ml=False, use_seir=False,
        )
        print('SUCCESS!')
        print(json.dumps(result.get('summary', {}), indent=2, ensure_ascii=False))
    except Exception as e:
        traceback.print_exc()
        print(f'\nERROR: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
