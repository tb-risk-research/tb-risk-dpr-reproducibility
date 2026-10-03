#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""结构化 JSON 导出。"""

import datetime
import json

from ._common import LOGGER, NUMPY_AVAILABLE, np


def export_to_json(results, filepath, schema_version='2.0'):
    """将评估结果导出为结构化 JSON 文件

    便于与其他系统集成或导入数据库。

    参数：
        results (dict): 评估结果字典
        filepath (str): 输出文件路径
        schema_version (str): schema 版本号

    返回：
        bool: 是否成功
    """
    try:
        export_data = {
            'schema_version': schema_version,
            'export_time': datetime.datetime.now().isoformat(),
            'assessment': {},
        }

        # 综合风险
        export_data['assessment']['overall'] = {
            'risk_level': results.get('overall_risk', 'N/A'),
            'risk_probability': results.get('base_infection_probability', 0),
            'suggestion': results.get('overall_suggestion', ''),
        }

        # 患者信息
        if results.get('patient_info'):
            export_data['assessment']['patient'] = results['patient_info']

        # 家庭接触者
        family = results.get('family_results', [])
        if family:
            export_data['assessment']['family_contacts'] = [
                {
                    'name': r.get('name', ''),
                    'age': r.get('age', ''),
                    'relationship': r.get('relationship', ''),
                    'risk_probability': r.get('disease_probability', 0),
                    'risk_level': r.get('priority', ''),
                    'recommendation': r.get('recommendation', ''),
                }
                for r in family
            ]

        # 社会接触者
        social = results.get('social_results', [])
        if social:
            export_data['assessment']['social_contacts'] = [
                {
                    'name': r.get('name', ''),
                    'age': r.get('age', ''),
                    'risk_probability': r.get('disease_probability', 0),
                    'risk_level': r.get('priority', ''),
                    'recommendation': r.get('recommendation', ''),
                    'is_high_risk': bool(r.get('is_high_risk')),
                }
                for r in social
            ]

        # ML 结果
        if results.get('ml_results'):
            ml = results['ml_results']
            export_data['assessment']['ml_analysis'] = {
                'model_performance': ml.get('model_performance', {}),
                'training_sample_count': ml.get('training_sample_count', 0),
            }

        # SEIR 结果
        if results.get('seir_results'):
            export_data['assessment']['seir_prediction'] = _serialize_for_json(
                results['seir_results'])

        # SDOH 结果
        if results.get('sdoh_results'):
            export_data['assessment']['sdoh_evaluation'] = _serialize_for_json(
                results['sdoh_results'])

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)

        LOGGER.info("结构化 JSON 报告已导出: %s", filepath)
        return True

    except Exception as e:
        LOGGER.warning("JSON 导出失败: %s", e, exc_info=True)
        return False


def _serialize_for_json(obj):
    """递归序列化对象为 JSON 兼容格式"""
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_json(v) for v in obj]
    elif NUMPY_AVAILABLE:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
    # 使用独立 if 而非 elif，确保 hasattr 检查在 NUMPY_AVAILABLE=True
    # 时也可达（修复：当 NUMPY_AVAILABLE=True 且 obj 为非 numpy 类型时，
    # 原 elif 链会导致 hasattr 检查被跳过，torch tensor 等对象无法序列化）
    if hasattr(obj, 'item'):
        return obj.item()
    return obj
