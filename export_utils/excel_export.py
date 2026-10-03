#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""结构化 Excel 导出（多工作表）。"""

import datetime

from ._common import LOGGER, PANDAS_AVAILABLE, pd


def export_to_excel(results, filepath, include_charts=False):
    """将评估结果导出为结构化 Excel 文件

    将患者信息、接触者风险、SEIR 预测、SDOH 评估分别输出到不同工作表，
    方便临床人员按需查看。

    参数：
        results (dict): 评估结果字典，包含以下可选键：
            - 'patient_info': 患者基本信息
            - 'family_members': 家庭成员列表
            - 'social_contacts': 社会接触者列表
            - 'family_results': 家庭成员风险评估结果
            - 'social_results': 社会接触者风险评估结果
            - 'ml_results': 机器学习评估结果
            - 'seir_results': SEIR 预测结果
            - 'sdoh_results': SDOH 评估结果
            - 'overall_risk': 综合风险等级
        filepath (str): 输出文件路径（.xlsx）
        include_charts (bool): 是否包含图表（需要 openpyxl）

    返回：
        bool: 是否成功
    """
    if not PANDAS_AVAILABLE:
        LOGGER.warning("pandas 未安装，无法导出 Excel。请安装: pip install pandas openpyxl")
        return False

    try:
        from pandas import ExcelWriter

        with ExcelWriter(filepath, engine='openpyxl') as writer:
            # Sheet 1: 综合风险摘要
            _write_summary_sheet(writer, results)

            # Sheet 2: 患者基本信息
            if results.get('patient_info'):
                _write_patient_sheet(writer, results['patient_info'])

            # Sheet 3: 家庭接触者风险评估
            if results.get('family_results'):
                _write_family_sheet(writer, results['family_results'])

            # Sheet 4: 社会接触者风险评估
            if results.get('social_results'):
                _write_social_sheet(writer, results['social_results'])

            # Sheet 5: ML 模型性能
            if results.get('ml_results'):
                _write_ml_performance_sheet(writer, results['ml_results'])

            # Sheet 6: SEIR 预测
            if results.get('seir_results'):
                _write_seir_sheet(writer, results['seir_results'])

            # Sheet 7: SDOH 评估
            if results.get('sdoh_results'):
                _write_sdoh_sheet(writer, results['sdoh_results'])

        LOGGER.info("结构化 Excel 报告已导出: %s", filepath)
        return True

    except Exception as e:
        LOGGER.warning("Excel 导出失败: %s", e, exc_info=True)
        return False


def _write_summary_sheet(writer, results):
    """写入综合风险摘要工作表"""
    # 患者数：检查 patient_info 是否为字典且有内容
    patient_count = 0
    pi = results.get('patient_info')
    if isinstance(pi, dict) and pi:
        patient_count = 1  # 单患者评估模式

    summary_data = {
        '指标': ['综合风险等级', '综合风险概率', '评估时间', '患者数', '家庭接触者数', '社会接触者数'],
        '值': [
            results.get('overall_risk', 'N/A'),
            f"{results.get('base_infection_probability', 0):.1f}%",
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            patient_count,
            len(results.get('family_results', [])),
            len(results.get('social_results', [])),
        ]
    }
    df = pd.DataFrame(summary_data)
    df.to_excel(writer, sheet_name='综合风险摘要', index=False)


def _write_patient_sheet(writer, patient_info):
    """写入患者基本信息工作表"""
    if isinstance(patient_info, dict):
        basic = patient_info.get('basic_info', patient_info)
        df = pd.DataFrame([basic])
        df.to_excel(writer, sheet_name='患者基本信息', index=False)


def _write_family_sheet(writer, family_results):
    """写入家庭接触者风险评估工作表"""
    rows = []
    for r in family_results:
        rows.append({
            '姓名': r.get('name', ''),
            '年龄': r.get('age', ''),
            '关系': r.get('relationship', ''),
            '风险概率': f"{r.get('disease_probability', 0):.1f}%",
            '风险等级': r.get('priority', ''),
            '建议': r.get('recommendation', ''),
            '接触时长(分钟)': r.get('single_duration', ''),
            '每周频次': r.get('freq_density', ''),
            '持续周期(周)': r.get('time_span', ''),
        })
    if rows:
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name='家庭接触者风险', index=False)


def _write_social_sheet(writer, social_results):
    """写入社会接触者风险评估工作表"""
    rows = []
    for r in social_results:
        rows.append({
            '姓名': r.get('name', ''),
            '年龄': r.get('age', ''),
            '风险概率': f"{r.get('disease_probability', 0):.1f}%",
            '风险等级': r.get('priority', ''),
            '建议': r.get('recommendation', ''),
            '是否高危': '是' if r.get('is_high_risk') else '否',
            '接触时长(分钟)': r.get('single_duration', ''),
            '每周频次': r.get('freq_density', ''),
            '持续周期(周)': r.get('time_span', ''),
        })
    if rows:
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name='社会接触者风险', index=False)


def _write_ml_performance_sheet(writer, ml_results):
    """写入 ML 模型性能工作表"""
    perf = ml_results.get('model_performance', {})
    rows = []
    for model_name, metrics in perf.items():
        rows.append({
            '模型': metrics.get('name', model_name),
            'AUROC': f"{metrics.get('AUROC', 0):.4f}",
            'AUPRC': f"{metrics.get('AUPRC', 0):.4f}",
            '训练样本量': ml_results.get('training_sample_count', ''),
        })
    if rows:
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name='ML模型性能', index=False)


def _write_seir_sheet(writer, seir_results):
    """写入 SEIR 预测结果工作表"""
    if isinstance(seir_results, dict):
        rows = [{
            '参数': k,
            '值': str(v)[:200] if not isinstance(v, (int, float)) else v,
        } for k, v in seir_results.items() if not isinstance(v, (list, dict))]
        if rows:
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name='SEIR预测', index=False)


def _write_sdoh_sheet(writer, sdoh_results):
    """写入 SDOH 评估结果工作表"""
    if isinstance(sdoh_results, dict):
        rows = []
        for domain, info in sdoh_results.items():
            if isinstance(info, dict):
                rows.append({
                    '领域': domain,
                    '评分': info.get('score', ''),
                    '风险等级': info.get('risk_level', ''),
                    '描述': str(info.get('description', ''))[:500],
                })
        if rows:
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name='SDOH评估', index=False)
