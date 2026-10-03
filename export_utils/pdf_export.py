#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PDF 报告导出（纯函数，GUI 和非 GUI 场景均可复用）。"""


def _export_pdf_report(results, patient_info, filepath):
    """导出完整的 PDF 报告（基于 reportlab 生成，含患者摘要、接触者风险列表）

    将 gui/mixins.py 中 GUI 耦合的 PDF 导出逻辑抽取为独立纯函数，
    使 PDF 导出能力可被非 GUI 场景（CLI、API、测试）复用。

    参数：
        results (dict): 评估结果字典，需包含：
            - 'overall_risk': 综合风险等级（str）
            - 'potential_patients': {'family': [...], 'social': [...]}
        patient_info (dict): 患者基本信息，需包含：
            - 'basic_info': {'age': int, 'sputum_smear': int, 'has_cavity': int}
        filepath (str): 输出 PDF 文件路径

    返回：
        bool: 是否成功
    """
    import datetime
    import logging
    _logger = logging.getLogger("tb_risk.export")

    if not results:
        _logger.warning("结果为空，无法导出 PDF")
        return False

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        )
        from reportlab.lib import colors
    except ImportError:
        _logger.warning(
            "PDF 导出需要 reportlab 库。请安装: pip install reportlab"
        )
        return False

    try:
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()

        # 标题
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1
        )
        elements.append(Paragraph("结核病风险评估报告", title_style))

        # 患者摘要
        elements.append(Paragraph("一、患者摘要", styles['Heading2']))
        basic = patient_info.get('basic_info', {}) if patient_info else {}
        patient_data = [
            ["项目", "数值"],
            ["年龄", str(basic.get('age', '未知'))],
            ["痰涂片结果",
             "涂阳" if basic.get('sputum_smear', 1) == 2 else "涂阴"],
            ["是否有空洞",
             "有" if basic.get('has_cavity', 1) == 2 else "无"],
            ["综合风险等级", results.get('overall_risk', '未知')]
        ]
        patient_table = Table(patient_data)
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(patient_table)
        elements.append(Spacer(1, 20))

        # 接触者风险列表
        elements.append(Paragraph("二、接触者风险列表", styles['Heading2']))
        potential = results.get('potential_patients', {})

        # 家庭成员
        elements.append(Paragraph("家庭成员：", styles['Heading3']))
        family_data = [["姓名", "关系", "风险等级", "感染概率(%)"]]
        for member in potential.get('family', []):
            family_data.append([
                member.get('name', '未知'),
                member.get('relationship', '未知'),
                member.get('priority', '未知'),
                f"{member.get('disease_probability', 0):.1f}"
            ])
        if len(family_data) > 1:
            family_table = Table(family_data)
            family_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(family_table)

        # 社会接触者
        elements.append(Paragraph("社会接触者：", styles['Heading3']))
        social_data = [["姓名", "类型", "风险等级", "感染概率(%)"]]
        for contact in potential.get('social', []):
            social_data.append([
                contact.get('name', '未知'),
                contact.get('relationship', '未知'),
                contact.get('priority', '未知'),
                f"{contact.get('disease_probability', 0):.1f}"
            ])
        if len(social_data) > 1:
            social_table = Table(social_data)
            social_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(social_table)

        elements.append(Spacer(1, 20))

        # 生成时间
        time_style = ParagraphStyle(
            'Time',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey
        )
        elements.append(Paragraph(
            f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            time_style
        ))

        doc.build(elements)
        _logger.info("PDF 报告已导出到: %s", filepath)
        return True

    except Exception as e:
        _logger.warning("PDF 报告导出失败: %s", e, exc_info=True)
        return False
