#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型训练报告（HTML/Markdown）与真实 vs 合成数据对比报告（HTML）生成。"""

import datetime

from ._common import LOGGER, np
from ..constants import (
    PREVALENCE_CALIBERS,
    PREVALENCE_CALIBER_MIXING_WARNING,
)
from ..core.ppv import format_decision_reference_summary


def export_model_training_report(predictor, filepath, format='html'):
    """生成模型训练报告（符合 TRIPOD+AI 报告规范）

    包含训练数据统计、特征重要性排序、验证集性能指标、
    混淆矩阵、ROC 曲线数据等。

    参数：
        predictor (MLRiskPredictor): 训练好的预测器实例
        filepath (str): 输出文件路径
        format (str): 输出格式 ('html' 或 'md')

    返回：
        bool: 是否成功
    """
    if not predictor or not predictor.is_trained:
        LOGGER.warning("模型未训练，无法生成报告")
        return False

    try:
        if format == 'html':
            content = _generate_html_report(predictor)
        else:
            content = _generate_markdown_report(predictor)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        LOGGER.info("模型训练报告已导出: %s", filepath)
        return True

    except Exception as e:
        LOGGER.warning("模型训练报告生成失败: %s", e, exc_info=True)
        return False


def _generate_markdown_report(predictor):
    """生成 Markdown 格式的模型训练报告"""
    lines = []
    lines.append("# 结核病风险预测模型训练报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("**报告版本**: TRIPOD+AI 兼容")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. 训练数据摘要
    lines.append("## 1. 训练数据摘要")
    lines.append("")
    lines.append(f"- **训练样本量**: {predictor.training_sample_count}")
    lines.append(f"- **数据来源**: {'真实数据' if predictor.use_real_data else '合成数据'}")
    lines.append(f"- **特征维度**: 基础 {len(predictor.FEATURE_NAMES)} 维 + 交互 {len(predictor.INTERACTION_FEATURE_NAMES)} 维 = {len(predictor.ALL_FEATURE_NAMES)} 维")
    lines.append("")

    # 2. 模型性能
    lines.append("## 2. 模型性能指标")
    lines.append("")
    lines.append("| 模型 | AUROC | AUPRC |")
    lines.append("|------|-------|-------|")
    for model_name, metrics in predictor.model_performance.items():
        name = metrics.get('name', model_name)
        auroc = metrics.get('AUROC', 0)
        auprc = metrics.get('AUPRC', 0)
        lines.append(f"| {name} | {auroc:.4f} | {auprc:.4f} |")
    lines.append("")

    # 3. 特征重要性（单模型 SHAP + 跨模型共识）
    _shap_ok = (predictor.shap_explainer is not None
                and predictor.last_shap_values is not None)
    _models = getattr(predictor, 'models', None)
    _feat_matrix = getattr(predictor, 'last_feature_matrix', None)
    _cross_ok = bool(_models and len(_models) >= 2 and _feat_matrix is not None)
    if _shap_ok or _cross_ok:
        lines.append("## 3. 特征重要性排序（SHAP）")
        lines.append("")
    if _shap_ok:
        try:
            shap_vals = predictor.last_shap_values
            if hasattr(shap_vals, 'values'):
                mean_abs_shap = np.abs(shap_vals.values).mean(axis=0)
            else:
                mean_abs_shap = np.abs(shap_vals).mean(axis=0)

            feature_importance = sorted(
                zip(predictor.ALL_FEATURE_NAMES, mean_abs_shap),
                key=lambda x: x[1], reverse=True
            )
            lines.append("| 排名 | 特征 | SHAP 重要性 |")
            lines.append("|------|------|-------------|")
            for i, (feat, imp) in enumerate(feature_importance[:10], 1):
                lines.append(f"| {i} | {feat} | {imp:.4f} |")
            lines.append("")
        except Exception as e:
            lines.append(f"*特征重要性计算失败: {e}*")
            lines.append("")

    # 跨模型 SHAP 共识（问题3：统一解释口径——各模型 feature_importances_
    # 的 gain/split 口径不可跨模型比较，一律以 SHAP 份额共识报告）
    if _cross_ok:
        try:
            from tb_risk.scoring.ml.shap_analysis import (
                compute_cross_model_shap, format_cross_model_shap_table)
            cross = compute_cross_model_shap(
                predictor, feature_matrix=_feat_matrix)
            if cross:
                lines.append("### 跨模型 SHAP 共识（统一解释口径）")
                lines.append("")
                lines.append(
                    f"全部 {cross['n_models']} 个模型在同一特征矩阵"
                    f"（{cross['n_samples']} 个样本）上的 mean|SHAP| 份额"
                    "共识；排名区间越宽表示模型间真实分歧越大"
                    "（SHAP 口径已消除 gain/split 不可比性）。")
                lines.append("")
                lines.append(format_cross_model_shap_table(cross))
                lines.append("")
        except Exception as e:
            LOGGER.warning("跨模型 SHAP 共识计算失败: %s", e)

    # 4. Bootstrap 置信区间
    lines.append("## 4. Bootstrap 95% 置信区间")
    lines.append("")
    try:
        ci_results = predictor.compute_bootstrap_ci(n_bootstrap=50)
        if ci_results:
            lines.append("| 模型 | AUROC [95% CI] | AUPRC [95% CI] |")
            lines.append("|------|---------------|----------------|")
            for model_name, ci in ci_results.items():
                name = ci.get('name', model_name)
                lines.append(
                    f"| {name} | {ci.get('AUROC_mean', 0):.3f} "
                    f"[{ci.get('AUROC_ci_lower', 0):.3f}-{ci.get('AUROC_ci_upper', 0):.3f}] | "
                    f"{ci.get('AUPRC_mean', 0):.3f} "
                    f"[{ci.get('AUPRC_ci_lower', 0):.3f}-{ci.get('AUPRC_ci_upper', 0):.3f}] |"
                )
    except Exception as e:
        LOGGER.warning("Bootstrap 置信区间计算失败: %s", e)
        lines.append("*Bootstrap 置信区间计算失败*")
    lines.append("")

    # 5. 校准评估
    lines.append("## 5. 模型校准")
    lines.append("")
    try:
        calib_results = predictor.compute_calibration()
        if calib_results:
            lines.append("| 模型 | Brier 评分 |")
            lines.append("|------|-----------|")
            for model_name, calib in calib_results.items():
                name = calib.get('name', model_name)
                lines.append(f"| {name} | {calib.get('brier_score', 0):.4f} |")
    except Exception as e:
        LOGGER.warning("校准评估计算失败: %s", e)
        lines.append("*校准评估计算失败*")
    lines.append("")

    # 6. 决策口径（问题4：排序+截断点为主要决策形式，绝对概率仅作参考）
    lines.append("## 6. 决策口径（排序 + 截断点）")
    lines.append("")
    lines.append("本模型以**排序 + 截断点**为主要决策形式：按风险分数降序覆盖"
                 "目标人群的前 k%，绝对概率仅作参考。阳性预测值（PPV）按"
                 "目标人群实际阳性率经贝叶斯换算（PPV = sens·π /"
                 "(sens·π + (1-spec)·(1-π))），两套口径**不可混用**：")
    lines.append("")
    lines.append("| 口径 | 阳性率 | 适用场景 | 来源 |")
    lines.append("|------|--------|----------|------|")
    for caliber in PREVALENCE_CALIBERS.values():
        lines.append(f"| {caliber['label']} | {caliber['display']} "
                     f"| {'筛查决策' if caliber['value'] > 0.005 else '对照参考'} "
                     f"| {caliber['source']} |")
    lines.append("")
    lines.append(f"> **口径警示**：{PREVALENCE_CALIBER_MIXING_WARNING}")
    lines.append("")

    # 6.1 各模型决策参考表与真实数据校准前后对照（训练内核自动生成；
    #      旧档案/未校准时无此数据，仅保留口径说明）
    has_ref = any('decision_reference' in m
                  for m in predictor.model_performance.values())
    has_cal = any('calibration' in m
                  for m in predictor.model_performance.values())
    if has_ref or has_cal:
        lines.append("### 各模型决策参考与校准对照")
        lines.append("")
        for model_name, metrics in predictor.model_performance.items():
            name = metrics.get('name', model_name)
            ref = metrics.get('decision_reference')
            if ref:
                lines.append(f"- **{name}**：{format_decision_reference_summary(ref)}")
            cal = metrics.get('calibration')
            if cal:
                lines.append(
                    f"  - 校准（{cal.get('method', '?')}，阳性样本 "
                    f"{cal.get('n_positive', '?')}）：Brier "
                    f"{cal.get('brier_before', 0):.4f} → "
                    f"{cal.get('brier_after', 0):.4f}，AUROC "
                    f"{cal.get('auroc_before', 0):.4f} → "
                    f"{cal.get('auroc_after', 0):.4f}（校准前 → 校准后，"
                    f"泄漏无关评估半区）")
        lines.append("")

    # 7. 特征列表
    lines.append("## 7. 特征列表")
    lines.append("")
    lines.append("### 基础特征")
    for feat in predictor.FEATURE_NAMES:
        desc = predictor.FEATURE_DESCRIPTIONS.get(feat, '')
        lines.append(f"- **{feat}**: {desc}")
    lines.append("")
    lines.append("### 交互特征")
    for feat in predictor.INTERACTION_FEATURE_NAMES:
        desc = predictor.INTERACTION_FEATURE_DESCRIPTIONS.get(feat, '')
        lines.append(f"- **{feat}**: {desc}")
    lines.append("")

    lines.append("---")
    lines.append("*报告由 tb_risk 自动生成*")
    return '\n'.join(lines)


def _generate_html_report(predictor):
    """生成 HTML 格式的模型训练报告"""
    md_content = _generate_markdown_report(predictor)
    # 尝试使用 markdown 库进行真正的 Markdown→HTML 转换
    html_body = md_content
    try:
        import markdown
        html_body = markdown.markdown(
            md_content,
            extensions=['tables', 'fenced_code', 'codehilite']
        )
    except ImportError:
        # 回退：简单的 <pre> 包裹（纯文本 Markdown 源码）
        html_body = f'<pre style="white-space: pre-wrap; font-family: inherit;">{md_content}</pre>'
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>结核病风险预测模型训练报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background-color: #f2f6fa; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 15px; border-radius: 4px; overflow-x: auto; }}
        .footer {{ color: #999; font-size: 0.9em; margin-top: 40px; border-top: 1px solid #eee; padding-top: 10px; }}
    </style>
</head>
<body>
{html_body}
<div class="footer">报告由 tb_risk 自动生成 | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</body>
</html>"""
    return html


def export_comparison_report(comparison_result, filepath):
    """导出真实 vs 合成数据对比报告（HTML 格式）

    接收 compare_real_vs_synthetic 的返回值，生成包含三种模式性能对比表、
    Bootstrap 置信区间、Wilcoxon 检验结果、统计建议的结构化 HTML 报告。

    参数：
        comparison_result (dict): compare_real_vs_synthetic 的返回值
        filepath (str): 输出 HTML 文件路径

    返回：
        bool: 是否成功
    """
    if not comparison_result or comparison_result.get('status') != 'ok':
        LOGGER.warning("对比实验结果无效，无法生成报告")
        return False

    try:
        html = _generate_comparison_html(comparison_result)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        LOGGER.info("对比报告已导出: %s", filepath)
        return True
    except Exception as e:
        LOGGER.warning("对比报告导出失败: %s", e, exc_info=True)
        return False


def _generate_comparison_html(result):
    """生成对比实验 HTML 报告"""
    modes = result.get('modes', {})
    wilcoxon = result.get('wilcoxon', {})
    bootstrap_ci = result.get('bootstrap_ci', {})
    recommendation = result.get('recommendation', '')
    timestamp = result.get('timestamp', datetime.datetime.now().isoformat())
    test_size = result.get('test_size', 0)

    mode_order = ['synthetic', 'real', 'mixed']

    # 构建 Bootstrap CI 表格行
    bootstrap_rows = ''
    for mode_key in mode_order:
        if mode_key not in bootstrap_ci:
            continue
        ci = bootstrap_ci[mode_key].get('auroc', {})
        label = modes.get(mode_key, {}).get('label', mode_key)
        bootstrap_rows += f"""
            <tr>
                <td>{label}</td>
                <td>{ci.get('mean', 0):.4f}</td>
                <td>{ci.get('ci_95_lower', 0):.4f}</td>
                <td>{ci.get('ci_95_upper', 0):.4f}</td>
                <td>{ci.get('std', 0):.4f}</td>
                <td>{ci.get('n_bootstrap', 0)}</td>
            </tr>"""

    # 构建三模式性能对比表
    performance_rows = ''
    for mode_key in mode_order:
        m = modes.get(mode_key, {})
        label = m.get('label', mode_key)
        auroc = m.get('auroc', 0)
        auprc = m.get('auprc', 0)
        brier = m.get('brier', 0)
        train_size = m.get('train_size', 0)
        n_models = m.get('n_models', 0)
        # 找出最佳 AUROC 以高亮
        best_auroc = max(
            (modes.get(k, {}).get('auroc', 0) for k in mode_order),
            default=0
        )
        best_class = ' class="best"' if auroc == best_auroc and auroc > 0 else ''
        performance_rows += f"""
            <tr{best_class}>
                <td>{label}</td>
                <td>{auroc:.4f}</td>
                <td>{auprc:.4f}</td>
                <td>{brier:.4f}</td>
                <td>{train_size}</td>
                <td>{n_models}</td>
            </tr>"""

    # 构建 Wilcoxon 检验表
    wilcoxon_rows = ''
    for pair_key, pair_info in wilcoxon.items():
        if 'error' in pair_info:
            wilcoxon_rows += f"""
            <tr>
                <td>{pair_info.get('label', pair_key)}</td>
                <td colspan="5" class="error">检验失败: {pair_info['error']}</td>
            </tr>"""
            continue
        label = pair_info.get('label', pair_key)
        p_val = pair_info.get('p_value', 1.0)
        sig = pair_info.get('significant_at_0_05', False)
        sig_class = ' class="significant"' if sig else ''
        sig_text = '显著 (p < 0.05)' if sig else '不显著 (p ≥ 0.05)'
        wilcoxon_rows += f"""
            <tr{sig_class}>
                <td>{label}</td>
                <td>{pair_info.get('statistic', 0):.1f}</td>
                <td>{p_val:.4f}</td>
                <td>{sig_text}</td>
                <td>{pair_info.get('mean_diff', 0):.4f}</td>
                <td>{pair_info.get('n_pairs', 0)}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>真实 vs 合成数据对比实验报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
            max-width: 960px;
            margin: 0 auto;
            padding: 30px 20px;
            color: #333;
            line-height: 1.6;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 12px;
            margin-bottom: 10px;
        }}
        h2 {{
            color: #2980b9;
            margin-top: 35px;
            padding-bottom: 6px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .meta {{
            color: #888;
            font-size: 0.9em;
            margin-bottom: 25px;
        }}
        .meta span {{
            margin-right: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0 25px 0;
            font-size: 0.95em;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px 14px;
            text-align: center;
        }}
        th {{
            background-color: #2c3e50;
            color: #fff;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e8f4fd;
        }}
        tr.best {{
            background-color: #d4edda !important;
            font-weight: 600;
        }}
        tr.significant {{
            background-color: #fff3cd !important;
        }}
        td.error {{
            color: #dc3545;
            font-style: italic;
        }}
        .recommendation {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            padding: 20px 25px;
            border-radius: 8px;
            margin: 20px 0;
            font-size: 1.05em;
            line-height: 1.7;
        }}
        .recommendation strong {{
            display: block;
            font-size: 1.1em;
            margin-bottom: 8px;
        }}
        .footer {{
            color: #aaa;
            font-size: 0.85em;
            margin-top: 50px;
            padding-top: 15px;
            border-top: 1px solid #eee;
            text-align: center;
        }}
        .note {{
            background: #f0f7ff;
            border-left: 4px solid #3498db;
            padding: 12px 18px;
            margin: 20px 0;
            color: #555;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>真实数据 vs 合成数据 性能对比实验报告</h1>
    <div class="meta">
        <span>📅 生成时间: {timestamp}</span>
        <span>📊 测试集样本量: {test_size}</span>
    </div>

    <div class="note">
        <strong>方法论说明：</strong>
        本报告基于 Bootstrap BCa 方法计算 95% 置信区间，
        使用 Wilcoxon 符号秩检验判断训练模式间差异的统计显著性。
        所有模型在相同测试集上评估，确保公平比较。
    </div>

    <h2>一、三种训练模式性能对比</h2>
    <table>
        <thead>
            <tr>
                <th>训练模式</th>
                <th>AUROC</th>
                <th>AUPRC</th>
                <th>Brier Score</th>
                <th>训练样本量</th>
                <th>模型数</th>
            </tr>
        </thead>
        <tbody>
            {performance_rows}
        </tbody>
    </table>
    <p style="color:#888;font-size:0.85em;">
        * 绿色高亮行表示 AUROC 最优模式。AUROC 越接近 1、Brier 越接近 0 表示性能越好。
    </p>

    <h2>二、Bootstrap 95% 置信区间（AUROC）</h2>
    <table>
        <thead>
            <tr>
                <th>训练模式</th>
                <th>均值</th>
                <th>CI 下界 (2.5%)</th>
                <th>CI 上界 (97.5%)</th>
                <th>标准差</th>
                <th>Bootstrap 次数</th>
            </tr>
        </thead>
        <tbody>
            {bootstrap_rows}
        </tbody>
    </table>
    <p style="color:#888;font-size:0.85em;">
        * Bootstrap BCa 方法，重采样 {result.get('bootstrap_ci', {}).get('synthetic', {}).get('auroc', {}).get('n_bootstrap', 0)} 次。
        置信区间不重叠表示模式间可能存在显著差异。
    </p>

    <h2>三、Wilcoxon 符号秩检验（成对比较）</h2>
    <table>
        <thead>
            <tr>
                <th>比较对</th>
                <th>统计量 W</th>
                <th>p 值</th>
                <th>显著性</th>
                <th>均差</th>
                <th>配对样本数</th>
            </tr>
        </thead>
        <tbody>
            {wilcoxon_rows}
        </tbody>
    </table>
    <p style="color:#888;font-size:0.85em;">
        * 黄色高亮行表示差异显著 (p < 0.05)。基于预测误差平方的成对比较。
    </p>

    <h2>四、统计建议</h2>
    <div class="recommendation">
        <strong>📋 基于统计证据的建议</strong>
        {recommendation}
    </div>

    <div class="footer">
        报告由 tb_risk 自动生成 | TRIPOD+AI 兼容 | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</body>
</html>"""
    return html


def generate_assessment_report(results, format='html', ai_enhanced=False,
                              ai_config=None):
    """生成评估摘要报告（HTML/JSON/text）

    问题七：从 IOMixin.generate_report 下沉至此，供 IOMixin 委托调用，
    消除 persistence/mixins.py 与 export_utils 的职责重叠。

    缺陷 #14：新增 ai_enhanced 参数。当 ai_enhanced=True 且 AI 已配置时，
    先调用 ai.reporter.generate_report() 生成自然语言摘要，再拼接到模板报告头部。
    AI 未配置或失败时优雅降级到模板报告（不抛异常）。

    参数：
        results (dict): 评估结果，应含 family_members / social_contacts /
                        patient_info / overall_risk / base_infection_probability 等键
        format (str): 'html' / 'json' / 'text'
        ai_enhanced (bool): 是否在报告头部插入 AI 自然语言摘要（默认 False）
        ai_config: 可选 AIConfig（传给 ai.reporter.generate_report），None 时从环境变量加载

    返回：
        str: 报告内容
    """
    import json as _json

    fm = results.get('family_members', [])
    sc = results.get('social_contacts', [])
    res = results.get('results', results)  # 允许 results 自身即为结果字典
    pi = results.get('patient_info', {})

    # 可选：调用 AI 生成自然语言摘要
    ai_summary = _try_generate_ai_summary(res, ai_enhanced, ai_config)

    if format == 'json':
        data = {
            'patient_info': pi, 'family_members': fm,
            'social_contacts': sc, 'results': res,
            'generated_at': datetime.datetime.now().isoformat(),
        }
        if ai_summary:
            data['ai_summary'] = ai_summary
        return _json.dumps(data, ensure_ascii=False, indent=2, default=str)
    elif format == 'text':
        lines = []
        if ai_summary:
            lines.extend([
                "=" * 60,
                "AI 风险摘要（由 LLM 生成，仅供参考）",
                "=" * 60,
                ai_summary,
                "=" * 60,
            ])
        lines.extend([
            "=" * 60, "TB Risk Assessment Report", "=" * 60,
            f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Risk Level: {res.get('overall_risk', '?')}",
            f"Infection Prob: {res.get('base_infection_probability', 0):.2f}%",
            f"Family: {len(fm)}, Social: {len(sc)}",
            "=" * 60,
        ])
        return '\n'.join(lines)
    else:
        # HTML 格式：AI 摘要作为独立 section 插入到 <body> 后
        ai_section = ''
        if ai_summary:
            # 简单转义：将 ## 标题转为 <h3>，保留段落换行
            escaped = ai_summary.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html_summary = escaped.replace('\n', '<br>\n')
            ai_section = f'''<div class="ai-summary">
<h2>AI 风险摘要</h2>
<p><em>由 LLM 生成，仅供参考</em></p>
<div class="ai-summary-content">{html_summary}</div>
</div>
<style>.ai-summary{{background:#f0f7ff;border-left:4px solid #3498db;padding:15px 20px;margin:20px 0}}.ai-summary h2{{color:#2980b9;margin-top:0}}</style>
'''
        return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>TB Risk Report</title>
<style>body{{font-family:sans-serif;margin:40px;color:#333}}h1{{color:#c0392b}}h2{{color:#2c3e50;margin-top:30px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px}}th{{background:#f5f5f5}}.footer{{margin-top:40px;color:#999;font-size:12px}}</style>
</head>
<body><h1>TB Risk Assessment Report</h1>
{ai_section}<p>Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<h2>Summary</h2>
<p>Risk Level: {res.get('overall_risk', '?')}</p>
<p>Infection Probability: {res.get('base_infection_probability', 0):.2f}%</p>
<h2>Data</h2>
<p>Family Members: {len(fm)}, Social Contacts: {len(sc)}</p>
<div class="footer"><p>Generated by tb_risk. For reference only.</p></div>
</body></html>'''


def _try_generate_ai_summary(result, ai_enhanced, ai_config):
    """尝试调用 AI 生成自然语言摘要，失败时返回 None（优雅降级）

    Args:
        result: 评估结果 dict（含 patient_score 等）
        ai_enhanced: 是否启用 AI 增强
        ai_config: 可选 AIConfig

    Returns:
        str or None: AI 生成的摘要文本；未启用/未配置/失败时返回 None
    """
    if not ai_enhanced:
        return None
    # result 必须是非空 dict（ai.reporter.generate_report 内部会再次校验）
    if not isinstance(result, dict) or not result:
        return None
    try:
        from ..ai.reporter import generate_report as _ai_gen
        return _ai_gen(result, config=ai_config)
    except ImportError:
        LOGGER.debug("ai.reporter 模块不可用，跳过 AI 摘要生成")
        return None
    except Exception as e:
        LOGGER.warning("AI 摘要生成失败，降级到模板报告: %s", e)
        return None
