#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端导入管线编排器

将 data_io 包中的七个模块串联为统一的一次调用流程：
  加载 → 列名映射 → 文本抽取 → 缺失值填充 → 质量评分 → 审计记录

用法：
    from tb_risk.data_io.pipeline import import_pipeline

    result = import_pipeline("data.csv")
    # result.records      — 标准化后的记录列表
    # result.quality      — 质量评分结果
    # result.audit_report — 审计报告路径
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

LOGGER = logging.getLogger("tb_risk.data_io.pipeline")


# ==============================================================================
# 一、管线结果
# ==============================================================================

@dataclass
class PipelineResult:
    """导入管线结果。

    属性：
        records (list[dict]): 标准化后的记录列表
        meta (dict): 加载元信息
        quality (dict|None): 质量评分结果（score_batch 返回值）
        audit_report (str|None): 审计报告路径
        success (bool): 是否全部成功
        errors (list[str]): 错误列表
        warnings (list[str]): 警告列表
    """
    records: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    quality: Optional[dict] = None
    audit_report: Optional[str] = None
    success: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


# ==============================================================================
# 二、主入口
# ==============================================================================

def import_pipeline(
    filepath,
    *,
    # 列名映射
    field_mapping=None,
    auto_map=True,
    # 文本抽取
    use_llm=False,
    model_path=None,
    # 缺失值填充
    impute_strategy='auto',
    default_type='family',
    # 质量评分
    enable_scoring=True,
    flagged_threshold=0.5,
    # 审计
    enable_audit=True,
    audit_dir=None,
    db_manager=None,
    # 其他
    required_fields=None,
    **kwargs,
):
    """端到端导入管线。

    一次调用完成：加载 → 列名映射 → 文本抽取 → 缺失值填充 → 质量评分 → 审计记录

    参数：
        filepath (str): 数据文件路径（支持 CSV/JSON/Excel/Parquet/纯文本）

        field_mapping (dict|None): 手动指定的字段映射 {原始名: 标准名}
        auto_map (bool): 是否自动映射未指定的字段

        use_llm (bool): 是否使用 LLM 增强文本抽取
        model_path (str|None): LLM 模型路径

        impute_strategy (str): 缺失值填充策略 ('auto' / 'mode' / 'conditional_median' / 'median')
        default_type (str): 默认值类型 ('family' / 'social' / 'patient')

        enable_scoring (bool): 是否启用质量评分
        flagged_threshold (float): 需人工复核的分数阈值

        enable_audit (bool): 是否启用审计记录
        audit_dir (str|None): 审计日志目录
        db_manager (object|None): 数据库管理器

        required_fields (list[str]|None): 必填字段列表
        **kwargs: 传递给 load_data_from_file 的额外参数

    返回：
        PipelineResult: 管线结果
    """
    # ---- 延迟导入，避免循环依赖 ----
    try:
        from ..io_utils import (
            load_data_from_file, rename_fields_with_details, validate_data_integrity
        )
    except ImportError:
        from io_utils import (
            load_data_from_file, rename_fields_with_details, validate_data_integrity
        )

    result = PipelineResult()
    column_mapping = {}   # {原始列名: 标准字段名}
    column_sources = {}   # {原始列名: (来源, 置信度)}
    field_sources_map = {}  # {record_index: {field_name: source}}

    # ======================================================================
    # 步骤 1：加载数据
    # ======================================================================
    LOGGER.info("导入管线启动: %s", filepath)

    try:
        records, meta = load_data_from_file(filepath, **kwargs)
    except Exception as e:
        result.success = False
        result.errors.append(f"加载失败: {e}")
        LOGGER.error("数据加载失败: %s", e)
        return result

    result.meta = meta
    result.records = records

    if not records:
        result.warnings.append("未加载到任何记录")
        LOGGER.warning("数据文件为空: %s", filepath)
        return result

    LOGGER.info("步骤 1/6: 加载完成 — %d 条记录", len(records))

    # ======================================================================
    # 步骤 2：列名映射 / 字段标准化
    # ======================================================================
    if auto_map or field_mapping:
        try:
            # 2a: 执行重命名并获取映射详情（一次调用取代 _collect_column_sources + rename_fields）
            records, mapping_details = rename_fields_with_details(
                records, field_mapping=field_mapping, auto_map=auto_map, verbose=False
            )
            result.records = records

            # 2b: 从 mapping_details 中拆解 column_mapping 和 column_sources
            # mapping_details: {原始列名: (标准字段名, 来源, 置信度)}
            for orig, (std_name, source, confidence) in mapping_details.items():
                column_mapping[orig] = std_name
                column_sources[orig] = (source, confidence)

            # 2c: 将列名映射来源写入每条记录的 _source 字典
            # 构建 (标准字段名 → 来源) 的反向映射
            std_source_map = {}
            for orig, (source, confidence) in column_sources.items():
                std_name = column_mapping.get(orig, orig)
                std_source_map[std_name] = source

            for rec in records:
                # 保留文本抽取产生的 _source 元数据
                existing_source = rec.get('_source', {})
                if not isinstance(existing_source, dict):
                    existing_source = {}
                # 合并列名映射来源（文本 _source 优先）
                merged = {}
                merged.update(std_source_map)  # 列名映射来源
                merged.update(existing_source)  # 文本已有来源优先
                rec['_source'] = merged

            # 2d: 构建 field_sources_map（重命名后，字段名已标准化）
            for i, rec in enumerate(records):
                field_sources = {}
                rec_source = rec.get('_source', {})
                rec_confidence = rec.get('_confidence', {})

                for field_name in rec.keys():
                    if field_name.startswith('_'):
                        continue
                    if field_name in rec_source:
                        field_sources[field_name] = rec_source[field_name]
                    elif field_name in rec_confidence:
                        field_sources[field_name] = 'regex'
                    else:
                        field_sources[field_name] = 'direct'
                field_sources_map[i] = field_sources

            LOGGER.info("步骤 2/6: 字段名标准化完成 — %d 个字段映射", len(column_mapping))
        except Exception as e:
            result.warnings.append(f"字段名标准化失败: {e}")
            LOGGER.warning("字段名标准化失败: %s", e)

    # ======================================================================
    # 步骤 3：缺失值智能填充
    # ======================================================================
    try:
        from .imputation import impute_missing, fill_missing_fields_batch
    except ImportError:
        impute_missing = None
        fill_missing_fields_batch = None

    if impute_missing is not None:
        imputation_log = {}

        # 3a: 收集所有至少有缺失值的字段集合（一次扫描）
        fields_with_missing = set()
        for record in records:
            for field_name, val in record.items():
                if field_name.startswith('_'):
                    continue
                if val is None or val == '':
                    fields_with_missing.add(field_name)

        # 3b: 对每个字段调用一次 impute_missing（O(字段数 × 记录数)）
        for field_name in sorted(fields_with_missing):
            # 3b-i: 记录填充前哪些记录该字段缺失（用于精确更新 field_sources_map）
            missing_indices = [
                i for i, rec in enumerate(records)
                if rec.get(field_name) is None or rec.get(field_name) == ''
            ]

            filled_count, strategy, fill_val = impute_missing(
                records, field_name, strategy=impute_strategy,
            )
            if filled_count > 0:
                imputation_log[field_name] = {
                    'strategy': strategy,
                    'filled_count': filled_count,
                    'fill_value': fill_val,
                }
                # 3b-ii: 仅对填充前实际缺失的记录更新来源标记
                # 注意：missing_marker 是终态，不需要 imputed_ 前缀
                source_label = strategy if strategy == 'missing_marker' else f'imputed_{strategy}'
                for i in missing_indices:
                    fs = field_sources_map.setdefault(i, {})
                    fs[field_name] = source_label
                    # 同步更新记录的 _source 字典，使下游消费者看到真实来源
                    rec_source = records[i].setdefault('_source', {})
                    if not isinstance(rec_source, dict):
                        rec_source = {}
                        records[i]['_source'] = rec_source
                    rec_source[field_name] = source_label

        # 3c: 统一默认值填充
        filled_count = fill_missing_fields_batch(records, default_type=default_type)
        if filled_count > 0:
            imputation_log['_default_fill'] = {'filled_count': filled_count}

        if imputation_log:
            LOGGER.info("步骤 3/6: 缺失值填充完成 — %d 个字段被填充", len(imputation_log))
            result.meta['imputation'] = imputation_log
        else:
            LOGGER.info("步骤 3/6: 无缺失值需要填充")
    else:
        LOGGER.info("步骤 3/6: 跳过（imputation 模块不可用）")

    # ======================================================================
    # 步骤 4：数据完整性验证
    # ======================================================================
    try:
        validation_report = validate_data_integrity(
            records, required_fields=required_fields, verbose=False
        )
        result.meta['validation'] = validation_report

        if validation_report.get('errors') or validation_report.get('warnings'):
            issue_count = len(validation_report.get('errors', [])) + len(validation_report.get('warnings', []))
            result.warnings.append(f"数据完整性验证发现 {issue_count} 个问题")
            LOGGER.warning("步骤 4/6: 完整性验证 — %d 个问题", issue_count)
        else:
            LOGGER.info("步骤 4/6: 数据完整性验证通过")
    except Exception as e:
        result.warnings.append(f"数据完整性验证失败: {e}")
        LOGGER.warning("步骤 4/6: 数据完整性验证失败: %s", e)

    # ======================================================================
    # 步骤 5：质量评分
    # ======================================================================
    if enable_scoring:
        try:
            from .scoring import DataQualityScorer

            scorer = DataQualityScorer(flagged_threshold=flagged_threshold)
            # 传入 field_sources_map 使评分正确反映各字段的来源置信度
            quality_result = scorer.score_batch(records, field_sources_map)
            result.quality = quality_result

            flagged = quality_result['summary']['flagged_count']
            mean_score = quality_result['summary']['mean_score']

            LOGGER.info("步骤 5/6: 质量评分 — 均值=%.2f, 标记=%d 条",
                        mean_score, flagged)

            if flagged > 0:
                result.warnings.append(
                    f"{flagged} 条记录需人工复核 (质量分均值={mean_score:.2f})"
                )
        except ImportError:
            LOGGER.info("步骤 5/6: 跳过（scoring 模块不可用）")
        except Exception as e:
            result.warnings.append(f"质量评分失败: {e}")
            LOGGER.warning("步骤 5/6: 质量评分失败: %s", e)
    else:
        LOGGER.info("步骤 5/6: 质量评分已禁用")

    # ======================================================================
    # 步骤 6：审计记录
    # ======================================================================
    if enable_audit:
        try:
            from .audit import ImportAuditor

            auditor = ImportAuditor(db_manager=db_manager, log_dir=audit_dir)

            # 检测输入类型
            input_type = meta.get('input_type', 'unknown')
            if input_type == 'unknown':
                try:
                    from .text_parser import detect_input_type
                    input_type = detect_input_type(filepath)
                except ImportError:
                    pass

            auditor.log_import_start(filepath, input_type)

            # 记录列名映射（步骤 2 的核心操作）
            if column_mapping:
                # column_sources 是 (source, confidence) 顺序，
                # log_column_mapping 期望 (标准字段名, 置信度, 来源) 三元组
                auditor.log_column_mapping(
                    {orig: (std,
                            column_sources.get(orig, (None, 1.0))[1],   # confidence
                            column_sources.get(orig, (None, None))[0])   # source
                     for orig, std in column_mapping.items()}
                )

            # 记录缺失值填充
            if 'imputation' in result.meta:
                for field_name, info in result.meta['imputation'].items():
                    if field_name == '_default_fill':
                        continue
                    auditor.log_imputation(
                        field_name,
                        info['strategy'],
                        info['filled_count'],
                        info.get('fill_value'),
                    )

            # 记录质量评分
            if result.quality:
                auditor.log_quality_summary(
                    result.quality['summary'],
                    result.quality['summary']['flagged_count'],
                )

            auditor.log_import_complete(
                len(records),
                warnings=result.warnings,
                errors=result.errors,
            )

            # 导出审计报告
            audit_filename = os.path.splitext(os.path.basename(filepath))[0]
            audit_path = os.path.join(
                audit_dir or os.path.join(os.getcwd(), 'logs'),
                f"audit_{audit_filename}.json",
            )
            os.makedirs(os.path.dirname(audit_path), exist_ok=True)
            if auditor.export_audit_report(audit_path):
                result.audit_report = audit_path

            LOGGER.info("步骤 6/6: 审计记录完成 — 报告: %s",
                        audit_path if result.audit_report else "N/A")
        except ImportError:
            LOGGER.info("步骤 6/6: 跳过（audit 模块不可用）")
        except Exception as e:
            result.warnings.append(f"审计记录失败: {e}")
            LOGGER.warning("步骤 6/6: 审计记录失败: %s", e)
    else:
        LOGGER.info("步骤 6/6: 审计记录已禁用")

    # ======================================================================
    # 步骤 7：record_id 统一生成（schema 契约在数据入口落地）
    # ======================================================================
    # 优先级1：在数据进入系统的第一个环节就用 ContactRecord.from_dict() 包装
    # 每条记录，使 record_id 在此处确定下来。消除不同消费者（contact_risk_
    # calculator / integrator / ml_runner）各自补生成 record_id 导致主键
    # 不一致的风险。to_dict() 保留裸 dict 形状以兼容既有下游消费者。
    #
    # contact_type 取 default_type（若为 family/social），否则默认 'family'。
    # from_dict 宽松构造：_id 优先作为 record_id，缺失时用 fallback_idx 合成
    # 稳定 id（如 "family_0"）；非契约字段透传到 extra，to_dict 原样输出。
    # _source/_confidence 等管线元数据进入 extra 并经 to_dict 保留。
    if result.records:
        try:
            from ..schemas import ContactRecord
            ct = default_type if default_type in ('family', 'social') else 'family'
            wrapped = []
            for i, rec in enumerate(result.records):
                obj = ContactRecord.from_dict(
                    rec, contact_type=ct, fallback_idx=i)
                wrapped.append(obj.to_dict())
            result.records = wrapped
            LOGGER.info("record_id 统一生成 — %d 条记录已包装", len(wrapped))
        except Exception as e:
            result.warnings.append(f"record_id 统一生成失败: {e}")
            LOGGER.warning("record_id 统一生成失败: %s", e)

    # ======================================================================
    # 完成
    # ======================================================================
    LOGGER.info("导入管线完成: %d 条记录, 成功=%s", len(result.records), result.success)
    return result


# ==============================================================================
# 四、便捷函数
# ==============================================================================

def quick_import(filepath, **kwargs):
    """快速导入（等同 import_pipeline，使用默认参数）。

    返回：
        (list[dict], dict): (记录列表, 质量评分 summary)
    """
    result = import_pipeline(filepath, **kwargs)
    return result.records, (result.quality['summary'] if result.quality else {})


def import_with_feedback(filepath, interactive=True, on_flagged=None, **kwargs):
    """交互式导入：低置信度映射时提示用户确认。

    参数：
        filepath (str): 数据文件路径
        interactive (bool): 是否启用交互确认
        on_flagged (callable|None): 回调函数，签名为 on_flagged(flagged_records, result)
                                     当检测到需复核记录时被调用。
                                     若为 None 且 interactive=True，则回退到 print 输出。
        **kwargs: 传递给 import_pipeline 的参数

    返回：
        PipelineResult
    """
    result = import_pipeline(filepath, **kwargs)

    if not interactive or not result.quality:
        return result

    flagged_indices = [
        r['record_index']
        for r in result.quality['record_scores']
        if r['flagged']
    ]

    if not flagged_indices:
        return result

    # 构建需复核记录列表
    flagged_records = []
    for i in flagged_indices:
        score_info = result.quality['record_scores'][i]
        record = result.records[i] if i < len(result.records) else {}
        flagged_records.append({
            'index': i,
            'score': score_info['score'],
            'rating': score_info['rating'],
            'missing_fields': score_info.get('missing_fields', []),
            'low_confidence_fields': score_info.get('low_confidence_fields', []),
            'record': record,
        })

    if on_flagged is not None:
        # GUI 回调模式
        on_flagged(flagged_records, result)
    else:
        # 控制台回退模式（使用logging而非print，避免污染GUI环境）
        LOGGER.warning("\n⚠️  %d 条记录需要人工复核:\n", len(flagged_indices))
        for fr in flagged_records[:10]:  # 最多显示 10 条
            LOGGER.warning("  记录 #%d: 质量分=%.2f (%s)", fr['index'], fr['score'], fr['rating'])
            if fr['missing_fields']:
                LOGGER.warning("    缺失字段: %s", ', '.join(fr['missing_fields'][:5]))
            if fr['low_confidence_fields']:
                LOGGER.warning("    低置信度字段: %s", ', '.join(fr['low_confidence_fields'][:5]))

        if len(flagged_indices) > 10:
            LOGGER.warning("  ... 还有 %d 条记录未显示", len(flagged_indices) - 10)

    return result