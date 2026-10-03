#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""部署期服务（core 层，纯 Python，不依赖 tkinter）。

为 GUI「部署与监控」面板提供三类能力：

  1. 批量接触者评估  —— ``score_contacts_batch``：
     对导入的一批接触者逐个打分（三层集成 → ML 集成 → 轻量评分三级回退），
     产出风险排序表所需的结构化结果，并写入评估日志（漂移监控数据源）。

  2. 批量模型对比    —— ``compare_models_on_cohort``：
     多个归档模型快照在同一确定性评估队列上对比 AUROC / AUPRC / Brier。

  3. 性能趋势        —— ``collect_performance_trend``：
     从训练档案读取各 run 指标，按 模型类型@数据集 分桶输出时间序列。

所有函数防御性返回 dict，绝不向 GUI 抛异常。
"""

import json
import logging
import os

from ..constants import (
    INFECTION_PROB_HIGH_RISK,
    INFECTION_PROB_MEDIUM_RISK,
)

LOGGER = logging.getLogger("tb_risk.core.deploy_services")


# ==============================================================================
# 通用工具
# ==============================================================================

def resolve_archive_dir():
    """解析训练档案目录。

    优先级：TB_RISK_ARCHIVE_DIR 环境变量 → ~/.tb_risk（有 training_log.jsonl）
    → 项目内 data/training_archive（存在 training_log.jsonl 时）→ ~/.tb_risk。
    """
    env_dir = os.environ.get("TB_RISK_ARCHIVE_DIR", "").strip()
    if env_dir:
        return env_dir
    home_dir = os.path.join(os.path.expanduser('~'), '.tb_risk')
    if os.path.exists(os.path.join(home_dir, 'training_log.jsonl')):
        return home_dir
    project_archive = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'training_archive')
    if os.path.exists(os.path.join(project_archive, 'training_log.jsonl')):
        return project_archive
    return home_dir


def classify_risk(prob_percent):
    """按全局阈值将 0-100 风险概率分为 高/中/低风险。"""
    if not isinstance(prob_percent, (int, float)):
        return '未知'
    if prob_percent >= INFECTION_PROB_HIGH_RISK:
        return '高风险'
    if prob_percent >= INFECTION_PROB_MEDIUM_RISK:
        return '中风险'
    return '低风险'


def _auroc(scores, labels):
    """秩和法 AUROC（处理并列；无正负样本时返回 None）。"""
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based 平均秩
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _auprc(scores, labels):
    """PR 曲线下面积（梯形近似；无正样本时返回 None）。"""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    total_pos = sum(1 for y in labels if y == 1)
    if total_pos == 0:
        return None
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for idx in order:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / total_pos
        precision = tp / (tp + fp)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return area


def _brier(scores, labels):
    """Brier 分数（scores 为 0-100 概率）。"""
    if not scores:
        return None
    return sum((s / 100.0 - y) ** 2 for s, y in zip(scores, labels)) / len(scores)


# ==============================================================================
# 1. 批量接触者评估
# ==============================================================================

def _score_one_contact(contact, contact_type, ml_predictor, integrator,
                       assessment):
    """为单个接触者打分（三级回退：三层集成 → ML 集成 → 轻量评分）。

    返回：
        dict: {final_score, ml_prob, gnn_prob, score_source}
        score_source ∈ three_layer / ml_ensemble / lightweight / unavailable
    """
    result = {
        'final_score': None, 'ml_prob': None, 'gnn_prob': None,
        'score_source': 'unavailable',
    }

    # 第一级：三方向集成（含三层递进架构）
    if integrator is not None and ml_predictor is not None \
            and getattr(ml_predictor, 'is_trained', False):
        try:
            ens = integrator.integrate_predictions(
                ml_predictor, assessment, contact, contact_type)
            tl = (ens or {}).get('three_layer') or {}
            decision = tl.get('decision', {}) if isinstance(tl, dict) else {}
            combined = decision.get('combined_probability')
            ml_info = (ens or {}).get('ml') or {}
            gnn_info = (ens or {}).get('gnn') or {}
            if isinstance(ml_info.get('risk_probability'), (int, float)):
                result['ml_prob'] = float(ml_info['risk_probability'])
            if isinstance(gnn_info.get('risk_probability'), (int, float)):
                result['gnn_prob'] = float(gnn_info['risk_probability'])
            if isinstance(combined, (int, float)):
                result['final_score'] = float(combined)
                result['score_source'] = 'three_layer'
                return result
            ens_info = (ens or {}).get('ensemble') or {}
            ens_prob = ens_info.get('risk_probability')
            if isinstance(ens_prob, (int, float)):
                result['final_score'] = float(ens_prob)
                result['score_source'] = 'ml_ensemble'
                return result
        except Exception as e:  # noqa: BLE001 — 批量场景逐条防御
            LOGGER.debug("集成打分失败，回退 ML: %s", e)

    # 第二级：仅 ML 集成（无集成器时）
    if ml_predictor is not None and getattr(ml_predictor, 'is_trained', False):
        try:
            pred = ml_predictor.predict_risk(contact, contact_type)
            ens = (pred or {}).get('ensemble') or {}
            prob = ens.get('risk_probability')
            if isinstance(prob, (int, float)):
                result['ml_prob'] = float(prob)
                result['final_score'] = float(prob)
                result['score_source'] = 'ml_ensemble'
                return result
        except Exception as e:  # noqa: BLE001
            LOGGER.debug("ML 打分失败，回退轻量评分: %s", e)

    # 第三级：轻量评分（NumPy 图风险预测器，无需训练，始终可用）
    try:
        from ..ml.gnn import NumPyGraphRiskPredictor
        predictor = NumPyGraphRiskPredictor()
        res = predictor.predict_contact_risk(contact, contact_type)
        prob = res.get('risk_probability')
        if isinstance(prob, (int, float)):
            # predict_contact_risk 输出 0-1 或 0-100 两种尺度均需兼容
            prob = float(prob)
            if prob <= 1.0:
                prob *= 100.0
            result['final_score'] = prob
            result['score_source'] = 'lightweight'
    except Exception as e:  # noqa: BLE001
        LOGGER.debug("轻量评分失败: %s", e)
    return result


def score_contacts_batch(contacts, ml_predictor=None, integrator=None,
                         assessment=None, source='batch', progress_cb=None,
                         log=True, log_base_dir=None):
    """批量接触者打分。

    参数：
        contacts (list[dict]): 接触者 entry 列表（需含 _contact_type 或
            contact_type 标记 'family'/'social'；缺省按 'social'）
        ml_predictor: MLRiskPredictor 实例（None/未训练时走轻量评分）
        integrator: ThreeDirectionIntegrator 实例（None 时跳过三层集成）
        assessment: 评估上下文（集成器 SEIR/GNN 方向的只读上下文，可为 None）
        source (str): 评估日志来源标记
        progress_cb (callable|None): 进度回调 fn(done, total)
        log (bool): 是否写入评估日志（漂移监控数据源）
        log_base_dir (str|None): 评估日志根目录

    返回：
        dict: {'available': bool, 'results': list[dict], 'summary': {...}}
    """
    results = []
    total = len(contacts)
    for idx, contact in enumerate(contacts, 1):
        ctype = contact.get('_contact_type') or contact.get('contact_type')
        ctype = 'family' if ctype == 'family' else 'social'
        scored = _score_one_contact(contact, ctype, ml_predictor, integrator,
                                    assessment)
        final = scored['final_score']
        results.append({
            'name': contact.get('name') or contact.get('member_name')
                    or contact.get('contact_name') or f'接触者{idx}',
            'contact_type': ctype,
            'age': contact.get('age'),
            'has_symptoms': contact.get('has_symptoms'),
            'ml_prob': scored['ml_prob'],
            'gnn_prob': scored['gnn_prob'],
            'final_score': final,
            'risk_level': classify_risk(final),
            'score_source': scored['score_source'],
            '_contact': contact,
        })
        if progress_cb is not None:
            try:
                progress_cb(idx, total)
            except Exception:  # noqa: BLE001
                pass

    scored_results = [r for r in results
                      if isinstance(r['final_score'], (int, float))]
    n_high = sum(1 for r in scored_results if r['risk_level'] == '高风险')
    n_mid = sum(1 for r in scored_results if r['risk_level'] == '中风险')
    n_low = sum(1 for r in scored_results if r['risk_level'] == '低风险')
    scores = [r['final_score'] for r in scored_results]
    summary = {
        'n_total': total,
        'n_scored': len(scored_results),
        'n_high': n_high,
        'n_medium': n_mid,
        'n_low': n_low,
        'mean_score': (sum(scores) / len(scores)) if scores else None,
        'score_source': scored_results[0]['score_source'] if scored_results
                        else 'unavailable',
    }

    if log and scored_results:
        try:
            from .assessment_log import (
                append_assessment_records, build_assessment_record)
            records = []
            for r in scored_results:
                contact = dict(r['_contact'])
                contact['_contact_type'] = r['contact_type']
                records.append(build_assessment_record(
                    contact, score=r['final_score'], ml_prob=r['ml_prob'],
                    source=source))
            append_assessment_records(records, base_dir=log_base_dir)
        except Exception as e:  # noqa: BLE001 — 日志失败不阻断打分
            LOGGER.debug("批量评估日志写入失败（非致命）: %s", e)

    # 结果按评分降序（风险排序）
    results.sort(key=lambda r: (r['final_score'] is None,
                                -(r['final_score'] or 0.0)))
    return {'available': bool(scored_results), 'results': results,
            'summary': summary}


# ==============================================================================
# 2. 批量模型对比
# ==============================================================================

def list_model_snapshots(archive_dir=None):
    """扫描档案目录下可用的 ML 模型快照（best.json 条目 + models/*.joblib）。

    返回：
        list[dict]: [{label, path, run_id, auroc, bucket, is_best}]
    """
    archive_dir = archive_dir or resolve_archive_dir()
    snapshots = []
    seen_paths = set()

    # best.json 条目（含指标与桶信息，最可信）
    best_path = os.path.join(archive_dir, 'best.json')
    try:
        with open(best_path, 'r', encoding='utf-8') as f:
            best = json.load(f)
    except (OSError, json.JSONDecodeError):
        best = {}
    if isinstance(best, dict):
        for bucket, entry in best.items():
            if not isinstance(entry, dict):
                continue
            path = entry.get('model_path')
            if not path or not os.path.exists(path):
                continue
            # 仅 ML 快照可参与对比（GNN .pt 需图构建上下文，不支持逐条打分）
            if not path.lower().endswith(('.joblib', '.pkl', '.pickle')):
                continue
            metrics = entry.get('metrics') or {}
            snapshots.append({
                'label': f"{entry.get('run_id', '?')} [{bucket}]",
                'path': path,
                'run_id': entry.get('run_id', ''),
                'auroc': metrics.get('AUROC'),
                'bucket': bucket,
                'is_best': True,
            })
            seen_paths.add(os.path.normcase(os.path.abspath(path)))

    # models/ 目录下的其他快照
    models_dir = os.path.join(archive_dir, 'models')
    try:
        filenames = sorted(os.listdir(models_dir))
    except OSError:
        filenames = []
    for fname in filenames:
        if not fname.endswith('.joblib') or not fname.startswith('best_ml_'):
            continue
        path = os.path.join(models_dir, fname)
        if os.path.normcase(os.path.abspath(path)) in seen_paths:
            continue
        run_id = fname[len('best_ml_'):-len('.joblib')]
        snapshots.append({
            'label': run_id,
            'path': path,
            'run_id': run_id,
            'auroc': None,
            'bucket': '',
            'is_best': False,
        })
    return snapshots


def compare_models_on_cohort(model_specs, n_contacts=200, n_cases=40,
                             random_state=42, progress_cb=None):
    """多个 ML 模型快照在同一确定性评估队列上对比。

    评估队列复用模型证据面板的确定性合成接触网络（含确诊标签），
    保证跨次对比可复现。

    参数：
        model_specs (list[dict]): [{label, path}, ...]（path 为 joblib 快照）
        n_contacts / n_cases / random_state: 队列生成参数
        progress_cb (callable|None): fn(done, total)

    返回：
        dict: {'available': bool, 'cohort': {...}, 'models': [ {...} ]}
    """
    from .model_evidence import _build_eval_cohort

    try:
        records, labels = _build_eval_cohort(n_contacts, n_cases, random_state)
    except Exception as e:  # noqa: BLE001
        return {'available': False,
                'reason': f'评估队列构建失败: {e}'}
    labels = [int(x) for x in labels]

    models_out = []
    total = len(model_specs)
    for idx, spec in enumerate(model_specs, 1):
        label = spec.get('label') or os.path.basename(spec.get('path', ''))
        path = spec.get('path', '')
        entry = {'label': label, 'path': path, 'available': False,
                 'auroc': None, 'auprc': None, 'brier': None,
                 'mean_score': None, 'reason': ''}
        try:
            from ..scoring.predictor import MLRiskPredictor
            predictor = MLRiskPredictor(random_state=random_state)
            if not predictor.load_model(path):
                entry['reason'] = '模型加载失败'
            else:
                scores = []
                for rec in records:
                    ctype = ('family' if rec.get('contact_type') == 'family'
                             else 'social')
                    pred = predictor.predict_risk(rec, ctype)
                    prob = ((pred or {}).get('ensemble') or {}).get(
                        'risk_probability')
                    scores.append(float(prob) if isinstance(prob, (int, float))
                                  else 0.0)
                entry.update({
                    'available': True,
                    'auroc': _auroc(scores, labels),
                    'auprc': _auprc(scores, labels),
                    'brier': _brier(scores, labels),
                    'mean_score': sum(scores) / len(scores) if scores else None,
                })
        except Exception as e:  # noqa: BLE001
            entry['reason'] = f'{type(e).__name__}: {e}'
        models_out.append(entry)
        if progress_cb is not None:
            try:
                progress_cb(idx, total)
            except Exception:  # noqa: BLE001
                pass

    n_ok = sum(1 for m in models_out if m['available'])
    return {
        'available': n_ok > 0,
        'reason': '' if n_ok else '没有可用的模型快照',
        'cohort': {
            'n_contacts': len(records),
            'n_cases': sum(labels),
            'random_state': random_state,
            'source': '确定性合成接触网络（build_synthetic_network）',
        },
        'models': models_out,
    }


# ==============================================================================
# 3. 性能趋势
# ==============================================================================

def collect_performance_trend(archive_dir=None):
    """从训练档案读取性能趋势（按 模型类型@数据集 分桶的时间序列）。

    返回：
        dict: {'available': bool, 'archive_dir': str,
               'series': {bucket: [ {timestamp, run_id, AUROC, AUPRC, Brier,
                                      is_best} ]},
               'best': {bucket: best_auroc}}
    """
    try:
        from ..gui.training_panel.training_log import TrainingLogger
    except ImportError as e:
        return {'available': False, 'reason': f'训练日志模块不可用: {e}'}

    archive_dir = archive_dir or resolve_archive_dir()
    try:
        logger = TrainingLogger(base_dir=archive_dir)
        entries = logger.read_history()
    except Exception as e:  # noqa: BLE001
        return {'available': False, 'reason': f'训练档案读取失败: {e}'}

    # best.json 当前最佳（用于趋势图参考线）
    best_auroc = {}
    try:
        with open(os.path.join(archive_dir, 'best.json'), 'r',
                  encoding='utf-8') as f:
            best = json.load(f)
        if isinstance(best, dict):
            for bucket, entry in best.items():
                if isinstance(entry, dict):
                    val = (entry.get('metrics') or {}).get('AUROC')
                    if isinstance(val, (int, float)):
                        best_auroc[bucket] = float(val)
    except (OSError, json.JSONDecodeError):
        best = {}

    series = {}
    for e in entries:
        if e.status != 'success':
            continue
        source = str((e.dataset or {}).get('source') or 'default')
        bucket = f'{e.model_type}@{source}'
        auroc = e.metrics.get('AUROC')
        if not isinstance(auroc, (int, float)):
            continue
        comparison = e.comparison or {}
        point = {
            'timestamp': e.timestamp,
            'run_id': e.run_id,
            'AUROC': float(auroc),
            'AUPRC': (float(e.metrics['AUPRC'])
                      if isinstance(e.metrics.get('AUPRC'), (int, float))
                      else None),
            'Brier': (float(e.metrics['Brier'])
                      if isinstance(e.metrics.get('Brier'), (int, float))
                      else None),
            'is_best': bool(comparison.get('is_new_best')),
        }
        series.setdefault(bucket, []).append(point)

    # 按时间正序（read_history 为倒序）
    for bucket in series:
        series[bucket].reverse()

    if not series:
        return {'available': False,
                'reason': f'训练档案（{archive_dir}）中暂无带 AUROC 的成功记录',
                'archive_dir': archive_dir}
    return {
        'available': True,
        'archive_dir': archive_dir,
        'series': series,
        'best': best_auroc,
    }


__all__ = [
    'resolve_archive_dir',
    'classify_risk',
    'score_contacts_batch',
    'list_model_snapshots',
    'compare_models_on_cohort',
    'collect_performance_trend',
]
