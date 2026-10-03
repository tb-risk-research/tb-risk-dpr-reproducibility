#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""7 队列 v4 主管线训练 + 部署级 checkpoint 归档（t9，2026-09-20）

背景：F2/P6 定版 v4 特征空间（per-cohort 22 维 + 队列专属 extras，
spec 单一真值源 ml/cohort_features.py）后，主管线仅做过单点验证
（kenya re-validation / brazil 冒烟测试），docstring 里的部署数字
（taiwan 0.874 / crp 0.781 / kenya 0.952）尚无可追溯的 run 记录与
部署工件——t7（2026-09-16）补全 run 记录；t8 验证 P1-a 告警清理
（数值与 t7 逐位一致）；t9 归档链修复重跑（P1-b 收口）。

产出：
  1. 部署级 checkpoint：data/processed/<cohort>_v4_checkpoint.joblib
     （save_model 全量落盘：6 模型 + v4 特征契约（v4_feature_names/
     v4_cohort/v4_fill_medians）+ deployment_model 路由 + 校准元信息
     + survival_model（peru_mdr，P5 生存路径））
  2. 训练档案：data/processed/cohort_v4_pipeline_<date>.json——每队列
     feature_space / model_selection / 6 模型 CV 指标（含 cv_imputation
     口径标注）/ 校准摘要（method/brier/auroc 前后 + 退化拦截状态）/
     peru 生存块（OOF Harrell C + Schoenfeld PH 检验）。
  3. 训练档案体系注册（t8 起接 model_saver；t9 修复落地）：results/
     {run_id}.json + training_log.jsonl + best.json 分桶 + 模型快照
     training_archive/models/best_ml_{run_id}.joblib。t8 实测暴露
     TrainingLogger 缺陷：is_new_best 严格大于，确定性重跑与 t7 逐位
     追平 → 快照回调不触发（注册表仍"有结论无归档"）。t9 修复
     （gui/training_panel/training_log.py）：追平且注册 best 无快照
     → best_artifact_repair 标记 + 同分新 run 携带快照接管桶注册
     （is_new_best 语义不变），本 run 落 7 快照并回填 model_path。

本次 run 的口径与修复（t9，训练前已落地并回归测试）：
  - 缺陷1（填补口径统一）：CV 指标折内中位填补
    （Pipeline(SimpleImputer(median)) 训练折拟合），与 F2 评估脚本
    判决口径一致；部署工件仍全数据中位（fill_medians 落盘）。
  - 缺陷2（校准拦截器同口径）：退化拦截 before = cal 半区 K 折
    未校准基线（泄漏无关），不再用 in-sample 乐观值误拦树模型。
  - P1-a（LGBM 特征名告警清理，2026-09-20）：CV 评估管线 imputer
    set_output(transform='pandas') + 校准输入 DataFrame 化，库级
    消除 "X does not have valid feature names" 告警；本脚本不再
    filterwarnings 压制（run 输出即告警审计面）。数值与 t7 逐位
    一致（固定种子；set_output 仅改输出容器）。
  - P1-c（treats 降级机制护栏）：treats checkpoint 训练时落
    research_only 标记，predict_risk 部署入口默认拦截（本 run 的
    treats 档案/快照为研究资产，部署面引用前见 constants.
    RESEARCH_ONLY_COHORTS 判决）。

用法：
    python data/run_cohort_v4_pipeline.py [--cohorts k1,k2] [--out PATH]
        [--checkpoint-dir PATH]
默认全 7 队列；kenya（63k）最久（约 4-5 分钟）。
"""
import argparse
import datetime
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

from tb_risk.scoring.predictor import MLRiskPredictor  # noqa: E402
from tb_risk.scoring.ml.training import (  # noqa: E402
    train_from_real_data, _get_git_hash,
)
from tb_risk.gui.training_panel.training_log import TrainingLogger  # noqa: E402

import pandas as pd  # noqa: E402

# 队列注册表（与 run_cohort_v4_features.py COHORTS 对齐：文件/组列/终点族）
COHORTS = {
    'nhanes':   {'file': 'ml_training_nhanes_real.csv',  'group_col': None,
                 'endpoint': 'LTBI (TST≥10mm)', 'family': 'ltbi'},
    'treats':   {'file': 'ml_training_treats_real.csv',  'group_col': 'group_id',
                 'endpoint': 'LTBI (TST/IGRA)', 'family': 'ltbi'},
    'crp':      {'file': 'ml_training_crp_real.csv',     'group_col': None,
                 'endpoint': '确诊 TB（细菌学）', 'family': 'confirmed_tb'},
    'kenya':    {'file': 'kenya_ml_training.csv',        'group_col': None,
                 'endpoint': '确诊 TB（细菌学）', 'family': 'confirmed_tb'},
    'taiwan':   {'file': 'ml_training_taiwan_real.csv',  'group_col': None,
                 'endpoint': '活动性 TB', 'family': 'confirmed_tb'},
    'brazil':   {'file': 'ml_training_brazil_real.csv',  'group_col': None,
                 'endpoint': 'LTBI (TST)', 'family': 'ltbi'},
    'peru_mdr': {'file': 'ml_training_peru_mdr_real.csv', 'group_col': 'family_id',
                 'endpoint': 'incident TB（time-to-event）', 'family': 'mdr_contacts'},
}

CALIB_FIELDS = ('method', 'n_positive', 'before_caliber', 'brier_before',
                'brier_after', 'auroc_before', 'auroc_after',
                'calibration_degraded')


def _calib_summary(entry):
    cal = entry.get('calibration')
    if not isinstance(cal, dict):
        return None
    return {k: cal.get(k) for k in CALIB_FIELDS}


def train_one(name, meta, ckpt_dir):
    csv = os.path.join(HERE, 'processed', meta['file'])
    if not os.path.exists(csv):
        return {'success': False, 'error_type': 'missing_csv',
                'note': csv}
    t0 = time.time()
    print(f'== [{name}] {meta["file"]} ==', flush=True)

    predictor = MLRiskPredictor()
    result = train_from_real_data(
        predictor, csv, target_column='tb_outcome', feature_set='v4',
        group_column=meta['group_col'])
    if not result.get('success'):
        return {'success': False,
                'error_type': result.get('error_type'),
                'note': result.get('diagnostics')}

    ckpt = os.path.join(ckpt_dir, f'{name}_v4_checkpoint.joblib')
    saved = predictor.save_model(ckpt)
    # round-trip 冒烟：加载成功 + 模型/契约在位（P6 已做深度
    # predict_risk/逐位一致性验证，此处只锁工件可用性）
    probe = MLRiskPredictor()
    loaded = probe.load_model(ckpt)
    roundtrip = bool(loaded and probe.is_trained and probe.models
                     and getattr(probe, 'v4_cohort', None) == name)

    models_out = {}
    for key, entry in predictor.model_performance.items():
        models_out[key] = {
            'AUROC': entry.get('AUROC'),
            'AUPRC': entry.get('AUPRC'),
            'AUROC_ci95': entry.get('AUROC_ci95'),
            'cv_imputation': entry.get('cv_imputation'),
            'cv_protocol': entry.get('cv_protocol'),
            'v4_cohort': entry.get('v4_cohort'),
            'n_features': entry.get('n_features'),
            'calibration': _calib_summary(entry),
            # 缺陷3（2026-09-17）：决策参考表随 run JSON 归档——
            # 审计不再需要加载 checkpoint（含 cohort_local 第三口径）
            'decision_reference': entry.get('decision_reference'),
        }
    out = {
        'success': True,
        'csv': meta['file'],
        'endpoint': meta['endpoint'],
        'endpoint_family': meta['family'],
        'n_samples': result.get('n_samples'),
        'n_positive': result.get('n_positive'),
        'group_column': result.get('group_column'),
        'cv_protocol': result.get('cv_protocol'),
        'n_groups': result.get('n_groups'),
        'feature_space': result.get('feature_space'),
        'model_selection': result.get('model_selection'),
        # P1-c：降级状态随管线 JSON 披露（审计不加载 checkpoint 即可
        # 查看——缺陷3 同惯例；treats 恒在场，其余队列为 None）
        'deployment_status': result.get('deployment_status'),
        'models': models_out,
        'checkpoint': os.path.relpath(ckpt, HERE),
        'checkpoint_bytes': (os.path.getsize(ckpt) if saved else None),
        'roundtrip_load': roundtrip,
        'survival': result.get('survival'),
        'runtime_s': round(time.time() - t0, 1),
    }

    # 缺陷3（2026-09-17）：v4 run 注册进训练档案体系（run_id 总表 +
    # results/{run_id}.json + best.json 分桶对比），历史对比链不再断档。
    # 写入 repo 内 data/training_archive（git 可追溯；同步到 ~/.tb_risk
    # 走 data/migrate_training_archive.py）
    try:
        logger = TrainingLogger(
            base_dir=os.path.join(HERE, 'training_archive'))
        best_key, best_entry = max(
            models_out.items(), key=lambda kv: kv[1]['AUROC'] or 0.0)
        metrics = {'AUROC': best_entry['AUROC'],
                   'AUPRC': best_entry['AUPRC']}
        surv = out.get('survival') or {}
        if surv.get('success') and surv.get('harrell_c_oof') is not None:
            metrics['harrell_c_oof'] = surv['harrell_c_oof']
        stamp = time.strftime('%Y%m%d_%H%M%S')
        run_id = f'{stamp}_v4_{name}'

        def _copy_checkpoint(dest):
            """model_saver：把部署 checkpoint 复制进训练档案 models/。

            t7 缺陷：未传 model_saver → run JSON/best.json 的 model_path
            为空，注册表有结论无工件级归档。TrainingLogger 仅在刷新
            历史最佳时调用本回调（snapshot 失败不阻断记录写入）。
            """
            shutil.copyfile(ckpt, dest)
            return True

        # P1-c：降级状态进训练档案 params（run JSON 审计面，不加载
        # checkpoint 即可看到 research_only 判决）
        _params = {'feature_set': 'v4', 'cohort': name,
                   'endpoint': meta['endpoint'],
                   'endpoint_family': meta['family'],
                   'group_column': meta['group_col'],
                   'cv_imputation': 'fold_internal_median',
                   'best_model': best_key}
        if result.get('deployment_status'):
            _params['deployment_status'] = \
                result['deployment_status']['status']

        logger.log(
            model_type='ml',
            params=_params,
            metrics=metrics,
            training_duration=out['runtime_s'],
            dataset={'source': meta['file'],
                     'n_samples': out['n_samples'],
                     'n_positive': out['n_positive'],
                     'positive_rate': (
                         out['n_positive'] / out['n_samples']
                         if out['n_samples'] else None),
                     'real_data': True,
                     'provenance': (
                         'v4 主管线定版 run（t9，归档链修复重跑：'
                         '追平 t7/t8 同分 + 快照接管）；checkpoint '
                         f'{out["checkpoint"]}；best.json 分桶 '
                         f'ml@{meta["file"]}')},
            run_id=run_id,
            model_saver=_copy_checkpoint,
            decision_reference=best_entry.get('decision_reference'))
        out['archive_run_id'] = run_id
        print(f'   档案注册: {run_id}', flush=True)
    except Exception as e:  # 档案失败不阻断 checkpoint/JSON 归档
        out['archive_run_id'] = None
        out['archive_error'] = str(e)
        print(f'   档案注册失败（不阻断）: {e}', flush=True)
    print(f'   n={out["n_samples"]} pos={out["n_positive"]} '
          f'[{out["runtime_s"]}s] ckpt={"OK" if saved and roundtrip else "FAIL"}'
          f' survival={"Y" if out["survival"] else "n"}', flush=True)
    if out['survival'] and out['survival'].get('success'):
        print(f'   生存路径: HarrellC_oof='
              f'{out["survival"].get("harrell_c_oof")}', flush=True)
    return out


def run():
    parser = argparse.ArgumentParser(description='7 队列 v4 主管线训练归档')
    parser.add_argument('--cohorts', type=str, default=None,
                        help='逗号分隔队列子集（冒烟用）')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default=None)
    args = parser.parse_args()

    date_tag = datetime.date.today().strftime('%Y%m%d')
    out_path = args.out or os.path.join(
        HERE, 'processed', f'cohort_v4_pipeline_{date_tag}.json')
    ckpt_dir = args.checkpoint_dir or os.path.join(HERE, 'processed')

    sel = (args.cohorts.split(',') if args.cohorts else list(COHORTS))
    unknown = [c for c in sel if c not in COHORTS]
    if unknown:
        sys.exit(f'未知队列: {unknown}（可用: {list(COHORTS)}）')

    print('== 7 队列 v4 主管线训练 + checkpoint 归档（t9）==')
    print(f'   队列: {sel}')
    print(f'   档案: {out_path}')
    print(f'   口径: CV 折内中位填补（缺陷1）/ 校准拦截同口径（缺陷2）')
    print(f'   t9: 归档链修复（追平接管） / t8: P1-a 告警清理'
          f'（无 filterwarnings 压制）/ t7: 首次注册', flush=True)

    t_start = time.time()
    archive = {
        'run': {
            'date': datetime.datetime.now().isoformat(timespec='seconds'),
            'script': os.path.basename(__file__),
            'git_commit': _get_git_hash(),
            'feature_set': 'v4',
            'cohorts_requested': sel,
            'caliber_notes': [
                'cv_imputation=fold_internal_median（缺陷1统一：与 F2 '
                'run_cohort_v4_features.py 判决口径一致；部署工件全数据'
                '中位 fill_medians 落盘）',
                'calibration before=cal_half_kfold_leak_free（缺陷2修复：'
                '拦截对照不再用 in-sample 乐观值）',
                'P1-a 特征名告警清理（t8，2026-09-20）：CV 评估管线 '
                'imputer set_output(pandas) + 校准输入 DataFrame 化；'
                '数值与 t7 逐位一致（固定种子）',
                't9 归档链修复（P1-b 收口）：TrainingLogger 追平修复'
                '（best_artifact_repair）——确定性重跑与注册 best 同分'
                '且 best 无快照 → 新 run 携带快照接管桶注册，'
                'model_path 回填非空；is_new_best 严格大于语义不变',
                'P1-c treats 护栏（t9）：treats 档案带 research_only '
                '标记（部署入口默认拦截，研究用途 allow_research=True）',
            ],
        },
        'cohorts': {},
    }
    for name in sel:
        archive['cohorts'][name] = train_one(name, COHORTS[name], ckpt_dir)

    archive['run']['runtime_s'] = round(time.time() - t_start, 1)
    n_ok = sum(1 for c in archive['cohorts'].values() if c.get('success'))
    print(f'\n== 完成: {n_ok}/{len(sel)} 队列成功 '
          f'[{archive["run"]["runtime_s"]}s] ==', flush=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(archive, f, ensure_ascii=False, indent=2)
    print(f'档案已写入: {out_path}')

    # 汇总表：队列 × 最优模型 CV AUROC（docstring 数字的 run 记录）
    print('\n== CV AUROC 汇总（折内中位口径）==')
    for name, c in archive['cohorts'].items():
        if not c.get('success'):
            print(f'  {name:9s} FAILED: {c.get("error_type")}')
            continue
        best = max(c['models'].items(),
                   key=lambda kv: kv[1]['AUROC'] or 0.0)
        lr = c['models'].get('logistic', {}).get('AUROC')
        print(f'  {name:9s} n={c["n_samples"]:>6} pos={c["n_positive"]:>5} '
              f'best={best[0]}({best[1]["AUROC"]:.4f}) '
              f'logistic={lr:.4f}')


if __name__ == '__main__':
    run()
