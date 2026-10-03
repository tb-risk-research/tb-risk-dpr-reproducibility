#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 训练面板 - 训练日志持久化与训练档案（Section VII）

将每次训练的参数配置、性能指标、训练耗时持久化到 JSONL 文件，
程序崩溃后可查看历史训练记录。

设计要点：
  - 日志文件路径：~/.tb_risk/training_log.jsonl（每行一条 JSON 记录，总表）
  - 训练档案目录：~/.tb_risk/results/{run_id}.json（每次训练一份完整归档）
  - 历史最佳快照：~/.tb_risk/best.json（按模型类型维护，原子写入）
  - 每次训练自动生成唯一 run_id（日期时间 + 关键超参数摘要）
  - 写入时自动与"上一次成功记录"和"历史最佳记录"对比，差值写入本次记录
  - 追加写入（不覆盖历史），使用原子写入单行（JSON 不含换行符）
  - 损坏的单行记录不影响其他记录的读取
  - 指标命名全局固定：主对比指标为 AUROC（项目全局约定）
"""

import json
import os
import re
import subprocess
import time
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable

from ...constants import PREVALENCE_CALIBERS


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

PRIMARY_METRIC = 'AUROC'
"""跨次对比的主指标名（全局固定，勿改名，否则历史对比会对不上）"""

BEST_FILENAME = 'best.json'
RESULTS_DIRNAME = 'results'


# ---------------------------------------------------------------------------
# 训练日志记录
# ---------------------------------------------------------------------------

@dataclass
class TrainingLogEntry:
    """单次训练的日志记录"""
    timestamp: str
    """ISO 格式时间戳（训练结束时间）"""

    model_type: str
    """模型类型：'ml' 或 'gnn'"""

    params: Dict[str, Any] = field(default_factory=dict)
    """训练参数配置"""

    metrics: Dict[str, float] = field(default_factory=dict)
    """性能指标（主指标 AUROC，命名全局固定）"""

    training_duration: float = 0.0
    """训练耗时（秒）"""

    status: str = 'success'
    """训练状态：'success' / 'failed' / 'cancelled'"""

    error_message: str = ''
    """失败时的错误信息"""

    # ======== 训练档案扩展字段（向后兼容：旧记录无这些字段） ========

    run_id: str = ''
    """唯一训练 ID：日期时间 + 关键超参数摘要，如 20260815_1432_ns2000_cv5"""

    end_time: str = ''
    """训练结束时间（ISO 格式，与 timestamp 相同，语义更明确）"""

    dataset: Dict[str, Any] = field(default_factory=dict)
    """数据集版本信息：{source, n_samples, real_data}"""

    code_version: str = ''
    """代码版本（git commit 短哈希，获取不到时为 'unknown'）"""

    model_path: str = ''
    """本次训练对应的模型权重文件路径（如有快照）"""

    comparison: Dict[str, Any] = field(default_factory=dict)
    """基准对比结果：{metric, vs_last, vs_best, is_new_best, last_run_id, best_run_id}

    P1-b（归档链修复）：best_artifact_repair——本次与注册 best 严格追平
    且 best 无权重快照（"注册表有结论无归档"缺口）时为 True，触发同分
    新 run 携带快照接管桶注册；is_new_best 保持严格大于语义不变。
    """

    prevalence_calibers: Dict[str, Any] = field(default_factory=dict)
    """阳性率双口径标注（问题4）：密接人群 2.85% vs 全人群 100/10万，
    由 log() 自动盖戳；训练档案/报告/界面三处口径统一的单一真值源
    （旧记录无此字段，加载时默认 {}）"""

    decision_reference: Dict[str, Any] = field(default_factory=dict)
    """决策参考表快照（缺陷3，2026-09-17）：排序+截断点操作特性 +
    多口径 PPV（含 cohort_local 训练队列实际阳性率）。此前只随
    checkpoint 落盘，审计需加载模型才能看到；现随 run 归档，
    results/{run_id}.json 直接可读（旧记录无此字段，默认 {}）"""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_jsonl(self) -> str:
        """序列化为单行 JSON（不含换行符）"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingLogEntry':
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        try:
            return cls(**filtered)
        except TypeError:
            return cls(timestamp='', model_type='?')

    def primary_metric_value(self) -> Optional[float]:
        """读取主指标值（缺失或不可比较时返回 None）"""
        value = self.metrics.get(PRIMARY_METRIC)
        if isinstance(value, (int, float)):
            return float(value)
        return None


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _sanitize_token(token: Any, max_len: int = 24) -> str:
    """将任意值转为 run_id 安全的短 token"""
    text = str(token)
    text = re.sub(r'[^A-Za-z0-9._-]', '', text)
    return text[:max_len]


def make_run_id(params: Optional[Dict[str, Any]] = None,
                model_type: str = 'ml',
                now: Optional[float] = None) -> str:
    """生成唯一训练 ID：日期时间 + 关键超参数摘要

    格式如：20260815_143205_ns2000_cv5（ML） / 20260815_143205_gnn_lr0.001_ep50（GNN）

    Args:
        params: 训练参数字典（TrainingConfig.to_dict()）
        model_type: 模型类型
        now: 时间戳（测试注入用）
    """
    stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(now))
    params = params or {}
    tokens = []
    if model_type == 'gnn':
        tokens.append('gnn')
        if 'gnn_learning_rate' in params:
            tokens.append(f"lr{_sanitize_token(params['gnn_learning_rate'])}")
        if 'gnn_n_epochs' in params:
            tokens.append(f"ep{_sanitize_token(params['gnn_n_epochs'])}")
    else:
        if 'n_samples' in params:
            tokens.append(f"ns{_sanitize_token(params['n_samples'])}")
        if 'ml_cv_folds' in params:
            tokens.append(f"cv{_sanitize_token(params['ml_cv_folds'])}")
        if params.get('ml_enable_hyperopt'):
            tokens.append('h1')
    tokens = [t for t in tokens if t]
    return '_'.join([stamp] + tokens) if tokens else stamp


def _get_code_version() -> str:
    """获取当前代码版本（git commit 短哈希）

    优先 subprocess git rev-parse；失败时直接读 .git 文件（无 git 环境也能工作）。
    任何失败返回 'unknown'，绝不抛异常。
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode == 0:
            return result.stdout.strip() or 'unknown'
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    # 回退：直接解析 .git 目录
    try:
        git_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.git')
        if not os.path.isdir(git_dir):
            return 'unknown'
        head_path = os.path.join(git_dir, 'HEAD')
        with open(head_path, 'r', encoding='utf-8') as f:
            head = f.read().strip()
        if head.startswith('ref: '):
            ref_path = os.path.join(git_dir, head[5:].replace('/', os.sep))
            with open(ref_path, 'r', encoding='utf-8') as f:
                return f.read().strip()[:7] or 'unknown'
        return head[:7] or 'unknown'
    except (OSError, ValueError):
        return 'unknown'


# ---------------------------------------------------------------------------
# 训练日志记录器
# ---------------------------------------------------------------------------

class TrainingLogger:
    """训练日志持久化记录器（训练档案系统）

    目录结构：
        ~/.tb_risk/
        ├── training_log.jsonl     # 总表（每行一条记录，追加写入）
        ├── best.json              # 按模型类型的历史最佳记录
        ├── results/               # 每次训练的完整档案
        │   ├── 20260815_143205_ns2000_cv5.json
        │   └── ...
        └── models/                # 模型权重（最佳快照由 ModelRepository 目录管理）

    用法：
        logger = TrainingLogger()
        logger.log(model_type='ml', params=config.to_dict(),
                   metrics={'AUROC': 0.95}, training_duration=12.5,
                   dataset={'source': 'synthetic', 'n_samples': 2000})
        entries = logger.read_history(limit=10)
        best = logger.get_best('ml')
    """

    LOG_FILENAME = 'training_log.jsonl'

    def __init__(self, base_dir: Optional[str] = None):
        """初始化训练日志记录器

        Args:
            base_dir: 日志根目录，默认为 ~/.tb_risk
        """
        if base_dir is None:
            # 与 core/assessment_log.py 同一惯例：优先 TB_RISK_ARCHIVE_DIR
            # （沙箱/自定义部署），否则 ~/.tb_risk
            base_dir = (os.environ.get('TB_RISK_ARCHIVE_DIR', '').strip()
                        or os.path.join(os.path.expanduser('~'), '.tb_risk'))
        self.base_dir = base_dir
        self.log_path = os.path.join(base_dir, self.LOG_FILENAME)
        self.best_path = os.path.join(base_dir, BEST_FILENAME)
        self.results_dir = os.path.join(base_dir, RESULTS_DIRNAME)
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    # ==================== 基准对比 ====================

    def _last_success(self, model_type: str,
                      dataset_source: Optional[str] = None) -> Optional[TrainingLogEntry]:
        """读取指定模型类型（可按数据集分桶）的最近一条成功记录

        dataset_source 非 None 时仅匹配 entry.dataset.source 相同的记录：
        不同数据集（半合成 / 肯尼亚 / NHANES）的 AUROC 跨集不可比，
        混桶比较会污染 vs_last/vs_best。
        """
        for entry in self.read_history(model_type=model_type):
            if entry.status != 'success':
                continue
            if dataset_source is not None:
                entry_src = str((entry.dataset or {}).get('source') or 'default')
                if entry_src != dataset_source:
                    continue
            return entry
        return None

    def _load_best(self) -> Dict[str, Any]:
        """加载 best.json（损坏时回退空字典）"""
        if not os.path.exists(self.best_path):
            return {}
        try:
            with open(self.best_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _atomic_write_json(self, path: str, data: Any) -> None:
        """原子写入 JSON 文件（先写临时文件再重命名）"""
        directory = os.path.dirname(path) or '.'
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix='.tmp',
                                         prefix='best_')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            raise

    def _build_comparison(self, entry: TrainingLogEntry,
                          last: Optional[TrainingLogEntry],
                          best_entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """计算本次记录相对上次/最佳的差值"""
        current = entry.primary_metric_value()
        comparison: Dict[str, Any] = {
            'metric': PRIMARY_METRIC,
            'vs_last': None,
            'vs_best': None,
            'is_new_best': False,
            'last_run_id': getattr(last, 'run_id', '') or '',
            'best_run_id': (best_entry or {}).get('run_id', ''),
        }

        if current is None:
            return comparison  # 无有效指标（失败/取消）不参与对比

        if last is not None:
            last_value = last.primary_metric_value()
            if last_value is not None:
                comparison['vs_last'] = round(current - last_value, 6)

        best_value = (best_entry or {}).get('metrics', {}).get(PRIMARY_METRIC)
        if isinstance(best_value, (int, float)):
            comparison['vs_best'] = round(current - float(best_value), 6)
            comparison['is_new_best'] = current > float(best_value)
            # P1-b（归档链修复，2026-09-20）：本次与历史最佳严格追平
            # （确定性重跑固定种子的常态）且注册 best 无权重快照
            # （t7 期"注册表有结论无归档"缺口）→ 标记修复候选：同分
            # 的新 run 携带快照接管桶注册。is_new_best 保持严格大于
            # （★新最佳语义与历史对比链不变）。
            comparison['best_artifact_repair'] = (
                not comparison['is_new_best']
                and abs(current - float(best_value)) == 0.0
                and not str((best_entry or {}).get('model_path')
                            or '').strip())
        else:
            # 无历史最佳：本次即为新的最佳
            comparison['is_new_best'] = True
        return comparison

    def _archive_result(self, entry: TrainingLogEntry) -> str:
        """将完整训练档案写入 results/{run_id}.json

        Returns:
            归档文件路径（写入失败返回空字符串）
        """
        run_id = entry.run_id or make_run_id(entry.params, entry.model_type)
        path = os.path.join(self.results_dir, f'{run_id}.json')
        # 极端情况下 run_id 重复：追加序号
        seq = 2
        while os.path.exists(path):
            path = os.path.join(self.results_dir, f'{run_id}_{seq}.json')
            seq += 1
        try:
            self._atomic_write_json(path, entry.to_dict())
            return path
        except OSError:
            return ''

    # ==================== 主入口 ====================

    def log(self, model_type: str,
            params: Optional[Dict[str, Any]] = None,
            metrics: Optional[Dict[str, float]] = None,
            training_duration: float = 0.0,
            status: str = 'success',
            error_message: str = '',
            dataset: Optional[Dict[str, Any]] = None,
            run_id: Optional[str] = None,
            model_saver: Optional[Callable[[str], bool]] = None,
            decision_reference: Optional[Dict[str, Any]] = None,
            ) -> TrainingLogEntry:
        """记录一条训练日志（含自动基准对比与最佳快照）

        Args:
            model_type: 模型类型 ('ml' / 'gnn')
            params: 训练参数配置
            metrics: 性能指标（主指标 AUROC）
            training_duration: 训练耗时（秒）
            status: 训练状态
            error_message: 失败时的错误信息
            dataset: 数据集版本信息 {source, n_samples, real_data}
            run_id: 唯一训练 ID（None 时自动生成）
            model_saver: 模型快照回调 model_saver(path)->bool，
                         仅当本次刷新历史最佳时被调用（用于保存最优权重）
            decision_reference: 决策参考表快照（缺陷3：随 run 归档，
                         审计不需要加载模型；None 时留空 {}）

        Returns:
            创建的 TrainingLogEntry
        """
        params = params or {}
        metrics = metrics or {}
        if run_id is None:
            run_id = make_run_id(params, model_type)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        entry = TrainingLogEntry(
            timestamp=timestamp,
            model_type=model_type,
            params=params,
            metrics=metrics,
            training_duration=training_duration,
            status=status,
            error_message=error_message,
            run_id=run_id,
            end_time=timestamp,
            dataset=dataset or {},
            code_version=_get_code_version(),
            decision_reference=decision_reference or {},
        )

        # 问题4：训练档案统一携带阳性率双口径戳（档案/报告/界面三处口径一致，
        # 旧记录无此字段保持向后兼容）
        entry.prevalence_calibers = {
            key: dict(caliber) for key, caliber in PREVALENCE_CALIBERS.items()
        }

        # 基准对比：先读旧基准，再决定是否刷新 best.json
        # v2 分桶：跨数据集比较无意义（半合成 0.757 vs 肯尼亚真实 0.70
        # 不可比），best.json 键升级为 "{model_type}@{dataset_source}"；
        # 旧扁平键按其自身 dataset.source 归入对应桶（仅当桶匹配时继承，
        # 保证半合成桶的历史最佳连续性）
        source_bucket = str((dataset or {}).get('source') or 'default')
        best_key = (model_type if source_bucket == 'default'
                    else f'{model_type}@{source_bucket}')
        last = self._last_success(model_type, source_bucket)
        best = self._load_best()
        best_entry = best.get(best_key)
        if best_entry is None and best_key != model_type:
            legacy = best.get(model_type)
            if (isinstance(legacy, dict)
                    and str((legacy.get('dataset') or {}).get('source')
                            or 'default') == source_bucket):
                best_entry = legacy
        entry.comparison = self._build_comparison(entry, last, best_entry)
        entry.comparison['dataset_bucket'] = source_bucket

        # 刷新历史最佳：保存权重快照并更新 best.json
        # P1-b（归档链修复）：除新最佳外，"追平修复"也走此分支——
        # 注册 best 无快照 + 本次严格追平（best_artifact_repair）→
        # 同分新 run 携带快照接管桶注册；无 model_saver 则无从产生
        # 快照，不触发（维持现状）
        _refresh_best = (
            entry.status == 'success'
            and (entry.comparison.get('is_new_best')
                 or (entry.comparison.get('best_artifact_repair')
                     and model_saver is not None)))
        if _refresh_best:
            snapshot_path = ''
            if model_saver is not None:
                try:
                    # GNN 权重为 torch checkpoint（.pt），ML 为 joblib
                    ext = 'pt' if model_type == 'gnn' else 'joblib'
                    candidate = os.path.join(
                        self.base_dir, 'models',
                        f'best_{model_type}_{run_id}.{ext}')
                    os.makedirs(os.path.dirname(candidate), exist_ok=True)
                    if model_saver(candidate):
                        snapshot_path = candidate
                except Exception:
                    snapshot_path = ''  # 快照失败不阻断记录写入
            entry.model_path = snapshot_path
            best[best_key] = {
                'run_id': run_id,
                'timestamp': timestamp,
                'metrics': dict(metrics),
                'model_path': snapshot_path,
                'params': dict(params),
                'dataset': dict(entry.dataset),
            }
            try:
                self._atomic_write_json(self.best_path, best)
            except OSError:
                pass  # best.json 写入失败不阻断总表写入

        # 归档完整档案到 results/
        self._archive_result(entry)

        # 追加总表
        self._append(entry)
        return entry

    def _append(self, entry: TrainingLogEntry) -> None:
        """追加一条日志到文件（单行 JSON + 换行符）"""
        line = entry.to_jsonl() + '\n'
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(line)
        except OSError:
            pass  # 日志写入失败不应影响主流程

    # ==================== 查询 ====================

    def read_history(self, limit: Optional[int] = None,
                      model_type: Optional[str] = None) -> List[TrainingLogEntry]:
        """读取训练历史

        Args:
            limit: 最多读取的记录数（None 时读取全部）
            model_type: 筛选模型类型

        Returns:
            按时间倒序排列的日志记录列表（最新在前）
        """
        if not os.path.exists(self.log_path):
            return []

        entries = []
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entry = TrainingLogEntry.from_dict(data)
                        if model_type is None or entry.model_type == model_type:
                            entries.append(entry)
                    except json.JSONDecodeError:
                        # 跳过损坏的单行记录
                        continue
        except OSError:
            return []

        # 倒序（最新在前）
        entries.reverse()
        if limit is not None:
            entries = entries[:limit]
        return entries

    def get_best(self, model_type: str) -> Optional[Dict[str, Any]]:
        """获取指定模型类型的历史最佳记录"""
        return self._load_best().get(model_type)

    def clear_history(self) -> int:
        """清空训练历史

        Returns:
            被删除的记录数
        """
        count = len(self.read_history())
        try:
            with open(self.log_path, 'w', encoding='utf-8') as f:
                f.write('')
        except OSError:
            pass
        return count

    def get_stats(self) -> Dict[str, Any]:
        """获取训练历史统计摘要

        Returns:
            包含 total_count, success_count, failure_count,
            avg_duration, last_training_time 的字典
        """
        entries = self.read_history()
        total = len(entries)
        success = sum(1 for e in entries if e.status == 'success')
        failure = sum(1 for e in entries if e.status == 'failed')
        durations = [e.training_duration for e in entries
                     if e.training_duration > 0]
        avg_dur = sum(durations) / len(durations) if durations else 0.0
        last_time = entries[0].timestamp if entries else ''

        return {
            'total_count': total,
            'success_count': success,
            'failure_count': failure,
            'avg_duration': avg_dur,
            'last_training_time': last_time,
        }


__all__ = ['TrainingLogEntry', 'TrainingLogger',
           'PRIMARY_METRIC', 'make_run_id']
