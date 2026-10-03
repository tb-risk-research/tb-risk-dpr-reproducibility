#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 训练面板 - 模型仓库管理（Section VII）

引入"模型仓库"概念：每次训练保存时自动生成版本号和时间戳，
维护一个模型列表，支持加载历史版本、对比性能指标、回滚到之前的版本。

设计要点：
  - 使用简单的 JSON 索引文件管理（~/.tb_risk/model_repository.json），无需数据库
  - 模型文件本身保存为 .joblib（ML）或 .pt（GNN），仓库仅管理索引
  - 每条记录包含：版本号、时间戳、模型类型、文件路径、性能指标、参数配置
  - 原子写入索引文件（先写临时文件再重命名），防止写入中途崩溃
  - 仓库目录结构：~/.tb_risk/models/{version}_{type}_{timestamp}.{ext}
"""

import json
import os
import time
import tempfile
import shutil
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


# ---------------------------------------------------------------------------
# 模型记录数据类
# ---------------------------------------------------------------------------

@dataclass
class ModelRecord:
    """单次模型训练的仓库记录"""
    version: str
    """版本号，格式如 'v001'、'v002'"""

    timestamp: str
    """ISO 格式时间戳"""

    model_type: str
    """模型类型：'ml' 或 'gnn'"""

    file_path: str
    """模型文件的绝对路径"""

    params: Dict[str, Any] = field(default_factory=dict)
    """训练参数配置（TrainingConfig.to_dict()）"""

    metrics: Dict[str, float] = field(default_factory=dict)
    """性能指标（如 accuracy, f1, auc 等）"""

    training_duration: float = 0.0
    """训练耗时（秒）"""

    notes: str = ''
    """用户备注"""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelRecord':
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        try:
            return cls(**filtered)
        except TypeError:
            return cls(version='?', timestamp='', model_type='?', file_path='')


# ---------------------------------------------------------------------------
# 模型仓库
# ---------------------------------------------------------------------------

class ModelRepository:
    """模型仓库：管理版本化的模型保存/加载/对比/回滚

    仓库目录结构：
        ~/.tb_risk/
        ├── model_repository.json    # 索引文件
        └── models/                   # 模型文件目录
            ├── v001_ml_20260715_120000.joblib
            ├── v002_gnn_20260715_130000.pt
            └── ...

    用法：
        repo = ModelRepository()
        record = repo.save_model(source_path='/tmp/model.joblib',
                                   model_type='ml', params=config.to_dict(),
                                   metrics={'accuracy': 0.95})
        records = repo.list_models(model_type='ml')
        repo.load_model(record.version)
    """

    INDEX_FILENAME = 'model_repository.json'
    MODELS_DIRNAME = 'models'

    def __init__(self, base_dir: Optional[str] = None):
        """初始化模型仓库

        Args:
            base_dir: 仓库根目录，默认为 ~/.tb_risk
        """
        if base_dir is None:
            base_dir = os.path.join(os.path.expanduser('~'), '.tb_risk')
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, self.MODELS_DIRNAME)
        self.index_path = os.path.join(base_dir, self.INDEX_FILENAME)

        # 确保目录存在
        os.makedirs(self.models_dir, exist_ok=True)

    # ==================== 索引文件读写 ====================

    def _load_index(self) -> List[ModelRecord]:
        """加载索引文件（损坏时回退空列表）"""
        if not os.path.exists(self.index_path):
            return []
        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                return []
            return [ModelRecord.from_dict(item) for item in data]
        except (json.JSONDecodeError, OSError, ValueError):
            # 索引文件损坏：回退空列表而非崩溃
            return []

    def _save_index(self, records: List[ModelRecord]) -> None:
        """原子写入索引文件（先写临时文件再重命名）"""
        os.makedirs(self.base_dir, exist_ok=True)
        data = [r.to_dict() for r in records]
        # 原子写入：先写到临时文件，再 rename
        fd, tmp_path = tempfile.mkstemp(
            dir=self.base_dir, suffix='.tmp', prefix='model_repo_')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # Windows 上 os.replace 可原子替换已存在文件
            os.replace(tmp_path, self.index_path)
        except Exception:
            # 清理临时文件
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            raise

    # ==================== 版本号管理 ====================

    def _next_version(self, records: List[ModelRecord]) -> str:
        """生成下一个版本号（v001, v002, ...）"""
        max_num = 0
        for r in records:
            v = r.version
            if v.startswith('v'):
                try:
                    num = int(v[1:])
                    max_num = max(max_num, num)
                except ValueError:
                    pass
        return f'v{max_num + 1:03d}'

    def _generate_filename(self, version: str, model_type: str,
                            timestamp: str, ext: str) -> str:
        """生成模型文件名"""
        # 时间戳格式化为文件名安全的字符串
        safe_ts = timestamp.replace(':', '').replace('-', '').replace(' ', '_')
        if ext.startswith('.'):
            ext = ext[1:]
        return os.path.join(
            self.models_dir, f'{version}_{model_type}_{safe_ts}.{ext}')

    # ==================== 公共 API ====================

    def save_model(self, source_path: str, model_type: str,
                    params: Optional[Dict[str, Any]] = None,
                    metrics: Optional[Dict[str, float]] = None,
                    training_duration: float = 0.0,
                    notes: str = '',
                    ext: Optional[str] = None) -> ModelRecord:
        """将模型文件保存到仓库

        Args:
            source_path: 模型源文件路径
            model_type: 'ml' 或 'gnn'
            params: 训练参数配置字典
            metrics: 性能指标字典
            training_duration: 训练耗时（秒）
            notes: 用户备注
            ext: 文件扩展名（自动推断如果未指定）

        Returns:
            ModelRecord: 新建的模型记录
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f'模型源文件不存在: {source_path}')

        records = self._load_index()
        version = self._next_version(records)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        # 推断扩展名
        if ext is None:
            _, ext = os.path.splitext(source_path)

        dest_path = self._generate_filename(version, model_type, timestamp, ext)

        # 复制模型文件到仓库
        shutil.copy2(source_path, dest_path)

        record = ModelRecord(
            version=version,
            timestamp=timestamp,
            model_type=model_type,
            file_path=os.path.abspath(dest_path),
            params=params or {},
            metrics=metrics or {},
            training_duration=training_duration,
            notes=notes,
        )

        records.append(record)
        self._save_index(records)
        return record

    def list_models(self, model_type: Optional[str] = None) -> List[ModelRecord]:
        """列出仓库中的所有模型

        Args:
            model_type: 筛选模型类型（None 时返回全部）

        Returns:
            按时间戳降序排列的模型记录列表
        """
        records = self._load_index()
        if model_type is not None:
            records = [r for r in records if r.model_type == model_type]
        # 按时间戳降序（最新的在前）
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records

    def get_model(self, version: str) -> Optional[ModelRecord]:
        """根据版本号获取模型记录"""
        for r in self._load_index():
            if r.version == version:
                return r
        return None

    def load_model_path(self, version: str) -> Optional[str]:
        """获取指定版本模型的文件路径

        Returns:
            模型文件路径，如果版本不存在或文件已删除则返回 None
        """
        record = self.get_model(version)
        if record is None:
            return None
        if not os.path.exists(record.file_path):
            return None
        return record.file_path

    def compare_models(self, versions: Optional[List[str]] = None) -> List[ModelRecord]:
        """对比多个版本的性能指标

        Args:
            versions: 要对比的版本号列表；None 时对比全部

        Returns:
            模型记录列表（含性能指标）
        """
        records = self._load_index()
        if versions is not None:
            vset = set(versions)
            records = [r for r in records if r.version in vset]
        # 按版本号升序
        records.sort(key=lambda r: r.version)
        return records

    def rollback(self, version: str) -> Optional[ModelRecord]:
        """回滚到指定版本（返回该版本的记录）

        回滚语义：返回指定版本的模型路径，调用方负责加载。
        仓库不会删除较新的版本记录（保留完整历史）。

        Args:
            version: 要回滚到的版本号

        Returns:
            该版本的模型记录，不存在则返回 None
        """
        return self.get_model(version)

    def record_rollback(self, version: str, reason: str) -> bool:
        """在目标版本记录的备注中追加回滚原因（审计留痕）。

        Args:
            version: 回滚目标版本号
            reason: 回滚原因（用户填写）

        Returns:
            是否成功记录
        """
        records = self._load_index()
        for r in records:
            if r.version == version:
                stamp = time.strftime('%Y-%m-%d %H:%M')
                entry = f"[回滚 {stamp}] {reason.strip() or '未填写原因'}"
                r.notes = f"{r.notes} | {entry}" if r.notes else entry
                self._save_index(records)
                return True
        return False

    def delete_model(self, version: str) -> bool:
        """删除指定版本的模型文件和索引记录

        Args:
            version: 要删除的版本号

        Returns:
            是否成功删除
        """
        records = self._load_index()
        target = None
        remaining = []
        for r in records:
            if r.version == version:
                target = r
            else:
                remaining.append(r)

        if target is None:
            return False

        # 删除模型文件
        if os.path.exists(target.file_path):
            try:
                os.remove(target.file_path)
            except OSError:
                pass

        self._save_index(remaining)
        return True

    def get_latest(self, model_type: Optional[str] = None) -> Optional[ModelRecord]:
        """获取最新的模型记录"""
        records = self.list_models(model_type=model_type)
        return records[0] if records else None

    def count(self, model_type: Optional[str] = None) -> int:
        """返回仓库中的模型数量"""
        return len(self.list_models(model_type=model_type))


__all__ = ['ModelRecord', 'ModelRepository']
