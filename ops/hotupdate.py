#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型/规则热更新 — 新版本放入指定目录即可加载，无需重装

职责：
- 监听"热更新目录"（默认 ~/.tb_risk/hotupdate/）。
- 检测到新文件后自动加载：
    - .joblib / .pkl  → 通过 ml_loader 加载为 ML 模型
    - .pth / .pt      → 通过 gnn_loader 加载为 GNN 模型
    - .json           → 通过 rule_loader 热更新规则配置
- 加载成功的文件归档到 models/ 并登记入模型仓库；失败的记录错误日志。
- 线程轮询实现，不依赖第三方库。

设计要点：
- 目录扫描 + 文件去重（记录已处理文件名），幂等。
- 通过注入回调（ml_loader/gnn_loader/rule_loader/on_update）解耦加载逻辑，
  便于在 GUI 中接线到 MLRiskPredictor。
- 原子移动（先 copy 再删除源文件），避免读到写入一半的文件。
"""

import copy
import json
import logging
import os
import shutil
import threading
import time
from typing import Callable, Dict, List, Optional

LOGGER = logging.getLogger("tb_risk.ops.hotupdate")

# 默认热更新目录（支持环境变量覆盖，便于容器/Docker 挂载）
DEFAULT_HOTUPDATE_DIR = os.environ.get(
    'TB_RISK_HOTUPDATE_DIR',
    os.path.join(os.path.expanduser('~'), '.tb_risk', 'hotupdate'))

# 支持的扩展名分类
ML_EXT = ('.joblib', '.pkl', '.pickle')
GNN_EXT = ('.pth', '.pt')
RULE_EXT = ('.json',)


def infer_file_kind(filename: str) -> Optional[str]:
    """根据扩展名推断文件类型：'ml' / 'gnn' / 'rule' / None。"""
    ext = os.path.splitext(filename)[1].lower()
    if ext in ML_EXT:
        return 'ml'
    if ext in GNN_EXT:
        return 'gnn'
    if ext in RULE_EXT:
        return 'rule'
    return None


# ---------------------------------------------------------------------------
# 运行时规则注册表（支持热更新覆盖）
# ---------------------------------------------------------------------------

class RuleRegistry:
    """可热更新的运行时规则存储。

    允许在运行期通过 JSON 文件覆盖评估/分级阈值等规则，无需重启或重装。
    未配置的键回退到默认值。
    """

    def __init__(self):
        self._rules: Dict[str, object] = {}
        self._lock = threading.Lock()
        self._version = 0

    @property
    def version(self) -> int:
        """规则版本号（每次热更新 +1）。"""
        with self._lock:
            return self._version

    def load_json(self, filepath: str) -> dict:
        """从 JSON 文件加载规则并合并到运行时规则。

        返回：
            dict: { ok, count, version }
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("规则文件顶层必须是 JSON 对象")
        with self._lock:
            self._rules.update(copy.deepcopy(data))
            self._version += 1
            version = self._version
        LOGGER.info("规则热更新: %s 合并 %d 个键, version=%d",
                    filepath, len(data), version)
        return {"ok": True, "count": len(data), "version": version}

    def set(self, key: str, value) -> None:
        with self._lock:
            self._rules[key] = value
            self._version += 1

    def get(self, key: str, default=None):
        """读取运行时规则，未覆盖时返回默认值。"""
        with self._lock:
            return self._rules.get(key, default)

    def get_all(self) -> Dict[str, object]:
        with self._lock:
            return dict(self._rules)

    def clear(self) -> None:
        with self._lock:
            self._rules.clear()
            self._version += 1


# 全局默认规则注册表
DEFAULT_RULE_REGISTRY = RuleRegistry()


# ---------------------------------------------------------------------------
# 热更新监听器
# ---------------------------------------------------------------------------

class HotUpdateWatcher:
    """热更新目录监听器（线程轮询）。"""

    def __init__(self, watch_dir: Optional[str] = None,
                 interval: float = 3.0,
                 ml_loader: Optional[Callable[[str], bool]] = None,
                 gnn_loader: Optional[Callable[[str], bool]] = None,
                 rule_loader: Optional[Callable[[str], object]] = None,
                 rule_registry: Optional[RuleRegistry] = None,
                 on_update: Optional[Callable[[dict], None]] = None,
                 auto_archive: bool = True):
        """
        参数：
            watch_dir: 监听目录，默认 ~/.tb_risk/hotupdate
            interval: 轮询间隔（秒）
            ml_loader: 加载 ML 模型文件的回调，返回 bool
            gnn_loader: 加载 GNN 模型文件的回调，返回 bool
            rule_loader: 加载规则 JSON 的回调，返回对象
            rule_registry: 规则注册表（默认全局 DEFAULT_RULE_REGISTRY）
            on_update: 每次成功处理文件后的回调（接收 result dict）
            auto_archive: 是否将成功加载的模型归档到模型仓库
        """
        self.watch_dir = watch_dir or DEFAULT_HOTUPDATE_DIR
        self.interval = interval
        self.ml_loader = ml_loader
        self.gnn_loader = gnn_loader
        self.rule_loader = rule_loader
        self.rule_registry = rule_registry or DEFAULT_RULE_REGISTRY
        self.on_update = on_update
        self.auto_archive = auto_archive

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._processed: set = set()
        self._history: List[dict] = []
        self._running = False
        self._lock = threading.Lock()

        os.makedirs(self.watch_dir, exist_ok=True)

    # ---------------- 生命周期 ----------------

    def start(self) -> bool:
        """启动监听线程。"""
        if self._running:
            return True
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True,
            name="tb-hotupdate")
        self._thread.start()
        LOGGER.info("热更新监听已启动: %s (间隔 %.1fs)", self.watch_dir, self.interval)
        return True

    def stop(self) -> None:
        """停止监听线程。"""
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1.0)
            self._thread = None
        LOGGER.info("热更新监听已停止")

    def is_running(self) -> bool:
        return self._running

    # ---------------- 扫描 ----------------

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                self.scan_once()
            except Exception as e:
                LOGGER.warning("热更新扫描出错: %s", e, exc_info=True)
            self._stop_event.wait(self.interval)

    @staticmethod
    def _file_signature(full_path: str, name: str) -> tuple:
        """生成文件签名（名称 + 修改时间 + 大小）。

        规则 JSON 文件在加载成功后会保留在目录中（不删除），仅按名称去重会
        导致"同名但内容已更新"的规则文件无法再次热加载。加入 mtime 与大小
        后可识别内容变更，实现真正的热更新。
        """
        try:
            st = os.stat(full_path)
            return (name, st.st_mtime_ns, st.st_size)
        except OSError:
            return (name, 0, 0)

    def scan_once(self) -> List[dict]:
        """执行一次扫描：处理热更新目录中的新文件。

        返回：
            本次处理结果列表
        """
        results = []
        if not os.path.isdir(self.watch_dir):
            return results
        try:
            names = sorted(os.listdir(self.watch_dir))
        except OSError as e:
            LOGGER.warning("读取热更新目录失败: %s", e)
            return results

        for name in names:
            if name.startswith('.'):
                continue
            full_path = os.path.join(self.watch_dir, name)
            if not os.path.isfile(full_path):
                continue

            sig = self._file_signature(full_path, name)
            if sig in self._processed:
                continue

            result = self._handle_file(full_path, name)
            result.setdefault('time', time.strftime('%Y-%m-%d %H:%M:%S'))
            results.append(result)
            with self._lock:
                self._processed.add(sig)
                self._history.append(result)
                self._history = self._history[-200:]  # 保留最近 200 条
        return results

    def _handle_file(self, full_path: str, name: str) -> dict:
        """处理单个新文件。"""
        kind = infer_file_kind(name)
        if kind is None:
            return {"ok": False, "name": name, "kind": None,
                    "message": f"不支持的扩展名，已跳过: {name}"}

        # 复制到安全临时文件再加载，避免读取写入一半的文件
        staged = full_path + ".staging"
        try:
            shutil.copy2(full_path, staged)
        except OSError as e:
            return {"ok": False, "name": name, "kind": kind,
                    "message": f"读取文件失败: {e}"}

        try:
            if kind == 'ml':
                ok = self._load_ml(staged)
            elif kind == 'gnn':
                ok = self._load_gnn(staged)
            else:
                ok = self._load_rule(full_path)
        except Exception as e:
            LOGGER.warning("热更新加载 %s 失败: %s", name, e, exc_info=True)
            self._cleanup_staging(staged)
            return {"ok": False, "name": name, "kind": kind,
                    "message": f"加载失败: {e}"}
        finally:
            self._cleanup_staging(staged)

        # 加载回调返回 False（加载未成功）：标记失败并保留源文件，避免误删
        if not ok:
            return {"ok": False, "name": name, "kind": kind,
                    "message": f"加载失败，源文件已保留: {name}"}

        # 成功处理：归档模型并移除源文件
        if ok and kind in ('ml', 'gnn'):
            self._archive_model(full_path, kind)
        elif ok and kind == 'rule':
            # 规则文件保留作为已生效标记，仍删除源避免重复
            pass
        try:
            if os.path.exists(full_path) and kind != 'rule':
                os.remove(full_path)
        except OSError as e:
            LOGGER.debug("删除已处理文件 %s 失败: %s", full_path, e)

        message = self._success_message(kind, name)
        result = {"ok": True, "name": name, "kind": kind, "message": message}

        if self.on_update is not None:
            try:
                self.on_update(result)
            except Exception as e:
                LOGGER.debug("on_update 回调失败: %s", e)
        return result

    # ---------------- 具体加载 ----------------

    def _load_ml(self, staged: str) -> bool:
        if self.ml_loader is None:
            LOGGER.warning("未配置 ML 加载回调，跳过 ML 模型热更新")
            return False
        try:
            return bool(self.ml_loader(staged))
        except Exception as e:
            LOGGER.warning("ML 模型加载失败: %s", e, exc_info=True)
            return False

    def _load_gnn(self, staged: str) -> bool:
        if self.gnn_loader is None:
            LOGGER.warning("未配置 GNN 加载回调，跳过 GNN 模型热更新")
            return False
        try:
            return bool(self.gnn_loader(staged))
        except Exception as e:
            LOGGER.warning("GNN 模型加载失败: %s", e, exc_info=True)
            return False

    def _load_rule(self, full_path: str) -> bool:
        try:
            self.rule_registry.load_json(full_path)
            if self.rule_loader is not None:
                self.rule_loader(full_path)
            return True
        except Exception as e:
            LOGGER.warning("规则热更新失败: %s", e, exc_info=True)
            return False

    # ---------------- 归档 ----------------

    def _archive_model(self, full_path: str, kind: str) -> None:
        if not self.auto_archive:
            return
        try:
            from ..gui.training_panel.model_repo import ModelRepository
            repo = ModelRepository()
            repo.save_model(source_path=full_path, model_type=kind,
                            notes="hot-update")
        except Exception as e:
            LOGGER.debug("模型归档到仓库失败: %s", e)

    @staticmethod
    def _cleanup_staging(staged: str) -> None:
        try:
            if os.path.exists(staged):
                os.remove(staged)
        except OSError:
            pass

    @staticmethod
    def _success_message(kind: str, name: str) -> str:
        labels = {'ml': 'ML 模型', 'gnn': 'GNN 模型', 'rule': '规则'}
        return f"{labels.get(kind, kind)}已热更新: {name}"

    # ---------------- 查询 ----------------

    def history(self) -> List[dict]:
        with self._lock:
            return list(self._history)

    def reset_processed(self) -> None:
        with self._lock:
            self._processed.clear()


__all__ = [
    "DEFAULT_HOTUPDATE_DIR", "infer_file_kind",
    "RuleRegistry", "DEFAULT_RULE_REGISTRY", "HotUpdateWatcher",
]
