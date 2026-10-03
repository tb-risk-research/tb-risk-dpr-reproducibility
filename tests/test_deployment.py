#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""部署与运维友好功能测试套件

覆盖 ops 子包四大能力：
- 可读错误：``errors.classify_error`` / ``format_error`` / ``render_error_dialog_text``
- 离线运行：``offline.check_network`` / ``ensure_cached`` / ``status`` / ``status_text``
- 热更新：``infer_file_kind`` / ``RuleRegistry`` / ``HotUpdateWatcher``
- 部署打包：``deploy.pack_distribution``

遵循项目 unittest + pytest 约定，纯标准库可测（不依赖 torch/sklearn）。
"""
import json
import os
import sys
import tempfile
import unittest
import zipfile

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ══════════════════════════════════════════════════════════════
# 导出检查
# ══════════════════════════════════════════════════════════════

class TestOpsExports(unittest.TestCase):
    """ops 子包公共 API 导出检查。"""

    def test_package_exports(self):
        from tb_risk.ops import (
            classify_error, format_error, render_error_dialog_text,
            check_network, offline_status, ensure_cached,
            HotUpdateWatcher, RuleRegistry, pack_distribution,
        )
        for fn in (classify_error, format_error, render_error_dialog_text,
                   check_network, offline_status, ensure_cached,
                   pack_distribution):
            self.assertTrue(callable(fn))
        for cls in (HotUpdateWatcher, RuleRegistry):
            self.assertTrue(isinstance(cls, type))

    def test_error_kind_constants(self):
        from tb_risk.ops import (
            KIND_MISSING_DEP, KIND_CUDA_OOM, KIND_OOM, KIND_IMPORT,
            KIND_FILE, KIND_DISK, KIND_PERMISSION, KIND_NETWORK,
            KIND_CUDA_UNAVAILABLE, KIND_GENERIC,
        )
        kinds = (KIND_MISSING_DEP, KIND_CUDA_OOM, KIND_OOM, KIND_IMPORT,
                 KIND_FILE, KIND_DISK, KIND_PERMISSION, KIND_NETWORK,
                 KIND_CUDA_UNAVAILABLE, KIND_GENERIC)
        self.assertEqual(len(set(kinds)), len(kinds))
        for k in kinds:
            self.assertIsInstance(k, str)


# ══════════════════════════════════════════════════════════════
# 可读错误
# ══════════════════════════════════════════════════════════════

class TestReadableErrors(unittest.TestCase):
    """异常 → 可读错误与恢复建议。"""

    def _classify(self, exc):
        from tb_risk.ops.errors import classify_error
        return classify_error(exc)

    def test_cuda_oom(self):
        from tb_risk.ops.errors import KIND_CUDA_OOM
        info = self._classify(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
        self.assertEqual(info["kind"], KIND_CUDA_OOM)
        self.assertTrue(info["message"])
        self.assertTrue(info["suggestions"])

    def test_memory_error(self):
        from tb_risk.ops.errors import KIND_OOM
        info = self._classify(MemoryError("Unable to allocate array"))
        self.assertEqual(info["kind"], KIND_OOM)
        self.assertTrue(info["suggestions"])

    def test_missing_dependency(self):
        from tb_risk.ops.errors import KIND_MISSING_DEP
        info = self._classify(ModuleNotFoundError("No module named 'torch'"))
        self.assertEqual(info["kind"], KIND_MISSING_DEP)
        self.assertIn("torch", info["message"])

    def test_import_error(self):
        # ImportError 实例会被归类为 missing_dependency；KIND_IMPORT 通过
        # 文本匹配命中（如动态加载失败但抛出的不是 ImportError 实例）。
        from tb_risk.ops.errors import KIND_IMPORT
        info = self._classify(
            Exception("ImportError: DLL load failed while importing _sqlite3"))
        self.assertEqual(info["kind"], KIND_IMPORT)

    def test_file_not_found(self):
        from tb_risk.ops.errors import KIND_FILE
        info = self._classify(FileNotFoundError("No such file or directory: x.pkl"))
        self.assertEqual(info["kind"], KIND_FILE)

    def test_disk_full(self):
        from tb_risk.ops.errors import KIND_DISK
        info = self._classify(OSError("No space left on device"))
        self.assertEqual(info["kind"], KIND_DISK)

    def test_permission_denied(self):
        from tb_risk.ops.errors import KIND_PERMISSION
        info = self._classify(PermissionError("Permission denied"))
        self.assertEqual(info["kind"], KIND_PERMISSION)

    def test_network_error(self):
        from tb_risk.ops.errors import KIND_NETWORK
        info = self._classify(TimeoutError("timed out"))
        self.assertEqual(info["kind"], KIND_NETWORK)

    def test_cuda_unavailable(self):
        from tb_risk.ops.errors import KIND_CUDA_UNAVAILABLE
        info = self._classify(RuntimeError("CUDA driver version is insufficient"))
        self.assertEqual(info["kind"], KIND_CUDA_UNAVAILABLE)

    def test_generic(self):
        from tb_risk.ops.errors import KIND_GENERIC
        info = self._classify(ValueError("some random business error"))
        self.assertEqual(info["kind"], KIND_GENERIC)
        self.assertTrue(info["suggestions"])

    def test_format_error_contains_suggestions(self):
        from tb_risk.ops.errors import format_error
        text = format_error(RuntimeError("CUDA out of memory"))
        self.assertIn("恢复建议", text)
        self.assertIn("•", text)
        self.assertIn("原始错误信息", text)

    def test_render_error_dialog_text(self):
        from tb_risk.ops.errors import render_error_dialog_text
        text = render_error_dialog_text(ModuleNotFoundError("No module named 'xgboost'"))
        self.assertIn("恢复建议", text)
        self.assertIn("•", text)


# ══════════════════════════════════════════════════════════════
# 离线运行
# ══════════════════════════════════════════════════════════════

class TestOfflineSupport(unittest.TestCase):
    """本地缓存与离线状态检测。"""

    def test_check_network_returns_bool(self):
        from tb_risk.ops.offline import check_network
        result = check_network(host="127.0.0.1", port=1, timeout=0.1)
        self.assertIsInstance(result, bool)

    def test_ensure_cached_structure(self):
        from tb_risk.ops.offline import ensure_cached
        cache = ensure_cached()
        for key in ("model_ready", "rule_ready", "model_count", "rules_file"):
            self.assertIn(key, cache)
        self.assertIsInstance(cache["model_ready"], bool)
        self.assertIsInstance(cache["rule_ready"], bool)
        self.assertIsInstance(cache["model_count"], int)

    def test_rules_ready_always_true(self):
        # 规则内嵌于 config.py，故核心始终可离线运行
        from tb_risk.ops.offline import rules_ready
        self.assertTrue(rules_ready())

    def test_status_structure(self):
        from tb_risk.ops.offline import status
        s = status()
        for key in ("online", "offline", "model_ready", "rule_ready",
                    "model_count", "rules_file", "data_dir",
                    "fully_offline_ready"):
            self.assertIn(key, s)
        self.assertIs(s["online"], not s["offline"])

    def test_status_text(self):
        from tb_risk.ops.offline import status_text
        text = status_text()
        self.assertIn("联网状态", text)
        self.assertIn("规则配置", text)


# ══════════════════════════════════════════════════════════════
# 热更新
# ══════════════════════════════════════════════════════════════

class TestInferFileKind(unittest.TestCase):
    """根据扩展名推断文件类型。"""

    def test_ml(self):
        from tb_risk.ops.hotupdate import infer_file_kind
        for name in ("model.joblib", "model.pkl", "model.pickle"):
            self.assertEqual(infer_file_kind(name), "ml")

    def test_gnn(self):
        from tb_risk.ops.hotupdate import infer_file_kind
        for name in ("model.pth", "model.pt"):
            self.assertEqual(infer_file_kind(name), "gnn")

    def test_rule(self):
        from tb_risk.ops.hotupdate import infer_file_kind
        self.assertEqual(infer_file_kind("rules.json"), "rule")

    def test_unknown(self):
        from tb_risk.ops.hotupdate import infer_file_kind
        self.assertIsNone(infer_file_kind("notes.txt"))


class TestRuleRegistry(unittest.TestCase):
    """运行时规则注册表。"""

    def test_set_get(self):
        from tb_risk.ops.hotupdate import RuleRegistry
        reg = RuleRegistry()
        self.assertIsNone(reg.get("threshold"))
        reg.set("threshold", 0.5)
        self.assertEqual(reg.get("threshold"), 0.5)
        self.assertEqual(reg.get("missing", "default"), "default")

    def test_load_json(self):
        from tb_risk.ops.hotupdate import RuleRegistry
        reg = RuleRegistry()
        v0 = reg.version
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False, encoding="utf-8") as f:
            json.dump({"risk_high": 15.0}, f)
            path = f.name
        try:
            result = reg.load_json(path)
        finally:
            os.remove(path)
        self.assertTrue(result["ok"])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["version"], v0 + 1)
        self.assertEqual(reg.get("risk_high"), 15.0)

    def test_load_json_non_object(self):
        from tb_risk.ops.hotupdate import RuleRegistry
        reg = RuleRegistry()
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False, encoding="utf-8") as f:
            f.write("[1, 2, 3]")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                reg.load_json(path)
        finally:
            os.remove(path)

    def test_clear(self):
        from tb_risk.ops.hotupdate import RuleRegistry
        reg = RuleRegistry()
        reg.set("a", 1)
        reg.clear()
        self.assertIsNone(reg.get("a"))


class TestHotUpdateWatcher(unittest.TestCase):
    """热更新目录监听器。"""

    def test_start_stop(self):
        from tb_risk.ops.hotupdate import HotUpdateWatcher
        with tempfile.TemporaryDirectory() as d:
            w = HotUpdateWatcher(watch_dir=d, auto_archive=False)
            self.assertTrue(w.start())
            self.assertTrue(w.is_running())
            w.stop()
            self.assertFalse(w.is_running())

    def test_scan_ml_with_loader(self):
        from tb_risk.ops.hotupdate import HotUpdateWatcher
        with tempfile.TemporaryDirectory() as d:
            loaded = []
            w = HotUpdateWatcher(
                watch_dir=d, auto_archive=False,
                ml_loader=lambda p: loaded.append(p) or True,
            )
            model_path = os.path.join(d, "model.joblib")
            with open(model_path, "w", encoding="utf-8") as f:
                f.write("dummy-model-bytes")
            results = w.scan_once()
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0]["ok"])
            self.assertEqual(results[0]["kind"], "ml")
            # 成功加载后源文件被移除，避免重复处理
            self.assertFalse(os.path.exists(model_path))
            self.assertEqual(len(loaded), 1)

    def test_scan_gnn_with_loader(self):
        from tb_risk.ops.hotupdate import HotUpdateWatcher
        with tempfile.TemporaryDirectory() as d:
            w = HotUpdateWatcher(
                watch_dir=d, auto_archive=False,
                gnn_loader=lambda p: True,
            )
            with open(os.path.join(d, "gnn.pth"), "w", encoding="utf-8") as f:
                f.write("dummy-gnn")
            results = w.scan_once()
            self.assertEqual(results[0]["kind"], "gnn")
            self.assertTrue(results[0]["ok"])

    def test_scan_rule(self):
        from tb_risk.ops.hotupdate import HotUpdateWatcher
        with tempfile.TemporaryDirectory() as d:
            w = HotUpdateWatcher(watch_dir=d, auto_archive=False)
            with open(os.path.join(d, "rules.json"), "w", encoding="utf-8") as f:
                json.dump({"beta": 0.3}, f)
            results = w.scan_once()
            self.assertEqual(results[0]["kind"], "rule")
            self.assertTrue(results[0]["ok"])
            self.assertEqual(w.rule_registry.get("beta"), 0.3)

    def test_scan_unknown_ext(self):
        from tb_risk.ops.hotupdate import HotUpdateWatcher
        with tempfile.TemporaryDirectory() as d:
            w = HotUpdateWatcher(watch_dir=d, auto_archive=False)
            with open(os.path.join(d, "notes.txt"), "w", encoding="utf-8") as f:
                f.write("hello")
            results = w.scan_once()
            self.assertEqual(len(results), 1)
            self.assertFalse(results[0]["ok"])

    def test_scan_loader_failure(self):
        from tb_risk.ops.hotupdate import HotUpdateWatcher
        with tempfile.TemporaryDirectory() as d:
            w = HotUpdateWatcher(
                watch_dir=d, auto_archive=False,
                ml_loader=lambda p: False,
            )
            with open(os.path.join(d, "bad.joblib"), "w", encoding="utf-8") as f:
                f.write("bad")
            results = w.scan_once()
            self.assertFalse(results[0]["ok"])

    def test_rule_same_name_reload(self):
        # 规则文件保留在目录中；同名但内容更新的文件应能再次热加载
        from tb_risk.ops.hotupdate import HotUpdateWatcher
        import time as _time
        with tempfile.TemporaryDirectory() as d:
            w = HotUpdateWatcher(watch_dir=d, auto_archive=False)
            path = os.path.join(d, "rules.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"beta": 0.3}, f)
            w.scan_once()
            self.assertEqual(w.rule_registry.get("beta"), 0.3)
            # 覆盖同名文件（更新内容），强制 mtime 变化
            _time.sleep(0.01)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"beta": 0.7}, f)
            w.scan_once()
            self.assertEqual(w.rule_registry.get("beta"), 0.7)
            self.assertEqual(len(w.history()), 2)

    def test_idempotent_scan(self):
        from tb_risk.ops.hotupdate import HotUpdateWatcher
        with tempfile.TemporaryDirectory() as d:
            w = HotUpdateWatcher(watch_dir=d, auto_archive=False)
            with open(os.path.join(d, "rules.json"), "w", encoding="utf-8") as f:
                json.dump({"a": 1}, f)
            w.scan_once()
            # 第二次扫描不应重复处理（源文件仍保留 for rule）
            w.scan_once()
            self.assertEqual(len(w.history()), 1)

    def test_history_cap(self):
        from tb_risk.ops.hotupdate import HotUpdateWatcher
        with tempfile.TemporaryDirectory() as d:
            w = HotUpdateWatcher(watch_dir=d, auto_archive=False)
            for i in range(210):
                name = f"file{i}.joblib"
                with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                    f.write("x")
            w.scan_once()
            self.assertLessEqual(len(w.history()), 200)


# ══════════════════════════════════════════════════════════════
# 部署打包
# ══════════════════════════════════════════════════════════════

class TestDeployPackaging(unittest.TestCase):
    """离线分发打包。"""

    def test_pack_distribution_creates_zip(self):
        from tb_risk.ops.deploy import pack_distribution
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "dist.zip")
            result = pack_distribution(out_path=out,
                                       include_models=False,
                                       include_config=False)
            self.assertEqual(result, out)
            self.assertTrue(os.path.exists(out))
            self.assertTrue(zipfile.is_zipfile(out))

    def test_pack_includes_source(self):
        from tb_risk.ops.deploy import PROJECT_ROOT, pack_distribution
        # 白名单前提是"源码检出布局"（requirements.lock/Dockerfile 位于
        # 项目根）。site-packages 安装布局（Docker 普通安装）下这些发行
        # 层文件不在包内，打包产物合法地不含它们——跳过而非报错
        if not os.path.exists(os.path.join(PROJECT_ROOT, 'requirements.lock')):
            self.skipTest('非源码检出布局（site-packages 安装），'
                          '打包白名单前提不成立')
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "dist.zip")
            pack_distribution(out_path=out, include_models=False,
                              include_config=False)
            with zipfile.ZipFile(out) as zf:
                names = [n.replace('\\', '/') for n in zf.namelist()]
            # 源码包（ops 模块）应被包含
            self.assertTrue(any("ops/errors.py" in n for n in names))
            self.assertTrue(any("requirements.lock" in n for n in names))
            self.assertTrue(any("Dockerfile" in n for n in names))


if __name__ == "__main__":
    unittest.main()
