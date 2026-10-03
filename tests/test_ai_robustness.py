#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — AI 健壮性改进 (缺陷 #6/#7/#8/#9/#17)

覆盖：
- #6: make_client 按 config 哈希缓存 + LocalLLMClient _load_model 加 threading.Lock
- #7: tenacity wait_exponential max 提升到 ≥10s
- #8: reporter.generate_report 调用 chat 时显式传 max_tokens=2048
- #9: reporter._build_context 体积约束（SEIR 关键标量 + top_features 前5 + 接触者前20）
- #17: _validate_schema 支持嵌套结构（list 内 dict 元素验证）
"""

import inspect
import os
import sys
import threading
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import tenacity
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

try:
    import transformers  # noqa: F401
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

AI_DEPS_AVAILABLE = HTTPX_AVAILABLE and TENACITY_AVAILABLE


# ==============================================================================
# #6: make_client 按 config 哈希缓存
# ==============================================================================

@unittest.skipUnless(AI_DEPS_AVAILABLE, "httpx/tenacity not installed")
class TestMakeClientCaching(unittest.TestCase):
    """make_client() 按 config 哈希缓存"""

    def setUp(self):
        """每个测试前清空缓存，避免相互干扰"""
        from tb_risk.ai import client as client_module
        if hasattr(client_module, '_clear_client_cache'):
            client_module._clear_client_cache()

    def tearDown(self):
        from tb_risk.ai import client as client_module
        if hasattr(client_module, '_clear_client_cache'):
            client_module._clear_client_cache()

    def test_same_config_returns_same_instance(self):
        """相同 config（相同字段值） → 返回同一实例"""
        from tb_risk.ai.client import make_client
        from tb_risk.ai.config import AIConfig
        cfg1 = AIConfig(api_key='sk-test-1', model='m1')
        cfg2 = AIConfig(api_key='sk-test-1', model='m1')
        c1 = make_client(cfg1)
        c2 = make_client(cfg2)
        self.assertIsNotNone(c1)
        self.assertIs(c1, c2, "相同 config 应命中缓存返回同一实例")

    def test_different_api_key_returns_different_instance(self):
        """不同 api_key → 不同实例"""
        from tb_risk.ai.client import make_client
        from tb_risk.ai.config import AIConfig
        cfg1 = AIConfig(api_key='sk-test-1')
        cfg2 = AIConfig(api_key='sk-test-2')
        c1 = make_client(cfg1)
        c2 = make_client(cfg2)
        self.assertIsNot(c1, c2)

    def test_different_model_returns_different_instance(self):
        """不同 model → 不同实例"""
        from tb_risk.ai.client import make_client
        from tb_risk.ai.config import AIConfig
        cfg1 = AIConfig(api_key='sk-test', model='m1')
        cfg2 = AIConfig(api_key='sk-test', model='m2')
        c1 = make_client(cfg1)
        c2 = make_client(cfg2)
        self.assertIsNot(c1, c2)

    def test_local_vs_cloud_returns_different_instance(self):
        """本地模式 vs 云端模式 → 不同实例"""
        from tb_risk.ai.client import make_client
        from tb_risk.ai.config import AIConfig
        cfg_cloud = AIConfig(api_key='sk-test')
        cfg_local = AIConfig(local_model_path='/path/to/model')
        c1 = make_client(cfg_cloud)
        c2 = make_client(cfg_local)
        self.assertIsNot(c1, c2)

    def test_unconfigured_returns_none_not_cached(self):
        """未配置 → None（不缓存，便于下次配置变更后重试）"""
        from tb_risk.ai.client import make_client
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig()  # 无 api_key 无 local
        c1 = make_client(cfg)
        c2 = make_client(cfg)
        self.assertIsNone(c1)
        self.assertIsNone(c2)

    def test_clear_cache_forces_new_instance(self):
        """清空缓存后强制创建新实例"""
        from tb_risk.ai import client as client_module
        from tb_risk.ai.client import make_client
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-test')
        c1 = make_client(cfg)
        client_module._clear_client_cache()
        c2 = make_client(cfg)
        self.assertIsNot(c1, c2)

    def test_cache_threadsafe_under_concurrent_calls(self):
        """并发调用 make_client 不应崩溃且应共享同一实例"""
        from tb_risk.ai.client import make_client
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-concurrent')
        results = []
        errors = []

        def worker():
            try:
                results.append(make_client(cfg))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 8)
        # 全部应为同一实例
        first = results[0]
        for r in results:
            self.assertIs(r, first)


# ==============================================================================
# #6: LocalLLMClient _load_model 加 threading.Lock
# ==============================================================================

class TestLocalLLMClientLoadModelLock(unittest.TestCase):
    """LocalLLMClient._load_model 使用 threading.Lock 保护"""

    def test_load_model_uses_lock(self):
        """_load_model 源码中应使用 self._load_lock 或 threading.Lock"""
        from tb_risk.ai.client import LocalLLMClient
        src = inspect.getsource(LocalLLMClient._load_model)
        # 必须显式 acquire/release（with self._load_lock: 或 with self._lock:）
        self.assertTrue(
            'with self.' in src and ('lock' in src.lower()),
            "_load_model 必须使用 with self.xxx_lock 保护并发加载"
        )

    def test_local_client_has_lock_attribute(self):
        """LocalLLMClient 实例应有 _load_lock（或类似锁）属性"""
        from tb_risk.ai.client import LocalLLMClient
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(local_model_path='/path/to/model')
        client = LocalLLMClient(cfg)
        # 找到任一 *lock* 属性
        lock_attrs = [a for a in dir(client) if 'lock' in a.lower()]
        self.assertTrue(lock_attrs, f"LocalLLMClient 应有锁属性，实际：{dir(client)}")

    def test_lock_is_threading_lock_type(self):
        """锁属性应是 threading.Lock 或 RLock 类型"""
        import threading as _threading
        from tb_risk.ai.client import LocalLLMClient
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(local_model_path='/path/to/model')
        client = LocalLLMClient(cfg)
        lock_attrs = [a for a in dir(client) if 'lock' in a.lower()]
        self.assertTrue(lock_attrs, "LocalLLMClient 应有锁属性")
        lock_value = getattr(client, lock_attrs[0])
        # threading.Lock() 返回 _thread.lock 类型；RLock 返回 _thread.RLock
        self.assertTrue(
            hasattr(lock_value, 'acquire') and hasattr(lock_value, 'release'),
            f"锁属性应是 threading.Lock 类型（有 acquire/release 方法），实际：{type(lock_value)}"
        )

    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers not installed")
    def test_concurrent_load_model_calls_serialized(self):
        """并发 _load_model 调用应串行化（仅真正加载一次）

        通过 mock AutoModelForCausalLM.from_pretrained 计数，
        并发触发 chat() 多次，from_pretrained 应仅被调用 1 次。
        """
        from tb_risk.ai.client import LocalLLMClient
        from tb_risk.ai.config import AIConfig

        # 不真正加载模型，用 mock 替换
        cfg = AIConfig(local_model_path='/fake/path')

        load_count = {'n': 0}
        load_lock = threading.Lock()

        class _FakeModel:
            def eval(self):
                return self

            def to(self, _device):
                return self

            def generate(self, **kwargs):
                # 返回 input_ids 长度的伪输出
                input_ids = kwargs.get('input_ids')
                if hasattr(input_ids, 'shape'):
                    return input_ids
                return [[0]]

        class _FakeTokenizer:
            eos_token_id = 0

            def apply_chat_template(self, messages, **kwargs):
                return "fake prompt"

            def __call__(self, text, **kwargs):
                # 模拟 torch tokenizer 输出
                class _Inputs(dict):
                    pass
                return _Inputs(input_ids=[[1, 2, 3]])

            def decode(self, tokens, **kwargs):
                return "fake response"

        def fake_from_pretrained(*args, **kwargs):
            with load_lock:
                load_count['n'] += 1
            # 模拟加载耗时，让其他线程有机会进入 _load_model
            import time
            time.sleep(0.05)
            return _FakeModel()

        with mock.patch('transformers.AutoModelForCausalLM.from_pretrained',
                        side_effect=fake_from_pretrained), \
                mock.patch('transformers.AutoTokenizer.from_pretrained',
                           return_value=_FakeTokenizer()):
            client = LocalLLMClient(cfg)

            def worker():
                try:
                    client._load_model()
                except Exception:
                    pass  # 测试关注加载次数，不关注其他错误

            threads = [threading.Thread(target=worker) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # 5 个线程并发，但真正加载应仅 1 次（其他被锁阻塞后看到 _model 已存在直接返回）
        self.assertEqual(load_count['n'], 1,
                         f"并发 _load_model 应仅加载 1 次，实际：{load_count['n']} 次")


# ==============================================================================
# #7: tenacity wait_exponential max ≥ 10s
# ==============================================================================

@unittest.skipUnless(TENACITY_AVAILABLE, "tenacity not installed")
class TestRetryBackoffMax(unittest.TestCase):
    """tenacity wait_exponential 上限提升"""

    def test_make_retryer_max_wait_at_least_10s(self):
        """_make_retryer 的 wait_exponential max 应 ≥ 10s

        原 max=0.5s 对 429 限速恢复过短，重试 3 次仍会失败。
        """
        from tb_risk.ai.client import LLMClient
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-test')
        client = LLMClient(cfg)
        retryer = client._make_retryer()
        # tenacity Retrying 暴露 wait 属性
        wait_obj = retryer.wait
        # wait_exponential 内部有 max_value 属性
        # 不同 tenacity 版本字段名可能是 'max' 或 'max_value'
        max_value = getattr(wait_obj, 'max', None) or getattr(wait_obj, 'max_value', None)
        self.assertIsNotNone(max_value, f"无法从 wait 对象提取 max：{wait_obj}")
        self.assertGreaterEqual(max_value, 10,
                                f"wait_exponential max 应 ≥ 10s，实际：{max_value}")


# ==============================================================================
# #8: reporter 调用 chat 时显式传 max_tokens=2048
# ==============================================================================

class TestReporterMaxTokens(unittest.TestCase):
    """reporter.generate_report 显式传 max_tokens"""

    def test_generate_report_passes_max_tokens_to_chat(self):
        """generate_report 调用 client.chat 时应传 max_tokens=2048"""
        from tb_risk.ai.reporter import generate_report

        captured = {}

        class _Client:
            def chat(self, messages, **kwargs):
                captured['kwargs'] = kwargs
                return "报告内容"

        result = {
            'patient_score': 75.0,
            'summary': {'overall_risk': '高'},
        }
        report = generate_report(result, client=_Client())
        self.assertEqual(report, "报告内容")
        self.assertIn('max_tokens', captured['kwargs'],
                      "generate_report 应显式传 max_tokens")
        self.assertGreaterEqual(captured['kwargs']['max_tokens'], 2048,
                                f"max_tokens 应 ≥ 2048，实际：{captured['kwargs']['max_tokens']}")


# ==============================================================================
# #9: reporter._build_context 体积约束
# ==============================================================================

class TestBuildContextSizeConstraints(unittest.TestCase):
    """_build_context 对上下文体积做约束"""

    def _make_large_result(self):
        """构造一个含大量数据的 result

        接触者 risk_score 用 100+i 整数（避免脱敏后名字无法匹配，
        通过 risk_score=XXX 在上下文中识别）。
        """
        # 50 个接触者
        contacts = [
            {'name': f'接触者{i}', 'risk_score': 100 + i,
             'priority': '高' if i < 10 else '中',
             'infection_probability': 30.0 + i}
            for i in range(50)
        ]
        # 20 个 top_features
        top_features = [
            {'feature': f'feat_{i}', 'importance': 0.1 - i * 0.005}
            for i in range(20)
        ]
        # 含 52 周时间序列的 SEIR 结果
        seir_with_timeseries = {
            'R0': 2.3,
            'peak_time': 45,
            'peak_size': 100,
            'weekly_incidence': [10 + i for i in range(52)],  # 52 周时间序列
            'weekly_prevalence': [50 + i for i in range(52)],
        }
        return {
            'patient_score': 75.0,
            'summary': {'overall_risk': '高', 'total_contacts': 50},
            'potential_patients': {
                'family': contacts[:30],
                'social': contacts[30:],
            },
            'ml_results': {
                'probability': 0.72,
                'top_features': top_features,
            },
            'seir_results': seir_with_timeseries,
        }

    def test_top_features_limited_to_5(self):
        """top_features 仅保留前 5 个"""
        from tb_risk.ai.reporter import _build_context
        ctx = _build_context(self._make_large_result())
        # 应包含前 5 个 feature 名
        for i in range(5):
            self.assertIn(f'feat_{i}', ctx)
        # 不应包含第 6 个之后的 feature 名
        for i in range(5, 20):
            self.assertNotIn(f'feat_{i}', ctx,
                             f"top_features 应仅保留前 5 个，但出现了 feat_{i}")

    def test_contacts_limited_to_20(self):
        """接触者列表仅展示前 20 条（按 risk_score=100+i 识别）"""
        from tb_risk.ai.reporter import _build_context
        ctx = _build_context(self._make_large_result())
        # 前 20 个接触者应出现（risk_score=100..119）
        for i in range(20):
            self.assertIn(f'风险评分={100 + i}', ctx,
                         f"接触者 {i} (risk_score={100+i}) 应在上下文中")
        # 第 21 个之后的接触者不应出现
        for i in range(20, 50):
            self.assertNotIn(f'风险评分={100 + i}', ctx,
                             f"接触者 {i} (risk_score={100+i}) 不应在上下文中（最多展示前 20 条）")

    def test_seir_excludes_timeseries(self):
        """SEIR 上下文应排除时间序列（weekly_incidence/weekly_prevalence）"""
        from tb_risk.ai.reporter import _build_context
        ctx = _build_context(self._make_large_result())
        # 时间序列不应出现在上下文中
        self.assertNotIn('weekly_incidence', ctx,
                         "上下文不应包含 weekly_incidence 时间序列")
        self.assertNotIn('weekly_prevalence', ctx,
                         "上下文不应包含 weekly_prevalence 时间序列")
        # 但应保留关键标量
        self.assertIn('R0', ctx)
        self.assertIn('peak_time', ctx)
        self.assertIn('peak_size', ctx)

    def test_seir_only_key_scalars_preserved(self):
        """SEIR 仅保留 R0/peak_time/peak_size 等关键标量"""
        from tb_risk.ai.reporter import _build_context
        result = {
            'patient_score': 75.0,
            'seir_results': {
                'R0': 2.3,
                'peak_time': 45,
                'peak_size': 100,
                'weekly_incidence': list(range(52)),
                'compartment_sizes': {'S': 1000, 'I': 50, 'R': 100},
                'long_field_name_that_should_be_excluded': 'x' * 200,
            }
        }
        ctx = _build_context(result)
        # 关键标量保留
        self.assertIn('2.3', ctx)
        self.assertIn('R0', ctx)
        # 时间序列与嵌套结构排除
        self.assertNotIn('weekly_incidence', ctx)
        self.assertNotIn('compartment_sizes', ctx)


# ==============================================================================
# #17: _validate_schema 支持嵌套结构
# ==============================================================================

class TestValidateSchemaNested(unittest.TestCase):
    """_validate_schema 支持嵌套结构"""

    def test_validates_flat_schema(self):
        """平面 schema 仍正常工作（向后兼容）"""
        from tb_risk.ai.client import _validate_schema
        schema = {'name': str, 'age': int}
        self.assertTrue(_validate_schema({'name': '张三', 'age': 35}, schema))
        self.assertFalse(_validate_schema({'name': '张三'}, schema))

    def test_validates_nested_dict_field(self):
        """支持嵌套 dict 字段验证"""
        from tb_risk.ai.client import _validate_schema
        # schema 中某字段是 dict → 表示该字段是嵌套 dict，需验证子结构
        schema = {
            'name': str,
            'meta': {'id': int, 'category': str},
        }
        # 合法
        self.assertTrue(_validate_schema(
            {'name': '张三', 'meta': {'id': 1, 'category': 'A'}}, schema))
        # meta 缺字段
        self.assertFalse(_validate_schema(
            {'name': '张三', 'meta': {'id': 1}}, schema))
        # meta 类型错误
        self.assertFalse(_validate_schema(
            {'name': '张三', 'meta': 'not a dict'}, schema))

    def test_validates_list_of_dicts(self):
        """支持 list[dict] 嵌套验证

        schema 中某字段是 list → 列表元素需符合 list[0] 描述的结构
        （list[0] 是 dict 视为元素 schema；list[0] 是 type 视为元素类型）
        """
        from tb_risk.ai.client import _validate_schema
        schema = {
            'issues': [{'field': str, 'problem': str}],
        }
        # 合法：list 内每个 dict 含 field/problem
        self.assertTrue(_validate_schema(
            {'issues': [{'field': 'age', 'problem': '缺失'},
                        {'field': 'bmi', 'problem': '异常'}]},
            schema))
        # 合法：空 list
        self.assertTrue(_validate_schema({'issues': []}, schema))
        # 不合法：list 内元素缺字段
        self.assertFalse(_validate_schema(
            {'issues': [{'field': 'age'}]}, schema))  # 缺 problem
        # 不合法：list 内元素非 dict
        self.assertFalse(_validate_schema(
            {'issues': ['not a dict']}, schema))

    def test_validates_list_of_scalars(self):
        """支持 list[type] 简单类型列表"""
        from tb_risk.ai.client import _validate_schema
        schema = {
            'tags': [str],
        }
        self.assertTrue(_validate_schema({'tags': ['a', 'b']}, schema))
        self.assertTrue(_validate_schema({'tags': []}, schema))
        self.assertFalse(_validate_schema({'tags': ['a', 1]}, schema))  # 混入 int
        self.assertFalse(_validate_schema({'tags': 'not a list'}, schema))

    def test_validates_optional_nested(self):
        """Optional + 嵌套（嵌套 dict 或 None）"""
        from tb_risk.ai.client import _validate_schema
        schema = {
            'meta': ({'id': int}, type(None)),  # meta 是 dict 或 None
        }
        self.assertTrue(_validate_schema({'meta': {'id': 1}}, schema))
        self.assertTrue(_validate_schema({'meta': None}, schema))
        self.assertFalse(_validate_schema({'meta': 'str'}, schema))
        self.assertFalse(_validate_schema({'meta': {'id': 'x'}}, schema))


if __name__ == '__main__':
    unittest.main()
