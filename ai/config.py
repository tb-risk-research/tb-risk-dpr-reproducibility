#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI 配置模块：环境变量、提供商预设、用户配置文件。

Layer 2-3a: 配置基础设施。

支持的提供商（均走 OpenAI 兼容协议）：
- deepseek:  https://api.deepseek.com/v1                默认
- openai:    https://api.openai.com/v1
- dashscope: https://dashscope.aliyuncs.com/compatible-mode/v1  (通义千问)
- zhipu:     https://open.bigmodel.cn/api/paas/v4       (智谱 GLM)
- moonshot:  https://api.moonshot.cn/v1                 (Kimi)

环境变量：
- TB_AI_PROVIDER:       提供商名（不识别时回退到 deepseek）
- TB_AI_API_KEY:        API Key（云端模式）
- TB_AI_BASE_URL:       覆盖预设 base_url（私有部署兼容）
- TB_AI_MODEL:          覆盖预设 model
- TB_AI_TIMEOUT:        请求超时秒数（默认 60）
- TB_AI_MAX_RETRIES:    最大重试次数（默认 3）
- TB_AI_TEMPERATURE:    采样温度（默认 0.2，抽取任务偏向确定性）
- TB_AI_LOCAL_MODEL:    本地模型路径（启用本地模式，敏感场景）
- TB_AI_LOCAL_DEVICE:   本地推理设备（默认 cpu）

用户配置文件：~/.tb_risk/ai_config.json
- 环境变量优先级高于配置文件（仅覆盖已设置的字段）
- API Key 不写入版本控制（.gitignore 已加入 ai_config.json）

安全：API Key 不会出现在 repr()/str()/to_dict() 输出中，避免日志泄露。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

LOGGER = logging.getLogger("tb_risk.ai.config")

# 用户配置文件默认路径（与 gui/app_config.py CONFIG_DIR 一致）
CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.tb_risk')
DEFAULT_AI_CONFIG_FILE = os.path.join(CONFIG_DIR, 'ai_config.json')


# ==============================================================================
# 提供商预设（OpenAI 兼容协议）
# ==============================================================================

PROVIDERS: Dict[str, Dict[str, str]] = {
    'deepseek': {
        'base_url': 'https://api.deepseek.com/v1',
        'model': 'deepseek-chat',
    },
    'openai': {
        'base_url': 'https://api.openai.com/v1',
        'model': 'gpt-4o-mini',
    },
    'dashscope': {
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'model': 'qwen-plus',
    },
    'zhipu': {
        'base_url': 'https://open.bigmodel.cn/api/paas/v4',
        'model': 'glm-4-flash',
    },
    'moonshot': {
        'base_url': 'https://api.moonshot.cn/v1',
        'model': 'moonshot-v1-8k',
    },
}

# 默认提供商（用户未指定时）
DEFAULT_PROVIDER = 'deepseek'


# ==============================================================================
# 工具函数
# ==============================================================================

def _safe_int(val: Any, default: int) -> int:
    """安全解析 int，失败返回 default"""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val: Any, default: float) -> float:
    """安全解析 float，失败返回 default"""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _resolve_provider(name: Optional[str]) -> str:
    """解析 provider 名称，未知时回退到 deepseek"""
    if not name or name not in PROVIDERS:
        if name and name not in PROVIDERS:
            LOGGER.warning("未知 AI 提供商 %r，回退到 %r", name, DEFAULT_PROVIDER)
        return DEFAULT_PROVIDER
    return name


def _mask_api_key(key: Optional[str]) -> Optional[str]:
    """API Key 掩码：仅保留首尾 2 字符，中间用 *** 替换"""
    if not key:
        return None
    if len(key) <= 4:
        return '***'
    return f"{key[:2]}***{key[-2:]}"


# ==============================================================================
# 配置字典字段类型校验（缺陷 #9）
# ==============================================================================
# 字段名 -> (允许的 Python 类型元组, 类型描述（用于错误信息）, 是否允许 None)
# bool 不在 int 字段的允许类型中：虽然 isinstance(True, int) 为 True，
# 但语义上 bool 不应作为 timeout/max_retries 使用（避免 True 被当作 1）。
_CONFIG_FIELD_TYPES: Dict[str, Tuple[Tuple[type, ...], str, bool]] = {
    'provider':         ((str,), 'str', False),
    'api_key':          ((str,), 'str or None', True),
    'base_url':         ((str,), 'str', False),
    'model':            ((str,), 'str', False),
    'timeout':          ((int,), 'int', False),
    'max_retries':      ((int,), 'int', False),
    'temperature':      ((int, float), 'int or float', False),
    'local_model_path': ((str,), 'str or None', True),
    'local_device':     ((str,), 'str', False),
    'prompts_dir':      ((str,), 'str or None', True),
}


def _validate_config_dict(data: Dict[str, Any]) -> None:
    """校验配置字典字段类型，类型不符时抛出 TypeError

    仅校验 data 中存在的字段，未提供的字段使用默认值（向后兼容部分配置）。
    用于 from_user_config_file 阶段拦截用户手写配置文件中的类型错误
    （如 max_retries: "3" 字符串），避免被 _safe_int 静默回退到默认值。

    Args:
        data: 配置字典（通常是 json.load 的结果）

    Raises:
        TypeError: 当字段类型不符时，错误信息包含字段名、期望类型与实际类型
    """
    for field, (allowed_types, type_desc, allow_none) in _CONFIG_FIELD_TYPES.items():
        if field not in data:
            continue
        val = data[field]
        if val is None:
            if not allow_none:
                raise TypeError(
                    f"AI 配置字段 {field!r} 不允许为 None，期望 {type_desc}"
                )
            continue
        # bool 排除（bool 是 int 子类，但语义上不应作为数值字段使用）
        if isinstance(val, bool) and bool not in allowed_types:
            raise TypeError(
                f"AI 配置字段 {field!r} 不接受 bool，期望 {type_desc}"
            )
        if not isinstance(val, allowed_types):
            raise TypeError(
                f"AI 配置字段 {field!r} 类型错误，"
                f"期望 {type_desc}，实际为 {type(val).__name__}"
            )


# ==============================================================================
# prompts_dir 路径安全校验（缺陷 #10）
# ==============================================================================
# 系统目录黑名单（跨平台）：prompts_dir 不允许指向这些目录或其子目录
def _build_forbidden_dirs():
    """构建系统目录黑名单（按平台）"""
    if os.name == 'nt':
        # Windows: 系统根、Program Files、ProgramData
        system_root = os.environ.get('SystemRoot', r'C:\Windows')
        return [
            Path(system_root),
            Path(r'C:\Program Files'),
            Path(r'C:\Program Files (x86)'),
            Path(r'C:\ProgramData'),
        ]
    # Unix-like: 标准系统目录
    return [
        Path('/etc'), Path('/usr'), Path('/bin'), Path('/sbin'),
        Path('/var'), Path('/sys'), Path('/proc'),
        Path('/boot'), Path('/dev'),
    ]


_FORBIDDEN_DIRS = _build_forbidden_dirs()

# 敏感目录名（出现在路径任意层级都拒绝）
_SENSITIVE_DIR_NAMES = {'.ssh', '.gnupg'}


def _is_path_under(path: Path, parent: Path) -> bool:
    """检查 path 是否在 parent 目录下（含 parent 本身）

    用 os.path.commonpath 而非字符串前缀匹配，避免 /etc 与 /etc-passed 误判。
    """
    try:
        # resolve 后再比较，避免符号链接和相对路径绕过
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_prompts_dir(path: Optional[str]) -> Optional[str]:
    """校验 prompts_dir 路径安全性

    允许：
    - None 或空字符串（视为未设置）
    - 用户主目录下的子目录
    - 当前工作目录（项目目录）下的子目录
    - 任意不在系统黑名单中的路径

    拒绝：
    - 路径遍历（Path.parts 含 '..'）
    - 系统目录（Windows: C:\\Windows、C:\\Program Files 等；
      Unix: /etc、/usr、/bin 等）
    - 敏感目录（.ssh、.gnupg）

    Args:
        path: prompts_dir 路径（None、相对路径或绝对路径）

    Returns:
        原样返回 path（None 或非空字符串），不强制绝对化

    Raises:
        ValueError: 当路径指向系统/敏感目录或包含路径遍历时
    """
    if not path:
        return None

    # 1. 路径遍历检查（相对路径中的 .. 不应允许）
    if '..' in Path(path).parts:
        raise ValueError(
            f"prompts_dir {path!r} 包含路径遍历（..），出于安全考虑被拒绝"
        )

    # 2. 解析为绝对路径（展开 ~ 和符号链接）
    abs_path = Path(path).expanduser().resolve()

    # 3. 系统目录黑名单检查
    for forbidden in _FORBIDDEN_DIRS:
        if _is_path_under(abs_path, forbidden):
            raise ValueError(
                f"prompts_dir {path!r} 指向系统目录 "
                f"{str(forbidden)!r}，出于安全考虑被拒绝"
            )

    # 4. 敏感目录检查（.ssh、.gnupg 等出现在路径任意层级都拒绝）
    for part in abs_path.parts:
        if part in _SENSITIVE_DIR_NAMES:
            raise ValueError(
                f"prompts_dir {path!r} 指向敏感目录 {part!r}，"
                f"出于安全考虑被拒绝"
            )

    return path


# ==============================================================================
# AIConfig 数据类
# ==============================================================================

@dataclass
class AIConfig:
    """AI 服务配置（dataclass，与项目 schemas.py 风格一致，不引入 pydantic）"""
    provider: str = DEFAULT_PROVIDER
    api_key: Optional[str] = None
    base_url: str = PROVIDERS[DEFAULT_PROVIDER]['base_url']
    model: str = PROVIDERS[DEFAULT_PROVIDER]['model']
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.2
    local_model_path: Optional[str] = None
    local_device: str = 'cpu'
    # 缺陷 #15：提示词外置目录（None 表示不使用外置提示词，使用 prompts.py 默认常量）
    # 设为 ~/.tb_risk/prompts/ 时，prompts.py 的 get_prompt() 会优先读取该目录下的 .txt 文件
    prompts_dir: Optional[str] = None

    @property
    def is_local(self) -> bool:
        """是否启用本地模型模式（敏感场景：数据不出本机）"""
        return self.local_model_path is not None

    def is_configured(self) -> bool:
        """是否已配置（有 api_key 或本地模型路径）"""
        return self.api_key is not None or self.local_model_path is not None

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> 'AIConfig':
        """从环境变量构造配置

        Args:
            env: 自定义环境变量字典（测试用），None 时读取 os.environ
        """
        if env is None:
            env = os.environ

        provider = _resolve_provider(env.get('TB_AI_PROVIDER'))
        preset = PROVIDERS[provider]
        # 缺陷 #10：prompts_dir 路径安全校验（拦截系统目录/路径遍历/.ssh）
        prompts_dir = _validate_prompts_dir(env.get('TB_AI_PROMPTS_DIR'))
        return cls(
            provider=provider,
            api_key=env.get('TB_AI_API_KEY'),
            base_url=env.get('TB_AI_BASE_URL', preset['base_url']),
            model=env.get('TB_AI_MODEL', preset['model']),
            timeout=_safe_int(env.get('TB_AI_TIMEOUT'), 60),
            max_retries=_safe_int(env.get('TB_AI_MAX_RETRIES'), 3),
            temperature=_safe_float(env.get('TB_AI_TEMPERATURE'), 0.2),
            local_model_path=env.get('TB_AI_LOCAL_MODEL'),
            local_device=env.get('TB_AI_LOCAL_DEVICE', 'cpu'),
            prompts_dir=prompts_dir,
        )

    @classmethod
    def from_user_config_file(cls, path: Optional[str] = None) -> 'AIConfig':
        """从用户配置文件加载（环境变量覆盖已设置的字段）

        Args:
            path: 配置文件路径，None 时使用 DEFAULT_AI_CONFIG_FILE
        """
        path = path or DEFAULT_AI_CONFIG_FILE
        file_data: Dict[str, Any] = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            if not isinstance(file_data, dict):
                LOGGER.warning("AI 配置文件 %s 顶层非 dict，忽略", path)
                file_data = {}
        except FileNotFoundError:
            file_data = {}
        except (json.JSONDecodeError, OSError) as e:
            LOGGER.warning("AI 配置文件 %s 损坏，忽略: %s", path, e)
            file_data = {}

        # 1. 从文件构建基础配置
        if file_data:
            # 缺陷 #9：在加载前校验字段类型，拦截用户手写配置文件的类型错误
            # （如 max_retries: "3" 字符串），避免被 _safe_int 静默回退到默认值
            _validate_config_dict(file_data)
            provider = _resolve_provider(file_data.get('provider'))
            preset = PROVIDERS[provider]
            cfg = cls(
                provider=provider,
                api_key=file_data.get('api_key'),
                base_url=file_data.get('base_url', preset['base_url']),
                model=file_data.get('model', preset['model']),
                timeout=_safe_int(file_data.get('timeout'), 60),
                max_retries=_safe_int(file_data.get('max_retries'), 3),
                temperature=_safe_float(file_data.get('temperature'), 0.2),
                local_model_path=file_data.get('local_model_path'),
                local_device=file_data.get('local_device', 'cpu'),
                # 缺陷 #10：prompts_dir 路径安全校验（拦截系统目录/路径遍历/.ssh）
                prompts_dir=_validate_prompts_dir(file_data.get('prompts_dir')),
            )
        else:
            cfg = cls()

        # 2. 环境变量覆盖（仅覆盖已设置的字段）
        env = os.environ
        if 'TB_AI_PROVIDER' in env:
            p = _resolve_provider(env['TB_AI_PROVIDER'])
            if p != cfg.provider:
                cfg.provider = p
                # 切换提供商时，base_url/model 重置为新提供商的预设值
                # （除非环境变量也显式覆盖了它们）
                preset = PROVIDERS[p]
                cfg.base_url = preset['base_url']
                cfg.model = preset['model']
        if 'TB_AI_API_KEY' in env:
            cfg.api_key = env['TB_AI_API_KEY']
        if 'TB_AI_BASE_URL' in env:
            cfg.base_url = env['TB_AI_BASE_URL']
        if 'TB_AI_MODEL' in env:
            cfg.model = env['TB_AI_MODEL']
        if 'TB_AI_TIMEOUT' in env:
            cfg.timeout = _safe_int(env['TB_AI_TIMEOUT'], cfg.timeout)
        if 'TB_AI_MAX_RETRIES' in env:
            cfg.max_retries = _safe_int(env['TB_AI_MAX_RETRIES'], cfg.max_retries)
        if 'TB_AI_TEMPERATURE' in env:
            cfg.temperature = _safe_float(env['TB_AI_TEMPERATURE'], cfg.temperature)
        if 'TB_AI_LOCAL_MODEL' in env:
            cfg.local_model_path = env['TB_AI_LOCAL_MODEL']
        if 'TB_AI_LOCAL_DEVICE' in env:
            cfg.local_device = env['TB_AI_LOCAL_DEVICE']
        if 'TB_AI_PROMPTS_DIR' in env:
            # 缺陷 #10：环境变量覆盖的 prompts_dir 也要校验路径安全
            cfg.prompts_dir = _validate_prompts_dir(env['TB_AI_PROMPTS_DIR'])

        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 dict（api_key 被掩码，可安全日志/打印）"""
        return {
            'provider': self.provider,
            'api_key': _mask_api_key(self.api_key),
            'base_url': self.base_url,
            'model': self.model,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'temperature': self.temperature,
            'local_model_path': self.local_model_path,
            'local_device': self.local_device,
            'prompts_dir': self.prompts_dir,
        }

    def save_to_file(self, path: Optional[str] = None) -> str:
        """将配置保存到 JSON 文件（原子写入，api_key 明文保存）

        Layer 1: GUI AISettingsDialog 确定后调用此方法回写配置到
        ~/.tb_risk/ai_config.json，供 AI 客户端读取使用。

        与 to_dict() 的关键差异：api_key 明文保存（不掩码），因为该文件
        供 AI 客户端读取使用，掩码后无法用于鉴权。文件本身不进入版本控制
        （.gitignore 已加入 ai_config.json）。

        保存前校验 prompts_dir 路径安全（与 from_user_config_file 一致），
        校验失败时抛出 ValueError 且不写入文件，保持原文件不变。

        Args:
            path: 配置文件路径，None 时使用 DEFAULT_AI_CONFIG_FILE

        Returns:
            实际写入的文件路径

        Raises:
            ValueError: prompts_dir 校验失败（路径遍历/系统目录/敏感目录）
            OSError: 文件写入失败
        """
        save_path = path or DEFAULT_AI_CONFIG_FILE

        # 1. 校验 prompts_dir 路径安全（与 from_user_config_file 一致）
        validated_prompts_dir = _validate_prompts_dir(self.prompts_dir)

        # 2. 构造保存字典（api_key 明文，不掩码）
        data = {
            'provider': self.provider,
            'api_key': self.api_key,
            'base_url': self.base_url,
            'model': self.model,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'temperature': self.temperature,
            'local_model_path': self.local_model_path,
            'local_device': self.local_device,
            'prompts_dir': validated_prompts_dir,
        }

        # 3. 类型校验（防御性，dataclass 已保证类型，但保持与加载路径一致）
        _validate_config_dict(data)

        # 4. 原子写入：先写临时文件再 rename，防止写入中途崩溃损坏原文件
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=save_dir or None, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # Windows 上 os.replace 可原子性覆盖已存在文件
            os.replace(tmp_path, save_path)
            # 设置文件权限：仅所有者可读写（Unix: 0o600；Windows 上 os.chmod 仅影响只读标志）
            try:
                os.chmod(save_path, 0o600)
            except OSError:
                pass  # 部分平台不支持 chmod，忽略
        except Exception:
            # 清理临时文件，避免残留
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        return save_path

    def __repr__(self) -> str:
        """repr 不泄露 api_key"""
        d = self.to_dict()
        return (f"AIConfig(provider={d['provider']!r}, "
                f"api_key={d['api_key']!r}, "
                f"base_url={d['base_url']!r}, model={d['model']!r}, "
                f"timeout={d['timeout']}, max_retries={d['max_retries']}, "
                f"temperature={d['temperature']}, "
                f"local_model_path={d['local_model_path']!r}, "
                f"local_device={d['local_device']!r}, "
                f"prompts_dir={d['prompts_dir']!r})")

    __str__ = __repr__


__all__ = ['AIConfig', 'PROVIDERS', 'DEFAULT_AI_CONFIG_FILE']
