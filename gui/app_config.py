#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 应用配置持久化与自动保存

问题七：本模块原为 ``gui/persistence.py``，因与 ``persistence/mixins.py``（数据
持久化）命名冲突而更名为 ``gui/app_config.py``，以消除命名歧义。

职责：
  - AppConfig：基于 dataclass 的应用配置项管理，JSON 读写 ~/.tb_risk/settings.json
  - AutoSaveManager：脏标记 + 定时器自动保存会话，原子写入防止崩溃损坏

使用方式：
    config = AppConfig.load()
    config.last_directory = "/path/to/file"
    config.save()

    auto_save = AutoSaveManager(app, interval=60)
    auto_save.mark_dirty()
    auto_save.start()  # 启动定时保存
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Optional

LOGGER = logging.getLogger(__name__)


# 配置目录（跨平台）
CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.tb_risk')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'settings.json')
AUTOSAVE_FILE = os.path.join(CONFIG_DIR, 'autosave.json')


@dataclass
class AppConfig:
    """应用配置（集中定义为数据类，新增配置项只需在此添加字段）"""
    # 窗口几何信息
    window_geometry: str = "1200x800"
    # 上次使用的目录
    last_directory: str = ""
    # 上次使用的场景
    last_scenario: str = "custom"
    # UI 模式偏好（classic / wizard）
    ui_mode_preference: str = "classic"
    # 自动保存间隔（秒）
    autosave_interval: int = 60
    # 最近打开的文件列表
    recent_files: list = field(default_factory=list)
    # AI 助手开关（用户须显式开启；关闭时所有 LLM 调用被旁路）
    ai_enabled: bool = False
    # AI 服务商（deepseek/openai/dashscope/zhipu/moonshot；仅作 UI 默认值，
    # 实际配置走 ~/.tb_risk/ai_config.json 或环境变量）
    ai_provider: str = "deepseek"
    # Base URL（留空使用服务商预设端点；私有部署或兼容协议时填写）
    ai_base_url: str = ""
    # 模型名称（留空使用服务商预设模型；如 deepseek-chat、gpt-4o-mini）
    ai_model: str = ""
    # 本地模型路径（敏感场景：数据不出本机；非空时优先本地模式）
    ai_local_model_path: str = ""
    # 请求超时秒数
    ai_timeout: int = 60
    # 最大重试次数
    ai_max_retries: int = 3
    # 采样温度（0.0 确定性 ~ 2.0 多样性）
    ai_temperature: float = 0.2
    # 外置提示词目录（空字符串表示不使用外置提示词，使用 prompts.py 默认常量）
    ai_prompts_dir: str = ""
    # 注：ai_api_key 不在 AppConfig 中（方案 A：仅存 ~/.tb_risk/ai_config.json，
    # 避免随项目配置分享泄露）

    # --- HL7 MLLP 接口服务配置 ---
    mllp_enabled: bool = False
    mllp_host: str = "0.0.0.0"
    mllp_port: int = 2575
    mllp_data_dir: str = "hl7_messages"
    mllp_auto_assess: bool = True  # 接收消息后自动风险评估

    # --- CDC Webhook 配置 ---
    webhook_enabled: bool = False
    webhook_host: str = "0.0.0.0"
    webhook_port: int = 8080
    webhook_secret_key: str = ""
    webhook_data_dir: str = "cdc_webhook"
    webhook_update_gnn: bool = True  # 回传数据自动更新GNN网络

    @property
    def ai_is_local(self) -> bool:
        """是否启用本地模型模式（只读派生属性）

        与 AIConfig.is_local 语义一致：local_model_path 非空时为 True。
        敏感场景下数据不出本机，使用本地 transformers 模型推理。
        """
        return bool(self.ai_local_model_path)

    @classmethod
    def load(cls) -> 'AppConfig':
        """从 JSON 文件加载配置，损坏时回退默认值"""
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 仅取 dataclass 中定义的字段，忽略多余键
            valid_fields = {k: v for k, v in data.items()
                           if k in cls.__dataclass_fields__}
            return cls(**valid_fields)
        except (FileNotFoundError, json.JSONDecodeError, TypeError, KeyError):
            return cls()

    def save(self):
        """保存配置到 JSON 文件（原子写入）"""
        os.makedirs(CONFIG_DIR, exist_ok=True)
        data = asdict(self)
        # 原子写入：先写临时文件再重命名，防止写入中途崩溃
        fd, tmp_path = tempfile.mkstemp(dir=CONFIG_DIR, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # Windows 上 os.replace 可以原子性地覆盖已存在文件
            os.replace(tmp_path, CONFIG_FILE)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError as e:
                LOGGER.debug("清理临时文件 %s 失败: %s", tmp_path, e)
            raise


class AutoSaveManager:
    """自动保存管理器（脏标记 + 定时器）

    任何数据变更调用 mark_dirty()，定时器每 interval 秒检查：
    若 dirty 则自动保存到 AUTOSAVE_FILE。程序启动时检查该文件是否存在，
    提示用户恢复会话。
    """

    def __init__(self, app, interval=60):
        """
        Args:
            app: 具有 family_entries, social_entries, basic_info_vars 等属性的主对象
            interval: 自动保存间隔（秒）
        """
        self.app = app
        self.interval = interval
        self._dirty = False
        self._timer_id = None
        self._after_callback = None  # root.after 回调

    def mark_dirty(self):
        """标记数据已修改，需要自动保存"""
        self._dirty = True

    def start(self):
        """启动定时自动保存"""
        self._schedule_next()

    def stop(self):
        """停止定时自动保存"""
        if self._timer_id is not None:
            try:
                self.app.root.after_cancel(self._timer_id)
            except Exception as e:
                LOGGER.debug("取消自动保存定时器失败: %s", e)
            self._timer_id = None

    def _schedule_next(self):
        """调度下一次自动保存检查"""
        if hasattr(self.app, 'root') and self.app.root:
            if self._timer_id is not None:
                try:
                    self.app.root.after_cancel(self._timer_id)
                except Exception as e:
                    LOGGER.debug("取消旧自动保存定时器失败: %s", e)
            try:
                self._timer_id = self.app.root.after(
                    self.interval * 1000, self._check_and_save
                )
            except Exception as e:
                LOGGER.warning("调度自动保存定时器失败: %s", e)

    def _check_and_save(self):
        """定时回调：若 dirty 则保存，然后调度下一次"""
        if self._dirty:
            self._save_now()
            self._dirty = False
        self._schedule_next()

    def _save_now(self):
        """执行实际保存（原子写入）"""
        try:
            data = self._collect_state()
            os.makedirs(CONFIG_DIR, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=CONFIG_DIR, suffix='.tmp')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2,
                             default=str)
                os.replace(tmp_path, AUTOSAVE_FILE)
            except Exception as e:
                try:
                    os.unlink(tmp_path)
                except OSError as cleanup_err:
                    LOGGER.debug("自动保存清理临时文件 %s 失败: %s",
                                 tmp_path, cleanup_err)
                LOGGER.warning("自动写入 autosave.json 失败: %s", e)
        except Exception as e:
            LOGGER.warning("自动保存失败（已忽略，不影响用户操作）: %s", e)

    def _collect_state(self):
        """收集当前应用状态（所有 tkinter 变量 + 接触者列表）"""
        data = {
            'family_entries': getattr(self.app, 'family_entries', []),
            'social_entries': getattr(self.app, 'social_entries', []),
        }
        basic_vars = getattr(self.app, 'basic_info_vars', {})
        if basic_vars:
            var_data = {}
            for key, var in basic_vars.items():
                try:
                    var_data[key] = var.get()
                except Exception as e:
                    LOGGER.debug("收集 basic_info_vars[%s] 失败: %s", key, e)
                    var_data[key] = None
            data['basic_info_vars'] = var_data
        if hasattr(self.app, 'contact_labels'):
            data['contact_labels'] = dict(self.app.contact_labels)
        return data

    @staticmethod
    def has_autosave() -> bool:
        """检查是否存在自动保存文件（用于启动时恢复提示）"""
        return os.path.exists(AUTOSAVE_FILE)

    @staticmethod
    def load_autosave() -> Optional[dict]:
        """加载自动保存的数据，失败返回 None"""
        try:
            with open(AUTOSAVE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as e:
            LOGGER.warning("自动保存文件损坏，无法恢复: %s", e)
            return None

    @staticmethod
    def clear_autosave():
        """清除自动保存文件（正常退出时调用）"""
        try:
            os.unlink(AUTOSAVE_FILE)
        except FileNotFoundError:
            LOGGER.debug("自动保存文件不存在，无需清除: %s", AUTOSAVE_FILE)
