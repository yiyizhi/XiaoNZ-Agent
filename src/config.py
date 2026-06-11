"""Configuration loader for XiaoNZ Agent.

Loads from config.yaml next to the project root. Single config file,
no env var overrides (MVP keeps it simple).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class _StrictModel(BaseModel):
    # 拒绝未知字段：allowed_open_ids 之类拼错/缩进错层时要在启动时报错，
    # 而不是静默用默认值（空白名单 = 安全口子）。
    model_config = ConfigDict(extra="forbid")


class FeishuConfig(_StrictModel):
    app_id: str
    app_secret: str
    # 长连接模式用不到，但 config.yaml.example 模板里带着这两项，
    # extra="forbid" 下必须声明成可选字段，否则照模板抄会启动即崩。
    verification_token: str = ""
    encrypt_key: str = ""
    # Sender allowlist. When non-empty, only messages whose sender
    # open_id is in this list are processed (applies to BOTH p2p and
    # group chats). Empty = allow everyone — anyone who can DM the bot
    # or share a group with it can drive run_command on this machine,
    # so keep it set.
    allowed_open_ids: list[str] = Field(default_factory=list)


class ModelConfig(_StrictModel):
    base_url: str
    auth_token: str
    model_id: str = "claude-opus-4-7"
    max_tokens: int = 8192
    # OpenAI-compatible path on the same gateway, used for image
    # generation. Leave unset to disable the generate_image tool.
    openai_auth_token: Optional[str] = None
    image_model: str = "gpt-image-2"
    # httpx timeout for anthropic calls. Past incidents had the loop
    # silently freeze for 1.5h after two back-to-back 120s timeouts,
    # so keep these tight and retry few times.
    connect_timeout: float = 10.0
    read_timeout: float = 60.0
    max_retries: int = 1
    # 整轮 LLM 调用的墙钟硬上限（秒）。流式下 read_timeout 只是
    # chunk 间不活动阈值，上游持续滴流时单次调用可以永远不超时，
    # 该 turn 会占着 session 锁把后续消息全堵死，必须有整轮兜底。
    turn_timeout: float = 600.0


class AgentConfig(_StrictModel):
    max_iterations: int = 20


class MemoryConfig(_StrictModel):
    max_session_turns: int = 20


class SearchConfig(_StrictModel):
    # Web search backend cascade: Tavily → Brave → DuckDuckGo. Each
    # layer kicks in only when the previous one errors or returns no
    # hits. Keys are optional — leave empty to skip that layer.
    tavily_api_key: Optional[str] = None
    brave_api_key: Optional[str] = None


class EmbeddingConfig(_StrictModel):
    # OpenAI-compatible /v1/embeddings endpoint. Default points at a
    # local Ollama instance (`ollama pull bge-m3` + `ollama serve`).
    # Disabled by default — set enabled=true to turn on semantic recall.
    enabled: bool = False
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "bge-m3"
    dim: int = 1024
    # How many top-k semantic hits to inject into the system prompt.
    recall_top_k: int = 5
    # Minimum cosine score to consider a hit relevant.
    recall_min_score: float = 0.45


class Settings(_StrictModel):
    feishu: FeishuConfig
    model: ModelConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    # Derived paths (filled by load())
    project_root: Path = Field(default_factory=Path.cwd)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "hermes_state.db"

    @property
    def memory_dir(self) -> Path:
        return self.data_dir / "memory"

    @property
    def skills_dir(self) -> Path:
        return self.data_dir / "skills"

    @property
    def soul_path(self) -> Path:
        return self.memory_dir / "SOUL.md"

    @property
    def memory_path(self) -> Path:
        return self.memory_dir / "MEMORY.md"


def load(config_path: Path | str | None = None) -> Settings:
    """Load settings from config.yaml.

    Search order:
      1. explicit path argument
      2. $XIAONZ_CONFIG env var (not implemented in MVP)
      3. <project_root>/config.yaml
    """
    project_root = Path(__file__).resolve().parent.parent
    path = Path(config_path) if config_path else project_root / "config.yaml"

    if not path.is_file():
        raise FileNotFoundError(
            f"config.yaml not found at {path}. "
            f"Copy config.yaml.example to config.yaml and fill in the values."
        )

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    settings = Settings(**raw)
    settings.project_root = project_root

    if not settings.feishu.allowed_open_ids:
        # 入口闸门对空白名单是 deny-all（见 client.py），这里把原因喊出来，
        # 免得排查"为什么谁的消息都不回"时绕远路。
        logger.warning(
            "feishu.allowed_open_ids 为空：将拒绝所有发送者。"
            "请在 config.yaml 里配置白名单 open_id。"
        )

    # Ensure data dirs exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.memory_dir.mkdir(parents=True, exist_ok=True)
    settings.skills_dir.mkdir(parents=True, exist_ok=True)

    return settings
