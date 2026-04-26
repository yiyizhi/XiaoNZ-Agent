"""Configuration loader for XiaoNZ Agent.

Loads from config.yaml next to the project root. Single config file,
no env var overrides (MVP keeps it simple).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class FeishuConfig(BaseModel):
    app_id: str
    app_secret: str


class ModelConfig(BaseModel):
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


class AgentConfig(BaseModel):
    max_iterations: int = 20


class MemoryConfig(BaseModel):
    max_session_turns: int = 20


class EmbeddingConfig(BaseModel):
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


class Settings(BaseModel):
    feishu: FeishuConfig
    model: ModelConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
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

    # Ensure data dirs exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.memory_dir.mkdir(parents=True, exist_ok=True)
    settings.skills_dir.mkdir(parents=True, exist_ok=True)

    return settings
