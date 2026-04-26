"""XiaoNZ Agent entry point.

Usage:
    venv/bin/python -m src.main

Wiring:
    config.load → SessionStore → ModelClient → AgentLoop → FeishuClient.start()

`FeishuClient.start()` is blocking: it owns the main thread to run
`lark.ws.Client.start()` (which expects to be on the thread whose
asyncio loop was bound at import time), and spawns a background
thread for the Agent's own asyncio loop.
"""
from __future__ import annotations

import logging
import signal
import sys

from . import config
from .agent.commands import CommandHandler
from .agent.embedder import Embedder
from .agent.loop import AgentLoop
from .agent.memory import MemoryStore
from .agent.model_client import ModelClient
from .agent.session import SessionStore
from .agent.skills import SkillStore
from .agent.tool_impls import (
    default_tools,
    make_browser_capture_tool,
    make_generate_image_tool,
    make_search_memory_semantic_tool,
    make_send_to_feishu_tool,
)
from .agent.vector_memory import VectorMemory
from .feishu.client import FeishuClient


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Quiet down lark's extremely chatty debug logs if any sneak in
    logging.getLogger("lark_oapi").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


def main() -> int:
    _setup_logging()
    log = logging.getLogger("xiaonz.main")

    settings = config.load()
    log.info(
        "xiaonz.start project_root=%s db=%s model=%s",
        settings.project_root,
        settings.db_path,
        settings.model.model_id,
    )

    store = SessionStore(settings.db_path)
    memory = MemoryStore(settings.memory_path)
    skills = SkillStore(settings.skills_dir)
    model = ModelClient(settings)

    vector_memory = None
    if settings.embedding.enabled:
        embedder = Embedder(
            base_url=settings.embedding.base_url,
            model=settings.embedding.model,
            dim=settings.embedding.dim,
        )
        vector_memory = VectorMemory(settings.db_path, embedder)
        log.info(
            "xiaonz.vector_memory enabled model=%s dim=%d top_k=%d min_score=%.2f",
            settings.embedding.model,
            settings.embedding.dim,
            settings.embedding.recall_top_k,
            settings.embedding.recall_min_score,
        )

    tools = default_tools(skills=skills, memory=memory)
    agent = AgentLoop(
        settings, store, model, skills=skills, tools=tools,
        vector_memory=vector_memory,
    )
    commands = CommandHandler(store, memory, skills)
    feishu = FeishuClient(settings, store, agent, commands)
    # Late wiring: CommandHandler doesn't import FeishuClient, so bind
    # the /cancel hook here after both are constructed. Same for the
    # send_to_feishu tool, which needs a live lark client reference.
    commands.set_cancel_callback(feishu.cancel_session)
    agent.register_tool(make_send_to_feishu_tool(feishu))
    agent.register_tool(make_browser_capture_tool(settings))
    log.info("xiaonz.tool.browser_capture registered")
    if settings.model.openai_auth_token:
        agent.register_tool(make_generate_image_tool(settings, feishu))
        log.info("xiaonz.tool.generate_image registered model=%s",
                 settings.model.image_model)
    if vector_memory is not None:
        agent.register_tool(make_search_memory_semantic_tool(vector_memory))
        log.info("xiaonz.tool.search_memory_semantic registered")

    # Graceful shutdown on SIGINT/SIGTERM — lark.ws.Client.start()
    # handles KeyboardInterrupt internally; we just want a clean log.
    def _handle_signal(signum, _frame):  # noqa: ANN001
        log.info("xiaonz.shutdown signal=%s", signum)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    feishu.start()  # blocks forever
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
