"""Smoke test: verify ModelClient can talk to the configured upstream end-to-end.

Run: venv/bin/python scripts/test_model.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Make `src` importable when run directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.agent.loop import AgentLoop  # noqa: E402
from src.agent.memory import MemoryStore  # noqa: E402
from src.agent.model_client import ModelClient  # noqa: E402
from src.agent.session import SessionStore  # noqa: E402
from src.agent.skills import SkillStore  # noqa: E402
from src.agent.tool_impls import default_tools  # noqa: E402


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    settings = config.load()
    store = SessionStore(settings.db_path)
    memory = MemoryStore(settings.memory_path)
    skills = SkillStore(settings.skills_dir)
    model = ModelClient(settings)
    tools = default_tools(skills=skills, memory=memory, db_path=settings.db_path)
    loop = AgentLoop(settings, store, model, skills=skills, tools=tools)

    session_id = "test:selfcheck"
    # Reset so repeated runs don't accumulate history
    store.reset_session(session_id)

    reply = await loop.handle_user_message(
        session_id=session_id,
        source="feishu_p2p",
        peer_id="local_test",
        user_text="你好，请用一句话介绍一下你自己。",
    )
    print("----- reply -----")
    print(reply)
    print("-----------------")
    return 0 if reply and reply != "(no response)" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
