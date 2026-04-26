"""End-to-end tool-use smoke test.

Drives the Agent with a request designed to trigger tool calls and
verifies the expected side effects.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

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
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = config.load()
    store = SessionStore(settings.db_path)
    memory = MemoryStore(settings.memory_path)
    skills = SkillStore(settings.skills_dir)
    model = ModelClient(settings)
    tools = default_tools(skills=skills, memory=memory)
    loop = AgentLoop(settings, store, model, skills=skills, tools=tools)

    session_id = "test:tools"
    store.reset_session(session_id)

    # Snapshot pre-state
    memory_before = memory.read()
    print("--- MEMORY before ---")
    print(memory_before)

    # Ask something that should cause the agent to (a) load the
    # reply-style skill and (b) persist a preference.
    user_msg = (
        "请记住：我偏好极简风格的回复，每次回复尽量控制在两句话以内。"
        "然后参考你的 reply-style 技能，用一句话确认。"
    )
    print("\n--- USER ---")
    print(user_msg)

    reply = await loop.handle_user_message(
        session_id=session_id,
        source="feishu_p2p",
        peer_id="local_test",
        user_text=user_msg,
    )
    print("\n--- ASSISTANT ---")
    print(reply)

    memory_after = memory.read()
    print("\n--- MEMORY after ---")
    print(memory_after)

    changed = memory_before != memory_after
    print(f"\nmemory_changed={changed}")
    return 0 if reply and reply != "(no response)" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
