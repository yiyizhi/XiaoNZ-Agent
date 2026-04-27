"""Compression smoke test.

Injects enough fake history into a throwaway session to cross the
compression threshold, then runs one real turn and checks that:
  - message count dropped to ~keep
  - session_summaries has a non-empty summary
  - the final reply still comes back
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


FILLER_USER = [
    "我叫 Alex，住在北京。",
    "我偏好简短的回复。",
    "我喜欢喝冰美式，加一份浓缩。",
    "我的生日是 3 月 14 号。",
    "我有一只橘猫，叫团子。",
    "我平时用 macOS 和 zsh。",
    "我在做一个叫 XiaoNZ 的 Agent 项目。",
    "我周末常去昌平徒步。",
    "我不吃香菜。",
    "我下周要去一趟上海。",
    "帮我记一下这些信息。",
    "你先不用回复每条，攒到一起就行。",
    "下面还有几条零散的。",
    "我习惯夜里工作，白天补觉。",
    "我不喝酒但喝咖啡。",
    "我最喜欢的电影是《花样年华》。",
    "我对樱花过敏。",
    "我妈叫我 Alexa。",
    "我常去的咖啡店叫「山丘」。",
    "我有个妹妹会用这个 bot。",
    "她的偏好和我一样：简短、直接。",
    "OK 信息差不多了。",
]


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
    tools = default_tools(skills=skills, memory=memory, db_path=settings.db_path)
    loop = AgentLoop(settings, store, model, skills=skills, tools=tools)

    session_id = "test:compress"
    store.reset_session(session_id)
    store.ensure_session(session_id, "feishu_p2p", "local_test")

    # Pump fake history directly into the DB. Each filler is paired
    # with a trivial assistant ack. With 22 pairs = 44 messages,
    # comfortably past the default threshold (max_session_turns*2*2 = 80)
    # — actually not. Let me crank it.
    for i, msg in enumerate(FILLER_USER):
        store.append_message(session_id, "user", msg)
        store.append_message(session_id, "assistant", f"收到第{i + 1}条。")
    # Add more filler to exceed threshold (threshold = keep*2 where
    # keep = max_session_turns*2. Default max_session_turns=20 → keep=40
    # → threshold=80). We already have 44. Add another 40 short pairs.
    for i in range(40):
        store.append_message(session_id, "user", f"继续闲聊 {i}")
        store.append_message(session_id, "assistant", f"嗯嗯 {i}")

    before = store.count_messages(session_id)
    print(f"--- before: {before} messages ---")

    reply = await loop.handle_user_message(
        session_id=session_id,
        source="feishu_p2p",
        peer_id="local_test",
        user_text="根据前面我告诉你的信息，总结一下我不吃什么？",
    )
    print("\n--- ASSISTANT ---")
    print(reply)

    after = store.count_messages(session_id)
    summary = store.get_summary(session_id)
    print(f"\n--- after: {after} messages ---")
    print(f"--- summary ({len(summary)} chars) ---")
    print(summary)

    ok = (
        after < before
        and bool(summary.strip())
        and reply
        and reply != "(no response)"
    )
    print(f"\ncompression_ok={ok}")

    # Clean up so we don't leave junk in the DB
    store.reset_session(session_id)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
