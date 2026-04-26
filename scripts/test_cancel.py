"""Cancellation smoke test.

Verifies the inflight-tracking + cancel pipeline end-to-end without
touching the real Feishu network. We:

  1. Build a FeishuClient with a stub AgentLoop whose handle_user_message
     sleeps for a long time.
  2. Start the agent thread.
  3. Dispatch a "turn" onto that loop.
  4. Assert the task shows up in the inflight dict.
  5. Call cancel_message / cancel_session from the main thread.
  6. Assert the task ends and the dict is cleaned up.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.agent.commands import CommandHandler  # noqa: E402
from src.agent.memory import MemoryStore  # noqa: E402
from src.agent.session import SessionStore  # noqa: E402
from src.agent.skills import SkillStore  # noqa: E402
from src.feishu.client import FeishuClient  # noqa: E402


class StubAgent:
    """Pretends to be AgentLoop but just sleeps — long enough for the
    test to observe inflight state and fire a cancel."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancelled = False

    async def handle_user_message(
        self,
        session_id: str,
        source: str,
        peer_id: str,
        user_text: str,
        attachments: list[dict] | None = None,
    ) -> str:
        self.started.set()
        try:
            await asyncio.sleep(30.0)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return "(shouldn't get here)"


def _patch_send_text(client: FeishuClient) -> list[str]:
    """Replace FeishuClient's outbound helpers with no-ops that record
    what would have been sent, so the test never talks to real Feishu.

    The current code path is: `_send_card` (placeholder) → agent →
    `_patch_card`. Both need stubbing. `_send_text` is kept as a
    last-resort fallback and also stubbed for safety.
    """
    sent: list[str] = []
    counter = {"n": 0}

    def stub_card(receive_id, receive_id_type, markdown):  # noqa: ANN001
        counter["n"] += 1
        mid = f"stub_card_{counter['n']}"
        sent.append(f"CARD[{mid}]:{markdown}")
        return mid

    def stub_patch(message_id, markdown):  # noqa: ANN001
        sent.append(f"PATCH[{message_id}]:{markdown}")
        return True

    def stub_text(receive_id, receive_id_type, text):  # noqa: ANN001
        sent.append(f"TEXT:{text}")

    client._send_card = stub_card  # type: ignore[method-assign]
    client._patch_card = stub_patch  # type: ignore[method-assign]
    client._send_text = stub_text  # type: ignore[method-assign]
    return sent


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    settings = config.load()
    store = SessionStore(settings.db_path)
    memory = MemoryStore(settings.memory_path)
    skills = SkillStore(settings.skills_dir)

    stub_agent = StubAgent()
    commands = CommandHandler(store, memory, skills)
    client = FeishuClient(settings, store, stub_agent, commands)  # type: ignore[arg-type]
    commands.set_cancel_callback(client.cancel_session)
    sent = _patch_send_text(client)

    # Spin up the agent thread without starting the ws loop
    client._start_agent_thread()
    assert client._agent_loop is not None
    loop = client._agent_loop

    session_id = "test:cancel"
    store.reset_session(session_id)
    store.ensure_session(session_id, "feishu_p2p", "local_test")

    # --- case 1: cancel_message by id ---------------------------------
    msg_id_a = "test_msg_A"
    fut_a = asyncio.run_coroutine_threadsafe(
        client._run_turn(
            message_id=msg_id_a,
            session_id=session_id,
            source="feishu_p2p",
            peer_id="local_test",
            user_text="hi",
            receive_id="local_test",
            receive_id_type="open_id",
        ),
        loop,
    )

    # Wait for the stub to actually start the sleep
    if not stub_agent.started.wait(timeout=5.0):
        print("FAIL: stub agent never started")
        return 1

    # Give the event dispatch + registration a tick to settle
    time.sleep(0.05)

    with client._inflight_lock:
        present = msg_id_a in client._inflight
    if not present:
        print("FAIL: task not registered in inflight dict")
        return 1
    print(f"OK: inflight registered for {msg_id_a}")

    # Fire cancel
    n = client.cancel_message(msg_id_a)
    print(f"cancel_message returned {n}")
    if n != 1:
        print("FAIL: expected cancel_message → 1")
        return 1

    # Wait for the task to actually end
    try:
        fut_a.result(timeout=5.0)
    except Exception as e:
        print(f"FAIL: task raised {type(e).__name__}: {e}")
        return 1

    if not stub_agent.cancelled:
        print("FAIL: CancelledError never reached stub agent")
        return 1

    with client._inflight_lock:
        still_there = msg_id_a in client._inflight
    if still_there:
        print("FAIL: inflight dict not cleaned up")
        return 1
    print("OK: cancel_message path — task cancelled and cleaned up")

    # --- case 2: cancel_session via /cancel command ------------------
    stub_agent.started.clear()
    stub_agent.cancelled = False

    msg_id_b = "test_msg_B"
    fut_b = asyncio.run_coroutine_threadsafe(
        client._run_turn(
            message_id=msg_id_b,
            session_id=session_id,
            source="feishu_p2p",
            peer_id="local_test",
            user_text="hello again",
            receive_id="local_test",
            receive_id_type="open_id",
        ),
        loop,
    )
    if not stub_agent.started.wait(timeout=5.0):
        print("FAIL: case 2 stub never started")
        return 1
    time.sleep(0.05)

    reply = commands.handle(session_id, "/cancel")
    print(f"/cancel → {reply!r}")
    if "已中断 1" not in reply:
        print("FAIL: /cancel reply didn't acknowledge 1 task")
        return 1

    try:
        fut_b.result(timeout=5.0)
    except Exception as e:
        print(f"FAIL: task B raised {type(e).__name__}: {e}")
        return 1

    if not stub_agent.cancelled:
        print("FAIL: case 2 — CancelledError never reached stub")
        return 1

    with client._inflight_lock:
        remaining = list(client._inflight_by_session.get(session_id, []))
    if remaining:
        print(f"FAIL: inflight_by_session still has {remaining}")
        return 1
    print("OK: /cancel path — task cancelled and cleaned up")

    # --- case 3: /cancel when nothing is running -------------------
    reply = commands.handle(session_id, "/cancel")
    print(f"/cancel (idle) → {reply!r}")
    if "当前没有正在处理" not in reply:
        print("FAIL: /cancel idle reply wrong")
        return 1
    print("OK: /cancel idle path")

    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
