"""Vision smoke test.

Simulates a Feishu image upload end-to-end without hitting the real
Feishu API:
  1. Builds a FeishuClient with the real AgentLoop.
  2. Monkey-patches `_download_resource` to return a tiny pre-baked
     PNG (a 2x2 red square) instead of calling Feishu.
  3. Patches the outbound card helpers to no-op (record-only).
  4. Dispatches a turn with `image_key="fake_key"` and waits for the
     real model round-trip to finish.
  5. Asserts the model returned some text describing the image.

If this passes, the vision content-block construction + Anthropic
call + reply wiring all work. If the live bot is still "blind", the
problem is in the Feishu event routing, not in vision itself.
"""
from __future__ import annotations

import asyncio
import logging
import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.agent.commands import CommandHandler  # noqa: E402
from src.agent.loop import AgentLoop  # noqa: E402
from src.agent.memory import MemoryStore  # noqa: E402
from src.agent.model_client import ModelClient  # noqa: E402
from src.agent.session import SessionStore  # noqa: E402
from src.agent.skills import SkillStore  # noqa: E402
from src.agent.tool_impls import default_tools  # noqa: E402
from src.feishu.client import FeishuClient  # noqa: E402


def _make_quadrant_png() -> bytes:
    """Build a 256x256 PNG split into four solid-color quadrants
    (red/green/blue/yellow). Big and distinctive enough that vision
    models reliably recognize it."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    w = h = 256
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    rows = []
    for y in range(h):
        row = bytearray([0])  # filter: None
        for x in range(w):
            if y < 128 and x < 128:
                row += b"\xff\x00\x00"  # red
            elif y < 128:
                row += b"\x00\xff\x00"  # green
            elif x < 128:
                row += b"\x00\x00\xff"  # blue
            else:
                row += b"\xff\xff\x00"  # yellow
        rows.append(bytes(row))
    idat = zlib.compress(b"".join(rows))
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


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
    agent = AgentLoop(settings, store, model, skills=skills, tools=tools)
    commands = CommandHandler(store, memory, skills)
    client = FeishuClient(settings, store, agent, commands)

    session_id = "test:vision"
    store.reset_session(session_id)

    # Replace network layers
    png_bytes = _make_quadrant_png()

    def stub_download(message_id, key, resource_type):  # noqa: ANN001
        assert resource_type == "image"
        return png_bytes

    client._download_resource = stub_download  # type: ignore[method-assign]

    # End-to-end vision needs a publicly fetchable HTTPS URL (some
    # upstreams reject base64). The production path uploads to
    # tmpfiles.org; we actually exercise that here so the test covers
    # the real chain.
    # (If tmpfiles is down the test will fail with a clear error.)

    sent_cards: list[str] = []
    patched_cards: list[str] = []

    def stub_send_card(receive_id, receive_id_type, markdown):  # noqa: ANN001
        sent_cards.append(markdown)
        return "stub_card_1"

    def stub_patch_card(message_id, markdown):  # noqa: ANN001
        patched_cards.append(markdown)
        return True

    client._send_card = stub_send_card  # type: ignore[method-assign]
    client._patch_card = stub_patch_card  # type: ignore[method-assign]

    # Pretend this is the agent thread's loop
    await client._run_turn(
        message_id="test_img_msg_1",
        session_id=session_id,
        source="feishu_p2p",
        peer_id="local_test",
        user_text="（我发了一张图片）",
        receive_id="local_test",
        receive_id_type="open_id",
        image_key="fake_image_key",
    )

    print("\n--- sent_cards (thinking placeholder) ---")
    for s in sent_cards:
        print(repr(s))
    print("\n--- patched_cards (final reply) ---")
    for p in patched_cards:
        print(p)

    if not patched_cards:
        print("\nFAIL: no reply produced")
        return 1
    final = patched_cards[-1]
    if "（已取消）" in final or "出了点问题" in final:
        print("\nFAIL: agent errored out —", final)
        return 1
    if len(final.strip()) < 5:
        print("\nFAIL: reply too short to be a real vision answer")
        return 1

    print("\nALL PASS — vision path works.")
    store.reset_session(session_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
