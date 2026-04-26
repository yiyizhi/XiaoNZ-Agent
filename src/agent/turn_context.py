"""Per-turn context carried via a ContextVar.

Tools that need to know *which* Feishu conversation they're running
inside (e.g. `send_file_to_feishu`) read from `current_turn`. The
FeishuClient sets it at the start of every `_run_turn`; because each
turn is its own asyncio.Task, ContextVars are naturally task-local
and don't bleed across concurrent conversations.
"""
from __future__ import annotations

import threading
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class TurnContext:
    session_id: str
    source: str
    peer_id: str
    receive_id: str
    receive_id_type: str
    # Per-turn counter guarded by `lock`. Incremented by generate_image
    # so a single turn can't spam the user with arbitrarily many images.
    generate_image_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


current_turn: ContextVar[TurnContext | None] = ContextVar(
    "xiaonz_current_turn", default=None
)
