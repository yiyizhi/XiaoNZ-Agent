"""In-chat slash commands.

Anything the user sends that starts with '/' is handled here and
returned as a plain text reply — it never reaches the LLM.

Supported:
  /help            — list commands
  /new  | /reset   — clear current session history
  /mem             — show MEMORY.md
  /skills          — list available skills

Unknown commands fall through to a friendly error string so the user
doesn't silently get no response.
"""
from __future__ import annotations

import logging
from typing import Callable

from .memory import MemoryStore
from .session import SessionStore
from .skills import SkillStore

logger = logging.getLogger(__name__)


HELP_TEXT = (
    "可用命令：\n"
    "/help        显示本帮助\n"
    "/new         开始新对话（清空当前会话历史）\n"
    "/mem         查看长期记忆（MEMORY.md）\n"
    "/skills      列出可用技能\n"
    "/cancel      中断当前正在处理的消息（也可直接撤回那条消息）"
)


# `session_id → number of tasks cancelled`. Injected by main.py after
# the FeishuClient is built, to avoid a circular import.
CancelCallback = Callable[[str], int]


class CommandHandler:
    def __init__(
        self,
        store: SessionStore,
        memory: MemoryStore,
        skills: SkillStore,
    ):
        self.store = store
        self.memory = memory
        self.skills = skills
        self._cancel_cb: CancelCallback | None = None

    def set_cancel_callback(self, cb: CancelCallback) -> None:
        self._cancel_cb = cb

    def is_command(self, text: str) -> bool:
        return text.startswith("/")

    def handle(self, session_id: str, text: str) -> str:
        # Split into command + rest (rest currently unused but kept
        # for future /skill <name> style commands)
        parts = text.strip().split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == "/help":
            return HELP_TEXT

        if cmd in ("/new", "/reset"):
            n = self.store.reset_session(session_id)
            return f"已开始新对话（清空了 {n} 条历史消息）。"

        if cmd == "/mem":
            content = self.memory.read().strip()
            if not content:
                return "长期记忆（MEMORY.md）为空。"
            # Truncate very long memory to keep the reply reasonable
            if len(content) > 3500:
                content = content[:3500] + "\n\n…(已截断)"
            return f"# 长期记忆\n\n{content}"

        if cmd == "/cancel":
            if self._cancel_cb is None:
                return "取消功能未启用。"
            n = self._cancel_cb(session_id)
            if n > 0:
                return f"已中断 {n} 个正在处理的请求。"
            return "当前没有正在处理的请求。"

        if cmd == "/skills":
            skills = self.skills.list_all()
            if not skills:
                return "目前还没有安装任何技能。"
            lines = ["# 可用技能"]
            for s in skills:
                desc = s.description or "(无描述)"
                lines.append(f"- **{s.name}** — {desc}")
            return "\n".join(lines)

        return f"未知命令：{cmd}。输入 /help 查看可用命令。"
