"""Agent Loop with tool-use support.

Flow per user turn:
    1. ensure_session + append user message (persisted)
    2. build system prompt from SOUL.md + MEMORY.md
    3. load normalized history from SQLite
    4. tool-use loop (bounded by max_iterations):
        - call model with current `messages` + tool specs
        - if stop_reason == "tool_use":
            * append assistant message (with tool_use blocks) to local `messages`
            * run each tool handler, collect tool_result blocks
            * append user message (with tool_result blocks) to local `messages`
            * continue
        - else:
            * extract final text, persist assistant message, return
    5. if loop exhausts, persist a fallback message and return

Persistence model: only the user message and the final assistant text
are written to SQLite. Intermediate tool_use / tool_result rounds live
purely in-memory for the duration of the turn — keeps the SessionStore
schema simple (role + string content) and keeps past turns free of
stale tool plumbing.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, timedelta
from typing import Any, Optional

from ..config import Settings
from .model_client import ModelClient
from .session import SessionStore
from .skills import SkillStore
from .tools import Tool
from .vector_memory import VectorMemory

logger = logging.getLogger(__name__)

# Track whether today's daily digest check has been done (avoid
# repeating the check + LLM call on every single turn).
_last_digest_check: str | None = None

# Strong refs for fire-and-forget tasks so the GC doesn't drop them
# mid-flight (asyncio only weakly refs scheduled tasks).
_bg_tasks: set[asyncio.Task] = set()


def _spawn_bg(coro, label: str) -> None:
    """Schedule a fire-and-forget coroutine. Logs exceptions instead of
    letting them get swallowed when the task is GC'd."""
    task = asyncio.create_task(coro, name=label)
    _bg_tasks.add(task)

    def _done(t: asyncio.Task) -> None:
        _bg_tasks.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.warning("bg_task_failed label=%s err=%r", label, exc)

    task.add_done_callback(_done)


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Convert an Anthropic SDK content block to a plain dict suitable
    for echoing back as `messages` input. We only need the block types
    we actually produce in this loop."""
    btype = getattr(block, "type", None)
    if btype == "text":
        return {"type": "text", "text": block.text}
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    # Unknown block — serialize best-effort via model_dump if available
    if hasattr(block, "model_dump"):
        return block.model_dump()
    return {"type": btype or "unknown"}


class AgentLoop:
    def __init__(
        self,
        settings: Settings,
        store: SessionStore,
        model: ModelClient,
        skills: SkillStore,
        tools: list[Tool] | None = None,
        vector_memory: Optional[VectorMemory] = None,
    ):
        self.settings = settings
        self.store = store
        self.model = model
        self.skills = skills
        self.tools: list[Tool] = list(tools or [])
        self._by_name: dict[str, Tool] = {t.name: t for t in self.tools}
        self.vector_memory = vector_memory

    def register_tool(self, tool: Tool) -> None:
        self.tools.append(tool)
        self._by_name[tool.name] = tool

    # ── prompt assembly ────────────────────────────────────────────

    def _build_system(
        self,
        session_id: str,
        recalled: Optional[list[dict]] = None,
    ) -> str:
        parts: list[str] = []
        soul_path = self.settings.soul_path
        if soul_path.is_file():
            parts.append(soul_path.read_text(encoding="utf-8").strip())
        mem_path = self.settings.memory_path
        if mem_path.is_file():
            mem = mem_path.read_text(encoding="utf-8").strip()
            if mem:
                parts.append("# 长期记忆\n\n" + mem)
        if recalled:
            lines = [
                "# 相关记忆片段（语义召回，按相关度排序）",
                "（这些是根据当前消息从历史对话/摘要/长期记忆里找到的最相关片段，",
                "可能跨越很久以前的对话，用于补充上下文。如果和当前话题无关可以忽略。）",
                "",
            ]
            for hit in recalled:
                kind = hit.get("kind", "?")
                score = hit.get("score", 0.0)
                ref = hit.get("source_ref") or ""
                text = (hit.get("text") or "").strip()
                if len(text) > 600:
                    text = text[:600] + "…"
                header = f"## [{kind} score={score:.2f}]"
                if ref:
                    header += f" {ref}"
                lines.append(header)
                lines.append(text)
                lines.append("")
            parts.append("\n".join(lines).rstrip())
        # Rolling summary of older turns (if compression has run)
        summary = self.store.get_summary(session_id)
        if summary.strip():
            parts.append("# 之前对话摘要\n\n" + summary.strip())
        # Inject recent daily digests so the model has cross-session
        # memory of past days (up to last 7 days).
        digests = self.store.list_daily_digests(limit=7)
        if digests:
            digest_lines = ["# 近期每日摘要"]
            for d in digests:
                text = self.store.get_daily_digest(d["date"])
                if text:
                    digest_lines.append(
                        f"## {d['date']}（{d['msg_count']} 条消息）\n{text}"
                    )
            if len(digest_lines) > 1:
                parts.append("\n\n".join(digest_lines))

        # Inject a lightweight skill index so the model knows what's
        # available without having to speculatively call list_skills.
        items = self.skills.list_all()
        if items:
            lines = ["# 可用技能（需要时调用 load_skill 读取全文）"]
            for s in items:
                desc = s.description or "(no description)"
                lines.append(f"- {s.name}: {desc}")
            parts.append("\n".join(lines))
        return "\n\n---\n\n".join(parts) if parts else ""

    # ── daily digest ────────────────────────────────────────────────

    async def _maybe_generate_daily_digest(self) -> None:
        """At the start of each day, compress ALL of yesterday's
        conversations into a single daily digest row. Skipped if the
        digest already exists or if today's check has already been done.
        """
        global _last_digest_check
        today = date.today().isoformat()
        if _last_digest_check == today:
            return
        _last_digest_check = today

        yesterday = (date.today() - timedelta(days=1)).isoformat()
        if self.store.has_daily_digest(yesterday):
            return

        messages = self.store.get_messages_for_date(yesterday)
        if not messages:
            logger.info("digest.skip date=%s no_messages", yesterday)
            return

        # Build a transcript grouped by session
        from collections import defaultdict
        by_session: dict[str, list[dict]] = defaultdict(list)
        for m in messages:
            by_session[m["session_id"]].append(m)

        transcript_parts: list[str] = []
        for sid, msgs in by_session.items():
            lines = []
            for m in msgs:
                role = "用户" if m["role"] == "user" else "助手"
                lines.append(f"{role}：{m['content']}")
            transcript_parts.append(
                f"--- 会话 {sid[:8]} ---\n" + "\n".join(lines)
            )
        transcript = "\n\n".join(transcript_parts)

        # Truncate very long transcripts to avoid blowing up the summarizer
        if len(transcript) > 30_000:
            transcript = transcript[:30_000] + "\n\n... (已截断)"

        summarizer_system = (
            "你是一个对话摘要助手。把下面一天内的所有对话压缩成简洁的中文摘要，"
            "方便日后回顾。\n\n"
            "要点：\n"
            "- 聚焦：用户请求、做出的决定、完成的任务、重要事实\n"
            "- 省略：寒暄、重复内容\n"
            "- 长度：不超过 600 字\n"
            "- 格式：纯文本段落或短列表"
        )
        user_prompt = f"日期：{yesterday}\n\n{transcript}"

        try:
            summary = await self.model.chat(
                messages=[{"role": "user", "content": user_prompt}],
                system=summarizer_system,
            )
        except Exception:
            logger.exception("digest.summarize_failed date=%s", yesterday)
            return

        if not summary or summary == "(no response)":
            logger.warning("digest.empty_summary date=%s", yesterday)
            return

        self.store.save_daily_digest(yesterday, summary, len(messages))
        logger.info(
            "digest.saved date=%s msgs=%d summary_chars=%d",
            yesterday,
            len(messages),
            len(summary),
        )

        # Embed digest so it becomes semantically recallable.
        if self.vector_memory is not None:
            _spawn_bg(
                self.vector_memory.add(
                    kind="digest",
                    text=summary,
                    source_ref=f"digest:{yesterday}",
                    meta={"date": yesterday, "msg_count": len(messages)},
                ),
                label=f"vec.add.digest:{yesterday}",
            )

    # ── compression ────────────────────────────────────────────────

    async def _maybe_compress(self, session_id: str) -> None:
        """When a session has more than `keep * 2` messages, summarize
        the oldest tail beyond `keep` and delete it from the messages
        table. The summary is merged into `session_summaries`.

        Called once per turn, before loading history. No-op if the
        session is still small.
        """
        keep = self.settings.memory.max_session_turns * 2  # keep N messages
        threshold = keep * 2  # trigger at 2× keep

        total = self.store.count_messages(session_id)
        if total <= threshold:
            return

        delete_n = total - keep
        old_messages = self.store.get_oldest_messages(session_id, delete_n)
        if not old_messages:
            return

        logger.info(
            "compress.trigger session=%s total=%d delete=%d keep=%d",
            session_id,
            total,
            delete_n,
            keep,
        )

        # Render the slice as a simple transcript for the summarizer
        transcript_lines: list[str] = []
        for m in old_messages:
            role = "用户" if m["role"] == "user" else "助手"
            transcript_lines.append(f"{role}：{m['content']}")
        transcript = "\n".join(transcript_lines)

        prior = self.store.get_summary(session_id).strip()

        summarizer_system = (
            "你是一个对话摘要助手。你的任务是把一段助理-用户对话压缩成"
            "简洁的中文摘要，方便助理在后续对话中保留关键上下文。\n\n"
            "要点：\n"
            "- 聚焦：用户偏好、事实、决定、未完成的事项\n"
            "- 省略：寒暄、重复内容、已解决且无长期价值的对话\n"
            "- 长度：不超过 400 字\n"
            "- 格式：纯文本段落或短列表，无标题"
        )

        user_prompt_parts: list[str] = []
        if prior:
            user_prompt_parts.append("之前的摘要（需要与下面新的对话合并）：\n" + prior)
        user_prompt_parts.append("下面是需要并入摘要的新一批对话：\n\n" + transcript)
        user_prompt = "\n\n".join(user_prompt_parts)

        try:
            new_summary = await self.model.chat(
                messages=[{"role": "user", "content": user_prompt}],
                system=summarizer_system,
            )
        except Exception:
            logger.exception("compress.summarize_failed session=%s", session_id)
            return

        if not new_summary or new_summary == "(no response)":
            logger.warning("compress.empty_summary session=%s", session_id)
            return

        self.store.set_summary(session_id, new_summary)
        deleted = self.store.delete_oldest_messages(session_id, delete_n)
        logger.info(
            "compress.done session=%s deleted=%d summary_chars=%d",
            session_id,
            deleted,
            len(new_summary),
        )

    def _tool_specs(self) -> list[dict] | None:
        if not self.tools:
            return None
        return [t.spec() for t in self.tools]

    # ── tool dispatch ──────────────────────────────────────────────

    async def _run_tool(self, name: str, input_data: dict | None) -> str:
        tool = self._by_name.get(name)
        if tool is None:
            logger.warning("tool.unknown name=%s", name)
            return f"ERROR: unknown tool '{name}'"
        try:
            result = await tool.handler(input_data or {})
            if not isinstance(result, str):
                result = str(result)
            logger.info("tool.ok name=%s out_chars=%d", name, len(result))
            return result
        except Exception as e:
            logger.exception("tool.failed name=%s", name)
            return f"ERROR: {type(e).__name__}: {e}"

    # ── main entry ─────────────────────────────────────────────────

    async def handle_user_message(
        self,
        session_id: str,
        source: str,
        peer_id: str,
        user_text: str,
        attachments: list[dict] | None = None,
    ) -> str:
        """Run one user turn.

        `attachments`, if given, is a list of Anthropic content blocks
        (e.g. `{"type": "image", "source": {...}}`) that should be
        attached to THIS user message only. They are not persisted —
        the text stored in SQLite stays as `user_text`, so subsequent
        turns will not re-inject the image bytes. For a plain-text
        history placeholder, `user_text` should already describe the
        attachments (e.g. "[图片：foo.png]").
        """
        self.store.ensure_session(session_id, source, peer_id)
        self.store.append_message(session_id, "user", user_text)

        # Fire-and-forget: embed the user message so future turns can
        # semantically recall it. Never blocks the main flow.
        if self.vector_memory is not None and user_text.strip():
            ts = time.time()
            _spawn_bg(
                self.vector_memory.add(
                    kind="message",
                    text=user_text,
                    source_ref=f"msg:{session_id}:{ts:.0f}:user",
                    meta={"role": "user", "session_id": session_id},
                ),
                label=f"vec.add.user:{session_id}",
            )

        # Generate yesterday's daily digest if we haven't already today.
        await self._maybe_generate_daily_digest()

        # Compress old turns into the rolling summary if the session
        # has grown past the threshold. Done BEFORE loading history so
        # the compressed tail doesn't go into the tool-use loop.
        await self._maybe_compress(session_id)

        # Semantic recall: pull top-k hits across past messages, digests,
        # and markdown memory. Soft-fail — if embedder is down we just
        # get an empty list and move on with the usual prompt.
        recalled: list[dict] = []
        if self.vector_memory is not None and user_text.strip():
            try:
                recalled = await self.vector_memory.search(
                    query=user_text,
                    top_k=self.settings.embedding.recall_top_k,
                    min_score=self.settings.embedding.recall_min_score,
                    kinds=["message", "digest", "md", "note"],
                )
                if recalled:
                    logger.info(
                        "vec.recall session=%s hits=%d top=%.3f",
                        session_id,
                        len(recalled),
                        recalled[0]["score"],
                    )
            except Exception:
                logger.exception("vec.recall_failed session=%s", session_id)
                recalled = []

        history = self.store.get_messages(
            session_id, max_turns=self.settings.memory.max_session_turns
        )
        system = self._build_system(session_id, recalled=recalled)
        tool_specs = self._tool_specs()

        # `messages` is the live list mutated across tool rounds.
        messages: list[dict] = list(history)

        # If this turn has multimodal attachments, rebuild the last
        # user message (which we just appended as plain text) as a
        # list-of-blocks with the attachments prepended.
        if attachments and messages and messages[-1].get("role") == "user":
            text = messages[-1].get("content", "")
            blocks: list[dict] = list(attachments)
            if isinstance(text, str) and text:
                blocks.append({"type": "text", "text": text})
            messages[-1] = {"role": "user", "content": blocks}

        max_iter = self.settings.agent.max_iterations
        logger.info(
            "loop.turn session=%s history_len=%d tools=%d max_iter=%d",
            session_id,
            len(history),
            len(self.tools),
            max_iter,
        )

        final_text: str = ""
        for iteration in range(max_iter):
            response = await self.model.create_message(
                messages=messages,
                system=system or None,
                tools=tool_specs,
            )

            text_parts: list[str] = []
            tool_uses: list[Any] = []
            for block in response.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    text_parts.append(block.text)
                elif btype == "tool_use":
                    tool_uses.append(block)

            if response.stop_reason == "tool_use" and tool_uses:
                # Echo assistant turn (with tool_use blocks) back into the log
                assistant_blocks = [_block_to_dict(b) for b in response.content]
                messages.append({"role": "assistant", "content": assistant_blocks})

                # Run every tool sequentially — MVP, no parallelism
                tool_results: list[dict] = []
                for tu in tool_uses:
                    logger.info(
                        "tool.call iter=%d name=%s id=%s",
                        iteration,
                        tu.name,
                        tu.id,
                    )
                    result_text = await self._run_tool(tu.name, tu.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": result_text,
                        }
                    )
                messages.append({"role": "user", "content": tool_results})
                continue

            # stop_reason in {"end_turn", "max_tokens", "stop_sequence"}
            final_text = "".join(text_parts).strip()
            if not final_text:
                final_text = "(no response)"
            break
        else:
            logger.warning(
                "loop.max_iter_reached session=%s iter=%d",
                session_id,
                max_iter,
            )
            final_text = (
                "（达到最大工具调用轮数，我先把能说的告诉你：无法在本轮完成。）"
            )

        self.store.append_message(session_id, "assistant", final_text)

        # Fire-and-forget embed of the assistant reply.
        if (
            self.vector_memory is not None
            and final_text.strip()
            and final_text != "(no response)"
        ):
            ts = time.time()
            _spawn_bg(
                self.vector_memory.add(
                    kind="message",
                    text=final_text,
                    source_ref=f"msg:{session_id}:{ts:.0f}:assistant",
                    meta={"role": "assistant", "session_id": session_id},
                ),
                label=f"vec.add.assistant:{session_id}",
            )

        return final_text
