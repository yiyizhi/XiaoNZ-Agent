"""Feishu long-connection (WebSocket) client.

Wraps `lark_oapi.ws.Client` so that:
- Incoming `im.message.receive_v1` events are routed to the Agent Loop.
- Outgoing replies go back via `lark_oapi.Client` HTTP API.
- Messages are deduped via `SessionStore.processed_events` so lark
  reconnections / retries don't double-process.

Replies are sent as interactive cards containing a single `markdown`
element. Each turn is a two-step dance:

    1. send a "thinking" card → get card_msg_id
    2. after the agent returns, patch that card with the final markdown

This gives the user a visible "..." placeholder while the LLM + any tool
calls are running, and also supports real markdown rendering (bold,
lists, code fences, …) that Feishu's plain `text` msg_type doesn't.

Threading model
---------------
`lark.ws.Client.start()` is blocking and drives its own asyncio loop
(bound to the thread that imported the module — main thread). The event
handler itself is a plain sync callback.

We want the Agent to be fully async (ModelClient uses AsyncAnthropic),
so we spin up a dedicated "agent loop" in a background thread and
marshal each incoming message via `asyncio.run_coroutine_threadsafe`.
The agent thread performs the LLM call and then sends the reply back
through the sync `lark.Client` HTTP API (blocking in its own thread is
fine — this is a single-user bot).

The main thread then calls `ws_client.start()` and blocks forever.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    GetMessageResourceRequest,
    PatchMessageRequest,
    PatchMessageRequestBody,
)

from ..agent.commands import CommandHandler
from ..agent.loop import AgentLoop
from ..agent.session import SessionStore
from ..agent.turn_context import TurnContext, current_turn
from ..config import Settings

logger = logging.getLogger(__name__)


THINKING_TEXT = "我想想…"

# Inline caps for attachment text injected into the user prompt. Kept
# tight to avoid blowing up context + history-replay cost. Anything
# larger gets saved to disk and exposed via the `anything_to_md` /
# `read_pdf` tools, so the model reads on-demand instead of every turn.
MAX_FILE_BYTES = 60_000       # ≈ 30k Chinese chars, ≈ 20k tokens
MAX_PDF_PAGES = 10            # cap long PDFs at ingest; full read via tool

_UNSAFE_NAME_CHARS = re.compile(r'[/\\:*?"<>|\x00-\x1f]')


def _decode_text_file(data: bytes) -> str | None:
    """Best-effort: decode as UTF-8 and reject anything that looks like
    binary (null bytes / decode failure). Returns the decoded text
    (possibly truncated) or None if it's not plain text."""
    truncated = False
    if len(data) > MAX_FILE_BYTES:
        data = data[:MAX_FILE_BYTES]
        truncated = True
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if "\x00" in text:
        return None
    if truncated:
        text += (
            f"\n\n... （已截断，超过 {MAX_FILE_BYTES // 1000}KB 的"
            f"部分未 inline；如需读全文可让我调 anything_to_md / "
            f"read_local_file 工具读原始文件）"
        )
    return text


def _extract_pdf_text(data: bytes) -> str | None:
    """Extract text from PDF bytes using pymupdf. Returns the text or
    None if extraction fails or yields nothing useful. Capped at
    MAX_PDF_PAGES and MAX_FILE_BYTES — the full PDF stays on disk for
    tool-based retrieval."""
    try:
        import pymupdf
    except ImportError:
        return None
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
        total_pages = len(doc)
        pages: list[str] = []
        for i, page in enumerate(doc):
            if i >= MAX_PDF_PAGES:
                pages.append(
                    f"\n... （只取前 {MAX_PDF_PAGES} 页，共 "
                    f"{total_pages} 页；如需完整内容让我调 read_pdf "
                    f"工具读原始文件）"
                )
                break
            text = page.get_text().strip()
            if text:
                pages.append(f"## 第 {i + 1} 页\n\n{text}")
        doc.close()
    except Exception:
        logger.exception("feishu.pdf_extract_failed")
        return None
    if not pages:
        return None
    full = "\n\n".join(pages)
    if len(full) > MAX_FILE_BYTES:
        full = full[:MAX_FILE_BYTES] + (
            f"\n\n... （正文超过 {MAX_FILE_BYTES // 1000}KB，已截断）"
        )
    return full


def _guess_image_media_type(data: bytes) -> str:
    """Sniff the magic bytes for formats Anthropic accepts. Falls back
    to image/png so the API call at least has a chance."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _parse_text_content(raw: str | None) -> str:
    """Feishu text messages arrive as JSON like `{"text":"hi @_user_1"}`.

    Return the plain text, stripping any @mentions we can detect. If
    parsing fails, return the raw string so the user still sees
    something.
    """
    if not raw:
        return ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()
    text = data.get("text", "")
    if not isinstance(text, str):
        return ""
    # Drop lark @-mention placeholders like "@_user_1 "
    parts = [p for p in text.split() if not p.startswith("@_user_")]
    return " ".join(parts).strip() or text.strip()


class FeishuClient:
    def __init__(
        self,
        settings: Settings,
        store: SessionStore,
        agent: AgentLoop,
        commands: CommandHandler,
    ):
        self.settings = settings
        self.store = store
        self.agent = agent
        self.commands = commands

        # Sync HTTP client for sending replies
        self._http = (
            lark.Client.builder()
            .app_id(settings.feishu.app_id)
            .app_secret(settings.feishu.app_secret)
            .log_level(lark.LogLevel.INFO)
            .build()
        )

        # Background asyncio loop for the agent work
        self._agent_loop: asyncio.AbstractEventLoop | None = None
        self._agent_thread: threading.Thread | None = None

        # Will be built in start()
        self._ws_client: lark.ws.Client | None = None

        # Inflight task tracking, for cancel-via-recall and /cancel.
        # Both dicts are mutated from the agent thread (register/unregister
        # inside `_run_turn`) and read from the ws thread (cancel paths),
        # so all access is protected by `_inflight_lock`.
        # `_inflight` maps message_id → Task; `_inflight_by_session` maps
        # session_id → {message_id, …} so /cancel can find every inflight
        # task belonging to a session.
        self._inflight: dict[str, asyncio.Task] = {}
        self._inflight_by_session: dict[str, set[str]] = {}
        self._inflight_lock = threading.Lock()

    # ── lifecycle ──────────────────────────────────────────────────

    def _start_agent_thread(self) -> None:
        ready = threading.Event()

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._agent_loop = loop
            ready.set()
            loop.run_forever()

        t = threading.Thread(target=_run, name="xiaonz-agent", daemon=True)
        t.start()
        ready.wait()
        self._agent_thread = t
        logger.info("feishu.agent_thread_started")

    def start(self) -> None:
        """Blocking: spin up agent thread then run lark ws.Client in
        the current (main) thread."""
        self._start_agent_thread()

        handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._on_message)
            .register_p2_im_message_recalled_v1(self._on_recalled)
            .build()
        )
        self._ws_client = lark.ws.Client(
            app_id=self.settings.feishu.app_id,
            app_secret=self.settings.feishu.app_secret,
            event_handler=handler,
            log_level=lark.LogLevel.INFO,
        )
        logger.info("feishu.ws_starting")
        self._ws_client.start()  # blocks forever

    # ── inbound ────────────────────────────────────────────────────

    def _on_message(self, event: Any) -> None:
        """Sync callback invoked by lark for every im.message.receive_v1.

        We dispatch the async work onto the agent thread and return
        immediately so the lark loop keeps pumping ping/pong.
        """
        try:
            msg = event.event.message
            sender = event.event.sender
            event_id = getattr(event.header, "event_id", None) or msg.message_id

            if self.store.is_event_processed(event_id):
                logger.info("feishu.dedup event_id=%s", event_id)
                return
            self.store.mark_event_processed(event_id)

            # Parse per-type payload: set user_text + file/image keys.
            user_text = ""
            file_key: str | None = None
            file_name: str | None = None
            image_key: str | None = None

            mtype = msg.message_type
            if mtype == "text":
                user_text = _parse_text_content(msg.content)
                if not user_text:
                    return
            elif mtype == "file":
                try:
                    payload = json.loads(msg.content or "{}")
                except json.JSONDecodeError:
                    payload = {}
                file_key = payload.get("file_key")
                file_name = payload.get("file_name") or "(未命名文件)"
                if not file_key:
                    return
                user_text = f"（我上传了一个文件：{file_name}）"
            elif mtype == "image":
                try:
                    payload = json.loads(msg.content or "{}")
                except json.JSONDecodeError:
                    payload = {}
                image_key = payload.get("image_key")
                if not image_key:
                    return
                user_text = "（我发了一张图片）"
            else:
                logger.info(
                    "feishu.skip_unsupported type=%s msg_id=%s",
                    mtype,
                    msg.message_id,
                )
                return

            chat_type = msg.chat_type  # 'p2p' | 'group'
            if chat_type == "p2p":
                source = "feishu_p2p"
                peer_id = sender.sender_id.open_id or ""
                session_id = f"feishu_p2p:{peer_id}"
                receive_id_type = "open_id"
                receive_id = peer_id
            else:
                source = "feishu_group"
                peer_id = msg.chat_id or ""
                session_id = f"feishu_group:{peer_id}"
                receive_id_type = "chat_id"
                receive_id = peer_id

            logger.info(
                "feishu.recv chat_type=%s peer=%s type=%s chars=%d msg_id=%s",
                chat_type,
                peer_id,
                mtype,
                len(user_text),
                msg.message_id,
            )

            # Slash commands are only routed for text messages (a file
            # upload cannot start with '/').
            if mtype == "text" and self.commands.is_command(user_text):
                self.store.ensure_session(session_id, source, peer_id)
                reply = self.commands.handle(session_id, user_text)
                # Send on the agent thread's executor to avoid blocking
                # the lark ws thread.
                assert self._agent_loop is not None
                asyncio.run_coroutine_threadsafe(
                    self._send_card_async(receive_id, receive_id_type, reply),
                    self._agent_loop,
                )
                return

            assert self._agent_loop is not None
            asyncio.run_coroutine_threadsafe(
                self._run_turn(
                    message_id=msg.message_id,
                    session_id=session_id,
                    source=source,
                    peer_id=peer_id,
                    user_text=user_text,
                    receive_id=receive_id,
                    receive_id_type=receive_id_type,
                    file_key=file_key,
                    file_name=file_name,
                    image_key=image_key,
                ),
                self._agent_loop,
            )
        except Exception:
            logger.exception("feishu._on_message failed")

    def _on_recalled(self, event: Any) -> None:
        """Sync callback: user recalled a previously sent message.

        We dedupe by event_id (so lark retries don't double-cancel)
        then cancel the inflight task for that message_id, if any.
        """
        try:
            event_id = getattr(event.header, "event_id", None)
            if event_id and self.store.is_event_processed(event_id):
                return
            if event_id:
                self.store.mark_event_processed(event_id)

            msg_id = event.event.message_id
            if not msg_id:
                return
            n = self.cancel_message(msg_id)
            logger.info(
                "feishu.recalled msg_id=%s cancelled=%d", msg_id, n
            )
        except Exception:
            logger.exception("feishu._on_recalled failed")

    async def _run_turn(
        self,
        message_id: str,
        session_id: str,
        source: str,
        peer_id: str,
        user_text: str,
        receive_id: str,
        receive_id_type: str,
        file_key: str | None = None,
        file_name: str | None = None,
        image_key: str | None = None,
    ) -> None:
        """Drive one user turn end-to-end: thinking card → agent → patch.

        Also registers the current asyncio.Task in the inflight dict so
        it can be cancelled externally (via /cancel or message recall).
        """
        task = asyncio.current_task()
        with self._inflight_lock:
            self._inflight[message_id] = task  # type: ignore[assignment]
            self._inflight_by_session.setdefault(session_id, set()).add(message_id)

        # Publish the turn context so tools (e.g. send_file_to_feishu)
        # can find the current user's receive_id without explicit wiring.
        # ContextVar is task-local, so no reset needed — dies with the task.
        current_turn.set(
            TurnContext(
                session_id=session_id,
                source=source,
                peer_id=peer_id,
                receive_id=receive_id,
                receive_id_type=receive_id_type,
            )
        )

        loop = asyncio.get_running_loop()
        card_msg_id: str | None = None

        try:
            # 1. Post the thinking-state card so the user immediately sees
            #    that we're working on it.
            card_msg_id = await loop.run_in_executor(
                None,
                self._send_card,
                receive_id,
                receive_id_type,
                THINKING_TEXT,
            )

            # 2. Resolve attachments (download + decode/encode) before
            #    handing off to the agent. Either mutates `user_text`
            #    (for files, inlined as code fence) or produces an
            #    Anthropic image content block.
            attachments: list[dict] = []
            if file_key:
                user_text = await self._resolve_file(
                    loop, message_id, session_id, file_key,
                    file_name or "file", user_text,
                )
            if image_key:
                block = await self._resolve_image(loop, message_id, image_key)
                if block is not None:
                    attachments.append(block)
                else:
                    user_text = f"{user_text}\n\n（图片下载失败）"

            # 3. Run the agent. This is the part that can take a while
            #    and is cancellable via task.cancel().
            reply = await self.agent.handle_user_message(
                session_id=session_id,
                source=source,
                peer_id=peer_id,
                user_text=user_text,
                attachments=attachments or None,
            )

            # 4. Patch the placeholder card with the final markdown reply.
            await self._deliver_reply(
                loop, card_msg_id, receive_id, receive_id_type, reply
            )

        except asyncio.CancelledError:
            logger.info(
                "feishu.turn_cancelled msg_id=%s session=%s",
                message_id,
                session_id,
            )
            # Best-effort: mark the placeholder card as cancelled so the
            # user isn't left staring at "正在思考…". Fire-and-forget
            # because the current task is tearing down — awaiting here
            # would re-raise CancelledError immediately on 3.11+.
            if card_msg_id is not None:
                loop.run_in_executor(
                    None, self._patch_card, card_msg_id, "（已取消）"
                )
            return

        except Exception as e:
            logger.exception("feishu.turn_failed msg_id=%s", message_id)
            err_text = f"抱歉，我这边出了点问题：{e}"
            try:
                await self._deliver_reply(
                    loop, card_msg_id, receive_id, receive_id_type, err_text
                )
            except Exception:
                logger.exception("feishu.err_delivery_failed")

        finally:
            with self._inflight_lock:
                self._inflight.pop(message_id, None)
                bucket = self._inflight_by_session.get(session_id)
                if bucket is not None:
                    bucket.discard(message_id)
                    if not bucket:
                        del self._inflight_by_session[session_id]

    async def _deliver_reply(
        self,
        loop: asyncio.AbstractEventLoop,
        card_msg_id: str | None,
        receive_id: str,
        receive_id_type: str,
        markdown_text: str,
    ) -> None:
        """Finalize a turn: patch the placeholder card, or send fresh
        (card → text) if patching isn't possible."""
        if card_msg_id is not None:
            ok = await loop.run_in_executor(
                None, self._patch_card, card_msg_id, markdown_text
            )
            if ok:
                return
        # No placeholder, or patch failed — send a fresh card.
        new_id = await loop.run_in_executor(
            None,
            self._send_card,
            receive_id,
            receive_id_type,
            markdown_text,
        )
        if new_id is None:
            # Card send also failed — last-resort plain text.
            await loop.run_in_executor(
                None,
                self._send_text,
                receive_id,
                receive_id_type,
                markdown_text,
            )

    # ── cancellation ───────────────────────────────────────────────

    def cancel_message(self, message_id: str) -> int:
        """Cancel the inflight task (if any) for a specific message_id.
        Returns 1 if a task was cancelled, 0 otherwise.

        Safe to call from any thread — uses `call_soon_threadsafe` to
        marshal the cancel onto the agent loop.
        """
        with self._inflight_lock:
            task = self._inflight.get(message_id)
        if task is None or task.done():
            return 0
        if self._agent_loop is None:
            return 0
        self._agent_loop.call_soon_threadsafe(task.cancel)
        return 1

    def cancel_session(self, session_id: str) -> int:
        """Cancel every inflight task for a session. Returns count."""
        with self._inflight_lock:
            msg_ids = list(self._inflight_by_session.get(session_id, set()))
        count = 0
        for mid in msg_ids:
            count += self.cancel_message(mid)
        return count

    # ── attachments ────────────────────────────────────────────────

    def _download_resource(
        self, message_id: str, key: str, resource_type: str
    ) -> bytes | None:
        """Blocking fetch of a file_key or image_key via the
        `im.v1.message_resource.get` endpoint. Returns raw bytes or
        None on failure."""
        req = (
            GetMessageResourceRequest.builder()
            .message_id(message_id)
            .file_key(key)
            .type(resource_type)
            .build()
        )
        resp = self._http.im.v1.message_resource.get(req)
        if not resp.success() or resp.file is None:
            logger.error(
                "feishu.resource_failed type=%s code=%s msg=%s log_id=%s",
                resource_type,
                getattr(resp, "code", "?"),
                getattr(resp, "msg", "?"),
                getattr(resp, "get_log_id", lambda: "?")(),
            )
            return None
        try:
            data = resp.file.read()
        except Exception:
            logger.exception("feishu.resource_read_failed key=%s", key)
            return None
        logger.info(
            "feishu.resource_ok type=%s key=%s bytes=%d",
            resource_type,
            key,
            len(data),
        )
        return data

    def _save_attachment(
        self, session_id: str, file_name: str, data: bytes
    ) -> Path | None:
        """Persist a downloaded attachment under
        `data/attachments/<sanitized_session>/<ts>_<filename>`. Returns
        the absolute path on success, None otherwise.

        Motivation: the inline content in `user_text` is capped tight
        (60 KB) and gets replayed on every turn. Saving the full file
        to disk lets the model fetch it on demand via
        `anything_to_md` / `read_pdf` / `read_local_file` without
        paying the context cost every turn.
        """
        safe_session = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)[:64]
        safe_name = _UNSAFE_NAME_CHARS.sub("_", file_name).strip(" .") or "file"
        ts = time.strftime("%Y%m%d_%H%M%S")
        dest_dir = self.settings.data_dir / "attachments" / safe_session
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{ts}_{safe_name}"
            dest.write_bytes(data)
        except OSError:
            logger.exception(
                "feishu.attachment_save_failed session=%s", session_id
            )
            return None
        logger.info(
            "feishu.attachment_saved session=%s bytes=%d path=%s",
            session_id, len(data), dest,
        )
        return dest

    async def _resolve_file(
        self,
        loop: asyncio.AbstractEventLoop,
        message_id: str,
        session_id: str,
        file_key: str,
        file_name: str,
        user_text: str,
    ) -> str:
        """Download a file attachment, save it to disk, and inline a
        (capped) text preview into `user_text`.

        Flow:
          1. download bytes from Feishu
          2. save to `data/attachments/<session>/<ts>_<name>`
          3. text files: fence + inline (≤ MAX_FILE_BYTES)
          4. PDFs: pymupdf text extract (≤ MAX_PDF_PAGES / MAX_FILE_BYTES)
          5. anything else (docx/xlsx/pptx/epub/…): skip inline, only
             surface the saved path so the model can call
             `anything_to_md` on demand
        """
        data = await loop.run_in_executor(
            None, self._download_resource, message_id, file_key, "file"
        )
        if data is None:
            return f"{user_text}\n\n（附件 {file_name} 下载失败）"

        saved_path = self._save_attachment(session_id, file_name, data)
        location_line = (
            f"\n（原件已保存到 `{saved_path}`，需要完整内容可调 "
            f"`anything_to_md` / `read_pdf` / `read_local_file`）"
            if saved_path else ""
        )

        # Try text decoding first
        decoded = _decode_text_file(data)
        if decoded is not None:
            lang = Path(file_name).suffix.lstrip(".").lower()
            fence = f"```{lang}\n{decoded}\n```" if lang else f"```\n{decoded}\n```"
            return (
                f"{user_text}\n\n# 附件：{file_name}{location_line}\n\n{fence}"
            )

        # Try PDF extraction
        if file_name.lower().endswith(".pdf") or data[:5] == b"%PDF-":
            pdf_text = _extract_pdf_text(data)
            if pdf_text:
                return (
                    f"{user_text}\n\n# 附件：{file_name}{location_line}"
                    f"\n\n{pdf_text}"
                )

        # Unknown binary (docx/xlsx/pptx/epub/csv/…). Don't inline —
        # hand the path over and let the model call `anything_to_md`.
        if saved_path:
            return (
                f"{user_text}\n\n"
                f"# 附件：{file_name}\n"
                f"已保存到 `{saved_path}`\n"
                f"（不是纯文本也不是 PDF，未 inline；"
                f"需要读内容请调用 `anything_to_md` 工具，"
                f"把这个路径作为 `input` 传进去。tomd 支持 "
                f"docx/xlsx/pptx/epub/csv/json 等格式。）"
            )
        return (
            f"{user_text}\n\n"
            f"（附件 {file_name} 既不是纯文本/PDF，保存也失败了；"
            f"可以试试导出成 .md/.txt/.pdf 再发给我。）"
        )

    async def _resolve_image(
        self,
        loop: asyncio.AbstractEventLoop,
        message_id: str,
        image_key: str,
    ) -> dict | None:
        """Download a Feishu image and produce an Anthropic vision
        content block. Returns None on failure.

        NOTE: Some upstream proxies reject `source.type == base64`
        images and ONLY accept `source.type == url` with an HTTPS URL.
        To stay compatible we always stage Feishu images through a
        public anonymous image bed (tmpfiles.org, 60-min retention)
        and pass that URL to the model. If your endpoint accepts
        base64 directly you can simplify and drop the upload step.
        """
        data = await loop.run_in_executor(
            None, self._download_resource, message_id, image_key, "image"
        )
        if data is None:
            return None
        url = await loop.run_in_executor(None, self._upload_image_bed, data)
        if url is None:
            return None
        return {"type": "image", "source": {"type": "url", "url": url}}

    def _upload_image_bed(self, data: bytes) -> str | None:
        """POST bytes to tmpfiles.org and return an HTTPS direct URL.

        tmpfiles returns `http://tmpfiles.org/<id>/<name>` which both
        (a) is HTTP and (b) serves an HTML preview page. We rewrite
        to `https://tmpfiles.org/dl/<id>/<name>` which serves the raw
        file over HTTPS — the form most Anthropic-compatible upstreams
        require for `source.type == url`.
        """
        media_type = _guess_image_media_type(data)
        ext = media_type.split("/")[-1]
        try:
            resp = httpx.post(
                "https://tmpfiles.org/api/v1/upload",
                files={"file": (f"image.{ext}", data, media_type)},
                timeout=20.0,
            )
            resp.raise_for_status()
            body = resp.json()
        except Exception:
            logger.exception("feishu.imgbed_upload_failed")
            return None
        raw_url = (body or {}).get("data", {}).get("url", "")
        if not raw_url:
            logger.error("feishu.imgbed_no_url body=%s", body)
            return None
        # http://tmpfiles.org/ID/name → https://tmpfiles.org/dl/ID/name
        https_url = raw_url.replace("http://", "https://", 1)
        if "tmpfiles.org/" in https_url and "/dl/" not in https_url:
            https_url = https_url.replace(
                "tmpfiles.org/", "tmpfiles.org/dl/", 1
            )
        logger.info(
            "feishu.imgbed_ok bytes=%d media=%s url=%s",
            len(data),
            media_type,
            https_url,
        )
        return https_url

    # ── outbound (cards) ───────────────────────────────────────────

    @staticmethod
    def _preprocess_markdown(text: str) -> str:
        """Feishu 旧版卡片的 markdown 元素不支持 `#` 标题和 GFM 表格：
        标题会原样显示井号；表格整段消失。在发送前把这两类构造转成
        它能渲染的形式（粗体 / 缩进列表）。
        """
        lines = text.split("\n")
        out: list[str] = []
        sep_re = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
        header_re = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

        def split_row(row: str) -> list[str]:
            row = row.strip()
            if row.startswith("|"):
                row = row[1:]
            if row.endswith("|"):
                row = row[:-1]
            return [c.strip() for c in row.split("|")]

        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                "|" in line
                and i + 1 < len(lines)
                and sep_re.match(lines[i + 1])
            ):
                headers = split_row(line)
                i += 2
                while (
                    i < len(lines)
                    and "|" in lines[i]
                    and lines[i].strip()
                ):
                    cells = split_row(lines[i])
                    parts: list[str] = []
                    for idx, cell in enumerate(cells):
                        if not cell:
                            continue
                        h = headers[idx] if idx < len(headers) else ""
                        parts.append(f"**{h}**: {cell}" if h else cell)
                    out.append("- " + "｜".join(parts))
                    i += 1
                continue
            m = header_re.match(line)
            if m:
                out.append(f"**{m.group(2).strip()}**")
            else:
                out.append(line)
            i += 1
        return "\n".join(out)

    @classmethod
    def _build_card(cls, markdown_text: str) -> str:
        """Build the JSON payload for an interactive card holding a
        single markdown element. `update_multi=True` allows the card
        to be patched later via `im.v1.message.patch`.
        """
        card = {
            "config": {"wide_screen_mode": True, "update_multi": True},
            "elements": [
                {"tag": "markdown", "content": cls._preprocess_markdown(markdown_text)}
            ],
        }
        return json.dumps(card, ensure_ascii=False)

    async def _send_card_async(
        self, receive_id: str, receive_id_type: str, markdown_text: str
    ) -> None:
        """Async wrapper for command-reply paths that don't need the
        returned message_id."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._send_card, receive_id, receive_id_type, markdown_text
        )

    def _send_card(
        self, receive_id: str, receive_id_type: str, markdown_text: str
    ) -> str | None:
        """Send an interactive card. Returns the new message_id on
        success, or None on failure so the caller can fall back."""
        body = (
            CreateMessageRequestBody.builder()
            .receive_id(receive_id)
            .msg_type("interactive")
            .content(self._build_card(markdown_text))
            .build()
        )
        req = (
            CreateMessageRequest.builder()
            .receive_id_type(receive_id_type)
            .request_body(body)
            .build()
        )
        resp = self._http.im.v1.message.create(req)
        if not resp.success():
            logger.error(
                "feishu.card_send_failed code=%s msg=%s log_id=%s",
                resp.code,
                resp.msg,
                getattr(resp, "get_log_id", lambda: "?")(),
            )
            return None
        msg_id = getattr(resp.data, "message_id", None) if resp.data else None
        logger.info(
            "feishu.card_sent chars=%d msg_id=%s to=%s",
            len(markdown_text),
            msg_id,
            receive_id,
        )
        return msg_id

    def _patch_card(self, message_id: str, markdown_text: str) -> bool:
        """Replace the body of a previously-sent card. Returns True on
        success, False otherwise (so caller can fall back to a fresh
        message)."""
        body = (
            PatchMessageRequestBody.builder()
            .content(self._build_card(markdown_text))
            .build()
        )
        req = (
            PatchMessageRequest.builder()
            .message_id(message_id)
            .request_body(body)
            .build()
        )
        resp = self._http.im.v1.message.patch(req)
        if not resp.success():
            logger.error(
                "feishu.card_patch_failed code=%s msg=%s log_id=%s",
                resp.code,
                resp.msg,
                getattr(resp, "get_log_id", lambda: "?")(),
            )
            return False
        logger.info(
            "feishu.card_patched chars=%d msg_id=%s",
            len(markdown_text),
            message_id,
        )
        return True

    # ── outbound (text fallback) ───────────────────────────────────

    def _send_text(
        self, receive_id: str, receive_id_type: str, text: str
    ) -> None:
        """Plain-text fallback when cards fail. Used only as a last
        resort; the default outbound path is `_send_card`."""
        body = (
            CreateMessageRequestBody.builder()
            .receive_id(receive_id)
            .msg_type("text")
            .content(json.dumps({"text": text}, ensure_ascii=False))
            .build()
        )
        req = (
            CreateMessageRequest.builder()
            .receive_id_type(receive_id_type)
            .request_body(body)
            .build()
        )
        resp = self._http.im.v1.message.create(req)
        if not resp.success():
            logger.error(
                "feishu.send_failed code=%s msg=%s log_id=%s",
                resp.code,
                resp.msg,
                getattr(resp, "get_log_id", lambda: "?")(),
            )
        else:
            logger.info("feishu.sent chars=%d to=%s", len(text), receive_id)
