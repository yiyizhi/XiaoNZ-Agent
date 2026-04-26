"""Concrete Tool instances wired to MemoryStore / SkillStore.

Kept separate from `tools.py` (which only defines the Tool dataclass)
so that the dataclass has no dependency on concrete stores.

Each factory takes the store it needs and returns a `Tool`. Call sites
collect them into a list passed to AgentLoop.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import re
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import html2text
import httpx
from lxml import html as lxml_html

from .memory import MemoryStore
from .skills import SkillStore
from .tools import Tool
from .turn_context import current_turn

logger = logging.getLogger(__name__)


# Browser-ish UA so sites that block python-requests still serve us.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Hard limits for web_fetch so a single call can't blow up the context.
_FETCH_MAX_BYTES = 2_000_000  # 2 MB raw body
_FETCH_MAX_OUT_CHARS = 8_000  # post-conversion truncation


# ── skills ─────────────────────────────────────────────────────────

def make_list_skills_tool(skills: SkillStore) -> Tool:
    async def handler(_: dict) -> str:
        items = skills.list_all()
        if not items:
            return "No skills installed."
        lines = []
        for s in items:
            desc = s.description or "(no description)"
            lines.append(f"- {s.name}: {desc}")
        return "\n".join(lines)

    return Tool(
        name="list_skills",
        description=(
            "List all available skills with their short descriptions. "
            "Call this first when you suspect a skill might help, then "
            "call load_skill to read the one you want."
        ),
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )


def make_load_skill_tool(skills: SkillStore) -> Tool:
    async def handler(input_data: dict) -> str:
        name = (input_data.get("name") or "").strip()
        if not name:
            return "ERROR: 'name' is required."
        skill = skills.get(name)
        if skill is None:
            available = ", ".join(s.name for s in skills.list_all()) or "(none)"
            return (
                f"ERROR: skill '{name}' not found. "
                f"Available: {available}"
            )
        return (
            f"# Skill: {skill.name}\n"
            f"{skill.description}\n\n"
            f"{skill.body}"
        )

    return Tool(
        name="load_skill",
        description=(
            "Load the full body of a skill by name. Returns the skill's "
            "complete instructions as markdown. Call list_skills first "
            "if you don't know the name."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact skill name from list_skills.",
                },
            },
            "required": ["name"],
        },
        handler=handler,
    )


def make_install_skill_tool(skills: SkillStore) -> Tool:
    async def handler(input_data: dict) -> str:
        name = (input_data.get("name") or "").strip()
        content = (input_data.get("content") or "").strip()
        if not name:
            return "ERROR: 'name' is required."
        if not content:
            return "ERROR: 'content' is required (the full SKILL.md text)."
        # Sanitize name: lowercase, hyphens only
        safe_name = re.sub(r"[^a-z0-9\-]", "-", name.lower()).strip("-")
        if not safe_name:
            return "ERROR: invalid skill name."
        skill_dir = skills.dir / safe_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(content, encoding="utf-8")
        # Verify it parses
        s = skills.get(safe_name)
        if s is None:
            return f"WARNING: 文件已写入 {skill_file}，但解析失败，请检查 frontmatter 格式。"
        return f"OK: 技能 '{s.name}' 已安装。描述：{s.description}"

    return Tool(
        name="install_skill",
        description=(
            "Install a new skill by writing its SKILL.md content to "
            "the skills directory. The content should be a complete "
            "SKILL.md file with YAML frontmatter (name, description) "
            "and a markdown body. Use this when the user asks to "
            "install a skill — you can fetch the content from a URL "
            "with web_fetch first, then pass it here. Works with any "
            "AgentSkills-format SKILL.md (e.g. from a shared skills "
            "repo or SkillHub)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Skill name (lowercase, hyphens). Will be used "
                        "as the directory name under data/skills/."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": (
                        "Complete SKILL.md content including YAML "
                        "frontmatter (---\\nname: ...\\ndescription: "
                        "...\\n---) and markdown body."
                    ),
                },
            },
            "required": ["name", "content"],
        },
        handler=handler,
    )


def make_uninstall_skill_tool(skills: SkillStore) -> Tool:
    async def handler(input_data: dict) -> str:
        name = (input_data.get("name") or "").strip()
        if not name:
            return "ERROR: 'name' is required."
        skill = skills.get(name)
        if skill is None:
            available = ", ".join(s.name for s in skills.list_all()) or "(none)"
            return f"ERROR: 技能 '{name}' 不存在。已安装的：{available}"
        import shutil
        skill_dir = skill.path.parent
        try:
            shutil.rmtree(str(skill_dir))
        except Exception as e:
            return f"ERROR: 删除失败：{e}"
        return f"OK: 技能 '{name}' 已卸载。"

    return Tool(
        name="uninstall_skill",
        description=(
            "Uninstall (delete) an installed skill by name. Removes "
            "the entire skill directory. Use when the user asks to "
            "remove a skill."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill to uninstall.",
                },
            },
            "required": ["name"],
        },
        handler=handler,
    )


# ── memory ─────────────────────────────────────────────────────────

def make_update_memory_tool(memory: MemoryStore) -> Tool:
    async def handler(input_data: dict) -> str:
        new_content = input_data.get("new_content")
        if not isinstance(new_content, str) or not new_content.strip():
            return "ERROR: 'new_content' must be a non-empty string."
        size = memory.write(new_content)
        return f"OK: MEMORY.md updated ({size} bytes)."

    return Tool(
        name="update_memory",
        description=(
            "Overwrite the entire long-term memory file (MEMORY.md) with "
            "new_content. Use this to persist facts the user wants you "
            "to remember across sessions (preferences, names, stable "
            "project context). Always include the FULL updated memory "
            "— partial writes will erase previous memory. Prefer Chinese "
            "for the content unless the user writes in English."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "new_content": {
                    "type": "string",
                    "description": (
                        "Complete replacement content for MEMORY.md. "
                        "Should be markdown. Organize by topic with "
                        "clear headings."
                    ),
                },
            },
            "required": ["new_content"],
        },
        handler=handler,
    )


# ── web ────────────────────────────────────────────────────────────

def _sync_web_fetch(url: str) -> str:
    """Blocking GET + light content conversion. Returns a single
    text blob, already truncated. Called from a thread via
    `asyncio.to_thread` so it doesn't block the agent loop.
    """
    try:
        with httpx.Client(
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": _BROWSER_UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
        ) as c:
            resp = c.get(url)
    except httpx.HTTPError as e:
        return f"ERROR: failed to fetch {url}: {type(e).__name__}: {e}"

    ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    body = resp.content[:_FETCH_MAX_BYTES]

    if ctype.startswith("text/html") or ctype.startswith("application/xhtml"):
        try:
            text = body.decode(resp.encoding or "utf-8", errors="replace")
        except LookupError:
            text = body.decode("utf-8", errors="replace")
        h = html2text.HTML2Text()
        h.ignore_images = True
        h.ignore_emphasis = False
        h.body_width = 0  # no hard wrapping
        out = h.handle(text).strip()
    elif (
        ctype.startswith("text/")
        or ctype.startswith("application/json")
        or ctype.startswith("application/xml")
        or ctype.startswith("application/yaml")
    ):
        try:
            out = body.decode(resp.encoding or "utf-8", errors="replace")
        except LookupError:
            out = body.decode("utf-8", errors="replace")
    else:
        return (
            f"HTTP {resp.status_code} {ctype or '(unknown)'} "
            f"bytes={len(resp.content)}\n\n"
            f"ERROR: 不支持的 content-type（只支持 text/* 和 application/json|xml|yaml）。"
        )

    truncated = ""
    if len(out) > _FETCH_MAX_OUT_CHARS:
        out = out[:_FETCH_MAX_OUT_CHARS]
        truncated = f"\n\n... (已截断，原文超过 {_FETCH_MAX_OUT_CHARS} 字符)"

    return (
        f"HTTP {resp.status_code} {ctype} final_url={str(resp.url)}\n\n"
        f"{out}{truncated}"
    )


def make_web_fetch_tool() -> Tool:
    async def handler(input_data: dict) -> str:
        url = (input_data.get("url") or "").strip()
        if not url:
            return "ERROR: 'url' is required."
        if not (url.startswith("http://") or url.startswith("https://")):
            return f"ERROR: url must start with http:// or https:// (got {url!r})"
        logger.info("tool.web_fetch url=%s", url)
        return await asyncio.to_thread(_sync_web_fetch, url)

    return Tool(
        name="web_fetch",
        description=(
            "Fetch the contents of a URL and return it as text. HTML "
            "pages are converted to markdown; plain-text and JSON are "
            "returned as-is. Use this when the user asks about a "
            "specific page, or when web_search returns an interesting "
            "link you want to read in full. Content is truncated to "
            "~8000 chars. Only text/* and application/json|xml|yaml "
            "content types are supported — binary files will be rejected."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Absolute http(s) URL to fetch.",
                },
            },
            "required": ["url"],
        },
        handler=handler,
    )


def _sync_web_search(query: str, max_results: int) -> str:
    """Scrape DuckDuckGo's plain-HTML endpoint. Returns a
    human-readable numbered list, one hit per entry with title / URL
    / snippet. No API key, no JSON, but works reliably on Python 3.9
    + LibreSSL where the `ddgs` package's TLS 1.3 requirement breaks."""
    try:
        with httpx.Client(
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": _BROWSER_UA},
        ) as c:
            resp = c.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
            )
    except httpx.HTTPError as e:
        return f"ERROR: web_search failed: {type(e).__name__}: {e}"

    if resp.status_code != 200:
        return f"ERROR: web_search got HTTP {resp.status_code}"

    try:
        doc = lxml_html.fromstring(resp.content)
    except Exception as e:
        return f"ERROR: web_search couldn't parse response: {e}"

    result_nodes = doc.xpath(
        "//div[contains(@class, 'result') and not(contains(@class, 'result--ad'))]"
    )

    hits: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for node in result_nodes:
        links = node.xpath(".//a[contains(@class, 'result__a')]")
        if not links:
            continue
        a = links[0]
        raw_href = a.get("href", "")
        if not raw_href:
            continue
        href = raw_href
        if href.startswith("//"):
            href = "https:" + href
        if href.startswith("/l/"):
            href = "https://duckduckgo.com" + href
        # DDG wraps external links as /l/?uddg=<encoded>
        if "duckduckgo.com/l/" in href:
            q = parse_qs(urlparse(href).query)
            uddg = q.get("uddg")
            if uddg:
                href = unquote(uddg[0])
        # Skip ad redirects
        if "/y.js" in href or "duckduckgo.com/y.js" in href:
            continue
        if href in seen:
            continue
        seen.add(href)
        title = a.text_content().strip()
        snip_nodes = node.xpath(".//*[contains(@class, 'result__snippet')]")
        snippet = (
            snip_nodes[0].text_content().strip() if snip_nodes else ""
        )
        hits.append((title, href, snippet))
        if len(hits) >= max_results:
            break

    if not hits:
        return f"(no results for {query!r})"

    lines = [f"# 搜索结果：{query}", ""]
    for i, (title, href, snippet) in enumerate(hits, 1):
        lines.append(f"{i}. **{title}**")
        lines.append(f"   {href}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")
    return "\n".join(lines).strip()


def make_web_search_tool() -> Tool:
    async def handler(input_data: dict) -> str:
        query = (input_data.get("query") or "").strip()
        if not query:
            return "ERROR: 'query' is required."
        logger.info("tool.web_search query=%s disabled=true", query)
        return (
            "web_search 暂不可用（DuckDuckGo 端点在当前网络不可达）。"
            "请不要再调用 web_search 或 web_fetch，"
            "直接基于已掌握的信息回答用户；"
            "如果确实需要最新数据，告诉用户能力暂时不可用。"
        )

    return Tool(
        name="web_search",
        description=(
            "Search the web via DuckDuckGo and return the top results "
            "(title + URL + snippet). Use this when the user asks about "
            "something that needs fresh information (news, versions, "
            "docs, prices, etc.) or when you're unsure and want to "
            "verify before answering. Follow up with web_fetch on the "
            "most relevant link to read the full page."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query. Plain natural language works; "
                        "for best results be specific and include "
                        "distinctive keywords."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "1–10, default 5.",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
        handler=handler,
    )


# ── local filesystem ──────────────────────────────────────────────

# Cap downloaded files at 50 MB — the agent is running on the user's
# own Mac, but we still don't want a runaway LLM to fill their disk.
_DOWNLOAD_MAX_BYTES = 50 * 1024 * 1024

# Default save directory when the model doesn't pass an absolute path.
_DEFAULT_DOWNLOAD_DIR = Path.home() / "Downloads"

# Extension inferred from content-type when the URL has no recognizable
# extension of its own. Keep this small — only the formats we actually
# want to auto-rename.
_CT_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
    "application/pdf": ".pdf",
    "application/json": ".json",
    "text/plain": ".txt",
    "text/html": ".html",
    "text/markdown": ".md",
}

_UNSAFE_NAME_CHARS = re.compile(r'[/\\:*?"<>|\x00-\x1f]')


def _safe_filename(name: str) -> str:
    name = _UNSAFE_NAME_CHARS.sub("_", name).strip(" .")
    return name or "download"


def _derive_filename(url: str, content_type: str) -> str:
    """Pick a reasonable filename from a URL + content-type. Strips
    query strings, ensures a sensible extension."""
    parsed = urlparse(url)
    base = Path(parsed.path).name
    base = _safe_filename(base)
    if not base or base == "download":
        base = "download"
    if "." not in base:
        ext = _CT_TO_EXT.get(content_type.split(";")[0].strip().lower(), "")
        base += ext
    return base


def _resolve_save_path(save_path: str | None, url: str, content_type: str) -> Path:
    """Turn the caller's (possibly None) save_path into an absolute
    Path. Rules:
      - None → ~/Downloads/<filename from url>
      - relative path → ~/Downloads/<that path>
      - absolute path ending in a separator or pointing at an existing
        directory → <dir>/<filename from url>
      - anything else → treat as the full target path
      - `~` is expanded.
    """
    filename = _derive_filename(url, content_type)
    if not save_path:
        return _DEFAULT_DOWNLOAD_DIR / filename
    p = Path(save_path).expanduser()
    if not p.is_absolute():
        p = _DEFAULT_DOWNLOAD_DIR / p
    # Treat trailing slash or existing-dir as "put the file in here"
    looks_like_dir = save_path.endswith(("/", "\\")) or (p.exists() and p.is_dir())
    if looks_like_dir:
        return p / filename
    return p


def _sync_download_to_disk(url: str, save_path: str | None) -> str:
    try:
        with httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": _BROWSER_UA},
        ) as c:
            with c.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    return f"ERROR: HTTP {resp.status_code} for {url}"
                ctype = (resp.headers.get("content-type") or "").strip()
                target = _resolve_save_path(save_path, str(resp.url), ctype)
                target.parent.mkdir(parents=True, exist_ok=True)
                total = 0
                with target.open("wb") as out:
                    for chunk in resp.iter_bytes():
                        total += len(chunk)
                        if total > _DOWNLOAD_MAX_BYTES:
                            out.close()
                            try:
                                target.unlink()
                            except OSError:
                                pass
                            return (
                                f"ERROR: 文件超过 {_DOWNLOAD_MAX_BYTES // (1024 * 1024)} MB 上限，已中止并删除。"
                            )
                        out.write(chunk)
    except httpx.HTTPError as e:
        return f"ERROR: 下载失败 {type(e).__name__}: {e}"
    except OSError as e:
        return f"ERROR: 写文件失败 {type(e).__name__}: {e}"
    return f"OK: 已保存到 {target} ({total} 字节)"


def make_download_to_disk_tool() -> Tool:
    async def handler(input_data: dict) -> str:
        url = (input_data.get("url") or "").strip()
        if not url:
            return "ERROR: 'url' is required."
        if not (url.startswith("http://") or url.startswith("https://")):
            return f"ERROR: url must start with http(s):// (got {url!r})"
        save_path = input_data.get("save_path")
        if save_path is not None and not isinstance(save_path, str):
            return "ERROR: 'save_path' must be a string if provided."
        logger.info("tool.download_to_disk url=%s save_path=%s", url, save_path)
        return await asyncio.to_thread(_sync_download_to_disk, url, save_path)

    return Tool(
        name="download_to_disk",
        description=(
            "Download a file or image from an http(s) URL and save it "
            "to the user's local filesystem. Default save directory is "
            "~/Downloads/. Use this when the user asks you to save "
            "something from the web onto their computer (e.g. "
            "'下载这张图保存到我电脑'). Returns the final absolute "
            "path on success. Max 50 MB per file."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Absolute http(s) URL to download.",
                },
                "save_path": {
                    "type": "string",
                    "description": (
                        "Optional. Target path on the user's filesystem. "
                        "Supports '~'. If it looks like a directory "
                        "(trailing slash or existing dir), the filename "
                        "is derived from the URL. If relative, it's "
                        "resolved under ~/Downloads/. Omit to drop the "
                        "file straight into ~/Downloads/ with a name "
                        "inferred from the URL."
                    ),
                },
            },
            "required": ["url"],
        },
        handler=handler,
    )


# ── feishu file sending ───────────────────────────────────────────

# Feishu file_type accepted values for im.v1.file.create
_OFFICE_EXT_TO_TYPE = {
    ".pdf": "pdf",
    ".doc": "doc",
    ".docx": "docx",
    ".xls": "xls",
    ".xlsx": "xlsx",
    ".ppt": "ppt",
    ".pptx": "pptx",
    ".mp4": "mp4",
    ".opus": "opus",
}


def _infer_feishu_file_type(path: Path) -> str:
    return _OFFICE_EXT_TO_TYPE.get(path.suffix.lower(), "stream")


def _is_image_path(path: Path) -> bool:
    mime, _ = mimetypes.guess_type(path.name)
    return bool(mime and mime.startswith("image/"))


def make_send_to_feishu_tool(feishu: Any) -> Tool:
    """Build a tool that uploads a local file and sends it as a Feishu
    message to the current conversation. `feishu` must expose
    `_http` (the lark.Client) and is only used for that purpose.
    """
    from lark_oapi.api.im.v1 import (
        CreateFileRequest,
        CreateFileRequestBody,
        CreateImageRequest,
        CreateImageRequestBody,
        CreateMessageRequest,
        CreateMessageRequestBody,
    )

    # Feishu upload limits: image/create ≤ 10 MB, file/create ≤ 30 MB.
    # We pre-check so the lark SDK doesn't blow up on a non-JSON error
    # page (which is what happened 2026-04-24 with a 36 MB zip).
    _IMAGE_MAX_BYTES = 10 * 1024 * 1024
    _FILE_MAX_BYTES = 30 * 1024 * 1024

    def _sync_send(path_str: str) -> str:
        ctx = current_turn.get()
        if ctx is None:
            return (
                "ERROR: 没有当前对话上下文，无法发送。这个工具只能在"
                "响应用户消息的回合里调用。"
            )

        p = Path(path_str).expanduser()
        if not p.exists():
            return f"ERROR: 文件不存在：{p}"
        if not p.is_file():
            return f"ERROR: 不是文件：{p}"

        size = p.stat().st_size
        is_image = _is_image_path(p)
        limit = _IMAGE_MAX_BYTES if is_image else _FILE_MAX_BYTES
        if size > limit:
            kind = "图片" if is_image else "文件"
            logger.warning(
                "tool.send_to_feishu.too_big path=%s size=%d limit=%d",
                p, size, limit,
            )
            return (
                f"ERROR: {kind}太大（{size / 1024 / 1024:.1f} MB），"
                f"超过飞书 {kind}上限 {limit // 1024 // 1024} MB。"
                f"请改成分批发送、压缩瘦身，或走外链。不要重试同一个文件。"
            )

        client = feishu._http  # lark.Client

        try:
            if is_image:
                with p.open("rb") as f:
                    img_body = (
                        CreateImageRequestBody.builder()
                        .image_type("message")
                        .image(f)
                        .build()
                    )
                    img_req = (
                        CreateImageRequest.builder()
                        .request_body(img_body)
                        .build()
                    )
                    img_resp = client.im.v1.image.create(img_req)
                if not img_resp.success():
                    return (
                        f"ERROR: 图片上传失败 code={img_resp.code} "
                        f"msg={img_resp.msg}"
                    )
                image_key = getattr(img_resp.data, "image_key", None)
                if not image_key:
                    return "ERROR: 图片上传后未返回 image_key"
                msg_type = "image"
                content = json.dumps({"image_key": image_key})
            else:
                file_type = _infer_feishu_file_type(p)
                with p.open("rb") as f:
                    file_body = (
                        CreateFileRequestBody.builder()
                        .file_type(file_type)
                        .file_name(p.name)
                        .file(f)
                        .build()
                    )
                    file_req = (
                        CreateFileRequest.builder()
                        .request_body(file_body)
                        .build()
                    )
                    file_resp = client.im.v1.file.create(file_req)
                if not file_resp.success():
                    return (
                        f"ERROR: 文件上传失败 code={file_resp.code} "
                        f"msg={file_resp.msg}"
                    )
                file_key = getattr(file_resp.data, "file_key", None)
                if not file_key:
                    return "ERROR: 文件上传后未返回 file_key"
                msg_type = "file"
                content = json.dumps({"file_key": file_key})

            send_body = (
                CreateMessageRequestBody.builder()
                .receive_id(ctx.receive_id)
                .msg_type(msg_type)
                .content(content)
                .build()
            )
            send_req = (
                CreateMessageRequest.builder()
                .receive_id_type(ctx.receive_id_type)
                .request_body(send_body)
                .build()
            )
            send_resp = client.im.v1.message.create(send_req)
        except Exception as e:
            logger.exception("tool.send_to_feishu_failed path=%s", p)
            return f"ERROR: {type(e).__name__}: {e}"

        if not send_resp.success():
            return (
                f"ERROR: 消息发送失败 code={send_resp.code} "
                f"msg={send_resp.msg}"
            )
        return f"OK: 已把 {p.name} 发到飞书。"

    async def handler(input_data: dict) -> str:
        path_str = (input_data.get("path") or "").strip()
        if not path_str:
            return "ERROR: 'path' is required."
        logger.info("tool.send_to_feishu path=%s", path_str)
        return await asyncio.to_thread(_sync_send, path_str)

    return Tool(
        name="send_to_feishu",
        description=(
            "Upload a local file from the user's computer and send it "
            "as a message in the current Feishu conversation. Auto-"
            "detects whether to send as image (png/jpg/webp/...) or "
            "file (anything else). Use this when the user asks you to "
            "send a file or image from their computer to their Feishu "
            "chat (e.g. '把这个文件发到我飞书'). The path can be "
            "absolute or use '~'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Absolute path (or path with ~) to a local "
                        "file on the user's computer."
                    ),
                },
            },
            "required": ["path"],
        },
        handler=handler,
    )


# ── image generation (OpenAI-compatible /v1/images/generations) ──

_IMAGE_GEN_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}


def make_generate_image_tool(settings: Any, feishu: Any) -> Tool:
    """Build a tool that generates an image via the gateway's OpenAI-
    compatible /v1/images/generations endpoint and sends it directly
    to the current Feishu conversation. Requires
    `settings.model.openai_auth_token` to be set; call site should
    skip registering this tool when it's empty.
    """
    from lark_oapi.api.im.v1 import (
        CreateImageRequest,
        CreateImageRequestBody,
        CreateMessageRequest,
        CreateMessageRequestBody,
    )

    base_url = settings.model.base_url.rstrip("/")
    auth_token = settings.model.openai_auth_token
    model = settings.model.image_model
    gen_dir = settings.data_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Tail appended to every ERROR return to stop Claude from silently
    # retrying with a new prompt when the first call fails. Without this
    # the model sees an error and treats it as "try again" — producing
    # the "发图发得没完没了" behaviour the user reported on 2026-04-24.
    _NO_RETRY = (
        "\n[重要：不要自动换 prompt 重试。把这条错误原样告诉用户，"
        "然后停下来等用户指示。]"
    )

    # Hard cap on images per single user turn. Session on 2026-04-24
    # generated 10 images from one "继续" message (whole office render
    # site) and spammed Feishu. Past the cap we refuse and instruct the
    # model to ask the user for explicit go-ahead.
    _MAX_IMAGES_PER_TURN = 5

    def _err(msg: str) -> str:
        logger.warning("tool.generate_image.failed reason=%s", msg)
        return msg + _NO_RETRY

    def _sync_generate_and_send(prompt: str, size: str) -> str:
        ctx = current_turn.get()
        if ctx is None:
            return _err(
                "ERROR: 没有当前对话上下文，无法发送。这个工具只能在"
                "响应用户消息的回合里调用。"
            )

        with ctx.lock:
            if ctx.generate_image_count >= _MAX_IMAGES_PER_TURN:
                logger.warning(
                    "tool.generate_image.cap_hit session=%s count=%d cap=%d",
                    ctx.session_id, ctx.generate_image_count,
                    _MAX_IMAGES_PER_TURN,
                )
                return _err(
                    f"ERROR: 这一轮已经生成过 {ctx.generate_image_count} "
                    f"张图了，达到单轮上限 {_MAX_IMAGES_PER_TURN} 张。"
                    f"请先把已有的图发给用户看，让用户确认是否继续生成。"
                )
            ctx.generate_image_count += 1
            current_count = ctx.generate_image_count

        logger.info(
            "tool.generate_image.progress session=%s count=%d/%d",
            ctx.session_id, current_count, _MAX_IMAGES_PER_TURN,
        )

        try:
            resp = httpx.post(
                f"{base_url}/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {auth_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "prompt": prompt,
                    "n": 1,
                    "size": size,
                },
                timeout=180.0,
            )
        except httpx.HTTPError as e:
            return _err(f"ERROR: 生图请求失败 {type(e).__name__}: {e}")

        if resp.status_code != 200:
            return _err(
                f"ERROR: 生图网关 HTTP {resp.status_code}: "
                f"{resp.text[:300]}"
            )
        try:
            data = resp.json()
        except Exception as e:
            return _err(f"ERROR: 生图返回不是合法 JSON: {e}")

        if "error" in data:
            return _err(f"ERROR: 生图失败 {data['error']}")
        items = data.get("data") or []
        if not items or "b64_json" not in items[0]:
            return _err(f"ERROR: 生图返回缺 b64_json: {str(data)[:300]}")

        b64 = items[0]["b64_json"]
        revised = items[0].get("revised_prompt") or ""
        try:
            raw = base64.b64decode(b64)
        except Exception as e:
            return _err(f"ERROR: base64 解码失败: {e}")

        ts = time.strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        path = gen_dir / f"{ts}_{uid}.png"
        try:
            path.write_bytes(raw)
        except OSError as e:
            return _err(f"ERROR: 写文件失败 {e}")

        client = feishu._http
        try:
            with path.open("rb") as f:
                img_body = (
                    CreateImageRequestBody.builder()
                    .image_type("message")
                    .image(f)
                    .build()
                )
                img_req = (
                    CreateImageRequest.builder()
                    .request_body(img_body)
                    .build()
                )
                img_resp = client.im.v1.image.create(img_req)
            if not img_resp.success():
                return _err(
                    f"ERROR: 图片上传失败 code={img_resp.code} "
                    f"msg={img_resp.msg}"
                )
            image_key = getattr(img_resp.data, "image_key", None)
            if not image_key:
                return _err("ERROR: 图片上传后未返回 image_key")

            send_body = (
                CreateMessageRequestBody.builder()
                .receive_id(ctx.receive_id)
                .msg_type("image")
                .content(json.dumps({"image_key": image_key}))
                .build()
            )
            send_req = (
                CreateMessageRequest.builder()
                .receive_id_type(ctx.receive_id_type)
                .request_body(send_body)
                .build()
            )
            send_resp = client.im.v1.message.create(send_req)
        except Exception as e:
            logger.exception("tool.generate_image.send_failed path=%s", path)
            return _err(f"ERROR: 飞书发送失败 {type(e).__name__}: {e}")

        if not send_resp.success():
            return _err(
                f"ERROR: 飞书消息发送失败 code={send_resp.code} "
                f"msg={send_resp.msg}"
            )

        out = f"OK: 图已生成（{len(raw)} 字节）并发到飞书，本地存档 {path}"
        if revised and revised.strip() != prompt.strip():
            out += f"\n模型改写后的 prompt：{revised}"
        return out

    async def handler(input_data: dict) -> str:
        prompt = (input_data.get("prompt") or "").strip()
        if not prompt:
            return "ERROR: 'prompt' is required."
        size = (input_data.get("size") or "1024x1024").strip()
        if size not in _IMAGE_GEN_SIZES:
            return (
                f"ERROR: size 只支持 {sorted(_IMAGE_GEN_SIZES)}，"
                f"收到 {size!r}"
            )
        logger.info(
            "tool.generate_image model=%s size=%s prompt=%r",
            model, size, prompt[:120],
        )
        return await asyncio.to_thread(_sync_generate_and_send, prompt, size)

    return Tool(
        name="generate_image",
        description=(
            "Generate an image with OpenAI gpt-image-2 via the internal "
            "gateway and send it directly to the current Feishu "
            "conversation. Use when the user asks you to draw / "
            "generate / make an image (e.g. '画一张秋天的图', "
            "'给我生成一张 logo'). The PNG is also saved locally for "
            "record-keeping. Longer and more specific prompts give "
            "better results; Chinese or English both work.\n\n"
            "IMPORTANT — only call this tool ONCE per user request by "
            "default (one image per ask). If the user explicitly asks "
            "for N images or for alternates, call it N times with "
            "different prompts; otherwise do NOT generate multiple "
            "variations on your own. If the tool returns ERROR, DO NOT "
            "retry with a different prompt — surface the error to the "
            "user and stop."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Detailed description of the image to generate."
                    ),
                },
                "size": {
                    "type": "string",
                    "enum": sorted(_IMAGE_GEN_SIZES),
                    "description": (
                        "Image dimensions. 1024x1024 (default, square), "
                        "1024x1536 (vertical), 1536x1024 (horizontal), "
                        "'auto' lets the model decide."
                    ),
                },
            },
            "required": ["prompt"],
        },
        handler=handler,
    )


# ── PDF extraction ────────────────────────────────────────────────

_PDF_MAX_CHARS = 60_000


def make_read_pdf_tool() -> Tool:
    async def handler(input_data: dict) -> str:
        raw = (input_data.get("path") or "").strip()
        if not raw:
            return "ERROR: 'path' is required."
        p = Path(raw).expanduser()
        if not p.is_file():
            return f"ERROR: 文件不存在：{p}"
        try:
            import pymupdf
        except ImportError:
            return "ERROR: pymupdf 未安装，无法读取 PDF。"
        try:
            doc = pymupdf.open(str(p))
            pages: list[str] = []
            for i, page in enumerate(doc):
                text = page.get_text().strip()
                if text:
                    pages.append(f"## 第 {i + 1} 页\n\n{text}")
            doc.close()
        except Exception as e:
            return f"ERROR: PDF 解析失败：{e}"
        if not pages:
            return f"PDF 没有可提取的文字（可能是扫描件/纯图片 PDF）。"
        full = "\n\n".join(pages)
        if len(full) > _PDF_MAX_CHARS:
            full = full[:_PDF_MAX_CHARS] + f"\n\n... (已截断，原文 {len(full)} 字符)"
        return full

    return Tool(
        name="read_pdf",
        description=(
            "Extract text from a local PDF file. Returns the text "
            "content organized by page. Works for text-based PDFs; "
            "scanned/image-only PDFs will return empty. Use when the "
            "user asks you to read a PDF on their computer (e.g. "
            "'帮我看看桌面那个 report.pdf'). Supports '~'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the PDF file. Supports '~'.",
                },
            },
            "required": ["path"],
        },
        handler=handler,
    )


# ── anything-to-md (tomd CLI) ─────────────────────────────────────

import shutil
import subprocess

# Resolve via PATH first; fall back to ~/.local/bin/tomd which is
# where `pip install --user tomd` lands and where launchd-style
# wrappers usually can't see (PATH doesn't include it by default).
_TOMD_BIN = shutil.which("tomd") or str(Path.home() / ".local/bin/tomd")

_TOMD_MAX_CHARS = 60_000
_TOMD_TIMEOUT = 180  # seconds — office docs + PDFs can take a while

_TOMD_FORCE_TYPES = {
    "webpage", "wechat", "youtube", "bilibili",
    "douyin", "xiaohongshu", "file",
}

# Homebrew's python@3.13 pyexpat is linked against /usr/lib/libexpat.1.dylib
# (system), which on Darwin 25 is too old and missing
# `_XML_SetAllocTrackerActivationThreshold`. That symbol is in homebrew
# expat 2.7+. Force the loader to prefer homebrew's copy so file paths
# (which trigger XML parsing inside tomd) don't blow up with dlopen errors.
_TOMD_DYLD_PATH = "/opt/homebrew/opt/expat/lib"


def make_anything_to_md_tool() -> Tool:
    async def handler(input_data: dict) -> str:
        raw = (input_data.get("input") or "").strip()
        if not raw:
            return "ERROR: 'input' is required (file path or URL)."
        is_url = raw.startswith("http://") or raw.startswith("https://")
        if is_url:
            target = raw
        else:
            p = Path(raw).expanduser()
            if not p.exists():
                return f"ERROR: 文件不存在：{p}"
            target = str(p)

        force_type = (input_data.get("force_type") or "").strip() or None
        if force_type and force_type not in _TOMD_FORCE_TYPES:
            return (
                f"ERROR: force_type 取值非法，可用："
                f"{sorted(_TOMD_FORCE_TYPES)}"
            )

        def _run() -> str:
            cmd = [_TOMD_BIN, target, "--stdout"]
            if force_type:
                cmd += ["--type", force_type]
            import os as _os
            env = {**_os.environ}
            existing = env.get("DYLD_LIBRARY_PATH", "")
            env["DYLD_LIBRARY_PATH"] = (
                f"{_TOMD_DYLD_PATH}:{existing}" if existing else _TOMD_DYLD_PATH
            )
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=_TOMD_TIMEOUT,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                return f"ERROR: tomd 超时（{_TOMD_TIMEOUT} 秒上限）"
            except FileNotFoundError:
                return f"ERROR: 找不到 tomd 可执行文件：{_TOMD_BIN}"
            except Exception as e:
                return f"ERROR: {type(e).__name__}: {e}"
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()[:400]
                return (
                    f"ERROR: tomd 退出码 {result.returncode}：{stderr}"
                )
            out = result.stdout or ""
            if not out.strip():
                return "(tomd 没有返回任何内容)"
            original_len = len(out)
            if original_len > _TOMD_MAX_CHARS:
                out = out[:_TOMD_MAX_CHARS] + (
                    f"\n\n... (已截断，原文 {original_len} 字符)"
                )
            return out

        logger.info(
            "tool.anything_to_md target=%s type=%s",
            target[:200], force_type,
        )
        return await asyncio.to_thread(_run)

    return Tool(
        name="anything_to_md",
        description=(
            "Convert almost any document or URL to clean Markdown via "
            "the locally installed `tomd` (anything-to-md) CLI. "
            "Supports DOCX / PPTX / XLSX / PDF / EPUB / CSV / JSON as "
            "well as webpages, WeChat articles, and YouTube / Bilibili "
            "/ Douyin / Xiaohongshu links. Use when the user sends a "
            "docx/xlsx/pptx attachment (path appears in the message "
            "as `data/attachments/…`) or shares a URL and wants the "
            "full cleaned-up content. Output is capped at "
            f"{_TOMD_MAX_CHARS // 1000}k chars."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": (
                        "Local file path (supports '~') or http(s) URL."
                    ),
                },
                "force_type": {
                    "type": "string",
                    "enum": sorted(_TOMD_FORCE_TYPES),
                    "description": (
                        "Optional. Force input type; usually omitted "
                        "— tomd auto-detects."
                    ),
                },
            },
            "required": ["input"],
        },
        handler=handler,
    )


# ── browser capture (playwright headless chromium) ───────────────

_BROWSER_MARKDOWN_CAP = 12_000              # chars returned as markdown
_BROWSER_GOTO_TIMEOUT = 30_000              # ms
_BROWSER_IDLE_TIMEOUT = 8_000               # ms — wait for network idle


def make_browser_capture_tool(settings: Any) -> Tool:
    """Headless chromium navigate → screenshot + page text.

    On-demand: a fresh browser is launched per call and closed after.
    No persistent login state (would need CDP attach to user's real
    Chrome — deliberately not done here to keep the bot from touching
    the user's live browsing session).
    """
    capture_dir = settings.data_dir / "browser_captures"
    capture_dir.mkdir(parents=True, exist_ok=True)

    async def _run(
        url: str,
        wait_selector: str | None,
        wait_ms: int,
        full_page: bool,
    ) -> str:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return (
                "ERROR: playwright 未安装。先跑："
                "venv/bin/pip install playwright && "
                "venv/bin/python -m playwright install chromium"
            )

        ts = time.strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        shot_path = capture_dir / f"{ts}_{uid}.png"
        markdown = ""
        final_url = url
        status: int | None = None
        html = ""

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                try:
                    context = await browser.new_context(
                        user_agent=_BROWSER_UA,
                        viewport={"width": 1280, "height": 900},
                        locale="zh-CN",
                    )
                    page = await context.new_page()
                    resp = await page.goto(
                        url,
                        wait_until="domcontentloaded",
                        timeout=_BROWSER_GOTO_TIMEOUT,
                    )
                    status = resp.status if resp else None
                    try:
                        await page.wait_for_load_state(
                            "networkidle", timeout=_BROWSER_IDLE_TIMEOUT
                        )
                    except Exception:
                        pass
                    if wait_selector:
                        try:
                            await page.wait_for_selector(
                                wait_selector, timeout=10_000
                            )
                        except Exception:
                            pass
                    if wait_ms > 0:
                        await page.wait_for_timeout(min(wait_ms, 30_000))

                    final_url = page.url
                    html = await page.content()
                    await page.screenshot(
                        path=str(shot_path),
                        full_page=full_page,
                    )
                finally:
                    await browser.close()
        except Exception as e:
            logger.exception("tool.browser_capture failed url=%s", url)
            return f"ERROR: 浏览器抓取失败 {type(e).__name__}: {e}"

        try:
            h = html2text.HTML2Text()
            h.ignore_images = True
            h.body_width = 0
            markdown = h.handle(html).strip()
        except Exception:
            markdown = ""
        original_len = len(markdown)
        if original_len > _BROWSER_MARKDOWN_CAP:
            markdown = (
                markdown[:_BROWSER_MARKDOWN_CAP]
                + f"\n\n... (已截断，原文 {original_len} 字符)"
            )

        try:
            shot_bytes = shot_path.stat().st_size
        except OSError:
            shot_bytes = 0

        head = (
            f"HTTP {status or '?'} final_url={final_url}\n"
            f"screenshot: {shot_path} ({shot_bytes} bytes)\n"
            f"— 需要把截图发给用户就调 send_to_feishu，"
            f"path 填上面那个 screenshot 路径。\n\n"
        )
        return head + (markdown or "(页面没有可提取的正文)")

    async def handler(input_data: dict) -> str:
        url = (input_data.get("url") or "").strip()
        if not url:
            return "ERROR: 'url' is required."
        if not (url.startswith("http://") or url.startswith("https://")):
            return f"ERROR: url 必须是 http(s):// (got {url!r})"
        wait_selector = input_data.get("wait_for_selector")
        if wait_selector is not None and not isinstance(wait_selector, str):
            return "ERROR: 'wait_for_selector' must be a string."
        try:
            wait_ms = int(input_data.get("wait_ms", 0))
        except (TypeError, ValueError):
            wait_ms = 0
        full_page = bool(input_data.get("full_page", False))
        logger.info(
            "tool.browser_capture url=%s selector=%s wait_ms=%d full_page=%s",
            url, wait_selector, wait_ms, full_page,
        )
        return await _run(url, wait_selector, wait_ms, full_page)

    return Tool(
        name="browser_capture",
        description=(
            "Open a URL in a headless chromium browser, wait for it to "
            "render, then return (a) a full-page screenshot saved to "
            "`data/browser_captures/<ts>_<id>.png` and (b) the rendered "
            "HTML converted to Markdown. Use this when `web_fetch` "
            "doesn't work — JS-heavy sites (SPAs, Twitter/X, LinkedIn, "
            "some news sites), pages that block curl UAs, or when you "
            "need a visual to show the user. No persistent login — "
            "each call is a fresh session.\n\n"
            "If the user wants to see the screenshot, call "
            "`send_to_feishu` afterwards with the returned screenshot "
            "path."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Absolute http(s) URL to open.",
                },
                "wait_for_selector": {
                    "type": "string",
                    "description": (
                        "Optional CSS selector to wait for before "
                        "capturing (e.g. 'article', '.main-content'). "
                        "Max 10s."
                    ),
                },
                "wait_ms": {
                    "type": "integer",
                    "description": (
                        "Additional fixed wait in ms after the page "
                        "settles (e.g. 2000 for lazy-loaded content). "
                        "0–30000."
                    ),
                    "minimum": 0,
                    "maximum": 30000,
                },
                "full_page": {
                    "type": "boolean",
                    "description": (
                        "Capture full scrollable page height. Default "
                        "false (viewport only — faster, smaller)."
                    ),
                },
            },
            "required": ["url"],
        },
        handler=handler,
    )


# ── command execution ─────────────────────────────────────────────

_CMD_TIMEOUT = 120  # seconds
_CMD_MAX_OUTPUT = 20_000  # chars returned to model


def make_run_command_tool() -> Tool:
    async def handler(input_data: dict) -> str:
        command = (input_data.get("command") or "").strip()
        if not command:
            return "ERROR: 'command' is required."
        cwd = input_data.get("working_directory")
        if cwd:
            cwd = str(Path(cwd).expanduser())

        logger.info("tool.run_command cmd=%s cwd=%s", command[:200], cwd)

        def _run() -> str:
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=_CMD_TIMEOUT,
                    cwd=cwd,
                    env={
                        **__import__("os").environ,
                        "PYTHONPATH": str(Path(__file__).resolve().parent.parent.parent),
                    },
                )
            except subprocess.TimeoutExpired:
                return f"ERROR: 命令超时（{_CMD_TIMEOUT} 秒限制）"
            except Exception as e:
                return f"ERROR: {type(e).__name__}: {e}"

            parts: list[str] = []
            if result.stdout:
                out = result.stdout
                if len(out) > _CMD_MAX_OUTPUT:
                    out = out[:_CMD_MAX_OUTPUT] + f"\n... (已截断，共 {len(result.stdout)} 字符)"
                parts.append(out)
            if result.stderr:
                err = result.stderr
                if len(err) > _CMD_MAX_OUTPUT:
                    err = err[:_CMD_MAX_OUTPUT] + "\n... (stderr 已截断)"
                parts.append(f"STDERR:\n{err}")
            if result.returncode != 0:
                parts.append(f"EXIT CODE: {result.returncode}")
            return "\n".join(parts) if parts else "(no output)"

        return await asyncio.to_thread(_run)

    return Tool(
        name="run_command",
        description=(
            "Execute a shell command on the user's computer and return "
            "the output. Use this to run scripts, install packages, "
            "or perform any terminal operation. The command runs in "
            "a shell (bash/zsh) with a 120-second timeout. Use "
            "working_directory to set the cwd. Examples: "
            "'python script.py', 'pip install package', 'ls -la'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
                "working_directory": {
                    "type": "string",
                    "description": (
                        "Optional. Working directory for the command. "
                        "Supports '~'. Defaults to the agent's project root."
                    ),
                },
            },
            "required": ["command"],
        },
        handler=handler,
    )


# ── local filesystem ops ──────────────────────────────────────────

# Max characters returned when reading a local file via tool.
_READ_FILE_MAX_CHARS = 60_000


def make_filesystem_tools() -> list[Tool]:
    """Return a bundle of tools for basic filesystem operations:
    create_directory, list_directory, move_path, copy_path,
    delete_path, read_local_file, write_local_file.
    """

    # ── create_directory ──────────────────────────────────────

    async def _create_dir(input_data: dict) -> str:
        raw = (input_data.get("path") or "").strip()
        if not raw:
            return "ERROR: 'path' is required."
        p = Path(raw).expanduser()
        if not p.is_absolute():
            return f"ERROR: 请提供绝对路径（收到的是 {raw!r}）"
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return f"ERROR: {e}"
        return f"OK: 目录已创建 {p}"

    create_dir_tool = Tool(
        name="create_directory",
        description=(
            "Create a directory (folder) on the user's computer. "
            "Creates parent directories automatically if they don't "
            "exist. Use this when the user asks you to create a folder "
            "(e.g. '在桌面建个文件夹叫 projects'). Path supports '~'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Absolute path for the new directory. "
                        "Use ~ for home dir (e.g. '~/Desktop/新文件夹')."
                    ),
                },
            },
            "required": ["path"],
        },
        handler=_create_dir,
    )

    # ── list_directory ────────────────────────────────────────

    async def _list_dir(input_data: dict) -> str:
        raw = (input_data.get("path") or "").strip()
        if not raw:
            return "ERROR: 'path' is required."
        p = Path(raw).expanduser()
        if not p.is_dir():
            return f"ERROR: 不是目录或不存在：{p}"
        try:
            entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except OSError as e:
            return f"ERROR: {e}"
        if not entries:
            return f"{p} 是空目录。"
        lines: list[str] = []
        for e in entries[:200]:
            if e.is_dir():
                lines.append(f"📁 {e.name}/")
            else:
                try:
                    size = e.stat().st_size
                except OSError:
                    size = 0
                if size < 1024:
                    sz = f"{size} B"
                elif size < 1024 * 1024:
                    sz = f"{size / 1024:.1f} KB"
                else:
                    sz = f"{size / (1024 * 1024):.1f} MB"
                lines.append(f"  {e.name}  ({sz})")
        result = "\n".join(lines)
        if len(entries) > 200:
            result += f"\n\n... 共 {len(entries)} 个条目，仅显示前 200 个"
        return result

    list_dir_tool = Tool(
        name="list_directory",
        description=(
            "List the contents of a directory on the user's computer. "
            "Shows folders first (with 📁), then files with sizes. "
            "Use when the user asks 'look at my Desktop' or 'what "
            "files are in ~/Documents'. Path supports '~'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the directory. Supports '~'.",
                },
            },
            "required": ["path"],
        },
        handler=_list_dir,
    )

    # ── move_path ─────────────────────────────────────────────

    async def _move_path(input_data: dict) -> str:
        src = (input_data.get("source") or "").strip()
        dst = (input_data.get("destination") or "").strip()
        if not src or not dst:
            return "ERROR: 'source' and 'destination' are both required."
        sp = Path(src).expanduser()
        dp = Path(dst).expanduser()
        if not sp.exists():
            return f"ERROR: 源路径不存在：{sp}"
        # If destination is an existing directory, move INTO it
        if dp.is_dir():
            dp = dp / sp.name
        try:
            dp.parent.mkdir(parents=True, exist_ok=True)
            sp.rename(dp)
        except OSError as e:
            # rename fails across filesystems; fall back to shutil
            import shutil
            try:
                shutil.move(str(sp), str(dp))
            except Exception as e2:
                return f"ERROR: {e2}"
        return f"OK: {sp.name} → {dp}"

    move_tool = Tool(
        name="move_path",
        description=(
            "Move or rename a file/folder on the user's computer. "
            "If destination is an existing directory, the source is "
            "moved inside it. Also works as rename. Supports '~'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Path to move from."},
                "destination": {"type": "string", "description": "Path to move to."},
            },
            "required": ["source", "destination"],
        },
        handler=_move_path,
    )

    # ── copy_path ─────────────────────────────────────────────

    async def _copy_path(input_data: dict) -> str:
        src = (input_data.get("source") or "").strip()
        dst = (input_data.get("destination") or "").strip()
        if not src or not dst:
            return "ERROR: 'source' and 'destination' are both required."
        sp = Path(src).expanduser()
        dp = Path(dst).expanduser()
        if not sp.exists():
            return f"ERROR: 源路径不存在：{sp}"
        if dp.is_dir():
            dp = dp / sp.name
        import shutil
        try:
            dp.parent.mkdir(parents=True, exist_ok=True)
            if sp.is_dir():
                shutil.copytree(str(sp), str(dp))
            else:
                shutil.copy2(str(sp), str(dp))
        except Exception as e:
            return f"ERROR: {e}"
        return f"OK: 已复制到 {dp}"

    copy_tool = Tool(
        name="copy_path",
        description=(
            "Copy a file or folder on the user's computer. If "
            "destination is an existing directory, copies into it. "
            "Supports '~'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Path to copy from."},
                "destination": {"type": "string", "description": "Path to copy to."},
            },
            "required": ["source", "destination"],
        },
        handler=_copy_path,
    )

    # ── delete_path ───────────────────────────────────────────

    async def _delete_path(input_data: dict) -> str:
        raw = (input_data.get("path") or "").strip()
        if not raw:
            return "ERROR: 'path' is required."
        p = Path(raw).expanduser()
        if not p.exists():
            return f"ERROR: 路径不存在：{p}"
        # Safety: refuse to delete home, root, or top-level system dirs
        resolved = p.resolve()
        dangerous = {Path.home().resolve(), Path("/").resolve()}
        if resolved in dangerous or len(resolved.parts) <= 2:
            return f"ERROR: 拒绝删除系统关键路径：{resolved}"
        import shutil
        try:
            if p.is_dir():
                shutil.rmtree(str(p))
            else:
                p.unlink()
        except Exception as e:
            return f"ERROR: {e}"
        kind = "目录" if p.is_dir() else "文件"
        return f"OK: 已删除{kind} {p}"

    delete_tool = Tool(
        name="delete_path",
        description=(
            "Delete a file or folder from the user's computer. "
            "Refuses to delete home directory or root-level system "
            "paths for safety. Supports '~'. Use with caution."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory to delete.",
                },
            },
            "required": ["path"],
        },
        handler=_delete_path,
    )

    # ── read_local_file ───────────────────────────────────────

    async def _read_local_file(input_data: dict) -> str:
        raw = (input_data.get("path") or "").strip()
        if not raw:
            return "ERROR: 'path' is required."
        p = Path(raw).expanduser()
        if not p.is_file():
            return f"ERROR: 文件不存在：{p}"
        try:
            data = p.read_bytes()
        except OSError as e:
            return f"ERROR: {e}"
        if b"\x00" in data[:8192]:
            return f"ERROR: {p.name} 是二进制文件，无法以文本读取。"
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = data.decode("gbk")
            except UnicodeDecodeError:
                return f"ERROR: {p.name} 编码无法识别（不是 UTF-8 也不是 GBK）。"
        if len(text) > _READ_FILE_MAX_CHARS:
            text = text[:_READ_FILE_MAX_CHARS] + f"\n\n... (已截断，原文 {len(text)} 字符)"
        return text

    read_file_tool = Tool(
        name="read_local_file",
        description=(
            "Read the text content of a file on the user's computer. "
            "Supports UTF-8 and GBK encodings. Rejects binary files. "
            "Use when the user says 'read ~/Desktop/notes.txt' or "
            "'look at this config file'. Supports '~'. Max ~60k chars."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the text file. Supports '~'.",
                },
            },
            "required": ["path"],
        },
        handler=_read_local_file,
    )

    # ── write_local_file ──────────────────────────────────────

    async def _write_local_file(input_data: dict) -> str:
        raw = (input_data.get("path") or "").strip()
        content = input_data.get("content")
        if not raw:
            return "ERROR: 'path' is required."
        if not isinstance(content, str):
            return "ERROR: 'content' must be a string."
        p = Path(raw).expanduser()
        if not p.is_absolute():
            return f"ERROR: 请提供绝对路径（收到的是 {raw!r}）"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except OSError as e:
            return f"ERROR: {e}"
        return f"OK: 已写入 {p} ({len(content)} 字符)"

    write_file_tool = Tool(
        name="write_local_file",
        description=(
            "Write text content to a file on the user's computer. "
            "Creates parent directories if needed. Overwrites existing "
            "file. Use when the user asks you to create or save a "
            "file (e.g. '帮我写个脚本保存到桌面'). Supports '~'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path for the file. Supports '~'.",
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write into the file.",
                },
            },
            "required": ["path", "content"],
        },
        handler=_write_local_file,
    )

    return [
        create_dir_tool,
        list_dir_tool,
        move_tool,
        copy_tool,
        delete_tool,
        read_file_tool,
        write_file_tool,
    ]


# ── memory search ─────────────────────────────────────────────────

def make_search_memory_tool(memory: MemoryStore) -> Tool:
    """Tool for the agent to search daily digests and memory backups."""
    from .session import SessionStore

    async def handler(input_data: dict) -> str:
        action = (input_data.get("action") or "").strip()

        if action == "search_digests":
            keyword = (input_data.get("keyword") or "").strip()
            if not keyword:
                return "ERROR: 'keyword' is required for search_digests."
            # We need the SessionStore — get it from the memory path's
            # sibling db. This is a bit roundabout but avoids plumbing
            # changes.
            db_path = memory.path.parent.parent / "xiaonz.db"
            if not db_path.is_file():
                return "没有找到数据库文件。"
            store = SessionStore(db_path)
            results = store.search_daily_digests(keyword)
            if not results:
                return f"没有找到包含「{keyword}」的每日摘要。"
            lines = []
            for r in results:
                lines.append(f"## {r['date']}（{r['msg_count']} 条消息）\n{r['summary']}")
            return "\n\n".join(lines)

        if action == "list_digests":
            db_path = memory.path.parent.parent / "xiaonz.db"
            if not db_path.is_file():
                return "没有找到数据库文件。"
            store = SessionStore(db_path)
            limit = int(input_data.get("limit", 30))
            digests = store.list_daily_digests(limit=limit)
            if not digests:
                return "还没有生成过每日摘要。"
            lines = []
            for d in digests:
                lines.append(f"- {d['date']}: {d['msg_count']} 条消息, {d['summary_chars']} 字摘要")
            return "\n".join(lines)

        if action == "read_digest":
            date_str = (input_data.get("date") or "").strip()
            if not date_str:
                return "ERROR: 'date' is required (格式 YYYY-MM-DD)."
            db_path = memory.path.parent.parent / "xiaonz.db"
            if not db_path.is_file():
                return "没有找到数据库文件。"
            store = SessionStore(db_path)
            text = store.get_daily_digest(date_str)
            return text if text else f"没有 {date_str} 的每日摘要。"

        if action == "list_backups":
            backups = memory.list_backups()
            if not backups:
                return "没有记忆备份文件。"
            return "\n".join(f"- {b}" for b in backups)

        if action == "read_backup":
            filename = (input_data.get("filename") or "").strip()
            if not filename:
                return "ERROR: 'filename' is required."
            text = memory.read_backup(filename)
            return text if text is not None else f"备份文件 {filename} 不存在。"

        if action == "read_memory":
            return memory.read() or "(MEMORY.md 为空)"

        return (
            "ERROR: 未知 action。可用值：search_digests, list_digests, "
            "read_digest, list_backups, read_backup, read_memory"
        )

    return Tool(
        name="search_memory",
        description=(
            "Search and retrieve the agent's long-term memory: daily "
            "conversation digests, MEMORY.md content, and memory backups. "
            "Use when you need to recall past conversations or facts. "
            "Actions: search_digests (keyword search), list_digests, "
            "read_digest (by date), list_backups, read_backup, read_memory."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "search_digests",
                        "list_digests",
                        "read_digest",
                        "list_backups",
                        "read_backup",
                        "read_memory",
                    ],
                    "description": "Which memory operation to perform.",
                },
                "keyword": {
                    "type": "string",
                    "description": "Search keyword (for search_digests).",
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format (for read_digest).",
                },
                "filename": {
                    "type": "string",
                    "description": "Backup filename (for read_backup).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (for list_digests, default 30).",
                },
            },
            "required": ["action"],
        },
        handler=handler,
    )


# ── semantic memory search ────────────────────────────────────────

_SEMANTIC_KINDS = {"message", "digest", "md", "note"}


def make_search_memory_semantic_tool(vector_memory: Any) -> Tool:
    """Explicit semantic recall tool. The system prompt already injects
    top-k hits automatically on every turn, so this is mostly for the
    model to use when the user is asking a direct memory question
    ('我们上次聊 X 是什么时候？') and needs more/filtered results."""

    async def handler(input_data: dict) -> str:
        query = (input_data.get("query") or "").strip()
        if not query:
            return "ERROR: 'query' is required."
        try:
            top_k = int(input_data.get("top_k", 8))
        except (TypeError, ValueError):
            top_k = 8
        top_k = max(1, min(20, top_k))
        try:
            min_score = float(input_data.get("min_score", 0.35))
        except (TypeError, ValueError):
            min_score = 0.35
        raw_kinds = input_data.get("kinds")
        kinds: list[str] | None
        if isinstance(raw_kinds, list) and raw_kinds:
            kinds = [k for k in raw_kinds if k in _SEMANTIC_KINDS]
            if not kinds:
                return (
                    f"ERROR: 'kinds' 取值非法，可用："
                    f"{sorted(_SEMANTIC_KINDS)}"
                )
        else:
            kinds = None
        try:
            hits = await vector_memory.search(
                query=query,
                top_k=top_k,
                kinds=kinds,
                min_score=min_score,
            )
        except Exception as e:
            return f"ERROR: 语义搜索失败 {type(e).__name__}: {e}"
        if not hits:
            return f"（没有和「{query}」相关的记忆，阈值 {min_score}）"
        lines = [f"# 语义搜索：{query}（{len(hits)} 条）", ""]
        for i, h in enumerate(hits, 1):
            text = (h.get("text") or "").strip()
            if len(text) > 800:
                text = text[:800] + "…"
            ref = h.get("source_ref") or "(no ref)"
            lines.append(
                f"## {i}. [{h['kind']}] score={h['score']:.2f} {ref}"
            )
            lines.append(text)
            lines.append("")
        return "\n".join(lines).rstrip()

    return Tool(
        name="search_memory_semantic",
        description=(
            "Semantic (vector) search across the agent's memory: past "
            "messages, daily digests, long-term markdown memory. Use "
            "this when the user asks about something that may have come "
            "up before but you don't see it in the current context — "
            "e.g. '我上次跟你说的那个项目叫什么？', '我们聊过"
            "关于 X 的事吗'. The system prompt already injects top "
            "auto-recall hits each turn, so only call this when you "
            "need more or want to filter by kind.\n\n"
            "kinds: 'message' = individual chat lines; 'digest' = daily "
            "conversation summaries; 'md' = chunks of MEMORY.md/SOUL.md "
            "and archived memory files; 'note' = explicit notes."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language question or keywords.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max hits to return, 1–20 (default 8).",
                    "minimum": 1,
                    "maximum": 20,
                },
                "kinds": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": sorted(_SEMANTIC_KINDS),
                    },
                    "description": (
                        "Optional filter. Omit to search all kinds."
                    ),
                },
                "min_score": {
                    "type": "number",
                    "description": (
                        "Minimum cosine similarity, default 0.35. "
                        "Lower to widen recall; raise to be strict."
                    ),
                },
            },
            "required": ["query"],
        },
        handler=handler,
    )


# ── convenience bundle ────────────────────────────────────────────

def default_tools(skills: SkillStore, memory: MemoryStore) -> list[Tool]:
    return [
        make_list_skills_tool(skills),
        make_load_skill_tool(skills),
        make_install_skill_tool(skills),
        make_uninstall_skill_tool(skills),
        make_update_memory_tool(memory),
        make_search_memory_tool(memory),
        make_web_search_tool(),
        make_web_fetch_tool(),
        make_download_to_disk_tool(),
        make_read_pdf_tool(),
        make_anything_to_md_tool(),
        make_run_command_tool(),
        *make_filesystem_tools(),
    ]
