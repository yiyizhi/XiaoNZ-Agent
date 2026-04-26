"""One-shot: embed every markdown file under data/memory/ + every
existing daily_digest + every existing message into the vector store.

Rationale: the live loop only embeds content going forward. This
script back-fills so semantic recall has something to find on day one.

Safe to re-run: skips source_refs that already exist in memory_vec.
Pass --reindex to drop and re-insert everything for a given kind.

Usage:
    venv/bin/python scripts/bootstrap_memory.py
    venv/bin/python scripts/bootstrap_memory.py --reindex md
    venv/bin/python scripts/bootstrap_memory.py --only digests

Chunks markdown by top-level and second-level headings, then falls
back to ~1500-char windows for long sections. Batches embeddings in
groups of 16 with a tiny sleep between to be gentle on small
single-threaded embedding services (e.g. a local Ollama instance).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterator

# Make `src` importable when run as `python scripts/bootstrap_memory.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.agent.embedder import Embedder  # noqa: E402
from src.agent.vector_memory import VectorMemory  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("bootstrap_memory")


# ── markdown chunking ─────────────────────────────────────────────

_CHUNK_TARGET = 1500   # chars per chunk (approx)
_CHUNK_HARD_MAX = 2500  # hard ceiling: split aggressively beyond this


def _split_by_heading(text: str) -> list[tuple[str, str]]:
    """Split a markdown blob into (heading_path, body) chunks.

    Heading path = "## A > ### B" style breadcrumb so each chunk is
    self-describing when retrieved in isolation.
    """
    lines = text.splitlines()
    chunks: list[tuple[str, str]] = []
    cur_path: list[str] = []  # stack of (level, title)
    cur_body: list[str] = []

    def _flush():
        body = "\n".join(cur_body).strip()
        if body:
            path = " > ".join(h for _, h in cur_path) if cur_path else ""
            chunks.append((path, body))

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m:
            _flush()
            cur_body = []
            level = len(m.group(1))
            title = f"{'#' * level} {m.group(2)}"
            # Pop headings at same or deeper level
            while cur_path and cur_path[-1][0] >= level:
                cur_path.pop()
            cur_path.append((level, title))
        else:
            cur_body.append(line)
    _flush()
    return chunks or [("", text.strip())]


def _window(text: str, size: int, overlap: int = 150) -> list[str]:
    out = []
    i = 0
    n = len(text)
    while i < n:
        out.append(text[i : i + size])
        if i + size >= n:
            break
        i += size - overlap
    return out


def chunk_markdown(path: Path, text: str) -> list[dict]:
    """Return a list of {text, source_ref, meta} ready to hand to
    VectorMemory.add_many. `source_ref` encodes file path + chunk idx
    so a later re-index can find and replace them."""
    rel = path.relative_to(ROOT).as_posix()
    results: list[dict] = []
    section_chunks = _split_by_heading(text)

    idx = 0
    for heading_path, body in section_chunks:
        combined = (heading_path + "\n\n" + body).strip() if heading_path else body
        if len(combined) <= _CHUNK_HARD_MAX:
            results.append({
                "text": combined,
                "source_ref": f"md:{rel}#{idx}",
                "meta": {"path": rel, "heading": heading_path},
            })
            idx += 1
            continue
        # Long section: window it; prepend heading to each piece so the
        # chunk remains self-describing out of context.
        for piece in _window(combined, _CHUNK_TARGET, overlap=150):
            results.append({
                "text": piece,
                "source_ref": f"md:{rel}#{idx}",
                "meta": {"path": rel, "heading": heading_path, "windowed": True},
            })
            idx += 1
    return results


# ── back-fill helpers ─────────────────────────────────────────────

async def _batched_add_many(
    vec: VectorMemory, kind: str, items: list[dict], batch_size: int = 16,
) -> int:
    total = 0
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        added = await vec.add_many(kind, batch)
        total += added
        log.info(
            "%s.batch %d/%d added=%d running=%d",
            kind,
            (i // batch_size) + 1,
            (len(items) + batch_size - 1) // batch_size,
            added,
            total,
        )
        # Be kind to single-process embedder
        await asyncio.sleep(0.3)
    return total


async def reindex_markdown(vec: VectorMemory, reindex: bool, settings) -> None:
    memory_dir = settings.memory_dir
    if not memory_dir.is_dir():
        log.warning("memory_dir not found: %s", memory_dir)
        return
    md_files = sorted(memory_dir.rglob("*.md"))
    log.info("md.scan found=%d dir=%s", len(md_files), memory_dir)

    all_items: list[dict] = []
    for p in md_files:
        try:
            text = p.read_text(encoding="utf-8").strip()
        except Exception as e:
            log.warning("md.read_failed %s: %s", p, e)
            continue
        if not text:
            continue
        rel = p.relative_to(ROOT).as_posix()
        source_prefix = f"md:{rel}#"
        if reindex:
            deleted = 0
            with sqlite3.connect(vec.db_path) as c:
                cur = c.execute(
                    "DELETE FROM memory_vec WHERE source_ref LIKE ?",
                    (source_prefix + "%",),
                )
                deleted = cur.rowcount
            if deleted:
                log.info("md.reindex path=%s deleted=%d", rel, deleted)
        else:
            # Skip if any chunk already exists for this file
            if vec.has_source(source_prefix + "0"):
                log.info("md.skip path=%s already_indexed", rel)
                continue
        chunks = chunk_markdown(p, text)
        log.info("md.chunk path=%s chunks=%d", rel, len(chunks))
        all_items.extend(chunks)

    if not all_items:
        log.info("md.nothing_to_do")
        return
    log.info("md.embed total_chunks=%d", len(all_items))
    await _batched_add_many(vec, "md", all_items)


async def reindex_digests(vec: VectorMemory, reindex: bool, settings) -> None:
    """Pull every row from daily_digests and embed it."""
    with sqlite3.connect(settings.db_path) as c:
        rows = c.execute(
            "SELECT date, summary, msg_count FROM daily_digests ORDER BY date ASC"
        ).fetchall()
    log.info("digest.scan found=%d", len(rows))
    items: list[dict] = []
    for d, summary, msg_count in rows:
        if not summary or not summary.strip():
            continue
        ref = f"digest:{d}"
        if reindex:
            with sqlite3.connect(vec.db_path) as c:
                c.execute("DELETE FROM memory_vec WHERE source_ref=?", (ref,))
        elif vec.has_source(ref):
            continue
        items.append({
            "text": summary,
            "source_ref": ref,
            "meta": {"date": d, "msg_count": msg_count},
        })
    if not items:
        log.info("digest.nothing_to_do")
        return
    log.info("digest.embed total=%d", len(items))
    await _batched_add_many(vec, "digest", items)


async def reindex_messages(
    vec: VectorMemory, reindex: bool, settings, min_chars: int = 10,
) -> None:
    """Embed raw messages so historical chat lines are recallable.

    We skip anything shorter than `min_chars` (e.g. 'ok', '好的') to
    avoid flooding the store with noise. Existing rows produced by the
    live loop (source_ref like 'msg:<sid>:<ts>:<role>') will be re-
    used via has_source when reindex=False.
    """
    with sqlite3.connect(settings.db_path) as c:
        rows = c.execute(
            """
            SELECT id, session_id, role, content, created_at
            FROM messages
            ORDER BY id ASC
            """,
        ).fetchall()
    log.info("msg.scan found=%d", len(rows))
    items: list[dict] = []
    for row_id, sid, role, content, created_at in rows:
        if not content or len(content.strip()) < min_chars:
            continue
        ref = f"msg_backfill:{row_id}"
        if reindex:
            with sqlite3.connect(vec.db_path) as c:
                c.execute("DELETE FROM memory_vec WHERE source_ref=?", (ref,))
        elif vec.has_source(ref):
            continue
        items.append({
            "text": content,
            "source_ref": ref,
            "meta": {
                "role": role,
                "session_id": sid,
                "created_at": created_at,
                "msg_id": row_id,
            },
        })
    if not items:
        log.info("msg.nothing_to_do")
        return
    log.info("msg.embed total=%d", len(items))
    await _batched_add_many(vec, "message", items)


# ── main ──────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["md", "digests", "messages"],
        help="Only run the named section.",
    )
    parser.add_argument(
        "--reindex",
        choices=["md", "digests", "messages", "all"],
        help="Delete existing rows for that kind before inserting.",
    )
    parser.add_argument(
        "--min-msg-chars",
        type=int,
        default=10,
        help="Skip messages shorter than this when back-filling (default 10).",
    )
    args = parser.parse_args()

    settings = config.load()
    embedder = Embedder(
        base_url=settings.embedding.base_url,
        model=settings.embedding.model,
        dim=settings.embedding.dim,
        timeout=60.0,
    )
    ok = await embedder.ping()
    if not ok:
        log.error(
            "embedder.unreachable url=%s/embeddings — start your "
            "embedding service first (e.g. `ollama serve`)",
            settings.embedding.base_url.rstrip("/"),
        )
        return 2
    log.info(
        "embedder.ok base_url=%s model=%s dim=%d",
        settings.embedding.base_url,
        settings.embedding.model,
        settings.embedding.dim,
    )

    vec = VectorMemory(settings.db_path, embedder)

    want_md = args.only in (None, "md")
    want_digests = args.only in (None, "digests")
    want_messages = args.only in (None, "messages")

    t0 = time.time()
    if want_md:
        await reindex_markdown(
            vec, reindex=(args.reindex in ("md", "all")), settings=settings,
        )
    if want_digests:
        await reindex_digests(
            vec, reindex=(args.reindex in ("digests", "all")), settings=settings,
        )
    if want_messages:
        await reindex_messages(
            vec,
            reindex=(args.reindex in ("messages", "all")),
            settings=settings,
            min_chars=args.min_msg_chars,
        )
    log.info("bootstrap.done elapsed=%.1fs stats=%s", time.time() - t0, vec.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
