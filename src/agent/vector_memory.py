"""SQLite-backed vector memory.

One table in the same hermes_state.db as SessionStore. bge-m3 1024-d
float32 vectors stored as raw bytes. Search loads all candidates of
the requested kinds into memory and does a single numpy matmul —
fine up to ~50k rows, which we're nowhere near. Swap to sqlite-vec
later if the corpus grows large.

Kinds used across the codebase:
    - 'message' : individual chat message (user or assistant)
    - 'digest'  : daily digest summary
    - 'md'      : chunked markdown memory file (SOUL / MEMORY / archive)
    - 'note'    : explicit note written via a tool (future)

Writes are best-effort: if the embedder is unreachable, add() returns
None and callers move on. Nothing here should ever be load-bearing for
the main message flow.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from .embedder import Embedder

logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_vec (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    source_ref  TEXT,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    created_at  REAL NOT NULL,
    meta        TEXT
);
CREATE INDEX IF NOT EXISTS idx_memvec_kind ON memory_vec(kind);
CREATE INDEX IF NOT EXISTS idx_memvec_source ON memory_vec(source_ref);
"""


class VectorMemory:
    def __init__(self, db_path: Path, embedder: Embedder):
        self.db_path = db_path
        self.embedder = embedder
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        try:
            yield conn
        finally:
            conn.close()

    # ── writes ─────────────────────────────────────────────────────

    async def add(
        self,
        kind: str,
        text: str,
        source_ref: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> Optional[int]:
        if not text or not text.strip():
            return None
        vec = await self.embedder.embed_one(text)
        if vec is None:
            logger.warning(
                "vec.add_skipped kind=%s ref=%s reason=embedder_unreachable",
                kind, source_ref,
            )
            return None
        return await asyncio.to_thread(
            self._insert_sync, kind, text, vec, source_ref, meta,
        )

    async def add_many(
        self,
        kind: str,
        items: List[dict],
    ) -> int:
        """Bulk add; each item: {text, source_ref?, meta?}. Returns
        number of rows inserted (0 on embedder failure)."""
        if not items:
            return 0
        texts = [it["text"] for it in items]
        vecs = await self.embedder.embed(texts)
        if vecs is None:
            logger.warning(
                "vec.add_many_skipped kind=%s count=%d", kind, len(items),
            )
            return 0
        def _bulk_insert() -> int:
            count = 0
            for it, v in zip(items, vecs):
                self._insert_sync(
                    kind, it["text"], v,
                    it.get("source_ref"), it.get("meta"),
                )
                count += 1
            return count

        return await asyncio.to_thread(_bulk_insert)

    def _insert_sync(
        self,
        kind: str,
        text: str,
        vec: np.ndarray,
        source_ref: Optional[str],
        meta: Optional[dict],
    ) -> int:
        blob = vec.astype(np.float32).tobytes()
        meta_json = json.dumps(meta, ensure_ascii=False) if meta else None
        with self._conn() as c:
            cur = c.execute(
                """
                INSERT INTO memory_vec
                    (kind, source_ref, text, embedding, created_at, meta)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (kind, source_ref, text, blob, time.time(), meta_json),
            )
            return int(cur.lastrowid)

    def delete_by_source(self, source_ref: str) -> int:
        """Remove rows matching a source_ref. Used by the bootstrap
        script when re-indexing a file whose content has changed."""
        with self._conn() as c:
            cur = c.execute(
                "DELETE FROM memory_vec WHERE source_ref = ?", (source_ref,),
            )
            return cur.rowcount

    # ── reads ──────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        top_k: int = 5,
        kinds: Optional[List[str]] = None,
        min_score: float = 0.35,
    ) -> List[dict]:
        """Semantic search. Returns list of
        {id, kind, source_ref, text, score, meta, created_at}.
        Filters out matches below `min_score` (cosine). Empty list on
        embedder failure — never raises.
        """
        if not query or not query.strip():
            return []
        qvec = await self.embedder.embed_one(query)
        if qvec is None:
            return []

        return await asyncio.to_thread(
            self._search_sync, qvec, top_k, kinds, min_score,
        )

    def _search_sync(
        self,
        qvec: np.ndarray,
        top_k: int,
        kinds: Optional[List[str]],
        min_score: float,
    ) -> List[dict]:
        where = ""
        params: list = []
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            where = f"WHERE kind IN ({placeholders})"
            params = list(kinds)

        with self._conn() as c:
            rows = c.execute(
                f"""
                SELECT id, kind, source_ref, text, embedding, created_at, meta
                FROM memory_vec {where}
                """,
                params,
            ).fetchall()

        if not rows:
            return []

        vecs = np.stack(
            [np.frombuffer(r[4], dtype=np.float32) for r in rows]
        )
        # bge-m3 returns unit vectors -> dot == cosine.
        scores = vecs @ qvec
        order = np.argsort(-scores)
        out: list[dict] = []
        for i in order:
            if scores[i] < min_score:
                break
            r = rows[i]
            out.append({
                "id": r[0],
                "kind": r[1],
                "source_ref": r[2],
                "text": r[3],
                "created_at": r[5],
                "meta": json.loads(r[6]) if r[6] else None,
                "score": float(scores[i]),
            })
            if len(out) >= top_k:
                break
        return out

    def count(self, kind: Optional[str] = None) -> int:
        with self._conn() as c:
            if kind:
                row = c.execute(
                    "SELECT COUNT(*) FROM memory_vec WHERE kind=?", (kind,),
                ).fetchone()
            else:
                row = c.execute(
                    "SELECT COUNT(*) FROM memory_vec",
                ).fetchone()
            return int(row[0]) if row else 0

    def has_source(self, source_ref: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM memory_vec WHERE source_ref=? LIMIT 1",
                (source_ref,),
            ).fetchone()
            return row is not None

    def stats(self) -> dict:
        with self._conn() as c:
            rows = c.execute(
                "SELECT kind, COUNT(*) FROM memory_vec GROUP BY kind",
            ).fetchall()
        return {k: n for k, n in rows}
