"""SQLite-backed session store.

Design borrowed from NousResearch/hermes-agent's `hermes_state.py`:
single-file SQLite, three tables (sessions, messages, processed_events).
Synchronous sqlite3 is fine — all writes happen from a single asyncio
event loop and individual calls are fast.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,        -- 'feishu_p2p' | 'feishu_group'
    peer_id     TEXT NOT NULL,        -- open_id or chat_id
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS processed_events (
    event_id    TEXT PRIMARY KEY,
    received_at REAL NOT NULL
);

-- Rolling summary per session. Populated by AgentLoop's compression
-- step when history exceeds the configured threshold; the summarized
-- messages are then deleted from `messages`.
CREATE TABLE IF NOT EXISTS session_summaries (
    session_id  TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    summary     TEXT NOT NULL DEFAULT '',
    updated_at  REAL NOT NULL
);

-- Daily conversation digests. One row per day, generated automatically
-- by the agent loop when a new day starts. Contains a compressed
-- summary of ALL sessions' conversations for that date.
CREATE TABLE IF NOT EXISTS daily_digests (
    date        TEXT PRIMARY KEY,   -- 'YYYY-MM-DD'
    summary     TEXT NOT NULL,
    msg_count   INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);
"""


class SessionStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, isolation_level=None)  # autocommit
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        try:
            yield conn
        finally:
            conn.close()

    # ── session lifecycle ──────────────────────────────────────────

    def ensure_session(self, session_id: str, source: str, peer_id: str) -> None:
        now = time.time()
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO sessions (id, source, peer_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (session_id, source, peer_id, now, now),
            )

    def reset_session(self, session_id: str) -> int:
        """Delete all messages AND the rolling summary for a session.
        Returns the number of message rows deleted (summary row is
        cleared silently)."""
        with self._conn() as c:
            cur = c.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            c.execute(
                "DELETE FROM session_summaries WHERE session_id = ?",
                (session_id,),
            )
            return cur.rowcount

    # ── messages ───────────────────────────────────────────────────

    def append_message(self, session_id: str, role: str, content: str) -> None:
        if role not in ("user", "assistant"):
            raise ValueError(f"invalid role: {role}")
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, time.time()),
            )
            c.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (time.time(), session_id),
            )

    def get_messages(
        self, session_id: str, max_turns: int = 20
    ) -> list[dict[str, str]]:
        """Return up to max_turns*2 messages (one turn = user+assistant),
        in chronological order, with strict role alternation enforced.
        """
        limit = max_turns * 2
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        # reverse to chronological order
        messages = [{"role": r, "content": ct} for r, ct in reversed(rows)]

        # Enforce role alternation: collapse consecutive same-role messages.
        # Anthropic API requires user/assistant to alternate.
        normalized: list[dict[str, str]] = []
        for m in messages:
            if normalized and normalized[-1]["role"] == m["role"]:
                # Merge content with a newline separator
                normalized[-1]["content"] += "\n\n" + m["content"]
            else:
                normalized.append(m)

        # Anthropic also requires the first message to be 'user'
        if normalized and normalized[0]["role"] != "user":
            normalized = normalized[1:]

        return normalized

    def count_messages(self, session_id: str) -> int:
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return int(row[0]) if row else 0

    def get_oldest_messages(
        self, session_id: str, limit: int
    ) -> list[dict[str, str]]:
        """Return the `limit` oldest messages in chronological order.

        Used by the compression step to build the text it sends to
        the summarizer.
        """
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [{"role": r, "content": ct} for r, ct in rows]

    def delete_oldest_messages(self, session_id: str, delete_n: int) -> int:
        """Delete the `delete_n` oldest messages for a session.

        Returns the number of rows actually deleted.
        """
        if delete_n <= 0:
            return 0
        with self._conn() as c:
            cur = c.execute(
                """
                DELETE FROM messages
                WHERE id IN (
                    SELECT id FROM messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                    LIMIT ?
                )
                """,
                (session_id, delete_n),
            )
            return cur.rowcount

    # ── summaries ──────────────────────────────────────────────────

    def get_summary(self, session_id: str) -> str:
        with self._conn() as c:
            row = c.execute(
                "SELECT summary FROM session_summaries WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row[0] if row else ""

    def set_summary(self, session_id: str, summary: str) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO session_summaries (session_id, summary, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary = excluded.summary,
                    updated_at = excluded.updated_at
                """,
                (session_id, summary, time.time()),
            )

    # ── event dedup ────────────────────────────────────────────────

    def is_event_processed(self, event_id: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM processed_events WHERE event_id = ?", (event_id,)
            ).fetchone()
            return row is not None

    def mark_event_processed(self, event_id: str) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT OR IGNORE INTO processed_events (event_id, received_at)
                VALUES (?, ?)
                """,
                (event_id, time.time()),
            )

    def prune_old_events(self, older_than_seconds: float = 86400) -> int:
        """Remove processed_events older than given age. Returns rows deleted."""
        cutoff = time.time() - older_than_seconds
        with self._conn() as c:
            cur = c.execute(
                "DELETE FROM processed_events WHERE received_at < ?", (cutoff,)
            )
            return cur.rowcount

    # ── daily digests ─────────────────────────────────────────────

    def get_messages_for_date(self, date_str: str) -> list[dict[str, str]]:
        """Return all messages from all sessions for a given date
        (format 'YYYY-MM-DD'), ordered chronologically."""
        from datetime import datetime
        try:
            day_start = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return []
        ts_start = day_start.timestamp()
        ts_end = ts_start + 86400
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT m.session_id, m.role, m.content
                FROM messages m
                WHERE m.created_at >= ? AND m.created_at < ?
                ORDER BY m.created_at ASC
                """,
                (ts_start, ts_end),
            ).fetchall()
        return [
            {"session_id": sid, "role": r, "content": ct}
            for sid, r, ct in rows
        ]

    def has_daily_digest(self, date_str: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM daily_digests WHERE date = ?", (date_str,)
            ).fetchone()
            return row is not None

    def save_daily_digest(
        self, date_str: str, summary: str, msg_count: int
    ) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO daily_digests (date, summary, msg_count, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    summary = excluded.summary,
                    msg_count = excluded.msg_count,
                    created_at = excluded.created_at
                """,
                (date_str, summary, msg_count, time.time()),
            )

    def get_daily_digest(self, date_str: str) -> str:
        with self._conn() as c:
            row = c.execute(
                "SELECT summary FROM daily_digests WHERE date = ?",
                (date_str,),
            ).fetchone()
            return row[0] if row else ""

    def list_daily_digests(self, limit: int = 30) -> list[dict]:
        """Return the most recent daily digests, newest first."""
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT date, msg_count, length(summary) as chars
                FROM daily_digests
                ORDER BY date DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {"date": d, "msg_count": n, "summary_chars": ch}
            for d, n, ch in rows
        ]

    def search_daily_digests(self, keyword: str, limit: int = 10) -> list[dict]:
        """Full-text search across daily digest summaries."""
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT date, summary, msg_count
                FROM daily_digests
                WHERE summary LIKE ?
                ORDER BY date DESC
                LIMIT ?
                """,
                (f"%{keyword}%", limit),
            ).fetchall()
        return [
            {"date": d, "summary": s, "msg_count": n}
            for d, s, n in rows
        ]
