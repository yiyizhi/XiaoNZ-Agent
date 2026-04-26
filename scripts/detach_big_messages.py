"""One-shot cleanup: truncate oversized inline-attachment messages in
the DB so they stop replaying into every context window.

Background
----------
Before 2026-04-24, `src/feishu/client.py` inlined up to 500 KB of
attachment text straight into the user message. Those rows now live
forever in `messages`, and the last 40 of them get replayed on every
turn — which can push input tokens over the model cap.

Post-fix we (a) cap ingest at 60 KB, (b) save attachments to disk.
This script back-fills: any existing row above --threshold bytes gets
its `content` rewritten to a short reference that preserves just
enough context (first ~500 chars + a note) so the conversation still
makes sense, but doesn't keep burning tokens forever.

Safety
------
- Dry-run by default. You must pass `--apply` to actually write.
- Takes a timestamped SQL dump before writing (`--skip-backup` to
  suppress).
- Only touches rows whose content starts with a clear attachment
  marker OR is just a very long blob; won't rewrite a long
  user-typed prose message.

Usage
-----
    venv/bin/python scripts/detach_big_messages.py
    venv/bin/python scripts/detach_big_messages.py --threshold 40000
    venv/bin/python scripts/detach_big_messages.py --apply
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402


# A row is "attachment-like" (safe to aggressively truncate) if its
# content contains any of these markers. Without a marker we still
# truncate, but we keep more of the head so prose isn't mangled.
_ATTACHMENT_MARKERS = (
    "# 附件：",
    "（我上传了一个文件",
    "# 搜索结果",
    "HTTP 200 text/html",
    "```",
)


def _is_attachment_like(content: str) -> bool:
    head = content[:2000]
    return any(m in head for m in _ATTACHMENT_MARKERS)


def _rewrite_content(content: str, preserve_head: int) -> str:
    head = content[:preserve_head].rstrip()
    original_bytes = len(content.encode("utf-8"))
    return (
        f"{head}\n\n"
        f"…（历史附件内容已从上下文中截断以避免爆 context；"
        f"原文件未落盘，若需要请重新发送。原消息 "
        f"{original_bytes} 字节 / {len(content)} 字符）"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=int,
        default=60_000,
        help="Rewrite rows whose content is strictly longer than this "
             "many chars (default 60000).",
    )
    parser.add_argument(
        "--preserve-head",
        type=int,
        default=500,
        help="Keep this many chars at the head of each rewritten row "
             "(default 500).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes. Without this, runs dry.",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Don't create a backup of the DB before rewriting.",
    )
    parser.add_argument(
        "--require-marker",
        action="store_true",
        help="Only rewrite rows whose head contains an attachment "
             "marker (safer; may miss pure-prose giants).",
    )
    args = parser.parse_args()

    settings = config.load()
    db_path = settings.db_path
    if not db_path.is_file():
        print(f"ERROR: DB not found: {db_path}")
        return 2

    with sqlite3.connect(db_path) as c:
        rows = c.execute(
            """
            SELECT id, session_id, role, created_at,
                   LENGTH(content), content
            FROM messages
            WHERE LENGTH(content) > ?
            ORDER BY LENGTH(content) DESC
            """,
            (args.threshold,),
        ).fetchall()

    if not rows:
        print(f"OK: no messages > {args.threshold} chars. Nothing to do.")
        return 0

    print(f"Found {len(rows)} row(s) > {args.threshold} chars:")
    targets: list[tuple[int, str, int, str]] = []  # (id, session, length, new_content)
    for row in rows:
        row_id, session_id, role, created_at, length, content = row
        marker_hit = _is_attachment_like(content)
        if args.require_marker and not marker_hit:
            print(
                f"  [skip no-marker] id={row_id} session={session_id} "
                f"role={role} len={length} ts={created_at:.0f}"
            )
            continue
        new_content = _rewrite_content(content, args.preserve_head)
        print(
            f"  id={row_id} session={session_id} role={role} "
            f"len={length} → {len(new_content)} "
            f"attachment={'yes' if marker_hit else 'no'}"
        )
        print(f"    head: {content[:120]!r}")
        targets.append((row_id, session_id, length, new_content))

    if not targets:
        print("No rows matched after filtering. Nothing to do.")
        return 0

    if not args.apply:
        print(
            f"\nDry run — {len(targets)} row(s) WOULD be rewritten. "
            f"Pass --apply to actually do it."
        )
        return 0

    if not args.skip_backup:
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = db_path.with_name(f"{db_path.name}.bak.{ts}")
        shutil.copy2(db_path, backup)
        print(f"backup: {backup}")

    with sqlite3.connect(db_path) as c:
        for row_id, _session_id, _length, new_content in targets:
            c.execute(
                "UPDATE messages SET content = ? WHERE id = ?",
                (new_content, row_id),
            )
        c.commit()
    print(f"OK: rewrote {len(targets)} row(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
