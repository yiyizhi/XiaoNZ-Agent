"""MEMORY.md store.

Thin wrapper around the single MEMORY.md file. Purposely dumb:
read returns raw text, write replaces the whole file atomically
(write-temp-then-rename). The LLM is responsible for producing a
sensible full replacement via the `update_memory` tool.

Before every write, the current file is backed up to an `archive/`
subdirectory with a timestamp. The 30 most recent backups are kept;
older ones are pruned automatically.

SOUL.md is read-only from the agent's perspective and lives in the
same directory — not managed here.
"""
from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_BACKUPS = 30


class MemoryStore:
    def __init__(self, memory_path: Path):
        self.path = memory_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._archive_dir = self.path.parent / "archive"

    def read(self) -> str:
        if not self.path.is_file():
            return ""
        return self.path.read_text(encoding="utf-8")

    def _backup(self) -> None:
        """Copy current MEMORY.md to archive/ with a timestamp suffix."""
        if not self.path.is_file():
            return
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        dst = self._archive_dir / f"MEMORY-{ts}.md"
        try:
            import shutil
            shutil.copy2(str(self.path), str(dst))
            logger.info("memory.backup path=%s", dst)
        except OSError:
            logger.exception("memory.backup_failed")
        # Prune old backups
        backups = sorted(self._archive_dir.glob("MEMORY-*.md"))
        for old in backups[:-_MAX_BACKUPS]:
            try:
                old.unlink()
            except OSError:
                pass

    def write(self, new_content: str) -> int:
        """Atomically replace MEMORY.md. Returns new byte length.

        A timestamped backup of the current file is created before
        the replacement so no data is ever permanently lost.
        """
        self._backup()

        data = new_content.rstrip() + "\n"
        fd, tmp_path = tempfile.mkstemp(
            prefix=".memory.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        size = len(data.encode("utf-8"))
        logger.info("memory.write bytes=%d", size)
        return size

    def list_backups(self) -> list[str]:
        """Return backup filenames sorted newest-first."""
        if not self._archive_dir.is_dir():
            return []
        return [f.name for f in sorted(
            self._archive_dir.glob("MEMORY-*.md"), reverse=True
        )]

    def read_backup(self, filename: str) -> str | None:
        """Read a specific backup by filename."""
        p = self._archive_dir / filename
        if not p.is_file():
            return None
        return p.read_text(encoding="utf-8")
