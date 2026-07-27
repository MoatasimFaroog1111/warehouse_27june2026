from __future__ import annotations

import os
import re
import threading
import time
import uuid
from pathlib import Path


_RESULT_ID = re.compile(r"^[0-9a-f]{32}$")


class ResultStorage:
    def __init__(self, root: Path, ttl_seconds: int) -> None:
        self.root = root
        self.ttl_seconds = ttl_seconds
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_cleanup = 0.0

    def save_jpeg(self, content: bytes) -> tuple[str, Path]:
        self.cleanup_if_due()
        result_id = uuid.uuid4().hex
        final_path = self.root / f"{result_id}.jpg"
        temp_path = self.root / f".{result_id}.tmp"

        with self._lock:
            temp_path.write_bytes(content)
            os.replace(temp_path, final_path)

        return result_id, final_path

    def get_jpeg(self, result_id: str) -> Path | None:
        if not _RESULT_ID.fullmatch(result_id):
            return None
        path = self.root / f"{result_id}.jpg"
        return path if path.is_file() else None

    def cleanup_if_due(self) -> int:
        now = time.time()
        if now - self._last_cleanup < min(300, self.ttl_seconds):
            return 0
        self._last_cleanup = now
        return self.cleanup(now=now)

    def cleanup(self, now: float | None = None) -> int:
        cutoff = (now or time.time()) - self.ttl_seconds
        removed = 0
        with self._lock:
            for path in self.root.glob("*.jpg"):
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink(missing_ok=True)
                        removed += 1
                except FileNotFoundError:
                    continue
        return removed
