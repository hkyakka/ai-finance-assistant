from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class CacheEntry:
    value: Any
    expires_at: float  # epoch seconds


class TTLCache:
    """
    Simple in-memory TTL cache.
    - Thread-safe
    - Stores arbitrary Python objects
    """

    def __init__(self, default_ttl_seconds: int = 1800, max_items: int = 2048) -> None:
        self.default_ttl_seconds = int(default_ttl_seconds)
        self.max_items = int(max_items)
        self._lock = threading.Lock()
        self._store: Dict[str, CacheEntry] = {}

    def _now(self) -> float:
        return time.time()

    def get(self, key: str) -> Optional[Tuple[Any, int]]:
        """
        Returns (value, remaining_ttl_seconds) if present and not expired, else None.
        """
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            if entry.expires_at <= self._now():
                self._store.pop(key, None)
                return None
            remaining = max(0, int(entry.expires_at - self._now()))
            return entry.value, remaining

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        ttl = self.default_ttl_seconds if ttl_seconds is None else int(ttl_seconds)
        expires_at = self._now() + max(1, ttl)

        with self._lock:
            # simple eviction: if over capacity, drop oldest-expiring entries
            if len(self._store) >= self.max_items:
                # sort by expires_at and remove 10%
                items = sorted(self._store.items(), key=lambda kv: kv[1].expires_at)
                for k, _ in items[: max(1, self.max_items // 10)]:
                    self._store.pop(k, None)

            self._store[key] = CacheEntry(value=value, expires_at=expires_at)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
