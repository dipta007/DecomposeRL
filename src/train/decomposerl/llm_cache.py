"""SQLite-backed LLM call cache — persists across runs, safe for parallel processes.

Usage:
    cache = LLMCache()                          # default: /tmp/llm_call_cache.db
    cache = LLMCache("/path/to/cache.db")       # custom path
    cache = LLMCache.from_env()                 # reads LLM_CACHE_DB env var

    result = cache.get(key)                     # returns None on miss or error
    cache.set(key, value)                       # no-op on error
    stats = cache.stats                         # {"hits": int, "misses": int}
"""

import logging
import os
import sqlite3
import threading
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class LLMCache:
    """Thread-safe, process-safe SQLite cache for LLM call results."""

    def __init__(self, db_path: str = "/tmp/llm_call_cache.db"):
        self._db_path = db_path
        self._local = threading.local()
        self._stats = {"hits": 0, "misses": 0}
        self._stats_lock = threading.Lock()

    @classmethod
    def from_env(cls, default: str = "/tmp/llm_call_cache.db") -> "LLMCache":
        return cls(os.getenv("LLM_CACHE_DB", default))

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local SQLite connection with WAL mode."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
            self._local.conn = sqlite3.connect(self._db_path, timeout=30)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache "
                "(key TEXT PRIMARY KEY, value TEXT, created_at TEXT)"
            )
            self._local.conn.commit()
        return self._local.conn

    def get(self, key: str) -> Optional[str]:
        """Get a cached value. Returns None on miss or error."""
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT value FROM llm_cache WHERE key = ?", (key,)
            ).fetchone()
            if row is not None:
                with self._stats_lock:
                    self._stats["hits"] += 1
                    if self._stats["hits"] % 10 == 0:
                        logger.info(
                            f"LLM cache: {self._stats['hits']} hits, "
                            f"{self._stats['misses']} misses"
                        )
                return row[0]
            with self._stats_lock:
                self._stats["misses"] += 1
            return None
        except Exception as e:
            logger.warning(f"LLM cache read error: {e}")
            return None

    def set(self, key: str, value: str) -> None:
        """Store a value. No-op on error."""
        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, value, created_at) VALUES (?, ?, ?)",
                (key, value, datetime.now().isoformat()),
            )
            conn.commit()
        except Exception as e:
            logger.warning(f"LLM cache write error: {e}")
