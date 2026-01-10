"""In-memory LRU cache layer (AfterImage pattern).

Provides a fast in-process cache for the hottest data:
- Identity facts (always needed, rarely change)
- Recent context lookups

This adds a third tier to our cache hierarchy:
1. In-memory LRU (hot) - microseconds
2. Redis (warm) - milliseconds
3. Postgres (cold) - tens of milliseconds
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Generic, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A cached entry with expiration tracking."""

    value: T
    created_at: float = field(default_factory=time.time)
    ttl: int = 300  # 5 minutes default

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.created_at + self.ttl


class LRUCache(Generic[T]):
    """
    Thread-safe LRU cache with TTL support.

    Features:
    - O(1) get/set operations
    - Automatic eviction of least recently used entries
    - TTL-based expiration
    - Thread-safe operations
    """

    def __init__(self, max_size: int = 100, default_ttl: int = 300) -> None:
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> T | None:
        """
        Get value from cache.

        Moves entry to end (most recently used) on access.

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: T, ttl: int | None = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override
        """
        with self._lock:
            # Remove if exists to update position
            if key in self._cache:
                del self._cache[key]

            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(
                value=value,
                ttl=ttl if ttl is not None else self.default_ttl,
            )

    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Returns:
            True if entry was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()

    def invalidate_prefix(self, prefix: str) -> int:
        """
        Invalidate all entries with keys starting with prefix.

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            to_delete = [k for k in self._cache.keys() if k.startswith(prefix)]
            for key in to_delete:
                del self._cache[key]
            return len(to_delete)

    @property
    def size(self) -> int:
        """Current number of entries."""
        with self._lock:
            return len(self._cache)

    @property
    def stats(self) -> dict[str, Any]:
        """Cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }

    def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired = [k for k, v in self._cache.items() if v.is_expired()]
            for key in expired:
                del self._cache[key]
            return len(expired)


class IdentityCache:
    """
    Specialized cache for user identity data.

    Identity data is accessed on every turn and rarely changes,
    making it ideal for aggressive caching.
    """

    def __init__(self, max_size: int = 100, ttl: int = 300) -> None:
        self._cache: LRUCache[dict[str, Any]] = LRUCache(
            max_size=max_size,
            default_ttl=ttl,
        )

    def get(self, user_id: str) -> dict[str, Any] | None:
        """Get identity for user."""
        return self._cache.get(f"identity:{user_id}")

    def set(self, user_id: str, identity: dict[str, Any]) -> None:
        """Cache identity for user."""
        self._cache.set(f"identity:{user_id}", identity)
        logger.debug("identity_cached", user_id=user_id)

    def invalidate(self, user_id: str) -> None:
        """Invalidate cached identity for user."""
        self._cache.delete(f"identity:{user_id}")
        logger.debug("identity_cache_invalidated", user_id=user_id)

    @property
    def stats(self) -> dict[str, Any]:
        """Cache statistics."""
        return self._cache.stats


class ContextCache:
    """
    Cache for recently retrieved context.

    Caches full MemoryContext objects for repeat queries
    within a short time window.
    """

    def __init__(self, max_size: int = 50, ttl: int = 60) -> None:
        # Shorter TTL for context (it's more dynamic)
        self._cache: LRUCache[dict[str, Any]] = LRUCache(
            max_size=max_size,
            default_ttl=ttl,
        )

    def _make_key(self, user_id: str, query: str | None, project_id: str | None) -> str:
        """Generate cache key from context parameters."""
        parts = [user_id]
        if query:
            # Hash the query for consistent key length
            parts.append(str(hash(query)))
        if project_id:
            parts.append(project_id)
        return ":".join(parts)

    def get(
        self,
        user_id: str,
        query: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Get cached context."""
        key = self._make_key(user_id, query, project_id)
        return self._cache.get(key)

    def set(
        self,
        user_id: str,
        context_data: dict[str, Any],
        query: str | None = None,
        project_id: str | None = None,
    ) -> None:
        """Cache context data."""
        key = self._make_key(user_id, query, project_id)
        self._cache.set(key, context_data)

    def invalidate_user(self, user_id: str) -> int:
        """Invalidate all cached contexts for a user."""
        return self._cache.invalidate_prefix(user_id)

    @property
    def stats(self) -> dict[str, Any]:
        """Cache statistics."""
        return self._cache.stats


class CortexCache:
    """
    Unified cache manager for Cortex.

    Provides a single interface to all in-memory caches.
    """

    _instance: "CortexCache | None" = None

    def __init__(
        self,
        identity_size: int = 100,
        context_size: int = 50,
        ttl: int = 300,
    ) -> None:
        self.identity = IdentityCache(max_size=identity_size, ttl=ttl)
        self.context = ContextCache(max_size=context_size, ttl=60)

    @classmethod
    def get_instance(
        cls,
        identity_size: int = 100,
        context_size: int = 50,
        ttl: int = 300,
    ) -> "CortexCache":
        """Get or create singleton cache instance."""
        if cls._instance is None:
            cls._instance = cls(identity_size, context_size, ttl)
        return cls._instance

    def clear_all(self) -> None:
        """Clear all caches."""
        self.identity._cache.clear()
        self.context._cache.clear()
        logger.info("all_caches_cleared")

    def cleanup_expired(self) -> dict[str, int]:
        """Cleanup expired entries from all caches."""
        return {
            "identity": self.identity._cache.cleanup_expired(),
            "context": self.context._cache.cleanup_expired(),
        }

    @property
    def stats(self) -> dict[str, Any]:
        """Combined cache statistics."""
        return {
            "identity": self.identity.stats,
            "context": self.context.stats,
        }
