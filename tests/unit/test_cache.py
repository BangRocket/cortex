"""Unit tests for LRU cache (AfterImage pattern)."""

from __future__ import annotations

import time

import pytest

from cortex.utils.cache import (
    CortexCache,
    ContextCache,
    IdentityCache,
    LRUCache,
)


class TestLRUCache:
    """Test LRUCache."""

    @pytest.fixture
    def cache(self) -> LRUCache:
        """Create test cache."""
        return LRUCache(max_size=5, default_ttl=300)

    def test_set_and_get(self, cache: LRUCache):
        """Test basic set and get."""
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_key(self, cache: LRUCache):
        """Test getting a missing key."""
        assert cache.get("nonexistent") is None

    def test_eviction_on_capacity(self, cache: LRUCache):
        """Test LRU eviction when at capacity."""
        # Fill cache
        for i in range(5):
            cache.set(f"key{i}", f"value{i}")

        # All should be present
        for i in range(5):
            assert cache.get(f"key{i}") == f"value{i}"

        # Add one more - should evict oldest (key0)
        cache.set("key5", "value5")

        assert cache.get("key0") is None
        assert cache.get("key5") == "value5"

    def test_lru_ordering(self, cache: LRUCache):
        """Test that access updates LRU order."""
        # Fill cache
        for i in range(5):
            cache.set(f"key{i}", f"value{i}")

        # Access key0 to make it recently used
        cache.get("key0")

        # Add new key - should evict key1 (oldest not accessed)
        cache.set("key5", "value5")

        assert cache.get("key0") == "value0"  # Still present
        assert cache.get("key1") is None  # Evicted

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = LRUCache(max_size=10, default_ttl=1)  # 1 second TTL
        cache.set("key1", "value1")

        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        assert cache.get("key1") is None

    def test_custom_ttl(self, cache: LRUCache):
        """Test custom TTL per entry."""
        cache.set("short", "value", ttl=1)
        cache.set("long", "value", ttl=300)

        time.sleep(1.1)

        assert cache.get("short") is None
        assert cache.get("long") == "value"

    def test_delete(self, cache: LRUCache):
        """Test deleting entries."""
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        result = cache.delete("key1")
        assert result is True
        assert cache.get("key1") is None

    def test_delete_missing(self, cache: LRUCache):
        """Test deleting non-existent key."""
        result = cache.delete("nonexistent")
        assert result is False

    def test_clear(self, cache: LRUCache):
        """Test clearing cache."""
        for i in range(3):
            cache.set(f"key{i}", f"value{i}")

        cache.clear()

        assert cache.size == 0
        for i in range(3):
            assert cache.get(f"key{i}") is None

    def test_invalidate_prefix(self, cache: LRUCache):
        """Test invalidating by prefix."""
        cache.set("user:1:identity", "data1")
        cache.set("user:1:session", "data2")
        cache.set("user:2:identity", "data3")

        removed = cache.invalidate_prefix("user:1")

        assert removed == 2
        assert cache.get("user:1:identity") is None
        assert cache.get("user:1:session") is None
        assert cache.get("user:2:identity") == "data3"

    def test_size(self, cache: LRUCache):
        """Test size property."""
        assert cache.size == 0

        cache.set("key1", "value1")
        assert cache.size == 1

        cache.set("key2", "value2")
        assert cache.size == 2

    def test_stats(self, cache: LRUCache):
        """Test statistics."""
        cache.set("key1", "value1")

        # Hit
        cache.get("key1")
        # Miss
        cache.get("nonexistent")

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cleanup_expired(self):
        """Test cleaning up expired entries."""
        cache = LRUCache(max_size=10, default_ttl=1)

        cache.set("key1", "value1")
        cache.set("key2", "value2", ttl=300)  # Won't expire

        time.sleep(1.1)

        removed = cache.cleanup_expired()
        assert removed == 1
        assert cache.get("key2") == "value2"


class TestIdentityCache:
    """Test IdentityCache."""

    @pytest.fixture
    def cache(self) -> IdentityCache:
        """Create test cache."""
        return IdentityCache(max_size=10, ttl=300)

    def test_set_and_get(self, cache: IdentityCache):
        """Test identity caching."""
        identity = {"name": "Test User", "email": "test@example.com"}
        cache.set("user1", identity)

        result = cache.get("user1")
        assert result == identity

    def test_get_missing(self, cache: IdentityCache):
        """Test getting missing identity."""
        assert cache.get("nonexistent") is None

    def test_invalidate(self, cache: IdentityCache):
        """Test invalidating identity."""
        cache.set("user1", {"name": "Test"})
        cache.invalidate("user1")
        assert cache.get("user1") is None

    def test_stats(self, cache: IdentityCache):
        """Test identity cache stats."""
        cache.set("user1", {"name": "Test"})
        cache.get("user1")  # Hit
        cache.get("user2")  # Miss

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1


class TestContextCache:
    """Test ContextCache."""

    @pytest.fixture
    def cache(self) -> ContextCache:
        """Create test cache."""
        return ContextCache(max_size=10, ttl=60)

    def test_set_and_get(self, cache: ContextCache):
        """Test context caching."""
        context = {"identity": {}, "session": {}}
        cache.set("user1", context)

        result = cache.get("user1")
        assert result == context

    def test_get_with_query(self, cache: ContextCache):
        """Test context with query."""
        context1 = {"query": "hello"}
        context2 = {"query": "world"}

        cache.set("user1", context1, query="hello")
        cache.set("user1", context2, query="world")

        assert cache.get("user1", query="hello") == context1
        assert cache.get("user1", query="world") == context2
        assert cache.get("user1", query="other") is None

    def test_get_with_project(self, cache: ContextCache):
        """Test context with project."""
        context = {"project": "test"}
        cache.set("user1", context, project_id="proj1")

        assert cache.get("user1", project_id="proj1") == context
        assert cache.get("user1", project_id="proj2") is None

    def test_invalidate_user(self, cache: ContextCache):
        """Test invalidating all contexts for a user."""
        cache.set("user1", {"a": 1}, query="q1")
        cache.set("user1", {"b": 2}, query="q2")
        cache.set("user2", {"c": 3})

        removed = cache.invalidate_user("user1")

        assert removed == 2
        assert cache.get("user1", query="q1") is None
        assert cache.get("user1", query="q2") is None
        assert cache.get("user2") is not None


class TestCortexCache:
    """Test CortexCache unified interface."""

    @pytest.fixture
    def cache(self) -> CortexCache:
        """Create test cache."""
        # Reset singleton for testing
        CortexCache._instance = None
        return CortexCache.get_instance(
            identity_size=10,
            context_size=5,
            ttl=300,
        )

    def test_singleton(self, cache: CortexCache):
        """Test singleton pattern."""
        cache2 = CortexCache.get_instance()
        assert cache is cache2

    def test_has_identity_cache(self, cache: CortexCache):
        """Test identity cache is available."""
        cache.identity.set("user1", {"name": "Test"})
        assert cache.identity.get("user1") == {"name": "Test"}

    def test_has_context_cache(self, cache: CortexCache):
        """Test context cache is available."""
        cache.context.set("user1", {"data": "test"})
        assert cache.context.get("user1") == {"data": "test"}

    def test_clear_all(self, cache: CortexCache):
        """Test clearing all caches."""
        cache.identity.set("user1", {"name": "Test"})
        cache.context.set("user1", {"data": "test"})

        cache.clear_all()

        assert cache.identity.get("user1") is None
        assert cache.context.get("user1") is None

    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        CortexCache._instance = None
        cache = CortexCache(identity_size=10, context_size=5, ttl=1)

        cache.identity.set("user1", {"name": "Test"})

        time.sleep(1.1)

        result = cache.cleanup_expired()
        assert result["identity"] >= 0

    def test_stats(self, cache: CortexCache):
        """Test combined stats."""
        cache.identity.set("user1", {"name": "Test"})
        cache.identity.get("user1")

        stats = cache.stats

        assert "identity" in stats
        assert "context" in stats
        assert stats["identity"]["hits"] >= 1
