"""Tests for RedisStore."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from cortex.config import RedisConfig
from cortex.models import Memory, MemoryType
from cortex.stores.redis_store import RedisStore


@pytest.fixture
async def redis_store():
    """Create a test Redis store using DB 15 for isolation."""
    config = RedisConfig(host="localhost", port=6379, db=15)
    store = RedisStore(config)
    await store.connect()
    yield store
    # Cleanup
    assert store.client is not None
    await store.client.flushdb()
    await store.close()


class TestIdentity:
    """Tests for identity operations."""

    @pytest.mark.asyncio
    async def test_set_and_get_identity(self, redis_store: RedisStore):
        user_id = "test-user-1"
        identity = {
            "name": "Josh",
            "family": {"children": ["Maddie", "Anne", "Thomas"]},
            "preferences": {"style": "direct"},
        }

        await redis_store.set_identity(user_id, identity)
        result = await redis_store.get_identity(user_id)

        assert result["name"] == "Josh"
        assert result["family"]["children"] == ["Maddie", "Anne", "Thomas"]
        assert "updated_at" in result

    @pytest.mark.asyncio
    async def test_update_identity_field(self, redis_store: RedisStore):
        user_id = "test-user-2"
        await redis_store.set_identity(user_id, {"name": "Josh"})
        await redis_store.update_identity_field(user_id, "location", "NYC")

        result = await redis_store.get_identity(user_id)
        assert result["name"] == "Josh"
        assert result["location"] == "NYC"

    @pytest.mark.asyncio
    async def test_get_empty_identity(self, redis_store: RedisStore):
        result = await redis_store.get_identity("nonexistent-user")
        assert result == {}

    @pytest.mark.asyncio
    async def test_delete_identity(self, redis_store: RedisStore):
        user_id = "test-user-3"
        await redis_store.set_identity(user_id, {"name": "Test"})
        await redis_store.delete_identity(user_id)

        result = await redis_store.get_identity(user_id)
        assert result == {}


class TestSession:
    """Tests for session operations."""

    @pytest.mark.asyncio
    async def test_session_lifecycle(self, redis_store: RedisStore):
        user_id = "test-user-session-1"

        # Start session
        await redis_store.set_session(
            user_id,
            {
                "started_at": datetime.utcnow().isoformat(),
                "current_topic": "testing",
            },
        )

        # Update session
        await redis_store.update_session(user_id, {"current_topic": "memory"})

        session = await redis_store.get_session(user_id)
        assert session["current_topic"] == "memory"
        assert "last_active" in session

        # Clear session
        await redis_store.clear_session(user_id)
        session = await redis_store.get_session(user_id)
        assert session == {}

    @pytest.mark.asyncio
    async def test_session_with_json_fields(self, redis_store: RedisStore):
        user_id = "test-user-session-2"
        await redis_store.set_session(
            user_id,
            {
                "active_goals": ["finish TDD", "write tests"],
                "context": {"project": "cortex"},
            },
        )

        session = await redis_store.get_session(user_id)
        assert session["active_goals"] == ["finish TDD", "write tests"]


class TestWorkingMemory:
    """Tests for working memory operations."""

    @pytest.mark.asyncio
    async def test_add_and_get_working_memory(self, redis_store: RedisStore):
        user_id = "test-user-working-1"

        memory = Memory(
            content="Test memory",
            user_id=user_id,
            memory_type=MemoryType.WORKING,
            emotional_score=0.5,
        )

        # Add with 1 hour TTL
        await redis_store.add_working(user_id, memory, ttl=3600)

        working = await redis_store.get_working(user_id)
        assert len(working) == 1
        assert working[0].content == "Test memory"

    @pytest.mark.asyncio
    async def test_expired_working_memory_cleaned(self, redis_store: RedisStore):
        user_id = "test-user-working-2"
        assert redis_store.client is not None

        # Manually add expired memory
        key = redis_store._key(user_id, "working")
        expired_data = {
            "id": "test-id",
            "content": "Expired",
            "type": "working",
            "emotion": 0.5,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
        }
        await redis_store.client.zadd(key, {json.dumps(expired_data): 1234567890})

        # Get should clean up
        working = await redis_store.get_working(user_id)
        assert len(working) == 0

    @pytest.mark.asyncio
    async def test_multiple_working_memories(self, redis_store: RedisStore):
        user_id = "test-user-working-3"

        for i in range(5):
            memory = Memory(
                content=f"Memory {i}",
                user_id=user_id,
                memory_type=MemoryType.WORKING,
                emotional_score=0.3 + i * 0.1,
            )
            await redis_store.add_working(user_id, memory, ttl=3600)

        working = await redis_store.get_working(user_id)
        assert len(working) == 5


class TestRecentBuffer:
    """Tests for recent buffer operations."""

    @pytest.mark.asyncio
    async def test_add_and_get_recent(self, redis_store: RedisStore):
        user_id = "test-user-recent-1"

        for i in range(5):
            await redis_store.add_recent(user_id, f"memory-{i}")

        recent = await redis_store.get_recent(user_id, limit=3)
        assert len(recent) == 3
        assert recent[0] == "memory-4"  # Most recent first

    @pytest.mark.asyncio
    async def test_recent_buffer_caps_at_max(self, redis_store: RedisStore):
        user_id = "test-user-recent-2"

        # Add more than max
        for i in range(60):
            await redis_store.add_recent(user_id, f"memory-{i}", max_size=50)

        recent = await redis_store.get_recent(user_id, limit=100)
        assert len(recent) == 50


class TestUtilities:
    """Tests for utility operations."""

    @pytest.mark.asyncio
    async def test_ping(self, redis_store: RedisStore):
        assert await redis_store.ping() is True

    @pytest.mark.asyncio
    async def test_get_active_user_ids(self, redis_store: RedisStore):
        # Create sessions for multiple users
        for i in range(3):
            await redis_store.set_session(f"user-{i}", {"topic": f"topic-{i}"})

        user_ids = await redis_store.get_active_user_ids()
        assert len(user_ids) == 3

    @pytest.mark.asyncio
    async def test_flush_user(self, redis_store: RedisStore):
        user_id = "test-user-flush"

        await redis_store.set_identity(user_id, {"name": "Test"})
        await redis_store.set_session(user_id, {"topic": "test"})

        await redis_store.flush_user(user_id)

        assert await redis_store.get_identity(user_id) == {}
        assert await redis_store.get_session(user_id) == {}
