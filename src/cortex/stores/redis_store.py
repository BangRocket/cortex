"""Redis store for hot/fast memory layer."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any

import redis.asyncio as redis
import structlog

from cortex.config import RedisConfig
from cortex.models import Memory, MemoryType

logger = structlog.get_logger(__name__)


class RedisStore:
    """
    Redis store for hot memory layer.

    Handles:
    - Identity (core facts, always loaded)
    - Session state (current conversation state)
    - Working memory (recent memories with TTL decay)
    - Recent buffer (last N memory IDs)
    """

    def __init__(self, config: RedisConfig) -> None:
        self.config = config
        self.prefix = config.prefix
        self.client: redis.Redis | None = None

    async def connect(self) -> None:
        """Initialize Redis connection."""
        self.client = redis.Redis(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            ssl=self.config.ssl,
            decode_responses=True,
            max_connections=self.config.max_connections,
        )
        await self.client.ping()
        logger.info("redis_connected", host=self.config.host, port=self.config.port)

    async def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            logger.info("redis_disconnected")

    def _key(self, user_id: str, suffix: str) -> str:
        """Generate namespaced key."""
        return f"{self.prefix}:{user_id}:{suffix}"

    # ==================== IDENTITY ====================

    async def get_identity(self, user_id: str) -> dict[str, Any]:
        """Get all identity fields for a user."""
        assert self.client is not None
        data = await self.client.hgetall(self._key(user_id, "identity"))
        if not data:
            return {}

        # Parse JSON fields
        result: dict[str, Any] = {}
        json_fields = {"family", "preferences", "key_facts"}
        for k, v in data.items():
            if k in json_fields:
                try:
                    result[k] = json.loads(v)
                except json.JSONDecodeError:
                    result[k] = v
            else:
                result[k] = v
        return result

    async def set_identity(self, user_id: str, identity: dict[str, Any]) -> None:
        """Set all identity fields (overwrites)."""
        assert self.client is not None
        key = self._key(user_id, "identity")

        # Serialize nested objects
        flat: dict[str, str] = {}
        for k, v in identity.items():
            if isinstance(v, (dict, list)):
                flat[k] = json.dumps(v)
            else:
                flat[k] = str(v) if v is not None else ""

        flat["updated_at"] = datetime.utcnow().isoformat()

        async with self.client.pipeline() as pipe:
            await pipe.delete(key)
            if flat:
                await pipe.hset(key, mapping=flat)
            await pipe.execute()

        logger.debug("identity_set", user_id=user_id, fields=len(flat))

    async def update_identity_field(self, user_id: str, field: str, value: Any) -> None:
        """Update a single identity field."""
        assert self.client is not None
        key = self._key(user_id, "identity")
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        await self.client.hset(key, field, str(value))
        await self.client.hset(key, "updated_at", datetime.utcnow().isoformat())
        logger.debug("identity_field_updated", user_id=user_id, field=field)

    async def delete_identity(self, user_id: str) -> None:
        """Delete all identity data for a user."""
        assert self.client is not None
        await self.client.delete(self._key(user_id, "identity"))

    # ==================== SESSION ====================

    async def get_session(self, user_id: str) -> dict[str, Any]:
        """Get current session state."""
        assert self.client is not None
        data = await self.client.hgetall(self._key(user_id, "session"))
        if not data:
            return {}

        # Parse JSON fields
        result: dict[str, Any] = {}
        json_fields = {"active_goals", "context"}
        for k, v in data.items():
            if k in json_fields:
                try:
                    result[k] = json.loads(v)
                except json.JSONDecodeError:
                    result[k] = v
            else:
                result[k] = v
        return result

    async def set_session(self, user_id: str, session: dict[str, Any], ttl: int = 86400) -> None:
        """Set session state (overwrites)."""
        assert self.client is not None
        key = self._key(user_id, "session")

        flat: dict[str, str] = {}
        for k, v in session.items():
            if isinstance(v, (dict, list)):
                flat[k] = json.dumps(v)
            else:
                flat[k] = str(v) if v is not None else ""

        async with self.client.pipeline() as pipe:
            await pipe.delete(key)
            if flat:
                await pipe.hset(key, mapping=flat)
            await pipe.expire(key, ttl)
            await pipe.execute()

        logger.debug("session_set", user_id=user_id)

    async def update_session(
        self, user_id: str, updates: dict[str, Any], ttl: int = 86400
    ) -> bool:
        """Update specific session fields."""
        assert self.client is not None
        key = self._key(user_id, "session")

        flat: dict[str, str] = {}
        for k, v in updates.items():
            if isinstance(v, (dict, list)):
                flat[k] = json.dumps(v)
            else:
                flat[k] = str(v) if v is not None else ""

        flat["last_active"] = datetime.utcnow().isoformat()

        await self.client.hset(key, mapping=flat)
        await self.client.expire(key, ttl)
        logger.debug("session_updated", user_id=user_id, fields=len(updates))
        return True

    async def clear_session(self, user_id: str) -> None:
        """Clear session data."""
        assert self.client is not None
        await self.client.delete(self._key(user_id, "session"))
        logger.debug("session_cleared", user_id=user_id)

    # ==================== WORKING MEMORY ====================

    async def add_working(self, user_id: str, memory: Memory, ttl: int) -> None:
        """Add a memory to working memory with TTL."""
        assert self.client is not None
        key = self._key(user_id, "working")

        # Serialize memory
        mem_data = {
            "id": memory.id or str(uuid.uuid4()),
            "content": memory.content,
            "type": memory.memory_type.value,
            "emotion": memory.emotional_score,
            "created_at": memory.created_at.isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(seconds=ttl)).isoformat(),
        }

        score = datetime.utcnow().timestamp()
        await self.client.zadd(key, {json.dumps(mem_data): score})
        logger.debug(
            "working_memory_added",
            user_id=user_id,
            memory_id=mem_data["id"],
            ttl=ttl,
        )

    async def _get_working_with_cleanup(self, user_id: str) -> tuple[list[Memory], int]:
        """Get non-expired working memories and cleanup expired ones.

        Returns:
            Tuple of (active memories, count of expired entries removed)
        """
        assert self.client is not None
        key = self._key(user_id, "working")

        # Get all members
        members = await self.client.zrange(key, 0, -1)

        now = datetime.utcnow()
        result: list[Memory] = []
        expired: list[str] = []

        for member in members:
            try:
                data = json.loads(member)
                expires_at = datetime.fromisoformat(data["expires_at"])

                if expires_at > now:
                    result.append(
                        Memory(
                            id=data["id"],
                            content=data["content"],
                            memory_type=MemoryType(data["type"]),
                            emotional_score=data["emotion"],
                            created_at=datetime.fromisoformat(data["created_at"]),
                            user_id=user_id,
                        )
                    )
                else:
                    expired.append(member)
            except (json.JSONDecodeError, KeyError, ValueError):
                expired.append(member)

        # Clean up expired
        expired_count = 0
        if expired:
            await self.client.zrem(key, *expired)
            expired_count = len(expired)
            logger.debug("working_memory_expired_removed", user_id=user_id, count=expired_count)

        return result, expired_count

    async def get_working(self, user_id: str) -> list[Memory]:
        """Get all non-expired working memories."""
        memories, _ = await self._get_working_with_cleanup(user_id)
        return memories

    async def cleanup_expired_working(self, user_id: str) -> int:
        """Remove expired working memories. Returns count removed."""
        _, expired_count = await self._get_working_with_cleanup(user_id)
        return expired_count

    async def clear_working(self, user_id: str) -> None:
        """Clear all working memory."""
        assert self.client is not None
        await self.client.delete(self._key(user_id, "working"))

    # ==================== RECENT BUFFER ====================

    async def add_recent(self, user_id: str, memory_id: str, max_size: int = 50) -> None:
        """Add a memory ID to the recent buffer."""
        assert self.client is not None
        key = self._key(user_id, "recent")
        async with self.client.pipeline() as pipe:
            await pipe.lpush(key, memory_id)
            await pipe.ltrim(key, 0, max_size - 1)
            await pipe.execute()

    async def get_recent(self, user_id: str, limit: int = 10) -> list[str]:
        """Get recent memory IDs."""
        assert self.client is not None
        key = self._key(user_id, "recent")
        return await self.client.lrange(key, 0, limit - 1)

    # ==================== META ====================

    async def get_meta(self, user_id: str) -> dict[str, Any]:
        """Get user metadata."""
        assert self.client is not None
        data = await self.client.hgetall(self._key(user_id, "meta"))
        return dict(data) if data else {}

    async def update_meta(self, user_id: str, updates: dict[str, Any]) -> None:
        """Update user metadata."""
        assert self.client is not None
        flat = {k: str(v) for k, v in updates.items()}
        await self.client.hset(self._key(user_id, "meta"), mapping=flat)

    # ==================== PROJECT ====================

    async def get_project_meta(self, project_id: str) -> dict[str, Any] | None:
        """Get project metadata."""
        assert self.client is not None
        data = await self.client.hgetall(f"{self.prefix}:project:{project_id}")
        if not data:
            return None

        result: dict[str, Any] = {}
        for k, v in data.items():
            if k == "members":
                try:
                    result[k] = json.loads(v)
                except json.JSONDecodeError:
                    result[k] = []
            else:
                result[k] = v
        return result

    async def set_project_meta(self, project_id: str, meta: dict[str, Any]) -> None:
        """Set project metadata."""
        assert self.client is not None
        key = f"{self.prefix}:project:{project_id}"

        flat: dict[str, str] = {}
        for k, v in meta.items():
            if isinstance(v, (dict, list)):
                flat[k] = json.dumps(v)
            else:
                flat[k] = str(v) if v is not None else ""

        await self.client.hset(key, mapping=flat)

    # ==================== UTILITIES ====================

    async def ping(self) -> bool:
        """Check if Redis is connected."""
        try:
            assert self.client is not None
            await self.client.ping()
            return True
        except Exception:
            return False

    async def get_active_user_ids(self) -> list[str]:
        """Get list of active user IDs (users with session data)."""
        assert self.client is not None
        pattern = f"{self.prefix}:*:session"
        user_ids: list[str] = []
        async for key in self.client.scan_iter(match=pattern):
            # Extract user_id from key pattern: prefix:user_id:session
            parts = key.split(":")
            if len(parts) == 3:
                user_ids.append(parts[1])
        return user_ids

    async def flush_user(self, user_id: str) -> None:
        """Delete all data for a user (use with caution)."""
        assert self.client is not None
        pattern = f"{self.prefix}:{user_id}:*"
        keys: list[str] = []
        async for key in self.client.scan_iter(match=pattern):
            keys.append(key)
        if keys:
            await self.client.delete(*keys)
            logger.warning("user_data_flushed", user_id=user_id, keys_deleted=len(keys))
