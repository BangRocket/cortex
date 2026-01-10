"""Main MemoryManager class - primary interface for Cortex."""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING

import structlog

from cortex.algorithms import calculate_retrieval_score, calculate_ttl
from cortex.config import CortexConfig
from cortex.models import (
    Memory,
    MemoryContext,
    MemoryStatus,
    MemoryType,
    SearchResult,
    StoreResult,
)
from cortex.stores.postgres_store import PostgresStore
from cortex.stores.redis_store import RedisStore
from cortex.utils.embedder import create_embedder
from cortex.utils.scorer import EmotionScorer

if TYPE_CHECKING:
    from cortex.utils.embedder import Embedder

logger = structlog.get_logger(__name__)


class MemoryManager:
    """
    Main interface for Cortex memory system.

    Usage:
        config = CortexConfig()
        manager = MemoryManager(config)
        await manager.initialize()

        # Every turn
        context = await manager.get_context(user_id, query="current topic")

        # Store new memories
        await manager.store(user_id, "User mentioned they have a meeting tomorrow")

        # Update session
        await manager.update_session(user_id, {"current_topic": "scheduling"})
    """

    def __init__(self, config: CortexConfig) -> None:
        self.config = config
        self.redis = RedisStore(config.redis)
        self.postgres = PostgresStore(config.postgres)
        self.embedder: Embedder = create_embedder(config.embedding)
        self.scorer = EmotionScorer(config.llm)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize connections to Redis and Postgres."""
        await self.redis.connect()
        await self.postgres.connect()
        await self.postgres.initialize_schema()
        self._initialized = True
        logger.info("memory_manager_initialized")

    async def close(self) -> None:
        """Clean shutdown."""
        await self.redis.close()
        await self.postgres.close()
        logger.info("memory_manager_closed")

    # ==================== CONTEXT ====================

    async def get_context(
        self,
        user_id: str,
        query: str | None = None,
        project_id: str | None = None,
        include_working: bool = True,
        max_retrieved: int = 20,
    ) -> MemoryContext:
        """
        Get full context for a user. Called every turn.

        Args:
            user_id: User identifier
            query: Optional query for semantic retrieval
            project_id: Optional project scope
            include_working: Include working memory
            max_retrieved: Max memories to retrieve

        Returns:
            MemoryContext with identity, session, working, and retrieved memories
        """
        # Always load from Redis (fast)
        identity = await self.redis.get_identity(user_id)
        session = await self.redis.get_session(user_id)

        working: list[Memory] = []
        if include_working:
            working = await self.redis.get_working(user_id)

        # Semantic retrieval from Postgres (if query provided)
        retrieved: list[Memory] = []
        if query:
            embedding = await self.embedder.embed(query)
            retrieved = await self.postgres.search(
                user_id=user_id,
                embedding=embedding,
                project_id=project_id,
                limit=max_retrieved,
            )
            # Re-rank with full scoring
            retrieved = self._rerank(retrieved)

            # Record access for retrieved memories
            for mem in retrieved[:5]:  # Top 5 most relevant
                if mem.id:
                    await self.postgres.record_access(mem.id)

        # Project context
        project = None
        if project_id:
            project_meta = await self.redis.get_project_meta(project_id)
            if project_meta:
                project = project_meta

        logger.debug(
            "context_retrieved",
            user_id=user_id,
            has_identity=bool(identity),
            working_count=len(working),
            retrieved_count=len(retrieved),
        )

        return MemoryContext(
            user_id=user_id,
            identity=identity,
            session=session,
            working=working,
            retrieved=retrieved,
            project=project,
        )

    # ==================== STORE ====================

    async def store(
        self,
        user_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        project_id: str | None = None,
        emotional_score: float | None = None,
        metadata: dict | None = None,
    ) -> StoreResult:
        """
        Store a new memory. Write-through pattern.

        1. Score emotion (if not provided)
        2. Write to Redis immediately (working memory)
        3. Generate embedding
        4. Write to Postgres
        """
        try:
            # Score emotion if not provided and enabled
            if emotional_score is None and self.config.enable_emotional_scoring:
                emotional_score = await self.scorer.score(content)
            elif emotional_score is None:
                emotional_score = 0.5

            # Calculate TTL for working memory
            ttl = calculate_ttl(emotional_score, self.config.ttl.base_ttl)

            # Create memory object
            memory = Memory(
                content=content,
                user_id=user_id,
                memory_type=memory_type,
                project_id=project_id,
                emotional_score=emotional_score,
                metadata=metadata or {},
                source="conversation",
            )

            # 1. Write to Redis immediately
            await self.redis.add_working(user_id, memory, ttl=ttl)

            # 2. Generate embedding
            memory.embedding = await self.embedder.embed(content)

            # 3. Write to Postgres
            memory.id = await self.postgres.store(memory)

            # 4. Add to recent buffer
            if memory.id:
                await self.redis.add_recent(user_id, memory.id)

            logger.debug(
                "memory_stored",
                user_id=user_id,
                memory_id=memory.id,
                emotional_score=emotional_score,
                ttl=ttl,
            )

            return StoreResult(
                success=True,
                memory_id=memory.id,
                emotional_score=emotional_score,
                ttl=ttl,
            )

        except Exception as e:
            logger.error("memory_store_failed", error=str(e), user_id=user_id)
            return StoreResult(success=False, error=str(e))

    # ==================== SEARCH ====================

    async def search(
        self,
        user_id: str,
        query: str,
        project_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 20,
    ) -> SearchResult:
        """Semantic search over memories."""
        start = time.time()

        embedding = await self.embedder.embed(query)
        memories = await self.postgres.search(
            user_id=user_id,
            embedding=embedding,
            project_id=project_id,
            memory_types=memory_types,
            limit=limit,
        )

        # Re-rank
        memories = self._rerank(memories)

        elapsed_ms = int((time.time() - start) * 1000)

        logger.debug(
            "search_completed",
            user_id=user_id,
            results=len(memories),
            elapsed_ms=elapsed_ms,
        )

        return SearchResult(
            memories=memories,
            total_count=len(memories),
            search_time_ms=elapsed_ms,
        )

    def _rerank(self, memories: list[Memory]) -> list[Memory]:
        """Re-rank memories using full scoring algorithm."""
        now = datetime.utcnow()

        scored = []
        for mem in memories:
            similarity = mem.metadata.get("similarity", 0.5)
            score = calculate_retrieval_score(mem, similarity, now)
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored]

    # ==================== SESSION ====================

    async def update_session(self, user_id: str, updates: dict) -> bool:
        """Update session state in Redis."""
        return await self.redis.update_session(user_id, updates)

    async def start_session(
        self, user_id: str, initial_context: dict | None = None
    ) -> dict:
        """Start a new session, loading identity."""
        session = {
            "started_at": datetime.utcnow().isoformat(),
            "last_active": datetime.utcnow().isoformat(),
            **(initial_context or {}),
        }
        await self.redis.set_session(user_id, session)

        logger.info("session_started", user_id=user_id)
        return session

    async def end_session(self, user_id: str) -> None:
        """End a session. Persist important working memories."""
        # Get working memories before they expire
        working = await self.redis.get_working(user_id)

        # Persist high-emotion working memories to episodic
        persisted = 0
        for mem in working:
            if mem.emotional_score > 0.6:
                mem.memory_type = MemoryType.EPISODIC
                mem.embedding = await self.embedder.embed(mem.content)
                await self.postgres.store(mem)
                persisted += 1

        # Clear session
        await self.redis.clear_session(user_id)

        logger.info(
            "session_ended",
            user_id=user_id,
            working_memories=len(working),
            persisted=persisted,
        )

    # ==================== IDENTITY ====================

    async def update_identity(self, user_id: str, key: str, value: any) -> bool:
        """
        Update a core identity fact.
        Also stores to Postgres for persistence.
        """
        # Update Redis
        await self.redis.update_identity_field(user_id, key, value)

        # Store as identity memory in Postgres
        memory = Memory(
            user_id=user_id,
            content=f"{key}: {value}",
            memory_type=MemoryType.IDENTITY,
            emotional_score=0.7,  # Identity facts are important
            source="identity_update",
        )
        memory.embedding = await self.embedder.embed(memory.content)
        await self.postgres.store(memory)

        logger.info("identity_updated", user_id=user_id, key=key)
        return True

    async def get_identity(self, user_id: str) -> dict:
        """Get current identity facts."""
        return await self.redis.get_identity(user_id)

    async def set_identity(self, user_id: str, identity: dict) -> None:
        """Set full identity (overwrites existing)."""
        await self.redis.set_identity(user_id, identity)

        # Also store each fact to Postgres
        for key, value in identity.items():
            if key != "updated_at":
                memory = Memory(
                    user_id=user_id,
                    content=f"{key}: {value}",
                    memory_type=MemoryType.IDENTITY,
                    emotional_score=0.7,
                    source="identity_set",
                )
                memory.embedding = await self.embedder.embed(memory.content)
                await self.postgres.store(memory)

        logger.info("identity_set", user_id=user_id, fields=len(identity))

    # ==================== UTILITIES ====================

    async def health_check(self) -> dict:
        """Check health of all components."""
        return {
            "redis": await self.redis.ping(),
            "postgres": await self.postgres.ping(),
            "embedder": await self.embedder.health_check(),
        }

    async def get_stats(self, user_id: str | None = None) -> dict:
        """Get statistics."""
        stats = {
            "total_users": await self.postgres.count_users(),
            "total_memories": await self.postgres.count_memories(),
        }

        if user_id:
            stats["user_memories"] = await self.postgres.count_memories(user_id)
            stats["user_identity"] = bool(await self.redis.get_identity(user_id))
            stats["user_working"] = len(await self.redis.get_working(user_id))

        return stats
