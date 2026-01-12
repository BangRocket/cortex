"""Main MemoryManager class - primary interface for Cortex."""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING

import structlog

from cortex.algorithms import (
    calculate_retrieval_score_with_churn,
    calculate_ttl,
    get_identity_promotion_candidates,
)
from cortex.config import CortexConfig
from cortex.models import (
    Entity,
    GraphContext,
    Memory,
    MemoryContext,
    MemoryStatus,
    MemoryType,
    Relationship,
    SearchResult,
    StoreResult,
)
from cortex.stores.postgres_store import PostgresStore
from cortex.stores.redis_store import RedisStore
from cortex.utils.budget import TokenBudgetManager
from cortex.utils.cache import CortexCache
from cortex.utils.embedder import create_embedder
from cortex.utils.scorer import EmotionScorer

if TYPE_CHECKING:
    from cortex.stores.graph_store import GraphStore
    from cortex.utils.embedder import Embedder
    from cortex.utils.entity_extractor import EntityExtractor

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

        # AfterImage-inspired features
        self.budget_manager = TokenBudgetManager(config.token_budget)
        self.cache = CortexCache.get_instance(
            identity_size=config.cache.identity_cache_size,
            context_size=config.cache.context_cache_size,
            ttl=config.cache.cache_ttl,
        ) if config.cache.enabled else None

        # Graph memory (mem0-style)
        self.graph: GraphStore | None = None
        self.entity_extractor: EntityExtractor | None = None
        if config.enable_graph_memory and config.graph.enabled:
            from cortex.stores.graph_store import GraphStore
            from cortex.utils.entity_extractor import EntityExtractor
            self.graph = GraphStore(config.neo4j)
            self.entity_extractor = EntityExtractor(config.llm, config.graph)

    async def initialize(self) -> None:
        """Initialize connections to Redis, Postgres, and optionally Neo4j."""
        await self.redis.connect()
        await self.postgres.connect()
        await self.postgres.initialize_schema()

        # Initialize graph store if enabled
        if self.graph:
            await self.graph.connect()
            await self.graph.initialize_schema()

        self._initialized = True
        logger.info(
            "memory_manager_initialized",
            graph_enabled=self.graph is not None,
        )

    async def close(self) -> None:
        """Clean shutdown."""
        await self.redis.close()
        await self.postgres.close()
        if self.graph:
            await self.graph.close()
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
        # Try identity from cache first (AfterImage pattern)
        identity = None
        if self.cache:
            identity = self.cache.identity.get(user_id)

        if identity is None:
            # Cache miss - load from Redis
            identity = await self.redis.get_identity(user_id)
            if self.cache and identity:
                self.cache.identity.set(user_id, identity)

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
            # Re-rank with full scoring including churn boost
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
            cache_enabled=self.cache is not None,
        )

        return MemoryContext(
            user_id=user_id,
            identity=identity,
            session=session,
            working=working,
            retrieved=retrieved,
            project=project,
        )

    def get_budgeted_context_string(self, context: MemoryContext) -> str:
        """
        Get context formatted as string with token budget applied.

        This is an alternative to context.to_prompt_string() that
        applies AfterImage-style token budget management.

        Args:
            context: The MemoryContext to format

        Returns:
            Formatted string respecting token budget
        """
        return self.budget_manager.format_context_with_budget(
            identity=context.identity,
            session=context.session,
            working=context.working,
            retrieved=context.retrieved,
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

            # 5. Extract entities and relationships (graph memory)
            if self.graph and self.entity_extractor and memory.id:
                try:
                    extracted = await self.entity_extractor.extract(
                        content, user_id, memory.id
                    )
                    for entity in extracted.entities:
                        await self.graph.create_entity(entity)
                    for relationship in extracted.relationships:
                        await self.graph.create_relationship(relationship)
                except Exception as e:
                    logger.warning("entity_extraction_failed", error=str(e))

            logger.debug(
                "memory_stored",
                user_id=user_id,
                memory_id=memory.id,
                emotional_score=emotional_score,
                ttl=ttl,
                graph_enabled=self.graph is not None,
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
        """Re-rank memories using full scoring algorithm with churn boost."""
        from datetime import timezone
        now = datetime.now(timezone.utc)

        scored = []
        for mem in memories:
            similarity = mem.metadata.get("similarity", 0.5)
            # Use churn-aware scoring (AfterImage pattern)
            score = calculate_retrieval_score_with_churn(
                mem,
                similarity,
                now,
                churn_threshold=self.config.churn.churn_threshold,
                churn_boost=self.config.churn.importance_boost,
            )
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

        # Invalidate cache (AfterImage pattern)
        if self.cache:
            self.cache.identity.invalidate(user_id)

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
        # Try cache first
        if self.cache:
            cached = self.cache.identity.get(user_id)
            if cached is not None:
                return cached

        identity = await self.redis.get_identity(user_id)

        # Populate cache
        if self.cache and identity:
            self.cache.identity.set(user_id, identity)

        return identity

    async def set_identity(self, user_id: str, identity: dict) -> None:
        """Set full identity (overwrites existing)."""
        await self.redis.set_identity(user_id, identity)

        # Invalidate cache (AfterImage pattern)
        if self.cache:
            self.cache.identity.invalidate(user_id)

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

        # Include cache stats if enabled
        if self.cache:
            stats["cache"] = self.cache.stats

        return stats

    # ==================== CHURN MANAGEMENT (AfterImage pattern) ====================

    async def process_identity_promotions(self, user_id: str) -> int:
        """
        Check for high-churn memories that should be promoted to identity.

        This implements the AfterImage pattern of detecting "churn" -
        memories accessed so frequently they likely represent core facts.

        Returns:
            Number of memories promoted
        """
        # Get recent memories with high access counts
        memories = await self.postgres.get_memories(
            user_id=user_id,
            limit=100,
        )

        # Find promotion candidates
        candidates = get_identity_promotion_candidates(
            memories,
            identity_threshold=self.config.churn.identity_promotion_threshold,
        )

        promoted = 0
        for mem, analysis in candidates:
            # Update memory type to identity
            await self.postgres.update(
                mem.id,
                {"memory_type": MemoryType.IDENTITY.value},
            )

            # Extract key fact for Redis identity store
            # Simple heuristic: use content as value, generate key from type
            key = f"fact_{promoted + 1}"
            await self.redis.update_identity_field(user_id, key, mem.content)

            logger.info(
                "memory_promoted_to_identity",
                user_id=user_id,
                memory_id=mem.id,
                access_count=analysis.access_count,
            )
            promoted += 1

        # Invalidate cache after promotions
        if promoted > 0 and self.cache:
            self.cache.identity.invalidate(user_id)

        return promoted

    async def get_cache_stats(self) -> dict | None:
        """Get in-memory cache statistics."""
        if self.cache:
            return self.cache.stats
        return None

    async def cleanup_caches(self) -> dict:
        """Cleanup expired cache entries."""
        if self.cache:
            return self.cache.cleanup_expired()
        return {}

    # ==================== GRAPH MEMORY ====================

    async def get_graph_context(
        self,
        user_id: str,
        entity_names: list[str],
        max_depth: int | None = None,
        max_entities: int | None = None,
    ) -> GraphContext | None:
        """
        Get graph-based context for a set of entity names.

        Args:
            user_id: User identifier
            entity_names: List of entity names to find and expand
            max_depth: Max hops for relationship traversal
            max_entities: Max entities to return

        Returns:
            GraphContext with entities and relationships, or None if graph disabled
        """
        if not self.graph:
            return None

        return await self.graph.get_graph_context(
            user_id=user_id,
            entity_names=entity_names,
            max_depth=max_depth or self.config.graph.max_hop_depth,
            max_entities=max_entities or self.config.graph.max_related_entities,
        )

    async def search_entities(
        self,
        user_id: str,
        query: str,
        entity_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[Entity]:
        """
        Search for entities by name/description.

        Args:
            user_id: User identifier
            query: Search query
            entity_types: Optional filter by entity types
            limit: Max results

        Returns:
            List of matching entities
        """
        if not self.graph:
            return []

        return await self.graph.search_entities(
            user_id=user_id,
            query=query,
            entity_types=entity_types,
            limit=limit,
        )

    async def get_entity_relationships(
        self,
        entity_id: str,
        direction: str = "both",
    ) -> list[Relationship]:
        """
        Get relationships for an entity.

        Args:
            entity_id: Entity ID
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of relationships
        """
        if not self.graph:
            return []

        return await self.graph.get_relationships(
            entity_id=entity_id,
            direction=direction,
        )

    async def get_related_entities(
        self,
        entity_id: str,
        max_depth: int = 2,
        limit: int = 20,
    ) -> list[Entity]:
        """
        Get entities related to a given entity within N hops.

        Args:
            entity_id: Starting entity ID
            max_depth: Max relationship hops
            limit: Max entities to return

        Returns:
            List of related entities
        """
        if not self.graph:
            return []

        return await self.graph.get_related_entities(
            entity_id=entity_id,
            max_depth=max_depth,
            limit=limit,
        )

    async def find_entity_path(
        self,
        source_entity_id: str,
        target_entity_id: str,
        max_depth: int = 4,
    ) -> list[str] | None:
        """
        Find shortest path between two entities.

        Args:
            source_entity_id: Starting entity ID
            target_entity_id: Target entity ID
            max_depth: Max path length

        Returns:
            List of entity names in the path, or None if no path found
        """
        if not self.graph:
            return None

        return await self.graph.find_path(
            source_id=source_entity_id,
            target_id=target_entity_id,
            max_depth=max_depth,
        )

    async def get_user_entities(
        self,
        user_id: str,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[Entity]:
        """
        Get all entities for a user.

        Args:
            user_id: User identifier
            entity_type: Optional filter by type
            limit: Max results

        Returns:
            List of entities
        """
        if not self.graph:
            return []

        return await self.graph.get_user_entities(
            user_id=user_id,
            entity_type=entity_type,
            limit=limit,
        )

    async def get_graph_stats(self, user_id: str | None = None) -> dict:
        """
        Get graph memory statistics.

        Returns:
            Dict with entity and relationship counts
        """
        if not self.graph:
            return {"enabled": False}

        return {
            "enabled": True,
            "entity_count": await self.graph.count_entities(user_id),
            "relationship_count": await self.graph.count_relationships(user_id),
        }
