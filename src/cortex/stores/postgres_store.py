"""Postgres store for warm/persistent memory layer with pgvector."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import asyncpg
import structlog
from pgvector.asyncpg import register_vector

from cortex.config import PostgresConfig
from cortex.models import (
    Contradiction,
    ConsolidationLog,
    Memory,
    MemoryStatus,
    MemoryType,
)

logger = structlog.get_logger(__name__)


class PostgresStore:
    """
    Postgres store for warm memory layer.

    Uses pgvector for semantic search over embeddings.
    """

    def __init__(self, config: PostgresConfig) -> None:
        self.config = config
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Initialize Postgres connection pool."""
        self.pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            min_size=self.config.min_pool_size,
            max_size=self.config.max_pool_size,
        )

        # Register pgvector type
        async with self.pool.acquire() as conn:
            await register_vector(conn)

        logger.info("postgres_connected", host=self.config.host, database=self.config.database)

    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("postgres_disconnected")

    async def initialize_schema(self) -> None:
        """Create tables if they don't exist."""
        assert self.pool is not None

        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Memories table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS memories (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id         VARCHAR(255) NOT NULL,
                    project_id      VARCHAR(255),
                    content         TEXT NOT NULL,
                    memory_type     VARCHAR(50) NOT NULL,
                    emotional_score FLOAT DEFAULT 0.5,
                    importance      FLOAT DEFAULT 0.5,
                    confidence      FLOAT DEFAULT 1.0,
                    created_at      TIMESTAMPTZ DEFAULT NOW(),
                    updated_at      TIMESTAMPTZ DEFAULT NOW(),
                    last_accessed   TIMESTAMPTZ,
                    access_count    INTEGER DEFAULT 0,
                    supersedes      UUID REFERENCES memories(id),
                    source          VARCHAR(100) DEFAULT 'conversation',
                    tags            TEXT[] DEFAULT ARRAY[]::TEXT[],
                    metadata        JSONB DEFAULT '{{}}',
                    embedding       vector({self.config.vector_dimensions}),
                    status          VARCHAR(50) DEFAULT 'active'
                )
            """)

            # Indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
                CREATE INDEX IF NOT EXISTS idx_memories_user_type ON memories(user_id, memory_type);
                CREATE INDEX IF NOT EXISTS idx_memories_user_status ON memories(user_id, status);
                CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id) WHERE project_id IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
            """)

            # Vector index (IVFFlat for performance)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_embedding
                ON memories USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)

            # Projects table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id              VARCHAR(255) PRIMARY KEY,
                    name            VARCHAR(255) NOT NULL,
                    owner_id        VARCHAR(255) NOT NULL,
                    created_at      TIMESTAMPTZ DEFAULT NOW(),
                    updated_at      TIMESTAMPTZ DEFAULT NOW(),
                    settings        JSONB DEFAULT '{}'
                )
            """)

            # Project members
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS project_members (
                    project_id      VARCHAR(255) REFERENCES projects(id) ON DELETE CASCADE,
                    user_id         VARCHAR(255) NOT NULL,
                    role            VARCHAR(50) DEFAULT 'member',
                    joined_at       TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (project_id, user_id)
                )
            """)

            # Contradictions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS contradictions (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id         VARCHAR(255) NOT NULL,
                    memory_a        UUID REFERENCES memories(id),
                    memory_b        UUID REFERENCES memories(id),
                    reason          TEXT NOT NULL,
                    resolution      VARCHAR(50),
                    resolution_note TEXT,
                    created_at      TIMESTAMPTZ DEFAULT NOW(),
                    resolved_at     TIMESTAMPTZ
                )
            """)

            # Consolidation logs
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_logs (
                    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id             VARCHAR(255) NOT NULL,
                    started_at          TIMESTAMPTZ NOT NULL,
                    completed_at        TIMESTAMPTZ,
                    memories_processed  INTEGER DEFAULT 0,
                    patterns_found      INTEGER DEFAULT 0,
                    identities_updated  INTEGER DEFAULT 0,
                    contradictions_found INTEGER DEFAULT 0,
                    memories_compacted  INTEGER DEFAULT 0,
                    success             BOOLEAN DEFAULT FALSE,
                    error               TEXT,
                    duration_ms         INTEGER DEFAULT 0
                )
            """)

        logger.info("postgres_schema_initialized")

    def _row_to_memory(self, row: asyncpg.Record) -> Memory:
        """Convert database row to Memory object."""
        return Memory(
            id=str(row["id"]),
            user_id=row["user_id"],
            project_id=row["project_id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            emotional_score=row["emotional_score"],
            importance=row["importance"],
            confidence=row["confidence"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            supersedes=str(row["supersedes"]) if row["supersedes"] else None,
            source=row["source"],
            tags=list(row["tags"]) if row["tags"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            embedding=list(row["embedding"]) if row["embedding"] else None,
            status=MemoryStatus(row["status"]),
        )

    # ==================== CORE CRUD ====================

    async def store(self, memory: Memory) -> str:
        """Store a memory and return its ID."""
        assert self.pool is not None

        query = """
            INSERT INTO memories (
                user_id, project_id, content, memory_type,
                emotional_score, importance, confidence,
                source, tags, metadata, embedding, status
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
            ) RETURNING id
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                memory.user_id,
                memory.project_id,
                memory.content,
                memory.memory_type.value,
                memory.emotional_score,
                memory.importance,
                memory.confidence,
                memory.source,
                memory.tags,
                json.dumps(memory.metadata),
                memory.embedding,
                memory.status.value,
            )
            memory_id = str(row["id"])
            logger.debug("memory_stored", memory_id=memory_id, user_id=memory.user_id)
            return memory_id

    async def get(self, memory_id: str) -> Memory | None:
        """Get a memory by ID."""
        assert self.pool is not None

        query = "SELECT * FROM memories WHERE id = $1"

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, memory_id)
            if row:
                return self._row_to_memory(row)
            return None

    async def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """Update specific fields of a memory."""
        assert self.pool is not None

        if not updates:
            return False

        # Build dynamic update query
        set_clauses = []
        params = []
        for i, (key, value) in enumerate(updates.items(), 1):
            if key == "metadata":
                value = json.dumps(value)
            set_clauses.append(f"{key} = ${i}")
            params.append(value)

        params.append(memory_id)
        query = f"""
            UPDATE memories
            SET {', '.join(set_clauses)}, updated_at = NOW()
            WHERE id = ${len(params)}
        """

        async with self.pool.acquire() as conn:
            result = await conn.execute(query, *params)
            return result == "UPDATE 1"

    async def update_status(self, memory_id: str, status: MemoryStatus) -> bool:
        """Update memory status."""
        return await self.update(memory_id, {"status": status.value})

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        assert self.pool is not None

        query = "DELETE FROM memories WHERE id = $1"
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, memory_id)
            return result == "DELETE 1"

    # ==================== SEARCH ====================

    async def search(
        self,
        user_id: str,
        embedding: list[float],
        project_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 20,
    ) -> list[Memory]:
        """Semantic search using pgvector."""
        assert self.pool is not None

        # Build query with filters
        conditions = ["user_id = $1", "status = 'active'"]
        params: list[Any] = [user_id]
        param_idx = 2

        if project_id:
            conditions.append(f"(project_id = ${param_idx} OR project_id IS NULL)")
            params.append(project_id)
            param_idx += 1

        if memory_types:
            type_values = [t.value for t in memory_types]
            conditions.append(f"memory_type = ANY(${param_idx})")
            params.append(type_values)
            param_idx += 1

        params.append(embedding)

        query = f"""
            SELECT *,
                   1 - (embedding <=> ${param_idx}) as similarity
            FROM memories
            WHERE {' AND '.join(conditions)}
                AND embedding IS NOT NULL
            ORDER BY embedding <=> ${param_idx}
            LIMIT {limit}
        """

        async with self.pool.acquire() as conn:
            await register_vector(conn)
            rows = await conn.fetch(query, *params)
            memories = []
            for row in rows:
                mem = self._row_to_memory(row)
                mem.metadata["similarity"] = row["similarity"]
                memories.append(mem)

            logger.debug("search_completed", user_id=user_id, results=len(memories))
            return memories

    async def get_memories(
        self,
        user_id: str | None = None,
        project_id: str | None = None,
        memory_type: MemoryType | None = None,
        memory_types: list[MemoryType] | None = None,
        since: datetime | None = None,
        before: datetime | None = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        limit: int = 100,
    ) -> list[Memory]:
        """Get memories with filters (no vector search)."""
        assert self.pool is not None

        conditions = ["status = $1"]
        params: list[Any] = [status.value]
        param_idx = 2

        if user_id:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        if project_id:
            conditions.append(f"project_id = ${param_idx}")
            params.append(project_id)
            param_idx += 1

        if memory_type:
            conditions.append(f"memory_type = ${param_idx}")
            params.append(memory_type.value)
            param_idx += 1

        if memory_types:
            conditions.append(f"memory_type = ANY(${param_idx})")
            params.append([t.value for t in memory_types])
            param_idx += 1

        if since:
            conditions.append(f"created_at >= ${param_idx}")
            params.append(since)
            param_idx += 1

        if before:
            conditions.append(f"created_at < ${param_idx}")
            params.append(before)
            param_idx += 1

        query = f"""
            SELECT * FROM memories
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
            LIMIT {limit}
        """

        async with self.pool.acquire() as conn:
            await register_vector(conn)
            rows = await conn.fetch(query, *params)
            return [self._row_to_memory(row) for row in rows]

    # ==================== ACCESS TRACKING ====================

    async def record_access(self, memory_id: str) -> None:
        """Record that a memory was accessed."""
        assert self.pool is not None

        query = """
            UPDATE memories
            SET access_count = access_count + 1,
                last_accessed = NOW()
            WHERE id = $1
        """
        async with self.pool.acquire() as conn:
            await conn.execute(query, memory_id)

    # ==================== PROJECT ====================

    async def create_project(self, project: dict[str, Any]) -> None:
        """Create a new project."""
        assert self.pool is not None

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO projects (id, name, owner_id, settings)
                VALUES ($1, $2, $3, $4)
                """,
                project["id"],
                project["name"],
                project["owner_id"],
                json.dumps(project.get("settings", {})),
            )

            # Add owner as member
            await conn.execute(
                """
                INSERT INTO project_members (project_id, user_id, role)
                VALUES ($1, $2, 'owner')
                """,
                project["id"],
                project["owner_id"],
            )

            # Add other members
            for member in project.get("members", []):
                if member != project["owner_id"]:
                    await conn.execute(
                        """
                        INSERT INTO project_members (project_id, user_id, role)
                        VALUES ($1, $2, 'member')
                        ON CONFLICT DO NOTHING
                        """,
                        project["id"],
                        member,
                    )

    async def add_project_member(self, project_id: str, user_id: str) -> None:
        """Add a member to a project."""
        assert self.pool is not None

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO project_members (project_id, user_id, role)
                VALUES ($1, $2, 'member')
                ON CONFLICT DO NOTHING
                """,
                project_id,
                user_id,
            )

    async def get_project_members(self, project_id: str) -> list[str]:
        """Get members of a project."""
        assert self.pool is not None

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT user_id FROM project_members WHERE project_id = $1",
                project_id,
            )
            return [row["user_id"] for row in rows]

    # ==================== CONTRADICTIONS ====================

    async def flag_contradiction(self, user_id: str, contradiction: Contradiction) -> str:
        """Store a detected contradiction."""
        assert self.pool is not None

        query = """
            INSERT INTO contradictions (user_id, memory_a, memory_b, reason)
            VALUES ($1, $2, $3, $4)
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                user_id,
                contradiction.memory_a,
                contradiction.memory_b,
                contradiction.reason,
            )
            return str(row["id"])

    async def get_contradictions(
        self, user_id: str | None = None, resolution: str | None = None
    ) -> list[Contradiction]:
        """Get contradictions, optionally filtered."""
        assert self.pool is not None

        conditions = []
        params: list[Any] = []
        param_idx = 1

        if user_id:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        if resolution == "pending":
            conditions.append("resolution IS NULL")
        elif resolution:
            conditions.append(f"resolution = ${param_idx}")
            params.append(resolution)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM contradictions {where_clause} ORDER BY created_at DESC"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [
                Contradiction(
                    id=str(row["id"]),
                    memory_a=str(row["memory_a"]),
                    memory_b=str(row["memory_b"]),
                    reason=row["reason"],
                    resolution=row["resolution"],
                )
                for row in rows
            ]

    async def resolve_contradiction(
        self, contradiction_id: str, resolution: str, note: str | None = None
    ) -> None:
        """Resolve a contradiction."""
        assert self.pool is not None

        query = """
            UPDATE contradictions
            SET resolution = $1, resolution_note = $2, resolved_at = NOW()
            WHERE id = $3
        """
        async with self.pool.acquire() as conn:
            await conn.execute(query, resolution, note, contradiction_id)

    # ==================== CONSOLIDATION LOGS ====================

    async def log_consolidation(self, log: ConsolidationLog) -> str:
        """Log a consolidation run."""
        assert self.pool is not None

        query = """
            INSERT INTO consolidation_logs (
                user_id, started_at, completed_at,
                memories_processed, patterns_found, identities_updated,
                contradictions_found, memories_compacted,
                success, error, duration_ms
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                log.user_id,
                log.started_at,
                log.completed_at,
                log.memories_processed,
                log.patterns_found,
                log.identities_updated,
                log.contradictions_found,
                log.memories_compacted,
                log.success,
                log.error,
                log.duration_ms,
            )
            return str(row["id"])

    async def get_consolidation_logs(
        self, user_id: str | None = None, since: datetime | None = None
    ) -> list[ConsolidationLog]:
        """Get consolidation logs."""
        assert self.pool is not None

        conditions = []
        params: list[Any] = []
        param_idx = 1

        if user_id:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        if since:
            conditions.append(f"started_at >= ${param_idx}")
            params.append(since)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM consolidation_logs {where_clause} ORDER BY started_at DESC"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [
                ConsolidationLog(
                    user_id=row["user_id"],
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                    memories_processed=row["memories_processed"],
                    patterns_found=row["patterns_found"],
                    identities_updated=row["identities_updated"],
                    contradictions_found=row["contradictions_found"],
                    memories_compacted=row["memories_compacted"],
                    success=row["success"],
                    error=row["error"],
                    duration_ms=row["duration_ms"],
                )
                for row in rows
            ]

    # ==================== UTILITIES ====================

    async def ping(self) -> bool:
        """Check if Postgres is connected."""
        try:
            assert self.pool is not None
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def count_users(self) -> int:
        """Count distinct users."""
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM memories")

    async def count_memories(self, user_id: str | None = None) -> int:
        """Count memories, optionally for a specific user."""
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            if user_id:
                return await conn.fetchval(
                    "SELECT COUNT(*) FROM memories WHERE user_id = $1", user_id
                )
            return await conn.fetchval("SELECT COUNT(*) FROM memories")

    async def get_active_user_ids(self) -> list[str]:
        """Get list of user IDs with recent activity."""
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT user_id FROM memories
                WHERE created_at > NOW() - INTERVAL '7 days'
                """
            )
            return [row["user_id"] for row in rows]
