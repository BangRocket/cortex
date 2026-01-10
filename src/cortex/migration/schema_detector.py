"""Schema detection for mem0 to Cortex migration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import asyncpg

logger = structlog.get_logger(__name__)


class SchemaType(str, Enum):
    """Type of database schema detected."""

    UNKNOWN = "unknown"
    MEM0 = "mem0"
    CORTEX = "cortex"
    EMPTY = "empty"


@dataclass
class SchemaInfo:
    """Information about detected schema."""

    schema_type: SchemaType
    has_memories_table: bool = False
    has_pgvector: bool = False
    memory_count: int = 0
    user_count: int = 0
    columns: list[str] | None = None
    vector_dimensions: int | None = None


class SchemaDetector:
    """
    Detects whether a PostgreSQL database contains mem0 or Cortex schema.

    mem0 schema:
    - memories table with: id (text), embedding (vector), metadata (jsonb),
      created_at, updated_at
    - Memory content stored in metadata->>'memory'
    - User ID stored in metadata->>'user_id'

    Cortex schema:
    - memories table with: id (uuid), user_id, content, memory_type,
      emotional_score, importance, embedding, etc.
    - Dedicated columns for all fields
    """

    # Columns that indicate Cortex schema
    CORTEX_COLUMNS = {"content", "user_id", "memory_type", "emotional_score"}

    # Columns that indicate mem0 schema (minimal set)
    MEM0_COLUMNS = {"id", "embedding", "metadata"}

    async def detect(self, pool: asyncpg.Pool) -> SchemaInfo:
        """
        Detect the schema type of the connected database.

        Args:
            pool: asyncpg connection pool

        Returns:
            SchemaInfo with detected schema details
        """
        info = SchemaInfo(schema_type=SchemaType.UNKNOWN)

        async with pool.acquire() as conn:
            # Check if pgvector extension exists
            info.has_pgvector = await self._check_pgvector(conn)

            # Check if memories table exists
            info.has_memories_table = await self._table_exists(conn, "memories")

            if not info.has_memories_table:
                info.schema_type = SchemaType.EMPTY
                logger.info("schema_detected", type="empty", has_pgvector=info.has_pgvector)
                return info

            # Get column names
            info.columns = await self._get_columns(conn, "memories")

            # Detect schema type based on columns
            if self._is_cortex_schema(info.columns):
                info.schema_type = SchemaType.CORTEX
                info.memory_count = await self._count_rows(conn, "memories")
                info.user_count = await self._count_distinct_users_cortex(conn)
            elif self._is_mem0_schema(info.columns):
                info.schema_type = SchemaType.MEM0
                info.memory_count = await self._count_rows(conn, "memories")
                info.user_count = await self._count_distinct_users_mem0(conn)
                info.vector_dimensions = await self._get_vector_dimensions(conn)
            else:
                info.schema_type = SchemaType.UNKNOWN

            logger.info(
                "schema_detected",
                type=info.schema_type.value,
                memory_count=info.memory_count,
                user_count=info.user_count,
                columns=info.columns,
            )

        return info

    async def get_mem0_stats(self, pool: asyncpg.Pool) -> dict:
        """
        Get statistics about mem0 data for migration planning.

        Args:
            pool: asyncpg connection pool

        Returns:
            Dictionary with mem0 statistics
        """
        stats = {
            "total_memories": 0,
            "users": [],
            "memories_by_user": {},
            "has_embeddings": 0,
            "missing_content": 0,
            "date_range": {"oldest": None, "newest": None},
        }

        async with pool.acquire() as conn:
            # Total count
            stats["total_memories"] = await self._count_rows(conn, "memories")

            # Users and their memory counts
            rows = await conn.fetch("""
                SELECT
                    COALESCE(metadata->>'user_id', 'unknown') as user_id,
                    COUNT(*) as count
                FROM memories
                GROUP BY metadata->>'user_id'
                ORDER BY count DESC
            """)

            for row in rows:
                user_id = row["user_id"]
                stats["users"].append(user_id)
                stats["memories_by_user"][user_id] = row["count"]

            # Embedding stats
            result = await conn.fetchrow("""
                SELECT
                    COUNT(*) FILTER (WHERE embedding IS NOT NULL) as has_embedding,
                    COUNT(*) FILTER (WHERE metadata->>'memory' IS NULL OR metadata->>'memory' = '') as missing_content
                FROM memories
            """)
            if result:
                stats["has_embeddings"] = result["has_embedding"]
                stats["missing_content"] = result["missing_content"]

            # Date range
            result = await conn.fetchrow("""
                SELECT
                    MIN(created_at) as oldest,
                    MAX(created_at) as newest
                FROM memories
            """)
            if result:
                stats["date_range"]["oldest"] = result["oldest"]
                stats["date_range"]["newest"] = result["newest"]

        return stats

    def _is_cortex_schema(self, columns: list[str]) -> bool:
        """Check if columns indicate Cortex schema."""
        return self.CORTEX_COLUMNS.issubset(set(columns))

    def _is_mem0_schema(self, columns: list[str]) -> bool:
        """Check if columns indicate mem0 schema."""
        column_set = set(columns)
        # Must have mem0 columns but NOT cortex-specific columns
        has_mem0 = self.MEM0_COLUMNS.issubset(column_set)
        has_cortex = "content" in column_set or "memory_type" in column_set
        return has_mem0 and not has_cortex

    async def _check_pgvector(self, conn: asyncpg.Connection) -> bool:
        """Check if pgvector extension is installed."""
        result = await conn.fetchval("""
            SELECT EXISTS(
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            )
        """)
        return bool(result)

    async def _table_exists(self, conn: asyncpg.Connection, table_name: str) -> bool:
        """Check if a table exists."""
        result = await conn.fetchval("""
            SELECT EXISTS(
                SELECT 1 FROM information_schema.tables
                WHERE table_name = $1
            )
        """, table_name)
        return bool(result)

    async def _get_columns(self, conn: asyncpg.Connection, table_name: str) -> list[str]:
        """Get column names for a table."""
        rows = await conn.fetch("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = $1
            ORDER BY ordinal_position
        """, table_name)
        return [row["column_name"] for row in rows]

    async def _count_rows(self, conn: asyncpg.Connection, table_name: str) -> int:
        """Count rows in a table."""
        # Use identifier quoting for safety
        result = await conn.fetchval(f'SELECT COUNT(*) FROM "{table_name}"')
        return result or 0

    async def _count_distinct_users_cortex(self, conn: asyncpg.Connection) -> int:
        """Count distinct users in Cortex schema."""
        result = await conn.fetchval("""
            SELECT COUNT(DISTINCT user_id) FROM memories
        """)
        return result or 0

    async def _count_distinct_users_mem0(self, conn: asyncpg.Connection) -> int:
        """Count distinct users in mem0 schema."""
        result = await conn.fetchval("""
            SELECT COUNT(DISTINCT metadata->>'user_id') FROM memories
        """)
        return result or 0

    async def _get_vector_dimensions(self, conn: asyncpg.Connection) -> int | None:
        """Get vector dimensions from first non-null embedding."""
        result = await conn.fetchval("""
            SELECT vector_dims(embedding)
            FROM memories
            WHERE embedding IS NOT NULL
            LIMIT 1
        """)
        return result


async def detect_schema(pool: asyncpg.Pool) -> SchemaInfo:
    """Convenience function to detect schema type."""
    detector = SchemaDetector()
    return await detector.detect(pool)
