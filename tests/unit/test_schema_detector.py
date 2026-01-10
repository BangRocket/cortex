"""Unit tests for schema detection."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cortex.migration.schema_detector import SchemaDetector, SchemaInfo, SchemaType


class AsyncContextManager:
    """Helper for async context manager mocking."""

    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *args):
        pass


class TestSchemaType:
    """Test SchemaType enum."""

    def test_values(self):
        """Test enum values."""
        assert SchemaType.UNKNOWN.value == "unknown"
        assert SchemaType.MEM0.value == "mem0"
        assert SchemaType.CORTEX.value == "cortex"
        assert SchemaType.EMPTY.value == "empty"


class TestSchemaInfo:
    """Test SchemaInfo dataclass."""

    def test_defaults(self):
        """Test default values."""
        info = SchemaInfo(schema_type=SchemaType.UNKNOWN)

        assert info.has_memories_table is False
        assert info.has_pgvector is False
        assert info.memory_count == 0
        assert info.user_count == 0
        assert info.columns is None

    def test_with_values(self):
        """Test with values."""
        info = SchemaInfo(
            schema_type=SchemaType.MEM0,
            has_memories_table=True,
            has_pgvector=True,
            memory_count=100,
            user_count=5,
            columns=["id", "embedding", "metadata"],
            vector_dimensions=1536,
        )

        assert info.schema_type == SchemaType.MEM0
        assert info.memory_count == 100
        assert info.vector_dimensions == 1536


class TestSchemaDetector:
    """Test SchemaDetector."""

    @pytest.fixture
    def detector(self) -> SchemaDetector:
        """Create test detector."""
        return SchemaDetector()

    @pytest.fixture
    def mock_pool(self):
        """Create mock connection pool."""
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value = AsyncContextManager(conn)
        return pool, conn

    @pytest.mark.asyncio
    async def test_detect_empty_database(self, detector: SchemaDetector, mock_pool):
        """Test detecting empty database."""
        pool, conn = mock_pool

        # pgvector exists, no tables
        conn.fetchval.side_effect = [
            True,   # pgvector check
            False,  # memories table check
        ]

        result = await detector.detect(pool)

        assert result.schema_type == SchemaType.EMPTY
        assert result.has_pgvector is True
        assert result.has_memories_table is False

    @pytest.mark.asyncio
    async def test_detect_cortex_schema(self, detector: SchemaDetector, mock_pool):
        """Test detecting Cortex schema."""
        pool, conn = mock_pool

        conn.fetchval.side_effect = [
            True,   # pgvector check
            True,   # memories table check
            100,    # count rows
            5,      # count distinct users (cortex)
        ]

        # Return Cortex columns
        conn.fetch.return_value = [
            {"column_name": "id"},
            {"column_name": "user_id"},
            {"column_name": "content"},
            {"column_name": "memory_type"},
            {"column_name": "emotional_score"},
            {"column_name": "embedding"},
        ]

        result = await detector.detect(pool)

        assert result.schema_type == SchemaType.CORTEX
        assert result.has_memories_table is True
        assert result.memory_count == 100
        assert result.user_count == 5
        assert "content" in result.columns
        assert "user_id" in result.columns

    @pytest.mark.asyncio
    async def test_detect_mem0_schema(self, detector: SchemaDetector, mock_pool):
        """Test detecting mem0 schema."""
        pool, conn = mock_pool

        conn.fetchval.side_effect = [
            True,   # pgvector check
            True,   # memories table check
            50,     # count rows
            3,      # count distinct users (mem0)
            1536,   # vector dimensions
        ]

        # Return mem0 columns (no content, memory_type columns)
        conn.fetch.return_value = [
            {"column_name": "id"},
            {"column_name": "embedding"},
            {"column_name": "metadata"},
            {"column_name": "created_at"},
            {"column_name": "updated_at"},
        ]

        result = await detector.detect(pool)

        assert result.schema_type == SchemaType.MEM0
        assert result.has_memories_table is True
        assert result.memory_count == 50
        assert result.vector_dimensions == 1536
        assert "content" not in result.columns
        assert "metadata" in result.columns

    @pytest.mark.asyncio
    async def test_detect_unknown_schema(self, detector: SchemaDetector, mock_pool):
        """Test detecting unknown schema."""
        pool, conn = mock_pool

        conn.fetchval.side_effect = [
            True,   # pgvector check
            True,   # memories table check
        ]

        # Return columns that don't match either schema
        conn.fetch.return_value = [
            {"column_name": "id"},
            {"column_name": "data"},
        ]

        result = await detector.detect(pool)

        assert result.schema_type == SchemaType.UNKNOWN

    def test_is_cortex_schema(self, detector: SchemaDetector):
        """Test Cortex schema detection logic."""
        cortex_columns = ["id", "user_id", "content", "memory_type", "emotional_score", "embedding"]
        assert detector._is_cortex_schema(cortex_columns) is True

        # Missing required columns
        incomplete = ["id", "user_id", "content"]
        assert detector._is_cortex_schema(incomplete) is False

    def test_is_mem0_schema(self, detector: SchemaDetector):
        """Test mem0 schema detection logic."""
        mem0_columns = ["id", "embedding", "metadata", "created_at"]
        assert detector._is_mem0_schema(mem0_columns) is True

        # Has cortex columns - not mem0
        mixed = ["id", "embedding", "metadata", "content"]
        assert detector._is_mem0_schema(mixed) is False

        # Missing required columns
        incomplete = ["id", "metadata"]
        assert detector._is_mem0_schema(incomplete) is False

    @pytest.mark.asyncio
    async def test_get_mem0_stats(self, detector: SchemaDetector, mock_pool):
        """Test getting mem0 statistics."""
        pool, conn = mock_pool

        # Total count
        conn.fetchval.return_value = 100

        # Users and counts
        conn.fetch.return_value = [
            {"user_id": "user1", "count": 60},
            {"user_id": "user2", "count": 30},
            {"user_id": "unknown", "count": 10},
        ]

        # Embedding stats
        conn.fetchrow.side_effect = [
            {"has_embedding": 95, "missing_content": 5},
            {"oldest": "2024-01-01", "newest": "2024-06-01"},
        ]

        stats = await detector.get_mem0_stats(pool)

        assert stats["total_memories"] == 100
        assert "user1" in stats["users"]
        assert stats["memories_by_user"]["user1"] == 60
        assert stats["has_embeddings"] == 95
        assert stats["missing_content"] == 5
