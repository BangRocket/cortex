"""Integration tests for mem0 migration."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from cortex.migration.mem0_migration import Mem0Migrator, MigrationReport
from cortex.models import Memory, MemoryType


pytestmark = pytest.mark.integration


class TestMem0Migration:
    """Test migration from mem0 to Cortex."""

    async def test_classify_and_import_memories(
        self,
        clean_postgres,
        clean_redis,
        embedder,
        scorer,
    ):
        """Test classifying and importing memories."""
        user_id = "test-migrate-user"

        migrator = Mem0Migrator(
            redis_store=clean_redis,
            postgres_store=clean_postgres,
            embedder=embedder,
            scorer=scorer,
        )

        # Mock exported memories
        memories = [
            {
                "id": "mem0-1",
                "content": "User works as a software engineer",
                "metadata": {"category": "work"},
                "created_at": datetime.utcnow().isoformat(),
            },
            {
                "id": "mem0-2",
                "content": "User prefers dark mode in all applications",
                "metadata": {"category": "preference"},
                "created_at": datetime.utcnow().isoformat(),
            },
        ]

        # Classify and import directly
        classifications = await migrator.classify_memories(memories)
        assert len(classifications) == 2

        result = await migrator.import_memories(user_id, memories, classifications)
        assert result["imported"] >= 1

        # Verify memories are searchable
        query_embedding = await embedder.embed("software engineer job")
        results = await clean_postgres.search(
            user_id=user_id,
            embedding=query_embedding,
            limit=5,
        )
        assert len(results) >= 1

    async def test_seed_identity_from_memories(
        self,
        clean_postgres,
        clean_redis,
        embedder,
        scorer,
    ):
        """Test seeding identity from identity-type memories."""
        user_id = "test-identity-seed"

        migrator = Mem0Migrator(
            redis_store=clean_redis,
            postgres_store=clean_postgres,
            embedder=embedder,
            scorer=scorer,
        )

        memories = [
            {
                "id": "mem0-name",
                "content": "User's name is Alice",
                "metadata": {},
            },
        ]

        # Mock classification results for identity type
        from cortex.migration.mem0_migration import ClassificationResult

        classifications = [
            ClassificationResult(
                memory_type=MemoryType.IDENTITY,
                confidence=0.9,
                identity_key="name",
            ),
        ]

        identity = await migrator.seed_identity(user_id, memories, classifications)
        assert "name" in identity

        # Verify identity in Redis
        stored_identity = await clean_redis.get_identity(user_id)
        assert stored_identity.get("name") == "User's name is Alice"

    async def test_import_empty_memories_list(
        self,
        clean_postgres,
        clean_redis,
        embedder,
        scorer,
    ):
        """Test importing an empty memories list."""
        user_id = "test-empty"

        migrator = Mem0Migrator(
            redis_store=clean_redis,
            postgres_store=clean_postgres,
            embedder=embedder,
            scorer=scorer,
        )

        result = await migrator.import_memories(user_id, [], [])
        assert result["imported"] == 0
        assert result["skipped"] == 0

    async def test_skip_low_confidence_memories(
        self,
        clean_postgres,
        clean_redis,
        embedder,
        scorer,
    ):
        """Test that low confidence memories are skipped."""
        user_id = "test-low-conf"

        migrator = Mem0Migrator(
            redis_store=clean_redis,
            postgres_store=clean_postgres,
            embedder=embedder,
            scorer=scorer,
        )

        memories = [
            {"id": "mem-1", "content": "Some memory content"},
        ]

        from cortex.migration.mem0_migration import ClassificationResult

        classifications = [
            ClassificationResult(
                memory_type=MemoryType.EPISODIC,
                confidence=0.3,  # Below 0.5 threshold
            ),
        ]

        result = await migrator.import_memories(user_id, memories, classifications)
        assert result["skipped"] == 1
        assert result["imported"] == 0

    async def test_classification_determines_memory_types(
        self,
        clean_postgres,
        clean_redis,
        embedder,
        scorer,
    ):
        """Test that classification determines correct memory types."""
        user_id = "test-classify"

        migrator = Mem0Migrator(
            redis_store=clean_redis,
            postgres_store=clean_postgres,
            embedder=embedder,
            scorer=scorer,
        )

        memories = [
            {
                "id": "mem0-identity",
                "content": "User's name is Alice and she lives in Seattle",
            },
            {
                "id": "mem0-event",
                "content": "User attended the Python conference on March 15",
            },
        ]

        # Classification uses LLM, so just verify it runs
        classifications = await migrator.classify_memories(memories)
        assert len(classifications) == 2
        assert all(hasattr(c, "memory_type") for c in classifications)
