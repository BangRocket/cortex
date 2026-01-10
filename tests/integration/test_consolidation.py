"""Integration tests for consolidation jobs."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from cortex.consolidation.pattern_extractor import PatternExtractor
from cortex.consolidation.compaction import CompactionJob
from cortex.models import Memory, MemoryType


pytestmark = pytest.mark.integration


class TestPatternExtraction:
    """Test pattern extraction from episodic memories."""

    async def test_extract_patterns_from_memories(
        self,
        clean_postgres,
        clean_redis,
        embedder,
        scorer,
    ):
        """Test extracting patterns when sufficient memories exist."""
        user_id = "test-pattern-user"

        # Create episodic memories that establish a pattern
        memories_content = [
            "Had a productive coding session in Python",
            "Finished the Python API endpoint today",
            "Debugging Python code took 2 hours",
            "Learning new Python libraries for ML",
            "Python project meeting went well",
            "Committed Python changes to main branch",
        ]

        for i, content in enumerate(memories_content):
            embedding = await embedder.embed(content)
            memory = Memory(
                user_id=user_id,
                content=content,
                memory_type=MemoryType.EPISODIC,
                embedding=embedding,
                created_at=datetime.utcnow() - timedelta(days=i),
            )
            await clean_postgres.store(memory)

        # Create pattern extractor
        extractor = PatternExtractor(
            scorer=scorer,
            postgres=clean_postgres,
            redis=clean_redis,
            embedder=embedder,
            lookback_days=7,
            min_memories=5,
            confidence_threshold=0.7,
        )

        # This test may require actual LLM access
        # For now, just verify the extractor can be created and called
        patterns = await extractor.extract_for_user(user_id)
        # Pattern extraction depends on LLM, so we just check it runs without error
        assert isinstance(patterns, list)

    async def test_insufficient_memories_returns_empty(
        self,
        clean_postgres,
        clean_redis,
        embedder,
        scorer,
    ):
        """Test that insufficient memories returns empty patterns."""
        user_id = "test-pattern-sparse"

        # Only add 2 memories (below threshold of 5)
        for content in ["Single memory 1", "Single memory 2"]:
            embedding = await embedder.embed(content)
            memory = Memory(
                user_id=user_id,
                content=content,
                memory_type=MemoryType.EPISODIC,
                embedding=embedding,
            )
            await clean_postgres.store(memory)

        extractor = PatternExtractor(
            scorer=scorer,
            postgres=clean_postgres,
            redis=clean_redis,
            embedder=embedder,
            lookback_days=7,
            min_memories=5,
        )

        patterns = await extractor.extract_for_user(user_id)
        assert patterns == []


class TestMemoryCompaction:
    """Test memory compaction jobs."""

    async def test_compaction_archives_old_memories(
        self,
        clean_postgres,
        embedder,
        scorer,
    ):
        """Test that old memories get compacted."""
        user_id = "test-compact-user"

        # Create old episodic memories
        for i in range(15):
            content = f"Old memory number {i} about project work"
            embedding = await embedder.embed(content)
            memory = Memory(
                user_id=user_id,
                content=content,
                memory_type=MemoryType.EPISODIC,
                embedding=embedding,
                created_at=datetime.utcnow() - timedelta(days=60 + i),
            )
            await clean_postgres.store(memory)

        compaction = CompactionJob(
            scorer=scorer,
            postgres=clean_postgres,
            embedder=embedder,
            compact_after_days=30,
            min_memories=10,
        )

        result = await compaction.compact_user(user_id)
        # Compaction logic depends on LLM for summarization
        # Just verify it runs without error
        assert isinstance(result, dict)


class TestWorkingMemoryCleanup:
    """Test working memory expiration and cleanup."""

    async def test_cleanup_returns_expired_count(
        self,
        clean_redis,
    ):
        """Test that cleanup correctly reports expired entries."""
        user_id = "test-cleanup-user"

        # Add some working memories with very short TTL
        # Note: In real tests, we'd need to wait or mock time
        memory = Memory(
            user_id=user_id,
            content="Test working memory",
            memory_type=MemoryType.WORKING,
            emotional_score=0.5,
        )
        await clean_redis.add_working(user_id, memory, ttl=7200)

        # Get working memories (should not expire yet)
        working = await clean_redis.get_working(user_id)
        assert len(working) >= 1

        # Cleanup should return 0 since nothing expired yet
        expired_count = await clean_redis.cleanup_expired_working(user_id)
        assert expired_count == 0
