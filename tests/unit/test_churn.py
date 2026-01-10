"""Unit tests for churn detection (AfterImage pattern)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from cortex.algorithms import (
    ChurnAnalysis,
    analyze_churn,
    calculate_retrieval_score_with_churn,
    get_high_churn_memories,
    get_identity_promotion_candidates,
)
from cortex.models import Memory, MemoryType


class TestAnalyzeChurn:
    """Test churn analysis."""

    @pytest.fixture
    def low_access_memory(self) -> Memory:
        """Memory with low access count."""
        return Memory(
            content="Low access memory",
            user_id="user1",
            access_count=5,
            memory_type=MemoryType.EPISODIC,
        )

    @pytest.fixture
    def high_access_memory(self) -> Memory:
        """Memory with high access count."""
        return Memory(
            content="High access memory",
            user_id="user1",
            access_count=15,
            memory_type=MemoryType.EPISODIC,
        )

    @pytest.fixture
    def very_high_access_memory(self) -> Memory:
        """Memory with very high access count."""
        return Memory(
            content="Very high access memory",
            user_id="user1",
            access_count=30,
            memory_type=MemoryType.EPISODIC,
        )

    def test_low_access_not_churn(self, low_access_memory: Memory):
        """Test that low access memories are not flagged as churn."""
        result = analyze_churn(low_access_memory, churn_threshold=10)

        assert result.is_high_churn is False
        assert result.importance_boost == 0.0
        assert result.should_promote_to_identity is False

    def test_high_access_is_churn(self, high_access_memory: Memory):
        """Test that high access memories are flagged as churn."""
        result = analyze_churn(high_access_memory, churn_threshold=10)

        assert result.is_high_churn is True
        assert result.importance_boost > 0.0
        assert result.should_promote_to_identity is False

    def test_very_high_access_promotes(self, very_high_access_memory: Memory):
        """Test that very high access suggests identity promotion."""
        result = analyze_churn(
            very_high_access_memory,
            churn_threshold=10,
            identity_threshold=25,
        )

        assert result.is_high_churn is True
        assert result.should_promote_to_identity is True

    def test_identity_not_promoted(self):
        """Test that identity memories are not promoted again."""
        memory = Memory(
            content="Already identity",
            user_id="user1",
            access_count=100,
            memory_type=MemoryType.IDENTITY,
        )

        result = analyze_churn(memory, identity_threshold=25)

        assert result.should_promote_to_identity is False

    def test_boost_scales_logarithmically(self):
        """Test that boost scales with access count."""
        mem1 = Memory(content="Test", user_id="user1", access_count=15)
        mem2 = Memory(content="Test", user_id="user1", access_count=100)

        result1 = analyze_churn(mem1, churn_threshold=10, importance_boost=0.2)
        result2 = analyze_churn(mem2, churn_threshold=10, importance_boost=0.2)

        # Higher access should have higher boost
        assert result2.importance_boost > result1.importance_boost

    def test_boost_capped_at_half(self):
        """Test that boost is capped at 0.5."""
        memory = Memory(
            content="Very high access",
            user_id="user1",
            access_count=10000,
        )

        result = analyze_churn(memory, churn_threshold=10, importance_boost=0.2)

        assert result.importance_boost <= 0.5

    def test_result_contains_memory_id(self, high_access_memory: Memory):
        """Test that result includes memory ID."""
        result = analyze_churn(high_access_memory)

        assert result.memory_id == high_access_memory.id

    def test_result_contains_access_count(self, high_access_memory: Memory):
        """Test that result includes access count."""
        result = analyze_churn(high_access_memory)

        assert result.access_count == high_access_memory.access_count


class TestCalculateRetrievalScoreWithChurn:
    """Test retrieval scoring with churn boost."""

    @pytest.fixture
    def memory(self) -> Memory:
        """Create test memory."""
        return Memory(
            content="Test memory",
            user_id="user1",
            emotional_score=0.5,
            access_count=20,
            created_at=datetime.utcnow() - timedelta(hours=1),
        )

    def test_includes_churn_boost(self, memory: Memory):
        """Test that score includes churn boost."""
        now = datetime.utcnow()

        score_with_churn = calculate_retrieval_score_with_churn(
            memory,
            similarity=0.8,
            current_time=now,
            churn_threshold=10,
            churn_boost=0.2,
        )

        # Score should be higher due to churn boost
        # Base score is roughly 0.4*0.8 + 0.25*recency + 0.2*0.5 + 0.15*reinforcement
        assert score_with_churn > 0.5

    def test_no_boost_for_low_access(self):
        """Test no boost for low access memories."""
        memory = Memory(
            content="Low access",
            user_id="user1",
            access_count=5,
            created_at=datetime.utcnow(),
        )

        now = datetime.utcnow()

        score = calculate_retrieval_score_with_churn(
            memory,
            similarity=0.8,
            current_time=now,
            churn_threshold=10,
        )

        # Should be similar to base score (no churn boost)
        assert score > 0
        assert score < 1.0

    def test_high_churn_scores_higher(self):
        """Test that high churn memories score higher."""
        now = datetime.utcnow()
        base_memory = Memory(
            content="Test",
            user_id="user1",
            access_count=5,
            created_at=now,
        )
        churn_memory = Memory(
            content="Test",
            user_id="user1",
            access_count=50,
            created_at=now,
        )

        score_low = calculate_retrieval_score_with_churn(
            base_memory,
            similarity=0.8,
            current_time=now,
            churn_threshold=10,
        )
        score_high = calculate_retrieval_score_with_churn(
            churn_memory,
            similarity=0.8,
            current_time=now,
            churn_threshold=10,
        )

        assert score_high > score_low


class TestGetHighChurnMemories:
    """Test filtering for high churn memories."""

    @pytest.fixture
    def mixed_memories(self) -> list[Memory]:
        """Create memories with varying access counts."""
        return [
            Memory(content="Low 1", user_id="user1", access_count=2),
            Memory(content="High 1", user_id="user1", access_count=15),
            Memory(content="Low 2", user_id="user1", access_count=5),
            Memory(content="High 2", user_id="user1", access_count=25),
            Memory(content="Medium", user_id="user1", access_count=10),
        ]

    def test_filters_high_churn(self, mixed_memories: list[Memory]):
        """Test that only high churn memories are returned."""
        results = get_high_churn_memories(mixed_memories, churn_threshold=10)

        # Should have 3: access counts 15, 25, and 10 (at threshold)
        assert len(results) == 3

        for mem, analysis in results:
            assert analysis.is_high_churn is True

    def test_sorted_by_access_count(self, mixed_memories: list[Memory]):
        """Test results are sorted by access count descending."""
        results = get_high_churn_memories(mixed_memories, churn_threshold=10)

        access_counts = [analysis.access_count for _, analysis in results]
        assert access_counts == sorted(access_counts, reverse=True)

    def test_empty_for_low_threshold(self, mixed_memories: list[Memory]):
        """Test no results when threshold is very high."""
        results = get_high_churn_memories(mixed_memories, churn_threshold=100)

        assert len(results) == 0


class TestGetIdentityPromotionCandidates:
    """Test identity promotion candidates."""

    @pytest.fixture
    def promotion_candidates(self) -> list[Memory]:
        """Create memories with varying access counts and types."""
        return [
            Memory(
                content="Should promote",
                user_id="user1",
                access_count=30,
                memory_type=MemoryType.EPISODIC,
            ),
            Memory(
                content="Already identity",
                user_id="user1",
                access_count=50,
                memory_type=MemoryType.IDENTITY,
            ),
            Memory(
                content="Not enough access",
                user_id="user1",
                access_count=10,
                memory_type=MemoryType.EPISODIC,
            ),
            Memory(
                content="Should also promote",
                user_id="user1",
                access_count=40,
                memory_type=MemoryType.SEMANTIC,
            ),
        ]

    def test_finds_promotion_candidates(self, promotion_candidates: list[Memory]):
        """Test finding promotion candidates."""
        results = get_identity_promotion_candidates(
            promotion_candidates,
            identity_threshold=25,
        )

        # Should find 2: the episodic and semantic with high access
        assert len(results) == 2

        for mem, analysis in results:
            assert analysis.should_promote_to_identity is True
            assert mem.memory_type != MemoryType.IDENTITY

    def test_excludes_identity_type(self, promotion_candidates: list[Memory]):
        """Test that identity types are excluded."""
        results = get_identity_promotion_candidates(
            promotion_candidates,
            identity_threshold=25,
        )

        for mem, _ in results:
            assert mem.memory_type != MemoryType.IDENTITY

    def test_respects_threshold(self, promotion_candidates: list[Memory]):
        """Test threshold is respected."""
        results_low = get_identity_promotion_candidates(
            promotion_candidates,
            identity_threshold=15,
        )
        results_high = get_identity_promotion_candidates(
            promotion_candidates,
            identity_threshold=35,
        )

        # Lower threshold should find more
        assert len(results_low) > len(results_high)
