"""Tests for core algorithms."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from cortex.algorithms import (
    calculate_importance,
    calculate_retrieval_score,
    calculate_ttl,
)
from cortex.models import Memory, MemoryType


class TestTTLCalculation:
    """Tests for TTL calculation."""

    def test_low_emotion_short_ttl(self):
        ttl = calculate_ttl(0.0)
        assert ttl == 1800  # 30 minutes

    def test_high_emotion_long_ttl(self):
        ttl = calculate_ttl(1.0)
        assert ttl == 21600  # 6 hours

    def test_mid_emotion_mid_ttl(self):
        ttl = calculate_ttl(0.5)
        assert 10000 < ttl < 13000  # ~3.25 hours

    def test_custom_base_ttl(self):
        ttl = calculate_ttl(0.0, base_ttl=3600)
        assert ttl == 3600

    def test_emotion_out_of_range_clamped(self):
        # Should still work with values outside 0-1
        ttl_negative = calculate_ttl(-0.5)
        ttl_over = calculate_ttl(1.5)
        assert ttl_negative > 0
        assert ttl_over > 0


class TestRetrievalScoring:
    """Tests for retrieval scoring."""

    def test_high_similarity_high_score(self):
        memory = Memory(
            content="test",
            user_id="test",
            emotional_score=0.5,
            access_count=0,
        )
        score = calculate_retrieval_score(memory, similarity=0.9, current_time=datetime.utcnow())
        assert score > 0.5

    def test_recent_memory_higher_score(self):
        now = datetime.utcnow()

        recent_memory = Memory(
            content="recent",
            user_id="test",
            emotional_score=0.5,
            created_at=now,
        )

        old_memory = Memory(
            content="old",
            user_id="test",
            emotional_score=0.5,
            created_at=now - timedelta(days=30),
        )

        recent_score = calculate_retrieval_score(recent_memory, 0.5, now)
        old_score = calculate_retrieval_score(old_memory, 0.5, now)

        assert recent_score > old_score

    def test_high_access_count_boosts_score(self):
        now = datetime.utcnow()

        accessed_memory = Memory(
            content="accessed",
            user_id="test",
            emotional_score=0.5,
            access_count=100,
        )

        fresh_memory = Memory(
            content="fresh",
            user_id="test",
            emotional_score=0.5,
            access_count=0,
        )

        accessed_score = calculate_retrieval_score(accessed_memory, 0.5, now)
        fresh_score = calculate_retrieval_score(fresh_memory, 0.5, now)

        assert accessed_score > fresh_score

    def test_custom_weights(self):
        memory = Memory(
            content="test",
            user_id="test",
            emotional_score=1.0,
        )

        # All weight on emotion
        score = calculate_retrieval_score(
            memory,
            similarity=0.0,
            current_time=datetime.utcnow(),
            weights={"similarity": 0, "recency": 0, "emotion": 1.0, "reinforcement": 0},
        )

        assert score == 1.0


class TestImportanceScoring:
    """Tests for importance scoring."""

    def test_identity_type_high_importance(self):
        identity_memory = Memory(
            content="name is Josh",
            user_id="test",
            memory_type=MemoryType.IDENTITY,
            emotional_score=0.5,
        )

        episodic_memory = Memory(
            content="had lunch",
            user_id="test",
            memory_type=MemoryType.EPISODIC,
            emotional_score=0.5,
        )

        identity_importance = calculate_importance(identity_memory)
        episodic_importance = calculate_importance(episodic_memory)

        assert identity_importance > episodic_importance

    def test_high_emotion_high_importance(self):
        high_emotion = Memory(
            content="test",
            user_id="test",
            emotional_score=1.0,
        )

        low_emotion = Memory(
            content="test",
            user_id="test",
            emotional_score=0.0,
        )

        assert calculate_importance(high_emotion) > calculate_importance(low_emotion)
