"""Core algorithms for Cortex memory system."""

from __future__ import annotations

import math
from datetime import datetime

from cortex.models import Memory, MemoryType


def calculate_ttl(emotional_score: float, base_ttl: int = 1800) -> int:
    """
    Calculate TTL in seconds based on emotional score.

    Base: 30 minutes (1800 seconds)
    Range: 30 minutes (emotion=0) to 6 hours (emotion=1)

    Formula: base_ttl * (1 + emotional_score * 11)
    - emotion=0.0 -> 1800 * 1.0 = 30 min
    - emotion=0.5 -> 1800 * 6.5 = 3.25 hours
    - emotion=1.0 -> 1800 * 12 = 6 hours
    """
    multiplier = 1 + (emotional_score * 11)
    return int(base_ttl * multiplier)


def calculate_retrieval_score(
    memory: Memory,
    similarity: float,
    current_time: datetime,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Combine signals for final retrieval ranking.

    Weights (default, tunable):
    - Similarity: 0.40 (semantic relevance)
    - Recency: 0.25 (temporal decay)
    - Emotion: 0.20 (emotional importance)
    - Reinforcement: 0.15 (access patterns)
    """
    if weights is None:
        weights = {
            "similarity": 0.40,
            "recency": 0.25,
            "emotion": 0.20,
            "reinforcement": 0.15,
        }

    # Recency decay: 1/(1 + log(1 + hours_old))
    age_hours = (current_time - memory.created_at).total_seconds() / 3600
    recency_score = 1 / (1 + math.log(1 + age_hours))

    # Reinforcement: log(1 + access_count) normalized
    reinforcement_score = math.log(1 + memory.access_count) / 10
    reinforcement_score = min(1.0, reinforcement_score)

    # Weighted combination
    final_score = (
        weights["similarity"] * similarity
        + weights["recency"] * recency_score
        + weights["emotion"] * memory.emotional_score
        + weights["reinforcement"] * reinforcement_score
    )

    return final_score


def calculate_importance(memory: Memory) -> float:
    """
    Calculate overall importance for a memory.

    Factors:
    - Emotional score (40%)
    - Recency (20%)
    - Access patterns (20%)
    - Type weight (20%)
    """
    type_weights = {
        MemoryType.IDENTITY: 1.0,
        MemoryType.SEMANTIC: 0.8,
        MemoryType.EPISODIC: 0.6,
        MemoryType.WORKING: 0.4,
        MemoryType.SESSION: 0.3,
        MemoryType.PROJECT: 0.7,
    }

    type_score = type_weights.get(memory.memory_type, 0.5)

    age_hours = (datetime.utcnow() - memory.created_at).total_seconds() / 3600
    recency_score = 1 / (1 + math.log(1 + age_hours))

    access_score = min(1.0, math.log(1 + memory.access_count) / 5)

    importance = (
        0.40 * memory.emotional_score
        + 0.20 * recency_score
        + 0.20 * access_score
        + 0.20 * type_score
    )

    return importance
