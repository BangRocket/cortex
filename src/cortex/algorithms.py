"""Core algorithms for Cortex memory system."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from cortex.models import Memory, MemoryType


@dataclass
class ChurnAnalysis:
    """Result of churn analysis for a memory."""

    memory_id: str
    access_count: int
    is_high_churn: bool
    importance_boost: float
    should_promote_to_identity: bool


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


# ==================== CHURN DETECTION (AfterImage pattern) ====================


def analyze_churn(
    memory: Memory,
    churn_threshold: int = 10,
    importance_boost: float = 0.2,
    identity_threshold: int = 25,
) -> ChurnAnalysis:
    """
    Analyze memory access patterns for churn detection.

    High-churn memories (frequently accessed) are likely important
    and should receive an importance boost. Very high-churn memories
    might be candidates for promotion to identity-level facts.

    Args:
        memory: The memory to analyze
        churn_threshold: Access count to consider "high churn"
        importance_boost: Importance boost for high-churn memories
        identity_threshold: Access count threshold for identity promotion

    Returns:
        ChurnAnalysis with recommendations
    """
    is_high_churn = memory.access_count >= churn_threshold
    should_promote = (
        memory.access_count >= identity_threshold
        and memory.memory_type != MemoryType.IDENTITY
    )

    # Calculate boost: scales logarithmically with access count
    if is_high_churn:
        # log(access_count / threshold) gives scaling factor
        scale = math.log(1 + memory.access_count / churn_threshold)
        boost = min(importance_boost * scale, 0.5)  # Cap at 0.5
    else:
        boost = 0.0

    return ChurnAnalysis(
        memory_id=memory.id or "",
        access_count=memory.access_count,
        is_high_churn=is_high_churn,
        importance_boost=boost,
        should_promote_to_identity=should_promote,
    )


def calculate_retrieval_score_with_churn(
    memory: Memory,
    similarity: float,
    current_time: datetime,
    weights: dict[str, float] | None = None,
    churn_threshold: int = 10,
    churn_boost: float = 0.2,
) -> float:
    """
    Calculate retrieval score with churn-based boost.

    Extends calculate_retrieval_score with AfterImage-inspired
    churn detection to boost frequently accessed memories.

    Args:
        memory: Memory to score
        similarity: Semantic similarity score
        current_time: Current time for recency calculation
        weights: Optional weight overrides
        churn_threshold: Access count for churn detection
        churn_boost: Maximum boost for high-churn memories

    Returns:
        Final retrieval score (0-1 range, but can exceed 1 with boost)
    """
    base_score = calculate_retrieval_score(memory, similarity, current_time, weights)

    # Apply churn boost
    churn = analyze_churn(memory, churn_threshold, churn_boost)
    boosted_score = base_score + churn.importance_boost

    return boosted_score


def get_high_churn_memories(
    memories: list[Memory],
    churn_threshold: int = 10,
) -> list[tuple[Memory, ChurnAnalysis]]:
    """
    Filter memories to find high-churn candidates.

    Useful for consolidation jobs to identify memories
    that should be reviewed for importance updates.

    Returns:
        List of (memory, analysis) tuples for high-churn memories
    """
    results = []
    for mem in memories:
        analysis = analyze_churn(mem, churn_threshold)
        if analysis.is_high_churn:
            results.append((mem, analysis))

    # Sort by access count descending
    results.sort(key=lambda x: x[1].access_count, reverse=True)
    return results


def get_identity_promotion_candidates(
    memories: list[Memory],
    identity_threshold: int = 25,
) -> list[tuple[Memory, ChurnAnalysis]]:
    """
    Find memories that should be promoted to identity type.

    These are memories accessed so frequently they likely represent
    core facts about the user.

    Returns:
        List of (memory, analysis) tuples for promotion candidates
    """
    results = []
    for mem in memories:
        analysis = analyze_churn(mem, identity_threshold=identity_threshold)
        if analysis.should_promote_to_identity:
            results.append((mem, analysis))

    return results
