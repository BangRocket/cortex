"""Observability: Prometheus metrics and structured logging for Cortex."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator

import structlog

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

# Try to import prometheus_client, but make it optional
try:
    from prometheus_client import Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None  # type: ignore
    Gauge = None  # type: ignore
    Histogram = None  # type: ignore


# ==================== METRICS ====================


class CortexMetrics:
    """Prometheus metrics for Cortex memory system."""

    _instance: "CortexMetrics | None" = None

    def __init__(self) -> None:
        if not PROMETHEUS_AVAILABLE:
            logger.warning("prometheus_client not installed, metrics disabled")
            return

        # Memory operations
        self.memories_stored = Counter(
            "cortex_memories_stored_total",
            "Total memories stored",
            ["memory_type", "user_id"],
        )
        self.memories_retrieved = Counter(
            "cortex_memories_retrieved_total",
            "Total memories retrieved",
            ["user_id"],
        )
        self.retrieval_latency = Histogram(
            "cortex_retrieval_seconds",
            "Memory retrieval latency in seconds",
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )
        self.store_latency = Histogram(
            "cortex_store_seconds",
            "Memory store latency in seconds",
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        # Working memory
        self.working_memory_size = Gauge(
            "cortex_working_memory_size",
            "Current working memory entries",
            ["user_id"],
        )
        self.working_memory_expired = Counter(
            "cortex_working_memory_expired_total",
            "Working memory entries expired",
            ["user_id"],
        )

        # Consolidation
        self.consolidation_runs = Counter(
            "cortex_consolidation_runs_total",
            "Consolidation job runs",
            ["status"],  # success, failure
        )
        self.patterns_extracted = Counter(
            "cortex_patterns_extracted_total",
            "Patterns extracted from memories",
        )
        self.contradictions_detected = Counter(
            "cortex_contradictions_detected_total",
            "Contradictions detected between memories",
        )
        self.memories_compacted = Counter(
            "cortex_memories_compacted_total",
            "Memories compacted in consolidation",
        )

        # Connection health
        self.redis_connected = Gauge(
            "cortex_redis_connected",
            "Redis connection status (1=connected, 0=disconnected)",
        )
        self.postgres_connected = Gauge(
            "cortex_postgres_connected",
            "Postgres connection status (1=connected, 0=disconnected)",
        )

        # Embeddings
        self.embedding_requests = Counter(
            "cortex_embedding_requests_total",
            "Embedding API requests",
            ["provider"],  # openai, local
        )
        self.embedding_latency = Histogram(
            "cortex_embedding_seconds",
            "Embedding generation latency",
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
        )

        # LLM scoring
        self.scoring_requests = Counter(
            "cortex_scoring_requests_total",
            "LLM scoring requests",
        )

    @classmethod
    def get_instance(cls) -> "CortexMetrics":
        """Get singleton metrics instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# ==================== TIMING CONTEXT MANAGERS ====================


@asynccontextmanager
async def track_retrieval_time(
    user_id: str,
) -> AsyncGenerator[None, None]:
    """Track memory retrieval timing."""
    metrics = CortexMetrics.get_instance()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if PROMETHEUS_AVAILABLE:
            metrics.retrieval_latency.observe(elapsed)
            metrics.memories_retrieved.labels(user_id=user_id).inc()
        logger.debug("retrieval_complete", user_id=user_id, duration_s=elapsed)


@asynccontextmanager
async def track_store_time(
    user_id: str,
    memory_type: str,
) -> AsyncGenerator[None, None]:
    """Track memory store timing."""
    metrics = CortexMetrics.get_instance()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if PROMETHEUS_AVAILABLE:
            metrics.store_latency.observe(elapsed)
            metrics.memories_stored.labels(memory_type=memory_type, user_id=user_id).inc()
        logger.debug("store_complete", user_id=user_id, memory_type=memory_type, duration_s=elapsed)


@asynccontextmanager
async def track_embedding_time(
    provider: str,
) -> AsyncGenerator[None, None]:
    """Track embedding generation timing."""
    metrics = CortexMetrics.get_instance()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if PROMETHEUS_AVAILABLE:
            metrics.embedding_latency.observe(elapsed)
            metrics.embedding_requests.labels(provider=provider).inc()


# ==================== HELPER FUNCTIONS ====================


def record_consolidation_run(success: bool) -> None:
    """Record a consolidation run."""
    metrics = CortexMetrics.get_instance()
    if PROMETHEUS_AVAILABLE:
        status = "success" if success else "failure"
        metrics.consolidation_runs.labels(status=status).inc()


def record_patterns_extracted(count: int) -> None:
    """Record patterns extracted."""
    metrics = CortexMetrics.get_instance()
    if PROMETHEUS_AVAILABLE:
        metrics.patterns_extracted.inc(count)


def record_contradictions_detected(count: int) -> None:
    """Record contradictions detected."""
    metrics = CortexMetrics.get_instance()
    if PROMETHEUS_AVAILABLE:
        metrics.contradictions_detected.inc(count)


def record_memories_compacted(count: int) -> None:
    """Record memories compacted."""
    metrics = CortexMetrics.get_instance()
    if PROMETHEUS_AVAILABLE:
        metrics.memories_compacted.inc(count)


def record_working_memory_expired(user_id: str, count: int) -> None:
    """Record expired working memory entries."""
    metrics = CortexMetrics.get_instance()
    if PROMETHEUS_AVAILABLE:
        metrics.working_memory_expired.labels(user_id=user_id).inc(count)


def set_working_memory_size(user_id: str, size: int) -> None:
    """Set current working memory size for a user."""
    metrics = CortexMetrics.get_instance()
    if PROMETHEUS_AVAILABLE:
        metrics.working_memory_size.labels(user_id=user_id).set(size)


def set_redis_connected(connected: bool) -> None:
    """Set Redis connection status."""
    metrics = CortexMetrics.get_instance()
    if PROMETHEUS_AVAILABLE:
        metrics.redis_connected.set(1 if connected else 0)


def set_postgres_connected(connected: bool) -> None:
    """Set Postgres connection status."""
    metrics = CortexMetrics.get_instance()
    if PROMETHEUS_AVAILABLE:
        metrics.postgres_connected.set(1 if connected else 0)


def record_scoring_request() -> None:
    """Record an LLM scoring request."""
    metrics = CortexMetrics.get_instance()
    if PROMETHEUS_AVAILABLE:
        metrics.scoring_requests.inc()
