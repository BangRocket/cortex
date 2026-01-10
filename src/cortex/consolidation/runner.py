"""Consolidation job runner with scheduling."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from cortex.config import ConsolidationConfig
from cortex.consolidation.compaction import CompactionJob
from cortex.consolidation.contradiction_detector import ContradictionDetector
from cortex.consolidation.pattern_extractor import PatternExtractor
from cortex.models import ConsolidationLog
from cortex.stores.postgres_store import PostgresStore
from cortex.stores.redis_store import RedisStore
from cortex.utils.embedder import Embedder
from cortex.utils.scorer import EmotionScorer

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class ConsolidationRunner:
    """Runs consolidation jobs on a schedule."""

    def __init__(
        self,
        postgres: PostgresStore,
        redis: RedisStore,
        embedder: Embedder,
        scorer: EmotionScorer,
        config: ConsolidationConfig | None = None,
    ) -> None:
        self.postgres = postgres
        self.redis = redis
        self.embedder = embedder
        self.scorer = scorer
        self.config = config or ConsolidationConfig()

        self.pattern_extractor = PatternExtractor(
            scorer=scorer,
            postgres=postgres,
            redis=redis,
            embedder=embedder,
            lookback_days=self.config.pattern_lookback_days,
            min_memories=self.config.pattern_min_memories,
            confidence_threshold=self.config.pattern_confidence_threshold,
        )

        self.contradiction_detector = ContradictionDetector(
            scorer=scorer,
            postgres=postgres,
            similarity_threshold=self.config.similarity_threshold,
        )

        self.compaction_job = CompactionJob(
            scorer=scorer,
            postgres=postgres,
            embedder=embedder,
            compact_after_days=self.config.compact_after_days,
            min_memories=self.config.compact_min_memories,
        )

        self.scheduler = AsyncIOScheduler()
        self._running = False

    def start(self) -> None:
        """Start the consolidation scheduler."""
        # Main consolidation job
        self.scheduler.add_job(
            self.run_all_users,
            "interval",
            hours=self.config.pattern_interval_hours,
            id="consolidation_main",
        )

        # Cleanup job
        self.scheduler.add_job(
            self.cleanup_expired,
            "interval",
            hours=self.config.cleanup_interval_hours,
            id="cleanup",
        )

        self.scheduler.start()
        self._running = True
        logger.info(
            "consolidation_scheduler_started",
            pattern_interval=self.config.pattern_interval_hours,
            cleanup_interval=self.config.cleanup_interval_hours,
        )

    def stop(self) -> None:
        """Stop the scheduler."""
        self.scheduler.shutdown()
        self._running = False
        logger.info("consolidation_scheduler_stopped")

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    async def run_all_users(self) -> None:
        """Run consolidation for all active users."""
        user_ids = await self.postgres.get_active_user_ids()

        logger.info("consolidation_run_starting", users=len(user_ids))

        for user_id in user_ids:
            try:
                await self.run_for_user(user_id)
            except Exception as e:
                logger.error("consolidation_user_failed", user_id=user_id, error=str(e))

        logger.info("consolidation_run_complete", users=len(user_ids))

    async def run_for_user(self, user_id: str) -> ConsolidationLog:
        """Run full consolidation for one user."""
        log = ConsolidationLog(user_id=user_id, started_at=datetime.utcnow())

        try:
            # 1. Extract patterns
            patterns = await self.pattern_extractor.extract_for_user(user_id)
            log.patterns_found = len(patterns)

            # Apply high-confidence patterns
            if patterns:
                log.identities_updated = await self.pattern_extractor.apply_patterns(
                    user_id, patterns
                )

            # 2. Detect contradictions
            contradictions = await self.contradiction_detector.detect_for_user(user_id)
            log.contradictions_found = len(contradictions)

            # Flag contradictions for review
            for c in contradictions:
                await self.postgres.flag_contradiction(user_id, c)

            # 3. Compact old memories
            compaction_result = await self.compaction_job.compact_user(user_id)
            log.memories_compacted = compaction_result.get("memories_archived", 0)

            log.success = True

        except Exception as e:
            log.success = False
            log.error = str(e)
            logger.error("consolidation_failed", user_id=user_id, error=str(e))

        log.completed_at = datetime.utcnow()
        log.duration_ms = int(
            (log.completed_at - log.started_at).total_seconds() * 1000
        )

        # Log to database
        await self.postgres.log_consolidation(log)

        logger.info(
            "consolidation_complete",
            user_id=user_id,
            patterns=log.patterns_found,
            contradictions=log.contradictions_found,
            compacted=log.memories_compacted,
            duration_ms=log.duration_ms,
        )

        return log

    async def cleanup_expired(self) -> None:
        """Clean up expired working memories."""
        user_ids = await self.redis.get_active_user_ids()

        for user_id in user_ids:
            await self.redis.cleanup_expired_working(user_id)

        logger.debug("cleanup_complete", users=len(user_ids))


class ConsolidationMetrics:
    """Aggregated metrics from consolidation runs."""

    def __init__(
        self,
        users_processed: int = 0,
        patterns_extracted: int = 0,
        contradictions_found: int = 0,
        memories_compacted: int = 0,
        errors: int = 0,
        duration_seconds: float = 0.0,
    ) -> None:
        self.users_processed = users_processed
        self.patterns_extracted = patterns_extracted
        self.contradictions_found = contradictions_found
        self.memories_compacted = memories_compacted
        self.errors = errors
        self.duration_seconds = duration_seconds


class ConsolidationMonitor:
    """Monitor consolidation health."""

    def __init__(self, postgres: PostgresStore) -> None:
        self.postgres = postgres

    async def get_metrics(self, hours: int = 24) -> ConsolidationMetrics:
        """Get aggregated metrics from recent runs."""
        from datetime import timedelta

        logs = await self.postgres.get_consolidation_logs(
            since=datetime.utcnow() - timedelta(hours=hours)
        )

        return ConsolidationMetrics(
            users_processed=len(logs),
            patterns_extracted=sum(log.patterns_found for log in logs),
            contradictions_found=sum(log.contradictions_found for log in logs),
            memories_compacted=sum(log.memories_compacted for log in logs),
            errors=sum(1 for log in logs if not log.success),
            duration_seconds=sum(log.duration_ms / 1000 for log in logs),
        )

    async def get_pending_contradictions(self, user_id: str | None = None) -> int:
        """Get count of unresolved contradictions."""
        contradictions = await self.postgres.get_contradictions(
            user_id=user_id, resolution="pending"
        )
        return len(contradictions)
