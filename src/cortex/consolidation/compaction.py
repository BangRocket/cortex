"""Compaction job for old episodic memories."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import structlog

from cortex.models import Memory, MemoryStatus, MemoryType
from cortex.stores.postgres_store import PostgresStore
from cortex.utils.embedder import Embedder
from cortex.utils.scorer import EmotionScorer

logger = structlog.get_logger(__name__)


SUMMARY_PROMPT = """Summarize this week's memories into a concise paragraph.
Preserve important facts, emotions, and events. Omit mundane details.

Week of {week_start}:
{memories}

Summary (2-3 sentences):"""


class CompactionJob:
    """Compacts old episodic memories into summaries."""

    def __init__(
        self,
        scorer: EmotionScorer,
        postgres: PostgresStore,
        embedder: Embedder,
        compact_after_days: int = 30,
        min_memories: int = 10,
    ) -> None:
        self.scorer = scorer
        self.postgres = postgres
        self.embedder = embedder
        self.compact_after_days = compact_after_days
        self.min_memories = min_memories

    async def compact_user(self, user_id: str) -> dict[str, int]:
        """Compact old episodic memories into summaries."""
        cutoff = datetime.utcnow() - timedelta(days=self.compact_after_days)

        # Get old episodic memories
        old_memories = await self.postgres.get_memories(
            user_id=user_id,
            memory_type=MemoryType.EPISODIC,
            before=cutoff,
            status=MemoryStatus.ACTIVE,
            limit=500,
        )

        if len(old_memories) < self.min_memories:
            logger.debug(
                "insufficient_memories_for_compaction",
                user_id=user_id,
                count=len(old_memories),
            )
            return {"weeks_processed": 0, "memories_archived": 0}

        # Group by week
        weeks = self._group_by_week(old_memories)

        weeks_processed = 0
        memories_archived = 0

        for week_start, memories in weeks.items():
            if len(memories) < 5:
                continue

            # Summarize the week
            summary = await self._summarize_week(memories, week_start)

            # Generate embedding for summary
            embedding = await self.embedder.embed(summary)

            # Store summary as semantic memory
            summary_memory = Memory(
                user_id=user_id,
                content=summary,
                memory_type=MemoryType.SEMANTIC,
                embedding=embedding,
                source="compaction",
                metadata={
                    "week_start": week_start.isoformat(),
                    "memories_compacted": len(memories),
                },
            )
            await self.postgres.store(summary_memory)

            # Archive originals
            for mem in memories:
                if mem.id:
                    await self.postgres.update_status(mem.id, MemoryStatus.ARCHIVED)

            weeks_processed += 1
            memories_archived += len(memories)

        logger.info(
            "compaction_complete",
            user_id=user_id,
            weeks=weeks_processed,
            archived=memories_archived,
        )

        return {"weeks_processed": weeks_processed, "memories_archived": memories_archived}

    def _group_by_week(self, memories: list[Memory]) -> dict[datetime, list[Memory]]:
        """Group memories by week."""
        weeks: dict[datetime, list[Memory]] = {}

        for mem in memories:
            # Get Monday of the week
            week_start = mem.created_at - timedelta(days=mem.created_at.weekday())
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

            if week_start not in weeks:
                weeks[week_start] = []
            weeks[week_start].append(mem)

        return weeks

    async def _summarize_week(
        self, memories: list[Memory], week_start: datetime
    ) -> str:
        """Generate summary of a week's memories."""
        memory_text = "\n".join([f"- {m.content}" for m in memories[:30]])

        prompt = SUMMARY_PROMPT.format(
            week_start=week_start.strftime("%B %d, %Y"),
            memories=memory_text,
        )

        try:
            response = await self.scorer.client.chat.completions.create(
                model=self.scorer.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,  # Slight creativity for natural summaries
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error("summarization_failed", error=str(e))
            # Fallback: concatenate top memories
            return " ".join([m.content for m in memories[:5]])
