"""Pattern extraction from episodic memories."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

import structlog

from cortex.models import Memory, MemoryType, Pattern
from cortex.stores.postgres_store import PostgresStore
from cortex.stores.redis_store import RedisStore
from cortex.utils.scorer import EmotionScorer

logger = structlog.get_logger(__name__)


PATTERN_EXTRACTION_PROMPT = """Analyze these recent memories and identify stable patterns about this user.

Memories:
{memories}

For each pattern, provide:
1. category: identity | preference | behavior | relationship
2. fact: The pattern statement
3. confidence: 0.0 to 1.0
4. evidence: Which memory indices support this (0-indexed)

Only report patterns with confidence >= 0.7.
Return as JSON array.

Example:
[
  {{"category": "identity", "fact": "works as software engineer", "confidence": 0.9, "evidence": [0, 5, 12]}},
  {{"category": "preference", "fact": "prefers direct communication", "confidence": 0.75, "evidence": [3, 8]}}
]"""


class PatternExtractor:
    """Extracts patterns from episodic memories."""

    def __init__(
        self,
        scorer: EmotionScorer,
        postgres: PostgresStore,
        redis: RedisStore,
        lookback_days: int = 7,
        min_memories: int = 5,
        confidence_threshold: float = 0.8,
    ) -> None:
        self.scorer = scorer
        self.postgres = postgres
        self.redis = redis
        self.lookback_days = lookback_days
        self.min_memories = min_memories
        self.confidence_threshold = confidence_threshold

    async def extract_for_user(self, user_id: str) -> list[Pattern]:
        """Extract patterns from recent episodic memories."""
        # Get last N days of episodic memories
        memories = await self.postgres.get_memories(
            user_id=user_id,
            memory_type=MemoryType.EPISODIC,
            since=datetime.utcnow() - timedelta(days=self.lookback_days),
            limit=100,
        )

        if len(memories) < self.min_memories:
            logger.debug(
                "insufficient_memories_for_patterns",
                user_id=user_id,
                count=len(memories),
            )
            return []

        # Ask LLM to identify patterns
        prompt = self._build_pattern_prompt(memories)

        try:
            response = await self.scorer.client.chat.completions.create(
                model=self.scorer.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            patterns = self._parse_patterns(text)

            logger.info(
                "patterns_extracted",
                user_id=user_id,
                memories=len(memories),
                patterns=len(patterns),
            )
            return patterns

        except Exception as e:
            logger.error("pattern_extraction_failed", error=str(e), user_id=user_id)
            return []

    def _build_pattern_prompt(self, memories: list[Memory]) -> str:
        """Build prompt for pattern extraction."""
        memory_text = "\n".join(
            [
                f"- [{i}] [{m.created_at.strftime('%Y-%m-%d')}] {m.content}"
                for i, m in enumerate(memories[:50])  # Limit to avoid token limits
            ]
        )
        return PATTERN_EXTRACTION_PROMPT.format(memories=memory_text)

    def _parse_patterns(self, response: str) -> list[Pattern]:
        """Parse patterns from LLM response."""
        # Handle markdown code blocks
        if "```" in response:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                response = match.group(1)

        try:
            data = json.loads(response)
            return [
                Pattern(
                    category=p["category"],
                    fact=p["fact"],
                    confidence=p["confidence"],
                    evidence=p.get("evidence", []),
                )
                for p in data
                if p.get("confidence", 0) >= 0.7
            ]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("pattern_parse_failed", error=str(e))
            return []

    async def apply_patterns(self, user_id: str, patterns: list[Pattern]) -> int:
        """Apply high-confidence patterns to identity."""
        applied = 0

        for pattern in patterns:
            if pattern.confidence < self.confidence_threshold:
                continue

            if pattern.category == "identity":
                # Update Redis identity
                key = self._infer_identity_key(pattern.fact)
                await self.redis.update_identity_field(user_id, key, pattern.fact)
                applied += 1

            # Store as semantic memory regardless of category
            from cortex.utils.embedder import create_embedder
            from cortex.config import EmbeddingConfig

            memory = Memory(
                user_id=user_id,
                content=pattern.fact,
                memory_type=MemoryType.SEMANTIC,
                confidence=pattern.confidence,
                source="consolidation",
                metadata={"pattern_category": pattern.category},
            )
            await self.postgres.store(memory)

        logger.info("patterns_applied", user_id=user_id, applied=applied)
        return applied

    def _infer_identity_key(self, fact: str) -> str:
        """Infer identity key from a fact statement."""
        fact_lower = fact.lower()

        # Common identity patterns
        if any(word in fact_lower for word in ["name is", "called", "goes by"]):
            return "name"
        if any(word in fact_lower for word in ["works as", "job", "occupation", "profession"]):
            return "occupation"
        if any(word in fact_lower for word in ["lives in", "located", "from"]):
            return "location"
        if any(word in fact_lower for word in ["child", "kid", "son", "daughter", "family"]):
            return "family"
        if any(word in fact_lower for word in ["prefer", "like", "enjoy", "love"]):
            return "preferences"

        # Default to key_facts
        return "key_facts"
