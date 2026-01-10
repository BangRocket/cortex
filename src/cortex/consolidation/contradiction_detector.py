"""Contradiction detection between memories."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import structlog

from cortex.models import Contradiction, Memory, MemoryStatus, MemoryType
from cortex.stores.postgres_store import PostgresStore
from cortex.utils.scorer import EmotionScorer

logger = structlog.get_logger(__name__)


CONTRADICTION_PROMPT = """Do any of these statements contradict each other?

Statements:
{statements}

If contradictions exist, return JSON array:
[{{"a": <index>, "b": <index>, "reason": "explanation"}}]

If no contradictions, return: []"""


class ContradictionDetector:
    """Detects contradictions between memories."""

    def __init__(
        self,
        scorer: EmotionScorer,
        postgres: PostgresStore,
        similarity_threshold: float = 0.8,
    ) -> None:
        self.scorer = scorer
        self.postgres = postgres
        self.similarity_threshold = similarity_threshold

    async def detect_for_user(self, user_id: str) -> list[Contradiction]:
        """Find contradicting memories for a user."""
        # Get facts that could contradict (identity + semantic)
        facts = await self.postgres.get_memories(
            user_id=user_id,
            memory_types=[MemoryType.IDENTITY, MemoryType.SEMANTIC],
            status=MemoryStatus.ACTIVE,
            limit=200,
        )

        if len(facts) < 2:
            return []

        # Cluster by embedding similarity
        clusters = self._cluster_by_similarity(facts)

        contradictions = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue

            # Check cluster for contradictions
            found = await self._check_cluster(cluster)
            contradictions.extend(found)

        logger.info(
            "contradiction_detection_complete",
            user_id=user_id,
            facts=len(facts),
            clusters=len(clusters),
            contradictions=len(contradictions),
        )
        return contradictions

    def _cluster_by_similarity(self, memories: list[Memory]) -> list[list[Memory]]:
        """Cluster memories by embedding similarity."""
        # Filter memories with embeddings
        with_embeddings = [m for m in memories if m.embedding]

        if len(with_embeddings) < 2:
            return []

        try:
            from sklearn.cluster import AgglomerativeClustering

            embeddings = np.array([m.embedding for m in with_embeddings])

            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - self.similarity_threshold,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(embeddings)

            clusters: dict[int, list[Memory]] = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(with_embeddings[i])

            return list(clusters.values())

        except Exception as e:
            logger.error("clustering_failed", error=str(e))
            return []

    async def _check_cluster(self, memories: list[Memory]) -> list[Contradiction]:
        """Ask LLM if any memories in cluster contradict."""
        memory_text = "\n".join([f"{i}. {m.content}" for i, m in enumerate(memories)])

        prompt = CONTRADICTION_PROMPT.format(statements=memory_text)

        try:
            response = await self.scorer.client.chat.completions.create(
                model=self.scorer.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if "```" in text:
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
                if match:
                    text = match.group(1)

            data = json.loads(text)

            contradictions = []
            for c in data:
                try:
                    mem_a = memories[c["a"]]
                    mem_b = memories[c["b"]]
                    if mem_a.id and mem_b.id:
                        contradictions.append(
                            Contradiction(
                                memory_a=mem_a.id,
                                memory_b=mem_b.id,
                                reason=c["reason"],
                            )
                        )
                except (KeyError, IndexError):
                    continue

            return contradictions

        except (json.JSONDecodeError, Exception) as e:
            logger.warning("contradiction_check_failed", error=str(e))
            return []

    async def resolve_contradiction(
        self,
        contradiction: Contradiction,
        resolution: str,  # "a_wins" | "b_wins" | "merge" | "both_valid"
        note: str | None = None,
    ) -> None:
        """Resolve a detected contradiction."""
        if resolution == "a_wins":
            await self.postgres.update_status(
                contradiction.memory_b, MemoryStatus.SUPERSEDED
            )
        elif resolution == "b_wins":
            await self.postgres.update_status(
                contradiction.memory_a, MemoryStatus.SUPERSEDED
            )
        elif resolution == "merge":
            # Create merged memory, supersede both
            mem_a = await self.postgres.get(contradiction.memory_a)
            mem_b = await self.postgres.get(contradiction.memory_b)

            if mem_a and mem_b:
                merged_content = await self._merge_memories(mem_a, mem_b)

                new_mem = Memory(
                    user_id=mem_a.user_id,
                    content=merged_content,
                    memory_type=mem_a.memory_type,
                    source="contradiction_merge",
                )
                await self.postgres.store(new_mem)

                await self.postgres.update_status(mem_a.id, MemoryStatus.SUPERSEDED)
                await self.postgres.update_status(mem_b.id, MemoryStatus.SUPERSEDED)

        # Log resolution
        if contradiction.id:
            await self.postgres.resolve_contradiction(contradiction.id, resolution, note)

        logger.info(
            "contradiction_resolved",
            contradiction_id=contradiction.id,
            resolution=resolution,
        )

    async def _merge_memories(self, mem_a: Memory, mem_b: Memory) -> str:
        """Merge two contradicting memories into one."""
        prompt = f"""Merge these two potentially contradicting statements into a single accurate statement:

Statement A: {mem_a.content}
Statement B: {mem_b.content}

Provide a single merged statement that resolves any contradiction:"""

        try:
            response = await self.scorer.client.chat.completions.create(
                model=self.scorer.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            # Fallback to newer memory
            return mem_b.content if mem_b.created_at > mem_a.created_at else mem_a.content
