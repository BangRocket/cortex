"""Migration scripts for importing memories from mem0."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from cortex.models import Memory, MemoryType
from cortex.stores.postgres_store import PostgresStore
from cortex.stores.redis_store import RedisStore
from cortex.utils.embedder import Embedder
from cortex.utils.scorer import EmotionScorer

logger = structlog.get_logger(__name__)


@dataclass
class MigrationReport:
    """Report from a migration run."""

    user_id: str
    started_at: datetime
    completed_at: datetime | None = None
    success: bool = False
    total_exported: int = 0
    identity_facts: int = 0
    imported: int = 0
    skipped: int = 0
    error: str | None = None


@dataclass
class ClassificationResult:
    """Result of classifying a memory."""

    memory_type: MemoryType
    confidence: float
    identity_key: str | None = None


class Mem0Migrator:
    """
    Migrates memories from mem0 to Cortex.

    Supports both the hosted mem0 client and self-hosted instances.
    """

    def __init__(
        self,
        redis_store: RedisStore,
        postgres_store: PostgresStore,
        embedder: Embedder,
        scorer: EmotionScorer,
        mem0_api_key: str | None = None,
        mem0_base_url: str | None = None,
    ) -> None:
        self.redis = redis_store
        self.postgres = postgres_store
        self.embedder = embedder
        self.scorer = scorer
        self.mem0_api_key = mem0_api_key
        self.mem0_base_url = mem0_base_url
        self._mem0_client = None

    @property
    def mem0_client(self):
        """Lazy load mem0 client."""
        if self._mem0_client is None:
            from mem0 import MemoryClient

            if self.mem0_api_key:
                self._mem0_client = MemoryClient(api_key=self.mem0_api_key)
            else:
                # Self-hosted - uses environment variables
                self._mem0_client = MemoryClient()
        return self._mem0_client

    async def export_from_mem0(self, user_id: str) -> list[dict[str, Any]]:
        """Export all memories for a user from mem0."""
        memories = []

        # mem0 client is synchronous
        results = self.mem0_client.get_all(user_id=user_id)

        for item in results:
            memories.append(
                {
                    "id": item.get("id"),
                    "content": item.get("memory"),
                    "metadata": item.get("metadata", {}),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "hash": item.get("hash"),
                }
            )

        logger.info("mem0_export_complete", user_id=user_id, count=len(memories))
        return memories

    async def classify_memories(
        self, memories: list[dict[str, Any]]
    ) -> list[ClassificationResult]:
        """Classify memories into Cortex types."""
        results = []

        for mem in memories:
            content = mem.get("content", "")
            classification = await self.scorer.classify(content)

            memory_type = MemoryType.EPISODIC
            try:
                memory_type = MemoryType(classification["type"])
            except ValueError:
                pass

            results.append(
                ClassificationResult(
                    memory_type=memory_type,
                    confidence=classification.get("confidence", 0.5),
                    identity_key=classification.get("identity_key"),
                )
            )

        logger.info("classification_complete", count=len(results))
        return results

    async def seed_identity(
        self,
        user_id: str,
        memories: list[dict[str, Any]],
        classifications: list[ClassificationResult],
    ) -> dict[str, Any]:
        """Extract identity facts and seed Redis."""
        identity: dict[str, Any] = {}

        for mem, classification in zip(memories, classifications):
            if classification.memory_type == MemoryType.IDENTITY:
                key = classification.identity_key
                content = mem.get("content", "")

                if key:
                    # For complex keys like 'family', accumulate
                    if key in ["family", "preferences", "key_facts"]:
                        if key not in identity:
                            identity[key] = []
                        identity[key].append(content)
                    else:
                        # Simple keys like 'name', 'occupation'
                        identity[key] = content

        # Post-process accumulated fields
        if "key_facts" in identity:
            identity["key_facts"] = identity["key_facts"][:20]  # Cap at 20

        if identity:
            await self.redis.set_identity(user_id, identity)

        logger.info("identity_seeded", user_id=user_id, facts=len(identity))
        return identity

    async def import_memories(
        self,
        user_id: str,
        memories: list[dict[str, Any]],
        classifications: list[ClassificationResult],
    ) -> dict[str, int]:
        """Import classified memories to Postgres."""
        imported = 0
        skipped = 0

        for mem, classification in zip(memories, classifications):
            # Skip low-confidence classifications
            if classification.confidence < 0.5:
                skipped += 1
                continue

            content = mem.get("content", "")
            if not content:
                skipped += 1
                continue

            try:
                # Score emotion
                emotional_score = await self.scorer.score(content)

                # Generate embedding
                embedding = await self.embedder.embed(content)

                # Create memory object
                memory = Memory(
                    content=content,
                    user_id=user_id,
                    memory_type=classification.memory_type,
                    emotional_score=emotional_score,
                    embedding=embedding,
                    source="migration",
                    metadata={
                        "original_id": mem.get("id"),
                        "migrated_at": datetime.utcnow().isoformat(),
                    },
                )

                await self.postgres.store(memory)
                imported += 1

            except Exception as e:
                logger.error("import_memory_failed", error=str(e), memory_id=mem.get("id"))
                skipped += 1

        logger.info("import_complete", user_id=user_id, imported=imported, skipped=skipped)
        return {"imported": imported, "skipped": skipped}

    async def migrate_user(self, user_id: str) -> MigrationReport:
        """Run full migration for a single user."""
        report = MigrationReport(user_id=user_id, started_at=datetime.utcnow())

        try:
            # 1. Export from mem0
            logger.info("migration_starting", user_id=user_id)
            memories = await self.export_from_mem0(user_id)
            report.total_exported = len(memories)

            if not memories:
                report.success = True
                report.completed_at = datetime.utcnow()
                return report

            # 2. Classify all memories
            classifications = await self.classify_memories(memories)

            # 3. Seed identity to Redis
            identity = await self.seed_identity(user_id, memories, classifications)
            report.identity_facts = len(identity)

            # 4. Import to Postgres
            result = await self.import_memories(user_id, memories, classifications)
            report.imported = result["imported"]
            report.skipped = result["skipped"]

            report.success = True

        except Exception as e:
            report.success = False
            report.error = str(e)
            logger.exception("migration_failed", user_id=user_id)

        report.completed_at = datetime.utcnow()
        return report

    async def validate_migration(self, user_id: str) -> dict[str, Any]:
        """Validate migration was successful."""
        validation: dict[str, Any] = {"user_id": user_id, "issues": []}

        # Check identity exists
        identity = await self.redis.get_identity(user_id)
        validation["has_identity"] = bool(identity)

        # Check memory counts
        original_memories = await self.export_from_mem0(user_id)
        migrated_count = await self.postgres.count_memories(user_id)

        validation["original_count"] = len(original_memories)
        validation["migrated_count"] = migrated_count

        # Allow for 10% variance due to filtering
        expected_min = len(original_memories) * 0.5  # At least 50% should migrate
        validation["count_valid"] = migrated_count >= expected_min

        if migrated_count < expected_min:
            validation["issues"].append(
                f"Only {migrated_count}/{len(original_memories)} memories migrated"
            )

        # Spot check: search for a known fact
        if identity.get("name"):
            from cortex.models import MemoryType

            results = await self.postgres.get_memories(
                user_id=user_id,
                memory_type=MemoryType.IDENTITY,
                limit=5,
            )
            validation["identity_memories_found"] = len(results)
            validation["search_works"] = len(results) > 0

        validation["valid"] = (
            validation["has_identity"]
            and validation["count_valid"]
            and not validation["issues"]
        )

        return validation


async def migrate_from_mem0_file(
    file_path: str,
    user_id: str,
    redis_store: RedisStore,
    postgres_store: PostgresStore,
    embedder: Embedder,
    scorer: EmotionScorer,
) -> MigrationReport:
    """
    Migrate from a JSON export file instead of live mem0.

    Useful for offline migration or when mem0 is not accessible.
    """
    report = MigrationReport(user_id=user_id, started_at=datetime.utcnow())

    try:
        # Load from file
        with open(file_path) as f:
            data = json.load(f)

        memories = data if isinstance(data, list) else data.get("memories", [])
        report.total_exported = len(memories)

        # Create migrator with stores
        migrator = Mem0Migrator(
            redis_store=redis_store,
            postgres_store=postgres_store,
            embedder=embedder,
            scorer=scorer,
        )

        # Run classification and import
        classifications = await migrator.classify_memories(memories)
        identity = await migrator.seed_identity(user_id, memories, classifications)
        report.identity_facts = len(identity)

        result = await migrator.import_memories(user_id, memories, classifications)
        report.imported = result["imported"]
        report.skipped = result["skipped"]

        report.success = True

    except Exception as e:
        report.success = False
        report.error = str(e)
        logger.exception("file_migration_failed", file_path=file_path, user_id=user_id)

    report.completed_at = datetime.utcnow()
    return report
