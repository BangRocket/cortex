"""Direct database migration from mem0 to Cortex schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from cortex.config import MigrationConfig, PostgresConfig
from cortex.migration.schema_detector import SchemaDetector, SchemaType

if TYPE_CHECKING:
    import asyncpg

logger = structlog.get_logger(__name__)


class Mem0MigrationError(Exception):
    """Error during mem0 migration."""

    pass


class Mem0SchemaDetectedError(Exception):
    """Raised when mem0 schema is detected but auto-migration is disabled."""

    pass


@dataclass
class MigrationReport:
    """Report from a direct migration."""

    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    success: bool = False

    # Counts
    total_source_records: int = 0
    migrated: int = 0
    skipped: int = 0
    errors: int = 0

    # Details
    backup_table: str | None = None
    users_migrated: list[str] = field(default_factory=list)
    error_details: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float | None:
        """Get duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class DirectMigrator:
    """
    Migrates mem0 data to Cortex schema directly via SQL.

    This is faster and more reliable than using the mem0 API,
    especially for large datasets.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        migration_config: MigrationConfig,
        postgres_config: PostgresConfig,
    ) -> None:
        self.pool = pool
        self.config = migration_config
        self.postgres_config = postgres_config
        self.detector = SchemaDetector()

    async def migrate(self, dry_run: bool = False) -> MigrationReport:
        """
        Run the full migration from mem0 to Cortex.

        Args:
            dry_run: If True, don't make changes, just report what would happen

        Returns:
            MigrationReport with results
        """
        report = MigrationReport()

        try:
            # Verify we have mem0 schema
            schema_info = await self.detector.detect(self.pool)

            if schema_info.schema_type != SchemaType.MEM0:
                raise Mem0MigrationError(
                    f"Expected mem0 schema, found: {schema_info.schema_type.value}"
                )

            report.total_source_records = schema_info.memory_count
            logger.info(
                "migration_starting",
                dry_run=dry_run,
                source_records=report.total_source_records,
                users=schema_info.user_count,
            )

            if dry_run:
                # Just get stats without making changes
                stats = await self.detector.get_mem0_stats(self.pool)
                report.users_migrated = stats["users"]
                report.migrated = stats["total_memories"] - stats["missing_content"]
                report.skipped = stats["missing_content"]
                report.success = True
                report.completed_at = datetime.now(timezone.utc)
                return report

            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # 1. Backup original table
                    if self.config.backup_before_migrate:
                        report.backup_table = await self._backup_mem0_table(conn)

                    # 2. Create Cortex schema
                    await self._create_cortex_schema(conn)

                    # 3. Migrate data
                    result = await self._transform_and_insert(conn)
                    report.migrated = result["migrated"]
                    report.skipped = result["skipped"]
                    report.users_migrated = result["users"]

                    # 4. Create indexes
                    await self._create_indexes(conn)

            report.success = True
            logger.info(
                "migration_complete",
                migrated=report.migrated,
                skipped=report.skipped,
                backup_table=report.backup_table,
            )

        except Exception as e:
            report.success = False
            report.error_details.append(str(e))
            logger.exception("migration_failed", error=str(e))

        report.completed_at = datetime.now(timezone.utc)
        return report

    async def _backup_mem0_table(self, conn: asyncpg.Connection) -> str:
        """Backup the mem0 memories table."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_name = f"memories_mem0_backup_{timestamp}"

        await conn.execute(f'ALTER TABLE memories RENAME TO "{backup_name}"')
        logger.info("mem0_table_backed_up", backup_table=backup_name)

        return backup_name

    async def _create_cortex_schema(self, conn: asyncpg.Connection) -> None:
        """Create the Cortex memories table schema."""
        vector_dims = self.postgres_config.vector_dimensions

        await conn.execute(f"""
            CREATE TABLE memories (
                id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id             VARCHAR(255) NOT NULL,
                project_id          VARCHAR(255),
                content             TEXT NOT NULL,
                memory_type         VARCHAR(50) NOT NULL DEFAULT 'episodic',
                emotional_score     FLOAT DEFAULT 0.5,
                importance          FLOAT DEFAULT 0.5,
                confidence          FLOAT DEFAULT 1.0,
                created_at          TIMESTAMPTZ DEFAULT NOW(),
                updated_at          TIMESTAMPTZ DEFAULT NOW(),
                last_accessed       TIMESTAMPTZ,
                access_count        INTEGER DEFAULT 0,
                supersedes          UUID,
                source              VARCHAR(100) DEFAULT 'mem0_migration',
                tags                TEXT[] DEFAULT ARRAY[]::TEXT[],
                metadata            JSONB DEFAULT '{{}}'::jsonb,
                embedding           vector({vector_dims}),
                status              VARCHAR(50) DEFAULT 'active'
            )
        """)

        logger.info("cortex_schema_created", vector_dimensions=vector_dims)

    async def _transform_and_insert(self, conn: asyncpg.Connection) -> dict[str, Any]:
        """Transform mem0 data and insert into Cortex schema."""
        default_user = self.config.default_user_id
        preserve_ids = self.config.preserve_mem0_ids

        # Get the backup table name (most recent)
        backup_table = await conn.fetchval("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE 'memories_mem0_backup_%'
            ORDER BY table_name DESC
            LIMIT 1
        """)

        if not backup_table:
            raise Mem0MigrationError("Backup table not found")

        # Build the metadata JSON based on config
        if preserve_ids:
            metadata_expr = f"""
                jsonb_build_object(
                    'mem0_id', id,
                    'mem0_hash', metadata->>'hash',
                    'migrated_at', '{datetime.now(timezone.utc).isoformat()}',
                    'original_metadata', metadata
                )
            """
        else:
            metadata_expr = f"""
                jsonb_build_object(
                    'migrated_at', '{datetime.now(timezone.utc).isoformat()}'
                )
            """

        # Insert with transformation
        result = await conn.execute(f"""
            INSERT INTO memories (
                user_id,
                content,
                memory_type,
                embedding,
                created_at,
                updated_at,
                metadata,
                source
            )
            SELECT
                COALESCE(NULLIF(metadata->>'user_id', ''), '{default_user}'),
                metadata->>'memory',
                'episodic',
                embedding,
                COALESCE(created_at, NOW()),
                COALESCE(updated_at, NOW()),
                {metadata_expr},
                'mem0_migration'
            FROM "{backup_table}"
            WHERE metadata->>'memory' IS NOT NULL
              AND metadata->>'memory' != ''
        """)

        # Parse the result to get count
        migrated = int(result.split()[-1]) if result else 0

        # Count skipped
        skipped = await conn.fetchval(f"""
            SELECT COUNT(*)
            FROM "{backup_table}"
            WHERE metadata->>'memory' IS NULL
               OR metadata->>'memory' = ''
        """)

        # Get users
        users = await conn.fetch("""
            SELECT DISTINCT user_id FROM memories ORDER BY user_id
        """)
        user_list = [row["user_id"] for row in users]

        return {
            "migrated": migrated,
            "skipped": skipped or 0,
            "users": user_list,
        }

    async def _create_indexes(self, conn: asyncpg.Connection) -> None:
        """Create indexes on the new Cortex table."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_user_type ON memories(user_id, memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_user_status ON memories(user_id, status)",
            "CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id) WHERE project_id IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC)",
        ]

        for idx_sql in indexes:
            await conn.execute(idx_sql)

        # Vector index (this can be slow for large datasets)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_embedding
            ON memories USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)

        logger.info("indexes_created")

    async def validate_migration(self) -> dict[str, Any]:
        """
        Validate that migration was successful.

        Returns:
            Validation results
        """
        validation: dict[str, Any] = {
            "valid": False,
            "checks": {},
            "issues": [],
        }

        async with self.pool.acquire() as conn:
            # Check Cortex table exists
            cortex_exists = await conn.fetchval("""
                SELECT EXISTS(
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'memories'
                )
            """)
            validation["checks"]["cortex_table_exists"] = cortex_exists

            if not cortex_exists:
                validation["issues"].append("Cortex memories table not found")
                return validation

            # Check backup exists
            backup_exists = await conn.fetchval("""
                SELECT EXISTS(
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name LIKE 'memories_mem0_backup_%'
                )
            """)
            validation["checks"]["backup_exists"] = backup_exists

            # Compare counts
            cortex_count = await conn.fetchval("SELECT COUNT(*) FROM memories")
            validation["checks"]["cortex_count"] = cortex_count

            if backup_exists:
                backup_table = await conn.fetchval("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_name LIKE 'memories_mem0_backup_%'
                    ORDER BY table_name DESC
                    LIMIT 1
                """)

                backup_count = await conn.fetchval(f"""
                    SELECT COUNT(*)
                    FROM "{backup_table}"
                    WHERE metadata->>'memory' IS NOT NULL
                      AND metadata->>'memory' != ''
                """)
                validation["checks"]["expected_count"] = backup_count

                if cortex_count != backup_count:
                    validation["issues"].append(
                        f"Count mismatch: {cortex_count} vs expected {backup_count}"
                    )

            # Check vector index exists
            index_exists = await conn.fetchval("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_indexes
                    WHERE indexname = 'idx_memories_embedding'
                )
            """)
            validation["checks"]["vector_index_exists"] = index_exists

            if not index_exists:
                validation["issues"].append("Vector index not created")

            # Sample search test
            try:
                await conn.fetchval("""
                    SELECT id FROM memories
                    WHERE embedding IS NOT NULL
                    LIMIT 1
                """)
                validation["checks"]["vector_query_works"] = True
            except Exception as e:
                validation["checks"]["vector_query_works"] = False
                validation["issues"].append(f"Vector query failed: {e}")

        validation["valid"] = len(validation["issues"]) == 0
        return validation

    async def rollback(self) -> bool:
        """
        Rollback migration by restoring from backup.

        Returns:
            True if rollback successful
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Find backup table
                backup_table = await conn.fetchval("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_name LIKE 'memories_mem0_backup_%'
                    ORDER BY table_name DESC
                    LIMIT 1
                """)

                if not backup_table:
                    logger.error("rollback_failed", reason="No backup table found")
                    return False

                # Drop Cortex table
                await conn.execute("DROP TABLE IF EXISTS memories CASCADE")

                # Restore backup
                await conn.execute(f'ALTER TABLE "{backup_table}" RENAME TO memories')

                logger.info("rollback_complete", restored_from=backup_table)
                return True


async def migrate_mem0_to_cortex(
    pool: asyncpg.Pool,
    migration_config: MigrationConfig,
    postgres_config: PostgresConfig,
    dry_run: bool = False,
) -> MigrationReport:
    """
    Convenience function to run mem0 to Cortex migration.

    Args:
        pool: asyncpg connection pool
        migration_config: Migration settings
        postgres_config: PostgreSQL settings
        dry_run: If True, don't make changes

    Returns:
        MigrationReport with results
    """
    migrator = DirectMigrator(pool, migration_config, postgres_config)
    return await migrator.migrate(dry_run=dry_run)
