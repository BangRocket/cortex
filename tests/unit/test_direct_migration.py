"""Unit tests for direct mem0 to Cortex migration."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cortex.config import MigrationConfig, PostgresConfig
from cortex.migration.direct_migration import (
    DirectMigrator,
    Mem0MigrationError,
    Mem0SchemaDetectedError,
    MigrationReport,
)
from cortex.migration.schema_detector import SchemaInfo, SchemaType


class AsyncContextManager:
    """Helper for async context manager mocking."""

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *args):
        pass


class SyncContextManager:
    """Sync context manager that returns async context manager."""

    def __init__(self, value):
        self.value = value
        self._acm = AsyncContextManager(value)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *args):
        pass


class TestMigrationReport:
    """Test MigrationReport dataclass."""

    def test_defaults(self):
        """Test default values."""
        report = MigrationReport()

        assert report.success is False
        assert report.total_source_records == 0
        assert report.migrated == 0
        assert report.skipped == 0
        assert report.errors == 0
        assert report.backup_table is None
        assert report.users_migrated == []

    def test_duration(self):
        """Test duration calculation."""
        report = MigrationReport()
        report.started_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        report.completed_at = datetime(2024, 1, 1, 12, 0, 30, tzinfo=timezone.utc)

        assert report.duration_seconds == 30.0

    def test_duration_not_completed(self):
        """Test duration when not completed."""
        report = MigrationReport()

        assert report.duration_seconds is None


class TestMem0SchemaDetectedError:
    """Test Mem0SchemaDetectedError exception."""

    def test_exception_message(self):
        """Test exception can be raised with message."""
        with pytest.raises(Mem0SchemaDetectedError) as exc_info:
            raise Mem0SchemaDetectedError("Test error message")

        assert "Test error message" in str(exc_info.value)


class TestDirectMigrator:
    """Test DirectMigrator."""

    @pytest.fixture
    def migration_config(self) -> MigrationConfig:
        """Create test migration config."""
        return MigrationConfig(
            auto_migrate_mem0=True,
            backup_before_migrate=True,
            preserve_mem0_ids=True,
            default_user_id="default_user",
        )

    @pytest.fixture
    def postgres_config(self) -> PostgresConfig:
        """Create test postgres config."""
        return PostgresConfig(
            host="localhost",
            port=5432,
            database="test_db",
            vector_dimensions=1536,
        )

    @pytest.fixture
    def mock_pool(self):
        """Create mock connection pool."""
        pool = MagicMock()
        conn = MagicMock()

        # Set up async methods on connection
        conn.execute = AsyncMock(return_value="INSERT 0 90")
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchval = AsyncMock(return_value=None)
        conn.fetchrow = AsyncMock(return_value=None)

        # Set up connection context manager
        pool.acquire.return_value = AsyncContextManager(conn)

        # Set up transaction context manager - sync function returning async cm
        conn.transaction.return_value = SyncContextManager(None)

        return pool, conn

    @pytest.fixture
    def migrator(
        self,
        mock_pool,
        migration_config: MigrationConfig,
        postgres_config: PostgresConfig,
    ) -> DirectMigrator:
        """Create test migrator."""
        pool, _ = mock_pool
        return DirectMigrator(pool, migration_config, postgres_config)

    @pytest.mark.asyncio
    async def test_migrate_wrong_schema(self, migrator: DirectMigrator):
        """Test migration fails if not mem0 schema."""
        with patch.object(migrator.detector, "detect") as mock_detect:
            mock_detect.return_value = SchemaInfo(schema_type=SchemaType.CORTEX)

            report = await migrator.migrate()

            assert report.success is False
            assert "Expected mem0 schema" in report.error_details[0]

    @pytest.mark.asyncio
    async def test_migrate_dry_run(self, migrator: DirectMigrator):
        """Test dry run doesn't make changes."""
        with patch.object(migrator.detector, "detect") as mock_detect:
            mock_detect.return_value = SchemaInfo(
                schema_type=SchemaType.MEM0,
                memory_count=100,
                user_count=5,
            )

            with patch.object(migrator.detector, "get_mem0_stats") as mock_stats:
                mock_stats.return_value = {
                    "total_memories": 100,
                    "users": ["user1", "user2"],
                    "memories_by_user": {"user1": 60, "user2": 40},
                    "missing_content": 5,
                }

                report = await migrator.migrate(dry_run=True)

                assert report.success is True
                assert report.migrated == 95  # 100 - 5 missing
                assert report.skipped == 5
                assert report.users_migrated == ["user1", "user2"]

    @pytest.mark.asyncio
    async def test_migrate_full(self, migrator: DirectMigrator, mock_pool):
        """Test full migration flow."""
        _, conn = mock_pool

        with patch.object(migrator.detector, "detect") as mock_detect:
            mock_detect.return_value = SchemaInfo(
                schema_type=SchemaType.MEM0,
                memory_count=100,
            )

            # Mock backup table name query
            conn.fetchval.side_effect = [
                "memories_mem0_backup_20240101_120000",  # backup table query
                10,  # skipped count
            ]

            # Mock users query
            conn.fetch.return_value = [
                {"user_id": "user1"},
                {"user_id": "user2"},
            ]

            # Mock INSERT result
            conn.execute.return_value = "INSERT 0 90"

            report = await migrator.migrate(dry_run=False)

            assert report.success is True
            assert report.backup_table is not None
            assert report.users_migrated == ["user1", "user2"]

    @pytest.mark.asyncio
    async def test_backup_table_creation(self, migrator: DirectMigrator, mock_pool):
        """Test backup table is created with timestamp."""
        _, conn = mock_pool

        backup_name = await migrator._backup_mem0_table(conn)

        assert backup_name.startswith("memories_mem0_backup_")
        conn.execute.assert_called()

    @pytest.mark.asyncio
    async def test_validate_migration(self, migrator: DirectMigrator, mock_pool):
        """Test migration validation."""
        _, conn = mock_pool

        conn.fetchval.side_effect = [
            True,   # cortex table exists
            True,   # backup exists
            90,     # cortex count
            "memories_mem0_backup_test",  # backup table name
            90,     # expected count from backup
            True,   # index exists
            "some-uuid",  # vector query
        ]

        validation = await migrator.validate_migration()

        assert validation["valid"] is True
        assert validation["checks"]["cortex_table_exists"] is True
        assert validation["checks"]["backup_exists"] is True
        assert validation["checks"]["cortex_count"] == 90

    @pytest.mark.asyncio
    async def test_validate_migration_count_mismatch(self, migrator: DirectMigrator, mock_pool):
        """Test validation detects count mismatch."""
        _, conn = mock_pool

        conn.fetchval.side_effect = [
            True,   # cortex table exists
            True,   # backup exists
            50,     # cortex count (less than expected)
            "memories_mem0_backup_test",  # backup table name
            100,    # expected count from backup
            True,   # index exists
            "some-uuid",  # vector query
        ]

        validation = await migrator.validate_migration()

        assert validation["valid"] is False
        assert any("mismatch" in issue.lower() for issue in validation["issues"])

    @pytest.mark.asyncio
    async def test_rollback(self, migrator: DirectMigrator, mock_pool):
        """Test rollback restores backup."""
        _, conn = mock_pool

        conn.fetchval.return_value = "memories_mem0_backup_test"

        result = await migrator.rollback()

        assert result is True
        # Should drop memories and rename backup
        assert conn.execute.call_count >= 2

    @pytest.mark.asyncio
    async def test_rollback_no_backup(self, migrator: DirectMigrator, mock_pool):
        """Test rollback fails if no backup exists."""
        _, conn = mock_pool

        conn.fetchval.return_value = None  # No backup found

        result = await migrator.rollback()

        assert result is False


class TestMigrationConfigDefaults:
    """Test MigrationConfig default values."""

    def test_defaults(self):
        """Test default configuration values."""
        config = MigrationConfig()

        assert config.auto_migrate_mem0 is False
        assert config.backup_before_migrate is True
        assert config.classify_on_migrate is True
        assert config.batch_size == 100
        assert config.preserve_mem0_ids is True
        assert config.default_user_id == "default"
        assert config.reembed_on_dimension_mismatch is True

    def test_env_prefix(self):
        """Test environment variable prefix."""
        assert MigrationConfig.model_config.get("env_prefix") == "CORTEX_MIGRATION_"
