"""Migration utilities for Cortex - importing from mem0."""

from cortex.migration.direct_migration import (
    DirectMigrator,
    Mem0MigrationError,
    Mem0SchemaDetectedError,
    MigrationReport,
    migrate_mem0_to_cortex,
)
from cortex.migration.mem0_migration import Mem0Migrator
from cortex.migration.schema_detector import (
    SchemaDetector,
    SchemaInfo,
    SchemaType,
    detect_schema,
)

__all__ = [
    # Schema detection
    "SchemaDetector",
    "SchemaInfo",
    "SchemaType",
    "detect_schema",
    # Direct migration
    "DirectMigrator",
    "Mem0MigrationError",
    "Mem0SchemaDetectedError",
    "MigrationReport",
    "migrate_mem0_to_cortex",
    # API-based migration (legacy)
    "Mem0Migrator",
]
