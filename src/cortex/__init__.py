"""Cortex - Tiered memory system for AI assistants.

A tiered memory architecture with:
- Hot layer (Redis): Always-loaded identity, session state, working memory with TTL
- Warm layer (Postgres + pgvector): Searchable long-term storage with embeddings
- Background consolidation: Pattern extraction, decay, deduplication

Usage:
    from cortex import MemoryManager, CortexConfig

    config = CortexConfig()
    manager = MemoryManager(config)
    await manager.initialize()

    # Get context for a conversation turn
    context = await manager.get_context(user_id, query="current topic")

    # Store a new memory
    await manager.store(user_id, "User mentioned they have a meeting tomorrow")
"""

from cortex.models import (
    Contradiction,
    ConsolidationLog,
    GroupContext,
    Memory,
    MemoryContext,
    MemoryStatus,
    MemoryType,
    Pattern,
    ProjectContext,
    SearchResult,
    StoreResult,
)
from cortex.manager import MemoryManager
from cortex.config import CortexConfig

__version__ = "0.1.0"

__all__ = [
    # Core
    "Memory",
    "MemoryContext",
    "MemoryManager",
    "MemoryStatus",
    "MemoryType",
    "SearchResult",
    "StoreResult",
    # Config
    "CortexConfig",
    # Additional models
    "Contradiction",
    "ConsolidationLog",
    "GroupContext",
    "Pattern",
    "ProjectContext",
]
