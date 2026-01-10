"""Core data models for Cortex memory system."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """Types of memories in the system."""

    IDENTITY = "identity"      # Core facts about the user (name, family, job)
    SEMANTIC = "semantic"      # General knowledge or learned facts
    EPISODIC = "episodic"      # Events or things that happened
    WORKING = "working"        # Temporary, decays with TTL
    SESSION = "session"        # Current session state
    PROJECT = "project"        # Project-specific memories


class MemoryStatus(str, Enum):
    """Status of a memory."""

    ACTIVE = "active"          # Normal, retrievable
    ARCHIVED = "archived"      # Old, not retrieved by default
    SUPERSEDED = "superseded"  # Replaced by newer memory
    FLAGGED = "flagged"        # Marked for review (contradiction)


@dataclass
class Memory:
    """A single memory unit."""

    content: str
    user_id: str
    memory_type: MemoryType = MemoryType.EPISODIC

    # Identifiers
    id: str | None = None
    project_id: str | None = None

    # Scoring
    emotional_score: float = 0.5
    importance: float = 0.5
    confidence: float = 1.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime | None = None

    # Tracking
    access_count: int = 0
    supersedes: str | None = None  # ID of memory this supersedes

    # Metadata
    source: str = "conversation"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Vector embedding (optional, populated when needed)
    embedding: list[float] | None = None

    # Status
    status: MemoryStatus = MemoryStatus.ACTIVE

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = str(uuid.uuid4())


@dataclass
class MemoryContext:
    """Complete context returned by get_context() - everything needed for a turn."""

    user_id: str
    identity: dict[str, Any]           # Core facts, always loaded
    session: dict[str, Any]            # Current session state
    working: list[Memory]              # Recent working memories
    retrieved: list[Memory]            # Semantically retrieved
    project: dict[str, Any] | None = None  # Project-specific context

    def to_prompt_string(self) -> str:
        """Format for inclusion in system prompt."""
        sections: list[str] = []

        if self.identity:
            sections.append("## User Identity")
            for k, v in self.identity.items():
                if k != "updated_at":
                    sections.append(f"- {k}: {v}")

        if self.session:
            sections.append("\n## Current Session")
            for k, v in self.session.items():
                if k not in ("started_at", "last_active"):
                    sections.append(f"- {k}: {v}")

        if self.working:
            sections.append("\n## Recent Context")
            for m in self.working[:10]:
                sections.append(f"- {m.content}")

        if self.retrieved:
            sections.append("\n## Relevant Memories")
            for m in self.retrieved[:15]:
                sections.append(f"- {m.content}")

        if self.project:
            sections.append("\n## Project Context")
            for k, v in self.project.items():
                sections.append(f"- {k}: {v}")

        return "\n".join(sections)


@dataclass
class StoreResult:
    """Result from store() operations."""

    success: bool
    memory_id: str | None = None
    error: str | None = None
    emotional_score: float | None = None
    ttl: int | None = None


@dataclass
class SearchResult:
    """Result from search() operations."""

    memories: list[Memory]
    total_count: int
    search_time_ms: int


@dataclass
class Pattern:
    """A pattern extracted from memories during consolidation."""

    category: str  # identity, preference, behavior, relationship
    fact: str
    confidence: float
    evidence: list[int] = field(default_factory=list)


@dataclass
class Contradiction:
    """A detected contradiction between memories."""

    memory_a: str  # Memory ID
    memory_b: str  # Memory ID
    reason: str
    id: str | None = None
    resolution: str | None = None  # a_wins, b_wins, merge, both_valid

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = str(uuid.uuid4())


@dataclass
class ConsolidationLog:
    """Log of a consolidation run."""

    user_id: str
    started_at: datetime
    completed_at: datetime | None = None

    # Stats
    memories_processed: int = 0
    patterns_found: int = 0
    identities_updated: int = 0
    contradictions_found: int = 0
    memories_compacted: int = 0

    # Result
    success: bool = False
    error: str | None = None
    duration_ms: int = 0


@dataclass
class ProjectContext:
    """Context for a shared project."""

    project_id: str
    name: str | None = None
    recent: list[Memory] = field(default_factory=list)
    retrieved: list[Memory] = field(default_factory=list)


@dataclass
class GroupContext:
    """Context for group chat messages."""

    personal: MemoryContext
    project: ProjectContext | None = None
    channel_id: str | None = None
