"""Token budget management utilities (AfterImage pattern).

This module handles intelligent context injection by:
1. Tracking token counts for memories
2. Summarizing lower-scored memories when over budget
3. Ensuring top N memories remain full while others are summarized
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from cortex.config import TokenBudgetConfig
    from cortex.models import Memory

logger = structlog.get_logger(__name__)


def estimate_tokens(text: str, tokens_per_char: float = 0.25) -> int:
    """
    Estimate token count for text.

    Uses a simple character-based estimation. For production,
    consider using tiktoken for exact counts.

    Args:
        text: The text to estimate
        tokens_per_char: Approximate tokens per character (default ~4 chars/token)

    Returns:
        Estimated token count
    """
    return int(len(text) * tokens_per_char)


def format_memory_for_context(memory: "Memory", full: bool = True) -> str:
    """
    Format a memory for context injection.

    Args:
        memory: The memory to format
        full: If True, include full content; if False, truncate

    Returns:
        Formatted string
    """
    if full:
        return f"- {memory.content}"

    # Truncated version: first 100 chars + ellipsis
    truncated = memory.content[:100]
    if len(memory.content) > 100:
        truncated += "..."
    return f"- {truncated}"


class TokenBudgetManager:
    """
    Manages token budget for context injection.

    When retrieved memories exceed the token budget, this class
    automatically summarizes lower-scored memories while keeping
    the top N memories at full fidelity.
    """

    def __init__(self, config: "TokenBudgetConfig") -> None:
        self.config = config
        self._summarizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return estimate_tokens(text, self.config.tokens_per_char)

    def count_memory_tokens(self, memory: "Memory") -> int:
        """Count tokens for a single memory."""
        return self.count_tokens(memory.content)

    def count_total_tokens(self, memories: list["Memory"]) -> int:
        """Count total tokens for a list of memories."""
        return sum(self.count_memory_tokens(m) for m in memories)

    def is_over_budget(self, memories: list["Memory"]) -> bool:
        """Check if memories exceed token budget."""
        return self.count_total_tokens(memories) > self.config.max_context_tokens

    def apply_budget(
        self,
        memories: list["Memory"],
        identity: dict | None = None,
        session: dict | None = None,
    ) -> tuple[list["Memory"], str | None]:
        """
        Apply token budget to memories.

        If over budget:
        1. Keep top N memories at full fidelity
        2. Summarize the rest into a single summary block
        3. Return the summary separately

        Args:
            memories: List of memories (assumed already ranked)
            identity: Identity dict (contributes to budget)
            session: Session dict (contributes to budget)

        Returns:
            Tuple of (full_memories, summary_of_rest or None)
        """
        if not self.config.enable_summarization:
            return memories, None

        # Calculate overhead from identity and session
        overhead = 0
        if identity:
            overhead += self.count_tokens(str(identity))
        if session:
            overhead += self.count_tokens(str(session))

        available_tokens = self.config.max_context_tokens - overhead

        # Check if we're under budget
        total_memory_tokens = self.count_total_tokens(memories)
        if total_memory_tokens <= available_tokens:
            logger.debug(
                "budget_under_limit",
                total_tokens=total_memory_tokens + overhead,
                limit=self.config.max_context_tokens,
            )
            return memories, None

        # We're over budget - split memories
        full_count = min(self.config.full_memory_count, len(memories))
        full_memories = memories[:full_count]

        # Summarize the rest
        remaining = memories[full_count:]
        if not remaining:
            return full_memories, None

        summary = self._create_summary(remaining)

        logger.info(
            "budget_applied",
            original_count=len(memories),
            full_count=len(full_memories),
            summarized_count=len(remaining),
            original_tokens=total_memory_tokens,
            final_tokens=self.count_total_tokens(full_memories) + self.count_tokens(summary),
        )

        return full_memories, summary

    def _create_summary(self, memories: list["Memory"]) -> str:
        """
        Create a summary of memories.

        This is a simple extractive summary - for production,
        consider using an LLM for better summaries.
        """
        if not memories:
            return ""

        # Group by type for better organization
        by_type: dict[str, list[str]] = {}
        for mem in memories:
            type_key = mem.memory_type.value
            if type_key not in by_type:
                by_type[type_key] = []
            # Extract key phrase (first sentence or first 80 chars)
            key_phrase = mem.content.split(".")[0][:80]
            by_type[type_key].append(key_phrase)

        # Build summary
        summary_parts = ["[Summary of additional context:]"]
        for mem_type, phrases in by_type.items():
            if len(phrases) <= 3:
                summary_parts.append(f"  {mem_type}: {'; '.join(phrases)}")
            else:
                # Show first 2 and count
                shown = "; ".join(phrases[:2])
                summary_parts.append(f"  {mem_type}: {shown} (+{len(phrases) - 2} more)")

        return "\n".join(summary_parts)

    def format_context_with_budget(
        self,
        identity: dict | None,
        session: dict | None,
        working: list["Memory"],
        retrieved: list["Memory"],
    ) -> str:
        """
        Format full context respecting token budget.

        This is an alternative to MemoryContext.to_prompt_string()
        that applies budget management.
        """
        sections: list[str] = []

        # Identity (always full)
        if identity:
            sections.append("## User Identity")
            for k, v in identity.items():
                if k != "updated_at":
                    sections.append(f"- {k}: {v}")

        # Session (always full, usually small)
        if session:
            sections.append("\n## Current Session")
            for k, v in session.items():
                if k not in ("started_at", "last_active"):
                    sections.append(f"- {k}: {v}")

        # Apply budget to working + retrieved
        all_memories = working + retrieved

        if all_memories:
            full_memories, summary = self.apply_budget(
                all_memories, identity, session
            )

            if full_memories:
                sections.append("\n## Relevant Context")
                for mem in full_memories:
                    sections.append(format_memory_for_context(mem, full=True))

            if summary:
                sections.append(f"\n{summary}")

        return "\n".join(sections)
