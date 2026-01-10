"""Unit tests for token budget management (AfterImage pattern)."""

from __future__ import annotations

import pytest

from cortex.config import TokenBudgetConfig
from cortex.models import Memory, MemoryType
from cortex.utils.budget import (
    TokenBudgetManager,
    estimate_tokens,
    format_memory_for_context,
)


class TestEstimateTokens:
    """Test token estimation."""

    def test_estimate_basic(self):
        """Test basic token estimation."""
        text = "Hello world"  # 11 chars
        tokens = estimate_tokens(text, tokens_per_char=0.25)
        assert tokens == 2  # 11 * 0.25 = 2.75, truncated to 2

    def test_estimate_empty(self):
        """Test empty string."""
        assert estimate_tokens("") == 0

    def test_estimate_long_text(self):
        """Test longer text estimation."""
        text = "a" * 1000
        tokens = estimate_tokens(text, tokens_per_char=0.25)
        assert tokens == 250


class TestFormatMemoryForContext:
    """Test memory formatting."""

    def test_format_full(self):
        """Test full memory formatting."""
        memory = Memory(
            content="This is a test memory",
            user_id="user1",
        )
        result = format_memory_for_context(memory, full=True)
        assert result == "- This is a test memory"

    def test_format_truncated(self):
        """Test truncated memory formatting."""
        long_content = "x" * 200
        memory = Memory(
            content=long_content,
            user_id="user1",
        )
        result = format_memory_for_context(memory, full=False)
        assert len(result) < len(long_content)
        assert result.endswith("...")


class TestTokenBudgetManager:
    """Test TokenBudgetManager."""

    @pytest.fixture
    def config(self) -> TokenBudgetConfig:
        """Create test config."""
        return TokenBudgetConfig(
            max_context_tokens=1000,
            full_memory_count=3,
            summary_target_tokens=200,
            enable_summarization=True,
            tokens_per_char=0.25,
        )

    @pytest.fixture
    def manager(self, config: TokenBudgetConfig) -> TokenBudgetManager:
        """Create manager instance."""
        return TokenBudgetManager(config)

    @pytest.fixture
    def small_memories(self) -> list[Memory]:
        """Create small test memories."""
        return [
            Memory(
                content=f"Small memory {i}",
                user_id="user1",
                memory_type=MemoryType.EPISODIC,
            )
            for i in range(5)
        ]

    @pytest.fixture
    def large_memories(self) -> list[Memory]:
        """Create large test memories that exceed budget."""
        return [
            Memory(
                content="x" * 500,  # ~125 tokens each
                user_id="user1",
                memory_type=MemoryType.EPISODIC,
            )
            for i in range(20)  # 20 * 125 = 2500 tokens, over 1000 budget
        ]

    def test_count_tokens(self, manager: TokenBudgetManager):
        """Test token counting."""
        text = "Hello world"
        tokens = manager.count_tokens(text)
        assert tokens == estimate_tokens(text, 0.25)

    def test_count_memory_tokens(self, manager: TokenBudgetManager, small_memories: list[Memory]):
        """Test counting tokens for a memory."""
        tokens = manager.count_memory_tokens(small_memories[0])
        assert tokens > 0

    def test_count_total_tokens(self, manager: TokenBudgetManager, small_memories: list[Memory]):
        """Test counting total tokens."""
        total = manager.count_total_tokens(small_memories)
        assert total > 0
        assert total == sum(manager.count_memory_tokens(m) for m in small_memories)

    def test_is_under_budget(self, manager: TokenBudgetManager, small_memories: list[Memory]):
        """Test budget check for small memories."""
        assert not manager.is_over_budget(small_memories)

    def test_is_over_budget(self, manager: TokenBudgetManager, large_memories: list[Memory]):
        """Test budget check for large memories."""
        assert manager.is_over_budget(large_memories)

    def test_apply_budget_under_limit(self, manager: TokenBudgetManager, small_memories: list[Memory]):
        """Test applying budget when under limit."""
        full, summary = manager.apply_budget(small_memories)
        assert full == small_memories
        assert summary is None

    def test_apply_budget_over_limit(self, manager: TokenBudgetManager, large_memories: list[Memory]):
        """Test applying budget when over limit."""
        full, summary = manager.apply_budget(large_memories)

        # Should keep only full_memory_count
        assert len(full) == 3
        assert full == large_memories[:3]

        # Should have a summary
        assert summary is not None
        assert "[Summary" in summary

    def test_apply_budget_disabled(self, config: TokenBudgetConfig, large_memories: list[Memory]):
        """Test that summarization can be disabled."""
        config.enable_summarization = False
        manager = TokenBudgetManager(config)

        full, summary = manager.apply_budget(large_memories)

        # Should return all memories unchanged
        assert full == large_memories
        assert summary is None

    def test_format_context_with_budget(self, manager: TokenBudgetManager):
        """Test full context formatting with budget."""
        identity = {"name": "Test User", "job": "Developer"}
        session = {"topic": "testing"}
        working = [
            Memory(content="Working memory 1", user_id="user1"),
        ]
        retrieved = [
            Memory(content="Retrieved memory 1", user_id="user1"),
            Memory(content="Retrieved memory 2", user_id="user1"),
        ]

        result = manager.format_context_with_budget(
            identity=identity,
            session=session,
            working=working,
            retrieved=retrieved,
        )

        assert "## User Identity" in result
        assert "name: Test User" in result
        assert "## Current Session" in result
        assert "## Relevant Context" in result

    def test_create_summary_groups_by_type(self, manager: TokenBudgetManager):
        """Test that summary groups memories by type."""
        memories = [
            Memory(content="Episodic 1", user_id="user1", memory_type=MemoryType.EPISODIC),
            Memory(content="Episodic 2", user_id="user1", memory_type=MemoryType.EPISODIC),
            Memory(content="Semantic 1", user_id="user1", memory_type=MemoryType.SEMANTIC),
        ]

        summary = manager._create_summary(memories)

        assert "episodic" in summary.lower()
        assert "semantic" in summary.lower()

    def test_create_summary_truncates_long_lists(self, manager: TokenBudgetManager):
        """Test that summary truncates long lists of memories."""
        memories = [
            Memory(content=f"Episodic {i}", user_id="user1", memory_type=MemoryType.EPISODIC)
            for i in range(10)
        ]

        summary = manager._create_summary(memories)

        # Should mention "+X more"
        assert "more" in summary.lower()
