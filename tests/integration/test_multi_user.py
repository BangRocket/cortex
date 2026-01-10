"""Integration tests for multi-user support."""

from __future__ import annotations

import pytest

from cortex.models import Memory, MemoryType
from cortex.multiuser import ProjectMemory, RejectionPipeline, GroupChatMemory


pytestmark = pytest.mark.integration


class TestUserIsolation:
    """Test that user data is properly isolated."""

    async def test_memories_isolated_between_users(
        self,
        clean_postgres,
        embedder,
    ):
        """Test that one user cannot see another's memories."""
        user1 = "test-iso-user-1"
        user2 = "test-iso-user-2"

        # Store memory for user1
        embedding = await embedder.embed("User 1's private memory")
        memory = Memory(
            user_id=user1,
            content="User 1's private memory about secret project",
            memory_type=MemoryType.SEMANTIC,
            embedding=embedding,
        )
        await clean_postgres.store(memory)

        # Store memory for user2
        embedding = await embedder.embed("User 2's private memory")
        memory = Memory(
            user_id=user2,
            content="User 2's private memory about different project",
            memory_type=MemoryType.SEMANTIC,
            embedding=embedding,
        )
        await clean_postgres.store(memory)

        # Search as user1
        query_embedding = await embedder.embed("secret project")
        results = await clean_postgres.search(
            user_id=user1,
            embedding=query_embedding,
            limit=10,
        )

        # Should only see user1's memories
        assert all(m.user_id == user1 for m in results)
        assert any("User 1" in m.content for m in results)

    async def test_identity_isolated_between_users(
        self,
        clean_redis,
    ):
        """Test that identity data is isolated between users."""
        user1 = "test-iso-ident-1"
        user2 = "test-iso-ident-2"

        await clean_redis.set_identity(user1, {"name": "Alice", "role": "admin"})
        await clean_redis.set_identity(user2, {"name": "Bob", "role": "user"})

        identity1 = await clean_redis.get_identity(user1)
        identity2 = await clean_redis.get_identity(user2)

        assert identity1["name"] == "Alice"
        assert identity2["name"] == "Bob"
        assert identity1["role"] != identity2["role"]


class TestProjectMemory:
    """Test project-level shared memory."""

    async def test_create_project_and_add_members(
        self,
        clean_postgres,
        clean_redis,
    ):
        """Test creating a project with members."""
        project_memory = ProjectMemory(clean_postgres, clean_redis)

        await project_memory.create_project(
            project_id="test-project-1",
            name="Test Project",
            owner_id="test-owner",
            members=["test-owner", "test-member-1"],
        )

        # Check access
        has_access = await project_memory.check_access("test-project-1", "test-owner")
        assert has_access

        has_access = await project_memory.check_access("test-project-1", "test-member-1")
        assert has_access

        has_access = await project_memory.check_access("test-project-1", "unauthorized-user")
        assert not has_access

    async def test_store_and_retrieve_project_memory(
        self,
        clean_postgres,
        clean_redis,
        embedder,
    ):
        """Test storing and retrieving project-scoped memories."""
        project_memory = ProjectMemory(clean_postgres, clean_redis)

        await project_memory.create_project(
            project_id="test-project-2",
            name="ML Project",
            owner_id="test-ml-owner",
        )

        # Store project memory
        memory_id = await project_memory.store_project_memory(
            project_id="test-project-2",
            content="Decided to use PyTorch for the ML pipeline",
            author_id="test-ml-owner",
            embedder=embedder,
        )
        assert memory_id is not None

        # Retrieve project context
        context = await project_memory.get_project_context(
            project_id="test-project-2",
            user_id="test-ml-owner",
        )
        assert len(context.recent) >= 1


class TestRejectionPipeline:
    """Test message rejection logic."""

    def test_quick_reject_trivial_messages(self):
        """Test that trivial messages are rejected."""
        pipeline = RejectionPipeline()

        # Should reject
        assert pipeline.quick_reject("hi")
        assert pipeline.quick_reject("hello!")
        assert pipeline.quick_reject("ok")
        assert pipeline.quick_reject("thanks")
        assert pipeline.quick_reject("lol")
        assert pipeline.quick_reject("  ")

        # Should not reject
        assert not pipeline.quick_reject("What's the status of the project?")
        assert not pipeline.quick_reject("Can you help me with this code?")
        assert not pipeline.quick_reject("I need to update the database schema")

    async def test_should_retrieve_respects_rejection(self):
        """Test should_retrieve applies fast rejection."""
        pipeline = RejectionPipeline(enable_llm_check=False)

        # Trivial messages should not trigger retrieval
        should = await pipeline.should_retrieve("hi", {})
        assert not should

        # Substantive messages should trigger retrieval
        should = await pipeline.should_retrieve("What did we discuss yesterday?", {})
        assert should

    async def test_should_respond_with_context(
        self,
        clean_redis,
        clean_postgres,
    ):
        """Test should_respond considers context."""
        from cortex.models import MemoryContext

        pipeline = RejectionPipeline(response_threshold=0.5)

        # With empty context, should not respond unless mentioned
        context = MemoryContext(
            user_id="test-user",
            identity={},
            session={},
            working=[],
            retrieved=[],
        )

        should = await pipeline.should_respond(context, "What's happening?", mentioned=False)
        assert not should  # No context, not mentioned

        should = await pipeline.should_respond(context, "What's happening?", mentioned=True)
        assert should  # Mentioned, always respond

        # With working memory context, should respond
        context.working = [
            Memory(
                user_id="test-user",
                content="Discussing project plans",
                memory_type=MemoryType.WORKING,
            )
        ]
        should = await pipeline.should_respond(context, "Tell me more", mentioned=False)
        assert should  # Has context
