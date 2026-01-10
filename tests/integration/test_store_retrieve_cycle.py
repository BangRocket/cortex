"""Integration tests for full store → retrieve cycle."""

from __future__ import annotations

import pytest

from cortex.models import Memory, MemoryType


pytestmark = pytest.mark.integration


class TestStoreRetrieveCycle:
    """Test complete memory storage and retrieval flow."""

    async def test_store_and_search_memory(
        self,
        clean_postgres,
        embedder,
    ):
        """Test storing a memory and finding it via semantic search."""
        user_id = "test-user-1"
        content = "I love programming in Python and building AI systems"

        # Generate embedding
        embedding = await embedder.embed(content)

        # Store memory
        memory = Memory(
            user_id=user_id,
            content=content,
            memory_type=MemoryType.SEMANTIC,
            embedding=embedding,
        )
        memory_id = await clean_postgres.store(memory)
        assert memory_id is not None

        # Search for related content
        query_embedding = await embedder.embed("What programming languages do I like?")
        results = await clean_postgres.search(
            user_id=user_id,
            embedding=query_embedding,
            limit=5,
        )

        assert len(results) >= 1
        assert any("Python" in m.content for m in results)

    async def test_store_multiple_and_retrieve_most_relevant(
        self,
        clean_postgres,
        embedder,
    ):
        """Test storing multiple memories and retrieving the most relevant."""
        user_id = "test-user-2"

        memories_content = [
            "I had pizza for dinner last night",
            "My favorite programming language is Rust",
            "I went to the gym this morning",
            "I'm learning about machine learning",
        ]

        # Store all memories
        for content in memories_content:
            embedding = await embedder.embed(content)
            memory = Memory(
                user_id=user_id,
                content=content,
                memory_type=MemoryType.EPISODIC,
                embedding=embedding,
            )
            await clean_postgres.store(memory)

        # Search for programming-related memories
        query_embedding = await embedder.embed("What do I know about programming?")
        results = await clean_postgres.search(
            user_id=user_id,
            embedding=query_embedding,
            limit=2,
        )

        assert len(results) >= 1
        # Rust or ML should be in top results
        contents = [m.content for m in results]
        assert any("Rust" in c or "learning" in c for c in contents)

    async def test_working_memory_with_ttl(
        self,
        clean_redis,
    ):
        """Test working memory TTL behavior."""
        user_id = "test-user-3"

        # Add working memory with short TTL
        memory = Memory(
            user_id=user_id,
            content="Currently discussing project architecture",
            memory_type=MemoryType.WORKING,
            emotional_score=0.3,
        )
        await clean_redis.add_working(user_id, memory, ttl=3600)

        # Retrieve working memories
        working = await clean_redis.get_working(user_id)
        assert len(working) == 1
        assert "architecture" in working[0].content

    async def test_identity_persistence(
        self,
        clean_redis,
    ):
        """Test identity storage and retrieval."""
        user_id = "test-user-4"

        identity = {
            "name": "Alice",
            "occupation": "Data Scientist",
            "preferences": {"theme": "dark", "language": "en"},
        }

        await clean_redis.set_identity(user_id, identity)
        retrieved = await clean_redis.get_identity(user_id)

        assert retrieved["name"] == "Alice"
        assert retrieved["occupation"] == "Data Scientist"
        assert retrieved["preferences"]["theme"] == "dark"

    async def test_session_lifecycle(
        self,
        clean_redis,
    ):
        """Test session creation, update, and clearing."""
        user_id = "test-user-5"

        # Create session
        await clean_redis.set_session(
            user_id,
            {"current_topic": "Memory Systems", "mood": "curious"},
            ttl=3600,
        )

        # Get session
        session = await clean_redis.get_session(user_id)
        assert session["current_topic"] == "Memory Systems"

        # Update session
        await clean_redis.update_session(
            user_id,
            {"mood": "excited"},
            ttl=3600,
        )
        session = await clean_redis.get_session(user_id)
        assert session["mood"] == "excited"

        # Clear session
        await clean_redis.clear_session(user_id)
        session = await clean_redis.get_session(user_id)
        assert session == {}

    async def test_memory_type_filtering(
        self,
        clean_postgres,
        embedder,
    ):
        """Test filtering memories by type during search."""
        user_id = "test-user-6"

        # Store different types
        for content, mem_type in [
            ("I learned Python at university", MemoryType.SEMANTIC),
            ("Had a meeting about Python project", MemoryType.EPISODIC),
            ("User prefers Python over Java", MemoryType.IDENTITY),
        ]:
            embedding = await embedder.embed(content)
            memory = Memory(
                user_id=user_id,
                content=content,
                memory_type=mem_type,
                embedding=embedding,
            )
            await clean_postgres.store(memory)

        # Search only semantic memories
        query_embedding = await embedder.embed("Python programming")
        results = await clean_postgres.search(
            user_id=user_id,
            embedding=query_embedding,
            memory_types=[MemoryType.SEMANTIC],
            limit=10,
        )

        assert len(results) >= 1
        assert all(m.memory_type == MemoryType.SEMANTIC for m in results)

    async def test_access_count_tracking(
        self,
        clean_postgres,
        embedder,
    ):
        """Test that access counts are incremented correctly."""
        user_id = "test-user-7"

        # Store memory
        embedding = await embedder.embed("Test memory for access tracking")
        memory = Memory(
            user_id=user_id,
            content="Test memory for access tracking",
            memory_type=MemoryType.SEMANTIC,
            embedding=embedding,
        )
        memory_id = await clean_postgres.store(memory)

        # Record accesses
        await clean_postgres.record_access(memory_id)
        await clean_postgres.record_access(memory_id)

        # Verify count
        retrieved = await clean_postgres.get(memory_id)
        assert retrieved is not None
        assert retrieved.access_count == 2
