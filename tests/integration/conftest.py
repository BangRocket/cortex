"""Fixtures for integration tests with real Redis and Postgres."""

from __future__ import annotations

import asyncio
import os
from typing import AsyncGenerator

import pytest

from cortex.config import (
    CortexConfig,
    EmbeddingConfig,
    LLMConfig,
    PostgresConfig,
    RedisConfig,
)
from cortex.manager import MemoryManager
from cortex.stores.postgres_store import PostgresStore
from cortex.stores.redis_store import RedisStore
from cortex.utils.embedder import create_embedder
from cortex.utils.scorer import EmotionScorer


# Skip all integration tests if databases aren't available
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "integration: mark test as integration test")


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for session scope."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def redis_config() -> RedisConfig:
    """Redis config for tests."""
    return RedisConfig(
        host=os.getenv("CORTEX_REDIS_HOST", "localhost"),
        port=int(os.getenv("CORTEX_REDIS_PORT", "6379")),
        db=15,  # Use DB 15 for tests to avoid conflicts
        prefix="cortex_test",
    )


@pytest.fixture(scope="session")
def postgres_config() -> PostgresConfig:
    """Postgres config for tests."""
    return PostgresConfig(
        host=os.getenv("CORTEX_POSTGRES_HOST", "localhost"),
        port=int(os.getenv("CORTEX_POSTGRES_PORT", "5432")),
        user=os.getenv("CORTEX_POSTGRES_USER", "cortex"),
        password=os.getenv("CORTEX_POSTGRES_PASSWORD", "cortex"),
        database=os.getenv("CORTEX_POSTGRES_DATABASE", "cortex_test"),
        vector_dimensions=384,  # Use smaller dimensions for tests
    )


@pytest.fixture(scope="session")
def embedding_config() -> EmbeddingConfig:
    """Embedding config for tests - uses local embeddings if available."""
    return EmbeddingConfig(
        provider="local",
        local_model="all-MiniLM-L6-v2",
        dimensions=384,
    )


@pytest.fixture(scope="session")
def llm_config() -> LLMConfig:
    """LLM config for tests."""
    return LLMConfig(
        provider="openai",
        api_key=os.getenv("OPENAI_API_KEY", "test-key"),
        model="gpt-4o-mini",
    )


@pytest.fixture(scope="session")
async def redis_store(redis_config: RedisConfig) -> AsyncGenerator[RedisStore, None]:
    """Create Redis store for tests."""
    store = RedisStore(redis_config)
    try:
        await store.connect()
        yield store
    finally:
        # Cleanup test keys
        if store.client:
            pattern = f"{redis_config.prefix}:*"
            keys = []
            async for key in store.client.scan_iter(match=pattern):
                keys.append(key)
            if keys:
                await store.client.delete(*keys)
        await store.close()


@pytest.fixture(scope="session")
async def postgres_store(postgres_config: PostgresConfig) -> AsyncGenerator[PostgresStore, None]:
    """Create Postgres store for tests."""
    store = PostgresStore(postgres_config)
    try:
        await store.connect()
        await store.initialize_schema()
        yield store
    finally:
        # Cleanup test data
        if store.pool:
            async with store.pool.acquire() as conn:
                await conn.execute("DELETE FROM memories WHERE user_id LIKE 'test-%'")
                await conn.execute("DELETE FROM contradictions WHERE user_id LIKE 'test-%'")
                await conn.execute("DELETE FROM consolidation_logs WHERE user_id LIKE 'test-%'")
                await conn.execute("DELETE FROM project_members WHERE user_id LIKE 'test-%'")
                await conn.execute("DELETE FROM projects WHERE id LIKE 'test-%'")
        await store.close()


@pytest.fixture
async def clean_redis(redis_store: RedisStore) -> AsyncGenerator[RedisStore, None]:
    """Provide Redis store with cleanup after each test."""
    yield redis_store
    # Clean up test user data after each test
    if redis_store.client:
        pattern = f"{redis_store.config.prefix}:test-*"
        keys = []
        async for key in redis_store.client.scan_iter(match=pattern):
            keys.append(key)
        if keys:
            await redis_store.client.delete(*keys)


@pytest.fixture
async def clean_postgres(postgres_store: PostgresStore) -> AsyncGenerator[PostgresStore, None]:
    """Provide Postgres store with cleanup after each test."""
    yield postgres_store
    # Clean up test data after each test
    if postgres_store.pool:
        async with postgres_store.pool.acquire() as conn:
            await conn.execute("DELETE FROM memories WHERE user_id LIKE 'test-%'")


@pytest.fixture
def embedder(embedding_config: EmbeddingConfig):
    """Create embedder for tests."""
    return create_embedder(embedding_config)


@pytest.fixture
def scorer(llm_config: LLMConfig) -> EmotionScorer:
    """Create scorer for tests."""
    return EmotionScorer(llm_config)


@pytest.fixture
async def memory_manager(
    redis_store: RedisStore,
    postgres_store: PostgresStore,
    embedding_config: EmbeddingConfig,
    llm_config: LLMConfig,
) -> AsyncGenerator[MemoryManager, None]:
    """Create full memory manager for integration tests."""
    config = CortexConfig(
        embedding=embedding_config,
        llm=llm_config,
    )

    manager = MemoryManager(config)
    manager.redis = redis_store
    manager.postgres = postgres_store
    manager.embedder = create_embedder(embedding_config)
    manager.scorer = EmotionScorer(llm_config)

    yield manager
