"""Configuration classes for Cortex."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisConfig(BaseSettings):
    """Redis connection configuration."""

    model_config = SettingsConfigDict(env_prefix="CORTEX_REDIS_")

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    ssl: bool = False

    # Connection pool
    max_connections: int = 20

    # Key prefix
    prefix: str = "cortex"


class PostgresConfig(BaseSettings):
    """Postgres connection configuration."""

    model_config = SettingsConfigDict(env_prefix="CORTEX_POSTGRES_")

    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    database: str = "cortex"

    # Connection pool
    min_pool_size: int = 5
    max_pool_size: int = 20

    # Vector dimensions (must match embedding model)
    vector_dimensions: int = 1536


class EmbeddingConfig(BaseSettings):
    """Embedding generation configuration."""

    model_config = SettingsConfigDict(env_prefix="CORTEX_EMBEDDING_")

    provider: str = "openai"  # openai, local
    api_key: str | None = None
    model: str = "text-embedding-3-small"
    dimensions: int = 1536

    # For local embeddings
    local_model: str = "all-MiniLM-L6-v2"


class LLMConfig(BaseSettings):
    """LLM configuration for scoring and classification."""

    model_config = SettingsConfigDict(env_prefix="CORTEX_LLM_")

    provider: str = "openai"  # openai, anthropic
    api_key: str | None = None
    model: str = "gpt-4o-mini"

    # For scoring
    temperature: float = 0.0
    max_tokens: int = 100


class ConsolidationConfig(BaseSettings):
    """Configuration for background consolidation jobs."""

    model_config = SettingsConfigDict(env_prefix="CORTEX_CONSOLIDATION_")

    # Scheduling
    pattern_interval_hours: int = 6
    cleanup_interval_hours: int = 1

    # Pattern extraction
    pattern_min_memories: int = 5
    pattern_confidence_threshold: float = 0.8
    pattern_lookback_days: int = 7

    # Compaction
    compact_after_days: int = 30
    compact_min_memories: int = 10

    # Contradiction detection
    similarity_threshold: float = 0.8

    # Limits
    max_memories_per_run: int = 500
    max_concurrent_users: int = 5


class TTLConfig(BaseSettings):
    """TTL configuration for working memory decay."""

    model_config = SettingsConfigDict(env_prefix="CORTEX_TTL_")

    # Base TTL in seconds (30 minutes)
    base_ttl: int = 1800

    # Max TTL multiplier (emotion=1.0 gives 12x = 6 hours)
    max_multiplier: float = 12.0

    # Session TTL (24 hours)
    session_ttl: int = 86400


class RetrievalConfig(BaseSettings):
    """Configuration for retrieval scoring weights."""

    model_config = SettingsConfigDict(env_prefix="CORTEX_RETRIEVAL_")

    # Weights must sum to 1.0
    similarity_weight: float = 0.40
    recency_weight: float = 0.25
    emotion_weight: float = 0.20
    reinforcement_weight: float = 0.15

    # Limits
    default_limit: int = 20
    max_limit: int = 100


class CortexConfig(BaseSettings):
    """Main configuration for Cortex memory system."""

    model_config = SettingsConfigDict(
        env_prefix="CORTEX_",
        env_nested_delimiter="__",
    )

    # Sub-configs
    redis: RedisConfig = Field(default_factory=RedisConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    ttl: TTLConfig = Field(default_factory=TTLConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    # Feature flags
    enable_consolidation: bool = True
    enable_emotional_scoring: bool = True
    enable_rejection_pipeline: bool = True

    # Debug
    debug: bool = False
    log_level: str = "INFO"
