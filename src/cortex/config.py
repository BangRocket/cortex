"""Configuration classes for Cortex."""

from __future__ import annotations

from pydantic import Field, model_validator
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

    @model_validator(mode="after")
    def validate_weights_sum_to_one(self) -> "RetrievalConfig":
        """Ensure retrieval weights sum to 1.0."""
        total = (
            self.similarity_weight
            + self.recency_weight
            + self.emotion_weight
            + self.reinforcement_weight
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Retrieval weights must sum to 1.0, got {total:.3f}")
        return self


class TokenBudgetConfig(BaseSettings):
    """Configuration for token budget management (AfterImage pattern)."""

    model_config = SettingsConfigDict(env_prefix="CORTEX_BUDGET_")

    # Maximum tokens for context injection
    max_context_tokens: int = 4000

    # How many top memories to keep full (rest get summarized)
    full_memory_count: int = 5

    # Target summary length in tokens
    summary_target_tokens: int = 500

    # Enable/disable automatic summarization
    enable_summarization: bool = True

    # Approximate tokens per character (for estimation)
    tokens_per_char: float = 0.25


class ChurnConfig(BaseSettings):
    """Configuration for churn detection and reinforcement (AfterImage pattern)."""

    model_config = SettingsConfigDict(env_prefix="CORTEX_CHURN_")

    # Access count threshold to consider "high churn"
    churn_threshold: int = 10

    # Importance boost for high-churn memories (added to base importance)
    importance_boost: float = 0.2

    # Access count threshold to auto-promote to identity
    identity_promotion_threshold: int = 25


class CacheConfig(BaseSettings):
    """Configuration for in-memory LRU cache (AfterImage pattern)."""

    model_config = SettingsConfigDict(env_prefix="CORTEX_CACHE_")

    # Enable in-memory cache layer
    enabled: bool = True

    # Max entries in identity cache
    identity_cache_size: int = 100

    # Max entries in context cache
    context_cache_size: int = 50

    # Cache TTL in seconds (5 minutes)
    cache_ttl: int = 300


class Neo4jConfig(BaseSettings):
    """Neo4j connection configuration for graph memory."""

    model_config = SettingsConfigDict(env_prefix="CORTEX_NEO4J_")

    # Connection
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    database: str = "neo4j"

    # Connection pool
    max_connection_pool_size: int = 50
    connection_timeout: float = 30.0


class GraphConfig(BaseSettings):
    """Configuration for graph memory (entity/relationship extraction)."""

    model_config = SettingsConfigDict(env_prefix="CORTEX_GRAPH_")

    # Enable graph memory
    enabled: bool = True

    # Entity extraction
    extract_entities: bool = True
    min_entity_confidence: float = 0.7

    # Relationship inference
    infer_relationships: bool = True
    min_relationship_confidence: float = 0.6

    # Graph traversal limits
    max_hop_depth: int = 2
    max_related_entities: int = 20

    # Entity types to extract
    entity_types: list[str] = [
        "person",
        "organization",
        "location",
        "project",
        "concept",
        "event",
    ]


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

    # AfterImage-inspired features
    token_budget: TokenBudgetConfig = Field(default_factory=TokenBudgetConfig)
    churn: ChurnConfig = Field(default_factory=ChurnConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    # Graph memory (mem0-style)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)

    # Feature flags
    enable_consolidation: bool = True
    enable_emotional_scoring: bool = True
    enable_rejection_pipeline: bool = True
    enable_graph_memory: bool = True

    # Debug
    debug: bool = False
    log_level: str = "INFO"
