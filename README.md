# Cortex

A tiered memory system for AI assistants with emotional decay and consolidation.

## Overview

Cortex provides a sophisticated memory architecture inspired by human cognition, designed for AI assistants like Clara. It solves common problems with embedding-only memory systems:

- **"Lost in the middle" effect**: Older memories get buried regardless of importance
- **No temporal awareness**: Recent vs. old treated equally
- **No emotional weighting**: Mundane facts retrieved alongside significant events
- **Contradiction accumulation**: Conflicting facts coexist without resolution

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        YOUR APPLICATION                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    MemoryManager                         │    │
│  │  - get_context(user_id) → always-loaded + retrieved      │    │
│  │  - store(user_id, memory) → write-through                │    │
│  │  - search(user_id, query) → semantic search              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐    ┌───────────────────┐
│     REDIS     │    │    POSTGRES    │    │   CONSOLIDATOR    │
│   (Hot/Fast)  │    │  (Warm/Search) │    │   (Background)    │
├───────────────┤    ├────────────────┤    ├───────────────────┤
│ • Identity    │    │ • memories     │    │ • Pattern extract │
│ • Session     │    │ • embeddings   │    │ • Contradiction   │
│ • Working mem │    │ • full history │    │ • Compaction      │
└───────────────┘    └────────────────┘    └───────────────────┘
```

### Key Innovations

- **Tiered Storage**: Redis for hot/instant access, Postgres+pgvector for searchable long-term storage
- **Emotional Decay**: Working memory TTL scales with emotional intensity (30min → 6hrs)
- **Always-Loaded Context**: Identity and session state loaded every turn, not retrieved
- **Background Consolidation**: Pattern extraction, contradiction detection, memory compaction
- **Multi-User Support**: Project sharing, group chat handling, rejection pipeline

## Installation

```bash
pip install cortex
```

Or install from source:

```bash
git clone https://github.com/joshheidorn/cortex.git
cd cortex
pip install -e ".[dev]"
```

## Requirements

- Python 3.11+
- Redis 7+
- PostgreSQL 15+ with pgvector extension
- OpenAI API key (for embeddings and scoring)

## Quick Start

```python
import asyncio
from cortex import CortexConfig, MemoryManager, MemoryType

async def main():
    config = CortexConfig()
    manager = MemoryManager(config)
    await manager.initialize()

    user_id = "user-123"

    # Set identity (always loaded, no retrieval needed)
    await manager.set_identity(user_id, {
        "name": "Josh",
        "occupation": "software engineer",
        "family": {"children": ["Maddie", "Anne", "Thomas"]},
    })

    # Store a memory (write-through to Redis + Postgres)
    result = await manager.store(
        user_id,
        "Had an important meeting about the new architecture",
        memory_type=MemoryType.EPISODIC,
    )
    print(f"Stored with emotion score: {result.emotional_score}")
    print(f"TTL: {result.ttl}s")  # Higher emotion = longer TTL

    # Get context for a conversation turn
    context = await manager.get_context(
        user_id,
        query="architecture decisions",
    )

    # Use in your system prompt
    print(context.to_prompt_string())

    await manager.close()

asyncio.run(main())
```

## Configuration

Configure via environment variables:

```bash
# Redis
CORTEX_REDIS_HOST=localhost
CORTEX_REDIS_PORT=6379
CORTEX_REDIS_PASSWORD=

# Postgres
CORTEX_POSTGRES_HOST=localhost
CORTEX_POSTGRES_PORT=5432
CORTEX_POSTGRES_USER=postgres
CORTEX_POSTGRES_PASSWORD=
CORTEX_POSTGRES_DATABASE=cortex

# Embeddings (OpenAI)
CORTEX_EMBEDDING_API_KEY=sk-...
CORTEX_EMBEDDING_MODEL=text-embedding-3-small

# LLM for scoring/classification
CORTEX_LLM_API_KEY=sk-...
CORTEX_LLM_MODEL=gpt-4o-mini

# Features
CORTEX_ENABLE_CONSOLIDATION=true
CORTEX_ENABLE_EMOTIONAL_SCORING=true
```

## CLI

```bash
# Initialize database schema
cortex init-db

# Check health
cortex health

# Get user identity
cortex get-identity user-123

# Search memories
cortex search user-123 "project architecture"

# Store a memory
cortex store user-123 "Learned about vector databases" --memory-type semantic

# Run consolidation for a user
cortex consolidate user-123

# Migrate from mem0
cortex migrate user-123
```

## Memory Types

| Type | Description | Example |
|------|-------------|---------|
| `identity` | Core facts, always loaded | "User's name is Josh" |
| `semantic` | Learned facts/knowledge | "Prefers direct communication" |
| `episodic` | Events that happened | "Had a meeting on Jan 5" |
| `working` | Temporary, decays with TTL | "Currently discussing architecture" |
| `session` | Current session state | "Topic: memory systems" |
| `project` | Project-specific context | "Cortex uses Redis + Postgres" |

## Emotional Decay

Working memory TTL scales with emotional intensity:

| Emotion Score | Example | TTL |
|---------------|---------|-----|
| 0.0 - 0.2 | "Grabbed coffee" | 30 min |
| 0.2 - 0.4 | "Good meeting" | 1.5 hours |
| 0.4 - 0.6 | "Finished project" | 3 hours |
| 0.6 - 0.8 | "Family emergency" | 5 hours |
| 0.8 - 1.0 | "Major life event" | 6+ hours |

## Retrieval Scoring

Memories are ranked by combining multiple signals:

- **Similarity** (40%): Semantic relevance from vector search
- **Recency** (25%): Temporal decay
- **Emotion** (20%): Emotional importance
- **Reinforcement** (15%): Access patterns

## Background Consolidation

Runs periodically (default: every 6 hours) to:

1. **Extract Patterns**: Identify stable facts from episodic memories
2. **Detect Contradictions**: Find conflicting facts using embedding clustering
3. **Compact Memories**: Summarize old episodes into semantic memories
4. **Update Identity**: Promote high-confidence patterns to identity

## Migration from mem0

```python
from cortex.migration import Mem0Migrator

migrator = Mem0Migrator(
    redis_store=redis,
    postgres_store=postgres,
    embedder=embedder,
    scorer=scorer,
)

report = await migrator.migrate_user("user-123")
print(f"Imported: {report.imported}, Skipped: {report.skipped}")
```

## Multi-User Support

```python
from cortex.multiuser import ProjectMemory, GroupChatMemory

# Create a shared project
project_memory = ProjectMemory(postgres, redis)
await project_memory.create_project(
    project_id="cortex",
    name="Cortex Memory System",
    owner_id="user-123",
    members=["user-123", "user-456"],
)

# Handle group chat with rejection pipeline
group_handler = GroupChatMemory(manager)
context = await group_handler.get_context_for_message(
    message="What's the status of the Redis integration?",
    author_id="user-123",
    channel_id="channel-789",
    mentioned=True,
)
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
ruff check src/

# Type checking
mypy src/
```

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by [Manus](https://manus.im) "context as RAM" pattern
- [HuixiangDou](https://github.com/InternLM/HuixiangDou) for rejection pipeline patterns
- [mem0](https://github.com/mem0ai/mem0) for the foundation we're building upon
