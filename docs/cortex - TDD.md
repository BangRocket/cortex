# Cortex - Technical Design Document

## Document Index

| Doc | Section | Description |
|-----|---------|-------------|
| 01 | Executive Summary & Architecture | Problem, solution, high-level architecture |
| 02 | Data Models | Redis schema, Postgres schema, Python objects |
| 03 | Core Algorithms | Emotional scoring, TTL, retrieval ranking, consolidation |
| 04 | API Design | Public interface, method signatures |
| 05 | Phase 1 | Redis foundation, basic read/write |
| 06 | Phase 2 | Migration from mem0, seeding identity |
| 07 | Phase 3 | Write-through pattern, session management |
| 08 | Phase 4 | Consolidation jobs, background processing |
| 09 | Phase 5 | Multi-user support, HuixiangDou patterns |
| 10 | Testing & Rollback | Testing strategy, rollback plan, success metrics |

---

# Section 1: Executive Summary & Architecture

## 1.1 Problem Statement

Current mem0-based memory relies solely on embedding similarity for retrieval. This leads to:

- **"Lost in the middle" effect**: Older memories get buried regardless of importance
- **No temporal awareness**: Recent vs. old treated equally in retrieval
- **No emotional weighting**: Mundane facts retrieved alongside significant events
- **No working memory**: Everything persists equally, no natural fade
- **Contradiction accumulation**: Conflicting facts coexist without resolution
- **Retrieval-only architecture**: Context is whatever floats to the top, not structured

## 1.2 Solution

A tiered memory architecture inspired by Manus's "context as RAM, filesystem as disk" pattern:

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Hot** | Redis | Always-loaded context, working memory with TTL, session state |
| **Warm** | Postgres + pgvector | Searchable long-term storage, embeddings, episodic/semantic facts |
| **Background** | Async workers | Pattern extraction, decay, deduplication, promotion |

## 1.3 Key Innovations

**Explicit context structure**: Core identity and session state are *always loaded*, not retrieved. Retrieval augments, not replaces.

**Emotional decay**: Memories fade based on emotional intensity. High-impact events persist in working memory longer than mundane observations.

**"Wait, what?" confirmation**: The only human review is for fact corrections—user naturally corrects wrong assumptions, system learns.

**Multi-user from day one**: Designed for multiple users with isolated memory spaces, inspired by HuixiangDou patterns.

## 1.4 Package Details

- **Name**: `cortex` (or `cortex`?)
- **Type**: Python package, imported directly by Clara
- **Repository**: New repo, standalone
- **Dependencies**: Redis, Postgres (pgvector), async HTTP client for embeddings

---

## 2. Architecture Overview

### 2.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLARA APPLICATION                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    MemoryManager                         │   │
│  │  - get_context(user_id) → always-loaded + retrieved      │   │
│  │  - store(user_id, memory) → write-through                │   │
│  │  - search(user_id, query) → semantic search              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐    ┌───────────────────┐
│     REDIS     │    │    POSTGRES    │    │   CONSOLIDATOR    │
│   (Hot/Fast)  │    │  (Warm/Search) │    │   (Background)    │
├───────────────┤    ├────────────────┤    ├───────────────────┤
│ • Identity    │    │ • memories     │    │ • Decay job       │
│ • Session     │    │ • Working mem  │    │ • Dedup job       │
│ • Recent buf  │    │ • metadata     │    │ • Promote job     │
│               │    │ • full history │    │ • Compact job     │
└───────────────┘    └────────────────┘    └───────────────────┘
       │                     │                      │
       └─────────────────────┴──────────────────────┘
                    Shared user_id namespace
```

### 2.2 Data Flow

**Write Path:**
```
New Memory 
    → Classify (type, emotion score via LLM)
    → Write to Redis (immediate, with TTL for working memory)
    → Queue for Postgres (async embed + store)
    → Return immediately
```

**Read Path (every turn):**
```
get_context(user_id, query?)
    → Load from Redis: identity + session + working (instant, always)
    → If query: semantic search Postgres (top-k by combined score)
    → Merge & dedupe
    → Update access timestamps
    → Return MemoryContext
```

### 2.3 Multi-User Isolation

All keys and queries are namespaced by `user_id`:

- Redis: `cortex:{user_id}:*`
- Postgres: `WHERE user_id = $1`

No cross-user data leakage possible at the data layer.

---

## 3. Design Principles

### 3.1 Always-Loaded vs Retrieved

| Category | Behavior | Example |
|----------|----------|---------|
| **Always-loaded** | Present every turn, no query needed | "Josh has 3 kids" |
| **Retrieved** | Fetched when semantically relevant | "On Jan 5, Thomas had a meltdown" |

### 3.2 Emotional Decay

Working memory TTL scales with emotional intensity:

| Emotion Score | Example | TTL |
|---------------|---------|-----|
| 0.0 - 0.2 | "Grabbed coffee" | 30 min |
| 0.2 - 0.4 | "Good meeting with team" | 1.5 hours |
| 0.4 - 0.6 | "Finished the TDD" | 3 hours |
| 0.6 - 0.8 | "Thomas had a meltdown" | 5 hours |
| 0.8 - 1.0 | "Major family emergency" | 6+ hours |

---

# Section 3: Core Algorithms

## 3.1 Emotional Scoring

At write time, memories are scored for emotional intensity using the LLM.

### Scoring Prompt

```python
EMOTION_SCORING_PROMPT = """
Rate the emotional intensity of this memory on a scale of 0.0 to 1.0:

- 0.0-0.2: Mundane, routine (grabbed coffee, checked email)
- 0.2-0.4: Mildly notable (had a good meeting, tried a new restaurant)
- 0.4-0.6: Significant (completed a project, had an argument)
- 0.6-0.8: Highly emotional (family emergency, major life event, conflict)
- 0.8-1.0: Profound impact (death, birth, trauma, breakthrough)

Memory: {content}

Return ONLY a float between 0.0 and 1.0.
"""
```

### Implementation

```python
async def score_emotion(content: str, llm_client) -> float:
    response = await llm_client.complete(
        EMOTION_SCORING_PROMPT.format(content=content),
        max_tokens=10
    )
    try:
        return max(0.0, min(1.0, float(response.strip())))
    except ValueError:
        return 0.5  # Default to middle if parsing fails
```

### Batch Scoring

For efficiency, score multiple memories in one call:

```python
async def score_emotions_batch(memories: list[str], llm_client) -> list[float]:
    prompt = "Rate each memory's emotional intensity (0.0-1.0). Return one float per line.\n\n"
    for i, mem in enumerate(memories):
        prompt += f"{i+1}. {mem}\n"
    
    response = await llm_client.complete(prompt, max_tokens=len(memories) * 5)
    scores = []
    for line in response.strip().split('\n'):
        try:
            scores.append(max(0.0, min(1.0, float(line.strip()))))
        except ValueError:
            scores.append(0.5)
    return scores
```

---

## 3.2 TTL Calculation (Working Memory)

Working memory fades based on emotional intensity. Higher emotion = longer persistence.

```python
def calculate_ttl(emotional_score: float, base_ttl: int = 1800) -> int:
    """
    Calculate TTL in seconds based on emotional score.
    
    Base: 30 minutes (1800 seconds)
    Range: 30 minutes (emotion=0) to 6 hours (emotion=1)
    
    Formula: base_ttl * (1 + emotional_score * 11)
    - emotion=0.0 → 1800 * 1.0 = 30 min
    - emotion=0.5 → 1800 * 6.5 = 3.25 hours
    - emotion=1.0 → 1800 * 12 = 6 hours
    """
    multiplier = 1 + (emotional_score * 11)
    return int(base_ttl * multiplier)
```

### TTL Table

| Emotional Score | Multiplier | TTL |
|-----------------|------------|-----|
| 0.0 | 1.0x | 30 minutes |
| 0.2 | 3.2x | 1.6 hours |
| 0.5 | 6.5x | 3.25 hours |
| 0.8 | 9.8x | 4.9 hours |
| 1.0 | 12x | 6 hours |

---

## 3.3 Retrieval Scoring

When retrieving memories, combine multiple signals for ranking:

```python
import math
from datetime import datetime

def calculate_retrieval_score(
    memory: Memory,
    similarity: float,  # From vector search, 0-1
    current_time: datetime
) -> float:
    """
    Combine signals for final retrieval ranking.
    
    Weights (tunable):
    - Similarity: 0.40 (semantic relevance)
    - Recency: 0.25 (temporal decay)
    - Emotion: 0.20 (emotional importance)
    - Reinforcement: 0.15 (access patterns)
    """
    # Recency decay: 1/(1 + log(1 + hours_old))
    age_hours = (current_time - memory.created_at).total_seconds() / 3600
    recency_score = 1 / (1 + math.log(1 + age_hours))
    
    # Reinforcement: log(1 + access_count) normalized
    reinforcement_score = math.log(1 + memory.access_count) / 10
    reinforcement_score = min(1.0, reinforcement_score)
    
    # Weighted combination
    final_score = (
        0.40 * similarity +
        0.25 * recency_score +
        0.20 * memory.emotional_score +
        0.15 * reinforcement_score
    )
    
    return final_score
```

### Weight Tuning

These weights are initial guesses. Tune based on user feedback:

```python
@dataclass
class RetrievalWeights:
    similarity: float = 0.40
    recency: float = 0.25
    emotion: float = 0.20
    reinforcement: float = 0.15
    
    def __post_init__(self):
        total = self.similarity + self.recency + self.emotion + self.reinforcement
        assert abs(total - 1.0) < 0.01, f"Weights must sum to 1.0, got {total}"
```

---

## 3.4 Consolidation Logic

Background job that runs periodically to maintain memory health.

```python
async def consolidate_user(user_id: str, stores: Stores):
    """
    Background job to consolidate memories for a user.
    
    Steps:
    1. Identify patterns in recent episodic memories
    2. Detect contradictions
    3. Update identity facts
    4. Compact old episodes
    5. Clean up expired working memory
    """
    log = ConsolidationLog(user_id=user_id, started_at=datetime.utcnow())
    
    # 1. Get recent episodic memories (last 7 days)
    recent = await stores.postgres.get_memories(
        user_id=user_id,
        memory_type=MemoryType.EPISODIC,
        since=datetime.utcnow() - timedelta(days=7)
    )
    log.memories_processed = len(recent)
    
    # 2. Extract patterns via LLM
    patterns = await extract_patterns(recent, stores.llm)
    log.patterns_found = len(patterns)
    
    # 3. Update identity for high-confidence patterns
    for pattern in patterns:
        if pattern.confidence > 0.8:
            await stores.redis.update_identity(user_id, pattern.fact, pattern.value)
            await stores.postgres.store_identity_update(user_id, pattern)
            log.identities_updated += 1
    
    # 4. Detect contradictions via embedding clustering
    contradictions = await detect_contradictions(user_id, stores.postgres)
    for c in contradictions:
        await stores.postgres.flag_contradiction(c)
    
    # 5. Compact old episodes (>30 days) into summaries
    old_episodes = await stores.postgres.get_memories(
        user_id=user_id,
        memory_type=MemoryType.EPISODIC,
        before=datetime.utcnow() - timedelta(days=30),
        status=MemoryStatus.ACTIVE
    )
    if old_episodes:
        summary = await summarize_episodes(old_episodes, stores.llm)
        await stores.postgres.store(Memory(
            user_id=user_id,
            content=summary,
            memory_type=MemoryType.SEMANTIC,
            source="consolidation"
        ))
        for ep in old_episodes:
            await stores.postgres.update_status(ep.id, MemoryStatus.ARCHIVED)
        log.memories_compacted = len(old_episodes)
    
    # 6. Clean expired working memory (Redis handles TTL, but audit)
    await stores.redis.cleanup_expired_working(user_id)
    
    # 7. Log consolidation run
    log.completed_at = datetime.utcnow()
    log.duration_ms = int((log.completed_at - log.started_at).total_seconds() * 1000)
    await stores.postgres.log_consolidation(log)
    
    return log
```

### Pattern Extraction Prompt

```python
PATTERN_EXTRACTION_PROMPT = """
Analyze these recent memories and identify stable patterns or facts about the user.

Memories:
{memories}

For each pattern you identify, provide:
1. The fact (e.g., "has three children", "works in software")
2. Confidence (0.0-1.0)
3. Evidence (which memories support this)

Only report patterns with confidence > 0.6.
Return as JSON array.
"""
```

### Contradiction Detection

```python
async def detect_contradictions(user_id: str, postgres: PostgresStore) -> list[Contradiction]:
    """
    Find memories that contradict each other.
    
    Approach:
    1. Cluster memories by embedding similarity
    2. Within clusters, ask LLM to identify conflicts
    """
    # Get identity and semantic memories (facts that could conflict)
    facts = await postgres.get_memories(
        user_id=user_id,
        memory_types=[MemoryType.IDENTITY, MemoryType.SEMANTIC],
        status=MemoryStatus.ACTIVE
    )
    
    # Cluster by embedding (DBSCAN or similar)
    clusters = cluster_by_embedding(facts, eps=0.3)
    
    contradictions = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        
        # Ask LLM to check for conflicts
        conflicts = await check_cluster_conflicts(cluster)
        contradictions.extend(conflicts)
    
    return contradictions
```

---

## 3.5 Importance Scoring

Derived score combining multiple factors:

```python
def calculate_importance(memory: Memory) -> float:
    """
    Calculate overall importance for a memory.
    
    Factors:
    - Emotional score (40%)
    - Recency (20%)
    - Access patterns (20%)
    - Type weight (20%)
    """
    type_weights = {
        MemoryType.IDENTITY: 1.0,
        MemoryType.SEMANTIC: 0.8,
        MemoryType.EPISODIC: 0.6,
        MemoryType.WORKING: 0.4,
        MemoryType.SESSION: 0.3,
    }
    
    type_score = type_weights.get(memory.memory_type, 0.5)
    
    age_hours = (datetime.utcnow() - memory.created_at).total_seconds() / 3600
    recency_score = 1 / (1 + math.log(1 + age_hours))
    
    access_score = min(1.0, math.log(1 + memory.access_count) / 5)
    
    importance = (
        0.40 * memory.emotional_score +
        0.20 * recency_score +
        0.20 * access_score +
        0.20 * type_score
    )
    
    return importance
```

---

## 3.6 Summary

| Algorithm | Purpose | Key Parameters |
|-----------|---------|----------------|
| Emotional Scoring | Tag memories at write time | LLM prompt, 0-1 scale |
| TTL Calculation | Working memory decay | base_ttl=1800s, max 6h |
| Retrieval Scoring | Rank search results | 4 weighted signals |
| Consolidation | Background maintenance | 7-day window, 0.8 confidence |
| Contradiction Detection | Find conflicts | Embedding clustering |
| Importance Scoring | Derived priority | 4 weighted factors |

# Section 4: API Design

## 4.1 Public Interface Overview

The MemoryManager is the primary interface for Clara's memory system. It provides a clean abstraction over the underlying Redis and Postgres stores.

## 4.2 Core Data Classes

```python
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

@dataclass
class MemoryContext:
    """Returned by get_context() - everything Clara needs for a turn."""
    user_id: str
    identity: dict                    # Core facts, always loaded
    session: dict                     # Current session state
    working: List[Memory]             # Recent working memories
    retrieved: List[Memory]           # Semantically retrieved
    project: Optional[dict] = None    # Project-specific context
    
    def to_prompt_string(self) -> str:
        """Format for inclusion in system prompt."""
        sections = []
        
        if self.identity:
            sections.append("## User Identity")
            for k, v in self.identity.items():
                sections.append(f"- {k}: {v}")
        
        if self.session:
            sections.append("\n## Current Session")
            for k, v in self.session.items():
                sections.append(f"- {k}: {v}")
        
        if self.working:
            sections.append("\n## Recent Context")
            for m in self.working[:10]:
                sections.append(f"- {m.content}")
        
        if self.retrieved:
            sections.append("\n## Relevant Memories")
            for m in self.retrieved[:15]:
                sections.append(f"- {m.content}")
        
        return "\n".join(sections)

@dataclass
class StoreResult:
    """Returned by store() operations."""
    success: bool
    memory_id: Optional[str] = None
    error: Optional[str] = None
    emotional_score: Optional[float] = None
    ttl: Optional[int] = None

@dataclass 
class SearchResult:
    """Returned by search() operations."""
    memories: List[Memory]
    total_count: int
    search_time_ms: int
```


## 4.3 MemoryManager Class

```python
class MemoryManager:
    """
    Main interface for Clara's memory system.
    
    Usage:
        manager = MemoryManager(config)
        await manager.initialize()
        
        # Every turn
        context = await manager.get_context(user_id, query="current topic")
        
        # Store new memories
        await manager.store(user_id, "User mentioned they have a meeting tomorrow")
        
        # Update session
        await manager.update_session(user_id, {"current_topic": "scheduling"})
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.redis: RedisStore = None
        self.postgres: PostgresStore = None
        self.embedder: Embedder = None
        self.scorer: EmotionScorer = None
    
    async def initialize(self):
        """Initialize connections to Redis and Postgres."""
        self.redis = RedisStore(self.config.redis)
        self.postgres = PostgresStore(self.config.postgres)
        self.embedder = Embedder(self.config.embedding)
        self.scorer = EmotionScorer(self.config.llm)
        
        await self.redis.connect()
        await self.postgres.connect()
    
    async def close(self):
        """Clean shutdown."""
        await self.redis.close()
        await self.postgres.close()
    
    async def get_context(
        self,
        user_id: str,
        query: Optional[str] = None,
        project_id: Optional[str] = None,
        include_working: bool = True,
        max_retrieved: int = 20
    ) -> MemoryContext:
        """
        Get full context for a user. Called every turn.
        
        Args:
            user_id: User identifier
            query: Optional query for semantic retrieval
            project_id: Optional project scope
            include_working: Include working memory
            max_retrieved: Max memories to retrieve
            
        Returns:
            MemoryContext with identity, session, working, and retrieved memories
        """
        # Always load from Redis (fast)
        identity = await self.redis.get_identity(user_id)
        session = await self.redis.get_session(user_id)
        
        working = []
        if include_working:
            working = await self.redis.get_working(user_id)
        
        # Semantic retrieval from Postgres (if query provided)
        retrieved = []
        if query:
            embedding = await self.embedder.embed(query)
            retrieved = await self.postgres.search(
                user_id=user_id,
                embedding=embedding,
                project_id=project_id,
                limit=max_retrieved
            )
            # Re-rank with full scoring
            retrieved = self._rerank(retrieved, query)
        
        # Project context
        project = None
        if project_id:
            project = await self.postgres.get_project_context(user_id, project_id)
        
        return MemoryContext(
            user_id=user_id,
            identity=identity,
            session=session,
            working=working,
            retrieved=retrieved,
            project=project
        )
```


    async def store(
        self,
        user_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        project_id: Optional[str] = None,
        emotional_score: Optional[float] = None,
        metadata: Optional[dict] = None
    ) -> StoreResult:
        """
        Store a new memory. Write-through pattern.
        
        1. Score emotion (if not provided)
        2. Write to Redis immediately (working memory)
        3. Generate embedding
        4. Write to Postgres async
        """
        try:
            # Score emotion if not provided
            if emotional_score is None:
                emotional_score = await self.scorer.score(content)
            
            # Calculate TTL for working memory
            ttl = calculate_ttl(emotional_score)
            
            # Create memory object
            memory = Memory(
                content=content,
                user_id=user_id,
                memory_type=memory_type,
                project_id=project_id,
                emotional_score=emotional_score,
                metadata=metadata or {},
                source="conversation"
            )
            
            # Write to Redis immediately
            await self.redis.add_working(user_id, memory, ttl=ttl)
            
            # Generate embedding and write to Postgres (can be async/queued)
            memory.embedding = await self.embedder.embed(content)
            memory.id = await self.postgres.store(memory)
            
            return StoreResult(
                success=True,
                memory_id=memory.id,
                emotional_score=emotional_score,
                ttl=ttl
            )
        except Exception as e:
            return StoreResult(success=False, error=str(e))
    
    async def search(
        self,
        user_id: str,
        query: str,
        project_id: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 20
    ) -> SearchResult:
        """
        Semantic search over memories.
        """
        import time
        start = time.time()
        
        embedding = await self.embedder.embed(query)
        memories = await self.postgres.search(
            user_id=user_id,
            embedding=embedding,
            project_id=project_id,
            memory_types=memory_types,
            limit=limit
        )
        
        # Re-rank
        memories = self._rerank(memories, query)
        
        elapsed_ms = int((time.time() - start) * 1000)
        
        return SearchResult(
            memories=memories,
            total_count=len(memories),
            search_time_ms=elapsed_ms
        )
```


## 4.4 Session Management

```python
    async def update_session(
        self,
        user_id: str,
        updates: dict
    ) -> bool:
        """Update session state in Redis."""
        return await self.redis.update_session(user_id, updates)
    
    async def start_session(
        self,
        user_id: str,
        initial_context: Optional[dict] = None
    ) -> dict:
        """Start a new session, loading identity."""
        session = {
            "started_at": datetime.utcnow().isoformat(),
            "last_active": datetime.utcnow().isoformat(),
            **(initial_context or {})
        }
        await self.redis.set_session(user_id, session)
        return session
    
    async def end_session(self, user_id: str):
        """
        End a session. Optionally persist important working memories.
        """
        # Get working memories before they expire
        working = await self.redis.get_working(user_id)
        
        # Persist high-emotion working memories to episodic
        for mem in working:
            if mem.emotional_score > 0.6:
                mem.memory_type = MemoryType.EPISODIC
                await self.postgres.store(mem)
        
        # Clear session
        await self.redis.clear_session(user_id)

## 4.5 Identity Management

    async def update_identity(
        self,
        user_id: str,
        key: str,
        value: any
    ) -> bool:
        """
        Update a core identity fact.
        Also stores to Postgres for persistence.
        """
        # Update Redis
        await self.redis.update_identity_field(user_id, key, value)
        
        # Store as identity memory in Postgres
        await self.postgres.store(Memory(
            user_id=user_id,
            content=f"{key}: {value}",
            memory_type=MemoryType.IDENTITY,
            emotional_score=0.7,  # Identity facts are important
            source="identity_update"
        ))
        
        return True
    
    async def get_identity(self, user_id: str) -> dict:
        """Get current identity facts."""
        return await self.redis.get_identity(user_id)
```


## 4.6 Internal Interfaces

### RedisStore

```python
class RedisStore:
    """Internal interface for Redis operations."""
    
    async def connect(self): ...
    async def close(self): ...
    
    # Identity
    async def get_identity(self, user_id: str) -> dict: ...
    async def set_identity(self, user_id: str, identity: dict): ...
    async def update_identity_field(self, user_id: str, key: str, value: any): ...
    
    # Session
    async def get_session(self, user_id: str) -> dict: ...
    async def set_session(self, user_id: str, session: dict): ...
    async def update_session(self, user_id: str, updates: dict) -> bool: ...
    async def clear_session(self, user_id: str): ...
    
    # Working Memory
    async def add_working(self, user_id: str, memory: Memory, ttl: int): ...
    async def get_working(self, user_id: str) -> List[Memory]: ...
    async def cleanup_expired_working(self, user_id: str): ...
    
    # Recent Buffer
    async def add_recent(self, user_id: str, memory: Memory): ...
    async def get_recent(self, user_id: str, limit: int = 10) -> List[Memory]: ...
```

### PostgresStore

```python
class PostgresStore:
    """Internal interface for Postgres operations."""
    
    async def connect(self): ...
    async def close(self): ...
    
    # Core CRUD
    async def store(self, memory: Memory) -> str: ...
    async def get(self, memory_id: str) -> Optional[Memory]: ...
    async def update(self, memory_id: str, updates: dict) -> bool: ...
    async def update_status(self, memory_id: str, status: MemoryStatus): ...
    
    # Search
    async def search(
        self,
        user_id: str,
        embedding: List[float],
        project_id: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 20
    ) -> List[Memory]: ...
    
    # Retrieval
    async def get_memories(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
        since: Optional[datetime] = None,
        before: Optional[datetime] = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        limit: int = 100
    ) -> List[Memory]: ...
    
    # Project
    async def get_project_context(self, user_id: str, project_id: str) -> dict: ...
    
    # Consolidation
    async def flag_contradiction(self, contradiction: Contradiction): ...
    async def log_consolidation(self, log: ConsolidationLog): ...
```

### Embedder

```python
class Embedder:
    """Embedding generation interface."""
    
    async def embed(self, text: str) -> List[float]: ...
    async def embed_batch(self, texts: List[str]) -> List[List[float]]: ...
```

### EmotionScorer

```python
class EmotionScorer:
    """Emotional intensity scoring."""
    
    async def score(self, content: str) -> float: ...
    async def score_batch(self, contents: List[str]) -> List[float]: ...
```
# Section 5: Phase 1 - Redis Foundation

## 5.1 Overview

Phase 1 establishes the hot storage layer using Redis. This provides:
- Always-loaded identity context
- Session state management
- Working memory with TTL-based decay
- Recent memory buffer

**Duration**: 1-2 weeks
**Dependencies**: Redis instance (existing or new)
**Deliverables**: RedisStore class, tests, basic CLI

---

## 5.2 Redis Key Schema

```
# Namespace prefix
PREFIX = "cortex"

# Per-user keys
{PREFIX}:{user_id}:identity    → HASH
{PREFIX}:{user_id}:session     → HASH
{PREFIX}:{user_id}:working     → ZSET (score = timestamp)
{PREFIX}:{user_id}:recent      → LIST (capped at N items)
{PREFIX}:{user_id}:meta        → HASH
```

### Identity Hash

```python
{
    "name": "Josh",
    "relationship": "user",
    "family": '{"children": ["Madeline", "Anne", "Thomas"]}',  # JSON string
    "occupation": "software engineer",
    "location": "unknown",
    "timezone": "America/New_York",
    "preferences": '{"communication": "direct", "humor": "dry"}',
    "key_facts": '["has three kids", "building Clara project"]',
    "updated_at": "2025-01-09T19:30:00Z"
}
```

### Session Hash

```python
{
    "started_at": "2025-01-09T18:00:00Z",
    "last_active": "2025-01-09T19:35:00Z",
    "current_topic": "memory system design",
    "active_goals": '["write TDD", "setup repo"]',
    "emotional_state": "focused",
    "context_summary": "Working on Clara memory architecture"
}
```

### Working Memory ZSET

Score is Unix timestamp, members are JSON-encoded memories:

```python
ZADD cortex:user123:working 1704825600 '{"content": "User mentioned meeting tomorrow", "emotion": 0.3, "type": "episodic"}'
```

### Recent Buffer LIST

Simple list of recent episodic memory IDs, capped at 50:

```python
LPUSH cortex:user123:recent "memory-uuid-here"
LTRIM cortex:user123:recent 0 49
```


## 5.3 RedisStore Implementation

```python
import redis.asyncio as redis
import json
from datetime import datetime
from typing import Optional, List
from dataclasses import asdict

class RedisStore:
    PREFIX = "cortex"
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self.client: redis.Redis = None
    
    async def connect(self):
        self.client = redis.Redis(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            decode_responses=True
        )
        await self.client.ping()
    
    async def close(self):
        if self.client:
            await self.client.close()
    
    def _key(self, user_id: str, suffix: str) -> str:
        return f"{self.PREFIX}:{user_id}:{suffix}"
    
    # ==================== IDENTITY ====================
    
    async def get_identity(self, user_id: str) -> dict:
        """Get all identity fields for a user."""
        data = await self.client.hgetall(self._key(user_id, "identity"))
        if not data:
            return {}
        
        # Parse JSON fields
        result = {}
        json_fields = ["family", "preferences", "key_facts"]
        for k, v in data.items():
            if k in json_fields:
                try:
                    result[k] = json.loads(v)
                except json.JSONDecodeError:
                    result[k] = v
            else:
                result[k] = v
        return result
    
    async def set_identity(self, user_id: str, identity: dict):
        """Set all identity fields (overwrites)."""
        key = self._key(user_id, "identity")
        
        # Serialize nested objects
        flat = {}
        for k, v in identity.items():
            if isinstance(v, (dict, list)):
                flat[k] = json.dumps(v)
            else:
                flat[k] = str(v) if v is not None else ""
        
        flat["updated_at"] = datetime.utcnow().isoformat()
        
        async with self.client.pipeline() as pipe:
            await pipe.delete(key)
            if flat:
                await pipe.hset(key, mapping=flat)
            await pipe.execute()
    
    async def update_identity_field(self, user_id: str, field: str, value: any):
        """Update a single identity field."""
        key = self._key(user_id, "identity")
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        await self.client.hset(key, field, str(value))
        await self.client.hset(key, "updated_at", datetime.utcnow().isoformat())
```


    # ==================== SESSION ====================
    
    async def get_session(self, user_id: str) -> dict:
        """Get current session state."""
        data = await self.client.hgetall(self._key(user_id, "session"))
        if not data:
            return {}
        
        # Parse JSON fields
        result = {}
        json_fields = ["active_goals"]
        for k, v in data.items():
            if k in json_fields:
                try:
                    result[k] = json.loads(v)
                except json.JSONDecodeError:
                    result[k] = v
            else:
                result[k] = v
        return result
    
    async def set_session(self, user_id: str, session: dict):
        """Set session state (overwrites)."""
        key = self._key(user_id, "session")
        
        flat = {}
        for k, v in session.items():
            if isinstance(v, (dict, list)):
                flat[k] = json.dumps(v)
            else:
                flat[k] = str(v) if v is not None else ""
        
        async with self.client.pipeline() as pipe:
            await pipe.delete(key)
            if flat:
                await pipe.hset(key, mapping=flat)
            await pipe.expire(key, 86400)  # 24 hour TTL
            await pipe.execute()
    
    async def update_session(self, user_id: str, updates: dict) -> bool:
        """Update specific session fields."""
        key = self._key(user_id, "session")
        
        flat = {}
        for k, v in updates.items():
            if isinstance(v, (dict, list)):
                flat[k] = json.dumps(v)
            else:
                flat[k] = str(v) if v is not None else ""
        
        flat["last_active"] = datetime.utcnow().isoformat()
        
        await self.client.hset(key, mapping=flat)
        await self.client.expire(key, 86400)  # Reset TTL
        return True
    
    async def clear_session(self, user_id: str):
        """Clear session data."""
        await self.client.delete(self._key(user_id, "session"))
    
    # ==================== WORKING MEMORY ====================
    
    async def add_working(self, user_id: str, memory: Memory, ttl: int):
        """Add a memory to working memory with TTL."""
        key = self._key(user_id, "working")
        
        # Serialize memory
        mem_data = {
            "id": memory.id or str(uuid.uuid4()),
            "content": memory.content,
            "type": memory.memory_type.value,
            "emotion": memory.emotional_score,
            "created_at": memory.created_at.isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(seconds=ttl)).isoformat()
        }
        
        score = datetime.utcnow().timestamp()
        await self.client.zadd(key, {json.dumps(mem_data): score})
        
        # Schedule cleanup (Redis doesn't auto-expire ZSET members)
        # We'll handle this in cleanup_expired_working()
    
    async def get_working(self, user_id: str) -> List[Memory]:
        """Get all non-expired working memories."""
        key = self._key(user_id, "working")
        
        # Get all members
        members = await self.client.zrange(key, 0, -1)
        
        now = datetime.utcnow()
        result = []
        expired = []
        
        for member in members:
            try:
                data = json.loads(member)
                expires_at = datetime.fromisoformat(data["expires_at"])
                
                if expires_at > now:
                    result.append(Memory(
                        id=data["id"],
                        content=data["content"],
                        memory_type=MemoryType(data["type"]),
                        emotional_score=data["emotion"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        user_id=user_id
                    ))
                else:
                    expired.append(member)
            except (json.JSONDecodeError, KeyError):
                expired.append(member)
        
        # Clean up expired
        if expired:
            await self.client.zrem(key, *expired)
        
        return result
    
    async def cleanup_expired_working(self, user_id: str) -> int:
        """Remove expired working memories. Returns count removed."""
        memories = await self.get_working(user_id)  # This cleans up as side effect
        # The get_working method already removes expired entries
        return 0  # Could track if needed
```


    # ==================== RECENT BUFFER ====================
    
    async def add_recent(self, user_id: str, memory_id: str):
        """Add a memory ID to the recent buffer."""
        key = self._key(user_id, "recent")
        async with self.client.pipeline() as pipe:
            await pipe.lpush(key, memory_id)
            await pipe.ltrim(key, 0, 49)  # Keep last 50
            await pipe.execute()
    
    async def get_recent(self, user_id: str, limit: int = 10) -> List[str]:
        """Get recent memory IDs."""
        key = self._key(user_id, "recent")
        return await self.client.lrange(key, 0, limit - 1)
    
    # ==================== META ====================
    
    async def get_meta(self, user_id: str) -> dict:
        """Get user metadata."""
        return await self.client.hgetall(self._key(user_id, "meta"))
    
    async def update_meta(self, user_id: str, updates: dict):
        """Update user metadata."""
        await self.client.hset(self._key(user_id, "meta"), mapping=updates)

---

## 5.4 Tests

```python
import pytest
from datetime import datetime, timedelta

@pytest.fixture
async def redis_store():
    store = RedisStore(RedisConfig(host="localhost", port=6379, db=15))
    await store.connect()
    yield store
    # Cleanup
    await store.client.flushdb()
    await store.close()

class TestIdentity:
    async def test_set_and_get_identity(self, redis_store):
        user_id = "test-user-1"
        identity = {
            "name": "Josh",
            "family": {"children": ["Maddie", "Anne", "Thomas"]},
            "preferences": {"style": "direct"}
        }
        
        await redis_store.set_identity(user_id, identity)
        result = await redis_store.get_identity(user_id)
        
        assert result["name"] == "Josh"
        assert result["family"]["children"] == ["Maddie", "Anne", "Thomas"]
        assert "updated_at" in result
    
    async def test_update_identity_field(self, redis_store):
        user_id = "test-user-2"
        await redis_store.set_identity(user_id, {"name": "Josh"})
        await redis_store.update_identity_field(user_id, "location", "NYC")
        
        result = await redis_store.get_identity(user_id)
        assert result["name"] == "Josh"
        assert result["location"] == "NYC"

class TestSession:
    async def test_session_lifecycle(self, redis_store):
        user_id = "test-user-3"
        
        # Start session
        await redis_store.set_session(user_id, {
            "started_at": datetime.utcnow().isoformat(),
            "current_topic": "testing"
        })
        
        # Update session
        await redis_store.update_session(user_id, {"current_topic": "memory"})
        
        session = await redis_store.get_session(user_id)
        assert session["current_topic"] == "memory"
        assert "last_active" in session
        
        # Clear session
        await redis_store.clear_session(user_id)
        session = await redis_store.get_session(user_id)
        assert session == {}

class TestWorkingMemory:
    async def test_working_memory_with_ttl(self, redis_store):
        user_id = "test-user-4"
        
        memory = Memory(
            content="Test memory",
            user_id=user_id,
            memory_type=MemoryType.WORKING,
            emotional_score=0.5
        )
        
        # Add with 1 hour TTL
        await redis_store.add_working(user_id, memory, ttl=3600)
        
        working = await redis_store.get_working(user_id)
        assert len(working) == 1
        assert working[0].content == "Test memory"
    
    async def test_expired_working_memory_cleaned(self, redis_store):
        user_id = "test-user-5"
        
        # Manually add expired memory
        key = redis_store._key(user_id, "working")
        expired_data = {
            "id": "test-id",
            "content": "Expired",
            "type": "working",
            "emotion": 0.5,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() - timedelta(hours=1)).isoformat()
        }
        await redis_store.client.zadd(key, {json.dumps(expired_data): 1234567890})
        
        # Get should clean up
        working = await redis_store.get_working(user_id)
        assert len(working) == 0
```


---

## 5.5 Acceptance Criteria

### Must Have
- [ ] RedisStore connects and handles connection failures gracefully
- [ ] Identity CRUD operations work with nested JSON
- [ ] Session management with automatic TTL refresh
- [ ] Working memory with emotional TTL decay
- [ ] All tests passing

### Should Have
- [ ] Connection pooling for high concurrency
- [ ] Metrics (operation latency, cache hits)
- [ ] CLI for manual inspection (`cortex redis get-identity <user>`)

### Nice to Have
- [ ] Redis Cluster support
- [ ] Pub/sub for real-time updates

---

## 5.6 Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Redis unavailable | Clara can't load context | Fallback to Postgres-only mode |
| Memory leak in ZSET | Unbounded growth | Periodic cleanup job, max size limit |
| JSON serialization issues | Data loss | Schema validation, error logging |

---

## 5.7 Definition of Done

1. RedisStore class implemented with full test coverage
2. Integration tests against real Redis instance
3. Documentation for Redis schema
4. Basic CLI for debugging
5. Deployed to staging environment

# Section 6: Phase 2 - Migration from mem0

## 6.1 Overview

Phase 2 migrates existing memories from mem0 to the new system. This is a one-time operation that:
1. Exports all memories from mem0
2. Classifies and scores each memory
3. Seeds Redis with identity facts
4. Imports to Postgres with embeddings

**Duration**: 1 week
**Dependencies**: Phase 1 complete, mem0 access
**Deliverables**: Migration scripts, validation report

---

## 6.2 Migration Strategy

### Approach: Parallel Operation
- New system runs alongside mem0
- Gradually shift reads to new system
- mem0 remains read-only backup
- Cut over when confidence is high

### Data Flow
```
mem0 → Export → Classify → Score → Transform → Redis + Postgres
```

---

## 6.3 Export Script

```python
from mem0 import MemoryClient

async def export_mem0_memories(user_id: str, mem0_client: MemoryClient) -> list[dict]:
    """Export all memories for a user from mem0."""
    
    memories = []
    
    # Get all memories (mem0 API)
    results = mem0_client.get_all(user_id=user_id)
    
    for item in results:
        memories.append({
            "id": item.get("id"),
            "content": item.get("memory"),
            "metadata": item.get("metadata", {}),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "hash": item.get("hash"),
        })
    
    return memories

async def export_all_users(mem0_client: MemoryClient) -> dict[str, list]:
    """Export memories for all users."""
    # This requires listing all users, which may need custom implementation
    # based on how user IDs are tracked
    
    all_exports = {}
    user_ids = await get_all_user_ids()  # Implementation depends on your tracking
    
    for user_id in user_ids:
        all_exports[user_id] = await export_mem0_memories(user_id, mem0_client)
    
    return all_exports
```


## 6.4 Classification Script

Classify each memory into the new type system:

```python
CLASSIFICATION_PROMPT = """
Classify this memory into one of these types:
- identity: Core fact about the user (name, family, job, preferences)
- semantic: General knowledge or learned fact
- episodic: An event or thing that happened
- project: Related to a specific project

Memory: {content}

Return JSON: {"type": "<type>", "confidence": <0.0-1.0>, "identity_key": "<key if identity>"}

Examples:
- "User's name is Josh" → {"type": "identity", "confidence": 0.95, "identity_key": "name"}
- "User has three kids" → {"type": "identity", "confidence": 0.9, "identity_key": "family"}
- "Had a meeting about the API" → {"type": "episodic", "confidence": 0.85, "identity_key": null}
"""

@dataclass
class ClassificationResult:
    memory_type: MemoryType
    confidence: float
    identity_key: Optional[str] = None

async def classify_memory(content: str, llm_client) -> ClassificationResult:
    response = await llm_client.complete(
        CLASSIFICATION_PROMPT.format(content=content),
        max_tokens=100
    )
    
    try:
        data = json.loads(response)
        return ClassificationResult(
            memory_type=MemoryType(data["type"]),
            confidence=data["confidence"],
            identity_key=data.get("identity_key")
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        # Default to episodic if classification fails
        return ClassificationResult(
            memory_type=MemoryType.EPISODIC,
            confidence=0.5
        )

async def classify_batch(memories: list[str], llm_client) -> list[ClassificationResult]:
    """Batch classify for efficiency."""
    results = []
    
    # Process in batches of 20
    for i in range(0, len(memories), 20):
        batch = memories[i:i+20]
        batch_results = await asyncio.gather(*[
            classify_memory(m, llm_client) for m in batch
        ])
        results.extend(batch_results)
    
    return results
```


## 6.5 Identity Seeding

Extract identity facts and seed Redis:

```python
async def seed_identity(
    user_id: str,
    memories: list[dict],
    classifications: list[ClassificationResult],
    redis_store: RedisStore
):
    """Extract identity facts from classified memories and seed Redis."""
    
    identity = {}
    
    for mem, classification in zip(memories, classifications):
        if classification.memory_type == MemoryType.IDENTITY:
            key = classification.identity_key
            if key:
                # For complex keys like 'family', accumulate
                if key in ["family", "preferences", "key_facts"]:
                    if key not in identity:
                        identity[key] = []
                    identity[key].append(mem["content"])
                else:
                    # Simple keys like 'name', 'occupation'
                    identity[key] = mem["content"]
    
    # Post-process accumulated fields
    if "family" in identity:
        identity["family"] = await summarize_family(identity["family"])
    if "key_facts" in identity:
        identity["key_facts"] = identity["key_facts"][:20]  # Cap at 20
    
    await redis_store.set_identity(user_id, identity)
    return identity

async def summarize_family(facts: list[str]) -> dict:
    """Use LLM to synthesize family facts into structured data."""
    prompt = f"""
    Synthesize these facts about a user's family into structured JSON:
    
    Facts:
    {chr(10).join(f'- {f}' for f in facts)}
    
    Return JSON like:
    {{"children": ["name1", "name2"], "spouse": "name", "pets": ["pet1"]}}
    """
    # ... LLM call
```

## 6.6 Import Script

```python
async def import_memories(
    user_id: str,
    memories: list[dict],
    classifications: list[ClassificationResult],
    postgres_store: PostgresStore,
    embedder: Embedder,
    scorer: EmotionScorer
):
    """Import classified memories to Postgres."""
    
    imported = 0
    skipped = 0
    
    for mem, classification in zip(memories, classifications):
        # Skip low-confidence classifications
        if classification.confidence < 0.5:
            skipped += 1
            continue
        
        # Score emotion
        emotional_score = await scorer.score(mem["content"])
        
        # Generate embedding
        embedding = await embedder.embed(mem["content"])
        
        # Create memory object
        memory = Memory(
            content=mem["content"],
            user_id=user_id,
            memory_type=classification.memory_type,
            emotional_score=emotional_score,
            embedding=embedding,
            source="migration",
            metadata={
                "original_id": mem["id"],
                "migrated_at": datetime.utcnow().isoformat()
            }
        )
        
        try:
            await postgres_store.store(memory)
            imported += 1
        except Exception as e:
            logger.error(f"Failed to import {mem['id']}: {e}")
            skipped += 1
    
    return {"imported": imported, "skipped": skipped}
```


## 6.7 Full Migration Runbook

```python
async def run_full_migration(
    user_id: str,
    mem0_client: MemoryClient,
    redis_store: RedisStore,
    postgres_store: PostgresStore,
    embedder: Embedder,
    scorer: EmotionScorer,
    llm_client
) -> MigrationReport:
    """Run full migration for a single user."""
    
    report = MigrationReport(user_id=user_id, started_at=datetime.utcnow())
    
    try:
        # 1. Export from mem0
        logger.info(f"Exporting memories for {user_id}")
        memories = await export_mem0_memories(user_id, mem0_client)
        report.total_exported = len(memories)
        
        # 2. Classify all memories
        logger.info(f"Classifying {len(memories)} memories")
        contents = [m["content"] for m in memories]
        classifications = await classify_batch(contents, llm_client)
        
        # 3. Seed identity to Redis
        logger.info("Seeding identity facts")
        identity = await seed_identity(user_id, memories, classifications, redis_store)
        report.identity_facts = len(identity)
        
        # 4. Import to Postgres
        logger.info("Importing to Postgres")
        result = await import_memories(
            user_id, memories, classifications,
            postgres_store, embedder, scorer
        )
        report.imported = result["imported"]
        report.skipped = result["skipped"]
        
        report.success = True
        
    except Exception as e:
        report.success = False
        report.error = str(e)
        logger.exception(f"Migration failed for {user_id}")
    
    report.completed_at = datetime.utcnow()
    return report

@dataclass
class MigrationReport:
    user_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    total_exported: int = 0
    identity_facts: int = 0
    imported: int = 0
    skipped: int = 0
    error: Optional[str] = None
```

---

## 6.8 Validation

After migration, validate data integrity:

```python
async def validate_migration(
    user_id: str,
    mem0_client: MemoryClient,
    redis_store: RedisStore,
    postgres_store: PostgresStore
) -> ValidationReport:
    """Validate migration was successful."""
    
    report = ValidationReport(user_id=user_id)
    
    # Check identity exists
    identity = await redis_store.get_identity(user_id)
    report.has_identity = bool(identity)
    
    # Check memory counts
    original = await export_mem0_memories(user_id, mem0_client)
    migrated = await postgres_store.get_memories(user_id=user_id)
    
    report.original_count = len(original)
    report.migrated_count = len(migrated)
    report.count_match = abs(len(original) - len(migrated)) < len(original) * 0.1  # Within 10%
    
    # Spot check: search for a known fact
    if identity.get("name"):
        results = await postgres_store.search(
            user_id=user_id,
            query=f"name is {identity['name']}",
            limit=5
        )
        report.search_works = len(results) > 0
    
    return report
```

---

## 6.9 Acceptance Criteria

- [ ] All users migrated without data loss
- [ ] Identity facts extracted and seeded to Redis
- [ ] Memories classified with >80% confidence on average
- [ ] Validation passes for all users
- [ ] mem0 remains accessible as read-only backup

# Section 7: Phase 3 - Write-Through Pattern

## 7.1 Overview

Phase 3 implements the core MemoryManager with write-through semantics:
- Writes go to Redis immediately (working memory)
- Async queue writes to Postgres (permanent storage)
- Reads merge both sources

**Duration**: 2 weeks
**Dependencies**: Phase 1 (Redis), Phase 2 (Migration)
**Deliverables**: MemoryManager class, Postgres store, embedder integration

---

## 7.2 PostgresStore Implementation

```python
import asyncpg
from pgvector.asyncpg import register_vector

class PostgresStore:
    def __init__(self, config: PostgresConfig):
        self.config = config
        self.pool: asyncpg.Pool = None
    
    async def connect(self):
        self.pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            min_size=5,
            max_size=20
        )
        
        # Register pgvector type
        async with self.pool.acquire() as conn:
            await register_vector(conn)
    
    async def close(self):
        if self.pool:
            await self.pool.close()
    
    async def store(self, memory: Memory) -> str:
        """Store a memory and return its ID."""
        query = """
            INSERT INTO memories (
                user_id, project_id, content, memory_type,
                emotional_score, importance, confidence,
                source, tags, metadata, embedding, status
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
            ) RETURNING id
        """
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                memory.user_id,
                memory.project_id,
                memory.content,
                memory.memory_type.value,
                memory.emotional_score,
                memory.importance,
                memory.confidence,
                memory.source,
                memory.tags,
                json.dumps(memory.metadata),
                memory.embedding,
                memory.status.value
            )
            return str(row["id"])
    
    async def get(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID."""
        query = "SELECT * FROM memories WHERE id = $1"
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, memory_id)
            if row:
                return self._row_to_memory(row)
            return None
    
    def _row_to_memory(self, row) -> Memory:
        """Convert database row to Memory object."""
        return Memory(
            id=str(row["id"]),
            user_id=row["user_id"],
            project_id=row["project_id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            emotional_score=row["emotional_score"],
            importance=row["importance"],
            confidence=row["confidence"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            supersedes=str(row["supersedes"]) if row["supersedes"] else None,
            source=row["source"],
            tags=row["tags"] or [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            embedding=list(row["embedding"]) if row["embedding"] else None,
            status=MemoryStatus(row["status"])
        )
```


## 7.3 Vector Search

```python
    async def search(
        self,
        user_id: str,
        embedding: List[float],
        project_id: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 20
    ) -> List[Memory]:
        """Semantic search using pgvector."""
        
        # Build query with filters
        conditions = ["user_id = $1", "status = 'active'"]
        params = [user_id]
        param_idx = 2
        
        if project_id:
            conditions.append(f"(project_id = ${param_idx} OR project_id IS NULL)")
            params.append(project_id)
            param_idx += 1
        
        if memory_types:
            type_values = [t.value for t in memory_types]
            conditions.append(f"memory_type = ANY(${param_idx})")
            params.append(type_values)
            param_idx += 1
        
        params.append(embedding)
        
        query = f"""
            SELECT *, 
                   1 - (embedding <=> ${param_idx}) as similarity
            FROM memories
            WHERE {' AND '.join(conditions)}
            ORDER BY embedding <=> ${param_idx}
            LIMIT {limit}
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            memories = []
            for row in rows:
                mem = self._row_to_memory(row)
                mem.metadata["similarity"] = row["similarity"]
                memories.append(mem)
            return memories
    
    async def get_memories(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
        memory_types: Optional[List[MemoryType]] = None,
        since: Optional[datetime] = None,
        before: Optional[datetime] = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        limit: int = 100
    ) -> List[Memory]:
        """Get memories with filters (no vector search)."""
        
        conditions = ["user_id = $1", "status = $2"]
        params = [user_id, status.value]
        param_idx = 3
        
        if memory_type:
            conditions.append(f"memory_type = ${param_idx}")
            params.append(memory_type.value)
            param_idx += 1
        
        if memory_types:
            conditions.append(f"memory_type = ANY(${param_idx})")
            params.append([t.value for t in memory_types])
            param_idx += 1
        
        if since:
            conditions.append(f"created_at >= ${param_idx}")
            params.append(since)
            param_idx += 1
        
        if before:
            conditions.append(f"created_at < ${param_idx}")
            params.append(before)
            param_idx += 1
        
        query = f"""
            SELECT * FROM memories
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
            LIMIT {limit}
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._row_to_memory(row) for row in rows]
```


## 7.4 Embedder Implementation

```python
from openai import AsyncOpenAI

class OpenAIEmbedder:
    def __init__(self, config: EmbeddingConfig):
        self.client = AsyncOpenAI(api_key=config.api_key)
        self.model = config.model or "text-embedding-3-small"
        self.dimensions = config.dimensions or 1536
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions
        )
        return response.data[0].embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        # OpenAI supports batch embedding
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions
        )
        return [item.embedding for item in response.data]

# Alternative: Local embeddings with sentence-transformers
class LocalEmbedder:
    def __init__(self, config: EmbeddingConfig):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(config.model or 'all-MiniLM-L6-v2')
    
    async def embed(self, text: str) -> List[float]:
        # Run in thread pool to not block
        import asyncio
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, 
            lambda: self.model.encode(text).tolist()
        )
        return embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        import asyncio
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(texts).tolist()
        )
        return embeddings
```

## 7.5 Emotion Scorer Implementation

```python
class EmotionScorer:
    def __init__(self, config: LLMConfig):
        self.client = AsyncOpenAI(api_key=config.api_key)
        self.model = config.model or "gpt-4o-mini"  # Fast and cheap
    
    async def score(self, content: str) -> float:
        """Score emotional intensity of content."""
        prompt = f"""Rate emotional intensity from 0.0 to 1.0:
- 0.0-0.2: Mundane (grabbed coffee)
- 0.2-0.4: Mildly notable (good meeting)
- 0.4-0.6: Significant (completed project)
- 0.6-0.8: Highly emotional (family emergency)
- 0.8-1.0: Profound (major life event)

Memory: {content}

Return ONLY a number."""
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5
    
    async def score_batch(self, contents: List[str]) -> List[float]:
        """Score multiple memories."""
        return await asyncio.gather(*[self.score(c) for c in contents])
```


## 7.6 Complete MemoryManager

```python
class MemoryManager:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.redis = RedisStore(config.redis)
        self.postgres = PostgresStore(config.postgres)
        self.embedder = OpenAIEmbedder(config.embedding)
        self.scorer = EmotionScorer(config.llm)
        self._initialized = False
    
    async def initialize(self):
        await self.redis.connect()
        await self.postgres.connect()
        self._initialized = True
    
    async def close(self):
        await self.redis.close()
        await self.postgres.close()
    
    async def get_context(
        self,
        user_id: str,
        query: Optional[str] = None,
        project_id: Optional[str] = None,
        include_working: bool = True,
        max_retrieved: int = 20
    ) -> MemoryContext:
        """Get full context for a conversation turn."""
        
        # Always load identity and session from Redis (fast)
        identity = await self.redis.get_identity(user_id)
        session = await self.redis.get_session(user_id)
        
        # Get working memory if requested
        working = []
        if include_working:
            working = await self.redis.get_working(user_id)
        
        # Semantic search if query provided
        retrieved = []
        if query:
            embedding = await self.embedder.embed(query)
            retrieved = await self.postgres.search(
                user_id=user_id,
                embedding=embedding,
                project_id=project_id,
                limit=max_retrieved
            )
            retrieved = self._rerank(retrieved)
        
        return MemoryContext(
            user_id=user_id,
            identity=identity,
            session=session,
            working=working,
            retrieved=retrieved
        )
    
    async def store(
        self,
        user_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        project_id: Optional[str] = None,
        emotional_score: Optional[float] = None,
        metadata: Optional[dict] = None
    ) -> StoreResult:
        """Store a new memory with write-through."""
        
        try:
            # Score emotion if not provided
            if emotional_score is None:
                emotional_score = await self.scorer.score(content)
            
            # Calculate TTL for working memory
            ttl = calculate_ttl(emotional_score)
            
            # Create memory
            memory = Memory(
                content=content,
                user_id=user_id,
                memory_type=memory_type,
                project_id=project_id,
                emotional_score=emotional_score,
                metadata=metadata or {}
            )
            
            # 1. Write to Redis immediately (fast)
            await self.redis.add_working(user_id, memory, ttl=ttl)
            
            # 2. Generate embedding
            memory.embedding = await self.embedder.embed(content)
            
            # 3. Write to Postgres
            memory.id = await self.postgres.store(memory)
            
            # 4. Add to recent buffer
            await self.redis.add_recent(user_id, memory.id)
            
            return StoreResult(
                success=True,
                memory_id=memory.id,
                emotional_score=emotional_score,
                ttl=ttl
            )
            
        except Exception as e:
            return StoreResult(success=False, error=str(e))
    
    def _rerank(self, memories: List[Memory]) -> List[Memory]:
        """Re-rank memories using full scoring algorithm."""
        now = datetime.utcnow()
        
        scored = []
        for mem in memories:
            similarity = mem.metadata.get("similarity", 0.5)
            score = calculate_retrieval_score(mem, similarity, now)
            scored.append((score, mem))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored]
```


## 7.7 Integration with Clara

```python
# In Clara's main message handler

class ClaraBot:
    def __init__(self):
        self.memory = MemoryManager(memory_config)
    
    async def handle_message(self, message: Message):
        user_id = f"discord-{message.author.id}"
        
        # Get context for this turn
        context = await self.memory.get_context(
            user_id=user_id,
            query=message.content,  # Use message as search query
            include_working=True
        )
        
        # Build system prompt with memory context
        system_prompt = BASE_SYSTEM_PROMPT + "\n\n" + context.to_prompt_string()
        
        # Generate response
        response = await self.llm.complete(
            system_prompt=system_prompt,
            user_message=message.content
        )
        
        # Store this interaction
        await self.memory.store(
            user_id=user_id,
            content=f"User said: {message.content[:200]}",
            memory_type=MemoryType.EPISODIC
        )
        
        # Extract and store any notable facts
        facts = await self.extract_facts(message.content, response)
        for fact in facts:
            await self.memory.store(
                user_id=user_id,
                content=fact,
                memory_type=MemoryType.SEMANTIC
            )
        
        return response
```

---

## 7.8 Tests

```python
@pytest.fixture
async def memory_manager():
    config = MemoryConfig(
        redis=RedisConfig(host="localhost", db=15),
        postgres=PostgresConfig(database="clara_test"),
        embedding=EmbeddingConfig(api_key=os.environ["OPENAI_API_KEY"]),
        llm=LLMConfig(api_key=os.environ["OPENAI_API_KEY"])
    )
    manager = MemoryManager(config)
    await manager.initialize()
    yield manager
    await manager.close()

class TestWriteThrough:
    async def test_store_appears_in_working_memory(self, memory_manager):
        user_id = "test-user"
        
        result = await memory_manager.store(
            user_id=user_id,
            content="Test memory content"
        )
        
        assert result.success
        
        # Should appear in working memory immediately
        context = await memory_manager.get_context(user_id)
        working_contents = [m.content for m in context.working]
        assert "Test memory content" in working_contents
    
    async def test_store_searchable_in_postgres(self, memory_manager):
        user_id = "test-user-2"
        
        await memory_manager.store(
            user_id=user_id,
            content="I love chocolate ice cream"
        )
        
        # Should be searchable
        context = await memory_manager.get_context(
            user_id=user_id,
            query="ice cream preferences"
        )
        
        assert len(context.retrieved) > 0
        assert any("chocolate" in m.content for m in context.retrieved)

---

## 7.9 Acceptance Criteria

- [ ] MemoryManager initializes and connects to both stores
- [ ] store() writes to Redis immediately, Postgres within 100ms
- [ ] get_context() returns combined identity + session + working + retrieved
- [ ] Semantic search returns relevant results
- [ ] Emotional scoring affects TTL correctly
- [ ] Integration with Clara message handler works

# Section 8: Phase 4 - Consolidation Jobs

## 8.1 Overview

Phase 4 implements background jobs that maintain memory health:
- Pattern extraction from episodic memories
- Identity fact updates
- Contradiction detection and resolution
- Old memory compaction
- Decay and cleanup

**Duration**: 2 weeks
**Dependencies**: Phase 3 complete
**Deliverables**: Consolidator class, job scheduler, monitoring

---

## 8.2 Consolidator Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     CONSOLIDATOR                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Pattern   │  │ Contradiction│  │  Compaction │    │
│  │  Extractor  │  │  Detector    │  │    Job      │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                │                │            │
│         ▼                ▼                ▼            │
│  ┌─────────────────────────────────────────────────┐   │
│  │                   Job Runner                     │   │
│  │  - Scheduled (cron-like)                        │   │
│  │  - Per-user queuing                             │   │
│  │  - Concurrency limits                           │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                             │
└──────────────────────────┼─────────────────────────────┘
                           ▼
              ┌─────────────────────────┐
              │   Redis + Postgres      │
              └─────────────────────────┘
```

---

## 8.3 Pattern Extractor

```python
class PatternExtractor:
    def __init__(self, llm_client, postgres: PostgresStore, redis: RedisStore):
        self.llm = llm_client
        self.postgres = postgres
        self.redis = redis
    
    async def extract_for_user(self, user_id: str) -> List[Pattern]:
        """Extract patterns from recent episodic memories."""
        
        # Get last 7 days of episodic memories
        memories = await self.postgres.get_memories(
            user_id=user_id,
            memory_type=MemoryType.EPISODIC,
            since=datetime.utcnow() - timedelta(days=7),
            limit=100
        )
        
        if len(memories) < 5:
            return []  # Not enough data
        
        # Ask LLM to identify patterns
        prompt = self._build_pattern_prompt(memories)
        response = await self.llm.complete(prompt, max_tokens=1000)
        
        patterns = self._parse_patterns(response)
        return patterns
    
    def _build_pattern_prompt(self, memories: List[Memory]) -> str:
        memory_text = "\n".join([
            f"- [{m.created_at.strftime('%Y-%m-%d')}] {m.content}"
            for m in memories[:50]  # Limit to avoid token limits
        ])
        
        return f"""Analyze these recent memories and identify stable patterns about this user.

Memories:
{memory_text}

For each pattern, provide:
1. category: identity | preference | behavior | relationship
2. fact: The pattern statement
3. confidence: 0.0 to 1.0
4. evidence: Which memories support this

Only report patterns with confidence >= 0.7.
Return as JSON array.

Example:
[
  {{"category": "identity", "fact": "works as software engineer", "confidence": 0.9, "evidence": [0, 5, 12]}},
  {{"category": "preference", "fact": "prefers direct communication", "confidence": 0.75, "evidence": [3, 8]}}
]"""
    
    def _parse_patterns(self, response: str) -> List[Pattern]:
        try:
            data = json.loads(response)
            return [
                Pattern(
                    category=p["category"],
                    fact=p["fact"],
                    confidence=p["confidence"],
                    evidence=p.get("evidence", [])
                )
                for p in data
            ]
        except (json.JSONDecodeError, KeyError):
            return []
    
    async def apply_patterns(self, user_id: str, patterns: List[Pattern]):
        """Apply high-confidence patterns to identity."""
        
        for pattern in patterns:
            if pattern.confidence < 0.8:
                continue
            
            if pattern.category == "identity":
                # Update Redis identity
                key = self._infer_identity_key(pattern.fact)
                await self.redis.update_identity_field(user_id, key, pattern.fact)
            
            # Store as semantic memory
            await self.postgres.store(Memory(
                user_id=user_id,
                content=pattern.fact,
                memory_type=MemoryType.SEMANTIC,
                confidence=pattern.confidence,
                source="consolidation",
                metadata={"pattern_category": pattern.category}
            ))
```


## 8.4 Contradiction Detector

```python
class ContradictionDetector:
    def __init__(self, llm_client, postgres: PostgresStore):
        self.llm = llm_client
        self.postgres = postgres
    
    async def detect_for_user(self, user_id: str) -> List[Contradiction]:
        """Find contradicting memories."""
        
        # Get facts that could contradict (identity + semantic)
        facts = await self.postgres.get_memories(
            user_id=user_id,
            memory_types=[MemoryType.IDENTITY, MemoryType.SEMANTIC],
            status=MemoryStatus.ACTIVE,
            limit=200
        )
        
        if len(facts) < 2:
            return []
        
        # Cluster by embedding similarity
        clusters = self._cluster_by_similarity(facts, threshold=0.8)
        
        contradictions = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            
            # Check cluster for contradictions
            found = await self._check_cluster(cluster)
            contradictions.extend(found)
        
        return contradictions
    
    def _cluster_by_similarity(
        self, 
        memories: List[Memory], 
        threshold: float
    ) -> List[List[Memory]]:
        """Cluster memories by embedding similarity."""
        from sklearn.cluster import AgglomerativeClustering
        import numpy as np
        
        embeddings = np.array([m.embedding for m in memories if m.embedding])
        
        if len(embeddings) < 2:
            return []
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - threshold,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)
        
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(memories[i])
        
        return list(clusters.values())
    
    async def _check_cluster(self, memories: List[Memory]) -> List[Contradiction]:
        """Ask LLM if any memories in cluster contradict."""
        
        memory_text = "\n".join([
            f"{i}. {m.content}" for i, m in enumerate(memories)
        ])
        
        prompt = f"""Do any of these statements contradict each other?

Statements:
{memory_text}

If contradictions exist, return JSON array:
[{{"a": <index>, "b": <index>, "reason": "explanation"}}]

If no contradictions, return: []"""
        
        response = await self.llm.complete(prompt, max_tokens=500)
        
        try:
            data = json.loads(response)
            return [
                Contradiction(
                    memory_a=memories[c["a"]].id,
                    memory_b=memories[c["b"]].id,
                    reason=c["reason"]
                )
                for c in data
            ]
        except (json.JSONDecodeError, KeyError, IndexError):
            return []
    
    async def resolve_contradiction(
        self, 
        contradiction: Contradiction,
        resolution: str,  # "a_wins" | "b_wins" | "merge" | "both_valid"
        note: Optional[str] = None
    ):
        """Resolve a detected contradiction."""
        
        if resolution == "a_wins":
            await self.postgres.update_status(
                contradiction.memory_b, 
                MemoryStatus.SUPERSEDED
            )
        elif resolution == "b_wins":
            await self.postgres.update_status(
                contradiction.memory_a,
                MemoryStatus.SUPERSEDED
            )
        elif resolution == "merge":
            # Create merged memory, supersede both
            mem_a = await self.postgres.get(contradiction.memory_a)
            mem_b = await self.postgres.get(contradiction.memory_b)
            
            merged_content = await self._merge_memories(mem_a, mem_b)
            
            new_mem = Memory(
                user_id=mem_a.user_id,
                content=merged_content,
                memory_type=mem_a.memory_type,
                source="contradiction_merge"
            )
            await self.postgres.store(new_mem)
            
            await self.postgres.update_status(mem_a.id, MemoryStatus.SUPERSEDED)
            await self.postgres.update_status(mem_b.id, MemoryStatus.SUPERSEDED)
        
        # Log resolution
        await self.postgres.log_contradiction_resolution(
            contradiction.id, resolution, note
        )
```


## 8.5 Compaction Job

```python
class CompactionJob:
    def __init__(self, llm_client, postgres: PostgresStore):
        self.llm = llm_client
        self.postgres = postgres
    
    async def compact_user(self, user_id: str, older_than_days: int = 30):
        """Compact old episodic memories into summaries."""
        
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        
        # Get old episodic memories
        old_memories = await self.postgres.get_memories(
            user_id=user_id,
            memory_type=MemoryType.EPISODIC,
            before=cutoff,
            status=MemoryStatus.ACTIVE,
            limit=500
        )
        
        if len(old_memories) < 10:
            return  # Not enough to compact
        
        # Group by week
        weeks = self._group_by_week(old_memories)
        
        for week_start, memories in weeks.items():
            if len(memories) < 5:
                continue
            
            # Summarize the week
            summary = await self._summarize_week(memories, week_start)
            
            # Store summary as semantic memory
            await self.postgres.store(Memory(
                user_id=user_id,
                content=summary,
                memory_type=MemoryType.SEMANTIC,
                source="compaction",
                metadata={
                    "week_start": week_start.isoformat(),
                    "memories_compacted": len(memories)
                }
            ))
            
            # Archive originals
            for mem in memories:
                await self.postgres.update_status(mem.id, MemoryStatus.ARCHIVED)
    
    def _group_by_week(self, memories: List[Memory]) -> dict:
        """Group memories by week."""
        weeks = {}
        for mem in memories:
            week_start = mem.created_at - timedelta(days=mem.created_at.weekday())
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            
            if week_start not in weeks:
                weeks[week_start] = []
            weeks[week_start].append(mem)
        
        return weeks
    
    async def _summarize_week(
        self, 
        memories: List[Memory], 
        week_start: datetime
    ) -> str:
        """Generate summary of a week's memories."""
        
        memory_text = "\n".join([
            f"- {m.content}" for m in memories[:30]
        ])
        
        prompt = f"""Summarize this week's memories into a concise paragraph.
Preserve important facts, emotions, and events. Omit mundane details.

Week of {week_start.strftime('%B %d, %Y')}:
{memory_text}

Summary (2-3 sentences):"""
        
        response = await self.llm.complete(prompt, max_tokens=200)
        return response.strip()

## 8.6 Job Runner

```python
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler

class ConsolidationRunner:
    def __init__(
        self,
        postgres: PostgresStore,
        redis: RedisStore,
        llm_client
    ):
        self.postgres = postgres
        self.redis = redis
        self.llm = llm_client
        
        self.pattern_extractor = PatternExtractor(llm_client, postgres, redis)
        self.contradiction_detector = ContradictionDetector(llm_client, postgres)
        self.compaction_job = CompactionJob(llm_client, postgres)
        
        self.scheduler = AsyncIOScheduler()
        self._running = False
    
    def start(self):
        """Start the consolidation scheduler."""
        
        # Run every 6 hours
        self.scheduler.add_job(
            self.run_all_users,
            'interval',
            hours=6,
            id='consolidation_main'
        )
        
        # Cleanup job every hour
        self.scheduler.add_job(
            self.cleanup_expired,
            'interval',
            hours=1,
            id='cleanup'
        )
        
        self.scheduler.start()
        self._running = True
    
    def stop(self):
        self.scheduler.shutdown()
        self._running = False
    
    async def run_all_users(self):
        """Run consolidation for all users."""
        user_ids = await self.postgres.get_active_user_ids()
        
        for user_id in user_ids:
            try:
                await self.run_for_user(user_id)
            except Exception as e:
                logger.error(f"Consolidation failed for {user_id}: {e}")
    
    async def run_for_user(self, user_id: str) -> ConsolidationLog:
        """Run full consolidation for one user."""
        log = ConsolidationLog(user_id=user_id, started_at=datetime.utcnow())
        
        # 1. Extract patterns
        patterns = await self.pattern_extractor.extract_for_user(user_id)
        await self.pattern_extractor.apply_patterns(user_id, patterns)
        log.patterns_found = len(patterns)
        
        # 2. Detect contradictions
        contradictions = await self.contradiction_detector.detect_for_user(user_id)
        for c in contradictions:
            await self.postgres.flag_contradiction(user_id, c)
        log.contradictions_found = len(contradictions)
        
        # 3. Compact old memories
        await self.compaction_job.compact_user(user_id)
        
        log.completed_at = datetime.utcnow()
        await self.postgres.log_consolidation(log)
        
        return log
    
    async def cleanup_expired(self):
        """Clean up expired working memories."""
        user_ids = await self.redis.get_active_user_ids()
        for user_id in user_ids:
            await self.redis.cleanup_expired_working(user_id)
```


---

## 8.7 Monitoring

```python
@dataclass
class ConsolidationMetrics:
    users_processed: int = 0
    patterns_extracted: int = 0
    contradictions_found: int = 0
    memories_compacted: int = 0
    errors: int = 0
    duration_seconds: float = 0.0

class ConsolidationMonitor:
    def __init__(self, postgres: PostgresStore):
        self.postgres = postgres
    
    async def get_recent_runs(self, hours: int = 24) -> List[ConsolidationLog]:
        """Get consolidation runs from last N hours."""
        return await self.postgres.get_consolidation_logs(
            since=datetime.utcnow() - timedelta(hours=hours)
        )
    
    async def get_metrics(self, hours: int = 24) -> ConsolidationMetrics:
        """Aggregate metrics from recent runs."""
        runs = await self.get_recent_runs(hours)
        
        return ConsolidationMetrics(
            users_processed=len(runs),
            patterns_extracted=sum(r.patterns_found for r in runs),
            contradictions_found=sum(r.contradictions_found for r in runs),
            memories_compacted=sum(r.memories_compacted for r in runs),
            errors=sum(1 for r in runs if r.error),
            duration_seconds=sum(
                (r.completed_at - r.started_at).total_seconds() 
                for r in runs if r.completed_at
            )
        )
    
    async def get_pending_contradictions(self) -> List[Contradiction]:
        """Get unresolved contradictions."""
        return await self.postgres.get_contradictions(resolution="pending")

---

## 8.8 Acceptance Criteria

- [ ] Pattern extraction identifies real patterns from episodic memories
- [ ] High-confidence patterns update Redis identity
- [ ] Contradiction detection finds conflicting facts
- [ ] Compaction summarizes old memories correctly
- [ ] Scheduler runs reliably every 6 hours
- [ ] Monitoring dashboard shows health metrics
- [ ] Errors are logged and don't crash the system

---

## 8.9 Configuration

```python
@dataclass
class ConsolidationConfig:
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
    
    # Contradiction
    similarity_threshold: float = 0.8
    
    # Limits
    max_memories_per_run: int = 500
    max_concurrent_users: int = 5
```
# Section 9: Phase 5 - Multi-User Support

## 9.1 Overview

Phase 5 adds proper multi-user support inspired by HuixiangDou patterns:
- Shared project memories across team members
- User-specific views of shared context
- Rejection pipeline for irrelevant queries
- Access control and permissions

**Duration**: 2 weeks
**Dependencies**: Phase 4 complete
**Deliverables**: Multi-user MemoryManager, project sharing, rejection pipeline

---

## 9.2 HuixiangDou Patterns

Key patterns from HuixiangDou we're adopting:

### 1. Rejection Pipeline
Not every message needs memory retrieval. Use a fast classifier to reject irrelevant queries:

```python
class RejectionPipeline:
    """Quickly reject messages that don't need memory."""
    
    REJECTION_PATTERNS = [
        r'^(hi|hello|hey|sup)$',  # Simple greetings
        r'^(thanks|thank you|thx)$',  # Thanks
        r'^(ok|okay|k|sure)$',  # Acknowledgments
        r'^\W*$',  # Empty or just punctuation
    ]
    
    def __init__(self, llm_client=None):
        self.patterns = [re.compile(p, re.I) for p in self.REJECTION_PATTERNS]
        self.llm = llm_client
    
    def quick_reject(self, message: str) -> bool:
        """Fast pattern-based rejection."""
        message = message.strip()
        
        # Too short
        if len(message) < 3:
            return True
        
        # Matches rejection pattern
        for pattern in self.patterns:
            if pattern.match(message):
                return True
        
        return False
    
    async def should_retrieve(self, message: str, session: dict) -> bool:
        """Determine if this message needs memory retrieval."""
        
        # Fast rejection first
        if self.quick_reject(message):
            return False
        
        # If in active task, might not need retrieval
        if session.get("active_task"):
            # LLM decides if this is continuation or new topic
            return await self._needs_context_switch(message, session)
        
        return True
    
    async def _needs_context_switch(self, message: str, session: dict) -> bool:
        """Use LLM to determine if topic is changing."""
        prompt = f"""Is this message continuing the current topic or switching to something new?

Current topic: {session.get('current_topic', 'none')}
Message: {message}

Reply: CONTINUE or SWITCH"""
        
        response = await self.llm.complete(prompt, max_tokens=10)
        return "SWITCH" in response.upper()
```


### 2. Group Chat Handling

```python
class GroupChatMemory:
    """Handle memory in group chat contexts."""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager
        self.rejection = RejectionPipeline()
    
    async def get_context_for_message(
        self,
        message: str,
        author_id: str,
        channel_id: str,
        mentioned: bool = False
    ) -> Optional[MemoryContext]:
        """Get context for a group chat message."""
        
        # If not mentioned in group, apply stricter filtering
        if not mentioned:
            if self.rejection.quick_reject(message):
                return None
        
        # Get author's personal context
        personal = await self.memory.get_context(
            user_id=author_id,
            query=message
        )
        
        # Get shared project context if channel is linked to project
        project_id = await self.get_channel_project(channel_id)
        project_context = None
        if project_id:
            project_context = await self.memory.get_project_context(
                project_id=project_id,
                query=message
            )
        
        return GroupContext(
            personal=personal,
            project=project_context,
            channel_id=channel_id
        )
    
    async def store_group_message(
        self,
        message: str,
        author_id: str,
        channel_id: str
    ):
        """Store a group message appropriately."""
        
        project_id = await self.get_channel_project(channel_id)
        
        # Store to author's personal memory
        await self.memory.store(
            user_id=author_id,
            content=message,
            project_id=project_id
        )
        
        # If project-related, also store to project memory
        if project_id:
            await self.memory.store_project_memory(
                project_id=project_id,
                content=message,
                author_id=author_id
            )
```

## 9.3 Project Sharing

```python
class ProjectMemory:
    """Shared memory for projects/teams."""
    
    def __init__(self, postgres: PostgresStore, redis: RedisStore):
        self.postgres = postgres
        self.redis = redis
    
    async def create_project(
        self,
        project_id: str,
        name: str,
        owner_id: str,
        members: List[str] = None
    ):
        """Create a new project with shared memory."""
        
        project = {
            "id": project_id,
            "name": name,
            "owner_id": owner_id,
            "members": members or [owner_id],
            "created_at": datetime.utcnow().isoformat()
        }
        
        await self.postgres.create_project(project)
        await self.redis.set_project_meta(project_id, project)
    
    async def add_member(self, project_id: str, user_id: str):
        """Add a member to a project."""
        await self.postgres.add_project_member(project_id, user_id)
        
        # Update Redis cache
        meta = await self.redis.get_project_meta(project_id)
        if meta:
            meta["members"].append(user_id)
            await self.redis.set_project_meta(project_id, meta)
    
    async def get_project_context(
        self,
        project_id: str,
        query: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> ProjectContext:
        """Get context for a project."""
        
        # Check access
        if user_id:
            has_access = await self.check_access(project_id, user_id)
            if not has_access:
                raise PermissionError(f"User {user_id} cannot access {project_id}")
        
        # Get project metadata
        meta = await self.redis.get_project_meta(project_id)
        
        # Get recent project memories
        recent = await self.postgres.get_memories(
            project_id=project_id,
            limit=20
        )
        
        # Search if query provided
        retrieved = []
        if query:
            retrieved = await self.postgres.search(
                project_id=project_id,
                query=query,
                limit=15
            )
        
        return ProjectContext(
            project_id=project_id,
            name=meta.get("name"),
            recent=recent,
            retrieved=retrieved
        )
    
    async def store_project_memory(
        self,
        project_id: str,
        content: str,
        author_id: str,
        memory_type: MemoryType = MemoryType.PROJECT
    ):
        """Store a memory to a project."""
        
        memory = Memory(
            content=content,
            user_id=author_id,  # Track who added it
            project_id=project_id,
            memory_type=memory_type,
            source="project"
        )
        
        await self.postgres.store(memory)
    
    async def check_access(self, project_id: str, user_id: str) -> bool:
        """Check if user has access to project."""
        meta = await self.redis.get_project_meta(project_id)
        if not meta:
            return False
        
        return user_id in meta.get("members", [])
```


## 9.4 Updated MemoryManager

```python
class MemoryManager:
    """Updated MemoryManager with multi-user support."""
    
    # ... previous methods ...
    
    def __init__(self, config: MemoryConfig):
        # ... previous init ...
        self.project_memory = ProjectMemory(self.postgres, self.redis)
        self.rejection = RejectionPipeline()
        self.group_handler = GroupChatMemory(self)
    
    async def get_context(
        self,
        user_id: str,
        query: Optional[str] = None,
        project_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        skip_retrieval: bool = False
    ) -> MemoryContext:
        """Get context with multi-user awareness."""
        
        # Check if we should skip retrieval
        if not skip_retrieval and query:
            session = await self.redis.get_session(user_id)
            if not await self.rejection.should_retrieve(query, session):
                skip_retrieval = True
        
        # Get personal context
        identity = await self.redis.get_identity(user_id)
        session = await self.redis.get_session(user_id)
        working = await self.redis.get_working(user_id)
        
        # Retrieved memories (if not skipped)
        retrieved = []
        if not skip_retrieval and query:
            embedding = await self.embedder.embed(query)
            retrieved = await self.postgres.search(
                user_id=user_id,
                embedding=embedding,
                project_id=project_id,
                limit=20
            )
        
        # Project context (if applicable)
        project = None
        if project_id:
            project = await self.project_memory.get_project_context(
                project_id=project_id,
                query=query,
                user_id=user_id
            )
        
        return MemoryContext(
            user_id=user_id,
            identity=identity,
            session=session,
            working=working,
            retrieved=retrieved,
            project=project
        )
    
    async def get_group_context(
        self,
        message: str,
        author_id: str,
        channel_id: str,
        mentioned: bool = False
    ) -> Optional[MemoryContext]:
        """Get context for group chat message."""
        return await self.group_handler.get_context_for_message(
            message=message,
            author_id=author_id,
            channel_id=channel_id,
            mentioned=mentioned
        )
```

## 9.5 Channel-Project Linking

```python
class ChannelProjectLink:
    """Link Discord channels to projects."""
    
    def __init__(self, redis: RedisStore):
        self.redis = redis
    
    async def link_channel(self, channel_id: str, project_id: str):
        """Link a channel to a project."""
        key = f"clara:channel_project:{channel_id}"
        await self.redis.client.set(key, project_id)
    
    async def unlink_channel(self, channel_id: str):
        """Unlink a channel from its project."""
        key = f"clara:channel_project:{channel_id}"
        await self.redis.client.delete(key)
    
    async def get_project_for_channel(self, channel_id: str) -> Optional[str]:
        """Get the project linked to a channel."""
        key = f"clara:channel_project:{channel_id}"
        return await self.redis.client.get(key)
    
    async def get_channels_for_project(self, project_id: str) -> List[str]:
        """Get all channels linked to a project."""
        # This requires scanning, consider a reverse index
        pattern = "clara:channel_project:*"
        channels = []
        async for key in self.redis.client.scan_iter(match=pattern):
            value = await self.redis.client.get(key)
            if value == project_id:
                channel_id = key.split(":")[-1]
                channels.append(channel_id)
        return channels
```


## 9.6 Database Changes

```sql
-- Projects table
CREATE TABLE projects (
    id              VARCHAR(255) PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    owner_id        VARCHAR(255) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    settings        JSONB DEFAULT '{}'
);

-- Project members
CREATE TABLE project_members (
    project_id      VARCHAR(255) REFERENCES projects(id),
    user_id         VARCHAR(255) NOT NULL,
    role            VARCHAR(50) DEFAULT 'member',  -- owner, admin, member
    joined_at       TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (project_id, user_id)
);

-- Add project_id index to memories
CREATE INDEX idx_memories_project ON memories(project_id) WHERE project_id IS NOT NULL;
```

---

## 9.7 Acceptance Criteria

- [ ] Projects can be created with owner and members
- [ ] Project memories are shared across members
- [ ] Non-members cannot access project memories
- [ ] Channels can be linked to projects
- [ ] Rejection pipeline skips irrelevant messages
- [ ] Group chat context merges personal + project memories
- [ ] Performance: rejection decision < 10ms

---

## 9.8 Usage Example

```python
# Create a project
await memory.project_memory.create_project(
    project_id="cortex",
    name="Cortex",
    owner_id="discord-123",
    members=["discord-123", "discord-456"]
)

# Link a channel
await memory.channel_links.link_channel(
    channel_id="discord-channel-789",
    project_id="cortex"
)

# In message handler
context = await memory.get_group_context(
    message=message.content,
    author_id=f"discord-{message.author.id}",
    channel_id=str(message.channel.id),
    mentioned=bot.user in message.mentions
)

if context:
    # Use context for response
    ...
else:
    # No context needed, simple response
    ...
```
# Section 10: Testing, Rollback, and Success Metrics

## 10.1 Testing Strategy

### Unit Tests

Every component has isolated unit tests:

```python
# tests/unit/test_redis_store.py
class TestRedisStore:
    async def test_identity_crud(self): ...
    async def test_session_lifecycle(self): ...
    async def test_working_memory_ttl(self): ...

# tests/unit/test_postgres_store.py
class TestPostgresStore:
    async def test_store_and_retrieve(self): ...
    async def test_vector_search(self): ...
    async def test_memory_types(self): ...

# tests/unit/test_algorithms.py
class TestAlgorithms:
    def test_ttl_calculation(self): ...
    def test_retrieval_scoring(self): ...
    def test_importance_scoring(self): ...
```

### Integration Tests

Test component interactions:

```python
# tests/integration/test_memory_manager.py
class TestMemoryManager:
    async def test_write_through(self):
        """Verify writes go to both Redis and Postgres."""
        ...
    
    async def test_get_context_merges_sources(self):
        """Verify context includes all sources."""
        ...
    
    async def test_session_affects_retrieval(self):
        """Verify session state influences retrieval."""
        ...

# tests/integration/test_consolidation.py
class TestConsolidation:
    async def test_pattern_extraction(self): ...
    async def test_contradiction_detection(self): ...
    async def test_compaction(self): ...
```

### End-to-End Tests

Test full Clara integration:

```python
# tests/e2e/test_clara_memory.py
class TestClaraMemory:
    async def test_conversation_flow(self):
        """Simulate multi-turn conversation."""
        # Send messages
        # Verify memory is stored
        # Verify retrieval includes relevant history
        ...
    
    async def test_identity_persistence(self):
        """Verify identity facts persist across sessions."""
        ...
    
    async def test_emotional_decay(self):
        """Verify high-emotion memories persist longer."""
        ...
```

### Performance Tests

```python
# tests/performance/test_latency.py
class TestLatency:
    async def test_get_context_under_100ms(self):
        """Context retrieval should be <100ms p95."""
        times = []
        for _ in range(100):
            start = time.time()
            await memory.get_context(user_id, query="test")
            times.append(time.time() - start)
        
        p95 = sorted(times)[95]
        assert p95 < 0.1  # 100ms
    
    async def test_store_under_50ms(self):
        """Store should be <50ms p95."""
        ...
```


## 10.2 Rollback Plan

### Phase 1: Parallel Operation

```
Week 1-2: New system deployed but not used
- Both systems receive writes
- Only mem0 serves reads
- Compare outputs

Week 3-4: Shadow Mode
- New system serves reads
- Results compared to mem0 but not used
- Log discrepancies

Week 5+: Gradual Cutover
- Route 10% → 50% → 100% of reads to new system
- mem0 remains read-only backup
```

### Rollback Triggers

Automatic rollback if:
- Error rate > 5% for 5 minutes
- Latency p95 > 500ms for 10 minutes
- Memory usage > 80% of limit
- User reports "Clara forgot me"

### Rollback Procedure

```python
class RollbackManager:
    async def initiate_rollback(self, reason: str):
        """Switch back to mem0."""
        
        # 1. Log rollback
        logger.critical(f"Initiating rollback: {reason}")
        await self.notify_oncall(reason)
        
        # 2. Switch reads to mem0
        await self.config.set("memory_backend", "mem0")
        
        # 3. Keep new system running for writes (dual-write)
        await self.config.set("dual_write", True)
        
        # 4. Alert for investigation
        await self.create_incident(reason)
    
    async def verify_rollback(self):
        """Verify mem0 is serving correctly."""
        # Test retrieval
        context = await self.mem0.get_context(test_user)
        assert context is not None
        
        # Verify latency
        assert await self.check_latency() < 0.2
```

### Data Preservation

```python
# Before any migration
async def backup_mem0():
    """Full backup of mem0 data."""
    export = await mem0_client.export_all()
    
    # Save to S3
    await s3.put_object(
        Bucket="clara-backups",
        Key=f"mem0/full-backup-{datetime.now().isoformat()}.json",
        Body=json.dumps(export)
    )
    
    return export
```


## 10.3 Success Metrics

### Core Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Retrieval Relevance | >80% | User feedback, A/B testing |
| Latency (p50) | <50ms | Application metrics |
| Latency (p95) | <100ms | Application metrics |
| Error Rate | <0.1% | Error logs |
| Memory Growth | <1GB/month/user | Database size monitoring |

### Quality Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| "Clara remembered" rate | >90% | Track when Clara correctly uses past context |
| Contradiction rate | <5% | Consolidation logs |
| Stale memory rate | <10% | Memories >90 days old still active |
| Identity accuracy | >95% | Manual review of identity facts |

### Operational Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Consolidation success | >99% | Job completion logs |
| Migration accuracy | >99% | Post-migration validation |
| Redis hit rate | >80% | Redis stats |
| Embedding API cost | <$0.01/user/day | OpenAI billing |

## 10.4 Monitoring Dashboard

```python
class MemoryDashboard:
    """Expose metrics for monitoring."""
    
    async def get_health(self) -> dict:
        return {
            "redis_connected": await self.redis.ping(),
            "postgres_connected": await self.postgres.ping(),
            "embedder_healthy": await self.embedder.health_check(),
            "consolidation_running": self.consolidator.is_running(),
            "last_consolidation": await self.get_last_consolidation_time(),
        }
    
    async def get_stats(self, hours: int = 24) -> dict:
        return {
            "total_users": await self.postgres.count_users(),
            "total_memories": await self.postgres.count_memories(),
            "memories_stored": await self.get_stores_count(hours),
            "retrievals": await self.get_retrieval_count(hours),
            "avg_latency_ms": await self.get_avg_latency(hours),
            "error_count": await self.get_error_count(hours),
        }
    
    async def get_user_stats(self, user_id: str) -> dict:
        return {
            "identity_facts": len(await self.redis.get_identity(user_id)),
            "total_memories": await self.postgres.count_user_memories(user_id),
            "working_memories": len(await self.redis.get_working(user_id)),
            "last_active": await self.get_user_last_active(user_id),
        }
```


## 10.5 Open Questions

Before implementation, decide:

1. **Embedding Model**: OpenAI (cost, quality) vs. local sentence-transformers (free, self-hosted)?
   - Recommendation: Start with OpenAI, migrate to local if cost becomes issue

2. **LLM for Scoring**: gpt-4o-mini (quality) vs. Claude Haiku (speed) vs. local (free)?
   - Recommendation: gpt-4o-mini for now, it's fast and cheap

3. **Redis Persistence**: RDB snapshots vs. AOF vs. both?
   - Recommendation: RDB + AOF for durability

4. **Consolidation Frequency**: 6 hours default, but configurable per-user?
   - Recommendation: Start with 6h global, add per-user later if needed

5. **Vector Dimensions**: 1536 (OpenAI default) vs. 384 (MiniLM)?
   - Recommendation: Match your embedding model

6. **Project Sharing UX**: How do users create/manage projects?
   - Recommendation: Discord commands first, web UI later

---

## 10.6 Implementation Checklist

### Phase 1: Redis Foundation
- [ ] RedisStore class with full test coverage
- [ ] Redis deployed and configured
- [ ] Basic CLI for debugging
- [ ] Documentation

### Phase 2: Migration
- [ ] Export script for mem0
- [ ] Classification pipeline
- [ ] Identity seeding
- [ ] Import to Postgres
- [ ] Validation report

### Phase 3: Write-Through
- [ ] PostgresStore with pgvector
- [ ] Embedder integration
- [ ] EmotionScorer
- [ ] MemoryManager
- [ ] Clara integration

### Phase 4: Consolidation
- [ ] PatternExtractor
- [ ] ContradictionDetector
- [ ] CompactionJob
- [ ] Scheduler
- [ ] Monitoring

### Phase 5: Multi-User
- [ ] Project model
- [ ] Rejection pipeline
- [ ] Group chat handler
- [ ] Channel linking
- [ ] Access control

### Launch
- [ ] Parallel operation verified
- [ ] Shadow mode tested
- [ ] Gradual rollout complete
- [ ] Rollback tested
- [ ] Monitoring active
- [ ] Documentation complete

---

## 10.7 Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Redis | 1-2 weeks | Redis instance |
| Phase 2: Migration | 1 week | Phase 1 |
| Phase 3: Write-Through | 2 weeks | Phase 2 |
| Phase 4: Consolidation | 2 weeks | Phase 3 |
| Phase 5: Multi-User | 2 weeks | Phase 4 |
| Testing & Rollout | 2 weeks | All phases |

**Total: 10-12 weeks**

---

## 10.8 Conclusion

This TDD provides a comprehensive blueprint for Clara's new memory system. The tiered architecture (Redis hot + Postgres warm) with emotional decay and consolidation should provide:

1. **Better relevance**: Emotional scoring + multi-signal retrieval
2. **Faster response**: Always-loaded identity/session from Redis
3. **Healthier data**: Contradiction detection + compaction
4. **Multi-user ready**: Project sharing + rejection pipeline

The parallel operation and rollback plan ensure we can safely transition from mem0 without risking data loss or user experience degradation.

Let's build it.