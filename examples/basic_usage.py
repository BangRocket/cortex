"""Basic usage example for Cortex memory system."""

import asyncio
import os

from cortex import CortexConfig, MemoryManager, MemoryType


async def main():
    """Demonstrate basic Cortex usage."""
    # Configuration can be set via environment variables:
    # CORTEX_REDIS_HOST, CORTEX_POSTGRES_HOST, CORTEX_EMBEDDING_API_KEY, etc.
    config = CortexConfig()

    # Create and initialize the memory manager
    manager = MemoryManager(config)
    await manager.initialize()

    try:
        user_id = "demo-user"

        # Set up identity
        print("Setting up identity...")
        await manager.set_identity(user_id, {
            "name": "Josh",
            "occupation": "software engineer",
            "preferences": {"style": "direct", "humor": "dry"},
        })

        # Start a session
        print("Starting session...")
        await manager.start_session(user_id, {"current_topic": "Cortex demo"})

        # Store some memories
        print("\nStoring memories...")

        result = await manager.store(
            user_id,
            "Had a great meeting about the new memory system architecture",
            memory_type=MemoryType.EPISODIC,
        )
        print(f"  Stored memory {result.memory_id}")
        print(f"  Emotional score: {result.emotional_score:.2f}")
        print(f"  TTL: {result.ttl}s")

        await manager.store(
            user_id,
            "Decided to use Redis for hot storage and Postgres for warm storage",
            memory_type=MemoryType.SEMANTIC,
        )

        await manager.store(
            user_id,
            "Working on the TDD document",
            memory_type=MemoryType.WORKING,
        )

        # Get context (like you would every conversation turn)
        print("\nGetting context...")
        context = await manager.get_context(
            user_id,
            query="memory system architecture",
        )

        print(f"\nIdentity: {context.identity}")
        print(f"Session: {context.session}")
        print(f"Working memories: {len(context.working)}")
        print(f"Retrieved memories: {len(context.retrieved)}")

        # Show what would go in a system prompt
        print("\n--- Prompt String ---")
        print(context.to_prompt_string())
        print("--- End ---")

        # Search memories
        print("\nSearching for 'Redis'...")
        results = await manager.search(user_id, "Redis storage")
        for mem in results.memories:
            print(f"  - {mem.content[:60]}... (score: {mem.metadata.get('similarity', 0):.2f})")

        # End session
        print("\nEnding session...")
        await manager.end_session(user_id)

        # Get stats
        stats = await manager.get_stats(user_id)
        print(f"\nStats: {stats}")

    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
