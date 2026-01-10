"""Example of integrating Cortex with a Clara-like AI assistant."""

import asyncio
from typing import Any

from cortex import CortexConfig, MemoryContext, MemoryManager, MemoryType


class ClaraBot:
    """Example AI assistant using Cortex for memory."""

    def __init__(self, config: CortexConfig):
        self.memory = MemoryManager(config)
        self.llm = None  # Your LLM client would go here

    async def initialize(self):
        """Initialize the bot."""
        await self.memory.initialize()

    async def close(self):
        """Clean shutdown."""
        await self.memory.close()

    async def handle_message(self, user_id: str, message: str) -> str:
        """Handle an incoming message."""
        # Get context for this turn
        context = await self.memory.get_context(
            user_id=user_id,
            query=message,  # Use message as search query
            include_working=True,
        )

        # Build system prompt with memory context
        system_prompt = self._build_system_prompt(context)

        # Generate response (placeholder - you'd use your LLM here)
        response = await self._generate_response(system_prompt, message)

        # Store this interaction
        await self.memory.store(
            user_id=user_id,
            content=f"User: {message[:200]}",
            memory_type=MemoryType.EPISODIC,
        )

        # Extract and store any notable facts from the conversation
        facts = await self._extract_facts(message, response)
        for fact in facts:
            await self.memory.store(
                user_id=user_id,
                content=fact,
                memory_type=MemoryType.SEMANTIC,
            )

        return response

    def _build_system_prompt(self, context: MemoryContext) -> str:
        """Build system prompt including memory context."""
        base_prompt = """You are Clara, a helpful AI assistant with persistent memory.
You remember past conversations and user preferences.

"""
        return base_prompt + context.to_prompt_string()

    async def _generate_response(self, system_prompt: str, user_message: str) -> str:
        """Generate a response using the LLM."""
        # Placeholder - integrate with your LLM
        return f"I understand you said: {user_message}"

    async def _extract_facts(self, message: str, response: str) -> list[str]:
        """Extract notable facts from conversation to store."""
        # Placeholder - you'd use an LLM to extract facts
        return []

    async def learn_identity(self, user_id: str, key: str, value: Any):
        """Update identity based on learned information."""
        await self.memory.update_identity(user_id, key, value)

    async def start_conversation(self, user_id: str, topic: str | None = None):
        """Start a new conversation session."""
        await self.memory.start_session(user_id, {"current_topic": topic})

    async def end_conversation(self, user_id: str):
        """End the current conversation, persisting important memories."""
        await self.memory.end_session(user_id)


async def demo():
    """Demonstrate Clara integration."""
    config = CortexConfig()
    bot = ClaraBot(config)
    await bot.initialize()

    try:
        user_id = "demo-user"

        # Set up initial identity
        await bot.learn_identity(user_id, "name", "Josh")
        await bot.learn_identity(user_id, "occupation", "software engineer")

        # Start conversation
        await bot.start_conversation(user_id, "AI memory systems")

        # Handle some messages
        response1 = await bot.handle_message(
            user_id,
            "I'm working on a new memory system for AI assistants"
        )
        print(f"Clara: {response1}")

        response2 = await bot.handle_message(
            user_id,
            "It uses Redis for hot storage and Postgres for long-term"
        )
        print(f"Clara: {response2}")

        # End conversation
        await bot.end_conversation(user_id)

    finally:
        await bot.close()


if __name__ == "__main__":
    asyncio.run(demo())
