"""Multi-user support with project sharing and rejection pipeline."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

from cortex.models import Memory, MemoryContext, MemoryType, ProjectContext

if TYPE_CHECKING:
    from cortex.manager import MemoryManager
    from cortex.stores.postgres_store import PostgresStore
    from cortex.stores.redis_store import RedisStore

logger = structlog.get_logger(__name__)


# ==================== REJECTION PIPELINE ====================


class RejectionPipeline:
    """
    Quickly reject messages that don't need memory retrieval.

    Inspired by HuixiangDou patterns for efficient group chat handling.
    """

    REJECTION_PATTERNS = [
        r"^(hi|hello|hey|sup|yo)[\s!.]*$",  # Simple greetings
        r"^(thanks|thank you|thx|ty)[\s!.]*$",  # Thanks
        r"^(ok|okay|k|sure|yep|yeah|yes|no|nope)[\s!.]*$",  # Acknowledgments
        r"^\W*$",  # Empty or just punctuation
        r"^(lol|lmao|haha|hehe)[\s!.]*$",  # Laughter
        r"^[\U0001F600-\U0001F64F\s]+$",  # Just emojis
    ]

    def __init__(
        self,
        enable_llm_check: bool = False,
        scorer=None,
        response_threshold: float = 0.5,
    ) -> None:
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.REJECTION_PATTERNS]
        self.enable_llm_check = enable_llm_check
        self.scorer = scorer
        self.response_threshold = response_threshold

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
        if session.get("active_task") and self.enable_llm_check and self.scorer:
            return await self._needs_context_switch(message, session)

        return True

    async def _needs_context_switch(self, message: str, session: dict) -> bool:
        """Use LLM to determine if topic is changing."""
        prompt = f"""Is this message continuing the current topic or switching to something new?

Current topic: {session.get('current_topic', 'none')}
Message: {message}

Reply: CONTINUE or SWITCH"""

        try:
            response = await self.scorer.client.chat.completions.create(
                model=self.scorer.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            return "SWITCH" in response.choices[0].message.content.upper()
        except Exception:
            return True  # Default to retrieving if check fails

    async def should_respond(
        self,
        context: MemoryContext,
        message: str,
        mentioned: bool = False,
    ) -> bool:
        """
        Determine if assistant should respond based on context.

        Args:
            context: The memory context for this user
            message: The incoming message
            mentioned: Whether the assistant was directly addressed

        Returns:
            True if assistant should respond, False to stay silent
        """
        # Always respond if directly mentioned
        if mentioned:
            return True

        # Fast rejection for trivial messages
        if self.quick_reject(message):
            return False

        # If no working memory context, wait to be addressed
        if not context.working:
            return False

        # If LLM check is enabled and we have a scorer, evaluate relevance
        if self.enable_llm_check and self.scorer:
            relevance = await self._calculate_relevance(message, context)
            return relevance >= self.response_threshold

        # Default: respond if we have context
        return True

    async def _calculate_relevance(self, message: str, context: MemoryContext) -> float:
        """Calculate relevance score for a message given context."""
        # Build context summary
        recent_topics = [m.content[:100] for m in context.working[:5]]
        context_summary = "\n".join(recent_topics) if recent_topics else "No recent context"

        prompt = f"""Rate how relevant this message is to the current conversation context.

Recent context:
{context_summary}

New message: {message}

Rate relevance from 0.0 (completely off-topic/not for me) to 1.0 (highly relevant).
Reply with just a number."""

        try:
            response = await self.scorer.client.chat.completions.create(
                model=self.scorer.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            match = re.search(r"(\d+\.?\d*)", text)
            if match:
                return float(match.group(1))
            return 0.5  # Default mid-range if parsing fails
        except Exception:
            return 0.5  # Default to mid-range on error


# ==================== PROJECT MEMORY ====================


class ProjectMemory:
    """Shared memory for projects/teams."""

    def __init__(self, postgres: "PostgresStore", redis: "RedisStore") -> None:
        self.postgres = postgres
        self.redis = redis

    async def create_project(
        self,
        project_id: str,
        name: str,
        owner_id: str,
        members: list[str] | None = None,
    ) -> None:
        """Create a new project with shared memory."""
        project = {
            "id": project_id,
            "name": name,
            "owner_id": owner_id,
            "members": members or [owner_id],
        }

        await self.postgres.create_project(project)
        await self.redis.set_project_meta(project_id, project)

        logger.info(
            "project_created",
            project_id=project_id,
            owner=owner_id,
            members=len(project["members"]),
        )

    async def add_member(self, project_id: str, user_id: str) -> None:
        """Add a member to a project."""
        await self.postgres.add_project_member(project_id, user_id)

        # Update Redis cache
        meta = await self.redis.get_project_meta(project_id)
        if meta:
            if "members" not in meta:
                meta["members"] = []
            if user_id not in meta["members"]:
                meta["members"].append(user_id)
            await self.redis.set_project_meta(project_id, meta)

        logger.info("project_member_added", project_id=project_id, user_id=user_id)

    async def get_project_context(
        self,
        project_id: str,
        query: str | None = None,
        user_id: str | None = None,
        embedder=None,
    ) -> ProjectContext:
        """Get context for a project."""
        # Check access
        if user_id:
            has_access = await self.check_access(project_id, user_id)
            if not has_access:
                raise PermissionError(f"User {user_id} cannot access project {project_id}")

        # Get project metadata
        meta = await self.redis.get_project_meta(project_id)

        # Get recent project memories
        recent = await self.postgres.get_memories(
            project_id=project_id,
            limit=20,
        )

        # Search if query provided and embedder available
        retrieved = []
        if query and embedder:
            embedding = await embedder.embed(query)
            retrieved = await self.postgres.search(
                user_id=user_id or "",  # Might need project-scoped search
                embedding=embedding,
                project_id=project_id,
                limit=15,
            )

        return ProjectContext(
            project_id=project_id,
            name=meta.get("name") if meta else None,
            recent=recent,
            retrieved=retrieved,
        )

    async def store_project_memory(
        self,
        project_id: str,
        content: str,
        author_id: str,
        embedder=None,
        memory_type: MemoryType = MemoryType.PROJECT,
    ) -> str:
        """Store a memory to a project."""
        memory = Memory(
            content=content,
            user_id=author_id,  # Track who added it
            project_id=project_id,
            memory_type=memory_type,
            source="project",
        )

        if embedder:
            memory.embedding = await embedder.embed(content)

        memory_id = await self.postgres.store(memory)
        logger.debug(
            "project_memory_stored",
            project_id=project_id,
            memory_id=memory_id,
            author=author_id,
        )
        return memory_id

    async def check_access(self, project_id: str, user_id: str) -> bool:
        """Check if user has access to project."""
        meta = await self.redis.get_project_meta(project_id)
        if not meta:
            # Check Postgres
            members = await self.postgres.get_project_members(project_id)
            return user_id in members

        return user_id in meta.get("members", [])


# ==================== GROUP CHAT HANDLER ====================


class GroupChatMemory:
    """Handle memory in group chat contexts."""

    def __init__(
        self,
        memory_manager: "MemoryManager",
        enable_rejection: bool = True,
    ) -> None:
        self.memory = memory_manager
        self.rejection = RejectionPipeline(
            enable_llm_check=False,  # Keep fast for group chats
        )
        self.project_memory = ProjectMemory(
            memory_manager.postgres,
            memory_manager.redis,
        )
        self.channel_links: ChannelProjectLink = ChannelProjectLink(memory_manager.redis)
        self.enable_rejection = enable_rejection

    async def get_context_for_message(
        self,
        message: str,
        author_id: str,
        channel_id: str,
        mentioned: bool = False,
    ) -> MemoryContext | None:
        """Get context for a group chat message."""
        # If not mentioned in group, apply stricter filtering
        if not mentioned and self.enable_rejection:
            if self.rejection.quick_reject(message):
                return None

        # Get author's personal context
        project_id = await self.channel_links.get_project_for_channel(channel_id)

        context = await self.memory.get_context(
            user_id=author_id,
            query=message,
            project_id=project_id,
        )

        return context

    async def store_group_message(
        self,
        message: str,
        author_id: str,
        channel_id: str,
    ) -> None:
        """Store a group message appropriately."""
        project_id = await self.channel_links.get_project_for_channel(channel_id)

        # Store to author's personal memory
        await self.memory.store(
            user_id=author_id,
            content=message,
            project_id=project_id,
        )

        # If project-related, also store to project memory
        if project_id:
            await self.project_memory.store_project_memory(
                project_id=project_id,
                content=message,
                author_id=author_id,
                embedder=self.memory.embedder,
            )


# ==================== CHANNEL LINKING ====================


class ChannelProjectLink:
    """Link Discord/chat channels to projects."""

    def __init__(self, redis: "RedisStore") -> None:
        self.redis = redis
        self.prefix = "cortex:channel_project"

    async def link_channel(self, channel_id: str, project_id: str) -> None:
        """Link a channel to a project."""
        key = f"{self.prefix}:{channel_id}"
        assert self.redis.client is not None
        await self.redis.client.set(key, project_id)
        logger.info("channel_linked", channel_id=channel_id, project_id=project_id)

    async def unlink_channel(self, channel_id: str) -> None:
        """Unlink a channel from its project."""
        key = f"{self.prefix}:{channel_id}"
        assert self.redis.client is not None
        await self.redis.client.delete(key)
        logger.info("channel_unlinked", channel_id=channel_id)

    async def get_project_for_channel(self, channel_id: str) -> str | None:
        """Get the project linked to a channel."""
        key = f"{self.prefix}:{channel_id}"
        assert self.redis.client is not None
        return await self.redis.client.get(key)

    async def get_channels_for_project(self, project_id: str) -> list[str]:
        """Get all channels linked to a project."""
        assert self.redis.client is not None
        pattern = f"{self.prefix}:*"
        channels = []
        async for key in self.redis.client.scan_iter(match=pattern):
            value = await self.redis.client.get(key)
            if value == project_id:
                channel_id = key.split(":")[-1]
                channels.append(channel_id)
        return channels
