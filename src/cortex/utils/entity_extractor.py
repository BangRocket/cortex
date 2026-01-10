"""Entity extraction from memories using LLM."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import structlog

from cortex.config import GraphConfig, LLMConfig
from cortex.models import Entity, ExtractedEntities, Relationship

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


ENTITY_EXTRACTION_PROMPT = """Extract entities and relationships from this memory.

Memory: {content}

Entity Types: {entity_types}

Return JSON with this structure:
{{
    "entities": [
        {{"name": "entity name", "type": "person|organization|location|project|concept|event", "description": "brief description"}}
    ],
    "relationships": [
        {{"source": "entity1 name", "target": "entity2 name", "type": "relationship type", "description": "relationship description"}}
    ]
}}

Common relationship types:
- person-organization: works_at, founded, leads, member_of
- person-person: knows, collaborates_with, reports_to, related_to
- person-location: lives_in, visited, from
- entity-project: works_on, created, owns
- concept-concept: related_to, part_of, depends_on

Only extract entities that are clearly mentioned. Be concise.
Return empty arrays if no entities found.

Example:
Memory: "Had lunch with Sarah from Acme Corp to discuss the Phoenix project"
{{
    "entities": [
        {{"name": "Sarah", "type": "person", "description": "Contact at Acme Corp"}},
        {{"name": "Acme Corp", "type": "organization", "description": "Company"}},
        {{"name": "Phoenix", "type": "project", "description": "Project being discussed"}}
    ],
    "relationships": [
        {{"source": "Sarah", "target": "Acme Corp", "type": "works_at", "description": "Sarah works at Acme Corp"}},
        {{"source": "Sarah", "target": "Phoenix", "type": "works_on", "description": "Sarah is involved with Phoenix project"}}
    ]
}}"""


class EntityExtractor:
    """
    Extract entities and relationships from memory content using LLM.

    Entities are people, organizations, locations, projects, concepts, etc.
    Relationships are connections between entities (works_at, knows, etc.)
    """

    def __init__(self, llm_config: LLMConfig, graph_config: GraphConfig) -> None:
        self.llm_config = llm_config
        self.graph_config = graph_config
        self._client = None

    @property
    def client(self):
        """Lazy load the LLM client."""
        if self._client is None:
            if self.llm_config.provider == "anthropic":
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(api_key=self.llm_config.api_key)
            else:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.llm_config.api_key)
        return self._client

    async def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """Call the LLM with the appropriate API."""
        if self.llm_config.provider == "anthropic":
            response = await self.client.messages.create(
                model=self.llm_config.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        else:
            response = await self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,
            )
            return response.choices[0].message.content.strip()

    async def extract(
        self,
        content: str,
        user_id: str,
        memory_id: str | None = None,
    ) -> ExtractedEntities:
        """
        Extract entities and relationships from memory content.

        Args:
            content: The memory text to analyze
            user_id: User ID for the entities
            memory_id: Optional memory ID for tracking

        Returns:
            ExtractedEntities with entities and relationships
        """
        if not self.graph_config.extract_entities:
            return ExtractedEntities(entities=[], relationships=[], source_memory_id=memory_id)

        prompt = ENTITY_EXTRACTION_PROMPT.format(
            content=content,
            entity_types=", ".join(self.graph_config.entity_types),
        )

        try:
            response = await self._call_llm(prompt, max_tokens=500)

            # Parse JSON from response
            data = self._parse_response(response)

            # Convert to Entity objects
            entities = []
            entity_name_to_id: dict[str, str] = {}

            for e in data.get("entities", []):
                entity = Entity(
                    name=e.get("name", ""),
                    entity_type=e.get("type", "concept"),
                    user_id=user_id,
                    description=e.get("description"),
                    source_memories=[memory_id] if memory_id else [],
                )
                entities.append(entity)
                entity_name_to_id[entity.name.lower()] = entity.id

            # Convert to Relationship objects
            relationships = []
            if self.graph_config.infer_relationships:
                for r in data.get("relationships", []):
                    source_name = r.get("source", "").lower()
                    target_name = r.get("target", "").lower()

                    source_id = entity_name_to_id.get(source_name)
                    target_id = entity_name_to_id.get(target_name)

                    if source_id and target_id:
                        relationship = Relationship(
                            source_id=source_id,
                            target_id=target_id,
                            relationship_type=r.get("type", "related_to"),
                            user_id=user_id,
                            description=r.get("description"),
                            source_memories=[memory_id] if memory_id else [],
                        )
                        relationships.append(relationship)

            logger.debug(
                "entities_extracted",
                memory_id=memory_id,
                entity_count=len(entities),
                relationship_count=len(relationships),
            )

            return ExtractedEntities(
                entities=entities,
                relationships=relationships,
                source_memory_id=memory_id,
            )

        except Exception as e:
            logger.warning("entity_extraction_failed", error=str(e))
            return ExtractedEntities(entities=[], relationships=[], source_memory_id=memory_id)

    def _parse_response(self, response: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Remove markdown code blocks if present
        if "```" in response:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                response = match.group(1)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return {"entities": [], "relationships": []}

    async def extract_batch(
        self,
        memories: list[tuple[str, str, str | None]],  # (content, user_id, memory_id)
    ) -> list[ExtractedEntities]:
        """
        Extract entities from multiple memories.

        For efficiency, could be parallelized or batched.
        """
        import asyncio

        tasks = [
            self.extract(content, user_id, memory_id)
            for content, user_id, memory_id in memories
        ]

        return await asyncio.gather(*tasks)

    def merge_entities(
        self,
        existing: Entity,
        new: Entity,
    ) -> Entity:
        """
        Merge a new entity mention with an existing entity.

        Increases mention count and adds source memories.
        """
        existing.mention_count += 1
        existing.source_memories.extend(new.source_memories)
        existing.updated_at = new.updated_at

        # Update description if new one is more detailed
        if new.description and (
            not existing.description or len(new.description) > len(existing.description)
        ):
            existing.description = new.description

        # Take higher confidence
        if new.confidence > existing.confidence:
            existing.confidence = new.confidence

        return existing


async def extract_entities_from_memory(
    content: str,
    user_id: str,
    memory_id: str | None,
    llm_config: LLMConfig,
    graph_config: GraphConfig,
) -> ExtractedEntities:
    """
    Convenience function to extract entities from a single memory.
    """
    extractor = EntityExtractor(llm_config, graph_config)
    return await extractor.extract(content, user_id, memory_id)
