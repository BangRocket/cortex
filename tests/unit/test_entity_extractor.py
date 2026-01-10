"""Unit tests for entity extraction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cortex.config import GraphConfig, LLMConfig
from cortex.models import ExtractedEntities
from cortex.utils.entity_extractor import EntityExtractor, extract_entities_from_memory


class TestEntityExtractor:
    """Test EntityExtractor."""

    @pytest.fixture
    def llm_config(self) -> LLMConfig:
        """Create test LLM config."""
        return LLMConfig(
            provider="openai",
            api_key="test-key",
            model="gpt-4o-mini",
        )

    @pytest.fixture
    def graph_config(self) -> GraphConfig:
        """Create test graph config."""
        return GraphConfig(
            enabled=True,
            extract_entities=True,
            infer_relationships=True,
            entity_types=["person", "organization", "location", "project"],
        )

    @pytest.fixture
    def extractor(self, llm_config: LLMConfig, graph_config: GraphConfig) -> EntityExtractor:
        """Create test extractor."""
        return EntityExtractor(llm_config, graph_config)

    def test_init(self, llm_config: LLMConfig, graph_config: GraphConfig):
        """Test extractor initialization."""
        extractor = EntityExtractor(llm_config, graph_config)

        assert extractor.llm_config == llm_config
        assert extractor.graph_config == graph_config
        assert extractor._client is None  # Lazy loaded

    def test_parse_response_simple_json(self, extractor: EntityExtractor):
        """Test parsing simple JSON response."""
        response = '{"entities": [{"name": "Test", "type": "person"}], "relationships": []}'

        result = extractor._parse_response(response)

        assert result["entities"] == [{"name": "Test", "type": "person"}]
        assert result["relationships"] == []

    def test_parse_response_with_markdown(self, extractor: EntityExtractor):
        """Test parsing JSON wrapped in markdown code blocks."""
        response = """```json
{
    "entities": [{"name": "Test", "type": "person"}],
    "relationships": []
}
```"""

        result = extractor._parse_response(response)

        assert result["entities"] == [{"name": "Test", "type": "person"}]

    def test_parse_response_with_extra_text(self, extractor: EntityExtractor):
        """Test parsing JSON with extra text around it."""
        response = """Here are the entities:
{"entities": [{"name": "Alice", "type": "person"}], "relationships": []}
That's all I found."""

        result = extractor._parse_response(response)

        assert result["entities"] == [{"name": "Alice", "type": "person"}]

    def test_parse_response_invalid_json(self, extractor: EntityExtractor):
        """Test handling invalid JSON."""
        response = "This is not valid JSON at all"

        result = extractor._parse_response(response)

        assert result["entities"] == []
        assert result["relationships"] == []

    @pytest.mark.asyncio
    async def test_extract_disabled(self, llm_config: LLMConfig):
        """Test extraction when disabled."""
        graph_config = GraphConfig(extract_entities=False)
        extractor = EntityExtractor(llm_config, graph_config)

        result = await extractor.extract("Test content", "user-1")

        assert result.entities == []
        assert result.relationships == []

    @pytest.mark.asyncio
    async def test_extract_with_openai(self, extractor: EntityExtractor):
        """Test extraction with OpenAI provider."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"entities": [{"name": "Alice", "type": "person", "description": "A person"}], "relationships": []}'
                )
            )
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch.object(extractor, "_client", mock_client):
            result = await extractor.extract(
                "Had a meeting with Alice today.",
                "user-1",
                "memory-1",
            )

        assert len(result.entities) == 1
        assert result.entities[0].name == "Alice"
        assert result.entities[0].entity_type == "person"
        assert result.entities[0].user_id == "user-1"
        assert "memory-1" in result.entities[0].source_memories

    @pytest.mark.asyncio
    async def test_extract_with_anthropic(self, graph_config: GraphConfig):
        """Test extraction with Anthropic provider."""
        llm_config = LLMConfig(
            provider="anthropic",
            api_key="test-key",
            model="claude-3-haiku-20240307",
        )
        extractor = EntityExtractor(llm_config, graph_config)

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='{"entities": [{"name": "Acme Corp", "type": "organization"}], "relationships": []}'
            )
        ]

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = mock_response

        with patch.object(extractor, "_client", mock_client):
            result = await extractor.extract(
                "Working with Acme Corp on a project.",
                "user-1",
            )

        assert len(result.entities) == 1
        assert result.entities[0].name == "Acme Corp"
        assert result.entities[0].entity_type == "organization"

    @pytest.mark.asyncio
    async def test_extract_with_relationships(self, extractor: EntityExtractor):
        """Test extraction with relationship inference."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="""{
                        "entities": [
                            {"name": "Alice", "type": "person", "description": "Developer"},
                            {"name": "Acme", "type": "organization", "description": "Tech company"}
                        ],
                        "relationships": [
                            {"source": "Alice", "target": "Acme", "type": "works_at", "description": "Alice works at Acme"}
                        ]
                    }"""
                )
            )
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch.object(extractor, "_client", mock_client):
            result = await extractor.extract(
                "Alice works at Acme Corp.",
                "user-1",
                "memory-1",
            )

        assert len(result.entities) == 2
        assert len(result.relationships) == 1
        assert result.relationships[0].relationship_type == "works_at"
        assert result.source_memory_id == "memory-1"

    @pytest.mark.asyncio
    async def test_extract_relationships_disabled(self, llm_config: LLMConfig):
        """Test extraction with relationships disabled."""
        graph_config = GraphConfig(
            extract_entities=True,
            infer_relationships=False,
        )
        extractor = EntityExtractor(llm_config, graph_config)

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="""{
                        "entities": [{"name": "Alice", "type": "person"}],
                        "relationships": [{"source": "Alice", "target": "Bob", "type": "knows"}]
                    }"""
                )
            )
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch.object(extractor, "_client", mock_client):
            result = await extractor.extract("Alice knows Bob.", "user-1")

        assert len(result.entities) == 1
        assert len(result.relationships) == 0  # Relationships disabled

    @pytest.mark.asyncio
    async def test_extract_handles_error(self, extractor: EntityExtractor):
        """Test extraction handles errors gracefully."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        with patch.object(extractor, "_client", mock_client):
            result = await extractor.extract("Test content", "user-1")

        assert result.entities == []
        assert result.relationships == []

    @pytest.mark.asyncio
    async def test_extract_batch(self, extractor: EntityExtractor):
        """Test batch extraction."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"entities": [{"name": "Test", "type": "person"}], "relationships": []}'
                )
            )
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch.object(extractor, "_client", mock_client):
            memories = [
                ("Memory 1", "user-1", "m1"),
                ("Memory 2", "user-1", "m2"),
                ("Memory 3", "user-1", "m3"),
            ]

            results = await extractor.extract_batch(memories)

        assert len(results) == 3
        assert all(isinstance(r, ExtractedEntities) for r in results)

    def test_merge_entities(self, extractor: EntityExtractor):
        """Test merging entities."""
        from cortex.models import Entity

        existing = Entity(
            name="Alice",
            entity_type="person",
            user_id="user-1",
            description="Short desc",
            confidence=0.7,
            mention_count=5,
            source_memories=["m1", "m2"],
        )

        new = Entity(
            name="Alice",
            entity_type="person",
            user_id="user-1",
            description="A longer description with more details",
            confidence=0.9,
            source_memories=["m3"],
        )

        result = extractor.merge_entities(existing, new)

        assert result.mention_count == 6  # Incremented
        assert result.source_memories == ["m1", "m2", "m3"]  # Extended
        assert result.description == new.description  # Longer description
        assert result.confidence == 0.9  # Higher confidence

    def test_merge_entities_keeps_existing_description(self, extractor: EntityExtractor):
        """Test merging keeps existing longer description."""
        from cortex.models import Entity

        existing = Entity(
            name="Alice",
            entity_type="person",
            user_id="user-1",
            description="A very long detailed description",
            confidence=0.9,
        )

        new = Entity(
            name="Alice",
            entity_type="person",
            user_id="user-1",
            description="Short",
            confidence=0.7,
        )

        result = extractor.merge_entities(existing, new)

        assert result.description == existing.description  # Kept longer


class TestExtractEntitiesFromMemory:
    """Test convenience function."""

    @pytest.mark.asyncio
    async def test_extract_entities_from_memory(self):
        """Test the convenience function."""
        llm_config = LLMConfig(provider="openai", api_key="test")
        graph_config = GraphConfig(extract_entities=False)  # Disable for quick test

        result = await extract_entities_from_memory(
            content="Test content",
            user_id="user-1",
            memory_id="memory-1",
            llm_config=llm_config,
            graph_config=graph_config,
        )

        assert isinstance(result, ExtractedEntities)
        assert result.entities == []
        assert result.source_memory_id == "memory-1"


class TestExtractedEntitiesModel:
    """Test ExtractedEntities model."""

    def test_extracted_entities_creation(self):
        """Test creating ExtractedEntities."""
        from cortex.models import Entity, Relationship

        entities = [
            Entity(name="Alice", entity_type="person", user_id="user-1"),
            Entity(name="Acme", entity_type="organization", user_id="user-1"),
        ]
        relationships = [
            Relationship(
                source_id=entities[0].id,
                target_id=entities[1].id,
                relationship_type="works_at",
                user_id="user-1",
            )
        ]

        result = ExtractedEntities(
            entities=entities,
            relationships=relationships,
            source_memory_id="memory-1",
        )

        assert len(result.entities) == 2
        assert len(result.relationships) == 1
        assert result.source_memory_id == "memory-1"

    def test_extracted_entities_empty(self):
        """Test empty ExtractedEntities."""
        result = ExtractedEntities(entities=[], relationships=[])

        assert result.entities == []
        assert result.relationships == []
        assert result.source_memory_id is None
