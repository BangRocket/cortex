"""Unit tests for Neo4j graph store."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cortex.config import Neo4jConfig
from cortex.models import Entity, Relationship


class TestGraphStoreImport:
    """Test GraphStore import behavior."""

    def test_neo4j_not_available_raises_error(self):
        """Test that GraphStore raises error when neo4j not installed."""
        with patch.dict("sys.modules", {"neo4j": None}):
            # Force reimport
            import cortex.stores.graph_store as gs

            # Store original value
            original = gs.NEO4J_AVAILABLE
            gs.NEO4J_AVAILABLE = False

            try:
                with pytest.raises(ImportError, match="neo4j package not installed"):
                    gs.GraphStore(Neo4jConfig())
            finally:
                gs.NEO4J_AVAILABLE = original


class AsyncSessionContextManager:
    """Helper class that implements async context manager protocol."""

    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *args):
        pass


class TestGraphStoreWithMock:
    """Test GraphStore with mocked Neo4j driver."""

    @pytest.fixture
    def config(self) -> Neo4jConfig:
        """Create test config."""
        return Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
            database="test",
        )

    @pytest.fixture
    def mock_driver(self):
        """Create mock Neo4j driver."""
        driver = MagicMock()
        session = AsyncMock()

        # Make session() return an async context manager
        driver.session.return_value = AsyncSessionContextManager(session)
        driver.close = AsyncMock()

        return driver, session

    @pytest.fixture
    def entity(self) -> Entity:
        """Create test entity."""
        return Entity(
            id="entity-1",
            name="Test Person",
            entity_type="person",
            user_id="user-1",
            description="A test person",
            confidence=0.9,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            mention_count=1,
            source_memories=["memory-1"],
            attributes={"key": "value"},
        )

    @pytest.fixture
    def relationship(self) -> Relationship:
        """Create test relationship."""
        return Relationship(
            id="rel-1",
            source_id="entity-1",
            target_id="entity-2",
            relationship_type="knows",
            user_id="user-1",
            description="Test relationship",
            confidence=0.8,
            strength=1.0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            source_memories=["memory-1"],
        )

    @pytest.mark.asyncio
    async def test_connect(self, config: Neo4jConfig, mock_driver):
        """Test connection to Neo4j."""
        driver, session = mock_driver

        with patch("cortex.stores.graph_store.NEO4J_AVAILABLE", True):
            with patch("cortex.stores.graph_store.AsyncGraphDatabase") as mock_gdb:
                mock_gdb.driver.return_value = driver

                from cortex.stores.graph_store import GraphStore
                store = GraphStore(config)
                await store.connect()

                mock_gdb.driver.assert_called_once_with(
                    config.uri,
                    auth=(config.user, config.password),
                    max_connection_pool_size=config.max_connection_pool_size,
                    connection_timeout=config.connection_timeout,
                )
                session.run.assert_called_once_with("RETURN 1")

    @pytest.mark.asyncio
    async def test_close(self, config: Neo4jConfig, mock_driver):
        """Test closing Neo4j connection."""
        driver, _ = mock_driver

        with patch("cortex.stores.graph_store.NEO4J_AVAILABLE", True):
            with patch("cortex.stores.graph_store.AsyncGraphDatabase") as mock_gdb:
                mock_gdb.driver.return_value = driver

                from cortex.stores.graph_store import GraphStore
                store = GraphStore(config)
                await store.connect()
                await store.close()

                driver.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_entity(self, config: Neo4jConfig, mock_driver, entity: Entity):
        """Test creating an entity."""
        driver, session = mock_driver

        # Mock the result
        mock_result = AsyncMock()
        mock_result.single.return_value = {"id": entity.id}
        session.run.return_value = mock_result

        with patch("cortex.stores.graph_store.NEO4J_AVAILABLE", True):
            with patch("cortex.stores.graph_store.AsyncGraphDatabase") as mock_gdb:
                mock_gdb.driver.return_value = driver

                from cortex.stores.graph_store import GraphStore
                store = GraphStore(config)
                store.driver = driver

                result = await store.create_entity(entity)

                assert result == entity.id
                session.run.assert_called()
                call_args = session.run.call_args
                assert "MERGE" in call_args[0][0]
                assert call_args[1]["name"] == entity.name
                assert call_args[1]["entity_type"] == entity.entity_type

    @pytest.mark.asyncio
    async def test_get_entity(self, config: Neo4jConfig, mock_driver, entity: Entity):
        """Test getting an entity by ID."""
        driver, session = mock_driver

        # Mock the result
        mock_result = AsyncMock()
        mock_result.single.return_value = {
            "e": {
                "id": entity.id,
                "name": entity.name,
                "entity_type": entity.entity_type,
                "user_id": entity.user_id,
                "description": entity.description,
                "confidence": entity.confidence,
                "created_at": entity.created_at.isoformat(),
                "updated_at": entity.updated_at.isoformat(),
                "mention_count": entity.mention_count,
                "source_memories": entity.source_memories,
                "attributes": "{}",
            }
        }
        session.run.return_value = mock_result

        with patch("cortex.stores.graph_store.NEO4J_AVAILABLE", True):
            with patch("cortex.stores.graph_store.AsyncGraphDatabase") as mock_gdb:
                mock_gdb.driver.return_value = driver

                from cortex.stores.graph_store import GraphStore
                store = GraphStore(config)
                store.driver = driver

                result = await store.get_entity(entity.id)

                assert result is not None
                assert result.id == entity.id
                assert result.name == entity.name
                assert result.entity_type == entity.entity_type

    @pytest.mark.asyncio
    async def test_get_entity_not_found(self, config: Neo4jConfig, mock_driver):
        """Test getting a non-existent entity."""
        driver, session = mock_driver

        mock_result = AsyncMock()
        mock_result.single.return_value = None
        session.run.return_value = mock_result

        with patch("cortex.stores.graph_store.NEO4J_AVAILABLE", True):
            with patch("cortex.stores.graph_store.AsyncGraphDatabase") as mock_gdb:
                mock_gdb.driver.return_value = driver

                from cortex.stores.graph_store import GraphStore
                store = GraphStore(config)
                store.driver = driver

                result = await store.get_entity("nonexistent")

                assert result is None

    @pytest.mark.asyncio
    async def test_delete_entity(self, config: Neo4jConfig, mock_driver):
        """Test deleting an entity."""
        driver, session = mock_driver

        mock_result = AsyncMock()
        mock_result.single.return_value = {"deleted": 1}
        session.run.return_value = mock_result

        with patch("cortex.stores.graph_store.NEO4J_AVAILABLE", True):
            with patch("cortex.stores.graph_store.AsyncGraphDatabase") as mock_gdb:
                mock_gdb.driver.return_value = driver

                from cortex.stores.graph_store import GraphStore
                store = GraphStore(config)
                store.driver = driver

                result = await store.delete_entity("entity-1")

                assert result is True
                call_args = session.run.call_args
                assert "DETACH DELETE" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_create_relationship(
        self, config: Neo4jConfig, mock_driver, relationship: Relationship
    ):
        """Test creating a relationship."""
        driver, session = mock_driver

        mock_result = AsyncMock()
        mock_result.single.return_value = {"id": relationship.id}
        session.run.return_value = mock_result

        with patch("cortex.stores.graph_store.NEO4J_AVAILABLE", True):
            with patch("cortex.stores.graph_store.AsyncGraphDatabase") as mock_gdb:
                mock_gdb.driver.return_value = driver

                from cortex.stores.graph_store import GraphStore
                store = GraphStore(config)
                store.driver = driver

                result = await store.create_relationship(relationship)

                assert result == relationship.id
                call_args = session.run.call_args
                assert "MERGE" in call_args[0][0]
                assert "RELATES_TO" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_relationships(self, config: Neo4jConfig, mock_driver):
        """Test getting relationships for an entity."""
        driver, session = mock_driver

        mock_result = AsyncMock()
        mock_result.data.return_value = [
            {
                "r": {
                    "id": "rel-1",
                    "relationship_type": "knows",
                    "user_id": "user-1",
                    "description": "Test",
                    "confidence": 0.9,
                    "strength": 1.0,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "source_memories": [],
                    "attributes": "{}",
                },
                "source_id": "entity-1",
                "target_id": "entity-2",
            }
        ]
        session.run.return_value = mock_result

        with patch("cortex.stores.graph_store.NEO4J_AVAILABLE", True):
            with patch("cortex.stores.graph_store.AsyncGraphDatabase") as mock_gdb:
                mock_gdb.driver.return_value = driver

                from cortex.stores.graph_store import GraphStore
                store = GraphStore(config)
                store.driver = driver

                result = await store.get_relationships("entity-1")

                assert len(result) == 1
                assert result[0].relationship_type == "knows"

    @pytest.mark.asyncio
    async def test_get_relationships_direction(self, config: Neo4jConfig, mock_driver):
        """Test getting relationships with direction filter."""
        driver, session = mock_driver

        mock_result = AsyncMock()
        mock_result.data.return_value = []
        session.run.return_value = mock_result

        with patch("cortex.stores.graph_store.NEO4J_AVAILABLE", True):
            with patch("cortex.stores.graph_store.AsyncGraphDatabase") as mock_gdb:
                mock_gdb.driver.return_value = driver

                from cortex.stores.graph_store import GraphStore
                store = GraphStore(config)
                store.driver = driver

                # Test outgoing
                await store.get_relationships("entity-1", direction="outgoing")
                call_args = session.run.call_args
                assert "-[r:RELATES_TO]->" in call_args[0][0]

                # Test incoming
                await store.get_relationships("entity-1", direction="incoming")
                call_args = session.run.call_args
                assert "-[r:RELATES_TO]->" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_ping(self, config: Neo4jConfig, mock_driver):
        """Test ping functionality."""
        driver, session = mock_driver

        with patch("cortex.stores.graph_store.NEO4J_AVAILABLE", True):
            with patch("cortex.stores.graph_store.AsyncGraphDatabase") as mock_gdb:
                mock_gdb.driver.return_value = driver

                from cortex.stores.graph_store import GraphStore
                store = GraphStore(config)
                store.driver = driver

                result = await store.ping()

                assert result is True
                session.run.assert_called_with("RETURN 1")

    @pytest.mark.asyncio
    async def test_count_entities(self, config: Neo4jConfig, mock_driver):
        """Test counting entities."""
        driver, session = mock_driver

        mock_result = AsyncMock()
        mock_result.single.return_value = {"count": 5}
        session.run.return_value = mock_result

        with patch("cortex.stores.graph_store.NEO4J_AVAILABLE", True):
            with patch("cortex.stores.graph_store.AsyncGraphDatabase") as mock_gdb:
                mock_gdb.driver.return_value = driver

                from cortex.stores.graph_store import GraphStore
                store = GraphStore(config)
                store.driver = driver

                result = await store.count_entities("user-1")

                assert result == 5

    @pytest.mark.asyncio
    async def test_count_relationships(self, config: Neo4jConfig, mock_driver):
        """Test counting relationships."""
        driver, session = mock_driver

        mock_result = AsyncMock()
        mock_result.single.return_value = {"count": 3}
        session.run.return_value = mock_result

        with patch("cortex.stores.graph_store.NEO4J_AVAILABLE", True):
            with patch("cortex.stores.graph_store.AsyncGraphDatabase") as mock_gdb:
                mock_gdb.driver.return_value = driver

                from cortex.stores.graph_store import GraphStore
                store = GraphStore(config)
                store.driver = driver

                result = await store.count_relationships("user-1")

                assert result == 3


class TestEntityModel:
    """Test Entity model."""

    def test_entity_creation(self):
        """Test creating an entity."""
        entity = Entity(
            name="Test",
            entity_type="person",
            user_id="user-1",
        )

        assert entity.name == "Test"
        assert entity.entity_type == "person"
        assert entity.id is not None  # Auto-generated
        assert entity.confidence == 1.0
        assert entity.mention_count == 1

    def test_entity_with_all_fields(self):
        """Test entity with all fields."""
        entity = Entity(
            id="custom-id",
            name="Test",
            entity_type="organization",
            user_id="user-1",
            description="Test org",
            confidence=0.9,
            mention_count=5,
            source_memories=["m1", "m2"],
            attributes={"industry": "tech"},
        )

        assert entity.id == "custom-id"
        assert entity.description == "Test org"
        assert entity.mention_count == 5
        assert entity.attributes["industry"] == "tech"


class TestRelationshipModel:
    """Test Relationship model."""

    def test_relationship_creation(self):
        """Test creating a relationship."""
        rel = Relationship(
            source_id="e1",
            target_id="e2",
            relationship_type="knows",
            user_id="user-1",
        )

        assert rel.source_id == "e1"
        assert rel.target_id == "e2"
        assert rel.relationship_type == "knows"
        assert rel.id is not None  # Auto-generated
        assert rel.strength == 1.0

    def test_relationship_with_all_fields(self):
        """Test relationship with all fields."""
        rel = Relationship(
            id="custom-id",
            source_id="e1",
            target_id="e2",
            relationship_type="works_at",
            user_id="user-1",
            description="Works at company",
            confidence=0.95,
            strength=2.5,
            source_memories=["m1"],
            attributes={"role": "engineer"},
        )

        assert rel.id == "custom-id"
        assert rel.description == "Works at company"
        assert rel.strength == 2.5
        assert rel.attributes["role"] == "engineer"
