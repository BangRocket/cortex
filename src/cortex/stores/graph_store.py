"""Neo4j graph store for entity/relationship memory."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import structlog

from cortex.config import Neo4jConfig
from cortex.models import Entity, GraphContext, Relationship

logger = structlog.get_logger(__name__)

# Try to import neo4j, but make it optional
try:
    from neo4j import AsyncGraphDatabase, AsyncDriver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    AsyncGraphDatabase = None  # type: ignore
    AsyncDriver = None  # type: ignore


class GraphStore:
    """
    Neo4j-based graph store for entity/relationship memory.

    Stores:
    - Entities (nodes): People, organizations, locations, concepts, etc.
    - Relationships (edges): Connections between entities

    This provides a knowledge graph layer on top of the vector memory,
    enabling relationship-based retrieval and reasoning.
    """

    def __init__(self, config: Neo4jConfig) -> None:
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "neo4j package not installed. Install with: pip install neo4j"
            )

        self.config = config
        self.driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Initialize Neo4j connection."""
        self.driver = AsyncGraphDatabase.driver(
            self.config.uri,
            auth=(self.config.user, self.config.password),
            max_connection_pool_size=self.config.max_connection_pool_size,
            connection_timeout=self.config.connection_timeout,
        )

        # Verify connection
        async with self.driver.session(database=self.config.database) as session:
            await session.run("RETURN 1")

        logger.info("neo4j_connected", uri=self.config.uri)

    async def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            logger.info("neo4j_disconnected")

    async def initialize_schema(self) -> None:
        """Create indexes and constraints."""
        assert self.driver is not None

        async with self.driver.session(database=self.config.database) as session:
            # Entity constraints and indexes
            await session.run("""
                CREATE CONSTRAINT entity_id IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.id IS UNIQUE
            """)
            await session.run("""
                CREATE INDEX entity_user_id IF NOT EXISTS
                FOR (e:Entity) ON (e.user_id)
            """)
            await session.run("""
                CREATE INDEX entity_name IF NOT EXISTS
                FOR (e:Entity) ON (e.name)
            """)
            await session.run("""
                CREATE INDEX entity_type IF NOT EXISTS
                FOR (e:Entity) ON (e.entity_type)
            """)

            # Full-text search index for entity names
            await session.run("""
                CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
                FOR (e:Entity) ON EACH [e.name, e.description]
            """)

        logger.info("neo4j_schema_initialized")

    # ==================== ENTITY OPERATIONS ====================

    async def create_entity(self, entity: Entity) -> str:
        """Create or update an entity node."""
        assert self.driver is not None

        query = """
            MERGE (e:Entity {user_id: $user_id, name: $name, entity_type: $entity_type})
            ON CREATE SET
                e.id = $id,
                e.description = $description,
                e.confidence = $confidence,
                e.created_at = datetime($created_at),
                e.updated_at = datetime($updated_at),
                e.mention_count = $mention_count,
                e.source_memories = $source_memories,
                e.attributes = $attributes
            ON MATCH SET
                e.description = COALESCE($description, e.description),
                e.confidence = CASE WHEN $confidence > e.confidence THEN $confidence ELSE e.confidence END,
                e.updated_at = datetime($updated_at),
                e.mention_count = e.mention_count + 1,
                e.source_memories = e.source_memories + $source_memories,
                e.attributes = apoc.map.merge(e.attributes, $attributes)
            RETURN e.id as id
        """

        async with self.driver.session(database=self.config.database) as session:
            result = await session.run(
                query,
                id=entity.id,
                user_id=entity.user_id,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                confidence=entity.confidence,
                created_at=entity.created_at.isoformat(),
                updated_at=entity.updated_at.isoformat(),
                mention_count=entity.mention_count,
                source_memories=entity.source_memories,
                attributes=json.dumps(entity.attributes),
            )
            record = await result.single()
            entity_id = record["id"] if record else entity.id

        logger.debug("entity_created", entity_id=entity_id, name=entity.name)
        return entity_id

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        assert self.driver is not None

        query = "MATCH (e:Entity {id: $id}) RETURN e"

        async with self.driver.session(database=self.config.database) as session:
            result = await session.run(query, id=entity_id)
            record = await result.single()

            if record:
                return self._record_to_entity(record["e"])
            return None

    async def get_entity_by_name(
        self, user_id: str, name: str, entity_type: str | None = None
    ) -> Entity | None:
        """Get an entity by name (and optionally type)."""
        assert self.driver is not None

        if entity_type:
            query = """
                MATCH (e:Entity {user_id: $user_id, name: $name, entity_type: $entity_type})
                RETURN e
            """
            params = {"user_id": user_id, "name": name, "entity_type": entity_type}
        else:
            query = "MATCH (e:Entity {user_id: $user_id, name: $name}) RETURN e"
            params = {"user_id": user_id, "name": name}

        async with self.driver.session(database=self.config.database) as session:
            result = await session.run(query, **params)
            record = await result.single()

            if record:
                return self._record_to_entity(record["e"])
            return None

    async def search_entities(
        self,
        user_id: str,
        query: str,
        entity_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[Entity]:
        """Search for entities by name/description."""
        assert self.driver is not None

        # Use full-text search
        cypher = """
            CALL db.index.fulltext.queryNodes('entity_name_fulltext', $query)
            YIELD node, score
            WHERE node.user_id = $user_id
        """

        if entity_types:
            cypher += " AND node.entity_type IN $entity_types"

        cypher += """
            RETURN node
            ORDER BY score DESC
            LIMIT $limit
        """

        async with self.driver.session(database=self.config.database) as session:
            result = await session.run(
                cypher,
                query=query,
                user_id=user_id,
                entity_types=entity_types,
                limit=limit,
            )
            records = await result.data()

            return [self._record_to_entity(r["node"]) for r in records]

    async def get_user_entities(
        self,
        user_id: str,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[Entity]:
        """Get all entities for a user."""
        assert self.driver is not None

        if entity_type:
            query = """
                MATCH (e:Entity {user_id: $user_id, entity_type: $entity_type})
                RETURN e
                ORDER BY e.mention_count DESC
                LIMIT $limit
            """
            params = {"user_id": user_id, "entity_type": entity_type, "limit": limit}
        else:
            query = """
                MATCH (e:Entity {user_id: $user_id})
                RETURN e
                ORDER BY e.mention_count DESC
                LIMIT $limit
            """
            params = {"user_id": user_id, "limit": limit}

        async with self.driver.session(database=self.config.database) as session:
            result = await session.run(query, **params)
            records = await result.data()

            return [self._record_to_entity(r["e"]) for r in records]

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships."""
        assert self.driver is not None

        query = """
            MATCH (e:Entity {id: $id})
            DETACH DELETE e
            RETURN count(e) as deleted
        """

        async with self.driver.session(database=self.config.database) as session:
            result = await session.run(query, id=entity_id)
            record = await result.single()
            return record["deleted"] > 0 if record else False

    # ==================== RELATIONSHIP OPERATIONS ====================

    async def create_relationship(self, relationship: Relationship) -> str:
        """Create or update a relationship between entities."""
        assert self.driver is not None

        # Use MERGE to update existing relationships
        query = """
            MATCH (source:Entity {id: $source_id})
            MATCH (target:Entity {id: $target_id})
            MERGE (source)-[r:RELATES_TO {relationship_type: $relationship_type}]->(target)
            ON CREATE SET
                r.id = $id,
                r.user_id = $user_id,
                r.description = $description,
                r.confidence = $confidence,
                r.strength = $strength,
                r.created_at = datetime($created_at),
                r.updated_at = datetime($updated_at),
                r.source_memories = $source_memories,
                r.attributes = $attributes
            ON MATCH SET
                r.description = COALESCE($description, r.description),
                r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END,
                r.strength = r.strength + 0.1,
                r.updated_at = datetime($updated_at),
                r.source_memories = r.source_memories + $source_memories
            RETURN r.id as id
        """

        async with self.driver.session(database=self.config.database) as session:
            result = await session.run(
                query,
                id=relationship.id,
                source_id=relationship.source_id,
                target_id=relationship.target_id,
                user_id=relationship.user_id,
                relationship_type=relationship.relationship_type,
                description=relationship.description,
                confidence=relationship.confidence,
                strength=relationship.strength,
                created_at=relationship.created_at.isoformat(),
                updated_at=relationship.updated_at.isoformat(),
                source_memories=relationship.source_memories,
                attributes=json.dumps(relationship.attributes),
            )
            record = await result.single()
            rel_id = record["id"] if record else relationship.id

        logger.debug(
            "relationship_created",
            relationship_id=rel_id,
            type=relationship.relationship_type,
        )
        return rel_id

    async def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",  # "outgoing", "incoming", "both"
        relationship_types: list[str] | None = None,
        limit: int = 50,
    ) -> list[Relationship]:
        """Get relationships for an entity."""
        assert self.driver is not None

        if direction == "outgoing":
            match = "(e:Entity {id: $entity_id})-[r:RELATES_TO]->(other)"
        elif direction == "incoming":
            match = "(other)-[r:RELATES_TO]->(e:Entity {id: $entity_id})"
        else:
            match = "(e:Entity {id: $entity_id})-[r:RELATES_TO]-(other)"

        query = f"MATCH {match}"

        if relationship_types:
            query += " WHERE r.relationship_type IN $relationship_types"

        query += """
            RETURN r, startNode(r).id as source_id, endNode(r).id as target_id
            ORDER BY r.strength DESC
            LIMIT $limit
        """

        async with self.driver.session(database=self.config.database) as session:
            result = await session.run(
                query,
                entity_id=entity_id,
                relationship_types=relationship_types,
                limit=limit,
            )
            records = await result.data()

            return [
                self._record_to_relationship(
                    r["r"], r["source_id"], r["target_id"]
                )
                for r in records
            ]

    # ==================== GRAPH QUERIES ====================

    async def get_related_entities(
        self,
        entity_id: str,
        max_depth: int = 2,
        limit: int = 20,
    ) -> list[Entity]:
        """Get entities related to a given entity within N hops."""
        assert self.driver is not None

        query = f"""
            MATCH (start:Entity {{id: $entity_id}})
            MATCH path = (start)-[*1..{max_depth}]-(related:Entity)
            WHERE related <> start
            RETURN DISTINCT related
            ORDER BY length(path), related.mention_count DESC
            LIMIT $limit
        """

        async with self.driver.session(database=self.config.database) as session:
            result = await session.run(query, entity_id=entity_id, limit=limit)
            records = await result.data()

            return [self._record_to_entity(r["related"]) for r in records]

    async def get_graph_context(
        self,
        user_id: str,
        entity_names: list[str],
        max_depth: int = 2,
        max_entities: int = 20,
    ) -> GraphContext:
        """Get graph context for a set of entity names."""
        assert self.driver is not None

        # Find matching entities
        entities: list[Entity] = []
        relationships: list[Relationship] = []
        seen_entity_ids: set[str] = set()

        for name in entity_names:
            entity = await self.get_entity_by_name(user_id, name)
            if entity and entity.id not in seen_entity_ids:
                entities.append(entity)
                seen_entity_ids.add(entity.id)

                # Get related entities
                related = await self.get_related_entities(
                    entity.id,
                    max_depth=max_depth,
                    limit=max_entities // len(entity_names),
                )
                for rel_entity in related:
                    if rel_entity.id not in seen_entity_ids:
                        entities.append(rel_entity)
                        seen_entity_ids.add(rel_entity.id)

                # Get relationships
                rels = await self.get_relationships(entity.id, limit=20)
                relationships.extend(rels)

        return GraphContext(
            entities=entities[:max_entities],
            relationships=relationships,
        )

    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 4,
    ) -> list[str] | None:
        """Find shortest path between two entities."""
        assert self.driver is not None

        query = f"""
            MATCH path = shortestPath(
                (source:Entity {{id: $source_id}})-[*1..{max_depth}]-(target:Entity {{id: $target_id}})
            )
            RETURN [n in nodes(path) | n.name] as path
        """

        async with self.driver.session(database=self.config.database) as session:
            result = await session.run(query, source_id=source_id, target_id=target_id)
            record = await result.single()

            if record:
                return record["path"]
            return None

    # ==================== UTILITIES ====================

    async def ping(self) -> bool:
        """Check if Neo4j is connected."""
        try:
            assert self.driver is not None
            async with self.driver.session(database=self.config.database) as session:
                await session.run("RETURN 1")
            return True
        except Exception:
            return False

    async def count_entities(self, user_id: str | None = None) -> int:
        """Count entities."""
        assert self.driver is not None

        if user_id:
            query = "MATCH (e:Entity {user_id: $user_id}) RETURN count(e) as count"
            params = {"user_id": user_id}
        else:
            query = "MATCH (e:Entity) RETURN count(e) as count"
            params = {}

        async with self.driver.session(database=self.config.database) as session:
            result = await session.run(query, **params)
            record = await result.single()
            return record["count"] if record else 0

    async def count_relationships(self, user_id: str | None = None) -> int:
        """Count relationships."""
        assert self.driver is not None

        if user_id:
            query = "MATCH ()-[r:RELATES_TO {user_id: $user_id}]->() RETURN count(r) as count"
            params = {"user_id": user_id}
        else:
            query = "MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count"
            params = {}

        async with self.driver.session(database=self.config.database) as session:
            result = await session.run(query, **params)
            record = await result.single()
            return record["count"] if record else 0

    async def clear_user_graph(self, user_id: str) -> int:
        """Delete all entities and relationships for a user."""
        assert self.driver is not None

        query = """
            MATCH (e:Entity {user_id: $user_id})
            DETACH DELETE e
            RETURN count(e) as deleted
        """

        async with self.driver.session(database=self.config.database) as session:
            result = await session.run(query, user_id=user_id)
            record = await result.single()
            count = record["deleted"] if record else 0

        logger.warning("user_graph_cleared", user_id=user_id, entities_deleted=count)
        return count

    def _record_to_entity(self, node: dict[str, Any]) -> Entity:
        """Convert Neo4j node to Entity."""
        return Entity(
            id=node.get("id"),
            name=node.get("name", ""),
            entity_type=node.get("entity_type", ""),
            user_id=node.get("user_id", ""),
            description=node.get("description"),
            confidence=node.get("confidence", 1.0),
            created_at=datetime.fromisoformat(str(node.get("created_at", datetime.utcnow().isoformat()))),
            updated_at=datetime.fromisoformat(str(node.get("updated_at", datetime.utcnow().isoformat()))),
            mention_count=node.get("mention_count", 1),
            source_memories=node.get("source_memories", []),
            attributes=json.loads(node.get("attributes", "{}")) if isinstance(node.get("attributes"), str) else node.get("attributes", {}),
        )

    def _record_to_relationship(
        self, rel: dict[str, Any], source_id: str, target_id: str
    ) -> Relationship:
        """Convert Neo4j relationship to Relationship."""
        return Relationship(
            id=rel.get("id"),
            source_id=source_id,
            target_id=target_id,
            relationship_type=rel.get("relationship_type", ""),
            user_id=rel.get("user_id", ""),
            description=rel.get("description"),
            confidence=rel.get("confidence", 1.0),
            strength=rel.get("strength", 1.0),
            created_at=datetime.fromisoformat(str(rel.get("created_at", datetime.utcnow().isoformat()))),
            updated_at=datetime.fromisoformat(str(rel.get("updated_at", datetime.utcnow().isoformat()))),
            source_memories=rel.get("source_memories", []),
            attributes=json.loads(rel.get("attributes", "{}")) if isinstance(rel.get("attributes"), str) else rel.get("attributes", {}),
        )
