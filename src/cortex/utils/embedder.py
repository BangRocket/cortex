"""Embedding generation utilities."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import structlog

from cortex.config import EmbeddingConfig

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class Embedder(ABC):
    """Abstract base class for embedding generation."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...


class OpenAIEmbedder(Embedder):
    """OpenAI embedding generator."""

    def __init__(self, config: EmbeddingConfig) -> None:
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=config.api_key)
        self.model = config.model
        self.dimensions = config.dimensions

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        # OpenAI supports batch embedding
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
        )
        return [item.embedding for item in response.data]

    async def health_check(self) -> bool:
        """Check if the embedder is working."""
        try:
            await self.embed("test")
            return True
        except Exception as e:
            logger.error("embedder_health_check_failed", error=str(e))
            return False


class LocalEmbedder(Embedder):
    """Local embedding generator using sentence-transformers."""

    def __init__(self, config: EmbeddingConfig) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(config.local_model)

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: self.model.encode(text).tolist()
        )
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self.model.encode(texts).tolist()
        )
        return embeddings

    async def health_check(self) -> bool:
        """Check if the embedder is working."""
        try:
            await self.embed("test")
            return True
        except Exception as e:
            logger.error("embedder_health_check_failed", error=str(e))
            return False


def create_embedder(config: EmbeddingConfig) -> Embedder:
    """Factory function to create the appropriate embedder."""
    if config.provider == "local":
        return LocalEmbedder(config)
    return OpenAIEmbedder(config)
