"""Unit tests for Embedder utilities."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cortex.config import EmbeddingConfig
from cortex.utils.embedder import (
    Embedder,
    LocalEmbedder,
    OpenAIEmbedder,
    create_embedder,
)


class TestOpenAIEmbedder:
    """Test OpenAI embedding generation."""

    @pytest.fixture
    def config(self) -> EmbeddingConfig:
        """Create test config."""
        return EmbeddingConfig(
            provider="openai",
            api_key="test-key",
            model="text-embedding-3-small",
            dimensions=1536,
        )

    async def test_embed_returns_vector(self, config: EmbeddingConfig):
        """Test that embed returns a vector of floats."""
        mock_embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=mock_embedding)]

        with patch("cortex.utils.embedder.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            embedder = OpenAIEmbedder(config)
            embedder.client = mock_client

            result = await embedder.embed("test text")
            assert len(result) == 1536
            assert all(isinstance(x, float) for x in result)

    async def test_embed_batch_returns_multiple_vectors(self, config: EmbeddingConfig):
        """Test that embed_batch returns multiple vectors."""
        mock_embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=mock_embedding),
            MagicMock(embedding=mock_embedding),
        ]

        with patch("cortex.utils.embedder.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            embedder = OpenAIEmbedder(config)
            embedder.client = mock_client

            result = await embedder.embed_batch(["text1", "text2"])
            assert len(result) == 2
            assert all(len(v) == 1536 for v in result)

    async def test_embed_batch_empty_list(self, config: EmbeddingConfig):
        """Test that embed_batch handles empty list."""
        with patch("cortex.utils.embedder.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client

            embedder = OpenAIEmbedder(config)
            result = await embedder.embed_batch([])
            assert result == []

    async def test_health_check_success(self, config: EmbeddingConfig):
        """Test health check when API is working."""
        mock_embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=mock_embedding)]

        with patch("cortex.utils.embedder.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            embedder = OpenAIEmbedder(config)
            embedder.client = mock_client

            result = await embedder.health_check()
            assert result is True

    async def test_health_check_failure(self, config: EmbeddingConfig):
        """Test health check when API fails."""
        with patch("cortex.utils.embedder.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(side_effect=Exception("API error"))
            mock_openai.return_value = mock_client

            embedder = OpenAIEmbedder(config)
            embedder.client = mock_client

            result = await embedder.health_check()
            assert result is False


class TestLocalEmbedder:
    """Test local embedding generation."""

    @pytest.fixture
    def config(self) -> EmbeddingConfig:
        """Create test config for local embedder."""
        return EmbeddingConfig(
            provider="local",
            local_model="all-MiniLM-L6-v2",
            dimensions=384,
        )

    async def test_embed_returns_vector(self, config: EmbeddingConfig):
        """Test that embed returns a vector."""
        mock_embedding = [0.1] * 384

        with patch("cortex.utils.embedder.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = MagicMock(tolist=lambda: mock_embedding)
            mock_st.return_value = mock_model

            embedder = LocalEmbedder(config)
            result = await embedder.embed("test text")

            assert len(result) == 384
            mock_model.encode.assert_called_once()

    async def test_embed_batch_returns_multiple_vectors(self, config: EmbeddingConfig):
        """Test that embed_batch returns multiple vectors."""
        mock_embeddings = [[0.1] * 384, [0.2] * 384]

        with patch("cortex.utils.embedder.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = MagicMock(tolist=lambda: mock_embeddings)
            mock_st.return_value = mock_model

            embedder = LocalEmbedder(config)
            result = await embedder.embed_batch(["text1", "text2"])

            assert len(result) == 2

    async def test_embed_batch_empty_list(self, config: EmbeddingConfig):
        """Test that embed_batch handles empty list."""
        with patch("cortex.utils.embedder.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            embedder = LocalEmbedder(config)
            result = await embedder.embed_batch([])

            assert result == []
            mock_model.encode.assert_not_called()


class TestCreateEmbedder:
    """Test embedder factory function."""

    def test_create_openai_embedder(self):
        """Test creating OpenAI embedder."""
        config = EmbeddingConfig(provider="openai", api_key="test")

        with patch("cortex.utils.embedder.AsyncOpenAI"):
            embedder = create_embedder(config)
            assert isinstance(embedder, OpenAIEmbedder)

    def test_create_local_embedder(self):
        """Test creating local embedder."""
        config = EmbeddingConfig(provider="local")

        with patch("cortex.utils.embedder.SentenceTransformer"):
            embedder = create_embedder(config)
            assert isinstance(embedder, LocalEmbedder)

    def test_default_is_openai(self):
        """Test that default provider is OpenAI."""
        config = EmbeddingConfig(api_key="test")

        with patch("cortex.utils.embedder.AsyncOpenAI"):
            embedder = create_embedder(config)
            assert isinstance(embedder, OpenAIEmbedder)
