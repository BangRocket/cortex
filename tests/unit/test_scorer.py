"""Unit tests for EmotionScorer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cortex.config import LLMConfig
from cortex.utils.scorer import EmotionScorer


class TestEmotionScorer:
    """Test EmotionScorer functionality."""

    @pytest.fixture
    def config(self) -> LLMConfig:
        """Create test config."""
        return LLMConfig(
            provider="openai",
            api_key="test-key",
            model="gpt-4o-mini",
        )

    @pytest.fixture
    def scorer(self, config: LLMConfig) -> EmotionScorer:
        """Create scorer instance."""
        return EmotionScorer(config)

    async def test_score_parses_float_response(self, scorer: EmotionScorer):
        """Test that score correctly parses float from LLM response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="0.75"))]

        with patch.object(scorer, "_client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            scorer._client = mock_client

            score = await scorer.score("I'm so happy about this!")
            assert score == 0.75

    async def test_score_handles_text_with_number(self, scorer: EmotionScorer):
        """Test parsing when response includes text around the number."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Score: 0.85"))]

        with patch.object(scorer, "_client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            scorer._client = mock_client

            score = await scorer.score("Major life event")
            assert score == 0.85

    async def test_score_clamps_to_valid_range(self, scorer: EmotionScorer):
        """Test that scores are clamped to 0.0-1.0 range."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="1.5"))]

        with patch.object(scorer, "_client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            scorer._client = mock_client

            score = await scorer.score("Something extreme")
            assert score == 1.0

    async def test_score_returns_default_on_error(self, scorer: EmotionScorer):
        """Test that score returns 0.5 on API error."""
        with patch.object(scorer, "_client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))
            scorer._client = mock_client

            score = await scorer.score("Any content")
            assert score == 0.5

    async def test_score_returns_default_on_unparseable(self, scorer: EmotionScorer):
        """Test that score returns 0.5 when response can't be parsed."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="I don't know"))]

        with patch.object(scorer, "_client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            scorer._client = mock_client

            score = await scorer.score("Confusing content")
            assert score == 0.5

    async def test_classify_parses_json_response(self, scorer: EmotionScorer):
        """Test that classify correctly parses JSON response."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"type": "identity", "confidence": 0.95, "identity_key": "name"}'
                )
            )
        ]

        with patch.object(scorer, "_client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            scorer._client = mock_client

            result = await scorer.classify("User's name is Josh")
            assert result["type"] == "identity"
            assert result["confidence"] == 0.95
            assert result["identity_key"] == "name"

    async def test_classify_handles_markdown_code_block(self, scorer: EmotionScorer):
        """Test that classify extracts JSON from markdown code block."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='```json\n{"type": "episodic", "confidence": 0.8, "identity_key": null}\n```'
                )
            )
        ]

        with patch.object(scorer, "_client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            scorer._client = mock_client

            result = await scorer.classify("Had a meeting today")
            assert result["type"] == "episodic"
            assert result["confidence"] == 0.8

    async def test_classify_returns_default_on_error(self, scorer: EmotionScorer):
        """Test that classify returns default on API error."""
        with patch.object(scorer, "_client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))
            scorer._client = mock_client

            result = await scorer.classify("Any content")
            assert result["type"] == "episodic"
            assert result["confidence"] == 0.5
            assert result["identity_key"] is None

    async def test_score_batch(self, scorer: EmotionScorer):
        """Test batch scoring."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="0.5"))]

        with patch.object(scorer, "_client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            scorer._client = mock_client

            scores = await scorer.score_batch(["content1", "content2", "content3"])
            assert len(scores) == 3
            assert all(s == 0.5 for s in scores)


class TestAnthropicProvider:
    """Test Anthropic provider support."""

    @pytest.fixture
    def anthropic_config(self) -> LLMConfig:
        """Create Anthropic test config."""
        return LLMConfig(
            provider="anthropic",
            api_key="test-anthropic-key",
            model="claude-3-haiku-20240307",
        )

    @pytest.fixture
    def anthropic_scorer(self, anthropic_config: LLMConfig) -> EmotionScorer:
        """Create scorer with Anthropic config."""
        return EmotionScorer(anthropic_config)

    async def test_anthropic_score(self, anthropic_scorer: EmotionScorer):
        """Test scoring with Anthropic provider."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="0.65")]

        with patch.object(anthropic_scorer, "_client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            anthropic_scorer._client = mock_client

            score = await anthropic_scorer.score("Moderately emotional content")
            assert score == 0.65

    async def test_anthropic_classify(self, anthropic_scorer: EmotionScorer):
        """Test classification with Anthropic provider."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"type": "semantic", "confidence": 0.9, "identity_key": null}')
        ]

        with patch.object(anthropic_scorer, "_client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            anthropic_scorer._client = mock_client

            result = await anthropic_scorer.classify("Python is a programming language")
            assert result["type"] == "semantic"
            assert result["confidence"] == 0.9
