"""Emotional intensity scoring utilities."""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

import structlog

from cortex.config import LLMConfig

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


EMOTION_SCORING_PROMPT = """Rate the emotional intensity of this memory on a scale of 0.0 to 1.0:

- 0.0-0.2: Mundane, routine (grabbed coffee, checked email)
- 0.2-0.4: Mildly notable (had a good meeting, tried a new restaurant)
- 0.4-0.6: Significant (completed a project, had an argument)
- 0.6-0.8: Highly emotional (family emergency, major life event, conflict)
- 0.8-1.0: Profound impact (death, birth, trauma, breakthrough)

Memory: {content}

Return ONLY a float between 0.0 and 1.0."""


CLASSIFICATION_PROMPT = """Classify this memory into one of these types:
- identity: Core fact about the user (name, family, job, preferences)
- semantic: General knowledge or learned fact
- episodic: An event or thing that happened
- project: Related to a specific project

Memory: {content}

Return JSON: {{"type": "<type>", "confidence": <0.0-1.0>, "identity_key": "<key if identity>"}}

Examples:
- "User's name is Josh" -> {{"type": "identity", "confidence": 0.95, "identity_key": "name"}}
- "User has three kids" -> {{"type": "identity", "confidence": 0.9, "identity_key": "family"}}
- "Had a meeting about the API" -> {{"type": "episodic", "confidence": 0.85, "identity_key": null}}"""


class EmotionScorer:
    """Emotional intensity scoring using LLM."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = None

    @property
    def client(self):
        """Lazy load the appropriate LLM client based on provider."""
        if self._client is None:
            if self.config.provider == "anthropic":
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(api_key=self.config.api_key)
            else:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.config.api_key)
        return self._client

    async def _call_llm(self, prompt: str, max_tokens: int = 100) -> str:
        """Call the LLM with the appropriate API."""
        if self.config.provider == "anthropic":
            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        else:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,
            )
            return response.choices[0].message.content.strip()

    async def score(self, content: str) -> float:
        """Score emotional intensity of content."""
        prompt = EMOTION_SCORING_PROMPT.format(content=content)

        try:
            text = await self._call_llm(prompt, max_tokens=5)
            # Extract float from response
            match = re.search(r"(\d+\.?\d*)", text)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            return 0.5
        except Exception as e:
            logger.warning("emotion_scoring_failed", error=str(e))
            return 0.5

    async def score_batch(self, contents: list[str]) -> list[float]:
        """Score multiple memories."""
        return await asyncio.gather(*[self.score(c) for c in contents])

    async def classify(self, content: str) -> dict:
        """Classify a memory into a type."""
        prompt = CLASSIFICATION_PROMPT.format(content=content)

        try:
            text = await self._call_llm(prompt, max_tokens=100)
            # Try to parse JSON from response
            # Handle markdown code blocks
            if "```" in text:
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
                if match:
                    text = match.group(1)
                else:
                    return {"type": "episodic", "confidence": 0.5, "identity_key": None}

            data = json.loads(text)
            return {
                "type": data.get("type", "episodic"),
                "confidence": data.get("confidence", 0.5),
                "identity_key": data.get("identity_key"),
            }
        except Exception as e:
            logger.warning("classification_failed", error=str(e))
            return {"type": "episodic", "confidence": 0.5, "identity_key": None}

    async def classify_batch(self, contents: list[str]) -> list[dict]:
        """Classify multiple memories."""
        return await asyncio.gather(*[self.classify(c) for c in contents])
