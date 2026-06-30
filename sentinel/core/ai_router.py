"""
AI Router — tiered model selection for SENTINEL.

Routing strategy (cheapest first):
  simple   -> Ollama Gemma 4 (free, local)
  standard -> xAI Grok-3-mini (cheap, fast)
  complex  -> xAI Grok-3 or GPT-5 (mid-cost)
  trading  -> Claude Opus 4.7 (premium, complex trading strategies only)

Usage:
    from sentinel.core.ai_router import get_ai_client, AITier

    client = get_ai_client(AITier.STANDARD)
    response = await client.chat("Summarize AAPL earnings")

    # Complex trading strategy — uses Claude Opus 4.7
    client = get_ai_client(AITier.TRADING)
    response = await client.chat("Build a pairs trading strategy for XLE/XOM...")
"""
from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from typing import Any

import httpx

from sentinel.core.config import get_settings
from sentinel.core.logging import get_logger

logger = get_logger(__name__)


class AITier(str, enum.Enum):
    SIMPLE   = "simple"    # Ollama Gemma — free, local, routine tasks
    STANDARD = "standard"  # xAI Grok-3-mini — cheap, most analysis
    COMPLEX  = "complex"   # xAI Grok-3 / GPT-5 — multi-step reasoning
    TRADING  = "trading"   # Claude Opus 4.7 — complex trading strategies only


@dataclass
class AIResponse:
    content: str
    model: str
    tier: AITier
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0


class OllamaClient:
    """Free local inference via Ollama."""

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def chat(self, prompt: str, system: str = "") -> AIResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False},
            )
            r.raise_for_status()
            data = r.json()

        content = data.get("message", {}).get("content", "")
        return AIResponse(
            content=content,
            model=self.model,
            tier=AITier.SIMPLE,
            provider="ollama",
        )


class XAIClient:
    """xAI Grok — cheap primary AI for standard and complex tasks."""

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def chat(self, prompt: str, system: str = "") -> AIResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={"model": self.model, "messages": messages},
            )
            r.raise_for_status()
            data = r.json()

        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return AIResponse(
            content=choice,
            model=self.model,
            tier=AITier.STANDARD,
            provider="xai",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )


class OpenAIClient:
    """OpenAI GPT-5 — complex multi-step workflows."""

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def chat(self, prompt: str, system: str = "") -> AIResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=180.0) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={"model": self.model, "messages": messages},
            )
            r.raise_for_status()
            data = r.json()

        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return AIResponse(
            content=choice,
            model=self.model,
            tier=AITier.COMPLEX,
            provider="openai",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )


class AnthropicClient:
    """Claude Opus 4.7 — premium, complex trading strategies only."""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    async def chat(self, prompt: str, system: str = "") -> AIResponse:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system

        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=body,
            )
            r.raise_for_status()
            data = r.json()

        content = data["content"][0]["text"]
        usage = data.get("usage", {})
        return AIResponse(
            content=content,
            model=self.model,
            tier=AITier.TRADING,
            provider="anthropic",
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )


def get_ai_client(tier: AITier | str = AITier.STANDARD) -> OllamaClient | XAIClient | OpenAIClient | AnthropicClient:
    """
    Return the appropriate AI client for the requested tier.

    Falls back gracefully:
      SIMPLE   -> Ollama, or xAI if Ollama unavailable
      STANDARD -> xAI Grok-3-mini, or Ollama if no xAI key
      COMPLEX  -> xAI Grok-3, or GPT-5, or Claude Haiku
      TRADING  -> Claude Opus 4.7 (no fallback — this is the premium tier)
    """
    s = get_settings()
    tier = AITier(tier)

    if tier == AITier.SIMPLE:
        if s.has_ollama:
            logger.debug("ai_router: SIMPLE -> Ollama", model=s.ollama_fast_model)
            return OllamaClient(s.ollama_fast_model, s.ollama_base_url)
        if s.has_xai:
            logger.debug("ai_router: SIMPLE fallback -> xAI grok-3-mini")
            return XAIClient(s.xai_model, s.xai_api_key, s.xai_base_url)

    if tier == AITier.STANDARD:
        if s.has_xai:
            logger.debug("ai_router: STANDARD -> xAI", model=s.xai_model)
            return XAIClient(s.xai_model, s.xai_api_key, s.xai_base_url)
        if s.has_ollama:
            logger.debug("ai_router: STANDARD fallback -> Ollama", model=s.ollama_default_model)
            return OllamaClient(s.ollama_default_model, s.ollama_base_url)

    if tier == AITier.COMPLEX:
        if s.has_xai:
            logger.debug("ai_router: COMPLEX -> xAI", model=s.xai_complex_model)
            return XAIClient(s.xai_complex_model, s.xai_api_key, s.xai_base_url)
        if s.has_openai:
            logger.debug("ai_router: COMPLEX -> GPT-5")
            return OpenAIClient(s.openai_model, s.openai_api_key, s.openai_base_url)
        if s.has_anthropic:
            logger.debug("ai_router: COMPLEX fallback -> Claude Haiku")
            return AnthropicClient(s.anthropic_fast_model, s.anthropic_api_key)

    if tier == AITier.TRADING:
        if s.has_anthropic:
            logger.debug("ai_router: TRADING -> Claude Opus 4.7")
            return AnthropicClient(s.anthropic_model, s.anthropic_api_key)
        raise RuntimeError(
            "TRADING tier requires ANTHROPIC_API_KEY. "
            "Set it in .env to enable complex trading strategy generation."
        )

    raise RuntimeError(
        f"No AI provider available for tier={tier}. "
        "Set at least one of: OLLAMA running locally, XAI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY"
    )


def route_by_task(task: str) -> AITier:
    """
    Suggest the appropriate tier based on task description keywords.

    Examples:
        route_by_task("summarize earnings call")      -> SIMPLE
        route_by_task("screen for momentum stocks")   -> STANDARD
        route_by_task("build pairs trading strategy") -> TRADING
    """
    task_lower = task.lower()

    trading_keywords = {
        "trading strategy", "pairs trade", "stat arb", "statistical arbitrage",
        "alpha signal", "portfolio construction", "factor model", "backtest strategy",
        "options strategy", "volatility strategy", "market making", "execution algo",
        "risk model", "regime strategy", "systematic strategy",
    }
    complex_keywords = {
        "multi-step", "compare", "analyze portfolio", "stress test",
        "attribution", "scenario", "cross-asset", "macro regime",
    }
    simple_keywords = {
        "summarize", "describe", "explain", "what is", "define",
        "list", "format", "extract", "parse",
    }

    for kw in trading_keywords:
        if kw in task_lower:
            return AITier.TRADING

    for kw in complex_keywords:
        if kw in task_lower:
            return AITier.COMPLEX

    for kw in simple_keywords:
        if kw in task_lower:
            return AITier.SIMPLE

    return AITier.STANDARD
