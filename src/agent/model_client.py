"""Anthropic Messages API client for XiaoNZ Agent.

Uses the official `anthropic` Python SDK against any Anthropic-compatible
Messages API endpoint (official `api.anthropic.com` or a self-hosted
gateway/proxy).

- Auth: Bearer token via `auth_token` (works for both x-api-key and
  Authorization: Bearer schemes; the SDK picks the right header).
- `cache_control` is kept in the prompt builder for forward compatibility;
  some upstreams (e.g. self-hosted proxies that strip the field) won't
  benefit, but it's harmless.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from anthropic import AsyncAnthropic
from anthropic.types import Message

from ..config import Settings

logger = logging.getLogger(__name__)


class ModelClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = AsyncAnthropic(
            base_url=settings.model.base_url,
            auth_token=settings.model.auth_token,
            timeout=httpx.Timeout(
                connect=settings.model.connect_timeout,
                read=settings.model.read_timeout,
                write=10.0,
                pool=10.0,
            ),
            max_retries=settings.model.max_retries,
        )

    async def create_message(
        self,
        messages: list[dict],
        system: str | list[dict] | None = None,
        tools: list[dict] | None = None,
    ) -> Message:
        """Low-level single API call. Returns the raw Message object so
        callers can inspect stop_reason and content blocks (needed for
        tool-use loops)."""
        kwargs: dict[str, Any] = {
            "model": self.settings.model.model_id,
            "max_tokens": self.settings.model.max_tokens,
            "messages": messages,
        }
        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        logger.debug(
            "model.request model=%s messages=%d tools=%d",
            self.settings.model.model_id,
            len(messages),
            len(tools) if tools else 0,
        )

        response = await self._client.messages.create(**kwargs)

        usage = getattr(response, "usage", None)
        if usage is not None:
            logger.info(
                "model.usage input=%s output=%s cache_read=%s cache_create=%s stop=%s",
                getattr(usage, "input_tokens", "?"),
                getattr(usage, "output_tokens", "?"),
                getattr(usage, "cache_read_input_tokens", 0),
                getattr(usage, "cache_creation_input_tokens", 0),
                getattr(response, "stop_reason", "?"),
            )

        return response

    async def chat(
        self,
        messages: list[dict],
        system: str | list[dict] | None = None,
    ) -> str:
        """Single-turn text-only completion. No tools.

        Used by the self-test script and any other caller that just
        wants a string back.
        """
        response = await self.create_message(messages=messages, system=system)
        parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts).strip() or "(no response)"
