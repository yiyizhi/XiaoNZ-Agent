"""Tool registry.

A Tool is just:
    name          — how the model calls it
    description   — what the model sees in the tool spec
    input_schema  — JSON Schema for input validation (sent to Anthropic)
    handler       — async callable taking the parsed input dict and
                    returning a string (the tool_result content)

AgentLoop owns a list of Tools and dispatches them inside its tool-use
loop. Handlers raise on failure — the loop catches and converts to an
error-string tool_result so the model can recover.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

ToolHandler = Callable[[dict], Awaitable[str]]


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler

    def spec(self) -> dict[str, Any]:
        """Render as an Anthropic tool spec dict."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
