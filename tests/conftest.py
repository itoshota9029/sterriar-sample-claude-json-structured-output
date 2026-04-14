"""Shared fixtures for building mock Anthropic responses without a live API.

These helpers keep tests readable and reduce the chance of drift if the
SDK changes response shape later — if that happens, adjust here only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock


def make_tool_use_block(
    name: str, tool_input: dict[str, Any], block_id: str = "toolu_test_001"
) -> SimpleNamespace:
    """Construct a fake ``tool_use`` content block."""
    return SimpleNamespace(
        type="tool_use",
        name=name,
        id=block_id,
        input=tool_input,
    )


def make_response(
    *,
    blocks: list[Any] | None = None,
    stop_reason: str = "tool_use",
) -> SimpleNamespace:
    """Construct a fake ``messages.create`` response."""
    return SimpleNamespace(
        content=blocks or [],
        stop_reason=stop_reason,
        role="assistant",
    )


def make_client_with_responses(*responses: Any) -> MagicMock:
    """Return a mock Anthropic client that returns ``responses`` in order.

    Example::

        client = make_client_with_responses(resp1, resp2)
        client.messages.create(...)  # -> resp1
        client.messages.create(...)  # -> resp2
    """
    client = MagicMock()
    client.messages.create.side_effect = list(responses)
    return client


def make_delta_event(partial_json: str) -> SimpleNamespace:
    """Construct a fake streaming ``content_block_delta`` event for tool_use input."""
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="input_json_delta", partial_json=partial_json),
    )


def make_stream_context(events: list[Any]) -> MagicMock:
    """Wrap ``events`` in a context-manager mock that mimics ``client.messages.stream``."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=iter(events))
    cm.__exit__ = MagicMock(return_value=False)
    return cm
