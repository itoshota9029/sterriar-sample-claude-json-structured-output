"""Pattern 4: Stream a partially-populated model to the UI as Claude writes.

Core insight: When Claude streams a tool_use, it emits
``input_json_delta`` events that accumulate into the final JSON. By
incrementally concatenating the deltas and trying to parse after each
chunk, we can surface useful partial state to the UI well before the
full response arrives — perfect for long extractions where the user
would otherwise stare at a spinner.

The key trick is handling ``json.JSONDecodeError`` gracefully: most
intermediate chunks will be syntactically incomplete. We only yield a
new partial when parsing succeeds (which happens at "natural" boundaries
— closing brackets, quoted strings, etc.).

Usage::

    for partial in stream_extract(client, text, Invoice):
        render_preview(partial)  # Partial Invoice with best-effort fields

Note: this uses ``partial_object_from_json`` which **tolerates missing
required fields** during streaming by calling ``model_cls.model_construct``
rather than ``model_validate``. You get a typed object but should not
treat it as validated until the generator completes.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from .basic import DEFAULT_MODEL, _build_tool

if TYPE_CHECKING:
    from anthropic import Anthropic

T = TypeVar("T", bound=BaseModel)


def _try_parse(accumulated: str) -> dict[str, Any] | None:
    """Attempt to parse ``accumulated`` as JSON, returning None on failure.

    We also try a forgiving variant: close trailing ``}`` / ``]`` / quotes
    when the last valid prefix can be completed cheaply. This avoids
    missing useful partials when Claude is mid-field.
    """
    try:
        parsed = json.loads(accumulated)
        if isinstance(parsed, dict):
            return parsed
        return None
    except json.JSONDecodeError:
        pass

    # Simple salvage attempts: close one unclosed brace or bracket.
    for suffix in ("}", '"}', '"]}', "]}"):
        try:
            parsed = json.loads(accumulated + suffix)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def stream_extract(
    client: Anthropic,
    text: str,
    model_cls: type[T],
    *,
    tool_name: str = "extract",
    tool_description: str | None = None,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
) -> Generator[T, None, None]:
    """Yield progressively populated ``model_cls`` instances as Claude streams.

    The generator yields every time a *new* partial JSON parses cleanly.
    The final yield corresponds to the fully-formed object and is the
    only one you should consider "validated"; intermediate yields use
    ``model_construct`` and may have missing required fields.

    Args:
        client: ``anthropic.Anthropic`` client (streaming-capable).
        text: Input to extract from.
        model_cls: Target Pydantic model.
        tool_name: Name of the forced tool.
        tool_description: Optional tool description.
        system: Optional system prompt.
        model: Claude model identifier.
        max_tokens: Upper bound on output tokens.

    Yields:
        Instances of ``model_cls`` — earlier ones via ``model_construct``,
        final one validated via ``model_validate``.
    """
    description = tool_description or f"Extract structured data matching {model_cls.__name__}"
    tool = _build_tool(tool_name, description, model_cls)

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "tools": [tool],
        "tool_choice": {"type": "tool", "name": tool_name},
        "messages": [{"role": "user", "content": text}],
    }
    if system is not None:
        kwargs["system"] = system

    accumulated = ""
    last_emitted: dict[str, Any] | None = None

    with client.messages.stream(**kwargs) as stream:
        for event in stream:
            etype = getattr(event, "type", None)
            if etype != "content_block_delta":
                continue
            delta = getattr(event, "delta", None)
            if getattr(delta, "type", None) != "input_json_delta":
                continue
            accumulated += getattr(delta, "partial_json", "") or ""

            parsed = _try_parse(accumulated)
            if parsed is not None and parsed != last_emitted:
                last_emitted = parsed
                yield model_cls.model_construct(**parsed)

        # On stream end, the accumulated string should be the final JSON.
        # Yield a *validated* instance as the final element — but only if
        # the content differs from the last partial. When content is
        # identical we validate silently (raising on schema violations)
        # to avoid emitting a duplicate to the caller.
        final = _try_parse(accumulated)
        if final is not None:
            if final != last_emitted:
                yield model_cls.model_validate(final)
            else:
                model_cls.model_validate(final)
