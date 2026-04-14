"""Pattern 3: Recover from ``max_tokens`` truncation without losing progress.

Core insight: When ``stop_reason == "max_tokens"``, Claude stopped mid-JSON.
The SDK still gives us the partial tool_use block with the keys it had
committed to. By asking Claude to continue from that partial state we
avoid re-generating (which is expensive) and keep output stable.

This helper exposes two tactics:

1. :func:`extract_with_continuation` — reruns with a larger budget,
   seeding the assistant turn with the partial tool_use so Claude
   only emits the missing fields.
2. :func:`salvage_partial` — best-effort: returns whatever fields
   were present in the truncated payload, with missing fields
   unset. Useful when the caller has sensible defaults.

Usage::

    result = extract_with_continuation(client, text, Invoice, max_tokens=4096)
    # or:
    partial_dict, was_truncated = salvage_partial(response, tool_name="extract")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from .basic import DEFAULT_MODEL, _build_tool, _extract_tool_input

if TYPE_CHECKING:
    from anthropic import Anthropic

T = TypeVar("T", bound=BaseModel)


def salvage_partial(response: Any, tool_name: str) -> tuple[dict[str, Any], bool]:
    """Best-effort extraction of the tool_use input.

    Returns a tuple ``(partial_input, was_truncated)``. ``was_truncated``
    is ``True`` when ``stop_reason == "max_tokens"``. When no tool_use
    block is present at all, ``partial_input`` is an empty dict — check
    ``was_truncated`` to distinguish "Claude never started" from "JSON cut
    mid-key".

    Args:
        response: ``anthropic.types.Message`` returned by ``messages.create``.
        tool_name: Name of the tool to extract.

    Returns:
        ``(partial_input, was_truncated)``.
    """
    was_truncated = getattr(response, "stop_reason", None) == "max_tokens"
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
            input_obj = getattr(block, "input", None) or {}
            return dict(input_obj), was_truncated
    return {}, was_truncated


def extract_with_continuation(
    client: Anthropic,
    text: str,
    model_cls: type[T],
    *,
    tool_name: str = "extract",
    tool_description: str | None = None,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
    continuation_budget: int = 4096,
) -> T:
    """Extract and transparently recover if the first call truncates.

    Flow:
        1. Try a normal ``messages.create`` call with ``max_tokens``.
        2. If ``stop_reason == "max_tokens"``, start a new turn feeding the
           partial tool_use back to Claude and asking it to finish.
        3. Re-parse the continuation output.

    Args:
        client: ``anthropic.Anthropic`` client.
        text: Input to extract from.
        model_cls: Target Pydantic model.
        tool_name: Name of the forced tool.
        tool_description: Optional tool description.
        system: Optional system prompt.
        model: Claude model identifier.
        max_tokens: First-attempt budget (keep tight for cost).
        continuation_budget: Second-attempt budget when continuing.

    Returns:
        A validated ``model_cls`` instance.

    Raises:
        ToolUseNotFoundError: If no tool_use block in either response.
        pydantic.ValidationError: If the final payload still doesn't validate.
    """
    description = tool_description or f"Extract structured data matching {model_cls.__name__}"
    tool = _build_tool(tool_name, description, model_cls)

    def _call(messages: list[dict[str, Any]], budget: int) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": budget,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": tool_name},
            "messages": messages,
        }
        if system is not None:
            kwargs["system"] = system
        return client.messages.create(**kwargs)

    messages: list[dict[str, Any]] = [{"role": "user", "content": text}]
    response = _call(messages, max_tokens)

    if getattr(response, "stop_reason", None) != "max_tokens":
        tool_input = _extract_tool_input(response, tool_name)
        return model_cls.model_validate(tool_input)

    # Truncated: feed the partial tool_use back and ask for completion.
    partial, _ = salvage_partial(response, tool_name)
    messages.append({"role": "assistant", "content": response.content})
    messages.append(
        {
            "role": "user",
            "content": (
                "The previous tool call was cut off before finishing. "
                "Please call the tool again with the complete object. "
                f"Fields already collected (re-emit them verbatim): {partial}"
            ),
        }
    )
    response = _call(messages, continuation_budget)
    tool_input = _extract_tool_input(response, tool_name)
    return model_cls.model_validate(tool_input)
