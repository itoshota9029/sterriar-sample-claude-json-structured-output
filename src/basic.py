"""Pattern 1: Force Tool Use to get reliably parsed JSON out of Claude.

Core insight: ``tool_choice={"type": "tool", "name": "<tool_name>"}`` forces
Claude to invoke the tool rather than fall back to prose. This eliminates
~95% of "the model returned free text" failures we see in naive setups.

Usage::

    from pydantic import BaseModel
    from src.basic import extract

    class Person(BaseModel):
        name: str
        age: int
        role: str

    person = extract(client, "山田太郎(30歳)はエンジニアです", Person,
                     tool_name="extract_person")
    # Person(name='山田太郎', age=30, role='engineer')
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from anthropic import Anthropic

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-sonnet-4-5"


class ToolUseNotFoundError(RuntimeError):
    """Raised when the assistant response has no ``tool_use`` block.

    This should not happen with ``tool_choice={"type": "tool", ...}`` but we
    surface it loudly so the caller can bisect setup problems (e.g. wrong
    tool name, outdated SDK, model unavailable) rather than silently
    receiving ``None``.
    """


def _build_tool(name: str, description: str, model_cls: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model into an Anthropic tool schema."""
    return {
        "name": name,
        "description": description,
        "input_schema": model_cls.model_json_schema(),
    }


def _extract_tool_input(response: Any, tool_name: str) -> dict[str, Any]:
    """Find the tool_use block with the expected name and return its input dict."""
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
            return dict(block.input)
    raise ToolUseNotFoundError(
        f"No tool_use block named {tool_name!r} found in response. "
        f"stop_reason={getattr(response, 'stop_reason', None)!r}"
    )


def extract(
    client: Anthropic,
    text: str,
    model_cls: type[T],
    *,
    tool_name: str = "extract",
    tool_description: str | None = None,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
) -> T:
    """Extract a Pydantic-typed object from ``text`` using forced Tool Use.

    Args:
        client: Initialized ``anthropic.Anthropic`` client.
        text: Natural-language input to extract structure from.
        model_cls: Pydantic model class describing the target schema.
        tool_name: Name of the synthetic tool Claude is forced to invoke.
        tool_description: Optional human-readable description for the tool.
            Defaults to ``"Extract structured data matching {ClassName}"``.
        system: Optional system prompt to steer extraction behavior.
        model: Claude model identifier (default: latest Sonnet).
        max_tokens: Upper bound on output tokens.

    Returns:
        An instance of ``model_cls`` populated from the tool_use input.

    Raises:
        ToolUseNotFoundError: If the response contains no matching tool_use block.
        pydantic.ValidationError: If the tool_use input fails Pydantic validation.
            Use :func:`src.retry.extract_with_retry` to auto-correct these.
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

    response = client.messages.create(**kwargs)
    tool_input = _extract_tool_input(response, tool_name)
    return model_cls.model_validate(tool_input)
