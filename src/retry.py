"""Pattern 2: Self-correcting retry loop on schema violations.

Core insight: When Pydantic raises ``ValidationError`` on the tool input
(e.g. the model emitted an out-of-enum value), we hand the error message
back to Claude as a user turn and let it retry. This is substantially
more reliable than blind ``try/except`` loops because Claude understands
*why* the previous output was wrong and corrects the specific field.

Usage::

    from src.retry import extract_with_retry

    result = extract_with_retry(client, text, Person, max_retries=3)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from .basic import DEFAULT_MODEL, ToolUseNotFoundError, _build_tool, _extract_tool_input

if TYPE_CHECKING:
    from anthropic import Anthropic

T = TypeVar("T", bound=BaseModel)


class RetriesExceededError(RuntimeError):
    """Raised when ``max_retries`` attempts have all failed validation.

    The :attr:`last_error` attribute exposes the final ``ValidationError``
    so callers can log it or surface to users.
    """

    def __init__(self, message: str, last_error: ValidationError) -> None:
        super().__init__(message)
        self.last_error = last_error


def _format_validation_errors(err: ValidationError) -> str:
    """Compact rendering of Pydantic errors suitable to send back to Claude."""
    lines = []
    for detail in err.errors():
        loc = ".".join(str(p) for p in detail["loc"])
        msg = detail["msg"]
        input_value = detail.get("input")
        lines.append(f"- {loc}: {msg} (got: {input_value!r})")
    return "\n".join(lines)


def extract_with_retry(
    client: Anthropic,
    text: str,
    model_cls: type[T],
    *,
    tool_name: str = "extract",
    tool_description: str | None = None,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
    max_retries: int = 3,
) -> T:
    """Extract with automatic correction when the model violates the schema.

    On each ``ValidationError`` the conversation grows by one assistant
    turn (the broken tool_use) and one user turn (a terse correction
    request). This gives Claude the full context it needs to emit a
    compliant retry.

    Args:
        client: ``anthropic.Anthropic`` client.
        text: Input to extract from.
        model_cls: Target Pydantic model.
        tool_name: Name of the forced tool.
        tool_description: Optional tool description.
        system: Optional system prompt.
        model: Claude model identifier.
        max_tokens: Upper bound per turn.
        max_retries: Number of correction attempts (0 = first call only).

    Returns:
        A validated ``model_cls`` instance.

    Raises:
        RetriesExceededError: If every attempt fails validation.
        ToolUseNotFoundError: If a response contains no tool_use block.
    """
    description = tool_description or f"Extract structured data matching {model_cls.__name__}"
    tool = _build_tool(tool_name, description, model_cls)

    messages: list[dict[str, Any]] = [{"role": "user", "content": text}]
    last_error: ValidationError | None = None

    for attempt in range(max_retries + 1):
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": tool_name},
            "messages": messages,
        }
        if system is not None:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)

        try:
            tool_input = _extract_tool_input(response, tool_name)
        except ToolUseNotFoundError:
            # Not a recoverable error — re-raise so caller sees setup issue.
            raise

        try:
            return model_cls.model_validate(tool_input)
        except ValidationError as e:
            last_error = e
            if attempt >= max_retries:
                break
            # Append the broken tool_use and ask Claude to fix it.
            # We reconstruct the assistant turn from response.content so
            # Claude sees exactly what it produced last turn.
            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous tool call failed schema validation:\n"
                        f"{_format_validation_errors(e)}\n\n"
                        "Please call the tool again with a corrected payload "
                        "that satisfies all constraints."
                    ),
                }
            )

    assert last_error is not None  # the loop only exits via success or this path
    raise RetriesExceededError(
        f"Failed to produce a valid {model_cls.__name__} after {max_retries + 1} attempts.",
        last_error=last_error,
    )
