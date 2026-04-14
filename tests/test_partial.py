"""Tests for src.partial — truncation recovery."""

from __future__ import annotations

from pydantic import BaseModel

from src.partial import extract_with_continuation, salvage_partial
from tests.conftest import make_client_with_responses, make_response, make_tool_use_block


class Invoice(BaseModel):
    invoice_number: str
    total: float
    note: str


def test_salvage_partial_returns_partial_and_flag_on_max_tokens():
    response = make_response(
        blocks=[make_tool_use_block("extract", {"invoice_number": "INV-001", "total": 10000.0})],
        stop_reason="max_tokens",
    )

    partial, was_truncated = salvage_partial(response, tool_name="extract")

    assert partial == {"invoice_number": "INV-001", "total": 10000.0}
    assert was_truncated is True


def test_salvage_partial_returns_empty_when_no_tool_use_block():
    response = make_response(blocks=[], stop_reason="max_tokens")

    partial, was_truncated = salvage_partial(response, tool_name="extract")

    assert partial == {}
    assert was_truncated is True


def test_extract_with_continuation_uses_single_call_when_not_truncated():
    full = make_response(
        blocks=[
            make_tool_use_block(
                "extract",
                {"invoice_number": "A", "total": 1.0, "note": "done"},
            )
        ],
        stop_reason="tool_use",
    )
    client = make_client_with_responses(full)

    result = extract_with_continuation(client, "foo", Invoice)

    assert result == Invoice(invoice_number="A", total=1.0, note="done")
    assert client.messages.create.call_count == 1


def test_extract_with_continuation_reruns_with_partial_feedback_on_truncation():
    truncated = make_response(
        blocks=[make_tool_use_block("extract", {"invoice_number": "B", "total": 2.0})],
        stop_reason="max_tokens",
    )
    full = make_response(
        blocks=[
            make_tool_use_block(
                "extract",
                {"invoice_number": "B", "total": 2.0, "note": "resumed"},
            )
        ],
        stop_reason="tool_use",
    )
    client = make_client_with_responses(truncated, full)

    result = extract_with_continuation(client, "foo", Invoice)

    assert result == Invoice(invoice_number="B", total=2.0, note="resumed")
    assert client.messages.create.call_count == 2

    # The second call must carry the partial as context.
    second_messages = client.messages.create.call_args_list[1].kwargs["messages"]
    assert any("invoice_number" in str(m.get("content", "")) for m in second_messages)
