"""Tests for src.retry — self-correcting retry loop."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from src.retry import RetriesExceededError, extract_with_retry
from tests.conftest import make_client_with_responses, make_response, make_tool_use_block


class Ticket(BaseModel):
    title: str
    severity: Literal["low", "medium", "high"]


def test_extract_with_retry_succeeds_on_first_attempt():
    good = make_response(
        blocks=[make_tool_use_block("extract", {"title": "down", "severity": "high"})]
    )
    client = make_client_with_responses(good)

    result = extract_with_retry(client, "incident: down", Ticket)

    assert result == Ticket(title="down", severity="high")
    assert client.messages.create.call_count == 1


def test_extract_with_retry_corrects_schema_violation_in_second_call():
    bad = make_response(
        blocks=[make_tool_use_block("extract", {"title": "down", "severity": "critical"})]
    )
    good = make_response(
        blocks=[make_tool_use_block("extract", {"title": "down", "severity": "high"})]
    )
    client = make_client_with_responses(bad, good)

    result = extract_with_retry(client, "incident", Ticket, max_retries=2)

    assert result == Ticket(title="down", severity="high")
    assert client.messages.create.call_count == 2
    # Second call should contain the correction user turn.
    second_messages = client.messages.create.call_args_list[1].kwargs["messages"]
    assert any(
        m["role"] == "user" and "schema validation" in str(m["content"]).lower()
        for m in second_messages
    )


def test_extract_with_retry_raises_retries_exceeded_when_all_attempts_fail():
    bad1 = make_response(
        blocks=[make_tool_use_block("extract", {"title": "x", "severity": "nope"})]
    )
    bad2 = make_response(
        blocks=[make_tool_use_block("extract", {"title": "x", "severity": "also-nope"})]
    )
    client = make_client_with_responses(bad1, bad2)

    with pytest.raises(RetriesExceededError) as excinfo:
        extract_with_retry(client, "x", Ticket, max_retries=1)

    # last_error must surface so the caller can log it.
    assert excinfo.value.last_error is not None
    assert client.messages.create.call_count == 2


def test_extract_with_retry_does_not_retry_when_max_retries_is_zero():
    bad = make_response(blocks=[make_tool_use_block("extract", {"title": "x", "severity": "nope"})])
    client = make_client_with_responses(bad)

    with pytest.raises(RetriesExceededError):
        extract_with_retry(client, "x", Ticket, max_retries=0)

    assert client.messages.create.call_count == 1
