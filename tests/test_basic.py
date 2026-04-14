"""Tests for src.basic — Tool Use forced extraction."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from src.basic import ToolUseNotFoundError, extract
from tests.conftest import make_client_with_responses, make_response, make_tool_use_block


class Person(BaseModel):
    name: str
    age: int
    role: str


def test_extract_happy_path_parses_tool_use_input():
    response = make_response(
        blocks=[
            make_tool_use_block(
                "extract_person",
                {"name": "山田太郎", "age": 30, "role": "engineer"},
            )
        ]
    )
    client = make_client_with_responses(response)

    result = extract(client, "山田太郎(30歳)はエンジニア", Person, tool_name="extract_person")

    assert result == Person(name="山田太郎", age=30, role="engineer")


def test_extract_passes_forced_tool_choice_to_api():
    response = make_response(
        blocks=[make_tool_use_block("extract_person", {"name": "A", "age": 1, "role": "x"})]
    )
    client = make_client_with_responses(response)

    extract(client, "foo", Person, tool_name="extract_person")

    # Inspect the kwargs the helper passed to the SDK.
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["tool_choice"] == {"type": "tool", "name": "extract_person"}
    assert kwargs["tools"][0]["name"] == "extract_person"
    assert "input_schema" in kwargs["tools"][0]


def test_extract_raises_when_response_has_no_tool_use_block():
    response = make_response(blocks=[], stop_reason="end_turn")
    client = make_client_with_responses(response)

    with pytest.raises(ToolUseNotFoundError):
        extract(client, "foo", Person, tool_name="extract_person")


def test_extract_ignores_unrelated_blocks_before_tool_use():
    text_block = type("TextBlock", (), {"type": "text", "text": "thinking..."})()
    response = make_response(
        blocks=[
            text_block,
            make_tool_use_block("extract_person", {"name": "B", "age": 2, "role": "y"}),
        ]
    )
    client = make_client_with_responses(response)

    result = extract(client, "foo", Person, tool_name="extract_person")

    assert result == Person(name="B", age=2, role="y")


def test_extract_system_prompt_is_forwarded_when_provided():
    response = make_response(
        blocks=[make_tool_use_block("extract_person", {"name": "C", "age": 3, "role": "z"})]
    )
    client = make_client_with_responses(response)

    extract(
        client,
        "foo",
        Person,
        tool_name="extract_person",
        system="You are an entity extractor.",
    )

    assert client.messages.create.call_args.kwargs["system"] == "You are an entity extractor."
