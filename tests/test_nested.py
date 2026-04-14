"""Tests for src.nested — complex schema stability."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.basic import extract
from src.nested import NestedExampleOutput
from tests.conftest import make_client_with_responses, make_response, make_tool_use_block


def test_nested_example_rejects_literal_outside_enum():
    with pytest.raises(ValidationError):
        NestedExampleOutput.model_validate(
            {
                "title": "x",
                "priority": "critical",  # not in Priority Literal
                "status": "draft",
            }
        )


def test_nested_example_accepts_discriminated_union_of_attachments():
    obj = NestedExampleOutput.model_validate(
        {
            "title": "Review launch checklist",
            "priority": "high",
            "status": "in_review",
            "tags": [{"name": "launch", "weight": 0.9}],
            "attachments": [
                {"kind": "text", "content": "pre-flight notes"},
                {"kind": "url", "url": "https://example.com/plan", "title": "Plan"},
            ],
        }
    )

    assert obj.priority == "high"
    assert len(obj.attachments) == 2
    assert obj.attachments[0].kind == "text"
    assert obj.attachments[1].kind == "url"


def test_nested_example_allows_omitted_optional_fields():
    obj = NestedExampleOutput.model_validate(
        {
            "title": "no assignee",
            "priority": "low",
            "status": "draft",
        }
    )

    assert obj.assignee is None
    assert obj.due_date is None
    assert obj.tags == []
    assert obj.attachments == []


def test_nested_example_end_to_end_via_extract():
    payload = {
        "title": "Ship v0.1.0",
        "priority": "high",
        "status": "approved",
        "tags": [{"name": "release", "weight": 1.0}],
        "assignee": "Alice",
        "due_date": "2026-05-01",
        "attachments": [
            {"kind": "url", "url": "https://example.com", "title": "Runbook"},
        ],
    }
    response = make_response(blocks=[make_tool_use_block("extract", payload)])
    client = make_client_with_responses(response)

    result = extract(client, "release plan", NestedExampleOutput, tool_name="extract")

    assert result.title == "Ship v0.1.0"
    assert result.priority == "high"
    assert len(result.attachments) == 1
