"""Tests for src.streaming — progressive parsing while Claude streams."""

from __future__ import annotations

from unittest.mock import MagicMock

from pydantic import BaseModel

from src.streaming import _try_parse, stream_extract
from tests.conftest import make_delta_event, make_stream_context


class Product(BaseModel):
    sku: str
    name: str
    price: float


def test_try_parse_returns_dict_for_complete_json():
    assert _try_parse('{"sku": "A", "name": "x", "price": 1.0}') == {
        "sku": "A",
        "name": "x",
        "price": 1.0,
    }


def test_try_parse_returns_none_for_incomplete_json():
    # Mid-key truncation can't be salvaged.
    assert _try_parse('{"sku": "A", "name":') is None


def test_try_parse_salvages_trailing_close_brace():
    # Missing closing brace — the salvage loop should recover.
    assert _try_parse('{"sku": "A", "name": "x", "price": 1.0') == {
        "sku": "A",
        "name": "x",
        "price": 1.0,
    }


def test_stream_extract_yields_progressive_partials_then_final_validated():
    events = [
        make_delta_event('{"sku":'),
        make_delta_event(' "A",'),
        make_delta_event(' "name": "bolt",'),
        make_delta_event(' "price": 99.5}'),
    ]
    client = MagicMock()
    client.messages.stream.return_value = make_stream_context(events)

    results = list(stream_extract(client, "a bolt, SKU A, 99.5 yen", Product))

    # Expect at least the final fully-populated instance.
    assert any(r.sku == "A" and r.name == "bolt" and r.price == 99.5 for r in results)
    # Final entry is the validated instance.
    final = results[-1]
    assert isinstance(final, Product)
    assert final.sku == "A"


def test_stream_extract_does_not_emit_duplicate_partials():
    events = [
        make_delta_event('{"sku": "A",'),
        # A delta that doesn't change the parseable state shouldn't double-emit.
        make_delta_event(" "),
        make_delta_event('"name": "bolt",'),
        make_delta_event(' "price": 1.0}'),
    ]
    client = MagicMock()
    client.messages.stream.return_value = make_stream_context(events)

    results = list(stream_extract(client, "x", Product))

    # With 4 deltas, we expect at most 4 unique parse states — not more.
    # (Duplicates from non-progressing deltas must be suppressed.)
    unique_payloads = {tuple(sorted(r.model_dump(exclude_none=True).items())) for r in results}
    assert len(unique_payloads) == len(results)
