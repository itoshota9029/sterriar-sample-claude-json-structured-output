"""Pattern 5: Schema design for complex outputs that survive in production.

This module doesn't expose a new extraction function — it demonstrates
**Pydantic schema patterns that stabilize Claude's tool_use output** on
the types of non-trivial shapes that are common in real apps.

Conventions that matter (from experience running this every day):

1. **Prefer ``Literal[...]`` over ``Enum``**. Claude is far more reliable
   with "must be one of these exact strings" in the JSON schema than
   with Enum-encoded integer values. When you need an enum, subclass
   ``str, Enum`` so the JSON serialization is the string.

2. **Use a discriminator field for Union types**. Bare ``Union[A, B]``
   confuses the model; ``Union[A, B]`` with ``discriminator="kind"``
   produces clean, unambiguous emission.

3. **Mark optional fields ``| None`` with an explicit default**. Without
   a default, Claude sometimes omits the field entirely, leading to
   Pydantic "field required" errors. ``Optional[...] = None`` is safer
   than ``Optional[...]`` alone.

4. **Add ``description`` to every field** via ``Field(description=...)``.
   This surfaces as part of the JSON schema Claude sees and noticeably
   improves extraction quality on ambiguous inputs.

The :class:`NestedExampleOutput` below is a realistic target that
combines all four techniques; pair it with :func:`src.basic.extract` or
:func:`src.retry.extract_with_retry` and it just works.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

# Convention #1: Literal over Enum.
Priority = Literal["low", "normal", "high", "urgent"]
Status = Literal["draft", "in_review", "approved", "archived"]


class Tag(BaseModel):
    """A tag attached to a task, with a weight so we can rank."""

    name: str = Field(description="Short human-readable tag, lowercase-kebab-case")
    weight: float = Field(
        description="Importance from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )


class TextAttachment(BaseModel):
    """A raw text blob attached to the task (notes, comments, etc)."""

    # Convention #2: discriminator field. The literal string is the
    # tag used by Pydantic to route Union types cleanly.
    kind: Literal["text"] = "text"
    content: str = Field(description="Plain text contents, up to 500 chars")


class UrlAttachment(BaseModel):
    """A URL reference attached to the task."""

    kind: Literal["url"] = "url"
    url: str = Field(description="Absolute https:// URL")
    title: str | None = Field(
        default=None,
        description="Optional human-readable title of the linked page",
    )


Attachment = Annotated[
    TextAttachment | UrlAttachment,
    Field(discriminator="kind"),
]


class NestedExampleOutput(BaseModel):
    """Realistic nested output — mirror this structure in your own schemas.

    This exercises Literal/enum handling, Union with discriminator,
    nested models, and optional fields with explicit defaults. Feed the
    resulting JSON schema to Claude via Tool Use and it comes back
    clean on the first try >>90% of the time.
    """

    title: str = Field(description="One-line summary, imperative mood")
    priority: Priority = Field(
        description="How urgent this task is, pick the best fit from the enum"
    )
    status: Status = Field(description="Current lifecycle state")

    tags: list[Tag] = Field(
        default_factory=list,
        description="Up to 5 topical tags, each with an importance weight",
    )

    # Convention #3: explicit None default for optional fields.
    assignee: str | None = Field(
        default=None,
        description="Full name of the assignee, or null if unassigned",
    )
    due_date: str | None = Field(
        default=None,
        description="ISO-8601 date (YYYY-MM-DD) when due, or null if no deadline",
    )

    attachments: list[Attachment] = Field(
        default_factory=list,
        description=(
            "Zero or more attachments. Each must include a 'kind' discriminator ('text' or 'url')."
        ),
    )
