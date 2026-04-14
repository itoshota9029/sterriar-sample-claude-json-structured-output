"""Production patterns for getting structured JSON out of the Anthropic Claude API.

See each module for detail:
    - basic:     Force Tool Use via ``tool_choice`` for reliable JSON
    - retry:     Self-correcting retry loop when the model violates the schema
    - partial:   Recover from ``max_tokens`` truncation without losing progress
    - streaming: Yield partial parsed models while Claude streams
    - nested:    Design tips for complex schemas (Enum / Union / Literal / nested)
"""

from .basic import extract
from .nested import NestedExampleOutput
from .partial import extract_with_continuation
from .retry import RetriesExceededError, extract_with_retry
from .streaming import stream_extract

__all__ = [
    "extract",
    "extract_with_retry",
    "RetriesExceededError",
    "extract_with_continuation",
    "stream_extract",
    "NestedExampleOutput",
]
