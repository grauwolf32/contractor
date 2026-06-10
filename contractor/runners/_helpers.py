"""Shared event/Part decoding helpers used by both ``task_runner`` and
``agent_runner``. Kept module-private (single-underscore) — outside the
runners package, prefer pulling text from public ``AgentRunResult`` /
``TaskResult`` shapes instead of poking at raw ADK events.
"""

from __future__ import annotations

import logging

from google.adk.events import Event
from google.genai import types

logger = logging.getLogger(__name__)


def _extract_final_text(event: Event) -> str:
    """Pull the concatenated text from a final-response event."""
    if not event.is_final_response():
        return ""

    parts = getattr(getattr(event, "content", None), "parts", None) or []
    return "\n".join(
        text for part in parts if (text := getattr(part, "text", None))
    ).strip()


def _decode_part_text(part: types.Part | None) -> str:
    """Best-effort text extraction from an artifact Part."""
    if part is None:
        return ""

    text = part.text
    if text is not None:
        return text

    inline_data = getattr(part, "inline_data", None)
    data = getattr(inline_data, "data", None) if inline_data else None
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if isinstance(data, (bytes, bytearray)):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning(
                "Artifact part bytes are not valid UTF-8; decoding with "
                "errors='replace' — invalid bytes appear as U+FFFD"
            )
            return data.decode("utf-8", errors="replace")
    return ""
