"""Centralized Langfuse / OpenInference instrumentation for Contractor.

All observability touchpoints flow through this module. Agents stay free of
Langfuse code; runtimes call init() once at startup and wrap each run with
run_context() so spans inherit pipeline-level metadata and tags.

Every public function is a no-op when Langfuse is disabled — safe to call
unconditionally from production code.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from contractor.utils.settings import get_settings

logger = logging.getLogger(__name__)

_initialized = False


def _enabled() -> bool:
    return bool(get_settings().use_langfuse)


def init() -> None:
    """Idempotent Langfuse + OpenInference ADK instrumentation.

    Safe to call multiple times. If Langfuse is disabled via settings,
    returns immediately without importing langfuse.
    """
    global _initialized
    if _initialized:
        return
    if not _enabled():
        _initialized = True
        return
    try:
        from langfuse import get_client
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor

        GoogleADKInstrumentor().instrument()
        get_client()
    except Exception as exc:
        logger.warning("Langfuse init failed: %s", exc)
    _initialized = True


def tag_trace(
    *,
    name: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    input: Any = None,
    output: Any = None,
) -> None:
    """Attach metadata to the *current* Langfuse trace.

    No-op if Langfuse is disabled or no span is currently active. Never raises
    — observability failures must not crash the pipeline.
    """
    if not _enabled():
        return
    kwargs: dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if user_id is not None:
        kwargs["user_id"] = user_id
    if session_id is not None:
        kwargs["session_id"] = session_id
    if tags:
        kwargs["tags"] = list(tags)
    if metadata:
        kwargs["metadata"] = dict(metadata)
    if input is not None:
        kwargs["input"] = input
    if output is not None:
        kwargs["output"] = output
    if not kwargs:
        return
    try:
        from langfuse import get_client

        get_client().update_current_trace(**kwargs)
    except Exception as exc:
        logger.debug("tag_trace failed: %s", exc)


@contextmanager
def run_context(
    *,
    name: str,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    """Open a top-level span and tag the resulting trace.

    All ADK / LLM / tool spans created inside the `with` block become children
    of this span. flush() runs on exit so short CLI runs don't drop spans.

    Yields the span object (or None if Langfuse is disabled).
    """
    if not _enabled():
        yield None
        return
    try:
        from langfuse import get_client

        client = get_client()
    except Exception as exc:
        logger.warning("run_context: langfuse client unavailable: %s", exc)
        yield None
        return

    try:
        with client.start_as_current_span(name=name) as span:
            tag_trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                tags=tags,
                metadata=metadata,
            )
            yield span
    finally:
        flush()


def flush() -> None:
    """Flush pending spans. No-op if Langfuse is disabled."""
    if not _enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:
        logger.debug("flush failed: %s", exc)
