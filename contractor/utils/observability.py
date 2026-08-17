"""Centralized Langfuse / OpenInference instrumentation for Contractor.

All observability touchpoints flow through this module. Agents stay free of
Langfuse code; runtimes call init() once at startup and wrap each run with
run_context() so spans inherit pipeline-level metadata and tags.

Every public function is a no-op when Langfuse is disabled — safe to call
unconditionally from production code.
"""
from __future__ import annotations

import logging
import sys
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
    # Intentionally set even after a failed init: retrying on every call
    # would re-attempt the import/instrumentation (and re-log the warning)
    # for the whole run. A broken Langfuse degrades to no-op observability.
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
    """Attach metadata to the current Langfuse observation and trace.

    No-op if Langfuse is disabled or no span is currently active. Never raises
    — observability failures must not crash the pipeline.
    """
    if not _enabled():
        return
    observation_kwargs: dict[str, Any] = {}
    if name is not None:
        observation_kwargs["name"] = name
    if metadata:
        observation_kwargs["metadata"] = dict(metadata)
    if input is not None:
        observation_kwargs["input"] = input
    if output is not None:
        observation_kwargs["output"] = output

    propagation_kwargs: dict[str, Any] = {}
    if name is not None:
        propagation_kwargs["trace_name"] = name
    if user_id is not None:
        propagation_kwargs["user_id"] = user_id
    if session_id is not None:
        propagation_kwargs["session_id"] = session_id
    if tags:
        propagation_kwargs["tags"] = list(tags)
    if metadata:
        propagation_kwargs["metadata"] = dict(metadata)
    if not observation_kwargs and not propagation_kwargs:
        return
    try:
        from langfuse import get_client, propagate_attributes

        with propagate_attributes(**propagation_kwargs):
            if observation_kwargs:
                get_client().update_current_span(**observation_kwargs)
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

    # Enter/exit the contexts manually so a broken Langfuse client degrades to
    # a no-op span instead of crashing the run (this module never raises).
    span_cm = None
    span = None
    try:
        span_cm = client.start_as_current_observation(name=name)
        span = span_cm.__enter__()
    except Exception as exc:
        logger.warning("run_context: failed to open span: %s", exc)
        span_cm = None

    attributes_cm = None
    try:
        from langfuse import propagate_attributes

        attributes_cm = propagate_attributes(
            trace_name=name,
            user_id=user_id,
            session_id=session_id,
            tags=list(tags) if tags else None,
            metadata=dict(metadata) if metadata else None,
        )
        attributes_cm.__enter__()
    except Exception as exc:
        logger.warning("run_context: failed to propagate trace attributes: %s", exc)
        attributes_cm = None

    try:
        yield span
    finally:
        exc_info = sys.exc_info()
        if attributes_cm is not None:
            try:
                attributes_cm.__exit__(*exc_info)
            except Exception as exc:
                logger.warning(
                    "run_context: failed to close trace attributes: %s", exc
                )
        if span_cm is not None:
            try:
                span_cm.__exit__(*exc_info)
            except Exception as exc:
                logger.warning("run_context: failed to close span: %s", exc)
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
