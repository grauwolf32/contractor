"""Agio (Agent Insights and Observations) event taxonomy.

A Pythonized adaptation of ByteDance's Agio protocol from UI-TARS-desktop
(``multimodal/tarko/agio/src/index.ts``, Apache-2.0). Standardises event
naming and the common base fields used by ``cli/metrics.MetricsSink`` and
the analyzer; per-event payloads remain open via ``extra='allow'``.

Unlike upstream Agio (which sanitises tool arguments/results to sizes for
privacy), contractor emits full ``arguments`` and ``result`` payloads —
analysis scripts depend on them.
"""

from __future__ import annotations

import time
from enum import StrEnum, unique
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from cli.utils import utc_now_iso


@unique
class AgioEventType(StrEnum):
    """All event types emitted by the runner + plugins, in one place.

    Core upstream-Agio events are listed first; contractor-specific
    extensions follow.
    """

    # ── Core Agio: lifecycle ──
    AGENT_INITIALIZED = "agent_initialized"
    AGENT_RUN_START = "agent_run_start"
    AGENT_RUN_END = "agent_run_end"

    # ── Core Agio: performance ──
    AGENT_TTFT = "agent_ttft"
    AGENT_TPS = "agent_tps"

    # ── Core Agio: execution ──
    AGENT_LOOP_START = "agent_loop_start"
    AGENT_LOOP_END = "agent_loop_end"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    # ── Core Agio: feedback ──
    USER_FEEDBACK = "user_feedback"

    # ── Contractor extensions: tool callbacks ──
    TOOL_EXCEPTION = "tool_exception"  # was metrics_tool_exception_error
    LLM_USAGE = "llm_usage"            # was metrics_llm_usage
    FS_COVERAGE = "fs_coverage"        # was metrics_fs_coverage
    RUN_SUMMARY = "run_summary"        # was metrics_summary
    CALLBACK_SUMMARY = "callback_summary"

    # ── Contractor extensions: ADK trace (state snapshots, full event objects) ──
    ADK_TOOL_CALL = "adk_tool_call"
    ADK_TOOL_RESULT = "adk_tool_result"
    ADK_TOOL_ERROR = "adk_tool_error"
    ADK_EVENT = "adk_event"

    # ── Contractor extensions: pipeline / runner / task lifecycle ──
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_FINISHED = "pipeline_finished"
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    AGENT_RUN_STARTED = "agent_run_started"   # AgentRunner.run() boundary (RouterPipeline)
    AGENT_RUN_FINISHED = "agent_run_finished"
    TASK_STARTED = "task_started"
    TASK_FINISHED = "task_finished"
    TASK_FAILED = "task_failed"
    TASK_SKIPPED = "task_skipped"
    GLOBAL_TASK_FINISHED = "global_task_finished"
    ITERATION_STARTED = "iteration_started"
    ITERATION_FINISHED = "iteration_finished"
    ITERATION_RESULT = "iteration_result"
    FINAL_TEXT = "final_text"


# Lifecycle + extension types that the metrics sink should persist alongside
# the per-tool metrics events. Mirrors the membership in the StrEnum above.
ALL_AGIO_EVENT_TYPES: frozenset[str] = frozenset(t.value for t in AgioEventType)


class TokenUsage(BaseModel):
    """Token counters reported with ``agent_run_end`` / ``agent_loop_end`` /
    ``llm_usage`` events. Field names mirror upstream Agio's payload shape
    (``input`` / ``output`` / ``total``) plus contractor extras."""

    model_config = ConfigDict(extra="allow")

    input: int = 0
    output: int = 0
    total: int = 0
    thoughts: int = 0
    cached: int = 0


class AgioEvent(BaseModel):
    """Canonical envelope every event lands in on disk.

    The base carries identification + timing fields; everything event-specific
    flows through ``extra='allow'`` so emitters can attach whatever payload
    is natural without bloating this schema.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    timestamp: float        # milliseconds since epoch
    ts_iso: str             # ISO-8601 UTC string (analyzer-friendly)

    # Identity
    session_id: Optional[str] = None
    invocation_id: Optional[str] = None  # ADK invocation
    run_id: Optional[str] = None
    task_name: Optional[str] = None
    task_id: Optional[int] = None
    iteration: Optional[int] = None
    agent_name: Optional[str] = None


def make_agio_record(
    event_type: AgioEventType | str,
    *,
    task_name: Optional[str] = None,
    task_id: Optional[int] = None,
    iteration: Optional[int] = None,
    session_id: Optional[str] = None,
    invocation_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    **payload: Any,
) -> dict[str, Any]:
    """Build a flat Agio record dict ready for JSONL serialisation."""
    base = AgioEvent(
        type=str(event_type),
        timestamp=time.time() * 1000.0,
        ts_iso=utc_now_iso(),
        session_id=session_id,
        invocation_id=invocation_id,
        task_name=task_name,
        task_id=task_id,
        iteration=iteration,
        agent_name=agent_name,
    )
    record = base.model_dump(mode="json", exclude_none=False)
    record.update(payload)
    return record
