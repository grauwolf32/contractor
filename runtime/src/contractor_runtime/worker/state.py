"""Bounded allocation-local Contractor State for one ADK Worker."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from collections.abc import Callable, Mapping
from typing import Any, Literal

from contractor_runtime.contracts import (
    API_VERSION,
    AgentStateSnapshot,
)
from contractor_runtime.contracts.base import json_byte_size
from contractor_runtime.shared_config import get_shared_config
from contractor_runtime.telemetry.metrics import MetricsState
from contractor_runtime.toolsets.common.http_observations import HTTPObservationSink
from contractor_runtime.toolsets.common.report_publications import ReportPublicationSink
from contractor_runtime.worker.execution import WorkerExecutionState
from contractor_runtime.worker.observations import validate_workspace_observation

WORKER_STATE_SCHEMA_VERSION = 2
_SUBTASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_INVOCATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_METRIC_IDENTIFIER = re.compile(r"^[a-z0-9_]{1,64}$")
_TERMINAL_PHASES = frozenset({"succeeded", "failed", "cancelled"})
_MAX_UINT64 = 2**64 - 1
_MAX_INVOCATION_TOOL_NAMES = 256
_INVOCATION_COUNTER_FIELDS = (
    "modelCalls",
    "modelErrors",
    "inputTokens",
    "outputTokens",
    "totalTokens",
    "cachedInputTokens",
    "tokenUsageUnavailable",
    "toolCalls",
    "toolErrors",
)
_INVOCATION_METRIC_FIELDS = frozenset((*_INVOCATION_COUNTER_FIELDS, "tools", "truncated"))
_SUMMARIZER_COUNTER_FIELDS = (
    "modelCalls",
    "inputTokens",
    "outputTokens",
    "totalTokens",
    "tokenUsageUnavailable",
)
_FAILURE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
# A summarization request whose invocation ends before the summarizer records
# an outcome closes as one cancelled attempt without a summarizer model call
# (WRK-R107, WRK-R155).
_UNSTARTED_SUMMARY: dict[str, Any] = {
    "succeeded": False,
    "model_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "token_usage_unavailable": 0,
    "failure_code": "cancelled",
}

InvocationPhase = Literal["running", "succeeded", "failed", "cancelled"]


class _Unset:
    __slots__ = ()


_UNSET = _Unset()


class WorkerStateError(RuntimeError):
    """A safe invariant error for the allocation-local State store."""


class WorkerStateStore:
    """Own one revisioned, JSON-safe Contractor subtree.

    ``MetricsState`` remains the allocation-wide report reducer used by the
    existing tool implementations. Only this store publishes its bounded
    projection into ADK State. All store writes use one optimistic mutation
    boundary; potentially large JSON admission runs outside the mutation lock.
    """

    def __init__(self, metrics: MetricsState | None = None) -> None:
        self.metrics = metrics or MetricsState()
        self.execution = WorkerExecutionState()
        # Private allocation-owned resource, excluded from JSON/model state.
        self.http_session: Any = None
        self.http_captures: Any = None
        self.http_observation_sink: HTTPObservationSink | None = None
        self.caido_observation_sink: HTTPObservationSink | None = None
        self.report_publication_sink: ReportPublicationSink | None = None
        self.sitemap_read_flush: Any = None
        self.before_completion: Any = None
        self._lock = asyncio.Lock()
        self._state: dict[str, Any] = {
            "schemaVersion": WORKER_STATE_SCHEMA_VERSION,
            "stateRevision": 1,
            "metrics": self.metrics.snapshot(),
            "currentInvocation": None,
            "lastCompletedInvocation": None,
        }
        self._state = _fit_snapshot(self._state)

    async def begin_invocation(
        self,
        *,
        invocation_id: str,
        subtask_id: str,
        metrics: Mapping[str, Any],
        workspace: Mapping[str, Any] | None = None,
        summarizer_enabled: bool = False,
    ) -> dict[str, Any]:
        _require_identifier("invocationId", invocation_id, _INVOCATION_ID)
        _require_identifier("subtaskId", subtask_id, _SUBTASK_ID)
        invocation_metrics = _invocation_metrics_copy(metrics)
        invocation_workspace = _workspace_observation_copy(workspace)
        if type(summarizer_enabled) is not bool:
            raise WorkerStateError("Worker summarizer enabled flag is invalid")

        def mutate(state: dict[str, Any]) -> None:
            if state["currentInvocation"] is not None:
                raise WorkerStateError("a Worker invocation is already active")
            state["currentInvocation"] = {
                "invocationId": invocation_id,
                "subtaskId": subtask_id,
                "phase": "running",
                "metrics": invocation_metrics,
                "summarizer": _empty_summarizer(enabled=summarizer_enabled),
                "workspace": invocation_workspace,
            }

        return await self._mutate(mutate)

    async def request_summarization(self, *, invocation_id: str) -> dict[str, Any]:
        def mutate(state: dict[str, Any]) -> None:
            current = _matching_current(state, invocation_id)
            summarizer = current["summarizer"]
            if summarizer["phase"] != "not_requested":
                raise WorkerStateError("Worker summarizer cannot be requested in this phase")
            summarizer["phase"] = "requested"
            # The preceding revision is the complete normal-loop checkpoint
            # that deterministically caused this request.
            summarizer["requestStateRevision"] = state["stateRevision"]

        return await self._mutate(mutate)

    async def complete_summarization(
        self,
        *,
        invocation_id: str,
        succeeded: bool,
        model_calls: int,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        token_usage_unavailable: int,
        failure_code: str | None = None,
    ) -> dict[str, Any]:
        summary = _summarizer_usage_copy(
            succeeded=succeeded,
            model_calls=model_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            token_usage_unavailable=token_usage_unavailable,
            failure_code=failure_code,
        )

        def mutate(state: dict[str, Any]) -> None:
            current = _matching_current(state, invocation_id)
            summarizer = current["summarizer"]
            if summarizer["phase"] != "requested":
                raise WorkerStateError("Worker summarizer completion is not requested")
            request_revision = summarizer["requestStateRevision"]
            summarizer.clear()
            summarizer.update(summary)
            summarizer["requestStateRevision"] = request_revision

        return await self._mutate(mutate)

    async def publish_invocation_metrics(
        self,
        *,
        invocation_id: str,
        metrics: Mapping[str, Any],
        workspace: Mapping[str, Any] | _Unset | None = _UNSET,
    ) -> dict[str, Any]:
        invocation_metrics = _invocation_metrics_copy(metrics)
        invocation_workspace = (
            _UNSET if workspace is _UNSET else _workspace_observation_copy(workspace)
        )

        def mutate(state: dict[str, Any]) -> None:
            current = _matching_current(state, invocation_id)
            current["metrics"] = invocation_metrics
            if invocation_workspace is not _UNSET:
                current["workspace"] = invocation_workspace

        return await self._mutate(mutate)

    async def complete_invocation(
        self,
        *,
        invocation_id: str,
        phase: InvocationPhase,
        metrics: Mapping[str, Any],
        workspace: Mapping[str, Any] | _Unset | None = _UNSET,
    ) -> dict[str, Any]:
        if phase not in _TERMINAL_PHASES:
            raise WorkerStateError("Worker invocation completion phase is invalid")
        invocation_metrics = _invocation_metrics_copy(metrics)
        invocation_workspace = (
            _UNSET if workspace is _UNSET else _workspace_observation_copy(workspace)
        )
        closed_request = False

        def mutate(state: dict[str, Any]) -> None:
            nonlocal closed_request
            current = _matching_current(state, invocation_id)
            completed = copy.deepcopy(current)
            summarizer = completed["summarizer"]
            # Every terminal path closes here, so no completed invocation keeps
            # a pending summarization request (WRK-R107): the deadline, a
            # sandbox failure or cancellation can end the invocation after the
            # request and before the summarizer records its own outcome.
            closed_request = summarizer["phase"] == "requested"
            if closed_request:
                completed["summarizer"] = {
                    **_summarizer_usage_copy(**_UNSTARTED_SUMMARY),
                    "requestStateRevision": summarizer["requestStateRevision"],
                }
            completed["phase"] = phase
            completed["metrics"] = invocation_metrics
            if invocation_workspace is not _UNSET:
                completed["workspace"] = invocation_workspace
            state["lastCompletedInvocation"] = completed
            state["currentInvocation"] = None

        snapshot = await self._mutate(mutate)
        if closed_request:
            # The allocation aggregate counts the closed request as one
            # cancelled attempt once the State close is committed (WRK-R155).
            self.metrics.record_summarizer_attempt(**_UNSTARTED_SUMMARY)
        return snapshot

    async def sync_metrics(self) -> dict[str, Any]:
        """Publish allocation metrics after a non-ADK lifecycle mutation."""

        return await self._mutate(lambda _state: None)

    async def snapshot(self) -> dict[str, Any]:
        """Return an independent deep snapshot without encoding under the lock."""

        async with self._lock:
            return copy.deepcopy(self._state)

    async def encoded_snapshot_size(self) -> int:
        snapshot = await self.snapshot()
        return await asyncio.to_thread(_envelope_size, snapshot)

    async def agent_state_snapshot(self) -> AgentStateSnapshot:
        """Validate an immutable snapshot through the exact JSON wire boundary."""

        snapshot = await self.snapshot()
        return await asyncio.to_thread(_agent_state_snapshot, snapshot)

    async def _mutate(self, transform: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        while True:
            metrics_snapshot = self.metrics.snapshot()
            async with self._lock:
                base_revision = int(self._state["stateRevision"])
                # Every revision replaces the allocation metrics with a fresh
                # projection, so only the sections a transform may change are
                # copied (WRK-R169). The candidate is private to this call.
                candidate = {
                    key: metrics_snapshot if key == "metrics" else copy.deepcopy(value)
                    for key, value in self._state.items()
                }

            transform(candidate)
            candidate["stateRevision"] = base_revision + 1
            candidate = await asyncio.to_thread(_fit_snapshot, candidate)

            async with self._lock:
                if self._state["stateRevision"] != base_revision:
                    continue
                self._state = candidate
                return copy.deepcopy(candidate)

    def __repr__(self) -> str:
        state = self._state
        return (
            "WorkerStateStore("
            f"schema_version={state['schemaVersion']!r}, "
            f"state_revision={state['stateRevision']!r}, "
            f"has_current_invocation={state['currentInvocation'] is not None!r})"
        )


def _matching_current(state: dict[str, Any], invocation_id: str) -> dict[str, Any]:
    current = state.get("currentInvocation")
    if not isinstance(current, dict) or current.get("invocationId") != invocation_id:
        raise WorkerStateError("Worker invocation correlation does not match active State")
    if current.get("phase") != "running":
        raise WorkerStateError("active Worker invocation is not running")
    return current


def _require_identifier(field: str, value: str, pattern: re.Pattern[str]) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise WorkerStateError(f"{field} is invalid")


def _invocation_metrics_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the closed content-free invocation metrics projection."""

    if set(value) != _INVOCATION_METRIC_FIELDS:
        raise WorkerStateError("Worker invocation metrics fields are invalid")
    result: dict[str, Any] = {}
    for field in _INVOCATION_COUNTER_FIELDS:
        counter = value.get(field)
        if type(counter) is not int or not 0 <= counter <= _MAX_UINT64:
            raise WorkerStateError(f"Worker invocation metric {field} is invalid")
        result[field] = counter
    tools = value.get("tools")
    if not isinstance(tools, Mapping) or len(tools) > _MAX_INVOCATION_TOOL_NAMES:
        raise WorkerStateError("Worker invocation tool metrics are invalid")
    projected_tools: dict[str, dict[str, int]] = {}
    if any(not isinstance(name, str) for name in tools):
        raise WorkerStateError("Worker invocation tool metric name is invalid")
    for name in sorted(tools):
        aggregate = tools[name]
        if not isinstance(name, str) or _METRIC_IDENTIFIER.fullmatch(name) is None:
            raise WorkerStateError("Worker invocation tool metric name is invalid")
        if not isinstance(aggregate, Mapping) or set(aggregate) != {"calls", "failures"}:
            raise WorkerStateError("Worker invocation tool metric aggregate is invalid")
        calls = aggregate.get("calls")
        failures = aggregate.get("failures")
        if (
            type(calls) is not int
            or type(failures) is not int
            or not 0 <= failures <= calls <= _MAX_UINT64
        ):
            raise WorkerStateError("Worker invocation tool metric counters are invalid")
        projected_tools[name] = {"calls": calls, "failures": failures}
    detail_calls = sum(item["calls"] for item in projected_tools.values())
    detail_failures = sum(item["failures"] for item in projected_tools.values())
    if detail_calls > result["toolCalls"] or detail_failures > result["toolErrors"]:
        raise WorkerStateError("Worker invocation tool detail exceeds its aggregate")
    truncated = value.get("truncated")
    if type(truncated) is not bool:
        raise WorkerStateError("Worker invocation metric truncation flag is invalid")
    if not truncated and (
        detail_calls != result["toolCalls"] or detail_failures != result["toolErrors"]
    ):
        raise WorkerStateError("complete Worker invocation tool detail is inconsistent")
    result["tools"] = projected_tools
    result["truncated"] = truncated
    return result


def _empty_summarizer(*, enabled: bool) -> dict[str, Any]:
    return {
        "phase": "not_requested" if enabled else "disabled",
        "modelCalls": 0,
        "inputTokens": 0,
        "outputTokens": 0,
        "totalTokens": 0,
        "tokenUsageUnavailable": 0,
    }


def _summarizer_usage_copy(
    *,
    succeeded: bool,
    model_calls: int,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    token_usage_unavailable: int,
    failure_code: str | None,
) -> dict[str, Any]:
    if type(succeeded) is not bool:
        raise WorkerStateError("Worker summarizer outcome is invalid")
    values = {
        "modelCalls": model_calls,
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "totalTokens": total_tokens,
        "tokenUsageUnavailable": token_usage_unavailable,
    }
    for field in _SUMMARIZER_COUNTER_FIELDS:
        value = values[field]
        maximum = 1 if field in {"modelCalls", "tokenUsageUnavailable"} else _MAX_UINT64
        if type(value) is not int or not 0 <= value <= maximum:
            raise WorkerStateError(f"Worker summarizer metric {field} is invalid")
    if token_usage_unavailable > model_calls:
        raise WorkerStateError("Worker summarizer missing usage exceeds model calls")
    if succeeded:
        if model_calls != 1 or failure_code is not None:
            raise WorkerStateError("successful Worker summarizer outcome is invalid")
        return {"phase": "succeeded", **values}
    if not isinstance(failure_code, str) or _FAILURE_CODE.fullmatch(failure_code) is None:
        raise WorkerStateError("Worker summarizer failure code is invalid")
    return {"phase": "failed", **values, "failureCode": failure_code}


def _workspace_observation_copy(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    try:
        return validate_workspace_observation(value)
    except (TypeError, ValueError) as error:
        raise WorkerStateError("Worker workspace observation is invalid") from error


def _fit_snapshot(candidate: dict[str, Any]) -> dict[str, Any]:
    """Admit optional metric detail while preserving the fixed State envelope.

    The caller owns ``candidate``. Overflow admission edits its containers in
    place and only re-slices the shared record projections, never mutating them.
    """

    if _envelope_size(candidate) <= get_shared_config().worker_state_max_bytes:
        return candidate

    metrics = candidate.get("metrics")
    if not isinstance(metrics, dict):
        raise WorkerStateError("Worker State metrics projection is invalid")
    tool_calls = metrics.get("toolCalls")
    errors = metrics.get("errors")
    if not isinstance(tool_calls, list) or not isinstance(errors, list):
        raise WorkerStateError("Worker State metric detail is invalid")

    metrics["toolCalls"] = []
    metrics["errors"] = []
    metrics["truncated"] = True
    _trim_workspace_detail(candidate)
    if _envelope_size(candidate) > get_shared_config().worker_state_max_bytes:
        raise WorkerStateError("mandatory Worker State exceeds the 4 MiB envelope")

    admitted_errors = _largest_fitting_prefix(candidate, metrics, "errors", errors)
    metrics["errors"] = errors[:admitted_errors]
    admitted_calls = _largest_fitting_prefix(candidate, metrics, "toolCalls", tool_calls)
    metrics["toolCalls"] = tool_calls[:admitted_calls]
    if _envelope_size(candidate) > get_shared_config().worker_state_max_bytes:
        raise WorkerStateError("Worker State admission produced an oversized envelope")
    return candidate


def _trim_workspace_detail(state: dict[str, Any]) -> None:
    """Deterministically reduce optional invocation path detail until it fits."""

    while _envelope_size(state) > get_shared_config().worker_state_max_bytes:
        choices: list[tuple[int, int, dict[str, Any], str]] = []
        for invocation_order, field_name in enumerate(
            ("lastCompletedInvocation", "currentInvocation")
        ):
            invocation = state.get(field_name)
            workspace = invocation.get("workspace") if isinstance(invocation, dict) else None
            if not isinstance(workspace, dict):
                continue
            for section_order, section in enumerate(("scopePaths", "interactions")):
                values = workspace.get(section)
                if isinstance(values, list) and values:
                    choices.append(
                        (len(values), -(invocation_order * 2 + section_order), invocation, section)
                    )
        if not choices:
            return
        _, _, invocation, section = max(choices, key=lambda item: (item[0], item[1]))
        workspace = invocation["workspace"]
        values = workspace[section]
        # Replace the observation rather than edit it: a transform that a
        # revision conflict runs again inserts the same validated observation.
        invocation["workspace"] = {
            **workspace,
            section: values[: len(values) // 2],
            "scopeComplete" if section == "scopePaths" else "detailComplete": False,
        }


def _largest_fitting_prefix(
    state: dict[str, Any],
    metrics: dict[str, Any],
    field: str,
    values: list[Any],
) -> int:
    low = 0
    high = len(values)
    while low < high:
        middle = (low + high + 1) // 2
        metrics[field] = values[:middle]
        if _envelope_size(state) <= get_shared_config().worker_state_max_bytes:
            low = middle
        else:
            high = middle - 1
    metrics[field] = []
    return low


def _envelope_size(state: Mapping[str, Any]) -> int:
    # The Server measures the snapshot bound with the same rule (WRK-R165).
    return json_byte_size({"apiVersion": API_VERSION, "state": state})


def _agent_state_snapshot(state: Mapping[str, Any]) -> AgentStateSnapshot:
    encoded = json.dumps(
        {"apiVersion": API_VERSION, "state": state},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return AgentStateSnapshot.model_validate_json(encoded)
