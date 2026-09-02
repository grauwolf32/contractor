"""Bounded allocation-local Contractor State for one ADK Worker."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from collections.abc import Callable, Mapping
from typing import Any, Literal

from contractor_runtime.contracts import API_VERSION
from contractor_runtime.metrics import MetricsState
from contractor_runtime.observations import validate_workspace_observation

WORKER_STATE_SCHEMA_VERSION = 1
MAX_AGENT_STATE_SNAPSHOT_BYTES = 4 * 1024 * 1024
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
    ) -> dict[str, Any]:
        _require_identifier("invocationId", invocation_id, _INVOCATION_ID)
        _require_identifier("subtaskId", subtask_id, _SUBTASK_ID)
        invocation_metrics = _invocation_metrics_copy(metrics)
        invocation_workspace = _workspace_observation_copy(workspace)

        def mutate(state: dict[str, Any]) -> None:
            if state["currentInvocation"] is not None:
                raise WorkerStateError("a Worker invocation is already active")
            state["currentInvocation"] = {
                "invocationId": invocation_id,
                "subtaskId": subtask_id,
                "phase": "running",
                "metrics": invocation_metrics,
                "workspace": invocation_workspace,
            }

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

        def mutate(state: dict[str, Any]) -> None:
            current = _matching_current(state, invocation_id)
            completed = copy.deepcopy(current)
            completed["phase"] = phase
            completed["metrics"] = invocation_metrics
            if invocation_workspace is not _UNSET:
                completed["workspace"] = invocation_workspace
            state["lastCompletedInvocation"] = completed
            state["currentInvocation"] = None

        return await self._mutate(mutate)

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

    async def _mutate(self, transform: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        while True:
            metrics_snapshot = self.metrics.snapshot()
            async with self._lock:
                base_revision = int(self._state["stateRevision"])
                candidate = copy.deepcopy(self._state)

            transform(candidate)
            candidate["stateRevision"] = base_revision + 1
            candidate["metrics"] = metrics_snapshot
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


def _workspace_observation_copy(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    try:
        return validate_workspace_observation(value)
    except (TypeError, ValueError) as error:
        raise WorkerStateError("Worker workspace observation is invalid") from error


def _fit_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Admit optional metric detail while preserving the fixed State envelope."""

    candidate = copy.deepcopy(state)
    if _envelope_size(candidate) <= MAX_AGENT_STATE_SNAPSHOT_BYTES:
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
    if _envelope_size(candidate) > MAX_AGENT_STATE_SNAPSHOT_BYTES:
        raise WorkerStateError("mandatory Worker State exceeds the 4 MiB envelope")

    admitted_errors = _largest_fitting_prefix(candidate, metrics, "errors", errors)
    metrics["errors"] = errors[:admitted_errors]
    admitted_calls = _largest_fitting_prefix(candidate, metrics, "toolCalls", tool_calls)
    metrics["toolCalls"] = tool_calls[:admitted_calls]
    if _envelope_size(candidate) > MAX_AGENT_STATE_SNAPSHOT_BYTES:
        raise WorkerStateError("Worker State admission produced an oversized envelope")
    return candidate


def _trim_workspace_detail(state: dict[str, Any]) -> None:
    """Deterministically reduce optional invocation path detail until it fits."""

    while _envelope_size(state) > MAX_AGENT_STATE_SNAPSHOT_BYTES:
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
                        (len(values), -(invocation_order * 2 + section_order), workspace, section)
                    )
        if not choices:
            return
        _, _, workspace, section = max(choices, key=lambda item: (item[0], item[1]))
        values = workspace[section]
        next_length = len(values) // 2
        workspace[section] = values[:next_length]
        workspace["scopeComplete" if section == "scopePaths" else "detailComplete"] = False


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
        if _envelope_size(state) <= MAX_AGENT_STATE_SNAPSHOT_BYTES:
            low = middle
        else:
            high = middle - 1
    metrics[field] = []
    return low


def _envelope_size(state: Mapping[str, Any]) -> int:
    return len(
        json.dumps(
            {"apiVersion": API_VERSION, "state": state},
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
