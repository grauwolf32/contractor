"""Finite real-Gateway workload, shared by current tests and historical measurements."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from test_adk_runtime import build_context, stage_request

from contractor_runtime.contracts import WorkerSessionMode
from contractor_runtime.llm.client import new_gateway_client
from contractor_runtime.llm.openai import OpenAICompatibleGatewayLlm
from contractor_runtime.worker.runtime import AdkWorkerRuntime
from fakes.worker_state import WorkerState

FINALIZER_MARKER = "Contractor Worker result finalization input (JSON):\n"


@dataclass
class Work:
    event_scans: int = 0
    event_visits: int = 0
    event_index_visits: int = 0
    max_event_scan: int = 0
    session_reads: int = 0
    read_event_copies: int = 0
    max_read_events: int = 0
    request_bytes: int = 0
    max_request_bytes: int = 0
    model_calls: int = 0
    finalizer_calls: int = 0
    provider_seconds: float = 0.0
    session_read_seconds: float = 0.0
    callback_seconds: dict[str, float] = field(default_factory=dict)
    callback_calls: dict[str, int] = field(default_factory=dict)
    construction_seconds: float = 0.0
    first_request_seconds: float | None = None
    created_session_ids: list[str] = field(default_factory=list)
    session_ids: list[str] = field(default_factory=list)


class MeteredEvents(list):
    """Observe public Session event-list reads without changing their values."""

    def __init__(self, events: list, work: Work) -> None:
        super().__init__(events)
        self.work = work

    def __iter__(self):
        self.work.event_scans += 1
        self.work.max_event_scan = max(self.work.max_event_scan, len(self))
        for event in super().__iter__():
            self.work.event_visits += 1
            yield event

    def __getitem__(self, key):
        result = super().__getitem__(key)
        self.work.event_index_visits += len(result) if isinstance(key, slice) else 1
        return result

    def __deepcopy__(self, memo):
        # Counter ownership is external to Session data. Copies remain independent.
        return MeteredEvents([copy.deepcopy(event, memo) for event in self], self.work)


class MeteredSessions(InMemorySessionService):
    def __init__(self, work: Work) -> None:
        super().__init__()
        self.work = work

    async def create_session(self, **kwargs):
        session = await super().create_session(**kwargs)
        self.work.created_session_ids.append(session.id)
        return session

    async def get_session(self, **kwargs):
        began = time.perf_counter()
        session = await super().get_session(**kwargs)
        self.work.session_read_seconds += time.perf_counter() - began
        if session is not None:
            count = len(session.events)
            self.work.session_reads += 1
            self.work.read_event_copies += count
            self.work.max_read_events = max(self.work.max_read_events, count)
            session.events = MeteredEvents(session.events, self.work)
        return session


def _reply(message: dict) -> dict:
    return {
        "choices": [{"finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


async def run_workload(
    folder: Path,
    *,
    turns: int,
    width: int,
    mode: str,
    subtasks: int,
) -> dict[str, Any]:
    """Execute main turns plus one finalizer per sequential logical invocation."""
    work = Work()
    sessions = MeteredSessions(work)
    state = WorkerState()
    active = 0
    step = 0
    effects: list[tuple[int, int, str]] = []
    completions = []
    request_sizes = []
    snapshot_sizes = []
    starts = []
    finishes = []
    workload_began = 0.0

    async def probe(index: int, tool_context: ToolContext) -> dict[str, Any]:
        """Record one deterministic index under the owning logical invocation."""
        if type(index) is not int:
            raise ValueError("index must be an integer")
        previous = tool_context.state.get("nested", {"steps": []})
        if previous["steps"] != [[owner, value] for owner, value, _ in effects]:
            raise RuntimeError("Nested allocation State lost a prior tool update")
        if index == 0 and tool_context.state.get("temp:discard") is not None:
            raise RuntimeError("Temporary State leaked between logical invocations")
        effects.append((active, index, tool_context.invocation_id))
        # Replace the nested value so the native State delta owns this update.
        tool_context.state["nested"] = {"steps": [*previous["steps"], [active, index]]}
        tool_context.state["app:owner"] = {"allocation": "scale"}
        tool_context.state["user:owner"] = {"allocation": "scale"}
        tool_context.state["temp:discard"] = "temporary-scale-value"
        return {"index": index, "ok": True}

    def gateway(request: httpx.Request) -> httpx.Response:
        nonlocal step
        began = time.perf_counter()
        if work.first_request_seconds is None:
            work.first_request_seconds = began - workload_began
        work.request_bytes += len(request.content)
        work.max_request_bytes = max(work.max_request_bytes, len(request.content))
        request_sizes.append(len(request.content))
        value = json.loads(request.content)
        # This is the real custom adapter's public OpenAI-compatible payload.
        if value.get("response_format") is not None:
            work.finalizer_calls += 1
            text = value["messages"][-1]["content"]
            payload = json.loads(text.split(FINALIZER_MARKER, 1)[1])
            message = {
                "role": "assistant",
                "content": json.dumps(
                    {"subtaskId": payload["subtaskId"], "result": payload["resultText"]}
                ),
            }
        else:
            step += 1
            work.model_calls += 1
            if step < turns:
                calls = [
                    {
                        "id": f"scale-{active}-{step}-{slot}",
                        "type": "function",
                        "function": {
                            "name": "probe",
                            "arguments": json.dumps({"index": (step - 1) * width + slot}),
                        },
                    }
                    for slot in range(width)
                ]
                message = {"role": "assistant", "content": None, "tool_calls": calls}
            else:
                message = {"role": "assistant", "content": f"Scale result {active}"}
        response = httpx.Response(200, json=_reply(message), request=request)
        work.provider_seconds += time.perf_counter() - began
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(gateway), trust_env=False) as http:
        handle = new_gateway_client(
            base_url="https://scale-gateway.test/v1",
            api_key="scale-fixture-credential-never-exported",
            timeout_seconds=30,
            http_client=http,
        )
        model = OpenAICompatibleGatewayLlm(model="worker-model", client_handle=handle)
        context = build_context(folder, state, {"probe": probe})
        context = replace(context, worker_session_mode=WorkerSessionMode(mode))
        context = replace(
            context,
            model_policy=context.model_policy.model_copy(
                update={
                    "max_model_calls": turns + 1,
                    "max_tool_calls": max(width * (turns - 1), 1),
                    "max_total_tokens": 100_000_000,
                }
            ),
        )
        construction_began = time.perf_counter()
        runtime = AdkWorkerRuntime(context, model, session_service=sessions)
        await runtime.start()
        work.construction_seconds = time.perf_counter() - construction_began
        for name in (
            "before_run_callback",
            "before_model_callback",
            "after_model_callback",
            "before_tool_callback",
            "after_tool_callback",
        ):
            original = getattr(runtime._plugin, name)

            async def measured(*args, _original=original, _name=name, **kwargs):
                started = time.perf_counter()
                try:
                    return await _original(*args, **kwargs)
                finally:
                    work.callback_calls[_name] = work.callback_calls.get(_name, 0) + 1
                    work.callback_seconds[_name] = work.callback_seconds.get(_name, 0.0) + (
                        time.perf_counter() - started
                    )

            setattr(runtime._plugin, name, measured)
        began = workload_began = time.perf_counter()
        try:
            for active in range(subtasks):
                step = 0
                starts.append(len(effects))
                completion = await runtime.invoke(
                    stage_request().model_copy(
                        update={
                            "subtask_id": str(active),
                            "objective": f"Measure deterministic scale subtask {active}",
                            "deadline": datetime.now(UTC) + timedelta(minutes=15),
                        }
                    )
                )
                if completion.failure is not None:
                    raise RuntimeError(f"Scale workload failed: {completion.failure.code}")
                if (
                    completion.result is None
                    or completion.result.result != f"Scale result {active}"
                ):
                    raise RuntimeError("Scale workload returned a different result")
                completions.append(completion.invocation_id)
                snapshot = await state.snapshot()
                snapshot_sizes.append(await state.encoded_snapshot_size())
                if snapshot["currentInvocation"] is not None:
                    raise RuntimeError("Scale workload did not settle its invocation")
                session_id = runtime._session_lifecycle.shared_session_id
                if session_id is not None:
                    work.session_ids.append(session_id)
                finishes.append(len(effects))
            elapsed = time.perf_counter() - began
            report = state.metrics.build_report(report_id="scale-report", duration_ms=1)
            retained = len(state.metrics.tool_calls)
            first_record_id = state.metrics.tool_calls[0].call_id if retained else None
            last_record_id = state.metrics.tool_calls[-1].call_id if retained else None
            reported_tools = report.metrics.tools.get("probe")
            canonical = await state.snapshot()
            old_snapshot = copy.deepcopy(canonical)
            canonical["metrics"]["toolCalls"].clear()
            if await state.snapshot() != old_snapshot:
                raise RuntimeError("Returned snapshot aliases canonical State")
        finally:
            await runtime.abort(datetime.now(UTC) + timedelta(minutes=2))
        if sessions.sessions.get(runtime._app_name) or sessions.app_state.get(runtime._app_name):
            raise RuntimeError("Scale workload retained allocation session/app state")
        if sessions.user_state.get(runtime._app_name):
            raise RuntimeError("Scale workload retained allocation user state")
    return {
        "turns": turns,
        "width": width,
        "mode": mode,
        "subtasks": subtasks,
        "work": {key: value for key, value in vars(work).items()},
        "effects": effects,
        "starts": starts,
        "finishes": finishes,
        "logical_ids": completions,
        "snapshot_bytes": snapshot_sizes,
        "retained_tool_records": retained,
        "retained_first_call_id": first_record_id,
        "retained_last_call_id": last_record_id,
        "reported_tool_calls": reported_tools.calls if reported_tools is not None else 0,
        "detail_truncated": report.truncated,
        "report_bytes": len(report.model_dump_json(by_alias=True).encode()),
        "request_sizes": request_sizes,
        "elapsed_seconds": elapsed,
        "framework_seconds_excluding_mock_handler": max(0.0, elapsed - work.provider_seconds),
    }
