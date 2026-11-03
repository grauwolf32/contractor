"""compact@1 through the production ADK Worker coordinator (V417-003)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fakes.model import json_result, scripted_model, text_result, tool_call
from fakes.worker_state import WorkerState
from google.adk.events import Event, EventActions
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from test_adk_runtime import (
    SECRET,
    CapturingScriptedLlm,
    assert_balanced_tool_calls,
    build_context,
    stage_request,
)

from contractor_runtime.contracts import (
    CompactWorkerContextManagement,
    ModelPolicyRef,
    ResolvedInstructions,
    ResolvedModelPolicy,
    WorkerSessionMode,
)
from contractor_runtime.factories import WorkerBuildContext
from contractor_runtime.reasoning import (
    MAX_REASONING_PART_BYTES,
    MAX_REASONING_TOTAL_BYTES,
    REASONING_TRUNCATED_MEMBER,
    TRUNCATION_MARKER,
    tagged_size,
)
from contractor_runtime.worker import compaction as compaction_module
from contractor_runtime.worker.compaction import (
    HISTORY_TOOL_NAME,
    MARKER_AUTHOR,
    MARKER_METADATA_KEY,
    MAX_COMPACTION_SUMMARY_BYTES,
    MAX_HISTORY_PAGE_BYTES,
    MAX_HISTORY_PAGE_GROUPS,
    MAX_RECENT_GROUPS_BYTES,
    MAX_RETAINED_HISTORY_BYTES,
    MAX_RETAINED_HISTORY_GROUPS,
    CompactionCoordinator,
    ContextArchive,
    WorkerCompactionError,
    _history_page,
    compaction_boundary,
)
from contractor_runtime.worker.factory import AdkWorkerRuntimeFactory
from contractor_runtime.worker.runtime import AdkWorkerRuntime

COMPACTOR_MODEL = "worker-compactor-model"
# min(floor(1000 * 0.5), 1000 - 100)
BOUNDARY = 500


def usage(response: LlmResponse, prompt: int | None) -> LlmResponse:
    response.usage_metadata = (
        None
        if prompt is None
        else types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt, candidates_token_count=3, total_token_count=prompt + 3
        )
    )
    return response


def step(tag: str, prompt: int | None, *, call_id: str | None = None) -> LlmResponse:
    return usage(tool_call("step", {"tag": tag}, call_id=call_id or f"call-{tag}"), prompt)


def summary(text: str) -> LlmResponse:
    return json_result({"summary": text})


def compactor_policy() -> ResolvedModelPolicy:
    return ResolvedModelPolicy(
        ref=ModelPolicyRef(policyId="compactor", version="1", digest="sha256:" + "c" * 64),
        model=COMPACTOR_MODEL,
        contextWindowTokens=131_072,
        maxOutputTokens=2048,
        maxModelCalls=1,
        maxTotalTokens=140_000,
        temperature=0.0,
    )


def compaction(
    *,
    keep: int = 1,
    max_compactions: int = 8,
    ratio: float = 0.5,
    include_reasoning: bool = False,
) -> CompactWorkerContextManagement:
    return CompactWorkerContextManagement(
        strategy="compact@1",
        modelPolicy=compactor_policy(),
        instructions=ResolvedInstructions(
            ref="instructions/worker-compaction.md",
            digest="sha256:" + "d" * 64,
            text='Keep {decisions} and exact references; JSON: {"summary": "text"}.',
        ),
        includeReasoning=include_reasoning,
        contextWindowRatio=ratio,
        keepRecentGroups=keep,
        maxCompactions=max_compactions,
    )


def compact_context(
    tmp_path: Path,
    state: WorkerState,
    tools: dict[str, object],
    *,
    keep: int = 1,
    max_compactions: int = 8,
    max_model_calls: int = 32,
    max_tool_calls: int = 32,
) -> WorkerBuildContext:
    tmp_path.mkdir(parents=True, exist_ok=True)
    context = build_context(tmp_path, state, tools)
    return replace(
        context,
        worker_session_mode=WorkerSessionMode.ISOLATED,
        model_policy=context.model_policy.model_copy(
            update={
                "context_window_tokens": 1000,
                "max_output_tokens": 100,
                "max_model_calls": max_model_calls,
                "max_tool_calls": max_tool_calls,
                "max_total_tokens": 1_000_000,
            }
        ),
        compaction=compaction(keep=keep, max_compactions=max_compactions),
    )


async def compact_runtime(
    tmp_path: Path,
    state: WorkerState,
    tools: dict[str, object],
    model: object,
    compactor: object,
    **options: Any,
) -> AdkWorkerRuntime:
    context = compact_context(tmp_path, state, tools, **options)

    def select_model(build: WorkerBuildContext) -> object:
        return compactor if build.model_policy.model == COMPACTOR_MODEL else model

    runtime = await AdkWorkerRuntimeFactory(select_model).create(context)  # type: ignore[arg-type]
    assert isinstance(runtime, AdkWorkerRuntime)
    return runtime


def recording_step(effects: list[str]):
    async def step(tag: str) -> dict[str, Any]:
        """Record one externally visible effect for the tag."""
        effects.append(tag)
        return {"recorded": tag, "ordinal": len(effects)}

    return step


async def session_events(runtime: AdkWorkerRuntime) -> list[Event]:
    session_id = runtime._session_lifecycle.active_session_id
    if session_id is None:
        raise AssertionError("no active session")
    session = await runtime._session_service.get_session(
        app_name=runtime._app_name, user_id=runtime._user_id, session_id=session_id
    )
    assert session is not None
    return list(session.events)


def markers(events: list[Event]) -> list[Event]:
    return [
        event
        for event in events
        if event.author == MARKER_AUTHOR and MARKER_METADATA_KEY in (event.custom_metadata or {})
    ]


def tool_ids(payload: dict[str, Any]) -> list[str]:
    return [message["tool_call_id"] for message in payload["messages"] if message["role"] == "tool"]


def page_bytes(page: dict[str, Any]) -> int:
    return len(json.dumps(page, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def finish(runtime: AdkWorkerRuntime) -> Any:
    return runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))


def test_boundary_uses_ratio_and_output_reserve() -> None:
    policy = compactor_policy().model_copy(
        update={"context_window_tokens": 1000, "max_output_tokens": 100}
    )
    assert compaction_boundary(policy, 0.5) == 500
    assert compaction_boundary(policy, 0.95) == 900


def test_isolated_worker_continues_through_two_checkpoints(tmp_path: Path) -> None:
    """A2: one logical invocation, side effects once, finalizer only at the end."""

    async def scenario() -> None:
        effects: list[str] = []
        state = WorkerState()
        model = CapturingScriptedLlm(
            model="worker-model",
            responses=[
                step("a", 100),
                step("b", 200),
                step("c", 600),
                step("d", 150),
                step("e", 650),
                usage(text_result("done"), 120),
            ],
        )
        compactor = scripted_model(
            [summary("S1: a and b recorded"), summary("S2: a to d recorded")],
            model=COMPACTOR_MODEL,
            auto_result_finalizer=False,
        )
        runtime = await compact_runtime(
            tmp_path, state, {"step": recording_step(effects)}, model, compactor
        )
        try:
            task = asyncio.current_task()
            result = await runtime.invoke(stage_request())
            assert asyncio.current_task() is task
            assert result.failure is None, result.failure
            assert result.result is not None
            assert result.result.result == "done"
            assert result.result.summarized is False
            assert effects == ["a", "b", "c", "d", "e"]

            assert len(compactor.requests) == 2
            assert compactor.requests[0]["toolNames"] == []
            assert compactor.requests[0]["hasResponseSchema"] is True
            assert compaction().instructions.text in compactor.requests[0]["systemInstruction"]
            first_input = compactor.requests[0]["contentText"]
            assert '"tag":"a"' in first_input and '"tag":"b"' in first_input
            assert '"tag":"c"' not in first_input
            second_input = compactor.requests[1]["contentText"]
            assert "S1: a and b recorded" in second_input
            assert '"tag":"c"' in second_input and '"tag":"d"' in second_input
            assert '"tag":"a"' not in second_input

            # Main requests: 6 normal + 1 finalizer; the projection replaces
            # compacted groups by the checkpoint and keeps calls balanced.
            payloads = model.payloads
            assert len(payloads) == 7
            for payload in payloads[:6]:
                assert_balanced_tool_calls(payload)
            assert tool_ids(payloads[3]) == ["call-c"]
            assert tool_ids(payloads[4]) == ["call-c", "call-d"]
            assert tool_ids(payloads[5]) == ["call-e"]
            checkpoint = payloads[5]["messages"]
            first_user = next(message for message in checkpoint if message["role"] == "user")
            text = json.dumps(first_user["content"])
            assert "Build a report" in text and "S2: a to d recorded" in text
            assert HISTORY_TOOL_NAME in text
            assert "S1: a and b recorded" not in json.dumps(checkpoint)
            assert HISTORY_TOOL_NAME in {
                tool["function"]["name"] for tool in payloads[0].get("tools", [])
            }

            report = state.metrics.build_report(report_id="worker-report", duration_ms=1)
            assert report.metrics.worker_budget is not None
            # 6 main + 2 compactor + 1 finalizer, each counted once.
            assert report.metrics.worker_budget.observed_model_calls == 9
            assert report.metrics.worker_budget.observed_tool_calls == 5
            assert runtime._compaction is not None and not runtime._compaction.active
        finally:
            await finish(runtime)

    asyncio.run(scenario())


def test_checkpoint_markers_commit_once_per_cycle(tmp_path: Path) -> None:
    async def scenario() -> None:
        effects: list[str] = []
        model = scripted_model(
            [step("a", 100), step("b", 600), step("c", 100), usage(text_result("ok"), 90)],
            model="worker-model",
        )
        compactor = scripted_model(
            [summary("S1")], model=COMPACTOR_MODEL, auto_result_finalizer=False
        )
        runtime = await compact_runtime(
            tmp_path, WorkerState(), {"step": recording_step(effects)}, model, compactor
        )
        observed: list[list[Event]] = []
        original = runtime._compaction.end  # type: ignore[union-attr]

        def capture_end() -> None:
            observed.append([])
            original()

        try:
            events_before_end: list[Event] = []

            async def snapshot_on_finalizer(*_args: Any, **_kwargs: Any) -> None:
                events_before_end.extend(await session_events(runtime))

            runtime._compaction.end = capture_end  # type: ignore[method-assign,union-attr]
            hook = runtime._plugin.before_result_finalizer_call

            async def before_finalizer(*, invocation_id: str) -> None:
                await snapshot_on_finalizer()
                await hook(invocation_id=invocation_id)

            runtime._plugin.before_result_finalizer_call = before_finalizer  # type: ignore[method-assign]
            result = await runtime.invoke(stage_request())
            assert result.result is not None, result.failure
            committed = markers(events_before_end)
            assert len(committed) == 1
            metadata = committed[0].custom_metadata[MARKER_METADATA_KEY]  # type: ignore[index]
            assert metadata == {"generation": 1, "firstGroup": 1, "lastGroup": 1}
            assert committed[0].content is None
            assert observed == [[]]
        finally:
            await finish(runtime)

    asyncio.run(scenario())


def test_missing_usage_never_triggers_an_estimated_compaction(tmp_path: Path) -> None:
    async def scenario() -> None:
        effects: list[str] = []
        model = scripted_model(
            [step("a", None), step("b", None), usage(text_result("ok"), None)],
            model="worker-model",
        )
        compactor = scripted_model([], model=COMPACTOR_MODEL, auto_result_finalizer=False)
        runtime = await compact_runtime(
            tmp_path, WorkerState(), {"step": recording_step(effects)}, model, compactor
        )
        try:
            result = await runtime.invoke(stage_request())
            assert result.result is not None, result.failure
            assert compactor.requests == []
            assert effects == ["a", "b"]
        finally:
            await finish(runtime)

    asyncio.run(scenario())


def test_high_usage_without_eligible_prefix_fails_explicitly(tmp_path: Path) -> None:
    async def scenario() -> None:
        effects: list[str] = []
        model = scripted_model([step("a", 900)], model="worker-model")
        compactor = scripted_model([], model=COMPACTOR_MODEL, auto_result_finalizer=False)
        runtime = await compact_runtime(
            tmp_path, WorkerState(), {"step": recording_step(effects)}, model, compactor, keep=4
        )
        try:
            result = await runtime.invoke(stage_request())
            assert result.failure is not None
            assert result.failure.code == "worker_compaction_failed"
            assert "ineffective" in result.failure.message
            assert compactor.requests == []
            assert len(model.requests) == 1
        finally:
            await finish(runtime)

    asyncio.run(scenario())


def test_ineffective_checkpoint_keeps_marker_and_stops_before_dispatch(tmp_path: Path) -> None:
    async def scenario() -> None:
        effects: list[str] = []
        model = scripted_model(
            [step("a", 100), step("b", 600), step("c", 700)], model="worker-model"
        )
        compactor = scripted_model(
            [summary("S1")], model=COMPACTOR_MODEL, auto_result_finalizer=False
        )
        state = WorkerState()
        runtime = await compact_runtime(
            tmp_path, state, {"step": recording_step(effects)}, model, compactor
        )
        appended: list[Event] = []
        append = runtime._session_service.append_event

        async def recording_append(session: Any, event: Event) -> Event:
            stored = await append(session, event)
            appended.append(event)
            return stored

        runtime._session_service.append_event = recording_append  # type: ignore[method-assign]
        try:
            result = await runtime.invoke(stage_request())
            assert result.failure is not None
            assert result.failure.code == "worker_compaction_failed"
            assert "ineffective" in result.failure.message
            # The response that proved ineffectiveness requested c; it never ran.
            assert effects == ["a", "b"]
            assert len(markers(appended)) == 1
            report = state.metrics.build_report(report_id="worker-report", duration_ms=1)
            assert report.metrics.worker_budget is not None
            assert report.metrics.worker_budget.observed_model_calls == 4
        finally:
            await finish(runtime)

    asyncio.run(scenario())


def oversized_summary() -> LlmResponse:
    return summary("x" * (MAX_COMPACTION_SUMMARY_BYTES + 1))


def output_limited() -> LlmResponse:
    response = summary("cut")
    response.finish_reason = types.FinishReason.MAX_TOKENS
    return response


@pytest.mark.parametrize(
    ("response", "cause"),
    [
        (text_result(""), "empty_output"),
        (text_result("not json"), "invalid_output"),
        (json_result({"summary": "S", "refs": []}), "invalid_output"),
        (json_result({"summary": "   "}), "invalid_output"),
        (oversized_summary(), "summary_too_large"),
        (output_limited(), "output_limit_exceeded"),
        (summary(SECRET), "unsafe_output"),
    ],
)
def test_rejected_compactor_output_adds_no_marker_and_no_later_calls(
    tmp_path: Path, response: LlmResponse, cause: str
) -> None:
    async def scenario() -> None:
        effects: list[str] = []
        model = scripted_model([step("a", 100), step("b", 600)], model="worker-model")
        compactor = scripted_model([response], model=COMPACTOR_MODEL, auto_result_finalizer=False)
        runtime = await compact_runtime(
            tmp_path, WorkerState(), {"step": recording_step(effects)}, model, compactor
        )
        appended: list[Event] = []
        append = runtime._session_service.append_event

        async def recording_append(session: Any, event: Event) -> Event:
            appended.append(event)
            return await append(session, event)

        runtime._session_service.append_event = recording_append  # type: ignore[method-assign]
        try:
            result = await runtime.invoke(stage_request())
            assert result.failure is not None
            assert result.failure.code == "worker_compaction_failed"
            assert f"({cause})" in result.failure.message
            assert result.failure.retryable
            assert markers(appended) == []
            assert len(model.requests) == 2
            assert effects == ["a", "b"]
        finally:
            await finish(runtime)

    asyncio.run(scenario())


def test_cycle_ceiling_is_a_non_retryable_context_limit(tmp_path: Path) -> None:
    async def scenario() -> None:
        effects: list[str] = []
        model = scripted_model(
            [step("a", 100), step("b", 600), step("c", 100), step("d", 600)],
            model="worker-model",
        )
        compactor = scripted_model(
            [summary("S1")], model=COMPACTOR_MODEL, auto_result_finalizer=False
        )
        runtime = await compact_runtime(
            tmp_path,
            WorkerState(),
            {"step": recording_step(effects)},
            model,
            compactor,
            max_compactions=1,
        )
        try:
            result = await runtime.invoke(stage_request())
            assert result.failure is not None
            assert result.failure.code == "worker_context_limit_exceeded"
            assert not result.failure.retryable
            assert len(compactor.requests) == 1
            assert effects == ["a", "b", "c", "d"]
        finally:
            await finish(runtime)

    asyncio.run(scenario())


def test_compactor_call_exhausting_the_worker_budget_keeps_budget_code(tmp_path: Path) -> None:
    async def scenario() -> None:
        model = scripted_model([step("a", 100), step("b", 600)], model="worker-model")
        compactor = scripted_model(
            [summary("S1")], model=COMPACTOR_MODEL, auto_result_finalizer=False
        )
        runtime = await compact_runtime(
            tmp_path,
            WorkerState(),
            {"step": recording_step([])},
            model,
            compactor,
            max_model_calls=2,
        )
        try:
            result = await runtime.invoke(stage_request())
            assert result.failure is not None
            assert result.failure.code == "worker_budget_exhausted"
            assert compactor.requests == []
        finally:
            await finish(runtime)

    asyncio.run(scenario())


def test_history_tool_reads_compacted_groups_with_tool_budget(tmp_path: Path) -> None:
    async def scenario() -> None:
        async def secret_step(tag: str) -> dict[str, Any]:
            """Return a value that carries a private setting."""
            return {"recorded": tag, "token": SECRET}

        model = CapturingScriptedLlm(
            model="worker-model",
            responses=[
                step("a", 100),
                step("b", 600),
                usage(tool_call(HISTORY_TOOL_NAME, {"group": 1}, call_id="history-1"), 100),
                usage(tool_call(HISTORY_TOOL_NAME, {"group": 2}, call_id="history-2"), 100),
                usage(text_result("ok"), 100),
            ],
        )
        compactor = scripted_model(
            [summary("S1")], model=COMPACTOR_MODEL, auto_result_finalizer=False
        )
        state = WorkerState()
        runtime = await compact_runtime(tmp_path, state, {"step": secret_step}, model, compactor)
        try:
            result = await runtime.invoke(stage_request())
            assert result.result is not None, result.failure
            history = {
                message["tool_call_id"]: json.loads(message["content"])
                for message in model.payloads[4]["messages"]
                if message["role"] == "tool"
            }
            page = history["history-1"]
            assert page["historical"] is True
            assert page["lastCompactedGroup"] == 1
            assert [item["group"] for item in page["groups"]] == [1]
            assert {"tag": "a"} in [
                part["functionCall"]["args"]
                for event in page["groups"][0]["events"]
                for part in event["parts"]
                if "functionCall" in part
            ]
            assert SECRET not in json.dumps(page)
            assert history["history-2"]["error"]["code"] == "invalid_history_group"
            report = state.metrics.build_report(report_id="worker-report", duration_ms=1)
            assert report.metrics.worker_budget is not None
            assert report.metrics.worker_budget.observed_tool_calls == 4
        finally:
            await finish(runtime)

    asyncio.run(scenario())


@pytest.mark.parametrize("stored", [True, False])
def test_lost_marker_acknowledgement_reconciles_or_fails_before_continuation(
    tmp_path: Path, stored: bool
) -> None:
    async def scenario() -> None:
        effects: list[str] = []
        model = scripted_model(
            [step("a", 100), step("b", 600), step("c", 100), usage(text_result("ok"), 90)],
            model="worker-model",
        )
        compactor = scripted_model(
            [summary("S1")], model=COMPACTOR_MODEL, auto_result_finalizer=False
        )
        runtime = await compact_runtime(
            tmp_path, WorkerState(), {"step": recording_step(effects)}, model, compactor
        )
        append = runtime._session_service.append_event

        async def lossy_append(session: Any, event: Event) -> Event:
            if not markers([event]):
                return await append(session, event)
            if stored:
                await append(session, event)
            raise ConnectionError("acknowledgement lost")

        runtime._session_service.append_event = lossy_append  # type: ignore[method-assign]
        try:
            result = await runtime.invoke(stage_request())
            if stored:
                assert result.result is not None, result.failure
                assert effects == ["a", "b", "c"]
            else:
                assert result.failure is not None
                assert result.failure.code == "worker_compaction_failed"
                assert "(marker_not_committed)" in result.failure.message
                assert effects == ["a", "b"]
                assert len(model.requests) == 2
            assert not runtime._session_lifecycle.failed
        finally:
            await finish(runtime)

    asyncio.run(scenario())


def test_conflicting_marker_identity_fences_the_session(tmp_path: Path) -> None:
    async def scenario() -> None:
        model = scripted_model([step("a", 100), step("b", 600)], model="worker-model")
        compactor = scripted_model(
            [summary("S1")], model=COMPACTOR_MODEL, auto_result_finalizer=False
        )
        runtime = await compact_runtime(
            tmp_path, WorkerState(), {"step": recording_step([])}, model, compactor
        )
        append = runtime._session_service.append_event

        async def conflicting_append(session: Any, event: Event) -> Event:
            if not markers([event]):
                return await append(session, event)
            # Same ID, different content: an equal ID never proves the append.
            await append(session, event.model_copy(update={"author": "other"}))
            raise ConnectionError("acknowledgement lost")

        runtime._session_service.append_event = conflicting_append  # type: ignore[method-assign]
        try:
            result = await runtime.invoke(stage_request())
            assert result.failure is not None
            assert result.failure.code == "worker_compaction_failed"
            assert "(marker_unverified)" in result.failure.message
            assert runtime._session_lifecycle.failed
            assert not runtime._accepting
        finally:
            await finish(runtime)

    asyncio.run(scenario())


def test_cancellation_during_compactor_call_adds_no_marker(tmp_path: Path) -> None:
    async def scenario() -> None:
        effects: list[str] = []
        model = scripted_model([step("a", 100), step("b", 600)], model="worker-model")
        compactor = scripted_model(
            [summary("S1")], model=COMPACTOR_MODEL, auto_result_finalizer=False, block=True
        )
        runtime = await compact_runtime(
            tmp_path, WorkerState(), {"step": recording_step(effects)}, model, compactor
        )
        appended: list[Event] = []
        append = runtime._session_service.append_event

        async def recording_append(session: Any, event: Event) -> Event:
            appended.append(event)
            return await append(session, event)

        runtime._session_service.append_event = recording_append  # type: ignore[method-assign]
        try:
            task = asyncio.create_task(runtime.invoke(stage_request()))
            await asyncio.wait_for(compactor.blocked.wait(), timeout=10)
            runtime.cancel_active(task)
            with pytest.raises(asyncio.CancelledError):
                await task
            assert markers(appended) == []
            assert effects == ["a", "b"]
            assert runtime._compaction is not None and not runtime._compaction.active
            assert runtime._invocation.task is None
            assert not runtime._invocation.lock.locked()
        finally:
            await finish(runtime)

    asyncio.run(scenario())


# Archive, projection and page bounds on the production structures.


def model_event(event_id: str, parts: list[types.Part], **fields: Any) -> Event:
    return Event(
        id=event_id,
        author="contractor_worker",
        invocation_id="sdk-1",
        content=types.Content(role="model", parts=parts),
        **fields,
    )


def call_event(event_id: str, *call_ids: str, name: str = "step") -> Event:
    return model_event(
        event_id,
        [
            types.Part(function_call=types.FunctionCall(id=call_id, name=name, args={}))
            for call_id in call_ids
        ],
    )


def reply_event(event_id: str, *call_ids: str, name: str = "step", value: Any = None) -> Event:
    return Event(
        id=event_id,
        author="contractor_worker",
        invocation_id="sdk-1",
        content=types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=call_id, name=name, response={"value": value}
                    )
                )
                for call_id in call_ids
            ],
        ),
    )


def archive() -> ContextArchive:
    return ContextArchive(secrets=(SECRET,), skill_tools=frozenset({"load_skill"}))


@pytest.mark.parametrize(
    ("events", "reason"),
    [
        ([call_event("e1", "c1"), call_event("e1", "c2")], "duplicate_event"),
        ([model_event("e1", [types.Part(text="x")], partial=True)], "partial_event"),
        ([model_event("e1", [types.Part(text="x")], error_code="ERR")], "error_event"),
        ([model_event("e1", [types.Part(text="x")], branch="other")], "foreign_scope"),
        (
            [
                model_event(
                    "e1",
                    [types.Part(text="x")],
                    actions=EventActions(
                        compaction={
                            "start_timestamp": 0.0,
                            "end_timestamp": 1.0,
                            "compacted_content": types.Content(
                                role="model", parts=[types.Part(text="native")]
                            ),
                        }
                    ),
                )
            ],
            "native_compaction",
        ),
        ([call_event("e1", "c1"), call_event("e2", "c2")], "interleaved_calls"),
        ([reply_event("e1", "c1")], "foreign_response"),
        ([call_event("e1", "c1"), model_event("e2", [types.Part(text="x")])], "split_group"),
    ],
)
def test_unsafe_history_fails_before_any_compactor_call(events: list[Event], reason: str) -> None:
    async def scenario() -> None:
        index = archive()
        for event in events:
            await index.observe(event)
        assert index.unsafe == reason
        coordinator = CompactionCoordinator(
            compaction(keep=1),
            worker_policy=compactor_policy().model_copy(
                update={"context_window_tokens": 1000, "max_output_tokens": 100}
            ),
            skill_tools=frozenset(),
        )
        coordinator.begin(invocation_id="inv-1", task_prompt="task", secrets=())
        coordinator._archive = index
        with pytest.raises(WorkerCompactionError) as raised:
            await coordinator.plan()
        assert raised.value.code == "worker_compaction_failed"
        assert raised.value.cause == reason

    asyncio.run(scenario())


def test_incomplete_group_fails_before_compactor_call() -> None:
    async def scenario() -> None:
        coordinator = new_coordinator()
        await coordinator.observe(call_event("e1", "c1", "c2"))
        await coordinator.observe(reply_event("e2", "c1"))
        with pytest.raises(WorkerCompactionError) as raised:
            await coordinator.plan()
        assert raised.value.cause == "incomplete_history"

    asyncio.run(scenario())


def new_coordinator(*, keep: int = 1, include_reasoning: bool = False) -> CompactionCoordinator:
    coordinator = CompactionCoordinator(
        compaction(keep=keep, include_reasoning=include_reasoning),
        worker_policy=compactor_policy().model_copy(
            update={"context_window_tokens": 1000, "max_output_tokens": 100}
        ),
        skill_tools=frozenset({"load_skill"}),
    )
    coordinator.begin(invocation_id="inv-1", task_prompt="task", secrets=(SECRET,))
    return coordinator


async def add_group(coordinator: CompactionCoordinator, ordinal: int, value: Any = None) -> None:
    await coordinator.observe(call_event(f"call-{ordinal}", f"c{ordinal}"))
    await coordinator.observe(reply_event(f"reply-{ordinal}", f"c{ordinal}", value=value))


def test_parallel_calls_form_one_group_and_markers_cover_exact_ids() -> None:
    async def scenario() -> None:
        coordinator = new_coordinator()
        await coordinator.observe(call_event("p", "c1", "c2"))
        await coordinator.observe(reply_event("r1", "c1"))
        await coordinator.observe(reply_event("r2", "c2"))
        await add_group(coordinator, 2)
        plan = await coordinator.plan()
        assert (plan.first_group, plan.last_group) == (1, 1)
        assert plan.event_ids == frozenset({"p", "r1", "r2"})
        marker = coordinator.marker(plan, sdk_invocation_id="sdk-1")
        coordinator.commit(plan, "S1", marker)
        events = [
            Event(id="u", author="user", content=types.Content(role="user", parts=[])),
            call_event("p", "c1", "c2"),
            reply_event("r1", "c1"),
            reply_event("r2", "c2"),
            call_event("call-2", "c2x"),
            marker,
        ]
        projected = coordinator.project(events)
        assert projected is not None
        assert projected[0].role == "user" and "S1" in (projected[0].parts[0].text or "")
        assert len(projected) == 2
        with pytest.raises(WorkerCompactionError) as raised:
            coordinator.project([*events, call_event("foreign", "f1")])
        assert raised.value.cause == "foreign_event"

    asyncio.run(scenario())


def test_skill_content_and_secrets_never_reach_history_pages() -> None:
    async def scenario() -> None:
        coordinator = new_coordinator()
        await coordinator.observe(call_event("s", "c1", name="load_skill"))
        await coordinator.observe(
            reply_event("sr", "c1", name="load_skill", value="SKILL-BODY " + SECRET)
        )
        await add_group(coordinator, 2, value="token " + SECRET)
        await add_group(coordinator, 3)
        plan = await coordinator.plan()
        coordinator.commit(plan, "S1", coordinator.marker(plan, sdk_invocation_id="sdk-1"))
        assert "SKILL-BODY" not in plan.document and SECRET not in plan.document
        page = coordinator.history_page("inv-1", 1, 0)
        rendered = json.dumps(page)
        assert "SKILL-BODY" not in rendered and SECRET not in rendered
        assert [item["group"] for item in page["groups"]] == [1, 2]
        with pytest.raises(ValueError):
            coordinator.history_page("inv-2", 1, 0)
        with pytest.raises(ValueError):
            coordinator.history_page("inv-1", 3, 0)
        coordinator.end()
        with pytest.raises(ValueError):
            coordinator.history_page("inv-1", 1, 0)

    asyncio.run(scenario())


def test_history_pages_split_large_multibyte_groups_and_resume_exactly() -> None:
    async def scenario() -> None:
        coordinator = new_coordinator()
        # 3-byte characters across several 64-KiB bounded text values.
        large = "€" * 30_000
        await coordinator.observe(call_event("big", "c1"))
        await coordinator.observe(
            reply_event("big-r", "c1", value={"a": large, "b": large, "c": large})
        )
        for ordinal in range(2, 12):
            await add_group(coordinator, ordinal)
        plan = await coordinator.plan()
        coordinator.commit(plan, "S1", coordinator.marker(plan, sdk_invocation_id="sdk-1"))
        projection = coordinator._archive.groups[0].projection  # type: ignore[union-attr]
        rebuilt, position, pages = "", 0, 0
        while True:
            page = coordinator.history_page("inv-1", 1, position)
            pages += 1
            assert page_bytes(page) <= MAX_HISTORY_PAGE_BYTES
            [item] = page["groups"]
            assert item["group"] == 1 and item["position"] == position
            rebuilt += item["fragment"]
            if item["complete"]:
                break
            assert page["nextGroup"] == 1
            position = page["nextPosition"]
        assert rebuilt == projection
        assert pages > 2
        assert page["nextGroup"] == 2 and page["nextPosition"] == 0
        small = coordinator.history_page("inv-1", 2, 0)
        assert len(small["groups"]) == MAX_HISTORY_PAGE_GROUPS
        assert small["nextGroup"] == 2 + MAX_HISTORY_PAGE_GROUPS

    asyncio.run(scenario())


def test_history_page_never_exceeds_byte_and_group_bounds_for_unit_pages() -> None:
    groups = tuple(
        compaction_module._Group(
            ordinal=index + 1,
            event_ids=(f"e{index}",),
            projection=json.dumps([{"role": "model", "parts": [{"text": "é" * 3000}]}]),
            size=0,
        )
        for index in range(20)
    )
    page = _history_page(groups, 1, 0)
    assert page_bytes(page) <= MAX_HISTORY_PAGE_BYTES
    assert 1 <= len(page["groups"]) <= MAX_HISTORY_PAGE_GROUPS
    assert page["nextGroup"] == 1 + len(page["groups"])


def test_recent_tail_over_its_byte_ceiling_fails_without_dropping_groups() -> None:
    async def scenario() -> None:
        coordinator = new_coordinator(keep=2)
        await add_group(coordinator, 1)
        # Two protected groups of about 64 KiB each exceed 128 KiB together.
        await add_group(coordinator, 2, value="x" * (64 * 1024))
        await add_group(coordinator, 3, value="y" * (64 * 1024))
        tail = sum(group.size for group in coordinator._archive.groups[1:])  # type: ignore[union-attr]
        assert tail > MAX_RECENT_GROUPS_BYTES
        with pytest.raises(WorkerCompactionError) as raised:
            await coordinator.plan()
        assert raised.value.code == "worker_context_limit_exceeded"
        assert raised.value.cause == "recent_groups"
        assert not raised.value.retryable

    asyncio.run(scenario())


def test_retained_history_accepts_its_group_ceiling_and_rejects_one_more() -> None:
    async def scenario() -> None:
        index = archive()
        for ordinal in range(1, MAX_RETAINED_HISTORY_GROUPS + 1):
            await index.observe(model_event(f"t{ordinal}", [types.Part(text="n")]))
        assert len(index.groups) == MAX_RETAINED_HISTORY_GROUPS
        with pytest.raises(WorkerCompactionError) as raised:
            await index.observe(model_event("one-more", [types.Part(text="n")]))
        assert raised.value.cause == "retained_history"
        assert raised.value.code == "worker_context_limit_exceeded"

    asyncio.run(scenario())


def test_retained_history_bytes_stop_at_the_ceiling() -> None:
    async def scenario() -> None:
        index = archive()
        # Each group carries one 64-KiB bounded text value.
        text = "z" * (64 * 1024)
        accepted = 0
        with pytest.raises(WorkerCompactionError) as raised:
            for ordinal in range(1, 1024):
                await index.observe(model_event(f"t{ordinal}", [types.Part(text=text)]))
                accepted += 1
        assert raised.value.cause == "retained_history"
        retained = sum(group.size for group in index.groups)
        assert retained <= MAX_RETAINED_HISTORY_BYTES
        assert retained + index.groups[0].size > MAX_RETAINED_HISTORY_BYTES
        assert accepted == len(index.groups)

    asyncio.run(scenario())


def thought_call_event(event_id: str, call_id: str, thought: str) -> Event:
    return model_event(
        event_id,
        [
            types.Part(text=thought, thought=True),
            types.Part(function_call=types.FunctionCall(id=call_id, name="step", args={})),
        ],
    )


async def add_thought_group(coordinator: CompactionCoordinator, ordinal: int, thought: str) -> None:
    await coordinator.observe(thought_call_event(f"call-{ordinal}", f"c{ordinal}", thought))
    await coordinator.observe(reply_event(f"reply-{ordinal}", f"c{ordinal}"))


def reasoning_parts(document: str) -> list[str]:
    payload = json.loads(document.split("\n", 1)[1])
    return [
        part["reasoning"]
        for group in payload["history"]
        for event in group["events"]
        for part in event["parts"]
        if "reasoning" in part
    ]


@pytest.mark.parametrize("include_reasoning", [False, True])
def test_reasoning_enters_compactor_input_only_when_selected(include_reasoning: bool) -> None:
    async def scenario() -> None:
        coordinator = new_coordinator(include_reasoning=include_reasoning)
        await add_thought_group(coordinator, 1, "consider the parser " + SECRET)
        await add_thought_group(coordinator, 2, "r" * (9 * 1024))
        await add_group(coordinator, 3)
        plan = await coordinator.plan()
        header = json.loads(plan.document.split("\n", 1)[1])["checkpoint"]
        parts = reasoning_parts(plan.document)
        assert SECRET not in plan.document
        if not include_reasoning:
            assert parts == []
            assert REASONING_TRUNCATED_MEMBER not in header
            return
        assert parts[0] == "consider the parser [REDACTED]"
        assert parts[1].endswith(TRUNCATION_MARKER)
        assert tagged_size(parts[1]) <= MAX_REASONING_PART_BYTES
        assert header[REASONING_TRUNCATED_MEMBER] is True
        # Reasoning is never replayed to the main model.
        coordinator.commit(plan, "S1", coordinator.marker(plan, sdk_invocation_id="sdk-1"))
        projected = coordinator.project(
            [
                thought_call_event("call-1", "c1", "x"),
                reply_event("reply-1", "c1"),
                thought_call_event("call-2", "c2", "y"),
                reply_event("reply-2", "c2"),
                call_event("call-3", "c3"),
                reply_event("reply-3", "c3"),
            ]
        )
        assert projected is not None
        assert not any(part.thought for content in projected for part in content.parts or [])

    asyncio.run(scenario())


def test_compactor_input_keeps_newest_reasoning_within_its_total_bound() -> None:
    async def scenario() -> None:
        coordinator = new_coordinator(include_reasoning=True)
        for ordinal in range(1, 13):
            await add_thought_group(coordinator, ordinal, f"{ordinal:02d}" + "q" * 7000)
        await add_group(coordinator, 13)
        plan = await coordinator.plan()
        parts = reasoning_parts(plan.document)
        assert sum(tagged_size(part) for part in parts) <= MAX_REASONING_TOTAL_BYTES
        assert 0 < len(parts) < 12
        # The newest groups keep their reasoning; evidence of every group stays.
        assert parts[-1].startswith("12")
        assert plan.document.count('"functionCall"') == 12
        header = json.loads(plan.document.split("\n", 1)[1])["checkpoint"]
        assert header[REASONING_TRUNCATED_MEMBER] is True

    asyncio.run(scenario())


def test_reasoning_does_not_count_against_the_protected_tail() -> None:
    async def scenario() -> None:
        coordinator = new_coordinator(keep=16, include_reasoning=True)
        await add_group(coordinator, 1)
        for ordinal in range(2, 18):
            await add_thought_group(coordinator, ordinal, "t" * 8000)
        tail = coordinator._archive.groups[1:]  # type: ignore[union-attr]
        assert sum(group.size for group in tail) > MAX_RECENT_GROUPS_BYTES
        plan = await coordinator.plan()
        assert (plan.first_group, plan.last_group) == (1, 1)

    asyncio.run(scenario())
