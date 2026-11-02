from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from fakes.worker_state import WorkerStateStore

from contractor_runtime.adapters import RuntimeSpan, TelemetryAttribute
from contractor_runtime.core.tool_metrics import RecordedToolCall
from contractor_runtime.telemetry.invocations import InvocationMetricsReducer
from contractor_runtime.telemetry.metrics import MetricsState
from contractor_runtime.worker.instrumentation import (
    WorkerInstrumentationPlugin,
    span_error_type,
)

SECRET = "instrumentation-secret-canary"
PROMPT = "prompt-body-that-must-not-survive"
SOURCE = "source-body-that-must-not-survive"
PRIVATE_URL = f"https://user:{SECRET}@private.example/api?token={SECRET}"
ALLOWED_TOOL_NAMES = frozenset(
    {
        "probe",
        "generic_probe",
        "rejected_probe",
        "adk_validation_probe",
        "validator_failure_probe",
        "cancelled_probe",
        "pending_probe",
        "self_recorded_probe",
    }
)


class ToolFailure(RuntimeError):
    code = "synthetic_tool_failure"
    retryable = True


@dataclass
class FakeSession:
    state: dict[str, Any] = field(default_factory=dict)
    id: str = "session"
    app_name: str = "app"
    user_id: str = "user"


@dataclass
class FakeContext:
    invocation_id: str
    session: FakeSession = field(default_factory=FakeSession)
    function_call_id: str | None = None


class FakeOwner:
    def __init__(
        self, rejection: Exception | None = None, *, observation_cursor: int | None = None
    ):
        self._rejection = rejection
        if observation_cursor is not None:
            self.artifact_observation_cursor = observation_cursor

    def contractor_raw_argument_error(self, _arguments: dict[str, Any]) -> Exception | None:
        return self._rejection


@dataclass
class FakeTool:
    name: str
    func: FakeOwner


class RecordingInstrumentation:
    def __init__(self) -> None:
        self.spans: list[RecordingSpan] = []

    def start_span(
        self,
        name: str,
        *,
        attributes: dict[str, TelemetryAttribute] | None = None,
    ) -> RuntimeSpan:
        span = RecordingSpan(name, attributes or {})
        self.spans.append(span)
        return span


class RecordingSpan:
    def __init__(self, name: str, attributes: dict[str, TelemetryAttribute]) -> None:
        self.name = name
        self.attributes = dict(attributes)
        self.outcome: str | None = None
        self.end_calls = 0

    def end(
        self,
        *,
        outcome: str,
        attributes: dict[str, TelemetryAttribute] | None = None,
    ) -> None:
        self.end_calls += 1
        self.outcome = outcome
        if attributes is not None:
            self.attributes.update(attributes)


def test_plugin_correlates_every_callback_path_without_double_counting(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        instrumentation = RecordingInstrumentation()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            allowed_tool_names=ALLOWED_TOOL_NAMES,
            budget=lambda: None,
            instrumentation=instrumentation,
            model_alias="safe-model-alias",
            observe_artifacts=lambda _owner, _cursor: (_ for _ in ()).throw(
                RuntimeError("optional projection failed")
            ),
        )
        run_context = FakeContext("worker-callbacks")
        plugin.prepare_invocation(invocation_id=run_context.invocation_id, subtask_id="1")
        await plugin.before_run_callback(invocation_context=run_context)

        model_context = FakeContext(run_context.invocation_id)
        await plugin.before_model_callback(callback_context=model_context, llm_request=PROMPT)
        await plugin.after_model_callback(
            callback_context=model_context,
            llm_response=SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=11,
                    candidates_token_count=7,
                    total_token_count=18,
                    cached_content_token_count=3,
                )
            ),
        )
        await plugin.before_model_callback(callback_context=model_context, llm_request=PROMPT)
        await plugin.on_model_error_callback(
            callback_context=model_context,
            llm_request=PROMPT,
            error=ToolFailure(f"provider returned {SECRET}"),
        )

        async def direct_tool(
            context_name: str,
            *,
            result: dict[str, Any],
            error: Exception | None = None,
            record_twice: bool = False,
            observation_cursor: int | None = None,
        ) -> None:
            context = FakeContext(run_context.invocation_id)
            tool = FakeTool("probe", FakeOwner(observation_cursor=observation_cursor))
            arguments = {"url": PRIVATE_URL, "body": SOURCE, "same": context_name}
            assert (
                await plugin.before_tool_callback(
                    tool=tool,
                    tool_args=arguments,
                    tool_context=context,
                )
                is None
            )
            state.metrics.record_tool_call(
                tool.name,
                arguments=arguments,
                result=result,
                error=error,
                secrets=(SECRET, SOURCE),
            )
            if record_twice:
                state.metrics.record_tool_call(
                    tool.name,
                    arguments=arguments,
                    result=result,
                    error=error,
                    secrets=(SECRET, SOURCE),
                )
            await plugin.after_tool_callback(
                tool=tool,
                tool_args=arguments,
                tool_context=context,
                result=result,
            )

        await direct_tool("success", result={"ok": True}, record_twice=True)
        failure = ToolFailure(f"tool returned {SECRET}")
        await direct_tool("returned-error", result={"ok": False}, error=failure)

        raised_context = FakeContext(run_context.invocation_id)
        raised_tool = FakeTool("probe", FakeOwner())
        assert (
            await plugin.before_tool_callback(
                tool=raised_tool,
                tool_args={"body": SOURCE},
                tool_context=raised_context,
            )
            is None
        )
        state.metrics.record_tool_call(
            raised_tool.name,
            arguments={"body": SOURCE},
            error=failure,
            secrets=(SECRET, SOURCE),
        )
        safe_error = await plugin.on_tool_error_callback(
            tool=raised_tool,
            tool_args={"body": SOURCE},
            tool_context=raised_context,
            error=failure,
        )
        assert safe_error == {
            "ok": False,
            "error": {
                "code": "synthetic_tool_failure",
                "message": "probe failed (ToolFailure)",
                "retryable": True,
            },
        }

        generic_context = FakeContext(run_context.invocation_id)
        generic_tool = FakeTool("generic_probe", FakeOwner(observation_cursor=0))
        await plugin.before_tool_callback(
            tool=generic_tool,
            tool_args={"body": SOURCE},
            tool_context=generic_context,
        )
        await plugin.after_tool_callback(
            tool=generic_tool,
            tool_args={"body": SOURCE},
            tool_context=generic_context,
            result={"ok": True, "body": SOURCE},
        )
        assert plugin.projection_failed

        rejected_context = FakeContext(run_context.invocation_id)
        rejected_tool = FakeTool("rejected_probe", FakeOwner(failure))
        rejected = await plugin.before_tool_callback(
            tool=rejected_tool,
            tool_args={"body": SOURCE},
            tool_context=rejected_context,
        )
        assert rejected is not None and rejected["ok"] is False
        await plugin.after_tool_callback(
            tool=rejected_tool,
            tool_args={"body": SOURCE},
            tool_context=rejected_context,
            result=rejected,
        )

        validation_context = FakeContext(run_context.invocation_id)
        validation_tool = FakeTool("adk_validation_probe", FakeOwner())
        await plugin.before_tool_callback(
            tool=validation_tool,
            tool_args={"body": SOURCE},
            tool_context=validation_context,
        )
        await plugin.after_tool_callback(
            tool=validation_tool,
            tool_args={"body": SOURCE},
            tool_context=validation_context,
            result={"error": SOURCE},
        )

        exploding_owner = FakeOwner()

        def explode_validation(_arguments: dict[str, Any]) -> Exception | None:
            raise RuntimeError(f"validator leaked {SECRET}")

        exploding_owner.contractor_raw_argument_error = explode_validation  # type: ignore[method-assign]
        exploding_context = FakeContext(run_context.invocation_id)
        exploding_tool = FakeTool("validator_failure_probe", exploding_owner)
        validation_failure = await plugin.before_tool_callback(
            tool=exploding_tool,
            tool_args={"body": SOURCE},
            tool_context=exploding_context,
        )
        assert validation_failure is not None
        assert validation_failure["error"]["code"] == "tool_argument_validation_failed"
        await plugin.after_tool_callback(
            tool=exploding_tool,
            tool_args={"body": SOURCE},
            tool_context=exploding_context,
            result=validation_failure,
        )

        unknown_context = FakeContext(run_context.invocation_id)
        unknown_tool = FakeTool(SECRET, FakeOwner())
        await plugin.before_tool_callback(
            tool=unknown_tool, tool_args={"body": SOURCE}, tool_context=unknown_context
        )
        unknown_response = await plugin.on_tool_error_callback(
            tool=unknown_tool,
            tool_args={"body": SOURCE},
            tool_context=unknown_context,
            error=ToolFailure(f"unknown model tool {SECRET}"),
        )
        assert unknown_response["error"]["message"] == "unknown_tool failed (ToolFailure)"

        async def parallel_duplicate() -> None:
            await direct_tool("identical", result={"ok": True})

        await asyncio.gather(parallel_duplicate(), parallel_duplicate())

        cancelled_context = FakeContext(run_context.invocation_id)
        cancelled_tool = FakeTool("cancelled_probe", FakeOwner())
        await plugin.before_tool_callback(
            tool=cancelled_tool,
            tool_args={"body": SOURCE},
            tool_context=cancelled_context,
        )
        completed = await plugin.complete_invocation(
            invocation_id=run_context.invocation_id,
            phase="cancelled",
        )
        assert completed is not None

        report = state.metrics.build_report(report_id="worker-report", duration_ms=1)
        assert report.metrics.model_calls == 2
        assert report.metrics.input_tokens == 11
        assert report.metrics.output_tokens == 7
        assert report.metrics.total_tokens == 18
        assert report.metrics.tools["probe"].calls == 5
        assert report.metrics.tools["probe"].failed == 2
        assert report.metrics.tools["generic_probe"].calls == 1
        assert report.metrics.tools["rejected_probe"].failed == 1
        assert report.metrics.tools["adk_validation_probe"].failed == 1
        assert report.metrics.tools["validator_failure_probe"].failed == 1
        assert report.metrics.tools["unknown_tool"].failed == 1
        assert report.metrics.tools["cancelled_probe"].failed == 1
        assert sum(tool.calls for tool in report.metrics.tools.values()) == 11

        last = completed["lastCompletedInvocation"]
        assert last["subtaskId"] == "1"
        assert last["phase"] == "cancelled"
        assert last["metrics"]["modelCalls"] == 2
        assert last["metrics"]["modelErrors"] == 1
        assert last["metrics"]["toolCalls"] == 11
        assert last["metrics"]["toolErrors"] == 7

        serialized = json.dumps(
            {
                "state": completed,
                "report": report.model_dump(mode="json", by_alias=True, exclude_none=True),
                "telemetry": [span.attributes for span in instrumentation.spans],
                "logs": caplog.text,
            }
        )
        for canary in (SECRET, PROMPT, SOURCE, PRIVATE_URL):
            assert canary not in serialized
        assert len(instrumentation.spans) == 13
        assert all(span.outcome is not None for span in instrumentation.spans)

        await plugin.close()
        await plugin.close()

    asyncio.run(scenario())


class UnclassifiedRetryable(RuntimeError):
    retryable = True


class TextCode(RuntimeError):
    code = "provider said no"


class Classified(RuntimeError):
    code = "classified_failure"
    retryable = False


@pytest.mark.parametrize(
    ("error", "code", "retryable"),
    [
        (TypeError(f"defect {SECRET}"), "tool_call_failed", False),
        (UnclassifiedRetryable(f"defect {SECRET}"), "tool_call_failed", True),
        (TextCode(f"defect {SECRET}"), "tool_call_failed", False),
        (Classified(f"defect {SECRET}"), "classified_failure", False),
    ],
    ids=["unclassified", "retryable-without-code", "non-identifier-code", "classified"],
)
def test_raised_tool_exception_is_a_bounded_failed_tool_result(
    error: Exception, code: str, retryable: bool
) -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            allowed_tool_names=ALLOWED_TOOL_NAMES,
            budget=lambda: None,
            instrumentation=RecordingInstrumentation(),
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
        )
        context = FakeContext("worker-raised-tool")
        plugin.prepare_invocation(invocation_id=context.invocation_id, subtask_id="1")
        await plugin.before_run_callback(invocation_context=context)
        tool = FakeTool("probe", FakeOwner())
        tool_context = FakeContext(context.invocation_id)
        assert (
            await plugin.before_tool_callback(
                tool=tool, tool_args={"body": SOURCE}, tool_context=tool_context
            )
            is None
        )
        with pytest.raises(type(error)), RecordedToolCall(state.metrics, "probe", {}):
            raise error

        response = await plugin.on_tool_error_callback(
            tool=tool, tool_args={"body": SOURCE}, tool_context=tool_context, error=error
        )

        class_name = type(error).__name__
        assert response == {
            "ok": False,
            "error": {
                "code": code,
                "message": f"probe failed ({class_name})",
                "retryable": retryable,
            },
        }
        report = state.metrics.build_report(report_id="worker-raised-tool", duration_ms=1)
        assert len(report.tool_calls) == 1
        recorded = report.tool_calls[0].error
        assert recorded is not None
        assert (recorded.code, recorded.message, recorded.retryable) == (
            code,
            f"Tool probe failed ({class_name})",
            retryable,
        )
        assert SECRET not in json.dumps(response) + report.model_dump_json()

    asyncio.run(scenario())


def test_plugin_close_records_pending_tool_calls_as_cancelled() -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        instrumentation = RecordingInstrumentation()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            allowed_tool_names=ALLOWED_TOOL_NAMES,
            budget=lambda: None,
            instrumentation=instrumentation,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
        )
        context = FakeContext("worker-close")
        plugin.prepare_invocation(invocation_id=context.invocation_id, subtask_id="1")
        await plugin.before_run_callback(invocation_context=context)
        # The plugin keys a pending call by its live tool context, as ADK keeps
        # it alive for the whole call; a freed context could reuse an id.
        tool_contexts = [FakeContext(context.invocation_id) for _ in range(2)]
        for name, tool_context in zip(
            ("pending_probe", "self_recorded_probe"), tool_contexts, strict=True
        ):
            pending = await plugin.before_tool_callback(
                tool=FakeTool(name, FakeOwner()),
                tool_args={"body": SOURCE},
                tool_context=tool_context,
            )
            assert pending is None
        # A toolset that records its own cancellation is not counted twice.
        state.metrics.record_tool_call(
            "self_recorded_probe", arguments={}, error=ToolFailure("cancelled")
        )

        await plugin.before_result_finalizer_call(invocation_id=context.invocation_id)
        await plugin.close()
        await plugin.close()

        report = state.metrics.build_report(report_id="worker-close", duration_ms=1)
        assert {name: tool.failed for name, tool in report.metrics.tools.items()} == {
            "pending_probe": 1,
            "self_recorded_probe": 1,
        }
        cancelled = next(call for call in report.tool_calls if call.tool == "pending_probe")
        assert cancelled.error is not None
        assert cancelled.error.code == "tool_call_cancelled"
        assert cancelled.error.retryable is True
        assert len(report.tool_calls) == 2
        assert [span.outcome for span in instrumentation.spans] == ["cancelled"] * 3
        assert all(span.end_calls == 1 for span in instrumentation.spans)

    asyncio.run(scenario())


def test_plugin_retains_only_latest_sequential_invocation() -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            allowed_tool_names=ALLOWED_TOOL_NAMES,
            budget=lambda: None,
            instrumentation=None,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
        )
        revisions: list[int] = []
        for index, subtask_id in enumerate(("0", "1")):
            invocation_id = f"worker-sequential-{index}"
            context = FakeContext(invocation_id)
            plugin.prepare_invocation(invocation_id=invocation_id, subtask_id=subtask_id)
            await plugin.before_run_callback(invocation_context=context)
            current = await state.snapshot()
            revisions.append(current["stateRevision"])
            assert current["currentInvocation"]["subtaskId"] == subtask_id
            await plugin.before_model_callback(callback_context=context, llm_request=PROMPT)
            await plugin.after_model_callback(
                callback_context=context,
                llm_response=SimpleNamespace(usage_metadata=None),
            )
            completed = await plugin.complete_invocation(
                invocation_id=invocation_id,
                phase="succeeded",
            )
            assert completed is not None
            revisions.append(completed["stateRevision"])
            assert completed["currentInvocation"] is None
            assert completed["lastCompletedInvocation"]["subtaskId"] == subtask_id

        assert revisions == sorted(set(revisions))
        snapshot = await state.snapshot()
        assert snapshot["metrics"]["counters"]["llm_calls"] == 2
        assert snapshot["lastCompletedInvocation"]["subtaskId"] == "1"
        assert await state.encoded_snapshot_size() <= 4 * 1024 * 1024

    asyncio.run(scenario())


class ProviderTyped(RuntimeError):
    def __init__(self, provider_error_type: object) -> None:
        super().__init__(f"provider failure {SECRET}")
        self.provider_error_type = provider_error_type


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ProviderTyped("RateLimitError"), "RateLimitError"),
        (ProviderTyped("a" * 128), "a" * 128),
        (ProviderTyped("a" * 129), "ProviderTyped"),
        (ProviderTyped(f"provider said {SECRET}"), "ProviderTyped"),
        (ProviderTyped(".leading-dot"), "ProviderTyped"),
        (ProviderTyped(429), "ProviderTyped"),
        (TimeoutError(SECRET), "TimeoutError"),
    ],
    ids=["safe", "longest", "too-long", "text", "leading-dot", "not-text", "no-provider-type"],
)
def test_span_error_type_admits_only_safe_provider_identifiers(
    error: Exception, expected: str
) -> None:
    assert span_error_type(error) == expected


@pytest.mark.parametrize(
    ("name", "identifier"),
    [
        ("read_source_file", "read_source_file"),
        ("Read-File.v2", "read_file_v2"),
        ("  __Mixed Case__  ", "mixed_case"),
        ("x" * 80, "x" * 64),
        ("!!!", "unknown"),
    ],
)
def test_invocation_and_allocation_metrics_key_a_tool_by_one_identifier(
    name: str, identifier: str
) -> None:
    invocation = InvocationMetricsReducer()
    invocation.record_tool_call(name, failed=False)
    allocation = MetricsState()
    allocation.record_tool_call(name, arguments={})
    report = allocation.build_report(report_id="worker-identifiers", duration_ms=1)

    assert list(invocation.snapshot()["tools"]) == [identifier]
    assert list(report.metrics.tools) == [identifier]


@pytest.mark.parametrize("phase", ["succeeded", "failed", "cancelled"])
@pytest.mark.parametrize("auxiliary_completed", [False, True])
def test_terminal_phase_survives_auxiliary_cleanup(phase: str, auxiliary_completed: bool) -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        sink = RecordingInstrumentation()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            allowed_tool_names=ALLOWED_TOOL_NAMES,
            budget=lambda: None,
            instrumentation=sink,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
        )
        context = FakeContext("worker-auxiliary")
        plugin.prepare_invocation(invocation_id=context.invocation_id, subtask_id="0")
        await plugin.before_run_callback(invocation_context=context)
        await plugin.before_result_finalizer_call(invocation_id=context.invocation_id)
        if auxiliary_completed:
            await plugin.after_result_finalizer_call(
                invocation_id=context.invocation_id, usage=None
            )
        snapshot = await plugin.complete_invocation(
            invocation_id=context.invocation_id, phase=phase
        )
        assert snapshot["currentInvocation"] is None
        assert snapshot["lastCompletedInvocation"]["phase"] == phase
        assert snapshot["lastCompletedInvocation"]["metrics"]["modelCalls"] == 1
        assert snapshot["lastCompletedInvocation"]["metrics"]["modelErrors"] == 0
        await plugin.on_result_finalizer_error(
            invocation_id=context.invocation_id, error=asyncio.CancelledError()
        )
        await plugin.close()
        await plugin.close()
        assert len(sink.spans) == 1
        assert sink.spans[0].end_calls == 1
        assert sink.spans[0].outcome == ("succeeded" if auxiliary_completed else "cancelled")

    asyncio.run(scenario())


def test_finalizer_startup_owns_partial_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        sink = RecordingInstrumentation()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            allowed_tool_names=ALLOWED_TOOL_NAMES,
            budget=lambda: None,
            instrumentation=sink,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
        )
        context = FakeContext("worker-startup")
        plugin.prepare_invocation(invocation_id=context.invocation_id, subtask_id="0")
        await plugin.before_run_callback(invocation_context=context)
        publish = state.publish_invocation_metrics
        error = asyncio.CancelledError()

        async def reject_publication(**_kwargs: Any) -> None:
            raise error

        monkeypatch.setattr(state, "publish_invocation_metrics", reject_publication)
        with pytest.raises(type(error)) as raised:
            await plugin.before_result_finalizer_call(invocation_id=context.invocation_id)
        assert raised.value is error
        # No caller received successful before-call ownership: cleanup must already be done.
        assert plugin._pending_auxiliary_models == {}
        assert sink.spans[0].end_calls == 1
        assert sink.spans[0].outcome == "cancelled"
        monkeypatch.setattr(state, "publish_invocation_metrics", publish)
        await plugin.on_result_finalizer_error(invocation_id=context.invocation_id, error=error)
        snapshot = await plugin.complete_invocation(
            invocation_id=context.invocation_id, phase="cancelled"
        )
        assert snapshot["lastCompletedInvocation"]["metrics"]["modelErrors"] == 0
        await plugin.close()
        assert sink.spans[0].end_calls == 1

    asyncio.run(scenario())


class CountingBudget:
    def __init__(self, *, summarize: bool = False) -> None:
        self.model_calls = 0
        self.tool_calls = 0
        self.usage: list[Any] = []
        self.recorded_prompts: list[bool] = []
        self.summarize_checks = 0
        self.summarize = summarize

    def before_model_call(self) -> None:
        self.model_calls += 1

    def before_tool_call(self) -> None:
        self.tool_calls += 1

    def after_model_response(self, usage: Any, *, record_prompt: bool = True) -> None:
        self.usage.append(usage)
        self.recorded_prompts.append(record_prompt)

    def should_summarize(self) -> bool:
        self.summarize_checks += 1
        return self.summarize


class RecordingHistory:
    def __init__(self) -> None:
        self.plugin: WorkerInstrumentationPlugin | None = None
        self.events: list[Any] = []
        self.admissions: list[FakeContext] = []
        self.completions: list[tuple[FakeContext, Any]] = []
        self.failures: list[tuple[FakeContext, Any, bool]] = []

    def observe_event(self, event: Any) -> None:
        self.events.append(event)

    def admitted(self, context: FakeContext) -> None:
        self.admissions.append(context)

    async def completed(self, context: FakeContext, result: Any) -> None:
        assert self.plugin is not None and not self.plugin._lock.locked()
        self.completions.append((context, result))

    async def failed(self, context: FakeContext, result: Any, *, cancelled: bool = False) -> None:
        assert self.plugin is not None and not self.plugin._lock.locked()
        self.failures.append((context, result, cancelled))


def test_sdk_turns_keep_one_logical_state_budget_and_tool_identity() -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        budget = CountingBudget()
        history = RecordingHistory()
        sink = RecordingInstrumentation()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            budget=lambda: budget,
            instrumentation=sink,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
            allowed_tool_names=frozenset({"probe"}),
            history=history,
        )
        history.plugin = plugin
        logical = "worker-logical"
        first = FakeContext(logical)
        plugin.prepare_invocation(invocation_id=logical, subtask_id="0")
        await plugin.before_run_callback(invocation_context=first)
        tool = FakeTool("probe", FakeOwner())
        for context in (first, FakeContext("sdk-second", session=first.session)):
            if context is not first:
                plugin.prepare_continuation(invocation_id=logical, sdk_invocation_id="sdk-second")
                with pytest.raises(RuntimeError, match="stale"):
                    await plugin.before_model_callback(callback_context=first, llm_request=PROMPT)
                await plugin.before_run_callback(invocation_context=context)
            await plugin.before_model_callback(callback_context=context, llm_request=PROMPT)
            await plugin.after_model_callback(
                callback_context=context,
                llm_response=SimpleNamespace(usage_metadata=None),
            )
            event = object()
            await plugin.on_event_callback(invocation_context=context, event=event)
            assert history.events[-1] is event
            tool_context = FakeContext(
                context.invocation_id,
                session=context.session,
                function_call_id=f"call-{budget.tool_calls}",
            )
            assert plugin.logical_invocation_id_for_tool(tool_context) == logical
            assert (
                await plugin.before_tool_callback(
                    tool=tool, tool_args={}, tool_context=tool_context
                )
                is None
            )
            pending = plugin._pending_tools[id(tool_context)]
            assert pending.invocation_id == logical
            assert pending.correlation_id == f"{logical}:tool:{budget.tool_calls}"
            result = {"ok": True, "exact": "result"}
            await plugin.after_tool_callback(
                tool=tool, tool_args={}, tool_context=tool_context, result=result
            )
            assert history.completions[-1] == (tool_context, result)
            assert history.completions[-1][1] is result
        with pytest.raises(RuntimeError, match="cannot continue"):
            plugin.prepare_continuation(invocation_id=logical, sdk_invocation_id=logical)
        await plugin.before_result_finalizer_call(invocation_id=logical)
        await plugin.after_result_finalizer_call(invocation_id=logical, usage=None)
        completed = await plugin.complete_invocation(invocation_id=logical, phase="succeeded")
        assert completed is not None
        assert completed["lastCompletedInvocation"]["invocationId"] == logical
        assert completed["lastCompletedInvocation"]["metrics"]["modelCalls"] == 3
        assert completed["lastCompletedInvocation"]["metrics"]["toolCalls"] == 2
        assert (budget.model_calls, budget.tool_calls) == (3, 2)
        assert len(history.admissions) == len(history.completions) == 2
        assert all(span.end_calls == 1 for span in sink.spans)
        assert "sdk-second" not in json.dumps(completed)
        with pytest.raises(RuntimeError, match="stale"):
            plugin.logical_invocation_id_for_tool(first)

    asyncio.run(scenario())


@pytest.mark.parametrize("callback", ["model", "tool"])
@pytest.mark.parametrize("foreign", ["sdk", "session", "app", "user"])
def test_stale_or_foreign_native_admission_has_no_charge_or_span(
    callback: str, foreign: str
) -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        budget = CountingBudget()
        sink = RecordingInstrumentation()
        admissions: list[bool] = []
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            budget=lambda: budget,
            instrumentation=sink,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
            allowed_tool_names=frozenset({"probe"}),
            check_admission=lambda: admissions.append(True),
        )
        initial = FakeContext("worker-logical")
        plugin.prepare_invocation(invocation_id=initial.invocation_id, subtask_id="0")
        await plugin.before_run_callback(invocation_context=initial)
        context = FakeContext(initial.invocation_id)
        if foreign == "sdk":
            context.invocation_id = "unprepared-sdk"
        elif foreign == "session":
            context.session.id = "foreign-session"
        elif foreign == "app":
            context.session.app_name = "foreign-app"
        else:
            context.session.user_id = "foreign-user"
        with pytest.raises(RuntimeError, match="stale"):
            if callback == "model":
                await plugin.before_model_callback(callback_context=context, llm_request=PROMPT)
            else:
                await plugin.before_tool_callback(
                    tool=FakeTool("probe", FakeOwner()), tool_args={}, tool_context=context
                )
        assert (budget.model_calls, budget.tool_calls) == (0, 0)
        assert admissions == []
        assert sink.spans == []
        assert plugin._pending_tools == {}

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["model", "tool", "finalizer"])
@pytest.mark.parametrize("error", [TimeoutError("expired"), asyncio.CancelledError()])
def test_semantic_admission_precedes_summary_budget_and_span(
    phase: str, error: BaseException
) -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        budget = CountingBudget(summarize=True)
        sink = RecordingInstrumentation()

        def refuse() -> None:
            raise error

        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            budget=lambda: budget,
            instrumentation=sink,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
            allowed_tool_names=frozenset({"probe"}),
            check_admission=refuse,
            summarizer_enabled=True,
        )
        context = FakeContext("worker-admission")
        plugin.prepare_invocation(invocation_id=context.invocation_id, subtask_id="0")
        await plugin.before_run_callback(invocation_context=context)
        with pytest.raises(type(error)) as raised:
            if phase == "model":
                await plugin.before_model_callback(callback_context=context, llm_request=PROMPT)
            elif phase == "tool":
                await plugin.before_tool_callback(
                    tool=FakeTool("probe", FakeOwner()), tool_args={}, tool_context=context
                )
            else:
                await plugin.before_result_finalizer_call(invocation_id=context.invocation_id)
        assert raised.value is error
        assert (budget.model_calls, budget.tool_calls, budget.summarize_checks) == (0, 0, 0)
        assert sink.spans == []
        assert plugin._pending_models == []
        assert plugin._pending_tools == {}
        assert plugin._pending_auxiliary_models == {}
        assert (await state.snapshot())["currentInvocation"]["phase"] == "running"

    asyncio.run(scenario())


@pytest.mark.parametrize("self_recorded", [False, True])
def test_cancelled_tool_error_is_accounted_once_and_releases_correlation(
    self_recorded: bool,
) -> None:
    from contractor_runtime.core.tool_metrics import ToolCallCancelled
    from contractor_runtime.telemetry.metrics import _TOOL_METRIC_CORRELATION

    async def scenario() -> None:
        state = WorkerStateStore()
        budget = CountingBudget()
        history = RecordingHistory()
        sink = RecordingInstrumentation()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            budget=lambda: budget,
            instrumentation=sink,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
            allowed_tool_names=frozenset({"probe"}),
            history=history,
        )
        history.plugin = plugin
        context = FakeContext("worker-cancel")
        plugin.prepare_invocation(invocation_id=context.invocation_id, subtask_id="0")
        await plugin.before_run_callback(invocation_context=context)
        tool = FakeTool("probe", FakeOwner())
        await plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=context)
        assert _TOOL_METRIC_CORRELATION.get() == "worker-cancel:tool:1"
        if self_recorded:
            state.metrics.record_tool_call("probe", arguments={}, error=ToolCallCancelled())
        error = asyncio.CancelledError(SECRET)
        with pytest.raises(asyncio.CancelledError) as raised:
            await plugin.on_tool_error_callback(
                tool=tool, tool_args={}, tool_context=context, error=error
            )
        assert raised.value is error
        assert _TOOL_METRIC_CORRELATION.get() is None
        assert len(history.failures) == 1
        assert history.failures[0][2] is True
        assert history.failures[0][1]["error"]["code"] == "tool_call_cancelled"
        assert history.failures[0][1]["error"]["retryable"] is True
        assert SECRET not in json.dumps(history.failures[0][1])
        await plugin.after_tool_callback(tool=tool, tool_args={}, tool_context=context, result={})
        with pytest.raises(asyncio.CancelledError):
            await plugin.on_tool_error_callback(
                tool=tool, tool_args={}, tool_context=context, error=error
            )
        report = state.metrics.build_report(report_id="cancelled", duration_ms=1)
        assert report.metrics.tools["probe"].calls == 1
        assert report.metrics.tools["probe"].failed == 1
        assert len(report.tool_calls) == 1
        assert report.tool_calls[0].error.code == "tool_call_cancelled"
        assert budget.tool_calls == 1
        assert sink.spans[0].outcome == "cancelled"
        assert sink.spans[0].end_calls == 1
        assert plugin._pending_tools == {}
        assert plugin._settling_tools == set()
        await plugin.complete_invocation(invocation_id=context.invocation_id, phase="cancelled")
        await plugin.close()
        assert sink.spans[0].end_calls == 1

    asyncio.run(scenario())


def test_history_serialization_releases_plugin_lock_and_fences_premature_completion() -> None:
    class BlockingHistory(RecordingHistory):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.resume = asyncio.Event()

        async def completed(self, context: FakeContext, result: Any) -> None:
            assert self.plugin is not None and not self.plugin._lock.locked()
            self.started.set()
            await self.resume.wait()
            await super().completed(context, result)

    async def scenario() -> None:
        state = WorkerStateStore()
        history = BlockingHistory()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            budget=lambda: None,
            instrumentation=None,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
            allowed_tool_names=frozenset({"probe"}),
            history=history,
        )
        history.plugin = plugin
        context = FakeContext("worker-history")
        plugin.prepare_invocation(invocation_id=context.invocation_id, subtask_id="0")
        await plugin.before_run_callback(invocation_context=context)
        tool = FakeTool("probe", FakeOwner())
        await plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=context)
        completed = asyncio.create_task(
            plugin.after_tool_callback(
                tool=tool, tool_args={}, tool_context=context, result={"ok": True}
            )
        )
        try:
            await asyncio.wait_for(history.started.wait(), 3)
            await asyncio.wait_for(
                plugin.capture_result_finalizer_content(invocation_id=context.invocation_id), 3
            )
            with pytest.raises(RuntimeError, match="cannot continue"):
                plugin.prepare_continuation(
                    invocation_id=context.invocation_id, sdk_invocation_id="sdk-next"
                )
            with pytest.raises(RuntimeError, match="still settling"):
                await plugin.complete_invocation(
                    invocation_id=context.invocation_id, phase="succeeded"
                )
            with pytest.raises(RuntimeError, match="still settling"):
                await plugin.close()
        finally:
            history.resume.set()
            await completed
        assert len(history.completions) == 1
        assert plugin._settling_tools == set()
        final = await plugin.complete_invocation(
            invocation_id=context.invocation_id, phase="succeeded"
        )
        assert final["lastCompletedInvocation"]["metrics"]["toolCalls"] == 1
        await plugin.close()

    asyncio.run(scenario())


def test_after_model_assigns_owned_ids_only_to_missing_function_call_ids() -> None:
    from google.genai import types

    async def scenario() -> None:
        state = WorkerStateStore()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            budget=lambda: None,
            instrumentation=None,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
            allowed_tool_names=frozenset({"probe"}),
        )
        context = FakeContext("worker-call-ids")
        plugin.prepare_invocation(invocation_id=context.invocation_id, subtask_id="0")
        await plugin.before_run_callback(invocation_context=context)
        calls = [types.FunctionCall(name="probe", args={}) for _ in range(2)]
        calls.append(types.FunctionCall(id="provider-id", name="probe", args={}))
        response = SimpleNamespace(
            content=types.Content(role="model", parts=[types.Part(function_call=c) for c in calls]),
            usage_metadata=None,
        )
        await plugin.before_model_callback(callback_context=context, llm_request=PROMPT)
        await plugin.after_model_callback(callback_context=context, llm_response=response)
        actual = [part.function_call.id for part in response.content.parts]
        assert all(identifier.startswith("contractor-tool-") for identifier in actual[:2])
        assert actual[0] != actual[1]
        assert actual[2] == "provider-id"
        assert all(not identifier.startswith("adk-") for identifier in actual)

    asyncio.run(scenario())


def test_unknown_tool_before_callback_is_one_bounded_attempt_without_content() -> None:
    class CapturedSpan(RecordingSpan):
        capture_content = True

        def __init__(self, name: str, attributes: dict[str, TelemetryAttribute]) -> None:
            super().__init__(name, attributes)
            self.content: dict[str, str] = {}

        def set_content(self, kind: str, value: str) -> None:
            self.content[kind] = value

    class CapturingInstrumentation(RecordingInstrumentation):
        def start_span(self, name: str, *, attributes=None) -> CapturedSpan:
            span = CapturedSpan(name, attributes or {})
            self.spans.append(span)
            return span

    async def scenario() -> None:
        state = WorkerStateStore()
        budget = CountingBudget()
        sink = CapturingInstrumentation()
        history = RecordingHistory()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            budget=lambda: budget,
            instrumentation=sink,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
            allowed_tool_names=frozenset({"probe"}),
            history=history,
        )
        history.plugin = plugin
        context = FakeContext("worker-unknown")
        plugin.prepare_invocation(invocation_id=context.invocation_id, subtask_id="0")
        await plugin.before_run_callback(invocation_context=context)
        owner = FakeOwner()

        def unexpected_validation(_args: Any) -> None:
            raise AssertionError("unknown placeholder ran a callable validator")

        owner.contractor_raw_argument_error = unexpected_validation  # type: ignore[method-assign]
        tool = FakeTool(SECRET, owner)
        assert (
            await plugin.before_tool_callback(
                tool=tool, tool_args={"body": SOURCE}, tool_context=context
            )
            is None
        )
        response = await plugin.on_tool_error_callback(
            tool=tool, tool_args={}, tool_context=context, error=ValueError(SECRET)
        )
        assert budget.tool_calls == 1
        assert len(history.admissions) == len(history.failures) == 1
        report = state.metrics.build_report(report_id="unknown", duration_ms=1)
        assert set(report.metrics.tools) == {"unknown_tool"}
        assert report.metrics.tools["unknown_tool"].calls == 1
        assert report.metrics.tools["unknown_tool"].failed == 1
        assert sink.spans[0].attributes["tool.name"] == "unknown_tool"
        assert sink.spans[0].content == {}
        rendered = (
            json.dumps(response) + report.model_dump_json() + json.dumps(await state.snapshot())
        )
        assert SECRET not in rendered
        assert SOURCE not in rendered
        assert sink.spans[0].end_calls == 1

    asyncio.run(scenario())


def test_spent_callbacks_settle_after_admission_deadline_without_new_charges() -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        budget = CountingBudget()
        sink = RecordingInstrumentation()
        expired = False
        checks = 0

        def check() -> None:
            nonlocal checks
            checks += 1
            if expired:
                raise TimeoutError("semantic deadline passed")

        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            budget=lambda: budget,
            instrumentation=sink,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
            allowed_tool_names=frozenset({"probe"}),
            check_admission=check,
        )
        context = FakeContext("worker-spent")
        plugin.prepare_invocation(invocation_id=context.invocation_id, subtask_id="0")
        await plugin.before_run_callback(invocation_context=context)
        await plugin.before_model_callback(callback_context=context, llm_request=PROMPT)
        expired = True
        await plugin.after_model_callback(
            callback_context=context, llm_response=SimpleNamespace(usage_metadata=None)
        )
        expired = False
        tool = FakeTool("probe", FakeOwner())
        await plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=context)
        expired = True
        await plugin.after_tool_callback(
            tool=tool, tool_args={}, tool_context=context, result={"ok": True}
        )
        expired = False
        await plugin.before_result_finalizer_call(invocation_id=context.invocation_id)
        expired = True
        await plugin.after_result_finalizer_call(invocation_id=context.invocation_id, usage=None)
        await plugin.complete_invocation(invocation_id=context.invocation_id, phase="failed")
        await plugin.close()
        assert checks == 3
        assert (budget.model_calls, budget.tool_calls) == (2, 1)
        assert len(budget.usage) == 2
        # Finalizer usage is charged but never replaces the main prompt measurement.
        assert budget.recorded_prompts == [True, False]
        assert all(span.end_calls == 1 and span.outcome == "succeeded" for span in sink.spans)

    asyncio.run(scenario())


@pytest.mark.parametrize("cleanup_error", [RuntimeError(SECRET), asyncio.CancelledError()])
def test_cancelled_tool_keeps_original_cancellation_when_history_cleanup_fails(
    cleanup_error: BaseException,
) -> None:
    class FailedHistory(RecordingHistory):
        async def failed(
            self, context: FakeContext, result: Any, *, cancelled: bool = False
        ) -> None:
            await super().failed(context, result, cancelled=cancelled)
            raise cleanup_error

    async def scenario() -> None:
        state = WorkerStateStore()
        history = FailedHistory()
        sink = RecordingInstrumentation()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            budget=lambda: None,
            instrumentation=sink,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
            allowed_tool_names=frozenset({"probe"}),
            history=history,
        )
        history.plugin = plugin
        context = FakeContext("worker-cancel-history")
        plugin.prepare_invocation(invocation_id=context.invocation_id, subtask_id="0")
        await plugin.before_run_callback(invocation_context=context)
        tool = FakeTool("probe", FakeOwner())
        await plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=context)
        original = asyncio.CancelledError("original")
        with pytest.raises(asyncio.CancelledError) as raised:
            await plugin.on_tool_error_callback(
                tool=tool, tool_args={}, tool_context=context, error=original
            )
        assert raised.value is original
        assert sink.spans[0].outcome == "cancelled"
        assert sink.spans[0].end_calls == 1
        assert plugin._pending_tools == {}
        assert plugin._settling_tools == set()
        report = state.metrics.build_report(report_id="cancel-history", duration_ms=1)
        assert report.metrics.tools["probe"].calls == 1
        assert SECRET not in report.model_dump_json()

    asyncio.run(scenario())


@pytest.mark.parametrize("callback", ["after_model", "model_error", "after_tool"])
@pytest.mark.parametrize("foreign", ["sdk", "session", "app", "user"])
def test_foreign_completion_callback_cannot_settle_an_owned_call_or_accept_text(
    callback: str, foreign: str
) -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        budget = CountingBudget()
        history = RecordingHistory()
        sink = RecordingInstrumentation()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            budget=lambda: budget,
            instrumentation=sink,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
            allowed_tool_names=frozenset({"probe"}),
            history=history,
        )
        history.plugin = plugin
        owned = FakeContext("worker-owned-completion")
        plugin.prepare_invocation(invocation_id=owned.invocation_id, subtask_id="0")
        await plugin.before_run_callback(invocation_context=owned)
        tool = FakeTool("probe", FakeOwner())
        if callback == "after_tool":
            await plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=owned)
        else:
            await plugin.before_model_callback(callback_context=owned, llm_request=PROMPT)
        outsider = FakeContext(owned.invocation_id)
        if foreign == "sdk":
            outsider.invocation_id = "unprepared-sdk"
        elif foreign == "session":
            outsider.session.id = "foreign-session"
        elif foreign == "app":
            outsider.session.app_name = "foreign-app"
        else:
            outsider.session.user_id = "foreign-user"
        response = SimpleNamespace(
            content=SimpleNamespace(parts=[SimpleNamespace(text=SECRET)]),
            usage_metadata=SimpleNamespace(
                prompt_token_count=11, candidates_token_count=7, total_token_count=18
            ),
        )
        with pytest.raises(RuntimeError, match="stale"):
            if callback == "after_model":
                await plugin.after_model_callback(callback_context=outsider, llm_response=response)
            elif callback == "model_error":
                await plugin.on_model_error_callback(
                    callback_context=outsider, llm_request=PROMPT, error=ToolFailure(SECRET)
                )
            else:
                await plugin.after_tool_callback(
                    tool=tool, tool_args={}, tool_context=outsider, result={"text": SECRET}
                )
        assert budget.usage == []
        assert history.completions == []
        assert sink.spans[0].end_calls == 0
        assert SECRET not in json.dumps(await state.snapshot())
        if callback == "after_tool":
            assert id(owned) in plugin._pending_tools
            await plugin.after_tool_callback(
                tool=tool, tool_args={}, tool_context=owned, result={"ok": True}
            )
            await plugin.after_tool_callback(
                tool=tool, tool_args={}, tool_context=owned, result={"ok": True}
            )
            assert len(history.completions) == 1
        elif callback == "after_model":
            assert len(plugin._pending_models) == 1
            await plugin.after_model_callback(callback_context=owned, llm_response=response)
            await plugin.after_model_callback(callback_context=owned, llm_response=response)
            assert len(budget.usage) == 1
        else:
            assert len(plugin._pending_models) == 1
            await plugin.on_model_error_callback(
                callback_context=owned, llm_request=PROMPT, error=ToolFailure(SECRET)
            )
            await plugin.on_model_error_callback(
                callback_context=owned, llm_request=PROMPT, error=ToolFailure(SECRET)
            )
        assert sink.spans[0].end_calls == 1
        assert (budget.model_calls, budget.tool_calls) == (
            (0, 1) if callback == "after_tool" else (1, 0)
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("callback", ["after_model", "model_error", "after_tool"])
def test_completed_invocation_rejects_late_native_completion(callback: str) -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            budget=lambda: None,
            instrumentation=None,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
            allowed_tool_names=frozenset({"probe"}),
        )
        context = FakeContext("worker-completed")
        plugin.prepare_invocation(invocation_id=context.invocation_id, subtask_id="0")
        await plugin.before_run_callback(invocation_context=context)
        await plugin.complete_invocation(invocation_id=context.invocation_id, phase="succeeded")
        settled = await state.snapshot()
        with pytest.raises(RuntimeError, match="stale"):
            if callback == "after_model":
                await plugin.after_model_callback(
                    callback_context=context, llm_response=SimpleNamespace(usage_metadata=None)
                )
            elif callback == "model_error":
                await plugin.on_model_error_callback(
                    callback_context=context, llm_request=PROMPT, error=ToolFailure(SECRET)
                )
            else:
                await plugin.after_tool_callback(
                    tool=FakeTool("probe", FakeOwner()),
                    tool_args={},
                    tool_context=context,
                    result={},
                )
        assert await state.snapshot() == settled

    asyncio.run(scenario())


def test_rejected_run_callback_cannot_clear_owned_output_limit_evidence() -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            budget=lambda: None,
            instrumentation=None,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
            allowed_tool_names=frozenset(),
        )
        context = FakeContext("worker-limited")
        plugin.prepare_invocation(invocation_id=context.invocation_id, subtask_id="0")
        await plugin.before_run_callback(invocation_context=context)
        await plugin.before_model_callback(callback_context=context, llm_request=PROMPT)
        await plugin.after_model_callback(
            callback_context=context,
            llm_response=SimpleNamespace(usage_metadata=None, finish_reason="MAX_TOKENS"),
        )
        assert plugin.model_output_limited
        with pytest.raises(RuntimeError, match="not prepared"):
            await plugin.before_run_callback(invocation_context=FakeContext("foreign"))
        assert plugin.model_output_limited
        plugin.prepare_continuation(invocation_id=context.invocation_id, sdk_invocation_id="sdk-2")
        with pytest.raises(RuntimeError, match="stale"):
            await plugin.before_run_callback(
                invocation_context=FakeContext("foreign", session=context.session)
            )
        assert plugin.model_output_limited
        await plugin.before_run_callback(
            invocation_context=FakeContext("sdk-2", session=context.session)
        )
        assert not plugin.model_output_limited
        await plugin.complete_invocation(invocation_id=context.invocation_id, phase="failed")

    asyncio.run(scenario())


@pytest.mark.parametrize("foreign", ["sdk", "session", "app", "user"])
def test_only_authorized_tool_cancellation_can_latch_current_turn(foreign):
    async def scenario():
        state = WorkerStateStore()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            budget=lambda: None,
            instrumentation=None,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
            allowed_tool_names=frozenset({"probe"}),
        )
        own = FakeContext("worker-child-cancel")
        plugin.prepare_invocation(invocation_id=own.invocation_id, subtask_id="0")
        await plugin.before_run_callback(invocation_context=own)
        outsider = FakeContext(own.invocation_id)
        if foreign == "sdk":
            outsider.invocation_id = "foreign-sdk"
        elif foreign == "session":
            outsider.session.id = "foreign-session"
        elif foreign == "app":
            outsider.session.app_name = "foreign-app"
        else:
            outsider.session.user_id = "foreign-user"
        tool = FakeTool("probe", FakeOwner())
        with pytest.raises(RuntimeError, match="stale"):
            await plugin.on_tool_error_callback(
                tool=tool, tool_args={}, tool_context=outsider, error=asyncio.CancelledError()
            )
        assert not plugin.tool_cancellation_observed
        await plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=own)
        with pytest.raises(asyncio.CancelledError):
            await plugin.on_tool_error_callback(
                tool=tool, tool_args={}, tool_context=own, error=asyncio.CancelledError()
            )
        assert plugin.tool_cancellation_observed
        await plugin.complete_invocation(invocation_id=own.invocation_id, phase="cancelled")
        plugin.prepare_invocation(invocation_id="worker-healthy-next", subtask_id="1")
        assert not plugin.tool_cancellation_observed

    asyncio.run(scenario())
