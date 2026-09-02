from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from contractor_runtime.adapters import RuntimeSpan, TelemetryAttribute
from contractor_runtime.instrumentation import WorkerInstrumentationPlugin
from contractor_runtime.worker_state import WorkerStateStore

SECRET = "instrumentation-secret-canary"
PROMPT = "prompt-body-that-must-not-survive"
SOURCE = "source-body-that-must-not-survive"
PRIVATE_URL = f"https://user:{SECRET}@private.example/api?token={SECRET}"


class ToolFailure(RuntimeError):
    code = "synthetic_tool_failure"
    retryable = True


@dataclass
class FakeSession:
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeContext:
    invocation_id: str
    session: FakeSession = field(default_factory=FakeSession)


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

    def end(
        self,
        *,
        outcome: str,
        attributes: dict[str, TelemetryAttribute] | None = None,
    ) -> None:
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
        unknown_response = await plugin.on_tool_error_callback(
            tool=FakeTool(SECRET, FakeOwner()),
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


def test_plugin_retains_only_latest_sequential_invocation() -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        plugin = WorkerInstrumentationPlugin(
            state=state,
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
