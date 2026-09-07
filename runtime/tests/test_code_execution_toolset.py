from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pytest
from google.adk.tools import FunctionTool

from contractor_runtime.sandbox.contracts import (
    ExecutionResult,
    ExecutionStatus,
    SandboxContractError,
    SandboxErrorCode,
)
from contractor_runtime.telemetry.execution import ContentFreeInstrumentation
from contractor_runtime.toolsets.code_execution.tools import (
    CodeExecutionToolsetFactory,
    ExecCommandTool,
)
from contractor_runtime.worker.execution import SandboxExecutionFailed
from contractor_runtime.worker.state import WorkerStateStore


class Executor:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or ExecutionResult(
            ExecutionStatus.COMPLETED, 3, "secret output", "", False, False, 5, stdout_bytes=13
        )

    async def execute(self, request, *, deadline):
        self.calls.append((request, deadline))
        return self.result


def test_exact_schema_nonzero_result_and_content_free_metrics():
    async def scenario():
        state = WorkerStateStore()
        executor = Executor()
        tool = ExecCommandTool(executor, state)
        declaration = FunctionTool(tool)._get_declaration()
        assert set(declaration.parameters_json_schema["properties"]) == {
            "command",
            "cwd",
            "timeout_seconds",
        }
        command = "printf '%s' 'private; $(shell)'; exit 3"
        result = await tool(command, "src")
        assert result == executor.result.observation()
        assert result["status"] == "completed" and result["exitCode"] == 3
        assert executor.calls[0][0].command == command
        assert 58 < (executor.calls[0][1] - datetime.now(UTC)).total_seconds() <= 60
        encoded = json.dumps(state.metrics.counters) + state.metrics.tool_calls[0].model_dump_json()
        assert "private" not in encoded and "secret output" not in encoded
        assert state.metrics.counters["sandbox.stdout_bytes"] == 13

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "args",
    [
        {"command": ""},
        {"command": "a" * 65537},
        {"command": "я" * 32769},
        {"command": "x\x00"},
        {"command": "\ud800"},
        {"command": 3},
        {"command": "true", "cwd": "../escape"},
        {"command": "true", "cwd": "/tmp"},
        {"command": "true", "cwd": "link/../src"},
        {"command": "true", "cwd": None},
        {"command": "true", "timeout_seconds": True},
        {"command": "true", "timeout_seconds": 0},
        {"command": "true", "timeout_seconds": 3601},
        {"command": "true", "timeout_seconds": 1.2},
    ],
)
def test_invalid_arguments_launch_nothing(args):
    async def scenario():
        executor = Executor()
        state = WorkerStateStore()
        tool = ExecCommandTool(executor, state)
        assert isinstance(tool.contractor_raw_argument_error(args), SandboxContractError)
        result = await tool(**args)
        assert result["status"] == "failed" and result["exitCode"] is None
        assert executor.calls == [] and state.execution.failure is None

    asyncio.run(scenario())


def test_selected_factory_never_advertises_or_hands_out_unselected_executor():
    async def scenario():
        factory = CodeExecutionToolsetFactory()
        kwargs = dict(
            allocation_id="a",
            run_id="r",
            namespace="n",
            runtime_settings=None,
            workspace=None,
            state=WorkerStateStore(),
        )
        assert await factory.probe() == frozenset()
        assert await factory.create_selected(selected=[], **kwargs) == {}
        for selected in (["unknown"], ["exec_command"]):
            with pytest.raises(SandboxContractError):
                await factory.create_selected(selected=selected, **kwargs)
        tools = await factory.create_selected(
            selected=["exec_command"], sandbox_executor=Executor(), **kwargs
        )
        assert set(tools) == {"exec_command"}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "code",
    [SandboxErrorCode.TIMEOUT, SandboxErrorCode.OUTPUT_LIMIT, SandboxErrorCode.OUTCOME_UNKNOWN],
)
def test_fatal_observation_blocks_further_calls(code):
    async def scenario():
        executor = Executor(
            ExecutionResult(ExecutionStatus.FAILED, None, "", "", False, False, 1, code)
        )
        state = WorkerStateStore()
        tool = ExecCommandTool(executor, state)
        assert (await tool("private"))["errorCode"] == code.value
        with pytest.raises(SandboxExecutionFailed):
            await tool("must not run")
        assert len(executor.calls) == 1

    asyncio.run(scenario())


def test_content_free_wrapper_does_not_forward_optional_content_authority():
    class Sink:
        capture_content = True

        def start_span(self, name, *, attributes=None):
            return self

        def set_content(self, *args):
            pytest.fail("execution content escaped to telemetry")

        def end(self, **kwargs):
            self.ended = kwargs

    sink = Sink()
    span = ContentFreeInstrumentation(sink).start_span("tool", attributes={"count": 1})
    assert not getattr(span, "capture_content", False)
    span.end(outcome="succeeded")
    assert sink.ended["outcome"] == "succeeded"


def test_adk_fatal_execution_stops_model_finalizer_and_later_invocations(tmp_path):
    from fakes.model import scripted_model, tool_call
    from test_adk_runtime import create_runtime, stage_request

    async def scenario():
        state = WorkerStateStore()
        executor = Executor(
            ExecutionResult(
                ExecutionStatus.TIMED_OUT,
                None,
                "partial",
                "",
                False,
                False,
                1,
                SandboxErrorCode.TIMEOUT,
            )
        )
        model = scripted_model(
            [tool_call("exec_command", {"command": "private"}, call_id="exec-1")]
        )
        runtime = await create_runtime(
            tmp_path, state, {"exec_command": ExecCommandTool(executor, state)}, model
        )
        try:
            completion = await runtime.invoke(stage_request())
            assert completion.result is None
            assert completion.failure.code == SandboxErrorCode.TIMEOUT.value
            assert not completion.failure.retryable
            assert state.metrics.counters["llm_calls"] == 1
            assert state.metrics.counters.get("llm_errors", 0) == 0
            assert (
                await runtime.invoke(stage_request())
            ).failure.code == SandboxErrorCode.TIMEOUT.value
            assert len(executor.calls) == 1
        finally:
            from datetime import timedelta

            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())


def test_adk_rejects_extra_authority_before_binding_without_metric_content(tmp_path):
    from fakes.model import scripted_model, tool_call
    from test_adk_runtime import create_runtime, stage_request, terminal_text

    async def scenario():
        state = WorkerStateStore()
        executor = Executor()
        model = scripted_model(
            [
                tool_call(
                    "exec_command",
                    {"command": "sensitive-shell", "env": "sensitive-env"},
                    call_id="invalid-exec",
                ),
                terminal_text("Argument rejected"),
            ]
        )
        runtime = await create_runtime(
            tmp_path, state, {"exec_command": ExecCommandTool(executor, state)}, model
        )
        try:
            assert (await runtime.invoke(stage_request())).result is not None
            assert executor.calls == []
            assert "sensitive" not in repr(state.metrics)
            assert state.metrics.counters["tool_calls"] == 1
        finally:
            from datetime import timedelta

            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())
