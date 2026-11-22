"""Repeat semantics through the real Worker Runner and direct concurrent admission."""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from fakes.model import text_result, tool_call
from fakes.worker_state import WorkerState
from test_adk_runtime import CapturingScriptedLlm, create_runtime, stage_request
from test_run_artifacts_toolset import FakeArtifactClient

from contractor_runtime.toolsets.run_artifacts.tools import ReadArtifactTool
from contractor_runtime.worker.repeats import ToolRepeatPlugin


def test_exact_reads_are_refused_and_work_continues(tmp_path):
    async def scenario():
        state = WorkerState()
        read = ReadArtifactTool(FakeArtifactClient(), state.metrics, ())
        args = {"namespace": "inputs", "name": "source", "revision": "revision-read"}
        sequence = [args, {**args, "max_lines": 200}, args, args, {**args, "max_lines": 1}, args]
        model = CapturingScriptedLlm(
            model="worker-model",
            responses=[
                *(
                    tool_call("read_artifact", arg, call_id=f"read-{i}")
                    for i, arg in enumerate(sequence)
                ),
                text_result("done"),
            ],
        )
        runtime = await create_runtime(
            tmp_path, state, {"read_artifact": read}, model, max_model_calls=10
        )
        try:
            result = await runtime.invoke(stage_request())
            assert result.result is not None
            assert state.metrics.counters["tool_calls"] == 6
            assert state.metrics.counters["tool_errors"] == 2
            assert (
                sum(
                    item.error is not None and item.error.code == "tool_repeat_limit"
                    for item in state.metrics.tool_calls
                )
                == 2
            )
            assert "tool_repeat_limit" in str(model.payloads[-2])
            assert not runtime._repeats._pending
        finally:
            await runtime.abort(datetime.now(UTC) + timedelta(seconds=5))

    asyncio.run(scenario())


def test_reservations_failure_interruption_and_checkpoint():
    async def scenario():
        read = ReadArtifactTool(FakeArtifactClient(), WorkerState().metrics, ())
        tool = SimpleNamespace(func=read)
        args = {"namespace": "inputs", "name": "source", "revision": "exact"}
        plugin = ToolRepeatPlugin()
        contexts = [object() for _ in range(8)]

        async def call(i, arguments=args):
            return await plugin.before_tool_callback(
                tool=tool, tool_args=arguments, tool_context=contexts[i]
            )

        assert await call(0) is None
        assert await call(1) is None
        assert (await call(2))["error"]["code"] == "tool_repeat_limit"
        plugin.settled(contexts[0], True)
        assert await call(3) is None
        plugin.interrupt()
        assert await call(4) is None
        plugin.settled(contexts[1], True)  # old series cannot decrement the new one
        plugin.settled(contexts[3], False)
        assert await call(5) is None
        assert (await call(6))["ok"] is False
        plugin.reset()  # the invocation/checkpoint owner calls this only after commit
        assert await call(7) is None
        assert await call(0, {**args, "revision": None}) is None
        assert plugin._series is None
        for invalid in ({"start_line": True}, {"max_lines": 401}, {"line_offset": -1}):
            assert read.contractor_repeat_rule({**args, **invalid}) is None

    asyncio.run(scenario())


def test_raw_rejection_breaks_series_without_bypassing_validation(tmp_path):
    async def scenario():
        state = WorkerState()
        read = ReadArtifactTool(FakeArtifactClient(), state.metrics, ())
        args = {"namespace": "inputs", "name": "source", "revision": "revision-read"}
        calls = [args, args, {**args, "secret_unknown": "do-not-echo"}, args, args, args]
        model = CapturingScriptedLlm(
            model="worker-model",
            responses=[
                *(
                    tool_call("read_artifact", arg, call_id=f"read-{i}")
                    for i, arg in enumerate(calls)
                ),
                text_result("done"),
            ],
        )
        runtime = await create_runtime(
            tmp_path, state, {"read_artifact": read}, model, max_model_calls=10
        )
        try:
            result = await runtime.invoke(stage_request())
            assert result.result is not None
            errors = [item.error.code for item in state.metrics.tool_calls if item.error]
            assert errors == ["tool_input_invalid", "tool_repeat_limit"]
            tool_messages = [
                message for message in model.payloads[-2]["messages"] if message["role"] == "tool"
            ]
            assert "do-not-echo" not in str(tool_messages)
        finally:
            await runtime.abort(datetime.now(UTC) + timedelta(seconds=5))

    asyncio.run(scenario())
