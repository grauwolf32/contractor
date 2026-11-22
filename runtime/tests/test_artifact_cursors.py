"""Current reads, immutable continuation and private cursor lifecycle."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fakes.model import scripted_model, text_result, tool_call
from fakes.worker_state import WorkerState
from test_adk_runtime import create_runtime, stage_request
from test_text_artifacts_toolset import MemoryArtifactClient, make_tools

from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.toolsets.common.versions import ModelArtifacts


def test_fresh_reads_observe_current_and_cursors_preserve_exact_windows(tmp_path):
    async def scenario():
        state, client = WorkerState(), MemoryArtifactClient()
        original = client.seed("inputs", "report", "text/plain", b"old-head\nold-tail\n")
        tools = await make_tools(tmp_path, client, state)
        read = tools["read_artifact"]
        first = await read("inputs", "report", max_lines=1)
        newer = client.seed("inputs", "report", "text/plain", b"new-head\nnew-tail\n")
        continuation = await read(cursor=first["nextCursor"])
        current = await read("inputs", "report", max_lines=1)
        assert continuation["text"] == "old-tail\n"
        assert continuation["artifact"]["revision"] == original.revision
        assert current["text"] == "new-head\n" and current["artifact"]["revision"] == newer.revision
        assert read.contractor_repeat_rule({"namespace": "inputs", "name": "report"}) is None
        assert read.contractor_repeat_rule({"cursor": first["readCursor"]}) is not None
        state.resources.selected_versions.assign([original])
        selected = await ModelArtifacts(client, state.resources.selected_versions).select(
            "inputs", "report"
        )
        assert selected == original
        before = client.read_calls
        for args in (
            {"cursor": first["nextCursor"] + "x"},
            {"cursor": first["nextCursor"], "start_line": True},
            {"cursor": first["nextCursor"], "namespace": "inputs"},
        ):
            with pytest.raises(ToolInputError):
                await read(**args)
        assert client.read_calls == before
        state.resources.artifact_cursors.reset()
        with pytest.raises(ToolInputError, match="expired"):
            await read(cursor=first["readCursor"])

    asyncio.run(scenario())


def test_worker_completion_revokes_previous_invocation_cursors(tmp_path):
    async def scenario():
        state, client = WorkerState(), MemoryArtifactClient()
        original = client.seed("inputs", "report", "text/plain", b"original\n")
        tools = await make_tools(tmp_path, client, state)
        read = tools["read_artifact"]
        cursor = state.resources.artifact_cursors.issue(original)
        model = scripted_model(
            [
                tool_call("read_artifact", {"cursor": cursor}, call_id="exact"),
                text_result("Read original"),
            ]
        )
        runtime = await create_runtime(tmp_path, state, {"read_artifact": read}, model)
        try:
            result = await runtime.invoke(
                stage_request().model_copy(update={"result_artifacts": {}})
            )
            assert result.failure is None
            with pytest.raises(ToolInputError, match="expired"):
                await read(cursor=cursor)
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())
