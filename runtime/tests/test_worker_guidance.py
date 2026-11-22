"""Current facts captured on the actual SDK request path."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes.model import scripted_model, text_result, tool_call
from fakes.worker_state import WorkerState
from google.genai import types
from test_adk_runtime import CapturingScriptedLlm, create_runtime, stage_request
from test_run_artifacts_toolset import FakeArtifactClient
from test_worker_compaction import COMPACTOR_MODEL, compact_runtime, summary, usage

from contractor_runtime.toolsets.run_artifacts.tools import ReadArtifactTool
from contractor_runtime.worker.guidance import GUIDANCE_PREAMBLE, MAX_GUIDANCE_BYTES


def facts(request):
    text = request["contentText"]
    assert text.count(GUIDANCE_PREAMBLE) == 1
    return json.loads(text.split(GUIDANCE_PREAMBLE)[1])


def test_shared_request_guidance_limit():
    fixture = json.loads(
        (Path(__file__).parents[2] / "api/testdata/request-guidance.json").read_text()
    )
    assert fixture["maxBytes"] == MAX_GUIDANCE_BYTES


@pytest.mark.parametrize("mode", ["none", "terminal@1", "compact@1"])
def test_current_capacity_and_checkpoint_reread(tmp_path, mode):
    compact = mode == "compact@1"

    async def scenario():
        state = WorkerState()
        read = ReadArtifactTool(FakeArtifactClient(), state.metrics, ())
        args = {"namespace": "inputs", "name": "source", "revision": "revision-read"}
        responses = [tool_call("read_artifact", args, call_id=f"read-{i}") for i in range(4)]
        # After the two successful reads and refusal, checkpoint before the fourth read.
        responses[2] = usage(responses[2], 600 if compact else 10)
        responses.append(usage(text_result("done"), None))
        model = CapturingScriptedLlm(model="worker-model", responses=responses)
        if compact:
            compactor = scripted_model(
                [summary("Exact artifact already read")],
                model=COMPACTOR_MODEL,
                auto_result_finalizer=False,
            )
            runtime = await compact_runtime(
                tmp_path, state, {"read_artifact": read}, model, compactor, keep=1
            )
        else:
            runtime = await create_runtime(
                tmp_path,
                state,
                {"read_artifact": read},
                model,
                max_model_calls=10,
                summary_model=(
                    scripted_model([text_result("summary")]) if mode == "terminal@1" else None
                ),
            )
        events = []
        original_append = runtime._session_service.append_event

        async def record_event(session, event):
            events.append(str(event))
            return await original_append(session, event)

        runtime._session_service.append_event = record_event
        try:
            result = await runtime.invoke(stage_request())
            assert result.result is not None
            main = [request for request in model.requests if not request["hasResponseSchema"]]
            views = [facts(request) for request in main]
            assert views[0]["contextManagement"] == mode
            assert views[0]["additionalModelCallsAfterThisCall"] >= 1
            assert views[1]["remainingToolCalls"] == views[0]["remainingToolCalls"] - 1
            assert views[3]["committedGeneration"] == int(compact)
            if compact:
                assert views[3]["remainingToolCalls"] == views[0]["remainingToolCalls"]
                assert (
                    views[3]["additionalModelCallsAfterThisCall"]
                    == views[0]["additionalModelCallsAfterThisCall"]
                )
            assert state.metrics.counters["tool_errors"] == (1 if compact else 2)
            assert all(
                len((GUIDANCE_PREAMBLE + json.dumps(view)).encode()) < MAX_GUIDANCE_BYTES
                for view in views
            )
            # The finalizer is a separate tool-free input, with no capacity hints.
            assert GUIDANCE_PREAMBLE not in model.requests[-1]["contentText"]
            assert GUIDANCE_PREAMBLE not in str(events)
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=5))

    asyncio.run(scenario())


def test_preparation_replaces_only_its_own_request_object(tmp_path):
    async def scenario():
        runtime = await create_runtime(
            tmp_path, WorkerState(), {}, scripted_model([text_result("done")])
        )
        runtime._invocation.budget = SimpleNamespace(
            max_model_calls=2,
            model_calls=1,
            checkpoint_model_calls=0,
            max_tool_calls=5,
            tool_calls=3,
            checkpoint_tool_calls=0,
            checkpoint_generation=0,
        )
        user = types.Content(role="user", parts=[types.Part(text=GUIDANCE_PREAMBLE + "user text")])
        request = SimpleNamespace(contents=[user])
        context = SimpleNamespace()
        try:
            await runtime._prepare_model_request(context, request)
            await runtime._prepare_model_request(context, request)
            assert len(request.contents) == 2 and request.contents[0] is user
            view = json.loads(request.contents[-1].parts[0].text.split(GUIDANCE_PREAMBLE)[1])
            assert view["additionalModelCallsAfterThisCall"] == 1
            assert view["remainingToolCalls"] == 2
        finally:
            runtime._invocation.budget = None
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=5))

    asyncio.run(scenario())
