"""Exact Stage input bindings cannot leak across Worker invocations."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fakes.model import scripted_model, text_result
from fakes.worker_state import WorkerState
from test_adk_runtime import create_runtime, stage_request


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
def test_exact_input_binding_is_private_and_cleared_on_every_outcome(tmp_path, outcome):
    async def scenario():
        state = WorkerState()
        model = scripted_model(
            [] if outcome == "failure" else [text_result("Completed")], block=True
        )
        runtime = await create_runtime(tmp_path, state, {}, model)
        request = stage_request().model_copy(update={"result_artifacts": {}})
        task = asyncio.create_task(runtime.invoke(request))
        try:
            await asyncio.wait_for(model.blocked.wait(), timeout=2)
            selected = state.resources.invocation_artifacts
            assert selected["source"] == request.artifacts["source"]
            assert selected["source"] is not request.artifacts["source"]
            with pytest.raises(TypeError):
                selected["unassigned"] = request.artifacts["source"]
            request.artifacts["source"].revision = "changed-after-admission"
            assert selected["source"].revision == "input-r1"
            if outcome == "cancelled":
                runtime.cancel_active(task)
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                model.release()
                result = await task
                assert (result.failure is None) == (outcome == "success")
            assert not state.resources.invocation_artifacts
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())
