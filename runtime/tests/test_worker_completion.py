"""Worker completion dispatch and ownership independent of any specific toolset."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fakes.model import scripted_model, text_result
from test_adk_runtime import build_context, stage_request

from contractor_runtime.contracts import (
    ArtifactRef,
    WorkerCompletionContract,
    WorkerFailure,
    WorkerObservations,
    WorkerResult,
)
from contractor_runtime.worker.completion import (
    CompleteCompletion,
    ContinueCompletion,
    PreparedWorkerCompletion,
    bind_worker_completion,
)
from contractor_runtime.worker.runtime import AdkWorkerRuntime
from contractor_runtime.worker.state import WorkerStateStore


class CollectedCompletion(PreparedWorkerCompletion):
    required_tools = frozenset({"collect_result", "read_assignment"})

    def __init__(self, *, fail=False):
        # Keep the current wire contract while replacing its host implementation.
        self.contract = WorkerCompletionContract(
            kind="audit-check-results@1",
            task=ArtifactRef(namespace="inputs", name="task", revision="pinned-r1"),
            executionManifest=ArtifactRef(
                namespace="inputs", name="manifest", revision="pinned-r1"
            ),
            resultArtifact=ArtifactRef(namespace="builder", name="report"),
        )
        self.diagnostics = None
        self.diagnostics_sink = None
        self.phase_sink = None
        self.begins = []
        self.finishes = 0
        self.ended = False
        self.fail = fail

    def reset_diagnostics(self):
        self.diagnostics = None

    async def record_phase(self, phase, failure_code=None):
        pass

    def begin(self, invocation_id):
        self.begins.append(invocation_id)

    async def end(self):
        self.ended = True

    def validate_request(self, request):
        assert request.result_artifacts["report"] == self.contract.result_artifact
        return None

    def failure(self, error):
        return WorkerFailure(
            code="custom_completion_failed", message="Completion failed safely", retryable=True
        )

    async def finish(self, *, request, deadline, check_active):
        check_active()
        assert deadline > asyncio.get_running_loop().time()
        self.finishes += 1
        if self.finishes == 1:
            return ContinueCompletion("Collect the remaining result")
        if self.fail:
            raise RuntimeError("internal detail must not reach the result")
        return CompleteCompletion(
            WorkerResult(
                subtaskId=request.subtask_id,
                result="Collected result",
                artifacts={
                    "report": self.contract.result_artifact.model_copy(
                        update={"revision": "published-r1"}
                    )
                },
                summarized=False,
                observations=WorkerObservations(
                    profile="lean@1", tools={}, workspace=None, truncated=False
                ),
            )
        )


def selected_tools(binding):
    async def collect_result() -> str:
        """Collect the current result."""
        return "collected"

    async def read_assignment() -> str:
        """Read the current assignment."""
        return "assignment"

    collect_result.completion_binding = binding
    read_assignment.completion_binding = binding
    return {"collect_result": collect_result, "read_assignment": read_assignment}


@pytest.mark.parametrize("fail", [False, True])
def test_worker_dispatches_custom_completion_and_always_releases_it(tmp_path, fail):
    async def scenario():
        binding = CollectedCompletion(fail=fail)
        state = WorkerStateStore()
        context = replace(
            build_context(tmp_path, state, selected_tools(binding)),
            completion_contract=binding.contract,
        )
        model = scripted_model([text_result("First response"), text_result("Second response")])
        runtime = AdkWorkerRuntime(context, model)
        await runtime.start()
        try:
            outcome = await runtime.invoke(stage_request())
            assert len(binding.begins) == 1 and binding.finishes == 2 and binding.ended
            assert "Collect the remaining result" in model.requests[-1]["contentText"]
            assert len(model.requests) == 2
            if fail:
                assert outcome.result is None
                assert outcome.failure.code == "custom_completion_failed"
                assert "internal detail" not in outcome.model_dump_json()
            else:
                assert outcome.failure is None
                assert outcome.result.result == "Collected result"
                assert outcome.result.artifacts["report"].revision == "published-r1"
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=5))

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid", ["missing_tool", "different_owner", "forged_binding"])
def test_completion_rejects_incomplete_or_untrusted_tool_bindings(invalid):
    binding = CollectedCompletion()
    tools = selected_tools(binding)
    if invalid == "missing_tool":
        del tools["read_assignment"]
    elif invalid == "different_owner":
        tools["read_assignment"].completion_binding = CollectedCompletion()
    else:
        forged = SimpleNamespace(contract=binding.contract, required_tools=binding.required_tools)
        tools = selected_tools(forged)
    with pytest.raises(ValueError, match="trusted prepared tool binding"):
        bind_worker_completion(tools=tools, contract=binding.contract, summarizer=None)
