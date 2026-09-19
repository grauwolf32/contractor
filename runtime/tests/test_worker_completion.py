"""Worker completion dispatch and ownership independent of any specific toolset."""

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fakes.model import scripted_model, text_result
from test_adk_runtime import build_context, stage_request

from contractor_runtime.contracts import (
    MAX_WORKER_RESULT_BYTES,
    ArtifactRef,
    WorkerCompletionContract,
    WorkerFailure,
    WorkerModelResult,
    WorkerObservations,
    WorkerResult,
)
from contractor_runtime.worker import runtime as worker_runtime
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


@pytest.fixture
def assembly_runtime(tmp_path):
    runtime = AdkWorkerRuntime(build_context(tmp_path, WorkerStateStore(), {}), scripted_model([]))
    yield runtime
    asyncio.run(runtime.finalize(datetime.now(UTC) + timedelta(seconds=2)))


@pytest.fixture
def collected_runtime(tmp_path):
    binding = CollectedCompletion()
    context = replace(
        build_context(tmp_path, WorkerStateStore(), selected_tools(binding)),
        completion_contract=binding.contract,
    )
    runtime = AdkWorkerRuntime(context, scripted_model([]))
    yield runtime, binding
    asyncio.run(runtime.finalize(datetime.now(UTC) + timedelta(seconds=2)))


@pytest.mark.parametrize(
    ("candidate", "code", "retryable"),
    [
        pytest.param(None, "worker_result_missing", True, id="missing"),
        pytest.param("private-decoder-canary", "worker_result_invalid", True, id="malformed"),
        pytest.param("[]", "worker_result_invalid", True, id="non-object"),
        pytest.param('{"subtaskId":"0"}', "worker_result_invalid", True, id="missing-text"),
        pytest.param(
            '{"subtaskId":"0","result":"done","artifacts":{}}',
            "worker_result_invalid",
            True,
            id="model-artifact-authority",
        ),
        pytest.param(
            '{"subtaskId":"0","result":7}',
            "worker_result_invalid",
            True,
            id="non-text",
        ),
        pytest.param(
            '{"subtaskId":"0","result":""}',
            "worker_result_invalid",
            True,
            id="empty-text",
        ),
        pytest.param(
            '{"subtaskId":"0","result":"  "}',
            "worker_result_invalid",
            True,
            id="blank-text",
        ),
        pytest.param(
            '{"subtaskId":"not valid","result":"done"}',
            "worker_result_invalid",
            True,
            id="invalid-subtask-schema",
        ),
        pytest.param(
            '{"subtaskId":"1","result":"done"}',
            "worker_result_subtask_mismatch",
            True,
            id="different-subtask",
        ),
        pytest.param(
            json.dumps(
                {"subtaskId": "0", "result": "é" * (MAX_WORKER_RESULT_BYTES // 2 + 1)},
                ensure_ascii=False,
            ),
            "worker_result_too_large",
            False,
            id="utf8-byte-overflow",
        ),
        pytest.param(
            json.dumps(
                {"subtaskId": "1", "result": "x" * (MAX_WORKER_RESULT_BYTES + 1), "extra": True}
            ),
            "worker_result_too_large",
            False,
            id="text-size-before-schema-and-id",
        ),
        pytest.param(
            '{"subtaskId":"1","result":""}',
            "worker_result_invalid",
            True,
            id="schema-before-id",
        ),
        pytest.param(
            " " * (worker_runtime.MAX_STAGE_RESULT_JSON_BYTES + 1),
            "worker_result_too_large",
            False,
            id="encoded-size-before-json",
        ),
    ],
)
def test_model_result_decoder_preserves_validation_errors(
    assembly_runtime, candidate, code, retryable
):
    decoded = worker_runtime._decode_model_result(candidate, expected_subtask_id="0")
    assert isinstance(decoded, WorkerFailure)
    assert (decoded.code, decoded.retryable) == (code, retryable)
    assert "private-decoder-canary" not in decoded.model_dump_json()
    result, exportable = assembly_runtime._build_runtime_result(stage_request(), candidate, ())
    assert result == decoded and exportable is False


@pytest.mark.parametrize("boundary", ["utf8-text", "encoded-json"])
def test_model_result_decoder_accepts_exact_size_boundaries(boundary):
    text = "é" * (MAX_WORKER_RESULT_BYTES // 2) if boundary == "utf8-text" else "done"
    candidate = json.dumps({"subtaskId": "0", "result": text}, ensure_ascii=False)
    if boundary == "encoded-json":
        candidate += " " * (worker_runtime.MAX_STAGE_RESULT_JSON_BYTES - len(candidate.encode()))
        assert len(candidate.encode()) == worker_runtime.MAX_STAGE_RESULT_JSON_BYTES
        overflow = worker_runtime._decode_model_result(candidate + " ", expected_subtask_id="0")
        assert isinstance(overflow, WorkerFailure)
        assert overflow.code == "worker_result_too_large" and overflow.retryable is False
    else:
        assert len(text.encode()) == MAX_WORKER_RESULT_BYTES
    decoded = worker_runtime._decode_model_result(candidate, expected_subtask_id="0")
    assert decoded == WorkerModelResult(subtaskId="0", result=text)


async def _finish_collected_result(runtime, binding, request, result):
    async def finish(*, request, deadline, check_active):
        check_active()
        assert deadline > asyncio.get_running_loop().time()
        return CompleteCompletion(result)

    binding.finish = finish
    return await runtime._complete_normal_finish(
        request=request,
        candidate="Model prose does not own collected results",
        invocation_id="completion-boundary",
        deadline=asyncio.get_running_loop().time() + 1,
    )


def _collected_result(text="Collected result"):
    return WorkerResult(
        subtaskId="999",
        result=text,
        artifacts={},
        summarized=True,
        observations=WorkerObservations(
            profile="lean@1",
            tools={"untrusted": {"calls": 9, "failures": 1}},
            workspace=None,
            truncated=True,
        ),
    )


def test_collected_completion_uses_typed_request_owned_result(collected_runtime, monkeypatch):
    runtime, binding = collected_runtime

    def reject_model_decoder(*args, **kwargs):
        pytest.fail("trusted completion must not enter the model JSON decoder")

    monkeypatch.setattr(worker_runtime, "_decode_model_result", reject_model_decoder)
    request = stage_request()
    request.result_artifacts.update(
        {
            "observed": ArtifactRef(namespace="builder", name="observed"),
            "missing": ArtifactRef(namespace="builder", name="missing"),
            "workspace": ArtifactRef(namespace="builder", name="workspace"),
        }
    )
    runtime._workspace_exporter = SimpleNamespace(reserved_slots=frozenset({"workspace"}))
    runtime._invocation_observed_refs = [
        ArtifactRef(namespace="builder", name="report", revision="observed-r1"),
        ArtifactRef(namespace="builder", name="observed", revision="observed-r1"),
        ArtifactRef(namespace="builder", name="report", revision="observed-r2"),
        ArtifactRef(namespace="builder", name="observed", revision="observed-r2"),
        ArtifactRef(namespace="builder", name="unrequested", revision="observed-r1"),
    ]
    decision = _collected_result()
    decision.artifacts.update(
        {
            "publisher-slot-one": ArtifactRef(
                namespace="builder", name="report", revision="published-r1"
            ),
            "publisher-slot-two": ArtifactRef(
                namespace="builder", name="report", revision="published-r2"
            ),
            "publisher-extra": ArtifactRef(
                namespace="builder", name="extra", revision="published-r1"
            ),
            "publisher-workspace": ArtifactRef(
                namespace="builder", name="workspace", revision="published-r1"
            ),
        }
    )
    outcome, exportable = asyncio.run(_finish_collected_result(runtime, binding, request, decision))
    assert exportable is True
    assert isinstance(outcome, WorkerResult) and outcome is not decision
    assert outcome.subtask_id == request.subtask_id == "0"
    assert outcome.result == decision.result
    assert outcome.artifacts == {
        "report": ArtifactRef(namespace="builder", name="report", revision="published-r2"),
        "observed": ArtifactRef(namespace="builder", name="observed", revision="observed-r2"),
    }
    assert outcome.summarized is False
    assert outcome.observations == WorkerObservations(
        profile="lean@1", tools={}, workspace=None, truncated=False
    )
    assert decision.subtask_id == "999" and decision.summarized is True


@pytest.mark.parametrize(
    ("mutated_text", "code", "retryable"),
    [
        pytest.param(None, "worker_result_invalid", True, id="none"),
        pytest.param(7, "worker_result_invalid", True, id="non-text"),
        pytest.param("", "worker_result_invalid", True, id="empty"),
        pytest.param(" \n ", "worker_result_invalid", True, id="blank"),
        pytest.param(
            "é" * (MAX_WORKER_RESULT_BYTES // 2 + 1),
            "worker_result_too_large",
            False,
            id="utf8-overflow",
        ),
    ],
)
def test_collected_completion_revalidates_mutated_result_text(
    collected_runtime, mutated_text, code, retryable
):
    runtime, binding = collected_runtime
    decision = _collected_result()
    # Pydantic DTOs remain mutable; prior construction is not permanent validation.
    decision.result = mutated_text
    outcome, exportable = asyncio.run(
        _finish_collected_result(runtime, binding, stage_request(), decision)
    )
    assert isinstance(outcome, WorkerFailure) and exportable is False
    assert (outcome.code, outcome.retryable) == (code, retryable)
    assert outcome.message in {
        "Worker returned an invalid structured result",
        "Worker structured result exceeds its limit",
    }


def test_collected_completion_removes_only_synthetic_json_encoding_bound(collected_runtime):
    runtime, binding = collected_runtime
    decision = _collected_result("\x7f" * 44_000)
    synthetic = json.dumps({"subtaskId": "0", "result": decision.result})
    assert len(decision.result.encode()) < MAX_WORKER_RESULT_BYTES
    assert len(synthetic.encode()) > worker_runtime.MAX_STAGE_RESULT_JSON_BYTES
    assert (
        len(decision.model_dump_json(by_alias=True).encode())
        < worker_runtime.MAX_STAGE_RESULT_JSON_BYTES
    )
    # The same bytes are still too large when supplied as actual model JSON.
    decoded = worker_runtime._decode_model_result(synthetic, expected_subtask_id="0")
    assert isinstance(decoded, WorkerFailure)
    assert decoded.code == "worker_result_too_large" and decoded.retryable is False
    outcome, exportable = asyncio.run(
        _finish_collected_result(runtime, binding, stage_request(), decision)
    )
    assert isinstance(outcome, WorkerResult) and exportable is True
    assert outcome.subtask_id == "0" and outcome.result == decision.result
    assert (
        len(outcome.model_dump_json(by_alias=True).encode())
        < worker_runtime.MAX_STAGE_RESULT_JSON_BYTES
    )


def test_collected_completion_retains_actual_worker_result_wire_bound(collected_runtime):
    runtime, binding = collected_runtime
    decision = _collected_result("\x01" * 44_000)
    assert len(decision.result.encode()) < MAX_WORKER_RESULT_BYTES
    assert (
        len(decision.model_dump_json(by_alias=True).encode())
        > worker_runtime.MAX_STAGE_RESULT_JSON_BYTES
    )
    outcome, exportable = asyncio.run(
        _finish_collected_result(runtime, binding, stage_request(), decision)
    )
    assert isinstance(outcome, WorkerFailure) and exportable is False
    assert outcome.code == "worker_result_too_large" and outcome.retryable is False
    assert outcome.message == "Worker result exceeds its limit"


@pytest.mark.parametrize(
    ("namespace", "name"),
    [
        ("inputs", "report"),
        ("outputs", "report"),
        ("skills", "report"),
        ("builder", "memory.shared"),
    ],
)
def test_result_assembly_preserves_reserved_binding_policy(assembly_runtime, namespace, name):
    request = stage_request()
    request.result_artifacts["report"] = ArtifactRef(namespace=namespace, name=name)
    fields = WorkerModelResult(subtaskId=request.subtask_id, result="done")
    observed = (ArtifactRef(namespace=namespace, name=name, revision="exact-r1"),)
    outcome, exportable = assembly_runtime._assemble_runtime_result(request, fields, observed)
    assert isinstance(outcome, WorkerFailure) and exportable is False
    assert outcome.code == "invalid_worker_result_binding" and outcome.retryable is False


def test_model_decoder_and_result_policy_keep_error_precedence(assembly_runtime):
    request = stage_request()
    request.result_artifacts["report"] = ArtifactRef(namespace="inputs", name="report")
    secret = assembly_runtime._context.runtime_settings.llm_gateway_token.get_secret_value()
    for subtask_id, extra, code in [
        ("1", {"extra": True}, "worker_result_invalid"),
        ("1", {}, "worker_result_subtask_mismatch"),
        ("0", {}, "unsafe_worker_result"),
    ]:
        candidate = json.dumps({"subtaskId": subtask_id, "result": secret, **extra})
        outcome, exportable = assembly_runtime._build_runtime_result(request, candidate, ())
        assert isinstance(outcome, WorkerFailure) and exportable is False
        assert outcome.code == code
        assert outcome.retryable is (code != "unsafe_worker_result")
        assert secret not in outcome.model_dump_json()
