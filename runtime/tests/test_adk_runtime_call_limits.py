from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fakes.spec import allocation_spec
from google.adk.runners import Runner
from test_adk_runtime import build_context, stage_request
from test_result_finalizer import RecordingObserver, policy

import contractor_runtime.worker.finalizer as finalizer_module
import contractor_runtime.worker.summarizer as summarizer_module
from contractor_runtime.allocation import WorkerState
from contractor_runtime.llm.client import new_gateway_client
from contractor_runtime.llm.openai import OpenAICompatibleGatewayLlm
from contractor_runtime.worker.finalizer import ResultFinalizerFailure, WorkerResultFinalizer
from contractor_runtime.worker.runtime import AdkWorkerRuntime
from contractor_runtime.worker.summarizer import SummarizerFailure, TerminalSummarizer


def _response(request, message, *, tools=False):
    return httpx.Response(
        200,
        request=request,
        json={
            "choices": [{"finish_reason": "tool_calls" if tools else "stop", "message": message}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        },
    )


@asynccontextmanager
async def _model(gateway):
    async with httpx.AsyncClient(transport=httpx.MockTransport(gateway), trust_env=False) as http:
        handle = new_gateway_client(
            base_url="https://owned-call-limits.test/v1",
            api_key="test-only-owned-call-limits",
            timeout_seconds=30,
            http_client=http,
        )
        yield OpenAICompatibleGatewayLlm(model="worker-model", client_handle=handle)


def _record_sdk_limits(monkeypatch):
    limits = []
    original = Runner.run_async

    async def observed(self, **kwargs):
        config = kwargs.get("run_config")
        limits.append(config.max_llm_calls if config is not None else None)
        async for event in original(self, **kwargs):
            yield event

    monkeypatch.setattr(Runner, "run_async", observed)
    return limits


async def _worker_workload(folder, *, maximum, terminal_at):
    counts = {"main": 0, "finalizer": 0, "tools": 0}

    async def probe() -> dict:
        """Complete exactly one bounded local operation."""
        counts["tools"] += 1
        return {"ok": True}

    def gateway(request):
        value = json.loads(request.content)
        if value.get("response_format") is not None:
            counts["finalizer"] += 1
            message = {
                "role": "assistant",
                "content": json.dumps({"subtaskId": "0", "result": "Done"}),
            }
            return _response(request, message)
        counts["main"] += 1
        if counts["main"] == terminal_at:
            return _response(request, {"role": "assistant", "content": "Done"})
        return _response(
            request,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"bounded-{counts['main']}",
                        "type": "function",
                        "function": {"name": "probe", "arguments": "{}"},
                    }
                ],
            },
            tools=True,
        )

    state = WorkerState()
    context = build_context(folder, state, {"probe": probe})
    context = replace(
        context,
        model_policy=context.model_policy.model_copy(
            update={
                "max_model_calls": maximum,
                "max_tool_calls": 10_000,
                "max_total_tokens": 100_000_000,
            }
        ),
    )
    async with _model(gateway) as model:
        runtime = AdkWorkerRuntime(context, model)
        await runtime.start()
        try:
            completion = await runtime.invoke(
                stage_request().model_copy(
                    update={"deadline": datetime.now(UTC) + timedelta(minutes=15)}
                )
            )
            snapshot = await state.snapshot()
        finally:
            await runtime.abort(datetime.now(UTC) + timedelta(minutes=2))
    return completion, snapshot, counts


def test_supported_model_ceiling_includes_999_main_calls_and_finalizer(tmp_path, monkeypatch):
    monkeypatch.delenv("ADK_MAX_LLM_CALLS", raising=False)
    limits = _record_sdk_limits(monkeypatch)
    completion, snapshot, counts = asyncio.run(
        _worker_workload(tmp_path, maximum=1000, terminal_at=999)
    )
    assert completion.failure is None and completion.result.result == "Done"
    assert counts == {"main": 999, "finalizer": 1, "tools": 998}
    metrics = snapshot["lastCompletedInvocation"]["metrics"]
    assert metrics["modelCalls"] == 1000 and metrics["modelErrors"] == 0
    assert (metrics["inputTokens"], metrics["outputTokens"], metrics["totalTokens"]) == (
        7000,
        3000,
        10000,
    )
    assert metrics["toolCalls"] == 998 and metrics["toolErrors"] == 0
    assert limits == [1000, 2]


def test_contractor_refuses_call_1001_before_dispatch_or_charge(tmp_path, monkeypatch):
    monkeypatch.delenv("ADK_MAX_LLM_CALLS", raising=False)
    limits = _record_sdk_limits(monkeypatch)
    completion, snapshot, counts = asyncio.run(
        _worker_workload(tmp_path, maximum=1000, terminal_at=None)
    )
    assert completion.result is None and completion.failure.code == "worker_budget_exhausted"
    assert completion.failure.retryable
    assert counts == {"main": 1000, "finalizer": 0, "tools": 1000}
    completed = snapshot["lastCompletedInvocation"]
    assert completed["phase"] == "failed"
    assert completed["metrics"]["modelCalls"] == 1000
    assert completed["metrics"]["totalTokens"] == 10000
    assert limits == [1000]
    budget = snapshot["metrics"]["workerBudget"]
    assert budget["exhausted"] == "model_calls" and budget["observedModelCalls"] == 1000


@pytest.mark.parametrize("ambient", ["1", "0", "1000000", "not-an-integer"])
def test_ambient_sdk_limit_cannot_replace_the_pinned_worker_budget(tmp_path, monkeypatch, ambient):
    monkeypatch.setenv("ADK_MAX_LLM_CALLS", ambient)
    limits = _record_sdk_limits(monkeypatch)
    completion, snapshot, counts = asyncio.run(
        _worker_workload(tmp_path, maximum=3, terminal_at=None)
    )
    assert completion.failure.code == "worker_budget_exhausted"
    assert counts == {"main": 3, "finalizer": 0, "tools": 3}
    assert snapshot["lastCompletedInvocation"]["metrics"]["modelCalls"] == 3
    assert limits == [3]


@pytest.mark.parametrize("role", ["finalizer", "summarizer"])
@pytest.mark.parametrize("ambient", ["1", "0", "1000000", "not-an-integer"])
@pytest.mark.parametrize("repair", [False, True], ids=["success", "refused-repair"])
def test_one_shot_sdk_limit_keeps_owned_outcomes_and_one_http_dispatch(
    monkeypatch, role, ambient, repair
):
    monkeypatch.setenv("ADK_MAX_LLM_CALLS", ambient)
    limits = _record_sdk_limits(monkeypatch)
    requests, guard_errors = [], []
    model_type = (
        finalizer_module._OneShotFinalizerModel
        if role == "finalizer"
        else summarizer_module._OneShotModel
    )
    original_guard = model_type._call_limit_error

    def guarded(self):
        error = original_guard(self)
        guard_errors.append(error)
        return error

    monkeypatch.setattr(model_type, "_call_limit_error", guarded)

    def gateway(request):
        requests.append(json.loads(request.content))
        if repair:
            return _response(
                request,
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "unavailable-one-shot-tool",
                            "type": "function",
                            "function": {"name": "unavailable", "arguments": "{}"},
                        }
                    ],
                },
                tools=True,
            )
        return _response(
            request,
            {"role": "assistant", "content": json.dumps({"subtaskId": "0", "result": "Done"})},
        )

    async def scenario():
        async with _model(gateway) as model:
            if role == "finalizer":
                observer = RecordingObserver()
                consumer = WorkerResultFinalizer(model=model, policy=policy(), observer=observer)
                operation = consumer.run(
                    subtask_id="0", result_text="Done", invocation_id="worker-invocation"
                )
            else:
                config = allocation_spec(summarizer=True).agent_template.summarizer
                consumer = TerminalSummarizer(
                    model=model, policy=config.model_policy, include_reasoning=False
                )
                operation = consumer.run(prompt="Summarize once", invocation_id="worker-invocation")
            if repair:
                expected_error = RuntimeError if role == "finalizer" else SummarizerFailure
                with pytest.raises(expected_error):
                    await operation
                assert len(guard_errors) == 1
                assert guard_errors[0].code == "call_limit_exceeded"
                assert isinstance(
                    guard_errors[0],
                    ResultFinalizerFailure if role == "finalizer" else SummarizerFailure,
                )
            else:
                candidate = await operation
                assert json.loads(candidate) == {"subtaskId": "0", "result": "Done"}
                assert guard_errors == []
            if role == "finalizer":
                assert observer.before == 1
                assert (observer.after, observer.errors) == ((0, 1) if repair else (1, 0))
            else:
                assert consumer.usage.model_calls == 1 and consumer.usage.total_tokens == 10

    asyncio.run(scenario())
    assert len(requests) == 1
    assert limits == [2]
