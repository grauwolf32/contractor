from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest
from fakes.model import json_result, scripted_model, tool_call
from google.genai import types

from contractor_runtime.contracts import API_VERSION, ModelPolicyRef, ResolvedModelPolicy
from contractor_runtime.contracts.planner import (
    PlannerCompaction,
    PlannerInvocation,
    PlannerOperationResult,
    PlannerTool,
)
from contractor_runtime.planner.host import PlannerStoppedError
from contractor_runtime.planner.loop import PlannerLoop


def planner_policy():
    return ResolvedModelPolicy(
        ref=ModelPolicyRef(policyId="planner-test", version="1", digest="sha256:" + "a" * 64),
        model="planner-test-model",
        maxOutputTokens=1024,
        maxModelCalls=4,
        maxWorkerCalls=2,
        maxTotalTokens=10000,
    )


def planner_input():
    return PlannerInvocation(
        input="Inspect the input and finish",
        tools=[
            {
                "name": "finish",
                "description": "Submit the confirmed result",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            }
        ],
    )


def test_planner_loop_runs_with_an_injected_local_host_without_server_bootstrap():
    async def scenario():
        calls = []

        class Host:
            async def execute(self, name, arguments):
                calls.append((name, arguments))
                return PlannerOperationResult(
                    apiVersion=API_VERSION,
                    result={"accepted": True} if name == "finish" else {"captureContent": True},
                    stop=name == "finish",
                    escalate=name == "finish",
                )

        model = scripted_model(
            [tool_call("finish", {}, call_id="finish-1")], model=planner_policy().model
        )
        from contractor_runtime.llm.cost import CostCollector

        costs = CostCollector()
        costs.observe(model.model, {"x-litellm-response-cost": "0.001"}, b"")
        loop = PlannerLoop(model, Host(), planner_policy(), cost_snapshot=lambda: costs.snapshot(0))
        await loop.run(planner_input(), "Use the provided tools; preserve literal {braces}.")
        assert [name for name, _ in calls] == [
            "before_model",
            "model_content",
            "after_model",
            "finish",
        ]
        assert "{braces}" in calls[1][1]["content"]
        assert calls[2][1]["cost"]["groups"][0]["amount"] == "0.001"
        assert calls[2][1]["cost"]["calls"] == 1
        assert "finish" in calls[2][1]["content"]
        assert loop.stopped
        assert len(model.requests) == 1
        assert "{braces}" in str(model.requests[0]["systemInstruction"])
        assert model.requests[0]["model"] == "planner-test-model"

    asyncio.run(scenario())


def test_planner_core_imports_with_server_adapters_blocked():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc, sys
class DenyServer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {
            "contractor_runtime.artifacts", "contractor_runtime.a2a_server",
            "contractor_runtime.allocation", "contractor_runtime.factories",
            "contractor_runtime.server", "contractor_runtime.planner.server_host",
        }:
            raise ImportError("Planner core imported a Server adapter: " + fullname)
sys.meta_path.insert(0, DenyServer())
from contractor_runtime.planner.loop import PlannerLoop
""",
        ],
        check=True,
        capture_output=True,
        timeout=10,
    )


def test_planner_host_refusal_prevents_any_model_call():
    async def scenario():
        class Host:
            async def execute(self, name, arguments):
                return PlannerOperationResult(
                    apiVersion=API_VERSION,
                    result={},
                    stop=True,
                    escalate=False,
                    failure={
                        "code": "planner_model_call_limit",
                        "message": "Planner budget exhausted",
                        "retryable": False,
                    },
                )

        model = scripted_model([tool_call("finish", {}, call_id="must-not-run")])
        with pytest.raises(PlannerStoppedError, match="budget exhausted"):
            await PlannerLoop(model, Host(), planner_policy()).run(planner_input(), "Inspect")
        assert model.requests == []

    asyncio.run(scenario())


def test_planner_cancellation_joins_model_before_returning():
    async def scenario():
        calls = []

        class Host:
            async def execute(self, name, arguments):
                calls.append(name)
                return PlannerOperationResult(
                    apiVersion=API_VERSION, result={}, stop=False, escalate=False
                )

        model = scripted_model([tool_call("finish", {}, call_id="cancelled-call")], block=True)
        task = asyncio.create_task(
            PlannerLoop(model, Host(), planner_policy()).run(planner_input(), "Inspect")
        )
        await model.blocked.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        model.release()
        assert calls == ["before_model"]
        assert task.done()

    asyncio.run(scenario())


def compaction_input():
    invocation = planner_input()
    invocation.tools.append(invocation.tools[0].model_copy(update={"name": "step"}))
    invocation.compaction = PlannerCompaction.model_validate(
        {
            "instructions": "Keep evidence and remaining work.",
            "threshold": 50,
            "keepRecentGroups": 1,
            "maxCompactions": 2,
            "policy": {
                "ref": {
                    "policyId": "compactor",
                    "version": "1",
                    "digest": "sha256:" + "b" * 64,
                },
                "model": "compactor-test-model",
                "contextWindowTokens": 100000,
                "maxOutputTokens": 1024,
                "maxModelCalls": 1,
                "maxTotalTokens": 100000,
            },
        }
    )
    return invocation


@pytest.mark.parametrize("protected_bytes", [0, 101000])
@pytest.mark.parametrize("rejected_after_checkpoint", [False, True])
def test_planner_compacts_with_local_host_and_injected_models_without_server(
    rejected_after_checkpoint,
    protected_bytes,
):
    async def scenario():
        calls = []

        class Host:
            async def execute(self, name, arguments):
                calls.append(name)
                result = {
                    "plan": {"revision": 2, "subtaskCounts": {"succeeded": 2}},
                    "retainedBytes": 0,
                    "objective": "Inspect and finish",
                }
                if name == "after_compactor":
                    result = {"summary": "Two earlier steps are complete."}
                if name == "after_model":
                    result = {"rejected": arguments["calls"] == ["unknown"]}
                return PlannerOperationResult(
                    apiVersion=API_VERSION,
                    result=result,
                    stop=name == "finish",
                    escalate=name == "finish",
                )

        invocation = compaction_input()
        invocation.input = json.dumps(
            {
                "objective": "Inspect and finish",
                "parameters": {"large": "protected-parameter:" + "p" * protected_bytes},
            }
        )
        steps = [tool_call("step", {}, call_id=f"step-{i}") for i in range(2)]
        for response in steps:
            response.usage_metadata.prompt_token_count = 100
            response.usage_metadata.total_token_count = 103
        if rejected_after_checkpoint:
            rejected = tool_call("unknown", {}, call_id="rejected")
            rejected.usage_metadata.prompt_token_count = 100
            rejected.usage_metadata.total_token_count = 103
            steps.append(rejected)
        model = scripted_model([*steps, tool_call("finish", {}, call_id="finish")])
        summary = json_result({"summary": "Two earlier steps are complete."})
        summary.finish_reason = types.FinishReason.STOP
        compactor = scripted_model([summary], model="compactor-test-model")
        loop = PlannerLoop(model, Host(), planner_policy(), compactor=compactor)
        if rejected_after_checkpoint:
            with pytest.raises(PlannerStoppedError, match="ineffective") as error:
                await loop.run(invocation, "Inspect and finish")
            assert error.value.code == "planner_compaction_failed"
            assert not error.value.retryable
            assert calls.count("checkpoint") == 1
            assert len(model.requests) == 3
            assert "finish" not in calls
            return
        await loop.run(invocation, "Inspect and finish")
        assert loop.stopped
        assert calls.count("before_model") == 3
        assert calls.count("before_compactor") == calls.count("checkpoint") == 1
        assert calls.index("checkpoint") < len(calls) - 3
        assert len(compactor.requests) == 1
        assert compactor.requests[0]["hasResponseSchema"]
        assert compactor.requests[0]["toolNames"] == []
        document = json.loads(compactor.requests[0]["contentText"])
        assert document["objective"] == "Inspect and finish"
        assert "parameters" not in document
        assert len(compactor.requests[0]["contentText"]) < 10000
        compactor_input = json.loads(compactor.requests[0]["contentText"])
        assert compactor_input["objective"] == "Inspect and finish"
        assert "protected-parameter:" not in compactor.requests[0]["contentText"]
        assert "Two earlier steps are complete." in model.requests[-1]["contentText"]
        assert invocation.input in model.requests[-1]["contentText"]

    asyncio.run(scenario())


@pytest.mark.parametrize("compact", [False, True])
def test_router_uses_the_same_loop_with_two_logical_workers_and_local_host(compact):
    async def scenario():
        routed = []
        operations = []

        class Host:
            async def execute(self, name, arguments):
                operations.append(name)
                if name == "execute_current_subtask":
                    routed.append(arguments["worker_name"])
                return PlannerOperationResult(
                    apiVersion=API_VERSION,
                    result={
                        "plan": {"revision": 2},
                        "retainedBytes": 0,
                        "objective": "Inspect and finish",
                        "summary": "Both Workers completed.",
                    },
                    stop=name == "finish",
                    escalate=name == "finish",
                )

        invocation = planner_input()
        if compact:
            invocation.compaction = compaction_input().compaction
        invocation.tools.append(
            PlannerTool.model_validate(
                {
                    "name": "execute_current_subtask",
                    "description": "Execute the selected Worker",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subtask_id": {"type": "string"},
                            "worker_name": {"type": "string", "enum": ["builder", "reviewer"]},
                        },
                        "required": ["subtask_id", "worker_name"],
                        "additionalProperties": False,
                    },
                }
            )
        )
        model = scripted_model(
            [
                tool_call(
                    "execute_current_subtask",
                    {"subtask_id": "0", "worker_name": "builder"},
                    call_id="route-0",
                ),
                tool_call(
                    "execute_current_subtask",
                    {"subtask_id": "1", "worker_name": "reviewer"},
                    call_id="route-1",
                ),
                tool_call("finish", {}, call_id="finish-router"),
            ],
            model=planner_policy().model,
        )
        if compact:
            for response in model.responses[:2]:
                response.usage_metadata.prompt_token_count = 100
                response.usage_metadata.total_token_count = 103
        summary = json_result({"summary": "Both Workers completed."})
        summary.finish_reason = types.FinishReason.STOP
        compactor = scripted_model([summary], model="compactor-test-model") if compact else None
        policy = planner_policy()
        policy.temperature = 0.15
        await PlannerLoop(model, Host(), policy, compactor=compactor).run(
            invocation, "Route to the named Workers"
        )
        assert operations.count("checkpoint") == int(compact)
        assert routed == ["builder", "reviewer"]
        assert len(model.requests) == 3
        assert all(request["temperature"] == 0.15 for request in model.requests)
        assert all(
            request["maxOutputTokens"] == policy.max_output_tokens for request in model.requests
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("count", [2, 65, 1000])
def test_planner_parallel_selection_is_bounded_and_recovers_before_any_effect(count):
    async def scenario():
        calls = []

        class Host:
            async def execute(self, name, arguments):
                calls.append((name, arguments))
                result = {"rejected": len(arguments.get("calls", [])) > 1}
                return PlannerOperationResult(
                    apiVersion=API_VERSION,
                    result=result,
                    stop=name == "finish",
                    escalate=name == "finish",
                )

        parallel = tool_call("finish", {}, call_id="parallel-0")
        parallel.content.parts = [
            types.Part(function_call=types.FunctionCall(name="finish", id=f"parallel-{i}", args={}))
            for i in range(count)
        ]
        model = scripted_model([parallel, tool_call("finish", {}, call_id="confirmed")])
        await PlannerLoop(model, Host(), planner_policy()).run(planner_input(), "Inspect")
        selections = [args for name, args in calls if name == "after_model"]
        assert selections[0]["calls"] == ["finish", "finish"]
        assert selections[0]["inputTokens"] == 7 and selections[0]["outputTokens"] == 3
        assert len(selections) == 2
        assert sum(name == "finish" for name, _ in calls) == 1
        assert len(model.requests) == 2

    asyncio.run(scenario())
