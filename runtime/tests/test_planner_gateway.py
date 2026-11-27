"""PLN-R86 regressions through the Runtime ADK loop and real Gateway adapter."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fakes.model import scripted_model, tool_call
from test_planner_loop import compaction_input, planner_input, planner_policy

from contractor_runtime.contracts import API_VERSION
from contractor_runtime.contracts.planner import PlannerOperationResult
from contractor_runtime.llm.client import new_gateway_client
from contractor_runtime.llm.openai import OpenAICompatibleGatewayLlm
from contractor_runtime.planner.host import PlannerStoppedError
from contractor_runtime.planner.loop import PlannerLoop


def gateway_reply(phase, case):
    message = {"content": '{"summary":"Earlier work is complete."}'}
    if phase == "main":
        message = {
            "tool_calls": [
                {
                    "id": "finish-1",
                    "function": {"name": "finish", "arguments": "{}"},
                }
            ]
        }
    value = {
        "choices": [{"finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 13, "completion_tokens": 4, "total_tokens": 17},
        "provider_extension": {"valid": True},
    }
    if case == "missing_usage":
        del value["usage"]
    elif case == "empty_usage":
        value["usage"] = {}
    elif case.startswith("missing_"):
        del value["usage"][case.removeprefix("missing_")]
    elif case == "inconsistent_usage":
        value["usage"]["total_tokens"] = 1
    elif case == "negative_usage":
        value["usage"]["prompt_tokens"] = -1
    elif case == "filtered":
        value["choices"][0] = {"finish_reason": "content_filter", "message": {"content": ""}}
    elif case.startswith("arguments_"):
        value["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = {
            "arguments_duplicate": '{"summary":"first","summary":"second"}',
            "arguments_escaped": '{"summary":"first","summar\\u0079":"second"}',
            "arguments_nested": '{"nested":{"x":1,"x":2}}',
            "arguments_trailing": "{} {}",
        }[case]
    raw = json.dumps(value)
    if case == "envelope_duplicate":
        raw = raw.replace('"total_tokens": 17', '"total_tokens": 999, "total_tokens": 17')
    elif case == "envelope_nested":
        raw = raw.replace('"valid": true', '"valid": false, "valid": true')
    elif case == "envelope_trailing":
        raw += " {}"
    return raw.encode()


CASES = [
    "valid",
    "missing_usage",
    "empty_usage",
    "missing_prompt_tokens",
    "missing_completion_tokens",
    "missing_total_tokens",
    "inconsistent_usage",
    "negative_usage",
    "envelope_duplicate",
    "envelope_nested",
    "envelope_trailing",
]


@pytest.mark.parametrize("phase", ["main", "compactor"])
@pytest.mark.parametrize("case", CASES)
def test_runtime_planner_gateway_rejects_untrusted_usage_and_envelopes(phase, case):
    asyncio.run(run_gateway_case(phase, case))


@pytest.mark.parametrize(
    "case",
    [
        "arguments_duplicate",
        "arguments_escaped",
        "arguments_nested",
        "arguments_trailing",
        "filtered",
    ],
)
def test_runtime_planner_gateway_rejects_arguments_and_filtered_main_response(case):
    asyncio.run(run_gateway_case("main", case))


async def run_gateway_case(phase, case):
    operations, requests = [], []

    class Host:
        async def execute(self, name, arguments):
            operations.append((name, arguments))
            result = {
                "plan": {"revision": 1},
                "retainedBytes": 0,
                "objective": "Inspect and finish",
            }
            failure = None
            if name == "model_failed":
                failure = {
                    "code": "planner_gateway_invalid_response",
                    "message": "Planner Gateway returned an invalid response",
                    "retryable": True,
                }
            elif name == "after_compactor":
                result = {"summary": "Earlier work is complete."}
            return PlannerOperationResult(
                apiVersion=API_VERSION,
                result=result,
                stop=name == "finish",
                escalate=name == "finish",
                **({"failure": failure} if failure else {}),
            )

    async def gateway(request):
        requests.append(request)
        return httpx.Response(
            200, content=gateway_reply(phase, case if len(requests) == 1 else "valid")
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
        model = OpenAICompatibleGatewayLlm(
            model="review-model",
            client_handle=new_gateway_client(
                base_url="https://gateway.invalid/v1",
                api_key="",
                timeout_seconds=10,
                http_client=http,
            ),
        )
        invocation = planner_input()
        # Native Go declarations omit an empty properties map during JSON encoding.
        invocation.tools[0].parameters.pop("properties")
        main, compactor = model, None
        if phase == "compactor":
            invocation = compaction_input()
            steps = [tool_call("step", {}, call_id=f"step-{i}") for i in range(2)]
            for response in steps:
                response.usage_metadata.prompt_token_count = 100
                response.usage_metadata.total_token_count = 103
            main = scripted_model([*steps, tool_call("finish", {}, call_id="finish")])
            compactor = model
        loop = PlannerLoop(main, Host(), planner_policy(), compactor=compactor)
        try:
            if case == "valid":
                await loop.run(invocation, "Use the provided tools.")
                assert loop.stopped
                assert any(name == "finish" for name, _ in operations)
                assert any(name == "checkpoint" for name, _ in operations) == (phase == "compactor")
            else:
                with pytest.raises(PlannerStoppedError) as error:
                    await loop.run(invocation, "Use the provided tools.")
                assert error.value.code == "planner_gateway_invalid_response"
                assert error.value.retryable
                assert not any(name in {"finish", "checkpoint"} for name, _ in operations)
                failures = [args for name, args in operations if name == "model_failed"]
                assert len(failures) == 1 and failures[0]["kind"] == "invalid_response"
                trusted = case.startswith("arguments_") or case == "filtered"
                assert failures[0]["usage"] == (
                    {"inputTokens": 13, "outputTokens": 4} if trusted else None
                )
            assert len(requests) == 1
            if phase == "main":
                declaration = json.loads(requests[0].content)["tools"][0]["function"]
                assert declaration["parameters"] == {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                }
        finally:
            await model.close()


@pytest.mark.parametrize(
    "case, reason, has_tools, rejection",
    [
        ("valid", "stop", False, None),
        ("tool_empty", "tool_calls", False, "tool_output"),
        ("function_empty", "function_call", False, "tool_output"),
        ("tool_valid", "tool_calls", True, "tool_output"),
        ("tool_bad_arguments", "tool_calls", True, "tool_output"),
        ("tool_many", "tool_calls", True, "tool_output"),
        ("empty", "stop", False, "empty"),
        ("filtered", "content_filter", False, "empty"),
        ("refusal", "content_filter", False, "empty"),
        ("length", "length", False, "output_limit"),
        ("unknown", "", False, "invalid"),
        ("missing_reason", "", False, "invalid"),
        ("invalid_summary", "stop", False, "invalid"),
    ],
)
def test_planner_compactor_preserves_provider_outcome_and_usage(case, reason, has_tools, rejection):
    async def scenario():
        from contractor_runtime.llm.cost import CostCollector

        operations, requests = [], []
        value = json.loads(gateway_reply("compactor", "valid"))
        choice = value["choices"][0]
        choice["finish_reason"] = reason or "unknown-provider-reason"
        if case == "missing_reason":
            del choice["finish_reason"]
        if case == "refusal":
            choice["finish_reason"] = "stop"
            choice["message"]["refusal"] = "Declined"
        if case in {"empty", "filtered"}:
            choice["message"]["content"] = ""
        if case == "invalid_summary":
            choice["message"]["content"] = "not valid summary JSON"
        if has_tools:
            choice["message"]["tool_calls"] = [
                {
                    "id": f"forbidden-{i}",
                    "function": {
                        "name": "forbidden_tool",
                        "arguments": ("{" if case == "tool_bad_arguments" else "{}"),
                    },
                }
                for i in range(65 if case == "tool_many" else 1)
            ]
        elif case in {"tool_empty", "function_empty"}:
            choice["message"]["tool_calls"] = []

        class Host:
            async def execute(self, name, arguments):
                operations.append((name, arguments))
                result = {"plan": {"revision": 1}, "objective": "Inspect", "retainedBytes": 0}
                failure = None
                if name == "after_compactor":
                    assert arguments["inputTokens"] == 13 and arguments["outputTokens"] == 4
                    assert arguments["finishReason"] == reason
                    assert arguments["toolCalls"] == int(has_tools)
                    assert arguments["cost"]["calls"] == 1
                    assert arguments["cost"]["groups"][0]["amount"] == "0.01"
                    if rejection:
                        failure = {
                            "code": "planner_compaction_failed",
                            "message": f"Compaction failed ({rejection})",
                            "retryable": False,
                        }
                    else:
                        result = {"summary": "Earlier work is complete."}
                if name == "model_failed":
                    pytest.fail(
                        "usable compactor outcome bypassed required classification/accounting"
                    )
                return PlannerOperationResult(
                    apiVersion=API_VERSION,
                    result=result,
                    stop=name == "finish",
                    escalate=name == "finish",
                    **({"failure": failure} if failure else {}),
                )

        async def gateway(request):
            requests.append(request)
            return httpx.Response(200, json=value, headers={"x-litellm-response-cost": "0.01"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            costs = CostCollector()
            client = new_gateway_client(
                base_url="https://gateway.invalid/v1",
                api_key="",
                timeout_seconds=10,
                http_client=http,
            )
            client.cost_observer = costs.observe
            compactor = OpenAICompatibleGatewayLlm(
                model="compactor-test-model", client_handle=client
            )
            steps = [tool_call("step", {}, call_id=f"step-{i}") for i in range(2)]
            for response in steps:
                response.usage_metadata.prompt_token_count = 100
                response.usage_metadata.total_token_count = 103
            main = scripted_model([*steps, tool_call("finish", {}, call_id="finish")])
            loop = PlannerLoop(
                main,
                Host(),
                planner_policy(),
                compactor=compactor,
                cost_snapshot=lambda: costs.snapshot(0),
            )
            try:
                if rejection:
                    with pytest.raises(PlannerStoppedError, match=rejection) as caught:
                        await loop.run(compaction_input(), "Inspect")
                    assert (
                        caught.value.code == "planner_compaction_failed"
                        and not caught.value.retryable
                    )
                else:
                    await loop.run(compaction_input(), "Inspect")
                assert len(requests) == 1
                names = [name for name, _ in operations]
                assert names.count("before_compactor") == names.count("after_compactor") == 1
                assert names.count("checkpoint") == int(rejection is None)
                assert "forbidden_tool" not in names
            finally:
                await compactor.close()

    asyncio.run(scenario())
