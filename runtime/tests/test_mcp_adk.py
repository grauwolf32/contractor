from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fakes.mcp import MCPFixture
from fakes.model import scripted_model, tool_call
from test_adk_runtime import (
    RecordingInstrumentation,
    create_runtime,
    stage_request,
    structured_result,
)
from test_mcp_lifecycle import assert_no_sdk_tasks
from test_mcp_session import SECRET, deadline
from test_mcp_toolset import prepared_tools

from contractor_runtime.allocation import WorkerState
from contractor_runtime.llm.client import new_gateway_client
from contractor_runtime.llm.openai import OpenAICompatibleGatewayLlm
from contractor_runtime.toolsets.common.deadlines import invocation_deadline


@pytest.mark.parametrize(
    ("case", "expected_error", "remote_calls"),
    [
        ("success", None, 1),
        ("business", None, 1),
        ("arguments", "invalid_arguments", 0),
        ("unsupported", "unsupported_content", 1),
        ("ambiguous", "outcome_unknown", 1),
        ("private", "private_content", 1),
    ],
)
def test_real_adk_gateway_declarations_call_envelope_accounting_and_finalizer(
    tmp_path, case, expected_error, remote_calls
):
    async def scenario():
        async with MCPFixture() as server, prepared_tools(server) as tools:
            if case == "business":
                server.result["isError"] = True
            elif case == "unsupported":
                server.result = {"content": [{"type": "future_unknown"}]}
            elif case == "ambiguous":
                server.drop_call = True
            elif case == "private":
                server.result = {"content": [{"type": "text", "text": SECRET}]}
            requests = []

            async def gateway(request):
                payload = json.loads(request.content)
                requests.append(payload)
                assert SECRET not in request.content.decode()
                assert "private-mcp-session" not in request.content.decode()
                if len(requests) == 1:
                    assert not server.calls  # Preparation spent no model call.
                    declarations = {
                        item["function"]["name"]: item["function"] for item in payload["tools"]
                    }
                    assert set(declarations) == {"search", "fetch"}
                    assert declarations["search"]["parameters"] == server.pages[0][0]["inputSchema"]
                    assert (
                        declarations["search"]["description"] == server.pages[0][0]["description"]
                    )
                    args = {"query": 123 if case == "arguments" else "safe query"}
                    message = {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "selected-call",
                                "type": "function",
                                "function": {"name": "search", "arguments": json.dumps(args)},
                            }
                        ],
                    }
                    finish = "tool_calls"
                elif len(requests) == 2:
                    messages = [m for m in payload["messages"] if m["role"] == "tool"]
                    assert len(messages) == 1
                    result = json.loads(messages[0]["content"])
                    if expected_error:
                        assert result["ok"] is False and result["error"]["code"] == expected_error
                    else:
                        assert result == server.result
                    message, finish = {"role": "assistant", "content": "Completed"}, "stop"
                else:
                    assert len(requests) == 3
                    assert not payload.get("tools")  # Existing result finalizer.
                    message, finish = (
                        {"role": "assistant", "content": '{"subtaskId":"0","result":"Completed"}'},
                        "stop",
                    )
                return httpx.Response(
                    200,
                    json={
                        "model": "worker-model",
                        "choices": [{"finish_reason": finish, "message": message}],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
                    },
                )

            async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as client:
                model = OpenAICompatibleGatewayLlm(
                    model="worker-model",
                    client_handle=new_gateway_client(
                        base_url="https://gateway.example/v1",
                        api_key="model-token",
                        timeout_seconds=10,
                        http_client=client,
                    ),
                )
                state = WorkerState()
                instrumentation = RecordingInstrumentation()
                runtime = await create_runtime(
                    tmp_path, state, tools, model, instrumentation=instrumentation
                )
                try:
                    result = await runtime.invoke(stage_request())
                    assert result.failure is None and result.result.result == "Completed"
                    observation = result.result.observations.tools["search"]
                    assert observation.calls == 1
                    assert observation.failures == (0 if case == "success" else 1)
                    report = state.metrics.build_report(report_id="mcp-worker", duration_ms=1)
                    assert report.metrics.worker_budget.observed_tool_calls == 1
                    assert len(server.calls) == remote_calls
                    assert len(requests) == 3
                    assert invocation_deadline.get() is None
                    assert SECRET not in report.model_dump_json()
                    spans = [
                        span
                        for span in instrumentation.spans
                        if span.name == "contractor.worker.tool"
                    ]
                    assert len(spans) == 1 and spans[0].outcome == (
                        "succeeded" if case == "success" else "failed"
                    )
                finally:
                    await runtime.finalize(datetime.now(UTC) + timedelta(seconds=2))
                    await model.close()

    asyncio.run(scenario())


def test_mcp_obeys_normal_tool_budget_without_dispatching_rejected_call(tmp_path):
    async def scenario():
        async with MCPFixture() as server, prepared_tools(server) as tools:
            model = scripted_model(
                [
                    tool_call("search", {"query": "first"}, call_id="first"),
                    tool_call("fetch", {"query": "second"}, call_id="second"),
                ]
            )
            state = WorkerState()
            runtime = await create_runtime(tmp_path, state, tools, model, max_tool_calls=1)
            result = await runtime.invoke(stage_request())
            assert result.failure.code == "worker_budget_exhausted"
            assert len(server.calls) == 1 and len(model.requests) == 2
            assert (
                state.metrics.build_report(
                    report_id="budget", duration_ms=1
                ).metrics.worker_budget.observed_tool_calls
                == 1
            )
            await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_mcp_summary_is_one_tool_free_call_after_completed_tool(tmp_path):
    async def scenario():
        async with MCPFixture() as server, prepared_tools(server) as tools:
            normal = scripted_model([tool_call("search", {"query": "first"}, call_id="first")])
            summary = scripted_model([structured_result("Summary")], model="worker-summary-model")
            runtime = await create_runtime(
                tmp_path, WorkerState(), tools, normal, summary_model=summary, cumulative_budget=10
            )
            result = await runtime.invoke(stage_request())
            assert result.result.summarized is True
            assert result.result.observations.tools["search"].calls == 1
            assert len(normal.requests) == len(summary.requests) == len(server.calls) == 1
            assert summary.requests[0]["toolNames"] == []
            assert SECRET not in summary.requests[0]["contentText"]
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_cancellation_cancels_mcp_without_replay_and_joins_its_tasks(tmp_path):
    async def scenario():
        async with MCPFixture() as server, prepared_tools(server) as tools:
            server.block_method = "tools/call"
            model = scripted_model([tool_call("search", {"query": "cancel"}, call_id="cancel")])
            runtime = await create_runtime(tmp_path, WorkerState(), tools, model)
            invocation = asyncio.create_task(runtime.invoke(stage_request()))
            await asyncio.wait_for(server.entered.wait(), 2)
            await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))
            await asyncio.gather(invocation, return_exceptions=True)
            await tools["search"].close_before(deadline=deadline())
            assert tools["search"]._owner.closed and len(server.calls) == 1
            assert_no_sdk_tasks()

    asyncio.run(scenario())
