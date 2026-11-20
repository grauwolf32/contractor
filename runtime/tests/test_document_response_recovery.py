import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fakes.model import scripted_model, text_result, thought_result
from fakes.worker_state import WorkerState
from google.genai import types
from test_adk_runtime import create_runtime, stage_request


@pytest.mark.parametrize("kind", ["empty", "space", "thought"])
@pytest.mark.parametrize("repeated", [False, True])
def test_document_worker_recovers_once(tmp_path, kind, repeated):
    async def scenario():
        def blank():
            response = (
                thought_result("private")
                if kind == "thought"
                else text_result(" " if kind == "space" else "")
            )
            response.finish_reason = types.FinishReason.STOP
            return response

        model = scripted_model([blank(), blank() if repeated else text_result("Complete document")])
        state = WorkerState()
        runtime = await create_runtime(tmp_path, state, {}, model, response_recovery="document@1")
        try:
            result = await runtime.invoke(stage_request())
            if repeated:
                assert result.failure.code == "worker_result_missing"
                assert len(model.requests) == 2
            else:
                assert result.failure is None
                assert len(model.requests) == 3  # two main calls and normal finalizer
            snapshot = await state.snapshot()
            metrics = snapshot["lastCompletedInvocation"]["metrics"]
            assert metrics["modelCalls"] == len(model.requests)
            assert metrics["totalTokens"] == 10 * len(model.requests)
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_document_recovery_honors_original_model_budget(tmp_path):
    async def scenario():
        blank = text_result("")
        blank.finish_reason = types.FinishReason.STOP
        model = scripted_model([blank])
        state = WorkerState()
        runtime = await create_runtime(
            tmp_path, state, {}, model, response_recovery="document@1", max_model_calls=1
        )
        try:
            result = await runtime.invoke(stage_request())
            assert result.failure.code == "worker_budget_exhausted"
            assert len(model.requests) == 1
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_shared_model_response_recovery():
    import json
    from pathlib import Path

    import httpx
    from google.adk.models.llm_request import LlmRequest

    from contractor_runtime.llm.client import new_gateway_client
    from contractor_runtime.llm.openai import GatewayModelError, OpenAICompatibleGatewayLlm
    from contractor_runtime.llm.response import empty_stop_response, model_response_failed

    cases = json.loads(
        (
            Path(__file__).parents[2] / "api" / "testdata" / "model-response-recovery.json"
        ).read_text()
    )

    async def scenario(case):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=case["response"]))
        ) as http:
            model = OpenAICompatibleGatewayLlm(
                model="test",
                client_handle=new_gateway_client(
                    base_url="https://gateway.example/v1",
                    api_key="fixture",
                    timeout_seconds=10,
                    http_client=http,
                ),
            )
            empty, accepted = False, False
            try:
                replies = [r async for r in model.generate_content_async(LlmRequest())]
                empty = empty_stop_response(replies[0])
                accepted = not empty and not model_response_failed(replies[0])
            except GatewayModelError as error:
                empty = error.empty_stop
            assert (empty, accepted) == (case["emptyStop"], case["accepted"]), case["name"]

    for case in cases:
        asyncio.run(scenario(case))


def test_document_recovery_cancellation_joins_second_call(tmp_path):
    async def scenario():
        blank = text_result("")
        blank.finish_reason = types.FinishReason.STOP
        model = scripted_model([blank, text_result("blocked")])
        model.block_call_number = 2
        runtime = await create_runtime(
            tmp_path, WorkerState(), {}, model, response_recovery="document@1"
        )
        try:
            task = asyncio.create_task(runtime.invoke(stage_request()))
            await asyncio.wait_for(model.blocked.wait(), 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(model.requests) == 2
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_shared_document_response_policy():
    import json
    from pathlib import Path

    from fakes.spec import allocation_spec

    from contractor_runtime.contracts import ResolvedAgentTemplate

    cases = json.loads(
        (
            Path(__file__).parents[2] / "api" / "testdata" / "document-response-policy.json"
        ).read_text()
    )
    for case in cases:
        value = allocation_spec().agent_template.model_dump(by_alias=True, exclude_none=True)
        toolset, version = case["ref"].split("@")
        value.update(
            responseRecovery=case["mode"],
            toolsets=[{"ref": {"toolsetId": toolset, "version": version}, "tools": case["tools"]}],
        )
        value["runtime"]["runtimeId"] = case["runtime"]
        if case["valid"]:
            ResolvedAgentTemplate.model_validate(value)
        else:
            with pytest.raises(ValueError):
                ResolvedAgentTemplate.model_validate(value)
