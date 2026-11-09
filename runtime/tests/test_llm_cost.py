from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from contractor_runtime.llm.client import new_gateway_client
from contractor_runtime.llm.cost import CostCollector, model_cost_phase
from contractor_runtime.llm.openai import GatewayModelError, OpenAICompatibleGatewayLlm


def test_physical_retries_and_auxiliary_phases_keep_cost_once(monkeypatch: pytest.MonkeyPatch):
    async def scenario():
        calls = 0
        closed = []

        async def send(request):
            nonlocal calls
            calls += 1
            response = httpx.Response(
                503 if calls == 1 else 200,
                headers={"llm_provider-x-litellm-response-cost": "0.00000625"},
                json={"usage": {"cost": 0.00000625}, "choices": []},
            )
            closed.append(response)
            return response

        async def no_delay(_):
            return None

        monkeypatch.setattr("contractor_runtime.llm.client.sleep", no_delay)
        collector = CostCollector()
        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            handle = new_gateway_client(
                base_url="http://gateway.test/v1",
                api_key="secret-canary",
                timeout_seconds=10,
                http_client=client,
            )
            handle.cost_observer = collector.observe
            payload = {"model": "openrouter/test", "messages": [{"role": "user", "content": "x"}]}
            await handle.complete(payload)
            for phase in ("compaction", "finalization", "summarization"):
                with model_cost_phase(phase):
                    await handle.complete(payload)
            await handle.close()
        cost = collector.snapshot(0)
        assert cost.calls == 5
        assert sum(g.calls for g in cost.groups) == 5
        assert {g.phase for g in cost.groups} == {
            "main",
            "compaction",
            "finalization",
            "summarization",
        }
        assert next(g for g in cost.groups if g.phase == "main").amount == "0.0000125"
        assert all(response.is_closed for response in closed)
        assert "secret-canary" not in cost.model_dump_json()

    asyncio.run(scenario())


def test_rejected_adk_response_keeps_cost_without_inventing_output():
    async def scenario():
        collector = CostCollector()

        async def send(request):
            return httpx.Response(200, json={"usage": {"cost": 0.1}, "choices": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            handle = new_gateway_client(
                base_url="http://gateway.test/v1",
                api_key=None,
                timeout_seconds=10,
                http_client=client,
            )
            handle.cost_observer = collector.observe
            model = OpenAICompatibleGatewayLlm(model="openrouter/test", client_handle=handle)
            request = LlmRequest(
                contents=[types.Content(role="user", parts=[types.Part(text="x")])]
            )
            with pytest.raises(GatewayModelError):
                async for _ in model.generate_content_async(request):
                    pytest.fail("invalid model output accepted")
            await model.close()
        assert collector.snapshot(0).groups[0].amount == "0.1"
        assert collector.snapshot(0).calls == 1

    asyncio.run(scenario())


def test_cost_bounds_missing_and_snapshot_isolation():
    collector = CostCollector()
    for i in range(65):
        collector.observe(f"model-{i}", {"x-litellm-response-cost": "1e-18"}, b"{}")
    snapshot = collector.snapshot(0)
    assert snapshot.calls == 65
    assert len(snapshot.groups) == 64
    assert sum(g.calls for g in snapshot.groups) == 64
    snapshot.groups[0].amount = "99"
    assert collector.snapshot(0).groups[0].amount == "0.000000000000000001"
    collector.observe("model-0", {"x-litellm-response-cost": "Infinity"}, b"{}")
    assert collector.snapshot(0).calls == 66
    assert sum(g.calls for g in collector.snapshot(0).groups) == 64
    assert "body" not in json.loads(collector.snapshot(0).model_dump_json())


def test_cancelled_response_keeps_observed_header_and_closes_stream():
    async def scenario():
        entered = asyncio.Event()
        closed = asyncio.Event()

        class PausedStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                await asyncio.Event().wait()
                yield b"{}"

            async def aclose(self):
                closed.set()

        async def send(request):
            return httpx.Response(
                200,
                headers={"llm_provider-x-litellm-response-cost": "0.1"},
                stream=PausedStream(),
            )

        collector = CostCollector()
        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            handle = new_gateway_client(
                base_url="http://gateway.test/v1",
                api_key=None,
                timeout_seconds=10,
                http_client=client,
            )
            handle.cost_observer = collector.observe
            task = asyncio.create_task(handle.complete({"model": "selected-model"}))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert closed.is_set()
            assert collector.snapshot(0).calls == 1
            assert collector.snapshot(0).groups[0].amount == "0.1"
            await handle.close()

    asyncio.run(scenario())
