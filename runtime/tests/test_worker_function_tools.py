"""ADK dispatch preserves raw domain values and logical business context."""

from __future__ import annotations

import asyncio
import functools
from types import SimpleNamespace
from typing import Annotated

import pytest
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel

from contractor_runtime.worker.function_tools import ContractorFunctionTool


class Payload(BaseModel):
    name: str


def annotated_probe(
    value: Annotated[int | None, "raw"], values: list[Annotated[int, "raw"]], payload: Payload
) -> dict[str, object]:
    """Observe original numeric values and an SDK-converted structured payload."""
    return {"value": value, "values": values, "payload": payload}


class CallableProbe:
    def __call__(self, value: int | None, values: list[int], payload: Payload) -> dict[str, object]:
        """Observe callable-instance arguments."""
        return {"value": value, "values": values, "payload": payload}


@pytest.mark.parametrize("kind", ["function", "method", "instance", "partial"])
@pytest.mark.parametrize("value", [0.0, 2.5, True, "1", None, 1])
def test_function_tool_keeps_raw_numbers_and_native_structured_conversion(
    kind: str, value: object
) -> None:
    async def scenario() -> None:
        instance = CallableProbe()
        func = {
            "function": annotated_probe,
            "method": instance.__call__,
            "instance": instance,
            "partial": functools.partial(annotated_probe, payload=Payload(name="bound")),
        }[kind]
        tool = ContractorFunctionTool(func, logical_invocation_id=lambda _: "logical")
        args = {"value": value, "values": [1.0, 2.5], "payload": {"name": "structured"}}
        result = await tool.run_async(args=args, tool_context=SimpleNamespace(invocation_id="sdk"))
        assert type(result["value"]) is type(value)
        assert result["value"] == value
        assert result["values"] == [1.0, 2.5]
        assert all(type(item) is float for item in result["values"])
        assert isinstance(result["payload"], Payload)
        assert result["payload"].name == "structured"
        assert args["payload"] == {"name": "structured"}

    asyncio.run(scenario())


@pytest.mark.parametrize("sync", [False, True])
def test_business_tool_gets_logical_context_with_original_state_and_actions(sync: bool) -> None:
    observed: list[object] = []

    def call(tool_context: ToolContext) -> dict[str, bool]:
        """Use business identity and the same mutable state/actions surface."""
        observed.append(tool_context.invocation_id)
        observed.append(tool_context.state)
        tool_context.actions.skip_summarization = True
        return {"ok": True}

    async def async_call(tool_context: ToolContext) -> dict[str, bool]:
        """Observe business context from an async callable."""
        return call(tool_context)

    async def scenario() -> None:
        context = SimpleNamespace(
            invocation_id="sdk-turn-2",
            state={"owned": True},
            actions=SimpleNamespace(skip_summarization=False),
        )
        tool = ContractorFunctionTool(
            call if sync else async_call, logical_invocation_id=lambda ctx: "logical-worker"
        )
        assert await tool.run_async(args={}, tool_context=context) == {"ok": True}
        assert observed[0] == "logical-worker"
        assert observed[1] is context.state
        assert context.invocation_id == "sdk-turn-2"
        assert context.actions.skip_summarization is True
        declaration = tool._get_declaration()
        assert declaration is not None
        schema = declaration.parameters
        properties = (
            schema.properties or {}
            if schema is not None
            else (declaration.parameters_json_schema or {}).get("properties", {})
        )
        assert "tool_context" not in properties

    asyncio.run(scenario())


def test_stale_context_rejection_prevents_callable_side_effect() -> None:
    effects: list[bool] = []

    async def probe() -> dict[str, bool]:
        """Record one admitted effect."""
        effects.append(True)
        return {"ok": True}

    def stale(_context: object) -> str:
        raise RuntimeError("Stale SDK context")

    async def scenario() -> None:
        tool = ContractorFunctionTool(probe, logical_invocation_id=stale)
        with pytest.raises(RuntimeError, match="Stale SDK context"):
            await tool.run_async(args={}, tool_context=SimpleNamespace(invocation_id="stale"))
        assert effects == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "arguments",
    [
        {"methdo": "POST"},
        {"tool_context": {"invocation_id": "injected"}},
        {"payload": {"name": "valid", "secret-extra": "never-echo-this"}},
        {"rows": [{"name": "valid", "extra": True}]},
        {"optional": {"name": "valid", "extra": True}},
        {"mapping": {"arbitrary": {"name": "valid", "extra": True}}},
    ],
)
def test_undeclared_fields_fail_before_sdk_conversion_or_callable(arguments) -> None:
    from contractor_runtime.core.input_errors import ToolInputError

    effects = []

    def probe(
        method: str = "GET",
        payload: Payload | None = None,
        rows: list[Payload] | None = None,
        optional: Payload | str | None = None,
        mapping: dict[str, Payload] | None = None,
        headers: dict[str, str] | None = None,
    ):
        """Observe defaults, structured records and arbitrary header names."""
        effects.append(method)
        return method

    async def scenario():
        tool = ContractorFunctionTool(probe, logical_invocation_id=lambda _: "logical")
        with pytest.raises(ToolInputError, match="undeclared field") as caught:
            await tool.run_async(args=arguments, tool_context=SimpleNamespace(invocation_id="sdk"))
        assert "never-echo-this" not in str(caught.value)
        assert "secret-extra" not in str(caught.value)
        assert not effects
        assert (
            await tool.run_async(
                args={"headers": {"X-Any": "value"}},
                tool_context=SimpleNamespace(invocation_id="sdk"),
            )
            == "GET"
        )
        assert effects == ["GET"]
        schema = tool._get_declaration().parameters_json_schema
        assert schema["additionalProperties"] is False
        assert schema["$defs"]["Payload"]["additionalProperties"] is False

    asyncio.run(scenario())


def test_callable_infrastructure_failure_keeps_its_exception() -> None:
    async def probe():
        """Raise an infrastructure failure."""
        raise RuntimeError("infrastructure unavailable")

    async def scenario():
        tool = ContractorFunctionTool(probe, logical_invocation_id=lambda _: "logical")
        with pytest.raises(RuntimeError, match="infrastructure unavailable"):
            await tool.run_async(args={}, tool_context=SimpleNamespace(invocation_id="sdk"))

    asyncio.run(scenario())


def test_worker_returns_repairable_error_and_executes_only_corrected_call(tmp_path) -> None:
    from datetime import UTC, datetime, timedelta

    import httpx
    from fakes.model import scripted_model, text_result, tool_call
    from fakes.worker_state import WorkerState
    from test_adk_runtime import create_runtime, stage_request

    requests = []

    async def transport(request):
        requests.append(request)
        return httpx.Response(200, text="ok")

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:

            async def send(method: str = "GET"):
                """Send one fixture request with the supplied method."""
                response = await client.request(method, "https://fixture.test/")
                return {"status": response.status_code}

            model = scripted_model(
                [
                    tool_call("send", {"methdo": "POST"}, call_id="typo"),
                    tool_call("send", {"method": "POST"}, call_id="corrected"),
                    text_result("Completed"),
                ]
            )
            runtime = await create_runtime(tmp_path, WorkerState(), {"send": send}, model)
            try:
                result = await runtime.invoke(
                    stage_request().model_copy(update={"result_artifacts": {}})
                )
                assert result.failure is None
                assert len(requests) == 1 and requests[0].method == "POST"
                session = await runtime._session_service.get_session(
                    app_name=runtime._app_name,
                    user_id=runtime._user_id,
                    session_id=runtime._session_lifecycle.shared_session_id,
                )
                responses = [
                    part.function_response.response
                    for event in session.events
                    for part in (event.content.parts if event.content else []) or []
                    if part.function_response is not None
                ]
                assert responses[0]["error"]["code"] == "tool_input_invalid"
                assert responses[1] == {"status": 200}
            finally:
                await runtime.finalize(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())
