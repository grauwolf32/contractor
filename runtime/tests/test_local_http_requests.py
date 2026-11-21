"""Local request selectors resolve mixed transports without changing saved evidence."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from google.adk.tools import FunctionTool
from http_cookie_store import HTTPToolsetFactory
from test_caido_read_tools import (
    FakeArtifactClient,
    close_tools,
    create_tools,
    representative_responses,
)
from test_finding_facades import finding_tool
from test_http_toolset import make_tools
from test_security_findings_toolset import FakeFindingClient

from contractor_runtime.contracts.base import MAX_EXACT_INTEGER
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.toolsets.caido.tools import CaidoToolError
from contractor_runtime.toolsets.common.http_capture import RecentHTTPCaptures
from contractor_runtime.toolsets.security_findings.facades import (
    GeneralFindingsToolsetFactory,
    HTTPFindingsToolsetFactory,
    RecordFactTool,
)


def test_local_locator_bound_nonreuse_wrong_source_and_exhaustion():
    captures = RecentHTTPCaptures()
    first = captures.local_id("caido", "1")
    http_id = captures.new_http_request()
    assert first != http_id
    assert captures.resolve(first) == ("caido", "1")
    assert captures.resolve(http_id) == ("http", str(http_id))
    with pytest.raises(ValueError):
        captures.resolve(http_id, source="caido")
    for index in range(200):
        captures.local_id("caido", f"provider-{index}")
    assert len(captures._local_ids) == len(captures._locators) == 128
    with pytest.raises(ValueError):
        captures.resolve(first)
    replacement = captures.local_id("caido", "1")
    assert replacement > http_id and replacement != first
    assert captures.local_id("caido", "1") == replacement
    captures.clear("caido")
    with pytest.raises(ValueError):
        captures.resolve(replacement)
    captures._next_local_id = MAX_EXACT_INTEGER + 1
    with pytest.raises(ValueError, match="exhausted"):
        captures.new_http_request()


@pytest.mark.parametrize("facade", ["general", "http", "fact"])
def test_mixed_local_ids_discovery_inspection_and_publication(tmp_path, facade):
    async def scenario():
        responses = representative_responses(
            b"GET /caido HTTP/1.1\r\nHost: target.example\r\n\r\n",
            b"HTTP/1.1 200 OK\r\n\r\ncaido",
        )
        responses["RequestsByOffset"]["requestsByOffset"]["nodes"][0]["id"] = "1"
        responses["RequestDetail"]["request"]["id"] = "1"
        operations = []

        async def caido_handler(request):
            payload = json.loads(request.content)
            operations.append(payload)
            return httpx.Response(200, json={"data": responses[payload["operationName"]]})

        caido, state, handle = await create_tools(tmp_path, caido_handler, FakeArtifactClient())
        http = await make_tools(
            HTTPToolsetFactory(
                lambda: httpx.AsyncClient(
                    transport=httpx.MockTransport(lambda _: httpx.Response(200, text="http"))
                )
            ),
            tmp_path,
            state=state,
        )
        context = SimpleNamespace(invocation_id="invocation-mixed", function_call_id="mixed")
        client = FakeFindingClient()
        factory = HTTPFindingsToolsetFactory if facade == "http" else GeneralFindingsToolsetFactory
        finding = await finding_tool(factory, client, state)
        if facade == "fact":
            finding = RecordFactTool(finding._publisher)
        args = {"title": "Observed", "description": "Transport observation"}
        if facade == "http":
            args.update(url="https://target.example/", method="GET")
        schema = FunctionTool(finding)._get_declaration().parameters_json_schema
        assert "caido_request_id" not in schema["properties"]
        try:
            sent = await http["http_request"]("https://target.example/", tool_context=context)
            discovered = await caido["caido_history"]()
            selected = discovered["requests"][0]["request_id"]
            assert selected != sent["request_id"]
            assert "id" not in discovered["requests"][0]
            before = len(operations)
            with pytest.raises(ToolInputError, match="unavailable"):
                await finding(**args, request_id=selected, tool_context=context)
            with pytest.raises(CaidoToolError):
                await caido["caido_request_detail"](sent["request_id"], tool_context=context)
            with pytest.raises(CaidoToolError):
                await caido["caido_request_detail"]("1", tool_context=context)
            assert len(operations) == before and not client.requests
            detail = await caido["caido_request_detail"](selected, tool_context=context)
            assert detail["request_id"] == selected
            assert operations[-1]["variables"] == {"id": "1"}
            before = len(operations)
            await finding(**args, request_id=selected, tool_context=context)
            assert client.requests[-1]["httpEvidence"]["source"] == "caido"
            assert client.requests[-1]["httpEvidence"]["request_id"] == "1"
            context.function_call_id = "http-second"
            await finding(**args, request_id=sent["request_id"], tool_context=context)
            assert client.requests[-1]["httpEvidence"]["source"] == "http"
            assert len(operations) == before
            context.invocation_id = "foreign"
            with pytest.raises(ToolInputError, match="unavailable"):
                await finding(**args, request_id=selected, tool_context=context)
        finally:
            await close_tools(http)
            await close_tools(caido)
            await handle.close()

    asyncio.run(scenario())
