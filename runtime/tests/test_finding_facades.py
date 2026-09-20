from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from google.adk.tools import FunctionTool
from google.genai import types
from test_http_toolset import FakeArtifactClient, create_tools
from test_security_findings_toolset import FakeFindingClient, _settings, _workspace

from contractor_runtime.allocation import WorkerState
from contractor_runtime.llm.openai import _tools
from contractor_runtime.toolsets.common.input_errors import ToolInputError
from contractor_runtime.toolsets.http.limits import MAX_HISTORY
from contractor_runtime.toolsets.security_findings.collection import _proposal
from contractor_runtime.toolsets.security_findings.facades import (
    CodeFindingsToolsetFactory,
    CodeFindingTool,
    GeneralFindingsToolsetFactory,
    GeneralFindingTool,
    HTTPFindingsToolsetFactory,
    HTTPFindingTool,
)
from contractor_runtime.toolsets.security_findings.locations import normalize_locations

LOCATION_CASES = json.loads(
    (
        Path(__file__).parents[2] / "internal/auditdomain/testdata/finding-locations.json"
    ).read_text()
)


@pytest.mark.parametrize("case", LOCATION_CASES, ids=lambda case: case["name"])
def test_locations_match_go_contract(case):
    if case["valid"]:
        assert normalize_locations([case["location"]]) == [case["location"]]
    else:
        with pytest.raises(ValueError):
            normalize_locations([case["location"]])


@pytest.mark.parametrize(
    "tool_type,required",
    [
        (GeneralFindingTool, {"title", "description"}),
        (CodeFindingTool, {"title", "description", "file"}),
        (HTTPFindingTool, {"title", "description", "url", "method"}),
    ],
)
def test_actual_provider_schema(tool_type, required):
    declaration = FunctionTool(tool_type(None))._get_declaration()
    wire = _tools([types.Tool(function_declarations=[declaration])])[0]["function"]
    assert wire["name"] == "finding"
    schema = wire["parameters"]
    assert set(schema["required"]) == required
    assert "tool_context" not in schema["properties"]
    evidence = schema["$defs"]["ExactEvidenceRef"]
    assert set(evidence["required"]) == {"namespace", "name", "revision"}
    assert evidence["additionalProperties"] is False
    if tool_type is GeneralFindingTool:
        assert schema["$defs"]["SourceLocation"]["additionalProperties"] is False
        assert schema["$defs"]["WebLocation"]["additionalProperties"] is False


async def finding_tool(factory_type, client, state):
    tools = await factory_type(lambda _allocation, _settings: client).create_selected(
        selected=["finding"],
        allocation_id="allocation-test",
        run_id="run-test",
        namespace="worker",
        runtime_settings=_settings(),
        workspace=_workspace(),
        state=state,
    )
    return tools["finding"]


def context(call_id="call-1", invocation_id="invocation-1"):
    return SimpleNamespace(invocation_id=invocation_id, function_call_id=call_id)


def test_code_call_normalizes_coordinates_and_keeps_retry_identity():
    async def scenario():
        client = FakeFindingClient()
        tool = await finding_tool(CodeFindingsToolsetFactory, client, WorkerState())
        adk = FunctionTool(tool)
        args = {
            "title": "Ownership guard missing",
            "description": "Source observation",
            "file": "src/order.py",
            "range": {"start_line": 10, "end_line": 12},
            "cwe": "CWE-639",
            "evidence_refs": [{"namespace": "worker", "name": "proof", "revision": "r1"}],
            "standard_refs": [
                {"scheme": "OWASP-ASVS", "version": "5.0.0", "requirement_id": "v5.0.0-1.1.1"}
            ],
        }
        await adk.run_async(args=args, tool_context=context())
        await adk.run_async(args=args, tool_context=context())
        assert client.requests[0] == client.requests[1]
        document = client.requests[0]["proposal"]
        assert document["subject"] is None
        assert document["locations"] == [{"file": "src/order.py", "range": args["range"]}]
        assert any(ref["requirement_id"] == "CWE-639" for ref in document["standard_refs"])
        assert client.requests[0]["evidenceRefs"] == args["evidence_refs"]
        assert _proposal(json.dumps(document).encode()) == document
        await adk.run_async(args=args, tool_context=context("call-2"))
        assert client.requests[2]["submissionId"] != client.requests[0]["submissionId"]
        with pytest.raises(ToolInputError, match="identity"):
            await adk.run_async(args=args, tool_context=context(None))

    asyncio.run(scenario())


def test_http_evidence_captures_actual_request_and_survives_history_eviction(tmp_path):
    async def scenario():
        observed = []

        async def handler(request):
            observed.append(request)
            if request.url.path == "/start":
                return httpx.Response(
                    303, headers={"Location": "https://other.example/end"}, request=request
                )
            return httpx.Response(200, text="observed response", request=request)

        artifacts = FakeArtifactClient()
        http, state = await create_tools(tmp_path, handler, artifacts=artifacts)
        client = FakeFindingClient()
        tool = await finding_tool(HTTPFindingsToolsetFactory, client, state)
        await http["http_session_set"](
            auth={"kind": "bearer", "token": "capture-secret"}, cookies={"sid": "cookie-secret"}
        )
        result = await http["http_request"](
            "https://target.example/start?a=%2f&a=2",
            method="POST",
            body_type="text",
            body="request body",
            tool_context=context(),
        )
        await tool(
            title="Finding",
            description="Observed behavior",
            url="https://target.example/start?a=%2f&a=2",
            method="POST",
            request_id=result["request_id"],
            tool_context=context(),
        )
        document = client.requests[0]["proposal"]
        assert _proposal(json.dumps(document).encode()) == document
        attempts = document["http_exchange"]["attempts"]
        assert len(attempts) == len(observed) == 2
        for captured, actual in zip(attempts, observed, strict=True):
            assert base64.b64decode(captured["body_base64"]) == actual.content
            assert captured["url"] == str(actual.url)
            assert [(row["name"], row["value"]) for row in captured["headers"]] == [
                (name.decode("ascii"), value.decode("latin-1"))
                for name, value in actual.headers.raw
            ]
        assert attempts[1]["method"] == "GET"
        assert not any(
            row["name"].lower() in {"authorization", "cookie"} for row in attempts[1]["headers"]
        )
        assert document["http_exchange"]["response_body_evidence_id"] == "evidence-1"
        assert client.requests[0]["evidenceRefs"] == [result["body_artifact"]]
        assert artifacts.writes == 1  # The existing response body; no request artifacts.
        assert "capture-secret" not in repr(state.metrics)
        assert "cookie-secret" not in repr(await http["http_history"]())

        with pytest.raises(ToolInputError, match="unavailable"):
            await tool(
                title="F",
                description="D",
                url="https://target.example",
                method="GET",
                request_id=result["request_id"],
                tool_context=context(invocation_id="other"),
            )
        retained = json.dumps(document)
        for _ in range(MAX_HISTORY):
            await http["http_request"]("https://target.example/later", tool_context=context())
        assert len(await http["http_history"]()) == MAX_HISTORY
        with pytest.raises(ToolInputError, match="unavailable"):
            await tool(
                title="F",
                description="D",
                url="https://target.example",
                method="GET",
                request_id=result["request_id"],
                tool_context=context("call-3"),
            )
        await tool(
            title="F",
            description="D",
            url="https://target.example",
            method="GET",
            tool_context=context("call-4"),
        )
        assert "http_exchange" not in client.requests[-1]["proposal"]
        await http["http_session_clear"]()
        await http["http_request"].close()
        assert json.dumps(document) == retained

    asyncio.run(scenario())


def test_general_finding_can_omit_location_and_classification():
    async def scenario():
        client = FakeFindingClient()
        tool = await finding_tool(GeneralFindingsToolsetFactory, client, WorkerState())
        await tool(
            title="Observation", description="Unknown affected subject", tool_context=context()
        )
        document = client.requests[0]["proposal"]
        assert "locations" not in document
        assert document["subject"] is None
        assert document["standard_refs"] == []

    asyncio.run(scenario())
