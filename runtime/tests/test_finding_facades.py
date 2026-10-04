from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google.adk.tools import FunctionTool
from google.genai import types
from test_audit_result_collector import CONTEXT, fixture_assignment, tool_for
from test_http_toolset import FakeArtifactClient, close_tools, create_tools, make_tools
from test_security_findings_toolset import FakeFindingClient, _settings, _workspace

from contractor_runtime.allocation import WorkerState
from contractor_runtime.contracts import HTTPOriginTargetSettings, RuntimeSettings
from contractor_runtime.llm.openai import _tools
from contractor_runtime.toolsets.audit_results.collector import AuditCollectionError
from contractor_runtime.toolsets.common.input_errors import ToolInputError
from contractor_runtime.toolsets.http.limits import MAX_HISTORY, MAX_REQUEST_HEADER_BYTES
from contractor_runtime.toolsets.http.tools import HTTPToolError, HTTPToolsetFactory
from contractor_runtime.toolsets.security_findings.classification import CWEReferenceError
from contractor_runtime.toolsets.security_findings.collection import _proposal
from contractor_runtime.toolsets.security_findings.facades import (
    CodeFindingsToolsetFactory,
    CodeFindingTool,
    GeneralFindingsToolsetFactory,
    GeneralFindingTool,
    HTTPFindingsToolsetFactory,
    HTTPFindingTool,
)
from contractor_runtime.toolsets.security_findings.http_evidence import HTTPAttempt
from contractor_runtime.toolsets.security_findings.locations import (
    StandardReference,
    normalize_locations,
)
from contractor_runtime.toolsets.security_findings.publisher import FindingPublisher
from contractor_runtime.worker.instrumentation import _safe_tool_response

LOCATION_CASES = json.loads(
    (Path(__file__).parents[2] / "internal/auditdomain/testdata/finding-locations.json").read_text()
)
HTTP_ATTEMPT_CASES = json.loads(
    (
        Path(__file__).parents[2] / "internal/auditdomain/testdata/finding-http-attempts.json"
    ).read_text()
)


@pytest.mark.parametrize("case", HTTP_ATTEMPT_CASES, ids=lambda case: case["name"])
def test_runtime_and_server_share_http_attempt_rules(case: dict[str, Any]) -> None:
    attempt = {
        "method": "GET",
        "url": case["url"] + case.get("repeat", "") * case.get("count", 0),
        "headers": [],
        "body_base64": "",
        "status": case["status"],
        "response_headers": [],
    }
    if case["valid"]:
        HTTPAttempt.model_validate(attempt)
    else:
        with pytest.raises(ValueError):
            HTTPAttempt.model_validate(attempt)


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


def test_code_finding_rejects_deprecated_cwe_before_submission():
    async def scenario():
        client = FakeFindingClient()
        state = WorkerState()
        tool = await finding_tool(CodeFindingsToolsetFactory, client, state)
        with pytest.raises(ToolInputError, match="pinned CWE catalog"):
            await FunctionTool(tool).run_async(
                args={
                    "title": "Debug log files exposed",
                    "description": "Source observation",
                    "file": "src/log.py",
                    "cwe": "CWE-534",
                },
                tool_context=context(),
            )
        assert client.requests == []
        assert state.metrics.counters["tool_errors"] == 1

    asyncio.run(scenario())


VERSION_REPAIR = "standard_refs: Use version 4.20 of the pinned CWE catalog"
WEAKNESS_REPAIR = "standard_refs: Use a non-deprecated weakness ID from the pinned CWE catalog"


@pytest.mark.parametrize(
    "factory_type,location_args",
    [
        (GeneralFindingsToolsetFactory, {}),
        (CodeFindingsToolsetFactory, {"file": "src/order.py"}),
        (HTTPFindingsToolsetFactory, {"url": "https://target.example", "method": "GET"}),
    ],
)
@pytest.mark.parametrize(
    "reference,repair_text",
    [
        ({"scheme": "CWE", "version": "4.20", "requirement_id": "CWE-534"}, WEAKNESS_REPAIR),
        ({"scheme": "CWE", "version": "4.20", "requirement_id": "CWE-999999"}, WEAKNESS_REPAIR),
        ({"scheme": "CWE", "version": "4.9", "requirement_id": "CWE-79"}, VERSION_REPAIR),
    ],
    ids=["deprecated", "unknown", "other-version"],
)
def test_cwe_standard_refs_outside_the_pinned_catalog_fail_before_submission(
    factory_type, location_args, reference, repair_text
):
    async def scenario():
        client = FakeFindingClient()
        state = WorkerState()
        tool = FunctionTool(await finding_tool(factory_type, client, state))
        args = {"title": "Candidate", "description": "Observation", **location_args}
        with pytest.raises(ToolInputError) as failure:
            await tool.run_async(
                args={**args, "standard_refs": [reference]}, tool_context=context()
            )
        reply = _safe_tool_response("finding", failure.value)["error"]
        assert reply["code"] == "tool_input_invalid" and reply["retryable"] is False
        assert reply["message"].startswith(repair_text)
        assert client.requests == []
        assert state.metrics.counters["tool_errors"] == 1

        valid = {"scheme": "CWE", "version": "4.20", "requirement_id": "CWE-639"}
        await tool.run_async(
            args={**args, "cwe": "CWE-639", "standard_refs": [valid]},
            tool_context=context("call-2"),
        )
        assert client.requests[0]["proposal"]["standard_refs"] == [valid]

    asyncio.run(scenario())


def test_audit_and_ordinary_findings_share_the_cwe_reference_check():
    async def scenario():
        collector, _, _ = tool_for(fixture_assignment("standard", "requirements-verification"))
        audit_state = WorkerState()
        audit_state.audit_completion_binding = SimpleNamespace(current=lambda: collector)
        reference = StandardReference(scheme="CWE", version="4.20", requirement_id="CWE-534")
        failures = []
        for state in (audit_state, WorkerState()):
            client = FakeFindingClient()
            finding = GeneralFindingTool(FindingPublisher(client, state.metrics, (), state))
            with pytest.raises(ToolInputError) as failure:
                await finding(
                    title="Candidate",
                    description="Observed weakness.",
                    standard_refs=[reference],
                    tool_context=context(invocation_id=CONTEXT.invocation_id),
                )
            assert client.requests == []
            failures.append(failure.value)
        audit, ordinary = failures
        assert isinstance(audit, AuditCollectionError) and audit.field == "standard_refs"
        assert isinstance(ordinary, CWEReferenceError)
        assert audit.reason == ordinary.reason and str(audit) == str(ordinary)
        assert str(ordinary).startswith(WEAKNESS_REPAIR)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "factory_type,location_args",
    [
        (GeneralFindingsToolsetFactory, {}),
        (CodeFindingsToolsetFactory, {"file": "src/order.py"}),
        (HTTPFindingsToolsetFactory, {"url": "https://target.example", "method": "GET"}),
    ],
)
@pytest.mark.parametrize(
    "field,refs,repair_text",
    [
        (
            "evidence_refs",
            [{"namespace": "worker", "name": "private-proof-canary"}],
            "non-empty revision",
        ),
        (
            "evidence_refs",
            [{"namespace": "worker", "name": "private-proof-canary", "revision": ""}],
            "non-empty revision",
        ),
        (
            "evidence_refs",
            [
                {
                    "namespace": "worker",
                    "name": "private-proof-canary",
                    "revision": "r1",
                    "mediaType": "private-media-canary",
                }
            ],
            "non-empty revision",
        ),
        (
            "evidence_refs",
            [
                {"namespace": "worker", "name": "valid", "revision": "r1"},
                {"namespace": "worker", "name": "private-proof-canary"},
            ],
            "non-empty revision",
        ),
        (
            "standard_refs",
            [
                {
                    "scheme": "private-scheme-canary",
                    "version": "1",
                    "requirement_id": "R1",
                    "title": "private-title-canary",
                }
            ],
            "standard_refs objects",
        ),
        (
            "standard_refs",
            [{"scheme": "private-scheme-canary", "version": "1"}],
            "standard_refs objects",
        ),
        (
            "standard_refs",
            [
                {"scheme": "VALID", "version": "1", "requirement_id": "R1"},
                {"scheme": "private-scheme-canary", "version": "1"},
            ],
            "standard_refs objects",
        ),
    ],
)
def test_invalid_finding_references_return_bounded_repair_error(
    factory_type, location_args, field, refs, repair_text
):
    async def scenario():
        client = FakeFindingClient()
        state = WorkerState()
        tool = await finding_tool(factory_type, client, state)
        with pytest.raises(ToolInputError) as failure:
            await FunctionTool(tool).run_async(
                args={
                    "title": "PrivateFindingCanary",
                    "description": "Observation",
                    field: refs,
                    **location_args,
                },
                tool_context=context(),
            )
        assert failure.value.code == "tool_input_invalid"
        assert repair_text in str(failure.value)
        assert client.requests == []
        assert state.metrics.counters["tool_errors"] == 1
        assert state.metrics.tool_calls[-1].error.code == "tool_input_invalid"
        diagnostics = str(failure.value) + repr(state.metrics.tool_calls[-1])
        for value in (
            "PrivateFindingCanary",
            "private-proof-canary",
            "private-media-canary",
            "private-scheme-canary",
            "private-title-canary",
        ):
            assert value not in diagnostics

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


def test_http_request_refuses_exchanges_finding_cannot_retain(tmp_path: Path) -> None:
    async def scenario() -> None:
        sent: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            sent.append(str(request.url))
            if request.url.path == "/redirect-long":
                return httpx.Response(
                    302, headers={"Location": "/" + "a" * 20_000}, request=request
                )
            if request.url.path == "/large-headers":
                return httpx.Response(
                    200,
                    headers={f"X-Fill-{index}": "x" * 7000 for index in range(10)},
                    request=request,
                )
            if request.url.path == "/bad-status":
                return httpx.Response(999, request=request)
            return httpx.Response(200, content=b"ok", request=request)

        http, state = await create_tools(tmp_path, handler)
        client = FakeFindingClient()
        finding = await finding_tool(HTTPFindingsToolsetFactory, client, state)
        try:
            for url in (
                "https://target.example/download?file=..\\..\\win.ini",
                "https://target.example/?q=50%",
                "https://target.example/?q=%u0027",
                "https://target.example/" + "é" * 3000,
            ):
                before = len(sent)
                with pytest.raises(HTTPToolError) as failure:
                    await http["http_request"](url, tool_context=context())
                assert failure.value.code == "http_request_invalid"
                assert len(sent) == before

            before = len(sent)
            with pytest.raises(HTTPToolError) as redirect_failure:
                await http["http_request"](
                    "https://target.example/redirect-long",
                    follow_redirects=True,
                    tool_context=context(),
                )
            assert redirect_failure.value.code == "http_request_invalid"
            assert len(sent) == before + 1

            for path in ("large-headers", "bad-status"):
                before = len(sent)
                with pytest.raises(HTTPToolError) as response_failure:
                    await http["http_request"](
                        f"https://target.example/{path}", tool_context=context()
                    )
                assert response_failure.value.code == "http_response_invalid"
                assert len(sent) == before + 1

            result = await http["http_request"]("https://target.example/ok", tool_context=context())
            await finding(
                title="Retained HTTP evidence",
                description="The valid exchange remains selectable",
                url="https://target.example/ok",
                method="GET",
                request_id=result["request_id"],
                tool_context=context(),
            )
            assert client.requests[-1]["proposal"]["http_exchange"]["attempts"][0]["status"] == 200
        finally:
            await close_tools(http)

    asyncio.run(scenario())


def test_every_sent_request_fits_finding_evidence_header_limits(tmp_path):
    async def scenario():
        observed = []

        async def handler(request):
            observed.append(request)
            return httpx.Response(200, text="ok", request=request)

        http, state = await create_tools(tmp_path, handler)
        client = FakeFindingClient()
        tool = await finding_tool(HTTPFindingsToolsetFactory, client, state)
        # Maximal session Basic auth plus model headers at their own budget.
        await http["http_session_set"](
            auth={"kind": "basic", "username": "u" * 256, "password": "p" * 8192}
        )
        headers = {f"X-Fill-{index}": "v" * 8000 for index in range(6)}
        filler = MAX_REQUEST_HEADER_BYTES - sum(len(k) + len(v) for k, v in headers.items())
        headers["X-Last"] = "v" * (filler - len("X-Last"))
        result = await http["http_request"](
            "https://target.example/", headers=headers, tool_context=context()
        )
        await tool(
            title="Finding",
            description="Observed",
            url="https://target.example/",
            method="GET",
            request_id=result["request_id"],
            tool_context=context(),
        )
        attempt = client.requests[0]["proposal"]["http_exchange"]["attempts"][0]
        assert len(attempt["headers"]) > len(headers)
        assert _proposal(json.dumps(client.requests[0]["proposal"]).encode())

        # Model headers beyond their budget are refused before sending.
        headers["X-Extra"] = "v"
        with pytest.raises(HTTPToolError) as invalid:
            await http["http_request"]("https://target.example/", headers=headers)
        assert invalid.value.code == "http_request_invalid"

        # Session cookies can still grow the block; such a request is never sent.
        await http["http_session_set"](cookies={f"c{index}": "x" * 8000 for index in range(7)})
        with pytest.raises(HTTPToolError) as oversized:
            await http["http_request"]("https://target.example/", headers={"X-A": "v" * 8000})
        assert oversized.value.code == "http_request_invalid"
        assert len(observed) == 1
        await http["http_request"].close()

    asyncio.run(scenario())


def test_http_evidence_never_retains_the_runtime_target_credential(tmp_path):
    target_secret = "project-target-secret"

    async def scenario():
        observed = []

        async def handler(request):
            observed.append(request)
            if request.url.path == "/start":
                return httpx.Response(
                    302, headers={"Location": "https://other.example/next"}, request=request
                )
            if request.url.path == "/next":
                return httpx.Response(
                    302, headers={"Location": "https://target.example/end"}, request=request
                )
            return httpx.Response(200, text="ok", request=request)

        def direct():
            return httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)

        state = WorkerState()
        factory = HTTPToolsetFactory(lambda _allocation, _settings: FakeArtifactClient(), direct)
        settings = RuntimeSettings(
            llmGatewayUrl="https://gateway.example/v1",
            artifactApiUrl="https://control.example/private/v1",
            httpOriginTarget=HTTPOriginTargetSettings(
                url="https://target.example/", bearerToken=target_secret
            ),
            requestTimeoutSeconds=30,
        )
        http = await make_tools(factory, tmp_path, state=state, settings=settings)
        client = FakeFindingClient()
        tool = await finding_tool(HTTPFindingsToolsetFactory, client, state)
        result = await http["http_request"](
            "https://target.example/start",
            headers={"Authorization": "Bearer model-value"},
            tool_context=context(),
        )
        other = await http["http_request"](
            "https://other.example/own",
            headers={"Authorization": "Bearer model-value"},
            tool_context=context(),
        )
        for request_id, call in ((result["request_id"], "call-1"), (other["request_id"], "call-2")):
            await tool(
                title="Finding",
                description="Observed",
                url="https://target.example/start",
                method="GET",
                request_id=request_id,
                tool_context=context(call),
            )
        # The target received the real credential on its own origin.
        assert observed[0].headers["authorization"] == f"Bearer {target_secret}"
        assert observed[2].headers["authorization"] == f"Bearer {target_secret}"
        assert target_secret not in json.dumps(client.requests)

        def authorization(attempt):
            return [
                row["value"] for row in attempt["headers"] if row["name"].lower() == "authorization"
            ]

        chain = client.requests[0]["proposal"]["http_exchange"]["attempts"]
        assert [authorization(attempt) for attempt in chain] == [
            ["[runtime-target-credential]"],
            [],
            ["[runtime-target-credential]"],
        ]
        # A credential the model supplied itself is evidence the model already knows.
        own = client.requests[1]["proposal"]["http_exchange"]["attempts"]
        assert authorization(own[0]) == ["Bearer model-value"]
        await close_tools(http)

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
