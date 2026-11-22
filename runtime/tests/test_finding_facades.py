from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from audit_fixtures import assigned
from fakes.source_selectors import artifact_sources, http_sources
from fakes.worker_state import WorkerState
from google.adk.tools import FunctionTool
from google.genai import types
from http_cookie_store import HTTPToolsetFactory
from test_audit_result_collector import CONTEXT, tool_for
from test_http_toolset import FakeArtifactClient, close_tools, create_tools, make_tools
from test_security_findings_toolset import FakeFindingClient, _settings, _workspace

from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import ArtifactRef, HTTPOriginTargetSettings, RuntimeSettings
from contractor_runtime.core.http_limits import MAX_HISTORY, MAX_REQUEST_HEADER_BYTES
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.llm.openai import _tools
from contractor_runtime.toolsets.audit_results.collector import AuditCollectionError
from contractor_runtime.toolsets.common.cwe import CWEReferenceError
from contractor_runtime.toolsets.common.finding_locations import (
    StandardReference,
)
from contractor_runtime.toolsets.http.tools import HTTPToolError
from contractor_runtime.toolsets.security_findings.collection import _proposal
from contractor_runtime.toolsets.security_findings.facades import (
    CodeFindingsToolsetFactory,
    CodeFindingTool,
    GeneralFindingsToolsetFactory,
    GeneralFindingTool,
    HTTPFindingsToolsetFactory,
    HTTPFindingTool,
)
from contractor_runtime.toolsets.security_findings.publisher import FindingPublisher
from contractor_runtime.worker.instrumentation import _safe_tool_response


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
    evidence = schema["$defs"]["ArtifactSource"]
    assert set(evidence["required"]) == {"kind", "namespace", "name"}
    assert evidence["additionalProperties"] is False
    check = schema["$defs"]["ProposedCheck"]
    assert set(check["required"]) == {"objective", "method"}
    assert set(check["properties"]) == {"objective", "method"}
    assert check["additionalProperties"] is False
    if tool_type is GeneralFindingTool:
        assert schema["$defs"]["SourceLocation"]["additionalProperties"] is False
        assert schema["$defs"]["WebLocation"]["additionalProperties"] is False


async def finding_tool(factory_type, client, state, *, name="finding"):
    # The toolset submits through the allocation's private API transport.
    tools = await factory_type(
        lambda _allocation, _settings: ArtifactClient("allocation-test", client)
    ).create_selected(
        selected=[name],
        allocation_id="allocation-test",
        run_id="run-test",
        namespace="worker",
        runtime_settings=_settings(),
        workspace=_workspace(),
        resources=state.resources,
    )
    return tools[name]


@pytest.mark.parametrize(
    "factory_type,location_args",
    [
        (GeneralFindingsToolsetFactory, {}),
        (CodeFindingsToolsetFactory, {"file": "src/order.py"}),
        (HTTPFindingsToolsetFactory, {"url": "https://target.example", "method": "GET"}),
    ],
)
def test_all_facades_publish_ordered_check_suggestions_through_adk(factory_type, location_args):
    async def scenario():
        client = FakeFindingClient()
        tool = await finding_tool(factory_type, client, WorkerState())
        checks = [
            {
                "objective": 'Проверить <owner> & "scope"\nagainst exact evidence.',
                "method": "verify",
            },
            {"objective": "Repeat the observation independently.", "method": "replay"},
            {"objective": "An unregistered suggestion remains data.", "method": "operator-idea"},
        ]
        args = {
            "title": "Observation",
            "description": "Needs an independently admitted follow-up",
            "proposed_checks": checks,
            "item_key": "item-01",
            "incidental": True,
            **location_args,
        }
        adk = FunctionTool(tool)
        await adk.run_async(args=args, tool_context=context())
        await adk.run_async(args=args, tool_context=context())
        assert client.requests[0] == client.requests[1]
        assert client.requests[0]["proposal"]["proposed_checks"] == checks
        assert client.requests[0]["itemKey"] == "item-01"
        assert client.requests[0]["incidental"] is True
        assert _proposal(json.dumps(client.requests[0]["proposal"]).encode())

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "checks",
    [
        "verify",
        {},
        [{"objective": "Missing method"}],
        [{"objective": "No role grant", "method": "verify", "workflow_role": "chosen"}],
        [{"objective": "No phase grant", "method": "verify", "phase": "verify"}],
        [{"objective": "No capability grant", "method": "verify", "capabilities": []}],
        [{"objective": "\u2003", "method": "verify"}],
        [{"objective": "x\x00y", "method": "verify"}],
        [{"objective": "é" * 32769, "method": "verify"}],
        [{"objective": "x\ud800", "method": "verify"}],
        [{"objective": "Question", "method": ""}],
        [{"objective": "Question", "method": "role/ref@1"}],
        [{"objective": "Question", "method": "m" * 161}],
        [{"objective": "Question", "method": True}],
        [{"objective": "Question", "method": "verify"}] * 513,
    ],
    ids=[
        "scalar",
        "object",
        "missing",
        "role",
        "phase",
        "capability",
        "blank",
        "nul",
        "utf8-bytes",
        "surrogate",
        "empty-method",
        "workflow-ref",
        "long-method",
        "boolean",
        "count",
    ],
)
def test_invalid_check_suggestions_publish_nothing(checks):
    async def scenario():
        client = FakeFindingClient()
        state = WorkerState()
        tool = await finding_tool(GeneralFindingsToolsetFactory, client, state)
        with pytest.raises(ToolInputError, match="proposed_checks"):
            await tool(
                title="Observation",
                description="Question",
                proposed_checks=checks,
                tool_context=context(),
            )
        assert client.requests == []
        assert state.metrics.counters["tool_errors"] == 1

    asyncio.run(scenario())


def test_check_suggestion_maxima_and_omission():
    async def scenario():
        client = FakeFindingClient()
        tool = await finding_tool(GeneralFindingsToolsetFactory, client, WorkerState())
        checks = [{"objective": "é" * 32768, "method": "m" * 160}]
        checks += [{"objective": "Question", "method": "verify"}] * 511
        await tool(
            title="Observation",
            description="Question",
            proposed_checks=checks,
            tool_context=context(),
        )
        assert client.requests[0]["proposal"]["proposed_checks"] == checks
        await tool(title="Observation", description="No follow-up", tool_context=context("call-2"))
        assert client.requests[1]["proposal"]["proposed_checks"] == []

    asyncio.run(scenario())


def request_attempts(submission):
    result = []
    for attempt in submission["httpEvidence"]["attempts"]:
        value = {**attempt["request"], **attempt["request"]["message"]}
        if "response" in attempt:
            value["status"] = attempt["response"]["status"]
        if "error" in attempt:
            value["error"] = attempt["error"]
        result.append(value)
    return result


def context(call_id="call-1", invocation_id="invocation-1"):
    return SimpleNamespace(invocation_id=invocation_id, function_call_id=call_id)


def test_invalid_final_response_keeps_prior_redirect_evidence(tmp_path):
    async def scenario():
        async def handler(request):
            if request.url.path == "/start":
                return httpx.Response(302, headers={"Location": "/final"}, request=request)
            return httpx.Response(600, request=request)

        http, state = await create_tools(tmp_path, handler)
        try:
            with pytest.raises(HTTPToolError, match="http_response_invalid"):
                await http["http_request"]("https://target.example/start", tool_context=context())
            history = await http["http_history"]()
            client = FakeFindingClient()
            finding = await finding_tool(HTTPFindingsToolsetFactory, client, state)
            await finding(
                title="Invalid final response",
                description="A redirect was followed by an invalid response.",
                url="https://target.example/start",
                method="GET",
                sources=http_sources(history[0]["request_id"]),
                tool_context=context(),
            )
            attempts = client.requests[0]["httpEvidence"]["attempts"]
            assert len(attempts) == 2
            assert attempts[0]["response"]["status"] == 302
            assert attempts[0]["response"]["complete"] is False
            assert "error" not in attempts[0]
            assert "response" not in attempts[1]
            assert attempts[1]["error"] == "response_invalid"
        finally:
            await close_tools(http)

    asyncio.run(scenario())


@pytest.mark.parametrize("has_response", [True, False])
@pytest.mark.parametrize("inspected", [True, False])
@pytest.mark.parametrize(
    "factory_type,name,location_args",
    [
        (GeneralFindingsToolsetFactory, "finding", {}),
        (GeneralFindingsToolsetFactory, "fact", {}),
        (
            HTTPFindingsToolsetFactory,
            "finding",
            {"url": "https://target.example/probe", "method": "POST"},
        ),
    ],
)
def test_caido_evidence_resolves_local_id_without_model_inspection(
    tmp_path, has_response, inspected, factory_type, name, location_args
):
    from test_caido_read_tools import FakeArtifactClient as CaidoArtifacts
    from test_caido_read_tools import close_tools as close_caido
    from test_caido_read_tools import create_tools as create_caido
    from test_caido_read_tools import representative_responses

    async def scenario():
        raw_request = b"POST /probe HTTP/1.1\r\nHost: target.example\r\n\r\n\x00\xff"
        raw_response = b"HTTP/1.1 403 Forbidden\r\nX-Test: one\r\nX-Test: two\r\n\r\n\x00\xff"
        responses = representative_responses(raw_request, raw_response)
        if not has_response:
            responses["RequestDetail"]["request"]["response"] = None
        observed = []

        async def handler(request):
            payload = json.loads(request.content)
            observed.append(payload["operationName"])
            return httpx.Response(200, json={"data": responses[payload["operationName"]]})

        artifacts = CaidoArtifacts()
        selected = {"caido_history", "caido_request_detail"} if inspected else {"caido_history"}
        caido, state, handle = await create_caido(tmp_path, handler, artifacts, selected=selected)
        try:
            history = await caido["caido_history"]()
            request_id = history["requests"][0]["request_id"]
            if inspected:
                await caido["caido_request_detail"](request_id, tool_context=context())
            else:
                assert "caido_request_detail" not in caido
                assert observed == ["RequestsByOffset"]
            client = FakeFindingClient()
            tool = await finding_tool(factory_type, client, state, name=name)
            result = await tool(
                title="Observed issue",
                description="Captured behavior",
                **location_args,
                sources=http_sources(request_id),
                tool_context=context(),
            )
            capture = client.requests[0]["httpEvidence"]
            attempt = capture["attempts"][0]
            assert capture["source"] == "caido"
            assert base64.b64decode(attempt["request"]["message"]["data_base64"]) == raw_request
            if has_response:
                assert (
                    base64.b64decode(attempt["response"]["message"]["data_base64"]) == raw_response
                )
            else:
                assert "response" not in attempt and attempt["error"] == "response_unavailable"
            assert result["evidence_refs"][-1]["namespace"] == "http-exchanges"
            assert observed == ["RequestsByOffset", "RequestDetail"]
            assert artifacts.payloads == {}
            await tool(
                title="F",
                description="D",
                **location_args,
                sources=http_sources(request_id),
                tool_context=context(invocation_id="later-invocation"),
            )
            assert client.requests[-1]["httpEvidence"] == capture
            assert client.requests[-1]["httpEvidenceInvocationId"] == "invocation-1"
            assert client.requests[-1]["invocationId"] == "later-invocation"
            assert observed == ["RequestsByOffset", "RequestDetail"]
        finally:
            await close_caido(caido)
            await handle.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure", ["unknown", "foreign", "evicted", "missing", "denied", "cancelled"]
)
def test_stored_http_evidence_failure_never_submits_a_finding(tmp_path, failure):
    from test_caido_read_tools import FakeArtifactClient as CaidoArtifacts
    from test_caido_read_tools import close_tools as close_caido
    from test_caido_read_tools import create_tools as create_caido
    from test_caido_read_tools import representative_responses

    from contractor_runtime.toolsets.caido.tools import CaidoToolError

    async def scenario():
        responses = representative_responses(
            b"GET /probe HTTP/1.1\r\nHost: target.example\r\n\r\n",
            b"HTTP/1.1 200 OK\r\n\r\n",
        )
        observed = []
        reading = asyncio.Event()
        joined = asyncio.Event()

        async def handler(request):
            operation = json.loads(request.content)["operationName"]
            observed.append(operation)
            if operation == "RequestDetail":
                if failure == "missing":
                    return httpx.Response(200, json={"data": {"request": None}})
                if failure == "denied":
                    return httpx.Response(403)
                if failure == "cancelled":
                    reading.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        joined.set()
            return httpx.Response(200, json={"data": responses[operation]})

        tools, state, handle = await create_caido(
            tmp_path, handler, CaidoArtifacts(), selected={"caido_history"}
        )
        client = FakeFindingClient()
        try:
            history = await tools["caido_history"]()
            request_id = history["requests"][0]["request_id"]
            selected_state = state
            if failure == "unknown":
                request_id += 1
            elif failure == "foreign":
                selected_state = WorkerState()
            elif failure == "evicted":
                state.resources.http_captures.clear("caido")
            tool = await finding_tool(GeneralFindingsToolsetFactory, client, selected_state)
            call = tool(
                title="Stored observation",
                description="The selected request supplies the evidence",
                sources=http_sources(request_id),
                tool_context=context(),
            )
            if failure == "cancelled":
                pending = asyncio.create_task(call)
                await reading.wait()
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                assert joined.is_set()
            else:
                expected = CaidoToolError if failure == "denied" else ToolInputError
                with pytest.raises(expected):
                    await call
            assert client.requests == []
            assert observed == (
                ["RequestsByOffset", "RequestDetail"]
                if failure in {"missing", "denied", "cancelled"}
                else ["RequestsByOffset"]
            )
        finally:
            await close_caido(tools)
            await handle.close()

    asyncio.run(scenario())


def test_code_call_normalizes_coordinates_and_keeps_retry_identity():
    async def scenario():
        client = FakeFindingClient()
        state = WorkerState()
        state.resources.selected_versions.observe(
            ArtifactRef(namespace="worker", name="proof", revision="r1")
        )
        tool = await finding_tool(CodeFindingsToolsetFactory, client, state)
        adk = FunctionTool(tool)
        args = {
            "title": "Ownership guard missing",
            "description": "Source observation",
            "file": "src/order.py",
            "range": {"start_line": 10, "end_line": 12},
            "cwe": "CWE-639",
            "sources": artifact_sources([{"namespace": "worker", "name": "proof"}]),
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
        assert client.requests[0]["evidenceRefs"] == [
            {"namespace": "worker", "name": "proof", "revision": "r1"}
        ]
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
        collector, _, _ = tool_for(assigned()[1])
        audit_state = WorkerState()
        audit_state.resources.proposal_collector = lambda: collector
        reference = StandardReference(scheme="CWE", version="4.20", requirement_id="CWE-534")
        failures = []
        for state in (audit_state, WorkerState()):
            client = FakeFindingClient()
            finding = GeneralFindingTool(
                FindingPublisher(client, state.metrics, (), state.resources)
            )
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
            "sources",
            [{"namespace": "worker", "name": "private-proof-canary"}],
            "sources",
        ),
        (
            "sources",
            [{"namespace": "worker", "name": "private-proof-canary", "revision": ""}],
            "sources",
        ),
        (
            "sources",
            [
                {
                    "namespace": "worker",
                    "name": "private-proof-canary",
                    "revision": "r1",
                    "mediaType": "private-media-canary",
                }
            ],
            "sources",
        ),
        (
            "sources",
            [
                {"namespace": "worker", "name": "valid", "revision": "r1"},
                {"namespace": "worker", "name": "private-proof-canary"},
            ],
            "sources",
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
                    field: artifact_sources(refs) if field == "sources" else refs,
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
            sources=http_sources(result["request_id"]),
            tool_context=context(),
        )
        document = client.requests[0]["proposal"]
        assert _proposal(json.dumps(document).encode()) == document
        attempts = request_attempts(client.requests[0])
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
        response = client.requests[0]["httpEvidence"]["attempts"][-1]["response"]
        assert base64.b64decode(response["message"]["body_base64"]) == b"observed response"
        assert response["complete"] is True
        assert client.requests[0]["evidenceRefs"] == []
        assert artifacts.writes == 0
        assert "capture-secret" not in repr(state.metrics)
        assert "cookie-secret" not in repr(await http["http_history"]())

        await tool(
            title="F",
            description="D",
            url="https://target.example",
            method="GET",
            sources=http_sources(result["request_id"]),
            tool_context=context(invocation_id="later-invocation"),
        )
        assert client.requests[-1]["httpEvidence"] == client.requests[0]["httpEvidence"]
        assert client.requests[-1]["httpEvidenceInvocationId"] == "invocation-1"
        assert client.requests[-1]["invocationId"] == "later-invocation"
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
                sources=http_sources(result["request_id"]),
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
                sources=http_sources(result["request_id"]),
                tool_context=context(),
            )
            assert request_attempts(client.requests[-1])[0]["status"] == 200
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
            sources=http_sources(result["request_id"]),
            tool_context=context(),
        )
        attempt = request_attempts(client.requests[0])[0]
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
        factory = HTTPToolsetFactory(direct)
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
                sources=http_sources(request_id),
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

        chain = request_attempts(client.requests[0])
        assert [authorization(attempt) for attempt in chain] == [
            ["[runtime-target-credential]"],
            [],
            ["[runtime-target-credential]"],
        ]
        # A credential the model supplied itself is evidence the model already knows.
        own = request_attempts(client.requests[1])
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


@pytest.mark.parametrize(
    "factory_type,location_args",
    [
        (GeneralFindingsToolsetFactory, {}),
        (CodeFindingsToolsetFactory, {"file": "src/order.py"}),
        (HTTPFindingsToolsetFactory, {"url": "https://target.example", "method": "GET"}),
    ],
)
def test_informational_findings_leave_evidence_equality_to_server(
    factory_type, location_args, monkeypatch
):
    async def forbidden_derivation(*_args, **_kwargs):
        raise AssertionError("informational equality must not parse source payloads")

    monkeypatch.setattr(
        "contractor_runtime.toolsets.security_findings.publisher.prepare_anchors",
        forbidden_derivation,
    )

    async def scenario():
        client = FakeFindingClient()
        state = WorkerState()
        state.resources.selected_versions.observe(
            ArtifactRef(namespace="proof", name="selected", revision="r1")
        )
        tool = await finding_tool(factory_type, client, state)
        result = await tool(
            title="Reported technology",
            description="Observed in selected material",
            severity_suggestion="informational",
            sources=artifact_sources([{"namespace": "proof", "name": "selected"}]),
            tool_context=context(),
            **location_args,
        )
        assert result["finding_id"] == "finding-1"
        assert "derived_anchors" not in client.requests[-1]
        assert "fingerprint" not in client.requests[-1]

    asyncio.run(scenario())
