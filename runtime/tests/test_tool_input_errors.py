"""Malformed model calls must be repairable, measured and free of side effects."""

import asyncio
from types import SimpleNamespace

import pytest
from test_caido_read_tools import FakeArtifactClient as CaidoArtifacts
from test_caido_read_tools import create_tools as caido_tools
from test_http_toolset import create_tools as http_tools
from test_likec4_toolset import MemoryArtifactClient as LikeC4Artifacts
from test_likec4_toolset import make_tools as likec4_tools
from test_openapi_toolset import MemoryArtifactClient as OpenAPIArtifacts
from test_openapi_toolset import make_tools as openapi_tools
from test_run_artifacts_toolset import FakeArtifactClient
from test_security_findings_toolset import FakeFindingClient
from test_source_analysis_toolset import ReadOnlyArtifactClient
from test_source_analysis_toolset import make_tools as source_tools
from test_text_artifacts_toolset import MemoryArtifactClient
from test_text_artifacts_toolset import make_tools as text_tools

from contractor_runtime.allocation import WorkerState
from contractor_runtime.telemetry.metrics import MetricsState
from contractor_runtime.toolsets.caido.tools import CaidoToolError
from contractor_runtime.toolsets.common.input_errors import ToolInputError
from contractor_runtime.toolsets.http.tools import HTTPToolError
from contractor_runtime.toolsets.run_artifacts.tools import WriteArtifactTool
from contractor_runtime.toolsets.security_findings.tools import FindingTool
from contractor_runtime.worker.instrumentation import _safe_tool_response


def assert_repair(error, field, metrics):
    reply = _safe_tool_response("test_tool", error)
    assert reply["error"]["code"] == "tool_input_invalid"
    assert field in reply["error"]["message"]
    assert not reply["error"]["retryable"]
    assert "failed (" not in reply["error"]["message"]
    assert metrics.counters["tool_calls"] == metrics.counters["tool_errors"] == 1
    assert field in metrics.tool_calls[0].error.message


@pytest.mark.parametrize(
    "field,value",
    [
        ("severity_suggestion", {"level": "high"}),
        ("severity_suggestion", False),
        ("severity_suggestion", []),
        ("hypothesis", []),
        ("hypothesis", False),
        ("description", "\ud800"),
        ("title", "\ud800"),
        ("client_key", "\ud800"),
        ("proposed_checks", "[]"),
        ("evidence_refs", "[]"),
    ],
)
def test_finding_rejects_invalid_types_and_unicode_with_repair(field, value):
    async def scenario():
        client, metrics = FakeFindingClient(), MetricsState()
        tool = FindingTool(client, metrics, ())
        arguments = {
            "client_key": "candidate-1",
            "title": "Candidate",
            "description": "Description",
            "subject": {"kind": "code", "key": "handler"},
            "evidence_refs": [],
            field: value,
        }
        with pytest.raises(ToolInputError) as caught:
            await tool(tool_context=SimpleNamespace(invocation_id="invocation-1"), **arguments)
        assert_repair(caught.value, field, metrics)
        assert client.requests == []

    asyncio.run(scenario())


@pytest.mark.parametrize("payload", [None, 12, {}, [], "%%%", "\ud800", "Zh=="])
def test_artifact_write_rejects_noncanonical_base64_without_write(payload):
    async def scenario():
        client, metrics = FakeArtifactClient(), MetricsState()
        tool = WriteArtifactTool(client, metrics, ())
        with pytest.raises(ToolInputError) as caught:
            await tool("worker", "artifact", "application/octet-stream", payload)
        assert_repair(caught.value, "data_base64", metrics)
        assert client.write_calls == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("value", [None, {}, "\ud800"])
def test_text_write_rejects_invalid_input_and_records_failure(tmp_path, value):
    async def scenario():
        client, state = MemoryArtifactClient(), WorkerState()
        tools = await text_tools(tmp_path, client, state)
        with pytest.raises(ToolInputError) as caught:
            await tools["write_text_artifact"]("result", value, "text/plain")
        assert_repair(caught.value, "text", state.metrics)
        assert not client.bindings

    asyncio.run(scenario())


@pytest.mark.parametrize("query", [None, {}, 5, "\ud800"])
def test_source_search_validates_before_encoding_metric_arguments(tmp_path, query):
    async def scenario():
        client, state = ReadOnlyArtifactClient(), WorkerState()
        tools = await source_tools(tmp_path, client, state)
        with pytest.raises(ToolInputError) as caught:
            await tools["search_source"](query)
        assert_repair(caught.value, "query", state.metrics)
        assert client.read_calls == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("content", [None, {}, "\ud800"])
def test_likec4_invalid_content_has_repair_message(tmp_path, content):
    async def scenario():
        client, state = LikeC4Artifacts(), WorkerState()
        tools = await likec4_tools(tmp_path, client, state, namespace="architecture")
        with pytest.raises(ToolInputError) as caught:
            await tools["write_likec4"](content)
        assert_repair(caught.value, "content", state.metrics)
        assert not client.bindings

    asyncio.run(scenario())


def test_openapi_invalid_title_has_repair_message(tmp_path):
    async def scenario():
        client, state = OpenAPIArtifacts(), WorkerState()
        tools = await openapi_tools(tmp_path, client, state, namespace="openapi")
        with pytest.raises(ToolInputError) as caught:
            await tools["initialize_openapi"](title={})
        assert_repair(caught.value, "title", state.metrics)
        assert not client.bindings

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("http_request", {"url": "https://target.example/\ud800"}),
        ("http_request", {"url": "https://target.example", "query": {"q": "\ud800"}}),
        ("http_request", {"url": "https://target.example", "query": {"\ud800": "q"}}),
        ("http_request", {"url": "https://target.example", "headers": {"X-Q": "\ud800"}}),
        ("http_request", {"url": "https://target.example", "timeout": 10**1000}),
        ("http_session_set", {"cookies": {"name": "\ud800"}}),
        ("http_session_set", {"auth": {"kind": "bearer", "token": "\ud800"}}),
    ],
)
def test_http_invalid_unicode_and_overflow_are_not_retryable_network_failures(
    tmp_path, name, arguments
):
    async def scenario():
        requests = []

        async def unexpected_request(request):
            requests.append(request)
            raise AssertionError("invalid request reached the network")

        tools, state = await http_tools(tmp_path, unexpected_request)
        try:
            with pytest.raises(HTTPToolError) as caught:
                await tools[name](**arguments)
            assert caught.value.code == "http_request_invalid"
            assert not caught.value.retryable
            assert state.metrics.counters["tool_errors"] == 1
            assert not requests
        finally:
            for tool in tools.values():
                await tool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("caido_history", {"filter": "\ud800"}),
        ("caido_request_detail", {"request_id": "\ud800"}),
        ("caido_replay", {"raw_request": "\ud800", "host": "target.example"}),
        ("caido_workflow_run", {"workflow_id": "workflow-1", "input": "\ud800"}),
        ("caido_scope", {"action": "create", "name": "\ud800", "allowlist": ["target.example"]}),
    ],
)
def test_caido_invalid_unicode_is_rejected_before_network_and_still_measured(
    tmp_path, name, arguments
):
    async def scenario():
        requests = []

        async def unexpected_request(request):
            requests.append(request)
            raise AssertionError("invalid request reached Caido")

        tools, state, handle = await caido_tools(
            tmp_path, unexpected_request, CaidoArtifacts(), selected={name}
        )
        try:
            with pytest.raises(CaidoToolError) as caught:
                await tools[name](**arguments)
            assert caught.value.code == "caido_request_invalid"
            assert not caught.value.retryable
            assert state.metrics.counters["tool_errors"] == 1
            assert not requests
        finally:
            for tool in tools.values():
                await tool.close()
            await handle.close()

    asyncio.run(scenario())


def test_untrusted_exceptions_stay_opaque_and_repair_messages_are_bounded():
    secret = "provider-secret"
    assert secret not in str(_safe_tool_response("probe", ValueError(secret)))
    reply = _safe_tool_response("probe", ToolInputError("Use valid text. " + "🧪" * 1000))
    assert len(reply["error"]["message"].encode("utf-8")) <= 512


def test_metrics_remain_serializable_with_invalid_unicode_in_arguments():
    metrics = MetricsState()
    metrics.record_tool_call("probe", arguments={"name": "\ud800"}, error=ValueError("\ud800"))
    report = metrics.build_report(report_id="invalid-unicode", duration_ms=1)
    assert report.model_dump_json().encode("utf-8")
    assert report.tool_calls[0].arguments["name"] == "?"
