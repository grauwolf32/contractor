from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_caido_read_tools import FakeArtifactClient, create_tools, representative_responses

import contractor_runtime.toolsets.caido.tools as caido
from contractor_runtime.toolsets.caido.tools import CaidoToolError


def test_invalid_inputs_fail_before_transport(
    tmp_path: Path,
) -> None:
    requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json={"data": {"scopes": []}}, request=request)

    async def scenario() -> None:
        artifacts = FakeArtifactClient()
        tools, _state, handle = await create_tools(tmp_path, handler, artifacts)
        calls = [
            lambda: tools["caido_scope"](action="create", name=""),
            lambda: tools["caido_history"](filter="x\nsecret"),
            lambda: tools["caido_history"](limit=101),
            lambda: tools["caido_request_detail"](""),
            lambda: tools["caido_automate_results"]("session", sort_by="UNKNOWN"),
            lambda: tools["caido_sitemap"](depth="RECURSIVE"),
            lambda: tools["caido_workflow_list"]("arbitrary"),
            lambda: tools["caido_workflow_findings"](offset=-1),
        ]
        for call in calls:
            with pytest.raises(CaidoToolError) as failure:
                await call()
            assert failure.value.code == "caido_request_invalid"
        assert requests == 0
        assert artifacts.payloads == {}
        await handle.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data["RequestDetail"].update({"unexpected": True}),
        lambda data: data["RequestDetail"].update({"request": None, "unexpected": True}),
        lambda data: data["RequestsByOffset"]["requestsByOffset"]["count"].update({"value": -1}),
        lambda data: data["Workflows"]["workflows"][0].update({"kind": "UNKNOWN"}),
        lambda data: data["RequestDetail"]["request"].update({"raw": "not-base64"}),
    ],
)
def test_malformed_or_partial_responses_fail_without_artifact_selection(
    tmp_path: Path, mutate: Any
) -> None:
    responses = representative_responses(b"GET / HTTP/1.1\r\n\r\n", b"HTTP/1.1 200 OK\r\n\r\n")
    mutate(responses)

    async def handler(request: httpx.Request) -> httpx.Response:
        operation = json.loads(request.content)["operationName"]
        return httpx.Response(200, json={"data": responses[operation]}, request=request)

    async def scenario() -> None:
        artifacts = FakeArtifactClient()
        tools, _state, handle = await create_tools(tmp_path, handler, artifacts)
        with pytest.raises(CaidoToolError) as failure:
            request_value = responses["RequestDetail"].get("request")
            if "unexpected" in responses["RequestDetail"] or (
                isinstance(request_value, dict) and request_value.get("raw") == "not-base64"
            ):
                await tools["caido_request_detail"]("request-1")
            elif responses["RequestsByOffset"]["requestsByOffset"]["count"]["value"] == -1:
                await tools["caido_history"]()
            else:
                await tools["caido_workflow_list"]()
        assert failure.value.code == "caido_response_invalid"
        assert artifacts.payloads == {}
        await handle.close()

    asyncio.run(scenario())


def test_null_domain_results_are_explicit_and_result_size_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = representative_responses(b"", b"")
    responses["RequestDetail"] = {"request": None}
    responses["AutomateSession"] = {"automateSession": None}

    async def handler(request: httpx.Request) -> httpx.Response:
        operation = json.loads(request.content)["operationName"]
        return httpx.Response(200, json={"data": responses[operation]}, request=request)

    async def scenario() -> None:
        tools, _state, handle = await create_tools(tmp_path, handler, FakeArtifactClient())
        assert await tools["caido_request_detail"]("missing") == {
            "request_id": "missing",
            "status": "not_found",
        }
        assert await tools["caido_automate_results"]("missing") == {
            "session_id": "missing",
            "status": "not_found",
        }

        monkeypatch.setattr(caido, "MAX_TOOL_RESULT_BYTES", 16)
        with pytest.raises(CaidoToolError) as oversized:
            await tools["caido_workflow_list"]()
        assert oversized.value.code == "caido_response_too_large"
        await handle.close()

    asyncio.run(scenario())


def test_noncanonical_blob_and_graphql_error_are_content_free(
    tmp_path: Path,
) -> None:
    server_secret = "recognizable-server-detail"
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            detail = representative_responses(b"x", b"")["RequestDetail"]
            detail["request"]["raw"] = base64.b64encode(b"x").decode() + "\n"
            return httpx.Response(200, json={"data": detail}, request=request)
        return httpx.Response(200, json={"errors": [{"message": server_secret}]}, request=request)

    async def scenario() -> None:
        tools, state, handle = await create_tools(tmp_path, handler, FakeArtifactClient())
        with pytest.raises(CaidoToolError) as invalid:
            await tools["caido_request_detail"]("request-1")
        assert invalid.value.code == "caido_response_invalid"
        with pytest.raises(CaidoToolError) as failed:
            await tools["caido_scope"]()
        assert failed.value.code == "caido_request_failed"
        assert server_secret not in repr(state.metrics)
        await handle.close()

    asyncio.run(scenario())
