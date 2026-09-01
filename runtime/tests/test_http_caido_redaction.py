"""Secret-retention regressions for HTTP/Caido tools and execution reports."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_caido_read_tools import CAIDO_TOKEN
from test_caido_read_tools import close_tools as close_caido_tools
from test_caido_read_tools import create_tools as create_caido_tools
from test_http_toolset import FakeArtifactClient, close_tools, create_tools

import contractor_runtime.metrics as runtime_metrics
from contractor_runtime.toolsets.caido import CaidoToolError
from contractor_runtime.toolsets.http_tools import HTTPToolError

HTTP_AUTH_SECRET = "canary-http-auth-07e42"
HTTP_COOKIE_SECRET = "canary-http-cookie-a8f13"
HTTP_HEADER_SECRET = "canary-http-header-631dd"
HTTP_REQUEST_BODY_SECRET = "canary-http-request-body-e590b"
HTTP_RESPONSE_BODY = "canary-http-response-body-2d9b8"
HTTP_RESPONSE_COOKIE = "canary-http-response-cookie-fc711"
HTTP_RESPONSE_HEADER = "canary-http-response-header-44cc0"
CAIDO_FILTER = "req.raw.cont:canary-caido-filter-5b81d"
CAIDO_SERVER_TEXT = "canary-caido-server-error-6730a"
CAIDO_ENDPOINT = "https://caido.example/graphql"


def test_http_secrets_are_absent_from_every_retained_projection(tmp_path: Path) -> None:
    artifacts = FakeArtifactClient()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {HTTP_AUTH_SECRET}"
        assert HTTP_COOKIE_SECRET in request.headers["cookie"]
        assert request.headers["x-api-key"] == HTTP_HEADER_SECRET
        assert request.content.decode() == HTTP_REQUEST_BODY_SECRET
        return httpx.Response(
            418,
            content=HTTP_RESPONSE_BODY.encode(),
            headers=[
                ("content-type", "text/plain"),
                ("set-cookie", f"response_session={HTTP_RESPONSE_COOKIE}; Path=/"),
                ("x-api-key", HTTP_RESPONSE_HEADER),
            ],
            request=request,
        )

    async def scenario() -> None:
        tools, state = await create_tools(tmp_path, handler, artifacts=artifacts)
        await tools["http_session_set"](
            auth={"kind": "bearer", "token": HTTP_AUTH_SECRET},
            cookies={"sid": HTTP_COOKIE_SECRET},
            headers={"X-API-Key": HTTP_HEADER_SECRET},
        )
        response = await tools["http_request"](
            "https://target.example/retention",
            method="POST",
            body_type="text",
            body=HTTP_REQUEST_BODY_SECRET,
        )
        assert response["body_preview"] == HTTP_RESPONSE_BODY
        assert response["headers"]["content-type"] == "text/plain"
        assert "set-cookie" not in response["headers"]
        assert "x-api-key" not in response["headers"]
        exact_body = await tools["http_read_body"](response["request_id"])
        assert exact_body["data"] == HTTP_RESPONSE_BODY

        retained = retained_worker_surfaces(
            state=state,
            session=await tools["http_session_get"](),
            history=await tools["http_history"](),
            extra_repr=repr(tools),
        )
        for forbidden in (
            HTTP_AUTH_SECRET,
            HTTP_COOKIE_SECRET,
            HTTP_HEADER_SECRET,
            HTTP_REQUEST_BODY_SECRET,
            HTTP_RESPONSE_BODY,
            HTTP_RESPONSE_COOKIE,
            HTTP_RESPONSE_HEADER,
        ):
            assert forbidden not in retained

        encoded_artifacts = b"\n".join(artifacts.payloads.values())
        assert HTTP_RESPONSE_BODY.encode() in encoded_artifacts
        for forbidden in (
            HTTP_AUTH_SECRET,
            HTTP_COOKIE_SECRET,
            HTTP_HEADER_SECRET,
            HTTP_REQUEST_BODY_SECRET,
            HTTP_RESPONSE_COOKIE,
            HTTP_RESPONSE_HEADER,
        ):
            assert forbidden.encode() not in encoded_artifacts
        await close_tools(tools)

    asyncio.run(scenario())


def test_caido_transport_discards_token_filter_endpoint_and_server_error(
    tmp_path: Path,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {CAIDO_TOKEN}"
        assert CAIDO_FILTER in request.content.decode()
        return httpx.Response(
            200,
            json={"data": None, "errors": [{"message": CAIDO_SERVER_TEXT}]},
            request=request,
        )

    async def scenario() -> None:
        artifacts = FakeArtifactClient()
        tools, state, handle = await create_caido_tools(
            tmp_path,
            handler,
            artifacts,
            selected={"caido_history"},
        )
        with pytest.raises(CaidoToolError) as failure:
            await tools["caido_history"](filter=CAIDO_FILTER)
        assert failure.value.code == "caido_request_failed"

        retained = retained_worker_surfaces(
            state=state,
            session={},
            history=[],
            extra_repr=f"{tools!r} {handle!r}",
        )
        retained += repr(handle._metrics)
        for forbidden in (CAIDO_TOKEN, CAIDO_FILTER, CAIDO_SERVER_TEXT, CAIDO_ENDPOINT):
            assert forbidden not in retained
        assert artifacts.payloads == {}
        await close_caido_tools(tools)
        await handle.close()

    asyncio.run(scenario())


def test_metric_history_is_bounded_and_redacted_under_repeated_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_metrics, "MAX_METRIC_TOOL_CALLS", 3)
    monkeypatch.setattr(runtime_metrics, "MAX_METRIC_ERRORS", 2)
    raw_canary = "canary-repeated-input-5eb37"

    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"invalid requests must not reach transport: {request!r}")

    async def scenario() -> None:
        tools, state = await create_tools(tmp_path, handler)
        await tools["http_session_set"](auth={"kind": "bearer", "token": HTTP_AUTH_SECRET})
        for index in range(6):
            with pytest.raises(HTTPToolError) as failure:
                await tools["http_request"](f"https://control.example/private/{raw_canary}/{index}")
            assert failure.value.code == "http_target_denied"

        snapshot = state.metrics.snapshot()
        report = state.metrics.build_report(report_id="report-redaction", duration_ms=1)
        assert len(snapshot["toolCalls"]) == 3
        assert len(snapshot["errors"]) == 2
        assert snapshot["truncated"] is True
        retained = json.dumps(
            {
                "snapshot": snapshot,
                "report": report.model_dump(mode="json", by_alias=True, exclude_none=True),
                "metricsRepr": repr(state.metrics),
            },
            sort_keys=True,
            default=str,
        )
        assert HTTP_AUTH_SECRET not in retained
        assert raw_canary not in retained
        await close_tools(tools)

    asyncio.run(scenario())


def retained_worker_surfaces(
    *,
    state: Any,
    session: dict[str, Any],
    history: list[dict[str, Any]],
    extra_repr: str,
) -> str:
    report = state.metrics.build_report(report_id="report-retention", duration_ms=7)
    return json.dumps(
        {
            "session": session,
            "history": history,
            "state": state.metrics.snapshot(),
            "report": report.model_dump(mode="json", by_alias=True, exclude_none=True),
            "metricsRepr": repr(state.metrics),
            "extraRepr": extra_repr,
        },
        sort_keys=True,
        default=str,
    )
