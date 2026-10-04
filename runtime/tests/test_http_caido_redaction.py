"""Secret-retention regressions for HTTP/Caido tools and execution reports."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_caido_read_tools import CAIDO_TOKEN, request_detail
from test_caido_read_tools import FakeArtifactClient as CaidoArtifactClient
from test_caido_read_tools import close_tools as close_caido_tools
from test_caido_read_tools import create_tools as create_caido_tools
from test_http_toolset import FakeArtifactClient, close_tools, create_tools

import contractor_runtime.telemetry.metrics as runtime_metrics
from contractor_runtime.contracts import (
    ArtifactRef,
    CaidoSettings,
    HTTPOriginTargetSettings,
    HTTPProxySettings,
    RuntimeSettings,
)
from contractor_runtime.toolsets.caido.tools import (
    CAIDO_TOOL_NAMES,
    CaidoToolError,
    CaidoToolsetFactory,
)
from contractor_runtime.toolsets.common.target_policy import TargetPolicyConfig, TargetUnresolved
from contractor_runtime.toolsets.http.tools import HTTPToolError

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
TARGET_TOKEN = "canary-target-bearer-4f1e9"
GATEWAY_TOKEN = "canary-gateway-token-8c2d7"
PROXY_USERNAME = "ops"
PROXY_PASSWORD = "pw"
PROXY_AUTHORIZATION = "Basic " + base64.b64encode(b"ops:pw").decode()


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


def test_caido_results_and_artifacts_scrub_runtime_injected_credentials(
    tmp_path: Path,
) -> None:
    """Caido records what the tool-http proxy route sent, Runtime credentials included."""

    raw_request = (
        "GET /api/me HTTP/1.1\r\n"
        "Host: target.example\r\n"
        f"Authorization: Bearer {TARGET_TOKEN}\r\n"
        f"Proxy-Authorization: {PROXY_AUTHORIZATION}\r\n"
        "X-Team: ops-pw\r\n\r\n"
    ).encode()
    raw_response = (
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n"
        f'{{"echo":"Bearer {TARGET_TOKEN}","gateway":"{GATEWAY_TOKEN}","team":"ops"}}'
    ).encode()
    convert_output = f"Authorization: Bearer {TARGET_TOKEN}\n" + "x" * 9000
    replay_raw: str | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal replay_raw
        payload = json.loads(request.content)
        operation = payload["operationName"]
        if operation == "RequestDetail":
            data: dict[str, Any] = {"request": request_detail(raw_request, raw_response)}
        elif operation == "CreateReplaySession":
            data = {
                "createReplaySession": {
                    "session": {"id": "session", "name": "replay", "activeEntry": None}
                }
            }
        elif operation == "StartReplayTask":
            replay_raw = payload["variables"]["input"]["raw"]
            data = {
                "startReplayTask": {
                    "error": None,
                    "task": {"id": "task", "replayEntry": {"id": "entry"}},
                }
            }
        elif operation == "ReplayEntry":
            data = {
                "replayEntry": {
                    "id": "entry",
                    "raw": replay_raw,
                    "error": None,
                    "request": {
                        "id": "replayed",
                        "method": "GET",
                        "host": "target.example",
                        "path": "/api/me",
                        "query": "",
                        "response": {
                            "statusCode": 200,
                            "length": len(raw_response),
                            "roundtripTime": 3,
                            "raw": base64.b64encode(raw_response).decode(),
                        },
                    },
                }
            }
        elif operation == "FindingsByOffset":
            data = {
                "findingsByOffset": {
                    "count": {"value": 1},
                    "nodes": [
                        {
                            "id": "finding-1",
                            "title": "Bearer token observed",
                            "description": f"Authorization: Bearer {TARGET_TOKEN}",
                            "host": "target.example",
                            "path": "/api/me",
                            "reporter": "passive-workflow",
                            "createdAt": "2026-09-01T00:00:00Z",
                            "request": None,
                        }
                    ],
                }
            }
        elif operation == "RunConvertWorkflow":
            data = {
                "runConvertWorkflow": {
                    "output": base64.b64encode(convert_output.encode()).decode(),
                    "error": None,
                }
            }
        else:
            raise AssertionError(f"unexpected operation {operation}")
        return httpx.Response(200, json={"data": data}, request=request)

    async def scenario() -> None:
        artifacts = CaidoArtifactClient()
        tools, _state, handle = await create_caido_tools(
            tmp_path,
            handler,
            artifacts,
            selected=CAIDO_TOOL_NAMES,
            factory=caido_credential_factory(artifacts),
            runtime_settings=caido_credential_settings(TARGET_TOKEN),
        )
        detail = await tools["caido_request_detail"]("request-1")
        replay = await tools["caido_replay"](request_id="request-1")
        findings = await tools["caido_workflow_findings"]()
        converted = await tools["caido_workflow_run"]("workflow-convert", input="source")

        request_preview = detail["raw"]["data"]
        assert "Host: target.example\r\n" in request_preview
        assert "Authorization: [runtime-target-credential]\r\n" in request_preview
        assert "Proxy-Authorization: [runtime-credential]\r\n" in request_preview
        # Short private values match only as a complete credential header value.
        assert "X-Team: ops-pw\r\n" in request_preview
        response_preview = detail["response"]["raw"]["data"]
        assert response_preview.endswith(
            '{"echo":"[REDACTED]","gateway":"[REDACTED]","team":"ops"}'
        )
        assert detail["raw"]["size"] == len(request_preview.encode())
        assert "Authorization: [runtime-target-credential]\r\n" in replay["raw"]["data"]
        assert replay["response_raw"]["data"] == response_preview
        assert findings["findings"][0]["description"] == "Authorization: [REDACTED]"
        assert converted["output"].startswith("Authorization: [runtime-target-credential]\n")

        exchange = ArtifactRef.model_validate(detail["raw_artifact"]).require_exact()
        stored = json.loads(artifacts.payloads[exact_key(exchange)])
        assert stored["request"]["text"] == request_preview
        assert stored["response"]["text"] == response_preview
        output = ArtifactRef.model_validate(converted["output_artifact"]).require_exact()
        assert artifacts.payloads[exact_key(output)].startswith(
            b"Authorization: [runtime-target-credential]\n"
        )

        returned = json.dumps([detail, replay, findings, converted])
        stored_bytes = b"\n".join(artifacts.payloads.values())
        assert len(artifacts.payloads) == 3
        for forbidden in (TARGET_TOKEN, GATEWAY_TOKEN, PROXY_AUTHORIZATION):
            assert forbidden not in returned
            assert forbidden.encode() not in stored_bytes
        await close_caido_tools(tools)
        await handle.close()

    asyncio.run(scenario())


def test_caido_short_injected_credentials_are_replaced_as_whole_header_values(
    tmp_path: Path,
) -> None:
    short_token = "s3cr3t-7"
    raw_request = (
        "GET / HTTP/2\r\n"
        "host: target.example\r\n"
        f"authorization: bearer {short_token}\r\n"
        "x-request: keep\r\n\r\n"
    ).encode()
    binary_response = b"HTTP/1.1 200 OK\r\n\r\n\xff" + GATEWAY_TOKEN.encode() + b"\x00"

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["operationName"] == "RequestDetail"
        data = {"request": request_detail(raw_request, binary_response)}
        return httpx.Response(200, json={"data": data}, request=request)

    async def scenario() -> None:
        artifacts = CaidoArtifactClient()
        tools, _state, handle = await create_caido_tools(
            tmp_path,
            handler,
            artifacts,
            selected={"caido_request_detail"},
            factory=caido_credential_factory(artifacts),
            runtime_settings=caido_credential_settings(short_token),
        )
        detail = await tools["caido_request_detail"]("request-1")
        assert detail["raw"]["data"] == (
            "GET / HTTP/2\r\n"
            "host: target.example\r\n"
            "authorization: [runtime-target-credential]\r\n"
            "x-request: keep\r\n\r\n"
        )
        response = base64.b64decode(detail["response"]["raw"]["data_b64"])
        assert response == b"HTTP/1.1 200 OK\r\n\r\n\xff[REDACTED]\x00"
        exchange = ArtifactRef.model_validate(detail["raw_artifact"]).require_exact()
        stored = json.loads(artifacts.payloads[exact_key(exchange)])
        assert stored["request"]["text"] == detail["raw"]["data"]
        assert base64.b64decode(stored["response"]["dataBase64"]) == response
        await close_caido_tools(tools)
        await handle.close()

    asyncio.run(scenario())


def caido_credential_settings(target_token: str) -> RuntimeSettings:
    return RuntimeSettings(
        llmGatewayUrl="https://gateway.example/v1",
        llmGatewayToken=GATEWAY_TOKEN,
        artifactApiUrl="https://control.example/private/v1",
        httpProxy=HTTPProxySettings(
            adapter="http-proxy@1",
            proxyUrl="http://caido-proxy.example:8080",
            basicAuth={"username": PROXY_USERNAME, "password": PROXY_PASSWORD},
            targets=["tool-http"],
        ),
        caido=CaidoSettings(
            adapter="caido-graphql@1",
            endpoint="https://caido.example",
            bearerToken=CAIDO_TOKEN,
            requestTimeoutSeconds=5,
        ),
        httpOriginTarget=HTTPOriginTargetSettings(
            url="http://target.example", bearerToken=target_token
        ),
        requestTimeoutSeconds=10,
    )


def caido_credential_factory(artifacts: CaidoArtifactClient) -> CaidoToolsetFactory:
    async def unresolved(_host: str, _port: int) -> tuple[()]:
        raise TargetUnresolved

    return CaidoToolsetFactory(
        lambda _allocation, _settings: artifacts,
        target_policy=TargetPolicyConfig(resolver=unresolved),
    )


def exact_key(ref: ArtifactRef) -> tuple[str, str, str]:
    if ref.revision is None:
        raise AssertionError("artifact ref is not exact")
    return ref.namespace, ref.name, ref.revision


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
