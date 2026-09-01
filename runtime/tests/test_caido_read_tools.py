from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.caido_graphql import CaidoGraphQLClient
from contractor_runtime.adapters.host import RuntimeAdapterMetricsState
from contractor_runtime.allocation import WorkerState
from contractor_runtime.contracts import (
    ArtifactRef,
    CaidoSettingsV2,
    RuntimeSettingsV2,
)
from contractor_runtime.toolsets.caido import (
    CAIDO_EXCHANGE_MEDIA_TYPE,
    CAIDO_READ_TOOL_NAMES,
    CaidoToolsetFactory,
)
from contractor_runtime.workspace import AllocationWorkspace

CAIDO_TOKEN = "recognizable-caido-tool-secret"
HTTPQL = 'req.host.eq:"target.example"'


def test_all_read_tools_use_static_operations_and_normalize_bounded_results(
    tmp_path: Path,
) -> None:
    raw_request = ("GET /long HTTP/1.1\r\nHost: target.example\r\n\r\n" + "r" * 9000).encode()
    raw_response = b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n\r\n\x00\xff"
    responses = representative_responses(raw_request, raw_response)
    observed: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        observed.append(payload)
        return httpx.Response(
            200,
            json={"data": responses[payload["operationName"]]},
            request=request,
        )

    async def scenario() -> None:
        artifacts = FakeArtifactClient()
        tools, state, handle = await create_tools(tmp_path, handler, artifacts)
        assert await CaidoToolsetFactory().probe() == CAIDO_READ_TOOL_NAMES

        scopes = await tools["caido_scope"]()
        assert scopes == {
            "scopes": [
                {
                    "id": "scope-1",
                    "name": "target",
                    "allowlist": ["*.target.example"],
                    "denylist": ["admin.target.example"],
                }
            ]
        }

        history = await tools["caido_history"](filter=HTTPQL, limit=20, offset=2)
        assert history["count"] == 1
        assert history["offset"] == 2
        assert history["requests"][0]["status_code"] == 200

        detail = await tools["caido_request_detail"]("request-1")
        assert detail["raw"]["truncated"] is True
        assert detail["response"]["raw"]["kind"] == "binary"
        exact = ArtifactRef.model_validate(detail["raw_artifact"]).require_exact()
        stored = json.loads(artifacts.payloads[(exact.namespace, exact.name, exact.revision)])
        assert stored["request"]["text"].encode() == raw_request
        assert base64.b64decode(stored["response"]["dataBase64"]) == raw_response
        assert artifacts.media_types[(exact.namespace, exact.name, exact.revision)] == (
            CAIDO_EXCHANGE_MEDIA_TYPE
        )

        automate = await tools["caido_automate_results"]("session-1", limit=10)
        assert automate["entry_id"] == "entry-2"
        assert automate["results"][0]["payloads"] == ["probe"]
        assert automate["results"][0]["status_code"] == 403

        root = await tools["caido_sitemap"](scope_id="scope-1")
        descendants = await tools["caido_sitemap"](parent_id="site-1", depth="ALL")
        assert root["entries"][0]["label"] == "target.example"
        assert descendants["entries"][0]["parent_id"] == "site-1"

        workflows = await tools["caido_workflow_list"]("active")
        assert workflows == {
            "workflows": [
                {
                    "id": "workflow-2",
                    "name": "CORS",
                    "kind": "active",
                    "enabled": True,
                    "global": False,
                }
            ]
        }
        findings = await tools["caido_workflow_findings"](limit=10)
        assert findings["findings"][0]["request_id"] == "request-1"

        assert [item["operationName"] for item in observed] == [
            "Scopes",
            "RequestsByOffset",
            "RequestDetail",
            "AutomateSession",
            "AutomateEntryRequests",
            "SitemapRoot",
            "SitemapDescendants",
            "Workflows",
            "FindingsByOffset",
        ]
        assert observed[1]["variables"] == {
            "filter": HTTPQL,
            "limit": 20,
            "offset": 2,
            "order": {"by": "ID", "ordering": "DESC"},
        }
        assert all(item["query"].startswith(("query ", "mutation ")) for item in observed)
        assert all(HTTPQL not in item["query"] for item in observed)
        assert state.metrics.counters["tool_calls"] == 8
        retained = repr(state.metrics)
        for forbidden in (CAIDO_TOKEN, HTTPQL, "request-1", raw_request.decode()):
            assert forbidden not in retained
        await close_tools(tools)
        await handle.close()

    asyncio.run(scenario())


async def create_tools(
    tmp_path: Path,
    handler: Any,
    artifacts: FakeArtifactClient,
) -> tuple[dict[str, Any], WorkerState, CaidoGraphQLClient]:
    metrics = RuntimeAdapterMetricsState()
    handle = CaidoGraphQLClient(
        endpoint="https://caido.example",
        bearer_token=CAIDO_TOKEN,
        ca_bundle_pem=None,
        timeout_seconds=5,
        metrics=metrics,
        transport=httpx.MockTransport(handler),
    )
    settings = RuntimeSettingsV2(
        llmGatewayUrl="https://gateway.example/v1",
        llmGatewayToken="gateway-token",
        artifactApiUrl="https://control.example/private/v1",
        caido=CaidoSettingsV2(
            adapter="caido-graphql@1",
            endpoint="https://caido.example",
            bearerToken=CAIDO_TOKEN,
            requestTimeoutSeconds=5,
        ),
        requestTimeoutSeconds=10,
    )
    state = WorkerState()
    factory = CaidoToolsetFactory(lambda _allocation, _settings: artifacts)
    result = await factory.create_selected(
        selected=sorted(CAIDO_READ_TOOL_NAMES),
        allocation_id="allocation-caido-tools",
        run_id="run-caido-tools",
        namespace="analyst",
        runtime_settings=settings,
        workspace=AllocationWorkspace(root=tmp_path, path=tmp_path / "allocation"),
        state=state,
        adapter_handles=AdapterHandles(caido_graphql=handle),
    )
    return dict(result), state, handle


async def close_tools(tools: dict[str, Any]) -> None:
    for tool in tools.values():
        await tool.close()


def representative_responses(request_raw: bytes, response_raw: bytes) -> dict[str, Any]:
    return {
        "Scopes": {
            "scopes": [
                {
                    "id": "scope-1",
                    "name": "target",
                    "allowlist": ["*.target.example"],
                    "denylist": ["admin.target.example"],
                }
            ]
        },
        "RequestsByOffset": {
            "requestsByOffset": {
                "count": {"value": 1},
                "nodes": [request_summary()],
            }
        },
        "RequestDetail": {"request": request_detail(request_raw, response_raw)},
        "AutomateSession": {
            "automateSession": {
                "id": "session-1",
                "name": "scan",
                "entries": [
                    {"id": "entry-1", "name": "old", "createdAt": "2026-09-01T00:00:00Z"},
                    {"id": "entry-2", "name": "new", "createdAt": "2026-09-01T00:01:00Z"},
                ],
                "settings": {"strategy": "ALL", "placeholders": [{"start": 1, "end": 2}]},
            }
        },
        "AutomateEntryRequests": {
            "automateEntry": {
                "id": "entry-2",
                "name": "new",
                "requestsByOffset": {
                    "count": {"value": 1},
                    "nodes": [
                        {
                            "sequenceId": 1,
                            "error": None,
                            "payloads": [
                                {"position": 0, "raw": base64.b64encode(b"probe").decode()}
                            ],
                            "request": {
                                "id": "automated-request-1",
                                "method": "GET",
                                "host": "target.example",
                                "path": "/probe",
                                "query": "",
                                "response": {
                                    "statusCode": 403,
                                    "length": 12,
                                    "roundtripTime": 4,
                                },
                            },
                        }
                    ],
                },
            }
        },
        "SitemapRoot": {
            "sitemapRootEntries": {
                "nodes": [
                    {
                        "id": "site-1",
                        "label": "target.example",
                        "kind": "DOMAIN",
                        "hasDescendants": True,
                        "metadata": {"isTls": True, "port": 443},
                    }
                ]
            }
        },
        "SitemapDescendants": {
            "sitemapDescendantEntries": {
                "nodes": [
                    {
                        "id": "site-2",
                        "label": "api",
                        "kind": "PATH",
                        "hasDescendants": False,
                        "parentId": "site-1",
                        "metadata": {},
                    }
                ]
            }
        },
        "Workflows": {
            "workflows": [
                {
                    "id": "workflow-1",
                    "name": "Convert",
                    "kind": "CONVERT",
                    "enabled": True,
                    "global": True,
                },
                {
                    "id": "workflow-2",
                    "name": "CORS",
                    "kind": "ACTIVE",
                    "enabled": True,
                    "global": False,
                },
            ]
        },
        "FindingsByOffset": {
            "findingsByOffset": {
                "count": {"value": 1},
                "nodes": [
                    {
                        "id": "finding-1",
                        "title": "Secret",
                        "description": "Potential secret",
                        "host": "target.example",
                        "path": "/config",
                        "reporter": "scanner",
                        "createdAt": "2026-09-01T00:02:00Z",
                        "request": {
                            "id": "request-1",
                            "method": "GET",
                            "host": "target.example",
                            "path": "/config",
                        },
                    }
                ],
            }
        },
    }


def request_summary() -> dict[str, Any]:
    return {
        "id": "request-1",
        "method": "GET",
        "host": "target.example",
        "path": "/long",
        "port": 443,
        "query": "",
        "isTls": True,
        "source": "PROXY",
        "createdAt": "2026-09-01T00:00:00Z",
        "response": {"statusCode": 200, "length": 9000, "roundtripTime": 12},
    }


def request_detail(request_raw: bytes, response_raw: bytes) -> dict[str, Any]:
    summary = request_summary()
    summary["raw"] = base64.b64encode(request_raw).decode()
    summary["response"] = {
        "id": "response-1",
        "statusCode": 200,
        "length": len(response_raw),
        "roundtripTime": 12,
        "raw": base64.b64encode(response_raw).decode(),
    }
    return summary


class FakeArtifactClient:
    def __init__(self) -> None:
        self.payloads: dict[tuple[str, str, str], bytes] = {}
        self.media_types: dict[tuple[str, str, str], str] = {}
        self._known: list[ArtifactRef] = []

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return tuple(self._known)

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> Any:
        assert expected_revision is None
        exact = target.model_copy(update={"revision": f"revision-{len(self._known) + 1}"})
        key = (exact.namespace, exact.name, exact.revision)
        self.payloads[key] = data
        self.media_types[key] = media_type
        self._known.append(exact)
        return SimpleNamespace(artifact=exact)
