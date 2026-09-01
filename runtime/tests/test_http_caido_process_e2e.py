"""Fast allocation-boundary companion to the PostgreSQL process E2E.

The Go build-tagged suite owns mTLS, label pinning, heterogeneous placement and
release-response loss.  This gate keeps the Python side fast while exercising
the complete AllocationSpec -> typed adapters -> ADK tools -> terminal cleanup
path with deterministic transports.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from fakes.model import scripted_model, text_result, tool_call
from fakes.spec import allocation_spec
from test_text_artifacts_toolset import MemoryArtifactClient

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.caido_graphql import CaidoGraphQLAdapter
from contractor_runtime.adapters.host import (
    RuntimeAdapterBuildContext,
    RuntimeAdapterMetricsState,
)
from contractor_runtime.adapters.http_proxy import ProxyHTTPClient
from contractor_runtime.allocation import AllocationService
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    ArtifactRef,
    CaidoSettingsV2,
    FinalizeAllocationRequest,
    HTTPProxySettingsV2,
    ReleaseAllocationRequest,
    RuntimeSettingsV2,
    StageContentRequest,
    ToolsetRef,
    ToolsetSelection,
)
from contractor_runtime.digests import _agent_template_digest
from contractor_runtime.factories import FactoryRegistry, built_in_factories
from contractor_runtime.state import ProcessState, RuntimeState

HTTP_SESSION_SECRET = "allocation-http-session-secret-canary"
PROXY_SECRET = "allocation-http-proxy-secret-canary"
CAIDO_SECRET = "allocation-caido-secret-canary"
RAW_BODY = "allocation-http-raw-body-canary"
REPORT = "# HTTP and Caido evidence\n\nBoth bounded operations completed.\n"


def test_http_caido_allocation_tools_finalize_release_and_reuse(tmp_path: Path) -> None:
    async def scenario() -> None:
        artifacts = MemoryArtifactClient()
        http_requests: list[httpx.Request] = []
        caido_operations: list[str] = []

        async def http_handler(request: httpx.Request) -> httpx.Response:
            http_requests.append(request)
            assert request.headers["authorization"] == f"Bearer {HTTP_SESSION_SECRET}"
            return httpx.Response(
                500,
                content=(RAW_BODY + "\n" + "bounded-" * 1200).encode(),
                headers={"content-type": "text/plain; charset=utf-8"},
                request=request,
            )

        async def caido_handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["authorization"] == f"Bearer {CAIDO_SECRET}"
            payload = json.loads(request.content)
            operation = payload["operationName"]
            caido_operations.append(operation)
            if operation == "RequestsByOffset":
                data: dict[str, Any] = {
                    "requestsByOffset": {
                        "count": {"value": 1},
                        "nodes": [
                            {
                                "id": "request-1",
                                "method": "GET",
                                "host": "target.example",
                                "path": "/observed",
                                "port": 443,
                                "query": "",
                                "isTls": True,
                                "source": "PROXY",
                                "createdAt": "2026-09-01T00:00:00Z",
                                "response": {
                                    "statusCode": 500,
                                    "length": 12000,
                                    "roundtripTime": 7,
                                },
                            }
                        ],
                    }
                }
            elif operation == "CreateScope":
                selected = payload["variables"]["input"]
                data = {
                    "createScope": {
                        "error": None,
                        "scope": {
                            "id": "scope-1",
                            "name": selected["name"],
                            "allowlist": selected["allowlist"],
                            "denylist": selected["denylist"],
                        },
                    }
                }
            else:  # pragma: no cover - an unexpected static operation is a hard failure
                raise AssertionError(operation)
            return httpx.Response(200, json={"data": data}, request=request)

        base = built_in_factories(
            tmp_path / "work",
            artifact_client_factory=lambda *_: artifacts,
            model_factory=lambda _: worker_model(),
            enabled_runtime_adapters=[],
        )
        http_factory = MockHTTPProxyAdapterFactory(http_handler)
        caido_factory = MockCaidoAdapterFactory(caido_handler)
        factories = FactoryRegistry(
            worker_runtimes=base.worker_runtimes,
            toolsets=base.toolsets,
            sandbox_profiles=base.sandbox_profiles,
            runtime_adapters={http_factory.ref: http_factory, caido_factory.ref: caido_factory},
            artifact_client_factory=base.artifact_client_factory,
        )
        capabilities = CapabilitySnapshot.create(
            runtimes=factories.worker_runtimes,
            toolsets={ref: factory.exported_tools for ref, factory in factories.toolsets.items()},
            sandbox_profiles=factories.sandbox_profiles,
            runtime_adapters=factories.runtime_adapters,
        )
        state = RuntimeState(instance_id="runtime-http-caido")
        await state.mark_registered()
        service = AllocationService(
            state,
            factories,
            capabilities,
            a2a_base_url="https://runtime.example",
        )
        spec = http_caido_spec()

        await service.prepare(spec)
        assert service._context is not None and service._context.worker is not None
        result = await service._context.worker.invoke(stage_request())

        assert result.outcome.value == "succeeded"
        assert result.artifacts["report"].revision == "revision-2"
        assert [request.url.host for request in http_requests] == ["target.example"]
        assert caido_operations == ["RequestsByOffset", "CreateScope"]
        body = artifacts.bindings[
            (
                "analysis",
                next(
                    name
                    for namespace, name in artifacts.bindings
                    if namespace == "analysis" and name.startswith("http.body.")
                ),
            )
        ]
        assert RAW_BODY.encode() in body.data

        terminal = await service.finalize(
            FinalizeAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                finalizationId="finalize-http-caido",
                deadline=datetime.now(UTC) + timedelta(seconds=3),
            )
        )
        await service.release(
            ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
        )
        await service.confirm_release(spec.allocation_id)

        assert terminal.report.worker.complete
        assert (await state.snapshot()).process_state is ProcessState.IDLE
        assert await service.snapshot() is None
        assert list((tmp_path / "work").iterdir()) == []
        assert all(adapter.closed for adapter in http_factory.adapters)
        assert all(adapter.handles.enabled_channels == () for adapter in caido_factory.adapters)
        retained = f"{terminal!r} {state!r} {http_factory.adapters!r} {caido_factory.adapters!r}"
        for secret in (HTTP_SESSION_SECRET, PROXY_SECRET, CAIDO_SECRET, RAW_BODY):
            assert secret not in retained

    asyncio.run(scenario())


def http_caido_spec() -> Any:
    spec = allocation_spec(allocation_id="allocation-http-caido")
    spec.logical_agent_name = "analyst"
    spec.namespace = "analysis"
    spec.agent_template.ref.template_id = "caido_analyst"
    spec.agent_template.description = "Performs bounded HTTP and Caido analysis"
    spec.agent_template.toolsets = [
        ToolsetSelection(
            ref=ToolsetRef(toolsetId="http-tools", version="1"),
            tools=[
                "http_history",
                "http_read_body",
                "http_request",
                "http_session_clear",
                "http_session_set",
            ],
        ),
        ToolsetSelection(
            ref=ToolsetRef(toolsetId="caido", version="1"),
            tools=["caido_history", "caido_scope"],
        ),
        ToolsetSelection(
            ref=ToolsetRef(toolsetId="text-artifacts", version="1"),
            tools=["write_text_artifact"],
        ),
    ]
    spec.agent_template.ref.digest = _agent_template_digest(spec.agent_template)
    spec.runtime_settings = RuntimeSettingsV2(
        llmGatewayUrl="https://gateway.example/v1",
        llmGatewayToken="gateway-token",
        artifactApiUrl="https://control.example/private/v1",
        httpProxy=HTTPProxySettingsV2(
            adapter="http-proxy@1",
            proxyUrl="https://proxy.example",
            bearerToken=PROXY_SECRET,
            targets=["tool-http"],
        ),
        caido=CaidoSettingsV2(
            adapter="caido-graphql@1",
            endpoint="https://caido.example",
            bearerToken=CAIDO_SECRET,
            requestTimeoutSeconds=5,
        ),
        requestTimeoutSeconds=10,
    )
    spec.resolved_runtime_config_provenance.runtime_adapters = [
        "caido-graphql@1",
        "http-proxy@1",
    ]
    return spec


def worker_model() -> object:
    return scripted_model(
        [
            tool_call(
                "http_session_set",
                {
                    "auth": {"kind": "bearer", "token": HTTP_SESSION_SECRET},
                    "replace_cookies": True,
                    "replace_headers": True,
                },
                call_id="session-set",
            ),
            tool_call(
                "http_request",
                {
                    "url": "https://target.example/failure",
                    "method": "POST",
                    "body_type": "none",
                    "follow_redirects": False,
                },
                call_id="http-request",
            ),
            tool_call(
                "http_read_body",
                {"request_id": 1, "offset": 0, "length": 8192},
                call_id="http-body",
            ),
            tool_call(
                "caido_history",
                {"filter": "", "limit": 5, "offset": 0},
                call_id="caido-history",
            ),
            tool_call(
                "caido_scope",
                {
                    "action": "create",
                    "name": "allocation-scope",
                    "allowlist": ["target.example"],
                    "denylist": [],
                },
                call_id="caido-scope",
            ),
            tool_call(
                "write_text_artifact",
                {
                    "name": "report",
                    "text": REPORT,
                    "media_type": "text/markdown",
                    "expected_revision": None,
                },
                call_id="write-report",
            ),
            tool_call("http_session_clear", {}, call_id="session-clear"),
            text_result("Bounded HTTP and Caido analysis completed"),
        ]
    )


def stage_request() -> StageContentRequest:
    return StageContentRequest(
        apiVersion=API_VERSION,
        objective="Collect bounded HTTP and Caido evidence",
        instructions="Use only selected tools and publish one report.",
        parameters={"target": "target.example"},
        artifacts={},
        resultArtifacts={"report": ArtifactRef(namespace="analysis", name="report")},
    )


class MockHTTPProxyAdapterFactory:
    ref = "http-proxy@1"

    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self.adapters: list[MockHTTPProxyAdapter] = []

    async def probe(self) -> bool:
        return True

    async def create(self, context: RuntimeAdapterBuildContext, settings: Any) -> Any:
        del context
        assert isinstance(settings, HTTPProxySettingsV2)
        adapter = MockHTTPProxyAdapter(self._handler)
        self.adapters.append(adapter)
        return adapter


class MockHTTPProxyAdapter:
    ref = "http-proxy@1"

    def __init__(self, handler: Any) -> None:
        self.metrics = RuntimeAdapterMetricsState()
        self._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
        self._handle = ProxyHTTPClient(self._client, metrics=self.metrics)
        self.handles = AdapterHandles(tool_http=self._handle)
        self.closed = False

    async def flush(self) -> None:
        return

    async def close(self) -> None:
        await self._client.aclose()
        self._handle.detach()
        self.handles = AdapterHandles()
        self.closed = True


class MockCaidoAdapterFactory:
    ref = "caido-graphql@1"

    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self.adapters: list[CaidoGraphQLAdapter] = []

    async def probe(self) -> bool:
        return True

    async def create(
        self, context: RuntimeAdapterBuildContext, settings: Any
    ) -> CaidoGraphQLAdapter:
        assert isinstance(settings, CaidoSettingsV2)
        adapter = CaidoGraphQLAdapter(
            context,
            settings,
            transport=httpx.MockTransport(self._handler),
        )
        self.adapters.append(adapter)
        return adapter
