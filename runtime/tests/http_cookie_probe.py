"""Production cookie Toolset against Go's private mTLS Artifact API and PostgreSQL."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.http_proxy import ProxyHTTPClient
from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import ArtifactAPIError, ArtifactClient
from contractor_runtime.contracts import ArtifactRef, HTTPOriginTargetSettings, RuntimeSettings
from contractor_runtime.toolsets.http.cookies import COOKIE_BINDING, CookieCheckpoint
from contractor_runtime.toolsets.http.tools import HTTPToolError, HTTPToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace


async def cookie_probe(client: ArtifactClient, action: str, proxy: bool) -> dict[str, Any]:
    if action == "faults":
        return await cookie_faults(client)
    calls = 0

    def target(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if request.url.path == "/login":
            return httpx.Response(
                201, headers={"set-cookie": "sid=target-session; Path=/; HttpOnly"}
            )
        if request.url.path == "/delete":
            return httpx.Response(200, headers={"set-cookie": "sid=; Max-Age=0; Path=/"})
        return httpx.Response(200 if request.headers.get("cookie") == "sid=target-session" else 401)

    async def create() -> tuple[dict[str, Any], httpx.AsyncClient]:
        transport = httpx.AsyncClient(transport=httpx.MockTransport(target), trust_env=False)
        factory = HTTPToolsetFactory(lambda: transport, artifact_client_factory=lambda *_: client)
        settings = RuntimeSettings(
            llmGatewayUrl="https://gateway.invalid/v1",
            artifactApiUrl="https://control.invalid",
            httpOriginTarget=HTTPOriginTargetSettings(url="https://target.example"),
            requestTimeoutSeconds=5,
        )
        tools = dict(
            await factory.create_selected(
                selected=sorted(factory.exported_tools),
                allocation_id=client.allocation_id,
                run_id="allocation-bound",
                namespace="builder",
                runtime_settings=settings,
                state=WorkerState(),
                workspace=AllocationWorkspace(
                    root=Path("/nonexistent/cookie-probe"),
                    path=Path("/nonexistent/cookie-probe/slot"),
                ),
                adapter_handles=AdapterHandles(tool_http=ProxyHTTPClient(transport))
                if proxy
                else AdapterHandles(),
            )
        )
        return tools, transport

    async def close(tools: dict[str, Any], transport: httpx.AsyncClient) -> None:
        await tools["http_request"].close()
        await transport.aclose()

    first, transport = await create()
    try:
        if action == "read":
            return {"count": (await first["http_session_get"]())["cookie_count"]}
        if action == "fenced":
            try:
                await first["http_session_clear"]()
            except HTTPToolError as error:
                return {"code": error.code, "retryable": error.retryable, "calls": calls}
            raise RuntimeError("fenced cookie write succeeded")
        if action == "delete":
            await first["http_request"]("https://target.example/delete")
            return {"count": (await first["http_session_get"]())["cookie_count"]}
        if action == "clear":
            return {"count": (await first["http_session_clear"]())["cookie_count"]}

        stale = CookieCheckpoint(client, "builder", ("https", "target.example", 443), lambda: 0, 5)
        await stale.restore()
        result = await first["http_request"]("https://target.example/login", method="POST")
        if "cookie_checkpoint" in result or result["status"] != 201:
            raise RuntimeError("cookie checkpoint did not reconcile")
        try:
            await stale.save(httpx.Cookies({"sid": "stale"}))
        except HTTPToolError as error:
            if error.code != "http_cookie_state_conflict":
                raise
        else:
            raise RuntimeError("stale cookie writer succeeded")
        value = await client.read_artifact(ArtifactRef(namespace="builder", name=COOKIE_BINDING))
        if b"target-session" not in value.data:
            raise RuntimeError("ordinary authorized read lost the cookie artifact")
    finally:
        await close(first, transport)
    # A newly constructed allocation client/toolset has no local jar, auth or
    # history; only the ordinary committed artifact can authenticate it.
    second, transport = await create()
    try:
        result = await second["http_request"]("https://target.example/private")
        return {
            "status": result["status"],
            "calls": calls,
            "revision": value.artifact.revision,
            "count": (await second["http_session_get"]())["cookie_count"],
        }
    finally:
        await close(second, transport)


async def cookie_faults(client: ArtifactClient) -> dict[str, Any]:
    class FaultClient:
        down = True
        entered: asyncio.Event
        release: asyncio.Event

        async def read_artifact(self, *args: Any, **kwargs: Any) -> Any:
            return await client.read_artifact(*args, **kwargs)

        async def write_artifact(self, *args: Any, **kwargs: Any) -> Any:
            if self.down:
                raise ArtifactAPIError(503, "storage_transaction_conflict", True)
            if hasattr(self, "entered"):
                self.entered.set()
                await self.release.wait()
            return await client.write_artifact(*args, **kwargs)

    fault = FaultClient()
    checkpoint = CookieCheckpoint(fault, "builder", ("https", "target.example", 443), lambda: 0, 5)
    await checkpoint.restore()
    try:
        await checkpoint.save(httpx.Cookies({"sid": "storage-recovered"}))
    except HTTPToolError as error:
        if error.code != "http_cookie_storage_unavailable":
            raise
    else:
        raise RuntimeError("storage fault unexpectedly succeeded")
    fault.down = False
    await checkpoint.recover()
    fault.entered, fault.release = asyncio.Event(), asyncio.Event()
    saving = asyncio.create_task(checkpoint.save(httpx.Cookies({"sid": "joined"})))
    await fault.entered.wait()
    saving.cancel()
    fault.release.set()
    try:
        await saving
    except asyncio.CancelledError:
        pass
    else:
        raise RuntimeError("cancelled checkpoint waiter returned success")
    replacement = CookieCheckpoint(
        client, "builder", ("https", "target.example", 443), lambda: 0, 5
    )
    cookies = await replacement.restore()
    if cookies.get("sid") != "joined":
        raise RuntimeError("cancelled write outlived its owner or lost its commit")
    return {"storage": "http_cookie_storage_unavailable", "joined": True}
