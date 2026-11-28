"""Real Chromium against real sockets, with the ordinary allocation HTTP owner."""

import asyncio
import io
import json
import time
from types import SimpleNamespace

import pytest
from fakes.engagement_socket import SocketTarget, preflight, socket_spec
from fakes.worker_state import WorkerState
from PIL import Image
from test_text_artifacts_toolset import MemoryArtifactClient

from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.toolsets.browser.tools import BrowserToolsetFactory
from contractor_runtime.toolsets.common.sources import SourcePreparer


async def browser_tools(owner):
    state = WorkerState()
    client = MemoryArtifactClient()
    tools = await BrowserToolsetFactory(lambda *_: client).create_selected(
        selected=sorted(BrowserToolsetFactory.exported_tools),
        allocation_id="a1",
        run_id="r1",
        namespace="frontend",
        runtime_settings=owner._settings,
        workspace=None,
        resources=state.resources,
        controlled_http=owner,
    )
    return tools, client, state


async def frontend(target, reader, writer, message):
    path = message.split(b" ")[1]
    if path == b"/allowed/api":
        data = b'{"loaded":true}'
        media = b"application/json"
    elif path == b"/allowed/redirect":
        writer.write(b"HTTP/1.1 302 Found\r\nLocation: /forbidden\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        return
    else:
        data = b"""<!doctype html><title>Local frontend</title>
        <h1>Orders</h1><button id="load" onclick="fetch('/allowed/api')
        .then(r=>r.json()).then(x=>document.querySelector('h1').textContent='Loaded orders')">
        Load</button>
        <button id="deny" onclick="fetch('/forbidden');fetch('http://other.example/no');fetch('/allowed/api',{method:'DELETE'})">Denied</button>
        <a href="/allowed/next">Next</a><input id="pw" type="password" value="private-password">
        <script>new WebSocket('ws://target.example/allowed/socket');
        if ('serviceWorker' in navigator)
        navigator.serviceWorker.register('/allowed/sw.js').catch(()=>{});</script>"""
        media = b"text/html"
    writer.write(
        b"HTTP/1.1 200 OK\r\nContent-Type: "
        + media
        + b"\r\nContent-Length: "
        + str(len(data)).encode()
        + b"\r\n\r\n"
        + data
    )
    await writer.drain()


def test_real_chromium_route_map_screenshot_denials_and_cleanup():
    async def scenario():
        async with SocketTarget(frontend) as target:
            spec = socket_spec(target)
            spec.engagement.http.routes[0].path_prefixes = ["/allowed"]
            spec.engagement.http.routes[0].methods = ["GET"]
            async with preflight(target, spec=spec) as (owner, _):
                owner.configure(authority_deadline=lambda: time.monotonic() + 120)
                tools, client, state = await browser_tools(owner)
                session = tools["browser_open"]._session
                try:
                    assert not target.requests
                    opened = await tools["browser_open"]()
                    context = opened["context_id"]
                    ctx = SimpleNamespace(invocation_id="browser-test")
                    nav = await tools["browser_navigate"](context, target.origin + "/allowed", ctx)
                    assert nav["outcome"] == "observed"
                    snap = await tools["browser_snapshot"](context)
                    assert snap["title"] == "Local frontend"
                    assert "private-password" not in json.dumps(snap)
                    page = session.state(context).page
                    assert await page.evaluate("typeof RTCPeerConnection") == "undefined"
                    assert await page.evaluate("typeof WebTransport") == "undefined"
                    action = await tools["browser_click"](context, "#load", ctx)
                    assert action["outcome"] == "observed"
                    shot = await tools["browser_screenshot"](context, "orders.png")
                    image_ref = shot["artifact"]
                    image = await client.read_artifact(ArtifactRef.model_validate(image_ref))
                    assert Image.open(io.BytesIO(image.data)).size == (1280, 720)
                    await tools["browser_click"](context, "#deny", ctx)
                    await tools["browser_navigate"](
                        context, target.origin + "/allowed/redirect", ctx
                    )
                    exported = await tools["browser_export_map"]()
                    ref = exported["artifact"]
                    artifact = await client.read_artifact(ArtifactRef.model_validate(ref))
                    route_map = json.loads(artifact.data)
                    api = [
                        r
                        for r in route_map["requests"]
                        if r["url"].endswith("/allowed/api") and r["method"] == "GET"
                    ]
                    assert len(api) == 1 and api[0]["action_id"] == action["id"]
                    assert api[0]["status"] == 200
                    captured = await state.resources.http_captures.get(
                        "http", str(api[0]["request_id"])
                    )
                    assert captured["attempts"][0]["response"]["status"] == 200
                    assert route_map["screenshots"][0]["artifact"] == image_ref
                    assert "websocket_blocked" in route_map["gaps"]
                    assert all(
                        b" /forbidden " not in r and b"DELETE " not in r and b"sw.js" not in r
                        for r in target.requests
                    )
                    assert (
                        state.resources.selected_versions.selected(
                            "frontend", "orders.png"
                        ).revision
                        == image_ref["revision"]
                    )
                    selected = await SourcePreparer(
                        client, state.resources, "r1", "frontend", 10
                    ).prepare(
                        [{"kind": "artifact", "namespace": "frontend", "name": "orders.png"}],
                        "browser-test",
                        "finding-call",
                    )
                    assert selected == [ArtifactRef.model_validate(image_ref)]
                    browser = session._browser
                finally:
                    await session.close()
                assert not browser.is_connected()
                assert not session.contexts

    asyncio.run(scenario())


def test_browser_requires_engagement_and_assigned_credentials():
    async def scenario():
        with pytest.raises(ValueError, match="engagement"):
            from contractor_runtime.toolsets.browser.session import BrowserSession

            BrowserSession(None, None)
        async with SocketTarget(frontend) as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            tools, _, _ = await browser_tools(owner)
            try:
                with pytest.raises(ToolInputError, match="Credential"):
                    await tools["browser_open"]("foreign")
                assert not target.requests
            finally:
                await tools["browser_open"].close()

    asyncio.run(scenario())


def test_browser_identities_storage_and_allocation_stop():
    from test_engagement_identity import identity_spec

    async def scenario():
        async with SocketTarget(frontend) as target:
            spec = identity_spec(
                target,
                [
                    ("account-a", "cookies", {"session": "canary-cookie-a"}, None),
                    ("account-b", "bearer", {"token": "canary-token-b"}, None),
                ],
            )
            async with preflight(target, spec=spec) as (owner, _):
                owner.configure(authority_deadline=lambda: time.monotonic() + 120)
                tools, _, _state = await browser_tools(owner)
                session = tools["browser_open"]._session
                contexts = []
                try:
                    for credential in [None, "credential-account-a", "credential-account-b"]:
                        opened = await tools["browser_open"](credential)
                        contexts.append(opened["context_id"])
                        assert "canary" not in json.dumps(opened)
                        await tools["browser_navigate"](
                            opened["context_id"], target.origin + "/allowed"
                        )
                    documents = [r for r in target.requests if r.startswith(b"GET /allowed ")]
                    assert len(documents) == 3
                    assert b"canary" not in documents[0]
                    assert (
                        b"canary-cookie-a" in documents[1] and b"canary-token-b" not in documents[1]
                    )
                    assert (
                        b"canary-token-b" in documents[2] and b"canary-cookie-a" not in documents[2]
                    )
                    # Target storage is separate; test-only evaluation cannot be
                    # requested by the model-facing browser tool interface.
                    await session.state(contexts[1]).page.evaluate(
                        "localStorage.setItem('identity','a')"
                    )
                    assert (
                        await session.state(contexts[0]).page.evaluate(
                            "localStorage.getItem('identity')"
                        )
                        is None
                    )
                    assert (
                        await session.state(contexts[2]).page.evaluate(
                            "localStorage.getItem('identity')"
                        )
                        is None
                    )
                    assert "canary" not in session.export().decode()
                    browser = session._browser
                    await owner.close()
                    assert not browser.is_connected() and not session.contexts
                    assert spec.runtime_settings.http_identities is None
                finally:
                    await session.close()

    asyncio.run(scenario())


def test_browser_cancellation_joins_inflight_request_and_lease_loss_denies():
    from contractor_runtime.adapters.http_transport import HTTPTransportError

    async def scenario():
        entered = asyncio.Event()

        async def hold(target, reader, writer, message):
            entered.set()
            await reader.read()

        async with SocketTarget(hold) as target, preflight(target) as (owner, _):
            lease = time.monotonic() + 120
            owner.configure(authority_deadline=lambda: lease)
            tools, _, _ = await browser_tools(owner)
            session = tools["browser_open"]._session
            context = (await tools["browser_open"]())["context_id"]
            browser = session._browser
            task = asyncio.create_task(
                tools["browser_navigate"](context, target.origin + "/allowed")
            )
            try:
                await asyncio.wait_for(entered.wait(), 15)
                lease = time.monotonic() - 1
                with pytest.raises(HTTPTransportError):
                    await tools["browser_snapshot"](context)
                await owner.close()
                await asyncio.gather(task, return_exceptions=True)
                assert task.done() and not browser.is_connected()
                assert not session.contexts and not owner._tasks
                await asyncio.wait_for(target.all_closed.wait(), 5)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await session.close()

    asyncio.run(scenario())


def test_browser_https_service_worker_blocking_and_dns_address_denial(tmp_path):
    from fakes.engagement_socket import resolver_for, tls_contexts

    async def scenario():
        server_tls, client_tls, _ = tls_contexts(tmp_path)
        async with SocketTarget(frontend, ssl_context=server_tls) as target:
            answers = [target.address]
            config = resolver_for(target, answers=lambda: answers)
            async with preflight(target, ssl_context=client_tls, config=config) as (owner, _):
                owner.configure(authority_deadline=lambda: time.monotonic() + 120)
                tools, _, _ = await browser_tools(owner)
                session = tools["browser_open"]._session
                try:
                    context = (await tools["browser_open"]())["context_id"]
                    action = await tools["browser_navigate"](context, target.origin + "/allowed")
                    assert action["outcome"] == "observed"
                    page = session.state(context).page
                    assert await page.evaluate("'serviceWorker' in navigator")
                    blocked = await page.evaluate(
                        "navigator.serviceWorker.register('/allowed/sw.js').then(()=>false,()=>true)"
                    )
                    assert blocked is True
                    assert not any(b"sw.js" in row for row in target.requests)
                    before = len(target.requests)
                    # DNS resolves the actual socket target, but the allowed
                    # address range no longer contains it. No application bytes.
                    answers[:] = ["10.255.255.254"]
                    await tools["browser_navigate"](context, target.origin + "/allowed/new")
                    assert len(target.requests) == before
                    assert session.requests[-1]["status"] is None
                finally:
                    await session.close()

    asyncio.run(scenario())


def test_browser_optional_http_limits_are_shared_with_ordinary_http_tools():
    from test_audit_http_credentials import tools_for

    from contractor_runtime.contracts.engagement import HTTPLimits

    async def scenario():
        async with SocketTarget(frontend) as target:
            spec = socket_spec(target)
            async with preflight(target, spec=spec, limits=HTTPLimits(responseBodyBytes=16)) as (
                owner,
                _,
            ):
                owner.configure(authority_deadline=lambda: time.monotonic() + 120)
                browser, _, _ = await browser_tools(owner)
                session = browser["browser_open"]._session
                try:
                    context = (await browser["browser_open"]())["context_id"]
                    result = await browser["browser_navigate"](context, target.origin + "/allowed")
                    assert result["outcome"] == "incomplete"
                    assert session.requests[0]["error"]
                    assert not session.requests[0]["status"]
                    assert owner.configured_limits() == {"responseBodyBytes": 16}
                    http, _ = await tools_for(owner)
                    bounded = await http["http_request"](target.origin + "/allowed")
                    assert bounded["body_truncated"] is True
                    assert bounded["content_length"] <= 16
                finally:
                    await session.close()

    asyncio.run(scenario())


def test_actual_worker_browser_tools_produce_required_route_map(tmp_path):
    from dataclasses import replace

    from fakes.model import scripted_model, text_result, tool_call
    from test_adk_runtime import build_context, stage_request

    from contractor_runtime.worker.factory import AdkWorkerRuntimeFactory

    async def scenario():
        async with SocketTarget(frontend) as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 120)
            tools, client, state = await browser_tools(owner)
            calls = [
                ("browser_open", {}),
                ("browser_navigate", {"context_id": "b1", "url": target.origin + "/allowed"}),
                ("browser_click", {"context_id": "b1", "selector": "#load"}),
                ("browser_screenshot", {"context_id": "b1", "name": "orders.png"}),
                ("browser_export_map", {"name": "route-map"}),
            ]
            model = scripted_model(
                [
                    *[
                        tool_call(name, arguments, call_id=f"browser-{i}")
                        for i, (name, arguments) in enumerate(calls)
                    ],
                    text_result("Frontend observations saved."),
                ]
            )
            context = replace(build_context(tmp_path, state, tools), namespace="frontend")
            runtime = await AdkWorkerRuntimeFactory(
                artifact_client_factory=lambda *_: client, model_factory=lambda _: model
            ).create(context)
            request = stage_request().model_copy(
                update={
                    "artifacts": {},
                    "result_artifacts": {
                        "route-map": ArtifactRef(namespace="frontend", name="route-map")
                    },
                }
            )
            try:
                result = await runtime.invoke(request)
                assert result.failure is None
                assert result.result.artifacts["route-map"].revision
                selected = result.result.artifacts["route-map"]
                data = json.loads((await client.read_artifact(selected)).data)
                assert any(
                    r["url"].endswith("/allowed/api") and r["action_id"] == "a2"
                    for r in data["requests"]
                )
                assert data["screenshots"][0]["after_action_id"] == "a2"
            finally:
                from datetime import UTC, datetime, timedelta

                await runtime.abort(datetime.now(UTC) + timedelta(seconds=10))
                await tools["browser_open"].close()

    asyncio.run(scenario())


def test_browser_timeout_retains_incomplete_action_without_replay():
    async def scenario():
        async def hold(target, reader, writer, message):
            await reader.read()

        async with SocketTarget(hold) as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            owner._settings.request_timeout_seconds = 1
            tools, _, _ = await browser_tools(owner)
            session = tools["browser_open"]._session
            try:
                context = (await tools["browser_open"]())["context_id"]
                result = await tools["browser_navigate"](context, target.origin + "/allowed")
                assert result["outcome"] == "incomplete"
                assert len(target.requests) == 1
                assert len(session.actions) == 1
                assert "action_incomplete_do_not_retry_automatically" in session.gaps
                await owner.close()
                assert not session.contexts and not owner._tasks
                await asyncio.wait_for(target.all_closed.wait(), 5)
            finally:
                await session.close()

    asyncio.run(scenario())
