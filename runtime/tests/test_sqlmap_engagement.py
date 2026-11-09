"""Installed SQLMap must use the same HTTP authority as ordinary HTTP tools."""

import asyncio
import json
import shutil
import time
from types import SimpleNamespace

import pytest
from fakes.engagement_socket import SocketTarget, resolver_for, socket_spec, tls_contexts
from fakes.worker_state import WorkerState
from test_engagement_identity import identity_spec

from contractor_runtime.adapters.http_transport import HTTPTransport
from contractor_runtime.contracts.engagement import HTTPLimits
from contractor_runtime.toolsets.scan import tools as scan
from contractor_runtime.workspace import AllocationWorkspace

pytestmark = pytest.mark.skipif(shutil.which("sqlmap") is None, reason="real SQLMap is absent")


async def scanner(tmp_path, target, *, spec=None, tls=None, document=None):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = spec or socket_spec(target)
    owner = HTTPTransport(spec, resolver=resolver_for(target).resolver, ssl_context=tls)
    owner.configure(authority_deadline=lambda: time.monotonic() + 60)

    async def read(_ref, **_kwargs):
        return SimpleNamespace(data=json.dumps(document).encode(), media_type="application/json")

    factory = scan.ScanToolsetFactory(lambda *_: SimpleNamespace(read_artifact=read))
    tools = await factory.create_selected(
        selected=["scan_sqlmap"],
        allocation_id=spec.allocation_id,
        run_id=spec.run_id,
        namespace="scanner",
        runtime_settings=spec.runtime_settings,
        workspace=AllocationWorkspace(root=tmp_path, path=workspace),
        resources=WorkerState().resources,
        controlled_http=owner,
    )
    return tools["scan_sqlmap"], owner


@pytest.mark.parametrize("scheme,method", [("http", "POST"), ("https", "PATCH")])
@pytest.mark.parametrize("kind", ["bearer", "api-keys"])
def test_sqlmap_prepared_request_uses_shared_identity_and_cancels(tmp_path, scheme, method, kind):
    async def scenario():
        server, client, names = tls_contexts(tmp_path) if scheme == "https" else (None, None, [])
        async with SocketTarget(ssl_context=server) as target:
            fields = (
                {"token": "private-canary"}
                if kind == "bearer"
                else {
                    "header:Authorization": "Bearer private-canary",
                    "query:api_key": "private-query-canary",
                    "cookie:client": "private-cookie-canary",
                }
            )
            spec = identity_spec(target, [("account", kind, fields, None)])
            spec.engagement.http.limits = HTTPLimits(requestsPerSecond=10, concurrentRequests=1)
            document = {
                "schemaVersion": 1,
                "method": method,
                "url": target.origin + "/allowed?keep=fixed",
                "headers": [{"name": "Content-Type", "value": "application/json"}],
                "body": '{"id":7,"keep":"unchanged"}',
                "testParameters": ["id"],
            }
            tool, owner = await scanner(tmp_path, target, spec=spec, tls=client, document=document)
            invocation = asyncio.create_task(
                tool(
                    request_ref={"namespace": "inputs", "name": "request", "revision": "1"},
                    credential_id="credential-account",
                    timeout_seconds=30,
                )
            )
            received = asyncio.create_task(target.request_received.wait())
            try:
                done, _ = await asyncio.wait(
                    (invocation, received), timeout=20, return_when=asyncio.FIRST_COMPLETED
                )
                assert received in done, invocation.result() if invocation.done() else "no request"
                first = target.requests[0]
                query = "&api_key=private-query-canary" if kind == "api-keys" else ""
                expected = f"{method} /allowed?keep=fixed{query} HTTP/1.1".encode()
                assert first.startswith(expected), first
                if kind == "api-keys":
                    assert b"client=private-cookie-canary" in first
                assert (
                    b"authorization: Bearer private-canary" in first
                    or b"Authorization: Bearer private-canary" in first
                )
                assert first.endswith(document["body"].encode())
                assert owner._request_count >= 1
                if scheme == "https":
                    assert names == ["target.example"]
            finally:
                received.cancel()
                await tool.close()
                await asyncio.gather(invocation, received, return_exceptions=True)
                await owner.close()
            assert not list((tmp_path / "workspace").iterdir())
            before = len(target.requests)
            await asyncio.sleep(0.05)
            assert len(target.requests) == before

    asyncio.run(scenario())


def test_sqlmap_native_pacing_and_shared_response_bound(tmp_path, monkeypatch):
    native = scan.run_process
    commands = []

    async def run(command, directory, timeout):
        commands.append(command)
        if "-c" in command:
            from pathlib import Path

            config = Path(command[command.index("-c") + 1])
            assert config.stat().st_mode & 0o777 == 0o600
            assert "proxyCred = runtime:" in config.read_text()
        assert not any("runtime:" in arg for arg in command)
        return await native(command, directory, timeout)

    monkeypatch.setattr(scan, "run_process", run)

    async def scenario():
        async with SocketTarget() as target:
            spec = socket_spec(target)
            spec.engagement.http.limits = HTTPLimits(
                requestsPerSecond=4, concurrentRequests=1, responseBodyBytes=1
            )
            tool, owner = await scanner(tmp_path, target, spec=spec)
            try:
                result = await tool(
                    target.origin + "/allowed?id=1", parameter="id", timeout_seconds=20
                )
                assert result["status"] == "failed", result
                assert result["errorCode"] == "scan_request_failed", result
                assert len(target.requests) == 1
                launch = next(command for command in commands if "--batch" in command)
                assert "--delay=0.25" in launch
                assert "--threads=1" in launch
                assert "--retries=0" in launch
                assert result["httpLimits"] == {
                    "requestsPerSecond": 4,
                    "concurrentRequests": 1,
                    "responseBodyBytes": 1,
                }
                assert "private-canary" not in json.dumps(result)
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


def test_sqlmap_scope_denial_never_reaches_target(tmp_path):
    async def scenario():
        async with SocketTarget() as target:
            tool, owner = await scanner(tmp_path, target)
            try:
                result = await tool(
                    "http://outside.example/allowed?id=1", parameter="id", timeout_seconds=20
                )
                assert result["status"] == "failed", result
                assert result["errorCode"] == "scan_target_denied", result
                assert not target.connections
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


def test_sqlmap_completes_anonymous_scan_and_preserves_explicit_cookie(tmp_path):
    async def scenario():
        async with SocketTarget() as target:
            tool, owner = await scanner(tmp_path, target)
            try:
                result = await tool(
                    target.origin + "/allowed?id=1",
                    parameter="id",
                    cookie="fixture=keep",
                    timeout_seconds=30,
                )
                assert result["status"] == "completed", result
                assert result["exitCode"] == 0
                assert target.requests
                assert b"cookie: fixture=keep" in target.requests[0].lower()
                assert owner._request_count == len(target.requests)
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())
