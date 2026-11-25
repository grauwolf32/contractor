"""Pinned ffuf conformance through the real allocation transport and TCP/TLS peers."""

import asyncio
import json
import shutil
import time
import tomllib
from pathlib import Path

import pytest
from fakes.engagement_socket import SocketTarget, resolver_for, socket_spec, tls_contexts
from fakes.http_transport import HTTPTransport
from fakes.worker_state import WorkerState
from test_engagement_identity import identity_spec
from test_tool_worker import ArtifactStore

from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.contracts.engagement import HTTPLimits
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.toolsets.common.report_observations import project_report
from contractor_runtime.toolsets.scan import tools as scan
from contractor_runtime.toolsets.scan.process import ProcessResult
from contractor_runtime.workspace import AllocationWorkspace

pytestmark = pytest.mark.skipif(shutil.which("ffuf") is None, reason="optional ffuf is absent")
REF = {"namespace": "inputs", "name": "words", "revision": "r1"}
MODEL_REF = {key: REF[key] for key in ("namespace", "name")}


def test_ffuf_unsupported_engine_fails_before_target_launch(tmp_path, monkeypatch):
    calls = []

    async def version(command, *_args, **_kwargs):
        calls.append(command)
        assert command[1:] == ["-V"]
        return ProcessResult(0, stdout=b"ffuf version: 2.1.0\n")

    monkeypatch.setattr(scan, "run_process", version)

    async def scenario():
        async with SocketTarget() as target:
            tool, owner = await scanner(tmp_path, target)
            try:
                result = await tool(target.origin + "/allowed/FUZZ", MODEL_REF)
                assert result["errorCode"] == "scan_proxy_unsupported"
                assert len(calls) == 1 and target.written == b""
                assert not list((tmp_path / "workspace").iterdir())
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


async def scanner(tmp_path, target, *, words=b"one\ntwo\n", spec=None, tls=None):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = spec or socket_spec(target)
    owner = HTTPTransport(spec, resolver=resolver_for(target).resolver, ssl_context=tls)
    owner.configure(authority_deadline=lambda: time.monotonic() + 60)
    store = ArtifactStore()
    store.values[("inputs", "words")] = (words, "text/vnd.contractor.wordlist", "r1")
    factory = scan.ScanToolsetFactory(
        lambda *_: ArtifactClient(spec.allocation_id, store), scanners=[scan.FFUFTool]
    )
    state = WorkerState()
    state.resources.selected_versions.observe(ArtifactRef.model_validate(REF))
    tools = await factory.create_selected(
        selected=["scan_ffuf"],
        allocation_id=spec.allocation_id,
        run_id=spec.run_id,
        namespace="scanner",
        runtime_settings=spec.runtime_settings,
        workspace=AllocationWorkspace(root=tmp_path, path=workspace),
        resources=state.resources,
        controlled_http=owner,
    )
    return tools["scan_ffuf"], owner


@pytest.mark.parametrize("limits", [None, HTTPLimits(requestsPerSecond=100, concurrentRequests=1)])
def test_ffuf_uses_shared_http_owner_and_preserves_filtered_results(tmp_path, monkeypatch, limits):
    commands = []
    native = scan.run_process

    async def run(command, *args, **kwargs):
        commands.append(command)
        return await native(command, *args, **kwargs)

    monkeypatch.setattr(scan, "run_process", run)

    async def respond(_, __, writer, request):
        status = b"404 Not Found" if b"/miss " in request else b"200 OK"
        writer.write(b"HTTP/1.1 " + status + b"\r\nContent-Length: 2\r\n\r\nok")
        await writer.drain()

    async def scenario():
        async with SocketTarget(respond) as target:
            spec = socket_spec(target)
            spec.engagement.http.limits = limits
            tool, owner = await scanner(tmp_path, target, words=b"one\nmiss\none\n", spec=spec)
            try:
                before, _ = await owner.request(target.origin + "/allowed/http")
                assert before.kind == "completed"
                result = await tool(
                    target.origin + "/allowed/FUZZ", MODEL_REF, filter_status="404", rate=999
                )
                assert result["status"] == "completed", result
                assert result["scanComplete"] and result["payloadsAttempted"] == 3
                assert result["wordlistArtifact"] == REF
                assert [r["input"]["FUZZ"] for r in result["results"]] == ["one", "one"]
                assert len(target.requests) == owner._request_count == 4
                expected = (
                    {} if limits is None else limits.model_dump(by_alias=True, exclude_none=True)
                )
                assert result["httpLimits"] == expected
                launch = next(c for c in commands if "-config" in c)
                assert launch[launch.index("-rate") + 1] == ("0" if limits is None else "100")
                assert launch[launch.index("-t") + 1] == "1"
                assert result["stdout"] == result["stderr"] == ""
                projected = project_report(
                    json.dumps(
                        {
                            "schemaVersion": 1,
                            "tool": "scan_ffuf",
                            "inputDigest": "sha256:" + "a" * 64,
                            "inputArtifacts": {"wordlist": REF},
                            "observation": result,
                        }
                    ).encode(),
                    "ffuf-report@1",
                    "2026-11-16T12:00:00Z",
                )
                assert len(projected["observations"]) == 2
                assert all(row["status"] == 200 for row in projected["observations"])
            finally:
                await tool.close()
                await owner.close()
            assert not list((tmp_path / "workspace").iterdir())

    asyncio.run(scenario())


@pytest.mark.parametrize("denied", ["path", "method", "address", "origin", "protected"])
def test_ffuf_scope_refusal_sends_no_target_bytes(tmp_path, denied):
    async def scenario():
        async with SocketTarget() as target:
            spec = socket_spec(target)
            route = spec.engagement.http.routes[0]
            route.path_prefixes = ["/allowed"]
            route.methods = ["GET"]
            url = target.origin + "/allowed/FUZZ"
            if denied == "path":
                url = target.origin + "/forbidden/FUZZ"
            elif denied == "method":
                route.methods = ["POST"]
            elif denied == "address":
                route.address_ranges = ["192.0.2.0/24"]
            elif denied == "origin":
                url = "http://foreign.invalid/allowed/FUZZ"
            else:
                url = "http://127.0.0.1/allowed/FUZZ"
            tool, owner = await scanner(tmp_path, target, spec=spec)
            try:
                result = await tool(url, MODEL_REF)
                assert result["status"] == "failed", result
                assert result["errorCode"] == "scan_target_denied"
                assert target.written == b""
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("trusted", [True, False])
def test_ffuf_https_uses_owner_dns_and_certificate_verification(tmp_path, trusted):
    async def scenario():
        server_tls, client_tls, names = tls_contexts(tmp_path)
        async with SocketTarget(ssl_context=server_tls) as target:
            tool, owner = await scanner(tmp_path, target, tls=client_tls if trusted else None)
            try:
                result = await tool(target.origin + "/allowed/FUZZ", MODEL_REF)
                if trusted:
                    assert result["status"] == "completed", result
                    assert len(target.requests) == 2 and set(names) == {"target.example"}
                else:
                    assert result["status"] == "failed", result
                    assert target.written == b""
                    assert owner._request_count == 2  # Only the owner's pre-send retry.
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["disconnect", "body_limit", "redirect", "http_status"])
def test_ffuf_never_replays_failed_exchange_or_follows_redirect(tmp_path, failure):
    async def respond(_, __, writer, ___):
        if failure == "disconnect":
            writer.transport.abort()
            return
        status = (
            b"302 Found"
            if failure == "redirect"
            else b"503 Unavailable"
            if failure == "http_status"
            else b"200 OK"
        )
        writer.write(
            b"HTTP/1.1 "
            + status
            + b"\r\nLocation: http://forbidden.invalid/\r\nContent-Length: 8\r\n\r\ncontents"
        )
        await writer.drain()

    async def scenario():
        async with SocketTarget(respond) as target:
            spec = socket_spec(target)
            if failure == "body_limit":
                spec.engagement.http.limits = HTTPLimits(responseBodyBytes=2)
            tool, owner = await scanner(tmp_path, target, words=b"one\n", spec=spec)
            try:
                result = await tool(target.origin + "/allowed/FUZZ", MODEL_REF)
                assert len(target.requests) == owner._request_count == 1
                if failure in {"disconnect", "body_limit"}:
                    assert result["status"] == "failed", result
                    assert result["errorCode"] == "scan_request_failed"
                else:
                    assert result["status"] == "completed", result
                    assert result["results"][0]["status"] == (302 if failure == "redirect" else 503)
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


def test_ffuf_private_proxy_and_credentials_keep_identity_cookie_isolation(tmp_path, monkeypatch):
    secret = "private-ffuf-assigned-canary"
    launches = []
    native = asyncio.create_subprocess_exec

    async def spawn(*command, **options):
        if "-config" in command:
            config = Path(command[command.index("-config") + 1])
            proxy = tomllib.loads(config.read_text())["http"]["proxyurl"]
            assert config.stat().st_mode & 0o777 == 0o600
            assert proxy not in repr(command) + repr(options["env"])
            assert secret not in repr(command) + repr(options["env"]) + config.read_text()
            launches.append(command)
        return await native(*command, **options)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    async def respond(_, __, writer, request):
        writer.write(
            b"HTTP/1.1 200 OK\r\nSet-Cookie: learned=private-jar; Path=/\r\n"
            + b"Location: /"
            + secret.encode()
            + b"\r\nContent-Length: 2\r\n\r\nok"
        )
        await writer.drain()

    async def scenario():
        async with SocketTarget(respond) as target:
            spec = identity_spec(
                target,
                [
                    (
                        "account",
                        "api-keys",
                        {
                            "header:X-API-Key": secret,
                            "query:api_key": secret,
                            "cookie:client": secret,
                        },
                        None,
                    )
                ],
            )
            tool, owner = await scanner(tmp_path, target, spec=spec)
            try:
                with pytest.raises(ToolInputError, match="credential"):
                    await tool(target.origin + "/allowed/FUZZ", MODEL_REF, credential_id="foreign")
                assert not launches and target.written == b""
                result = await tool(
                    target.origin + "/allowed/FUZZ", MODEL_REF, credential_id="credential-account"
                )
                assert result["status"] == "completed", result
                assert all(
                    secret.encode() in r and b"Proxy-Authorization" not in r
                    for r in target.requests
                )
                assert b"learned=private-jar" in target.requests[1]
                assert secret not in json.dumps(result)
                anonymous = await tool(target.origin + "/allowed/FUZZ", MODEL_REF)
                assert anonymous["status"] == "completed", anonymous
                assert secret.encode() not in target.requests[2]
                assert b"learned=private-jar" not in target.requests[2]
                assert len(launches) == 2
            finally:
                await tool.close()
                await owner.close()
            assert spec.runtime_settings.http_identities is None

    asyncio.run(scenario())


@pytest.mark.parametrize("finish", ["cancel", "lease_loss", "timeout"])
def test_ffuf_joins_process_target_and_private_files_on_termination(tmp_path, monkeypatch, finish):
    children = []
    native = asyncio.create_subprocess_exec

    async def spawn(*command, **options):
        process = await native(*command, **options)
        if "-config" in command:
            children.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    async def stall(_, reader, __, ___):
        await reader.read()

    async def scenario():
        async with SocketTarget(stall) as target:
            tool, owner = await scanner(tmp_path, target)
            task = asyncio.create_task(
                tool(
                    target.origin + "/allowed/FUZZ",
                    MODEL_REF,
                    timeout_seconds=3 if finish == "timeout" else 30,
                )
            )
            try:
                await asyncio.wait_for(target.request_received.wait(), 15)
                if finish == "cancel":
                    task.cancel()
                elif finish == "lease_loss":
                    await owner.close()
                if finish == "timeout":
                    result = await task
                    assert result["errorCode"] == "scan_timeout", result
                else:
                    with pytest.raises(asyncio.CancelledError):
                        await task
                await asyncio.wait_for(target.all_closed.wait(), 5)
                assert len(target.requests) == 1
                assert children and all(p.returncode is not None for p in children)
                assert not owner._tasks
                assert not list((tmp_path / "workspace").iterdir())
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await tool.close()
                await owner.close()

    asyncio.run(scenario())
