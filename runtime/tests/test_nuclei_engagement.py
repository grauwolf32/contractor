"""Real installed Nuclei uses the production allocation transport and socket peer."""

import asyncio
import base64
import json
import shutil
import time
from itertools import pairwise
from pathlib import Path

import pytest
import yaml
from fakes.engagement_socket import SocketTarget, resolver_for, socket_spec, tls_contexts
from fakes.http_transport import HTTPTransport
from fakes.worker_state import WorkerState
from test_audit_http_credentials import tools_for
from test_engagement_identity import identity_spec
from test_tool_worker import ArtifactStore

from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.contracts.engagement import HTTPLimits
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.toolsets.scan.tools import ScanToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace

pytestmark = pytest.mark.skipif(
    shutil.which("nuclei") is None, reason="optional real Nuclei binary is absent"
)


def template(*, path="{{RootURL}}/allowed", method="GET"):
    return {
        "id": "scoped-http",
        "info": {"name": "Scoped HTTP fixture", "author": "contractor", "severity": "info"},
        "http": [
            {
                "method": method,
                "path": [path],
                "matchers": [{"type": "status", "status": [200]}],
            }
        ],
    }


async def scanner(tmp_path, target, *, spec=None, templates=None, tls=None):
    source, workspace = tmp_path / "templates", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    for index, value in enumerate(templates or [template()]):
        (source / f"fixture-{index}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))
    spec = spec or socket_spec(target)
    owner = HTTPTransport(spec, resolver=resolver_for(target).resolver, ssl_context=tls)
    owner.configure(authority_deadline=lambda: time.monotonic() + 60)
    store = ArtifactStore()
    factory = ScanToolsetFactory(
        lambda *_: ArtifactClient(spec.allocation_id, store),
        templates_directory=source,
    )
    tools = await factory.create_selected(
        selected=["scan_nuclei"],
        allocation_id=spec.allocation_id,
        run_id=spec.run_id,
        namespace="scanner",
        runtime_settings=spec.runtime_settings,
        workspace=AllocationWorkspace(root=tmp_path, path=workspace),
        resources=WorkerState().resources,
        controlled_http=owner,
    )
    return tools["scan_nuclei"], owner, store


def literal_origin(target):
    target.origin = f"http://{target.address}:{target.port}"


def test_real_nuclei_delegates_to_allocation_http(tmp_path):
    async def scenario():
        async with SocketTarget() as target:
            literal_origin(target)
            tool, owner, store = await scanner(tmp_path, target)
            try:
                result = await tool(target.origin + "/allowed", timeout_seconds=30)
                assert result["status"] == "completed", result
                assert [entry["template-id"] for entry in result["results"]] == ["scoped-http"]
                assert len(target.requests) == 1
                assert target.requests[0].startswith(b"GET /allowed HTTP/1.1")
                assert owner._request_count == 1
                assert result["httpLimits"] == {}
                assert result["templateManifest"]["revision"]
                assert ("scanner", result["templateManifest"]["name"]) in store.values
            finally:
                await tool.close()
                await owner.close()
        assert not list((tmp_path / "workspace").iterdir())

    asyncio.run(scenario())


def test_real_nuclei_keeps_published_finding_before_later_transport_failure(tmp_path, monkeypatch):
    async def scenario():
        matched = asyncio.Event()
        original_read = asyncio.StreamReader.read

        async def read(self, n=-1):
            data = await original_read(self, n)
            if b'"template-id":"scoped-http"' in data:
                matched.set()
            return data

        monkeypatch.setattr(asyncio.StreamReader, "read", read)

        async def respond(_, __, writer, request):
            if b"/allowed/fail " in request:
                await asyncio.wait_for(matched.wait(), 20)
                writer.close()
            else:
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                await writer.drain()

        async with SocketTarget(respond) as target:
            literal_origin(target)
            bad = template(path="{{RootURL}}/allowed/fail")
            bad["id"] = "later-failure"
            finding = template()
            finding["info"]["severity"] = "medium"
            tool, owner, store = await scanner(tmp_path, target, templates=[finding, bad])
            try:
                result = await tool(target.origin + "/allowed", timeout_seconds=30)
                assert matched.is_set(), result
                assert result["status"] == "failed", result
                assert result["errorCode"] == "scan_request_failed"
                assert [row["template-id"] for row in result["results"]] == ["scoped-http"]
                published = await tool.publish_outputs(
                    result,
                    {
                        name: ArtifactRef(namespace="scanner", name=name)
                        for name in ("candidates", "evidence")
                    },
                )
                batch = json.loads(store.values[("scanner", "candidates")][0])
                assert [row["rule"] for row in batch["candidates"]] == ["scoped-http"]
                evidence = store.values[("scanner", "evidence")][0]
                assert b"scoped-http" in evidence
                assert published["status"] == "failed"
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


def test_real_nuclei_custom_artifact_rules_use_the_same_http_boundary(tmp_path):
    async def scenario():
        async with SocketTarget() as target:
            literal_origin(target)
            tool, owner, store = await scanner(tmp_path, target)
            data = yaml.safe_dump(template(), sort_keys=False).encode()
            store.values[("inputs", "rules")] = (data, "application/yaml", "r1")
            try:
                result = await tool.contractor_trusted_call(
                    target.origin + "/allowed",
                    rulepack_ref={"namespace": "inputs", "name": "rules", "revision": "r1"},
                    timeout_seconds=30,
                )
                assert result["status"] == "completed", result
                assert result["source"] == "rulepack"
                assert result["sourceContentDigest"].startswith("sha256:")
                assert len(target.requests) == owner._request_count == 1
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


def test_real_nuclei_refuses_foreign_credential_before_launch(tmp_path):
    async def scenario():
        async with SocketTarget() as target:
            literal_origin(target)
            tool, owner, _ = await scanner(tmp_path, target)
            try:
                with pytest.raises(ToolInputError, match="credential"):
                    await tool(target.origin + "/allowed", credential_id="unassigned")
                assert target.written == b""
                assert owner._request_count == 0
                assert not list((tmp_path / "workspace").iterdir())
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("trusted", [True, False])
def test_real_nuclei_https_uses_owner_tls_verification(tmp_path, trusted):
    async def scenario():
        server_tls, client_tls, names = tls_contexts(tmp_path)
        async with SocketTarget(ssl_context=server_tls) as target:
            tool, owner, _ = await scanner(tmp_path, target, tls=client_tls if trusted else None)
            try:
                result = await tool(target.origin + "/allowed", timeout_seconds=30)
                if trusted:
                    assert result["status"] == "completed", result
                    assert len(target.requests) == 1 and names == ["target.example"]
                else:
                    assert result["status"] == "failed"
                    assert result["errorCode"] == "scan_request_failed"
                    assert target.written == b""
                    assert owner._request_count == 2
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("denied", ["path", "method", "address", "origin", "exclusion"])
def test_real_nuclei_scope_refuses_target_bytes(tmp_path, denied):
    async def scenario():
        async with SocketTarget() as target, SocketTarget() as foreign:
            literal_origin(target)
            literal_origin(foreign)
            spec = socket_spec(target)
            route = spec.engagement.http.routes[0]
            route.path_prefixes = ["/allowed"]
            route.methods = ["GET"]
            rule = template()
            if denied == "path":
                rule["http"][0]["path"] = ["{{RootURL}}/forbidden"]
            elif denied == "method":
                rule["http"][0]["method"] = "POST"
            elif denied == "address":
                route.address_ranges = ["203.0.113.9/32"]
            elif denied == "origin":
                rule["http"][0]["path"] = [foreign.origin + "/allowed"]
            else:
                spec.engagement.http.exclusions.append(route.model_copy(deep=True))
            tool, owner, _ = await scanner(tmp_path, target, spec=spec, templates=[rule])
            try:
                result = await tool(target.origin + "/allowed", timeout_seconds=30)
                assert result["status"] == "failed", result
                assert result["errorCode"] in {"scan_target_denied", "scan_request_failed"}
                assert target.written == foreign.written == b""
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


def test_real_nuclei_and_http_tools_share_transport_without_total_quota(tmp_path):
    async def scenario():
        async with SocketTarget() as target:
            literal_origin(target)
            spec = socket_spec(target)
            spec.engagement.http.limits = HTTPLimits(requestsPerSecond=1000)
            tool, owner, _ = await scanner(tmp_path, target, spec=spec)
            tools, _ = await tools_for(owner)
            try:
                await tools["http_request"](target.origin + "/allowed")
                assert len(target.requests) == 1
                result = await tool(target.origin + "/allowed", timeout_seconds=30)
                assert result["status"] == "completed", result
                assert len(target.requests) == owner._request_count == 2
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


def test_real_nuclei_explicit_calls_keep_distinct_manifests_and_one_owner(tmp_path):
    async def scenario():
        async with SocketTarget() as target:
            literal_origin(target)
            tool, owner, store = await scanner(tmp_path, target)
            try:
                first = await tool(target.origin + "/allowed", timeout_seconds=30)
                second = await tool(target.origin + "/allowed", timeout_seconds=30)
                assert first["status"] == second["status"] == "completed"
                assert first["templateManifest"] != second["templateManifest"]
                assert len(store.values) == 2
                assert owner._request_count == len(target.requests) == 2
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [401, 429, 503])
def test_real_nuclei_http_statuses_never_authorize_hidden_retry(tmp_path, status):
    async def respond(_, __, writer, ___):
        writer.write(f"HTTP/1.1 {status} Result\r\nContent-Length: 0\r\n\r\n".encode())
        await writer.drain()

    async def scenario():
        async with SocketTarget(respond) as target:
            literal_origin(target)
            tool, owner, _ = await scanner(tmp_path, target)
            try:
                result = await tool(target.origin + "/allowed", timeout_seconds=30)
                assert result["status"] == "completed", result
                assert len(target.requests) == owner._request_count == 1
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("bound", ["request", "response"])
def test_real_nuclei_body_limits_cannot_be_bypassed(tmp_path, bound):
    async def respond(_, __, writer, ___):
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\ncontents")
        await writer.drain()

    async def scenario():
        async with SocketTarget(respond) as target:
            literal_origin(target)
            spec = socket_spec(target)
            rule = template(method="POST" if bound == "request" else "GET")
            if bound == "request":
                spec.engagement.http.limits = HTTPLimits(requestBodyBytes=2)
                rule["http"][0]["body"] = "larger-than-two"
            else:
                spec.engagement.http.limits = HTTPLimits(responseBodyBytes=2)
            tool, owner, _ = await scanner(tmp_path, target, spec=spec, templates=[rule])
            try:
                result = await tool(target.origin + "/allowed", timeout_seconds=30)
                assert result["status"] == "failed", result
                assert result["errorCode"] == "scan_request_failed"
                assert len(target.requests) == (0 if bound == "request" else 1)
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("method,attempts", [("GET", 2), ("POST", 1)])
def test_real_nuclei_leaves_retry_to_http_owner(tmp_path, method, attempts, monkeypatch):
    async def scenario():
        async with SocketTarget() as target:
            literal_origin(target)
            tool, owner, _ = await scanner(tmp_path, target, templates=[template(method=method)])
            calls = 0
            original = asyncio.open_connection

            async def refuse(host, port, **options):
                nonlocal calls
                if host == target.address and port == target.port:
                    calls += 1
                    raise ConnectionRefusedError("fixture pre-send connection failure")
                return await original(host, port, **options)

            monkeypatch.setattr(asyncio, "open_connection", refuse)
            try:
                result = await tool(target.origin + "/allowed", timeout_seconds=30)
                assert result["status"] == "failed", result
                assert owner._request_count == calls == attempts
                assert target.written == b""
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


def test_real_nuclei_never_repeats_an_inflight_write(tmp_path):
    async def drop(_, __, writer, ___):
        writer.transport.abort()

    async def scenario():
        async with SocketTarget(drop) as target:
            literal_origin(target)
            rule = template(method="POST")
            rule["http"][0]["body"] = "write-once"
            tool, owner, _ = await scanner(tmp_path, target, templates=[rule])
            try:
                result = await tool(target.origin + "/allowed", timeout_seconds=30)
                assert result["status"] == "failed", result
                assert len(target.requests) == owner._request_count == 1
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "feature", ["raw", "unsafe", "pipeline", "race", "flow", "dns", "helper", "payload_file"]
)
def test_real_nuclei_unsupported_features_fail_before_any_target_request(tmp_path, feature):
    async def scenario():
        async with SocketTarget() as target:
            literal_origin(target)
            rule = template()
            if feature == "raw":
                rule["http"][0]["raw"] = ["GET /allowed HTTP/1.1\r\nHost: {{Hostname}}\r\n"]
            elif feature in {"unsafe", "pipeline", "race"}:
                rule["http"][0][feature] = True
            elif feature == "flow":
                rule["flow"] = "http(1)"
            elif feature == "dns":
                rule["dns"] = [{"name": "never-resolve.invalid", "type": "A"}]
            elif feature == "helper":
                rule["variables"] = {"derived": "{{resolve('never-resolve.invalid', 'a')}}"}
            else:
                rule["http"][0]["payloads"] = {"word": "/not-an-inline-payload.txt"}
            tool, owner, _ = await scanner(tmp_path, target, templates=[rule])
            try:
                result = await tool(target.origin + "/allowed", timeout_seconds=30)
                assert result["status"] == "failed", result
                assert result["errorCode"] == "nuclei_selection_invalid"
                assert target.written == b"" and owner._request_count == 0
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["basic", "bearer", "cookies", "api-keys"])
def test_real_nuclei_uses_private_credentials_and_runtime_cookie_state(tmp_path, monkeypatch, kind):
    secret = "private-assigned-nuclei-credential-canary"
    observed = []
    child_commands = []
    original = asyncio.create_subprocess_exec

    async def spawn(*command, **options):
        if "-l" in command:
            config = Path(command[command.index("-config") + 1])
            assert config.stat().st_mode & 0o777 == 0o600
            assert secret not in repr(command) + repr(options["env"])
            assert "proxy-internal: true" in config.read_text()
            child_commands.append(command)
        return await original(*command, **options)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    async def reflect(_, __, writer, raw):
        observed.append(raw)
        body = secret.encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\nSet-Cookie: learned=runtime-jar; Path=/\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        await writer.drain()

    async def scenario():
        async with SocketTarget(reflect) as target:
            literal_origin(target)
            fields = (
                {"username": "scanner-user", "password": secret}
                if kind == "basic"
                else {"token": secret}
                if kind == "bearer"
                else {"session": secret}
                if kind == "cookies"
                else {"header:X-API-Key": secret, "query:api_key": secret, "cookie:client": secret}
            )
            spec = identity_spec(target, [("account", kind, fields, None)])
            rule = template()
            rule["http"][0]["path"] = ["{{RootURL}}/allowed/one", "{{RootURL}}/allowed/two"]
            rule["http"][0]["extractors"] = [{"type": "regex", "part": "body", "regex": [".+"]}]
            tool, owner, store = await scanner(tmp_path, target, spec=spec, templates=[rule])
            try:
                result = await tool(
                    target.origin + "/allowed",
                    credential_id="credential-account",
                    timeout_seconds=30,
                )
                assert result["status"] == "completed", result
                assert len(observed) == 2
                assert len(child_commands) == 1
                assert child_commands[0][child_commands[0].index("-retries") + 1] == "0"
                assert child_commands[0][child_commands[0].index("-rate-limit") + 1] == "0"
                expected = (
                    b"Authorization: Basic " + base64.b64encode(("scanner-user:" + secret).encode())
                    if kind == "basic"
                    else b"Authorization: Bearer " + secret.encode()
                    if kind == "bearer"
                    else b"Cookie: session=" + secret.encode()
                    if kind == "cookies"
                    else b"X-API-Key: " + secret.encode()
                )
                assert all(expected.lower() in raw.lower() for raw in observed)
                if kind == "api-keys":
                    assert all(b"?api_key=" + secret.encode() in raw for raw in observed)
                    assert all(b"client=" + secret.encode() in raw for raw in observed)
                assert b"learned=runtime-jar" in observed[1]
                assert secret not in json.dumps(result)
                assert all(secret.encode() not in value[0] for value in store.values.values())
            finally:
                await tool.close()
                await owner.close()
            assert spec.runtime_settings.http_identities is None

    asyncio.run(scenario())


@pytest.mark.parametrize("finish", ["cancel", "lease_loss", "timeout"])
def test_real_nuclei_process_and_socket_join_on_owner_termination(tmp_path, monkeypatch, finish):
    children = []
    original = asyncio.create_subprocess_exec

    async def spawn(*command, **options):
        process = await original(*command, **options)
        if "-l" in command:
            children.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    async def stall(_, reader, __, ___):
        await reader.read()

    async def scenario():
        async with SocketTarget(stall) as target:
            literal_origin(target)
            tool, owner, _ = await scanner(tmp_path, target)
            task = asyncio.create_task(
                tool(target.origin + "/allowed", timeout_seconds=5 if finish == "timeout" else 30)
            )
            try:
                await asyncio.wait_for(target.request_received.wait(), 15)
                if finish == "cancel":
                    task.cancel()
                elif finish == "lease_loss":
                    await owner.close()
                if finish == "timeout":
                    result = await task
                    assert result["errorCode"] == "scan_timeout"
                else:
                    with pytest.raises(asyncio.CancelledError):
                        await task
                await asyncio.wait_for(target.all_closed.wait(), 5)
                assert len(target.requests) == 1
                assert children and all(child.returncode is not None for child in children)
                assert not owner._tasks
                assert not list((tmp_path / "workspace").iterdir())
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["bridge", "lease"])
def test_real_nuclei_preserves_fatal_transport_signals(tmp_path, monkeypatch, failure):
    from contractor_runtime.core.tool_errors import ToolExecutionError

    async def scenario():
        async with SocketTarget() as target:
            literal_origin(target)
            tool, owner, _ = await scanner(tmp_path, target)
            try:
                if failure == "lease":
                    owner.configure(authority_deadline=lambda: 0.0)
                else:
                    await owner._initialize()
                    original = owner._engine.handle_hook

                    async def broken(pool, hook):
                        if hook.name == "requestheaders":
                            raise RuntimeError("private-scanner-bridge-canary")
                        return await original(pool, hook)

                    monkeypatch.setattr(owner._engine, "handle_hook", broken)
                with pytest.raises(ToolExecutionError) as raised:
                    await tool(target.origin + "/allowed", timeout_seconds=30)
                assert raised.value.code == (
                    "http_allocation_closed" if failure == "lease" else "http_bridge_failed"
                )
                assert not owner._tasks
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


def test_real_nuclei_proxy_listener_startup_failure_is_fatal(tmp_path, monkeypatch):
    from contractor_runtime.core.tool_errors import ToolExecutionError

    async def scenario():
        async with SocketTarget() as target:
            literal_origin(target)
            tool, owner, _ = await scanner(tmp_path, target)
            try:
                await owner._initialize()

                async def fail_bind(*_args, **_kwargs):
                    raise OSError(24, "private-listener-startup-canary")

                monkeypatch.setattr(asyncio, "start_server", fail_bind)
                with pytest.raises(ToolExecutionError) as raised:
                    await tool(target.origin + "/allowed", timeout_seconds=30)
                assert raised.value.code == "http_bridge_failed"
                assert "private-listener-startup-canary" not in str(raised.value)
                assert not owner._tasks
                assert not owner._engine.ingresses
                assert not target.requests
                assert not list((tmp_path / "workspace").iterdir())
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())


def test_native_nuclei_paces_requests_from_engagement(tmp_path, monkeypatch):
    from contractor_runtime.toolsets.scan import tools as scan

    native = scan.run_process
    commands, arrivals = [], []

    async def run(command, directory, timeout, **kwargs):
        commands.append(command)
        return await native(command, directory, timeout, **kwargs)

    monkeypatch.setattr(scan, "run_process", run)

    async def respond(_, __, writer, ___):
        arrivals.append(time.monotonic())
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        await writer.drain()

    async def scenario():
        async with SocketTarget(respond) as target:
            literal_origin(target)
            spec = socket_spec(target)
            spec.engagement.http.limits = HTTPLimits(requestsPerSecond=2, concurrentRequests=1)
            rule = template()
            rule["http"][0]["path"] = [f"{{{{RootURL}}}}/allowed/{i}" for i in range(3)]
            tool, owner, _ = await scanner(tmp_path, target, spec=spec, templates=[rule])
            try:
                result = await tool(target.origin + "/allowed", rate_limit=999, timeout_seconds=30)
                assert result["status"] == "completed", result
                assert len(arrivals) == 3
                assert all(b - a >= 0.45 for a, b in pairwise(arrivals))
                launch = next(command for command in commands if "-config" in command)
                assert launch[launch.index("-rate-limit") + 1] == "2"
                assert launch[launch.index("-concurrency") + 1] == "1"
                assert launch[launch.index("-retries") + 1] == "0"
                assert result["httpLimits"] == {"requestsPerSecond": 2, "concurrentRequests": 1}
            finally:
                await tool.close()
                await owner.close()

    asyncio.run(scenario())
