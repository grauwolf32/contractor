"""Optional transport checks against an installed sqlmap and loopback targets only."""

from __future__ import annotations

import asyncio
import json
import shutil
import ssl
import subprocess
from contextlib import suppress
from types import SimpleNamespace

import pytest

from contractor_runtime.allocation import WorkerState
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.toolsets.scan.tools import ScanToolsetFactory, SQLMapTool
from contractor_runtime.workspace import AllocationWorkspace


@pytest.mark.skipif(shutil.which("sqlmap") is None, reason="optional sqlmap binary is absent")
@pytest.mark.parametrize("scheme,method", [("http", "POST"), ("https", "POST"), ("https", "PATCH")])
def test_installed_sqlmap_preserves_prepared_request_on_wire(tmp_path, scheme, method):
    tls = None
    if scheme == "https":
        if shutil.which("openssl") is None:
            pytest.skip("openssl is needed for the local TLS fixture")
        certificate, key = tmp_path / "cert.pem", tmp_path / "key.pem"
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-keyout",
                str(key),
                "-out",
                str(certificate),
                "-days",
                "1",
                "-subj",
                "/CN=localhost",
            ],
            check=True,
            timeout=10,
            capture_output=True,
        )
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(certificate, key)

    async def scenario():
        captured = []
        handlers = set()
        probe_seen = asyncio.Event()
        body = '{\n  "id":7, "token":"body-canary-ключ"\n}'

        async def handle(reader, writer):
            handlers.add(asyncio.current_task())
            try:
                head = await reader.readuntil(b"\r\n\r\n")
                first, *lines = head.decode("ascii").split("\r\n")
                headers = dict(
                    (name.lower(), value.strip())
                    for line in lines
                    if line
                    for name, _, value in [line.partition(":")]
                )
                data = await reader.readexactly(int(headers.get("content-length", "0")))
                captured.append(
                    (first, headers, data, writer.get_extra_info("ssl_object") is not None)
                )
                # Wait until sqlmap actually exercises the selected JSON parameter.
                try:
                    parsed = json.loads(data)
                except ValueError:
                    parsed = {}
                if parsed.get("id") != 7 and parsed.get("token") == "body-canary-ключ":
                    probe_seen.set()
                response = b"fixed local fixture response"
                writer.write(
                    b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: text/plain\r\n"
                    + f"Content-Length: {len(response)}\r\n\r\n".encode()
                    + response
                )
                await writer.drain()
            except (ConnectionError, asyncio.IncompleteReadError):
                pass
            finally:
                writer.close()
                with suppress(ConnectionError):
                    await writer.wait_closed()
                handlers.discard(asyncio.current_task())

        async with await asyncio.start_server(handle, "127.0.0.1", 0, ssl=tls) as server:
            port = server.sockets[0].getsockname()[1]
            document = {
                "schemaVersion": 1,
                "method": method,
                "url": f"{scheme}://127.0.0.1:{port}/items%2Fsearch?keep=unchanged",
                "headers": [
                    {"name": "Authorization", "value": "Bearer header-canary"},
                    {"name": "Cookie", "value": "session=cookie-canary"},
                    {"name": "Content-Type", "value": "application/json; charset=utf-8"},
                    {"name": "X-Fixture", "value": "kept"},
                ],
                "body": body,
                "testParameters": ["id"],
            }
            ref = {"namespace": "inputs", "name": "request", "revision": "request-1"}

            async def read_artifact(exact_ref, *, max_bytes):
                assert exact_ref.model_dump(by_alias=True) == ref
                data = json.dumps(document).encode()
                assert len(data) <= max_bytes
                return SimpleNamespace(media_type="application/json", data=data)

            workspace = tmp_path / "workspace"
            workspace.mkdir()
            state = WorkerState()
            factory = ScanToolsetFactory(
                lambda *_: SimpleNamespace(read_artifact=read_artifact), scanners=[SQLMapTool]
            )
            tools = await factory.create_selected(
                selected=["scan_sqlmap"],
                allocation_id="allocation",
                run_id="run",
                namespace="scanner",
                runtime_settings=RuntimeSettings(
                    artifactApiUrl="https://artifacts.invalid", requestTimeoutSeconds=30
                ),
                state=state,
                workspace=AllocationWorkspace(root=tmp_path, path=workspace),
            )
            tool = tools["scan_sqlmap"]
            invocation = asyncio.create_task(tool(request_ref=ref, timeout_seconds=20))
            seen = asyncio.create_task(probe_seen.wait())
            try:
                done, _ = await asyncio.wait(
                    [invocation, seen], timeout=20, return_when=asyncio.FIRST_COMPLETED
                )
                assert seen in done, (
                    await invocation
                    if invocation.done()
                    else "no selected-parameter probe observed"
                )
            finally:
                seen.cancel()
                await tool.close()
                await asyncio.gather(invocation, seen, return_exceptions=True)
                if handlers:
                    await asyncio.gather(*handlers)

        assert captured
        first, headers, data, encrypted = captured[0]
        assert first == f"{method} /items%2Fsearch?keep=unchanged HTTP/1.1"
        assert headers["host"] == f"127.0.0.1:{port}"
        assert headers["authorization"] == "Bearer header-canary"
        assert headers["cookie"] == "session=cookie-canary"
        assert headers["content-type"] == "application/json; charset=utf-8"
        assert headers["x-fixture"] == "kept"
        assert data == body.encode()
        assert encrypted == (scheme == "https")
        # The scanner may mutate id; unrelated query/header/body values remain pinned.
        for first, headers, _data, encrypted in captured:
            assert first == f"{method} /items%2Fsearch?keep=unchanged HTTP/1.1"
            assert headers["authorization"] == "Bearer header-canary"
            assert headers["cookie"] == "session=cookie-canary"
            assert encrypted == (scheme == "https")
        assert not list(workspace.iterdir())
        assert "canary" not in repr(state.metrics)

    asyncio.run(scenario())
