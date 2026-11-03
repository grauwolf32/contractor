"""Native protocol and pool behavior observed on actual target sockets."""

import asyncio
import time

import pytest
from fakes.engagement_socket import SocketTarget, preflight, resolver_for, tls_contexts
from fakes.native_http import NativeTarget


@pytest.mark.parametrize("http2", [False, True], ids=["http11", "http2"])
def test_native_pool_reuses_verified_connection_and_rechecks_dns(tmp_path, http2, monkeypatch):
    from mitmproxy.net import tls

    writes = []
    monkeypatch.setattr(tls, "log_master_secret", lambda *_: writes.append(True))

    async def scenario():
        server, trust, _ = tls_contexts(tmp_path)
        server.set_alpn_protocols(["h2"] if http2 else ["http/1.1"])
        async with NativeTarget(ssl_context=server) as target:
            answers = [target.address]
            calls = []
            config = resolver_for(target, lambda: answers, calls=calls)
            async with preflight(target, config=config, ssl_context=trust) as (owner, _):
                owner.configure(authority_deadline=lambda: time.monotonic() + 60)
                for _ in range(3):
                    outcome, response = await owner.request(target.origin + "/allowed")
                    assert outcome.kind == "completed" and response.body == b"ok"
                assert len(target.connections) == 1
                assert len(target.trace) == 3
                assert {row[2] for row in target.trace} == {b"HTTP/2" if http2 else b"HTTP/1.1"}
                answers.append("127.0.0.1")
                outcome, _ = await owner.request(target.origin + "/allowed")
                assert outcome.kind == "denied"
                assert len(target.trace) == 3
                assert calls.count(("target.example", target.port)) == 4
                assert owner._engine.flows == {}
                # Native dial tasks live for the connection's lifetime. They
                # may keep the first exchange's small authority marker, never
                # its response/future or secret-bearing flow.
                for pool in owner._engine.pools.values():
                    for old in pool.connecting.values():
                        assert old.future is old.response is old.flow is old.request is None
                assert all(
                    not p.exchanges and not p.layer.streams for p in owner._engine.pools.values()
                )
            await target.all_closed.wait()
            assert not writes

    asyncio.run(scenario())


def test_native_http2_multiplexes_same_identity_and_resets_only_cancelled_stream(tmp_path):
    async def scenario():
        server, trust, _ = tls_contexts(tmp_path)
        server.set_alpn_protocols(["h2"])
        async with (
            NativeTarget(ssl_context=server) as target,
            preflight(target, ssl_context=trust) as (owner, _),
        ):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            assert (await owner.request(target.origin + "/allowed"))[0].kind == "completed"
            held = asyncio.create_task(owner.request(target.origin + "/hold"))
            await target.held.wait()
            # The first stream cannot complete until released. The second
            # must complete on the same connection while it is outstanding.
            outcome, response = await owner.request(target.origin + "/allowed")
            assert outcome.kind == "completed" and response.body == b"ok"
            assert not held.done() and len(target.connections) == 1
            held.cancel()
            with pytest.raises(asyncio.CancelledError):
                await held
            assert (await owner.request(target.origin + "/allowed"))[0].kind == "completed"
            assert len(target.connections) == 1
            assert len(target.resets) == 1
            assert owner._request_count == 4

    async def bounded():
        async with asyncio.timeout(15):
            await scenario()

    asyncio.run(bounded())


def test_native_rejects_http10_response():
    async def respond(_, __, writer, ___):
        writer.write(b"HTTP/1.0 200 OK\r\nContent-Length: 2\r\n\r\nok")
        await writer.drain()

    async def scenario():
        async with SocketTarget(respond) as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            outcome, response = await owner.request(target.origin + "/allowed")
            assert outcome.kind == "failed" and outcome.reason == "response_invalid"
            assert response is None and owner._request_count == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["tcp", "tls"])
def test_native_expiry_after_connection_setup_prevents_http_bytes(tmp_path, monkeypatch, phase):
    from contractor_runtime.adapters.http_engine import _Pool

    async def scenario():
        now = time.monotonic()
        server, trust, _ = tls_contexts(tmp_path)
        original_connect = asyncio.open_connection
        original_hook = _Pool.handle_hook

        async with SocketTarget(ssl_context=server if phase == "tls" else None) as target:

            async def connect(host, port, **options):
                nonlocal now
                result = await original_connect(host, port, **options)
                if phase == "tcp" and port == target.port:
                    now += 61
                return result

            async def hook(pool, event):
                nonlocal now
                if phase == "tls" and event.name == "tls_established_server":
                    now += 61
                await original_hook(pool, event)

            monkeypatch.setattr(asyncio, "open_connection", connect)
            monkeypatch.setattr(_Pool, "handle_hook", hook)
            async with preflight(target, ssl_context=trust, monotonic=lambda: now) as (owner, _):
                owner.configure(authority_deadline=lambda: now + 60)
                outcome, response = await owner.request(target.origin + "/allowed")
                assert outcome.kind == "failed" and response is None
                assert target.written == b"" and owner._request_count == 1

    async def bounded():
        async with asyncio.timeout(15):
            await scenario()

    asyncio.run(bounded())


def test_native_http2_cancel_removes_a_request_queued_by_remote_settings(tmp_path, monkeypatch):
    from mitmproxy.proxy.layers.http import RequestHeaders

    from contractor_runtime.adapters.http_engine_protocol import BoundedHttp2Client

    async def scenario():
        server, trust, _ = tls_contexts(tmp_path)
        server.set_alpn_protocols(["h2"])
        queued = asyncio.Event()
        original = BoundedHttp2Client._handle_event

        def observe(parser, event):
            stream_id = getattr(event, "stream_id", None)
            yield from original(parser, event)
            if isinstance(event, RequestHeaders) and stream_id in parser.stream_queue:
                queued.set()

        monkeypatch.setattr(BoundedHttp2Client, "_handle_event", observe)
        async with (
            NativeTarget(ssl_context=server, max_streams=1) as target,
            preflight(target, ssl_context=trust) as (owner, _),
        ):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            await owner.request(target.origin + "/allowed")
            held = asyncio.create_task(owner.request(target.origin + "/hold"))
            await target.held.wait()
            pending = asyncio.create_task(owner.request(target.origin + "/never"))
            await queued.wait()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            target.release.set()
            assert (await held)[0].kind == "completed"
            assert (await owner.request(target.origin + "/allowed"))[0].kind == "completed"
            assert b"/never" not in [row[1] for row in target.trace]
            assert len(target.connections) == 1
            parser = next(iter(next(iter(owner._engine.pools.values())).parsers.values()))
            assert not parser.stream_queue and not parser.our_stream_id

    async def bounded():
        async with asyncio.timeout(15):
            await scenario()

    asyncio.run(bounded())


def test_native_startup_failure_restores_context_and_allows_clean_next_allocation(monkeypatch):
    from pathlib import Path

    from mitmproxy import ctx
    from mitmproxy.addons.tlsconfig import TlsConfig

    from contractor_runtime.adapters import http_engine

    original = TlsConfig.configure
    previous = {key: getattr(ctx, key, None) for key in ("master", "options", "log")}
    directories = []

    def refuse(tls, updated):
        if "confdir" in updated:
            directories.append(Path(ctx.options.confdir))
            raise RuntimeError("injected native startup failure")
        return original(tls, updated)

    async def scenario():
        async with SocketTarget() as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            monkeypatch.setattr(TlsConfig, "configure", refuse)
            with pytest.raises(RuntimeError, match="injected native startup failure"):
                await owner.request(target.origin + "/allowed")
            assert http_engine._active is None and owner._engine is None
            assert all(not directory.exists() for directory in directories)
            assert all(getattr(ctx, key, None) is value for key, value in previous.items())
            assert target.written == b""
            monkeypatch.setattr(TlsConfig, "configure", original)
            assert (await owner.request(target.origin + "/allowed"))[0].kind == "completed"
        assert http_engine._active is None

    asyncio.run(scenario())


@pytest.mark.parametrize("http2", [False, True], ids=["http11", "http2"])
def test_native_pools_isolate_actual_credentials_and_exact_origins(tmp_path, http2):
    from test_engagement_identity import identity_spec

    async def scenario():
        server, trust, _ = tls_contexts(tmp_path)
        server.set_alpn_protocols(["h2"] if http2 else ["http/1.1"])
        async with (
            NativeTarget(ssl_context=server) as first,
            NativeTarget(ssl_context=server) as second,
        ):
            spec = identity_spec(
                first,
                [
                    ("a", "bearer", {"token": "private-a"}, None),
                    ("b", "bearer", {"token": "private-b"}, None),
                ],
            )
            spec.engagement.http.routes.append(
                spec.engagement.http.routes[0].model_copy(update={"origin": second.origin})
            )
            for identity in spec.engagement.http.identities:
                identity.origins.append(second.origin)
            async with preflight(first, spec=spec, ssl_context=trust) as (owner, _):
                owner.configure(authority_deadline=lambda: time.monotonic() + 60)
                for target in (first, second):
                    for credential in (None, "credential-a", "credential-b"):
                        for _ in range(2):
                            assert (
                                await owner.request(
                                    target.origin + "/allowed", credential_id=credential
                                )
                            )[0].kind == "completed"
                    assert len(target.connections) == 3
                    auth = [
                        (
                            row[3].get(b"authorization")
                            if http2
                            else next(
                                (
                                    line.split(b":", 1)[1].strip()
                                    for line in row[3].split(b"\r\n")
                                    if line.lower().startswith(b"authorization:")
                                ),
                                None,
                            )
                        )
                        for row in target.trace
                    ]
                    assert auth == [
                        None,
                        None,
                        b"Bearer private-a",
                        b"Bearer private-a",
                        b"Bearer private-b",
                        b"Bearer private-b",
                    ]
                assert len(owner._engine.pools) == 6
                assert owner._request_count == 12

    asyncio.run(scenario())


@pytest.mark.parametrize("case", ["body", "headers"])
def test_native_http2_applies_common_body_and_header_bounds(tmp_path, case):
    from contractor_runtime.contracts.engagement import HTTPLimits

    async def scenario():
        server, trust, _ = tls_contexts(tmp_path)
        server.set_alpn_protocols(["h2"])
        headers = [(b"x-bound", b"v")] * 64 if case == "headers" else []
        async with (
            NativeTarget(ssl_context=server, payload=b"bounded-body", headers=headers) as target,
            preflight(target, ssl_context=trust, limits=HTTPLimits(responseBodyBytes=3)) as (
                owner,
                _,
            ),
        ):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            outcome, response = await owner.request(target.origin + "/allowed")
            if case == "body":
                assert (
                    outcome.kind == "completed" and response.body == b"bou" and response.truncated
                )
            else:
                assert outcome.kind == "failed" and response is None
            assert owner._request_count == 1

    asyncio.run(scenario())


def test_native_callback_bug_fails_closed_without_logging_content(monkeypatch, caplog):
    from contractor_runtime.adapters.http_engine import HTTPEngine

    original = HTTPEngine.handle_hook

    async def fail(engine, pool, event):
        if event.name == "requestheaders":
            raise RuntimeError("private-callback-canary")
        return await original(engine, pool, event)

    async def scenario():
        monkeypatch.setattr(HTTPEngine, "handle_hook", fail)
        async with SocketTarget() as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            with pytest.raises(RuntimeError, match="native HTTP engine callback failed") as caught:
                await owner.request(target.origin + "/allowed")
            assert "private-callback-canary" not in str(caught.value)
            assert target.written == b""
        assert "private-callback-canary" not in caplog.text

    asyncio.run(scenario())


@pytest.mark.parametrize("http2", [False, True], ids=["http11", "http2"])
def test_native_connect_waiters_progress_when_first_responses_become_idle(tmp_path, http2):
    async def scenario():
        server, trust, _ = tls_contexts(tmp_path)
        server.set_alpn_protocols(["h2"] if http2 else ["http/1.1"])
        async with (
            NativeTarget(ssl_context=server) as target,
            preflight(target, ssl_context=trust) as (owner, _),
        ):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            responses = await asyncio.gather(
                *(owner.request(target.origin + "/allowed") for _ in range(8))
            )
            assert all(
                outcome.kind == "completed" and response.body == b"ok"
                for outcome, response in responses
            )
            assert len(target.trace) == owner._request_count == 8

    async def bounded():
        async with asyncio.timeout(15):
            await scenario()

    asyncio.run(bounded())
