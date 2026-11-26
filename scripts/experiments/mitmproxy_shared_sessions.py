"""Exercise production shared HTTP sessions with real socket and TLS fixtures.

The original extraction experiment and its evidence remain at e40825690.
This matrix now uses the ordinary Runtime owner and HTTPTransport initialization.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import importlib.metadata
import json
import platform
import ssl
import sys
import tempfile
import threading
import time
import weakref
from contextlib import AsyncExitStack, suppress
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "runtime/src"), str(ROOT / "runtime/tests")]

import httpx
from contractor_runtime.adapters.bridge import HTTPProxyBridge
from contractor_runtime.adapters.bridge.mitmproxy.runtime import (
    MitmproxyEngine,
)
from contractor_runtime.adapters.http_session import PrivateResponse
from contractor_runtime.adapters.http_transport import HTTPTransport
from contractor_runtime.contracts import (
    AllocationSpec,
    SharedConfig,
    decode_private,
)
from contractor_runtime.shared_config import install_shared_config
from fakes.engagement_socket import (
    SocketTarget,
    resolver_for,
    socket_spec,
    tls_contexts,
)
from fakes.native_http import NativeTarget
from mitmproxy import ctx, master
from mitmproxy.addons.tlsconfig import TlsConfig
from mitmproxy.net import tls as native_tls


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


async def open_engine(trust, *, contextual=True):
    engine = MitmproxyEngine()
    await engine.start()
    # Deliberately give every owner the same trust only for the negative control.
    engine.assessment_trust = None if contextual else trust
    try:
        _ = ctx.options.ssl_verify_upstream_trusted_ca
    except LookupError:
        pass
    else:
        raise RuntimeError("unowned TLS options must fail closed")
    return engine


async def close_engine(engine):
    await engine.close()


async def owner_for(stack, engine, target, trust, tag):
    raw = socket_spec(target).model_dump(mode="json", by_alias=True, exclude_none=True)
    raw["allocationId"] = "allocation-" + tag
    ref = {"credentialId": "credential-shared", "kind": "target-bearer@1"}
    raw["engagement"]["http"]["identities"].append(
        {
            "id": "same-name",
            "kind": "bearer",
            "origins": [target.origin],
            "credential": ref,
        }
    )
    raw["runtimeSettings"]["httpIdentities"] = {
        "same-name": {"credential": ref, "fields": {"token": tag}}
    }
    owner = HTTPTransport(
        AllocationSpec.model_validate_json(json.dumps(raw)),
        ssl_context=engine.assessment_trust or trust,
        native_engine=engine,
        resolver=resolver_for(target).resolver,
    )
    owner.configure(authority_deadline=lambda: time.monotonic() + 120)
    await owner._initialize()
    identity = owner._sessions[1]
    identity.receive(
        PrivateResponse(
            target.origin + "/",
            200,
            [(b"set-cookie", f"session={tag}; Path=/".encode())],
        ),
        identity.epoch,
    )
    stack.push_async_callback(owner.close)
    return owner


async def call(owner, target, path="/allowed"):
    return await owner.request(target.origin + path, credential_id="credential-shared")


def header(row, name):
    headers = row[3]
    if isinstance(headers, dict):
        return headers.get(name, b"")
    return next(
        (
            line.split(b":", 1)[1].strip()
            for line in headers.split(b"\r\n")
            if line.lower().startswith(name + b":")
        ),
        b"",
    )


def verify_trace(target, tags):
    for row, tag in zip(target.trace, tags, strict=True):
        require(
            header(row, b"authorization") == b"Bearer " + tag.encode(),
            "auth owner mismatch",
        )
        require(
            header(row, b"cookie") == b"session=" + tag.encode(),
            "cookie owner mismatch",
        )


async def ordinary_sessions(directory, http2):
    server, trust, _ = tls_contexts(directory)
    server.set_alpn_protocols(["h2"] if http2 else ["http/1.1"])
    engine = await open_engine(trust)
    engine_path = Path(engine.directory.name)
    refs = []
    try:
        async with AsyncExitStack() as stack:
            target = await stack.enter_async_context(NativeTarget(ssl_context=server))
            owners = [
                await owner_for(stack, engine, target, trust, f"fixture-{i}") for i in range(4)
            ]
            refs.extend(weakref.ref(owner._engine) for owner in owners)
            outcomes = await asyncio.gather(
                *(call(owner, target, f"/allocation-{index}") for index, owner in enumerate(owners))
            )
            require(
                all(o.kind == "completed" for o, _ in outcomes),
                "concurrent request failed",
            )
            observed = [
                header(row, b"authorization").removeprefix(b"Bearer ").decode()
                for row in target.trace
            ]
            require(set(observed) == {f"fixture-{i}" for i in range(4)}, "missing session")
            expected = [
                f"fixture-{row[1].decode().removeprefix('/allocation-')}" for row in target.trace
            ]
            verify_trace(target, expected)
            require(len(target.connections) == 4, "connections shared across owners")
            require(
                all(row[2] == (b"HTTP/2" if http2 else b"HTTP/1.1") for row in target.trace),
                "wrong protocol",
            )
            # A deliberate wrong-owner dispatch must fail the target-side checker.
            start = len(target.trace)
            await call(owners[1], target)
            detected = False
            try:
                verify_trace(type("Trace", (), {"trace": target.trace[start:]})(), ["fixture-0"])
            except RuntimeError:
                detected = True
            require(detected, "negative routing control did not detect wrong owner")
            for owner in owners:
                require(not owner._engine.flows, "completed flow retained")
            sessions = [owner._engine for owner in owners]
            await owners[0].close()
            require(
                sessions[0].closed and not sessions[0].pools,
                "session cleanup incomplete",
            )
            require(
                all([(await call(owner, target))[0].kind == "completed" for owner in owners[1:]]),
                "neighbor stopped",
            )
            require(engine_path.exists(), "shared engine files removed by session")
            del sessions
        del owner, owners
    finally:
        await close_engine(engine)
    gc.collect()
    require(all(ref() is None for ref in refs), "closed session retained")
    require(not engine_path.exists(), "engine files retained")
    return {
        "sessions": 4,
        "target_connections": 4,
        "negative_control_detected": detected,
        "private_auth_cookie_and_pool": True,
        "neighbor_cleanup": True,
        "released_refs": True,
    }


async def proxy_sessions(directory):
    server, trust, _ = tls_contexts(directory)
    server.set_alpn_protocols(["h2", "http/1.1"])
    engine = await open_engine(trust)
    try:
        async with AsyncExitStack() as stack:
            target = await stack.enter_async_context(NativeTarget(ssl_context=server))
            owners = [await owner_for(stack, engine, target, trust, f"proxy-{i}") for i in range(2)]
            bridges = [
                await stack.enter_async_context(
                    HTTPProxyBridge(
                        owner,
                        credential_id="credential-shared",
                        deadline=time.monotonic() + 60,
                    )
                )
                for owner in owners
            ]
            for index, bridge in enumerate(bridges):
                async with httpx.AsyncClient(
                    proxy=bridge.proxy_url,
                    verify=ssl.create_default_context(
                        cafile=Path(owners[index]._engine.directory.name) / "mitmproxy-ca-cert.pem"
                    ),
                    http2=True,
                    trust_env=False,
                ) as client:
                    response = await client.get(target.origin + "/allowed")
                    require(
                        response.status_code == 200 and response.http_version == "HTTP/2",
                        "proxy HTTP/2 failed",
                    )
                require(
                    header(target.trace[-1], b"authorization") == f"Bearer proxy-{index}".encode(),
                    "proxy identity mixed",
                )
            # Move a token to the other listener. Neither a foreign token nor revoked
            # listener may grant access to the target. Do not record capability URLs.
            from urllib.parse import urlsplit, urlunsplit

            first, second = (urlsplit(bridge.proxy_url) for bridge in bridges)
            wrong = urlunsplit(
                first._replace(
                    netloc=first.netloc.split("@", 1)[0] + "@" + second.netloc.split("@", 1)[1]
                )
            )
            before = len(target.trace)
            async with httpx.AsyncClient(proxy=wrong, verify=False, trust_env=False) as client:
                with suppress(httpx.HTTPError):
                    await client.get(target.origin + "/allowed")
            require(len(target.trace) == before, "foreign token reached target")
            stale = bridges[0].proxy_url
            async with httpx.AsyncClient(
                proxy=stale, verify=False, http2=True, trust_env=False
            ) as client:
                active = asyncio.create_task(client.get(target.origin + "/hold"))
                try:
                    await target.held.wait()
                    await bridges[0].__aexit__()
                    result = await asyncio.gather(active, return_exceptions=True)
                    require(isinstance(result[0], httpx.HTTPError), "active scanner survived close")
                finally:
                    target.release.set()
                    await asyncio.gather(active, return_exceptions=True)
            before = len(target.trace)
            async with httpx.AsyncClient(proxy=stale, verify=False, trust_env=False) as client:
                with suppress(httpx.HTTPError):
                    await client.get(target.origin + "/allowed")
            require(len(target.trace) == before, "revoked token reached target")
            require(
                (await call(owners[1], target))[0].kind == "completed",
                "neighbor bridge stopped",
            )
        return {
            "sessions": 2,
            "connect_http2": True,
            "session_proxy_ca_verified": True,
            "active_scanner_joined": True,
            "foreign_token_rejected": True,
            "revoked_token_rejected": True,
            "neighbor_continues": True,
        }
    finally:
        await close_engine(engine)


async def trust_sessions(directory, *, contextual=False):
    first_dir, second_dir = directory / "first", directory / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    server_a, trust_a, _ = tls_contexts(first_dir)
    server_b, trust_b, _ = tls_contexts(second_dir)
    engine = await open_engine(trust_a, contextual=contextual)
    try:
        async with AsyncExitStack() as stack:
            a = await stack.enter_async_context(NativeTarget(ssl_context=server_a))
            b = await stack.enter_async_context(NativeTarget(ssl_context=server_b))
            # All policy routes are authorized. The only distinction is each owner's
            # effective TLS trust. Fresh sessions avoid connection reuse masking trust.
            a_a = await owner_for(stack, engine, a, trust_a, "trust-a")
            a_b = await owner_for(stack, engine, a, trust_b, "trust-b")
            b_b = await owner_for(stack, engine, b, trust_b, "trust-b")
            (good_a, _), (wrong_a, _), (good_b, _) = await asyncio.gather(
                call(a_a, a), call(a_b, a), call(b_b, b)
            )
            require(good_a.kind == "completed", "trust fixture baseline failed")
            return {
                "assigned_ca_a": good_a.kind,
                "unassigned_ca_a_with_session_trusting_b": wrong_a.kind,
                "assigned_ca_b_with_session_trusting_b": good_b.kind,
                "isolated": wrong_a.kind != "completed"
                and good_b.kind == "completed"
                and len(a.trace) == 1,
                "target_observed_unassigned_ca_request": len(a.trace) > 1,
            }
    finally:
        await close_engine(engine)


async def pending_tls(directory):
    server, trust, _ = tls_contexts(directory)
    engine = await open_engine(trust)
    release = threading.Event()
    original = TlsConfig.tls_start_server
    entered, cleaning = asyncio.Event(), asyncio.Event()
    loop = asyncio.get_running_loop()
    first = True

    def held(config, data):
        nonlocal first
        if first:
            first = False
            loop.call_soon_threadsafe(entered.set)
            require(release.wait(10), "TLS barrier timeout")
        return original(config, data)

    try:
        async with AsyncExitStack() as stack:
            target = await stack.enter_async_context(NativeTarget(ssl_context=server))
            a = await owner_for(stack, engine, target, trust, "held-a")
            b = await owner_for(stack, engine, target, trust, "live-b")
            original_release = a._engine._release_exchange

            async def observe_cleanup(*args):
                cleaning.set()
                await original_release(*args)

            a._engine._release_exchange = observe_cleanup
            with patch.object(TlsConfig, "tls_start_server", held):
                task = asyncio.create_task(call(a, target))
                try:
                    await entered.wait()
                    task.cancel()
                    await cleaning.wait()
                    require(not task.done(), "TLS work abandoned on cancellation")
                    require(
                        (await call(b, target))[0].kind == "completed",
                        "neighbor TLS blocked",
                    )
                    release.set()
                    result = await asyncio.gather(task, return_exceptions=True)
                    require(
                        isinstance(result[0], asyncio.CancelledError),
                        "cancellation lost",
                    )
                    require(len(target.trace) == 1, "cancelled request reached target")
                    require(not a._engine.flows, "cancelled flow retained")
                finally:
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)
            await a.close()
            require((await call(b, target))[0].kind == "completed", "neighbor closed")
        return {
            "thread_joined": True,
            "neighbor_progress": True,
            "late_send_blocked": True,
        }
    finally:
        release.set()
        await close_engine(engine)


async def redirects_retry(directory):
    async def respond(target, reader, writer, message):
        path = message.split(b" ", 2)[1]
        status = b"302 Found\r\nLocation: /allowed\r\n" if path == b"/redirect" else b"200 OK\r\n"
        writer.write(b"HTTP/1.1 " + status + b"Content-Length: 2\r\n\r\nok")
        await writer.drain()

    _, trust, _ = tls_contexts(directory)
    engine = await open_engine(trust)
    try:
        async with AsyncExitStack() as stack:
            target = await stack.enter_async_context(SocketTarget(respond))
            owners = [await owner_for(stack, engine, target, trust, f"retry-{i}") for i in range(2)]
            outcomes = await asyncio.gather(*(call(owner, target, "/redirect") for owner in owners))
            require(all(o.kind == "completed" for o, _ in outcomes), "redirect failed")
            require(
                all(owner._request_count == 2 for owner in owners),
                "redirect accounting mixed",
            )
            for message in target.requests:
                auth = next(
                    line for line in message.split(b"\r\n") if line.startswith(b"Authorization:")
                )
                tag = auth.removeprefix(b"Authorization: Bearer ")
                require(b"session=" + tag in message, "redirect cookie identity mixed")
            # Fresh connections force a pre-send retry for each owner.
            for owner in owners:
                for pool in owner._engine.pools.values():
                    await pool.close()
                owner._engine.pools.clear()
            original = asyncio.open_connection
            refusals = 2

            async def connect(host, port, **kwargs):
                nonlocal refusals
                if port == target.port and refusals:
                    refusals -= 1
                    raise ConnectionRefusedError("local assessment pre-send refusal")
                return await original(host, port, **kwargs)

            with patch.object(asyncio, "open_connection", connect):
                results = await asyncio.gather(*(call(owner, target) for owner in owners))
            require(all(o.kind == "completed" for o, _ in results), "retry failed")
            require(
                sum(owner._request_count for owner in owners) == 8,
                "retry accounting changed",
            )
        return {
            "redirect_auth_cookie_local": True,
            "presend_refusals": 2,
            "attempts_independently_counted": True,
        }
    finally:
        await close_engine(engine)


async def pending_dns_and_queue(directory):
    server, trust, _ = tls_contexts(directory)
    server.set_alpn_protocols(["h2"])
    engine = await open_engine(trust)
    try:
        async with AsyncExitStack() as stack:
            target = await stack.enter_async_context(
                NativeTarget(ssl_context=server, max_streams=1)
            )
            a = await owner_for(stack, engine, target, trust, "dns-a")
            b = await owner_for(stack, engine, target, trust, "dns-b")
            entered, resume = asyncio.Event(), asyncio.Event()
            original = a._policy._resolver

            async def resolve(host, port):
                entered.set()
                await resume.wait()
                return await original(host, port)

            a._policy._resolver = resolve
            request = asyncio.create_task(call(a, target))
            try:
                await entered.wait()
                request.cancel()
                require((await call(b, target))[0].kind == "completed", "DNS blocks peer")
                require(not request.done(), "DNS abandoned")
                resume.set()
                result = await asyncio.gather(request, return_exceptions=True)
                require(
                    isinstance(result[0], asyncio.CancelledError),
                    "DNS cancellation lost",
                )
                require(len(target.trace) == 1, "cancelled DNS sent request")
            finally:
                resume.set()
                await asyncio.gather(request, return_exceptions=True)
            a._policy._resolver = original
            await call(a, target)
            held = asyncio.create_task(call(a, target, "/hold"))
            await target.held.wait()
            queued_entered = asyncio.Event()
            original_request = a._engine.request

            async def queued_request(*args):
                queued_entered.set()
                return await original_request(*args)

            a._engine.request = queued_request
            queued = asyncio.create_task(call(a, target))
            try:
                await queued_entered.wait()
                await a.close()
                results = await asyncio.gather(held, queued, return_exceptions=True)
                require(
                    all(isinstance(result, asyncio.CancelledError) for result in results),
                    "close did not cancel session requests",
                )
                require(
                    (await call(b, target))[0].kind == "completed",
                    "queue close stops peer",
                )
                require(len(target.trace) == 4, "queued send escaped close")
            finally:
                target.release.set()
                await asyncio.gather(held, queued, return_exceptions=True)
        return {
            "dns_joined": True,
            "queued_http2_cancelled": True,
            "peer_continues": True,
        }
    finally:
        await close_engine(engine)


async def churn(directory):
    server, trust, _ = tls_contexts(directory)
    engine = await open_engine(trust)
    refs, paths = [], []
    try:
        async with NativeTarget(ssl_context=server) as target:
            for cycle in range(8):
                async with AsyncExitStack() as stack:
                    owners = [
                        await owner_for(stack, engine, target, trust, f"churn-{cycle}-{i}")
                        for i in range(4)
                    ]
                    refs.extend(weakref.ref(owner._engine) for owner in owners)
                    paths.extend(Path(owner._engine.directory.name) for owner in owners)
                    results = await asyncio.gather(*(call(owner, target) for owner in owners))
                    require(
                        all(outcome.kind == "completed" for outcome, _ in results),
                        "churn request failed",
                    )
                del owners
                await target.all_closed.wait()
            # Run done callbacks before checking weak references to joined tasks.
            await asyncio.sleep(0)
            gc.collect()
            require(all(ref() is None for ref in refs), "churn session retained")
            require(all(not path.exists() for path in paths), "churn TLS files retained")
            require(
                native_tls.create_proxy_server_context.cache_info().currsize == 0,
                "native target TLS cache retained",
            )
            require(
                native_tls.create_client_proxy_context.cache_info().currsize == 0,
                "native ingress TLS cache retained",
            )
        return {
            "cycles": 8,
            "sessions": 32,
            "session_refs_released": True,
            "private_directories_removed": True,
            "tls_caches_empty": True,
            "target_sockets_closed": True,
        }
    finally:
        await close_engine(engine)


async def assess():
    install_shared_config(
        decode_private(
            SharedConfig,
            (ROOT / "runtime/tests/testdata/shared-config.json").read_bytes(),
        ).snapshot()
    )
    matrix = {}
    with tempfile.TemporaryDirectory(prefix="contractor-shared-http-assessment-") as scratch:
        root = Path(scratch)
        for name, probe in [
            ("http11", lambda p: ordinary_sessions(p, False)),
            ("http2", lambda p: ordinary_sessions(p, True)),
            ("scanner", proxy_sessions),
            ("tls_cancellation", pending_tls),
            ("distinct_trust", trust_sessions),
            ("contextual_trust", lambda p: trust_sessions(p, contextual=True)),
            ("redirect_retry", redirects_retry),
            ("dns_queue_cleanup", pending_dns_and_queue),
            ("churn", churn),
        ]:
            directory = root / name
            directory.mkdir()
            original_master = master.Master.__init__
            starts = 0

            def counted_start(instance, *args, original_start=original_master, **kwargs):
                nonlocal starts
                starts += 1
                original_start(instance, *args, **kwargs)

            async with asyncio.timeout(40):
                with patch.object(master.Master, "__init__", counted_start):
                    matrix[name] = await probe(directory)
            require(starts == 1, "case did not use exactly one native engine")
            matrix[name]["native_master_creations"] = starts
    require(
        not matrix["distinct_trust"]["isolated"],
        "baseline negative control disappeared",
    )
    isolated = matrix["contextual_trust"]["isolated"]
    return {
        "verdict": "supported" if isolated else "unsupported",
        "scope": (
            "one native Master, task-local options adapter, allocation TLS/pools "
            "and ordinary HTTPTransport owners"
        ),
        "native_master_count_per_case": 1,
        "runtime_process_count": 1,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mitmproxy", "cryptography", "pyOpenSSL", "httpx")
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "lock_sha256": hashlib.sha256((ROOT / "runtime/uv.lock").read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "matrix": matrix,
        "remaining_proof": [],
        "recommendation": (
            "Use one pinned native Master with allocation-owned TLS/connection state "
            "and a stable task-local ctx.options adapter. Guard task/thread context "
            "propagation and release with these real-socket regressions."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(assess())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verdict": result["verdict"], "cases": len(result["matrix"])}))


if __name__ == "__main__":
    main()
