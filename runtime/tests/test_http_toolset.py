from __future__ import annotations

import asyncio
import base64
import os
import select
import socket
import ssl
import sys
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h11
import httpcore
import httpx
import idna
import pytest
from credential_field_values import read_credential_field_value_cases
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from fakes.worker_state import WorkerState
from http_cookie_store import HTTPToolsetFactory

import contractor_runtime.toolsets.http.tools as http_tools
from contractor_runtime import shared_config
from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.adapters.http_proxy import ProxyHTTPClient
from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactTransportError,
)
from contractor_runtime.contracts import (
    ArtifactRef,
    HTTPOriginTargetSettings,
    HTTPProxySettings,
    RuntimeSettings,
)
from contractor_runtime.contracts.fields import is_credential_field_value, is_field_value
from contractor_runtime.core.target_policy import TargetDenied
from contractor_runtime.toolsets.http.fields import is_cookie_pair
from contractor_runtime.toolsets.http.tools import (
    HTTPToolError,
)
from contractor_runtime.worker.instrumentation import WorkerInstrumentationPlugin
from contractor_runtime.workspace import AllocationWorkspace

SECRET = "recognizable-http-session-secret"
TARGET_SECRET = "recognizable-project-origin-secret"
# Header values httpx or h11 refuse to send.
UNSENDABLE_FIELD_VALUES = ("caf\u00e9", "trailing ", "trailing\t", " leading", "a\x0bb", "a\x0cb")


@pytest.mark.parametrize(
    "raw_query",
    ["q=%FF", "q=%20&flag", "q=a%2fb", "q=one&q=two", "flag&empty=&", ""],
)
@pytest.mark.parametrize("query", [None, {}, {"added": ["a b", "/"]}])
def test_request_preserves_raw_query_and_only_encodes_additions(
    tmp_path: Path, raw_query: str, query: dict[str, Any] | None
) -> None:
    async def scenario() -> None:
        observed: list[bytes] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            observed.append(request.url.raw_path)
            return httpx.Response(200, request=request)

        tools, _ = await create_tools(tmp_path, handler)
        try:
            await tools["http_request"]("https://target.example/path?" + raw_query, query=query)
            expected = raw_query
            if query:
                expected += ("&" if raw_query else "") + "added=a+b&added=%2F"
            assert observed == [b"/path?" + expected.encode("ascii")]
        finally:
            await tools["http_request"].close()

    asyncio.run(scenario())


def test_raw_query_and_added_parameters_share_request_bounds(tmp_path: Path) -> None:
    async def scenario() -> None:
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, request=request)

        tools, _ = await create_tools(tmp_path, handler)
        try:
            raw = "&".join("flag" for _ in range(http_tools.MAX_QUERY_KEYS))
            await tools["http_request"]("https://target.example/?" + raw)
            for url, query in (
                ("https://target.example/?" + raw + "&one", None),
                ("https://target.example/?" + raw, {"one": "more"}),
                ("https://target.example/?q=" + "x" * http_tools.MAX_URL_BYTES, None),
                ("https://target.example/?q=" + "x" * 8100, {"added": "x" * 100}),
            ):
                with pytest.raises(HTTPToolError, match="http_request_invalid"):
                    await tools["http_request"](url, query=query)
            assert calls == 1
        finally:
            await tools["http_request"].close()

    asyncio.run(scenario())


def test_direct_text_binary_status_redirect_and_exact_body_reads(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/redirect":
            return httpx.Response(302, headers={"Location": "/text"}, request=request)
        if request.url.path == "/text":
            return httpx.Response(
                404,
                content="hello world",
                headers={"Content-Type": "text/plain; charset=utf-8", "Set-Cookie": "sid=x"},
                request=request,
            )
        return httpx.Response(
            500,
            content=b"\x00\xffpayload",
            headers={"Content-Type": "application/octet-stream"},
            request=request,
        )

    async def scenario() -> None:
        artifacts = FakeArtifactClient()
        tools, state = await create_tools(tmp_path, handler, artifacts=artifacts)
        request = tools["http_request"]
        read = tools["http_read_body"]

        text = await request("https://target.example/redirect", follow_redirects=True)
        assert text["status"] == 404
        assert text["body_kind"] == "text"
        assert text["body_preview"] == "hello world"
        assert text["redirects"] == 1
        assert "body_artifact" not in text
        assert artifacts.writes == 0
        text_slice = await read(text["request_id"], offset=6, length=5)
        assert text_slice == {
            "request_id": 1,
            "kind": "text",
            "unit": "characters",
            "offset": 6,
            "length": 5,
            "total": 11,
            "eof": True,
            "body_truncated": False,
            "data": "world",
        }

        binary = await request("https://target.example/binary")
        assert binary["status"] == 500
        assert binary["retries"] == 2
        assert binary["body_kind"] == "binary"
        assert binary["body_preview"] is None
        binary_slice = await read(binary["request_id"], offset=1, length=3)
        assert base64.b64decode(binary_slice["data_b64"]) == b"\xffpa"
        assert [item.url.path for item in calls] == [
            "/redirect",
            "/text",
            "/binary",
            "/binary",
            "/binary",
        ]
        assert state.metrics.counters["tool_calls"] == 4
        assert SECRET not in repr(state.metrics)

    asyncio.run(scenario())


def test_http_body_reads_cache_pages_and_evict_between_requests(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/text":
            return httpx.Response(
                200,
                content="a😀éz".encode(),
                headers={"content-type": "text/plain"},
                request=request,
            )
        return httpx.Response(
            200,
            content=b"\x00\xffpayload",
            headers={"content-type": "application/octet-stream"},
            request=request,
        )

    async def scenario() -> None:
        artifacts = FakeArtifactClient()
        tools, _state = await create_tools(tmp_path, handler, artifacts=artifacts)
        read = tools["http_read_body"]
        try:
            text = await tools["http_request"]("https://target.example/text")
            binary = await tools["http_request"]("https://target.example/binary")
            text_id, binary_id = text["request_id"], binary["request_id"]

            assert await read(text_id, offset=1, length=2) == {
                "request_id": text_id,
                "kind": "text",
                "unit": "characters",
                "offset": 1,
                "length": 2,
                "total": 4,
                "eof": False,
                "body_truncated": False,
                "data": "😀é",
            }
            assert (await read(text_id, offset=3, length=2))["data"] == "z"
            assert artifacts.reads == 0

            binary_page = await read(binary_id, offset=1, length=3)
            assert binary_page == {
                "request_id": binary_id,
                "kind": "binary",
                "unit": "bytes",
                "offset": 1,
                "length": 3,
                "total": 9,
                "eof": False,
                "body_truncated": False,
                "data_b64": base64.b64encode(b"\xffpa").decode(),
            }
            assert base64.b64decode((await read(binary_id, offset=4))["data_b64"]) == b"yload"
            assert artifacts.reads == 0
            assert tools["http_request"]._session._cached_body.request_id == binary_id
            assert sys.getsizeof(tools["http_request"]._session._cached_body.content) <= (
                http_tools.decoded_body_cache_max_bytes()
            )

            assert (await read(text_id, offset=4))["eof"] is True
            assert artifacts.reads == 0
            assert tools["http_request"]._session._cached_body.request_id == text_id
        finally:
            await close_tools(tools)

        assert tools["http_request"]._session._cached_body is None
        with pytest.raises(HTTPToolError) as failure:
            await read(text_id)
        assert failure.value.code == "http_request_failed"
        assert artifacts.reads == 0

    asyncio.run(scenario())


def test_http_body_read_skips_content_above_cache_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"body", headers={"content-type": "text/plain"}, request=request
        )

    async def scenario() -> None:
        artifacts = FakeArtifactClient()
        tools, _state = await create_tools(tmp_path, handler, artifacts=artifacts)
        try:
            result = await tools["http_request"]("https://target.example/body")
            monkeypatch.setattr(
                http_tools, "decoded_body_cache_max_bytes", lambda: sys.getsizeof("body") - 1
            )
            assert (await tools["http_read_body"](result["request_id"]))["data"] == "body"
            assert tools["http_request"]._session._cached_body is None
            assert (await tools["http_read_body"](result["request_id"]))["data"] == "body"
            assert artifacts.reads == 0
        finally:
            await close_tools(tools)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "error",
    [
        ArtifactTransportError("temporary transport failure"),
        ArtifactAPIError(503, "unavailable", True),
    ],
)
def test_http_body_read_uses_buffer_during_artifact_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    async def scenario() -> None:
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                200, content=b"stored body", headers={"content-type": "text/plain"}, request=request
            )

        artifacts = FakeArtifactClient()
        tools, _ = await create_tools(tmp_path, handler, artifacts=artifacts)
        try:
            result = await tools["http_request"]("https://target.example/body")

            async def fail_read(_ref: ArtifactRef) -> Any:
                raise error

            with monkeypatch.context() as patch:
                patch.setattr(artifacts, "read_artifact", fail_read)
                assert (await tools["http_read_body"](result["request_id"]))[
                    "data"
                ] == "stored body"
                assert artifacts.reads == 0

            recovered = await tools["http_read_body"](result["request_id"])
            assert recovered["data"] == "stored body"
            assert calls == 1
        finally:
            await close_tools(tools)

    asyncio.run(scenario())


def test_http_body_read_expires_with_capture_eviction_and_session_clear(tmp_path: Path) -> None:
    async def scenario() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="buffered body", request=request)

        artifacts = FakeArtifactClient()
        tools, state = await create_tools(tmp_path, handler, artifacts=artifacts)
        try:
            first = await tools["http_request"]("https://target.example/first")
            assert (await tools["http_read_body"](first["request_id"]))["data"] == "buffered body"
            for number in range(http_tools.MAX_HISTORY):
                await tools["http_request"](f"https://target.example/item-{number}")
            with pytest.raises(HTTPToolError) as expired:
                await tools["http_read_body"](first["request_id"])
            assert expired.value.code == "http_body_not_found"
            assert len(await tools["http_history"]()) == http_tools.MAX_HISTORY
            await tools["http_session_clear"]()
            assert not state.resources.http_captures._entries
            assert tools["http_request"]._session._cached_body is None
            assert artifacts.writes == artifacts.reads == 0
        finally:
            await close_tools(tools)

    asyncio.run(scenario())


def test_request_validation_private_origin_and_retry_policy(tmp_path: Path) -> None:
    counts: dict[str, int] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        counts[path] = counts.get(path, 0) + 1
        if path == "/retry" and counts[path] < 3:
            return httpx.Response(503, content=b"retry", request=request)
        return httpx.Response(503 if path == "/post" else 200, content=b"done", request=request)

    async def scenario() -> None:
        tools, _state = await create_tools(tmp_path, handler)
        request = tools["http_request"]

        retried = await request("https://target.example/retry")
        assert retried["status"] == 200
        assert retried["retries"] == 2
        posted = await request(
            "https://target.example/post",
            method="POST",
            body_type="json",
            body={"safe": True},
        )
        assert posted["status"] == 503
        assert posted["retries"] == 0
        assert counts == {"/retry": 3, "/post": 1}

        for kwargs, code in (
            ({"url": "https://control.example/private"}, "http_target_denied"),
            (
                {"url": "https://target.example", "headers": {"X-Test": "x\r\ny"}},
                "http_request_invalid",
            ),
            (
                {"url": "https://target.example", "headers": {"Host": "elsewhere"}},
                "http_request_invalid",
            ),
            ({"url": "file:///tmp/private"}, "http_request_invalid"),
        ):
            with pytest.raises(HTTPToolError) as failure:
                await request(**kwargs)
            assert failure.value.code == code

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["direct", "proxy"])
def test_malformed_urls_fail_before_send_without_proxy_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    async def scenario() -> None:
        calls: list[str] = []
        delays: list[float] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            if request.url.path == "/transport":
                raise httpx.ConnectError("transport unavailable")
            return httpx.Response(200, request=request)

        async def record_retry(delay: float) -> None:
            delays.append(delay)

        monkeypatch.setattr(http_tools, "_retry_sleep", record_retry)
        proxy_client: httpx.AsyncClient | None = None
        if route == "direct":
            tools, _ = await create_tools(tmp_path, handler)
        else:
            proxy_client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler), trust_env=False
            )
            factory = HTTPToolsetFactory()
            tools = await make_tools(
                factory,
                tmp_path,
                settings=proxy_runtime_settings(),
                adapter_handles=AdapterHandles(tool_http=ProxyHTTPClient(proxy_client)),
            )
        try:
            for url in (
                "https://target.example/a\x00b",
                "https://target.example/a\x01b",
                "https://target.example/a\x7fb",
                "https://target.example/a\tb",
                "https://xn--zz.example/",
            ):
                with pytest.raises(HTTPToolError) as failure:
                    await tools["http_request"](url)
                assert failure.value.code == "http_request_invalid", repr(url)
                assert failure.value.retryable is False
                assert calls == []
                assert delays == []

            original_build_request = httpx.AsyncClient.build_request

            def failing_build_request(
                client: httpx.AsyncClient, method: str, url: str, **kwargs: Any
            ) -> httpx.Request:
                if url.endswith("/invalid-url"):
                    raise httpx.InvalidURL("invalid URL")
                if url.endswith("/invalid-idna"):
                    raise idna.IDNAError("invalid A-label")
                return original_build_request(client, method, url, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(httpx.AsyncClient, "build_request", failing_build_request)
                for suffix in ("invalid-url", "invalid-idna"):
                    with pytest.raises(HTTPToolError) as failure:
                        await tools["http_request"](f"https://target.example/{suffix}")
                    assert failure.value.code == "http_request_invalid"
                    assert failure.value.retryable is False
                    assert calls == []
                    assert delays == []

            # Input from which no request can be sent is invalid on both routes:
            # an unusable port, or a header value the HTTP/1.1 field grammar
            # rejects (non-ASCII, VT, FF, leading or trailing space or tab).
            for url, headers in (
                ("https://target.example:99999/", None),
                *(
                    ("https://target.example/", {"X-Note": value})
                    for value in UNSENDABLE_FIELD_VALUES
                ),
            ):
                with pytest.raises(HTTPToolError) as failure:
                    await tools["http_request"](url, headers=headers)
                assert failure.value.code == "http_request_invalid", repr((url, headers))
                assert failure.value.retryable is False
                assert calls == []
                assert delays == []
            # The session rejects the same input before storing it, so a stored
            # session value can never make later requests invalid.
            for update in (
                *({"headers": {"X-Note": value}} for value in UNSENDABLE_FIELD_VALUES),
                # A cookie's name=value pair is one field value: a leading
                # space only matters before the name.
                *(
                    {"cookies": {"session": value}}
                    for value in UNSENDABLE_FIELD_VALUES
                    if not value.startswith(" ")
                ),
                {"cookies": {" session": "value"}},
                *(
                    {"auth": {"kind": "bearer", "token": value}}
                    for value in UNSENDABLE_FIELD_VALUES
                ),
            ):
                with pytest.raises(HTTPToolError) as rejected:
                    await tools["http_session_set"](**update)
                assert rejected.value.code == "http_request_invalid", repr(update)
            session = await tools["http_session_get"]()
            assert (session["auth_kind"], session["default_headers"], session["cookie_count"]) == (
                "none",
                {},
                0,
            )
            assert calls == []

            if route == "proxy":
                with pytest.raises(HTTPToolError) as transport_failure:
                    await tools["http_request"]("https://target.example/transport")
                assert transport_failure.value.code == "http_request_failed"
                assert transport_failure.value.retryable is True
                assert len(calls) == 3
                assert delays == [0.25, 0.5]
        finally:
            await close_tools(tools)
            if proxy_client is not None:
                await proxy_client.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("route", "injection"),
    [
        ("direct", "transport"),
        ("direct", "client"),
        ("proxy", "client"),
        ("direct", "capture"),
        ("proxy", "capture"),
    ],
)
def test_unexpected_send_exception_reaches_tool_error_callback_unclassified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str, injection: str
) -> None:
    async def scenario() -> None:
        calls: list[str] = []
        delays: list[float] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            if request.url.path == "/bug" and injection == "transport":
                raise TypeError(f"injected implementation failure {SECRET}")
            if request.url.path == "/connect":
                raise httpx.ConnectError("recognizable connection failure")
            return httpx.Response(200, content=b"ok", request=request)

        async def record_retry(delay: float) -> None:
            delays.append(delay)

        original_from_request = http_tools.CapturedAttempt.from_request.__func__

        def failing_from_request(
            cls: type, request: httpx.Request, *, target_credential: bool = False
        ) -> Any:
            if request.url.path == "/bug":
                raise TypeError(f"injected implementation failure {SECRET}")
            return original_from_request(cls, request, target_credential=target_credential)

        original_extract_cookies = httpx.Cookies.extract_cookies

        def failing_extract_cookies(cookies: httpx.Cookies, response: httpx.Response) -> None:
            # httpx client code that runs after the transport on both routes.
            if response.request.url.path == "/bug":
                raise TypeError(f"injected implementation failure {SECRET}")
            original_extract_cookies(cookies, response)

        monkeypatch.setattr(http_tools, "_retry_sleep", record_retry)
        if injection == "capture":
            monkeypatch.setattr(
                http_tools.CapturedAttempt, "from_request", classmethod(failing_from_request)
            )
        if injection == "client":
            monkeypatch.setattr(httpx.Cookies, "extract_cookies", failing_extract_cookies)
        state = WorkerState()
        proxy_client: httpx.AsyncClient | None = None
        if route == "direct":
            tools = await make_tools(
                HTTPToolsetFactory(
                    lambda: httpx.AsyncClient(
                        transport=httpx.MockTransport(handler), trust_env=False
                    )
                ),
                tmp_path,
                state=state,
            )
        else:
            proxy_client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler), trust_env=False
            )
            tools = await make_tools(
                HTTPToolsetFactory(),
                tmp_path,
                state=state,
                settings=proxy_runtime_settings(),
                adapter_handles=AdapterHandles(tool_http=ProxyHTTPClient(proxy_client)),
            )
        callbacks = await WorkerToolCallbacks.start(state, tools)
        try:
            request = tools["http_request"]
            # Even an idempotent request is not retried: the exception keeps
            # its identity and is not a transport failure.
            response, raised = await callbacks.call(request, url="https://target.example/bug")
            assert type(raised) is TypeError
            assert response == {
                "ok": False,
                "error": {
                    "code": "tool_call_failed",
                    "message": "http_request failed (TypeError)",
                    "retryable": False,
                },
            }
            assert delays == []
            assert calls == ([] if injection == "capture" else ["/bug"])

            # A known connection failure keeps its classified retryability and
            # the idempotent retry policy.
            response, raised = await callbacks.call(request, url="https://target.example/connect")
            assert isinstance(raised, HTTPToolError)
            assert response == {
                "ok": False,
                "error": {
                    "code": "http_request_failed",
                    "message": "http_request failed (HTTPToolError)",
                    "retryable": True,
                },
            }
            assert delays == [0.25, 0.5]

            # Request IDs stay monotonic; the failed send keeps its explicit record.
            response, raised = await callbacks.call(request, url="https://target.example/ok")
            assert raised is None
            assert response["request_id"] == 3
            history = {item["request_id"]: item for item in await tools["http_history"]()}
            if injection == "capture":
                assert 1 not in history
            else:
                assert history[1]["error"] == "transport_error"
            assert history[2]["error"] == "transport_error"
            assert history[3]["status"] == 200
            retained = repr(state.metrics)
            assert "tool_call_failed" in retained
            assert SECRET not in retained
            assert "injected implementation failure" not in retained
        finally:
            await close_tools(tools)
            if proxy_client is not None:
                await proxy_client.aclose()

    asyncio.run(scenario())


def h11_sends(value: str) -> bool:
    try:
        h11.Request(method="GET", target="/", headers=[("Host", "t"), ("X-Note", value)])
    except (h11.LocalProtocolError, UnicodeEncodeError):
        return False
    return True


def test_field_grammar_matches_the_h11_serializer() -> None:
    # Every ASCII character alone, inside, leading and trailing a value, plus
    # whitespace runs and non-ASCII text.
    corpus = {"", "a b", "a \t b", "a\t\tb", "caf\u00e9", "\u00e9"}
    for code in range(128):
        char = chr(code)
        corpus.update({char, f"a{char}b", f"{char}a", f"a{char}"})
    disagreements = [value for value in sorted(corpus) if is_field_value(value) != h11_sends(value)]
    assert disagreements == []
    assert (
        is_cookie_pair("sid", "a b") and is_cookie_pair("sid", "") and is_cookie_pair("sid", None)
    )
    assert not is_cookie_pair("", "value")
    assert not is_cookie_pair(" sid", "value")
    assert not is_cookie_pair("sid", "value ")
    assert not is_cookie_pair("sid", "caf\u00e9")


def test_shared_credential_field_values_are_sent_unchanged_by_h11() -> None:
    # The corpus of the shared credential field-value grammar: is_field_value
    # still agrees with h11 on all of it, and h11 sends every credential value
    # the grammar accepts, so no configured token fails at send time.
    cases = read_credential_field_value_cases().grammar
    disagreements = [
        case.name for case in cases if is_field_value(case.value) != h11_sends(case.value)
    ]
    assert disagreements == []
    assert [case.name for case in cases if case.valid and not h11_sends(case.value)] == []
    assert [
        case.name
        for case in cases
        if is_credential_field_value(case.value) and not is_field_value(case.value)
    ] == []


def loopback_tls_transport(certificate: tuple[Path, Path], port: int) -> httpx.AsyncHTTPTransport:
    """A transport whose pool connects target.example to a local server over real TLS."""

    transport = httpx.AsyncHTTPTransport(trust_env=False, retries=0)
    # As PolicyHTTPTransport does, replace httpx's private pool to choose
    # the network backend; the TLS stack and exception mapping stay real.
    transport._pool = httpcore.AsyncConnectionPool(
        ssl_context=ssl.create_default_context(cafile=str(certificate[0])),
        retries=0,
        network_backend=LoopbackBackend(port),
    )
    return transport


def loopback_tls_client(certificate: tuple[Path, Path], port: int) -> httpx.AsyncClient:
    """A client whose pool connects target.example to a local server over real TLS."""

    return httpx.AsyncClient(transport=loopback_tls_transport(certificate, port), trust_env=False)


class RecordingTLSTarget:
    """A real TLS target that records each request head and answers 200 OK."""

    def __init__(self, tmp_path: Path) -> None:
        self.certificate = issue_target_certificate(tmp_path)
        self._context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._context.load_cert_chain(*self.certificate)
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        self.heads: list[bytes] = []
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                raw, _ = self._listener.accept()
            except OSError:
                return
            try:
                with self._context.wrap_socket(raw, server_side=True) as tls:
                    head = b""
                    while b"\r\n\r\n" not in head:
                        chunk = tls.recv(65536)
                        if not chunk:
                            break
                        head += chunk
                    self.heads.append(head)
                    tls.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
                    )
            except OSError:
                pass

    def client(self) -> httpx.AsyncClient:
        return loopback_tls_client(self.certificate, self.port)

    def close(self) -> None:
        self._listener.close()
        self._thread.join(30)


def test_project_target_token_is_sent_unchanged_over_real_tls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A configured token with inner space and tab runs is a credential field
    # value (RCF-R114): it reaches the target as configured, on one attempt.
    token = "tok\ten  inner-runs!~"
    assert is_credential_field_value(token)

    async def scenario() -> None:
        delays: list[float] = []

        async def record_retry(delay: float) -> None:
            delays.append(delay)

        monkeypatch.setattr(http_tools, "_retry_sleep", record_retry)
        target = RecordingTLSTarget(tmp_path)
        state = WorkerState()
        settings = RuntimeSettings(
            llmGatewayUrl="https://gateway.example/v1",
            artifactApiUrl="https://control.example/private/v1",
            httpOriginTarget=HTTPOriginTargetSettings(
                url="https://target.example", bearerToken=token
            ),
            requestTimeoutSeconds=30,
        )
        tools = await make_tools(
            HTTPToolsetFactory(target.client), tmp_path, state=state, settings=settings
        )
        callbacks = await WorkerToolCallbacks.start(state, tools)
        try:
            response, raised = await callbacks.call(
                tools["http_request"], url="https://target.example/"
            )
            assert raised is None
            assert response["status"] == 200
        finally:
            await close_tools(tools)
            target.close()
        assert delays == []
        assert len(target.heads) == 1
        lines = target.heads[0].split(b"\r\n")
        authorization = [
            line.split(b":", 1)[1] for line in lines if line.lower().startswith(b"authorization:")
        ]
        assert authorization == [b" Bearer " + token.encode("ascii")]

    asyncio.run(scenario())


class CorruptTLSRecordTarget:
    """A real TLS target that answers with a record the client cannot decrypt.

    In the ``headers`` phase the corrupt record replaces the response; in the
    ``body`` phase it follows the response head and part of the body. httpcore
    lets the resulting ssl.SSLError (an OSError) escape unmapped.
    """

    def __init__(self, tmp_path: Path, phase: str) -> None:
        self.phase = phase
        self.certificate = issue_target_certificate(tmp_path)
        self._context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._context.load_cert_chain(*self.certificate)
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        self.connections = 0
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                raw, _ = self._listener.accept()
            except OSError:
                return
            self.connections += 1
            try:
                with self._context.wrap_socket(raw, server_side=True) as tls:
                    tls.recv(65536)
                    if self.phase == "body":
                        tls.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\nabc")
                    # An application-data record with a wrong authentication tag.
                    os.write(tls.fileno(), b"\x17\x03\x03\x00\x40" + b"\x00" * 64)
                    # Close only after the client has: closing first could reset
                    # the connection before the client reads the corrupt record.
                    while select.select([tls.fileno()], [], [], 30)[0]:
                        if not os.read(tls.fileno(), 65536):
                            break
            except OSError:
                pass

    def client(self) -> httpx.AsyncClient:
        return loopback_tls_client(self.certificate, self.port)

    def transport(self) -> httpx.AsyncHTTPTransport:
        return loopback_tls_transport(self.certificate, self.port)

    def close(self) -> None:
        self._listener.close()
        self._thread.join(30)


class LoopbackBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, port: int) -> None:
        self._port = port
        self._backend = httpcore.AnyIOBackend()

    async def connect_tcp(self, host: str, port: int, **kwargs: Any) -> httpcore.AsyncNetworkStream:
        return await self._backend.connect_tcp("127.0.0.1", self._port, **kwargs)

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


def issue_target_certificate(tmp_path: Path) -> tuple[Path, Path]:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "target.example")])
    now = datetime.now(UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("target.example")]), False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), True)
        .sign(key, hashes.SHA256())
    )
    certificate_path = tmp_path / "target-cert.pem"
    key_path = tmp_path / "target-key.pem"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return certificate_path, key_path


@pytest.mark.parametrize("route", ["direct", "proxy"])
@pytest.mark.parametrize("phase", ["headers", "body"])
def test_tls_record_failure_from_a_target_is_a_classified_transport_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str, phase: str
) -> None:
    async def scenario() -> None:
        delays: list[float] = []

        async def record_retry(delay: float) -> None:
            delays.append(delay)

        monkeypatch.setattr(http_tools, "_retry_sleep", record_retry)
        target = CorruptTLSRecordTarget(tmp_path, phase)
        state = WorkerState()
        proxy_client: httpx.AsyncClient | None = None
        if route == "direct":
            tools = await make_tools(HTTPToolsetFactory(target.client), tmp_path, state=state)
        else:
            proxy_client = target.client()
            tools = await make_tools(
                HTTPToolsetFactory(),
                tmp_path,
                state=state,
                settings=proxy_runtime_settings(),
                adapter_handles=AdapterHandles(tool_http=ProxyHTTPClient(proxy_client)),
            )
        callbacks = await WorkerToolCallbacks.start(state, tools)
        try:
            response, raised = await callbacks.call(
                tools["http_request"], url="https://target.example/"
            )
            assert isinstance(raised, HTTPToolError)
            assert response == {
                "ok": False,
                "error": {
                    "code": "http_request_failed",
                    "message": "http_request failed (HTTPToolError)",
                    "retryable": True,
                },
            }
            if phase == "headers":
                # A failed send of an idempotent request is retried on both routes.
                assert delays == [0.25, 0.5]
                assert target.connections == 3
            else:
                # A response body that fails after delivery is not re-requested.
                assert delays == []
                assert target.connections == 1
            history = await tools["http_history"]()
            assert history[-1]["error"] == "transport_error"
        finally:
            await close_tools(tools)
            if proxy_client is not None:
                await proxy_client.aclose()
            target.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["direct", "proxy"])
def test_host_names_under_the_shared_host_rule_are_sent(tmp_path: Path, route: str) -> None:
    async def scenario() -> None:
        hosts: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            hosts.append(request.url.raw_host.decode("ascii"))
            return httpx.Response(200, request=request)

        proxy_client: httpx.AsyncClient | None = None
        if route == "direct":
            tools, _ = await create_tools(tmp_path, handler)
        else:
            proxy_client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler), trust_env=False
            )
            factory = HTTPToolsetFactory()
            tools = await make_tools(
                factory,
                tmp_path,
                settings=proxy_runtime_settings(),
                adapter_handles=AdapterHandles(tool_http=ProxyHTTPClient(proxy_client)),
            )
        # The longest label the shared host rule accepts; see test_host_syntax.
        long_label = "a" * 63
        try:
            for url in (
                "http://juice_shop:3000/",
                "https://web_app.example/",
                "https://ab--cd.example/",
                f"https://{long_label}.example/",
                "https://xn--bcher-kva.example/",
                "https://bücher.example/",
            ):
                await tools["http_request"](url)
        finally:
            await close_tools(tools)
            if proxy_client is not None:
                await proxy_client.aclose()
        assert hosts == [
            "juice_shop",
            "web_app.example",
            "ab--cd.example",
            f"{long_label}.example",
            "xn--bcher-kva.example",
            "xn--bcher-kva.example",
        ]

    asyncio.run(scenario())


def test_connect_time_target_denial_does_not_retain_an_exchange(tmp_path: Path) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise TargetDenied

    async def scenario() -> None:
        tools, _ = await create_tools(tmp_path, handler)
        try:
            request = tools["http_request"]
            session = request._session
            with pytest.raises(HTTPToolError) as failure:
                await request(
                    "https://target.example/denied",
                    tool_context=SimpleNamespace(invocation_id="invocation-1"),
                )
            assert failure.value.code == "http_target_denied"
            assert failure.value.retryable is False
            assert not session._captures.contains("http", "1")
            assert await tools["http_history"]() == []
        finally:
            await close_tools(tools)

    asyncio.run(scenario())


def test_session_is_redacted_serialized_and_history_is_bounded(tmp_path: Path) -> None:
    active = 0
    maximum_active = 0
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        observed.append(request)
        active -= 1
        if request.url.path == "/redirect-other":
            return httpx.Response(
                302,
                headers={"Location": "https://other.example/final"},
                request=request,
            )
        return httpx.Response(200, content=b"ok", request=request)

    async def scenario() -> None:
        tools, state = await create_tools(tmp_path, handler)
        await tools["http_session_set"](
            cookies={"sid": SECRET},
            headers={"X-API-Key": SECRET, "X-Safe": "visible"},
            auth={"kind": "bearer", "token": SECRET},
        )
        view = await tools["http_session_get"]()
        assert view == {
            "auth_kind": "bearer",
            "default_headers": {"X-API-Key": "[REDACTED]", "X-Safe": "visible"},
            "cookie_names": ["sid"],
            "cookie_count": 1,
            "history_count": 0,
        }
        assert SECRET not in repr(view)

        await asyncio.gather(
            tools["http_request"]("https://target.example/one"),
            tools["http_request"]("https://target.example/two"),
        )
        assert maximum_active == 1
        assert len(observed) == 2
        assert all(item.headers["authorization"] == f"Bearer {SECRET}" for item in observed)
        assert all("sid=" in item.headers["cookie"] for item in observed)
        await tools["http_request"]("https://target.example/redirect-other")
        assert observed[-1].url.host == "other.example"
        assert "authorization" not in observed[-1].headers
        assert "cookie" not in observed[-1].headers
        history = await tools["http_history"]()
        assert [item["request_id"] for item in history] == [1, 2, 3]
        assert all("body_preview" not in item for item in history)

        cleared = await tools["http_session_clear"]()
        assert cleared["auth_kind"] == "none"
        assert cleared["cookie_count"] == 0
        assert cleared["history_count"] == 0
        assert SECRET not in repr(state.metrics)

    asyncio.run(scenario())


def test_project_authorization_is_exact_origin_hidden_and_erased(tmp_path: Path) -> None:
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.url.path == "/redirect":
            return httpx.Response(
                302,
                headers={"Location": "https://other.example/final"},
                request=request,
            )
        return httpx.Response(200, content=b"ok", request=request)

    async def scenario() -> None:
        state = WorkerState()

        def direct() -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)

        factory = HTTPToolsetFactory(direct)
        settings = RuntimeSettings(
            llmGatewayUrl="https://gateway.example/v1",
            artifactApiUrl="https://control.example/private/v1",
            httpOriginTarget=HTTPOriginTargetSettings(
                url="https://target.example/application",
                bearerToken=TARGET_SECRET,
            ),
            requestTimeoutSeconds=30,
        )
        tools = await make_tools(factory, tmp_path, state=state, settings=settings)

        await tools["http_request"](
            "https://target.example/one",
            headers={"Authorization": "Bearer model-supplied"},
        )
        await tools["http_request"](
            "https://target.example:444/two",
            headers={"Authorization": "Bearer model-other-origin"},
        )
        await tools["http_request"]("https://target.example/redirect", follow_redirects=True)

        assert observed[0].headers["authorization"] == f"Bearer {TARGET_SECRET}"
        assert observed[1].headers["authorization"] == "Bearer model-other-origin"
        assert observed[2].headers["authorization"] == f"Bearer {TARGET_SECRET}"
        assert observed[3].url.host == "other.example"
        assert "authorization" not in observed[3].headers
        session = await tools["http_session_get"]()
        assert TARGET_SECRET not in repr(session)
        assert TARGET_SECRET not in repr(state)

        private_session = tools["http_request"]._session
        await close_tools(tools)
        assert private_session._target_origin is None
        assert private_session._target_authorization is None

    asyncio.run(scenario())


def test_proxy_route_is_required_and_target_errors_are_returned(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, content=b"target failure", request=request)

    async def scenario() -> None:
        settings = proxy_runtime_settings()
        factory = HTTPToolsetFactory()
        with pytest.raises(RuntimeError, match="resolved tool-http"):
            await make_tools(factory, tmp_path, settings=settings)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
        handle = ProxyHTTPClient(client)
        tools = await make_tools(
            factory,
            tmp_path,
            settings=settings,
            adapter_handles=AdapterHandles(tool_http=handle),
        )
        result = await tools["http_request"]("https://target.example/failure")
        assert result["status"] == 502
        assert result["retries"] == 2
        await close_tools(tools)
        await client.aclose()

    asyncio.run(scenario())


def test_oversized_response_and_proxy_auth_failure_leave_no_body_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        shared_config,
        "_snapshot",
        replace(shared_config.get_shared_config(), http_response_body_max_bytes=32),
    )

    async def oversized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 33, request=request)

    async def proxy_auth(request: httpx.Request) -> httpx.Response:
        return httpx.Response(407, content=b"must not be surfaced", request=request)

    async def scenario() -> None:
        artifacts = FakeArtifactClient()
        tools, _state = await create_tools(tmp_path, oversized, artifacts=artifacts)
        with pytest.raises(HTTPToolError) as too_large:
            await tools["http_request"]("https://target.example/large")
        assert too_large.value.code == "http_response_too_large"
        assert artifacts.writes == 0

        client = httpx.AsyncClient(transport=httpx.MockTransport(proxy_auth), trust_env=False)
        proxy = ProxyHTTPClient(client)
        factory = HTTPToolsetFactory(
            lambda: (_ for _ in ()).throw(AssertionError("direct fallback")),
        )
        proxied = await make_tools(
            factory,
            tmp_path,
            settings=proxy_runtime_settings(),
            adapter_handles=AdapterHandles(tool_http=proxy),
        )
        # A forwarded plain-HTTP 407 may be the proxy's own refusal.
        with pytest.raises(HTTPToolError) as failed:
            await proxied["http_request"]("http://target.example/private")
        assert failed.value.code == "http_request_failed"
        assert artifacts.writes == 0
        # Through an HTTPS tunnel the proxy cannot answer 407; it is target data.
        tunneled = await proxied["http_request"]("https://target.example/private")
        assert tunneled["status"] == 407
        assert tunneled["retries"] == 0
        await client.aclose()

    asyncio.run(scenario())


async def create_tools(
    tmp_path: Path,
    handler: Any,
    *,
    artifacts: FakeArtifactClient | None = None,
) -> tuple[dict[str, Any], WorkerState]:
    artifacts = artifacts or FakeArtifactClient()
    state = WorkerState()

    def direct() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)

    factory = HTTPToolsetFactory(direct)
    tools = await make_tools(factory, tmp_path, state=state)
    return tools, state


async def make_tools(
    factory: HTTPToolsetFactory,
    tmp_path: Path,
    *,
    state: WorkerState | None = None,
    settings: RuntimeSettings | None = None,
    adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
) -> dict[str, Any]:
    state = state or WorkerState()
    settings = settings or RuntimeSettings(
        llmGatewayUrl="https://gateway.example/v1",
        llmGatewayToken="gateway-token",
        artifactApiUrl="https://control.example/private/v1",
        requestTimeoutSeconds=30,
    )
    workspace_path = tmp_path / "allocation"
    result = await factory.create_selected(
        selected=sorted(factory.exported_tools),
        allocation_id="allocation-http",
        run_id="run-http",
        namespace="worker",
        runtime_settings=settings,
        workspace=AllocationWorkspace(root=tmp_path, path=workspace_path),
        resources=state.resources,
        adapter_handles=adapter_handles,
    )
    return dict(result)


async def close_tools(tools: dict[str, Any]) -> None:
    for tool in tools.values():
        await tool.close()


class WorkerToolCallbacks:
    """Call real tools between the Worker's ADK callbacks, exactly as ADK does.

    ADK runs before_tool_callback, awaits the tool and hands a raised exception
    to on_tool_error_callback, whose response replaces the tool result.
    """

    def __init__(self, plugin: WorkerInstrumentationPlugin, invocation_id: str) -> None:
        self._plugin = plugin
        self._context = SimpleNamespace(
            invocation_id=invocation_id,
            session=SimpleNamespace(id="session", app_name="http-tools", user_id="user", state={}),
        )

    @classmethod
    async def start(cls, state: WorkerState, tools: dict[str, Any]) -> WorkerToolCallbacks:
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            budget=lambda: None,
            instrumentation=None,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
            allowed_tool_names=frozenset(tools),
        )
        callbacks = cls(plugin, "worker-tool-errors")
        plugin.prepare_invocation(invocation_id="worker-tool-errors", subtask_id="1")
        await plugin.before_run_callback(invocation_context=callbacks._context)
        return callbacks

    async def call(self, tool: Any, **arguments: Any) -> tuple[Any, Exception | None]:
        adk_tool = SimpleNamespace(name=tool.name, func=tool)
        rejection = await self._plugin.before_tool_callback(
            tool=adk_tool, tool_args=arguments, tool_context=self._context
        )
        assert rejection is None
        try:
            result = await tool(**arguments)
        except Exception as error:
            response = await self._plugin.on_tool_error_callback(
                tool=adk_tool, tool_args=arguments, tool_context=self._context, error=error
            )
            return response, error
        await self._plugin.after_tool_callback(
            tool=adk_tool, tool_args=arguments, tool_context=self._context, result=result
        )
        return result, None


def proxy_runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        llmGatewayUrl="https://gateway.example/v1",
        llmGatewayToken="gateway-token",
        artifactApiUrl="https://control.example/private/v1",
        httpProxy=HTTPProxySettings(
            adapter="http-proxy@1",
            proxyUrl="https://proxy.example",
            targets=["tool-http"],
        ),
        requestTimeoutSeconds=30,
    )


class FakeArtifactClient:
    def __init__(self) -> None:
        self.payloads: dict[tuple[str, str, str], bytes] = {}
        self.media_types: dict[tuple[str, str, str], str] = {}
        self.writes = 0
        self.reads = 0

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> Any:
        assert expected_revision is None
        self.writes += 1
        exact = target.model_copy(update={"revision": f"revision-{self.writes}"})
        key = (exact.namespace, exact.name, exact.revision)
        self.payloads[key] = data
        self.media_types[key] = media_type
        return SimpleNamespace(artifact=exact)

    async def read_artifact(self, ref: ArtifactRef) -> Any:
        self.reads += 1
        exact = ref.require_exact()
        key = (exact.namespace, exact.name, exact.revision)
        if key not in self.payloads:
            raise RuntimeError("not found")
        return SimpleNamespace(data=self.payloads[key], media_type=self.media_types[key])
