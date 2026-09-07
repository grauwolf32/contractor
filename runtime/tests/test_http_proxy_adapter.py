from __future__ import annotations

import asyncio
import json
import os
import shutil
import ssl
import subprocess
import sys
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import httpx
import pytest
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import RuntimeAdapterBuildContext
from contractor_runtime.adapters.http_proxy import (
    HTTPProxyAdapter,
    HTTPProxyAdapterFactory,
    ProxyHTTPClient,
    ProxyRequestError,
    ProxySubprocessError,
    ProxySubprocessLauncher,
)
from contractor_runtime.contracts import HTTPProxySettingsV2
from contractor_runtime.factories import FactoryRegistry
from contractor_runtime.llm.factory import gateway_model
from contractor_runtime.llm.openai import GatewayModelError, OpenAICompatibleGatewayLlm

PROXY_PASSWORD = "recognizable-proxy-password"
PROXY_BEARER = "recognizable-proxy-bearer"
GATEWAY_TOKEN = "recognizable-gateway-token"
BACKEND_SECRET = "recognizable-backend-error"


def test_llm_gateway_target_routes_only_model_traffic_and_changes_no_globals() -> None:
    async def scenario() -> None:
        proxy_environment = {
            name: os.environ.get(name)
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "SSL_CERT_FILE")
        }
        assert await HTTPProxyAdapterFactory().probe()
        async with fake_proxy(openai_response) as proxy:
            adapter = HTTPProxyAdapter(
                adapter_context(),
                proxy_settings(proxy.url, targets=["llm-gateway"]),
            )
            assert isinstance(adapter.handles.model_http, ProxyHTTPClient)
            assert adapter.handles.tool_http is None
            assert adapter.handles.tool_subprocess is None
            context = SimpleNamespace(
                model_policy=SimpleNamespace(model="worker-model"),
                runtime_settings=SimpleNamespace(
                    llm_gateway_url="http://gateway.example/v1",
                    llm_gateway_token=SimpleSecret(GATEWAY_TOKEN),
                    request_timeout_seconds=2,
                ),
                adapter_handles=adapter.handles,
            )
            model = gateway_model(context)
            assert isinstance(model, OpenAICompatibleGatewayLlm)
            request = LlmRequest(
                contents=[types.Content(role="user", parts=[types.Part(text="Return ok")])],
                config=types.GenerateContentConfig(max_output_tokens=32),
            )
            responses = [response async for response in model.generate_content_async(request)]
            assert responses[-1].content is not None
            assert responses[-1].content.parts[0].text == "ok"

            direct_calls = 0

            async def private_transport(request: httpx.Request) -> httpx.Response:
                nonlocal direct_calls
                direct_calls += 1
                return httpx.Response(200, json={"ok": True}, request=request)

            async with httpx.AsyncClient(
                transport=httpx.MockTransport(private_transport), trust_env=False
            ) as private_client:
                assert (await private_client.get("https://control.example/heartbeat")).is_success
                assert (await private_client.get("https://control.example/artifacts")).is_success
            completed = subprocess.run(
                [sys.executable, "-c", "print('local')"],
                capture_output=True,
                check=True,
            )
            assert completed.stdout.strip() == b"local"
            assert direct_calls == 2
            assert len(proxy.requests) == 1
            assert proxy.requests[0].target.startswith("http://gateway.example/v1/chat/completions")
            assert proxy.requests[0].headers["proxy-authorization"].startswith("Basic ")
            await model.close()
            await adapter.close()
        assert {name: os.environ.get(name) for name in proxy_environment} == proxy_environment

    asyncio.run(scenario())


def test_tool_http_and_subprocess_use_authenticated_tls_proxy_and_remove_ca(
    tmp_path: Path,
) -> None:
    if shutil.which("openssl") is None:
        pytest.skip("openssl is required for the TLS proxy fixture")
    certificate, key = issue_local_certificate(tmp_path)

    async def scenario() -> None:
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(certificate, key)
        ca_bundle = certificate.read_text(encoding="utf-8")
        async with fake_proxy(text_response, tls=tls, host="localhost") as proxy:
            adapter = HTTPProxyAdapter(
                adapter_context(),
                proxy_settings(
                    proxy.url,
                    targets=["tool-http", "tool-subprocess"],
                    ca_bundle=ca_bundle,
                ),
            )
            http_handle = adapter.handles.tool_http
            launcher = adapter.handles.tool_subprocess
            assert isinstance(http_handle, ProxyHTTPClient)
            assert isinstance(launcher, ProxySubprocessLauncher)
            with pytest.raises(ProxyRequestError):
                await http_handle.request("GET", "https://control.example/private/v1")
            assert proxy.requests == []
            response = await http_handle.request("GET", "http://public.example/tool")
            assert response.text == "proxied"

            script = (
                "import os, urllib.request; "
                "p=os.environ['SSL_CERT_FILE']; "
                "print(p); "
                "print(urllib.request.urlopen('http://public.example/child').read().decode())"
            )
            completed = await asyncio.to_thread(
                launcher.run,
                [sys.executable, "-c", script],
                env={"PATH": os.environ.get("PATH", os.defpath), "LANG": "C.UTF-8"},
                timeout=5,
            )
            lines = completed.stdout.decode().splitlines()
            assert lines[-1] == "proxied"
            child_ca_path = Path(lines[0])
            assert not child_ca_path.exists()
            assert launcher.active_temporary_roots == ()
            assert len(proxy.requests) == 2
            assert all(
                request.headers["proxy-authorization"].startswith("Basic ")
                for request in proxy.requests
            )
            assert adapter.metrics.failed_operations == 1
            assert adapter.metrics.operations == 3
            await adapter.close()
            with pytest.raises(ProxyRequestError):
                _ = http_handle.async_client
            assert PROXY_PASSWORD not in repr(adapter)
            assert PROXY_PASSWORD not in repr(launcher)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["auth", "disconnect", "tls"])
def test_proxy_failures_are_bounded_and_never_fall_back(
    tmp_path: Path,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_retry_delay(_seconds: float) -> None:
        return None

    monkeypatch.setattr("openai._base_client.anyio.sleep", no_retry_delay)

    async def scenario() -> None:
        backend_calls = 0

        async def backend(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            nonlocal backend_calls
            backend_calls += 1
            writer.close()
            await writer.wait_closed()

        backend_server = await asyncio.start_server(backend, "127.0.0.1", 0)
        backend_port = backend_server.sockets[0].getsockname()[1]
        backend_url = f"http://127.0.0.1:{backend_port}/must-not-be-called"
        try:
            if failure == "auth":
                async with fake_proxy(auth_failure_response) as proxy:
                    await assert_safe_proxy_failure(proxy.url, backend_url)
                    await assert_safe_model_proxy_failure(proxy.url, backend_url)
            elif failure == "disconnect":
                async with fake_proxy(disconnect_response) as proxy:
                    await assert_safe_proxy_failure(proxy.url, backend_url)
                    await assert_safe_model_proxy_failure(proxy.url, backend_url)
            else:
                if shutil.which("openssl") is None:
                    pytest.skip("openssl is required for the TLS proxy fixture")
                certificate, key = issue_local_certificate(tmp_path)
                tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                tls.load_cert_chain(certificate, key)
                async with fake_proxy(text_response, tls=tls, host="localhost") as proxy:
                    await assert_safe_proxy_failure(proxy.url, backend_url)
                    await assert_safe_model_proxy_failure(proxy.url, backend_url)
            assert backend_calls == 0
        finally:
            backend_server.close()
            await backend_server.wait_closed()

    asyncio.run(scenario())


def test_bearer_subprocess_fails_closed_and_registry_rejects_channel_mismatch() -> None:
    async def scenario() -> None:
        async with fake_proxy(text_response) as proxy:
            settings = HTTPProxySettingsV2(
                adapter="http-proxy@1",
                proxyUrl=proxy.url,
                bearerToken=PROXY_BEARER,
                targets=["tool-http", "tool-subprocess"],
            )
            adapter = HTTPProxyAdapter(adapter_context(), settings)
            http_handle = adapter.handles.tool_http
            launcher = adapter.handles.tool_subprocess
            assert isinstance(http_handle, ProxyHTTPClient)
            assert isinstance(launcher, ProxySubprocessLauncher)
            assert (await http_handle.request("GET", "http://public.example/bearer")).is_success
            assert proxy.requests[0].headers["proxy-authorization"] == (f"Bearer {PROXY_BEARER}")
            with pytest.raises(ProxySubprocessError) as captured:
                await asyncio.to_thread(launcher.run, [sys.executable, "-c", "print('x')"])
            assert PROXY_BEARER not in repr(captured.value)
            assert len(proxy.requests) == 1
            await adapter.close()

    asyncio.run(scenario())

    with pytest.raises(ValueError, match="unknown tool"):
        registry_with_toolset(DescriptorToolset({"missing": frozenset({"runtime-http-client"})}))
    with pytest.raises(ValueError, match="invalid channels"):
        registry_with_toolset(DescriptorToolset({"local": frozenset({"ambient-network"})}))

    local = DescriptorToolset({})
    registry_with_toolset(local)
    handles = AdapterHandles(tool_http=object(), tool_subprocess=object())
    assert handles.for_tool_channels(frozenset()).enabled_channels == ()


def test_proxy_handle_preserves_target_http_errors_as_responses() -> None:
    async def scenario() -> None:
        async with fake_proxy(lambda _request: (503, b"target unavailable")) as proxy:
            adapter = HTTPProxyAdapter(
                adapter_context(),
                proxy_settings(proxy.url, targets=["tool-http"]),
            )
            handle = adapter.handles.tool_http
            assert isinstance(handle, ProxyHTTPClient)
            response = await handle.request("GET", "http://public.example/failure")
            assert response.status_code == 503
            assert response.text == "target unavailable"
            assert adapter.metrics.operations == 1
            assert adapter.metrics.failed_operations == 0
            await adapter.close()

    asyncio.run(scenario())


async def assert_safe_proxy_failure(proxy_url: str, backend_url: str) -> None:
    adapter = HTTPProxyAdapter(
        adapter_context(private_bypass_hosts=("control.example",)),
        proxy_settings(proxy_url, targets=["tool-http"]),
    )
    handle = adapter.handles.tool_http
    assert isinstance(handle, ProxyHTTPClient)
    with pytest.raises(ProxyRequestError) as captured:
        await handle.request("GET", backend_url)
    rendered = f"{captured.value!s} {captured.value!r} {adapter!r} {adapter.metrics!r}"
    for forbidden in (PROXY_PASSWORD, BACKEND_SECRET, proxy_url):
        assert forbidden not in rendered
    assert adapter.metrics.failed_operations == 1
    assert adapter.metrics.last_error_code == "request_failed"
    await adapter.close()


async def assert_safe_model_proxy_failure(proxy_url: str, backend_url: str) -> None:
    adapter = HTTPProxyAdapter(
        adapter_context(private_bypass_hosts=("control.example",)),
        proxy_settings(proxy_url, targets=["llm-gateway"]),
    )
    context = SimpleNamespace(
        model_policy=SimpleNamespace(model="worker-model"),
        runtime_settings=SimpleNamespace(
            llm_gateway_url=backend_url,
            llm_gateway_token=SimpleSecret(GATEWAY_TOKEN),
            request_timeout_seconds=2,
        ),
        adapter_handles=adapter.handles,
    )
    model = gateway_model(context)
    request = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text="Return ok")])],
        config=types.GenerateContentConfig(max_output_tokens=32),
    )
    with pytest.raises(GatewayModelError) as captured:
        async for _response in model.generate_content_async(request):
            pass
    rendered = f"{captured.value!s} {captured.value!r} {adapter!r} {adapter.metrics!r}"
    for forbidden in (PROXY_PASSWORD, BACKEND_SECRET, proxy_url, GATEWAY_TOKEN):
        assert forbidden not in rendered
    # The SDK owns the bounded 1 + 3 attempt policy. Every failure remains on
    # this allocation's proxy route; there is no direct fallback.
    assert adapter.metrics.failed_operations == 4
    await model.close()
    await adapter.close()


def adapter_context(
    *,
    private_bypass_hosts: tuple[str, ...] = (
        "127.0.0.1",
        "control.example",
        "localhost",
    ),
) -> RuntimeAdapterBuildContext:
    return RuntimeAdapterBuildContext(
        allocation_id="allocation-proxy",
        run_id="run-proxy",
        stage_execution_id="stage-proxy",
        logical_agent_name="worker",
        request_timeout_seconds=3,
        runtime_config_refs=("proxy@1",),
        runtime_config_digests=("sha256:" + "a" * 64,),
        run_labels=("caido",),
        agent_labels=(),
        run_metadata_labels=MappingProxyType({}),
        runtime_adapter_refs=("http-proxy@1",),
        private_bypass_hosts=private_bypass_hosts,
    )


def proxy_settings(
    proxy_url: str,
    *,
    targets: list[str],
    ca_bundle: str | None = None,
) -> HTTPProxySettingsV2:
    return HTTPProxySettingsV2(
        adapter="http-proxy@1",
        proxyUrl=proxy_url,
        basicAuth={"username": "proxy-user", "password": PROXY_PASSWORD},
        caBundlePem=ca_bundle,
        targets=targets,
    )


@dataclass(frozen=True)
class CapturedRequest:
    target: str
    headers: Mapping[str, str]
    body: bytes


@dataclass
class FakeProxy:
    url: str
    requests: list[CapturedRequest]


ProxyResponse = Callable[[CapturedRequest], tuple[int, bytes] | None]


@asynccontextmanager
async def fake_proxy(
    responder: ProxyResponse,
    *,
    tls: ssl.SSLContext | None = None,
    host: str = "127.0.0.1",
) -> AsyncIterator[FakeProxy]:
    requests: list[CapturedRequest] = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            header_bytes = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5)
            lines = header_bytes.decode("iso-8859-1").split("\r\n")
            method, target, _version = lines[0].split(" ", 2)
            headers = {
                name.strip().lower(): value.strip()
                for line in lines[1:]
                if line and (name_value := line.split(":", 1))
                for name, value in [name_value]
            }
            length = int(headers.get("content-length", "0"))
            body = await reader.readexactly(length) if length else b""
            request = CapturedRequest(target=target, headers=MappingProxyType(headers), body=body)
            requests.append(request)
            selected = responder(request)
            if selected is None:
                return
            status, response_body = selected
            reason = "OK" if status == 200 else "Proxy Authentication Required"
            writer.write(
                (
                    f"HTTP/1.1 {status} {reason}\r\n"
                    f"Content-Length: {len(response_body)}\r\n"
                    "Content-Type: application/json\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("ascii")
                + response_body
            )
            await writer.drain()
            del method
        except (asyncio.IncompleteReadError, ConnectionError, TimeoutError):
            return
        finally:
            writer.close()
            with suppress(ConnectionError, ssl.SSLError):
                await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0, ssl=tls)
    port = server.sockets[0].getsockname()[1]
    scheme = "https" if tls is not None else "http"
    value = FakeProxy(url=f"{scheme}://{host}:{port}", requests=requests)
    try:
        yield value
    finally:
        server.close()
        await server.wait_closed()


def openai_response(_request: CapturedRequest) -> tuple[int, bytes]:
    return 200, json.dumps(
        {
            "id": "chatcmpl-proxy",
            "object": "chat.completion",
            "created": 1,
            "model": "worker-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        separators=(",", ":"),
    ).encode()


def text_response(_request: CapturedRequest) -> tuple[int, bytes]:
    return 200, b"proxied"


def disconnect_response(_request: CapturedRequest) -> None:
    return None


def auth_failure_response(_request: CapturedRequest) -> tuple[int, bytes]:
    return 407, BACKEND_SECRET.encode()


def issue_local_certificate(tmp_path: Path) -> tuple[Path, Path]:
    certificate = tmp_path / "proxy-cert.pem"
    key = tmp_path / "proxy-key.pem"
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
            "-addext",
            "subjectAltName=DNS:localhost",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
        timeout=10,
    )
    return certificate, key


class SimpleSecret:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


class DescriptorToolset:
    ref = "descriptor@1"
    exported_tools = frozenset({"local"})

    def __init__(self, channels: Mapping[str, frozenset[str]]) -> None:
        self.infrastructure_channels = channels

    async def probe(self) -> frozenset[str]:
        return self.exported_tools

    async def create_selected(self, **_values: Any) -> Mapping[str, Any]:
        return {}


class DescriptorRuntime:
    ref = "runtime@1"


class DescriptorSandbox:
    ref = "sandbox@1"


def registry_with_toolset(toolset: DescriptorToolset) -> FactoryRegistry:
    return FactoryRegistry(
        worker_runtimes={"runtime@1": DescriptorRuntime()},
        toolsets={"descriptor@1": toolset},
        sandbox_profiles={"sandbox@1": DescriptorSandbox()},
    )
