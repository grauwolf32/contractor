"""One host-syntax rule for http_request, Caido and every scanner entry point."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from test_caido_read_tools import FakeArtifactClient as CaidoArtifactClient
from test_caido_read_tools import close_tools as close_caido_tools
from test_caido_read_tools import create_tools as create_caido_tools
from test_http_toolset import FakeArtifactClient, make_tools, proxy_runtime_settings
from test_http_toolset import close_tools as close_http_tools
from test_http_toolset import create_tools as create_http_tools
from test_scan_toolset import install_echo_scanners
from test_scan_toolset import make_tools as make_scan_tools

import contractor_runtime.toolsets.scan.tools as scan
from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.http_proxy import ProxyHTTPClient
from contractor_runtime.toolsets.caido.tools import CaidoToolError, CaidoToolsetFactory
from contractor_runtime.toolsets.common.input_errors import ToolInputError
from contractor_runtime.toolsets.common.target_policy import (
    TargetDenied,
    TargetPolicy,
    TargetPolicyConfig,
    TargetUnresolved,
    validate_host_syntax,
)
from contractor_runtime.toolsets.http.tools import HTTPToolError, HTTPToolsetFactory
from contractor_runtime.toolsets.scan.ffuf import validate_ffuf_url
from contractor_runtime.toolsets.scan.http_request import parse_http_request
from contractor_runtime.toolsets.scan.katana import canonical_url

# Each case is (bare host, host as written in a URL authority).
ACCEPTED_HOSTS = [
    ("target.example", "target.example"),
    ("juice_shop", "juice_shop"),
    ("web_app.example", "web_app.example"),
    ("ab--cd.example", "ab--cd.example"),
    ("xn--bcher-kva.example", "xn--bcher-kva.example"),
    ("bücher.example", "bücher.example"),
    ("10.0.0.5", "10.0.0.5"),
    ("2001:db8::5", "[2001:db8::5]"),
    ("a" * 63 + ".example", "a" * 63 + ".example"),
    ("trailing.example.", "trailing.example."),
]
REJECTED_HOSTS = [
    ("169.254.169.%32%35%34", "169.254.169.%32%35%34"),
    ("%6c%6fcalhost", "%6c%6fcalhost"),
    ("a!b$c.example", "a!b$c.example"),
    ("a" * 64 + ".example", "a" * 64 + ".example"),
    (".".join(["a" * 63] * 4), ".".join(["a" * 63] * 4)),
    ("-option.example", "-option.example"),
    ("xn--zz.example", "xn--zz.example"),
    ("bad..example", "bad..example"),
    ("fe80::1%eth0", "[fe80::1%25eth0]"),
]
CASES = [(*case, True) for case in ACCEPTED_HOSTS] + [(*case, False) for case in REJECTED_HOSTS]


def _sqlmap_request(authority: str) -> object:
    document = {
        "schemaVersion": 1,
        "method": "GET",
        "url": f"http://{authority}:8080/items?id=7",
        "headers": [],
        "body": "",
        "testParameters": ["id"],
    }
    return parse_http_request(json.dumps(document).encode())


def _ffuf_url(authority: str) -> None:
    url = f"http://{authority}:8080/FUZZ"
    validate_ffuf_url(url)
    scan._url(url)


# Scanner argument validators, by entry point. ffuf and sqlmap request
# artifacts carry wire bytes and accept only ASCII URLs (A-labels).
SCANNER_ENTRY_POINTS: dict[str, Callable[[str, str], object]] = {
    "naabu-host": lambda host, _authority: scan._host(host),
    "nuclei-url": lambda _host, authority: scan._url(f"http://{authority}:8080/path"),
    "katana-url": lambda _host, authority: canonical_url(f"http://{authority}:8080/path"),
    "ffuf-url": lambda _host, authority: _ffuf_url(authority),
    "sqlmap-request": lambda _host, authority: _sqlmap_request(authority),
}
ASCII_ONLY_ENTRY_POINTS = frozenset({"ffuf-url", "sqlmap-request"})


@pytest.mark.parametrize(("host", "authority", "accepted"), CASES)
def test_one_host_rule_for_http_caido_and_scanners(
    host: str, authority: str, accepted: bool
) -> None:
    del authority
    if accepted:
        validate_host_syntax(host)
    else:
        with pytest.raises(ValueError) as failure:
            validate_host_syntax(host)
        assert host not in str(failure.value)
        # The policy itself never classifies such text as a name.
        with pytest.raises(TargetDenied):
            TargetPolicy().check_host(host, 80)


@pytest.mark.parametrize("entry_point", sorted(SCANNER_ENTRY_POINTS))
@pytest.mark.parametrize(("host", "authority", "accepted"), CASES)
def test_scanner_entry_points_apply_the_shared_host_rule(
    entry_point: str, host: str, authority: str, accepted: bool
) -> None:
    validate = SCANNER_ENTRY_POINTS[entry_point]
    if accepted and (host.isascii() or entry_point not in ASCII_ONLY_ENTRY_POINTS):
        validate(host, authority)
        return
    with pytest.raises(ToolInputError) as failure:
        validate(host, authority)
    assert failure.value.retryable is False
    assert host not in str(failure.value)


@pytest.mark.parametrize("route", ["direct", "proxy"])
@pytest.mark.parametrize(("host", "authority"), REJECTED_HOSTS)
def test_http_request_shares_the_scanner_host_rule(
    tmp_path: Path, route: str, host: str, authority: str
) -> None:
    del host
    sent: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        return httpx.Response(200, request=request)

    async def scenario() -> None:
        proxy_client: httpx.AsyncClient | None = None
        if route == "direct":
            tools, _ = await create_http_tools(tmp_path, handler)
        else:
            proxy_client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler), trust_env=False
            )
            tools = await make_tools(
                HTTPToolsetFactory(lambda _allocation, _settings: FakeArtifactClient()),
                tmp_path,
                settings=proxy_runtime_settings(),
                adapter_handles=AdapterHandles(tool_http=ProxyHTTPClient(proxy_client)),
            )
        try:
            with pytest.raises(HTTPToolError) as failure:
                await tools["http_request"](f"http://{authority}:8080/")
            # Invalid input, never a resolver or proxy failure that retries.
            assert failure.value.code == "http_request_invalid"
            assert failure.value.retryable is False
        finally:
            await close_http_tools(tools)
            if proxy_client is not None:
                await proxy_client.aclose()
        assert sent == []

    asyncio.run(scenario())


@pytest.mark.parametrize(("host", "authority", "accepted"), CASES)
def test_caido_raw_replay_applies_the_shared_host_rule(
    tmp_path: Path, host: str, authority: str, accepted: bool
) -> None:
    del authority
    operations: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        operation = json.loads(request.content)["operationName"]
        operations.append(operation)
        if operation == "CreateReplaySession":
            data = {
                "createReplaySession": {
                    "session": {"id": "session", "name": "replay", "activeEntry": None}
                }
            }
        elif operation == "StartReplayTask":
            data = {
                "startReplayTask": {
                    "error": None,
                    "task": {"id": "task", "replayEntry": {"id": "entry"}},
                }
            }
        else:
            raise AssertionError(operation)
        return httpx.Response(200, json={"data": data}, request=request)

    async def unresolved(_host: str, _port: int) -> tuple[()]:
        raise TargetUnresolved

    async def scenario() -> None:
        artifacts = CaidoArtifactClient()
        tools, _state, handle = await create_caido_tools(
            tmp_path,
            handler,
            artifacts,
            selected={"caido_replay"},
            factory=CaidoToolsetFactory(
                lambda _allocation, _settings: artifacts,
                target_policy=TargetPolicyConfig(resolver=unresolved),
            ),
        )
        try:
            replay = tools["caido_replay"]
            raw_request = "GET / HTTP/1.1\r\nHost: target\r\n\r\n"
            if accepted:
                result = await replay(raw_request=raw_request, host=host, port=8080, wait=False)
                assert result["status"] == "started"
                assert operations == ["CreateReplaySession", "StartReplayTask"]
            else:
                with pytest.raises(CaidoToolError) as failure:
                    await replay(raw_request=raw_request, host=host, port=8080, wait=False)
                assert failure.value.code == "caido_request_invalid"
                assert failure.value.retryable is False
                assert operations == []
        finally:
            await close_caido_tools(tools)
            await handle.close()

    asyncio.run(scenario())


def test_caido_connection_hosts_accept_only_bracketed_ipv6_literals(tmp_path: Path) -> None:
    operations: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        operation = json.loads(request.content)["operationName"]
        operations.append(operation)
        if operation == "CreateReplaySession":
            data = {
                "createReplaySession": {
                    "session": {"id": "session", "name": "replay", "activeEntry": None}
                }
            }
        else:
            data = {
                "startReplayTask": {
                    "error": None,
                    "task": {"id": "task", "replayEntry": {"id": "entry"}},
                }
            }
        return httpx.Response(200, json={"data": data}, request=request)

    async def scenario() -> None:
        artifacts = CaidoArtifactClient()
        tools, _state, handle = await create_caido_tools(
            tmp_path, handler, artifacts, selected={"caido_replay"}
        )
        raw_request = "GET / HTTP/1.1\r\nHost: x\r\n\r\n"
        try:
            for host in ("[target.example]", "[10.0.0.5]", "[2001:db8::5", "[fe80::1%eth0]"):
                with pytest.raises(CaidoToolError) as failure:
                    await tools["caido_replay"](raw_request=raw_request, host=host, wait=False)
                assert failure.value.code == "caido_request_invalid"
            assert operations == []
            started = await tools["caido_replay"](
                raw_request=raw_request, host="[2001:db8::5]", wait=False
            )
            assert started["status"] == "started"
        finally:
            await close_caido_tools(tools)
            await handle.close()

    asyncio.run(scenario())


def test_scanners_launch_docker_compose_service_names(tmp_path: Path, monkeypatch) -> None:
    install_echo_scanners(tmp_path, monkeypatch)

    async def scenario() -> None:
        tools, _state = await make_scan_tools(
            tmp_path, selected=["scan_naabu", "scan_nuclei", "scan_sqlmap"]
        )
        naabu = await tools["scan_naabu"]("juice_shop", ports="3000")
        assert naabu["status"] == "completed", naabu
        arguments = naabu["results"][0]["args"]
        assert arguments[arguments.index("-host") + 1] == "juice_shop"
        nuclei = await tools["scan_nuclei"]("http://juice_shop:3000/")
        assert nuclei["status"] == "completed", nuclei
        assert nuclei["results"][0]["targetFile"] == "http://juice_shop:3000/\n"
        sqlmap = await tools["scan_sqlmap"]("http://web_app:8080/?id=1", parameter="id")
        assert sqlmap["status"] == "completed", sqlmap
        assert "--url=http://web_app:8080/?id=1" in json.loads(sqlmap["stdout"])["args"]
        assert canonical_url("http://Juice_Shop:3000") == "http://juice_shop:3000/"
        assert canonical_url("https://BÜCHER.example/a") == "https://xn--bcher-kva.example/a"
        for tool in tools.values():
            await tool.close()

    asyncio.run(scenario())
