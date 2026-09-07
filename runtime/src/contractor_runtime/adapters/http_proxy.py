"""Allocation-owned fail-closed HTTP proxy clients and subprocess launcher."""

from __future__ import annotations

import asyncio
import os
import ssl
import subprocess
import tempfile
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from contractor_runtime.adapters.host import (
    AdapterFactoryError,
    AdapterHandles,
    AdapterSettings,
    RuntimeAdapterBuildContext,
    RuntimeAdapterMetricsState,
)
from contractor_runtime.contracts import HTTPProxySettings, RuntimeAdapterRef

MAX_SUBPROCESS_ARGUMENTS = 128
MAX_SUBPROCESS_ARGUMENT_BYTES = 4096
MAX_SUBPROCESS_INPUT_BYTES = 4 * 1024 * 1024
MAX_SUBPROCESS_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_CHILD_ENVIRONMENT_BYTES = 64 * 1024
MAX_CHILD_ENVIRONMENT_VALUE_BYTES = 32 * 1024
MAX_COMBINED_CA_BYTES = 2 * 1024 * 1024
_ALLOWED_CHILD_ENV = frozenset(
    {
        "CI",
        "LANG",
        "LC_ALL",
        "NO_COLOR",
        "NO_UPDATE_NOTIFIER",
        "PATH",
        "PYTHONIOENCODING",
    }
)


class ProxyRequestError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("allocation proxy request failed")


class ProxySubprocessError(RuntimeError):
    code = "proxy_subprocess_failed"
    retryable = True

    def __init__(self) -> None:
        super().__init__("proxied subprocess failed")


class ProxyCloseError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("allocation proxy close failed")


class _ObservedProxyTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        transport: httpx.AsyncBaseTransport,
        metrics: RuntimeAdapterMetricsState,
    ) -> None:
        self._transport: httpx.AsyncBaseTransport | None = transport
        self._metrics = metrics

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        transport = self._transport
        if transport is None:
            self._metrics.record_operation(succeeded=False, error_code="request_failed")
            raise ProxyRequestError from None
        try:
            response = await transport.handle_async_request(request)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._metrics.record_operation(succeeded=False, error_code="request_failed")
            raise ProxyRequestError from None
        # A target 4xx/5xx is application data for model-facing HTTP tools.
        # 407 is the only response status that unambiguously belongs to the
        # configured forward-proxy hop; tunnel/routing failures surface as
        # transport exceptions above.
        if response.status_code == 407:
            self._metrics.record_operation(succeeded=False, error_code="request_failed")
            await response.aclose()
            raise ProxyRequestError from None
        self._metrics.record_operation(succeeded=True)
        return response

    async def aclose(self) -> None:
        transport = self._transport
        self._transport = None
        if transport is not None:
            await transport.aclose()


class ProxyHTTPClient:
    """Narrow allocation handle; callers cannot change proxy routing."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        forbidden_hosts: Sequence[str] = (),
        metrics: RuntimeAdapterMetricsState | None = None,
    ) -> None:
        self._client: httpx.AsyncClient | None = client
        self._forbidden_hosts = frozenset(forbidden_hosts)
        self._metrics = metrics

    @property
    def async_client(self) -> httpx.AsyncClient:
        client = self._client
        if client is None:
            raise ProxyRequestError
        return client

    async def request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        parsed = urlsplit(url)
        if parsed.hostname in self._forbidden_hosts or parsed.netloc in self._forbidden_hosts:
            if self._metrics is not None:
                self._metrics.record_operation(succeeded=False, error_code="request_failed")
            raise ProxyRequestError
        try:
            response = await self.async_client.request(method, url, **kwargs)
            if response.status_code == 407:
                await response.aclose()
                raise ProxyRequestError
            return response
        except asyncio.CancelledError:
            raise
        except ProxyRequestError:
            raise
        except Exception:
            raise ProxyRequestError from None

    async def stream_request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        """Send one routed request without buffering its response body."""

        parsed = urlsplit(url)
        if parsed.hostname in self._forbidden_hosts or parsed.netloc in self._forbidden_hosts:
            if self._metrics is not None:
                self._metrics.record_operation(succeeded=False, error_code="request_failed")
            raise ProxyRequestError
        try:
            client = self.async_client
            request = client.build_request(method, url, **kwargs)
            response = await client.send(request, stream=True, follow_redirects=False)
            if response.status_code == 407:
                await response.aclose()
                raise ProxyRequestError
            return response
        except asyncio.CancelledError:
            raise
        except ProxyRequestError:
            raise
        except Exception:
            raise ProxyRequestError from None

    def clear_cookies(self) -> None:
        """Erase allocation-session cookies retained by httpx."""

        self.async_client.cookies.clear()

    def detach(self) -> None:
        self._client = None
        self._forbidden_hosts = frozenset()
        self._metrics = None

    def __repr__(self) -> str:
        return f"ProxyHTTPClient(active={self._client is not None!r})"


class ProxySubprocessLauncher:
    """Synchronous bounded launcher intended to run in a Worker thread."""

    def __init__(
        self,
        *,
        proxy_url: str,
        basic_auth: tuple[str, str] | None,
        bearer_token: str | None,
        combined_ca_bundle: bytes | None,
        bypass_hosts: Sequence[str],
        timeout_seconds: float,
        metrics: RuntimeAdapterMetricsState,
    ) -> None:
        self._proxy_url = proxy_url
        self._basic_auth = basic_auth
        self._bearer_token = bearer_token
        self._combined_ca_bundle = combined_ca_bundle
        self._bypass_hosts = tuple(bypass_hosts)
        self._timeout_seconds = timeout_seconds
        self._metrics = metrics
        self._temporary_roots: set[Path] = set()
        self._lock = threading.Lock()
        self._closed = False

    def run(
        self,
        command: Sequence[str],
        *,
        input: bytes | None = None,
        cwd: Path | str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_output_bytes: int = MAX_SUBPROCESS_OUTPUT_BYTES,
    ) -> subprocess.CompletedProcess[bytes]:
        selected_command = _validate_command(command)
        if input is not None and not isinstance(input, bytes):
            raise ProxySubprocessError
        selected_input = input or b""
        if len(selected_input) > MAX_SUBPROCESS_INPUT_BYTES:
            raise ProxySubprocessError
        if not 1 <= max_output_bytes <= MAX_SUBPROCESS_OUTPUT_BYTES:
            raise ProxySubprocessError
        selected_timeout = min(
            self._timeout_seconds,
            self._timeout_seconds if timeout is None else max(0.001, timeout),
        )

        ca_root: Path | None = None
        with self._lock:
            if self._closed:
                raise ProxySubprocessError
            try:
                child_env, ca_root = self._child_environment(env)
            except Exception:
                self._metrics.record_operation(succeeded=False, error_code="request_failed")
                raise ProxySubprocessError from None

        try:
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                process = subprocess.Popen(
                    selected_command,
                    stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    cwd=cwd,
                    env=child_env,
                    close_fds=True,
                )
                try:
                    process.communicate(input=input, timeout=selected_timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise ProxySubprocessError from None
                stdout_size = stdout.tell()
                stderr_size = stderr.tell()
                if stdout_size + stderr_size > max_output_bytes:
                    raise ProxySubprocessError
                stdout.seek(0)
                stderr.seek(0)
                result = subprocess.CompletedProcess(
                    selected_command,
                    process.returncode,
                    stdout.read(max_output_bytes + 1),
                    stderr.read(max_output_bytes + 1),
                )
        except asyncio.CancelledError:
            raise
        except ProxySubprocessError:
            self._metrics.record_operation(succeeded=False, error_code="request_failed")
            raise
        except Exception:
            self._metrics.record_operation(succeeded=False, error_code="request_failed")
            raise ProxySubprocessError from None
        finally:
            if ca_root is not None and not self._remove_temporary_root(ca_root):
                self._metrics.record_operation(succeeded=False, error_code="request_failed")
                raise ProxySubprocessError from None

        succeeded = result.returncode == 0
        self._metrics.record_operation(
            succeeded=succeeded,
            error_code=None if succeeded else "request_failed",
        )
        return result

    def close(self) -> None:
        with self._lock:
            self._closed = True
            roots = tuple(self._temporary_roots)
            self._temporary_roots.clear()
            self._proxy_url = ""
            self._basic_auth = None
            self._bearer_token = None
            self._combined_ca_bundle = None
            self._bypass_hosts = ()
        for root in roots:
            if not _remove_private_root(root):
                raise ProxyCloseError from None

    @property
    def active_temporary_roots(self) -> tuple[Path, ...]:
        with self._lock:
            return tuple(sorted(self._temporary_roots))

    def _child_environment(
        self,
        source: Mapping[str, str] | None,
    ) -> tuple[dict[str, str], Path | None]:
        if self._bearer_token is not None:
            # Generic proxy environment variables cannot express bearer proxy
            # authentication. Fail closed instead of silently sending direct or
            # downgrading it to Basic authentication.
            raise ProxySubprocessError
        environment: dict[str, str] = {}
        total_environment_bytes = 0
        for name, value in (source or {}).items():
            if (
                name not in _ALLOWED_CHILD_ENV
                or not isinstance(value, str)
                or "\x00" in value
                or len(value.encode("utf-8")) > MAX_CHILD_ENVIRONMENT_VALUE_BYTES
            ):
                raise ProxySubprocessError
            environment[name] = value
            total_environment_bytes += len(name) + len(value.encode("utf-8"))
        if total_environment_bytes > MAX_CHILD_ENVIRONMENT_BYTES:
            raise ProxySubprocessError
        environment.setdefault("PATH", os.defpath)
        proxy_url = _proxy_environment_url(self._proxy_url, self._basic_auth)
        environment.update(
            {
                "HTTP_PROXY": proxy_url,
                "HTTPS_PROXY": proxy_url,
                "NO_PROXY": ",".join(self._bypass_hosts),
            }
        )
        if (
            any(
                len(value.encode("utf-8")) > MAX_CHILD_ENVIRONMENT_VALUE_BYTES
                for value in environment.values()
            )
            or sum(len(name) + len(value.encode("utf-8")) for name, value in environment.items())
            > MAX_CHILD_ENVIRONMENT_BYTES
        ):
            raise ProxySubprocessError
        ca_root: Path | None = None
        if self._combined_ca_bundle is not None:
            ca_root = Path(tempfile.mkdtemp(prefix="contractor-proxy-ca-"))
            os.chmod(ca_root, 0o700)
            bundle = ca_root / "ca-bundle.pem"
            try:
                with bundle.open("xb") as stream:
                    os.chmod(bundle, 0o600)
                    stream.write(self._combined_ca_bundle)
                self._temporary_roots.add(ca_root)
            except Exception:
                _remove_private_root(ca_root)
                raise
            environment["SSL_CERT_FILE"] = str(bundle)
        return environment, ca_root

    def _remove_temporary_root(self, root: Path) -> bool:
        removed = _remove_private_root(root)
        if removed:
            with self._lock:
                self._temporary_roots.discard(root)
        return removed

    def __repr__(self) -> str:
        with self._lock:
            return (
                "ProxySubprocessLauncher("
                f"active_roots={len(self._temporary_roots)}, closed={self._closed!r})"
            )


class HTTPProxyAdapterFactory:
    ref = "http-proxy@1"

    async def probe(self) -> bool:
        client: httpx.AsyncClient | None = None
        try:
            context = ssl.create_default_context()
            transport = httpx.AsyncHTTPTransport(verify=context, trust_env=False, retries=0)
            client = httpx.AsyncClient(transport=transport, trust_env=False)
        except Exception:
            return False
        finally:
            if client is not None:
                await client.aclose()
        return True

    async def create(
        self,
        context: RuntimeAdapterBuildContext,
        settings: AdapterSettings,
    ) -> HTTPProxyAdapter:
        if not isinstance(settings, HTTPProxySettings):
            raise AdapterFactoryError(retryable=False)
        try:
            return HTTPProxyAdapter(context, settings)
        except Exception:
            raise AdapterFactoryError(retryable=False) from None

    def __repr__(self) -> str:
        return "HTTPProxyAdapterFactory(ref='http-proxy@1')"


class HTTPProxyAdapter:
    ref: RuntimeAdapterRef = "http-proxy@1"

    def __init__(
        self,
        context: RuntimeAdapterBuildContext,
        settings: HTTPProxySettings,
    ) -> None:
        self.metrics = RuntimeAdapterMetricsState()
        basic_auth: tuple[str, str] | None = None
        if settings.basic_auth is not None:
            basic_auth = (
                settings.basic_auth.username.get_secret_value(),
                settings.basic_auth.password.get_secret_value(),
            )
        bearer_token = (
            settings.bearer_token.get_secret_value() if settings.bearer_token is not None else None
        )
        tls_context = ssl.create_default_context()
        if settings.ca_bundle_pem is not None:
            tls_context.load_verify_locations(cadata=settings.ca_bundle_pem)
        proxy = _httpx_proxy(settings.proxy_url, tls_context, basic_auth, bearer_token)
        timeout = httpx.Timeout(float(context.request_timeout_seconds))
        limits = httpx.Limits(max_connections=4, max_keepalive_connections=2)

        self._clients: list[ProxyHTTPClient] = []
        model_http = (
            self._new_http_client(proxy, tls_context, timeout, limits)
            if "llm-gateway" in settings.targets
            else None
        )
        tool_http = (
            self._new_http_client(
                proxy,
                tls_context,
                timeout,
                limits,
                forbidden_hosts=context.private_bypass_hosts,
            )
            if "tool-http" in settings.targets
            else None
        )

        combined_ca = (
            _combined_ca_bundle(tls_context, settings.ca_bundle_pem)
            if settings.ca_bundle_pem is not None and "tool-subprocess" in settings.targets
            else None
        )
        tool_subprocess = (
            ProxySubprocessLauncher(
                proxy_url=settings.proxy_url,
                basic_auth=basic_auth,
                bearer_token=bearer_token,
                combined_ca_bundle=combined_ca,
                bypass_hosts=context.private_bypass_hosts,
                timeout_seconds=float(context.request_timeout_seconds),
                metrics=self.metrics,
            )
            if "tool-subprocess" in settings.targets
            else None
        )
        self._launcher = tool_subprocess
        self.handles = AdapterHandles(
            model_http=model_http,
            tool_http=tool_http,
            tool_subprocess=tool_subprocess,
        )
        self._closed = False

    def _new_http_client(
        self,
        proxy: httpx.Proxy,
        tls_context: ssl.SSLContext,
        timeout: httpx.Timeout,
        limits: httpx.Limits,
        *,
        forbidden_hosts: Sequence[str] = (),
    ) -> ProxyHTTPClient:
        transport = httpx.AsyncHTTPTransport(
            verify=tls_context,
            trust_env=False,
            limits=limits,
            proxy=proxy,
            retries=0,
        )
        observed = _ObservedProxyTransport(transport, self.metrics)
        handle = ProxyHTTPClient(
            httpx.AsyncClient(
                transport=observed,
                trust_env=False,
                follow_redirects=False,
                timeout=timeout,
                limits=limits,
            ),
            forbidden_hosts=forbidden_hosts,
            metrics=self.metrics,
        )
        self._clients.append(handle)
        return handle

    async def flush(self) -> None:
        return

    async def close(self) -> None:
        if self._closed:
            return
        failed = False
        clients = tuple(self._clients)
        launcher = self._launcher
        try:
            for handle in clients:
                try:
                    await handle.async_client.aclose()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    failed = True
        finally:
            for handle in clients:
                handle.detach()
            self._clients.clear()
            if launcher is not None:
                try:
                    launcher.close()
                except Exception:
                    failed = True
            self._launcher = None
            self.handles = AdapterHandles()
            self._closed = True
        if failed:
            raise ProxyCloseError from None

    def __repr__(self) -> str:
        return f"HTTPProxyAdapter(ref={self.ref!r}, closed={self._closed!r})"


def _httpx_proxy(
    proxy_url: str,
    tls_context: ssl.SSLContext,
    basic_auth: tuple[str, str] | None,
    bearer_token: str | None,
) -> httpx.Proxy:
    headers: dict[str, str] = {}
    auth = basic_auth
    if bearer_token is not None:
        headers["Proxy-Authorization"] = f"Bearer {bearer_token}"
    return httpx.Proxy(
        proxy_url,
        ssl_context=(tls_context if urlsplit(proxy_url).scheme == "https" else None),
        auth=auth,
        headers=headers,
    )


def _proxy_environment_url(
    proxy_url: str,
    basic_auth: tuple[str, str] | None,
) -> str:
    if basic_auth is None:
        return proxy_url
    parsed = urlsplit(proxy_url)
    username, password = basic_auth
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    authority = f"{quote(username, safe='')}:{quote(password, safe='')}@{hostname}"
    if parsed.port is not None:
        authority += f":{parsed.port}"
    return urlunsplit((parsed.scheme, authority, parsed.path, "", ""))


def _validate_command(command: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(command)
    if not 1 <= len(selected) <= MAX_SUBPROCESS_ARGUMENTS:
        raise ProxySubprocessError
    if any(
        not isinstance(item, str)
        or not item
        or "\x00" in item
        or len(item.encode("utf-8")) > MAX_SUBPROCESS_ARGUMENT_BYTES
        for item in selected
    ):
        raise ProxySubprocessError
    executable = Path(selected[0])
    if not executable.is_absolute() or not executable.is_file():
        raise ProxySubprocessError
    return selected


def _combined_ca_bundle(context: ssl.SSLContext, extra_pem: str) -> bytes:
    roots = b"".join(
        ssl.DER_cert_to_PEM_cert(certificate).encode("ascii")
        for certificate in context.get_ca_certs(binary_form=True)
    )
    result = roots + extra_pem.encode("utf-8")
    if len(result) > MAX_COMBINED_CA_BYTES:
        raise ValueError("combined proxy CA bundle exceeds its bound")
    return result


def _remove_private_root(root: Path) -> bool:
    try:
        bundle = root / "ca-bundle.pem"
        if bundle.exists() or bundle.is_symlink():
            bundle.unlink()
        root.rmdir()
        return True
    except OSError:
        return False
