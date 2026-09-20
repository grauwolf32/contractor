"""Sequenced Runtime Agent registration and heartbeat client."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import ssl
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

from contractor_runtime.contracts import (
    AgentRegistrationResponse,
    HeartbeatResponse,
    PrivateProtocolDecodeError,
    ReconciliationAction,
    decode_private,
)
from contractor_runtime.lease import LeaseWatchdog
from contractor_runtime.mtls import verify_control_plane_peer
from contractor_runtime.settings import Settings
from contractor_runtime.state import ProcessState, RuntimeState

logger = logging.getLogger(__name__)

MAX_CONTROL_RESPONSE_BYTES = 1 << 20
MAX_CONTROL_HEADERS = 64
MAX_CONTROL_HEADER_BYTES = 64 * 1024
MAX_CONTROL_LINE_BYTES = 8192
CONNECTION_CLOSE_TIMEOUT_SECONDS = 0.5

_HTTP_FIELD_NAME = re.compile(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+")


class ControlTransport(Protocol):
    async def post_json(
        self, path: str, payload: Mapping[str, Any]
    ) -> Mapping[str, Any] | bytes: ...


class ReconciliationHandler(Protocol):
    async def reconcile_drain(self, allocation_id: str, shutdown_grace_seconds: float) -> None: ...

    async def confirm_release(self, allocation_id: str | None) -> None: ...


class ControlClientError(Exception):
    """A bounded, non-secret private protocol error."""


class ControlHTTPError(ControlClientError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"Control Plane returned HTTP status {status_code}")
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class ControlTiming:
    heartbeat_interval_seconds: float
    confirmed_lease_seconds: float


class MTLSJSONTransport:
    """Minimal HTTPS/1.1 transport with a pre-request URI SAN role check.

    A generic HTTP client verifies chain and hostname but does not expose a
    portable hook between TLS completion and the first request byte. This
    transport intentionally opens a fresh bounded connection per control call,
    verifies the Control Plane role, and only then writes the HTTP request.
    """

    def __init__(self, base_url: str, context: ssl.SSLContext, timeout_seconds: float) -> None:
        if not _is_visible_ascii(base_url):
            raise ValueError("Control Plane base URL must contain only visible ASCII characters")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or "?" in base_url
            or "#" in base_url
        ):
            raise ValueError(
                "Control Plane base URL must be HTTPS without userinfo, query or fragment"
            )
        self._host = parsed.hostname
        self._port = parsed.port or 443
        self._base_path = parsed.path.rstrip("/")
        self._context = context
        self._timeout = timeout_seconds

    async def post_json(self, path: str, payload: Mapping[str, Any]) -> bytes:
        request = self._encode_request(path, payload)
        writer: asyncio.StreamWriter | None = None
        try:
            # The timeout covers the entire exchange, including TLS setup.
            async with asyncio.timeout(self._timeout):
                reader, writer = await asyncio.open_connection(
                    self._host,
                    self._port,
                    ssl=self._context,
                    server_hostname=self._host,
                    limit=MAX_CONTROL_HEADER_BYTES,
                )
                ssl_object = writer.get_extra_info("ssl_object")
                if not isinstance(ssl_object, ssl.SSLObject | ssl.SSLSocket):
                    raise ssl.SSLCertVerificationError("private connection has no TLS peer")
                verify_control_plane_peer(ssl_object)
                writer.write(request)
                await writer.drain()
                status, headers = await _read_response_head(reader)
                response_body = await _read_response_body(reader, headers)
        finally:
            if writer is not None:
                await _close_connection(writer, self._timeout)
        if not 200 <= status < 300:
            raise ControlHTTPError(status)
        return response_body

    def _encode_request(self, path: str, payload: Mapping[str, Any]) -> bytes:
        if not path.startswith("/") or "?" in path or "#" in path or not _is_visible_ascii(path):
            raise ValueError("private control path must be an absolute ASCII path without query")
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        target = self._base_path + path
        host_name = f"[{self._host}]" if ":" in self._host else self._host
        host = host_name if self._port == 443 else f"{host_name}:{self._port}"
        return (
            f"POST {target} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Accept: application/json\r\n"
            f"X-Request-ID: request_{uuid.uuid4().hex}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii") + body


class ControlClient:
    """Register the process and exchange one sequenced heartbeat at a time."""

    def __init__(
        self,
        settings: Settings,
        state: RuntimeState,
        transport: ControlTransport,
        *,
        watchdog: LeaseWatchdog | None = None,
        reconciliation: ReconciliationHandler | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self._settings = settings
        self._state = state
        self._transport = transport
        self._watchdog = watchdog
        self._reconciliation = reconciliation
        self._sleep = sleep
        self._jitter = jitter
        self._heartbeat_lock = asyncio.Lock()
        self._sequence = 0
        self._echoed_ack = 0
        self._timing = ControlTiming(
            heartbeat_interval_seconds=settings.heartbeat_interval_seconds,
            confirmed_lease_seconds=settings.confirmed_lease_seconds,
        )

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def echoed_ack(self) -> int:
        return self._echoed_ack

    @property
    def timing(self) -> ControlTiming:
        return self._timing

    async def register(self) -> AgentRegistrationResponse:
        registration = await self._state.registration(self._settings)
        raw = await self._transport.post_json(
            "/private/v1/agents/register",
            registration.model_dump(mode="json", by_alias=True, exclude_none=True),
        )
        try:
            response = decode_private(AgentRegistrationResponse, _response_json(raw))
        except PrivateProtocolDecodeError:
            raise ControlClientError("invalid registration response") from None
        self._timing = ControlTiming(
            heartbeat_interval_seconds=float(response.heartbeat_interval_seconds),
            confirmed_lease_seconds=float(response.confirmed_lease_seconds),
        )
        snapshot = await self._state.snapshot()
        if snapshot.process_state is ProcessState.STARTING:
            await self._state.mark_registered()
        elif snapshot.process_state is ProcessState.FENCED and snapshot.allocation_id is None:
            # Registration has established that the Control Plane owns no
            # allocation for this identity, so an idle lease loss may clear.
            await self._state.confirm_release(None)
        if self._watchdog is not None:
            await self._watchdog.arm(self._timing.confirmed_lease_seconds)
        return response

    async def register_until_stopped(self, stop: asyncio.Event) -> bool:
        failures = 0
        while not stop.is_set():
            try:
                await self.register()
                return True
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failures += 1
                logger.warning("Control Plane registration failed (%s)", type(error).__name__)
                await self._sleep_until_stopped(stop, self._backoff(failures))
        return False

    async def heartbeat_once(self) -> HeartbeatResponse:
        # Keep the request, acknowledgement and reconciliation in sequence even
        # when callers invoke heartbeat_once concurrently.
        async with self._heartbeat_lock:
            return await self._exchange_heartbeat()

    async def _exchange_heartbeat(self) -> HeartbeatResponse:
        self._sequence += 1
        request = await self._state.heartbeat(self._sequence, self._echoed_ack)
        path = f"/private/v1/agents/{quote(self._state.instance_id, safe='')}/heartbeat"
        raw = await self._transport.post_json(
            path,
            request.model_dump(mode="json", by_alias=True, exclude_none=True),
        )
        try:
            response = decode_private(HeartbeatResponse, _response_json(raw))
        except PrivateProtocolDecodeError:
            raise ControlClientError("invalid heartbeat response") from None
        if response.ack_seq != request.heartbeat_seq:
            raise ControlClientError("heartbeat response acknowledged the wrong sequence")
        self._echoed_ack = response.ack_seq
        if self._watchdog is not None:
            await self._watchdog.acknowledge(response.ack_seq, self._timing.confirmed_lease_seconds)
        await self._apply_reconciliation(response)
        return response

    async def _apply_reconciliation(self, response: HeartbeatResponse) -> None:
        if response.action is ReconciliationAction.DRAIN:
            if self._reconciliation is None or response.allocation_id is None:
                raise ControlClientError("heartbeat drain action cannot be applied")
            await self._reconciliation.reconcile_drain(
                response.allocation_id, self._settings.shutdown_grace_seconds
            )
        elif response.action is ReconciliationAction.RELEASE:
            if self._reconciliation is None:
                raise ControlClientError("heartbeat release action cannot be applied")
            await self._reconciliation.confirm_release(response.allocation_id)
        elif response.action is ReconciliationAction.REREGISTER:
            await self.register()

    async def run_heartbeats(self, stop: asyncio.Event) -> None:
        failures = 0
        while not stop.is_set():
            try:
                await self.heartbeat_once()
                failures = 0
                delay = self._timing.heartbeat_interval_seconds
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failures += 1
                logger.warning("Control Plane heartbeat failed (%s)", type(error).__name__)
                delay = self._backoff(failures)
            await self._sleep_until_stopped(stop, delay)

    async def _sleep_until_stopped(self, stop: asyncio.Event, delay: float) -> None:
        if stop.is_set():
            return
        sleeping = asyncio.ensure_future(self._sleep(delay))
        stopped = asyncio.create_task(stop.wait())
        try:
            done, _ = await asyncio.wait({sleeping, stopped}, return_when=asyncio.FIRST_COMPLETED)
            if sleeping in done:
                await sleeping
        finally:
            for task in (sleeping, stopped):
                if not task.done():
                    task.cancel()
            await asyncio.gather(sleeping, stopped, return_exceptions=True)

    def _backoff(self, failures: int) -> float:
        ceiling = min(5.0, self._timing.heartbeat_interval_seconds)
        base = min(ceiling, 0.5 * (2 ** min(failures - 1, 10)))
        return min(ceiling, base * (0.75 + 0.5 * self._jitter()))


def _response_json(raw: Mapping[str, Any] | bytes) -> bytes:
    if isinstance(raw, bytes):
        return raw
    try:
        return json.dumps(raw, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError):
        raise ControlClientError("Control Plane returned invalid JSON") from None


def _is_visible_ascii(value: str) -> bool:
    return all(" " < character < "\x7f" for character in value)


async def _close_connection(writer: asyncio.StreamWriter, timeout_seconds: float) -> None:
    writer.close()
    current_task = asyncio.current_task()
    if current_task is not None and current_task.cancelling():
        writer.transport.abort()
        return
    try:
        await asyncio.wait_for(
            writer.wait_closed(), timeout=min(CONNECTION_CLOSE_TIMEOUT_SECONDS, timeout_seconds)
        )
    except asyncio.CancelledError:
        writer.transport.abort()
        raise
    except Exception:
        writer.transport.abort()


async def _read_http_line(reader: asyncio.StreamReader, *, limit: int, error: str) -> bytes:
    try:
        line = await reader.readline()
    except ValueError:
        # StreamReader raises ValueError when its own line limit is exceeded.
        raise ControlClientError(error) from None
    if len(line) > limit or not line.endswith(b"\r\n"):
        raise ControlClientError(error)
    return line


async def _read_exactly(reader: asyncio.StreamReader, length: int) -> bytes:
    try:
        return await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        raise ControlClientError("incomplete HTTP response body") from None


async def _read_response_head(reader: asyncio.StreamReader) -> tuple[int, dict[str, str]]:
    status_line = await _read_http_line(
        reader, limit=MAX_CONTROL_LINE_BYTES, error="invalid HTTP status line"
    )
    parts = status_line[:-2].split(b" ", 2)
    if len(parts) != 3 or parts[0] not in {b"HTTP/1.0", b"HTTP/1.1"}:
        raise ControlClientError("invalid HTTP status line")
    if len(parts[1]) != 3 or not parts[1].isdigit():
        raise ControlClientError("invalid HTTP status code")
    status = int(parts[1])
    if not 100 <= status <= 599:
        raise ControlClientError("invalid HTTP status code")
    headers: dict[str, str] = {}
    total = len(status_line)
    while True:
        line = await _read_http_line(
            reader,
            limit=MAX_CONTROL_HEADER_BYTES - total,
            error="invalid or oversized HTTP response headers",
        )
        total += len(line)
        if line == b"\r\n":
            return status, headers
        if len(headers) >= MAX_CONTROL_HEADERS:
            raise ControlClientError("too many HTTP response headers")
        name, separator, value = line[:-2].partition(b":")
        if not separator or not _HTTP_FIELD_NAME.fullmatch(name):
            raise ControlClientError("invalid HTTP response header")
        key = name.decode("ascii").lower()
        if key in headers:
            raise ControlClientError("invalid or duplicate HTTP response header")
        if any((byte < 32 and byte != 9) or byte == 127 for byte in value):
            raise ControlClientError("invalid HTTP response header value")
        headers[key] = value.decode("latin-1").strip(" \t")


async def _read_response_body(reader: asyncio.StreamReader, headers: Mapping[str, str]) -> bytes:
    if "transfer-encoding" in headers:
        if "content-length" in headers:
            raise ControlClientError("ambiguous HTTP response framing")
        transfer_encoding = headers["transfer-encoding"].lower()
        if transfer_encoding != "chunked":
            raise ControlClientError("unsupported HTTP transfer encoding")
        return await _read_chunked_body(reader)
    if "content-length" in headers:
        raw_length = headers["content-length"]
        if not raw_length.isascii() or not raw_length.isdecimal():
            raise ControlClientError("invalid HTTP content length")
        try:
            length = int(raw_length)
        except ValueError:
            raise ControlClientError("invalid HTTP content length") from None
        if length > MAX_CONTROL_RESPONSE_BYTES:
            raise ControlClientError("Control Plane response is too large")
        return await _read_exactly(reader, length)
    result = bytearray()
    while True:
        chunk = await reader.read(min(64 * 1024, MAX_CONTROL_RESPONSE_BYTES + 1 - len(result)))
        if not chunk:
            return bytes(result)
        result.extend(chunk)
        if len(result) > MAX_CONTROL_RESPONSE_BYTES:
            raise ControlClientError("Control Plane response is too large")


async def _read_chunked_body(reader: asyncio.StreamReader) -> bytes:
    result = bytearray()
    while True:
        size_line = await _read_http_line(
            reader, limit=MAX_CONTROL_LINE_BYTES, error="invalid chunk size"
        )
        raw_size = size_line[:-2].partition(b";")[0].rstrip(b" \t")
        if not re.fullmatch(rb"[0-9a-fA-F]+", raw_size):
            raise ControlClientError("invalid chunk size")
        size = int(raw_size, 16)
        if len(result) + size > MAX_CONTROL_RESPONSE_BYTES:
            raise ControlClientError("Control Plane response is too large")
        if size == 0:
            if await _read_exactly(reader, 2) != b"\r\n":
                raise ControlClientError("chunked trailers are not supported")
            return bytes(result)
        result.extend(await _read_exactly(reader, size))
        if await _read_exactly(reader, 2) != b"\r\n":
            raise ControlClientError("invalid chunk delimiter")
