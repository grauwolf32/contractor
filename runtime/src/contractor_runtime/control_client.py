"""Sequenced Runtime Agent registration and heartbeat client."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import ssl
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

from pydantic import ValidationError

from contractor_runtime.contracts import (
    AgentRegistrationResponse,
    HeartbeatResponse,
    ReconciliationAction,
)
from contractor_runtime.lease import LeaseWatchdog
from contractor_runtime.mtls import verify_control_plane_peer
from contractor_runtime.settings import Settings
from contractor_runtime.state import ProcessState, RuntimeState

logger = logging.getLogger(__name__)

MAX_CONTROL_RESPONSE_BYTES = 1 << 20
MAX_CONTROL_HEADERS = 64


class ControlTransport(Protocol):
    async def post_json(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


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
        parsed = urlsplit(base_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError("Control Plane base URL must be HTTPS without query or fragment")
        self._host = parsed.hostname
        self._port = parsed.port or 443
        self._base_path = parsed.path.rstrip("/")
        self._context = context
        self._timeout = timeout_seconds

    async def post_json(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if not path.startswith("/") or "?" in path or "#" in path:
            raise ValueError("private control path must be absolute and contain no query")
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        reader: asyncio.StreamReader
        writer: asyncio.StreamWriter
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                self._host,
                self._port,
                ssl=self._context,
                server_hostname=self._host,
                limit=64 * 1024,
            ),
            timeout=self._timeout,
        )
        try:
            ssl_object = writer.get_extra_info("ssl_object")
            if not isinstance(ssl_object, ssl.SSLObject | ssl.SSLSocket):
                raise ssl.SSLCertVerificationError("private connection has no TLS peer")
            verify_control_plane_peer(ssl_object)
            target = (self._base_path + path) or "/"
            host_name = f"[{self._host}]" if ":" in self._host else self._host
            host = host_name if self._port == 443 else f"{host_name}:{self._port}"
            request = (
                f"POST {target} HTTP/1.1\r\n"
                f"Host: {host}\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Accept: application/json\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii") + body
            writer.write(request)
            await asyncio.wait_for(writer.drain(), timeout=self._timeout)
            status, headers = await asyncio.wait_for(
                _read_response_head(reader), timeout=self._timeout
            )
            response_body = await asyncio.wait_for(
                _read_response_body(reader, headers), timeout=self._timeout
            )
        finally:
            writer.close()
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                writer.transport.abort()
            else:
                try:
                    await asyncio.wait_for(writer.wait_closed(), timeout=min(0.5, self._timeout))
                except Exception:
                    writer.transport.abort()
        if not 200 <= status < 300:
            raise ControlHTTPError(status)
        try:
            decoded = json.loads(response_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ControlClientError("Control Plane returned invalid JSON") from error
        if not isinstance(decoded, dict):
            raise ControlClientError("Control Plane response must be a JSON object")
        return decoded


class ControlClient:
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
            response = AgentRegistrationResponse.model_validate_json(json.dumps(raw))
        except ValidationError as error:
            raise ControlClientError("invalid registration response") from error
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
                await self._sleep(self._backoff(failures))
        return False

    async def heartbeat_once(self) -> HeartbeatResponse:
        self._sequence += 1
        request = await self._state.heartbeat(self._sequence, self._echoed_ack)
        path = f"/private/v1/agents/{quote(self._state.instance_id, safe='')}/heartbeat"
        raw = await self._transport.post_json(
            path,
            request.model_dump(mode="json", by_alias=True, exclude_none=True),
        )
        try:
            response = HeartbeatResponse.model_validate_json(json.dumps(raw))
        except ValidationError as error:
            raise ControlClientError("invalid heartbeat response") from error
        if response.ack_seq != self._sequence:
            raise ControlClientError("heartbeat response acknowledged the wrong sequence")
        self._echoed_ack = response.ack_seq
        if self._watchdog is not None:
            await self._watchdog.acknowledge(response.ack_seq, self._timing.confirmed_lease_seconds)
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
        return response

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
            if not stop.is_set():
                await self._sleep(delay)

    def _backoff(self, failures: int) -> float:
        ceiling = min(5.0, self._timing.heartbeat_interval_seconds)
        base = min(ceiling, 0.5 * (2 ** min(failures - 1, 10)))
        return min(ceiling, base * (0.75 + 0.5 * self._jitter()))


async def _read_response_head(reader: asyncio.StreamReader) -> tuple[int, dict[str, str]]:
    status_line = await reader.readline()
    if not status_line.endswith(b"\r\n") or len(status_line) > 8192:
        raise ControlClientError("invalid HTTP status line")
    parts = status_line.decode("ascii", errors="strict").strip().split(" ", 2)
    if len(parts) < 2 or parts[0] not in {"HTTP/1.0", "HTTP/1.1"}:
        raise ControlClientError("invalid HTTP status line")
    try:
        status = int(parts[1])
    except ValueError as error:
        raise ControlClientError("invalid HTTP status code") from error
    headers: dict[str, str] = {}
    total = len(status_line)
    for _ in range(MAX_CONTROL_HEADERS):
        line = await reader.readline()
        total += len(line)
        if total > 64 * 1024 or not line.endswith(b"\r\n"):
            raise ControlClientError("invalid or oversized HTTP response headers")
        if line == b"\r\n":
            return status, headers
        name, separator, value = line.partition(b":")
        key = name.decode("ascii", errors="strict").strip().lower()
        if not separator or not key or key in headers:
            raise ControlClientError("invalid or duplicate HTTP response header")
        headers[key] = value.decode("ascii", errors="strict").strip()
    raise ControlClientError("too many HTTP response headers")


async def _read_response_body(reader: asyncio.StreamReader, headers: Mapping[str, str]) -> bytes:
    transfer_encoding = headers.get("transfer-encoding", "").lower()
    if transfer_encoding:
        if transfer_encoding != "chunked":
            raise ControlClientError("unsupported HTTP transfer encoding")
        return await _read_chunked_body(reader)
    if "content-length" in headers:
        try:
            length = int(headers["content-length"])
        except ValueError as error:
            raise ControlClientError("invalid HTTP content length") from error
        if not 0 <= length <= MAX_CONTROL_RESPONSE_BYTES:
            raise ControlClientError("Control Plane response is too large")
        return await reader.readexactly(length)
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
        size_line = await reader.readline()
        raw_size = size_line.partition(b";")[0].strip()
        try:
            size = int(raw_size, 16)
        except ValueError as error:
            raise ControlClientError("invalid chunk size") from error
        if size < 0 or len(result) + size > MAX_CONTROL_RESPONSE_BYTES:
            raise ControlClientError("Control Plane response is too large")
        if size == 0:
            if await reader.readline() != b"\r\n":
                raise ControlClientError("chunked trailers are not supported")
            return bytes(result)
        result.extend(await reader.readexactly(size))
        if await reader.readexactly(2) != b"\r\n":
            raise ControlClientError("invalid chunk delimiter")
