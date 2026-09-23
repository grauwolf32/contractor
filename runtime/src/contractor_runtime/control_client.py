"""Sequenced Runtime Agent registration and heartbeat client."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import ssl
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

from contractor_runtime import _https
from contractor_runtime.backoff import bounded_backoff
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
MAX_CONTROL_HEADERS = _https.MAX_RESPONSE_HEADERS
MAX_CONTROL_HEADER_BYTES = _https.MAX_RESPONSE_HEADER_BYTES


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

    Each control call opens a fresh bounded connection through the shared
    private exchange, which verifies the Control Plane role before the first
    request byte is written.
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
        try:
            response = await _https.exchange(
                self._host,
                self._port,
                self._context,
                request,
                timeout_seconds=self._timeout,
                max_response_bytes=MAX_CONTROL_RESPONSE_BYTES,
                verify_peer=verify_control_plane_peer,
            )
        except _https.HTTPResponseTooLargeError:
            raise ControlClientError("Control Plane response is too large") from None
        except _https.HTTPResponseError as error:
            raise ControlClientError(str(error)) from None
        if not 200 <= response.status_code < 300:
            raise ControlHTTPError(response.status_code)
        return response.body

    def _encode_request(self, path: str, payload: Mapping[str, Any]) -> bytes:
        if not path.startswith("/") or "?" in path or "#" in path or not _is_visible_ascii(path):
            raise ValueError("private control path must be an absolute ASCII path without query")
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        headers = {
            "Host": _https.host_header(self._host, self._port),
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "Accept": "application/json",
            "X-Request-ID": _https.new_request_id(),
            "Connection": "close",
        }
        return _https.encode_request("POST", self._base_path + path, headers, body)


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
        # Reconciliation may wait for a long allocation operation. It runs in
        # one serialized background task so heartbeats keep renewing the lease.
        self._pending_reconciliation: deque[tuple[ReconciliationAction, str | None]] = deque()
        self._reconciliation_task: asyncio.Task[None] | None = None
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
        sent_at = self._watchdog.now() if self._watchdog is not None else None
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
            await self._watchdog.arm(self._timing.confirmed_lease_seconds, sent_at)
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
        # Keep the request, acknowledgement and reconciliation order in
        # sequence even when callers invoke heartbeat_once concurrently.
        async with self._heartbeat_lock:
            return await self._exchange_heartbeat()

    async def _exchange_heartbeat(self) -> HeartbeatResponse:
        self._sequence += 1
        request = await self._state.heartbeat(self._sequence, self._echoed_ack)
        path = f"/private/v1/agents/{quote(self._state.instance_id, safe='')}/heartbeat"
        sent_at = self._watchdog.now() if self._watchdog is not None else None
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
            await self._watchdog.acknowledge(
                response.ack_seq, self._timing.confirmed_lease_seconds, sent_at
            )
        await self._apply_reconciliation(response)
        return response

    async def _apply_reconciliation(self, response: HeartbeatResponse) -> None:
        if response.action is ReconciliationAction.DRAIN:
            if self._reconciliation is None or response.allocation_id is None:
                raise ControlClientError("heartbeat drain action cannot be applied")
            self._schedule_reconciliation(response.action, response.allocation_id)
        elif response.action is ReconciliationAction.RELEASE:
            if self._reconciliation is None:
                raise ControlClientError("heartbeat release action cannot be applied")
            self._schedule_reconciliation(response.action, response.allocation_id)
        elif response.action is ReconciliationAction.REREGISTER:
            await self.register()

    def _schedule_reconciliation(
        self, action: ReconciliationAction, allocation_id: str | None
    ) -> None:
        # Repeated heartbeats re-send the same authoritative action. Queue it
        # once so a long-running allocation operation cannot grow the backlog.
        if (action, allocation_id) not in self._pending_reconciliation:
            self._pending_reconciliation.append((action, allocation_id))
        if self._reconciliation_task is None or self._reconciliation_task.done():
            self._reconciliation_task = asyncio.create_task(
                self._run_reconciliation(), name="runtime-control-reconciliation"
            )
            self._reconciliation_task.add_done_callback(_consume_background_task)

    async def _run_reconciliation(self) -> None:
        assert self._reconciliation is not None
        while self._pending_reconciliation:
            action, allocation_id = self._pending_reconciliation.popleft()
            try:
                if action is ReconciliationAction.DRAIN:
                    assert allocation_id is not None
                    await self._reconciliation.reconcile_drain(
                        allocation_id, self._settings.shutdown_grace_seconds
                    )
                else:
                    await self._reconciliation.confirm_release(allocation_id)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning("Control Plane reconciliation failed (%s)", type(error).__name__)

    async def wait_for_reconciliation(self) -> None:
        """Wait until every reconciliation action received so far was attempted."""

        while self._reconciliation_task is not None and not self._reconciliation_task.done():
            await asyncio.wait({self._reconciliation_task})

    async def close(self) -> None:
        """Cancel queued and in-flight reconciliation on shutdown."""

        self._pending_reconciliation.clear()
        task = self._reconciliation_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def run_heartbeats(self, stop: asyncio.Event) -> None:
        failures = 0
        try:
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
        finally:
            await self.close()

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
        return bounded_backoff(failures, ceiling, self._jitter)


def _consume_background_task(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    task.exception()


def _response_json(raw: Mapping[str, Any] | bytes) -> bytes:
    if isinstance(raw, bytes):
        return raw
    try:
        return json.dumps(raw, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError):
        raise ControlClientError("Control Plane returned invalid JSON") from None


def _is_visible_ascii(value: str) -> bool:
    return all(" " < character < "\x7f" for character in value)
