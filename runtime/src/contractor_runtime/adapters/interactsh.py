"""Allocation-owned Interactsh protocol client; no ambient collectors or proxy fallback."""

from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import math
import secrets
import ssl
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from contractor_runtime import threads
from contractor_runtime.adapters.host import (
    AdapterFactoryError,
    AdapterHandles,
    AdapterSettings,
    RuntimeAdapterBuildContext,
    RuntimeAdapterMetricsState,
)
from contractor_runtime.contracts import InteractshSettings
from contractor_runtime.strict_json import strict_json_loads as loads

# LIM-INT-1..8, docs/spec/platform/limits.md. Upstream protocol is pinned by
# scripts/test_interactsh_protocol.sh (RSA-OAEP/SHA256 and AES-CTR).
MAX_REGISTRATIONS = 16
MAX_EVENTS = 100
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_EVENT_BYTES = 64 * 1024
REGISTRATION_TTL_SECONDS = 3600
POLL_INTERVAL_SECONDS = 2
CLEANUP_SECONDS = 10
MAX_ACTION_BYTES = 1024


class InteractshError(RuntimeError):
    """A stable, detail-free tool failure. Provider text never escapes this client."""

    def __init__(
        self, code: str, *, retryable: bool = False, retry_after_seconds: int | None = None
    ) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds


@dataclass(slots=True, repr=False)
class Registration:
    handle: str
    correlation: str
    nonce: str
    secret: str = field(repr=False)
    key: rsa.RSAPrivateKey | None = field(repr=False)
    created: float
    state: str = "outcome_unknown"
    last_poll: float = -float("inf")
    coverage_gap: bool = False
    reconciled: PollBatch | None = None

    def erase(self) -> None:
        self.secret = ""
        self.key = None
        self.nonce = ""
        self.reconciled = None


@dataclass(frozen=True, slots=True)
class PollBatch:
    events: tuple[bytes, ...] = ()
    discarded: int = 0


class InteractshClient:
    def __init__(
        self,
        settings: InteractshSettings,
        *,
        timeout_seconds: float,
        metrics: RuntimeAdapterMetricsState,
        transport: httpx.AsyncBaseTransport | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        tls = ssl.create_default_context()
        if settings.ca_bundle_pem is not None:
            tls.load_verify_locations(cadata=settings.ca_bundle_pem)
        headers = {"Accept": "application/json", "Accept-Encoding": "identity"}
        if settings.token is not None:
            headers["Authorization"] = settings.token.get_secret_value()
        self._client: httpx.AsyncClient | None = httpx.AsyncClient(
            transport=transport or httpx.AsyncHTTPTransport(verify=tls, trust_env=False, retries=0),
            trust_env=False,
            follow_redirects=False,
            headers=headers,
            timeout=timeout_seconds,
        )
        self._endpoint = settings.endpoint.rstrip("/")
        self._domain = urlsplit(settings.endpoint).hostname or ""
        self._timeout = timeout_seconds
        self._metrics = metrics
        self._now = monotonic
        self._registrations: dict[str, Registration] = {}
        self._lock = asyncio.Lock()
        self._active: set[asyncio.Task[Any]] = set()
        self._closing = False

    @property
    def timeout_seconds(self) -> float:
        return self._timeout

    def registration(self, handle: str) -> Registration:
        item = self._registrations.get(handle) if isinstance(handle, str) else None
        if self._closing or item is None:
            raise InteractshError("interactsh_handle_invalid")
        if self._now() - item.created >= REGISTRATION_TTL_SECONDS or item.state == "expired":
            item.state = "expired"
            raise InteractshError("interactsh_registration_expired")
        return item

    def public_registration(self, item: Registration) -> dict[str, Any]:
        result: dict[str, Any] = {"handle": item.handle, "state": item.state}
        if item.state == "registered":
            result["payload"] = f"{item.correlation}{item.nonce}.{self._domain}"
            result["correlation_id"] = item.correlation
            result["expires_in_seconds"] = max(
                0, int(REGISTRATION_TTL_SECONDS - (self._now() - item.created))
            )
        return result

    async def register(self) -> Registration:
        task = asyncio.current_task()
        if task is None or self._closing:
            raise InteractshError("interactsh_handle_invalid")
        self._active.add(task)
        item: Registration | None = None
        try:
            async with asyncio.timeout(self._timeout), self._lock:
                if len(self._registrations) >= MAX_REGISTRATIONS:
                    raise InteractshError("interactsh_limit_exceeded")
                key = await threads.to_thread_until_done(
                    rsa.generate_private_key,
                    65537,
                    2048,
                    name="interactsh-key",
                )
                item = Registration(
                    handle=secrets.token_hex(16),
                    correlation=secrets.token_hex(10),
                    nonce="".join(
                        secrets.choice("ybndrfg8ejkmcpqxot1uwisza345h769") for _ in range(13)
                    ),
                    secret=secrets.token_hex(32),
                    key=key,
                    created=self._now(),
                )
                # Ownership precedes the remote mutation, including a lost response.
                self._registrations[item.handle] = item
                public = (
                    key.public_key()
                    .public_bytes(
                        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
                    )
                    .replace(b"PUBLIC KEY", b"RSA PUBLIC KEY")
                )
                try:
                    result = await self._request(
                        "POST",
                        "/register",
                        json={
                            "public-key": base64.b64encode(public).decode(),
                            "secret-key": item.secret,
                            "correlation-id": item.correlation,
                        },
                    )
                    if result.get("message") != "registration successful":
                        raise InteractshError("interactsh_response_invalid")
                    item.state = "registered"
                except InteractshError as error:
                    if error.code in {"interactsh_auth_failed", "interactsh_request_invalid"}:
                        item.state = "rejected"
                        item.erase()
                        raise
                    # A same-identity poll authenticates the registration. Its result
                    # is kept for the first tool poll; no new identity is generated.
                    try:
                        batch = await self._poll(item)
                        item.reconciled = batch
                        item.state = "registered"
                    except InteractshError:
                        pass
                return item
        except TimeoutError:
            # If generated, the owned identity remains available for cleanup.
            if item is not None:
                return item
            raise InteractshError("interactsh_request_failed", retryable=True) from None
        finally:
            self._active.discard(task)

    async def poll(self, handle: str) -> PollBatch:
        task = asyncio.current_task()
        if task is None:
            raise InteractshError("interactsh_handle_invalid")
        self._active.add(task)
        try:
            async with asyncio.timeout(self._timeout), self._lock:
                item = self.registration(handle)
                if item.reconciled is not None:
                    batch = item.reconciled
                    item.reconciled = None
                    return batch
                if self._now() - item.last_poll < POLL_INTERVAL_SECONDS:
                    raise InteractshError("interactsh_rate_limited", retryable=True)
                item.last_poll = self._now()
                try:
                    batch = await self._poll(item)
                    item.state = "registered"
                    return batch
                except BaseException:
                    # Upstream drains its queue before returning the response.
                    item.coverage_gap = True
                    raise
        except TimeoutError:
            raise InteractshError("interactsh_request_failed", retryable=True) from None
        finally:
            self._active.discard(task)

    async def _poll(self, item: Registration) -> PollBatch:
        result = await self._request(
            "GET",
            "/poll",
            params={
                "id": item.correlation,
                "secret": item.secret,
            },
        )
        return await threads.to_thread_until_done(
            _decrypt_batch, result, item, name="interactsh-decrypt"
        )

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        client = self._client
        if client is None:
            raise InteractshError("interactsh_handle_invalid")
        try:
            async with client.stream(method, self._endpoint + path, **kwargs) as response:
                if response.status_code in {401, 403}:
                    raise InteractshError("interactsh_auth_failed")
                if response.status_code != 200:
                    if response.status_code == 429:
                        # Surface Retry-After rather than sleeping inside a destructive poll.
                        raise InteractshError(
                            "interactsh_rate_limited",
                            retryable=True,
                            retry_after_seconds=_retry_after(response.headers.get("retry-after")),
                        )
                    if response.status_code >= 500:
                        raise InteractshError("interactsh_request_failed", retryable=True)
                    # Upstream's missing-id text is stable within the pinned protocol.
                    raw = await _read_response(response)
                    if path == "/poll" and b"could not get correlation-id from cache" in raw:
                        raise InteractshError("interactsh_registration_expired")
                    raise InteractshError("interactsh_request_invalid")
                raw = await _read_response(response)
            result = await threads.to_thread_until_done(loads, raw, name="interactsh-json")
            if not isinstance(result, dict):
                raise InteractshError("interactsh_response_invalid")
        except InteractshError:
            self._metrics.record_operation(succeeded=False, error_code="request_failed")
            raise
        except (httpx.HTTPError, OSError):
            self._metrics.record_operation(succeeded=False, error_code="request_failed")
            raise InteractshError("interactsh_request_failed", retryable=True) from None
        except (ValueError, TypeError, RecursionError):
            raise InteractshError("interactsh_response_invalid") from None
        self._metrics.record_operation(succeeded=True)
        return result

    async def close(self) -> None:
        self._closing = True
        tasks = tuple(task for task in self._active if task is not asyncio.current_task())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._lock:
            client = self._client
            if client is None:
                return
            try:
                async with asyncio.timeout(CLEANUP_SECONDS):
                    for item in self._registrations.values():
                        if not item.secret:
                            continue
                        try:
                            await self._request(
                                "POST",
                                "/deregister",
                                json={
                                    "correlation-id": item.correlation,
                                    "secret-key": item.secret,
                                },
                            )
                        except InteractshError:
                            # Remote cleanup is best effort; the session has upstream TTL.
                            self._metrics.record_operation(
                                succeeded=False, error_code="close_failed"
                            )
            except TimeoutError:
                self._metrics.record_operation(succeeded=False, error_code="close_failed")
            finally:
                for item in self._registrations.values():
                    item.erase()
                self._registrations.clear()
                self._client = None
                self._endpoint = self._domain = ""
                client.headers.clear()
                await client.aclose()

    def __repr__(self) -> str:
        return f"InteractshClient(active={not self._closing!r})"


async def _read_response(response: httpx.Response) -> bytes:
    if response.headers.get("content-encoding", "identity").lower() != "identity":
        raise InteractshError("interactsh_response_invalid")
    data = bytearray()
    async for chunk in response.aiter_bytes(chunk_size=65536):
        if len(data) + len(chunk) > MAX_RESPONSE_BYTES:
            raise InteractshError("interactsh_response_too_large")
        data.extend(chunk)
    return bytes(data)


def _decrypt_batch(result: dict[str, Any], item: Registration) -> PollBatch:
    try:
        if set(result) - {"data", "extra", "aes_key", "tlddata"} or not {
            "data",
            "extra",
            "aes_key",
        } <= set(result):
            raise ValueError
        data = result["data"]
        if data is None:
            data = []
        if not isinstance(data, list) or any(not isinstance(value, str) for value in data):
            raise ValueError
        if item.key is None:
            raise ValueError
        key = item.key.decrypt(
            base64.b64decode(result["aes_key"], validate=True),
            padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
        )
        if len(key) not in {16, 24, 32}:
            raise ValueError
        events = []
        discarded = max(0, len(data) - MAX_EVENTS)
        # Token-wide/root-domain events are never allocation-owned evidence.
        for name in ("extra", "tlddata"):
            extra = result.get(name)
            if extra is None:
                extra = []
            if not isinstance(extra, list):
                raise ValueError
            discarded += len(extra)
        for value in data[:MAX_EVENTS]:
            cipher = base64.b64decode(value, validate=True)
            if not 16 < len(cipher) <= MAX_EVENT_BYTES + 16:
                discarded += 1
                continue
            decryptor = Cipher(algorithms.AES(key), modes.CTR(cipher[:16])).decryptor()
            raw = decryptor.update(cipher[16:]) + decryptor.finalize()
            event = loads(raw)
            if (
                not isinstance(event, dict)
                or event.get("unique-id") != item.correlation + item.nonce
            ):
                discarded += 1
                continue
            if event.get("protocol") not in {"dns", "http", "https", "smtp", "smtps"}:
                discarded += 1
                continue
            if datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")).tzinfo is None:
                raise ValueError
            ipaddress.ip_address(event["remote-address"])
            events.append(raw)
        return PollBatch(tuple(events), discarded)
    except (ValueError, TypeError, KeyError, binascii.Error, AttributeError, RecursionError):
        raise InteractshError("interactsh_response_invalid") from None


def _retry_after(value: str | None) -> int:
    try:
        if value is None:
            return POLL_INTERVAL_SECONDS
        seconds = (
            float(value)
            if value.isascii() and value.isdecimal()
            else (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
        )
        if math.isfinite(seconds):
            return max(POLL_INTERVAL_SECONDS, min(120, math.ceil(seconds)))
    except (ValueError, TypeError, OverflowError):
        pass
    return POLL_INTERVAL_SECONDS


class InteractshHTTPAdapterFactory:
    ref = "interactsh-http@1"

    async def probe(self) -> bool:
        try:
            async with httpx.AsyncClient(verify=ssl.create_default_context(), trust_env=False):
                return hasattr(rsa, "generate_private_key") and hasattr(modes, "CTR")
        except Exception:
            return False

    async def create(self, context: RuntimeAdapterBuildContext, settings: AdapterSettings):
        if not isinstance(settings, InteractshSettings):
            raise AdapterFactoryError(retryable=False)
        try:
            return InteractshHTTPAdapter(context, settings)
        except Exception:
            raise AdapterFactoryError(retryable=False) from None


class InteractshHTTPAdapter:
    ref = "interactsh-http@1"

    def __init__(self, context: RuntimeAdapterBuildContext, settings: InteractshSettings):
        self.metrics = RuntimeAdapterMetricsState()
        self._client = InteractshClient(
            settings,
            timeout_seconds=min(context.request_timeout_seconds, settings.request_timeout_seconds),
            metrics=self.metrics,
        )
        self.handles = AdapterHandles(interactsh=self._client)

    async def flush(self) -> None:
        return

    async def close(self) -> None:
        await self._client.close()
        self.handles = AdapterHandles()
