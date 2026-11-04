"""Allocation-owned, request-driven HTTP with explicit credential selection."""

from __future__ import annotations

import asyncio
import ssl
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from urllib.parse import urlsplit

from contractor_runtime.adapters.bridge import BridgeFailure
from contractor_runtime.adapters.http_policy import HTTPRoutePolicy
from contractor_runtime.adapters.http_session import IdentitySession
from contractor_runtime.adapters.http_stream import RequestBoundary
from contractor_runtime.contracts import AllocationSpec
from contractor_runtime.contracts.engagement import MAX_URL_BYTES, HTTPLimits, resolve_redirect_url
from contractor_runtime.core.http_limits import MAX_CONTROLLED_HEADER_BYTES, MAX_HEADERS
from contractor_runtime.core.http_session import basic_authorization
from contractor_runtime.core.target_policy import Resolver, TargetDenied, TargetUnresolved
from contractor_runtime.shared_config import get_shared_config
from contractor_runtime.threads import to_thread_until_done

TARGET_ADAPTER = "http-target@1"
REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class HTTPTransportError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class HTTPOutcome:
    kind: str
    reason: str = ""


class HTTPTransportFactory:
    ref = TARGET_ADAPTER

    def __init__(
        self,
        protected_urls: tuple[str, ...] = (),
        resolver: Resolver | None = None,
    ) -> None:
        self._protected_urls = protected_urls
        self._resolver = resolver

    async def probe(self) -> bool:
        return True

    def create(self, spec: AllocationSpec) -> HTTPTransport:
        return HTTPTransport(
            spec,
            protected_urls=self._protected_urls,
            resolver=self._resolver,
        )


class HTTPTransport:
    def __init__(
        self,
        spec: AllocationSpec,
        *,
        protected_urls: tuple[str, ...] = (),
        resolver: Resolver | None = None,
        ssl_context: ssl.SSLContext | None = None,
        monotonic=time.monotonic,
        sleep=asyncio.sleep,
    ) -> None:
        projection = spec.engagement
        if projection is None or projection.http is None:
            raise ValueError("controlled HTTP requires an explicit assignment")
        self._assignment = projection.http
        self._http_limits = projection.http.limits or HTTPLimits()
        self._settings = spec.runtime_settings
        self._policy = HTTPRoutePolicy(
            self._assignment, protected_urls=protected_urls, resolver=resolver
        )
        self._clock, self._sleep = monotonic, sleep
        self._engine = None
        self._ssl = ssl_context
        self._tasks: set[asyncio.Task] = set()
        self._response_cleanup: list[Callable[[], None]] = []
        self._request_count = 0
        self._next_request_at = 0.0
        self._rate_lock = asyncio.Lock()
        self._initialize_lock = asyncio.Lock()
        self._initialized = False
        self._boundaries: set[RequestBoundary] = set()
        self._sessions: list[IdentitySession] = []
        try:
            for identity in self._assignment.identities:
                self._sessions.append(
                    IdentitySession(
                        identity,
                        (self._settings.http_identities or {}).get(identity.id),
                    )
                )
        except Exception:
            for session in self._sessions:
                session.close()
            for material in (self._settings.http_identities or {}).values():
                material.fields.clear()
            self._settings.http_identities = None
            raise
        self._semaphore = (
            asyncio.Semaphore(self._http_limits.concurrent_requests)
            if self._http_limits.concurrent_requests is not None
            else nullcontext()
        )
        self._session_locks = {session.identity.id: asyncio.Lock() for session in self._sessions}
        self._authority_deadline: Callable[[], float | None] = lambda: None
        self._stopped = False

    def configure(self, *, authority_deadline: Callable[[], float | None]) -> None:
        """Bind authority without DNS, connecting or issuing target requests."""
        if self._stopped:
            raise HTTPTransportError("http_allocation_closed")
        self._authority_deadline = authority_deadline

    async def close(self) -> None:
        self._stopped = True
        tasks = tuple(self._tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))
        await self._close_transport()
        for cleanup in self._response_cleanup:
            cleanup()
        self._response_cleanup.clear()
        # Stop may precede the Server's release acknowledgement. Drop all
        # allocation settings and target material as soon as the owner joins;
        # failed tasks may retain private request/response traceback frames.
        if self._settings is not None:
            # AllocationContext and the Worker share this private settings
            # object. The target credentials must not survive a joined stop
            # there until the later allocation release.
            self._settings.http_identities = None
        self._tasks.clear()
        self._settings = None
        self._assignment = None
        self._policy = None
        self._boundaries.clear()
        for session in self._sessions:
            session.close()
        self._sessions.clear()
        self._ssl = None
        self._authority_deadline = lambda: None

    def _require_authority(self) -> float:
        if self._stopped:
            raise HTTPTransportError("http_allocation_closed")
        lease = self._authority_deadline()
        if lease is None or self._clock() >= lease:
            raise HTTPTransportError("http_allocation_closed")
        return lease

    def attach_response_buffer(self, cleanup: Callable[[], None]) -> None:
        """Allocation teardown also erases model-facing response buffers."""
        self._response_cleanup.append(cleanup)

    def own_invocation(self) -> bool:
        self._require_authority()
        task = asyncio.current_task()
        added = task not in self._tasks
        self._tasks.add(task)
        return added

    def finish_invocation(self, added: bool) -> None:
        if added:
            self._tasks.discard(asyncio.current_task())

    async def _close_boundary(self, boundary: RequestBoundary) -> None:
        self._boundaries.discard(boundary)

    async def _close_transport(self) -> None:
        if self._engine is not None:
            await self._engine.close()
            self._engine = None

    async def _initialize(self) -> None:
        async with self._initialize_lock:
            if self._initialized:
                return
            await self._policy.initialize(self._settings)
            if self._engine is None:
                from contractor_runtime.adapters.bridge.mitmproxy import MitmproxyEngine

                self._engine = MitmproxyEngine(
                    require_authority=self._require_authority, ssl_context=self._ssl
                )
                try:
                    await self._engine.start()
                except BaseException:
                    await self._close_transport()
                    raise
            self._initialized = True

    async def open_proxy(self):
        """Acquire private scanner ingress on this allocation's native engine."""
        self._require_authority()
        await self._initialize()
        return self._engine

    def _session(self, credential_id: str | None) -> IdentitySession:
        self._require_authority()
        if credential_id is not None and (not isinstance(credential_id, str) or not credential_id):
            raise ValueError("credential is unavailable to this Audit")
        for session in self._sessions:
            ref = session.identity.credential
            if (credential_id is None and ref is None) or (
                ref is not None and ref.credential_id == credential_id
            ):
                return session
        raise ValueError("credential is unavailable to this Audit")

    def credentials(self) -> list[dict]:
        """List only the exact credentials assigned by this Audit's private prepare."""
        self._require_authority()
        return [
            {
                "credential_id": session.identity.credential.credential_id,
                "kind": session.identity.kind,
                "origins": list(session.identity.origins),
                "role": session.identity.role,
                "tenant": session.identity.tenant,
            }
            for session in self._sessions
            if session.identity.credential is not None
        ]

    def configured_limits(self) -> dict[str, int | float]:
        """Configured slot limits for execution provenance; omitted values stay absent."""
        return self._http_limits.model_dump(by_alias=True, exclude_none=True)

    def private_values(self) -> tuple[str, ...]:
        """Static assigned credentials and their HTTP authentication encodings."""
        values = []
        for session in self._sessions:
            material = session.material
            if material is None:
                continue
            fields = {key: value.get_secret_value() for key, value in material.fields.items()}
            values.extend(fields.values())
            if session.identity.kind == "basic":
                authorization = basic_authorization(fields["username"], fields["password"])
                values.extend((authorization, authorization.removeprefix("Basic ")))
        return tuple(value for value in values if value)

    def read_credential(self, credential_id: str) -> dict:
        """Explicit model access is restricted to the immutable Audit assignment."""
        if not isinstance(credential_id, str) or not credential_id:
            raise ValueError("credential is unavailable to this Audit")
        session = self._session(credential_id)
        if session.material is None:
            raise ValueError("credential is unavailable to this Audit")
        return {
            "credential_id": credential_id,
            "kind": session.identity.kind,
            "values": {
                name: value.get_secret_value() for name, value in session.material.fields.items()
            },
        }

    async def clear_cookies(self) -> None:
        """Drop learned cookies; a cookie credential cannot silently regain its jar."""
        self._require_authority()
        for session in self._sessions:
            async with self._session_locks[session.identity.id]:
                session.clear()
                if session.identity.kind == "cookies":
                    session.state = "blocked"

    async def session_summaries(self) -> list[dict]:
        self._require_authority()
        result = []
        for session in self._sessions:
            async with self._session_locks[session.identity.id]:
                cookies = await to_thread_until_done(
                    session.cookie_summary, name="audit-cookie-summary"
                )
                result.append(
                    {
                        "credential_id": session.identity.credential.credential_id
                        if session.identity.credential
                        else None,
                        "state": session.state,
                        **cookies,
                    }
                )
        return result

    async def request(self, url: str, *, credential_id: str | None = None, **options):
        """Execute one explicit tool call; preparation never invokes this method."""
        session = self._session(credential_id)
        if session.state == "blocked" or not session.allows_origin(url):
            raise ValueError("selected credential does not authorize this origin")
        if (
            self._http_limits.request_body_bytes is not None
            and len(options.get("body", b"")) > self._http_limits.request_body_bytes
        ):
            raise ValueError("request body exceeds the Runtime slot bound")
        if credential_id is not None and any(
            name.lower() in {b"authorization", b"cookie"} for name, _ in options.get("headers", ())
        ):
            raise ValueError("selected credential conflicts with explicit authentication headers")
        added = self.own_invocation()
        timeout = min(options.pop("timeout", 30), 120)
        deadline = min(self._require_authority(), self._clock() + timeout)
        try:
            async with asyncio.timeout(max(0, deadline - self._clock())):
                async with self._semaphore:
                    await self._initialize()
                    return await self._execute_session(session, url, deadline=deadline, **options)
        finally:
            self.finish_invocation(added)

    async def _execute_session(self, session, url, **options):
        outcome, response = await self._request_chain(session, url, **options)
        return outcome, response

    async def _admit_request(self) -> None:
        async with self._rate_lock:
            self._require_authority()
            if (
                self._http_limits.target_requests is not None
                and self._request_count >= self._http_limits.target_requests
            ):
                raise HTTPTransportError("http_limit_exceeded")
            delay = self._next_request_at - self._clock()
            if delay > 0:
                await self._sleep(delay)
            self._require_authority()
            self._request_count += 1
            if self._http_limits.requests_per_second is not None:
                self._next_request_at = self._clock() + 1 / self._http_limits.requests_per_second

    @staticmethod
    def _wire_headers(url, request):
        names = {name.lower() for name, _ in request.headers}
        headers = [
            (b"Host", urlsplit(url).netloc.encode("ascii")),
            *([] if b"accept" in names else [(b"Accept", b"*/*")]),
            *([] if b"accept-encoding" in names else [(b"Accept-Encoding", b"identity")]),
            *request.headers,
        ]
        if request.body or request.method in {"POST", "PUT", "PATCH"}:
            headers.append((b"Content-Length", str(len(request.body)).encode()))
        if (
            len(headers) > MAX_HEADERS
            or sum(len(name) + len(value) + 4 for name, value in headers)
            > MAX_CONTROLLED_HEADER_BYTES
        ):
            raise ValueError("controlled HTTP headers exceed bound")
        return headers

    async def _request_chain(
        self,
        session,
        url,
        *,
        method="GET",
        headers=(),
        body=b"",
        follow_redirects=True,
        deadline=None,
    ):
        if deadline is None:
            deadline = min(self._require_authority(), self._clock() + 30)
        hop = retry = 0
        while True:
            self._require_authority()
            async with self._session_locks[session.identity.id]:
                epoch = session.epoch
                try:
                    private_request = await to_thread_until_done(
                        session.request, url, name="engagement-private-request"
                    )
                except (ValueError, KeyError, UnicodeError):
                    return HTTPOutcome(kind="denied", reason="policy_denied"), None

            def require_session(epoch=epoch):
                if epoch is not session.epoch or session.state == "blocked":
                    raise TargetDenied

            private_request.method = method
            private_request.body = body
            private_request.follow_redirects = follow_redirects
            private_request.headers = [*headers, *private_request.headers]
            private_request.headers = self._wire_headers(url, private_request)
            # Scope is checked before local budget admission, and every retry
            # and redirect obtains its own slot request allowance.
            try:
                await to_thread_until_done(
                    self._policy.for_url,
                    url,
                    session.identity,
                    method,
                    name="engagement-request-admission",
                )
            except TargetDenied:
                return HTTPOutcome(kind="denied", reason="policy_denied"), None
            await self._admit_request()
            boundary = RequestBoundary(
                deadline,
                authority_deadline=self._authority_deadline,
                monotonic=self._clock,
                require_session=require_session,
            )
            self._boundaries.add(boundary)
            next_url = None
            response = None
            try:
                async with asyncio.timeout(max(0, boundary.deadline - self._clock())):
                    outcome, next_url, response = await self._request(
                        url, boundary, session, private_request
                    )
            except asyncio.CancelledError:
                raise
            except TargetDenied:
                outcome = self._failure(boundary, "policy_denied")
            except TimeoutError:
                outcome = self._failure(boundary, "timeout")
            except BridgeFailure as error:
                outcome = self._failure(boundary, error.reason)
            except TargetUnresolved:
                outcome = self._failure(boundary, "connect_failed")
            except ssl.SSLError:
                outcome = self._failure(boundary, "tls_failed")
            except (HTTPTransportError, ValueError, OSError):
                outcome = self._failure(boundary, "transport_failed")
            finally:
                # Retain ownership after a close failure so allocation teardown
                # can finish closing the exact stream.
                await self._close_boundary(boundary)
            if response is not None:
                response.method = method
                response.redirects = hop
                response.retries = retry
                async with self._session_locks[session.identity.id]:
                    try:
                        await to_thread_until_done(
                            session.receive, response, epoch, name="engagement-private-response"
                        )
                    except (ValueError, KeyError, IndexError, UnicodeError, RecursionError):
                        response.session_valid = False
                    if not response.session_valid and epoch is session.epoch:
                        session.clear()
                        if session.identity.kind == "cookies":
                            session.state = "blocked"
            if response is not None and not response.session_valid:
                return outcome, response
            if outcome.kind == "completed" and next_url is not None and hop < 5:
                if (
                    urlsplit(url).netloc != urlsplit(next_url).netloc
                    or urlsplit(url).scheme != urlsplit(next_url).scheme
                ):
                    headers = [
                        (name, value)
                        for name, value in headers
                        if name.lower() not in {b"authorization", b"cookie"}
                    ]
                if response.status == 303 or (response.status in {301, 302} and method == "POST"):
                    method, body = "GET", b""
                    headers = [
                        (name, value)
                        for name, value in headers
                        if name.lower() not in {b"content-type", b"content-encoding"}
                    ]
                hop += 1
                url = next_url
                continue
            if (
                not boundary.request_started
                and method in {"GET", "HEAD", "OPTIONS"}
                and retry == 0
                and outcome.reason in {"connect_failed", "tls_failed"}
            ):
                retry = 1
                continue
            return outcome, response

    @staticmethod
    def _failure(boundary, reason):
        return HTTPOutcome(kind="denied" if reason == "policy_denied" else "failed", reason=reason)

    async def _request(self, url, boundary, session, request):
        policy = await to_thread_until_done(
            self._policy.for_url,
            url,
            session.identity,
            request.method,
            name="engagement-request-scope",
        )
        if not url.isascii():
            raise ValueError("unsupported target URL encoding")
        limit = min(
            get_shared_config().http_response_body_max_bytes,
            self._http_limits.response_body_bytes
            or get_shared_config().http_response_body_max_bytes,
        )
        response = await self._engine.request(
            url, boundary, request, policy, session.identity.id, limit
        )
        next_url = None
        if request.follow_redirects and response.status in REDIRECT_STATUSES:
            locations = [value for name, value in response.headers if name.lower() == b"location"]
            if len(locations) == 1 and len(locations[0]) <= MAX_URL_BYTES:
                try:
                    candidate = resolve_redirect_url(url, locations[0].decode("ascii"))
                    if not session.allows_origin(candidate):
                        raise TargetDenied
                    await to_thread_until_done(
                        self._policy.for_url,
                        candidate,
                        session.identity,
                        "GET"
                        if response.status == 303
                        or (response.status in {301, 302} and request.method == "POST")
                        else request.method,
                        name="engagement-redirect-scope",
                    )
                    next_url = candidate
                except (ValueError, UnicodeError, TargetDenied):
                    pass
        return HTTPOutcome(kind="completed"), next_url, response
