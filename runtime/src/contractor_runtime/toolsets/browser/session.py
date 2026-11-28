"""Browser observations and target I/O owned by one allocation.

Playwright never continues a target request. The allocation HTTP owner performs
it, and a mandatory controlled proxy covers Chromium's non-route network paths.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright

from contractor_runtime.adapters.bridge.proxy import HTTPProxyBridge
from contractor_runtime.adapters.http_transport import HTTPTransportError
from contractor_runtime.core.async_tasks import join_owned_task
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.redaction import redact_private_values
from contractor_runtime.threads import consume_task_exception
from contractor_runtime.toolsets.common.http_capture import HTTPObservation
from contractor_runtime.toolsets.common.http_exchange import TransportExchangeCapture

MAX_CONTEXTS = 4
MAX_ACTIONS = 256
MAX_REQUESTS = 1024
MAX_SCREENSHOTS = 64
MAX_MAP_BYTES = 1024 * 1024
MAX_SCREENSHOT_BYTES = 4 * 1024 * 1024
BROWSER_ARGS = [
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-domain-reliability",
    "--disable-quic",
    "--dns-prefetch-disable",
    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
    "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1",
    "--proxy-bypass-list=<-loopback>",
]
# Do not offer arbitrary JavaScript/eval to the model. Page scripts are target
# content; disable transports which do not have the allocation HTTP contract.
INIT_SCRIPT = """(() => {
 if (globalThis.ServiceWorkerContainer) {
   Object.defineProperty(ServiceWorkerContainer.prototype, 'register', {
     value: () => Promise.reject(new DOMException(
       'Service Workers are disabled', 'NotSupportedError')),
     configurable: false
   });
 }
 for (const key of ['RTCPeerConnection', 'webkitRTCPeerConnection', 'WebTransport']) {
   Object.defineProperty(globalThis, key, {value: undefined, configurable: false});
 }
})()"""
SNAPSHOT = """() => ({
 title: document.title.slice(0, 256),
 text: document.body ? document.body.innerText.slice(0, 12000) : '',
 links: Array.from(document.querySelectorAll('a[href]')).slice(0, 100)
   .map(a => ({text: a.innerText.slice(0, 160), url: a.href.slice(0, 4096)})),
 controls: Array.from(document.querySelectorAll('button,input,select,textarea,[role=button]'))
   .slice(0, 100).map(e => ({tag:e.tagName.toLowerCase(), id:e.id.slice(0,160),
    name:(e.getAttribute('name')||'').slice(0,160), type:(e.getAttribute('type')||'').slice(0,80),
    text:(e.innerText||e.getAttribute('aria-label')||'').slice(0,160)}))
})"""


@dataclass
class BrowserContextState:
    id: str
    credential_id: str | None
    context: object
    page: object
    bridge: HTTPProxyBridge
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_action: str | None = None
    invocation_id: str | None = None
    pending: set = field(default_factory=set)


class BrowserSession:
    def __init__(self, transport, captures, *, timeout=15):
        if transport is None:
            raise ValueError("browser-tools@1 requires an engagement HTTP allocation")
        self.transport = transport
        self.captures = captures
        self.timeout = min(timeout, 30)
        self.contexts = {}
        self.actions = []
        self.requests = []
        self.screenshots = []
        self.screenshot_attempts = 0
        self.gaps = []
        self._pw = None
        self._browser = None
        self._launch_lock = asyncio.Lock()
        self._open_lock = asyncio.Lock()
        self._closed = False
        self._close_task = None
        self._owned = set()
        transport.attach_consumer_cleanup(self.close)

    def safe(self, text):
        return redact_private_values(text, self.transport.private_values())

    def gap(self, code):
        if code not in self.gaps and len(self.gaps) < 32:
            self.gaps.append(code)

    async def start(self):
        async with self._launch_lock:
            if self._closed:
                raise ToolInputError("Browser session is closed")
            if self._browser is None:
                self._pw = await async_playwright().start()
                try:
                    self._browser = await self._pw.chromium.launch(
                        headless=True,
                        args=BROWSER_ARGS,
                        proxy={"server": "http://127.0.0.1:9"},
                    )
                except BaseException:
                    await self._pw.stop()
                    self._pw = None
                    raise

    async def open(self, credential_id):
        async with self._open_lock:
            return await self._open(credential_id)

    async def _open(self, credential_id):
        if credential_id is not None and credential_id not in {
            row["credential_id"] for row in self.transport.credentials()
        }:
            raise ToolInputError("Credential is not assigned to this allocation")
        async with self._launch_lock:
            if len(self.contexts) >= MAX_CONTEXTS:
                raise ToolInputError("Browser context limit reached")
        if any(s.credential_id == credential_id for s in self.contexts.values()):
            raise ToolInputError("Identity already has an open context; use its context_id")
        await self.start()
        bridge = HTTPProxyBridge(
            self.transport,
            credential_id=credential_id,
            deadline=time.monotonic() + 1800,
        )
        await bridge.__aenter__()
        context = None
        try:
            proxy = urlsplit(bridge.proxy_url)
            context = await self._browser.new_context(
                viewport={"width": 1280, "height": 720},
                service_workers="block",
                accept_downloads=False,
                proxy={
                    "server": f"http://{proxy.hostname}:{proxy.port}",
                    "username": proxy.username,
                    "password": proxy.password,
                    "bypass": "<-loopback>",
                },
            )
            await context.add_init_script(INIT_SCRIPT)
            page = await context.new_page()
            state = BrowserContextState(
                f"b{len(self.contexts) + 1}", credential_id, context, page, bridge
            )
            await context.route("**/*", lambda route: self._route(state, route))
            await context.route_web_socket("**/*", self._websocket)
            context.on("page", lambda popup: self._popup(state, popup))
            self.contexts[state.id] = state
            return {
                "context_id": state.id,
                "credential_id": credential_id,
                "scope": self.transport.model_scope(),
                "credentials": self.transport.credentials(),
            }
        except BaseException:
            if context is not None:
                await context.close()
            await bridge.__aexit__()
            raise

    async def _websocket(self, socket):
        self.gap("websocket_blocked")
        await socket.close(code=1008, reason="Unsupported transport")

    def _popup(self, state, popup):
        self.gap("popup_blocked")
        # The route handler independently denies popup requests before closing.
        task = asyncio.create_task(popup.close())
        self._owned.add(task)
        task.add_done_callback(self._owned.discard)
        task.add_done_callback(consume_task_exception)

    async def _route(self, state, route):
        task = asyncio.current_task()
        state.pending.add(task)
        request = route.request
        action = state.active_action
        try:
            try:
                source_page = request.frame.page
            except PlaywrightError:
                # Worker requests have no frame. Keep this explicitly unsupported
                # rather than guessing whether an unowned popup initiated it.
                self.gap("worker_request_blocked")
                await route.abort("blockedbyclient")
                return
            if self._closed or source_page != state.page:
                await route.abort("blockedbyclient")
                return
            if len(self.requests) >= MAX_REQUESTS:
                self.gap("request_limit_reached")
                await route.abort("blockedbyclient")
                return
            request_id = self.captures.new_http_request()
            capture = TransportExchangeCapture(request_id, f"browser-{request_id}")
            row = {
                "request_id": request_id,
                "context_id": state.id,
                "action_id": action,
                "association": "observed-during-action" if action else "background",
                "method": request.method,
                "url": self.safe(request.url),
                "resource_type": request.resource_type,
                "status": None,
            }
            self.requests.append(row)
            # Browser cookies never override a managed identity's private jar.
            headers = [
                (k.encode("ascii"), v.encode("latin-1"))
                for k, v in (await request.all_headers()).items()
                if k.lower()
                not in {
                    "host",
                    "cookie",
                    "authorization",
                    "proxy-authorization",
                    "connection",
                    "content-length",
                    "accept-encoding",
                }
            ]
            try:
                outcome, response = await self.transport.request(
                    request.url,
                    credential_id=state.credential_id,
                    method=request.method,
                    headers=headers,
                    body=request.post_data_buffer or b"",
                    timeout=self.timeout,
                    follow_redirects=False,
                    capture=capture,
                )
            finally:
                if capture.exchange.attempts:
                    await self.captures.put(
                        capture.exchange.evidence(),
                        state.invocation_id,
                        observation=HTTPObservation(capture.exchange.attempts[0].at),
                    )
            if (
                response is None
                or outcome.kind != "completed"
                or response.truncated
                or not response.session_valid
            ):
                row["error"] = outcome.reason or "response_unavailable"
                self.gap("request_failed_or_denied")
                await route.abort("blockedbyclient")
                return
            row["status"] = response.status
            # Playwright fulfill has a header dictionary. Preserve Set-Cookie
            # duplicates using its newline representation; wire payload stays exact
            # in the shared HTTP evidence buffer.
            selected = {}
            for name, value in response.headers:
                key = name.decode("ascii").lower()
                if key not in {"transfer-encoding", "connection", "content-length"}:
                    decoded = value.decode("latin-1")
                    selected[key] = (
                        selected[key] + ("\n" if key == "set-cookie" else ", ") + decoded
                        if key in selected
                        else decoded
                    )
            if selected.get("content-type", "").startswith("text/event-stream"):
                row["error"] = "streaming_unsupported"
                self.gap("streaming_unsupported")
                await route.abort("blockedbyclient")
                return
            await route.fulfill(status=response.status, headers=selected, body=response.body)
        except HTTPTransportError:
            self.gap("allocation_authority_lost")
            await route.abort("blockedbyclient")
        except PlaywrightError:
            self.gap("browser_request_interrupted")
        finally:
            state.pending.discard(task)

    def state(self, context_id):
        if self._closed or context_id not in self.contexts:
            raise ToolInputError("Unknown or closed browser context")
        return self.contexts[context_id]

    async def action(self, context_id, kind, value, invocation_id):
        state = self.state(context_id)
        async with state.lock:
            if len(self.actions) >= MAX_ACTIONS:
                raise ToolInputError("Browser action limit reached")
            row = {
                "id": f"a{len(self.actions) + 1}",
                "context_id": context_id,
                "kind": kind,
                "target": self.safe(value) if kind != "fill" else "[input]",
                "before_url": self.safe(state.page.url),
                "started_at": time.time(),
                "outcome": "incomplete",
            }
            self.actions.append(row)
            state.active_action = row["id"]
            state.invocation_id = invocation_id
            try:
                async with asyncio.timeout(self.timeout):
                    if kind == "navigate":
                        if len(value) > 4096 or urlsplit(value).scheme not in {"http", "https"}:
                            raise ToolInputError("Navigation requires a bounded HTTP(S) URL")
                        await state.page.goto(
                            value, wait_until="domcontentloaded", timeout=self.timeout * 1000
                        )
                    elif kind == "click":
                        await state.page.locator(value).click(timeout=self.timeout * 1000)
                    else:
                        selector, text = value
                        await state.page.locator(selector).fill(text, timeout=self.timeout * 1000)
                    # Finite observation window; this is temporal association,
                    # never a claim that all future requests were caused by action.
                    await asyncio.sleep(0.2)
                    while state.pending:
                        await asyncio.gather(*tuple(state.pending))
                row["outcome"] = "observed"
            except (PlaywrightError, TimeoutError):
                row["outcome"] = "incomplete"
                self.gap("action_incomplete_do_not_retry_automatically")
            finally:
                state.active_action = None
                row["after_url"] = self.safe(state.page.url)
                row["ended_at"] = time.time()
            return row

    async def snapshot(self, context_id):
        state = self.state(context_id)
        async with state.lock:
            result = await state.page.evaluate(SNAPSHOT)

            # Do not echo password/input values. DOM text can still contain
            # target data; handle this artifact like other target evidence.
            def safe_value(value):
                if isinstance(value, str):
                    return self.safe(value)
                if isinstance(value, list):
                    return [safe_value(row) for row in value]
                if isinstance(value, dict):
                    return {key: safe_value(row) for key, row in value.items()}
                return value

            return safe_value({"url": state.page.url, **result})

    async def screenshot(self, context_id):
        state = self.state(context_id)
        async with state.lock:
            data = await state.page.screenshot(
                type="png",
                full_page=False,
                mask=[state.page.locator("input[type=password]")],
                timeout=self.timeout * 1000,
            )
            if len(data) > MAX_SCREENSHOT_BYTES:
                raise ToolInputError("Screenshot exceeds the 4 MiB limit")
            return data, self.safe(state.page.url)

    def export(self):
        if any(state.bridge.requests for state in self.contexts.values()):
            self.gap("proxy_fallback_requests_not_in_action_map")
        value = {
            "schema": "contractor.frontend-route-map.v1",
            "association": "temporal observation, not proof of causality",
            "unsupported": [
                "service-workers",
                "websocket",
                "webrtc",
                "webtransport",
                "streaming",
                "worker-requests",
                "popups",
                "downloads",
            ],
            "contexts": [
                {"id": s.id, "credential_id": s.credential_id} for s in self.contexts.values()
            ],
            "routes": sorted(
                {
                    a["after_url"]
                    for a in self.actions
                    if a.get("after_url", "").startswith(("http://", "https://"))
                }
            ),
            "actions": self.actions,
            "requests": self.requests,
            "screenshots": self.screenshots,
            "screenshot_attempts": self.screenshot_attempts,
            "gaps": self.gaps,
        }
        data = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        if len(data) > MAX_MAP_BYTES:
            raise ToolInputError("Route map exceeds the 1 MiB limit; export fewer observations")
        return data

    async def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(), name="browser-close")
        cancelled = await join_owned_task(self._close_task)
        self._close_task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _close(self):
        self._closed = True
        # Attempt every owner even when one close fails. Never leave the driver
        # or another context running because the first context disconnected.
        failures = []

        async def attempt(operation):
            try:
                await operation()
            except Exception as error:
                failures.append(error)

        for state in self.contexts.values():
            await attempt(state.context.close)
            for task in tuple(state.pending):
                task.cancel()
            if state.pending:
                await asyncio.gather(*tuple(state.pending), return_exceptions=True)
            await attempt(state.bridge.__aexit__)
        for task in tuple(self._owned):
            task.cancel()
        if self._owned:
            await asyncio.gather(*self._owned, return_exceptions=True)
        if self._browser is not None:
            await attempt(self._browser.close)
        if self._pw is not None:
            await attempt(self._pw.stop)
        self.contexts.clear()
        self.requests.clear()
        self.actions.clear()
        self.screenshots.clear()
        self.gaps.clear()
        self.captures.clear("http")
        if failures:
            raise RuntimeError("Browser cleanup failed") from None
