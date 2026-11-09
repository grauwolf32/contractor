"""Private per-invocation scanner capability on the allocation native engine."""

from __future__ import annotations

import asyncio
import base64
import json
import secrets
import time
from typing import TYPE_CHECKING

from contractor_runtime.adapters.bridge.errors import BridgeInfrastructureError
from contractor_runtime.core.async_tasks import join_owned_task
from contractor_runtime.redaction import redact_private_values
from contractor_runtime.threads import consume_task_exception

if TYPE_CHECKING:
    from contractor_runtime.adapters.http_transport import HTTPTransport


class HTTPProxyBridge:
    def __init__(
        self,
        transport: HTTPTransport,
        *,
        credential_id: str | None,
        deadline: float,
        strip_cookies: bool = True,
    ) -> None:
        self._transport = transport
        self._credential_id = credential_id
        self._deadline = deadline
        self.strip_cookies = strip_cookies
        self._token = secrets.token_hex(32)
        encoded = base64.b64encode(("runtime:" + self._token).encode()).decode()
        self._authorization = ("Basic " + encoded).encode()
        private_values = (*transport.private_values(), self._token, encoded)
        self._private_values = tuple(
            {
                representation
                for value in private_values
                for representation in (
                    value,
                    json.dumps(value)[1:-1],
                    json.dumps(value, ensure_ascii=False)[1:-1],
                )
            }
        )
        self._server: asyncio.Server | None = None
        self._tasks: set[asyncio.Task] = set()
        self._writers: set[asyncio.StreamWriter] = set()
        self._handlers = set()
        self._engine = None
        self._exception = None
        self._close_task = None
        self._error_code: str | None = None
        self.failed = asyncio.Event()
        self.requests = 0
        self._proxy_url = ""

    @property
    def error_code(self) -> str | None:
        return self._error_code

    @error_code.setter
    def error_code(self, value: str) -> None:
        self._error_code = self._error_code or value
        self.failed.set()

    async def __aenter__(self):
        catalog = self._transport.credentials()
        if self._credential_id is not None and not any(
            item["credential_id"] == self._credential_id for item in catalog
        ):
            raise ValueError("credential is not assigned to this allocation")
        self._engine = await self._transport.open_proxy()
        self._engine.ingresses.add(self)
        try:
            try:
                self._server = await asyncio.start_server(self._accept, "127.0.0.1", 0)
            except OSError:
                raise BridgeInfrastructureError() from None
            port = self._server.sockets[0].getsockname()[1]
            self._proxy_url = f"http://runtime:{self._token}@127.0.0.1:{port}"
            return self
        except BaseException:
            await self.__aexit__()
            raise

    @property
    def proxy_url(self) -> str:
        """Private IPC capability; adapters pass it through a private config file."""
        if not self._proxy_url:
            raise RuntimeError("HTTP bridge is not open")
        return self._proxy_url

    def _accept(self, reader, writer):
        self._writers.add(writer)
        handler = self._engine.create_ingress(self, reader, writer)
        self._handlers.add(handler)
        task = asyncio.create_task(self._handle(handler, writer), name="http-bridge-request")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(consume_task_exception)

    async def _handle(self, handler, writer):
        try:
            async with asyncio.timeout_at(min(self._deadline, time.monotonic() + 30)):
                await handler.handle_client()
        except asyncio.CancelledError:
            if handler.pending:
                self.error_code = self.error_code or "scan_request_failed"
            raise
        except (TimeoutError, OSError):
            self.error_code = self.error_code or "scan_request_failed"
        finally:
            await handler.join()
            if handler.received and (handler.pending or not handler.completed):
                self.error_code = self.error_code or "scan_request_failed"
            self._handlers.discard(handler)
            self._writers.discard(writer)

    async def __aexit__(self, *_):
        if self._close_task is None or self._close_task.done():
            self._close_task = asyncio.create_task(self._close(), name="http-bridge-close")
        cancelled = await join_owned_task(self._close_task)
        self._close_task.result()
        if cancelled:
            raise asyncio.CancelledError
        if self._exception is not None:
            raise self._exception

    async def _close(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for handler in tuple(self._handlers):
            await handler.join()
            self._handlers.discard(handler)
        for writer in tuple(self._writers):
            writer.close()
            await writer.wait_closed()
            self._writers.discard(writer)
        for result in results:
            if isinstance(result, Exception):
                raise result
        if self._engine is not None:
            self._engine.ingresses.discard(self)
        self._private_values = ()
        self._token = self._proxy_url = ""
        self._authorization = b""
        self._engine = None

    def redact(self, data: bytes) -> bytes:
        return redact_private_values(
            data.decode("utf-8", errors="replace"), self._private_values
        ).encode()
