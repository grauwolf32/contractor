"""One task owns SDK contexts from first initialization through final close."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import httpx
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

from contractor_runtime.adapters.http_proxy import ProxyHTTPClient
from contractor_runtime.contracts.toolset_settings import MCPToolsetSettings
from contractor_runtime.toolsets.discovery import valid_allocation_tool_names
from contractor_runtime.toolsets.mcp.errors import MCPCleanupError, MCPError
from contractor_runtime.toolsets.mcp.redaction import install_private_log_filter, mcp_private_task
from contractor_runtime.toolsets.mcp.schema import (
    ToolDefinition,
    bounded_json,
    declarations_digest,
    validate_schema,
)
from contractor_runtime.toolsets.mcp.transport import create_mcp_http_client

MAX_PAGES = 16
MAX_TOOLS = 1024
MAX_CURSOR_BYTES = 2048
MAX_DESCRIPTION_BYTES = 4096
DEFAULT_CLOSE_SECONDS = 5.0


@dataclass(slots=True, repr=False)
class _Call:
    name: str
    arguments: dict[str, Any]
    deadline: float
    result: asyncio.Future[types.CallToolResult]
    dispatched: bool = False


def _consume_future(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


class MCPSessionOwner:
    def __init__(
        self,
        settings: MCPToolsetSettings,
        selected: Sequence[str],
        *,
        proxy: ProxyHTTPClient | None = None,
        forbidden_hosts: Sequence[str] = (),
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not valid_allocation_tool_names(selected):
            raise MCPError("invalid_selection")
        self._settings: MCPToolsetSettings | None = settings.model_copy(deep=True)
        self._selected = tuple(selected)
        self._proxy = proxy
        self._forbidden_hosts = tuple(forbidden_hosts)
        self._transport = transport
        self._task: asyncio.Task[None] | None = None
        self._started = False
        self._client: httpx.AsyncClient | None = None
        self._stack: AsyncExitStack | None = None
        self._ready: asyncio.Future[None] | None = None
        self._queue: asyncio.Queue[_Call] = asyncio.Queue(maxsize=64)
        self._active: _Call | None = None
        self._definitions: dict[str, ToolDefinition] = {}
        self.schema_digest: str | None = None
        self._closing = False
        self._closed = False
        self._close_deadline: float | None = None
        self._cleanup_failed = False
        self._request_timeout = float(settings.request_timeout_seconds)

    @property
    def closed(self) -> bool:
        return self._closed

    def definition(self, name: str) -> ToolDefinition:
        if self.schema_digest is None or name not in self._definitions:
            raise MCPError("session_not_ready")
        return copy.deepcopy(self._definitions[name])

    async def prepare(self, *, deadline: float) -> None:
        if self._closing:
            raise MCPError("session_closed")
        loop = asyncio.get_running_loop()
        if self._task is None:
            self._ready = loop.create_future()
            self._ready.add_done_callback(_consume_future)
            install_private_log_filter()
            self._task = asyncio.create_task(self._run(deadline), name="contractor-mcp-session")
        assert self._ready is not None
        try:
            async with asyncio.timeout_at(deadline):
                await asyncio.shield(self._ready)
        except asyncio.CancelledError:
            self._request_close()
            raise
        except TimeoutError:
            self._request_close()
            raise MCPError("prepare_timeout", retryable=True) from None

    async def call(
        self, name: str, arguments: dict[str, Any], *, deadline: float
    ) -> types.CallToolResult:
        if self._closing or self.schema_digest is None:
            raise MCPError("session_closed")
        if name not in self._definitions:
            raise MCPError("unselected_tool")
        bounded_json(arguments, 64 * 1024, code="invalid_arguments")
        loop = asyncio.get_running_loop()
        deadline = min(deadline, loop.time() + self._request_timeout)
        future: asyncio.Future[types.CallToolResult] = loop.create_future()
        future.add_done_callback(_consume_future)
        call = _Call(name, copy.deepcopy(arguments), deadline, future)
        try:
            async with asyncio.timeout_at(deadline):
                await self._queue.put(call)
                return await asyncio.shield(future)
        except (TimeoutError, asyncio.CancelledError) as error:
            future.cancel()
            if call.dispatched:
                self._request_close()
            if isinstance(error, asyncio.CancelledError):
                raise
            raise MCPError("outcome_unknown" if call.dispatched else "deadline_exceeded") from None

    def _request_close(self, deadline: float | None = None) -> None:
        now = asyncio.get_running_loop().time()
        deadline = min(
            now + DEFAULT_CLOSE_SECONDS,
            deadline if deadline is not None else now + DEFAULT_CLOSE_SECONDS,
        )
        self._close_deadline = min(self._close_deadline or deadline, deadline)
        if not self._closing:
            self._closing = True
            if self._task is not None and not self._task.done():
                self._task.cancel()

    async def close(self, *, deadline: float | None = None) -> None:
        self._request_close(deadline)
        if self._task is None:
            self._erase()
            self._closed = True
            return
        assert self._close_deadline is not None
        if self._task.cancelled() and not self._started:
            self._erase()
            self._closed = True
            return
        try:
            async with asyncio.timeout_at(self._close_deadline):
                await asyncio.shield(self._task)
        except asyncio.CancelledError:
            if self._task.cancelled() and not self._started:
                self._erase()
                self._closed = True
                return
            raise
        except TimeoutError:
            raise MCPCleanupError from None
        if self._cleanup_failed or not self._closed:
            raise MCPCleanupError

    async def _discover(self, session: ClientSession) -> dict[str, ToolDefinition]:
        names: set[str] = set()
        cursors: set[str] = set()
        selected: dict[str, ToolDefinition] = {}
        cursor: str | None = None
        for _ in range(MAX_PAGES):
            page = await session.list_tools(cursor=cursor)
            for tool in page.tools:
                if tool.name in names or len(tool.name.encode()) > 256:
                    raise MCPError("invalid_inventory")
                names.add(tool.name)
                if len(names) > MAX_TOOLS:
                    raise MCPError("inventory_too_large")
                description = tool.description or ""
                if len(description.encode()) > MAX_DESCRIPTION_BYTES:
                    raise MCPError("description_too_large")
                bounded_json(tool.inputSchema, 64 * 1024, code="schema_too_large")
                if tool.outputSchema is not None:
                    bounded_json(tool.outputSchema, 64 * 1024, code="schema_too_large")
                if tool.name in self._selected:
                    selected[tool.name] = ToolDefinition(
                        tool.name,
                        description,
                        validate_schema(tool.inputSchema),
                        validate_schema(tool.outputSchema)
                        if tool.outputSchema is not None
                        else None,
                    )
            cursor = page.nextCursor
            if cursor is None:
                if set(selected) != set(self._selected):
                    raise MCPError("missing_tool")
                return selected
            if not cursor or len(cursor.encode()) > MAX_CURSOR_BYTES or cursor in cursors:
                raise MCPError("invalid_cursor")
            cursors.add(cursor)
        raise MCPError("too_many_pages")

    async def _serve(self, session: ClientSession) -> None:
        while not self._closing:
            call = await self._queue.get()
            self._active = call
            if call.result.cancelled():
                self._active = None
                continue
            if call.deadline <= asyncio.get_running_loop().time():
                call.result.set_exception(MCPError("deadline_exceeded"))
                self._active = None
                continue
            try:
                async with asyncio.timeout_at(call.deadline):
                    call.dispatched = True
                    # Use the SDK's typed request API. ClientSession.call_tool
                    # may refresh schemas on a cache miss; the allocation owns
                    # frozen validation and must never discover during a call.
                    result = await session.send_request(
                        types.ClientRequest(
                            types.CallToolRequest(
                                params=types.CallToolRequestParams(
                                    name=call.name,
                                    arguments=call.arguments,
                                )
                            )
                        ),
                        types.CallToolResult,
                        request_read_timeout_seconds=timedelta(
                            seconds=max(0.001, call.deadline - asyncio.get_running_loop().time())
                        ),
                    )
                if not call.result.done():
                    call.result.set_result(result)
                self._active = None
            except BaseException as error:
                if not call.result.done():
                    call.result.set_exception(_safe_failure(error, dispatched=True))
                self._active = None
                # A failed/ambiguous transport never reconnects or replays.
                return

    async def _run(self, prepare_deadline: float) -> None:
        self._started = True
        token = mcp_private_task.set(True)
        stack = self._stack = AsyncExitStack()
        transport = None
        try:
            assert self._settings is not None
            client, transport = create_mcp_http_client(
                self._settings,
                proxy=self._proxy,
                forbidden_hosts=self._forbidden_hosts,
                transport=self._transport,
            )
            self._client = client
            read, write, _ = await stack.enter_async_context(
                streamable_http_client(
                    self._settings.endpoint,
                    http_client=client,
                )
            )
            session = await stack.enter_async_context(
                ClientSession(
                    read,
                    write,
                    read_timeout_seconds=timedelta(seconds=self._request_timeout),
                )
            )
            async with asyncio.timeout_at(prepare_deadline):
                await session.initialize()
                self._definitions = await self._discover(session)
                self.schema_digest = declarations_digest(self._definitions)
                transport.discovery = False
            assert self._ready is not None
            self._ready.set_result(None)
            await self._serve(session)
        except BaseException as error:
            self._fail_waiters(
                transport.failure
                if transport is not None and transport.failure is not None
                else _safe_failure(error)
            )
        finally:
            self._closing = True
            self._fail_waiters(MCPError("session_closed"))
            self._close_deadline = self._close_deadline or (
                asyncio.get_running_loop().time() + DEFAULT_CLOSE_SECONDS
            )
            try:
                # All SDK cancel-scope/task-group exits run in this task. The
                # SDK owns optional session DELETE and its protocol headers.
                async with asyncio.timeout_at(self._close_deadline):
                    try:
                        await stack.aclose()
                    except BaseExceptionGroup as error:
                        # A TaskGroup reports already joined request failures at
                        # exit. They are operation failures, not failed teardown.
                        if not _only_transport_failures(error):
                            raise
                    if self._client is not None:
                        await self._client.aclose()
            except BaseException:
                self._cleanup_failed = True
            if not self._cleanup_failed:
                self._erase()
                self._stack = None
                self._client = None
                self._closed = True
            # On failed cleanup keep handles in the fenced allocation owner;
            # neither a second close nor a fresh deadline confirms this release.
            mcp_private_task.reset(token)

    def _fail_waiters(self, error: MCPError) -> None:
        if self._ready is not None and not self._ready.done():
            self._ready.set_exception(error)
        if self._active is not None and not self._active.result.done():
            self._active.result.set_exception(MCPError("outcome_unknown"))
        self._active = None
        while not self._queue.empty():
            call = self._queue.get_nowait()
            if not call.result.done():
                call.result.set_exception(MCPError("session_closed"))

    def _erase(self) -> None:
        if self._settings is not None:
            self._settings.headers.clear()
        self._settings = None
        self._proxy = None
        self._transport = None
        self._definitions.clear()

    def __repr__(self) -> str:
        return f"MCPSessionOwner(ready={self.schema_digest is not None!r}, closed={self._closed!r})"


def _safe_failure(error: BaseException, *, dispatched: bool = False) -> MCPError:
    if isinstance(error, MCPError):
        return MCPError(error.code, retryable=error.retryable)
    if isinstance(error, BaseExceptionGroup):
        for nested in error.exceptions:
            if isinstance(nested, MCPError):
                return _safe_failure(nested, dispatched=dispatched)
    if dispatched:
        return MCPError("outcome_unknown")
    if isinstance(error, (TimeoutError, httpx.TimeoutException, asyncio.CancelledError)):
        return MCPError("prepare_timeout", retryable=True)
    if isinstance(error, httpx.HTTPStatusError) and error.response.status_code in {401, 403}:
        return MCPError("authentication_failed")
    return MCPError("protocol_error")


def _only_transport_failures(error: BaseException) -> bool:
    if isinstance(error, BaseExceptionGroup):
        return all(_only_transport_failures(item) for item in error.exceptions)
    return isinstance(error, MCPError)
