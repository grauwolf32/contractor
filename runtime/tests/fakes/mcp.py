"""Controlled TCP Streamable HTTP endpoint; no external services or model."""

from __future__ import annotations

import asyncio
import json
import ssl
from contextlib import suppress
from typing import Any


def remote_tool(name: str, **overrides: Any) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Controlled {name}",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        **overrides,
    }


class MCPFixture:
    def __init__(
        self,
        *,
        sse: bool = False,
        stateful: bool = True,
        tls: ssl.SSLContext | None = None,
        host: str = "127.0.0.1",
        pages: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        self.sse, self.stateful, self.tls, self.host = sse, stateful, tls, host
        self.pages = pages or [
            [remote_tool("search")],
            [remote_tool("fetch"), remote_tool("delete_document")],
        ]
        self.requests: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.sessions: list[str] = []
        self.deleted: list[str] = []
        self.writers: set[asyncio.StreamWriter] = set()
        self.tasks: set[asyncio.Task[Any]] = set()
        self.block_method: str | None = None
        self.entered = asyncio.Event()
        self.unblock = asyncio.Event()
        self.drop_call = False
        self.redirect = False
        self.status: int | None = None
        self.repeated_cursor = False
        self.raw_response: bytes | None = None
        self.required_header: str | None = None
        self.result: dict[str, Any] = {
            "content": [{"type": "text", "text": "found"}],
            "structuredContent": {"found": True},
            "isError": False,
        }

    async def __aenter__(self) -> MCPFixture:
        self.server = await asyncio.start_server(self._handle, self.host, 0, ssl=self.tls)
        port = self.server.sockets[0].getsockname()[1]
        self.url = f"{'https' if self.tls else 'http'}://{self.host}:{port}/mcp"
        return self

    async def __aexit__(self, *_args: object) -> None:
        self.server.close()
        for writer in tuple(self.writers):
            writer.close()
        for task in tuple(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self.tasks.add(task)
        self.writers.add(writer)
        try:
            raw = await reader.readuntil(b"\r\n\r\n")
            lines = raw.decode("iso-8859-1").split("\r\n")
            method, target, _ = lines[0].split(" ")
            headers = {
                name.lower(): value.strip()
                for line in lines[1:]
                if line
                for name, value in [line.split(":", 1)]
            }
            body = await reader.readexactly(int(headers.get("content-length", "0")))
            request = json.loads(body) if body else {}
            rpc = request.get("method", method)
            self.requests.append(
                {
                    "method": method,
                    "target": target,
                    "headers": headers,
                    "body": request,
                    "rpc": rpc,
                }
            )
            if rpc == "tools/call":
                self.calls.append(request)
            if rpc == self.block_method:
                self.entered.set()
                await self.unblock.wait()
            if self.status is not None or (
                self.required_header is not None
                and headers.get("authorization") != self.required_header
            ):
                await self._respond(writer, self.status or 401, b"private authentication failure")
                return
            if self.redirect:
                await self._respond(writer, 307, b"", {"Location": self.url + "/redirect"})
                return
            if method == "GET":
                if not self.stateful:
                    await self._respond(writer, 405, b"")
                    return
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    b"Connection: close\r\n\r\n"
                )
                writer.write(
                    b'data: {"jsonrpc":"2.0","method":"notifications/tools/list_changed"}\n\n'
                )
                await writer.drain()
                await reader.read()
                return
            if method == "DELETE":
                self.deleted.append(headers.get("mcp-session-id", ""))
                await self._respond(writer, 200, b"")
                return
            response_headers = {
                "Content-Type": "text/event-stream" if self.sse else "application/json"
            }
            if rpc == "initialize":
                result = {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {"name": "controlled-mcp", "version": "1"},
                }
                if self.stateful:
                    sid = f"private-mcp-session-{len(self.sessions) + 1}"
                    self.sessions.append(sid)
                    response_headers["Mcp-Session-Id"] = sid
                    response_headers["Set-Cookie"] = f"mcp_private={sid}; Path=/"
            elif rpc == "tools/list":
                page = int(request.get("params", {}).get("cursor", "0"))
                result = {"tools": self.pages[min(page, len(self.pages) - 1)]}
                if self.repeated_cursor or page + 1 < len(self.pages):
                    result["nextCursor"] = "1" if self.repeated_cursor else str(page + 1)
            elif rpc == "tools/call":
                if self.drop_call:
                    return
                result = self.result
            else:
                await self._respond(writer, 202, b"")
                return
            encoded = (
                self.raw_response
                or json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}).encode()
            )
            if self.sse:
                encoded = b"event: message\r\ndata: " + encoded + b"\r\n\r\n"
            await self._respond(writer, 200, encoded, response_headers)
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.CancelledError, ssl.SSLError):
            pass
        finally:
            writer.close()
            with suppress(ConnectionError, ssl.SSLError):
                await writer.wait_closed()
            self.writers.discard(writer)
            self.tasks.discard(task)

    @staticmethod
    async def _respond(
        writer: asyncio.StreamWriter,
        status: int,
        body: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        fields = {"Content-Length": str(len(body)), "Connection": "close", **(headers or {})}
        writer.write(
            f"HTTP/1.1 {status} Fixture\r\n".encode()
            + b"".join(f"{key}: {value}\r\n".encode() for key, value in fields.items())
            + b"\r\n"
            + body
        )
        await writer.drain()
