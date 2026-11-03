"""Bounded, allocation-private state of one native target exchange."""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass, field
from typing import Any

from contractor_runtime.adapters.http_session import PrivateResponse


class EngineFailure(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(eq=False, repr=False)
class Exchange:
    url: str
    boundary: Any
    request: Any
    addresses: tuple
    body_limit: int
    future: asyncio.Future
    flow: Any = None
    stream_id: int = 0
    connection: Any = None
    reason: str = ""
    exception: Exception | None = None
    response: PrivateResponse | None = None
    buffer: io.BytesIO = field(default_factory=io.BytesIO)
    hooks: set = field(default_factory=set)
    truncated: bool = False

    def receive(self, data: bytes) -> bool:
        available = self.body_limit - self.buffer.tell()
        self.buffer.write(data[:available])
        self.truncated = len(data) > available
        return self.truncated

    def finish(self) -> None:
        if self.future is None or self.future.done():
            return
        if self.exception is not None:
            self.future.set_exception(self.exception)
        elif self.response is not None and (not self.reason or self.truncated):
            self.response.body = self.buffer.getvalue()
            self.response.truncated = self.truncated
            self.future.set_result(self.response)
        else:
            self.future.set_exception(EngineFailure(self.reason or "response_invalid"))

    def clear(self) -> None:
        self.buffer.close()
        if self.flow is not None:
            if self.flow.request is not None:
                self.flow.request.headers.clear()
                self.flow.request.raw_content = b""
            self.flow.response = None
        if self.request is not None:
            self.request.headers.clear()
            self.request.body = b""
        self.flow = self.request = self.connection = None
        self.response = self.exception = self.future = None
