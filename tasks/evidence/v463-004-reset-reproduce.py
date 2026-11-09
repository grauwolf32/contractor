"""Isolated real TCP reset through the production HTTP bridge; no LLM or demo writes."""

import asyncio
import json
import socket
import struct
import time
import traceback
from pathlib import Path

from contractor_runtime.adapters.bridge.mitmproxy import engine as module
from contractor_runtime.contracts import SharedConfig, decode_private
from contractor_runtime.shared_config import install_shared_config
from fakes.engagement_socket import SocketTarget, preflight

install_shared_config(
    decode_private(
        SharedConfig,
        (
            Path(__file__).resolve().parents[2]
            / "runtime/tests/testdata/shared-config.json"
        ).read_bytes(),
    ).snapshot()
)


async def scenario():
    reset = asyncio.Event()
    reset_sent = asyncio.Event()

    async def respond(target, reader, writer, message):
        writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 2\r\n\r\n{}")
        await writer.drain()
        await reset.wait()
        writer.get_extra_info("socket").setsockopt(
            socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
        )
        writer.transport.abort()
        reset_sent.set()

    result = {}
    async with SocketTarget(respond) as target, preflight(target) as (owner, _):
        owner.configure(authority_deadline=lambda: time.monotonic() + 60)
        outcome, response = await owner.request(
            target.origin + "/allowed", method="POST", body=b'{"amount":1}'
        )
        result["request"] = {"outcome": outcome.kind, "status": response.status}
        engine = owner._engine
        pool = next(iter(engine.pools.values()))
        io = next(io for io in pool.transports.values() if io.writer is not None)
        handler = io.handler
        writer = io.writer.inner
        reset.set()
        await reset_sent.wait()
        finished = await asyncio.gather(handler, return_exceptions=True)
        result["native_handler_exception"] = (
            type(finished[0]).__name__
            if isinstance(finished[0], BaseException)
            else None
        )
        if isinstance(finished[0], BaseException):
            result["native_handler_traceback"] = "".join(
                traceback.format_exception(finished[0])
            )
        result["socket_after_handler"] = {
            "is_closing": writer.is_closing(),
            "fileno": writer.get_extra_info("socket").fileno(),
            "reader_at_eof": io.reader.inner.at_eof(),
        }
        failures = []
        for _ in range(2):
            try:
                await writer.wait_closed()
            except ConnectionResetError as e:
                failures.append(e)
        result["repeated_wait_closed"] = [type(e).__name__ for e in failures]
        result["same_exception_object"] = (
            len(failures) == 2 and failures[0] is failures[1]
        )
        result["retained_after_reset"] = {
            "writers": len(pool.writers),
            "transports": len(pool.transports),
            "parsers": len(pool.parsers),
        }
        result["close_attempts"] = []
        for _ in range(2):
            try:
                await owner.close()
                result["close_attempts"].append("succeeded")
            except ConnectionResetError as e:
                result["close_attempts"].append(type(e).__name__)
                result["cleanup_traceback"] = "".join(traceback.format_exception(e))
        result["engine_closed_after_retries"] = engine.closed
        result["exclusive_process_owner_retained"] = module._active is engine
        # Experiment only: verify the classification hypothesis without editing production.
        # This real socket is already closed. Keep the original close-wait join,
        # handling only its observed reset; all other failures still escape.
        original = writer.wait_closed

        async def classified_wait():
            try:
                await original()
            except ConnectionResetError:
                if writer.get_extra_info("socket").fileno() != -1:
                    raise

        writer.wait_closed = classified_wait
        await owner.close()
        result["after_closed_reset_classification"] = {
            "engine_closed": engine.closed,
            "exclusive_owner_released": module._active is None,
            "writers": len(pool.writers),
        }
    print(
        json.dumps({k: v for k, v in result.items() if "traceback" not in k}, indent=2)
    )


async def main():
    async with asyncio.timeout(20):
        await scenario()


asyncio.run(main())
