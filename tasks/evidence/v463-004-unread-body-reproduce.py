"""Real kernel reset from an early HTTP denial, without SO_LINGER or mocked I/O."""

import asyncio
import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from contractor_runtime.adapters.bridge.mitmproxy import engine as module
from contractor_runtime.contracts import SharedConfig, decode_private
from contractor_runtime.shared_config import install_shared_config
from fakes.engagement_socket import preflight, private_interface

install_shared_config(
    decode_private(
        SharedConfig,
        (
            Path(__file__).resolve().parents[2]
            / "runtime/tests/testdata/shared-config.json"
        ).read_bytes(),
    ).snapshot()
)


async def main():
    observed = {}
    failures = []
    listener = socket.socket()
    listener.settimeout(5)
    listener.bind((private_interface(), 0))
    listener.listen(1)
    address, port = listener.getsockname()
    target = SimpleNamespace(
        address=address, port=port, origin=f"http://target.example:{port}"
    )

    def reject():
        try:
            connection, _ = listener.accept()
            connection.settimeout(5)
            with connection:
                headers = b""
                while not headers.endswith(b"\r\n\r\n"):
                    headers += connection.recv(1)
                observed["unread_body_present"] = bool(
                    connection.recv(1, socket.MSG_PEEK)
                )
                connection.sendall(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}"
                )
                # Closing a socket with unread inbound data is the only reset trigger.
            observed["target_closed"] = True
        except OSError as error:
            observed["target_error"] = type(error).__name__
        finally:
            listener.close()

    thread = threading.Thread(target=reject, daemon=True)
    thread.start()
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    loop.set_exception_handler(
        lambda _, ctx: failures.append(
            {
                "message": ctx.get("message"),
                "exception": type(ctx.get("exception")).__name__,
            }
        )
    )
    try:
        async with preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            try:
                outcome, response = await owner.request(
                    target.origin + "/allowed", method="POST", body=b'{"amount":1}'
                )
                observed["request"] = {
                    "outcome": outcome.kind,
                    "status": response.status if response else None,
                }
            except ConnectionResetError as error:
                observed["request_exception"] = type(error).__name__
            await asyncio.to_thread(thread.join, 5)
            engine = owner._engine
            try:
                await owner.close()
                observed["cleanup"] = "succeeded"
            except ConnectionResetError as error:
                observed["cleanup"] = type(error).__name__
            observed["retained_closed_writers"] = []
            for pool in engine.pools.values():
                for writer in list(pool.writers):
                    inner = getattr(writer, "inner", writer)
                    observed["retained_closed_writers"].append(
                        {
                            "is_closing": inner.is_closing(),
                            "fileno": inner.get_extra_info("socket").fileno(),
                        }
                    )
                    original = inner.wait_closed

                    async def classified_wait(original=original, inner=inner):
                        try:
                            await original()
                        except ConnectionResetError:
                            if inner.get_extra_info("socket").fileno() != -1:
                                raise

                    inner.wait_closed = classified_wait
            await owner.close()
            observed["classification_cleared_owner"] = module._active is None
    finally:
        loop.set_exception_handler(previous)
    observed["unhandled_native_tasks"] = failures

    print(json.dumps(observed, indent=2))


async def bounded():
    async with asyncio.timeout(20):
        await main()


asyncio.run(bounded())
