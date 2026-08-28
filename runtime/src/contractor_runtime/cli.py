"""Runtime Agent command-line entry point."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from collections.abc import Generator, Sequence

import uvicorn

from contractor_runtime.app import create_app
from contractor_runtime.log import configure_logging
from contractor_runtime.settings import Settings, parse_settings

logger = logging.getLogger(__name__)


class RuntimeServer(uvicorn.Server):
    """Uvicorn server whose signals are coordinated by the asyncio entry point."""

    @contextlib.contextmanager
    def capture_signals(self) -> Generator[None]:
        yield


async def serve(settings: Settings) -> None:
    config = uvicorn.Config(
        create_app(),
        host=settings.host,
        port=settings.port,
        access_log=False,
        log_config=None,
    )
    server = RuntimeServer(config)
    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    handled_signals = (signal.SIGINT, signal.SIGTERM)
    for handled_signal in handled_signals:
        loop.add_signal_handler(handled_signal, stop_requested.set)

    logger.info("runtime agent listening on %s", settings.listen_address)
    server_task = asyncio.create_task(server.serve(), name="runtime-http-server")
    stop_task = asyncio.create_task(stop_requested.wait(), name="runtime-stop-signal")
    try:
        done, _ = await asyncio.wait({server_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        if stop_task in done and not server_task.done():
            server.should_exit = True
        await server_task
        if not server.started:
            raise RuntimeError("runtime HTTP server failed to start")
    finally:
        stop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stop_task
        for handled_signal in handled_signals:
            loop.remove_signal_handler(handled_signal)
        logger.info("runtime agent stopped")


def main(argv: Sequence[str] | None = None) -> None:
    settings = parse_settings(argv)
    configure_logging(settings.log_level)
    try:
        asyncio.run(serve(settings))
    except KeyboardInterrupt:
        logger.info("runtime agent interrupted")
