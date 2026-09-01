"""Runtime Agent process entry point and bounded shutdown orchestration."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from collections.abc import Callable, Sequence

from contractor_runtime.allocation import AllocationService
from contractor_runtime.artifacts import ArtifactClient, MTLSArtifactTransport
from contractor_runtime.capabilities import discover_capabilities
from contractor_runtime.control_client import ControlClient, ControlTransport, MTLSJSONTransport
from contractor_runtime.factories import built_in_factories
from contractor_runtime.lease import LeaseWatchdog
from contractor_runtime.log import configure_logging
from contractor_runtime.mtls import runtime_agent_client_context, runtime_agent_server_context
from contractor_runtime.server import RuntimeServer, create_app, create_server_config
from contractor_runtime.settings import Settings, parse_settings
from contractor_runtime.state import RuntimeState
from contractor_runtime.workspace import cleanup_orphan_workdirs

logger = logging.getLogger(__name__)


async def serve(
    settings: Settings,
    *,
    state: RuntimeState | None = None,
    transport: ControlTransport | None = None,
    stop_requested: asyncio.Event | None = None,
    server_factory: Callable[..., RuntimeServer] = RuntimeServer,
    install_signal_handlers: bool = True,
) -> None:
    runtime_state = state or RuntimeState()
    stop = stop_requested or asyncio.Event()
    cleanup_orphan_workdirs(settings.work_root)
    outgoing_tls = runtime_agent_client_context(
        ca_file=settings.ca_file,
        certificate_file=settings.certificate_file,
        private_key_file=settings.private_key_file,
    )
    incoming_tls = runtime_agent_server_context(
        ca_file=settings.ca_file,
        certificate_file=settings.certificate_file,
        private_key_file=settings.private_key_file,
    )
    control_transport = transport or MTLSJSONTransport(
        settings.control_plane_url, outgoing_tls, settings.request_timeout_seconds
    )
    factories = built_in_factories(
        settings.work_root,
        artifact_client_factory=lambda allocation_id, runtime_settings: ArtifactClient(
            allocation_id,
            MTLSArtifactTransport(
                runtime_settings.artifact_api_url,
                outgoing_tls,
                runtime_settings.request_timeout_seconds,
                runtime_state.instance_id,
            ),
        ),
    )
    allocation_service = AllocationService(
        runtime_state,
        factories,
        a2a_base_url=settings.advertised_a2a_url,
    )
    watchdog = LeaseWatchdog(
        lambda: allocation_service.expire_control_lease(settings.shutdown_grace_seconds)
    )
    control = ControlClient(
        settings,
        runtime_state,
        control_transport,
        watchdog=watchdog,
        reconciliation=allocation_service,
    )
    application = create_app(runtime_state, allocation_service=allocation_service)
    server = server_factory(create_server_config(settings, application, incoming_tls))

    loop = asyncio.get_running_loop()
    handled_signals = (signal.SIGINT, signal.SIGTERM)
    if install_signal_handlers:
        for handled_signal in handled_signals:
            loop.add_signal_handler(handled_signal, stop.set)

    server_task = asyncio.create_task(server.serve(), name="runtime-private-server")
    registration_task: asyncio.Task[bool] | None = None
    heartbeat_task: asyncio.Task[None] | None = None
    watchdog_task: asyncio.Task[None] | None = None
    stop_task = asyncio.create_task(stop.wait(), name="runtime-stop-signal")
    try:
        await _wait_until_listening(server, server_task, settings.request_timeout_seconds)
        logger.info("runtime agent private listener is accepting")
        capabilities = await discover_capabilities(factories)
        await runtime_state.install_capabilities(capabilities)
        registration_task = asyncio.create_task(
            control.register_until_stopped(stop), name="runtime-registration"
        )
        done, _ = await asyncio.wait(
            {registration_task, server_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if server_task in done:
            await server_task
            raise RuntimeError("runtime private server stopped before shutdown")
        if stop_task in done:
            return
        if not await registration_task:
            return
        logger.info("runtime agent registered", extra={"instanceId": runtime_state.instance_id})
        heartbeat_task = asyncio.create_task(
            control.run_heartbeats(stop), name="runtime-heartbeats"
        )
        watchdog_task = asyncio.create_task(watchdog.run(stop), name="runtime-lease-watchdog")
        done, _ = await asyncio.wait(
            {heartbeat_task, watchdog_task, server_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if server_task in done:
            await server_task
            raise RuntimeError("runtime private server stopped before shutdown")
        if heartbeat_task in done:
            await heartbeat_task
            raise RuntimeError("runtime heartbeat loop stopped before shutdown")
        if watchdog_task in done:
            await watchdog_task
            raise RuntimeError("runtime lease watchdog stopped before shutdown")
    finally:
        stop.set()
        for task in (registration_task, heartbeat_task):
            if task is not None and not task.done():
                task.cancel()
        for task in (registration_task, heartbeat_task, stop_task):
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if watchdog_task is not None:
            # Any watchdog failure was already raised from the main wait path;
            # do not let a second await skip listener cleanup in this finally.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await watchdog_task
        await runtime_state.begin_stopping()
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=settings.shutdown_grace_seconds)
        except TimeoutError:
            server.force_exit = True
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server_task
        if install_signal_handlers:
            for handled_signal in handled_signals:
                loop.remove_signal_handler(handled_signal)
        logger.info("runtime agent stopped")


async def _wait_until_listening(
    server: RuntimeServer,
    server_task: asyncio.Task[None],
    timeout_seconds: float,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while not server.started:
        if server_task.done():
            await server_task
            raise RuntimeError("runtime private listener failed to start")
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("runtime private listener readiness timed out")
        await asyncio.sleep(0.01)


def main(argv: Sequence[str] | None = None) -> None:
    settings = parse_settings(argv)
    configure_logging(settings.log_level)
    try:
        asyncio.run(serve(settings))
    except KeyboardInterrupt:
        logger.info("runtime agent interrupted")
    except Exception as error:
        logger.error("runtime agent failed (%s)", type(error).__name__)
        raise SystemExit(1) from None
