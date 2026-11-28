"""Runtime Agent process entry point and bounded shutdown orchestration."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import time
from collections.abc import Callable, Sequence

from contractor_runtime._grpc_client import ControlRPC
from contractor_runtime._rpc_connections import RPCConnections
from contractor_runtime.allocation import AllocationService
from contractor_runtime.archive_cache import (
    ImmutableArchiveCache,
    archive_trust_domain,
    tls_trust_fingerprint,
)
from contractor_runtime.artifacts import ArtifactClient, MTLSArtifactTransport
from contractor_runtime.capabilities import discover_capabilities
from contractor_runtime.control_client import (
    ControlClient,
    ControlTransport,
    RegistrationRejectedError,
    SharedConfigError,
)
from contractor_runtime.core.async_tasks import join_owned_task
from contractor_runtime.core.target_policy import TargetPolicyConfig
from contractor_runtime.factories import built_in_factories
from contractor_runtime.lease import LeaseWatchdog
from contractor_runtime.log import configure_logging
from contractor_runtime.mtls import (
    log_runtime_certificate_expiry,
    runtime_agent_client_context,
    runtime_agent_server_context,
)
from contractor_runtime.sandbox.podman.lifecycle import PodmanLifecycle
from contractor_runtime.sandbox.podman.workroots import check_root_policy
from contractor_runtime.server import RuntimeServer, create_app, create_server_config
from contractor_runtime.settings import Settings, parse_settings
from contractor_runtime.state import RuntimeState
from contractor_runtime.threads import consume_task_exception, to_thread_until_done
from contractor_runtime.work_root_lock import WorkRootLockError, hold_work_roots
from contractor_runtime.workspace import cleanup_orphan_workdirs

logger = logging.getLogger(__name__)


class _ShutdownDeadline:
    """One process-exit fence, including waits for cancellation settlement."""

    def __init__(
        self,
        grace_seconds: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        force_exit: Callable[[int], object] = os._exit,
    ) -> None:
        self._grace_seconds = grace_seconds
        self._monotonic = monotonic
        self._force_exit = force_exit
        self._deadline: float | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._expired = False

    @property
    def started(self) -> bool:
        return self._deadline is not None

    def start(self) -> float:
        if self._deadline is None:
            self._deadline = self._monotonic() + self._grace_seconds
            self._timer = asyncio.get_running_loop().call_later(self._grace_seconds, self._expire)
        return self._deadline

    def remaining(self) -> float:
        remaining = self.start() - self._monotonic()
        if remaining <= 0:
            self._expire()
            # A real process exit never returns. Keep injected exits from
            # authorizing another cleanup phase after the budget is exhausted.
            raise RuntimeError("Runtime shutdown deadline exceeded")
        return remaining

    def _expire(self) -> None:
        if not self._expired:
            self._expired = True
            logger.error("runtime shutdown deadline exceeded; exiting process")
            self._force_exit(70)

    def close(self) -> None:
        if self._deadline is not None and self._monotonic() >= self._deadline:
            self._expire()
        if self._timer is not None:
            self._timer.cancel()
        if self._expired:
            raise RuntimeError("Runtime shutdown deadline exceeded")


async def serve(
    settings: Settings,
    *,
    state: RuntimeState | None = None,
    transport: ControlTransport | None = None,
    stop_requested: asyncio.Event | None = None,
    server_factory: Callable[..., RuntimeServer] = RuntimeServer,
    install_signal_handlers: bool = True,
) -> None:
    project_root = settings.workspace.work_root if settings.workspace is not None else None
    with hold_work_roots(settings.work_root, project_root):
        shutdown = _ShutdownDeadline(settings.shutdown_grace_seconds)
        task = asyncio.create_task(
            _bootstrap(
                settings,
                state=state,
                transport=transport,
                stop_requested=stop_requested,
                server_factory=server_factory,
                install_signal_handlers=install_signal_handlers,
                shutdown=shutdown,
            ),
            name="runtime-process-owner",
        )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not shutdown.started:
                shutdown.start()
                task.cancel()
            # Later cancellations belong to this waiter, not to the task
            # closing allocation resources and holding the work-root locks.
            await join_owned_task(task)
            consume_task_exception(task)
            raise
        finally:
            shutdown.close()
        logger.info("runtime agent stopped")


async def _bootstrap(
    settings: Settings,
    *,
    state: RuntimeState | None,
    transport: ControlTransport | None,
    stop_requested: asyncio.Event | None,
    server_factory: Callable[..., RuntimeServer],
    install_signal_handlers: bool,
    shutdown: _ShutdownDeadline,
) -> None:
    stop = stop_requested or asyncio.Event()
    runtime_state = state or RuntimeState(allocation_slots=settings.allocation_slots)
    control_rpc = None
    loop = asyncio.get_running_loop()
    handled_signals = (signal.SIGINT, signal.SIGTERM)
    if install_signal_handlers:
        for handled_signal in handled_signals:
            loop.add_signal_handler(handled_signal, stop.set)
    try:
        if transport is None:
            control_rpc = ControlRPC(
                settings.control_plane_url,
                runtime_agent_client_context(
                    ca_file=settings.ca_file,
                    certificate_file=settings.certificate_file,
                    private_key_file=settings.private_key_file,
                ),
                settings.request_timeout_seconds,
            )
            transport = control_rpc
        control = ControlClient(settings, runtime_state, transport)
        if not await control.configure_until_stopped(stop) or stop.is_set():
            return
        await _serve_configured(
            settings,
            state=runtime_state,
            transport=transport,
            stop_requested=stop,
            server_factory=server_factory,
            shutdown=shutdown,
        )
    finally:
        shutdown.start()
        if install_signal_handlers:
            for handled_signal in handled_signals:
                loop.remove_signal_handler(handled_signal)
        if control_rpc is not None:
            await control_rpc.aclose()


async def _serve_configured(
    settings: Settings,
    *,
    state: RuntimeState,
    transport: ControlTransport,
    stop_requested: asyncio.Event,
    server_factory: Callable[..., RuntimeServer],
    shutdown: _ShutdownDeadline,
) -> None:
    lifecycle = PodmanLifecycle(settings.podman) if settings.podman.enabled else None
    project_root = settings.workspace.work_root if settings.workspace is not None else None
    archive_cache = None
    try:
        roots = [settings.work_root]
        if project_root is not None:
            roots.append(project_root)
        for root in roots:
            await to_thread_until_done(
                check_root_policy,
                root,
                settings.podman.owner if lifecycle is not None else None,
                name="runtime-root-policy",
            )
        if lifecycle is not None:
            await lifecycle.recover(deadline=time.monotonic() + settings.podman.prepare_max_seconds)
        if project_root is not None and settings.workspace.storage == "local":
            archive_cache = await ImmutableArchiveCache.open(project_root)
        await _serve(
            settings,
            state=state,
            transport=transport,
            stop_requested=stop_requested,
            server_factory=server_factory,
            lifecycle=lifecycle,
            archive_cache=archive_cache,
            shutdown=shutdown,
        )
    finally:
        shutdown.start()
        try:
            if archive_cache is not None:
                await archive_cache.close()
        finally:
            if lifecycle is not None:
                # EOF still delegates cleanup to the surviving owner if this
                # bounded graceful close cannot confirm it. Never kill the owner.
                await lifecycle.close(deadline=shutdown.start())


async def _serve(
    settings: Settings,
    *,
    state: RuntimeState,
    transport: ControlTransport,
    stop_requested: asyncio.Event,
    server_factory: Callable[..., RuntimeServer],
    lifecycle: PodmanLifecycle | None,
    archive_cache: ImmutableArchiveCache | None,
    shutdown: _ShutdownDeadline,
) -> None:
    runtime_state = state
    stop = stop_requested
    await to_thread_until_done(
        cleanup_orphan_workdirs, settings.work_root, name="runtime-orphan-cleanup"
    )
    log_runtime_certificate_expiry(
        settings.certificate_file,
        ca_file=settings.ca_file,
        warning_days=settings.certificate_expiry_warning_days,
    )
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
    # The process owns its private connections and closes them last (CTL-R164):
    # registration and heartbeats never wait behind Artifact transfers.
    bulk_connections = RPCConnections(outgoing_tls, bulk=True)
    allocation_connections = RPCConnections(
        runtime_agent_client_context(
            ca_file=settings.ca_file,
            certificate_file=settings.certificate_file,
            private_key_file=settings.private_key_file,
        )
    )
    factories = None
    try:
        trust_fingerprint = await to_thread_until_done(
            tls_trust_fingerprint, outgoing_tls, name="archive-cache-trust"
        )
        factories = built_in_factories(
            settings.work_root,
            artifact_client_factory=lambda allocation_id, runtime_settings: ArtifactClient(
                allocation_id,
                MTLSArtifactTransport(
                    runtime_settings.artifact_api_url,
                    allocation_connections,
                    bulk_connections,
                    runtime_settings.request_timeout_seconds,
                    runtime_state.instance_id,
                ),
                archive_cache=archive_cache,
                archive_trust_domain=archive_trust_domain(
                    runtime_settings.artifact_api_url, trust_fingerprint
                ),
            ),
            enabled_runtime_adapters=settings.enabled_runtime_adapters,
            workspace_settings=settings.workspace,
            target_policy=_target_policy(settings),
            execution_lifecycle=lifecycle,
        )
        allocation_service = AllocationService(
            runtime_state,
            factories,
            a2a_base_url=settings.advertised_a2a_url,
            confirmed_control_deadline=lambda: watchdog.confirmed_deadline,
            private_bypass_urls=(
                settings.control_plane_url,
                settings.advertised_control_url,
                settings.advertised_a2a_url,
            ),
        )
        watchdog = LeaseWatchdog(
            lambda: allocation_service.expire_control_lease(settings.shutdown_grace_seconds)
        )
        if lifecycle is not None:
            lifecycle.bind_health(lambda: watchdog.confirmed_deadline, stop.set)
        control = ControlClient(
            settings,
            runtime_state,
            transport,
            watchdog=watchdog,
            reconciliation=allocation_service,
        )
        application = create_app(runtime_state, allocation_service=allocation_service)
        server = server_factory(
            create_server_config(
                settings,
                application,
                incoming_tls,
                state=runtime_state,
                allocation_service=allocation_service,
            )
        )

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
            shutdown.start()
            stop.set()
            for task in (registration_task, heartbeat_task):
                if task is not None and not task.done():
                    task.cancel()
            for task in (registration_task, heartbeat_task, stop_task):
                if task is not None:
                    # A failure, such as a permanent registration rejection, was
                    # already raised from the main wait path; awaiting it again
                    # must not skip the listener cleanup below.
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
            if watchdog_task is not None:
                # Any watchdog failure was already raised from the main wait path;
                # do not let a second await skip listener cleanup in this finally.
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await watchdog_task
            # Every profile owns an in-process Worker and tool resources. Stop them
            # while Artifact connections and the private listener still exist.
            with contextlib.suppress(Exception):
                await allocation_service.close(shutdown.remaining())
            await runtime_state.begin_stopping()
            server.should_exit = True
            await server_task
    finally:
        shutdown.start()
        try:
            if factories is not None and factories.http_transport_factory is not None:
                await factories.http_transport_factory.close()
        finally:
            try:
                await allocation_connections.aclose()
            finally:
                await bulk_connections.aclose()


def _target_policy(settings: Settings) -> TargetPolicyConfig:
    """Runtime-owned endpoints plus operator networks for model-selected targets."""

    listen_host = f"[{settings.host}]" if ":" in settings.host else settings.host
    protected = [
        settings.control_plane_url,
        settings.advertised_control_url,
        settings.advertised_a2a_url,
    ]
    if settings.port:
        protected.append(f"https://{listen_host}:{settings.port}")
    return TargetPolicyConfig(
        protected_urls=tuple(protected),
        allowed_networks=settings.allowed_target_networks,
    )


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
    except (WorkRootLockError, RegistrationRejectedError, SharedConfigError) as error:
        # These messages contain only safe, bounded fields for the operator.
        logger.error("runtime agent failed: %s", error)
        raise SystemExit(1) from None
    except Exception as error:
        logger.error("runtime agent failed (%s)", type(error).__name__)
        raise SystemExit(1) from None
