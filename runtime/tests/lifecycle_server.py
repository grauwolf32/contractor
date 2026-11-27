"""Standalone mTLS Runtime Agent lifecycle server for Go integration tests."""

from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path

from contractor_runtime.allocation import AllocationService
from contractor_runtime.capabilities import discover_capabilities
from contractor_runtime.factories import built_in_factories
from contractor_runtime.mtls import runtime_agent_server_context
from contractor_runtime.server import RuntimeServer, create_app, create_server_config
from contractor_runtime.settings import Settings
from contractor_runtime.state import RuntimeState
from contractor_runtime.telemetry.resources import ProcessReading, ResourceCollector
from contractor_runtime.workspace import cleanup_orphan_workdirs


async def serve(args: argparse.Namespace) -> None:
    advertised = f"https://127.0.0.1:{args.port}"
    settings = Settings(
        control_plane_url="https://127.0.0.1:1",
        advertised_control_url=advertised,
        advertised_a2a_url=advertised,
        ca_file=args.ca,
        certificate_file=args.certificate,
        private_key_file=args.private_key,
        host="127.0.0.1",
        port=args.port,
        work_root=args.work_root,
        allocation_slots=1,
    )
    # Component probes bypass the production CLI bootstrap deliberately.
    from contractor_runtime.contracts import SharedConfig, decode_private
    from contractor_runtime.shared_config import install_shared_config

    install_shared_config(
        decode_private(
            SharedConfig, (Path(__file__).parent / "testdata/shared-config.json").read_bytes()
        ).snapshot()
    )
    cleanup_orphan_workdirs(settings.work_root)
    factories = built_in_factories(settings.work_root)
    faults = None
    if args.fault_mode is not None:
        from fakes.adk_process_faults import NativeFaults

        if args.fault_control is None:
            raise ValueError("fault mode requires an owned control directory")
        faults = NativeFaults(args.fault_mode, args.fault_control)
        factories = faults.factories(factories)
    capabilities = await discover_capabilities(factories)
    state = RuntimeState(
        instance_id="runtime-cross-language",
        capabilities=capabilities,
        allocation_slots=settings.allocation_slots,
    )
    await state.mark_registered()
    service = AllocationService(
        state,
        factories,
        capabilities,
        a2a_base_url=settings.advertised_a2a_url,
        resource_collector_factory=lambda: ResourceCollector(
            reader=lambda _: ProcessReading(1.0, 0.5, 4096)
        ),
    )
    tls_context = runtime_agent_server_context(
        ca_file=settings.ca_file,
        certificate_file=settings.certificate_file,
        private_key_file=settings.private_key_file,
    )
    app = create_app(state, allocation_service=service)
    server = RuntimeServer(
        create_server_config(
            settings,
            app,
            tls_context,
            state=state,
            allocation_service=service,
        )
    )
    loop = asyncio.get_running_loop()
    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(handled_signal, setattr, server, "should_exit", True)
    if faults is not None:
        await faults.start(service)
    try:
        await server.serve()
    finally:
        if faults is not None:
            await faults.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ca", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--fault-mode")
    parser.add_argument("--fault-control", type=Path)
    args = parser.parse_args()
    if args.fault_mode is None:
        asyncio.run(serve(args))
    else:
        from fakes.adk_process_faults import MODES
        from fakes.clock import run_with_advanceable_clock

        if args.fault_mode not in MODES:
            parser.error("unsupported native ADK fault mode")
        run_with_advanceable_clock(serve(args))


if __name__ == "__main__":
    main()
