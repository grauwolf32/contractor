"""Standalone mTLS Runtime Agent lifecycle server for Go integration tests."""

from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path

from contractor_runtime.allocation import AllocationService
from contractor_runtime.factories import built_in_factories
from contractor_runtime.mtls import runtime_agent_server_context
from contractor_runtime.server import RuntimeServer, create_app, create_server_config
from contractor_runtime.settings import Settings
from contractor_runtime.state import RuntimeState
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
    )
    cleanup_orphan_workdirs(settings.work_root)
    state = RuntimeState(instance_id="runtime-cross-language")
    await state.mark_registered()
    service = AllocationService(
        state,
        built_in_factories(settings.work_root),
        a2a_base_url=settings.advertised_a2a_url,
    )
    tls_context = runtime_agent_server_context(
        ca_file=settings.ca_file,
        certificate_file=settings.certificate_file,
        private_key_file=settings.private_key_file,
    )
    server = RuntimeServer(
        create_server_config(
            settings,
            create_app(state, allocation_service=service),
            tls_context,
        )
    )
    loop = asyncio.get_running_loop()
    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(handled_signal, setattr, server, "should_exit", True)
    await server.serve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ca", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    asyncio.run(serve(parser.parse_args()))


if __name__ == "__main__":
    main()
