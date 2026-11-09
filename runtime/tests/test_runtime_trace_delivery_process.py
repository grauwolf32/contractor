"""Observe received protobuf spans across actual process exit, including exit 70."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from fakes.runtime_trace_delivery import SECRET
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest


@pytest.mark.parametrize("operation", ["finalize", "abort", "lease", "reconcile"])
@pytest.mark.parametrize("fault", ["none", "worker", "http", "tool", "execution"])
def test_worker_budget_failure_exports_before_normal_or_forced_exit(tmp_path, operation, fault):
    async def scenario():
        received = []
        handlers = set()
        errors = []

        async def collect(reader, writer):
            task = asyncio.current_task()
            handlers.add(task)
            try:
                async with asyncio.timeout(5):
                    header = await reader.readuntil(b"\r\n\r\n")
                    length = next(
                        int(line.split(b":", 1)[1])
                        for line in header.split(b"\r\n")
                        if line.lower().startswith(b"content-length:")
                    )
                    payload = await reader.readexactly(length)
                    request = ExportTraceServiceRequest.FromString(payload)
                    received.extend(
                        span
                        for resource in request.resource_spans
                        for scope in resource.scope_spans
                        for span in scope.spans
                    )
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/x-protobuf\r\n"
                        b"Content-Length: 0\r\nConnection: close\r\n\r\n"
                    )
                    await writer.drain()
            except Exception as error:
                errors.append(error)
            finally:
                writer.close()
                await writer.wait_closed()
                handlers.discard(task)

        server = await asyncio.start_server(collect, "127.0.0.1", 0)
        try:
            endpoint = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/traces"
            runtime = Path(__file__).resolve().parents[1]
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join([str(runtime / "src"), str(runtime / "tests")])
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "fakes.runtime_trace_delivery",
                endpoint,
                str(tmp_path / "allocation"),
                operation,
                fault,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20)
            except BaseException:
                process.kill()
                await process.wait()
                raise
            if handlers:
                await asyncio.wait_for(asyncio.gather(*tuple(handlers)), timeout=5)
            assert process.returncode == (0 if fault == "none" else 70), stderr.decode()
            events = [json.loads(line) for line in stdout.splitlines()]
            assert events[0]["failure"] == "worker_budget_exhausted"
            assert not errors
            assert len(received) == events[0]["queued"]
            assert len({span.span_id for span in received}) == len(received)
            assert any(span.name == "contractor.worker.error" for span in received)
            assert any(
                attr.key == "error.type" and attr.value.string_value == "worker_budget_exhausted"
                for span in received
                for attr in span.attributes
            )
            assert SECRET.encode() not in stderr
            assert b"private.invalid" not in stderr
            if fault != "none":
                assert f"phase={fault}" in stderr.decode()
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())
