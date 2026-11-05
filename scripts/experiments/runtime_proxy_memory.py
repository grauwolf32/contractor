# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = []
# ///
"""Opt-in RSS measurements; children use the locked Runtime plus a proxy overlay.

These are import and local HTTP probes, not registered or live-model allocations.
No production dependency/source is changed. No bodies or certificates are retained.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import tempfile
import weakref
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = "12.3.1"
BODY_BYTES = 8 * 1024 * 1024
CONCURRENCY = 4
ROUNDS = 3
IMPORT_CASES = ("transport", "toolworker", "allocation", "cli", "mitm_min", "mitm_dump", "cli_mitm")
HTTP_CASES = ("direct", "proxy_buffered", "proxy_streamed")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def mark(stage):
    import psutil

    gc.collect()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if platform.system() == "Darwin" else peak * 1024
    return {
        "stage": stage,
        "rss_mib": round(psutil.Process().memory_info().rss / 1048576, 2),
        "peak_rss_mib": round(peak_bytes / 1048576, 2),
        "modules": len(sys.modules),
        "adk_loaded": "google.adk" in sys.modules,
        "mitmproxy_loaded": "mitmproxy" in sys.modules,
    }


def import_minimal_proxy():
    for name in (
        "mitmproxy.master",
        "mitmproxy.options",
        "mitmproxy.addons.proxyserver",
        "mitmproxy.addons.next_layer",
        "mitmproxy.addons.tlsconfig",
    ):
        importlib.import_module(name)


def imports(case):
    rows = [mark("python")]
    modules = {
        "transport": "contractor_runtime.adapters.http_transport",
        "toolworker": "contractor_runtime.worker.tool_runtime",
        "allocation": "contractor_runtime.allocation.service",
        "cli": "contractor_runtime.cli",
        "cli_mitm": "contractor_runtime.cli",
    }
    if case in modules:
        importlib.import_module(modules[case])
        rows.append(mark("runtime_import"))
    if case in {"mitm_min", "cli_mitm"}:
        import_minimal_proxy()
        rows.append(mark("proxy_import"))
    elif case == "mitm_dump":
        importlib.import_module("mitmproxy.tools.dump")
        rows.append(mark("proxy_import"))
    return {"stages": rows}


class Barrier:
    """Hold completed exchanges at the same logical stage for RSS sampling."""

    def __init__(self):
        self.values = []
        self.ready = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.departed = 0

    async def arrive(self, retained_bytes):
        self.values.append(retained_bytes)
        if len(self.values) == CONCURRENCY:
            self.ready.set()
        await self.release.wait()
        self.departed += 1
        if self.departed == CONCURRENCY:
            self.finished.set()


class Origin:
    def __init__(self):
        self.tasks = set()
        self.writers = set()
        self.requests = 0
        self.server = None

    async def start(self):
        self.server = await asyncio.start_server(self.accept, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}/body"

    def accept(self, reader, writer):
        task = asyncio.create_task(self.serve(reader, writer))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def serve(self, reader, writer):
        self.writers.add(writer)
        try:
            async with asyncio.timeout(30):
                await reader.readuntil(b"\r\n\r\n")
                self.requests += 1
                writer.write(
                    f"HTTP/1.1 200 OK\r\nContent-Length: {BODY_BYTES}\r\n"
                    "Content-Type: application/octet-stream\r\nConnection: close\r\n\r\n".encode()
                )
                chunk = b"x" * 65536
                for _ in range(BODY_BYTES // len(chunk)):
                    writer.write(chunk)
                    await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            self.writers.discard(writer)

    async def close(self):
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        for writer in tuple(self.writers):
            writer.close()
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class Probe:
    def __init__(self):
        self.running_event = asyncio.Event()
        self.barrier = None
        self.flows = []

    def running(self):
        self.running_event.set()

    async def response(self, flow):
        self.flows.append(weakref.ref(flow))
        await self.barrier.arrive(len(flow.response.raw_content or b""))


class Proxy:
    def __init__(self, directory, streamed):
        from mitmproxy import ctx, master, options
        from mitmproxy.addons import next_layer, proxyserver, tlsconfig

        self.ctx = ctx
        self.previous = {name: getattr(ctx, name, None) for name in ("master", "log", "options")}
        opts = options.Options(listen_host="127.0.0.1", listen_port=0, confdir=str(directory))
        self.master = master.Master(opts, with_termlog=False)
        self.server = proxyserver.Proxyserver()
        self.probe = Probe()
        self.master.addons.add(
            self.server, next_layer.NextLayer(), tlsconfig.TlsConfig(), self.probe
        )
        opts.update(
            connection_strategy="lazy",
            body_size_limit="16m",
            stream_large_bodies="1m" if streamed else None,
            store_streamed_bodies=False,
        )
        self.task = None

    async def start(self):
        self.task = asyncio.create_task(self.master.run())
        await asyncio.wait_for(self.probe.running_event.wait(), 10)
        require(not self.task.done(), "proxy exited before workload")
        host, port = self.server.listen_addrs()[0]
        return f"http://{host}:{port}"

    async def close(self):
        from contractor_runtime.core.async_tasks import join_owned_task

        async def cleanup():
            if self.task is not None:
                self.master.shutdown()
                await self.task
            await self.server.servers.update([])
            tasks = {
                transport.handler
                for handler in tuple(self.server.connections.values())
                for transport in handler.transports.values()
                if transport.handler is not None and not transport.handler.done()
            }
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for name, value in self.previous.items():
                if value is None:
                    self.ctx.__dict__.pop(name, None)
                else:
                    setattr(self.ctx, name, value)
            require(not self.server.connections, "proxy connections survived cleanup")

        task = asyncio.create_task(cleanup())
        cancelled = await join_owned_task(task)
        task.result()
        if cancelled:
            raise asyncio.CancelledError


async def workload(case):
    rows = [mark("python")]
    importlib.import_module("contractor_runtime.cli")
    rows.append(mark("runtime_import"))
    import httpx

    origin = Origin()
    proxy = None
    jobs = []
    barrier = None
    rounds = []
    flow_refs = []
    with tempfile.TemporaryDirectory(prefix="contractor-rss-proxy-") as temporary:
        try:
            url = await origin.start()
            rows.append(mark("fixture_ready"))
            proxy_url = None
            if case != "direct":
                import_minimal_proxy()
                rows.append(mark("proxy_import"))
                proxy = Proxy(Path(temporary), case == "proxy_streamed")
                proxy_url = await proxy.start()
            async with httpx.AsyncClient(
                proxy=proxy_url,
                trust_env=False,
                timeout=30,
                limits=httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=0),
            ) as client:
                rows.append(mark("ready"))
                for index in range(ROUNDS):
                    barrier = Barrier()
                    if proxy is not None:
                        proxy.probe.barrier = barrier

                    async def request(barrier=barrier):
                        count = 0
                        async with client.stream("GET", url) as response:
                            require(response.status_code == 200, "unexpected HTTP status")
                            async for data in response.aiter_raw():
                                count += len(data)
                        require(count == BODY_BYTES, "incomplete response transfer")
                        if proxy is None:
                            await barrier.arrive(0)
                        return count

                    jobs = [asyncio.create_task(request()) for _ in range(CONCURRENCY)]
                    await asyncio.wait_for(barrier.ready.wait(), 30)
                    sample = mark(f"round_{index + 1}_responses_held")
                    sample["retained_response_bytes"] = sum(barrier.values)
                    expected = CONCURRENCY * BODY_BYTES if case == "proxy_buffered" else 0
                    require(sum(barrier.values) == expected, "unexpected body retention")
                    barrier.release.set()
                    counts = await asyncio.gather(*jobs)
                    await asyncio.wait_for(barrier.finished.wait(), 10)
                    jobs = []
                    sample["transferred_bytes"] = sum(counts)
                    rounds.append(sample)
                    rows.append(mark(f"round_{index + 1}_completed"))
        finally:
            if barrier is not None:
                barrier.release.set()
            for job in jobs:
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            if proxy is not None:
                flow_refs = proxy.probe.flows
                await proxy.close()
                proxy = None
            await origin.close()
        rows.append(mark("cleanup"))
    require(origin.requests == ROUNDS * CONCURRENCY, "incorrect target request count")
    require(not origin.tasks and not origin.writers, "origin resources survived cleanup")
    live_flows = sum(reference() is not None for reference in flow_refs)
    require(live_flows == 0, "proxy flows survived cleanup")
    return {
        "stages": rows,
        "rounds": rounds,
        "target_requests": origin.requests,
        "fixture_cleanup_confirmed": True,
        "live_proxy_flows_after_cleanup": live_flows,
    }


def child(case):
    require(importlib.metadata.version("mitmproxy") == VERSION, "unexpected proxy version")
    result = imports(case) if case in IMPORT_CASES else asyncio.run(workload(case))
    return {
        "case": case,
        "python": platform.python_version(),
        "mitmproxy": importlib.metadata.version("mitmproxy"),
        "google_adk": importlib.metadata.version("google-adk"),
        "psutil": importlib.metadata.version("psutil"),
        **result,
    }


def run_child(case):
    command = [
        "uv",
        "run",
        "--project",
        str(ROOT / "runtime"),
        "--locked",
        "--with",
        f"mitmproxy=={VERSION}",
        "python",
        str(Path(__file__).resolve()),
        "--child",
        case,
    ]
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=90)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        require(process.returncode == 0, f"measurement failed: {case}\n{stderr[-4000:]}")
    return json.loads(stdout)


def main(output):
    source_paths = [ROOT / "runtime" / name for name in ("pyproject.toml", "uv.lock")]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths]
    result = {
        "schema": "runtime-proxy-memory-assessment@1",
        "platform": platform.system() + " " + platform.machine(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "body_bytes": BODY_BYTES,
        "concurrency": CONCURRENCY,
        "rounds": ROUNDS,
        "samples": [],
    }
    for repetition in range(3):
        for case in (*IMPORT_CASES, *HTTP_CASES):
            sample = run_child(case)
            sample["repetition"] = repetition + 1
            result["samples"].append(sample)
            print(json.dumps({"repetition": repetition + 1, "case": case}), flush=True)
    after = [hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths]
    require(before == after, "production dependencies changed")
    result["production_dependencies_unchanged"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", choices=(*IMPORT_CASES, *HTTP_CASES))
    args = parser.parse_args()
    if args.child is not None:
        print(json.dumps(child(args.child)))
    elif args.output is not None:
        main(args.output)
    else:
        parser.error("--output or --child is required")
