"""Production controlled HTTP memory probe; four retained 8 MiB responses.

Run from the repository: cd runtime && uv run --locked python
../scripts/experiments/mitmproxy_production_memory.py --output /tmp/native-memory.json
Each of three samples runs in a fresh process, with real private-interface TCP
sockets, the production policy/engine and test-only assignment/shared fixtures.
"""

import argparse
import asyncio
import gc
import json
import subprocess
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime/tests"))
MIB = 1024 * 1024


def memory():
    gc.collect()
    return round(psutil.Process().memory_info().rss / MIB, 2)


async def measure():
    from fakes.engagement_socket import SocketTarget, preflight

    import contractor_runtime.cli  # noqa: F401
    from contractor_runtime import shared_config
    from contractor_runtime.contracts import SharedConfig, decode_private

    shared_config._snapshot = decode_private(
        SharedConfig, (ROOT / "runtime/tests/testdata/shared-config.json").read_bytes()
    ).snapshot()
    rows = {"runtime_and_fixtures_mib": memory()}
    payload = b"x" * (8 * MIB)

    async def respond(_, __, writer, ___):
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 8388608\r\n\r\n")
        # Keep the fixture's own send buffers small instead of duplicating
        # every complete payload inside each asyncio transport.
        for start in range(0, len(payload), 65536):
            writer.write(payload[start : start + 65536])
            await writer.drain()

    async with SocketTarget(respond) as target, preflight(target) as (owner, _):
        owner.configure(authority_deadline=lambda: asyncio.get_running_loop().time() + 120)
        await owner.open_proxy()
        rows["engine_and_fixture_ready_mib"] = memory()
        for index in range(3):
            responses = await asyncio.gather(
                *(owner.request(target.origin + "/allowed") for _ in range(4))
            )
            if any(
                outcome.kind != "completed" or len(response.body) != len(payload)
                for outcome, response in responses
            ):
                raise RuntimeError("native memory workload did not complete")
            rows[f"round_{index + 1}_retained_mib"] = memory()
            rows[f"round_{index + 1}_retained_bytes"] = sum(len(r.body) for _, r in responses)
            responses.clear()
            rows[f"round_{index + 1}_discarded_mib"] = memory()
        rows["live_exchange_flows"] = len(owner._engine.flows)
        rows["idle_connections"] = sum(
            len(pool.transports) for pool in owner._engine.pools.values()
        )
        directory = Path(owner._engine.directory.name)
        await owner.close()
        await target.all_closed.wait()
        rows["after_joined_close_mib"] = memory()
        rows["certificate_directory_removed"] = not directory.exists()
        rows["target_requests"] = len(target.requests)
        if rows["live_exchange_flows"] or directory.exists() or len(target.requests) != 12:
            raise RuntimeError("native memory probe cleanup failed")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(asyncio.run(measure())))
        return
    if args.output is None:
        parser.error("--output is required")
    samples = []
    for _ in range(3):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        samples.append(json.loads(result.stdout))
    args.output.write_text(json.dumps({"samples": samples}, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "samples": samples}))


if __name__ == "__main__":
    main()
