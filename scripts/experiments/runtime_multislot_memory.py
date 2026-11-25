"""Opt-in Linux comparison through real Server/Runtime/PostgreSQL entry points.

Requires the locked Runtime, Go, a disposable CONTRACTOR_TEST_DATABASE_URL and
an already provisioned rootless CONTRACTOR_TEST_PODMAN_IMAGE. No live model is
called. Missing prerequisites fail; measurements are never synthesized.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import platform
import signal
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import psutil

from contractor_runtime.sandbox.contracts import PODMAN_PROFILE
from contractor_runtime.sandbox.podman.engine import LABEL_PREFIX
from contractor_runtime.sandbox.podman.settings import PodmanSettings

ROOT = Path(__file__).resolve().parents[2]
MIB = 1 << 20
BODY = b"x" * 65536
CAPABILITY_FIELDS = (
    "softwareVersion",
    "supportedRuntimes",
    "supportedToolsets",
    "supportedSandboxProfiles",
    "supportedRuntimeAdapters",
    "workspaceCapabilities",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(command, *, timeout=120):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    require(result.returncode == 0, f"{command[0]} failed: {result.stderr[-4000:]}")
    return result.stdout.strip()


def source_digest():
    paths = [
        *sorted((ROOT / "runtime/src").rglob("*.py")),
        ROOT / "runtime/uv.lock",
        ROOT / "runtime/pyproject.toml",
        Path(__file__).resolve(),
        *(
            ROOT / "tests/e2e" / name
            for name in (
                "runtime_memory_test.go",
                "runtime_slots_test.go",
                "runtime_placement_test.go",
            )
        ),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def summary(values):
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def phase_summary(samples):
    require(samples, "empty process-memory phase")
    return {
        "samples": len(samples),
        **{key: summary([sample[key] for sample in samples]) for key in samples[0]},
    }


def measure_case(binary, directory, slots, target):
    directory.mkdir()
    environment = {
        **os.environ,
        "CONTRACTOR_RUNTIME_MULTISLOT_MEASUREMENT": str(directory),
        "CONTRACTOR_RUNTIME_MEMORY_SLOTS": str(slots),
        "CONTRACTOR_RUNTIME_MEMORY_TARGET": target,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    identities = {}
    load_average_before = os.getloadavg()
    with (
        (directory / "harness.log").open("w") as log,
        subprocess.Popen(
            [
                str(binary),
                "-test.v",
                "-test.count=1",
                "-test.timeout=5m",
                "-test.run=^TestRuntimeAllocationSlotsAcrossProductionProcesses$",
            ],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process,
    ):
        deadline = time.monotonic() + 240

        def check():
            if process.poll() is not None:
                detail = (directory / "harness.log").read_text()[-8000:]
                raise RuntimeError(f"measurement process exited before its barrier:\n{detail}")
            require(time.monotonic() < deadline, "measurement phase deadline expired")

        def wait(name):
            path = directory / (name + ".json")
            while not path.exists():
                check()
                time.sleep(0.05)
            return json.loads(path.read_bytes())

        try:
            ready = wait("ready")
            roots = set(ready["runtime_pids"])
            require(len(roots) == 4 // slots, "wrong Runtime process count")
            capabilities = {key: ready["agents"][0].get(key) for key in CAPABILITY_FIELDS}
            require(
                PODMAN_PROFILE in capabilities["supportedSandboxProfiles"],
                "the ordinary rootless Podman capability was not advertised",
            )
            require(
                capabilities["workspaceCapabilities"] is not None,
                "the local workspace and archive cache were not enabled",
            )
            for agent in ready["agents"]:
                require(agent["allocationSlots"] == slots, "capacity changed")
                require(
                    capabilities == {key: agent.get(key) for key in CAPABILITY_FIELDS},
                    "Runtime capabilities differ within the layout",
                )

            def sample():
                processes = {}
                for pid in roots:
                    root = psutil.Process(pid)
                    for child in [root, *root.children(recursive=True)]:
                        processes[child.pid] = child
                require(len(processes) <= 128, "unbounded measurement process tree")
                result = dict.fromkeys(
                    (
                        "rss_bytes",
                        "pss_bytes",
                        "private_bytes",
                        "runtime_pss_bytes",
                        "helper_pss_bytes",
                        "podman_backend_pss_bytes",
                        "scanner_pss_bytes",
                    ),
                    0,
                )
                backends = scanners = 0
                for pid, child in processes.items():
                    identities[pid] = child.create_time()
                    memory = child.memory_full_info()
                    result["rss_bytes"] += memory.rss
                    result["pss_bytes"] += memory.pss
                    result["private_bytes"] += memory.uss
                    result["runtime_pss_bytes" if pid in roots else "helper_pss_bytes"] += (
                        memory.pss
                    )
                    if pid not in roots:
                        command = child.cmdline()
                        if "contractor_runtime.sandbox.podman.owner" in command:
                            backends += 1
                            result["podman_backend_pss_bytes"] += memory.pss
                        if any(Path(argument).name == "nuclei.py" for argument in command):
                            scanners += 1
                            result["scanner_pss_bytes"] += memory.pss
                require(
                    backends == len(roots) and scanners == 4,
                    "the measured backend/scanner process topology changed",
                )
                result["podman_backends"] = backends
                result["scanner_children"] = scanners
                result["processes"] = len(processes)
                result["helpers"] = len(processes) - len(roots)
                return result

            def held(seconds):
                samples, until = [], time.monotonic() + seconds
                while time.monotonic() < until:
                    check()
                    samples.append(sample())
                    time.sleep(0.25)
                return samples

            warm = held(15)
            (directory / "load").touch()
            load = []
            while not (directory / "computed.json").exists():
                check()
                load.append(sample())
                time.sleep(0.25)
            computed = wait("computed")
            post = held(2)
            results = [
                json.loads(path.read_bytes())
                for path in sorted((directory / "workers").glob("result-*.json"))
            ]
            inputs = [
                json.loads(path.read_bytes())
                for path in (directory / "workers").glob("ready-*.json")
            ]
            require(len(results) == len(inputs) == 4, "incomplete four-worker workload")
            digest = hashlib.sha256(BODY).hexdigest()
            require(all(row["input_sha256"] == digest for row in inputs), "input bytes changed")
            require(
                all(
                    row["digest"] == digest and row["units"] == ready["units_per_worker"]
                    for row in results
                ),
                "workload or result changed",
            )
            require(computed["heartbeat_progress"], "Runtime heartbeat did not progress")
            elapsed = max(row["end"] for row in results) - min(row["start"] for row in results)
            (directory / "release").touch()
            released = wait("released")
            require(process.wait(timeout=75) == 0, "measurement harness failed during cleanup")
        except BaseException:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=75)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise

    for pid, created in identities.items():
        try:
            child = psutil.Process(pid)
            require(child.create_time() != created, f"owned process survived cleanup: {pid}")
        except psutil.NoSuchProcess:
            pass
    for owner in ready["podman_owners"]:
        require(
            not run(["podman", "ps", "-aq", "--filter", f"label={LABEL_PREFIX}owner={owner}"]),
            "measurement-owned Podman container survived cleanup",
        )
    heartbeat = []
    for before in computed["heartbeats_before"]:
        after = next(
            row for row in computed["heartbeats_after"] if row["instanceId"] == before["instanceId"]
        )
        heartbeat.append(
            {"before": before["lastAcceptedHeartbeat"], "after": after["lastAcceptedHeartbeat"]}
        )
    return {
        "layout": f"{4 // slots}x{slots}",
        "capabilities": capabilities,
        "server_binary_sha256": ready["server_binary_sha256"],
        "warm": phase_summary(warm),
        "load": phase_summary(load),
        "post": phase_summary(post),
        "cpu_hash_units_per_worker": ready["units_per_worker"],
        "bytes_per_unit": len(BODY),
        "load_average_before": load_average_before,
        "load_average_after": os.getloadavg(),
        "load_seconds": elapsed,
        "worker_latency_seconds": summary([row["end"] - row["start"] for row in results]),
        "cpu_hash_units_per_second": sum(row["units"] for row in results) / elapsed,
        "heartbeats": heartbeat,
        "cleanup": {**released, "all_sampled_processes_gone": True, "owned_containers_after": 0},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, choices=range(1, 6), default=3)
    args = parser.parse_args()
    require(
        platform.system() == "Linux" and Path("/proc/self/smaps_rollup").is_file(),
        "PSS measurements require Linux /proc smaps_rollup",
    )
    require(os.environ.get("CONTRACTOR_TEST_DATABASE_URL"), "disposable PostgreSQL is required")
    image = os.environ.get("CONTRACTOR_TEST_PODMAN_IMAGE")
    PodmanSettings(enabled=True, image=image, owner="measurement")
    require(
        run(["podman", "info", "--format", "{{.Host.Security.Rootless}}"], timeout=20) == "true",
        "a rootless Podman engine is required",
    )
    run(["podman", "image", "inspect", image], timeout=20)
    revision = os.environ.get("CONTRACTOR_SOURCE_REVISION")
    if not revision:
        revision = run(["git", "rev-parse", "HEAD"])
    require(
        len(revision) == 40 and all(c in "0123456789abcdef" for c in revision),
        "a full source revision is required for an exported source tree",
    )
    before = source_digest()
    expected = os.environ.get("CONTRACTOR_RUNTIME_SOURCE_SHA256")
    require(expected is None or expected == before, "exported measurement sources differ")
    samples = []

    class Origin(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            self.wfile.write(BODY)

        def log_message(self, *_):
            pass

    with tempfile.TemporaryDirectory(prefix="contractor-multislot-memory-") as temporary:
        root = Path(temporary)
        binary = root / "process-tests"
        run(["go", "test", "-c", "-tags=e2e", "-o", str(binary), "./tests/e2e"], timeout=180)
        binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Origin)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        target = f"http://127.0.0.1:{server.server_port}/input"
        try:
            for repetition in range(args.repetitions):
                for slots in (1, 4) if repetition % 2 == 0 else (4, 1):
                    sample = measure_case(
                        binary, root / f"trial-{repetition + 1}-{slots}", slots, target
                    )
                    sample["repetition"] = repetition + 1
                    if samples:
                        require(
                            sample["capabilities"] == samples[0]["capabilities"], "capability drift"
                        )
                        require(
                            sample["server_binary_sha256"] == samples[0]["server_binary_sha256"],
                            "Server binaries differ between layouts",
                        )
                    samples.append(sample)
                    print(
                        json.dumps(
                            {
                                "repetition": repetition + 1,
                                "layout": sample["layout"],
                                "warm_pss_mib": sample["warm"]["pss_bytes"]["median"] / MIB,
                            }
                        ),
                        flush=True,
                    )
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
    require(source_digest() == before, "measured sources or locked dependencies changed")
    paired = []
    for repetition in range(1, args.repetitions + 1):
        pair = {row["layout"]: row for row in samples if row["repetition"] == repetition}
        paired.append(
            100
            * (
                1
                - pair["1x4"]["warm"]["pss_bytes"]["median"]
                / pair["4x1"]["warm"]["pss_bytes"]["median"]
            )
        )
    result = {
        "schema": "runtime-multislot-memory@1",
        "status": "passed",
        "revision": revision,
        "source_sha256": before,
        "test_binary_sha256": binary_sha256,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "podman_image": image,
        "input_sha256": hashlib.sha256(BODY).hexdigest(),
        "repetitions": args.repetitions,
        "method": {
            "topology": (
                "Real Server, PostgreSQL, mTLS, Scheduler and ordinary tool Workers; same compiled "
                "test binary and identical capabilities, input bytes and work units."
            ),
            "phases": (
                "Four scanner children fetch 64 KiB, warm up with 64 hashes, hold for 15 seconds, "
                "execute fixed SHA256 work, then hold for 2 seconds before ordinary release."
            ),
            "memory": (
                "250 ms samples of Runtime trees including Podman backends and scanner children; "
                "PSS, USS (private clean/dirty pages) and RSS are summed once per PID. "
                "RSS includes shared pages repeatedly."
            ),
            "excluded": (
                "Identical Server/driver/origin and existing engine infrastructure. No live model, "
                "workload container or retained HTTP body load; HTTP Master remains lazy and "
                "the archive cache is enabled but empty."
            ),
            "uncertainty": (
                "Three alternating-order repetitions by default; min/median/max show variability. "
                "Shared VM, CPU frequency, file-page sharing and sampling affect values. "
                "No fixed savings or performance threshold."
            ),
        },
        "warm_pss_savings_percent": summary(paired),
        "samples": samples,
        "cleanup_confirmed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "warm_pss_savings_percent": summary(paired)}))


if __name__ == "__main__":
    main()
