from __future__ import annotations

import json
import os
import signal
import struct
import sys
import time
from pathlib import Path


def read_request() -> dict[str, object]:
    header = sys.stdin.buffer.read(4)
    if len(header) != 4:
        raise SystemExit(2)
    length = struct.unpack(">I", header)[0]
    payload = sys.stdin.buffer.read(length)
    if len(payload) != length:
        raise SystemExit(2)
    return json.loads(payload)


def write(document: dict[str, object]) -> None:
    payload = json.dumps(document, separators=(",", ":"), sort_keys=True).encode()
    sys.stdout.buffer.write(struct.pack(">I", len(payload)) + payload)
    sys.stdout.buffer.flush()


def success(request: dict[str, object]) -> None:
    arguments = request["arguments"]
    assert isinstance(arguments, dict)
    write(
        {
            "schemaVersion": "1.0",
            "requestId": request["requestId"],
            "ok": True,
            "result": {
                "snapshotDigest": arguments["snapshotDigest"],
                "coverage": arguments["coverage"],
                "languages": ["python"],
                "nodeCount": 1,
                "edgeCount": 0,
                "callEdgeCount": 0,
                "entrypointCount": 0,
                "dependencyCount": 0,
                "rssKiB": 1,
            },
        }
    )


def main() -> int:
    mode = sys.argv[1]
    if mode == "exit-once-before-read":
        marker = Path(sys.argv[2])
        try:
            descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
            return 8
    request = read_request()
    if mode == "hang":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(60)
    if mode == "crash":
        return 9
    if mode == "oom":
        os.kill(os.getpid(), signal.SIGKILL)
    if mode == "oversized":
        sys.stdout.buffer.write(struct.pack(">I", 2 * 1024 * 1024 + 1))
        sys.stdout.buffer.flush()
        time.sleep(60)
    if mode == "malformed":
        sys.stdout.buffer.write(struct.pack(">I", 1) + b"{")
        sys.stdout.buffer.flush()
        return 0
    if mode == "partial":
        sys.stdout.buffer.write(struct.pack(">I", 100) + b"{")
        sys.stdout.buffer.flush()
        return 0
    if mode == "wrong-id":
        write(
            {
                "schemaVersion": "1.0",
                "requestId": "different-request",
                "ok": True,
                "result": {},
            }
        )
        time.sleep(60)
    if mode == "bad-result":
        write(
            {
                "schemaVersion": "1.0",
                "requestId": request["requestId"],
                "ok": True,
                "result": {},
            }
        )
        time.sleep(60)
    if mode == "ignore-term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        success(request)
        time.sleep(60)
    if mode == "stderr-flood":
        for _ in range(512):
            sys.stderr.buffer.write(b"x" * 4096)
        sys.stderr.buffer.flush()
        success(request)
        time.sleep(60)
    if mode == "exit-once-before-read":
        success(request)
        time.sleep(60)
    if mode == "hang-query":
        success(request)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        read_request()
        time.sleep(60)
    if mode == "crash-query":
        success(request)
        read_request()
        return 9
    raise SystemExit(3)


if __name__ == "__main__":
    raise SystemExit(main())
