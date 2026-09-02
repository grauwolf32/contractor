"""Reproducibly characterize the bounded allocation-local Trailmark child."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

from contractor_runtime.projectfs.storage import (
    WorkspaceSnapshot,
    WorkspaceTextFile,
    workspace_digest,
)
from contractor_runtime.toolsets.trailmark_host import MAX_GRAPH_FILES, TrailmarkChildHost

FIXTURE_FILE_COUNTS = {
    "small": 3,
    "medium": 500,
    "bounded-overflow": MAX_GRAPH_FILES + 100,
}


def main() -> int:
    parser = argparse.ArgumentParser(prog="measure_trailmark.py")
    parser.add_argument("--fixture", choices=tuple(FIXTURE_FILE_COUNTS), default="small")
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    result = asyncio.run(_measure(arguments.fixture))
    if arguments.json:
        print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0


async def _measure(fixture: str) -> dict[str, object]:
    snapshot = _fixture(FIXTURE_FILE_COUNTS[fixture])
    with tempfile.TemporaryDirectory(prefix="contractor-trailmark-measure-") as directory:
        host = TrailmarkChildHost(Path(directory))
        started = time.perf_counter()
        try:
            result = await host.build(snapshot)
        finally:
            elapsed = time.perf_counter() - started
            await host.close()
    return {
        "fixture": fixture,
        "inputFiles": len(snapshot.files),
        "buildSeconds": round(elapsed, 6),
        "rssKiB": result.rss_kib,
        "nodeCount": result.node_count,
        "edgeCount": result.edge_count,
        "entrypointCount": result.entrypoint_count,
        "coverage": result.coverage.wire(),
    }


def _fixture(count: int) -> WorkspaceSnapshot:
    texts = {
        f"src/module_{index:05d}.py": (f"def function_{index:05d}():\n    return {index}\n")
        for index in range(count)
    }
    files = tuple(
        WorkspaceTextFile(path=path, text=text, size=len(text.encode("utf-8")))
        for path, text in sorted(texts.items())
    )
    return WorkspaceSnapshot(
        directories=("src",),
        files=files,
        binary_paths=(),
        digest=workspace_digest({"src"}, texts),
    )


if __name__ == "__main__":
    raise SystemExit(main())
