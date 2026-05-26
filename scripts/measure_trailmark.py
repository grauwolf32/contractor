"""Measure trailmark cost on a target codebase.

Reports: language-detected file count, parse+index wall-time, peak RSS,
peak tracemalloc, node/edge counts, on-disk JSON size.

Usage:
    poetry run python scripts/measure_trailmark.py <project_path> [--language auto]
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
import tracemalloc
from pathlib import Path


def _rss_mb() -> float:
    """Current process RSS in MiB (Linux: ru_maxrss is in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project_path", type=Path)
    ap.add_argument("--language", default="auto")
    ap.add_argument("--dump-json", type=Path, default=None,
                    help="Optional path to write engine.to_json() output")
    args = ap.parse_args()

    project = args.project_path.resolve()
    if not project.is_dir():
        print(f"not a directory: {project}", file=sys.stderr)
        return 2

    from trailmark.parse import detect_languages, parse_directory
    from trailmark.parsers import _common as _tm_common
    from trailmark.query.api import QueryEngine

    # Skip files that crash the C++/C parser with UnicodeDecodeError.
    _orig_parse_dir = _tm_common.parse_directory
    skipped: list[str] = []

    def _safe_parse_directory(parse_file_fn, language, dir_path, extensions):
        def _wrapped(p):
            try:
                return parse_file_fn(p)
            except UnicodeDecodeError as e:
                skipped.append(f"{p} ({e})")
                from trailmark.models.graph import CodeGraph
                return CodeGraph(language=language, root_path=str(p))

        return _orig_parse_dir(_wrapped, language, dir_path, extensions)

    _tm_common.parse_directory = _safe_parse_directory

    print(f"# target: {project}")
    print(f"# language arg: {args.language}")

    t0 = time.perf_counter()
    detected = detect_languages(str(project))
    t_detect = time.perf_counter() - t0
    print(f"detect_languages: {t_detect*1000:.1f} ms -> {detected}")

    gc.collect()
    tracemalloc.start()
    rss_before = _rss_mb()

    t0 = time.perf_counter()
    graph = parse_directory(str(project), language=args.language)
    t_parse = time.perf_counter() - t0

    rss_after_parse = _rss_mb()
    _, peak_traced_parse = tracemalloc.get_traced_memory()

    t0 = time.perf_counter()
    engine = QueryEngine.from_graph(graph)
    t_index = time.perf_counter() - t0

    rss_after_index = _rss_mb()
    _, peak_traced_index = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print()
    print(f"parse_directory:     {t_parse:.2f} s")
    print(f"QueryEngine.build:   {t_index*1000:.1f} ms")
    print(f"nodes:               {len(graph.nodes):,}")
    print(f"edges:               {len(graph.edges):,}")
    print(f"entrypoints:         {len(graph.entrypoints):,}")
    print(f"languages:           {graph.language if hasattr(graph,'language') else '?'}")
    print()
    print(f"RSS before:          {rss_before:8.1f} MiB")
    print(f"RSS after parse:     {rss_after_parse:8.1f} MiB  (Δ {rss_after_parse-rss_before:+.1f})")
    print(f"RSS after index:     {rss_after_index:8.1f} MiB  (Δ {rss_after_index-rss_before:+.1f})")
    print(f"tracemalloc peak (parse): {peak_traced_parse/1024/1024:8.1f} MiB")
    print(f"tracemalloc peak (total): {peak_traced_index/1024/1024:8.1f} MiB")
    if skipped:
        print(f"skipped files (UnicodeDecodeError): {len(skipped)}")
        for s in skipped[:5]:
            print(f"  - {s}")

    # Sample queries — measure cost of typical agent-style calls.
    sample_node_id = next(iter(graph.nodes), None)
    if sample_node_id is not None:
        name_for_lookup = graph.nodes[sample_node_id].name
        t0 = time.perf_counter()
        cs = engine.callers_of(name_for_lookup)
        t_callers = time.perf_counter() - t0
        t0 = time.perf_counter()
        cs2 = engine.callees_of(name_for_lookup)
        t_callees = time.perf_counter() - t0
        t0 = time.perf_counter()
        anc = engine.ancestors_of(name_for_lookup)
        t_anc = time.perf_counter() - t0
        print()
        print(f"sample query target: {name_for_lookup}")
        print(f"  callers_of:    {t_callers*1000:7.2f} ms  ({len(cs)} hits)")
        print(f"  callees_of:    {t_callees*1000:7.2f} ms  ({len(cs2)} hits)")
        print(f"  ancestors_of:  {t_anc*1000:7.2f} ms  ({len(anc)} hits)")

    if args.dump_json is not None:
        t0 = time.perf_counter()
        payload = engine.to_json()
        t_serialize = time.perf_counter() - t0
        text = json.dumps(payload) if not isinstance(payload, str) else payload
        args.dump_json.write_text(text)
        sz = args.dump_json.stat().st_size
        print()
        print(f"engine.to_json:      {t_serialize:.2f} s")
        print(f"json on-disk size:   {sz/1024/1024:.1f} MiB  ({args.dump_json})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
