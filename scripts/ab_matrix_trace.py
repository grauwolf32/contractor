#!/usr/bin/env python3
"""Multi-arm A/B matrix of observation configs on the trace_annotation task eval.

Runs a set of named observation arms (each an ObservationConfig overlay) on one
or more fixtures via the planner-based trace task eval, with high per-path
timeouts so no path is truncated (clean, apples-to-apples comparison). Streams a
per-arm scoreboard and, at the end, ranks arms by a composite quality score
(annotation F1 + vuln F1) with cost (tokens) as the tiebreaker, and prints the
recommended "best combination".

Single run per (arm, fixture): a local model is noisy, so treat the ranking as
indicative — large deltas and cost trends are the trustworthy signal. Results
stream to <out>/matrix.jsonl and the final ranking to stdout.

Usage::

    AB_FIXTURE=vulnyapi AB_PER_PATH_TIMEOUT=900 poetry run python scripts/ab_matrix_trace.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_trace_task_eval import run_eval  # noqa: E402

# Each arm is (name, overlay-dict). Defaults: track_tools/files/skills=True,
# include_tool_errors=False, malformed_only=False, in_record/in_result=True.
ARMS: list[tuple[str, dict]] = [
    ("off", {"enabled": False}),
    ("full", {"enabled": True, "include_tool_errors": True}),
    ("lean_no_errors", {"enabled": True, "include_tool_errors": False}),
    ("lean_memories", {"enabled": True, "include_tool_errors": False, "track_memories": True}),
    ("lean_paths", {"enabled": True, "include_tool_errors": False, "track_file_paths": True}),
    ("record_only", {"enabled": True, "include_tool_errors": True, "in_result": False}),
    ("malformed_only", {"enabled": True, "include_tool_errors": True, "malformed_only": True}),
    ("files_skills", {"enabled": True, "track_tools": False, "track_files": True, "track_skills": True}),
]


def _quality(r: dict) -> float:
    """Composite: annotation F1 (primary task) + vuln F1 (secondary)."""
    return 0.6 * float(r.get("f1", 0) or 0) + 0.4 * float((r.get("vuln") or {}).get("f1", 0) or 0)


def _row(name: str, r: dict) -> str:
    v = r.get("vuln") or {}
    return (
        f"{name:16} | annotF1={r.get('f1', 0):.3f} (P={r.get('precision', 0):.3f} "
        f"R={r.get('recall', 0):.3f}) | vulnF1={v.get('f1', 0):.3f} "
        f"| quality={_quality(r):.3f} | tools={r.get('total_tool_calls', 0)} "
        f"tok={r.get('total_tokens', 0)} llm={r.get('llm_calls', 0)} "
        f"errs={r.get('tool_errors', 0)} | {r.get('duration_s', 0):.0f}s"
        f"{' TIMEOUT' if r.get('timed_out') else ''}"
    )


async def main() -> None:
    fixture = os.environ.get("AB_FIXTURE", "vulnyapi")
    arm_filter = os.environ.get("AB_ARMS")  # comma-separated subset of arm names
    if arm_filter:
        wanted = {a.strip() for a in arm_filter.split(",") if a.strip()}
        globals()["ARMS"] = [(n, o) for n, o in ARMS if n in wanted]
    out = REPO / "eval_runs" / "ab_matrix" / fixture
    out.mkdir(parents=True, exist_ok=True)
    per_path = float(os.environ.get("AB_PER_PATH_TIMEOUT", "900"))
    outer = float(os.environ.get("AB_TIMEOUT", "21600"))
    max_attempts = int(os.environ.get("AB_MAX_ATTEMPTS", "2"))
    jsonl = out / "matrix.jsonl"

    print(f"A/B MATRIX  fixture={fixture}  arms={[a for a, _ in ARMS]}  "
          f"per_path={per_path}s outer={outer}s", flush=True)

    collected: list[tuple[str, dict]] = []
    for name, overlay in ARMS:
        os.environ["CONTRACTOR_EVAL_OBSERVATIONS"] = json.dumps(overlay)
        print(f"\n###### arm={name}  {json.dumps(overlay)} ######", flush=True)
        try:
            res = await run_eval([fixture], out / name, outer, None, max_attempts, 1, per_path)
            r = res[0] if res else {}
        except Exception as exc:
            print(f"  [arm {name}] ERROR: {exc!r}", flush=True)
            r = {}
        collected.append((name, r))
        print("ROW " + _row(name, r), flush=True)
        with jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"arm": name, "overlay": overlay, "result": r}, default=str) + "\n")

    # Rank: quality desc, then fewer tokens (cheaper) as tiebreaker.
    ranked = sorted(
        [c for c in collected if c[1]],
        key=lambda c: (-_quality(c[1]), c[1].get("total_tokens", 1e18)),
    )
    base = next((r for n, r in collected if n == "off"), {})

    print("\n\n===================== A/B MATRIX RANKING =====================", flush=True)
    print(f"(fixture={fixture}; quality = 0.6*annotF1 + 0.4*vulnF1; n=1 per arm)\n", flush=True)
    for rank, (name, r) in enumerate(ranked, 1):
        dq = _quality(r) - _quality(base) if base else 0.0
        dtok = (r.get("total_tokens", 0) - base.get("total_tokens", 0)) if base else 0
        print(f"#{rank} {_row(name, r)}  | dQ_vs_off={dq:+.3f} dTok={dtok:+d}", flush=True)

    if ranked:
        best_name, best_r = ranked[0]
        print(f"\nBEST: {best_name}", flush=True)
        print(f"BEST_CONFIG: {json.dumps(dict(ARMS)[best_name])}", flush=True)
        (out / "ranking.json").write_text(
            json.dumps(
                {"fixture": fixture, "best": best_name,
                 "best_config": dict(ARMS)[best_name],
                 "ranked": [{"arm": n, "quality": _quality(r), **{k: r.get(k) for k in
                            ("f1", "precision", "recall", "total_tokens",
                             "total_tool_calls", "llm_calls", "duration_s")},
                            "vuln_f1": (r.get("vuln") or {}).get("f1")} for n, r in ranked]},
                indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
