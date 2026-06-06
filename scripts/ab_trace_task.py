#!/usr/bin/env python3
"""Interleaved A/B of deterministic observations on the trace_annotation task eval.

For each fixture we run the planner-based trace task eval twice — arm ``off``
(baseline, observations disabled) then arm ``on`` (fully enabled) — and stream a
scoreboard so progress is visible online. Arms are interleaved per fixture (not
all-off-then-all-on) so per-fixture deltas land as soon as a pair completes and
model/order drift is controlled.

The arm is selected by setting CONTRACTOR_EVAL_OBSERVATIONS in-process before
each call to ``run_eval`` (which resolves it onto the TaskRunner).

Usage::

    AB_FIXTURES=fastapi,spring poetry run python scripts/ab_trace_task.py
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

ARMS = {
    "off": '{"enabled": false}',
    "on": '{"enabled": true, "include_tool_errors": true, '
          '"track_skills": true, "track_files": true}',
}


def _row(slug: str, arm: str, r: dict) -> str:
    v = r.get("vuln") or {}
    return (
        f"{slug:10} {arm:3} | F1={r.get('f1', 0):.3f} P={r.get('precision', 0):.3f} "
        f"R={r.get('recall', 0):.3f} | vulnF1={v.get('f1', 0):.3f} "
        f"| tools={r.get('total_tool_calls', 0)} errs={r.get('tool_errors', 0)} "
        f"tok={r.get('total_tokens', 0)} llm={r.get('llm_calls', 0)} "
        f"| {r.get('duration_s', 0):.0f}s {'TIMEOUT' if r.get('timed_out') else ''}"
    )


async def main() -> None:
    fixtures = [s.strip() for s in os.environ.get("AB_FIXTURES", "fastapi,spring").split(",") if s.strip()]
    out = REPO / "eval_runs" / "ab_trace_obs"
    timeout_s = float(os.environ.get("AB_TIMEOUT", "3600"))
    per_path = float(os.environ.get("AB_PER_PATH_TIMEOUT", "420"))
    max_attempts = int(os.environ.get("AB_MAX_ATTEMPTS", "2"))

    summary: dict[str, dict[str, dict]] = {}
    print(f"A/B trace_annotation observations  fixtures={fixtures}", flush=True)

    for slug in fixtures:
        summary[slug] = {}
        for arm, val in ARMS.items():
            os.environ["CONTRACTOR_EVAL_OBSERVATIONS"] = val
            print(f"\n###### {slug} / arm={arm}  ({val}) ######", flush=True)
            try:
                res = await run_eval(
                    [slug], out / arm, timeout_s, None, max_attempts, 1, per_path
                )
                r = res[0] if res else {}
            except Exception as exc:  # keep the A/B going if one arm errors
                print(f"  [{slug}/{arm}] ERROR: {exc!r}", flush=True)
                r = {}
            summary[slug][arm] = r
            print("ROW " + _row(slug, arm, r), flush=True)

        off, on = summary[slug].get("off", {}), summary[slug].get("on", {})
        if off and on:
            print(
                f"DELTA {slug}: dF1={on.get('f1', 0) - off.get('f1', 0):+.3f} "
                f"dP={on.get('precision', 0) - off.get('precision', 0):+.3f} "
                f"dR={on.get('recall', 0) - off.get('recall', 0):+.3f} "
                f"dVulnF1={(on.get('vuln') or {}).get('f1', 0) - (off.get('vuln') or {}).get('f1', 0):+.3f} "
                f"dTools={on.get('total_tool_calls', 0) - off.get('total_tool_calls', 0):+d} "
                f"dTok={on.get('total_tokens', 0) - off.get('total_tokens', 0):+d}",
                flush=True,
            )
        out.mkdir(parents=True, exist_ok=True)
        (out / "ab_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

    print("\n===== A/B SUMMARY (observations off vs on) =====", flush=True)
    for slug, arms in summary.items():
        for arm in ("off", "on"):
            if arm in arms:
                print(_row(slug, arm, arms[arm]), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
