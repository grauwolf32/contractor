#!/usr/bin/env python3
"""Compare two A/B eval arms produced by run_ab_limit_eval.sh.

Reads per-case metrics.json under eval_runs_default/ and eval_runs_new/ and
prints a per-case pass/score diff plus regression/improvement summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Override arm dirs from argv: compare_ab_eval.py <armA_label>:<dir> <armB_label>:<dir>
_argv = sys.argv[1:]
if len(_argv) == 2 and all(":" in a for a in _argv):
    ARMS = {a.split(":", 1)[0]: ROOT / a.split(":", 1)[1] for a in _argv}
else:
    ARMS = {"default": ROOT / "eval_runs_default", "new": ROOT / "eval_runs_new"}
_A, _B = list(ARMS)  # arm labels in order

# numeric detail keys worth surfacing per domain
SCORE_KEYS = ("precision", "recall", "f1", "endpoint_precision", "endpoint_recall",
              "tp", "fp", "fn", "captured", "expected", "score")


def load_arm(base: Path) -> dict[tuple[str, str], dict]:
    """(unit, fixture__case) -> case dict."""
    out: dict[tuple[str, str], dict] = {}
    if not base.is_dir():
        return out
    for metrics in base.glob("*/cases/*/metrics.json"):
        unit = metrics.parts[len(base.parts)]            # eval_runs_X/<unit>/cases/...
        case_key = metrics.parent.name                   # <fixture>__<case>
        try:
            out[(unit, case_key)] = json.loads(metrics.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  ! skip {metrics}: {e}")
    return out


def score_str(case: dict) -> str:
    d = {**(case.get("detail") or {}), **(case.get("metrics") or {})}
    bits = [f"{k}={d[k]}" for k in SCORE_KEYS if k in d]
    return ", ".join(bits) or "-"


def main() -> int:
    arms = {name: load_arm(path) for name, path in ARMS.items()}
    for name, data in arms.items():
        print(f"[{name}] {ARMS[name]}: {len(data)} cases"
              + ("" if data else "  (MISSING — arm may not have finished)"))
    keys = sorted(set(arms[_A]) | set(arms[_B]))
    if not keys:
        print("No cases found in either arm.")
        return 1

    regressions, improvements = [], []
    by_unit: dict[str, list[int]] = {}
    print(f"\n{'unit/case':54} {'default':>8} {'new':>8}")
    print("-" * 74)
    for unit, case_key in keys:
        dc = arms[_A].get((unit, case_key))
        nc = arms[_B].get((unit, case_key))
        dp = dc.get("passed") if dc else None
        npass = nc.get("passed") if nc else None
        tag = "  "
        if dp is not None and npass is not None:
            by_unit.setdefault(unit, [0, 0, 0])
            by_unit[unit][0] += int(bool(dp))
            by_unit[unit][1] += int(bool(npass))
            by_unit[unit][2] += 1
            if dp and not npass:
                regressions.append((unit, case_key, dc, nc))
            elif npass and not dp:
                improvements.append((unit, case_key, dc, nc))
        label = f"{unit}/{case_key}"[:54]
        print(f"{label:54} {str(dp):>8} {str(npass):>8} {tag}")

    print("\n=== per-unit pass counts (default → new) ===")
    for unit, (d, n, t) in sorted(by_unit.items()):
        flag = "  ⚠ DROP" if n < d else ("  ↑" if n > d else "")
        print(f"  {unit:20} {d}/{t} → {n}/{t}{flag}")

    print(f"\n=== regressions (passed default, failed new): {len(regressions)} ===")
    for unit, ck, dc, nc in regressions:
        print(f"  {unit}/{ck}\n      default: {score_str(dc)}\n      new    : {score_str(nc)}")
    print(f"\n=== improvements (failed default, passed new): {len(improvements)} ===")
    for unit, ck, _dc, _nc in improvements:
        print(f"  {unit}/{ck}")

    print("\n=== score deltas on cases both arms ran ===")
    for unit, case_key in keys:
        dc, nc = arms[_A].get((unit, case_key)), arms[_B].get((unit, case_key))
        if dc and nc:
            ds, ns = score_str(dc), score_str(nc)
            if ds != ns:
                print(f"  {unit}/{case_key}\n      default: {ds}\n      new    : {ns}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
