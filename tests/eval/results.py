"""Standardized eval result envelope (``eval/v1``) shared by all three
eval scenarios.

There are exactly three eval scenarios, distinguished by *what unit is under
test* and tagged with a pass@X repetition parameter:

* **agent**   — one ``LlmAgent`` driven directly (``harness.run_agent``); pass@N.
* **task**    — one ``TaskRunner`` task (planner + worker; ``task_harness``); pass@M.
* **pipeline**— a whole workflow / CLI run; pass@K.

N/M/K are mechanically identical — a per-scenario default + knob name. The
*semantics* are uniform: a case is attempted ``pass_at`` times and counts as
**passed** if *any* attempt passes (pass@X = "solved at least once in X tries").

Every producer converges on a single on-disk shape — ``eval_results.json`` in
the ``eval/v1`` envelope below — so ``analytics-ui`` renders all evals
uniformly, grouped by scenario, with a small domain panel keyed by
``metric_kind`` (detection / verdict / capture / diff / generic).

This module owns:

* the dataclasses (:class:`AttemptOutcome` → :class:`CaseResult` →
  :class:`FixtureResult` → :class:`EvalRun`),
* the :func:`pass_at` repeater (built on whatever an attempt reports as
  ``passed`` — typically ``EvalResult.passed`` from ``tests/eval/scoring.py``),
* :func:`write_eval_results` which serializes the envelope to
  ``eval_runs/<run>/eval_results.json`` (and embeds a derived headline/totals
  snapshot so the JSON is self-describing offline),
* the pure headline/totals derivation (:func:`derive_headline`,
  :func:`derive_totals`) reused by the UI's normalizer.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Optional

SCHEMA = "eval/v1"

Scenario = Literal["agent", "task", "pipeline"]
# How a case is scored — drives which domain panel the UI shows. "generic"
# means pass/fail only (no domain panel).
MetricKind = Literal["detection", "verdict", "capture", "diff", "generic"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = _REPO_ROOT / "eval_runs"


# ───────────────────────── per-attempt / per-case ─────────────────────────


@dataclass
class AttemptOutcome:
    """The result of running the unit-under-test once for a case.

    ``passed`` is the only field :func:`pass_at` needs; ``metrics`` and
    ``detail`` are carried through so the representative attempt can populate
    the case record. ``metrics`` is the standard analytics bag (tokens, tool
    calls, …); ``detail`` is domain-specific (verdict, tp/fp/fn, captured, …).
    """

    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": bool(self.passed), "metrics": self.metrics, "detail": self.detail}


@dataclass
class CaseResult:
    """One scored case, aggregated over ``attempts`` pass@X repetitions."""

    id: str
    passed: bool           # pass@X: any attempt passed
    pass_count: int        # how many attempts passed
    attempts: int          # X
    metrics: dict[str, Any] = field(default_factory=dict)   # representative attempt
    detail: dict[str, Any] = field(default_factory=dict)    # representative attempt
    # Per-attempt breakdown, present only when attempts > 1.
    runs: Optional[list[dict[str, Any]]] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "passed": bool(self.passed),
            "pass_count": int(self.pass_count),
            "attempts": int(self.attempts),
            "metrics": self.metrics,
            "detail": self.detail,
        }
        if self.runs is not None:
            d["runs"] = self.runs
        return d


@dataclass
class FixtureResult:
    """A group of cases (usually one fixture / target codebase)."""

    slug: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def cases_total(self) -> int:
        return len(self.cases)

    @property
    def cases_passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "cases_total": self.cases_total,
            "cases_passed": self.cases_passed,
            "cases": [c.to_dict() for c in self.cases],
        }


@dataclass
class EvalRun:
    """A whole eval run — the thing serialized to ``eval_results.json``."""

    scenario: Scenario
    unit: str                          # agent / task-template / workflow name
    pass_at: int
    metric_kind: MetricKind = "generic"
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    timestamp: Optional[str] = None
    fixtures: list[FixtureResult] = field(default_factory=list)
    # Free-form run-level extras (e.g. agent_variant, axes, notes).
    meta: dict[str, Any] = field(default_factory=dict)

    def to_envelope(self) -> dict[str, Any]:
        fixtures = [f.to_dict() for f in self.fixtures]
        body = {
            "schema": SCHEMA,
            "scenario": self.scenario,
            "unit": self.unit,
            "metric_kind": self.metric_kind,
            "pass_at": int(self.pass_at),
            "model": self.model,
            "prompt_version": self.prompt_version,
            "timestamp": self.timestamp or _now_iso(),
            "meta": self.meta,
            "fixtures": fixtures,
        }
        # Embed a derived snapshot so the JSON is self-describing offline; the
        # UI recomputes the same values from ``fixtures`` for legacy files.
        body["headline"] = derive_headline(self.metric_kind, fixtures)
        body["totals"] = derive_totals(fixtures)
        return body


# ───────────────────────── pass@X repeater ─────────────────────────

AttemptFn = Callable[[int], Awaitable[AttemptOutcome]]


async def pass_at(case_id: str, attempt_fn: AttemptFn, n: int) -> CaseResult:
    """Run ``attempt_fn(i)`` ``n`` times and fold into a :class:`CaseResult`.

    ``attempt_fn`` receives the 0-based attempt index and returns an
    :class:`AttemptOutcome`. The case passes if any attempt passes; the
    representative attempt (whose metrics/detail populate the case) is the
    first passing one, else the first attempt.
    """
    if n < 1:
        raise ValueError("pass_at n must be >= 1")
    outcomes: list[AttemptOutcome] = []
    for i in range(n):
        outcomes.append(await attempt_fn(i))

    pass_count = sum(1 for o in outcomes if o.passed)
    rep = next((o for o in outcomes if o.passed), outcomes[0])
    return CaseResult(
        id=case_id,
        passed=pass_count > 0,
        pass_count=pass_count,
        attempts=n,
        metrics=rep.metrics,
        detail=rep.detail,
        runs=[o.to_dict() for o in outcomes] if n > 1 else None,
    )


# ───────────────────────── headline / totals derivation ─────────────────────────


def _round(x: float, n: int = 3) -> float:
    return round(x, n)


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": _round(p), "recall": _round(r), "f1": _round(f1)}


def _iter_cases(fixtures: list[dict[str, Any]]):
    for f in fixtures:
        for c in f.get("cases") or []:
            yield c


def derive_headline(metric_kind: str, fixtures: list[dict[str, Any]]) -> dict[str, Any]:
    """The scalars the eval card shows. Always pass@X; plus domain extras."""
    cases = list(_iter_cases(fixtures))
    total = len(cases)
    passed = sum(1 for c in cases if c.get("passed"))
    head: dict[str, Any] = {
        "pass_rate": _round(passed / total, 3) if total else 0.0,
        "passed": passed,
        "total": total,
    }
    if metric_kind == "detection":
        tp = sum(int((c.get("detail") or {}).get("tp", 0) or 0) for c in cases)
        fp = sum(int((c.get("detail") or {}).get("fp", 0) or 0) for c in cases)
        fn = sum(int((c.get("detail") or {}).get("fn", 0) or 0) for c in cases)
        head.update(_prf(tp, fp, fn))
    elif metric_kind == "verdict":
        ev = sum(1 for c in cases if (c.get("detail") or {}).get("has_evidence"))
        head["evidence_rate"] = _round(ev / total, 3) if total else 0.0
    elif metric_kind == "capture":
        ch = sum(1 for c in cases if (c.get("detail") or {}).get("chain"))
        head["chain_rate"] = _round(ch / total, 3) if total else 0.0
    elif metric_kind == "diff":
        f1s = [float((c.get("detail") or {}).get("f1", 0) or 0) for c in cases]
        head["mean_f1"] = _round(sum(f1s) / len(f1s), 3) if f1s else 0.0
    return head


def derive_totals(fixtures: list[dict[str, Any]]) -> dict[str, Any]:
    cases = list(_iter_cases(fixtures))
    in_tok = out_tok = tot_tok = tool_calls = llm_calls = http = 0
    dur = 0.0
    tools: Counter[str] = Counter()
    for c in cases:
        m = c.get("metrics") or {}
        in_tok += int(m.get("input_tokens", 0) or 0)
        out_tok += int(m.get("output_tokens", 0) or 0)
        tot_tok += int(m.get("total_tokens", 0) or 0)
        tool_calls += int(m.get("total_tool_calls", m.get("tool_calls", 0)) or 0)
        llm_calls += int(m.get("llm_calls", 0) or 0)
        http += int(m.get("http_requests", 0) or 0)
        dur += float(m.get("duration_s", 0) or 0)
        tc = m.get("tool_counts")
        if isinstance(tc, dict):
            for k, v in tc.items():
                if isinstance(v, (int, float)):
                    tools[k] += int(v)
    return {
        "fixtures": len(fixtures),
        "cases": len(cases),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": tot_tok or (in_tok + out_tok),
        "total_tool_calls": tool_calls,
        "llm_calls": llm_calls,
        "http_requests": http,
        "duration_s": _round(dur, 1),
        "tool_counts": dict(tools),
    }


# ───────────────────────── metrics helper ─────────────────────────


def metrics_from_task(metrics: dict[str, Any]) -> dict[str, Any]:
    """Fold a ``{task_ref: TaskMetrics}`` mapping (from ``task_harness``) into
    the standard per-case metrics bag."""
    tool_counts: Counter[str] = Counter()
    tot = llm = in_tok = out_tok = toks = 0
    dur_ms = 0.0
    for m in metrics.values():
        tool_counts.update(getattr(m, "tool_counts", {}) or {})
        tot += int(getattr(m, "total_tool_calls", 0) or 0)
        llm += int(getattr(m, "llm_calls", 0) or 0)
        in_tok += int(getattr(m, "input_tokens", 0) or 0)
        out_tok += int(getattr(m, "output_tokens", 0) or 0)
        toks += int(getattr(m, "total_tokens", 0) or 0)
        dur_ms += float(getattr(m, "tool_time_ms", 0) or 0)
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": toks or (in_tok + out_tok),
        "total_tool_calls": tot,
        "llm_calls": llm,
        "tool_time_ms": _round(dur_ms, 1),
        "tool_counts": dict(tool_counts),
    }


# ───────────────────────── serialization ─────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_eval_results(run: EvalRun, out_dir: Path | str) -> Path:
    """Write ``run`` to ``<out_dir>/eval_results.json`` and return the path.

    ``out_dir`` may be absolute or relative; a bare name is placed under
    ``eval_runs/``. The directory is created if missing.
    """
    out = Path(out_dir)
    if not out.is_absolute() and out_dir == out.name:
        out = EVAL_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    path = out / "eval_results.json"
    path.write_text(
        json.dumps(run.to_envelope(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
