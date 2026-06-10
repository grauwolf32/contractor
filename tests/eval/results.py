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
import logging
import os
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

SCHEMA = "eval/v1"

Scenario = Literal["agent", "task", "pipeline"]
# How a case is scored — drives which domain panel the UI shows. "generic"
# means pass/fail only (no domain panel).
MetricKind = Literal["detection", "verdict", "capture", "diff", "generic"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = _REPO_ROOT / "eval_runs"


def _compute_run_stamp() -> str:
    """A per-process run id used to namespace the *archive* of every run so
    results are never overwritten. ``CONTRACTOR_EVAL_RUN_STAMP`` overrides it
    (e.g. to label an A/B: ``0607-qw3off``); otherwise ``mmdd-HHMMSS`` (UTC).
    Computed once at import — one pytest process == one run == one stamp.
    """
    env = os.environ.get("CONTRACTOR_EVAL_RUN_STAMP")
    if env:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in env)
    return datetime.now(UTC).strftime("%m%d-%H%M%S")


# Per-run archive namespace (never overwritten). Re-read via the module global
# so tests can monkeypatch it; production sets it once at import.
RUN_STAMP = _compute_run_stamp()


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
    runs: list[dict[str, Any]] | None = None

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
    """A group of cases (usually one fixture / target codebase).

    ``tokens`` / ``latency_ms`` are *optional* per-fixture cost measurements
    (total tokens spent and wall-clock latency for the run that produced this
    fixture). They are summed into the envelope ``totals`` by
    :func:`derive_totals` only when at least one fixture supplies them; left
    ``None`` everywhere they are absent, so the envelope is unchanged for
    producers that don't measure cost.
    """

    slug: str
    cases: list[CaseResult] = field(default_factory=list)
    tokens: int | None = None
    latency_ms: float | None = None

    @property
    def cases_total(self) -> int:
        return len(self.cases)

    @property
    def cases_passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "slug": self.slug,
            "cases_total": self.cases_total,
            "cases_passed": self.cases_passed,
            "cases": [c.to_dict() for c in self.cases],
        }
        # Emit cost fields only when measured, so the per-fixture shape is
        # byte-for-byte unchanged for producers that don't supply them.
        if self.tokens is not None:
            d["tokens"] = int(self.tokens)
        if self.latency_ms is not None:
            d["latency_ms"] = float(self.latency_ms)
        return d


@dataclass
class EvalRun:
    """A whole eval run — the thing serialized to ``eval_results.json``."""

    scenario: Scenario
    unit: str                          # agent / task-template / workflow name
    pass_at: int
    metric_kind: MetricKind = "generic"
    model: str | None = None
    prompt_version: str | None = None
    timestamp: str | None = None
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
        yield from f.get("cases") or []


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
    in_tok = out_tok = tot_tok = tool_calls = tool_errors = llm_calls = http = skill_reads = 0
    dur = 0.0
    tools: Counter[str] = Counter()
    for c in cases:
        m = c.get("metrics") or {}
        in_tok += int(m.get("input_tokens", 0) or 0)
        out_tok += int(m.get("output_tokens", 0) or 0)
        tot_tok += int(m.get("total_tokens", 0) or 0)
        tool_calls += int(m.get("total_tool_calls", m.get("tool_calls", 0)) or 0)
        tool_errors += int(m.get("tool_errors", 0) or 0)
        llm_calls += int(m.get("llm_calls", 0) or 0)
        http += int(m.get("http_requests", 0) or 0)
        dur += float(m.get("duration_s", 0) or 0)
        tc = m.get("tool_counts")
        if isinstance(tc, dict):
            for k, v in tc.items():
                if isinstance(v, (int, float)):
                    tools[k] += int(v)
            skill_reads += int(tc.get("skills_read", 0) or 0)
    totals: dict[str, Any] = {
        "fixtures": len(fixtures),
        "cases": len(cases),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": tot_tok or (in_tok + out_tok),
        "total_tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "llm_calls": llm_calls,
        "http_requests": http,
        "skill_reads": skill_reads,
        "duration_s": _round(dur, 1),
        "tool_counts": dict(tools),
    }
    # Optional per-fixture cost (QW4/F2): surface summed tokens + latency only
    # when at least one fixture measured it. Absent on all → no new keys, so the
    # envelope is byte-for-byte back-compatible for non-measuring producers.
    cost_tokens = [f["tokens"] for f in fixtures if f.get("tokens") is not None]
    cost_latency = [f["latency_ms"] for f in fixtures if f.get("latency_ms") is not None]
    if cost_tokens:
        totals["tokens"] = sum(int(t) for t in cost_tokens)
    if cost_latency:
        totals["latency_ms"] = _round(sum(float(latency) for latency in cost_latency), 1)
    return totals


# ───────────────────────── metrics helper ─────────────────────────


def metrics_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold ADK metrics-plugin events (``AgentRun.metrics_events`` /
    ``metrics.jsonl`` rows) into the standard per-case metrics bag.

    Each event is a dict with ``event_type`` ∈ {tool_call, tool_result,
    llm_usage}. Tolerant of an empty list (→ zeros) so harnesses that don't
    attach the metrics plugin still produce a valid case record.
    """
    tool_counts: Counter[str] = Counter()
    tot = llm = in_tok = out_tok = toks = errors = 0
    dur_ms = 0.0
    for ev in events or []:
        et = str(ev.get("event_type", ""))
        if et == "tool_call":
            name = str(ev.get("tool_name", ""))
            if name:
                tool_counts[name] += 1
                tot += 1
        elif et == "tool_result":
            dur_ms += float(ev.get("execution_time_ms", 0) or 0)
            if ev.get("result_error"):
                errors += 1
        elif et == "tool_exception":
            errors += 1
        elif et == "llm_usage":
            usage = ev.get("usage") or {}
            llm += 1
            in_tok += int(usage.get("input", 0) or 0)
            out_tok += int(usage.get("output", 0) or 0)
            toks += int(usage.get("total", 0) or 0)
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": toks or (in_tok + out_tok),
        "total_tool_calls": tot,
        "tool_errors": errors,
        "llm_calls": llm,
        "tool_time_ms": _round(dur_ms, 1),
        "tool_counts": dict(tool_counts),
    }


def metrics_from_task(metrics: dict[str, Any]) -> dict[str, Any]:
    """Fold a ``{task_ref: TaskMetrics}`` mapping (from ``task_harness``) into
    the standard per-case metrics bag."""
    tool_counts: Counter[str] = Counter()
    tot = llm = in_tok = out_tok = toks = errors = 0
    dur_ms = 0.0
    for m in metrics.values():
        tool_counts.update(getattr(m, "tool_counts", {}) or {})
        tot += int(getattr(m, "total_tool_calls", 0) or 0)
        errors += int(getattr(m, "tool_errors", 0) or 0)
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
        "tool_errors": errors,
        "llm_calls": llm,
        "tool_time_ms": _round(dur_ms, 1),
        "tool_counts": dict(tool_counts),
    }


# ───────────────────────── live trace location ─────────────────────────


def _run_slug(scenario: str, unit: str, fixture: str | None = None) -> str:
    """``<scenario>-<unit>[-eval-<fixture>]`` — the per-fixture archive folder."""
    slug = f"{scenario}-{_safe_name(unit)}"
    if fixture:
        slug += f"-eval-{_safe_name(fixture)}"
    return slug


def run_archive_dir(
    scenario: str, unit: str, fixture: str | None = None, *, stamp: str | None = None
) -> Path:
    """Dated, never-overwritten archive dir for one run:
    ``eval_runs/<RUN_STAMP>/<scenario>-<unit>-eval-<fixture>/``.

    ``RUN_STAMP`` is per-process, so every run lands in its own folder and no
    eval information is ever overwritten (the data-loss fix). The flat
    ``eval_runs/<unit>/`` path is kept separately as a "latest" pointer.
    """
    return EVAL_ROOT / (stamp or RUN_STAMP) / _run_slug(scenario, unit, fixture)


def case_artifact_dir(unit: str, fixture: str, case_id: str, *, scenario: str = "agent") -> Path:
    """Directory for a case's *live* on-disk artifact trace, co-located with the
    eval_sink per-case metrics under the dated archive:
    ``eval_runs/<RUN_STAMP>/<scenario>-<unit>-eval-<fixture>/cases/<case>/artifacts``.

    Pass this as ``artifact_dir`` to ``run_agent`` / ``run_task_pipeline`` so the
    full ADK artifact tree persists during the run (survives a timeout/crash),
    sitting next to the ``metrics.json`` that :class:`EvalSink` writes afterward.
    """
    return run_archive_dir(scenario, unit, fixture) / "cases" / _safe_name(case_id) / "artifacts"


# ───────────────────────── serialization ─────────────────────────


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_eval_results(run: EvalRun, out_dir: Path | str) -> Path:
    """Write ``run`` to ``<out_dir>/eval_results.json`` and return the path.

    ``out_dir`` may be absolute or relative; a bare name is placed under
    ``eval_runs/``. The directory is created if missing.
    """
    out = Path(out_dir)
    if not out.is_absolute() and str(out_dir) == out.name:
        out = EVAL_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    path = out / "eval_results.json"
    path.write_text(
        json.dumps(run.to_envelope(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


# ───────────────────────── session sink (for pytest gates) ─────────────────────────


def _safe_name(unit: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in unit)


def _default_run_name(scenario: str, unit: str, metric_kind: str) -> str:
    """Default "latest pointer" dir for one ``(scenario, unit, metric_kind)``
    bucket: ``<scenario>-<unit>[-<metric_kind>]`` (metric_kind only when it
    isn't the ``generic`` default), matching the established
    ``<scenario>-<unit>-eval-<fixture>`` archive naming. Buckets are keyed on
    all three fields, so the default name must carry all three — a bare
    ``_safe_name(unit)`` made two buckets sharing a unit overwrite each
    other's ``eval_runs/<unit>/eval_results.json`` in the same flush.
    """
    name = _run_slug(scenario, unit)
    if metric_kind != "generic":
        name += f"-{metric_kind}"
    return name


class EvalSink:
    """Accumulates per-case results across a pytest session and, on flush,
    writes one ``eval_results.json`` envelope per ``(scenario, unit,
    metric_kind)`` group.

    Per-fixture pytest evals are isolated test invocations, so they can't build
    an aggregate run themselves. Each records a single :class:`CaseResult` here
    via :meth:`record`; the session-scoped fixture flushes them at the end. A
    partial pytest selection simply produces a partial (but valid) envelope.
    """

    def __init__(self) -> None:
        self._runs: dict[tuple[str, str, str], dict[str, Any]] = {}

    def record(
        self,
        *,
        scenario: Scenario,
        unit: str,
        metric_kind: MetricKind,
        fixture: str,
        case: CaseResult,
        model: str | None = None,
        prompt_version: str | None = None,
        pass_at: int = 1,
        run_name: str | None = None,
        meta: dict[str, Any] | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> None:
        # Key on metric_kind too: detection/diff/verdict cases carry
        # incompatible `detail` shapes, so two record() calls sharing
        # (scenario, unit) but differing in metric_kind must not merge into one
        # bucket (the first kind would silently win via setdefault).
        key = (scenario, unit, metric_kind)
        run = self._runs.setdefault(key, {
            "scenario": scenario, "unit": unit, "metric_kind": metric_kind,
            "model": model, "prompt_version": prompt_version, "pass_at": pass_at,
            "run_name": run_name or _default_run_name(scenario, unit, metric_kind),
            "meta": meta or {},
            "fixtures": {},
        })
        # The first record() seeds model/prompt_version for the whole bucket;
        # backfill missing values and warn loudly when later cases disagree
        # (the envelope can only carry one value per run).
        for field_name, value in (("model", model), ("prompt_version", prompt_version)):
            if value is None:
                continue
            if run[field_name] is None:
                run[field_name] = value
            elif run[field_name] != value:
                logger.warning(
                    "eval_sink: bucket %r already has %s=%r; case %r recorded %r "
                    "(first value wins in the envelope)",
                    key, field_name, run[field_name], case.id, value,
                )
        run["fixtures"].setdefault(fixture, []).append(case)
        run["pass_at"] = max(run["pass_at"], pass_at)
        # Persist this case immediately (crash-safe) into the dated, never-
        # overwritten archive: eval_runs/<RUN_STAMP>/<scenario>-<unit>-eval-<fixture>/cases/<case>/.
        self._persist_case(scenario, unit, fixture, case, artifacts)

    @staticmethod
    def _persist_case(scenario: str, unit: str, fixture: str, case: CaseResult,
                      artifacts: dict[str, str] | None) -> None:
        base = run_archive_dir(scenario, unit, fixture) / "cases" / _safe_name(case.id)
        base.mkdir(parents=True, exist_ok=True)
        (base / "metrics.json").write_text(
            json.dumps({"fixture": fixture, **case.to_dict()}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        for name, text in (artifacts or {}).items():
            if not text:
                continue
            (base / _safe_name(name)).write_text(
                text if isinstance(text, str) else json.dumps(text, default=str, ensure_ascii=False),
                encoding="utf-8")

    def flush(self) -> list[Path]:
        paths = []
        for run in self._runs.values():
            fixtures = [FixtureResult(slug=s, cases=cs)
                        for s, cs in run["fixtures"].items()]

            def _mk(fxs: list[FixtureResult], _run: dict[str, Any] = run) -> EvalRun:
                return EvalRun(
                    scenario=_run["scenario"], unit=_run["unit"], pass_at=_run["pass_at"],
                    metric_kind=_run["metric_kind"], model=_run["model"],
                    prompt_version=_run["prompt_version"], fixtures=fxs, meta=_run["meta"],
                )

            # (1) "latest" pointer — overwritten each run, at the stable
            #     eval_runs/<scenario>-<unit>[-<metric_kind>]/ path (or the
            #     caller-supplied run_name) that analytics-ui reads.
            paths.append(write_eval_results(_mk(fixtures), run["run_name"]))
            # (2) dated, per-fixture archive — NEVER overwritten (one folder per
            #     run via RUN_STAMP), so eval history is never lost.
            for fx in fixtures:
                paths.append(write_eval_results(
                    _mk([fx]),
                    run_archive_dir(run["scenario"], run["unit"], fx.slug),
                ))
        return paths
