"""Unit tests for optional per-fixture cost (tokens + latency) in eval/v1
``totals`` (QW4/F2).

The cost fields are *optional*: when at least one fixture supplies them
:func:`derive_totals` sums them into ``totals``; when no fixture supplies them
the envelope is byte-for-byte unchanged (no new keys), so the change is fully
back-compatible.
"""
from __future__ import annotations

from tests.eval.results import (
    CaseResult,
    EvalRun,
    FixtureResult,
    derive_totals,
)


def _case(cid: str, passed: bool = True) -> CaseResult:
    return CaseResult(cid, passed, 1 if passed else 0, 1, metrics={"total_tokens": 10})


def test_totals_sum_tokens_and_latency_when_present():
    run = EvalRun(
        scenario="agent", unit="codereview_agent", pass_at=1, metric_kind="generic",
        fixtures=[
            FixtureResult("alpha", [_case("alpha")], tokens=1_000, latency_ms=1_200.0),
            FixtureResult("beta", [_case("beta")], tokens=2_500, latency_ms=800.5),
        ],
    )
    env = run.to_envelope()
    totals = env["totals"]

    assert totals["tokens"] == 3_500
    assert totals["latency_ms"] == 2_000.5
    # Existing keys untouched.
    assert totals["fixtures"] == 2
    assert totals["cases"] == 2

    # Per-fixture serialization carries the cost fields when set.
    fixtures = {f["slug"]: f for f in env["fixtures"]}
    assert fixtures["alpha"]["tokens"] == 1_000
    assert fixtures["alpha"]["latency_ms"] == 1_200.0


def test_totals_partial_cost_sums_only_present_fixtures():
    # Only one fixture measured cost; the other contributes nothing.
    run = EvalRun(
        scenario="agent", unit="codereview_agent", pass_at=1, metric_kind="generic",
        fixtures=[
            FixtureResult("alpha", [_case("alpha")], tokens=1_000, latency_ms=500.0),
            FixtureResult("beta", [_case("beta")]),  # no cost
        ],
    )
    totals = run.to_envelope()["totals"]
    assert totals["tokens"] == 1_000
    assert totals["latency_ms"] == 500.0


def test_totals_unchanged_when_no_cost_fields():
    run = EvalRun(
        scenario="agent", unit="codereview_agent", pass_at=1, metric_kind="generic",
        fixtures=[
            FixtureResult("alpha", [_case("alpha")]),
            FixtureResult("beta", [_case("beta", passed=False)]),
        ],
    )
    env = run.to_envelope()
    totals = env["totals"]

    # No cost keys when absent on all fixtures — schema stable / back-compatible.
    assert "tokens" not in totals
    assert "latency_ms" not in totals

    # And the per-fixture dicts keep their legacy shape (no cost keys).
    for f in env["fixtures"]:
        assert "tokens" not in f
        assert "latency_ms" not in f

    # derive_totals over the same fixtures matches: stable, no crash.
    assert derive_totals(env["fixtures"]) == totals


def test_derive_totals_empty_run_has_no_cost_keys():
    env = EvalRun(scenario="pipeline", unit="trace", pass_at=1,
                  metric_kind="diff", fixtures=[]).to_envelope()
    assert "tokens" not in env["totals"]
    assert "latency_ms" not in env["totals"]
