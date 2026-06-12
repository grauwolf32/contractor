#!/usr/bin/env python3
"""Run a vuln-finding *workflow* end-to-end on eval fixtures and score detection.

Pipeline-scenario counterpart to ``run_vuln_eval.py`` (agent) and
``run_trace_task_eval.py`` (task): each run spins up a whole workflow —
``vuln-sweep``, ``trace-postdiff``, or any registry entry that persists
findings via ``report_vulnerability`` / trace result texts — against a
fixture's source tree, then scores the reported vulnerabilities with
``tests.eval.trace_vuln_scoring`` against ``vulnerabilities.expected.json``.

A timeout salvages partial results: scoring reads the on-disk artifact
store, so whatever the workflow persisted before the deadline still counts.

Usage::

    poetry run python scripts/run_pipeline_vuln_eval.py
    poetry run python scripts/run_pipeline_vuln_eval.py \
        --workflows vuln-sweep,trace-postdiff --fixtures vulnyapi --runs 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_WORKFLOWS = "vuln-sweep,trace-postdiff"
DEFAULT_FIXTURES = "vulnyapi"

logger = logging.getLogger("pipeline_vuln_eval")


def _resolve_model() -> str:
    override = os.environ.get("CONTRACTOR_EVAL_MODEL")
    if override:
        return override
    from contractor.utils.settings import DEFAULT_MODEL

    return str(DEFAULT_MODEL.model)


async def _run_workflow(
    *,
    workflow_name: str,
    fixture: Any,
    fixture_dir: Path,
    model_alias: str,
    timeout_s: float,
    artifact_root: Path,
) -> dict[str, Any]:
    """One workflow run; returns run stats. Artifacts land in *artifact_root*."""
    from google.adk.artifacts import FileArtifactService

    from cli.fs import RootedLocalFileSystem
    from contractor.utils import observability
    from contractor.workflows import WorkflowContext, get_workflows

    workflow_cls = get_workflows()[workflow_name]

    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_service = FileArtifactService(root_dir=str(artifact_root))
    fs = RootedLocalFileSystem(str(fixture.source_root))

    oas_path = fixture_dir / "oas.expected.yaml"
    oas_yaml = oas_path.read_text(encoding="utf-8") if oas_path.is_file() else None

    user_id = "eval-user"
    ctx = WorkflowContext(
        project_path=fixture.source_root,
        folder_name="/",
        model=model_alias,
        app_name="eval-pipeline",
        user_id=user_id,
        artifact_service=artifact_service,
        fs=fs,
        artifact=oas_yaml,
        # Local single-GPU backends serialize concurrent requests (the sweep
        # fans out), so a queued call can far exceed the 300s default.
        timeout=600,
    )
    workflow = workflow_cls(ctx)

    events: list[dict[str, Any]] = []

    async def _on_event(event: Any) -> None:
        events.append(
            {"type": event.type, "task": event.task_name, **event.payload}
        )

    timed_out = False
    error: str | None = None
    t0 = time.monotonic()
    with observability.run_context(
        name=f"eval.pipeline.{workflow_name}",
        session_id=f"pipeline-eval-{fixture.slug}-{workflow_name}",
        tags=["eval", "pipeline", f"workflow:{workflow_name}"],
        metadata={"fixture": fixture.slug, "workflow": workflow_name},
    ):
        try:
            await asyncio.wait_for(
                workflow.run(user_id=user_id, on_event=_on_event),
                timeout=timeout_s,
            )
        except TimeoutError:
            timed_out = True
            logger.warning(
                "%s on %s timed out after %.0fs — scoring partial artifacts",
                workflow_name,
                fixture.slug,
                timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 — salvage partial artifacts
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("%s on %s failed", workflow_name, fixture.slug)
    wallclock_s = time.monotonic() - t0

    events_path = artifact_root / "events.jsonl"
    with events_path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev, default=str) + "\n")

    in_tok = sum(
        int(ev.get("input_tokens", 0) or 0)
        for ev in events
        if ev.get("type") == "metrics_llm_usage"
    )
    out_tok = sum(
        int(ev.get("output_tokens", 0) or 0)
        for ev in events
        if ev.get("type") == "metrics_llm_usage"
    )
    llm_calls = sum(1 for ev in events if ev.get("type") == "metrics_llm_usage")
    tool_calls = sum(1 for ev in events if ev.get("type") == "metrics_tool_call")

    return {
        "wallclock_s": round(wallclock_s, 1),
        "timed_out": timed_out,
        "error": error,
        "event_count": len(events),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
    }


# Multi-stage workflows persist intermediate findings (e.g. vuln-sweep's
# low-confidence nominations) under the same ``vulnerability-reports/`` tree
# as their final output. Score only the final stage's namespaces — counting
# nominations as output would credit the pipeline for findings it never
# confirmed. ``diagnostic`` namespaces get a separate recall-only score
# (the nomination stage's recall is the trace stage's upper bound).
_STAGE_FILTERS: dict[str, dict[str, str | None]] = {
    "vuln-sweep": {"final": ":trace:", "diagnostic": ":sweep:"},
    # Single producer stage, but report-artifact findings still need the
    # per-instance extractor (the blanket one collapses same-family reports).
    "trace-postdiff": {"final": "trace-postdiff:", "diagnostic": None},
}


def _extract_reports(artifacts_dir: Path, *, namespace_substr: str = "") -> list[Any]:
    """ReportedVulns from report-tool artifacts whose namespace dir matches.

    Unlike ``extract_from_run_dir``, dedup keys include the finding's code
    location (``place``): report-tool findings carry no API path, and a
    ``(family, "")`` key would collapse every same-family finding into one —
    capping measurable recall at one TP per family. Path attribution is
    attempted from ``METHOD /path`` mentions in the finding text.
    """
    import yaml

    from tests.eval.trace_vuln_scoring import (
        _METHODPATH_RE,
        ReportedVuln,
        _latest_versions,
        _norm_path,
        family_of_text,
    )

    by_key: dict[tuple[str, str, str], Any] = {}
    pattern = str(artifacts_dir / "**" / "vulnerability-reports" / "**" / "versions" / "*" / "*")
    for f in _latest_versions(pattern):
        if f.endswith("metadata.json"):
            continue
        namespace = f.split("/vulnerability-reports/", 1)[-1].split("/versions/", 1)[0]
        if namespace_substr and namespace_substr not in namespace:
            continue
        try:
            reports = yaml.safe_load(Path(f).read_text(encoding="utf-8", errors="ignore")) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(reports, dict):
            continue
        for name, r in reports.items():
            if not isinstance(r, dict):
                continue
            text = f"{r.get('title', '')} {r.get('details', '')} {r.get('summary', '')}"
            fam = family_of_text(text)
            if fam is None:
                continue
            pm = _METHODPATH_RE.search(text)
            path = _norm_path(pm.group(2)) if pm else ""
            place = str(r.get("place", "") or name)
            rv = ReportedVuln(
                family=fam, path=path, title=str(r.get("title", ""))[:120],
                source="report_tool",
            )
            by_key.setdefault((fam, path, place), rv)
    return list(by_key.values())


def _extract_trace_results(artifacts_dir: Path) -> list[Any]:
    """ReportedVulns from per-finding trace result texts.

    Fan-out trace phases publish under ``trace_annotation/<slug>/result``
    (unique ``artifact_key`` per finding), which the stock
    ``trace_annotation/result`` glob in ``extract_from_run_dir`` never
    matches. A trace task that confirms a finding often expresses it only
    as a Shape block in its result text — without this, the trace stage's
    confirmations are invisible to scoring.
    """
    from tests.eval.trace_vuln_scoring import _latest_versions, extract_from_result

    out: list[Any] = []
    pattern = str(
        artifacts_dir / "**" / "trace_annotation" / "**" / "result" / "versions" / "*" / "result"
    )
    for f in _latest_versions(pattern):
        out.extend(
            extract_from_result(Path(f).read_text(encoding="utf-8", errors="ignore"))
        )
    return out


def _score_run(
    artifact_root: Path, fixture_dir: Path, workflow_name: str
) -> tuple[Any, Any | None]:
    """Score the workflow's final output; returns ``(final, diagnostic)``.

    ``diagnostic`` is the intermediate stage's score (None for single-stage
    workflows, which fall back to the blanket extractor).
    """
    from tests.eval.trace_vuln_scoring import (
        extract_from_run_dir,
        load_expected,
        score_vulns,
    )

    expected = load_expected(fixture_dir / "vulnerabilities.expected.json")
    stage = _STAGE_FILTERS.get(workflow_name)
    if stage is None:
        return score_vulns(extract_from_run_dir(artifact_root), expected), None

    # Final output = trace-stage report artifacts ∪ trace result Shape
    # blocks, deduped by (family, path) so a finding expressed both ways
    # doesn't double-count as an FP.
    by_key: dict[tuple[str, str], Any] = {}
    for rv in (
        _extract_reports(artifact_root, namespace_substr=stage["final"])
        + _extract_trace_results(artifact_root)
    ):
        by_key.setdefault(rv.key(), rv)
    final = score_vulns(list(by_key.values()), expected)
    diagnostic = None
    if stage["diagnostic"] is not None:
        diagnostic = score_vulns(
            _extract_reports(artifact_root, namespace_substr=stage["diagnostic"]),
            load_expected(fixture_dir / "vulnerabilities.expected.json"),
        )
    return final, diagnostic


async def _eval_unit(
    *,
    workflow_name: str,
    fixture_slugs: list[str],
    model_alias: str,
    runs: int,
    timeout_s: float,
    min_precision: float,
    min_recall: float,
) -> Path | None:
    from tests.eval.conftest import FIXTURES_ROOT, _load_fixture
    from tests.eval.results import (
        CaseResult,
        EvalRun,
        FixtureResult,
        case_artifact_dir,
        write_eval_results,
    )

    fixture_results: list[FixtureResult] = []

    for slug in fixture_slugs:
        fixture = _load_fixture(slug)
        fixture_dir = FIXTURES_ROOT / slug
        expected_path = fixture_dir / "vulnerabilities.expected.json"
        if not expected_path.is_file():
            logger.warning("%s has no vulnerabilities.expected.json — skipping", slug)
            continue

        attempts: list[dict[str, Any]] = []
        for i in range(1, runs + 1):
            artifact_root = case_artifact_dir(
                workflow_name, slug, f"attempt-{i}", scenario="pipeline"
            )
            print(f"\n→ {workflow_name} on {slug} (attempt {i}/{runs}) …", flush=True)
            stats = await _run_workflow(
                workflow_name=workflow_name,
                fixture=fixture,
                fixture_dir=fixture_dir,
                model_alias=model_alias,
                timeout_s=timeout_s,
                artifact_root=artifact_root,
            )
            score, diag = _score_run(artifact_root, fixture_dir, workflow_name)
            attempts.append({"stats": stats, "score": score, "diag": diag})
            flags = (
                (" TIMEOUT" if stats["timed_out"] else "")
                + (f" ERROR={stats['error']}" if stats["error"] else "")
            )
            print(
                f"  {score.explain()}  wallclock={stats['wallclock_s']}s"
                f" tokens={stats['input_tokens']}+{stats['output_tokens']}{flags}",
                flush=True,
            )
            if diag is not None:
                print(f"  nomination stage: {diag.explain()}", flush=True)

        best = max(attempts, key=lambda a: a["score"].f1)
        bs = best["score"]
        pass_count = sum(
            1
            for a in attempts
            if a["score"].precision >= min_precision
            and a["score"].recall >= min_recall
        )
        case = CaseResult(
            id=slug,
            passed=pass_count > 0,
            pass_count=pass_count,
            attempts=len(attempts),
            metrics={
                "duration_s": best["stats"]["wallclock_s"],
                "input_tokens": best["stats"]["input_tokens"],
                "output_tokens": best["stats"]["output_tokens"],
                "llm_calls": best["stats"]["llm_calls"],
                "tool_calls": best["stats"]["tool_calls"],
            },
            detail={
                "tp": bs.tp,
                "fp": bs.fp,
                "fn": bs.fn,
                "precision": bs.precision,
                "recall": bs.recall,
                "f1": bs.f1,
                "per_family": bs.per_family,
                "matched": bs.matched,
                "missed": bs.missed,
                "extra": bs.extra,
                "timed_out": best["stats"]["timed_out"],
                "error": best["stats"]["error"],
                "nomination_stage": (
                    {
                        "tp": best["diag"].tp,
                        "fn": best["diag"].fn,
                        "recall": best["diag"].recall,
                        "missed": best["diag"].missed,
                    }
                    if best["diag"] is not None
                    else None
                ),
                "all_attempts": [
                    {
                        "precision": a["score"].precision,
                        "recall": a["score"].recall,
                        "f1": a["score"].f1,
                        "duration_s": a["stats"]["wallclock_s"],
                        "timed_out": a["stats"]["timed_out"],
                        "error": a["stats"]["error"],
                    }
                    for a in attempts
                ],
            },
        )
        fixture_results.append(FixtureResult(slug=slug, cases=[case]))

    if not fixture_results:
        logger.warning("no scoreable fixtures for %s", workflow_name)
        return None

    env_path = write_eval_results(
        EvalRun(
            scenario="pipeline",
            unit=workflow_name,
            pass_at=runs,
            metric_kind="detection",
            model=model_alias,
            fixtures=fixture_results,
            meta={
                "min_precision": min_precision,
                "min_recall": min_recall,
                "timeout_s": timeout_s,
            },
        ),
        f"pipeline-{workflow_name}",
    )
    print(f"\nenvelope: {env_path}")
    return env_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflows", default=DEFAULT_WORKFLOWS)
    parser.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    parser.add_argument("--runs", type=int, default=1, help="attempts per fixture (pass@N)")
    parser.add_argument("--timeout", type=float, default=5400.0, help="per-run timeout (s)")
    parser.add_argument("--min-precision", type=float, default=0.3)
    parser.add_argument("--min-recall", type=float, default=0.4)
    parser.add_argument(
        "--score-only",
        metavar="ARTIFACT_DIR",
        help="re-score an existing run's artifact dir (uses --workflows/--fixtures, "
        "first entry each) instead of running anything",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if args.score_only:
        from tests.eval.conftest import FIXTURES_ROOT

        workflow_name = args.workflows.split(",")[0].strip()
        slug = args.fixtures.split(",")[0].strip()
        score, diag = _score_run(
            Path(args.score_only), FIXTURES_ROOT / slug, workflow_name
        )
        print(f"{workflow_name} on {slug}: {score.explain()}")
        if diag is not None:
            print(f"nomination stage: {diag.explain()}")
        for label, items in (("matched", score.matched), ("missed", score.missed),
                             ("extra", score.extra)):
            for it in items:
                print(f"  {label}: {it}")
        return 0

    model_alias = _resolve_model()
    workflows = [w.strip() for w in args.workflows.split(",") if w.strip()]
    fixtures = [f.strip() for f in args.fixtures.split(",") if f.strip()]
    print(f"model={model_alias} workflows={workflows} fixtures={fixtures} runs={args.runs}")

    for workflow_name in workflows:
        asyncio.run(
            _eval_unit(
                workflow_name=workflow_name,
                fixture_slugs=fixtures,
                model_alias=model_alias,
                runs=args.runs,
                timeout_s=args.timeout,
                min_precision=args.min_precision,
                min_recall=args.min_recall,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
