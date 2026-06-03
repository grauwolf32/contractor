#!/usr/bin/env python3
"""Run the ``trace_annotation`` **task** eval (planner + worker via TaskRunner).

This is the task-scenario sibling of ``scripts/run_vuln_eval.py`` (which drives
the bare trace_agent). Here we run the real ``TraceAnnotationWorkflow``: for each
OpenAPI path the planner spawns subtasks and the worker annotates the traced
flow. We score the produced ``(file, function)`` annotations against each
fixture's ``expected_annotated`` ground truth and emit the canonical ``eval/v1``
envelope (``scenario=task``, ``unit=trace_annotation``, ``metric_kind=diff``).

The task only runs on fixtures that ship an ``oas.expected.yaml`` (the spec
drives the per-path loop) AND a ``trace-cases.json`` (ground-truth annotations).

Usage::

    poetry run python scripts/run_trace_task_eval.py
    poetry run python scripts/run_trace_task_eval.py --prompt shannon
    poetry run python scripts/run_trace_task_eval.py --fixtures vulnyapi --timeout 1800 \
        --output eval_runs/trace-task-shannon
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_FIXTURES = "vulnyapi,vaultpay,fastapi,spring"


def _score(actual: set, expected: set) -> dict[str, Any]:
    """Precision/recall/f1 of annotated ``(file, function)`` tuples."""
    if not expected:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "matched": 0,
                "total_expected": 0, "total_actual": len(actual),
                "missing": [], "extra": []}
    actual_t = {a.as_tuple() for a in actual}
    expected_t = {e.as_tuple() for e in expected}
    matched = actual_t & expected_t
    precision = len(matched) / len(actual_t) if actual_t else 0.0
    recall = len(matched) / len(expected_t) if expected_t else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "matched": len(matched),
        "total_expected": len(expected_t),
        "total_actual": len(actual_t),
        "missing": sorted(f"{f}::{fn}" for f, fn in (expected_t - actual_t)),
        "extra": sorted(f"{f}::{fn}" for f, fn in (actual_t - expected_t)),
    }


async def run_eval(
    fixture_slugs: list[str],
    output_dir: Path,
    timeout_s: float,
    prompt_version: str | None,
    max_attempts: int,
    iterations: int,
    per_path_timeout_s: float,
) -> list[dict[str, Any]]:
    from dotenv import load_dotenv
    for p in (REPO_ROOT / "cli" / ".env", REPO_ROOT / ".env"):
        if p.exists():
            load_dotenv(p, override=False)

    import os

    from contractor.utils import observability
    observability.init()

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)

    override = os.environ.get("CONTRACTOR_EVAL_MODEL")
    if override:
        model_alias = override
    else:
        from contractor.utils.settings import DEFAULT_MODEL
        model_alias = DEFAULT_MODEL.model

    import yaml
    from google.adk.artifacts import FileArtifactService

    from cli.fs import RootedLocalFileSystem
    from contractor.agents.trace_agent.agent import build_trace_agent
    from contractor.runners.models import TaskRunnerEvent
    from contractor.runners.task_runner import TaskRunner
    from contractor.utils import load_prompt_with_version
    from contractor.workflows import WorkflowContext
    from contractor.workflows.trace_annotation.workflow import (
        TraceAnnotationWorkflow,
        extract_openapi_paths,
    )
    from tests.eval.conftest import _load_fixture
    from tests.eval.results import (
        CaseResult,
        EvalRun,
        FixtureResult,
        metrics_from_task,
        write_eval_results,
    )
    from tests.eval.task_harness import _aggregate_metrics
    from tests.eval.trace_harness import (
        extract_annotations_from_overlay,
    )

    prompt_text, resolved_version = load_prompt_with_version("trace_agent", prompt_version)

    class _EvalTraceAnnotationWorkflow(TraceAnnotationWorkflow):
        """TraceAnnotationWorkflow with eval-tunable prompt + task budgets.

        Overrides ``_run_path_analysis`` solely to (a) pin the worker prompt to a
        chosen version and (b) override the per-task ``max_attempts``/``iterations``
        budget; everything else (overlay, per-path loop, artifacts) is inherited.
        """

        def __init__(self, ctx: WorkflowContext) -> None:
            super().__init__(ctx)
            self._events: list[TaskRunnerEvent] = []

        async def _run_path_analysis(
            self, api_path, *, user_id="cli-user", on_event=None,
        ) -> None:
            if not api_path.operations:
                return
            ctx = self.ctx
            trace_builder = partial(
                build_trace_agent,
                name="trace_agent",
                fs=self.overlayfs,
                model=self.llm,
                max_tokens=80_000,
                enable_vuln_reporting=True,
                with_graph_tools=True,
                prompt=prompt_text,
            )
            runner = TaskRunner(
                name="contractor",
                artifact_service=ctx.artifact_service,
                checkpoint_path=ctx.checkpoint_path,
            )
            runner.add_variable(name="project_path", value=ctx.folder_name)
            operation_ids, operation_schema_yaml = self._build_path_task_payload(api_path)
            runner.add_variable(name="operation_id", value=operation_ids)
            runner.add_variable(name="operation_schema", value=operation_schema_yaml)
            runner.add_task(
                name="trace_annotation",
                ref=f"trace_annotation:{self.namespace}:{api_path.path_key}",
                worker_builder=trace_builder,
                iterations=iterations,
                max_attempts=max_attempts,
                max_steps=20,
                artifacts=[],
                skills=["trace"],
                namespace=f"trace-annotation:{self.namespace}",
                model=self.llm,
            )
            # Bound each path independently: a stuck path is cancelled and we move
            # on, keeping annotations already written to the shared overlay.
            try:
                await asyncio.wait_for(
                    runner.run(user_id=user_id, on_event=on_event),
                    timeout=per_path_timeout_s,
                )
            except TimeoutError:
                print(f"    [path {api_path.path_key}] timed out "
                      f"(>{per_path_timeout_s:.0f}s) — skipping to next path")

    results: list[dict[str, Any]] = []

    for slug in fixture_slugs:
        fixture = _load_fixture(slug)
        oas_path = REPO_ROOT / "tests" / "eval" / "fixtures" / slug / "oas.expected.yaml"
        if not oas_path.is_file():
            print(f"  [{slug}] skipped — no oas.expected.yaml")
            continue

        expected = set()
        from tests.eval.trace_harness import Annotation
        for case in fixture.trace_cases:
            for entry in case.get("expected_annotated", []):
                expected.add(Annotation(file=entry["file"], function=entry["function"]))

        oas_yaml = oas_path.read_text(encoding="utf-8")
        openapi = yaml.safe_load(oas_yaml)
        n_paths = len(extract_openapi_paths(json.loads(json.dumps(openapi))))

        print(f"\n{'='*60}")
        print(f"  {slug}: {n_paths} paths, {len(expected)} expected annotations "
              f"(prompt=trace_agent@{resolved_version}, "
              f"max_attempts={max_attempts}, iterations={iterations})")
        print(f"{'='*60}")

        case_dir = output_dir / slug
        case_dir.mkdir(parents=True, exist_ok=True)
        artifact_service = FileArtifactService(root_dir=str(case_dir / "artifacts"))
        fs = RootedLocalFileSystem(str(fixture.source_root))
        ctx = WorkflowContext(
            project_path=fixture.source_root,
            folder_name="/",
            model=model_alias,
            app_name="contractor",
            user_id="eval-user",
            artifact_service=artifact_service,
            fs=fs,
            artifact=oas_yaml,
            timeout=600,
        )

        events: list[TaskRunnerEvent] = []

        async def _on_event(event: TaskRunnerEvent, _sink=events) -> None:
            _sink.append(event)

        wf = _EvalTraceAnnotationWorkflow(ctx)
        t0 = time.monotonic()
        timed_out = False
        try:
            with observability.run_context(
                name="eval.trace_annotation_task",
                session_id=f"trace-task-{slug}",
                tags=["eval", "agent:task_runner", "task:trace_annotation",
                      f"prompt:trace_agent@{resolved_version}"],
                metadata={"fixture": slug, "prompt_version": resolved_version},
            ):
                await asyncio.wait_for(
                    wf.run(user_id="eval-user", on_event=_on_event), timeout=timeout_s,
                )
        except TimeoutError:
            # Outer budget exhausted — score whatever paths completed rather
            # than discarding the run. Partial annotations live in the overlay.
            timed_out = True
            print(f"  [{slug}] outer timeout (>{timeout_s:.0f}s) — scoring partial overlay")
        except Exception as exc:
            print(f"  [{slug}] ERROR: {exc!r}")
            continue
        duration = time.monotonic() - t0

        annotations = extract_annotations_from_overlay(wf.overlayfs)
        score = _score(annotations, expected)
        metrics = metrics_from_task(_aggregate_metrics(events))

        fr = {
            "slug": slug,
            "prompt_version": resolved_version,
            "model": model_alias,
            "duration_s": round(duration, 1),
            "expected_count": len(expected),
            "annotation_count": len(annotations),
            "annotations": sorted(f"{a.file}::{a.function}" for a in annotations),
            "timed_out": timed_out,
            **score,
            **metrics,
        }
        results.append(fr)
        (case_dir / "metrics.json").write_text(
            json.dumps(fr, indent=2, default=str), encoding="utf-8")

        print(f"  duration: {duration:.1f}s  annotations: {len(annotations)}"
              f"  / {len(expected)} expected")
        print(f"  P={score['precision']:.3f} R={score['recall']:.3f} F1={score['f1']:.3f}")
        print(f"  tools: {metrics.get('total_tool_calls', 0)}  "
              f"tokens: {metrics.get('total_tokens', 0)}  "
              f"llm_calls: {metrics.get('llm_calls', 0)}  "
              f"errors: {metrics.get('tool_errors', 0)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()

    fixtures = []
    for r in results:
        case = CaseResult(
            id=r["slug"], passed=r["f1"] >= 0.5, pass_count=int(r["f1"] >= 0.5),
            attempts=1, metrics=r,
            detail={"precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
                    "matched": r["matched"], "total_expected": r["total_expected"],
                    "total_actual": r["total_actual"], "missing": r["missing"],
                    "extra": r["extra"], "prompt_version": r["prompt_version"]})
        fixtures.append(FixtureResult(slug=r["slug"], cases=[case]))

    eval_run = EvalRun(
        scenario="task", unit="trace_annotation", pass_at=1, metric_kind="diff",
        model=model_alias,
        prompt_version=(results[0]["prompt_version"] if results else prompt_version),
        timestamp=timestamp, fixtures=fixtures,
        meta={"max_attempts": max_attempts, "iterations": iterations},
    )
    results_path = write_eval_results(eval_run, output_dir)
    print(f"\nResults saved to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run the trace_annotation task eval (TaskRunner planner+worker)")
    parser.add_argument("--fixtures", default=DEFAULT_FIXTURES,
                        help=f"Comma-separated fixture slugs (default: {DEFAULT_FIXTURES})")
    parser.add_argument("--output", default=str(REPO_ROOT / "eval_runs" / "trace_task_eval"))
    parser.add_argument("--timeout", type=float, default=7200.0,
                        help="outer per-fixture wall-clock budget (all paths)")
    parser.add_argument("--per-path-timeout", type=float, default=420.0,
                        help="per-path budget; a stuck path is cancelled and skipped")
    parser.add_argument("--prompt", default=None,
                        help="trace_agent prompt version (e.g. v7, shannon); default = active")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()

    slugs = [s.strip() for s in args.fixtures.split(",") if s.strip()]
    asyncio.run(run_eval(
        slugs, Path(args.output), args.timeout, args.prompt,
        args.max_attempts, args.iterations, args.per_path_timeout,
    ))


if __name__ == "__main__":
    main()
