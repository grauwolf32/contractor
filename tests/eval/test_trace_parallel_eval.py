"""A/B eval: sequential vs parallel trace workflows.

Runs ``trace-graph``, ``trace-graph-pathpar``, and ``trace-graph-opspar``
on the same fixture and compares:

- **Wallclock time** (the whole point of parallelism)
- **Annotation quality** (precision/recall vs ground-truth trace cases)
- **Merge conflicts** (files touched by >1 parallel fork)

The fixture's ``oas.expected.yaml`` is used as the input OpenAPI spec so
the eval doesn't depend on the ``build`` workflow.

Usage::

    CONTRACTOR_RUN_EVAL=1 poetry run pytest tests/eval/test_trace_parallel_eval.py -k vulnyapi -s
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from google.adk.artifacts import FileArtifactService
from google.adk.models.lite_llm import LiteLlm

from cli.fs import RootedLocalFileSystem
from contractor.runners.models import TaskRunnerEvent
from contractor.utils import observability
from contractor.workflows import Workflow, WorkflowContext
from contractor.workflows.trace_graph import TraceGraphWorkflow
from contractor.workflows.trace_graph_pathpar import TraceGraphPathParWorkflow
from tests.eval.conftest import FIXTURES_ROOT, EvalFixture, _load_fixture
from tests.eval.trace_harness import (Annotation,
                                      extract_annotations_from_overlay,
                                      overlay_modified_files)

WORKFLOW_VARIANTS: list[tuple[str, type[Workflow]]] = [
    ("trace-graph", TraceGraphWorkflow),
    ("trace-graph-pathpar", TraceGraphPathParWorkflow),
]


@dataclass
class WorkflowRunResult:
    variant: str
    annotations: set[Annotation]
    modified_files: set[str]
    wallclock_s: float
    events: list[dict[str, Any]] = field(default_factory=list)


def _expected_annotations(fixture: EvalFixture) -> set[Annotation]:
    """Union of all expected annotations across trace cases."""
    annotations: set[Annotation] = set()
    for case in fixture.trace_cases:
        for entry in case.get("expected_annotated", []):
            annotations.add(Annotation(file=entry["file"], function=entry["function"]))
    return annotations


async def _run_workflow_variant(
    workflow_cls: type[Workflow],
    *,
    fixture: EvalFixture,
    model: LiteLlm,
    oas_yaml: str,
) -> WorkflowRunResult:
    """Run a single workflow variant end-to-end and capture results."""
    tmp = tempfile.mkdtemp(prefix="eval_parallel_")
    artifact_service = FileArtifactService(root_dir=Path(tmp))
    fs = RootedLocalFileSystem(str(fixture.source_root))
    app_name = "eval-parallel"
    user_id = "eval-user"

    ctx = WorkflowContext(
        project_path=fixture.source_root,
        folder_name="/",
        model=model.model,
        app_name=app_name,
        user_id=user_id,
        artifact_service=artifact_service,
        fs=fs,
        artifact=oas_yaml,
    )

    workflow = workflow_cls(ctx)

    events: list[dict[str, Any]] = []

    async def _on_event(event: TaskRunnerEvent) -> None:
        events.append({"type": event.type, "task": event.task_name, **event.payload})

    t0 = time.monotonic()
    await workflow.run(user_id=user_id, on_event=_on_event)
    wallclock = time.monotonic() - t0

    overlay = workflow.overlayfs
    annotations = extract_annotations_from_overlay(overlay)
    modified = overlay_modified_files(overlay)

    return WorkflowRunResult(
        variant=workflow_cls.__name__,
        annotations=annotations,
        modified_files=modified,
        wallclock_s=wallclock,
        events=events,
    )


def _score(actual: set[Annotation], expected: set[Annotation]) -> dict[str, Any]:
    if not expected:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "matched": 0, "total_expected": 0}

    actual_tuples = {a.as_tuple() for a in actual}
    expected_tuples = {e.as_tuple() for e in expected}

    matched = actual_tuples & expected_tuples
    precision = len(matched) / len(actual_tuples) if actual_tuples else 0.0
    recall = len(matched) / len(expected_tuples) if expected_tuples else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "matched": len(matched),
        "total_expected": len(expected_tuples),
        "total_actual": len(actual_tuples),
        "missing": sorted(expected_tuples - actual_tuples),
        "extra": sorted(actual_tuples - expected_tuples),
    }


def _discover_parallel_slugs() -> list[str]:
    """Fixtures that have both trace-cases.json AND oas.expected.yaml."""
    return [
        slug
        for slug in sorted(d.name for d in FIXTURES_ROOT.iterdir() if d.is_dir())
        if (FIXTURES_ROOT / slug / "trace-cases.json").is_file()
        and (FIXTURES_ROOT / slug / "oas.expected.yaml").is_file()
    ]


@pytest.fixture(
    scope="session",
    params=_discover_parallel_slugs(),
    ids=lambda s: s,
)
def parallel_fixture(request: pytest.FixtureRequest) -> EvalFixture:
    return _load_fixture(request.param)


@pytest.mark.eval
@pytest.mark.asyncio
async def test_trace_parallel_workflows(parallel_fixture: EvalFixture, eval_model: LiteLlm):
    """Compare trace-graph, trace-graph-pathpar, and trace-graph-opspar."""
    fixture = parallel_fixture

    oas_path = FIXTURES_ROOT / fixture.slug / "oas.expected.yaml"
    oas_yaml = oas_path.read_text(encoding="utf-8")

    expected = _expected_annotations(fixture)
    results: list[WorkflowRunResult] = []

    for variant_name, workflow_cls in WORKFLOW_VARIANTS:
        with observability.run_context(
            name=f"eval.parallel.{variant_name}",
            session_id=f"parallel-eval-{fixture.slug}-{variant_name}",
            tags=["eval", "parallel-workflow", f"variant:{variant_name}"],
            metadata={"fixture": fixture.slug, "variant": variant_name},
        ):
            result = await _run_workflow_variant(
                workflow_cls,
                fixture=fixture,
                model=eval_model,
                oas_yaml=oas_yaml,
            )
        result.variant = variant_name
        results.append(result)

    # ── Report ────────────────────────────────────────────────────────
    header = f"\n{'='*70}\nParallel workflow comparison — fixture={fixture.slug}\n{'='*70}"
    lines = [header]

    baseline = results[0]
    for r in results:
        score = _score(r.annotations, expected)
        speedup = baseline.wallclock_s / r.wallclock_s if r.wallclock_s > 0 else 0
        lines.append(
            f"\n  {r.variant}:\n"
            f"    wallclock:   {r.wallclock_s:7.1f}s"
            f"  (speedup: {speedup:.2f}x vs baseline)\n"
            f"    annotations: {len(r.annotations):3d} actual"
            f"  / {score['total_expected']} expected\n"
            f"    precision:   {score['precision']:.3f}\n"
            f"    recall:      {score['recall']:.3f}\n"
            f"    f1:          {score['f1']:.3f}\n"
            f"    files:       {len(r.modified_files)}"
        )

    report = "\n".join(lines)
    print(report)

    # Save report to fixture runs dir.
    runs_dir = FIXTURES_ROOT / fixture.slug / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    report_path = runs_dir / "parallel_comparison.txt"
    report_path.write_text(report, encoding="utf-8")

    # Save structured results for downstream analysis.
    structured = {
        "fixture": fixture.slug,
        "expected_count": len(expected),
        "variants": [
            {
                "name": r.variant,
                "wallclock_s": round(r.wallclock_s, 2),
                "annotation_count": len(r.annotations),
                "modified_files": len(r.modified_files),
                **_score(r.annotations, expected),
            }
            for r in results
        ],
    }
    (runs_dir / "parallel_comparison.json").write_text(
        json.dumps(structured, indent=2, default=str),
        encoding="utf-8",
    )
