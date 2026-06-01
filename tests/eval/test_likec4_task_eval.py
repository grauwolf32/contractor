"""End-to-end eval for the ``likec4_build`` task.

Runs the likec4_build task (optionally preceded by dependency_information
+ project_information when precomputed artifacts are not available) and
scores the output.

Scoring is two-pronged:

1. **DSL validity.** We hand the published ``likec4_build/result`` text
   to ``Likec4Linter().validate(...)`` and fail when the number of
   reported errors exceeds ``max_validation_errors`` (default 0). The
   linter shells out to the ``likec4`` CLI; if it's not installed the
   test skips so unrelated CI doesn't go red.
2. **Structural keyword coverage.** Every well-formed LikeC4 source
   must contain at minimum ``specification``, ``model``, ``views``
   blocks -- we match each as a substring on the result body. Per-case
   overrides come from ``task-cases.json``.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from functools import partial

import pytest
from google.adk.models.lite_llm import LiteLlm

from contractor.agents.likec4_builder_agent.agent import \
    build_likec4_builder_agent
from contractor.agents.swe_agent.agent import build_swe_agent
from contractor.tools.fs import MemoryOverlayFileSystem
from contractor.tools.likec4 import (DEFAULT_LIKEC4_PATH, Likec4Error,
                                     Likec4Linter, Likec4NotFoundError)
from contractor.utils.prompt import load_prompt_with_version
from tests.eval.conftest import FIXTURES_ROOT
from tests.eval.results import CaseResult, metrics_from_task
from tests.eval.scorers import diff_detail, score_likec4_build
from tests.eval.task_harness import render_metrics_table, run_task_pipeline

_LIKEC4_FENCE_RE = re.compile(
    r"```(?:likec4|c4)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)


def _extract_likec4_source(result_text: str) -> str:
    matches = _LIKEC4_FENCE_RE.findall(result_text)
    if matches:
        return matches[-1].strip()
    return result_text


def _case_for(fixture, task: str) -> dict | None:
    for case in fixture.task_cases:
        if case.get("task") == task:
            return case
    return None


def _load_precomputed(slug: str) -> dict[str, str] | None:
    """Load precomputed dep/project artifacts from disk, if available."""
    art_dir = FIXTURES_ROOT / slug / "artifacts"
    mapping: dict[str, str] = {}
    for task_key in ("dependency_information", "project_information"):
        path = art_dir / f"{task_key}_result.txt"
        if not path.is_file():
            return None
        mapping[f"{task_key}/result"] = path.read_text(encoding="utf-8")
    return mapping


@pytest.mark.eval
@pytest.mark.asyncio
async def test_likec4_task(fixture, eval_model: LiteLlm, eval_sink):
    case = _case_for(fixture, "likec4_build")
    if case is None:
        pytest.skip(f"no likec4_build case for fixture {fixture.slug}")

    if shutil.which("likec4") is None and not any(
        shutil.which(r) for r in ("bunx", "pnpx", "npx")
    ):
        pytest.skip("likec4 CLI / JS runner not available")

    from cli.fs import RootedLocalFileSystem

    fs = RootedLocalFileSystem(str(fixture.source_root))
    overlay_fs = MemoryOverlayFileSystem(fs=fs)

    precomputed = _load_precomputed(fixture.slug)

    _, prompt_version = load_prompt_with_version("likec4_builder_agent")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = (
        FIXTURES_ROOT / fixture.slug / "runs" / f"{prompt_version}_{ts}"
    )

    def queue(runner) -> None:
        likec4_builder = partial(
            build_likec4_builder_agent,
            name="likec4_builder",
            fs=overlay_fs,
            model=eval_model,
            max_tokens=120_000,
        )

        runner.add_variable(name="project_path", value=str(fixture.source_root))

        if precomputed is None:
            swe_builder = partial(
                build_swe_agent,
                name="swe_agent",
                fs=fs,
                model=eval_model,
                max_tokens=100_000,
            )
            runner.add_task(
                name="dependency_information",
                worker_builder=swe_builder,
                iterations=1, max_attempts=2, max_steps=20,
                namespace="dependency_information", model=eval_model,
            )
            runner.add_task(
                name="project_information",
                worker_builder=swe_builder,
                iterations=1, max_attempts=2, max_steps=20,
                artifacts=["dependency_information/result"],
                namespace="project_information", model=eval_model,
            )

        runner.add_task(
            name="likec4_build",
            worker_builder=likec4_builder,
            iterations=3, max_attempts=6, max_steps=20,
            artifacts=[
                "dependency_information/result",
                "project_information/result",
            ],
            namespace="likec4-building", model=eval_model,
        )

    run = await run_task_pipeline(
        queue_fn=queue,
        artifact_keys=["likec4_build/result"],
        namespace=f"task-eval-{fixture.slug}-{case['id']}",
        timeout_s=float(case.get("timeout_s", 2400.0)),
        runner_name=f"likec4-{fixture.slug}",
        preloaded_artifacts=precomputed,
        output_dir=run_dir,
    )

    if overlay_fs.exists(DEFAULT_LIKEC4_PATH):
        dsl = overlay_fs.read_text(DEFAULT_LIKEC4_PATH, encoding="utf-8")
        source_origin = "overlay"
    else:
        result_text = run.result_text("likec4_build")
        assert result_text, (
            "no LikeC4 artifact produced\n"
            + render_metrics_table(run.metrics)
        )
        dsl = _extract_likec4_source(result_text)
        source_origin = "result-fence"

    (run_dir / "architecture.c4").write_text(dsl, encoding="utf-8")

    validation_errors: list[dict] = []
    if case.get("must_validate", True):
        try:
            validation_errors = Likec4Linter().validate(dsl)
        except Likec4NotFoundError:
            pytest.skip("likec4 binary disappeared between check and validate")
        except Likec4Error as exc:
            pytest.fail(
                f"likec4 validator failed to run: {exc}\n"
                f"dsl_chars={len(dsl)} source_origin={source_origin}\n"
                f"metrics:\n{render_metrics_table(run.metrics)}"
            )

    result = score_likec4_build(dsl, case, validation_errors)

    summary = (
        f"fixture={fixture.slug} case={case['id']}\n"
        f"  source_origin={source_origin}\n"
        f"{result.explain()}\n"
        f"  precomputed={'yes' if precomputed else 'no'}\n\n"
        f"metrics:\n{render_metrics_table(run.metrics)}"
    )
    print(f"\n{'='*60}\n{summary}\n{'='*60}")

    eval_sink.record(
        scenario="task", unit="likec4_build", metric_kind="diff",
        fixture=fixture.slug, model=str(eval_model.model),
        case=CaseResult(id=case["id"], passed=result.passed,
                        pass_count=int(result.passed), attempts=1,
                        metrics=metrics_from_task(run.metrics), detail=diff_detail(result)),
    )
    assert result.passed, f"likec4 eval failed\n{summary}"
