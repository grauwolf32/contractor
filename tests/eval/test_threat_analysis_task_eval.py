"""End-to-end eval for the ``threat_analysis`` (STRIDE) task.

The threat-model task consumes four upstream artifacts — likec4, openapi,
project_information, dependency_information — and persists each credible threat
as a vulnerability report. This eval runs the task in isolation by *preloading*
those upstream artifacts (so it measures the threat reasoning, not the quality
of the precompute chain) and scores the reports it produces.

Inputs per fixture:
  - dependency_information / project_information: precomputed text under
    ``<fixture>/artifacts/*_result.txt`` (required → skip if absent).
  - openapi: the fixture's ground-truth ``oas.expected.yaml`` (the ideal spec,
    so a weak threat model can't be blamed on a weak OAS) → skip if absent.
  - likec4: precomputed ``<fixture>/artifacts/likec4_build_result.txt`` when
    present; otherwise omitted (the task tolerates a missing artifact).

Scoring is structural (see ``score_threat_analysis``): report count, STRIDE
breadth, the mandated report shape, valid severity/confidence enums, plus a
soft check that the threats reference the known-vulnerable OpenAPI paths from
``vulnerabilities.expected.json`` when that ground truth exists.

The OAS is fed through the task inbox (not the OpenAPI tool store), so the
agent reads it as context; native ``openapi`` enumeration tools are therefore
disabled for this eval. A future revision could preload the OpenAPI store to
exercise those tools natively.
"""

from __future__ import annotations

from functools import partial

import pytest
import yaml
from google.adk.models.lite_llm import LiteLlm

from contractor.agents.threat_model_agent.agent import build_threat_model_agent
from tests.eval.conftest import FIXTURES_ROOT
from tests.eval.results import CaseResult, case_artifact_dir, metrics_from_task
from tests.eval.scorers import diff_detail, score_threat_analysis
from tests.eval.task_harness import render_metrics_table, run_task_pipeline

NAMESPACE = "threat-model"
VULN_ARTIFACT_KEY = f"user:vulnerability-reports/{NAMESPACE}"


def _case_for(fixture, task: str) -> dict | None:
    for case in fixture.task_cases:
        if case.get("task") == task:
            return case
    return None


def _read_artifact(slug: str, task_key: str) -> str | None:
    path = FIXTURES_ROOT / slug / "artifacts" / f"{task_key}_result.txt"
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _parse_reports(text: str) -> list[dict]:
    """The vuln store persists a ``{name: report}`` YAML map; return the rows."""
    raw = yaml.safe_load(text or "") or {}
    if not isinstance(raw, dict):
        return []
    out: list[dict] = []
    for name, body in raw.items():
        if isinstance(body, dict):
            out.append({"name": name, **body})
    return out


@pytest.mark.eval
@pytest.mark.asyncio
async def test_threat_analysis_task(fixture, eval_model: LiteLlm, eval_sink):
    case = _case_for(fixture, "threat_analysis")
    if case is None:
        pytest.skip(f"no threat_analysis case for fixture {fixture.slug}")

    dep = _read_artifact(fixture.slug, "dependency_information")
    proj = _read_artifact(fixture.slug, "project_information")
    if dep is None or proj is None:
        pytest.skip(
            f"{fixture.slug}: precomputed dependency/project artifacts required"
        )

    # openapi + likec4 are optional inputs (the task tolerates missing
    # artifacts). When oas.expected.yaml exists we feed it as the ideal spec.
    expected_oas = fixture.expected_oas
    openapi_text = (
        yaml.safe_dump(expected_oas, sort_keys=False, allow_unicode=True)
        if expected_oas
        else None
    )
    likec4 = _read_artifact(fixture.slug, "likec4_build")  # optional

    from cli.fs import RootedLocalFileSystem

    fs = RootedLocalFileSystem(str(fixture.source_root))

    preloaded = {
        "dependency_information/result": dep,
        "project_information/result": proj,
    }
    artifacts = [
        "dependency_information/result",
        "project_information/result",
    ]
    if openapi_text:
        preloaded["oas_update/result"] = openapi_text
        artifacts.append("oas_update/result")
    if likec4:
        preloaded["likec4_build/result"] = likec4
        artifacts.insert(0, "likec4_build/result")

    def queue(runner) -> None:
        threat_builder = partial(
            build_threat_model_agent,
            name="threat_model",
            fs=fs,
            model=eval_model,
            max_tokens=120_000,
            with_openapi=False,  # OAS arrives via the inbox, not the OAS store
            with_graph_tools=True,
        )
        runner.add_variable(name="project_path", value=str(fixture.source_root))
        # The threat template substitutes these refs; the content itself is
        # injected via the artifacts= inbox below.
        runner.add_variable(
            name="likec4_artifact",
            value="likec4_build/result" if likec4 else "(not available)",
        )
        runner.add_variable(
            name="openapi_artifact",
            value="oas_update/result" if openapi_text else "(not available)",
        )
        runner.add_variable(
            name="project_information", value="project_information/result"
        )
        runner.add_variable(
            name="dependency_information", value="dependency_information/result"
        )
        runner.add_task(
            name="threat_analysis",
            worker_builder=threat_builder,
            iterations=1,
            max_attempts=2,
            max_steps=40,
            artifacts=artifacts,
            namespace=NAMESPACE,
            model=eval_model,
        )

    run = await run_task_pipeline(
        queue_fn=queue,
        artifact_keys=["threat_analysis/result", VULN_ARTIFACT_KEY],
        namespace=f"task-eval-{fixture.slug}-{case['id']}",
        timeout_s=float(case.get("timeout_s", 2400.0)),
        runner_name=f"threat_analysis-{fixture.slug}",
        preloaded_artifacts=preloaded,
        artifact_dir=case_artifact_dir("threat_analysis", fixture.slug, case["id"]),
    )

    reports_text = run.artifacts.get(VULN_ARTIFACT_KEY, "")
    reports = _parse_reports(reports_text)
    assert reports, (
        "threat_analysis persisted no vulnerability reports\n"
        f"(looked under {VULN_ARTIFACT_KEY})\n"
        + render_metrics_table(run.metrics)
    )

    result = score_threat_analysis(reports, case, fixture.expected_vulnerabilities)

    summary = (
        f"fixture={fixture.slug} case={case['id']} "
        f"likec4={'yes' if likec4 else 'no'}\n"
        f"{result.explain()}\n\n"
        f"metrics:\n{render_metrics_table(run.metrics)}"
    )
    print(f"\n{'=' * 60}\n{summary}\n{'=' * 60}")

    eval_sink.record(
        scenario="task",
        unit="threat_analysis",
        metric_kind="diff",
        fixture=fixture.slug,
        model=str(eval_model.model),
        case=CaseResult(
            id=case["id"],
            passed=result.passed,
            pass_count=int(result.passed),
            attempts=1,
            metrics=metrics_from_task(run.metrics),
            detail=diff_detail(result),
        ),
        artifacts=run.artifacts,
    )
    assert result.passed, f"threat_analysis eval failed\n{summary}"
