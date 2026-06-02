"""Per-eval-type scorer functions.

Each function takes a run result (or extracted output) plus a case dict
and returns an ``EvalResult`` with named checks.  Tests assert
``result.passed``; scripts call the same functions for programmatic access
to precision/recall/etc.
"""

from __future__ import annotations

from typing import Any

from tests.eval.adk_evals import score_tool_trajectory
from tests.eval.scoring import (
    EvalCheck,
    EvalResult,
    _score_sets,
    score_components,
    score_endpoints,
    score_markdown_sections,
    score_phrases,
    score_vulnerabilities,
)


def diff_detail(result: EvalResult) -> dict[str, Any]:
    """Extract a ``diff`` case detail (precision/recall/f1) from an EvalResult.

    Scans ``result.meta`` for the first ``Score``-like object (one carrying an
    ``f1`` attribute) and returns its rounded metrics. Used to populate the
    ``eval/v1`` case detail for set-diff scorers (OAS / LikeC4 / project-info /
    trace). Returns ``{}`` when no Score is present (→ headline mean_f1 unaffected).
    """
    for value in result.meta.values():
        if hasattr(value, "f1"):
            return {"precision": round(value.precision, 3),
                    "recall": round(value.recall, 3),
                    "f1": round(value.f1, 3)}
    return {}

# ---------------------------------------------------------------------------
# Trace agent
# ---------------------------------------------------------------------------


def score_trace_run(run, case: dict) -> EvalResult:
    """Score a ``TraceAgentRun`` against a trace case.

    Checks annotation precision/recall and optional tool trajectory.
    """
    from tests.eval.trace_harness import Annotation

    expected = {
        Annotation(file=item["file"], function=item["function"])
        for item in case["expected_annotated"]
    }
    actual_keys = {a.as_tuple() for a in run.annotations}
    expected_keys = {a.as_tuple() for a in expected}
    annotation_score = _score_sets(actual_keys, expected_keys)

    min_p = float(case.get("min_precision", 0.5))
    min_r = float(case.get("min_recall", 0.5))

    checks = [
        EvalCheck(
            name="annotations",
            passed=annotation_score.passes(min_precision=min_p, min_recall=min_r),
            details=annotation_score.explain("annotations"),
        ),
    ]

    expected_trajectory = case.get("expected_tool_trajectory")
    if expected_trajectory:
        ordered = bool(case.get("trajectory_ordered", True))
        traj = score_tool_trajectory(
            run.agent_run, expected_trajectory, ordered=ordered,
        )
        checks.append(EvalCheck(
            name="trajectory",
            passed=traj.matched,
            details=traj.explain(),
        ))

    return EvalResult(
        checks=checks,
        meta={
            "prompt_version": run.prompt_version,
            "modified_files": sorted(run.modified_files),
            "tools_used": sorted(set(run.agent_run.tool_names())),
            "annotation_score": annotation_score,
        },
    )


# ---------------------------------------------------------------------------
# SWE agent
# ---------------------------------------------------------------------------


def score_swe_run(run, case: dict) -> EvalResult:
    """Score an ``AgentRun`` against a SWE case.

    Checks phrase recall, required/any-of tool presence, and optional
    tool trajectory.
    """
    names = set(run.tool_names())
    required_all = set(case.get("expected_tools_all") or [])
    required_any = set(case.get("expected_tools_any") or [])

    missing_all = required_all - names
    any_satisfied = (not required_any) or bool(required_any & names)

    phrase_score = score_phrases(run.final_text, case.get("expected_phrases", []))
    min_phrase_recall = float(case.get("min_phrase_recall", 0.66))

    checks = [
        EvalCheck(
            name="phrases",
            passed=phrase_score.recall >= min_phrase_recall,
            details=phrase_score.explain("phrases"),
        ),
        EvalCheck(
            name="required_tools",
            passed=not missing_all,
            details=f"missing={sorted(missing_all)}" if missing_all else "all present",
        ),
        EvalCheck(
            name="any_of_tools",
            passed=any_satisfied,
            details=(
                f"needed any of {sorted(required_any)}, "
                f"had={sorted(required_any & names)}"
                if required_any
                else "no requirement"
            ),
        ),
    ]

    expected_trajectory = case.get("expected_tool_trajectory")
    if expected_trajectory:
        ordered = bool(case.get("trajectory_ordered", True))
        traj = score_tool_trajectory(run, expected_trajectory, ordered=ordered)
        checks.append(EvalCheck(
            name="trajectory",
            passed=traj.matched,
            details=traj.explain(),
        ))

    return EvalResult(
        checks=checks,
        meta={
            "tools_used": sorted(names),
            "phrase_score": phrase_score,
            "final_text_preview": run.final_text[:300],
        },
    )


# ---------------------------------------------------------------------------
# OAS schema (builder / build-task / enrich-task)
# ---------------------------------------------------------------------------


def score_oas_schema(
    actual_schema: dict[str, Any],
    expected_schema: dict[str, Any],
    *,
    min_endpoint_precision: float = 0.7,
    min_endpoint_recall: float = 0.8,
    min_schema_recall: float = 0.5,
) -> EvalResult:
    """Score an OpenAPI schema against ground truth.

    Reused by ``test_oas_builder``, ``test_oas_build_task``, and
    ``test_oas_enrich_task``.
    """
    endpoint_score = score_endpoints(actual_schema, expected_schema)
    schemas_score = score_components(actual_schema, expected_schema, "schemas")

    return EvalResult(
        checks=[
            EvalCheck(
                name="endpoints",
                passed=endpoint_score.passes(
                    min_precision=min_endpoint_precision,
                    min_recall=min_endpoint_recall,
                ),
                details=endpoint_score.explain("endpoints"),
            ),
            EvalCheck(
                name="schemas",
                passed=schemas_score.recall >= min_schema_recall,
                details=schemas_score.explain("schemas"),
            ),
        ],
        meta={
            "endpoint_score": endpoint_score,
            "schemas_score": schemas_score,
        },
    )


# ---------------------------------------------------------------------------
# OAS analyzer
# ---------------------------------------------------------------------------


def score_oas_analysis(
    vulnerabilities: list[dict[str, Any]],
    expected_vulnerabilities: list[dict[str, Any]],
    *,
    min_precision: float = 0.4,
    min_recall: float = 0.5,
) -> EvalResult:
    """Score OAS analyzer vulnerability findings."""
    score = score_vulnerabilities(vulnerabilities, expected_vulnerabilities)

    return EvalResult(
        checks=[
            EvalCheck(
                name="vulnerabilities",
                passed=score.passes(
                    min_precision=min_precision, min_recall=min_recall,
                ),
                details=score.explain("vulnerabilities"),
            ),
        ],
        meta={
            "vulnerability_score": score,
            "reported_count": len(vulnerabilities),
        },
    )


# ---------------------------------------------------------------------------
# Project information task
# ---------------------------------------------------------------------------


def score_project_info(result_text: str, case: dict) -> EvalResult:
    """Score a ``project_information`` task result."""
    section_score = score_markdown_sections(
        result_text, case["expected_sections"],
    )
    min_section_recall = float(case.get("min_section_recall", 0.7))

    expected_phrases_list = case.get("expected_phrases_any", [])
    phrase_score = score_phrases(result_text, expected_phrases_list)
    min_phrase_recall = float(case.get("min_phrase_recall", 0.5))

    return EvalResult(
        checks=[
            EvalCheck(
                name="sections",
                passed=section_score.recall >= min_section_recall,
                details=section_score.explain("sections"),
            ),
            EvalCheck(
                name="phrases",
                passed=phrase_score.recall >= min_phrase_recall,
                details=phrase_score.explain("phrases"),
            ),
        ],
        meta={
            "result_chars": len(result_text),
            "section_score": section_score,
            "phrase_score": phrase_score,
        },
    )


# ---------------------------------------------------------------------------
# LikeC4 build task
# ---------------------------------------------------------------------------


def score_exploitability_run(run, case: dict) -> EvalResult:
    """Score an ``ExploitabilityRun`` against an exploitability case.

    Checks verdict presence, verdict correctness (with tolerance for
    ``exploitable_unverified`` when ``exploitable`` is expected), and
    evidence quality.
    """
    expected_verdict = case["expected_verdict"]
    finding_name = case["vulnerability_name"]

    verdict_obj = run.verdicts.get(finding_name)

    checks: list[EvalCheck] = []

    checks.append(EvalCheck(
        name="verdict_present",
        passed=verdict_obj is not None,
        details=f"verdict={'present' if verdict_obj else 'missing'} for {finding_name}",
    ))

    if verdict_obj is not None:
        actual = verdict_obj.get("verdict", "")

        if expected_verdict == "exploitable":
            verdict_match = actual in ("exploitable", "exploitable_unverified")
        else:
            verdict_match = actual == expected_verdict

        checks.append(EvalCheck(
            name="verdict_correct",
            passed=verdict_match,
            details=f"expected={expected_verdict} actual={actual}",
        ))

        has_evidence = bool(verdict_obj.get("entry_point")) and bool(verdict_obj.get("summary"))
        checks.append(EvalCheck(
            name="evidence_present",
            passed=has_evidence,
            details=(
                f"entry_point={bool(verdict_obj.get('entry_point'))} "
                f"summary={bool(verdict_obj.get('summary'))}"
            ),
        ))

    return EvalResult(
        checks=checks,
        meta={
            "prompt_version": getattr(run, "prompt_version", None),
            "tools_used": sorted(set(run.agent_run.tool_names())),
            "verdict": verdict_obj.get("verdict") if verdict_obj else None,
        },
    )


# ---------------------------------------------------------------------------
# LikeC4 build task
# ---------------------------------------------------------------------------


def score_likec4_build(
    dsl: str,
    case: dict,
    validation_errors: list[dict],
) -> EvalResult:
    """Score a ``likec4_build`` task result.

    The caller runs the linter and passes validation errors in — the
    scorer itself does no I/O.
    """
    max_errors = int(case.get("max_validation_errors", 0))
    keyword_score = score_phrases(dsl, case.get("expected_keywords", []))
    min_keyword_recall = float(case.get("min_keyword_recall", 1.0))

    return EvalResult(
        checks=[
            EvalCheck(
                name="validation",
                passed=len(validation_errors) <= max_errors,
                details=(
                    f"errors={len(validation_errors)} max={max_errors}"
                    + (f" first={validation_errors[:3]}" if validation_errors else "")
                ),
            ),
            EvalCheck(
                name="keywords",
                passed=keyword_score.recall >= min_keyword_recall,
                details=keyword_score.explain("keywords"),
            ),
        ],
        meta={
            "dsl_chars": len(dsl),
            "keyword_score": keyword_score,
        },
    )
