"""End-to-end eval for ``trace_agent``.

Each case anchors a trace at a specific entrypoint (route handler or
function) and lists the functions the agent is expected to annotate.  We
score the produced ``(file, function)`` set against ground truth using
precision/recall.

Every ``(fixture, case)`` pair is its own pytest item via the
``trace_case`` fixture (parametrized in ``conftest.pytest_generate_tests``).
This gives independent timeouts, xdist parallelism, and per-case CI
visibility.

Prompt version: a case may specify ``prompt_version`` to pin a specific
entry from ``contractor/agents/trace_agent/prompt.yml``.  The env var
``CONTRACTOR_EVAL_TRACE_PROMPT_VERSION`` overrides every case (handy for
sweeping a candidate version across all fixtures).  When neither is set
the manifest's ``active`` version is used.
"""

from __future__ import annotations

import os

import pytest

from tests.eval.results import CaseResult, case_artifact_dir, metrics_from_events
from tests.eval.scorers import diff_detail, score_trace_run
from tests.eval.trace_harness import run_trace_agent


def _user_message(case: dict) -> str:
    entry = case["entrypoint"]
    where = entry.get("file", "?")
    func = entry.get("function") or entry.get("route") or "?"
    intent = case.get("intent", "Annotate every function on the relevant flow.")
    return (
        f"Trace the request flow that begins at `{func}` in `{where}`. "
        f"{intent} "
        "Insert `# @trace target=... args=... calls=...` comments above each "
        "function definition you confidently identify as part of the path. "
        "Use the `insert_line` tool to mutate files. Stop once the path is "
        "covered."
    )


def _resolve_prompt_version(case: dict) -> str | None:
    return os.environ.get("CONTRACTOR_EVAL_TRACE_PROMPT_VERSION") or case.get(
        "prompt_version"
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_trace_agent(trace_case, eval_model, eval_sink):
    fixture, case = trace_case

    run = await run_trace_agent(
        fixture_root=fixture.source_root,
        user_message=_user_message(case),
        model=eval_model,
        namespace=f"trace-eval-{fixture.slug}-{case['id']}",
        timeout_s=float(case.get("timeout_s", 900.0)),
        prompt_version=_resolve_prompt_version(case),
        with_graph_tools=bool(case.get("with_graph_tools", True)),
        artifact_dir=case_artifact_dir("trace_agent", fixture.slug, case["id"]),
    )

    result = score_trace_run(run, case)
    _agent_run = getattr(run, "agent_run", None)
    eval_sink.record(
        scenario="agent", unit="trace_agent", metric_kind="diff",
        fixture=fixture.slug, model=str(eval_model.model),
        prompt_version=_resolve_prompt_version(case),
        case=CaseResult(id=case["id"], passed=result.passed,
                        pass_count=int(result.passed), attempts=1,
                        metrics=metrics_from_events(getattr(_agent_run, "metrics_events", [])),
                        detail=diff_detail(result)),
        artifacts=getattr(_agent_run, "artifacts", {}) or {},
    )
    assert result.passed, f"trace_agent eval failed: case={case['id']}\n{result.explain()}"
