"""End-to-end eval for `trace_agent`.

Each case anchors a trace at a specific entrypoint (route handler or
function) and lists the functions the agent is expected to annotate. We
score the produced `(file, function)` set against ground truth using the
same precision/recall machinery as the other agents.
"""

from __future__ import annotations

import pytest

from tests.eval.scoring import _score_sets
from tests.eval.trace_harness import Annotation, run_trace_agent


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


def _expected_set(case: dict) -> set[Annotation]:
    return {
        Annotation(file=item["file"], function=item["function"])
        for item in case["expected_annotated"]
    }


@pytest.mark.eval
@pytest.mark.asyncio
async def test_trace_agent_cases(fixture, eval_model):
    if not fixture.trace_cases:
        pytest.skip(f"no trace cases defined for fixture {fixture.slug}")

    failures: list[str] = []

    for case in fixture.trace_cases:
        run = await run_trace_agent(
            fixture_root=fixture.source_root,
            user_message=_user_message(case),
            model=eval_model,
            namespace=f"trace-eval-{fixture.slug}-{case['id']}",
            timeout_s=float(case.get("timeout_s", 900.0)),
        )

        expected = _expected_set(case)
        actual_keys = {a.as_tuple() for a in run.annotations}
        expected_keys = {a.as_tuple() for a in expected}
        score = _score_sets(actual_keys, expected_keys)

        min_precision = float(case.get("min_precision", 0.5))
        min_recall = float(case.get("min_recall", 0.5))

        if not score.passes(min_precision=min_precision, min_recall=min_recall):
            failures.append(
                f"case={case['id']}\n"
                f"  modified_files={sorted(run.modified_files)}\n"
                f"  {score.explain('annotations')}\n"
                f"  tools_used={sorted(set(run.agent_run.tool_names()))}"
            )

    assert not failures, "trace_agent eval failures:\n\n" + "\n\n".join(failures)
