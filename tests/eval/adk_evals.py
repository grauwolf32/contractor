"""Bridge between contractor eval harnesses and ADK evaluation framework.

Provides:
  - ``agent_run_to_invocation``: convert ``AgentRun`` → ADK ``Invocation``
  - ``score_tool_name_trajectory``: name-only ordered trajectory scoring
    (ADK's ``TrajectoryEvaluator`` requires exact args match, which is too
    strict for our eval cases where tool arguments are dynamic)
"""

from __future__ import annotations

from google.adk.evaluation.eval_case import IntermediateData, Invocation
from google.genai import types

from tests.eval.harness import AgentRun


def agent_run_to_invocation(
    run: AgentRun,
    *,
    user_message: str,
    invocation_id: str = "eval-0",
) -> Invocation:
    """Convert an ``AgentRun`` to an ADK ``Invocation``.

    The returned object carries ``user_content``, ``final_response``, and
    ``intermediate_data`` (tool calls + responses) — enough to feed any
    ADK evaluator (``TrajectoryEvaluator``, ``HallucinationsV1``, etc.).
    """
    tool_uses = [
        types.FunctionCall(name=tc.name, args=tc.args)
        for tc in run.tool_calls
    ]

    tool_responses = [
        types.FunctionResponse(name=tr["name"], response={"ok": tr.get("ok", True)})
        for tr in run.tool_responses
    ]

    return Invocation(
        invocation_id=invocation_id,
        user_content=types.Content(
            role="user", parts=[types.Part(text=user_message)]
        ),
        final_response=types.Content(
            role="model", parts=[types.Part(text=run.final_text)]
        )
        if run.final_text
        else None,
        intermediate_data=IntermediateData(
            tool_uses=tool_uses,
            tool_responses=tool_responses,
        ),
    )


# ── Name-only trajectory scoring ─────────────────────────────────────────


def _actual_tool_names(run: AgentRun) -> list[str]:
    return [tc.name for tc in run.tool_calls]


def check_trajectory_in_order(
    run: AgentRun,
    expected: list[str],
) -> bool:
    """Check that *expected* tool names appear in *run*'s calls in order.

    Extra calls between expected ones are allowed (subsequence match).
    """
    if not expected:
        return True
    it = iter(expected)
    current = next(it)
    for name in _actual_tool_names(run):
        if name == current:
            try:
                current = next(it)
            except StopIteration:
                return True
    return False


def check_trajectory_any_order(
    run: AgentRun,
    expected: list[str],
) -> bool:
    """Check that every tool name in *expected* appears at least once.

    Each expected name consumes one actual call (handles duplicates).
    """
    remaining = list(_actual_tool_names(run))
    for name in expected:
        try:
            remaining.remove(name)
        except ValueError:
            return False
    return True


def score_tool_trajectory(
    run: AgentRun,
    expected: list[str],
    *,
    ordered: bool = True,
) -> TrajectoryResult:
    """Score tool-name trajectory with a detailed result.

    ``ordered=True`` uses subsequence matching (IN_ORDER);
    ``ordered=False`` uses multiset containment (ANY_ORDER).
    """
    actual = _actual_tool_names(run)
    matched = check_trajectory_in_order(run, expected) if ordered else check_trajectory_any_order(run, expected)

    actual_set = set(actual)
    expected_set = set(expected)
    missing = sorted(expected_set - actual_set)

    return TrajectoryResult(
        matched=matched,
        actual_names=actual,
        expected_names=expected,
        missing_names=missing,
        ordered=ordered,
    )


class TrajectoryResult:
    __slots__ = ("matched", "actual_names", "expected_names", "missing_names", "ordered")

    def __init__(
        self,
        *,
        matched: bool,
        actual_names: list[str],
        expected_names: list[str],
        missing_names: list[str],
        ordered: bool,
    ) -> None:
        self.matched = matched
        self.actual_names = actual_names
        self.expected_names = expected_names
        self.missing_names = missing_names
        self.ordered = ordered

    def explain(self) -> str:
        mode = "in_order" if self.ordered else "any_order"
        status = "PASS" if self.matched else "FAIL"
        lines = [f"trajectory({mode}): {status}"]
        if self.missing_names:
            lines.append(f"  missing: {self.missing_names}")
        if not self.matched:
            lines.append(f"  expected: {self.expected_names}")
            lines.append(f"  actual:   {self.actual_names}")
        return "\n".join(lines)
