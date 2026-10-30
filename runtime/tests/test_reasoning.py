"""The one provider-reasoning projection against its shared cases (WRK-R182)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from contractor_runtime.reasoning import (
    MAX_REASONING_PART_BYTES,
    MAX_REASONING_TOTAL_BYTES,
    REASONING_TAG,
    TRUNCATION_MARKER,
    ReasoningBudget,
    reasoning_part,
    tagged_size,
)

CASES = json.loads(
    (Path(__file__).parents[2] / "api/testdata/reasoning-projection-cases.json").read_text(
        encoding="utf-8"
    )
)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "".join(
        segment if isinstance(segment, str) else segment["repeat"] * segment["count"]
        for segment in value
    )


def test_shared_reasoning_limits_match_the_registry() -> None:
    limits = CASES["limits"]
    assert limits == {
        "partBytes": MAX_REASONING_PART_BYTES,
        "totalBytes": MAX_REASONING_TOTAL_BYTES,
        "tag": REASONING_TAG,
        "truncationMarker": TRUNCATION_MARKER,
    }


@pytest.mark.parametrize("case", CASES["parts"], ids=lambda case: case["name"])
def test_shared_reasoning_part_cases(case: dict[str, Any]) -> None:
    part = reasoning_part(_text(case["text"]), case["secrets"])
    expected = case["expected"]
    if expected is None:
        assert part is None
        return
    assert part is not None
    assert part.text == _text(expected["text"])
    assert part.size == expected["size"] == tagged_size(part.text)
    assert part.size <= MAX_REASONING_PART_BYTES
    assert part.truncated is expected["truncated"]
    assert part.tagged() == {REASONING_TAG: part.text}
    encoded = json.dumps(part.tagged(), ensure_ascii=False, separators=(",", ":"))
    assert len(encoded.encode("utf-8")) == part.size
    for secret in case["secrets"]:
        assert secret not in part.text


@pytest.mark.parametrize("case", CASES["budgets"], ids=lambda case: case["name"])
def test_shared_reasoning_budget_cases(case: dict[str, Any]) -> None:
    budget = ReasoningBudget()
    assert [budget.admit(size) for size in case["sizes"]] == case["admitted"]
    assert budget.omitted is case["omitted"]
    assert budget.used == sum(
        size for size, admitted in zip(case["sizes"], case["admitted"], strict=True) if admitted
    )
    assert budget.used <= MAX_REASONING_TOTAL_BYTES
