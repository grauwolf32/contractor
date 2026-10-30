"""Budget release and large-input bounds of the provider-reasoning projection."""

from __future__ import annotations

import pytest

from contractor_runtime.reasoning import (
    MAX_REASONING_PART_BYTES,
    MAX_REASONING_TOTAL_BYTES,
    TRUNCATION_MARKER,
    ReasoningBudget,
    reasoning_part,
)


def test_reasoning_budget_release_returns_bytes_and_marks_an_omission() -> None:
    budget = ReasoningBudget()
    assert budget.admit(MAX_REASONING_TOTAL_BYTES)
    assert not budget.fits(1)
    budget.release(8192)
    assert budget.omitted is True
    assert budget.fits(8192) and not budget.fits(8193)
    with pytest.raises(RuntimeError):
        budget.release(MAX_REASONING_TOTAL_BYTES)
    with pytest.raises(RuntimeError):
        budget.admit(-1)


def test_reasoning_part_bounds_a_large_provider_text_without_scanning_past_the_cut() -> None:
    # A multi-megabyte provider reasoning string is redacted completely, but
    # the bounded search measures only a part-sized prefix per probe.
    secret = "z" * 32
    part = reasoning_part(("界" * 1_000_000) + secret, [secret])
    assert part is not None and part.truncated is True
    assert part.size <= MAX_REASONING_PART_BYTES
    assert part.text.endswith(TRUNCATION_MARKER)
