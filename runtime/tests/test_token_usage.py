from __future__ import annotations

from types import SimpleNamespace

import pytest

from contractor_runtime.token_usage import MAX_TOKEN_COUNT, project_token_usage


@pytest.mark.parametrize("usage", [None, SimpleNamespace(total_token_count=0)])
def test_missing_or_zero_total_usage_is_explicitly_unavailable(usage: object) -> None:
    projected = project_token_usage(usage)

    assert projected.total_tokens is None
    assert projected.total_unavailable is True


def test_partial_usage_keeps_independently_reported_prompt_evidence() -> None:
    projected = project_token_usage(SimpleNamespace(prompt_token_count=7168))

    assert projected.prompt_tokens == 7168
    assert projected.total_tokens is None
    assert projected.total_unavailable is True


@pytest.mark.parametrize(
    "usage",
    [
        SimpleNamespace(
            prompt_token_count=11,
            candidates_token_count=7,
            total_token_count=17,
        ),
        SimpleNamespace(
            prompt_token_count=11,
            candidates_token_count=1,
            total_token_count=12,
            cached_content_token_count=12,
        ),
    ],
)
def test_inconsistent_usage_is_dropped_closed(usage: object) -> None:
    projected = project_token_usage(usage)

    assert projected.prompt_tokens is None
    assert projected.output_tokens is None
    assert projected.total_tokens is None
    assert projected.cached_input_tokens is None
    assert projected.total_unavailable is True


def test_valid_usage_is_bounded_to_wire_counter_capacity() -> None:
    projected = project_token_usage(
        SimpleNamespace(
            prompt_token_count=MAX_TOKEN_COUNT + 10,
            candidates_token_count=3,
            total_token_count=MAX_TOKEN_COUNT + 20,
            cached_content_token_count=2,
        )
    )

    assert projected.prompt_tokens == MAX_TOKEN_COUNT
    assert projected.output_tokens == 3
    assert projected.total_tokens == MAX_TOKEN_COUNT
    assert projected.cached_input_tokens == 2
    assert projected.total_unavailable is False
