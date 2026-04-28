import pytest

from contractor.callbacks.context import (
    FunctionResultsRemovalCallback,
    SummarizationLimitCallback,
)
from tests.units.contractor_tests.helpers import (
    mk_callback_context,
    mk_function_response_part,
    mk_llm_request,
    mk_text_part,
    MockContent,
)


# ---------------------------------------------------------------------------
# SummarizationLimitCallback
# ---------------------------------------------------------------------------


TOKEN_STATE_KEY = "::TokenUsageCallback"


def _seed_token_state(ctx, total: int) -> None:
    """Mirror the layout TokenUsageCallback writes via save_to_state."""
    ctx.state.setdefault("callbacks", {})
    ctx.state["callbacks"][TOKEN_STATE_KEY] = {
        "counter": {"input": 0, "output": 0, "total": total},
        "invocation_id": ctx.invocation_id,
    }


def test_summarization_no_op_when_under_limit():
    ctx = mk_callback_context()
    _seed_token_state(ctx, total=100)

    cb = SummarizationLimitCallback(message="please summarize", max_tokens=1000)
    request = mk_llm_request()

    cb(ctx, request)

    assert request.contents == []  # nothing appended
    state = ctx.state["callbacks"][f"::{cb.name}"]
    assert state["token_count"] == 100
    assert state["history"] == []


def test_summarization_appends_message_when_over_limit():
    ctx = mk_callback_context()
    _seed_token_state(ctx, total=2000)

    cb = SummarizationLimitCallback(message="summarize now", max_tokens=1000)
    request = mk_llm_request()

    cb(ctx, request)

    assert len(request.contents) == 1
    appended = request.contents[0]
    assert appended.role == "user"
    assert appended.parts[0].text == "summarize now"

    state = ctx.state["callbacks"][f"::{cb.name}"]
    assert len(state["history"]) == 1
    assert isinstance(state["history"][0], int)


def test_summarization_threshold_is_strict_less_than():
    # Exactly at max_tokens should still trigger summarization.
    ctx = mk_callback_context()
    _seed_token_state(ctx, total=1000)

    cb = SummarizationLimitCallback(message="hit", max_tokens=1000)
    request = mk_llm_request()

    cb(ctx, request)

    assert len(request.contents) == 1


def test_summarization_handles_missing_token_state():
    # No token state seeded — should default to 0 and not append.
    ctx = mk_callback_context()
    cb = SummarizationLimitCallback(message="m", max_tokens=10)
    request = mk_llm_request()

    cb(ctx, request)

    assert request.contents == []
    state = ctx.state["callbacks"][f"::{cb.name}"]
    assert state["token_count"] == 0


def test_summarization_respects_custom_summarization_key():
    ctx = mk_callback_context()
    ctx.state.setdefault("callbacks", {})
    ctx.state["callbacks"][TOKEN_STATE_KEY] = {
        "counter": {"input": 999, "output": 1, "total": 1000},
        "invocation_id": ctx.invocation_id,
    }

    cb = SummarizationLimitCallback(
        message="m", max_tokens=500, summarization_key="input"
    )
    request = mk_llm_request()

    cb(ctx, request)

    assert len(request.contents) == 1  # 999 input >= 500 max


def test_summarization_to_state_shape():
    cb = SummarizationLimitCallback(message="m", max_tokens=100)
    state = cb.to_state()
    assert set(state.keys()) == {"max_tokens", "token_count", "message", "history"}


# ---------------------------------------------------------------------------
# FunctionResultsRemovalCallback
# ---------------------------------------------------------------------------


def test_results_removal_requires_keep_last_n_above_one():
    with pytest.raises(AssertionError):
        FunctionResultsRemovalCallback(keep_last_n=1)


def test_results_removal_no_op_on_empty_contents():
    ctx = mk_callback_context()
    cb = FunctionResultsRemovalCallback(keep_last_n=2)
    request = mk_llm_request()

    cb(ctx, request)

    assert request.contents == []
    assert cb.counter == 0


def test_results_removal_preserves_recent_function_responses():
    ctx = mk_callback_context()
    cb = FunctionResultsRemovalCallback(keep_last_n=2)

    parts = [
        mk_function_response_part(response={"v": 1}),
        mk_function_response_part(response={"v": 2}),
        mk_function_response_part(response={"v": 3}),
    ]
    request = mk_llm_request([MockContent(role="tool", parts=parts)])

    cb(ctx, request)

    # parts is iterated in reverse: the LAST function_response counts as #1.
    # With keep_last_n=2, we keep #1 (v=3) only — wait: the loop checks
    # `func_count < self.keep_last_n` to *skip* removal. So when func_count==1
    # (last one, v=3), 1 < 2 so it skips removal. When func_count==2 (v=2),
    # 2 < 2 is false → it strips. When func_count==3 (v=1), it strips.
    # So only the most-recent one (v=3) is preserved.
    assert parts[-1].function_response.response == {"v": 3}
    assert parts[-2].function_response.response == {}
    assert parts[-3].function_response.response == {}
    assert cb.counter == 2  # two responses cleared


def test_results_removal_skips_text_only_parts():
    ctx = mk_callback_context()
    cb = FunctionResultsRemovalCallback(keep_last_n=2)

    parts = [
        mk_text_part("hello"),
        mk_function_response_part(response={"v": 1}),
        mk_function_response_part(response={"v": 2}),
    ]
    request = mk_llm_request([MockContent(role="tool", parts=parts)])

    cb(ctx, request)

    # Text part is untouched.
    assert parts[0].text == "hello"
    # Most recent function_response is preserved.
    assert parts[-1].function_response.response == {"v": 2}
    # Older one is wiped.
    assert parts[-2].function_response.response == {}


def test_results_removal_handles_content_with_no_parts():
    ctx = mk_callback_context()
    cb = FunctionResultsRemovalCallback(keep_last_n=3)

    request = mk_llm_request([MockContent(role="user", parts=None)])

    cb(ctx, request)  # must not raise

    assert cb.counter == 0


def test_results_removal_state_persists_counter():
    ctx = mk_callback_context()
    cb = FunctionResultsRemovalCallback(keep_last_n=2)

    parts = [
        mk_function_response_part(response={"v": 1}),
        mk_function_response_part(response={"v": 2}),
        mk_function_response_part(response={"v": 3}),
    ]
    request = mk_llm_request([MockContent(role="tool", parts=parts)])

    cb(ctx, request)

    state = ctx.state["callbacks"][f"::{cb.name}"]
    assert state["keep_last_n"] == 2
    assert state["counter"] == 2
