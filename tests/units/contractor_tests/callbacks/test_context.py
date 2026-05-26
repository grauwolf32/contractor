import json

import pytest

from contractor.callbacks.context import (
    FunctionResultsRemovalCallback,
    SummarizationLimitCallback,
)
from tests.units.contractor_tests.helpers import (
    mk_callback_context,
    mk_function_call_part,
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
# FunctionResultsRemovalCallback — construction
# ---------------------------------------------------------------------------


def test_results_removal_rejects_both_zero():
    with pytest.raises(ValueError, match="at least one"):
        FunctionResultsRemovalCallback(keep_last_n=0, keep_budget_chars=0)


def test_results_removal_rejects_negative():
    with pytest.raises(ValueError, match="must not be negative"):
        FunctionResultsRemovalCallback(keep_last_n=-1)


# ---------------------------------------------------------------------------
# FunctionResultsRemovalCallback — count-based (keep_last_n)
# ---------------------------------------------------------------------------


def test_results_removal_no_op_on_empty_contents():
    ctx = mk_callback_context()
    cb = FunctionResultsRemovalCallback(keep_last_n=2)
    request = mk_llm_request()

    cb(ctx, request)

    assert request.contents == []
    assert cb.counter == 0


def test_results_removal_preserves_recent_function_responses():
    ctx = mk_callback_context()
    cb = FunctionResultsRemovalCallback(keep_last_n=2, deduplicate=False)

    parts = [
        mk_function_response_part(response={"v": 1}, name="tool_a"),
        mk_function_response_part(response={"v": 2}, name="tool_a"),
        mk_function_response_part(response={"v": 3}, name="tool_a"),
    ]
    request = mk_llm_request([MockContent(role="tool", parts=parts)])

    cb(ctx, request)

    # keep_last_n=2 keeps exactly 2 most-recent results.
    assert parts[-1].function_response.response == {"v": 3}
    assert parts[-2].function_response.response == {"v": 2}
    assert parts[-3].function_response.response == {"elided": True, "tool": "tool_a"}
    assert cb.counter == 1


def test_results_removal_skips_text_only_parts():
    ctx = mk_callback_context()
    cb = FunctionResultsRemovalCallback(keep_last_n=1, deduplicate=False)

    parts = [
        mk_text_part("hello"),
        mk_function_response_part(response={"v": 1}, name="tool_a"),
        mk_function_response_part(response={"v": 2}, name="tool_a"),
    ]
    request = mk_llm_request([MockContent(role="tool", parts=parts)])

    cb(ctx, request)

    assert parts[0].text == "hello"
    assert parts[-1].function_response.response == {"v": 2}
    assert parts[-2].function_response.response == {"elided": True, "tool": "tool_a"}


def test_results_removal_handles_content_with_no_parts():
    ctx = mk_callback_context()
    cb = FunctionResultsRemovalCallback(keep_last_n=3)

    request = mk_llm_request([MockContent(role="user", parts=None)])

    cb(ctx, request)  # must not raise

    assert cb.counter == 0


def test_results_removal_state_persists_counter():
    ctx = mk_callback_context()
    cb = FunctionResultsRemovalCallback(keep_last_n=2, deduplicate=False)

    parts = [
        mk_function_response_part(response={"v": 1}),
        mk_function_response_part(response={"v": 2}),
        mk_function_response_part(response={"v": 3}),
    ]
    request = mk_llm_request([MockContent(role="tool", parts=parts)])

    cb(ctx, request)

    state = ctx.state["callbacks"][f"::{cb.name}"]
    assert state["keep_last_n"] == 2
    assert state["counter"] == 1


# ---------------------------------------------------------------------------
# FunctionResultsRemovalCallback — budget-based (keep_budget_chars)
# ---------------------------------------------------------------------------


def _big_response(n_chars: int, tag: str = "x") -> dict:
    return {"data": tag * n_chars}


def test_budget_elides_when_over_char_limit():
    ctx = mk_callback_context()
    cb = FunctionResultsRemovalCallback(
        keep_budget_chars=100, deduplicate=False,
    )

    parts = [
        mk_function_response_part(response=_big_response(80, "a"), name="t"),
        mk_function_response_part(response=_big_response(80, "b"), name="t"),
        mk_function_response_part(response=_big_response(80, "c"), name="t"),
    ]
    request = mk_llm_request([MockContent(role="tool", parts=parts)])

    cb(ctx, request)

    # Most recent (c) is always kept. Next (b) would push over 100 → elided.
    assert parts[-1].function_response.response == _big_response(80, "c")
    assert parts[-2].function_response.response == {"elided": True, "tool": "t"}
    assert parts[-3].function_response.response == {"elided": True, "tool": "t"}


def test_budget_always_keeps_at_least_one():
    ctx = mk_callback_context()
    cb = FunctionResultsRemovalCallback(
        keep_budget_chars=10, deduplicate=False,
    )

    parts = [
        mk_function_response_part(response=_big_response(9999, "a"), name="t"),
    ]
    request = mk_llm_request([MockContent(role="tool", parts=parts)])

    cb(ctx, request)

    assert parts[0].function_response.response == _big_response(9999, "a")
    assert cb.counter == 0


def test_budget_and_count_both_apply():
    ctx = mk_callback_context()
    cb = FunctionResultsRemovalCallback(
        keep_last_n=2, keep_budget_chars=999999, deduplicate=False,
    )

    parts = [
        mk_function_response_part(response={"v": 1}, name="t"),
        mk_function_response_part(response={"v": 2}, name="t"),
        mk_function_response_part(response={"v": 3}, name="t"),
    ]
    request = mk_llm_request([MockContent(role="tool", parts=parts)])

    cb(ctx, request)

    # Budget is huge but count caps at 2.
    assert parts[-1].function_response.response == {"v": 3}
    assert parts[-2].function_response.response == {"v": 2}
    assert parts[-3].function_response.response == {"elided": True, "tool": "t"}


# ---------------------------------------------------------------------------
# FunctionResultsRemovalCallback — staleness / deduplication
# ---------------------------------------------------------------------------


def _make_call_response_pair(name, args, response):
    """Return (model Content with function_call, tool Content with function_response)."""
    return (
        MockContent(role="model", parts=[mk_function_call_part(name=name, args=args)]),
        MockContent(
            role="tool",
            parts=[mk_function_response_part(response=response, name=name)],
        ),
    )


def test_dedup_elides_older_duplicate():
    ctx = mk_callback_context()
    cb = FunctionResultsRemovalCallback(keep_last_n=99)

    c1_call, c1_resp = _make_call_response_pair(
        "read_file", {"path": "/a.py"}, {"content": "old"}
    )
    c2_call, c2_resp = _make_call_response_pair(
        "read_file", {"path": "/a.py"}, {"content": "new"}
    )
    request = mk_llm_request([c1_call, c1_resp, c2_call, c2_resp])

    cb(ctx, request)

    # Second (newer) call is kept; first (older, same args) is stale.
    fr_old = c1_resp.parts[0].function_response
    fr_new = c2_resp.parts[0].function_response
    assert fr_new.response == {"content": "new"}
    assert fr_old.response["elided"] is True
    assert fr_old.response["reason"] == "stale"


def test_dedup_keeps_different_args():
    ctx = mk_callback_context()
    cb = FunctionResultsRemovalCallback(keep_last_n=99)

    c1_call, c1_resp = _make_call_response_pair(
        "read_file", {"path": "/a.py"}, {"content": "aaa"}
    )
    c2_call, c2_resp = _make_call_response_pair(
        "read_file", {"path": "/b.py"}, {"content": "bbb"}
    )
    request = mk_llm_request([c1_call, c1_resp, c2_call, c2_resp])

    cb(ctx, request)

    # Different args → both kept.
    assert c1_resp.parts[0].function_response.response == {"content": "aaa"}
    assert c2_resp.parts[0].function_response.response == {"content": "bbb"}
    assert cb.counter == 0


def test_dedup_with_multiple_stale_calls():
    ctx = mk_callback_context()
    cb = FunctionResultsRemovalCallback(keep_last_n=99)

    pairs = [
        _make_call_response_pair("read_file", {"path": "/a.py"}, {"v": 1}),
        _make_call_response_pair("read_file", {"path": "/a.py"}, {"v": 2}),
        _make_call_response_pair("read_file", {"path": "/a.py"}, {"v": 3}),
    ]
    contents = [c for pair in pairs for c in pair]
    request = mk_llm_request(contents)

    cb(ctx, request)

    # Only the latest (v=3) survives; v=1 and v=2 are stale.
    assert pairs[0][1].parts[0].function_response.response["elided"] is True
    assert pairs[1][1].parts[0].function_response.response["elided"] is True
    assert pairs[2][1].parts[0].function_response.response == {"v": 3}
    assert cb.counter == 2


def test_dedup_disabled():
    ctx = mk_callback_context()
    cb = FunctionResultsRemovalCallback(keep_last_n=99, deduplicate=False)

    c1_call, c1_resp = _make_call_response_pair(
        "read_file", {"path": "/a.py"}, {"content": "old"}
    )
    c2_call, c2_resp = _make_call_response_pair(
        "read_file", {"path": "/a.py"}, {"content": "new"}
    )
    request = mk_llm_request([c1_call, c1_resp, c2_call, c2_resp])

    cb(ctx, request)

    # Both kept when dedup is off.
    assert c1_resp.parts[0].function_response.response == {"content": "old"}
    assert c2_resp.parts[0].function_response.response == {"content": "new"}
    assert cb.counter == 0


def test_dedup_plus_budget():
    ctx = mk_callback_context()
    cb = FunctionResultsRemovalCallback(keep_budget_chars=200)

    pairs = [
        _make_call_response_pair("read_file", {"path": "/a.py"}, _big_response(80, "a")),
        _make_call_response_pair("read_file", {"path": "/a.py"}, _big_response(80, "b")),
        _make_call_response_pair("grep", {"pattern": "x"}, _big_response(80, "c")),
    ]
    contents = [c for pair in pairs for c in pair]
    request = mk_llm_request(contents)

    cb(ctx, request)

    # read_file /a.py called twice: first is stale, always elided.
    # grep is unique. Budget: grep(~90 chars) + read_file/b(~90 chars) ≈ 180 < 200.
    assert pairs[0][1].parts[0].function_response.response["elided"] is True
    assert pairs[0][1].parts[0].function_response.response["reason"] == "stale"
    assert pairs[1][1].parts[0].function_response.response == _big_response(80, "b")
    assert pairs[2][1].parts[0].function_response.response == _big_response(80, "c")


def test_to_state_includes_new_fields():
    cb = FunctionResultsRemovalCallback(
        keep_last_n=5, keep_budget_chars=10000, target_tools=["read_file"],
    )
    state = cb.to_state()
    assert state["keep_last_n"] == 5
    assert state["keep_budget_chars"] == 10000
    assert state["deduplicate"] is True
    assert state["counter"] == 0
    assert state["target_tools"] == ["read_file"]
