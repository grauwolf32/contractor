from types import SimpleNamespace

from contractor.callbacks.tokens import TokenCounter, TokenUsageCallback
from tests.units.contractor_tests.helpers import mk_callback_context, mk_llm_response


def test_series_same_interaction_then_change_and_more():
    """
    Several requests within one invocation_id, then the invocation_id
    changes, then more requests. Checks:
      - the global counter always accumulates
      - the current counter accumulates only within the current invocation_id
      - history tracks the per-invocation totals, including the in-progress
        invocation (flushed on every call, keyed by invocation_id)
    """
    ctx = mk_callback_context()
    token_usage_callback = TokenUsageCallback()
    ctx.invocation_id = "A"
    # interaction A: 2 calls
    token_usage_callback(ctx, mk_llm_response(total=10, prompt=6, candidates=4))
    token_usage_callback(ctx, mk_llm_response(total=3, prompt=1, candidates=2))
    state_key = "::" + token_usage_callback.name

    s = ctx.state["callbacks"][state_key]
    assert s["counter"] == {"input": 7, "output": 6, "total": 13}

    g = TokenUsageCallback.get_global_counter(ctx)
    assert g.input == 7
    assert g.output == 6
    assert g.total == 13

    # The in-progress invocation is already visible in history — consumers
    # reading mid-run (or after the final invocation) never undercount.
    h = TokenUsageCallback.get_history(ctx)
    assert h == {"A": {"input": 7, "output": 6, "total": 13}}

    # interaction B: the first call with a new id starts a fresh current
    # counter; A's totals stay frozen in history.
    ctx.invocation_id = "B"
    new_invocation_id = ctx.invocation_id
    token_usage_callback(ctx, mk_llm_response(total=5, prompt=2, candidates=3))

    s = ctx.state["callbacks"][state_key]
    assert s["invocation_id"] == new_invocation_id
    assert s["counter"] == {"input": 2, "output": 3, "total": 5}

    g = TokenUsageCallback.get_global_counter(ctx)
    assert g.input == 9
    assert g.output == 9
    assert g.total == 18

    h = TokenUsageCallback.get_history(ctx)
    assert h == {
        "A": {"input": 7, "output": 6, "total": 13},
        "B": {"input": 2, "output": 3, "total": 5},
    }

    # interaction B: one more call — the current counter accumulates and the
    # history entry follows it.
    token_usage_callback(ctx, mk_llm_response(total=2, prompt=1, candidates=1))

    s = ctx.state["callbacks"][state_key]
    assert s["invocation_id"] == new_invocation_id
    assert s["counter"] == {"input": 3, "output": 4, "total": 7}

    g = TokenUsageCallback.get_global_counter(ctx)
    assert g.input == 10
    assert g.output == 10
    assert g.total == 20

    h = TokenUsageCallback.get_history(ctx)
    assert h == {
        "A": {"input": 7, "output": 6, "total": 13},
        "B": {"input": 3, "output": 4, "total": 7},
    }


def test_history_includes_final_invocation_without_id_change():
    # Regression: history used to be written only when invocation_id changed,
    # so the LAST invocation of a run was never flushed and consumers
    # undercounted by one invocation. The flush-on-every-call seam keeps the
    # final invocation's entry present and accurate without double-counting.
    ctx = mk_callback_context()
    ctx.invocation_id = "only"
    cb = TokenUsageCallback()

    cb(ctx, mk_llm_response(total=10, prompt=6, candidates=4))
    cb(ctx, mk_llm_response(total=3, prompt=1, candidates=2))

    h = TokenUsageCallback.get_history(ctx)
    assert h == {"only": {"input": 7, "output": 6, "total": 13}}
    # History totals equal the global counter — nothing lost, nothing doubled.
    g = TokenUsageCallback.get_global_counter(ctx)
    assert h["only"] == {"input": g.input, "output": g.output, "total": g.total}


# ─── TokenCounter ────────────────────────────────────────────────────────────


class TestTokenCounter:
    def test_is_empty_on_default(self):
        assert TokenCounter().is_empty()

    def test_not_empty_when_any_field_nonzero(self):
        assert not TokenCounter(input=1).is_empty()
        assert not TokenCounter(output=1).is_empty()
        assert not TokenCounter(total=1).is_empty()

    def test_add_accumulates_per_field(self):
        a = TokenCounter(input=1, output=2, total=3)
        a.add(TokenCounter(input=10, output=20, total=30))
        assert (a.input, a.output, a.total) == (11, 22, 33)


# ─── TokenUsageCallback edge cases ────────────────────────────────────────────


class TestTokenUsageCallbackEdges:
    def test_no_usage_metadata_is_noop(self):
        ctx = mk_callback_context()
        ctx.invocation_id = "x"
        cb = TokenUsageCallback()
        # An LlmResponse without `usage_metadata` (or with it None) must not
        # mutate state — important for upstream providers that occasionally
        # return only an error or a partial response.
        cb(ctx, SimpleNamespace(usage_metadata=None))
        assert "callbacks" in ctx.state
        # No counter state was created for this callback.
        assert TokenUsageCallback.global_counter_key() not in ctx.state

    def test_none_token_fields_treated_as_zero(self):
        # The Anthropic / LiteLLM responses sometimes return None for the
        # individual usage fields. The callback must coerce to 0 rather than
        # propagate None into the counter.
        ctx = mk_callback_context()
        ctx.invocation_id = "x"
        cb = TokenUsageCallback()
        usage = SimpleNamespace(
            prompt_token_count=None,
            candidates_token_count=None,
            total_token_count=None,
        )
        cb(ctx, SimpleNamespace(usage_metadata=usage))
        g = TokenUsageCallback.get_global_counter(ctx)
        assert (g.input, g.output, g.total) == (0, 0, 0)
        # The callback latched onto the invocation_id even with zero counts.
        state_key = "::" + cb.name
        assert ctx.state["callbacks"][state_key]["invocation_id"] == "x"

    def test_first_call_adopts_ctx_invocation_id(self):
        ctx = mk_callback_context()
        ctx.invocation_id = "first"
        cb = TokenUsageCallback()
        cb(ctx, mk_llm_response(total=10, prompt=4, candidates=6))
        state_key = "::" + cb.name
        assert ctx.state["callbacks"][state_key]["invocation_id"] == "first"
        assert ctx.state["callbacks"][state_key]["counter"] == {
            "input": 4, "output": 6, "total": 10,
        }
