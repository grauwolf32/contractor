from types import SimpleNamespace

from contractor.callbacks.tokens import TokenCounter, TokenUsageCallback
from tests.units.contractor_tests.helpers import mk_callback_context, mk_llm_response


def test_series_same_interaction_then_change_and_more():
    """
    1) Несколько запросов в рамках одного invocation_id, потом invocation_id меняется, и ещё запросы.
    Проверяем:
      - common суммируется всегда
      - current суммируется только в рамках текущего invocation_id
      - при смене invocation_id прошлый current уходит в history под старым id
    """
    ctx = mk_callback_context()
    token_usage_callback = TokenUsageCallback()
    ctx.invocation_id = "A"
    # interaction A: 2 вызова
    token_usage_callback(ctx, mk_llm_response(total=10, prompt=6, candidates=4))
    token_usage_callback(ctx, mk_llm_response(total=3, prompt=1, candidates=2))
    state_key = "::" + token_usage_callback.name

    s = ctx.state["callbacks"][state_key]
    assert s["counter"] == {"input": 7, "output": 6, "total": 13}

    g = TokenUsageCallback.get_global_counter(ctx)
    assert g.input == 7
    assert g.output == 6
    assert g.total == 13

    h = TokenUsageCallback.get_history(ctx)
    assert h == {}

    # interaction B: первый вызов с новым id переносит A->history и ставит current = token_count(B)
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
    assert h == {"A": {"input": 7, "output": 6, "total": 13}}

    # interaction B: ещё один вызов — current накапливается
    token_usage_callback(ctx, mk_llm_response(total=2, prompt=1, candidates=1))

    s = ctx.state["callbacks"][state_key]
    assert s["invocation_id"] == new_invocation_id
    assert s["counter"] == {"input": 3, "output": 4, "total": 7}

    g = TokenUsageCallback.get_global_counter(ctx)
    assert g.input == 10
    assert g.output == 10
    assert g.total == 20

    h = TokenUsageCallback.get_history(ctx)
    assert h == {"A": {"input": 7, "output": 6, "total": 13}}


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
