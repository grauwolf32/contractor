from uuid import uuid4
from contractor.callbacks.tokens import TokenUsageCallback
from tests.contractor.callbacks.helpers import mk_callback_context, mk_llm_response


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
