from contractor.callbacks.tokens import TokenUsageCallback
from tests.contractor.callbacks.helpers import mk_callback_context, mk_llm_response


def test_series_same_interaction_then_change_and_more():
    """
    1) Несколько запросов в рамках одного interaction_id, потом interaction_id меняется, и ещё запросы.
    Проверяем:
      - common суммируется всегда
      - current суммируется только в рамках текущего interaction_id
      - при смене interaction_id прошлый current уходит в history под старым id
    """
    ctx = mk_callback_context()
    token_usage_callback = TokenUsageCallback()

    # interaction A: 2 вызова
    token_usage_callback(ctx, mk_llm_response("A", total=10, prompt=6, candidates=4))
    token_usage_callback(ctx, mk_llm_response("A", total=3, prompt=1, candidates=2))

    s = ctx.state["callbacks"][token_usage_callback.name]
    assert s["interaction_id"] == "A"
    assert s["common"] == {"input": 7, "output": 6, "total": 13}
    assert s["current"] == {"input": 7, "output": 6, "total": 13}
    assert s["history"] == {}

    # interaction B: первый вызов с новым id переносит A->history и ставит current = token_count(B)
    token_usage_callback(ctx, mk_llm_response("B", total=5, prompt=2, candidates=3))

    s = ctx.state["callbacks"][token_usage_callback.name]
    assert s["interaction_id"] == "B"
    assert s["common"] == {"input": 9, "output": 9, "total": 18}
    assert s["current"] == {"input": 2, "output": 3, "total": 5}
    assert s["history"] == {"A": {"input": 7, "output": 6, "total": 13}}

    # interaction B: ещё один вызов — current накапливается
    token_usage_callback(ctx, mk_llm_response("B", total=2, prompt=1, candidates=1))

    s = ctx.state["callbacks"][token_usage_callback.name]
    assert s["interaction_id"] == "B"
    assert s["common"] == {"input": 10, "output": 10, "total": 20}
    assert s["current"] == {"input": 3, "output": 4, "total": 7}
    assert s["history"] == {"A": {"input": 7, "output": 6, "total": 13}}


def test_request_when_state_interaction_id_none():
    """
    2) Делается запрос при interaction_id=None в state.
    Ожидаем: callback возьмёт interaction_id из llm_response.interaction_id,
    и будет корректно вести common/current.
    """
    ctx = mk_callback_context()
    token_usage_callback = TokenUsageCallback()

    # Первый вызов: state interaction_id=None -> должен стать "X"
    token_usage_callback(ctx, mk_llm_response("X", total=4, prompt=3, candidates=1))

    s = ctx.state["callbacks"][token_usage_callback.name]
    assert s["interaction_id"] == "X"
    assert s["common"] == {"input": 3, "output": 1, "total": 4}
    assert s["current"] == {"input": 3, "output": 1, "total": 4}
    assert s["history"] == {}

    # Второй вызов в рамках "X": current должен накопиться
    token_usage_callback(ctx, mk_llm_response("X", total=6, prompt=2, candidates=4))

    s = ctx.state["callbacks"][token_usage_callback.name]
    assert s["interaction_id"] == "X"
    assert s["common"] == {"input": 5, "output": 5, "total": 10}
    assert s["current"] == {"input": 5, "output": 5, "total": 10}
    assert s["history"] == {}
