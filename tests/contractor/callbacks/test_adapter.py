from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.tokens import TokenUsageCallback
from tests.contractor.callbacks.helpers import mk_callback_context, mk_llm_response


def test_middleware_registration():
    middleware = CallbackAdapter()
    middleware.register(TokenUsageCallback())


def test_callback_chain_call():
    middleware = CallbackAdapter()
    middleware.register(TokenUsageCallback())

    chain = middleware.get_chain(TokenUsageCallback().cb_type)

    ctx = mk_callback_context()

    chain(ctx, mk_llm_response("A", total=10, prompt=6, candidates=4))
    chain(ctx, mk_llm_response("A", total=3, prompt=1, candidates=2))

    s = ctx.state["callbacks"][TokenUsageCallback().name]
    assert s["interaction_id"] == "A"
    assert s["common"] == {"input": 7, "output": 6, "total": 13}
    assert s["current"] == {"input": 7, "output": 6, "total": 13}
    assert s["history"] == {}
