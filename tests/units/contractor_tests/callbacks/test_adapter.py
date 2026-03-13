from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.tokens import TokenUsageCallback
from tests.units.contractor_tests.helpers import mk_callback_context, mk_llm_response


def test_middleware_registration():
    middleware = CallbackAdapter()
    middleware.register(TokenUsageCallback())


def test_callback_chain_call():
    middleware = CallbackAdapter()
    middleware.register(TokenUsageCallback())

    chain = middleware.get_chain(TokenUsageCallback().cb_type)

    ctx = mk_callback_context()

    chain(ctx, mk_llm_response(total=10, prompt=6, candidates=4))
    chain(ctx, mk_llm_response(total=3, prompt=1, candidates=2))

    state_key = "::" + TokenUsageCallback().name
    s = ctx.state["callbacks"][state_key]
    invocation_id = ctx.invocation_id

    assert s["invocation_id"] == invocation_id
    assert s["counter"] == {"input": 7, "output": 6, "total": 13}
