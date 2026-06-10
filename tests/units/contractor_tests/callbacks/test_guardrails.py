from types import SimpleNamespace

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.guardrails import (
    InvalidToolCallGuardrailCallback,
    MandatoryToolCallback,
    RepeatedToolCallCallback,
)
from tests.units.contractor_tests.helpers import (
    MockContent,
    mk_callback_context,
    mk_function_call_part,
    mk_text_part,
    mk_tool_context,
)


def _mk_tool(name: str):
    return SimpleNamespace(name=name)


def test_repeated_tool_call_below_threshold_passes_through():
    cb = RepeatedToolCallCallback(threshold=3)
    ctx = mk_tool_context()
    tool = _mk_tool("read_file")

    assert cb(tool, {"path": "a"}, ctx) is None
    assert cb(tool, {"path": "a"}, ctx) is None
    assert cb.run_length == 2


def test_repeated_tool_call_triggers_at_threshold():
    cb = RepeatedToolCallCallback(threshold=3)
    ctx = mk_tool_context()
    tool = _mk_tool("read_file")
    args = {"path": "a"}

    cb(tool, args, ctx)
    cb(tool, args, ctx)
    result = cb(tool, args, ctx)

    assert isinstance(result, dict)
    assert "warning" in result
    assert "read_file" in result["warning"]
    assert "3" in result["warning"]
    assert len(cb.history) == 1


def test_repeated_tool_call_keeps_warning_after_threshold_but_history_unchanged():
    cb = RepeatedToolCallCallback(threshold=2)
    ctx = mk_tool_context()
    tool = _mk_tool("ls")
    args = {"path": "/"}

    cb(tool, args, ctx)
    cb(tool, args, ctx)  # triggers, history grows
    cb(tool, args, ctx)  # still warns, history stays

    assert cb.run_length == 3
    assert len(cb.history) == 1


def test_different_args_resets_run_length():
    cb = RepeatedToolCallCallback(threshold=3)
    ctx = mk_tool_context()
    tool = _mk_tool("read_file")

    cb(tool, {"path": "a"}, ctx)
    cb(tool, {"path": "a"}, ctx)
    cb(tool, {"path": "b"}, ctx)

    assert cb.run_length == 1
    assert cb.last_signature is not None
    assert "b" in cb.last_signature


def test_different_tool_resets_run_length():
    cb = RepeatedToolCallCallback(threshold=3)
    ctx = mk_tool_context()

    cb(_mk_tool("read_file"), {"path": "a"}, ctx)
    cb(_mk_tool("read_file"), {"path": "a"}, ctx)
    cb(_mk_tool("ls"), {"path": "a"}, ctx)

    assert cb.run_length == 1


def test_arg_order_does_not_matter():
    cb = RepeatedToolCallCallback(threshold=2)
    ctx = mk_tool_context()
    tool = _mk_tool("grep")

    cb(tool, {"pattern": "x", "path": "/"}, ctx)
    result = cb(tool, {"path": "/", "pattern": "x"}, ctx)

    assert isinstance(result, dict)
    assert "warning" in result


def test_unhashable_args_does_not_crash():
    cb = RepeatedToolCallCallback(threshold=2)
    ctx = mk_tool_context()
    tool = _mk_tool("foo")

    class NotJsonable:
        pass

    args = {"obj": NotJsonable()}
    # falls back to repr; repeating the same object identity should still match
    cb(tool, args, ctx)
    result = cb(tool, args, ctx)

    assert isinstance(result, dict)


def test_empty_args_pass_through_and_never_trigger():
    cb = RepeatedToolCallCallback(threshold=2)
    ctx = mk_tool_context()
    tool = _mk_tool("execute_current_subtask")

    assert cb(tool, {}, ctx) is None
    assert cb(tool, {}, ctx) is None
    assert cb(tool, None, ctx) is None  # type: ignore[arg-type]

    assert cb.run_length == 0
    assert cb.last_signature is None
    assert cb.history == []


def test_empty_args_do_not_break_existing_streak():
    cb = RepeatedToolCallCallback(threshold=3)
    ctx = mk_tool_context()
    add = _mk_tool("add_subtask")
    execute = _mk_tool("execute_current_subtask")

    cb(add, {"description": "x"}, ctx)
    cb(execute, {}, ctx)  # transparent
    cb(add, {"description": "x"}, ctx)
    cb(execute, {}, ctx)  # transparent
    result = cb(add, {"description": "x"}, ctx)

    assert isinstance(result, dict)
    assert "warning" in result
    assert "add_subtask" in result["warning"]


# ---------------------------------------------------------------------------
# InvalidToolCallGuardrailCallback — return-value contract
# ---------------------------------------------------------------------------


def _named_tool(name: str):
    def _tool():
        pass

    _tool.__name__ = name
    return _tool


def _mk_invalid_tool_cb(
    tool_names: tuple[str, ...] = ("default_tool", "submit_verdict", "read_file"),
) -> InvalidToolCallGuardrailCallback:
    return InvalidToolCallGuardrailCallback(
        tools=[_named_tool(n) for n in tool_names],
        default_tool_name="default_tool",
        default_tool_arg="meta",
    )


def _mk_model_response(parts):
    return SimpleNamespace(content=MockContent(role="model", parts=list(parts)))


def test_invalid_tool_cb_returns_none_when_nothing_modified():
    cb = _mk_invalid_tool_cb()
    ctx = mk_callback_context()
    resp = _mk_model_response(
        [
            mk_text_part("thinking..."),
            mk_function_call_part(name="read_file", args={"path": "a"}),
        ]
    )

    assert cb(ctx, resp) is None
    assert cb.history == []
    # state is still saved even when nothing was rewritten
    assert "::InvalidToolCallGuardrailCallback" in ctx.state["callbacks"]


def test_invalid_tool_cb_returns_none_for_text_only_response():
    cb = _mk_invalid_tool_cb()
    ctx = mk_callback_context()
    resp = _mk_model_response([mk_text_part("final answer")])

    assert cb(ctx, resp) is None
    assert cb.history == []


def test_invalid_tool_cb_returns_response_when_it_rewrites_a_part():
    cb = _mk_invalid_tool_cb()
    ctx = mk_callback_context()
    resp = _mk_model_response(
        [mk_function_call_part(name="no_such_tool", args={"x": 1})]
    )

    result = cb(ctx, resp)

    assert result is resp
    fc = resp.content.parts[0].function_call
    assert fc.name == "default_tool"
    assert fc.args["meta"]["func_name"] == "no_such_tool"
    assert len(cb.history) == 1


def test_invalid_tool_cb_rewrites_malformed_args():
    cb = _mk_invalid_tool_cb()
    ctx = mk_callback_context()
    part = mk_function_call_part(name="read_file")
    part.function_call.args = "not-a-dict"  # type: ignore[assignment]
    resp = _mk_model_response([part])

    result = cb(ctx, resp)

    assert result is resp
    fc = resp.content.parts[0].function_call
    assert fc.name == "default_tool"
    assert "error" in fc.args["meta"]


def test_chain_runs_downstream_callback_when_nothing_modified():
    class SentinelCallback(MandatoryToolCallback):
        """Downstream after_model callback that records it was reached."""

        def __init__(self):
            super().__init__(tool_names=["submit_verdict"])
            self.seen = 0

        def __call__(self, callback_context, llm_response):
            self.seen += 1
            return super().__call__(callback_context, llm_response)

    sentinel = SentinelCallback()
    adapter = CallbackAdapter(agent_name="worker")
    adapter.register(_mk_invalid_tool_cb())
    adapter.register(sentinel)
    chain = adapter()["after_model_callback"]

    ctx = mk_callback_context()
    resp = _mk_model_response(
        [mk_function_call_part(name="read_file", args={"path": "a"})]
    )

    assert chain(callback_context=ctx, llm_response=resp) is None
    assert sentinel.seen == 1


def test_exploitability_chaining_lets_mandatory_tool_callback_nudge():
    """Mirrors exploitability_agent/agent.py: the worker's after_model chain
    (ending with InvalidToolCallGuardrailCallback) is wrapped by ``_chain``,
    which only runs MandatoryToolCallback when the chain returns None."""
    adapter = CallbackAdapter(agent_name="worker")
    adapter.register(_mk_invalid_tool_cb())
    worker_chain = adapter()["after_model_callback"]

    mandatory = MandatoryToolCallback(tool_names=["submit_verdict"], max_nudges=3)

    def _chain(callback_context, llm_response):
        result = worker_chain(
            callback_context=callback_context, llm_response=llm_response
        )
        if result is not None:
            return result
        return mandatory(
            callback_context=callback_context, llm_response=llm_response
        )

    ctx = mk_callback_context()

    # turn 1: a valid tool call — tracked, no nudge
    resp = _mk_model_response(
        [mk_function_call_part(name="read_file", args={"path": "a"})]
    )
    assert _chain(ctx, resp) is None
    assert mandatory.step_count == 1

    # turn 2: text-only final answer without submit_verdict — must nudge
    final = _mk_model_response([mk_text_part("verdict: exploitable")])
    nudge = _chain(ctx, final)

    assert nudge is not None
    assert mandatory.nudge_count == 1
    assert "submit_verdict" in nudge.content.parts[0].text


def test_state_is_persisted():
    cb = RepeatedToolCallCallback(threshold=2)
    ctx = mk_tool_context()
    tool = _mk_tool("read_file")

    cb(tool, {"path": "a"}, ctx)
    cb(tool, {"path": "a"}, ctx)

    saved = ctx.state["callbacks"]["::RepeatedToolCallCallback"]
    assert saved["run_length"] == 2
    assert saved["threshold"] == 2
    assert len(saved["history"]) == 1
