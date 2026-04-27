from types import SimpleNamespace

from contractor.callbacks.guardrails import RepeatedToolCallCallback
from tests.units.contractor_tests.helpers import mk_tool_context


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
