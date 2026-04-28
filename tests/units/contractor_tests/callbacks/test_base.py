from typing import Any

from contractor.callbacks.base import (
    BaseCallback,
    CallbackTypes,
    _callback_name,
)
from tests.units.contractor_tests.helpers import mk_callback_context


class _DummyCb(BaseCallback):
    cb_type: CallbackTypes = CallbackTypes.before_model_callback
    deps: list[str] = []

    def __init__(self, value: int = 0, agent_name: str | None = None):
        self.value = value
        self.agent_name = agent_name

    def to_state(self) -> dict[str, Any]:
        return {"value": self.value}

    def __call__(self, *args, **kwargs) -> None:
        return None


# ---------------------------------------------------------------------------
# _callback_name
# ---------------------------------------------------------------------------


def test_callback_name_uses_qualname_for_function():
    def my_cb():
        pass

    name = _callback_name(my_cb)
    assert "my_cb" in name


def test_callback_name_uses_class_name_for_lambdas():
    f = lambda: None  # noqa: E731
    # Lambdas have a __name__ ("<lambda>") so we get that.
    assert "<lambda>" == _callback_name(f) or _callback_name(f).endswith("<lambda>")


def test_callback_name_uses_class_for_callable_object():
    class Callable:
        def __call__(self):
            pass

    obj = Callable()
    # Callable instances have no __qualname__ or __name__ on the instance,
    # so fall through to class name.
    assert _callback_name(obj) == "Callable"


# ---------------------------------------------------------------------------
# BaseCallback.name and state keying
# ---------------------------------------------------------------------------


def test_name_property_returns_class_name():
    cb = _DummyCb(value=1)
    assert cb.name == "_DummyCb"


def test_state_key_uses_empty_when_agent_name_unset():
    cb = _DummyCb(value=1)
    assert cb._callback_state_key("foo") == "::foo"


def test_state_key_uses_agent_name_when_set():
    cb = _DummyCb(value=1, agent_name="planner")
    assert cb._callback_state_key("foo") == "planner::foo"


# ---------------------------------------------------------------------------
# save_to_state / get_from_cb_state
# ---------------------------------------------------------------------------


def test_save_to_state_writes_under_callbacks_namespace():
    ctx = mk_callback_context()
    cb = _DummyCb(value=42)

    cb.save_to_state(ctx)

    assert "callbacks" in ctx.state
    assert ctx.state["callbacks"]["::_DummyCb"] == {"value": 42}


def test_save_to_state_preserves_other_callback_entries():
    ctx = mk_callback_context()
    ctx.state["callbacks"] = {"::OtherCb": {"x": 1}}

    cb = _DummyCb(value=7)
    cb.save_to_state(ctx)

    assert ctx.state["callbacks"]["::OtherCb"] == {"x": 1}
    assert ctx.state["callbacks"]["::_DummyCb"] == {"value": 7}


def test_get_from_cb_state_reads_back_what_was_saved():
    ctx = mk_callback_context()
    cb = _DummyCb(value=11)
    cb.save_to_state(ctx)

    result = cb.get_from_cb_state(ctx, "_DummyCb")
    assert result == {"value": 11}


def test_get_from_cb_state_returns_none_for_missing():
    ctx = mk_callback_context()
    cb = _DummyCb()
    assert cb.get_from_cb_state(ctx, "Nonexistent") is None


def test_save_to_state_scopes_by_agent_name():
    ctx = mk_callback_context()
    cb_a = _DummyCb(value=1, agent_name="a")
    cb_b = _DummyCb(value=2, agent_name="b")

    cb_a.save_to_state(ctx)
    cb_b.save_to_state(ctx)

    assert ctx.state["callbacks"]["a::_DummyCb"] == {"value": 1}
    assert ctx.state["callbacks"]["b::_DummyCb"] == {"value": 2}


# ---------------------------------------------------------------------------
# get_invocation_id, dependencies
# ---------------------------------------------------------------------------


def test_get_invocation_id_passes_through():
    ctx = mk_callback_context(invocation_id="inv-1")
    cb = _DummyCb()
    assert cb.get_invocation_id(ctx) == "inv-1"


def test_get_dependencies_returns_deps_list():
    cb = _DummyCb()
    cb.deps = ["TokenUsageCallback"]
    assert cb.get_dependencies() == ["TokenUsageCallback"]
