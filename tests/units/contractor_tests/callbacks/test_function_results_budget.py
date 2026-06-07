"""Byte-bounded heavy-result retention (QW3/A2).

Two layers are exercised:

1. The ``FunctionResultsRemovalCallback`` budget branch directly: a small
   ``keep_budget_chars`` evicts the *oldest* heavy-tool results once the
   cumulative kept-char total is exceeded, even when the count is still
   within ``keep_last_n``; and ``keep_budget_chars=0`` reproduces pure
   count-only behaviour (nothing elided by budget).
2. The ``build_worker`` wiring: the new
   ``Settings.fs_heavy_keep_budget_chars`` / ``elide_keep_budget_chars``
   knob is threaded into the constructed callback, and its default (0) is a
   no-op (count-only).
"""

from __future__ import annotations

from contractor.callbacks.context import FunctionResultsRemovalCallback
from tests.units.contractor_tests.helpers import (
    MockContent,
    mk_callback_context,
    mk_function_response_part,
    mk_llm_request,
)


def _big_response(n_chars: int, tag: str = "x") -> dict:
    return {"data": tag * n_chars}


# ---------------------------------------------------------------------------
# Budget vs. count interaction on the real callback
# ---------------------------------------------------------------------------


def test_budget_elides_oldest_even_when_count_within_keep_last_n():
    """keep_last_n is large; the char budget is what forces eviction."""
    ctx = mk_callback_context()
    # Count axis would keep all 4; budget axis is the binding constraint.
    cb = FunctionResultsRemovalCallback(
        keep_last_n=100,
        keep_budget_chars=250,
        target_tools=["read_file"],
        deduplicate=False,
    )

    # Each response ~ len(json.dumps({"data": "?"*100})) ≈ 113 chars.
    parts = [
        mk_function_response_part(response=_big_response(100, "a"), name="read_file"),
        mk_function_response_part(response=_big_response(100, "b"), name="read_file"),
        mk_function_response_part(response=_big_response(100, "c"), name="read_file"),
        mk_function_response_part(response=_big_response(100, "d"), name="read_file"),
    ]
    request = mk_llm_request([MockContent(role="tool", parts=parts)])

    cb(ctx, request)

    # Reverse scan keeps d (always), then c (≈226 total, under 250); b would
    # push over 250 → elided; a → elided. Count never bound (4 <= 100).
    assert parts[3].function_response.response == _big_response(100, "d")
    assert parts[2].function_response.response == _big_response(100, "c")
    assert parts[1].function_response.response == {"elided": True, "tool": "read_file"}
    assert parts[0].function_response.response == {"elided": True, "tool": "read_file"}
    # The two OLDEST were evicted by budget, not count.
    assert cb.counter == 2


def test_budget_zero_is_count_only_no_op():
    """keep_budget_chars=0 → nothing elided by budget; count keeps all."""
    ctx = mk_callback_context()
    cb = FunctionResultsRemovalCallback(
        keep_last_n=100,
        keep_budget_chars=0,
        target_tools=["read_file"],
        deduplicate=False,
    )

    parts = [
        mk_function_response_part(response=_big_response(100, "a"), name="read_file"),
        mk_function_response_part(response=_big_response(100, "b"), name="read_file"),
        mk_function_response_part(response=_big_response(100, "c"), name="read_file"),
        mk_function_response_part(response=_big_response(100, "d"), name="read_file"),
    ]
    request = mk_llm_request([MockContent(role="tool", parts=parts)])

    cb(ctx, request)

    # Budget disabled, count cap not reached → everything retained verbatim.
    assert parts[0].function_response.response == _big_response(100, "a")
    assert parts[1].function_response.response == _big_response(100, "b")
    assert parts[2].function_response.response == _big_response(100, "c")
    assert parts[3].function_response.response == _big_response(100, "d")
    assert cb.counter == 0


# ---------------------------------------------------------------------------
# build_worker wiring
# ---------------------------------------------------------------------------


def _capture_build_worker(monkeypatch):
    """Patch the heavy LlmAgent + capture the FunctionResultsRemovalCallback.

    Returns a dict that, after build_worker runs, holds the
    ``keep_budget_chars`` / ``keep_last_n`` the callback was constructed with.
    """
    import contractor.agents.worker_factory as wf

    captured: dict = {}
    real_cls = wf.FunctionResultsRemovalCallback

    def _spy(*args, **kwargs):
        cb = real_cls(*args, **kwargs)
        captured["keep_budget_chars"] = cb.keep_budget_chars
        captured["keep_last_n"] = cb.keep_last_n
        return cb

    monkeypatch.setattr(wf, "FunctionResultsRemovalCallback", _spy)
    # Stub out LlmAgent so we don't need the model/ADK machinery.
    monkeypatch.setattr(wf, "LlmAgent", lambda **kw: kw)
    return captured


def default_tool():  # noqa: D401 - guardrail requires a tool named "default_tool"
    """Placeholder tool the InvalidToolCallGuardrail falls back to."""


def _build(wf_module, **overrides):
    kwargs = {
        "name": "spy_worker",
        "instruction": "do things",
        "description": "spy",
        "tools": [default_tool],
        "_format": "json",
        "summarization_bullets": "You have reached the context limit.\n1. x\n",
        "elide_tool_results": ["read_file"],
    }
    kwargs.update(overrides)
    return wf_module.build_worker(**kwargs)


def test_build_worker_defaults_budget_to_settings(monkeypatch):
    import contractor.agents.worker_factory as wf
    from contractor.utils.settings import Settings

    captured = _capture_build_worker(monkeypatch)
    # Isolate from ambient env so we assert the field default, not a stray
    # FS_HEAVY_KEEP_BUDGET_CHARS in the environment.
    monkeypatch.delenv("FS_HEAVY_KEEP_BUDGET_CHARS", raising=False)
    monkeypatch.setattr(wf, "get_settings", lambda: Settings())
    # Default Settings.fs_heavy_keep_budget_chars is 0 → no-op (count-only).
    _build(wf)

    assert captured["keep_last_n"] == 15
    assert captured["keep_budget_chars"] == 0


def test_build_worker_reads_settings_override(monkeypatch):
    import contractor.agents.worker_factory as wf
    from contractor.utils.settings import Settings

    captured = _capture_build_worker(monkeypatch)

    # Simulate FS_HEAVY_KEEP_BUDGET_CHARS=120000 via the settings object.
    monkeypatch.setattr(
        wf, "get_settings", lambda: Settings(fs_heavy_keep_budget_chars=120_000)
    )
    _build(wf)

    assert captured["keep_last_n"] == 15
    assert captured["keep_budget_chars"] == 120_000


def test_build_worker_explicit_arg_overrides_settings(monkeypatch):
    import contractor.agents.worker_factory as wf
    from contractor.utils.settings import Settings

    captured = _capture_build_worker(monkeypatch)
    # Settings says one thing; explicit kwarg must win.
    monkeypatch.setattr(
        wf, "get_settings", lambda: Settings(fs_heavy_keep_budget_chars=999)
    )
    _build(wf, elide_keep_budget_chars=50_000)

    assert captured["keep_budget_chars"] == 50_000
