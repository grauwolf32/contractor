"""Unit tests for the eval auto-skip gate helpers in ``tests/eval/conftest.py``.

The gate must only be bypassed when the ``-m`` expression actually selects the
``eval`` marker, or when ``CONTRACTOR_RUN_EVAL`` is a truthy boolean — an
unrelated ``-m "not slow"`` or ``CONTRACTOR_RUN_EVAL=0`` must NOT silently
enable the LLM-bound suite.
"""
from __future__ import annotations

from tests.eval.conftest import _markexpr_selects_eval, _run_eval_env_enabled


def test_markexpr_eval_selects():
    assert _markexpr_selects_eval("eval")
    assert _markexpr_selects_eval("eval and trace")
    assert _markexpr_selects_eval("slow or eval")
    # `not eval` matches too — harmless, pytest deselects the items itself
    assert _markexpr_selects_eval("not eval")


def test_markexpr_unrelated_does_not_select():
    assert not _markexpr_selects_eval(None)
    assert not _markexpr_selects_eval("")
    assert not _markexpr_selects_eval("not slow")
    assert not _markexpr_selects_eval("integration")
    # word-boundary match: marker names merely containing "eval" don't count
    assert not _markexpr_selects_eval("evaluation")
    assert not _markexpr_selects_eval("not pre_eval_check")


def test_run_eval_env_truthy_values():
    for v in ("1", "true", "TRUE", " yes ", "on"):
        assert _run_eval_env_enabled(v)


def test_run_eval_env_falsy_values():
    for v in (None, "", "0", "false", "no", "off", "anything-else"):
        assert not _run_eval_env_enabled(v)
