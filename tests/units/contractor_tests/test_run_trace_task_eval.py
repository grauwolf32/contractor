"""Unit tests for ``scripts/run_trace_task_eval._score`` — the trace_annotation
task eval's annotation scorer with partial-label (GT-scoped) precision."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "rtte",
    Path(__file__).resolve().parents[3] / "scripts" / "run_trace_task_eval.py",
)
rtte = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rtte)


class _Ann:
    """Minimal stand-in for trace_harness.Annotation (file/function + as_tuple)."""

    def __init__(self, file: str, function: str):
        self.file, self.function = file, function

    def as_tuple(self):
        return (self.file, self.function)


def _score(actual, expected):
    return rtte._score({_Ann(*a) for a in actual}, {_Ann(*e) for e in expected})


def test_precision_scoped_to_labeled_files():
    # GT labels 2 funcs in 1 file. The run also annotates a second, un-labeled
    # file (a different operation) — those must NOT count against precision.
    expected = [("/src/a.ts", "f1"), ("/src/a.ts", "f2")]
    actual = [
        ("/src/a.ts", "f1"),        # match
        ("/src/a.ts", "wrong"),     # in-scope FP (labeled file, wrong func)
        ("/src/b.ts", "x"),         # out-of-scope (un-labeled file) — ignored
        ("/src/b.ts", "y"),         # out-of-scope — ignored
    ]
    s = _score(actual, expected)
    assert s["matched"] == 1
    assert s["recall"] == 0.5                       # 1 / 2 expected
    assert s["scored_actual"] == 2 and s["out_of_scope"] == 2
    assert s["precision"] == 0.5                    # 1 / 2 in-scope (a.ts only)
    assert s["precision_raw"] == 0.25               # 1 / 4 over all annotations
    # extra = in-scope FPs only; out-of-scope annotations are not penalized
    assert s["extra"] == ["/src/a.ts::wrong"]


def test_empty_expected_is_neutral():
    s = _score([("/src/a.ts", "f")], [])
    assert s["recall"] == 0.0 and s["out_of_scope"] == 1


def test_perfect_in_scope_with_unlabeled_noise_is_f1_one():
    # All labeled funcs matched; extra annotations only on un-labeled files.
    expected = [("/src/a.ts", "f1")]
    actual = [("/src/a.ts", "f1"), ("/src/other.ts", "z")]
    s = _score(actual, expected)
    assert s["precision"] == 1.0 and s["recall"] == 1.0 and s["f1"] == 1.0
    assert s["precision_raw"] == 0.5                # raw still sees the noise
