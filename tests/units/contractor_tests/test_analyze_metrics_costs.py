"""Unit tests for cost estimation in ``scripts/analyze_metrics.py``.

Costs are only computed for models with a known pricing entry; rows from
unknown models (e.g. local lm-studio aliases) get no estimate and are
excluded from ``llm_with_cost`` so cost charts/tables skip them.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "analyze_metrics",
    Path(__file__).resolve().parents[3] / "scripts" / "analyze_metrics.py",
)
analyze_metrics = importlib.util.module_from_spec(_SPEC)
# Register before exec: dataclass slots resolution looks the module up in
# sys.modules while the module body runs.
sys.modules[_SPEC.name] = analyze_metrics
_SPEC.loader.exec_module(analyze_metrics)


def _row(model: str) -> pd.Series:
    return pd.Series(
        {
            "model": model,
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "cached_tokens": 0,
        }
    )


def test_known_model_gets_cost():
    cost = analyze_metrics._estimate_row_cost(_row("gemini-2.5-flash"))
    assert cost == pytest.approx(0.15 + 0.60)


def test_unknown_model_gets_no_cost():
    assert analyze_metrics._estimate_row_cost(_row("lm-studio-qwen3.6")) is None


def _llm_record(model: str | None) -> dict:
    return {
        "type": "llm_usage",
        "ts_iso": "2026-01-01T00:00:00Z",
        "model": model,
        "usage": {"input": 1000, "output": 1000, "total": 2000},
    }


def test_slices_drop_unpriced_rows():
    df = analyze_metrics.normalize_records(
        [_llm_record("gemini-2.5-flash"), _llm_record("lm-studio-qwen3.6")]
    )
    slices = analyze_metrics.MetricSlices.build(df)
    assert len(slices.llm_with_cost) == 1
    assert slices.llm_with_cost["model"].tolist() == ["gemini-2.5-flash"]


def test_slices_all_unknown_models_leave_cost_table_empty():
    df = analyze_metrics.normalize_records(
        [_llm_record("lm-studio-qwen3.6"), _llm_record(None)]
    )
    slices = analyze_metrics.MetricSlices.build(df)
    assert slices.llm_with_cost.empty

    summary = analyze_metrics.compute_summary(df, slices)
    # Rendered as "n/a" instead of a fictional $0-or-default cost.
    assert summary["estimated_total_cost"] is None
