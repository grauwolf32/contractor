#!/usr/bin/env python3
"""Analyze contractor metrics.jsonl and produce charts, tables, and a report."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

_TOKEN_COLS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "thoughts_tokens",
    "cached_tokens",
)

_METRIC_EVENT_TYPES = {
    "tool_call": "metrics_tool_call",
    "tool_result": "metrics_tool_result",
    "tool_exc": "metrics_tool_exception_error",
    "llm": "metrics_llm_usage",
    "summary": "metrics_summary",
    "before_run": "adk_before_run",
    "after_run": "adk_after_run",
    "adk_event": "adk_event",
}

_FALLBACKS = {
    "task": "unknown_task",
    "agent": "unknown_agent",
    "tool": "unknown_tool",
    "type": "unknown_type",
    "model": "unknown_model",
    "invocation": "unknown_invocation",
    "session": "unknown_session",
    "error": "unknown_error",
}

_MAX_HEATMAP_ROWS = 12
_MAX_HEATMAP_COLS = 15
_MIN_CALLS_FOR_ERROR_RATE = 3
_TOP_N_DEFAULT = 12

# Approximate pricing per 1M tokens — adjust to actual rates
_PRICE_PER_M_TOKENS: dict[str, dict[str, float]] = {
    "gemini-2.5-pro": {"prompt": 1.25, "completion": 10.00, "cached": 0.3125},
    "gemini-2.5-flash": {"prompt": 0.15, "completion": 0.60, "cached": 0.0375},
    "gemini-2.0-flash": {"prompt": 0.10, "completion": 0.40, "cached": 0.025},
}
_DEFAULT_PRICE = {"prompt": 1.00, "completion": 3.00, "cached": 0.25}


# ─── Output paths ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OutputPaths:
    root: Path
    charts: Path
    tables: Path

    @classmethod
    def from_args(cls, input_file: Path, output_dir: Path | None) -> OutputPaths:
        input_file = input_file.resolve()
        root = (
            output_dir.resolve()
            if output_dir
            else input_file.parent / "metrics_report"
        )
        charts = root / "charts"
        tables = root / "tables"
        charts.mkdir(parents=True, exist_ok=True)
        tables.mkdir(parents=True, exist_ok=True)
        return cls(root=root, charts=charts, tables=tables)


# ─── Data loading & normalisation ─────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value) if not isinstance(value, str) else value


def _safe_dict(mapping: Any) -> dict[str, Any]:
    return mapping if isinstance(mapping, dict) else {}


def normalize_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    norm: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        payload = _safe_dict(rec.get("payload"))
        usage = _safe_dict(payload.get("usage"))

        norm.append(
            {
                "row_id": idx,
                "ts": pd.to_datetime(rec.get("ts"), utc=True, errors="coerce"),
                "type": _opt_str(rec.get("type")),
                "task_name": _opt_str(rec.get("task_name")),
                "task_id": rec.get("task_id"),
                "iteration": rec.get("iteration", payload.get("iteration")),
                "session_id": _opt_str(
                    rec.get("session_id", payload.get("session_id"))
                ),
                "invocation_id": _opt_str(
                    rec.get("invocation_id", payload.get("invocation_id"))
                ),
                "agent_name": _opt_str(
                    rec.get("agent_name", payload.get("agent_name"))
                ),
                "tool_name": _opt_str(
                    rec.get("tool_name", payload.get("tool_name"))
                ),
                "author": _opt_str(rec.get("author", payload.get("author"))),
                "result_error": bool(payload.get("result_error", False)),
                "error": _opt_str(
                    payload.get("error") or payload.get("error_message")
                ),
                "model": _opt_str(payload.get("model")),
                **{
                    col: pd.to_numeric(
                        usage.get(col.removesuffix("_tokens")), errors="coerce"
                    )
                    for col in _TOKEN_COLS
                },
                "tool_args": _safe_dict(payload.get("tool_args")),
                "result": payload.get("result"),
                "payload": payload,
                "raw": rec,
            }
        )

    df = pd.DataFrame(norm)
    if df.empty:
        return df

    for col in ("task_id", "iteration"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in _TOKEN_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if df["ts"].notna().any():
        df = df.sort_values(["ts", "row_id"], kind="stable").reset_index(drop=True)

    return df


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _fill_label(series: pd.Series, fallback: str) -> pd.Series:
    out = series.fillna(fallback).astype(str)
    return out.replace({"nan": fallback, "None": fallback, "": fallback})


def _top_n(series: pd.Series, n: int = _TOP_N_DEFAULT) -> pd.Series:
    s = series.sort_values(ascending=False)
    if len(s) <= n:
        return s
    head = s.iloc[:n].copy()
    head.loc["other"] = s.iloc[n:].sum()
    return head


def _args_hash(args: Any) -> str:
    try:
        raw = json.dumps(args, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:
        raw = repr(args)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _estimate_row_cost(row: pd.Series) -> float:
    model = str(row.get("model", ""))
    prices = _PRICE_PER_M_TOKENS.get(model, _DEFAULT_PRICE)
    return (
        row.get("prompt_tokens", 0) * prices["prompt"]
        + row.get("completion_tokens", 0) * prices["completion"]
        + row.get("cached_tokens", 0) * prices.get("cached", 0)
    ) / 1_000_000


# ─── Pre-sliced views of the dataframe ───────────────────────────────────────


@dataclass
class MetricSlices:
    """Pre-filtered views so each chart doesn't re-filter the full frame."""

    all: pd.DataFrame
    tool_calls: pd.DataFrame
    tool_results: pd.DataFrame
    tool_exc: pd.DataFrame
    llm: pd.DataFrame
    summaries: pd.DataFrame
    before_runs: pd.DataFrame
    after_runs: pd.DataFrame
    adk_events: pd.DataFrame

    # Aggregated tables (computed once, used by many charts)
    tools_by_calls: pd.Series = field(
        default_factory=lambda: pd.Series(dtype="int64")
    )
    err_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    tokens_by_agent: pd.DataFrame = field(default_factory=pd.DataFrame)
    tokens_by_task: pd.DataFrame = field(default_factory=pd.DataFrame)
    calls_by_agent_tool: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Duration tables
    tool_durations: pd.DataFrame = field(default_factory=pd.DataFrame)
    invocation_durations: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Cost table
    llm_with_cost: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Retry detection
    tool_calls_with_retries: pd.DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def build(cls, df: pd.DataFrame) -> MetricSlices:
        df = df.copy()
        for raw, label in [
            ("task_name", "task"),
            ("agent_name", "agent"),
            ("tool_name", "tool"),
            ("type", "type"),
            ("invocation_id", "invocation"),
            ("session_id", "session"),
        ]:
            df[f"{raw}_f"] = _fill_label(df[raw], _FALLBACKS[label])

        def _where(event_key: str) -> pd.DataFrame:
            etype = _METRIC_EVENT_TYPES.get(event_key)
            if etype is None:
                return pd.DataFrame()
            return df[df["type_f"] == etype].copy()

        slices = cls(
            all=df,
            tool_calls=_where("tool_call"),
            tool_results=_where("tool_result"),
            tool_exc=_where("tool_exc"),
            llm=_where("llm"),
            summaries=_where("summary"),
            before_runs=_where("before_run"),
            after_runs=_where("after_run"),
            adk_events=_where("adk_event"),
        )
        slices._compute_aggregates()
        slices._compute_durations()
        slices._compute_costs()
        slices._compute_retries()
        return slices

    def _compute_aggregates(self) -> None:
        tc, tr, te, llm = (
            self.tool_calls,
            self.tool_results,
            self.tool_exc,
            self.llm,
        )

        if not tc.empty:
            self.tools_by_calls = (
                tc.groupby("tool_name_f")
                .size()
                .sort_values(ascending=False)
                .rename("calls")
            )

        self.err_table = self._build_error_table(tc, tr, te)

        if not llm.empty:
            self.tokens_by_agent = (
                llm.groupby("agent_name_f")[list(_TOKEN_COLS)]
                .sum()
                .sort_values("total_tokens", ascending=False)
            )
            self.tokens_by_task = (
                llm.groupby("task_name_f")[
                    ["prompt_tokens", "completion_tokens", "total_tokens"]
                ]
                .sum()
                .sort_values("total_tokens", ascending=False)
            )

        if not tc.empty:
            self.calls_by_agent_tool = tc.pivot_table(
                index="agent_name_f",
                columns="tool_name_f",
                values="row_id",
                aggfunc="count",
                fill_value=0,
            )

    def _compute_durations(self) -> None:
        tc = self.tool_calls
        tr = self.tool_results

        # Tool call durations: pair before_tool → after_tool
        if (
            not tc.empty
            and not tr.empty
            and tc["ts"].notna().any()
            and tr["ts"].notna().any()
        ):
            starts = tc[tc["ts"].notna()][
                ["ts", "invocation_id_f", "agent_name_f", "tool_name_f"]
            ].copy()
            ends = tr[tr["ts"].notna()][
                ["ts", "invocation_id_f", "agent_name_f", "tool_name_f"]
            ].copy()

            key_cols = ["invocation_id_f", "agent_name_f", "tool_name_f"]
            starts["_seq"] = starts.sort_values("ts").groupby(key_cols).cumcount()
            ends["_seq"] = ends.sort_values("ts").groupby(key_cols).cumcount()

            merged = starts.merge(
                ends, on=key_cols + ["_seq"], suffixes=("_start", "_end")
            )
            merged["duration_s"] = (
                merged["ts_end"] - merged["ts_start"]
            ).dt.total_seconds()
            self.tool_durations = merged[merged["duration_s"] >= 0].copy()

        # Invocation durations: before_run → after_run
        br, ar = self.before_runs, self.after_runs
        if (
            not br.empty
            and not ar.empty
            and br["ts"].notna().any()
            and ar["ts"].notna().any()
        ):
            br_ts = (
                br[br["ts"].notna()]
                .groupby("invocation_id_f")["ts"]
                .min()
                .rename("start_ts")
            )
            ar_ts = (
                ar[ar["ts"].notna()]
                .groupby("invocation_id_f")["ts"]
                .max()
                .rename("end_ts")
            )
            inv_dur = pd.concat([br_ts, ar_ts], axis=1).dropna()
            inv_dur["duration_s"] = (
                inv_dur["end_ts"] - inv_dur["start_ts"]
            ).dt.total_seconds()
            self.invocation_durations = inv_dur[inv_dur["duration_s"] >= 0].copy()

    def _compute_costs(self) -> None:
        if self.llm.empty:
            return
        llm = self.llm.copy()
        llm["estimated_cost"] = llm.apply(_estimate_row_cost, axis=1)
        self.llm_with_cost = llm

    def _compute_retries(self) -> None:
        tc = self.tool_calls
        if tc.empty:
            return

        tc = tc.copy()
        tc["args_hash"] = tc["tool_args"].apply(_args_hash)
        tc = tc.sort_values(["invocation_id_f", "ts", "row_id"])
        tc["prev_tool"] = tc.groupby("invocation_id_f")["tool_name_f"].shift(1)
        tc["prev_hash"] = tc.groupby("invocation_id_f")["args_hash"].shift(1)
        tc["is_retry"] = (tc["tool_name_f"] == tc["prev_tool"]) & (
            tc["args_hash"] == tc["prev_hash"]
        )
        self.tool_calls_with_retries = tc

    @staticmethod
    def _build_error_table(
        tc: pd.DataFrame, tr: pd.DataFrame, te: pd.DataFrame
    ) -> pd.DataFrame:
        if tc.empty and tr.empty and te.empty:
            return pd.DataFrame(
                columns=[
                    "calls",
                    "result_errors",
                    "exception_errors",
                    "total_errors",
                    "error_rate",
                ]
            )

        parts = {}
        if not tc.empty:
            parts["calls"] = tc.groupby("tool_name_f").size().rename("calls")
        if not tr.empty:
            parts["result_errors"] = (
                tr.groupby("tool_name_f")["result_error"]
                .sum()
                .rename("result_errors")
            )
        if not te.empty:
            parts["exception_errors"] = (
                te.groupby("tool_name_f").size().rename("exception_errors")
            )

        tbl = (
            pd.concat(parts.values(), axis=1)
            .reindex(
                columns=["calls", "result_errors", "exception_errors"], fill_value=0
            )
            .fillna(0)
        )
        for col in ("calls", "result_errors", "exception_errors"):
            tbl[col] = pd.to_numeric(tbl[col], errors="coerce").fillna(0)

        tbl["total_errors"] = tbl["result_errors"] + tbl["exception_errors"]
        tbl["error_rate"] = tbl["total_errors"] / tbl["calls"].replace(0, pd.NA)
        return tbl.fillna(0).sort_values(["total_errors", "calls"], ascending=False)


# ─── Plotting primitives ─────────────────────────────────────────────────────


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _trim_heatmap(
    matrix: pd.DataFrame,
    max_rows: int = _MAX_HEATMAP_ROWS,
    max_cols: int = _MAX_HEATMAP_COLS,
) -> pd.DataFrame:
    if matrix.shape[0] > max_rows:
        keep = matrix.sum(axis=1).sort_values(ascending=False).head(max_rows).index
        matrix = matrix.loc[keep]
    if matrix.shape[1] > max_cols:
        keep = matrix.sum(axis=0).sort_values(ascending=False).head(max_cols).index
        matrix = matrix[keep]
    return matrix


def _bar(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    *,
    horizontal: bool = False,
    top_n: int = _TOP_N_DEFAULT,
) -> None:
    if series.empty:
        return
    series = _top_n(series, top_n)
    h = max(4, min(12, 0.45 * len(series) + 2)) if horizontal else 6
    fig, ax = plt.subplots(figsize=(10, h))
    if horizontal:
        series = series.sort_values()
        ax.barh(series.index.astype(str), series.values)
    else:
        ax.bar(series.index.astype(str), series.values)
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save_fig(fig, path)


def _stacked_bar(
    df: pd.DataFrame,
    layers: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    *,
    max_items: int = 10,
    sort_by: str | None = None,
) -> None:
    if df.empty:
        return
    sort_col = sort_by or layers[0]
    if sort_col not in df.columns:
        sort_col = df.columns[0]
    top = df.sort_values(sort_col, ascending=False).head(max_items)
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = None
    for col in layers:
        if col not in top.columns:
            continue
        vals = top[col].fillna(0)
        ax.bar(top.index.astype(str), vals, bottom=bottom, label=col)
        bottom = vals if bottom is None else bottom + vals
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    _save_fig(fig, path)


def _line(
    df: pd.DataFrame,
    x: str,
    y: str | list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    if df.empty:
        return
    cols = [y] if isinstance(y, str) else y
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for col in cols:
        if col in df.columns:
            ax.plot(df[x], df[col], label=col if len(cols) > 1 else None)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(cols) > 1:
        ax.legend()
    ax.tick_params(axis="x", rotation=25)
    _save_fig(fig, path)


def _hist(
    series: pd.Series,
    bins: int,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(s, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save_fig(fig, path)


def _scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    *,
    label_col: str | None = None,
    max_labels: int = 40,
    color_col: str | None = None,
) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    if color_col and color_col in df.columns:
        categories = df[color_col].unique()
        cmap = plt.cm.get_cmap("tab10", len(categories))
        for i, cat in enumerate(categories):
            sub = df[df[color_col] == cat]
            ax.scatter(sub[x], sub[y], alpha=0.7, label=str(cat), color=cmap(i))
        ax.legend(fontsize=7, loc="best")
    else:
        ax.scatter(df[x], df[y], alpha=0.7)
    if label_col and label_col in df.columns and len(df) <= max_labels:
        for _, row in df.iterrows():
            ax.annotate(str(row[label_col]), (row[x], row[y]), fontsize=7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save_fig(fig, path)


def _heatmap(
    matrix: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    matrix = _trim_heatmap(matrix)
    if matrix.empty:
        return
    w = max(8, min(18, 0.7 * matrix.shape[1] + 3))
    h = max(5, min(16, 0.5 * matrix.shape[0] + 2))
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(matrix.values, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(
        [str(c) for c in matrix.columns], rotation=45, ha="right"
    )
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([str(i) for i in matrix.index])
    fig.colorbar(im, ax=ax)
    _save_fig(fig, path)


def _boxplot(
    groups: dict[str, pd.Series],
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    clean = {
        k: pd.to_numeric(v, errors="coerce").dropna() for k, v in groups.items()
    }
    clean = {k: v for k, v in clean.items() if not v.empty}
    if not clean:
        return
    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(clean)), 6))
    ax.boxplot(
        [v.values for v in clean.values()],
        tick_labels=list(clean.keys()),
        vert=True,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    _save_fig(fig, path)


def _pie(
    series: pd.Series,
    title: str,
    path: Path,
    *,
    max_slices: int = 8,
) -> None:
    top = series.sort_values(ascending=False).head(max_slices)
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(top.values, labels=top.index.astype(str), autopct="%1.1f%%")
    ax.set_title(title)
    _save_fig(fig, path)


# ─── Summary computation ─────────────────────────────────────────────────────


def compute_summary(df: pd.DataFrame, slices: MetricSlices | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rows": len(df),
        "date_min": None,
        "date_max": None,
        "event_types": {},
        "tool_calls": 0,
        "tool_result_errors": 0,
        "tool_exception_errors": 0,
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "agents": 0,
        "tools": 0,
        "tasks": 0,
        "estimated_total_cost": 0.0,
        "retry_count": 0,
        "avg_invocation_duration_s": None,
        "avg_tool_duration_s": None,
        "cache_hit_rate": None,
        "thinking_overhead": None,
        "useful_work_ratio": None,
    }
    if df.empty:
        return out

    if df["ts"].notna().any():
        out["date_min"] = df["ts"].min().isoformat()
        out["date_max"] = df["ts"].max().isoformat()

    out["event_types"] = df["type"].fillna("unknown").value_counts().to_dict()

    masks = {k: df["type"].eq(v) for k, v in _METRIC_EVENT_TYPES.items()}

    out["tool_calls"] = int(masks.get("tool_call", pd.Series(dtype=bool)).sum())
    out["tool_result_errors"] = int(
        (masks.get("tool_result", pd.Series(dtype=bool)) & df["result_error"].fillna(False)).sum()
    )
    out["tool_exception_errors"] = int(masks.get("tool_exc", pd.Series(dtype=bool)).sum())
    out["llm_calls"] = int(masks.get("llm", pd.Series(dtype=bool)).sum())

    llm_mask = masks.get("llm", pd.Series(dtype=bool))
    llm_rows = df.loc[llm_mask]
    for col in ("prompt_tokens", "completion_tokens", "total_tokens"):
        out[col] = int(llm_rows[col].sum())

    out["agents"] = int(df["agent_name"].dropna().nunique())
    out["tools"] = int(df["tool_name"].dropna().nunique())
    out["tasks"] = int(df["task_name"].dropna().nunique())

    # Cost
    if slices is not None and not slices.llm_with_cost.empty:
        out["estimated_total_cost"] = round(
            float(slices.llm_with_cost["estimated_cost"].sum()), 6
        )

    # Retries
    if slices is not None and not slices.tool_calls_with_retries.empty:
        out["retry_count"] = int(
            slices.tool_calls_with_retries["is_retry"].sum()
        )

    # Durations
    if slices is not None and not slices.invocation_durations.empty:
        out["avg_invocation_duration_s"] = round(
            float(slices.invocation_durations["duration_s"].mean()), 2
        )
    if slices is not None and not slices.tool_durations.empty:
        out["avg_tool_duration_s"] = round(
            float(slices.tool_durations["duration_s"].mean()), 4
        )

    # Efficiency ratios
    prompt_total = float(llm_rows["prompt_tokens"].sum())
    cached_total = float(llm_rows["cached_tokens"].sum())
    total_total = float(llm_rows["total_tokens"].sum())
    thoughts_total = float(llm_rows["thoughts_tokens"].sum())

    if prompt_total > 0:
        out["cache_hit_rate"] = round(cached_total / prompt_total, 4)
    if total_total > 0:
        out["thinking_overhead"] = round(thoughts_total / total_total, 4)

    tool_total = out["tool_calls"]
    err_total = out["tool_result_errors"] + out["tool_exception_errors"]
    if tool_total > 0:
        out["useful_work_ratio"] = round((tool_total - err_total) / tool_total, 4)

    return out


# ─── Markdown report ──────────────────────────────────────────────────────────


def write_markdown_report(
    df: pd.DataFrame, summary: dict[str, Any], path: Path
) -> None:
    def _fmt(key: str, label: str, fmt: str = "d") -> str:
        val = summary.get(key)
        if val is None:
            return f"- **{label}:** n/a"
        if fmt == "d":
            return f"- **{label}:** {int(val):,}"
        if fmt == ",":
            return f"- **{label}:** {int(val):,}"
        if fmt == "f":
            return f"- **{label}:** {val:,.4f}"
        if fmt == "f2":
            return f"- **{label}:** {val:,.2f}"
        if fmt == "$":
            return f"- **{label}:** ${val:,.6f}"
        if fmt == "%":
            return f"- **{label}:** {val:.2%}"
        return f"- **{label}:** {val}"

    lines = [
        "# Metrics Report",
        "",
        "## Summary",
        "",
        f"- **Rows:** {summary['rows']:,}",
        f"- **Time range:** {summary['date_min'] or 'n/a'} → {summary['date_max'] or 'n/a'}",
        "",
        "### Tool Metrics",
        "",
        _fmt("tool_calls", "Tool calls"),
        _fmt("tool_result_errors", "Tool result errors"),
        _fmt("tool_exception_errors", "Tool exception errors"),
        _fmt("retry_count", "Detected retries"),
        _fmt("useful_work_ratio", "Useful work ratio", "%"),
        "",
        "### LLM Metrics",
        "",
        _fmt("llm_calls", "LLM calls"),
        _fmt("prompt_tokens", "Prompt tokens", ","),
        _fmt("completion_tokens", "Completion tokens", ","),
        _fmt("total_tokens", "Total tokens", ","),
        _fmt("cache_hit_rate", "Cache hit rate", "%"),
        _fmt("thinking_overhead", "Thinking overhead", "%"),
        _fmt("estimated_total_cost", "Estimated total cost", "$"),
        "",
        "### Duration",
        "",
        _fmt("avg_invocation_duration_s", "Avg invocation duration (s)", "f2"),
        _fmt("avg_tool_duration_s", "Avg tool call duration (s)", "f"),
        "",
        "### Scale",
        "",
        _fmt("agents", "Distinct agents"),
        _fmt("tools", "Distinct tools"),
        _fmt("tasks", "Distinct tasks"),
        "",
    ]

    if not df.empty:
        tc_mask = df["type"].eq(_METRIC_EVENT_TYPES["tool_call"])
        llm_mask = df["type"].eq(_METRIC_EVENT_TYPES["llm"])

        top_tools = (
            _fill_label(df.loc[tc_mask, "tool_name"], _FALLBACKS["tool"])
            .value_counts()
            .head(10)
        )
        lines += ["## Top Tools by Calls", ""]
        lines += [f"- {tool}: {int(n):,}" for tool, n in top_tools.items()]
        lines.append("")

        top_agents = (
            df.loc[llm_mask]
            .assign(
                agent=_fill_label(
                    df.loc[llm_mask, "agent_name"], _FALLBACKS["agent"]
                )
            )
            .groupby("agent", dropna=False)["total_tokens"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        lines += ["## Top Agents by Total Tokens", ""]
        lines += [
            f"- {agent}: {int(n):,}" for agent, n in top_agents.items()
        ]
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ─── Chart functions ──────────────────────────────────────────────────────────
#
# Each chart is a small function: (MetricSlices, OutputPaths) → None
# Collected into _CHART_PIPELINE for isolated execution.


def _chart_event_counts(s: MetricSlices, p: OutputPaths) -> None:
    _bar(
        s.all["type_f"].value_counts(),
        "Event counts by type",
        "Event type",
        "Count",
        p.charts / "01_event_counts_by_type.png",
    )


def _chart_tool_calls_by_tool(s: MetricSlices, p: OutputPaths) -> None:
    if s.tool_calls.empty:
        return
    _bar(
        s.tool_calls["tool_name_f"].value_counts(),
        "Tool calls by tool",
        "Tool",
        "Calls",
        p.charts / "02_tool_calls_by_tool.png",
        horizontal=True,
        top_n=15,
    )


def _chart_tool_calls_by_agent(s: MetricSlices, p: OutputPaths) -> None:
    if s.tool_calls.empty:
        return
    _bar(
        s.tool_calls["agent_name_f"].value_counts(),
        "Tool calls by agent",
        "Agent",
        "Calls",
        p.charts / "03_tool_calls_by_agent.png",
        horizontal=True,
    )


def _chart_llm_calls_by_agent(s: MetricSlices, p: OutputPaths) -> None:
    if s.llm.empty:
        return
    _bar(
        s.llm["agent_name_f"].value_counts(),
        "LLM calls by agent",
        "Agent",
        "Calls",
        p.charts / "04_llm_calls_by_agent.png",
        horizontal=True,
    )


def _chart_total_tokens_by_agent(s: MetricSlices, p: OutputPaths) -> None:
    if s.tokens_by_agent.empty:
        return
    _bar(
        s.tokens_by_agent["total_tokens"],
        "Total tokens by agent",
        "Agent",
        "Tokens",
        p.charts / "05_total_tokens_by_agent.png",
        horizontal=True,
    )


def _chart_prompt_vs_completion_by_agent(
    s: MetricSlices, p: OutputPaths
) -> None:
    if s.tokens_by_agent.empty:
        return
    _stacked_bar(
        s.tokens_by_agent,
        ["prompt_tokens", "completion_tokens"],
        "Prompt vs completion tokens by agent",
        "Agent",
        "Tokens",
        p.charts / "06_prompt_vs_completion_by_agent.png",
        sort_by="total_tokens",
    )


def _chart_total_tokens_by_task(s: MetricSlices, p: OutputPaths) -> None:
    if s.tokens_by_task.empty:
        return
    _bar(
        s.tokens_by_task["total_tokens"],
        "Total tokens by task",
        "Task",
        "Tokens",
        p.charts / "07_total_tokens_by_task.png",
        horizontal=True,
    )


def _chart_token_histogram(s: MetricSlices, p: OutputPaths) -> None:
    if s.llm.empty:
        return
    _hist(
        s.llm["total_tokens"],
        25,
        "Distribution of total tokens per LLM call",
        "Total tokens",
        "Calls",
        p.charts / "08_total_tokens_hist.png",
    )


def _chart_prompt_completion_scatter(s: MetricSlices, p: OutputPaths) -> None:
    if s.llm.empty:
        return
    _scatter(
        s.llm[["prompt_tokens", "completion_tokens", "agent_name_f"]].copy(),
        "prompt_tokens",
        "completion_tokens",
        "Prompt vs completion tokens per LLM call",
        "Prompt tokens",
        "Completion tokens",
        p.charts / "09_prompt_vs_completion_scatter.png",
        color_col="agent_name_f",
    )


def _chart_cumulative_tokens(s: MetricSlices, p: OutputPaths) -> None:
    if s.llm.empty or not s.llm["ts"].notna().any():
        return
    ts = (
        s.llm[["ts", "prompt_tokens", "completion_tokens", "total_tokens"]]
        .sort_values("ts")
        .copy()
    )
    for col in ("prompt_tokens", "completion_tokens", "total_tokens"):
        ts[f"cum_{col}"] = ts[col].cumsum()
    _line(
        ts,
        "ts",
        [f"cum_{c}" for c in ("prompt_tokens", "completion_tokens", "total_tokens")],
        "Cumulative tokens over time",
        "Time",
        "Tokens",
        p.charts / "10_cumulative_tokens_over_time.png",
    )


def _chart_cumulative_tool_calls_errors(
    s: MetricSlices, p: OutputPaths
) -> None:
    if not s.all["ts"].notna().any():
        return
    t = (
        s.all[["ts", "type_f", "result_error"]]
        .dropna(subset=["ts"])
        .sort_values("ts")
        .copy()
    )
    t["cum_tool_calls"] = (
        (t["type_f"] == _METRIC_EVENT_TYPES["tool_call"]).astype(int).cumsum()
    )
    t["cum_tool_exc_errors"] = (
        (t["type_f"] == _METRIC_EVENT_TYPES["tool_exc"]).astype(int).cumsum()
    )
    t["cum_tool_result_errors"] = (
        (
            (t["type_f"] == _METRIC_EVENT_TYPES["tool_result"])
            & t["result_error"].fillna(False)
        )
        .astype(int)
        .cumsum()
    )
    _line(
        t,
        "ts",
        ["cum_tool_calls", "cum_tool_exc_errors", "cum_tool_result_errors"],
        "Cumulative tool calls and errors over time",
        "Time",
        "Count",
        p.charts / "11_cumulative_tool_calls_errors.png",
    )


def _chart_errors_by_tool(s: MetricSlices, p: OutputPaths) -> None:
    if s.err_table.empty:
        return
    _bar(
        s.err_table["total_errors"],
        "Total errors by tool",
        "Tool",
        "Errors",
        p.charts / "12_errors_by_tool.png",
        horizontal=True,
        top_n=15,
    )


def _chart_error_rate_by_tool(s: MetricSlices, p: OutputPaths) -> None:
    if s.err_table.empty:
        return
    filtered = s.err_table[
        s.err_table["calls"] >= _MIN_CALLS_FOR_ERROR_RATE
    ].sort_values("error_rate", ascending=False)
    if filtered.empty:
        return
    _bar(
        filtered["error_rate"],
        f"Tool error rate (min {_MIN_CALLS_FOR_ERROR_RATE} calls)",
        "Tool",
        "Error rate",
        p.charts / "13_error_rate_by_tool.png",
        horizontal=True,
        top_n=15,
    )


def _chart_result_vs_exception_errors(
    s: MetricSlices, p: OutputPaths
) -> None:
    if s.err_table.empty:
        return
    _stacked_bar(
        s.err_table,
        ["result_errors", "exception_errors"],
        "Result vs exception errors by tool",
        "Tool",
        "Errors",
        p.charts / "14_result_vs_exception_errors_by_tool.png",
        sort_by="total_errors",
    )


def _chart_calls_heatmap(s: MetricSlices, p: OutputPaths) -> None:
    if s.calls_by_agent_tool.empty:
        return
    _heatmap(
        s.calls_by_agent_tool,
        "Tool calls heatmap: agent × tool",
        "Tool",
        "Agent",
        p.charts / "15_calls_heatmap_agent_tool.png",
    )


def _chart_error_heatmap(s: MetricSlices, p: OutputPaths) -> None:
    tr, te = s.tool_results, s.tool_exc
    if tr.empty and te.empty:
        return

    parts: list[pd.DataFrame] = []
    if not tr.empty:
        sub = tr[tr["result_error"].fillna(False)]
        if not sub.empty:
            parts.append(
                sub.pivot_table(
                    index="agent_name_f",
                    columns="tool_name_f",
                    values="row_id",
                    aggfunc="count",
                    fill_value=0,
                )
            )
    if not te.empty:
        parts.append(
            te.pivot_table(
                index="agent_name_f",
                columns="tool_name_f",
                values="row_id",
                aggfunc="count",
                fill_value=0,
            )
        )

    if not parts:
        return
    combined = parts[0]
    for extra in parts[1:]:
        combined = combined.add(extra, fill_value=0)
    if combined.empty:
        return
    _heatmap(
        combined,
        "Tool error heatmap: agent × tool",
        "Tool",
        "Agent",
        p.charts / "16_error_heatmap_agent_tool.png",
    )


def _chart_tool_calls_by_task(s: MetricSlices, p: OutputPaths) -> None:
    if s.tool_calls.empty:
        return
    _bar(
        s.tool_calls["task_name_f"].value_counts(),
        "Tool calls by task",
        "Task",
        "Calls",
        p.charts / "17_tool_calls_by_task.png",
        horizontal=True,
    )


def _chart_tokens_by_iteration(s: MetricSlices, p: OutputPaths) -> None:
    if s.llm.empty or not s.llm["iteration"].notna().any():
        return
    by_iter = (
        s.llm.groupby("iteration")[
            ["prompt_tokens", "completion_tokens", "total_tokens"]
        ]
        .sum()
        .sort_index()
        .reset_index()
    )
    _line(
        by_iter,
        "iteration",
        ["prompt_tokens", "completion_tokens", "total_tokens"],
        "Tokens by iteration",
        "Iteration",
        "Tokens",
        p.charts / "18_tokens_by_iteration.png",
    )


def _chart_tool_calls_by_iteration(s: MetricSlices, p: OutputPaths) -> None:
    if s.tool_calls.empty or not s.tool_calls["iteration"].notna().any():
        return
    by_iter = (
        s.tool_calls.groupby("iteration")
        .size()
        .rename("tool_calls")
        .reset_index()
    )
    _line(
        by_iter,
        "iteration",
        "tool_calls",
        "Tool calls by iteration",
        "Iteration",
        "Calls",
        p.charts / "19_tool_calls_by_iteration.png",
    )


def _chart_avg_tokens_by_agent(s: MetricSlices, p: OutputPaths) -> None:
    if s.llm.empty:
        return
    avg = (
        s.llm.groupby("agent_name_f")["total_tokens"]
        .mean()
        .sort_values(ascending=False)
    )
    _bar(
        avg,
        "Avg total tokens per LLM call by agent",
        "Agent",
        "Avg tokens",
        p.charts / "20_avg_tokens_per_llm_call_by_agent.png",
        horizontal=True,
    )


def _chart_avg_tokens_by_task(s: MetricSlices, p: OutputPaths) -> None:
    if s.llm.empty:
        return
    avg = (
        s.llm.groupby("task_name_f")["total_tokens"]
        .mean()
        .sort_values(ascending=False)
    )
    _bar(
        avg,
        "Avg total tokens per LLM call by task",
        "Task",
        "Avg tokens",
        p.charts / "21_avg_tokens_per_task.png",
        horizontal=True,
    )


def _chart_token_boxplot_by_agent(s: MetricSlices, p: OutputPaths) -> None:
    if s.llm.empty:
        return
    top = (
        s.llm.groupby("agent_name_f")
        .size()
        .sort_values(ascending=False)
        .head(8)
        .index
    )
    groups = {
        agent: s.llm.loc[s.llm["agent_name_f"] == agent, "total_tokens"]
        for agent in top
    }
    _boxplot(
        groups,
        "Token distribution per LLM call by agent",
        "Agent",
        "Total tokens",
        p.charts / "22_token_boxplot_by_agent.png",
    )


def _chart_task_calls_vs_tokens(s: MetricSlices, p: OutputPaths) -> None:
    if s.tool_calls.empty and s.llm.empty:
        return
    parts = {}
    if not s.tool_calls.empty:
        parts["tool_calls"] = (
            s.tool_calls.groupby("task_name_f").size().rename("tool_calls")
        )
    if not s.llm.empty:
        parts["total_tokens"] = (
            s.llm.groupby("task_name_f")["total_tokens"]
            .sum()
            .rename("total_tokens")
        )
    if not parts:
        return
    merged = pd.concat(parts.values(), axis=1).fillna(0)
    _scatter(
        merged.reset_index(),
        "tool_calls",
        "total_tokens",
        "Task: tool calls vs total tokens",
        "Tool calls",
        "Total tokens",
        p.charts / "23_task_tool_calls_vs_tokens.png",
        label_col="task_name_f",
    )


def _chart_tool_calls_vs_errors(s: MetricSlices, p: OutputPaths) -> None:
    if s.err_table.empty:
        return
    scatter_df = s.err_table.reset_index().rename(columns={"index": "tool_name"})
    label = (
        "tool_name_f"
        if "tool_name_f" in scatter_df.columns
        else "tool_name"
    )
    _scatter(
        scatter_df,
        "calls",
        "total_errors",
        "Tool: calls vs total errors",
        "Calls",
        "Errors",
        p.charts / "24_tool_calls_vs_errors.png",
        label_col=label if label in scatter_df.columns else None,
    )


def _chart_summary_agent_count(s: MetricSlices, p: OutputPaths) -> None:
    if s.summaries.empty or not s.summaries["ts"].notna().any():
        return
    t = s.summaries[["ts", "payload"]].copy()
    t["agent_count"] = t["payload"].map(
        lambda pl: len(pl.get("agents", {}))
        if isinstance(pl, dict) and isinstance(pl.get("agents"), dict)
        else 0
    )
    _line(
        t,
        "ts",
        "agent_count",
        "Agent count in metrics_summary over time",
        "Time",
        "Agents",
        p.charts / "25_summary_agent_count_over_time.png",
    )


def _chart_event_volume_per_minute(s: MetricSlices, p: OutputPaths) -> None:
    if not s.all["ts"].notna().any():
        return
    t = s.all[["ts"]].dropna().copy()
    t["minute"] = t["ts"].dt.floor("min")
    volume = t.groupby("minute").size().rename("events").reset_index()
    _line(
        volume,
        "minute",
        "events",
        "Event volume over time (per minute)",
        "Time",
        "Events",
        p.charts / "26_event_volume_per_minute.png",
    )


def _chart_llm_by_model(s: MetricSlices, p: OutputPaths) -> None:
    if s.llm.empty or s.llm["model"].dropna().nunique() == 0:
        return
    _bar(
        s.llm["model"].fillna(_FALLBACKS["model"]).value_counts(),
        "LLM usage by model",
        "Model",
        "Calls",
        p.charts / "27_llm_usage_by_model.png",
        horizontal=True,
        top_n=10,
    )


def _chart_thoughts_vs_cached(s: MetricSlices, p: OutputPaths) -> None:
    ta = s.tokens_by_agent
    if ta.empty:
        return
    if not (
        (ta["thoughts_tokens"] > 0).any() or (ta["cached_tokens"] > 0).any()
    ):
        return
    _stacked_bar(
        ta,
        ["thoughts_tokens", "cached_tokens"],
        "Thoughts vs cached tokens by agent",
        "Agent",
        "Tokens",
        p.charts / "28_thoughts_vs_cached_tokens_by_agent.png",
        sort_by="total_tokens",
    )


def _chart_tool_share_pie(s: MetricSlices, p: OutputPaths) -> None:
    if s.tool_calls.empty:
        return
    _pie(
        s.tool_calls["tool_name_f"].value_counts(),
        "Tool call share (top 8 tools)",
        p.charts / "29_tool_call_share_pie.png",
    )


def _chart_calls_by_invocation(s: MetricSlices, p: OutputPaths) -> None:
    if s.tool_calls.empty:
        return
    inv = s.tool_calls["invocation_id_f"].value_counts()
    _bar(
        inv,
        "Tool calls by invocation",
        "Invocation",
        "Calls",
        p.charts / "30_tool_calls_by_invocation.png",
        horizontal=True,
        top_n=15,
    )


# ── NEW: Latency & Duration charts ───────────────────────────────────────────


def _chart_tool_duration_by_tool(s: MetricSlices, p: OutputPaths) -> None:
    if s.tool_durations.empty:
        return
    avg = (
        s.tool_durations.groupby("tool_name_f")["duration_s"]
        .mean()
        .sort_values(ascending=False)
    )
    _bar(
        avg,
        "Avg tool call duration by tool",
        "Tool",
        "Duration (s)",
        p.charts / "31_avg_tool_duration_by_tool.png",
        horizontal=True,
        top_n=15,
    )


def _chart_tool_duration_histogram(s: MetricSlices, p: OutputPaths) -> None:
    if s.tool_durations.empty:
        return
    _hist(
        s.tool_durations["duration_s"],
        30,
        "Distribution of tool call durations",
        "Duration (s)",
        "Calls",
        p.charts / "32_tool_duration_hist.png",
    )


def _chart_tool_duration_boxplot(s: MetricSlices, p: OutputPaths) -> None:
    if s.tool_durations.empty:
        return
    top = (
        s.tool_durations.groupby("tool_name_f")
        .size()
        .sort_values(ascending=False)
        .head(8)
        .index
    )
    groups = {
        tool: s.tool_durations.loc[
            s.tool_durations["tool_name_f"] == tool, "duration_s"
        ]
        for tool in top
    }
    _boxplot(
        groups,
        "Tool duration distribution (top 8 tools)",
        "Tool",
        "Duration (s)",
        p.charts / "33_tool_duration_boxplot.png",
    )


def _chart_tool_duration_percentiles(
    s: MetricSlices, p: OutputPaths
) -> None:
    if s.tool_durations.empty:
        return
    pcts = (
        s.tool_durations.groupby("tool_name_f")["duration_s"]
        .quantile([0.5, 0.9, 0.99])
        .unstack()
    )
    if pcts.empty:
        return
    pcts.columns = ["p50", "p90", "p99"]
    _save_table(pcts.reset_index(), p.tables / "tool_duration_percentiles.csv")

    top = pcts.sort_values("p90", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(11, 6))
    x_pos = range(len(top))
    width = 0.25
    ax.bar(
        [i - width for i in x_pos], top["p50"], width=width, label="p50"
    )
    ax.bar(x_pos, top["p90"], width=width, label="p90")
    ax.bar(
        [i + width for i in x_pos], top["p99"], width=width, label="p99"
    )
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(top.index.astype(str), rotation=45, ha="right")
    ax.set_title("Tool duration percentiles (p50/p90/p99)")
    ax.set_ylabel("Duration (s)")
    ax.legend()
    _save_fig(fig, p.charts / "34_tool_duration_percentiles.png")


def _chart_invocation_duration_histogram(
    s: MetricSlices, p: OutputPaths
) -> None:
    if s.invocation_durations.empty:
        return
    _hist(
        s.invocation_durations["duration_s"],
        20,
        "Distribution of invocation durations",
        "Duration (s)",
        "Invocations",
        p.charts / "35_invocation_duration_hist.png",
    )


def _chart_invocation_duration_by_invocation(
    s: MetricSlices, p: OutputPaths
) -> None:
    if s.invocation_durations.empty:
        return
    dur = s.invocation_durations["duration_s"].sort_values(ascending=False)
    _bar(
        dur.head(15),
        "Invocation duration (top 15)",
        "Invocation",
        "Duration (s)",
        p.charts / "36_invocation_duration_by_invocation.png",
        horizontal=True,
        top_n=15,
    )


def _chart_duration_vs_tokens_scatter(
    s: MetricSlices, p: OutputPaths
) -> None:
    if s.invocation_durations.empty or s.llm.empty:
        return
    tokens_per_inv = (
        s.llm.groupby("invocation_id_f")["total_tokens"]
        .sum()
        .rename("total_tokens")
    )
    merged = s.invocation_durations[["duration_s"]].join(
        tokens_per_inv, how="inner"
    )
    if merged.empty:
        return
    _scatter(
        merged.reset_index(),
        "total_tokens",
        "duration_s",
        "Invocation duration vs total tokens",
        "Total tokens",
        "Duration (s)",
        p.charts / "37_duration_vs_tokens_scatter.png",
        label_col="invocation_id_f",
    )


def _chart_time_to_first_tool(s: MetricSlices, p: OutputPaths) -> None:
    if (
        s.before_runs.empty
        or s.tool_calls.empty
        or not s.before_runs["ts"].notna().any()
        or not s.tool_calls["ts"].notna().any()
    ):
        return
    run_start = (
        s.before_runs[s.before_runs["ts"].notna()]
        .groupby("invocation_id_f")["ts"]
        .min()
        .rename("run_start")
    )
    first_tool = (
        s.tool_calls[s.tool_calls["ts"].notna()]
        .groupby("invocation_id_f")["ts"]
        .min()
        .rename("first_tool_ts")
    )
    merged = pd.concat([run_start, first_tool], axis=1).dropna()
    merged["ttft_s"] = (
        merged["first_tool_ts"] - merged["run_start"]
    ).dt.total_seconds()
    merged = merged[merged["ttft_s"] >= 0]
    if merged.empty:
        return
    _hist(
        merged["ttft_s"],
        20,
        "Time to first tool call",
        "Time (s)",
        "Invocations",
        p.charts / "38_time_to_first_tool_hist.png",
    )
    _save_table(
        merged[["ttft_s"]].reset_index(),
        p.tables / "time_to_first_tool.csv",
    )


# ── NEW: Cost charts ─────────────────────────────────────────────────────────


def _chart_cost_by_agent(s: MetricSlices, p: OutputPaths) -> None:
    if s.llm_with_cost.empty:
        return
    cost = (
        s.llm_with_cost.groupby("agent_name_f")["estimated_cost"]
        .sum()
        .sort_values(ascending=False)
    )
    _bar(
        cost,
        "Estimated cost by agent",
        "Agent",
        "Cost ($)",
        p.charts / "39_cost_by_agent.png",
        horizontal=True,
    )


def _chart_cost_by_task(s: MetricSlices, p: OutputPaths) -> None:
    if s.llm_with_cost.empty:
        return
    cost = (
        s.llm_with_cost.groupby("task_name_f")["estimated_cost"]
        .sum()
        .sort_values(ascending=False)
    )
    _bar(
        cost,
        "Estimated cost by task",
        "Task",
        "Cost ($)",
        p.charts / "40_cost_by_task.png",
        horizontal=True,
    )


def _chart_cost_by_model(s: MetricSlices, p: OutputPaths) -> None:
    if s.llm_with_cost.empty:
        return
    cost = (
        s.llm_with_cost.groupby(
            s.llm_with_cost["model"].fillna(_FALLBACKS["model"])
        )["estimated_cost"]
        .sum()
        .sort_values(ascending=False)
    )
    _bar(
        cost,
        "Estimated cost by model",
        "Model",
        "Cost ($)",
        p.charts / "41_cost_by_model.png",
        horizontal=True,
        top_n=10,
    )


def _chart_cost_by_invocation(s: MetricSlices, p: OutputPaths) -> None:
    if s.llm_with_cost.empty:
        return
    cost = (
        s.llm_with_cost.groupby("invocation_id_f")["estimated_cost"]
        .sum()
        .sort_values(ascending=False)
    )
    _bar(
        cost,
        "Estimated cost by invocation (top 15)",
        "Invocation",
        "Cost ($)",
        p.charts / "42_cost_by_invocation.png",
        horizontal=True,
        top_n=15,
    )


def _chart_cumulative_cost(s: MetricSlices, p: OutputPaths) -> None:
    if s.llm_with_cost.empty or not s.llm_with_cost["ts"].notna().any():
        return
    t = (
        s.llm_with_cost[["ts", "estimated_cost"]]
        .dropna(subset=["ts"])
        .sort_values("ts")
        .copy()
    )
    t["cum_cost"] = t["estimated_cost"].cumsum()
    _line(
        t,
        "ts",
        "cum_cost",
        "Cumulative estimated cost over time",
        "Time",
        "Cost ($)",
        p.charts / "43_cumulative_cost.png",
    )


def _chart_cost_per_iteration(s: MetricSlices, p: OutputPaths) -> None:
    if s.llm_with_cost.empty or not s.llm_with_cost["iteration"].notna().any():
        return
    by_iter = (
        s.llm_with_cost.groupby("iteration")["estimated_cost"]
        .sum()
        .sort_index()
        .reset_index()
    )
    _line(
        by_iter,
        "iteration",
        "estimated_cost",
        "Estimated cost by iteration",
        "Iteration",
        "Cost ($)",
        p.charts / "44_cost_per_iteration.png",
    )


# ── NEW: Retry & repetition charts ───────────────────────────────────────────


def _chart_retry_rate_by_tool(s: MetricSlices, p: OutputPaths) -> None:
    tc = s.tool_calls_with_retries
    if tc.empty:
        return
    retries = tc.groupby("tool_name_f")["is_retry"].sum().rename("retries")
    total = tc.groupby("tool_name_f").size().rename("calls")
    tbl = pd.concat([total, retries], axis=1).fillna(0)
    tbl["retry_rate"] = tbl["retries"] / tbl["calls"].replace(0, pd.NA)
    tbl = tbl.fillna(0)
    _save_table(tbl.reset_index(), p.tables / "retry_rate_by_tool.csv")
    filtered = tbl[tbl["retries"] > 0].sort_values(
        "retry_rate", ascending=False
    )
    if filtered.empty:
        return
    _bar(
        filtered["retry_rate"],
        "Retry rate by tool",
        "Tool",
        "Retry rate",
        p.charts / "45_retry_rate_by_tool.png",
        horizontal=True,
        top_n=15,
    )


def _chart_retries_by_tool(s: MetricSlices, p: OutputPaths) -> None:
    tc = s.tool_calls_with_retries
    if tc.empty:
        return
    retries = tc[tc["is_retry"]].groupby("tool_name_f").size()
    if retries.empty:
        return
    _bar(
        retries.sort_values(ascending=False),
        "Retry count by tool",
        "Tool",
        "Retries",
        p.charts / "46_retries_by_tool.png",
        horizontal=True,
        top_n=15,
    )


def _chart_tool_calls_per_invocation_hist(
    s: MetricSlices, p: OutputPaths
) -> None:
    if s.tool_calls.empty:
        return
    per_inv = s.tool_calls.groupby("invocation_id_f").size()
    _hist(
        per_inv,
        20,
        "Tool calls per invocation distribution",
        "Tool calls",
        "Invocations",
        p.charts / "47_tool_calls_per_invocation_hist.png",
    )


def _chart_llm_calls_per_invocation_hist(
    s: MetricSlices, p: OutputPaths
) -> None:
    if s.llm.empty:
        return
    per_inv = s.llm.groupby("invocation_id_f").size()
    _hist(
        per_inv,
        20,
        "LLM calls per invocation distribution",
        "LLM calls",
        "Invocations",
        p.charts / "48_llm_calls_per_invocation_hist.png",
    )


def _chart_calls_before_first_success(
    s: MetricSlices, p: OutputPaths
) -> None:
    tc = s.tool_calls_with_retries
    tr = s.tool_results
    if tc.empty or tr.empty:
        return

    success_results = tr[~tr["result_error"].fillna(False)]
    if success_results.empty:
        return

    # For each (invocation, tool), find the row_id of the first success
    first_success = (
        success_results.sort_values(["invocation_id_f", "ts", "row_id"])
        .groupby(["invocation_id_f", "tool_name_f"])["row_id"]
        .first()
        .rename("first_success_row")
    )

    # Count how many calls happened before that row_id
    call_counts: list[dict[str, Any]] = []
    for (inv, tool), success_row in first_success.items():
        mask = (
            (tc["invocation_id_f"] == inv)
            & (tc["tool_name_f"] == tool)
            & (tc["row_id"] <= success_row)
        )
        call_counts.append(
            {"tool": tool, "calls_before_success": int(mask.sum())}
        )

    if not call_counts:
        return
    cdf = pd.DataFrame(call_counts)
    avg = cdf.groupby("tool")["calls_before_success"].mean().sort_values(ascending=False)
    if avg.empty:
        return
    _bar(
        avg.head(15),
        "Avg calls before first success by tool",
        "Tool",
        "Calls",
        p.charts / "49_calls_before_first_success.png",
        horizontal=True,
        top_n=15,
    )


# ── NEW: Agent interaction & flow charts ──────────────────────────────────────


def _chart_agent_transition_heatmap(
    s: MetricSlices, p: OutputPaths
) -> None:
    events = s.adk_events
    if events.empty or not events["author"].notna().any():
        return

    t = (
        events[events["author"].notna()]
        .sort_values(["invocation_id_f", "ts", "row_id"])
        .copy()
    )
    t["prev_author"] = t.groupby("invocation_id_f")["author"].shift(1)
    transitions = t.dropna(subset=["prev_author"])
    if transitions.empty:
        return

    matrix = transitions.pivot_table(
        index="prev_author",
        columns="author",
        values="row_id",
        aggfunc="count",
        fill_value=0,
    )
    if matrix.empty:
        return
    _heatmap(
        matrix,
        "Agent transition heatmap (from → to)",
        "To agent",
        "From agent",
        p.charts / "50_agent_transition_heatmap.png",
    )
    _save_table(
        matrix.reset_index(), p.tables / "agent_transition_matrix.csv"
    )


def _chart_tool_bigrams(s: MetricSlices, p: OutputPaths) -> None:
    if s.tool_calls.empty:
        return
    sequences = (
        s.tool_calls.sort_values(["invocation_id_f", "ts", "row_id"])
        .groupby("invocation_id_f")["tool_name_f"]
        .apply(list)
    )
    bigrams: list[str] = []
    for seq in sequences:
        for i in range(len(seq) - 1):
            bigrams.append(f"{seq[i]} → {seq[i + 1]}")
    if not bigrams:
        return
    counts = pd.Series(bigrams).value_counts()
    _bar(
        counts.head(15),
        "Top tool call bigrams",
        "Sequence",
        "Count",
        p.charts / "51_tool_bigrams.png",
        horizontal=True,
        top_n=15,
    )
    _save_table(
        counts.head(30).reset_index(), p.tables / "tool_bigrams.csv"
    )


def _chart_tool_trigrams(s: MetricSlices, p: OutputPaths) -> None:
    if s.tool_calls.empty:
        return
    sequences = (
        s.tool_calls.sort_values(["invocation_id_f", "ts", "row_id"])
        .groupby("invocation_id_f")["tool_name_f"]
        .apply(list)
    )
    trigrams: list[str] = []
    for seq in sequences:
        for i in range(len(seq) - 2):
            trigrams.append(f"{seq[i]} → {seq[i + 1]} → {seq[i + 2]}")
    if not trigrams:
        return
    counts = pd.Series(trigrams).value_counts()
    _bar(
        counts.head(15),
        "Top tool call trigrams",
        "Sequence",
        "Count",
        p.charts / "52_tool_trigrams.png",
        horizontal=True,
        top_n=15,
    )
    _save_table(
        counts.head(30).reset_index(), p.tables / "tool_trigrams.csv"
    )


def _chart_agents_per_invocation(s: MetricSlices, p: OutputPaths) -> None:
    df = s.all
    if df.empty:
        return
    agents_per = (
        df[df["agent_name"].notna()]
        .groupby("invocation_id_f")["agent_name"]
        .nunique()
    )
    if agents_per.empty:
        return
    _hist(
        agents_per,
        max(5, min(20, int(agents_per.max()))),
        "Distinct agents per invocation",
        "Agent count",
        "Invocations",
        p.charts / "53_agents_per_invocation_hist.png",
    )


# ── NEW: Error depth charts ──────────────────────────────────────────────────


def _chart_error_recovery_rate(s: MetricSlices, p: OutputPaths) -> None:
    tr = s.tool_results
    if tr.empty:
        return
    t = tr.sort_values(["invocation_id_f", "tool_name_f", "ts", "row_id"]).copy()
    t["prev_error"] = t.groupby(["invocation_id_f", "tool_name_f"])[
        "result_error"
    ].shift(1)
    recoveries = t[t["prev_error"].fillna(False) & ~t["result_error"].fillna(True)]
    fails_followed = t[t["prev_error"].fillna(False)]

    if fails_followed.empty:
        return

    recovery_by_tool = recoveries.groupby("tool_name_f").size().rename("recoveries")
    fail_follow_by_tool = (
        fails_followed.groupby("tool_name_f").size().rename("after_fail")
    )
    tbl = pd.concat([fail_follow_by_tool, recovery_by_tool], axis=1).fillna(0)
    tbl["recovery_rate"] = tbl["recoveries"] / tbl["after_fail"].replace(
        0, pd.NA
    )
    tbl = tbl.fillna(0).sort_values("recovery_rate", ascending=False)
    _save_table(tbl.reset_index(), p.tables / "error_recovery_rate.csv")

    if (tbl["recoveries"] > 0).any():
        _bar(
            tbl[tbl["recoveries"] > 0]["recovery_rate"],
            "Error recovery rate by tool",
            "Tool",
            "Recovery rate",
            p.charts / "54_error_recovery_rate.png",
            horizontal=True,
        )


def _chart_error_clustering(s: MetricSlices, p: OutputPaths) -> None:
    df = s.all
    if not df["ts"].notna().any():
        return
    t = df[["ts", "type_f", "result_error"]].dropna(subset=["ts"]).copy()
    t["is_error"] = (
        (t["type_f"] == _METRIC_EVENT_TYPES["tool_exc"])
        | (
            (t["type_f"] == _METRIC_EVENT_TYPES["tool_result"])
            & t["result_error"].fillna(False)
        )
    ).astype(int)
    t["minute"] = t["ts"].dt.floor("min")
    by_min = t.groupby("minute")["is_error"].sum().rename("errors").reset_index()
    if by_min.empty or by_min["errors"].sum() == 0:
        return
    _line(
        by_min,
        "minute",
        "errors",
        "Error clustering over time (per minute)",
        "Time",
        "Errors",
        p.charts / "55_error_clustering_per_minute.png",
    )


def _chart_errors_per_iteration(s: MetricSlices, p: OutputPaths) -> None:
    df = s.all
    if not df["iteration"].notna().any():
        return
    t = df.copy()
    t["is_error"] = (
        (t["type_f"] == _METRIC_EVENT_TYPES["tool_exc"])
        | (
            (t["type_f"] == _METRIC_EVENT_TYPES["tool_result"])
            & t["result_error"].fillna(False)
        )
    ).astype(int)
    by_iter = (
        t.groupby("iteration")["is_error"].sum().rename("errors").reset_index()
    )
    if by_iter.empty or by_iter["errors"].sum() == 0:
        return
    _line(
        by_iter,
        "iteration",
        "errors",
        "Errors by iteration",
        "Iteration",
        "Errors",
        p.charts / "56_errors_per_iteration.png",
    )


def _chart_first_error_position(s: MetricSlices, p: OutputPaths) -> None:
    tc = s.tool_calls
    tr = s.tool_results
    te = s.tool_exc
    if tc.empty:
        return

    # Number each tool call within its invocation
    numbered = (
        tc.sort_values(["invocation_id_f", "ts", "row_id"])
        .copy()
    )
    numbered["call_seq"] = numbered.groupby("invocation_id_f").cumcount() + 1

    error_rows = set()
    if not tr.empty:
        error_rows |= set(
            tr[tr["result_error"].fillna(False)]["row_id"].tolist()
        )
    if not te.empty:
        error_rows |= set(te["row_id"].tolist())

    # For each invocation, find the position of the first error
    # (by matching row_ids that are also call rows, or by sequential proximity)
    # Simplified: match tool_results back to tool_calls by invocation + tool + sequence
    if not error_rows:
        return

    # Merge results back to calls to find which call_seq had the first error
    positions: list[int] = []
    for inv, group in numbered.groupby("invocation_id_f"):
        inv_error_tools = set()
        if not tr.empty:
            inv_errors = tr[
                (tr["invocation_id_f"] == inv)
                & tr["result_error"].fillna(False)
            ]
            inv_error_tools |= set(inv_errors["tool_name_f"].tolist())
        if not te.empty:
            inv_exc = te[te["invocation_id_f"] == inv]
            inv_error_tools |= set(inv_exc["tool_name_f"].tolist())

        if inv_error_tools:
            first = group[group["tool_name_f"].isin(inv_error_tools)][
                "call_seq"
            ].min()
            if pd.notna(first):
                positions.append(int(first))

    if not positions:
        return
    _hist(
        pd.Series(positions),
        max(5, min(20, max(positions))),
        "First error position within invocation",
        "Call sequence #",
        "Invocations",
        p.charts / "57_first_error_position_hist.png",
    )


def _chart_top_error_messages(s: MetricSlices, p: OutputPaths) -> None:
    errors = s.tool_exc["error"].dropna()
    if errors.empty:
        return
    truncated = errors.str[:120]
    counts = truncated.value_counts()
    _save_table(
        counts.head(30).reset_index().rename(
            columns={"index": "error_message", "count": "count"}
        ),
        p.tables / "top_error_messages.csv",
    )
    _bar(
        counts.head(15),
        "Top error messages",
        "Error",
        "Count",
        p.charts / "58_top_error_messages.png",
        horizontal=True,
    )


# ── NEW: Efficiency ratio charts ─────────────────────────────────────────────


def _chart_tokens_per_tool_call(s: MetricSlices, p: OutputPaths) -> None:
    if s.llm.empty or s.tool_calls.empty:
        return
    tokens = (
        s.llm.groupby("invocation_id_f")["total_tokens"]
        .sum()
        .rename("total_tokens")
    )
    calls = (
        s.tool_calls.groupby("invocation_id_f")
        .size()
        .rename("tool_calls")
    )
    merged = pd.concat([tokens, calls], axis=1).dropna()
    merged = merged[merged["tool_calls"] > 0]
    merged["tokens_per_call"] = merged["total_tokens"] / merged["tool_calls"]
    if merged.empty:
        return
    _hist(
        merged["tokens_per_call"],
        20,
        "Tokens per tool call (per invocation)",
        "Tokens / tool call",
        "Invocations",
        p.charts / "59_tokens_per_tool_call_hist.png",
    )


def _chart_cache_hit_rate_by_agent(s: MetricSlices, p: OutputPaths) -> None:
    if s.tokens_by_agent.empty:
        return
    ta = s.tokens_by_agent.copy()
    ta["cache_hit_rate"] = ta["cached_tokens"] / ta["prompt_tokens"].replace(
        0, pd.NA
    )
    ta = ta.dropna(subset=["cache_hit_rate"])
    if ta.empty or (ta["cache_hit_rate"] == 0).all():
        return
    _bar(
        ta["cache_hit_rate"].sort_values(ascending=False),
        "Cache hit rate by agent (cached / prompt tokens)",
        "Agent",
        "Cache hit rate",
        p.charts / "60_cache_hit_rate_by_agent.png",
        horizontal=True,
    )


def _chart_thinking_overhead_by_agent(
    s: MetricSlices, p: OutputPaths
) -> None:
    if s.tokens_by_agent.empty:
        return
    ta = s.tokens_by_agent.copy()
    ta["thinking_ratio"] = ta["thoughts_tokens"] / ta["total_tokens"].replace(
        0, pd.NA
    )
    ta = ta.dropna(subset=["thinking_ratio"])
    if ta.empty or (ta["thinking_ratio"] == 0).all():
        return
    _bar(
        ta["thinking_ratio"].sort_values(ascending=False),
        "Thinking overhead by agent (thoughts / total tokens)",
        "Agent",
        "Thinking ratio",
        p.charts / "61_thinking_overhead_by_agent.png",
        horizontal=True,
    )


def _chart_useful_work_ratio_by_tool(
    s: MetricSlices, p: OutputPaths
) -> None:
    if s.err_table.empty:
        return
    tbl = s.err_table.copy()
    tbl["success"] = tbl["calls"] - tbl["total_errors"]
    tbl["useful_ratio"] = tbl["success"] / tbl["calls"].replace(0, pd.NA)
    tbl = tbl.dropna(subset=["useful_ratio"])
    filtered = tbl[tbl["calls"] >= _MIN_CALLS_FOR_ERROR_RATE].sort_values(
        "useful_ratio", ascending=True
    )
    if filtered.empty:
        return
    _bar(
        filtered["useful_ratio"],
        f"Useful work ratio by tool (min {_MIN_CALLS_FOR_ERROR_RATE} calls)",
        "Tool",
        "Success ratio",
        p.charts / "62_useful_work_ratio_by_tool.png",
        horizontal=True,
        top_n=15,
    )


def _chart_tokens_per_successful_call(
    s: MetricSlices, p: OutputPaths
) -> None:
    if s.llm.empty or s.err_table.empty:
        return
    tokens = (
        s.llm.groupby("invocation_id_f")["total_tokens"]
        .sum()
        .rename("total_tokens")
    )
    success_results = s.tool_results[
        ~s.tool_results["result_error"].fillna(False)
    ]
    if success_results.empty:
        return
    successes = (
        success_results.groupby("invocation_id_f")
        .size()
        .rename("successes")
    )
    merged = pd.concat([tokens, successes], axis=1).dropna()
    merged = merged[merged["successes"] > 0]
    merged["tokens_per_success"] = (
        merged["total_tokens"] / merged["successes"]
    )
    if merged.empty:
        return
    _hist(
        merged["tokens_per_success"],
        20,
        "Tokens per successful tool call (per invocation)",
        "Tokens / success",
        "Invocations",
        p.charts / "63_tokens_per_successful_call_hist.png",
    )


# ── NEW: Session & cross-iteration charts ────────────────────────────────────


def _chart_error_rate_by_iteration(s: MetricSlices, p: OutputPaths) -> None:
    tc = s.tool_calls
    tr = s.tool_results
    te = s.tool_exc
    if tc.empty or not tc["iteration"].notna().any():
        return

    calls_per = tc.groupby("iteration").size().rename("calls")
    err_count = pd.Series(dtype="float64")
    if not tr.empty and tr["iteration"].notna().any():
        err_count = err_count.add(
            tr[tr["result_error"].fillna(False)]
            .groupby("iteration")
            .size()
            .rename("errors"),
            fill_value=0,
        )
    if not te.empty and te["iteration"].notna().any():
        err_count = err_count.add(
            te.groupby("iteration").size().rename("errors"), fill_value=0
        )

    tbl = pd.concat(
        [calls_per, err_count.rename("errors")], axis=1
    ).fillna(0)
    tbl["error_rate"] = tbl["errors"] / tbl["calls"].replace(0, pd.NA)
    tbl = tbl.fillna(0).sort_index().reset_index()
    if tbl.empty:
        return
    _line(
        tbl,
        "iteration",
        "error_rate",
        "Error rate trend across iterations",
        "Iteration",
        "Error rate",
        p.charts / "64_error_rate_by_iteration.png",
    )
    _save_table(tbl, p.tables / "error_rate_by_iteration.csv")


def _chart_session_duration(s: MetricSlices, p: OutputPaths) -> None:
    df = s.all
    if df.empty or not df["ts"].notna().any():
        return
    t = df[df["ts"].notna()].copy()
    session_dur = t.groupby("session_id_f")["ts"].agg(["min", "max"])
    session_dur["duration_s"] = (
        session_dur["max"] - session_dur["min"]
    ).dt.total_seconds()
    session_dur = session_dur[session_dur["duration_s"] >= 0]
    if session_dur.empty:
        return
    _hist(
        session_dur["duration_s"],
        20,
        "Session duration distribution",
        "Duration (s)",
        "Sessions",
        p.charts / "65_session_duration_hist.png",
    )


def _chart_events_per_session(s: MetricSlices, p: OutputPaths) -> None:
    df = s.all
    if df.empty:
        return
    per_session = df.groupby("session_id_f").size()
    if per_session.empty:
        return
    _hist(
        per_session,
        max(5, min(25, int(per_session.max()))),
        "Events per session distribution",
        "Event count",
        "Sessions",
        p.charts / "66_events_per_session_hist.png",
    )


# ─── Chart pipeline ──────────────────────────────────────────────────────────

_CHART_PIPELINE: Sequence[tuple[str, Callable[[MetricSlices, OutputPaths], None]]] = [
    # ── Original charts (01–30) ───────────────────────────────────────────
    ("event_counts", _chart_event_counts),
    ("tool_calls_by_tool", _chart_tool_calls_by_tool),
    ("tool_calls_by_agent", _chart_tool_calls_by_agent),
    ("llm_calls_by_agent", _chart_llm_calls_by_agent),
    ("total_tokens_by_agent", _chart_total_tokens_by_agent),
    ("prompt_vs_completion_by_agent", _chart_prompt_vs_completion_by_agent),
    ("total_tokens_by_task", _chart_total_tokens_by_task),
    ("token_histogram", _chart_token_histogram),
    ("prompt_completion_scatter", _chart_prompt_completion_scatter),
    ("cumulative_tokens", _chart_cumulative_tokens),
    ("cumulative_tool_calls_errors", _chart_cumulative_tool_calls_errors),
    ("errors_by_tool", _chart_errors_by_tool),
    ("error_rate_by_tool", _chart_error_rate_by_tool),
    ("result_vs_exception_errors", _chart_result_vs_exception_errors),
    ("calls_heatmap", _chart_calls_heatmap),
    ("error_heatmap", _chart_error_heatmap),
    ("tool_calls_by_task", _chart_tool_calls_by_task),
    ("tokens_by_iteration", _chart_tokens_by_iteration),
    ("tool_calls_by_iteration", _chart_tool_calls_by_iteration),
    ("avg_tokens_by_agent", _chart_avg_tokens_by_agent),
    ("avg_tokens_by_task", _chart_avg_tokens_by_task),
    ("token_boxplot_by_agent", _chart_token_boxplot_by_agent),
    ("task_calls_vs_tokens", _chart_task_calls_vs_tokens),
    ("tool_calls_vs_errors", _chart_tool_calls_vs_errors),
    ("summary_agent_count", _chart_summary_agent_count),
    ("event_volume_per_minute", _chart_event_volume_per_minute),
    ("llm_by_model", _chart_llm_by_model),
    ("thoughts_vs_cached", _chart_thoughts_vs_cached),
    ("tool_share_pie", _chart_tool_share_pie),
    ("calls_by_invocation", _chart_calls_by_invocation),
    # ── Latency & duration (31–38) ────────────────────────────────────────
    ("tool_duration_by_tool", _chart_tool_duration_by_tool),
    ("tool_duration_histogram", _chart_tool_duration_histogram),
    ("tool_duration_boxplot", _chart_tool_duration_boxplot),
    ("tool_duration_percentiles", _chart_tool_duration_percentiles),
    ("invocation_duration_histogram", _chart_invocation_duration_histogram),
    ("invocation_duration_by_invocation", _chart_invocation_duration_by_invocation),
    ("duration_vs_tokens_scatter", _chart_duration_vs_tokens_scatter),
    ("time_to_first_tool", _chart_time_to_first_tool),
    # ── Cost (39–44) ─────────────────────────────────────────────────────
    ("cost_by_agent", _chart_cost_by_agent),
    ("cost_by_task", _chart_cost_by_task),
    ("cost_by_model", _chart_cost_by_model),
    ("cost_by_invocation", _chart_cost_by_invocation),
    ("cumulative_cost", _chart_cumulative_cost),
    ("cost_per_iteration", _chart_cost_per_iteration),
    # ── Retry & repetition (45–49) ───────────────────────────────────────
    ("retry_rate_by_tool", _chart_retry_rate_by_tool),
    ("retries_by_tool", _chart_retries_by_tool),
    ("tool_calls_per_invocation_hist", _chart_tool_calls_per_invocation_hist),
    ("llm_calls_per_invocation_hist", _chart_llm_calls_per_invocation_hist),
    ("calls_before_first_success", _chart_calls_before_first_success),
    # ── Agent interaction & flow (50–53) ──────────────────────────────────
    ("agent_transition_heatmap", _chart_agent_transition_heatmap),
    ("tool_bigrams", _chart_tool_bigrams),
    ("tool_trigrams", _chart_tool_trigrams),
    ("agents_per_invocation", _chart_agents_per_invocation),
    # ── Error depth (54–58) ──────────────────────────────────────────────
    ("error_recovery_rate", _chart_error_recovery_rate),
    ("error_clustering", _chart_error_clustering),
    ("errors_per_iteration", _chart_errors_per_iteration),
    ("first_error_position", _chart_first_error_position),
    ("top_error_messages", _chart_top_error_messages),
    # ── Efficiency ratios (59–63) ────────────────────────────────────────
    ("tokens_per_tool_call", _chart_tokens_per_tool_call),
    ("cache_hit_rate_by_agent", _chart_cache_hit_rate_by_agent),
    ("thinking_overhead_by_agent", _chart_thinking_overhead_by_agent),
    ("useful_work_ratio_by_tool", _chart_useful_work_ratio_by_tool),
    ("tokens_per_successful_call", _chart_tokens_per_successful_call),
    # ── Session & cross-iteration (64–66) ────────────────────────────────
    ("error_rate_by_iteration", _chart_error_rate_by_iteration),
    ("session_duration", _chart_session_duration),
    ("events_per_session", _chart_events_per_session),
]


# ─── Tables pipeline ─────────────────────────────────────────────────────────


def _emit_tables(s: MetricSlices, p: OutputPaths) -> None:
    _save_table(
        s.all.drop(
            columns=["raw", "payload", "tool_args", "result"], errors="ignore"
        ),
        p.tables / "events_flat.csv",
    )

    if not s.tools_by_calls.empty:
        _save_table(
            s.tools_by_calls.reset_index(), p.tables / "tools_by_calls.csv"
        )

    if not s.err_table.empty:
        _save_table(
            s.err_table.reset_index(), p.tables / "tool_error_table.csv"
        )

    if not s.tokens_by_agent.empty:
        _save_table(
            s.tokens_by_agent.reset_index(),
            p.tables / "tokens_by_agent.csv",
        )

    if not s.tokens_by_task.empty:
        _save_table(
            s.tokens_by_task.reset_index(), p.tables / "tokens_by_task.csv"
        )

    if not s.llm.empty:
        llm_calls = (
            s.llm.groupby("agent_name_f")
            .size()
            .rename("llm_calls")
            .sort_values(ascending=False)
        )
        _save_table(
            llm_calls.reset_index(), p.tables / "llm_calls_by_agent.csv"
        )

    if not s.calls_by_agent_tool.empty:
        _save_table(
            s.calls_by_agent_tool.reset_index(),
            p.tables / "calls_by_agent_tool.csv",
        )

    # Duration tables
    if not s.tool_durations.empty:
        _save_table(
            s.tool_durations[
                ["invocation_id_f", "agent_name_f", "tool_name_f", "duration_s"]
            ],
            p.tables / "tool_durations.csv",
        )

    if not s.invocation_durations.empty:
        _save_table(
            s.invocation_durations[["duration_s"]].reset_index(),
            p.tables / "invocation_durations.csv",
        )

    # Cost tables
    if not s.llm_with_cost.empty:
        cost_by_agent = (
            s.llm_with_cost.groupby("agent_name_f")["estimated_cost"]
            .sum()
            .sort_values(ascending=False)
        )
        _save_table(
            cost_by_agent.reset_index(), p.tables / "cost_by_agent.csv"
        )

        cost_by_task = (
            s.llm_with_cost.groupby("task_name_f")["estimated_cost"]
            .sum()
            .sort_values(ascending=False)
        )
        _save_table(
            cost_by_task.reset_index(), p.tables / "cost_by_task.csv"
        )

        cost_by_model = (
            s.llm_with_cost.groupby(
                s.llm_with_cost["model"].fillna(_FALLBACKS["model"])
            )["estimated_cost"]
            .sum()
            .sort_values(ascending=False)
        )
        _save_table(
            cost_by_model.reset_index(), p.tables / "cost_by_model.csv"
        )

    # Retry tables
    if not s.tool_calls_with_retries.empty:
        retries = s.tool_calls_with_retries[
            s.tool_calls_with_retries["is_retry"]
        ]
        if not retries.empty:
            _save_table(
                retries[
                    [
                        "invocation_id_f",
                        "agent_name_f",
                        "tool_name_f",
                        "ts",
                    ]
                ].reset_index(drop=True),
                p.tables / "detected_retries.csv",
            )


# ─── Main orchestrator ───────────────────────────────────────────────────────


def analyze(df: pd.DataFrame, paths: OutputPaths) -> None:
    if df.empty:
        summary = compute_summary(df)
        (paths.root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_markdown_report(df, summary, paths.root / "report.md")
        _save_table(df, paths.tables / "events_flat.csv")
        return

    slices = MetricSlices.build(df)

    # Summary + report
    summary = compute_summary(df, slices)
    (paths.root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown_report(df, summary, paths.root / "report.md")

    # Tables
    _emit_tables(slices, paths)

    # Charts — each isolated so one failure doesn't abort the rest
    generated = 0
    skipped = 0
    failed = 0
    for chart_name, chart_fn in _CHART_PIPELINE:
        try:
            chart_fn(slices, paths)
            generated += 1
        except Exception:
            failed += 1
            logger.warning(
                "Chart '%s' failed:\n%s", chart_name, traceback.format_exc()
            )

    logger.info(
        "Charts complete: %d generated, %d failed out of %d",
        generated,
        failed,
        len(_CHART_PIPELINE),
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze contractor metrics.jsonl → charts, tables, report.",
    )
    parser.add_argument("input", type=Path, help="Path to metrics.jsonl")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input_dir>/metrics_report)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    input_path = args.input.resolve()
    paths = OutputPaths.from_args(input_path, args.output_dir)
    records = load_jsonl(input_path)
    df = normalize_records(records)
    analyze(df, paths)
    print(f"Done. Report written to: {paths.root}")


if __name__ == "__main__":
    main()