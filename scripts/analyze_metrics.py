from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class Paths:
    input_file: Path
    output_dir: Path
    charts_dir: Path
    tables_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze contractor metrics.jsonl and build charts/tables/report."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to metrics.jsonl",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write charts and tables. Defaults to <input_dir>/metrics_report",
    )
    return parser.parse_args()


def ensure_dirs(input_file: Path, output_dir: Path | None) -> Paths:
    input_file = input_file.resolve()
    out = output_dir.resolve() if output_dir else input_file.parent / "metrics_report"
    charts = out / "charts"
    tables = out / "tables"
    charts.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return Paths(input_file=input_file, output_dir=out, charts_dir=charts, tables_dir=tables)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {i}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def normalize_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    norm_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        tool_args = payload.get("tool_args") if isinstance(payload.get("tool_args"), dict) else {}
        result = payload.get("result")

        row = {
            "row_id": idx,
            "ts": pd.to_datetime(record.get("ts"), utc=True, errors="coerce"),
            "type": _stringify(record.get("type")),
            "task_name": _stringify(record.get("task_name")),
            "task_id": record.get("task_id"),
            "iteration": record.get("iteration", payload.get("iteration")),
            "session_id": _stringify(record.get("session_id", payload.get("session_id"))),
            "invocation_id": _stringify(record.get("invocation_id", payload.get("invocation_id"))),
            "agent_name": _stringify(record.get("agent_name", payload.get("agent_name"))),
            "tool_name": _stringify(record.get("tool_name", payload.get("tool_name"))),
            "result_error": bool(payload.get("result_error", False)),
            "error": _stringify(payload.get("error") or payload.get("error_message")),
            "model": _stringify(payload.get("model")),
            "prompt_tokens": pd.to_numeric(usage.get("prompt"), errors="coerce"),
            "completion_tokens": pd.to_numeric(usage.get("completion"), errors="coerce"),
            "total_tokens": pd.to_numeric(usage.get("total"), errors="coerce"),
            "thoughts_tokens": pd.to_numeric(usage.get("thoughts"), errors="coerce"),
            "cached_tokens": pd.to_numeric(usage.get("cached"), errors="coerce"),
            "tool_args": tool_args,
            "result": result,
            "payload": payload,
            "raw": record,
        }
        norm_rows.append(row)

    df = pd.DataFrame(norm_rows)
    if df.empty:
        return df

    for col in ["task_id", "iteration"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "thoughts_tokens",
        "cached_tokens",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "ts" in df.columns:
        df = df.sort_values(["ts", "row_id"], kind="stable").reset_index(drop=True)

    return df


def safe_group_label(series: pd.Series, fallback: str) -> pd.Series:
    out = series.fillna(fallback).astype(str)
    out = out.replace({"nan": fallback, "None": fallback, "": fallback})
    return out


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def save_plot(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def top_n(series: pd.Series, n: int = 12) -> pd.Series:
    s = series.sort_values(ascending=False)
    if len(s) <= n:
        return s
    head = s.iloc[:n].copy()
    head.loc["__other__"] = s.iloc[n:].sum()
    head.index = ["other" if x == "__other__" else x for x in head.index]
    return head


def make_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, path: Path, horizontal: bool = False) -> None:
    if series.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, min(12, 0.45 * len(series) + 2)) if horizontal else 6))
    if horizontal:
        series = series.sort_values()
        ax.barh(series.index.astype(str), series.values)
    else:
        ax.bar(series.index.astype(str), series.values)
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_plot(fig, path)


def make_line(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(df[x], df[y])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    save_plot(fig, path)


def make_multi_line(df: pd.DataFrame, x: str, lines: list[str], title: str, xlabel: str, ylabel: str, path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for col in lines:
        if col in df.columns:
            ax.plot(df[x], df[col], label=col)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.tick_params(axis="x", rotation=25)
    save_plot(fig, path)


def make_hist(series: pd.Series, bins: int, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(s, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_plot(fig, path)


def make_scatter(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, path: Path, label_col: str | None = None) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df[x], df[y], alpha=0.7)
    if label_col and label_col in df.columns and len(df) <= 40:
        for _, row in df.iterrows():
            ax.annotate(str(row[label_col]), (row[x], row[y]), fontsize=7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_plot(fig, path)


def make_heatmap(matrix: pd.DataFrame, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    if matrix.empty:
        return
    fig_width = max(8, min(18, 0.7 * max(1, matrix.shape[1]) + 3))
    fig_height = max(5, min(16, 0.5 * max(1, matrix.shape[0]) + 2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix.values, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([str(c) for c in matrix.columns], rotation=45, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([str(i) for i in matrix.index])
    fig.colorbar(im, ax=ax)
    save_plot(fig, path)


def make_boxplot(groups: dict[str, pd.Series], title: str, xlabel: str, ylabel: str, path: Path) -> None:
    clean = {k: pd.to_numeric(v, errors="coerce").dropna() for k, v in groups.items()}
    clean = {k: v for k, v in clean.items() if not v.empty}
    if not clean:
        return
    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(clean)), 6))
    ax.boxplot([v.values for v in clean.values()], tick_labels=list(clean.keys()), vert=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    save_plot(fig, path)


def compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rows": int(len(df)),
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
    }
    if df.empty:
        return out

    if df["ts"].notna().any():
        out["date_min"] = df["ts"].min().isoformat()
        out["date_max"] = df["ts"].max().isoformat()

    out["event_types"] = df["type"].fillna("unknown").value_counts().to_dict()

    tool_call_mask = df["type"].eq("metrics_tool_call")
    tool_result_mask = df["type"].eq("metrics_tool_result")
    tool_exc_mask = df["type"].eq("metrics_tool_exception_error")
    llm_mask = df["type"].eq("metrics_llm_usage")

    out["tool_calls"] = int(tool_call_mask.sum())
    out["tool_result_errors"] = int((tool_result_mask & df["result_error"].fillna(False)).sum())
    out["tool_exception_errors"] = int(tool_exc_mask.sum())
    out["llm_calls"] = int(llm_mask.sum())
    out["prompt_tokens"] = int(df.loc[llm_mask, "prompt_tokens"].sum())
    out["completion_tokens"] = int(df.loc[llm_mask, "completion_tokens"].sum())
    out["total_tokens"] = int(df.loc[llm_mask, "total_tokens"].sum())
    out["agents"] = int(df["agent_name"].dropna().nunique())
    out["tools"] = int(df["tool_name"].dropna().nunique())
    out["tasks"] = int(df["task_name"].dropna().nunique())
    return out


def write_markdown_report(df: pd.DataFrame, summary: dict[str, Any], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Metrics report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Rows: {summary['rows']}")
    lines.append(f"- Time range: {summary['date_min'] or 'n/a'} → {summary['date_max'] or 'n/a'}")
    lines.append(f"- Tool calls: {summary['tool_calls']}")
    lines.append(f"- Tool result errors: {summary['tool_result_errors']}")
    lines.append(f"- Tool exception errors: {summary['tool_exception_errors']}")
    lines.append(f"- LLM calls: {summary['llm_calls']}")
    lines.append(f"- Prompt tokens: {summary['prompt_tokens']}")
    lines.append(f"- Completion tokens: {summary['completion_tokens']}")
    lines.append(f"- Total tokens: {summary['total_tokens']}")
    lines.append(f"- Distinct agents: {summary['agents']}")
    lines.append(f"- Distinct tools: {summary['tools']}")
    lines.append(f"- Distinct tasks: {summary['tasks']}")
    lines.append("")

    if not df.empty:
        tool_call_mask = df["type"].eq("metrics_tool_call")
        llm_mask = df["type"].eq("metrics_llm_usage")
        top_tools = (
            safe_group_label(df.loc[tool_call_mask, "tool_name"], "unknown_tool")
            .value_counts()
            .head(10)
        )
        lines.append("## Top tools by calls")
        lines.append("")
        for tool, n in top_tools.items():
            lines.append(f"- {tool}: {int(n)}")
        lines.append("")

        top_agents_tokens = (
            df.loc[llm_mask]
            .assign(agent=safe_group_label(df.loc[llm_mask, "agent_name"], "unknown_agent"))
            .groupby("agent", dropna=False)["total_tokens"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        lines.append("## Top agents by total tokens")
        lines.append("")
        for agent, n in top_agents_tokens.items():
            lines.append(f"- {agent}: {int(n)}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def analyze(df: pd.DataFrame, paths: Paths) -> None:
    summary = compute_summary(df)
    (paths.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(df, summary, paths.output_dir / "report.md")

    save_table(df.drop(columns=["raw", "payload", "tool_args", "result"], errors="ignore"), paths.tables_dir / "events_flat.csv")

    if df.empty:
        return

    df = df.copy()
    df["task_name_f"] = safe_group_label(df["task_name"], "unknown_task")
    df["agent_name_f"] = safe_group_label(df["agent_name"], "unknown_agent")
    df["tool_name_f"] = safe_group_label(df["tool_name"], "unknown_tool")
    df["type_f"] = safe_group_label(df["type"], "unknown_type")

    tool_calls = df[df["type_f"] == "metrics_tool_call"].copy()
    tool_results = df[df["type_f"] == "metrics_tool_result"].copy()
    tool_exc = df[df["type_f"] == "metrics_tool_exception_error"].copy()
    llm = df[df["type_f"] == "metrics_llm_usage"].copy()
    summaries = df[df["type_f"] == "metrics_summary"].copy()

    # Tables
    if not tool_calls.empty:
        tools_by_calls = tool_calls.groupby("tool_name_f").size().sort_values(ascending=False).rename("calls")
        save_table(tools_by_calls.reset_index(), paths.tables_dir / "tools_by_calls.csv")
    else:
        tools_by_calls = pd.Series(dtype="int64")

    if not tool_results.empty or not tool_exc.empty or not tool_calls.empty:
        result_errors = (
            tool_results.groupby("tool_name_f")["result_error"].sum().rename("result_errors")
            if not tool_results.empty
            else pd.Series(dtype="float64")
        )
        exc_errors = (
            tool_exc.groupby("tool_name_f").size().rename("exception_errors")
            if not tool_exc.empty
            else pd.Series(dtype="int64")
        )
        calls = (
            tool_calls.groupby("tool_name_f").size().rename("calls")
            if not tool_calls.empty
            else pd.Series(dtype="int64")
        )

        err_table = pd.concat([calls, result_errors, exc_errors], axis=1)
        err_table = err_table.reindex(
            columns=["calls", "result_errors", "exception_errors"],
            fill_value=0,
        )
        err_table = err_table.fillna(0)

        err_table["calls"] = pd.to_numeric(err_table["calls"], errors="coerce").fillna(0)
        err_table["result_errors"] = pd.to_numeric(err_table["result_errors"], errors="coerce").fillna(0)
        err_table["exception_errors"] = pd.to_numeric(err_table["exception_errors"], errors="coerce").fillna(0)

        err_table["total_errors"] = err_table["result_errors"] + err_table["exception_errors"]
        err_table["error_rate"] = err_table["total_errors"] / err_table["calls"].replace(0, pd.NA)
        err_table = err_table.fillna(0).sort_values(["total_errors", "calls"], ascending=False)
        save_table(err_table.reset_index(), paths.tables_dir / "tool_error_table.csv")
    else:
        err_table = pd.DataFrame(
            columns=["calls", "result_errors", "exception_errors", "total_errors", "error_rate"]
        )

    if not llm.empty:
        tokens_by_agent = llm.groupby("agent_name_f")[["prompt_tokens", "completion_tokens", "total_tokens", "thoughts_tokens", "cached_tokens"]].sum().sort_values("total_tokens", ascending=False)
        save_table(tokens_by_agent.reset_index(), paths.tables_dir / "tokens_by_agent.csv")
        tokens_by_task = llm.groupby("task_name_f")[["prompt_tokens", "completion_tokens", "total_tokens"]].sum().sort_values("total_tokens", ascending=False)
        save_table(tokens_by_task.reset_index(), paths.tables_dir / "tokens_by_task.csv")
        llm_calls_by_agent = llm.groupby("agent_name_f").size().rename("llm_calls").sort_values(ascending=False)
        save_table(llm_calls_by_agent.reset_index(), paths.tables_dir / "llm_calls_by_agent.csv")
    else:
        tokens_by_agent = pd.DataFrame()
        tokens_by_task = pd.DataFrame()
        llm_calls_by_agent = pd.Series(dtype="int64")

    if not tool_calls.empty:
        calls_by_agent_tool = tool_calls.pivot_table(index="agent_name_f", columns="tool_name_f", values="row_id", aggfunc="count", fill_value=0)
        save_table(calls_by_agent_tool.reset_index(), paths.tables_dir / "calls_by_agent_tool.csv")
    else:
        calls_by_agent_tool = pd.DataFrame()

    if not err_table.empty and not calls_by_agent_tool.empty:
        pass

    # Charts 1: event counts by type
    event_counts = top_n(df["type_f"].value_counts(), 12)
    make_bar(event_counts, "Event counts by type", "Event type", "Count", paths.charts_dir / "01_event_counts_by_type.png")

    # Charts 2: tool calls by tool
    if not tool_calls.empty:
        make_bar(top_n(tool_calls["tool_name_f"].value_counts(), 15), "Tool calls by tool", "Tool", "Calls", paths.charts_dir / "02_tool_calls_by_tool.png", horizontal=True)

    # Charts 3: tool calls by agent
    if not tool_calls.empty:
        make_bar(top_n(tool_calls["agent_name_f"].value_counts(), 12), "Tool calls by agent", "Agent", "Calls", paths.charts_dir / "03_tool_calls_by_agent.png", horizontal=True)

    # Charts 4: llm calls by agent
    if not llm.empty:
        make_bar(top_n(llm["agent_name_f"].value_counts(), 12), "LLM calls by agent", "Agent", "Calls", paths.charts_dir / "04_llm_calls_by_agent.png", horizontal=True)

    # Charts 5: total tokens by agent
    if not tokens_by_agent.empty:
        make_bar(top_n(tokens_by_agent["total_tokens"], 12), "Total tokens by agent", "Agent", "Tokens", paths.charts_dir / "05_total_tokens_by_agent.png", horizontal=True)

    # Charts 6: prompt vs completion by agent
    if not tokens_by_agent.empty:
        top_agents = tokens_by_agent.sort_values("total_tokens", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(top_agents.index.astype(str), top_agents["prompt_tokens"], label="prompt_tokens")
        ax.bar(top_agents.index.astype(str), top_agents["completion_tokens"], bottom=top_agents["prompt_tokens"], label="completion_tokens")
        ax.set_title("Prompt vs completion tokens by agent")
        ax.set_xlabel("Agent")
        ax.set_ylabel("Tokens")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        save_plot(fig, paths.charts_dir / "06_prompt_vs_completion_by_agent.png")

    # Charts 7: total tokens by task
    if not tokens_by_task.empty:
        make_bar(top_n(tokens_by_task["total_tokens"], 12), "Total tokens by task", "Task", "Tokens", paths.charts_dir / "07_total_tokens_by_task.png", horizontal=True)

    # Charts 8: token histogram per llm call
    if not llm.empty:
        make_hist(llm["total_tokens"], 25, "Distribution of total tokens per LLM call", "Total tokens", "Calls", paths.charts_dir / "08_total_tokens_hist.png")

    # Charts 9: prompt/completion scatter per llm call
    if not llm.empty:
        sample = llm[["prompt_tokens", "completion_tokens", "agent_name_f"]].copy()
        make_scatter(sample, "prompt_tokens", "completion_tokens", "Prompt vs completion tokens per LLM call", "Prompt tokens", "Completion tokens", paths.charts_dir / "09_prompt_vs_completion_scatter.png")

    # Charts 10: cumulative tokens over time
    if not llm.empty and llm["ts"].notna().any():
        ts_df = llm[["ts", "prompt_tokens", "completion_tokens", "total_tokens"]].copy().sort_values("ts")
        ts_df["cum_prompt_tokens"] = ts_df["prompt_tokens"].cumsum()
        ts_df["cum_completion_tokens"] = ts_df["completion_tokens"].cumsum()
        ts_df["cum_total_tokens"] = ts_df["total_tokens"].cumsum()
        make_multi_line(ts_df, "ts", ["cum_prompt_tokens", "cum_completion_tokens", "cum_total_tokens"], "Cumulative tokens over time", "Time", "Tokens", paths.charts_dir / "10_cumulative_tokens_over_time.png")

    # Charts 11: cumulative tool calls and errors over time
    if df["ts"].notna().any():
        timeline = df[["ts", "type_f", "result_error"]].copy().sort_values("ts")
        timeline["tool_calls"] = (timeline["type_f"] == "metrics_tool_call").astype(int)
        timeline["tool_exc_errors"] = (timeline["type_f"] == "metrics_tool_exception_error").astype(int)
        timeline["tool_result_errors"] = ((timeline["type_f"] == "metrics_tool_result") & timeline["result_error"].fillna(False)).astype(int)
        timeline["cum_tool_calls"] = timeline["tool_calls"].cumsum()
        timeline["cum_tool_exc_errors"] = timeline["tool_exc_errors"].cumsum()
        timeline["cum_tool_result_errors"] = timeline["tool_result_errors"].cumsum()
        make_multi_line(timeline, "ts", ["cum_tool_calls", "cum_tool_exc_errors", "cum_tool_result_errors"], "Cumulative tool calls and errors over time", "Time", "Count", paths.charts_dir / "11_cumulative_tool_calls_errors.png")

    # Charts 12: error counts by tool
    if not err_table.empty:
        make_bar(top_n(err_table["total_errors"], 15), "Total errors by tool", "Tool", "Errors", paths.charts_dir / "12_errors_by_tool.png", horizontal=True)

    # Charts 13: error rate by tool (min calls filter)
    if not err_table.empty:
        filtered = err_table[err_table["calls"] >= 3].sort_values("error_rate", ascending=False)
        if not filtered.empty:
            make_bar(top_n(filtered["error_rate"], 15), "Tool error rate by tool (min 3 calls)", "Tool", "Error rate", paths.charts_dir / "13_error_rate_by_tool.png", horizontal=True)

    # Charts 14: result vs exception errors by tool
    if not err_table.empty:
        top_err = err_table.sort_values("total_errors", ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(top_err.index.astype(str), top_err["result_errors"], label="result_errors")
        ax.bar(top_err.index.astype(str), top_err["exception_errors"], bottom=top_err["result_errors"], label="exception_errors")
        ax.set_title("Result vs exception errors by tool")
        ax.set_xlabel("Tool")
        ax.set_ylabel("Errors")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        save_plot(fig, paths.charts_dir / "14_result_vs_exception_errors_by_tool.png")

    # Charts 15: calls heatmap agent x tool
    if not calls_by_agent_tool.empty:
        matrix = calls_by_agent_tool.copy()
        if matrix.shape[0] > 12:
            top_agents = matrix.sum(axis=1).sort_values(ascending=False).head(12).index
            matrix = matrix.loc[top_agents]
        if matrix.shape[1] > 15:
            top_tools_cols = matrix.sum(axis=0).sort_values(ascending=False).head(15).index
            matrix = matrix[top_tools_cols]
        make_heatmap(matrix, "Tool calls heatmap: agent × tool", "Tool", "Agent", paths.charts_dir / "15_calls_heatmap_agent_tool.png")

    # Charts 16: error heatmap agent x tool
    if not tool_results.empty or not tool_exc.empty:
        result_err_agent_tool = (
            tool_results[tool_results["result_error"].fillna(False)]
            .pivot_table(index="agent_name_f", columns="tool_name_f", values="row_id", aggfunc="count", fill_value=0)
        ) if not tool_results.empty else pd.DataFrame()
        exc_err_agent_tool = (
            tool_exc.pivot_table(index="agent_name_f", columns="tool_name_f", values="row_id", aggfunc="count", fill_value=0)
        ) if not tool_exc.empty else pd.DataFrame()
        err_heat = result_err_agent_tool.add(exc_err_agent_tool, fill_value=0)
        if not err_heat.empty:
            if err_heat.shape[0] > 12:
                top_agents = err_heat.sum(axis=1).sort_values(ascending=False).head(12).index
                err_heat = err_heat.loc[top_agents]
            if err_heat.shape[1] > 15:
                top_tools_cols = err_heat.sum(axis=0).sort_values(ascending=False).head(15).index
                err_heat = err_heat[top_tools_cols]
            make_heatmap(err_heat, "Tool error heatmap: agent × tool", "Tool", "Agent", paths.charts_dir / "16_error_heatmap_agent_tool.png")

    # Charts 17: tool calls by task
    if not tool_calls.empty:
        make_bar(top_n(tool_calls["task_name_f"].value_counts(), 12), "Tool calls by task", "Task", "Calls", paths.charts_dir / "17_tool_calls_by_task.png", horizontal=True)

    # Charts 18: tokens over iterations
    if not llm.empty and llm["iteration"].notna().any():
        toks_by_iter = llm.groupby("iteration")[["prompt_tokens", "completion_tokens", "total_tokens"]].sum().sort_index().reset_index()
        make_multi_line(toks_by_iter, "iteration", ["prompt_tokens", "completion_tokens", "total_tokens"], "Tokens by iteration", "Iteration", "Tokens", paths.charts_dir / "18_tokens_by_iteration.png")

    # Charts 19: tool calls over iterations
    if not tool_calls.empty and tool_calls["iteration"].notna().any():
        calls_by_iter = tool_calls.groupby("iteration").size().rename("tool_calls").reset_index()
        make_line(calls_by_iter, "iteration", "tool_calls", "Tool calls by iteration", "Iteration", "Calls", paths.charts_dir / "19_tool_calls_by_iteration.png")

    # Charts 20: average tokens per llm call by agent
    if not llm.empty:
        avg = llm.groupby("agent_name_f")["total_tokens"].mean().sort_values(ascending=False)
        make_bar(top_n(avg, 12), "Average total tokens per LLM call by agent", "Agent", "Avg tokens", paths.charts_dir / "20_avg_tokens_per_llm_call_by_agent.png", horizontal=True)

    # Charts 21: average tokens per task
    if not llm.empty:
        avg_task = llm.groupby("task_name_f")["total_tokens"].mean().sort_values(ascending=False)
        make_bar(top_n(avg_task, 12), "Average total tokens per LLM call by task", "Task", "Avg tokens", paths.charts_dir / "21_avg_tokens_per_task.png", horizontal=True)

    # Charts 22: token distribution by agent (boxplot)
    if not llm.empty:
        top_agents = llm.groupby("agent_name_f").size().sort_values(ascending=False).head(8).index
        groups = {agent: llm.loc[llm["agent_name_f"] == agent, "total_tokens"] for agent in top_agents}
        make_boxplot(groups, "Distribution of total tokens per LLM call by agent", "Agent", "Total tokens", paths.charts_dir / "22_token_boxplot_by_agent.png")

    # Charts 23: task summary calls vs tokens scatter
    if not tool_calls.empty or not llm.empty:
        task_tool_calls = tool_calls.groupby("task_name_f").size().rename("tool_calls") if not tool_calls.empty else pd.Series(dtype="int64")
        task_tokens = llm.groupby("task_name_f")["total_tokens"].sum().rename("total_tokens") if not llm.empty else pd.Series(dtype="float64")
        task_scatter = pd.concat([task_tool_calls, task_tokens], axis=1).fillna(0)
        if not task_scatter.empty:
            make_scatter(task_scatter.reset_index(), "tool_calls", "total_tokens", "Task-level tool calls vs total tokens", "Tool calls", "Total tokens", paths.charts_dir / "23_task_tool_calls_vs_tokens.png", label_col="task_name_f")

    # Charts 24: tool-level calls vs errors scatter
    if not err_table.empty:
        scatter = err_table.reset_index().rename(columns={"index": "tool_name"})
        make_scatter(scatter, "calls", "total_errors", "Tool-level calls vs total errors", "Calls", "Errors", paths.charts_dir / "24_tool_calls_vs_errors.png", label_col="tool_name_f" if "tool_name_f" in scatter.columns else None)

    # Charts 25: summary event agent counts over time
    if not summaries.empty and summaries["ts"].notna().any():
        temp = summaries[["ts", "payload"]].copy()
        temp["agent_count"] = temp["payload"].map(lambda p: len(p.get("agents", {})) if isinstance(p, dict) and isinstance(p.get("agents"), dict) else 0)
        make_line(temp, "ts", "agent_count", "Agent count in metrics_summary over time", "Time", "Agents", paths.charts_dir / "25_summary_agent_count_over_time.png")

    # Charts 26: event volume over time bucketed by minute
    if df["ts"].notna().any():
        temp = df[["ts"]].dropna().copy()
        temp["minute"] = temp["ts"].dt.floor("min")
        volume = temp.groupby("minute").size().rename("events").reset_index()
        make_line(volume, "minute", "events", "Event volume over time (per minute)", "Time", "Events", paths.charts_dir / "26_event_volume_per_minute.png")

    # Charts 27: model usage if multiple models
    if not llm.empty and llm["model"].dropna().nunique() > 0:
        make_bar(top_n(llm["model"].fillna("unknown_model").value_counts(), 10), "LLM usage by model", "Model", "Calls", paths.charts_dir / "27_llm_usage_by_model.png", horizontal=True)

    # Charts 28: thoughts vs cached tokens by agent
    if not tokens_by_agent.empty and ((tokens_by_agent["thoughts_tokens"] > 0).any() or (tokens_by_agent["cached_tokens"] > 0).any()):
        top_agents = tokens_by_agent.sort_values("total_tokens", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(top_agents.index.astype(str), top_agents["thoughts_tokens"], label="thoughts_tokens")
        ax.bar(top_agents.index.astype(str), top_agents["cached_tokens"], bottom=top_agents["thoughts_tokens"], label="cached_tokens")
        ax.set_title("Thoughts vs cached tokens by agent")
        ax.set_xlabel("Agent")
        ax.set_ylabel("Tokens")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        save_plot(fig, paths.charts_dir / "28_thoughts_vs_cached_tokens_by_agent.png")

    # Charts 29: tool share pie for top tools
    if not tool_calls.empty:
        pie = tool_calls["tool_name_f"].value_counts().head(8)
        if not pie.empty:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(pie.values, labels=pie.index.astype(str), autopct="%1.1f%%")
            ax.set_title("Tool call share (top 8 tools)")
            save_plot(fig, paths.charts_dir / "29_tool_call_share_pie.png")

    # Charts 30: calls per invocation
    if not tool_calls.empty:
        inv = tool_calls.groupby(safe_group_label(tool_calls["invocation_id"], "unknown_invocation")).size().sort_values(ascending=False)
        make_bar(top_n(inv, 15), "Tool calls by invocation", "Invocation", "Calls", paths.charts_dir / "30_tool_calls_by_invocation.png", horizontal=True)


def main() -> None:
    args = parse_args()
    paths = ensure_dirs(args.input, args.output_dir)
    records = load_jsonl(paths.input_file)
    df = normalize_records(records)
    analyze(df, paths)
    print(f"Done. Report written to: {paths.output_dir}")


if __name__ == "__main__":
    main()
