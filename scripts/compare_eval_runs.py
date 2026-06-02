"""Compare two ``trace_agent`` configurations head-to-head on a single
eval fixture/case.

Used to A/B prompt versions and the ``with_graph_tools`` opt-in. Each
side runs ``run_trace_agent`` with its own ``prompt_version`` and
``with_graph_tools`` knob; the script prints a side-by-side table of
tool counts, token usage (input/output), tool execution time, and
precision/recall against the case's ground truth.

Usage:
    poetry run python scripts/compare_eval_runs.py \\
        --fixture vulnyapi --case notes-search-sqli \\
        --a-prompt v5 \\
        --b-prompt v7 --b-graph-tools
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


async def _run_one(
    *,
    label: str,
    fixture_root: Path,
    user_message: str,
    case: dict[str, Any],
    model,
    prompt_version: str | None,
    with_graph_tools: bool,
) -> dict[str, Any]:
    from tests.eval.trace_harness import run_trace_agent

    namespace = f"compare-{case['id']}-{label}"
    result = await run_trace_agent(
        fixture_root=fixture_root,
        user_message=user_message,
        model=model,
        namespace=namespace,
        timeout_s=float(case.get("timeout_s", 900.0)),
        prompt_version=prompt_version,
        with_graph_tools=with_graph_tools,
    )

    # Ground-truth tool counts come from AdkMetricsPlugin's tool_call
    # events (captures every call regardless of how ADK packaged it).
    # Fall back to the harness's ADK-event reader when the plugin
    # wasn't attached.
    metrics_events = result.agent_run.metrics_events
    tc_events = [e for e in metrics_events if str(e.get("event_type")) == "tool_call"]
    tr_events = [e for e in metrics_events if str(e.get("event_type")) == "tool_result"]
    llm_events = [e for e in metrics_events if str(e.get("event_type")) == "llm_usage"]
    if tc_events:
        tool_counts = Counter(e["tool_name"] for e in tc_events)
    else:
        tool_counts = Counter(c.name for c in result.agent_run.tool_calls)

    total_input_tokens = sum(int(e.get("usage", {}).get("input", 0)) for e in llm_events)
    total_output_tokens = sum(int(e.get("usage", {}).get("output", 0)) for e in llm_events)
    total_tokens = sum(int(e.get("usage", {}).get("total", 0)) for e in llm_events)
    total_tool_time_ms = sum(float(e.get("execution_time_ms", 0)) for e in tr_events)
    total_args_bytes = sum(int(e.get("arguments_size", 0)) for e in tc_events)
    total_result_bytes = sum(int(e.get("result_size", 0)) for e in tr_events)

    expected = {
        (item["file"], item["function"]) for item in case["expected_annotated"]
    }
    actual = {(a.file, a.function) for a in result.annotations}
    tp = len(expected & actual)
    precision = tp / len(actual) if actual else 0.0
    recall = tp / len(expected) if expected else 0.0

    return {
        "label": label,
        "prompt_version": result.prompt_version,
        "with_graph_tools": with_graph_tools,
        "tool_counts": dict(tool_counts),
        "total_tool_calls": sum(tool_counts.values()),
        "llm_calls": len(llm_events),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "tool_time_ms": total_tool_time_ms,
        "args_bytes": total_args_bytes,
        "result_bytes": total_result_bytes,
        "annotations": [(a.file, a.function) for a in sorted(result.annotations, key=lambda x: (x.file, x.function))],
        "expected": sorted(expected),
        "modified_files": sorted(result.modified_files),
        "precision": precision,
        "recall": recall,
    }


def _user_message(case: dict[str, Any]) -> str:
    entry = case["entrypoint"]
    where = entry.get("file", "?")
    func = entry.get("function") or entry.get("route") or "?"
    intent = case.get("intent", "Annotate every function on the relevant flow.")
    return (
        f"Trace the request flow that begins at `{func}` in `{where}`. "
        f"{intent} "
        "Insert `# @trace target=... args=... calls=...` comments above each "
        "function definition you confidently identify as part of the path. "
        "Use the `insert_line` tool to mutate files. Stop once the path is "
        "covered."
    )


def _render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    a = report["a"]
    b = report["b"]
    lines.append(f"# case: {report['case_id']}  fixture: {report['fixture']}")
    lines.append(
        f"A: prompt={a['prompt_version']} graph_tools={a['with_graph_tools']}"
    )
    lines.append(
        f"B: prompt={b['prompt_version']} graph_tools={b['with_graph_tools']}"
    )
    lines.append("")
    lines.append(f"{'metric':30s} {'A':>16s} {'B':>16s}  delta")
    lines.append("-" * 80)
    metrics = [
        ("total tool calls", a["total_tool_calls"], b["total_tool_calls"]),
        ("llm calls", a["llm_calls"], b["llm_calls"]),
        ("input tokens", a["input_tokens"], b["input_tokens"]),
        ("output tokens", a["output_tokens"], b["output_tokens"]),
        ("total tokens", a["total_tokens"], b["total_tokens"]),
        ("tool time (ms)", a["tool_time_ms"], b["tool_time_ms"]),
        ("args size (bytes)", a["args_bytes"], b["args_bytes"]),
        ("result size (bytes)", a["result_bytes"], b["result_bytes"]),
        ("precision", a["precision"], b["precision"]),
        ("recall", a["recall"], b["recall"]),
        ("annotated count", len(a["annotations"]), len(b["annotations"])),
        ("files modified", len(a["modified_files"]), len(b["modified_files"])),
    ]
    for name, va, vb in metrics:
        if isinstance(va, float):
            lines.append(f"{name:30s} {va:16.1f} {vb:16.1f}  {vb - va:+.1f}")
        else:
            lines.append(f"{name:30s} {va:16d} {vb:16d}  {vb - va:+d}")

    lines.append("")
    lines.append("## tool call counts")
    all_tools = sorted(set(a["tool_counts"]) | set(b["tool_counts"]))
    lines.append(f"{'tool':30s} {'A':>16s} {'B':>16s}")
    lines.append("-" * 65)
    for tool in all_tools:
        va = a["tool_counts"].get(tool, 0)
        vb = b["tool_counts"].get(tool, 0)
        lines.append(f"{tool:30s} {va:16d} {vb:16d}")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="vulnyapi")
    ap.add_argument("--case", required=True, help="trace-case id")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--a-prompt", default="v5")
    ap.add_argument(
        "--a-graph-tools", action="store_true",
        help="attach graph tools to side A",
    )
    ap.add_argument("--b-prompt", default="v7")
    ap.add_argument(
        "--b-graph-tools", action="store_true", default=True,
        help="attach graph tools to side B (default true)",
    )
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / "cli" / ".env", override=False)

    from tests.eval.conftest import select_fixture

    fixture = select_fixture(args.fixture)
    if fixture is None:
        print(f"unknown fixture: {args.fixture}", file=sys.stderr)
        return 2

    matches = [c for c in fixture.trace_cases if c["id"] == args.case]
    if not matches:
        print(f"unknown case id: {args.case}", file=sys.stderr)
        print(
            "available:",
            [c["id"] for c in fixture.trace_cases],
            file=sys.stderr,
        )
        return 2
    case = matches[0]

    from contractor.utils.settings import DEFAULT_MODEL

    override = os.environ.get("CONTRACTOR_EVAL_MODEL")
    if override:
        from google.adk.models.lite_llm import LiteLlm

        model = LiteLlm(model=override, timeout=600)
    else:
        model = DEFAULT_MODEL

    user_msg = _user_message(case)

    async def _both() -> tuple[dict[str, Any], dict[str, Any]]:
        a = await _run_one(
            label="A",
            fixture_root=fixture.source_root,
            user_message=user_msg,
            case=case,
            model=model,
            prompt_version=args.a_prompt,
            with_graph_tools=args.a_graph_tools,
        )
        b = await _run_one(
            label="B",
            fixture_root=fixture.source_root,
            user_message=user_msg,
            case=case,
            model=model,
            prompt_version=args.b_prompt,
            with_graph_tools=args.b_graph_tools,
        )
        return a, b

    a, b = asyncio.run(_both())

    report = {
        "fixture": args.fixture,
        "case_id": args.case,
        "a": a,
        "b": b,
    }

    text = _render(report)
    print(text)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nfull report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
