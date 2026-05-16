"""Probe trace_agent on an arbitrary project + entrypoint, without a
fixture / ground-truth.

Runs two configurations of trace_agent back to back on the same
project + entrypoint and prints a side-by-side report of tool counts,
LLM token usage, and execution time. Useful for measuring the impact
of a prompt or tool-set change on real codebases where we don't have a
labelled trace_case.

Usage:
    poetry run python scripts/probe_trace.py \\
        --project-path tests/playground/cloud/cloud-core \\
        --file /core/Controller/LoginController.php \\
        --function tryLogin

Both runs:
    A: trace_agent v5, no graph tools (baseline)
    B: trace_agent v6 (or another version), opt-in graph tools

Override per side via --a-prompt / --a-graph-tools / --b-prompt /
--b-graph-tools flags.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _user_message(file: str, function: str, intent: str) -> str:
    return (
        f"Trace the request flow that begins at `{function}` in `{file}`. "
        f"{intent} "
        "Insert `# @trace target=... args=... calls=...` comments above each "
        "function definition you confidently identify as part of the path. "
        "Use the `insert_line` tool to mutate files. Stop once the path is "
        "covered."
    )


async def _run_one(
    *,
    label: str,
    project_root: Path,
    user_message: str,
    prompt_version: Optional[str],
    with_graph_tools: bool,
    model,
    timeout_s: float,
) -> dict[str, Any]:
    from tests.eval.trace_harness import run_trace_agent

    namespace = f"probe-{label}"
    result = await run_trace_agent(
        fixture_root=project_root,
        user_message=user_message,
        model=model,
        namespace=namespace,
        timeout_s=timeout_s,
        prompt_version=prompt_version,
        with_graph_tools=with_graph_tools,
    )

    events = result.agent_run.metrics_events
    tc_events = [e for e in events if str(e.get("event_type")) == "tool_call"]
    tr_events = [e for e in events if str(e.get("event_type")) == "tool_result"]
    llm_events = [e for e in events if str(e.get("event_type")) == "llm_usage"]
    if tc_events:
        tool_counts = Counter(e["tool_name"] for e in tc_events)
    else:
        tool_counts = Counter(c.name for c in result.agent_run.tool_calls)

    return {
        "label": label,
        "prompt_version": result.prompt_version,
        "with_graph_tools": with_graph_tools,
        "tool_counts": dict(tool_counts),
        "total_tool_calls": sum(tool_counts.values()),
        "llm_calls": len(llm_events),
        "input_tokens": sum(int(e.get("usage", {}).get("input", 0)) for e in llm_events),
        "output_tokens": sum(int(e.get("usage", {}).get("output", 0)) for e in llm_events),
        "total_tokens": sum(int(e.get("usage", {}).get("total", 0)) for e in llm_events),
        "tool_time_ms": sum(float(e.get("execution_time_ms", 0)) for e in tr_events),
        "args_bytes": sum(int(e.get("arguments_size", 0)) for e in tc_events),
        "result_bytes": sum(int(e.get("result_size", 0)) for e in tr_events),
        "annotations": [(a.file, a.function) for a in sorted(result.annotations, key=lambda x: (x.file, x.function))],
        "modified_files": sorted(result.modified_files),
    }


def _render(a: dict[str, Any], b: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# probe: trace_agent A vs B")
    lines.append(f"A: prompt={a['prompt_version']} graph_tools={a['with_graph_tools']}")
    lines.append(f"B: prompt={b['prompt_version']} graph_tools={b['with_graph_tools']}")
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
    ap.add_argument("--project-path", type=Path, required=True)
    ap.add_argument("--file", required=True, help="entrypoint file (virtual /relative)")
    ap.add_argument("--function", required=True, help="entrypoint function/method name")
    ap.add_argument("--intent", default="Annotate every function on the relevant flow.")
    ap.add_argument("--a-prompt", default="v5")
    ap.add_argument("--a-graph-tools", action="store_true")
    ap.add_argument("--b-prompt", default="v6")
    ap.add_argument("--b-graph-tools", action="store_true", default=True)
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / "cli" / ".env", override=False)

    project = args.project_path.resolve()
    if not project.is_dir():
        print(f"not a directory: {project}", file=sys.stderr)
        return 2

    from contractor.utils.settings import DEFAULT_MODEL

    override = os.environ.get("CONTRACTOR_EVAL_MODEL")
    if override:
        from google.adk.models.lite_llm import LiteLlm
        model = LiteLlm(model=override, timeout=600)
    else:
        model = DEFAULT_MODEL

    user_msg = _user_message(args.file, args.function, args.intent)

    async def _both() -> tuple[dict[str, Any], dict[str, Any]]:
        a = await _run_one(
            label="A",
            project_root=project,
            user_message=user_msg,
            prompt_version=args.a_prompt,
            with_graph_tools=args.a_graph_tools,
            model=model,
            timeout_s=args.timeout,
        )
        b = await _run_one(
            label="B",
            project_root=project,
            user_message=user_msg,
            prompt_version=args.b_prompt,
            with_graph_tools=args.b_graph_tools,
            model=model,
            timeout_s=args.timeout,
        )
        return a, b

    a, b = asyncio.run(_both())

    print(_render(a, b))

    if args.output:
        args.output.write_text(json.dumps({"a": a, "b": b}, indent=2, default=str))
        print(f"\nfull report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
