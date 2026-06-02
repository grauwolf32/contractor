"""Run ``probe_trace`` N times and aggregate (mean / std / min / max).

Each invocation runs both A and B sides on the same project + entrypoint,
so N samples → 2N LLM runs. Picks up the same CLI flags as
``probe_trace.py`` so a single command can swap from a one-shot probe
to a variance band.

Usage:
    poetry run python scripts/probe_variance.py \\
        --project-path tests/playground/cloud/cloud-core \\
        --file /core/Controller/LoginController.php \\
        --function tryLogin \\
        --a-prompt v5 \\
        --b-prompt v7 --b-graph-tools \\
        --samples 3 \\
        --output eval_runs/variance_login.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


_METRIC_KEYS: tuple[str, ...] = (
    "total_tool_calls",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "tool_time_ms",
    "args_bytes",
    "result_bytes",
)


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/std/min/max per numeric metric across N samples."""
    summary: dict[str, Any] = {}
    for key in _METRIC_KEYS:
        values = [float(s.get(key, 0) or 0) for s in samples]
        if not values:
            continue
        summary[key] = {
            "mean": statistics.fmean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    # Annotation / files-modified counts collapse to set-of-tuples and
    # union sizes — useful to see whether the agent always lands on the
    # same locations.
    ann_seen: set[tuple[str, str]] = set()
    file_seen: set[str] = set()
    ann_counts: list[int] = []
    file_counts: list[int] = []
    for s in samples:
        anns = {(a[0], a[1]) for a in s.get("annotations", [])}
        files = set(s.get("modified_files", []))
        ann_seen |= anns
        file_seen |= files
        ann_counts.append(len(anns))
        file_counts.append(len(files))
    summary["annotations_count"] = {
        "mean": statistics.fmean(ann_counts) if ann_counts else 0,
        "stdev": statistics.stdev(ann_counts) if len(ann_counts) > 1 else 0,
        "min": min(ann_counts) if ann_counts else 0,
        "max": max(ann_counts) if ann_counts else 0,
    }
    summary["files_modified_count"] = {
        "mean": statistics.fmean(file_counts) if file_counts else 0,
        "stdev": statistics.stdev(file_counts) if len(file_counts) > 1 else 0,
        "min": min(file_counts) if file_counts else 0,
        "max": max(file_counts) if file_counts else 0,
    }
    summary["unique_annotations_across_runs"] = sorted(ann_seen)
    summary["unique_files_across_runs"] = sorted(file_seen)
    return summary


def _user_message(file: str, function: str, intent: str) -> str:
    return (
        f"Trace the request flow that begins at `{function}` in `{file}`. "
        f"{intent} "
        "Insert `# @trace target=... args=... calls=...` comments above each "
        "function definition you confidently identify as part of the path. "
        "Use the `insert_line` tool to mutate files. Stop once the path is "
        "covered."
    )


def _format_row(name: str, agg: dict[str, Any]) -> str:
    return (
        f"{name:24s} mean={agg['mean']:10.1f}  stdev={agg['stdev']:8.1f}  "
        f"min={agg['min']:10.1f}  max={agg['max']:10.1f}"
    )


def _render(label: str, samples: list[dict[str, Any]], agg: dict[str, Any]) -> str:
    lines = [f"# {label}   N={len(samples)}"]
    for key in _METRIC_KEYS:
        if key in agg:
            lines.append(_format_row(key, agg[key]))
    lines.append(_format_row("annotations", agg["annotations_count"]))
    lines.append(_format_row("files_modified", agg["files_modified_count"]))
    lines.append(
        f"unique_files_across_runs: {agg['unique_files_across_runs']}"
    )
    lines.append("")
    return "\n".join(lines)


def _delta(a: dict, b: dict) -> str:
    lines = ["# A vs B mean-on-mean delta"]
    for key in _METRIC_KEYS + ("annotations_count", "files_modified_count"):
        if key in a and key in b:
            ma, mb = a[key]["mean"], b[key]["mean"]
            pct = "n/a" if ma == 0 else f"{(mb - ma) / ma * 100:+.1f}%"
            lines.append(
                f"  {key:24s} A={ma:10.1f}  B={mb:10.1f}  Δ={mb-ma:+10.1f}  ({pct})"
            )
    return "\n".join(lines)


async def main_async(args) -> int:
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

    from scripts.probe_trace import _run_one  # reuse

    user_msg = _user_message(args.file, args.function, args.intent)

    a_samples: list[dict[str, Any]] = []
    b_samples: list[dict[str, Any]] = []
    for i in range(args.samples):
        print(
            f"\n=== sample {i+1}/{args.samples} ===", file=sys.stderr, flush=True
        )
        a = await _run_one(
            label=f"A-{i+1}",
            project_root=project,
            user_message=user_msg,
            prompt_version=args.a_prompt,
            with_graph_tools=args.a_graph_tools,
            model=model,
            timeout_s=args.timeout,
        )
        b = await _run_one(
            label=f"B-{i+1}",
            project_root=project,
            user_message=user_msg,
            prompt_version=args.b_prompt,
            with_graph_tools=args.b_graph_tools,
            model=model,
            timeout_s=args.timeout,
        )
        print(
            f"  A: ann={len(a['annotations'])} tokens={a['total_tokens']}  "
            f"B: ann={len(b['annotations'])} tokens={b['total_tokens']}",
            file=sys.stderr,
            flush=True,
        )
        a_samples.append(a)
        b_samples.append(b)

    a_agg = _aggregate(a_samples)
    b_agg = _aggregate(b_samples)

    print(_render(f"A (prompt={args.a_prompt} graph={args.a_graph_tools})", a_samples, a_agg))
    print(_render(f"B (prompt={args.b_prompt} graph={args.b_graph_tools})", b_samples, b_agg))
    print(_delta(a_agg, b_agg))

    if args.output:
        args.output.write_text(json.dumps({
            "a_samples": a_samples,
            "b_samples": b_samples,
            "a_aggregate": a_agg,
            "b_aggregate": b_agg,
        }, indent=2, default=str))
        print(f"\nfull report written to {args.output}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-path", type=Path, required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--function", required=True)
    ap.add_argument("--intent", default="Annotate every function on the relevant flow.")
    ap.add_argument("--a-prompt", default="v5")
    ap.add_argument("--a-graph-tools", action="store_true")
    ap.add_argument("--b-prompt", default="v7")
    ap.add_argument("--b-graph-tools", action="store_true", default=True)
    ap.add_argument("--timeout", type=float, default=1500.0)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
