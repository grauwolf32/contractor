"""Unified A/B comparison CLI for eval runs.

Consolidates compare_eval_runs, compare_task_runs, probe_trace, and
probe_variance into subcommands sharing metric extraction and rendering.

Subcommands:
  trace     — fixture-based A/B trace agent (has ground truth)
  task      — A/B task pipeline with manifest pinning
  probe     — A/B trace agent on arbitrary project (no ground truth)
  variance  — multi-sample variance measurement for trace agent

Usage:
    poetry run python scripts/compare.py trace \\
        --fixture vulnyapi --case notes-search-sqli \\
        --a-prompt v5 --b-prompt v7 --b-graph-tools

    poetry run python scripts/compare.py task \\
        --fixture vulnyapi --pipeline project_information \\
        --a-pin planning_agent=v5 --b-pin planning_agent=v6

    poetry run python scripts/compare.py probe \\
        --project-path ./target --file /src/app.py --function main \\
        --a-prompt v5 --b-prompt v7 --b-graph-tools

    poetry run python scripts/compare.py variance \\
        --project-path ./target --file /src/app.py --function main \\
        --samples 5 --a-prompt v5 --b-prompt v7 --b-graph-tools
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Shared: model setup
# ---------------------------------------------------------------------------


def _load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / "cli" / ".env", override=False)


def _get_model():
    override = os.environ.get("CONTRACTOR_EVAL_MODEL")
    if override:
        from google.adk.models.lite_llm import LiteLlm

        return LiteLlm(model=override, timeout=600)
    from contractor.utils.settings import DEFAULT_MODEL

    return DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Shared: metric extraction
# ---------------------------------------------------------------------------


def extract_trace_metrics(run) -> dict[str, Any]:
    """Extract metrics from a TraceAgentRun's plugin events."""
    events = run.agent_run.metrics_events or []
    tool_counts: Counter[str] = Counter()
    totals = {
        "total_tool_calls": 0,
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "tool_time_ms": 0.0,
        "args_bytes": 0,
        "result_bytes": 0,
    }

    for ev in events:
        kind = ev.get("event_type", "")
        if kind == "tool_call":
            name = ev.get("tool_name", "")
            if name:
                tool_counts[name] += 1
                totals["total_tool_calls"] += 1
            totals["args_bytes"] += int(ev.get("arguments_size", 0) or 0)
        elif kind == "tool_result":
            totals["tool_time_ms"] += float(ev.get("execution_time_ms", 0) or 0)
            totals["result_bytes"] += int(ev.get("result_size", 0) or 0)
        elif kind == "llm_usage":
            usage = ev.get("usage", {})
            totals["llm_calls"] += 1
            totals["input_tokens"] += int(usage.get("input", 0) or 0)
            totals["output_tokens"] += int(usage.get("output", 0) or 0)
            totals["total_tokens"] += int(usage.get("total", 0) or 0)

    if not events:
        for tc in run.agent_run.tool_calls:
            tool_counts[tc.name] += 1
            totals["total_tool_calls"] += 1

    return {"tool_counts": dict(tool_counts), **totals}


def extract_task_metrics(run) -> dict[str, Any]:
    """Extract aggregate metrics from a TaskAgentRun."""
    totals = {
        "total_tool_calls": 0,
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "tool_time_ms": 0.0,
        "args_bytes": 0,
        "result_bytes": 0,
    }
    per_task: dict[str, dict] = {}
    for ref, m in run.metrics.items():
        totals["total_tool_calls"] += m.total_tool_calls
        totals["llm_calls"] += m.llm_calls
        totals["input_tokens"] += m.input_tokens
        totals["output_tokens"] += m.output_tokens
        totals["total_tokens"] += m.total_tokens
        totals["tool_time_ms"] += m.tool_time_ms
        totals["args_bytes"] += m.args_bytes
        totals["result_bytes"] += m.result_bytes
        per_task[ref] = {
            "total_tool_calls": m.total_tool_calls,
            "llm_calls": m.llm_calls,
            "input_tokens": m.input_tokens,
            "output_tokens": m.output_tokens,
            "tool_time_ms": m.tool_time_ms,
            "tool_counts": dict(m.tool_counts),
        }
    artifact_sizes = {key: len(text) for key, text in run.artifacts.items()}
    return {**totals, "per_task": per_task, "artifact_sizes": artifact_sizes}


# ---------------------------------------------------------------------------
# Shared: table rendering
# ---------------------------------------------------------------------------

_METRIC_KEYS = (
    "total_tool_calls",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "tool_time_ms",
    "args_bytes",
    "result_bytes",
)


def render_ab_table(a: dict, b: dict, *, extra_keys: list[str] | None = None) -> str:
    """Render metric name | A | B | delta table."""
    keys = list(_METRIC_KEYS) + (extra_keys or [])
    lines = [f"{'metric':30s} {'A':>16s} {'B':>16s}  delta", "-" * 80]
    for key in keys:
        va, vb = a.get(key, 0), b.get(key, 0)
        if isinstance(va, float) or isinstance(vb, float):
            lines.append(f"{key:30s} {va:16.1f} {vb:16.1f}  {vb - va:+.1f}")
        else:
            lines.append(f"{key:30s} {va:16d} {vb:16d}  {vb - va:+d}")
    return "\n".join(lines)


def render_tool_counts_table(a_counts: dict, b_counts: dict) -> str:
    """Render per-tool call count comparison."""
    all_tools = sorted(set(a_counts) | set(b_counts))
    if not all_tools:
        return ""
    lines = [f"\n{'tool':30s} {'A':>8s} {'B':>8s}", "-" * 50]
    for tool in all_tools:
        va, vb = a_counts.get(tool, 0), b_counts.get(tool, 0)
        lines.append(f"{tool:30s} {va:8d} {vb:8d}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared: variance aggregation
# ---------------------------------------------------------------------------


def aggregate_samples(samples: list[dict]) -> dict[str, dict[str, float]]:
    """Compute mean/stdev/min/max for each numeric metric across samples."""
    if not samples:
        return {}
    result: dict[str, dict[str, float]] = {}
    for key in _METRIC_KEYS:
        values = [float(s.get(key, 0)) for s in samples]
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0
        result[key] = {
            "mean": mean,
            "stdev": math.sqrt(variance),
            "min": min(values),
            "max": max(values),
        }
    return result


def render_variance_table(agg: dict[str, dict[str, float]], label: str) -> str:
    """Render variance stats for one side."""
    lines = [f"\n# {label}", f"{'metric':24s} {'mean':>10s} {'stdev':>8s} {'min':>10s} {'max':>10s}"]
    lines.append("-" * 66)
    for key, stats in agg.items():
        lines.append(
            f"{key:24s} {stats['mean']:10.1f} {stats['stdev']:8.1f} "
            f"{stats['min']:10.1f} {stats['max']:10.1f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared: manifest pinning
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _pin_manifest_active(manifest_path: Path, version: str):
    """Temporarily rewrite ``active: <ver>`` in a prompt/task manifest."""
    if version is None:
        yield
        return
    original = manifest_path.read_text(encoding="utf-8")
    pinned = re.sub(
        r"^active:\s*\S+", f"active: {version}", original, count=1, flags=re.MULTILINE
    )
    if pinned == original:
        print(
            f"warning: {manifest_path} has no `active:` line; "
            f"version={version} ignored",
            file=sys.stderr,
        )
        yield
        return
    manifest_path.write_text(pinned, encoding="utf-8")
    try:
        yield
    finally:
        manifest_path.write_text(original, encoding="utf-8")


def pin_versions(versions: dict[str, str | None]):
    """Stack manifest pins for ``{name: version}`` dict.

    Name formats: ``task:<key>``, ``prompt:<agent>``, or bare name
    (auto-discovers task first, then agent).
    """
    stack = contextlib.ExitStack()
    for name, ver in versions.items():
        if not ver:
            continue
        kind, _, ident = name.partition(":")
        if not ident:
            kind, ident = "auto", name
        candidates: list[Path] = []
        if kind in ("task", "auto"):
            candidates.append(REPO_ROOT / "contractor" / "tasks" / f"{ident}.yml")
        if kind in ("prompt", "auto"):
            candidates.append(REPO_ROOT / "contractor" / "agents" / ident / "prompt.yml")
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            raise SystemExit(f"could not locate manifest for {name!r}")
        stack.enter_context(_pin_manifest_active(found, ver))
    return stack


# ---------------------------------------------------------------------------
# Shared: trace user message
# ---------------------------------------------------------------------------


def build_trace_message(*, file: str, function: str, intent: str) -> str:
    return (
        f"Trace the request flow that begins at `{function}` in `{file}`. "
        f"{intent} "
        "Insert `# @trace target=... args=... calls=...` comments above each "
        "function definition you confidently identify as part of the path. "
        "Use the `insert_line` tool to mutate files. Stop once the path is "
        "covered."
    )


# ---------------------------------------------------------------------------
# Subcommand: trace
# ---------------------------------------------------------------------------


async def _cmd_trace(args) -> int:
    from tests.eval.conftest import select_fixture
    from tests.eval.trace_harness import run_trace_agent

    fixture = select_fixture(args.fixture)
    if fixture is None:
        print(f"unknown fixture: {args.fixture}", file=sys.stderr)
        return 2

    matches = [c for c in fixture.trace_cases if c["id"] == args.case]
    if not matches:
        print(
            f"case {args.case!r} not found in {args.fixture}; "
            f"available: {[c['id'] for c in fixture.trace_cases]}",
            file=sys.stderr,
        )
        return 2
    case = matches[0]

    model = _get_model()
    entry = case["entrypoint"]
    msg = build_trace_message(
        file=entry.get("file", "?"),
        function=entry.get("function") or entry.get("route") or "?",
        intent=case.get("intent", "Annotate every function on the relevant flow."),
    )

    async def _run(label, prompt, graph_tools):
        return await run_trace_agent(
            fixture_root=fixture.source_root,
            user_message=msg,
            model=model,
            namespace=f"compare-{args.fixture}-{args.case}-{label}",
            timeout_s=args.timeout,
            prompt_version=prompt,
            with_graph_tools=graph_tools,
        )

    run_a = await _run("A", args.a_prompt, args.a_graph_tools)
    run_b = await _run("B", args.b_prompt, args.b_graph_tools)

    ma, mb = extract_trace_metrics(run_a), extract_trace_metrics(run_b)

    expected = {
        (item["file"], item["function"]) for item in case["expected_annotated"]
    }
    actual_a = {a.as_tuple() for a in run_a.annotations}
    actual_b = {a.as_tuple() for a in run_b.annotations}

    ma["precision"] = len(actual_a & expected) / len(actual_a) if actual_a else 0
    ma["recall"] = len(actual_a & expected) / len(expected) if expected else 1
    ma["annotations"] = len(run_a.annotations)
    mb["precision"] = len(actual_b & expected) / len(actual_b) if actual_b else 0
    mb["recall"] = len(actual_b & expected) / len(expected) if expected else 1
    mb["annotations"] = len(run_b.annotations)

    print(f"# trace: fixture={args.fixture} case={args.case}")
    print(f"A: prompt={args.a_prompt} graph_tools={args.a_graph_tools}")
    print(f"B: prompt={args.b_prompt} graph_tools={args.b_graph_tools}")
    print()
    print(render_ab_table(ma, mb, extra_keys=["precision", "recall", "annotations"]))
    print(render_tool_counts_table(ma.get("tool_counts", {}), mb.get("tool_counts", {})))

    if args.output:
        report = {"fixture": args.fixture, "case": args.case, "a": ma, "b": mb}
        args.output.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nreport written to {args.output}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: task
# ---------------------------------------------------------------------------


async def _cmd_task(args) -> int:
    from functools import partial

    from tests.eval.conftest import select_fixture
    from tests.eval.task_harness import run_task_pipeline

    fixture = select_fixture(args.fixture)
    if fixture is None:
        print(f"unknown fixture: {args.fixture}", file=sys.stderr)
        return 2

    model = _get_model()

    async def _run_pipeline(label, pinned):
        from cli.fs import RootedLocalFileSystem
        from contractor.agents.swe_agent.agent import build_swe_agent

        fs = RootedLocalFileSystem(str(fixture.source_root))

        if args.pipeline == "project_information":

            def queue(runner) -> None:
                swe = partial(build_swe_agent, name="swe_agent", fs=fs, model=model, max_tokens=100_000)
                runner.add_variable(name="project_path", value=str(fixture.source_root))
                runner.add_task(
                    name="dependency_information", worker_builder=swe,
                    iterations=1, max_attempts=2, max_steps=20,
                    namespace="dependency_information", model=model,
                )
                runner.add_task(
                    name="project_information", worker_builder=swe,
                    iterations=1, max_attempts=2, max_steps=20,
                    artifacts=["dependency_information/result"],
                    namespace="project_information", model=model,
                )

            artifact_keys = ["dependency_information/result", "project_information/result"]
            timeout = 1800.0

        elif args.pipeline == "likec4":
            from contractor.agents.likec4_builder_agent.agent import (
                build_likec4_builder_agent,
            )
            from contractor.tools.fs import MemoryOverlayFileSystem

            overlay = MemoryOverlayFileSystem(fs=fs)

            def queue(runner) -> None:
                swe = partial(build_swe_agent, name="swe_agent", fs=fs, model=model, max_tokens=100_000)
                lc4 = partial(build_likec4_builder_agent, name="likec4_builder", fs=overlay, model=model, max_tokens=120_000)
                runner.add_variable(name="project_path", value=str(fixture.source_root))
                runner.add_task(name="dependency_information", worker_builder=swe, iterations=1, max_attempts=2, max_steps=20, namespace="dependency_information", model=model)
                runner.add_task(name="project_information", worker_builder=swe, iterations=1, max_attempts=2, max_steps=20, artifacts=["dependency_information/result"], namespace="project_information", model=model)
                runner.add_task(name="likec4_build", worker_builder=lc4, iterations=3, max_attempts=6, max_steps=20, artifacts=["dependency_information/result", "project_information/result"], namespace="likec4-building", model=model)

            artifact_keys = ["dependency_information/result", "project_information/result", "likec4_build/result"]
            timeout = 2400.0
        else:
            print(f"unknown pipeline: {args.pipeline}", file=sys.stderr)
            return 2

        with pin_versions(pinned):
            return await run_task_pipeline(
                queue_fn=queue,
                artifact_keys=artifact_keys,
                namespace=f"compare-{args.pipeline}-{fixture.slug}-{label}",
                runner_name=f"compare-{args.pipeline}-{label}",
                timeout_s=timeout,
            )

    def _parse_pin(raw_list):
        out = {}
        for raw in raw_list or []:
            if "=" not in raw:
                raise SystemExit(f"bad --pin {raw!r}; expected name=version")
            name, ver = raw.split("=", 1)
            out[name.strip()] = ver.strip() or None
        return out

    run_a = await _run_pipeline("A", _parse_pin(args.a_pin))
    run_b = await _run_pipeline("B", _parse_pin(args.b_pin))

    ma, mb = extract_task_metrics(run_a), extract_task_metrics(run_b)

    print(f"# task: pipeline={args.pipeline} fixture={args.fixture}")
    print(f"A pinned: {args.a_pin}")
    print(f"B pinned: {args.b_pin}")
    print()
    print(render_ab_table(ma, mb))

    # Per-task breakdown
    all_refs = sorted(set(ma.get("per_task", {})) | set(mb.get("per_task", {})))
    if all_refs:
        print(f"\n{'task_ref':35s} {'A':>10s} {'B':>10s}")
        print("-" * 60)
        for ref in all_refs:
            va = ma.get("per_task", {}).get(ref, {}).get("total_tool_calls", 0)
            vb = mb.get("per_task", {}).get(ref, {}).get("total_tool_calls", 0)
            print(f"{ref:35s} {va:10d} {vb:10d}")

    # Artifact sizes
    all_arts = sorted(set(ma.get("artifact_sizes", {})) | set(mb.get("artifact_sizes", {})))
    if all_arts:
        print(f"\n{'artifact':40s} {'A':>10s} {'B':>10s}")
        print("-" * 65)
        for k in all_arts:
            va = ma.get("artifact_sizes", {}).get(k, 0)
            vb = mb.get("artifact_sizes", {}).get(k, 0)
            print(f"{k:40s} {va:10d} {vb:10d}")

    if args.output:
        report = {"fixture": args.fixture, "pipeline": args.pipeline, "a": ma, "b": mb}
        args.output.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nreport written to {args.output}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: probe
# ---------------------------------------------------------------------------


async def _run_probe_once(*, project_path, msg, model, label, prompt, graph_tools, timeout):
    from tests.eval.trace_harness import run_trace_agent

    run = await run_trace_agent(
        fixture_root=project_path,
        user_message=msg,
        model=model,
        namespace=f"probe-{label}",
        timeout_s=timeout,
        prompt_version=prompt,
        with_graph_tools=graph_tools,
    )
    metrics = extract_trace_metrics(run)
    metrics["annotations"] = len(run.annotations)
    metrics["files_modified"] = len(run.modified_files)
    metrics["annotation_list"] = [a.as_tuple() for a in run.annotations]
    metrics["modified_file_list"] = sorted(run.modified_files)
    return metrics


async def _cmd_probe(args) -> int:
    model = _get_model()
    project_path = Path(args.project_path).resolve()
    if not project_path.is_dir():
        print(f"not a directory: {project_path}", file=sys.stderr)
        return 2

    msg = build_trace_message(
        file=args.file, function=args.function, intent=args.intent,
    )

    ma = await _run_probe_once(
        project_path=project_path, msg=msg, model=model,
        label="A", prompt=args.a_prompt, graph_tools=args.a_graph_tools,
        timeout=args.timeout,
    )
    mb = await _run_probe_once(
        project_path=project_path, msg=msg, model=model,
        label="B", prompt=args.b_prompt, graph_tools=args.b_graph_tools,
        timeout=args.timeout,
    )

    print(f"# probe: {project_path.name} {args.file}::{args.function}")
    print(f"A: prompt={args.a_prompt} graph_tools={args.a_graph_tools}")
    print(f"B: prompt={args.b_prompt} graph_tools={args.b_graph_tools}")
    print()
    print(render_ab_table(ma, mb, extra_keys=["annotations", "files_modified"]))
    print(render_tool_counts_table(ma.get("tool_counts", {}), mb.get("tool_counts", {})))

    if args.output:
        report = {"project": str(project_path), "a": ma, "b": mb}
        args.output.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nreport written to {args.output}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: variance
# ---------------------------------------------------------------------------


async def _cmd_variance(args) -> int:
    model = _get_model()
    project_path = Path(args.project_path).resolve()
    if not project_path.is_dir():
        print(f"not a directory: {project_path}", file=sys.stderr)
        return 2

    msg = build_trace_message(
        file=args.file, function=args.function, intent=args.intent,
    )

    a_samples: list[dict] = []
    b_samples: list[dict] = []

    for i in range(1, args.samples + 1):
        print(f"  sample {i}/{args.samples} ...", file=sys.stderr, end=" ")
        ma = await _run_probe_once(
            project_path=project_path, msg=msg, model=model,
            label=f"A-{i}", prompt=args.a_prompt, graph_tools=args.a_graph_tools,
            timeout=args.timeout,
        )
        mb = await _run_probe_once(
            project_path=project_path, msg=msg, model=model,
            label=f"B-{i}", prompt=args.b_prompt, graph_tools=args.b_graph_tools,
            timeout=args.timeout,
        )
        a_samples.append(ma)
        b_samples.append(mb)
        print(
            f"A: {ma['annotations']} ann / {ma['total_tokens']} tok  "
            f"B: {mb['annotations']} ann / {mb['total_tokens']} tok",
            file=sys.stderr,
        )

    agg_a = aggregate_samples(a_samples)
    agg_b = aggregate_samples(b_samples)

    print(f"# variance: {project_path.name} {args.file}::{args.function}  N={args.samples}")
    print(f"A: prompt={args.a_prompt} graph_tools={args.a_graph_tools}")
    print(f"B: prompt={args.b_prompt} graph_tools={args.b_graph_tools}")
    print(render_variance_table(agg_a, f"A (prompt={args.a_prompt} graph={args.a_graph_tools})"))
    print(render_variance_table(agg_b, f"B (prompt={args.b_prompt} graph={args.b_graph_tools})"))

    # Delta summary
    print("\n# A vs B delta (mean)")
    print(f"{'metric':24s} {'A':>10s} {'B':>10s} {'delta':>10s}")
    print("-" * 60)
    for key in _METRIC_KEYS:
        va = agg_a[key]["mean"]
        vb = agg_b[key]["mean"]
        delta = vb - va
        print(f"{key:24s} {va:10.1f} {vb:10.1f} {delta:+10.1f}")

    if args.output:
        report = {
            "project": str(project_path),
            "samples": args.samples,
            "a_samples": a_samples,
            "b_samples": b_samples,
            "a_aggregate": agg_a,
            "b_aggregate": agg_b,
        }
        args.output.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nreport written to {args.output}")
    return 0


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _add_trace_args(parser: argparse.ArgumentParser) -> None:
    """Add common A/B trace agent arguments."""
    parser.add_argument("--a-prompt", default="v5")
    parser.add_argument("--a-graph-tools", action="store_true", default=False)
    parser.add_argument("--b-prompt", default="v7")
    parser.add_argument("--b-graph-tools", action="store_true", default=True)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--output", type=Path, default=None)


def _add_probe_args(parser: argparse.ArgumentParser) -> None:
    """Add probe-specific arguments (arbitrary project)."""
    parser.add_argument("--project-path", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--function", required=True)
    parser.add_argument("--intent", default="Annotate every function on the relevant flow.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Unified A/B comparison CLI for eval runs.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # -- trace --
    p_trace = sub.add_parser("trace", help="Fixture-based A/B trace comparison")
    p_trace.add_argument("--fixture", default="vulnyapi")
    p_trace.add_argument("--case", required=True)
    _add_trace_args(p_trace)

    # -- task --
    p_task = sub.add_parser("task", help="A/B task pipeline with manifest pinning")
    p_task.add_argument("--fixture", default="vulnyapi")
    p_task.add_argument("--pipeline", choices=["project_information", "likec4"], default="project_information")
    p_task.add_argument("--a-pin", action="append", default=[], help="name=version pin for A")
    p_task.add_argument("--b-pin", action="append", default=[], help="name=version pin for B")
    p_task.add_argument("--output", type=Path, default=None)

    # -- probe --
    p_probe = sub.add_parser("probe", help="A/B trace on arbitrary project (no ground truth)")
    _add_probe_args(p_probe)
    _add_trace_args(p_probe)

    # -- variance --
    p_var = sub.add_parser("variance", help="Multi-sample variance measurement")
    _add_probe_args(p_var)
    _add_trace_args(p_var)
    p_var.add_argument("--samples", type=int, default=3)

    args = ap.parse_args()
    _load_env()

    dispatch = {
        "trace": _cmd_trace,
        "task": _cmd_task,
        "probe": _cmd_probe,
        "variance": _cmd_variance,
    }
    return asyncio.run(dispatch[args.command](args))


if __name__ == "__main__":
    raise SystemExit(main())
