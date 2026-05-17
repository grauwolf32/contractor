"""A/B two task-pipeline configurations head-to-head on a single fixture.

Same shape as ``compare_eval_runs.py`` but for the planner+worker
chains exercised by ``tests/eval/test_*_task_eval.py``. Each side runs
the chosen pipeline through ``run_task_pipeline`` with its own task /
prompt versions pinned; the script prints a side-by-side table of tool
counts, token usage (input/output), tool execution time, and (for
project_information) section / phrase recall, or (for likec4) DSL
validity + keyword recall.

Usage:
    poetry run python scripts/compare_task_runs.py \\
        --fixture vulnyapi --pipeline project_information \\
        --a-task-version v1 --b-task-version v1 \\
        --b-planner-version v6

The ``--a-*`` / ``--b-*`` overrides are passed straight to the prompt /
task loaders, which temporarily pin the requested version via the
existing version-resolution machinery (``load_prompt_with_version`` /
``TaskTemplate.load(version=...)``). Per-pipeline knobs (which tasks
to queue, which artifacts to score) live below in ``_PIPELINES``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@contextlib.contextmanager
def _pin_manifest_active(manifest_path: Path, version: str):
    """Temporarily rewrite ``active: <ver>`` in a prompt/task manifest.

    The TaskRunner / load_prompt_with_version path reads the manifest
    each call, so swapping the ``active:`` line is enough to pin a
    different version for the duration of one run. Restores the file
    afterwards (even on exception).
    """
    if version is None:
        yield
        return
    original = manifest_path.read_text(encoding="utf-8")
    pinned = re.sub(
        r"^active:\s*\S+", f"active: {version}", original, count=1, flags=re.MULTILINE
    )
    if pinned == original:
        # No ``active:`` line found — leave the file untouched but warn.
        print(
            f"warning: {manifest_path} has no `active:` line; "
            f"--pin-version={version} ignored",
            file=sys.stderr,
        )
        yield
        return
    manifest_path.write_text(pinned, encoding="utf-8")
    try:
        yield
    finally:
        manifest_path.write_text(original, encoding="utf-8")


def _pin_versions(versions: dict[str, Optional[str]]):
    """Stack manifest pins for a whole {name: version} dict.

    ``name`` is either ``task:<key>`` (rewrites
    ``contractor/tasks/<key>.yml``) or ``prompt:<agent>`` (rewrites
    ``contractor/agents/<agent>/prompt.yml``). Bare names fall back to
    task-first, then agent.
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
            candidates.append(
                REPO_ROOT / "contractor" / "agents" / ident / "prompt.yml"
            )
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            raise SystemExit(f"could not locate manifest for {name!r}")
        stack.enter_context(_pin_manifest_active(found, ver))
    return stack


async def _run_project_information(label: str, fixture, model, namespace_suffix: str):
    from contractor.agents.swe_agent.agent import build_swe_agent

    from tests.eval.task_harness import run_task_pipeline
    from cli.fs import RootedLocalFileSystem

    fs = RootedLocalFileSystem(str(fixture.source_root))

    def queue(runner) -> None:
        swe_builder = partial(
            build_swe_agent, name="swe_agent", fs=fs, model=model, max_tokens=100_000
        )
        runner.add_variable(name="project_path", value=str(fixture.source_root))
        runner.add_task(
            name="dependency_information",
            worker_builder=swe_builder,
            iterations=1, max_attempts=2, max_steps=20,
            namespace="dependency_information", model=model,
        )
        runner.add_task(
            name="project_information",
            worker_builder=swe_builder,
            iterations=1, max_attempts=2, max_steps=20,
            artifacts=["dependency_information/result"],
            namespace="project_information", model=model,
        )

    return await run_task_pipeline(
        queue_fn=queue,
        artifact_keys=[
            "dependency_information/result",
            "project_information/result",
        ],
        namespace=f"compare-project_information-{namespace_suffix}-{label}",
        runner_name=f"compare-project_information-{label}",
        timeout_s=1800.0,
    )


async def _run_likec4(label: str, fixture, model, namespace_suffix: str):
    from contractor.agents.likec4_builder_agent.agent import \
        build_likec4_builder_agent
    from contractor.agents.swe_agent.agent import build_swe_agent
    from contractor.tools.fs import MemoryOverlayFileSystem

    from tests.eval.task_harness import run_task_pipeline
    from cli.fs import RootedLocalFileSystem

    fs = RootedLocalFileSystem(str(fixture.source_root))
    overlay = MemoryOverlayFileSystem(fs=fs)

    def queue(runner) -> None:
        swe_builder = partial(
            build_swe_agent, name="swe_agent", fs=fs, model=model, max_tokens=100_000
        )
        likec4_builder = partial(
            build_likec4_builder_agent,
            name="likec4_builder", fs=overlay, model=model, max_tokens=120_000,
        )
        runner.add_variable(name="project_path", value=str(fixture.source_root))
        runner.add_task(
            name="dependency_information", worker_builder=swe_builder,
            iterations=1, max_attempts=2, max_steps=20,
            namespace="dependency_information", model=model,
        )
        runner.add_task(
            name="project_information", worker_builder=swe_builder,
            iterations=1, max_attempts=2, max_steps=20,
            artifacts=["dependency_information/result"],
            namespace="project_information", model=model,
        )
        runner.add_task(
            name="likec4_build", worker_builder=likec4_builder,
            iterations=3, max_attempts=6, max_steps=20,
            artifacts=[
                "dependency_information/result",
                "project_information/result",
            ],
            namespace="likec4-building", model=model,
        )
        runner.add_task(
            name="likec4_validate", worker_builder=likec4_builder,
            iterations=1, max_attempts=2, max_steps=20,
            artifacts=[
                "dependency_information/result",
                "project_information/result",
                "likec4_build/result",
            ],
            namespace="likec4-building", model=model,
        )

    return await run_task_pipeline(
        queue_fn=queue,
        artifact_keys=[
            "dependency_information/result",
            "project_information/result",
            "likec4_build/result",
            "likec4_validate/result",
        ],
        namespace=f"compare-likec4-{namespace_suffix}-{label}",
        runner_name=f"compare-likec4-{label}",
        timeout_s=2400.0,
    )


_PIPELINES = {
    "project_information": _run_project_information,
    "likec4": _run_likec4,
}


async def _run_side(
    *,
    label: str,
    fixture,
    model,
    pipeline: str,
    pinned: dict[str, Optional[str]],
) -> dict[str, Any]:
    with _pin_versions(pinned):
        runner_fn = _PIPELINES[pipeline]
        run = await runner_fn(label, fixture, model, namespace_suffix=fixture.slug)

    totals = {
        "total_tool_calls": sum(m.total_tool_calls for m in run.metrics.values()),
        "llm_calls": sum(m.llm_calls for m in run.metrics.values()),
        "input_tokens": sum(m.input_tokens for m in run.metrics.values()),
        "output_tokens": sum(m.output_tokens for m in run.metrics.values()),
        "total_tokens": sum(m.total_tokens for m in run.metrics.values()),
        "tool_time_ms": sum(m.tool_time_ms for m in run.metrics.values()),
        "args_bytes": sum(m.args_bytes for m in run.metrics.values()),
        "result_bytes": sum(m.result_bytes for m in run.metrics.values()),
    }
    per_task = {
        ref: {
            "total_tool_calls": m.total_tool_calls,
            "llm_calls": m.llm_calls,
            "input_tokens": m.input_tokens,
            "output_tokens": m.output_tokens,
            "tool_time_ms": m.tool_time_ms,
            "tool_counts": dict(m.tool_counts),
        }
        for ref, m in run.metrics.items()
    }
    artifact_sizes = {key: len(text) for key, text in run.artifacts.items()}

    return {
        "label": label,
        "pinned": pinned,
        "totals": totals,
        "per_task": per_task,
        "artifact_sizes": artifact_sizes,
    }


def _render(report: dict[str, Any]) -> str:
    a, b = report["a"], report["b"]
    lines: list[str] = []
    lines.append(f"# pipeline: {report['pipeline']}  fixture: {report['fixture']}")
    lines.append(f"A pinned: {a['pinned']}")
    lines.append(f"B pinned: {b['pinned']}")
    lines.append("")
    lines.append(f"{'metric':30s} {'A':>16s} {'B':>16s}  delta")
    lines.append("-" * 80)
    for key in (
        "total_tool_calls",
        "llm_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "tool_time_ms",
        "args_bytes",
        "result_bytes",
    ):
        va, vb = a["totals"][key], b["totals"][key]
        if isinstance(va, float) or isinstance(vb, float):
            lines.append(f"{key:30s} {va:16.1f} {vb:16.1f}  {vb - va:+.1f}")
        else:
            lines.append(f"{key:30s} {va:16d} {vb:16d}  {vb - va:+d}")

    lines.append("")
    lines.append("## per-task tool calls")
    all_refs = sorted(set(a["per_task"]) | set(b["per_task"]))
    lines.append(f"{'task_ref':35s} {'A':>10s} {'B':>10s}")
    lines.append("-" * 60)
    for ref in all_refs:
        va = a["per_task"].get(ref, {}).get("total_tool_calls", 0)
        vb = b["per_task"].get(ref, {}).get("total_tool_calls", 0)
        lines.append(f"{ref:35s} {va:10d} {vb:10d}")

    lines.append("")
    lines.append("## artifact sizes (chars)")
    all_keys = sorted(set(a["artifact_sizes"]) | set(b["artifact_sizes"]))
    lines.append(f"{'artifact':40s} {'A':>10s} {'B':>10s}")
    lines.append("-" * 65)
    for k in all_keys:
        va = a["artifact_sizes"].get(k, 0)
        vb = b["artifact_sizes"].get(k, 0)
        lines.append(f"{k:40s} {va:10d} {vb:10d}")
    return "\n".join(lines)


def _parse_pin(arg: list[str] | None) -> dict[str, Optional[str]]:
    out: dict[str, Optional[str]] = {}
    for raw in arg or []:
        if "=" not in raw:
            raise SystemExit(f"bad --pin {raw!r}; expected name=version")
        name, ver = raw.split("=", 1)
        out[name.strip()] = ver.strip() or None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="vulnyapi")
    ap.add_argument(
        "--pipeline", choices=sorted(_PIPELINES), default="project_information"
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--a-pin", action="append", default=[],
        help="Version pin for side A, e.g. task:project_information=v1",
    )
    ap.add_argument(
        "--b-pin", action="append", default=[],
        help="Version pin for side B (same syntax as --a-pin).",
    )
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / "cli" / ".env", override=False)

    from tests.eval.conftest import select_fixture

    fixture = select_fixture(args.fixture)
    if fixture is None:
        print(f"unknown fixture: {args.fixture}", file=sys.stderr)
        return 2

    if args.pipeline == "likec4":
        if shutil.which("likec4") is None and not any(
            shutil.which(r) for r in ("bunx", "pnpx", "npx")
        ):
            print(
                "likec4 CLI not on PATH and no JS runner available; "
                "the likec4 pipeline will fail mid-run.",
                file=sys.stderr,
            )

    from contractor.utils.settings import DEFAULT_MODEL

    override = os.environ.get("CONTRACTOR_EVAL_MODEL")
    if override:
        from google.adk.models.lite_llm import LiteLlm
        model = LiteLlm(model=override, timeout=600)
    else:
        model = DEFAULT_MODEL

    async def _both():
        a = await _run_side(
            label="A", fixture=fixture, model=model,
            pipeline=args.pipeline, pinned=_parse_pin(args.a_pin),
        )
        b = await _run_side(
            label="B", fixture=fixture, model=model,
            pipeline=args.pipeline, pinned=_parse_pin(args.b_pin),
        )
        return a, b

    a, b = asyncio.run(_both())
    report = {
        "fixture": args.fixture,
        "pipeline": args.pipeline,
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
