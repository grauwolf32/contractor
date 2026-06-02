#!/usr/bin/env python3
"""Run a contractor pipeline across multiple prompt/task version combinations.

Given one or more `--vary <name>=<v1>,<v2>,...` axes, the script Cartesian-products
them, temporarily pins each combo's versions in the relevant manifests, spawns
the CLI as a subprocess, restores the manifests, and finally produces a
Markdown comparison table.

Manifest pinning works at the `active:` line of:
  contractor/agents/<name>/prompt.yml         (when <name> is an agent)
  contractor/tasks/<name>.yml                 (when <name> is a task)

Axis names auto-detect — use `prompt:<name>` or `task:<name>` to disambiguate.

Usage:
    poetry run python scripts/eval_sweep.py \\
        --pipeline oas_build \\
        --project-path tests/playground/python/vulnyapi \\
        --model lm-studio-qwen3.6 \\
        --vary planning_agent=v4,v5 \\
        --vary oas_builder_agent=v3,v4 \\
        --out eval_runs/sweep-$(date +%F)

Combinations run sequentially; each lives in its own subdir under --out, with
its full metrics.jsonl, stdout.log, stderr.log, and (if produced) artifacts.
The final report is at <out>/sweep.md + <out>/sweep.json.
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = REPO_ROOT / "contractor" / "agents"
TASKS_DIR = REPO_ROOT / "contractor" / "tasks"

# Make scripts/diagnose.py importable as a module.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import diagnose  # type: ignore  # noqa: E402

# ─── Axis specification ───────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class Axis:
    kind: str             # "prompt" | "task"
    name: str             # agent / task name
    versions: list[str]   # versions to sweep
    manifest_path: Path   # absolute path to the manifest


def _parse_axis_head(head: str) -> tuple[str, str]:
    """Return (kind, name) for an axis head; auto-detect when no prefix."""
    if ":" in head:
        kind, name = head.split(":", 1)
        if kind not in ("prompt", "task"):
            raise ValueError(
                f"--vary kind must be 'prompt' or 'task', got {kind!r} (in {head!r})"
            )
        return kind, name

    prompt_path = AGENTS_DIR / head / "prompt.yml"
    task_path = TASKS_DIR / f"{head}.yml"
    found_kinds = []
    if prompt_path.exists():
        found_kinds.append("prompt")
    if task_path.exists():
        found_kinds.append("task")
    if len(found_kinds) == 0:
        raise ValueError(
            f"--vary {head}: no manifest found at {prompt_path} or {task_path}. "
            f"Use prompt:<name> or task:<name>."
        )
    if len(found_kinds) > 1:
        raise ValueError(
            f"--vary {head}: ambiguous — both a prompt and a task manifest exist. "
            f"Use prompt:{head} or task:{head}."
        )
    return found_kinds[0], head


def _manifest_for(kind: str, name: str) -> Path:
    if kind == "prompt":
        return AGENTS_DIR / name / "prompt.yml"
    return TASKS_DIR / f"{name}.yml"


def resolve_axis(spec: str) -> Axis:
    if "=" not in spec:
        raise ValueError(f"--vary expects 'name=v1,v2,...', got {spec!r}")
    head, vers_str = spec.split("=", 1)
    versions = [v.strip() for v in vers_str.split(",") if v.strip()]
    if not versions:
        raise ValueError(f"--vary {head}=  has no versions listed")

    kind, name = _parse_axis_head(head)
    manifest_path = _manifest_for(kind, name)

    # Validate every requested version is declared in the manifest.
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    declared = set((manifest.get("versions") or {}).keys())
    if "active" not in manifest:
        raise ValueError(f"{manifest_path}: missing 'active:' line")
    unknown = [v for v in versions if v not in declared]
    if unknown:
        available = ", ".join(sorted(declared)) or "(none)"
        raise ValueError(
            f"--vary {head}: version(s) {unknown!r} not declared in {manifest_path}. "
            f"Available: {available}"
        )

    return Axis(kind=kind, name=name, versions=versions, manifest_path=manifest_path)


# ─── Manifest pinning ─────────────────────────────────────────────────────────


def _replace_active_line(text: str, version: str) -> str:
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("active:"):
            lines[i] = f"active: {version}\n"
            return "".join(lines)
    raise ValueError("manifest has no `active:` line")


@contextmanager
def pin_manifest_versions(pins: dict[Path, str]) -> Iterator[None]:
    """Temporarily set ``active: <version>`` in each manifest; restore on exit.

    Backups are kept in memory; if the script is killed mid-run the on-disk
    manifest will be left in the pinned state — guard with try/finally where
    you call this. Use ``--restore-only`` to recover from a stale state.
    """
    backups: dict[Path, str] = {}
    try:
        for path, version in pins.items():
            original = path.read_text(encoding="utf-8")
            backups[path] = original
            path.write_text(_replace_active_line(original, version), encoding="utf-8")
        yield
    finally:
        for path, original in backups.items():
            try:
                path.write_text(original, encoding="utf-8")
            except OSError as exc:
                print(
                    f"warning: failed to restore {path}: {exc}", file=sys.stderr
                )


# ─── Combo execution ──────────────────────────────────────────────────────────


@dataclass
class RunResult:
    combo: dict[str, str]            # axis-name → version
    rep: int
    out_dir: str
    status: str                      # "ok" | "error" | "timeout"
    exit_code: int | None
    duration_s: float
    cmd: list[str]
    summary: dict[str, Any] = field(default_factory=dict)
    diagnose: dict[str, int] = field(default_factory=dict)


def _run_label(axes: list[Axis], combo: tuple[str, ...], rep: int, n_reps: int) -> str:
    parts = []
    for axis, ver in zip(axes, combo, strict=False):
        parts.append(f"{axis.kind[0]}{axis.name}-{ver}")
    base = "__".join(parts)
    return f"{base}__rep{rep}" if n_reps > 1 else base


def spawn_run(
    *,
    pipeline: str,
    project_path: Path,
    model: str,
    folder_name: str,
    artifact: Path | None,
    out_dir: Path,
    extra_args: list[str],
    timeout: int | None,
) -> tuple[str, int | None, float, list[str]]:
    """Spawn the CLI as a subprocess and capture stdout/stderr."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "cli.main",
        "--pipeline",
        pipeline,
        "--project-path",
        str(project_path),
        "--model",
        model,
        "--folder-name",
        folder_name,
        "--output",
        str(out_dir),
        "--no-ui",
        *extra_args,
    ]
    if artifact is not None:
        cmd.extend(["--artifact", str(artifact)])

    started = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        # subprocess.TimeoutExpired returns stdout/stderr as bytes even with
        # text=True (Python ≥3.6 quirk). Decode defensively.
        def _decode(b: Any) -> str:
            if b is None:
                return ""
            if isinstance(b, bytes):
                return b.decode("utf-8", errors="replace")
            return str(b)
        (out_dir / "stdout.log").write_text(_decode(exc.stdout), encoding="utf-8")
        (out_dir / "stderr.log").write_text(_decode(exc.stderr), encoding="utf-8")
        return "timeout", None, time.time() - started, cmd

    duration = time.time() - started
    (out_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (out_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
    status = "ok" if result.returncode == 0 else "error"
    return status, result.returncode, duration, cmd


def summarize_run(out_dir: Path) -> dict[str, Any]:
    """Aggregate token usage, tool stats, and task outcomes from metrics.jsonl."""
    metrics_path = out_dir / "metrics.jsonl"
    summary: dict[str, Any] = {
        "metrics_path": str(metrics_path),
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tool_calls": 0,
        "tool_errors": 0,
        "task_finished": 0,
        "task_failed": 0,
    }
    if not metrics_path.exists():
        summary["metrics_present"] = False
        return summary
    summary["metrics_present"] = True

    with metrics_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get("type")
            payload = rec.get("payload") or {}
            if t == "metrics_llm_usage":
                summary["total_tokens"] += int(payload.get("total_tokens") or 0)
                summary["prompt_tokens"] += int(payload.get("prompt_tokens") or 0)
                summary["completion_tokens"] += int(payload.get("completion_tokens") or 0)
            elif t == "metrics_tool_call":
                summary["tool_calls"] += 1
            elif t == "metrics_tool_result":
                if payload.get("result_error"):
                    summary["tool_errors"] += 1
                else:
                    result = payload.get("result")
                    if isinstance(result, dict) and "error" in result:
                        summary["tool_errors"] += 1
            elif t == "task_finished":
                summary["task_finished"] += 1
            elif t == "task_failed":
                summary["task_failed"] += 1
    return summary


def run_diagnose(metrics_path: Path) -> dict[str, int]:
    """Return ``{rule: hit_count}`` by calling diagnose detectors directly."""
    if not metrics_path.exists():
        return {}
    events = list(diagnose.iter_events(metrics_path))
    counts: dict[str, int] = {}
    for rule, detector in diagnose.DETECTORS.items():
        finding = detector(events)
        if finding is not None:
            counts[rule] = finding.count
    return counts


# ─── Report rendering ─────────────────────────────────────────────────────────


def _tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _err_rate(summary: dict[str, Any]) -> str:
    calls = summary.get("tool_calls") or 0
    if calls == 0:
        return "—"
    return f"{(summary.get('tool_errors', 0) / calls) * 100:.0f}%"


def render_report(
    runs: list[RunResult],
    axes: list[Axis],
    pipeline: str,
    project_path: Path,
    out_dir: Path,
) -> str:
    out = []
    out.append("# Contractor Eval Sweep Report\n")
    out.append(f"- Workflow: `{pipeline}`")
    out.append(f"- Project: `{project_path}`")
    out.append(f"- Output: `{out_dir}`")
    out.append(f"- Runs: **{len(runs)}**")
    out.append("- Axes:")
    for axis in axes:
        out.append(
            f"  - `{axis.kind}:{axis.name}` → {', '.join(f'`{v}`' for v in axis.versions)}"
        )
    out.append("")

    # ── Main comparison table ────────────────────────────────────────────────
    axis_cols = [f"{a.kind}:{a.name}" for a in axes]
    header = (
        ["#"]
        + axis_cols
        + ["status", "dur(s)", "tokens", "tool calls", "err%", "✓", "✗"]
    )
    out.append("## Combo Summary\n")
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join("---" for _ in header) + "|")
    for i, r in enumerate(runs):
        row = [str(i)]
        for a in axes:
            row.append(f"`{r.combo.get(a.name, '?')}`")
        row.append(r.status)
        row.append(f"{r.duration_s:.0f}")
        row.append(_tokens(r.summary.get("total_tokens", 0)))
        row.append(str(r.summary.get("tool_calls", 0)))
        row.append(_err_rate(r.summary))
        row.append(str(r.summary.get("task_finished", 0)))
        row.append(str(r.summary.get("task_failed", 0)))
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    # ── Diagnose findings table ──────────────────────────────────────────────
    all_rules: list[str] = []
    seen: set[str] = set()
    for r in runs:
        for rule in r.diagnose:
            if rule not in seen:
                seen.add(rule)
                all_rules.append(rule)
    if all_rules:
        out.append("## Diagnose Findings (counts per rule)\n")
        header = ["#"] + axis_cols + all_rules
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join("---" for _ in header) + "|")
        for i, r in enumerate(runs):
            row = [str(i)]
            for a in axes:
                row.append(f"`{r.combo.get(a.name, '?')}`")
            for rule in all_rules:
                row.append(str(r.diagnose.get(rule, 0)))
            out.append("| " + " | ".join(row) + " |")
        out.append("")

    # ── Per-run pointers ─────────────────────────────────────────────────────
    out.append("## Run Artifacts\n")
    for i, r in enumerate(runs):
        bits = ", ".join(f"`{k}={v}`" for k, v in r.combo.items())
        out.append(f"- **#{i}** ({bits}) → `{r.out_dir}`")
    out.append("")

    return "\n".join(out)


# ─── Main ─────────────────────────────────────────────────────────────────────


def _default_out_dir() -> Path:
    return REPO_ROOT / "eval_runs" / datetime.datetime.now().strftime("sweep-%Y%m%d-%H%M%S")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a contractor pipeline across multiple prompt/task version "
            "combinations and produce a comparison report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pipeline",
        required=True,
        help="Workflow name (oas_build, oas_update, trace, trace-direct, likec4, router).",
    )
    parser.add_argument(
        "--project-path",
        required=True,
        type=Path,
        help="Path to the project to analyse.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="LiteLLM model alias (must be configured in the LiteLLM proxy).",
    )
    parser.add_argument(
        "--folder-name",
        default="/",
        help="Project-relative folder path forwarded to the CLI (default: /).",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Optional --artifact path forwarded to the CLI (for enrich/trace).",
    )
    parser.add_argument(
        "--vary",
        action="append",
        required=True,
        metavar="NAME=V1,V2,...",
        help=(
            "Axis: agent or task name + versions. Repeatable. "
            "Use prompt:<name> or task:<name> to disambiguate."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for the sweep. Default: eval_runs/sweep-<timestamp>/",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run each combo N times. Default: 1.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-run timeout in seconds (default: no timeout).",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help=(
            "Extra arg(s) to forward verbatim to contractor (repeatable). "
            "Example: --extra-arg=--rm"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned combos and exit without running.",
    )
    args = parser.parse_args(argv)

    # ── Resolve axes ─────────────────────────────────────────────────────────
    try:
        axes = [resolve_axis(spec) for spec in args.vary]
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    seen_names: set[str] = set()
    for axis in axes:
        if axis.name in seen_names:
            print(f"error: --vary {axis.name} listed twice", file=sys.stderr)
            return 2
        seen_names.add(axis.name)

    combos = list(itertools.product(*[axis.versions for axis in axes]))
    total_runs = len(combos) * args.repeat

    out_dir = args.out or _default_out_dir()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("# Sweep plan")
    print(f"  pipeline:     {args.pipeline}")
    print(f"  project:      {args.project_path}")
    print(f"  model:        {args.model}")
    print(f"  out:          {out_dir}")
    print(f"  combos:       {len(combos)} × repeat={args.repeat} = {total_runs} runs")
    print("  axes:")
    for axis in axes:
        print(f"    {axis.kind}:{axis.name}  -> {axis.versions}  ({axis.manifest_path})")
    print("  combinations:")
    for i, combo in enumerate(combos):
        bits = ", ".join(f"{a.name}={v}" for a, v in zip(axes, combo, strict=False))
        print(f"    #{i}  {bits}")

    if args.dry_run:
        return 0

    # ── Execute ──────────────────────────────────────────────────────────────
    runs: list[RunResult] = []
    started_all = time.time()
    for _combo_idx, combo in enumerate(combos):
        pins = {axis.manifest_path: ver for axis, ver in zip(axes, combo, strict=False)}
        for rep in range(args.repeat):
            label = _run_label(axes, combo, rep, args.repeat)
            run_out = out_dir / label
            print(f"\n=== [{len(runs) + 1}/{total_runs}] {label} ===")
            for axis, ver in zip(axes, combo, strict=False):
                print(f"   pin {axis.kind}:{axis.name} → {ver}")

            with pin_manifest_versions(pins):
                status, exit_code, duration, cmd = spawn_run(
                    pipeline=args.pipeline,
                    project_path=args.project_path,
                    model=args.model,
                    folder_name=args.folder_name,
                    artifact=args.artifact,
                    out_dir=run_out,
                    extra_args=args.extra_arg,
                    timeout=args.timeout,
                )

            summary = summarize_run(run_out)
            diagnose_counts = run_diagnose(run_out / "metrics.jsonl")

            print(
                f"   status={status} exit={exit_code} dur={duration:.0f}s "
                f"tokens={_tokens(summary.get('total_tokens', 0))} "
                f"tool_calls={summary.get('tool_calls', 0)} "
                f"err_rate={_err_rate(summary)} "
                f"finished={summary.get('task_finished', 0)} failed={summary.get('task_failed', 0)}"
            )

            runs.append(
                RunResult(
                    combo={axis.name: ver for axis, ver in zip(axes, combo, strict=False)},
                    rep=rep,
                    out_dir=str(run_out),
                    status=status,
                    exit_code=exit_code,
                    duration_s=duration,
                    cmd=cmd,
                    summary=summary,
                    diagnose=diagnose_counts,
                )
            )

            # Save partial results after each combo for crash-safety.
            (out_dir / "sweep.json").write_text(
                json.dumps([asdict(r) for r in runs], indent=2, default=str),
                encoding="utf-8",
            )

    total_elapsed = time.time() - started_all
    print(f"\nSweep complete in {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")

    report = render_report(runs, axes, args.pipeline, args.project_path, out_dir)
    (out_dir / "sweep.md").write_text(report, encoding="utf-8")

    # Canonical eval/v1 envelope (scenario=pipeline, metric_kind=generic): each
    # swept combo is a case; it passes if the run finished with no failed task.
    from tests.eval.results import (
        CaseResult,
        EvalRun,
        FixtureResult,
        write_eval_results,
    )
    cases = []
    for r in runs:
        s = r.summary or {}
        label = "__".join(f"{k}={v}" for k, v in r.combo.items()) or "default"
        label = f"{label}#rep{r.rep}"
        passed = (r.status == "ok" and int(s.get("task_finished", 0) or 0) > 0
                  and int(s.get("task_failed", 0) or 0) == 0)
        cases.append(CaseResult(
            id=label, passed=passed, pass_count=int(passed), attempts=1,
            metrics={"total_tokens": int(s.get("total_tokens", 0) or 0),
                     "total_tool_calls": int(s.get("tool_calls", 0) or 0),
                     "duration_s": round(r.duration_s, 1)},
            detail={**r.combo, "status": r.status, "tool_errors": s.get("tool_errors"),
                    "diagnose": r.diagnose}))
    write_eval_results(
        EvalRun(scenario="pipeline", unit=args.pipeline, pass_at=1, metric_kind="generic",
                fixtures=[FixtureResult(slug=args.pipeline, cases=cases)],
                meta={"axes": [a.name for a in axes]}),
        out_dir,
    )

    print(f"\n{report}")
    print(f"\nReport: {out_dir / 'sweep.md'}")
    print(f"JSON:   {out_dir / 'sweep.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
