#!/usr/bin/env python3
"""Surface concrete failure-mode fingerprints from contractor metrics.jsonl.

Reads one or more metrics.jsonl files (or directories containing them) and
emits a Markdown report listing detected anti-patterns: default_tool routing
errors, tool-retry streaks, suspect upsert args, premature finish, etc.

Usage:
    python scripts/diagnose.py path/to/metrics.jsonl [more.jsonl ...]
    python scripts/diagnose.py path/to/project_dir   # auto-finds .contractor/metrics.jsonl
    python scripts/diagnose.py --rule upsert_args path/to/run/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ─── Event model ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Event:
    source: str
    type: str
    task_name: str
    task_id: int
    payload: dict[str, Any]

    @property
    def session_id(self) -> str | None:
        return self.payload.get("session_id") or _get(self.payload, "session_id")

    @property
    def invocation_id(self) -> str | None:
        return self.payload.get("invocation_id")

    @property
    def agent_name(self) -> str | None:
        return self.payload.get("agent_name")

    @property
    def tool_name(self) -> str | None:
        return self.payload.get("tool_name")

    @property
    def args_hash(self) -> str | None:
        return self.payload.get("args_hash")


def _get(d: dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default


def iter_events(path: Path) -> Iterator[Event]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield Event(
                source=str(path),
                type=rec.get("type") or "?",
                task_name=rec.get("task_name") or "?",
                task_id=rec.get("task_id") if rec.get("task_id") is not None else -1,
                payload=rec.get("payload") or {},
            )


def collect_paths(inputs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for raw in inputs:
        if raw.is_file():
            out.append(raw)
        elif raw.is_dir():
            out.extend(sorted(raw.rglob("metrics.jsonl")))
        else:
            print(f"warning: {raw} is not a file or directory", file=sys.stderr)
    return out


# ─── Detector helpers ─────────────────────────────────────────────────────────


def _is_error_result(payload: dict[str, Any]) -> bool:
    """True if a metrics_tool_result payload represents a tool error."""
    if payload.get("result_error"):
        return True
    result = payload.get("result")
    if isinstance(result, dict):
        if "error" in result:
            return True
        # Some tools nest the error inside {"result": {...}}
        nested = result.get("result")
        if isinstance(nested, dict) and "error" in nested:
            return True
    return False


def _truncate(value: Any, n: int = 120) -> str:
    s = repr(value) if not isinstance(value, str) else value
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[: n - 1] + "…"


def _format_loc(ev: Event) -> str:
    """Short locator: task::iteration::session::invocation."""
    bits = [str(ev.task_name)]
    iteration = ev.payload.get("iteration")
    if iteration is not None:
        bits.append(f"iter={iteration}")
    if ev.session_id:
        bits.append(f"sess={ev.session_id[:8]}")
    return " · ".join(bits)


# ─── Detection rules ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class Finding:
    rule: str
    severity: str          # "high" | "medium" | "low"
    summary: str           # one-line summary
    detail: str            # multi-line detail (Markdown)
    count: int = 0


def detect_default_tool_calls(events: list[Event]) -> Finding | None:
    hits = [
        e for e in events
        if e.type in ("metrics_tool_call", "metrics_tool_result")
        and e.tool_name == "default_tool"
    ]
    # Deduplicate by (call, session, invocation) since call + result are paired.
    seen: set[tuple] = set()
    unique = []
    for e in hits:
        key = (e.session_id, e.invocation_id, e.args_hash, e.type)
        if key in seen:
            continue
        seen.add(key)
        unique.append(e)
    if not unique:
        return None

    examples = []
    for e in unique[:10]:
        if e.type == "metrics_tool_call":
            meta = (e.payload.get("tool_args") or {}).get("meta") or {}
            func = meta.get("func_name") or "?"
            examples.append(f"- `{_format_loc(e)}`  intended `{func}`")
    return Finding(
        rule="default_tool_calls",
        severity="high",
        summary=f"{len(unique)} default_tool routing(s) — malformed tool calls that the harness deflected.",
        detail=(
            "Each occurrence means the model called a tool name that does not exist or passed "
            "non-dict arguments. Fix the originating prompt or the model's tool selection.\n\n"
            + "\n".join(examples)
            + ("\n…" if len(unique) > 10 else "")
        ),
        count=len(unique),
    )


def detect_retry_streaks(
    events: list[Event], min_streak: int = 3
) -> Finding | None:
    """Same (agent, tool, args_hash) repeated N+ times consecutively per session."""
    by_session: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        if e.type != "metrics_tool_call":
            continue
        sid = e.session_id or "?"
        by_session[sid].append(e)

    streaks: list[tuple[int, Event]] = []  # (streak_length, first_event_of_streak)
    for _sid, evs in by_session.items():
        cur_key: tuple = ()
        cur_start: Event | None = None
        cur_len = 0
        for e in evs:
            k = (e.agent_name, e.tool_name, e.args_hash)
            if k == cur_key:
                cur_len += 1
            else:
                if cur_len >= min_streak and cur_start is not None:
                    streaks.append((cur_len, cur_start))
                cur_key = k
                cur_start = e
                cur_len = 1
        if cur_len >= min_streak and cur_start is not None:
            streaks.append((cur_len, cur_start))

    if not streaks:
        return None

    streaks.sort(key=lambda t: t[0], reverse=True)
    examples = []
    for streak_len, ev in streaks[:10]:
        tool = ev.tool_name or "?"
        agent = ev.agent_name or "?"
        examples.append(
            f"- `{_format_loc(ev)}`  agent=`{agent}` tool=`{tool}` repeated **{streak_len}×**"
        )
    return Finding(
        rule="tool_retry_streaks",
        severity="medium",
        summary=f"{len(streaks)} streaks of identical tool calls (≥{min_streak} consecutive).",
        detail=(
            "Models stuck in a loop with identical args. Often a sign that an elided result was "
            "interpreted as a failure, or that the model is unable to make progress on a path. "
            "Compare against the RepeatedToolCallCallback threshold (5).\n\n"
            + "\n".join(examples)
            + ("\n…" if len(streaks) > 10 else "")
        ),
        count=len(streaks),
    )


def detect_elided_repeats(events: list[Event]) -> Finding | None:
    """Same heavy tool with same args returning {"elided": true} ≥2 times."""
    seen: dict[tuple, int] = defaultdict(int)
    first: dict[tuple, Event] = {}
    for e in events:
        if e.type != "metrics_tool_result":
            continue
        result = e.payload.get("result")
        if not isinstance(result, dict) or not result.get("elided"):
            continue
        key = (e.session_id, e.agent_name, e.tool_name, e.args_hash)
        seen[key] += 1
        if key not in first:
            first[key] = e

    repeats = {k: c for k, c in seen.items() if c >= 2}
    if not repeats:
        return None

    examples = []
    for key, count in sorted(repeats.items(), key=lambda kv: -kv[1])[:10]:
        ev = first[key]
        tool = ev.tool_name or "?"
        agent = ev.agent_name or "?"
        examples.append(
            f"- `{_format_loc(ev)}`  agent=`{agent}` tool=`{tool}` elided **{count}×**"
        )
    return Finding(
        rule="elided_result_repeats",
        severity="medium",
        summary=f"{len(repeats)} heavy-tool args returned elided results ≥2 times.",
        detail=(
            "When `read_file`/`grep`/`glob`/`list_symbols` returns `{\"elided\": true}` the call "
            "succeeded but the body was dropped to save context. Repeating with identical args "
            "re-elides — the model needs to narrow the range or change the query.\n\n"
            + "\n".join(examples)
            + ("\n…" if len(repeats) > 10 else "")
        ),
        count=len(repeats),
    )


_BANNED_EXTS = (".yaml", ".yml", ".json", ".md")


def detect_suspect_upsert_args(events: list[Event]) -> Finding | None:
    """OAS builder anti-patterns documented in oas_builder_agent prompt."""
    issues: list[tuple[str, Event, str]] = []
    for e in events:
        if e.type != "metrics_tool_call":
            continue
        if e.tool_name not in ("upsert_path", "upsert_component"):
            continue
        args = e.payload.get("tool_args") or {}

        # 1. *_def passed as a string instead of a dict
        for key in ("path_def", "component_def"):
            val = args.get(key)
            if isinstance(val, str) and val.strip()[:1] in ("{", "["):
                issues.append((f"{key} passed as JSON string", e, _truncate(val, 80)))

        # 2. provenance file with banned extension
        for key in ("path_files", "component_files"):
            files = args.get(key) or []
            if isinstance(files, list):
                bad = [f for f in files if isinstance(f, str) and f.lower().endswith(_BANNED_EXTS)]
                if bad:
                    issues.append(
                        (f"{key} contains banned extension", e, ", ".join(bad))
                    )

        # 3. x-path-files / x-component-files embedded inside *_def
        for key, bad_key in (
            ("path_def", "x-path-files"),
            ("component_def", "x-component-files"),
        ):
            val = args.get(key)
            if isinstance(val, dict) and bad_key in val:
                issues.append(
                    (f"{bad_key} embedded inside {key}", e, _truncate(val.get(bad_key), 80))
                )

    if not issues:
        return None
    examples = [
        f"- `{_format_loc(ev)}`  **{kind}** — `{detail}`"
        for kind, ev, detail in issues[:15]
    ]
    by_kind = Counter(kind for kind, _, _ in issues)
    return Finding(
        rule="suspect_upsert_args",
        severity="high",
        summary=f"{len(issues)} suspect upsert_path/upsert_component arg shapes.",
        detail=(
            "Each item is a violation of the OAS builder contract documented in "
            "`contractor/agents/oas_builder_agent/prompts/v4.md` (HARD RULES 2-4).\n\n"
            "**By kind:** "
            + ", ".join(f"`{k}`={v}" for k, v in by_kind.most_common())
            + "\n\n"
            + "\n".join(examples)
            + ("\n…" if len(issues) > 15 else "")
        ),
        count=len(issues),
    )


def detect_premature_finish(events: list[Event]) -> Finding | None:
    """`finish` calls that returned an error (e.g. open subtasks remaining)."""
    hits: list[Event] = []
    for e in events:
        if e.type != "metrics_tool_result":
            continue
        if e.tool_name != "finish":
            continue
        if _is_error_result(e.payload):
            hits.append(e)
    if not hits:
        return None
    examples = []
    for e in hits[:10]:
        result = e.payload.get("result") or {}
        err = result.get("error") if isinstance(result, dict) else None
        examples.append(f"- `{_format_loc(e)}`  error: `{_truncate(err, 100)}`")
    return Finding(
        rule="premature_finish",
        severity="medium",
        summary=f"{len(hits)} `finish` call(s) returned an error.",
        detail=(
            "Planner tried to finalize while open subtasks remained or other "
            "completion criteria were unmet. Often signals the planner gave up early.\n\n"
            + "\n".join(examples)
            + ("\n…" if len(hits) > 10 else "")
        ),
        count=len(hits),
    )


def detect_get_schema_after_upsert(
    events: list[Event], window: int = 5
) -> Finding | None:
    """`get_full_openapi_schema` called within N tool calls after a successful upsert.

    Documented anti-pattern in oas_builder_agent prompt: verifying via
    get_full_openapi_schema after a successful upsert is wasteful.
    """
    by_session: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        if e.type != "metrics_tool_call":
            continue
        by_session[e.session_id or "?"].append(e)

    hits: list[Event] = []
    for _sid, evs in by_session.items():
        # Walk pairwise. For each upsert_path/upsert_component call, check the
        # next `window` calls in the same session for get_full_openapi_schema.
        for i, e in enumerate(evs):
            if e.tool_name not in ("upsert_path", "upsert_component"):
                continue
            for follow in evs[i + 1 : i + 1 + window]:
                if follow.tool_name == "get_full_openapi_schema":
                    hits.append(follow)
                    break

    if not hits:
        return None
    examples = [f"- `{_format_loc(e)}`" for e in hits[:10]]
    return Finding(
        rule="get_schema_after_upsert",
        severity="low",
        summary=f"{len(hits)} `get_full_openapi_schema` call(s) within {window} tool calls after an upsert.",
        detail=(
            "Anti-pattern: verifying via `get_full_openapi_schema` right after an upsert "
            "succeeded is wasteful — the diff is already in the upsert result.\n\n"
            + "\n".join(examples)
            + ("\n…" if len(hits) > 10 else "")
        ),
        count=len(hits),
    )


def detect_tool_error_rate(
    events: list[Event], min_calls: int = 5, error_pct_threshold: float = 0.25
) -> Finding | None:
    """Tools whose error rate exceeds a threshold (per agent, per tool)."""
    total: Counter = Counter()
    errors: Counter = Counter()
    for e in events:
        if e.type != "metrics_tool_result":
            continue
        key = (e.agent_name or "?", e.tool_name or "?")
        total[key] += 1
        if _is_error_result(e.payload):
            errors[key] += 1
    flagged = []
    for key, n in total.items():
        if n < min_calls:
            continue
        rate = errors.get(key, 0) / n
        if rate >= error_pct_threshold:
            flagged.append((key, n, errors.get(key, 0), rate))
    if not flagged:
        return None
    flagged.sort(key=lambda t: -t[3])
    examples = [
        f"- `{agent}` · `{tool}`  {err}/{n} = **{rate*100:.0f}%** error rate"
        for (agent, tool), n, err, rate in flagged[:20]
    ]
    return Finding(
        rule="high_tool_error_rate",
        severity="medium",
        summary=f"{len(flagged)} (agent, tool) pairs with ≥{int(error_pct_threshold*100)}% error rate (min {min_calls} calls).",
        detail="\n".join(examples) + ("\n…" if len(flagged) > 20 else ""),
        count=len(flagged),
    )


def detect_version_outcomes(events: list[Event]) -> Finding | None:
    """Cross-tab (template_key, template_version) × (finished, failed)."""
    finished: Counter = Counter()
    failed: Counter = Counter()
    for e in events:
        key = (
            e.payload.get("template_key") or "?",
            e.payload.get("template_version") or "?",
        )
        if e.type == "task_finished":
            finished[key] += 1
        elif e.type == "task_failed":
            failed[key] += 1
    keys = sorted(set(finished) | set(failed))
    if not keys:
        return None
    rows = ["| template | version | finished | failed |", "|---|---|---:|---:|"]
    for k in keys:
        f_n = finished.get(k, 0)
        x_n = failed.get(k, 0)
        rows.append(f"| `{k[0]}` | `{k[1]}` | {f_n} | {x_n} |")
    return Finding(
        rule="version_outcomes",
        severity="low",
        summary=f"Outcome counts for {len(keys)} (template, version) pair(s).",
        detail="\n".join(rows),
        count=len(keys),
    )


def detect_prompt_versions(events: list[Event]) -> Finding | None:
    """Surface the `prompt_versions` snapshot from RUN_STARTED, if present."""
    snapshots: list[tuple[str, dict[str, str]]] = []
    for e in events:
        if e.type != "run_started":
            continue
        pv = e.payload.get("prompt_versions")
        if isinstance(pv, dict) and pv:
            snapshots.append((e.source, pv))
    if not snapshots:
        return None
    blocks = []
    for source, pv in snapshots:
        rows = "\n".join(f"  - `{name}` = `{ver}`" for name, ver in sorted(pv.items()))
        blocks.append(f"**{source}**\n{rows}")
    return Finding(
        rule="prompt_versions_seen",
        severity="low",
        summary=f"Prompt-version snapshots from {len(snapshots)} run(s).",
        detail="\n\n".join(blocks),
        count=len(snapshots),
    )


DETECTORS = {
    "prompt_versions": detect_prompt_versions,
    "version_outcomes": detect_version_outcomes,
    "default_tool_calls": detect_default_tool_calls,
    "tool_retry_streaks": detect_retry_streaks,
    "elided_repeats": detect_elided_repeats,
    "suspect_upsert_args": detect_suspect_upsert_args,
    "premature_finish": detect_premature_finish,
    "get_schema_after_upsert": detect_get_schema_after_upsert,
    "tool_error_rate": detect_tool_error_rate,
}


# ─── Report ───────────────────────────────────────────────────────────────────


_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def render_report(findings: list[Finding], n_events: int, sources: list[Path]) -> str:
    out: list[str] = []
    out.append("# Contractor Diagnose Report\n")
    out.append(f"- Events scanned: **{n_events:,}**")
    out.append(f"- Sources ({len(sources)}):")
    for p in sources:
        out.append(f"  - `{p}`")
    out.append("")
    if not findings:
        out.append("_No anti-patterns detected._")
        return "\n".join(out)

    findings.sort(key=lambda f: (_SEVERITY_RANK.get(f.severity, 9), -f.count))
    out.append("## Findings\n")
    for f in findings:
        out.append(f"### `{f.rule}` — {f.severity}")
        out.append(f.summary)
        out.append("")
        out.append(f.detail)
        out.append("")
    return "\n".join(out)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose contractor metrics.jsonl for known failure-mode fingerprints."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more metrics.jsonl files or directories (searched recursively).",
    )
    parser.add_argument(
        "--rule",
        action="append",
        choices=list(DETECTORS.keys()),
        help="Run only the named detector(s). Repeatable. Default: all detectors.",
    )
    args = parser.parse_args(argv)

    files = collect_paths(args.paths)
    if not files:
        print("error: no metrics.jsonl files found", file=sys.stderr)
        return 2

    all_events: list[Event] = []
    for p in files:
        all_events.extend(iter_events(p))

    rules = args.rule or list(DETECTORS.keys())
    findings: list[Finding] = []
    for rule in rules:
        detector = DETECTORS[rule]
        result = detector(all_events)
        if result is not None:
            findings.append(result)

    print(render_report(findings, len(all_events), files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
