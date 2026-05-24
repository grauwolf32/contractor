#!/usr/bin/env python3
"""Render subtask graphs from a contractor run's ``metrics.jsonl``.

Each ``task_finished`` (and ``task_failed``) event carries the streamline
planner's full subtask pool. This tool reconstructs the parent→child tree
from dotted ``task_id``s (``1`` → ``1.2`` → ``1.2.3``) and writes one PNG
+ one DOT file per task, with nodes coloured by terminal status.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# Maps subtask status → (fill colour, label). Mirrors SUBTASK_STATUS_TRANSITIONS
# in contractor/tools/tasks.py: terminal states are done/decomposed/skipped;
# incomplete/malformed records appear when a subtask was retried by being
# decomposed but the parent record still carries the failed status.
STATUS_COLOURS: dict[str, str] = {
    "done": "#4caf50",         # green — success
    "incomplete": "#e53935",   # red — failed
    "malformed": "#b71c1c",    # dark red — failed parse
    "skipped": "#ffb300",      # amber — explicitly skipped
    "decomposed": "#90caf9",   # light blue — broke into children
    "new": "#bdbdbd",          # grey — never executed
    "unknown": "#9e9e9e",
}

# task_finished payloads embed records directly; task_failed nests them under
# last_result. ITERATION_FINISHED is used as a fallback for runs that were
# cancelled before task_finished/failed was emitted.
_RECORD_EVENT_TYPES: frozenset[str] = frozenset(
    {"task_finished", "task_failed", "iteration_finished"}
)

_FILENAME_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def _sanitize(name: str) -> str:
    return _FILENAME_SAFE.sub("_", str(name)).strip("_") or "task"


# ─── Data extraction ─────────────────────────────────────────────────────────


@dataclass
class SubtaskRun:
    """One reconstructed task run with its subtask pool."""

    event_type: str
    task_name: str
    task_id: Any
    template_key: str | None
    records: list[dict[str, Any]] = field(default_factory=list)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _extract_records(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull the subtask records list out of a task_(finished|failed) event."""
    records = event.get("records")
    if isinstance(records, list):
        return [r for r in records if isinstance(r, dict)]

    # task_failed nests the iteration's records under last_result.records,
    # and iteration_finished does the same under result.records.
    for wrapper_key in ("last_result", "result"):
        nested = event.get(wrapper_key)
        if isinstance(nested, dict):
            nested_records = nested.get("records")
            if isinstance(nested_records, list):
                return [r for r in nested_records if isinstance(r, dict)]
    return []


def extract_runs(events: Iterable[dict[str, Any]]) -> list[SubtaskRun]:
    """Build one SubtaskRun per recorded task attempt with non-empty records.

    Prefers task_finished/task_failed over iteration_finished for the same
    (task_name, task_id) so we only render the final, authoritative pool.
    """
    by_key: dict[tuple[str, Any], SubtaskRun] = {}
    priority = {"task_finished": 2, "task_failed": 2, "iteration_finished": 1}

    for event in events:
        etype = event.get("type")
        if etype not in _RECORD_EVENT_TYPES:
            continue
        records = _extract_records(event)
        if not records:
            continue

        task_name = str(event.get("task_name") or "unknown_task")
        task_id = event.get("task_id")
        key = (task_name, task_id)

        new_run = SubtaskRun(
            event_type=etype,
            task_name=task_name,
            task_id=task_id,
            template_key=event.get("template_key"),
            records=records,
        )

        existing = by_key.get(key)
        if existing is None or priority[etype] >= priority[existing.event_type]:
            by_key[key] = new_run

    return list(by_key.values())


# ─── Graph construction & layout ─────────────────────────────────────────────


@dataclass
class Node:
    task_id: str
    title: str
    status: str
    children: list[str] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0


def _parent_id(task_id: str) -> str | None:
    if "." not in task_id:
        return None
    return task_id.rsplit(".", 1)[0]


def build_tree(records: list[dict[str, Any]]) -> tuple[dict[str, Node], list[str]]:
    """Build node table + ordered list of roots from a record pool.

    Records share their schema with Subtask.model_dump() plus execution
    result fields; we only need task_id/title/status here.
    """
    nodes: dict[str, Node] = {}

    for rec in records:
        tid = rec.get("task_id")
        if tid is None:
            continue
        tid = str(tid)
        status = str(rec.get("status") or "unknown")
        # Later records for the same task_id win — decompose writes a
        # "decomposed" parent record after the original status flip.
        nodes[tid] = Node(
            task_id=tid,
            title=str(rec.get("title") or ""),
            status=status,
        )

    # Synthesize any missing ancestors as 'unknown' so the tree is connected.
    for tid in list(nodes):
        cur = tid
        while True:
            parent = _parent_id(cur)
            if parent is None:
                break
            if parent not in nodes:
                nodes[parent] = Node(task_id=parent, title="", status="unknown")
            cur = parent

    # Wire children, preserving numeric order.
    def _sort_key(tid: str) -> list[int]:
        return [int(part) if part.isdigit() else 0 for part in tid.split(".")]

    for tid, node in nodes.items():
        parent = _parent_id(tid)
        if parent is not None:
            nodes[parent].children.append(tid)
    for node in nodes.values():
        node.children.sort(key=_sort_key)

    roots = sorted([tid for tid in nodes if _parent_id(tid) is None], key=_sort_key)
    return nodes, roots


def layout_tree(nodes: dict[str, Node], roots: list[str]) -> None:
    """Assign x/y coordinates via a simple leaf-packing tree layout.

    Each leaf claims one x slot; each internal node centres above its
    children. Depth grows downward (y = -depth).
    """
    counter = {"x": 0}

    def _assign(tid: str, depth: int) -> float:
        node = nodes[tid]
        node.y = float(-depth)
        if not node.children:
            x = float(counter["x"])
            counter["x"] += 1
            node.x = x
            return x
        child_xs = [_assign(c, depth + 1) for c in node.children]
        node.x = (child_xs[0] + child_xs[-1]) / 2.0
        return node.x

    for root in roots:
        _assign(root, 0)
        counter["x"] += 1  # gap between root subtrees


# ─── Rendering ───────────────────────────────────────────────────────────────


def _wrap(text: str, width: int = 28, max_lines: int = 3) -> str:
    text = text.strip()
    if not text:
        return ""
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        candidate = f"{cur} {word}".strip()
        if len(candidate) <= width:
            cur = candidate
            continue
        if cur:
            lines.append(cur)
        cur = word
        if len(lines) >= max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    consumed = sum(len(line.split()) for line in lines)
    if len(lines) == max_lines and (cur or len(words) > consumed):
        lines[-1] = lines[-1][: max(0, width - 1)] + "…"
    return "\n".join(lines)


def render_tree(
    nodes: dict[str, Node],
    roots: list[str],
    title: str,
    output_path: Path,
) -> None:
    if not nodes:
        logger.info("No subtasks to render for %s", title)
        return

    layout_tree(nodes, roots)

    xs = [n.x for n in nodes.values()]
    ys = [n.y for n in nodes.values()]
    width = max(8.0, (max(xs) - min(xs) + 2) * 1.6)
    height = max(4.0, (max(ys) - min(ys) + 2) * 1.4)

    fig, ax = plt.subplots(figsize=(width, height))

    # Edges first so node boxes draw on top.
    for node in nodes.values():
        for child_id in node.children:
            child = nodes[child_id]
            ax.plot(
                [node.x, child.x],
                [node.y, child.y],
                color="#555555",
                linewidth=1.0,
                zorder=1,
            )

    for node in nodes.values():
        colour = STATUS_COLOURS.get(node.status, STATUS_COLOURS["unknown"])
        label_lines = [f"#{node.task_id}"]
        wrapped_title = _wrap(node.title)
        if wrapped_title:
            label_lines.append(wrapped_title)
        label_lines.append(f"[{node.status}]")
        label = "\n".join(label_lines)
        ax.text(
            node.x,
            node.y,
            label,
            ha="center",
            va="center",
            fontsize=7,
            zorder=3,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": colour,
                "edgecolor": "#222222",
                "linewidth": 0.8,
            },
        )

    statuses_present = sorted({n.status for n in nodes.values()})
    legend_handles = [
        mpatches.Patch(
            color=STATUS_COLOURS.get(s, STATUS_COLOURS["unknown"]),
            label=s,
        )
        for s in statuses_present
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)

    ax.set_title(title)
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    ax.set_axis_off()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_dot(
    nodes: dict[str, Node],
    title: str,
    output_path: Path,
) -> None:
    """Write a Graphviz DOT file alongside the PNG for editable rendering."""
    lines = [f'digraph "{title}" {{', "  rankdir=TB;", '  node [shape=box, style="rounded,filled", fontname="Helvetica"];']
    for node in nodes.values():
        colour = STATUS_COLOURS.get(node.status, STATUS_COLOURS["unknown"])
        title_part = node.title.replace('"', '\\"')
        label = f"#{node.task_id}\\n{title_part}\\n[{node.status}]"
        lines.append(
            f'  "{node.task_id}" [label="{label}", fillcolor="{colour}"];'
        )
    for node in nodes.values():
        for child_id in node.children:
            lines.append(f'  "{node.task_id}" -> "{child_id}";')
    lines.append("}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _resolve_input(input_arg: Path) -> Path:
    if input_arg.is_dir():
        candidate = input_arg / "metrics.jsonl"
        if not candidate.exists():
            raise FileNotFoundError(f"No metrics.jsonl found in {input_arg}")
        return candidate
    return input_arg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render subtask graph(s) from a contractor metrics.jsonl.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to metrics.jsonl or a directory containing it",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input_dir>/subtask_graphs)",
    )
    parser.add_argument(
        "--no-dot",
        action="store_true",
        help="Skip writing .dot files alongside PNGs",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    input_path = _resolve_input(args.input.resolve())
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else input_path.parent / "subtask_graphs"
    )

    events = _load_jsonl(input_path)
    runs = extract_runs(events)
    if not runs:
        logger.warning("No subtask records found in %s", input_path)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, int] = {}

    for run in runs:
        nodes, roots = build_tree(run.records)
        if not nodes:
            continue

        slug_base = _sanitize(f"{run.task_name}__{run.task_id}")
        count = seen.get(slug_base, 0)
        seen[slug_base] = count + 1
        slug = slug_base if count == 0 else f"{slug_base}__{count + 1}"

        title = f"{run.task_name} (task_id={run.task_id}, {run.event_type})"
        png_path = output_dir / f"{slug}.png"
        render_tree(nodes, roots, title, png_path)
        if not args.no_dot:
            render_dot(nodes, title, output_dir / f"{slug}.dot")

        logger.info("Wrote %s (%d subtasks)", png_path, len(nodes))

    print(f"Done. {len(runs)} graph(s) written under: {output_dir}")


if __name__ == "__main__":
    main()
