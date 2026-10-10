"""Validate the task records under tasks/ against tasks/index.yml.

Open tasks (pending, in_progress, deferred) live directly in tasks/; finished
tasks (completed, archived) live in tasks/done/. The task files are the
index: this check keeps them consistent instead of a hand-maintained list.
Every task names an epic declared in tasks/epics.yml and a kind; a task that
repairs earlier work names that task in introduced_by. tasks/BACKLOG.md is
rendered from the records (``--write-backlog``, ``make tasks-backlog``) and
must stay current.

PyYAML comes from the Runtime environment, so run it from the repository root
as ``cd runtime && uv run python ../scripts/check_tasks.py`` (``make lint``).
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "tasks"
DONE = "done"
FINISHED = frozenset({"completed", "archived"})
KINDS = frozenset({"feature", "defect", "chore"})
NOT_TASKS = frozenset({"index.yml", "epics.yml"})
BACKLOG = "BACKLOG.md"
TASK_ID = re.compile(r"^(MVP|V\d+)-\d{3}[A-Z]?$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
REQUIRED = (
    "schema_version",
    "id",
    "title",
    "milestone",
    "epic",
    "kind",
    "priority",
    "status",
    "depends_on",
    "goal",
    "context",
    "scope",
    "acceptance_criteria",
    "tests",
)


@dataclass(frozen=True)
class Record:
    path: str  # relative to tasks/, e.g. "done/v1-001-cancellation-abort.yml"
    data: Any


@dataclass(frozen=True)
class Policy:
    statuses: frozenset[str]
    priorities: frozenset[str]
    exit_tasks: Mapping[str, str]  # milestone id -> exit task id
    epics: frozenset[str] = frozenset()
    epic_titles: Mapping[str, str] = field(default_factory=dict)  # in tasks/epics.yml order


def validate(records: Iterable[Record], policy: Policy) -> list[str]:
    errors: list[str] = []
    by_id: dict[str, Record] = {}
    records = list(records)
    for record in records:
        data = record.data
        if not isinstance(data, dict):
            errors.append(f"{record.path}: task file is not a mapping")
            continue
        missing = [key for key in REQUIRED if key not in data]
        if missing:
            errors.append(f"{record.path}: missing {', '.join(missing)}")
        task_id = data.get("id")
        if not isinstance(task_id, str) or not TASK_ID.match(task_id):
            errors.append(f"{record.path}: invalid id {task_id!r}")
            continue
        name = record.path.rsplit("/", 1)[-1]
        if not name.startswith(task_id.lower() + "-"):
            errors.append(f"{record.path}: file name does not start with {task_id.lower()}-")
        if task_id in by_id:
            errors.append(f"{record.path}: duplicate id {task_id} (also {by_id[task_id].path})")
            continue
        by_id[task_id] = record

        status = data.get("status")
        if status not in policy.statuses:
            errors.append(f"{record.path}: status {status!r} is not declared")
        if "epic" in data and data["epic"] not in policy.epics:
            errors.append(f"{record.path}: epic {data['epic']!r} is not declared in tasks/epics.yml")
        if "kind" in data and data["kind"] not in KINDS:
            errors.append(f"{record.path}: kind {data['kind']!r} is not one of {', '.join(sorted(KINDS))}")
        priority = data.get("priority")
        if priority not in policy.priorities:
            errors.append(f"{record.path}: priority {priority!r} is not declared")
        finished = status in FINISHED
        in_done = record.path.startswith(DONE + "/")
        if finished and not in_done:
            errors.append(f"{record.path}: {status} task belongs in tasks/{DONE}/")
        if in_done and not finished:
            errors.append(f"{record.path}: {status} task belongs in tasks/, not tasks/{DONE}/")
        if status == "completed" and not COMMIT.match(str(data.get("implementation_commit", ""))):
            errors.append(f"{record.path}: completed task needs a full implementation_commit")

    for record in records:
        data = record.data
        if not isinstance(data, dict):
            continue
        for dependency in data.get("depends_on") or []:
            if dependency not in by_id:
                errors.append(f"{record.path}: depends_on unknown task {dependency}")
        introducer = data.get("introduced_by")
        if introducer is not None and (introducer not in by_id or introducer == data.get("id")):
            errors.append(f"{record.path}: introduced_by {introducer!r} must name another existing task")

    for milestone, exit_task in sorted(policy.exit_tasks.items()):
        if exit_task not in by_id:
            errors.append(f"tasks/index.yml: milestone {milestone} exit_task {exit_task} has no task file")
    return errors


def _order(task_id: str) -> tuple[int, int, str]:
    prefix, number = task_id.split("-")
    version = 0 if prefix == "MVP" else int(prefix[1:])
    return version, int(number[:3]), number[3:]


def _cell(value: Any) -> str:
    return " ".join(str(value).split()).replace("|", "\\|")


def _link(record: Record) -> str:
    return f"[{record.data['id']}]({record.path})"


def render_backlog(records: Iterable[Record], policy: Policy) -> str:
    """Render tasks/BACKLOG.md from the task records."""
    tasks = sorted(
        (r for r in records if isinstance(r.data, dict)), key=lambda r: _order(r.data["id"])
    )
    by_id = {r.data["id"]: r.data for r in tasks}
    by_status: dict[str, list[Record]] = {}
    for record in tasks:
        by_status.setdefault(record.data["status"], []).append(record)
    count = {status: len(items) for status, items in by_status.items()}
    open_total = sum(count.get(s, 0) for s in ("in_progress", "pending", "deferred"))
    lines = [
        "# Backlog",
        "",
        "Generated from the task files by `make tasks-backlog`; do not edit by hand.",
        "`make lint` fails when this file is stale. New tasks start from",
        "[templates/](templates/); the task policy is in [index.yml](index.yml) and",
        "the epics are in [epics.yml](epics.yml).",
        "",
        f"{open_total} open tasks: {count.get('in_progress', 0)} in progress, "
        f"{count.get('pending', 0)} pending, {count.get('deferred', 0)} deferred. "
        f"{count.get('completed', 0) + count.get('archived', 0)} finished: "
        f"{count.get('completed', 0)} completed, {count.get('archived', 0)} archived.",
        "",
        "## In progress",
        "",
        "| Task | Epic | Kind | Priority | Title |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in by_status.get("in_progress", []):
        d = r.data
        lines.append(
            f"| {_link(r)} | {d['epic']} | {d['kind']} | {d['priority']} | {_cell(d['title'])} |"
        )
    lines += ["", "## Pending"]
    pending = by_status.get("pending", [])
    for epic, title in policy.epic_titles.items():
        rows = [r for r in pending if r.data["epic"] == epic]
        if not rows:
            continue
        lines += [
            "",
            f"### {title} (`{epic}`)",
            "",
            "| Task | Kind | Priority | Waiting for | Title |",
            "| --- | --- | --- | --- | --- |",
        ]
        for r in rows:
            d = r.data
            waiting = [
                dep
                for dep in d.get("depends_on") or []
                if by_id.get(dep, {}).get("status") != "completed"
            ]
            lines.append(
                f"| {_link(r)} | {d['kind']} | {d['priority']} | "
                f"{', '.join(waiting) or '—'} | {_cell(d['title'])} |"
            )
    lines += [
        "",
        "## Deferred",
        "",
        "| Task | Epic | Since | Reactivate when |",
        "| --- | --- | --- | --- |",
    ]
    for r in by_status.get("deferred", []):
        deferral = r.data.get("deferral") or {}
        lines.append(
            f"| {_link(r)} | {r.data['epic']} | {_cell(deferral.get('date', '—'))} | "
            f"{_cell(deferral.get('reactivate_when', '—'))} |"
        )
    caused: dict[str, int] = {}
    for d in by_id.values():
        introducer = by_id.get(d.get("introduced_by") or "")
        if introducer:
            caused[introducer["epic"]] = caused.get(introducer["epic"], 0) + 1
    lines += [
        "",
        "## Epics",
        "",
        "All tasks of each epic by kind, and how many later tasks name one of its",
        "tasks in `introduced_by`.",
        "",
        "| Epic | Open | Feature | Defect | Chore | Later fixes caused |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for epic, title in policy.epic_titles.items():
        mine = [d for d in by_id.values() if d["epic"] == epic]
        kinds = {kind: sum(1 for d in mine if d["kind"] == kind) for kind in KINDS}
        opened = sum(1 for d in mine if d["status"] not in FINISHED)
        lines.append(
            f"| {title} (`{epic}`) | {opened} | {kinds['feature']} | {kinds['defect']} | "
            f"{kinds['chore']} | {caused.get(epic, 0)} |"
        )
    return "\n".join(lines) + "\n"


def validate_templates(templates: Mapping[str, Any]) -> list[str]:
    """Check that each tasks/templates/<kind>.yml still carries the schema."""
    errors: list[str] = []
    if set(templates) != KINDS:
        errors.append(f"templates/: expected one template per kind ({', '.join(sorted(KINDS))})")
    for kind, data in sorted(templates.items()):
        path = f"templates/{kind}.yml"
        if not isinstance(data, dict):
            errors.append(f"{path}: template is not a mapping")
            continue
        missing = [key for key in REQUIRED if key not in data]
        if missing:
            errors.append(f"{path}: missing {', '.join(missing)}")
        if data.get("kind") != kind:
            errors.append(f"{path}: kind must be {kind}")
        if data.get("status") != "pending":
            errors.append(f"{path}: status must be pending")
    return errors


def load_templates(tasks: Path = TASKS) -> dict[str, Any]:
    import yaml  # Provided by the Runtime environment; see the module docstring.

    return {
        path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted((tasks / "templates").glob("*.yml"))
    }


def load(tasks: Path = TASKS) -> tuple[list[Record], Policy]:
    import yaml  # Provided by the Runtime environment; see the module docstring.

    index = yaml.safe_load((tasks / "index.yml").read_text(encoding="utf-8"))
    execution = index["execution_policy"]
    epics = yaml.safe_load((tasks / "epics.yml").read_text(encoding="utf-8"))["epics"]
    for epic in epics:
        for spec in epic.get("specs") or []:
            if not (tasks.parent / spec).is_file():
                print(f"epics.yml: epic {epic['id']} names missing spec {spec}", file=sys.stderr)
                raise SystemExit(1)
    policy = Policy(
        statuses=frozenset(execution["task_statuses"]),
        priorities=frozenset(execution["priorities"]),
        exit_tasks={m["id"]: m["exit_task"] for m in index.get("milestones") or [] if "exit_task" in m},
        epics=frozenset(epic["id"] for epic in epics),
        epic_titles={epic["id"]: epic["title"] for epic in epics},
    )
    records: list[Record] = []
    for path in sorted([*tasks.glob("*.yml"), *(tasks / DONE).glob("*.yml")]):
        relative = path.relative_to(tasks).as_posix()
        if relative in NOT_TASKS:
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            records.append(Record(relative, None))
            print(f"{relative}: invalid YAML: {error}", file=sys.stderr)
            continue
        records.append(Record(relative, data))
    return records, policy


def main() -> int:
    records, policy = load()
    errors = validate(records, policy) + validate_templates(load_templates())
    backlog = TASKS / BACKLOG
    if not errors:
        rendered = render_backlog(records, policy)
        if "--write-backlog" in sys.argv[1:]:
            backlog.write_text(rendered, encoding="utf-8")
            print(f"wrote tasks/{BACKLOG}")
        elif not backlog.is_file() or backlog.read_text(encoding="utf-8") != rendered:
            errors.append(f"{BACKLOG}: stale; run make tasks-backlog")
    for error in errors:
        print(f"tasks/{error}" if not error.startswith("tasks/") else error, file=sys.stderr)
    if errors:
        print(f"{len(errors)} task record problem(s)", file=sys.stderr)
        return 1
    print(f"{len(records)} task records are consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
