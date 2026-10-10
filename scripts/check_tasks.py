"""Validate the task records under tasks/ against tasks/index.yml.

Open tasks (pending, in_progress, deferred) live directly in tasks/; finished
tasks (completed, archived) live in tasks/done/. The task files are the
index: this check keeps them consistent instead of a hand-maintained list.

PyYAML comes from the Runtime environment, so run it from the repository root
as ``cd runtime && uv run python ../scripts/check_tasks.py`` (``make lint``).
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "tasks"
DONE = "done"
FINISHED = frozenset({"completed", "archived"})
TASK_ID = re.compile(r"^(MVP|V\d+)-\d{3}[A-Z]?$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
REQUIRED = (
    "schema_version",
    "id",
    "title",
    "milestone",
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

    for milestone, exit_task in sorted(policy.exit_tasks.items()):
        if exit_task not in by_id:
            errors.append(f"tasks/index.yml: milestone {milestone} exit_task {exit_task} has no task file")
    return errors


def load(tasks: Path = TASKS) -> tuple[list[Record], Policy]:
    import yaml  # Provided by the Runtime environment; see the module docstring.

    index = yaml.safe_load((tasks / "index.yml").read_text(encoding="utf-8"))
    execution = index["execution_policy"]
    policy = Policy(
        statuses=frozenset(execution["task_statuses"]),
        priorities=frozenset(execution["priorities"]),
        exit_tasks={m["id"]: m["exit_task"] for m in index.get("milestones") or [] if "exit_task" in m},
    )
    records: list[Record] = []
    for path in sorted([*tasks.glob("*.yml"), *(tasks / DONE).glob("*.yml")]):
        relative = path.relative_to(tasks).as_posix()
        if relative == "index.yml":
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
    errors = validate(records, policy)
    for error in errors:
        print(f"tasks/{error}" if not error.startswith("tasks/") else error, file=sys.stderr)
    if errors:
        print(f"{len(errors)} task record problem(s)", file=sys.stderr)
        return 1
    print(f"{len(records)} task records are consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
