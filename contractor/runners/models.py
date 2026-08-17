from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum, unique
from pathlib import Path
from typing import Any, Literal

import yaml
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import AgentTool

from contractor.tools.observations import ObservationConfig

_checkpoint_logger = logging.getLogger(__name__ + ".checkpoint")

# ─── Enums ────────────────────────────────────────────────────────────────────


@unique
class EventType(StrEnum):
    """All event types emitted by TaskRunner, in one discoverable place."""

    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    TASK_STARTED = "task_started"
    TASK_FINISHED = "task_finished"
    TASK_FAILED = "task_failed"
    GLOBAL_TASK_FINISHED = "global_task_finished"
    ITERATION_STARTED = "iteration_started"
    ITERATION_FINISHED = "iteration_finished"
    ITERATION_RESULT = "iteration_result"
    FINAL_TEXT = "final_text"


@unique
class TaskStatus(StrEnum):
    RUNNING = "running"
    DONE = "done"


# ─── Data Structures ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class TaskResult:
    """Structured result returned from each iteration / task."""

    invocation_id: str
    task_ref: str
    task_key: str
    task_title: str
    template_key: str
    task_id: int
    session_id: str
    final_response: str
    state: dict[str, Any]
    carry_state: dict[str, Any]
    status: str | None
    result: Any
    summary: str | None
    records: list[Any]
    params: dict[str, Any]
    input_artifacts: dict[str, str]
    published_artifacts: dict[ArtifactKind, str]
    # True only when TaskRunner reused a validated checkpoint entry instead of
    # executing the task in this run. Downstream postconditions use this to
    # distinguish an intentionally reused persisted side effect from a stale
    # artifact that a freshly-run worker failed to update.
    restored: bool = False


@dataclass(slots=True, frozen=True)
class TaskRunnerEvent:
    type: EventType | str
    task_name: str
    task_id: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskInvocation:
    id: str
    ref: str
    template_key: str
    template_version: str
    worker_builder: WorkerBuilder

    params: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)

    # Publish key for result/summary/records artifacts. ``None`` keeps the
    # historical behavior (``template_key``); fan-out workflows that queue
    # several tasks from the same template set a unique, stable key per task
    # so siblings don't overwrite each other's artifacts.
    artifact_key: str | None = None

    iterations: int = 1
    max_attempts: int = 1
    max_steps: int = 15
    # Wallclock cap per attempt (seconds). ``max_steps`` bounds planner
    # subtasks, but nothing bounds a worker run's tool-call loop — without
    # this, one spinning task starves every task queued after it.
    timeout_s: float | None = None

    namespace: str | None = None
    model: LiteLlm | None = None
    observations: ObservationConfig | None = None

    def effective_namespace(self, fallback: str) -> str:
        return self.namespace or fallback

    @property
    def effective_artifact_key(self) -> str:
        """Key under which this invocation's artifacts are published."""
        return self.artifact_key or self.template_key

    def effective_model(self, fallback: LiteLlm) -> LiteLlm:
        return self.model or fallback

    @property
    def template_cache_key(self) -> tuple[str, str]:
        """Composite key used by ``TaskRunner.templates``."""
        return (self.template_key, self.template_version)


# ─── Key-name helpers ────────────────────────────────────────────────────────


class TaskScopedKeys:
    """
    Generates the task::{id}::* keys for a specific task_id.

    Usage:
        keys = TaskScopedKeys(task_id)
        state[keys.status] = TaskStatus.RUNNING
        state[keys.result] = "..."
    """

    def __init__(self, task_id: int) -> None:
        self._prefix = f"task::{task_id}"

    def _k(self, suffix: str) -> str:
        return f"{self._prefix}::{suffix}"

    # ── Task-scoped keys (written by TaskRunner, read by StreamlineManager) ──

    @property
    def objective(self) -> str:
        return self._k("objective")

    @property
    def status(self) -> str:
        return self._k("status")

    @property
    def current(self) -> str:
        """Current active subtask; None while not yet assigned."""
        return self._k("current")

    @property
    def result(self) -> str:
        """Final task result text; written by StreamlineManager.finish()."""
        return self._k("result")

    @property
    def summary(self) -> str:
        """Concise handoff summary; written by StreamlineManager.finish()."""
        return self._k("summary")

    @property
    def pool(self) -> str:
        """List of task record dicts; appended by StreamlineManager.save_record()."""
        return self._k("pool")


# ─── Global key constant ─────────────────────────────────────────────────────

GLOBAL_TASK_ID_KEY = "_global_task_id"


# ─── Session-state builder ───────────────────────────────────────────────────
#
# Per-iteration state is a flat dict keyed by:
#   _global_task_id    — sentinel read by StreamlineManager
#   task::{id}::*      — per-task live state, owned by planning agent


def build_active_state(*, task_id: int, task: RenderedTask) -> dict[str, Any]:
    """Initial flat state dict for a new task iteration."""
    keys = TaskScopedKeys(task_id)
    return {
        GLOBAL_TASK_ID_KEY: task_id,
        keys.objective: task.objective,
        keys.status: TaskStatus.RUNNING,
        keys.current: None,
        keys.result: "",
        keys.summary: "",
        keys.pool: [],
    }


# ─── Constants ────────────────────────────────────────────────────────────────

ArtifactKind = Literal["result", "summary", "records"]
WorkerBuilder = Callable[..., LlmAgent | AgentTool]
TaskRunnerEventHandler = Callable[[TaskRunnerEvent], Awaitable[None]]

# --- Task Models ---------------------------------------------------------------


TASKS_BASE_DIR = Path(__file__).parent.parent / "tasks"


def _normalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
    return normalized or "task"


def _artifact_var_name(artifact_ref: str) -> str:
    return "artifact__" + "__".join(
        _normalize_name(part) for part in artifact_ref.split("/") if part.strip()
    )


@dataclass(slots=True, frozen=True)
class TaskTemplate:
    key: str
    version: str
    title: str
    objective: str
    instructions: str
    output_format: str
    default_artifacts: list[str] = field(default_factory=list)
    default_skills: list[str] = field(default_factory=list)
    default_iterations: int = 1
    format: str = "json"
    # Declared parameter placeholders with default values. A task body that
    # references ``{focus}`` declares ``params: {focus: "<default text>"}`` so
    # the placeholder renders even when no caller supplies it; a workflow
    # overrides it via ``add_task(params={"focus": "..."})``. This is what makes
    # a placeholder *optional* — plain ``str.format`` KeyErrors on a missing key.
    default_params: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, name: str, version: str | None = None) -> TaskTemplate:
        """Load a task template by name, optionally pinning the version.

        Layout (mirrors the prompt manifest system):
          contractor/tasks/<name>.yml          — manifest with `active:` + `versions:`
          contractor/tasks/<name>/<vN>.yml     — task body for each version

        When ``version`` is ``None`` the manifest's ``active`` version is used.
        """
        template_key = Path(name).stem
        manifest_path, resolved_version, body_path = _resolve_task_version(
            template_key, version
        )

        with open(body_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if "task" not in data:
            raise ValueError(
                f"Task body {body_path} missing top-level 'task:' key "
                f"(manifest: {manifest_path})"
            )
        raw = data["task"]

        for required in ("objective", "instructions", "output_format"):
            if required not in raw:
                raise ValueError(
                    f"Task body {body_path} missing required '{required}:' "
                    f"field under 'task:' (manifest: {manifest_path})"
                )

        return cls(
            key=template_key,
            version=resolved_version,
            title=raw.get("name", template_key) or template_key,
            objective=raw["objective"],
            instructions=raw["instructions"],
            output_format=raw["output_format"],
            default_artifacts=list(raw.get("artifacts", []) or []),
            default_skills=list(raw.get("skills", []) or []),
            default_iterations=int(raw.get("iterations", 1) or 1),
            format=raw.get("format", "json") or "json",
            default_params={
                str(k): "" if v is None else str(v)
                for k, v in (raw.get("params", {}) or {}).items()
            },
        )


# ─── Task manifest resolution ─────────────────────────────────────────────────


def _resolve_task_version(
    template_key: str, version: str | None
) -> tuple[Path, str, Path]:
    """Resolve a task name (+ optional version) to a body file on disk.

    Returns ``(manifest_path, resolved_version, body_path)``. Raises
    ``ValueError`` with a clear message on missing manifest, missing
    version, or missing body file.
    """
    manifest_path = TASKS_BASE_DIR / f"{template_key}.yml"
    if not manifest_path.exists():
        raise ValueError(f"Task template {template_key} not found at {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}

    if "active" not in manifest or "versions" not in manifest:
        raise ValueError(
            f"Task manifest {manifest_path} must declare 'active:' and 'versions:' "
            f"(see contractor/tasks/dependency_information.yml for the layout)"
        )

    versions = manifest["versions"] or {}
    if not isinstance(versions, dict):
        raise ValueError(
            f"Task manifest {manifest_path} 'versions:' must be a mapping"
        )

    # Explicit arg wins; then an env override (CONTRACTOR_TASK_VERSION_<NAME>,
    # for A/B eval-gating a task version without flipping `active`); then active.
    resolved = (
        version
        or os.environ.get(f"CONTRACTOR_TASK_VERSION_{template_key.upper()}")
        or manifest["active"]
    )
    if resolved not in versions:
        available = ", ".join(sorted(versions.keys())) or "(none)"
        raise ValueError(
            f"Task version {resolved!r} not declared in {manifest_path}. "
            f"Available versions: {available}"
        )

    entry = versions[resolved] or {}
    file_ref = entry.get("file") if isinstance(entry, dict) else None
    if not file_ref:
        raise ValueError(
            f"Task manifest {manifest_path} version {resolved!r} missing 'file:' field"
        )

    body_path = TASKS_BASE_DIR / file_ref
    if not body_path.exists():
        raise ValueError(
            f"Task body for {template_key}@{resolved} not found at {body_path} "
            f"(referenced by {manifest_path})"
        )

    return manifest_path, str(resolved), body_path


@dataclass(slots=True, frozen=True)
class RenderedTask:
    key: str
    title: str
    objective: str
    instructions: str
    output_format: str
    format: str
    artifacts: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_template(
        cls,
        template: TaskTemplate,
        *,
        variables: Mapping[str, Any],
        params: Mapping[str, Any],
        artifacts: Mapping[str, str],
    ) -> RenderedTask:
        # Precedence (low → high): template param defaults, then runner-level
        # variables, then per-invocation params. So a workflow's
        # ``add_task(params=...)`` overrides the YAML default, and an omitted
        # param falls back to the declared default instead of raising KeyError.
        scope: dict[str, Any] = dict(template.default_params)
        scope.update(variables)
        scope.update(params)

        scope["artifacts"] = yaml.safe_dump(
            dict(artifacts),
            allow_unicode=True,
            sort_keys=False,
        )

        # Distinct artifact refs can normalize to the same template variable
        # (e.g. "oas-build/result" and "oas_build/result" both become
        # "artifact__oas_build__result"); the later one would silently win,
        # so refuse the ambiguity instead.
        var_sources: dict[str, str] = {}
        for artifact_ref, value in artifacts.items():
            var_name = _artifact_var_name(artifact_ref)
            if var_name in var_sources:
                raise ValueError(
                    f"Artifact refs {var_sources[var_name]!r} and "
                    f"{artifact_ref!r} both normalize to template variable "
                    f"{var_name!r} — rename one so the substitutions don't "
                    f"collide"
                )
            var_sources[var_name] = artifact_ref
            scope[var_name] = value

        return cls(
            key=template.key,
            title=template.title,
            objective=template.objective.format(**scope),
            instructions=template.instructions.format(**scope),
            output_format=template.output_format.format(**scope),
            artifacts=dict(artifacts),
            format=template.format,
        )

    def _format_artifacts(self) -> str:
        fmt: str = "artifacts from previous tasks, stored as memories:\n"
        for name in self.artifacts:
            fmt += f"* {name}\n"
        return fmt

    def _format_task(self) -> str:
        task: str = (
            f"TASK:\n{self.title}\n\n"
            f"OBJECTIVE:\n{self.objective}\n\n"
            f"INSTRUCTIONS:\n{self.instructions}\n\n"
            f"OUTPUT FORMAT:\n{self.output_format}\n\n"
        )
        if self.artifacts:
            task += f"INBOX:\n{self._format_artifacts()}"
        return task


# ─── Checkpoint ──────────────────────────────────────────────────────────────


_CHECKPOINT_VERSION = 1
_CHECKPOINT_SAVE_LOCK = threading.Lock()


@contextmanager
def _checkpoint_file_lock(path: Path) -> Iterator[None]:
    """Hold a cross-process advisory lock for one checkpoint path."""
    lock_path = path.with_name(f".{path.name}.lock")
    with open(lock_path, "a+b") as lock_file:
        if os.name == "nt":  # pragma: no cover - exercised on Windows CI only
            import msvcrt

            lock_file.seek(0)
            if not lock_file.read(1):
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@dataclass(slots=True)
class CheckpointEntry:
    task_id: int
    ref: str
    template_key: str
    template_version: str
    published_artifacts: dict[str, str]


@dataclass
class Checkpoint:
    workflow: str
    entries: list[CheckpointEntry] = field(default_factory=list)
    # Only entries changed by this in-memory snapshot may overwrite the latest
    # on-disk value during a merge.  A checkpoint loaded before a parallel
    # sibling finishes contains stale copies of every pre-existing entry; if we
    # blindly replay all of them, that stale snapshot can undo the sibling's
    # newer update even though the sibling refs themselves are preserved.
    _dirty_refs: set[str] = field(default_factory=set, init=False, repr=False)
    # Snapshot loaded from disk. Comparing it at save time catches callers
    # that mutate ``entries`` or an entry's artifact mapping directly instead
    # of going through ``mark_done``; without this, those valid public-dataclass
    # mutations were silently discarded by the concurrent merge.
    _baseline_entries: dict[str, tuple[int, str, str, str]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        # Directly-constructed checkpoints treat their supplied entries as new
        # data.  ``load`` clears this set after reconstructing an on-disk
        # snapshot, making loaded historical entries clean.
        self._dirty_refs.update(entry.ref for entry in self.entries)

    def get(self, ref: str) -> CheckpointEntry | None:
        for e in self.entries:
            if e.ref == ref:
                return e
        return None

    def mark_done(self, entry: CheckpointEntry) -> None:
        self.entries = [e for e in self.entries if e.ref != entry.ref]
        self.entries.append(entry)
        self._dirty_refs.add(entry.ref)

    @staticmethod
    def _entry_snapshot(entry: CheckpointEntry) -> tuple[int, str, str, str]:
        return (
            entry.task_id,
            entry.template_key,
            entry.template_version,
            json.dumps(
                entry.published_artifacts,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )

    def _current_snapshots(self) -> dict[str, tuple[int, str, str, str]]:
        snapshots: dict[str, tuple[int, str, str, str]] = {}
        for entry in self.entries:
            if entry.ref in snapshots:
                raise ValueError(
                    f"checkpoint contains duplicate task ref {entry.ref!r}"
                )
            snapshots[entry.ref] = self._entry_snapshot(entry)
        return snapshots

    def _capture_baseline(self) -> None:
        self._baseline_entries = self._current_snapshots()

    def save(self, path: Path) -> None:
        # Parallel TaskRunners (notably vuln-sweep's per-class fan-out) each
        # hold their own in-memory snapshot. Merge the latest on-disk entries
        # while serializing the read/replace sequence so the last finisher does
        # not erase siblings that completed earlier.
        with _CHECKPOINT_SAVE_LOCK, _checkpoint_file_lock(path):
            entries_to_save = self.entries
            current_snapshots = self._current_snapshots()
            dirty_refs = self._dirty_refs | {
                ref
                for ref, snapshot in current_snapshots.items()
                if self._baseline_entries.get(ref) != snapshot
            }
            deleted_refs = self._baseline_entries.keys() - current_snapshots.keys()

            existing = Checkpoint.load(path)
            if existing is not None and existing.workflow == self.workflow:
                merged = Checkpoint(
                    workflow=self.workflow,
                    entries=[
                        entry
                        for entry in existing.entries
                        if entry.ref not in deleted_refs
                    ],
                )
                for entry in self.entries:
                    if entry.ref in dirty_refs:
                        merged.mark_done(entry)
                entries_to_save = merged.entries

            data = {
                "version": _CHECKPOINT_VERSION,
                "workflow": self.workflow,
                "updated_at": datetime.now(UTC).isoformat(),
                "tasks": [
                    {
                        "task_id": e.task_id,
                        "ref": e.ref,
                        "template_key": e.template_key,
                        "template_version": e.template_version,
                        "published_artifacts": e.published_artifacts,
                    }
                    for e in entries_to_save
                ],
            }
            fd, tmp_name = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
            )
            tmp = Path(tmp_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
            # Adopt imported sibling entries only after the atomic replacement
            # succeeds. On a failed write, retaining the caller's old snapshot
            # and baseline prevents a retry from treating imported (clean)
            # siblings as locally dirty and overwriting a newer writer.
            self.entries = entries_to_save
            self._dirty_refs.clear()
            self._capture_baseline()

    @classmethod
    def load(cls, path: Path) -> Checkpoint | None:
        if not path.is_file():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            _checkpoint_logger.warning("ignoring corrupt checkpoint %s: %s", path, exc)
            return None

        # Structurally malformed data (valid JSON but not the expected shape —
        # entries missing task_id/ref/template_key, non-dict entries, …) must
        # follow the same "ignoring corrupt checkpoint" path, not raise.
        try:
            if data.get("version") != _CHECKPOINT_VERSION:
                _checkpoint_logger.warning(
                    "ignoring checkpoint %s with unsupported version %s",
                    path,
                    data.get("version"),
                )
                return None

            checkpoint = cls(
                workflow=data.get("workflow", ""),
                entries=[
                    CheckpointEntry(
                        task_id=t["task_id"],
                        ref=t["ref"],
                        template_key=t["template_key"],
                        template_version=t["template_version"],
                        published_artifacts=t.get("published_artifacts", {}),
                    )
                    for t in data.get("tasks", [])
                ],
            )
            checkpoint._dirty_refs.clear()
            checkpoint._capture_baseline()
            return checkpoint
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            _checkpoint_logger.warning(
                "ignoring corrupt checkpoint %s: %r", path, exc
            )
            return None
