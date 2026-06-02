from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext

from contractor.runners.models import GLOBAL_TASK_ID_KEY, TaskScopedKeys
from contractor.tools.tasks.formatters import SubtaskFormatter
from contractor.tools.tasks.models import (
    NO_ACTIVE_SUBTASKS_MSG,
    NO_SUBTASKS_EXIST_MSG,
    InvalidStatusTransitionError,
    Subtask,
    SubtaskExecutionResult,
    SubtaskSpec,
    SubtaskStatus,
    validate_status_transition,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# StreamlineManager
# ═══════════════════════════════════════════════════════════════════
@dataclass
class StreamlineManager:
    name: str
    max_tasks: int
    fmt: SubtaskFormatter

    # ── State key helpers ───────────────────────────────────────────
    def _state_key(self, ctx: ToolContext | CallbackContext) -> str:
        global_task_id = ctx.state.get(GLOBAL_TASK_ID_KEY, 0)
        invocation_id = ctx.invocation_id
        return f"task::{global_task_id}::{invocation_id}::{self.name}"

    def _subtasks_key(self, ctx: ToolContext | CallbackContext) -> str:
        return self._state_key(ctx) + "::tasks"

    def _current_idx_key(self, ctx: ToolContext | CallbackContext) -> str:
        return self._state_key(ctx) + "::idx"

    @staticmethod
    def _task_keys(ctx: ToolContext | CallbackContext) -> TaskScopedKeys:
        global_task_id = ctx.state.get(GLOBAL_TASK_ID_KEY, 0)
        return TaskScopedKeys(global_task_id)

    # ── ID generation ───────────────────────────────────────────────
    @staticmethod
    def _next_task_id(subtasks: list[Subtask]) -> str:
        """Return the next root-level task ID (0-based)."""
        if not subtasks:
            return "0"
        max_root = max(int(s.task_id.split(".")[0]) for s in subtasks)
        return str(max_root + 1)

    # ── Subtask persistence ─────────────────────────────────────────
    def get_subtasks(self, ctx: ToolContext | CallbackContext) -> list[Subtask]:
        key = self._subtasks_key(ctx)
        ctx.state.setdefault(key, [])
        return [Subtask(**sub) for sub in ctx.state[key]]

    def _save_subtasks(
        self,
        subtasks: list[Subtask],
        ctx: ToolContext | CallbackContext,
    ) -> None:
        ctx.state[self._subtasks_key(ctx)] = [sub.model_dump() for sub in subtasks]

    @contextmanager
    def _subtasks_session(
        self, ctx: ToolContext | CallbackContext
    ) -> Generator[list[Subtask], None, None]:
        """Load subtasks, yield the mutable list, persist on clean exit.

        Note: this is *not* a lock — `ctx.state` is single-threaded by contract.
        On exception, the persisted state is left untouched, which preserves
        the pre-mutation snapshot when a transition fails partway through.
        """
        subtasks = self.get_subtasks(ctx)
        yield subtasks
        self._save_subtasks(subtasks, ctx)

    # ── Index helpers ───────────────────────────────────────────────
    def _get_idx(self, ctx: ToolContext | CallbackContext) -> int | None:
        return ctx.state.get(self._current_idx_key(ctx))

    def _set_idx(self, ctx: ToolContext | CallbackContext, idx: int) -> None:
        ctx.state[self._current_idx_key(ctx)] = idx

    # ── Status transition ───────────────────────────────────────────
    @staticmethod
    def _apply_status_transition(subtask: Subtask, new_status: str) -> None:
        validate_status_transition(subtask.status, new_status)
        subtask.status = cast(SubtaskStatus, new_status)

    # ── Core operations ─────────────────────────────────────────────
    def add_subtask(
        self,
        subtask_spec: SubtaskSpec,
        ctx: ToolContext | CallbackContext,
    ) -> Subtask | None:
        with self._subtasks_session(ctx) as subtasks:
            if len(subtasks) >= self.max_tasks:
                logger.warning(
                    "Task limit reached",
                    extra={"max_tasks": self.max_tasks, "current": len(subtasks)},
                )
                return None

            new = Subtask(
                task_id=self._next_task_id(subtasks),
                title=subtask_spec.title,
                description=subtask_spec.description,
                status="new",
            )

            idx = self._get_idx(ctx)
            should_advance = (
                idx is None
                or idx < 0
                or idx >= len(subtasks)
                or subtasks[idx].status in ("done", "skipped", "decomposed")
            )
            if should_advance:
                self._set_idx(ctx, len(subtasks))

            subtasks.append(new)
            logger.info(
                "Subtask added",
                extra={"task_id": new.task_id, "title": new.title},
            )
        return new

    def get_current_subtask(
        self, ctx: ToolContext | CallbackContext
    ) -> Subtask | None:
        subtasks = self.get_subtasks(ctx)
        idx = self._get_idx(ctx)
        if idx is None or idx < 0 or idx >= len(subtasks):
            return None
        return subtasks[idx]

    def get_remaining_subtasks(
        self, ctx: ToolContext | CallbackContext
    ) -> list[Subtask]:
        """Return the current subtask and any later ones (the actionable plan).

        Empty when there is no current subtask, or when only the trailing
        subtask remains and it has already been resolved.
        """
        subtasks = self.get_subtasks(ctx)
        idx = self._get_idx(ctx)
        if idx is None or idx < 0 or idx >= len(subtasks):
            return []
        if idx == len(subtasks) - 1 and subtasks[idx].status != "new":
            return []
        return subtasks[idx:]

    def get_records(
        self,
        ctx: ToolContext | CallbackContext,
    ) -> list[Any]:
        pool_key = self._task_keys(ctx).pool
        ctx.state.setdefault(pool_key, [])
        return ctx.state[pool_key]

    def save_record(
        self,
        record: str | dict[str, Any],
        ctx: ToolContext | CallbackContext,
    ) -> None:
        records = self.get_records(ctx)
        records.append(record)
        ctx.state[self._task_keys(ctx).pool] = records

    def skip(
        self,
        reason: str,
        ctx: ToolContext | CallbackContext,
    ) -> Subtask | None:
        """Skip the current subtask. Returns the next subtask or None."""
        idx = self._get_idx(ctx)
        if idx is None:
            return None

        with self._subtasks_session(ctx) as subtasks:
            if idx < 0 or idx >= len(subtasks):
                return None

            current = subtasks[idx]
            try:
                self._apply_status_transition(current, "skipped")
            except InvalidStatusTransitionError as exc:
                logger.warning(
                    "Invalid skip transition",
                    extra={
                        "task_id": current.task_id,
                        "current_status": current.status,
                        "error": str(exc),
                    },
                )
                return None

            # Determine the next subtask, if any
            next_subtask: Subtask | None = None
            if idx + 1 < len(subtasks):
                self._set_idx(ctx, idx + 1)
                next_subtask = subtasks[idx + 1]

        # Build record directly — SubtaskExecutionResult doesn't allow "skipped"
        record: dict[str, Any] = {
            **current.model_dump(),
            "status": "skipped",
            "output": reason,
            "summary": f"Skipped: {reason}",
        }
        self.save_record(record, ctx)

        logger.info(
            "Subtask skipped",
            extra={"task_id": current.task_id, "reason": reason},
        )
        return next_subtask

    def decompose_current_subtask(
        self,
        new_subtasks: list[SubtaskSpec],
        ctx: ToolContext | CallbackContext,
    ) -> list[Subtask] | None:
        """
        Returns:
            list[Subtask] on success,
            None if preconditions not met or task limit would be exceeded.
        """
        idx = self._get_idx(ctx)
        if idx is None:
            return None

        with self._subtasks_session(ctx) as subtasks:
            if idx < 0 or idx >= len(subtasks):
                return None

            if len(subtasks) + len(new_subtasks) > self.max_tasks:
                logger.warning(
                    "Decomposition would exceed task limit",
                    extra={
                        "current_count": len(subtasks),
                        "new_count": len(new_subtasks),
                        "max_tasks": self.max_tasks,
                    },
                )
                return None

            current = subtasks[idx]
            if current.status not in ("incomplete", "malformed"):
                logger.warning(
                    "Cannot decompose non-decomposable subtask",
                    extra={
                        "task_id": current.task_id,
                        "status": current.status,
                    },
                )
                return None

            current_id = current.task_id
            try:
                self._apply_status_transition(current, "decomposed")
            except InvalidStatusTransitionError as exc:
                logger.warning(
                    "Invalid decompose transition",
                    extra={
                        "task_id": current.task_id,
                        "current_status": current.status,
                        "error": str(exc),
                    },
                )
                return None

            insertion: list[Subtask] = []
            for ind, spec in enumerate(new_subtasks, start=1):
                insertion.append(
                    Subtask(
                        task_id=f"{current_id}.{ind}",
                        title=spec.title,
                        description=spec.description,
                        status="new",
                    )
                )

            for i, sub in enumerate(insertion):
                subtasks.insert(idx + 1 + i, sub)

            parent_record = {
                **current.model_dump(),
                "status": "decomposed",
                "output": (
                    f"Decomposed into {len(insertion)} child subtasks: "
                    + ", ".join(s.task_id for s in insertion)
                ),
                "summary": (
                    f"Subtask {current_id} was decomposed into "
                    f"{len(insertion)} child subtasks."
                ),
            }

        self._set_idx(ctx, idx + 1)
        self.save_record(parent_record, ctx)

        logger.info(
            "Subtask decomposed",
            extra={
                "parent_task_id": current_id,
                "child_count": len(insertion),
                "child_ids": [s.task_id for s in insertion],
            },
        )
        return insertion

    def complete_current_subtask(
        self,
        subtask_result: SubtaskExecutionResult,
        ctx: ToolContext | CallbackContext,
    ) -> tuple[bool, str | None]:
        """Apply execution result. Returns (success, error_message)."""
        idx = self._get_idx(ctx)
        if idx is None:
            return False, NO_SUBTASKS_EXIST_MSG

        with self._subtasks_session(ctx) as subtasks:
            if idx < 0:
                return False, NO_SUBTASKS_EXIST_MSG
            if idx >= len(subtasks):
                return False, NO_ACTIVE_SUBTASKS_MSG

            current = subtasks[idx]
            try:
                self._apply_status_transition(current, subtask_result.status)
            except InvalidStatusTransitionError as exc:
                logger.warning(
                    "Invalid status transition during completion",
                    extra={
                        "task_id": current.task_id,
                        "from_status": current.status,
                        "to_status": subtask_result.status,
                        "error": str(exc),
                    },
                )
                return False, str(exc)

            # Advance inside the lock for consistency
            can_advance = idx + 1 < len(subtasks)
            if can_advance and subtask_result.status not in ("incomplete",):
                self._set_idx(ctx, idx + 1)

        record = self.fmt.format_task_record(current, subtask_result)
        self.save_record(record, ctx)

        logger.info(
            "Subtask completed",
            extra={
                "task_id": current.task_id,
                "status": subtask_result.status,
                "advanced": can_advance and subtask_result.status != "incomplete",
            },
        )
        return True, None

    def complete_current_subtask_from_runtime_result(
        self,
        runtime_result: dict[str, Any],
        ctx: ToolContext | CallbackContext,
    ) -> tuple[bool, str | None]:
        """Apply a runtime-generated result (e.g. malformed)."""
        idx = self._get_idx(ctx)
        if idx is None:
            return False, NO_SUBTASKS_EXIST_MSG

        with self._subtasks_session(ctx) as subtasks:
            if idx < 0:
                return False, NO_SUBTASKS_EXIST_MSG
            if idx >= len(subtasks):
                return False, NO_ACTIVE_SUBTASKS_MSG

            current = subtasks[idx]
            new_status = runtime_result["status"]
            try:
                self._apply_status_transition(current, new_status)
            except InvalidStatusTransitionError as exc:
                logger.warning(
                    "Invalid status transition during runtime completion",
                    extra={
                        "task_id": current.task_id,
                        "from_status": current.status,
                        "to_status": new_status,
                        "error": str(exc),
                    },
                )
                return False, str(exc)

            # Advance inside the lock for consistency
            can_advance = idx + 1 < len(subtasks)
            if can_advance and new_status not in ("incomplete", "malformed"):
                self._set_idx(ctx, idx + 1)

        record: dict[str, Any] = {
            **current.model_dump(),
            "status": runtime_result["status"],
            "output": runtime_result["output"],
            "summary": runtime_result["summary"],
        }
        self.save_record(record, ctx)

        logger.info(
            "Subtask completed from runtime result",
            extra={
                "task_id": current.task_id,
                "status": new_status,
                "advanced": can_advance
                and new_status not in ("incomplete", "malformed"),
            },
        )
        return True, None

    def finish(
        self,
        status: str,
        result: str,
        summary: str,
        ctx: ToolContext | CallbackContext,
    ) -> None:
        keys = self._task_keys(ctx)
        ctx.state[keys.result] = result
        ctx.state[keys.summary] = summary
        ctx.state[keys.status] = status
        logger.info("Task finished", extra={"status": status})
