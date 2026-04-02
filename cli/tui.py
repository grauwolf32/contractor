# cli/live_render.py
from __future__ import annotations

import json
import textwrap
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional

from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text

from cli import EventView

from cli.helpers import (
    _normalize_event,
    _fmt_tool_call_event,
    _fmt_tool_result_event,
    _fmt_tool_error_event,
    _fmt_tool_data,
    _try_parse_json_like,
    _fmt_dict,
    _fmt_list,
    _fmt_value,
    _fmt_blob,
    _clamp_lines,
    _wrap_text
)


TOP_PANEL_HEIGHT = 6
EVENT_HISTORY_LIMIT = 200



@dataclass
class UiState:
    pipeline_name: str
    total_tasks: int = 0
    completed_tasks: int = 0

    current_task_name: str = "—"
    current_task_index: int = 0
    current_iteration: Optional[int] = None
    current_max_attempts: Optional[int] = None

    events: Deque[EventView] = field(
        default_factory=lambda: deque(maxlen=EVENT_HISTORY_LIMIT)
    )


class LiveRenderer:
    def __init__(self, pipeline_name: str) -> None:
        self.console = Console()
        self.state = UiState(pipeline_name=pipeline_name)
        self.live: Optional[Live] = None

    def start(self) -> None:
        self.live = Live(
            self.render(),
            console=self.console,
            auto_refresh=False,
            transient=False,
        )
        self.live.start(refresh=True)

    def stop(self) -> None:
        if self.live is not None:
            self.live.update(self.render(), refresh=True)
            self.live.stop()
            self.live = None

    def on_event(self, event: Any) -> None:
        self._apply_state(event)

        view = _normalize_event(event)
        if view is not None:
            self.state.events.append(view)

        if self.live is not None:
            self.live.update(self.render(), refresh=True)

    def render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="top", size=TOP_PANEL_HEIGHT),
            Layout(name="bottom", ratio=1),
        )
        layout["top"].update(self._render_status())
        layout["bottom"].update(self._render_events())
        return layout

    def _apply_state(self, event: Any) -> None:
        event_type = getattr(event, "type", "") or ""
        payload = getattr(event, "payload", {}) or {}

        if event_type == "run_started":
            self.state.total_tasks = int(payload.get("total_tasks") or 0)
            self.state.completed_tasks = int(payload.get("completed_tasks") or 0)
            return

        if event_type == "task_started":
            self.state.current_task_name = getattr(event, "task_name", "—")
            self.state.current_task_index = int(getattr(event, "task_id", 0)) + 1
            self.state.total_tasks = int(
                payload.get("total_tasks") or self.state.total_tasks
            )
            self.state.completed_tasks = int(
                payload.get("completed_tasks") or self.state.completed_tasks
            )
            self.state.current_iteration = None
            self.state.current_max_attempts = payload.get("max_attempts")
            return

        if event_type == "iteration_started":
            self.state.current_iteration = payload.get("iteration")
            return

        if event_type == "iteration_result":
            self.state.current_iteration = payload.get(
                "iteration",
                self.state.current_iteration,
            )
            self.state.current_max_attempts = payload.get(
                "max_attempts",
                self.state.current_max_attempts,
            )
            return

        if event_type == "global_task_finished":
            self.state.total_tasks = int(
                payload.get("total_tasks") or self.state.total_tasks
            )
            self.state.completed_tasks = int(
                payload.get("completed_tasks") or self.state.completed_tasks
            )

    def _render_status(self) -> Panel:
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            expand=True,
        )
        progress.add_task(
            description="Pipeline",
            total=max(1, self.state.total_tasks),
            completed=min(self.state.completed_tasks, self.state.total_tasks),
        )

        task_position = (
            f"{self.state.current_task_index}/{self.state.total_tasks}"
            if self.state.total_tasks > 0
            else "—"
        )

        attempts = "—"
        if (
            self.state.current_iteration is not None
            and self.state.current_max_attempts is not None
        ):
            attempts = f"{self.state.current_iteration}/{self.state.current_max_attempts}"
        elif self.state.current_iteration is not None:
            attempts = str(self.state.current_iteration)

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)

        left = Group(
            Text(
                f"Pipeline: {self.state.pipeline_name}",
                style="bold cyan",
                no_wrap=False,
                overflow="fold",
            ),
            Text(
                f"Task: {self.state.current_task_name}",
                style="bold white",
                no_wrap=False,
                overflow="fold",
            ),
        )
        right = Group(
            Text(
                f"Task position: {task_position}",
                style="yellow",
                no_wrap=False,
                overflow="fold",
            ),
            Text(
                f"Attempts: {attempts}",
                style="magenta",
                no_wrap=False,
                overflow="fold",
            ),
        )

        grid.add_row(left, right)
        grid.add_row(progress, Text(""))

        return Panel(
            grid,
            title="Status",
            border_style="blue",
            padding=(0, 1),
            expand=True,
        )

    def _render_events(self) -> Panel:
        events = self._visible_events()

        if not events:
            return Panel(
                Text("No events yet", style="dim"),
                title="Events",
                border_style="green",
                padding=(0, 1),
                expand=True,
            )

        blocks: list[RenderableType] = []
        for idx, event in enumerate(events):
            blocks.extend(self._render_event_block(event))
            if idx < len(events) - 1:
                blocks.append(Text(""))

        return Panel(
            Group(*blocks),
            title=f"Events ({len(events)} visible)",
            border_style="green",
            padding=(0, 1),
            expand=True,
        )

    def _render_event_block(self, event: EventView) -> list[RenderableType]:
        tone_to_style = {
            "info": "cyan",
            "success": "green",
            "warning": "yellow",
            "error": "red",
            "muted": "dim",
            "default": "white",
        }

        title_style = tone_to_style.get(event.tone, "white")
        body_style = "dim" if event.tone == "muted" else "white"

        block: list[RenderableType] = [
            Text(
                event.title,
                style=f"bold {title_style}",
                no_wrap=False,
                overflow="fold",
            )
        ]

        if event.body:
            for line in event.body.splitlines():
                block.append(
                    Text(
                        f"  {line}",
                        style=body_style,
                        no_wrap=False,
                        overflow="fold",
                    )
                )

        return block

    def _visible_events(self) -> list[EventView]:
        events = list(self.state.events)
        if not events:
            return []

        available_lines = self._available_event_lines()
        selected: list[EventView] = []
        used_lines = 0

        for event in reversed(events):
            event_height = self._event_height(event)

            if not selected:
                selected.append(event)
                used_lines += event_height
                continue

            if used_lines + event_height > available_lines:
                break

            selected.append(event)
            used_lines += event_height

        selected.reverse()
        return selected

    def _available_event_lines(self) -> int:
        total_height = self.console.size.height

        # Резерв под:
        # - верхнюю панель
        # - рамки
        # - title нижней панели
        reserve = TOP_PANEL_HEIGHT + 4

        available = total_height - reserve
        return max(available, 4)

    def _event_height(self, event: EventView) -> int:
        panel_inner_width = max(self.console.size.width - 4, 20)

        height = self._wrapped_line_count(event.title, panel_inner_width)

        if event.body:
            body_width = max(panel_inner_width - 2, 10)
            height += self._wrapped_line_count(event.body, body_width)

        height += 1  # пустая строка-разделитель
        return height

    @staticmethod
    def _wrapped_line_count(text: str, width: int) -> int:
        if width <= 1:
            return 1

        total = 0
        lines = text.splitlines() or [""]

        for line in lines:
            total += max(1, len(_wrap_text(line, width)))

        return total


def _normalize_event(event: Any) -> Optional[EventView]:
    event_type = getattr(event, "type", "") or ""
    payload = getattr(event, "payload", {}) or {}

    if event_type == "run_started":
        return _fmt_run_started_event(payload)

    if event_type == "task_started":
        return _fmt_task_started_event(event, payload)

    if event_type == "iteration_started":
        return _fmt_iteration_started_event(payload)

    if event_type == "tool_call":
        return _fmt_tool_call_event(payload)

    if event_type == "tool_result":
        return _fmt_tool_result_event(payload)

    if event_type == "tool_error":
        return _fmt_tool_error_event(payload)

    if event_type == "final_text":
        return _fmt_final_text_event(payload)

    if event_type == "iteration_result":
        return _fmt_iteration_result_event(payload)

    if event_type == "global_task_finished":
        return _fmt_global_task_finished_event(event, payload)

    if event_type == "task_failed":
        return _fmt_task_failed_event(event, payload)

    return None


def _fmt_run_started_event(payload: dict[str, Any]) -> EventView:
    total_tasks = payload.get("total_tasks")
    body = f"Tasks total: {total_tasks}" if total_tasks is not None else None

    return EventView(
        event_type="run_started",
        title="Runner launch",
        body=body,
        tone="info",
    )


def _fmt_task_started_event(event: Any, payload: dict[str, Any]) -> EventView:
    iterations = payload.get("iterations")
    max_attempts = payload.get("max_attempts")

    meta: list[str] = []
    if iterations is not None:
        meta.append(f"runs={iterations}")
    if max_attempts is not None:
        meta.append(f"attempts={max_attempts}")

    title = f"Task #{event.task_id}: {event.task_name}"
    if meta:
        title += f" ({', '.join(meta)})"

    return EventView(
        event_type="task_started",
        title=title,
        tone="info",
    )


def _fmt_iteration_started_event(payload: dict[str, Any]) -> EventView:
    iteration = payload.get("iteration")
    objective = payload.get("objective", "")

    return EventView(
        event_type="iteration_started",
        title=f"Iteration {iteration}",
        body=_fmt_blob(objective, max_lines=4, max_width=100) if objective else None,
        tone="warning",
    )


def _fmt_tool_call_event(payload: dict[str, Any]) -> EventView:
    tool_name = payload.get("tool_name", "unknown_tool")
    tool_args = (
        payload.get("tool_args")
        or payload.get("args")
        or payload.get("arguments")
        or payload.get("input")
    )

    return EventView(
        event_type="tool_call",
        title=f"Tool call: {tool_name}",
        body=_fmt_tool_data(tool_args, max_lines=8, max_width=100),
        tone="muted",
    )


def _fmt_tool_result_event(payload: dict[str, Any]) -> EventView:
    tool_name = payload.get("tool_name", "unknown_tool")
    result = (
        payload.get("result")
        or payload.get("output")
        or payload.get("response")
        or payload.get("data")
    )

    return EventView(
        event_type="tool_result",
        title=f"Tool result: {tool_name}",
        body=_fmt_tool_data(result, max_lines=10, max_width=100),
        tone="success",
    )


def _fmt_tool_error_event(payload: dict[str, Any]) -> EventView:
    tool_name = payload.get("tool_name", "unknown_tool")
    error = payload.get("error") or payload.get("message")

    return EventView(
        event_type="tool_error",
        title=f"Tool error: {tool_name}",
        body=_fmt_blob(error, max_lines=8, max_width=100),
        tone="error",
    )


def _fmt_final_text_event(payload: dict[str, Any]) -> Optional[EventView]:
    text = (payload.get("text") or "").strip()
    if not text:
        return None

    return EventView(
        event_type="final_text",
        title="Final answer",
        body=_fmt_blob(text, max_lines=8, max_width=100),
        tone="success",
    )


def _fmt_iteration_result_event(payload: dict[str, Any]) -> EventView:
    iteration = payload.get("iteration")
    completed = "yes" if payload.get("completed") else "no"
    status = payload.get("status")
    summary = payload.get("summary") or "—"

    body = _fmt_blob(
        "\n".join(
            [
                f"status={status}",
                f"completed={completed}",
                f"summary={summary}",
            ]
        ),
        max_lines=5,
        max_width=100,
    )

    return EventView(
        event_type="iteration_result",
        title=f"Iteration result: {iteration}",
        body=body,
        tone="warning",
    )


def _fmt_global_task_finished_event(
    event: Any,
    payload: dict[str, Any],
) -> EventView:
    summary = payload.get("summary") or "—"

    return EventView(
        event_type="global_task_finished",
        title=f"Task finished: {event.task_name}",
        body=_fmt_blob(summary, max_lines=6, max_width=100),
        tone="success",
    )


def _fmt_task_failed_event(event: Any, payload: dict[str, Any]) -> EventView:
    last_result = payload.get("last_result")

    return EventView(
        event_type="task_failed",
        title=f"Task failed: {event.task_name}",
        body=_fmt_tool_data(last_result, max_lines=8, max_width=100),
        tone="error",
    )


def _fmt_tool_data(value: Any, max_lines: int = 8, max_width: int = 100) -> str:
    if value is None:
        return "—"

    parsed = _try_parse_json_like(value)

    if isinstance(parsed, dict):
        return _fmt_dict(parsed, max_lines=max_lines, max_width=max_width)

    if isinstance(parsed, list):
        return _fmt_list(parsed, max_lines=max_lines, max_width=max_width)

    return _fmt_blob(parsed, max_lines=max_lines, max_width=max_width)


def _try_parse_json_like(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value

    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return text

    if text[0] not in "[{":
        return value

    try:
        return json.loads(text)
    except Exception:
        return value


def _fmt_dict(data: dict[str, Any], max_lines: int = 8, max_width: int = 100) -> str:
    lines: list[str] = []

    for key, value in data.items():
        lines.append(f"{key}: {_fmt_value(value)}")

    return _clamp_lines(lines, max_lines=max_lines, max_width=max_width)


def _fmt_list(items: list[Any], max_lines: int = 8, max_width: int = 100) -> str:
    lines: list[str] = []

    for item in items:
        lines.append(f"- {_fmt_value(item)}")

    return _clamp_lines(lines, max_lines=max_lines, max_width=max_width)


def _fmt_value(value: Any) -> str:
    if isinstance(value, dict):
        keys = list(value.keys())
        preview = ", ".join(map(str, keys[:4]))
        if len(keys) > 4:
            preview += ", …"
        return "{" + preview + "}"

    if isinstance(value, list):
        preview = ", ".join(str(v) for v in value[:3])
        if len(value) > 3:
            preview += ", …"
        return "[" + preview + "]"

    if isinstance(value, str):
        return " ".join(value.split())

    return str(value)


def _fmt_blob(value: Any, max_lines: int = 8, max_width: int = 100) -> str:
    if value is None:
        return "—"

    if isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            text = str(value)
    else:
        text = str(value)

    raw_lines = text.splitlines() or [text]
    wrapped_lines: list[str] = []

    for raw in raw_lines:
        line = raw.rstrip()
        if not line:
            continue
        wrapped_lines.extend(_wrap_text(line, max_width))

    if not wrapped_lines:
        return "—"

    if len(wrapped_lines) > max_lines:
        wrapped_lines = wrapped_lines[:max_lines]
        wrapped_lines[-1] = wrapped_lines[-1].rstrip() + " …"

    return "\n".join(wrapped_lines)


def _clamp_lines(lines: list[str], max_lines: int = 8, max_width: int = 100) -> str:
    cooked: list[str] = []

    for line in lines:
        normalized = str(line).rstrip()
        if not normalized:
            continue
        cooked.extend(_wrap_text(normalized, max_width))

    if not cooked:
        return "—"

    if len(cooked) > max_lines:
        cooked = cooked[:max_lines]
        cooked[-1] = cooked[-1].rstrip() + " …"

    return "\n".join(cooked)


def _wrap_text(text: str, width: int) -> list[str]:
    text = text.rstrip()
    if not text:
        return [""]

    if width <= 1:
        return [text]

    return textwrap.wrap(
        text,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
        replace_whitespace=False,
        drop_whitespace=False,
    ) or [text]