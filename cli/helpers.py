from typing import Any, Optional
from cli import EventView

def _normalize_event(event: Any) -> Optional[EventView]:
    event_type = getattr(event, "type", "") or ""
    payload = getattr(event, "payload", {}) or {}

    if event_type == "run_started":
        total_tasks = payload.get("total_tasks")
        body = f"Tasks total: {total_tasks}" if total_tasks is not None else None
        return EventView(
            event_type=event_type,
            title="Runner launch",
            body=body,
            tone="info",
        )

    if event_type == "task_started":
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
            event_type=event_type,
            title=title,
            tone="info",
        )

    if event_type == "iteration_started":
        iteration = payload.get("iteration")
        objective = payload.get("objective", "")
        return EventView(
            event_type=event_type,
            title=f"Iteration {iteration}",
            body=_fmt_blob(objective, max_lines=4, max_width=100) if objective else None,
            tone="warning",
        )

    if event_type == "tool_call":
        return _fmt_tool_call_event(payload)

    if event_type == "tool_result":
        return _fmt_tool_result_event(payload)

    if event_type == "tool_error":
        return _fmt_tool_error_event(payload)

    if event_type == "final_text":
        text = (payload.get("text") or "").strip()
        if not text:
            return None

        return EventView(
            event_type=event_type,
            title="Final answer",
            body=_fmt_blob(text, max_lines=8, max_width=100),
            tone="success",
        )

    if event_type == "iteration_result":
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
            event_type=event_type,
            title=f"Iteration result: {iteration}",
            body=body,
            tone="warning",
        )

    if event_type == "global_task_finished":
        summary = payload.get("summary") or "—"
        return EventView(
            event_type=event_type,
            title=f"Task finished: {event.task_name}",
            body=_fmt_blob(summary, max_lines=6, max_width=100),
            tone="success",
        )

    if event_type == "task_failed":
        last_result = payload.get("last_result")
        return EventView(
            event_type=event_type,
            title=f"Task failed: {event.task_name}",
            body=_fmt_tool_data(last_result, max_lines=8, max_width=100),
            tone="error",
        )

    return None


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

    return _clamp_lines(wrapped_lines, max_lines=max_lines, max_width=max_width)


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