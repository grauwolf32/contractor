from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from contractor.runners.task_runner import TaskRunnerEvent

from cli.utils import utc_now_iso


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]

    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _jsonable(method())
            except Exception:
                pass

    if hasattr(value, "__dict__"):
        try:
            return _jsonable(vars(value))
        except Exception:
            pass

    return repr(value)


def _event_to_record(event: TaskRunnerEvent) -> dict[str, Any]:
    payload = _jsonable(getattr(event, "payload", {}) or {})
    payload_dict = payload if isinstance(payload, dict) else {}

    return {
        "ts": utc_now_iso(),
        "type": getattr(event, "type", None),
        "task_name": getattr(event, "task_name", None),
        "task_id": getattr(event, "task_id", None),
        "payload": payload,
        "iteration": payload_dict.get("iteration"),
        "session_id": payload_dict.get("session_id"),
        "invocation_id": payload_dict.get("invocation_id"),
        "agent_name": payload_dict.get("agent_name"),
        "tool_name": payload_dict.get("tool_name"),
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


class MetricsSink:
    """Filters metrics_* events and persists them to a JSONL file."""

    def __init__(self, output_dir: Path) -> None:
        self._path = output_dir / "metrics.jsonl"
        self._lock = asyncio.Lock()

    @staticmethod
    def matches(event: TaskRunnerEvent) -> bool:
        event_type = getattr(event, "type", "") or ""
        return event_type.startswith("metrics_")

    async def write(self, event: TaskRunnerEvent) -> None:
        record = _event_to_record(event)
        async with self._lock:
            await asyncio.to_thread(_append_jsonl, self._path, record)
