from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from cli.utils import utc_now_iso
from contractor.runners.agio import ALL_AGIO_EVENT_TYPES
from contractor.runners.task_runner import TaskRunnerEvent


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
    """Flatten a ``TaskRunnerEvent`` into an Agio-shaped record.

    The Agio convention is one flat dict per line — no nested ``payload``.
    Identification fields (``type``, ``task_name``, ``task_id``) are pulled
    from the event envelope; everything else from ``event.payload`` is
    lifted to the top level so analysis scripts can index by name directly.
    """
    payload = _jsonable(event.payload or {})
    payload_dict: dict[str, Any] = payload if isinstance(payload, dict) else {}

    record: dict[str, Any] = {
        "type": getattr(event, "type", None),
        "timestamp": time.time() * 1000.0,
        "ts_iso": utc_now_iso(),
        "task_name": getattr(event, "task_name", None),
        "task_id": getattr(event, "task_id", None),
    }
    for key, value in payload_dict.items():
        record.setdefault(key, value)
    return record


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


class MetricsSink:
    """Filters Agio events and persists them to a JSONL file."""

    def __init__(self, output_dir: Path) -> None:
        self._path = output_dir / "metrics.jsonl"
        self._lock = asyncio.Lock()

    @staticmethod
    def matches(event: TaskRunnerEvent) -> bool:
        event_type = getattr(event, "type", "") or ""
        return event_type in ALL_AGIO_EVENT_TYPES

    async def write(self, event: TaskRunnerEvent) -> None:
        record = _event_to_record(event)
        async with self._lock:
            await asyncio.to_thread(_append_jsonl, self._path, record)
