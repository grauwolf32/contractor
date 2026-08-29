"""Bounded, secret-safe in-memory metrics for one Worker allocation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from contractor_runtime.contracts import TerminationError

MAX_METRIC_TOOL_CALLS = 128
MAX_METRIC_ERRORS = 128
MAX_METRIC_ITEMS = 32
MAX_METRIC_DEPTH = 4
MAX_METRIC_TEXT = 256
SENSITIVE_METRIC_KEYS = frozenset(
    {"body", "bytes", "content", "data", "data_base64", "password", "payload", "secret", "token"}
)


@dataclass(slots=True)
class MetricsState:
    counters: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[TerminationError] = field(default_factory=list)
    final_outcome: str | None = None
    truncated: bool = False

    def record_tool_call(
        self,
        name: str,
        *,
        arguments: Mapping[str, Any],
        result: Mapping[str, Any] | None = None,
        error: Exception | None = None,
        secrets: tuple[str, ...] = (),
    ) -> None:
        self._increment("tool_calls")
        self._increment(f"tool_calls.{_metric_identifier(name)}")
        entry: dict[str, Any] = {
            "name": _bounded_text(name, secrets),
            "arguments": _sanitize_metric_value(arguments, secrets=secrets),
        }
        if result is not None:
            entry["result"] = _sanitize_metric_value(result, secrets=secrets)
        if error is not None:
            self._increment("tool_errors")
            code = getattr(error, "code", "tool_call_failed")
            retryable = bool(getattr(error, "retryable", False))
            entry["error"] = {
                "type": type(error).__name__,
                "code": _bounded_text(str(code), secrets),
                "retryable": retryable,
            }
            self._append_error(
                TerminationError(
                    code=f"tool_{_metric_identifier(name)}_failed",
                    message=f"Tool {name} failed ({type(error).__name__})",
                    retryable=retryable,
                )
            )
        if len(self.tool_calls) < MAX_METRIC_TOOL_CALLS:
            self.tool_calls.append(entry)
        else:
            self.truncated = True

    def record_model_call(self) -> None:
        self._increment("llm_calls")

    def record_model_usage(self, usage: Any) -> None:
        for attribute, counter in (
            ("prompt_token_count", "input_tokens"),
            ("candidates_token_count", "output_tokens"),
            ("total_token_count", "total_tokens"),
            ("cached_content_token_count", "cached_input_tokens"),
        ):
            value = getattr(usage, attribute, None)
            if isinstance(value, int) and value >= 0:
                self._increment(counter, value)

    def record_model_error(self, error: Exception) -> None:
        self._increment("llm_errors")
        self._append_error(
            TerminationError(
                code="model_call_failed",
                message=f"Model call failed ({type(error).__name__})",
                retryable=True,
            )
        )

    def record_outcome(self, outcome: str) -> None:
        normalized = _metric_identifier(outcome)
        self.final_outcome = normalized
        self._increment(f"outcomes.{normalized}")

    def snapshot(self) -> dict[str, Any]:
        return {
            "counters": dict(sorted(self.counters.items())),
            "toolCalls": list(self.tool_calls),
            "errors": [error.model_dump(by_alias=True) for error in self.errors],
            "finalOutcome": self.final_outcome,
            "truncated": self.truncated,
        }

    def _increment(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def _append_error(self, error: TerminationError) -> None:
        if len(self.errors) < MAX_METRIC_ERRORS:
            self.errors.append(error)
        else:
            self.truncated = True


def _sanitize_metric_value(
    value: Any,
    *,
    secrets: tuple[str, ...],
    key: str | None = None,
    depth: int = 0,
) -> Any:
    if key is not None and key.lower() in SENSITIVE_METRIC_KEYS:
        return {"redacted": True, "size": _value_size(value)}
    if depth >= MAX_METRIC_DEPTH:
        return "[TRUNCATED]"
    if isinstance(value, bytes | bytearray | memoryview):
        return {"bytes": len(value)}
    if isinstance(value, str):
        return _bounded_text(value, secrets)
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (item_key, item_value) in enumerate(value.items()):
            if index >= MAX_METRIC_ITEMS:
                result["__truncated__"] = True
                break
            normalized_key = _bounded_text(str(item_key), secrets)
            result[normalized_key] = _sanitize_metric_value(
                item_value,
                secrets=secrets,
                key=str(item_key),
                depth=depth + 1,
            )
        return result
    if isinstance(value, list | tuple):
        result = [
            _sanitize_metric_value(item, secrets=secrets, depth=depth + 1)
            for item in value[:MAX_METRIC_ITEMS]
        ]
        if len(value) > MAX_METRIC_ITEMS:
            result.append("[TRUNCATED]")
        return result
    return f"<{type(value).__name__}>"


def _bounded_text(value: str, secrets: tuple[str, ...]) -> str:
    if any(secret and secret in value for secret in secrets):
        return "[REDACTED]"
    if "://" in value:
        parsed = urlsplit(value)
        if (
            parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            return "[REDACTED_URL]"
    if len(value) > MAX_METRIC_TEXT:
        return value[:MAX_METRIC_TEXT] + "…"
    return value


def _value_size(value: Any) -> int | None:
    if isinstance(value, str | bytes | bytearray | memoryview | list | tuple | Mapping):
        return len(value)
    return None


def _metric_identifier(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
    return normalized[:64] or "unknown"
