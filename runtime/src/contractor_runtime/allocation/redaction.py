"""Values and URL hosts used to redact allocation observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from contractor_runtime.contracts.redaction import (
    runtime_setting_values as _runtime_setting_values,  # noqa: F401
)


def _url_hosts(urls: Sequence[str]) -> tuple[str, ...]:
    values: set[str] = set()
    for value in urls:
        parsed = urlsplit(value)
        if parsed.hostname is not None:
            values.add(parsed.hostname)
        if parsed.netloc:
            values.add(parsed.netloc)
    return tuple(sorted(values))


def _nested_strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        return {item for nested in value.values() for item in _nested_strings(nested)}
    if isinstance(value, (list, tuple)):
        return {item for nested in value for item in _nested_strings(nested)}
    return set()
