"""Nuclei input contract shared with Server admission fixtures."""

from __future__ import annotations

import re

from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.toolsets.scan.http_request import validate_http_url

_FILTER = re.compile(r"[A-Za-z0-9_*][A-Za-z0-9_.*-]*(,[A-Za-z0-9_*][A-Za-z0-9_.*-]*)*")


def validate_nuclei_arguments(arguments: dict) -> None:
    if set(arguments) - {
        "url",
        "template_ids",
        "tags",
        "severity",
        "rate_limit",
        "timeout_seconds",
        "templates_digest",
    }:
        raise ToolInputError("unknown Nuclei argument")
    validate_http_url(arguments.get("url"))
    for name in ("template_ids", "tags", "severity"):
        value = arguments.get(name, "")
        if (
            not isinstance(value, str)
            or len(value.encode("utf-8")) > 2048
            or (value and not _FILTER.fullmatch(value))
        ):
            raise ToolInputError("invalid Nuclei filter")
        if (
            name == "severity"
            and value
            and set(value.split(",")) - {"info", "low", "medium", "high", "critical", "unknown"}
        ):
            raise ToolInputError("invalid Nuclei severity")
    for name, default, maximum in (("rate_limit", 10, 1000), ("timeout_seconds", 300, 3600)):
        value = arguments.get(name, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ToolInputError("invalid Nuclei integer")
    digest = arguments.get("templates_digest", "")
    if not isinstance(digest, str) or (digest and not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)):
        raise ToolInputError("invalid Nuclei package digest")
