"""Source-independent preparation of bounded HTTP observation metadata."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta, timezone
from urllib.parse import quote, unquote_plus, urlsplit, urlunsplit

from contractor_runtime.core.http_limits import MAX_URL_BYTES
from contractor_runtime.core.target_policy import validate_host_syntax
from contractor_runtime.toolsets.common.finding_locations import validate_web_url

MAX_OBSERVATIONS = 100
MAX_PARAMETERS = 100
MAX_PARAMETER_BYTES = 64
_FIELDS = {"url", "method", "outcome", "status", "at"}
_METHOD = re.compile(r"[A-Z]{1,16}")
_TIME = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})",
    re.ASCII,
)


def _timestamp(value: object) -> str:
    match = _TIME.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError("invalid observation timestamp")
    year, month, day, hour, minute, second, fraction, offset = match.groups()
    zone = UTC
    if offset != "Z":
        hours, minutes = int(offset[1:3]), int(offset[4:6])
        if hours > 23 or minutes > 59:
            raise ValueError("invalid observation timezone")
        difference = timedelta(hours=hours, minutes=minutes)
        zone = timezone(difference if offset[0] == "+" else -difference)
    instant = datetime(
        int(year), int(month), int(day), int(hour), int(minute), int(second), tzinfo=zone
    ).astimezone(UTC)
    # datetime would truncate nanoseconds; timezone conversion only changes the
    # integral second, so retain the original fraction and strip trailing zeroes.
    normalized = (
        f"{instant.year:04d}-{instant.month:02d}-{instant.day:02d}"
        f"T{instant.hour:02d}:{instant.minute:02d}:{instant.second:02d}"
    )
    fraction = (fraction or "").rstrip("0")
    return normalized + (f".{fraction}" if fraction else "") + "Z"


def _parameter(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value.encode("utf-8")) <= MAX_PARAMETER_BYTES
        and not any(unicodedata.category(character) == "Cc" for character in value)
    )


def _url(value: object) -> tuple[str, set[str]]:
    if (
        not isinstance(value, str)
        or len(value.encode("utf-8")) > MAX_URL_BYTES
        or not value.startswith(("http://", "https://"))
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("invalid observation URL")
    parts = urlsplit(value)
    authority = parts.netloc.rsplit("@", 1)[-1]
    if authority.startswith("[") and re.fullmatch(r"\[[^]]+\](?::[0-9]+)?", authority) is None:
        raise ValueError("invalid observation authority")
    # These locations never enter the metadata model. Query values stay
    # transient until the common domain reducer applies finite selectors.
    normalized = urlunsplit(
        (
            parts.scheme,
            authority,
            quote(parts.path or "/", safe="!$&'()*+,-./:;=@_~%"),
            parts.query,
            "",
        )
    )
    validate_web_url(normalized)
    validate_host_syntax(parts.hostname or "")
    # The common Go URL contract percent-encodes a Unicode authority before
    # validating prepared-request bytes; account for that expansion too.
    encoded = urlunsplit(
        (
            parts.scheme,
            quote(authority, safe="[]:._-"),
            quote(parts.path or "/", safe="!$&'()*+,-./:;=@_~%"),
            parts.query,
            "",
        )
    )
    if len(encoded.encode("utf-8")) > MAX_URL_BYTES:
        raise ValueError("observation URL exceeds its byte bound")
    # Go's prepared-request grammar leaves RawQuery opaque; unlike a path,
    # literal non-ASCII query bytes cannot be represented by that contract.
    if not parts.query.isascii() or ";" in parts.query:
        raise ValueError("invalid observation query")
    names = set()
    for field in parts.query.split("&"):
        if not field:
            continue
        name = unquote_plus(field.partition("=")[0], encoding="utf-8", errors="strict")
        if not _parameter(name):
            raise ValueError("invalid observation parameter")
        names.add(name)
        if len(names) > MAX_PARAMETERS:
            raise ValueError("too many observation parameters")
    return normalized, names


def normalize_http_observation(value: Mapping[str, object]) -> dict | None:
    """Return prepared metadata, or None for an unrepresentable source row.

    This function has no source/provider knowledge and never changes a bad
    response into an attempt. It performs no DNS, target or artifact I/O.
    """
    try:
        if not isinstance(value, Mapping) or not _FIELDS <= value.keys() <= _FIELDS | {
            "parameters"
        }:
            return None
        method, outcome, status = value["method"], value["outcome"], value["status"]
        if not isinstance(method, str) or _METHOD.fullmatch(method) is None:
            return None
        if type(status) is not int or not (
            (outcome == "response" and 100 <= status <= 599)
            or (outcome in {"attempt", "timeout", "transport-error"} and status == 0)
        ):
            return None
        url, names = _url(value["url"])
        result = {
            "url": url,
            "method": method,
            "outcome": outcome,
            "status": status,
            "at": _timestamp(value["at"]),
        }
        if "parameters" in value:
            parameters = value["parameters"]
            if (
                not isinstance(parameters, list)
                or len(parameters) > MAX_PARAMETERS
                or any(not _parameter(name) for name in parameters)
            ):
                return None
            names.update(parameters)
            result["parameters"] = sorted(set(parameters))
        return result if len(names) <= MAX_PARAMETERS else None
    except (ValueError, TypeError, UnicodeError, OverflowError):
        return None


def normalize_http_observations(values: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """Prepare one selected page, preserving good rows and one omission count."""
    accepted = []
    omitted = max(0, len(values) - MAX_OBSERVATIONS)
    for value in values[:MAX_OBSERVATIONS]:
        normalized = normalize_http_observation(value)
        if normalized is None:
            omitted += 1
        else:
            accepted.append(normalized)
    return accepted, ([{"reason": "projection", "count": omitted}] if omitted else [])
