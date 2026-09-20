"""Bounded ffuf filters and result projection; process ownership stays in ScanTool."""

from __future__ import annotations

import base64
import binascii
import json
import re
from urllib.parse import urlsplit

from contractor_runtime.toolsets.common.input_errors import ToolInputError
from contractor_runtime.toolsets.scan.process import ProcessResult
from contractor_runtime.toolsets.scan.wordlist import MAX_WORDLIST_PAYLOAD_BYTES

MAX_FFUF_RESULTS = 100
MAX_FFUF_RESULTS_BYTES = 128 * 1024
_PROGRESS = re.compile(
    rb":: Progress: \[(\d{1,6})/(\d{1,6})\] :: Job \[1/1\] :: [^\r\n]*? :: Errors: (\d{1,6}) ::"
)


def validate_ffuf_url(value: str) -> None:
    # Validate text before the shared URL validator, so Unicode errors cannot echo inputs.
    if (
        not isinstance(value, str)
        or len(value) > 8192
        or any(ord(char) <= 32 or ord(char) >= 127 for char in value)
        or "\\" in value
        or "#" in value
        or "FFUFHASH" in value
        or re.search(r"%(?![0-9a-fA-F]{2})", value)
    ):
        raise ToolInputError("ffuf url must be a bounded ASCII HTTP(S) URL")
    try:
        parts = urlsplit(value)
    except ValueError:
        raise ToolInputError("ffuf url has invalid authority") from None
    if "FUZZ" in parts.netloc or "FUZZ" not in parts.path + parts.query:
        raise ToolInputError("ffuf url requires FUZZ in the path or query, not the authority")


def validate_ffuf_filter(value: str, name: str, *, status: bool = False, all_: bool = False):
    if not isinstance(value, str) or len(value) > 1024:
        raise ToolInputError(f"{name} must be a bounded numeric filter")
    if value == "all" and all_:
        return
    if not value and not all_:
        return
    if not re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", value):
        raise ToolInputError(f"{name} must contain comma-separated numbers or ordered ranges")
    items = value.split(",")
    if len(items) > 64:
        raise ToolInputError(f"{name} exceeds its filter count limit")
    minimum, maximum = (100, 599) if status else (0, 1_000_000_000)
    for item in items:
        first, _, last = item.partition("-")
        if (
            len(first) > 10
            or len(last) > 10
            or not minimum <= int(first) <= int(last or first) <= maximum
        ):
            raise ToolInputError(f"{name} contains an out-of-range or unordered filter")


def _result(value: object, entries: int) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("input"), dict):
        raise ValueError
    payload = base64.b64decode(value["input"]["FUZZ"], validate=True)
    if len(payload) > MAX_WORDLIST_PAYLOAD_BYTES:
        raise ValueError
    item = {"input": {"FUZZ": payload.decode("utf-8")}}
    for name in ("position", "status", "length", "words", "lines", "duration"):
        number = value[name]
        low, high = {"position": (1, entries), "status": (100, 599)}.get(name, (0, 2**63 - 1))
        if type(number) is not int or not low <= number <= high:
            raise ValueError
        item["durationNs" if name == "duration" else name] = number
    for source, target in (
        ("url", "url"),
        ("content-type", "contentType"),
        ("redirectlocation", "redirectLocation"),
    ):
        text = value[source]
        if not isinstance(text, str):
            raise ValueError
        # A malformed scanner string must not make report serialization fail later.
        text.encode("utf-8")
        item[target] = text
    parsed_url = urlsplit(item["url"])
    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        raise ValueError
    return item


def ffuf_observation(result: ProcessResult, entries: int) -> dict:
    response = result.observation()
    items, size, invalid, truncated = [], 2, 0, False  # Include the JSON array brackets.
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        try:
            item = _result(json.loads(line), entries)
            encoded = json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode()
        except (ValueError, TypeError, KeyError, RecursionError, binascii.Error):
            invalid += 1
            continue
        item_size = len(encoded) + bool(items)  # Include the comma between records.
        if len(items) >= MAX_FFUF_RESULTS or size + item_size > MAX_FFUF_RESULTS_BYTES:
            truncated = True
            continue
        items.append(item)
        size += item_size

    attempted, errors, total = None, None, None
    for match in _PROGRESS.finditer(result.stderr):
        attempted, total, errors = (int(value) for value in match.groups())
    complete = (
        result.error_code is None
        and result.exit_code == 0
        and not invalid
        and attempted == total == entries
        and errors == 0
    )
    if response["errorCode"] is None:
        error = (
            "invalid_scanner_output"
            if invalid
            else "scan_request_failed"
            if errors
            else "scan_incomplete"
            if not complete
            else None
        )
        if error:
            response.update(status="failed", errorCode=error)
    # Diagnostics contain the target and may echo payloads. Return structured matches only.
    response.update(
        scanner="ffuf",
        stdout="",
        stderr="",
        diagnosticsRedacted=True,
        results=items,
        resultsTruncated=truncated,
        invalidResultLines=invalid,
        scanComplete=complete,
        payloadsAttempted=attempted,
        requestErrors=errors,
        wordlistEntries=entries,
    )
    return response
