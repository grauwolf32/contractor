"""Bounded, allowlisted Katana observations and reusable target-list bytes."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from urllib.parse import urlsplit, urlunsplit

from contractor_runtime.toolsets.common.input_errors import ToolInputError
from contractor_runtime.toolsets.scan.process import ProcessResult

TARGET_LIST_MEDIA_TYPE = "text/vnd.contractor.target-list"
MAX_TARGETS = 100
MAX_TARGET_BYTES = 128 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024


def canonical_url(value: str) -> str:
    """Keep concrete paths/queries, normalize only the HTTP origin and empty path."""
    try:
        if (
            not isinstance(value, str)
            or len(value.encode("utf-8")) > 8192
            or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value)
            or "\\" in value
            or re.search(r"%(?![0-9a-fA-F]{2})", value)
        ):
            raise ValueError
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError
        host = parsed.hostname.lower()
        try:
            address = ipaddress.ip_address(host)
            if "%" in host:
                raise ValueError
            host = f"[{address.compressed}]" if address.version == 6 else str(address)
        except ValueError:
            if len(host) > 253 or any(
                not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                for label in host.removesuffix(".").split(".")
            ):
                raise ValueError from None
        port = parsed.port
        if port is not None and not 1 <= port <= 65535:
            raise ValueError
        if port is not None and port != (80 if parsed.scheme == "http" else 443):
            host += f":{port}"
        return urlunsplit((parsed.scheme, host, parsed.path or "/", parsed.query, ""))
    except (ValueError, UnicodeError):
        raise ToolInputError(
            "url must be a valid HTTP(S) URL without credentials or fragment"
        ) from None


def origin(url: str) -> str:
    parsed = urlsplit(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def scope_regex(url: str) -> str:
    parsed = urlsplit(url)
    authority = re.escape(parsed.netloc)
    if parsed.port is None:
        # Accept explicit default ports without widening scheme/host/port scope.
        authority = re.escape(parsed.netloc) + f"(?::{80 if parsed.scheme == 'http' else 443})?"
    return f"(?i)^{parsed.scheme}://{authority}(?:/|\\?|$)"


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError


def katana_observation(
    result: ProcessResult,
    *,
    seed: str,
    max_depth: int,
    max_pages: int,
    rate_limit: int,
    timeout_seconds: int,
) -> tuple[dict, bytes]:
    """Never retain raw responses, headers, error strings or unvisited URLs."""
    response = result.observation()
    response.update(
        scanner="katana",
        stdout="",
        stderr="",
        diagnosticsRedacted=True,
        artifacts={},
        targetsArtifact=None,
        targetsDigest=None,
        source={"seed": seed, "origin": origin(seed)},
        limits={
            "maxDepth": max_depth,
            "maxPages": max_pages,
            "rateLimit": rate_limit,
            "timeoutSeconds": timeout_seconds,
            "crawlDurationSeconds": max(1, timeout_seconds - 2),
            "requestTimeoutSeconds": min(10, max(1, timeout_seconds // 3)),
            "maxTargets": MAX_TARGETS,
            "maxTargetBytes": MAX_TARGET_BYTES,
            "maxResponseBytes": MAX_RESPONSE_BYTES,
        },
        # Katana does not certify queue exhaustion or report all skipped pages.
        discoveryComplete=False,
    )
    counts = dict.fromkeys(
        (
            "observations",
            "responses",
            "requestErrors",
            "depthLimited",
            "outOfScope",
            "unsupportedMethods",
            "unvisited",
            "invalidResultLines",
            "duplicates",
        ),
        0,
    )
    candidates: dict[str, dict] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        counts["observations"] += 1
        try:
            value = json.loads(line, object_pairs_hook=_object, parse_constant=_reject_constant)
            if not isinstance(value, dict) or not isinstance(value.get("request"), dict):
                raise ValueError
            request = value["request"]
            endpoint = canonical_url(request.get("endpoint"))
            if origin(endpoint) != origin(seed):
                counts["outOfScope"] += 1
                continue
            if request.get("method") != "GET":
                counts["unsupportedMethods"] += 1
                continue
            error = value.get("error")
            if error == "max depth reached":
                counts["depthLimited"] += 1
                continue
            if error:
                counts["requestErrors"] += 1
                continue
            received = value.get("response")
            if received is None:
                counts["unvisited"] += 1
                continue
            if not isinstance(received, dict):
                raise ValueError
            status = received.get("status_code")
            if type(status) is not int or not 100 <= status <= 599:
                raise ValueError
            parent = canonical_url(request.get("source") or seed)
            if origin(parent) != origin(seed):
                raise ValueError
            counts["responses"] += 1
            item = {"url": endpoint, "method": "GET", "statusCode": status, "source": parent}
            if endpoint in candidates:
                counts["duplicates"] += 1
                # Reordered observations produce the same retained provenance.
                item = min(
                    (item, candidates[endpoint]), key=lambda x: (x["source"], x["statusCode"])
                )
            candidates[endpoint] = item
        except (ValueError, TypeError, RecursionError, ToolInputError):
            counts["invalidResultLines"] += 1

    selected, size = [], 0
    for _url, item in sorted(candidates.items()):
        item_size = len(json.dumps(item, ensure_ascii=False).encode())
        if len(selected) >= MAX_TARGETS or size + item_size > MAX_TARGET_BYTES:
            break
        selected.append(item)
        size += item_size
    data = "".join(item["url"] + "\n" for item in selected).encode()
    truncated = len(selected) < len(candidates)
    reasons = [
        "bounded_crawl",
        "exhaustion_unverified",
        "response_size_limit",
        "default_extension_filter",
    ]
    for condition, reason in (
        (counts["depthLimited"], "depth_limit"),
        (counts["responses"] + counts["requestErrors"] >= max_pages, "page_limit"),
        (counts["requestErrors"], "request_errors"),
        (counts["outOfScope"], "out_of_scope_observations"),
        (counts["unsupportedMethods"], "unsupported_methods"),
        (counts["unvisited"], "unvisited_observations"),
        (counts["invalidResultLines"], "invalid_output"),
        (truncated, "target_export_limit"),
        (result.error_code, "process_incomplete"),
    ):
        if condition:
            reasons.append(reason)
    response.update(
        results=selected,
        resultsTruncated=truncated,
        invalidResultLines=counts["invalidResultLines"],
        requestErrors=counts["requestErrors"],
        coverage={
            **counts,
            "uniqueTargets": len(candidates),
            "exportedTargets": len(selected),
            "omittedTargets": len(candidates) - len(selected),
            "limitations": reasons,
        },
    )
    if response["errorCode"] is None:
        if counts["invalidResultLines"]:
            response.update(status="failed", errorCode="invalid_scanner_output")
        elif counts["requestErrors"]:
            response.update(status="failed", errorCode="scan_request_failed")
        elif not selected:
            response.update(status="failed", errorCode="no_discovered_targets")
    if data:
        response["targetsDigest"] = "sha256:" + hashlib.sha256(data).hexdigest()
    return response, data
