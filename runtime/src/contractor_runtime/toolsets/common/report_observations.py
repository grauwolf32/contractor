"""Project exact normalized scanner reports into source-neutral HTTP metadata."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import Field

from contractor_runtime import threads
from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactClient,
    ArtifactResponseLimitError,
)
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.contracts.base import (
    DIGEST_PATTERN,
    MAX_EXACT_INTEGER,
    NAME_PATTERN,
    StrictWireModel,
)
from contractor_runtime.strict_json import strict_json_loads
from contractor_runtime.toolsets.common.observation_metadata import normalize_http_observation

REPORT_FORMATS = {
    "scan_ffuf": "ffuf-report@1",
    "scan_katana": "katana-report@1",
    "scan_nuclei": "nuclei-report@1",
}
MAX_REPORT_BYTES = 512 << 10
MAX_RESULT_BYTES = 128 << 10
MAX_RESULTS = 100
MAX_COUNT = MAX_EXACT_INTEGER
Count = Annotated[int, Field(ge=0, le=MAX_COUNT)]
Nonnegative = Annotated[int, Field(ge=0, le=2**63 - 1)]
Digest = Annotated[str, Field(pattern=DIGEST_PATTERN.pattern)]


class _Envelope(StrictWireModel):
    schema_version: int
    tool: str
    input_digest: Digest
    input_artifacts: dict[str, ArtifactRef]
    observation: dict[str, Any]


class _Process(StrictWireModel):
    status: Literal["completed", "failed"]
    exit_code: Annotated[int, Field(ge=-(2**63), le=2**63 - 1)] | None
    error_code: str | None
    stdout: Literal[""]
    stderr: Literal[""]
    stdout_truncated: bool
    stderr_truncated: bool
    output_limit_exceeded: bool
    duration_ms: Nonnegative
    scanner: str
    diagnostics_redacted: bool
    results_truncated: bool
    invalid_result_lines: Count


class _Payload(StrictWireModel):
    fuzz: str = Field(alias="FUZZ")


class _FFUFMatch(StrictWireModel):
    input: _Payload
    position: int
    status: int
    length: Nonnegative
    words: Nonnegative
    lines: Nonnegative
    url: str
    duration_ns: Nonnegative
    content_type: str
    redirect_location: str


class _FFUF(_Process):
    results: list[_FFUFMatch]
    wordlist_artifact: ArtifactRef
    wordlist_entries: Annotated[int, Field(ge=1, le=10000)]
    payloads_attempted: Count | None
    request_errors: Count | None
    scan_complete: bool


class _Source(StrictWireModel):
    seed: str
    origin: str


class _Limits(StrictWireModel):
    max_depth: Annotated[int, Field(ge=1, le=5)]
    max_pages: Annotated[int, Field(ge=1, le=1000)]
    rate_limit: Annotated[int, Field(ge=1, le=1000)]
    timeout_seconds: Annotated[int, Field(ge=1, le=3600)]
    crawl_duration_seconds: int
    request_timeout_seconds: int
    max_targets: int
    max_target_bytes: int
    max_response_bytes: int


class _Coverage(StrictWireModel):
    observations: Count
    responses: Count
    request_errors: Count
    depth_limited: Count
    out_of_scope: Count
    unsupported_methods: Count
    unvisited: Count
    invalid_result_lines: Count
    duplicates: Count
    unique_targets: Count
    exported_targets: Count
    omitted_targets: Count
    limitations: list[str]


class _KatanaMatch(StrictWireModel):
    url: str
    method: Literal["GET"]
    status_code: int
    source: str


class _Katana(_Process):
    results: list[_KatanaMatch]
    artifacts: dict[str, ArtifactRef]
    targets_artifact: ArtifactRef | None
    targets_digest: Digest | None
    artifact_error_code: str | None = None
    source: _Source
    limits: _Limits
    discovery_complete: bool
    request_errors: Count
    coverage: _Coverage


async def prepare_report(client: ArtifactClient, format_: str, artifact: ArtifactRef) -> dict:
    """Read once with ordinary exact Artifact authority; join bounded parsing."""
    if format_ not in REPORT_FORMATS.values():
        raise ValueError("unsupported normalized report format")
    try:
        value = await client.read_artifact(artifact.require_exact(), max_bytes=MAX_REPORT_BYTES)
    except ArtifactResponseLimitError:
        raise ArtifactAPIError(400, "invalid_request", False) from None
    if value.media_type != "application/json":
        raise ArtifactAPIError(400, "invalid_request", False)
    at = value.revision_created_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    try:
        batch = await threads.to_thread_until_done(
            project_report, value.data, format_, at, name="sitemap-report-project"
        )
    except (ValueError, TypeError, KeyError, RecursionError, UnicodeError):
        raise ArtifactAPIError(400, "invalid_request", False) from None
    batch["artifact"] = value.artifact.model_dump(by_alias=True)
    return batch


def _bounded(value: object, depth: int = 0) -> int:
    if depth > 16:
        raise ValueError("report nesting exceeds bound")
    if isinstance(value, str):
        value.encode("utf-8")
    if isinstance(value, dict):
        for name in value:
            name.encode("utf-8")
    children = (
        value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
    )
    nodes = 1 + (len(value) if isinstance(value, dict) else 0)
    for child in children:
        nodes += _bounded(child, depth + 1)
        if nodes > 100000:
            raise ValueError("report nodes exceed bound")
    return nodes


def _row(url: str, status: int, at: str) -> dict:
    parts = urlsplit(url)
    if parts.username is not None or parts.password is not None or parts.fragment:
        raise ValueError("report URL is not normalized")
    row = normalize_http_observation(
        {"url": url, "method": "GET", "status": status, "outcome": "response", "at": at}
    )
    if row is None:
        raise ValueError("report observation is invalid")
    return row


def _origin(url: str, at: str) -> str:
    parts = urlsplit(_row(url, 200, at)["url"])
    host = parts.hostname
    if host is None:
        raise ValueError("report URL has no host")
    host = f"[{host}]" if ":" in host else host
    port = parts.port
    suffix = (
        f":{port}" if port is not None and port != (443 if parts.scheme == "https" else 80) else ""
    )
    return f"{parts.scheme}://{host}{suffix}"


def project_report(data: bytes, format_: str, at: str) -> dict:
    """Strictly consume the existing versioned envelope, never scanner stdout."""
    if len(data) > MAX_REPORT_BYTES:
        raise ValueError("report exceeds bound")
    raw = strict_json_loads(data.decode("utf-8"))
    _bounded(raw)
    envelope = _Envelope.model_validate(raw)
    kind = {value: tool.removeprefix("scan_") for tool, value in REPORT_FORMATS.items()}[format_]
    if envelope.schema_version != 1 or envelope.tool != "scan_" + kind:
        raise ValueError("report format differs from selection")
    for name, ref in envelope.input_artifacts.items():
        if NAME_PATTERN.fullmatch(name) is None:
            raise ValueError("report input name is invalid")
        ref.require_exact()
    if kind == "nuclei":
        return _project_nuclei_surface(envelope.observation, at)
    report = (_FFUF if kind == "ffuf" else _Katana).model_validate(envelope.observation)
    rows = envelope.observation["results"]
    if (
        len(rows) > MAX_RESULTS
        or len(
            json.dumps(rows, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        )
        > MAX_RESULT_BYTES
    ):
        raise ValueError("report results exceed bound")
    if (
        report.scanner != kind
        or not report.diagnostics_redacted
        or not (
            (report.status == "completed" and report.error_code is None and report.exit_code == 0)
            or (report.status == "failed" and report.error_code)
        )
    ):
        raise ValueError("report process metadata is inconsistent")
    gaps: dict[str, int] = {}

    def gap(reason: str, count: int) -> None:
        if count:
            gaps[reason] = min(MAX_COUNT, gaps.get(reason, 0) + count)

    gap("projection", report.invalid_result_lines)
    gap("report-truncated", int(report.results_truncated or report.output_limit_exceeded))
    if isinstance(report, _FFUF):
        report.wordlist_artifact.require_exact()
        if any(
            count is not None and count > report.wordlist_entries
            for count in (report.payloads_attempted, report.request_errors)
        ) or (
            report.scan_complete
            and (
                report.status != "completed"
                or report.exit_code != 0
                or report.invalid_result_lines != 0
                or report.payloads_attempted != report.wordlist_entries
                or report.request_errors != 0
            )
        ):
            raise ValueError("report completion counters are inconsistent")
        observations = []
        for row in report.results:
            if (
                not 1 <= row.position <= report.wordlist_entries
                or len(row.input.fuzz.encode()) > 4096
            ):
                raise ValueError("report match is outside its wordlist")
            observations.append(_row(row.url, row.status, at))
        gap("incomplete", int(not report.scan_complete))
        gap("request-errors", report.request_errors or 0)
    else:
        _validate_katana(report, at)
        seen = set()
        observations = []
        for row in report.results:
            if (
                _origin(row.url, at) != report.source.origin
                or _origin(row.source, at) != report.source.origin
                or row.url in seen
            ):
                raise ValueError("report target is duplicate or outside its origin")
            seen.add(row.url)
            observations.append(_row(row.url, row.status_code, at))
        digest = (
            "sha256:"
            + hashlib.sha256("".join(row.url + "\n" for row in report.results).encode()).hexdigest()
        )
        if report.targets_digest != (digest if observations else None):
            raise ValueError("report targets digest differs from retained results")
        gap(
            "incomplete",
            int(report.status == "failed" and report.error_code != "scan_artifact_failed"),
        )
        gap("request-errors", report.request_errors)
        gap("members", report.coverage.omitted_targets)
        gap(
            "projection",
            sum(
                getattr(report.coverage, name)
                for name in ("depth_limited", "out_of_scope", "unsupported_methods", "unvisited")
            ),
        )
    return {
        "kind": kind,
        "observations": observations,
        **(
            {"gaps": [{"reason": reason, "count": count} for reason, count in gaps.items()]}
            if gaps
            else {}
        ),
    }


def _validate_katana(report: _Katana, at: str) -> None:
    limits, coverage = report.limits, report.coverage
    if (
        _origin(report.source.seed, at) != report.source.origin
        or report.discovery_complete
        or limits.crawl_duration_seconds != max(1, limits.timeout_seconds - 2)
        or limits.request_timeout_seconds != min(10, max(1, limits.timeout_seconds // 3))
        or limits.max_targets != MAX_RESULTS
        or limits.max_target_bytes != MAX_RESULT_BYTES
        or limits.max_response_bytes != 1 << 20
        or len(report.results) != coverage.exported_targets
        or coverage.request_errors != report.request_errors
        or coverage.invalid_result_lines != report.invalid_result_lines
        or coverage.unique_targets != coverage.exported_targets + coverage.omitted_targets
        or coverage.responses != coverage.unique_targets + coverage.duplicates
        or report.results_truncated != (coverage.omitted_targets > 0)
        or coverage.observations
        != sum(
            getattr(coverage, name)
            for name in (
                "responses",
                "request_errors",
                "depth_limited",
                "out_of_scope",
                "unsupported_methods",
                "unvisited",
                "invalid_result_lines",
            )
        )
    ):
        raise ValueError("report crawl bounds or counters are inconsistent")
    required = {
        "bounded_crawl",
        "exhaustion_unverified",
        "response_size_limit",
        "default_extension_filter",
    }
    conditions = {
        "depth_limit": coverage.depth_limited > 0,
        "page_limit": coverage.responses + coverage.request_errors >= limits.max_pages,
        "request_errors": coverage.request_errors > 0,
        "out_of_scope_observations": coverage.out_of_scope > 0,
        "unsupported_methods": coverage.unsupported_methods > 0,
        "unvisited_observations": coverage.unvisited > 0,
        "invalid_output": coverage.invalid_result_lines > 0,
        "target_export_limit": coverage.omitted_targets > 0,
    }
    limitations = set(coverage.limitations)
    if (
        len(limitations) != len(coverage.limitations)
        or not required <= limitations
        or limitations - required - conditions.keys() - {"process_incomplete"}
        or any((reason in limitations) != present for reason, present in conditions.items())
    ):
        raise ValueError("report crawl limitations are inconsistent")
    if report.targets_artifact is None:
        if report.artifacts or (
            coverage.exported_targets and report.artifact_error_code != "scan_artifact_failed"
        ):
            raise ValueError("report target publication is inconsistent")
    elif (
        report.targets_artifact.require_exact() != report.artifacts.get("targets")
        or len(report.artifacts) != 1
        or report.artifact_error_code is not None
    ):
        raise ValueError("report target ref differs from published artifacts")


def _project_nuclei_surface(observation: dict, at: str) -> dict:
    """Consume already-normalized adapter facts; never parse native Nuclei fields."""
    normalized = observation.get("findingSummary")
    if not isinstance(normalized, dict):
        raise ValueError("normalized Nuclei surface is unavailable")
    surface = normalized.get("surface")
    if not isinstance(surface, list) or len(surface) > MAX_RESULTS:
        raise ValueError("invalid bounded surface metadata")
    rows = []
    omitted = normalized.get("surfaceOmissions", 0) + normalized.get("skipped", {}).get(
        "invalid_row", 0
    )
    if (
        not isinstance(omitted, int)
        or isinstance(omitted, bool)
        or not 0 <= omitted <= 2 * MAX_RESULTS
    ):
        raise ValueError("invalid bounded surface omissions")
    for fact in surface:
        if not isinstance(fact, dict) or set(fact) != {"url", "method", "status", "note"}:
            raise ValueError("invalid surface fact")
        method, status, note = fact["method"], fact["status"], fact["note"]
        if not method:
            omitted += 1
            continue
        row = normalize_http_observation(
            {
                "url": fact["url"],
                "method": method,
                "status": status or 0,
                "outcome": "response" if status is not None else "attempt",
                "at": at,
            }
        )
        if row is None or not isinstance(note, str) or len(note.encode()) > 4096:
            raise ValueError("invalid normalized surface observation")
        if note:
            row["note"] = note
        rows.append(row)
    gaps = []
    if omitted:
        gaps.append({"reason": "projection", "count": omitted})
    if normalized.get("complete") is not True:
        gaps.append({"reason": "incomplete", "count": 1})
    return {"kind": "nuclei", "observations": rows, **({"gaps": gaps} if gaps else {})}
