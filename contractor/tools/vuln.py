from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar
from xml.sax.saxutils import escape as xml_escape

import yaml
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

from contractor.tools.result import aguard, err
from contractor.utils import utc_now_iso

# Passthrough type for the code-fence helpers: a non-str payload is returned
# unchanged, a str may be wrapped — so the caller's narrower type is preserved.
_T = TypeVar("_T")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Severity = Literal["info", "low", "medium", "high", "critical"]
Confidence = Literal["low", "medium", "high"]
PlaceType = Literal["file", "url"]
OutputFormat = Literal["json", "markdown", "yaml", "xml"]
Verdict = Literal[
    "exploitable",
    "exploitable_unverified",
    "not_exploitable",
    "inconclusive",
]
AttackerControl = Literal["full", "partial", "none"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
#
# Both records are frozen pydantic models: the ``Literal`` fields enforce the
# constrained vocabularies (severity / confidence / verdict / attacker control)
# at construction — no hand-rolled ``__post_init__`` validators — and freezing
# makes the stored records genuinely immutable, as their docstrings promise.


class VulnerabilityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    place_type: PlaceType
    place: str
    title: str
    summary: str
    severity: Severity
    confidence: Confidence
    details: str
    ordinal: int = 0
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


# ---------------------------------------------------------------------------
# Shared serialisation helpers
# ---------------------------------------------------------------------------


class _NamedRecord(Protocol):
    """Structural type for the two record models: a ``name`` + ``model_dump``."""

    name: str

    def model_dump(self) -> dict[str, Any]: ...


_R = TypeVar("_R", bound=_NamedRecord)


def _preview_dict(
    model: BaseModel, fields: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """Serialise a model to a dict, optionally projected to preview ``fields``."""
    data = model.model_dump()
    if fields is None:
        return data
    return {k: data[k] for k in fields}


def _single_record_yaml(key: str, payload: dict[str, Any]) -> str:
    """YAML-dump one record under a single top-level ``key``."""
    return yaml.safe_dump({key: payload}, sort_keys=False, allow_unicode=True)


def _dump_records_yaml(records: Iterable[_NamedRecord]) -> str:
    """YAML-dump a ``{record.name: model_dump}`` map preserving the given order."""
    payload = {r.name: r.model_dump() for r in records}
    return yaml.safe_dump(
        payload, sort_keys=False, allow_unicode=True, default_flow_style=False
    )


async def _load_artifact_records(
    ctx: ToolContext | CallbackContext,
    *,
    artifact_key: str,
    normalize: Callable[[str, dict[str, Any], int], _R],
) -> dict[str, _R]:
    """Load a ``{name: record}`` map from a YAML artifact, skipping non-dicts.

    ``normalize`` is called with ``(name, item, index)`` where ``index`` is the
    1-based position (a fallback ordinal); records are keyed by their own
    ``.name``.
    """
    artifact = await ctx.load_artifact(filename=artifact_key)
    if artifact is None:
        return {}
    raw: dict[str, Any] = yaml.safe_load(artifact.text or "") or {}
    records: dict[str, _R] = {}
    for index, (name, item) in enumerate(raw.items(), start=1):
        if not isinstance(item, dict):
            continue
        record = normalize(name, item, index)
        records[record.name] = record
    return records


async def _save_artifact_text(
    ctx: ToolContext | CallbackContext, artifact_key: str, text: str
) -> None:
    """Persist ``text`` to the artifact store under ``artifact_key``."""
    await ctx.save_artifact(
        filename=artifact_key, artifact=types.Part.from_text(text=text)
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

# Fields included in preview mode (everything except `details`)
_PREVIEW_FIELDS: tuple[str, ...] = (
    "name",
    "title",
    "place_type",
    "place",
    "severity",
    "confidence",
    "summary",
    "ordinal",
    "created_at",
    "updated_at",
)


def _report_to_dict(
    report: VulnerabilityReport, *, preview: bool = False
) -> dict[str, Any]:
    """Serialise a report to a plain dictionary."""
    return _preview_dict(report, _PREVIEW_FIELDS if preview else None)


def _report_to_markdown(report: VulnerabilityReport, *, preview: bool = False) -> str:
    """Render a report as a Markdown block."""
    lines = [
        f"### {report.title}",
        f"**Name**: {report.name}",
        f"**Place**: `{report.place_type}` — `{report.place}`",
        f"**Severity**: {report.severity}",
        f"**Confidence**: {report.confidence}",
        f"**Summary**: {report.summary}",
        f"**Created At**: {report.created_at or '-'}",
        f"**Updated At**: {report.updated_at or '-'}",
    ]
    if not preview:
        lines.append(f"**Details**:\n{report.details}")
    return "\n".join(lines) + "\n"


def _report_to_yaml(report: VulnerabilityReport, *, preview: bool = False) -> str:
    """Render a report as a YAML document."""
    return _single_record_yaml(
        f"vulnerability_{report.name}", _report_to_dict(report, preview=preview)
    )


def _xml_tag(tag: str, value: str, pad: str) -> str:
    """Wrap an escaped value in an XML tag with indentation."""
    return f"{pad}<{tag}>{xml_escape(value)}</{tag}>"


def _report_to_xml(
    report: VulnerabilityReport,
    *,
    preview: bool = False,
    indent: int = 0,
) -> str:
    """Render a report as an XML fragment."""
    pad = " " * (indent * 4)
    pad2 = " " * ((indent + 1) * 4)

    children = [
        _xml_tag("title", report.title, pad2),
        f'{pad2}<place type="{xml_escape(report.place_type)}">'
        f"{xml_escape(report.place)}</place>",
        _xml_tag("severity", report.severity, pad2),
        _xml_tag("confidence", report.confidence, pad2),
        _xml_tag("summary", report.summary, pad2),
        _xml_tag("created_at", report.created_at, pad2),
        _xml_tag("updated_at", report.updated_at, pad2),
    ]
    if not preview:
        children.append(_xml_tag("details", report.details, pad2))

    inner = "\n".join(children)
    return (
        f'{pad}<vulnerability name="{xml_escape(report.name)}">\n'
        f"{inner}\n"
        f"{pad}</vulnerability>"
    )


@dataclass
class VulnerabilityReportFormat:
    """Formats one or many :class:`VulnerabilityReport` instances."""

    _format: OutputFormat = "json"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_code_fence(
        output: _T,
        fmt: str,
        *,
        type_hint: bool,
    ) -> _T | str:
        """Optionally wrap string output in a Markdown code fence."""
        if not type_hint or not isinstance(output, str):
            return output
        return f"```{fmt}\n{output}\n```"

    # ------------------------------------------------------------------
    # Single-report formatting
    # ------------------------------------------------------------------

    def format_report(
        self,
        report: VulnerabilityReport,
        *,
        preview: bool = False,
        type_hint: bool = False,
    ) -> str | dict[str, Any]:
        """Format a single report, optionally omitting the details field."""
        output: str | dict[str, Any]

        if self._format == "json":
            output = _report_to_dict(report, preview=preview)
        elif self._format == "markdown":
            output = _report_to_markdown(report, preview=preview)
        elif self._format == "yaml":
            output = _report_to_yaml(report, preview=preview)
        elif self._format == "xml":
            output = _report_to_xml(report, preview=preview)
        else:
            output = _report_to_dict(report, preview=preview)

        return self._wrap_code_fence(output, self._format, type_hint=type_hint)

    # ------------------------------------------------------------------
    # Multi-report formatting
    # ------------------------------------------------------------------

    def format_reports(
        self,
        reports: list[VulnerabilityReport],
        *,
        preview: bool = False,
        type_hint: bool = False,
    ) -> str | list[dict[str, Any]]:
        """Format a collection of reports."""
        if self._format == "json":
            # JSON returns a native list — no string wrapping needed.
            return [_report_to_dict(r, preview=preview) for r in reports]

        if self._format in {"markdown", "yaml"}:
            # In these branches format_report yields a str; str() is identity
            # but lets mypy see a str generator without an arg-type ignore.
            output = "\n".join(
                str(self.format_report(r, preview=preview, type_hint=False))
                for r in reports
            )
            return self._wrap_code_fence(output, self._format, type_hint=type_hint)

        if self._format == "xml":
            inner = "\n".join(
                _report_to_xml(r, preview=preview, indent=1) for r in reports
            )
            output = f"<vulnerabilities>\n{inner}\n</vulnerabilities>"
            return self._wrap_code_fence(output, self._format, type_hint=type_hint)

        # Fallback — should never be reached with strict typing.
        return [_report_to_dict(r, preview=preview) for r in reports]


# ---------------------------------------------------------------------------
# Storage / business logic
# ---------------------------------------------------------------------------


@dataclass
class VulnerabilityReportTools:
    name: str
    fmt: VulnerabilityReportFormat = field(
        default_factory=lambda: VulnerabilityReportFormat("json")
    )
    reports: dict[str, VulnerabilityReport] = field(default_factory=dict)
    # asyncio.Lock must not be shared across event loops; create lazily.
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def artifact_key(self) -> str:
        """Storage key used with the artifact API."""
        return f"user:vulnerability-reports/{self.name}"

    def _next_ordinal(self) -> int:
        if not self.reports:
            return 1
        return max(r.ordinal for r in self.reports.values()) + 1

    @staticmethod
    def _normalize_report(
        name: str,
        item: dict[str, Any],
        fallback_ordinal: int,
    ) -> VulnerabilityReport:
        """Build a :class:`VulnerabilityReport` from a raw YAML dictionary."""
        return VulnerabilityReport(
            name=item.get("name", name),
            place_type=item.get("place_type", "file"),
            place=item.get("place", ""),
            title=item.get("title", name),
            summary=item.get("summary", ""),
            severity=item.get("severity", "medium"),
            confidence=item.get("confidence", "medium"),
            details=item.get("details", ""),
            ordinal=item.get("ordinal", fallback_ordinal),
            created_at=item.get("created_at", utc_now_iso()),
            updated_at=item.get("updated_at", utc_now_iso()),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def load(self, ctx: ToolContext | CallbackContext) -> None:
        """Load reports from the artifact store into memory."""
        async with self._lock:
            self.reports = await _load_artifact_records(
                ctx,
                artifact_key=self.artifact_key,
                normalize=self._normalize_report,
            )

    def dump(self) -> str:
        """Serialise all reports to a YAML string, ordered by (ordinal, name)."""
        # Bind to a typed local first so the sort key sees VulnerabilityReport
        # (with .ordinal), not the _NamedRecord protocol expected downstream.
        ordered: list[VulnerabilityReport] = sorted(
            self.reports.values(), key=lambda r: (r.ordinal, r.name)
        )
        return _dump_records_yaml(ordered)

    async def save(self, ctx: ToolContext | CallbackContext) -> None:
        """Persist in-memory reports to the artifact store."""
        async with self._lock:
            await _save_artifact_text(ctx, self.artifact_key, self.dump())

    # ------------------------------------------------------------------
    # Public CRUD API
    # ------------------------------------------------------------------

    async def list_reports(
        self, ctx: ToolContext | CallbackContext
    ) -> list[VulnerabilityReport]:
        await self.load(ctx)
        return sorted(self.reports.values(), key=lambda r: (r.ordinal, r.name))

    async def get_report(
        self,
        name: str,
        ctx: ToolContext | CallbackContext,
    ) -> VulnerabilityReport | None:
        await self.load(ctx)
        return self.reports.get(name)

    async def write_report(
        self,
        name: str,
        place_type: PlaceType,
        place: str,
        title: str,
        summary: str,
        severity: Severity,
        confidence: Confidence,
        details: str,
        ctx: ToolContext | CallbackContext,
    ) -> VulnerabilityReport:
        """Create or fully overwrite a vulnerability report."""
        await self.load(ctx)
        now = utc_now_iso()
        existing = self.reports.get(name)

        report = VulnerabilityReport(
            name=name,
            place_type=place_type,
            place=place,
            title=title,
            summary=summary,
            severity=severity,
            confidence=confidence,
            details=details,
            ordinal=existing.ordinal if existing else self._next_ordinal(),
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )

        self.reports[name] = report
        await self.save(ctx)
        return report

    async def delete_report(
        self,
        name: str,
        ctx: ToolContext | CallbackContext,
    ) -> bool:
        """Remove a report by name. Returns *True* if it existed."""
        await self.load(ctx)
        existed = self.reports.pop(name, None) is not None
        if existed:
            await self.save(ctx)
        return existed


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


def vulnerability_report_tools(
    name: str,
    fmt: VulnerabilityReportFormat | None = None,
) -> list[Any]:
    """
    Return a list of async tool functions bound to a named report store.

    Args:
        name: Logical name of the vulnerability report collection.
        fmt:  Output format descriptor; defaults to JSON.
    """
    vr = VulnerabilityReportTools(
        name=name,
        fmt=fmt if fmt is not None else VulnerabilityReportFormat("json"),
    )

    async def report_vulnerability(
        name: str,
        place_type: PlaceType,
        place: str,
        title: str,
        summary: str,
        severity: Severity,
        confidence: Confidence,
        details: str,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Write or update a vulnerability report.

        Args:
            name:        Stable unique key, e.g. "sqli-login" or "xss-profile".
            place_type:  Location kind — "file" or "url".
            place:       File path or URL where the issue was found.
            title:       Human-readable vulnerability title.
            summary:     One-sentence description.
            severity:    "info", "low", "medium", "high", or "critical".
            confidence:  "low", "medium", or "high".
            details:     Technical details, reproduction steps, impact, and remediation hints.
        """

        async def _impl() -> Any:
            report = await vr.write_report(
                name=name,
                place_type=place_type,
                place=place,
                title=title,
                summary=summary,
                severity=severity,
                confidence=confidence,
                details=details,
                ctx=tool_context,
            )
            return vr.fmt.format_report(report)

        return await aguard(_impl)

    async def get_vulnerability(
        name: str,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Read a vulnerability report by its unique name.

        Args:
            name: Exact report key.
        """

        async def _impl() -> Any:
            report = await vr.get_report(name, tool_context)
            if report is None:
                return err(f"Vulnerability report {name!r} not found.")
            return vr.fmt.format_report(report)

        return await aguard(_impl)

    async def list_vulnerabilities(
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        List vulnerability reports in insertion order.

        Returns:
            A preview list with title, location, severity, confidence, and summary.
        """

        async def _impl() -> Any:
            reports = await vr.list_reports(tool_context)
            return vr.fmt.format_reports(reports, preview=True)

        return await aguard(_impl)

    return [
        report_vulnerability,
        get_vulnerability,
        list_vulnerabilities,
    ]


# ---------------------------------------------------------------------------
# Verification — second-stage attacker-role-play assessment of a finding
# ---------------------------------------------------------------------------


class VerifiedFinding(BaseModel):
    """Outcome of a verifier-stage assessment.

    Verifier consumes a :class:`VulnerabilityReport` and produces this record,
    which is stored separately so the original finding remains immutable and
    multiple verification attempts can coexist. The model is frozen, so that
    immutability is enforced rather than merely conventional.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    source_namespace: str
    verdict: Verdict
    summary: str
    attacker_control_at_sink: AttackerControl
    sink_reached: bool
    entry_point: str
    data_flow: list[str] = Field(default_factory=list)
    path_broken_at: str | None = None
    impact: str = ""
    notes: str = ""
    evidence_request_ids: list[str] = Field(default_factory=list)
    verified_at: str = Field(default_factory=utc_now_iso)


_VERIFICATION_PREVIEW_FIELDS: tuple[str, ...] = (
    "name",
    "source_namespace",
    "verdict",
    "summary",
    "attacker_control_at_sink",
    "sink_reached",
    "verified_at",
)


def _verification_to_dict(
    finding: VerifiedFinding, *, preview: bool = False
) -> dict[str, Any]:
    return _preview_dict(finding, _VERIFICATION_PREVIEW_FIELDS if preview else None)


def _verification_to_yaml(
    finding: VerifiedFinding, *, preview: bool = False
) -> str:
    return _single_record_yaml(
        f"verification_{finding.name}",
        _verification_to_dict(finding, preview=preview),
    )


def _verification_to_markdown(
    finding: VerifiedFinding, *, preview: bool = False
) -> str:
    lines = [
        f"### {finding.name} — {finding.verdict}",
        f"**Source namespace**: `{finding.source_namespace}`",
        f"**Entry point**: `{finding.entry_point}`",
        f"**Sink reached**: {finding.sink_reached}",
        f"**Attacker control at sink**: {finding.attacker_control_at_sink}",
        f"**Summary**: {finding.summary}",
        f"**Verified at**: {finding.verified_at or '-'}",
    ]
    if not preview:
        if finding.data_flow:
            flow_md = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(finding.data_flow))
            lines.append(f"**Data flow**:\n{flow_md}")
        if finding.path_broken_at:
            lines.append(f"**Path broken at**: {finding.path_broken_at}")
        if finding.impact:
            lines.append(f"**Impact**: {finding.impact}")
        if finding.evidence_request_ids:
            lines.append(
                "**Evidence request ids**: "
                + ", ".join(f"`{r}`" for r in finding.evidence_request_ids)
            )
        if finding.notes:
            lines.append(f"**Notes**:\n{finding.notes}")
    return "\n".join(lines) + "\n"


def _verification_to_xml(
    finding: VerifiedFinding,
    *,
    preview: bool = False,
    indent: int = 0,
) -> str:
    pad = " " * (indent * 4)
    pad2 = " " * ((indent + 1) * 4)
    children = [
        _xml_tag("source_namespace", finding.source_namespace, pad2),
        _xml_tag("verdict", finding.verdict, pad2),
        _xml_tag("summary", finding.summary, pad2),
        _xml_tag("entry_point", finding.entry_point, pad2),
        _xml_tag("sink_reached", str(finding.sink_reached).lower(), pad2),
        _xml_tag("attacker_control_at_sink", finding.attacker_control_at_sink, pad2),
        _xml_tag("verified_at", finding.verified_at, pad2),
    ]
    if not preview:
        flow_inner = "".join(
            f"\n{pad2}    <step>{xml_escape(step)}</step>" for step in finding.data_flow
        )
        children.append(f"{pad2}<data_flow>{flow_inner}\n{pad2}</data_flow>")
        if finding.path_broken_at:
            children.append(_xml_tag("path_broken_at", finding.path_broken_at, pad2))
        if finding.impact:
            children.append(_xml_tag("impact", finding.impact, pad2))
        if finding.evidence_request_ids:
            ids_inner = "".join(
                f"\n{pad2}    <id>{xml_escape(r)}</id>"
                for r in finding.evidence_request_ids
            )
            children.append(
                f"{pad2}<evidence_request_ids>{ids_inner}\n"
                f"{pad2}</evidence_request_ids>"
            )
        if finding.notes:
            children.append(_xml_tag("notes", finding.notes, pad2))
    inner = "\n".join(children)
    return (
        f'{pad}<verification name="{xml_escape(finding.name)}">\n'
        f"{inner}\n"
        f"{pad}</verification>"
    )


@dataclass
class VerifiedFindingFormat:
    """Formats one or many :class:`VerifiedFinding` instances."""

    _format: OutputFormat = "json"

    def format_finding(
        self,
        finding: VerifiedFinding,
        *,
        preview: bool = False,
    ) -> str | dict[str, Any]:
        if self._format == "json":
            return _verification_to_dict(finding, preview=preview)
        if self._format == "yaml":
            return _verification_to_yaml(finding, preview=preview)
        if self._format == "markdown":
            return _verification_to_markdown(finding, preview=preview)
        return _verification_to_xml(finding, preview=preview)

    def format_findings(
        self,
        findings: list[VerifiedFinding],
        *,
        preview: bool = False,
    ) -> str | list[Any]:
        if self._format == "json":
            return [_verification_to_dict(f, preview=preview) for f in findings]
        if self._format == "yaml":
            payload = {
                f"verification_{f.name}": _verification_to_dict(f, preview=preview)
                for f in findings
            }
            return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        if self._format == "markdown":
            return "\n".join(
                _verification_to_markdown(f, preview=preview) for f in findings
            )
        body = "\n".join(
            _verification_to_xml(f, preview=preview, indent=1) for f in findings
        )
        return f"<verifications>\n{body}\n</verifications>"


@dataclass
class VerifiedFindingsTools:
    """Per-namespace store for :class:`VerifiedFinding` records.

    Findings are keyed by ``name`` (which mirrors the upstream
    :class:`VulnerabilityReport.name`); writing a record for the same name
    overwrites the previous verdict.
    """

    name: str
    fmt: VerifiedFindingFormat = field(
        default_factory=lambda: VerifiedFindingFormat("json")
    )
    findings: dict[str, VerifiedFinding] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @property
    def artifact_key(self) -> str:
        return f"user:vulnerability-verifications/{self.name}"

    @staticmethod
    def _normalize(name: str, item: dict[str, Any]) -> VerifiedFinding:
        return VerifiedFinding(
            name=item.get("name", name),
            source_namespace=item.get("source_namespace", ""),
            verdict=item.get("verdict", "inconclusive"),
            summary=item.get("summary", ""),
            attacker_control_at_sink=item.get("attacker_control_at_sink", "none"),
            sink_reached=bool(item.get("sink_reached", False)),
            entry_point=item.get("entry_point", ""),
            data_flow=list(item.get("data_flow", []) or []),
            path_broken_at=item.get("path_broken_at"),
            impact=item.get("impact", ""),
            notes=item.get("notes", ""),
            evidence_request_ids=list(item.get("evidence_request_ids", []) or []),
            verified_at=item.get("verified_at", utc_now_iso()),
        )

    async def load(self, ctx: ToolContext | CallbackContext) -> None:
        async with self._lock:
            self.findings = await _load_artifact_records(
                ctx,
                artifact_key=self.artifact_key,
                # Findings have no fallback ordinal — the position is ignored.
                normalize=lambda name, item, _index: self._normalize(name, item),
            )

    def dump(self) -> str:
        return _dump_records_yaml(self.findings.values())

    async def save(self, ctx: ToolContext | CallbackContext) -> None:
        async with self._lock:
            await _save_artifact_text(ctx, self.artifact_key, self.dump())

    async def list_findings(
        self, ctx: ToolContext | CallbackContext
    ) -> list[VerifiedFinding]:
        await self.load(ctx)
        return sorted(self.findings.values(), key=lambda f: f.name)

    async def get_finding(
        self, name: str, ctx: ToolContext | CallbackContext
    ) -> VerifiedFinding | None:
        await self.load(ctx)
        return self.findings.get(name)

    async def write_finding(
        self,
        *,
        name: str,
        source_namespace: str,
        verdict: Verdict,
        summary: str,
        attacker_control_at_sink: AttackerControl,
        sink_reached: bool,
        entry_point: str,
        data_flow: list[str],
        path_broken_at: str | None,
        impact: str,
        notes: str,
        ctx: ToolContext | CallbackContext,
        evidence_request_ids: list[str] | None = None,
    ) -> VerifiedFinding:
        await self.load(ctx)
        finding = VerifiedFinding(
            name=name,
            source_namespace=source_namespace,
            verdict=verdict,
            summary=summary,
            attacker_control_at_sink=attacker_control_at_sink,
            sink_reached=sink_reached,
            entry_point=entry_point,
            data_flow=list(data_flow or []),
            path_broken_at=path_broken_at,
            impact=impact,
            notes=notes,
            evidence_request_ids=list(evidence_request_ids or []),
        )
        self.findings[name] = finding
        await self.save(ctx)
        return finding


def verification_tools(
    name: str,
    fmt: VerifiedFindingFormat | None = None,
) -> list[Any]:
    """Tools the verifier uses to persist its verdict for a finding.

    Args:
        name:  Logical namespace key — typically the same string used for the
               upstream ``vulnerability_report_tools`` so verifications and
               their source findings share scope.
        fmt:   Output format descriptor; defaults to JSON.
    """
    vt = VerifiedFindingsTools(
        name=name,
        fmt=fmt if fmt is not None else VerifiedFindingFormat("json"),
    )

    async def report_verification(
        name: str,
        source_namespace: str,
        verdict: Verdict,
        summary: str,
        attacker_control_at_sink: AttackerControl,
        sink_reached: bool,
        entry_point: str,
        data_flow: list[str],
        impact: str,
        notes: str,
        tool_context: ToolContext,
        path_broken_at: str | None = None,
        request_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Persist a verifier verdict for a single upstream finding.

        Args:
            name:                       Same as the upstream VulnerabilityReport.name.
            source_namespace:           Namespace where the upstream report lives.
            verdict:                    "exploitable", "exploitable_unverified",
                                        "not_exploitable", or "inconclusive".
            summary:                    One-sentence verdict rationale.
            attacker_control_at_sink:   "full", "partial", or "none".
            sink_reached:               True iff attacker-controlled data reaches the sink.
            entry_point:                Concrete entry point used (URL or function:line).
            data_flow:                  Ordered list of code locations through which
                                        the data travels — file:line or func name per step.
            impact:                     What an external attacker gains. Required for
                                        "exploitable" — must harm someone OTHER than the
                                        attacker themselves.
            notes:                      Attacker-narrative reasoning, alternative attempts,
                                        and refuted hypotheses.
            path_broken_at:             For "not_exploitable" / "inconclusive": which step
                                        the path breaks at (function name or guard). None
                                        when sink is reached.
            request_ids:                The request_tag values (from http_request /
                                        caido_replay) of the requests that prove this
                                        verdict. Used to collect the raw HTTP chain.
        """

        async def _impl() -> Any:
            finding = await vt.write_finding(
                name=name,
                source_namespace=source_namespace,
                verdict=verdict,
                summary=summary,
                attacker_control_at_sink=attacker_control_at_sink,
                sink_reached=sink_reached,
                entry_point=entry_point,
                data_flow=data_flow,
                path_broken_at=path_broken_at,
                impact=impact,
                notes=notes,
                evidence_request_ids=request_ids,
                ctx=tool_context,
            )
            return vt.fmt.format_finding(finding)

        return await aguard(_impl)

    async def get_verification(
        name: str,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Read a previously-stored verification by its finding name.

        Args:
            name: Finding name whose verification to retrieve.

        Returns the stored verification, or an error if none exists for that
        name.
        """

        async def _impl() -> Any:
            finding = await vt.get_finding(name, tool_context)
            if finding is None:
                return err(f"Verification for {name!r} not found.")
            return vt.fmt.format_finding(finding)

        return await aguard(_impl)

    async def list_verifications(
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        List all verifications stored in this namespace.

        Returns a preview (truncated) of each stored verification; use
        get_verification for the full detail of one.
        """

        async def _impl() -> Any:
            findings = await vt.list_findings(tool_context)
            return vt.fmt.format_findings(findings, preview=True)

        return await aguard(_impl)

    async def submit_verdict(
        name: str,
        verdict: Verdict,
        summary: str,
        entry_point: str,
        evidence: str,
        tool_context: ToolContext,
        sink_reached: bool = True,
        attacker_control: AttackerControl = "full",
        request_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Simplified verdict submission — use this instead of report_verification.

        Args:
            name:               Finding name (from get_vulnerability).
            verdict:            "exploitable", "exploitable_unverified",
                                "not_exploitable", or "inconclusive".
            summary:            One-sentence verdict rationale.
            entry_point:        The URL or function:line you probed.
            evidence:           Full evidence: HTTP requests/responses sent,
                                observed behavior, code references.
            sink_reached:       True if attacker data reaches the sink.
            attacker_control:   "full", "partial", or "none".
            request_ids:        The request_tag values (from http_request /
                                caido_replay) of the requests that prove this
                                verdict. Used to collect the raw HTTP chain
                                as an artifact — list every probe that matters.
        """

        async def _impl() -> Any:
            finding = await vt.write_finding(
                name=name,
                source_namespace=vt.name,
                verdict=verdict,
                summary=summary,
                attacker_control_at_sink=attacker_control,
                sink_reached=sink_reached,
                entry_point=entry_point,
                data_flow=[],
                path_broken_at=None,
                impact=summary
                if verdict in ("exploitable", "exploitable_unverified")
                else "",
                notes=evidence,
                evidence_request_ids=request_ids,
                ctx=tool_context,
            )
            return vt.fmt.format_finding(finding)

        return await aguard(_impl)

    return [
        submit_verdict,
        report_verification,
        get_verification,
        list_verifications,
    ]
