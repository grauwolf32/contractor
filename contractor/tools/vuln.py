from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Union
from xml.sax.saxutils import escape as xml_escape

import yaml
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.genai import types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Severity = Literal["info", "low", "medium", "high", "critical"]
Confidence = Literal["low", "medium", "high"]
PlaceType = Literal["file", "url"]
OutputFormat = Literal["json", "markdown", "yaml", "xml"]

_VALID_SEVERITIES: frozenset[str] = frozenset(
    {"info", "low", "medium", "high", "critical"}
)
_VALID_CONFIDENCES: frozenset[str] = frozenset({"low", "medium", "high"})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class VulnerabilityReport:
    name: str
    place_type: PlaceType
    place: str
    title: str
    summary: str
    severity: Severity
    confidence: Confidence
    details: str
    ordinal: int = 0
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity {self.severity!r}. "
                f"Must be one of {sorted(_VALID_SEVERITIES)}."
            )
        if self.confidence not in _VALID_CONFIDENCES:
            raise ValueError(
                f"Invalid confidence {self.confidence!r}. "
                f"Must be one of {sorted(_VALID_CONFIDENCES)}."
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
    data = asdict(report)
    if preview:
        return {k: data[k] for k in _PREVIEW_FIELDS}
    return data


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
    payload = {f"vulnerability_{report.name}": _report_to_dict(report, preview=preview)}
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


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
        output: Union[str, dict[str, Any], list[Any]],
        fmt: str,
        *,
        type_hint: bool,
    ) -> Union[str, dict[str, Any], list[Any]]:
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
    ) -> Union[str, dict[str, Any]]:
        """Format a single report, optionally omitting the details field."""
        output: Union[str, dict[str, Any]]

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
    ) -> Union[str, list[dict[str, Any]]]:
        """Format a collection of reports."""
        if self._format == "json":
            # JSON returns a native list — no string wrapping needed.
            return [_report_to_dict(r, preview=preview) for r in reports]

        if self._format in {"markdown", "yaml"}:
            output = "\n".join(
                self.format_report(r, preview=preview, type_hint=False)  # type: ignore[arg-type]
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
            artifact = await ctx.load_artifact(filename=self.artifact_key)
            if artifact is None:
                self.reports = {}
                return

            raw: dict[str, Any] = yaml.safe_load(artifact.text) or {}
            self.reports = {
                report.name: report
                for index, (name, item) in enumerate(raw.items(), start=1)
                if isinstance(item, dict)
                for report in (
                    self._normalize_report(
                        name=name, item=item, fallback_ordinal=index
                    ),
                )
            }

    def dump(self) -> str:
        """Serialise all reports to a YAML string."""
        sorted_reports = sorted(
            self.reports.values(), key=lambda r: (r.ordinal, r.name)
        )
        payload = {r.name: asdict(r) for r in sorted_reports}
        return yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )

    async def save(self, ctx: ToolContext | CallbackContext) -> None:
        """Persist in-memory reports to the artifact store."""
        async with self._lock:
            artifact = types.Part.from_text(text=self.dump())
            await ctx.save_artifact(filename=self.artifact_key, artifact=artifact)

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
    ) -> Optional[VulnerabilityReport]:
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
    fmt: Optional[VulnerabilityReportFormat] = None,
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

    async def write_vulnerability_report(
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
            name:        Stable unique key, e.g. ``"sqli-login"`` or ``"xss-profile"``.
            place_type:  Location kind — ``"file"`` or ``"url"``.
            place:       File path or URL where the issue was found.
            title:       Human-readable vulnerability title.
            summary:     One-sentence description.
            severity:    ``"info"``, ``"low"``, ``"medium"``, ``"high"``, or ``"critical"``.
            confidence:  ``"low"``, ``"medium"``, or ``"high"``.
            details:     Technical details, reproduction steps, impact, and remediation hints.
        """
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
        return {"result": vr.fmt.format_report(report)}

    async def get_vulnerability_report(
        name: str,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Read a vulnerability report by its unique name.

        Args:
            name: Exact report key.
        """
        report = await vr.get_report(name, tool_context)
        if report is None:
            return {"error": f"Vulnerability report {name!r} not found."}
        return {"result": vr.fmt.format_report(report)}

    async def list_vulnerabilities(
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        List vulnerability reports in insertion order.

        Returns:
            A preview list with title, location, severity, confidence, and summary.
        """
        reports = await vr.list_reports(tool_context)
        return {"result": vr.fmt.format_reports(reports, preview=True)}

    return [
        write_vulnerability_report,
        get_vulnerability_report,
        list_vulnerabilities,
    ]
