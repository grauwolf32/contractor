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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


Severity = Literal["info", "low", "medium", "high", "critical"]
Confidence = Literal["low", "medium", "high"]
PlaceType = Literal["file", "url"]


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
    created_at: str = ""
    updated_at: str = ""


@dataclass
class VulnerabilityReportFormat:
    _format: Literal["json", "markdown", "yaml", "xml"] = "json"

    @staticmethod
    def _type_hint(
        output: Union[str, dict[str, Any], list[Any]],
        fmt: str,
        type_hint: bool = False,
    ) -> Union[str, dict[str, Any], list[Any]]:
        if not type_hint or not isinstance(output, str):
            return output
        return f"```{fmt}\n{output}\n```"

    @staticmethod
    def _report_to_json(report: VulnerabilityReport, **kwargs) -> dict[str, Any]:
        return asdict(report)

    @staticmethod
    def _report_preview_to_json(report: VulnerabilityReport, **kwargs) -> dict[str, Any]:
        return {
            "name": report.name,
            "title": report.title,
            "place_type": report.place_type,
            "place": report.place,
            "severity": report.severity,
            "confidence": report.confidence,
            "summary": report.summary,
            "ordinal": report.ordinal,
            "created_at": report.created_at,
            "updated_at": report.updated_at,
        }

    @staticmethod
    def _report_to_markdown(report: VulnerabilityReport, **kwargs) -> str:
        return (
            f"### {report.title}\n"
            f"**Name**: {report.name}\n"
            f"**Place**: `{report.place_type}` — `{report.place}`\n"
            f"**Severity**: {report.severity}\n"
            f"**Confidence**: {report.confidence}\n"
            f"**Summary**: {report.summary}\n"
            f"**Created At**: {report.created_at or '-'}\n"
            f"**Updated At**: {report.updated_at or '-'}\n"
            f"**Details**:\n{report.details}\n"
        )

    @staticmethod
    def _report_preview_to_markdown(report: VulnerabilityReport, **kwargs) -> str:
        return (
            f"### {report.title}\n"
            f"**Name**: {report.name}\n"
            f"**Place**: `{report.place_type}` — `{report.place}`\n"
            f"**Severity**: {report.severity}\n"
            f"**Confidence**: {report.confidence}\n"
            f"**Summary**: {report.summary}\n"
            f"**Created At**: {report.created_at or '-'}\n"
            f"**Updated At**: {report.updated_at or '-'}\n"
        )

    @staticmethod
    def _report_to_yaml(report: VulnerabilityReport, **kwargs) -> str:
        payload = {
            f"vulnerability_{report.name}": {
                "name": report.name,
                "place_type": report.place_type,
                "place": report.place,
                "title": report.title,
                "summary": report.summary,
                "severity": report.severity,
                "confidence": report.confidence,
                "details": report.details,
                "ordinal": report.ordinal,
                "created_at": report.created_at,
                "updated_at": report.updated_at,
            }
        }
        return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)

    @staticmethod
    def _report_preview_to_yaml(report: VulnerabilityReport, **kwargs) -> str:
        payload = {
            f"vulnerability_{report.name}": {
                "name": report.name,
                "place_type": report.place_type,
                "place": report.place,
                "title": report.title,
                "summary": report.summary,
                "severity": report.severity,
                "confidence": report.confidence,
                "ordinal": report.ordinal,
                "created_at": report.created_at,
                "updated_at": report.updated_at,
            }
        }
        return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)

    @staticmethod
    def _report_to_xml(report: VulnerabilityReport, indent: int = 0, **kwargs) -> str:
        pad = " " * (indent * 4)
        pad2 = " " * ((indent + 1) * 4)
        return (
            f'{pad}<vulnerability name="{xml_escape(report.name)}">\n'
            f"{pad2}<title>{xml_escape(report.title)}</title>\n"
            f"{pad2}<place type=\"{xml_escape(report.place_type)}\">{xml_escape(report.place)}</place>\n"
            f"{pad2}<severity>{xml_escape(report.severity)}</severity>\n"
            f"{pad2}<confidence>{xml_escape(report.confidence)}</confidence>\n"
            f"{pad2}<summary>{xml_escape(report.summary)}</summary>\n"
            f"{pad2}<created_at>{xml_escape(report.created_at)}</created_at>\n"
            f"{pad2}<updated_at>{xml_escape(report.updated_at)}</updated_at>\n"
            f"{pad2}<details>{xml_escape(report.details)}</details>\n"
            f"{pad}</vulnerability>"
        )

    @staticmethod
    def _report_preview_to_xml(report: VulnerabilityReport, indent: int = 0, **kwargs) -> str:
        pad = " " * (indent * 4)
        pad2 = " " * ((indent + 1) * 4)
        return (
            f'{pad}<vulnerability name="{xml_escape(report.name)}">\n'
            f"{pad2}<title>{xml_escape(report.title)}</title>\n"
            f"{pad2}<place type=\"{xml_escape(report.place_type)}\">{xml_escape(report.place)}</place>\n"
            f"{pad2}<severity>{xml_escape(report.severity)}</severity>\n"
            f"{pad2}<confidence>{xml_escape(report.confidence)}</confidence>\n"
            f"{pad2}<summary>{xml_escape(report.summary)}</summary>\n"
            f"{pad2}<created_at>{xml_escape(report.created_at)}</created_at>\n"
            f"{pad2}<updated_at>{xml_escape(report.updated_at)}</updated_at>\n"
            f"{pad}</vulnerability>"
        )

    def format_report(
        self,
        report: VulnerabilityReport,
        *,
        type_hint: bool = False,
    ) -> Union[str, dict[str, Any]]:
        formatters = {
            "json": self._report_to_json,
            "markdown": self._report_to_markdown,
            "yaml": self._report_to_yaml,
            "xml": self._report_to_xml,
        }
        formatter = formatters.get(self._format, self._report_to_json)
        output = formatter(report)
        return self._type_hint(output, self._format, type_hint)

    def format_report_preview(
        self,
        report: VulnerabilityReport,
        *,
        type_hint: bool = False,
    ) -> Union[str, dict[str, Any]]:
        formatters = {
            "json": self._report_preview_to_json,
            "markdown": self._report_preview_to_markdown,
            "yaml": self._report_preview_to_yaml,
            "xml": self._report_preview_to_xml,
        }
        formatter = formatters.get(self._format, self._report_preview_to_json)
        output = formatter(report)
        return self._type_hint(output, self._format, type_hint)

    def format_reports(
        self,
        reports: list[VulnerabilityReport],
        *,
        type_hint: bool = False,
        preview: bool = False,
    ) -> Union[str, list[dict[str, Any]]]:
        if self._format == "json":
            if preview:
                return [self._report_preview_to_json(r) for r in reports]
            return [self._report_to_json(r) for r in reports]

        if self._format in {"markdown", "yaml"}:
            formatter = self.format_report_preview if preview else self.format_report
            output = "\n".join(
                item for item in (formatter(r, type_hint=False) for r in reports) if isinstance(item, str)
            )
            return self._type_hint(output, self._format, type_hint)

        if self._format == "xml":
            formatter = self._report_preview_to_xml if preview else self._report_to_xml
            output = "<vulnerabilities>\n" + "\n".join(formatter(r, indent=1) for r in reports) + "\n</vulnerabilities>"
            return self._type_hint(output, self._format, type_hint)

        if preview:
            return [self._report_preview_to_json(r) for r in reports]
        return [self._report_to_json(r) for r in reports]


@dataclass
class VulnerabilityReportTools:
    name: str
    fmt: VulnerabilityReportFormat = field(default_factory=VulnerabilityReportFormat)
    reports: dict[str, VulnerabilityReport] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def artifact_key(self) -> str:
        return f"user:vulnerability-reports/{self.name}"

    def _normalize_report(
        self,
        name: str,
        item: dict[str, Any],
        fallback_ordinal: int,
    ) -> VulnerabilityReport:
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
            created_at=item.get("created_at", ""),
            updated_at=item.get("updated_at", ""),
        )

    def _next_ordinal(self) -> int:
        if not self.reports:
            return 1
        return max(report.ordinal for report in self.reports.values()) + 1

    async def load(self, ctx: ToolContext | CallbackContext):
        async with self._lock:
            artifact = await ctx.load_artifact(filename=self.artifact_key())
            if artifact is None:
                self.reports = {}
                return

            raw = yaml.safe_load(artifact.text) or {}
            reports: dict[str, VulnerabilityReport] = {}

            for index, (name, item) in enumerate(raw.items(), start=1):
                if not isinstance(item, dict):
                    continue
                report = self._normalize_report(name=name, item=item, fallback_ordinal=index)
                reports[report.name] = report

            self.reports = reports

    def dump(self) -> str:
        payload = {
            name: asdict(report)
            for name, report in sorted(
                self.reports.items(),
                key=lambda pair: (pair[1].ordinal, pair[0]),
            )
        }
        return yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )

    async def save(self, ctx: ToolContext | CallbackContext):
        async with self._lock:
            artifact = types.Part.from_text(text=self.dump())
            await ctx.save_artifact(filename=self.artifact_key(), artifact=artifact)

    async def list_reports(self, ctx: ToolContext | CallbackContext) -> list[VulnerabilityReport]:
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
        await self.load(ctx)
        now = utc_now_iso()
        existing = self.reports.get(name)

        if existing is not None:
            report = VulnerabilityReport(
                name=name,
                place_type=place_type,
                place=place,
                title=title,
                summary=summary,
                severity=severity,
                confidence=confidence,
                details=details,
                ordinal=existing.ordinal,
                created_at=existing.created_at or now,
                updated_at=now,
            )
        else:
            report = VulnerabilityReport(
                name=name,
                place_type=place_type,
                place=place,
                title=title,
                summary=summary,
                severity=severity,
                confidence=confidence,
                details=details,
                ordinal=self._next_ordinal(),
                created_at=now,
                updated_at=now,
            )

        self.reports[name] = report
        await self.save(ctx)
        return report


def vulnerability_report_tools(
    name: str,
    fmt: VulnerabilityReportFormat = VulnerabilityReportFormat("json"),
):
    vr = VulnerabilityReportTools(name=name, fmt=fmt)

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
        Writes or updates a vulnerability report.

        Args:
            name: Stable unique key of the report, for example "sqli-login" or "xss-profile-preview".
            place_type: Where the issue is located: "file" or "url".
            place: File path or URL where the issue was found.
            title: Human-readable vulnerability title.
            summary: Short description of the issue.
            severity: One of "info", "low", "medium", "high", "critical".
            confidence: One of "low", "medium", "high".
            details: Technical details, reproduction notes, impact, or remediation hints.
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
        Reads a vulnerability report by its unique name.

        Args:
            name: Exact report key.
        """
        report = await vr.get_report(name, tool_context)
        if report is None:
            return {"error": f"vulnerability report {name} not found"}
        return {"result": vr.fmt.format_report(report)}

    async def list_vulnerability_reports(
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Lists vulnerability reports in insertion order.

        Returns:
            A preview list with title, location, severity, confidence and summary.
        """
        reports = await vr.list_reports(tool_context)
        return {"result": vr.fmt.format_reports(reports, preview=True)}

    return {
        "vulnerability_report_write": write_vulnerability_report,
        "vulnerability_report_get": get_vulnerability_report,
        "vulnerability_report_list": list_vulnerability_reports,
    }