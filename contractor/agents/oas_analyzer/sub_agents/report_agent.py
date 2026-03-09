from __future__ import annotations

import logging
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types
from typing_extensions import override

from contractor.agents.oas_analyzer.models import (
    EndpointVulnerability,
    ServiceBasicInfo,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def format_vulnerability(vulnerability: EndpointVulnerability) -> str:
    """
    Format vulnerability into Markdown table format
    """

    severity_to_emoji = {
        "low": "🟢",
        "medium": "🟡",
        "high": "🔴",
        "critical": "💀",
    }

    return (
        "<table>\n"
        "<tr>\n"
        "<th>Path</th>\n"
        f"<td>{vulnerability.path}</td>\n"
        "<th>Method</th>\n"
        f"<td>{vulnerability.method}</td>\n"
        "<th>Severity</th>\n"
        f"<td>{severity_to_emoji.get(vulnerability.severity, '🟡')}</td>\n"
        "<th>Confidence</th>\n"
        f"<td>{vulnerability.confidence}</td>\n"
        "</tr>\n"
        "<tr>\n"
        "<th>Parameters</th>\n"
        f'<td colspan="7">{vulnerability.parameters}</td>\n'
        "</tr>\n"
        "<tr>\n"
        "<th>Vulnerability</th>\n"
        f'<td colspan="7">{vulnerability.vulnerability}</td>\n'
        "</tr>\n"
        "<tr>\n"
        "<th>Description</th>\n"
        f'<td colspan="7">{vulnerability.description}</td>\n'
        "</tr>\n"
        "</table>\n"
    )


def format_vulnerabilities(vulnerabilities: list[dict]) -> str:
    """
    Format the vulnerabilities into a string.
    """
    tags = {vulnerability["tag"] for vulnerability in vulnerabilities}
    tags = sorted(tags)
    result = ""

    vulns_by_tag = {tag: [] for tag in tags}
    for vulnerability in vulnerabilities:
        vulns_by_tag[vulnerability["tag"]].append(vulnerability)

    for tag in vulns_by_tag:
        vulns_by_tag[tag] = sorted(vulns_by_tag[tag], key=lambda x: x["severity"])

    for tag in tags:
        result += f"\n\n## {tag} \n\n"
        for vulnerability in vulns_by_tag[tag]:
            result += (
                format_vulnerability(EndpointVulnerability(**vulnerability)) + "\n"
            )
        result += "\n"

    return result


def extract_mermaid_diagram(text: str) -> str:
    """
    Extract the mermaid diagram from the text.
    """
    start = text.find("```mermaid")
    end = text.find("```", start + 10)
    if start == -1 or end == -1:
        return ""
    return text[start + 10 : end].strip()


def format_service_information(service_information: dict) -> str:
    service_information = ServiceBasicInfo(**service_information)
    mermaid_diagram = extract_mermaid_diagram(service_information.diagram)

    return f"""
    # Service Information

    ## Name
    {service_information.name}

    ## Description
    {service_information.description}

    ## Summary
    {service_information.summary}

    ## Diagram
    ```mermaid
    {mermaid_diagram}
    ```

    ## Criticality
    {service_information.criticality}

    ## Criticality Reason
    {service_information.criticality_reason}
    """


class ReportAgent(BaseAgent):
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name: str):
        super().__init__(name=name)

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        service_information = ctx.session.state.get("oas_analyzer::service_information")
        vulnerabilities = ctx.session.state.get("oas_analyzer::vulnerabilities", [])

        report = format_service_information(service_information)

        if vulnerabilities:
            report += "\n\n"
            report += format_vulnerabilities(vulnerabilities)

        content = types.Content(
            parts=[types.Part.from_text(text=report)],
            role="system",
        )

        await ctx.artifact_service.save_artifact(
            app_name=ctx.app_name,
            user_id=ctx.user_id,
            filename="oas_vulnerabilities.md",
            artifact=content,
        )

        yield Event(
            content=content,
            turn_complete=True,
            invocation_id=ctx.invocation_id,
            author=self.name,
        )


report_agent = ReportAgent(name="report_agent")
