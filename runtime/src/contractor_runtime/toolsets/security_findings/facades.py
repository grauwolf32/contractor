"""Alternative model-facing finding schemas, selected explicitly by toolset."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from google.adk.tools.tool_context import ToolContext

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.finding_proposals import FindingProposalClient
from contractor_runtime.redaction import runtime_secrets
from contractor_runtime.toolsets.common.artifacts import ArtifactClientFactory
from contractor_runtime.toolsets.common.check_plans import ProposedCheck
from contractor_runtime.toolsets.common.factory import require_metrics, require_selected_tools
from contractor_runtime.toolsets.common.finding_locations import (
    ExactEvidenceRef,
    FindingSeverity,
    LineNumber,
    LineRange,
    Location,
    StandardReference,
)
from contractor_runtime.toolsets.common.resources import ToolResources
from contractor_runtime.toolsets.security_findings.project_reader import (
    ListProjectFindingsTool,
    ProjectFindingTools,
    ReadFindingSelection,
    ReadProjectFindingTool,
    SelectProjectFindingsTool,
)
from contractor_runtime.toolsets.security_findings.publisher import FindingPublisher
from contractor_runtime.toolsets.security_findings.reader import prepare_reader
from contractor_runtime.workspace import AllocationWorkspace


class _FindingFacade:
    name = "finding"
    description: str

    def __init__(self, publisher: FindingPublisher) -> None:
        self._publisher = publisher
        self.__name__ = self.name
        self.__doc__ = self.description

    async def close(self) -> None:
        self._publisher.close()


class GeneralFindingTool(_FindingFacade):
    description = """Record one finding with optional source/web locations.

    A proposal records an observation, not a confirmed vulnerability. Describe
    the impact and supporting observations. Locations may combine code and HTTP.
    Each location is either file with optional line/range, or url with optional
    method. Omit unknown coordinates and classification instead of guessing.

    Args:
        title: Short non-empty finding title.
        description: Observation, impact, prerequisites and reproduction details.
        locations: Optional explicit source or web locations.
        request_id: Optional local HTTP/Caido request_id inspected in this invocation.
            The selected captured exchange is saved automatically.
        cwe: Optional CWE-NNN weakness identifier from the pinned catalog.
        evidence_refs: Optional exact artifact references copied from tool output.
            With any selected exchange they total at most 64 MiB; a larger set
            fails with invalid_finding_request, so cite fewer artifacts.
        standard_refs: Optional exact standard identities copied from an assigned
            Audit task. A finding linked to a standard-mapped Audit result must
            cite every assigned entry ID.
        proposed_checks: Optional follow-up questions, each with objective and
            method. At most 512; these suggestions grant no execution authority.
        item_key: Optional actual Audit member key from read_audit_task. Required
            for task-bound batches; omit for a general incidental observation.
        incidental: An independent discovery that does not test the assigned hypothesis.
            Without item_key, retains Run/Audit origin only. Selecting a member adds
            Task origin; incidental calls never associate its hypothesis.
        severity_suggestion: Optional producer severity; leaves analyst assessment unset.
            Saving always requires an active Project and selected exact evidence.
    """

    async def __call__(
        self,
        title: str,
        description: str,
        tool_context: ToolContext,
        locations: list[Location] | None = None,
        request_id: LineNumber | None = None,
        cwe: str | None = None,
        evidence_refs: list[ExactEvidenceRef] | None = None,
        standard_refs: list[StandardReference] | None = None,
        proposed_checks: list[ProposedCheck] | None = None,
        severity_suggestion: FindingSeverity | None = None,
        item_key: str | None = None,
        incidental: bool = False,
    ) -> dict[str, Any]:
        return await self._publisher.submit(
            title=title,
            description=description,
            locations=locations or [],
            context=tool_context,
            cwe=cwe,
            evidence_refs=evidence_refs,
            standard_refs=standard_refs,
            proposed_checks=proposed_checks,
            severity_suggestion=severity_suggestion,
            item_key=item_key,
            incidental=incidental,
            request_id=request_id,
        )


class RecordFactTool(_FindingFacade):
    name = "record_fact"
    description = """Save one informational Project observation with exact evidence.

    An active Project and selected exact evidence are required. Describe what the
    source actually reports, including uncertainty; a note is a reported claim.
    The record has no automatic analyst verdict or severity assessment.

    Args:
        title: Short non-empty observation title.
        description: Observation, source quality, conditions and limitations.
        evidence_refs: Exact artifact references copied from tool output. At most four
            sources including a selected HTTP exchange; authored content is at most 32 KiB.
        request_id: Optional local HTTP/Caido request ID; saves its captured exchange.
        item_key: Optional assigned Audit member key to retain Task origin.
        incidental: True retains Run/Audit origin without a hypothesis. Set false
            to derive the selected Task's exact hypothesis; batch tasks need item_key.
    """

    async def __call__(
        self,
        title: str,
        description: str,
        tool_context: ToolContext,
        evidence_refs: list[ExactEvidenceRef] | None = None,
        request_id: LineNumber | None = None,
        item_key: str | None = None,
        incidental: bool = True,
    ) -> dict[str, Any]:
        return await self._publisher.submit(
            title=title,
            description=description,
            locations=[],
            context=tool_context,
            cwe=None,
            evidence_refs=evidence_refs,
            request_id=request_id,
            severity_suggestion="informational",
            _tool_name=self.name,
            item_key=item_key,
            incidental=incidental,
        )


class CodeFindingTool(_FindingFacade):
    description = """Record one code finding at an explicit source file.

    Explain the observation, impact and prerequisites. This records a proposal;
    it does not confirm exploitability. Use line or range, never both. Coordinates
    are one-based and inclusive. Omit unknown line numbers rather than guessing.

    Args:
        title: Short non-empty finding title.
        description: Source observation, reasoning, impact and prerequisites.
        file: Exact case-sensitive relative POSIX path in the inspected source.
        line: Optional one-based line number.
        range: Optional inclusive start_line/end_line object.
        cwe: Optional CWE-NNN weakness identifier from the pinned catalog.
        evidence_refs: Optional exact artifact references copied from tool output.
            They total at most 64 MiB; a larger set fails with
            invalid_finding_request, so cite fewer artifacts.
        standard_refs: Optional exact standard identities copied from an assigned
            Audit task. A finding linked to a standard-mapped Audit result must
            cite every assigned entry ID.
        proposed_checks: Optional follow-up questions, each with objective and
            method. At most 512; these suggestions grant no execution authority.
        item_key: Optional actual Audit member key from read_audit_task. Required
            for task-bound batches; omit for a general incidental observation.
        incidental: An independent discovery that does not test the assigned hypothesis.
            Without item_key, retains Run/Audit origin only. Selecting a member adds
            Task origin; incidental calls never associate its hypothesis.
        severity_suggestion: Optional producer severity; leaves analyst assessment unset.
            Saving always requires an active Project and selected exact evidence.
    """

    async def __call__(
        self,
        title: str,
        description: str,
        file: str,
        tool_context: ToolContext,
        line: LineNumber | None = None,
        range: LineRange | None = None,
        cwe: str | None = None,
        evidence_refs: list[ExactEvidenceRef] | None = None,
        standard_refs: list[StandardReference] | None = None,
        proposed_checks: list[ProposedCheck] | None = None,
        severity_suggestion: FindingSeverity | None = None,
        item_key: str | None = None,
        incidental: bool = False,
    ) -> dict[str, Any]:
        location: dict[str, Any] = {"file": file}
        if line is not None:
            location["line"] = line
        if range is not None:
            location["range"] = range.model_dump() if isinstance(range, LineRange) else range
        return await self._publisher.submit(
            title=title,
            description=description,
            locations=[location],
            context=tool_context,
            cwe=cwe,
            evidence_refs=evidence_refs,
            standard_refs=standard_refs,
            proposed_checks=proposed_checks,
            severity_suggestion=severity_suggestion,
            item_key=item_key,
            incidental=incidental,
        )


class HTTPFindingTool(_FindingFacade):
    description = """Record one HTTP finding at an explicit URL and method.

    Describe the observed behavior, impact and authorization context. An optional
    request_id attaches the captured outgoing request and response evidence from
    this invocation's recent HTTP history. Omit the ID to report without captured
    evidence. Reporting never sends or replays an HTTP request.

    Args:
        title: Short non-empty finding title.
        description: Observations, impact, prerequisites and reproduction details.
        url: Absolute HTTP/HTTPS URL of the affected endpoint.
        method: Explicit HTTP method; no default method is inferred.
        request_id: Optional local HTTP/Caido request_id inspected in this invocation.
            Saves the selected captured exchange automatically without sending traffic.
        cwe: Optional CWE-NNN weakness identifier from the pinned catalog.
        evidence_refs: Optional exact artifact references copied from tool output.
            With any selected exchange they total at most 64 MiB; a larger set
            fails with invalid_finding_request, so cite fewer artifacts.
        standard_refs: Optional exact standard identities copied from an assigned
            Audit task. A finding linked to a standard-mapped Audit result must
            cite every assigned entry ID.
        proposed_checks: Optional follow-up questions, each with objective and
            method. At most 512; these suggestions grant no execution authority.
        item_key: Optional actual Audit member key from read_audit_task. Required
            for task-bound batches; omit for a general incidental observation.
        incidental: An independent discovery that does not test the assigned hypothesis.
            Without item_key, retains Run/Audit origin only. Selecting a member adds
            Task origin; incidental calls never associate its hypothesis.
        severity_suggestion: Optional producer severity; leaves analyst assessment unset.
            Saving always requires an active Project and selected exact evidence.
    """

    async def __call__(
        self,
        title: str,
        description: str,
        url: str,
        method: str,
        tool_context: ToolContext,
        request_id: LineNumber | None = None,
        cwe: str | None = None,
        evidence_refs: list[ExactEvidenceRef] | None = None,
        standard_refs: list[StandardReference] | None = None,
        proposed_checks: list[ProposedCheck] | None = None,
        severity_suggestion: FindingSeverity | None = None,
        item_key: str | None = None,
        incidental: bool = False,
    ) -> dict[str, Any]:
        return await self._publisher.submit(
            title=title,
            description=description,
            locations=[{"url": url, "method": method}],
            context=tool_context,
            cwe=cwe,
            evidence_refs=evidence_refs,
            standard_refs=standard_refs,
            proposed_checks=proposed_checks,
            severity_suggestion=severity_suggestion,
            item_key=item_key,
            incidental=incidental,
            request_id=request_id,
        )


class GeneralFindingsToolsetFactory:
    ref = "security-findings@1"
    exported_tools = frozenset(
        {
            "record_fact",
            "finding",
            "list_findings",
            "list_project_findings",
            "read_project_finding",
            "select_project_findings",
            "read_finding_selection",
        }
    )
    infrastructure_channels = MappingProxyType({})
    tool_class = GeneralFindingTool

    def __init__(self, client_factory: ArtifactClientFactory | None = None) -> None:
        self._client_factory = client_factory or _unconfigured_client

    async def probe(self) -> frozenset[str]:
        return self.exported_tools

    async def create_selected(
        self,
        *,
        selected: Sequence[str],
        allocation_id: str,
        run_id: str,
        namespace: str,
        runtime_settings: RuntimeSettings,
        workspace: AllocationWorkspace,
        resources: ToolResources,
        adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
        project_workspace: Any = None,
    ) -> Mapping[str, Any]:
        del run_id, workspace, adapter_handles, project_workspace
        require_selected_tools(
            selected, self.exported_tools, description="unknown selected finding tools"
        )
        metrics = require_metrics(resources, "finding toolset")
        client = self._client_factory(allocation_id, runtime_settings)
        publisher = FindingPublisher(
            FindingProposalClient(client.allocation_id, client.transport),
            metrics,
            runtime_secrets(runtime_settings),
            resources,
            artifacts=client,
        )
        result = {}
        if "finding" in selected:
            result["finding"] = self.tool_class(publisher)
        if "record_fact" in selected:
            result["record_fact"] = RecordFactTool(publisher)
        if "list_findings" in selected:
            result["list_findings"] = await prepare_reader(
                client, metrics, runtime_secrets(runtime_settings)
            )
        project = ProjectFindingTools(client, metrics, runtime_secrets(runtime_settings), namespace)
        for tool_class in (
            ListProjectFindingsTool,
            ReadProjectFindingTool,
            SelectProjectFindingsTool,
        ):
            if tool_class.name in selected:
                result[tool_class.name] = tool_class(
                    client, metrics, runtime_secrets(runtime_settings), namespace
                )
        if "read_finding_selection" in selected:
            result["read_finding_selection"] = ReadFindingSelection(project)
        return result


class CodeFindingsToolsetFactory(GeneralFindingsToolsetFactory):
    ref = "security-findings-code@1"
    exported_tools = frozenset({"finding"})
    tool_class = CodeFindingTool


class HTTPFindingsToolsetFactory(GeneralFindingsToolsetFactory):
    ref = "security-findings-http@1"
    exported_tools = frozenset({"finding"})
    tool_class = HTTPFindingTool


def _unconfigured_client(_allocation_id: str, _settings: RuntimeSettings) -> ArtifactClient:
    raise RuntimeError("finding toolsets require an ArtifactClient factory")
