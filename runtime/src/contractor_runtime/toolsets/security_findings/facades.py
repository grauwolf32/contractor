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
    FindingSeverity,
    LineNumber,
    LineRange,
    Location,
    StandardReference,
)
from contractor_runtime.toolsets.common.resources import ToolResources
from contractor_runtime.toolsets.common.sources import SourcePreparer, SourceSelector
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
        cwe: Optional CWE-NNN weakness identifier from the pinned catalog.
        sources: Selected code, HTTP or artifact objects, native array or serialized once.
            Artifact selectors use kind=artifact and namespace/name; Runtime supplies
            the selected revision. Code selects file/start_line/end_line in the opened
            pinned archive; workspace=true verifies current workspace coordinates.
            HTTP selects kind=http and a local request_id without replay; at most one
            local exchange. Retained sources preserve their original observation context.
            A fact statement or Memory note does not independently verify its claim.
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
        cwe: str | None = None,
        sources: list[SourceSelector] | str | None = None,
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
            source_selection=sources,
            standard_refs=standard_refs,
            proposed_checks=proposed_checks,
            severity_suggestion=severity_suggestion,
            item_key=item_key,
            incidental=incidental,
        )


class FactTool(_FindingFacade):
    name = "fact"
    description = """Save one informational Project observation with exact evidence.

    An active Project and selected exact evidence are required. Describe what the
    source actually reports, including uncertainty; a note is a reported claim.
    The record has no automatic analyst verdict or severity assessment.

    Args:
        title: Short non-empty observation title.
        description: Observation, source quality, conditions and limitations.
        sources: Selected code, HTTP or artifact objects, native array or serialized once.
            Artifact selectors use kind=artifact and namespace/name; Runtime supplies
            the selected revision. Code selects file/start_line/end_line in the opened
            pinned archive; workspace=true verifies current workspace coordinates.
            HTTP selects kind=http and a local request_id without replay; at most one
            local exchange. Retained sources preserve their original observation context.
            A fact statement or Memory note does not independently verify its claim.
        memory_name: Optional exact memory name from the assigned memories snapshot.
            Attaches its complete original content, description and tags; no quotation
            is required. Describe uncertainty and inference in your own words.
        item_key: Optional assigned Audit member key to retain Task origin.
        incidental: True retains Run/Audit origin without a hypothesis. Set false
            to derive the selected Task's exact hypothesis; batch tasks need item_key.
    """

    async def __call__(
        self,
        title: str,
        description: str,
        tool_context: ToolContext,
        sources: list[SourceSelector] | str | None = None,
        memory_name: str | None = None,
        item_key: str | None = None,
        incidental: bool = True,
    ) -> dict[str, Any]:
        return await self._publisher.submit(
            title=title,
            description=description,
            locations=[],
            context=tool_context,
            cwe=None,
            source_selection=sources,
            memory_name=memory_name,
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
        sources: Selected code, HTTP or artifact objects, native array or serialized once.
            Artifact selectors use kind=artifact and namespace/name; Runtime supplies
            the selected revision. Code selects file/start_line/end_line in the opened
            pinned archive; workspace=true verifies current workspace coordinates.
            HTTP selects kind=http and a local request_id without replay; at most one
            local exchange. Retained sources preserve their original observation context.
            A fact statement or Memory note does not independently verify its claim.
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
        sources: list[SourceSelector] | str | None = None,
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
            source_selection=sources,
            standard_refs=standard_refs,
            proposed_checks=proposed_checks,
            severity_suggestion=severity_suggestion,
            item_key=item_key,
            incidental=incidental,
        )


class HTTPFindingTool(_FindingFacade):
    description = """Record one HTTP finding at an explicit URL and method.

    Describe the observed behavior, impact and authorization context. An optional
    HTTP source selector attaches the captured outgoing request and response
    from the allocation's recent HTTP history. Reporting never sends or replays
    an HTTP request.

    Args:
        title: Short non-empty finding title.
        description: Observations, impact, prerequisites and reproduction details.
        url: Absolute HTTP/HTTPS URL of the affected endpoint.
        method: Explicit HTTP method; no default method is inferred.
        cwe: Optional CWE-NNN weakness identifier from the pinned catalog.
        sources: Selected code, HTTP or artifact objects, native array or serialized once.
            Artifact selectors use kind=artifact and namespace/name; Runtime supplies
            the selected revision. Code selects file/start_line/end_line in the opened
            pinned archive; workspace=true verifies current workspace coordinates.
            HTTP selects kind=http and a local request_id without replay; at most one
            local exchange. Retained sources preserve their original observation context.
            A fact statement or Memory note does not independently verify its claim.
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
        cwe: str | None = None,
        sources: list[SourceSelector] | str | None = None,
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
            source_selection=sources,
            standard_refs=standard_refs,
            proposed_checks=proposed_checks,
            severity_suggestion=severity_suggestion,
            item_key=item_key,
            incidental=incidental,
        )


class GeneralFindingsToolsetFactory:
    workspace_access = "source"
    ref = "security-findings@1"
    exported_tools = frozenset(
        {
            "fact",
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
        del workspace, adapter_handles
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
            namespace=namespace,
            source_preparer=SourcePreparer(
                client,
                resources,
                run_id,
                namespace,
                runtime_settings.request_timeout_seconds,
                project_workspace,
            ),
        )
        result = {}
        if "finding" in selected:
            result["finding"] = self.tool_class(publisher)
        if "fact" in selected:
            result["fact"] = FactTool(publisher)
        if "list_findings" in selected:
            result["list_findings"] = await prepare_reader(
                client, metrics, runtime_secrets(runtime_settings)
            )
        project = ProjectFindingTools(
            client,
            metrics,
            runtime_secrets(runtime_settings),
            namespace,
            resources.selected_versions,
        )
        for tool_class in (
            ListProjectFindingsTool,
            ReadProjectFindingTool,
            SelectProjectFindingsTool,
        ):
            if tool_class.name in selected:
                result[tool_class.name] = tool_class(
                    client,
                    metrics,
                    runtime_secrets(runtime_settings),
                    namespace,
                    resources.selected_versions,
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
