"""Alternative model-facing finding schemas, selected explicitly by toolset."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from google.adk.tools.tool_context import ToolContext

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.toolsets.common.artifacts import ArtifactClientFactory, gateway_secrets
from contractor_runtime.toolsets.security_findings.locations import (
    ExactEvidenceRef,
    LineNumber,
    LineRange,
    Location,
    StandardReference,
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
        evidence_refs: Optional exact artifact references copied from tool output.
        standard_refs: Optional exact standard identities copied from an assigned Audit task.
    """

    async def __call__(
        self,
        title: str,
        description: str,
        tool_context: ToolContext,
        locations: list[Location] | None = None,
        cwe: str | None = None,
        evidence_refs: list[ExactEvidenceRef] | None = None,
        standard_refs: list[StandardReference] | None = None,
    ) -> dict[str, str]:
        return await self._publisher.submit(
            title=title,
            description=description,
            locations=locations or [],
            context=tool_context,
            cwe=cwe,
            evidence_refs=evidence_refs,
            standard_refs=standard_refs,
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
        standard_refs: Optional exact standard identities copied from an assigned Audit task.
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
    ) -> dict[str, str]:
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
        request_id: Optional ID returned by http_request in this invocation.
        cwe: Optional CWE-NNN weakness identifier from the pinned catalog.
        evidence_refs: Optional exact artifact references copied from tool output.
        standard_refs: Optional exact standard identities copied from an assigned Audit task.
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
    ) -> dict[str, str]:
        return await self._publisher.submit(
            title=title,
            description=description,
            locations=[{"url": url, "method": method}],
            context=tool_context,
            cwe=cwe,
            evidence_refs=evidence_refs,
            standard_refs=standard_refs,
            request_id=request_id,
        )


class GeneralFindingsToolsetFactory:
    ref = "security-findings@1"
    exported_tools = frozenset({"finding", "list_findings"})
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
        state: Any,
        adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
        project_workspace: Any = None,
    ) -> Mapping[str, Any]:
        del run_id, namespace, workspace, adapter_handles, project_workspace
        if set(selected) - self.exported_tools:
            raise ValueError("finding toolset selected an unknown tool")
        metrics = getattr(state, "metrics", None)
        if metrics is None or not callable(getattr(metrics, "record_tool_call", None)):
            raise TypeError("finding toolset requires State.metrics")
        client = self._client_factory(allocation_id, runtime_settings)
        publisher = FindingPublisher(
            client,
            metrics,
            gateway_secrets(runtime_settings),
            state,
        )
        result = {}
        if "finding" in selected:
            result["finding"] = self.tool_class(publisher)
        if "list_findings" in selected:
            result["list_findings"] = await prepare_reader(
                client, metrics, gateway_secrets(runtime_settings)
            )
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
