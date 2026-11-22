"""Workspace-coordinate finding facade over the ordinary finding publisher."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import jcs
from pydantic import Field, ValidationError, model_validator

from contractor_runtime import threads
from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.artifacts import ArtifactAPIError, ArtifactTransportError
from contractor_runtime.contracts import DIGEST_PATTERN, ArtifactRef, RuntimeSettings
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.core.tool_metrics import RecordedToolCall
from contractor_runtime.digests import sha256_digest
from contractor_runtime.finding_proposals import FindingProposalClient
from contractor_runtime.projectfs.storage import WorkspaceSourceReader, WorkspaceStorageError
from contractor_runtime.redaction import runtime_secrets
from contractor_runtime.strict_json import strict_json_loads
from contractor_runtime.toolsets.common.artifact_write import (
    ArtifactReconciliationFailed,
    definite_write_rejection,
    write_artifact_exact,
)
from contractor_runtime.toolsets.common.call_identity import call_client_key
from contractor_runtime.toolsets.common.factory import require_metrics, require_selected_tools
from contractor_runtime.toolsets.common.finding_locations import (
    ClosedModel,
    ExactEvidenceRef,
    LineNumber,
    SourceLocation,
    normalize_locations,
)
from contractor_runtime.toolsets.common.resources import ToolResources
from contractor_runtime.toolsets.common.sources import SourcePreparer
from contractor_runtime.toolsets.common.trace_source import (
    MAX_TRACE_BYTES,
    MAX_TRACE_LINES,
    trace_source,
)
from contractor_runtime.toolsets.security_findings.facades import (
    CodeFindingsToolsetFactory,
    CodeFindingTool,
)
from contractor_runtime.toolsets.security_findings.publisher import (
    FindingPublisher,
    _bounded_finding_error,
)
from contractor_runtime.workspace import AllocationWorkspace

# The envelope includes JSON escaping of the bounded source text and small metadata.
MAX_PROOF_BYTES = 8 * MAX_TRACE_BYTES


class ProofLocation(ClosedModel):
    file: str
    start_line: LineNumber = Field(alias="startLine")
    end_line: LineNumber = Field(alias="endLine")
    digest: str = Field(pattern=DIGEST_PATTERN.pattern)

    @model_validator(mode="after")
    def check_location(self):
        SourceLocation(
            file=self.file, range={"start_line": self.start_line, "end_line": self.end_line}
        )
        return self


class ProofSource(ProofLocation):
    artifact: ExactEvidenceRef


class ProofWorkspace(ProofLocation):
    snippet_start: LineNumber = Field(alias="snippetStartLine")
    snippet_end: LineNumber = Field(alias="snippetEndLine")
    text: str

    @model_validator(mode="after")
    def check_snippet(self):
        if not (self.snippet_start <= self.start_line <= self.end_line <= self.snippet_end):
            raise ValueError("trace proof window does not contain its selection")
        if (
            self.snippet_end - self.snippet_start >= MAX_TRACE_LINES
            or len(self.text.encode()) > MAX_TRACE_BYTES
        ):
            raise ValueError("trace proof exceeds its source window bounds")
        return self


class TraceProof(ClosedModel):
    schema_id: Literal["contractor.trace-finding-evidence.v1"] = Field(alias="schema")
    source: ProofSource
    workspace: ProofWorkspace
    request_digest: str = Field(alias="requestDigest", pattern=DIGEST_PATTERN.pattern)


def _read_proof(data: bytes, digest: str) -> dict:
    try:
        proof = TraceProof.model_validate(strict_json_loads(data))
    except (ValueError, ValidationError, UnicodeDecodeError, RecursionError):
        raise ToolInputError("trace finding proof is invalid") from None
    if proof.request_digest != digest:
        raise ToolInputError("trace finding tool-call identity was reused with different arguments")
    return proof.model_dump(by_alias=True)


def _wire(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, list):
        return [_wire(item) for item in value]
    if isinstance(value, dict):
        return {key: _wire(item) for key, item in value.items()}
    return value


class TraceFindingPublisher(FindingPublisher):
    def __init__(self, *args, reader: WorkspaceSourceReader, namespace: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._reader = reader
        self._namespace = namespace

    async def submit(self, **kwargs) -> dict[str, Any]:
        with RecordedToolCall(
            self.metrics,
            "finding",
            {},
            secrets=self.secrets,
            bound_error=_bounded_finding_error,
        ) as call:
            kwargs = await self.prepare_source_selection(kwargs)
            prepared = await self._prepare(kwargs)
            return await super()._submit(**prepared, _call=call)

    async def _prepare(self, kwargs: dict) -> dict:
        context = kwargs["context"]
        locations = normalize_locations(kwargs["locations"])
        if len(locations) != 1 or "file" not in locations[0]:
            raise ToolInputError("trace finding requires one source location")
        location = locations[0]
        if "line" not in location and "range" not in location:
            raise ToolInputError("trace finding requires a current workspace line or range")
        request = _wire(
            {
                key: value
                for key, value in kwargs.items()
                if key != "context" and key != "_source_normalization"
            }
        )
        request_digest = await threads.to_thread_until_done(
            lambda: sha256_digest(jcs.canonicalize(request)), name="trace-request-digest"
        )
        identity = jcs.canonicalize([context.invocation_id, call_client_key(context)])
        target = ArtifactRef(
            namespace=self._namespace,
            name="trace-proof-" + sha256_digest(identity).removeprefix("sha256:"),
        )
        client = self.artifacts
        if client is None:
            raise RuntimeError("trace findings require an Artifact client")
        # Replay the immutable call proof before consulting a changed workspace.
        try:
            saved = await client.read_artifact(target, max_bytes=MAX_PROOF_BYTES)
        except ArtifactAPIError as error:
            if error.status_code != 404:
                raise
            saved = None
        if saved is None:
            try:
                capture = await self._reader.capture_source_file(location["file"])
            except WorkspaceStorageError:
                raise ToolInputError("trace finding source is unavailable") from None
            start = location.get("line", location.get("range", {}).get("start_line"))
            end = location.get("range", {}).get("end_line", start)
            proof = await threads.to_thread_until_done(
                trace_source, capture, start, end, name="trace-finding-source"
            )
            proof["requestDigest"] = request_digest
            TraceProof.model_validate(proof)
            data = await threads.to_thread_until_done(
                jcs.canonicalize, proof, name="trace-proof-encode"
            )
            if len(data) > MAX_PROOF_BYTES:
                raise ToolInputError("trace finding proof exceeds the evidence byte limit")
            try:
                saved = await write_artifact_exact(
                    client,
                    target,
                    data=data,
                    media_type="application/json",
                    expected_revision=None,
                    max_bytes=MAX_PROOF_BYTES,
                    reject_first=lambda error: (
                        definite_write_rejection(error) and error.code != "artifact_conflict"
                    ),
                    reject_replay=lambda error: (
                        definite_write_rejection(error) and error.code != "artifact_conflict"
                    ),
                )
            except ArtifactReconciliationFailed as error:
                if error.read_error is not None:
                    raise error.read_error from None
                if error.current is None:
                    raise ArtifactTransportError(
                        "trace finding proof write is unresolved"
                    ) from None
                raise ToolInputError("trace finding proof conflicts with this tool call") from None
        else:
            if saved.media_type != "application/json":
                raise ToolInputError("trace finding proof has an invalid media type")
            proof = await threads.to_thread_until_done(
                _read_proof, saved.data, request_digest, name="trace-proof-read"
            )
        source = proof["source"]
        original_location = {"file": source["file"]}
        if "range" in location:
            original_location["range"] = {
                "start_line": source["startLine"],
                "end_line": source["endLine"],
            }
        else:
            original_location["line"] = source["startLine"]
        kwargs["locations"] = [original_location]
        refs = [_wire(ref) for ref in kwargs.get("evidence_refs") or []]
        # Ordinary intake retains both the original source and the annotated proof.
        for ref in (source["artifact"], saved.artifact.model_dump(exclude_none=True)):
            if ref not in refs:
                refs.append(ref)
        kwargs["evidence_refs"] = refs
        return kwargs


class TraceFindingTool(CodeFindingTool):
    description = (
        CodeFindingTool.description.replace(
            "at an explicit source file.", "from the current annotated workspace."
        )
        + """
    Supply file and line/range from the current workspace. The Runtime saves exact
    annotated evidence and translates coordinates to the pinned original source.
    A line or range is required. Unsupported code edits return a tool input error.
    """
    )


class TraceFindingsToolsetFactory(CodeFindingsToolsetFactory):
    ref = "security-findings-trace@1"
    requires_workspace = True
    workspace_access = "source"

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
        project_workspace: WorkspaceSourceReader | None = None,
    ) -> Mapping[str, Any]:
        del workspace, adapter_handles
        require_selected_tools(
            selected, self.exported_tools, description="unknown trace finding tools"
        )
        if "finding" not in selected:
            return {}
        if project_workspace is None:
            raise RuntimeError("trace findings require an overlay source reader")
        client = self._client_factory(allocation_id, runtime_settings)
        publisher = TraceFindingPublisher(
            FindingProposalClient(client.allocation_id, client.transport),
            require_metrics(resources, "finding toolset"),
            runtime_secrets(runtime_settings),
            resources,
            artifacts=client,
            source_preparer=SourcePreparer(
                client,
                resources,
                run_id,
                namespace,
                runtime_settings.request_timeout_seconds,
                project_workspace,
            ),
            reader=project_workspace,
            namespace=namespace,
        )
        return {"finding": TraceFindingTool(publisher)} if "finding" in selected else {}
