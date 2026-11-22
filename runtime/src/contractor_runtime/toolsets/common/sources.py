"""Explicit code, HTTP and material selection shared by claim-owning tools."""

from __future__ import annotations

import hashlib
from typing import Annotated, Literal

import jcs
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from contractor_runtime import threads
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.contracts.base import MAX_EXACT_INTEGER
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.core.tool_errors import ToolResultDiagnostic
from contractor_runtime.digests import sha256_digest
from contractor_runtime.toolsets.common.argument_diagnostics import argument_details
from contractor_runtime.toolsets.common.arguments import (
    MAX_CONTAINER_DEPTH,
    MAX_CONTAINER_NODES,
    container_argument,
)
from contractor_runtime.toolsets.common.artifact_write import (
    definite_write_rejection,
    write_artifact_exact,
)
from contractor_runtime.toolsets.common.http_publication import HTTPExchangePublisher
from contractor_runtime.toolsets.common.trace_source import trace_source
from contractor_runtime.toolsets.common.versions import MaterialSelector


class ArtifactSource(MaterialSelector):
    kind: Literal["artifact"]


class HTTPSource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["http"]
    request_id: int = Field(gt=0, le=MAX_EXACT_INTEGER)


class CodeSource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["code"]
    file: str = Field(min_length=1, max_length=512)
    start_line: int = Field(ge=1, le=MAX_EXACT_INTEGER)
    end_line: int = Field(ge=1, le=MAX_EXACT_INTEGER)
    workspace: bool = False


SourceSelector = Annotated[ArtifactSource | HTTPSource | CodeSource, Field(discriminator="kind")]
_ADAPTER = TypeAdapter(list[SourceSelector])


class SourceArgumentError(ToolInputError, ToolResultDiagnostic):
    def __init__(self, field, message, **details):
        super().__init__(field + ": " + message)
        self.model_details = argument_details(
            field,
            **{
                key: value
                for key, value in details.items()
                if key in {"expected_type", "actual_type"}
            },
        )


def parse_sources(value, notices=None, *, allow_empty=False):
    if isinstance(value, list):
        value = [item.model_dump() if isinstance(item, BaseModel) else item for item in value]
    values, normalized = container_argument(
        value,
        field="sources",
        kinds=(list,),
        max_bytes=8 * 1024 * 1024,
        max_depth=MAX_CONTAINER_DEPTH,
        max_nodes=MAX_CONTAINER_NODES,
        invalid=SourceArgumentError,
    )
    if notices is not None:
        notices.record("sources", normalized, "array")
    if (not values and not allow_empty) or len(values) > 256:
        raise SourceArgumentError("sources", "Select between 1 and 256 sources.")
    try:
        sources = _ADAPTER.validate_python(values)
    except ValidationError as error:
        location = error.errors(include_input=False)[0]["loc"]
        field = "sources" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in location
            if isinstance(part, int)
            or part
            in {
                "artifact",
                "http",
                "code",
                "kind",
                "namespace",
                "name",
                "request_id",
                "file",
                "start_line",
                "end_line",
                "workspace",
            }
        )
        raise SourceArgumentError(
            field, "Use the declared code, http or artifact selector fields."
        ) from None
    seen = set()
    for source in sources:
        key = source.model_dump_json()
        if key in seen:
            raise SourceArgumentError("sources", "Remove duplicate source selectors.")
        seen.add(key)
        if isinstance(source, CodeSource) and not 0 <= source.end_line - source.start_line < 400:
            raise ToolInputError("sources: code range must contain 1 to 400 lines")
    return sources


class SourcePreparer:
    """No traffic, authority or independent registry; call only after owner admission."""

    def __init__(self, client, resources, run_id, namespace, timeout, workspace=None):
        self.client, self.resources = client, resources
        self.run_id, self.namespace, self.timeout = run_id, namespace, timeout
        self.workspace = workspace
        self.http = HTTPExchangePublisher(client, resources.http_captures, run_id, timeout)

    async def prepare_for_finding(
        self,
        selected,
        invocation_id,
        call_id,
        *,
        max_sources=256,
        additional_sources=0,
        notices=None,
    ):
        values = parse_sources(selected, notices, allow_empty=True)
        http = [value for value in values if isinstance(value, HTTPSource)]
        if len(http) > 1:
            raise ToolInputError(
                "sources: a Finding or fact selects at most one local HTTP exchange"
            )
        other = [value.model_dump() for value in values if not isinstance(value, HTTPSource)]
        remaining = max_sources - additional_sources - len(http)
        if remaining < 0:
            raise SourceArgumentError("sources", "Selected sources exceed the claim source limit.")
        refs = (
            await self.prepare(other, invocation_id, call_id, max_sources=remaining)
            if other
            else []
        )
        return refs, http[0].request_id if http else None

    async def prepare(self, selected, invocation_id, call_id, *, max_sources=256, notices=None):
        sources = parse_sources(selected, notices)
        refs = []
        # Resolve all selectors before publishing any prepared bytes.
        prepared = []
        for index, source in enumerate(sources):
            if isinstance(source, ArtifactSource):
                try:
                    ref = self.resources.selected_versions.resolve(
                        {"namespace": source.namespace, "name": source.name}
                    )
                except ToolInputError as error:
                    raise SourceArgumentError("sources", str(error)) from None
                refs.append(ref)
            elif isinstance(source, HTTPSource):
                pending = await self.http.prepare(
                    invocation_id, [source.request_id], max_bytes=8 * 1024 * 1024
                )
                prepared.append(("http", source.request_id, pending[0][1]))
            else:
                target = ArtifactRef(
                    namespace=self.namespace,
                    name="source-proof-"
                    + hashlib.sha256(f"{invocation_id}:{call_id}:{index}".encode()).hexdigest(),
                )
                selector = source.model_dump()
                digest = sha256_digest(jcs.canonicalize(selector))
                if source.workspace:
                    if self.workspace is None:
                        raise ToolInputError("sources: workspace source capture is unavailable")
                    capture = await self.workspace.capture_source_file(source.file)
                    proof = await threads.to_thread_until_done(
                        trace_source,
                        capture,
                        source.start_line,
                        source.end_line,
                        name="selected-code-source",
                    )
                else:
                    capture = self.resources.source_evidence
                    if capture is None:
                        raise ToolInputError(
                            "sources: open a pinned source archive before selecting code"
                        )
                    proof = await capture(source.file, source.start_line, source.end_line)
                proof["selectionDigest"] = digest
                raw = await threads.to_thread_until_done(
                    jcs.canonicalize, proof, name="selected-code-encode"
                )
                if len(raw) > 1024 * 1024:
                    raise ToolInputError("sources: code proof exceeds its byte bound")
                refs.append(ArtifactRef.model_validate(proof["source"]["artifact"]))
                prepared.append(("code", target, raw))
        addresses = {(ref.namespace, ref.name, ref.revision) for ref in refs}
        if len(addresses) + len(prepared) > max_sources:
            raise SourceArgumentError("sources", "Selected sources exceed the claim source limit.")
        for kind, key, raw in prepared:
            if kind == "http":
                from contractor_runtime.strict_json import strict_json_loads

                material = strict_json_loads(await self.http.publish(invocation_id, key, raw))
                refs.append(ArtifactRef.model_validate(material["ref"]))
            else:
                saved = await write_artifact_exact(
                    self.client,
                    key,
                    data=raw,
                    media_type="application/json",
                    expected_revision=None,
                    max_bytes=1024 * 1024,
                    reject_first=lambda error: (
                        definite_write_rejection(error) and error.code != "artifact_conflict"
                    ),
                    reject_replay=lambda error: (
                        definite_write_rejection(error) and error.code != "artifact_conflict"
                    ),
                )
                refs.append(saved.artifact)
        unique = {ref.model_dump_json(): ref for ref in refs}
        for ref in unique.values():
            self.resources.selected_versions.observe(ref)
        return list(unique.values())
