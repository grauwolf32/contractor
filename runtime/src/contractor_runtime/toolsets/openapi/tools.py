"""Namespace-bound, CAS-backed OpenAPI 3 tools for one Worker allocation."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any

import yaml
from pydantic import ValidationError

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.adapters.http_proxy import ProxySubprocessLauncher
from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.core.process import ProcessOutputLimitError, run_tool_command
from contractor_runtime.probe import executable_responds
from contractor_runtime.projectfs.storage import (
    WorkspaceReader,
    WorkspaceStorageError,
)
from contractor_runtime.redaction import runtime_secrets
from contractor_runtime.strict_json import reject_json_constant, unique_json_object
from contractor_runtime.threads import to_thread_until_done
from contractor_runtime.toolsets.common.artifacts import (
    ArtifactClientFactory,
    _unconfigured_client,
)
from contractor_runtime.toolsets.common.document_session import (
    DocumentSession,
    validate_target_name,
)
from contractor_runtime.toolsets.common.factory import require_metrics, require_selected_tools
from contractor_runtime.toolsets.common.tool_base import SessionArtifactTool
from contractor_runtime.toolsets.openapi import yaml_scalars
from contractor_runtime.toolsets.openapi.models import (
    PathItem,
    RequestBody,
    Response,
    SecurityScheme,
)
from contractor_runtime.workspace import AllocationWorkspace

# The source-document bounds of Server scan preparation, so that no document
# this Toolset accepts or writes exceeds them (SCN-R253): UTF-8 bytes
# (LIM-SCN-39); depth, with the top-level value at depth 1 and every member
# name, member value and array element one deeper than its container
# (LIM-SCN-41); and nodes, counting every value and every member name
# (LIM-SCN-42). api/testdata/openapi-document-bounds.json holds the Runtime and
# internal/scanplan to the same verdicts.
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_DOCUMENT_DEPTH = 64
MAX_DOCUMENT_NODES = 100_000
DOCUMENT_BYTES_MESSAGE = "OpenAPI document exceeds the 2 MiB domain limit"
DOCUMENT_DEPTH_MESSAGE = "OpenAPI document exceeds the maximum nesting depth"
DOCUMENT_NODES_MESSAGE = "OpenAPI document exceeds the node limit"
MAX_TARGETED_RESULT_BYTES = 256 * 1024
MAX_LIST_ITEMS = 500
MAX_VACUUM_OUTPUT_BYTES = 1024 * 1024
MAX_VALIDATION_ISSUES = 100
VACUUM_TIMEOUT_SECONDS = 30
DEFAULT_TARGET_NAME = "openapi"
TARGET_MEDIA_TYPE = "application/yaml"
SOURCE_EVIDENCE_DENIED_SUFFIXES = frozenset({".json", ".yaml", ".yml", ".md", ".c4"})
OPENAPI_30_COMPONENT_SECTIONS = frozenset(
    {
        "schemas",
        "responses",
        "parameters",
        "examples",
        "requestBodies",
        "headers",
        "securitySchemes",
        "links",
        "callbacks",
    }
)
ALLOWED_COMPONENT_SECTIONS = OPENAPI_30_COMPONENT_SECTIONS | {"pathItems"}


BASE_DOCUMENT: dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {"title": "", "description": "", "version": "1.0.0"},
    "paths": {},
    "components": {section: {} for section in sorted(OPENAPI_30_COMPONENT_SECTIONS)},
}


class OpenAPIToolsetFactory:
    ref = "openapi@1"
    workspace_access = "read"
    requires_workspace = True
    exported_tools = frozenset(
        {
            "load_openapi",
            "initialize_openapi",
            "get_openapi_info",
            "set_openapi_info",
            "list_openapi_servers",
            "set_openapi_servers",
            "list_openapi_tags",
            "set_openapi_tags",
            "list_openapi_paths",
            "get_openapi_path",
            "upsert_openapi_path",
            "remove_openapi_path",
            "list_openapi_components",
            "get_openapi_component",
            "upsert_openapi_component",
            "remove_openapi_component",
            "read_openapi_document",
            "validate_openapi",
        }
    )
    infrastructure_channels = MappingProxyType(
        {"validate_openapi": frozenset({"runtime-subprocess-launcher"})}
    )

    def __init__(self, client_factory: ArtifactClientFactory | None = None) -> None:
        self._client_factory = client_factory or _unconfigured_client

    async def probe(self) -> frozenset[str]:
        available = set(self.exported_tools)
        if not await executable_responds("vacuum", ("version",)):
            available.discard("validate_openapi")
        return frozenset(available)

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
        project_workspace: WorkspaceReader | None = None,
    ) -> Mapping[str, Any]:
        del run_id, workspace
        require_selected_tools(selected, self.exported_tools)
        if project_workspace is None:
            raise ValueError("openapi@1 requires a project workspace")
        metrics = require_metrics(state, "openapi@1")
        client = self._client_factory(allocation_id, runtime_settings)
        launcher = adapter_handles.tool_subprocess
        if launcher is not None and not isinstance(launcher, ProxySubprocessLauncher):
            raise TypeError("openapi@1 received an invalid subprocess handle")
        session = _OpenAPISession(
            client,
            namespace,
            launcher,
            project_workspace=project_workspace,
        )
        secrets = runtime_secrets(runtime_settings)
        constructors = {
            "load_openapi": LoadOpenAPITool,
            "initialize_openapi": InitializeOpenAPITool,
            "get_openapi_info": GetOpenAPIInfoTool,
            "set_openapi_info": SetOpenAPIInfoTool,
            "list_openapi_servers": ListOpenAPIServersTool,
            "set_openapi_servers": SetOpenAPIServersTool,
            "list_openapi_tags": ListOpenAPITagsTool,
            "set_openapi_tags": SetOpenAPITagsTool,
            "list_openapi_paths": ListOpenAPIPathsTool,
            "get_openapi_path": GetOpenAPIPathTool,
            "upsert_openapi_path": UpsertOpenAPIPathTool,
            "remove_openapi_path": RemoveOpenAPIPathTool,
            "list_openapi_components": ListOpenAPIComponentsTool,
            "get_openapi_component": GetOpenAPIComponentTool,
            "upsert_openapi_component": UpsertOpenAPIComponentTool,
            "remove_openapi_component": RemoveOpenAPIComponentTool,
            "read_openapi_document": ReadOpenAPIDocumentTool,
            "validate_openapi": ValidateOpenAPITool,
        }
        return {name: constructors[name](session, client, metrics, secrets) for name in selected}


class _OpenAPISession(DocumentSession[dict[str, Any]]):
    media_type = TARGET_MEDIA_TYPE
    seed_media_types = frozenset({"application/yaml", "application/json"})
    max_bytes = MAX_DOCUMENT_BYTES
    reload_tool = "load_openapi"
    read_tool = "read_openapi_document"
    seed_revision_message = (
        "an OpenAPI seed outside the Worker namespace requires an exact revision"
    )
    seed_too_large_message = DOCUMENT_BYTES_MESSAGE
    seed_media_type_message = "OpenAPI seed media type must be application/yaml or application/json"
    not_loaded_message = "load_openapi or initialize_openapi must be called first"

    def __init__(
        self,
        client: ArtifactClient,
        namespace: str,
        launcher: ProxySubprocessLauncher | None,
        *,
        project_workspace: WorkspaceReader,
    ) -> None:
        super().__init__(client, namespace)
        self._project_workspace = project_workspace
        self._launcher = launcher

    def _decode_seed(self, data: bytes, media_type: str) -> dict[str, Any]:
        # _parse_document already checked the bounds and the JSON value types.
        document = _parse_document(data, media_type)
        _validate_structure(document, require_provenance=False)
        return document

    def _serialize(self, document: dict[str, Any]) -> bytes:
        return _dump_document(document)

    def _document_state(
        self,
        artifact: ArtifactRef,
        target_name: str,
        size: int,
        *,
        changed: bool,
        copied: bool,
    ) -> dict[str, Any]:
        del size
        return _document_state(artifact, target_name, changed=changed, copied=copied)

    async def initialize(
        self,
        *,
        title: str,
        version: str,
        description: str,
        target_name: str,
        expected_revision: str | None,
    ) -> dict[str, Any]:
        _require_nonempty("title", title)
        _require_nonempty("version", version)
        validate_target_name(target_name)
        document = copy.deepcopy(BASE_DOCUMENT)
        document["info"] = {"title": title, "description": description, "version": version}
        async with self._lock:
            data = await self._prepare(_prepare_document, document)
            written = await self._write(
                data, target_name=target_name, expected_revision=expected_revision
            )
            self._remember(document, target_name, written.artifact)
            return _document_state(written.artifact, target_name, changed=True, copied=False)

    async def get_info(self) -> dict[str, Any]:
        async with self._lock:
            document, artifact = self._require_document()
            value = await self._prepare(_targeted_copy, document["info"])
            return {"artifact": artifact.model_dump(by_alias=True), "info": value}

    async def set_info(
        self,
        *,
        title: str,
        version: str | None,
        description: str | None,
        framework: str | None,
        code_language: str | None,
    ) -> dict[str, Any]:
        _require_nonempty("title", title)

        def modify(document: dict[str, Any]) -> None:
            info = document.setdefault("info", {})
            info["title"] = title
            if version is not None:
                _require_nonempty("version", version)
                info["version"] = version
            if description is not None:
                info["description"] = description
            if framework is not None:
                info["x-framework"] = framework
            if code_language is not None:
                info["x-code-language"] = code_language

        return await self._mutate(modify, operation="set_info")

    async def list_servers(self) -> dict[str, Any]:
        async with self._lock:
            document, artifact = self._require_document()
            servers = await self._prepare(_targeted_copy, document.get("servers", []))
            return {"artifact": artifact.model_dump(by_alias=True), "servers": servers}

    async def set_servers(self, servers: list[dict[str, Any]]) -> dict[str, Any]:
        async with self._lock:
            normalized = await self._prepare(_checked_servers, servers)
            return await self._mutate_locked(
                lambda document: document.__setitem__("servers", normalized),
                operation="set_servers",
            )

    async def list_tags(self) -> dict[str, Any]:
        async with self._lock:
            document, artifact = self._require_document()
            tags = await self._prepare(_targeted_copy, document.get("tags", []))
            return {"artifact": artifact.model_dump(by_alias=True), "tags": tags}

    async def set_tags(self, tags: list[dict[str, Any]]) -> dict[str, Any]:
        async with self._lock:
            normalized = await self._prepare(_checked_tags, tags)
            return await self._mutate_locked(
                lambda document: document.__setitem__("tags", normalized),
                operation="set_tags",
            )

    async def list_paths(self) -> dict[str, Any]:
        async with self._lock:
            document, artifact = self._require_document()
            return await self._prepare(
                _bounded_name_listing,
                {"artifact": artifact.model_dump(by_alias=True)},
                "paths",
                document["paths"],
            )

    async def get_path(self, path: str) -> dict[str, Any]:
        normalized = _validate_api_path(path)
        async with self._lock:
            document, artifact = self._require_document()
            if normalized not in document["paths"]:
                raise ToolInputError("OpenAPI path is absent", code="openapi_path_not_found")
            value = await self._prepare(_targeted_copy, document["paths"][normalized])
            return {
                "artifact": artifact.model_dump(by_alias=True),
                "path": normalized,
                "pathItem": value,
            }

    async def upsert_path(
        self, path: str, path_item: dict[str, Any], evidence_files: list[str]
    ) -> dict[str, Any]:
        normalized = _validate_api_path(path)
        if not isinstance(path_item, dict):
            raise ToolInputError("path_item must be an object")
        evidence = await self._validate_patch(path_item, evidence_files)

        def modify(document: dict[str, Any]) -> None:
            # The supplied object is a merge patch, not a standalone Path
            # Item. The merged candidate is validated with the whole document.
            current = document["paths"].get(normalized, {})
            document["paths"][normalized] = _deep_merge(
                current, {**path_item, "x-path-files": evidence}
            )

        result = await self._mutate(modify, operation="upsert_path")
        result["path"] = normalized
        return result

    async def remove_path(self, path: str) -> dict[str, Any]:
        normalized = _validate_api_path(path)

        def modify(document: dict[str, Any]) -> None:
            if normalized not in document["paths"]:
                raise ToolInputError("OpenAPI path is absent", code="openapi_path_not_found")
            del document["paths"][normalized]

        result = await self._mutate(modify, operation="remove_path")
        result["path"] = normalized
        return result

    async def list_components(self, section: str) -> dict[str, Any]:
        normalized = _validate_component_section(section)
        async with self._lock:
            document, artifact = self._require_document()
            return await self._prepare(
                _bounded_name_listing,
                {"artifact": artifact.model_dump(by_alias=True), "section": normalized},
                "components",
                document.get("components", {}).get(normalized, {}),
            )

    async def get_component(self, section: str, name: str) -> dict[str, Any]:
        normalized = _validate_component_section(section)
        _validate_component_name(name)
        async with self._lock:
            document, artifact = self._require_document()
            components = document.get("components", {}).get(normalized, {})
            if name not in components:
                raise ToolInputError(
                    "OpenAPI component is absent", code="openapi_component_not_found"
                )
            value = await self._prepare(_targeted_copy, components[name])
            return {
                "artifact": artifact.model_dump(by_alias=True),
                "section": normalized,
                "name": name,
                "component": value,
            }

    async def upsert_component(
        self,
        section: str,
        name: str,
        component: dict[str, Any],
        evidence_files: list[str],
    ) -> dict[str, Any]:
        normalized = _validate_component_section(section)
        _validate_component_name(name)
        if not isinstance(component, dict):
            raise ToolInputError("component must be an object")
        evidence = await self._validate_patch(component, evidence_files)

        def modify(document: dict[str, Any]) -> None:
            update = dict(component)
            # A 3.0 Reference Object cannot carry extension siblings. The
            # evidence is checked above, but the alias must stay a pure $ref.
            if not _reference_only_component(update, document["openapi"]):
                update["x-component-files"] = evidence
            # The merged candidate is validated with the whole document.
            values = document.setdefault("components", {}).setdefault(normalized, {})
            values[name] = _deep_merge(values.get(name, {}), update)

        result = await self._mutate(modify, operation="upsert_component")
        result.update({"section": normalized, "name": name})
        return result

    async def remove_component(self, section: str, name: str) -> dict[str, Any]:
        normalized = _validate_component_section(section)
        _validate_component_name(name)

        def modify(document: dict[str, Any]) -> None:
            values = document.setdefault("components", {}).setdefault(normalized, {})
            if name not in values:
                raise ToolInputError(
                    "OpenAPI component is absent", code="openapi_component_not_found"
                )
            del values[name]

        result = await self._mutate(modify, operation="remove_component")
        result.update({"section": normalized, "name": name})
        return result

    async def read_document(self) -> dict[str, Any]:
        async with self._lock:
            document, artifact = self._require_document()
            rendered = await self._prepare(_render_full_read, document)
            return {
                "artifact": artifact.model_dump(by_alias=True),
                "mediaType": TARGET_MEDIA_TYPE,
                "document": rendered,
            }

    async def validate(self) -> dict[str, Any]:
        project_evidence_paths = await self._project_evidence_paths()
        async with self._lock:
            # close() cancels this task from its preparation through Vacuum.
            return await self._validating(self._validate_current(project_evidence_paths))

    async def _validate_current(
        self, project_evidence_paths: frozenset[str] | None
    ) -> dict[str, Any]:
        document, artifact = self._require_document()
        rendered, structural_errors = await self._prepare(
            self._validation_input, document, project_evidence_paths
        )
        if self._launcher is None:
            vacuum = await _run_vacuum(rendered)
        else:
            vacuum = await _run_vacuum(rendered, self._launcher)
        valid = (
            not structural_errors
            and vacuum["available"]
            and vacuum["executionError"] is None
            and not vacuum["issues"]
        )
        return {
            "artifact": artifact.model_dump(by_alias=True),
            "valid": valid,
            "structuralErrors": structural_errors,
            "validator": "vacuum",
            "validatorAvailable": vacuum["available"],
            "validatorExecutionError": vacuum["executionError"],
            "issues": vacuum["issues"],
            "issuesTruncated": vacuum["truncated"],
        }

    def _validation_input(
        self, document: dict[str, Any], project_evidence_paths: frozenset[str] | None
    ) -> tuple[str, list[str]]:
        # The document bounds fail the call before Vacuum runs; only the
        # structure checks below are reported beside its findings.
        rendered = _dump_document(document).decode("utf-8")
        _validate_json_tree(document)
        structural_errors: list[str] = []
        try:
            _validate_structure(document, require_provenance=True)
            self._validate_current_provenance(document, project_evidence_paths)
        except ValueError as error:
            structural_errors.append(str(error))
        return rendered, structural_errors

    async def _mutate(
        self, modifier: Callable[[dict[str, Any]], None], *, operation: str
    ) -> dict[str, Any]:
        async with self._lock:
            return await self._mutate_locked(modifier, operation=operation)

    async def _mutate_locked(
        self, modifier: Callable[[dict[str, Any]], None], *, operation: str
    ) -> dict[str, Any]:
        document, artifact = self._require_document()
        prepared = await self._prepare(_prepare_mutation, document, modifier)
        if prepared is None:
            return _document_state(
                artifact, self._target_name or DEFAULT_TARGET_NAME, changed=False, copied=False
            ) | {"operation": operation}
        working, data = prepared
        target_name, revision = self._require_target()
        written = await self._write(data, target_name=target_name, expected_revision=revision)
        self._remember(working, target_name, written.artifact)
        return _document_state(written.artifact, target_name, changed=True, copied=False) | {
            "operation": operation
        }

    async def _validate_patch(self, patch: dict[str, Any], evidence_files: list[str]) -> list[str]:
        """Check a merge patch, then its source evidence; return the evidence paths."""

        async with self._lock:
            await self._prepare(_validate_json_tree, patch)
        project_evidence_paths = await self._project_evidence_paths()
        async with self._lock:
            return await self._prepare(
                self._validate_evidence_from_paths, evidence_files, project_evidence_paths
            )

    async def _project_evidence_paths(self) -> frozenset[str]:
        try:
            snapshot = await self._project_workspace.snapshot()
        except WorkspaceStorageError:
            raise ValueError("project workspace is unavailable for OpenAPI evidence") from None
        return frozenset([file.path for file in snapshot.files] + list(snapshot.binary_paths))

    def _validate_evidence_from_paths(
        self,
        evidence_files: list[str],
        project_evidence_paths: frozenset[str],
    ) -> list[str]:
        if not isinstance(evidence_files, list) or not evidence_files:
            raise ToolInputError("evidence_files must be a JSON array of at least one source path")
        if len(evidence_files) > 100:
            raise ToolInputError("source evidence exceeds the 100-file limit")
        result: list[str] = []
        seen: set[str] = set()
        for raw in evidence_files:
            normalized = _validate_relative_source_path(raw)
            if PurePosixPath(normalized).suffix.lower() in SOURCE_EVIDENCE_DENIED_SUFFIXES:
                raise ToolInputError(
                    "OpenAPI provenance must reference implementation source files"
                )
            if normalized in seen:
                continue
            if normalized not in project_evidence_paths:
                raise ToolInputError(f"source evidence file does not exist: {normalized}")
            seen.add(normalized)
            result.append(normalized)
        return result

    def _validate_current_provenance(
        self,
        document: dict[str, Any],
        project_evidence_paths: frozenset[str],
    ) -> None:
        for path_item in document["paths"].values():
            self._validate_evidence_from_paths(path_item["x-path-files"], project_evidence_paths)
        for section, values in document.get("components", {}).items():
            if section.startswith("x-"):
                continue
            for component in values.values():
                if _reference_only_component(component, document["openapi"]):
                    continue
                self._validate_evidence_from_paths(
                    component["x-component-files"], project_evidence_paths
                )


_BaseOpenAPITool = SessionArtifactTool[_OpenAPISession]


def _artifact_metric(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "artifact": result.get("artifact"),
        "changed": result.get("changed"),
        "operation": result.get("operation"),
    }


class LoadOpenAPITool(_BaseOpenAPITool):
    name = "load_openapi"
    description = """Load an OpenAPI YAML or JSON artifact into the current document session.

    Copies the source into the Worker's namespace or resumes the same binding.
    Call this or initialize_openapi before other OpenAPI operations.

    Args:
        namespace: Source artifact namespace.
        name: Source artifact binding name.
        revision: Source revision; required for a source outside the Worker's namespace.
        target_name: Destination binding in the Worker's namespace; defaults to "openapi".
        expected_revision: Current destination revision when copying over an existing
            binding; omit to create it. Resuming the same binding uses its revision.

    Returns:
        Exact current artifact reference, mediaType, targetName and changed/copied flags.
    """

    async def __call__(
        self,
        namespace: str,
        name: str,
        revision: str | None = None,
        target_name: str = DEFAULT_TARGET_NAME,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        arguments = {
            "namespace": namespace,
            "name": name,
            "revision": revision,
            "target_name": target_name,
            "expected_revision": expected_revision,
        }
        return await self._call(
            arguments,
            self._session.load(
                namespace=namespace,
                name=name,
                revision=revision,
                target_name=target_name,
                expected_revision=expected_revision,
            ),
            _artifact_metric,
        )


class InitializeOpenAPITool(_BaseOpenAPITool):
    name = "initialize_openapi"
    description = """Create a minimal OpenAPI 3.0.3 YAML document in the Worker's namespace.

    Initializes the current document session for subsequent OpenAPI operations.

    Args:
        title: Non-empty API title.
        version: Non-empty API version; defaults to "1.0.0".
        description: API description; defaults to an empty string.
        target_name: Destination artifact binding; defaults to "openapi".
        expected_revision: Current destination revision to replace; omit only to
            create a new binding.

    Returns:
        Exact saved artifact reference, mediaType, targetName and changed/copied flags.
    """

    async def __call__(
        self,
        title: str,
        version: str = "1.0.0",
        description: str = "",
        target_name: str = DEFAULT_TARGET_NAME,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        arguments = {
            "content": {"title": title, "version": version, "description": description},
            "target_name": target_name,
            "expected_revision": expected_revision,
        }
        return await self._call(
            arguments,
            self._session.initialize(
                title=title,
                version=version,
                description=description,
                target_name=target_name,
                expected_revision=expected_revision,
            ),
            _artifact_metric,
        )


class GetOpenAPIInfoTool(_BaseOpenAPITool):
    name = "get_openapi_info"
    description = """Read the current OpenAPI info object.

    Load or initialize the document first.

    Returns:
        The info object and exact artifact reference.
    """

    async def __call__(self) -> dict[str, Any]:
        return await self._call(
            {},
            self._session.get_info(),
            lambda result: {"artifact": result["artifact"]},
        )


class SetOpenAPIInfoTool(_BaseOpenAPITool):
    name = "set_openapi_info"
    description = """Update the current OpenAPI info object and save with a revision check.

    Load or initialize the document first. Omitted optional fields are preserved.

    Args:
        title: Non-empty API title.
        version: New non-empty API version; omit to preserve it.
        description: New API description; omit to preserve it, or use empty to clear.
        framework: New x-framework value; omit to preserve it.
        code_language: New x-code-language value; omit to preserve it.

    Returns:
        Exact saved artifact metadata, operation and changed flag.
    """

    async def __call__(
        self,
        title: str,
        version: str | None = None,
        description: str | None = None,
        framework: str | None = None,
        code_language: str | None = None,
    ) -> dict[str, Any]:
        arguments = {
            "content": {
                "title": title,
                "version": version,
                "description": description,
                "framework": framework,
                "code_language": code_language,
            }
        }
        return await self._call(
            arguments,
            self._session.set_info(
                title=title,
                version=version,
                description=description,
                framework=framework,
                code_language=code_language,
            ),
            _artifact_metric,
        )


class ListOpenAPIServersTool(_BaseOpenAPITool):
    name = "list_openapi_servers"
    description = """Read the current OpenAPI server list.

    Load or initialize the document first.

    Returns:
        Server objects and the exact artifact reference.
    """

    async def __call__(self) -> dict[str, Any]:
        return await self._call(
            {},
            self._session.list_servers(),
            lambda result: {"artifact": result["artifact"], "count": len(result["servers"])},
        )


class SetOpenAPIServersTool(_BaseOpenAPITool):
    name = "set_openapi_servers"
    description = """Replace the entire OpenAPI server list and save with a revision check.

    Load or initialize the document first.

    Args:
        servers: Up to 100 server objects, each with a unique non-empty url and
            optional description. An empty list removes all servers.

    Returns:
        Exact saved artifact metadata, operation and changed flag.
    """

    async def __call__(self, servers: list[dict[str, Any]]) -> dict[str, Any]:
        return await self._call(
            {"content": servers}, self._session.set_servers(servers), _artifact_metric
        )


class ListOpenAPITagsTool(_BaseOpenAPITool):
    name = "list_openapi_tags"
    description = """Read the current top-level OpenAPI tag declarations.

    Load or initialize the document first.

    Returns:
        Tag objects and the exact artifact reference.
    """

    async def __call__(self) -> dict[str, Any]:
        return await self._call(
            {},
            self._session.list_tags(),
            lambda result: {"artifact": result["artifact"], "count": len(result["tags"])},
        )


class SetOpenAPITagsTool(_BaseOpenAPITool):
    name = "set_openapi_tags"
    description = """Replace all top-level OpenAPI tag declarations and save with a revision check.

    Load or initialize the document first.

    Args:
        tags: Up to 100 objects with unique non-empty name, optional description,
            externalDocs and x-* extensions. An empty list removes all declarations.

    Returns:
        Exact saved artifact metadata, operation and changed flag.
    """

    async def __call__(self, tags: list[dict[str, Any]]) -> dict[str, Any]:
        return await self._call({"content": tags}, self._session.set_tags(tags), _artifact_metric)


class ListOpenAPIPathsTool(_BaseOpenAPITool):
    name = "list_openapi_paths"
    description = """List path keys in the current OpenAPI document.

    Load or initialize the document first. Use get_openapi_path for one definition.

    Returns:
        Sorted path strings, total, truncated and the exact artifact reference.
        At most 500 paths are listed, fewer when long paths reach the output
        limit; truncated then marks the omitted suffix.
    """

    async def __call__(self) -> dict[str, Any]:
        return await self._call(
            {},
            self._session.list_paths(),
            lambda result: {
                "artifact": result["artifact"],
                "total": result["total"],
                "returned": len(result["paths"]),
                "truncated": result["truncated"],
            },
        )


class GetOpenAPIPathTool(_BaseOpenAPITool):
    name = "get_openapi_path"
    description = """Read one Path Item from the current OpenAPI document.

    Load or initialize the document first.

    Args:
        path: Exact API path key, such as "/users/{id}".

    Returns:
        path, pathItem and the exact artifact reference. A missing path fails.
    """

    async def __call__(self, path: str) -> dict[str, Any]:
        return await self._call(
            {"path": path},
            self._session.get_path(path),
            lambda result: {"artifact": result["artifact"], "path": result["path"]},
        )


class UpsertOpenAPIPathTool(_BaseOpenAPITool):
    name = "upsert_openapi_path"
    description = """Create or merge an OpenAPI Path Item and save with a revision check.

    Load or initialize the document first. Fields merge recursively; supplied
    lists and reference objects replace existing values. Local $ref values must
    resolve. An OpenAPI 3.0 Path Item cannot itself be a $ref; inline its
    operations instead. Implementation-source evidence is required.

    Args:
        path: API path key, such as "/users/{id}".
        path_item: Structured Path Item object with operations; pass an object,
            not a JSON string.
        evidence_files: Non-empty list of existing relative implementation source paths
            in the project workspace. Files ending in
            .json, .yaml, .yml, .md or .c4 are rejected.

    Returns:
        Exact saved artifact metadata, path, operation and changed flag.
    """

    async def __call__(
        self, path: str, path_item: dict[str, Any], evidence_files: list[str]
    ) -> dict[str, Any]:
        return await self._call(
            {"path": path, "content": path_item, "evidence_files": evidence_files},
            self._session.upsert_path(path, path_item, evidence_files),
            _artifact_metric,
        )


class RemoveOpenAPIPathTool(_BaseOpenAPITool):
    name = "remove_openapi_path"
    description = """Remove an existing OpenAPI path and save with a revision check.

    Load or initialize the document first.

    Args:
        path: Exact API path key to remove, such as "/users/{id}".

    Returns:
        Exact saved artifact metadata, path, operation and changed flag.
    """

    async def __call__(self, path: str) -> dict[str, Any]:
        return await self._call({"path": path}, self._session.remove_path(path), _artifact_metric)


class ListOpenAPIComponentsTool(_BaseOpenAPITool):
    name = "list_openapi_components"
    description = """List component names in a section of the current OpenAPI document.

    Load or initialize the document first.

    Args:
        section: schemas, responses, parameters, examples, requestBodies, headers,
            securitySchemes, links, callbacks, or pathItems for OpenAPI 3.1.

    Returns:
        Sorted component names, section, total, truncated and exact artifact reference.
        At most 500 names are listed, fewer when long names reach the output
        limit; truncated then marks the omitted suffix.
    """

    async def __call__(self, section: str) -> dict[str, Any]:
        return await self._call(
            {"section": section},
            self._session.list_components(section),
            lambda result: {
                "artifact": result["artifact"],
                "section": result["section"],
                "total": result["total"],
                "returned": len(result["components"]),
                "truncated": result["truncated"],
            },
        )


class GetOpenAPIComponentTool(_BaseOpenAPITool):
    name = "get_openapi_component"
    description = """Read one named component from the current OpenAPI document.

    Load or initialize the document first.

    Args:
        section: schemas, responses, parameters, examples, requestBodies, headers,
            securitySchemes, links, callbacks, or pathItems for OpenAPI 3.1.
        name: Exact component name within the section.

    Returns:
        The component object, section, name and exact artifact reference.
    """

    async def __call__(self, section: str, name: str) -> dict[str, Any]:
        return await self._call(
            {"section": section, "name": name},
            self._session.get_component(section, name),
            lambda result: {
                "artifact": result["artifact"],
                "section": result["section"],
                "name": result["name"],
            },
        )


class UpsertOpenAPIComponentTool(_BaseOpenAPITool):
    name = "upsert_openapi_component"
    description = """Create or merge an OpenAPI component and save with a revision check.

    Load or initialize the document first. Fields merge recursively; supplied
    lists and reference objects replace existing values. Local $ref values must
    resolve. Implementation-source evidence is required.

    Args:
        section: schemas, responses, parameters, examples, requestBodies, headers,
            securitySchemes, links, callbacks, or pathItems for OpenAPI 3.1.
        name: Component name within the section.
        component: Structured component object; pass an object, not a JSON string.
        evidence_files: Non-empty list of existing relative implementation source paths
            in the project workspace. Files ending in
            .json, .yaml, .yml, .md or .c4 are rejected.

    Returns:
        Exact saved artifact metadata, section, name, operation and changed flag.
    """

    async def __call__(
        self,
        section: str,
        name: str,
        component: dict[str, Any],
        evidence_files: list[str],
    ) -> dict[str, Any]:
        return await self._call(
            {
                "section": section,
                "name": name,
                "content": component,
                "evidence_files": evidence_files,
            },
            self._session.upsert_component(section, name, component, evidence_files),
            _artifact_metric,
        )


class RemoveOpenAPIComponentTool(_BaseOpenAPITool):
    name = "remove_openapi_component"
    description = """Remove an existing OpenAPI component and save with a revision check.

    Load or initialize the document first. Remaining local references must resolve.

    Args:
        section: schemas, responses, parameters, examples, requestBodies, headers,
            securitySchemes, links, callbacks, or pathItems for OpenAPI 3.1.
        name: Exact component name to remove.

    Returns:
        Exact saved artifact metadata, section, name, operation and changed flag.
    """

    async def __call__(self, section: str, name: str) -> dict[str, Any]:
        return await self._call(
            {"section": section, "name": name},
            self._session.remove_component(section, name),
            _artifact_metric,
        )


class ReadOpenAPIDocumentTool(_BaseOpenAPITool):
    name = "read_openapi_document"
    description = """Read the full canonical YAML of the current OpenAPI document for global review.

    Load or initialize the document first. Prefer targeted list/get tools for
    individual paths or components; full reads have a model output size limit.

    Returns:
        Canonical YAML in document, mediaType and the exact artifact reference.
    """

    async def __call__(self) -> dict[str, Any]:
        return await self._call(
            {},
            self._session.read_document(),
            lambda result: {
                "artifact": result["artifact"],
                "mediaType": result["mediaType"],
                "utf8Bytes": len(result["document"].encode("utf-8")),
            },
        )


class ValidateOpenAPITool(_BaseOpenAPITool):
    name = "validate_openapi"
    description = """Validate the current OpenAPI structure, local references and source provenance.

    Load or initialize the document first. Also runs Vacuum to check serious
    lint findings. Validation does not change the document.

    Returns:
        Exact artifact reference, valid, structuralErrors, bounded issues,
        issuesTruncated, validatorAvailable and validatorExecutionError.
        valid is false when Vacuum is unavailable or fails.
    """

    async def __call__(self) -> dict[str, Any]:
        return await self._call(
            {},
            self._session.validate(),
            lambda result: {
                "artifact": result["artifact"],
                "valid": result["valid"],
                "structuralErrors": len(result["structuralErrors"]),
                "validatorAvailable": result["validatorAvailable"],
                "validatorExecutionError": result["validatorExecutionError"] is not None,
                "issues": len(result["issues"]),
                "issuesTruncated": result["issuesTruncated"],
            },
        )


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader that reads an OpenAPI document in one bounded pass.

    Composition rejects anchors, aliases and tags other than the core tags,
    and stops at the first node beyond the depth or node bound, before PyYAML
    composes a deeper or larger tree. A node is one deeper than the
    collections open around it, so a mapping key is a node at the depth of its
    value, as in _validate_json_tree; the count covers every document of the
    stream.

    Scalars resolve and keys read under the rule Server scan preparation and
    the Audit inventory apply (SCN-R124, yaml_scalars), not under PyYAML's
    YAML 1.1 resolvers: ``yes`` and ``1:30`` are strings, ``0o17`` and ``1e3``
    are numbers, a plain date such as ``version: 2023-01-01`` stays its source
    text, and a merge key ``<<`` is rejected.
    """

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._open_collections = 0
        self._nodes = 0
        # Nodes with an explicit tag. A plain key is the string of its source
        # text, an explicitly tagged key must be !!str.
        self._tagged: set[int] = set()

    def compose_node(self, parent: yaml.Node | None, index: Any) -> yaml.Node:
        event = self.peek_event()
        if getattr(event, "anchor", None) is not None:
            raise ToolInputError("OpenAPI YAML aliases and anchors are not supported")
        tag = getattr(event, "tag", None)
        if tag is not None and tag not in _EXPLICIT_TAGS.get(type(event), frozenset()):
            raise ToolInputError(
                "OpenAPI YAML tags other than !!str, !!int, !!float, !!bool, !!null, "
                "!!map and !!seq are not supported"
            )
        self._nodes += 1
        if self._open_collections + 1 > MAX_DOCUMENT_DEPTH:
            raise ToolInputError(DOCUMENT_DEPTH_MESSAGE)
        if self._nodes > MAX_DOCUMENT_NODES:
            raise ToolInputError(DOCUMENT_NODES_MESSAGE)
        self._open_collections += 1
        try:
            node = super().compose_node(parent, index)
        finally:
            self._open_collections -= 1
        if tag is not None:
            self._tagged.add(id(node))
        return node

    def resolve(self, kind: type[yaml.Node], value: Any, implicit: Any) -> str:
        if kind is yaml.ScalarNode and implicit[0]:
            return yaml_scalars.resolve_plain(value)
        return str(super().resolve(kind, value, implicit))

    def process_directives(self) -> Any:
        directives = super().process_directives()
        # go-yaml reads only YAML 1.1 version directives.
        if self.yaml_version not in (None, (1, 1)):
            raise ToolInputError(
                "OpenAPI YAML version directives other than %YAML 1.1 are not supported"
            )
        return directives


# The double-quoted escapes of go-yaml: PyYAML's set without \/ and with \'.
_StrictSafeLoader.ESCAPE_REPLACEMENTS = {
    **{
        escape: value
        for escape, value in yaml.SafeLoader.ESCAPE_REPLACEMENTS.items()
        if escape != "/"
    },
    "'": "'",
}

_EXPLICIT_TAGS: Mapping[type[yaml.Event], frozenset[str]] = {
    yaml.ScalarEvent: yaml_scalars.SCALAR_TAGS,
    yaml.MappingStartEvent: frozenset({yaml_scalars.MAP_TAG}),
    yaml.SequenceStartEvent: frozenset({yaml_scalars.SEQ_TAG}),
}


class _OpenAPIDumper(yaml.SafeDumper):
    """SafeDumper that writes only YAML every reader reads back unchanged.

    A string is written plain only when the rule of yaml_scalars and YAML 1.1
    both read it as that string; every other string is quoted (SCN-R124).
    """

    def ignore_aliases(self, data: Any) -> bool:
        del data
        return True

    def resolve(self, kind: type[yaml.Node], value: Any, implicit: Any) -> str:
        if kind is yaml.ScalarNode and implicit[0]:
            return yaml_scalars.writer_tag(value)
        return str(super().resolve(kind, value, implicit))


def _construct_scalar(loader: _StrictSafeLoader, node: yaml.ScalarNode) -> Any:
    del loader
    return yaml_scalars.scalar_value(node.tag, node.value)


def _reject_tag(loader: _StrictSafeLoader, node: yaml.Node) -> Any:
    del loader, node
    raise ToolInputError("OpenAPI YAML tags other than the core tags are not supported")


def _construct_unique_mapping(
    loader: _StrictSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        if not isinstance(key_node, yaml.ScalarNode):
            raise ToolInputError("OpenAPI YAML mapping keys must be scalar strings")
        if id(key_node) in loader._tagged:
            if key_node.tag != yaml_scalars.STR_TAG:
                raise ToolInputError("OpenAPI YAML mapping keys must be strings")
        elif key_node.tag == yaml_scalars.MERGE_TAG:
            raise ToolInputError("OpenAPI YAML merge keys (<<) are not supported")
        # JSON object keys are strings: a plain key such as 200:, 1.5:, on: or
        # 0o17: is the string of its source text.
        key = key_node.value
        if key in result:
            raise ToolInputError(f"OpenAPI YAML contains duplicate key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


# Only the core tags construct values; resolve() never yields another tag
# without composition having rejected it first.
_StrictSafeLoader.yaml_constructors = {
    **dict.fromkeys(yaml_scalars.SCALAR_TAGS, _construct_scalar),
    yaml_scalars.NUMERIC_STRING_TAG: _construct_scalar,
    yaml_scalars.MERGE_TAG: _construct_scalar,
    yaml_scalars.MAP_TAG: _construct_unique_mapping,
    yaml_scalars.SEQ_TAG: yaml.SafeLoader.yaml_constructors[yaml_scalars.SEQ_TAG],
    None: _reject_tag,
}
_StrictSafeLoader.yaml_multi_constructors = {}


def _parse_document(data: bytes, media_type: str = TARGET_MEDIA_TYPE) -> dict[str, Any]:
    """Parse document bytes once and check its bounds and JSON value types."""

    if len(data) > MAX_DOCUMENT_BYTES:
        raise ToolInputError(DOCUMENT_BYTES_MESSAGE)
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ToolInputError("OpenAPI document must be valid UTF-8") from error
    if media_type == "application/json":
        try:
            document = json.loads(
                text,
                object_pairs_hook=unique_json_object,
                parse_constant=reject_json_constant,
                parse_int=_json_number,
                parse_float=_json_number,
            )
        except ToolInputError:
            raise
        except RecursionError as error:
            raise ToolInputError(DOCUMENT_DEPTH_MESSAGE) from error
        except ValueError as error:
            raise ToolInputError("OpenAPI document is not valid JSON") from error
    else:
        document = _parse_yaml_document(text)
    _validate_json_tree(document)
    if not isinstance(document, dict):
        raise ToolInputError("OpenAPI artifact must contain exactly one object document")
    return document


def _json_number(text: str) -> int | float:
    # Check the source spelling before float rounding, like both Server readers.
    value = yaml_scalars.document_float(text)
    if value is None:
        raise ToolInputError(yaml_scalars.NUMBER_MESSAGE)
    # Keep integer tokens integral in YAML output, preserving the sign of -0.
    if "." not in text and "e" not in text.lower() and text != "-0":
        return int(text)
    return value


def _parse_yaml_document(text: str) -> Any:
    try:
        # The reader rejects non-printable characters while it is created.
        loader = _StrictSafeLoader(text)
    except yaml.YAMLError as error:
        raise ToolInputError("OpenAPI document is not valid YAML or JSON") from error
    try:
        # Compose the whole stream under the bounds before constructing any
        # value, so a later document's anchor or bound fails first.
        nodes = []
        while loader.check_node():
            nodes.append(loader.get_node())
        documents = [loader.construct_document(node) for node in nodes]
    except ToolInputError:
        raise
    except (yaml.YAMLError, ValueError) as error:
        # PyYAML raises ValueError for a \U escape beyond U+10FFFF.
        raise ToolInputError("OpenAPI document is not valid YAML or JSON") from error
    finally:
        loader.dispose()
    if len(documents) != 1:
        raise ToolInputError("OpenAPI artifact must contain exactly one object document")
    return documents[0]


def _validate_json_tree(value: Any) -> None:
    """Check the document bounds and the JSON value types of a parsed tree.

    The top-level value is at depth 1; every member name, member value and
    array element is one deeper than its container and counts as one node.
    """
    count = 0
    active: set[int] = set()

    def node(depth: int) -> None:
        nonlocal count
        count += 1
        if depth > MAX_DOCUMENT_DEPTH:
            raise ToolInputError(DOCUMENT_DEPTH_MESSAGE)
        if count > MAX_DOCUMENT_NODES:
            raise ToolInputError(DOCUMENT_NODES_MESSAGE)

    def visit(item: Any, depth: int) -> None:
        node(depth)
        if isinstance(item, str):
            _reject_surrogates(item)
            return
        if item is None or isinstance(item, bool):
            return
        if isinstance(item, int | float):
            # Scan preparation and the Audit inventory accept only finite
            # numbers within the exact integer range (LIM-SCN-67, AUD-R146).
            if abs(item) > yaml_scalars.MAX_SAFE_INTEGER or not math.isfinite(item):
                raise ToolInputError(yaml_scalars.NUMBER_MESSAGE)
            return
        if isinstance(item, dict):
            identity = id(item)
            if identity in active:
                raise ToolInputError("OpenAPI document contains a cycle")
            active.add(identity)
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ToolInputError("OpenAPI object keys must be strings")
                _reject_surrogates(key)
                node(depth + 1)
                visit(child, depth + 1)
            active.remove(identity)
            return
        if isinstance(item, list):
            identity = id(item)
            if identity in active:
                raise ToolInputError("OpenAPI document contains a cycle")
            active.add(identity)
            for child in item:
                visit(child, depth + 1)
            active.remove(identity)
            return
        raise ToolInputError(f"OpenAPI document contains unsupported {type(item).__name__} value")

    visit(value, 1)


def _reject_surrogates(text: str) -> None:
    """Reject surrogate code points, which UTF-8 and the Server readers cannot carry."""

    if not text.isascii() and _SURROGATE.search(text) is not None:
        raise ToolInputError("OpenAPI document strings must not contain surrogate code points")


_SURROGATE = re.compile("[\ud800-\udfff]")


def _validate_document(document: dict[str, Any], *, require_provenance: bool) -> None:
    _validate_json_tree(document)
    _validate_structure(document, require_provenance=require_provenance)


def _validate_structure(document: dict[str, Any], *, require_provenance: bool) -> None:
    """Check OpenAPI structure of a document already within the document bounds."""

    version = document.get("openapi")
    if not isinstance(version, str) or not (
        version.startswith("3.0.") or version.startswith("3.1.")
    ):
        raise ToolInputError("OpenAPI document must declare a supported 3.0.x or 3.1.x version")
    info = document.get("info")
    if not isinstance(info, dict):
        raise ToolInputError("OpenAPI info must be an object")
    _require_nonempty("info.title", info.get("title"))
    _require_nonempty("info.version", info.get("version"))
    paths = document.get("paths")
    if not isinstance(paths, dict):
        raise ToolInputError("OpenAPI paths must be an object")
    for path, path_item in paths.items():
        _validate_api_path(path)
        _validate_path_item(path_item)
        if version.startswith("3.0.") and "$ref" in path_item:
            # Required x-path-files provenance cannot sit beside a 3.0 $ref,
            # and a local alias duplicates its target's operationIds. The path
            # comes last so a long path cannot truncate the guidance.
            raise ToolInputError(
                "OpenAPI 3.0 Path Item $ref is not supported; inline the Path Item with "
                f"its own operations and unique operationIds at path {path}"
            )
        if require_provenance:
            _validate_provenance(path_item.get("x-path-files"), f"path {path}")
    components = document.get("components", {})
    if not isinstance(components, dict):
        raise ToolInputError("OpenAPI components must be an object")
    for section, values in components.items():
        if section.startswith("x-"):
            # Specification Extensions are allowed on the Components Object
            # and carry arbitrary JSON rather than named components.
            continue
        if section not in ALLOWED_COMPONENT_SECTIONS:
            # An unknown component bucket is almost certainly a model mistake.
            raise ToolInputError(f"unsupported OpenAPI component section: {section}")
        if version.startswith("3.0.") and section == "pathItems":
            raise ToolInputError("OpenAPI components.pathItems requires OpenAPI 3.1")
        if not isinstance(values, dict):
            raise ToolInputError(f"OpenAPI components.{section} must be an object")
        for name, component in values.items():
            _validate_component_name(name)
            _validate_component(section, component)
            if require_provenance and not _reference_only_component(component, version):
                _validate_provenance(
                    component.get("x-component-files"), f"component {section}.{name}"
                )
    if "servers" in document:
        _validate_servers(document["servers"])
    if "tags" in document:
        _validate_tags(document["tags"])
    _validate_schema_shapes(document, version)
    _validate_reference_siblings(document, version)
    _validate_local_refs(document)


def _validate_path_item(value: Any) -> None:
    if not isinstance(value, dict):
        raise ToolInputError("OpenAPI Path Item must be an object")
    try:
        PathItem.model_validate(value)
    except ValidationError as error:
        raise ToolInputError(_validation_message("PathItem", error)) from error


def _validate_component(section: str, value: Any) -> None:
    if not isinstance(value, dict):
        raise ToolInputError("OpenAPI component must be an object")
    model: Any | None = {
        "securitySchemes": SecurityScheme,
        "requestBodies": RequestBody,
        "responses": Response,
    }.get(section)
    if model is not None and "$ref" not in value:
        try:
            model.model_validate(value)
        except ValidationError as error:
            raise ToolInputError(_validation_message(model.__name__, error)) from error


def _reference_only_component(value: Any, version: str) -> bool:
    return version.startswith("3.0.") and isinstance(value, dict) and set(value) == {"$ref"}


def _validate_reference_siblings(value: Any, version: str) -> None:
    """Reject Reference Object siblings which OpenAPI 3.0 ignores."""

    if isinstance(value, dict):
        if version.startswith("3.0.") and "$ref" in value and len(value) != 1:
            raise ToolInputError("OpenAPI 3.0 $ref objects cannot contain sibling fields")
        for child in value.values():
            _validate_reference_siblings(child, version)
    elif isinstance(value, list):
        for child in value:
            _validate_reference_siblings(child, version)


def _validate_schema_shapes(document: dict[str, Any], version: str) -> None:
    """Validate high-value Schema Object invariants without a second CLI call."""

    components = document.get("components", {})
    schemas = components.get("schemas", {}) if isinstance(components, dict) else {}
    if isinstance(schemas, dict):
        for schema in schemas.values():
            _validate_schema_shape(schema, version)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "schema" and isinstance(child, dict):
                    _validate_schema_shape(child, version)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(document)


def _validate_schema_shape(value: Any, version: str) -> None:
    if isinstance(value, bool) and version.startswith("3.1."):
        # OpenAPI 3.1 uses JSON Schema 2020-12, where true/false are schemas.
        return
    if not isinstance(value, dict):
        raise ToolInputError("OpenAPI schema must be an object")
    schema_type = value.get("type")
    if "type" in value:
        valid_type = isinstance(schema_type, str) and bool(schema_type)
        if version.startswith("3.1.") and isinstance(schema_type, list):
            valid_type = bool(schema_type) and all(
                isinstance(item, str) and bool(item) for item in schema_type
            )
        if not valid_type:
            raise ToolInputError("OpenAPI schema type must not be null or empty")
    properties = value.get("properties")
    if "properties" in value and not isinstance(properties, dict):
        raise ToolInputError("OpenAPI schema properties must be an object")
    required = value.get("required")
    if "required" in value and (
        not isinstance(required, list)
        or not required
        or any(not isinstance(item, str) or not item for item in required)
    ):
        raise ToolInputError("OpenAPI schema required must be a non-empty string list")
    for keyword in ("allOf", "anyOf", "oneOf"):
        members = value.get(keyword)
        if keyword in value and (
            not isinstance(members, list)
            or not members
            or any(not _schema_value(item, version) for item in members)
        ):
            raise ToolInputError(f"OpenAPI schema {keyword} must contain at least one schema")
        if isinstance(members, list):
            for member in members:
                _validate_schema_shape(member, version)
    if isinstance(properties, dict):
        for property_schema in properties.values():
            _validate_schema_shape(property_schema, version)
    if "items" in value:
        _validate_schema_shape(value["items"], version)
    additional = value.get("additionalProperties")
    if "additionalProperties" in value and not isinstance(additional, bool | dict):
        raise ToolInputError("OpenAPI schema additionalProperties must be a boolean or schema")
    if isinstance(additional, dict):
        _validate_schema_shape(additional, version)


def _schema_value(value: Any, version: str) -> bool:
    return isinstance(value, dict) or (isinstance(value, bool) and version.startswith("3.1."))


def _validation_message(kind: str, error: ValidationError) -> str:
    details = [
        {"field": ".".join(str(item) for item in issue["loc"]), "error": issue["msg"]}
        for issue in error.errors()[:20]
    ]
    return json.dumps({"component": kind, "errors": details}, separators=(",", ":"))


def _validate_local_refs(document: dict[str, Any]) -> None:
    refs: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            ref = value.get("$ref")
            if ref is not None:
                if not isinstance(ref, str) or not ref.startswith("#/"):
                    raise ToolInputError("OpenAPI $ref must be a local JSON pointer")
                refs.append(ref)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(document)
    for ref in refs:
        current: Any = document
        for encoded in ref[2:].split("/"):
            token = encoded.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict) and token in current:
                current = current[token]
            elif (
                isinstance(current, list)
                and token.isascii()
                and token.isdigit()
                and int(token) < len(current)
            ):
                current = current[int(token)]
            else:
                raise ToolInputError(f"OpenAPI local reference is unresolved: {ref}")


def _validate_provenance(value: Any, kind: str) -> None:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ToolInputError(f"OpenAPI {kind} is missing source provenance")


def _validate_servers(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 100:
        raise ToolInputError("OpenAPI servers must be a list of at most 100 entries")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for server in value:
        if not isinstance(server, dict):
            raise ToolInputError("OpenAPI server entries must be objects")
        url = server.get("url")
        _require_nonempty("server.url", url)
        if not (isinstance(url, str)):
            raise RuntimeError("Expected isinstance(url, str)")
        if url in seen:
            raise ToolInputError("OpenAPI server URLs must be unique")
        seen.add(url)
        description = server.get("description")
        if description is not None and not isinstance(description, str):
            raise ToolInputError("OpenAPI server description must be a string")
        result.append(copy.deepcopy(server))
    return result


def _validate_tags(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 100:
        raise ToolInputError("OpenAPI tags must be a list of at most 100 entries")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tag in value:
        if not isinstance(tag, dict):
            raise ToolInputError("OpenAPI tag entries must be objects")
        unsupported = [
            key
            for key in tag
            if key not in {"name", "description", "externalDocs"}
            and not (isinstance(key, str) and key.startswith("x-"))
        ]
        if unsupported:
            raise ToolInputError("OpenAPI tag contains unsupported fields")
        name = tag.get("name")
        _require_nonempty("tag.name", name)
        if not (isinstance(name, str)):
            raise RuntimeError("Expected isinstance(name, str)")
        if len(name) > 256 or name in seen:
            raise ToolInputError("OpenAPI tag names must be unique and at most 256 characters")
        seen.add(name)
        description = tag.get("description")
        if description is not None and not isinstance(description, str):
            raise ToolInputError("OpenAPI tag description must be a string")
        external_docs = tag.get("externalDocs")
        if external_docs is not None:
            if not isinstance(external_docs, dict) or set(external_docs) - {
                "url",
                "description",
            }:
                raise ToolInputError("OpenAPI tag externalDocs is invalid")
            _require_nonempty("tag.externalDocs.url", external_docs.get("url"))
            external_description = external_docs.get("description")
            if external_description is not None and not isinstance(external_description, str):
                raise ToolInputError("OpenAPI tag externalDocs description must be a string")
        result.append(copy.deepcopy(tag))
    return result


def _prepare_document(document: dict[str, Any]) -> bytes:
    """Validate a complete new document once and render it for writing."""

    _validate_document(document, require_provenance=False)
    return _dump_document(document)


def _prepare_mutation(
    document: dict[str, Any], modifier: Callable[[dict[str, Any]], None]
) -> tuple[dict[str, Any], bytes] | None:
    """Apply a modifier to a copy, validate the whole result once and render it.

    Returns None when the valid result equals the current document. The
    current document is only read, and the copy is new.
    """

    working = copy.deepcopy(document)
    modifier(working)
    _validate_document(working, require_provenance=False)
    if working == document:
        return None
    return working, _dump_document(working)


def _checked_servers(value: Any) -> list[dict[str, Any]]:
    _validate_json_tree(value)
    return _validate_servers(value)


def _checked_tags(value: Any) -> list[dict[str, Any]]:
    _validate_json_tree(value)
    return _validate_tags(value)


def _render_full_read(document: dict[str, Any]) -> str:
    rendered = _dump_document(document)
    if len(rendered) > MAX_TARGETED_RESULT_BYTES:
        raise ToolInputError("OpenAPI document exceeds the full-read model output limit")
    return rendered.decode("utf-8")


def _dump_document(document: dict[str, Any]) -> bytes:
    rendered = yaml.dump(
        document,
        Dumper=_OpenAPIDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=120,
    ).encode("utf-8")
    if len(rendered) > MAX_DOCUMENT_BYTES:
        raise ToolInputError(DOCUMENT_BYTES_MESSAGE)
    return rendered


def _deep_merge(base: Any, update: Any) -> Any:
    if isinstance(base, dict) and isinstance(update, dict):
        # A Reference Object replaces the value it points from. Retaining an
        # inline schema's old fields beside a new $ref creates an invalid 3.0
        # document and makes later repair ambiguous.
        if "$ref" in update or "$ref" in base:
            return copy.deepcopy(update)
        result = copy.deepcopy(base)
        for key, value in update.items():
            result[key] = _deep_merge(result[key], value) if key in result else copy.deepcopy(value)
        return result
    return copy.deepcopy(update)


def _validate_api_path(value: Any) -> str:
    if not isinstance(value, str) or value != value.strip() or not value.startswith("/"):
        raise ToolInputError("OpenAPI path must be a normalized string beginning with /")
    if not value or len(value) > 2048 or "\x00" in value:
        raise ToolInputError("OpenAPI path must be at most 2048 characters without NUL characters")
    return value


def _validate_component_section(value: Any) -> str:
    if not isinstance(value, str) or value not in ALLOWED_COMPONENT_SECTIONS:
        raise ToolInputError(
            "unsupported OpenAPI component section; use "
            + ", ".join(sorted(ALLOWED_COMPONENT_SECTIONS))
        )
    return value


def _validate_component_name(value: Any) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256 or "\x00" in value:
        raise ToolInputError(
            "OpenAPI component name must be non-empty, at most 256 characters, without NUL"
        )


def _validate_relative_source_path(raw: Any) -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw or "\x00" in raw:
        raise ToolInputError("source evidence path must be a non-empty relative string using /")
    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ToolInputError("source evidence path must be normalized and relative")
    path = PurePosixPath(raw)
    if path.is_absolute() or path.as_posix() != raw:
        raise ToolInputError("source evidence path must be normalized and relative")
    return raw


def _require_nonempty(field: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError(f"{field} must be a non-empty string")


def _targeted_copy(value: Any) -> Any:
    """Return a copy of one document value that fits model output."""

    if _encoded_size(value) > MAX_TARGETED_RESULT_BYTES:
        raise ToolInputError("targeted OpenAPI result exceeds the model output limit")
    return copy.deepcopy(value)


def _encoded_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _bounded_name_listing(
    envelope: dict[str, Any], field: str, keys: Iterable[str]
) -> dict[str, Any]:
    """Return the longest sorted prefix whose complete result fits model output."""

    names = sorted(keys)
    selected: list[str] = []
    # "truncated": false is the longer encoding, so the sized envelope never
    # grows when the final flag is filled in.
    result = {**envelope, field: selected, "total": len(names), "truncated": False}
    size = _encoded_size(result)
    for name in names[:MAX_LIST_ITEMS]:
        cost = _encoded_size(name) + (1 if selected else 0)
        if size + cost > MAX_TARGETED_RESULT_BYTES:
            break
        selected.append(name)
        size += cost
    result["truncated"] = len(selected) < len(names)
    return result


def _document_state(
    artifact: ArtifactRef, target_name: str, *, changed: bool, copied: bool
) -> dict[str, Any]:
    return {
        "artifact": artifact.require_exact().model_dump(by_alias=True),
        "mediaType": TARGET_MEDIA_TYPE,
        "targetName": target_name,
        "changed": changed,
        "copied": copied,
    }


async def _run_vacuum(
    source_text: str,
    launcher: ProxySubprocessLauncher | None = None,
) -> dict[str, Any]:
    executable = shutil.which("vacuum")
    if executable is None:
        return {
            "available": False,
            "executionError": "Vacuum executable is unavailable",
            "issues": [],
            "truncated": False,
        }
    try:
        command = [executable, "spectral-report", "-i", "-o"]
        process = await run_tool_command(
            command,
            launcher=launcher,
            input=source_text.encode("utf-8"),
            env={"PATH": os.environ.get("PATH", os.defpath), "LANG": "C.UTF-8"},
            timeout=VACUUM_TIMEOUT_SECONDS,
            max_output_bytes=2 * MAX_VACUUM_OUTPUT_BYTES,
        )
    except subprocess.TimeoutExpired:
        return {
            "available": True,
            "executionError": "Vacuum validation timed out",
            "issues": [],
            "truncated": False,
        }
    except (OSError, ProcessOutputLimitError):
        return {
            "available": True,
            "executionError": "Vacuum could not be executed",
            "issues": [],
            "truncated": False,
        }
    # Parsing the output and cutting snippets scale with the output and the
    # document; a cancelled caller still waits for the thread.
    return await to_thread_until_done(
        _vacuum_report,
        process.returncode,
        process.stdout,
        source_text,
        name="openapi-validator-output",
    )


def _vacuum_report(returncode: int, stdout: bytes, source_text: str) -> dict[str, Any]:
    """Turn a finished Vacuum process into the bounded validator result."""

    if returncode not in {0, 1}:
        return {
            "available": True,
            "executionError": "Vacuum returned an execution failure",
            "issues": [],
            "truncated": False,
        }
    if not stdout or len(stdout) > MAX_VACUUM_OUTPUT_BYTES:
        return {
            "available": True,
            "executionError": "Vacuum returned empty or oversized output",
            "issues": [],
            "truncated": False,
        }
    try:
        parsed = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {
            "available": True,
            "executionError": "Vacuum returned invalid JSON",
            "issues": [],
            "truncated": False,
        }
    if not isinstance(parsed, list):
        return {
            "available": True,
            "executionError": "Vacuum returned an unexpected JSON shape",
            "issues": [],
            "truncated": False,
        }
    serious = [
        item
        for item in parsed
        if isinstance(item, dict)
        and type(item.get("severity")) is int
        and item["severity"] in {0, 1}
    ]
    serious.sort(key=lambda item: item["severity"])
    truncated = len(serious) > MAX_VALIDATION_ISSUES
    source_lines = source_text.splitlines()
    return {
        "available": True,
        "executionError": None,
        "issues": [
            _issue_with_snippet(item, source_lines) for item in serious[:MAX_VALIDATION_ISSUES]
        ],
        "truncated": truncated,
    }


def _issue_with_snippet(issue: dict[str, Any], lines: list[str]) -> dict[str, Any]:
    result = {key: copy.deepcopy(value) for key, value in issue.items() if key != "range"}
    result["snippet"] = ""
    coordinates = issue.get("range")
    if not isinstance(coordinates, dict):
        return result
    start = coordinates.get("start")
    end = coordinates.get("end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return result
    values = (start.get("line"), start.get("character"), end.get("line"), end.get("character"))
    if not all(type(value) is int for value in values):
        return result
    start_line, start_char, end_line, end_char = values
    if not (1 <= start_line <= end_line <= len(lines) and start_char >= 1 and end_char >= 1):
        return result
    if start_line == end_line:
        snippet = lines[start_line - 1][start_char - 1 : end_char - 1]
    else:
        selected = [lines[start_line - 1][start_char - 1 :]]
        selected.extend(lines[start_line : end_line - 1])
        selected.append(lines[end_line - 1][: end_char - 1])
        snippet = "\n".join(selected)
    result["snippet"] = snippet[:2000]
    return result
