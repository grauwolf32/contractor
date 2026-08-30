"""Namespace-bound, CAS-backed OpenAPI 3 tools for one Worker allocation."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

import yaml
from pydantic import ValidationError

from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.toolsets.openapi_models import (
    PathItem,
    RequestBody,
    Response,
    SecurityScheme,
)
from contractor_runtime.toolsets.run_artifacts import ArtifactClientFactory, ToolMetrics
from contractor_runtime.workspace import AllocationWorkspace

MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
MAX_DOCUMENT_DEPTH = 128
MAX_DOCUMENT_ITEMS = 100_000
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

HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")

BASE_DOCUMENT: dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {"title": "", "description": "", "version": "1.0.0"},
    "paths": {},
    "components": {section: {} for section in sorted(OPENAPI_30_COMPONENT_SECTIONS)},
}


class OpenAPIToolsetFactory:
    ref = "openapi@1"
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

    def __init__(self, client_factory: ArtifactClientFactory | None = None) -> None:
        self._client_factory = client_factory or _unconfigured_client

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
    ) -> Mapping[str, Any]:
        del run_id
        unknown = sorted(set(selected) - self.exported_tools)
        if unknown:
            raise ValueError(f"unknown selected tools: {', '.join(unknown)}")
        metrics = getattr(state, "metrics", None)
        if metrics is None or not callable(getattr(metrics, "record_tool_call", None)):
            raise TypeError("openapi@1 requires State.metrics")
        client = self._client_factory(allocation_id, runtime_settings)
        session = _OpenAPISession(client, namespace, workspace)
        secrets = (runtime_settings.llm_gateway_token.get_secret_value(),)
        builders: dict[str, Callable[[], Any]] = {
            "load_openapi": lambda: LoadOpenAPITool(session, client, metrics, secrets),
            "initialize_openapi": lambda: InitializeOpenAPITool(session, client, metrics, secrets),
            "get_openapi_info": lambda: GetOpenAPIInfoTool(session, client, metrics, secrets),
            "set_openapi_info": lambda: SetOpenAPIInfoTool(session, client, metrics, secrets),
            "list_openapi_servers": lambda: ListOpenAPIServersTool(
                session, client, metrics, secrets
            ),
            "set_openapi_servers": lambda: SetOpenAPIServersTool(session, client, metrics, secrets),
            "list_openapi_tags": lambda: ListOpenAPITagsTool(session, client, metrics, secrets),
            "set_openapi_tags": lambda: SetOpenAPITagsTool(session, client, metrics, secrets),
            "list_openapi_paths": lambda: ListOpenAPIPathsTool(session, client, metrics, secrets),
            "get_openapi_path": lambda: GetOpenAPIPathTool(session, client, metrics, secrets),
            "upsert_openapi_path": lambda: UpsertOpenAPIPathTool(session, client, metrics, secrets),
            "remove_openapi_path": lambda: RemoveOpenAPIPathTool(session, client, metrics, secrets),
            "list_openapi_components": lambda: ListOpenAPIComponentsTool(
                session, client, metrics, secrets
            ),
            "get_openapi_component": lambda: GetOpenAPIComponentTool(
                session, client, metrics, secrets
            ),
            "upsert_openapi_component": lambda: UpsertOpenAPIComponentTool(
                session, client, metrics, secrets
            ),
            "remove_openapi_component": lambda: RemoveOpenAPIComponentTool(
                session, client, metrics, secrets
            ),
            "read_openapi_document": lambda: ReadOpenAPIDocumentTool(
                session, client, metrics, secrets
            ),
            "validate_openapi": lambda: ValidateOpenAPITool(session, client, metrics, secrets),
        }
        return {name: builders[name]() for name in selected}


class _OpenAPISession:
    def __init__(
        self,
        client: ArtifactClient,
        namespace: str,
        workspace: AllocationWorkspace,
    ) -> None:
        self._client = client
        self._namespace = namespace
        self._source_root = workspace.path / "source"
        self._lock = asyncio.Lock()
        self._document: dict[str, Any] | None = None
        self._target_name: str | None = None
        self._revision: str | None = None

    async def load(
        self,
        *,
        namespace: str,
        name: str,
        revision: str | None,
        target_name: str,
        expected_revision: str | None,
    ) -> dict[str, Any]:
        _validate_target_name(target_name)
        if namespace != self._namespace and revision is None:
            raise ValueError(
                "an OpenAPI seed outside the Worker namespace requires an exact revision"
            )
        source_ref = ArtifactRef(namespace=namespace, name=name, revision=revision)
        async with self._lock:
            value = await self._client.read_artifact(source_ref)
            if revision is not None and value.artifact.revision != revision:
                raise ValueError("Artifact API did not preserve the requested exact revision")
            if value.media_type not in {"application/yaml", "application/json"}:
                raise ValueError(
                    "OpenAPI seed media type must be application/yaml or application/json"
                )
            document = _parse_document(value.data)
            _validate_document(document, require_provenance=False)
            same_binding = namespace == self._namespace and name == target_name
            if same_binding and value.media_type == TARGET_MEDIA_TYPE:
                self._document = document
                self._target_name = target_name
                self._revision = value.artifact.require_exact().revision
                return _document_state(value.artifact, target_name, changed=False, copied=False)
            written = await self._write_document(
                document,
                target_name=target_name,
                expected_revision=(value.artifact.revision if same_binding else expected_revision),
            )
            self._document = document
            self._target_name = target_name
            self._revision = written.artifact.require_exact().revision
            return _document_state(written.artifact, target_name, changed=True, copied=True)

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
        _validate_target_name(target_name)
        document = copy.deepcopy(BASE_DOCUMENT)
        document["info"] = {"title": title, "description": description, "version": version}
        async with self._lock:
            written = await self._write_document(
                document, target_name=target_name, expected_revision=expected_revision
            )
            self._document = document
            self._target_name = target_name
            self._revision = written.artifact.require_exact().revision
            return _document_state(written.artifact, target_name, changed=True, copied=False)

    async def get_info(self) -> dict[str, Any]:
        async with self._lock:
            document, artifact = self._require_document()
            value = copy.deepcopy(document["info"])
            _bound_targeted_result(value)
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
            servers = copy.deepcopy(document.get("servers", []))
            _bound_targeted_result(servers)
            return {"artifact": artifact.model_dump(by_alias=True), "servers": servers}

    async def set_servers(self, servers: list[dict[str, Any]]) -> dict[str, Any]:
        normalized = _validate_servers(servers)
        return await self._mutate(
            lambda document: document.__setitem__("servers", normalized),
            operation="set_servers",
        )

    async def list_tags(self) -> dict[str, Any]:
        async with self._lock:
            document, artifact = self._require_document()
            tags = copy.deepcopy(document.get("tags", []))
            _bound_targeted_result(tags)
            return {"artifact": artifact.model_dump(by_alias=True), "tags": tags}

    async def set_tags(self, tags: list[dict[str, Any]]) -> dict[str, Any]:
        normalized = _validate_tags(tags)
        return await self._mutate(
            lambda document: document.__setitem__("tags", normalized),
            operation="set_tags",
        )

    async def list_paths(self) -> dict[str, Any]:
        async with self._lock:
            document, artifact = self._require_document()
            paths = sorted(document["paths"])
            return {
                "artifact": artifact.model_dump(by_alias=True),
                "paths": paths[:MAX_LIST_ITEMS],
                "total": len(paths),
                "truncated": len(paths) > MAX_LIST_ITEMS,
            }

    async def get_path(self, path: str) -> dict[str, Any]:
        normalized = _validate_api_path(path)
        async with self._lock:
            document, artifact = self._require_document()
            if normalized not in document["paths"]:
                raise ValueError("OpenAPI path is absent")
            value = copy.deepcopy(document["paths"][normalized])
            _bound_targeted_result(value)
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
            raise ValueError("path_item must be an object")
        validated_evidence = self._validate_evidence(evidence_files)
        candidate = copy.deepcopy(path_item)
        candidate["x-path-files"] = validated_evidence
        _validate_path_item(candidate)

        def modify(document: dict[str, Any]) -> None:
            current = document["paths"].get(normalized, {})
            document["paths"][normalized] = _deep_merge(current, candidate)
            _validate_path_item(document["paths"][normalized])

        result = await self._mutate(modify, operation="upsert_path")
        result["path"] = normalized
        return result

    async def remove_path(self, path: str) -> dict[str, Any]:
        normalized = _validate_api_path(path)

        def modify(document: dict[str, Any]) -> None:
            if normalized not in document["paths"]:
                raise ValueError("OpenAPI path is absent")
            del document["paths"][normalized]

        result = await self._mutate(modify, operation="remove_path")
        result["path"] = normalized
        return result

    async def list_components(self, section: str) -> dict[str, Any]:
        normalized = _validate_component_section(section)
        async with self._lock:
            document, artifact = self._require_document()
            values = sorted(document.get("components", {}).get(normalized, {}))
            return {
                "artifact": artifact.model_dump(by_alias=True),
                "section": normalized,
                "components": values[:MAX_LIST_ITEMS],
                "total": len(values),
                "truncated": len(values) > MAX_LIST_ITEMS,
            }

    async def get_component(self, section: str, name: str) -> dict[str, Any]:
        normalized = _validate_component_section(section)
        _validate_component_name(name)
        async with self._lock:
            document, artifact = self._require_document()
            components = document.get("components", {}).get(normalized, {})
            if name not in components:
                raise ValueError("OpenAPI component is absent")
            value = copy.deepcopy(components[name])
            _bound_targeted_result(value)
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
            raise ValueError("component must be an object")
        validated_evidence = self._validate_evidence(evidence_files)
        candidate = copy.deepcopy(component)
        candidate["x-component-files"] = validated_evidence
        _validate_component(normalized, candidate)

        def modify(document: dict[str, Any]) -> None:
            values = document.setdefault("components", {}).setdefault(normalized, {})
            values[name] = _deep_merge(values.get(name, {}), candidate)
            _validate_component(normalized, values[name])

        result = await self._mutate(modify, operation="upsert_component")
        result.update({"section": normalized, "name": name})
        return result

    async def remove_component(self, section: str, name: str) -> dict[str, Any]:
        normalized = _validate_component_section(section)
        _validate_component_name(name)

        def modify(document: dict[str, Any]) -> None:
            values = document.setdefault("components", {}).setdefault(normalized, {})
            if name not in values:
                raise ValueError("OpenAPI component is absent")
            del values[name]

        result = await self._mutate(modify, operation="remove_component")
        result.update({"section": normalized, "name": name})
        return result

    async def read_document(self) -> dict[str, Any]:
        async with self._lock:
            document, artifact = self._require_document()
            rendered = _dump_document(document).decode("utf-8")
            if len(rendered.encode("utf-8")) > MAX_TARGETED_RESULT_BYTES:
                raise ValueError("OpenAPI document exceeds the full-read model output limit")
            return {
                "artifact": artifact.model_dump(by_alias=True),
                "mediaType": TARGET_MEDIA_TYPE,
                "document": rendered,
            }

    async def validate(self) -> dict[str, Any]:
        async with self._lock:
            document, artifact = self._require_document()
            rendered = _dump_document(document).decode("utf-8")
            structural_errors: list[str] = []
            try:
                _validate_document(document, require_provenance=True)
                self._validate_current_provenance(document)
            except ValueError as error:
                structural_errors.append(str(error))
            vacuum = await asyncio.to_thread(_run_vacuum, rendered)
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

    async def close(self) -> None:
        async with self._lock:
            self._document = None
            self._target_name = None
            self._revision = None

    async def _mutate(
        self, modifier: Callable[[dict[str, Any]], None], *, operation: str
    ) -> dict[str, Any]:
        async with self._lock:
            document, artifact = self._require_document()
            working = copy.deepcopy(document)
            modifier(working)
            _validate_document(working, require_provenance=False)
            if working == document:
                return _document_state(
                    artifact, self._target_name or DEFAULT_TARGET_NAME, changed=False, copied=False
                ) | {"operation": operation}
            assert self._target_name is not None and self._revision is not None
            written = await self._write_document(
                working,
                target_name=self._target_name,
                expected_revision=self._revision,
            )
            self._document = working
            self._revision = written.artifact.require_exact().revision
            return _document_state(
                written.artifact, self._target_name, changed=True, copied=False
            ) | {"operation": operation}

    async def _write_document(
        self,
        document: dict[str, Any],
        *,
        target_name: str,
        expected_revision: str | None,
    ) -> Any:
        _validate_document(document, require_provenance=False)
        data = _dump_document(document)
        return await self._client.write_artifact(
            ArtifactRef(namespace=self._namespace, name=target_name),
            data=data,
            media_type=TARGET_MEDIA_TYPE,
            expected_revision=expected_revision,
        )

    def _require_document(self) -> tuple[dict[str, Any], ArtifactRef]:
        if self._document is None or self._target_name is None or self._revision is None:
            raise ValueError("load_openapi or initialize_openapi must be called first")
        return self._document, ArtifactRef(
            namespace=self._namespace,
            name=self._target_name,
            revision=self._revision,
        )

    def _validate_evidence(self, evidence_files: list[str]) -> list[str]:
        if not isinstance(evidence_files, list) or not evidence_files:
            raise ValueError("at least one source evidence file is required")
        if len(evidence_files) > 100:
            raise ValueError("source evidence exceeds the 100-file limit")
        root = self._source_root.resolve()
        if self._source_root.is_symlink() or not self._source_root.is_dir():
            raise ValueError("open_source_archive must materialize source before mutation")
        result: list[str] = []
        seen: set[str] = set()
        for raw in evidence_files:
            normalized = _validate_relative_source_path(raw)
            if PurePosixPath(normalized).suffix.lower() in SOURCE_EVIDENCE_DENIED_SUFFIXES:
                raise ValueError("OpenAPI provenance must reference implementation source files")
            if normalized in seen:
                continue
            candidate = self._source_root.joinpath(*PurePosixPath(normalized).parts)
            resolved = candidate.resolve()
            if (
                not resolved.is_relative_to(root)
                or candidate.is_symlink()
                or not candidate.is_file()
            ):
                raise ValueError(f"source evidence file does not exist: {normalized}")
            seen.add(normalized)
            result.append(normalized)
        return result

    def _validate_current_provenance(self, document: dict[str, Any]) -> None:
        for path_item in document["paths"].values():
            self._validate_evidence(path_item["x-path-files"])
        for values in document.get("components", {}).values():
            for component in values.values():
                self._validate_evidence(component["x-component-files"])


class _BaseOpenAPITool:
    name: str
    description: str

    def __init__(
        self,
        session: _OpenAPISession,
        client: ArtifactClient,
        metrics: ToolMetrics,
        secrets: tuple[str, ...],
    ) -> None:
        self._session = session
        self._client = client
        self._metrics = metrics
        self._secrets = secrets
        self.__name__ = self.name
        self.__doc__ = self.description

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return tuple(getattr(self._client, "known_exact_refs", ()))

    async def close(self) -> None:
        await self._session.close()
        self._secrets = ()

    async def _call(
        self,
        arguments: Mapping[str, Any],
        operation: Any,
        metric_result: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        try:
            result = await operation
            self._metrics.record_tool_call(
                self.name,
                arguments=arguments,
                result=metric_result(result),
                secrets=self._secrets,
                duration_ms=_elapsed_ms(started_ns),
            )
            return result
        except Exception as error:
            self._metrics.record_tool_call(
                self.name,
                arguments=arguments,
                error=error,
                secrets=self._secrets,
                duration_ms=_elapsed_ms(started_ns),
            )
            raise


def _artifact_metric(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "artifact": result.get("artifact"),
        "changed": result.get("changed"),
        "operation": result.get("operation"),
    }


class LoadOpenAPITool(_BaseOpenAPITool):
    name = "load_openapi"
    description = (
        "Load an OpenAPI YAML/JSON artifact and copy or resume it in this Agent Namespace."
    )

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
    description = "Create a minimal OpenAPI 3.0.3 YAML document in this Agent Namespace."

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
    description = "Read the current OpenAPI info object."

    async def __call__(self) -> dict[str, Any]:
        return await self._call(
            {},
            self._session.get_info(),
            lambda result: {"artifact": result["artifact"]},
        )


class SetOpenAPIInfoTool(_BaseOpenAPITool):
    name = "set_openapi_info"
    description = "CAS-update OpenAPI info while preserving fields not explicitly supplied."

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
    description = "Read the current OpenAPI server list."

    async def __call__(self) -> dict[str, Any]:
        return await self._call(
            {},
            self._session.list_servers(),
            lambda result: {"artifact": result["artifact"], "count": len(result["servers"])},
        )


class SetOpenAPIServersTool(_BaseOpenAPITool):
    name = "set_openapi_servers"
    description = "Validate and CAS-replace the OpenAPI server list."

    async def __call__(self, servers: list[dict[str, Any]]) -> dict[str, Any]:
        return await self._call(
            {"content": servers}, self._session.set_servers(servers), _artifact_metric
        )


class ListOpenAPITagsTool(_BaseOpenAPITool):
    name = "list_openapi_tags"
    description = "Read the current top-level OpenAPI tag declarations."

    async def __call__(self) -> dict[str, Any]:
        return await self._call(
            {},
            self._session.list_tags(),
            lambda result: {"artifact": result["artifact"], "count": len(result["tags"])},
        )


class SetOpenAPITagsTool(_BaseOpenAPITool):
    name = "set_openapi_tags"
    description = "Validate and CAS-replace top-level OpenAPI tag declarations."

    async def __call__(self, tags: list[dict[str, Any]]) -> dict[str, Any]:
        return await self._call({"content": tags}, self._session.set_tags(tags), _artifact_metric)


class ListOpenAPIPathsTool(_BaseOpenAPITool):
    name = "list_openapi_paths"
    description = "List current OpenAPI path keys without returning whole path definitions."

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
    description = "Read one current OpenAPI Path Item."

    async def __call__(self, path: str) -> dict[str, Any]:
        return await self._call(
            {"path": path},
            self._session.get_path(path),
            lambda result: {"artifact": result["artifact"], "path": result["path"]},
        )


class UpsertOpenAPIPathTool(_BaseOpenAPITool):
    name = "upsert_openapi_path"
    description = (
        "Validate and merge one OpenAPI Path Item with required implementation-source evidence, "
        "then CAS-save canonical YAML."
    )

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
    description = "Remove one existing OpenAPI path and CAS-save the document."

    async def __call__(self, path: str) -> dict[str, Any]:
        return await self._call({"path": path}, self._session.remove_path(path), _artifact_metric)


class ListOpenAPIComponentsTool(_BaseOpenAPITool):
    name = "list_openapi_components"
    description = "List component names in one allowed OpenAPI component section."

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
    description = "Read one named OpenAPI component from an allowed section."

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
    description = (
        "Validate and merge one OpenAPI component with required implementation-source "
        "evidence, then CAS-save canonical YAML."
    )

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
    description = "Remove one existing named OpenAPI component and CAS-save the document."

    async def __call__(self, section: str, name: str) -> dict[str, Any]:
        return await self._call(
            {"section": section, "name": name},
            self._session.remove_component(section, name),
            _artifact_metric,
        )


class ReadOpenAPIDocumentTool(_BaseOpenAPITool):
    name = "read_openapi_document"
    description = (
        "Read the full canonical OpenAPI YAML only for global review; prefer targeted tools."
    )

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
    description = (
        "Validate current structure, local refs, provenance, and serious Vacuum findings. "
        "valid is false when Vacuum is unavailable or fails."
    )

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


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


class _NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        del data
        return True


def _construct_unique_mapping(
    loader: _UniqueSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ValueError("OpenAPI YAML mapping keys must be scalar strings") from error
        if duplicate:
            raise ValueError(f"OpenAPI YAML contains duplicate key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _parse_document(data: bytes) -> dict[str, Any]:
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError("OpenAPI document exceeds the 4 MiB domain limit")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("OpenAPI document must be valid UTF-8") from error
    try:
        events = list(yaml.parse(text, Loader=yaml.SafeLoader))
        if any(getattr(event, "anchor", None) is not None for event in events):
            raise ValueError("OpenAPI YAML aliases and anchors are not supported")
        documents = list(yaml.load_all(text, Loader=_UniqueSafeLoader))
    except yaml.YAMLError as error:
        raise ValueError("OpenAPI document is not valid YAML or JSON") from error
    if len(documents) != 1 or not isinstance(documents[0], dict):
        raise ValueError("OpenAPI artifact must contain exactly one object document")
    document = documents[0]
    _validate_json_tree(document)
    return document


def _validate_json_tree(value: Any) -> None:
    count = 0
    active: set[int] = set()

    def visit(item: Any, depth: int) -> None:
        nonlocal count
        if depth > MAX_DOCUMENT_DEPTH:
            raise ValueError("OpenAPI document exceeds the maximum nesting depth")
        count += 1
        if count > MAX_DOCUMENT_ITEMS:
            raise ValueError("OpenAPI document exceeds the collection item limit")
        if item is None or isinstance(item, str | bool):
            return
        if isinstance(item, int):
            if abs(item) > (1 << 53) - 1:
                raise ValueError("OpenAPI document contains a non-I-JSON integer")
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("OpenAPI document contains a non-finite number")
            return
        if isinstance(item, dict):
            identity = id(item)
            if identity in active:
                raise ValueError("OpenAPI document contains a cycle")
            active.add(identity)
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("OpenAPI object keys must be strings")
                visit(child, depth + 1)
            active.remove(identity)
            return
        if isinstance(item, list):
            identity = id(item)
            if identity in active:
                raise ValueError("OpenAPI document contains a cycle")
            active.add(identity)
            for child in item:
                visit(child, depth + 1)
            active.remove(identity)
            return
        raise ValueError(f"OpenAPI document contains unsupported {type(item).__name__} value")

    visit(value, 0)


def _validate_document(document: dict[str, Any], *, require_provenance: bool) -> None:
    _validate_json_tree(document)
    version = document.get("openapi")
    if not isinstance(version, str) or not (
        version.startswith("3.0.") or version.startswith("3.1.")
    ):
        raise ValueError("OpenAPI document must declare a supported 3.0.x or 3.1.x version")
    info = document.get("info")
    if not isinstance(info, dict):
        raise ValueError("OpenAPI info must be an object")
    _require_nonempty("info.title", info.get("title"))
    _require_nonempty("info.version", info.get("version"))
    paths = document.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("OpenAPI paths must be an object")
    for path, path_item in paths.items():
        _validate_api_path(path)
        _validate_path_item(path_item)
        if require_provenance:
            _validate_provenance(path_item.get("x-path-files"), "path")
    components = document.get("components", {})
    if not isinstance(components, dict):
        raise ValueError("OpenAPI components must be an object")
    for section, values in components.items():
        if section not in ALLOWED_COMPONENT_SECTIONS:
            # OpenAPI extensions can add x-* component-adjacent data, but an
            # unknown component bucket is almost certainly a model mistake.
            raise ValueError(f"unsupported OpenAPI component section: {section}")
        if version.startswith("3.0.") and section == "pathItems":
            raise ValueError("OpenAPI components.pathItems requires OpenAPI 3.1")
        if not isinstance(values, dict):
            raise ValueError(f"OpenAPI components.{section} must be an object")
        for name, component in values.items():
            _validate_component_name(name)
            _validate_component(section, component)
            if require_provenance:
                _validate_provenance(component.get("x-component-files"), "component")
    if "servers" in document:
        _validate_servers(document["servers"])
    if "tags" in document:
        _validate_tags(document["tags"])
    _validate_schema_shapes(document, version)
    _validate_reference_siblings(document, version)
    _validate_local_refs(document)


def _validate_path_item(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("OpenAPI Path Item must be an object")
    try:
        PathItem.model_validate(value)
    except ValidationError as error:
        raise ValueError(_validation_message("PathItem", error)) from error


def _validate_component(section: str, value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("OpenAPI component must be an object")
    model: Any | None = {
        "securitySchemes": SecurityScheme,
        "requestBodies": RequestBody,
        "responses": Response,
    }.get(section)
    if model is not None and "$ref" not in value:
        try:
            model.model_validate(value)
        except ValidationError as error:
            raise ValueError(_validation_message(model.__name__, error)) from error


def _validate_reference_siblings(value: Any, version: str) -> None:
    """Reject Reference Object siblings which OpenAPI 3.0 ignores."""

    if isinstance(value, dict):
        if version.startswith("3.0.") and "$ref" in value and len(value) != 1:
            raise ValueError("OpenAPI 3.0 $ref objects cannot contain sibling fields")
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
    if not isinstance(value, dict):
        raise ValueError("OpenAPI schema must be an object")
    schema_type = value.get("type")
    if "type" in value:
        valid_type = isinstance(schema_type, str) and bool(schema_type)
        if version.startswith("3.1.") and isinstance(schema_type, list):
            valid_type = bool(schema_type) and all(
                isinstance(item, str) and bool(item) for item in schema_type
            )
        if not valid_type:
            raise ValueError("OpenAPI schema type must not be null or empty")
    properties = value.get("properties")
    if "properties" in value and not isinstance(properties, dict):
        raise ValueError("OpenAPI schema properties must be an object")
    required = value.get("required")
    if "required" in value and (
        not isinstance(required, list)
        or not required
        or any(not isinstance(item, str) or not item for item in required)
    ):
        raise ValueError("OpenAPI schema required must be a non-empty string list")
    for keyword in ("allOf", "anyOf", "oneOf"):
        members = value.get(keyword)
        if keyword in value and (
            not isinstance(members, list)
            or len(members) < 2
            or any(not isinstance(item, dict) for item in members)
        ):
            raise ValueError(f"OpenAPI schema {keyword} must contain at least two schemas")
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
        raise ValueError("OpenAPI schema additionalProperties must be a boolean or schema")
    if isinstance(additional, dict):
        _validate_schema_shape(additional, version)


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
                    raise ValueError("OpenAPI $ref must be a local JSON pointer")
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
            elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
                current = current[int(token)]
            else:
                raise ValueError(f"OpenAPI local reference is unresolved: {ref}")


def _validate_provenance(value: Any, kind: str) -> None:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"OpenAPI {kind} is missing source provenance")


def _validate_servers(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 100:
        raise ValueError("OpenAPI servers must be a list of at most 100 entries")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for server in value:
        if not isinstance(server, dict):
            raise ValueError("OpenAPI server entries must be objects")
        url = server.get("url")
        _require_nonempty("server.url", url)
        assert isinstance(url, str)
        if url in seen:
            raise ValueError("OpenAPI server URLs must be unique")
        seen.add(url)
        description = server.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError("OpenAPI server description must be a string")
        result.append(copy.deepcopy(server))
    return result


def _validate_tags(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 100:
        raise ValueError("OpenAPI tags must be a list of at most 100 entries")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tag in value:
        if not isinstance(tag, dict):
            raise ValueError("OpenAPI tag entries must be objects")
        unsupported = [
            key
            for key in tag
            if key not in {"name", "description", "externalDocs"}
            and not (isinstance(key, str) and key.startswith("x-"))
        ]
        if unsupported:
            raise ValueError("OpenAPI tag contains unsupported fields")
        name = tag.get("name")
        _require_nonempty("tag.name", name)
        assert isinstance(name, str)
        if len(name) > 256 or name in seen:
            raise ValueError("OpenAPI tag names must be unique and at most 256 characters")
        seen.add(name)
        description = tag.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError("OpenAPI tag description must be a string")
        external_docs = tag.get("externalDocs")
        if external_docs is not None:
            if not isinstance(external_docs, dict) or set(external_docs) - {
                "url",
                "description",
            }:
                raise ValueError("OpenAPI tag externalDocs is invalid")
            _require_nonempty("tag.externalDocs.url", external_docs.get("url"))
            external_description = external_docs.get("description")
            if external_description is not None and not isinstance(external_description, str):
                raise ValueError("OpenAPI tag externalDocs description must be a string")
        result.append(copy.deepcopy(tag))
    return result


def _dump_document(document: dict[str, Any]) -> bytes:
    rendered = yaml.dump(
        document,
        Dumper=_NoAliasSafeDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=120,
    ).encode("utf-8")
    if len(rendered) > MAX_DOCUMENT_BYTES:
        raise ValueError("OpenAPI document exceeds the 4 MiB domain limit")
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
        raise ValueError("OpenAPI path must be a normalized string beginning with /")
    if not value or len(value) > 2048 or "\x00" in value:
        raise ValueError("OpenAPI path is invalid or oversized")
    return value


def _validate_component_section(value: Any) -> str:
    if not isinstance(value, str) or value not in ALLOWED_COMPONENT_SECTIONS:
        raise ValueError("unsupported OpenAPI component section")
    return value


def _validate_component_name(value: Any) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256 or "\x00" in value:
        raise ValueError("OpenAPI component name is invalid or oversized")


def _validate_target_name(value: str) -> None:
    ArtifactRef(namespace="openapi", name=value)


def _validate_relative_source_path(raw: Any) -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw or "\x00" in raw:
        raise ValueError("source evidence path is invalid")
    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("source evidence path must be normalized and relative")
    path = PurePosixPath(raw)
    if path.is_absolute() or path.as_posix() != raw:
        raise ValueError("source evidence path must be normalized and relative")
    return raw


def _require_nonempty(field: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _bound_targeted_result(value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_TARGETED_RESULT_BYTES:
        raise ValueError("targeted OpenAPI result exceeds the model output limit")


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


def _run_vacuum(source_text: str) -> dict[str, Any]:
    executable = shutil.which("vacuum")
    if executable is None:
        return {
            "available": False,
            "executionError": "Vacuum executable is unavailable",
            "issues": [],
            "truncated": False,
        }
    try:
        process = subprocess.run(
            [executable, "spectral-report", "-i", "-o"],
            input=source_text.encode("utf-8"),
            capture_output=True,
            timeout=VACUUM_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "available": True,
            "executionError": "Vacuum validation timed out",
            "issues": [],
            "truncated": False,
        }
    except OSError:
        return {
            "available": True,
            "executionError": "Vacuum could not be executed",
            "issues": [],
            "truncated": False,
        }
    if process.returncode not in {0, 1}:
        return {
            "available": True,
            "executionError": "Vacuum returned an execution failure",
            "issues": [],
            "truncated": False,
        }
    if not process.stdout or len(process.stdout) > MAX_VACUUM_OUTPUT_BYTES:
        return {
            "available": True,
            "executionError": "Vacuum returned empty or oversized output",
            "issues": [],
            "truncated": False,
        }
    try:
        parsed = json.loads(process.stdout)
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
        _issue_with_snippet(item, source_text)
        for item in parsed
        if isinstance(item, dict)
        and type(item.get("severity")) is int
        and item["severity"] in {0, 1}
    ]
    serious.sort(key=lambda item: item.get("severity", 99))
    truncated = len(serious) > MAX_VALIDATION_ISSUES
    return {
        "available": True,
        "executionError": None,
        "issues": serious[:MAX_VALIDATION_ISSUES],
        "truncated": truncated,
    }


def _issue_with_snippet(issue: dict[str, Any], source_text: str) -> dict[str, Any]:
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
    lines = source_text.splitlines()
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


def _unconfigured_client(allocation_id: str, runtime_settings: RuntimeSettings) -> ArtifactClient:
    del runtime_settings
    return ArtifactClient(allocation_id, _UnavailableTransport())


class _UnavailableTransport:
    async def request(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("Artifact transport is not configured")


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)
