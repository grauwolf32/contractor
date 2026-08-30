from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

import contractor_runtime.toolsets.openapi as openapi_module
from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import ArtifactValue
from contractor_runtime.contracts import (
    API_VERSION,
    ArtifactRef,
    ArtifactWriteResult,
    RuntimeSettings,
)
from contractor_runtime.factories import built_in_factories
from contractor_runtime.toolsets.openapi import (
    MAX_DOCUMENT_BYTES,
    MAX_DOCUMENT_DEPTH,
    OpenAPIToolsetFactory,
    _run_vacuum,
    _validate_json_tree,
)
from contractor_runtime.workspace import AllocationWorkspace

SECRET = "openapi-tool-recognizable-secret"


def test_openapi_tools_build_validate_and_expose_exact_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        source = tmp_path / "source" / "src"
        source.mkdir(parents=True)
        (source / "app.py").write_text("@app.get('/health')\ndef health(): ...\n")
        client = MemoryArtifactClient()
        state = WorkerState()
        tools = await make_tools(tmp_path, client, state, namespace="openapi")
        monkeypatch.setattr(openapi_module, "_run_vacuum", clean_vacuum)

        initialized = await tools["initialize_openapi"](
            title=f"Service {SECRET}", description="Generated from implementation"
        )
        first_revision = initialized["artifact"]["revision"]

        info = await tools["set_openapi_info"](
            title="Health Service",
            framework="FastAPI",
            code_language="Python",
        )
        servers = await tools["set_openapi_servers"](
            [{"url": "https://api.example.test", "description": "Production"}]
        )
        tags = await tools["set_openapi_tags"](
            [{"name": "health", "description": "Health operations"}]
        )
        component = await tools["upsert_openapi_component"](
            "schemas",
            "HealthResponse",
            {
                "type": "object",
                "required": ["ok"],
                "properties": {"ok": {"type": "boolean"}},
            },
            ["src/app.py"],
        )
        path = await tools["upsert_openapi_path"](
            "/health",
            {
                "get": {
                    "operationId": "getHealth",
                    "responses": {
                        "200": {
                            "description": "Healthy",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HealthResponse"}
                                }
                            },
                        }
                    },
                }
            },
            ["src/app.py"],
        )

        revisions = [
            first_revision,
            info["artifact"]["revision"],
            servers["artifact"]["revision"],
            tags["artifact"]["revision"],
            component["artifact"]["revision"],
            path["artifact"]["revision"],
        ]
        assert len(set(revisions)) == 6
        assert client.write_count == 6
        assert (await tools["get_openapi_info"]())["info"]["x-framework"] == "FastAPI"
        assert (await tools["list_openapi_servers"]())["servers"][0]["url"].startswith("https://")
        assert (await tools["list_openapi_tags"]())["tags"] == [
            {"name": "health", "description": "Health operations"}
        ]
        assert (await tools["list_openapi_paths"]())["paths"] == ["/health"]
        assert (await tools["get_openapi_path"]("/health"))["pathItem"]["x-path-files"] == [
            "src/app.py"
        ]
        assert (await tools["list_openapi_components"]("schemas"))["components"] == [
            "HealthResponse"
        ]
        assert (await tools["get_openapi_component"]("schemas", "HealthResponse"))["component"][
            "x-component-files"
        ] == ["src/app.py"]

        validation = await tools["validate_openapi"]()
        assert validation["valid"]
        assert validation["issues"] == []
        rendered = await tools["read_openapi_document"]()
        document = yaml.safe_load(rendered["document"])
        assert "pathItems" not in document["components"]
        assert document["paths"]["/health"]["x-path-files"] == ["src/app.py"]
        assert document["components"]["schemas"]["HealthResponse"]["x-component-files"] == [
            "src/app.py"
        ]
        current = await client.read_artifact(ArtifactRef(namespace="openapi", name="openapi"))
        assert current.data == rendered["document"].encode()

        observed = {
            (ref.namespace, ref.name, ref.revision)
            for tool in tools.values()
            for ref in tool.known_exact_refs
        }
        assert ("openapi", "openapi", path["artifact"]["revision"]) in observed
        serialized_metrics = repr(state.metrics.snapshot())
        assert SECRET not in serialized_metrics
        assert "operationId" not in serialized_metrics
        assert state.metrics.counters["tool_calls.validate_openapi"] == 1

    asyncio.run(scenario())


def test_minimal_openapi_30_shape_is_clean_with_real_vacuum(tmp_path: Path) -> None:
    if openapi_module.shutil.which("vacuum") is None:
        pytest.skip("Vacuum executable is unavailable")

    async def scenario() -> None:
        source = tmp_path / "source" / "src"
        source.mkdir(parents=True)
        (source / "app.py").write_text("@app.get('/health')\ndef health(): ...\n")
        tools = await make_tools(
            tmp_path, MemoryArtifactClient(), WorkerState(), namespace="openapi"
        )
        await tools["initialize_openapi"]("Health Service", description="HTTP health endpoint")
        await tools["set_openapi_servers"]([{"url": ".", "description": "Current origin"}])
        await tools["set_openapi_tags"]([{"name": "health", "description": "Health operations"}])
        await tools["upsert_openapi_path"](
            "/health",
            {
                "get": {
                    "summary": "Read service health",
                    "description": "Returns the current service health.",
                    "operationId": "getHealth",
                    "tags": ["health"],
                    "responses": {"200": {"description": "Service is healthy."}},
                }
            },
            ["src/app.py"],
        )

        validation = await tools["validate_openapi"]()

        assert validation["valid"], validation
        assert validation["issues"] == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("media_type", "serialize"),
    [
        ("application/json", lambda value: json.dumps(value).encode()),
        ("application/yaml", lambda value: yaml.safe_dump(value).encode()),
    ],
)
def test_exact_seed_is_copied_and_later_stage_resumes_same_binding(
    tmp_path: Path, media_type: str, serialize: Any
) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        seed_bytes = serialize(minimal_document("Seed API"))
        seed = client.seed("inputs", "existing-openapi", media_type, seed_bytes)
        first_stage = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")

        with pytest.raises(ValueError, match="requires an exact revision"):
            await first_stage["load_openapi"]("inputs", "existing-openapi")
        copied = await first_stage["load_openapi"]("inputs", "existing-openapi", seed.revision)
        assert copied["copied"] and copied["changed"]
        assert copied["artifact"]["namespace"] == "openapi"
        assert client.history[("inputs", "existing-openapi", seed.revision)].data == seed_bytes
        writes_after_copy = client.write_count

        later_stage = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        resumed = await later_stage["load_openapi"](
            "openapi", "openapi", copied["artifact"]["revision"]
        )
        assert not resumed["copied"] and not resumed["changed"]
        assert client.write_count == writes_after_copy
        updated = await later_stage["set_openapi_info"]("Updated API")
        assert updated["artifact"]["revision"] != copied["artifact"]["revision"]
        assert client.write_count == writes_after_copy + 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        b"openapi: 3.0.3\ninfo:\n  title: first\n  title: second\n  version: 1\npaths: {}\n",
        b"openapi: 3.0.3\ninfo: &info\n  title: Demo\n  version: '1'\ncopy: *info\npaths: {}\n",
        b"openapi: 3.0.3\ninfo:\n  title: Demo\n  version: .nan\npaths: {}\n",
        b"openapi: 2.0\ninfo:\n  title: Demo\n  version: '1'\npaths: {}\n",
        (
            b"openapi: 3.0.3\ninfo:\n  title: Demo\n  version: '1'\npaths: {}\n"
            b"components:\n  pathItems: {}\n"
        ),
        b"openapi: 3.0.3\ninfo:\n  title: Demo\n  version: '1'\npaths: {}\n1: invalid-key\n",
        b"[]\n",
    ],
)
def test_malformed_or_unsafe_seed_never_creates_target(tmp_path: Path, payload: bytes) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        seed = client.seed("inputs", "seed", "application/yaml", payload)
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        with pytest.raises(ValueError):
            await tools["load_openapi"]("inputs", "seed", seed.revision)
        assert ("openapi", "openapi") not in client.bindings
        assert client.write_count == 0

    asyncio.run(scenario())


def test_parser_enforces_byte_depth_and_item_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def oversized_scenario() -> None:
        client = MemoryArtifactClient()
        seed = client.seed(
            "inputs",
            "oversized",
            "application/yaml",
            b"x" * (MAX_DOCUMENT_BYTES + 1),
        )
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        with pytest.raises(ValueError, match="4 MiB"):
            await tools["load_openapi"]("inputs", "oversized", seed.revision)
        assert client.write_count == 0

    asyncio.run(oversized_scenario())

    nested: Any = "leaf"
    for _ in range(MAX_DOCUMENT_DEPTH + 1):
        nested = [nested]
    with pytest.raises(ValueError, match="nesting depth"):
        _validate_json_tree(nested)

    monkeypatch.setattr(openapi_module, "MAX_DOCUMENT_ITEMS", 3)
    with pytest.raises(ValueError, match="item limit"):
        _validate_json_tree([1, 2, 3])


def test_invalid_mutations_and_stale_cas_leave_document_unchanged(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "source" / "src"
        source.mkdir(parents=True)
        (source / "app.py").write_text("route = '/health'\n")
        (source / "spec.yaml").write_text("not implementation evidence\n")
        client = MemoryArtifactClient()
        state = WorkerState()
        tools = await make_tools(tmp_path, client, state, namespace="openapi")
        initialized = await tools["initialize_openapi"]("Demo")
        revision = initialized["artifact"]["revision"]
        writes = client.write_count

        failures = [
            (ValueError, "at least one", ("/empty", valid_path_item(), [])),
            (
                ValueError,
                "implementation source",
                ("/banned", valid_path_item(), ["src/spec.yaml"]),
            ),
            (ValueError, "does not exist", ("/missing", valid_path_item(), ["src/missing.py"])),
            (ValueError, "responses", ("/invalid", {"get": {}}, ["src/app.py"])),
            (
                ValueError,
                "unresolved",
                (
                    "/unresolved",
                    valid_path_item("#/components/schemas/Missing"),
                    ["src/app.py"],
                ),
            ),
            (
                ValueError,
                "local JSON pointer",
                (
                    "/remote",
                    valid_path_item("https://example.test/schema.yaml"),
                    ["src/app.py"],
                ),
            ),
        ]
        for error_type, message, arguments in failures:
            with pytest.raises(error_type, match=message):
                await tools["upsert_openapi_path"](*arguments)
            assert client.write_count == writes
            assert client.bindings[("openapi", "openapi")].revision == revision

        externally_advanced = await client.write_artifact(
            ArtifactRef(namespace="openapi", name="openapi"),
            data=yaml.safe_dump(minimal_document("Concurrent update")).encode(),
            media_type="application/yaml",
            expected_revision=revision,
        )
        with pytest.raises(ValueError, match="CAS"):
            await tools["set_openapi_info"]("Stale model update")
        current = client.bindings[("openapi", "openapi")]
        assert current.revision == externally_advanced.artifact.revision
        assert yaml.safe_load(current.data)["info"]["title"] == "Concurrent update"
        assert "operationId" not in repr(state.metrics.snapshot())

    asyncio.run(scenario())


def test_validation_checks_seed_provenance_and_never_treats_missing_vacuum_as_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        source = tmp_path / "source" / "src"
        source.mkdir(parents=True)
        (source / "app.py").write_text("route = '/health'\n")
        client = MemoryArtifactClient()
        document = minimal_document("Seed")
        document["paths"]["/health"] = valid_path_item()
        document["paths"]["/health"]["x-path-files"] = ["src/missing.py"]
        seed = client.seed("inputs", "seed", "application/yaml", yaml.safe_dump(document).encode())
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        await tools["load_openapi"]("inputs", "seed", seed.revision)
        monkeypatch.setattr(openapi_module, "_run_vacuum", clean_vacuum)
        invalid = await tools["validate_openapi"]()
        assert not invalid["valid"]
        assert "does not exist" in invalid["structuralErrors"][0]

        await tools["initialize_openapi"](
            "Replacement", expected_revision=invalid["artifact"]["revision"]
        )
        monkeypatch.setattr(
            openapi_module,
            "_run_vacuum",
            lambda _source: {
                "available": False,
                "executionError": "Vacuum executable is unavailable",
                "issues": [],
                "truncated": False,
            },
        )
        unavailable = await tools["validate_openapi"]()
        assert not unavailable["valid"]
        assert not unavailable["validatorAvailable"]
        assert unavailable["validatorExecutionError"]

    asyncio.run(scenario())


def test_schema_mutations_replace_refs_atomically_and_reject_vacuum_shape_errors(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        source = tmp_path / "source" / "src"
        source.mkdir(parents=True)
        (source / "app.py").write_text("value = 'example'\n")
        client = MemoryArtifactClient()
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        await tools["initialize_openapi"]("Schema Service")
        await tools["upsert_openapi_component"](
            "schemas", "StringValue", {"type": "string"}, ["src/app.py"]
        )
        await tools["upsert_openapi_component"](
            "schemas",
            "Envelope",
            {
                "type": "object",
                "properties": {"value": {"type": "string", "description": "Inline value"}},
            },
            ["src/app.py"],
        )
        replaced = await tools["upsert_openapi_component"](
            "schemas",
            "Envelope",
            {"properties": {"value": {"$ref": "#/components/schemas/StringValue"}}},
            ["src/app.py"],
        )
        envelope = (await tools["get_openapi_component"]("schemas", "Envelope"))["component"]
        assert envelope["properties"]["value"] == {"$ref": "#/components/schemas/StringValue"}

        revision = replaced["artifact"]["revision"]
        writes = client.write_count
        with pytest.raises(ValueError, match="type must not be null"):
            await tools["upsert_openapi_component"](
                "schemas",
                "Envelope",
                {
                    "properties": {
                        "value": {
                            "$ref": "#/components/schemas/StringValue",
                            "type": None,
                        }
                    }
                },
                ["src/app.py"],
            )
        with pytest.raises(ValueError, match=r"\$ref objects cannot contain sibling"):
            await tools["upsert_openapi_component"](
                "schemas",
                "Envelope",
                {
                    "properties": {
                        "value": {
                            "$ref": "#/components/schemas/StringValue",
                            "description": "Ignored sibling",
                        }
                    }
                },
                ["src/app.py"],
            )
        with pytest.raises(ValueError, match="at least two schemas"):
            await tools["upsert_openapi_component"](
                "schemas",
                "Envelope",
                {"properties": {"value": {"anyOf": [{"type": "string"}]}}},
                ["src/app.py"],
            )
        assert client.write_count == writes
        assert client.bindings[("openapi", "openapi")].revision == revision

    asyncio.run(scenario())


def test_vacuum_adapter_bounds_and_orders_serious_issues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, kwargs))
        issues = [
            {"severity": 2, "message": "info"},
            {
                "severity": 1,
                "message": "warning",
                "range": {
                    "start": {"line": 2, "character": 1},
                    "end": {"line": 2, "character": 5},
                },
            },
            {
                "severity": 0,
                "message": "error",
                "range": {
                    "start": {"line": 1, "character": 1},
                    "end": {"line": 1, "character": 6},
                },
            },
            {"severity": 3, "message": "hint"},
        ]
        return subprocess.CompletedProcess(command, 1, json.dumps(issues).encode(), b"")

    monkeypatch.setattr(openapi_module.shutil, "which", lambda _name: "/opt/bin/vacuum")
    monkeypatch.setattr(openapi_module.subprocess, "run", run)
    result = _run_vacuum("alpha\nbeta\n")
    assert result["executionError"] is None
    assert [item["severity"] for item in result["issues"]] == [0, 1]
    assert [item["snippet"] for item in result["issues"]] == ["alpha", "beta"]
    assert all("range" not in item for item in result["issues"])
    assert calls[0][0] == ["/opt/bin/vacuum", "spectral-report", "-i", "-o"]
    assert calls[0][1]["input"] == b"alpha\nbeta\n"
    assert calls[0][1]["timeout"] == 30
    assert "shell" not in calls[0][1]


def test_vacuum_adapter_reports_unavailable_timeout_and_bad_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openapi_module.shutil, "which", lambda _name: None)
    assert not _run_vacuum("openapi: 3.0.3")["available"]

    monkeypatch.setattr(openapi_module.shutil, "which", lambda _name: "/bin/vacuum")

    def timeout(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired("vacuum", 30)

    monkeypatch.setattr(openapi_module.subprocess, "run", timeout)
    assert "timed out" in _run_vacuum("openapi: 3.0.3")["executionError"]

    monkeypatch.setattr(
        openapi_module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, b"not-json", b""),
    )
    assert "invalid JSON" in _run_vacuum("openapi: 3.0.3")["executionError"]


def test_factory_rejects_unknown_tools_and_builtin_registry_matches(tmp_path: Path) -> None:
    async def scenario() -> None:
        factory = OpenAPIToolsetFactory(lambda _allocation, _settings: MemoryArtifactClient())
        selected = await factory.create_selected(
            selected=["get_openapi_info"],
            allocation_id="allocation-1",
            run_id="run-1",
            namespace="openapi",
            runtime_settings=runtime_settings(),
            workspace=AllocationWorkspace(root=tmp_path.parent, path=tmp_path),
            state=WorkerState(),
        )
        assert set(selected) == {"get_openapi_info"}
        with pytest.raises(ValueError, match="unknown selected tools"):
            await factory.create_selected(
                selected=["execute_openapi"],
                allocation_id="allocation-1",
                run_id="run-1",
                namespace="openapi",
                runtime_settings=runtime_settings(),
                workspace=AllocationWorkspace(root=tmp_path.parent, path=tmp_path),
                state=WorkerState(),
            )

    asyncio.run(scenario())
    assert built_in_factories(tmp_path).toolsets["openapi@1"].exported_tools == {
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


async def make_tools(
    tmp_path: Path,
    client: MemoryArtifactClient,
    state: WorkerState,
    *,
    namespace: str,
) -> dict[str, Any]:
    factory = OpenAPIToolsetFactory(lambda _allocation, _settings: client)
    selected = await factory.create_selected(
        selected=sorted(factory.exported_tools),
        allocation_id="allocation-1",
        run_id="run-1",
        namespace=namespace,
        runtime_settings=runtime_settings(),
        workspace=AllocationWorkspace(root=tmp_path.parent, path=tmp_path),
        state=state,
    )
    return dict(selected)


def runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        llmGatewayUrl="https://gateway.example/v1",
        llmGatewayToken=SECRET,
        artifactApiUrl="https://control.example/private/v1",
        requestTimeoutSeconds=30,
    )


def minimal_document(title: str) -> dict[str, Any]:
    return {
        "openapi": "3.0.3",
        "info": {"title": title, "version": "1.0.0"},
        "paths": {},
        "components": {},
    }


def valid_path_item(ref: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object"}
    if ref is not None:
        schema = {"$ref": ref}
    return {
        "get": {
            "responses": {
                "200": {
                    "description": "OK",
                    "content": {"application/json": {"schema": schema}},
                }
            }
        }
    }


def clean_vacuum(_source: str) -> dict[str, Any]:
    return {
        "available": True,
        "executionError": None,
        "issues": [],
        "truncated": False,
    }


@dataclass(slots=True)
class StoredArtifact:
    revision: str
    media_type: str
    data: bytes = field(repr=False)


class MemoryArtifactClient:
    def __init__(self) -> None:
        self.bindings: dict[tuple[str, str], StoredArtifact] = {}
        self.history: dict[tuple[str, str, str], StoredArtifact] = {}
        self._known: dict[tuple[str, str, str], ArtifactRef] = {}
        self._next_revision = 1
        self.write_count = 0

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return tuple(self._known.values())

    def seed(self, namespace: str, name: str, media_type: str, data: bytes) -> ArtifactRef:
        return self._store(namespace, name, media_type, data)

    async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue:
        if ref.revision is None:
            stored = self.bindings[(ref.namespace, ref.name)]
        else:
            stored = self.history[(ref.namespace, ref.name, ref.revision)]
        exact = ArtifactRef(
            namespace=ref.namespace,
            name=ref.name,
            revision=stored.revision,
        )
        self._remember(exact)
        return ArtifactValue(artifact=exact, media_type=stored.media_type, data=stored.data)

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> ArtifactWriteResult:
        key = (target.namespace, target.name)
        current = self.bindings.get(key)
        if expected_revision is None:
            if current is not None:
                raise ValueError("create-only CAS precondition failed")
        elif current is None or current.revision != expected_revision:
            raise ValueError("update CAS precondition failed")
        exact = self._store(target.namespace, target.name, media_type, data)
        self.write_count += 1
        return ArtifactWriteResult(
            apiVersion=API_VERSION,
            artifact=exact,
            mediaType=media_type,
            size=len(data),
        )

    def _store(self, namespace: str, name: str, media_type: str, data: bytes) -> ArtifactRef:
        revision = f"revision-{self._next_revision}"
        self._next_revision += 1
        stored = StoredArtifact(revision=revision, media_type=media_type, data=data)
        self.bindings[(namespace, name)] = stored
        self.history[(namespace, name, revision)] = stored
        exact = ArtifactRef(namespace=namespace, name=name, revision=revision)
        self._remember(exact)
        return exact

    def _remember(self, ref: ArtifactRef) -> None:
        assert ref.revision is not None
        self._known[(ref.namespace, ref.name, ref.revision)] = ref
