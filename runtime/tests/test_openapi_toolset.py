from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import re
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import yaml
from document_preparation_probes import (
    PreparationGate,
    loop_callback,
    observe_threads,
    yield_to_loop,
)
from document_write_faults import FaultInjectingArtifactClient
from fakes.worker_state import WorkerState
from shared_config_defaults import MAX_ARTIFACT_BYTES

import contractor_runtime.toolsets.openapi.tools as openapi_module
from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactResponseLimitError,
    ArtifactValue,
)
from contractor_runtime.contracts import (
    API_VERSION,
    ArtifactRef,
    ArtifactWriteResult,
    RuntimeSettings,
)
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.factories import built_in_factories
from contractor_runtime.projectfs.storage import WorkspaceSnapshot, WorkspaceTextFile
from contractor_runtime.toolsets.openapi.tools import (
    DOCUMENT_BYTES_MESSAGE,
    DOCUMENT_DEPTH_MESSAGE,
    DOCUMENT_NODES_MESSAGE,
    MAX_DOCUMENT_BYTES,
    MAX_DOCUMENT_DEPTH,
    MAX_DOCUMENT_NODES,
    OpenAPIToolsetFactory,
    _run_vacuum,
    _validate_json_tree,
)
from contractor_runtime.worker.instrumentation import _safe_tool_response
from contractor_runtime.workspace import AllocationWorkspace

SECRET = "openapi-tool-recognizable-secret"
DOCUMENT_BOUNDS = Path(__file__).resolve().parents[2] / "api/testdata/openapi-document-bounds.json"
OPENAPI_YAML_CASES = (
    Path(__file__).resolve().parents[2] / "api/testdata/openapi-yaml-scalar-cases.json"
)
OPENAPI_JSON_CASES = (
    Path(__file__).resolve().parents[2] / "api/testdata/openapi-json-escape-cases.json"
)


def test_openapi_tools_build_validate_and_expose_exact_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        source = tmp_path / "project" / "src"
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


def test_openapi_rejects_missing_workspace_before_creating_clients(tmp_path: Path) -> None:
    async def scenario() -> None:
        scratch = tmp_path / "source"
        scratch.mkdir()
        (scratch / "app.py").write_text("def handler(): pass\n")

        def forbidden_client(_allocation, _settings):
            raise AssertionError("missing workspace reached client construction")

        factory = OpenAPIToolsetFactory(forbidden_client)
        with pytest.raises(ValueError, match="requires a project workspace"):
            await factory.create_selected(
                selected=["upsert_openapi_path"],
                allocation_id="allocation-1",
                run_id="run-1",
                namespace="openapi",
                runtime_settings=runtime_settings(),
                workspace=AllocationWorkspace(root=tmp_path.parent, path=tmp_path),
                resources=WorkerState().resources,
            )

    asyncio.run(scenario())


def test_project_workspace_is_authoritative_for_openapi_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        scratch_source = tmp_path / "source"
        scratch_source.mkdir()
        (scratch_source / "scratch-only.py").write_text("legacy = True\n")
        project = StaticWorkspaceReader({"src/app.py": "@app.get('/health')\ndef health(): ...\n"})
        client = MemoryArtifactClient()
        tools = await make_tools(
            tmp_path,
            client,
            WorkerState(),
            namespace="openapi",
            project_workspace=project,
        )
        monkeypatch.setattr(openapi_module, "_run_vacuum", clean_vacuum)

        await tools["initialize_openapi"](title="Workspace API")
        component = await tools["upsert_openapi_component"](
            "schemas",
            "HealthResponse",
            {"type": "object", "properties": {"ok": {"type": "boolean"}}},
            ["src/app.py"],
        )
        assert component["name"] == "HealthResponse"
        validation = await tools["validate_openapi"]()
        assert validation["valid"]
        assert project.file_paths_calls == 2
        assert project.snapshot_calls == 0

        with pytest.raises(ValueError, match="source evidence file does not exist"):
            await tools["upsert_openapi_component"](
                "schemas",
                "LegacyOnly",
                {"type": "object"},
                ["scratch-only.py"],
            )

    asyncio.run(scenario())


def test_minimal_openapi_30_shape_is_clean_with_real_vacuum(tmp_path: Path) -> None:
    if openapi_module.shutil.which("vacuum") is None:
        pytest.skip("Vacuum executable is unavailable")

    async def scenario() -> None:
        source = tmp_path / "project" / "src"
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


@pytest.mark.parametrize("same_binding", [False, True])
@pytest.mark.parametrize(
    ("fragment", "key", "value"),
    [
        (r'"x-\/url":"https:\/\/example.test"', "x-/url", "https://example.test"),
        (r'"x-\ud83d\ude00":"\ud83d\ude00"', "x-😀", "😀"),
    ],
)
def test_json_seed_escapes_survive_load_edit_and_yaml_reload(
    tmp_path: Path, same_binding: bool, fragment: str, key: str, value: str
) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        namespace = "openapi" if same_binding else "inputs"
        payload = json.dumps(minimal_document("Seed"))[:-1] + "," + fragment + "}"
        seed = client.seed(namespace, "openapi", "application/json", payload.encode())
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        loaded = await tools["load_openapi"](namespace, "openapi", seed.revision)
        assert loaded["copied"] is True
        assert tools["load_openapi"]._session._document[key] == value
        await tools["set_openapi_info"]("Edited")
        written = client.bindings[("openapi", "openapi")]
        assert written.media_type == "application/yaml"
        assert yaml.safe_load(written.data)[key] == value
        resumed = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        writes = client.write_count
        assert (await resumed["load_openapi"]("openapi", "openapi"))["copied"] is False
        assert resumed["load_openapi"]._session._document[key] == value
        assert (await resumed["get_openapi_info"]())["info"]["title"] == "Edited"
        assert client.write_count == writes

    asyncio.run(scenario())


def test_openapi_tools_enforce_shared_memory_source_target_and_ref_visibility(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        payload = yaml.safe_dump(minimal_document("Seed API")).encode()
        hidden = client.seed("analysis", "memory.hidden", "application/yaml", payload)
        allowed = client.seed("inputs", "memory.workflow_input", "application/yaml", payload)
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        session = tools["load_openapi"]._session
        calls_before = (client.read_count, client.write_count)

        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["load_openapi"](
                hidden.namespace,
                hidden.name,
                hidden.revision,
                target_name="safe_target",
            )
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["load_openapi"](
                allowed.namespace,
                allowed.name,
                allowed.revision,
                target_name="memory.hidden_target",
            )
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["initialize_openapi"]("Blocked", target_name="memory.hidden_target")
        assert (client.read_count, client.write_count) == calls_before
        assert session._document is None and session._target_name is None

        copied = await tools["load_openapi"](allowed.namespace, allowed.name, allowed.revision)
        assert copied["artifact"]["namespace"] == "openapi"
        observed = {
            (ref.namespace, ref.name) for tool in tools.values() for ref in tool.known_exact_refs
        }
        assert (hidden.namespace, hidden.name) not in observed
        assert (allowed.namespace, allowed.name) in observed

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
        b"openapi: 3.0.3\ninfo:\n  title: Demo\n  version: '1'\npaths: {}\n? [1]\n: invalid-key\n",
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


VALID_31_SEED = b"""\
openapi: 3.1.0
info:
  title: Dated
  version: 2023-01-01
paths:
  /items:
    x-path-files: [src/app.py]
    get:
      responses:
        200:
          description: OK
          content:
            application/json:
              schema:
                type: array
                prefixItems: [{type: string}]
                items: false
        404:
          description: Missing
components:
  x-internal:
    owner: platform
  schemas:
    Open:
      x-component-files: [src/app.py]
      type: object
      properties:
        anything: true
        nothing: false
        on: {type: string}
      allOf: [true]
"""


def test_json_compatible_yaml_seed_with_31_boolean_schemas_loads_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        source = tmp_path / "project" / "src"
        source.mkdir(parents=True)
        (source / "app.py").write_text("route = '/items'\n")
        client = MemoryArtifactClient()
        seed = client.seed("inputs", "seed", "application/yaml", VALID_31_SEED)
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        await tools["load_openapi"]("inputs", "seed", seed.revision)

        info = await tools["get_openapi_info"]()
        assert info["info"]["version"] == "2023-01-01"
        path = await tools["get_openapi_path"]("/items")
        assert list(path["pathItem"]["get"]["responses"]) == ["200", "404"]
        schema = (await tools["get_openapi_component"]("schemas", "Open"))["component"]
        assert schema["properties"] == {
            "anything": True,
            "nothing": False,
            "on": {"type": "string"},
        }
        monkeypatch.setattr(openapi_module, "_run_vacuum", clean_vacuum)
        validated = await tools["validate_openapi"]()
        assert validated["structuralErrors"] == []
        assert validated["valid"] is True

        # The stored target still reads back with the same JSON meaning.
        stored = client.bindings[("openapi", "openapi")].data
        assert b"'2023-01-01'" in stored
        reloaded = openapi_module._parse_document(stored)
        assert reloaded["info"]["version"] == "2023-01-01"
        assert reloaded["components"]["x-internal"] == {"owner": "platform"}
        assert "200" in reloaded["paths"]["/items"]["get"]["responses"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("fragment", "message"),
    [
        ("      items: false\n", "schema must be an object"),
        ("      properties: {a: true}\n", "schema must be an object"),
        ("      allOf: [true]\n", "at least one schema"),
    ],
)
def test_boolean_schemas_remain_invalid_for_openapi_30(fragment: str, message: str) -> None:
    payload = (
        "openapi: 3.0.3\ninfo: {title: Demo, version: '1'}\npaths: {}\n"
        "components:\n  schemas:\n    Value:\n      type: array\n" + fragment
    ).encode()
    with pytest.raises(ValueError, match=message):
        openapi_module._validate_document(
            openapi_module._parse_document(payload), require_provenance=False
        )


def test_yaml_keys_are_json_strings_without_weakening_structure_checks() -> None:
    base = "openapi: 3.0.3\ninfo: {title: Demo, version: '1'}\npaths: {}\n"
    parsed = openapi_module._parse_document(
        (base + "x-codes: {200: ok, 1.5: half, true: kept, null: none}\n").encode()
    )
    assert parsed["x-codes"] == {"200": "ok", "1.5": "half", "true": "kept", "null": "none"}
    for payload, message in (
        (base + "x-codes: {200: a, '200': b}\n", "duplicate key"),
        (base + "? [1, 2]\n: sequence-key\n", "keys must be scalar strings"),
        (base + "!!binary aGk=: binary-key\n", "tags other than"),
        (base + "!!int 200: int-key\n", "keys must be strings"),
        (base + "released: !!timestamp 2023-01-01\n", "tags other than"),
        (base + "x-m: {<<: {a: 1}, b: 2}\n", "merge keys"),
        (base + "x-v: 0x10000000000000000\n", "quote a string that is spelled like a number"),
        (base + "x-v: !!bool yes\n", "does not match its explicit tag"),
        ("%YAML 1.2\n---\n" + base, "version directives"),
        (base + 'x-v: "\\ud83d"\n', "surrogate code points"),
        (base + 'x-v: "\\U00110000"\n', "not valid YAML or JSON"),
        (base + "x-v: \x07\n", "not valid YAML or JSON"),
        (base + "x-ref: &a 1\nx-copy: *a\n", "aliases and anchors"),
        (base + "x-only: &unused {a: 1}\n", "aliases and anchors"),
        (base + "x-copy: *missing\n", "aliases and anchors"),
        (base + "&key x-key: value\n", "aliases and anchors"),
        (base + "x-list: [&item 1, 2]\n", "aliases and anchors"),
        # The whole stream is composed before any value is constructed.
        (base + "x-dup: {a: 1, a: 2}\n---\nx-later: &later 1\n", "aliases and anchors"),
    ):
        with pytest.raises(ToolInputError, match=message):
            openapi_module._parse_document(payload.encode())


def openapi_yaml_cases() -> dict[str, Any]:
    fixture = json.loads(OPENAPI_YAML_CASES.read_text(encoding="utf-8"))
    assert set(fixture) == {"description", "values", "keys", "documents", "writes"}
    assert all(fixture[group] for group in ("values", "keys", "documents", "writes"))
    return fixture


def same_json(left: Any, right: Any) -> bool:
    """JSON equality that keeps booleans apart from numbers and the sign of zero."""

    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, int | float) and isinstance(right, int | float):
        return float(left) == float(right) and math.copysign(1, left) == math.copysign(1, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_json(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(map(same_json, left, right))
    return type(left) is type(right) and left == right


def test_shared_openapi_yaml_scalar_cases_match_server_readers() -> None:
    """openapi@1 reads each shared case as scan preparation and the inventory do."""

    fixture = openapi_yaml_cases()
    documents: list[tuple[str, str, dict[str, Any] | None]] = []
    for case in fixture["values"]:
        expected = {"v": case["value"]} if case["verdict"] == "accepted" else None
        documents.append((case["yaml"], "v: " + case["yaml"] + "\n", expected))
    for case in fixture["keys"]:
        expected = {case["member"]: "v"} if case["verdict"] == "accepted" else None
        documents.append((case["yaml"], case["yaml"] + ": v\n", expected))
    for case in fixture["documents"]:
        expected = case["value"] if case["verdict"] == "accepted" else None
        documents.append((case["name"], case["yaml"], expected))
    for label, document, expected in documents:
        assert expected is not None or label  # every case is labelled
        if expected is None:
            with pytest.raises(ToolInputError) as rejected:
                openapi_module._parse_document(document.encode())
            assert rejected.value.code == "tool_input_invalid", label
            assert rejected.value.retryable is False, label
            continue
        assert same_json(openapi_module._parse_document(document.encode()), expected), label


def test_shared_openapi_json_escape_cases_match_server_readers(tmp_path: Path) -> None:
    async def scenario() -> None:
        fixture = json.loads(OPENAPI_JSON_CASES.read_text(encoding="utf-8"))
        assert fixture["cases"]
        for case in fixture["cases"]:
            client = MemoryArtifactClient()
            tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
            await tools["initialize_openapi"]("Before")
            session = tools["load_openapi"]._session
            before = session._document
            writes = client.write_count
            seed = client.seed("inputs", "seed", case["mediaType"], case["document"].encode())
            if not case["accepted"]:
                with pytest.raises(ToolInputError) as rejected:
                    await tools["load_openapi"]("inputs", "seed", seed.revision)
                assert rejected.value.code == "tool_input_invalid", case["name"]
                assert rejected.value.retryable is False, case["name"]
                assert client.write_count == writes, case["name"]
                assert session._document is before, case["name"]
                continue
            await tools["load_openapi"]("inputs", "seed", seed.revision, target_name="copy")
            assert same_json(session._document, case["value"]), case["name"]
            written = client.bindings[("openapi", "copy")]
            assert same_json(openapi_module._parse_document(written.data), case["value"])

    asyncio.run(scenario())


def test_shared_openapi_yaml_writer_emits_the_scalars_every_reader_reads_back() -> None:
    """openapi@1 writes each shared string as the fixture's scalar, value and member name."""

    for case in openapi_yaml_cases()["writes"]:
        string, scalar = case["string"], case["scalar"]
        for document, written in (
            ({"v": string}, f"v: {scalar}\n"),
            ({string: "v"}, f"{scalar}: v\n"),
        ):
            assert openapi_module._dump_document(document).decode() == written, string
            assert openapi_module._parse_document(written.encode()) == document, string
            # YAML 1.1 readers, PyYAML's among them, read the same string.
            assert yaml.safe_load(written) == document, string


def test_mutations_write_strings_every_reader_reads_back_and_bound_numbers(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        source = tmp_path / "project" / "src"
        source.mkdir(parents=True)
        (source / "app.py").write_text("@app.get('/health')\ndef health(): ...\n")
        client = MemoryArtifactClient()
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        await tools["initialize_openapi"](title="Service")
        item = valid_path_item()
        item["x-values"] = {"octal": "0o17", "exponent": "1e3", "flag": "yes", "1:30": 1.5}
        await tools["upsert_openapi_path"]("/health", item, ["src/app.py"])
        stored = client.bindings[("openapi", "openapi")].data
        for written in (b"octal: '0o17'", b"exponent: '1e3'", b"flag: 'yes'", b"'1:30': 1.5"):
            assert written in stored
        reloaded = openapi_module._parse_document(stored)
        assert reloaded["paths"]["/health"]["x-values"] == item["x-values"]

        writes = client.write_count
        for number in (1e300, 2.0**53, -(2.0**53)):
            item["x-values"]["number"] = number
            with pytest.raises(ToolInputError, match="within ±9007199254740991"):
                await tools["upsert_openapi_path"]("/health", item, ["src/app.py"])
        assert client.write_count == writes

    asyncio.run(scenario())


def test_merge_key_seed_at_the_same_binding_is_rejected_before_it_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A YAML seed loaded at its own binding keeps its bytes, so the rule must reject it.

    Scan preparation and the Audit inventory reject a merge key; openapi@1 used
    to merge it, keep the original bytes and validate them.
    """

    async def scenario() -> None:
        vacuum = AsyncMock(side_effect=clean_vacuum)
        monkeypatch.setattr(openapi_module, "_run_vacuum", vacuum)
        client = MemoryArtifactClient()
        payload = (
            b"openapi: 3.0.3\ninfo: {title: Merge, version: 1.0.0}\npaths: {}\n"
            b"components: {}\nx-defaults: {<<: {owner: platform}, tier: internal}\n"
        )
        client.seed("openapi", "openapi", "application/yaml", payload)
        writes = client.write_count
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        with pytest.raises(ToolInputError) as rejected:
            await tools["load_openapi"]("openapi", "openapi")
        assert _safe_tool_response("load_openapi", rejected.value)["error"] == {
            "code": "tool_input_invalid",
            "message": "OpenAPI YAML merge keys (<<) are not supported",
            "retryable": False,
        }
        with pytest.raises(ToolInputError, match="must be called first"):
            await tools["validate_openapi"]()
        assert client.write_count == writes
        assert client.bindings[("openapi", "openapi")].data == payload
        vacuum.assert_not_awaited()

    asyncio.run(scenario())


def test_components_accept_extensions_but_not_unknown_sections() -> None:
    document = minimal_document("Extensions")
    document["components"] = {"x-owner": "platform", "x-flags": ["internal"]}
    openapi_module._validate_document(document, require_provenance=True)
    document["components"]["schema"] = {}
    with pytest.raises(ValueError, match="unsupported OpenAPI component section"):
        openapi_module._validate_document(document, require_provenance=False)


def test_local_ref_array_index_rejects_non_ascii_digits() -> None:
    document = minimal_document("Refs")
    document["x-values"] = [{"type": "string"}]
    document["x-ref"] = {"$ref": "#/x-values/0"}
    openapi_module._validate_document(document, require_provenance=False)
    # Superscript two, Arabic-Indic one and fullwidth zero all satisfy isdigit().
    for token in (chr(0x00B2), chr(0x0661), chr(0xFF10)):
        document["x-ref"] = {"$ref": f"#/x-values/{token}"}
        with pytest.raises(ValueError, match="unresolved"):
            openapi_module._validate_document(document, require_provenance=False)


def test_path_and_component_listings_fit_the_model_output_limit(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        document = minimal_document("Long paths")
        for index in range(openapi_module.MAX_LIST_ITEMS):
            path = f"/{index:04d}/" + "p" * 2000
            document["paths"][path] = valid_path_item()
        document["components"]["schemas"] = {
            f"{index:04d}" + chr(0x1F600) * 252: {"type": "string"}
            for index in range(openapi_module.MAX_LIST_ITEMS)
        }
        # Unescaped UTF-8 keeps the seed within the 2 MiB document bound while
        # each listing still exceeds the model output limit.
        payload = yaml.safe_dump(document, allow_unicode=True).encode()
        assert len(payload) <= openapi_module.MAX_DOCUMENT_BYTES
        seed = client.seed("inputs", "seed", "application/yaml", payload)
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        await tools["load_openapi"]("inputs", "seed", seed.revision)

        paths = await tools["list_openapi_paths"]()
        components = await tools["list_openapi_components"]("schemas")
        for result, key in ((paths, "paths"), (components, "components")):
            encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
            assert len(encoded.encode("utf-8")) <= openapi_module.MAX_TARGETED_RESULT_BYTES
            assert 0 < len(result[key]) < openapi_module.MAX_LIST_ITEMS
            assert result["total"] == openapi_module.MAX_LIST_ITEMS
            assert result["truncated"] is True
            assert result[key] == sorted(result[key])
        assert paths["paths"][0].startswith("/0000/")

    asyncio.run(scenario())


@pytest.mark.parametrize("mutation", ["path", "component", "servers", "tags"])
def test_mutation_depth_rejection_preserves_the_published_document(
    tmp_path: Path, mutation: str
) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        initialized = await tools["initialize_openapi"](title="Service")
        writes = client.write_count
        nested: Any = "leaf"
        for _ in range(600):
            nested = {"child": nested}
        calls = {
            "path": lambda: tools["upsert_openapi_path"](
                "/health", {"x-deep": nested}, ["src/app.py"]
            ),
            "component": lambda: tools["upsert_openapi_component"](
                "schemas", "Deep", {"x-deep": nested}, ["src/app.py"]
            ),
            "servers": lambda: tools["set_openapi_servers"](
                [{"url": "https://example.test", "x-deep": nested}]
            ),
            "tags": lambda: tools["set_openapi_tags"]([{"name": "health", "x-deep": nested}]),
        }
        with pytest.raises(ToolInputError, match="maximum nesting depth") as rejected:
            await calls[mutation]()
        assert _safe_tool_response("mutation", rejected.value)["error"]["retryable"] is False
        assert client.write_count == writes
        assert (await tools["get_openapi_info"]())["artifact"] == initialized["artifact"]

    asyncio.run(scenario())


def test_parser_enforces_byte_depth_and_node_limits(
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
        with pytest.raises(ToolInputError, match="2 MiB"):
            await tools["load_openapi"]("inputs", "oversized", seed.revision)
        assert client.read_limits == [MAX_DOCUMENT_BYTES]
        assert client.write_count == 0

    asyncio.run(oversized_scenario())
    with pytest.raises(ToolInputError, match="2 MiB"):
        openapi_module._parse_document(b"x" * (MAX_DOCUMENT_BYTES + 1))

    # The top-level value is at depth 1, so the leaf of 63 nested arrays is at
    # the bound and one more array puts it beyond.
    nested: Any = "leaf"
    for _ in range(MAX_DOCUMENT_DEPTH - 1):
        nested = [nested]
    _validate_json_tree(nested)
    with pytest.raises(ToolInputError, match="nesting depth"):
        _validate_json_tree([nested])
    keyed: Any = "leaf"
    for _ in range(MAX_DOCUMENT_DEPTH - 1):
        keyed = {"a": keyed}
    _validate_json_tree(keyed)
    with pytest.raises(ToolInputError, match="nesting depth"):
        _validate_json_tree({"a": keyed})

    # Member names are nodes: {"a": 1} is three nodes although it holds two values.
    monkeypatch.setattr(openapi_module, "MAX_DOCUMENT_NODES", 3)
    _validate_json_tree({"a": 1})
    _validate_json_tree([1, 2])
    with pytest.raises(ToolInputError, match="node limit"):
        _validate_json_tree({"a": 1, "b": 2})
    with pytest.raises(ToolInputError, match="node limit"):
        _validate_json_tree([1, 2, 3])


def test_shared_openapi_document_bounds_match_scan_preparation(tmp_path: Path) -> None:
    """Source parsing agrees with both Server readers; readiness stays separate."""

    fixture = json.loads(DOCUMENT_BOUNDS.read_text(encoding="utf-8"))
    assert fixture["bounds"] == {
        "sourceBytes": MAX_DOCUMENT_BYTES,
        "depth": MAX_DOCUMENT_DEPTH,
        "nodes": MAX_DOCUMENT_NODES,
    }
    assert len(fixture["cases"]) >= 12
    messages = {
        "bytes": DOCUMENT_BYTES_MESSAGE,
        "depth": DOCUMENT_DEPTH_MESSAGE,
        "nodes": DOCUMENT_NODES_MESSAGE,
    }

    async def verdict(document: bytes, media_type: str) -> str | None:
        client = MemoryArtifactClient()
        seed = client.seed("inputs", "bounds", media_type, document)
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        try:
            loaded = await tools["load_openapi"]("inputs", "bounds", seed.revision)
        except ToolInputError as error:
            assert client.write_count == 0
            return str(error)
        assert loaded["copied"] is True
        assert len(client.bindings[("openapi", "openapi")].data) <= MAX_DOCUMENT_BYTES
        return None

    async def scenario() -> None:
        for case in fixture["cases"]:
            for form in ("json", "yaml"):
                document = bounds_document(
                    fixture["prefix"][form], case, compact_json=form == "json"
                )
                label = f"{case['name']}/{form}"
                assert hashlib.sha256(document).hexdigest() == case["sha256"][form], label
                expected = None if case["violation"] is None else messages[case["violation"]]
                media_type = f"application/{form}"
                try:
                    openapi_module._parse_document(document, media_type)
                except ToolInputError as error:
                    assert str(error) == expected, label
                else:
                    assert expected is None, label
                string_case = case.get("string", {})
                if string_case.get("kind") == "path" and string_case["bytes"] > 2048:
                    # SCN-R253 is the shared source-bound contract, not an
                    # authoring-readiness guarantee. Keep this long case: it
                    # parses everywhere while load_openapi's existing path
                    # rule rejects it without writing. This task removes the
                    # inventory-only 512-byte path cap, not authoring policy.
                    assert await verdict(document, media_type) == (
                        "OpenAPI path must be at most 2048 characters without NUL characters"
                    ), label
                else:
                    assert await verdict(document, media_type) == expected, label

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("case", "message"),
    [
        # The size of the document Server preparation rejected in the review.
        ({"pad": {"bytes": 3145826, "fill": "a"}}, DOCUMENT_BYTES_MESSAGE),
        ({"deep": {"kind": "mapping", "depth": 65}}, DOCUMENT_DEPTH_MESSAGE),
        # 50,007 nodes counting values alone, 100,013 with the member names.
        ({"members": 50000}, DOCUMENT_NODES_MESSAGE),
    ],
    ids=["reproduced-3MiB", "depth-65", "member-names"],
)
def test_document_beyond_the_bounds_is_rejected_before_vacuum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: dict[str, Any], message: str
) -> None:
    async def scenario() -> None:
        vacuum = AsyncMock(side_effect=clean_vacuum)
        monkeypatch.setattr(openapi_module, "_run_vacuum", vacuum)
        client = MemoryArtifactClient()
        payload = bounds_document(BOUNDS_YAML_PREFIX, case)
        seed = client.seed("inputs", "seed", "application/yaml", payload)
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        with pytest.raises(ToolInputError) as rejected:
            await tools["load_openapi"]("inputs", "seed", seed.revision)
        assert _safe_tool_response("load_openapi", rejected.value)["error"] == {
            "code": "tool_input_invalid",
            "message": message,
            "retryable": False,
        }
        with pytest.raises(ToolInputError, match="must be called first"):
            await tools["validate_openapi"]()
        assert client.write_count == 0
        vacuum.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("bound", "value", "message"),
    [
        ("MAX_DOCUMENT_BYTES", 64, DOCUMENT_BYTES_MESSAGE),
        ("MAX_DOCUMENT_DEPTH", 2, DOCUMENT_DEPTH_MESSAGE),
        ("MAX_DOCUMENT_NODES", 8, DOCUMENT_NODES_MESSAGE),
    ],
)
def test_validation_checks_the_bounds_before_vacuum_and_keeps_semantic_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound: str, value: int, message: str
) -> None:
    async def scenario() -> None:
        vacuum = AsyncMock(side_effect=clean_vacuum)
        monkeypatch.setattr(openapi_module, "_run_vacuum", vacuum)
        client = MemoryArtifactClient()
        document = minimal_document("Seed")
        document["paths"]["/health"] = valid_path_item()
        seed = client.seed("inputs", "seed", "application/yaml", yaml.safe_dump(document).encode())
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        await tools["load_openapi"]("inputs", "seed", seed.revision)

        # A document within the bounds keeps its semantic validation beside Vacuum.
        checked = await tools["validate_openapi"]()
        assert checked["valid"] is False
        assert checked["structuralErrors"] == ["OpenAPI path /health is missing source provenance"]
        vacuum.assert_awaited_once()

        monkeypatch.setattr(openapi_module, bound, value)
        with pytest.raises(ToolInputError) as rejected:
            await tools["validate_openapi"]()
        assert str(rejected.value) == message
        vacuum.assert_awaited_once()

    asyncio.run(scenario())


def test_mutation_beyond_the_byte_bound_preserves_the_published_document(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        initialized = await tools["initialize_openapi"](title="Service")
        writes = client.write_count
        with pytest.raises(ToolInputError) as rejected:
            await tools["set_openapi_info"]("Service", description="a" * MAX_DOCUMENT_BYTES)
        assert str(rejected.value) == DOCUMENT_BYTES_MESSAGE
        assert client.write_count == writes
        assert (await tools["get_openapi_info"]())["artifact"] == initialized["artifact"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("media_type", "payload"),
    [
        ("application/json", b"[" * 2000 + b"]" * 2000),
        (
            "application/yaml",
            b"openapi: 3.0.3\ninfo: {title: Deep, version: '1'}\npaths: {}\nx-deep: "
            + b"{a: " * 2000
            + b"1"
            + b"}" * 2000
            + b"\n",
        ),
    ],
    ids=["sequences", "mappings"],
)
def test_seed_nested_beyond_the_parser_stack_reports_the_depth_limit(
    tmp_path: Path, media_type: str, payload: bytes
) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        seed = client.seed("inputs", "deep", media_type, payload)
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        with pytest.raises(ToolInputError) as rejected:
            await tools["load_openapi"]("inputs", "deep", seed.revision)
        reply = _safe_tool_response("load_openapi", rejected.value)["error"]
        assert reply == {
            "code": "tool_input_invalid",
            "message": "OpenAPI document exceeds the maximum nesting depth",
            "retryable": False,
        }
        assert client.write_count == 0

    asyncio.run(scenario())


def test_partial_path_updates_validate_after_merge_and_reject_invalid_result(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        source = tmp_path / "project" / "src"
        source.mkdir(parents=True)
        (source / "app.py").write_text("route = '/health'\n")
        client = MemoryArtifactClient()
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        await tools["initialize_openapi"]("Demo")
        await tools["upsert_openapi_path"]("/health", valid_path_item(), ["src/app.py"])
        parameters = [{"name": "verbose", "in": "query", "schema": {"type": "boolean"}}]
        await tools["upsert_openapi_path"]("/health", {"parameters": parameters}, ["src/app.py"])
        await tools["upsert_openapi_path"](
            "/health",
            {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"properties": {"ok": {"type": "boolean"}}}
                                }
                            }
                        }
                    }
                }
            },
            ["src/app.py"],
        )
        item = (await tools["get_openapi_path"]("/health"))["pathItem"]
        assert item["parameters"] == parameters
        response = item["get"]["responses"]["200"]
        assert response["description"] == "OK"
        assert response["content"]["application/json"]["schema"] == {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
        }
        writes = client.write_count
        for path, patch in [
            ("/new", {"parameters": parameters}),
            ("/health", {"get": {"responses": {"200": None}}}),
            ("/health", {"post": {"summary": "missing responses"}}),
        ]:
            with pytest.raises(ValueError):
                await tools["upsert_openapi_path"](path, patch, ["src/app.py"])
            assert client.write_count == writes
            assert (await tools["get_openapi_path"]("/health"))["pathItem"] == item

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "section,initial,patch",
    [
        (
            "responses",
            {"description": "OK"},
            {"content": {"application/json": {"schema": {"type": "string"}}}},
        ),
        (
            "requestBodies",
            {"content": {"application/json": {"schema": {"type": "string"}}}},
            {"description": "Input"},
        ),
        ("securitySchemes", {"type": "http", "scheme": "bearer"}, {"description": "Access token"}),
    ],
)
def test_partial_component_updates_validate_after_merge(
    tmp_path: Path, section: str, initial: dict[str, Any], patch: dict[str, Any]
) -> None:
    async def scenario() -> None:
        source = tmp_path / "project" / "src"
        source.mkdir(parents=True)
        (source / "app.py").write_text("route = '/health'\n")
        client = MemoryArtifactClient()
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        await tools["initialize_openapi"]("Demo")
        await tools["upsert_openapi_component"](section, "Example", initial, ["src/app.py"])
        await tools["upsert_openapi_component"](section, "Example", patch, ["src/app.py"])
        component = (await tools["get_openapi_component"](section, "Example"))["component"]
        assert component == {**initial, **patch, "x-component-files": ["src/app.py"]}
        writes = client.write_count
        with pytest.raises(ValueError):
            await tools["upsert_openapi_component"](section, "Missing", patch, ["src/app.py"])
        assert client.write_count == writes

    asyncio.run(scenario())


def test_invalid_mutations_and_stale_cas_leave_document_unchanged(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "project" / "src"
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
        source = tmp_path / "project" / "src"
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
            "Replacement"
        )
        monkeypatch.setattr(
            openapi_module,
            "_run_vacuum",
            AsyncMock(
                side_effect=lambda _source: {
                    "available": False,
                    "executionError": "Vacuum executable is unavailable",
                    "issues": [],
                    "truncated": False,
                }
            ),
        )
        unavailable = await tools["validate_openapi"]()
        assert not unavailable["valid"]
        assert not unavailable["validatorAvailable"]
        assert unavailable["validatorExecutionError"]

    asyncio.run(scenario())


PATH_ITEM_REF_GUIDANCE = (
    "OpenAPI 3.0 Path Item $ref is not supported; inline the Path Item with its own "
    "operations and unique operationIds at path"
)


def test_openapi_30_component_aliases_mutate_and_path_item_refs_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        source = tmp_path / "project" / "src"
        source.mkdir(parents=True)
        (source / "app.py").write_text("route = '/health'\n")
        client = MemoryArtifactClient()
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        monkeypatch.setattr(openapi_module, "_run_vacuum", clean_vacuum)
        await tools["initialize_openapi"]("Aliases")

        components = (
            ("schemas", "User", {"type": "object"}),
            ("responses", "User", {"description": "OK"}),
            ("parameters", "User", {"name": "user", "in": "query", "schema": {"type": "string"}}),
        )
        for section, name, target in components:
            await tools["upsert_openapi_component"](section, name, target, ["src/app.py"])
            alias = {"$ref": f"#/components/{section}/{name}"}
            await tools["upsert_openapi_component"](section, "Alias", alias, ["src/app.py"])
            assert (await tools["get_openapi_component"](section, "Alias"))["component"] == alias

        await tools["upsert_openapi_path"]("/health", valid_path_item(), ["src/app.py"])
        assert (await tools["validate_openapi"]())["valid"]
        health = (await tools["get_openapi_path"]("/health"))["pathItem"]

        writes = client.write_count
        # A new alias, a merge that turns an existing Path Item into one, and
        # a path long enough that the bounded model reply truncates it.
        for path in ("/alias", "/health", "/" + "p" * 2000):
            with pytest.raises(ToolInputError) as rejected:
                await tools["upsert_openapi_path"](
                    path, {"$ref": "#/paths/~1health"}, ["src/app.py"]
                )
            assert str(rejected.value) == f"{PATH_ITEM_REF_GUIDANCE} {path}"
            reply = _safe_tool_response("upsert_openapi_path", rejected.value)["error"]
            assert reply["code"] == "tool_input_invalid" and reply["retryable"] is False
            assert reply["message"].startswith(PATH_ITEM_REF_GUIDANCE)
        assert client.write_count == writes
        assert (await tools["list_openapi_paths"]())["paths"] == ["/health"]
        assert (await tools["get_openapi_path"]("/health"))["pathItem"] == health

        invalid_path = valid_path_item("#/components/schemas/User")
        invalid_path["get"]["responses"]["200"]["content"]["application/json"]["schema"][
            "description"
        ] = "Ignored sibling"
        with pytest.raises(ValueError, match=r"\$ref objects cannot contain sibling"):
            await tools["upsert_openapi_path"]("/bad", invalid_path, ["src/app.py"])
        with pytest.raises(ValueError, match="does not exist"):
            await tools["upsert_openapi_component"](
                "schemas", "BadAlias", {"$ref": "#/components/schemas/User"}, ["src/missing.py"]
            )
        assert client.write_count == writes

    asyncio.run(scenario())


def test_seeded_openapi_30_aliases_validate_and_missing_provenance_names_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def validate_seed(document: dict[str, Any]) -> dict[str, Any]:
        client = MemoryArtifactClient()
        seed = client.seed("inputs", "seed", "application/yaml", yaml.safe_dump(document).encode())
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        await tools["load_openapi"]("inputs", "seed", seed.revision)
        return await tools["validate_openapi"]()

    async def scenario() -> None:
        source = tmp_path / "project" / "src"
        source.mkdir(parents=True)
        (source / "app.py").write_text("route = '/health'\n")
        monkeypatch.setattr(openapi_module, "_run_vacuum", clean_vacuum)
        document = minimal_document("Seeded aliases")
        document["paths"]["/health"] = valid_path_item() | {"x-path-files": ["src/app.py"]}
        document["components"] = {
            "schemas": {
                "User": {"type": "object", "x-component-files": ["src/app.py"]},
                "Alias": {"$ref": "#/components/schemas/User"},
            },
            "responses": {
                "User": {"description": "OK", "x-component-files": ["src/app.py"]},
                "Alias": {"$ref": "#/components/responses/User"},
            },
            "parameters": {
                "User": {
                    "name": "user",
                    "in": "query",
                    "schema": {"type": "string"},
                    "x-component-files": ["src/app.py"],
                },
                "Alias": {"$ref": "#/components/parameters/User"},
            },
        }
        assert (await validate_seed(document))["valid"]

        no_path_source = copy.deepcopy(document)
        del no_path_source["paths"]["/health"]["x-path-files"]
        path_error = await validate_seed(no_path_source)
        assert not path_error["valid"]
        assert "path /health" in path_error["structuralErrors"][0]

        no_component_source = copy.deepcopy(document)
        del no_component_source["components"]["responses"]["User"]["x-component-files"]
        component_error = await validate_seed(no_component_source)
        assert not component_error["valid"]
        assert "component responses.User" in component_error["structuralErrors"][0]

        for alias in (
            {"$ref": "#/paths/~1health"},
            {"$ref": "#/paths/~1health", "x-path-files": ["src/app.py"]},
        ):
            path_ref = copy.deepcopy(document)
            path_ref["paths"]["/alias"] = alias
            with pytest.raises(ToolInputError, match=re.escape(PATH_ITEM_REF_GUIDANCE)):
                await validate_seed(path_ref)

        # OpenAPI 3.1 Path Item references are unchanged.
        version_31 = minimal_document("Seeded 3.1 path alias") | {"openapi": "3.1.0"}
        version_31["paths"] = {
            "/health": valid_path_item() | {"x-path-files": ["src/app.py"]},
            "/alias": {"$ref": "#/paths/~1health", "x-path-files": ["src/app.py"]},
        }
        assert (await validate_seed(version_31))["valid"]

    asyncio.run(scenario())


def test_openapi_30_path_item_ref_guidance_holds_against_real_vacuum(tmp_path: Path) -> None:
    if openapi_module.shutil.which("vacuum") is None:
        pytest.skip("Vacuum executable is unavailable")

    def operation(operation_id: str, summary: str, schema: str) -> dict[str, Any]:
        return {
            "summary": summary,
            "description": f"{summary}.",
            "operationId": operation_id,
            "tags": ["health"],
            "responses": {
                "200": {
                    "description": "Service is healthy.",
                    "content": {
                        "application/json": {"schema": {"$ref": f"#/components/schemas/{schema}"}}
                    },
                }
            },
        }

    async def scenario() -> None:
        source = tmp_path / "project" / "src"
        source.mkdir(parents=True)
        (source / "app.py").write_text("@app.get('/health')\ndef health(): ...\n")
        tools = await make_tools(
            tmp_path, MemoryArtifactClient(), WorkerState(), namespace="openapi"
        )
        await tools["initialize_openapi"]("Health Service", description="HTTP health endpoint")
        await tools["set_openapi_servers"]([{"url": ".", "description": "Current origin"}])
        await tools["set_openapi_tags"]([{"name": "health", "description": "Health operations"}])
        health = {
            "type": "object",
            "description": "Service health.",
            "properties": {"ok": {"type": "boolean", "example": True}},
            "example": {"ok": True},
        }
        await tools["upsert_openapi_component"]("schemas", "Health", health, ["src/app.py"])
        alias = {"$ref": "#/components/schemas/Health"}
        await tools["upsert_openapi_component"]("schemas", "HealthAlias", alias, ["src/app.py"])
        await tools["upsert_openapi_path"](
            "/health", {"get": operation("getHealth", "Read health", "Health")}, ["src/app.py"]
        )
        with pytest.raises(ToolInputError, match=re.escape(PATH_ITEM_REF_GUIDANCE)):
            await tools["upsert_openapi_path"](
                "/alias", {"$ref": "#/paths/~1health"}, ["src/app.py"]
            )

        # Vacuum rejects the refused shape whether or not provenance sits beside $ref.
        document = yaml.safe_load((await tools["read_openapi_document"]())["document"])
        document["paths"]["/alias"] = {"$ref": "#/paths/~1health", "x-path-files": ["src/app.py"]}
        sourced = await _run_vacuum(openapi_module._dump_document(document).decode())
        assert {"no-$ref-siblings", "operation-operationId-unique"} <= {
            issue["code"] for issue in sourced["issues"]
        }
        del document["paths"]["/alias"]["x-path-files"]
        bare = await _run_vacuum(openapi_module._dump_document(document).decode())
        assert "operation-operationId-unique" in {issue["code"] for issue in bare["issues"]}

        # Following the guidance keeps the component alias and validates cleanly.
        inline = operation("getHealthAlias", "Read health by alias", "HealthAlias")
        await tools["upsert_openapi_path"]("/alias", {"get": inline}, ["src/app.py"])
        validation = await tools["validate_openapi"]()
        assert validation["valid"], validation
        assert validation["issues"] == []

    asyncio.run(scenario())


def test_schema_mutations_replace_refs_atomically_and_reject_vacuum_shape_errors(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        source = tmp_path / "project" / "src"
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
        with pytest.raises(ValueError, match="at least one schema"):
            await tools["upsert_openapi_component"](
                "schemas",
                "Envelope",
                {"properties": {"value": {"anyOf": []}}},
                ["src/app.py"],
            )
        assert client.write_count == writes
        assert client.bindings[("openapi", "openapi")].revision == revision

    asyncio.run(scenario())


@pytest.mark.parametrize("keyword", ["allOf", "anyOf", "oneOf"])
def test_single_schema_combinations_are_preserved_and_empty_ones_rejected(tmp_path, keyword):
    async def scenario():
        source = tmp_path / "project" / "src"
        source.mkdir(parents=True)
        (source / "app.py").write_text("def handler(): pass\n")
        client, state = MemoryArtifactClient(), WorkerState()
        tools = await make_tools(tmp_path, client, state, namespace="openapi")
        await tools["initialize_openapi"](title="Service")
        schema = {keyword: [{"type": "string"}]}
        await tools["upsert_openapi_component"]("schemas", "Value", schema, ["src/app.py"])
        component = await tools["get_openapi_component"]("schemas", "Value")
        assert component["component"][keyword] == [{"type": "string"}]
        writes = client.write_count
        with pytest.raises(ValueError, match="at least one schema"):
            await tools["upsert_openapi_component"](
                "schemas", "Value", {keyword: []}, ["src/app.py"]
            )
        assert client.write_count == writes

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
    monkeypatch.setattr("contractor_runtime.core.process.run_command", AsyncMock(side_effect=run))
    result = asyncio.run(_run_vacuum("alpha\nbeta\n"))
    assert result["executionError"] is None
    assert [item["severity"] for item in result["issues"]] == [0, 1]
    assert [item["snippet"] for item in result["issues"]] == ["alpha", "beta"]
    assert all("range" not in item for item in result["issues"])
    assert calls[0][0][:7] == [
        "/opt/bin/vacuum",
        "spectral-report",
        "-i",
        "-o",
        "--remote=false",
        "--no-update-check",
        "--config",
    ]
    assert str(calls[0][1]["cwd"]) in calls[0][0][7]
    assert calls[0][1]["input"] == b"alpha\nbeta\n"
    assert calls[0][1]["timeout"] == 30
    assert "shell" not in calls[0][1]


def test_vacuum_adapter_builds_snippets_only_for_returned_issues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = openapi_module.MAX_VALIDATION_ISSUES
    issues = [{"severity": 1, "message": f"warning {index}"} for index in range(limit + 50)]
    issues.append({"severity": 0, "message": "error"})

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 1, json.dumps(issues).encode(), b"")

    snippets: list[str] = []
    original = openapi_module._issue_with_snippet

    def counting(issue: dict[str, Any], lines: Any) -> dict[str, Any]:
        snippets.append(issue["message"])
        return original(issue, lines)

    monkeypatch.setattr(openapi_module.shutil, "which", lambda _name: "/opt/bin/vacuum")
    monkeypatch.setattr("contractor_runtime.core.process.run_command", AsyncMock(side_effect=run))
    monkeypatch.setattr(openapi_module, "_issue_with_snippet", counting)
    result = asyncio.run(_run_vacuum("alpha\n"))
    assert result["truncated"]
    assert len(result["issues"]) == len(snippets) == limit
    assert result["issues"][0]["message"] == "error"


def test_vacuum_adapter_reports_unavailable_timeout_and_bad_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openapi_module.shutil, "which", lambda _name: None)
    assert not asyncio.run(_run_vacuum("openapi: 3.0.3"))["available"]

    monkeypatch.setattr(openapi_module.shutil, "which", lambda _name: "/bin/vacuum")

    def timeout(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired("vacuum", 30)

    monkeypatch.setattr(
        "contractor_runtime.core.process.run_command", AsyncMock(side_effect=timeout)
    )
    assert "timed out" in asyncio.run(_run_vacuum("openapi: 3.0.3"))["executionError"]

    monkeypatch.setattr(
        "contractor_runtime.core.process.run_command",
        AsyncMock(
            side_effect=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                [], 0, b"not-json", b""
            )
        ),
    )
    assert "invalid JSON" in asyncio.run(_run_vacuum("openapi: 3.0.3"))["executionError"]


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
            resources=WorkerState().resources,
            project_workspace=StaticWorkspaceReader({}),
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
                resources=WorkerState().resources,
            )

    asyncio.run(scenario())
    assert OpenAPIToolsetFactory.workspace_access == "read"
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


@pytest.mark.parametrize("failure", ["lost_transport", "lost_500"])
def test_openapi_ambiguous_mutation_adopts_committed_revision(tmp_path: Path, failure: str) -> None:
    async def scenario() -> None:
        client = FaultInjectingArtifactClient(MemoryArtifactClient())
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        await tools["initialize_openapi"]("Initial")
        client.failure = failure
        before = len(client.attempts)
        changed = await tools["set_openapi_info"]("Updated")
        assert changed["artifact"]["revision"] == "revision-2"
        assert client.inner.write_count == 2
        assert client.attempts[before:] == [client.attempts[before]] * 2
        read = await tools["read_openapi_document"]()
        assert yaml.safe_load(read["document"])["info"]["title"] == "Updated"
        assert read["artifact"]["revision"] == "revision-2"
        later = await tools["set_openapi_servers"]([{"url": "https://api.example.test"}])
        assert later["artifact"]["revision"] == "revision-3"
        assert client.inner.write_count == 3

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["conflict", "diverged"])
def test_openapi_concurrent_change_requires_reload(tmp_path: Path, failure: str) -> None:
    async def scenario() -> None:
        client = FaultInjectingArtifactClient(MemoryArtifactClient())
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        await tools["initialize_openapi"]("Initial")
        client.concurrent_data = yaml.safe_dump(minimal_document("Concurrent")).encode()
        client.failure = failure
        before = len(client.attempts)
        with pytest.raises(ToolInputError) as rejected:
            await tools["set_openapi_info"]("Mine")
        assert rejected.value.code == "document_changed"
        assert rejected.value.retryable is False
        assert "load_openapi" in str(rejected.value)
        assert "read_openapi_document" in str(rejected.value)
        visible = _safe_tool_response("set_openapi_info", rejected.value)["error"]
        assert visible["code"] == "document_changed" and visible["retryable"] is False
        assert "load_openapi" in visible["message"]
        assert len(client.attempts) - before == (1 if failure == "conflict" else 2)
        loaded = await tools["load_openapi"]("openapi", "openapi")
        assert (
            loaded["artifact"]["revision"] == client.inner.bindings[("openapi", "openapi")].revision
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        ("fenced", "allocation_write_fenced"),
        ("forbidden", "artifact_access_denied"),
        ("invalid", "artifact_invalid"),
    ],
)
def test_openapi_definite_rejection_is_not_replayed(
    tmp_path: Path, failure: str, code: str
) -> None:
    async def scenario() -> None:
        client = FaultInjectingArtifactClient(MemoryArtifactClient())
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        await tools["initialize_openapi"]("Initial")
        client.failure = failure
        before = len(client.attempts)
        with pytest.raises(ArtifactAPIError) as rejected:
            await tools["set_openapi_info"]("Denied")
        assert rejected.value.code == code
        assert len(client.attempts) == before + 1
        assert client.inner.write_count == 1

    asyncio.run(scenario())


def test_openapi_document_work_runs_on_worker_threads_while_the_loop_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        order: list[str] = []
        gate = PreparationGate(openapi_module._parse_document, "decoded", order)
        threads = observe_threads(
            monkeypatch,
            openapi_module,
            {
                "_parse_document": ("decode", gate),
                "_prepare_document": ("initialize", None),
                "_prepare_mutation": ("mutation", None),
                "_dump_document": ("serialize", None),
                "_targeted_copy": ("targeted read", None),
                "_bounded_name_listing": ("listing", None),
                "_validate_structure": ("validate", None),
                "_vacuum_report": ("validator output", None),
            },
        )
        monkeypatch.setattr(openapi_module.shutil, "which", lambda _name: "/opt/bin/vacuum")
        monkeypatch.setattr(
            "contractor_runtime.core.process.run_command",
            AsyncMock(return_value=subprocess.CompletedProcess(["vacuum"], 0, b"[]", b"")),
        )
        source = tmp_path / "project" / "src"
        source.mkdir(parents=True)
        (source / "app.py").write_text("route = '/health'\n")
        client = MemoryArtifactClient()
        seed = client.seed(
            "inputs", "seed", "application/json", json.dumps(minimal_document("Seed")).encode()
        )
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")

        load = asyncio.create_task(tools["load_openapi"]("inputs", "seed", seed.revision))
        try:
            await gate.wait_entered(load)
            await loop_callback(order)
            assert not load.done()
        finally:
            gate.release.set()
        assert (await load)["copied"] is True
        assert order == ["loop callback", "decoded"]

        await tools["upsert_openapi_path"]("/health", valid_path_item(), ["src/app.py"])
        await tools["get_openapi_path"]("/health")
        await tools["list_openapi_paths"]()
        await tools["read_openapi_document"]()
        assert (await tools["validate_openapi"]())["valid"] is True
        await tools["initialize_openapi"]("Next", target_name="next")
        assert set(threads) == {
            "decode",
            "initialize",
            "mutation",
            "serialize",
            "targeted read",
            "listing",
            "validate",
            "validator output",
        }
        assert all(threading.get_ident() not in seen for seen in threads.values()), threads

    asyncio.run(scenario())


@pytest.mark.parametrize("media_type", ["application/yaml", "application/json"])
def test_openapi_seed_is_parsed_and_validated_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, media_type: str
) -> None:
    async def scenario() -> None:
        document = minimal_document("Seed")
        document["paths"]["/health"] = valid_path_item()
        payload = (
            yaml.safe_dump(document) if media_type == "application/yaml" else json.dumps(document)
        )
        client = MemoryArtifactClient()
        seed = client.seed("inputs", "seed", media_type, payload.encode())
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        session = tools["load_openapi"]._session
        parses = 0
        start_reader = yaml.reader.Reader.__init__
        json_loads = json.loads

        def counted_reader(self: Any, stream: Any) -> None:
            # Every PyYAML pass over the text, a parse or a load, starts a Reader.
            nonlocal parses
            parses += 1
            start_reader(self, stream)

        def counted_json(*args: Any, **kwargs: Any) -> Any:
            nonlocal parses
            parses += 1
            return json_loads(*args, **kwargs)

        monkeypatch.setattr(yaml.reader.Reader, "__init__", counted_reader)
        monkeypatch.setattr(json, "loads", counted_json)
        calls = record_calls(
            monkeypatch,
            {
                "_validate_json_tree": 0,
                "_validate_structure": 0,
                "_validate_path_item": 0,
                "_dump_document": 0,
            },
        )

        assert (await tools["load_openapi"]("inputs", "seed", seed.revision))["copied"] is True
        loaded = session._document
        assert parses == 1
        assert [value is loaded for value in calls["_validate_json_tree"]] == [True]
        assert [value is loaded for value in calls["_validate_structure"]] == [True]
        assert len(calls["_validate_path_item"]) == 1
        assert [value is loaded for value in calls["_dump_document"]] == [True]

        # Resuming the copied binding parses its YAML once and renders nothing.
        writes = client.write_count
        assert (await tools["load_openapi"]("openapi", "openapi"))["copied"] is False
        assert parses == 2
        assert len(calls["_validate_structure"]) == 2
        assert len(calls["_dump_document"]) == 1
        assert client.write_count == writes

    asyncio.run(scenario())


@pytest.mark.parametrize("target", ["path", "component"])
def test_accepted_upsert_validates_its_merged_document_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    async def scenario() -> None:
        source = tmp_path / "project" / "src"
        source.mkdir(parents=True)
        (source / "app.py").write_text("route = '/health'\n")
        client = MemoryArtifactClient()
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        session = tools["upsert_openapi_path"]._session
        await tools["initialize_openapi"]("Counted")
        await tools["upsert_openapi_path"]("/other", valid_path_item(), ["src/app.py"])
        await tools["upsert_openapi_path"]("/health", valid_path_item(), ["src/app.py"])
        await tools["upsert_openapi_component"](
            "responses", "Ok", {"description": "OK"}, ["src/app.py"]
        )
        await tools["upsert_openapi_component"](
            "requestBodies",
            "Input",
            {"content": {"application/json": {"schema": {"type": "object"}}}},
            ["src/app.py"],
        )
        calls = record_calls(
            monkeypatch,
            {
                "_validate_json_tree": 0,
                "_validate_structure": 0,
                "_validate_path_item": 0,
                "_validate_component": 1,
                "_dump_document": 0,
            },
        )
        writes = client.write_count
        if target == "path":
            parameters = [{"name": "verbose", "in": "query", "schema": {"type": "boolean"}}]
            await tools["upsert_openapi_path"](
                "/health", {"parameters": parameters}, ["src/app.py"]
            )
        else:
            await tools["upsert_openapi_component"](
                "responses",
                "Ok",
                {"content": {"application/json": {"schema": {"type": "string"}}}},
                ["src/app.py"],
            )
        assert client.write_count == writes + 1
        document = session._document
        if target == "path":
            changed, entries = document["paths"]["/health"], "_validate_path_item"
            assert "parameters" in changed
        else:
            changed, entries = document["components"]["responses"]["Ok"], "_validate_component"
            assert "content" in changed
        # One pass over the merged document: one tree walk, one structure check
        # and one rendering. The patch has its own walk before the merge.
        assert [value is document for value in calls["_validate_json_tree"]].count(True) == 1
        assert [value is document for value in calls["_validate_structure"]] == [True]
        assert [value is document for value in calls["_dump_document"]] == [True]
        # The structure check covers each entry once, the changed one included.
        assert [value is changed for value in calls[entries]].count(True) == 1
        assert len(calls["_validate_path_item"]) == len(document["paths"]) == 2
        assert len(calls["_validate_component"]) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_load,close_session", [(True, False), (True, True), (False, True)])
def test_json_load_cancellation_and_close_join_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_load: bool, close_session: bool
) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        initialized = await tools["initialize_openapi"]("Before")
        session = tools["load_openapi"]._session
        before = session._document
        seed = client.seed(
            "inputs", "seed", "application/json", json.dumps(minimal_document("After")).encode()
        )
        writes = client.write_count
        order: list[str] = []
        gate = PreparationGate(openapi_module._parse_document, "decoded", order)
        monkeypatch.setattr(openapi_module, "_parse_document", gate)

        async def close() -> None:
            await tools["load_openapi"].close()
            order.append("closed")

        loading = asyncio.create_task(
            tools["load_openapi"]("inputs", "seed", seed.revision, target_name="copy")
        )
        closing = None
        try:
            await gate.wait_entered(loading)
            if cancel_load:
                loading.cancel()
            if close_session:
                closing = asyncio.create_task(close())
            await yield_to_loop()
            assert not loading.done()
            assert closing is None or not closing.done()
            assert session._document is before
            assert session._revision == initialized["artifact"]["revision"]
        finally:
            gate.release.set()
        if cancel_load:
            with pytest.raises(asyncio.CancelledError):
                await loading
        else:
            await loading
        if closing is not None:
            await closing
        assert order == (["decoded", "closed"] if close_session else ["decoded"])
        assert client.write_count == writes + (not cancel_load)
        assert session._document is (None if close_session else before)

    asyncio.run(scenario())


def test_close_joins_a_cancelled_mutation_preparation_before_clearing_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        session = tools["set_openapi_info"]._session
        initialized = await tools["initialize_openapi"]("Before")
        with pytest.raises(RuntimeError, match="session lock"):
            await session._prepare(len, "unlocked")
        writes = client.write_count
        order: list[str] = []
        gate = PreparationGate(openapi_module._prepare_mutation, "prepared", order)
        monkeypatch.setattr(openapi_module, "_prepare_mutation", gate)

        async def close() -> None:
            await tools["set_openapi_info"].close()
            order.append("closed")

        mutation = asyncio.create_task(tools["set_openapi_info"]("After"))
        try:
            await gate.wait_entered(mutation)
            mutation.cancel()
            closing = asyncio.create_task(close())
            await yield_to_loop()
            # The cancelled call keeps the lock, so close() waits for the thread.
            assert not mutation.done()
            assert not closing.done()
            assert session._revision == initialized["artifact"]["revision"]
        finally:
            gate.release.set()
        with pytest.raises(asyncio.CancelledError):
            await mutation
        await closing
        assert order == ["prepared", "closed"]
        assert client.write_count == writes
        assert session._document is None and session._revision is None

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["preparation", "vacuum"])
def test_close_cancels_validation_after_joining_its_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    async def scenario() -> None:
        client = MemoryArtifactClient()
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        session = tools["validate_openapi"]._session
        await tools["initialize_openapi"]("Validated")
        order: list[str] = []
        vacuum_started = asyncio.Event()
        finish_vacuum = asyncio.Event()

        async def pending_vacuum(source: str) -> dict[str, Any]:
            vacuum_started.set()
            try:
                await finish_vacuum.wait()
            except asyncio.CancelledError:
                order.append("vacuum cancelled")
                raise
            return await clean_vacuum(source)

        vacuum = AsyncMock(side_effect=pending_vacuum)
        monkeypatch.setattr(openapi_module, "_run_vacuum", vacuum)
        gate = PreparationGate(openapi_module._validate_structure, "prepared", order)
        if phase == "preparation":
            monkeypatch.setattr(openapi_module, "_validate_structure", gate)

        async def close() -> None:
            await tools["validate_openapi"].close()
            order.append("closed")

        validation = asyncio.create_task(tools["validate_openapi"]())
        try:
            if phase == "preparation":
                await gate.wait_entered(validation)
            else:
                await vacuum_started.wait()
            closing = asyncio.create_task(close())
            await yield_to_loop()
            if phase == "preparation":
                # close() cancelled the validation, which still waits for its thread.
                assert not validation.done()
                assert not closing.done()
            else:
                assert closing.done()
        finally:
            # Released without a cancellation, the validation would complete.
            gate.release.set()
            finish_vacuum.set()
        with pytest.raises(asyncio.CancelledError):
            await validation
        await closing
        if phase == "preparation":
            # Cancellation joined the thread and then stopped before Vacuum.
            assert order == ["prepared", "closed"]
            vacuum.assert_not_awaited()
        else:
            assert order == ["vacuum cancelled", "closed"]
        assert session._validation_task is None
        assert session._document is None

    asyncio.run(scenario())


def test_openapi_written_bytes_and_noop_results_are_unchanged(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "project" / "src"
        source.mkdir(parents=True)
        (source / "app.py").write_text("route = '/health'\n")
        client = MemoryArtifactClient()
        seed = client.seed(
            "inputs",
            "seed",
            "application/json",
            json.dumps(
                {
                    "openapi": "3.0.3",
                    "info": {"title": "Seed", "version": "1"},
                    "paths": {},
                    "components": {
                        "schemas": {
                            "Thing": {"type": "object", "x-component-files": ["src/app.py"]}
                        }
                    },
                    "x-extra": [1, 2.5, True, None, "2023-01-01"],
                }
            ).encode(),
        )
        tools = await make_tools(tmp_path, client, WorkerState(), namespace="openapi")
        await tools["load_openapi"]("inputs", "seed", seed.revision)
        await tools["set_openapi_info"]("Service", description="desc", framework="fastapi")
        await tools["set_openapi_servers"]([{"url": "https://a.example", "description": "A"}])
        await tools["set_openapi_tags"]([{"name": "health", "x-order": 1}])
        await tools["upsert_openapi_path"](
            "/health",
            {"x-path-files": ["stale.py"], **valid_path_item("#/components/schemas/Thing")},
            ["src/app.py"],
        )
        await tools["upsert_openapi_path"](
            "/health",
            {"parameters": [{"name": "v", "in": "query", "schema": {"type": "boolean"}}]},
            ["src/app.py"],
        )
        await tools["upsert_openapi_component"](
            "responses", "Ok", {"description": "OK"}, ["src/app.py"]
        )
        await tools["upsert_openapi_component"](
            "schemas", "Alias", {"$ref": "#/components/schemas/Thing"}, ["src/app.py"]
        )
        writes = client.write_count
        unchanged = await tools["set_openapi_tags"]([{"name": "health", "x-order": 1}])
        assert unchanged["changed"] is False and client.write_count == writes
        await tools["remove_openapi_component"]("responses", "Ok")
        stored = client.bindings[("openapi", "openapi")].data
        # The bytes of the implementation before off-loop preparation (V407-031).
        assert stored == WRITTEN_DOCUMENT
        assert (await tools["read_openapi_document"]())["document"] == stored.decode()

    asyncio.run(scenario())


WRITTEN_DOCUMENT = b"""\
openapi: 3.0.3
info:
  title: Service
  version: '1'
  description: desc
  x-framework: fastapi
paths:
  /health:
    x-path-files:
    - src/app.py
    get:
      responses:
        '200':
          description: OK
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Thing'
    parameters:
    - name: v
      in: query
      schema:
        type: boolean
components:
  schemas:
    Thing:
      type: object
      x-component-files:
      - src/app.py
    Alias:
      $ref: '#/components/schemas/Thing'
  responses: {}
x-extra:
- 1
- 2.5
- true
- null
- '2023-01-01'
servers:
- url: https://a.example
  description: A
tags:
- name: health
  x-order: 1
"""


def record_calls(
    monkeypatch: pytest.MonkeyPatch, functions: dict[str, int]
) -> dict[str, list[Any]]:
    """Record the positional argument at the given index of each module-function call."""

    calls: dict[str, list[Any]] = {name: [] for name in functions}
    for name, position in functions.items():
        function = getattr(openapi_module, name)

        def run(
            *args: Any,
            _name: str = name,
            _position: int = position,
            _function: Callable[..., Any] = function,
            **kwargs: Any,
        ) -> Any:
            calls[_name].append(args[_position])
            return _function(*args, **kwargs)

        monkeypatch.setattr(openapi_module, name, run)
    return calls


async def make_tools(
    tmp_path: Path,
    client: MemoryArtifactClient,
    state: WorkerState,
    *,
    namespace: str,
    project_workspace: Any = None,
) -> dict[str, Any]:
    factory = OpenAPIToolsetFactory(lambda _allocation, _settings: client)
    if project_workspace is None:
        project = tmp_path / "project"
        project_workspace = StaticWorkspaceReader(
            {
                path.relative_to(project).as_posix(): path.read_text()
                for path in project.rglob("*")
                if path.is_file()
            }
        )
    selected = await factory.create_selected(
        selected=sorted(factory.exported_tools),
        allocation_id="allocation-1",
        run_id="run-1",
        namespace=namespace,
        runtime_settings=runtime_settings(),
        workspace=AllocationWorkspace(root=tmp_path.parent, path=tmp_path),
        resources=state.resources,
        project_workspace=project_workspace,
    )
    return dict(selected)


def runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        llmGatewayUrl="https://gateway.example/v1",
        llmGatewayToken=SECRET,
        artifactApiUrl="https://control.example/private/v1",
        requestTimeoutSeconds=30,
    )


BOUNDS_YAML_PREFIX = "openapi: 3.0.3\ninfo:\n  title: Bounds\n  version: 1.0.0\npaths: {}\n"


def bounds_document(prefix: str, case: dict[str, Any], *, compact_json: bool = False) -> bytes:
    """Render one case of api/testdata/openapi-document-bounds.json after prefix.

    The fixture's description states the rule; Go's TestSharedOpenAPIDocumentBounds*
    renders the same bytes, which the fixture's sha256 values pin.
    """

    if "string" in case:
        text = "a" * case["string"]["bytes"]
        path, extra_json, extra_yaml = "/bounds", "", ""
        match case["string"]["kind"]:
            case "value":
                extra_json = ',"description":"' + text + '"'
                extra_yaml = '      description: "' + text + '"\n'
            case "name":
                name = "x-" + text[2:]
                extra_json = ',"' + name + '":0'
                extra_yaml = '      ? "' + name + '"\n      : 0\n'
            case "path":
                path = "/" + text[1:]
            case "operationId":
                extra_json = ',"operationId":"' + text + '"'
                extra_yaml = '      operationId: "' + text + '"\n'
            case _:
                raise ValueError("unknown string kind")
        if compact_json:
            return (
                '{"openapi":"3.0.3","info":{"title":"Bounds","version":"1.0.0"},"paths":{"'
                + path
                + '":{"get":{"responses":{"200":{"description":"ok"}}'
                + extra_json
                + "}}}}"
            ).encode()
        return (
            'openapi: 3.0.3\ninfo:\n  title: Bounds\n  version: 1.0.0\npaths:\n  ? "'
            + path
            + "\"\n  :\n    get:\n      responses:\n        '200':\n          description: ok\n"
            + extra_yaml
        ).encode()
    text = prefix
    deep = case.get("deep")
    if deep is not None:
        levels = deep["depth"] - 2
        if deep["kind"] == "mapping":
            if compact_json:
                text += ',"x-deep":' + '{"a":' * levels + "0" + "}" * levels
            else:
                text += "x-deep:\n" + "".join("  " * level + "a:\n" for level in range(1, levels))
                text += "  " * levels + "a: 0\n"
        elif deep["kind"] == "sequence":
            nested = "[" * levels + "0" + "]" * levels
            text += (',"x-deep":' + nested) if compact_json else ("x-deep: " + nested + "\n")
        else:
            raise ValueError(f"unknown deep kind {deep['kind']!r}")
    elements = case.get("elements", 0)
    if elements:
        if compact_json:
            text += ',"x-items":[' + ",".join(["0"] * elements) + "]"
        else:
            text += "x-items:\n" + "- 0\n" * elements
    members = case.get("members", 0)
    if members:
        if compact_json:
            text += ',"x-keys":{' + ",".join(f'"k{index:06d}":0' for index in range(members)) + "}"
        else:
            text += "x-keys:\n" + "".join(f"  k{index:06d}: 0\n" for index in range(members))
    pad = case.get("pad")
    if pad is None:
        return (text + "}" if compact_json else text).encode()
    head, tail = (text + ',"x-pad":"', '"}') if compact_json else (text + "x-pad: ", "\n")
    room = pad["bytes"] - len(head.encode()) - len(tail.encode())
    fill = pad["fill"].encode()
    if room < 0 or not fill:
        raise ValueError(f"a pad of {pad['bytes']} bytes cannot hold the document")
    return head.encode() + fill * (room // len(fill)) + b"a" * (room % len(fill)) + tail.encode()


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


async def clean_vacuum(_source: str) -> dict[str, Any]:
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


class StaticWorkspaceReader:
    def __init__(self, files: dict[str, str]) -> None:
        self._files = dict(files)
        self.snapshot_calls = 0
        self.file_paths_calls = 0

    async def file_paths(self) -> frozenset[str]:
        self.file_paths_calls += 1
        return frozenset(self._files)

    async def snapshot(self) -> WorkspaceSnapshot:
        self.snapshot_calls += 1
        return WorkspaceSnapshot(
            directories=("src",),
            files=tuple(
                WorkspaceTextFile(
                    path=path,
                    text=text,
                    size=len(text.encode("utf-8")),
                )
                for path, text in sorted(self._files.items())
            ),
            binary_paths=(),
            digest="sha256:" + "0" * 64,
        )


class MemoryArtifactClient:
    def __init__(self) -> None:
        self.bindings: dict[tuple[str, str], StoredArtifact] = {}
        self.history: dict[tuple[str, str, str], StoredArtifact] = {}
        self._known: dict[tuple[str, str, str], ArtifactRef] = {}
        self._next_revision = 1
        self.write_count = 0
        self.read_count = 0
        self.read_limits: list[int] = []

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return tuple(self._known.values())

    def seed(self, namespace: str, name: str, media_type: str, data: bytes) -> ArtifactRef:
        return self._store(namespace, name, media_type, data)

    async def read_artifact(
        self, ref: ArtifactRef, *, max_bytes: int = MAX_ARTIFACT_BYTES
    ) -> ArtifactValue:
        self.read_count += 1
        self.read_limits.append(max_bytes)
        if ref.revision is None:
            stored = self.bindings[(ref.namespace, ref.name)]
        else:
            stored = self.history[(ref.namespace, ref.name, ref.revision)]
        if len(stored.data) > max_bytes:
            raise ArtifactResponseLimitError("Artifact API response exceeds the read byte limit")
        exact = ArtifactRef(
            namespace=ref.namespace,
            name=ref.name,
            revision=stored.revision,
        )
        self._remember(exact)
        return ArtifactValue(
            artifact=exact,
            media_type=stored.media_type,
            data=stored.data,
            binding_created_at=datetime.now(UTC),
            revision_created_at=datetime.now(UTC),
        )

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


@pytest.mark.parametrize(
    "reference", ["https://example.test/schema", "file:///etc/passwd", "../schema.yaml"]
)
def test_local_validation_rejects_external_references(reference: str) -> None:
    with pytest.raises(ToolInputError, match="only internal references"):
        openapi_module._require_internal_references(
            {"components": {"nested": [{"$ref": reference}]}}
        )
    openapi_module._require_internal_references({"$ref": "#/components/schemas/Item"})
