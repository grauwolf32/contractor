from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import pytest
import yaml

from contractor_runtime.capabilities import (
    CapabilityDiscoveryError,
    CapabilitySnapshot,
    discover_capabilities,
)
from contractor_runtime.factories import (
    FactoryRegistry,
    StubADKWorkerRuntimeFactory,
    built_in_factories,
)
from contractor_runtime.settings import Settings, WorkspaceLimits, WorkspaceSettings
from contractor_runtime.state import RuntimeState
from contractor_runtime.toolsets import code_analysis_languages, likec4, openapi
from contractor_runtime.toolsets.code_analysis import (
    CODE_ANALYSIS_REF,
    CORE_GRAPH_TOOLS,
    EXPORTED_TOOLS,
    GRAPH_TOOLS,
    SHALLOW_TOOLS,
    dependency_versions_match,
)
from contractor_runtime.workspace import LocalWorkdirFactory


def test_builtin_toolset_infrastructure_channels_match_parity_fixture(
    tmp_path: Path,
) -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "api"
        / "descriptor-parity"
        / "toolset-infrastructure-channels.yaml"
    )
    fixture = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert fixture["schemaVersion"] == "1.0"
    factories = built_in_factories(tmp_path).toolsets
    actual = {
        ref: {name: sorted(channels) for name, channels in factory.infrastructure_channels.items()}
        for ref, factory in factories.items()
    }
    assert actual == fixture["toolsets"]


def test_code_analysis_static_contract_and_shallow_factory_are_exact(
    tmp_path: Path,
) -> None:
    assert {"list_symbols", "search_def"} == SHALLOW_TOOLS
    assert {
        "attack_surface",
        "complexity_hotspots",
        "entrypoint_paths_to",
        "find_callees",
        "find_callers",
        "find_symbol",
        "functions_that_raise",
        "graph_summary",
        "paths_between",
    } == GRAPH_TOOLS
    assert {
        "find_callees",
        "find_callers",
        "find_symbol",
        "graph_summary",
    } == CORE_GRAPH_TOOLS
    assert EXPORTED_TOOLS == SHALLOW_TOOLS | GRAPH_TOOLS
    assert dependency_versions_match()
    factory = built_in_factories(tmp_path).toolsets[CODE_ANALYSIS_REF]
    assert factory.exported_tools == EXPORTED_TOOLS
    assert factory.infrastructure_channels == {}
    assert factory.requires_workspace is True  # type: ignore[attr-defined]
    assert factory.workspace_access == "read"  # type: ignore[attr-defined]


def test_builtin_discovery_keeps_editing_tools_without_optional_validators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes: list[tuple[str, tuple[str, ...]]] = []

    async def unavailable(name: str, arguments: tuple[str, ...]) -> bool:
        probes.append((name, arguments))
        return False

    monkeypatch.setattr(likec4, "executable_responds", unavailable)
    monkeypatch.setattr(openapi, "executable_responds", unavailable)

    async def scenario() -> None:
        work_root = tmp_path / "work"
        snapshot = await discover_capabilities(built_in_factories(work_root))

        assert snapshot.runtimes == ("adk@1",)
        assert snapshot.sandbox_profiles == ("local-workdir@1",)
        assert snapshot.runtime_adapters == (
            "caido-graphql@1",
            "http-proxy@1",
            "otlp-http@1",
        )
        assert snapshot.workspace is None
        toolsets = {item.ref: item.tools for item in snapshot.toolsets}
        assert CODE_ANALYSIS_REF not in toolsets
        assert "write_likec4" in toolsets["likec4@1"]
        assert "validate_likec4" not in toolsets["likec4@1"]
        assert "upsert_openapi_path" in toolsets["openapi@1"]
        assert "validate_openapi" not in toolsets["openapi@1"]
        assert probes == [("likec4", ("version",)), ("vacuum", ("version",))]
        assert list(work_root.iterdir()) == []

        state = RuntimeState(instance_id="stable-runtime", capabilities=snapshot)
        first = await state.registration(make_settings(tmp_path))
        second = await state.registration(make_settings(tmp_path))
        assert first.model_dump_json(by_alias=True) == second.model_dump_json(by_alias=True)

        await state.mark_registered()
        with pytest.raises(RuntimeError, match="only while starting"):
            await state.install_capabilities(snapshot)

    asyncio.run(scenario())


@pytest.mark.parametrize("storage", ["local", "memory"])
def test_workspace_capability_is_probed_frozen_and_registered(
    tmp_path: Path, storage: Literal["local", "memory"]
) -> None:
    async def scenario() -> None:
        workspace_settings = WorkspaceSettings(
            storage=storage,
            work_root=tmp_path / "project-workspaces" if storage == "local" else None,
            limits=WorkspaceLimits(
                max_files=123,
                max_expanded_bytes=4096,
                max_managed_text_bytes=2048,
                max_file_bytes=1024,
            ),
        )
        factories = built_in_factories(tmp_path / "sandbox", workspace_settings=workspace_settings)
        snapshot = await discover_capabilities(factories)
        assert snapshot.workspace is not None
        assert snapshot.workspace.storage == storage
        assert snapshot.workspace.modes == ("direct", "overlay")
        assert snapshot.workspace.limits.max_files == 123
        toolsets = {item.ref: item.tools for item in snapshot.toolsets}
        expected_code_tools = SHALLOW_TOOLS | (CORE_GRAPH_TOOLS if storage == "local" else set())
        assert toolsets[CODE_ANALYSIS_REF] == tuple(sorted(expected_code_tools))

        registration = await RuntimeState(
            instance_id=f"workspace-{storage}", capabilities=snapshot
        ).registration(make_settings(tmp_path))
        assert registration.workspace_capabilities is not None
        assert registration.workspace_capabilities.storage == storage
        assert registration.workspace_capabilities.modes == ["direct", "overlay"]
        assert registration.workspace_capabilities.limits.max_files == 123
        assert str(tmp_path / "project-workspaces") not in registration.model_dump_json(
            by_alias=True
        )

    asyncio.run(scenario())


def test_broken_code_analysis_parser_probe_omits_only_that_toolset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(code_analysis_languages, "probe_all_parsers", lambda: False)

    async def scenario() -> None:
        factories = built_in_factories(
            tmp_path / "sandbox",
            workspace_settings=WorkspaceSettings(
                storage="memory",
                limits=WorkspaceLimits(
                    max_files=100,
                    max_expanded_bytes=4096,
                    max_managed_text_bytes=2048,
                    max_file_bytes=1024,
                ),
            ),
        )
        snapshot = await discover_capabilities(factories)
        toolsets = {item.ref: item.tools for item in snapshot.toolsets}
        assert CODE_ANALYSIS_REF not in toolsets
        assert "filesystem@1" in toolsets

    asyncio.run(scenario())


def test_workspace_required_toolset_probe_is_gated_by_provider() -> None:
    toolset = WorkspaceRequiredToolset()
    factories = FactoryRegistry(
        worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
        sandbox_profiles={"local-workdir@1": PassingSandbox()},
        toolsets={toolset.ref: toolset},
    )

    async def scenario() -> None:
        snapshot = await discover_capabilities(factories)
        assert snapshot.workspace is None
        assert snapshot.toolsets == ()
        assert toolset.probes == 0

    asyncio.run(scenario())


def test_builtin_runtime_adapter_subset_is_the_only_probed_capability(tmp_path: Path) -> None:
    async def scenario() -> None:
        factories = built_in_factories(
            tmp_path / "work", enabled_runtime_adapters=("http-proxy@1",)
        )
        snapshot = await discover_capabilities(factories)
        assert tuple(factories.runtime_adapters) == ("http-proxy@1",)
        assert snapshot.runtime_adapters == ("http-proxy@1",)

    asyncio.run(scenario())


def test_builtin_runtime_adapter_subset_rejects_unknown_ref(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not built in"):
        built_in_factories(tmp_path, enabled_runtime_adapters=("unknown@1",))


def test_optional_probe_timeout_and_failure_are_omitted_without_leaking_details(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "probe-secret-canary"
    local_path = "/recognizable/private/probe/path"
    factories = FactoryRegistry(
        worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
        sandbox_profiles={"local-workdir@1": LocalWorkdirFactory(tmp_path / "work")},
        toolsets={
            "failed@1": FailedToolset(secret, local_path),
            "hanging@1": HangingToolset(),
        },
        runtime_adapters={
            "http-proxy@1": PassingAdapter("http-proxy@1"),
            "otlp-http@1": FailedAdapter("otlp-http@1", secret),
        },
    )

    async def scenario() -> None:
        with caplog.at_level(logging.INFO, logger="contractor_runtime.capabilities"):
            snapshot = await discover_capabilities(
                factories,
                per_factory_timeout_seconds=0.01,
                total_timeout_seconds=1,
            )
        assert snapshot.runtimes == ("adk@1",)
        assert snapshot.sandbox_profiles == ("local-workdir@1",)
        assert snapshot.toolsets == ()
        assert snapshot.runtime_adapters == ("http-proxy@1",)

    asyncio.run(scenario())
    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert secret not in rendered
    assert local_path not in rendered
    assert "synthetic command output" not in rendered


def test_missing_core_capability_stops_registration_with_safe_error(tmp_path: Path) -> None:
    secret = "sandbox-probe-secret"
    factories = FactoryRegistry(
        worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
        sandbox_profiles={"broken@1": FailedSandbox(secret)},
        toolsets={"empty@1": EmptyToolset()},
    )

    async def scenario() -> None:
        with pytest.raises(CapabilityDiscoveryError) as failure:
            await discover_capabilities(factories)
        assert str(failure.value) == "no usable SandboxProfile capability"
        assert secret not in str(failure.value)

        state = RuntimeState(instance_id="unprobed-runtime")
        with pytest.raises(RuntimeError, match="have not been discovered"):
            await state.registration(make_settings(tmp_path))

    asyncio.run(scenario())


def test_runtime_adapter_probe_timeout_is_omitted(tmp_path: Path) -> None:
    factories = FactoryRegistry(
        worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
        sandbox_profiles={"local-workdir@1": LocalWorkdirFactory(tmp_path / "work")},
        toolsets={"empty@1": EmptyToolset()},
        runtime_adapters={"http-proxy@1": HangingAdapter("http-proxy@1")},
    )

    async def scenario() -> None:
        snapshot = await discover_capabilities(
            factories,
            per_factory_timeout_seconds=0.01,
            total_timeout_seconds=1,
        )
        assert snapshot.runtime_adapters == ()

    asyncio.run(scenario())


def test_snapshot_normalizes_and_rejects_replacement_for_one_instance() -> None:
    async def scenario() -> None:
        original = CapabilitySnapshot.create(
            runtimes=["adk@1", "adk@1"],
            toolsets={"tools@1": ["write", "read", "write"], "empty@1": []},
            sandbox_profiles=["local@1", "local@1"],
            runtime_adapters=["otlp-http@1", "http-proxy@1", "otlp-http@1"],
        )
        replacement = CapabilitySnapshot.create(
            runtimes=["other@1"],
            toolsets={},
            sandbox_profiles=["local@1"],
            runtime_adapters=[],
        )
        assert original.runtimes == ("adk@1",)
        assert original.toolsets[0].tools == ("read", "write")
        assert original.runtime_adapters == ("http-proxy@1", "otlp-http@1")

        state = RuntimeState(instance_id="immutable-runtime")
        await state.install_capabilities(original)
        await state.install_capabilities(original)
        with pytest.raises(RuntimeError, match="immutable"):
            await state.install_capabilities(replacement)

    asyncio.run(scenario())


class HangingToolset:
    ref = "hanging@1"
    exported_tools = frozenset({"hang"})
    infrastructure_channels = MappingProxyType({})

    async def probe(self) -> frozenset[str]:
        await asyncio.Event().wait()
        return self.exported_tools

    async def create_selected(self, **_: object) -> dict[str, object]:
        raise AssertionError("probe must not construct tools")


class FailedToolset:
    ref = "failed@1"
    exported_tools = frozenset({"fail"})
    infrastructure_channels = MappingProxyType({})

    def __init__(self, secret: str, local_path: str) -> None:
        self._secret = secret
        self._local_path = local_path

    async def probe(self) -> frozenset[str]:
        raise RuntimeError(f"{self._secret} {self._local_path} synthetic command output")

    async def create_selected(self, **_: object) -> dict[str, object]:
        raise AssertionError("probe must not construct tools")


class EmptyToolset:
    ref = "empty@1"
    exported_tools = frozenset({"optional"})
    infrastructure_channels = MappingProxyType({})

    async def probe(self) -> frozenset[str]:
        return frozenset()

    async def create_selected(self, **_: object) -> dict[str, object]:
        raise AssertionError("probe must not construct tools")


class WorkspaceRequiredToolset:
    ref = "filesystem@1"
    exported_tools = frozenset({"read_file"})
    infrastructure_channels = MappingProxyType({})
    requires_workspace = True

    def __init__(self) -> None:
        self.probes = 0

    async def probe(self) -> frozenset[str]:
        self.probes += 1
        return self.exported_tools

    async def create_selected(self, **_: object) -> dict[str, object]:
        raise AssertionError("probe must not construct tools")


class PassingSandbox:
    ref = "local-workdir@1"

    async def probe(self) -> bool:
        return True

    async def prepare(self) -> object:
        raise AssertionError("probe must not prepare an allocation")

    async def cleanup(self, workspace: object) -> None:
        del workspace


class FailedSandbox:
    ref = "broken@1"

    def __init__(self, secret: str) -> None:
        self._secret = secret

    async def probe(self) -> bool:
        raise RuntimeError(self._secret)

    async def prepare(self) -> object:
        raise AssertionError("probe must not prepare an allocation")

    async def cleanup(self, workspace: object) -> None:
        del workspace


class PassingAdapter:
    def __init__(self, ref: str) -> None:
        self.ref = ref

    async def probe(self) -> bool:
        return True


class FailedAdapter:
    def __init__(self, ref: str, secret: str) -> None:
        self.ref = ref
        self._secret = secret

    async def probe(self) -> bool:
        raise RuntimeError(self._secret)


class HangingAdapter:
    def __init__(self, ref: str) -> None:
        self.ref = ref

    async def probe(self) -> bool:
        await asyncio.Event().wait()
        return True


def make_settings(tmp_path: Path) -> Settings:
    placeholder = tmp_path / "unused-pki"
    return Settings(
        control_plane_url="https://control.example",
        advertised_control_url="https://runtime.example:9443",
        advertised_a2a_url="https://runtime.example:9444",
        ca_file=placeholder,
        certificate_file=placeholder,
        private_key_file=placeholder,
    )
