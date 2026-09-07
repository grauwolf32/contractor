"""Startup capability discovery for one immutable Runtime Agent process."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from contractor_runtime.contracts import RUNTIME_ADAPTER_REFS, ToolsetCapability
from contractor_runtime.factories import FactoryRegistry
from contractor_runtime.projectfs import WorkspaceCapabilitySnapshot
from contractor_runtime.sandbox.contracts import EXECUTION_TOOLSET, PODMAN_PROFILE
from contractor_runtime.sandbox.podman.probe import PROBE_CLEANUP_SECONDS, PROBE_TIMEOUT_SECONDS
from contractor_runtime.telemetry.resources import SUPPORTED_PERFORMANCE_METRICS_VERSIONS

logger = logging.getLogger(__name__)

DEFAULT_FACTORY_PROBE_TIMEOUT_SECONDS = 5.0
DEFAULT_TOTAL_PROBE_TIMEOUT_SECONDS = 30.0
PODMAN_TOTAL_PROBE_TIMEOUT_SECONDS = 60.0


class CapabilityDiscoveryError(RuntimeError):
    """Safe startup failure that contains no local dependency details."""


@dataclass(frozen=True, slots=True)
class ToolsetCapabilitySnapshot:
    ref: str
    tools: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CapabilitySnapshot:
    """Normalized positive capabilities frozen for one process instance."""

    runtimes: tuple[str, ...]
    toolsets: tuple[ToolsetCapabilitySnapshot, ...]
    sandbox_profiles: tuple[str, ...]
    runtime_adapters: tuple[str, ...] = ()
    workspace: WorkspaceCapabilitySnapshot | None = None
    performance_metrics_versions: tuple[int, ...] = SUPPORTED_PERFORMANCE_METRICS_VERSIONS
    completion_contracts: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        runtimes: Iterable[str],
        toolsets: Mapping[str, Iterable[str]],
        sandbox_profiles: Iterable[str],
        runtime_adapters: Iterable[str] = (),
        workspace: WorkspaceCapabilitySnapshot | None = None,
        completion_contracts: Iterable[str] = (),
    ) -> CapabilitySnapshot:
        normalized_runtimes = tuple(sorted(set(runtimes)))
        normalized_sandboxes = tuple(sorted(set(sandbox_profiles)))
        normalized_adapters = tuple(sorted(set(runtime_adapters)))
        completion = tuple(sorted(set(completion_contracts)))
        if set(completion) - {"audit-check-results@1"} or (
            completion
            and (
                "adk@1" not in normalized_runtimes
                or not {"read_audit_task", "submit_check_result"}
                <= set(toolsets.get("audit-results@2", ()))
            )
        ):
            raise CapabilityDiscoveryError("invalid Worker completion capability snapshot")
        if len(normalized_adapters) > 64 or any(
            ref not in RUNTIME_ADAPTER_REFS for ref in normalized_adapters
        ):
            raise CapabilityDiscoveryError("invalid RuntimeAdapter capability snapshot")
        if not normalized_runtimes:
            raise CapabilityDiscoveryError("no usable WorkerRuntime capability")
        if not normalized_sandboxes:
            raise CapabilityDiscoveryError("no usable SandboxProfile capability")
        normalized_toolsets: list[ToolsetCapabilitySnapshot] = []
        for ref in sorted(toolsets):
            tools = tuple(sorted(set(toolsets[ref])))
            if tools:
                normalized_toolsets.append(ToolsetCapabilitySnapshot(ref=ref, tools=tools))
        return cls(
            runtimes=normalized_runtimes,
            toolsets=tuple(normalized_toolsets),
            sandbox_profiles=normalized_sandboxes,
            runtime_adapters=normalized_adapters,
            workspace=workspace,
            completion_contracts=completion,
        )

    def wire_toolsets(self) -> list[ToolsetCapability]:
        return [
            ToolsetCapability(ref=capability.ref, tools=list(capability.tools))
            for capability in self.toolsets
        ]

    def supports_runtime(self, ref: str) -> bool:
        return ref in self.runtimes

    def supports_sandbox(self, ref: str) -> bool:
        return ref in self.sandbox_profiles

    def has_toolset(self, ref: str) -> bool:
        return any(capability.ref == ref for capability in self.toolsets)

    def supports_tools(self, ref: str, tools: Iterable[str]) -> bool:
        selected = set(tools)
        for capability in self.toolsets:
            if capability.ref == ref:
                return selected <= set(capability.tools)
        return False

    def supports_runtime_adapters(self, refs: Iterable[str]) -> bool:
        return set(refs) <= set(self.runtime_adapters)

    def supports_workspace_mode(self, mode: str) -> bool:
        return self.workspace is not None and mode in self.workspace.modes


async def discover_capabilities(
    factories: FactoryRegistry,
    *,
    per_factory_timeout_seconds: float = DEFAULT_FACTORY_PROBE_TIMEOUT_SECONDS,
    total_timeout_seconds: float | None = None,
    podman_timeout_seconds: float = PROBE_TIMEOUT_SECONDS,
) -> CapabilitySnapshot:
    """Probe enabled factories once and return their normalized positive set."""

    if total_timeout_seconds is None:
        total_timeout_seconds = (
            PODMAN_TOTAL_PROBE_TIMEOUT_SECONDS
            if factories.execution_lifecycle is not None
            else DEFAULT_TOTAL_PROBE_TIMEOUT_SECONDS
        )
    if any(
        not math.isfinite(value) or value <= 0
        for value in (per_factory_timeout_seconds, total_timeout_seconds, podman_timeout_seconds)
    ):
        raise ValueError("capability probe timeouts must be positive")

    runtimes: list[str] = []
    toolsets: dict[str, frozenset[str]] = {}
    sandboxes: list[str] = []
    runtime_adapters: list[str] = []
    workspace: WorkspaceCapabilitySnapshot | None = None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + total_timeout_seconds

    # This is a startup gate, not an optional capability probe: failure must
    # prevent workspace-provider initialization and its orphan deletion.
    if factories.execution_lifecycle is not None:
        await factories.execution_lifecycle.recover(deadline=deadline)

    for ref, factory in sorted(factories.worker_runtimes.items()):
        result = await _probe_one(
            "runtime", ref, factory.probe, deadline, per_factory_timeout_seconds
        )
        if result is True:
            runtimes.append(ref)

    for ref, factory in sorted(factories.sandbox_profiles.items()):
        if ref == PODMAN_PROFILE:
            continue  # requires the real local workspace probe first
        result = await _probe_one(
            "sandbox", ref, factory.probe, deadline, per_factory_timeout_seconds
        )
        if result is True:
            sandboxes.append(ref)

    if factories.workspace_provider is not None:
        result = await _probe_one(
            "workspace",
            "workspace-storage",
            factories.workspace_provider.probe,
            deadline,
            per_factory_timeout_seconds,
        )
        if result is True:
            workspace = factories.workspace_provider.capability

    if (
        factories.execution_lifecycle is not None
        and workspace is not None
        and workspace.storage == "local"
        and "direct" in workspace.modes
    ):
        await _probe_podman(factories, deadline, podman_timeout_seconds)
        factory = factories.sandbox_profiles.get(PODMAN_PROFILE)
        if factory is not None and factories.execution_lifecycle.probe_available:
            result = await _probe_one(
                "sandbox", PODMAN_PROFILE, factory.probe, deadline, per_factory_timeout_seconds
            )
            if result is True:
                sandboxes.append(PODMAN_PROFILE)

    for ref, factory in sorted(factories.toolsets.items()):
        if ref == EXECUTION_TOOLSET and PODMAN_PROFILE not in sandboxes:
            continue
        if getattr(factory, "requires_workspace", False) and workspace is None:
            continue
        result = await _probe_one(
            "toolset", ref, factory.probe, deadline, per_factory_timeout_seconds
        )
        if not isinstance(result, (set, frozenset)):
            continue
        available = frozenset(result)
        if not available or not available <= factory.exported_tools:
            if available:
                _log_probe(ref, "toolset", "invalid_result", 0)
            continue
        toolsets[ref] = available

    for ref, factory in sorted(factories.runtime_adapters.items()):
        result = await _probe_one(
            "runtime_adapter", ref, factory.probe, deadline, per_factory_timeout_seconds
        )
        if result is True:
            runtime_adapters.append(ref)

    if PODMAN_PROFILE in sandboxes and (
        EXECUTION_TOOLSET not in toolsets or not factories.execution_lifecycle.probe_available
    ):
        sandboxes.remove(PODMAN_PROFILE)
        toolsets.pop(EXECUTION_TOOLSET, None)

    return CapabilitySnapshot.create(
        runtimes=runtimes,
        toolsets=toolsets,
        sandbox_profiles=sandboxes,
        runtime_adapters=runtime_adapters,
        workspace=workspace,
        completion_contracts=(
            ("audit-check-results@1",)
            if "adk@1" in runtimes
            and getattr(factories.worker_runtimes.get("adk@1"), "supports_worker_completion", False)
            and {"read_audit_task", "submit_check_result"}
            <= set(toolsets.get("audit-results@2", ()))
            else ()
        ),
    )


async def _probe_podman(factories, deadline, timeout):
    loop = asyncio.get_running_loop()
    end = min(deadline, loop.time() + timeout)
    if end - loop.time() <= PROBE_CLEANUP_SECONDS:
        _log_probe(PODMAN_PROFILE, "sandbox", "total_timeout", 0)
        return
    lifecycle, provider = factories.execution_lifecycle, factories.workspace_provider
    started = loop.time()
    try:
        async with asyncio.timeout(end - started):
            storage = await provider.create("podman-capability-probe")
            root = Path(storage.root) / "run_workdir"
            await asyncio.to_thread(root.mkdir, mode=0o700)
            result = await lifecycle.probe(root, deadline=end)
            # The only receipt authorizing bind deletion, including negative
            # capabilities. Cancellation/uncertainty deliberately retains it.
            await provider.cleanup(storage)
    except asyncio.CancelledError:
        lifecycle.reject_probe()
        raise
    except Exception:
        lifecycle.reject_probe()
        raise CapabilityDiscoveryError("Podman probe cleanup is unconfirmed") from None
    _log_probe(
        PODMAN_PROFILE,
        "sandbox",
        "available" if result["available"] else result["failure"],
        _duration_ms(started, loop.time()),
    )
    if result["available"]:
        logger.info(
            "Runtime Podman effective policy verified",
            extra=getattr(lifecycle, "probe_diagnostics", {}),
        )


async def _probe_one(
    kind: str,
    ref: str,
    probe: object,
    deadline: float,
    per_factory_timeout_seconds: float,
) -> object | None:
    loop = asyncio.get_running_loop()
    started = loop.time()
    remaining = deadline - started
    if remaining <= 0:
        _log_probe(ref, kind, "total_timeout", 0)
        return None
    timeout = min(per_factory_timeout_seconds, remaining)
    try:
        result = await asyncio.wait_for(probe(), timeout=timeout)  # type: ignore[operator]
    except TimeoutError:
        _log_probe(ref, kind, "timeout", _duration_ms(started, loop.time()))
        return None
    except asyncio.CancelledError:
        raise
    except Exception:
        _log_probe(ref, kind, "failed", _duration_ms(started, loop.time()))
        return None
    _log_probe(ref, kind, "available", _duration_ms(started, loop.time()))
    return result


def _duration_ms(started: float, finished: float) -> int:
    return max(0, min(int((finished - started) * 1000), 2_147_483_647))


def _log_probe(ref: str, kind: str, outcome: str, duration_ms: int) -> None:
    logger.info(
        "Runtime capability probe %s",
        outcome,
        extra={
            "capabilityRef": ref,
            "capabilityKind": kind,
            "probeOutcome": outcome,
            "durationMs": duration_ms,
        },
    )
