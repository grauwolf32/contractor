"""Startup capability discovery for one immutable Runtime Agent process."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from contractor_runtime.contracts import RUNTIME_ADAPTER_REFS, ToolsetCapability
from contractor_runtime.factories import FactoryRegistry

logger = logging.getLogger(__name__)

DEFAULT_FACTORY_PROBE_TIMEOUT_SECONDS = 5.0
DEFAULT_TOTAL_PROBE_TIMEOUT_SECONDS = 30.0


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

    @classmethod
    def create(
        cls,
        *,
        runtimes: Iterable[str],
        toolsets: Mapping[str, Iterable[str]],
        sandbox_profiles: Iterable[str],
        runtime_adapters: Iterable[str] = (),
    ) -> CapabilitySnapshot:
        normalized_runtimes = tuple(sorted(set(runtimes)))
        normalized_sandboxes = tuple(sorted(set(sandbox_profiles)))
        normalized_adapters = tuple(sorted(set(runtime_adapters)))
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


async def discover_capabilities(
    factories: FactoryRegistry,
    *,
    per_factory_timeout_seconds: float = DEFAULT_FACTORY_PROBE_TIMEOUT_SECONDS,
    total_timeout_seconds: float = DEFAULT_TOTAL_PROBE_TIMEOUT_SECONDS,
) -> CapabilitySnapshot:
    """Probe enabled factories once and return their normalized positive set."""

    if per_factory_timeout_seconds <= 0 or total_timeout_seconds <= 0:
        raise ValueError("capability probe timeouts must be positive")

    runtimes: list[str] = []
    toolsets: dict[str, frozenset[str]] = {}
    sandboxes: list[str] = []
    runtime_adapters: list[str] = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + total_timeout_seconds

    for ref, factory in sorted(factories.worker_runtimes.items()):
        result = await _probe_one(
            "runtime", ref, factory.probe, deadline, per_factory_timeout_seconds
        )
        if result is True:
            runtimes.append(ref)

    for ref, factory in sorted(factories.sandbox_profiles.items()):
        result = await _probe_one(
            "sandbox", ref, factory.probe, deadline, per_factory_timeout_seconds
        )
        if result is True:
            sandboxes.append(ref)

    for ref, factory in sorted(factories.toolsets.items()):
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

    return CapabilitySnapshot.create(
        runtimes=runtimes,
        toolsets=toolsets,
        sandbox_profiles=sandboxes,
        runtime_adapters=runtime_adapters,
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
