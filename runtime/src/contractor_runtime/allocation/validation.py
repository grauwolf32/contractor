"""Allocation admission against resolved configuration and frozen capabilities."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from contractor_runtime.allocation.errors import AllocationError
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import AllocationSpec, normalize_run_metadata_labels
from contractor_runtime.digests import (
    TemplateDigestMismatch,
    verify_model_policy_digest,
    verify_template_digests,
)
from contractor_runtime.factories import (
    FactoryRegistry,
    WorkerRuntimeFactory,
)
from contractor_runtime.sandbox.contracts import SandboxContractError, validate_sandbox_selection

RESERVED_NAMESPACES = frozenset({"inputs", "outputs", "skills"})


def validate_spec(
    spec: AllocationSpec,
    *,
    capabilities: CapabilitySnapshot,
    factories: FactoryRegistry,
    now: Callable[[], datetime],
    runtime_factory: Callable[[AllocationSpec], WorkerRuntimeFactory],
) -> None:
    audit_tools = any(
        selection.ref.toolset_id == "audit-results" and selection.ref.version == "2"
        for selection in spec.agent_template.toolsets
    )
    contract = spec.completion_contract
    if contract is not None:
        try:
            contract.validate_allocation(spec.namespace, spec.agent_template)
        except ValueError:
            raise AllocationError(
                "invalid_worker_completion",
                "Invalid Audit completion selection",
                retryable=False,
                status_code=422,
            ) from None
    if (audit_tools and contract is None) or (
        contract is not None
        and (
            contract.kind not in capabilities.completion_contracts
            or not getattr(runtime_factory(spec), "supports_worker_completion", False)
        )
    ):
        raise AllocationError(
            "unsupported_worker_completion",
            "Audit completion requires trusted supported preparation",
            retryable=False,
            status_code=422,
        )
    provider = factories.workspace_provider
    try:
        validate_sandbox_selection(spec, provider.capability.storage if provider else None)
    except SandboxContractError as error:
        raise AllocationError(
            error.code.value,
            "incompatible sandbox, Toolset or workspace selection",
            retryable=False,
            status_code=422,
        ) from None
    try:
        normalize_run_metadata_labels(spec.run_metadata_labels)
    except (TypeError, ValueError, AttributeError, UnicodeError):
        raise AllocationError(
            "invalid_run_metadata_labels",
            "allocation Run metadata labels are invalid",
            retryable=False,
            status_code=422,
        ) from None
    if spec.namespace in RESERVED_NAMESPACES:
        raise AllocationError(
            "invalid_agent_namespace",
            "agent allocation cannot use a Run-reserved namespace",
            retryable=False,
            status_code=422,
        )
    if spec.lease_expires_at <= now():
        raise AllocationError(
            "allocation_lease_expired",
            "allocation lease has already expired",
            retryable=True,
            status_code=409,
        )
    try:
        verify_template_digests(spec.agent_template)
        verify_model_policy_digest(spec.model_policy)
    except TemplateDigestMismatch:
        raise AllocationError(
            "template_digest_mismatch",
            "resolved AgentTemplate integrity verification failed",
            retryable=False,
            status_code=422,
        ) from None

    runtime = spec.agent_template.runtime
    runtime_ref = f"{runtime.runtime_id}@{runtime.version}"
    if not capabilities.supports_runtime(runtime_ref):
        raise AllocationError(
            "unsupported_worker_runtime",
            "AgentTemplate selects an unavailable WorkerRuntime",
            retryable=False,
            status_code=422,
        )
    runtime_factory = factories.worker_runtimes.get(runtime_ref)
    if spec.resolved_skills and not bool(getattr(runtime_factory, "supports_agent_skills", False)):
        raise AllocationError(
            "skill_runtime_unsupported",
            "selected WorkerRuntime does not support Agent Skills",
            retryable=False,
            status_code=422,
        )
    sandbox = spec.agent_template.sandbox_profile
    sandbox_ref = f"{sandbox.sandbox_profile_id}@{sandbox.version}"
    if not capabilities.supports_sandbox(sandbox_ref):
        raise AllocationError(
            "unsupported_sandbox_profile",
            "AgentTemplate selects an unavailable SandboxProfile",
            retryable=False,
            status_code=422,
        )
    if sandbox_ref == "podman@1" and factories.execution_lifecycle is None:
        raise AllocationError(
            "sandbox_unavailable",
            "sandbox lifecycle is unavailable",
            retryable=False,
            status_code=422,
        )
    required_infrastructure_channels: set[str] = set()
    for selection in spec.agent_template.toolsets:
        toolset_ref = f"{selection.ref.toolset_id}@{selection.ref.version}"
        if not capabilities.has_toolset(toolset_ref):
            raise AllocationError(
                "unsupported_toolset",
                "AgentTemplate selects an unavailable Toolset",
                retryable=False,
                status_code=422,
            )
        if not capabilities.supports_tools(toolset_ref, selection.tools):
            raise AllocationError(
                "unsupported_tool",
                "AgentTemplate selects an unavailable tool",
                retryable=False,
                status_code=422,
            )
        factory = factories.toolsets.get(toolset_ref)
        if factory is not None:
            required_infrastructure_channels.update(
                channel
                for tool in selection.tools
                for channel in factory.infrastructure_channels.get(tool, frozenset())
            )
        if getattr(factory, "requires_workspace", False) and spec.workspace is None:
            raise AllocationError(
                "workspace_required",
                "selected Toolset requires an allocation project workspace",
                retryable=False,
                status_code=422,
            )
        if (
            toolset_ref == "workspace-changes@1"
            and spec.workspace is not None
            and spec.workspace.mode != "overlay"
        ):
            raise AllocationError(
                "workspace_mode_unsupported",
                "selected Toolset requires overlay workspace mode",
                retryable=False,
                status_code=422,
            )

    if (
        "caido-graphql-client" in required_infrastructure_channels
        and spec.runtime_settings.caido is None
    ):
        raise AllocationError(
            "caido_not_configured",
            "selected Caido tools require resolved Runtime configuration",
            retryable=False,
            status_code=422,
        )
    if spec.workspace is not None:
        provider = factories.workspace_provider
        if (
            provider is None
            or capabilities.workspace is None
            or capabilities.workspace != provider.capability
            or not capabilities.supports_workspace_mode(spec.workspace.mode)
        ):
            raise AllocationError(
                "workspace_mode_unsupported",
                "allocation workspace mode is not available on this Runtime Agent",
                retryable=False,
                status_code=422,
            )
        if factories.artifact_client_factory is None:
            raise AllocationError(
                "workspace_source_unavailable",
                "allocation workspace Artifact client is unavailable",
                retryable=True,
                status_code=503,
            )
    required = set(spec.resolved_runtime_config_provenance.runtime_adapters)
    configured: set[str] = set()
    if spec.runtime_settings.telemetry is not None:
        configured.add(spec.runtime_settings.telemetry.adapter)
    if spec.runtime_settings.http_proxy is not None:
        configured.add(spec.runtime_settings.http_proxy.adapter)
    if spec.runtime_settings.caido is not None:
        configured.add(spec.runtime_settings.caido.adapter)
    if (
        required != configured
        or not capabilities.supports_runtime_adapters(required)
        or not required <= set(factories.runtime_adapters)
    ):
        raise AllocationError(
            "unsupported_runtime_adapter",
            "allocation Runtime adapter settings do not match frozen capabilities",
            retryable=False,
            status_code=422,
        )
