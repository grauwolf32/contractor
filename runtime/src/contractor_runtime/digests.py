"""Cross-language RFC 8785 digest verification for resolved templates."""

from __future__ import annotations

import hashlib
from typing import Any

import jcs

from contractor_runtime.contracts import (
    API_VERSION,
    ResolvedAgentTemplate,
    ResolvedLLMGatewayConfig,
    ResolvedModelPolicy,
)


class TemplateDigestMismatch(ValueError):
    """A trusted Control Plane sent an internally inconsistent template body."""


class GatewayDigestMismatch(ValueError):
    """A trusted Control Plane sent an inconsistent non-secret Gateway body."""


def verify_template_digests(template: ResolvedAgentTemplate) -> None:
    instruction_digest = _digest_bytes(template.instructions.text.encode("utf-8"))
    if instruction_digest != template.instructions.digest:
        raise TemplateDigestMismatch("resolved instruction digest does not match its text")

    policy_digest = _model_policy_digest(template.model_policy)
    if policy_digest != template.model_policy.ref.digest:
        raise TemplateDigestMismatch("resolved ModelPolicy digest does not match its body")

    template_digest = _agent_template_digest(template)
    if template_digest != template.ref.digest:
        raise TemplateDigestMismatch("resolved AgentTemplate digest does not match its body")


def verify_gateway_config_digest(gateway: ResolvedLLMGatewayConfig) -> None:
    if _llm_gateway_config_digest(gateway) != gateway.ref.digest:
        raise GatewayDigestMismatch("resolved LLMGatewayConfig digest does not match its body")


def _model_policy_digest(policy: ResolvedModelPolicy) -> str:
    spec: dict[str, Any] = {
        "model": policy.model,
        "maxOutputTokens": policy.max_output_tokens,
        "maxModelCalls": policy.max_model_calls,
        "maxToolCalls": policy.max_tool_calls,
        "maxTotalTokens": policy.max_total_tokens,
    }
    if policy.temperature is not None:
        spec["temperature"] = policy.temperature
    manifest = {
        "apiVersion": API_VERSION,
        "kind": "ModelPolicy",
        "metadata": {"name": policy.ref.policy_id, "version": policy.ref.version},
        "spec": spec,
    }
    return _digest_jcs(manifest)


def _llm_gateway_config_digest(gateway: ResolvedLLMGatewayConfig) -> str:
    spec: dict[str, Any] = {"protocol": gateway.protocol, "url": gateway.url}
    if gateway.credential_manager is not None:
        spec["credentialManager"] = {
            "implementation": gateway.credential_manager.implementation,
            "managementUrl": gateway.credential_manager.management_url,
        }
    return _digest_jcs(
        {
            "apiVersion": API_VERSION,
            "kind": "LLMGatewayConfig",
            "metadata": {"name": gateway.ref.gateway_id, "version": gateway.ref.version},
            "spec": spec,
        }
    )


def _agent_template_digest(template: ResolvedAgentTemplate) -> str:
    policy: dict[str, Any] = {
        "ref": {
            "policyId": template.model_policy.ref.policy_id,
            "version": template.model_policy.ref.version,
            "digest": template.model_policy.ref.digest,
        },
        "model": template.model_policy.model,
        "maxOutputTokens": template.model_policy.max_output_tokens,
        "maxModelCalls": template.model_policy.max_model_calls,
        "maxToolCalls": template.model_policy.max_tool_calls,
        "maxTotalTokens": template.model_policy.max_total_tokens,
    }
    if template.model_policy.temperature is not None:
        policy["temperature"] = template.model_policy.temperature

    toolsets = [
        {
            "ref": {
                "toolsetId": selection.ref.toolset_id,
                "version": selection.ref.version,
            },
            "tools": sorted(selection.tools),
        }
        for selection in sorted(
            template.toolsets,
            key=lambda item: f"{item.ref.toolset_id}@{item.ref.version}",
        )
    ]
    manifest = {
        "apiVersion": API_VERSION,
        "kind": "AgentTemplate",
        "metadata": {"name": template.ref.template_id, "version": template.ref.version},
        "spec": {
            "description": template.description,
            "runtime": {
                "runtimeId": template.runtime.runtime_id,
                "version": template.runtime.version,
            },
            "instructions": {
                "ref": template.instructions.ref,
                "digest": template.instructions.digest,
            },
            "modelPolicy": policy,
            "toolsets": toolsets,
            "sandboxProfile": {
                "sandboxProfileId": template.sandbox_profile.sandbox_profile_id,
                "version": template.sandbox_profile.version,
            },
        },
    }
    return _digest_jcs(manifest)


def _digest_jcs(value: Any) -> str:
    canonical = jcs.canonicalize(value)
    return _digest_bytes(canonical)


def _digest_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"
