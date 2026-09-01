"""Resolved allocation fixtures with valid Contractor digests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from contractor_runtime.contracts import (
    API_VERSION,
    AgentTemplateRef,
    AllocationSpecV2,
    LLMGatewayConfigRef,
    ModelPolicyRef,
    ResolvedAgentTemplate,
    ResolvedInstructions,
    ResolvedModelPolicy,
    ResolvedRuntimeConfigProvenanceV2,
    RuntimeConfigRefV2,
    RuntimeLabelBindingProvenanceV2,
    RuntimeSettingsV2,
    SandboxProfileRef,
    ToolsetRef,
    ToolsetSelection,
    WorkerRuntimeRef,
)
from contractor_runtime.digests import (
    _agent_template_digest,
    _digest_bytes,
    _model_policy_digest,
)


def allocation_spec(
    *,
    allocation_id: str = "allocation-1",
    secret: str = "recognizable-a2a-test-token",
    tools: list[str] | None = None,
) -> AllocationSpecV2:
    instructions = "Use only the selected tools and return a strict result."
    policy = ResolvedModelPolicy(
        ref=ModelPolicyRef(policyId="worker", version="1", digest="sha256:" + "0" * 64),
        model="worker-model",
        maxOutputTokens=4096,
        maxModelCalls=8,
        maxToolCalls=16,
        maxTotalTokens=32768,
        temperature=0.1,
    )
    policy.ref.digest = _model_policy_digest(policy)
    template = ResolvedAgentTemplate(
        ref=AgentTemplateRef(
            templateId="artifact_builder", version="1", digest="sha256:" + "0" * 64
        ),
        description="Builds a requested artifact",
        runtime=WorkerRuntimeRef(runtimeId="adk", version="1"),
        instructions=ResolvedInstructions(
            ref="instructions/artifact-builder.md",
            digest=_digest_bytes(instructions.encode()),
            text=instructions,
        ),
        modelPolicy=policy,
        toolsets=[
            ToolsetSelection(
                ref=ToolsetRef(toolsetId="run-artifacts", version="1"),
                tools=tools or ["read_artifact"],
            )
        ],
        sandboxProfile=SandboxProfileRef(sandboxProfileId="local-workdir", version="1"),
    )
    template.ref.digest = _agent_template_digest(template)
    return AllocationSpecV2(
        apiVersion=API_VERSION,
        allocationId=allocation_id,
        runId="run-1",
        stageExecutionId="stage-execution-1",
        logicalAgentName="builder",
        namespace="builder",
        leaseExpiresAt=datetime.now(UTC) + timedelta(minutes=5),
        agentTemplate=template,
        resolvedSkills=[],
        modelPolicy=policy.model_copy(deep=True),
        runtimeSettings=RuntimeSettingsV2(
            llmGatewayUrl="https://llm.example/v1",
            llmGatewayToken=secret,
            artifactApiUrl="https://control.example/private/v1",
            requestTimeoutSeconds=30,
        ),
        resolvedRuntimeConfigProvenance=ResolvedRuntimeConfigProvenanceV2(
            default=RuntimeLabelBindingProvenanceV2(
                label="default",
                bindingRevision=1,
                config=RuntimeConfigRefV2(
                    name="contractor-empty",
                    version="1",
                    digest="sha256:" + "a" * 64,
                ),
            ),
            runLabels=[],
            agentLabels=[],
            runtimeAdapters=[],
            llmGatewayConfig=LLMGatewayConfigRef(
                gatewayId="local-litellm",
                version="1",
                digest="sha256:" + "b" * 64,
            ),
            runtimeCredentialRefs=[],
        ),
    )
