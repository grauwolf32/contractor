"""Resolved allocation fixtures with valid Contractor digests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from contractor_runtime.contracts import (
    API_VERSION,
    AgentTemplateRef,
    AllocationSpec,
    ModelPolicyRef,
    ResolvedAgentTemplate,
    ResolvedInstructions,
    ResolvedModelPolicy,
    RuntimeSettings,
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
) -> AllocationSpec:
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
    return AllocationSpec(
        apiVersion=API_VERSION,
        allocationId=allocation_id,
        runId="run-1",
        stageExecutionId="stage-execution-1",
        logicalAgentName="builder",
        namespace="builder",
        leaseExpiresAt=datetime.now(UTC) + timedelta(minutes=5),
        agentTemplate=template,
        runtimeSettings=RuntimeSettings(
            llmGatewayUrl="https://llm.example/v1",
            llmGatewayToken=secret,
            artifactApiUrl="https://control.example/private/v1",
            requestTimeoutSeconds=30,
        ),
    )
