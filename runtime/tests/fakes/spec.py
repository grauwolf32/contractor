"""Resolved allocation fixtures with valid Contractor digests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from contractor_runtime.contracts import (
    API_VERSION,
    AgentTemplateRef,
    AllocationSpec,
    ArtifactRef,
    ModelPolicyRef,
    NoWorkerContextManagement,
    ResolvedAgentTemplate,
    ResolvedInstructions,
    ResolvedModelPolicy,
    ResolvedRuntimeConfigProvenance,
    ResolvedSkill,
    RuntimeSettings,
    SandboxProfileRef,
    TerminalWorkerContextManagement,
    ToolsetRef,
    ToolsetSelection,
    WorkerContextManagement,
    WorkerRuntimeRef,
    WorkerSessionMode,
)
from contractor_runtime.contracts.settings import WorkerSummarizerSettings
from contractor_runtime.digests import (
    _agent_template_digest,
    _model_policy_digest,
    sha256_digest,
)


def allocation_spec(
    *,
    allocation_id: str = "allocation-1",
    secret: str = "recognizable-a2a-test-token",
    tools: list[str] | None = None,
    skills: list[ResolvedSkill] | None = None,
    summarizer: bool = False,
    context_management: WorkerContextManagement | None = None,
) -> AllocationSpec:
    skills = skills or []
    instructions = "Use only the selected tools and return a strict result."
    policy = ResolvedModelPolicy(
        ref=ModelPolicyRef(policyId="worker", version="1", digest="sha256:" + "0" * 64),
        model="worker-model",
        **({"contextWindowTokens": 131_072} if summarizer else {}),
        maxOutputTokens=4096,
        maxModelCalls=8,
        maxToolCalls=16,
        maxTotalTokens=32768,
        temperature=0.1,
    )
    policy.ref.digest = _model_policy_digest(policy)
    summarizer_config: WorkerSummarizerSettings | None = None
    summarizer_policy = None
    if summarizer:
        summarizer_policy = ResolvedModelPolicy(
            ref=ModelPolicyRef(
                policyId="terminal_summarizer",
                version="1",
                digest="sha256:" + "0" * 64,
            ),
            model="worker-summarizer-model",
            contextWindowTokens=131_072,
            maxOutputTokens=2048,
            maxModelCalls=1,
            temperature=0.1,
        )
        summarizer_policy.ref.digest = _model_policy_digest(summarizer_policy)
        summarizer_config = WorkerSummarizerSettings(
            contextWindowRatio=0.9,
            cumulativeBudget=20_000,
            includeReasoning=False,
        )
    template = ResolvedAgentTemplate(
        ref=AgentTemplateRef(
            templateId="artifact_builder", version="1", digest="sha256:" + "0" * 64
        ),
        description="Builds a requested artifact",
        runtime=WorkerRuntimeRef(runtimeId="adk", version="1"),
        instructions=ResolvedInstructions(
            ref="instructions/artifact-builder.md",
            digest=sha256_digest(instructions.encode()),
            text=instructions,
        ),
        **({"summarizer": summarizer_config} if summarizer_config is not None else {}),
        toolsets=[
            ToolsetSelection(
                ref=ToolsetRef(toolsetId="run-artifacts", version="1"),
                tools=tools or ["read_artifact"],
            )
        ],
        skills=[ArtifactRef(namespace="skills", name=skill.name) for skill in skills],
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
        workerSessionMode=WorkerSessionMode.ISOLATED,
        contextManagement=context_management
        or (
            TerminalWorkerContextManagement(strategy="terminal@1")
            if summarizer
            else NoWorkerContextManagement(strategy="none")
        ),
        runMetadataLabels={},
        leaseExpiresAt=datetime.now(UTC) + timedelta(minutes=5),
        agentTemplate=template,
        resolvedSkills=skills,
        modelPolicy=policy.model_copy(deep=True),
        **({"summarizerModelPolicy": summarizer_policy} if summarizer_policy else {}),
        runtimeSettings=RuntimeSettings(
            llmGatewayUrl="https://llm.example/v1",
            llmGatewayToken=secret,
            artifactApiUrl="https://control.example/private/v1",
            requestTimeoutSeconds=30,
        ),
        resolvedRuntimeConfigProvenance=runtime_provenance(),
    )


# Read once at import: tests use the shared fixture as input data, so no test
# opens it (tests/shared_fixtures.py).
_EMPTY_PROVENANCE = (
    Path(__file__).resolve().parents[3]
    / "api/testdata/v1alpha1/valid/runtime-provenance-empty.json"
).read_bytes()


def runtime_provenance() -> ResolvedRuntimeConfigProvenance:
    return ResolvedRuntimeConfigProvenance.model_validate_json(_EMPTY_PROVENANCE)


def summarizer_config():
    from contractor_runtime.allocation.preparation import _effective_summarizer

    result = _effective_summarizer(allocation_spec(summarizer=True))
    if result is None:
        raise RuntimeError("test allocation has no effective summarizer")
    return result
