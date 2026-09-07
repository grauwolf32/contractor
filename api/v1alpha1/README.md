# Contractor private wire contracts v1alpha1

These JSON Schema documents describe Contractor-owned payloads exchanged
between the Go Server and Python Runtime Agent. Every standalone request and
response carries `"apiVersion": "contractor/v1alpha1"`, uses camelCase fields,
and rejects unknown fields. Reusable resolved values such as
`ResolvedLLMGatewayConfig` are embedded components rather than standalone
messages, so their version is carried by their exact digest-bearing ref.

A2A envelopes themselves are owned by the official A2A 1.0 SDKs. The deployed
adapter receives `StageContentRequest` in an A2A DataPart with media type
`application/vnd.contractor.stage-content+json` and returns the Runtime-owned
`WorkerCompletion` with media type
`application/vnd.contractor.worker-completion+json`.

The schemas are review artifacts and compatibility contracts. Go and Python
DTOs are maintained explicitly and are checked against shared golden fixtures
under `api/testdata/v1alpha1`.

`AllocationSpec.agentTemplate.modelPolicy` remains the immutable template
default covered by the AgentTemplate digest. The separate required
`AllocationSpec.modelPolicy` is the effective Worker policy pinned by the Run;
Runtime validates its own digest and uses it without mutating or re-signing the
AgentTemplate. `RuntimeSettings.llmGatewayToken` may be the empty string for an
explicitly unauthenticated Gateway.

Every `AllocationSpec` also carries a required `runMetadataLabels` object. An
unlabeled Run sends `{}`. Server and Runtime validate the shared 32-entry,
key-shape and UTF-8 byte bounds; Runtime exposes a detached immutable copy only
to allocation telemetry, never to Worker instructions, ADK State or tools.

Every `AllocationSpec` carries a required `workerSessionMode` equal to
`isolated` or `shared`. It is the already-resolved immutable Stage policy; the
private wire has no omission default and Runtime rejects a missing or unknown
value before constructing the Worker.

`AllocationSpec.agentTemplate.summarizer`, when present, pins one separately
digested tool-free ModelPolicy and a normalized context-window trigger. It is part
of the AgentTemplate digest and uses the same allocation `RuntimeSettings`
Gateway route and credential as the effective Worker. Its `cumulativeBudget`
must remain below both the template-default and effective Worker
`maxTotalTokens`; it measures cumulative provider-reported total tokens across
one normal Worker invocation. `contextWindowRatio` is required on the resolved
wire (authoring omission defaults to `0.9`) and combines with the effective
Worker ModelPolicy's required `contextWindowTokens` and `maxOutputTokens` to
derive the prompt boundary. Omission of the complete summarizer block is the
only disabled representation.

An optional `summarizer.instructions` contains the resolved `ref`, `digest`, and
`text`, like Worker instructions. Its text is nonblank and at most 8,000 Unicode
characters, and its ref/digest participate in the AgentTemplate digest. Runtime
verifies the instruction text digest and uses the text literally as the summary
agent's instruction. Omission preserves the legacy built-in instruction. This
character limit does not guarantee a 2,000-token upper bound.

`AgentStateSnapshot.state.schemaVersion` is `2`: invocation metrics include a
required nullable `latestPromptTokens`, and every current/completed invocation
contains a closed summarizer phase/usage record. The HTTP ETag's existing
`contractor-agent-state-v1-*` prefix versions cache semantics independently.

`llm-gateway-config-manifest.schema.json` describes the strict non-secret YAML
manifest after YAML-to-JSON conversion. `llm-gateway-config.schema.json`
describes its normalized, digest-bearing resolved value on private wires.
`execution-config-manifest.schema.json` describes immutable Stage-local
escalation profiles, while `workflow-transition.schema.json` describes the
strict `on` block including bounded retry and inline-or-ref escalation.
