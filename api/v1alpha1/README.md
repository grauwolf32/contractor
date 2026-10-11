# Contractor private wire contracts v1alpha1

These JSON Schema documents describe Contractor-owned payloads exchanged
between the Go Server and Python Runtime Agent. Every standalone request and
response carries `"apiVersion": "contractor/v1alpha1"`, uses camelCase fields,
and rejects unknown fields. Allocation messages carry an exact digest-bearing
`LLMGatewayConfigRef`; the resolved Gateway body stays on the Go Server.

A2A envelopes themselves are owned by the official A2A 1.0 SDKs. The deployed
adapter receives `StageContentRequest` in an A2A DataPart with media type
`application/vnd.contractor.stage-content+json` and returns the Runtime-owned
`WorkerCompletion` with media type
`application/vnd.contractor.worker-completion+json`.
`StageContentRequest.deadline` is optional for older callers; production
Planners set it to the effective Worker-call deadline. Runtime uses it as
control data and does not include it in the model prompt.

This is the single private contract catalog. There is no separate protocol
number, alternate DTO family or conversion path. HTTP routes retain their
`/private/v1` prefix. The schemas are review artifacts and wire definitions. Go and Python
DTOs are maintained explicitly and are checked against shared golden fixtures
under `api/testdata/v1alpha1`. Its `index.json` assigns each fixture a message
type and schema; each implementation validates the wire types it owns and
checks the listed rejections. Gateway body fixtures remain in this shared
catalog for Go Server schema and digest checks; Python Runtime validates the
Gateway ref and effective runtime settings instead. A new fixture needs an
index entry, and its type needs a Runtime model or an explicit Go-only reason
in `runtime/tests/test_contracts.py`. `digest-cases.json` holds resolved ModelPolicy and AgentTemplate
bodies checked by Go and Python, plus Server-only LLMGatewayConfig bodies,
including non-ASCII text and floating-point temperatures and ratios.
`agent-card-secret-scan-cases.json` holds the WorkerHandle Agent Card secret
scan both implementations apply: the trusted path table, the Runtime's card and
the cases each must accept or reject.
`runtime-capability-cases.json` holds the registration and heartbeat values
Operations shows: the tool-name and identity grammars and the capability
bounds that Go registration, the registration and heartbeat schemas, the
Runtime models and the public `RuntimeAgentObservation` schema share.

Three text rules hold for every field that applies them, with one definition
per language. `workspace-path-cases.json` pins the relative workspace path
grammar (WSP-R13) of workspace targets, the files a Worker reports reading and
Worker State paths (`contracts.RelativePathProblem`,
`contracts.base.relative_path_problem`, which the Runtime's project path
normalizer applies to the NFC form it records, and Studio's
`isWorkspaceTarget`): a Windows drive is an ASCII letter and a colon, so
`ж:docs` is an ordinary directory. `blank-text-cases.json` pins blank text
(CTL-R149): a text field that must not be blank holds more than the Unicode
White_Space characters Go's `strings.TrimSpace` trims; the information
separators U+001C..U+001F are text, although Python's `str.isspace` counts
them, so the Runtime uses `WHITESPACE`, `is_blank` and `trim_whitespace`
instead of `str.strip`. `json-byte-size-cases.json` pins the compact UTF-8
JSON size that byte bounds measure (WRK-R37, WRK-R164;
`contracts.ResultJSONSize`, `json_byte_size`): U+2028 and U+2029 count three
bytes and `<`, `>` and `&` one.

`ArtifactListResult` is one page of versionless refs from the private
allocation listing. Its optional `nextCursor` (1 to 512 unpadded base64url
characters, never on an empty page) continues a paged listing and is absent on
the last page and from a `namePrefix` listing; the listing limit and cursor
grammar are pinned by `api/testdata/v1alpha1/artifact-list-cases.json`, which
also lists the bindings the paged listing hides from model tools.

`AllocationSpec.agentTemplate.modelPolicy` remains the immutable template
default covered by the AgentTemplate digest. The separate required
`AllocationSpec.modelPolicy` is the effective Worker policy pinned by the Run;
Runtime validates its own digest and uses it without mutating or re-signing the
AgentTemplate. `RuntimeSettings.llmGatewayToken` is normally omitted for an
unauthenticated Gateway. The private wire validators also accept an explicit
empty string, which Runtime treats like an omitted token. The Server does not
emit empty resolved credentials.

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
agent's instruction. Omission selects the built-in instruction. This
character limit does not guarantee a 2,000-token upper bound.

`AgentStateSnapshot.state.schemaVersion` is `2`: invocation metrics include a
required nullable `latestPromptTokens`, and every current/completed invocation
contains a closed summarizer phase/usage record. The HTTP ETag's existing
`contractor-agent-state-v1-*` prefix versions cache semantics independently.

`llm-gateway-config-manifest.schema.json` describes the strict non-secret YAML
manifest after YAML-to-JSON conversion. `llm-gateway-config.schema.json`
describes its normalized, digest-bearing resolved value inside the Go Server.
`execution-config-manifest.schema.json` describes immutable Stage-local
escalation profiles, while `workflow-transition.schema.json` describes the
strict `on` block including bounded retry and inline-or-ref escalation.

Registration requires `initialLabels` and `supportedRuntimeAdapters`, including
empty arrays when nothing is selected. The response carries the durable
`runtimeAgentId`, authoritative `labels` and `labelRevision` alongside lease timing.
Every allocation requires `resolvedRuntimeConfigProvenance`; endpoint and credential
material belongs only to the separate `runtimeSettings` value. Optional settings
select telemetry, HTTP proxy, HTTP origin authentication and Caido adapters.

`runtime-report.schema.json` describes the current Runtime report and is referenced
by allocation final responses. Reports include an `adapters` object, which is empty
when no adapters were selected. Producer schemas reject malformed resource metrics;
receivers isolate invalid resource measurements and preserve valid execution facts
with a safe `invalid_report` diagnostic. Duplicate keys, malformed JSON and unknown
envelope fields remain errors.

Optional `supportedPerformanceMetricsVersions: [1]` advertises resource sampling.
An allocation requests it with `performanceMetrics: {version: 1, intervalSeconds: 15}`;
omission disables sampling. Shared producer fragments live in `performance.schema.json`.
Resource integers are bounded by `2^53 - 1`. A complete report requires CPU/RSS
boundaries, two successful RSS observations and no uncovered interval above 30 seconds.
Shared Go/Python cases also verify cross-field invariants beyond JSON Schema.
