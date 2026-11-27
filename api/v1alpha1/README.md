# Contractor private wire contracts v1alpha1

`project-sitemap.schema.json` includes allocation-only typed mutations, snapshot
commands and one source-independent normalized collection command. Runtime tools
preprocess source reports and provider metadata. `SitemapCollectionRequest`
carries `input` (source label, observations, optional omission counts and an exact
current-Run evidence ArtifactRef) and a required composition-owned `automatic`
flag. Automatic collection and explicit imports both return `SitemapMutationResponse`.
Server has no report-format enum or provider-specific import endpoint. The shared direction-aware
`sitemap-wire-cases.json` exercises these wrappers; domain document definitions
are generated from the public owning schema.

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
`StageContentRequest.deadline` is required: every Planner sets it to the
effective invocation deadline (Worker call or Planner Stage). Runtime uses it as control data and does not
include it in the model prompt. The Stage result a Planner returns to the
Scheduler (`StageContentResult`) never crosses this boundary and is not part of
this catalog.

`planner@1` uses an ordinary separately accounted allocation and A2A invocation.
The optional `StageContentRequest.planner` contains the initial input, bounded
tool declarations and optional pinned compactor configuration; ordinary Workers
reject it. `planner-execution.schema.json` defines ordered `PlannerOperation`
and `PlannerOperationResult` messages for the existing mTLS allocation transport
at `POST /private/v1/allocations/{allocationId}/planner-operations`. A live host
caches only its last exact request/reply, permitting one identical transport
retry; process loss follows ordinary Stage interruption. The shared
`planner-execution-limits.json` pins the byte, tool and sequence limits.

The official A2A SDK represents DataPart numbers through protobuf Struct.
The A2A adapter restores exact integral values before strict DTO validation;
fractional and unsafe integral values remain invalid. Private HTTP JSON keeps
its original strict numeric rules. Planner invocation data is bounded to
512 KiB, Worker invocation data to 256 KiB; the A2A envelope is bounded to
4 MiB to account for SDK framing and escaping.

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
`required-member-cases.json` is generated from these schemas and the valid
fixtures: for every valid fixture it lists each member whose omission the
schema rejects, and every object member or map value whose `null` the schema rejects. Go,
Python and the schema reject every case. Go DTO fields declare such members
with `strictjson:"required"` (or `strictjson:"required,nullable"` where the
schema allows `null`), which `strictjson.Decode` enforces, and a test pins the
tags to the `required` lists. Private decoding uses `strictjson.DecodeNonNull`,
with `strictjson:"nullable"` for an optional schema-declared nullable member;
ordinary public `strictjson.Decode` keeps its nullable request semantics.
Python models declare required members without defaults and reject an explicit
null for optional members outside their schema-declared `nullable_fields`.
`mutation-cases.json` is generated the same way (`go test ./internal/contracts
-run TestSharedMutationCasesAreRejected -args -update-mutation-cases`): for
every valid fixture it lists the members whose value outside an `enum` or
`const`, the strings one character over `maxLength`, and the objects whose
added member or repeated key the schema, Go and Python must all reject.
The same catalog probes schema-forbidden numeric zeroes, including omitted
optional ModelPolicy limits, and both renamed and shadowing snake_case aliases
of closed-object camelCase members.
`maxLength` counts characters. Where Go and Python bound a string in UTF-8
bytes, the schema also states the byte bound as `x-maxBytes`, which the Go and
Python schema tests enforce, and the cases record the verdict all three reach
on a string of `maxLength` characters that is one byte too long.

A test that reads anything under `api/testdata` is named `TestShared*` (Go,
reading through `internal/contracts/contractstest`) or `test_shared_*`
(Runtime; see [Testing](../../docs/testing/README.md#focused-checks) for
fixtures read at import), and [`readers.json`](../testdata/readers.json)
declares which sides read each fixture. `make test-wire-cross-language` runs
every such test on both sides and fails when a side has no shared test, when
one did not run and pass, when a fixture is read by a side it does not declare
or not read by one it declares, or when a fixture has one reader and no
recorded reason.
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
(CTL-R152): a text field that must not be blank holds more than the Unicode
White_Space characters Go's `strings.TrimSpace` trims; the information
separators U+001C..U+001F are text, although Python's `str.isspace` counts
them, so the Runtime uses `WHITESPACE`, `is_blank` and `trim_whitespace`
instead of `str.strip`. `json-byte-size-cases.json` pins the compact UTF-8
JSON size that byte bounds measure (WRK-R37, WRK-R165;
`contracts.ResultJSONSize`, `json_byte_size`): U+2028 and U+2029 count three
bytes and `<`, `>` and `&` one.
`execution-error-cases.json` pins the 4 KiB UTF-8 text bound of error messages
and the compact JSON bound of tool arguments in both final reports and live
State, including multibyte boundaries and JSON escaping.

`memory-reservation-cases.json` holds the purpose-reserved Run Namespaces and,
per binding, whether it is a reserved Memory binding and whether a Stage or
Worker result may name it (ART-R114). Go `artifactpolicy` and the Worker
result check in `contracts`, and the Runtime's `contracts.artifacts`, which the
Worker, MemoryTools and allocation admission use, read it.

`runtime-settings-bounds-cases.json` holds every numeric `RuntimeSettings`
bound, the required `telemetry.export` and `export.retry` objects, and the
export values the Server sends when a Worker RuntimeConfig omits them
(RCF-R102); Go decoding, the Runtime model and this schema are checked against
it.

`credential-field-value-cases.json` holds the grammar of a credential token or
header value (RCF-R114): every ASCII character at each position, named values
and the byte bound of each field. Go `contracts.ValidCredentialFieldValue`, the
Runtime credential constructors, stored material decoding, LLM credential
tokens, the public create request, Go RuntimeSettings decoding, the Runtime's
`contracts.fields` and model, and the `CredentialFieldValue` definition of this
schema are checked against it; the Runtime also checks that h11 sends every
value it accepts.

`target-credential-cases.json` holds the engagement target credentials (RCF-R184).
Its references are `RuntimeCredentialRef` texts checked by Go decoding, the
Runtime model and the `RuntimeCredentialRef` definition of this schema; its
materials are the creation material of each target kind, checked by the
Server's constructors, stored material decoding, the public create request and
its OpenAPI material schemas; the `target-basic@1` cases are checked for
`http-origin-basic@1` and `http-proxy-basic@1` too, which share the Basic rule
(RCF-R114). The Runtime reads no target material yet.

`gateway-recovery-participant-cases.json` holds the model recovery participant
protocol of `POST /private/v1/allocations/{allocationID}/gateway-recovery`
(SCH-R168): the updates and which are terminal, the codes a `failed` update may
carry, the update grammar, the update each attempt outcome produces and the
update re-delivered after cancellation. The Server's `gatewayrecovery`, used by
the private API and the Server-side Planner, and the Runtime's recovery client
are checked against it.

`ArtifactListResult` is one page of versionless refs from the private
allocation listing. Its optional `nextCursor` (1 to 512 unpadded base64url
characters, never on an empty page) continues a paged listing and is absent on
the last page and from a `namePrefix` listing; the listing limit and cursor
grammar are pinned by `api/testdata/v1alpha1/artifact-list-cases.json`, which
also lists the bindings the paged listing hides from model tools.
`ArtifactReadOrWriteResult` is the body of a private write. The private
metadata read `GET …/artifacts/{namespace}/{name}/metadata` returns
`ArtifactMetadataResult` with the exact revision, media type, size and opaque
content cache key of a current or exact ref without the payload.

Raw and metadata reads default to the current Run. The exact query pair
`scope=project&operation=<Name>` selects an explicitly granted concrete binding
in its owning Project; the Server derives that Project from the durable Run.
`artifact-read-target-cases.json` holds the shared strict selector cases. The
same binding's `/access` endpoint accepts only this pair and returns `204` with
no body after current authority checks, without reading artifact content, metadata
or lineage. It allows domain-owned immutable views to check authority after
source history eviction; it creates no hold and cannot recover an evicted revision.
The owning contract is ART-R135.

AgentTemplate carries reusable behavior without main or summary model policies.
Required `AllocationSpec.modelPolicy` is the effective Worker policy selected by
Workflow and pinned by the Run. A terminal allocation also requires the separate
`summarizerModelPolicy`. Runtime validates their digests independently of the
AgentTemplate and rejects the removed template fields, including null.
`RuntimeSettings.llmGatewayToken` is omitted for an
unauthenticated Gateway; when present it is non-empty, and the Server, Runtime
and schema reject an empty token.

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
only disabled representation. Within a present summarizer, `includeReasoning`
is a required JSON boolean, including `false`. With `true` it admits the Worker
model's provider reasoning as tagged, redacted and bounded ordinary text.
Authoring omission normalizes to `false`; both sides reject omission, `null`
and non-boolean values on the resolved wire. Both values join the exact
AgentTemplate digest manifest.

An optional `summarizer.instructions` contains the resolved `ref`, `digest`, and
`text`, like Worker instructions. Its text is nonblank and at most 8,000 Unicode
characters, and its ref/digest participate in the AgentTemplate digest. Runtime
verifies the instruction text digest and uses the text literally as the summary
agent's instruction. Omission selects the built-in instruction. This
character limit does not guarantee a 2,000-token upper bound.

Every `AllocationSpec` carries a required `contextManagement` object, the one
effective context branch of the Worker (WRK-R184): `{"strategy": "none"}`
when the template has no summarizer, `{"strategy": "terminal@1"}` when it
has one, or `compact@1` with every member explicit beside the strategy:
the pinned tool-free compactor `modelPolicy` (exact ref, digest and body),
`instructions` (`ref`, `digest`, `text`), a JSON boolean `includeReasoning`,
`contextWindowRatio`, `keepRecentGroups` and `maxCompactions`. Other strategies
carry no members. Both sides reject a branch that disagrees with the template
summarizer, `compact@1` for a `tool@1` template, a `shared` session or Audit
completion, and `summarizerModelPolicy` outside `terminal@1`. The compactor
uses the allocation's Gateway route and credential, which the Server
authorizes for the compactor model before placement (CTL-R154). The shared cases
are `api/testdata/workflow-context-management-cases.json` and the
`allocation-spec-compact*` fixtures.

`AgentStateSnapshot.state.schemaVersion` is `2`: every current/completed invocation
contains bounded invocation metrics and a closed summarizer phase/usage record.
Prompt-token trigger tracking and workspace integrity digests stay internal to
their Runtime owners; the State contract rejects these unused exported fields. The Server cache ETag's existing
`contractor-agent-state-v1-*` prefix versions cache semantics independently.

`llm-gateway-config-manifest.schema.json` describes the strict non-secret YAML
manifest after YAML-to-JSON conversion. `llm-gateway-config.schema.json`
describes its normalized, digest-bearing resolved value inside the Go Server.
Escalation execution settings are inline in the owning Workflow and may reuse
standard YAML anchors within that document.

Registration requires `supportedRuntimeAdapters`, including an empty array when
none are available, and carries no RuntimeConfig label assignment. A new
certificate principal starts with empty labels at revision 1; assignments use
the public Operations API. The response contains only `apiVersion`,
`heartbeatIntervalSeconds` and `confirmedLeaseSeconds`. Durable `runtimeAgentId`,
labels and their revision remain Server-owned and are exposed through Operations.
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

`http-exchange.schema.json` describes the captured exchange submitted as
`httpEvidence` with an allocation finding. Its required paired
`httpEvidenceInvocationId` identifies the original capture invocation, independently
of the Finding's current `invocationId`; `#/$defs/selection` describes that pair.
The Runtime capture owner supplies it, and Server derives the Run from the allocation.
`#/$defs/artifact` describes the
frozen selected evidence, whose finding association is supplied by Server
intake. Go and Runtime validate `api/testdata/http-capture-cases.json`; the
Runtime schema check also pins the closed object structure.

The finding publisher may attach hidden `derived_anchors` (schema
`contractor.finding.anchors-derived.v1`) to the existing allocation-bound
finding submission. This is Runtime-owned private metadata, absent from all
Worker tool declarations and receipt/collection responses. The available basis
is the closed `FindingAnchorBasis`/`FindingStructuralAnchor` contract in the
public schema; unavailable values contain only schema, availability and reason
from FND-R173. Shared `finding-anchor-cases.json` exercises actual Runtime
parsing and strict Go validation, including unknown/null/missing members and
source/provenance mismatch. Generic Artifact writes have no producer authority.

`artifact-metadata.schema.json` defines the private allocation metadata response
`ArtifactMetadataResult`. Its required opaque `contentCacheKey` identifies the
immutable Server version across fork bindings, grants no access and is absent
from `ArtifactRef`, shared read/write results, public APIs and model tools.
The shared exact-content cache authorizes the consuming exact ref through this
strict response. Domain read views use the authority-only `/access` check described
above when retention-independent paging is required. The golden metadata fixture and invalid key fixtures test
both language readers; generated required-member and mutation cases cover
missing, duplicate, unknown and overlong members.

The engagement is a typed Project artifact. Its private allocation projection
in `engagement-allocation.schema.json` carries exact attribution and an optional
`http` assignment with scope, assigned identities, protected infrastructure and
any explicitly configured `limits`. HTTP limits come only from the engagement;
an omitted field is not supplied by process configuration or a default. Every
allocation enforces its own limits in Runtime. There is no per-request Server
permit/journal protocol or separate target transport-stop RPC. Ordinary
allocation cancellation/release joins target work and clears private state.

`shared-config.schema.json` describes the domain value returned by the typed
startup `ControlPlane.Bootstrap` RPC.
The Runtime verifies the digest of canonical `{limits,version}` and freezes
that snapshot before preparation and registration (CTL-R171, RTA-R126).
`make shared-config-generate` derives the golden fixture, Runtime test defaults
and Artifact schema/SQL ceilings from the Server-owned Go definition.


Allocation Project finding reads use the existing private Artifact transport:
`GET /allocations/{allocationID}/findings?limit=20&cursor=...`,
`GET /allocations/{allocationID}/findings/{findingID}?revision=N`, and
`POST /allocations/{allocationID}/finding-selections`. Requests never select a
Project ID. The selection body has `clientKey`, `findings` (explicit `findingId`
and exact `revision`) and `receiptIds`; arrays are required, with at most 128
selectors total. Live pages return `items`, `projectRevision`, `nextCursor`;
selection returns the existing exact collection publication descriptor.
FND-R224 through FND-R229 own bounds, owner/Project authority, replay and failures.
The shared findings collection fixture pins Project source kinds in both languages.


Native control uses `runtime.proto`: Bootstrap, Register, Heartbeat and the
allocation lifecycle including conditional ReadState. State carries a typed
revision and snapshot/not-modified oneof; the JSON document keeps its 4 MiB
ceiling inside a 4 MiB + 1 KiB protobuf ceiling. The old private HTTP routes
for these operations are removed. `grpc-limits.json`, `grpc-sequences.json`
and `private-rpc-origin-cases.json` pin cross-language transport limits, exact
unsigned heartbeat sequences and HTTPS origin grammar. Existing JSON report
schemas remain the validation boundary for named protobuf documents.
