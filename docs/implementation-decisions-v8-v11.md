# V8–V12 implementation decision log

This is a non-normative engineering log for the implementation sequence that
starts after the 2026-09-01 specification review. The working agreements in
[`docs/spec`](spec/README.md) remain authoritative. This file records choices
that are easy to lose in commit history, especially where two implementations
could both appear to satisfy a short task title.

Each completed implementation task still records the full hash of its final
task commit in `tasks/<id>.yml`. If implementation discovers a contract change,
the specification and task are updated before code relies on it, and the
reason is appended here.

Private protocol rollout entries D012/D013 describe the historical implementation
sequence. [V50-001](../tasks/v50-001-unified-private-contracts.yml) supersedes their
parallel DTOs and protocol number: all current fields now use the original model
names and the sole `contractor/v1alpha1` catalog. No old protocol is retained.

## Decisions accepted during the pre-implementation review

### D001 — Workflow owns workspace composition; Runtime owns storage

- Applies to: V11.
- Decision: a Stage declares logical source/state artifact aliases,
  `direct|overlay` semantics and optional overlay result slots. Scheduler pins
  exact RunScope refs into AllocationSpec. Runtime startup selects private
  `local|memory` storage and immutable resource limits.
- Reason: source artifacts and their revisions are Run execution authority,
  while disk versus RAM is a physical Runtime capability. Treating either as
  the other produced hidden project affinity and non-reproducible executions.

### D002 — Workspace is a separate allocation service

- Applies to: V11-004 through V11-012.
- Decision: preserve `contractor_runtime.workspace.AllocationWorkspace` as
  general `local-workdir@1` scratch. Implement project workspace behavior in
  `contractor_runtime.projectfs` and attach an optional `WorkspaceSession` to
  the Worker/Toolset build context.
- Reason: skills and existing domain tools already rely on allocation scratch;
  overloading it with semantic source state would mix cleanup and authority.

### D003 — ADK script support is absent, not merely inert

- Applies to: V10-003 and V10-005.
- Decision: retain Google ADK native Skill list/load/resource behavior, expose
  no `run_skill_script`, disable registry/search and use script-free guidance.
- Reason: the accepted Skill package profile contains nothing executable, so
  advertising an impossible tool only creates model errors.

### D004 — Source ZIP is the only v1 workspace import format

- Applies to: V11-005.
- Decision: each source is an exact `application/zip` Run artifact mapped below
  one non-overlapping relative target. Preserve archive structure and reject
  traversal, duplicates, links, special entries and bound violations. Local
  keeps ordinary binary files; memory skips them.
- Reason: a single deterministic archive format makes pinning, validation and
  multi-source composition testable without guessing source layout.

### D005 — Overlay state is cumulative text, not a revision chain

- Applies to: V11-006 and V11-010.
- Decision: one versioned JSON state artifact expresses canonical text-only
  operations from exact sources `S` to final tree `F` and carries base/result
  digests. It contains no previous artifact revision. Imported state plus `S`
  is sufficient on its own.
- Reason: ArtifactStore already owns revisions; leaking them into the model or
  state format would couple workspace semantics to storage history.

### D006 — Planner and Worker Memory share one ordered terminal barrier

- Applies to: V9-003 and V9-006.
- Decision: Planner memory mutations lock Run then Stage; Scheduler fences
  Worker grants before the same durable terminal order.
- Reason: this prevents late memory commits and database deadlocks.

### D007 — Router Workers receive independent physical workspaces

- Applies to: V11-003 and V11-011.
- Decision: every logical Worker gets the same exact initial refs but a separate
  local/memory copy. There is no live synchronization within a StageExecution.
  A later Stage may consume an explicitly persisted state revision.
- Reason: sharing one mutable tree would make routing order and parallelism
  change execution meaning.

### D008 — Direct mode is disposable; overlay is artifact-persisted

- Applies to: V11-006 through V11-010.
- Decision: direct edits immediately mutate the private allocation copy and are
  never auto-exported. Overlay edits remain in an upper; graceful terminal
  results export cumulative state plus checkpoint diff through the existing
  Artifact API. No tool mutates an operator checkout and no `materialize`
  operation exists.
- Reason: the useful persistence boundary is a versioned Run artifact, not a
  crash-recovery journal around external host files.

### D009 — Checkpoint advances only after complete graceful export

- Applies to: V11-010.
- Decision: imported state establishes checkpoint `B`. `diff`/rollback use
  `B`; exported state is `S -> F` and diff is `B -> F`. Both writes finish and
  exact refs are injected before terminal A2A response; then `F` becomes the
  next checkpoint. Failed/partial export does not advance it.
- Reason: this orders artifact mutation before Scheduler's write fence and
  supports sequential A2A tasks without losing cumulative reconstruction.

### D010 — Implementation and commit order

- Applies to: all unfinished V11–V12 tasks.
- Decision: implement in dependency-valid ID order. Each task gets one primary
  completion commit; a later metadata commit records its full hash without
  replacing that implementation hash.
- Reason: preserve bisectability and self-contained task evidence.

### D011 — HTTP forward proxy and Caido control API are distinct

- Applies to: V12.
- Decision: generic `http-tools@1` optionally consumes the existing `tool-http`
  forward-proxy handle. `caido@1` requires a new `caido-graphql@1` typed
  Runtime adapter configured by atomic labels with optional
  `caido-bearer@1` credential. Static operations only; large/raw bodies become
  artifacts and session secrets remain allocation-memory-only.
- Reason: a proxy URL cannot safely represent authenticated GraphQL control,
  and infrastructure selection must remain outside model input.

## Decisions discovered during implementation

Append entries here as `D012`, `D013`, and so on. Each entry should name the
owning task, alternatives considered, chosen behavior, compatibility impact and
tests that make the choice observable.

### D012 — Private protocol v2 lands inert and errors are opaque

- Applies to: V8-003 and V8-004.
- Decision: v2 has separate Go/Python DTO names and codecs while production
  registration remains v1 for all of V8-003. V8-004 must switch both peers and
  durable principal handling atomically. A v2 decode failure exposes only one
  of `version`, `duplicate_key`, `schema` or `invariant`; the original parser or
  validator exception is deliberately not unwrap-able.
- Provenance shape: default, Run-label and Agent-label bindings carry only
  revisions plus exact RuntimeConfig refs; adapter refs, exact Gateway/LLM
  credential refs and runtime credential ID/kind pairs are separate safe
  fields. Endpoint and secret-bearing settings cannot fit this closed model.
- Reason: activating request fields before the authoritative response and
  principal store exist creates an unsafe half-protocol. Retaining detailed
  secret-bearing validation causes is unnecessary once cross-language reason
  classes are test fixtures.
- Observable tests at the V8-003 boundary proved production still emitted v1;
  the retained fixture tests continue to prove canonical parity, duplicate
  rejection, error-class parity, redacted formatting, immutable adapter
  discovery and detached Operations projection. D013 owns the later cut-over.

### D013 — Protocol-v2 activation follows authoritative data ownership

- Applies to: V8-004, V8-007 and V8-010.
- Decision: V8-004 atomically activates the complete v2 registration request
  and response on Go and Python because durable principal labels now exist. It
  does not invent an AllocationSpec provenance document: AllocationSpec is
  activated by V8-007 only after the candidate-specific transaction can commit
  every exact binding/config/credential ref before delivery. RuntimeReport
  adapter counters become authoritative with the adapter lifecycle host in
  V8-010. The private HTTP routes and shared `apiVersion` remain unchanged;
  there is no v1/v2 registration negotiation or compatibility response.
- Alternatives rejected: an all-empty “bootstrap provenance” would make audit
  data false, while accepting optional provenance would turn a required
  security record into a silent compatibility mode. Activating secret-bearing
  settings before V8-007 would also send values that have no durable
  allocation-time linearization point.
- Compatibility impact: a v1 Runtime registration now fails closed. Existing
  Allocation lifecycle messages remain at their prior shape only until the
  already-planned V8-007 synchronized switch.
- Observable tests: `make test-runtime-wire` covers the v1 registration
  rejection and v2 emission/response parsing; V8-007's placement tests must
  cover the AllocationSpec cut-over before its completion commit.

### D014 — A Run pins a compact exact reference snapshot

- Applies to: V8-005 through V8-007.
- Decision: `workflow_runs` owns an immutable normalized JSON snapshot containing
  the mandatory default pin, sorted explicit label pins, binding revisions,
  exact RuntimeConfig refs/digests and sorted non-secret credential IDs. Exact
  RuntimeConfig bodies remain in the already immutable version table and are
  loaded by ref; physical version deletion remains deferred. Public Run JSON
  projects only labels, binding revisions and config refs, never credential IDs.
- Transaction boundary: `RunWriter.PinRuntimeLabels` locks default plus explicit
  bindings lexically and validates their immutable bodies under the same
  PostgreSQL transaction that inserts the Run and forks inputs. The existing
  shared credential lifecycle reader surrounds workflow resolution, pinning and
  commit, so both managed LLM and Runtime credential deletion serialize without
  acquiring a recursive read lock.
- Optional route representation: `ResolvedConsumerExecutionConfig.llmGateway`
  is now an actual optional pointer. Only a Worker may retain absence after Run
  initialization; Planner Gateway/ModelPolicy completeness is unchanged.
  Candidate resolution in V8-007 must complete every Worker route before
  reservation/prepare.
- Conservative deletion fence: a non-terminal Run fences every credential ID
  named by one of its pinned exact RuntimeConfigs, even when a higher layer may
  later clear that lower value. This prefers deterministic replay over early
  deletion; the fence disappears when the Run becomes terminal.
- Upgrade behavior: Runs created before migration 000018 had no label input.
  Migration therefore records an empty explicit set and pins the default
  binding current at migration time. It does not fabricate historical labels.
  Canonical request hashing omits only the empty label set, preserving the old
  digest for exact response-loss replay; every non-empty set is included in
  sorted form.
- Alternatives rejected: copying full RuntimeConfig documents into every Run
  duplicates immutable bodies and endpoints; retaining only mutable label names
  breaks replay; a normalized child table plus a duplicate read projection adds
  two authorities before immutable-version garbage collection exists.
- Observable tests: real PostgreSQL tests hold a pin transaction open while a
  rebind blocks, prove old/new Runs observe whole old/new refs, verify both
  credential deletion queries, and reject mutation of the committed snapshot.

### D015 — The pure resolver consumes authorization metadata, not secrets

- Applies to: V8-006 and V8-007.
- Decision: the pure resolver accepts exact pinned RuntimeConfig bodies,
  execution-route patches, immutable Gateway bodies, LLM credential effective
  policy metadata and Runtime credential kinds. It returns effective physical
  settings, typed field origins and safe Runtime provenance. Token/header/
  password material is neither an input nor an output; V8-007 resolves only the
  chosen IDs after durable placement.
- Planner boundary: default and Run-label Planner telemetry is resolved and
  kind-checked in the same deterministic call, but it contributes neither a
  candidate Runtime adapter requirement nor a Worker credential provenance
  ref. Agent-label Planner blocks are ignored after requiring that each Agent
  label contain at least one Worker-applicable operation.
- Authorization: a present LLM credential must match the final exact Gateway
  and its effective policy must contain both the already selected exact
  ModelPolicy ref and model alias. The resolver has no operation that can
  replace that policy, its budgets or the alias.
- Alternatives rejected: resolving secrets inside the pure merge would make
  candidate comparison side-effectful and broaden secret lifetime; requiring
  Planner OTLP support from a Python Runtime would conflate Server and Worker
  adapter capabilities; retaining only the credential/Gateway pair would miss
  a label-driven policy bypass.
- Observable tests: every permutation of equal/conflicting label layers has an
  identical result/error, explicit clears retain their winning origin, Agent
  Planner input cannot win, and route/kind/policy failures expose only stable
  codes plus typed paths.

### D016 — Candidate placement is provisional until provenance commits

- Applies to: V8-007.
- Decision: candidate-specific resolution reads detached live snapshots and
  immutable catalogs outside the Registry mutex, passes only compact compatible
  edges into complete matching, and creates a provisional all-or-nothing slot
  batch. The SQL transaction then locks referenced Agent-label bindings in
  lexical order and principals by sorted SPKI fingerprint, re-resolves the
  selected edges, locks the `preparing` StageExecution and persists every exact
  non-secret allocation snapshot. Only after commit does Registry expose the
  batch to Scheduler. A changed revision/catalog observation discards the
  never-prepared batch and is ordinary temporary capacity for the same
  StageExecution.
- Failure classification: typed candidate policy/reference/kind failures remove
  only that edge. Opaque database/provider errors remain visible Scheduler
  errors; they are not converted to insufficient capacity. Static process-local
  development LLM credentials are explicitly `unrestricted`, while managed
  credentials contribute their immutable effective ModelPolicy/model allowlist.
  Both still require the exact selected Gateway ref.
- Secret boundary: allocation rows retain only ModelPolicy ref, typed field
  origins and safe Runtime provenance. The Scheduler decrypts selected
  tokens/headers/passwords only after commit, supplies them in
  `PrepareAllocationRequest`, rejects a WorkerHandle containing any supplied
  value, and clears all reachable request-setting references immediately after
  the bounded prepare call. Credential deletion shares one reader/writer fence
  with allocation commit and remains blocked until durable release completion.
- Upgrade behavior: migration 000019 leaves pre-V8 allocation rows as an
  all-null legacy provenance tuple rather than fabricating a principal or
  historical label selection. Every new application write requires the full
  v2 tuple. Readers accept old rows for release/report recovery; new placement
  never emits them.
- Deadline correction discovered by review: the documented Stage/Planner
  deadline had previously begun only after capacity was found. V8-007 now
  derives one deadline from immutable `StageExecution.created_at` plus the
  configured `planner-timeout`, so capacity waits and revision churn consume
  the same finite budget as preparation and Planner execution. Expiry before a
  Planner exists records retryable preparing termination
  `stage_deadline_exceeded`. Revision churn before that deadline creates neither
  a Planner nor an attempt; once the shared deadline itself expires, normal
  Workflow retry policy decides whether a later StageExecution is created.
- Alternatives rejected: holding the Registry mutex across SQL/decryption
  would couple fleet liveness to external latency; persisting endpoints or
  secrets would broaden compromise and deletion surfaces; exposing provisional
  reservations would allow prepare before audit commit; backfilling guessed
  historical principals would produce false provenance.
- Observable tests: PostgreSQL placement tests cover an unlabeled adapter
  candidate, post-match principal revision mutation and exact durable pinning;
  Registry tests cover non-greedy injective matching; credential lifecycle
  tests cover live allocation fences; Scheduler tests cover the immutable
  capacity deadline; Go/Python protocol tests cover v2 delivery and secret-free
  returned handles.

### D017 — RuntimeConfig versions remain immutable API history

- Applies to: V8-008.
- Decision: the first Operations API publishes and reads immutable
  RuntimeConfig versions but does not delete them. Delete/CAS behavior applies
  to mutable label bindings and write-only Runtime credentials. Binding
  mutations use a PostgreSQL-backed idempotency record in the same transaction
  as the CAS mutation; a transaction-scoped advisory lock serializes the brief
  interval before that immutable record exists.
- Reason: V8-008's original A3 accidentally named config deletion even though
  its endpoint list, owning spec and explicit out-of-scope list defer immutable
  version garbage collection. Adding a destructive endpoint here would create
  retention semantics not reviewed by the spec. Durable binding idempotency is
  necessary because a same-target no-op is not sufficient evidence that a
  response-loss replay is the original mutation.
- Compatibility impact: an initial binding PUT now fails its
  `If-None-Match: *` precondition whenever the label already exists, including
  when it already points at the requested ref. Exact replay still succeeds by
  idempotency key without consuming another revision.

### D018 — Durable principal labels advance the live placement snapshot

- Applies to: V8-009.
- Decision: Operations reads principal identity, labels and audit metadata from
  PostgreSQL and joins them to a current process observation held only by the
  in-memory Registry. After a successful durable label CAS, Control Plane
  advances every matching live Registry entry to the committed revision. It
  sends no label command to Runtime and does not alter any active reservation's
  pinned RuntimeSettings or provenance. A replay of an older successful
  mutation re-reads and applies the newest durable principal revision, while
  returning the original mutation result.
- Reason: leaving the Registry on the registration-time revision would make
  every later placement fail its database recheck until the process restarted.
  Persisting heartbeat or capabilities to solve that mismatch would instead
  create a second liveness authority. The post-commit update has a safe race:
  placement observing the old revision is rejected by the existing SQL
  principal lock/recheck, and placement after the update sees the new complete
  set.
- Availability projection: `offline` means no live or allocation-bound process
  observation exists. For a live process, intrinsic adapter requirements
  contributed by its Agent labels are compared with its frozen registration
  capabilities; `adapter_capability_mismatch` takes precedence over busy slot
  status so an operator can see why later label-resolved work cannot use it.
  This is not a claim that the Agent can run every Workflow: ordinary runtime,
  Toolset, SandboxProfile and Run-specific adapter matching remains placement
  work.
- Deletion boundary: an empty offline principal is still fenced by every
  durable allocation whose release is incomplete. Terminal allocation rows are
  immutable historical provenance values, not live foreign-key ownership, and
  do not retain the principal configuration row forever. Principal-row locking
  serializes deletion with candidate placement before either inserts or removes
  that live durable reference.

### D019 — Adapter close authority is stronger than telemetry delivery

- Applies to: V8-010.
- Decision: `AllocationAdapterHost` is constructed before sandbox preparation
  and owns four explicit allocation channels (`model_http`, `tool_http`,
  `tool_subprocess`, `instrumentation`). Typed settings determine exactly which
  channels a factory must return. No adapter or setting is installed in a
  process global. Worker and Toolset factory contexts receive the merged
  handles; model-visible objects receive neither adapters nor settings.
- Teardown boundary: telemetry flush gets at most half of the remaining outer
  lifecycle deadline, reserving the rest for credential-erasing close. A flush
  timeout/failure is recorded only in saturating allowlisted metrics. A close
  timeout/failure instead fences the slot and requests process exit, because an
  in-process coroutine that ignored cancellation could otherwise keep a client
  or credential reachable after an incorrect return to `idle`.
- Wire compatibility: Python's active `RuntimeReport` and Go's strict response
  DTO both carry the typed adapter map. Go serializes a nil internal map as an
  empty object so old synthesized reports remain valid. Missing or malformed
  adapter maps in already durable pre-V8 history are retained as an incomplete
  runtime report rather than rejecting the semantic final report. Control Plane
  then keeps only valid metrics whose refs occur in the reservation's pinned
  provenance; unexpected, malformed or missing metrics set `complete=false`
  and cannot block terminalization/release.
  The separate canonical `RuntimeReport` fixture is part of the shared
  v1alpha1 parity catalog. This small Go change is necessary even though V8-010 was
  originally marked Python-only: otherwise `DisallowUnknownFields` would
  reject the Runtime's first adapter report.
- Secret handling: allocation replay fingerprints use a per-process keyed HMAC
  rather than a reusable raw SHA-256 of secret-bearing JSON. Agent Card checks
  compare every private settings leaf (and scan distinctive values when
  embedded), while repr/snapshot expose only adapter refs/channel names and
  safe codes. Preparation failures distinguish an explicitly classified
  transient factory error; an unclassified implementation error is
  non-retryable.
- Alternatives rejected: letting flush consume the complete deadline could
  make an instant close appear unconfirmed; treating close like best-effort
  telemetry would violate secret erasure; constructing adapters after sandbox
  would leave avoidable local resources on configuration failure; and adding a
  second final-report envelope would fork the already active lifecycle route.

### D020 — Worker telemetry is a bounded allocation-local OTLP projection

- Applies to: V8-011.
- Decision: `otlp-http@1` builds official OTLP/HTTP trace protobuf messages
  directly instead of installing a second tracing runtime or LangChain. Each
  allocation owns one exporter, one HTTP client and an encoded FIFO bounded by
  2048 spans and 2 MiB. The queue has no background worker or durable spool; it
  is flushed once through the existing bounded terminal adapter lifecycle.
- Content boundary: instrumentation accepts only four fixed span names and a
  closed attribute allowlist. It records correlation identifiers, exact safe
  refs/labels, model aliases, tool names, outcomes, durations and aggregate
  counters. Prompts, responses, tool arguments/results, artifact bytes,
  provider bodies and URLs have no representable field. Header values and the
  selected endpoint are additionally treated as secrets and cause matching
  attribute values to be omitted.
- Provenance encoding: exact RuntimeConfig refs and digests are exported as
  parallel bounded arrays. A combined `name@version#digest` can exceed the
  normative 256-byte string bound, while the parallel representation retains
  exact identity without truncating a digest. Run labels, Agent labels and
  adapter refs are independently bounded arrays from the already pinned
  allocation context.
- Transport boundary: `httpx` is an explicit Runtime dependency. The client
  disables environment proxy/trust inheritance and redirects, uses one
  connection, sends secret headers only to the exact configured endpoint, and
  never receives Runtime mTLS material or Artifact grants. A successful HTTP
  response is accepted only when its bounded protobuf body reports no partial
  rejection; response text is never surfaced.
- Failure semantics: queue overflow, encoding failure, backend rejection,
  disconnect and final flush timeout affect only allowlisted adapter metrics.
  They cannot change the semantic Worker result or prevent release. Close
  retains the stronger V8-010 erasure/fencing rule. ADK hook failures are also
  swallowed at the instrumentation boundary. While wiring this path, optional
  LLM Gateway credentials were corrected so a configuration without a token
  does not attempt to dereference `None`.
- Alternatives rejected: the full OpenTelemetry SDK would add global provider
  and background-exporter lifecycle questions; JSON OTLP would weaken wire
  interoperability; redirects or ambient proxies could disclose configured
  headers to an unintended authority; recording arbitrary callback attributes
  would make the no-content policy depend on every model/tool implementation.

### D021 — Proxying is an explicit typed channel, never ambient process state

- Applies to: V8-012.
- Decision: `http-proxy@1` owns separate allocation-local HTTP clients for the
  selected model and tool channels plus an optional bounded subprocess
  launcher. It configures `httpx.AsyncHTTPTransport` directly with zero
  connection retries, disabled ambient trust/proxy state and no redirects.
  `AdapterHandles.for_worker()` exposes only model HTTP and instrumentation;
  each Toolset receives only the union of channels declared by its actually
  selected tools. Configuration therefore cannot make a local/artifact-only
  tool network-capable.
- Descriptor parity: the Go configuration descriptors and Python factories
  both declare `runtime-http-client` and/or `runtime-subprocess-launcher` per
  exported tool. Invalid or unknown declarations reject factory/catalog
  construction. Both implementations are checked against the same committed
  non-wire YAML parity fixture. The existing LikeC4 and OpenAPI validators are
  the first subprocess consumers; all other current tools declare no
  infrastructure channel.
- Private boundary: CLI startup supplies Control Plane and advertised Runtime
  hosts to the allocation service, while each allocation adds its Artifact API
  host and loopback names. Tool HTTP handles reject those destinations, and
  child-only `NO_PROXY` contains them. Registration, heartbeat, Artifact mTLS
  and A2A traffic keep their pre-existing clients and trust contexts and never
  receive a proxy handle. No module global, `os.environ` value or system trust
  store is mutated.
- Trust and subprocess boundary: optional CA certificates augment a fresh
  default SSL context for targeted clients. A subprocess gets a per-invocation
  mode-0600 combined public CA file and a closed allowlist of environment
  variables; the file is removed after the bounded process exits and all
  retained material is cleared again on adapter close. Commands require an
  absolute executable, bounded arguments/input/output and the smaller of the
  tool and allocation timeouts.
- Authentication decision: HTTP clients use the proxy library's dedicated
  Basic auth or `Proxy-Authorization: Bearer` channel, so credentials are sent
  only while connecting to the exact proxy authority. Standard
  `HTTP_PROXY`/`HTTPS_PROXY` environment semantics can encode Basic credentials
  but cannot faithfully express Bearer proxy authentication. A
  bearer-authenticated `tool-subprocess` therefore fails closed in v1 instead
  of silently going direct or downgrading the credential; model/tool HTTP
  bearer proxying remains supported. A future subprocess relay would require a
  separate reviewed adapter version.
- Failure semantics: proxy refusal, disconnect, TLS failure, private-host
  misuse and subprocess bounds become content-free `request_failed` adapter
  metrics plus bounded model/tool errors. There is no direct retry path.
  Transport/launcher close remains under V8-010's confirmed-erasure fence.

### D022 — Adapter availability may be narrowed only at immutable startup

- Applies to: V8-013.
- Decision: Runtime enables all built-in adapter factories when no selector is
  supplied. A repeated `--runtime-adapter` option, or its comma-separated
  environment equivalent, may select a strict non-empty subset before probes
  and registration. Unknown and duplicate refs fail startup. The resulting
  registry is frozen with the rest of the process capability snapshot; labels
  and allocations cannot mutate it.
- Reason: deployments can intentionally package or expose different local
  infrastructure integrations even when they share one Python environment.
  Treating this as immutable factory enablement makes heterogeneous placement
  testable without a test-only probe hook and without claiming that a label
  creates capability. A configured factory must still pass its real probe.
- Alternatives rejected: dynamically enabling adapters after registration
  would invalidate Scheduler's positive snapshot; inferring availability from
  labels would conflate desired infrastructure with installed code; requiring
  a separate virtualenv for each subset would hide an operator-level boundary
  behind packaging mechanics and make a one-VM deployment unnecessarily hard.

### D023 — Planner OTLP is invocation-local and wholly supplementary

- Applies to: V8-014.
- Decision: Server owns one immutable PlannerTelemetryAdapter registry shared
  by RuntimeConfig publication and Scheduler. `otlp-http@1` encodes a bounded
  closed span vocabulary directly as OTLP/HTTP protobuf for one Planner
  invocation; it does not install a global OpenTelemetry provider, inherit an
  HTTP proxy, follow redirects, retry, or maintain a background/durable spool.
  Passthrough, Streamline and Router receive only a narrow instrumentation
  handle. Resource correlation is built from the pinned default/Run snapshot
  plus Planner model/Gateway/credential refs; candidate Agent labels and
  Worker proxy settings are structurally absent.
- Lifecycle: Scheduler makes one flush after Planner returns, bounded by the
  minimum of config timeout, remaining Stage deadline and finalization timeout,
  then destroys secret-bearing adapter state. Creation, JIT credential,
  encoding, queue, delivery and flush failures never become Planner failures.
  If a durable Planner report identity exists, the safe result is appended as
  a `telemetry.export` tool call/metric rather than a semantic report error.
- Alternatives rejected: a process-global SDK provider would create shared
  mutable routing and shutdown ownership; background batching could outlive the
  pinned invocation or delay allocation release; treating a telemetry secret
  outage as Stage preparation failure would let supplementary observability
  choose retry/escalation semantics. The Worker exporter remains allocation-
  local and independent.
- Observable tests decode real protobuf for all three Planner types, prove the
  closed no-content projection and exact Run-only provenance, cap the queue,
  bound a hanging collector, and show rejected delivery leaves Run success and
  slot release unchanged while producing only the safe durable export outcome.

### D024 — Run detail projects committed allocation provenance, not live Operations

- Applies to: V8-015.
- Decision: `StageAttempt.runtimeConfiguration` is absent until at least one
  allocation has committed durable Runtime provenance. It then projects one
  entry per logical Worker with Agent-label pins, exact adapter refs, closed
  field origins and release status. The projection is read from immutable
  `stage_allocations`; it never joins the current principal label set or live
  Registry snapshot.
- Safety boundary: the public shape has no representation for allocation IDs,
  certificate principals, physical Runtime instance IDs, endpoint URLs,
  RuntimeSettings, headers, tokens or proxy authentication. Frontend parsing
  also rejects unknown fields before storing this nested projection in query
  state.
- Reason: the V8 API already persisted the required provenance but the Run
  response exposed only creation-time default/Run pins. Without this small
  public projection, V8-015 could only guess a final Agent override from
  mutable Operations state, which is both historically wrong and contrary to
  the spec. Returning the full private provenance was rejected because it
  would couple owner-facing Run views to physical deployment and credential
  metadata.

### D025 — Runtime-configuration hardening is one executable composed matrix

- Applies to: V8-016.
- Decision: keep one production-boundary process scenario for the complete
  Server/PostgreSQL/Scheduler/Planner/mTLS/Python Runtime/adapter path, then
  compose deterministic PostgreSQL, race-detector, real-certificate and Python
  lifecycle tests for fault points that cannot be injected from an external
  client. A strict YAML matrix names every mutation/failure contract and its
  owning test; the release target executes the matrix, focused hardening,
  process and browser gates together.
- Process additions: every new public mutation is immediately replayed as if
  its first response had been lost; a second instance using the same live
  certificate is rejected; another CA-valid Agent cannot borrow a copied
  allocation/instance identity; Planner and Worker export to distinct pinned
  collectors; required proxy rejection fails the Run; two disjoint adapter
  specialists both execute a final successful reuse probe.
- Alternatives rejected: one monolithic process with internal SQL/network
  hooks would either replace production resolution/release code, introduce
  test-only mutation endpoints or make response partitions and nanosecond CAS
  races timing-dependent. Merely listing existing test commands in prose could
  silently lose coverage after a rename. Running every browser permutation
  would add time without reaching the private race linearization points.
- Compatibility impact: `make verify` remains database-independent and checks
  the executable matrix; `make test-runtime-configuration-e2e` requires
  PostgreSQL and runs the full deterministic increment; `make release-verify`
  is the CI aggregate. External LM Studio, Langfuse, Caido, cloud services and
  internet access remain outside the gate.

### D026 — Memory ordinal is a uint64 inside the exact JSON integer range

- Applies to: V9-001 and later MemoryTools tasks.
- Decision: keep `ordinal` as an unsigned 64-bit domain value, but reject a
  stored or encoded value above `2^53-1`. The ordinary `v1` allocator can only
  produce `0..127`, because a Namespace has at most 128 notes and no delete.
- Reason: RFC 8785 canonicalizes JSON numbers with ECMAScript/IEEE-754
  semantics. Go's JCS implementation does not accept `uint64` directly and
  Python's implementation rounds larger integers; accepting the full uint64
  wire range would silently change an ordinal during canonicalization.
- Alternatives rejected: a JSON string would contradict the agreed note
  schema, while allowing rounded values would break cross-language identity.
  Reducing the domain type itself to a small integer would unnecessarily
  constrain a future version that adds deletion or a larger quota.
- Observable tests: shared Go/Python fixtures accept `2^53-1`, reject `2^53`,
  and prove byte-identical canonical output for the values used by `v1`.

### D027 — Worker MemoryTools reconciles only transport-uncertain mutations

- Applies to: V9-002 and later MemoryTools hardening.
- Decision: all six selected Worker operations share one allocation-local
  serialized Memory session and the allocation-bound Artifact client. A direct
  Artifact CAS conflict immediately becomes retryable `memory_changed`. Only a
  transport failure with an unknown commit outcome authorizes one replay of the
  byte-identical canonical payload with the identical create/update
  precondition. If that replay also fails, an exact current payload means the
  original mutation completed; a different current payload is
  `memory_changed`.
- Error boundary: allocation/grant/write-fence rejection becomes the bounded
  non-retryable `memory_forbidden`; transport, malformed stored state and other
  Artifact failures become retryable `memory_unavailable`. Low-level
  `artifact_*` codes and messages never enter the model-facing Memory contract.
- Provenance boundary: Memory tool instances intentionally expose no
  `known_exact_refs` projection. `run-artifacts@1` filters the reserved
  `memory.` prefix from list results and its accumulated exact refs, and rejects
  generic read/write before making an Artifact call.
- Reason: retrying an ordinary conflict could overwrite a real concurrent
  semantic change, while recomputing append after response loss could duplicate
  its fragment. A single exact replay plus equality reconciliation preserves
  at-most-once logical mutation without a Memory-specific replay table or
  exposing revisions to the model.

### D028 — Planner Memory uses a closed ADK adapter and global lifecycle lock order

- Applies to: V9-003 and later MemoryTools tasks.
- Decision: Streamline and Router install custom structural ADK tools for the
  selected Memory operations. They advertise the exact JSON Schema but reduce
  malformed raw arguments themselves to the closed `memory_invalid` or
  `memory_forbidden` vocabulary before storage access. Successful values keep
  the same Worker logical projections; durable Planner facts retain only safe
  names, counts, byte sizes and outcomes. Logical Worker bindings that resolve
  to one Agent Namespace reuse one serialized Namespace object while retaining
  their distinct operation allowlists. Passthrough receives no Planner tools.
- Lifecycle decision: every Scheduler transition that can race an active
  Planner Memory call takes PostgreSQL locks in WorkflowRun-then-StageExecution
  order. In particular, abort entry is an atomic Scheduler persistence
  operation rather than a direct Stage-first RunStore update, because the
  Stage event trigger also updates the WorkflowRun sequence.
- Reason: ADK's generic function wrapper can surface framework validation text
  before the handler runs, and a Stage-first abort update deadlocks with the
  Planner's required Run-first authority check. Both details are adapter and
  persistence mechanics, not additions to the public Memory contract.

### D029 — Memory hardening uses namespace-aware visibility and pre-binding validation

- Applies to: V9-005 and every later artifact-backed Runtime Toolset.
- Decision: one dependency-neutral Artifact policy defines a Memory binding as
  `memory.*` only outside `inputs`, `outputs` and `skills`. Generic, text,
  source-analysis, OpenAPI and LikeC4 tools apply it to source and target names
  before I/O and to every accumulated exact-ref projection. Runtime Worker,
  Planner and Scheduler independently reject such a ref as a Stage result, and
  Workflow loading rejects it as StageContext. The three purpose Namespaces
  retain their separate rules.
- ADK boundary: Memory callables expose a trusted raw-argument validator that
  runs inside the ordinary budgeted Worker tool wrapper before ADK filters
  unknown fields or emits its own missing-parameter text. Invalid shape records
  one content-free failed call and returns the closed `memory_invalid` result;
  advertised function declarations remain unchanged.
- Integrity boundary: global views require the exact ordinal set
  `0..count-1`, append validates only its real fragment and final canonical
  note, exact reads require the requested ETag, timestamps project as UTC-Z,
  and internal errors are reduced to the fixed code/retryability table.
  Targeted reads and existing-note mutations remain available when unrelated
  ordinal state is corrupt so repair and diagnosis do not require 128 reads on
  every call.
- Telemetry boundary: content, description, tags and revisions are still
  absent, while validated names, counts and non-negative derived byte sizes are
  retained. The generic sanitizer accepts those size fields only when their
  values are integers, so renaming a sensitive payload key cannot bypass
  redaction.
- Alternatives rejected: filtering by artifact name alone incorrectly hid
  ordinary `inputs/memory.*`; relying on ADK schema binding leaked framework
  errors and silently discarded extra fields; relying only on exact-ref
  provenance left alternate Toolsets and recovery prompts as bypasses.

### D030 — The V9 release audit closes purpose, error and reconciliation boundaries

- Applies to: V9-001 through V9-007.
- Purpose Namespace correction: `inputs`, `outputs` and `skills` are one closed
  purpose-reserved set and can never be a Stage Agent Namespace. Workflow
  loading, persisted Workflow decoding, Runtime allocation preparation and both
  Memory factories reject all three. This prevents `skills/memory.*` from being
  created through MemoryTools while generic visibility correctly treats that
  same name as an ordinary Skill artifact.
- Error correction: Worker tool entry points normalize every unexpected Python
  exception before both model dispatch and metrics recording. A custom
  exception's arbitrary `code`, `retryable` value or message therefore cannot
  expand the seven-code Memory error vocabulary or become retained detail.
- Reconciliation correction: after an uncertain exact replay, the final read is
  authoritative for access. `memory_changed` requires an observed current value
  whose canonical bytes differ. A fenced/forbidden read becomes
  `memory_forbidden`; an unreadable value becomes `memory_unavailable`.
- Telemetry wording: the enforceable non-retention promise covers automatic
  Memory adapter diagnostics. The LLM Gateway is an intentional content channel,
  and a model may explicitly copy note data into another semantic destination.
  Preventing such a copy would require information-flow tracking, which is not
  claimed by `v1`.
- Reason: the earlier implementation was internally consistent on its main
  path, but these three boundary mismatches could respectively bypass model
  visibility, leak an unbounded error vocabulary, and misclassify lost
  authority as a semantic CAS conflict. The former absolute telemetry sentence
  also promised a property no prompt-only implementation can prove.

### D031 — Workspace hardening preserves behavior, not legacy escape hatches

- Applies to: V11-012 and later workspace implementations.
- Decision: retain the useful contractor-old read, Edit, newline and overlay
  behavior behind the narrow allocation-scoped handles. Do not restore raw
  fsspec delegation, materialize, fork/merge, implicit host roots or shared
  global memory state. `maxFiles` counts the complete managed tree, including
  implicit directories.
- Local defense in depth: direct text writes use descriptor-relative,
  no-follow traversal and atomic replacement. A swapped final symlink is
  replaced and a swapped parent fails closed without changing outside bytes;
  failed replacements remove their private temporary files. Runtime `workRoot`
  remains private to the Runtime OS identity because same-UID filesystem
  mutation is inside the host trust boundary.
- Verification: the retained backend matrix is accompanied by adversarial ZIP,
  Unicode/path, replacement, concurrency, cancellation and stale-export tests.
  Tool telemetry is asserted not to retain fixture paths, patterns or contents.

### D032 — HTTP/Caido hardening separates application checks from network policy

- Applies to: V12-008 and later HTTP/Caido Toolset revisions.
- Egress decision: generic HTTP rejects invalid URLs, Contractor's exact
  configured infrastructure origins, localhost names and dangerous IP
  literals, then repeats those checks on every redirect and strips credentials
  cross-origin. It does not perform DNS pinning or claim RFC1918/ULA isolation.
  Deployments that require a closed destination set use the already mandatory
  `tool-http` forward-proxy route and/or a restricted Runtime network
  namespace; proxy failure never falls back to direct egress.
- Compatibility decision: Caido compatibility is the reviewed set of static
  operation documents, variables and strict response shapes, not an inferred
  product version. Registration probes only the local adapter implementation.
  There is no remote introspection or generated-query fallback; schema drift
  returns a bounded error until a new reviewed adapter/tool revision and its
  fixtures are published.
- State decision: HTTP session updates preflight the complete prospective
  header/cookie/auth state before mutation. The allocation session owns the
  authoritative cookie jar and clears transport scratch cookies before reuse.
  Request IDs and Caido action tags are consumed monotonically even when a
  mutation is cancelled or its response is ambiguous. A body committed before
  cancellation cannot become selected under a later ID.
- Verification: one strict YAML matrix maps all three V12-008 acceptance
  requirements and thirteen injected fault classes to named Go/Python owners.
  Focused tests cover exact byte/count limits, redirects, mandatory proxying,
  response bombs, GraphQL shapes, concurrency, cancellation and retained
  canaries; the release gate adds PostgreSQL, mTLS and two real Python Runtime
  processes with release-response loss and clean slot reuse.

### D033 — Taint annotations are structured workspace edits, not graph state

- Applies to: V16-001 through V16-005 and later trace-oriented templates.
- Identity decision: the portable `taint-annotations@1` resolver uses a
  normalized workspace path, exact unqualified function-like symbol and an
  optional current line selector. It does not consume Trailmark `symbolId`:
  those IDs are allocation/digest-local and unavailable on memory Runtime
  Agents, while source annotations must work with the common Tree-sitter
  capability. Same-name definitions fail as ambiguous instead of first-match.
- Mutation decision: parsing occurs off the asyncio loop, followed by an exact
  source comparison inside the existing atomic `WorkspaceWriter.update_text`.
  This closes lost updates against independently selected Edit Toolsets without
  expanding the workspace interface or adding a Server endpoint. Exact replay
  is a no-op; a different body for the same trace target conflicts rather than
  silently rewriting evidence.
- Syntax decision: annotation placement treats decorators, attributes,
  exports and templates as a declaration prefix, permits one trace line per
  target in insertion order, and uses the actual v1 line comment marker for
  every supported language. The old decorator split, duplicate rejection,
  PHP-marker mismatch and arbitrary exception text are not compatibility
  requirements.
- Persistence decision: direct-mode edits remain disposable. Durable output is
  the already specified cumulative overlay state and text diff auto-export;
  annotations create no graph mutation, Artifact type, table or API.
