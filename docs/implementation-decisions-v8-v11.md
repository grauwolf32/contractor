# V8–V11 implementation decision log

This is a non-normative engineering log for the implementation sequence that
starts after the 2026-09-01 specification review. The working agreements in
[`docs/spec`](spec/README.md) remain authoritative. This file records choices
that are easy to lose in commit history, especially where two implementations
could both appear to satisfy a short task title.

Each completed implementation task still records the full hash of its final
task commit in `tasks/<id>.yml`. If implementation discovers a contract change,
the specification and task are updated before code relies on it, and the
reason is appended here.

## Decisions accepted during the pre-implementation review

### D001 — Project filesystems are Runtime-only

- Applies to: V11.
- Decision: host roots, backend mode, imported baseline, overlay upper,
  materialization journal and `hostWrite` authority exist only in the Python
  Runtime process.
- Server impact: only the existing static Toolset/tool-name descriptor registry
  learns `filesystem@1`, `edit-files@1` and `workspace-changes@1`, so
  AgentTemplate validation can remain exact. No path, mount, mode, digest,
  change set, API, SQL state, RuntimeSettings field or AllocationSpec field is
  added.
- Reason: physical project access is an environment capability of a Runtime,
  not Workflow data or Control Plane state.

### D002 — New Python package is `contractor_runtime.projectfs`

- Applies to: V11-002 through V11-012.
- Decision: keep `contractor_runtime.workspace` unchanged for the existing
  `local-workdir@1` allocation scratch implementation. Put project filesystem
  protocols/backends in `contractor_runtime.projectfs`.
- Reason: a module and package with the same import name would collide, and the
  two concepts have different authority/lifecycle. `WorkspaceSession` remains
  the domain term; `projectfs` is the implementation namespace.

### D003 — ADK script support is absent, not merely inert

- Applies to: V10-003 and V10-005.
- Decision: retain Google ADK's native Skill models and list/load/resource tool
  implementations, but expose exactly `list_skills`, `load_skill` and
  `load_skill_resource`. Filter `run_skill_script`, disable registry/search and
  replace ADK 2.8.0's stock script-bearing system instruction with bounded
  script-free progressive-disclosure guidance.
- Reason: accepted packages contain no scripts and the model should not be
  instructed to call a function that can never succeed. Merely omitting an
  executor is insufficient because the pinned ADK instruction still advertises
  scripts.
- Compatibility guard: the `adk@1` startup probe verifies both declarations and
  generated instructions so an upstream ADK change fails closed.

### D004 — Project import shares the existing startup deadline

- Applies to: V11-003 and V11-008.
- Decision: all configured memory-backed roots share the existing 30-second
  complete Runtime capability-discovery deadline. Per-mount file/count/byte/
  depth limits remain independent. Filesystem Toolset probes reuse the prepared
  provider and do not rescan source trees under their five-second individual
  budgets.
- Reason: “30 seconds per mount” conflicted with the already normative global
  Runtime startup bound and allowed eight mounts to delay registration for
  minutes.

### D005 — Startup host-write probing never mutates the project

- Applies to: V11-008.
- Decision: startup validates required descriptor-relative/no-follow platform
  primitives and opens the root with requested authority, but does not create a
  hidden probe file inside an operator project. Generic probe mutations remain
  below Runtime work roots. Real host permission failures remain bounded tool
  or preparation failures because permissions may change after any probe.
- Reason: capability discovery must not alter user data, and a successful
  create/delete test could not prove future availability anyway.

### D006 — Planner Memory mutations use the Stage write fence

- Applies to: V9-003 and V9-005.
- Decision: the Server-side Planner MemoryTools view is bound to the active
  StageExecution. Its CAS write and the durable finalizing/aborting fence are
  checked in one Artifact transaction, exactly like Worker authority at the
  private API boundary.
- Reason: a Run-scoped direct ArtifactStore view without a Stage fence could
  commit a late note after cancellation even though Worker writes were already
  revoked.

### D007 — No hidden project affinity in V11

- Applies to: V11-008 through V11-012.
- Decision: placement continues to use only ordinary exact Toolset/tool
  capability subsets. Until a separate source-artifact/affinity design exists,
  one filesystem-enabled AgentTemplate may target either one eligible Runtime
  or a set of Runtimes with equivalent logical mounts, content, modes and write
  authority.
- Reason: sending a project identifier while claiming a Runtime-only contract
  would create an accidental fifth placement dimension and incomplete Server
  ownership.

### D008 — Direct local writes and overlay materialization stay distinct

- Applies to: V11-002, V11-006 and V11-010.
- Decision: `local + hostWrite=true` performs explicit immediate edits and
  promises atomic whole-file replacement plus bounded in-process operation
  rollback, but no allocation-level diff/rollback and no crash-safe multi-file
  transaction. `overlay-local + hostWrite=true` is the path for reviewable
  changes and crash-recoverable explicit materialization.
- Reason: claiming portable atomic host-tree mutation for direct local mode
  would be false. Operators choose overlay-local when those properties matter.

### D009 — Reject hard-linked project files in the strict profile

- Applies to: V11-002 and V11-012.
- Decision: rooted local access rejects multiply linked regular files as well
  as symlinks and special files.
- Reason: an in-root hard link can alias an inode that is also named outside the
  configured tree, violating the intended path authority for both reads and
  writes.

### D010 — Implementation and commit order

- Applies to: all unfinished V8–V11 tasks.
- Decision: implement in dependency-valid ID order, V8 before V9 before V10
  before V11. One task gets one primary completion commit unless an inseparable
  schema/wire change must remain buildable. After the commit, update that task's
  `status` and full `implementation_commit` in a small metadata commit; the
  recorded hash remains the implementation commit, not the metadata commit.
- Reason: this preserves bisectability and lets a future worker verify task
  evidence without reconstructing the entire session.

### D011 — Materialization has explicit pre-write and recovery bounds

- Applies to: V11-010 and V11-011.
- Decision: one host materialization is capped at 10,000 affected paths,
  16 MiB per backed-up file, 64 MiB staged replacements, 128 MiB backup,
  8 MiB journal metadata and 60 seconds. Startup recovery scans at most eight
  entries/256 MiB under the shared 30-second startup deadline and persists
  rollback progress before yielding.
- Reason: “bounded journal” was not an executable contract. Large deleted host
  files can make backup much larger than the 64 MiB overlay upper, so both
  per-file and aggregate backup limits are necessary before the first write.

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
  does not invent an AllocationSpec provenance document: AllocationSpecV2 is
  activated by V8-007 only after the candidate-specific transaction can commit
  every exact binding/config/credential ref before delivery. RuntimeReportV2
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
- Observable tests: `make test-runtime-wire-v2` covers the v1 registration
  rejection and v2 emission/response parsing; V8-007's placement tests must
  cover the AllocationSpecV2 cut-over before its completion commit.

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
  settings, typed field origins and protocol-v2 safe provenance. Token/header/
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
  origins and protocol-v2 safe provenance. The Scheduler decrypts selected
  tokens/headers/passwords only after commit, supplies them in
  `PrepareAllocationRequestV2`, rejects a WorkerHandle containing any supplied
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
