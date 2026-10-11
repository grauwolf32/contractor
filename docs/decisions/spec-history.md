# Specification history

Delivery plans, increments, acceptance narratives and implementation status
that the feature-ordered specifications carried until the 2026-10 migration
to the domain layout. Kept verbatim for reference; current status lives in
the task files and the backlog, current behaviour in [the specification](../spec/README.md).

## From `00-workflow-and-planner.md`

### 00 — Workflow Scheduler and Planner

Status: **Working agreement**

## From `01-agent-template.md`

### 01 — AgentTemplate

Status: **Working agreement**

## From `02-runtime-and-a2a.md`

### 02 — Runtime and A2A

Status: **Working agreement**

## From `03-artifact-plane.md`

### 03 — Artifact plane

Status: **Working agreement**

## From `04-execution-lifecycle-and-metrics.md`

### 04 — Stage execution lifecycle, sessions and metrics

Status: **Working agreement**

## From `05-first-slice-and-open-decisions.md`

### 05 — First slice and open decisions

Status: **Working agreement**

### 05 — First slice and open decisions › First implementation slice

The first slice proves boundaries, not the final distributed runtime:

```text
one user-scoped uploaded input
  -> one Workflow Run with that exact version forked into RunScope inputs
  -> Workflow Scheduler selects one Stage
  -> one exact AgentTemplate
  -> one local single-slot Runtime Agent
  -> one in-process Worker runtime instance
  -> one PassthroughPlanner invocation
  -> one A2A invocation producing an immediate Message or terminal Task
  -> one committed working artifact
  -> one persisted candidate StageResult and finalizing StageExecution
  -> one graceful Worker finalization report
  -> one terminal StageExecution and one Scheduler-bound Run output
  -> one atomically succeeded WorkflowRun with frozen required outputs
  -> allocation cleanup
```

This slice must demonstrate:

- Server recursively loads the fixed `workflows`, `agent-templates`,
  `model-policies`, `llm-gateways` and `execution-configs` YAML subtrees from
  the operator and managed configuration roots, plus relative resources under
  `instructions`; document lookup uses
  `kind + metadata.name + metadata.version`, not the file name, and duplicate
  identities or any invalid dependency reject the complete configuration set;
- Operations publishes new ModelPolicy and LLMGatewayConfig identities as
  canonical YAML in the managed root through full-set validation, durable
  atomic rename and an atomic in-memory snapshot swap; neither root overrides
  a duplicate identity from the other;
- Web UI is built and deployed as a separate Node.js service with its own
  version and Server-API compatibility check; Go Server embeds no frontend
  assets and remains healthy when UI is unavailable; Node serves only the
  client bundle and non-secret runtime API URL, while the browser calls Go
  Server directly through its exact CORS origin allowlist;
- Web UI is a strict-TypeScript React SPA built by Vite, routed with React
  Router Data Mode and backed by TanStack Query as its only shared Server-state
  cache; it has no SSR/BFF or duplicate global lifecycle store and never
  optimistically invents authoritative execution or publication transitions;
- one committed OpenAPI 3.1 public `/v1` contract generates the pinned
  TypeScript client through a thin cookie/CSRF/idempotency/CAS transport
  adapter; strict type checking, regeneration-drift tests, Vitest/Testing
  Library coverage and Playwright against the real Node UI plus Go Server are
  required first-slice gates;
- one direct, cookie-authenticated, exact-Origin WebSocket multiplexes Run and
  Operations observation while every command remains HTTP; reconnect replays
  the owning Run's committed ordered lifecycle and redacted Planner facts from
  a cursor, whereas a lost process-local Operations revision requires a REST
  resync, and backpressure can never block Scheduler or Planner persistence;
- one Argon2id bootstrap user logs in directly to Go Server and receives an
  in-memory, idle/absolute-expiring HttpOnly session plus a session-bound CSRF
  token; browser mutation requires exact Origin and CSRF checks, Server restart
  invalidates sessions, and the principal owns both user and Operations APIs;
- an LLMGatewayConfig may declare `litellm-virtual-keys@1` plus its non-secret
  management origin; Server Operations can manage it only when operator
  bootstrap binds that exact digest-bearing Gateway ref to a protected LiteLLM
  admin-key file;
- idempotent credential creation obtains a LiteLLM virtual key through
  `/key/generate`, encrypts it in Contractor PostgreSQL and exposes neither it
  nor the LiteLLM master key; deletion removes the exact remote `token_id`
  through `/key/delete` and then the local row, while the deterministic alias
  lets restart recovery clean an interrupted creation without exposing a
  partial credential state;
- virtual-key creation derives a non-empty LiteLLM model allowlist from exact
  ModelPolicy refs and forwards optional spend/reset, TPM, RPM and concurrency
  fields; it stores the effective policy returned by LiteLLM for display and
  provenance but leaves all Gateway-quota enforcement to LiteLLM, while
  ModelPolicy limits only bound one Planner or Worker execution;
- Run creation selects an exact Workflow `<id>@<version>`, resolves Workflow
  execution defaults plus reference-only Run overrides, and stores the complete
  base and declared escalation-variant per-consumer snapshots; later
  configuration-file edits cannot reinterpret that Run;
- escalation executionConfig is either one inline Stage-local
  `planner`/`agents` override or an exact digest-bearing `ExecutionConfig` ref;
  both normalize to the same override, mixed ref/inline form is invalid, and
  every referenced body is resolved and pinned before Run execution;
- public Run creation requires an owner-scoped `Idempotency-Key`; response-loss
  retry returns the same Run without repeating input forks or Scheduler wake,
  while reuse with different validated content conflicts;
- template resolution happens before allocation and the exact ref is retained;
- `planner`, `template` and ExecutionConfig `ref` accept only the exact
  `<id>@<version>` grammar; malformed, unversioned, `latest`, range and unknown
  selectors are rejected, and the Run snapshot retains resolved refs instead
  of re-resolving strings;
- every declared Run parameter is a string slot with explicit `required`;
  missing required, unknown and non-string values are rejected without
  coercion, while the validated mapping is persisted immutably and supplied
  read-only to every Planner through StageContext but not copied automatically
  to Workers; there are no defaults, optional absence is preserved and an empty
  string counts as a supplied value;
- every Stage has a non-empty literal `objective` and mandatory
  configuration-root-relative Planner `instructions.ref`; resolution stores
  exact text/digest in the Run snapshot, and PassthroughPlanner combines
  objective and instructions into one deterministic Worker task while keeping
  parameters and artifact refs structured;
- every Stage accepts only `session: isolated|shared`; new omission resolves to
  explicit `isolated` before persistence, persisted snapshots without an
  explicit value are unsupported under
  [00](../spec/configuration/workflows.md), and retry or
  escalation retains the pinned mode;
- every AgentTemplate has a mandatory configuration-root-relative Worker
  `instructions.ref`; its resolved text contributes to the template digest and
  reaches Runtime Agent inside AllocationSpec rather than through
  configuration-file I/O;
- every AgentTemplate selects an exact registered SandboxProfile. The baseline
  `local-workdir@1` creates fresh allocation scratch, exposes no host path
  through declarative contracts, cleans it before the slot returns to `idle`,
  and promises no OS, process or network isolation. The opt-in `podman@1`
  extension in [21](../spec/tools/code-execution.md) adds verified rootless code execution
  only for local direct project workspaces, with container removal before files;
- every `adk@1` AgentTemplate selects an exact digest-bearing default
  ModelPolicy through `modelPolicy`; reference-only Run executionConfig may
  select another compatible published policy, and AllocationSpec carries the
  effective resolved body separately while Gateway URL and token remain absent
  from AgentTemplate and ModelPolicy;
- one shared ModelPolicy kind serves Planner and Worker; `model` is mandatory,
  portable generation/budget fields are optional in the shared schema, and
  each consumer rejects a policy missing its required finite bounds rather than
  interpreting absence as unbounded execution;
- AgentTemplate `toolsets` explicitly selects versioned groups and a non-empty
  allowlist of exported tool names within each group; wildcard/all defaults,
  duplicate refs/names, unknown tools and cross-group name collisions are
  rejected, and unselected tools are not constructed;
- before first registration, Runtime Agent probes every enabled local runtime,
  Toolset and sandbox factory under bounded startup time and freezes the
  normalized positive result for that process `instance_id`; a Toolset may
  advertise only the subset of exported tools whose complete prerequisites
  passed, while probe failures remain local redacted diagnostics;
- the registered environment and capability snapshot are immutable for the
  process lifetime; changing dependencies or enabled factories requires a
  Runtime Agent restart and new `instance_id`, and heartbeat never carries a
  capability mutation;
- the built-in `run-artifacts@1` Toolset exports exactly `list_artifacts`,
  `read_artifact` and `write_artifact`; no generic Artifact tool is implicit,
  read-only selection is possible, and selecting a tool never expands the
  allocation's Server-side Artifact grant;
- instruction digests are SHA-256 over exact UTF-8 source bytes and
  AgentTemplate digests are SHA-256 over an RFC 8785 JCS normalized resolved
  manifest; tests prove that YAML presentation changes preserve identity while
  semantic or instruction-byte changes alter it;
- Runtime Agent verifies the instruction and AgentTemplate digests before
  Worker creation and fails preparation on a mismatch;
- every first-slice Runtime Agent runs the same Contractor code but advertises
  `adk@1` only after its local runtime probe passes; two processes may advertise
  different Toolset/tool subsets because their immutable environments differ;
- Workflow defaults plus reference-only Run overrides resolve an exact
  ModelPolicy for every modeled consumer and a complete Gateway route for every
  model-backed Planner; a Worker route may be completed or physically
  overridden by the pinned Run/Agent-label boundary in [07](../spec/configuration/runtime-config.md). Control Plane
  supplies the resulting allocation URL and optional token in RuntimeSettings
  over mTLS, and Agent clears secrets on release;
- Workflow Scheduler records `preparing -> running -> finalizing -> terminal`;
- every StageExecution has an immutable persisted `CreatedAt`. Its preparation
  and Planner deadline derives from that timestamp; missing creation time is
  invalid scheduler state, never a new budget from the time of observation;
- Scheduler requires contextual allocation reservations with complete resolved
  Runtime configuration, identity/label revision and an explicit performance
  collection policy. Missing placement inputs fail before Worker preparation;
  Scheduler does not invent configuration provenance or default a missing
  collection policy to disabled. Stored allocation reads require the same
  identity, positive label revision, current configuration schema and explicit
  policy; incomplete historical records fail without rewriting stored bytes.
  Model-free Workers retain a complete Runtime configuration with no LLM route;
- WorkflowRun records `initializing -> pending -> running -> succeeded`, with the final
  Stage acceptance, required output bindings and Run success in one transaction;
- cancellation or participant loss records
  `preparing/running -> aborting -> cancelled/interrupted`, with a durable
  StageTermination and bounded abort deadline;
- Run cancellation records `initializing/pending/running/waiting -> cancelling -> cancelled`,
  starts no new Stage, and cannot be held open by an unreachable Worker after
  the Stage abort deadline;
- tests cover both outcomes of the Run cancel-versus-success race: the first
  durable Run transition wins, and a finalizing Stage accepted after cancel
  does not create Workflow outputs;
- a candidate exact artifact revision remains authoritative if its logical
  binding advances before finalization; Scheduler never substitutes the newer
  current revision;
- bounded retry tests show that `maxAttempts` includes the first execution,
  every retry creates a new StageExecution, non-retryable outcomes and exhausted
  attempts execute `then`, and Run cancellation prevents another attempt;
- each Stage has a non-empty object-valued `agents` mapping; its keys become
  logical Worker names, order is ignored, and duplicate YAML keys, list and
  scalar shorthand forms are rejected;
- no Planner or Worker sees a physical Runtime Agent identity: Planner knows
  only logical Worker names and prepared WorkerHandles, while placement remains
  inside Control Plane;
- `passthrough@1` and `streamline@1` require exactly one logical Worker;
  `router@1` accepts a fixed non-empty mapping and cannot add or replace a
  binding;
- `streamline@1` and `router@1` use the Go ADK behind the existing
  Planner interface and share the bounded subtask-plan and validated
  `finish(StageResult)` operation;
- Streamline's dispatch function is exactly
  `execute_current_subtask(subtask_id)`, while Router's is exactly
  `execute_current_subtask(subtask_id, worker_name)` with `worker_name`
  constrained to the immutable Stage binding keys; explicitly selected
  MemoryTools may add only the surface derived by [08](../spec/tools/memory.md);
- Router's system instruction is deterministically augmented with every logical
  binding's name and purpose from the resolved immutable
  `AgentTemplate.description`, never placement or credential information;
- neither execution function accepts objective, instructions, parameters or
  ArtifactRefs; its adapter supplies the exact current stored subtask and
  complete immutable StageContext through the existing sequential A2A boundary;
- an unknown/stale subtask or Worker name is rejected before side effects,
  while a valid but poor Router selection remains an observable Planner routing
  decision rather than being silently corrected;
- model-backed Planners can report succeeded or failed only through `finish`;
  they expose no `escalate` operation, and Scheduler alone applies independently
  declared `failed` or `interrupted` escalation configuration by creating a new
  StageExecution with a pinned executionConfig override; escalation eligibility
  does not depend on a model-backed Planner candidate's `retryable` value;
- Streamline and Router enforce the selected ModelPolicy's model-call,
  cumulative-token, per-response and Worker-call limits plus a finite
  Stage/Planner wall deadline; exhaustion produces an interrupted, retryable
  running-phase StageTermination through Scheduler's bounded abort;
- Streamline and Router use their own exact ModelPolicy and LLMGatewayConfig
  selections from ResolvedExecutionConfig and never inherit a Worker
  token/model implicitly;
  both factories require valid per-invocation ModelAccess before constructing
  a provider client. Model/Worker-call and token budgets come from that policy;
  an omitted temperature stays omitted, while explicit zero is forwarded;
  successful responses must include consistent token usage, while provider
  error bodies and tokens never enter durable session events, reports, logs or
  the public API;
- Worker runtime code/heavy dependencies are absent from Server and live in the
  Runtime Agent deployment;
- A2A reaches the allocated Runtime Agent's own A2A Server; there is no proxy to
  a child Worker process;
- Planner completion produces a candidate result while Workflow Scheduler owns
  its durable acceptance;
- Planner uses one database-backed Contractor Session associated with the
  StageExecution; ADK-based Planner events are persisted only as redacted facts,
  receive ordered per-session and per-Run sequences before becoming visible on
  the public WebSocket, and never expose raw model or tool payloads, while
  Worker ADK state remains in the allocated Runtime Agent process and only its
  bounded Contractor-owned subtree is readable by Control Plane through the
  allocation-correlated private endpoint in
  [14](../spec/execution/worker.md);
- Runtime Agent finalizes and destroys its in-process Worker instance through
  its private control endpoint and returns bounded reports before terminal
  acceptance;
- the generic `adk@1` Worker lazily creates allocation-local sessions only for
  admitted invocations; isolated mode carries at most 4 MiB of eligible deep-
  copied State without events while shared mode retains one conversation, and
  either mode clears every session, app/user value and identifier before slot
  reuse;
- the same private endpoint supports idempotent bounded abort without requiring
  A2A `CancelTask` to succeed or reach a terminal Task state;
- private Control Plane/Runtime Agent traffic uses the deployment CA for mTLS;
  Runtime Agents additionally require the Control Plane URI SAN prefix
  `urn:contractor:control-plane:` while Control Plane treats all valid Runtime
  Agent certificates as peers with equal private-API privileges; their reported
  execution capabilities and assigned infrastructure labels may differ;
- each Runtime Agent process registers one in-memory `instance_id`; restarting
  it creates a new process identity (while retaining its certificate principal)
  and interrupts rather than adopts its old allocation;
- Runtime Agent sends a monotonic heartbeat and echoes the last received ack;
  both sides advance and expire the confirmed control lease only through that
  round trip, with the interval and lease duration owned by
  [02](../spec/execution/control-plane.md);
- local lease loss drains and ultimately kills Worker, then leaves Runtime
  Agent `fenced` with the old allocation ID until Control Plane explicitly
  acknowledges release; it cannot silently reuse the slot as `idle`;
- a response-only heartbeat-loss test demonstrates that unchanged echoed acks
  do not renew Control Plane's confirmed lease, the Stage enters `aborting` and
  the fenced slot is not reused before reconciliation;
- live Runtime Agent registrations, issued/confirmed heartbeat sequences,
  control leases and slot availability are held in Control Plane memory rather
  than written to PostgreSQL;
- Control Plane reserves the complete Stage Worker set atomically and returns
  either every ready WorkerHandle or no handles; preparation is idempotent for
  `stage_execution_id`;
- placement tests cover exact runtime and sandbox matching, per-Toolset tool-set
  containment, partial Toolset availability and a specialist/generalist
  multi-binding case where a complete matching exists but first-fit greedy
  assignment would report false insufficient capacity;
- one ArtifactStore binds public calls to UserScope and Worker calls to
  RunScope;
- input fork records an exact source version and lineage without requiring a
  blob copy, and Worker mutation leaves the user source unchanged;
- each `context.artifacts` entry explicitly declares `namespace`, `name` and
  `required`; Stage preparation pins present refs before allocation, records
  optional absence, and terminates without Planner on a missing required ref;
- retry creates a fresh StageContext snapshot, while later binding changes do
  not alter the exact refs already recorded for an execution;
- artifact bytes use the Server's private allocation-bound Artifact API while
  A2A carries only the ref;
- Artifact API reads/writes return a versioned ArtifactRef; Runtime maps only refs
  observed through trusted tools to server-declared Stage result `from` bindings,
  after which Planner and Scheduler preserve that revision through candidate
  acceptance without re-resolving current;
- Workflow loading validates that every `workflowOutputs` key is a declared
  Workflow output and every value is a result artifact declared by that Stage;
- every Workflow input/output and Stage-result artifact slot explicitly defines
  `required` and a non-empty `mediaTypes`; every non-workspace Stage result also
  defines a versionless `from` binding in an assigned Agent Namespace. Tests cover
  missing/invalid bindings, exact media-type matching, explicit `*/*` and
  incompatible output mappings;
- accepted output is frozen under `outputs/<slot>` and is not implicitly
  published back to UserScope;
- release clears the allocation's A2A identity, State, tools, access context
  and `local-workdir@1` directory; cleanup failure keeps the Runtime Agent
  fenced, while startup removes recognized orphan allocation directories
  before advertising an idle slot;
- public/private HTTP and A2A responses carry a bounded correlation ID, REST
  failures repeat it in their redacted body, and 5xx logs omit raw paths,
  bodies, secrets and provider exception messages;
- the executable fault matrix covers every durable lifecycle phase and every
  mutating first-slice operation under response loss, restart, races, fencing,
  mTLS/tampering attacks and bounded resource shutdown.

### 05 — First slice and open decisions › Label-driven infrastructure increment

The next implementation increment extends the runnable slice without changing
Workflow graph, Planner/Worker or artifact semantics. It must demonstrate:

- PostgreSQL stores immutable typed RuntimeConfig versions, revisioned
  label-to-config bindings, encrypted adapter credentials and durable Agent
  label sets;
- a Runtime Agent principal is derived from its certificate public key while
  every process incarnation retains a fresh `instance_id`; two concurrent
  single-slot agents use two CA-signed certificate/key pairs;
- Run creation always pins the default RuntimeConfig, accepts optional immutable
  labels, pins their exact binding/config versions in its idempotent
  transaction and survives later rebinding;
- Runtime registration reports immutable `otlp-http@1` and `http-proxy@1`
  adapter capabilities after bounded local probes;
- Control Plane overlays pinned Run-selected Runtime labels on the default and
  allocation-time Agent Runtime labels on both, rejects same-layer typed
  conflicts, performs complete
  capability placement and sends no unresolved label string as Runtime
  behavior;
- a `debug` Run emits bounded Worker telemetry to its pinned OTLP endpoint,
  content-free by default or unredacted with explicit `captureContent=true`
  for a trusted sink; optional Planner telemetry uses the same pinned Run
  config on Server;
- a `caido` Run applies HTTP proxying only to explicitly selected Worker model,
  tool-client or tool-subprocess targets while all Contractor private traffic
  bypasses it;
- rebinding either label requires no Workflow/AgentTemplate edit or Runtime
  restart, affects the next defined resolution boundary and leaves an active
  allocation unchanged;
- Operations/API/UI create and inspect safe config versions/bindings, assign
  Agent Runtime labels, select Run Runtime labels and never return credentials;
- a process-level end-to-end test proves old/new binding pinning, heterogeneous
  adapter placement, secret redaction, bounded exporter failure and complete
  release/slot reuse.

### 05 — First slice and open decisions › Shared-memory increment

The shared-memory increment adds a bounded artifact-backed notebook without
changing Stage or Planner ownership. It must demonstrate:

- `memory-tools@1` is selected through an AgentTemplate's exact operation
  allowlist and Runtime capability placement like any other Toolset;
- one logical note is one canonical RunScope artifact under the reserved
  `memory.` prefix, while generic model-visible Artifact tools cannot observe
  or mutate that prefix;
- Streamline Planner and its Worker share one `(Run, Agent Namespace)` view;
  Router receives per-operation `worker_name` enums and cannot access an
  unselected Worker's operation;
- retry, escalation and an intentional later-Stage Namespace reuse observe the
  same Run notes, while another Run observes none;
- the serialized `v1` call model enforces the 128-note count and unique creation
  ordinal, hidden Artifact CAS reports `memory_changed`, and canonical-payload
  reconciliation cannot duplicate an append after response loss;
- the durable Artifact fence rejects late Worker mutations, Planner call
  ordering prevents a late Planner mutation, and all retained reports, events,
  logs and telemetry exclude content, description and tags;
- a process-level test covers one Streamline Worker, two Router Workers,
  response loss, same-binding CAS conflict, quota and terminal release.

### 05 — First slice and open decisions › Agent Skills increment

The Agent Skills increment adds centrally updateable Worker guidance without
changing Planner, Workflow graph or A2A Task semantics. It must demonstrate:

- AgentTemplate has an optional sorted set of versionless `skills/*`
  ArtifactRefs; omission and an explicit empty set are identical;
- reviewable packages under `configs/skills/<name>` create only missing
  configured-owner UserScope artifacts through `SkillCatalog` before Server
  readiness; existing Artifact API updates are never overwritten and differing
  bytes surface as `seed_drift`;
- a user/operator updates a package through the existing User Artifact API with
  ordinary CAS and receives a new exact ArtifactRef; there is no Skill API;
- Run initialization resolves every referenced current binding once, records
  exact package refs/digests and forks them into the reserved RunScope `skills`
  Namespace before scheduling;
- current UserScope Artifact updates affect new Runs while retries and later Stages of
  an existing Run retain their exact pinned packages;
- AllocationSpec carries only exact RunScope refs/digests, and Runtime fetches
  bytes through the existing private Artifact API before constructing Google
  ADK's native SkillToolset with no remote registry;
- packages accept `SKILL.md`, `references/**` and `assets/**`, while dual
  Server/Runtime validation rejects scripts, executable ADK metadata, traversal,
  links, duplicates and bounded-size violations;
- model-visible generic Artifact tools cannot observe or mutate `skills`, a
  Skill cannot add a domain tool, and Runtime filters both `run_skill_script`
  and ADK's stock script-bearing guidance;
- final reports retain bounded Skill tool names/paths/counts/outcomes without
  instruction or resource content, and release removes all allocation-local
  extracted files and ADK activation State;
- the initial `contractor-old` Skill corpus is converted under `configs/skills`,
  validated, initialized as artifacts and connected only to the intended
  AgentTemplates rather than copied into MemoryTools.

### 05 — First slice and open decisions › Runtime filesystem increment

The workspace increment mounts exact Run artifacts into one allocation-private
logical `run_workdir` without giving Server access to a filesystem. It must
demonstrate:

- Workflow validates logical ZIP/state aliases, target directories,
  `direct|overlay` mode and overlay result slots;
- Scheduler pins exact RunScope refs and places the immutable workspace
  projection in every corresponding AllocationSpec;
- Runtime startup freezes only private `local|memory` storage limits and
  advertises modes plus the exact filesystem/Edit/change tool subsets;
- secure ZIP hydration and cumulative text overlay import are bounded,
  deterministic and isolated for every logical Worker;
- read and Edit tools consume narrow interfaces and behave consistently across
  local/memory/direct/overlay implementations;
- overlay diff/rollback use the current invocation checkpoint and graceful
  terminal results auto-export cumulative state plus human diff through the
  existing Artifact API before Scheduler fences writes;
- release, abort and lease loss erase disposable state; there is no host
  checkout mutation or model-visible materialization.

### 05 — First slice and open decisions › HTTP and Caido tools increment

The next Runtime tools increment ports bounded generic HTTP and Caido behavior:

- `http-tools@1` uses direct transport or the existing label-selected
  `tool-http` forward-proxy handle without exposing proxy configuration;
- `caido@1` uses a distinct label-selected `caido-graphql@1` adapter with a
  typed endpoint and optional `caido-bearer@1` credential;
- AgentTemplate grants exact tools while labels only configure infrastructure;
- response bodies/raw exchanges become bounded ordinary artifacts instead of
  unbounded prompt content;
- session secrets remain allocation-memory-only and teardown erases all
  clients/credentials;
- static GraphQL operations and deterministic fake-service/process tests cover
  the migrated Caido surface.

### 05 — First slice and open decisions › Workspace code-analysis increment

The structural analysis increment adds one explicitly selected
`code-analysis@1` Toolset without changing Artifact or Scheduler semantics:

- local and memory workspace Runtime Agents can advertise bounded Tree-sitter
  definition/symbol operations after an offline dependency probe;
- only a local workspace Runtime may advertise the pinned Trailmark graph
  operations, which execute in a killable allocation-local child;
- every analysis uses the current exact effective `WorkspaceSnapshot`, so
  direct and overlay edits invalidate derived state by digest;
- operation-level capability placement has no hidden shallow fallback;
- all results, scans, paths, child resources, caches and lifecycle cleanup are
  finite, explicit and content-redacted outside model tool results.

### 05 — First slice and open decisions › Structured taint-annotation increment

The next workspace increment ports the useful `contractor-old` annotation
surface without adding mutation to `code-analysis@1`:

- `taint-annotations@1` exports only `annotate_trace`, `annotate_validate` and
  `annotate_sink` over a narrowed workspace Writer;
- all three operations resolve a real function-like Tree-sitter definition,
  preserve decorators/attributes, indentation and newline style, and never
  accept a host path or ArtifactRef;
- exact replay is a no-op, ambiguous definitions require a line selector, and
  a concurrent workspace mutation fails with a retryable stable error rather
  than overwriting it;
- direct mode remains disposable, while ordinary overlay auto-export produces
  the cumulative state and diff artifacts selected by Workflow.

### 05 — First slice and open decisions › Deliberately deferred

Each decision should be added to its owning spec only when a concrete first-
slice implementation needs it.

### 05 — First slice and open decisions › Exit question

The slice is successful when the `adk@1` runtime executes different
AgentTemplates, `streamline@1` plans through one prepared Worker and `router@1`
routes current subtasks over a fixed prepared logical Worker set without
exposing placement or changing Workflow Scheduler, PassthroughPlanner, A2A or
artifact contracts.

The label-driven increment is successful when a compatible Runtime with no
Agent labels executes a `debug`/`caido` Run from typed pinned settings, an
Agent-specific label overrides the next allocation without changing semantic
policy, a live allocation keeps its old snapshot across rebind, and all
adapters release without exposing or retaining secrets.

The shared-memory increment is successful when Planner and corresponding
Worker coordinate through bounded notes without seeing revisions, Router
cannot cross a logical Worker's selected surface, Artifact CAS reconciliation
cannot duplicate an append, and a new Run starts with an empty Memory
Namespace.

The Agent Skills increment is successful when one ordinary Artifact CAS update
is observed by a new Run on any compatible Runtime Agent, an existing Run keeps
its prior exact package, native ADK progressive disclosure reads its reference
through no new content API, and no accepted first-profile package can execute a
script or expand the AgentTemplate's domain-tool authority.

The Runtime filesystem increment is successful when exact Run ZIP artifacts
hydrate equivalent isolated local/memory workspaces, direct/overlay tools
cannot escape `run_workdir`, and a graceful overlay result exports cumulative
state plus its checkpoint diff before terminal A2A without new Artifact APIs.

The HTTP/Caido increment is successful when one label retargets a Caido-backed
Run without changing AgentTemplate, only compatible Runtime Agents are chosen,
generic target HTTP statuses remain observable through direct/proxied paths,
and all response/session/credential bounds survive release and slot reuse.

The Worker-session increment is successful when new omitted Stage authoring is
observably isolated, an explicit shared Stage preserves sequential conversation
history, eligible State crosses only invocation boundaries inside the same
allocation, and cleanup/failure tests prove that no event, State value or
session identity survives allocation release or Runtime-slot reuse.

## From `06-server-ui-and-operations.md`

### 06 — Server UI and Operations

Status: **Working agreement; being refined in dialogue**

### 06 — Server UI and Operations › Delivered usability and remaining work

Verification: the UI journey gate
[V37-012](../../tasks/done/v37-012-ui-journey-verification.yml), the follow-up
tasks [V46-001](../../tasks/done/v46-001-run-draft-submission-generation.yml)
through [V46-004](../../tasks/done/v46-004-catalog-previous-page.yml), the
Performance gate [V32-008](../../tasks/done/v32-008-performance-release-gate.yml),
the Git gate [V35-005](../../tasks/done/v35-005-git-artifacts-release-gate.yml) and
the Evals gate [V38-010](../../tasks/done/v38-010-eval-release-gate.yml) record the
implementation and verification boundaries of this surface.

## From `07-runtime-labels-and-infrastructure-config.md`

### 07 — Runtime labels and infrastructure configuration

Status: **Working agreement**

## From `08-memory-tools.md`

### 08 — Shared MemoryTools

Status: **Working agreement**

## From `09-agent-skills.md`

### 09 — Agent Skills

Status: **Working agreement**

### 09 — Agent Skills › Allocation contract

Making `resolvedSkills` mandatory is a lockstep private-wire increment for the
single-VM deployment. Upgrade first lets every non-terminal Run terminate and
releases its allocations, then replaces Server and Runtime Agent processes
together. Missing/new fields fail closed during prepare; rolling mixed-version
operation is not promised.

### 09 — Agent Skills › Initial increment acceptance

1. valid `configs/skills` sources initialize missing configured-owner artifacts,
   survive crash replay and never overwrite an Artifact API update;
2. ordinary Artifact PUT creates a new exact skill ArtifactRef, a future Run
   uses it and an existing Run retains its prior source/fork;
3. two AgentTemplates share one logical owner artifact while one Run forks it
   once and each allocation receives only its declared subset;
4. Runtime fetches exact RunScope packages, verifies them, constructs native
   SkillToolset and Worker can list, load and read a reference;
5. dual validation rejects traversal, links, duplicates, oversize, scripts and
   behavioral ADK metadata;
6. generic Run Artifact tools cannot observe or mutate reserved `skills`, and a
   stale/fenced allocation cannot advance it;
7. `run_skill_script`, `search_skills` and script-bearing ADK guidance are
   absent from the model-visible request;
8. retry, later Stage and escalation use Run-pinned package after owner current
   advances;
9. release removes extraction and ADK activation State;
10. repeated list/load/resource disclosure calls stop at the allocation budget;
11. metrics expose bounded names/counts/outcomes but no content or refs.

## From `10-runtime-filesystems-and-edit-tools.md`

### 10 — Run workspaces and filesystem tools

Status: **Working agreement**

### 10 — Run workspaces and filesystem tools › Initial acceptance

1. A Runtime without workspace configuration continues to run existing
   artifact workflows and reports no workspace capability.
2. Go and Python strict fixtures round-trip the same WorkspaceContext,
   AllocationSpec projection and capability snapshot; unknown fields fail.
3. Scheduler pins exact source/state refs and never sends a logical mutable
   binding in AllocationSpec.
4. Local and memory sessions hydrate safe ZIP fixtures deterministically and
   remain isolated across two allocations.
5. Direct and overlay read/edit behavior is backend-independent; overlay lower
   data is unchanged and rollback restores the invocation checkpoint.
6. Exported cumulative state alone plus sources reconstructs the final text
   tree and validates both digests; `text/x-diff` describes only the current
   checkpoint delta.
7. Router Workers cannot see one another's unexported changes.
8. Traversal, link/special entry, archive bomb, binary text edit, quota,
   cancellation, lease loss and cleanup fault suites fail closed with no path
   escape or retained content leak.

### 10 — Run workspaces and filesystem tools › Initial acceptance › Local direct amendment acceptance

These cases use actual files modified outside WorkspaceWriter while the same
allocation remains alive, and cover both narrow handles and Toolset consumers.

1. External writes, creates, renames, deletes and empty-directory changes are
   visible on the next applicable call, without a refresh or executor signal.
   A same-size rewrite with restored mtime still returns the new contents.
2. Exact edit, append, copy and move use current contents. Unrelated external
   changes remain untouched; deleted files are never resurrected from memory.
3. Text/binary transitions update classification. Snapshots and observation
   digests reflect the current managed-text projection.
4. Search and analysis see external edits on their next call. Changed digests
   invalidate caches/cursors; previously returned snapshots remain immutable.
5. Single-file reads do not load unrelated source texts. Complete acquisitions
   fail safely on limits without fallback to the last valid snapshot.
6. Root/parent/leaf symlinks, hard links, special files, invalid names and
   adversarial path replacement cannot escape the root, leak outside contents
   or block the event loop indefinitely.
7. Workspace API calls preserve operation ordering. Cancellation cannot race
   release against an owned filesystem mutation. Detected external races fail
   safely without claiming protection from arbitrary concurrent writers.
8. Validation failure leaves files unchanged. Injected I/O/recovery failures
   report failure and fence on unconfirmed teardown; retained memory cannot
   be used as proof that disk state was restored.
9. Direct state import reaches disk before readiness; later external changes
   take precedence. Release removes the private copy. Persistence remains
   explicit ordinary Artifact writes.
10. Memory/direct and both overlay variants retain existing parity,
    checkpoint, export and isolation behavior. The local direct rule alone
    adds no execution capability or model-visible tool.

## From `11-http-and-caido-tools.md`

### 11 — HTTP and Caido tools

Status: **Working agreement**

The implementation ports useful behavior from `contractor-old`; it does not
preserve old unbounded responses, secret-bearing session artifacts or direct
environment-based infrastructure configuration.

### 11 — HTTP and Caido tools › Initial acceptance

1. Strict Go/Python RuntimeConfig, private settings, credential and provenance
   fixtures round-trip `caido-graphql@1`; all secret values remain redacted.
2. Label precedence can retarget Caido for a Run or Runtime Agent without
   changing AgentTemplate or Runtime startup environment.
3. A selected Caido Toolset cannot reserve/prepare on an Agent lacking the
   adapter or when effective configuration omits it.
4. Generic HTTP direct and mandatory-proxy paths produce equivalent bounded
   records, preserve target 4xx/5xx, never proxy private Contractor traffic and
   never fall back after proxy failure.
5. Session auth is erased at allocation teardown and absent from artifacts,
   metrics, logs, reports and tool views.
6. Every Caido tool passes against a deterministic fake GraphQL server with
   exact variables, pagination/bounds, mutation and error fixtures.
7. The migrated `configs/skills/caido` package is assigned only to
   `caido_analyst@1`, whose exact Toolset allowlist includes every HTTP/Caido
   operation named by the package; removing one fails the repository release
   gate and native Skill loading introduces no additional execution authority.
8. Real process tests cover two Runtime Agents where only one advertises Caido,
   label-based placement, response loss/release, and subsequent clean slot
   reuse.
9. A strict executable hardening matrix owns every input/response bound,
   proxy/redirect boundary, cancellation/lifecycle fault and retained-secret
   assertion; `make release-verify` composes that matrix with the real-process
   gate.

## From `12-code-analysis-tools.md`

### 12 — Workspace code-analysis tools

Status: **Working agreement**

### 12 — Workspace code-analysis tools › Port boundary

The initial port retains the two read-only Tree-sitter tools and nine read-only
Trailmark tools listed above. Existing unit fixtures may be adapted as semantic
references, but old architecture is not authoritative.

### 12 — Workspace code-analysis tools › Initial acceptance

1. Server and Runtime agree on exactly eleven exported operation names, while
   Runtime registration may report only a positive subset.
2. Memory and local processes with shallow dependencies advertise and execute
   `search_def`/`list_symbols` over equal snapshots with equal normalized rows.
3. Only a successfully probed local process advertises graph operations; a
   graph-requiring Stage waits rather than landing on memory or silently using
   shallow analysis.
4. Direct and overlay edits are visible on the next call because every engine
   builds from the effective snapshot rather than the provider underlay.
5. Shallow parsing leaves no retained AST/source cache and never blocks the
   asyncio heartbeat during a deliberately slow parse.
6. Trailmark runs in one allocation-local child, receives no RuntimeSettings or
   credentials, exposes no host path and is reaped on digest change, terminal
   completion, abort, lease loss and release.
7. Duplicate symbol names produce multiple `find_symbol` candidates; every
   graph relationship query accepts only one returned opaque ID.
8. Adversarial high-branching graphs prove path traversal stops at `limit + 1`
   and depth, response, step and deadline bounds without first materializing
   all simple paths; an unreachable target in a dense cyclic graph answers
   without enumeration.
9. File/byte/symbol/parser limits produce deterministic explicit incomplete
   coverage; pagination never claims an exact total for unexamined data.
10. Binary, unsupported, oversized, malformed and polyglot fixtures preserve
    confinement and report bounded coverage without a global parser patch.
11. Child hang, crash, OOM, malformed protocol, teardown timeout and concurrent
    edit tests fail closed, keep heartbeats responsive and prevent dirty slot
    reuse.
12. A real two-process heterogeneous test runs a shallow template on memory,
    holds a graph Stage until local capacity appears, then completes it and
    proves both slots can be reused.

## From `13-taint-annotations.md`

### 13 — Structured taint annotations

Status: **Working agreement**

### 13 — Structured taint annotations › Port boundary and acceptance

The old implementation and tests are semantic references, not authoritative
architecture. Version 1 intentionally changes these behaviors:

- raw fsspec/root access becomes one narrowed allocation Writer;
- `file/function` become normalized `path/symbol`, with a line selector for
  real ambiguity;
- first-match resolution and grep fallback are rejected;
- decorators/attributes/exports/templates remain syntactically intact;
- different trace targets may coexist and exact replay is a no-op;
- source/comment text and arbitrary filesystem/parser errors are not returned
  or retained;
- PHP follows the checked-in trace Skill and uses `//`; Haskell and Lua use
  their actual `--` line comment marker.

The initial increment is accepted when:

1. Go and Python agree on the exact ref/tools and local/memory Runtime Agents
   advertise them only after a complete parser/workspace probe.
2. All three operations produce byte-minimal valid edits across the nineteen
   language fixtures, including decorator/wrapper and CRLF cases.
3. Undefined and ambiguous symbols, unsafe inputs, capacity limits, exact
   replay, conflicts and cross-Toolset write races have deterministic outcomes.
4. Direct and overlay providers behave equally; overlay state/diff auto-export
   remains the only durable output mechanism.
5. Cancellation/close/release cannot leave work running or state retained, and
   metrics/logs/reports contain no content/path/value canaries.
6. A checked-in trace AgentTemplate/Workflow resolves the bundled Skill,
   selects no generic source writer and completes through PostgreSQL, mTLS,
   Scheduler, A2A and a real Runtime process with a reusable clean slot.

## From `14-worker-results-and-live-state.md`

### 14 — Worker results, deterministic observations and live state

Status: **Working agreement**

## From `15-worker-summarization.md`

### 15 — Optional terminal Worker summarization

Status: **Working agreement; post-observation increment**

## From `16-run-metadata-labels.md`

### 16 — WorkflowRun metadata labels › Conformance gate

The executable ownership matrix is
`tests/e2e/run_metadata_labels_matrix.yml`. It binds public validation,
transactional storage, idempotency, indexed owner-scoped querying, private-wire
validation, placement independence, Planner/Worker telemetry, lifecycle
recovery and both UI surfaces to named tests. Matrix validation is part of the
database-independent repository verification, so deleting or renaming an
owning test fails before the production-process gate is run.

The production-process scenario creates an unlabeled baseline, labeled A/B
Runs over one exact input revision and a final unlabeled reuse Run through the
ordinary Go Server/Scheduler, PostgreSQL, mTLS Python Runtime and public API. A
real `debug` Runtime label configures Planner and Worker OTLP while `eval.*`
metadata only annotates the two trace roots. Deterministic model and collector
fixtures prove that metadata does not enter model requests, process Resource
attributes or retained execution payloads, and that exporter failure neither
changes the result nor prevents reuse of the same Runtime slot. A second Server
identity over the same database proves owner-scoped list and detail behavior.

## From `17-projects-and-queue.md`

### 17 — Projects, reusable artifacts and the global queue

Status: **Working agreement**

### 17 — Projects, reusable artifacts and the global queue › First-release conformance

The executable conformance inventory is
`tests/e2e/project_workspace_matrix.yml`. Every listed case names a repository
test symbol; the matrix test fails if an ownership, exact-lineage, recovery,
placement, secret, browser, publication-race or standalone-regression category
is missing.

The production process gate uses real PostgreSQL, one restarted Go Server and
two certificate-distinct Python Runtime Agents. Their fixed startup
environments expose `validate_openapi` and `validate_likec4` on different
agents. One Project launches the two Workflows independently, while the gate
checks request replay, stale Project CAS, Queue membership, exact input forks,
create-only output publication, Runtime cleanup and safe target-credential
retention. Project metadata must not make an otherwise incompatible Runtime a
placement candidate.

The browser gate keeps one standalone Run as regression coverage, then creates
a Project through the independently served Node UI, uploads exact inputs,
launches a recommended Project Workflow, observes it in Queue and verifies the
published primary output and explicit **Run again** state. These gates are
aggregated by `make test-project-workspaces-release`.

## From `18-run-and-workspace-lifecycle-controls.md`

### 18 — Run and workspace lifecycle controls

Status: **Working agreement**

## From `19-audits.md`

### 19 — Audits: checks, findings, and iterative assessment

Status: **Working agreement; preparation storage, controller dispatch and generated initial inventory are implemented ([V62-002](../../tasks/done/v62-002-audit-preparation-store.yml), [V62-003](../../tasks/done/v62-003-audit-preparation-controller.yml), [V62-004](../../tasks/done/v62-004-audit-prepared-inventory.yml)). Minimal public controls work before a Round exists. Richer provenance, preparation UI and the generated-flow process/browser gate remain V62-010–012.**

### 19 — Audits: checks, findings, and iterative assessment › 3. Alignment with the current implementation

Verification: the baseline release gates are recorded in
[V25-012](../../tasks/done/v25-012-audits-release-gate.yml),
[V26-005](../../tasks/done/v26-005-audit-program-library-release-gate.yml) and
[V39-007](../../tasks/done/v39-007-audit-completion-release-gate.yml); the
preparation contract in [V62-001](../../tasks/done/v62-001-audit-composition-contracts.yml)
and the OpenAPI scan profile in
[V62-009](../../tasks/done/v62-009-audit-openapi-scan-profile.yml).

### 19 — Audits: checks, findings, and iterative assessment › 4. AuditProfile › 4.4 Preparation contract

Verification: preparation execution is delivered by
[V62-002](../../tasks/done/v62-002-audit-preparation-store.yml),
[V62-003](../../tasks/done/v62-003-audit-preparation-controller.yml) and
[V62-004](../../tasks/done/v62-004-audit-prepared-inventory.yml).

### 19 — Audits: checks, findings, and iterative assessment › 19. Additive implementation changes

| Component | Required change |
| --- | --- |
| Configuration | Read-only AuditProfile catalog plus owner-safe compatibility projection with resolved Workflow closures and digest |
| Server model | Audit entities, API, Controller, package/import and intake services |
| PostgreSQL | Durable work, claims, uniqueness, events, receipts, and retention holds |
| Run Service | Trusted pinned Workflow/Runtime/Skill creation, Audit provenance, managed publication, deletion guard |
| Scheduler | Optional generic Audit eligibility gate and terminal wake hint; unchanged Stage semantics |
| Runtime | Selected `security-findings@1` adapter, ordered Audit task-set decoding/result publication, and bounded receipt reconciliation |
| Artifact plane | Package import, proposal/fence transaction, protected Audit bindings and retention |
| Project deletion | Cancel/drain Audits before existing Run/ProjectScope purge |
| Frontend | Audit views, review actions, coverage and finding distinctions |

### 19 — Audits: checks, findings, and iterative assessment › 20. Acceptance gates

1. Multiple Audit modes coexist in one Project without crossing identities,
   baselines, reports, or queue state.
2. Exact replay of create/start/submission/collection returns the original result;
   same key with different payload conflicts.
3. OpenAPI canonical inventory and operation keys are stable under equivalent
   JSON/YAML and mapping-key reordering; exact source-content provenance still
   differs, method/path remain distinct, and `operationId` is not identity.
4. An invalid manifest creates zero child Runs; accepted manifests are immutable.
5. N checks create N logical items and N execution-item records. With
   `batchSize: 1` and `maxItemRunAttempts: 1` they create exactly N Runs; bounded
   compatible batches create fewer Runs, and bounded retries create additional
   execution-item records without creating additional logical items. A complete
   result is required for every batch member; proposals cannot alter the current
   barrier.
6. Crash injection around each submission/collection/assessment durable boundary
   neither loses work nor creates an extra Run.
7. Periodic reconcile recovers with all wake hints lost; two Controllers
   converge on the same accepted outcomes.
8. Finding submission survives response loss/Worker failure; stale replay does
   not create another artifact and write fence rejects a new submission.
9. Generic artifact content cannot forge a proposal receipt, review decision,
   Audit association, or confirmed finding.
10. `inconclusive` never confirms a finding; failed Run never satisfies a
    requirement.
11. A remote OpenAPI ref required by a selected operation, oversized package,
    foreign evidence, or mixed scope fails before dispatch/import; unsupported
    callbacks/webhooks and their untraversed refs become explicit gaps.
12. Review of an old subject revision cannot authorize a new revision and
    waiting review holds no allocation.
13. Owner pause, Audit pause, cancel, and Project delete have tested concurrency
    boundaries; all cleanup progresses while paused.
14. Child Run hard-delete is blocked before terminal release and a durable
    collection receipt, and is safe afterward even for missing, invalid,
    failed, or cancelled output; accepted exact evidence remains retained.
15. Public Project Artifact mutations cannot alter Audit-managed results.
16. Partial coverage, exclusions, manual items, and empty inventory are explicit;
    empty inventory never reports 100%.
17. Concurrent dispatch cannot exceed the current derived outstanding-Run
    window or finite item/proposal/total-Run budgets, and unknown crashed
    consumption is not counted as zero.
18. Specialist checks wait for compatible Runtime capability without downgrade
    and preserve coverage gaps on exhaustion.
19. Catalog-visible but unsupported profiles cannot start; profile reads expose
    stable compatibility reasons and start rechecks them atomically.
20. Exact Skill package revisions selected at Audit start are used by later
    child Runs even after the current Skill binding changes.
21. Analyst TP/FP and severity edits retain exact-subject history and reject
    stale writes; duplicate/reopen and new assessment do not inherit a verdict.
22. An API-only script traces a rated finding to its source Workflow, all
    verification attempts and exact checklist entries after catalog replacement
    and permitted Run deletion. Pagination detects intervening revisions.
23. Retry, duplicate and multi-requirement fixtures preserve attribution without
    multiplying distinct confirmed findings; a standards mapping alone never
    becomes a causal checklist source.
24. A direct Workflow finding needs no prior hypothesis or synthetic check item;
    accepted verification traces to that same Run's exact result/contract, while
    absent or invalid verification evidence never implies TP.
25. A profile-gated active Workflow item cannot create an execution intent
    without an accepted, unexpired decision for its exact task/profile digest;
    inventory data cannot weaken that gate.
26. Human report review displays immutable proposed bytes. Accept publishes
    those exact revisions, reject/expiry terminates with a stable reason, and
    finding/import mutations cannot cross the pending candidate boundary.
27. The curated Top 10 and ASVS pilot release gate names every executable
    definition, PostgreSQL, capability, lifecycle, production-process and
    independent-browser owner; both historical reports reopen with their exact
    denominator and non-certifying language after current program catalog
    entries disappear and the Server restarts.
28. Start/resume distinguish an omitted allowance, zero and a positive bounded
    value in validation and idempotency. Pauses preserve remaining time across
    collection updates and restart; owner Queue waiting consumes it. Time-limit
    expiry pauses admission while existing Runs drain and collect.
29. Terminal Audits, including old `completed`/`failed` records with
    `deadline_exhausted`, reject Resume without changing reports, accepted
    results, exact baseline, evidence, attempts, holds or reviews. A current
    deadline pause can resume with the default allowance, a new limit or no
    limit. Expired approvals require fresh exact-subject decisions; replay
    does not renew a review or change the deadline twice.
30. Audit input autoselection waits for complete pagination, preserves explicit
    choices and does not infer uniqueness after a load error. Start/Continue
    dialogs submit the selected allowance, preserve ordinary paused time by
    default, and perform no mutation when dismissed before confirmation.

### 19 — Audits: checks, findings, and iterative assessment › 21. Delivery increments and deferred work

**Increment 1:** strict AuditProfile catalog and compatibility gate;
deterministic checklist/OpenAPI inventory; one fixed immutable Round;
one-item bounded check Runs; PostgreSQL reconciliation; exact collection
receipts and package retention; coverage report; two executable demo profiles;
polling UI; mandatory ownership, deletion, Skill/credential, idempotency, and
Run-deletion gates.

**Increment 2:** `security-findings@1` for Audit and ordinary Runs; finding
triage; exact human review; bounded discovery/assessment and multiple rounds;
analyst TP/FP and severity history; API backtrace from finding through proposals
and verification attempts to exact Workflow/checklist origins, including after
source Run deletion; exact item-action/report decisions; a documented
external-script example.

**Increment 3:** the curated licensed versioned standard package catalog,
canonical seeding, exact Audit pins, license-aware read API, bounded OWASP Top
10:2025 source-risk program, and selected five-requirement ASVS 5.0.0 Level 1
pilot. Their release matrix binds exact identity, denominator,
mixed coverage, restart, catalog replacement, lifecycle, production-process and
independent-browser checks to executable owners. Comparison of Audits for the
same system, retest of accepted findings against a new baseline, the complete
ASVS corpus, and broader program coverage remain follow-up work.

**Increment 4:** bounded compatible check-item batches in one ordinary Run;
deterministic task-set packages; complete atomic result import; per-item
attempt, evidence, proposal, coverage and provenance retention; and a two-item
checklist demo while preserving the one-item wire path.

Verification: increments 1–2 are recorded in
[V25-012](../../tasks/done/v25-012-audits-release-gate.yml) with
[V25-010](../../tasks/done/v25-010-security-findings-intake.yml) and
[V25-011](../../tasks/done/v25-011-audit-review-and-multiround.yml); increment 3 in
[V26-005](../../tasks/done/v26-005-audit-program-library-release-gate.yml); increment 4
in [V26-004](../../tasks/done/v26-004-batched-audit-executions.yml) and
[V26-006](../../tasks/done/v26-006-audit-batch-byte-budget.yml); Worker completion in
[V39-007](../../tasks/done/v39-007-audit-completion-release-gate.yml); preparation
in section 4.4.

### 19 — Audits: checks, findings, and iterative assessment › 22. Sources and decision status

- The current Contractor specifications, especially [00](../spec/configuration/workflows.md),
  [02](../spec/execution/control-plane.md), [03](../spec/artifacts/artifact-store.md),
  [04](../spec/execution/scheduler.md),
  [12](../spec/tools/code-analysis.md), [13](../spec/tools/code-analysis.md),
  [14](../spec/execution/worker.md),
  [17](../spec/artifacts/projects.md), and
  [18](../spec/artifacts/retention-and-deletion.md), and
  [20](../spec/execution/scheduler.md).
- [OWASP Top 10:2025](https://owasp.org/Top10/2025/), verified 2026-09-05.
- [OWASP ASVS](https://owasp.org/www-project-application-security-verification-standard/)
  and its [official repository](https://github.com/OWASP/ASVS), including stable
  version 5.0.0 and version-qualified requirement identifiers, verified
  2026-09-05.

## From `20-scheduler-concurrency-control.md`

### 20 — Workflow Scheduler concurrency control

Status: **Working agreement**

## From `21-podman-sandbox.md`

### 21 — Allocation-scoped Podman execution sandbox

Status: **Implemented; execution remains opt-in** (verified by [V31-008](../../tasks/done/v31-008-podman-release-gate.yml))

### 21 — Allocation-scoped Podman execution sandbox › Acceptance and implementation sequence

The release gate is `make test-podman-release` on a rootless Linux host. The
[executable matrix](../../tests/e2e/podman_sandbox_matrix.yml) maps all eleven
cases below to concrete tests. Unsupported host configurations and deferred
skill/network features are not certified by this gate.

Verification: the local direct prerequisite
[V30-004](../../tasks/done/v30-004-local-direct-release-gate.yml), the Podman tasks
[V31-001](../../tasks/done/v31-001-podman-contracts-settings-placement.yml) through
[V31-008](../../tasks/done/v31-008-podman-release-gate.yml) and the
[verification record](../../runtime/PODMAN.md#release-verification-v31-008)
document the environment, effective policy and verification scope.

Required acceptance cases:

1. Valid local direct placement succeeds. Missing workspace, overlay, memory,
   unselected tools, missing image and failed prerequisites never execute code
   or silently select a different backend.
2. Prepare creates exactly one container. Repeated prepare, uncertain create
   response and partial failures preserve idempotency and owner-scoped cleanup.
3. A file edited through `edit-files` is immediately visible to a command; files
   created/modified/deleted by commands are visible to filesystem, observation
   and analysis calls without refresh. Other allocations remain isolated.
4. Tool input cannot select host paths, engine options, mounts, credentials,
   execution user or network. Host shell metacharacters stay container input.
   Mount boundaries, UID mapping and root/parent/leaf path attacks are tested.
5. Stdout/stderr floods and invalid UTF-8 remain bounded. Nonzero command exit,
   engine failure and timeout produce distinct observations; no command is
   automatically replayed after uncertainty.
6. Normal completion, fork/detach, cancellation, timeout and output-limit tests
   prove no workload writer survives before workspace ownership is released.
   An unconfirmed stop fences the allocation and prevents reuse.
7. Two overlapping tool calls and command/edit/snapshot races preserve the
   common workspace ordering. Heartbeats remain responsive during engine and
   filesystem I/O.
8. `INPUT_REQUIRED` and sequential invocations reuse one allocation container;
   finalize stops it, release removes it before mounted files and repeated
   finalize/release remain idempotent.
9. Runtime crash without immediate restart stops workload execution within the
   supervisor liveness bound. Restart cleans only its predecessor resources,
   including containers whose create response was lost.
10. CPU/memory/pids/tmpfs/network policy is verified in the real supported
    rootless deployment; undeclared disk-quota guarantees are never asserted.
11. Existing `local-workdir@1`, memory/overlay workspaces, trusted host
    validators and script-free skills retain their behavior and authority.

## From `22-performance-metrics-and-profiling.md`

### 22 — Operations performance metrics and Go profiling

Status: **Implemented** (verified by [V32-008](../../tasks/done/v32-008-performance-release-gate.yml))

### 22 — Operations performance metrics and Go profiling › Implementation status

Verification: the task files
[V32-001](../../tasks/done/v32-001-performance-contracts-and-settings.yml) through
[V32-008](../../tasks/done/v32-008-performance-release-gate.yml) record the
implementation and release-gate evidence; see
[Verification and delivery](#verification-and-delivery). Allocation performance
policy is a separate forward-only migration from Run resumption. Historical
allocation resources cannot be reconstructed when they were not collected.

### 22 — Operations performance metrics and Go profiling › Verification and delivery

Verification: the task files
[V32-001](../../tasks/done/v32-001-performance-contracts-and-settings.yml),
[V32-002](../../tasks/done/v32-002-server-performance-collection.yml),
[V32-003](../../tasks/done/v32-003-postgres-performance-history.yml),
[V32-004](../../tasks/done/v32-004-allocation-resource-sampling.yml),
[V32-005](../../tasks/done/v32-005-performance-api-and-allocation-history.yml),
[V32-006](../../tasks/done/v32-006-operations-performance-ui.yml),
[V32-007](../../tasks/done/v32-007-opt-in-go-profiling.yml) and
[V32-008](../../tasks/done/v32-008-performance-release-gate.yml) record the
implementation, dependencies and release-gate evidence.

Required acceptance:

1. All four startup-switch combinations work; metrics off creates no new
   collectors, metric queues/timers or diagnostic DB connections; pprof off
   creates no listener. Existing execution accounting/budgets still work.
2. Fake-clock tests prove 15 s / 60 s / 300 s cadence, bounded memory/queues,
   non-overlap, truthful missing/zero/reset semantics and minute aggregation.
3. HTTP accounting preserves streaming/WebSockets, concurrent totals and
   quantiles while avoiding unbounded dimensions and observer-traffic feedback.
4. Disposable PostgreSQL tests cover limited roles, hidden activity, lock waits,
   counter resets, database outage, budgets, retention, restart and query plans.
5. Allocation tests cover short/long/sequential runs, unsupported/disabled
   collection, partial reads, failure/loss, retries, cleanup and old/new contracts.
6. Authenticated API/browser tests cover charts, disabled/stale/gap states,
   bounded queries, owner isolation and durable history after authoritative release.
7. Profiling tests cover loopback-only exposure, no public/private/cmdline route,
   finite durations/concurrency, valid readable output and cancellation/shutdown.
8. A reproducible enabled/disabled load comparison records CPU time, allocations,
   RSS, throughput, p95 latency and DB query/write counts, with toolchain/hardware/
   workload/repetitions. Measure idle pprof and active profiling separately.
   Establish before/after evidence, not an unmeasured universal overhead percentage.

The release gate includes race-enabled Go and PostgreSQL checks, Python Runtime
checks, real cross-language mTLS finalize/abort resource reports and the built
Node UI against the production browser stack.

## From `23-artifact-blob-backends.md`

### 23 — Server Artifact blob backends

Status: **Implemented — PostgreSQL/filesystem; S3 deferred** (verified by [V34-005](../../tasks/done/v34-005-blob-backends-release-gate.yml))

### 23 — Server Artifact blob backends › Verification and delivery

- Run the same Artifact contract tests against PostgreSQL and filesystem,
  including exact 64 MiB read/write, oversize refusal, CAS, write fences, scope
  isolation, forks, Audit/Skill retention and concurrent last-reference purge.
- Inject failure before file publication, before/after registry commit and
  before unlink; referenced successful bytes are exact or explicitly missing,
  and orphans are removable through the offline command.
- Demonstrate Kubernetes-compatible PostgreSQL startup with no PVC and a
  read-only root; restart the Server against the same DB and read prior bytes.
- Demonstrate filesystem with an ephemeral directory; after replacing it,
  missing payloads are reported truthfully and no automatic recovery is claimed.
- Validate backend mismatch, unsupported S3, unwritable root, traversal/symlink
  attacks, transfer saturation/cancellation and same-digest delete/recreate races.

Verification: the task files
[V34-001](../../tasks/done/v34-001-blob-contracts-and-postgres.yml) through
[V34-005](../../tasks/done/v34-005-blob-backends-release-gate.yml) record this
increment; [V34-006](../../tasks/done/v34-006-overlay-export-size-contract.yml)
records the overlay export limit contract.

## From `24-git-artifacts.md`

### 24 — Git repositories as Artifact inputs

Status: **Implemented** (verified by [V35-005](../../tasks/done/v35-005-git-artifacts-release-gate.yml))

### 24 — Git repositories as Artifact inputs › Network and resource limits › Reader implementation and evidence

Integration tests run against native Git fixture servers over verified HTTPS
and authenticated SSH, advance the branch between advertisement and fetch, and
check the original commit, annotated tags, deterministic ZIP bytes, wrong owner
and changed host keys. Adversarial tests cover transport refusal, redirects,
TLS, advertisement/object/delta caps, malformed packs, unsafe content and
cancellation. The importer must also complete in a read-only container with no
writable checkout or temporary directory.

Verification: [V35-002](../../tasks/done/v35-002-bounded-memory-git-snapshots.yml)
and [V35-005](../../tasks/done/v35-005-git-artifacts-release-gate.yml) record the
reader evidence and measured memory, together with the
[Git artifacts guide](../guides/git-artifacts.md).

### 24 — Git repositories as Artifact inputs › Delivery and verification

Verification: the task files
[V35-001](../../tasks/done/v35-001-git-key-settings-and-contracts.yml),
[V35-002](../../tasks/done/v35-002-bounded-memory-git-snapshots.yml),
[V35-003](../../tasks/done/v35-003-git-artifact-publication-and-provenance.yml),
[V35-004](../../tasks/done/v35-004-git-settings-and-import-ui.yml) and
[V35-005](../../tasks/done/v35-005-git-artifacts-release-gate.yml) record key
storage, the bounded reader, import API/provenance, the UI and the release gate.

The gate must demonstrate public HTTPS and private SSH imports, key replacement
and removal, unknown/changed host keys, forbidden destinations/redirects,
branch movement and exact commit provenance, CAS races, malformed/oversized
packs, cancellation and saturated capacity. Verify selected ZIPs with the
existing Runtime source consumers. Test the actual Server with read-only root,
PostgreSQL blobs and no writable Git/blob/tmp directory or PVC, then repeat
Artifact publication with filesystem storage. Measure memory and verify that
key/canary material appears nowhere in responses, artifacts or captured logs.

The reproducible gate, deployment instructions, failure matrix and measured
memory are documented in [Git artifacts](../guides/git-artifacts.md).

## From `25-audit-worker-finalization.md`

### Audit Worker completion contracts

Status: **Implemented opt-in; deterministic release gate verified ([V39-007](../../tasks/done/v39-007-audit-completion-release-gate.yml)).**

### Audit Worker completion contracts › 6. Diagnostics and release requirements

Rollout is capability-first: install compatible Server/Runtime/report readers,
verify the release gate, then create an Audit from the current catalog.
All bundled Audit profiles pin this strategy at version 1. Distinct scenarios
have separate names. Existing
resolved snapshots are not rewritten by a catalog reload. There is no
durable recovery of a partially collected invocation, target summarizer, or
multi-Worker completion strategy in this version. These examples are not live
deployment or certification evidence.

Release tests must include ordinary/no-contract regression, spoofed activation,
unsupported Runtime placement, per-item validation/correction/replay, parallel
submissions, reversed submission order producing identical bytes, no-call and
partial batches, invalid calls not satisfying the gate, shared-session reuse,
hard budgets and summarizer configuration rejection, cancellation at each
boundary, dropped write replies, conflicting artifacts, and process loss.
End-to-end tests must pass a runtime-produced ZIP through the real Go importer,
including the missing-required-evidence regression. No live model or demo
mutation is required for deterministic acceptance tests.

Provide a dedicated release target that requires an isolated PostgreSQL test
database and executes the new persistence/restart and importer bridge tests
without skips. Plain go test ./... and make verify are not sufficient evidence:
existing database tests skip when CONTRACTOR_TEST_DATABASE_URL is absent.
Include same-Run versus new-Run retry, escalation variants, mixed batch/scalar
replays, exact-budget completion, input-alias changes, and real ADK continuation.

The executable gate is `make test-audit-completion-e2e`, included in
`make release-verify`. Its fixed required-case matrix and no-skip report
validation are documented in the
[Audit completion release gate](../testing/release-gates.md#audit-completion-release-gate).
The bridge uses production Run creation, allocation grants, mTLS Artifact API,
PostgreSQL artifact storage and Audit importer, with a separate pinned ADK
Runner process and scripted model. Scheduler driving is test orchestration; the
gate does not establish live-model quality or a full Server/Planner/A2A
deployment test.

## From `26-portable-evaluation-format.md`

### 26 — Portable evaluation format and execution bindings

Status: **Implemented in playground-v2 and offline-verified ([V41-008](../../tasks/done/v41-008-eval-format-release-gate.yml)).**
The managed Evals UX is owned by [30](../spec/evals/managed-evals.md). Offline conformance
does not prove instruction quality or a successful live target campaign.

### 26 — Portable evaluation format and execution bindings › 12. Delivery gate before evaluations

The delivery gate covers schemas, typed records, converters, adapter separation,
frozen plans, recovery, scoring/comparison and publication using **offline/recorded
fixtures and deterministic API conformance tests**. It does not run paid/model quality evals,
launch benchmark campaigns, publish candidate default configurations or change
running Audits. Ground-truth sentinels and fault fixtures are mandatory.

Conformance evidence must retain the hashes of the implementation and inputs
actually tested. A later source change requires new evidence rather than
updating an old report to appear current.

The release gate covers schemas/examples/negative fixtures, old v1 readers,
one unchanged case/scorer across Contractor and recorded providers, complete fault
walkthroughs, owner-safe publication and stable comparison arithmetic. It records
exact implementation/schema/scorer build digests in the readiness report. Those
pins identify the tested revision, including its retained specification bytes.
Compatibility passing is not evidence that new instructions improve model
quality; a model-quality evaluation is a separate task with its own explicit
budget and environment.

Verification: the gate result is recorded in
[V41-008](../../tasks/done/v41-008-eval-format-release-gate.yml); instruction
fixtures, provider bindings/pilot plan and the adoption decision are tracked by
[V40-001](../../tasks/done/v40-001-agent-instruction-eval-fixtures.yml),
[V40-002](../../tasks/v40-002-agent-instruction-paired-runner.yml) and
[V40-003](../../tasks/v40-003-agent-instruction-eval-decision.yml).

## From `27-findings-tools-and-collections.md`

### 27 — Finding tools and immutable collections

Status: **Implemented and verified** — see the
[findings contract gate task](../../tasks/done/v43-005-findings-contract-gate.yml).

### 27 — Finding tools and immutable collections › Verification and rollout

Producer guidance lives in a separate instruction file; the `trace` Skill and
instruction-eval baselines are unchanged by it. Neither configuration changes
default deployment or establishes model-quality improvements.

Required integration coverage: PostgreSQL publication/retention tests and an
HTTP test that passes the ZIP into ordinary Run creation, covering concurrent
replay, later review changes, all finding contributions, foreign/missing
sources, source Run deletion and rollback after the ZIP write but before its
receipt; they run with `-tags=integration` and fail when explicitly selected
without a test database. Runtime tests cover Python decoding of the same
fixture, real ArtifactClient preparation behavior, previews, pagination and
independent tool selection. The findings process gate, `make test-findings-e2e`,
proves producer → publication → ordinary input fork → preparation → reader
across Server/Runtime processes. Model quality evals remain separate and follow
the portable evaluation format in [26](../spec/evals/format.md).

### 27 — Finding tools and immutable collections › Finding proposal and selected authoring interfaces › Development contract and verification

The shared location fixture exercises Go, Python and UI validation. The findings
process gate includes a real facade→intake→API→collection→reader round trip after
producer deletion. Runtime tests cover actual provider declarations, hidden call
identity, full HTTP capture, invocation isolation, eviction and immutable snapshots.

## From `28-finding-analysis-and-sarif.md`

### 28 — Structured finding analysis and SARIF export

Status: **Draft; Specified target; not implemented.**

## From `29-tool-workers.md`

### 29 — Deterministic tool Workers

Status: **Implemented**; [V55-003](../../tasks/done/v55-003-model-free-tool-worker.yml) records verification.

### 29 — Deterministic tool Workers › Verification matrix

Release verification must exercise valid/invalid bindings and Go/Python/schema/digest parity;
no-gateway config resolution and placement with broken irrelevant model routes;
real A2A invocation with an artifact transport and controlled tool; successful
publication; missing capabilities; timeout/cancel/close; report/receipt failures;
in-memory and recreated-allocation replay; and changed-input rejection. Existing
ADK configuration, allocation and runtime suites must pass unchanged.

## From `30-managed-evals.md`

### 30 — Independent managed Evals and external producers

Status: **Implemented and deterministically verified ([V38-010](../../tasks/done/v38-010-eval-release-gate.yml)): native setup, comparison/review, optional Playground client and independent process acceptance.**

### 30 — Independent managed Evals and external producers › Dataset and variant selection › Workflow and Audit parity

Conformance fixtures and the release gate include all four kind/control combinations,
an Audit with multiple children/rounds/retries, overlapping child durations,
missing child usage and duplicate observation replay. UI labels and drill-down
follow the selected kind; they never suggest an Audit is just one child Run.

### 30 — Independent managed Evals and external producers › Compatibility and external clients

This contract is producer-neutral. An independent minimal HTTP client must pass
the same external registration/submission/replay/assessment tests without importing
Playground. Native UI process tests run with Playground entirely absent. These
are the release tests for the selected independence boundary.

## From `32-scan-planning.md`

### Deterministic scan planning › Results and coverage

Verification uses fabricated local inputs and fake Worker/Artifact adapters:
`go test ./internal/scanplan/... ./internal/planner/... ./internal/scheduler/...`.
These checks do not launch scanners, model calls or active network probes.

## From `33-autonomous-pentest-audits.md`

### 33 — Autonomous web and API pentest Audits

Status: **Draft; specified implementation target; not implemented.**

### 33 — Autonomous web and API pentest Audits › 14. Delivery and existing work

Reuse existing work by contract, with task files retaining their own status.
This table does not mark any task complete or create new task records.

| Existing direction | Required integration |
| --- | --- |
| V54 proposed 001–004 | Implement the engagement, enforced target transport, identities and trusted evidence specified here; source is optional |
| V54 proposed 005–007 | URL-only discovery plus source enrichment, typed plans, explicit round routing, verification and replay |
| V54 proposed 008–010 | Shared API/UI, fixtures/scorer and API profile release gate |
| V54 proposed 011–012 | Browser and callback extensions with full web acceptance |
| V55 and [scan planning](../spec/tools/scanners.md) | Reuse prepared inputs, fixed Workers and durable scan-job recovery only after scope/identity/capture integration; scan planner owns scanner fanout, Audit owns hypotheses |
| V55 crawler/ranking work | Optional discovery/ranking adapters; neither is required for direct HTTP MVP |
| V61 toolset configuration | Reuse exact settings, credential holds and MCP lifecycle; target principals are identities within a Run, not conflicting per-agent infrastructure settings |
| [Portable Evals](../spec/evals/format.md), [managed Evals](../spec/evals/managed-evals.md), V38 | Add pentest fixtures/scorers and attributed comparison through existing engines |
| V60 recovery/runtime/evidence/security reviews | Include the new invariants in their relevant checks; the full review backlog is not a prerequisite for contract work |
| [Finding analysis](../proposals/0028-finding-analysis-and-sarif.md) | Align evidence decoders and vocabulary; SARIF remains independently deliverable |

Implement in this dependency order:

1. Closed schemas, shared valid/invalid fixtures, profile extensions, baseline
   projection, credential revisions and permit/journal persistence contracts.
2. Enforced direct HTTP transport, shared accounting, cancellation and unknown
   outcome handling; prove denial/fault cases before model-driven probing.
3. Independent identities, login, redacted capture and trusted observation
   registration, including anonymous suppression of Project auth.
4. URL-only discovery and separate source enrichment with explicit coverage.
5. Typed hypothesis admission, retained cross-round dependencies, deterministic
   predicates and fresh-session replay.
6. Both API profile versions, UI/API/CLI projections and deterministic release
   gate; then frozen live quality evaluation.
7. Browser egress/context/trace capability and XSS/web-auth gates.
8. Callback lifecycle/SSRF proof and complete web v2 gate.

Fixtures and independent oracles can be implemented after step 1 while transport
work proceeds. Direct HTTP MVP does not wait for every scanner, MCP integration,
SARIF exporter or complete Evals browser journey.

### 33 — Autonomous web and API pentest Audits › 15. Acceptance and evaluation gates

The release gate exercises the production service/transport path with real
Runtime processes and deterministic targets. Mocked model output may propose
plans but cannot manufacture accepted capture receipts or oracle success.

| Gate | Required positive and negative cases |
| --- | --- |
| Inputs and separation | URL-only with no source/OpenAPI; exact source-assisted input; anonymous/A/B in both modes; source/ground-truth inputs rejected from blackbox closures |
| Scope | Allowed HTTP and explicit private test network; denied redirect, rebinding, encoded path, protected alias, cross-origin credential transfer and alternate egress; unsupported adapter rejected before sending |
| Budgets | Concurrent Workers share request/rate/concurrency/token bounds; retries/login/subresources counted; duplicate settlement and restart cannot refund unknown usage |
| Sessions | Correct A/B controls, truly anonymous requests, cookie/storage separation, expiry/refresh failure, replacement login and replay generation separation |
| Reconnaissance | HTTP links/forms/JS and optional API inputs; missing surface, crawl limits, parse errors and denied routes produce retained gaps |
| Admission and rounds | Receipt/ordinal matching, exact role routing, dedup contributions, retained dependencies after source deletion, immutable accepted worklists and bounded later hypotheses |
| Proof | Injection/authz positives pass observation and replay; paired fixed/negative fixtures fail the same oracle; status-only, fabricated trace, missing control, changed predicate and inconsistent replay cannot yield reproduced |
| Evidence | Forged invocation/permit/receipt rejected; redaction/truncation propagates unknown; credential canaries absent from tool views, events, artifacts, reports and logs |
| Recovery | Crash before/after send and before/after capture receipt; duplicate delivery; lease loss; cancel; unknown mutation never automatically repeated; unproven teardown quarantines capacity |
| State and cleanup | Isolated objects or serialization; failed cleanup and unknown residue reported; fresh session not confused with fresh target |
| Reporting | Proof distinct from human TP/FP; incomplete zero-result Audit remains incomplete; deterministic report survives permitted source Run deletion |
| Browser/callback | Required for web v2: actual execution marker, isolated contexts, blocked subrequest bypasses, authentic nonce/window receipt, denied direct collector access, teardown and late-arrival behavior |

Use existing Playground target lifecycle and evaluation services. Start with
selected VAmPI injection/authz cases, then suitable DSVW/crAPI cases and small
fixtures for missing controls. A fixed branch is a negative control for its
specific case, not a declaration that an entire target is secure. Runtime
reachability is a separate setup assertion; localhost Compose URLs are not
assumed reachable from a placed Worker.

For an end-to-end authorization case, create a disposable object owned by B,
prove B's identity and ownership, attempt access as A and anonymous against
paired vulnerable/fixed endpoints, then replay on fresh sessions and objects.
The external evaluator checks actual target state and the report's proof chain.
The Worker receives the permitted task and account bindings, not private
verifier/reset hooks or expected answers. Injection fixtures similarly use a
controlled target-side marker and a matched negative case.

Keep protocol conformance separate from model quality. Deterministic gates
require every required case to pass, zero out-of-scope sends, zero leaked
credential canaries and zero unjustified proof promotions. A single failure
blocks release of the affected capability.

Before the first live experiment, freeze an evaluation manifest with exact
profiles/models/tools/Skills, fixture/scorer revisions, source/auth matrix,
repeat count, held-out cases, budgets, success criteria and numerical quality
thresholds. Quality thresholds are an experiment input; they are not invented
from the observed score. A missing manifest blocks a quality claim, not local
protocol development.

Compare the existing URL-only Workflow with the new Audit under equal inputs
and budgets. Compare source-assisted in a separate pair whose sole intended
change is source access. Measure unique externally confirmed findings, negative
control false positives, successful replay rate, attempted/completed/gap
coverage, time to first reproduced finding, requests, tokens and cost. Flag
capture remains a benchmark-specific metric rather than a replacement for
proof-chain/report quality. Retain every failed, cancelled and inconclusive
sample in the denominator.

Implementation readiness requires the schemas and gates above, published exact
profile closures and a recorded deterministic release result. Live quality
claims additionally require the frozen evaluation and its retained results.
Writing this specification establishes neither implementation nor measured
model quality.

## From `34-audit-check-prioritization.md`

### 34 — Contextual Audit checklist prioritization

Status: **Draft integration target; pure core implemented ([V64-000](../../tasks/done/v64-000-audit-priority-core.yml)); Audit capability not enabled.**

### 34 — Contextual Audit checklist prioritization › 2. Ownership and prerequisite contracts

Verification: integration depends on the preparation contract, store and
controller tasks
([V62-001](../../tasks/done/v62-001-audit-composition-contracts.yml),
[V62-002](../../tasks/done/v62-002-audit-preparation-store.yml),
[V62-003](../../tasks/done/v62-003-audit-preparation-controller.yml)).

### 34 — Contextual Audit checklist prioritization › 5. Verdict and comparison rubric › Pure core boundary

Verification: the pure core is recorded in
[V64-000](../../tasks/done/v64-000-audit-priority-core.yml).

### 34 — Contextual Audit checklist prioritization › 10. Required verification

The implementation gate uses scripted model responses, disposable PostgreSQL,
production Server/Scheduler boundaries and controlled check Workers. No paid
model or external target is required. Mandatory cases include:

- Counts 0/3/9/10/11/100; `topN=9` rejection, `topN=12`, and all-if-fewer.
- 100 inventory items with maxItemsPerRound 10; full denominator and exactly ten
  accepted items, with no deferred item budget consumption.
- Identical scores, stable tie order, high-priority position 11 retaining `high`,
  absent optional context, and advisory non-applicability retaining membership.
- One malformed/foreign-ID/tool-call/truncated verdict prevents ranked selection;
  no-context alone does not lower priority and no failure becomes a fabricated score. Oversized late-candidate input or
  insufficient known whole-pool token budget gives zero model calls.
- Check capacity three with desired ten gives explicit blocked selection, no
  three-item Round; ranking overhead and retry/batch reservations are accounted. Eleven submission
  slots permit one ranking Run plus ten one-item checks, without double charging
  on acceptance/replay or loss of reservations to another retry.
- Selected high-priority item awaiting approval; no substitution or weakened gate.
- Restart at intent/send/result/selection boundaries, lost commit acknowledgements,
  two Controllers/claims, whole-Run retry attempts and actual provider call counts.
  Reject tampered aggregate output, wrong Run/cycle/source revisions and output
  collection/deletion races before selection.
- Pause/cancel/deadline/delete before Round1 and between rounds; Audit admission
  expiry drains ranking then resume reuses its receipt, while Run/Stage timeout
  yields incomplete ranking; unknown usage,
  ownership loss, source Run deletion, stale pagination and context review races.
- Ranking without any Runtime connected still respects global queue/concurrency;
  existing Worker-backed Stages retain allocation-loss detection.
- Two passes of 10 from 25 leave five not-tested after maxRounds=2, use refreshed
  context, keep prior verdicts, and work with finding confirmation disabled.
- Browser upload/configuration/ranking/progress/selection/deferred explanation/
  incomplete recovery and full-inventory report against production public API.
- Current-schema non-priority profiles retain execution/review behavior and
  deterministic canonicalization; obsolete profile fields and snapshots are
  rejected. Current public clients and storage upgrade/recovery fixtures pass
  without compatibility readers. Tests explicitly distinguish planning mechanics
  from model quality.

The release task must register discoverable mandatory process/browser cases,
fail on missing prerequisites/zero selected tests/skips, and retain actual
commands, revisions and results. Creating this document or task files is not
implementation or test evidence.

## From `README.md`

### Contractor v2 — working specifications

Status: **Working agreement**

### Contractor v2 — working specifications › Reading order

| Document | Owns |
|---|---|
| [00](../spec/configuration/workflows.md) | Workflow Scheduler, Stage, Planner and StageResult |
| [01](../spec/configuration/agent-templates.md) | Reusable Worker behavior and Stage binding |
| [02](../spec/execution/control-plane.md) | Allocation and the single-slot Runtime Agent acting as Worker/A2A Server |
| [03](../spec/artifacts/artifact-store.md) | ArtifactStore scopes, RunArtifactSpace, input/output forks, Namespace and CAS |
| [04](../spec/execution/scheduler.md) | StageExecution identity, StageTermination, sessions, finalization, recovery and metrics |
| [05](spec-history.md) | First implementation slice and deliberately deferred decisions |
| [06](../spec/ui/ui.md) | Separate Node.js Web UI, Operations visibility, process budgets and published execution configuration selection |
| [07](../spec/configuration/runtime-config.md) | Run-selected/Agent Runtime labels, database-backed infrastructure configs, adapter placement and allocation-scoped settings |
| [08](../spec/tools/memory.md) | Run-scoped shared Memory Namespace and the artifact-backed `memory-tools@1` contract |
| [09](../spec/tools/skills.md) | AgentTemplate-selected, artifact-pinned Agent Skills loaded through native Google ADK |
| [10](../spec/tools/workspace.md) | Run-artifact workspaces, Runtime local/memory storage, overlay export and filesystem/Edit tools |
| [11](../spec/tools/http-caido.md) | Allocation-scoped HTTP exploration and label-configured Caido GraphQL tools |
| [12](../spec/tools/code-analysis.md) | Workspace Tree-sitter analysis and local-only allocation-scoped Trailmark graph tools |
| [13](../spec/tools/code-analysis.md) | Structured, atomic `@trace`/`@validate`/`@sink` workspace annotations |
| [14](../spec/execution/worker.md) | Typed Worker results, deterministic observations, volatile Worker State and explicit Planner projections |
| [15](../spec/execution/worker.md) | Optional one-shot terminal Worker summarization at a deterministic soft limit |
| [16](../spec/execution/runs.md) | Immutable queryable WorkflowRun metadata labels and the eval correlation convention |
| [17](../spec/artifacts/projects.md) | Optional Projects, ProjectScope artifact reuse, Workflow recommendations, Evals and the global Queue |
| [18](../spec/artifacts/retention-and-deletion.md) | Consolidated Runs UI, durable queue pause/resume and safe Run/Project deletion |
| [19](../spec/audits/audits.md) | Project-bound multi-Run Audits, deterministic inventories, findings, review, coverage and recovery |
| [20](../spec/execution/scheduler.md) | Durable Operations-controlled concurrency across ordinary WorkflowRuns and Audit dispatch backpressure |
| [21](../spec/tools/code-execution.md) | Implemented: opt-in local Podman execution, direct workspace sharing and verified container cleanup |
| [22](../spec/operations/operations.md) | Implemented: Server/DB performance charts, completed allocation resources, retained history and independent Go profiling |
| [23](../spec/artifacts/blob-storage.md) | Startup-selected PostgreSQL/filesystem blobs, Kubernetes without PVC, bounded memory and deferred S3 |
| [24](../spec/artifacts/git-import.md) | Owner SSH-key Settings, bounded in-memory Git snapshots, Workflow/Project import UI and immutable commit provenance |
| [25](../spec/audits/worker-completion.md) | Implemented opt-in: collection, Runtime completion, ZIP publication and diagnostics; [V39-007](../../tasks/done/v39-007-audit-completion-release-gate.yml) records the completed deterministic release gate |
| [26](../spec/evals/format.md) | Implemented in playground-v2 and offline-gated: portable documents, execution bindings, v1 compatibility, recovery and comparison; model quality evals remain |
| [27](../spec/findings/findings.md) | Implemented: findings collection codec, Server publication, Runtime materialization, independent finding/list_findings tools and producer/analyst Workflows |
| [28](../proposals/0028-finding-analysis-and-sarif.md) | Draft; not implemented: structured finding analysis, exact-evidence resolution and deterministic SARIF export |
| [29](../spec/tools/scanners.md) | Implemented: deterministic model-free tool Workers, typed execution bindings and durable invocation receipts |
| [30](../spec/evals/managed-evals.md) | Implemented: native Evals execution, assessments, comparison APIs, browser UI and optional independent Playground client |
| [31](../spec/tools/scanners.md) | Implemented: bounded deterministic OpenAPI request preparation, RequestSet identities and coverage gaps |
| [32](../spec/tools/scanners.md) | Implemented: bounded model-free scan plans, fixed Worker routing, durable job intent and conservative recovery |
| [33](../proposals/0033-autonomous-pentest-audits.md) | Draft; not implemented: source-optional web/API pentest Audits, enforced scope, isolated identities, live proof and replay, recovery and release gates |
| [34](../proposals/0034-audit-check-prioritization.md) | Draft integration target; V64-000 pure verdict/selection core implemented, Audit capability pending |
| [Database ERD](../spec/database-erd.html) | Interactive PostgreSQL schema at migration 000096: 71 tables, keys, each table's purpose and the processes that use it, and where the tables removed by V365 went; regenerate with `make docs-erd` ([generator](../../scripts/erd/README.md)) |
| [LikeC4](../spec/artitecture.likec4) | Component map and focused architecture views |
| [OpenAPI Audit scans](../spec/audits/scans.md) | Released: assigned SQLMap requests and pinned Nuclei URLs; prepare-role OpenAPI generation specified but not delivered |
| [UI user stories](../spec/ui/journeys.md) | User goals and acceptance scenarios; individual stories retain their implementation status |

### Contractor v2 — working specifications › Specification rule

Each decision has one owning document. Other documents link to it instead of
repeating a second normative version. Unresolved behavior is recorded
explicitly rather than inferred from historical designs: cross-cutting
decisions stay in [05](spec-history.md), and a document
may keep its own deferred section for decisions inside its scope; 05 links
those sections. Implementation status lives only in a document's `Status:`
line, in the reading-order table above and in task files; normative body text
describes the accepted target without task identifiers, dates or evidence.

## From `openapi-audit-scans.md`

### OpenAPI Audit scan adapter

Status: **Supplied-input scans released and verified ([V62-009](../../tasks/done/v62-009-audit-openapi-scan-profile.yml)). Preparation execution and generated initial inventory are implemented by V62-003–004 under [19 §4.4](../spec/audits/scans.md); the generated-flow process/browser acceptance gate remains V62-012.**

### OpenAPI Audit scan adapter › Assigned inputs and authority › Settings and inventory contract

The checked-in [local input examples](../../configs/scan/examples/audit-openapi-scan/README.md)
cover path/query values and an authenticated POST. Library and executor tests
verify preparation and canonical results without scanner/network calls. The
registered profiles must also pass the mandatory production-process gate below.

### OpenAPI Audit scan adapter › Required release evidence

`make test-openapi-audit-scan-e2e` requires all seven cases of
`TestOpenAPIAuditScanAcrossProductionProcesses` to execute and pass; missing or
skipped cases fail. Real Server/Scheduler/Runtime/public API tests with local
controlled SQLMap and Nuclei fixtures prove exact dispatch, accepted
Audit packages, distinct scan coverage, bounded retries and recovery without
implicit redispatch of completed or unknown executions across Run/Stage attempts.
The matrix includes missing path/body/auth values, excluded SQLMap parameters,
Nuclei POST-derived URLs, credential rejection, unknown outcomes, cancelled
execution, lost acknowledgements and retained evidence after source Run deletion.
Scripted tests do not establish model-generated OpenAPI quality or scanner recall.
The recorded gate result is [`tasks/evidence/v62-009.json`](../../tasks/evidence/v62-009.json).

## From `ui-user-stories.md`

### Contractor UI: user stories and improvement plan

Status: **Implemented for US-01 through US-11; separate follow-up scope
remains** (verified by [V37-012](../../tasks/done/v37-012-ui-journey-verification.yml))

### Contractor UI: user stories and improvement plan › Implementation order

Verification: the task files below record each story's implementation,
dependencies, acceptance criteria and verification commands. New follow-up
tasks start with `status: pending` and `priority: P2` under the repository
policy of "after the first releasable slice".

| Wave | Task | Outcome | Rationale |
| --- | --- | --- | --- |
| 1 | [V37-001](../../tasks/done/v37-001-accessible-dialogs.yml) | Shared Dialog, focus management and nested forms | UX-05 |
| 1 | [V37-002](../../tasks/done/v37-002-run-draft-continuity.yml) | Drafts within the current tab, local upload to an input slot and Git integration | UX-02 |
| 1 | [V37-003](../../tasks/done/v37-003-project-workflow-primary-actions.yml) | Add-materials and launch actions visible at the top of the page | UX-01 |
| 1 | [V37-004](../../tasks/done/v37-004-input-suggestion-review.yml) | Explicit review of inputs suggested by format | UX-03 |
| 1 | [V37-005](../../tasks/done/v37-005-audit-action-confirmations.yml) | Cancel/Delete Audit confirmations | UX-04 |
| 2 | [V37-006](../../tasks/done/v37-006-catalog-discovery-contracts.yml) | Metadata and bounded search, version and usage-link APIs | UX-06 |
| 2 | [V37-007](../../tasks/done/v37-007-catalog-discovery-ui.yml) | Workflow selection by purpose, agent version selection and Where used | UX-06 |
| 2 | [V37-008](../../tasks/done/v37-008-run-repeat-and-next-actions.yml) | Prefilled repeat and next steps for errors/waits | UX-07 |
| 2 | [V37-009](../../tasks/done/v37-009-primary-result-preview.yml) | Primary results and preview in one action | UX-08 |
| 3 | [V37-010](../../tasks/done/v37-010-audit-review-workspace.yml) | Audit summary and decision work queue | UX-09 |
| 3 | [V37-011](../../tasks/done/v37-011-operations-progressive-forms.yml) | Operations overview, forms opened by action and the impact of changes | UX-10 |
| V37 acceptance | [V37-012](../../tasks/done/v37-012-ui-journey-verification.yml) | Verification of connected journeys, keyboard access and mobile viewport | US-01…09, US-11 within V37 |
| Separate design | [V38-001](../../tasks/done/v38-001-evals-experience-contract.yml) | Selected native experiment and independent producer contract | UX-11 / US-10 |
| V38 acceptance | [V38-010](../../tasks/done/v38-010-eval-release-gate.yml) | Full Evals setup, launch, review, comparison and release verification | UX-11 / US-10 |
| Project sections | [Project section plan](../plans/2026-09-project-section-navigation.md) | Project Overview and independent section screens | US-01 |

Git Settings/import ([V35-004](../../tasks/done/v35-004-git-settings-and-import-ui.yml))
and the Performance UI ([V32-006](../../tasks/done/v32-006-operations-performance-ui.yml))
are owned by their own specifications; the stories reuse them without
duplication. Journey verification does not establish performance at production
inventory sizes.

### Contractor UI: user stories and improvement plan › Verification

Verification: [V37-012](../../tasks/done/v37-012-ui-journey-verification.yml) with
its [acceptance report](../plans/2026-09-19-v37-journey-verification.md), and
[V38-010](../../tasks/done/v38-010-eval-release-gate.yml) with its
[acceptance record](../plans/2026-09-20-managed-evals-ui.md), record connected
browser journeys, real Server/Runtime/PostgreSQL and API checks, restart and
response-loss recovery, owner isolation and desktop/mobile viewports. Fixture
evidence is separate from real API evidence. These results verify behavior,
not model quality or human task-success rates; no participant usability study
has been performed.

The repeatable acceptance approach is to verify a connected journey rather than
only an individual screen:
choose a Workflow → missing file → fill in the form → navigate away and return →
launch; error → new draft; completed Run → primary result; waiting_review →
evidence → decision; dismiss Audit confirmation without mutation.

Audit checks also cover artifact pagination and ambiguous matches, starting
without a time limit, preserving time through a pause, and continuing a legacy
deadline closure without changing accepted results. Configuration checks cover
keyboard opening/closing of dialogs, exact-version binding selection, Caido
read projections and clearing an unsaved credential on close.

Acceptance covers keyboard access, 1440×1000 and 390×844 viewports, real data
constraints, errors, empty states and recovery after an ambiguous response.
Synthetic browser fixtures verify reproducible interactions; integration checks
verify API, ownership and exact-revision boundaries.

To assess usability, have a new user and an experienced user perform the same
tasks before and after the changes. Measure success without hints, time spent,
backtracking, incorrect input choices and lost input. Set numerical targets
after the baseline measurement; expert review does not replace user testing.

## Origins in contractor-old

The legacy specifications named the predecessor implementation where a
contract was ported from it. The reference text keeps the rules and drops the
lineage; the lineage is kept here.

- `12-code-analysis-tools.md`: the first read-only structural code-analysis
  Toolset for a hydrated allocation workspace ports the useful Tree-sitter and
  Trailmark behavior from `contractor-old` while preserving the v2 Runtime
  boundary. It supports the retained v1 language names from `contractor-old`.
- `13-taint-annotations.md`: the taint annotation Toolset ports the useful
  `contractor-old` annotation operations while retaining the v2 ownership model.
- `09-agent-skills.md`: the initial contract preserves the useful content shape
  from `contractor-old` without preserving its implementation. The old system
  copied Markdown into reserved Memory notes and exposed `skills_list` and
  `skills_read`.
