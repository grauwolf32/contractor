# 05 — First slice and open decisions

Status: **Working agreement**

Depends on: [00](00-workflow-and-planner.md) through
[04](04-execution-lifecycle-and-metrics.md),
[06](06-server-ui-and-operations.md) and
[07](07-runtime-labels-and-infrastructure-config.md), plus the later
shared-memory increment in [08](08-memory-tools.md) and Agent Skills increment
in [09](09-agent-skills.md), and the Runtime filesystem increment in
[10](10-runtime-filesystems-and-edit-tools.md)

## First implementation slice

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

Server, PostgreSQL and Runtime Agent may run on one VM. Runtime Agent and its
current Worker runtime are one process; tool execution may still use a separate
sandbox where a tool requires it.

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
- every AgentTemplate has a mandatory configuration-root-relative Worker
  `instructions.ref`; its resolved text contributes to the template digest and
  reaches Runtime Agent inside AllocationSpec rather than through
  configuration-file I/O;
- every AgentTemplate selects the exact registered SandboxProfile
  `local-workdir@1`; it creates a fresh allocation directory, exposes no host
  path through declarative contracts, cleans it before the slot returns to
  `idle`, and explicitly promises no OS, process or network isolation;
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
  overridden by the pinned Run/Agent-label boundary in [07]. Control Plane
  supplies the resulting allocation URL and optional token in RuntimeSettings
  over mTLS, and Agent clears secrets on release;
- Workflow Scheduler records `preparing -> running -> finalizing -> terminal`;
- WorkflowRun records `initializing -> running -> succeeded`, with the final
  Stage acceptance, required output bindings and Run success in one transaction;
- cancellation or participant loss records
  `preparing/running -> aborting -> cancelled/interrupted`, with a durable
  StageTermination and bounded abort deadline;
- Run cancellation records `initializing/running -> cancelling -> cancelled`,
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
- `streamline@1` and `router@1` pin Google ADK Go v1.6.0 behind the existing
  Planner interface and share the bounded subtask-plan and validated
  `finish(StageResult)` operation;
- Streamline's dispatch function is exactly
  `execute_current_subtask(subtask_id)`, while Router's is exactly
  `execute_current_subtask(subtask_id, worker_name)` with `worker_name`
  constrained to the immutable Stage binding keys; explicitly selected
  MemoryTools may add only the surface derived by [08](08-memory-tools.md);
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
  [14](14-worker-results-and-live-state.md);
- Runtime Agent finalizes and destroys its in-process Worker instance through
  its private control endpoint and returns bounded reports before terminal
  acceptance;
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
- Runtime Agent sends a monotonic heartbeat every 10 seconds and echoes the
  last received ack; both sides advance the confirmed control lease only
  through that round trip and expire it after 60 seconds;
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

## Label-driven infrastructure increment

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
- a `debug` Run emits content-free bounded Worker telemetry to its pinned OTLP
  endpoint, and optional Planner telemetry uses the same pinned Run config on
  Server;
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

## Shared-memory increment

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

## Agent Skills increment

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

## Runtime filesystem increment

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

[10](10-runtime-filesystems-and-edit-tools.md) owns the complete contract.

## HTTP and Caido tools increment

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

[11](11-http-and-caido-tools.md) owns the complete contract.

## Workspace code-analysis increment

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

[12](12-code-analysis-tools.md) owns the complete contract.

## Structured taint-annotation increment

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

[13](13-taint-annotations.md) owns the complete contract.

## Deliberately deferred

The following decisions remain open; no legacy document defines them
implicitly:

- remaining Workflow YAML details outside the transition, Agent-binding,
  artifact-slot and output-mapping contracts;
- exact count, name-length, value-length and total-size limits for string Run
  parameters;
- exact maximum size for a resolved UTF-8 instruction resource;
- Planner authority for generic artifacts and domain Toolsets other than the
  explicitly specified MemoryTools mirror;
- optional incremental Worker metric delivery for retaining detail across a
  hard process crash;
- S3 blob backend, streaming uploads, and longer-term Artifact retention;
- explicit Run-output publication endpoint;
- concrete graceful-drain timeout before forced Worker termination after lease
  loss;
- shared Runtime Agent Registry and coordination for multiple active Control
  Plane replicas;
- Contractor credential-encryption master-key rotation/re-encryption and
  external Vault/KMS adapters;
- LiteLLM virtual-key TTL/automatic rotation and policy fields beyond the
  explicit first-slice model, spend/reset, TPM, RPM and concurrency allowlist;
- concrete CA bootstrap, certificate delivery, lifetime, rotation and
  revocation procedures;
- multi-tenant authorization and quota policy.

Each decision should be added to its owning spec only when a concrete first-
slice implementation needs it.

## Explicit non-goals for the first slice

- Kubernetes CRDs/controllers or autoscaling;
- multi-host placement optimization;
- dynamic Planner expansion of the Worker pool;
- concurrent StageExecutions within one WorkflowRun;
- Planner-selected AgentTemplates;
- concurrent Tasks inside one Worker allocation;
- resuming a Planner invocation from its persisted ADK Session or private plan;
- artifact change subscriptions or semantic merge service;
- dynamic or concurrent Streamline/Router Worker expansion;
- cross-Run/user-scoped memory, automatic note injection, semantic retrieval,
  deletion or automatic eviction.
- Agent Skill script execution, remote registry search, Run/Stage/Planner skill
  selection, multi-owner shared skills or hot replacement inside an active Run.
- Server-managed project paths, operator-checkout mutation or automatic
  synchronization between allocation workspaces.

## Exit question

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
