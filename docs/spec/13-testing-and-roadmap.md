# 13 — Testing, delivery roadmap and v1 migration

Status: **Draft**  
Depends on: all subsystem specifications

## Required test layers

- domain unit/property tests for state machines, DAG and policies;
- JSON/OpenAPI contract fixtures for every public DTO and API;
- ADK capability tests pinned to the locked dependency version;
- repository integration tests against real PostgreSQL;
- A2A integration tests with real Server and Agent adapters;
- one framework-neutral Worker-strategy invariant suite applied to every Worker
  implementation, plus adapter-specific cases;
- one shared Planner-strategy invariant suite applied to static, passthrough and
  decomposing strategies, plus strategy-specific cases;
- scheduled/release, workflow-specific quality evaluations for substitutable
  strategy configurations;
- one deterministic CI fixture for the evaluation manifest/report schema;
- sandbox isolation/cancellation tests;
- end-to-end and fault-injection scenarios;
- load tests for connection, telemetry and concurrency budgets.

Mocks can isolate domain tests. Acceptance tests use real PostgreSQL and the
real adapters under test; a fake LLM Proxy is acceptable for deterministic
model responses.

## Requirements

- **TST-001** — Acceptance tests MUST use real PostgreSQL and the real adapter
  under test; domain tests MAY use fakes.
- **TST-002** — The release candidate MUST automate every global end-to-end
  acceptance scenario below.
- **TST-003** — CI MUST maintain a machine-readable mapping from every
  requirement ID to one or more tests.
- **TST-004** — A phase MUST NOT close until its required compatibility,
  failure and cancellation tests pass.
- **TST-005** — Fleet-control contract tests MUST cover mTLS identity,
  registration idempotency/reordered generations, heartbeat replay, lease
  expiry, Server epoch change and drain before multi-Agent routing is enabled.
- **TST-006** — Every Worker implementation MUST pass the shared start,
  duplicate-delivery, result, cancellation, deadline, restart/reconciliation,
  artifact, usage and fencing invariant suite. Framework- or adapter-specific
  behavior MUST have additional tests rather than weakening the common suite.
- **TST-007** — CI MUST run a core profile with a minimal non-ADK test Worker and
  prove that domain, A2A profile and `AgentRuntime` do not import ADK.
- **TST-008** — Static, passthrough and decomposing Planner strategies MUST pass
  the shared typed-command, RunState/CAS, outbox, retry, cancellation and
  recovery invariant suite. Strategy-specific Planner-session, model-budget and
  manifest tests remain separate.
- **TST-009** — Passthrough MUST propose exactly one root `TaskSpec`. The
  scheduler creates one immutable `WorkerJob` per Attempt; transport redelivery
  reuses that exact job, while a policy retry creates a new Attempt and
  `WorkerJob` for the same Task and `TaskSpec`, never a second root Task.
- **TST-010** — A Worker MAY plan, work or maintain private subtasks internally.
  If it does, all internal activity MUST remain within one Attempt and share its
  grants, reserved budget, deadline, cancellation and aggregate usage.
- **TST-011** — Architecture tests MUST prove Worker strategies cannot import or
  receive `WorkerRegistry`, `WorkerGateway`, Server Attempt repositories or
  fleet-control clients.
- **TST-012** — Every scheduled/release comparison MUST be driven by an
  immutable, versioned evaluation manifest. Each candidate pins its exact
  `PlannerStrategyRef`, case-specific accepted `RunSpec`, Worker routing,
  capability and implementation constraints, budget/deadline policy and all
  relevant prompt, model, tool and sandbox policy digests. Each case pins its ID
  and exact input `ArtifactRef` values. The manifest also pins the scorer name,
  version and configuration digest, evaluation-tool version and configuration
  digest,
  repetition count and order, the seed assigned to each repetition (or an
  explicit unsupported marker), and requested model-resolution policy.
- **TST-013** — Evaluation tooling MUST derive stable child-run submission and
  evaluator idempotency keys from the manifest digest, candidate, case and
  repetition. It MUST derive the final-report publication key from the manifest
  digest, report-schema version and evaluation-tool configuration digest. A final
  report MUST be published only after every referenced child Run is terminal and
  accounting-final under `TST-020`. Repeating publication with the same key and
  canonical bytes MUST return the existing report `ArtifactRef`; different bytes
  under that key MUST fail closed. The report MUST reference the exact manifest
  digest, every child Run ID and every evaluator artifact. Scores MUST come from
  accepted output and evaluator artifacts; reliability, timing and usage MUST
  come from authoritative lifecycle and ledger records. Sampled or dropped
  telemetry MUST NOT change the report.
- **TST-014** — A stochastic report MUST retain every raw repetition, including
  failures, resolved model identity and per-run quality/usage/timing values. It
  MUST report distributions and paired case deltas and MUST NOT collapse the
  comparison to pass@N or select a passing repetition as representative. A
  single run MUST NOT be presented as a general winner.
- **TST-015** — A report MUST include output-contract validity, the declared
  workflow quality measures, accepted-output rate, wall-clock duration, model
  invocation/token totals, tool-call totals and Task/Attempt/retry counts.
  Monetary cost MAY be reported only when the manifest pins the price source,
  version, currency and effective date; otherwise cost is `unknown` rather than
  inferred from a mutable price table. The operational section MUST also report
  database write-amplification and load measurements described below, using
  explicit `unknown` or `not_applicable` values when a measurement does not
  exist for a candidate.
- **TST-016** — Normal CI MUST validate manifest/report schemas, aggregation,
  idempotent child-run/evaluator/report recovery and finalization with
  deterministic fixtures and a fake LLM Proxy. Fixtures MUST cover a crash
  before and after report publication and every accounting disposition in
  `TST-020`. Stochastic product-quality comparisons run in a scheduled or release
  evaluation lane and are not global end-to-end acceptance scenarios.
- **TST-017** — CI MUST validate a versioned v1 workflow migration ledger with
  exactly one entry for every canonical public v1 workflow key. A workflow MUST
  NOT be reported as migrated until its entry is `validated` and references its
  passing behavioral fixture, output-quality evidence, compatibility gaps,
  enabled versioned workflow-catalog fixture and authenticated submission/
  authorization fixture.
- **TST-018** — Exploit/live-target workflow fixtures MUST use an authorized
  sandbox, replay server or disposable target. A migration MUST separately test
  scope authorization, egress denial and non-idempotent/unconfirmed tool-effect
  recovery; automated quality evaluation MUST NOT repeat live external effects.
  Automatic Agent-loss or policy retry for an effect-capable `TaskSpec` MUST NOT
  be enabled until durable intent/outcome/reconciliation evidence proves replay
  safe; absent or ambiguous evidence yields `external_effect_unconfirmed`.
- **TST-019** — Before execution, a comparison manifest MUST declare its
  primary quality measure, minimum case/repetition coverage, reliability and
  resource guardrails, equivalence/non-inferiority margins and decision rule.
  The report MUST show paired deltas with uncertainty and conclude only
  `preferred(candidate_id)`, `non_inferior(candidate_id, baseline_id)`,
  `equivalent-within-margin` or `inconclusive`. `non_inferior` requires a
  declared one-sided margin; `equivalent-within-margin` requires its declared
  two-sided equivalence test. The report MUST NOT choose a strategy by an
  undeclared post-hoc metric or automatically change production routing.
- **TST-020** — A terminal child Run is accounting-final only when every Planner
  invocation and Attempt reservation has one authoritative disposition:
  settled from accepted usage, conservatively charged to the reserved upper
  bound or stricter authoritative evidence with ambiguity/unknowns retained, or
  released/zero from durable evidence that no metered model/tool/execution work
  began under it. This rule covers `succeeded`, `failed` and `cancelled` Runs;
  `succeeded`, `failed`, `cancelled` and `lost` Attempts; Planner failure; and a
  Run that terminates before its first Attempt. A report MUST remain pending or
  fail its evaluation lane on an unresolved reservation; it MUST NOT infer zero
  from terminal status, missing telemetry or missing ADK session detail.
- **TST-021** — The first planning-placement study MUST include (a) the retained
  Server decomposing Planner dispatching narrow Tasks and (b) passthrough
  dispatching one root Task to a Worker that plans/works internally (for example
  a ReAct strategy). Where the workflow has a stable known graph, it SHOULD also
  include the static manifest baseline. Where one root objective can also be
  executed directly, it SHOULD include passthrough plus that direct Worker as an
  ablation of Agent-local planning. Cases, output scorer, allowed model/tool/
  sandbox policy and total budget class MUST be paired; unavoidable Task-
  granularity/placement differences are recorded and the conclusion is labeled
  an end-to-end coordination-strategy comparison, not a Planner-only causal claim.

## Scheduled/release strategy evaluation

Strategy evaluation is release/test tooling, not a runtime orchestration
protocol. The tooling owns versioned manifest and report artifact schemas; they
are not public execution DTOs, are not embedded in `PlannerRunState`, and do not
authorize planning, dispatch or result acceptance.

The unit of comparison is a complete candidate configuration. A comparison may
be called Planner-controlled only when the `RunSpec`, Worker constraints,
resolved model policy, tools, budgets and every other material dimension are
held fixed. When planning placement, `TaskSpec` granularity or Worker behavior
also changes, the report labels the result an end-to-end configuration
comparison and attributes no delta solely to the Planner.

The initial comparison matrix is deliberately small and useful: retained
Server decomposition plus narrow Workers; one-root passthrough plus an
Agent-local plan/work loop; and, for workflows with a known stable graph, a
deterministic static manifest. A non-ADK Worker is another implementation of a
candidate, not another evaluation protocol. This matrix answers both “where
should planning live?” and “does dynamic planning beat the static baseline?”
while keeping every wire/runtime invariant fixed. When the root contract also
has a direct implementation, passthrough plus that direct Worker is the preferred
ablation for measuring the incremental value of the Agent-local planning loop.

The manifest identifies candidates, paired cases and exact initial artifact
versions. For every candidate/case/repetition cell it pins or derives the exact
accepted `RunSpec` and stable submission idempotency key. The report records the
requested model alias and the Proxy-resolved provider/model identity for every
model invocation; a seed is recorded only when the provider supports it, and
unsupported seeding is explicit. Automated cases use sandboxed or replayable
inputs and MUST NOT repeat unsafe live side effects.

The manifest also predeclares what “better” means for that workflow: one primary
quality measure, coverage/sample minimums, non-inferiority or equivalence
margins, and reliability/resource guardrails. Reports show paired deltas and
uncertainty and may remain inconclusive. A one-sided non-inferiority conclusion
is not reported as two-sided equivalence. These reports are release evidence;
they never silently mutate the Planner profile used by production submissions.

Evaluators validate declared output contracts before scoring product quality.
The following are reusable candidate measure families. Each manifest selects
exactly one primary measure and identifies any remaining measures as secondary
quality or diagnostic evidence:

| Workflow or output | Candidate quality measures |
|---|---|
| `oas_build`, `oas_update` / OpenAPI | endpoint precision, recall and F1; schema recall; validator success |
| `trace`, `trace-direct`, `trace-graph`, `trace-graph-pathpar` | annotation precision, recall and F1; required finding-artifact validity |
| `trace-postdiff` | vulnerability precision, recall and F1; overlay/diff and finding-artifact validity |
| `trace-verify` | verification-verdict correctness; required evidence presence; authoritative verification-artifact validity/freshness |
| vulnerability discovery | TP/FP/FN, precision, recall, F1 and F2 |
| exploitability | verdict correctness and required evidence presence |
| LikeC4 / threat analysis | parser/validator success, required-structure and coverage measures |
| `router` | specialist-routing accuracy and final objective/output-contract acceptance |

Planner subtask count, depth, skip rate and Planner-tool calls are diagnostics
for the decomposing strategy, not cross-strategy product-quality measures.
Attempts and retries are reported as total work per case, not treated as quality
by themselves. All raw repetitions remain in the report; aggregate success,
quality, latency and resource distributions are derived from them only after
terminal state and `TST-020` accounting finalization.

Evaluation accounting is finalization evidence, not another runtime owner. For
each terminal child Run, tooling reads the authoritative Planner-invocation,
Attempt and budget ledgers and records every reservation disposition required by
`TST-020`. A pre-dispatch failure records an evidence-backed zero/release, while
lost or ambiguous model/Worker activity is conservatively charged and retains
unknown fields. A terminal lifecycle state alone is insufficient. The canonical
report is published under its stable idempotency key only after this check; a
publication replay returns the same report ref rather than creating another
“final” result.

The operational section also compares database write amplification and load:
RunState version rows/bytes and CAS latency, ADK session-event rows/bytes when
ADK sessions are enabled, Contractor telemetry rollup write rows/bytes/batches,
peak pool/connection use and representative query-latency distributions under
the same workload. These values are diagnostic and non-authoritative. Collectors
SHOULD use database statistics and bounded counters/rollups; retaining or
replaying raw ADK/telemetry events is not required evidence and cannot change
quality, usage settlement or Run acceptance.

## V1 workflow migration ledger

The repository owns a versioned migration-ledger artifact. It is release
evidence, not runtime state. It contains exactly one entry for each canonical v1
public workflow key:

```text
oas_build              oas_update              exploit
likec4                 trace                   trace-direct
trace-graph             trace-graph-pathpar     trace-postdiff
trace-verify            vuln-assess             vuln-scan
vuln-scan-fast          vuln-scan-trace         vuln-sweep
router
```

Each entry records:

- the public key, immutable v1 repository revision and build/package digest,
  exact source package/class, workflow-configuration digest, resolved
  environment overrides and task-template/objective-contract versions;
- source objective, input-artifact and output-artifact schemas/contracts, using
  exact refs/digests when typed and an explicit `untyped` gap plus migration
  issue when v1 has no schema;
- task graph and ordering, conditional artifact reuse/checkpoint behavior,
  fan-out/concurrency/merge behavior and failure-isolation unit;
- direct, planner-driven or static source execution behavior and the candidate
  v2 `PlannerStrategyRef` (`static`, its `passthrough` specialization, or
  `decomposing`) plus its implementation/configuration digests and intended
  `TaskSpec` graph mapping;
- the exact v2 workflow/execution/catalog profile version and digest, RunSpec
  contract, Worker capability/implementation constraints and resolved tool,
  prompt, requested/resolved model and sandbox-policy versions/digests,
  separately from the corresponding resolved v1 values;
- for any external effect, the source behavior and candidate v2 effect class,
  target/environment, operation, egress, credential and time authorization,
  stable operation-key scope and retry/reconciliation rule;
- the acceptance fixture, scorer version/configuration and thresholds, exact v1
  baseline evidence and latest v2 comparison report when one is required;
- status (`not_assessed`, `planned`, `in_progress`, `validated`, `deferred` or
  `retired`), rationale, unresolved compatibility gaps and evidence refs.

`validated` means the versioned workflow-catalog fixture exposes the same public
key in an enabled profile and an authenticated submission with exact authorized
inputs resolves the recorded RunSpec and PlannerStrategyRef. Anonymous,
wrong-scope and wrong-contract submissions must fail before RunState creation.
An implementation absent from the tested enabled catalog remains `in_progress`
or is deliberately `deferred`; it is not `validated` merely because an internal
Worker fixture passes.

The `oas_build` entry identifies its static four-stage manifest and exact-ref
reuse rules. The `trace` entry records its planner-driven per-path behavior.
`trace-direct` and `trace-graph` each map the complete public workflow to one
passthrough root Task; their Worker deterministically processes operations in
order over one shared overlay and path-scoped memory, differing by the exact
graph-tool policy. `trace-graph-pathpar` also maps to one passthrough root Task;
inside its boundary Attempt the Worker creates bounded private route-group
workspace views, preserves per-group failure isolation and performs a versioned
deterministic merge before returning the final patch/diff and finding refs.

This root-Worker mapping is deliberate. The legacy overlay, ordered namespace
memory and fork/merge state cross operation boundaries, while the current
`StaticPlanManifest` contract has no collection-expansion primitive for deriving
an operation-count-dependent Server DAG. Splitting operations into Server Tasks
would require a separately specified deterministic expansion, exact overlay
checkpoint and merge contract. An operation-level Worker fixture remains useful
for controlled quality evaluation, but it is not a public workflow Run and
cannot by itself validate a migration-ledger entry. A `retired` entry names its
supported replacement and cutover evidence; it is not silently counted as
migrated.

## Delivery phases

### Phase 0 — Repository foundation

- package skeleton with `domain`, `application`, `adapters` and process apps;
- lint, type-check, test and architecture dependency checks;
- core dependency lock with the A2A and database bindings, plus a separately
  locked Google ADK `2.7.1` adapter profile;
- `PlannerStrategy` and framework-neutral `WorkerStrategy` ports plus a no-ADK
  import/conformance profile;
- DTO fixtures and generated Control Plane API skeleton.

Exit: `SCP`, `ARC` and `CON` core tests pass without ADK; enabling the ADK
profile also passes its `ADK` capability tests.

### Phase 1 — Durable foundation

- shared `AsyncEngine` lifecycle;
- migrations and run projection repository;
- Server-owned Attempt operation-start gates, per-generation Agent persistence
  capability guards and scoped continuation/recovery-grant schema;
- chunked `PgArtifactService`, RLS, staged promotion and RunState CAS;
- authenticated principal/tenant mapping, append-only audit storage and
  fail-closed run/artifact authorization primitives;
- transactional dispatch outbox and crash-window recovery;
- authoritative lifecycle queries plus bounded telemetry rollup/sample
  projection, without row-for-row ADK event duplication.

Exit: concurrent CAS, restart and connection-budget tests pass.

### Phase 2 — First vertical slice

- minimal Control Plane submit/get/cancel;
- authenticated, idempotent blob/project-snapshot publication and exact input
  selection; no Server-local project path;
- one deterministic one-node static plan (the passthrough specialization) that
  proposes one root `TaskSpec` through the shared Planner state machine;
- private HTTPS/mTLS registry API with one Agent capability and durable lease;
- persistence-capability rotation plus new-nonce replacement cleanup-only tests;
  the first Worker declares `non_resumable`, so Phase 2 does not implement
  same-incarnation business-execution continuation;
- durable Server/Agent A2A mappings, restartable Attempt I/O, committed
  context-based recovery and Contractor execute/stream/cancel wrapper;
- co-hosted mTLS Agent Attempt-control endpoint for pre-task-ID cancellation;
- one real `non_resumable` Worker implementation, one harmless typed tool and a
  local test sandbox; fakes exercise the `PlannerStrategy`, `WorkerStrategy` and
  gateway ports without adding a second production execution stack;
- staged result artifact, Server-side promotion CAS and correlated telemetry.

Exit:

```text
submit → static/passthrough → one TaskSpec → one Task → first Attempt
       → WorkerJob → A2A → Worker → staged artifact → accepted terminal status
```

passes, including Server restart, duplicate transport delivery and fake-port
conformance.

### Phase 3 — Safe concurrency

- N Agents, heartbeat leases, capability routing and fencing;
- one enabled durable Worker provider, if product need justifies it, plus
  `recover_start` and externally single-owned same-incarnation continuation
  tests; otherwise the production profile remains `non_resumable` and this
  optional capability stays disabled;
- Podman sandbox leases and production policy;
- fan-out/join, deadlines and budget enforcement;
- durable tool-effect intent/outcome/reconciliation records, Server read-only
  evidence checks, gate-serialized intent creation, bounded late evidence and
  default-deny live-target/egress enforcement;
- Agent-loss and policy retries for `read_only` or proven replay-safe tool
  contracts; non-idempotent or ambiguous effects remain
  `external_effect_unconfirmed` and block automatic retry;
- the decomposing Planner behind the same typed-command/state-machine boundary,
  including Planner LLM budget and pre-CAS recovery tests;
- a second Worker implementation and cross-implementation conformance tests;
- cancellation and aggregate-budget tests for a Worker that may plan, work or
  maintain private subtasks inside one boundary Attempt;
- telemetry aggregation, sampling, backpressure, retention and diagnostic
  queries.

Exit: Agent-loss, stale-result, cancellation, effect-reconciliation, live-target
denial, isolation and load tests pass. No effect-capable retry is enabled without
durable evidence that replay is safe.

### Phase 4 — v1 workflow migration

Migrate one workflow at a time. Start with the canonical public workflow
`oas_build` as a static four-stage manifest:

```text
dependency_information → project_information → oas_update → oas_validate
```

Preserve the business ability to reuse the first two stages, not v1's mutable
logical-name cache test. Reuse is allowed only when the accepted RunSpec supplies
the exact discovery output refs and the scheduler commits a `reused`
`TaskResolution` whose fingerprint covers the TaskSpec, project snapshot,
resolved inputs, effective policies and permitted producer provenance. Absent or
mismatched evidence executes the stage. The ledger records v1's “non-empty
artifact with this name” check as an intentional incompatibility. Preserve the
final validator and authoritative `oas-openapi-building` publication. The
migration MUST document how the v1 `oas_update` requirement for two successful
iterations is represented; intentional refinement passes are explicit workflow/
Worker behavior, never retries of a failed Attempt.

Next migrate the complete public `trace-graph` workflow as one passthrough root
Task. Its Worker enumerates the exact OpenAPI operations deterministically,
processes them in legacy order over a shared overlay/path namespace, isolates an
ordinary failure at the operation boundary and returns the declared aggregate
patch/diff and finding artifacts. `trace-direct` uses the same public/root
mapping with graph tools disabled. An operation-level Worker contract may be
compared as a controlled ablation, but no external evaluation child-Run loop is
the runtime implementation of the public workflow.

Then migrate `trace-graph-pathpar` as one passthrough root whose Worker uses
Attempt-private route-group workspace views, bounded concurrency, per-group
failure isolation and the pinned deterministic merge policy. Server observes one
Attempt and promotes only the final aggregate refs. This preserves the legacy
cross-operation overlay/memory and fork/merge boundary without inventing an
unspecified dynamic static-manifest expansion. A migration maps:

- v1 string artifact keys to typed `ArtifactRef` contracts;
- v1 Worker builders to versioned Agent capability implementations/providers;
- retained v1 Planner semantics to typed Planner commands, or explicitly
  retires them when a static/passthrough candidate meets the same business
  acceptance criteria;
- v1 global sandbox cleanup to attempt leases;
- retained v1 event facts to authoritative domain/audit records or bounded
  derived telemetry, without copying every legacy event dictionary.

For each migrated logical unit, compare the same objective and exact initial
artifact versions. Required outputs must validate against the same schemas and
meet the same workflow-specific thresholds; stochastic text or artifact bytes
need not be identical. Compare two v2 strategies only when they are genuine
substitutes for that same logical unit and evaluator. Operation-level trace
fixtures evaluate the internal Worker contract only; the migration gate also
exercises the complete public root objective, ordered/shared state, partial
operation failure behavior and final aggregate outputs.

The existing v1 `oas_build` task eval exercises `oas_update` and does not cover
the complete public workflow. Reuse its OpenAPI scorer, but the v2 migration
fixture covers all four stages, conditional reuse, validation, final artifact
publication and fail-fast semantics. Evaluation retains every repetition; v1
pass@N aggregation or a selected passing repetition MUST NOT be used to compute
strategy deltas.

Do not copy `TaskRunner` as a second durable scheduler or expose Agent-private
planning/working steps as Server Tasks.

Exit per workflow: its migration-ledger entry is `validated`; its enabled
workflow-catalog profile and authenticated/negative submission fixtures pass;
behavioral fixtures pass, outputs are documented, failure/effect semantics are
explicit, the chosen v2 migration meets the v1-baseline quality gate, any
genuine substitute comparison has a finalized scheduled/release report, and no
v1 runtime import remains.

### Phase 5 — Production readiness

- authentication/authorization hardening, credential rotation and operator
  views (the externally reachable Phase 2 API is already authenticated and
  authorized);
- backup/restore and migration rehearsal;
- deployment manifests, dashboards, alerts and runbooks;
- performance budgets, scheduled evaluation baselines/reports and a
  release/cutover decision.

Release/cutover gate: every canonical v1 workflow has a ledger disposition of
`validated`, explicitly approved `deferred`, or `retired` with replacement and
cutover evidence; no entry remains `not_assessed`, `planned` or `in_progress`.
Every `validated` key is enabled in the release workflow catalog and passes its
authenticated submission/authorization fixture. A key absent from that catalog
must be `deferred` or `retired`, never `validated`.

## Global end-to-end acceptance

The release candidate automates all of the following:

1. Submit an idempotent run and observe one Run ID.
2. Every enabled Planner strategy commits typed `TaskSpec` values through the
   same RunState state machine; the scheduler materializes Attempt-specific
   `WorkerJob` values and dispatches them through the same A2A path.
3. Enabled ADK sessions, artifacts and bounded telemetry projections use the
   same PostgreSQL database through one pool per process without duplicating
   every ADK event into Contractor telemetry.
4. Duplicate delivery, Server restart and Agent death do not duplicate a
   committed task result.
5. Cancellation stops active work and cleans only the target sandbox.
6. PostgreSQL or Proxy outage causes bounded, classified failure/retry behavior.
7. Final status, output artifacts and the authoritative lifecycle remain
   queryable; sampled/dropped diagnostic detail is identified explicitly.
8. Backup/restore preserves the completed run and its references.
9. Crash before/after RunState CAS and before/after A2A send obeys the outbox
   protocol without duplicating logical execution.
10. Runtime database roles have no DDL grants and Agent artifact access remains
    constrained by RLS.
11. Duplicate registration, heartbeat replay, Agent restart and Server epoch
    change leave exactly one eligible logical Agent and never revive a fenced
    lease.
12. Crash after Agent accepts an execute but before Server receives the A2A task
    ID is recovered by endpoint/context; concurrent cancellation durably wins or
    follows the Agent mapping through one tombstone/execute transaction, never
    starts a post-cancel task and stops a recovered task exactly once.
13. Passthrough commits one root Task from one root `TaskSpec`; its Worker may
    execute directly, plan/work or maintain private subtasks, but Server observes
    one boundary Attempt and one aggregate `WorkerResult`.
14. Retrying that root creates a new Attempt, `WorkerJob` and fence for the same
    Task/`TaskSpec`; redelivery reuses the exact job for the current Attempt, and
    cancellation stops its Worker and sandbox work.
15. The golden Worker conformance suite passes through each enabled Worker
    implementation without changing Server Planner/domain code or A2A routes.
16. A client publishes an exact project snapshot and submits it without a host
    path; corrupt, cross-principal and traversal inputs fail before dispatch.
17. Anonymous or wrongly scoped principals cannot submit/cancel Runs, read
    artifacts/events or access fleet control, and deleting optional telemetry
    or ADK session detail leaves append-only security audit queries unchanged.
18. A live stale Agent pool cannot authenticate by copying a replacement's
    registration fields; persistence-capability rotation fences ordinary writes,
    exact recovery grants permit only their declared mapping actions and a
    non-open operation-start gate prevents every late model/tool start before
    external I/O.

## Traceability rule

CI maintains a machine-readable mapping from every requirement ID to one or
more tests. A missing mapping fails the release gate even if implementation code
exists. A phase is complete only when its specs are marked `Implemented` and
the corresponding architecture model still validates.
