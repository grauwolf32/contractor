**Contractor: repeat architectural review and validation of recommendations**

Date: 2026-09-19. Reviewed HEAD `dc4d6f3aa58522dbdf67b7c2164d7dfb638a0316`
and the available working tree. This document was revised after the user's
feedback that the ADK rationale for the finalizer was found too late. This
revision replaces the initial recommendations; R01–R19 identifiers are retained
for traceability. The [task plan](../plans/2026-09-19-architecture-review-followups.md)
separates confirmed fixes from decisions and experiments.

After the review, V57-001 (R03, model selection/provenance) and V57-002 (R04,
status summaries) were completed at the user's separate request. The analysis
of the original HEAD is retained below; current implementation and checks are
recorded in the task files and linked plan.

Status: non-normative analysis. Specifications and task files remain the owners
of contracts and statuses. This review does not authorize changing current
completion, reassign occupied tasks or launch live evaluations.

**The initial analysis was too confident.** The mistake was not limited to
missing V21-001. Several implementation properties were called shortcomings
without checking why they had been chosen. The repeat review compared code,
the original task, subsequent changes, specification, existing backlog and a
possible counterargument for every recommendation.

Particularly significant corrections:

- Runtime already assembled results from ordinary text in V13; V17 deliberately
  revoked that behavior, and V21 separated tools from schema. Removing the
  finalizer requires revisiting a decision, not merely deleting a redundant call.
- Failure-only evidence in the old live harness, blob buffering, fixed allocations,
  whole-batch retry and the absence of a cross-allocation graph cache were accepted
  limitations. Their cost may justify development, but does not itself prove a defect.
- In-source annotations are an intended useful output. The proposal to replace
  them with a primary sidecar is withdrawn. Private DTO generation is also removed
  from the recommended plan because its benefit is unproven.
- Simply enabling keep-alive is insufficient: actual reuse, transport retries,
  principal binding and cleanup must be checked.
- Two groups of fixes are confirmed: model-selection/attribution regression in
  the live harness and incorrect current status summaries. Other items have
  different statuses: local refactor, new feature or hypothesis.

**A full rewrite is not justified now.** No requirement was found that already
proves a need to replace all of Contractor. This does not establish simplicity
or safe migration for every one of the 19 ideas. Resume, per-item settlement and
allocation on demand change fundamental execution boundaries; their cost and
transition have not yet been designed.

Starting again, I would establish verifiable domain outcomes, separation of
model-authored data from Runtime authority, and typical journey costs earlier.
But I would not make “Check an application” the only product model: source
analysis, OpenAPI/LikeC4, reusable reports and annotated source have independent value.

Preserve exact inputs/provenance, PostgreSQL ownership of durable state, atomic
result acceptance, lease/fencing, credential isolation, the distinction between
technical success / assessment / human decision, explicit artifact reuse and
same-Run Memory. Replacing Go/Python/React does not itself improve any of these guarantees.

**Complete findings matrix after verification**

“Local” means a bounded implementation preserving contracts. “Incremental” means
coordinated changes across components. “Subsystem” means changing the unit of
recovery, result acceptance or resource lifetime. Scope estimates are conditional
for designs that have not yet been selected.

| ID | Verified conclusion | Verdict | Next step and scope |
| --- | --- | --- | --- |
| R01 | Necessity of the ordinary exact-copy LLM call remains a contract question; typed assembler is implemented | V57-003 investigation archived on 2026-09-20 | Standalone fake-model environment is excessive; focused checks are needed for a concrete change |
| R02 | Every Invoke disables keep-alive; reuse benefit and safety are unmeasured | Experiment | V57-005; bounded transport path |
| R03 | Model override does not reach the current Worker policy; evidence records the requested alias | Confirmed defect, narrowed scope | V57-001; local. Success retention separate |
| R04 | Some summaries describe completed work as pending/planned | Confirmed inaccuracy | V57-002; documentation, no mandatory generator |
| R05 | Shared CI gate is large, but costly duplicate work is unproven | Investigation | Timing/execution manifest if needed |
| R06 | Model-free Worker exists; SDK footprint is not a scanner-runtime defect | Investigation | Isolated profile only after comparative measurement |
| R07 | Handwritten private DTOs and golden fixtures were deliberate; no drift demonstrated | Recommendation withdrawn | No new codegen task |
| R08 | A bundle simplifies delivery, but no installation failure was reproduced | Deferred feature | One fresh-install profile on separate request |
| R09 | Pentest journey extends the product and requires new domain decisions | Narrowed | Existing V54 draft; contracts first, then integration |
| R10 | Evidence plane exists; a specific durable HTTP exchange/replay contract is missing | Narrowed | Include in V54 contract/capture, without a new general epic |
| R11 | Annotated source is a required output; sidecar-primary changes the product | Recommendation withdrawn | Investigate invalidation/optional index if cost is demonstrated |
| R12 | Full bounded blob buffering is allowed by the specification | Investigation | Transfer benchmark first; streaming not assigned |
| R13 | Intra-allocation cache and explicit report reuse exist; shared graph cache was excluded | Investigation | Measure after the digest fix, then make a separate storage decision |
| R14 | Audit-aware admission protects authority; some merge work is already in V56 | Deferred | Use V61-003; no shared compiler assigned |
| R15 | Tool receipts exist; bounded restart belongs to V55-008 | Narrowed | Existing task; transparent Stage resume is a separate subsystem |
| R16 | Whole-batch retry was deliberate; local collector progress is not an accepted result | Deferred | Measure retry cost, then consider settlement redesign |
| R17 | Fixed allocations provide complete placement and stable sessions | Deferred | Representative workload before changing dispatch/allocation |
| R18 | A shared model executor does not remove distinct Planner/Worker authority | Investigation | No implementation task |
| R19 | An internal Audit JSON round trip exists; boundary checks remain justified | Narrowed locally | V57-004 complete: decoder/assembler and agreed removal of the synthetic Audit bound |

**R01 — ordinary completion: revisit the requirement, then implementation**

History changes the assessment:

1. [V13-001](../../tasks/v13/v13-001-runtime-owned-worker-result.yml) already removed
   the LLM serializer: plain terminal text became a Runtime-owned result,
   with exact artifacts selected from trusted observations.
2. [V17-003](../../tasks/v17/v17-003-structured-worker-results.yml) explicitly revoked
   that plain-text success projection. Requirement R1:
   “No arbitrary final model text can become a successful WorkerResult”.
   The model had to return `WorkerModelResult(subtaskId,result)`.
3. [V21-001](../../tasks/v21/v21-001-adk-worker-result-finalizer.yml) fixed an observed
   LikeC4/LiteLLM → LM Studio grammar failure when combining tools and schema.
   A separate tool-free serializer preserved the strict boundary. It deliberately
   extracts no new meaning and must copy the text exactly.

The current design is therefore the result of several decisions, not a forgotten
helper. The main Worker must remain without `output_schema`; an adapter's
advertised capability does not establish downstream-route compatibility.
The [ADK documentation](https://adk.dev/agents/llm-agents/#data-handling) also
distinguishes model support for tools/schema from fallback through a special tool.
That does not replace checking the specific installed combination.

The schema indeed now contains only a request-owned ID and a string. The second
model does not establish the analysis's truth: its output is checked for exact
copying. Runtime assembly therefore remains a reasonable candidate. But the
decision must explicitly state which V17 guarantee is preserved, replaced or no
longer needed. Semantic extraction and exact-copy behavior cannot both be required.

In [runtime.py](../../runtime/src/contractor_runtime/worker/runtime.py), the
ordinary finalizer, Audit collector completion and terminal summarizer have
different inputs. Ordinary completion could use the current serializer or typed
assembly from completed text; Audit already assembles submissions; the summarizer
actually creates new content. They cannot be unified mechanically.

**Reassessment on 2026-09-20: V57-003 is archived as an excessive standalone
investigation.** It is not a complete duplicate of V57-004: the ordinary finalizer
still runs, and no decision to remove it has been made. However, history is
already reconstructed, typed assembly implemented, and existing tests cover
malformed/changed copies, the mandatory extra call, its budget, cancellation
and truncation.

The original V57-003 required an alternative test-only pipeline even for
keep-current. That experiment would mainly reproduce its predetermined
differences: without the copying step, its errors and extra call disappear.
It would not establish whether the product needs the stage, measure production
cost or prove the future implementation's lifecycle. New transition checks are
more useful against a concrete production patch, using V57-004's assembler
and fixtures.

One question remains: **does an ordinary Worker need to copy completed text
through another LLM call when Runtime already validates fields and owns refs?**
Revisit the change through an explicit choice of semantics or observations of
ordinary-finalizer failures/cost. The known ADK tools/schema limitation still
matters for the current structured-output path; its disappearance is not assumed.
Archiving proves neither the value of retaining nor removing the copying step
and does not change the current contract.

If Runtime assembly is selected for ordinary completion, the concrete task must
cover spec 14, focused regressions, accounting and completion at the budget
boundary. Configured budgets are not automatically lowered. Reaching the call
count limit, exceeding a token budget and receiving a truncated response are
different cases. Archived V53 investigations are not resumed. The comparison
module, decision document and evidence from V57-003's original scope are absent;
the task is not declared complete.

Cutover needs a separate decision. Registration `softwareVersion` is informational;
placement checks runtime/toolset/capability refs, not an exact build. Draining
only active allocations does not protect queued Runs or later Stages of an old
Run. A coordinated transition after all affected nonterminal Runs finish is
possible in the local project; coexistence needs an explicit pin and mixed-fleet
checks. No legacy DTO family is required.
Sources: [private contracts](../../internal/contracts/private.go),
[registry](../../internal/controlplane/registry.go).

**R02 — A2A reuse: check the whole transport**

The [client](../../internal/planner/a2a/client.go) creates a Transport within Invoke
with `DisableKeepAlives: true`. The flag arrived with principal binding in
[V8-004](../../tasks/v8/v8-004-runtime-agent-principals.yml); no separate rationale
for the flag was found in the task. That does not establish it as accidental.
The 100 ms polling interval explains potential handshake cost, but is neither
a load measurement nor proof of the main bottleneck.

In pinned A2A Go SDK v2.5.0, both SendMessage and GetTask are JSON-RPC POSTs.
The body can be rewound through GetBody; Go Transport allows certain retries
on reused connections. With reuse only inside Invoke, initial SendMessage uses
a fresh connection, and later polls are logically read-only. This analysis
has not demonstrated a dangerous repeat of already executed model work.

SDK `Destroy()` is a no-op for JSON-RPC, and our wrapper provides no idle-connection
cleanup. The raw Transport needs an explicit owner. Reading one JSON object
also does not prove full body draining or actual reuse.

V57-005 is an experiment with real local TLS, a connection counter, zero/partial
write, lost response, malformed/oversized/chunked body, cancellation and
principal/allocation-change checks. Conditions: no repeated semantic SendMessage,
no bytes sent to the wrong SPKI, preserved response bounds and a connection
lifetime limited to the invocation. Certificates are not reverified on every
poll over one persistent connection; that is part of the decision too. Neither
a global pool, streaming nor a production toggle is part of the experiment.

**R03 — live harness: fix the specific regression**

[V2-008](../../tasks/v2/v2-008-project-workflows-live-eval.yml) requires env overrides
for URL/token/model. However,
[copyLiveConfiguration](../../tests/eval/project_workflows/live_process_test.go)
changes only legacy `domain_worker*` policies. Selected
The historical `openapi-from-workspace@5` and `likec4-from-workspace@5`
versions used `worker@2`
from [worker.yaml](../../configs/model-policies/worker.yaml).
No override is passed to CreateRun.
[ModelSHA256](../../tests/eval/project_workflows/live_test.go) is computed from
the requested name; it does not prove the model alias actually sent, much less
upstream weights/revision.

V57-001 restores model selection and accurately separates requested alias,
effective policy/route and unknown upstream identity. A focused offline fixture
must observe the actual Gateway request for both workflows; checking a string
in copied YAML is insufficient.

Failure-only evidence retention was explicitly part of V2-008. Success retention
is therefore excluded from the bug fix. A full experiment, if needed, should use
the already implemented V41 portable format. Fixing the old harness does not
close [V40-002](../../tasks/v40/v40-002-agent-instruction-paired-runner.yml), whose
own pins, mappings, aggregate dispatch and observation collection remain open.

**R04 — correct summaries without creating another status system**

In the [task index](../../tasks/index.yml), UI planning still says planned and
“All new tasks are pending”, although V37 is complete. The
[spec index](../spec/README.md) calls V39-007 a remaining release gate, although
the [task](../../tasks/v39/v39-007-audit-completion-release-gate.yml) is complete.
These are specific documentation inconsistencies.

However, `active_delivery_order` is dated 2026-09-06 and does not override
dependencies/ownership. A retained delivery sequence need not be a live queue.
V57-002 clarifies historical/current summary meaning and reconciles summaries
with task files. No roadmap generator, automatic prioritization or new source
of status is needed.

**R05 — CI: establish the critical path and checks actually executed first**

[CI](../../.github/workflows/ci.yml) has a shared job with a 35-minute timeout;
the [Makefile](../../Makefile) has a large aggregate. Timeout is not measured
duration. Overlapping packages and repeated dependency sync do not prove costly
duplicate work either: make deduplicates shared prerequisites, and some setup
may be a cheap no-op.

Broad process/race/fault gates were deliberately required, for example in
[V8-016](../../tasks/v8/v8-016-runtime-configuration-hardening-e2e.yml).
The [findings runner](../../scripts/test-findings-e2e.py) already verifies actual
run/pass/skips. Its absence from the mandatory pipeline matches the scope of
[V43-005](../../tasks/v43/v43-005-findings-contract-gate.yml).

If CI obstructs delivery, first collect durations, cold/warm setup costs, actual
selected/pass/skip cases and flaky failures. Split jobs or unify gate scripts
around an identified costly step. No implementation task is created now.

**R06 — scanner-only Runtime: a separate measurable capability**

[V55-003](../../tasks/v55/v55-003-model-free-tool-worker.yml) requires tool execution
without model services/credentials, not an installation without ADK.
Full [factory composition](../../runtime/src/contractor_runtime/factories.py)
does not contradict that requirement.

The [local memory report](../operations/runtime-memory-2026-09-19.md) shows a
substantial SDK import footprint but does not prove a need for a new installation
profile. The inclusive LLM factory figure of 46.4 MiB already includes OpenAI's
19.8 MiB; additional ADK factory import costs 24.7 MiB. Nested figures cannot
simply be added together. Thirty ADK lifecycles in that report showed no continuing growth.

Before assigning a lightweight profile, compare identical startup/probe, first
tool Run and repeated allocations for full and scanner-only compositions.
Preserve exact advertised capabilities and the default full profile. For a
model Worker, lazy imports only defer an unavoidable cost.

**R07 — private DTO codegen: recommendation withdrawn**

The [private API README](../../api/v1alpha1/README.md) defines explicit Go/Python
DTOs and shared golden fixtures.
[V50-001](../../tasks/v50/v50-001-unified-private-contracts.yml), at the user's
request, removed parallel DTO/conversion families. This is not a prohibition on
generation, but a reason not to add a layer without demonstrated benefit.
Public Go/TypeScript generation already exists.

The initial analysis demonstrated neither specific drift, regression frequency
nor the cost of manually changing a selected schema. A generator pilot is not
a necessary fix. Actual drift first needs a shared failing fixture and model
correction; discuss codegen when recurring cost is measured. A generator does
not replace semantic validation.

**R08 — release bundle: a delivery feature, not discovered debt**

Single-host topology, health/capability checks and a
[deployment guide](../deployment.md) already exist. The review reproduced
neither installation failure nor manual-installation cost. A new installer is
reasonable under a separate distribution requirement.

The initial scope would then be one fresh-install host profile, repeated
bootstrap and reboot preserving identity/data on a disposable VM. The original
update/rollback promise is too broad: migrations are forward-only, and upgrades
require coordinated transition. An installer does not make arbitrary binary
rollback safe. Upgrade/restore need separate requirements and checks.

**R09 — “Check an application”: a new journey with open contracts**

Existing [Audit compatibility](../../internal/auditservice/compatibility.go)
deliberately rejects automatic active checks and initial finding-candidates.
These are not forgotten switch branches.

The [native pentest plan](../plans/2026-09-19-native-pentest-audit.md) is a draft;
its listed V54 task files do not yet exist. It still needs decisions on
first-round inventory before discovery, cross-round recon reuse, check routing,
anonymous versus Project authorization, identity/scope and the distinction
between supported / replayed / human-confirmed.

If pentest becomes the next product priority, make those decisions first, then
build a bounded capture → verification → UI path on existing Audit/Runs/artifacts.
“Incremental integration” applies to a bounded, agreed profile; universal
pentest cost is not yet established. R09 does not become a new dependency for
finishing V55 scans.

**R10 — evidence: close a specific gap for a specific consumer**

The [findings specification](../spec/27-findings-tools-and-collections.md) already
describes observations/hypotheses/findings, exact refs, retention and collections.
Generic ArtifactRef and optional reproduction instructions are deliberate
generality, not the absence of an evidence plane.

A narrow gap is confirmed: the
[HTTP tool](../../runtime/src/contractor_runtime/toolsets/http/tools.py) stores
response bodies as artifacts but history in an allocation-local deque cleared
during cleanup. [Spec 28](../spec/28-finding-analysis-and-sarif.md) explicitly
notes that the body does not reconstruct the request, headers and credentials.

Replay requires durable exchange/verification evidence with exact
identity/scope/deployment refs, explicit completeness and a separate replay
receipt. This is already the subject of draft V54 contract/capture work, not a
second general evidence epic. Source evidence separately needs snapshot-bound
capture; a typed artifact does not turn a model claim into a trusted fact.
Do not combine HTTP replay, source lineage and full SARIF into one mandatory
new contract before a working consumer exists.

**R11 — annotations: preserve the output's purpose**

[Spec 13](../spec/13-taint-annotations.md),
[V16-004](../../tasks/v16/v16-004-trace-annotation-config.yml) and the
[trace skill](../../configs/skills/trace/SKILL.md) intentionally provide source
mutation, cumulative overlay state, diff and report. No retained direct user
quote saying “in-source only” was found; it must not be attributed to the user.
The deliberate contract is nevertheless confirmed.

The [annotation-index proposal](annotation-index-proposal.md) proposes an optional
participation/provenance index while retaining annotations in source. It does
not justify turning annotated source into an optional export.

The cost of annotate → reanalysis may be substantial, but must be measured:
repeated reads without edits, changed=false, actual insertion, stale snapshot
and one sink in different contexts. If necessary, optimize invalidation or add
an optional index while preserving output. Unchanged code behavior after adding
a comment does not justify reusing old graph coordinates.

**R12 — streaming artifacts: accepted bounds are not an unbounded buffer**

[Spec 23](../spec/23-artifact-blob-backends.md) permits one complete payload for
hash verification: at most 64 MiB and four concurrent transfers, with admission
before reading. This is not a 256 MiB process cap: additional driver/serialization
buffers exist. The [filesystem backend](../../internal/artifacts/blob_filesystem.go)
checks size, an extra byte and digest before returning data.
[V34-001](../../tasks/v34/v34-001-blob-contracts-and-postgres.yml) preserves PostgreSQL
mode without a mandatory temporary volume.

Streaming changes when integrity is established and how consumers behave.
First benchmark backend × payload size × concurrency 1/4: RSS, copies, latency
and cancellation. Only if bounds fail actual workload requirements should one
end-to-end path with verify-before-success be designed. The Runtime workspace
digest memory report does not measure Server blob transfers and cannot justify
this recommendation.

**R13 — source indexes: account for existing reuse**

[V15-003](../../tasks/v15/v15-003-trailmark-child-host.yml) introduced a child lifecycle,
in part to return graph RSS to the OS.
[Spec 12](../spec/12-code-analysis-tools.md) explicitly excludes persisted/shared
graph artifacts and cross-allocation caching. However, a cached graph for the
current digest and a shallow cache already exist within an allocation.
Each tool call does not imply a new graph build.

[V2-009](../../tasks/v2/v2-009-precomputed-analysis-variants.yml) implemented explicit
from-analysis workflows with exact discovery reports. Model-analysis reuse is
already available. [Memory](../spec/08-memory-tools.md) survives allocation release
within its Run scope; a different Run receives previous conclusions through
explicitly promoted artifacts, not hidden note reads.

First measure snapshot/hash/parse/build/cache hits after the current
[workspace digest optimization](../operations/runtime-memory-2026-09-19.md).
That separate local replay reduced peak memory from 361.1 → 205.0 MiB with the
same digest; it is uncommitted concurrent work, not reproduced here, and does
not guarantee per-allocation RSS. No duplicate task is needed.

If repeated indexing remains significant, decide source/engine pins, completeness,
owner access, disk limits, eviction and invalidation. A shared graph process or
hidden cross-Run reuse does not follow automatically from a benchmark result.

**R14 — execution compiler: authority checks are not automatically layer leakage**

[V39-002](../../tasks/v39/v39-002-audit-completion-pinning-placement.yml) requires
trusted task/manifest/output pins and a selected passthrough single-Worker target.
[Scheduler checks](../../internal/scheduler/audit_completion.go) protect precisely
that boundary. Frontloading does not replace checking the selected attempt at
escalation, allocation and secret materialization.

[V61-003](../../tasks/v61/v61-003-run-toolset-pinning-allocation.yml) already includes
pure merge and selected per-allocation projection. Another compiler epic would
duplicate some work. For now, complete its existing acceptance and extract a
helper for specific duplication. Generic completion is better designed around
a second real consumer while preserving Audit authority.

**R15 — receipts/resume: do not promise more than the selected job mode**

At-most-one Planner invocation and interruption after process loss are original
constraints confirmed by [V1-006](../../tasks/v1/v1-006-streamline-planner.yml),
not accidentally unfinished persistence.

[Tool Workers](../spec/29-tool-workers.md) already have durable receipts, unknown
outcomes and a prohibition on unsafe automatic rescanning.
[V55-008](../../tasks/v55/v55-008-deterministic-scan-planner.yml) contains the required
bounded deterministic planner/restart scope. It must check logical job identity
together with Stage-attempt identity, completed-receipt replay and unknown
outcomes without rescanning. stageExecutionId must not be removed from the key.

This is not resume for arbitrary ADK transcripts, external side effects and all
Stages. No general recovery engine is assigned without a concrete consumer.
Transparent Stage resume remains a substantial subsystem redesign.

**R16 — AuditItem checkpoints: distinguish local progress from authoritative acceptance**

[V26-004](../../tasks/v26/v26-004-batched-audit-executions.yml) deliberately chose
small bounded batches to reduce repeated preparation and share analysis;
partial durable checkpoints were left out of scope.
[Spec 19](../spec/19-audits.md) and
[spec 25](../spec/25-audit-worker-finalization.md) preserve terminal collection
and atomic batch acceptance.

Items accepted by an allocation-local collector have not yet been accepted by
the authoritative importer. Process loss loses unsealed local progress; it does
not erase accepted terminal collections. The original phrase “loss of already
accepted results” was inaccurate.

First measure preparation/context cost and retry amplification for the selected
profile. If losses justify checkpoints, a new receipt/pin/settlement design is
needed, covering cancel, supersession, late results, current round and retention.
This changes the unit of result acceptance; it is not partial ZIP import.

**R17 — allocation on demand: a different admission model**

[V3-004](../../tasks/v3/v3-004-router-planner.yml) deliberately hides physical capacity
from Planner and excludes parallel dispatch. The
[core spec](../spec/02-runtime-and-a2a.md) requires complete injective matching
before reservation, preventing partial resource holds and greedy-placement errors.
A fixed set provides stable sessions/workspaces and avoids waiting for capacity
mid-Planner.

Idle slots may have a cost, but this review did not measure a representative
Router workload. With demonstrated demand, a separate stateless-job mode could
be designed: capacity waits, deadlines, fairness, recovery and fencing.
That redesigns allocation/dispatch; it is outside V55-008 and cannot be achieved
simply by removing matching checks.

**R18 — unified executor: still a research hypothesis**

Go Planner owns durable orchestration facts; Python Worker owns tool/runtime
execution. Different roles have different budgets, secret scopes and lifecycles.
One language or framework does not itself remove these differences.

The [Stateflow draft](stateflow-1.md) is a new opt-in Planner, not a ready migration
to a unified executor. It preserves fixed Workers and excludes transparent restart
resume/concurrency. Do not combine stateflow, a language change and recovery into
one “simplification” project. Parity fixtures and demonstrated maintenance-cost
reduction are needed; no implementation task is created now.

**R19 — validation: separate a redundant conversion from independent protection**

A specific internal conversion exists in
[runtime.py](../../runtime/src/contractor_runtime/worker/runtime.py):
Audit completion already has a typed decision but creates a JSON wrapper that
the shared helper immediately parses. It can be removed by separating model JSON
decoding from trusted result assembly. This is V57-004, a refactor independent
of R01 that preserves current semantics and safe failure codes.

| Check | Boundary protected | Decision after review |
| --- | --- | --- |
| Model-result JSON/schema/subtask echo | Untrusted finalizer/summarizer output | Preserve now; the ordinary portion disappears only if R01 is accepted |
| Exact-copy equality | Current serializer must not rewrite text | Preserve with the current finalizer; do not treat as a truth assessment |
| Audit-generated wrapper decoding | Runtime's own temporary format | V57-004: typed handoff; only the artificial wrapper-size bound removed at the user's explicit choice |
| Empty text, UTF-8 size, truncation | Valid completion and bounded data | Preserve; a typed object does not prove the response is complete |
| Final encoded envelope limit | Escaping, observations and artifact refs enlarge the wire payload | Preserve separately from the 64 KiB result limit |
| Completed invocation-local exact refs, reserved bindings | The model does not assign artifact or namespace authority | Preserve, including auto-export |
| Secret checks, cancellation, lease/fencing, budget failure | Safe acceptance at completion | Preserve |
| WorkerCompletion validation at the Planner/Server boundary | A framework-neutral Invoker is not trusted | Preserve |
| Repeated validation of a mutable DTO within one helper path | Possible local repeated work | Only after ownership is proven; currently out of scope |
| Two private-codec passes | Duplicate keys/constants and strict JSON conversions | Do not remove as an “obvious duplicate” |

The last row matters:
[codec.py](../../runtime/src/contractor_runtime/contracts/codec.py) first detects
duplicate keys/nonstandard constants, then `model_validate_json` preserves strict
JSON conversions, particularly RFC3339 datetimes. Replacing it with
`model_validate` on an already parsed dict is not necessarily equivalent.
The [Planner boundary](../../internal/planner/worker_result.go) also explicitly
treats Invoker implementations as untrusted. Removing similar-looking checks
across processes is not the same as removing an internal round trip.

V57-004 does not apply broad skip-validation/`model_construct`, disable the
finalizer or promise a measured speedup. Its rationale is clearer decoder/assembler
ownership and removal of one specific unnecessary internal format.

V57-004 implementation was completed in `7f553b22`. After separate discussion,
the user chose to remove only the artificial Audit JSON limit. Shared
64 KiB/256 KiB contracts were preserved for future discussion.
[Change and origins of the limits](2026-09-19-worker-result-assembly.md).

**Verification limits and currency**

V37, V39, V41, V45 and V50 are complete; their historical shortcomings are not
entered as new tasks. V40-002 remains in_progress with live_ready: false,
V40-003 pending. V55-001–003 are complete, V55-004 in_progress and the other V55
tasks pending. V56 consists of separate pending task specifications.
V53-001/002 are archived at the user's request and do not return to the queue.
The source is the corresponding task files in the [ledger](../../tasks/index.yml),
a working-tree snapshot on the review date. Concurrent uncommitted changes are
not considered delivered.

This review validates findings and the plan rather than implementing them.
The initial analysis ran only
`runtime/.venv/bin/pytest -W error runtime/tests/test_result_finalizer.py`
— 3 passed. That checks the current finalizer, not the proposed replacement.
No new live-model campaigns, production deployment or performance experiments
ran during the repeat review. Historical gate reports and the separate memory
report are not presented here as newly obtained results. Document and new task-plan
checks are recorded in the [plan](../plans/2026-09-19-architecture-review-followups.md).
