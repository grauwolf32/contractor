# Code-as-agent harness improvement plan

> **Status:** analysis and implementation proposal; not current runtime behavior.
> **Source:** [*Code as Agent Harness* (arXiv:2605.18747)](https://arxiv.org/abs/2605.18747).
> **Last reviewed:** 2026-08-05.

This document translates the paper's harness principles into a concrete
hardening plan for Contractor. It is aimed at contributors changing the task
runtime, tool policy, persistence, verification, observability, or evaluation
systems.

The normative description of the current project remains in the
[reconstruction specification](../spec/README.md). When a proposal here is implemented, the
affected specification chapters and acceptance tests must be updated in the
same change.

## TL;DR

Contractor already has most of the *components* described by the paper:
planner/worker orchestration, typed tools, artifacts, memory, isolated code
execution, workflow-specific validators, detailed traces, and model
evaluations. The principal gap is enforcement at the boundaries where a model
decision becomes accepted shared state.

The recommended direction is therefore:

1. do not add more agents or a more elaborate workflow topology yet;
2. route side effects through a central, context-sensitive policy gateway;
3. treat model-reported completion as `candidate_done`, then require scoped
   verifier evidence before acceptance;
4. publish artifacts, checkpoints, memories, and merged work against explicit
   revisions using atomic transactions;
5. bind every run to an immutable manifest and evaluate harness reliability in
   addition to final task quality; and
6. permit harness evolution only through offline, governed experiments after
   the preceding controls are reliable.

The target invariant is:

> Every accepted agent action is a versioned, permissioned, observable, and
> independently verifiable state transition.

---

## 1. Evidence boundary

The paper is a survey and position paper, not an evaluated reference
implementation. Its comparisons summarize systems built with different
models, tasks, tools, budgets, and correctness oracles. It does not establish
that any single mechanism will improve Contractor by a predictable amount.

Consequently:

- paper-derived recommendations in this document are design hypotheses;
- Contractor's existing fixtures and adversarial fault tests are the source of
  promotion evidence;
- every mechanism should be introduced in shadow mode or behind a reversible
  configuration boundary where practical; and
- domain quality, safety, consistency, latency, and cost must be measured
  together.

The most relevant paper sections are the
[Plan–Execute–Verify loop](https://arxiv.org/html/2605.18747#S3.SS4),
[managed memory](https://arxiv.org/html/2605.18747#S3.SS2),
[tool lifecycle](https://arxiv.org/html/2605.18747#S3.SS3),
[harness telemetry and optimization](https://arxiv.org/html/2605.18747#S3.SS5),
and the open problems covering
[evaluation](https://arxiv.org/html/2605.18747#S5.SS2.SSS1),
[scoped verification](https://arxiv.org/html/2605.18747#S5.SS2.SSS2),
[safe evolution](https://arxiv.org/html/2605.18747#S5.SS2.SSS3),
[transactional shared state](https://arxiv.org/html/2605.18747#S5.SS2.SSS4),
and [human accountability](https://arxiv.org/html/2605.18747#S5.SS2.SSS5).

---

## 2. Contractor baseline

Contractor is already a substantial code-based agent harness. The following
assessment distinguishes existing mechanisms from the enforcement still
needed around them.

| Property | Existing foundation | Remaining gap |
|---|---|---|
| Executable | Typed agent tools, a rooted project filesystem, copy-on-write overlays, HTTP tools, and isolated command execution. | Execution authority is not described and enforced by one common side-effect policy. |
| Inspectable | Task events, full tool/model traces, metrics, records, and deterministic observations. | A trace is not yet bound to a complete immutable harness identity and cannot always be replayed or audited end to end. |
| Stateful | Task state, artifacts, memories, overlays, checkpoints, findings, and verification records. | Several stores are versionless or use non-transactional load/modify/save and multi-artifact publication. |
| Verifiable | OpenAPI and LikeC4 validators, source evidence gates, finding records, and independent verifier agents. | Generic task success still follows model-controlled terminal state rather than a runtime-wide verifier stack. |
| Governed | Rooted filesystems, read-only project mounts, clean sandbox environments, budgets, and callback guardrails. | Live network destinations, redirects, host networking, approvals, and privileged side effects need central runtime enforcement. |

The current behavior and known compatibility gaps are described in:

- [runtime orchestration](../spec/03-runtime-orchestration.md), including completion,
  artifact publication, and checkpoint restoration;
- [agents and callbacks](../spec/04-agents-callbacks-skills.md), including mandatory
  tool calls and verifier agents;
- [tools and filesystems](../spec/06-tools-and-filesystems.md), including overlay
  fork/merge behavior;
- [artifacts, HTTP, and security records](../spec/07-artifacts-http-openapi-security.md),
  including persistence and network trust boundaries;
- [observability and deployment](../spec/09-configuration-observability-deployment.md);
- [testing and acceptance](../spec/10-testing-and-acceptance.md).

### 2.1 Concrete gaps in the current implementation

1. `finish(done)` requires at least one completed subtask and no `new`
   subtasks, but it does not prove that the overall objective or all required
   output properties hold. See
   [`contractor/tools/tasks/tools.py`](../../contractor/tools/tasks/tools.py).
2. `TaskRunner` counts the task-scoped model state as completion and publishes
   its result before any runtime-wide deterministic acceptance gate. See
   [`contractor/runners/task_runner.py`](../../contractor/runners/task_runner.py).
3. The HTTP tool accepts a caller-supplied URL and follows redirects by default.
   See [`contractor/tools/http.py`](../../contractor/tools/http.py).
4. The code-execution container mounts the source read-only and receives a
   clean environment, but uses host networking. See
   [`contractor/tools/podman.py`](../../contractor/tools/podman.py).
5. A task result's `result`, `summary`, and `records` artifacts are written
   sequentially, so a later failure can leave a partially published group. See
   [`contractor/runners/artifacts.py`](../../contractor/runners/artifacts.py).
6. Checkpoint restore validates task-template identity and artifact existence,
   but not the exact source, inputs, policy, harness, and output content hashes.
7. Different writes to the same path from parallel overlay forks are reported
   as a conflict but the longest byte sequence is still promoted. See
   [`contractor/tools/fs/merge.py`](../../contractor/tools/fs/merge.py).
8. Memory records do not carry source revision, evidence provenance,
   validation status, expiry, or supersession metadata. See
   [`contractor/tools/memory.py`](../../contractor/tools/memory.py).
9. Evaluation records capture domain outcomes and useful resource metrics, but
   do not define a first-class harness reliability block. See
   [`tests/eval/results.py`](../../tests/eval/results.py).

These observations identify the initial integration points; they are not a
complete threat model.

---

## 3. Target lifecycle and common identity model

The desired lifecycle is:

```text
User intent
    │
    ▼
TaskContract ──────────────── source + harness base revisions
    │
    ▼
Policy gateway ────────────── durable approval ledger
    │ allow
    ▼
Bounded execution ─────────── sandbox + typed tools
    │
    ▼
Staged ArtifactTransaction
    │
    ▼
Verifier stack ────────────── deterministic sensors + scoped reviewers
    │
    ├── fail ──────────────── repair or roll back
    ├── unknown ───────────── insufficient evidence or escalate
    └── pass
         │
         ▼
AcceptanceBundle ──────────── atomic state commit + checkpoint
         │
         ▼
Run manifest, replay trace, and governed memory
```

All subsystems should use a shared set of stable identities:

| Identity | Meaning |
|---|---|
| `run_id` | One top-level workflow execution. |
| `attempt_id` | One model-driven attempt within a task invocation. |
| `action_id` | One proposed tool or state-changing action. |
| `source_digest` | Exact inspected source state: Git commit plus a digest of scoped tracked and untracked working-tree content, or a deterministic Merkle digest when Git identity is unavailable. |
| `harness_digest` | Exact prompts, tasks, skills, tool schemas, workflow configuration, policies, model parameters, and relevant runtime versions. |
| `state_revision` | Monotonic logical revision of the shared harness state. |
| `artifact_generation` | Immutable generation of one logical artifact key. |
| `approval_id` | Durable authorization decision scoped to an action or bounded action class. |
| `verifier_id` | Stable identity and version of a verification mechanism. |

Large values and secrets must not be copied into every event. Store immutable
payloads once, apply existing redaction rules, and refer to them by digest and
artifact identity.

---

## 4. P0 — Central side-effect policy and durable approvals

### 4.1 Objective

Every tool call that may read sensitive data, mutate state, execute code, use a
credential, or contact an external system must receive a policy decision based
on its concrete arguments and environment before execution.

Prompt text such as “only test the authorized target” is guidance, not an
authorization boundary. Likewise, `CONTRACTOR_TARGET_URL` identifies an
intended target but does not prove authority to interact with it.

### 4.2 Proposed contracts

```text
RunPolicy {
  policy_id
  policy_digest
  mode                         // inspect | sandbox | live
  allowed_tools[]
  allowed_origins[]
  allowed_cidrs[]
  allowed_ports[]
  allowed_methods[]
  blocked_networks[]           // default block; explicit scope may override
  allow_cross_origin_redirects
  allow_host_network
  request_budget
  payload_budget
  credential_policy
  approval_rules[]
  valid_until
}

ActionIntent {
  action_id
  run_id
  attempt_id
  task_contract_id
  tool_name
  arguments_digest
  normalized_resources[]       // paths, origins, addresses, artifact keys
  effects[]                    // read, local_write, sandbox_exec,
                               // network_read, network_mutate, privileged
  data_classes[]
  reversible
  risk_tier                    // observe | sandbox | external | privileged
}

PolicyDecision {
  action_id
  decision                     // allow | deny | needs_approval
  rule_id
  reason
  policy_digest
  approval_id?
  decided_at
}

ApprovalRecord {
  approval_id
  action_fingerprint
  authorized_scope
  actor
  decision
  evidence_refs[]
  valid_until
  recorded_at
}
```

### 4.3 Enforcement rules

1. Register a common policy callback before the existing token,
   summarization, invalid-call, and repetition callbacks assembled in
   [`contractor/agents/worker_factory.py`](../../contractor/agents/worker_factory.py).
2. Classify tools by effects, but make the final decision from tool identity,
   normalized arguments, target environment, credentials, sensitivity,
   reversibility, and current authorization scope.
3. Permit read-only project inspection automatically when it stays inside the
   rooted filesystem and policy scope.
4. Default local code execution to no network. Use a scoped egress mechanism
   for approved targets; do not use host networking as the default.
5. Validate every HTTP destination before sending. Revalidate every redirect
   destination and all resolved addresses. Private, loopback, link-local,
   metadata, and other sensitive networks are denied unless explicitly named
   in the authorization scope.
6. Bind approvals to a normalized action fingerprint or a deliberately bounded
   action class. An approval for one origin, method, or credential does not
   authorize a different one.
7. Persist allows, denials, approval requests, approval responses, expiration,
   and revocation as run state and trace events.
8. Non-interactive mode must not silently promote `needs_approval` to `allow`.
   It requires a pre-authorized durable scope or returns a policy denial.
9. Never place raw secrets in decisions, manifests, traces, or evidence
   bundles. Record a redacted identity or credential reference.

### 4.4 Failure semantics

- A policy denial is not a tool execution failure and should not trigger an
  identical retry.
- A task may select a lower-risk alternative, request approval, or terminate as
  `policy_denied`/`insufficient_authority`.
- Expired or revoked approval invalidates any not-yet-executed action.
- A policy engine failure is fail-closed for side effects and fail-open only for
  explicitly classified local read-only inspection.

### 4.5 Acceptance criteria

- Every external or privileged action has a persisted policy decision created
  before execution.
- Out-of-scope direct URLs, redirects, DNS answers, and sandbox-originated
  network attempts emit no off-scope packet in the adversarial test suite.
- An expired or mismatched approval cannot authorize an action.
- Non-interactive execution cannot bypass a human gate.
- Authorized fixtures retain their expected functionality and domain quality.

---

## 5. P0 — Evidence-gated completion and objective convergence

### 5.1 Objective

An agent may report that work is finished, but the runtime owns acceptance.
`done` becomes a candidate state rather than proof of success.

The terminal state vocabulary should distinguish at least:

```text
accepted
failed
insufficient_evidence
policy_denied
stale_state
conflict
rolled_back
escalated
```

Do not convert “the verifier could not determine this” into success, safety, or
a negative security conclusion.

### 5.2 Proposed contracts

```text
VerificationSpec {
  verifier_id
  kind                         // schema | parser | command | static |
                               // dynamic | invariant | reviewer | custom
  required
  verifies                     // explicit scope
  does_not_verify              // explicit blind spots
  inputs[]
  timeout
  configuration_digest
}

VerificationEvidence {
  verifier_id
  verifier_version
  source_digest
  state_revision
  checked_artifacts[]          // logical key + generation + digest
  outcome                      // pass | fail | unknown
  observations[]
  raw_evidence_refs[]
  assumptions[]
  limitations[]
  residual_risks[]
  checked_at
}

AcceptanceBundle {
  task_contract_id
  run_id
  attempt_id
  reported_status
  accepted
  evidence[]
  untested_regions[]
  residual_risks[]
  accepted_by                 // runtime rule, reviewer, or human approval
  accepted_at
}
```

### 5.3 Runtime transition

```text
agent reports done
    → candidate_done
    → stage outputs
    → run all required VerificationSpec entries
       ├── all pass: create AcceptanceBundle and commit
       ├── any fail: return structured diagnostics to a repair attempt
       └── any required unknown: insufficient_evidence or escalation
```

A failed verification consumes the configured repair budget but does not count
as a successful iteration. Repeated verifier failure must stop through an
explicit budget or stagnation rule, not model confidence.

### 5.4 Initial verifier stack

Use cheap deterministic sensors before expensive or probabilistic reviewers:

1. Parse `result` and `records` against their declared output schemas.
2. Confirm required artifact keys, generations, and content digests.
3. Run the existing OpenAPI parser/Vacuum and LikeC4 validation where those
   outputs are produced.
4. Validate that reported source paths and locations exist in `source_digest`.
5. Validate finding and verification namespace relationships.
6. Require a live `exploitable` verdict to cite same-run request/response
   evidence within the approved target scope.
7. Require independent review or durable human approval for configured
   high-impact conclusions.

An LLM verifier can contribute evidence but is not a deterministic oracle. Its
scope, input revision, uncertainty, and blind spots must remain explicit.

### 5.5 Security convergence

Security workflows converge only within a declared scope. Acceptance should
require the workflow-specific combination of applicable evidence, for example:

- structural/schema checks;
- static source and data-flow evidence;
- dynamic request evidence when live testing is authorized;
- invariant, differential, metamorphic, or fuzz checks where configured;
- independent verifier review; and
- human review for policy-defined high-risk findings or actions.

The result must state uncovered endpoints, source regions, attack classes, or
runtime conditions. “No verified finding” and “verified safe” are different
claims.

### 5.6 Acceptance criteria

- Calling `finish(done)` with invalid output, stale inputs, missing required
  evidence, or a failed verifier never publishes an accepted result.
- Every accepted task has an `AcceptanceBundle` tied to exact revisions.
- Verifier failure produces actionable structured repair feedback.
- Required `unknown` outcomes result in `insufficient_evidence` or escalation.
- Existing domain pass/F1 metrics remain within the approved non-regression
  budget.

---

## 6. P0 — Transactional, versioned shared state

### 6.1 Objective

Agents and workflows must never observe a logically mixed artifact generation,
silently overwrite a stale base, or promote an unresolved semantic conflict.

### 6.2 Proposed contracts

```text
ArtifactRef {
  logical_key
  generation
  sha256
  media_type
  producer_run_id
  producer_task_contract_id
  source_digest
  harness_digest
  dependencies[]               // artifact refs and state revisions
}

ArtifactTransaction {
  transaction_id
  base_state_revision
  reads[]                      // key + expected generation
  writes[]                     // immutable staged ArtifactRef values
  assumptions[]
  verifier_obligations[]
  conflict_policy
}

ConflictRecord {
  conflict_id
  transaction_ids[]
  base_revision
  affected_resources[]
  candidate_refs[]
  conflict_kind                // content | semantic | delete/write | stale base
  resolution
  resolved_by
  reverification_refs[]
}
```

### 6.3 Commit protocol

1. Capture `source_digest`, `harness_digest`, input generations, and the current
   `state_revision` before execution.
2. Write new payloads under immutable content identities; they are not visible
   through logical artifact keys yet.
3. Execute required verifiers over the staged payloads.
4. Compare the transaction's expected reads and base revision with current
   state.
5. On a match, atomically compare-and-swap one logical bundle manifest and
   advance `state_revision`.
6. On a mismatch, produce a visible conflict or stale-state outcome. Do not
   overwrite the newer state.
7. Checkpoint only the committed bundle manifest and its complete identity.

This protocol must cover the task result triplet as one logical group. It
should also be reused by whole-map stores such as memories, vulnerability
reports, and verification records, or those stores should move to individually
versioned immutable rows.

### 6.4 Checkpoint restoration

A checkpoint entry is reusable only when all of the following match:

- workflow and task invocation identity;
- task-template version and content digest;
- source digest;
- input artifact generations and digests;
- relevant prompt, skill, tool-schema, configuration, policy, and model
  identities from the harness digest;
- committed output bundle digest; and
- any expiry or authorization constraints relevant to the task.

Artifact existence alone is insufficient. A stale entry should be retained for
audit but must cause re-execution rather than restoration.

### 6.5 Merge behavior

The current longest-content heuristic may remain available only as an explicit
legacy compatibility mode. In the target behavior:

- identical writes coalesce;
- disjoint structural writes merge deterministically;
- conflicting writes create `ConflictRecord` and are not promoted;
- a domain-specific resolver may propose a merged candidate;
- unresolved conflicts require human resolution or fail the affected task; and
- every merge invalidates and reruns verifiers whose scope intersects the
  changed resources.

### 6.6 Acceptance criteria

- Crash injection after any payload write exposes no mixed logical generation.
- Concurrent writers cannot lose a sibling update silently.
- A stale-base transaction is rejected visibly.
- A source or input change prevents checkpoint restoration.
- Parallel incompatible writes never select a winner solely by byte length.
- Resolved merges cannot commit without affected verifier evidence.

### 6.7 Compatibility migration

Artifact revisioning is a storage migration, not a silent replacement of the
current key/value contract. Preserve existing logical artifact names through a
resolution layer while new writers produce immutable generations and a bundle
manifest. Introduce a versioned checkpoint schema rather than reinterpreting
legacy entries in place.

Legacy artifacts may remain readable, but absence of a source digest,
transaction manifest, or acceptance bundle must remain visible. A legacy
artifact must not be relabeled as verified or revision-safe merely because its
content can be loaded. Explorer export, resume, cleanup, and downstream
artifact injection must be tested against both migration states.

---

## 7. P1 — Run manifests, replay, and harness evaluation

### 7.1 Immutable run manifest

Contractor already records rich events. Add a final manifest binding those
events to the complete execution identity:

```text
RunManifest/v1 {
  run_id
  parent_run_id?
  started_at
  ended_at
  outcome
  source_digest
  harness_digest
  model_identity
  model_parameters_digest
  workflow_config_digest
  prompt_versions_and_digests
  task_versions_and_digests
  skill_digests
  tool_schema_digest
  policy_digest
  sandbox_image_digest
  external_tool_versions
  input_artifact_refs[]
  output_artifact_refs[]
  approval_refs[]
  acceptance_bundle_refs[]
  event_stream_ref
  event_count
  final_event_sequence
}
```

Every event must carry `schema_version`, `run_id`, a monotonically increasing
sequence, and event time. Large content should remain in separately addressed
artifacts. Existing secret redaction must run before persistence.

Existing runtime events are best-effort telemetry and must not silently become
the authoritative audit log. Either add a separate durable audit sink, or give
the manifest/event journal explicit persistence guarantees while keeping
ordinary UI/metrics delivery non-blocking. The terminal manifest must reveal
any incomplete event journal rather than claiming a complete replay.

### 7.2 Replay levels

“Replay” should not imply that a nondeterministic model or external target will
produce identical live output. Define three explicit modes:

1. **Audit replay:** reconstruct state transitions, decisions, artifact
   identities, verifier inputs, and terminal reasoning from recorded events.
2. **Simulation replay:** feed recorded model/network/tool results through the
   harness to reproduce control-flow and state behavior.
3. **Live replay:** execute external actions again only under a new valid policy
   and approval; never as an automatic audit operation.

### 7.3 Harness metrics

Extend the existing evaluation envelope compatibly with a `harness` block and
aggregate raw counters across *all* pass@K attempts:

```text
harness {
  trajectory  { tokens, model_calls, tool_calls, retries, wall_ms, cost }
  verification { checks, failures, unknowns, oracle_types,
                 covered_scope, false_accepts }
  recovery    { invalid_actions, repaired_actions, dead_ends }
  consistency { stale_reads, stale_restores, conflicts,
                 revalidations, rollbacks }
  safety      { allows, denials, approvals, violations }
  replay      { manifest_valid, expected_events, present_events,
                 artifact_digest_failures }
}
```

Keep current domain metrics as separate headline outcomes. A safer harness that
cannot complete the task is not sufficient; a high-scoring workflow that
bypasses policy or accepts invalid output is also not sufficient.

Preserve the meaning of existing representative-attempt totals for historical
comparisons and add separate all-attempt totals. Evaluation aggregation must
partition or reject records with different `harness_digest` values; warning
while combining them would make the comparison uninterpretable.

### 7.4 Acceptance criteria

- Success, failure, timeout, cancellation, policy denial, and injected-crash
  runs all produce a valid terminal manifest.
- Every referenced artifact verifies by digest.
- Event sequences expose missing or duplicated events.
- Audit replay reconstructs every acceptance and policy decision.
- Secret-canary tests prove that protected values do not survive redaction.
- Evaluation reports total pass@K resource and safety cost, not only the first
  representative success.

---

## 8. P1 — Executable task contracts

`SubtaskSpec` currently carries most scope and expected-output information in
prose. Introduce an optional machine-readable contract shared by the planner,
policy gateway, transaction layer, and verifier stack:

```text
TaskContract {
  contract_id
  goal
  base_state_revision
  source_digest
  inputs[]
  read_set[]
  write_set[]
  dependencies[]
  assumptions[]
  invariants[]
  expected_outputs[]
  verification_specs[]
  risk_tier
  required_permissions[]
  required_approvals[]
  execution_budget
  rollback_revision
  success_criteria[]
  termination_criteria[]
}
```

### 8.1 Rollout

1. Add the fields as optional and retain current task templates.
2. In shadow mode, record declared versus actual reads, writes, destinations,
   permissions, and verifiers.
3. Report contract coverage and violations without blocking low-risk work.
4. Enforce external and privileged actions first.
5. Revalidate assumptions and input revisions before retry and commit.
6. Give direct `AgentRunner` workflows a synthetic top-level contract so they
   cannot bypass policy, state, or acceptance rules merely by omitting the
   planner.
7. Promote fields to required only after workflow-specific coverage is high
   enough to avoid misleading enforcement.

The contract externalizes intent without exposing or storing private model
chain-of-thought.

### 8.2 Acceptance criteria

- Contract declarations are revisioned and included in the run manifest.
- Actual high-risk writes and external actions cannot exceed declared scope.
- A retry cannot reuse an assumption invalidated by intervening state changes.
- Contract violations return structured repair information.
- Rollback restores the declared revision or fails visibly.

---

## 9. P1 — Governed memory and provenance-preserving compaction

### 9.1 Memory model

Separate memory by lifecycle rather than treating all notes equally:

| Kind | Purpose | Default lifetime |
|---|---|---|
| Working | Current trajectory, hypotheses, and immediate evidence. | Attempt or task. |
| Semantic | Repository structure and source-backed facts. | Source revision. |
| Experiential | Successful and failed strategies with outcome evidence. | Until revalidated or superseded. |
| Long-term | Deliberately promoted, validated project knowledge. | Explicit expiry/invalidation. |
| Shared | Coordination state and decisions visible across agents. | Run or workflow-defined scope. |

Every durable memory record should add:

```text
kind
source_digest
state_revision
provenance_refs[]
evidence_refs[]
confidence
validation_status            // candidate | verified | promoted | stale | revoked
validated_by[]
expires_at?
supersedes[]
invalidated_by[]
```

### 9.2 Lifecycle rules

- Agent-written experience begins as `candidate`.
- Promotion requires evidence from a successful accepted run or explicit
  review.
- Failures and unsuccessful strategies are retained when they prevent repeated
  mistakes; they are not rewritten as positive advice.
- Source changes invalidate dependent semantic and experiential memories.
- Retrieval filters by project, workflow, task, source revision, validation
  state, and applicable artifact generation.
- Concurrent mutation uses the same revision/CAS rules as other shared state.

### 9.3 Compaction

Generalize the existing practice of offloading large HTTP bodies: compact
context contains a bounded summary plus immutable references to the full
evidence, relevant locations, failed checks, open questions, and omitted
regions. A summary without resolvable provenance is not durable knowledge.

### 9.4 Acceptance criteria

- A source change prevents stale memory from being presented as current fact.
- Every promoted memory resolves to original evidence.
- Concurrent updates cannot lose a sibling record.
- Compaction preserves links to all evidence required by an acceptance bundle.
- Retrieval quality and token reduction are measured together.

---

## 10. P2 — Unified static and dynamic evidence graph

Contractor currently stores useful information across source files, OpenAPI
artifacts, call graphs, trace annotations, vulnerability records, verification
records, and HTTP request history. A later improvement is a revisioned evidence
graph that joins these views without replacing their authoritative stores.

Possible nodes:

```text
source snapshot, file, symbol, route, OpenAPI operation, call edge,
trace step, source, sink, control, finding, HTTP exchange, verifier check
```

Possible edges:

```text
defines, calls, maps_to, observed_in, supports, contradicts,
derived_from, verified_by, invalidated_by
```

Every node and edge must carry revision and provenance. This is experimental:
the paper identifies unification of repository structure and execution behavior
as an open systems problem rather than a solved reference design. Build it only
after artifact identity and transaction semantics are stable.

Acceptance should be based on concrete queries, for example:

- find live request evidence supporting a source finding at this exact source
  revision;
- list routes whose static trace changed since their last dynamic verification;
- identify accepted conclusions depending on a superseded artifact; and
- compute verifier coverage without reconstructing relationships from prose.

---

## 11. P2 — Governed harness evolution

Do not permit runtime agents to mutate production prompts, task templates,
tool schemas, permission rules, or verifier thresholds directly.

First introduce an offline `HarnessChange` proposal:

```text
HarnessChange {
  change_id
  base_harness_digest
  changed_components[]
  target_failure_mode
  causal_hypothesis
  predicted_effects
  preserved_invariants[]
  falsifying_evaluations[]
  held_out_suites[]
  acceptance_thresholds
  canary_plan
  rollback_target
  required_approvals[]
}
```

Promotion requires:

1. paired or interleaved comparison against a fixed baseline;
2. held-out domain and adversarial safety suites;
3. no regression in permission enforcement, false acceptance, artifact
   consistency, or redaction;
4. explicit human approval for policy, credential, network, or verifier
   changes;
5. a bounded canary; and
6. deterministic rollback to the prior harness digest.

An automated “evolution agent” may eventually propose changes and analyze
traces, but it must not approve or deploy its own production mutations.

---

## 12. Component impact map

| Proposal | Primary implementation areas | Specification chapters to update when implemented |
|---|---|---|
| Side-effect policy and approvals | `contractor/callbacks/`, `contractor/agents/worker_factory.py`, HTTP/code/Caido tools, workflow context, CLI approval input | [04](../spec/04-agents-callbacks-skills.md), [06](../spec/06-tools-and-filesystems.md), [07](../spec/07-artifacts-http-openapi-security.md), [08](../spec/08-cli-and-explorer.md), [09](../spec/09-configuration-observability-deployment.md) |
| Evidence-gated completion | task models/tools, `TaskRunner`, workflow verifier registration, finding/verification models | [03](../spec/03-runtime-orchestration.md), [04](../spec/04-agents-callbacks-skills.md), [05](../spec/05-workflows.md), [07](../spec/07-artifacts-http-openapi-security.md) |
| Transactional state | artifact service adapter, result publisher, checkpoint schema, memory/finding/verification stores, overlay merge | [03](../spec/03-runtime-orchestration.md), [06](../spec/06-tools-and-filesystems.md), [07](../spec/07-artifacts-http-openapi-security.md) |
| Run manifest and replay | runner events/plugins, CLI metrics sink, artifact export, analytics UI | [03](../spec/03-runtime-orchestration.md), [08](../spec/08-cli-and-explorer.md), [09](../spec/09-configuration-observability-deployment.md), [10](../spec/10-testing-and-acceptance.md) |
| Task contracts | runner and task models, planner tools, task-template schema, policy and verifier adapters | [03](../spec/03-runtime-orchestration.md), [04](../spec/04-agents-callbacks-skills.md), [06](../spec/06-tools-and-filesystems.md) |
| Governed memory | memory models/tools, artifact store, retrieval and compaction callbacks | [04](../spec/04-agents-callbacks-skills.md), [07](../spec/07-artifacts-http-openapi-security.md) |
| Harness evaluation/evolution | `tests/eval/`, metrics aggregation, version manifests, tuning scripts | [09](../spec/09-configuration-observability-deployment.md), [10](../spec/10-testing-and-acceptance.md) |

New configuration values must follow the existing ownership rules: workflow
budgets and per-agent options belong in workflow `config.yaml`; environment and
tool defaults belong in the central settings model. Safety policies should be
explicit run inputs with a stable digest, not scattered environment checks.

---

## 13. Fault-injection and acceptance matrix

These tests should compare the current baseline and proposed harness while
holding the model, prompts, task versions, fixtures, and sampling parameters
fixed.

| Fault or adversarial condition | Required result | Primary metric |
|---|---|---|
| Source changes after a checkpoint is written. | Restore is rejected and the affected task reruns. | `stale_restores = 0` |
| Failure after writing one or two members of a task result group. | No reader observes a committed partial generation. | visible partial commits `= 0` |
| Parallel forks write incompatible bytes to one path. | A conflict record is created; no heuristic winner is accepted. | silent conflict promotions `= 0` |
| Model reports `done` with invalid schema or missing evidence. | Result remains unaccepted and receives verifier diagnostics. | false accepts `= 0` for the injected suite |
| Verifier returns `unknown`. | Outcome is `insufficient_evidence` or escalation, never “safe.” | unknown-to-success conversions `= 0` |
| HTTP request targets an unauthorized origin. | Policy denial before any network effect. | off-scope packets `= 0` |
| Allowed URL redirects to loopback, link-local, metadata, or another origin. | Destination is re-evaluated and denied unless explicitly authorized. | redirect escapes `= 0` |
| DNS result changes after approval. | Every concrete destination remains within approved scope. | DNS scope escapes `= 0` |
| Code execution attempts arbitrary outbound traffic. | Default network isolation blocks it; scoped egress permits only approved targets. | sandbox egress violations `= 0` |
| Approval is expired, revoked, or bound to different arguments. | Action is denied or requests a new approval. | mismatched approval accepts `= 0` |
| Memory depends on a superseded source revision. | Retrieval rejects it or marks it explicitly stale. | stale promoted memories `= 0` |
| Process crashes, times out, or is cancelled. | Terminal manifest is valid and committed artifacts remain consistent. | manifest validity `= 100%` |
| Secret canary appears in a tool result. | Persisted events/manifests contain only permitted redacted form. | secret leakage `= 0` |

Zero-tolerance gates apply to authorization bypass, false acceptance in the
defined adversarial suite, silent state corruption, and secret leakage.
Latency, token, and cost budgets should be established from baseline
measurements per workflow rather than copied from the paper. Any quality
trade-off must be explicit; an initial promotion should require no material
domain pass/F1 regression.

---

## 14. Delivery sequence

The priorities above describe risk. The implementation order must also respect
dependencies between identity, policy, verification, and state.

| Phase | Deliverable | Enforcement level | Exit condition |
|---|---|---|---|
| 0 | Freeze representative domain fixtures; add the fault-injection suite and baseline harness metrics. | Observe only. | Baseline is reproducible with fixed harness inputs. |
| 1 | Add stable run/action IDs, source and harness digests, event sequence numbers, and draft schemas. | Observe only; backward-compatible serialization. | All current runs emit complete identities. |
| 2 | Add the policy gateway and approval ledger, starting with HTTP and sandbox networking. | Enforce external/privileged actions; shadow lower-risk classifications. | Adversarial network tests emit zero off-scope effects. |
| 3 | Add `candidate_done`, verifier registration, evidence bundles, and workflow-specific acceptance gates. | Enforce required deterministic verifiers. | Injected false acceptance is zero and domain quality is within budget. |
| 4 | Add immutable artifact generations, atomic bundle manifests, CAS state revisions, and strict merge conflicts. | Enforce publication and restore consistency. | Crash, stale-restore, and concurrent-writer suites pass. |
| 5 | Finalize `RunManifest/v1`, audit/simulation replay, and all-attempt harness evaluation. | Required for promoted runs. | Success and all failure modes are audit-replayable. |
| 6 | Roll out task contracts in shadow mode, then enforce high-risk declarations; add governed memory invalidation. | Progressive, workflow-specific. | Contract coverage and stale-memory tests meet promotion thresholds. |
| 7 | Prototype the evidence graph and offline harness-change workflow. | Experimental only. | Held-out and safety gates prove value over simpler state representations. |

The first three P0 outcomes—policy, verification, and transactional
state—should be treated as one hardening program even if their implementation
lands incrementally.

---

## 15. Risks and explicit non-goals

### 15.1 Risks introduced by this plan

- More verification can add latency and tool cost or reject valid but
  difficult-to-prove work.
- Incomplete plan contracts can create false confidence if declarations are
  trusted before coverage is measured.
- Transactional storage adds schema and migration complexity.
- Replay logs can expand the sensitive data footprint if redaction and
  retention are not designed first.
- Central policy becomes a high-value correctness and availability boundary.
- A weak deterministic oracle can make outcomes consistently wrong rather than
  merely variable.

Mitigate these through shadow deployment, explicit verifier scope, fail-closed
side-effect policy, immutable evidence, held-out fixtures, and reversible
schema migration.

### 15.2 Non-goals

This plan does not prioritize:

- adding specialist agents merely to increase parallelism or debate;
- replacing deterministic evidence with agent consensus;
- unrestricted agent-created reusable tools;
- autonomous mutation of production prompts, tools, policies, or verifiers;
- treating a larger context window or ungoverned vector store as memory
  architecture;
- claiming formal correctness from a passing test whose scope is narrower than
  the conclusion;
- browser/GUI automation unless Contractor's product scope expands to require
  it; or
- formal proof machinery except for future assurance-critical use cases with a
  concrete oracle and cost justification.

The paper's multi-agent analysis suggests that elaborate topology can
compensate for weak shared state. Contractor already has extensive role
specialization; strengthening the shared substrate has higher priority than
adding more roles.

---

## References

- [*Code as Agent Harness* — abstract and version metadata](https://arxiv.org/abs/2605.18747)
- [*Code as Agent Harness* — full PDF](https://arxiv.org/pdf/2605.18747)
- [*Code as Agent Harness* — accessible HTML](https://arxiv.org/html/2605.18747)
- [Plan–Execute–Verify control loop, §3.4](https://arxiv.org/html/2605.18747#S3.SS4)
- [Harness optimization and telemetry, §3.5](https://arxiv.org/html/2605.18747#S3.SS5)
- [Multi-agent shared state and convergence, §4](https://arxiv.org/html/2605.18747#S4)
- [Harness-level evaluation, §5.2.1](https://arxiv.org/html/2605.18747#S5.SS2.SSS1)
- [Scoped verification, §5.2.2](https://arxiv.org/html/2605.18747#S5.SS2.SSS2)
- [Regression-controlled harness evolution, §5.2.3](https://arxiv.org/html/2605.18747#S5.SS2.SSS3)
- [Transactional shared state, §5.2.4](https://arxiv.org/html/2605.18747#S5.SS2.SSS4)
- [Human accountability and durable intervention, §5.2.5](https://arxiv.org/html/2605.18747#S5.SS2.SSS5)
