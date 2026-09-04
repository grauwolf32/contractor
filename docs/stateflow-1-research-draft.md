# `stateflow@1` Planner research draft

Status: **RESEARCH DRAFT — NOT PRODUCTION-READY**

This document is non-normative. It does not define an executable configuration,
a registered Planner factory, a compatibility promise, or an implementation
commitment. The name `stateflow@1` is provisional. Production code and
configuration must not assume that this profile exists.

The purpose of this draft is to preserve a design hypothesis and the alternatives
that should be evaluated before a working agreement is written. If the experiment
is successful, the accepted design must be restated in the focused documents under
[`docs/spec`](spec/README.md), assigned implementation tasks, and reviewed against
the existing lifecycle, security, artifact and telemetry contracts.

## Source and motivation

The primary research reference is:

- Sanket Badhe, Priyanka Tiwari and Jonghyun Chung,
  [“SKILL.state: Scalable Long-Horizon Agent Skills,” arXiv:2608.26263v2](https://arxiv.org/pdf/2608.26263v2).

The paper replaces an append-only model conversation with an explicit mutable
execution state. At step `t`, the model receives only the immutable procedure `P`,
the current structured state `Σt`, and the latest observation `Ot`. It proposes a
validated state patch and one next action. Intermediate reasoning, older actions
and older observations do not enter the next model request.

This draft explores applying that idea at Contractor's **Planner-to-Worker macro
boundary**, rather than inside every Worker tool call:

```text
(immutable Stage procedure P, Planner state Σt, latest Worker observation Ot)
                                  |
                                  v
                    one Planner transition call
                                  |
                  validated patch + one next action
                                  |
                                  v
                    deterministic context builder
                                  |
                                  v
                         one Worker invocation
                                  |
                                  v
                    WorkerCompletion becomes Ot+1
```

In this mapping, one Worker invocation is a macro action. A Worker remains a
bounded micro-agent that can use several tools internally. The Planner owns the
state required to decide the next macro action.

“Skill” in the source paper does not mean Contractor's Agent Skill package.
Contractor Agent Skills are progressively disclosed guidance and resources, not
execution runtimes or state-schema owners. This proposal therefore belongs at the
Planner/Worker execution boundary and does not change the Agent Skills package
format by itself.

## Research hypothesis

A semi-deterministic Planner can maintain bounded structured Stage state and
construct each Worker request from that state, allowing long Stage executions to
avoid relying on the Planner's accumulated ADK conversation.

The expected benefits are:

- bounded Planner prompt size with respect to the number of completed macro steps;
- fewer repeated Worker dispatches caused by forgotten or contradictory history;
- explicit validation of the model's progress representation;
- clearer separation between immutable procedure, current state and untrusted
  observations;
- a natural checkpoint boundary after each completed Worker invocation;
- preservation of Contractor's fixed Worker allocation and Scheduler ownership.

The hypothesis is conditional. State and observation values must have explicit
size and cardinality limits. A fixed JSON schema containing an unbounded list does
not provide a bounded prompt. A structurally valid patch may also be semantically
wrong, so evaluation must measure state-loss and false-state failures rather than
token use alone.

## Non-goals for the first investigation

The first experiment should not attempt to provide:

- exact SKILL.state semantics inside every Worker tool call;
- arbitrary user-authored state schemas;
- transparent continuation of an in-flight Stage after Server process loss;
- exactly-once external side effects across a crash boundary;
- concurrent Worker dispatch or shared-state conflict resolution;
- automatic persistence of model-authored state into MemoryTools;
- a generic state query or mutation API for models, users or Operations;
- a replacement for Workflow Scheduler transitions;
- a replacement for artifacts, workspace state or domain systems of record;
- production defaults or migration of existing `passthrough@1`, `streamline@1`
  or `router@1` Stages.

## Proposed ownership boundary

The candidate `stateflow@1` implementation would be a new Planner factory behind
the existing framework-neutral Planner interface. Workflow Scheduler would still
prepare a fixed set of logical Workers, invoke one Planner for one StageExecution,
and accept exactly one validated terminal candidate.

The Planner would own:

- the bounded canonical Planner state for the active StageExecution;
- the latest bounded Worker observation presented to the Planner model;
- revision checks and state-patch validation;
- selection of exactly one dispatch or terminal action per transition;
- deterministic construction of the next `StageContentRequest`;
- Planner model, transition and Worker-call budgets;
- conversion of a terminal transition into a `StageContentResult` candidate.

The Planner would not own:

- physical Runtime Agent selection;
- mutable Workflow topology, retry or escalation policy;
- Worker tool execution or Worker-local budgets;
- artifact revision authority;
- workspace truth or external system truth;
- acceptance of a final Stage result.

The implementation should reuse the validated parts of `streamline@1`, especially
the plan controller, Worker invoker, artifact inspection, candidate validation,
execution reports and session fact recording. `passthrough@1` is a useful example
of the minimal PlannerFactory boundary, but its single-call lifecycle is not a
sufficient implementation base for a multi-step Planner.

## Candidate execution model

### Canonical inputs

Every Planner transition call would be constructed from exactly three logical
inputs:

```text
P  immutable Stage procedure
Σ  current bounded Planner state
O  newest bounded observation, or an explicit initial-observation marker
```

`P` may include only immutable Run-snapshot data required by the Planner:

- Stage objective and Stage instructions;
- fixed logical Worker names and descriptions;
- exact input artifact references and result contract;
- immutable string parameters;
- the transition/state schema and action rules.

`Σ` is the current validated state snapshot. `O` is one of:

- the initial Stage context marker;
- one validated `WorkerCompletion` projection;
- one bounded dispatch failure;
- one bounded state-transition validation error used for a permitted retry.

Older observations may affect a future decision only if a prior accepted
transition projected their relevant information into `Σ` or into an external
artifact referenced by `Σ`.

### Candidate state envelope

The first experiment should use one code-owned closed schema rather than generic
JSON Schema authoring. A possible shape is:

```json
{
  "schemaVersion": 1,
  "revision": 7,
  "phase": "verify",
  "completedWork": [
    {
      "id": "0",
      "status": "succeeded",
      "summary": "Located the HTTP router",
      "evidenceRefs": [
        {"namespace": "analysis", "name": "router-map", "revision": "..."}
      ]
    }
  ],
  "currentDispatch": null,
  "facts": [
    {
      "key": "api_entrypoint",
      "value": "internal/httpapi/router.go",
      "confidence": "observed",
      "evidenceRefs": []
    }
  ],
  "hypotheses": [],
  "blockers": [],
  "remainingWork": ["verify route registration"]
}
```

This is illustrative, not a proposed wire contract. Every string, collection,
nested object and complete encoded snapshot would need a bound. Large prose,
source content, diffs, HTTP bodies and analysis graphs belong in artifacts or
their owning external systems; state should retain only short facts and exact
references.

Planner execution state should initially remain separate from the exported
Contractor Worker State. The latter is a Runtime-authored metrics/observation
surface whose current contract deliberately excludes arbitrary model output and
complete tool payloads.

### Candidate transition envelope

The Planner model should propose one closed transition rather than a free-form
context string:

```json
{
  "expectedRevision": 7,
  "statePatch": {
    "phase": "verify",
    "remainingWork": ["verify route registration"]
  },
  "action": {
    "kind": "dispatch",
    "workerName": "reviewer",
    "objective": "Verify the discovered HTTP entrypoint",
    "instructions": "Check route registration and report contradictory evidence."
  }
}
```

Candidate action variants are:

```text
dispatch(workerName, objective, instructions)
finish(outcome, summary, artifact slot mappings, optional safe error)
```

A later experiment may split plan mutation from dispatch, but the first version
should prefer one transition per model call. The runtime would:

1. reject a stale `expectedRevision`;
2. validate the patch against the closed state schema and all size limits;
3. validate exactly one action and its relationship to the patched state;
4. record dispatch intent, never unconfirmed success, in state;
5. apply the patch and increment the authoritative revision;
6. deterministically build the Worker request or validate the terminal candidate;
7. execute the selected action;
8. present the bounded result as the next and only observation.

The model must not author an arbitrary serialized `workerContext`. If it can copy
history into a context string, prompt growth and authority mixing return under a
different field name.

### Deterministic Worker context construction

For a dispatch, trusted code would combine:

- the stored subtask objective and instructions selected by the transition;
- immutable Stage string parameters and exact artifact refs;
- a named, bounded projection of current Planner state;
- an explicit statement that the state projection is context, not a replacement
  for the Worker's mandatory AgentTemplate instructions;
- any result bindings already declared by Server.

For an initial spike, the bounded state projection can be encoded into the
existing task instructions. That avoids a wire change but provides weak type
separation and should not automatically become the production contract.

A production candidate may add a closed structured context member to
`StageContentRequest`. Such a change would require coordinated Go/Python wire
schemas, size limits, canonical encoding, compatibility tests and explicit rules
about which fields are model-visible. Another alternative is to pass one exact
state artifact ref, but making every Worker read its control state through a tool
adds latency and makes state availability depend on tool selection.

## Independent design axes

Worker history isolation, Worker-session continuity and Planner-state durability
are independent choices. They should not be bundled into one all-or-nothing
implementation.

### Axis A: Worker history isolation

#### A0 — Planner context only

The new Planner stops using accumulated conversation, but the existing Worker
runtime remains unchanged.

Advantages:

- Planner-only implementation;
- fastest way to validate state quality and Planner token behavior;
- no change to Worker Agent Skills or Worker session lifecycle.

Limitations:

- repeated calls to the same allocated Worker still use its allocation-scoped
  ADK session and may accumulate earlier A2A invocations;
- total system token use may remain superlinear even if Planner token use becomes
  linear;
- old Worker observations can still conflict with newly composed context.

This variant is suitable only as an instrumentation and state-schema spike.

#### A1 — Current-turn-only Worker context

The Worker excludes earlier A2A turns from each model request while retaining the
current invocation's user message, tool calls and tool results.

Advantages:

- small conceptual change to Worker construction;
- one Worker invocation remains a normal bounded ADK micro-agent;
- allocation-scoped clients, workspace and metrics can remain intact;
- previous Planner macro steps no longer enter the next Worker prompt.

Risks and questions:

- tool history still grows inside one large Worker invocation;
- an earlier `load_skill` tool result is absent from the filtered history;
  carrying its activation marker does not by itself restore the disclosed
  `SKILL.md` guidance, so the Worker must load it again or Runtime must provide a
  separately specified trusted projection;
- the exact behavior of the pinned Python ADK version must be covered by a
  Contractor regression test rather than assumed from an upstream option;
- ModelPolicy limits must keep one micro-episode bounded.

This remains a useful prompt-filtering comparison, but it does not provide the
exact fresh-session boundary selected for the preferred experiment below.

#### A2 — Fresh ADK session at a reset boundary

Runtime creates a fresh ADK session while retaining allocation-owned tools,
workspace, clients and metrics outside that session. It also carries eligible
ADK State forward between session epochs inside the same allocation; conversation
events and invocation-scoped `temp:` State are not carried. When the allocation
mode is `isolated`, this means one session per Worker invocation. A `shared`
allocation does not use this A2 boundary and retains one session until release.

Advantages:

- strongest separation between macro actions;
- conversation events can be destroyed immediately after one completion;
- behavior is explicit rather than dependent on ADK history filtering.

Costs and risks:

- Runtime needs an explicit allocation-scoped ADK State handoff between fresh
  session identities;
- snapshot, commit and merge behavior must exclude `temp:` State and remain
  correct on cancellation or a failed invocation;
- Agent Skill activation and other carried State need pinned-ADK regression
  coverage;
- cancellation, cleanup and instrumentation require additional lifecycle tests;
- session creation/destruction becomes part of every invocation boundary in an
  isolated allocation.

This variant is also required when the Stage contract promises a genuinely fresh
ADK session rather than only a prompt-history filter.

#### A3 — Exact per-tool state transitions

Every Worker tool call carries a state patch and only its result becomes the next
observation. This is closest to the source paper.

It requires wrapping every tool schema, coordinating patch application with side
effects, adapting Agent Skills, changing final-result production and handling
tool-level retry/idempotency. It is deliberately outside the initial
`stateflow@1` Planner investigation.

### Candidate allocation-scoped Worker session mode

This question applies to Streamline and Router as well as stateflow. The current
Python Worker creates one ADK session with the allocation ID and reuses it for
sequential A2A invocations. Because the Worker's `LlmAgent` uses ADK's default
history mode, a later subtask can receive model-visible events from earlier
subtasks in the same allocation.

The focused specifications currently establish that one allocation may serve a
sequence of A2A Tasks in
[Stage execution scope](spec/00-workflow-and-planner.md#stage-execution-scope)
and refer to the allocation's ADK session in
[Contractor-owned Worker State](spec/14-worker-results-and-live-state.md#contractor-owned-worker-state).
They do not explicitly choose whether Worker conversation continuity is isolated
per invocation or shared for the allocation. This draft records that as a contract
gap; it does not claim that the current implementation already supports both
modes.

The mode is fixed before Worker preparation. Stage configuration is resolved into
the immutable Run snapshot; Scheduler carries the selected mode into the
allocation requirements; Control Plane includes it in `AllocationSpec`; Runtime
enforces it for every invocation of that allocation. Planner receives the
prepared WorkerHandle and neither selects nor changes the mode.

An illustrative, non-executable Stage field is:

```yaml
session: isolated | shared
```

In the current Workflow shape, a stateflow Stage using the default isolated mode
would look like:

```yaml
spec:
  stages:
    analyze:
      objective: Analyze the supplied project
      instructions: {ref: instructions/analyze.md}
      planner: stateflow@1
      # session is omitted and resolves to isolated
      agents:
        analyst: {template: code_analyzer@1}
      # Existing context, result, workflowOutputs and on fields are unchanged.
```

Only deliberate conversational continuation needs an override:

```yaml
session: shared
```

The Stage-level value applies to every logical Worker allocation prepared for
that Stage. For newly authored Workflows, an absent field resolves to `isolated`
before the immutable Run snapshot is created. The resulting `AllocationSpec`
always contains an explicit resolved mode. Existing persisted snapshots that
predate the field must retain their former `shared` behavior through an explicit
legacy decoder, migration or version boundary; they must not be silently
reinterpreted as isolated. Per-Agent overrides should be added only if
representative Router workloads require different modes in one Stage.

The field name and encoding are provisional. `shared` is preferred to `common`:
it states that sequential invocations share one session, while `common` does not
identify the sharing scope. A more mechanical alternative is
`sessionScope: invocation | allocation`. Candidate meanings are:

- `isolated`: every Worker invocation starts in a fresh ADK session with no
  earlier conversation events; eligible ADK State is carried within the same
  allocation;
- `shared`: all sequential invocations of the logical Worker allocation reuse
  one ADK session until allocation release.

`isolated` is the default for newly authored Workflows: unrelated subtasks should
not inherit conversation accidentally. `shared` is useful when a sequence
deliberately builds on model/tool context from an earlier subtask.

The two fixed lifecycles are:

```text
isolated allocation: invocation 0 -> session 0
                     invocation 1 -> session 1
                     invocation 2 -> session 2

shared allocation:   invocation 0 -> session 0
                     invocation 1 -> session 0
                     invocation 2 -> session 0

next Stage:           new allocation -> no carried ADK State
```

Runtime generates opaque session IDs. Neither Planner nor `StageContentRequest`
provides a session mode, reset command or raw session identifier.

#### Capability and validation rules

The selected mode is part of the immutable Stage and allocation snapshots. A
Runtime Agent that does not support it must be rejected during placement or
allocation preparation; Server must not silently substitute the other mode.

For Router, each logical Worker has a separate allocation and therefore a
separate fixed mode, current session and carried State. Session replacement in an
isolated allocation occurs only between invocations; Runtime must not replace or
destroy a session while that Worker has an active A2A Task.

`passthrough@1` performs only one Worker invocation, so `isolated` versus `shared`
has no observable conversational distinction. Planner implementations require no
session-mode-specific tool or descriptor behavior.

#### Exact reset boundary

A Worker-session reset means a new ADK session identity and no conversation
events inherited from the previous epoch. Eligible ADK State is intentionally
carried between the fresh sessions when they belong to the same allocation.
Reusing the same session with `include_contents="none"` implements A1 prompt
isolation, but it is not equivalent to this stronger A2 reset contract.

Resetting the ADK session must not by itself replace the prepared Worker or
allocation. The following remain allocation-owned unless another explicit
contract resets them:

- WorkerHandle and logical Worker binding;
- model client and ModelPolicy budgets/configuration;
- selected tools and infrastructure adapters;
- workspace and sandbox contents;
- artifact grants and exact input/result bindings;
- Runtime-owned metrics and observation reducers;
- allocation-scoped domain clients such as an HTTP session, where selected.

This distinction must be visible in naming and documentation: a Worker ADK
session reset is not an allocation, workspace, sandbox, HTTP-cookie or domain
transaction reset.

The State handoff has a different boundary from conversation isolation:

- non-temporary ADK Session State may be copied into the next session epoch only
  within the same allocation;
- ADK `temp:` State remains scoped to one invocation and must not be copied;
- conversation events are never part of the State snapshot;
- exported Contractor Worker State remains canonical in `WorkerStateStore` and
  is rehydrated or reconciled from that store rather than treating an old ADK
  mirror as its authority;
- Agent Skill activation State may continue across the reset, but the old
  `load_skill` result does not; the current adapter must load the guidance again
  or a later contract must define an explicit trusted reinjection;
- carried State is discarded when the allocation is released and must never be
  imported into another allocation.

The implementation still needs one explicit snapshot/commit rule for partial,
failed and cancelled invocations. The reset contract must not accidentally turn
ADK State into a cross-allocation persistence mechanism.

#### Compatibility requirement

Today `streamline@1` effectively continues one allocation-scoped Worker ADK
session across subtasks. Changing that behavior in place would change prompts,
conversation context and potentially task results for existing Run snapshots. The
new authoring default is `isolated`, while snapshots created before the field
existed must retain `shared`. This requires an explicit legacy decoder, snapshot
migration or Workflow/schema version boundary. A new exact `AllocationSpec`
must not use field absence to decide between those two meanings. This research
draft does not select the migration mechanism.

#### AllocationSpec transport

No behavior-specific WorkerRuntime ref is proposed. A StageExecution prepares
fresh Worker allocations and releases them before another Stage executes, so an
allocation is already the exact lifetime boundary for carried ADK State. Session
continuity is Stage execution policy, not a different Worker implementation.

The resolved Stage mode should be copied into every affected `AllocationSpec`.
That immutable field fixes what Runtime does for the lifetime of the allocation.
For example:

```text
AllocationSpec.session = isolated | shared
```

`StageContentRequest` already exists as the strict Go/Python A2A payload for one
Worker invocation. Today it carries the subtask ID, objective, instructions,
string parameters, exact input ArtifactRefs and result bindings. It remains
unchanged: Worker session lifecycle is not task content and no per-invocation
decision crosses A2A.

Candidate validation is:

- Scheduler derives the selected mode from the immutable Stage snapshot and
  includes it in the allocation request path;
- Control Plane copies the mode into the exact `AllocationSpec` sent during
  preparation;
- Runtime rejects an absent, unknown or unsupported mode before constructing the
  Worker;
- `isolated` creates a fresh session for every `Worker.invoke` call, while
  `shared` creates one session and reuses it until release;
- Runtime generates opaque session IDs and never accepts one from Planner or
  task content.

#### Proposed Runtime call trace

The following trace applies after an allocation has already been prepared with
`AllocationSpec.session=isolated`. It names current Runtime entry points but the
session-lifecycle operations are proposed:

Implementation of the session-mode prerequisite is tracked independently from
the stateflow experiment in
[`V19-001`](../tasks/v19-001-worker-session-modes.yml).

```text
Scheduler / Control Plane
  -> prepare allocation with session=isolated
  -> Runtime constructs Worker, Runner, model, tools and one allocation-local
     SessionService
  -> Worker.start()
       does not create an invocation session
       initializes an empty allocation-local carried-State snapshot

A2A POST /private/v1/allocations/{allocation_id}/a2a
  -> AllocationA2AGateway
       resolves the active allocation application
  -> A2A request handler
  -> ContractorAgentExecutor.execute()
       verifies A2A task/context IDs and allocation tenant
       strictly decodes one StageContentRequest DataPart
  -> AdkWorkerRuntime.invoke(request)
       rejects draining/busy work before creating a session
       acquires the allocation's invocation lock
       creates a new invocation_id
       SessionLifecycle.begin_invocation()
         creates a new opaque session_id
         creates an ADK session from:
           eligible State carried from the prior invocation
           + a fresh canonical `contractor` snapshot from WorkerStateStore
           + no prior events
       prepares invocation metrics and budgets
       validates the bounded request
       _run_adk(request, invocation_id, session_id)
         renders the task prompt
         Runner.run_async(..., invocation_id=..., session_id=...)
         model/tool events and State deltas belong only to this ADK session
       validates the structured Worker result
       optionally exports workspace changes
       shielded invocation finalizer
         completes canonical WorkerStateStore state
         mirrors its latest `contractor` snapshot into the active ADK session
         snapshots a bounded deep copy of eligible non-temporary ADK State
           for the next invocation
         deletes the ADK session, including all of its events
       releases the invocation lock
       returns WorkerCompletion(invocationId, stateRevision, result|failure)
  -> ContractorAgentExecutor completes or fails the A2A Task
```

Consequently, `allocation_id`, A2A `task_id`/`context_id`, Contractor
`invocation_id` and ADK `session_id` belong to separate identity domains. In
isolated mode every accepted invocation gets a different `invocation_id` and a
different `session_id`; only the allocation identity, prepared Worker resources
and eligible State continue. The next request repeats `begin_invocation` with
zero inherited events.

The allocation-owned lifecycle helper needs only three pieces of state: the
fixed mode, an optional active session ID and the carried State snapshot. It
should expose operations equivalent to `start`, `begin_invocation`,
`finish_invocation` and `close`. `_run_adk` and State synchronization should
receive the active session ID explicitly rather than read a mutable Worker-wide
ID.

The finalizer must cover the whole accepted invocation, not only
`Runner.run_async`. Model, tool, budget, result-validation and workspace-export
failures still complete the canonical invocation record, snapshot any eligible
State already committed by ADK and delete the isolated session. Cancellation
must shield that bounded cleanup. If State extraction or session deletion fails,
the Worker should stop accepting work rather than run the next invocation from
an ambiguous State epoch.

In isolated mode, requests rejected for a stale route, wrong tenant, invalid
`StageContentRequest`, draining Worker or busy Worker do not create an ADK
session. Runtime may still update canonical rejection metrics; in isolated mode
there is no active ADK session that needs a mirrored metrics event.

For `session=shared`, the outer A2A trace is the same. The lifecycle helper
instead lazily creates one session during the first admitted `begin_invocation`,
returns that same ID from later calls, retains its State and events after
`finish_invocation`, and deletes it only at allocation release.

This requires a coordinated extension of the Go and Python `AllocationSpec`
schemas, Scheduler/Control Plane construction and strict cross-language fixtures.
It does not change the Worker A2A payload or Planner tool schemas.

Runtime support must still be discoverable during placement. The current
capability snapshot distinguishes only exact WorkerRuntime refs, so a production
rollout must either increment the generic ADK runtime contract or advertise a
separate closed session-control feature. It must not encode isolated/shared as two
behavior-named Runtime factories.

At allocation release, Runtime already destroys the Worker, SessionService,
prepared skills and allocation-owned resources. No ADK State is transferred to a
new Stage allocation.

### Axis B: Planner-state durability

#### B0 — Volatile StageExecution state

Canonical state lives only inside the active Planner process. Existing reduced
plan/events remain durable, but an interrupted Planner is not resumed. Workflow
policy may start a fresh StageExecution.

Advantages:

- preserves the current partial-invocation recovery boundary;
- no new database or artifact contract;
- cheapest and safest experiment.

Limitations:

- a process failure loses accepted semantic patches that were not externalized;
- retry must reconstruct useful state from immutable Stage context, artifacts and
  other already durable domain state.

This is the preferred first variant.

#### B1 — Checkpoint between completed macro steps

After a transition and its WorkerCompletion have both reached a safe boundary,
Planner stores a bounded state snapshot plus correlation metadata. Recovery may
continue only from such a completed boundary; an in-flight dispatch is never
assumed complete.

Open design work includes:

- whether the checkpoint belongs in Planner session state or a reserved artifact;
- versioning, canonical encoding and maximum size;
- whether a new process may resume the same StageExecution under current ownership
  and fencing rules;
- how a completed Worker side effect whose response was lost is represented.

B1 is a possible second research increment, not a requirement for the first
prototype.

#### B2 — Exact mid-stage resume

Recovery preserves an in-flight transition and guarantees safe continuation even
when a Worker performed a side effect before Planner persisted its completion.

This requires durable dispatch identities, response-loss handling, idempotency or
reconciliation for every side-effecting Worker operation, and integration with
Stage cancellation/fencing. It is high-risk and explicitly not production-ready
in this draft.

### Axis C: state authorship

#### C0 — Fully deterministic reducer

Code derives all state changes from known typed Worker outcomes. This provides the
strongest correctness but works only for narrow domain-specific workflows whose
result vocabulary is known in advance.

#### C1 — Model-proposed, runtime-validated patch

The Planner model proposes a patch from current state and one observation. Runtime
validates structure, bounds, revision and allowed transitions. This is the main
research candidate: semi-deterministic rather than fully deterministic.

Structural validation cannot prove that a fact is true. Candidate schemas should
distinguish observed facts, hypotheses, decisions and pending intent, and retain
evidence refs where possible.

#### C2 — Worker-proposed patch

Worker returns a semantic state patch together with its result. This reduces a
Planner model call but makes every Worker template understand the Planner state
schema, weakens separation between execution and orchestration, and does not fit
the current two-field Worker model result cleanly. It is not recommended for the
first experiment.

### Axis D: context transport

#### D0 — Encode into existing task instructions

Useful for a spike. It requires no wire change but must use unmistakable,
deterministically rendered sections and the existing request-size limit.

#### D1 — Add typed Planner context to `StageContentRequest`

The likely production direction if the experiment succeeds. It preserves
authority and type separation but changes public cross-language contracts and
therefore requires a formal specification increment.

#### D2 — Store state as an exact artifact

Useful when Workers need a larger immutable snapshot, but unsuitable as the only
control path because reading it requires an appropriate selected tool and consumes
an additional model/tool turn. State should still remain bounded; artifact storage
must not be used to hide unbounded conversational history.

## Preferred first experiment

The recommended combination is:

```text
A2 via AllocationSpec session=isolated for the fixed baseline
B0 volatile StageExecution state
C1 model-proposed, runtime-validated patch
D0 existing task instructions for the spike
```

The Planner implementation would avoid a long-running ADK conversation. It would
make one independent Planner model request for each transition, containing `P`,
`Σ` and `O`, and accept one typed transition function call or structured result.

The initial experiment should use:

- one code-owned state schema;
- one Workflow/Stage family with an existing fixed Worker set;
- sequential dispatch only;
- no cross-retry state continuation;
- a conservative state size such as 32–64 KiB, subject to measurement;
- existing exact ArtifactRefs for large outputs and evidence;
- existing Stage/Planner/Worker model, token, call and wall-time limits;
- explicit metrics for transition retries and state quality.

The first implementation should remain opt-in and must not silently reinterpret
an existing Planner selector.

Worker session mode is fixed by allocation rather than selected by the Planner.
The benchmark should run otherwise equivalent `isolated` and `shared`
allocations and compare them without changing the stateflow action schema.

## Failure and cancellation semantics to investigate

### Invalid Planner transition

A malformed, stale, oversized or schema-invalid transition must not mutate state
or dispatch a Worker. The experiment should compare:

- immediate retryable Planner failure;
- one bounded corrective retry using the same `P`, `Σ` and a safe validation-error
  observation;
- grammar/constrained decoding when supported by the selected model path.

There must be no unmetered repair loop.

### Worker failure

A valid `WorkerFailure` becomes the next bounded observation. Planner may propose
a corrective dispatch or terminal failure subject to its existing limits.
Workflow retry/escalation remains Scheduler-owned.

### Dispatch intent versus success

The pre-dispatch state patch may record only intent and correlation, for example:

```json
{
  "currentDispatch": {
    "callId": "dispatch-0042",
    "workerName": "reviewer",
    "purpose": "verify_api_entrypoint",
    "status": "pending"
  }
}
```

It must not claim that the Worker succeeded. The next transition resolves the
pending intent from the trusted completion or bounded failure.

### Cancellation

Cancellation must stop new Planner model and Worker calls, cancel an active Worker
call best-effort, and follow existing Stage abort/fencing behavior. Stateflow must
not add an independent lifecycle state machine outside Scheduler and Planner
session ownership.

## Candidate invariants

These are research guardrails, not yet normative requirements:

1. Immutable procedure `P` is never rewritten by model output.
2. A Planner model call receives one current state and at most one latest
   observation, not reconstructed conversational history.
3. State has a closed schema, positive revision and complete encoded size bound.
4. Every transition names the revision it read and selects exactly one action.
5. Runtime validates a transition before any Worker side effect starts.
6. A pre-dispatch patch records pending intent, never unobserved success.
7. Worker context is built by trusted code; the model cannot submit an arbitrary
   context transcript.
8. Worker output is untrusted semantic input even after wire validation.
9. Exact ArtifactRefs and domain systems remain the authority for durable data.
10. Operational events may remain available for audit while being excluded from
    future model prompts.
11. State never contains credentials, headers, RuntimeSettings, physical Runtime
    identity or hidden reasoning.
12. Planner/Worker budgets and Scheduler lifecycle rules remain independently
    enforceable.
13. Scheduler carries the immutable Stage session mode into each allocation, and
    Runtime applies it uniformly to every invocation of that allocation.
14. Planner supplies neither a session mode nor a session identifier.
15. Session reset occurs only between Worker invocations and does not reset the
    allocation, workspace or external domain state implicitly.
16. Conversation events and `temp:` State never cross a reset boundary; carried
    ADK State never crosses an allocation boundary.

## Relationship to existing features

### `passthrough@1`

Passthrough deterministically invokes one Worker and maps one completion. It is a
useful minimal factory/lifecycle example. Stateflow differs by running a bounded
multi-step transition loop and possibly using a Planner model.

### `streamline@1` and `router@1`

These are the closest existing behavioral relatives. They already provide bounded
typed subtasks, sequential Worker dispatch, exact logical Worker selection and a
validated `finish` operation. Their live model reasoning still uses an ADK
conversation. Stateflow should reuse their domain controllers while replacing the
conversation as the primary reasoning substrate.

The current `streamline@1` Worker path reuses one allocation-scoped ADK session
across subtasks. The desired design is isolated-by-default with an explicit Stage
mode that may instead select a shared allocation session. Planner behavior and
tool schemas are identical in both modes. This is a proposed versioned contract,
not a description of current `streamline@1` behavior.

Router-style logical Worker selection could become a later stateflow action
variant. The first experiment should prefer one Worker binding unless multi-Worker
routing is essential to the selected evaluation workload.

### MemoryTools

Memory notes are explicit durable coordination artifacts and are not automatically
injected into model context. Stateflow must not silently reinterpret MemoryTools as
its canonical state store. A trusted checkpoint design may use a reserved artifact
in the future, but that requires its own contract.

### Agent Skills

Agent Skills remain progressively disclosed Worker guidance. Under current-turn
Worker isolation, a later invocation cannot rely on an old `load_skill` tool
result remaining in conversation context. Its activation marker may continue in
carried ADK State, but the current Contractor adapter exposes a fixed tool surface
and does not reconstruct the disclosed `SKILL.md` result from that marker. State
carry-over alone is therefore insufficient: an isolated invocation must load the
guidance again unless a later contract defines a bounded trusted reinjection.

### Worker State and Planner projection tools

Runtime-owned Worker State records metrics and typed observations. Planner
projection tools expose narrow read-only views. Neither is currently a mutable
model-authored execution state. They may provide trusted observation inputs to a
future stateflow context builder without becoming the canonical `Σ` automatically.

### Terminal Worker summarization

Terminal summarization stops a long Worker and emits one final result. Stateflow
instead continues a Stage through multiple bounded Worker macro actions. The two
mechanisms are complementary and should have separate metrics and configuration.

## Evaluation plan

Token reduction alone is insufficient. The experiment should compare an existing
Planner baseline and stateflow under the same models, tool selection, task inputs,
budgets and deterministic decoding settings where available.

Required measurements should include:

- prompt tokens per Planner transition and per Worker model call;
- cumulative Planner, Worker and total tokens;
- task/Stage success rate;
- repeated Worker dispatches and repeated tool actions;
- invalid, stale, oversized and semantically rejected state patches;
- facts lost because they were not projected into state;
- facts incorrectly retained after contradictory observations;
- state encoded size and per-field cardinality over time;
- Worker context size over time;
- configured Worker session mode and observed session-epoch lengths;
- quality and token deltas between `isolated` and `shared` allocations;
- corrective transition count;
- side effects repeated after failures or cancellations;
- latency per macro step and complete Stage duration;
- behavior with irrelevant noisy Worker results;
- behavior when the relevance of an earlier observation becomes apparent later.

Useful evaluation horizons include 10, 25, 50 and 100 macro steps, while Worker
micro-episode budgets remain fixed. At least one workload should exercise source
analysis or application-security behavior rather than only a synthetic inventory
state.

The experiment should report Planner and Worker slopes separately. A bounded
Planner prompt does not prove bounded total execution if the reused Worker's
session still accumulates earlier invocations.

## Decision gates

The proposal should not advance to a working specification unless an evaluation
shows all of the following:

- Planner prompt size stabilizes after initialization;
- total token growth is materially closer to linear over the tested horizon;
- success quality is not materially worse than the selected baseline;
- invalid patch behavior is bounded and observable;
- state omissions and false-state errors are understood and acceptable for the
  selected domain;
- no test demonstrates duplicate external side effects caused by stateflow retry;
- Worker context isolation is verified against the exact pinned ADK build;
- every configured Worker session mode is either supported explicitly or
  rejected during placement or preparation, with no implicit fallback;
- session mode affects only the documented ADK-session boundary;
- state and observation payloads remain within explicit limits without covert
  transcript accumulation;
- telemetry and durable events contain no newly exposed model content or secrets.

Failure to satisfy these gates should leave existing Planner profiles unchanged.

## Rough engineering estimates

These are exploratory estimates for one engineer familiar with Contractor, not
delivery commitments:

| Scope | Rough effort |
|---|---:|
| Domain-specific Planner spike, fixed schema, A0/B0/C1/D0 | 1–2 weeks |
| Add and verify A1 Worker prompt isolation | 1 additional week |
| Stage/Scheduler/AllocationSpec mode plus ADK session lifecycle | 1–2 additional weeks |
| Reusable opt-in profile, metrics, failure tests and representative evals | 4–8 weeks total |
| B1 safe-boundary checkpoint research | 2–4 additional weeks |
| B2 exact mid-stage resume | 3–5+ additional weeks after B1 |

Arbitrary schema authoring, UI configuration and migration of existing Workflows
would expand this estimate and should be considered only after the fixed-schema
experiment succeeds.

## Open questions

1. Is one code-owned state schema sufficient for the first representative
   Contractor workload?
2. Should stateflow own a separate bounded plan, extend `PlannerPlanController`,
   or compose an immutable plan projection with a separate working-state object?
3. Does the Planner model return a structured result or call one synthetic
   `transition` function?
4. Is one corrective transition retry useful, or does it hide model/schema
   incompatibility?
5. Which state fields may be projected into Worker context, and must projections
   differ by logical Worker?
6. Can A1 reliably isolate prior A2A turns with the pinned Python ADK while
   preserving current-turn tool execution?
7. Should the final Stage spelling be `session: isolated | shared`, or the more
   mechanical `sessionScope: invocation | allocation`?
8. Is the mode Stage-wide or configurable independently for each logical Worker
   binding?
9. Should session-control support be advertised through a generic ADK runtime
   contract increment or a separate closed Runtime capability?
10. Which legacy decoder, snapshot migration or schema-version boundary maps old
    snapshots without a resolved session mode to `shared`, while omission in new
    Workflow authoring resolves to `isolated`?
11. What snapshot and commit boundary carries eligible ADK State into a fresh
   session while reliably excluding invocation-scoped `temp:` keys?
12. Must an isolated invocation load active Agent Skill guidance again, or should
   Runtime define a bounded trusted reinjection independent of event history?
13. Is B0 acceptable for the first production candidate, or is safe-boundary
   checkpointing a prerequisite?
14. Which facts should be runtime-derived from trusted artifact/observation data
   instead of model-authored?
15. What state size and collection bounds produce a useful but genuinely bounded
    prompt for code-analysis and application-security tasks?

## Path from research draft to production

If the preferred experiment passes its decision gates:

1. record the evaluation fixtures and findings;
2. resolve the open schema, transport, retry and durability choices;
3. write focused normative changes for Workflow/Planner, Runtime/A2A, lifecycle,
   metrics and any wire contracts;
4. create ordered implementation tasks with explicit compatibility and security
   acceptance criteria;
5. register a final Planner ref only when Server descriptors, configuration
   validation, Runtime compatibility and end-to-end tests exist;
6. retain existing Planner behavior for every previously valid Run snapshot.

Until those steps complete, `stateflow@1` remains only a research name in this
document and is not production-ready.
