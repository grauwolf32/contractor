# 00 — Workflow Scheduler and Planner

Status: **Working agreement**

Depends on: none

## Goal

Workflow defines the product-specific sequence or graph. Workflow Scheduler
executes that graph and owns Stage lifecycle. Planner owns decisions inside one
prepared Stage. Worker runtimes and their heavy dependencies remain outside
Server, Scheduler and Planner packages.

The execution model has no Server-side Task DAG in addition to the Workflow.
Planner may keep a private plan, but the durable outer unit is a
`StageExecution`: one attempt with at most one Planner invocation and exactly
one terminal `StageResult` or `StageTermination`. Retry creates another
StageExecution rather than reopening the previous one. Lifecycle, conditional
cardinalities and recovery are owned by
[04](04-execution-lifecycle-and-metrics.md).

## Workflow, Scheduler and Stage

One Workflow invocation creates one `WorkflowRun` (`Run` below). A Workflow
definition declares:

- named string input parameters with required/optional status;
- named input artifact slots, including required/optional status and accepted
  media types;
- named output artifact slots and their acceptance contract;
- Stages, dependencies and product transition/escalation rules.

Workflow definitions are YAML files. At Server startup `WorkflowCatalog` loads,
parses and validates them into internal DTOs; the baseline has no hot reload.
Creating a Run stores the complete validated Workflow snapshot used by that Run,
including its resolved Planner factory refs and exact digest-bearing
AgentTemplate, ModelPolicy and LLMGatewayConfig dependencies plus the resolved
`executionConfig`, so later file edits cannot change its graph or execution
contract. Every Workflow has an exact configuration key `(name, version)`,
while the durable snapshot rather than a separately computed Workflow digest
is the execution authority in the first slice.

### Configuration files

The configuration/UI slice uses one operator-configured root with five fixed
manifest subtrees plus instruction resources:

```text
configs/
  workflows/          # Workflow YAML manifests
  agent-templates/    # AgentTemplate YAML manifests
  model-policies/     # ModelPolicy YAML manifests
  llm-gateways/       # LLMGatewayConfig YAML manifests; never secret values
  instructions/       # UTF-8 instruction resources
```

The local loader recursively discovers regular files ending in `.yaml` below
the first four subtrees. Each file contains exactly one non-empty YAML
document; multi-document streams are invalid. Its `kind` must match its
subtree. The loader rejects duplicate YAML mapping keys and validates the exact
schema selected by `apiVersion` and `kind` rather than retaining unknown
fields.

Every YAML manifest has the envelope `apiVersion`, `kind`, `metadata` and
`spec`. `metadata` contains mandatory `name` and `version`, using respectively
the shared selector `id` and `version` grammars below. The configuration lookup
key is `(kind, metadata.name, metadata.version)`. File name and nesting below
the kind subtree are organizational only; two files with the same lookup key
make the complete configuration set invalid.

Configuration loading is all-or-nothing. Server first loads and validates
ModelPolicies and LLMGatewayConfigs, then resolves AgentTemplates, and finally
resolves Workflows. It publishes no partially resolved in-memory configuration
set if any manifest, instruction ref or cross-document selector is invalid. The
baseline performs this process only at startup.

Instruction resources retain a different identity rule: their normalized path
relative to the configuration root, such as
`instructions/openapi/planner.md`, is the ref embedded into the resolved parent
document. They need not be YAML and are not indexed by `kind/name/version`.
Toolsets, SandboxProfiles, WorkerRuntime factories and Planner factories are
registered code plus Server-visible descriptors, not additional configuration
subtrees.

A future S3-backed loader maps the same five logical manifest/resource
subtrees to prefixes and
preserves the same document keys and relative instruction refs. No YAML
manifest may depend on a local absolute path, inode or file name for its
identity.

### ExecutionConfig defaults and Run overrides

Workflow supplies reference-only execution defaults, while a Run request may
override those defaults with other already published configurations. Neither
surface accepts an inline URL, token, model name, budget or provider-specific
parameter. The common authoring shape is:

```yaml
spec:
  executionConfig:
    planner:
      modelPolicy: planner-balanced@1
      llmGateway: local-lm-studio@1
    workers:
      llmGateway: local-lm-studio@1
    stages:
      build:
        planner:
          modelPolicy: planner-strong@1
          llmGateway: paid-gateway@1
        agents:
          builder:
            modelPolicy: domain-worker-strong@1
            llmGateway: paid-gateway@1
```

`planner` and `workers` are optional Workflow-wide defaults. `stages` is an
optional mapping keyed by an existing Stage name; each `agents` entry is keyed
by an existing logical binding in that Stage. Every leaf contains only optional
exact `<id>@<version>` selectors `modelPolicy` and `llmGateway` and must contain
at least one of them. Unknown fields, Stages, bindings or configurations are
invalid.

`POST /v1/runs` accepts an optional top-level `executionConfig` object whose
content has the same `planner`/`workers`/`stages` shape. It is an override
document, not a way to author configuration. Server resolves one effective
configuration for every modeled consumer in this order, with later values
winning:

1. the AgentTemplate's ModelPolicy for a Worker;
2. Workflow-wide execution defaults;
3. Workflow Stage/binding defaults;
4. Run-request-wide overrides;
5. Run-request Stage/binding overrides.

Every `adk@1` Worker must resolve one compatible ModelPolicy and one
LLMGatewayConfig. Every `streamline@1` Planner must independently resolve one
compatible ModelPolicy and one LLMGatewayConfig. `passthrough@1` has no model
client, so a Planner model/Gateway selection for that Stage is invalid. The
consumer-specific ModelPolicy requirements are defined in
[01](01-agent-template.md).

Run initialization resolves complete ModelPolicy and LLMGatewayConfig bodies,
their exact digest-bearing refs, and the non-secret exact credential revision.
It stores the fully expanded per-Stage/per-binding `ResolvedExecutionConfig`
with the immutable WorkflowRun snapshot. That snapshot, rather than later
configuration edits or UI state, is authoritative for every attempt. Secret
bytes are never stored in the Run snapshot; Control Plane resolves the pinned
credential revision only when constructing RuntimeSettings or the Planner
model client.

The canonical idempotency digest for `POST /v1/runs` includes the supplied
executionConfig selectors. Metrics and audit records identify the effective
ModelPolicy, LLMGatewayConfig and credential refs, allowing consumption to be
grouped without exposing a token. A caller may select only published
configurations it is authorized to use; the single-owner first UI slice exposes
all active published configurations, while multi-tenant policy remains
deferred.

The first schema is an explicit state-machine graph with a versioned document
envelope, one entry Stage, a mapping of stable Stage names and Stage-local typed
transitions:

```yaml
apiVersion: contractor/v1alpha1
kind: Workflow

metadata:
  name: openapi-analysis
  version: "1"

spec:
  parameters:
    objective:
      required: true
    mode:
      required: false

  entryStage: build
  stages:
    build:
      objective: Build an OpenAPI description from the service source
      instructions:
        ref: instructions/build-openapi.md
      planner: passthrough@1
      agents:
        builder:
          template: oas_builder@2
      on:
        succeeded:
          next: review
        failed:
          fail: {}
        interrupted:
          retry:
            maxAttempts: 2
            then:
              fail: {}

    review:
      objective: Review and improve the produced OpenAPI description
      instructions:
        ref: instructions/review-openapi.md
      planner: streamline@1
      agents:
        reviewer:
          template: oas_reviewer@1
      on:
        succeeded:
          succeed: {}
        failed:
          fail: {}
        interrupted:
          fail: {}
```

`entryStage` must name one key in `stages`. Each transition selects exactly one
typed action: `next`, `retry`, `succeed` or `fail`. `next` names another Stage;
`retry` creates a new StageExecution for the same Stage rather than reopening
the prior attempt. Workflow cancellation is not a graph edge: once Run is
`cancelling`, Stage cancellation does not evaluate `on` transitions.

In `v1alpha1`, `on` must contain exactly the three keys `succeeded`, `failed`
and `interrupted`, each with one action object. There are no ordered rules,
error-code predicates, expression language or implicit fallback. A StageResult
selects `succeeded` or `failed`; a StageTermination selects `interrupted`.
`cancelled` is intentionally absent because it is handled only by the enclosing
WorkflowRun cancellation path.

Action validity is constrained by outcome:

- `succeeded` allows `next` or `succeed`;
- `failed` allows `next`, `retry` or `fail`;
- `interrupted` allows `next`, `retry` or `fail`.

Every `next` target must exist, and every graph path must reach `succeed`,
`fail` or a bounded retry exhaustion action. A future schema version may add
ordered predicates without changing snapshots validated as `v1alpha1`.

### Versioned selectors

The Workflow selector accepted when creating a Run, the `planner` field and
every Agent binding's `template` field use the same compact authoring syntax:

```text
<id>@<version>
```

Its first-slice grammar is:

```text
id      = [a-z][a-z0-9_-]*
version = [A-Za-z0-9][A-Za-z0-9._+-]*
```

A selector contains exactly one `@`. Both parts are non-empty; whitespace and
`/` are consequently invalid. Version comparison is opaque, case-sensitive
exact equality: Contractor does not interpret semantic-version ordering.
`latest`, version ranges and an omitted version are invalid. This selector is
an authoring/API lookup value only, never the complete durable execution
identity.

At Run creation, WorkflowCatalog resolves the supplied selector to the exact
Workflow document whose `metadata.name` and `metadata.version` match it. The
validated and dependency-resolved Workflow snapshot is then stored with the
Run; Scheduler never consults configuration files to reinterpret that Run.

WorkflowCatalog resolves `planner` through the configured PlannerFactory
registry to an exact `(planner_id, version)` implementation ref. It resolves
each `template` through AgentTemplateCatalog to an exact
`AgentTemplateRef(template_id, version, digest)` and validated template body.
Unknown or ambiguous selectors make the Workflow invalid. The normalized Run
snapshot stores the resolved refs and template dependencies; Scheduler never
re-resolves the authoring strings while executing that Run.

### Workflow parameters

`spec.parameters` is a mapping of stable parameter names to string-slot
contracts. Each contract contains exactly one mandatory boolean field,
`required`; it has no default. `v1alpha1` intentionally has no parameter type
declaration because every accepted parameter value is a string.

The Run request supplies a string mapping separately from its artifact-slot
bindings:

```yaml
parameters:
  objective: Find authorization vulnerabilities
  mode: strict

artifacts:
  source:
    namespace: projects
    name: payment-service-source
```

Before the Run becomes schedulable, Server rejects missing required parameters,
unknown names and every non-string value, including numbers, booleans, arrays,
objects and null. It performs no coercion, trimming, interpolation or structured
parsing; an empty string is a valid supplied string. The exact validated mapping
is stored durably with the Run and is immutable for its lifetime. Retrying a
Stage does not re-read or modify it.

Every Planner receives the complete parameter object as read-only
`StageContext` data. `v1alpha1` has no Stage-local parameter projection or
expression language. Parameters are not automatically included in
`AllocationSpec` or delivered to a Worker: Planner deliberately selects what to
put in each A2A request.

Parameters are ordinary persisted Run data, not a secret channel. Callers must
not place credentials or provider tokens in them; deployment and
allocation-scoped secrets enter Runtime Agent only through `RuntimeSettings`.
Structured, repeated or binary input belongs in a declared Workflow artifact
slot, using an appropriate media type such as `application/json`, rather than
inside the parameter contract. Contractor does not inspect a string merely
because its contents happen to resemble JSON.

### Retry action

The only `v1alpha1` retry form is:

```yaml
retry:
  maxAttempts: 3
  then:
    next: fallback
```

`maxAttempts` is an integer greater than or equal to two and counts the initial
StageExecution. Thus `maxAttempts: 3` permits at most two new retry executions.
Every retry creates a new StageExecution with the next attempt number, a
`previous_execution_id` pointing to the prior attempt and, after successful
preparation, new allocations, Planner instance and Planner Session.

Both StageError and StageTermination carry a required boolean `retryable`.
`retry` creates the next attempt only when that value is `true`, the attempt
limit is not exhausted and WorkflowRun is still `running`. Otherwise Scheduler
immediately evaluates `then`. `then` contains exactly one `next` or `fail`
action; nested retry and successful termination are invalid there.

`retryable` does not trigger retry by itself. A `failed` or `interrupted`
transition using `next` or `fail` follows that declared action regardless of the
flag. Workflow `v1alpha1` has no retry delay, backoff, error-code predicate or
runtime override. Capacity backoff while preparing one attempt is an
infrastructure concern and does not consume another Workflow attempt.

Arbitrary cycles through `next` are rejected in `v1alpha1`; the only permitted
cycle is the explicitly bounded `retry` action. This leaves a direct mapping to
a future block editor without introducing a second Scheduler DAG.

### Stage objective and instructions

Every Stage declares a mandatory non-empty literal `objective` and one mandatory
Planner-instruction resource ref:

```yaml
objective: |
  Produce an OpenAPI description that matches the service implementation.
instructions:
  ref: instructions/build-openapi.md
```

`objective` states the Stage-specific desired outcome. The resolved
`instructions` text gives Planner Stage-specific operating guidance and
constraints, including how the logical Worker names may be used. Scheduler
performs no parameter interpolation, expression evaluation or prompt-template
rendering in either value. Empty or whitespace-only objective/instruction text
is invalid.

The authoring `instructions` object contains exactly one `ref`. It is not an
ArtifactRef and is never resolved through UserScope or RunScope. It names a
UTF-8 text resource under the configured `configs/` root. The normalized ref is
a non-empty `/`-separated relative path: absolute paths, URI schemes,
backslashes, empty segments, `.` and `..` segments are invalid. A local
configuration loader also rejects a symlink whose resolved target escapes the
configured root. The same logical resource rule can later be backed by an S3
prefix without changing Workflow YAML.

Conceptually, authoring and resolved forms are distinct:

```python
class InstructionsRef(BaseModel):
    ref: str


class ResolvedInstructions(BaseModel):
    ref: str       # normalized configuration-root-relative ref
    digest: str    # "sha256:" + 64 lowercase hexadecimal characters
    text: str
```

The instruction digest is SHA-256 over the exact source bytes and is encoded as
`sha256:<64 lowercase hex>`. The loader first verifies that those bytes decode
as strict UTF-8 and that the decoded text is not whitespace-only. It performs
no Unicode, line-ending, BOM or trailing-newline normalization before hashing.
Consequently the stored text and digest identify exactly the bytes loaded from
the configuration resource.

WorkflowCatalog resolves the ref while loading the Workflow, requires non-empty
text, and records the normalized ref, content digest and resolved text. It does
not read the file again while creating or executing a Run. The complete
resolved instruction dependency is embedded in the immutable Run snapshot, so
editing or deleting the configuration resource cannot change an existing Run.

Planner receives `objective` and resolved instructions from the immutable
StageSpec, the Run's string parameters from StageContext, and the separately
pinned artifact refs. A planning strategy such as Streamline uses objective as
its goal and the instruction text as its operating guidance.

PassthroughPlanner has no reasoning step of its own. It creates one Worker task
text deterministically as:

```text
Objective:
<Stage objective>

Instructions:
<resolved Stage instructions text>
```

That single text plus the separate structured Run-parameter mapping and pinned
artifact refs form its initial A2A request. This is the explicit passthrough
adapter behavior; it does not turn the Stage instruction into reusable Worker
configuration.

These strings are semantic input, not executable policy. Scheduler does not try
to infer success from their text, and they cannot override result contracts,
Workflow transitions, budgets, artifact grants, RuntimeSettings or platform
safety constraints. Worker-wide reusable behavior remains in the selected
`AgentTemplate`; Stage instructions do not mutate that template.

### Artifact slots and mappings

Workflow artifact contracts have three distinct levels:

- `spec.inputs` and `spec.outputs` declare the public input and output slots of
  the reusable Workflow;
- `stage.result.artifacts` declares the Stage-local artifact names that may
  appear as keys in its `StageResult.artifacts`;
- `stage.workflowOutputs` explicitly maps Workflow output slots to those
  Stage-local result names.

For example:

```yaml
spec:
  inputs:
    source:
      required: true
      mediaTypes: [application/zip]

  outputs:
    openapi:
      required: true
      mediaTypes: [application/yaml, application/json]

  entryStage: build
  stages:
    build:
      objective: Build an OpenAPI description from the source
      instructions:
        ref: instructions/build-openapi.md
      planner: passthrough@1
      agents:
        oas_builder:
          template: oas_builder@2

      context:
        artifacts:
          source:
            namespace: inputs
            name: source
            required: true

      result:
        artifacts:
          candidate:
            required: true
            mediaTypes: [application/yaml, application/json]

      workflowOutputs:
        openapi: candidate

      on:
        succeeded:
          succeed: {}
        failed:
          fail: {}
        interrupted:
          fail: {}
```

All slot names and Stage-local context/result names are stable identifiers in
their containing mapping. At Run creation, the caller supplies bindings only
for declared Workflow input slots, which Scheduler forks to
`inputs/<input slot>`.

`context.artifacts` is a mapping from a Stage-local name to an authoring
binding with exactly three fields: `namespace`, `name` and the mandatory boolean
`required`. Scope is implicitly the current RunScope. A `revision` is invalid in
Workflow YAML because revisions belong to one concrete Run:

```yaml
context:
  artifacts:
    source:
      namespace: inputs
      name: source
      required: true
    previous_report:
      namespace: analysis
      name: report
      required: false
```

When creating each StageExecution, before requesting any allocation, Scheduler
resolves every present binding to its exact current revision and records that
version in the durable StageContext. It also records explicit absence for each
missing optional binding; the resolved artifact mapping passed to Planner omits
those absent entries. A missing required binding enters the preparation abort
path with `StageTermination(code="context_artifact_missing",
retryable=false, phase="preparing")`, without allocating a Worker or creating a
Planner/Session.

The context snapshot is immutable. A retry is a new StageExecution and resolves
a fresh snapshot, so it may deliberately observe artifacts committed by the
previous attempt. Writes during the new attempt may advance logical bindings,
but cannot change the exact refs already present in its StageContext.
`context.artifacts` is data dependency/provenance, not an authorization
allowlist: Worker access remains governed by its allocation-bound RunScope
tools and grants in [03](03-artifact-plane.md).

Each `workflowOutputs` key must name a declared Workflow output, and its value
must name a result artifact declared by the same Stage. Scheduler considers the
mapping only for an accepted successful StageResult while WorkflowRun is still
`running`. The StageResult value is an exact versioned ArtifactRef; Scheduler
binds that version to `outputs/<output slot>` without resolving the source's
current binding again. There is no implicit output mapping based on artifact
Namespace, name, list position or matching slot names. Workers cannot write the
reserved `outputs` Namespace directly.

The same minimal artifact-slot contract is used in `spec.inputs`,
`spec.outputs` and `stage.result.artifacts`:

- `required` is a mandatory boolean; it has no default;
- `mediaTypes` is a mandatory, non-empty list of unique media types;
- a media type is a canonical lowercase `type/subtype` without parameters;
- matching is exact, except that the single value `*/*` explicitly accepts any
  media type; other wildcards are invalid, and `*/*` cannot be combined with
  specific values;
- `v1alpha1` has no artifact cardinality, size, filename-extension or structured
  payload schema in a slot contract. One slot binds at most one ArtifactRef.

For a required Workflow input, the Run request must supply a binding; an
optional input may be absent. Scheduler validates the exact selected source
version's media type before creating any Run input forks. For a successful
StageResult, every required Stage result slot must be present. Optional result
slots may be absent, but every present result ref is validated against its
declared media types regardless of Stage outcome. Unknown result names are
invalid.

A required Workflow output must be bound before Run success; an optional output
may remain absent. When applying `workflowOutputs`, Scheduler validates the
selected artifact against both the Stage result slot and the destination
Workflow output slot. Thus an incompatible mapping is rejected at catalog load
when the two declared media-type sets cannot intersect, and an incompatible
concrete artifact can never be accepted through that mapping.

The reusable definition owns these slot contracts; concrete input and output
bindings belong to one Run. A Stage declares:

- one literal objective and one set of Stage-specific instructions;
- one exact Planner factory/version;
- one or more uniquely named Worker bindings;
- its Stage context and references to available Run artifacts;
- named result-artifact slots and their required/optional and media-type
  contracts.

Conceptually:

```yaml
objective: Build and review an OpenAPI description
instructions:
  ref: instructions/build-and-review.md
planner: passthrough@1
agents:
  oas_builder:
    template: oas_builder@2
  reviewer:
    template: oas_reviewer@1
    namespace: shared_oas
context: ...
result: ...
workflowOutputs: ...
```

In Workflow authoring, `template` is the exact versioned selector defined above.
It resolves to the digest-bearing `AgentTemplateRef` owned by
[01](01-agent-template.md). The key in the non-empty `agents` mapping is the
stable logical name visible to Planner, not a physical Runtime Agent or process
identity. Each value is an object with the required `template` field and an
optional `namespace`; scalar shorthand is invalid. `namespace` defaults to the
mapping key and affects artifact tools only. Mapping order has no execution
semantics. WorkflowCatalog rejects duplicate YAML mapping keys before schema
validation rather than accepting a parser's last value.

At Run creation, the caller maps Workflow input slots to artifacts in its
authenticated UserScope. Before selecting the first Stage, Workflow Scheduler
pins their exact versions and forks independent working bindings into the Run's
reserved `inputs` Namespace. Run output bindings live in reserved `outputs` and
are created from accepted Stage results. The one-store, scoped fork and
publication contract is owned by [03](03-artifact-plane.md).

Workflow Scheduler owns this sequence for the normal, successfully prepared
StageExecution path:

1. resolve and durably pin the StageContext artifact snapshot;
2. load and verify every exact AgentTemplate dependency of the selected Stage;
3. ask Control Plane to prepare all required Worker allocations;
4. construct one Planner through the selected `PlannerFactory`;
5. invoke the Planner once with the fixed prepared Worker set;
6. validate and durably record the candidate `StageResult` plus exact referenced
   artifact versions, entering `finalizing`;
7. ask Control Plane to drain the Workers and collect execution/runtime reports;
8. atomically persist the terminal result and, only while the Run is still
   `running`, apply any declared Workflow output mapping and progression;
9. release every Stage allocation.

Planner is not invoked when required context cannot be resolved or any required
allocation fails to become ready. Context resolution occurs before allocation;
already prepared allocations from a later preparation failure are released
before Workflow Scheduler applies the Workflow's declared Stage failure/retry
policy. A non-capacity preparation failure records a `StageTermination`; it
does not synthesize a Planner, Planner Session or StageResult.

After a Planner terminates, Workflow Scheduler interprets `StageResult` under
that policy and, while the Run remains `running`, selects the next Stage, a
retry or an explicit escalation. The Planner cannot advance or rewrite the
Workflow graph itself. Retry creates a new StageExecution with an incremented
attempt number and, if preparation succeeds, a new Planner and ADK Session.
When Scheduler records a `StageTermination`, Workflow policy makes the same
outer progression decision without pretending that Planner returned a semantic
result, again only while the Run is `running`. In `cancelling`, terminal Stage
outcomes contribute only to reaching Run quiescence.

For every accepted StageResult or committed interrupted StageTermination while
the Run remains `running`, Scheduler records one immutable transition decision
keyed by the source `stage_execution_id`. The decision is exactly one of
`next`, `retry`, `succeed`, or `fail`. A `next`/`retry` decision includes the
target Stage name and newly created StageExecution ID; terminal decisions have
no target. Completing the source execution, applying successful output
mappings, recording this decision, and either creating/pinning the target
execution or making the Run terminal are one database transaction. Recovery
therefore observes either the old active source or the complete committed
progression and never manufactures a second attempt for the same outcome.

## WorkflowRun lifecycle

WorkflowRun has an explicit durable lifecycle:

```text
initializing -> running -> succeeded
      |             \----> failed
      \------------------> failed

initializing / running -> cancelling -> cancelled
```

```python
class WorkflowRunError(BaseModel):
    code: str
    message: str
    stage_execution_id: StageExecutionId | None = None


class WorkflowRunCancellation(BaseModel):
    code: Literal["user_cancelled"]
    requested_at: datetime
    requested_by: str | None = None
    reason: str | None = None
```

`failed` requires WorkflowRunError. `cancelling` and `cancelled` require the
same immutable WorkflowRunCancellation recorded by the winning cancel request.

- `initializing`: the validated Workflow snapshot, immutable Run parameters,
  fully resolved executionConfig and exact input forks are being committed; no
  Stage may start;
- `running`: Workflow Scheduler may select and create StageExecutions;
- `cancelling`: cancellation intent is durable, no new StageExecution or
  Workflow-output mapping is allowed, and active executions are stopped through
  their bounded `aborting` path unless they already entered `finalizing`;
- `succeeded`, `failed` and `cancelled` are immutable terminal states.

An initialization error moves the Run to `failed` with a stable error and no
StageExecution. Cancellation is accepted from `initializing` or `running` by
compare-and-set. Repeating cancellation is idempotent; cancellation of any
terminal Run returns that existing terminal state without mutation.

Public `POST /v1/runs` requires exactly one bounded `Idempotency-Key`. The
validated request includes Workflow selector, parameters, input refs and the
optional reference-only executionConfig override. Server binds the key to the
authenticated owner and a canonical digest of that request in the same
transaction as Run creation, execution-config resolution and exact input
forks. A retry with the same owner, key and digest returns the existing Run and
does not repeat input forks or Scheduler notification. Reusing the key with a
different digest is a conflict. Thus losing the successful HTTP response cannot
create a second WorkflowRun or semantic execution.

Run success is committed only when all of the following are true:

- Workflow policy selected an explicit successful terminal transition;
- no StageExecution remains active and policy selects no further Stage;
- every required Workflow output slot has an exact versioned source ref;
- the corresponding `outputs/<slot name>` bindings are created and frozen.

Accepting the StageResult that causes success, applying its Workflow-output
mapping, freezing all required outputs and changing `running -> succeeded` are
one database transaction. Frozen Run outputs are never implicitly published or
overwritten in UserScope.

Run failure is committed with a stable machine-readable error when Workflow
policy explicitly selects failure, retry/escalation paths are exhausted, Run
initialization fails, or the graph is quiescent without a valid successful
terminal transition. Graph exhaustion with a missing required output is
`failed`, not an indefinitely running Run. The first-slice Scheduler executes
one StageExecution at a time, so a policy-selected failure is committed only
after that execution is terminal; future parallel-Stage scheduling must add a
bounded failure-drain state before allowing other active executions.

`cancelling -> cancelled` occurs after all active StageExecutions are terminal
and their Server-side routes and grants are released or durably fenced. An
unreachable Runtime Agent cannot keep the Run in `cancelling` after its bounded
Stage abort deadline.

### Run cancellation and result race

Run cancellation and successful progression serialize on the WorkflowRun row:

- if the transaction accepting the final StageResult, mapping outputs and
  setting `succeeded` commits first, a later cancel is an idempotent no-op;
- if `cancelling` commits first, Scheduler starts no new Stage and accepts no
  later Workflow-output mapping;
- a StageExecution already in `finalizing` still reaches its own terminal
  StageResult for audit, but that result neither advances the cancelling Run nor
  creates its Workflow output bindings;
- a StageExecution still in `preparing` or `running` enters `aborting` with a
  cancelled StageTermination.

The terminal transition that commits first is authoritative; message arrival
time and volatile Planner state do not determine the winner.

## Planner

Planner is constructed for one prepared Stage. The replaceable Contractor
abstraction is the Go `PlannerFactory`; Workflow Scheduler sees only the
framework-neutral `Planner.Run(context.Context) (StageContentResult, error)`
boundary. `passthrough@1` is deterministic Go code. `streamline@1` constructs
one Google ADK Go `LlmAgent` behind that same boundary; ADK types do not enter
Scheduler, RunStore, A2A, or artifact contracts.

`streamline@1` pins `google.golang.org/adk` v1.6.0. It exposes every prepared
logical Worker as a fixed function tool whose internal adapter calls the
existing `WorkerInvoker` A2A boundary. The deterministic model-visible mapping
from logical name to tool name is included in Planner context. Contractor does
not depend on ADK's experimental remote-agent/A2A API, so changing that API
cannot change the Planner or Scheduler domain interfaces. Allocation routing
still comes only from the prepared `WorkerHandle` and its Agent Card.

The ADK agent and its live conversation stay inside Server memory. Invoking a
fixed Worker tool crosses the process boundary through A2A. Planner may
decompose, iterate and route among the fixed names, but it cannot:

- add or replace an Agent binding;
- choose a physical Runtime Agent;
- reserve capacity or initialize a Runtime Agent's Worker instance;
- change an AgentTemplate or Namespace;
- access a Worker sandbox directly.

Cross-Stage state passes through explicit Workflow context, Stage results or
artifacts, never through an implicitly shared Planner instance.

### Completion semantics

Planner strategy determines when its invocation is semantically complete, while
Workflow Scheduler owns the durable StageExecution transition. Planner returns
a candidate StageResult and does not write RunStore directly.

`PassthroughPlanner` completes when its required remote Worker invocation
produces an immediate A2A `Message`, a terminal Task, or a Task state requiring
interaction the baseline cannot provide. `INPUT_REQUIRED` and `AUTH_REQUIRED`
therefore become stable failed candidates in the passthrough baseline rather
than an unbounded wait. A more capable Planner may satisfy the requested
interaction and continue within its own budget.

A Streamline-style Planner completes successfully only through an explicit
`finish`; `escalate` produces an explicit semantic failed candidate. Both tools
validate declared result slots, exact revisions and media types but cannot
accept or advance Artifact bindings. An invalid completion call returns a
bounded tool error so the model may correct it within the remaining budget.
Scheduler independently repeats candidate validation before acceptance.

Each prepared ADK Worker independently enforces the cumulative model-call,
tool-call, and provider-reported token ceilings embedded in its exact
ModelPolicy. The optional tool-free result-finalization call consumes the same
budget. Exhaustion stops before the next side effect and returns retryable
failed code `worker_budget_exhausted`; it is a Worker candidate, not a Planner
budget termination and not a direct StageExecution write.

`streamline@1` receives `maxOutputTokens`, `maxModelCalls`, `maxWorkerCalls`,
and `maxTotalTokens` from its exact resolved ModelPolicy. The Stage/Planner
deadline remains an execution limit outside ModelPolicy and is always finite.
An incompatible or incomplete policy fails Run initialization before capacity
is prepared. Exhaustion without a valid terminal tool produces no semantic
candidate: Planner returns a stable safe error and Scheduler enters bounded
`aborting` with an interrupted `StageTermination`, phase `running`, and
`retryable: true`. The stable codes are
`planner_model_call_limit`, `planner_token_limit`,
`planner_worker_call_limit`, and `planner_deadline_exceeded`.
Every successful Planner Gateway response must include non-negative prompt,
completion and total token usage; missing or inconsistent usage is a retryable
invalid-response failure rather than a way to bypass the cumulative budget.

The Planner model alias comes from that ModelPolicy. Gateway URL/protocol and
the non-secret credential reference come from the Planner's independently
resolved LLMGatewayConfig selection. Planner resolves the pinned credential
revision through the Server secret boundary and never inherits a Worker's
token, Gateway or policy merely because both participate in the same Stage.

Only one tool call is executed per model turn. A parallel or unknown tool
selection is rejected before any Worker side effect and the model may correct
it. Each Worker call supplies a focused objective and instructions plus an
explicitly selected mapping of string parameters and exact ArtifactRefs. The
adapter verifies every added ref inside the same RunScope and applies the same
Stage deadline. Authoritative live lease loss cancels that context independently.

Once Planner has produced a candidate it performs no further semantic work. It
may request cancellation of outstanding A2A Tasks, but candidate delivery does
not depend on `CancelTask` succeeding or on observing a terminal Task state.
Workflow Scheduler's bounded `finalizing` path authoritatively stops the Worker
allocations and collects their reports. Scheduler-owned cancellation or
interruption instead uses the bounded `aborting` path.

### Passthrough baseline

`PassthroughPlanner` is the baseline integration path. It requires one prepared
Worker, sends the deterministic Stage input through the direct A2A
`WorkerInvoker`, waits for the remote invocation to complete and maps the
response and artifact refs into `StageResult`.

```text
StageSpec + StageContext + WorkerHandle
  -> A2A SendMessage
  -> immediate Message or Worker A2A Task
  -> completed response / versioned artifact refs
  -> StageResult
```

It knows neither which Runtime Agent hosts the Worker nor how that process
configured the selected runtime.
More capable Planners preserve the same outer Stage contract.

## StageResult

A normally completed Planner invocation produces one framework-neutral
semantic result candidate:

```python
class StageError(BaseModel):
    code: str
    message: str
    retryable: bool


class StageResult(BaseModel):
    outcome: StageOutcome
    artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)  # revision required
    content: StageContent | None = None
    error: StageError | None = None
```

`StageContent` is a bounded Contractor presentation DTO; an ADK adapter may map
to and from `google.genai.types.Content`, but the core result does not expose an
ADK type. Its exact text/structured-part shape remains an implementation
decision. `StageMetrics` and report completeness are stored separately on
StageExecution and are defined in [04](04-execution-lifecycle-and-metrics.md).

Planner produces a candidate. Scheduler validates and stores one normalized
`StageResult` while entering `finalizing`; final report collection may update
StageMetrics but never rewrites this semantic result.

`StageOutcome` is `succeeded` or `failed`.

- success has no `error` and satisfies the Stage's required artifact contract;
- failure has an `error`;
- failed results may retain useful partial content or artifacts;
- presentation `content` is bounded and excludes thoughts/function protocol;
- large or durable output is an `ArtifactRef`, never inline content;
- every result ref contains a revision and identifies the exact version selected
  by Planner;
- Scheduler validates result refs in the current RunScope, never directly in
  UserScope, and never replaces their revision with the current binding;
- every `artifacts` key is a Stage-local result name declared by the Stage;
  Workflow output mappings select results by that name rather than by list
  position or by parsing an ArtifactRef;
- metrics are not part of StageResult and cannot change its outcome.

Exact AgentTemplate refs, WorkerRuntime refs, Runtime Agent process identities
and allocation IDs are recorded with the StageExecution for provenance; they
are not duplicated inside presentation content.

Scheduler-driven cancellation or interruption is not represented as a
StageResult. It uses the distinct `StageTermination` contract defined in
[04](04-execution-lifecycle-and-metrics.md). This keeps a preparation failure,
lost Worker or external cancellation from looking like a semantic Planner
decision.

## Stage execution scope

One StageExecution has at most one root Planner instance, one Planner invocation
and one database-backed Planner Session. All three are created only after the
fixed allocation set is ready and are required from `running` onward on the
normal result path. An allocation lives through Planner invocation and
finalization and may serve multiple sequential A2A Tasks while running. A
Worker handles at most one Task at a time; different prepared Workers run on
different allocated Runtime Agent processes and may execute concurrently.

Workflow Scheduler persists the candidate before stopping Workers, accepts the
terminal StageResult or records a StageTermination, then releases allocations.
Planner-private plans and A2A Task IDs do not become additional Workflow
stages. `finalizing` and `aborting` are mutually exclusive durable paths as
defined in [04](04-execution-lifecycle-and-metrics.md).
