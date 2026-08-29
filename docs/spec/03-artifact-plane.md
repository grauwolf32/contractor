# 03 — Artifact plane

Status: **Working agreement**

Depends on: [00](00-workflow-and-planner.md), [02](02-runtime-and-a2a.md)

## Model

Contractor has one physical `ArtifactStore`, not separate `UserArtifactStore`
and `RunArtifactStore` services. Every store operation is evaluated in one
authenticated scope:

- `UserScope(user_id)` is the user's durable artifact library. The public API
  derives `user_id` from authentication rather than accepting an arbitrary
  scope from the request. It supports upload, read, list and revision-checked
  binding updates;
- `RunScope(run_id)` is the logical `RunArtifactSpace` of one Workflow Run.
  Planner and all allocated Runtime Agents acting as Workers across every Stage
  share this scope.

`RunArtifactSpace` is therefore a scoped view of `ArtifactStore`, not another
storage component. Artifacts in it are durable Stage outputs and the data plane
for concurrent Worker collaboration. A Worker never reads another Worker's
sandbox.

The public `ArtifactRef` uses `(namespace, name, revision?)`. Scope is
structural context and must not be encoded into any field. Omitting `revision`
addresses the current logical binding; including it addresses one exact
historical binding revision. The fully qualified internal binding key is:

```text
(scope_kind, scope_id, namespace, name)
```

Adding `revision` to that key identifies one exact retained binding revision
and its immutable internal version.

This keeps one ref model while preventing a Worker from selecting another Run
or UserScope in a tool call. A revision is an opaque selection token, not an
authorization credential.

## Namespaces

The space contains Namespaces. A Namespace is:

- part of artifact addressing;
- the fixed scope of specialized domain tools;
- unique only within its UserScope or RunScope;
- not a security boundary against explicitly granted current-Run tools.

RunScope reserves two Namespaces:

- `inputs` contains mutable working copies of declared Workflow inputs;
- `outputs` contains declared Workflow results and is managed by Workflow
  Scheduler.

All other Namespaces hold intermediate artifacts. They normally use the Stage
Agent Namespace selected by its binding. For display and authoring,
`namespace/name` denotes one `ArtifactRef`; `/` does not create a filesystem or
an additional hierarchy. For example:

```text
inputs/source                 ArtifactRef(namespace="inputs", name="source")
oas/review-report             ArtifactRef(namespace="oas", name="review-report")
outputs/openapi               ArtifactRef(namespace="outputs", name="openapi")
```

`inputs` and `outputs` cannot be selected as Stage Agent Namespaces. Their names
are Workflow input/output slot names rather than arbitrary Worker-selected
paths.

## Stage Namespace binding

Workflow Scheduler resolves each Stage Agent's Namespace from its Stage binding
before allocation. An explicit `namespace` wins; otherwise the logical Agent
name is used. The result travels in AllocationSpec and is fixed when Worker
tools are constructed.

Specialized namespace-bound toolsets must resolve to unique Namespaces within
one Stage. A later or different Stage may assign the same Namespace to another
Agent and continue that data scope.

There is no Namespace-wide writer lock: different artifact names are
independent, while conflicting writes by authorized participants to the same
logical binding use compare-and-swap. Whether multiple Stages of one Run execute
concurrently is a separate Workflow Scheduler decision and is not required by
this artifact contract.

## Tool authority

An AgentTemplate may explicitly select two kinds of model-visible tools:

1. the exact domain tools selected from AgentTemplate Toolsets and fixed to its
   resolved Namespace, for example selected members of
   `openapi_tools(namespace)` or `memory_tools(namespace)`;
2. selected members of the built-in `run-artifacts@1` Toolset, which exports
   `list_artifacts`, `read_artifact` and `write_artifact` for explicit
   current-Run refs.

No generic Artifact tool is injected implicitly. An AgentTemplate that selects
only `list_artifacts` and `read_artifact` exposes a model-visible read-only
interface; one that omits `run-artifacts@1` exposes none of those generic
operations. A domain tool may still use the allocation-bound private Artifact
client internally as part of its registered implementation.

Namespace is not a model-selected argument of domain tools. The generic tools
deliberately have broader read authority inside the current Run. The baseline
write grant permits `inputs` and non-reserved Namespaces, so Workers can modify
their working input copies and intermediate artifacts. It denies writes to
`outputs`, whose bindings are created only by Workflow Scheduler from accepted
Stage results.

The Server's private Artifact API authenticates the Runtime Agent through mTLS,
binds the request to its active allocation and enforces the hard Run boundary,
reserved-Namespace policy, byte limits and write preconditions.

The initial private HTTP binding has one unambiguous write protocol. Creating a
binding requires `If-None-Match: *`; updating it requires exactly one quoted,
strong `If-Match` value containing the expected opaque revision. A PUT without
one of those preconditions, a weak validator, or a list of validators is
rejected. Successful reads return the resolved revision in `ETag`; successful
writes return the same exact revision in both `ETag` and the versioned
`ArtifactWriteResult`. The allocation ID is the only authority-bearing path
value: neither URL nor body accepts a Run or User scope selector.

Tool selection and Artifact authorization are independent checks. Selecting
`write_artifact` only constructs and exposes that model tool; it never broadens
the Server-side allocation grant. A grant likewise does not cause an
unselected tool to be constructed or exposed.

When that StageExecution durably enters `finalizing` or `aborting`, Scheduler
establishes a Server-side write fence for every associated allocation. From
that transition onward the private Artifact API rejects its writes even if a
Worker or A2A Task has not yet stopped. Waiting until allocation release is not
a conforming write-revocation boundary.

There are no separate `read_any_artifact` aliases. The authority is evident in
the explicitly selected generic tool, its explicit ref and the allocation
grant.

## Public contract

An `ArtifactRef` has two explicit modes: versionless means the current logical
binding, while a non-null `revision` means the exact immutable version selected
through that binding revision:

```python
class ArtifactRef(BaseModel):
    namespace: NamespaceId
    name: str
    revision: ArtifactRevision | None = None


class ArtifactPayload(BaseModel):
    media_type: str
    data: bytes  # transport/storage may stream this


class ArtifactReadResult(BaseModel):
    ref: ArtifactRef  # revision is always present
    payload: ArtifactPayload


class ArtifactWriteResult(BaseModel):
    ref: ArtifactRef  # revision is always present
```

Scope comes from authenticated API, Run/Stage or allocation context and is not
duplicated in the ref. Blob locations, digests and immutable storage-version
IDs remain internal. The Server resolves
`(scope, namespace, name, revision)` to that internal version.

`ArtifactPayload.media_type` is normalized and stored with every immutable
artifact version as a lowercase `type/subtype` without parameters. Invalid or
parameterized values are rejected by the first-slice API. Workflow input,
Stage-result and Workflow-output slot validation uses this stored value and the
strict matching rules defined in [00](00-workflow-and-planner.md).

The scope-bound operations are conceptually:

```python
async def read_artifact(ref: ArtifactRef) -> ArtifactReadResult: ...

async def write_artifact(
    ref: ArtifactRef,
    payload: ArtifactPayload,
    expected_revision: ArtifactRevision | None,
) -> ArtifactWriteResult: ...

async def list_artifacts(
    namespace: NamespaceId | None = None,
) -> list[ArtifactRef]: ...
```

`list_artifacts` returns versionless refs. Reading a versionless ref resolves
the current binding and returns a ref with the resolved revision. Reading a
versioned ref returns that exact historical revision and never falls forward to
the current binding.

The target `ref` passed to `write_artifact` must be versionless. CAS remains a
separate write concern: `expected_revision=None` is create-only, while updating
an existing binding requires the opaque revision returned by read/write. A
versioned write target is invalid because `ref.revision` and
`expected_revision` would have conflicting meanings. A stale expected revision
returns `ArtifactConflict`; infrastructure never silently overwrites or
semantically merges the winner.

ArtifactReadResult and ArtifactWriteResult always contain a versioned ref. An
old revision remains resolvable to its immutable internal version for as long
as Run input provenance, a StageContext snapshot, a StageResult, a Run output
or retention policy keeps that version alive.

One resolved, versioned ref identifies one payload with one media type.
Contractor has no compound artifact/`parts[]` contract. Multi-file output is an
archive or a manifest that names exact versioned refs.

The same contract backs the User Artifact API and Run clients, but scope binding
and grants differ. A public user client is bound to its authenticated
`UserScope`; a Worker client is bound to the `RunScope` in its allocation. Only
purpose-specific trusted Server operations may perform a controlled cross-scope
fork; there is no generic client operation that accepts arbitrary source and
target scopes.

## Workflow inputs and scope fork

A Workflow definition declares named input and output slots. Concrete input and
output bindings belong to one Workflow Run, not to the reusable Workflow
definition.

A source tree, repository archive, OpenAPI document or other domain input is an
ordinary user-scoped artifact selected for a declared slot. Contractor core has
no `ProjectSnapshot` or `ProjectSnapshotManifest` domain type. Generic digest,
streaming, retention and size behavior belongs to ArtifactStore; safe unpacking
or interpretation belongs to the selected Worker tools and sandbox policy.

At Run creation, the caller maps input slots to `ArtifactRef` values in its
authenticated UserScope. A versionless input selects the binding current at Run
creation; a versioned input explicitly selects that retained historical
revision. Before the Run becomes schedulable, Workflow Scheduler records the
exact source version of every supplied input, validates its media type, then
forks it to:

```text
UserScope(user_id): <source namespace>/<source name>
  -> RunScope(run_id): inputs/<workflow input name>
```

A fork creates an independent target binding and lineage edge to the exact
source version. It may initially reuse the same immutable blob; copying bytes is
not required. The first or any later Run write creates a new immutable version
and advances only the Run binding through CAS. It never mutates the source user
binding.

The initialization of all required inputs is atomic from the execution model's
point of view: no Stage is selected until every fork succeeds and its exact
source version is recorded. Optional missing inputs remain absent. A failed
initialization leaves no runnable partial RunArtifactSpace. Recovery retries
the recorded source versions; it never re-resolves a newer current UserScope
binding for the same Run.

## Workflow outputs and explicit publication

Every ArtifactRef in a candidate StageResult must contain a revision. When
entering StageExecution `finalizing`, Workflow Scheduler validates each exact
ref, including its stored media type against the declared result slot, and pins
the internal version identified by that revision. It never
re-resolves the current binding, and the selected revision need not still be
current. Worker finalization cannot mutate artifacts. When Scheduler accepts
the terminal successful result while WorkflowRun is still `running`, the same
database transaction creates or advances the corresponding
`outputs/<workflow output name>` binding to those already pinned versions. This
also validates the destination Workflow output slot. The binding is a metadata
fork inside the same RunScope and does not copy bytes. If Run
cancellation committed first, StageResult may still be accepted for audit but
its Workflow-output mapping is not applied.

Workers may read declared outputs produced by earlier accepted Stages but cannot
write the reserved Namespace. Once a Run succeeds, all required output bindings
are present, their exact versions are retained in Run provenance, and those
bindings are frozen in the same transaction as `running -> succeeded`. When a
Run instead becomes `failed` or `cancelled`, any output bindings accepted from
earlier Stages are also frozen for audit but do not constitute successful Run
completion and are not implicitly published.

A successful Run does not implicitly overwrite the user's artifact library.
Publishing a result is an explicit controlled fork from the exact Run output
version to a caller-selected versionless `ArtifactRef` in UserScope. Creating a
new user binding is create-only; replacing an existing one requires its current
opaque revision as a separate CAS precondition. Publication records lineage and
may reuse the existing blob.

## OpenAPI Workflow example

Assume the authenticated user's library contains:

```text
UserScope(user-42)
  projects/payment-service-source
  openapi/payment-service-current
```

The Run request maps those refs to the Workflow's `source` and
`existing_openapi` input slots. Initialization produces independent bindings:

```text
RunScope(run-17)
  inputs/source
  inputs/existing_openapi
```

An `oas_builder` Worker in Namespace `oas` reads `inputs/source`, updates
`inputs/existing_openapi` with CAS and may write `oas/source-map`. A later
reviewer may update the same working OpenAPI binding and write
`oas/review-report`. A validator reads the resulting current version and writes
`validation/report`.

Suppose the final accepted StageResult contains these named result bindings:

```yaml
artifacts:
  openapi:
    namespace: inputs
    name: existing_openapi
    revision: 4
  validation_report:
    namespace: validation
    name: report
    revision: 2
```

Here `openapi` and `validation_report` are Stage-local result names, while the
values are exact, versioned `ArtifactRef`s. For example, the producing Stage can
declare this explicit output mapping:

```yaml
workflowOutputs:
  openapi: openapi
  validation_report: validation_report
```

The mapping keys are Workflow output slots and the values are Stage-local
result names. It causes Scheduler to bind those exact revisions as:

```text
outputs/openapi
outputs/validation_report
```

Neither user source changed during this Run. If requested after success,
`outputs/openapi` can be explicitly published to a new UserScope ref such as
`openapi/payment-service-v2`; it is not copied back to
`openapi/payment-service-current` automatically.

## Publication and visibility

ArtifactStore keeps immutable content versions and one current binding for each
fully qualified internal key:

```text
uploading -> committed
uploading -> aborted
committed -> expired by its scope's retention policy
```

Only a complete committed payload is readable. A successful RunScope commit is
immediately visible to all authorized participants of the Run. StageResult
acceptance is therefore not a general artifact visibility event; only its
optional binding into the reserved `outputs` Namespace is Scheduler-controlled.

Normal Worker reads through a versionless ref resolve the current committed
binding and return the exact versioned ref that was read. Entering
StageExecution `finalizing` validates and pins the revisions already named by
the candidate StageResult. Terminal acceptance may bind those versions into
`outputs`; it does not re-resolve a newer current binding.

ArtifactStore exposes no Worker subscription/event-stream contract. Planner
coordinates meaningful refs through A2A; Workers may discover current state
through list/read. Internal registry history and audit events do not imply a
public change feed.

## A2A and ADK

Normal durable artifact bytes stay in ArtifactStore. A2A progress/results carry
a small structured `ArtifactRef` extension. A Worker forwards the versioned ref
returned by its artifact read/write operation when it identifies a concrete
result. An A2A message may still carry a versionless ArtifactRef when it
deliberately means "read current"; this is unrelated to the catalog
instruction-resource refs used by Workflow and AgentTemplate configuration.

ADK `BaseArtifactService` is an adapter, not the authoritative contract:

```python
class AdkArtifactServiceAdapter(BaseArtifactService):
    store: ContractorArtifactClient
    run_id: RunId
    stage_execution_id: StageExecutionId
    principal: ArtifactPrincipal
```

Planner/Worker Runners receive an explicitly configured adapter connected to
the shared RunArtifactSpace; an exported ADK A2A Worker must not fall back to an
in-memory artifact service. One ADK `types.Part` maps to one Contractor payload.

ADK `save_artifact` lacks CAS/grant semantics. The adapter may use it for
append-only creation. Shared binding updates use the richer Contractor client
with explicit revision preconditions.

## Service and storage boundaries

```text
ArtifactStore
  -> ArtifactRegistry
       scopes, immutable versions, logical bindings, CAS, grants, lineage
  -> ArtifactBlobStore
       immutable payload bytes
```

The initial registry is PostgreSQL because binding CAS, scope forks and
Stage-output recording need transactions. `ArtifactBlobStore` is replaceable:
bytes may initially live in PostgreSQL and later in S3 without changing
ArtifactRef.

For the first slice, RunStore and ArtifactRegistry participate in the same
PostgreSQL transaction when Scheduler accepts a StageResult, applies Workflow
output mappings and possibly makes WorkflowRun terminal. A future storage split
must preserve the same atomic contract rather than exposing a partially mapped
terminal Run.

The User Artifact API binds operations to the authenticated UserScope. An
allocated Runtime Agent uses the Server's private Artifact API with its existing
mTLS identity and active allocation ID rather than database/S3 credentials. The
Server binds requests to that allocation's RunScope, enforces
grants/limits/preconditions and streams bytes. Neither public nor private
interface lets its caller supply an arbitrary scope ID.

Workflow Scheduler uses an internal ArtifactStore capability to pin versions,
fork UserScope inputs into RunScope and bind accepted versions into `outputs`.
The User Artifact API exposes a separate, purpose-specific publication
operation for an authorized Run output. ArtifactStore and its APIs do not
interpret domain artifact semantics.

```python
class ArtifactBlobStore(Protocol):
    async def begin_upload(...) -> UploadSession: ...
    async def complete_upload(...) -> BlobRef: ...
    async def open(blob: BlobRef) -> AsyncIterator[bytes]: ...


class ArtifactRegistry(Protocol):
    async def commit_version(...) -> StoredArtifactVersion: ...
    async def resolve(
        binding: ArtifactBinding,
        revision: ArtifactRevision | None = None,
    ) -> StoredArtifactVersion | None: ...
    async def fork_version(...) -> StoredArtifactVersion: ...
    async def record_stage_outputs(...) -> None: ...
```

`resolve(binding, None)` resolves current. Supplying a revision resolves only
that retained historical binding revision and never substitutes another
version.

## Invariants

1. One ArtifactStore serves UserScope and RunScope; they are scoped views, not
   separate storage services.
2. ArtifactRef contains Namespace, name and an optional opaque revision;
   authenticated context supplies scope and callers cannot select another
   scope.
3. All Stage participants in one Run use the same RunArtifactSpace.
4. Every required input is pinned and forked before the Run is schedulable;
   Run mutations never modify its UserScope source.
5. Generic Worker grants never cross a Run and cannot write `outputs`.
6. Allocation writes are rejected from the durable transition to `finalizing`
   or `aborting`, independently of Worker shutdown and allocation release.
7. Domain tool Namespace is fixed before Worker allocation and cannot be a
   reserved Namespace.
8. Artifact content is immutable; only its logical current binding changes.
9. A versionless ref resolves current; a versioned ref resolves exactly and
   never falls forward to a newer binding.
10. Writes require a versionless target and use a separate explicit CAS
    precondition; same-binding conflicts are returned to the caller.
11. Every candidate StageResult ref is versioned before `finalizing`; Scheduler
    pins that exact revision rather than resolving current.
12. Declared Run outputs are Scheduler-managed bindings to accepted versions.
13. Final required-output binding/freezing and `running -> succeeded` are one
    transaction; a StageResult accepted after Run cancellation cannot create
    Workflow output bindings.
14. Run-to-user publication is explicit and revision-checked.
15. A2A normally carries refs, not artifact bytes.
16. Runtime Agent receives no Artifact Registry or blob-backend credential.
17. Every present StageContext artifact is pinned to an exact retained revision
    before allocation; missing optional bindings are recorded as absent, while
    a missing required binding prevents Planner creation.
