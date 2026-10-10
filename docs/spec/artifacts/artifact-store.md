# Artifact store

Requirement prefix: `ART` · [Specification index](../README.md)

## Purpose and scope

This document owns the ArtifactStore: its structural scopes, artifact
addressing, Namespaces and reserved names, binding revisions and
compare-and-swap (CAS) writes, the Artifact authority of Workers and their
tools, the public and private Artifact API contract, Workflow input forks,
Workflow output binding and freezing, and explicit publication of Run outputs.
The artifact plane owns data access, authenticated scope and Namespace grants.
<!-- from: 02#115, 03#3 -->

It leaves to other documents:

- Workflow input and output slot declarations, slot media-type matching, Stage
  binding fields and `workflowOutputs` validation:
  [Workflows](../configuration/workflows.md).
- StageExecution lifecycle, StageContext snapshots and retries, the preparation
  abort path and finalization records: [Scheduler](../execution/scheduler.md).
- WorkflowRun lifecycle, Run creation, repeat drafts and manual continuation
  eligibility: [Runs](../execution/runs.md).
- Projection of observed refs into Worker results and the Worker model
  boundary: [Worker](../execution/worker.md).
- Runtime Agent principals and mTLS identity:
  [Control Plane](../execution/control-plane.md) and
  [Security](../platform/security.md).
- The Project resource, Project Run creation and Project output publication:
  [Projects](projects.md).
- Scope purge and garbage collection:
  [Retention and deletion](retention-and-deletion.md).
- Payload backends: [Blob storage](blob-storage.md).
- Git repositories as artifact inputs: [Git import](git-import.md).
- The common Toolset selection contract: [Toolsets](../tools/toolsets.md);
  purpose-specific artifact-backed tools: [Memory](../tools/memory.md),
  [Skills](../tools/skills.md), [Workspace](../tools/workspace.md),
  [HTTP and Caido](../tools/http-caido.md), [Scanners](../tools/scanners.md).
- Audit-managed artifacts: [Audits](../audits/audits.md) and
  [Review and reports](../audits/review-and-reports.md).
- Transaction conventions: [Persistence](../platform/persistence.md).

## Concepts

Terms are defined once in the [glossary](../glossary.md).

| Term | Meaning here |
| --- | --- |
| ArtifactStore | The one physical artifact store of Contractor. |
| Scope | The structural context of a store operation: `UserScope`, `ProjectScope` or `RunScope`. |
| RunArtifactSpace | The RunScope of one WorkflowRun, seen as the shared data plane of its Stages. |
| ArtifactRef | The public artifact address `(namespace, name, revision?)`. |
| Binding | The logical, mutable `(scope, namespace, name)` entry that points at one current immutable version. |
| Revision | An opaque token that selects one exact retained binding revision. |
| Namespace | The first component of artifact addressing and the fixed scope of domain tools. |
| Stage Agent Namespace | The Namespace resolved for one Stage Agent from its Stage binding. |
| Fork | A new independent binding with a lineage edge to an exact source version. |
| Publication | A controlled fork of an exact Run output version into UserScope (explicit, on request) or ProjectScope (on Project Run success). |
| Write fence | The Server-side rejection of an allocation's Artifact writes once its StageExecution enters `finalizing` or `aborting`. |
| Allocation grant | The Server-side record that authorizes one Runtime Agent process to use the private Artifact API for one allocation. |
| Frozen binding | A binding that can no longer advance: for example a Run `outputs` binding once the Run is terminal (except through manual continuation of an eligible failed Run, ART-R97), or an Audit-managed `audit-*` Project binding, which is created frozen ([Review and reports](../audits/review-and-reports.md)). |

<!-- from: 03#3, 03#5, 03#6, 03#11, 03#32, 03#64, 03#67, 03#68, 03#70, 03#101, 19#168 -->

## Model

### Scopes

| Scope | What it holds |
| --- | --- |
| `UserScope(user_id)` | The user's durable artifact library. |
| `ProjectScope(project_id)` | The owner's long-lived reusable workspace for one [Project](projects.md). |
| `RunScope(run_id)` | The `RunArtifactSpace` of one Workflow Run: durable Stage outputs and the data plane for concurrent Worker collaboration. |

<!-- from: 03#4, 03#5, 17#11 -->

`RunArtifactSpace` is a scoped view of `ArtifactStore`, not another storage
component.
<!-- from: 03#5 -->

### References and bindings

The public `ArtifactRef` is `(namespace, name, revision?)`. The fully
qualified internal binding key is:

```text
(scope_kind, scope_id, namespace, name)
```

<!-- from: 03#6, 03#7 -->

Adding `revision` to that key identifies one exact retained binding revision
and its immutable internal version. ArtifactStore keeps immutable content
versions and one current binding for each fully qualified internal key.
<!-- from: 03#8, 03#83 -->

### Namespaces

A Namespace is:

- part of artifact addressing;
- the fixed scope of specialized domain tools;
- unique only within its UserScope, ProjectScope or RunScope;
- not a security boundary against explicitly granted current-Run tools.

<!-- from: 03#11, 03#12 -->

Reserved Namespaces and name prefixes:

| Name | Kind | Scope | Holds | Rules |
| --- | --- | --- | --- | --- |
| `inputs` | Namespace | RunScope | Mutable working copies of declared Workflow inputs | ART-R14, ART-R76 |
| `outputs` | Namespace | RunScope | Declared Workflow results, managed by Workflow Scheduler | ART-R14, ART-R90 |
| `skills` | Namespace | RunScope | Exact read-only Agent Skill packages ([Skills](../tools/skills.md)) | ART-R14, ART-R39, ART-R108, ART-R109 |
| `contractor-system` | Namespace, hidden | RunScope | Bounded Control Plane records coupled to a Run | ART-R17, ART-R18 |
| `memory.` | Name prefix | Non-reserved RunScope Namespaces | Memory notes ([Memory](../tools/memory.md)) | ART-R19 to ART-R22 |
| `tool-invocation.` | Name prefix | RunScope | Runtime-owned receipts of model-free tool Workers ([Scanners](../tools/scanners.md)) | ART-R23 |
| `http.body.` | Name prefix | The allocation's logical Namespace | HTTP response bodies ([HTTP and Caido](../tools/http-caido.md)) | ART-R24 |
| `audit-standards` | Namespace | UserScope | Curated Audit standard packages ([Audits](../audits/audits.md)) | ART-R25 |
| `audit-*` | Namespace prefix | ProjectScope | Audit-managed results, evidence and reports ([Review and reports](../audits/review-and-reports.md)) | ART-R26 |

<!-- from: 03#13, 03#14, 03#18, 03#25, 08#15, 11#34, 19#34, 19#168 -->

### Version states

```text
uploading -> committed
uploading -> aborted
committed -> expired by its scope's retention policy
```

<!-- from: 03#84 -->

```mermaid
stateDiagram-v2
    [*] --> uploading
    uploading --> committed
    uploading --> aborted
    committed --> expired: scope retention policy
```

<!-- from: 03#83, 03#84 -->

### Storage components

```text
ArtifactStore
  -> ArtifactRegistry
       scopes, immutable versions, logical bindings, CAS, grants, lineage
  -> ArtifactBlobStore
       immutable payload bytes
```

<!-- from: 03#96 -->

### Scope forks

An input fork moves an exact source version into a Run:

```text
UserScope(user_id) or ProjectScope(project_id): <source namespace>/<source name>
  -> RunScope(run_id): inputs/<workflow input name>
```

<!-- from: 03#63 -->

```mermaid
flowchart LR
    subgraph SRC["UserScope or ProjectScope"]
        S["source namespace/name, exact version"]
        K["owner UserScope skills/*"]
    end
    subgraph RUN["RunScope(run_id)"]
        I["inputs/input-slot"]
        W["intermediate Namespaces"]
        SK["skills"]
        O["outputs/output-slot"]
    end
    U["UserScope versionless ref"]
    P["ProjectScope"]
    S -->|"input fork at Run creation, before the Run is schedulable"| I
    K -->|"Skill fork, Scheduler, same initialization"| SK
    I -->|"exact StageResult revision, output_bind"| O
    W -->|"exact StageResult revision, output_bind"| O
    O -->|"explicit publication, CAS"| U
    O -->|"Project Run success, project_output_publish"| P
```

<!-- from: 00#113, 03#62, 03#64, 03#66, 03#70, 03#106, 17#21 -->

## Requirements

### Store and scopes

**ART-R1.** Contractor has one physical `ArtifactStore`, not separate
User, Project and Run artifact services; UserScope, ProjectScope and RunScope
are scoped views of it.
<!-- from: 03#3, 17#11, 03#106 -->

**ART-R2.** Every store operation is evaluated in one
authenticated or trusted internal scope.
<!-- from: 03#3 -->

**ART-R3.** The public API derives the UserScope `user_id` from
authentication rather than accepting an arbitrary scope from the request.
UserScope supports upload, read, list and revision-checked binding updates.
<!-- from: 03#4 -->

**ART-R4.** The ProjectScope public API derives the owner and
the Project scope from authentication and the Project resource. ProjectScope
allows arbitrary artifacts, and exact selected versions may be forked into
Runs ([Projects](projects.md)).
<!-- from: 03#4 -->

**ART-R5.** Planner and all allocated Runtime Agents acting as
Workers, across every Stage of one Workflow Run, share that Run's RunScope:
all Stage participants in one Run use the same RunArtifactSpace.
<!-- from: 03#4, 03#106 -->

**ART-R6.** A Worker never reads another Worker's sandbox.
<!-- from: 03#5 -->

### References and names

**ART-R7.** An `ArtifactRef` is `(namespace, name, revision?)`.
Scope comes from the authenticated API, Run/Stage or allocation context; it is
structural context and must not be encoded into any ref field or duplicated in
the ref.
<!-- from: 03#6, 03#36 -->

**ART-R8.** An `ArtifactRef` has two explicit modes. A versionless
ref (no `revision`) addresses the current logical binding. A ref with a
non-null `revision` addresses one exact historical binding revision and the
immutable version selected through it.
<!-- from: 03#6, 03#34 -->

**ART-R9.** The Server resolves `(scope, namespace, name, revision)`
to one immutable internal version. Blob locations, digests and immutable
storage-version IDs remain internal.
<!-- from: 03#8, 03#36 -->

**ART-R10.** A revision is an opaque selection token, not an
authorization credential.
<!-- from: 03#9 -->

Rationale: one ref model is kept while a Worker cannot select another Run or
UserScope in a tool call.
<!-- from: 03#9 -->

**ART-R11.** Both `namespace` and `name` are case-sensitive ASCII
identifiers of 1 through 128 characters (LIM-ART-1): the first character is a
letter or digit, followed only by letters, digits, `_`, `-` or `.`
(`[A-Za-z0-9][A-Za-z0-9_.-]{0,127}`). Spaces, non-ASCII characters and control
characters are rejected.
<!-- from: 03#10 -->

**ART-R12.** Workflow validation, wire DTOs, ArtifactStore and
the public and private APIs enforce the same grammar. Names are never trimmed
or normalized.
<!-- from: 03#10 -->

Rationale: Workflow inputs can depend on the exact spelling of a name.
<!-- from: 03#10 -->

**ART-R13.** For display and authoring, `namespace/name` denotes one
`ArtifactRef`; `/` does not create a filesystem or an additional hierarchy.
For example:
<!-- from: 03#15 -->

```text
inputs/source                 ArtifactRef(namespace="inputs", name="source")
oas/review-report             ArtifactRef(namespace="oas", name="review-report")
outputs/openapi               ArtifactRef(namespace="outputs", name="openapi")
```

<!-- from: 03#16 -->

### Namespaces and reserved names

**ART-R14.** RunScope reserves three model/data Namespaces:

- `inputs` contains mutable working copies of declared Workflow inputs;
- `outputs` contains declared Workflow results and is managed by Workflow
  Scheduler;
- `skills` contains exact read-only Agent Skill packages pinned and forked by
  Workflow Scheduler ([Skills](../tools/skills.md)).

<!-- from: 03#13, 03#14, 09#58 -->

**ART-R15.** All other RunScope Namespaces hold intermediate
artifacts. They normally use the Stage Agent Namespace selected by the Stage
binding.
<!-- from: 03#15 -->

**ART-R16.** `inputs`, `outputs` and `skills` cannot be
selected as Stage Agent Namespaces. The names in them are Workflow slot names
or trusted runtime-resource names, not arbitrary Worker-selected paths.
<!-- from: 03#17, 01#94 -->

**ART-R17.** The physical store may additionally contain the exact
hidden `contractor-system` Namespace for bounded Control Plane records coupled
to a Run; its `repeat-request` binding retains the non-secret request fragment
defined by [Runs](../execution/runs.md). It is written only through a
purpose-specific trusted service operation, is hidden from every public
Artifact list, read, metadata, version and lineage route and from every private
allocation Artifact operation, and is removed with the containing RunScope
(LIFE-ART-1).
<!-- from: 03#18 -->

**ART-R18.** `contractor-system` is not an Agent Namespace, a
Workflow input or output, a collaboration channel or a general internal
key/value store. Owner access is only through the validated repeat-draft
projection owned by [Runs](../execution/runs.md), which rechecks authorization
and source scopes.
<!-- from: 03#18 -->

**ART-R19.** The `memory.` artifact-name prefix is reserved at the
model-visible Toolset layer in every non-reserved RunScope Namespace.
`run-artifacts@1` and every other model-visible artifact-backed Toolset,
including bounded text, source-analysis, OpenAPI and LikeC4 tools, reject a
source or target that resolves to such a binding and filter such bindings from
lists and from any accumulated exact-ref projection, even when the same
AgentTemplate selects MemoryTools. Selecting a domain tool must not become an
alternate route around MemoryTools.
<!-- from: 03#25, 08#15, 03#106 -->

**ART-R20.** No StageContext declaration or StageResult can
expose a RunScope binding whose name begins `memory.` in a non-reserved
Namespace.
<!-- from: 03#106 -->

**ART-R21.** The lower-level allocation-bound private Artifact
API stays domain-neutral: the trusted MemoryTools implementation uses
`memory.` bindings through it without another endpoint, grant kind or Artifact
authorization mode, and their bytes and immutable revisions remain ordinary
ArtifactStore behaviour. Authenticated Run owners and operator diagnostics may
inspect the underlying Artifact metadata and revisions. The note mapping,
bounds, hidden CAS and response-loss reconciliation are owned by
[Memory](../tools/memory.md).
<!-- from: 03#25, 08#15, 03#106 -->

Rationale: this keeps revisions out of model context without adding another
network service or Artifact authorization mode.
<!-- from: 08#15 -->

**ART-R22.** An ordinary artifact named `memory.*` in the
purpose-reserved `inputs`, `outputs` or `skills` Namespace is governed by that
Namespace's own contract and is not a Memory note.
<!-- from: 08#15 -->

**ART-R23.** The `tool-invocation.` name prefix in RunScope is
reserved for the Runtime-owned receipts of model-free tool Workers
([Scanners](../tools/scanners.md)); bindings with that prefix cannot be
declared as Stage result or Workflow output targets.
<!-- from: 03#25, 03#106 -->

**ART-R24.** HTTP response-body bindings use the reserved
`http.body.` name prefix ([HTTP and Caido](../tools/http-caido.md)): generic
model-visible Artifact tools neither list nor read or write them. This is a
Toolset authority boundary, not a new Artifact API or storage class.
<!-- from: 11#34 -->

**ART-R25.** The UserScope Namespace `audit-standards` is
reserved for curated Audit standard packages. Generic Artifact APIs cannot
write it or read or list raw package bytes; only the trusted catalog seed path
can create a binding. The catalog rules are owned by
[Audits](../audits/audits.md).
<!-- from: 19#34 -->

**ART-R26.** The Server derives and reserves the `audit-*` Project
Namespace prefix. Generic Project list, read, metadata, version, lineage and
mutation routes hide the entire reserved prefix, and ordinary public Project
Artifact mutation cannot write or delete Audit-managed bindings. Owner uploads
continue through non-reserved Namespaces. Frozen Audit bindings, recovery
staging, artifact links and crash replay are owned by
[Review and reports](../audits/review-and-reports.md).
<!-- from: 19#168, 19#169 -->

### Stage Agent Namespaces

**ART-R27.** Workflow Scheduler resolves each Stage Agent's
Namespace from its Stage binding ([Workflows](../configuration/workflows.md))
before allocation: an explicit `namespace` wins; otherwise the logical Agent
name is used. The result travels in AllocationSpec and is fixed when Worker
tools are constructed.
<!-- from: 03#19, 01#94, 03#106 -->

**ART-R28.** Namespace is not a model-selected argument of domain
tools.
<!-- from: 03#27 -->

**ART-R29.** Specialized namespace-bound toolsets must resolve to
unique Namespaces within one Stage. A later or different Stage may assign the
same Namespace to another Agent and continue that data scope.
<!-- from: 03#20, 01#94 -->

**ART-R30.** There is no Namespace-wide writer lock: different artifact
names are independent, while conflicting writes by authorized participants to
the same logical binding use compare-and-swap. Whether multiple Stages of one
Run execute concurrently is a [Scheduler](../execution/scheduler.md) decision
and is not required by this contract.
<!-- from: 03#21 -->

### Tool authority

**ART-R31.** An AgentTemplate may explicitly select two kinds of
model-visible artifact tools:

1. the exact domain tools selected from AgentTemplate Toolsets and fixed to its
   resolved Namespace, for example selected members of
   `openapi_tools(namespace)` or the purpose-specific `memory-tools@1` wrapper
   ([Memory](../tools/memory.md));
2. selected members of the built-in current-Run Artifact Toolset
   `run-artifacts@1`, which exports exactly `list_artifacts`, `read_artifact`
   and `write_artifact` for explicit current-Run refs.

<!-- from: 03#22, 03#23, 01#47 -->

**ART-R32.** No generic Artifact tool is injected implicitly: an
AgentTemplate must select `run-artifacts@1` and list each operation it wants
the model to see ([Toolsets](../tools/toolsets.md)). Selecting only
`list_artifacts` and `read_artifact` exposes a model-visible read-only
interface; omitting `run-artifacts@1` exposes none of the generic operations.
<!-- from: 01#47, 03#24 -->

**ART-R33.** A domain Toolset implementation may use the
allocation-bound private Artifact client internally, within its own registered
contract, without exposing the generic `run-artifacts@1` operations.
<!-- from: 01#53, 03#24 -->

**ART-R34.** There are no separate `read_any_artifact` aliases. The
authority is evident in the explicitly selected generic tool, its explicit ref
and the allocation grant.
<!-- from: 03#33 -->

**ART-R35.** Tool selection and Artifact authorization are
independent checks. Selecting `write_artifact` only constructs and exposes that
model tool; it never broadens the Server-side allocation grant, bypasses the
`outputs` Namespace reservation or crosses the write fence (ART-R86).
Conversely, a grant never causes an unselected tool to be constructed or
exposed.
<!-- from: 01#53, 03#31 -->

**ART-R36.** The generic tools deliberately have broader read
authority inside the current Run, except for the purpose-reserved `skills`
Namespace and the `memory.` and `http.body.` names.
<!-- from: 03#27, 11#34 -->

**ART-R37.** The baseline allocation write grant permits `inputs`
and non-reserved Namespaces, so Workers can modify their working input copies
and intermediate artifacts. It denies writes to `outputs` and `skills`, whose
bindings are created only by trusted Scheduler operations.
<!-- from: 03#27, 00#101, 03#67, 09#58, 03#106 -->

**ART-R107.** Workers may read declared outputs produced by earlier accepted
Stages.
<!-- from: 03#67 -->

**ART-R38.** Generic Worker grants never cross a Run. Of a Project's or an
Audit's artifacts, Workers see only the exact versions forked into RunScope
`inputs`; they have no Audit or Project Artifact authority
([Review and reports](../audits/review-and-reports.md)).
<!-- from: 03#106, 19#169 -->

**ART-R39.** RunScope `skills` bindings are exact forks from the
Run owner's UserScope created only by Scheduler. The trusted Runtime loader may
read exact package refs through its allocation-bound private client, and
allocations read `skills` only through this trusted Skill loading. Package
validation, the owner-UserScope fork and ADK access are owned by
[Skills](../tools/skills.md).
<!-- from: 03#26, 09#58, 03#106 -->

**ART-R108.** The private allocation Artifact API rejects every write to
`skills`.
<!-- from: 03#26, 09#58, 03#106 -->

**ART-R109.** `run-artifacts@1` filters the `skills` Namespace from
model-visible list, read and write operations.
<!-- from: 03#26, 09#58, 03#106 -->

**ART-R110.** Authenticated Run-owner inspection of the exact metadata, lineage
and bytes of `skills` bindings through public Artifact routes remains ordinary
public Artifact behaviour and is not model authority.
<!-- from: 03#26, 09#58 -->

### Private Artifact API

**ART-R40.** The Server's private Artifact API authenticates
the Runtime Agent through mTLS, binds each request to its active allocation and
that allocation's RunScope, enforces the hard Run boundary, reserved-Namespace
policy, grants, byte limits and write preconditions, and streams bytes.
<!-- from: 03#28, 03#100 -->

**ART-R41.** An allocated Runtime Agent uses the private
Artifact API with its existing mTLS identity and active allocation ID; it
receives no Artifact Registry, database, S3 or other blob-backend credential.
<!-- from: 03#100, 03#106 -->

**ART-R42.** The allocation grant records the
certificate-derived Runtime Agent principal and its process `instance_id`
beside `allocation_id`, and each request must match all three. Another
otherwise valid CA-signed Runtime Agent certificate cannot borrow an allocation
ID. The principal has no broader scope or role: the check only proves that the
caller is the process to which Control Plane issued the grant
([Control Plane](../execution/control-plane.md)).
<!-- from: 03#101 -->

**ART-R43.** The allocation remains the path identity, while
the client repeats its process identity in the bounded
`X-Contractor-Runtime-Instance-ID` header. The Server validates that header
against the mTLS principal-bound grant before reading a request body.
<!-- from: 03#102 -->

**ART-R44.** The allocation ID is the only
authority-bearing path value of the private API: neither URL nor body accepts
a Run or User scope selector. Neither the public nor the private interface lets
its caller supply an arbitrary scope ID.
<!-- from: 03#29, 03#100, 03#106 -->

**ART-R45.** The private HTTP binding has one unambiguous
write protocol. Creating a binding requires `If-None-Match: *`; updating it
requires exactly one quoted, strong `If-Match` value containing the expected
opaque revision. A PUT without one of those preconditions, with a weak
validator or with a list of validators is rejected.
<!-- from: 03#29 -->

**ART-R46.** Successful private reads return the resolved revision in
`ETag`; successful writes return the same exact revision in both `ETag` and the
versioned `ArtifactWriteResult`.
<!-- from: 03#29 -->

**ART-R47.** `GET /private/v1/allocations/{allocationID}/artifacts`
supports an unfiltered listing with an optional `namespace` query. Trusted
adapters may instead supply all three of `namespace`, `namePrefix` and
`limit`. This form requires non-empty portable ASCII namespace and prefix
components and an integer `limit` in `1..256` (LIM-ART-3), applies a literal
case-sensitive name prefix in storage, and returns at most `limit` versionless
refs ordered by name. Underscores are literal, not SQL wildcards. Missing
paired parameters, duplicates and invalid values are rejected.
<!-- from: 03#30 -->

**ART-R48.** The filtered listing is a bounded query, not an
assertion that the result is complete: quota-aware callers request one extra
row to detect overflow. Run system Namespaces remain hidden. The filtering adds
no scope selector and does not change the unfiltered client contract.
<!-- from: 03#30 -->

### Versions, reads and writes

**ART-R49.** Artifact content is immutable; only its logical current
binding changes.
<!-- from: 03#83, 03#106 -->

**ART-R50.** Only a complete committed payload is readable.
<!-- from: 03#85 -->

**ART-R51.** A successful RunScope commit is immediately visible to
all authorized participants of the Run. StageResult acceptance is therefore not
a general artifact visibility event; only its optional binding into the
reserved `outputs` Namespace is Scheduler-controlled.
<!-- from: 03#85 -->

**ART-R52.** `list_artifacts` returns versionless refs.
<!-- from: 03#41 -->

**ART-R53.** Reading a versionless ref resolves the current
committed binding and returns the exact versioned ref that was read. Reading a
versioned ref returns that exact historical revision and never falls forward to
the current or a newer binding.
<!-- from: 03#41, 03#86, 03#106 -->

**ART-R111.** In the ArtifactRegistry, `resolve(binding, None)` resolves the
current binding; supplying a revision resolves only that retained historical
binding revision and never substitutes another version.
<!-- from: 03#105 -->

**ART-R54.** `ArtifactReadResult` and `ArtifactWriteResult`
always contain a versioned ref plus the binding and revision timestamps.
<!-- from: 03#35, 03#44 -->

**ART-R55.** ArtifactStore assigns both timestamps from its
authoritative database clock. `binding_created_at` is fixed when the logical
`(scope, namespace, name)` binding is first created. `revision_created_at`
identifies when the exact returned binding revision was committed and advances
on update. They are metadata, not part of ArtifactRef identity or of the
caller-supplied payload.
<!-- from: 03#38, 08#19 -->

**ART-R56.** The raw private GET returns the timestamps as
strict RFC 3339 UTC values in `X-Contractor-Binding-Created-At` and
`X-Contractor-Revision-Created-At`, and the private PUT returns the same two
headers. Both use RFC 3339 with nanosecond-capable precision and canonical UTC
`Z`. The timestamps do not change the JSON bodies or ETags; the trusted client
assembles the conceptual result metadata from the headers and exposes it to
trusted wrappers without changing ArtifactRef or exposing the values as model
arguments.
<!-- from: 03#38, 08#19 -->

Rationale: purpose-specific wrappers such as MemoryTools can project stable
creation and update order without trusting a Runtime Agent clock or adding
another endpoint ([Memory](../tools/memory.md)).
<!-- from: 03#38, 08#19 -->

**ART-R57.** The target `ref` passed to `write_artifact` must be
versionless. A versioned write target is invalid, because `ref.revision` and
`expected_revision` would have conflicting meanings.
<!-- from: 03#42, 03#106 -->

**ART-R58.** CAS is a separate, explicit write precondition:
`expected_revision=None` is create-only, while updating an existing binding
requires the opaque revision returned by a read or write.
<!-- from: 03#42, 03#106 -->

**ART-R59.** A stale expected revision returns `ArtifactConflict`
(FAIL-ART-1). Same-binding conflicts are returned to the caller;
infrastructure never silently overwrites or semantically merges the winner.
<!-- from: 03#42, 03#106 -->

**ART-R60.** CAS is also the response-loss contract. If a create
or update committed but its response was lost, repeating the already-consumed
`If-None-Match: *` or `If-Match: <old revision>` fails with `ArtifactConflict`
and creates no new version. The caller reconciles through a current or exact
read; the Server does not guess whether a different payload under the same
stale precondition was an intended replay (FAIL-ART-1).
<!-- from: 03#43 -->

**ART-R61.** Concurrent first writes to the same scope, and
concurrent writes of identical content to different bindings, must both observe
the committed scope or blob after a uniqueness wait rather than return a false
CAS conflict (FAIL-ART-5).
<!-- from: 03#69 -->

How long an old revision remains resolvable to its immutable internal version
is owned by [Retention and deletion](retention-and-deletion.md).
<!-- from: 03#44 -->

### Payloads and media types

**ART-R62.** One resolved, versioned ref identifies one payload with
one media type. Contractor has no compound artifact or `parts[]` contract;
multi-file output is an archive or a manifest that names exact versioned refs.
<!-- from: 03#45 -->

**ART-R63.** `ArtifactPayload.media_type` is normalized and stored
with every immutable artifact version as a canonical lowercase `type/subtype`
without parameters, each part an RFC 6838 restricted name (a letter or digit
followed by letters, digits and `!#$&^_.+-`), at most 255 characters in total
(LIM-ART-2). The API rejects invalid or parameterized values. The artifact
store, Audit packages, schemas, Runtime and UI share
[these cases](../../../api/testdata/v1alpha1/media-type-cases.json).
<!-- from: 03#37, 00#103 -->

**ART-R64.** Workflow input, Stage-result and Workflow-output slot
validation uses the stored media type and the strict slot matching rules owned
by [Workflows](../configuration/workflows.md).
<!-- from: 03#37, 00#103 -->

**ART-R65.** The generic Artifact payload limit is 64 MiB
(67,108,864 bytes) (LIM-ART-4), enforced by the public and private APIs,
clients and persistence. Source archive uploads, including source ZIPs, may use
the full generic limit.
<!-- from: 03#95, 23#13 -->

**ART-R66.** Specialized package and tool limits remain
independent of the generic payload limit: raising the transport limit does not
raise Skill or Audit package budgets, model disclosure (model-context) limits
or workspace expansion limits.
<!-- from: 03#95, 23#13 -->

**ART-R67.** ArtifactStore and its APIs do not interpret
domain artifact semantics. The generic Artifact API enforces bytes, CAS, owner
isolation and immutable history but does not infer, for example, that an
artifact is a Skill ([Skills](../tools/skills.md)).
<!-- from: 03#103, 09#50 -->

**ART-R68.** A source tree, repository archive, OpenAPI document
or other domain input is an ordinary artifact selected for a declared slot.
Generic digest, streaming, retention and size behaviour belongs to
ArtifactStore; safe unpacking or interpretation belongs to the selected Worker
tools and sandbox policy.
<!-- from: 03#61 -->

### Artifact API clients

**ART-R69.** The same contract backs the User and Project
Artifact APIs, Run clients and trusted Scheduler operations, but scope binding
and grants differ: a public user client is bound to its authenticated
UserScope, a Project client is owner-bound to one ProjectScope, and a Worker
client is bound to the RunScope in its allocation. The Project Artifact routes
and their scope binding are owned by [Projects](projects.md).
<!-- from: 03#46, 03#100, 17#18 -->

**ART-R70.** In the public Artifact API, create uses
`If-None-Match: *` and update uses the strong current revision ETag; the
response contains the new exact `ArtifactRef`.
<!-- from: 09#50 -->

**ART-R71.** The public UserScope Artifact list accepts an exact
`namespace` filter and an `excludeNamespace` exclusion. The Server evaluates
the exclusion before keyset pagination, so a page cannot become short or
incorrectly claim completion after browser-side filtering. The exact
`namespace` filter and the exclusion are part of the cursor identity. The UI
routes that use them are owned by [UI](../ui/ui.md).
<!-- from: 06#47 -->

**ART-R72.** Only purpose-specific trusted Server operations may
perform a controlled source-to-RunScope or successful-Run-to-ProjectScope fork,
including Workflow input and output forks ([Projects](projects.md)) and Agent
Skill forks ([Skills](../tools/skills.md)). There is no generic client
operation that accepts arbitrary source and target scopes.
<!-- from: 03#46 -->

**ART-R73.** Workflow Scheduler uses an internal
ArtifactStore capability to verify exact versions, fork UserScope inputs into
RunScope and bind accepted versions into `outputs`. The User Artifact API
exposes a separate, purpose-specific publication operation for an authorized
Run output.
<!-- from: 03#103 -->

### Workflow input forks

**ART-R74.** At Run creation, the caller supplies bindings only for declared
Workflow input slots ([Workflows](../configuration/workflows.md)), as
`ArtifactRef` values in the source scope selected by the endpoint: a
standalone Run selects from the caller's authenticated UserScope, and a Project
Run selects from its owner-authorized ProjectScope ([Projects](projects.md)).
<!-- from: 00#96, 00#113, 03#61, 03#62, 17#21 -->

**ART-R75.** A versionless input selects the binding current at
Run creation and is resolved to that exact revision; a versioned input
explicitly selects that retained historical revision directly.
<!-- from: 03#62, 17#21 -->

**ART-R76.** Before the Run becomes schedulable, and so before the
first Stage is selected, the trusted Run-creation operation records the exact
source version of every supplied input, validates its media type against the
Workflow input slot, then forks it to
`RunScope(run_id): inputs/<workflow input name>`. For a standalone Run that
operation is Workflow Scheduler; for a Project Run it is the Project endpoint,
after verifying ownership ([Projects](projects.md)). In the same
initialization boundary Scheduler pins and forks every selected Agent Skill
into `skills` ([Skills](../tools/skills.md)).
<!-- from: 00#96, 00#113, 03#62, 03#63, 17#21, 03#106 -->

**ART-R77.** A fork creates an independent target binding and a
lineage edge to the exact source version. It may initially reuse the same
immutable blob; copying bytes is not required. The first or any later Run write
creates a new immutable version and advances only the Run binding through CAS;
Run mutations never mutate the source binding.
<!-- from: 03#64, 03#106 -->

**ART-R78.** The initialization of all required inputs is atomic
from the execution model's point of view: no Stage is selected until every fork
succeeds and its exact source version is recorded. Optional missing inputs
remain absent. A failed initialization leaves no runnable partial
RunArtifactSpace (FAIL-ART-2).
<!-- from: 03#65, 03#106 -->

**ART-R79.** Recovery retries the recorded exact source versions;
it never re-resolves a newer current source binding, in UserScope or
ProjectScope, for the same Run (FAIL-ART-2).
<!-- from: 03#65, 17#21 -->

**ART-R80.** Run creation forks each declared input as one ordinary
artifact through this mechanism. It does not follow refs inside a payload,
such as the JSON refs of a findings collection ZIP, and has no
collection-specific recursive input handling
([Findings](../findings/findings.md)).
<!-- from: 27#33 -->

### Stage context artifacts

**ART-R81.** When creating each StageExecution, before
requesting any allocation, Scheduler resolves every present context binding to
its exact current revision, which must resolve to a retained version, and
records that version in the durable StageContext. It records explicit absence
for each missing optional binding, and the resolved artifact mapping passed to
Planner omits those absent entries. A missing required binding prevents Worker
allocation and Planner or Session creation; the abort path and its
`context_artifact_missing` StageTermination are owned by
[Scheduler](../execution/scheduler.md).
<!-- from: 00#99, 03#106 -->

**ART-R82.** Writes during a StageExecution may advance
logical bindings but cannot change the exact refs already present in its
StageContext. Snapshot and retry semantics are owned by
[Scheduler](../execution/scheduler.md).
<!-- from: 00#100 -->

**ART-R83.** `context.artifacts` is data dependency and
provenance, not an authorization allowlist: Worker access remains governed by
the allocation-bound RunScope tools and grants of this document.
<!-- from: 00#100 -->

### Stage results and the write fence

**ART-R84.** Every ArtifactRef in a candidate StageResult must
already contain the revision selected by Planner.
<!-- from: 03#66, 04#37, 03#106 -->

**ART-R85.** When the StageExecution enters `finalizing`,
Workflow Scheduler verifies that each exact `(RunScope, namespace, name,
revision)` exists and resolves to a retained immutable version, validates its
stored media type against the declared result slot, and locks that internal
version for the same transition. It never re-resolves or substitutes the
current binding, even when that binding has advanced since Planner selected the
result; the selected revision need not still be current.
<!-- from: 03#66, 03#86, 04#36, 04#37, 03#106 -->

**ART-R86.** When a StageExecution durably enters `finalizing` or
`aborting`, Scheduler establishes a Server-side write fence for every
associated allocation. From that transition onward the private Artifact API
rejects its writes even if a Worker or A2A Task has not yet stopped
(FAIL-ART-3). Waiting until allocation release is not a conforming
write-revocation boundary; the fence is independent of Worker shutdown and
allocation release.
<!-- from: 03#32, 04#36, 04#38, 03#106 -->

**ART-R87.** Fence establishment is serialized with the
complete Artifact mutation, not merely with grant lookup: when the fence
operation returns, no earlier authorized write may commit afterward
(FAIL-ART-3).
<!-- from: 03#32 -->

**ART-R88.** After the transition to `finalizing`, neither
Planner nor Worker may change the semantic result or its referenced artifact
versions. Worker finalization cannot mutate artifacts.
<!-- from: 03#66, 04#38 -->

### Workflow outputs

**ART-R89.** Declared Run outputs are Scheduler-managed
bindings in `outputs` to accepted versions, created only through the Stage's
explicit `workflowOutputs` mapping ([Workflows](../configuration/workflows.md)).
There is no implicit output mapping based on artifact Namespace, name, list
position or matching slot names.
<!-- from: 00#101, 03#106 -->

**ART-R90.** Scheduler applies the `workflowOutputs` mapping only
for an accepted terminal successful StageResult while the WorkflowRun is still
`running`. In the same database transaction it creates or advances each
`outputs/<workflow output name>` binding to the exact version already validated
for the StageResult, without resolving the source's current binding again, and
validates the destination Workflow output slot. The binding is a metadata fork
inside the same RunScope with lineage edge kind `output_bind` and does not copy
bytes.
<!-- from: 00#101, 03#66, 03#70, 03#86, 04#38 -->

**ART-R91.** If Run cancellation committed first, the StageResult
may still be accepted for audit, but its Workflow-output mapping is not applied
and it cannot create Run output bindings (FAIL-ART-4). Its effect on Workflow
advancement is owned by [Scheduler](../execution/scheduler.md).
<!-- from: 03#66, 04#38, 03#106 -->

**ART-R92.** Accepting the StageResult that causes success,
applying its Workflow-output mapping, freezing all required outputs and
changing the WorkflowRun `running -> succeeded` are one database transaction:
RunStore and ArtifactRegistry participate in the same PostgreSQL transaction
whenever Scheduler accepts a StageResult, applies Workflow output mappings and
possibly makes the WorkflowRun terminal. A storage split must preserve the same
atomic contract rather than exposing a partially mapped terminal Run
(FAIL-ART-6).
<!-- from: 00#128, 03#67, 03#99, 03#106 -->

**ART-R93.** Once a Run succeeds, all required output bindings
are present and their exact versions are retained in Run provenance (LIFE-ART-2).
<!-- from: 03#67 -->

**ART-R94.** When a Run instead becomes `failed` or `cancelled`,
any output bindings accepted from earlier Stages are also frozen for audit;
they do not constitute successful Run completion and are not implicitly
published (LIFE-ART-2).
<!-- from: 03#67 -->

**ART-R95.** Frozen Run outputs are never implicitly
published or overwritten in UserScope; a successful standalone Run does not
implicitly overwrite the user's artifact library.
<!-- from: 00#128, 03#70 -->

**ART-R96.** A Run with no Artifact bindings has an empty output
set. Freezing that set is a successful no-op; absence of an Artifact scope row
cannot prevent a terminal WorkflowRun transition.
<!-- from: 03#69 -->

**ART-R97.** Manual continuation of an eligible failed Run
([Runs](../execution/runs.md)) is the explicit exception to output-binding
freezing. The same transaction records the continuation receipt, creates the
new attempt and unfreezes that Run's `outputs` bindings without changing their
current revisions; later Scheduler publication may advance them normally.
Immutable artifact revisions, terminal Stage facts and other frozen bindings
remain protected, and Workers gain no output-write authority. Cancelled,
succeeded and already-published Runs cannot use this exception (LIFE-ART-2).
<!-- from: 03#68 -->

### Publication

**ART-R98.** Publishing a Run result to UserScope is an explicit
controlled fork from the exact Run output version to a caller-selected
versionless `ArtifactRef` in UserScope. Creating a new user binding is
create-only; replacing an existing one requires its current opaque revision as
a separate CAS precondition. Publication records lineage and may reuse the
existing blob.
<!-- from: 03#70, 03#106 -->

**ART-R99.** A successful Project Run additionally performs the
create-only `outputs/<slot>` publication contract of [Projects](projects.md);
it never silently replaces an existing Project binding or changes the semantic
Run outcome. The successful cross-scope lineage edge has kind
`project_output_publish`, distinct from the Run-internal `output_bind` edge.
<!-- from: 03#70 -->

### Deletion

Deletion of committed artifact versions, including the controlled scope purge,
its gates and who may invoke it, is owned by
[Retention and deletion](retention-and-deletion.md).
<!-- from: 03#88 -->

### A2A and ADK

**ART-R100.** Normal durable artifact bytes stay in ArtifactStore and
are never sent through A2A; A2A normally carries refs.
<!-- from: 02#115, 03#90, 03#106 -->

**ART-R112.** Agent Skill package bytes follow ART-R100: AllocationSpec
carries exact refs and digests, and Runtime reads the bytes through the private
Artifact API ([Skills](../tools/skills.md)).
<!-- from: 02#115 -->

**ART-R113.** Workspace source and state bytes follow ART-R100: their exact
refs are private construction data, while ordinary Stage task content remains
A2A ([Workspace](../tools/workspace.md)).
<!-- from: 02#115 -->

**ART-R101.** Private A2A task input carries exact input refs plus
Server-declared versionless result bindings. A private A2A input may still
carry a versionless ArtifactRef when it deliberately means "read current"; this
is unrelated to the catalog instruction-resource refs used by Workflow and
AgentTemplate configuration.
<!-- from: 03#90 -->

Runtime, not the Worker model, matches each declared result binding to the
exact versioned ref observed through completed allocation-bound tool calls in
the same invocation; that projection and the Worker model boundary are owned by
[Worker](../execution/worker.md).
<!-- from: 03#90, 14#35 -->

**ART-R102.** ADK `BaseArtifactService` is an adapter, not the
authoritative contract:
<!-- from: 03#91 -->

```python
class AdkArtifactServiceAdapter(BaseArtifactService):
    store: ContractorArtifactClient
    run_id: RunId
    stage_execution_id: StageExecutionId
    principal: ArtifactPrincipal
```

<!-- from: 03#92 -->

**ART-R103.** Planner and Worker Runners receive an explicitly
configured adapter connected to the shared RunArtifactSpace; an exported ADK
A2A Worker must not fall back to an in-memory artifact service.
<!-- from: 03#93 -->

**ART-R104.** One ADK `types.Part` maps to one Contractor payload.
<!-- from: 03#93 -->

**ART-R105.** ADK `save_artifact` lacks CAS and grant semantics. The
adapter may use it for append-only creation; shared binding updates use the
richer Contractor client with explicit revision preconditions.
<!-- from: 03#94 -->

### Storage boundary

**ART-R106.** The ArtifactRegistry is PostgreSQL, because binding
CAS, scope forks and Stage-output recording need transactions.
<!-- from: 03#97 -->

The replaceable ArtifactBlobStore, its payload backends and their independence
from ArtifactRef are owned by [Blob storage](blob-storage.md).
<!-- from: 03#96, 03#97 -->

## Interfaces

Public User and Project Artifact routes are defined in
[contractor-public-v1.yaml](../../../api/openapi/contractor-public-v1.yaml).
The conceptual data model shared by every client:

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
    binding_created_at: datetime
    revision_created_at: datetime


class ArtifactWriteResult(BaseModel):
    ref: ArtifactRef  # revision is always present
    binding_created_at: datetime
    revision_created_at: datetime
```

<!-- from: 03#35 -->

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
    *, name_prefix: str | None = None, limit: int | None = None,
) -> list[ArtifactRef]: ...
```

<!-- from: 03#39, 03#40 -->

Private allocation Artifact API (ART-R40 to ART-R48,
ART-R56):

| Element | Value |
| --- | --- |
| Authentication | mTLS Runtime Agent principal plus `X-Contractor-Runtime-Instance-ID` |
| Path identity | `allocationID` |
| List | `GET /private/v1/allocations/{allocationID}/artifacts` with optional `namespace`, or all of `namespace`, `namePrefix`, `limit` |
| Create precondition | `If-None-Match: *` |
| Update precondition | exactly one quoted strong `If-Match: "<revision>"` |
| Revision on read and write | `ETag` |
| Timestamp headers | `X-Contractor-Binding-Created-At`, `X-Contractor-Revision-Created-At` |

<!-- from: 03#29, 03#30, 03#38, 03#102 -->

The registry protocol behind ArtifactStore; `resolve` follows ART-R111 (the
matching ArtifactBlobStore protocol is in [Blob storage](blob-storage.md)):

```python
class ArtifactRegistry(Protocol):
    async def commit_version(...) -> StoredArtifactVersion: ...
    async def resolve(
        binding: ArtifactBinding,
        revision: ArtifactRevision | None = None,
    ) -> StoredArtifactVersion | None: ...
    async def fork_version(...) -> StoredArtifactVersion: ...
    async def record_stage_outputs(...) -> None: ...
```

<!-- from: 03#104, 03#105 -->

CAS create, update and response-loss reconciliation (ART-R45,
ART-R60):

```mermaid
sequenceDiagram
    participant C as Caller
    participant A as Artifact API
    C->>A: PUT with If-None-Match *
    A-->>C: ETag revision 1, binding created
    C->>A: PUT with If-Match revision 1
    Note over A: revision 2 commits, response is lost
    A--xC: response lost
    C->>A: repeat PUT with If-Match revision 1
    A-->>C: ArtifactConflict, no new version
    C->>A: GET current binding
    A-->>C: ETag revision 2
```

<!-- from: 03#29, 03#42, 03#43 -->

## Limits, failures and errors

Limits ([registry](../platform/limits.md)):

- LIM-ART-1 — `namespace` and `name`: 1 through 128 characters
  (ART-R11).
- LIM-ART-2 — media type: at most 255 characters (ART-R63).
- LIM-ART-3 — filtered private list `limit`: 1 through 256 refs
  (ART-R47).
- LIM-ART-4 — generic Artifact payload: 64 MiB (67,108,864 bytes)
  (ART-R65).

<!-- from: 03#10, 00#103, 03#30, 03#95, 23#13 -->

Errors ([registry](../platform/errors.md)):

- `ArtifactConflict` — a stale or already-consumed CAS precondition; no new
  version is created (ART-R59, ART-R60).
- `context_artifact_missing` is owned by [Scheduler](../execution/scheduler.md).

<!-- from: 03#42, 03#43, 00#99 -->

Failure outcomes ([failure model](../platform/failure-model.md)):

- FAIL-ART-1 — CAS write with a stale precondition or a lost response
  (ART-R59, ART-R60).
- FAIL-ART-2 — Workflow input initialization and its recovery
  (ART-R78, ART-R79).
- FAIL-ART-3 — allocation Artifact write after the write fence (ART-R86,
  ART-R87).
- FAIL-ART-4 — Workflow-output mapping after Run cancellation committed
  (ART-R91).
- FAIL-ART-5 — concurrent first writes and identical-content writes
  (ART-R61).
- FAIL-ART-6 — StageResult acceptance, output mapping and terminal transition
  (ART-R92).

<!-- from: 03#42, 03#43, 03#65, 03#32, 03#66, 03#69, 03#99 -->

Lifecycle ([registry](../platform/lifecycle.md)):

- LIFE-ART-1 — `contractor-system` bindings (ART-R17).
- LIFE-ART-2 — RunScope `outputs` bindings (ART-R93,
  ART-R94, ART-R97).
- Artifact versions, RunScope and ProjectScope deletion and retention are
  owned by [Retention and deletion](retention-and-deletion.md).

<!-- from: 03#18, 03#67, 03#68, 03#44 -->

## Security

The security-relevant rules of this document are:

- callers never select a scope: ART-R3, ART-R7,
  ART-R10, ART-R44, ART-R69,
  ART-R72;
- Runtime Agent authentication and allocation binding: ART-R40,
  ART-R41, ART-R42, ART-R43;
  certificate identity itself is owned by [Security](../platform/security.md);
- model tools versus grants: ART-R35, ART-R36,
  ART-R37, ART-R38, ART-R83, ART-R108, ART-R110;
- write revocation at finalization: ART-R86, ART-R87;
- reserved names hidden from model tools and generic routes:
  ART-R17, ART-R19, ART-R24, ART-R25, ART-R26, ART-R39, ART-R109;
- delete authority: [Retention and deletion](retention-and-deletion.md).

<!-- from: 03#4, 03#9, 03#28, 03#100, 03#101 -->

## Invariants

These properties follow from the cited requirements, which are authoritative.

1. One ArtifactStore serves every scope; scopes are views, not separate
   storage services (ART-R1).
2. ArtifactRef contains Namespace, name and an optional opaque revision;
   authenticated context supplies scope and callers cannot select another
   scope (ART-R7, ART-R44).
3. All Stage participants in one Run use the same RunArtifactSpace
   (ART-R5).
4. Every required input is forked at an exact revision before the Run is
   schedulable; Run mutations never modify the source binding
   (ART-R76, ART-R77).
5. Generic Worker grants never cross a Run and cannot write `outputs`
   (ART-R37, ART-R38).
6. Allocation writes are rejected from the durable transition to `finalizing`
   or `aborting`, independently of Worker shutdown and allocation release
   (ART-R86).
7. A domain tool Namespace is fixed before Worker allocation and cannot be a
   reserved Namespace (ART-R16, ART-R27).
8. Artifact content is immutable; only its logical current binding changes
   (ART-R49).
9. A versionless ref resolves current; a versioned ref resolves exactly and
   never falls forward (ART-R53, ART-R111).
10. Writes require a versionless target and a separate explicit CAS
    precondition; same-binding conflicts are returned to the caller
    (ART-R57, ART-R58, ART-R59).
11. Every candidate StageResult ref is versioned before `finalizing`, and
    Scheduler verifies that exact revision rather than resolving current
    (ART-R84, ART-R85).
12. Declared Run outputs are Scheduler-managed bindings to accepted versions
    (ART-R89).
13. Final required-output binding and freezing and `running -> succeeded` are
    one transaction; a StageResult accepted after Run cancellation creates no
    Workflow output binding (ART-R91, ART-R92).
14. Run-to-user publication is explicit and revision-checked
    (ART-R98).
15. A2A normally carries refs, not artifact bytes (ART-R100).
16. A Runtime Agent receives no Artifact Registry or blob-backend credential
    (ART-R41).
17. Every present StageContext artifact resolves to an exact retained revision
    verified before allocation; missing optional bindings are recorded as
    absent, and a missing required binding prevents Planner creation
    (ART-R81).
18. `memory.` names in non-reserved RunScope Namespaces cannot be exposed by
    any artifact-backed model Toolset, StageContext declaration or StageResult;
    `tool-invocation.` names cannot be declared result or output bindings
    (ART-R19, ART-R20, ART-R23).
19. RunScope `skills` bindings are Scheduler-created exact forks; allocations
    read them only through trusted Skill loading, never write them, and generic
    Artifact tools cannot observe them (ART-R39, ART-R108, ART-R109).
20. Controlled lifecycle purge is scope-complete and collects only versions and
    blobs with no reference outside its deletion set
    ([Retention and deletion](retention-and-deletion.md)).

<!-- from: 03#106 -->

## Examples

### OpenAPI Workflow

Assume the authenticated user's library contains:

```text
UserScope(user-42)
  projects/payment-service-source
  openapi/payment-service-current
```

<!-- from: 03#71, 03#72 -->

The Run request maps those refs to the Workflow's `source` and
`existing_openapi` input slots. Initialization produces independent bindings:

```text
RunScope(run-17)
  inputs/source
  inputs/existing_openapi
```

<!-- from: 03#73, 03#74 -->

An `oas_builder` Worker in Namespace `oas` reads `inputs/source`, updates
`inputs/existing_openapi` with CAS and may write `oas/source-map`. A later
reviewer may update the same working OpenAPI binding and write
`oas/review-report`. A validator reads the resulting current version and writes
`validation/report`.
<!-- from: 03#75 -->

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

<!-- from: 03#76, 03#77 -->

Here `openapi` and `validation_report` are Stage-local result names, while the
values are exact, versioned `ArtifactRef`s. For example, the producing Stage can
declare this explicit output mapping:

```yaml
workflowOutputs:
  openapi: openapi
  validation_report: validation_report
```

<!-- from: 03#78, 03#79 -->

The mapping keys are Workflow output slots and the values are Stage-local
result names. It causes Scheduler to bind those exact revisions as:

```text
outputs/openapi
outputs/validation_report
```

<!-- from: 03#80, 03#81 -->

Neither user source changed during this Run. If requested after success,
`outputs/openapi` can be explicitly published to a new UserScope ref such as
`openapi/payment-service-v2`; it is not copied back to
`openapi/payment-service-current` automatically.
<!-- from: 03#82 -->

## Non-goals

- ArtifactStore exposes no Worker subscription or event-stream contract.
  Planner coordinates meaningful refs through A2A; Workers may discover current
  state through list and read. Internal registry history and audit events do
  not imply a public change feed.
  <!-- from: 03#87 -->
- Contractor core has no `ProjectSnapshot` or `ProjectSnapshotManifest` domain
  type; Project inputs are ordinary artifacts ([Projects](projects.md)).
  <!-- from: 03#61 -->
