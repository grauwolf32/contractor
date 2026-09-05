# 17 — Projects, reusable artifacts and the global queue

Status: **Working agreement**

Depends on: [00](00-workflow-and-planner.md),
[03](03-artifact-plane.md), [06](06-server-ui-and-operations.md),
[07](07-runtime-labels-and-infrastructure-config.md),
[09](09-agent-skills.md) and [16](16-run-metadata-labels.md)

## Purpose and boundary

A Project is an owner-scoped, long-lived workspace that groups reusable
artifacts and the ordinary WorkflowRuns created from them. It is primarily an
organization and reuse surface for Web UI. It is not a Workflow, Scheduler,
queue, execution state machine, authorization role or Runtime boundary.

A WorkflowRun may still be created without a Project. Such a Run is
`standalone`; the UI may group standalone Runs, but Server does not create a
physical or privileged default Project for them.

Projects do not introduce `ProjectSnapshot`, `ProjectSnapshotManifest` or a
compound source domain object. Exact `ArtifactRef` revisions and the existing
Run snapshot remain the execution provenance.

## Project resource

The durable owner-scoped resource is:

```text
Project
  project_id       opaque stable ID
  owner_id         derived from authentication
  kind             project | evaluation
  name             non-empty display name
  description      optional bounded text
  http_target      optional safe URL + RuntimeCredential ref
  lifecycle_state  active | deleting
  revision         opaque optimistic-concurrency token
  created_at
  updated_at
```

`kind` is immutable. While active, `name` and `description` may be updated with
`If-Match` against the current Project revision. Deletion is a durable
cancel/drain/purge operation rather than a synchronous foreign-key cascade;
[18](18-run-and-workspace-lifecycle-controls.md) owns its contract.

`evaluation` is the same storage and execution composition with a distinct UI
entry point. It does not create an eval-only Scheduler path. The Evals view
uses Project kind plus the generic `purpose=eval` and `eval.*` Run-label
convention in [16](16-run-metadata-labels.md).

The public Project collection is owner-isolated and supports a `kind` filter.
The Projects UI asks for `project`; the Evals UI asks for `evaluation`.

## ProjectScope

The one physical ArtifactStore gains a third structural scope:

```text
ProjectScope(project_id)
```

Only the Project owner may list, read, create or revision-check update its
bindings. An `ArtifactRef` remains exactly `(namespace, name, revision?)`;
scope is derived from the authenticated Project endpoint and is never encoded
inside the ref.

ProjectScope accepts arbitrary artifact names and supported media types. The
Server has no closed enum of `sources`, `openapi`, `likec4`, `docs` or `diff`.
Those are useful UI affordances and interoperable naming conventions, not an
authorization or schema boundary. A generic Other upload path is always
available.

The UI presents the initial shortcuts below. Clicking one opens the same
Project Artifact upload dialog with a suggested namespace, media type and icon;
all suggested fields remain reviewable before upload.

| Shortcut | Suggested namespace | Typical media types |
|---|---|---|
| Sources | `sources` | `application/zip`, source text |
| OpenAPI | `openapi` | `application/yaml`, `application/json` |
| LikeC4 | `likec4` | LikeC4/plain UTF-8 text |
| Docs | `docs` | Markdown, text, PDF |
| Diffs | `diffs` | unified text diff |
| Other | caller-selected | any ArtifactStore-supported type |

The shortcut classification is presentation metadata inferred from namespace,
media type and known Workflow slots. It is not copied to AllocationSpec and a
Worker never receives an icon/category enum.

Project Artifact APIs reuse the User/Run Artifact representation, streaming,
ETag/CAS, revision history, bounds and content-disposition behavior. They are
scope-bound below `/v1/projects/{project_id}/artifacts`; no request accepts an
arbitrary `scope_kind` or `owner_id`.

## Creating a Run from a Project

`POST /v1/projects/{project_id}/runs` accepts the ordinary create-Run body and
idempotency header. The selected Workflow, string parameters, Runtime labels,
metadata labels and execution overrides retain their existing meanings. Only
the source scope of `artifacts` changes:

```text
standalone POST /v1/runs
  artifacts refs resolve in UserScope(owner_id)

project POST /v1/projects/{project_id}/runs
  artifacts refs resolve in ProjectScope(project_id)
```

The Project endpoint verifies ownership, resolves every supplied versionless
ref to its then-current exact revision, validates the Workflow input slot and
forks that exact source to `RunScope(run_id): inputs/<slot>`. A versioned ref
selects that retained revision directly. Recovery uses the recorded exact
source; it never follows a later Project binding.

The resulting WorkflowRun stores nullable `project_id`. This is immutable
membership and query metadata, not a Scheduler input. Scheduler, Planner,
Control Plane and Runtime execute a Project Run exactly like a standalone Run.
Runtime still sees only RunScope and has no Project API authority.

A Project Run cannot source an input directly from another Project or mix
UserScope and ProjectScope refs in one request. The user may explicitly fork or
upload the desired artifact into the Project first. Global Skills are the
deliberate exception handled by trusted Run initialization: they continue to
resolve from the owner's UserScope under [09](09-agent-skills.md) and are not
Project artifacts.

## Workflow compatibility and recommendations

Workflow recommendations are a UI read projection over the exact published
Workflow contracts and the Project's current Artifact bindings. They never
replace Server validation.

For each required Workflow input slot, the UI finds Project artifacts whose
stored media type satisfies the strict slot matching rules in [00]. The
Workflow is runnable when every required artifact slot has at least one
candidate. One candidate is preselected; multiple candidates require an
explicit user choice. Missing optional inputs do not block the suggestion.
Required string parameters remain form fields and do not make artifact
compatibility unknowable.

Each Workflow card has its own Run action. There is no batch ProjectLaunch
entity and no implicit chain of all compatible Workflows. The ordinary
Workflows view and a Project's **All workflows** view always permit an explicit
Run, including recomputation.

Workflow output slots gain optional `primary: true`. Omission means `false`.
This flag is presentation/publication intent only; it does not alter Stage
acceptance or Run success. Built-in generator Workflows use canonical semantic
slot names, including `openapi`, `likec4`, `docs` and domain-qualified report
names. Internal handoff slots such as `workspace_state` and `workspace_diff`
are not primary.

A compatible Workflow is omitted from the default Recommended group when all
of its primary destinations already exist at
`ProjectScope(project_id): outputs/<slot>`. If it has no primary output it is
not suppressed by this rule. A partially complete set remains recommended.
This is only a convenience heuristic: stale content, changed inputs or changed
Workflow versions may still make **Run again** useful.

## Successful Project outputs

Every successful Project Run retains its exact frozen outputs in RunScope.
Additionally, Server attempts a controlled metadata fork for each declared
Workflow output to:

```text
RunScope(run_id): outputs/<slot>@<exact revision>
  -> ProjectScope(project_id): outputs/<slot>
```

Automatic publication is create-only. It fills a missing Project binding but
never silently replaces an existing result. Concurrent successful Runs may
race; exactly one creates the destination and the other records a publication
conflict while both Runs remain `succeeded`. A Run that was cancelled or failed
does not auto-publish.

The terminal Run transaction records one bounded publication result per present
declared output (`published`, `already_present` or `failed`) together with the
exact source and, when published, target revision. A legitimately absent
optional output has no receipt because it has no exact source revision. An
existing Project output or a publication failure cannot rewrite semantic
WorkflowRun success. The UI exposes the result and may offer an explicit
CAS-protected replacement using the ordinary Project Artifact publication
operation. Retrying publication is idempotent against the recorded exact
source.

This rule publishes technical declared outputs as reusable artifacts as well;
only `primary` outputs affect recommendations. A future Workflow composition or
join feature may select exact `workspace_state`/`workspace_diff` revisions
without changing Project or Scheduler identity.

## URL and authentication configuration

A Project may store one optional application URL and a reference to an existing
encrypted `RuntimeCredential`. The public field is `httpTarget`; setting or
clearing it is part of the same revision-checked Project PATCH as display
metadata. The URL is safe Project metadata; secret
material is not an Artifact, parameter, metadata label or Project response.
The Web UI edits this pair in a dedicated dialog and may show the value while
the user is entering it. After submission, public read APIs return only the URL
and credential ID/kind, never plaintext.

The URL is a bounded absolute HTTP(S) URL without userinfo, query or fragment.
Authorization scope is its normalized exact origin: scheme, lowercase host and
effective port. Its path is safe application metadata but does not narrow the
authentication scope.

This feature reuses the existing RuntimeCredential encrypted store and
lifecycle. Target-origin Basic and Bearer authentication use
`http-origin-basic@1` and `http-origin-bearer@1`; forward-proxy credentials must
not be reinterpreted as origin authorization. Attaching a credential validates
its exact active ID/kind under the shared credential-reference fence. Deletion
requires first detaching it from every Project and remains blocked while a
nonterminal Run or unreleased allocation pins it.

At Project Run creation, Server pins the safe URL and credential reference into
the immutable Run input/configuration provenance, exposed safely as
`projectHttpTarget`. After placement, Control Plane decrypts it only for a
logical agent whose resolved AgentTemplate selects `http_request`; other agents
in the same Stage receive no target setting. The typed resolved value is put inside
`AllocationSpec.runtimeSettings`. It travels over mTLS, remains
allocation-private in Runtime memory and is erased on finalize, abort, release
or lease loss. It is never rendered into Planner/Worker prompts, ADK State,
events, metrics, errors or durable sessions.

At Runtime, configured target authorization replaces a model-supplied
`Authorization` header only on that exact origin. It is not injected on another
port or subdomain and is removed before following a cross-origin redirect.
Model-visible HTTP session inspection reports neither the target nor its
credential. Allocation cleanup clears the target URL, derived Authorization
header and source secret before slot reuse.

The UI can select an existing origin credential or create one from Basic/Bearer
material inside the target dialog. It clears entered secrets on submission and
never writes them to URL state, browser storage or query-cache response data. If
credential creation succeeds but the Project CAS update fails, the safe active
credential ID is shown so the user can select it and retry without resubmitting
the secret.

The first implementation may support one target and Basic/Bearer auth. Multiple
named targets, interactive OAuth and an in-memory-only Server credential backend
are later extensions; they do not require a new Project artifact type.

## Global queue

Queue is a top-level UI/read API over ordinary nonterminal WorkflowRuns across
standalone Projects, evaluation Projects and standalone execution:

```text
initializing | running | cancelling
```

It is not another durable queue table or Scheduler. Scheduler continues to
claim WorkflowRuns using its authoritative lifecycle query. Queue items expose
Run identity/state/timestamps, Workflow identity and optional safe Project
identity/kind. They do not promise an exact numeric position or start time:
recovery, cancellation priority, claims, Runtime capability placement and
another Server process can change which Run advances next.

The owner-scoped queue endpoint uses stable keyset pagination and never exposes
foreign Projects or Runs. Live invalidation may reuse the existing Run event
channel; polling remains a correct fallback. The consolidated Runs UI and the
durable owner pause/resume gate are specified in
[18](18-run-and-workspace-lifecycle-controls.md).

## UI information architecture

The target top-level navigation is:

```text
Projects
Evals
Runs
Workflows
Artifacts
Skills
Operations
```

Project detail contains Overview, Artifacts and Runs plus Recommended/All
Workflow launch surfaces. Evals renders evaluation Projects and eval-specific
label grouping without duplicating execution APIs. Skills is a global
owner-level view over `skills/*` UserScope artifacts and the existing
SkillCatalog rules. A Project may show the Skills required by a candidate
Workflow read-only, but cannot own, copy or override them.

## Failure behavior

| Failure | Result |
|---|---|
| Unknown/foreign Project | owner-safe `404` |
| Stale Project `If-Match` | `409`/`412`; no metadata mutation |
| Missing/incompatible Project input | Run create fails atomically |
| Duplicate Project Run idempotency replay | original Run and Project membership returned |
| Same idempotency key used in another Project or with changed input | conflict |
| Automatic output destination already exists | Run succeeds; publication records `already_present` |
| Automatic publication infrastructure failure | Run succeeds; publication records/retries a bounded failure |
| Project credential missing at Run initialization | Run remains non-runnable and fails initialization safely |
| Credential disappears after immutable Run pin | normal resolved-secret failure; no fallback to another credential |

## Invariants

1. A Project organizes artifacts and Runs; it never executes work.
2. Standalone Runs remain first-class and no hidden default Project exists.
3. ProjectScope is structural authorization; Runtime receives only RunScope.
4. Project input forks and output publication always name exact revisions.
5. Automatic output publication is create-only and cannot change Run outcome.
6. Workflow recommendation is advisory, media-type based and bypassable.
7. `primary` affects recommendation only, not Workflow correctness.
8. Skills are global owner artifacts, never Project-owned copies.
9. Secret target auth uses RuntimeCredential and appears in plaintext only in
   allocation-private RuntimeSettings after placement.
10. Queue is a read model over WorkflowRun lifecycle, not a second Scheduler;
    its pause control is only a durable admission gate under [18].

## First-release conformance

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
