# Fast-forward publication of revised Project outputs

Proposal 0035 · Status: **Draft; not implemented.**

This proposal lets several Workflows refine one Project artifact in turn, for
example an OpenAPI document that one Workflow generates from source and
another refines from recorded traffic, without giving up the guarantee that
automatic publication never silently replaces a Project result. It changes
only how a successful Project Run publishes a declared output; Run execution,
Run-internal output binding and Audit-managed publication are unchanged.

## User outcome

The shipped `openapi-from-workspace` Workflow already reads an optional
`existing_openapi` and writes `openapi`. In a Project:

1. The first Run, without `existing_openapi`, publishes `outputs/openapi` at
   v1.
2. A second Run of the same or another Workflow selects `outputs/openapi` (v1)
   as `existing_openapi` and produces a better `openapi`. The Project moves to
   v2 automatically, because the Run read v1 and v1 is still the Project's
   version.
3. Any later Run that selects `outputs/openapi` gets v2.

If two refinement Runs start from v1, the first to finish moves the Project to
its result; the second does not overwrite it, and its Run shows that the
Project version changed since it started, with a way to compare, to make its
own result the Project version explicitly, or to run again on the current
version. An older Run that finishes late can never replace a newer result.

## Current behaviour and the gap

- Automatic publication is create-only
  ([PRJ-R53](../spec/artifacts/projects.md), LIFE-PRJ-1, FAIL-PRJ-4): the first
  successful Run creates `ProjectScope(project_id): outputs/<slot>`; every later
  Run records `already_present` and its better version stays only in its own
  RunScope.
- The only way to move the Project forward is the ordinary Project Artifact
  write (`putProjectArtifact`) with `If-Match`: download the Run output and
  upload its bytes. That creates a new version with no lineage to the Run that
  produced it, and the UI offers no such action; it shows the publication
  result `Already present` (PRJ-R60 allows an explicit CAS-protected
  replacement but does not define one for Run outputs).
- The Recommended group hides a Workflow whose primary outputs already exist
  (PRJ-R49), so a refining Workflow disappears from the default view exactly
  when it becomes useful.

Create-only exists for good reasons, and this proposal keeps them: two Runs
racing for one destination, a Run that read stale input, or an old Run that
finishes late must not replace a newer result without anyone deciding so.

## Design

The rule follows git's fast-forward: a Run that read the Project's version of
an artifact and produced a new version of it is a commit on top of that
version, and publication may advance the Project binding only while the
binding still points to the version the Run read.

### Declaring a revision

A Workflow output slot may declare that it revises one of the Workflow's input
slots:

```yaml
inputs:
  existing_openapi:
    required: false
    mediaTypes: [application/yaml, application/json]
outputs:
  openapi:
    required: true
    primary: true
    mediaTypes: [application/yaml]
    revises: existing_openapi
```

Catalog load rejects a `revises` that names no declared input slot, an input
slot named by two outputs, and media-type sets of the input and the output
that cannot intersect. Nothing is inferred from matching slot names
([WFL-R72](../spec/configuration/workflows.md)); a Workflow without `revises`
publishes exactly as today.

### Base revision

The base of a revising output is the exact source revision the Project
endpoint resolved for its input slot when the Run was created
([PRJ-R23](../spec/artifacts/projects.md)). It is already recorded: the
`input_fork` lineage edge into `RunScope(run_id): inputs/<slot>` names the
source binding and its revision. The base counts only when that source
binding is the publication destination itself,
`ProjectScope(project_id): outputs/<output slot>`.

### Publication rule

The terminal Run transaction keeps one bounded publication result per present
declared output (LIM-PRJ-2). For an output that declares `revises`, it
locks the destination binding and decides:

| Destination | Effect | Result |
| --- | --- | --- |
| absent | create, as today | `published` |
| current revision is the base | advance to the Run's exact output revision | `advanced` (new) |
| current revision is this Run's own publication (replay) | none | the result first recorded (PRJ-R59) |
| any other revision, or the Run's input did not come from the destination, or the optional revised input was not supplied | none | `superseded` (new) |

Outputs without `revises` keep `published`, `already_present` and `failed`.
An advance is a metadata fork like a create: no bytes are copied, the binding
gains a new immutable revision, the previous revision stays in its history,
and the new revision gets the `project_output_publish` lineage edge from the
exact Run output (ART-R99). A `superseded` or `already_present` result never
changes semantic Run success (PRJ-R58).

### Explicit promotion

A Run result that publication did not apply can be made the Project version
deliberately:

```text
POST /v1/projects/{projectId}/artifacts/outputs/{slot}/promote
If-Match: "<current binding revision>"      (or If-None-Match: * to create)
{"runId": "run_…"}
```

The Server forks the exact frozen output of that succeeded Run of the same
Project into the destination, with the same lineage edge kind. A stale
`If-Match` fails with the ordinary Artifact CAS conflict (`409 conflict`,
ART-R59) and changes nothing. This
is the CAS-protected replacement PRJ-R60 already allows, without downloading
and re-uploading bytes.

### UI

- The Run's publication results show `advanced` as "Project version updated"
  and `superseded` as "Project version changed since this Run started", with
  *Compare*, *Make Project version* and *Run again on the current version*.
- The Project artifact view shows the binding's versions with the Run that
  produced each one (existing `/versions` and `/lineage` reads).
- PRJ-R49 keeps a Workflow in the Recommended group when its existing primary
  destinations are outputs it revises.

### Unchanged

- Inside one Run each `outputs/<slot>` is bound once
  ([ART-R90](../spec/artifacts/artifact-store.md)); revising happens between
  Runs, at the Project level.
- Standalone Runs and UserScope are unaffected.
- `publication_mode=audit-managed` still suppresses publication (PRJ-R62).
- Recovery keeps using the recorded exact source and never follows a later
  Project binding (PRJ-R24).

## Failure outcomes

| Outcome | Behaviour |
| --- | --- |
| Success | `advanced` or `published`, with the target revision and lineage in the terminal Run transaction |
| Definite permanent | Destination moved or input did not come from it → `superseded`; exact source unavailable, Project unavailable or deleting → `failed`, as today; the Run succeeds in every case |
| Retryable | Deadlock, serialization failure, lock, statement or pool timeout, lost connection: aborts the terminal transaction and is replayed under the existing LIM-PRJ-4 bound (FAIL-PRJ-4) |
| Ambiguous | A lost commit acknowledgement is reconciled by the replay, which finds the destination at this Run's own revision and records nothing new |
| Cancelled | Claim loss aborts the terminal transaction like a transient failure; nothing is advanced |
| Timeout | The Scheduler operation budget owns each replay, as today |
| Explicit promotion | Stale `If-Match` → `409 conflict`; Run not succeeded, not in the Project or deleted → not found or invalid; transient failure → the ordinary retryable `503 storage_transaction_conflict` |

## Bounds and lifecycle

- Work per terminal transaction is unchanged in kind: one lock and one
  compare per present declared output, plus the Run's own input lineage for
  revising outputs, bounded by the number of declared inputs.
- Binding history grows by one revision per advance and is retained until
  the Project's ProjectScope purge, like every other Project binding revision.
- Deleting a Run does not change Project binding revisions, which reference
  their own versions; an explicit promotion of a deleted Run is refused.

## Specification changes when implemented

- [Projects](../spec/artifacts/projects.md): PRJ-R53, PRJ-R54, PRJ-R56 and
  PRJ-R58 describe the revising rule and the two results; PRJ-R49 the
  Recommended group; PRJ-R60 names the promotion operation.
- [ArtifactStore](../spec/artifacts/artifact-store.md): ART-R99 becomes
  "create-only, or fast-forward for a declared revision".
- [Workflows](../spec/configuration/workflows.md): the `revises` field and its
  load-time checks.
- Registries: LIFE-PRJ-1 (an existing binding changes only by a declared
  fast-forward or an explicit CAS-protected replacement) and FAIL-PRJ-4.
- Public API: the two publication result values and the promotion operation
  (additive); UI language for both results.
- Schema: the publication journal's status check gains `advanced` and
  `superseded`, and `advanced` records its target revision and base revision.

## Alternatives considered

- **Keep create-only and add only explicit promotion.** Safe and smaller, but
  every refinement needs a manual step, and the common sequential case gains
  nothing.
- **Last writer wins.** Rejected: races and late Runs silently replace newer
  results, which LIFE-PRJ-1 exists to prevent.
- **Advance Workflow outputs inside a Run** (option B of the 2026-10-12
  output-binding decision). Rejected for this purpose: it concerns one Run, and
  iteration across Workflows happens between Runs.
- **Infer the revision from matching input and output slot names.** Rejected
  by WFL-R72: mappings are explicit.
- **Named channels or tags on Project artifacts.** More general than needed;
  the binding history already provides versions.

## Open questions

- Whether explicit promotion should also restore an older revision of the same
  Project binding (rollback), or whether the ordinary write covers that.
- Whether `superseded` should offer an automatic "run again on the current
  version", or only the action described above.
- Whether a revising output should also be allowed for outputs that are not
  primary.

## Required verification

- PostgreSQL: create, advance, replay after a lost acknowledgement, two Runs
  from the same base (one `advanced`, one `superseded`), a late older Run, an
  input from another binding, a deleting Project, Audit-managed Runs, and the
  LIM-PRJ-4 bound with a revising output.
- Catalog load: every invalid `revises` form is rejected.
- Public API: the promotion operation with current, stale and missing
  `If-Match`, a foreign Project's Run and a deleted Run.
- UI: both results and their actions at 390 px in every theme, and the
  Recommended group rule.
