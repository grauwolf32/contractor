# Reuse annotations across Runs

Proposal 0037 · Status: **Draft alternatives; not approved for implementation.**

Recorded on 2026-10-23. This document collects the alternatives discussed for
carrying source annotations between Runs, especially independent checks of
OpenAPI operations that reach shared functions. The recommendation is a narrow,
deterministic merge of annotation-only overlay changes. It is a design
recommendation, not an enabled capability or an implementation commitment.

## User outcome

A later Run can start with annotations produced by earlier research against
the same source baseline. Parallel checks can contribute annotations to a
common function without overwriting one another. Each check still produces
its own evidence, findings, gaps and coverage assessment.

Annotation reuse does not certify the correctness of a prior assertion or that
the current Worker inspected a path. Function descriptions may be shared;
argument states, entry conditions and effective controls remain specific to a
path or call site. An imported annotation is research context.

## Current boundaries

- [Workspace overlays](../spec/tools/workspace.md) already export cumulative
  state and an invocation diff. A declared state input can reconstruct a
  workspace from the exact sources. Cross-Run merge is not defined today.
- The state stores full `write_file` contents, not an annotation operation log.
  Applying two states successively can overwrite one branch with the other.
- [Annotation tools](../spec/tools/code-analysis.md) resolve declarations and
  mutate the effective allocation workspace. They have no persistent annotation
  registry or participation index.
- [taint-trace-from-workspace](../../configs/workflows/taint_trace_from_workspace_v4_memory.yaml)
  declares workspace state/diff outputs. The checked-in
  [OpenAPI operation trace Workflow](../../configs/workflows/audit_openapi_operation_trace.yaml)
  does not currently select annotation tools or declare those exports.
- [Memory](../spec/tools/memory.md) remains scoped to one Run and Agent
  Namespace. None of these alternatives changes Memory lifetime or implicitly
  copies notes into another Run.
- [Findings](../spec/findings/findings.md) distinguish annotation identity from
  participation in research. A comment or `changed=false` response establishes
  neither path coverage nor vulnerability reachability.

## Alternatives

| Option | Representation carried between Runs | Best fit | Main limitation |
| --- | --- | --- | --- |
| A. Sequential overlay reuse | Existing cumulative `workspace_state` | Continue one research chain | Does not combine independent branches |
| B. General three-way merge | Reconstructed trees plus a common ancestor | Workflows that intentionally edit arbitrary source | Textual success does not establish annotation compatibility or preserved behavior |
| C. Annotation-aware overlay merge | Existing states, merged under an additions-only policy | Parallel operation checks on one baseline | Requires structural validation and an explicit common checkpoint |
| D. Structured annotation artifact | Versioned records, optionally projected into source comments | Attribution, selective retrieval and incremental research | Adds a record/export/import contract and reconciliation machinery |

### A. Sequential overlay reuse

Pass the exact source inputs and the previous Run's exact state revision into
the next Workflow's declared workspace state input. The next Run receives a
private workspace and exports a new cumulative state. No merge is needed.

This uses the existing state codec and preparation boundary. The selected
Workflow still needs an optional state input, the corresponding workspace
binding and declared exports; this is not automatically wired today.

It is the smallest option for a deliberate `Run A -> Run B -> Run C` chain.
It serializes research and does not solve the case where A and B independently
annotate the same file. The state also carries any other permitted source
edits, so annotation-only reuse requires a separate validation policy.

### B. General three-way merge

Reconstruct both branch trees, compare them with a declared common ancestor,
and apply a conventional three-way text merge. Changes to distinct paths can
be combined; overlapping replacements, delete/edit and type changes need
explicit conflict handling.

This preserves the generality of workspace artifacts. However, two independent
insertions immediately above the same function can conflict textually even
when both annotations are useful. Conversely, a textually clean merge can
combine inconsistent research assertions or change code behavior.

This option needs its own contract for arbitrary source edits and should not
silently become the fallback for a failed annotation merge. Conflict markers
must never enter a successful workspace artifact.

### C. Annotation-aware overlay merge — recommended first parallel option

Consume exact state artifacts and allow only additions of canonical annotation
comments relative to one known common checkpoint. Keep the existing overlay
format as the interchange representation. Publish a new cumulative state and
a bounded merge report as ordinary artifacts.

The merge is deterministic code, not a model judgement. It belongs in an
explicit preparation/merge step using the workspace and annotation boundaries;
the Scheduler does not acquire annotation-specific behavior. No shared writable
workspace, new Artifact endpoint or independent receipt table is needed.

#### Sources and common checkpoint are different identities

Use these names from the workspace contract:

- `S`: the exact hydrated original sources;
- `B`: the common checkpoint, reconstructed from `S` and the exact imported
  state shared by the branches, or `S` when that state is absent;
- `F_i`: a branch's final tree, reconstructed from `S` and its cumulative state;
- `M`: the merged final tree.

Each cumulative artifact describes `S -> F_i`. Merge eligibility and additions
are checked against `B -> F_i`. The output describes `S -> M`.

Matching `baseWorkspaceDigest` alone proves a shared `S`, not a shared `B`.
The merger needs trusted exact input/checkpoint provenance for every branch.
Otherwise an annotation added in an earlier checkpoint and then removed by one
branch can be mistaken for an independent addition and resurrected by union.
Version one rejects branches without the same provable checkpoint. It does
not guess an ancestor or rebase branches that started from different overlays.

#### Merge procedure

1. Pin a finite manifest of source, common-checkpoint and branch ArtifactRefs.
   Resolve ownership and scope through the existing artifact boundary. Verify
   each state's schema, base digest, result digest and canonical operations.
2. Reconstruct `B` once and identify paths changed by each branch relative to
   it. Reuse the source tree rather than reparsing every unchanged file per
   branch.
3. Validate every branch change as additions-only canonical annotations. Use
   language parsing and declaration attachment rules, not a search for text
   that resembles `@trace` inside strings. Removing only the accepted new
   comment lines must recover the checkpoint file byte for byte. Preserve
   annotations and unrelated comments already present in `B`.
4. Match additions to unambiguous declarations in `B`. A line number from an
   already annotated branch or a temporary graph symbol ID is not sufficient.
   Equal additions collapse; compatible additions are retained together.
5. Preserve the checkpoint's existing order. Give new additions a documented,
   deterministic order independent of branch completion order. Revalidate the
   resulting placement and comment inertness: individually safe insertions can
   become unsafe when reordered, including first-line encoding directives.
6. Validate the whole result against workspace limits. Encode one canonical
   `S -> M` state with recomputed digests. Publish the state only after the
   entire merge succeeds; publish conflict diagnostics separately on failure.

The narrow initial policy rejects code edits, file creation/deletion, type
changes, and removal or modification of checkpoint annotations. A later policy
may extend these cases only with an explicit decision and tests.

#### Compatibility and conflicts

| Case | Proposed result |
| --- | --- |
| Two branches add the same canonical line to the same declaration | Keep one line; retain both source references in the merge report |
| Branches annotate distinct declarations or files | Keep both after additions-only validation |
| One function has trace lines for different targets | Keep each line; do not combine targets into one field |
| Same declaration and trace target have different argument states or callees | Conflict; do not pick the latest writer or have a model choose silently |
| Distinct canonical validation or sink lines | Permit coexistence under the existing tool contract; do not infer semantic agreement |
| A branch removes or rewrites an annotation present in the checkpoint | Conflict; version one has no deletion/tombstone policy |
| Non-annotation bytes change, the location is ambiguous, or placement is unsafe | Reject the merge without a usable merged state |

`target` remains model-authored annotation content. It cannot substitute for
trusted Audit item identity or establish which operation checked a function.
The report records exact branch artifacts and origin Run identifiers; it is
not a claim that every contributing Run verified every resulting annotation.

The union is idempotent for duplicate inputs and insensitive to input ordering
within one declared source/checkpoint context. Reusing a previously merged
state as an input requires retaining that same ancestor context and provenance;
arbitrary nesting or different checkpoints has no implicit merge guarantee.

### D. Structured annotation artifact

Export a specialized, versioned artifact containing annotation records. A
record would bind a source baseline, an unambiguous declaration location,
canonical content and trusted producer provenance. Identity would account for
baseline, location and normalized content; stability across arbitrary
refactoring is not required in the first version.

Runtime would assemble accepted tool records and reconcile them with the final
workspace snapshot before export. Rolled-back or deleted comments must not
remain active records. Existing comments with unknown origin cannot be
presented as tool-verified research. Model text and diagnostic metrics are not
authoritative reconstruction sources.

At the next Run's preparation boundary, validate an exact selected revision
and optionally materialize comments into a private overlay. Records can also
be read selectively without inserting every comment into source.

A separate participation relation could associate a record with a Run, task,
path or explicit research result. Import, observation, use as context and
verified coverage must remain distinguishable. This option becomes useful
when the product needs attribution, selective retrieval, annotation
replacement/deletion or incremental reconciliation across source changes.

## Shared publication, failure and lifecycle rules

All alternatives use exact immutable artifact revisions and the existing
[ArtifactStore](../spec/artifacts/artifact-store.md). A Run's inputs do not
change when another branch finishes. Each parallel batch reads a pinned
checkpoint; a later batch may consume a newly accepted merged revision.

[Project output publication](../spec/artifacts/projects.md) is currently
create-only, and Audit child Runs suppress ordinary Project-output publication.
This proposal therefore does not assume a mutable, automatically advancing
`latest annotations` binding. Selection/import is explicit. Automatic promotion
would need its own CAS and acceptance decision; see the separate
[fast-forward output proposal](0035-fast-forward-project-outputs.md).

| Outcome | Proposed handling |
| --- | --- |
| Successful merge | Publish the complete canonical state and report with exact lineage |
| Missing, foreign, corrupt or incompatible explicitly selected input | Permanent preparation failure; no silent omission of that branch |
| Semantic/structural conflict | Retain bounded diagnostics and input references; publish no usable merged state |
| Transient artifact/storage failure | Bounded retry of the same pinned inputs through existing artifact mechanisms |
| Publication may have committed | Reconcile through the existing keyed publication and exact artifact content; do not add an append receipt mechanism |
| Cancellation or deadline | Join parsing and workspace work before cleanup; no usable partial result or advancing shared checkpoint |
| Previously unseen result under an occupied publication identity | Explicit conflict; no overwrite |

Retain or fork source/checkpoint/state payloads needed for future reconstruction
through existing scope retention and exact artifact holds. Deleting a source
Run must not leave an accepted downstream snapshot unusable. Research evidence
needed by a finding must already have retained exact references when the finding
is published; a future merged snapshot cannot retroactively become evidence.
[Retention and deletion](../spec/artifacts/retention-and-deletion.md) continues
to own purge and garbage collection.

## Work bounds and verification before implementation

Reuse the existing workspace reconstruction, per-file annotation and overlay
payload ceilings from the [limits registry](../spec/platform/limits.md).
Before implementation, define additional finite caps for branch count,
aggregate input bytes, annotation count, parsing work, report bytes and one
merge deadline. A per-artifact size ceiling alone does not bound a merge.
Explicit input manifests replace scans of all Project Runs. Parsing and large
tree processing run off the Runtime event loop; avoid whole-tree validation
after every annotation or branch.

The selected alternative needs regression coverage for:

- sequential exact-state transfer and reconstruction of cumulative output;
- local/memory workspace parity and the selected Workflow's real process path;
- distinct files, the same declaration, duplicate inputs and reordered inputs;
- baseline annotations, decorators, duplicate symbol names, LF/CRLF and
  language-active comment/encoding cases;
- different checkpoints over the same source and deletion-resurrection cases;
- changed trace bodies, code edits, deletion/type conflicts and ambiguous anchors;
- exact byte/count limits and realistic multi-operation workloads;
- cancellation, failed reads, ambiguous publication and source Run deletion;
- independent path coverage: reuse must not manufacture completed checks or
  deduplicate findings solely because they share a function.

## Recommendation and decision points

Use A for a deliberate sequential chain. For parallel operation checks, start
with C over one baseline and a proven shared checkpoint, keeping the transport
as ordinary workspace artifacts. General arbitrary-edit merge B is a separate
capability. Move to D when a concrete consumer needs attribution, selective
retrieval or updates that snapshot union cannot express safely.

Owners for a future implementation are code analysis/annotations and
workspaces, with ArtifactStore and Audit owners for publication, checkpoint
provenance, retention and result references. Before creating implementation
tasks, decide the first Workflow variant, exact input/report schemas, budgets,
common-checkpoint proof, conflict presentation and publication receipt boundary.
The existing OpenAPI trace variant needs explicit annotation selection and
state/diff outputs if it is chosen as the producer.

The recommendation should be revisited if representative checks rarely share
code, the cost of importing annotations exceeds the saved research, merges
frequently conflict, or users need per-operation attribution. Compare repeated
samples with and without reuse at the same configured model and research
budget, measuring useful results, false positives, gaps, repeated reads and
total cost including merge preparation.

Cross-baseline rebasing, shared writable workspaces, automatic coverage reuse,
cross-Run Memory, implicit finding grouping and automatic global knowledge
promotion are outside this proposal's first implementation scope.
