# Finding tools and immutable collections

Status: V43-001 specifies the contract and implements the Go codec, package
validation and deterministic destination mapping. Server publication (V43-002),
Runtime preparation and `list_findings` (V43-003), agent integration (V43-004)
and the process gate (V43-005) remain pending. This document does not activate
a tool, endpoint or configuration version.

## Responsibilities and compatibility

`finding(...)` preserves the existing `security-findings@1` operation and its
allocation-bound intake receipt. The same client key with identical content is
idempotent; changed content conflicts. A proposal may describe an observation,
a hypothesis or a finding with evidence. Hypothesis, proposed checks, annotations
and reproduction instructions remain optional under the generic contract.
Scenario instructions can require more detail without changing the tool schema.

An explicitly selected new version of `security-findings` adds `list_findings`.
AgentTemplate selection can expose creation, reading or both. Selecting reading
does not imply finding emission, Audit confirmation or review authority.
The existing [Audit review contract](19-audits.md) owns those mutations.

A collection is a frozen selection of existing receipts and available review
observations. It is an ordinary input artifact, usable by both ordinary Workflows
and Audit child Runs. It creates no verification inventory or checks. Annotations,
diffs, logs and reports are possible evidence documents; their semantic validation
belongs to the consumer. There is no implicit deduplication by function, subject,
description or evidence digest.

## Package format

Media type: `application/vnd.contractor.findings-collection+zip`.
Reuse the bounded `contractor.audit.package.v1` ZIP transport with a new
`kind: finding-collection` and no `entry_point`. Reusing this transport does not
make the collection an Audit execution result. Existing result importers continue
to require their own package kind and envelopes.

| Member ID | Path | Content |
| --- | --- | --- |
| `collection` | `collection.json` | Canonical collection metadata, `application/json` |
| Each document ID | `documents/<document ID>` | Exact retained proposal or evidence bytes |

The transport manifest is `manifest.json`. Its `package_id` is `collection-`
followed by the full lowercase SHA-256 of canonical `collection.json` bytes.
Each document occurs once, including documents shared by multiple entries.
No undeclared members or unused documents are accepted. ZIP integrity and paths
are validated by the existing package codec before any document is exposed.

Both metadata documents use RFC 8785 canonical JSON: UTF-8, no BOM or trailing
newline. Collection decoding rejects noncanonical representations, unknown or
duplicate keys, wrong key case, missing required fields, invalid Unicode and
unsupported schema versions. Optional members are omitted, never `null` or an
empty placeholder. Embedded proposal bytes retain their original representation
and are independently decoded under `contractor.audit.finding-proposal.v1`.

### Collection metadata

All fields below are required unless marked optional. Arrays are always arrays,
including when empty. ID values use the existing Audit-domain identifier contract
(1–160 ASCII bytes, `[A-Za-z0-9][A-Za-z0-9._:-]*`). Exact ArtifactRefs retain the
Artifact plane's own namespace/name/revision constraints.

| Object | Fields |
| --- | --- |
| Collection | `schema: contractor.findings.collection.v1`, `snapshot_at`, `sources`, `documents`, `entries` |
| Source | `kind: run\|audit`, `id` |
| Document | `id`, `scope: {kind: run\|project\|user, id}`, exact `ref`, `digest`, `media_type`, `size_bytes` |
| Entry | `receipt_id`, `proposal_id`, `run_id`, `invocation_id`, optional `audit_origin`, `retention`, `proposal_document_id`, `evidence`, `reviews` |
| Audit origin | `audit_id`, `execution_id`, `role` |
| Evidence link | `evidence_id`, `document_id` |
| Review observation | `audit_id`, `finding_id`, `revision`, `state`; optional `decision_id`, `assessment_id`, `duplicate_target_id` |

`snapshot_at` is UTC RFC3339Nano ending in `Z`, with insignificant fractional
zeros omitted. It records when the selection was captured; it is not a query for
the latest database state. `sources` records the requested Run/Audit scopes.
Every entry must match a Run source through `run_id`, or an Audit source through
its Audit origin or a review observation. This is a consistency check, not proof
of server authority.

Document `scope` identifies where the exact retained bytes were read. It can be
a producer Run or an authorized retained Project/User artifact. Entry origin
still identifies the producing Run/invocation even when retained bytes came from
an Audit hold in ProjectScope. No owner identity or source scope in this JSON
grants read access.

`retention` captures the existing receipt state: `source-held`, `audit-held` or
`discarded`. It never substitutes for package byte availability. A discarded
receipt can be included only if all selected exact bytes remain available through
an authorized retained copy; otherwise publication fails.

Review `revision` is an integer in 1–9007199254740991. `state` is `proposed`,
`confirmed`, `rejected`, `duplicate` or `needs-evidence`. IDs are copied only when
present in the captured server record. In particular, current projections can
have `duplicate` or `needs-evidence` without `current_decision_id`. The collection
does not infer a decision or assessment. These IDs are provenance, not dereference
capabilities or full decision documents. A later review requires a new snapshot.
An ordinary Run proposal has no invented Audit origin, finding ID or review state.

Sort sources by `(kind, id)`, documents by `id`, entries by `receipt_id`, each
entry's evidence by `evidence_id`, and its reviews by `(audit_id, finding_id)`;
comparison is ascending ASCII lexical order. Each key is unique. Proposal IDs
are also unique across entries. If source selections overlap, the publisher
keeps one receipt entry and merges its distinct review observations. Different
receipts remain distinct even when they have identical subjects or bytes.

An entry's proposal document must be nonempty JSON and decode as FindingProposal.
Its `evidence_ids` set must equal the entry's evidence-link IDs exactly; proposal
array order need not be lexical. The publisher preserves intake's mapping of
`evidence-1`, `evidence-2`, etc. to the receipt's sorted exact refs; it does not
renumber links after sorting the collection. Review observations never replace
or rewrite the original proposal.

### Document identity and bounds

`media_type` is the normalized lowercase MIME type without parameters, as stored
by the Artifact plane. `digest` is `sha256:` plus 64 lowercase hex characters;
`size_bytes` is the exact byte length, including zero for empty evidence.

The document ID is `doc-` plus the full lowercase SHA-256 of the RFC 8785
encoding of this array, in this exact order:

```text
[scope.kind, scope.id, ref.namespace, ref.name, ref.revision,
 digest, media_type, size_bytes]
```

Including scope prevents two Runs with identical binding names and revisions
from colliding, even when their content bytes are identical. Documents with the
same entire identity can be shared within the package.

| Limit | Value |
| --- | --- |
| Sources | 1–64 |
| Entries | 0–256 |
| Documents | 0–1023 |
| Evidence links per entry | 0–256 |
| Review observations per entry | 0–32 |
| Collection JSON | 1 MiB |
| Proposal document | 1 byte–8 MiB |
| Evidence document | 0–16 MiB |
| Sum of document bytes | At most 16 MiB |
| Complete ZIP | At most 16 MiB, including metadata and headers |

The existing transport additionally limits manifest bytes, expansion, JSON depth
and node count. These are simultaneous bounds; the entry limit is not a promise
that every selection of 256 large proposals fits. Oversized selections fail and
can be explicitly narrowed. Empty successful selections contain empty `entries`
and `documents`; missing bytes never turn a nonempty selection into an empty one.

## Server publication and reader preparation

V43-002 owns owner-authorized source selection, snapshot capture, assembly and
publication through the existing Artifact plane. Selection is explicit about
receipts versus admitted Audit findings and any retention/review filters.
Selecting an Audit finding includes all its contributing selected-source
receipts, not just `FirstProposal`. Selection is fully enumerated before fetching
bytes; page limits, missing retained bytes or exceeded package limits cannot
produce a successful partial snapshot.

The server checks source ownership and correspondence to existing records,
captures available review revisions, and reads exact retained bytes with matching
digest, MIME type and size. Publication exposes a complete package atomically.
Retries reuse an already published exact snapshot or a previously fixed selection;
if that selection cannot be recovered exactly, they fail explicitly. They never
silently recompute a newer collection under the same publication identity.
Retention of the published ZIP retains the required bytes independently of
later source-Run deletion. No new findings database is introduced.

The ZIP is passed as a declared ordinary Workflow input. Public Run creation
continues to fork declared artifacts through its existing mechanism. It does not
follow JSON refs or gain collection-specific recursive input handling.

V43-003 owns allocation-local reader-toolset preparation:

1. Resolve the selected declared collection input to an exact current-Run ref.
   Validate the complete package and all proposal/evidence links before writes.
2. For each embedded document, derive a versionless current-Run destination:
   namespace `findings-<full SHA-256 hex of exact ZIP bytes>`, name equal to the
   document ID. `FindingCollectionTargets` specifies this pure mapping; its
   output is a write target, not an exact read receipt.
3. Materialize through the existing allocation ArtifactClient using create-only
   writes. For a replay/conflict, read the existing destination and verify exact
   bytes, digest, MIME type and size. Identical content can be reused; different
   content fails without overwriting it. Record actual returned exact revisions.
4. Expose `list_findings` only after every document has a verified exact ref.
   Failed preparation may leave ordinary unexposed intermediate artifacts;
   replay verifies/reuses them and existing Run cleanup owns their lifecycle.

The toolset may perform this internal materialization even in a reader-only
configuration. This grants no model-visible creation operation. It uses the
existing `WriteInputsAndIntermediates` allocation policy and leaves
`ReadCurrentRun` intact. Source refs stay separate as provenance. No Runtime read
of a foreign Run is needed. Full cross-process readability is a V43-005 gate;
V43-001 tests the deterministic mapping, not artifact I/O.

## Reader tool contract

```text
list_findings(subject_kind?, subject_key?, limit?, cursor?)
    -> {items: [...], next_cursor: string|null}
```

The collection input is fixed by the selected toolset configuration, never by
model-supplied Run, Project, Audit or owner IDs. V1 binds one collection per
prepared reader. Filtering compares the original proposal subject exactly and
case-sensitively. `subject_key` requires `subject_kind`; absent values mean no
filter, while empty strings are invalid. Limit defaults to 20, range 1–100.

Each item returns `receipt_id`, `proposal_id`, `run_id`, `invocation_id`, optional
`audit_origin`, `retention`, the original `subject`, `title_preview`,
`description_preview`, `has_hypothesis`, `proposal`, `evidence` and `reviews`.
`proposal` is `{ref, digest, media_type, size_bytes, source: {scope, ref}}`;
`evidence` items add `evidence_id` to that same shape. The top-level `ref` is the
verified exact consumer ref; `source.ref` preserves the retained source ref.
Reviews have the metadata shape above. Each preview is `{text, truncated}`:
title at most 512 UTF-8 bytes, description at most 2048 bytes, cut only at a code
point boundary. Full proposal text, hypothesis and checks remain available via
`read_artifact(proposal.ref)`; all evidence remains available through its exact refs.

Results retain receipt order after filtering. The canonical JSON response is at
most 256 KiB. Whole items are added until the item or byte bound is reached,
accounting for the final cursor. If one complete item cannot fit, return an
explicit limit error, never an empty page or truncated evidence/review arrays.
`next_cursor` is `null` exactly when no matching entries remain. Empty selections
and filters with no matches return `items: [], next_cursor: null`.

Cursor is unpadded base64url of canonical JSON with exactly:

```text
{version: 1, collection_digest: "sha256:...", subject_kind: string|null,
 subject_key: string|null, after_receipt_id: string}
```

Decoded JSON is at most 1024 bytes; encoded cursor at most 1366 ASCII bytes.
Cursor values must match the bound exact ZIP digest and requested filters, and
`after_receipt_id` must exist among matching entries. Limit may change between
calls. Wrong version, malformed cursor, unknown receipt or changed filters or
snapshot fail explicitly. Cursor is a position token, not an authorization grant;
it does not need a signature because the collection boundary is already fixed.

Invalid arguments, invalid collection/cursor, missing or conflicting prepared
artifacts, byte limits and a successful empty page are distinct outcomes.
Codec errors reuse the existing bounded `audit_invalid`, `audit_schema_unsupported`,
`audit_limit_exceeded`, `audit_reference_invalid`, `audit_package_invalid` and
`audit_digest_mismatch` codes. Tool transport maps these to its existing error
envelope; it never returns source bytes in error text. Preparation and reads cannot
confirm findings or mutate review state.

## Verification and rollout

The [shared fixture](../../internal/auditdomain/testdata/finding-collection-v1.fixture.json)
contains generic function/configuration subjects, ordinary Run and Audit origins,
optional hypothesis, a review without decision ID, and identical evidence bindings
from different Runs. Document and package hashes were independently generated
using Python; Go codec tests require exact conformance and reject inconsistent
metadata, byte content and evidence membership.

V43-002 adds database publication/retention tests. V43-003 consumes this same
fixture for Python decoding, real ArtifactClient preparation behavior, previews,
pagination and independent tool selection. V43-004 selects new immutable config
versions; old versions remain reproducible. V43-005 proves producer → publication
→ ordinary input fork → preparation → reader across Server/Runtime processes.
Model quality evals remain separate and follow the V41 portable-format gate.
