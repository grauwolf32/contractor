# 27 — Finding tools and immutable collections

Status: **Implemented and verified** — see the
[findings contract gate task](../../tasks/v43-005-findings-contract-gate.yml).

The application uses one current finding schema and interface set; old
development shapes are not maintained as compatibility variants.

## Responsibilities

`finding(...)` uses the selected general/code/HTTP interface and its
allocation-bound intake receipt. The same client key with identical content is
idempotent; changed content conflicts. A proposal may describe an observation,
a hypothesis or a finding with evidence. Hypothesis, proposed checks, annotations
and reproduction instructions remain optional under the generic contract.
Scenario instructions can require more detail without changing the tool schema.

`security-findings@1` also exports independently selectable `list_findings`.
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
and are independently decoded under the current `contractor.audit.finding-proposal.v1` contract.

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
| Entry | `receipt_id`, `proposal_id`, `run_id`, `invocation_id`, optional `audit_origin`, optional `audit_holds`, `retention`, `proposal_document_id`, `evidence`, `reviews` |
| Audit origin | `audit_id`, `execution_id`, `role` |
| Evidence link | `evidence_id`, `document_id` |
| Review observation | `audit_id`, `finding_id`, `revision`, `state`; optional `decision_id`, `assessment_id`, `duplicate_target_id` |

`snapshot_at` is UTC RFC3339Nano ending in `Z`, with insignificant fractional
zeros omitted. It records when the selection was captured; it is not a query for
the latest database state. `sources` records the requested Run/Audit scopes.
Every entry must match a Run source through `run_id`, or an Audit source through
its Audit origin, an Audit hold or a review observation. This is a consistency check, not proof
of server authority.

Document `scope` identifies where the exact retained bytes were read. It can be
a producer Run or an authorized retained Project/User artifact. Entry origin
still identifies the producing Run/invocation even when retained bytes came from
an Audit hold in ProjectScope. No owner identity or source scope in this JSON
grants read access.

Optional `audit_holds` contains up to 64 sorted unique Audit IDs with retained
copies of this receipt. It preserves source membership independently of producer
origin and review state; it does not create a finding ID or imply confirmation.
Absent or empty holds encode by omitting the member.

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

The owner-authenticated `POST /v1/finding-collections` request is:

```json
{
  "clientKey": "review-snapshot-1",
  "sources": [
    {"kind": "run", "id": "run-1", "receiptIds": ["receipt-1"], "findings": []},
    {"kind": "audit", "id": "audit-1", "receiptIds": [],
     "findings": [{"findingId": "finding-2", "revision": 3}]}
  ]
}
```

`clientKey` is an Artifact name (1–128 bytes). Selector identities use 1–128
ASCII bytes under the identifier pattern above. Sources are unique by kind/ID;
receipts and finding IDs are unique within a source. Both arrays are required.
Run sources have an empty `findings` array. The request has at most 256 selectors
across all sources; expansion of finding contributions must also fit the package
entry/byte bounds. Overlap across sources is permitted and deduplicates only the
same receipt identity. Input order is normalized for replay identity.

The caller enumerates its desired receipt IDs using existing paginated APIs.
Empty arrays explicitly select nothing; there is no implicit “all”, live filter
or fallback to the first page. Admission/retention/review policy is expressed by
the selected IDs and finding revisions. A missing or foreign source/receipt is
404; a stale finding revision is 409. Selected rejected, unreviewed or discarded
records are not silently skipped. An unavailable exact document fails the entire
publication. Invalid selectors are 400; package codec failures retain their
bounded 422 codes.

A successful response contains `artifact` (`ref`, `digest`, `mediaType`,
`sizeBytes`), `snapshotAt`, `entryCount` and `replayed`. Creation returns 201;
replay returns 200. The exact ZIP is written to the authenticated owner's
`finding-collections/<clientKey>` binding. A small ordinary JSON artifact in
`finding-collection-receipts/<clientKey>` records the normalized request digest
and exact ZIP metadata. Both writes commit together. This receipt is publication
metadata, not a second finding store or review authority. The public API schema
owns the precise HTTP field spelling and error envelopes.

The same key and normalized request replay the original exact ZIP before looking
up sources or current reviews. A changed request conflicts. A pre-existing ZIP
without the paired receipt conflicts; missing or inconsistent retained ZIP bytes
fail explicitly. Existing Artifact-plane lifecycle rules apply to these UserScope
bindings and their exact revisions; publication does not modify them on replay.
If a transaction aborts before publication, the explicit source IDs remain fixed;
retry validates the requested finding revisions again and either captures them
or conflicts. Concurrent snapshots use repeatable-read transactions and bounded
retry on definite PostgreSQL serialization aborts.

Server owns owner-authorized source selection, snapshot capture, assembly and
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

For `security-findings@1`, the declared input slot is `findings`, required and
accepting `application/vnd.contractor.findings-collection+zip`. Workflow validation
requires this slot only when an agent selects `list_findings`. Preparation resolves
`inputs/findings` once to an exact revision; later binding changes do not alter
the prepared snapshot. The existing ToolsetSelection contract needs no new fields.

```yaml
# Workflow spec
inputs:
  findings:
    required: true
    mediaTypes: [application/vnd.contractor.findings-collection+zip]

# AgentTemplate spec
toolsets:
  - ref: security-findings@1
    tools: [list_findings] # Add finding when this agent also creates proposals.
  - ref: run-artifacts@1
    tools: [read_artifact]
```

Runtime owns allocation-local reader-toolset preparation:

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
of a foreign Run is needed. Full cross-process readability is covered by the
findings process gate below.

## Reader tool contract

```text
list_findings(subject_kind?, subject_key?, limit?, cursor?)
    -> {items: [...], next_cursor: string|null}
```

The collection input is fixed by the selected toolset version, never by
model-supplied Run, Project, Audit or owner IDs. The reader binds one collection per
prepared reader. Filtering compares the original proposal subject exactly and
case-sensitively. `subject_key` requires `subject_kind`; absent values mean no
filter, while empty strings are invalid. Limit defaults to 20, range 1–100.

Each item returns `receipt_id`, `proposal_id`, `run_id`, `invocation_id`, optional
`audit_origin` and `audit_holds`, `retention`, the original `subject`, `title_preview`,
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

Opt-in configurations:

- `openapi-operation-trace@3` binds `audit-openapi-operation-trace@2` and
  `audit_openapi_operation_tracer@2`. It retains structural graph/source tools,
  adds text evidence writing and `finding`, and uses human-required finding
  confirmation. Canonical results carry successful proposal client keys;
  `operation-resolution` coverage retains its existing meaning.
- `findings-review@1` uses `findings_analyst@1`, accepting only the required
  `findings` collection input and publishing a Markdown `report`. It selects
  `list_findings`, `read_artifact` and text report writing. Its generic analysis
  requires no hypothesis, annotation, OpenAPI input or Audit origin.

Producer guidance lives in a separate instruction file; the `trace` Skill and
instruction-eval baselines are unchanged by it. Neither configuration changes
default deployment or establishes model-quality improvements.

The [shared fixture](../../internal/auditdomain/testdata/finding-collection-v1.fixture.json)
is the conformance reference for the Go and Python codecs and for Runtime
preparation. It contains generic function/configuration subjects, ordinary Run
and Audit origins, optional hypothesis, a review without decision ID, and
identical evidence bindings from different Runs. Codec tests require exact
conformance and reject inconsistent metadata, byte content and evidence
membership.

Required integration coverage: PostgreSQL publication/retention tests and an
HTTP test that passes the ZIP into ordinary Run creation, covering concurrent
replay, later review changes, all finding contributions, foreign/missing
sources, source Run deletion and rollback after the ZIP write but before its
receipt; they run with `-tags=integration` and fail when explicitly selected
without a test database. Runtime tests cover Python decoding of the same
fixture, real ArtifactClient preparation behavior, previews, pagination and
independent tool selection. The findings process gate, `make test-findings-e2e`,
proves producer → publication → ordinary input fork → preparation → reader
across Server/Runtime processes. Model quality evals remain separate and follow
the portable evaluation format in [26](26-portable-evaluation-format.md).

## Finding proposal and selected authoring interfaces

The single current `contractor.audit.finding-proposal.v1` schema carries the
proposal fields and intake/receipt authority and permits an explicitly unknown
`subject: null`, and adds optional `locations` and `http_exchange`. A missing
subject is never synthesized from a URL, title or run ID. Verification inventory
uses the existing receipt identity when a subject is unknown; this does not
rewrite the proposal's subject.

Select exactly one creation interface in `AgentTemplate.toolsets`:

| Toolset | Required model arguments | Optional location arguments |
| --- | --- | --- |
| `security-findings@1` | `title`, `description` | typed `locations` |
| `security-findings-code@1` | `title`, `description`, `file` | `line` or `range` |
| `security-findings-http@1` | `title`, `description`, `url`, `method` | `request_id` |

Each exports the single visible name `finding`. Selection is fixed before the
worker starts; duplicate visible tool names are rejected. Each also accepts
optional `cwe`, exact `evidence_refs` and typed `standard_refs`. The publisher
shares normalization, intake and receipt handling. Reading via
`security-findings@1` can be selected independently of a creation facade.
There is no parallel old authoring implementation or alternate proposal schema.

The hidden `client_key` is `call-` plus the full SHA-256 of the nonempty ADK
function-call ID. The existing invocation namespace and submission identity
remain unchanged. A transport retry of the same event reuses exact submission
bytes; different tool calls remain different proposals. Missing call identity
fails before submission. Results return `proposal_id`, `receipt_id`, `client_key`;
Audit workers copy the returned key into `submit_check_result(proposal_keys=...)`.
No content-based deduplication or guessed subject is introduced.

Optional `cwe` maps to `standard_refs` with scheme `CWE`, version `4.20` and the
explicit `CWE-NNN` weakness ID. The bundled catalog records MITRE's versioned XML
archive URL, archive/XML SHA-256 and its weakness IDs. Unknown IDs fail locally;
no live taxonomy lookup occurs. Updating the catalog/version requires reviewing
the mapping and the external benchmark binding together.

### Typed locations

A location is exactly one closed object:

- Source: required `file`, optional `line` or `range: {start_line, end_line}`.
- Web: required `url`, optional `method`.

Source coordinates are one-based and inclusive; range end cannot precede start.
File paths are case-sensitive relative POSIX paths. Reject backslashes, colons,
empty/dot/parent components and control characters. URLs must be absolute
HTTP/HTTPS URLs with a host, valid nonzero port if present and no userinfo,
backslash, whitespace, control characters or malformed percent escapes. Methods
are nonempty ASCII HTTP tokens. Preserve authored spelling, query order,
escaping and fragments; do not fetch, normalize, infer GET or remap paths.

| Bound | Value and basis |
| --- | --- |
| Locations | 256, matching existing per-finding evidence capacity |
| File | 4096 UTF-8 bytes, matching the retained path bound |
| URL | 8192 UTF-8 bytes, matching the HTTP tool URL bound |
| Method | 64 ASCII bytes, matching the bounded HTTP method field |
| Line/range endpoints | 1–9007199254740991, JSON/JavaScript exact integers |

Absent or empty locations encode by omission. Explicit null, unknown fields,
file+URL, line+range and malformed coordinates fail. Locations are producer
claims, not proof of source identity, reachability or exploitation. The benchmark
must separately verify exact source bytes. Array order is retained.

### Selected HTTP evidence

The HTTP session keeps the latest 128 outgoing request records in memory, under
its existing history capacity. Each contains actual HTTPX request method/URL,
ordered headers (including duplicate fields), complete outgoing body bytes,
response status/headers or a transport outcome, and the redirect/retry attempts.
The store is allocation-owned; resolution additionally requires the same
invocation. Failed/incomplete exchanges cannot be attached as successful evidence.
Clearing or closing the session erases the store; IDs are never reused.

The HTTP facade requires URL and method even without an ID. A missing ID permits
a finding without captured evidence. A supplied expired, incomplete or foreign ID
returns a repairable error; callers may choose a current ID or omit it. Creating
a finding never sends or replays traffic. Authored location and observed request
remain separate facts; neither silently replaces the other.

`http_exchange` embeds a copied snapshot in the selected proposal:
`request_id`, `request_tag`, nonempty `attempts`, and optional
`response_body_evidence_id`. Each attempt has `method`, `url`, `headers`
(`{name,value}` rows), canonical `body_base64`, exactly one of `status` or `error`,
and optional `response_headers`. Errors are `transport_error` or `cancelled`.
An existing final response-body artifact is pinned through normal exact evidence
references. There is no new artifact for every outgoing request. Selected target
Authorization/Cookie headers and request bodies are retained in full in the
proposal; they are not added to history summaries, metrics or serialized worker
state. Private transport credentials remain outside the target request capture.

Limits mirror existing HTTP bounds: 1 MiB outgoing body per attempt, 64 KiB per
header block and at most ten redirects plus three attempts (13 total records).
The request header block is the complete set actually sent, not only the
model-supplied headers: `http_request` limits those to 48 KiB, reserving the
rest for Runtime-added headers, and refuses before sending any request whose
complete block would exceed 64 KiB.
The existing 8 MiB proposal/submission limit still applies to the encoded
snapshot. An oversized finding fails explicitly; evidence is not silently
truncated. Original response artifact size/truncation semantics are unchanged.

### Development contract and verification

Server, Runtime, configurations, clients and benchmark binding use this current
contract together. Old development shapes are not supported through adapters or
migrations. Collections retain exact proposal bytes that satisfy this contract.
API and machine reports expose typed coordinates and selected exchange evidence.
The UI displays authored coordinates without inferred links or web previews.
SARIF export remains a separate draft.

The ordinary `source-findings-review@1` workflow uses the code facade. Source
verifier/tracer templates select the code facade; the HTTP verifier selects the
HTTP facade. No duplicate template/workflow versions exist for compatibility.
External benchmark bindings consume the ordinary workflow's retained public
receipts.

The shared location fixture exercises Go, Python and UI validation. The findings
process gate includes a real facade→intake→API→collection→reader round trip after
producer deletion. Runtime tests cover actual provider declarations, hidden call
identity, full HTTP capture, invocation isolation, eviction and immutable snapshots.
