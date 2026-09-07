# V39-004: deterministic Audit result encoding and publication

Status: implemented and verified offline on 2026-09-07; production activation
remains V39-005 after Server propagation and collection. No Runtime capability,
model call, tool factory or deployed configuration is enabled by this change.

## Encoding and compatibility

`toolsets/audit_results/packages.py` contains the existing pure Audit package codecs,
extracted from the @1 tool, now in `toolsets/audit_results/v1.py`. The @1 tool
retains its arguments, reads,
publication and telemetry behavior. `CanonicalAuditPackageEncoder` uses those
same package conventions over immutable V39-001 inputs and snapshots. It verifies
input membership/order, requested coverage and encodable normalized values.
Task-local evidence acceptance is V39-003; independent Audit import is unchanged.

Results use trusted task order, canonical JSON, deterministic evidence IDs,
fixed ZIP time/Unix permissions, path order and ZIP_STORED. Revisions, arrival
order and invocation IDs without proposal selections do not affect bytes.
Proposal invocation identities remain semantic inputs. The two-item proposal
fixture was encoded using the original @1 implementation at
`f17476a792a6bc29eab44a9fcaf5d615a71f1ea2` in a separate checkout: 2428 bytes,
SHA-256 `50b0e966a0f6c01477464a0c16ca4668385ea5697e5df8759f70fa45f208c5ea`.
The new encoder matches that pinned value exactly.

The encoder accepts empty/partial prospective snapshots for collector admission;
those packages cannot be published. Publication accepts only a complete sealed
snapshot with matching allocation/invocation ownership and trusted input bytes.
The collector must call this same encoder before committing creates/replacements.

Measured collected bytes are the sum of actual result/evidence content members,
including rendered evidence text. The 8 MiB budget therefore includes evidence
that occurs in both its JSON record and text member. Per-member limits include
the manifest; the 16 MiB package limit includes ZIP overhead. Tests exercise
exact size boundaries and an actual 256-evidence case that exceeds the collected
budget despite individually bounded summaries.

A planning contradiction was resolved against the existing Go importer:
`MaximumProposalsPerItem` is **128**, while coverage lists allow **512** values.
The normalized item contract and V39-003 task now preserve both distinct limits;
no aggregate 512-value limit is introduced and the importer is not relaxed.

## Publication and recovery

`DeterministicAuditResultPublisher` receives a Server-validated versionless output
binding, immutable owner, allocation-scoped ArtifactClient and the invocation's
fatal-state check. It copies the binding and rejects input namespaces, exact write
targets and foreign allocation clients. Preparation/Artifact API remain responsible
for grants and StageContentRequest ownership; the publisher grants no new access.

Only create-only CAS is used. An authoritative successful write response binds
sent bytes/media/size to an exact revision. On a create conflict or ambiguous
transport outcome, bounded read-back compares complete bytes and media type.
Matching content yields the returned exact revision; different content fails with
`audit_result_publication_conflict`, without overwrite. A missing/unavailable
reply can allow one more identical create-only attempt. Every path is bounded by
**two writes and two reads total**, the invocation deadline and a 16 MiB read cap.

Authentication, authorization, write fencing and invalid requests stop immediately,
even when the server supplies a retryable flag. Unresolved transport outcomes yield
retryable `audit_result_publication_failed`; input/size/conflicting-content failures
are non-retryable. Error text contains only fixed codes. Production ArtifactClient
performs the ordinary ETag, response identity, timestamp, media and size validation;
the publisher also enforces its bounded exact receipt contract.

A later Stage attempt must independently collect and seal its full assignment.
An existing result never becomes a collector checkpoint. Equal bytes may be
recognized; changed content or proposal invocation IDs conflict. A new child Run
uses its own allocation-scoped output storage. Cancellation/fatal-state checks run
before and after publication I/O and before returning a receipt, with a final
cooperative yield to deliver pending cancellation even for an inline transport.
A written artifact can remain diagnostic without implying Worker success or Audit
acceptance. No model repair, fake tool metric or terminal WorkerResult is emitted.

## Validation and remaining integration

An isolated checkout passed **120 tests, zero skipped**:

```sh
cd runtime
uv run --frozen pytest tests/test_audit_result_publication.py \
  tests/test_audit_results_toolset.py tests/test_audit_completion_contracts.py \
  tests/test_artifacts.py tests/test_capabilities.py
```

The task's required publication/@1 subset contains 59 tests. All changed Python
files pass Ruff. Publication tests use the production ArtifactClient and a scripted
ArtifactTransport, covering dropped replies, repeated ambiguity, conflicting
content/media, prior/new Run attempts, read limits, malformed receipts, deadlines,
fencing and cancellation. No model service or external artifact store is contacted.

V39-002 owns durable authority propagation and placement; V39-003 owns validated
collection and the model-facing tools; V39-005 wires completion and advertisement.
V39-007 must still validate Runtime-produced ZIPs through the real Go importer and
isolated PostgreSQL release gate. These unit/contract results do not replace it.
