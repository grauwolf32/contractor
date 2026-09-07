# V39-003: invocation-local Audit result collection

`InvocationAuditCollector` implements the V39-001 collector interface using the
V39-004 canonical package encoder for each prospective mutation. It owns immutable
input bytes and normalized result values outside ADK State. One async lock covers
scalar admission, ordered atomic batches, sealing and discard. Read results are
copies of the same pinned tasks used for validation; neither tool has an artifact
client or rereads current input aliases.

Creates start at revision 1. Canonically equivalent retries preserve revisions;
corrections require the current revision. Only the immediately preceding explicit
update can replay its previous revision/content pair. The collector retains the
current value and one replay revision per item. Mixed batch/scalar use creates
missing items and replays identical ones atomically; changed existing members
return an error including the current revision for scalar correction.

Local validation covers trusted membership, coverage subsets, bounded UTF-8
summaries and arrays, proposal identities, conclusive checklist evidence, pinned
standard assessment/kind/count requirements, and not-tested operation coverage.
Coverage/proposal sets are sorted during normalization. Evidence order remains
part of canonical content. Partial or gapped coverage may be valid. Profile,
proposal, retention and final Audit acceptance remain importer responsibilities.

Receipts say `recorded`, list accepted revisions and missing keys, and contain no
result content or ArtifactRef. Errors identify fields and trusted item keys without
echoing invalid payloads. Tool metrics use those safe projections and count errors
as failures. Prospective size checks use actual canonical members and ZIP overhead,
including replacement of previous values; rejected submissions preserve revisions.

Both tool instances bind to one collector and reject a different invocation ID.
Sealing rejects all later writes, including identical retries. Discard clears
accepted values and replay state and permanently closes the collector. V39-005's
common completion owner must create one collector per allocation/invocation/task
digest and call discard in cancellation/failure cleanup. There is no persisted
partial state to carry into a new invocation or recover after process loss.

Shared data lives in `api/testdata/audit-completion`, independent of `configs`.
Nineteen positive/negative cases run through both Python collection and Go importer
validation. Additional tests cover concurrent replay/conflicts, atomic mixed
batches, immutable copies, stale invocation calls, cancellation cleanup, real ADK
tool schemas, exact encoder bounds and the actual 8 MiB aggregate limit. Legacy
@1 byte-compatibility and publication tests remain part of the regression command.

Validation passed: 140 Python tests, the Go shared-fixture test (19 cases), and
Ruff for all changed Python files. Commands:

```sh
cd runtime
uv run --frozen pytest tests/test_audit_result_collector.py tests/test_audit_results_toolset.py tests/test_audit_result_publication.py tests/test_audit_completion_contracts.py
uv run --frozen ruff check src/contractor_runtime/audit_result_collector.py src/contractor_runtime/toolsets/audit_results_v2.py tests/test_audit_result_collector.py tests/test_audit_results_toolset.py
```

```sh
go test ./internal/auditimport -run TestAuditCompletionSharedTaskLocalValidation -count=1
```

Registration, A2A finalization, publication and capability activation remain
V39-005 work. This change leaves the legacy @1 implementation untouched.
