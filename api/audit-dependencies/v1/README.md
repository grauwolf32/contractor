# Exact Audit item dependencies

`manifest.schema.json` describes `contractor.audit.retained-dependencies.v1`.
The Server constructs and verifies the manifest against its accepted baseline,
preparation outputs, item results and held finding receipts. An origin in JSON
alone is never authority. The header pins the consumer's baseline, profile,
Workflow and selected Skill closure digests.

At most 32 named descriptors and 256 KiB of canonical metadata describe the
complete closure. Names and each `requires` array are sorted and unique; the
explicit graph is closed and acyclic. `projection: exact` requires identical
source/retained digests, media types and sizes and an Audit-owned retained
namespace. Other projections and origin variants require a production adapter.
A bare JSON reference is provenance, not a loading instruction or access grant.

The same manifest is pinned in the item task and execution item. Runtime checks
that agreement before accepting check results. Server admission and dispatch
also verify ownership, accepted source membership, consumer authorization,
media compatibility and every exact retained revision. Required sources are
materialized as ordinary Run inputs before scheduling.

The Go/Python cases in `api/testdata/audit-dependencies/` pin grammar, closed
origins, exact metadata, closure failures and limits. The schema describes
structural validation; codecs also enforce graph and cross-field invariants.
