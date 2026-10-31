# Common Audit documents

`contract.schema.json` is the strict single program/task/hypothesis/context/result
contract. Its shared primitive definitions come from the existing v1alpha1 common
schema, embedded alongside it by the Go validator.

`contractor-server config validate --root <catalog> --audit-document <JSON>`
validates real Workflow bindings. For conclusions or accepted results add
`--audit-context <trusted-context.json>`; local files establish conformance only,
not live authority. This operation creates no runnable profile or Run.

Go rejects malformed/duplicate/case-aliased/unknown JSON, enforces UTF-8/raw/
canonical/node/depth bounds and derives or verifies exact per-item provenance.
JSON Schema alone cannot establish provenance authority, distinguish duplicate
JSON keys or count UTF-8/encoded bytes; these additional constraints are tested
in api/testdata/audit-common on both sides. No Runtime completion is registered.

The owning specification is docs/spec/audits/common-contract.md. The removal
map is docs/proposals/0050-audit-removal-map.yml. Existing executable profiles
are migrated by their implementation owners; this format has no legacy reader.
