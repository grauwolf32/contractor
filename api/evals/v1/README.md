# Managed evaluation data contract

V38-002 introduces data codecs and fixtures; it does **not** expose HTTP routes.
[Spec 29](../../../docs/spec/29-managed-evals.md) owns the managed behavior and
[spec 26](../../../docs/spec/26-portable-evaluation-format.md) owns the portable
format. Public OpenAPI and generated clients land with the V38-005/006 handlers.

`managed.schema.json` is the closed Draft 2020-12 catalog. A DTO is selected by
`urn:contractor:eval:v1#/$defs/<name>`. The embedded local catalog is validated by
`internal/evaldomain`; reference loading is disabled outside embedded resources.
The code checks semantic invariants after schema validation. Ownership, resource
resolution, evaluator registration and observed execution provenance still require
the service layer; passing a schema never grants authority.

The `portable` directory is a byte-for-byte schema snapshot from the implementation
of spec 26, pinned by commit and exact file hashes in `portable/provenance.json`.
The historical `playground.*` schema IDs name a format, not a required producer.
The Server embeds these files and does not import, locate or execute Playground.
Refresh this snapshot only with an explicit compatible format change and shared
conformance evidence. Existing frozen documents keep their original bytes/digests.

## Requests and projections

- Authoring: `DatasetInput`, `Draft`, `CreateExperiment`, `DraftUpdate`,
  `ExternalRegistration`. Private check data is present only in authoring/storage
  and the explicit owner `Review` context. `Dataset`, `CasePage`, the execution
  projection and `PublicPlan` have no private oracle partition.
- Control: `Command`, `Submission`, `Delete` (empty JSON representation of a
  bodyless deletion), and their durable receipts. Every mutation carries an
  idempotency key; revisioned mutations also require a quoted positive `If-Match`.
  Replay is looked up within authenticated owner/resource scope before stale-CAS
  rejection. Body/operation/revision changes under a key conflict.
- Evidence: `ResultInput`, `AssessmentInput`, `CheckRequest`, `SelectionInput`.
  Result claims are checked against authoritative execution and usage at ingestion.
  Native check requests carry check IDs rather than caller-supplied verdicts.
  Reviewer identity always comes from authentication, not the body.
- Reads: `Capabilities`, `Experiment`, collection pages, `MemberPage`,
  `ExecutionPage`, `Pair`, `PairPage`, `Review`, `Report` and `Chart`. Raw private
  portable records are never serialized as these read DTOs.
- `APIError` is the safe domain error payload; the HTTP layer places it in the
  existing public error envelope and supplies request correlation metadata.

`qualityPassed` counts passing assessments, including allowed assessment of failed
executions. `endToEndPassed` additionally requires eligible/succeeded/complete
execution. Their denominators cannot be substituted. Charts label matching pair
coverage separately from whole-arm totals. Audit normalization counts each leaf
Run/stage execution once, includes all owned roles/retries and uses parent wall
time; it never adds parent aggregates or overlapping child durations.

Schemas bound documents to spec 26's matrix limits; the decoder additionally caps
bytes at 1 MiB and nesting at 32. Both must be satisfied. Imported source hashes
remain producer assertions; visible imported bytes and frozen local bytes get
separate computed hashes. Pure codecs cannot verify hidden external plan contents.

The [fixture index](../../testdata/evals/cases.json) includes all four
Workflow/Audit and server/external combinations, portable identities, invalid
records and explicit privacy markers. `http-mutations.json` gives complete
method/path/header/body examples. `audit-accounting.json` fixes multi-role retry,
deduplication and overlapping-duration expectations. No example authorizes a
live model or target run.

Run from the repository root:

```sh
go test -race -count=1 ./internal/evaldomain
```
