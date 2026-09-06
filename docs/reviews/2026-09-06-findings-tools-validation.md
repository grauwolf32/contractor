# Findings producer, collection and reader validation

Validated on 2026-09-07. V43 separates finding creation, collection publication,
collection reading and human review. These checks establish contract behavior;
they do not measure model quality or activate a deployment.

## Executed contracts

`TestFindingsProducerAndReaderAcrossProcesses` uses the authored
`openapi-operation-trace@3` → `audit-openapi-operation-trace@2` producer and
`findings-review@1` consumer. It launches the Go Server, Python Runtime, disposable
PostgreSQL schema, mTLS control plane and a local scripted model gateway. The
producer writes evidence, receives an intake receipt, and includes its exact
invocation/client key in the canonical result accepted by the real importer.
Public collection publication and ordinary Run input forking deliver full proposal
and evidence bytes to the consumer. Audit review state is unchanged by analysis.

| Required case | Evidence and observed behavior |
| --- | --- |
| Ordinary Run and optional hypothesis | `TestFindingsCollectionsRetainOrdinaryAndAuditReceipts/ordinary-hypothesis-and-intake-replay`: the ordinary producer has no Audit origin; repeating identical `finding` arguments within the invocation returns the same receipt. |
| Shared function across operations | `shared-function-preserves-operations`: GET and POST reach the same function and produce distinct receipts with equal evidence digests. No receipt is removed by semantic deduplication. |
| Source deletion and retention | `snapshot-survives-source-deletion`: the published ZIP replays unchanged after ordinary/Audit child Runs are deleted; retained Audit bytes also support a new collection. |
| Idempotency conflict | `changed-selection-conflicts`: changing the selected receipts under the same collection key returns HTTP 409. |
| Pagination and full reads | `pagination-and-exact-evidence`: three one-item pages reach the analyst; every proposal and evidence document is read by its current-Run exact ref and all receipts remain in the report. |
| Empty collection | `TestFindingsReaderBoundariesAcrossProcesses/empty-is-explicit-success`: empty arrays produce an empty report inventory without fabrication. |
| Interrupted preparation state | `interrupted-preparation-reuses-exact-ref`: before a Runtime registers, the production Run Artifact store receives one verified document representing a prior partial preparation. Runtime reuses that actual receipt, materializes the remaining documents, then exposes the reader. |
| Conflicting document | `conflicting-document-fails`: seeded conflicting bytes remain unchanged; preparation fails and publishes neither report nor proposal. |
| Invalid ZIP | `invalid-zip-fails`: malformed input fails before any model invocation. |
| Inaccessible input | `inaccessible-input`: an absent exact input is rejected at the ordinary Run-create boundary. |

All process cases use actual Server/Runtime implementations. Seeding a partial
Artifact store is deterministic fault injection, not a process-kill timing test.
Lost write responses, interrupted materialization, source-scope collisions,
foreign/changed cursors, invalid manifests, byte bounds and exact-read ETag
mismatches also run through the real Python ArtifactClient with a scripted
transport in `runtime/tests/test_findings_reader_toolset.py`.

## Gate and prerequisites

Run with a prepared `runtime/.venv`, Go, Python 3 and disposable PostgreSQL:

```sh
CONTRACTOR_TEST_DATABASE_URL='postgres://…/disposable_test_db' make test-findings-e2e
```

The gate runs the writer/reader Python contracts, then the three production-process
Go tests above with `-count=1`. It checks pytest JUnit membership, including minimum
parametrized case counts, and Go JSON events for all 13 mandatory test/subtest
names. Passing parent tests cannot hide skipped or missing cases. Missing database
configuration, missing Runtime prerequisites, no matching tests, failed cases and
skipped cases all fail the gate. Offline `TestFindingsGate*` tests verify those
failure paths. Go tests create isolated schemas and clean them up.

Additional checks:

```sh
go test ./internal/findingintake ./internal/auditservice ./internal/config ./tests/e2e
cd runtime && uv run --frozen pytest tests/test_security_findings_toolset.py tests/test_findings_reader_toolset.py
```

The workflow catalogue and frozen trace Skill were checked during V43-004 through
`go test ./internal/config ./internal/agentskills ./tests/eval/agent_instructions ./tests/e2e`
and `go run ./cmd/contractor-server config validate --root ./configs`.

## Model evaluation cases to freeze after V41-008

These are scenario descriptions for a future portable plan. They are not an
executed evaluation suite, a benchmark score or an implicit approval for live
model calls. Each case needs explicit fixture, configuration, policy/Skill pins,
repeats, budget, assessment criteria and completeness requirements in the V41
format. Keep execution correctness separate from quality assessments.

| Case | Fixture and expected reasoning | Assessment evidence |
| --- | --- | --- |
| External middleware control | Route body omits authorization, but an applied middleware validates ownership. Inspect middleware and call chain before proposing a missing-control finding. | Source citations for the effective control; no unsupported exploit claim. |
| Safe sink | Input reaches a parameterized query or a context-correct escaping API. Distinguish the dangerous capability from an unsafe use. | Argument-level evidence; no finding based on the API name alone. |
| Source-inferred reproduction | Static route/schema and a supported issue are available; the target cannot be executed. Describe request shape and required identity/data, clearly marking expected observations as unexecuted. | Reproduction plausibility, preconditions and absence of invented responses. |
| Multiple independent findings | One operation contains two independently supported issues. Record both with distinct client keys, evidence and canonical proposal selections. | Complete receipt inventory; no silent omission or conflation. |
| Shared function, different controls | Two operations reach one function through different authorization paths. Preserve each operation's context; recommend grouping only when conditions and impact justify it. | Both original receipts remain cited; shared code alone is insufficient. |
| Generic observation without hypothesis | Collection contains configuration or function observations and report/log evidence, with no annotation or proposed check. Analyze the available evidence without manufacturing missing scenario fields. | Faithful scope, provenance and limitation reporting. |
| Truncated preview and contrary evidence | Preview appears severe but full proposal/evidence states a limiting condition. Open exact full documents before judging the claim. | Conclusions reflect complete retained text; conflicting evidence is surfaced. |
| Grounded analyst report | Mix supported, uncertain and duplicate-candidate findings with captured reviews. Compare evidence and suggest next inspection steps while distinguishing historical review state from current knowledge. | Exact citations, calibrated uncertainty and advisory grouping; no review mutation. |

The existing `operation-resolution` key still certifies only the operation/source
mapping. Findings, optional annotations and analyst reports do not expand that
coverage contract or imply vulnerability confirmation.
