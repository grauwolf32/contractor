# Public OpenAPI corrections — V59 results

Verification date: 2026-09-20. Based on the [R01–R12 review](2026-09-19-public-openapi-review.md)
at `b779dedf285f5d6645f064d423e8873b5c382ab6` and the
[V59 plan](../plans/2026-09-19-public-openapi-corrections.md).

All 12 findings were confirmed with clarifications and corrected in the existing
project. These fixes required neither an API rewrite nor a version change.
Verification covered schemas, actual HTTP handlers, domain constraints,
generated Go/TypeScript clients and the UI. Machine-readable results are in
[V59-005 evidence](../../tasks/evidence/v59-005.json).

## Corrections

| Finding / task | Cause and final fix | Verification |
| --- | --- | --- |
| R01 / [V59-001](../../tasks/v59-001-audit-report-and-pagination.yml) | The public DTO dropped `review`, which the service had already returned. The DTO and handler now preserve it; the schema requires review for `status=proposed`. | Real HTTP handler: four statuses, preserved ID/revision/state, authorization; negative schema case for proposed without review. |
| R02 / [V59-002](../../tasks/v59-002-published-configuration-contracts.yml) | The closed Summarizer schema omitted the existing `instructions` field. An optional exact ref was added; the UI preserves only safe ref/digest values. | Real catalog: 41 AgentTemplates, four ModelPolicies and one Gateway via list/detail; 30 AgentTemplate versions have Summarizer instructions. Go decoding preserves the ref; the UI does not copy instruction text. |
| R03 / V59-002 | The closed telemetry schema omitted the existing `export.retry`; the UI also dropped it. Optional fields spanning 1–60000 ms and existing defaults 100/1000 are documented. The UI preserves valid values and applies the existing Server ordering check after defaults. | Publication/list/detail: absent retry, empty object, custom values, minimum and maximum; scalar/null/unknown and invalid ordering are rejected. Defaults are described in prose so the generator does not make optional request fields mandatory. |
| R04 / [V59-003](../../tasks/v59-003-runtime-config-author-and-nullable-client.yml) | One union mixed the input selector and stored exact ref. Author/read documents were separated. The UI form was also found to send a gateway object where the Server accepts a selector; the form was fixed. | 11 shapes compared against the real author parser and both schemas. Real POST resolves a selector; GET returns an exact ref and preserves permitted clear fields. UI typecheck and form test. |
| R05 / V59-001 | A public page of 200 needs 201 internal rows; two limits interfered: service validation and receipt batch loading. The internal limit now permits the extra row, with receipts loaded in batches of 200. | Isolated PostgreSQL and race detector: 201 findings/reviews/provenance records, pages of 199/200 with 2/1 remaining, complete traversal without omissions or duplicates; owner isolation, stale cursor, public 201 remains rejected. |
| R06 / [V59-004](../../tasks/v59-004-request-and-history-contracts.yml) | The schema did not express verdict-dependent fields. A closed `oneOf` now requires severity for true_positive and a target for duplicate; other verdicts forbid unrelated fields. The UI builds the matching branch. | 18 shared wire cases for the schema and real domain validator, five valid generated Go builders, UI tests. |
| R07 / V59-004 | The regex rejected a supported empty label. Both Run-list schemas now accept empty values, additional `=` characters and newlines. | Eight HTTP scenarios across two list endpoints select exactly the intended Run. Independent review corrected the initial `.*` pattern, which incorrectly rejected newlines. |
| R08 / V59-004 | JSON Schema counts characters; the Server counts UTF-8 bytes. Incompatible character bounds were removed and units documented explicitly; the UI no longer requires 12 characters. | Real bootstrap/hash/login for a 12-byte Cyrillic password and a 1024-byte supplementary-Unicode password; 10/1028 bytes rejected. UI input and submission tests. |
| R09 / V59-003 | `*T` with `omitempty` did not distinguish omission from explicit null. Enabled `nullable-type`, supported by the pinned generator, and adapted CLI calls. | Real typed HTTP builders: 15 absent/null/value cases for five RuntimeConfig fields and six for Run credential selectors; serialized bytes and the production parser are checked. |
| R10 / V59-004 | Four Git mutations did not declare existing Origin/CSRF parameters for cookie auth. Reusable parameters were added; regenerated builders send them. | Four operations × five authentication modes; rejected requests never reach the domain mutation. Generated-client header serialization is checked separately. |
| R11 / V59-004 | `maxItems: 1024` was not an actual retained Run-history limit. The false schema cap was removed; documentation describes the full retained history. | Real config accepts retry=1025; the HTTP handler preserves 1025 fixture attempts/transitions, including the final entries. This does not execute 1025 real Scheduler executions. |
| R12 / V59-004 | The description incorrectly required an explicit create precondition. Documentation now reflects existing create-only behavior without headers and CAS for updates. | Real import precondition logic with a fixture repository: absent/existing binding, nil/stale/exact revision. |

## Why YAML edits alone were insufficient

R01 and R05 were primarily Server projection and pagination-pipeline defects.
R02/R03/R04/R08 affected client normalization or forms; regenerating types alone
was insufficient. R09 required checking the actual JSON sent by the generated
Go client and correcting affected CLI call sites.

The proposed → review condition is expressed by rejecting the invalid combination
with `not`: the kin-openapi version in use did not enforce the required `if/then`
check even with JSON Schema 2020 mode enabled. A negative test exposed this
before task completion. The chosen form also preserves convenient Go/TypeScript
types without introducing a large union for the entire report.

Checks were not removed just to make tests pass. Closed objects remain closed;
owner/CAS boundaries and the password byte policy are preserved. Only constraints
that contradicted the existing contract were removed: mixed author/read shapes,
character-based password bounds, the empty-label prohibition and the unenforced
history cap. No null-clearing support was added for gateway or llmGateway.
Shared WorkerCompletion limits, the ADK finalizer and model budgets were not revised.

## Verification and its limits

- OpenAPI 3.1 / YAML 1.2: unique keys, 88 paths, 111 operations,
  346 component schemas; validation against the Draft 2020-12 metaschema.
- `make verify-public-api`: passes; the permanent gate includes new contract
  cases and a previously omitted Project Run handler test.
- `make verify-public-api-postgres`: passes with the race detector and a separate
  local PostgreSQL 17. The command must fail without `CONTRACTOR_TEST_DATABASE_URL`;
  this was also verified. The gate is included in `release-verify`.
- Go public HTTP, auditservice, runtimeconfig, publicclient, CLI, config and auth
  were checked with a real test PostgreSQL instance. Application service calls
  in some HTTP contract cases use fixtures; those cases are not claimed as DB tests.
- `go test -count=1 ./...` and `go vet ./...` pass. The first full Go run exposed
  a missing `runtime/.venv` in the isolated worktree; after connecting the existing
  locked environment, the repeated full run passed.
- Full UI typecheck, lint, Vitest, build and static-server tests pass. After
  integrating concurrent V58-013 work, the entire UI suite was rerun; exact
  counts and results are recorded in the evidence.
- Go generator 2.8.0 and openapi-typescript 7.13.0 were rerun; both generated
  artifacts are byte-for-byte reproducible, with SHA-256 values in the evidence.

The run used Go 1.25.6, Node 26.7.0 and pnpm 11.24.0. The UI declares Node
24.20.x: these results apply to the actual Node 26.7.0 environment, despite all
checks passing. Browser E2E, live-model evals, deployment and production workloads
were not checked by this run. It does not prove the absence of other project
defects or an improvement in model-result quality.

## Integration and further review

V59-001–004 have separate implementation commits and subsequent completion
metadata with full original hashes. V59-005 adds permanent gates, this analysis
and a combined verification record. Concurrent V58-013 work is included in the
combined state; existing uncommitted plans in the main worktree are preserved
separately from V59 fixes.

The next review should examine user journeys and trust boundaries, including
recovery and concurrent operations. A passing OpenAPI gate verifies known
contract scenarios; it does not certify the correctness of every response,
state transition or future configuration.
