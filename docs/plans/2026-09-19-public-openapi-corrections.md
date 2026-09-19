# Public OpenAPI corrections — V59

Implemented and verified on 2026-09-20. See the
[finding-by-finding results](../research/2026-09-20-public-openapi-corrections-results.md)
and [durable verification record](../../tasks/evidence/v59-005.json).
Task files record implementation hashes and integration completion separately.

The user authorized detailed analysis, task creation and implementation of the
[2026-09-19 review](../research/2026-09-19-public-openapi-review.md). This plan
records the selected corrections to the current implementation. Existing domain
specifications remain authoritative; a permissive or restrictive OpenAPI shape
does not justify changing valid stored data or established mutation semantics.

| Task | Findings | Implementation boundary |
| --- | --- | --- |
| V59-001 | R01, R05 | Audit public report projection and database-backed pagination |
| V59-002 | R02, R03 | Published configuration projections and optional telemetry retry |
| V59-003 | R04, R09 | RuntimeConfig author/read schemas and Go nullable request semantics |
| V59-004 | R06, R07, R08, R10, R11, R12 | Request conditions, bounds, labels, auth and Git documentation |
| V59-005 | All | Cross-client verification, finding resolution and integration |

**Audit behavior.** For a proposed report the service already owns the current
review request. Preserve that request in the public DTO and require it when the
response status is proposed. Do not fabricate a review in the UI. Pagination
must keep its public bound of 200 while allowing an internal extra record for
cursor detection. Trace that record through receipt hydration: increasing the
first validator alone does not resolve the complete pipeline. Verify real
database pages at 199/200 with a 201st record and stable cursor traversal.

**Published configuration.** The two missing fields are existing behavior:
Summarizer instructions are safe exact references, and telemetry retry is
optional configuration with normalization and defaults. Add their precise
schemas without opening objects to arbitrary properties. Validate the actual
repository configuration catalog and both submitted and normalized runtime
documents. Historical documents omitting the new fields remain valid.

**RuntimeConfig authoring.** Publication takes exact selector strings and
resolves them into immutable gateway references. Reading returns those refs.
Use separate author/read document types with shared components only where the
accepted values are the same. Gateway and llmGateway cannot be cleared with
null; telemetry, proxy, Caido and the supported credential field can. Preserve
those distinctions in both validators and generated clients.

**Nullable Go requests.** The pinned generator supports `nullable-type: true`
for the current OpenAPI 3.1 null unions. Enabling it changes nullable fields
outside RuntimeConfig as well. Inspect and update affected callsites, preserve
their actual request bytes and test absent/null/value through the generated
HTTP request builder. Raw-body request support is not a substitute for a
correct typed client. The TypeScript author/read distinction must compile
through current forms and normalization helpers.

**Request conditions and bounds.** Encode finding verdict requirements with
positive and negative cases. Empty metadata-label values must be selectable in
both owner and project Run lists. Password validation remains 12–1024 UTF-8
bytes; remove incompatible character-count restrictions from the public schema
and any affected login client without changing stored credentials. Declare
Origin/CSRF requirements for Git cookie mutations using the existing reusable
parameters. Document missing Git preconditions as create-only, preserving CAS
for updates. Remove the unsupported 1024-item Run history schema cap; global
history retention and pagination policy are separate work, so no truncation or
new runtime quota is introduced here.

**Verification.** Each task records its focused checks and original
implementation commit. The final gate regenerates both clients, tests Go
consumers and the complete UI, and includes the new contract cases in the
normal public API verification command. Structural validation alone is not the
acceptance criterion. PostgreSQL checks use an isolated local test database;
live services, models and evaluations are outside this work.

V59-001 through V59-004 can be developed independently in isolated worktrees.
They all touch the bundled OpenAPI or generated files, so integration resolves
source schemas first and regenerates outputs with the pinned generators.
V59-005 starts after the four implementation tasks are complete. Existing
uncommitted planning documents in main are preserved during integration.
