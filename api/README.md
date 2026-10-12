# Public API contracts

`openapi/contractor-public-v1.yaml` is the single committed, already bundled
OpenAPI 3.1 contract for public HTTP endpoints. It deliberately has no external
`$ref`; the separately versioned WebSocket frame contract is linked as
`events/contractor-events-v1.schema.json` rather than embedded as an HTTP body.

The Server serves every operation of the contract, and nothing else: the route
table in `internal/httpapi/public/routes.go` registers each route with its
`operationId`, and `TestRouteTableMatchesOpenAPI` fails when an operation has no
route or a route is not an operation. The contract therefore carries no
planned operations; a design that is not served yet belongs in
`docs/proposals/`, not here.

Every operation also needs a test response checked against this contract. The
Go tests of `internal/httpapi/public` record each response that
`serveAndValidatePublicContract` validates, and a run of the whole package
fails when an operation has none. Each operation needs a validated success
response, except the WebSocket upgrade, whose `101` needs a real connection
(the event handler tests complete the handshake), and, while
`CONTRACTOR_TEST_DATABASE_URL` is unset, the Eval operations, whose success
responses come from the Eval service on PostgreSQL. A filtered run enforces the
same coverage when it sets `CONTRACTOR_REQUIRE_OPERATION_COVERAGE=1`, as
`make verify-public-api` does.

Within `/v1`, an additive change may introduce an optional request field, an
optional response field, or a new endpoint. Existing fields, statuses,
semantics, security requirements, and enum members cannot be removed, renamed,
narrowed, reinterpreted, or extended with a value that an exhaustive client
cannot handle. Changing a required field, making an optional field required,
changing a field type, or otherwise invalidating an existing conforming client
requires a new public API version and a separately versioned event subprotocol
when applicable.

Run `make verify-public-api` after either contract changes. The pinned Go module
validator checks OpenAPI structure, examples, formats and patterns, verifies
that the YAML is self-contained, validates the standalone Draft 2020-12 event
schema and examples, enforces the public/private and secret-free boundaries,
and validates a response of every operation as described above, without a
database. The gate also checks actual repository configuration projections,
RuntimeConfig author/read distinctions, generated-client field preservation,
finding decision variants, multibyte passwords, label selectors and retained
Run history. These fixtures establish their tested cases, not exhaustive
equivalence between every schema and domain validator.

Every operation whose handler reaches PostgreSQL documents the retryable `503`
(`storage_transaction_conflict`) returned when definite serialization or
deadlock aborts exhaust their retries. The contract test lists the few
process-local operations that cannot return it, so a new operation must either
document the response or be added there with its reason.

Every error code of the HTTP error envelope and of a WebSocket error frame is
an entry of the Server's error registry (`internal/apierror`), with its status
and `retryable` flag; a domain error reports its entry, and the public handler
answers only with registered entries. A new code is added there, then
`make docs-error-registry` regenerates its table in
`docs/spec/platform/errors.md`. `make verify-public-api` checks the registry
against that table, the Eval error code enum, the error details kinds and the
error statuses of this contract, the error frame codes of the event schema, and
the error codes the browser client compares.

`make verify-public-api-postgres` requires an explicit disposable
`CONTRACTOR_TEST_DATABASE_URL` and checks Audit pagination at its maximum page
size, including receipt hydration, next cursors and owner/revision fences. It
runs with the race detector and is included in `make release-verify`; skipping
the database test is not acceptance of that boundary.

RuntimeConfig publication uses `RuntimeConfigAuthorDocument` with exact Gateway
selector strings. Read responses use `RuntimeConfigDocument` with resolved refs.
The Go generator preserves absent/null/value using `nullable.Nullable[T]` for
supported nullable patches; callers must deliberately choose omission, `SetNull`
or `Set(value)`. The typed request tests verify emitted HTTP JSON.

No derived bundle is committed, so there is no second generated
contract that can drift from the YAML source of truth.

The [managed Eval data catalog](evals/v1/README.md) is a closed, embedded codec
contract introduced by V38-002. Its public `/v1/eval-*` and
`/v1/projects/{projectId}/eval-*` operations are defined in
[`contractor-public-v1.yaml`](openapi/contractor-public-v1.yaml) and served by
the Server.

The [scan RequestSet artifact contract](scan/v1/README.md) defines neutral
prepared HTTP inputs, content identities, exact source provenance and coverage
gaps. It is an Artifact format consumed by preparation/planning code, not a new
public HTTP endpoint.
