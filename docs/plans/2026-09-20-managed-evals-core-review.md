# Managed Evals core review follow-up

Implementation: `e7183230bda98cea2b84c3f457394d00472b80c7` on
`feat/v38-managed-evals`. This follow-up corrects V38-002–005 independently of
unfinished V38-006; it does not mark result collection/comparison complete.

## Fixed behavior

- Deadline and token checks remain active in settling and interrupted recovery
  until accepted work drains. Four PostgreSQL regressions cover Workflow/Audit ×
  deadline/tokens, each with eight active accepted executions. Every execution
  receives an ordinary cancellation request before the fixture completes drain.
- Semantic validation selects known DTOs and child fields, with typed Summary,
  Counts, Quality, Execution and MemberView contracts. Valid dataset output names
  `counts` and `conclusion` no longer select Summary validation or cause a panic.
- Unknown preparation failures remain pending and return their original cause to
  the coordinator. Public diagnostics expose `eval_preparation_unavailable` and
  `wait`; successful retry clears the diagnostic. Confirmed configuration errors
  fail the command and return to draft. Tests cover both classification branches
  and an actual PostgreSQL error/recovery through the real preflight resolver.

## Responsibilities and compatibility

`evaldomain.Lifecycle` centralizes typed state/command rules, UI actions,
transition validation, admission, budget stops and recovery acknowledgement.
Workflow and Audit adapters share durable operation recovery. Artifact copying and
Audit workspace creation belong to member resources; metric SQL belongs to store.

Plan building has typed case, suite, binding, experiment and plan documents plus
separate check/case/variant/member builders. Pin values are optional strings.
Dataset/experiment/command/submission receipts have distinct types. Old immutable
receipts remain readable and retain their original bytes. Existing frozen plans
are not rewritten. Spec 26, execution ownership and global limits are unchanged.

## Verification

All checks passed on 2026-09-20:

- Real disposable PostgreSQL 17, isolated random schemas:
  `go test -race -count=1 ./internal/evaldomain ./internal/evalstore ./internal/evalservice ./internal/evalcoordinator ./internal/httpapi/public ./internal/projectlifecycle`.
- The additional real database preparation recovery regression passed with `-race`.
- Focused lifecycle, DTO and legacy receipt regressions passed with `-race`.
- `go vet ./internal/evaldomain ./internal/evalstore ./internal/evalservice ./internal/evalcoordinator ./internal/httpapi/public`.
- `go test ./... -run '^$'` (all packages compile).
- `make verify-public-api ui-typecheck` using Node 24.20.0.
- `git diff --check`.

No live models, targets or shared demo data were used. The unfinished V38-006 work
was saved separately before this follow-up and is not part of its implementation
commit.

## Credential recovery and integration follow-up

The second review found a remaining failure path: `PinRunSnapshot` joins
`ErrInvalid` to credential lookup errors, including database failures. Commit
`0169c0f7` records dependency failures before that wrapper for both LLM and Runtime
credentials. Unknown failures keep preparation pending; missing credentials and
incompatible identities return to draft. Runtime credential reads preserve the
database cause without rendering SQL or credential content in the public message.

Regression coverage exercises a real unavailable `runtime_credentials` table,
restores it and checks automatic preparation recovery. Additional Audit fixtures
exercise Runtime-selected LLM credentials with transient lookup errors, missing
credentials and mismatched identities.

The follow-up also separates preflight binding selection, dependency pinning,
case eligibility, artifact checks and comparison pins. Member construction is
split by case and computes each shuffle hash once. Result scope validation uses
typed execution references. Query, polling and cursor limits use named constants;
state transitions and HTTP handlers reuse their existing constants. SQL and Go
literals are formatted for review. The OpenAPI generator now emits indented
blocks; a parsed-document comparison confirms the contract is unchanged.

Integration verification on 2026-09-20:

- Disposable PostgreSQL 17 with isolated test schemas: `go test -race -count=1
  -p 4 ./internal/evaldomain ./internal/evalstore ./internal/evalservice
  ./internal/evalcoordinator ./internal/httpapi/public ./internal/projectlifecycle
  ./internal/credentials ./internal/runtimeconfig ./internal/auditservice
  ./internal/runservice ./internal/app ./internal/persistence/postgres` passed.
- `go vet ./...` and compilation of every Go package passed.
- Both UI TypeScript configurations passed. OpenAPI synchronization and Go client
  generation are repeatable; the TypeScript client was regenerated from the
  merged schema. The Python generator passes Ruff formatting and lint checks.
- SQL token comparison against the reviewed implementation confirms that the
  query formatting changes preserve SQL semantics.
- After including the concurrently merged V60-012 correction from `main`
  (`e2d1fbdc`), the PostgreSQL race suite passed again for `evalservice`, public
  HTTP, `runstore`, `auditstore`, `findingintake` and `projectlifecycle`.

The merged scope remains V38-002–005. Uncommitted V38-006 stays in its original
worktree and is not included in this integration.
