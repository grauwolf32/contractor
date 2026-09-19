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
