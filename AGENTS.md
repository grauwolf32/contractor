# AGENTS.md

Guide for coding agents working in this repository. Read it before you plan or
change anything. It tells you where the truth lives, which rules are not
negotiable, and which mistakes this project has already paid for.

Most of the backlog (805 tasks, 2026-08-29 → 2026-10-10) was not planned
feature work. About 63% of tasks were reactions: GitHub issues, code reviews,
schema reviews and follow-ups. Most of those could have been avoided by
stating a contract, a failure outcome, a scale bound or a real-store test up
front. Section 3 lists those lessons; treat them as a checklist.

## 1. What Contractor is

Contractor is an Application Security orchestration platform. Versioned
**Workflows** define security checks and supporting analyses (OpenAPI and
LikeC4 generation, source review, taint tracing, HTTP/Caido analysis, scanner
checks). A **Run** executes one Workflow. An **Audit** coordinates many
ordinary Runs inside a **Project** against a pinned baseline, and tracks
coverage, evidence, findings and human review.

| Component | Language | Owns |
| --- | --- | --- |
| Server (`cmd/`, `internal/`) | Go 1.26, `net/http`, pgx v5 | Public API, Scheduler, Control Plane, Planners, Audit/Eval controllers, persistence |
| Runtime Agent (`runtime/`) | Python 3.13, uv, Google ADK, A2A | One allocation slot, one Worker, toolsets, workspaces, sandbox |
| UI (`ui/`) | React, Node 24, pnpm | Browser client of the public API; Node serves static files only |
| PostgreSQL | — | All durable state; artifact payloads by default |
| LLM Gateway | external | OpenAI-compatible endpoint (LiteLLM, LM Studio) |

Design direction: make small local models (27b–80b) useful. Deterministic
code owns structure, identity, validation and result assembly; the model does
bounded judgement. New work keeps moving responsibility away from the model
(tool Workers without a model, deterministic scan planning, Runtime-assembled
results, the model as a ranker). Do not reintroduce "the model emits the final
structured package".

The project is **pre-production**. There are no external users, deployments
or legacy data to protect.

### Where the truth lives

| Question | Source of truth |
| --- | --- |
| How should it behave? | `docs/spec/` by domain: start at `docs/spec/README.md`; registries of limits, error codes, failure outcomes and entity lifecycle in `docs/spec/platform/` |
| Designs not implemented yet | `docs/proposals/` |
| Public HTTP/WebSocket contract | `api/openapi/contractor-public-v1.yaml`, `api/events/` |
| Server ↔ Runtime wire contract | `api/v1alpha1/` plus shared golden fixtures |
| What is done, in progress or deferred? | `tasks/BACKLOG.md` (generated), `tasks/*.yml` (open), `tasks/done/*.yml` (finished), policy in `tasks/index.yml` |
| Settled architecture decisions | `docs/decisions/fixed-decisions.yml` |
| Why past epics were shaped as they were | `docs/decisions/planning-history.yml` (non-normative) |
| Shipped Workflows, templates, policies, Skills | `configs/` (see `configs/README.md`) |
| Architecture model | `docs/spec/artitecture.likec4` (`make verify-architecture`) |
| Database schema picture | `docs/spec/database-erd.html` (`make docs-erd`) |
| UI language and layout | `docs/design/ui/README.md`, `ui/src/ui/README.md` |

`docs/research/` and `docs/plans/` are non-normative. A historical report does
not prove current behaviour. If code and spec disagree, find out which one is
wrong and fix that one in the same change; do not leave them disagreeing.

The specification is organised by domain (`docs/spec/conventions.md` has the
rules): `platform/` (limits, errors, failure model, lifecycle, security,
observability, persistence) · `configuration/` (catalog, Workflows,
AgentTemplates, RuntimeConfig) · `execution/` (Runs, Scheduler, Control Plane,
Runtime Agent, Planners, Worker) · `artifacts/` (ArtifactStore, Projects,
deletion, blobs, Git import) · `tools/` (toolsets, Memory, Skills, workspaces,
code analysis, HTTP/Caido, scanners, code execution) · `audits/` · `findings/` ·
`evals/` · `api/` · `ui/` · `operations/`. Each concept has one owning
document; requirements are numbered `PREFIX-Rn` and never renumbered. A spec
never cites tasks; a task cites the spec as
`docs/spec/<domain>/<document>.md#PREFIX-Rn`. `make lint` runs
`scripts/check_spec.py`. Legacy `NN-*.md` names in old task records map to
their new owners in `docs/decisions/spec-migration.md`; unresolved
contradictions found during the migration are in
`docs/decisions/open-questions.md`.

## 2. Ground rules

These are settled decisions. Do not reopen them without the user.

- **No backward compatibility.** No shims, fallbacks, lenient legacy readers or
  "old records" paths. Validate strictly on write and on read. If you find
  legacy-support code, propose removing it. An acceptance criterion that
  demands legacy behaviour is itself a defect; say so.
- **Never edit generated files.** `ui/src/api/generated/public.ts` comes from
  `make ui-generate`; `internal/publicclient/generated` from
  `make generate-public-client`.
- **Tests never read the live `configs/` tree.** Put what you need in
  `testdata/` fixtures. Shipped-config validity is checked by
  `contractor server config validate --root ./configs`. Exceptions:
  `tests/e2e`, `tests/ui-stack`, `tests/eval`. For Skill tests, use a test-only
  pseudo-Skill fixture.
- **Config identities published at runtime are immutable.** Managed
  publication rejects other content under an existing identity, Skill seeds
  and Audit standard packages report drift, and Eval pins reject a changed
  selection; changed Audit standard content under `configs/audit-standards`
  still needs a new version. The rest of the shipped catalog under `configs/`
  (manifests and instructions) is pre-production: until the first release it
  may change in place without a version bump (CAT-R74 in
  `docs/spec/configuration/catalog.md`). Run snapshots and Eval pins record
  digests, so compare digests, not `name@version`, when exact content matters.
  The first release replaces this exception with a digest lock.
- **No `assert` in production Python.** Use `if not cond: raise RuntimeError(...)`
  (Ruff `S101` enforces it outside `runtime/tests/`). In Go, never call
  `os.Exit` below `main`; honour context cancellation.
- **Never log secrets or environment values.** Credentials stay in allocation
  memory or encrypted storage; telemetry content is redacted.
- **Long SQL lives in `*_sql.go`.** Any statement of 10 or more lines goes into
  a commented `*_sql.go` file next to its caller
  (`internal/persistence/sql_placement_test.go` enforces it).
- **Dependencies point down the layer stack.** Go: `internal/archtest/layers_test.go`
  (foundation → kernel → platform → catalog → domain → execution → product →
  edge → composition). Python: `[tool.importlinter]` in `runtime/pyproject.toml`.
  Toolsets share code only through `toolsets.common`. Fix an upward import by
  inverting the dependency or moving the shared type down; a new package must
  be classified; delete a known violation once it is fixed.
- **Model choice in live runs is not yours to change.** Keep the project's
  configured model; if a live run is slow, raise the timeout instead.
- **Do not touch the user's work.** The primary checkout often holds the
  user's uncommitted files. Never `git add -A`, `git stash`, `git reset --hard`
  or `git clean` there. Stage only the paths you changed.
- **Never migrate or write to the demo stand's database.** Use a disposable
  test database or a scratch schema.

## 3. Lessons from the backlog

Each item below is a class of defect that produced dozens of follow-up tasks.
Before you call a change done, check it against every item that applies.

### 3.1 One rule, one definition (largest class, ~100 tasks)

The same grammar, limit or enum was implemented separately in Go, Python,
OpenAPI, the UI and the CLI client, and the copies drifted. Examples: host and
target syntax (V162 → V310 → V356), config version grammar (V323 → V324 →
V362 → V363), secret redaction matching (V179-005 → V311 → V357), CWE catalog,
media-type length, page size and `limit+1` reads, retryability of Gateway
errors, Server vs Runtime result-size accounting.

When you add or change a constrained value (ID or version grammar, size limit,
enum, state set, error code, retry class):

1. Search every implementation: `internal/`, `runtime/src/`,
   `api/openapi/`, `ui/src/`, `internal/publicclient/`, `docs/spec/`.
2. Change all of them in the same commit, or make one of them the source and
   generate or test the others against it.
3. Add or extend a shared fixture that every side validates (see
   `api/testdata/`, for example `scan-host-syntax.json`, and
   `make test-wire-cross-language`).
4. For public API fields, update the OpenAPI document and run
   `make verify-public-api`. Within `/v1`, changes must be additive.

### 3.2 State every failure outcome (~70 tasks)

For each external call, store mutation or cross-process request, decide what
happens on each outcome and test it:

| Outcome | Decide |
| --- | --- |
| Success | Result and side effects |
| Definite permanent failure | Error code, HTTP status, no retry. A deterministic store rejection (`ErrInvalid`) is permanent, never an endless retry (V280) |
| Definite retryable failure | Backoff, attempt and deadline bounds, `Retry-After` |
| Ambiguous (may have committed, response lost) | Reconcile through the subsystem's existing receipt or idempotency key; never blindly repeat an append (V48-001, V120) |
| Cancelled in flight | Who joins the work before cleanup; nothing may outlive release (V60-013) |
| Timeout | Which timeout owns it; do not reuse one timeout for unrelated operations (V49-001) |

Reuse the receipt, lease and idempotency mechanism the subsystem already has.
The 2026-10-11 hub audit counted at least 16 such mechanisms, not the nine an
earlier review found. V370-007 folded three receipt tables and two claim
implementations; ten remain. Do not add an eleventh:

- receipts: a keyed mutation records its receipt in `mutation_receipts`
  through `persistencepostgres.KeyedMutation` (lock, replay, claim;
  PER-R24); a request keeps its key on the row it creates (Runs, Projects,
  RuntimeConfig versions, Runtime credentials, Audit executions, Eval
  members, Audit review requests and decisions); natural keys make proposal
  receipts, Stage resumptions, transition decisions, output publications,
  recovery failures and collection receipts idempotent; journals of
  outbound calls (`credential_operations`, `eval_suboperations`) record a
  call's key before the call; a finding collection's receipt is an Artifact
  binding;
- leases: controller claims through `persistencepostgres.ControllerClaims`
  (PER-R25), the Scheduler's Run claim, the Project deletion claim, the
  Gateway recovery probe and the active Server's session lease.

Every advisory lock key comes from `persistencepostgres.NewAdvisoryKey`,
except the session lease and the per-schema migration lock.
Leases, claims and the single-active-Server assumption are recurring sources
of races (V107, V111, V114, V168, V288): read
`docs/spec/execution/scheduler.md` and `docs/spec/platform/persistence.md`
before touching them.

### 3.3 Bound the work, not just the input (~70 tasks)

Domain limits existed (items per Round, members per Eval), but work per
controller tick, per transaction, per page and per UI request did not. Typical
results: N+1 queries, whole-collection rescans on every poll, per-row triggers
dominating inserts, pickers that only show the first 50 entries, Memory
disabled because a Run had 7000 ordinary artifacts (V48-002).

- Write down the expected and maximum cardinality of every collection you
  read, including the containing collection.
- Bound each loop iteration, transaction and response by rows and bytes; page
  every list, and make UI pickers follow every page.
- Prove it with a test at a realistic size, not only with three rows.
- In the Runtime, run blocking I/O and CPU-heavy parsing off the event loop
  with `threads.to_thread_until_done` and keep lock hold times short
  (V14-003, V221, V346).

### 3.4 Lifecycle: delete, purge, pause, resume, cancel (~20 tasks)

For every new entity or state, decide who can delete it, what blocks
deletion, what cascades (FK `ON DELETE`), what is retained and for how long,
and what happens to pending reviews or requests when the parent closes. Test
purge of a fully populated parent, including human decisions. Audit Rounds and
reviews hit every unplanned combination (V205, V236, V278, V279, V285).

### 3.5 Tests must be able to fail (~60 tasks)

- Persistence behaviour is tested against real PostgreSQL. A fake store must
  enforce the same validation as the real one; V280 passed for weeks because
  its fake did not validate.
- No wall-clock assertions; use injected clocks and deterministic budgets.
- A database test skips only while `CONTRACTOR_TEST_DATABASE_URL` is unset.
  Once it is set, an unreachable database must fail the test.
- A gate or `-run` selection that matches zero tests is a failure.
- If you rename a test, change a catalog entry, a config default, a Gateway
  error class or a UI selector, run every gate that consumes it. In the V60
  review, 14 of 41 tasks only repaired gates broken this way.

### 3.6 Fix the class, not the instance

The 2026-10-04 review found 28 of 127 completed fix tasks partial or
regressing, and 118 tasks in total exist to repair earlier fixes or review
tasks. Before closing a defect:

- find every sibling code path, language and encoding that applies the same
  rule, and fix or explicitly rule out each one;
- add a regression test on the real path;
- run the full suite of every package you touched, not only the new test.

### 3.7 Build only what has a caller (~50 tasks)

Many tasks only removed exports, states, API operations and tables that no
production code used, or merged tables that duplicated a mechanism (91 → 70
tables). Every new export, state, endpoint and table needs a production caller
or reader in the same change. `staticcheck -checks U1000` with the `e2e` and
`integration` tags runs in `make lint`, and so does `tools/testonlyexports`,
which fails on an exported identifier that only tests reference: put a test
hook in the package's `export_test.go` and a helper that other packages' tests
share in a test helper package named `*test`; its baseline only shrinks.
Prefer column-scoped immutability triggers over 1:1 side tables.

### 3.8 UI must tell the truth (~45 tasks)

- Refetch when an object reaches a terminal state; stop polling settled
  objects; back off on failed reads.
- Show stale-revision conflicts instead of overwriting a newer draft.
- Never claim a state the Server has not reported (V93-002 claimed a Run was
  cancelled while it was still active).
- Confirm destructive actions before sending `DELETE`.
- Use the user's words from `docs/design/ui/README.md` (materials, check,
  possible issue); keep internals behind *Technical details*. Check 390 px
  width and the light, dark and black themes.

### 3.9 Accepted risks need an owner

If you accept a risk or declare a non-goal, write down how it is enforced or
detected and when to reopen it. "One active Server", "128 notes, no
pagination needed" and "whole-batch retry is acceptable" were accepted
without enforcement, and each came back as several defect tasks.

### 3.10 Record decisions at boundaries

The Worker result boundary was decided four times (MVP-012 → V5-001 → V13-001
→ V17-003 → V21-001, then V57-003 reopened it). When you choose between
designs at a component boundary, record the guarantee, the alternatives, the
evidence and the reversal condition in the spec or the task, so the next agent
does not argue it again.

## 4. Tasks

`tasks/` is the project tracker (`gh` is not available). The rules are in
`execution_policy` at the top of `tasks/index.yml`; read them before you
create or change a task.

- **Layout.** Open tasks (`pending`, `in_progress`, `deferred`) live in
  `tasks/`; finished tasks (`completed`, `archived`) live in `tasks/done/`.
  The task files are the index; there is no separate list to maintain.
  `tasks/BACKLOG.md` is generated from them: run `make tasks-backlog` after
  you add a task or change a status, priority, epic or dependency.
  `scripts/check_tasks.py` (part of `make lint`) enforces the layout, declared
  statuses, priorities, epics and kinds, IDs matching file names, full
  implementation commits, known dependencies and `introduced_by` references,
  and milestone exit tasks.
- **Status.** Set `in_progress` before starting. Set `completed` only when
  every acceptance criterion passes and every listed test command succeeds,
  and record `implementation_commit` as the full hash of the last commit whose
  primary purpose completed the task. In that same commit, `git mv` the file
  into `tasks/done/` and repoint links to it. Do not pick up another agent's
  `in_progress` task. `deferred` and `archived` tasks are not eligible and do
  not satisfy dependencies; reactivating them needs the user.
- **Classification.** Every task sets `epic` (one of the areas in
  `tasks/epics.yml`, the area that owns the change) and `kind`: `feature`
  adds or extends a capability, `defect` corrects behaviour that was wrong,
  `chore` changes no intended behaviour (tests, CI, dead code, refactoring,
  docs, schema consolidation). A defect or chore that repairs earlier work
  sets `introduced_by` to the task whose implementation caused it; find it
  with `git log`/`git blame` on the code you fix. This is how the project
  learns which epics leave gaps, so do not skip it when it is knowable.
- **Intake.** GitHub issues are an intake channel only: turn an issue into a
  task file (link it from `context.issue`) and close the issue with a link to
  the task. The task file is what agents work from.
- **Inputs.** A task may start only when everything in `depends_on` is
  complete. Documents named in `context.specification`, `context.design`,
  `context.format` or `context.proposal` are required reading.
- **Create the task first, even for small fixes.** Work done without a task
  file (about 180 commits on 2026-09-22/23, the V3B redesign, the layer checks)
  later produced defects that nobody could trace.
- **Start from a template.** Copy `tasks/templates/feature.yml`, `defect.yml`
  or `chore.yml`. The templates carry the sections that were missing when
  later fixes became necessary.
- **Write tasks as small specifications, not one-line tickets.** Early tasks
  (scope in/out, requirements, given/when/then acceptance, test commands with
  `covers`) held up; later tickets with two flat criteria did not. A feature
  task states failure outcomes, cardinality and lifecycle cases. A defect task
  names the finding with `file:line`, lists every sibling path in scope, and
  requires a regression test on the real store.
- **Keep the index consistent.** Use the owning epic's milestone, or declare a
  new milestone with `exit_task` and `outcome` under `milestones:` in
  `tasks/index.yml`; do not create one milestone per defect. Add an epic to
  `tasks/epics.yml` only for a new lasting area. Use only declared
  priorities (P0–P3).
- **One task per commit**, unless two tasks must land together to keep the
  build green.
- **Land your branch.** V61 (9 tasks) was completed on a feature branch and
  never merged, so `main` still lists it as pending. When you finish work on a
  branch, say whether it is merged and what is left.

## 5. Build and verify

Run commands from the repository root. Pick checks by the boundary you changed
(`docs/testing/README.md` has the full table).

| Purpose | Command |
| --- | --- |
| Format | `make fmt` |
| Lint, layers, release-graph guard | `make lint` |
| Go tests (includes PostgreSQL tests when the URL is set) | `make test-go` |
| Runtime tests | `make test-runtime` |
| UI generate-check, lint, typecheck, tests, build | `make ui-verify` |
| Everything above plus builds | `make verify` |
| PostgreSQL repositories | `make test-postgres` |
| Server + Runtime process tests | `make test-e2e` |
| Public API contract | `make verify-public-api`, `make verify-public-api-postgres` |
| Cross-language wire fixtures | `make test-wire-cross-language` |
| Shipped configuration | `contractor server config validate --root ./configs` |
| Architecture model | `make verify-architecture` |
| Full CI gate | `make release-verify` (`make -k` to see every failing stage) |

Notes:

- Database tests need a disposable database whose user can create and drop
  schemas:
  `export CONTRACTOR_TEST_DATABASE_URL='postgres://…/contractor_test?sslmode=disable'`.
  Parallel agents use separate databases or schemas.
- Build tags: `integration` marks PostgreSQL-only integration tests (picked up
  by the release gate automatically); `e2e` marks process tests, which must run
  in `release-verify` or be allowlisted with a reason in
  `scripts/check_release_verify_graph.py`.
- Family targets (`test-audits-*`, `test-*-e2e`, …) run their focused Go
  suites when invoked directly but skip them inside `release-verify`
  (`RELEASE_CONSOLIDATED=1`). Do not conclude a suite ran from a release log
  without checking.
- Live-model checks (`docs/testing/live-models.md`) call a real Gateway and
  are never part of the deterministic gate.
- Report results as they are: name the commands you ran and their outcome. A
  skipped database test is not a pass.

## 6. Component conventions

### Go server

- Decode untrusted JSON with `strictjson.Decode`. Anywhere in the value it
  rejects duplicate keys and a member whose name matches no field exactly (a
  case alias such as `Reason` for `reason` is unknown, although
  `encoding/json` would accept it, and so is any other unknown member); it
  also rejects data after the value. A type with its own `UnmarshalJSON` is
  checked for duplicate keys only, so such a method must decode its members
  with `strictjson.Decode` itself. Plain `json.Unmarshal` and `json.Decoder`
  check none of this.
- Return classified errors; map them to stable API codes and statuses. Every
  handler that reaches PostgreSQL documents the retryable `503`
  `storage_transaction_conflict`.
- Migrations live in `internal/persistence/migrations/NNNNNN_name.sql`,
  numbered sequentially. A migration that moves or folds data has a
  `*_migration_test.go` in `internal/persistence/postgres`.
  `000001_baseline.sql` is the whole schema as of 2026-10-11 (V367-001);
  a database migrated before that must be recreated.
- A table needs a named reader; a transaction-scoped lookup uses the
  transaction, never a second pool connection (nested acquisition deadlocked
  under `MaxConns=1`, V60-003, V60-015).

### Python Runtime

- `uv sync --locked`; Ruff (line length 100) and import-linter must pass.
- Agent-facing tool functions are what the model sees: give them accurate
  docstrings, keep the `Returns` description general, and make examples match
  real behaviour (V178, V184). Internal helpers do not need that.
- Keep a Python exception (a bug or infrastructure failure) separate from a
  tool-result error returned to the model; they are different signals in
  events and UI.
- The Worker passes its instruction to ADK as a string, so ADK replaces any
  bare `{identifier}` with a session-state value and fails when it is missing.
  Avoid bare braces in instruction text, or pass a provider as the summarizer
  does (`instruction=lambda _: text`).
- Tree-sitter behaviour differs per language; add a case for each supported
  language you touch (V161, V173, V313–V315).

### UI

- Node 24.20 with Corepack and pnpm (`corepack pnpm --dir ui …`).
- Shared V3B building blocks live in `ui/src/ui`; colours come only from
  `ui/src/app/theme.css` tokens. Shared CSS must not live in lazily loaded
  route CSS.
- Bump `ui/package.json` for each UI release (patch, minor, major).

### Configuration, prompts and evaluations

- AgentTemplate instruction text describes how the Worker uses its tools.
  Planner objectives and instructions carry per-subtask context. Skills hold
  domain knowledge and non-trivial tool instructions. Put a fix in the surface
  that owns the behaviour.
- When an evaluation fails, teach general methodology. Never encode a
  benchmark's specific bugs, sinks or file names into prompts or Skills; that
  overfits and invalidates the evaluation.
- Keep ground truth outside the project folder the agent scans.
- A single sample is noise. Compare variants over repeated samples before
  claiming an improvement.

## 7. Git and delivery

- Commit subjects follow `type(scope): summary` in the imperative, e.g.
  `fix(audit): …`, `feat(ui): …`, `refactor(contracts): …`, `test(arch): …`,
  `docs(tasks): …`. The body says why.
- Do not commit straight to `main` unless asked. Work on a branch (one
  worktree and one test database per parallel agent); the user merges into
  `main` with `--no-ff`.
- CI runs on pull requests, `v*` tags and manual `workflow_dispatch`. A push
  to `main` does not run it. Branch protection (V347) is not in place yet, so
  nothing remote stops a broken merge: run the relevant gates locally first.
- Never force-push or rewrite shared history without the user.

## 8. Before you say "done"

- [ ] The spec, OpenAPI, shared fixtures and every implementation of a changed
      rule agree (3.1).
- [ ] Every failure outcome of new calls and mutations is decided and tested
      (3.2).
- [ ] Reads and loops are bounded, and tested at a realistic size (3.3).
- [ ] Delete, purge and cancel paths are covered (3.4).
- [ ] Tests use the real store or process and can fail; gates you affected
      were run (3.5).
- [ ] Sibling paths of a fixed defect are fixed or ruled out (3.6).
- [ ] Nothing new is without a production caller (3.7).
- [ ] `make lint` and the tests for every touched package pass; you report
      exactly what you ran.
- [ ] The task file is updated (status, commit hash, moved to `tasks/done/`
      when finished) and `make lint` accepts it.
