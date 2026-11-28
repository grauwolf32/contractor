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
| Runtime Agent (`runtime/`) | Python 3.13, uv, Google ADK, A2A | 1–4 allocation slots per process, one Worker per allocation, shared service owners, private workspaces/sandboxes |
| UI (`ui/`) | React, Node 24, pnpm | Browser client of the public API; Node serves static files only |
| PostgreSQL | — | Durable metadata and inline artifact payloads |
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
| Shipped Workflows, templates, policies, Skills | `configs/` (see `configs/README.md`) |
| Architecture model | `docs/spec/artitecture.likec4` (`make verify-architecture`) |
| Database schema picture | `docs/spec/database-erd.html` (`make docs-erd`) |
| UI language and layout | `docs/design/ui/README.md`, `ui/src/ui/README.md` |

`docs/design/`, `docs/research/` and `docs/plans/` are non-normative. A
historical report does not prove current behaviour. If code and spec disagree,
find out which one is wrong and fix that one in the same change; do not leave
them disagreeing.

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
`scripts/check_spec.py`.

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
  still needs a new version. The shipped catalog under `configs/` is pinned by
  `catalog.lock.json` (CAT-R74). Changed content needs a new manifest version
  or instruction path. `config lock` adds identities but refuses to rewrite
  existing digests; release validation requires the lock. Run snapshots and
  Eval pins remain exact.
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
- **Live evals and research use the demo stand.** User decision 2026-11-28:
  run them on the existing demo deployment; do not start separate Server,
  Runtime, PostgreSQL or catalog copies for an eval/research campaign. Create
  task-owned Projects, Evals and immutable config identities on that stand,
  preserve its business data and configured models, and enable debug traces
  before live execution. Verify trace delivery before the repeated series.
  This does not change the disposable-database rule for deterministic tests
  or recoverable-backup verification.
- **Do not touch the user's work.** The primary checkout often holds the
  user's uncommitted files. Never `git add -A`, `git stash`, `git reset --hard`
  or `git clean` there. Stage only the paths you changed.
- Before migrating the demo database, take and verify a recoverable backup,
  preserve existing business data and record the applied migration versions
  in the release manifest. A user-requested demo deployment authorizes the
  schema migrations required by that revision and normal Server startup
  writes; do not ask for separate migration approval. Resetting, bulk-deleting
  or replacing demo business data outside required migrations needs separate,
  explicit user authorization.

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
   `make verify-public-api`. Until the first release `/v1` may change
   incompatibly (API-R3 in `docs/spec/api/public-api.md`): remove an unused
   field, parameter or value end to end (Server, OpenAPI, both generated
   clients, UI) in one commit and record it in the task, rather than keep a
   variant only to stay additive.

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
  User decision 2026-11-05: the named disk-heavy fixtures are an explicit
  exception. They skip before resource creation unless the caller supplies
  `CONTRACTOR_TEST_HEAVY_IO=1`, including full verification and CI. Never enable
  this flag implicitly or count their disabled skips as passing coverage.
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

### 3.11 Keep source details at their boundary

Before implementation, identify who owns source interpretation, domain rules,
authority and persistence; record that boundary under 3.10. First distinguish
external-provider adaptation from a domain operation over existing artifacts.
An artifact's producer does not determine where its consumers execute: a
tool-generated OpenAPI document is still an API contract artifact. Joining it
with a sitemap is a domain transformation, not another provider ingestion path.

For tool-produced observations and reports, provider-specific parsing, field
mapping and malformed row handling belong in the owning Runtime adapter. Server
validates the common normalized input and enforces authorization, domain invariants, bounds, canonical
identities, grouping and receipts. Adding or replacing a provider should require
an adapter and composition changes, without new Server branches to interpret its
format. Provider labels are provenance; enums expressing actual domain rules
remain appropriate.

Place artifact transformations with the module that owns their domain rules,
reusing existing parsers and algorithms; they may execute on Server. A separate
Workflow stage defines sequencing and inputs/outputs, not execution location.
Inspect the actual stage execution path before choosing its integration. Do not
add a Runtime round trip, duplicate a parser or introduce a model merely to fit
an existing tool Worker. If a concrete operation needs a missing execution
capability, extend the common mechanism while keeping domain logic in its owner.

Build artifact-based features on the ordinary Artifact read, metadata, revision,
publication and retention mechanisms. Keep domain transformations in their owning
module. When a concrete caller needs a missing shared capability, extend the
existing owner rather than introducing a parallel feature-specific store or API;
the caller requirement in 3.7 still applies. This does not justify a speculative
plugin framework or a new abstraction for hypothetical future providers.

When requirements change, first identify old paths, states and contracts that
can be deleted. Passing tests establish behavior under those tests, not the need
for the architecture. Review responsibilities and existing mechanisms before
building and testing an additional one; do not preserve an unnecessary mechanism
merely because it is already implemented and covered by tests.

## 4. Tasks

### 4.1 Shaping work

This applies when you turn a request, proposal or finding into specifications
and tasks, and when work in progress outgrows its task. A task that already
states its scope and acceptance is the contract for executing it
(`task_files_are_self_contained` in `tasks/index.yml`): do not restate or
re-plan it; come back here only when it proves wrong or too small.

- **Start from the outcome.** State the user-visible outcome, the smallest
  end-to-end scenario through real components that shows it, and the
  assumptions that would change either, in product terms before
  implementation details. In a feature task these are `goal` and
  `demonstration`.
- **Keep track of who asked for what.** A requirement comes from a user
  decision, the specification or an agent proposal. Record user decisions with
  their date in `context.authority` (as V397-009 does) and label agent
  proposals as proposals. Do not turn an inferred need, a stronger guarantee or
  a possible future use into a requirement; 3.7 counts what removing those
  later costs.
- **Settle consequential choices before specifying them.** A choice is
  consequential when it changes user-visible behaviour, a guarantee, component
  ownership or the expected amount of work. Give the recommended option, a
  simpler viable alternative and the trade-off, and record the outcome as 3.10
  describes. Ask only when instructions and evidence leave the choice open;
  make routine choices within the agreed direction yourself.
- **When nobody can answer now** (a parallel lane, a background agent, a
  queued task), neither guess nor stall: write the open choice, its options and
  your recommendation into the task as `context.decision_needed` (V371-004,
  V55-009), leave or return the task to `pending`, and continue with work that
  does not depend on it.
- **Demonstrate early.** For a feature that spans several tasks, make the first
  deliverable a narrow end-to-end path through real components and let its
  result shape the later tasks. A working demonstration is not release
  readiness; the milestone's release gate is (V410-007, V411-020).
- **Report scope growth instead of absorbing it.** When the work needs a
  prerequisite the task did not foresee, or materially widens, stop and say
  what you found, why the current approach is not enough and which smaller
  options remain. A new prerequisite gets its own task; do not grow supporting
  subsystems under the original one.
- **Re-plan when direction changes.** The last paragraph of 3.11 applies to
  plans as well as code: review the open dependent tasks, specifications and
  checks, and revise or remove what the new decision makes obsolete, within
  the authorized scope. An older acceptance criterion does not override a newer
  user decision; record that decision in each affected task's
  `context.authority`.
- **Report against the outcome.** Say what now works for the user, what is
  still unproven and whether scope or complexity changed. Finished
  infrastructure tasks alone do not make a feature ready.

### 4.2 Task files

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
- **Create the task before changing code, even for small fixes.** Tasks track
  code changes, including their tests and accompanying specification or API
  contract updates. They are not a tracker for every user request. Deploying
  an existing revision, restarting services, checking logs or status, answering
  questions, read-only investigation and editing agent instructions do not
  require a task. Record deployment results in the release manifest or
  operational log. If an investigation or deployment requires a code change,
  create its task before changing code. Code work done without a task file
  (about 180 commits on 2026-09-22/23, the V3B redesign, the layer checks) later
  produced defects that nobody could trace.
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

- **Do not monitor healthy test progress.** Run tests with output saved to a
  log or report, then consume only completion status and the required final
  coverage/skip summary. Inspect detailed logs only after a failure, timeout
  or abnormal termination. Do not repeatedly poll logs, count passing cases,
  read test code to narrate progress, or send per-test progress updates. While
  tests run, do independent work or wait for completion without analysing
  intermediate output. User decision 2026-11-27.
- Database tests need a disposable database whose user can create and drop
  schemas and databases (`CREATEDB`; read-budget tests measure in a database
  of their own, see `docs/testing/README.md`):
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
- Large disk-write fixtures are optional even in full runs. Explicitly use
  `CONTRACTOR_TEST_HEAVY_IO=1` to include them; the CI manual dispatch input is
  `heavy_io` (default false). See `scripts/disk_heavy_policy.py` and
  `docs/testing/README.md`; ordinary database regressions remain required.
- Reuse one disposable test database per lane; after its processes and pools
  stop, explicitly drop the database you created. Do not leave a database per
  command or infer that another lane's disconnected database is abandoned.
  `make test-storage-report` inventories storage; `make test-cache-prune`
  previews cold Go cache pruning (`APPLY=1` applies it). See the local test
  storage section of `docs/testing/README.md` for database and explicit Podman
  cleanup; `make test-podman-storage-report` inventories the selected engine.
  Remove owned test containers, volumes and networks when their lane finishes;
  never use global Podman prune/reset against a shared development engine.

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
  `000001_baseline.sql` is the whole schema as of 2026-11-23 (V480-001);
  other stale ledgers fail drift; PER-R30 records the explicitly authorized
  pre-production demo reset.
- A table needs a named reader; a transaction-scoped lookup uses the
  transaction, never a second pool connection (nested acquisition deadlocked
  under `MaxConns=1`, V60-003, V60-015).
- Every table and view has one owner package, the only one whose SQL writes
  it (PER-R23); the registry is
  `internal/persistence/table_ownership.txt`. A new or renamed table needs, in
  the same change: its `[relations]` line (owner and ERD group), its row in
  the "Table owners" table of `docs/spec/platform/persistence.md`, its entry in
  `scripts/erd/meta.json`, and a page rebuilt with `make docs-erd`. Another
  package's read needs a `[reads]` line with its reason; a write goes through
  the owner's Go API. `make docs-erd` also runs whenever a package starts or
  stops using a table, since the page lists the packages that use each one.

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

- Node 24.20 with Corepack and pnpm, run from `ui/` as make and CI do
  (`cd ui && corepack pnpm …`). From the root, `corepack pnpm --dir ui …`
  starts Corepack's default pnpm, which refuses the pinned version.
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
- [ ] Source details stay in adapters, shared mechanisms are reused, and
      superseded paths are removed (3.11).
- [ ] Progress is reported against the user-visible outcome, and any scope
      growth or open choice is recorded in the task (4.1).
- [ ] `make lint` and the tests for every touched package pass; you report
      exactly what you ran.
- [ ] The task file is updated (status, commit hash, moved to `tasks/done/`
      when finished) and `make lint` accepts it.
