# Managed experiment orchestration

`Service` prepares exact native plans and accepts native commands or external
registrations/submissions. `Driver` reconciles only previously admitted members
through separate Workflow and Audit adapters. Shared operation persistence owns
intent/replay/acknowledgement; member resources own artifact copying and Audit
workspaces. Metric SQL lives in `evalstore`, not in execution adapters.

`evaldomain.Lifecycle` owns command targets, allowed UI actions, observed
transitions, admission, budget stops and command recovery. Deadline/token stops
continue through settling until all accepted work drains. Unknown preparation
failures stay pending for retry, with a safe diagnostic and the original cause
returned to the coordinator. Confirmed configuration failures return to draft.
Preflight records both LLM and Runtime credential lookup failures before catalog
validation or Runtime pinning can obscure their classification. Database failures
remain retryable even when a downstream validator adds `ErrInvalid`.

Portable documents and operation receipts have named types. Plan construction
uses separate check/case/variant/member builders. Nested validation follows known
DTO fields; arbitrary output roles and producer data never choose a validator.
`evalcoordinator` claims experiments in PostgreSQL and calls bounded ticks; the
ordinary Scheduler and Audit controller continue to own execution.

Preparation resolves exact Workflow/AuditProfile, Runtime, Skills, standards and
artifact selections without creating an execution, artifact or target. It saves
the portable plan and its private resource closure atomically. Required equality
pins must be observed, present and equal; unavailable provider model revisions
are not inferred from model names. The fixed registry supports pinned human review,
required artifacts, media types and bounded allowlisted JSON schemas. No uploaded
evaluator is executed.
Binding selection and dependency pinning, case eligibility, artifact checks and
comparison pins have separate implementations; `Resolve` owns their transaction.

Input/output mappings run from case role to executable slot. Omitted input roles
use their own name; collisions and unknown mappings fail. Variant parameters
override case parameters; `$task.objective` resolves the case objective. Each
Audit has a newly owned kind=project workspace, and inputs are copied atomically
into a member-specific namespace. An Audit variant currently selects an exact
AuditProfile: execution overrides are rejected because ordinary Audit creation
has no override contract. Native Workflow variants use ordinary execution patches.

The ordinary creation services verify prepared selection hashes again inside
their transactions, after checking existing idempotency receipts. Drift cannot
silently change an experiment; a lost response still recovers the original effect
after mutable catalogs change. External registration retains its attributed
source manifest and separate local binding snapshots. It never imports authority
from a producer's labels, host paths, journal or claimed binding hashes.

Submission keys include the managed experiment lookup identity and member ID, so
identical portable invocation IDs in separate Projects cannot collide in the
ordinary owner-wide Run key namespace. The frozen portable CLI format is unchanged.
Run creation and Audit workspace/create/start/cancel each retain their own exact
request. Unknown outcomes remain outstanding. Ordinary deletion captures an
exact-identity tombstone so a deleted execution cannot be recreated after a lost
response. An Audit deleted before Start remains explicitly never-started.

Pause drains accepted work, resume retains the original clock and cumulative
known usage, and explicit optional token limits do not change global defaults.
Budget accounting is a bounded per-member high-water lower bound over unique
stage metrics across all owned Audit roles. Nullable per-dimension usage preserves
missing reports, exact source digests and parent wall duration. Terminal members
are never resampled. Duplicate copies native
authoring intent into a fresh draft; external producers create new invocations.

Native ticks never admit external members. A failed member observation rotates
within bounded polling and cannot prevent the deadline fence or reconciliation
of other accepted members. Ready/terminal states with pending command receipts
remain claimable to recover a crash after a lifecycle transition commits.

Verification uses a disposable PostgreSQL and isolated schemas:

```sh
test -n "$CONTRACTOR_TEST_DATABASE_URL" &&
  go test -race -count=1 ./internal/evalservice ./internal/evalcoordinator ./internal/projectlifecycle
```

The integration fixtures call real ordinary Run/Audit services but never run
models or contact targets. Tests lose committed creation/start responses, replace
coordinators, enforce pin checks and deletion fences, exercise native/external
ownership, and verify eight-member plans without duplicate effects. Regression
checks cover deadline/token cancellation of eight active members already in
settling (Workflow and Audit), a real PostgreSQL preparation failure followed by
recovery, safe diagnostics, legacy receipt replay and dataset role names that
coincide with Summary fields.

Public authoring/control and bounded observation routes are provided through the
ordinary authenticated HTTP adapter (V38-005). Mutation responses are retained
receipts; GET returns current state. Collection lists expose safe summaries,
while owner detail may include visible draft inputs. Exact capabilities are
paginated from the existing catalogs. V38-006 adds exact result/assessment revisions,
owner review, explicit CAS selection, pair pages, charts, reports and full member
execution inventories. No assessment is inferred from execution success.

Collection separates authoritative observations, native record construction and
selected-record validation. The first native result/check pair is selected by an
audited system CAS; later, human and external records require explicit selection.
Record receipt replay precedes evidence revalidation. Human review persists if a
subsequent selection races. External result ingestion cannot overwrite execution
state, timestamps or measured usage; producer activity advances only on new
accepted operations, not receipt replay.

Read methods use one repeatable-read snapshot. Safe reports omit private rubrics
and free-text review reasons; only the dedicated owner review endpoint returns
pinned private criteria. Charts and filtered pages retain the same unfiltered
summary. Signed bin tokens bind owner, experiment, snapshot, suite and scope.

V38-006 verification adds native structural failures, deleted evidence, retained
reviews, external assessment revisions, all-role Audit inventories, chart/bin
consistency and bounded observation gaps. The 10,000-member persistence fixture
measures selected-page query counts and response sizes; it does not claim an
end-to-end 10,000-member portable authoring journey. Browser setup/comparison,
Playground integration and full release acceptance remain V38-007–010.
