# Availability of managed-credential deletion and recovery

Proposal 0044 · Status: **Draft; recommendation only. RCF-R52, RCF-R53 and
RCF-R72 stay in force until a separately accepted feature task changes them.**

Recorded on 2026-10-23 for V407-018, from findings D1 and D13 of the
[2026-10-23 Server and Runtime review](../research/2026-10-23-server-runtime-review.md),
and checked against `925a614c`, which includes V407-019's reconciliation of the
process-wide deletion fence. It describes how managed LLM credential deletion
and startup recovery trade availability for safety today, maps every credential
reader and writer to the fence that protects it, and compares narrower
alternatives. It changes no behaviour and authorizes no implementation.

## 1. Decisions

| ID | Question | Current contract |
| --- | --- | --- |
| D1 | How long does a managed delete hold the process-wide credential-lifecycle barrier? | From its first reference check through the Gateway request and the local tombstone commit ([RCF-R52](../spec/configuration/runtime-config.md)) |
| D2 | When reconciliation of an uncertain delete intent finds the intent durable, what stays fenced? | Every Run initialization and allocation reference, for every credential ([RCF-R53](../spec/configuration/runtime-config.md)) |
| D3 | Who completes a durable prepared operation whose client never retries with the same Idempotency-Key? | Startup recovery only (RCF-R53, RCF-R72) |
| D4 | What happens at startup when a prepared operation exists and the Gateway management API does not answer? | Startup fails ([RCF-R72](../spec/configuration/runtime-config.md)) |

Each decision keeps the current contract as a legitimate outcome (option A),
with its detection and reopening condition.

## 2. Safety property

Every option must keep these, which are the task's R2 and scope limits:

- **S1.** A delete records its intent only while no committed or committing
  non-terminal Run, live allocation, active label binding or Audit dispatch
  hold references the credential (RCF-R48, RCF-R128).
- **S2.** Once a delete intent for credential C may be durable, no new Run,
  allocation, binding or Audit hold commits a reference to C until the deletion
  completes or durable state proves the intent absent. The Gateway is called
  only after the intent COMMIT is acknowledged (`executeDelete` runs after the
  intent transaction returns), so "the remote key may be gone" implies "the
  intent may be durable".
- **S3.** The operation journal (`credential_operations`), tombstones,
  reference checks, single-active-Server ownership and secret handling stay as
  they are.

## 3. Current mechanisms

Line numbers are those of `925a614c`.

| | Mechanism | Covers | Code |
| --- | --- | --- | --- |
| M1 | Credential-lifecycle barrier: one process-local `sync.RWMutex` | every LLM and Runtime credential | `internal/credentials/lifecycle_barrier.go`; one instance shared by both services (`internal/app/composition_credentials.go:115-136`) |
| M2 | `lifecycleMu`, a `sync.Mutex` | managed LLM credential create, delete and recovery | `internal/credentials/lifecycle.go:127`, taken at `:344`, `:416`, `:486` |
| M3 | Journal singleton: a prepared operation under another key refuses every managed create and delete with `credential_recovery_required`, so the journal holds at most one prepared operation | managed LLM credential mutations | `preparedFor`, `lifecycle.go:869-883` |
| M4 | Per-credential durable fence: every metadata lookup refuses a credential with a prepared delete | references to one LLM credential | `internal/credentials/encrypted_provider.go:48-58`, pooled and transaction-bound (`transaction_lookup.go`); `CompositeProvider.ResolveLLMCredential` looks up first (`composite.go:63-77`) |
| M5 | Per-credential advisory lock `llm-credential-reference:<id>` | RuntimeConfig references to one LLM credential | `internal/runtimeconfig/credentials.go:163-209`: shared in publication and binding transactions, exclusive in the delete-intent transaction (`lifecycle.go:566-568`), tried without waiting by reconciliation (`:285`) |
| M6 | Process-wide uncertain-delete fence | the Run-initialization and allocation guards only | raised when the intent transaction fails without a definite outcome (`lifecycle.go:610-614`); a guard reconciles under M5's try-lock and clears it only when no prepared delete remains (`:212-327`) |
| M7 | Readiness | the M6 guards and managed credential reads | `ready` is false until `Recover` drains the journal (`lifecycle.go:343-403`); `Recover` runs only from `configureCredentials` (`composition_credentials.go:126`), before any listener opens (`internal/app/app.go:180`) |

Holders of M1: the shared side is taken by the reference paths of section 4;
the exclusive side by a managed delete (`lifecycle.go:491-494`), by recovery of
a prepared delete (`:391-395`) and by Runtime credential create and delete
(`runtime_service.go:99`, `:275`). Managed credential creation takes M2 only: a
new credential has no reference to protect.

## 4. Reference inventory and operation matrix

Availability columns: **(a)** a managed delete of C is waiting for or holding
M1 (Gateway request in flight); **(b)** C has a durable prepared delete after a
Gateway failure; **(c)** M6 is raised (intent outcome unknown, or found durable
after a lost COMMIT reply); **(d)** startup with a prepared operation while the
Gateway management API is down. "Other" means a credential other than C.

| # | Path | Pins | Fence today | (a) | (b) | (c) | (d) | Safety proof if M1's hold or M6 is narrowed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Run creation: `runservice.CreatePublic`/`CreatePublicIn` for `POST /v1/runs`, Eval members and scan sweeps (`internal/runservice/public.go:62`) | LLM (Workflow, RuntimeConfig), Runtime (RuntimeConfig, Project target) | M1 shared, M6, M7; Workflow credentials through the pooled M4 lookup before the Run transaction, RuntimeConfig credentials through the transaction lookup | waits | other: admitted; C: 503 `credential_recovery_required` | 503 for every credential | no Server | M4 read after entering M1 |
| 2 | Audit-managed Run creation (`internal/runservice/audit.go:34`, called from `internal/auditcontroller/controller.go:610`) | the Audit's pinned baseline | M1 shared, M6, M7; no lookup: the Audit's dispatch hold is a counted reference | waits | unaffected (a held credential cannot get an intent) | refused; the controller retries on its next pass (`controller.go:710-713`) | no Server | S1 through the hold |
| 3 | Audit start and resume (`internal/auditservice/start.go:64`, transaction lookup `:274`; `resume.go:38`) | baseline credentials (`start.go:342`) | M1 shared, M6, M7, M4 in the start transaction | waits | C refused | 503 for every Audit | no Server | M4 (start), hold (resume) |
| 4 | Eval preflight (`internal/evalservice/preflight.go:58`) | none durable; members create Runs through row 1 | M1 shared, M6, M7, M4 in its transaction | waits | C refused | refused | no Server | M4 |
| 5 | Allocation placement (`internal/controlplane/placement.go:299`, transaction lookup `:377`, `:706-726`) | LLM and Runtime credentials of the Run selection and of default, Run and Agent labels | M1 shared, M6, M7, M4 in the placement transaction | waits | C refused | every Stage placement of every Run fails; the lane logs an error, releases its claim and retries on the next poll, 1 s by default (`internal/scheduler/supervisor.go:85`, `:198-202`); this error path does not evaluate the Stage deadline (`stage_preparation.go:164-188`) | no Server | M4 |
| 6 | RuntimeConfig publication (`internal/runtimeconfig/publisher.go:89`, `:126`) | LLM, Runtime | M1 shared through the Runtime credential service, M5 shared, M4 in its transaction; **not** M6 or M7 | waits | C refused | admitted | no Server | M5: the lock waits for an open intent transaction, then the transaction lookup reads its committed outcome |
| 7 | Binding create, rebind (`management.go:96`, `:143`; `binding_service.go:47`) | LLM, Runtime | as row 6 | waits | C refused | admitted | no Server | as row 6 |
| 8 | Binding delete (`management.go:227`) | removes a reference | M1 shared | waits | unaffected | unaffected | no Server | removes references only |
| 9 | Agent-label replacement (`internal/agentprincipal/service.go:237`) | a label, never a credential | binding row locks only | unaffected | unaffected | unaffected | no Server | the binding is the counted reference; row 5 resolves its credentials |
| 10 | Project HTTP target attachment (`internal/httpapi/public/project_handlers.go:227`) | Runtime | M1 shared | waits | unaffected | unaffected | no Server | Runtime credentials have no remote step and no prepared state |
| 11 | Runtime credential use for allocation preparation (`runtime_service.go:199`; `internal/scheduler/execution_preparation.go:337`) | an already pinned Runtime credential | M1 shared | waits: Stage preparation stalls for every Run that uses an OTLP, proxy, Caido, Interactsh or Project-origin credential | unaffected | unaffected | no Server | as row 10 |
| 12 | LLM token decryption for Worker settings and the Planner (`execution_preparation.go:158`, `:405`) | an already counted reference | M4 only | unaffected | C refused | unaffected | no Server | S1: a counted reference blocks every intent |
| 13 | Managed credential create (`lifecycle.go:406`) | — | M2, M3 | waits for M2 | 503 (M3) | 503 (M3) | no Server | — |
| 14 | Managed credential delete of another credential, or of C under another key (`lifecycle.go:478`) | — | M2, M1 exclusive, M3, M5 exclusive | waits | 503 (M3) | 503 (M3) | no Server | — |
| 15 | Runtime credential create and delete (`runtime_service.go:72`, `:260`) | — | M1 exclusive | waits | unaffected | unaffected | no Server | one local transaction |
| 16 | Managed credential list and read (`lifecycle.go:190-204`) | — | M7 | unaffected | unaffected | unaffected | no Server | read only |

M6 fences only rows 1-5. A managed delete of C also stalls rows 6-8, 10, 11
and 15, which touch only Runtime credentials (rows 10, 11 and 15) or may touch
only other LLM credentials (rows 6-8).

## 5. Current outcomes

### 5.1 Work done under the barrier

A managed delete waits for M2 without a context check, then for the exclusive
side of M1 (`sync.RWMutex.Lock`, context checked only before and after). From
the moment it asks, `sync.RWMutex` refuses every new reader: one slow reference
already inside M1 ahead of the delete stops every later reference of every
credential until both finish.

Under M1 the delete runs four reads (journal singleton, idempotency record,
credential record, at most 128 non-terminal Runs) and the Audit-hold read (at
most 128), the intent transaction (M5 exclusive, record recheck, prepared check,
at most 128 bindings, insert, COMMIT), three reference rechecks, one Gateway
`POST /key/delete` and the completion transaction. Measured in
`TestCredentialDeleteWorkUnderTheBarrierDoesNotGrowWithHistory`: **18
PostgreSQL statements and one Gateway request** in a fixture without an
Audit-hold reader (production adds the two hold reads), unchanged with 5,000
finished journal operations and 200 non-terminal Runs of another credential.

Budgets, none of which bounds the barrier as a whole:

| Budget | Default | Maximum | Owner |
| --- | --- | --- | --- |
| Gateway connect | 3 s | 1 min | `spec.credentialManagement.connectTimeout` (`internal/app/operational_settings.go:159-162`, `:271-273`) |
| Gateway request | 15 s | 2 min | `spec.credentialManagement.requestTimeout`, the `http.Client` timeout (`internal/credentials/litellm/manager.go:84-105`) |
| Statement / query / lock wait | 15 s / 20 s / 2 s | operator setting | database budgets (`operational_settings.go:150-154`) |
| Wait for M1 or M2 | none | none | `sync.RWMutex`, `sync.Mutex`; a waiting request cannot be cancelled |

The public HTTP server sets no request or write timeout
(`internal/app/system.go:59-62`), so a reference request waits as long as the
delete ahead of it.

A recovery step is bounded the same way. `Recover` reads at most 200 prepared
operations per batch (`maximumPreparedOperationBatch`,
`internal/credentials/repository.go:28`), although M3 keeps one at most in
practice, and completes them one at a time under M2, so no two Gateway
requests run concurrently. A prepared delete costs the reads above (each
reference list capped at 128 rows, `lifecycle.go:28`) and one Gateway request;
a prepared create costs up to two (remove the deterministic alias, then
generate). Each request body is at most 64 KiB and each response at most
64 KiB by default and 1 MiB at most (`litellm/manager.go:33-35`). The first
failed step ends recovery without a retry.

### 5.2 Outcomes of one managed delete

| Outcome | Result today | Fence afterwards | Completed by |
| --- | --- | --- | --- |
| Success | 204; row deleted; the completed operation is the tombstone | none | — |
| Referenced | 409 `credential_in_use` with at most 128 Run, Audit and label IDs; no side effect | none | — |
| Invalid request, missing master key, unknown credential, changed record, reused key | 400, 503 `credential_key_unavailable` (not retryable), 404, 409; no intent | none | — |
| Gateway without a usable credential manager | 502 `gateway_unavailable` from inside the intent transaction, before the insert (`lifecycle.go:594-596`, `:820-832`) | none | — |
| Gateway error, timeout or unreadable reply after a durable intent | 502 `gateway_unavailable` (retryable); intent prepared | M4 for C; M3 for every managed create and delete | the same key, or startup recovery |
| Intent INSERT or COMMIT without a definite outcome | storage error; M6 raised | M6 for rows 1-5; M3 and M6 refuse other deletes | reconciliation clears M6 when no prepared delete remains; a durable intent waits for the same key or startup recovery (D2, D3) |
| Process stops after the Gateway deleted the key, before the completion commit | intent prepared | at the next start | startup recovery (`already absent` counts as success) |
| Caller cancelled while waiting for M2 or M1 | waits for the holder, then returns its cancellation with no side effect | none | — |
| Caller cancelled during the intent transaction | possibly ambiguous, as the INSERT/COMMIT row | M6 | as above |
| Caller cancelled during the Gateway request | the HTTP client's cancellation is reported as `gateway_unavailable` (`lifecycle.go:785-790`); intent prepared | M4, M3 | the same key, or startup recovery |
| Shutdown | the public server waits for the handler up to `shutdownTimeout`, then closes its connection; nothing joins the delete; a completion commit that fails leaves the intent prepared | M4, M3 at the next start | startup recovery |

### 5.3 Ordinary Gateway outage and outage with a prepared operation

- **No prepared operation.** Startup recovery makes no Gateway call and the
  Server starts (`TestCredentialStartupRecoveryNeedsTheGatewayOnlyForAPreparedOperation`).
  During the outage a managed create or delete answers `gateway_unavailable`
  and, once its intent is recorded, leaves a prepared operation: a create's
  validation is local, so a create that fails at `/key/generate` is prepared
  too.
- **A prepared create or delete.** At the next start, the first failed Gateway
  call ends `Recover`, `configureCredentials` returns the error and the process
  exits before it serves anything. The unready Service would refuse every M6
  guard and every managed credential read with `credential_recovery_required`,
  but it is never served. The whole public API and the UI stay unavailable
  until the Gateway management API answers and the Server is restarted. A prepared
  create has no credential row, so no reference to it can exist; a prepared
  delete fences C through M4.

## 6. Verification findings

1. **Claims rechecked.** The exclusive hold through `manager.Delete` is at
   `lifecycle.go:491-494` and `:785` (the review cited `:326-329` and `:611-615`
   on `dc7012e8`); the startup stop is at `composition_credentials.go:126-127`.
   Creation takes M2 but not M1. No remote rotate path exists.
2. **RCF-R52 named Agent-label mutation as a barrier holder; the code never
   took the barrier there.** It needs none (row 9). This change corrects the
   holder list in RCF-R52, its definition table and the glossary; behaviour is
   unchanged.
3. **M6 fences only rows 1-5.** RuntimeConfig publication and binding, Project
   targets and Runtime credential use take M1 directly and stay available while
   M6 is raised (`TestCredentialDurableIntentFoundByReconciliationKeepsTheProcessWideFenceUntilRestart`).
   They are safe without M6: Runtime credentials have no remote step, and LLM
   references in rows 6-7 go through M5 and a transaction lookup, which read
   the true outcome of an intent transaction once it ends. This database-level
   proof already exists and is what D1-C generalizes.
4. **A durable prepared operation refuses every managed create and delete
   (M3)**, including a delete of the same credential under another key, with
   or without M6 (`TestCredentialPreparedDeleteFencesOwnReferencesAndEveryManagedMutation`).
   "A Gateway failure fences only its own credential" holds for references,
   not for credential mutations.
5. **The same key is rarely available again.** The UI keeps a delete's
   Idempotency-Key only in the credential page's component state
   (`ui/src/routes/operations/credentials/detail.tsx:51-59`); a reload, another
   tab or another operator gets a new key, and the CLI has no credential
   delete. After a lost COMMIT reply, a Server restart is in practice the only
   way to complete the deletion and lower M6.
6. **No signal for M6.** Readiness checks only the database
   (`internal/app/performance.go:35-50`); Operations does not show the fence.
   It is visible only as 503 `credential_recovery_required` answers and as one
   Scheduler error log per lane and poll.
7. **Startup recovery trusts its journal read.** A fresh process has no M6, so
   `Recover` does not reconcile an intent transaction of a previous owner that
   is still committing on a connection that outlived the lease handover. If
   such an intent committed after the new owner admitted a Run of C,
   `executeDelete` would answer `credential_in_use` and every later start would
   fail until that Run ends, which needs a running Server. Reproduced by
   V414-010 with a COMMIT held across a server-side end of the previous
   owner's lease session, and fixed there: managed credential writes are
   lease-fenced and startup recovery reads the journal only after the lease
   handover ([PER-R34](../spec/platform/persistence.md),
   [RCF-R72](../spec/configuration/runtime-config.md)). A reference of the
   previous owner in flight across the handover remains an accepted risk
   ([PER-R36](../spec/platform/persistence.md)).
8. **FAIL-RCF-2 leaves the timeout outcome "not specified".** In code a Gateway
   timeout and a cancellation during the Gateway request both answer
   `gateway_unavailable` and keep the intent prepared. A task that changes
   RCF-R53 or RCF-R72 should state it there.

## 7. Alternatives

### D1. Barrier hold during remote deletion

**D1-A. Keep RCF-R52.** *Guarantee:* S1 and S2 hold structurally: any
reference path under M1 is safe whatever lookup it uses, including future
ones. *Availability:* every row marked "waits" in column (a) waits for one
managed delete: about 15 s with default settings, up to 2 min with maximal
ones, plus the longest reference already in flight ahead of the delete.
Deletes are operator actions and rare: a credential ID is deleted at most once.
*Evidence:* `TestCredentialDeleteBlocksReferencesAndMutationsOfOtherCredentialsAcrossItsRemoteCall`,
`TestCredentialDeleteWaitingForOneReferenceStopsEveryLaterReference`, section
5.1. *Detection:* none specific; request durations of `POST /v1/runs` and of
placement when performance collection is on. *Reopen when:* a delete is
observed to stall Run creation or placement longer than the default Gateway
request timeout; credential deletion becomes automated or frequent;
`credentialManagement.requestTimeout` is raised above its default; or D1-C is
chosen for another reason.

**D1-B. Release M1 once the intent commit is acknowledged.** Hold M1
exclusively through the reference checks and the intent transaction (or until
M6 is raised), release it before `executeDelete`, and keep M2 through the
completion commit. *Guarantee:* S1 unchanged (same exclusive window). S2 by
inventory: every path that commits an LLM reference reads C through M4 after
entering M1 (rows 1, 3-5), holds a counted reference that S1 already protects
(row 2), or uses M5 (rows 6-7). A reference that enters after the release
reads the acknowledged intent with a later READ COMMITTED snapshot and is
refused; one that entered before finished before the delete obtained M1. An
uncertain COMMIT raises M6 before the release, as today. *Availability:*
references and Runtime credential mutations wait for the reference checks and
the intent transaction (the first 11 of the 18 statements, 13 with the
Audit-hold reads; lock wait at most 2 s), never for the Gateway; managed creates and deletes still wait for M2. *Failure outcomes:*
section 5.2 unchanged, except that a stalled Gateway request no longer delays
other work. *Bounds:* unchanged statements, one Gateway request per delete, no
new state. *Lifecycle:* the delete still runs in its request; nothing new to
join. *Risk:* the proof is an inventory. A future reference path that skips M4
would break S2 silently, so this option needs a fitness test that enumerates
every caller of the guards and of `WithCredentialReferences`. *Reversal:*
restore the deferred unlock; no data change.

**D1-C. Make the database fence authoritative for LLM references.** Every
transaction that commits a reference to an LLM credential takes M5's shared
lock for each pinned credential in sorted order and reads its prepared state
in the same transaction, as rows 6-7 already do. The delete keeps M5 exclusive
in its intent transaction; M1 remains only for Runtime credentials.
*Guarantee:* S1 and S2 hold in PostgreSQL, across processes, and an ambiguous
COMMIT is resolved by the lock itself, so M6 becomes unnecessary for
references and D2 disappears. *Availability:* references to other credentials
never wait for a delete; a reference to C waits at most for C's intent
transaction (`lock_timeout`, 2 s); a lock timeout already maps to the
transient timeout class (`internal/persistence/postgres/sqlstate.go:124`), and
the task must decide its public answer. *Cost:* one advisory-lock statement per pinned LLM
credential per reference transaction; Run creation must move its Workflow
credential lookup, now pooled and before the Run transaction, into that
transaction. It reuses M5 and adds no mechanism (AGENTS 3.2). *Bounds:* at most
the number of distinct credentials a Run pins. *Reversal:* drop the shared
locks and restore M1 for LLM credentials. The largest change: `runservice`,
`auditservice`, `evalservice`, `controlplane` and RCF-R52/R53.

### D2. Fence after reconciliation finds a durable intent

**D2-A. Keep RCF-R53.** *Guarantee:* the conservative rule that predates
V407-019. *Availability:* until the same key retries or the Server restarts,
rows 1-5 are refused for every credential: no Run starts, no Audit starts,
resumes or dispatches, and no Stage of any in-flight Run is placed. There is no
time bound, and finding 5 makes the restart the usual exit. *Evidence:*
`TestCredentialDurableIntentFoundByReconciliationKeepsTheProcessWideFenceUntilRestart`
and V407-019's `TestCredentialDeleteIntentCommittedLateStaysFencedUntilItsDeletionCompletes`.
*Detection:* finding 6. *Reopen when:* the state is observed once, or D3 is
accepted.

**D2-B. Narrow to the credential.** When reconciliation takes M5's try-lock and
reads a prepared delete, clear M6 as it does when it reads none. *Guarantee:*
the resulting state is the one a Gateway failure after a durable intent
produces, which RCF-R53 already accepts ("A durable prepared intent rejects
only references to its own credential"). The try-lock proves that the intent
transaction ended; the read after it sees the commit; every later reference
reads C as prepared through M4 (rows 1, 3-5) or M5 (rows 6-7); row 2 is
covered by S1. *Evidence:* both routes leave the same journal row and the same
M4 refusal (`TestCredentialPreparedDeleteFencesOwnReferencesAndEveryManagedMutation`,
`TestCredentialDurableIntentFoundByReconciliationKeepsTheProcessWideFenceUntilRestart`).
*Availability:* other credentials are admitted after one reconciliation;
managed mutations stay refused by M3 until D3. *Failure outcomes:* a
reconciliation that cannot take the lock or cannot read keeps M6, as today; a
cancelled caller keeps its cancellation. *Bounds:* unchanged: one short
transaction per guard while M6 is raised, shared by concurrent guards.
*Lifecycle:* nothing new. *Change:* `settleUncertainDeleteLocked` clears on a
prepared read; `Recover`'s re-read after an empty journal (`lifecycle.go:347-374`)
adapts; RCF-R53, FAIL-RCF-2 and the `credential_recovery_required` text change.
*Reversal:* restore the condition.

**D2-C. Narrow and complete.** D2-B together with D3-B in one change.

### D3. Completing a prepared operation without the same key

**D3-A. Keep.** The same key or startup recovery. *Availability:* managed
creates and deletes are refused by M3 indefinitely; under D2-A rows 1-5 too.
*Evidence:* finding 5. *Reopen when:* D2-B is accepted (this is then the only
remaining stall) or a stuck operation is observed.

**D3-B. A delete of the same credential under any key completes its prepared
delete.** The new request executes the stored request (stored actor and
deletion time stay on the tombstone) and then answers as the completing
request. *Bounds:* M3 keeps at most one prepared operation, so each attempt
makes one Gateway request under M2; there is no retry loop. *Failure
outcomes:* a Gateway failure answers `gateway_unavailable` and keeps the
intent; a cancellation leaves it prepared; both as today's same-key retry.
*Idempotency:* the new key records nothing; the task must define the answer to
a later replay of that key, which finds the credential deleted. *Lifecycle:*
runs in the request; nothing outlives it. Covers the reload case of finding 5
with no new component. Does not cover a prepared create, whose credential ID
is reserved by the original create operation.

**D3-C. Any managed create or delete completes the prepared operation
first**, a create as startup recovery does (remove the alias, create again),
then continues with its own request. Bounds as D3-B. Covers creates as well,
but one operator's request completes another's operation and pays its Gateway
latency.

**D3-D. Background completer.** One runner owned by `ServeSystem` retries the
single prepared operation with bounded exponential backoff (for example 5 s
doubling to 5 min) under M2 while one exists, logs each failure, exposes the
pending state to Operations and is joined on shutdown (its request is bounded
by `requestTimeout`). Adds a lasting component, a limit entry and a signal; it
is the natural companion of D4-C.

**D3-E. Explicit Operations action** to complete the pending operation. Needs a
public endpoint, OpenAPI and UI; out of proportion for a rare state.

### D4. Startup with a prepared operation and no Gateway

**D4-A. Keep RCF-R72.** *Guarantee:* no traffic before every prepared
operation completes. *Availability:* a Gateway management outage at a restart,
while a prepared create or delete exists, keeps the whole Server down; an
ordinary outage does not (`TestCredentialStartupRecoveryNeedsTheGatewayOnlyForAPreparedOperation`).
*Detection:* the startup error `recover LLM credential operations` and the
process exit. *Reopen when:* such a startup failure is observed, the Gateway
management API is deployed with lower availability than the Server, or D2-B
and D3 make a running Server tolerate the same state.

**D4-B. Bounded startup retry** of `Recover` within a startup budget. Keeps
RCF-R72 and absorbs short outages, but only delays the same failure and holds
the lease meanwhile.

**D4-C. Restricted start.** When recovery fails at the Gateway, mark the
Service ready with the prepared operation still in the journal. The Server then
starts in the state a running Server reaches after a Gateway failure:
references to the affected credential are refused (M4; a prepared create has
no row to reference), managed creates and deletes are refused (M3), everything
else serves, and a completer (D3-B or D3-D) finishes the operation. RCF-R72
changes from "before Run or Operations traffic" to "before traffic that can
reference the affected credential". *Safety:* as D2-B; no new state.
*Needs:* D2-B first (otherwise a fresh start would be more permissive than a
running Server with M6 raised), and a signal for the pending operation.

## 8. Recommendation

Labels: **E** evidence from code or tests at `925a614c`, **A** assumption.

1. **D2: accept D2-B in a separate feature task.** E: the durable state equals
   the Gateway-failure state that RCF-R53 already narrows; today's fence
   refuses all execution with no time bound and no signal (findings 5, 6). A:
   the inventory of section 4 is complete; the task must add a test that
   enumerates the guard callers so that a new reference path fails the build.
2. **D3: accept D3-B together with D2-B (D2-C).** E: finding 5; M3 bounds the
   work to one Gateway request per attempt. A: prepared creates are rarer and
   harmless to references (no row exists), so D3-A stays for creates; reopen
   D3-C or D3-D if a stuck prepared create is observed.
3. **D1: keep D1-A now.** E: the stall is bounded by the Gateway request budget
   and occurs only during an operator's delete. D1-B would replace a structural
   guarantee by an inventory. A: no deployment raises `requestTimeout` or
   automates deletion. If D1 is reopened, prefer D1-C, which also removes M6,
   over D1-B.
4. **D4: keep D4-A now.** E: it needs a restart during a Gateway management
   outage while a prepared operation exists. Reopen after D2-B and D3 land if
   such a startup failure is observed; D4-C is then a small step consistent
   with the running Server. D4-B is not recommended: it delays the failure
   without removing its cause.

RCF-R52 (with its holder list corrected), RCF-R53 and RCF-R72 stay in force.
Accepting any option needs a feature task that changes the owning
requirements and FAIL-RCF-1/2 in the same commit.

## 9. Decision record

| Item | Content |
| --- | --- |
| Decision owner | The project owner (the user) accepts or declines each of D1-D4. Owning area: `runtime-config`; `execution-core` for placement and Scheduler effects; `operations` for detection |
| Unresolved product decision | Whether to trade the conservative fences of RCF-R53 (D2) and RCF-R72 (D4) for availability, and whether a managed mutation may complete another request's prepared operation (D3) |
| Guarantee kept by every option | S1, S2, S3 |
| Alternatives | D1-A to D1-C, D2-A to D2-C, D3-A to D3-E, D4-A to D4-C |
| Evidence | Sections 3-6 and the tests of section 11, at `925a614c` |
| Reversal conditions | Per option in section 7; section 10 for the current policy |

## 10. Reopening the current policy

Reopen with the owner above when any of these holds:

1. `credential_recovery_required` answers or Scheduler placement errors persist
   after a credential delete (D2, D3).
2. A Server start fails with `recover LLM credential operations` during a
   Gateway management outage (D4).
3. A credential delete stalls Run creation, Audit start or placement for longer
   than the default Gateway request timeout, or `requestTimeout` is raised (D1).
4. Credential deletion becomes automated, frequent, or performed by more than
   one Server (D1, D2; the single-active-Server assumption of M1, M2 and M6).
5. A new path commits a credential reference outside rows 1-15 (every option).

## 11. Characterization and what a change must preserve

Real-store tests (PostgreSQL, synchronization by channels and lock probes, no
elapsed-time assertions) in `internal/credentials`:

| Test | Shows |
| --- | --- |
| `TestCredentialDeleteBlocksReferencesAndMutationsOfOtherCredentialsAcrossItsRemoteCall` | While the Gateway request is in flight, M1 and M2 refuse entry; Run creation and allocation of another LLM credential, RuntimeConfig/Project and Runtime credential references, Runtime credential creation and managed creation all pass only after the request returns |
| `TestCredentialDeleteWaitingForOneReferenceStopsEveryLaterReference` | A delete waiting behind one reference stops every later reference until the delete, Gateway request included, finishes |
| `TestCredentialPreparedDeleteFencesOwnReferencesAndEveryManagedMutation` | After a Gateway failure only C's references are refused, but every managed create and delete is (M3); Runtime credentials are unaffected; startup recovery reopens mutations |
| `TestCredentialDurableIntentFoundByReconciliationKeepsTheProcessWideFenceUntilRestart` | A durable intent found by reconciliation keeps rows 1-5 closed for every credential across repeated reconciliations, leaves rows 6-11 open, refuses every managed mutation, and is lowered by a restarted Server's recovery |
| `TestCredentialDeleteWorkUnderTheBarrierDoesNotGrowWithHistory` | 18 statements and one Gateway request under M1, independent of journal and Run history |
| `TestCredentialStartupRecoveryNeedsTheGatewayOnlyForAPreparedOperation` | An ordinary outage does not affect startup; a prepared delete makes recovery fail and the Service unready until the Gateway answers |

Existing tests that cover the same contract: V407-019's
`lifecycle_delete_fence_postgres_test.go`, and
`TestCredentialFailedDeleteFencesOnlyItsOwnCredential`,
`TestCredentialDeleteSerializesAgainstRunSnapshotCommit` and
`TestCredentialLifecycleRecoversDeleteCrashAndBlocksRunCreation` in
`lifecycle_postgres_test.go`.

An accepted change keeps every safety assertion of these tests (a credential
with a prepared delete is refused through both lookups; a reference committed
before an intent makes the delete answer `credential_in_use`; an intent whose
COMMIT may still land is never passed; recovery completes a prepared delete)
and rewrites only the availability assertions it changes: D1 the first two
tests, D2 the fourth test and V407-019's late-commit test, D3 the
different-key refusals, D4 the unready refusal of the last test. It also adds
an operation-matrix test that drives every row of section 4 against a delete in
flight, a durable prepared delete and an uncertain intent, and asserts S2 for C
and the intended availability for other credentials.
