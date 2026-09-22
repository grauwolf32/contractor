# Deterministic release gates

[Testing overview](README.md) · [Development setup](../development.md)

The commands below document focused verification and failure diagnosis.
Set up dependencies and a disposable PostgreSQL database as described in the
testing overview. The exact aggregate target is maintained in the
[Makefile](../../Makefile); task files record completed runs of these checks.

- [Hardening gates](#hardening-gates)
- [Runtime-configuration hardening and release gate](#runtime-configuration-hardening-and-release-gate)
- [WorkflowRun metadata-label release gate](#workflowrun-metadata-label-release-gate)
- [Complete browser stack gate](#complete-browser-stack-gate)
- [Run and Project lifecycle release gate](#run-and-project-lifecycle-release-gate)
- [Workflow Scheduler concurrency release gate](#workflow-scheduler-concurrency-release-gate)
- [Audit completion release gate](#audit-completion-release-gate)
- [Shared MemoryTools release boundary](#shared-memorytools-release-boundary)
- [HTTP/Caido hardening boundary](#httpcaido-hardening-boundary)
- [Streamline Planner evidence](#streamline-planner-evidence)
- [Automated MVP evidence](#automated-mvp-evidence)

## Audit completion release gate

Prepare the pinned Runtime environment with `cd runtime && uv sync --locked`,
then set `CONTRACTOR_TEST_DATABASE_URL` to a disposable PostgreSQL instance and run:

```shell
make test-audit-completion-e2e
```

The database role must be allowed to create and drop test databases. The gate
creates randomly named databases, applies production migrations, and removes
them after testing. Do not point it at the demo database. No live provider,
demo deployment, or immutable catalog update is involved.

This target is included in `make release-verify`. It requires Go, uv and the
prepared Runtime environment. Its fixed
[required-case matrix](../../scripts/audit-completion-matrix.json) is checked
against Go JSON events and pytest JUnit results: missing tests, insufficient
parameter coverage, failures and **any selected skip** fail the gate. The three
PostgreSQL authority/recovery tests and the Runtime/importer bridge must run.
Plain `go test ./...` and `make verify` can skip database tests and cannot
substitute for this gate. The standalone Python bridge test also skips outside
its Go parent; inside the gate a missing child result is a failure.

`TestAuditCompletionRuntimeZIPImporter` creates a trusted two-item Audit Run
through `runservice.CreateAudit`, passes its exact completion contract to the
pinned ADK Runner in a separate Python process, and serves artifacts through
the production mTLS Artifact API backed by PostgreSQL. The actual Runtime ZIP
is bound and frozen as a Workflow output, then collected by the production Go
importer into durable coverage and one batch receipt. The external model and
Scheduler driving are deterministic test orchestration; this is not a live
provider benchmark or a full Server/Planner/A2A deployment test.

Bridge cases cover reversed separate submissions, correction after missing
evidence, no calls, invalid/partial collection, dropped write reply, cancellation
before/after write, deliberately injected invalid packages, abrupt post-write
process loss, same-Run recognition/conflict (including proposal invocation IDs),
and a new child Run in the same Audit. The component suites additionally cover
parallel/revisioned submissions, batch atomicity, shared fixtures and sessions,
standard evidence rules, escalation, spoofing, unsupported Runtimes, rollback,
restart, hard budgets, real-Runner continuations, and ordinary/legacy completion.

Each invocation stores sanitized logs, `go.jsonl`, `runtime.xml`, and the exact
executed test names in `executed.json` under a fresh directory in
`.local/audit-completion-gate/`. `executed.json` is written only after every
check passes; old reports cannot make a later failed invocation pass. Use
`python3 scripts/test-audit-completion-e2e.py --output-dir <directory>` to retain
these artifacts elsewhere. [V39-007](../../tasks/v39/v39-007-audit-completion-release-gate.yml)
records the implementation commit and verification evidence.

## Hardening gates

The executable fault inventory is
[`tests/faults/matrix.yml`](../../tests/faults/matrix.yml). It documents every
mutating first-slice operation's response-loss semantics and maps lifecycle
crashes, retries, races, stale-allocation attacks, mTLS failures, tampering,
redaction, and resource shutdown to concrete tests. Its schema test also fails
if a referenced test is renamed or removed.

Run the deterministic local gates with the same PostgreSQL prerequisite:

```shell
make verify
go test -race ./...
CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
  make test-postgres test-streamline test-faults test-e2e
make test-control-integration
make test-artifact-integration
```

`make test-faults` runs the relevant Go packages under the race detector and
the complete Runtime Agent suite as `pytest -W error`. Only precise warnings
emitted by the pinned ADK/A2A/Starlette dependencies are quarantined; every
other warning fails the suite. The Runtime shutdown test also asserts that no
asyncio task remains pending after an in-flight heartbeat is cancelled.

All API responses carry a bounded `X-Request-ID`. REST error bodies repeat it
as `requestId`; A2A responses carry it in the HTTP header. Public requests never
get to choose this value. Trusted private hops propagate one valid incoming
value, while malformed or repeated values are replaced. Server-side 5xx logs
record the request ID, boundary, method, status, and safe error type without
raw request paths, bodies, artifact bytes, or exception messages.

## Runtime-configuration hardening and release gate

The complete deterministic gate is:

```shell
CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
  make test-runtime-configuration-e2e
```

It first runs the strict executable matrix in
`tests/e2e/runtime_configuration_matrix.yml`, all relevant Go packages under
the race detector and the Python adapter/lease failure suites. It then runs the
two-certificate process scenario and the real Chromium stack. `make
release-verify` prepends the ordinary Go/Python/UI verification and is the CI
release command. `make verify` remains database-independent and validates the
matrix references, so removing or renaming an owning hardening test cannot
silently narrow the release gate.

The process scenario uses production RuntimeConfig resolution, placement,
Scheduler, Planner, mTLS, Artifact API and adapter host code. Only the external
LLM, OTLP collector and HTTP proxy endpoints are deterministic local fixtures.
It deliberately gives one Runtime only `otlp-http@1` and the other only
`http-proxy@1`, so the final successful probe on each adapter proves both slots
were actually released after the injected telemetry and proxy failures.

If this gate fails:

- an HTTP `409` from the deliberate duplicate-certificate registration is
  expected; a successful registration is an identity regression;
- a proxy-failure Run must be `failed`, while an OTLP-failure Run must still be
  `succeeded`; reversing either outcome is a policy regression;
- a Run stuck in `preparing` usually means the advertised adapter sets no
  longer match the test's intentionally disjoint Runtime processes;
- a terminal Run with a non-idle Agent indicates release reconciliation, not a
  reason to increase an unbounded timeout; inspect the final Operations
  snapshot and the bounded child-process diagnostics emitted by the test;
- PostgreSQL must allow isolated schema create/drop, and Chromium plus
  `runtime/.venv` must already be installable as described in the [development setup](../development.md).

Accepted first-slice boundaries remain explicit: there is one active Control
Plane, no Vault/KMS or immutable RuntimeConfig garbage collection, no dynamic
Runtime capability changes, no content-bearing telemetry and no dependency on a real LM Studio, Langfuse, Caido or cloud
service. Performance measurements have their own [guide](../operations/performance.md)
and release target.

## WorkflowRun metadata-label release gate

Run metadata labels have their own strict, executable ownership matrix:

```shell
make test-run-metadata-labels-matrix
```

The matrix maps every accepted contract boundary to a named Go, Python or
TypeScript test: public validation, atomic immutable storage, canonical
idempotency, indexed owner-scoped queries, private AllocationSpec validation,
placement/configuration independence, Planner and Worker trace projection,
cancellation/release and the standalone UI. Renaming or deleting an owner
therefore fails `make verify`; the YAML is not a prose-only checklist.

The focused deterministic language-level gate is:

```shell
make test-run-metadata-labels-hardening
```

To cross real process and trust boundaries, provide PostgreSQL and run:

```shell
CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
  make test-run-metadata-labels-e2e
```

This starts the production Go Server and Scheduler, a certificate-distinct
Python Runtime Agent, an isolated migrated schema, and deterministic external
model/OTLP fixtures. It executes one unlabeled baseline, two ordinary A/B Runs
with a shared `eval.id`, and a final reuse Run over the same exact source
revision. The scenario verifies exact owner/state/cursor filters, response-loss
replay, a value containing `=`, independent `debug` Runtime configuration,
trace-root attributes, failed-export neutrality, artifact/result provenance,
foreign-owner isolation, retained-surface exclusion and cleanup of the single
Runtime slot.

Label values are expected to be visible in Run APIs and execution-root
telemetry. The test canaries prove absence from model requests, Worker State,
artifacts, reports, Planner events, child spans, process Resource attributes
and process logs; they do not turn labels into a supported secret store. Run
credentials and source fragments must still never be supplied as labels.

`make test-e2e` includes this process scenario, and `make release-verify`
combines it with the existing real-browser stack. The process test requires the
locked `runtime/.venv`; its target runs `uv sync --locked` first. Only the
external LLM-compatible and OTLP endpoints are fakes—Run creation, PostgreSQL,
mTLS, scheduling, allocation, Artifact API and Runtime lifecycle are production
implementations.

## Complete browser stack gate

`make test-ui-stack` builds and starts the production Go Server, production
Node static service and production Python Runtime Agent entry points. It uses
an isolated PostgreSQL schema, temporary deployment CA, deterministic
OpenAI-compatible model fixture and deterministic LiteLLM management fixture.
The browser connects directly to the Go API; Node is started from a clean
environment that contains no Server bearer token, model token, credential
manager key, password or CSRF value.

The harness exposes local `https://ui.contractor.test:<port>` and
`https://api.contractor.test:<port>` origins through temporary TLS reverse
proxies. Chromium receives an explicit process-local mapping of both reserved
`.test` hosts to loopback, so no host configuration or external DNS is used.
They are same-site for the Server's `SameSite=Lax` session cookie but have
different host-only cookie scopes. No paid model or container runtime is used.
PostgreSQL is caller-owned and must permit creation and deletion of an isolated
schema.

The Chromium flow covers login, reload recovery and logout; exact-origin CORS,
preflight and CSRF rejection without side effects; Artifact upload/download;
live Streamline planning, WebSocket interruption/reconnect and metrics;
four-stage OpenAPI execution and frozen output download; immutable
ModelPolicy publication; fake-manager credential creation/deletion; independent
Node restart; Server restart, session loss and Operations generation change;
and incompatible public-API configuration. It scans bounded HTTP bodies,
WebSocket frames, browser storage, DOM text, process logs, database-safe
columns, Playwright trace and screenshot bytes for seeded secret canaries.
Screenshot OCR is additionally used when a working Tesseract language pack is
installed.

The first run downloads the pinned Chromium build. If browser startup fails,
rerun `make ui-browser-install` and install the host libraries reported by
Playwright. A failure includes bounded, canary-redacted child-process output;
the most common setup failures are an unavailable PostgreSQL URL, missing
`runtime/.venv` (fixed by `uv sync --locked`), or missing Chromium. The test
always releases held model calls, terminates child processes, removes temporary
keys/certificates/workspaces and drops its isolated schema.

To verify independent releases outside the harness, keep Server running while
rebuilding/restarting only the service documented in
[`deploy/ui/README.md`](../../deploy/ui/README.md). A compatible UI reload must
retain the Server session and any active Run; restarting Server intentionally
invalidates the in-memory browser session and changes the Operations event
generation.

## Run and Project lifecycle release gate

The executable ownership and fault map is
[`tests/e2e/lifecycle_controls_matrix.yml`](../../tests/e2e/lifecycle_controls_matrix.yml).
It binds artifact-list exclusion, queue pause serialization, Stage drain,
terminal release, Run purge, shared-content retention, durable Project
deletion, Server restart and browser reconciliation to named tests. Keep every
new lifecycle failure mode in this matrix; prose or an unreferenced test is not
release evidence.

Run the deterministic database and component gate with a disposable PostgreSQL
database. It executes production repositories and controllers under the race
detector and never calls a test-only deletion path:

```shell
CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
  make test-lifecycle-controls-hardening
```

The process gate restarts the Go Server around real PostgreSQL and two distinct
Python specialist Runtime Agents. The browser gate builds and serves the Node
UI separately from the Go API, then proves pause/resume, eligible Run trash and
typed Project deletion through authoritative polling without a manual reload.
Run the complete boundary before releasing lifecycle changes:

```shell
CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
  make test-lifecycle-controls-release
```

Operationally, `paused` means only that new Stage admission is fenced. Already
admitted work drains and cancellation remains available. A terminal Run is
deletable only after every allocation is durably released. Project deletion is
asynchronous: the UI may be closed while the controller cancels active Runs,
waits for release, purges Run state, then removes ProjectScope bindings. A
Server restart resumes the stored phase. Do not retry by bypassing the phase or
manually deleting rows; inspect the Project deletion phase and unresolved
allocation release first.

## Workflow Scheduler concurrency release gate

The executable policy and fault map is
[`tests/e2e/scheduler_concurrency_matrix.yml`](../../tests/e2e/scheduler_concurrency_matrix.yml).
It binds the seeded serial default, PostgreSQL CAS/restart, lane claims,
dynamic resize drain, Runtime capacity, deferred-owner rotation, lifecycle
maintenance and the Operations browser flow to named tests. Keep new
concurrency failure modes in this matrix so the release evidence cannot drift
into prose-only claims.

Run the complete boundary with a disposable PostgreSQL database:

```shell
CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
  make test-scheduler-concurrency-e2e
```

The process test starts the production Go Server, two certificate-distinct
single-slot Python Runtime Agents and a deterministic OpenAI-compatible
Gateway. Before mutation it proves the migration default admits only one Run.
After the Operations API changes the setting to two, two one-Worker Runs block
inside separate agents while a third has no allocation or Planner invocation;
all three then finish, release their claims and allocations, and a restarted
Server reads the same durable setting. The multi-Worker production fixture
separately proves that one Router Stage may consume both Runtime slots while
still counting as one Run lane.

For the local demo, keep the product migration unchanged and explicitly select
`2` from **Operations → Settings → Workflow scheduling** after the Server is
ready. Lowering the value is a drain: it does not terminate work already
admitted. Owner **Queue Pause** remains a separate admission control, and two
configured lanes are only an upper bound when fewer compatible Runtime slots
are idle.

Public Run creation requires exactly one `Idempotency-Key` of 1–128 safe ASCII
characters. Retrying the same semantic request with the same authenticated
user/key returns the original Run and `Idempotency-Replayed: true`; reusing the
key for different content returns `409 Conflict`. Artifact PUT remains CAS:
after a lost successful response, repeating the consumed `If-None-Match` or
`If-Match` precondition returns conflict and creates no second revision; use a
GET to reconcile the accepted exact revision.

## Shared MemoryTools release boundary

`memory-tools@1` stores each logical note as one ordinary canonical
`memory.<name>` RunScope Artifact. There is no Memory HTTP API, SQL table,
credential, replay record or cross-Run store. The only intended durable payload
copy is `artifact_blobs.payload`; owner/operator Artifact inspection can read it
under the existing Run authorization contract.

The `v1` quota and ordinal allocator rely on serialized execution: one Planner
function call, at most one active Worker dispatch and no concurrent Stages in a
Run. Enabling concurrent different-name creates in one Memory Namespace first
requires a generic ArtifactStore atomic batch/CAS primitive.

The redaction guarantee applies to copies made automatically by Memory
adapters: reports, Planner durable facts, events, HTTP failures, process logs and
decoded OTLP retain safe operation/name/size/outcome dimensions but not note
content, description or tags. A selected LLM Gateway necessarily receives note
data returned to the model, and a model can explicitly author that data into a
different semantic result. Preventing such an intentional copy would require
information-flow tracking and is outside `v1`.

The matrix and contract checks are database-independent:

```shell
make test-shared-memory-matrix
make test-memory-contracts
```

The fault and process gates require an explicit PostgreSQL test URL. The
hardening target composes the matrix, shared codec, race/fault tests and the
two-Runtime process scenario; the aggregate release target also runs the
Runtime-configuration and browser gates.

```shell
export CONTRACTOR_TEST_DATABASE_URL='postgres://postgres:contractor@127.0.0.1:5432/contractor?sslmode=disable'
make test-shared-memory-hardening
make release-verify
```

## HTTP/Caido hardening boundary

The generic HTTP Toolset's private-origin checks are not a DNS or network
sandbox. For untrusted destinations, configure the allocation's mandatory
`tool-http` proxy route and enforce DNS/address policy at that proxy or in the
Runtime network namespace. Caido control access is separate: it uses the typed
`caido-graphql@1` adapter and reviewed static GraphQL documents. Runtime does
not introspect a Caido installation or negotiate a schema version, so an
upgrade must pass the checked-in compatibility fixtures before deployment.

The matrix and focused Runtime suite are deterministic and need no external
Caido, target service or PostgreSQL:

```shell
make test-http-caido-matrix
make test-http-caido-runtime
make test-http-caido-architecture
```

The complete gate adds PostgreSQL, mTLS, two real Python Runtime processes and
local fake target/proxy/Caido/Gateway services. It requires only the same test
database URL as other process gates:

```shell
export CONTRACTOR_TEST_DATABASE_URL='postgres://postgres:contractor@127.0.0.1:5432/contractor?sslmode=disable'
make test-http-caido-hardening
```

No external LM Studio, Caido instance, internet target or cloud credential is
part of this gate. The fake services inject response loss, schema errors,
oversized bodies and secret canaries deterministically.

## Streamline Planner evidence

`make test-streamline` exercises the real Google ADK event loop through a
deterministic OpenAI-compatible Gateway, one fixed fake Worker, deterministic
full Stage-context propagation, succeeded/failed `finish` candidates,
Scheduler-owned transition selection, ordered subtask IDs with stale-dispatch
rejection, budget termination, secret redaction, and
PostgreSQL completed-session recovery. The example Workflow is
[`streamline_review_workflow.yaml`](../../configs/examples/streamline_review_workflow.yaml).

`go test -race ./internal/planner/router/...` exercises Router through the same
real ADK loop. It verifies the exact two-argument execute schema and lexical
Available-agents prompt, rejects unknown, stale, and parallel selections before
A2A, and proves that model requests and reports contain logical binding names
but no allocation IDs, Runtime URLs, or credentials. The strict loadable
example is
[`router_openapi_workflow.yaml`](../../configs/examples/router_openapi_workflow.yaml).

`make test-e2e` additionally starts two identical single-slot Runtime Agent
processes and runs three workflows in sequence. `streamline-copy@1` selects its
sole builder, `router-review@1` prepares builder and reviewer but invokes only
reviewer, and `escalating-copy@1` first records a non-retryable failed candidate
before Scheduler creates one linked `failed_escalation` execution with the
exact `strong_planner@1` and `strong_worker@1` settings. The harness checks the
PostgreSQL event journal and reports as well as the public result. Physical
allocation IDs, instance IDs, Runtime URLs, model content, and credentials are
not accepted in the UI-safe plan events.


## Automated MVP evidence

The harness allocates loopback ports dynamically and removes its schema,
certificates, process state, and work directories on exit. PostgreSQL itself is
supplied by the caller; the test never starts Docker or silently substitutes a
different repository. Recent child-process logs are bounded and known test
tokens are redacted from failure output.

The scenario uploads `contractor-e2e-input\n`, executes
`artifact-copy@1`, and verifies all of the following:

- the public input is forked into Run scope without changing User scope;
- one `passthrough@1` Planner invocation reaches one Python `adk@1` Worker over
  A2A JSON-RPC 1.0 and mTLS;
- only `read_artifact` and `write_artifact` are visible to the model;
- the exact Worker result is linked to a distinct frozen Workflow output;
- the bounded Worker `ExecutionReport` contains model, token and tool
  aggregates plus bounded diagnostic records;
- the Planner, Worker and Runtime reports form durable `StageMetrics`, while
  public Run status exposes only the safe aggregate summary;
- release removes the allocation route, in-memory Worker state and local
  workspace;
- private listeners reject a client that trusts the CA but has no certificate.
