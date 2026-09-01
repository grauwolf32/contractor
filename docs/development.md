# Local development

## Implementation checkpoint

As of 2026-09-01, implementation tasks through `V8-016` are complete. The
repository contains the runnable Go Server/Python Runtime Agent MVP plus:

- durable Run cancellation and bounded `aborting` cleanup;
- symmetric 10-second heartbeat / 60-second confirmed allocation leases,
  Runtime self-fencing, and two-phase release reconciliation;
- validated serial multi-Stage Workflow graphs and explicit bounded retry;
- one immutable PostgreSQL transition decision per completed StageExecution,
  atomically committed with either the next/retry execution or terminal Run;
- fresh exact StageContext artifact pins for every retry attempt;
- bounded Python Worker and Go Planner reports with independent Runtime facts;
- immutable PostgreSQL telemetry, idempotent StageMetrics aggregation,
  terminal-only 30-day retention cleanup, and a safe public summary;
- response-loss-safe public Run creation through a required `Idempotency-Key`;
- exact candidate-revision output acceptance under concurrent binding changes;
- bounded correlation IDs and redacted diagnostics on public and private HTTP
  boundaries;
- an executable lifecycle crash/retry/race/security matrix under Go's race
  detector and Python's strict warning/task-shutdown gate;
- a model-backed, single-Worker `streamline@1` Planner built with Google ADK
  Go, one `execute_current_subtask(subtask_id)` adapter that supplies immutable
  Stage context, a bounded typed subtask plan with immutable dispatch IDs, one
  `finish` operation for successful or failed candidates, and bounded model,
  token, Worker-call, and wall-time budgets;
- a model-backed `router@1` Planner that reuses the same bounded execution
  engine and exposes only
  `execute_current_subtask(subtask_id, worker_name)`, with a schema constrained
  to the immutable logical Stage bindings and no physical placement data;
- strict immutable `LLMGatewayConfig` manifests with normalized cross-language
  digests, non-secret resolved wire values, and a loopback LiteLLM example;
- a PostgreSQL-backed Planner session adapter that persists only redacted ADK
  facts and recovers a completed decision without repeating model or Worker
  calls;
- bounded UTF-8 Run-artifact tools and safe ZIP source exploration;
- namespace-bound, CAS-backed OpenAPI construction with source provenance and
  Vacuum validation;
- namespace-bound, CAS-backed LikeC4 editing with direct CLI validation;
- executable four-Stage `openapi-from-source@1` and `likec4-from-source@1`
  Workflow configurations;
- an opt-in production-stack live-model quality gate and explicit two-Stage
  variants for caller-supplied reviewed analysis reports;
- exact ModelPolicy-bound cumulative model-call, tool-call, and token budgets
  with safe failure reports and deterministic release coverage;
- a deterministic production-boundary gate with two real single-slot Runtime
  Agents that proves Streamline, Router selection, Scheduler-owned escalation,
  exact stronger execution settings, durable safe plan events, frozen outputs,
  complete metrics, and slot reuse;
- an opt-in live Router contract evaluation and a local LiteLLM profile for LM
  Studio that preserve finite model/Worker/time budgets.
- a versioned owner-scoped public OpenAPI with managed configuration,
  encrypted credential lifecycle, Operations snapshots, exact local browser
  sessions, and bounded resumable Run/Operations WebSocket streams;
- an independently built React UI foundation with generated OpenAPI types,
  direct cookie/CSRF transport, guarded routes, explicit API compatibility,
  and a dependency-free Node static service that never acts as a BFF or proxy;
- a deterministic Chromium gate that drives the separate Node UI and Go API
  through local HTTPS while PostgreSQL, Scheduler, Control Plane and the real
  Python Runtime Agent execute streamline and OpenAPI workflows.
- immutable Runtime startup capability probes, complete heterogeneous
  specialist/generalist placement, positive capability visibility in
  Operations, and a real two-environment capacity-waiting gate.
- immutable database-backed RuntimeConfig versions, revisioned default/Run/
  Agent label bindings and write-only encrypted OTLP/proxy credentials;
- certificate-SPKI Runtime principals, candidate-specific exact configuration
  resolution, atomic allocation provenance and just-in-time secret delivery;
- allocation-local Python OTLP and explicit HTTP proxy adapters plus
  invocation-local Go Planner OTLP, all with bounded cleanup and no ambient
  process configuration;
- complete Runtime configuration Operations UI, Run label selection and safe
  historical Stage allocation provenance;
- an executable Runtime-configuration hardening matrix covering response-loss
  replay, PostgreSQL/CAS races, mTLS impersonation, adapter/lease/release
  failures, secret retention, browser operation and reuse of two disjoint
  specialist Runtime slots.

The first-slice and initial project-workflow milestones are complete. The
authoritative checkpoint is
[`tasks/index.yml`](../tasks/index.yml); the current Streamline contract and
completion evidence are recorded in
[`v3-001-planner-finish-contract.yml`](../tasks/v3-001-planner-finish-contract.yml),
[`v3-002-typed-planner-plan.yml`](../tasks/v3-002-typed-planner-plan.yml), and
[`v3-003-single-worker-streamline.yml`](../tasks/v3-003-single-worker-streamline.yml).
Router completion evidence is recorded in
[`v3-004-router-planner.yml`](../tasks/v3-004-router-planner.yml), immutable
escalation in
[`v3-008-scheduler-escalation.yml`](../tasks/v3-008-scheduler-escalation.yml),
the public-safe Planner journal in
[`v3-009-planner-plan-events.yml`](../tasks/v3-009-planner-plan-events.yml), and
the complete boundary proof in
[`v3-010-routing-escalation-e2e.yml`](../tasks/v3-010-routing-escalation-e2e.yml).
The public/browser foundation and its completion evidence are recorded in
[`v4-001-public-openapi-contract.yml`](../tasks/v4-001-public-openapi-contract.yml),
[`v4-007-run-event-websocket.yml`](../tasks/v4-007-run-event-websocket.yml), and
[`v4-008-react-ui-foundation.yml`](../tasks/v4-008-react-ui-foundation.yml).
The complete browser-stack proof is recorded in
[`v4-011-browser-e2e.yml`](../tasks/v4-011-browser-e2e.yml).

## Bundled Agent Skills

Reviewable built-in Agent Skill sources live only below
`configs/skills/<name>`. Validate a source tree and create its deterministic,
script-free package with:

```shell
go run ./cmd/contractor-skill validate configs/skills/<name>
go run ./cmd/contractor-skill package configs/skills/<name> /tmp/<name>.zip
```

`AgentTemplate` is a Server-side configuration term, not Worker guidance.
Never refer to it from bundled `SKILL.md` or reference content. Conditional
guidance says that an operation must be visible in the current Worker
invocation and states what to do when it is absent. The aggregate package gate
enforces this boundary for all nine bundled packages.

At startup the Server validates the complete bundled set before writing any
artifact, then creates only missing `skills/<name>` bindings in the configured
local user's UserScope. A restart never treats the filesystem as desired state:
an existing artifact with different bytes or media type is reported as
`seed_drift` and remains current. To adopt an edited bundled source, package it
explicitly and upload the resulting ZIP through the ordinary user Artifact PUT
with media type `application/vnd.contractor.agent-skill+zip` and the current
revision precondition. Future Runs use that new binding; existing Runs keep
their pinned revision.

Inspect the current binding and immutable version history, then use the current
revision as the compare-and-swap precondition for an operator-authored update:

```shell
SKILL_NAME=likec4
SKILL_BASE="http://127.0.0.1:8080/v1/artifacts/skills/$SKILL_NAME"

curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "$SKILL_BASE/metadata" | jq .
curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "$SKILL_BASE/versions?limit=100" | jq .

CURRENT_REVISION="$(curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "$SKILL_BASE/metadata" | jq -r .artifact.revision)"
go run ./cmd/contractor-skill validate "configs/skills/$SKILL_NAME"
go run ./cmd/contractor-skill package \
  "configs/skills/$SKILL_NAME" "/tmp/$SKILL_NAME.zip"
curl --fail --silent --show-error -X PUT \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  -H 'Content-Type: application/vnd.contractor.agent-skill+zip' \
  -H "If-Match: \"$CURRENT_REVISION\"" \
  --data-binary "@/tmp/$SKILL_NAME.zip" "$SKILL_BASE" | jq .
```

Create a Run through the ordinary Workflow API after uploading exact `source`
and optional `existing_likec4` inputs as described below. Version 3 assigns the
bundled LikeC4 skill to its builder and validator AgentTemplates:

```shell
jq -n --argjson source "$SOURCE_REF" --arg objective 'Model the architecture' \
  '{workflow:"likec4-from-source@3",parameters:{objective:$objective},artifacts:{source:$source}}' | \
  curl --fail --silent --show-error \
    -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
    -H "Idempotency-Key: likec4-skilled-$(date +%s)" \
    -H 'Content-Type: application/json' --data-binary @- \
    http://127.0.0.1:8080/v1/runs | jq .
```

The bounded real-process proof covers startup seeding, restart idempotency,
native ADK disclosure tools, an Artifact CAS update between source selection
and Run initialization, retry pinning, release cleanup, and empty-skill slot
reuse:

```shell
CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
  make test-agent-skills-mvp
```

Before publishing any bundled-package or Runtime Skill change, run the complete
bounded gate. It validates the shared adversarial corpus in independent Go and
Python implementations, packages all nine sources twice, checks the exact
server-side selection topology, exercises PostgreSQL/CAS and write-fence races,
and finishes with the real-process proof:

```shell
CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
  make test-agent-skills-hardening
```

The executable ownership map is
[`tests/e2e/agent_skills_matrix.yml`](../tests/e2e/agent_skills_matrix.yml).
Keep a new fault or invariant in that matrix and give it one concrete test
owner; a prose-only hardening claim is not a release gate.

Operational recovery is deliberately Artifact-first:

- `seed_drift` means the database current binding won. Inspect its metadata and
  immutable versions; never restart repeatedly expecting filesystem content to
  overwrite it. Publish the intended package with an ordinary `If-Match` CAS.
- A Run in `initializing` with reason `skill_initialization_pending` is not
  schedulable. Retryable Artifact failures are recovered by Scheduler using the
  already selected exact source. Invalid media, digest, archive or missing
  source terminates only that Run with a bounded `skill_*` code and logical
  `skills/<name>`; package content and parser text must not appear in logs.
- A CAS conflict during operator update means current changed independently.
  Re-read metadata and decide from the new exact revision; do not replay an
  update without a fresh precondition.
- `worker_stop_unconfirmed`, `allocation_cleanup_failed`, or a fenced Runtime
  after Skill extraction cleanup is a process-isolation event. Do not return
  that slot to service. Replace the Runtime process and inspect/remove its
  allocation work directory before reusing the same work root.

`resolvedSkills` is a mandatory private allocation field. Even an
AgentTemplate without skills is sent as `"resolvedSkills": []`; a non-empty
value contains only exact RunScope `skills/<name>` revisions and package
digests. It never contains the owner's source ref, package bytes or catalog
authority. This private-wire change is deliberately fail-closed: mixed Server
and Runtime Agent versions are unsupported. Before deploying a version that
adds or changes the allocation shape, stop new Run admission, let active
allocations finish (or cancel them), confirm every Runtime slot is released,
then replace the Server and all Runtime Agents together. Do not attempt a
rolling upgrade with live allocations.

The automated MVP test is the shortest proof that the actual Go Server and
Python Runtime Agent interoperate. It starts both production entry points,
creates a temporary deployment CA, uses an isolated PostgreSQL schema, and
replaces only the external OpenAI-compatible LLM Gateway with a deterministic
loopback fake.

The project-workflow end-to-end gate extends the same production boundary to
all eight OpenAPI and LikeC4 Stage executions. It uses one reusable Runtime slot,
a generated source ZIP and optional seeds, deterministic function-calling model
responses, and fixed child `vacuum`/`likec4` executables. It verifies real domain
tools, exact artifact lineage, frozen outputs, metrics, and workspace cleanup;
it does not score model quality.

An additional opt-in live quality gate runs the same production Server and
Runtime against a real OpenAI-compatible Gateway and the real validators. It
scores route, security, architecture, relationship, and source-evidence
coverage without comparing prose byte-for-byte or invoking a judge model.

## Prerequisites

- Go 1.25 or newer;
- Python 3.13 and `uv`;
- Node 24.20, Corepack 0.36 and pnpm 11.24 for the independent browser UI;
- a reachable PostgreSQL database in which the test user may create and drop
  schemas.

Install the locked Runtime Agent environment once:

```shell
cd runtime
uv sync --locked
cd ..
```

Install the pinned UI package manager and dependencies once:

```shell
npm install --global corepack@0.36.0
make ui-install
make ui-browser-install
```

Build and run the static UI service independently of Server:

```shell
make ui-build
CONTRACTOR_UI_API_BASE_URL=http://127.0.0.1:8080 \
  corepack pnpm --dir ui start
```

The loopback Go Server must be configured with
`http://127.0.0.1:4173` as an exact browser origin. The Node process exposes
only static files, `/runtime-config.json`, and `/healthz`; the browser sends
session, CSRF, Artifact and WebSocket traffic directly to Go Server.

Run the complete MVP gate:

```shell
CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
  make test-e2e

CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
  make test-capability-e2e

CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
  make test-runtime-labels-e2e

CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
  make test-runtime-configuration-e2e

CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
  make test-project-workflows

CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
  make test-ui-stack
```

## Environment-specific Runtime Agent capabilities

A Runtime Agent probes its enabled WorkerRuntime, Toolset/tool and
SandboxProfile factories once, after its private listener starts and before it
registers. The resulting positive snapshot is fixed for that `instanceId`.
Keep each process environment fixed: in particular, give it an explicit
executable search path containing only the validators and tools intended for
that deployment.

For example, two otherwise identical processes may be started by the service
manager with different fixed environments:

```shell
env PATH=/opt/contractor/source-tools/bin:/usr/bin \
  runtime/.venv/bin/python -m contractor_runtime [normal Runtime flags...]

env PATH=/opt/contractor/architecture-tools/bin:/usr/bin \
  runtime/.venv/bin/python -m contractor_runtime [normal Runtime flags...]
```

The second directory might contain a directly invokable `likec4`, while the
first does not. Both processes still advertise independent LikeC4 editing
tools; only the process whose startup probe succeeds advertises
`validate_likec4`. Do not install or replace executables underneath a running
process. Stop it, change the environment, and start a new process so it gets a
new `instanceId` and a newly probed immutable snapshot.

Verify the accepted positive snapshots in the UI at
`/operations/runtime-agents`, or through the read-only
`GET /v1/operations/runtime-agents` endpoint. The view intentionally contains
only exact runtime refs, SandboxProfile refs and Toolset/tool names. It does
not expose failed probe output, executable paths, environment values or a
remote re-probe control.

`make test-capability-e2e` proves this boundary with two production Python
Runtime Agent processes. One receives an isolated empty executable path and
remains idle while a Stage requiring `validate_likec4` waits in its first
`StageExecution`; a later process receives a deterministic fake LikeC4 CLI,
is selected without a retry, and completes normal prepare, A2A, finalization,
release and Operations reconciliation.

## Label-driven Runtime infrastructure

Runtime labels select physical Worker infrastructure without changing a
Workflow's ModelPolicy, budgets, Planner or tool allowlist. The checked-in
[deployment examples](../deploy/runtime-labels/README.md) show secret-free OTLP
and proxy documents. Publish credential material through the write-only
Operations mutation, publish each immutable RuntimeConfig, and then bind its
short label. A missing label always means only the pinned `default` binding;
the bootstrap `contractor-empty@1` default preserves authored executionConfig.

Every concurrently connected process must have a unique CA-signed identity:

```shell
go run ./cmd/contractor-pki issue-agent --name agent-proxy
go run ./cmd/contractor-pki issue-agent --name agent-telemetry
```

After binding an optional startup label such as `site-proxy`, start the first
process with its own certificate and an immutable adapter subset:

```shell
cd runtime
uv run contractor-runtime \
  --control-plane-url https://127.0.0.1:8443 \
  --advertised-control-url https://127.0.0.1:9443 \
  --advertised-a2a-url https://127.0.0.1:9443 \
  --ca-file ../.local/pki/ca.crt \
  --certificate-file ../.local/pki/agents/agent-proxy.crt \
  --private-key-file ../.local/pki/agents/agent-proxy.key \
  --listen 127.0.0.1:9443 \
  --work-root ../.local/runtime/agent-proxy \
  --initial-label site-proxy \
  --runtime-adapter http-proxy@1
```

Start the second process on another listen/advertised port with
`agent-telemetry.{crt,key}`. Omit `--runtime-adapter` to probe all built-ins, or
repeat it for an explicit subset. Startup labels seed only a previously unseen
certificate principal; subsequent assignments use CAS on
`PUT /v1/operations/runtime-agent-principals/{runtimeAgentId}/labels`.

Inspect certificate identity, authoritative labels and safe adapter capability
without receiving an endpoint or credential value:

```shell
curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  'http://127.0.0.1:8080/v1/operations/runtime-agent-principals?limit=50' | jq .
curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  'http://127.0.0.1:8080/v1/operations/runtime-labels?limit=50' | jq .
```

A Run label such as `debug` may require `otlp-http@1` even when neither Agent
is named or labeled `debug`; placement selects any capable idle candidate. An
Agent label is the highest physical Worker layer and can replace that Run's
endpoint/credential on its next allocation. Rebinding a label while work is
active changes only future Runs (for Run labels) or future allocations (for
Agent labels). `make test-runtime-labels-e2e` proves these rules with real
Server, PostgreSQL, two uniquely certified Runtime processes, OTLP protobuf,
authenticated proxying, exporter failure and complete slot reuse.

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
  `runtime/.venv` must already be installable as described below.

Accepted first-slice boundaries remain explicit: there is one active Control
Plane, no Vault/KMS or immutable RuntimeConfig garbage collection, no dynamic
Runtime capability changes, no content-bearing telemetry, no performance
benchmark and no dependency on a real LM Studio, Langfuse, Caido or cloud
service. These are deferred product/deployment features, not gaps hidden by
the hardening gate.

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
[`deploy/ui/README.md`](../deploy/ui/README.md). A compatible UI reload must
retain the Server session and any active Run; restarting Server intentionally
invalidates the in-memory browser session and changes the Operations event
generation.

## Hardening gates

The executable fault inventory is
[`tests/faults/matrix.yml`](../tests/faults/matrix.yml). It documents every
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

Public Run creation requires exactly one `Idempotency-Key` of 1–128 safe ASCII
characters. Retrying the same semantic request with the same authenticated
user/key returns the original Run and `Idempotency-Replayed: true`; reusing the
key for different content returns `409 Conflict`. Artifact PUT remains CAS:
after a lost successful response, repeating the consumed `If-None-Match` or
`If-Match` precondition returns conflict and creates no second revision; use a
GET to reconcile the accepted exact revision.

## Streamline Planner evidence

`make test-streamline` exercises the real Google ADK event loop through a
deterministic OpenAI-compatible Gateway, one fixed fake Worker, deterministic
full Stage-context propagation, succeeded/failed `finish` candidates,
Scheduler-owned transition selection, ordered subtask IDs with stale-dispatch
rejection, budget termination, secret redaction, and
PostgreSQL completed-session recovery. The example Workflow is
[`streamline_review_workflow.yaml`](../configs/examples/streamline_review_workflow.yaml).

`go test -race ./internal/planner/router/...` exercises Router through the same
real ADK loop. It verifies the exact two-argument execute schema and lexical
Available-agents prompt, rejects unknown, stale, and parallel selections before
A2A, and proves that model requests and reports contain logical binding names
but no allocation IDs, Runtime URLs, or credentials. The strict loadable
example is
[`router_openapi_workflow.yaml`](../configs/examples/router_openapi_workflow.yaml).

`make test-e2e` additionally starts two identical single-slot Runtime Agent
processes and runs three workflows in sequence. `streamline-copy@1` selects its
sole builder, `router-review@1` prepares builder and reviewer but invokes only
reviewer, and `escalating-copy@1` first records a non-retryable failed candidate
before Scheduler creates one linked `failed_escalation` execution with the
exact `strong_planner@1` and `strong_worker@1` settings. The harness checks the
PostgreSQL event journal and reports as well as the public result. Physical
allocation IDs, instance IDs, Runtime URLs, model content, and credentials are
not accepted in the UI-safe plan events.

The live Gateway dialect check is opt-in so normal tests remain deterministic.
It verifies that a deployed LiteLLM or LM Studio model returns a real function
tool call through the same Go adapter:

```shell
CONTRACTOR_STREAMLINE_LIVE_GATEWAY_URL='http://127.0.0.1:4000/v1' \
CONTRACTOR_STREAMLINE_LIVE_GATEWAY_TOKEN='replace-with-gateway-token' \
CONTRACTOR_STREAMLINE_LIVE_MODEL='planner-model' \
  go test -count=1 -run TestLiveGatewayToolCall ./tests/integration/streamline
```

The 2026-08-30 implementation gate also passed this smoke against LiteLLM and
LM Studio `qwen/qwen3.8-27b`; that external model is evidence, not a CI
dependency.

## Live Router evaluation through LiteLLM

The repository includes a small local proxy profile in
[`deploy/litellm/litellm_config.yaml`](../deploy/litellm/litellm_config.yaml).
It exposes `qwen/qwen3.8-27b` plus the shipped Planner/Worker aliases and
forwards all of them to LM Studio. The run script uses digest-pinned LiteLLM
and PostgreSQL images, retains LiteLLM's virtual-key database in a named Podman
volume, and keeps the proxy itself in the foreground. Create two owner-only
local bootstrap keys before the first start:

```shell
umask 077
mkdir -p .local/secrets
printf 'sk-%s\n' "$(openssl rand -hex 32)" > .local/secrets/litellm-master-key
printf 'sk-%s\n' "$(openssl rand -hex 32)" > .local/secrets/litellm-salt-key
CONTRACTOR_LITELLM_MASTER_KEY_FILE="$(pwd)/.local/secrets/litellm-master-key" \
CONTRACTOR_LITELLM_SALT_KEY_FILE="$(pwd)/.local/secrets/litellm-salt-key" \
CONTRACTOR_LM_STUDIO_URL='http://192.168.1.217:1234/v1' \
  deploy/litellm/run.sh
```

`Ctrl-C` removes the proxy container but deliberately leaves
`contractor-litellm-postgres` and its named volume intact. The master key file
is also the `adminKeyFile` used by Contractor's exact Gateway binding; the salt
key remains LiteLLM-only. Neither file belongs in repository YAML.

In another terminal, run the bounded real-model Router scenario:

```shell
CONTRACTOR_LIVE_LLM_URL='http://127.0.0.1:4000/v1' \
CONTRACTOR_LIVE_LLM_MODEL='qwen/qwen3.8-27b' \
  make test-live-routing
```

With the documented loopback Gateway and model defaults, the equivalent short
command is `scripts/test-live-routing.sh`.

Set `CONTRACTOR_LIVE_LLM_TOKEN` only when the local Gateway requires one. The
evaluation uses the production Go ADK Router and exact four-tool contract,
offers builder and reviewer descriptions, and requires exactly one dispatch to
reviewer. The Worker and artifact lookup are deterministic because the normal
E2E gate already covers A2A and mTLS. Success prints only a model-name SHA-256,
bounded call count, selected logical Worker, and outcome. Failure reports a
stable Contractor error code; inspect local LiteLLM/LM Studio logs separately
when provider diagnostics are needed. Provider bodies and tokens are never
printed by the test command.

The live check found and now guards an important dialect detail: no-argument
tools still carry an explicit closed JSON object schema with `properties: {}`.
LM Studio rejects the otherwise equivalent schema when that field is omitted.

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

To smoke-test the real ADK/LiteLLM adapter and provider-supplied token usage
without making the deterministic suite depend on a live model, point the
opt-in test at any OpenAI-compatible gateway:

```shell
cd runtime
CONTRACTOR_LIVE_LLM_GATEWAY_URL='http://127.0.0.1:4000/v1' \
CONTRACTOR_LIVE_LLM_GATEWAY_TOKEN='replace-with-gateway-token' \
CONTRACTOR_LIVE_LLM_MODEL='worker-model' \
  uv run pytest tests/test_live_gateway.py
```

## Manual local stack

The default local ports are public HTTP `127.0.0.1:8080`, Control Plane mTLS
`127.0.0.1:8443`, and Runtime Agent mTLS `127.0.0.1:9443`. PostgreSQL and an
OpenAI-compatible LLM Gateway must already be running. Its immutable API base
is authored in `configs/llm-gateways/` and its model aliases and limits are
authored in `configs/model-policies/`; LiteLLM is one supported backend, but it
is not required by the architecture.

Create deployment identities:

```shell
go run ./cmd/contractor-pki init-ca
go run ./cmd/contractor-pki issue-control-plane
go run ./cmd/contractor-pki issue-agent --name agent-local
```

Create the owner-only local principal bootstrap. The command reads the password
twice from the terminal without echo; the file contains only its exact Argon2id
hash, not the password:

```shell
umask 077
mkdir -p .local/secrets
go run ./cmd/contractor-server auth hash-password \
  --user-id=local-user --username=admin > .local/secrets/local-auth.yaml
```

In the first terminal, export Server settings and start it. The explicit
loopback mode uses the non-production `contractor_loopback_session` cookie and
accepts only a loopback public listener and loopback HTTP browser origins.
Production omits that mode, uses HTTPS origins, and issues only the Secure
`__Host-contractor_session` cookie.

```shell
export CONTRACTOR_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor?sslmode=disable'
export CONTRACTOR_PUBLIC_USER_ID='local-user'
export CONTRACTOR_PUBLIC_BEARER_TOKEN='replace-with-a-local-api-token'
export CONTRACTOR_LOCAL_AUTH_FILE="$(pwd)/.local/secrets/local-auth.yaml"
export CONTRACTOR_BROWSER_ORIGINS='http://127.0.0.1:5173'
export CONTRACTOR_INSECURE_LOOPBACK_COOKIE='true'
export CONTRACTOR_LLM_GATEWAY_TOKEN='replace-with-the-gateway-token'
export CONTRACTOR_CA_FILE='.local/pki/ca.crt'
export CONTRACTOR_CONTROL_PLANE_CERT_FILE='.local/pki/control-plane.crt'
export CONTRACTOR_CONTROL_PLANE_KEY_FILE='.local/pki/control-plane.key'
make run-local
```

`CONTRACTOR_PUBLIC_BEARER_TOKEN` remains the non-browser authentication path;
CLI and automation clients send it only in `Authorization: Bearer ...` and do
not send browser Origin, cookie, or CSRF headers. It maps to the `userId` loaded
from local-auth. `CONTRACTOR_PUBLIC_USER_ID` is a temporary compatibility
assertion and, when present, must equal that value.

The two LLM token variables are a development bootstrap only. A non-empty
`CONTRACTOR_LLM_GATEWAY_TOKEN` creates the in-memory credential ID
`development-worker`; `CONTRACTOR_PLANNER_LLM_GATEWAY_TOKEN` creates
`development-planner` and defaults to the Worker token. Select those IDs in a
Workflow or Run `executionConfig`. URL, model and limits never come from these
environment variables and cannot override an immutable Run snapshot. Omit the
credential selector for a Gateway that intentionally accepts unauthenticated
requests.

Managed Gateway credentials use a separate database-encryption key. Generate
it once as an owner-only file and pass only its absolute path to Server:

```shell
umask 077
mkdir -p .local/secrets
openssl rand -base64 32 > .local/secrets/credential-master-key
cat > .local/secrets/llm-gateway-admin-bindings.yaml <<EOF
bindings:
  - llmGateway:
      gatewayId: local-litellm
      version: "1"
      digest: sha256:6e1bcf93a5d1fc64307dcd256a5c120f1bac54fe85e9d23d0aaafb84d4f14376
    adminKeyFile: $(pwd)/.local/secrets/litellm-master-key
EOF
go run ./cmd/contractor-server serve \
  --config-root ./configs/e2e \
  --credential-master-key-file="$(pwd)/.local/secrets/credential-master-key" \
  --llm-gateway-admin-bindings-file="$(pwd)/.local/secrets/llm-gateway-admin-bindings.yaml"
```

There is deliberately no environment variable or command-line literal for the
key bytes. The file must contain RFC 4648 base64 for exactly 32 bytes, with at
most one trailing newline, and must not be a symlink or readable by group or
world. The flag is optional while no encrypted credential rows exist; once one
exists, a missing file or a key whose fingerprint differs from the stored rows
makes Server startup fail before accepting traffic.

The admin-binding document is strict YAML and contains no key bytes. Each
entry names the complete digest-bearing Gateway ref and an absolute owner-only
`sk-` key file. Server resolves the ref at startup and remembers its exact
management origin; a missing/wrong digest, insecure file, redirect, malformed
provider response, or unbound Gateway fails closed. Changing an immutable
LLMGatewayConfig therefore requires a new binding entry and Server restart.
The LiteLLM manager uses finite timeouts and bounded bodies and accepts only
the response shape verified by `make test-litellm-contract` against the pinned
image.

Once Server is running, create an active virtual key without ever sending its
token through the public API. This example derives both exact refs from the
safe configuration API:

```shell
export CONTRACTOR_API_TOKEN='replace-with-a-local-api-token'
GATEWAY_REF="$(curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  http://127.0.0.1:8080/v1/configurations/llm-gateways/local-litellm/versions/1 | \
  jq -c '.ref | {gatewayId:.name,version,digest}')"
WORKER_POLICY_REF="$(curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  http://127.0.0.1:8080/v1/configurations/model-policies/worker/versions/1 | \
  jq -c '.ref | {policyId:.name,version,digest}')"
jq -n --argjson gateway "$GATEWAY_REF" --argjson policy "$WORKER_POLICY_REF" \
  '{credentialId:"managed-worker",llmGateway:$gateway,
    label:"Local Worker",gatewayPolicy:{modelPolicies:[$policy],
    maxBudget:10,budgetDuration:"1d",rpmLimit:30,maxParallelRequests:2}}' | \
  curl --fail --silent --show-error \
    -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
    -H 'Idempotency-Key: create-managed-worker-1' \
    -H 'Content-Type: application/json' --data-binary @- \
    http://127.0.0.1:8080/v1/operations/credentials | jq .
```

The response contains only ID, exact Gateway ref, label, effective policy and
creation time. LiteLLM's generated virtual key is encrypted immediately in
PostgreSQL. Select `managed-worker` in Workflow or Run `executionConfig`; keep
the development token environment variables unset when testing this path.

The read-only Operations API exposes one coherent current Control Plane view:

```shell
curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  http://127.0.0.1:8080/v1/operations/snapshot | jq .
curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  'http://127.0.0.1:8080/v1/operations/runtime-agents?limit=50' | jq .
curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  'http://127.0.0.1:8080/v1/operations/allocations?limit=50' | jq .
```

Each response carries the process generation and revision used later by the
Operations WebSocket. Page cursors are bound to that pair and return a safe
`400 invalid_request` after concurrent state change; restart pagination from a
fresh snapshot. These endpoints cannot release, fence, reassign or otherwise
mutate an allocation.

In the second terminal, start the single-slot Runtime Agent:

```shell
cd runtime
uv run contractor-runtime \
  --control-plane-url https://127.0.0.1:8443 \
  --advertised-control-url https://127.0.0.1:9443 \
  --advertised-a2a-url https://127.0.0.1:9443 \
  --ca-file ../.local/pki/ca.crt \
  --certificate-file ../.local/pki/agents/agent-local.crt \
  --private-key-file ../.local/pki/agents/agent-local.key \
  --listen 127.0.0.1:9443 \
  --work-root ../.local/runtime/work
```

The Server advertises the agreed heartbeat interval of 10 seconds and confirmed
lease of 60 seconds. A new Runtime becomes placement-eligible after it echoes
the first heartbeat acknowledgement, so allow roughly one heartbeat interval
before expecting a newly submitted Run to leave `preparing`.

Upload and execute the fixture with `curl` and `jq`:

```shell
export CONTRACTOR_API_TOKEN='replace-with-a-local-api-token'
INPUT_REF="$(curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  -H 'Content-Type: text/plain' -H 'If-None-Match: *' \
  --data-binary $'contractor-local-input\n' \
  http://127.0.0.1:8080/v1/artifacts/projects/source | jq -c .artifact)"

RUN_ID="$(jq -n --argjson source "$INPUT_REF" \
  '{workflow:"artifact-copy@1",parameters:{},artifacts:{source:$source}}' | \
  curl --fail --silent --show-error \
    -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
    -H "Idempotency-Key: local-run-$(date +%s)" \
    -H 'Content-Type: application/json' --data-binary @- \
    http://127.0.0.1:8080/v1/runs | jq -r .runId)"

curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "http://127.0.0.1:8080/v1/runs/$RUN_ID"

curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "http://127.0.0.1:8080/v1/runs/$RUN_ID/outputs/result"
```

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

## OpenAPI from a source archive

`openapi-from-source@1` runs four serial Stages: dependency discovery, project
discovery, incremental OpenAPI construction, and final validation/repair. Each
Stage gets its own allocation and workspace. The reports and document move
between Stages as exact RunScope artifact revisions rather than model memory.

The Runtime Agent host must have `vacuum` on its `PATH`:

```shell
command -v vacuum
vacuum version
```

Create a ZIP with normalized relative entries. For a Git project without
tracked symbolic links, `git archive` is the simplest safe option; it includes
only the committed tree, so commit or otherwise package intentional local
changes first.

```shell
export PROJECT_ROOT='/absolute/path/to/project'
export SOURCE_ZIP='/tmp/contractor-project-source.zip'
git -C "$PROJECT_ROOT" archive --format=zip --output="$SOURCE_ZIP" HEAD
```

The source Toolset deliberately rejects path traversal, absolute/backslash
paths, duplicate entries, symbolic/special entries, encrypted members, more
than 10,000 members, archives above 16 MiB compressed, and archives above 64
MiB declared uncompressed.

Upload the source as a UserScope artifact. The Run creation transaction copies
the exact revision to `inputs/source`; Workers never read the UserScope binding
directly.

```shell
SOURCE_REF="$(curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  -H 'Content-Type: application/zip' -H 'If-None-Match: *' \
  --data-binary "@$SOURCE_ZIP" \
  http://127.0.0.1:8080/v1/artifacts/projects/project-source | jq -c .artifact)"
```

An existing OpenAPI 3.0/3.1 YAML or JSON document is optional. When supplied,
the build Stage reads its exact input revision and creates an independent
`openapi/openapi` Run binding; the uploaded UserScope value is never mutated.

```shell
OPENAPI_SEED_REF="$(curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  -H 'Content-Type: application/yaml' -H 'If-None-Match: *' \
  --data-binary '@/absolute/path/to/openapi.yaml' \
  http://127.0.0.1:8080/v1/artifacts/projects/existing-openapi | jq -c .artifact)"
```

Submit a Run with the optional seed:

```shell
RUN_ID="$(jq -n \
  --argjson source "$SOURCE_REF" \
  --argjson seed "$OPENAPI_SEED_REF" \
  --arg objective 'Document the implemented public HTTP API' \
  '{workflow:"openapi-from-source@1",parameters:{objective:$objective},artifacts:{source:$source,existing_openapi:$seed}}' | \
  curl --fail --silent --show-error \
    -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
    -H "Idempotency-Key: openapi-run-$(date +%s)" \
    -H 'Content-Type: application/json' --data-binary @- \
    http://127.0.0.1:8080/v1/runs | jq -r .runId)"
```

For a new document, omit `existing_openapi` and build the artifact map as
`{source:$source}`. Poll `/v1/runs/$RUN_ID`; after success, retrieve the two
frozen Workflow outputs:

```shell
curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "http://127.0.0.1:8080/v1/runs/$RUN_ID/outputs/openapi" \
  --output /tmp/generated-openapi.yaml

curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "http://127.0.0.1:8080/v1/runs/$RUN_ID/outputs/validation_report" \
  --output /tmp/openapi-validation-report.md
```

Intermediate Run bindings are `analysis/dependencies`, `analysis/project`,
`openapi/openapi`, and `openapi/validation-report`. The Workflow succeeds only
when the final `validate_openapi` call reports a structurally clean document and
no serious Vacuum findings. Missing Vacuum is an explicit failed validation,
never an implicit pass.

### Workspace-backed OpenAPI variant

Use `openapi-from-workspace@1` when the Worker should inspect the hydrated tree
through `filesystem@1` rather than open the ZIP through `source-analysis@1`.
The Run request is otherwise identical. Every Stage receives a new isolated
workspace reconstructed from the exact source ref and, after the first Stage,
the exact cumulative state exported by its predecessor. The final Run freezes
`openapi`, `validation_report`, `workspace_state`, and `workspace_diff`.

Workspace support is an immutable Runtime startup capability. Enable exactly
one backend when starting an agent:

```shell
# Disposable files below a dedicated local root; binary ZIP members are retained.
contractor-runtime \
  --workspace-storage local \
  --workspace-work-root /var/lib/contractor/project-workspaces \
  ...

# Isolated in-process fsspec storage; binary ZIP members are intentionally skipped.
contractor-runtime --workspace-storage memory ...
```

The optional `--workspace-max-files`, `--workspace-max-expanded-bytes`,
`--workspace-max-managed-text-bytes`, and `--workspace-max-file-bytes` flags
replace backend defaults. They are probed and advertised at registration and
cannot change for the lifetime of that Runtime process. Physical workspace
paths never cross the private Runtime boundary. Direct-mode edits affect only
the disposable hydrated copy; overlay-mode changes become durable only through
the Workflow-declared state/diff Artifact slots.

Workspace v1 intentionally has three sharp boundaries. Text tools cannot read,
modify or encode binary files; a future binary patch is a separate artifact
type. No shell/subprocess tool or automatic temporary-checkout adapter exists.
Finally, allocations—including Router siblings—never share a live workspace;
coordination happens only through exact exported artifacts in a later
allocation. Keep the local `workRoot` private to the Runtime OS identity: a
same-UID process with write access is part of that host's trust boundary.

## LikeC4 from a source archive

`likec4-from-source@1` reuses the same dependency- and project-discovery
contracts as the OpenAPI workflow, then builds and repair-validates one
single-file architecture model. Reports and the model cross Stage boundaries as
exact artifact revisions; no Worker relies on another allocation's memory or
workspace.

`likec4-from-workspace@1` is the explicit workspace-backed equivalent. It keeps
the same four semantic Stages and domain outputs, selects the workspace-aware
AgentTemplates, carries cumulative overlay state by exact Artifact revision, and
also freezes `workspace_state` and `workspace_diff`. Existing
`likec4-from-source@*` IDs are unchanged.

Install the LikeC4 CLI directly on every Runtime Agent host and make it visible
on `PATH`. The Runtime never invokes `npx` or downloads a validator while a Run
is executing.

```shell
command -v likec4
likec4 version
```

Create and upload `SOURCE_REF` with the safe ZIP procedure in the OpenAPI
section above. A pre-existing single-file model is optional. When supplied, the
build Stage copies its exact revision into `likec4/architecture`; it never
modifies the UserScope artifact. Both the canonical LikeC4 media type and plain
UTF-8 text are accepted as seeds and normalized to `text/vnd.likec4`.

```shell
LIKEC4_SEED_REF="$(curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  -H 'Content-Type: text/vnd.likec4' -H 'If-None-Match: *' \
  --data-binary '@/absolute/path/to/architecture.c4' \
  http://127.0.0.1:8080/v1/artifacts/projects/existing-likec4 | jq -c .artifact)"
```

Submit a Run with the optional seed:

```shell
RUN_ID="$(jq -n \
  --argjson source "$SOURCE_REF" \
  --argjson seed "$LIKEC4_SEED_REF" \
  --arg objective 'Model the implemented architecture and trust boundaries' \
  '{workflow:"likec4-from-source@1",parameters:{objective:$objective},artifacts:{source:$source,existing_likec4:$seed}}' | \
  curl --fail --silent --show-error \
    -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
    -H "Idempotency-Key: likec4-run-$(date +%s)" \
    -H 'Content-Type: application/json' --data-binary @- \
    http://127.0.0.1:8080/v1/runs | jq -r .runId)"
```

For a new model, omit `existing_likec4` and use
`artifacts:{source:$source}`. Poll `/v1/runs/$RUN_ID`; after success, retrieve
the immutable Workflow outputs:

```shell
curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "http://127.0.0.1:8080/v1/runs/$RUN_ID/outputs/architecture" \
  --output /tmp/generated-architecture.c4

curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "http://127.0.0.1:8080/v1/runs/$RUN_ID/outputs/validation_report" \
  --output /tmp/likec4-validation-report.md
```

Intermediate bindings are `analysis/dependencies`, `analysis/project`,
`likec4/architecture`, and `likec4/validation-report`. The supported MVP is one
self-contained file: includes, multi-file projects, rendering, and layout
artifacts are outside this workflow. It succeeds only after direct
`likec4 validate` reports a clean model. A missing or failed CLI is an explicit
retryable validation failure, never an implicit pass.

## Reusing explicit analysis reports

`openapi-from-analysis@1` and `likec4-from-analysis@1` are two-Stage variants
for callers that already have reviewed dependency and project reports. They do
not search prior Runs or choose a current artifact implicitly. The caller must
upload exact `text/markdown` reports to UserScope and select their revisions
together with the exact source revision. Run creation copies them to
`inputs/dependency_report` and `inputs/project_report`; Workers may mutate only
RunScope artifacts and the uploaded values remain unchanged.

```shell
DEPENDENCY_REPORT_REF="$(curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  -H 'Content-Type: text/markdown' -H 'If-None-Match: *' \
  --data-binary '@/absolute/path/to/dependency-report.md' \
  http://127.0.0.1:8080/v1/artifacts/projects/dependency-report | jq -c .artifact)"

PROJECT_REPORT_REF="$(curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  -H 'Content-Type: text/markdown' -H 'If-None-Match: *' \
  --data-binary '@/absolute/path/to/project-report.md' \
  http://127.0.0.1:8080/v1/artifacts/projects/project-report | jq -c .artifact)"

RUN_ID="$(jq -n \
  --argjson source "$SOURCE_REF" \
  --argjson dependencies "$DEPENDENCY_REPORT_REF" \
  --argjson project "$PROJECT_REPORT_REF" \
  '{workflow:"openapi-from-analysis@1",parameters:{},artifacts:{source:$source,dependency_report:$dependencies,project_report:$project}}' | \
  curl --fail --silent --show-error \
    -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
    -H "Idempotency-Key: openapi-analysis-run-$(date +%s)" \
    -H 'Content-Type: application/json' --data-binary @- \
    http://127.0.0.1:8080/v1/runs | jq -r .runId)"
```

Use `likec4-from-analysis@1` in the same request to produce LikeC4. Optional
`existing_openapi` and `existing_likec4` inputs retain the contracts described
above. Contractor v1alpha1 does not prove that a report revision was derived
from the selected source revision; the caller owns that compatibility decision.
A future provenance/cache policy can automate it without changing these
explicit workflow contracts.

## Live project-workflow quality evaluation

This gate is deliberately excluded from `make verify` and ordinary CI. It
requires PostgreSQL, `vacuum`, `likec4`, the locked Python environment, and a
model with reliable OpenAI-compatible function calling. The fixture is a small
FastAPI service with authenticated GET/POST routes, PostgreSQL persistence, and
an outbound Inventory HTTP client. Both workflows start from source only, with
no prebuilt OpenAPI or LikeC4 seed.

Run directly against LM Studio:

```shell
CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
CONTRACTOR_WORKFLOWS_LIVE_GATEWAY_URL='http://192.168.1.217:1234/v1' \
CONTRACTOR_WORKFLOWS_LIVE_GATEWAY_TOKEN='unused' \
CONTRACTOR_WORKFLOWS_LIVE_MODEL='qwen/qwen3.8-27b' \
  make test-project-workflows-live
```

For LiteLLM, map an alias such as `project-workflow-model` to the LM Studio
upstream in its configuration, start the proxy, and use the alias at the proxy
API base:

```shell
CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
CONTRACTOR_WORKFLOWS_LIVE_GATEWAY_URL='http://127.0.0.1:4000/v1' \
CONTRACTOR_WORKFLOWS_LIVE_GATEWAY_TOKEN='replace-with-litellm-master-key' \
CONTRACTOR_WORKFLOWS_LIVE_MODEL='project-workflow-model' \
  make test-project-workflows-live
```

Allow up to 30 minutes for each four-Stage workflow, plus setup and cleanup.
`domain_worker@1` permits up to 16,384 output tokens per model response and one
Worker invocation permits at most 24 model calls, 96 tool calls, and 250,000
provider-reported cumulative tokens. Actual model/tool/token counters are
reported by Stage in the Run status. On failure,
the harness writes bounded generated documents, analysis reports, predicate
codes, and counters below ignored `.local/eval-results/`. It deliberately does
not persist the source archive, Gateway URL/token, prompts, or provider response
bodies. A successful run removes its isolated schema, certificates, processes,
and Runtime workspaces without retaining evaluation artifacts.
During local diagnosis only, set `CONTRACTOR_WORKFLOWS_LIVE_ONLY` to either
`openapi-from-source@1` or `likec4-from-source@1`; the default and documented
quality gate always execute both.

## Worker invocation budgets

Every shipped ModelPolicy declares `maxModelCalls`, `maxToolCalls`, and
`maxTotalTokens` in addition to per-response `maxOutputTokens`. The Runtime
applies the cumulative limits to one Worker A2A invocation, including its
optional tool-free result-finalization call. Model and tool capacity is checked
before starting the next operation. Provider `total_tokens` is accumulated
after each response; a response that crosses the limit cannot execute its tool
call. Missing usage is reported as unavailable and model/tool limits remain
active.

Budget exhaustion returns a retryable failed Worker result with stable code
`worker_budget_exhausted`. The Worker report includes configured and observed
limits plus `model_calls`, `tool_calls`, or `total_tokens` as the exhausted
dimension, without retaining prompts or payloads. Workflow `maxAttempts` still
owns whole-Stage retry and the Server Planner deadline remains the outer wall
limit. Tune the YAML policy and create a new policy digest; do not patch Runtime
constants or treat `maxOutputTokens` as a cumulative ceiling.

Cancellation is an idempotent durable request. It interrupts an active local
Planner immediately; another Server process observes the same `cancelling`
state while renewing its claim. To cancel instead of waiting for the result:

```shell
jq -n --arg reason 'no longer needed' '{reason:$reason}' | \
  curl --fail --silent --show-error \
    -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
    -H 'Content-Type: application/json' --data-binary @- \
    "http://127.0.0.1:8080/v1/runs/$RUN_ID/cancel"
```

The response is `202 Accepted` while bounded cleanup is in progress and `200 OK`
when the Run was already terminal. Repeating the request never replaces
the first cancellation reason or changes a successful terminal Run.

Using a real LiteLLM/provider is deliberately opt-in. Publish its non-secret
endpoint as an exact `LLMGatewayConfig`, publish model aliases and bounds as
exact `ModelPolicy` documents, then select them independently for Planner and
Workers through `executionConfig`. For local bootstrap, the Worker and Planner
may use the separately named `development-worker` and `development-planner`
credentials described above. CI and `make test-e2e` use a temporary authored
Gateway manifest and a deterministic fake model rather than an external
provider.

## Troubleshooting

- A Run that remains in `preparing` usually has no confirmed, compatible idle
  Runtime Agent. Check registration and two successive heartbeat exchanges.
- TLS hostname failures mean the URL host is absent from the leaf certificate
  SANs. The local PKI commands include `localhost`, `127.0.0.1`, and `::1` by
  default.
- A private request rejected as the wrong role usually means an Agent
  certificate was used where the Control Plane certificate with its dedicated
  URI SAN is required.
- Validate fixture manifests independently with
  `go run ./cmd/contractor-server config validate --root ./configs/e2e`.
- Gateway 404 responses usually mean the configured base omitted or duplicated
  the provider's `/v1` path.
