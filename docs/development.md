# Local development

## Implementation checkpoint

As of 2026-08-30, implementation tasks through `V3-010` are complete. The
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
- a reachable PostgreSQL database in which the test user may create and drop
  schemas.

Install the locked Runtime Agent environment once:

```shell
cd runtime
uv sync --locked
cd ..
```

Run the complete MVP gate:

```shell
CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
  make test-e2e

CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
  make test-project-workflows
```

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

In the first terminal, export Server settings and start it. Keep secrets in the
environment or a local ignored `.env`; do not place them in YAML manifests.

```shell
export CONTRACTOR_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor?sslmode=disable'
export CONTRACTOR_PUBLIC_USER_ID='local-user'
export CONTRACTOR_PUBLIC_BEARER_TOKEN='replace-with-a-local-api-token'
export CONTRACTOR_LLM_GATEWAY_TOKEN='replace-with-the-gateway-token'
export CONTRACTOR_CA_FILE='.local/pki/ca.crt'
export CONTRACTOR_CONTROL_PLANE_CERT_FILE='.local/pki/control-plane.crt'
export CONTRACTOR_CONTROL_PLANE_KEY_FILE='.local/pki/control-plane.key'
make run-local
```

The two token variables are a development bootstrap only. A non-empty
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

## LikeC4 from a source archive

`likec4-from-source@1` reuses the same dependency- and project-discovery
contracts as the OpenAPI workflow, then builds and repair-validates one
single-file architecture model. Reports and the model cross Stage boundaries as
exact artifact revisions; no Worker relies on another allocation's memory or
workspace.

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
