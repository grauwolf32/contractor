# Local development

## Implementation checkpoint

As of 2026-08-30, implementation tasks through `V1-006` are complete and the
project-workflow increment is underway. The repository contains the runnable Go
Server/Python Runtime Agent MVP plus:

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
- a model-backed `streamline@1` Planner built with Google ADK Go, fixed
  sequential A2A Worker tools, explicit `finish`/`escalate`, and bounded model,
  token, Worker-call, and wall-time budgets;
- a PostgreSQL-backed Planner session adapter that persists only redacted ADK
  facts and recovers a completed decision without repeating model or Worker
  calls;
- bounded UTF-8 Run-artifact tools and safe ZIP source exploration;
- namespace-bound, CAS-backed OpenAPI construction with source provenance and
  Vacuum validation;
- namespace-bound, CAS-backed LikeC4 editing with direct CLI validation;
- executable four-Stage `openapi-from-source@1` and `likec4-from-source@1`
  Workflow configurations.

The first-slice milestone is complete; OpenAPI and LikeC4 project workflows are
the current product increment. The authoritative checkpoint is
[`tasks/index.yml`](../tasks/index.yml); detailed Streamline completion evidence
is recorded in
[`v1-006-streamline-planner.yml`](../tasks/v1-006-streamline-planner.yml).

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
deterministic OpenAI-compatible Gateway, two fixed fake Workers, exact artifact
selection, explicit `finish`, budget termination, secret redaction, and
PostgreSQL completed-session recovery. The example Workflow is
[`streamline_review_workflow.yaml`](../configs/examples/streamline_review_workflow.yaml).

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
OpenAI-compatible LLM Gateway must already be running. The gateway URL should
be its OpenAI API base, commonly ending in `/v1`; LiteLLM is one supported
backend, but it is not required by the architecture.

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
export CONTRACTOR_LLM_GATEWAY_URL='http://127.0.0.1:4000/v1'
export CONTRACTOR_LLM_GATEWAY_TOKEN='replace-with-the-gateway-token'
export CONTRACTOR_PLANNER_MODEL='planner-model'
export CONTRACTOR_CA_FILE='.local/pki/ca.crt'
export CONTRACTOR_CONTROL_PLANE_CERT_FILE='.local/pki/control-plane.crt'
export CONTRACTOR_CONTROL_PLANE_KEY_FILE='.local/pki/control-plane.key'
make run-local
```

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

Using a real LiteLLM/provider is deliberately opt-in. Worker settings use
`CONTRACTOR_LLM_GATEWAY_URL` and `CONTRACTOR_LLM_GATEWAY_TOKEN`; Planner URL and
token default to those values, while `CONTRACTOR_PLANNER_MODEL` selects its
Gateway model. Separate deployments may set
`CONTRACTOR_PLANNER_LLM_GATEWAY_URL` and
`CONTRACTOR_PLANNER_LLM_GATEWAY_TOKEN`. CI and `make test-e2e` never use
provider credentials or make an external model call.

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
