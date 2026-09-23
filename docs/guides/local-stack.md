# Run a local stack

[Documentation index](../README.md) · [Development setup](../development.md)

Install the dependencies in the development guide first. Commands below start
the checked-in loopback configuration and execute one artifact-copy Workflow.
Use separate terminals for the Server, Runtime Agent and UI.

## Prepare and start Server

The default local ports are public HTTP `127.0.0.1:8080`, Control Plane mTLS
`127.0.0.1:8443`, and Runtime Agent mTLS `127.0.0.1:9443`. PostgreSQL and an
OpenAI-compatible LLM Gateway must already be running. Its immutable API base
is authored in `configs/llm-gateways/` and its model aliases and limits are
authored in `configs/model-policies/`; LiteLLM is one supported backend. The
checked-in Gateway uses `http://127.0.0.1:4000/v1` with `worker-model` and
`planner-model` aliases. See [the local LiteLLM setup](../deployment.md#llm-gateway)
if you need to start that Gateway.

Create deployment identities:

```shell
contractor pki init-ca
contractor pki issue-control-plane
contractor pki issue-runtime --name agent-local
```

Create the owner-only local principal bootstrap. The command reads the password
twice from the terminal without echo; the file contains only its exact Argon2id
hash, not the password:

```shell
umask 077
mkdir -p .local/secrets
contractor server auth hash-password \
  --user-id=local-user --username=admin > .local/secrets/local-auth.yaml
```

In the first terminal, export only the local secret values, migrate, and start
the Server with the checked-in non-secret
[`ServerConfig`](../../configs/server.local.yaml). Relative paths in that file are
resolved from the file's directory. The explicit loopback mode uses the
non-production `contractor_loopback_session` cookie and requires a loopback
public listener. This configuration allows the exact loopback UI origin.
Production omits that mode, uses HTTPS origins, and issues only the Secure
`__Host-contractor_session` cookie.

```shell
export CONTRACTOR_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor?sslmode=disable'
export CONTRACTOR_PUBLIC_BEARER_TOKEN='replace-with-a-local-api-token'
export CONTRACTOR_LLM_GATEWAY_TOKEN='replace-with-the-gateway-token'
contractor server migrate
contractor server run --config ./configs/server.local.yaml
```

`CONTRACTOR_SERVER_CONFIG` and `--server-config` are aliases for `--config`.
Effective process settings use `built-in defaults < ServerConfig < environment
< command-line flags`. The YAML contract rejects unknown and duplicate fields,
multiple documents, and secret-bearing fields such as database URLs or tokens.

`CONTRACTOR_PUBLIC_BEARER_TOKEN` remains the non-browser authentication path;
CLI and automation clients send it only in `Authorization: Bearer ...` and do
not send browser Origin, cookie, or CSRF headers. It maps to the `userId` loaded
from local-auth.

The two LLM token variables are a development bootstrap only. A non-empty
`CONTRACTOR_LLM_GATEWAY_TOKEN` creates the in-memory credential ID
`development-worker`; `CONTRACTOR_PLANNER_LLM_GATEWAY_TOKEN` creates
`development-planner` and defaults to the Worker token. The shipped local
Workflow manifests select those IDs by role; custom Workflow or Run
`executionConfig` must do the same explicitly when it uses the bootstrapped
credentials. URL, model and limits never come from these environment variables
and cannot override an immutable Run snapshot. Omit the credential selector
for a Gateway that intentionally accepts unauthenticated requests. A selected
but unavailable development credential rejects Run creation before execution.

For managed virtual keys, follow [Gateway credentials](../operations/gateway-credentials.md).

## Inspect Operations

The read-only Operations API exposes one coherent current Control Plane view:

Use another terminal at the repository root. Set `CONTRACTOR_API_TOKEN` to the
same value as the Server's `CONTRACTOR_PUBLIC_BEARER_TOKEN`:

```shell
export CONTRACTOR_API_TOKEN='replace-with-a-local-api-token'
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

## Start a Runtime Agent

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

HTTP and scanner tools refuse loopback and link-local targets by default;
private-network and public targets are allowed. A Project HTTP target on this
host is allowed for Runs of that Project. To reach other targets on this
machine, directly or through a local Caido forward proxy, add
`--allowed-target-network 127.0.0.0/8`; the Server, Runtime and proxy endpoints
stay refused. See the
[target policy](../spec/11-http-and-caido-tools.md#target-policy).

The Server advertises the agreed heartbeat interval of 10 seconds and confirmed
lease of 60 seconds. A new Runtime becomes placement-eligible after it echoes
the first heartbeat acknowledgement, so allow roughly one heartbeat interval
before expecting a newly submitted Run to leave `preparing`.

## Run the artifact-copy example

In a third terminal at the repository root, upload and execute the fixture with
`curl` and `jq`. Use the same bearer token as the Server:

```shell
export CONTRACTOR_API_TOKEN='replace-with-a-local-api-token'
INPUT_REF="$(curl --fail --silent --show-error -X PUT \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  -H 'Content-Type: text/plain' -H 'If-None-Match: *' \
  --data-binary $'contractor-local-input\n' \
  http://127.0.0.1:8080/v1/artifacts/projects/source | jq -c .artifact)"

RUN_ID="$(jq -n --argjson source "$INPUT_REF" \
  '{workflow:"artifact-copy@2",parameters:{},artifacts:{source:$source}}' | \
  curl --fail --silent --show-error \
    -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
    -H "Idempotency-Key: local-run-$(date +%s)" \
    -H 'Content-Type: application/json' --data-binary @- \
    http://127.0.0.1:8080/v1/runs | jq -r .runId)"

curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "http://127.0.0.1:8080/v1/runs/$RUN_ID"

contractor --server http://127.0.0.1:8080 \
  run watch "$RUN_ID" --wait-timeout 30m
```

After the Run succeeds, download its frozen output:

```shell
curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "http://127.0.0.1:8080/v1/runs/$RUN_ID/outputs/result"
```

## Start the browser UI

In another terminal at the repository root:

```shell
make ui-build
CONTRACTOR_UI_API_BASE_URL=http://127.0.0.1:8080 \
  corepack pnpm --dir ui start
```

Open `http://127.0.0.1:4173` and sign in as `admin` with the password used to
create `local-auth.yaml`. The checked-in ServerConfig allows this exact browser
origin. The browser sends API and WebSocket traffic directly to the Go Server.
For deployment settings, see the [UI guide](../../deploy/ui/README.md).

For source-code Workflows, configure the additional
[workspace capabilities](project-workflows.md#runtime-workspace-requirements)
before starting the Runtime Agent.

For an explicitly trusted RFC 1918 development network, configure exact
IP-literal UI origins and the corresponding API origin. Keep Go Server on
loopback behind a local TCP forwarder and expose only its public HTTP port.
Public HTTP addresses and hostnames are rejected by this development mode.

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
  `contractor server config validate --root ./configs/e2e`.
- Gateway 404 responses usually mean the configured base omitted or duplicated
  the provider's `/v1` path.

## Model availability and queued Runs

A newly initialized Run is `pending` until the scheduler acquires its resources.
It becomes `running` at admission. If a started invocation loses its model, the
Run becomes `waiting`; its findings and conversation stay in that invocation.
Server coordinates recovery for the resolved owner/gateway/model/credential and
model proxy route, across Runtime processes and Server-side modeled planners.
Other Runs on that route remain queued. A single probe checks recovery.

These process settings are configurable under `ServerConfig.spec`:

```yaml
llmRecovery:
  requestTimeout: 60s
  initialDelay: 1s
  maxDelay: 30s
  automaticWindow: 5m
```

The defaults allow a short local model reload, cap each request at one minute,
use exponential retry delays from one to thirty seconds, and stop automatic
probes after five minutes. For a slower model, increase `requestTimeout` to cover
its expected response time. Matching environment variables are
`CONTRACTOR_LLM_RECOVERY_REQUEST_TIMEOUT`, `CONTRACTOR_LLM_RECOVERY_INITIAL_DELAY`,
`CONTRACTOR_LLM_RECOVERY_MAX_DELAY`, and `CONTRACTOR_LLM_RECOVERY_AUTOMATIC_WINDOW`.
CLI flags use the corresponding `--llm-recovery-...` names.

After the automatic window expires, restore the model and choose **Retry model
connection** on the Run page, or POST `{}` to `/v1/runs/{runId}/retry-gateway`.
This retries the existing model call; it does not create a new Stage attempt or
replay tools. Cancellation remains available while waiting. While Server is
unreachable, a waiting Runtime keeps reconnecting with jittered backoff (at most
5s apart) until its allocation lease expires; a Server that stays reachable but
keeps answering the recovery endpoint with 5xx for 120s fails the model call
with `recovery_authority_unavailable`. Reconnects appear in the Runtime log as
`Model recovery authority unavailable (...)` at most every 30s and in the
allocation metrics as `llm_recovery_authority_retries` counters. The configured Stage
wall-clock deadline starts at resource admission and continues during recovery;
queue residence does not consume it. Runtime process loss does not preserve the
in-memory conversation.

Transient failures include transport errors, HTTP 408/409/429/5xx and the
Gateway's declared `failureSignatures`; the `openai-compatible@1` default
covers the exact LM Studio model-unloaded HTTP 400 messages observed through
LiteLLM. An ordinary HTTP 400, authentication failure, exhausted quota or
context limit is a permanent error. A provider that reports availability with
another exact message (an Ollama `404 model not found`, for example) is
declared on a new Gateway version, as the commented example in
[`configs/llm-gateways/local_litellm.yaml`](../../configs/llm-gateways/local_litellm.yaml)
shows; declaring the block changes that Gateway's digest. Only safe error codes
and timing are retained; provider response bodies are not published in
recovery status.
