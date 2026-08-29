# Local development

The automated MVP test is the shortest proof that the actual Go Server and
Python Runtime Agent interoperate. It starts both production entry points,
creates a temporary deployment CA, uses an isolated PostgreSQL schema, and
replaces only the external OpenAI-compatible LLM Gateway with a deterministic
loopback fake.

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
```

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
- the bounded Worker `ExecutionReport` contains model, token, tool, and outcome
  counters;
- release removes the allocation route, in-memory Worker state and local
  workspace;
- private listeners reject a client that trusts the CA but has no certificate.

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
    -H 'Content-Type: application/json' --data-binary @- \
    http://127.0.0.1:8080/v1/runs | jq -r .runId)"

curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "http://127.0.0.1:8080/v1/runs/$RUN_ID"

curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "http://127.0.0.1:8080/v1/runs/$RUN_ID/outputs/result"
```

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

Using a real LiteLLM/provider is deliberately opt-in: point
`CONTRACTOR_LLM_GATEWAY_URL` and `CONTRACTOR_LLM_GATEWAY_TOKEN` at that gateway
and run the manual stack above. CI and `make test-e2e` never use provider
credentials or make an external model call.

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
