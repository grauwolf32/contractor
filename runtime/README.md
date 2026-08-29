# Contractor Runtime Agent

The Runtime Agent is the Python, single-slot execution process for Contractor
v2. It registers a fresh process identity with the Go Control Plane, maintains
confirmed sequenced heartbeats, and exposes one private mTLS listener. Worker,
Google ADK, A2A 1.0 JSON-RPC, and artifact tools run together in that process;
each Runtime Agent has exactly one allocation slot.

```shell
uv sync
CONTRACTOR_CONTROL_PLANE_URL=https://localhost:8443 \
CONTRACTOR_ADVERTISED_CONTROL_URL=https://localhost:9443 \
CONTRACTOR_ADVERTISED_A2A_URL=https://localhost:9443 \
CONTRACTOR_CA_FILE=../.local/pki/ca.crt \
CONTRACTOR_CERTIFICATE_FILE=../.local/pki/agents/agent-local.crt \
CONTRACTOR_PRIVATE_KEY_FILE=../.local/pki/agents/agent-local.key \
uv run contractor-runtime --listen 127.0.0.1:9443
```

The listener requires both a deployment-CA client certificate and the reserved
Control Plane URI SAN before HTTP dispatch. Readiness remains false until the
listener is accepting and registration has succeeded.
