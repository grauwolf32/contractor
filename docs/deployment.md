# Deployment

[Documentation](README.md) · [Local walkthrough](guides/local-stack.md)

The baseline deployment is one host with PostgreSQL, one Contractor Server,
one or more Python Runtime Agents, and an optional Node UI. An external
OpenAI-compatible Gateway supplies the models. This guide covers the service
layout and installation sequence; the local walkthrough includes a first Run.

## Install the release

Build the commands and install locked Runtime/UI dependencies using
[development setup](development.md#build-the-commands). The following commands
assume `contractor` is on `PATH`. On a service host, install the built binaries,
configuration tree, Runtime environment and UI build under paths owned by its
service account; use absolute executable paths in the service manager.

Server and ordinary Runtime target Linux and macOS. The deterministic CI stack
runs on Linux; a cross-compiled binary alone does not verify another host.
Podman execution requires a separately provisioned Linux host with local
rootless Podman and the cgroup capabilities in the [Podman guide](../deploy/podman/README.md).

## Services and network

| Service | Baseline address | Access |
| --- | --- | --- |
| Server public API | `127.0.0.1:8080` | HTTPS reverse proxy for browsers and CLI |
| Server private Control Plane | `https://127.0.0.1:8443` | Runtime Agents with deployment mTLS identities |
| Runtime control and A2A | `https://127.0.0.1:9443` | Server with its Control Plane mTLS identity |
| Node UI | `127.0.0.1:4173` | HTTPS reverse proxy serving static UI |
| PostgreSQL | Operator-provided DSN | Server only |
| LLM Gateway | Configured API base | Server Planners and Runtime Workers |

Use separate same-site HTTPS origins such as `https://ui.contractor.example`
and `https://api.contractor.example`. The browser calls the API directly;
configure its exact UI origin in Server's `browserOrigins`. Forward public API
HTTP and WebSocket traffic to Server. Node receives the public API origin and
no bearer token or model credentials. See [UI deployment](../deploy/ui/README.md).

For remote Runtime hosts, change private listen/advertised addresses and issue
certificates whose SANs match those hosts. Use a unique Runtime certificate,
listener port and work root for each process. Keep private mTLS listeners off
the public reverse-proxy routes. A Runtime has one allocation slot; adding a
process adds capacity only for the capabilities it advertises.

## Identity, configuration and storage

For the one-host layout, initialize deployment identities and the local login
from the repository/release root:

```shell
contractor pki init-ca
contractor pki issue-control-plane
contractor pki issue-runtime --name runtime-local
umask 077
mkdir -p .local/secrets
contractor server auth hash-password \
  --user-id=local-user --username=admin > .local/secrets/local-auth.yaml
```

The PKI files live under `.local/pki`; keep the CA private key and leaf keys
protected. Password hashing reads the password from the terminal and stores
only its hash. Perform this bootstrap once for a fresh installation.

Create `.local/server.yaml` with non-secret process settings. Relative paths
are resolved from that file, so this example uses `../configs` for the catalog:

```yaml
apiVersion: contractor/v1alpha1
kind: ServerConfig
spec:
  listen: 127.0.0.1:8080
  privateListen: 127.0.0.1:8443
  privateUrl: https://127.0.0.1:8443
  operatorConfigRoot: ../configs
  managedConfigRoot: managed-configs
  localAuthFile: secrets/local-auth.yaml
  browserOrigins:
    - https://ui.contractor.example
  insecureLoopbackCookie: false
  trustedProxies:
    - 127.0.0.1
  caFile: pki/ca.crt
  certificateFile: pki/control-plane.crt
  privateKeyFile: pki/control-plane.key
  artifactBlobBackend: postgresql
```

Replace the example origin with the real proxy origin, and list the reverse
proxy's own source address in `trustedProxies` so failed logins are throttled
per browser client from `X-Forwarded-For` rather than per proxy; leave the list
empty when clients connect to Server directly. The
[local configuration](../configs/server.local.yaml) shows additional timeout
and diagnostics settings, with an explicitly insecure loopback browser mode.
Process settings follow `defaults < YAML < environment < flags` and require a
restart. [Timeout configuration](operations/timeout-configuration.md) owns the
settings table and independent deadline rules.

| Data | Persistence requirement |
| --- | --- |
| Execution, artifact metadata and PostgreSQL-backed payloads | Back up PostgreSQL |
| Filesystem-backed payloads | Back up the separate blob root along with its PostgreSQL metadata |
| Operator and managed configurations | Retain both roots; managed storage must support hard links and durable publication |
| Login, PKI and encrypted-credential master key | Retain owner-only files; the master key is needed to decrypt stored credentials |
| Runtime workspaces | Disposable allocation data; cleanup must finish before reusing a slot |

Choose the payload backend for a fresh installation. Changing the flag does
not migrate an existing store. PostgreSQL is the default; filesystem requires a
dedicated absolute path; S3 is not implemented. Storage, backup implications and
offline cleanup are described in [artifact blob storage](operations/artifact-blob-storage.md).

The Gateway endpoint and model aliases belong to immutable documents under
`configs/llm-gateways` and `configs/model-policies`. The shipped catalog uses
`local-litellm@1`, `planner-model` and `worker-model`; verify that both Server
and Runtime hosts can reach the selected Gateway. Tokens remain outside YAML.
For the shipped development credential selectors, set the bootstrap token
variables described in [the local walkthrough](guides/local-stack.md).
Their exact Gateway selector defaults to `local-litellm@1`. Override it with
ServerConfig `spec.developmentLlmGateway`, `CONTRACTOR_DEVELOPMENT_LLM_GATEWAY`
or `--development-llm-gateway`, using the same file/environment/flag precedence.
This binds only the development bootstrap tokens; ordinary managed credentials
keep their own pinned Gateway identity. Registry heartbeat and confirmed-lease
defaults remain 10 and 60 seconds and are not new ServerConfig settings.
For managed LiteLLM keys, configure the persistent encryption key and exact
admin binding in [Gateway credentials](operations/gateway-credentials.md), then
select the issued credential in the Workflow or Run execution configuration.

## LLM Gateway

The repository includes a small local proxy profile in
[`deploy/litellm/litellm_config.yaml`](../deploy/litellm/litellm_config.yaml).
It exposes `planner-model` and `worker-model`, both mapped to the configured
Qwen model in LM Studio. The run script uses digest-pinned LiteLLM
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
CONTRACTOR_LM_STUDIO_URL='http://lm-studio-host:1234/v1' \
  deploy/litellm/run.sh
```

`Ctrl-C` removes the proxy container but deliberately leaves
`contractor-litellm-postgres` and its named volume intact. The master key file
is also the `adminKeyFile` used by Contractor's exact Gateway binding; the salt
key remains LiteLLM-only. Neither file belongs in repository YAML.

## Start and verify

Supply the database DSN and, for CLI access, a bearer token through the service
environment. These example values must be replaced:

```shell
export CONTRACTOR_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor?sslmode=disable'
export CONTRACTOR_PUBLIC_BEARER_TOKEN='replace-with-an-api-token'
contractor server config validate --root ./configs
contractor server migrate
contractor server run --config .local/server.yaml
```

The DSN above assumes a local database; configure PostgreSQL TLS for a remote
database. Migrations are forward-only and run in one transaction. Each
statement is limited to 120 seconds and each lock wait to 10 seconds by
default; for a large upgrade raise them with `--statement-timeout` and
`--lock-timeout` (or `CONTRACTOR_MIGRATE_STATEMENT_TIMEOUT` and
`CONTRACTOR_MIGRATE_LOCK_TIMEOUT`, which the flags override). The lock timeout
must be shorter than the statement timeout, and the whole run is allowed at
least 15 minutes or the statement timeout plus about 5 minutes. These settings
apply only to `migrate`, not to ServerConfig database budgets. Keep Server
running while starting a Runtime in another terminal:

```shell
runtime/.venv/bin/contractor-runtime \
  --control-plane-url https://127.0.0.1:8443 \
  --advertised-control-url https://127.0.0.1:9443 \
  --advertised-a2a-url https://127.0.0.1:9443 \
  --listen 127.0.0.1:9443 \
  --ca-file .local/pki/ca.crt \
  --certificate-file .local/pki/agents/runtime-local.crt \
  --private-key-file .local/pki/agents/runtime-local.key \
  --work-root .local/runtime/runtime-local
```

For document-generation Workflows add the required
[local workspace and validator capabilities](guides/project-workflows.md#runtime-workspace-requirements)
before startup. Capabilities are probed once; restart the process after changing
its tool environment. [Runtime configuration](operations/runtime-configuration.md)
covers labels and adapter placement.

Start the built UI behind its HTTPS proxy:

```shell
make ui-build
CONTRACTOR_UI_API_BASE_URL=https://api.contractor.example \
  node ui/server/index.mjs
```

Verify Server `/readyz`, UI `/healthz`, browser login and the Runtime's confirmed
idle slot in Operations. Follow the [CLI guide](guides/cli.md) to configure a
context, then inspect `contractor check` and `contractor ops agents`. Execute
the artifact-copy example in the local walkthrough to check the full boundary.

## Upgrades and operational references

For a Server/private-contract change, pause new work, drain or cancel active
allocations and confirm release before replacing Server and Runtime together.
Back up durable state, apply migrations, restart and verify before resuming
admission. Follow [Runtime upgrade rules](operations/runtime-configuration.md#worker-session-mode-upgrade)
for immutable snapshots and rollback constraints. A compatible UI can be
rebuilt and restarted independently.

For running services, use the [operations index](operations/README.md) for
timeouts, credentials, metrics, profiling and cleanup. Use [testing](testing/README.md)
to choose release checks. Deployment examples under `deploy/` cover the
[independent UI](../deploy/ui/README.md), [Server blob storage without PVC](operations/artifact-blob-storage.md#deployment-without-pvc),
[Runtime labels](../deploy/runtime-labels/README.md) and [Podman](../deploy/podman/README.md);
they are component examples and require the remaining services above.
