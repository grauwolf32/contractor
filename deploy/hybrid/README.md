# Local Control Plane and cloud Runtime Agents

These scripts extend an existing local Server deployment. They reuse
`deploy/ansible` for Runtime installation and the existing deployment CA. They
do not deploy another Server or migrate the Contractor database.

Default topology:

```text
Mac 10.77.0.10                  cloudru1 10.77.0.11
  existing Server                one Runtime slot :9443
  PostgreSQL                     LiteLLM :4000 -> isolated wg-egress -> OpenRouter
  local Runtime Agents           persistent LiteLLM PostgreSQL
  mesh :8443 -> loopback :8443
             \
              WireGuard mesh -- cloudru2 10.77.0.12
                                  one Runtime slot :9443
```

The mesh has individual keys and /32 routes; it does not replace any host's
default route. LiteLLM alone uses the provider VPN profile. One provider profile
must not be brought up on two hosts simultaneously.

Requirements: SSH aliases with verified host keys, Ubuntu with passwordless
remote sudo, local `wg`, `wg-quick`, `wireguard-go`, `uv`, `git`, `openssl`, and
the existing `contractor` binary and CA. Use Python 3.11 or newer. `server.py`
also needs PyYAML 6.0.3; `uv run --script deploy/hybrid/server.py ...` supplies it.
Run commands from the repository root. All `--state` options precede the
subcommand; the default is the gitignored `.local/hybrid/` directory.

## Bring up the network

```sh
python3 deploy/wireguard/mesh.py init cloudru1 cloudru2
python3 deploy/wireguard/mesh.py apply
# Only for Cloud.ru images with the unavailable vendor Ubuntu mirror:
# python3 deploy/wireguard/mesh.py apply --ubuntu-mirror https://ru.archive.ubuntu.com/ubuntu
python3 deploy/wireguard/mesh.py local-up
python3 deploy/wireguard/mesh.py status
python3 deploy/wireguard/egress.py apply --config /absolute/path/provider.conf
python3 deploy/wireguard/egress.py check-kill-switch
```

`init` refuses an existing deployment; `render` preserves existing private keys.
The local `local-up` step asks for sudo in your terminal. On macOS the actual
interface is named `utunN`, even though the configuration is `wg-mesh.conf`.
Remote peers listen on UDP 51821; allow this in the cloud security group. The
Mac can be behind NAT and initiates connections with keepalives. On cloudru1
the provider tunnel additionally needs outbound UDP to its configured endpoint.

Runtime and LiteLLM listen on the mesh; the public Server API stays on loopback.
The namespace uses 172.30.91.0/30; choose a different subnet in the scripts
before installation if this overlaps an existing route. The current scripts
assume this fixed private subnet and a stable mesh topology. To add peers,
render and reapply all peer configurations, then restart remote `wg-quick`
units during a maintenance window so new routes are installed.

## Bring up the gateway

```sh
python3 deploy/hybrid/gateway.py apply --openrouter-key-file /absolute/path/openrouter-key
python3 deploy/hybrid/gateway.py check
python3 deploy/hybrid/bootstrap.py tunnel
python3 deploy/hybrid/bootstrap.py --token-file /absolute/path/api-token publish
```

The OpenRouter key file must be owner-only (0600). Images use the existing
digest pins and model aliases from `deploy/litellm`. Existing model selections
are preserved. The gateway is shared by the cloud workers. Its persistent
PostgreSQL volume survives service restarts; do not rotate its database
password by editing a file alone.

LiteLLM's management origin is a loopback SSH forward at 127.0.0.1:14000.
This meets the Server's HTTPS-or-loopback management contract while inference
uses HTTP inside WireGuard. This SSH process must remain running for managed
key provisioning, revocation and reconciliation. Rerun `bootstrap.py tunnel`
after losing the connection or rebooting the Mac. Supervise this process and
the local ingress before relying on unattended operation.

## Prepare the existing local Server

```sh
uv run --script deploy/hybrid/server.py prepare \
  --config /absolute/path/existing/server.yaml \
  --pki-root /absolute/path/existing/pki \
  --contractor /absolute/path/existing/bin/contractor
```

This writes a separate Server profile and a separate leaf certificate with the
mesh IP, signed by the existing CA. The original configuration and certificates
are preserved. The private listener stays on 127.0.0.1:8443, so existing local
Runtime URLs continue to work. A bounded raw TCP ingress exposes that same
mTLS listener on 10.77.0.10:8443 without terminating TLS.

For the optional guarded restart helper, create an owner-only JSON file:

```json
{
  "values": {"PGPASSFILE": "/absolute/path/pgpass", "CONTRACTOR_DATABASE_URL": "postgres://user@127.0.0.1:5432/database?sslmode=disable"},
  "files": {
    "CONTRACTOR_PUBLIC_BEARER_TOKEN": "/absolute/path/api-token",
    "CONTRACTOR_LLM_GATEWAY_TOKEN": "/absolute/path/worker-inference-key",
    "CONTRACTOR_PLANNER_LLM_GATEWAY_TOKEN": "/absolute/path/planner-inference-key"
  }
}
```

Use the environment variable names from your existing deployment. `files` reads
values into process memory; it does not put them in command arguments.

```sh
uv run --script deploy/hybrid/server.py restart \
  --contractor /absolute/path/existing/bin/contractor \
  --pid-file /absolute/path/existing/server.pid \
  --environment /absolute/path/server-environment.json \
  --token-file /absolute/path/api-token
```

The helper requires an idle queue, idle Runtime Agents, an expected Server PID,
and the local mesh address before stopping anything. It reuses the binary and
database; it does not execute migrations. Failed readiness starts the original
configuration again. An interrupted restart still requires operator inspection.

## Install and route the cloud workers

```sh
python3 deploy/hybrid/bootstrap.py --token-file /absolute/path/api-token configure
python3 deploy/hybrid/runtime.py \
  --contractor /absolute/path/existing/bin/contractor \
  --pki-root /absolute/path/existing/pki \
  --version FULL_RUNTIME_COMMIT
python3 deploy/hybrid/bootstrap.py --token-file /absolute/path/api-token assign \
  --pki-root /absolute/path/existing/pki
```

The Runtime revision must contain `runtime/scripts/prefetch_parsers.py`, required
by the current Ansible role. Pin a compatible Server/Runtime revision; do not
assume arbitrary versions interoperate. The wrapper checks machine identities
and remote reachability of the local private API before deployment.

Start with one slot per host: these VPS have only 2 CPUs and 4 GB RAM, and
cloudru1 also hosts LiteLLM. The default profile provides source analysis with
local workspaces and tree-sitter parsers. It omits Podman, external scanners,
and external OpenAPI/LikeC4 validators. `--full-capabilities` installs the full
Ansible profile when needed. A label configures execution settings; it does not
by itself restrict scheduling to particular hosts.

To enable all five scanners, both validators, and offline/networked rootless
Podman on the existing cloud hosts, rerun the deployment with the same pinned
revision and add `--full-capabilities`:

```sh
python3 deploy/hybrid/runtime.py \
  --contractor /absolute/path/existing/bin/contractor \
  --pki-root /absolute/path/existing/pki \
  --version FULL_RUNTIME_COMMIT --full-capabilities
```

Keep that flag on subsequent wrapper runs; omitting it selects the minimal
profile again. Drain the affected slots before applying changed capabilities.
Ansible verifies the startup probes, and the wrapper checks every mTLS
readiness endpoint, including unchanged instances. For a test on one specific
cloud host, a Runtime label alone provides no scheduling affinity: temporarily
stop other compatible idle cloud slots, wait for their Control Plane leases
to expire, and restore them after the test. Local Runtime Agents without a
Podman capability cannot receive a Podman Workflow.

New cloud principals register with the initial `cloud-llm` label. Create it
before starting the workers: registration rejects an unknown label, so a new
worker cannot accidentally receive the local loopback gateway configuration.
For an already registered principal, labels are authoritative Server state;
changing startup arguments does not replace them. Use `assign` to reconcile it.

`configure` provisions a managed inference credential through Contractor with a
$5 budget, 30 RPM and at most two parallel requests. It reads the exact Worker,
summarizer and Audit completion policies from the Server's current model choices
(including role overrides), rather than assuming the base `worker@2` alias is
the selected policy. `--model-policies` permits an explicit selection. It
initially publishes `cloud-llm@1` and its label. `assign` adds that label
only to the cloud certificate principals and preserves unrelated labels.
Existing identities with different policy/configuration fail instead of being
silently overwritten. Change budget or routing through explicit new identities
or the normal Operations UI. For example:

```sh
python3 deploy/hybrid/bootstrap.py --token-file /absolute/path/api-token configure \
  --credential-id cloud-workers-selected --config-version 2 --rebind-label
```

The explicit rebind uses the label's current revision. Successful routing
identities are remembered in `cloud-routing.json`, so subsequent plain
`configure` calls check that same route. A future change to selected models can
require another credential/config version; it never changes model defaults.

## Secrets and recovery

Generated WireGuard keys and gateway master/salt/database keys live in
`.local/hybrid/` with mode 0600, in directories created with mode 0700. Remote
gateway secrets live in root-owned `/etc/contractor-hybrid/*.env` (0600).
Runtime private keys belong to the dedicated Runtime service account; the CA
private key stays on the local machine. Runtime inference keys are managed by
Contractor and stored using its existing encryption mechanism.

Create an encrypted backup; keep its password in a password manager:

```sh
python3 deploy/hybrid/vault.py seal /absolute/path/hybrid-secrets.vault
# Restore into an empty state directory; keep deployment.json separately:
python3 deploy/hybrid/vault.py --state /absolute/path/restored-state restore /absolute/path/hybrid-secrets.vault
```

The vault contains `secrets/` and `wireguard/` only. Separately back up the
deployment manifest, the original Contractor CA, Server credential master key,
Contractor database, and LiteLLM PostgreSQL volume. No encrypted backup exists
until `seal` succeeds. Never commit plaintext state, `.env` files or VPN profiles.

Stopping services does not remove persisted data. For rollback, stop the local
ingress PID recorded in state and start the previous Server binary/config from
`server-previous.json` with the original environment. Stop remote Runtime user
units before taking down the mesh. On cloudru1 stop LiteLLM and its PostgreSQL
before stopping `contractor-wg-egress`. Namespace cleanup removes only resources
owned by these scripts; it does not delete either database volume.

## Checks

```sh
make test-hybrid
make test-ansible
```

The deployment tests use fixture keys and a temporary Vault password, never the
live `.local/` state. The Vault round trip requires `uvx` and ansible-core 2.21.5.
