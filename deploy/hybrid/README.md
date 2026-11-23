# Local Control Plane and cloud Runtime Agents

These scripts extend an existing local Server deployment. They reuse
`deploy/ansible` for Runtime installation and the existing deployment CA. They
do not deploy another Server or migrate the Contractor database.

Illustrative topology (choose your addresses and SSH aliases):

```text
Controller 10.42.0.10           worker-a 10.42.0.11
  existing Server                one Runtime slot :9443
  PostgreSQL                     LiteLLM :4000 -> isolated wg-egress -> OpenRouter
  local Runtime Agents           persistent LiteLLM PostgreSQL
  mesh :8443 -> loopback :8443
             \
              WireGuard mesh -- worker-b 10.42.0.12
                                  one Runtime slot :9443
```

The example host aliases must exist in your local SSH configuration.

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
python3 deploy/wireguard/mesh.py init worker-a worker-b \
  --network 10.42.0.0/24 --egress-network 172.25.18.0/30 \
  --egress-check-url https://provider.example/health
python3 deploy/wireguard/mesh.py apply
# Optional: replace one exact unavailable Ubuntu mirror (both URLs required):
# python3 deploy/wireguard/mesh.py apply \
#   --ubuntu-mirror-source https://old-mirror.example/ubuntu \
#   --ubuntu-mirror https://new-mirror.example/ubuntu
python3 deploy/wireguard/mesh.py local-up
python3 deploy/wireguard/mesh.py status
python3 deploy/wireguard/egress.py apply --config /absolute/path/provider.conf
python3 deploy/wireguard/egress.py check-kill-switch
```

`init` refuses an existing deployment; `render` preserves existing private keys.
The local `local-up` step asks for sudo in your terminal. On macOS the actual
interface is named `utunN`, even though the configuration is `wg-mesh.conf`.
On macOS, `local-up` also restores an explicit route for the deployment's private
mesh /24. It discovers the interface by the controller address, verifies its
WireGuard public identity, and checks that each peer routes through it. It also
reconciles one exact host route for the controller's own mesh address through
`lo0` (for example, `10.42.0.10/32`). Local Runtime calls to the advertised
private API can then use local delivery, while remote peer routes still use
the mesh interface. Both route classes are read back before reporting readiness.
On macOS the commands pass `-ifp` to select the actual output interface as well
as `-interface` to set the directly reachable link gateway. A changed gateway
alone does not prove that an existing route moved to another interface.
Repeating the command repairs routing without restarting an existing matching CLI tunnel:
run it again after activating a personal VPN that captures mesh traffic. The
personal VPN, default route, DNS and keys are unchanged. A conflicting or
unverifiable interface, denied administrator privileges, or a failed route check
fails the command. Graphical-client tunnels remain manually managed.

Stop the CLI mesh with `python3 deploy/wireguard/mesh.py local-down`. This helper
verifies the mesh identity and removes the controller's exact `lo0` host route
before invoking `wg-quick down` to remove the tunnel and its interface routes.
It also removes a leftover controller host route when the mesh is already down;
repeating teardown is safe. If the interface has an unexpected identity, no
routes are changed. Failed or unverified host-route removal leaves the mesh
running for an explicit retry. Direct `wg-quick down` does not own the separate
loopback route; rerun `local-down` to clean it after an out-of-band teardown.
All these routes are temporary and do not survive a reboot.
There is no background route monitor for VPN changes or automatic activation
added by this helper; rerun `local-up` after such a change. Linux cloud startup
continues to use `wg-quick@wg-mesh` and the per-peer /32 routes.
Remote peers listen on UDP 51821; allow this in the cloud security group. The
Mac can be behind NAT and initiates connections with keepalives. On the gateway host
the provider tunnel additionally needs outbound UDP to its configured endpoint.

Runtime and LiteLLM listen on the mesh; the public Server API stays on loopback.
The mesh and namespace networks are explicit operator inputs recorded in
`.local/hybrid/deployment.json`; the values above are examples. Replace the
probe URL with a real HTTPS endpoint reachable through the provider VPN.
The manifest shape is illustrated in [deployment.example.json](deployment.example.json).
The namespace requires a private IPv4 /30 outside the private mesh /24; choose
both to avoid existing host routes. Its first two usable addresses become the
host and namespace addresses across routing, firewall, database and SSH tunnel.
Do not change an installed namespace network in place: stop its dependent services
and tear down its owned network resources before installing a different network.
The current scripts assume a stable mesh topology. To add peers,
render and reapply all peer configurations, then restart remote `wg-quick`
units during a maintenance window so new routes are installed.

## Bring up the gateway

```sh
python3 deploy/hybrid/gateway.py apply --openrouter-key-file /absolute/path/openrouter-key
python3 deploy/hybrid/gateway.py check
python3 deploy/hybrid/bootstrap.py tunnel
python3 deploy/hybrid/bootstrap.py --token-file /absolute/path/api-token publish
```

The provider profile must declare its DNS servers explicitly.
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
mTLS listener on 10.42.0.10:8443 without terminating TLS.

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
uv run --locked --project runtime python deploy/hybrid/runtime.py \
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

`runtime.py` uses the locked Runtime environment to validate location labels
with the same grammar as registration. `mesh.py init` explicitly writes
`"placement_labels": ["cloud"]` on each host in `deployment.json`. Before
deploying, review these declarations in that source manifest. Use, for example,
`["cloud/aws"]` for an AWS host or `["cloud/hetzner/site-a"]` for a narrower
location. Each host may declare its own location; its slots share it. No provider
is inferred from its address or SSH alias. Labels must be sorted, unique and
form one ancestor chain (at most 32 paths of 63 ASCII bytes). An explicit empty
list declares no location and cannot serve a check constrained to `cloud`.
Missing, null or malformed declarations fail before inventory replacement or
remote changes; add the explicit field to a manifest that lacks it.

The wrapper reads those source declarations on every invocation, including
`--inventory-only` and `--syntax-check`; do not edit `runtime-inventory.yml`.
They become `--placement-label` arguments independently of the Server-managed
`cloud-llm` RuntimeConfig assignment. Changing a declaration requires an ordinary
Runtime restart/new incarnation. With `--defer-restarts`, ready processes still
have their previous labels until restarted; Operations shows their actual
registered declarations. Drain affected slots and rerun without that flag to
apply the new declaration. A failed or interrupted deployment can be rerun from
the same source manifest without changing the desired location.

Start with one slot per host and size additional slots against measured CPU
and memory capacity, including LiteLLM on the gateway host. The default profile provides source analysis with
local workspaces and tree-sitter parsers. It omits Podman, external scanners,
and external OpenAPI/LikeC4 validators. `--full-capabilities` installs the full
Ansible profile when needed. A label configures execution settings; it does not
by itself restrict scheduling to particular hosts.

To enable all five scanners, both validators, and offline/networked rootless
Podman on the existing cloud hosts, rerun the deployment with the same pinned
revision and add `--full-capabilities`:

The full wrapper profile enables exact systemd scope bounds for the dedicated
Runtime account (needed by Ubuntu 24.04 / Podman 4.9). CPU, memory and PID
defaults come from the deployed Runtime parser; slots sharing that account
must use the same resource bounds. Ansible prepares cold keep-id image layers
before starting the Runtime, without starting a container or binding host data.

```sh
uv run --locked --project runtime python deploy/hybrid/runtime.py \
  --contractor /absolute/path/existing/bin/contractor \
  --pki-root /absolute/path/existing/pki \
  --version FULL_RUNTIME_COMMIT --full-capabilities
```

Keep that flag on subsequent wrapper runs; omitting it selects the minimal
profile again. Drain the affected slots before applying changed capabilities.
Ansible verifies the startup probes, and the wrapper checks every mTLS
readiness endpoint, including unchanged instances. To constrain a test to a
particular declared site, use the ordinary check label `cloud:aws/site-a` with
a host declaration such as `cloud/aws/site-a`. The `cloud-llm` RuntimeConfig
label alone provides no scheduling affinity. Local Runtime Agents without a
Podman capability cannot receive a Podman Workflow.

New cloud principals register with an empty configuration-label set. After
registration, use `assign` to apply `cloud-llm` through Operations before
submitting work to those agents. Keep affected work paused during provisioning:
registration alone does not establish the cloud Gateway route. For an already
registered principal, `assign` reconciles the Server-owned labels; changing
Runtime startup arguments never assigns or replaces them.

`configure` provisions a managed inference credential through Contractor with a
$5 budget, 30 RPM and at most two parallel requests. It reads the exact Worker,
summarizer and Audit completion policies from the Server's current model choices
(including role overrides), rather than assuming the base `worker@1` alias is
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

Bootstrap derives a stable mutation key from the HTTP method, resource, canonical
body and `If-Match`/`If-None-Match` preconditions. Restoring an earlier binding or
principal label set at a later revision therefore starts a distinct operation;
an exact retry retains its original key. Explicit caller keys remain unchanged.
After a timeout or lost response, rerun `configure` or `assign`: each first reads
the current resource and only writes if the requested state is still absent.
Cancellation does not undo a possibly committed write. Stale revisions,
validation errors and idempotency conflicts remain failures; bootstrap does not
discard the precondition or invent a fresh key to hide them. Requests retain the
30-second timeout and bounded response reads, and error output suppresses
provider bodies and credentials.

## Secrets and recovery

Keep concrete SSH aliases, addresses, mirror mappings, local launch commands
and operational reports in a private `.local/` runbook. The repository contains
reusable scripts and illustrative examples; it must not become the record of a
particular installation. The shared `deploy/models/` catalog remains versioned.

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
units before taking down the mesh. On the gateway host, stop LiteLLM and its PostgreSQL
before stopping `contractor-wg-egress`. Namespace cleanup removes only resources
owned by these scripts; it does not delete either database volume.

## Checks

```sh
make test-hybrid
make test-ansible
```

The deployment tests use fixture keys and a temporary Vault password, never the
live `.local/` state. The Vault round trip requires `uvx` and ansible-core 2.21.5.
