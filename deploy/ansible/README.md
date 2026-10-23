# Runtime Agent hosts with Ansible

`runtime.yml` and the `contractor_runtime` role provision Linux hosts with one
or more Runtime Agents (one allocation slot each) and every prerequisite of the
capabilities you choose per host or per process. The Server, PostgreSQL, the
UI and the LLM Gateway are out of scope; see [deployment](../../docs/deployment.md).

For an existing local Control Plane and cloud Runtime hosts, see the
[hybrid deployment scripts](../hybrid/README.md). They configure WireGuard
separately and generate an inventory for this role.

A run fails unless every process it started or restarted registered with the
Control Plane and reported an available probe for each capability it was
configured for.

## Requirements

- Controller: `ansible-core` 2.19 or later (the gate pins 2.21.5), `git`, and
  the repository checkout to deploy. Only committed content is deployed.
- PKI host (the controller by default): the `contractor` command and the
  deployment CA under `contractor_runtime_pki_root`
  (`contractor pki init-ca`). The CA key never leaves this host.
- Runtime hosts: Linux with systemd 248 or later (user managers reached with
  `systemctl --machine=<user>@.host --user`), cgroup v2 for Podman, `sudo`, and
  Python 3 for Ansible. Package groups ship for the Debian (Debian, Ubuntu),
  RedHat (Fedora, RHEL 9 and rebuilds) and Archlinux families; set
  `contractor_runtime_system_packages` for another one. `x86_64` and `aarch64`
  have pinned downloads.
- Network from Runtime hosts to GitHub, `nodejs.org`, the npm registry, PyPI
  and the base-image registry, or the mirrors described below.

## Run

```shell
cp deploy/ansible/inventory.example.yml ~/contractor-runtime.yml   # then edit
ANSIBLE_CONFIG=deploy/ansible/ansible.cfg \
  ansible-playbook -i ~/contractor-runtime.yml deploy/ansible/runtime.yml
```

Then check placement in **Operations → Runtime Agents** or with
`contractor ops agents`.

## What the role does, in order

1. **Resolve.** Validates the inventory and resolves each instance before
   touching a host: names, ports, capability switches, adapter decision,
   advertise address, values a unit file can carry literally, pinned downloads
   for the host's architecture, and instance names unique across the play.
2. **Packages and account.** Installs the package groups the host's
   capabilities need, creates the system account
   (`contractor_runtime_user`), its subordinate IDs, linger, and cgroup
   delegation of cpu, cpuset, io, memory and pids to user managers.
3. **Tools.** Downloads each needed tool by exact version, verifies its
   sha256, and installs it under `<prefix>/tools/<name>-<version>`; likec4 is
   installed with `npm ci` from the lockfile in
   `roles/contractor_runtime/files/likec4`.
4. **Release.** `git archive` of the resolved commit (Runtime sources, lock
   and `deploy/podman`), `uv sync --locked --no-dev` into
   `<prefix>/releases/<commit>` with a uv-managed Python, and the tree-sitter
   parser prefetch as the service account (`runtime/scripts/prefetch_parsers.py`).
5. **Podman.** Checks `podman info` (rootless, cgroup v2, systemd cgroup
   manager, seccomp) and `pasta`, then builds `deploy/podman` offline as the
   service account and uses the image's `repository@sha256` digest, unless
   `contractor_runtime_podman_image` names one. Prepares the image's keep-id
   layers with an unstarted provisioning container, without host binds or
   inherited credentials (300 s create budget, 30 s removal budget). This
   avoids spending the Runtime self-test budget on the first image copy.
   Only that exact provisioning container is removed. On uncertain failure,
   deployment stops and reports its name for reconciliation.
6. **Instances.** For each instance: issue its certificate once on the PKI
   host and copy it, create its roots, link the executables of its enabled
   capabilities into a bin directory named by their digest, run the Runtime's
   own `parse_settings` over the exact argv and the Runtime's tool probes
   under the instance's `PATH`, and write its user unit. When enabled, derive
   shared Podman scope bounds with the deployed parser and reject mixed
   CPU/memory/PID policies before changing the scope configuration. The scope
   option applies to a dedicated account only; setting it back to false leaves
   the existing host policy unmanaged. Remove that drop-in explicitly after
   draining the account if it is no longer needed.
7. **Start.** Stops and removes instances marked `state: absent`, then starts
   stopped instances and restarts changed ones one at a time, each verified
   from its journal before the next.
8. **Prune.** Removes releases beyond `contractor_runtime_keep_releases`.

## Capabilities

Effective switches are `contractor_runtime_capabilities_default`, overridden
by `contractor_runtime_capabilities` (host or group) and then by an instance's
`capabilities`:

| Switch | Values | Gives | Needs on the host |
| --- | --- | --- | --- |
| `workspace_storage` | `local`, `memory`, `none` | filesystem@1, edit-files@1, workspace-changes@1 and the Trailmark tools of code-analysis@1 (local only) | `local`: a workspace root per instance |
| `openapi_validator` | bool | `validate_openapi` in openapi@1 | vacuum |
| `likec4_validator` | bool | `validate_likec4` in likec4@1 | Node and the likec4 CLI |
| `scanners` | subset of `nuclei`, `sqlmap`, `naabu`, `ffuf`, `katana` | the matching `scan_*` operations of scan@1 | the scanners; nuclei templates for nuclei |
| `podman` | bool | podman@1 and code-execution@1 | rootless Podman, subordinate IDs, delegated controllers, the supervisor image; requires `local` storage |
| `podman_network` | bool | podman-network@1 | pasta; requires `podman` |

code-analysis@1 and taint-annotations@1 need only the release and the parser
cache, so every instance has them.

A capability is advertised whenever its executable is found, so each instance
gets its own `PATH`: its bin directory followed by
`contractor_runtime_system_path` (default `/usr/bin:/bin`). The preflight fails
when a disabled scanner or validator is still found there, for example a
distribution `sqlmap` in `/usr/bin`; remove it or narrow the system path.

## Variables

`roles/contractor_runtime/defaults/main.yml` documents every variable. The
ones most inventories set:

| Variable | Default | Meaning |
| --- | --- | --- |
| `contractor_runtime_control_plane_url` | required | The Server's private Control Plane URL (`https://…:8443`). |
| `contractor_runtime_instances` | one instance `<host>-1` on port 9443 | List of `{name, port, state, …}`; any per-instance key below. |
| `contractor_runtime_listen_host` | `127.0.0.1` | Listen address. A wildcard (`0.0.0.0`, `::`) needs `advertise_host`. |
| `contractor_runtime_advertise_host` | listen host | Address the Server dials; part of the certificate SANs. |
| `contractor_runtime_capabilities` | `{}` | Host or group overrides of the switches above. |
| `contractor_runtime_labels` | `[]` | Startup labels; they seed a previously unseen principal only. Bind each label on the Server first (**Operations → Runtime labels**): an unbound one fails registration with `invalid_runtime_labels`. |
| `contractor_runtime_adapters` | `null` | `null` probes every built-in adapter; a list selects a subset. |
| `contractor_runtime_allowed_target_networks` | `[]` | CIDRs passed as `--allowed-target-network`. |
| `contractor_runtime_workspace_limits`, `contractor_runtime_podman_limits` | `{}` | `--workspace-<key>` / `--podman-<key>` settings, e.g. `{cpus: 4}`. |
| `contractor_runtime_podman_scope_limits` | `false` | Set CPU, memory, swap and PID bounds on all `libpod-*.scope` units of the dedicated Runtime account. Needed on Ubuntu 24.04 with Podman 4.9. All Podman instances in that account must have identical CPU, memory and PID limits; the deployed Runtime parser supplies defaults and validates this before the scope policy changes. |
| `contractor_runtime_extra_args`, `contractor_runtime_extra_environment` | empty | Further Runtime flags and environment. |
| `contractor_runtime_version` | `HEAD` | Commit-ish of `contractor_runtime_source_repo` to deploy. |
| `contractor_runtime_pki_host`, `contractor_runtime_pki_root`, `contractor_runtime_pki_command` | `localhost`, `<repo>/.local/pki`, `contractor` | Where certificates are issued. |
| `contractor_runtime_restart` | `true` | Restart running instances whose settings, executables, certificate or release changed. |
| `contractor_runtime_verify`, `contractor_runtime_verify_timeout_seconds` | `true`, `180` | Journal verification after each start. |

Per instance you can set `listen_host`, `advertise_host`, `cert_dns`,
`cert_ips`, `labels`, `adapters`, `allowed_target_networks`, `log_level`,
`extra_args`, `extra_environment`, `workspace_limits`, `podman_limits`,
`podman_owner` and `capabilities`.

The role checks only what it interprets itself. Labels, adapters, CIDRs and
Podman or workspace bounds are validated by the Runtime's own parser during the
preflight, so a bad value fails before any restart with the Runtime's message.

## Restarts and draining

A restart aborts the allocation the slot is running. The role restarts only
instances whose unit, executables, certificate or release changed, one at a
time, but it does not drain them. Either drain the Runtime Agent first, or set
`contractor_runtime_restart: false`: changed instances then keep running their
previous settings, the run lists them, and releases are not pruned until they
were restarted. The deferral is recorded under
`<config_dir>/restart-pending/<name>`, so any later run with
`contractor_runtime_restart: true` restarts and verifies exactly those
instances, even when nothing else changed. A manual
`systemctl --user restart` leaves the record, and the next run restarts the
instance once more.

The bin directory of a running instance never changes: a different tool set is
a new directory and a new `PATH` in the unit, and the old directory is removed
after the restart.

Changing cgroup delegation restarts the account's user manager, which stops
its running instances; the start phase starts them again.

## Certificates

Each instance's certificate is issued once with
`contractor pki issue-runtime --name <name>` and never re-issued, because a new
key changes the Runtime principal and its Operations labels. SANs are fixed at
issue time from `cert_dns`/`cert_ips` or, when empty, from the advertise host
plus `localhost`, `127.0.0.1` and `::1`. To change them, re-issue on the PKI
host (`--force`, a key rotation) and rerun the playbook. Renewals
(`contractor pki renew-runtime --name <name>`) keep the key; the next run
copies the renewed certificate and restarts the instance.

The PKI tasks run on the PKI host as the controller user with `throttle: 1`.

## Removing an instance

Set `state: absent` on its entry. The unit is stopped, disabled and removed;
its certificate, key and roots stay, because a work-root lock or Podman owner
marker must never be removed implicitly and reusing the name would reuse the
principal. Delete them by hand once the instance is gone for good.

## Mirrors and air-gapped hosts

- Override download URLs (keeping each `sha256`) through
  `contractor_runtime_tools`, which is merged recursively over
  `contractor_runtime_tools_default`.
- `uv` installs Python and the packages pinned in `runtime/uv.lock`, whose
  sources name PyPI. `contractor_runtime_uv_environment` passes uv's own
  settings, for example `UV_PYTHON_INSTALL_MIRROR`, `HTTPS_PROXY` or
  `SSL_CERT_FILE`; a package mirror must serve the locked distributions.
- Set `contractor_runtime_podman_pull_base_image: false` and load the base
  image named in `deploy/podman/Containerfile` into the service account's
  store, or provide `contractor_runtime_podman_image` as `repository@sha256:…`.
- Point `contractor_runtime_nuclei_templates_dir` at templates you manage.

## Updating pins

Each entry of `contractor_runtime_tools_default` names a version, per
architecture asset URL and sha256, the extracted executable `path` and the
archive's `strip` depth. Keep scanner versions inside what the Runtime accepts
(for example katana 1.7.x or 1.8.x). To move likec4, change its version in the
defaults and regenerate the lockfile with the pinned Node:

```shell
cd deploy/ansible/roles/contractor_runtime/files/likec4
# edit package.json, then:
npm install --package-lock-only --ignore-scripts
```

## Tests

`make test-ansible` (also part of `make lint`) runs the syntax check,
ansible-lint, the controller-only resolution tests in `tests/resolve.yml`, and
the tool probe against the locked Runtime environment.
