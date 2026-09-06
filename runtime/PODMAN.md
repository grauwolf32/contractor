# Podman startup policy

V31-001 registers authoring/placement contracts and parses immutable startup
policy. It does **not** install an execution factory, invoke Podman, pull images
or advertise `podman@1` / `code-execution@1`. V31-002 adds a private engine
adapter, still without startup/allocation wiring or capability advertisement.
The supervisor, allocation lifecycle and full startup probes are subsequent tasks.

The profile requires an explicit Stage project workspace in `direct` mode and
local Runtime workspace storage. Ordinary `local-workdir@1` Workers keep their
existing memory/direct and overlay behavior, including in a mixed Stage.
Selecting `podman@1` alone adds no tools. `code-execution@1` selects only
`exec_command` and requires that exact profile; there is no host fallback.

## Operator settings

All options have environment equivalents: replace `--podman-` with
`CONTRACTOR_PODMAN_`, uppercase the suffix and replace hyphens with underscores.
For example, `--podman-command-max-seconds` is
`CONTRACTOR_PODMAN_COMMAND_MAX_SECONDS`. CLI values override environment values.

| CLI option | Default | Accepted bound/policy |
|---|---|---|
| `--podman-enabled` | `false` | Explicit `true` or `false` |
| `--podman-image` | unset | Required when enabled; at most 512 characters, `repository@sha256:` plus 64 lowercase hex digits |
| `--podman-owner` | unset | Required when enabled; stable unique service ID, `[a-z0-9][a-z0-9_-]{0,62}` |
| `--podman-cpus` | 2 | Finite positive number, at most 64 |
| `--podman-memory-bytes` | 2147483648 | Positive integer, at most 64 GiB |
| `--podman-pids` | 256 | Positive integer, at most 4096 |
| `--podman-tmpfs-bytes` | 268435456 | Positive integer, at most 4 GiB and no larger than memory |
| `--podman-command-max-seconds` | 300 | Integer, 60–3600; accommodates the tool's fixed 60-second default |
| `--podman-prepare-max-seconds` | 30 | Positive integer, at most 120 |
| `--podman-stop-grace-seconds` | 5 | Positive integer, at most 30 |
| `--podman-preview-bytes` | 32768 | Positive integer, at most 1 MiB per stream |
| `--podman-output-max-bytes` | 1048576 | Positive integer, at most 16 MiB; at least twice the preview limit |

Image repository names use the conservative grammar
`[a-z0-9]+(?:[._:/-][a-z0-9]+)*` before the digest suffix. Tags without a digest,
URLs, host paths, whitespace and credentials are rejected. Supplied policy is
validated even when disabled. Enabled policy also requires
`--workspace-storage local`; the existing workspace root settings still apply.
Parsing verifies syntax only: image availability and service-owner locking must
be verified by the later engine/probe implementation before advertisement.

There are no image/resource/environment fields in Workflow or model tool input.
Network is fixed to `none`, engine access is local rootless Podman, shell is
`/bin/sh -c`, and mounts/user mapping are operator-owned implementation policy.
No remote engine, host socket, arbitrary mount, host environment inheritance,
privileged mode or skill-script mount is introduced. The workspace bind has no
per-allocation hard disk quota; API acquisition limits do not enforce disk quota.

## Private execution boundary

`ExecutionRequest(command, cwd="", timeout_seconds=60)` bounds UTF-8 shell text
to 64 KiB and canonicalizes cwd using the existing workspace path grammar.
The future executor must also validate the on-disk directory under the shared
workspace guard: lexical validation alone does not exclude symlink races.
Its effective deadline must include lock wait and cleanup and be bounded by
request, operator, invocation and confirmed lease deadlines.

The `sandbox-execution` channel is distinct from `runtime-subprocess-launcher`.
The selected tool receives only `SandboxExecutor`; lifecycle code retains
`AllocationSandbox` with full ownership identity, lease renewal, stop and removal.
No lifecycle/engine handle is added to ordinary tools or the ADK Worker context.
These are private contracts, not active execution implementations.

The structured result has `status`, nullable `exitCode`, `stdout`, `stderr`,
`stdoutTruncated`, `stderrTruncated`, `durationMs` and nullable stable `errorCode`.
Nonzero program exit is still `completed`; infrastructure failures have no
trusted program exit code. Request/output and ownership fields are excluded
from diagnostic reprs. Preview limits count captured bytes before UTF-8
replacement; the private result ceiling allows the worst-case 3× replacement
expansion. The executor must enforce the configured capture limits.

Verification:

```sh
cd runtime
uv run pytest -W error tests/test_podman_settings.py tests/test_podman_contracts.py tests/test_capabilities.py
```

From the repository root, also run
`go test -count=1 ./internal/config ./internal/scheduler ./internal/controlplane`.

## Private engine adapter (V31-002)

`PodmanEngine` exposes `open`, `create`, `inspect`, `start`, `stop`, `remove`,
`discover` and `close`. Calls take absolute monotonic deadlines, with a 120-second
implementation ceiling and the configured shorter preparation ceiling for
creation. Future lifecycle wiring converts its lease/invocation deadlines once
and retains the workspace guard/content until removal is confirmed.

The adapter rejects a root/elevated host identity and checks local engine
rootless status. It uses `/usr/bin/podman --remote=false` with structured argv,
not a host shell. Host environment is constructed from the current passwd entry,
fixed executable search path, `/run/user/<uid>`, the corresponding local D-Bus
address and locale. Runtime secrets, proxies, remote-engine selectors and custom
environment variables are not copied. A private executable/transport seam exists
for deterministic tests, not for Workflow configuration.

Creation uses the pinned image with pull disabled, keep-id UID/GID mapping,
private namespaces, no network, dropped capabilities, no-new-privileges,
explicit resource limits and disabled persistent container logging. Image volumes,
health checks, proxy inheritance and default container environment are disabled.
The fixed container environment is PATH, HOME=/tmp and LANG. The only project
bind is the trusted hydrated `run_workdir` at `/workspace`; `/tmp` is bounded
tmpfs. Bind propagation is private and nonrecursive, with private SELinux
relabeling rather than disabled host confinement or recursive ownership changes.
The approved image entrypoint remains the responsibility of V31-003.

The service owner holds a nonblocking exclusive flock in the private directory
`/run/user/<uid>/contractor-podman-owners`. All services sharing a local engine
must use that same lock namespace. The directory and lock file must be owned by
the Runtime user and have modes 0700 and 0600. Symlinks, hard-linked lock files,
inode replacement and permission changes fail closed. Lock files are never
unlinked on release, avoiding concurrent owners locking different inodes.

Every create attempt gets a generated immutable creation ID before CLI launch.
Creation records five labels under `io.contractor.sandbox.`: managed=1, owner,
incarnation, allocation and creation. The generated name is
`contractor-<creation-id>`. Engine operations retain full container IDs and verify
all labels plus the exact name/ID from inspect. Discovery filters by owner and
managed label but independently verifies every returned container. It includes
predecessors for later recovery; they can be removed, not started as new work.

An uncertain create is reconciled by its original creation label, not replayed.
If no resource can yet be found, the attempt remains owned: absence does not
authorize a second create or unlocking the service. This deliberately sacrifices
availability on ambiguous failures. When discovery later finds the resource,
verified removal clears the attempt, even if create never returned its ID.
Repeated start does not restart exited work. Stop/remove exit codes alone are
not confirmation: inspect verifies termination/removal, and failed inspect only
means absence if `podman container exists` returns its documented code 1. Code
125 is uncertainty. No broad or forced removal is used; `ps --all` is read-only,
owner-filtered discovery. At most 1024 active attempts/discovered resources are
accepted; confirmed removal releases the active record and pinned descriptor.

Engine operations are serialized by an owning task. Cancelling/timing out its
caller leaves that task and the service lock retained, including delayed CLI
spawn and kernel reaping. Waiting callers have their own finite deadlines and
cannot launch after cancellation. CLI stdout/stderr are drained concurrently,
with a combined 1 MiB hard bound; stderr is discarded, never logged. A CLI
timeout/overflow closes pipes and kills/reaps the CLI child while its
leader is still alive. This is **not** evidence that its container has stopped.
An unconfirmed kernel operation prevents owner release rather than reporting a
reusable resource. Inherited output pipes are closed at the deadline; no group
signal is sent using a remembered PID after the CLI leader has exited.

This adapter does not claim isolation against arbitrary same-user host writers
replacing a bind path between checks. Only trusted allocation hydration code may
supply the root; pin checks complement, not replace, lifecycle/guard ownership.
It also does not yet guarantee Runtime-death cleanup: V31-003/V31-004 must fence
or terminate in-flight CLI operations as well as workload processes before
successor recovery. Mock CLI tests do not establish effective kernel resource,
UID mapping, confinement or descendant/liveness guarantees; the real rootless
image/probe gates remain mandatory before advertisement.

Run `cd runtime && uv run pytest -W error tests/test_podman_engine.py tests/test_podman_contracts.py`.
The gate includes a scripted engine transport, real fake CLI child processes,
delayed spawn, bounded output, cancelled waiters and cross-process owner locks.
It does not create or delete real Podman containers.

Flag and exit-code references: [Podman create](https://docs.podman.io/en/latest/markdown/podman-create.1.html),
[local/remote engine selection](https://docs.podman.io/en/latest/markdown/podman.1.html),
[container exists](https://docs.podman.io/en/latest/markdown/podman-container-exists.1.html).
