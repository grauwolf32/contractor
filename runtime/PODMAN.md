# Podman startup policy

V31-001 registers authoring/placement contracts and parses immutable startup
policy. It does **not** install an execution factory, invoke Podman, pull images
or advertise `podman@1` / `code-execution@1`. V31-002 adds a private engine
adapter, still without startup/allocation wiring or capability advertisement.
V31-003 adds the approved image and independent guardian/completion proof;
V31-004 adds the allocation lifecycle and startup recovery gate. Command tooling
and positive capability advertisement remain V31-005/V31-006; enabling policy
currently starts recovery, but does not advertise `podman@1` or `code-execution@1`.

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
The approved image entrypoint is the isolated inert PID 1 described below.
Only PID 1 runs as namespace root (a subordinate host UID), with no capabilities;
workload exec must explicitly select the nonzero keep-id UID/GID.
`--dns=none` is not supplied: Podman 6.1 rejects it with `--network=none`.

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
The engine alone does not guarantee Runtime-death cleanup: the V31-004 owner
below retains in-flight CLI ownership across Runtime death. Mock CLI tests do not establish effective kernel resource,
UID mapping, confinement or descendant/liveness guarantees; the real rootless
image/probe gates remain mandatory before advertisement.

Run `cd runtime && uv run pytest -W error tests/test_podman_engine.py tests/test_podman_contracts.py`.
The gate includes a scripted engine transport, real fake CLI child processes,
delayed spawn, bounded output, cancelled waiters and cross-process owner locks.
It does not create or delete real Podman containers.

Flag and exit-code references: [Podman create](https://docs.podman.io/en/latest/markdown/podman-create.1.html),
[local/remote engine selection](https://docs.podman.io/en/latest/markdown/podman.1.html),
[container exists](https://docs.podman.io/en/latest/markdown/podman-container-exists.1.html).

## Host guardian and image (V31-003)

The selected trust boundary is a separate, isolated host Python process holding
an inherited `SOCK_SEQPACKET` socket, a pinned exact container cgroup directory
and a PID 1 pidfd. No filesystem socket or secret token exists. These descriptors
are close-on-exec and explicitly passed **only** to the guardian; neither the
image nor Podman/workload children receive them. Container PID 1 has a different
UID from workload code, no capabilities, an immutable isolated-mode interpreter,
no command API and no project imports. It only waits and reaps orphans. Workload
code cannot signal/ptrace that PID 1, see host guardian PIDs, or write cgroup
controls through its read-only cgroup mount.

`open_fence` accepts an already ownership-verified running inspect record. It
pins only the exact `libpod-<full container ID>.scope` below this user's systemd
delegation, validates PID membership and subordinate UID, opens a real pidfd
and verifies freeze/kill write authority. Broader cgroup parents are never
destructive targets. Unsupported cgroup layouts fail; this MVP requires rootless
Podman with a systemd-managed cgroup v2 delegation. Python builds without
`os.pidfd_open` use the named libc API, not hard-coded syscall numbers or PID-only
liveness checks. These filesystem checks must run off the Runtime event loop
when allocation wiring is added.

Private protocol messages are bounded to 256 bytes and serialized:

| Request | Trusted acknowledgement | Meaning |
|---|---|---|
| Start with confirmed monotonic lease deadline | `ready` | Independent expiry is armed |
| `renew` with confirmed monotonic lease deadline | `renewed` | Deadline is min(confirmed lease, now + 10 s) |
| `check` after every launch has settled | `clean` | Frozen recursive cgroup inventory contained only the live, pinned PID 1 |
| `stop` | `stopped` | Entire cgroup is unpopulated and PID 1 has exited |

Freeze waits at most one second and never beyond liveness expiry. Inventory
includes descendant cgroups, irrespective of fork/reparent/double-fork/setsid
or inherited output descriptors. Only a clean, still-live scope is unfrozen.
A survivor, invalid packet, expired/missing renewal, Runtime socket EOF or
failed proof triggers `cgroup.kill` for the whole scope, independently of Podman
CLI responsiveness. Kernel-confirmed empty state, not CLI exit or process-group
signals, authorizes a stopped receipt. Unconfirmed termination produces no
success receipt. There is no restart or reconnect/revival protocol.

The guardian polls at 25 ms; the real gate exercises both finite expiry with a
living controller and SIGKILL of the Runtime-side controller without a successor.
This assumes the trusted guardian and kernel remain scheduled/functional; host
failure or uninterruptible kernel operations are not magically bounded. No
unconfirmed cleanup is advertised as successful. Deployment must preserve the
guardian long enough to enforce shutdown when stopping the Runtime service.

`CompletionGate.confirm` reuses `ExecutionResult`: it promotes a private
transport's completed result only after a guardian receipt **and** a fresh
ownership-verified running engine state. Workload output is never parsed as
control messages. Any exception permanently invalidates that gate and disconnects
the guardian to terminate execution. The caller retains the common workspace
guard until termination is confirmed; an exception is not permission to release
files or reuse the allocation. Guardian spawn owns duplicated descriptors and
reaps late children even after caller cancellation.

Lifecycle/executor integration obligations remain explicit:

- Hold the service owner, pin/hydrate content, start only the inert image and
  arm/verify the guardian before permitting any workload launch.
- Fence in-flight create/start/exec operations during recovery and teardown;
  do not treat this attachment primitive as a replacement for lifecycle ownership.
- Serialize commands and filesystem operations under the common workspace guard.
  A `check` is valid only after all launch attempts have settled: an unrelated
  host actor starting a new exec after the frozen inventory invalidates that proof.
- Run every workload with the nonzero keep-id UID/GID, no interactive stdin/PTY,
  fixed environment and bounded concurrent output capture. Feed completion only
  trusted command status, never output text. Podman's ambiguous 125/126/127 exit
  classes must remain infrastructure uncertainty unless separately proven.
- Convert confirmed lease deadlines once to monotonic time, renew more frequently
  than ten seconds and fail the allocation on renewal/protocol uncertainty.
- On timeout, cancellation, output overflow or uncertain launch, terminate the
  container and confirm kernel/engine cleanup before releasing workspace ownership.
  Remove the exact owned container before deleting mounted content.

Build/provision instructions are in [deploy/podman](../deploy/podman/README.md).
The mandatory task gates are `uv run pytest -W error tests/test_podman_supervisor.py`
from `runtime`, and `make test-podman-supervisor` from the repository root with
an explicit preinstalled digest-pinned `CONTRACTOR_TEST_PODMAN_IMAGE`. Passing
the ordinary suite's opt-in skips does not satisfy the real gate. The gate also
checks actual CPU throttling, memory OOM enforcement, PID exhaustion, tmpfs size,
network isolation, seccomp/no-new-privileges and mutual host/container file edits.

Kernel semantics: [cgroup v2 freezer, kill and populated state](https://docs.kernel.org/admin-guide/cgroup-v2.html).

## Allocation lifecycle and surviving owner (V31-004)

`PodmanWorkdirFactory` prepares ordinary private scratch; its path is never a
container ID. After project hydration, `AllocationService` records a private
`PreparedExecution` handle before awaiting container creation/start and guardian
readiness. Only then are selected tools and the Worker constructed. Identical
prepare reuses the allocation; conflicting prepare is rejected. Individual A2A
calls, including an `INPUT_REQUIRED` pause, do not stop or recreate its container.
Neither the Worker context nor ordinary tools receive lifecycle/engine handles.

Finalize rejects execution, lets the existing Worker termination/artifact path
finish and confirms container stop, retaining bind data. Abort and confirmed
lease loss remove the container before discarding project files. Release retains
one shielded cleanup task and removes container, project files, scratch and
adapters in that order. A timeout/cancelled response does not cancel ownership or
authorize idle capacity. Failed prepare also retains its context when removal is
uncertain, so authoritative release cannot bypass its outstanding resources.
The existing artifact write fence and release-confirmation edge are unchanged.

An independent host owner process runs the engine and holds its service flock.
Runtime communicates over two inherited, close-on-exec `SOCK_SEQPACKET` pairs:
serialized lifecycle RPC and a separately serviced health/rejection channel.
Packets are bounded to 8 KiB. There is no listening filesystem/network socket,
shell RPC, model-visible channel, inherited Runtime environment or stdout
protocol. Caller cancellation retains a late RPC response until it can be
drained; a subsequent remove cannot mistake a late prepare response for proof.

Runtime sends a pulse every 250 ms using `LeaseWatchdog.confirmed_deadline`.
The owner caps authority at the earlier of that confirmed lease and receipt
time plus three seconds; it may not renew its own liveness indefinitely. The
allocation-spec lease is the initial confirmed-lease snapshot, not a permanent
allocation TTL: successful subsequent heartbeat acknowledgements extend the
guardian's authority through the watchdog. Expired or replayed acknowledgements
do not revive execution. Runtime event-loop stalls therefore expire the guardian
even while the independent owner is alive.

On Runtime EOF the owner disconnects the guardian immediately, joins uncertain
engine operations, discovers/removes exact owned containers and only then
releases its flock. A late create may finish, but cannot proceed to start after
authority loss. An unconfirmed/absent create remains fenced; no second create or
global prune is issued. The owner never deletes workspace files. A successor
must acquire the same owner and complete recovery before scratch orphan cleanup,
workspace-provider initialization or capability discovery. Cleanup uncertainty
fails startup instead of advertising idle capacity.

Dedicated scratch/project roots retain `.contractor-podman-owner-v1`, outside
the mounted content. A disabled or differently configured owner cannot silently
delete those roots' predecessors. The marker is not automatically removed;
changing owner or disabling this deployment on the same roots requires an
operator migration after confirmed teardown. Embedders using built-in factories
receive the same project-provider recovery gate. Manually constructed providers
must supply their trusted `before_initialize` recovery hook.

Deployment must let both the host owner and guardian survive Runtime termination
long enough to finish their work; killing the whole service cgroup defeats that
assumption. Killing the owner/guardian themselves, arbitrary same-user host
mutation, host failure and indefinitely blocked kernel operations are not
bounded-success guarantees. No cleanup receipt is synthesized for uncertainty.
Service-manager configuration and positive capability probes remain later tasks.

Verification from `runtime`:

```sh
uv run pytest -W error tests/test_podman_allocation.py tests/test_podman_recovery.py
uv run pytest -W error tests/test_allocation.py tests/test_abort.py tests/test_lease_watchdog.py tests/test_projectfs_storage.py
CONTRACTOR_RUN_PODMAN_SUPERVISOR_GATE=1 uv run pytest -W error tests/test_podman_owner_integration.py
```

The last command additionally requires the explicitly preinstalled digest-pinned
`CONTRACTOR_TEST_PODMAN_IMAGE`. It proves owner-lock exclusivity, graceful owner
close, Runtime SIGKILL cleanup and Runtime SIGSTOP lease expiry with a real
workload writer. It never pulls/builds an image and does not delete bind data.
