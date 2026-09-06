# 21 — Allocation-scoped Podman execution sandbox

Status: **Contracts/settings and private engine implemented in V31-001/V31-002; execution profile not enabled**

Depends on: [01](01-agent-template.md), [02](02-runtime-and-a2a.md),
[04](04-execution-lifecycle-and-metrics.md),
[09](09-agent-skills.md), [10](10-runtime-filesystems-and-edit-tools.md).

## Purpose and scope

One local Runtime Agent creates one rootless Podman container when preparing
an allocation that selects `podman@1`. The container shares that allocation's
local direct project directory. Explicit execution tools run commands inside
it; finalization stops it and release removes it before deleting mounted files.

The ADK Worker, A2A listener, Artifact client and existing trusted Toolsets
continue to run in the Runtime process. This profile isolates executed code;
it does not move or sandbox the whole Runtime Agent. It does not replace
Scheduler, introduce another queue or create an independent Worker lifecycle.

The first version uses a single container, not a multi-container Podman pod.
Remote engines, Daytona, sidecars, interactive terminals, background services
and execution of skill scripts are outside this version.

## Configuration and compatibility

AgentTemplate selects registered behavior using existing exact selectors:

```yaml
spec:
  sandboxProfile: podman@1
  toolsets:
    - ref: filesystem@1
      tools: [ls, read_file, grep]
    - ref: edit-files@1
      tools: [write_file, edit]
    - ref: code-execution@1
      tools: [exec_command]
```

These are registered authoring refs, not currently installed capabilities. Selecting the
profile does not implicitly add execution or filesystem tools. The profile
may be selected without `exec_command`, but still owns its container lifecycle.
Selecting `exec_command` requires `podman@1` in this first version; it cannot
execute on the host or silently fall back to `local-workdir@1`.

Every `podman@1` Worker requires a declared Stage `context.workspace` with
`mode: direct` and a Runtime configured with `workspace.storage: local`.
Sources and optional initial state use the existing artifact hydration
contract. No operator checkout, empty-workspace schema extension or additional
mount paths can be supplied by the model.

Compatibility is checked at three boundaries:

1. Authoring rejects `podman@1` with missing workspace or `mode: overlay`, and
   rejects execution tools with an incompatible SandboxProfile.
2. Placement requires the exact runtime, profile, selected tools and local
   direct workspace capability. These are binding-specific requirements; a
   different Worker in the Stage may continue to use its own compatible profile.
3. Runtime repeats all checks before resource creation. Unknown or incompatible
   combinations fail prepare without starting code.

This forbids Contractor overlay mode with the execution sandbox, including a
temporary materialization followed by implicit merge-back. The container
engine may still use its ordinary image-layer storage internally.

Profile/tool compatibility belongs in registered descriptors and their Runtime
counterparts. It is not inferred from instructions or from a tool-name prefix.
The current rule that storage is diagnostic-only for placement must be extended
for this explicitly local profile; ordinary workspace placement remains as in
[10](10-runtime-filesystems-and-edit-tools.md).

## Operator-owned process configuration

Podman support is opt-in immutable Runtime startup configuration. The following
is the logical settings contract; exact CLI/environment spelling and implementation
ceilings are documented in [Runtime Podman policy](../../runtime/PODMAN.md).
Settings are parsed at process startup, not from a public Workflow document:

| Setting | Initial policy |
|---|---|
| enabled | False unless explicitly enabled |
| image | Required preinstalled OCI image pinned by digest |
| engine | Local rootless Podman; no remote URL/socket selection |
| owner identity | Stable unique local Runtime service identity, held under an exclusive ownership lock |
| cpu | 2 vCPU by default; positive operator-configured limit |
| memory | 2 GiB by default; positive operator-configured hard limit |
| pids | 256 by default; includes command children and sandbox supervisor |
| tmpfs | 256 MiB by default for private temporary files |
| network | `none` in `podman@1` version 1 |
| command timeout | 60 seconds default, 300 seconds operator maximum by default |
| stdout/stderr previews | 32 KiB each |
| combined output limit | 1 MiB per command; exceeding it terminates execution |
| preparation timeout | 30 seconds maximum by default, bounded by allocation prepare/lease deadlines |
| stop grace | 5 seconds maximum by default, bounded by remaining teardown deadline |

Limits are finite, validated at startup and bounded by implementation ceilings.
There is no model-selected image, user, mount, Podman flag, host executable,
credential, network mode or resource override. The Runtime does not pull or
build images during allocation prepare. Packages and the execution supervisor
are provisioned in the approved image; project-local environments can be
created from already available dependencies.

Network access is deliberately a first-version constraint: package downloads
and live target HTTP requests from executed code are unavailable. A later
networked profile needs explicit egress/proxy semantics and Audit active-check
classification. Existing `http-tools@1`/`caido@1` remain separate explicitly
selected capabilities; their credentials/proxy configuration are not injected
into this sandbox.

The resolved image digest and effective limits are captured in allocation-owned
diagnostic state. Reports may include the image digest and safe limits, but not
host paths or secrets. This first version does not introduce Run-selected image
or resource policies via Runtime labels.

## Container and mount boundary

Runtime creates an unprivileged rootless container with private namespaces,
dropped capabilities, no-new-privileges and the host's supported confinement
profile. It does not disable SELinux/AppArmor/seccomp globally to make mounts
work. Required UID/GID mapping and mount labels are established and probed
before advertising the profile. Host and container tools must be able to read
and edit each other's project files without recursively changing ownership of
an operator directory.

| Container path | Access and lifetime |
|---|---|
| `/workspace` | Read/write bind of the allocation's project content directory |
| `/tmp` | Bounded private tmpfs, disposed with the container |
| image root | Read-only approved tools and supervisor |

No parent provider root, ownership marker, general Runtime scratch, home
directory, engine socket, Artifact token, mTLS key or LLM credential is mounted.
Additional device access, privileged execution and host namespaces are not
available. Environment variables are constructed from an explicit safe
allowlist, including a writable temporary HOME; the Runtime environment is
never copied wholesale and bare Podman environment-inheritance flags are not
used. No command may choose its execution user.

Regular Linux container isolation shares the host kernel; this is not a VM
profile. CPU, memory, process and tmpfs limits are enforced by the runtime/OS,
not by model instructions. The first version does not provide a per-allocation
hard disk quota for the `/workspace` bind mount. Workspace acquisition limits
in [10] do not stop an executing process from filling that filesystem; operators
own dedicated workspace capacity and any filesystem quotas. This limitation
must be explicit in deployment documentation and capability diagnostics must
not claim a disk limit that is not enforced.

## Allocation preparation and ownership

Preparation follows this order:

1. Validate exact profile/tool/workspace compatibility and remaining deadlines.
2. Acquire the Runtime's allocation slot and prepare its ordinary private
   scratch and local project provider resources.
3. Hydrate exact source artifacts and optional state onto disk under [10].
4. Create and start the owned container with `/workspace` bound to that content
   directory; verify the supervisor, effective limits and mount access.
5. Construct selected Toolsets with narrow filesystem and execution handles,
   then construct the Worker.
6. Return prepared/ready only after all resources are usable.

`SandboxFactory.prepare()` currently returns only local scratch and runs before
project hydration. Implementation must separate scratch preparation from
starting the execution sandbox, or introduce equivalent lifecycle hooks. A
remote/container ID must not masquerade as an `AllocationWorkspace.path`.
The container is owned by the allocation lifecycle, not by an individual tool
instance whose `close()` might run multiple times.

Each container has a unique, internally generated name and labels identifying
Contractor ownership, the stable local Runtime service instance, process
incarnation and allocation ID. Runtime retains the full engine container ID;
subsequent operations resolve and verify that exact owned resource. There are
no `--latest`, broad `--all` or name-prefix-only destructive operations.

The stable service owner is held under an exclusive local ownership lock so
one Runtime's startup recovery cannot collect another live Runtime's resources.
Creation records ownership labels atomically with engine creation. If the CLI
times out before returning the ID, Runtime reconciles by the unique creation
identity before retrying or cleaning up; it must not create a second container
blindly. No Run/user text is interpolated into host shell commands.

Retries of an identical prepare return the already owned allocation resources.
A mismatching request remains a conflict. Partial prepare removes its container
before deleting project files; cleanup uncertainty fences the slot.

## Execution Toolset

`code-execution@1` initially exports only:

```text
exec_command(command: string, cwd: string = "", timeout_seconds: integer = 60)
```

`command` is a bounded UTF-8 shell program (maximum 64 KiB) passed as one
argument to a fixed `/bin/sh -c` inside the container. Podman itself is invoked
with structured argv and no host shell. `cwd` uses the existing normalized
workspace-relative path grammar; `""` selects `/workspace`. It cannot select a
host path or a directory outside the mounted project through a link.

Runtime supplies a private allocation-owned execution handle only to selected
execution tools. This is a sandbox channel, not the existing host
`runtime-subprocess-launcher`: existing validators must not automatically move
into the container or lend host execution privileges to this tool. Infrastructure
descriptors and factory validation must distinguish these channels. The model
never receives the handle, container ID or raw engine client.

Commands run without a PTY or interactive stdin. They may invoke Python or any
other installed executable through the shell. Separate calls preserve project
files and temporary files for the allocation's lifetime, but do not promise
persistent shell cwd/environment, Python globals or background services.

There is at most one active command per allocation. Waiting for the workspace
operation lock is bounded by the command's effective deadline. The deadline is
the minimum of requested/operator limits and the applicable invocation/lease
bounds; the command cannot extend a lease or a shutdown grace period.

Each completed command returns a bounded structured observation:

```text
status: completed | timed_out | output_limit_exceeded | failed
exitCode: integer | null
stdout: string
stderr: string
stdoutTruncated: boolean
stderrTruncated: boolean
durationMs: nonnegative integer
errorCode: stable code | null
```

Output is untrusted content, decoded with explicit replacement for invalid UTF-8.
Stdout and stderr are drained concurrently and counted before preview truncation.
Truncation alone does not stop a command; exceeding the combined hard limit does.
No unbounded full log is retained in memory or written to host/container log
storage. Preview truncation flags remain true when bytes were omitted.

A normal program exit with a nonzero code is `completed` with that exit code;
it is evidence available to the Worker, not an infrastructure retry. Engine,
supervisor, timeout and output-limit failures are distinguishable. Runtime must
not infer engine success/failure from command stdout or a single ambiguous CLI
exit code: command status comes through the managed execution protocol and is
checked against engine state. Raw Podman errors are mapped to content-free
stable codes. Commands are never automatically replayed after an uncertain
outcome because filesystem or other effects may already have happened.

Files intended to outlive allocation release are explicitly written as ordinary
Run artifacts through existing Artifact tools and grants while writes remain
permitted. A sandbox path is not an ArtifactRef, and this version adds no
automatic output upload or workspace export.

## Command completion and workspace coordination

`exec_command` owns the same coordination boundary as local direct reads,
edits and snapshot acquisition for its entire command lifetime. It may release
that ownership only after there are no remaining workload processes that can
write the project tree. A Python asyncio lock or exit of the attached Podman
client alone is not evidence of that condition.

Version 1 has no background-command API. The managed execution supervisor
must account for command descendants, including reparented/detached children,
and confirm their termination before reporting completion. A process-group kill
alone is insufficient if children can detach into another group/session. A
workload that leaves live descendants is terminated as unsupported background
execution. If descendant cleanup cannot be confirmed, Runtime stops the entire
container and makes the allocation unusable rather than releasing workspace
ownership and continuing. The supervisor protocol and this guarantee must be
tested with adversarial fork/detach cases on the supported Podman environment.

On timeout, cancellation, hard output-limit violation or uncertain engine
communication after launch, the first version conservatively stops the whole
container. It does not restart it or run another command in that allocation.
Normal nonzero program exits can continue if cleanup is confirmed. Runtime
reports fatal execution infrastructure failures through the existing Worker
failure/termination path; an interrupted command's partial filesystem effects
are not rolled back or labelled as a completed result.

After any command outcome, current project files remain authoritative under
[10]. Any retained snapshot is invalid for a subsequent call until reacquired.
The filesystem contract also works for completed external changes that did
not originate from `exec_command`.

## Finalize, abort, release and crash recovery

The container lives for the allocation, including sequential A2A invocations
and `INPUT_REQUIRED`. An individual tool return or A2A invocation completion
does not remove it. Allocation finalization and release remain distinct:

```text
prepare:  hydrate project → start container → construct tools/Worker → ready
run:      acquire workspace → execute → confirm no writers → release workspace
finalize: reject new work → stop Worker/commands → stop container → report
release:  confirm container removal → remove mounted project/scratch → acknowledge
```

Finalize/abort/lease loss first reject new commands. Stop includes a finite
grace and forced termination within the remaining lifecycle deadline, followed
by engine-state verification. Host client cancellation alone never completes
this step. A lost lease cannot be extended just to finish a user command.

Normal result artifacts are published before Contractor's existing write fence,
not after entering finalizing/aborting. Stopping the container retains the
project bind contents until allocation release; shutdown does not introduce a
late export window. Abort and lease-loss cleanup may remove resources earlier
according to [02]/[04], but must preserve container-before-files ordering.

Release is idempotent and owns one cleanup task: close tool entry points,
confirm container stopped/removed, dispose mounted project files, then release
scratch/adapters. The current project-before-sandbox cleanup order must change
for this profile. A timeout retains the owned cleanup task and fences the slot;
retries join it. Only authoritative release confirmation makes the slot idle.

Startup recovery runs before advertising idle capacity. It acquires the stable
service ownership lock, lists only Contractor containers for that owner,
validates ownership and removes predecessor allocation containers before the
workspace provider removes their files. Containers belonging to other Runtime
services or unrelated Podman users remain untouched. Engine operations use
full resolved IDs and bounded verification. Global prune is never recovery.

The deployment must also stop owned containers when the Runtime service dies;
a stopped Python process does not imply stopped containers. The approved
container supervisor must enforce a finite Runtime-renewed liveness deadline
bounded by the confirmed control lease, so code cannot keep running indefinitely
while the Runtime is down. It must stop accepting commands and terminate the
container when that deadline expires. Startup cleanup remains responsible for
removing stopped orphan resources. No workload can renew or disable the
supervisor's liveness deadline or forge command-completion acknowledgements.
The control channel and supervisor authority must be separated from workload
processes, not merely hidden in a file readable by the same execution user.
A host guardian tied to the Runtime service or an equivalently isolated
supervisor can implement this boundary; the implementation must prove it before
advertising the profile. This is a required implementation/probe item, not a
guarantee supplied automatically by `podman exec`.

## Probes, errors and metrics

Enabling a setting alone is not a capability. Bounded startup discovery checks
local engine access, rootless configuration, cgroup/resource support, pinned
image availability, compatible supervisor, bind access/UID mapping, execution,
descendant cleanup, liveness expiry and confirmed container removal. Probe
containers have separate ownership identities and must be cleaned before the
Runtime advertises `podman@1` and the dependent tool operation.

A failed probe omits those capabilities. It must not degrade to privileged
Podman, host execution, memory storage or overlay. If probe cleanup cannot be
confirmed, Runtime remains fenced/unready. Existing usable profiles can remain
available when the failed probe left no resources behind.

Stable failure classes include invalid command/cwd, preparation failure,
sandbox unavailable, timeout, output limit, unsupported background execution
and cleanup failure. Launch/outcome uncertainty is never an automatic command
retry. Allocation preparation errors retain existing retryable/nonretryable
classification and shutdown/cleanup uses existing lifecycle reporting.

Metrics record command count, elapsed time, exit/failure category, captured-byte
counts, truncation and sandbox prepare/stop/cleanup outcomes. Aggregate logs
exclude command text, output, host paths and secrets. Bounded command/output
content belongs only to the existing authorized tool observation surfaces.

## Future skill mounts

The planned extension is a read-only `/skills/<name>` mount containing selected
Run-pinned skill packages, with writable outputs directed to `/workspace` or
`/tmp`. Selection and pinning continue to belong to [09](09-agent-skills.md).
No global skill directory or mutable catalog is mounted.

Current packages deliberately exclude scripts. Enabling script members,
validation, disclosure, interpreter requirements, invocation and capability
checks needs a separate amendment to [09]. Merely selecting `podman@1`,
installing ADK or loading SKILL.md does not enable script execution. A skill
cannot add tools, images, mounts, privileges or network authority.

## Acceptance and implementation sequence

Implementation follows V30-001 through V30-004 and V31-001 through V31-008 in
the [task catalog](../../tasks/index.yml). Podman work starts after the local
direct gate V30-004 passes. V31 separates settings/placement, engine ownership,
the approved image and supervisor, allocation lifecycle, execution tools,
capability probes, deployment examples and the real-container release gate.
V31-006 enables advertisement only after the lifecycle and tool implementation
are present; V31-008 is the feature completion gate. Task files contain their
own requirements, dependencies and executable acceptance commands.

Required acceptance cases:

1. Valid local direct placement succeeds. Missing workspace, overlay, memory,
   unselected tools, missing image and failed prerequisites never execute code
   or silently select a different backend.
2. Prepare creates exactly one container. Repeated prepare, uncertain create
   response and partial failures preserve idempotency and owner-scoped cleanup.
3. A file edited through `edit-files` is immediately visible to a command; files
   created/modified/deleted by commands are visible to filesystem, observation
   and analysis calls without refresh. Other allocations remain isolated.
4. Tool input cannot select host paths, engine options, mounts, credentials,
   execution user or network. Host shell metacharacters stay container input.
   Mount boundaries, UID mapping and root/parent/leaf path attacks are tested.
5. Stdout/stderr floods and invalid UTF-8 remain bounded. Nonzero command exit,
   engine failure and timeout produce distinct observations; no command is
   automatically replayed after uncertainty.
6. Normal completion, fork/detach, cancellation, timeout and output-limit tests
   prove no workload writer survives before workspace ownership is released.
   An unconfirmed stop fences the allocation and prevents reuse.
7. Two overlapping tool calls and command/edit/snapshot races preserve the
   common workspace ordering. Heartbeats remain responsive during engine and
   filesystem I/O.
8. `INPUT_REQUIRED` and sequential invocations reuse one allocation container;
   finalize stops it, release removes it before mounted files and repeated
   finalize/release remain idempotent.
9. Runtime crash without immediate restart stops workload execution within the
   supervisor liveness bound. Restart cleans only its predecessor resources,
   including containers whose create response was lost.
10. CPU/memory/pids/tmpfs/network policy is verified in the real supported
    rootless deployment; undeclared disk-quota guarantees are never asserted.
11. Existing `local-workdir@1`, memory/overlay workspaces, trusted host
    validators and script-free skills retain their behavior and authority.

## Upstream references

These references describe engine primitives; the lifecycle and policy choices
above are Contractor requirements, not claims that Podman implements them all.

- [Podman run](https://docs.podman.io/en/stable/markdown/podman-run.1.html):
  container creation, mounts, namespaces and resource controls.
- [Podman exec](https://docs.podman.io/en/stable/markdown/podman-exec.1.html):
  execution inside a running container, attached streams and exit statuses.
- [Podman stop](https://docs.podman.io/en/stable/markdown/podman-stop.1.html):
  bounded graceful stop followed by forced termination.
- [Podman API service](https://docs.podman.io/en/stable/markdown/podman-system-service.1.html):
  engine access grants the service user's authority; it is not a workload API.
