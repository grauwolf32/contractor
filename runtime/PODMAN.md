# Podman startup policy

V31-001 registers authoring/placement contracts and parses immutable startup
policy. It does **not** install an execution factory, invoke Podman, pull images
or advertise `podman@1` / `code-execution@1`. Actual engine, supervisor,
allocation lifecycle and startup probes are subsequent V31 tasks.

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
