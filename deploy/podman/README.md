# Rootless Podman image and local deployment

Provision explicitly on a supported **rootless Podman + systemd cgroup v2** host.
The exact allocation cgroup must expose writable `cgroup.freeze` and
`cgroup.kill`. A pidfd-capable kernel/libc is required. Unsupported delegation
fails closed; there is no privileged or process-group-only fallback.

The Containerfile pins the Python 3.12 base by digest. `/bin/sh`, Python and the
base's other tools are fixed by that digest; no package installation occurs.
The only added executable is a small inert, isolated Python PID 1 that reaps
orphans. It runs as namespace root, mapped to a **subordinate host UID**, with
all capabilities dropped. Workloads must use `podman exec --user=<host UID>:<host
GID>` under keep-id; they must never use the image's root default. The root
filesystem is read-only and HOME is writable `/tmp` tmpfs. There is no command
socket, token, interpreter session or workload-controlled input to PID 1.

The host guardian lives in the Runtime package, **not in this image**. Neither
its inherited socket/pidfd/cgroup descriptors nor host engine access is mounted
or passed to workload processes. See [the trust and protocol contract](../../runtime/PODMAN.md#host-guardian-and-image-v31-003).

Use a Linux host with local Podman (not a remote connection or `podman machine`),
a working user systemd session/DBus, subordinate UID/GID mappings in `/etc/subuid`
and `/etc/subgid`, and an installed seccomp policy. Keep the user manager alive
while Runtime owners/guardians recover; an administrator may provision lingering
for a dedicated service account. Check under that same unprivileged account:

```sh
id -u
podman --remote=false info --format json
systemctl --user is-system-running
```

Require rootless=true, cgroupVersion=v2, cgroupManager=systemd and
seccompEnabled=true. A degraded user manager is not itself proof of incompatibility;
the mandatory gate tests the actual delegated container cgroup and mount access.
No `sudo podman`, disabling SELinux/seccomp, recursive chown or privileged fallback
is part of this setup. Dedicated binds use private SELinux relabeling when applicable.

Provision the pinned base explicitly (this one operator step can use the registry;
allocation and startup never pull):

```sh
podman pull docker.io/library/python@sha256:bd55e06128e2743d526738dd5cf5db924e49ead66d4fe2ad15f2aac8ad86aae9
```

For an air-gapped host, transfer a trusted image archive and load it into this
same user's store before building. Build from the repository root, offline:

```sh
podman build --pull=never --network=none \
  -t localhost/contractor-supervisor:v31-003 \
  -f deploy/podman/Containerfile deploy/podman
podman image inspect localhost/contractor-supervisor:v31-003 --format '{{.RepoDigests}}'
```

Copy a displayed `localhost/contractor-supervisor@sha256:...` reference into
`CONTRACTOR_TEST_PODMAN_IMAGE` (do not use a mutable tag or image ID), then run:

```sh
podman image inspect "${CONTRACTOR_TEST_PODMAN_IMAGE:?set the preinstalled digest reference}"
make test-podman-supervisor
make test-podman-workflow
```

If `RepoDigests` is empty, do not guess from the image ID: provision a digest-addressable
manifest via your approved image distribution process, then verify `image inspect`
by that exact reference. The [Podman build reference](https://docs.podman.io/en/latest/markdown/podman-build.1.html)
documents the explicit offline build options. The gate
does not build, pull or repair prerequisites. Missing configuration, an unpinned
image, incompatible image/host or failed isolation test fails the gate. It
creates uniquely owned test containers and removes only those exact resources.
Ordinary pytest intentionally skips this opt-in real-host module; that is not
a substitute for the explicit real gate.

After the gates pass, follow the exact [Runtime environment and sample workflow
recipe](../../runtime/README.md#local-podman-workflow). Positive capabilities are
opt-in and require local/direct storage. Default limits are 2 CPUs, 2 GiB memory,
256 PIDs, 256 MiB `/tmp`, 300 seconds maximum command duration (60 seconds when a
tool call omits its timeout), 30 seconds prepare, 5 seconds stop grace, 32 KiB
preview per stream and 1 MiB combined command output. Scope swap has a verified
finite upper bound; these numbers do not impose a bind disk quota. Each selected
allocation gets one container after hydration; it is stopped during finalization
and removed before workspace cleanup/release. Run `make test-podman-release`
with both `CONTRACTOR_TEST_PODMAN_IMAGE` and `CONTRACTOR_TEST_DATABASE_URL` set
for the mandatory real-container and PostgreSQL evidence. Neither prerequisites
nor real checks are skipped by that gate. See the [verification record](../../runtime/PODMAN.md#release-verification-v31-008);
Both that gate and repository-wide `make verify` passed for V31-008; rerun
them when changing the implementation or validating a deployment host.

## Service-manager lifetime and recovery

For foreground development, signal only the Runtime process with SIGTERM and
allow cleanup to finish. For a user service, adapt
[`contractor-runtime.service.example`](contractor-runtime.service.example), using
the installed virtualenv entrypoint directly (no `uv run`/shell wrapper as MainPID).
Copy the Runtime recipe's environment values into the referenced private
EnvironmentFile as `NAME=value` lines, with absolute paths and the real image
digest; shell `$PWD` expansion is not supported there. Do not put model credentials
in the image or mounts. Install/start the unit explicitly only after configuring
its working directory, executable and environment file.

The example uses `KillMode=mixed` with `SendSIGKILL=no`: stop signals the Runtime,
but must not kill its independently recovering owner/guardian descendants.
Unlike the default cgroup-wide kill, this preserves their cleanup authority.
`Restart=no` leaves restart/recovery an explicit operator decision. See
[systemd's kill-policy definition](https://github.com/systemd/systemd/blob/main/man/systemd.kill.xml).
Do not replace these settings with `control-group`, enable cgroup-wide timeout
escalation, or put the whole user slice under a watchdog that kills the guardian.
`systemctl stopped` is not a receipt that container writers are absent.

Keep the stable owner, nonzero UID, image policy and dedicated scratch/project
roots unchanged across restarts. Owner-scoped labels plus exact ID/name checks
identify only this Runtime's containers. On Runtime EOF or a stalled heartbeat,
the surviving owner and independent kernel guardian stop writers. Restart with
the same settings: recovery takes the owner lock and confirms removal before
any orphan bind deletion, probes or registration. If the previous owner still
holds the lock, startup fails safely; let it finish and retry. If the Runtime
itself is stuck, first isolate/drain its slot; any operator SIGKILL must target
the resolved Runtime PID only, never the service cgroup.

For teardown: drain the slot in Control Plane, stop Runtime, and allow its owner
to finish. Inspect only the configured owner, for example (read-only):

```sh
podman ps -a --filter label=io.contractor.sandbox.managed=1 \
  --filter label=io.contractor.sandbox.owner=local-podman-runtime
```

No global `podman prune`, bulk label deletion, PID guessing or deletion of mounted
directories is a recovery mechanism. If cleanup is uncertain, retain the slot
fence, owner records and both roots; fix the host prerequisite and retry recovery.
Never remove `.contractor-podman-owner-v1` markers to bypass a different owner or
disable Podman against an old root. Migration to another owner/account requires
confirmed old-container removal and new dedicated roots. Same-user host tampering,
killing owners/guardians, user-manager shutdown and indefinitely blocked kernel
operations are outside the bounded-success guarantee.
