# Approved supervisor image (V31-003)

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

Build only as an explicit provisioning operation (the pinned base must already
be installed for this offline example):

```sh
podman build --pull=never --network=none \
  -t localhost/contractor-supervisor:v31-003 \
  -f deploy/podman/Containerfile deploy/podman
podman image inspect localhost/contractor-supervisor:v31-003 --format '{{.RepoDigests}}'
```

Copy the displayed `localhost/contractor-supervisor@sha256:...` reference into
`CONTRACTOR_TEST_PODMAN_IMAGE`, then run `make test-podman-supervisor`. The gate
does not build, pull or repair prerequisites. Missing configuration, an unpinned
image, incompatible image/host or failed isolation test fails the gate. It
creates uniquely owned test containers and removes only those exact resources.
Ordinary pytest intentionally skips this opt-in real-host module; that is not
a substitute for the explicit real gate.

This image and gate do not enable production `podman@1`. Allocation lifecycle,
bounded command transport/tool integration, startup probes, deployment service
fencing and the full release gate remain V31-004 through V31-008 work.
