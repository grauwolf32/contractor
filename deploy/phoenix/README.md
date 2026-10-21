# Local trace viewer

Phoenix runs the UI, OTLP/HTTP collector and SQLite database in one container.
This local profile avoids Langfuse's ClickHouse, Redis and S3 dependencies.
Persistent state is bind-mounted from the checkout's ignored `.local/phoenix`
to `/data`; Contractor's PostgreSQL is not used or changed.

Requires Podman, a running Podman machine on macOS, and an available host port
6006. No Compose provider is needed. Run from any directory:

```sh
deploy/phoenix/run.sh up
deploy/phoenix/run.sh status
deploy/phoenix/run.sh down
```

The UI is at **http://127.0.0.1:6006**. The exact OTLP/HTTP traces endpoint is
**http://127.0.0.1:6006/v1/traces**. Only HTTP is published, on loopback; this
profile has authentication disabled and is intended for local use.
Phoenix telemetry, unsolicited external resources, the agent assistant and its
MCP server are disabled in this profile.

`up` starts or reuses `contractor-phoenix`, then runs bounded health probes.
`down` stops and removes that container while retaining all SQLite history.
It does not install a login service or automatically start after a host reboot.
The image is pinned by digest in `run.sh`; use `down` followed by `up` after
changing that pin or the host port. Failed startup leaves the container and
data available for inspection with `podman logs contractor-phoenix`.

Set `CONTRACTOR_PHOENIX_PORT` for another port (`0` allocates a free port and
prints the actual address). `CONTRACTOR_PHOENIX_CONTAINER` changes the container
name. A checkout's data directory belongs to one viewer at a time.

The script uses Podman's selected connection and does not change it. On a
macOS machine where the rootless connection has an existing UID/GID mapping
problem, select the already configured rootful connection for each command:

```sh
CONTAINER_CONNECTION=podman-machine-default-root deploy/phoenix/run.sh up
```

## Connect Contractor

Publish [`runtime-config.example.yaml`](runtime-config.example.yaml) through
the Operations RuntimeConfig UI/API, bind it to a Runtime label, and select
that label for a new Run. The [Runtime label guide](../runtime-labels/README.md)
describes publication and binding. No telemetry credential is needed for this
local unauthenticated collector.

The example assumes both Server and Runtime run on the host. When either runs
inside a container, its endpoint must resolve to the host from that container
(for Podman, normally `http://host.containers.internal:6006/v1/traces`), and
the engine must make the loopback-published service reachable. Verify that
route rather than exposing this unauthenticated profile on all interfaces.

Content capture stays off by default. To inspect prompts and responses, opt
into `captureContent: true` for the relevant exporter; Contractor's sensitive
tool rules still apply. Traces can then contain secrets and remain on local
disk until explicitly removed. The viewer does not change live Run settings.

Contractor emits standard OTLP spans with custom `langfuse.observation.*`
content attributes. Phoenix can retain them as raw span attributes. Its
specialized OpenInference LLM input/output and token views require the matching
semantic attributes; this deployment does not rewrite Contractor's exporter.

## Verify

```sh
sh deploy/phoenix/test-run.sh
CONTAINER_CONNECTION=podman-machine-default-root sh deploy/phoenix/test-smoke.sh
```

The smoke test uses a temporary checkout and container, sends synthetic OTLP
spans and verifies retained content after replacing the container. It never
reads the live `.local/phoenix`, modifies RuntimeConfig bindings or calls a
model. The test container and temporary data are removed on exit.

Sources: [Phoenix architecture](https://arize.com/docs/phoenix/self-hosting/deployment),
[official Docker and SQLite deployment](https://arize.com/docs/phoenix/self-hosting/deployment-options/docker).
