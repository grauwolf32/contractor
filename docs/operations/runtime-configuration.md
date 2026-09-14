# Runtime capabilities, labels and upgrades

[Documentation index](../README.md) · [Local stack](../guides/local-stack.md)

Run commands from the repository root unless a block changes directory.
API examples use the local Server and `CONTRACTOR_API_TOKEN` from the local-stack
guide. For typed process deadlines, see [timeout configuration](timeout-configuration.md).

## Environment-specific Runtime Agent capabilities

A Runtime Agent probes its enabled WorkerRuntime, Toolset/tool and
SandboxProfile factories once, after its private listener starts and before it
registers. The resulting positive snapshot is fixed for that `instanceId`.
Keep each process environment fixed: in particular, give it an explicit
executable search path containing only the validators and tools intended for
that deployment.

For example, two otherwise identical processes may be started by the service
manager with different fixed environments:

```shell
env PATH=/opt/contractor/source-tools/bin:/usr/bin \
  runtime/.venv/bin/python -m contractor_runtime [normal Runtime flags...]

env PATH=/opt/contractor/architecture-tools/bin:/usr/bin \
  runtime/.venv/bin/python -m contractor_runtime [normal Runtime flags...]
```

The second directory might contain a directly invokable `likec4`, while the
first does not. Both processes still advertise independent LikeC4 editing
tools; only the process whose startup probe succeeds advertises
`validate_likec4`. Do not install or replace executables underneath a running
process. Stop it, change the environment, and start a new process so it gets a
new `instanceId` and a newly probed immutable snapshot.

Verify the accepted positive snapshots in the UI at
`/operations/runtime-agents`, or through the read-only
`GET /v1/operations/runtime-agents` endpoint. The view intentionally contains
only exact runtime refs, SandboxProfile refs and Toolset/tool names. It does
not expose failed probe output, executable paths, environment values or a
remote re-probe control.

`make test-capability-e2e` proves this boundary with two production Python
Runtime Agent processes. One receives an isolated empty executable path and
remains idle while a Stage requiring `validate_likec4` waits in its first
`StageExecution`; a later process receives a deterministic fake LikeC4 CLI,
is selected without a retry, and completes normal prepare, A2A, finalization,
release and Operations reconciliation.

## Label-driven Runtime infrastructure

Runtime labels select physical Worker infrastructure without changing a
Workflow's ModelPolicy, budgets, Planner or tool allowlist. The checked-in
[deployment examples](../../deploy/runtime-labels/README.md) show secret-free OTLP
and proxy documents. Publish credential material through the write-only
Operations mutation, publish each immutable RuntimeConfig, and then bind its
short label. A missing label always means only the pinned `default` binding;
the bootstrap `contractor-empty@1` default preserves authored executionConfig.

Every concurrently connected process must have a unique CA-signed identity:

```shell
contractor pki issue-runtime --name agent-proxy
contractor pki issue-runtime --name agent-telemetry
```

After binding an optional startup label such as `site-proxy`, start the first
process with its own certificate and an immutable adapter subset:

```shell
cd runtime
uv run contractor-runtime \
  --control-plane-url https://127.0.0.1:8443 \
  --advertised-control-url https://127.0.0.1:9443 \
  --advertised-a2a-url https://127.0.0.1:9443 \
  --ca-file ../.local/pki/ca.crt \
  --certificate-file ../.local/pki/agents/agent-proxy.crt \
  --private-key-file ../.local/pki/agents/agent-proxy.key \
  --listen 127.0.0.1:9443 \
  --work-root ../.local/runtime/agent-proxy \
  --initial-label site-proxy \
  --runtime-adapter http-proxy@1
```

Start the second process on another listen/advertised port with
`agent-telemetry.{crt,key}`. Omit `--runtime-adapter` to probe all built-ins, or
repeat it for an explicit subset. Startup labels seed only a previously unseen
certificate principal; subsequent assignments use CAS on
`PUT /v1/operations/runtime-agent-principals/{runtimeAgentId}/labels`.

Inspect certificate identity, authoritative labels and safe adapter capability
without receiving an endpoint or credential value:

```shell
curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  'http://127.0.0.1:8080/v1/operations/runtime-agent-principals?limit=50' | jq .
curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  'http://127.0.0.1:8080/v1/operations/runtime-labels?limit=50' | jq .
```

A Run-selected Runtime label such as `debug` may require `otlp-http@1` even when neither Agent
is named or labeled `debug`; placement selects any capable idle candidate. An
Agent label is the highest physical Worker layer and can replace that Run's
endpoint/credential on its next allocation. Rebinding a label while work is
active changes only future Runs (for Run-selected Runtime labels) or future
allocations (for Agent Runtime labels). `make test-runtime-labels-e2e` proves
these rules with real
Server, PostgreSQL, two uniquely certified Runtime processes, OTLP protobuf,
authenticated proxying, exporter failure and complete slot reuse.

The public Run field is `runtimeLabels`. The former top-level `labels` array is
not a compatibility alias: upgrade Server, bundled UI and automation clients
together after draining request traffic, and discard any unsubmitted browser
draft created against the old shape. Existing Run rows and active allocation
snapshots keep their pinned Runtime configuration and require no data rewrite.

## Worker session mode upgrade

`workerSessionMode` is another mandatory private AllocationSpec field. The Go
Server, every Python Runtime Agent and the generated UI/API contract must be
deployed in lockstep; an old Runtime rejects the new field and a new Runtime
rejects its absence. The deployment procedure is:

1. pause new Run admission and let current Stage allocations finish, or cancel
   them through the normal bounded abort path;
2. confirm that Operations shows no authoritative allocation and every Runtime
   slot has completed release reconciliation;
3. take the normal PostgreSQL backup, then stop Server and Runtime Agents;
4. deploy Server, Runtime Agents and UI from the same release and restart them;
5. validate configuration and run the focused gate below before resuming
   admission.

No data migration rewrites existing WorkflowRun or StageExecution snapshots.
The new Server decodes a genuinely absent `session` field in those persisted
pre-feature snapshots as `shared`, preserving their original conversation
behavior. Newly loaded YAML with omitted `session` instead resolves and stores
explicit `isolated`; explicit `shared` remains available when a Stage needs one
sequential conversation. Explicit null, empty and unknown values fail in both
authoring and persisted data.

A binary-only rollback after admitting a new Run is unsafe because old Server
code does not understand the explicit field in its immutable snapshots. Before
resuming admission, rollback may replace the whole lockstep release. Afterward,
use a forward fix or restore the pre-upgrade database backup together with the
old binaries; never delete the field manually from selected rows.

Run the focused verification before release:

```shell
go test -race ./internal/config/... ./internal/scheduler/... \
  ./internal/controlplane/... ./internal/httpapi/public/...
make verify-wire-contracts
make test-wire-cross-language
make test-worker-session-modes-e2e
```

The process gate exercises default-isolated and explicit-shared Stages through
the real Scheduler, Control Plane, private allocation transport and Python
Runtime, including sequential invocations, allocation cleanup and Runtime-slot
reuse.
