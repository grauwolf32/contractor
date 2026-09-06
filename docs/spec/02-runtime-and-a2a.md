# 02 — Runtime and A2A

Status: **Working agreement**

Depends on: [00](00-workflow-and-planner.md), [01](01-agent-template.md)

## Control Plane

Control Plane owns Runtime Agent registration, availability and allocation. It
checks the WorkerRuntime selected by each AgentTemplate, chooses free Runtime
Agent capacity and returns ready `WorkerHandle` values to Workflow Scheduler.

It manages allocation lifecycle, not Planner algorithms, Worker objectives or
A2A Task semantics.

The partially implemented operational extension in
[22](22-performance-metrics-and-profiling.md) owns optional allocation resource
collection requests and their diagnostic capability advertisement. That
capability does not constrain placement or change the lifecycle defined here.

## Control-plane trust and mTLS

The planned Audit-check extension in [25](25-audit-worker-finalization.md)
adds a Server-pinned optional AllocationSpec completion contract and explicit
Runtime completion capabilities. Unlike resource diagnostics, support for a
requested completion contract constrains placement and direct preparation.
Omission retains ordinary behavior; labels, A2A task text and model tool calls
cannot activate it. Unsupported contracts fail closed without downgrade.

Server and Runtime Agents form one deployment-owned trust domain rooted in a
single CA. The private control protocol uses mutual TLS. All Runtime Agents are
equally trusted and have the same control-protocol privileges; the certificate
does not encode per-agent roles or an authorization policy. They may still
advertise different execution capabilities because their immutable process
environments contain different runtime, tool or sandbox dependencies.

Control Plane accepts any otherwise-valid Runtime Agent certificate that chains
to this CA. A Runtime Agent accepts a Control Plane peer only when both of the
following hold:

- its certificate chains to the same CA;
- one URI SAN starts with the exact canonical prefix
  `urn:contractor:control-plane:` and has a non-empty instance identifier after
  the prefix, for example
  `urn:contractor:control-plane:01951d4e-ec24-7df2-a9c1-18d55d123456`.

The URI SAN, rather than CN or OU, distinguishes Control Plane certificates from
the otherwise-equivalent Runtime Agent certificates. Prefix matching permits
certificate rotation and multiple future Control Plane instances without an
agent-side allowlist. CA issuance must ensure that Runtime Agent certificates do
not contain the reserved Control Plane URI SAN prefix.

For HTTPS/TCP, normal endpoint-name verification against a DNS/IP SAN still
applies; the reserved URI SAN is an additional Control Plane role check, not a
replacement for standard TLS endpoint validation. The same peer check protects
Server calls to both the Runtime Agent control endpoint and its A2A endpoint.

This is a node trust boundary, not user authentication: user clients do not
receive certificates from this CA, and an in-process Worker has no identity or
certificate separate from its Runtime Agent. Separately from TLS, Control Plane
keeps the normal allocation invariant that lifecycle messages and final reports
are associated with the Runtime Agent to which the allocation was issued. That
association protects execution-state consistency; it does not give different
Runtime Agents different privileges.

## Registration, principal and process identity

Runtime Agent registration represents one running process. On process start
the agent generates a fresh random `instance_id`. It retains that value in
memory across registration retries and Control Plane reconnects, but does not
reuse it after its own restart.

Separately, Control Plane derives a stable `runtime_agent_id` from the
authenticated leaf certificate's public-key fingerprint. It binds durable
Agent-label configuration to a trusted peer but grants no different Runtime
authorization. Each concurrently running logical Runtime Agent therefore uses
its own certificate/key pair, even on one VM. The complete principal and label
contract is owned by [07](07-runtime-labels-and-infrastructure-config.md).

The minimal registration describes:

```text
RuntimeAgentRegistration
  instance_id
  software_version
  started_at
  control endpoint
  A2A endpoint
  initial labels used only to seed a previously unseen certificate principal
  frozen startup WorkerRuntime, Toolset/tool, SandboxProfile and RuntimeAdapter capabilities
  observed_state
  allocation_id?
  private protocol version
```

`software_version` is a bounded version string reported by the Runtime Agent
binary itself. It is observation metadata for Operations and diagnostics; the
Server neither substitutes its own build version nor treats this string as a
placement capability or trust assertion.

Registration with the same `instance_id` is idempotent. The `instance_id` is a
correlation and routing identity, not an authentication credential; mTLS
authenticates the agent and derives its stable principal. Control Plane records
the last accepted
registration/heartbeat request, issued and confirmed heartbeat sequences, plus
observed and authoritative single-slot facts for the live instance. Placement
checks the explicitly reported exact runtime, tool, sandbox and adapter
capabilities;
those capabilities do not grant a different authorization role to otherwise
equal CA-trusted Runtime Agents.

The accepted registration binds `(runtime_agent_id, instance_id)` for the
process lifetime. Every later registration retry, heartbeat, private Artifact
call and final report carrying that `instance_id`/allocation must arrive through
the same certificate-derived principal; a different CA-valid peer is rejected.
This is allocation consistency, not per-Agent RBAC.

The registered control and A2A endpoints must present a CA-valid leaf with the
same SPKI fingerprint when Control Plane connects to them. Normal DNS/IP SAN
verification still applies. A Runtime Agent cannot register another trusted
agent's endpoint and cause Control Plane to deliver that allocation's
RuntimeSettings or A2A traffic to the wrong peer.

A restarted Runtime Agent under the same certificate principal registers with
a new `instance_id` and retains the Control Plane's durable Agent labels. It
cannot adopt a Worker or allocation belonging to its previous process instance.
The old
registration becomes unavailable after its confirmed control lease expires,
and any active StageExecution using that allocation enters `aborting`; Workflow
Scheduler records an interrupted StageTermination, completes the bounded abort
path and then lets Workflow Scheduler apply the declared interrupted policy:
retry with a new StageExecution, use a configured escalation executionConfig,
or fail the Run.

The stable principal is not an incarnation or Worker-continuation identity.
This first model still has no registration generation, Server epoch, rotated
fleet token or cross-process Worker continuation protocol.

## Startup capability discovery

An installed factory and a usable capability are different facts. Before its
first registration, each Runtime Agent builds its enabled local FactoryRegistry
and runs bounded capability probes against the process environment. The result
is one immutable `CapabilitySnapshot`:

```text
CapabilitySnapshot
  supported_runtimes: sorted set of exact WorkerRuntime refs
  supported_toolsets: sorted map of exact Toolset ref -> sorted non-empty tool set
  supported_sandbox_profiles: sorted set of exact SandboxProfile refs
  supported_runtime_adapters: sorted set of exact RuntimeAdapter refs
```

The snapshot maps directly to the existing registration fields; there is no
second capability document or Server-side environment catalog. Only positive
capabilities are registered. Probe failures remain bounded, redacted Runtime
Agent startup diagnostics and are not sent as placement facts. A Toolset with
no usable exported tool is omitted. If no WorkerRuntime or no SandboxProfile
passes, the process fails startup readiness and does not register an `idle`
slot. Empty Toolset and RuntimeAdapter lists are valid and can serve templates
and resolved settings that require neither. The RuntimeAdapter list contains at
most 64 sorted unique exact refs; the complete request remains subject to the
private registration body bound.

Every enabled factory owns its local probe semantics:

- a WorkerRuntime probe verifies that the adapter and its Runtime-owned local
  prerequisites can construct that exact runtime contract; for `adk@1` after
  the Agent Skills increment this includes the pinned ADK Skill models, package
  loader and native SkillToolset required by [09](09-agent-skills.md), but no
  particular skill package;
- a Toolset probe returns the exact subset of `exported_tools` that can honor
  their complete contracts, including required local executables, libraries
  and compatible versions; filesystem Toolsets additionally require the
  immutable workspace storage provider from
  [10](10-runtime-filesystems-and-edit-tools.md);
- a SandboxProfile probe proves that the profile can prepare and clean an
  isolated probe resource under its operator-owned root. A cleanup failure is
  a failed probe;
- a RuntimeAdapter probe proves that its local code/dependencies can construct
  that exact typed adapter contract without contacting a configured endpoint.

Ordinary factory probes have a five-second timeout and the complete startup
probe phase has a thirty-second timeout when Podman is disabled. Opt-in
[Podman discovery](21-podman-sandbox.md#probes-errors-and-metrics) has a
sixty-second total budget and a shared profile/execution probe bounded to
thirty seconds, including an eight-second cleanup reserve. Unconfirmed Podman
cleanup prevents registration; it is not an optional capability omission.
Timeout, cancellation or an unexpected
exception means that factory or affected tool is unavailable; an optional
failure does not prevent unrelated capabilities from being advertised. Probes
must not use Workflow/User artifacts, make an LLM call or perform domain work.
They may create only uniquely named, bounded resources below a Runtime-owned
probe/work root and must remove them before reporting success. External
programs are invoked directly without a shell, with closed stdin, bounded
output and the same effective executable search/configuration that the real
tool will use.

Runtime workspace-provider initialization under [10] is shared preparation
inside that same complete thirty-second phase. It validates private local or
memory storage and cleanup primitives but does not download Run artifacts or
hydrate a tree before an allocation. Filesystem factory probes reuse the
frozen provider and therefore remain within the five-second per-factory bound.

A positive Toolset capability means more than “the function can be called”. If
an operation requires an external validator to fulfill its declared behavior,
returning `validator unavailable` is not that operation's successful startup
capability. The factory omits that tool while retaining independent tools that
passed their own prerequisites.

`code-analysis@1` is one concrete use of partial positive capability. Its
Tree-sitter operations may pass on either workspace provider, while its
Trailmark graph operations additionally require a local provider and a bounded
offline child-process probe. The child is internal Toolset machinery rather
than a `runtime-subprocess-launcher` channel. Exact placement and lifecycle are
owned by [12](12-code-analysis-tools.md).

Capability probes cover only Runtime-owned prerequisites available before an
allocation. They do not probe an LLM Gateway URL/token, OTLP endpoint, HTTP
proxy, Artifact API grant or other RuntimeSettings supplied later in
AllocationSpec. Availability of those per-allocation dependencies follows the
ordinary bounded preparation or execution failure path and is not a
physical-agent placement capability.

The computed snapshot is frozen for the lifetime of `instance_id` and reused
byte-for-byte across registration retries and Control Plane reconnects.
Control Plane rejects a registration retry that reuses the `instance_id` with
a different snapshot. Heartbeats carry no capability update. The deployment
contract requires the process environment, enabled factory set, executable
resolution and local dependency configuration to remain unchanged after
registration. Applying an environment change requires restarting Runtime
Agent, which creates a new `instance_id`, probes again and registers a new
snapshot. Dynamic re-probing, capability withdrawal and in-place
re-registration are outside the first slice.

The `v1alpha1` registration shape treats the four capability dimensions as
composable sets. A Runtime Agent must therefore advertise only runtimes,
Toolsets/tools, sandboxes and Runtime adapters that can be combined safely
within that process.
An environment with combination-specific incompatibilities must expose their
common safe subset or run separate Runtime Agent processes with compatible
registries; capability-profile expressions are deferred.

## Heartbeat and confirmed control lease

The first-slice heartbeat interval is 10 seconds and the confirmed Runtime
Agent control lease is 60 seconds. Control Plane returns both values on
registration so that the Server remains their authority and the agent does not
depend on matching local configuration. Each side measures expiry with its own
monotonic clock; no shared wall-clock timestamp is a lease authority.

A Runtime Agent starts a strictly increasing heartbeat `sequence` at process
start and does not reset it across reconnects. A request contains:

```text
RuntimeAgentHeartbeat
  instance_id
  sequence
  last_ack_sequence?
  observed_state       idle | allocated | draining | fenced
  allocation_id?
  worker_stopped?
```

Control Plane responds with:

```text
RuntimeAgentHeartbeatAck
  ack_sequence         equal to the accepted request sequence
  action               continue | drain | release | reregister
  allocation_id?
```

Heartbeat interval and lease duration are returned by registration, not
repeated in every acknowledgement.

`last_ack_sequence` echoes the newest response the Runtime Agent actually
received. Control Plane maintains two different liveness facts:

- registration `last_seen` advances on every valid authenticated heartbeat
  request and is useful for observation and routing diagnostics;
- the confirmed control lease advances only when `last_ack_sequence` advances
  and names an ack that this current in-memory registration entry previously
  issued. Repeating the same ack never extends the lease again.

Runtime Agent likewise renews its local confirmed lease only after receiving a
new ack for a heartbeat sequence that it actually sent. A duplicate, delayed or
unknown ack does not extend the watchdog. Consequently requests reaching
Control Plane while all responses are lost cannot keep the confirmed lease
alive indefinitely: the echoed ack stops advancing and both sides expire.

An active allocation is live only while the confirmed control lease is valid
and the observed allocation matches Control Plane's authoritative record. An
idle process is eligible for placement only after the same round trip is
confirmed and both sides agree that no allocation exists. A heartbeat is only
an observation: `idle`, `fenced` or a different `allocation_id` can trigger
reconciliation but can never release or replace the authoritative allocation.

When the confirmed lease expires on Control Plane, it excludes the Runtime
Agent from placement. Any active allocation is irreversibly reported lost to
Workflow Scheduler, which enters `aborting` with an interrupted
StageTermination. A late ack cannot revive that allocation; only reconciliation
and a new allocation may return the slot to service.

Runtime Agent maintains the symmetric monotonic watchdog. If 60 seconds pass
without a new valid ack, the agent:

1. rejects new A2A Tasks and enters `draining`;
2. asks its active in-process Worker runtime to abort and snapshots the
   accumulated report without further semantic work;
3. stops and destroys that Worker instance within the configured shutdown grace;
4. if in-process termination cannot be guaranteed, exits the Runtime Agent
   process instead of reusing the slot;
5. otherwise deactivates the allocation's A2A identity, secrets and access
   context, retains its allocation ID and cached final report, and enters
   `fenced` rather than `idle`.

The fenced agent continues registration and heartbeat retries. It cannot accept
another allocation until Control Plane explicitly acknowledges release of the
old allocation. If no allocation existed when the local lease expired, the
agent still remains fenced until a confirmed response establishes that the
authoritative slot is unallocated. A lost release response is safe: the agent
continues reporting the old allocation as fenced and Control Plane repeats the
idempotent release action.

### Observed/authoritative reconciliation

- matching live allocation IDs and a confirmed lease allow `continue`;
- after reservation and before the private `prepare` commits locally, an
  authoritative allocation may transiently coexist with an observed idle slot;
  this bounded preparation transition receives `continue` and is not treated
  as allocation loss;
- observed `draining` is valid only when Control Plane has issued finalization
  or abort for that allocation;
- an authoritative allocation whose grant is write-fenced is never allowed to
  continue Worker execution: every matching `allocated` heartbeat receives
  `drain` until the Worker reports `draining`/`fenced` or the Runtime process
  exits; this is the fail-safe when a private finalize/abort request is lost;
- observed `fenced`, `idle` or another allocation while Control Plane still
  owns an active allocation makes that allocation lost and starts/resumes the
  bounded abort path;
- an allocation reported for a `finalizing` or `aborting` StageExecution causes
  the corresponding idempotent control command to be reissued;
- an allocation reported for a terminal or unknown StageExecution receives an
  idempotent drain/release action;
- no mismatched process is offered for placement until it reports the released
  state and completes a fresh confirmed round trip.

A late report may be accepted as incomplete telemetry, but lease loss cannot be
reversed into successful completion of the interrupted StageExecution.

## Runtime Agent Registry persistence

The first implementation keeps the live Runtime Agent Registry in Control Plane
memory. Registrations, last-seen times, issued/confirmed heartbeat sequences,
control leases and current slot availability are not PostgreSQL state, and
heartbeat processing does not write one database row every 10 seconds.

PostgreSQL retains only the durable execution facts that refer to allocations,
including their StageExecution association and terminal outcome. After a Server
restart the in-memory Registry starts empty. A Runtime Agent process that
remained alive registers again with its existing process-scoped `instance_id`;
an agent process that also restarted uses a new one. Recovery still interrupts
every affected `preparing` or `running` StageExecution through durable
`aborting` because its Planner is not resumable, then drains any Worker reported
by the reconnected agent.

After Registry loss, an echoed ack from the former Control Plane process is not
recognized because the new in-memory entry did not issue it. After
re-registration, Control Plane acknowledges the next heartbeat; only echoing
that new response establishes the fresh confirmed control lease. The
process-monotonic heartbeat sequence avoids collision with a delayed response
without introducing a durable Server epoch or registration generation.

Registration or heartbeat reporting an active `allocation_id` triggers
reconciliation with durable StageExecution state before that slot can receive
new work. A matching `finalizing` execution reissues its recorded finalization;
a matching `aborting` execution reissues its recorded abort; a terminal or
unknown allocation is drained and released. Reconnection never adopts a Worker
for continued Planner execution.

This design permits one active Control Plane instance in the first slice.
Shared fleet coordination, durable liveness state and multiple active Control
Plane replicas are deferred together; they must not be approximated by sharing
an unfenced `runtime_agents` table. Durable certificate principals and their
label assignments under [07] are configuration records only; they never imply
that a process is live, leased or eligible.

The in-memory Registry is a current live/reconciliation index, not process
history. A superseded or control-lease-expired process entry is retired from
memory as soon as it owns no authoritative allocation. An entry that still
owns an allocation remains until that allocation is reconciled and released.
Registration performs the same expiry/retirement sweep before enforcing its
bounded live-index capacity, so process-scoped IDs accumulated across Runtime
restarts cannot permanently exhaust admission. Historical connection/audit
data, if introduced, belongs in a separate bounded durable facility.

## Runtime Agent

A Runtime Agent is one long-running process with one exclusive slot. Runtime
Agent and Worker behavior are implemented by the same code and execute in that
same process. `Worker` names the temporary allocation-scoped role/runtime
instance, not another service, daemon, subprocess or container.

For one allocation the Runtime Agent:

1. reserves its only slot and validates `AllocationSpec` against its frozen
   startup runtime, Toolset/tool, SandboxProfile and RuntimeAdapter capability
   snapshot, including the effective digest-bearing ModelPolicy and exact
   `resolvedSkills` manifest and mandatory `workerSessionMode`;
2. constructs the selected allocation-scoped infrastructure adapters and
   validates their typed handle set before creating any sandbox, Toolset or
   Worker resource;
3. asks the selected SandboxProfile to prepare the allocation-local scratch;
4. if `AllocationSpec.workspace` is present, creates a private WorkspaceSession,
   downloads its exact RunScope ZIP/state refs through the existing Artifact
   API, hydrates the selected local/memory storage and establishes a checkpoint.
   For `podman@1`, prepares the allocation-owned container only after local
   direct hydration, before any Toolset or Worker receives execution access;
5. binds the private Artifact client and fetches/validates the exact RunScope
   Agent Skill packages selected under [09](09-agent-skills.md);
6. validates the resolved AgentTemplate projection, prepares allocation-local
   Contractor State, instrumentation reducers, selected tools and native ADK
   SkillToolset, then passes only the minimal `WorkerBuildContext` plus the
   effective ModelPolicy selected by the Run's ResolvedExecutionConfig to the
   in-process Worker runtime factory;
7. configures its own A2A Server and Agent Card for that allocation;
8. binds model access to the allocation context;
9. reports ready and handles the Worker's A2A Tasks itself;
10. before each graceful terminal A2A `WorkerCompletion`, closes the matching
   observation snapshot and performs any declared overlay auto export; on
   finalization or abort, rejects new Tasks, requests
   cancellation of active work, destroys the Worker runtime instance, closes
   every selected Toolset, confirms the Podman execution container stopped
   when present, bounded-flushes/destroys allocation adapters and
   serializes the report;
11. after the terminal Stage outcome is committed, an idempotent private
   release confirms Toolset cleanup and Podman container removal before
   disposing mounted files. It removes allocation State, the
   WorkspaceSession and every
   disposable local/memory/overlay file, loaded Skill objects, extracted
   allocation-local skill files, RuntimeSettings, access tokens and the profile
   workspace, but retains the allocation identity and cached report in
   `fenced` state; only exact marker-owned Runtime work directories are cleanup
   targets;
12. after Control Plane has removed its authority, a subsequent heartbeat
   `release` action confirms that edge; only then does Runtime Agent clear the
   retained identity/report, become `idle` and free the slot.

The Worker runtime may use ADK or another future in-process adapter without
changing Workflow Scheduler or Planner contracts. Its dependencies live in the
Runtime Agent deployment, not in Contractor Server. It receives no PostgreSQL
or S3 credential and cannot outlive the Runtime Agent process.

Finalization and abort enter through the Runtime Agent's existing private
control endpoint. They perform no new semantic work: the same process stops its
current Worker runtime, converts accumulated metrics to `ExecutionReport` and
returns the report. Neither operation is exposed as an A2A skill or waits for
A2A cancellation to reach a terminal Task state.

The first-slice `local-workdir@1` SandboxProfile is prepared before any Toolset
factory or Worker can run. Failure to create its fresh allocation directory is
an allocation-preparation failure. During release, Runtime Agent acknowledges
local cleanup only after removing that directory, while continuing to report
the old allocation identity as `fenced`. Cleanup failure keeps the slot
unavailable and an idempotent release retry repeats cleanup. A lost HTTP
response is therefore safe: neither side can infer an idle slot. Before
registering an idle slot after process startup, Runtime Agent also removes
recognized orphan allocation directories
under its dedicated configured work root. The profile does not add a process,
container, filesystem-permission or network security boundary.

### RuntimeSettings from Control Plane

Control Plane includes a resolved `RuntimeSettings` snapshot in every
AllocationSpec. It may contain the selected LLM Gateway URL and optional token,
the Server private Artifact API endpoint, timeouts, limits and typed
telemetry/HTTP proxy adapter settings. Gateway values come either from the
WorkflowRun's
immutable `ResolvedExecutionConfig` or from a higher-precedence Run/Agent label
layer that completes or overrides its physical Worker route. Runtime Agent does
not resolve these values from AgentTemplate, label strings, local Worker
configuration, environment defaults or a mutable configuration alias. The
complete precedence, pinning, adapter isolation and credential rules are owned
by [07](07-runtime-labels-and-infrastructure-config.md).

`LLM Gateway` is a backend-neutral role in this contract. LiteLLM is the initial
backend, not a required Contractor component; a compatible backend can replace
it without changing Workflow, AgentTemplate or allocation semantics.

Secret fields are accepted only over the private mTLS control channel, retained
in memory for the active allocation and redacted from logs, Agent Cards,
WorkerHandle, metrics and durable StageExecution state. They are erased during
release and never copied into ADK State or the private live-State snapshot.
Durable execution and metrics may retain the non-secret
LLMGatewayConfig and credential refs, never their resolved token. A token that
expires during a long allocation fails through the ordinary bounded Gateway
error path; silently replacing the active settings snapshot is not allowed.

Runtime Agent applies Worker proxy and telemetry settings only through
allocation-owned adapter objects. It does not mutate process-global proxy
environment or trust stores, and registration, heartbeat, control, A2A and
Artifact API traffic always bypasses the Worker proxy. MemoryTools uses that
same Artifact client and adds no proxied endpoint. Finalization/abort closes
adapters after a bounded best-effort telemetry flush; release idempotently
ensures they remain closed and erases their retained settings secrets.

### Correlation and redacted boundary failures

Every public and private HTTP response carries one bounded `X-Request-ID`.
Public ingress always generates its own value. Private Control Plane, lifecycle,
Artifact and A2A hops propagate one syntactically valid incoming value and
replace missing, duplicated or malformed values. REST error objects repeat it
as `requestId`; A2A retains its protocol error envelope and carries correlation
in the HTTP header.

Server-side 5xx diagnostics record the request ID, boundary, method, status and
safe error type. They do not record raw URL paths, request bodies, artifact
bytes, provider exception messages, RuntimeSettings or tokens. This gives an
operator a stable lookup key without turning a user-controlled path or nested
tool argument into a logging channel.

## WorkerHandle

A ready handle contains only what Workflow Scheduler and Planner need to address
the prepared Worker:

- exact AgentTemplate ref;
- exact WorkerRuntime ref;
- globally unique allocation_id;
- external Agent Card;
- lease/deadline information.

The ready handle echoes the exact `lease_expires_at` supplied in
`AllocationSpec`; it does not renew or reinterpret that authoritative Control
Plane lease.

That timestamp proves preparation happened inside the then-current confirmed
lease; it is not a frozen lifetime deadline for later Planner calls. Successful
sequenced heartbeat acknowledgements renew the live Control Plane and Runtime
watchdogs without mutating the immutable WorkerHandle. Every A2A call keeps the
Stage deadline, while authoritative lease loss independently cancels the
Scheduler invocation and Runtime Agent self-fences its Worker.

It contains no host path, tool-sandbox handle or in-process Worker object.
It also contains no LLM Gateway token or other RuntimeSettings secret.

## Preparation and execution flow

```text
Workflow Scheduler selects a ready Stage and resolves its AgentTemplates
  -> Control Plane resolves pinned default/Run-selected Runtime labels plus candidate Agent Runtime labels
  -> Control Plane matches every template/adapter requirement against frozen capability snapshots
  -> Control Plane atomically selects a complete set of free Runtime Agents
  -> Runtime Agents reserve their slots and accept AllocationSpecs
  -> each Runtime Agent creates one in-process Worker instance
  -> each Runtime Agent publishes its allocation-scoped Agent Card
  -> Control Plane returns ready WorkerHandles
  -> Workflow Scheduler constructs the selected Planner over fixed WorkerHandles
  -> Planner WorkerInvoker tools <-> A2A <-> allocated Runtime Agents acting as Workers
  -> Planner returns candidate StageResult
  -> Workflow Scheduler durably enters finalizing
  -> Control Plane asks Runtime Agents to finalize their Worker instances
  -> Workflow Scheduler accepts terminal StageResult
  -> Control Plane releases allocation contexts, leases and slots
```

Infrastructure preparation completes before Planner starts. If any required
Worker fails preparation, Workflow Scheduler receives no partial Planner
subagent set.

### Capability-aware placement

The implemented Podman extension [21](21-podman-sandbox.md) adds an explicit local
storage requirement for bindings selecting `podman@1`, as well as registered
profile/tool compatibility checks. Preparation creates the container after
hydration; teardown removes it before mounted files. Only a verified, opt-in
Runtime advertises this capacity. Existing profiles retain the behavior below.

The currently connected fleet is deliberately not consulted while loading a
Workflow, resolving an AgentTemplate or creating a WorkflowRun. Those steps
validate exact refs and tool names against Server descriptors. Physical
availability is ephemeral and is evaluated only when each StageExecution is in
`preparing`, so an eligible Runtime Agent may connect after the Run was
created.

For one resolved Stage Agent binding, the placement requirement is exactly:

```text
required runtime = AgentTemplate.runtime
required sandbox = AgentTemplate.sandboxProfile
required tools   = each AgentTemplate.toolsets[ref].tools
required adapters = RuntimeAdapters referenced by resolved Run + Agent label settings
required workspace mode = Stage context.workspace.mode, when workspace exists
required workspace storage = local for podman@1; otherwise no storage constraint
```

A Runtime Agent is a candidate only when it is `idle`, has a confirmed control
lease, is otherwise placement-eligible, advertises the exact runtime and
sandbox refs, its advertised tool set is a superset of every selected tool for
each exact Toolset ref, and its advertised adapter set contains every adapter
required after combining that principal's Agent labels with the pinned Run
labels. Extra capabilities neither activate settings, change Worker behavior
nor become model-visible.

Placement never requires a Run label name to appear in the candidate's Agent
label set. A Run `debug` configuration applies to an unlabeled Runtime Agent
when that process advertises the resulting required adapter; Agent labels are
only the higher-precedence candidate-specific configuration layer defined by
[07].

For a multi-Agent Stage, Control Plane must find a complete injective matching
between logical bindings and eligible single-slot Runtime Agents before
reserving anything. It must not greedily consume a broadly capable agent when
that would hide an existing complete assignment. For example, if agent A can
perform source analysis and LikeC4 validation while agent B can perform only
source analysis, a Stage needing one source-only Worker and one LikeC4
validator is placeable as source-only -> B and validator -> A. A deterministic
maximum bipartite matching or an equivalent complete algorithm satisfies this
contract; the particular equally valid assignment is not Workflow semantics.

If no complete matching currently exists, Control Plane returns temporary
insufficient compatible capacity. StageExecution remains `preparing`, no slot
is reserved and the existing bounded capacity backoff applies. This covers
both busy compatible agents and an environment capability that is not yet
represented in the live fleet; the Stage deadline or Run cancellation remains
the bound. Planner never chooses, observes or changes physical placement.

### Atomic Stage reservation

Control Plane reserves capacity for the complete Stage Worker set as one
in-memory operation. Under the single Control Plane instance's registry lock it
either moves enough `idle` slots to `reserved` or changes no slot at all. A
Stage requiring three Workers therefore never holds two slots while waiting for
the third.

Temporary lack of sufficient capacity is not a Stage failure. StageExecution
remains `preparing`, no slot is reserved, and Workflow Scheduler retries with
bounded backoff until capacity appears, the execution deadline expires or
cancellation is requested.

After reserving the full set, Control Plane creates one globally unique
`allocation_id` and one complete AllocationSpec per logical Stage Agent binding,
including that binding's effective ModelPolicy and resolved RuntimeSettings
plus exact label/config provenance and the exact Run-pinned `resolvedSkills`
manifest from [09] and the Stage's pinned `workerSessionMode`, then may initialize
those Runtime Agents concurrently. The mode participates in reservation replay
identity: a retry of the same `stage_execution_id` cannot reinterpret it. Planner
starts only after every Runtime Agent reports ready and Control Plane can return
the complete `WorkerHandle` map.

If any initialization fails, times out or returns an incompatible Agent Card,
Control Plane drains/releases every allocation from that batch, including
Runtime Agents that became ready. Workflow Scheduler receives no partial map and
moves that StageExecution through `aborting` with a retryable interrupted
StageTermination whose code identifies the infrastructure error. When the
batch is cleared or the abort deadline expires, the execution becomes terminal
and Workflow policy decides whether to create a new attempt.

`prepare_stage` is idempotent for `stage_execution_id`. A retry after a lost
response observes the same in-memory reservation/preparation operation and
cannot create a second Worker set. Loss of the Control Plane process instead
uses the already-defined interruption and cleanup recovery path.

```python
class ControlPlane(Protocol):
    async def prepare_stage(
        self,
        bindings: Collection[ResolvedStageAgentBinding],
        context: StageAllocationContext,
    ) -> Mapping[AgentName, WorkerHandle]: ...

    async def finalize_stage(
        self,
        handles: Collection[WorkerHandle],
        finalization_id: FinalizationId,
        deadline: datetime,
    ) -> Mapping[AgentName, AllocationFinalReport]: ...

    async def abort_stage(
        self,
        handles: Collection[WorkerHandle],
        abort_id: AbortId,
        deadline: datetime,
    ) -> Mapping[AgentName, AllocationFinalReport]: ...

    async def release_stage(
        self,
        handles: Collection[WorkerHandle],
        reason: ReleaseReason,
    ) -> Mapping[AgentName, AllocationReleaseAck]: ...
```

Transport DTOs are defined by the versioned schemas under `api/v1alpha1`.
Preparation, finalization, abort and acknowledged release are always a private
control protocol rather than A2A skills. The StageExecution lifecycle and
report contracts are defined by [04](04-execution-lifecycle-and-metrics.md).

## Capacity and execution scope

One Runtime Agent has one active slot. A Stage declaring three named Workers
needs three available Runtime Agent processes before Planner starts. Those
processes may all run on the same VM.

One allocation lasts for the complete Planner invocation and its finalization.
Planner may send multiple sequential A2A Tasks to its Worker while the
allocation is active. A Worker executes at most one Task at a time; separate
Stage allocations may execute concurrently. A draining allocation rejects new
A2A work and remains reserved until its StageExecution is terminal. A fenced
slot rejects all work and remains unavailable until authoritative
reconciliation and release acknowledgement complete.

Every allocation receives a globally unique `allocation_id` that is never
reused. Scheduler-owned retry, configured escalation and replacement create new
StageExecutions and new allocation IDs. This ID is the sole lifecycle, routing and report-correlation
key; there is no separate allocation generation. Repeated control commands for
an active `allocation_id` are idempotent, while commands and A2A requests for a
released or unknown ID are rejected.

Planner cannot expand or replace the prepared set during the invocation.
Changing the set requires a new StageExecution decision by Workflow Scheduler
under the Workflow policy.

Allocation identity, A2A Task/context identity, Contractor Worker invocation
identity and ADK session identity are four separate domains. Runtime creates
opaque ADK session IDs according to the pinned Stage mode; none is derived from,
equal by contract to, or exposed through the allocation ID, A2A payload,
WorkerCompletion, Agent Card, public API or Planner tools. A rejected request
therefore cannot infer whether a prior session exists. The allocation-local
session and State lifecycle is owned by [04](04-execution-lifecycle-and-metrics.md).

## Allocation-scoped A2A endpoint

Runtime Agent itself exposes one stable authenticated A2A address. While its
slot is allocated, the same process configures its A2A Server and external Agent
Card from the active AgentTemplate. An
opaque allocation routing key may be carried in the
[`AgentInterface.tenant`](https://a2a-protocol.org/v1.0.0/specification) field
defined by A2A 1.0; the protocol leaves its routing semantics opaque. It is not
an authentication credential.

```text
external Agent Card
  skills = [contractor_stage_content]
  supportedInterfaces[0].url = https://runtime-agent-7.internal/private/v1/allocations/allocation-123/a2a
  supportedInterfaces[0].protocolBinding = JSONRPC
  supportedInterfaces[0].protocolVersion = 1.0
  supportedInterfaces[0].tenant = allocation-123

Runtime Agent active slot
  allocation-123 -> in-process Worker runtime created from AgentTemplate
```

The MVP card exposes one strict Contractor stage-content skill. Its input mode
is `application/vnd.contractor.stage-content+json` and its output mode is
`application/vnd.contractor.worker-completion+json`, as owned by [14]. The
AgentTemplate's selected tools remain internal model capabilities and are not
advertised as independent A2A skills.

The external card declares the endpoint's mutual-TLS security requirement. The
Planner client is configured with the Server's Control Plane certificate; the
advertised `tenant` remains only the opaque route selector and is never treated
as a credential.

The A2A endpoint responsibilities are limited to:

- mTLS authentication of the Server as a Control Plane peer;
- validation that `tenant` matches the active allocation and lease;
- delivery of every A2A operation and stream to the current in-process Worker
  runtime;
- rejection of stale, expired or released allocations;
- deactivation during finalization, abort and release.

The transport layer does not plan work or assemble results. The in-process
Worker runtime owns Task semantics while the surrounding Runtime Agent owns
transport, allocation and lease checks.

## Protocol ownership

The private control protocol owns registration, heartbeat, reservation,
AllocationSpec and RuntimeSettings delivery, readiness, lease, idempotent Worker
finalization and abort, execution/runtime report transport, release and cleanup.
While a Worker is live it also owns the read-only allocation-correlated
`GET /private/v1/allocations/{allocation_id}/agent-state` transport. That route
returns only bounded Contractor-owned State to Control Plane; it is neither an
A2A operation nor a public/generic state API. [14](14-worker-results-and-live-state.md)
owns its schema and lifecycle.

A2A owns interaction with a Runtime Agent's already prepared Worker instance:
messages/streams, Task state and continuation, cancellation, progress and
artifact refs.

The artifact plane owns data access, authenticated scope and Namespace grants,
and never sends normal durable artifact bytes through A2A. See
[03](03-artifact-plane.md). Agent Skill package bytes follow the same rule:
AllocationSpec carries exact refs/digests and Runtime reads them through the
private Artifact API under [09](09-agent-skills.md).
Workspace source/state bytes follow the same rule under
[10](10-runtime-filesystems-and-edit-tools.md); their exact refs are private
construction data while ordinary Stage task content remains A2A.

## Single-VM baseline

The baseline deployment may use:

```text
one VM
  Contractor Server process
  PostgreSQL
  Runtime Agent process 1 (one slot, Worker runtime executes in-process)
  Runtime Agent process 2 (one slot, only if concurrency is needed)
```

Control, A2A and artifact transports may use loopback on the single VM.
Kubernetes, distributed scheduling, a child Worker process and a remotely
managed AgentTemplate service are not prerequisites.

## Invariants

1. Physical placement belongs to Control Plane, not Workflow Scheduler or
   Planner.
2. One declared Stage Agent name maps to one allocation and handle.
3. One Runtime Agent process has one slot and at most one in-process Worker
   instance; it never hosts multiple Workers concurrently. It may create the
   next Worker only after the prior allocation is released and fully cleared.
4. Runtime Agent owns transport/allocation checks; its current Worker runtime
   owns every A2A Task under the allocation.
5. Allocation identity is not A2A Task identity; one allocation may serve a
   sequence of Tasks.
6. Planner completion first moves its StageExecution to durable `finalizing`;
   Scheduler-owned cancellation/interruption instead moves it to durable
   `aborting`. The two transitions are mutually exclusive.
7. Worker finalization and abort cannot perform model/tool calls or artifact
   mutations, wait indefinitely for A2A Task cancellation, or exceed their
   durable control deadline.
8. Workflow Scheduler releases allocations only after recording terminal Stage
   outcome.
9. Stale, draining or released allocation identities cannot address the current
   or a replacement Worker instance.
10. A Runtime Agent recognizes a Control Plane peer by the reserved URI SAN
    prefix in a certificate signed by the deployment CA; Control Plane applies
    no per-agent authorization policy beyond allocation ownership consistency.
11. Runtime Agent process restart creates a new process `instance_id`, may
    retain its certificate-derived principal and durable labels, and never
    resumes or adopts the previous process instance's allocation.
12. The confirmed control lease advances only through a new monotonic
    request/ack/echo round trip; receiving heartbeat requests alone does not
    keep an allocation live or a slot eligible for placement.
13. Local lease expiry drains and destroys Worker, then leaves Runtime Agent
    fenced with the old allocation identity until explicit release is
    acknowledged; it never silently becomes idle.
14. Observed and authoritative allocation mismatch triggers reconciliation and
    cannot itself release, adopt or replace an allocation.
15. Live registrations, heartbeat sequence state and slot availability are
    in-memory Control Plane state; PostgreSQL does not serve as a heartbeat
    registry in the first slice.
16. An `allocation_id` is globally unique and never reused; no allocation
    generation exists in the first model.
17. Capacity for one Stage is reserved all-or-nothing; Planner never receives a
    partial WorkerHandle map.
18. Runtime Agent advertises only runtime, Toolset/tool, SandboxProfile and
    RuntimeAdapter capabilities proven before first registration; that
    normalized snapshot is immutable for its process-scoped `instance_id` and
    is never changed by heartbeat.
19. Capability-aware placement requires set containment for each binding and a
    complete injective matching for the whole Stage; extra advertised tools are
    never exposed implicitly.
20. Runtime Agent's process environment is immutable after registration.
    Changing dependencies, executable resolution or enabled factories requires
    a process restart, new `instance_id` and new startup probes.
21. A label/config change affects only a future allocation; Runtime Agent never
    interprets the label name or mutates an active RuntimeSettings snapshot.
22. Allocation proxy settings never intercept registration, heartbeat,
    control, A2A or Artifact API traffic.
23. Agent Skill packages are allocation data pinned by WorkflowRun, not Runtime
    environment capabilities or A2A Agent Card operations; Runtime never
    resolves an owner UserScope current skill binding.
24. Runtime exposes live Worker State only through the read-only
    allocation-correlated Control Plane endpoint; no private request can write
    an arbitrary ADK State key or replace pinned allocation configuration.
25. Every AllocationSpec carries the Stage's explicit immutable Worker session
    mode. Reservation replay with another mode conflicts rather than replacing
    an existing allocation.
26. Allocation, A2A Task/context, Contractor invocation and ADK session IDs are
    distinct; only Runtime may create or observe the ADK session identity.
