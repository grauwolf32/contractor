# Server operations

Requirement prefix: `SOP` · [Specification index](../README.md)

A Server operation is a deterministic Stage execution path. Operation modules
adapt declared inputs, arguments and outputs; their domain modules own algorithms
and mutations. Scheduler invokes operations directly. Planner and Runtime Worker
execution do not participate in this path.

**SOP-R1.** A Stage selects exactly one of `planner` or `operation`. An operation
contains an exact registered `ref` and an `arguments` map using the common
literal/parameter binding grammar. Its `context`, `result`, Workflow output
mappings and transitions use the ordinary Stage contracts. Arbitrary module paths,
commands, unknown operations, Worker bindings, model settings, instructions,
workspace and shared Worker sessions are rejected.

**SOP-R2.** Stage context resolution pins exact current-Run artifacts before
execution through existing Artifact mechanisms. Scheduler binds arguments from
the immutable Run parameters and configuration. An operation receives only its
Run/Stage identity, current Scheduler claim, declared exact inputs and output
bindings. Configuration cannot select a foreign owner or arbitrary artifact scope.

**SOP-R3.** Scheduler admits and starts an operation through the existing Stage
state machine and Run claim. It allocates no Runtime, creates no Planner session,
calls no model and records no synthetic Worker or Planner execution. Operations
consume the existing model-free Stage lane and admitted Stage deadline.

**SOP-R4.** `internal/operations` owns dispatch and the common call contract.
Per-operation modules adapt calls to domain services; application composition
registers modules. Scheduler contains no operation-name or artifact-format
branches. Artifact reads, publication, retention and accounting remain owned by
ArtifactStore. Domain services own their classified mutations and common receipts.

**SOP-R5.** A call finishes synchronously under Stage cancellation/deadline; no
operation work survives its return. Domain writes verify the current Run claim,
Stage execution and lifecycle in the same transaction. Cancellation, claim loss
or Project deletion prevents later writes. Existing Scheduler cleanup and terminal
transitions settle the Stage without allocation cleanup or Planner recovery.

**SOP-R6.** Successful operations return the common bounded Stage result with
exact declared outputs. Scheduler validates the result and accepts it through the
ordinary finalization/output progression. The shared Stage result contract lives
outside Planner so both execution paths use one definition.

**SOP-R7.** Recovery may execute the same interrupted call with unchanged exact
inputs, arguments, Stage identity and output bindings. Side effects reconcile
through the domain's existing keyed mutation owner or ordinary Artifact natural
identity; no operation journal, receipt table, claim or independent retry clock
is introduced. A lost acknowledgement must not repeat an accepted mutation.

**SOP-R8.** Invalid input and domain conflicts are permanent operation outcomes.
Storage unavailability retains its existing classification; retries preserve
original intent and the remaining Stage deadline. An unavailable exact artifact
never resolves to latest or is recreated from cached bytes. Operations are bounded
by their domain's bytes, rows and computation limits plus the shared Stage result
limits; a registry entry alone does not authorize unbounded work.

**SOP-R9.** A failed operation Stage ends with an interrupted StageTermination
that carries the operation's own classified failure, or one of these codes:
`server_operation_unavailable` — the Server has no operation execution;
non-retryable; `invalid_request` — the arguments cannot be bound from the Run
parameters; non-retryable; `server_operation_timeout` — a deadline inside the
operation expired before the Stage deadline; retryable;
`server_operation_failed` — any other failure; retryable only for a transient
database failure; and `stage_deadline_exceeded`
([SCH-R184](scheduler.md)) once the Stage deadline has passed.

The Workflow form is explicit:

```yaml
stages:
  join-api:
    operation:
      ref: sitemap-openapi-join@1
      arguments:
        origin: {source: parameter, name: api_origin}
        contract: {source: literal, value: public-api}
    context:
      artifacts:
        openapi: {namespace: inputs, name: openapi, required: true}
        sitemap: {namespace: inputs, name: sitemap, required: true}
    result:
      artifacts:
        sitemap:
          required: true
          mediaTypes: [application/vnd.contractor.project-sitemap+json]
          from: {namespace: join-api, name: sitemap}
    on:
      succeeded: {succeed: {}}
      failed: {fail: {}}
      interrupted: {fail: {}}
```

The complete Workflow declares the input slots and `api_origin` parameter.
Use `sitemap-apply-openapi@1` for explicit live publication. With
`result: {artifacts: {}}` either operation returns only its bounded Stage summary;
no map or report copy is created. Declared outputs may be mapped to ordinary
Workflow outputs or selected by subsequent Stage context. The concrete operation
contracts are [SMT-R26](../artifacts/project-sitemap.md) through
[SMT-R30](../artifacts/project-sitemap.md).

**SOP-R10.** `openapi-prepare-work@1` reads the declared exact `openapi` input
and, for `scan-requests` selection, JSON `settings`. Its bound string arguments
are `workflow` (checking alias), `selection` (`operations` or `scan-requests`),
and canonical positive decimal `max_items` within LIM-ACT-5. Operation selection
uses the existing OpenAPI domain parser/resolver; scan-request selection uses
its existing settings/request preparation. Source archives retain their owning
source-format validation. Conditions, independent operation keys/order and gaps
are preserved; this Stage performs no target activity or Audit mutation.

The required JSON `work` output follows [ACT-R13](../audits/common-contract.md).
Per-operation JSON material and needed original inputs are selected by exact
descriptors. No task ZIP, Round, execution manifest, model, Runtime allocation
or second inventory/admission mechanism is constructed. Domain computation runs
outside database row locks. Each ordinary content-named material publication
checks Project/Run/Stage/claim ownership in its transaction; the declared list is
published only after its complete content passes validation. Identical ordinary
bindings reconcile their original exact versions; collisions fail permanently.
Interrupted intermediate material remains owner-local and normally purgeable.
The existing Stage deadline/cancellation owns all work; no new journal, receipt
or cleanup controller is introduced (FAIL-ACT-2, LIFE-ACT-2).


**SOP-R11.** `publish-findings@1` binds either exactly `candidates` and `evidence`
current-Run context inputs, or one exact `materials` input containing a flat
common material set with at most 16 candidate/evidence pairs. It accepts no
arguments and declares no Artifact outputs. Every selected envelope is validated
before the first mutation; the live claim and exact selection are fenced for
each item. The Finding
owner consumes the normalized contract under FND-R232/R233/R234. The operation
returns bounded JSON counts and source/Finding references in its ordinary Stage
summary. Complete domain detail uses the existing paginated Finding/source
readers. Scanner reports remain on their concrete Run.

**SOP-R12.** `memory-snapshot@1` has no context inputs and exactly one required
JSON output named `snapshot`, from an ordinary non-reserved binding. It binds
exactly `namespace` and `notes` using existing string literal/parameter bindings
(without a parser). Memory owns selection, canonical note decoding, bounds and
snapshot reconciliation under MEM-R100 through MEM-R103. The operation adapter
binds current Run/Stage/claim; the Project owner fences an attached Project's
lifecycle before the Run/Stage lock. This narrow explicit read of logical Memory
names does not permit reserved refs in StageContext or ordinary Artifact tools.
Its result contains the exact ordinary output and a count-only summary. No new
endpoint, receipt, allocation, Planner session or model is introduced.
