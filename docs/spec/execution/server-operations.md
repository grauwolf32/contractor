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
