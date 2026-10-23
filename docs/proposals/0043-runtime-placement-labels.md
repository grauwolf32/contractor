# Runtime placement labels: cloud and local

Proposal 0043 · Status: **Partially implemented. Startup declarations and
Operations observations are delivered; Run/Audit selection, matching and UI
remain planned.** Recorded on 2026-10-23 from the user's decisions.

## 1. Accepted direction

Runtime Agents declare placement labels at process startup. A user chooses
required placement labels in the UI when launching a WorkflowRun or starting
an Audit. Control Plane evaluates those requirements against the current
Runtime fleet whenever it assigns Workers.

The launch stores the required labels, not a list of Runtime IDs. A newly
connected matching Runtime can serve later assignments without changing the
launch. Placement labels do not belong in Workflow definitions or
RuntimeConfig documents.

## 2. Fixed namespaces

The only top-level placement namespaces are **`cloud`** and **`local`**.
Namespace names and matching are case-sensitive; accepted labels use lowercase
ASCII segments with the existing `[a-z][a-z0-9_-]*` segment grammar.

| Namespace | Meaning | Examples |
| --- | --- | --- |
| `cloud` | Runtime executes on cloud infrastructure. | `cloud`, `cloud/aws`, `cloud/hetzner`, `cloud/yandex` |
| `local` | Runtime executes on local infrastructure. | `local`, `local/workstation`, `local/lab` |

A concrete label is a namespace alone or a namespace followed by non-empty
segments separated by `/`. Leading, trailing and repeated `/` are invalid.
Provider and local-site names are operator-defined segments, not a fixed
catalog. Additional top-level namespaces require an explicit design change;
`region/eu-central`, `cloud-test` and `cloudy` are not valid placement labels.

A Runtime may declare only the broad `cloud` or `local` label when it does
not need a more specific location. A label such as `cloud/aws` already belongs
to `cloud`; declaring the parent as another label is unnecessary.

Each subject's labels form one ancestor chain: `cloud` with `cloud/aws` is
valid, while mixed roots or disjoint provider/site branches are rejected.
The bounded grammar and registration declaration are owned by
[CTL-R156 through CTL-R158](../spec/execution/control-plane.md).

This namespace restriction applies to placement labels only. Existing
RuntimeConfig labels such as `debug` and `caido` retain their own grammar and
configuration meaning.

## 3. Parent matching

A required label matches an actual Runtime label when:

```text
actual == required || actual.startsWith(required + "/")
```

The `/` boundary is significant. A bare `cloud` requirement selects every
Runtime labelled `cloud` or a descendant such as `cloud/aws`. No `*` syntax
is needed or accepted in either advertised or required labels.

| Required label | Actual Runtime label | Matches |
| --- | --- | --- |
| `cloud` | `cloud` | Yes |
| `cloud` | `cloud/aws` | Yes |
| `cloud` | `cloud/hetzner` | Yes |
| `cloud/aws` | `cloud/aws` | Yes |
| `cloud/aws` | `cloud/aws/eu-central` | Yes |
| `cloud/aws` | `cloud` | No |
| `cloud/aws` | `cloud/hetzner` | No |
| `cloud/aws` | `cloud/aws-test` | No |
| `local` | `local/workstation` | Yes |
| `cloud` | `local/workstation` | No |

For several actual labels, each required label must match at least one actual
label. All required labels are combined with AND. An empty requirement adds
no placement restriction. Tool, sandbox, adapter, workspace and slot-capacity
checks still apply to every candidate.

## 4. Startup, launch and UI boundaries

- **Runtime declaration.** Startup is authoritative for the placement labels
  of one process incarnation. Registration publishes them; within one
  `instanceId`, including repeated registration, they remain immutable. A
  restart may declare a different set. Operations displays these labels.
- **Launch selection.** The UI submits required labels for one WorkflowRun
  or Audit. An Audit pins the selection at start and passes it to all child
  Runs, including preparation and later rounds. Retry and resume preserve the
  original requirement; a new launch may choose another requirement.
- **Dynamic matching.** Every new assignment evaluates the pinned requirement
  against current Runtime instances. Existing allocations retain their
  assigned instance; a newly connected Runtime does not move active Workers.
- **UI choices.** Suggestions include advertised concrete labels and their
  parents, so a Runtime advertising `cloud/aws` makes both `cloud` and
  `cloud/aws` available. Manual input is allowed for an offline deployment.
  A selected requirement remains visible when its last Runtime disconnects.
- **Unavailable capacity.** A valid requirement with no matching available
  Runtime uses the existing capacity wait and backoff. It never broadens to
  another namespace. The UI explains the wait. Queue waiting does not consume
  the Stage execution budget before admission ([RUN-R28](../spec/execution/runs.md)).

## 5. Existing contracts and implementation follow-up

This is a separate placement concept from both WorkflowRun metadata `labels`
and RuntimeConfig `runtimeLabels`. The existing contracts deliberately exclude
those labels from affinity matching ([CTL-R75 and CTL-R82](../spec/execution/control-plane.md),
[RCF-R149 and RCF-R182](../spec/configuration/runtime-config.md)).

Existing startup `initialLabels` seed a durable principal only on its first
registration; later registrations use Server-authoritative configuration labels
([CTL-R26](../spec/execution/control-plane.md)). That field cannot implement the
accepted startup authority for placement labels without changing its meaning.

Registration publishes optional non-null `placementLabels`; startup uses
`--placement-label` and `CONTRACTOR_PLACEMENT_LABELS`. Grammar bounds and
incompatible-chain rejection are defined by CTL-R156. The additive launch
request field `requiredPlacementLabels` remains a working name until the Run
selection task publishes it. Placement selection still needs persistence,
matching, Audit propagation and UI contracts.

Before implementation, create the owning task and cover grammar and parent
matching through shared Go/Python fixtures. Apply the predicate to candidate
construction, final reservation checks and model-Stage capacity holds. Preserve
complete multi-Worker matching, record actual placement labels in allocation
history, and verify Audit propagation, retry/resume, restart and cancellation
on the real store and process paths. Bound the dynamic UI inventory and every
page it reads. Fold the implemented guarantees into the owning specification
documents when the capability lands.

## 6. Implementation tasks

- [V405-001](../../tasks/done/v405-001-runtime-placement-declarations.yml): Publish startup-owned cloud and local Runtime placement labels.
- [V405-002](../../tasks/v405-002-run-placement-requirements.yml): Pin Run placement requirements and enforce dynamic Runtime matching.
- [V405-003](../../tasks/v405-003-audit-placement-propagation.yml): Pin Audit placement selection and propagate it to every child Run.
- [V405-004](../../tasks/v405-004-placement-label-picker-ui.yml): Select Runtime placement labels in Workflow and Audit launch forms.
- [V405-005](../../tasks/v405-005-placement-process-release-gate.yml): Verify cloud and local placement across real processes and fold proposal 0043.
