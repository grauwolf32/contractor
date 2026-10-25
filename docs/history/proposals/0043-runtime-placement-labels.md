# Runtime placement labels: cloud and local

[Historical documents](../README.md). Retained design/review evidence; current
contracts belong in the specification and current status in the task records.

Proposal 0043 · Status: **Implemented.** Startup declarations,
Operations observations, Run/Audit labels, inheritance, matching, UI and the
deterministic process release gate are delivered.
Recorded on 2026-10-23 from the user's decisions.

## 1. Accepted direction

Runtime Agents declare placement labels at process startup. A user chooses
ordinary labels in the UI when launching a WorkflowRun or starting
an Audit. The reserved cloud/local keys derive placement requirements. Control Plane evaluates those requirements against the current
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
[CTL-R156 through CTL-R158](../../spec/execution/control-plane.md).

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
- **Launch selection.** The UI uses ordinary key/value `labels`. `cloud` with
  an empty value derives `cloud`; `cloud=aws` derives `cloud/aws`; `local=lab`
  derives `local/lab`. All other labels remain ordinary metadata. Audit start
  fixes its entire label map, and every child Run inherits it plus system
  `audit.*` provenance. Retry, resume and later rounds preserve this snapshot.
- **Dynamic matching.** Every new assignment evaluates the pinned requirement
  against current Runtime instances. Existing allocations retain their
  assigned instance; a newly connected Runtime does not move active Workers.
- **UI labels.** The user clarified on 2026-10-23 that a separate location
  selector and live-fleet inventory are unnecessary. Add ordinary labels to
  Audit and use the existing Run metadata editor. Valid offline cloud/local
  values remain accepted, and failed reads never clear typed labels.
- **Unavailable capacity.** A valid requirement with no matching available
  Runtime uses the existing capacity wait and backoff. It never broadens to
  another namespace. The UI explains the wait. Queue waiting does not consume
  the Stage execution budget before admission ([RUN-R28](../../spec/execution/runs.md)).

## 5. Contract ownership

Runtime startup `placementLabels` remain a separate incarnation declaration
from RuntimeConfig `runtimeLabels`. Launch placement is derived from the
reserved `cloud`/`local` keys of ordinary metadata labels. This supersedes the
initial proposal's separate writable `requiredPlacementLabels` field. The
requirement remains a read-only, immutable derived snapshot for scheduling and
provenance; there is one caller-controlled source, `labels`.

Audit accepts at most 28 ordinary labels and rejects `audit.*` owner keys,
leaving four of the Run's 32 entries for trusted child-specific provenance.
Start fixes all Audit labels; changing them afterwards requires a new Audit.

Existing startup `initialLabels` seed a durable principal only on its first
registration; later registrations use Server-authoritative configuration labels
([CTL-R26](../../spec/execution/control-plane.md)). That field cannot implement the
accepted startup authority for placement labels without changing its meaning.

Registration publishes optional non-null `placementLabels`; startup uses
`--placement-label` and `CONTRACTOR_PLACEMENT_LABELS`. Grammar bounds and
incompatible-chain rejection are defined by CTL-R156. Run and Audit requests
use `labels`; read/repeat projections expose the derived requirement. Apply
matching at candidate construction, final reservation and model-capacity
holds, retain actual allocation labels, and verify lifecycle guarantees on
real stores and processes. Current guarantees belong to CTL-R156–161,
RUN-R107–109, AUD-R229–232, UI-R203 and SYS-R9 in their owning specifications.
`make test-runtime-placement-e2e` is included in the release process selections;
the graph guard rejects a missing or empty test selection.

## 6. Implementation tasks

- [V405-001](../../../tasks/done/v405-001-runtime-placement-declarations.yml): Publish startup-owned cloud and local Runtime placement labels.
- [V405-002](../../../tasks/done/v405-002-run-placement-requirements.yml): Pin Run placement requirements and enforce dynamic Runtime matching.
- [V405-003](../../../tasks/done/v405-003-audit-placement-propagation.yml): Pin Audit placement selection and propagate it to every child Run.
- [V405-004](../../../tasks/done/v405-004-placement-label-picker-ui.yml): Add ordinary Audit labels, freeze them at start and inherit them into child Runs.
- [V405-005](../../../tasks/done/v405-005-placement-process-release-gate.yml): Verify cloud and local placement across real processes and fold proposal 0043.
