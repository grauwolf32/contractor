# Operations tabs · native OpenDesign exploration

User approved the Operations overview and requested the other tabs on 2026-11-03.
This experiment extends that design with all seven remaining top-level sections
and the existing Completed allocation view. It leaves production code unchanged.

## Ownership and source references

Existing components and contracts define the field meanings and supported actions:

| Section | Source | Important boundary |
| --- | --- | --- |
| Runtime Agents | `ui/src/routes/operations/runtime-agents/` | Durable principal/labels are separate from the current single-slot process; availability is reported, not Workflow compatibility. |
| Current allocations | `ui/src/routes/operations/allocations/index.tsx` | Authoritative phase and Runtime observation remain separately labelled. Prepared is not a call in flight. |
| Completed allocations | `ui/src/routes/operations/allocations/completed.tsx`, `performance/resources.tsx` | Include available, partial, disabled, unsupported, pending and missing reports. Runtime process resources exclude subprocesses and containers. |
| Performance | `ui/src/routes/operations/performance/`, `ui/src/api/performance.ts` | Snapshot freshness and metric units are explicit. Unknown GPU/measurements stay unknown. |
| Configuration | `ui/src/routes/operations/runtime-configs/` | Publishing an immutable version does not bind a label. Runtime service credentials are a distinct kind of resource. |
| LLM configurations | `ui/src/routes/operations/llm-configurations/` | Kind filter, exact identity/digest, inspect/clone policies and gateways; templates read-only. |
| Credentials | `ui/src/routes/operations/credentials/` | Managed LLM gateway credentials and observed budgets. Absent consumption is not zero. |
| Settings | `ui/src/routes/operations/settings/`, `ui/src/routes/settings/` | Server-wide Scheduler/model defaults, personal Git SSH key and browser appearance are separate scopes. |

The Operations specification owns these semantics. The preserved prior overview
seed supplies public-schema-validated process/allocation snapshots, shared Icon
markup, exact global fonts and theme definitions. Additional tab fixtures are
explicit synthetic **view examples**, not wire fixtures or live inventory. The
candidate does not imply complete production form behavior: dialogs illustrate
labels/bindings/publication/credential layout, while secret payloads and exact
kind-dependent document editors are deliberately absent. Local forms report only
local validation and never claim a Server save, generate keys or change models.
The Runtime publish example chooses a base version; production transfer must
retain the full existing authoring document and validation. The clone text area
is a labelled layout placeholder, not a publishable configuration document.

## Review

```sh
node docs/design/ui/experiments/operations-tabs-opendesign-2026-11-03/serve.mjs 17569
node docs/design/ui/experiments/operations-tabs-opendesign-2026-11-03/verify.mjs
```

Preview has real section switching through the left navigation and toolbar;
Current/Completed, connection and LLM-kind filters; native diagnostic disclosures;
local dialogs; history filtering; Performance range selection; and local setting
controls. 15-minute fixed samples are illustrative; other ranges show no demo
samples. The wrapper supports 1440/1100/390/320px or a live viewport and all themes.
Read-failure state uses unknown values; empty inventory is explicit. Reconnection
keeps the last snapshot, and Performance labels its measurements stale.

The reviewed overview is served from its original artifact, with a separate
navigation bridge to the new wrapper. That file and its native inputs are never
modified. The new experiment inherits the original overview CSS as a base; the
new native contribution is a separate CSS block for the new tabs and dialogs.
Every section still uses the project font families and theme tokens.

The wrapper and sandboxed iframe CSP prevent external requests, form navigation,
remote images/fonts and production writes. The sandbox permits local form events; form-action none still forbids actual
form navigation and the controller cancels submissions. Unsupported deep destinations announce
the actual path locally. This is a design preview; production transfer, complete
form behavior and main merge require subsequent user direction.

## Native output and bounded recovery

Native project `contractor-operations-tabs-20261103`, conversation
`e755ae95-9804-4026-8ae6-6cd3bbda8677`, run
`24594eb2-4e5f-4756-9c15-7ba2409f4732` used the unchanged Contractor model.
The run produced complete `opendesign.css` and then **failed** on the fixed
24-request gateway budget before assembling HTML or writing its rationale.
No second run or larger model budget was used.

`opendesign.css` is the exact native output. `assembled.html` and byte-identical
`preview.html` are host-assembled: the only change from `seed-submitted.html` is
one style element containing those exact CSS bytes immediately before `</head>`.
`assembly.json` records this boundary and all hashes. No manually corrected CSS
is presented as native output; no successful native finalization is claimed.
The native project/output and failed receipt stay intact, including in recovery
archives. The generated CSS refines inventory density, segmented filters,
allocation phase styling, metric typography, settings scopes and native dialogs.

Inputs, exact CSS, assembly manifest, prompt, run identity, verified before/after
archives and proof that all old files/history were preserved are retained here.


## Verification

`verify.mjs` passes 96 section/theme/width layouts and 24 empty/read-failure/
reconnecting boundaries, plus native CSS/input/assembly hashes, known tokens,
local filters, history resource states, dialogs, scheduling bounds, labels,
configuration clone, section and Current/Completed navigation, keyboard mobile
menu and live resize. No horizontal overflow, page errors or external requests.
Sixteen selected dark screenshots and all layout metrics are retained in
`review/`; the ignored UI test output also retains light/black screenshots.
Task/spec validation and `git diff --check` pass. No production UI code changed,
so the production UI build/test gate was not run for this artifact.

Both full recovery archives passed independent SQLite integrity verification.
All 189 existing native project files and all prior Project/Conversation/Message
rows are identical. Nine new project files were added. The unchanged isolated
container has its read-only root and sole ordinary state volume; the 24-request
relay has stopped and its upstream credential was removed. Provenance retains
the failed native run status and the local assembly recovery explicitly.

Review starts at <http://127.0.0.1:17569/?section=runtime-agents&version=new&case=attention&theme=dark&width=1440>.
This branch contains the design artifacts; production transfer and main merge
remain subsequent work after user feedback.
