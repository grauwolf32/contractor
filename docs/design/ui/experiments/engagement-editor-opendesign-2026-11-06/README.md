# Engagement editor · OpenDesign preview — 2026-11-06

V429-035 is the independently deliverable design part of the requested HTTP configuration work. V411-017 owns API/CLI, and V411-018 owns the complete HTTP Audit setup/evidence journey. Their pending Workflow/report/common-API/UI prerequisites remain pending; this preview does not claim those integrations are implemented. Completed V444-003 and V449-003 provide the current reduced policy and ordinary typed Run-input boundary.

The policy contains only `schema`, `egress`, `identities` and optional `limits`. Egress exposes exact origins, path prefixes, methods and CIDRs, with separate exclusions. One anonymous identity remains required; named Basic/Bearer/cookie identities select matching non-secret Runtime credential references. Changing kind clears the selected credential instead of silently choosing another. Role/tenant remain declarations. Materials, execution budgets/deadlines and Workflow capabilities stay outside engagement.

All five optional HTTP limits begin empty and stay omitted until explicitly configured. Clearing the last configured limit removes the `limits` object. Values are positive uint32 integers and apply independently to each allocation. There is no default quota, global usage counter, login/refresh recipe or automatic authentication claim.

## Genuine native contribution

OpenDesign generated `opendesign.css` in the new project `contractor-engagement-design-20261106`, with the existing configured Contractor model unchanged. Only synthetic references and the current theme were supplied. Exact inputs are retained in `native-inputs/`, original CSS is unchanged, and `provenance.json` records its SHA-256 and terminal native Run. `seed.html`, `editor.js` and fixture wiring are host-owned. `host-polish.css` separately fixes mobile section-heading/action wrapping and the native error-link token after visual QA; it is not described as generated output.

The existing unprivileged, read-only container keeps its internal-only network and has no host repository mount. Its expiring single-model relay used 18 of 24 permitted requests, then stopped and had its upstream credential removed. Recoverable pre/post state archives remain outside Git. `preservation.json` checks previous files and stable SQLite rows, including OpenCode history. Only the shared OpenCode project's access timestamp and append-only operational log change normally; original content/history is retained.

## Interactive review

```sh
node docs/design/ui/experiments/engagement-editor-opendesign-2026-11-06/prepare.mjs
node docs/design/ui/experiments/engagement-editor-opendesign-2026-11-06/verify.mjs
```

Open `/test-results/engagement-editor/index.html` through the local Vite server on 127.0.0.1:17560. Version offers OpenDesign, baseline and comparison. State shows local validation, stale revision, missing credential, Server rejection and read-only examples. Theme and width cover light/dark/black and 1440/1100/390/320px.

This preview edits a local synthetic draft and exports JSON; Save never publishes a revision or sends traffic. Conflict/rejection examples are explicitly simulated. It reads no credential values, uses no browser persistence for secrets and calls no public API. The only browser reads are local UI assets/schema/modules. Local field checks help editing without claiming canonical CIDR/DNS/Server admission or authentication success. Production integration must reuse ordinary exact-CAS Artifact writes, paginated safe credential metadata and authoritative Server validation.

Preparation reads the actual shared schema and uses the project's existing JSON-schema library for shape/bound validation. Label/help association and form interactions are host-owned; changing a field keeps focus. `verification.json` records responsive/state checks, exact export, optional-limit omission, kind/reference selection, route removal and conflict draft preservation. The exported example is also accepted by the existing Runtime Engagement model. Captures are retained in `review/`. No production source, API contract or UI package version changes in this design-only task, and nothing is merged into main.
