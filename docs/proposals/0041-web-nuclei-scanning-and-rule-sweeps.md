# Web-oriented Nuclei scanning and single-rule project sweeps

Proposal 0041 · Status: **Draft.** Recorded on 2026-10-23 and checked against
`main @ 5677eda0`. It describes a web-oriented Nuclei scan primitive, a custom
rulepack input, and a way to run one chosen rule across a scope of Projects. It
is a design recommendation, not an enabled capability or an implementation
commitment.

The driving case: a new Nuclei template (for example a fresh CVE check) is
published, and the operator wants to run **only that rule** against the live
endpoints of **several Projects** at once — the seven reconstructed DOM.RF app
backends being the concrete example, each a Project with an `httpTarget` set.
Today Contractor cannot do this:

- the only loaded Nuclei path, the Audit profile `openapi-nuclei-scan@1`, pins
  one fixed template (`http-missing-security-headers`) as a literal in
  `configs/agent-templates/audit_nuclei_scan.yaml`, caps the run at 16 GET
  operations (`configs/audit-profiles/openapi_nuclei_scan.yaml`,
  `maxItemsTotal: 16`), replays no POST/body/auth, and is in practice a
  contract fixture — its example README states the inputs "exercise V62-009
  preparation, inventory and trusted scan execution";
- the model-free standalone catalog `configs/scan` (which has `nuclei-target@1`)
  is not loaded by the running Server, whose `operatorConfigRoot` points at the
  main `configs/` tree; and even `nuclei-scan@1` does not expose the rule
  selection as a parameter;
- an Audit is strictly per-Project (`POST /v1/projects/{id}/audits`) and its
  only scan inventory, `openapi-scans@1`, is OpenAPI-driven and Go-registered
  (`internal/config/auditprofile/audit_inventory.go:9`,
  `internal/auditdomain/inventory.go:279`, allowed only in `risk-assessment`
  mode, `internal/config/auditprofile/audit_profile.go:790`). There is no
  origin/host inventory and no cross-Project launch; `/v1/audits` is read-only.

## User outcome

An operator who learns of a new Nuclei rule uploads or names it, selects a set
of Projects, and starts one scan sweep. Each selected Project runs that rule
against its own `httpTarget`, and the per-Project result is grouped under that
Project. The operator controls the ruleset and the active-traffic limits as
ordinary, reviewable settings rather than hand-written JSON, and can supply a
custom rule the installed template set does not contain.

A single-URL web scan with a chosen ruleset is also available on its own,
without an OpenAPI document and without the Audit coordination machinery.

## Current boundary

- **The scanner already accepts the knobs.** `scan_nuclei`
  (`runtime/src/contractor_runtime/toolsets/scan/tools.py:314`) takes
  `template_ids`, `tags`, `severity` (validated against
  info/low/medium/high/critical/unknown), `rate_limit` (1..1000, default 10)
  and `timeout_seconds` (1..3600, default 300), mapping them to `-id`, `-tags`,
  `-severity` and `-rate-limit`. Empty `template_ids` selects all installed
  templates. It already disables template updates, redirects, the proxy route
  and OAST, serializes calls, and enforces the Runtime target policy
  (`scan_target_denied` for metadata, Runtime-service, loopback and link-local
  destinations unless the operator allowed the network). The gap is *plumbing*:
  the shipped templates pin these as literals, so no Run or Project setting can
  choose them.
- **Rule source is a directory, not an artifact.** `scan_nuclei.prepare`
  requires an installed templates directory and passes `-t <dir>`
  (`tools.py:307`). There is no path for an operator to supply a single custom
  template at Run time. The precedent for an artifact-supplied scanner input
  already exists: `scan_ffuf(url, wordlist_ref)` resolves an exact artifact,
  writes it into the private workspace and passes it to the tool
  (`tools.py:599`, SCN-R94).
- **Two planners, one literal rule constraint.** A single-target scan uses the
  `passthrough@1` Planner, whose AgentTemplate `arguments` may bind
  `{source: parameter, ...}` (the existing `url` binding does). A multi-target
  scan uses `scan-plan@1` over a `text/vnd.contractor.target-list` artifact,
  but there **"Other arguments must be literals in the pinned template"**
  (`configs/scan/SCAN_PLAN.md`): per-job `template_ids` cannot vary. So an
  *arbitrary* chosen rule can be a parameter only on a `passthrough@1`
  single-target Workflow; breadth across Projects must come from launching many
  such Runs, not from one target-list job.
- **No Project target plumbing for scanners.** `httpTarget` is decrypted only
  for a logical agent whose AgentTemplate selects `http_request`, and travels
  as allocation-private `runtimeSettings` (PRJ-R81, PRJ-R82). Scan tools take
  their `url` from a parameter and never see `httpTarget`. A scan of a
  Project's target must therefore pass the URL explicitly today.
- **Standalone vs Project Runs.** A standalone `POST /v1/runs` resolves
  artifacts in UserScope and has no Project membership; a `POST
  /v1/projects/{id}/runs` groups the Run under the Project and resolves
  artifacts in ProjectScope (PRJ-R22, PRJ-R97). Either accepts string
  parameters unchanged.

## Design

The design is a single-target web scan primitive (Phase 1), a custom rulepack
input (Phase 2), and a first-class scope sweep with a settings widget
(Phase 3). Phases 1 and 2 are standalone-Run features; ASC-R91 (which forbids a
*model* from choosing scanner arguments and templates in an Audit) does not
apply, because the operator supplies the values at Run creation and they are
pinned in the Run snapshot — not returned by a model, and not an Audit.

### 1. Phase 1 — a parameterized single-target web scan Workflow

A new Workflow `nuclei-web@1` and AgentTemplate `nuclei-web-scan@1` in the
loaded `configs/` tree, built on `passthrough@1` and the existing `scan@1` /
`scan_nuclei` / `local-workdir@1` stack, exposing the scanner knobs as typed
Workflow string parameters:

```text
nuclei-web@1
  parameters:
    target        required   # absolute http(s) URL
    template_ids  optional   # comma-separated IDs/wildcards; empty = all installed
    tags          optional   # comma-separated template tags
    severity      optional   # comma-separated nuclei severities
    rate_limit    optional   # 1..1000, default 10
  inputs: {}
  outputs: { report: application/json, primary }
```

The AgentTemplate binds each argument with `{source: parameter, name: ...}` and
keeps `timeout_seconds` a literal matched to the Stage deadline. Parameter
values are validated on the Server against the same bounds the tool enforces,
so an out-of-range `rate_limit` or an unknown `severity` fails Run creation, not
the Worker. "Run only the new rule" is then
`run create nuclei-web@1 --param target=<url> --param template_ids=<rule-id>`.

This does not touch the Audit scan path or the `openapi-nuclei-scan@1` fixture.

### 2. Phase 1 — a single-rule sweep across a scope of Projects

Because breadth cannot be one job (the literal-rule constraint above), a sweep
is a bounded fan-out of Phase 1 Runs, one **Project Run** per selected Project:

```text
for each Project p in scope:
  POST /v1/projects/{p}/runs
    workflow = nuclei-web@1
    params   = { target: p.httpTarget, template_ids: <rule-id>, ... }
    idempotency-key = sweep-scoped per (sweep, project)
```

Each Run is grouped under its Project and visible in that Project's Runs and the
global Queue. A Project with no `httpTarget` is skipped with an explicit
reason. In Phase 1 the fan-out is a documented client loop (a small script, as
the CLI already supports `project` and `run create`); making it one server call
is Phase 3. The target URL is passed explicitly because scanners do not read
`httpTarget` (see Current boundary); letting `target` default to the Project
`httpTarget` is a later enhancement recorded under Phase 3.

### 3. Phase 2 — a custom rulepack artifact

An optional `rulepack` artifact input to `nuclei-web@1`, carrying one or more
Nuclei templates, resolved and materialized into the private workspace and
passed to the scanner as an additional `-t <dir>`, mirroring the ffuf
`wordlist_ref` pattern exactly (`tools.py:599`):

- a new `scan_nuclei(url, ..., rulepack_ref=...)` parameter taking an exact
  ArtifactRef; the AgentTemplate binds it `{source: artifact, name: rulepack}`;
- a dedicated bounded media type, `application/vnd.contractor.nuclei-rulepack+zip`
  for a pack and `application/yaml` for a single template, with explicit limits
  (total bytes, entry count, per-entry bytes), a flat extraction that rejects
  absolute paths, `..`, symlinks and non-template entries, and a parsed-as-YAML
  check before launch;
- when `rulepack_ref` is set and `template_ids` is empty, only the custom pack
  runs (`-t <rulepack-dir>` alone, not the installed set), so "run only my new
  rule" needs no knowledge of installed IDs;
- all existing target-policy, no-proxy, no-OAST and serialization guarantees are
  unchanged; the rulepack is data, never a way to reach a denied destination.

The report records the exact `rulepack` ref and digest alongside the existing
input digest, so a result names the rules it ran.

### 4. Phase 3 — first-class scope sweep and a settings widget

- **Scope sweep resource.** A server-side sweep that accepts a rule selection
  (and optional `rulepack` ref) plus a Project scope, and creates the per-Project
  Phase 1 Runs under one sweep identity with per-(sweep, Project) idempotency,
  bounded scope size, and a read view of each Project's Run and result. It adds
  no new Scheduler path: it is a create-many-Runs convenience over the existing
  Project Run endpoint, like the Audit Controller is over child Runs, not a new
  execution engine.
- **`target` defaulting.** The sweep fills each Run's `target` from the
  Project `httpTarget`, so the operator supplies only the rule and the scope.
- **Settings widget.** A UI editor that authors the rule selection and limits
  as a form — ruleset (IDs/tags/severity), `rate_limit`, optional rulepack
  upload — instead of hand-written JSON, for both a single `nuclei-web@1` Run
  and a sweep. "General settings" (a reusable default ruleset and limits per
  Project or owner) are presentation/orchestration state over these same typed
  parameters; they never become model-visible or change the pinned Run snapshot
  semantics.

The existing `openapi-scan-settings.v1` artifact keeps its narrow job —
selecting OpenAPI operations and their path/query values for the OpenAPI-driven
Audit scan — and is not extended to carry rulesets or limits; those belong to
the Workflow parameters and the rulepack input.

## Failure, lifecycle and bounds

- Invalid parameters (`rate_limit` out of range, unknown `severity`,
  non-absolute or non-http(s) `target`, a `target` the Runtime target policy
  denies) fail Run creation or Stage preparation deterministically, never a
  silent clean scan (the tool already treats timeout and overflow as failures).
- A `rulepack` that is too large, has too many or oversized entries, escapes
  its directory, or does not parse as a Nuclei template fails before launch with
  a bounded error; no partial pack is scanned.
- A sweep is bounded by a maximum scope size; each per-Project Run is an
  ordinary Run with the existing lifecycle (cancel, retry, recovery). A Project
  that is deleting or lacks an `httpTarget` is skipped with a recorded reason
  and never blocks the rest of the sweep.
- Active traffic: parallel Runs of one sweep are paced by the model-free
  `tool` lane pool (`maxConcurrentToolRuns`, proposal 0040). Per-job
  `rate_limit` does not combine across parallel Runs, so a sweep that targets
  several origins of the same host multiplies load there; this is the accepted
  risk already recorded for scans (0040 §6) and the sweep adds no per-host
  limit of its own.
- Rule/target/rulepack/policy changes change the input digest and the dedup
  receipt, so a changed rule is new work, not a replay (SCN-R33 analogue).

## Alternatives

| Option | Why not |
| --- | --- |
| Extend the Audit `openapi-nuclei-scan@1` profile to run arbitrary rules | It is OpenAPI-driven, per-Project, capped at 16 GET operations, and pins templates for the execution-contract fixture; widening it reintroduces model/arbitrary-string template selection that ASC-R91 forbids |
| A new origin/host **Audit** inventory builder | A new Go inventory implementation and Audit coordination for what is a single-URL scan; the user's own view is that origin scanning is a separate scan, not an Audit |
| One `scan-plan@1` job over a target-list of all Project origins | `scan-plan@1` requires `template_ids` to be a literal in the template, so an arbitrary new rule could not be a parameter; good for a fixed ruleset, not for "run this one new rule" |
| Put ruleset and limits into the `openapi-scan-settings.v1` artifact | Conflates operation selection with scanner policy, needs a cross-language schema change, and still does not address the non-OpenAPI web case |
| Install the custom rule on the Runtime instead of an artifact | Not per-Run, not auditable, and requires operator host access for every new rule; the ffuf-wordlist artifact pattern already solves this cleanly |

## Decision record

- **Guarantee.** An operator can run a chosen Nuclei rule (installed ID, tag,
  severity, or a custom rulepack artifact) against one URL, and across a bounded
  scope of Projects via one Project Run each, with the rule and limits pinned in
  each Run snapshot and never chosen by a model.
- **Evidence.** `scan_nuclei` arguments (`tools.py:314`), the ffuf wordlist
  precedent (`tools.py:599`, SCN-R94), the `scan-plan@1` literal-argument rule
  (`SCAN_PLAN.md`), the per-Project Audit/Run endpoints (PRJ-R22), and the
  OpenAPI-only scan inventory (`audit_inventory.go:9`).
- **Reversal condition.** Revisit the standalone-Run shape if a sweep needs
  cross-Project coordination (shared baseline, coverage, joint review) that only
  the Audit machinery provides; revisit the fan-out if a single target-list job
  with a fixed ruleset turns out to be the common case.

## Decisions

Taken by the user on 2026-10-23:

1. Build a web-oriented Nuclei scan as a **standalone scan, not an Audit**; the
   Audit path stays OpenAPI-driven.
2. The shipped `openapi-nuclei-scan@1` is treated as a test/contract fixture and
   is not promoted into the real scanner.
3. Deliver origin/single-URL scanning first; OpenAPI-driven parameterization, if
   wanted, is a later, separate task.
4. Write this proposal first; implement after review.

Open for the user to confirm during review: the sweep as a client loop in
Phase 1 vs. a first-class resource in Phase 3; the maximum sweep scope size; and
whether `target` should ever default from the Project `httpTarget`.

## Specification and contract changes

- [scanners](../spec/tools/scanners.md): `scan_nuclei` gains the
  `rulepack_ref` input and its media types, bounds and validation beside the
  ffuf wordlist rules (SCN-R94 area); new LIM-SCN entries for rulepack size,
  entry count and per-entry bytes.
- [Workflows](../spec/configuration/workflows.md) /
  [AgentTemplates](../spec/configuration/agent-templates.md): the
  `nuclei-web@1` parameters and the parameter→argument bindings; Server-side
  parameter bound validation.
- [Projects](../spec/artifacts/projects.md) and the public API: the Phase 3
  sweep resource and the optional `target`-from-`httpTarget` defaulting; no
  change to the per-Project Run contract itself.
- OpenAPI: the sweep endpoints and the rulepack media types; within `/v1`,
  additive only.
- `configs/`: `nuclei-web@1`, `nuclei-web-scan@1`, and scan instructions; these
  are new catalog identities (additive, CAT-R74).
- Shared fixtures: a rulepack validation fixture exercised by Go and Python
  (`make test-wire-cross-language` analogue), per AGENTS.md §3.1.
- UI: the scan settings widget and the sweep launch/results views.

## Verification

- Phase 1: `nuclei-web@1` runs with each knob, including empty `template_ids`
  (all installed), a single `template_ids`, a tag set and a severity set;
  out-of-range `rate_limit`, unknown `severity` and a policy-denied `target`
  fail at creation/preparation, not as clean scans; the report records the
  resolved arguments and input digest.
- Sweep: a fan-out over a scope creates one grouped Project Run per Project,
  skips a Project without `httpTarget` with a reason, is idempotent per
  (sweep, Project) on replay, and respects the scope-size bound.
- Phase 2 (real Runtime with Nuclei installed): a custom single-template
  rulepack detects what the installed set does not; an oversized, escaping or
  non-template pack fails before launch; `rulepack` with empty `template_ids`
  runs only the custom pack; the report names the rulepack ref and digest.
- Phase 3: the sweep resource and `target` defaulting; the settings widget
  authors valid parameters and never writes secrets or model-visible state;
  render at 390 px in light, dark and black themes.
- The mandatory scan process gate (`make test-openapi-audit-scan-e2e` analogue
  for the new Workflow) with disposable PostgreSQL and installed Nuclei.

## Proposed tasks

1. Server + configs: `nuclei-web@1`, `nuclei-web-scan@1`, Server parameter
   bound validation, spec and OpenAPI; the Workflow has a caller only with the
   parameters, so they land together (AGENTS.md §3.7).
2. Runtime + contract: `scan_nuclei` `rulepack_ref`, media types, extraction and
   validation, the shared cross-language fixture.
3. Server + API: the Phase 3 sweep resource, scope bounds, idempotency, and
   `target`-from-`httpTarget` defaulting.
4. UI: the scan settings widget and the sweep launch/results views.
