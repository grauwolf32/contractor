# Web-oriented Nuclei scanning and single-rule project sweeps

Proposal 0041 · Status: **Accepted for implementation.** Recorded on 2026-10-23 and checked against
`main @ 5677eda0`. It describes a web-oriented Nuclei scan primitive, a custom
rulepack input, and a way to run one chosen rule across a scope of Projects. It
is the accepted implementation plan; capabilities remain unavailable until their
implementation tasks and verification complete.

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

An operator who learns of a new Nuclei rule uploads it into their **UserScope**
or names an installed rule, selects a set of Projects, and starts one scan
sweep. The uploaded pack stays an owner-level reusable input; the operator does
not upload a separate copy into each Project. Each selected Project runs that
rule against its own `httpTarget`, and the per-Project result is grouped under that
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
  destinations unless the operator allowed the network). These are pre-launch
  checks of the supplied URL, not enforcement of every destination a template
  can contact. The shipped templates pin the knobs as literals, so no Run or
  Project setting can choose them.
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
  parameters unchanged. PRJ-R29 currently forbids a Project Run from sourcing an
  ordinary input directly from UserScope; rulepacks therefore require an
  explicit contract exception, described below.
- **Bindings do not convert or default missing inputs.** Workflow parameters
  are string slots without defaults (WFL-R4, WFL-R5, RUN-R12). A tool
  `parameter` binding passes that string unchanged, and an `artifact` binding
  requires the named artifact to be present. `ToolWorkerRuntime._resolve`
  (`runtime/src/contractor_runtime/worker/tool_runtime.py:179`) indexes both
  mappings before strict callable-schema validation. Binding a string to the
  integer `rate_limit`, or omitting a bound optional parameter or artifact,
  therefore produces `tool_input_invalid` today. Phase 1 needs the explicit
  binding extension below, not only new YAML configurations.
- **Installed IDs do not pin template bytes.** The Runtime chooses a local
  templates directory, which can differ between Agents or change independently
  of a Run. The existing tool input digest covers arguments and refs, not that
  directory. A reproducible web scan must pin the installed package content as
  well as its selection filters.

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
`scan_nuclei` / `local-workdir@1` stack, exposing the scanner knobs as Workflow
string parameters with explicit tool-argument conversion and defaults:

```text
nuclei-web@1
  parameters:
    target        required   # absolute http(s) URL
    template_ids  optional   # comma-separated IDs/wildcards; empty = all installed
    tags          optional   # comma-separated template tags
    severity      optional   # comma-separated nuclei severities
    rate_limit    optional   # canonical decimal string, 1..1000; omitted = 10
  inputs: {}
  outputs: { report: application/json, primary }
```

The existing parameter transport stays string-to-string. Extend the tool
argument binding contract explicitly, before shipping this Workflow:

```yaml
url: {source: parameter, name: target}
template_ids: {source: parameter, name: template_ids, default: ""}
tags: {source: parameter, name: tags, default: ""}
severity: {source: parameter, name: severity, default: ""}
rate_limit: {source: parameter, name: rate_limit, parse: integer, default: 10}
```

- `parse: integer` is an explicit conversion available only to a `parameter`
  binding. A supplied value must match `0|[1-9][0-9]*`, then satisfy the
  scanner's range. Signs, whitespace, leading zeroes, decimal points, exponents
  and an empty string fail. Bindings without `parse` still pass supplied
  strings unchanged; there is no implicit conversion.
- `default` is allowed only on a `parameter` binding and is a non-null scalar
  of the resulting argument type. It applies only when the parameter is absent.
  An explicitly supplied empty or invalid value never selects the default.
  A missing parameter without a default fails before invocation.
- Server validates the projected Nuclei arguments at Run admission using the
  resolved bindings and the scanner-owned input contract. Runtime performs the
  same projection and validation before execution. Shared fixtures cover the
  grammar, bounds, defaults and severity/token rules. The persisted public
  `Run.parameters` mapping retains exactly the supplied strings; defaults and
  integers belong to the resolved tool arguments and their input digest.

This extends SCN-R6 to SCN-R10 and clarifies RUN-R12 for explicit tool binding
validation; it does not introduce typed Workflow parameter values or Workflow
slot defaults. The Server must not special-case the Workflow's name to perform
conversion. `timeout_seconds` remains a literal matched to the Stage deadline.
"Run only the new rule" is then
`run create nuclei-web@1 --param target=<url> --param template_ids=<rule-id>`.

**Installed rule identity.** For installed-rule mode, the new AgentTemplate
pins a literal `templates_digest` for the operator-provisioned installed
template package.
The digest covers a canonical, sorted manifest of package-relative names and
file digests, including resources consumed by templates. It is part of the
resolved AgentTemplate and therefore every Run snapshot and tool input digest.
Before launch, Runtime materializes a bounded private snapshot and verifies its
manifest against that digest; scanning uses those verified bytes. A missing or
different package fails preparation without substituting the local installed
set. A changed package needs an updated pinned template configuration for new
Runs. Reports record the expected and actual package digest, selected template
IDs and content digests, and the Nuclei engine version. Replacement allocations
must satisfy the same pin.

This does not touch the Audit scan path or the `openapi-nuclei-scan@1` fixture.

### 2. Phase 1 — a single-rule sweep across a scope of Projects

Because breadth cannot be one job (the literal-rule constraint above), a sweep
is a bounded fan-out of Phase 1 Runs, one **Project Run** per selected Project:

```text
manifest = persist(sweep_id, selected_project_ids, exact_child_requests)
for each pending Project p in manifest:
  POST /v1/projects/{p}/runs
    workflow = nuclei-web@1
    params   = manifest[p].params  # includes the captured p.httpTarget.url
    idempotency-key = sweep-scoped per (sweep, project)
```

Before the first POST, the client captures the bounded Project selection,
target URLs, supplied parameters, exact artifact revisions and the keys in a
local resumable manifest. It records skipped Projects and created Run IDs there.
After a lost response it repeats the exact captured request with the same key.
Resume never re-reads a newer `httpTarget` or a newer artifact binding to rebuild
a child request. Changing inputs creates a new sweep identity and new keys.

Each Run is grouped under its Project and visible in that Project's Runs and the
global Queue. A Project with no `httpTarget` is skipped with an explicit
reason. In Phase 1 the fan-out is a documented client loop (a small script, as
the CLI already supports `project` and `run create`); making it one server call
is Phase 3. The target URL is passed explicitly because scanners do not read
`httpTarget` (see Current boundary); letting `target` default to the Project
`httpTarget` is a later enhancement recorded under Phase 3.

### 3. Phase 2 — a custom rulepack artifact

An optional `rulepack` artifact input to `nuclei-web@1`, carrying one or more
Nuclei templates, uploaded once into the authenticated owner's **UserScope**.
It is selected by an artifact ref, resolved to an exact revision, forked into
RunScope and materialized into the private workspace for `-t <dir>`. The ffuf
`wordlist_ref` pattern supplies the Runtime materialization precedent
(`tools.py:599`); the owner-level source for Project Runs is a new contract.

**Source scope and authorization.** The Workflow's `rulepack` input declares
`sourceScope: user` alongside `required: false` and its allowed media types.
This is a narrow, catalog-declared exception to PRJ-R29 for scanner rulepacks.
Both standalone and Project Run admission derive UserScope from the
authenticated owner, check access and media type, resolve the source once and
fork that exact version to `RunScope(run_id): inputs/rulepack`. Other Project
inputs retain their ProjectScope semantics. The Run request gains no arbitrary
owner/scope selector, and Runtime continues to see only RunScope. Source
revision and content digest are recorded in Run provenance. The user-owned
source is reusable across Projects and is never copied into ProjectScope.

**Optional binding.** Extend artifact bindings with `optional: true`:

```yaml
rulepack_ref: {source: artifact, name: rulepack, optional: true}
```

When the input is absent, the resolver omits this callable argument and the
callable's `None` default applies. Allocation preparation must check that the
selected callable permits omission. A supplied but invalid, inaccessible,
wrong-media-type or missing-revision ref fails; it never becomes an absent
input. `optional` is valid only for artifact bindings. This requires a
Server/Runtime wire and validation change; merely marking the Workflow slot
optional is insufficient.

The scanner contract adds:

- a new `scan_nuclei(url, ..., rulepack_ref=None)` argument taking an exact
  RunScope ArtifactRef when present;
- a dedicated bounded media type, `application/vnd.contractor.nuclei-rulepack+zip`
  for a pack and `application/yaml` for a single template, with explicit limits
  (total bytes, entry count, per-entry bytes), a flat extraction that rejects
  absolute paths, `..`, symlinks and non-template entries. Validate the complete
  pack as supported Nuclei HTTP templates before scanning; YAML syntax alone
  is insufficient. One invalid entry rejects the whole pack;
- when `rulepack_ref` is present, the pack is the sole template source. The
  IDs/tags/severity filters apply within it; empty `template_ids` selects the
  pack without adding installed templates. There is no implicit union or
  fallback to the installed package. Custom-pack mode does not require an
  installed templates directory; the installed-package digest check applies
  only in installed-rule mode;
- existing pre-launch target checks, no-proxy, no-OAST and serialization
  behavior remain. Isolation of destinations chosen inside a custom template
  is outside the acceptance scope of this proposal, by the user's review
  decision on 2026-10-23. The owner chooses the pack, and its recorded digest
  makes content changes visible. Reopen that scope if pack execution crosses
  an owner trust boundary or destination isolation becomes a required guarantee.

The report records the exact RunScope `rulepack` ref, its content digest and
selected template IDs/digests alongside the input digest. Run provenance links
that fork to the owner's exact UserScope source. Editing or deleting the source
binding after admission does not change a Run's fork; ordinary Run retention
and deletion own that fork, and the owner's source keeps its independent
lifecycle. Scratch files are removed after completion, failure, timeout,
cancellation and allocation release.

### 4. Phase 3 — first-class scope sweep and a settings widget

- **Scope sweep resource.** A server-side sweep that accepts a rule selection
  (and optional owner-UserScope `rulepack` ref) plus a Project scope, and creates
  the per-Project Phase 1 Runs under one sweep identity with per-(sweep, Project) idempotency,
  bounded scope size, and a read view of each Project's Run and result. It adds
  no new Scheduler path: it is a create-many-Runs convenience over the existing
  Project Run endpoint, like the Audit Controller is over child Runs, not a new
  execution engine.
- **Admission snapshot and recovery.** Before fan-out, persist the owner,
  concrete Project IDs, resolved Workflow/template pins, target URLs, defaulted
  arguments, exact source revisions/digests and canonical per-Project requests
  with their keys. Pending, admitted, skipped and failed members and their Run
  IDs form the resumable read view. Persist a retained ArtifactStore fork of
  selected source inputs before dispatch, so deleting or updating the original
  UserScope binding cannot change pending members. The trusted child-admission
  path uses this authorized snapshot and its exact bytes without resolving a
  current UserScope binding. These are sweep-owned input snapshots, never
  ProjectScope copies or a new model-visible scope.
- **Mutation ownership.** Use the existing keyed-mutation mechanism for sweep
  creation and existing controller claims for bounded fan-out. A lost child
  creation response is reconciled with the captured request and key. Controller
  restart resumes pending members; it never repeats active scans independently
  of the ordinary Run machinery. No additional receipt or lease mechanism is
  introduced.
- **`target` defaulting.** The sweep fills each Run's `target` from the
  captured Project `httpTarget.url`, so the operator supplies only the rule and
  the scope.
- **Settings widget.** A UI editor that authors the rule selection and limits
  as a form — ruleset (IDs/tags/severity), `rate_limit`, optional rulepack
  selection/upload in the owner's UserScope — instead of hand-written JSON,
  for both a single `nuclei-web@1` Run and a sweep. "General settings" (a reusable
  default ruleset and limits per Project or owner) are presentation/orchestration
  state over these same string parameters and artifact inputs; they never
  become model-visible or change the pinned Run snapshot semantics.

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
- Fan-out ticks, input snapshotting and result lists have independent row/byte
  bounds and pagination. Permanent admission errors settle the affected member;
  retryable storage failures use bounded backoff. A lost acknowledgement keeps
  the original key and captured request. Sweep cancellation stops new child
  admissions and requests ordinary cancellation of admitted Runs; it reports
  their actual states until settlement. Sweep input snapshots remain retained
  through recovery and child admission, then follow sweep retention. Purge
  joins controller work and removes only sweep-owned state and snapshots; child
  Runs and the owner's source rulepack keep their ordinary lifecycles.
- Active traffic: parallel Runs of one sweep are paced by the model-free
  `tool` lane pool (`maxConcurrentToolRuns`, proposal 0040). Per-job
  `rate_limit` does not combine across parallel Runs, so a sweep that targets
  several origins of the same host multiplies load there; this is the accepted
  risk already recorded for scans (0040 §6) and the sweep adds no per-host
  limit of its own.
- Replay preserves the captured request, source revisions, installed-package
  digest and child key. A changed input under an existing create key is
  `idempotency_conflict`; a changed resolved tool input under an existing
  execution key is `tool_input_conflict` (SCN-R33). A changed digest does not
  create a new receipt. Changed rules, targets, limits or pinned execution
  policy require an explicitly new Run or sweep identity. Runtime deployment
  target policy is rechecked before launch and may refuse execution; the
  existing input digest does not claim to fingerprint that external policy.

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
  scope of Projects via one Project Run each. Parameters are projected through
  explicit typed/defaulted bindings. The installed package digest or exact
  UserScope rulepack revision and content digest, plus limits, are pinned for
  execution. The owner uploads a custom pack once; trusted Server input forks
  supply it to each Run. Models choose none of these inputs. Replay reuses the
  captured input snapshot and identity.
- **Evidence.** `scan_nuclei` arguments (`tools.py:314`), the ffuf wordlist
  precedent (`tools.py:599`, SCN-R94), the `scan-plan@1` literal-argument rule
  (`SCAN_PLAN.md`), the per-Project Audit/Run endpoints (PRJ-R22), and the
  OpenAPI-only scan inventory (`audit_inventory.go:9`). Review additionally
  checked `ToolWorkerRuntime._resolve`, WFL-R4/WFL-R5, RUN-R12, PRJ-R29 and
  SCN-R31 to SCN-R34: they establish the required binding/scope extensions and
  existing conflict behavior; they do not already provide the proposed features.
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
5. The user uploads a custom rulepack into their UserScope. Reusing it across
   Projects must not require separate ProjectScope uploads.
6. Template-controlled custom-rulepack egress is outside this proposal's
   acceptance scope for now. Keep the actual pre-launch checks, remove the
   unsupported claim that they confine every request made by a custom template,
   and record the owner and reopening condition above.

The binding fields (`parse`, `default`, `optional`), the rulepack input's
`sourceScope: user` exception, installed-package digest pinning and immutable
sweep admission snapshot are proposed resolutions of the remaining review
findings. They require implementation and contract updates; none is described
as an existing facility.

Implementation uses the phased client/server sweep and bounds recorded below.
Ordinary Run target defaulting is outside scope; sweep admission captures the
Project URL explicitly.

## Specification and contract changes

- [scanners](../spec/tools/scanners.md): `scan_nuclei` gains the
  `templates_digest` and `rulepack_ref` inputs, exclusive source selection,
  package/template manifests and report provenance. Extend tool bindings with
  explicit integer parsing, typed defaults and optional artifact omission;
  document preparation failures. New LIM-SCN entries cover rulepack size,
  entry count, per-entry bytes and bounded installed-package snapshots.
- [Workflows](../spec/configuration/workflows.md) /
  [AgentTemplates](../spec/configuration/agent-templates.md): the
  `nuclei-web@1` string parameters, explicit binding projection and
  `sourceScope: user` on the optional rulepack input. Resolved definitions and
  digests cover these fields. [Runs](../spec/execution/runs.md) clarifies
  validation of the projected tool arguments while preserving supplied strings.
- [Projects](../spec/artifacts/projects.md) /
  [ArtifactStore](../spec/artifacts/artifact-store.md): explicitly amend
  PRJ-R29 with the catalog-declared rulepack UserScope-to-RunScope exception,
  owner authorization, exact source provenance, retention and replay. All other
  ordinary Project inputs keep their current source scope.
- [Public API](../spec/api/public-api.md), OpenAPI and Server/Runtime wire
  schemas: additive binding and input-slot fields and rulepack media types;
  Phase 3 sweep endpoints, admission snapshot, per-member state, pagination and
  `target`-from-`httpTarget` capture. No arbitrary source-scope request field.
- [Failure model](../spec/platform/failure-model.md),
  [lifecycle](../spec/platform/lifecycle.md) and
  [limits](../spec/platform/limits.md): binding/package validation failures,
  sweep replay/cancellation/purge, snapshot retention, bounded fan-out and
  numeric scope, page, row and byte limits before implementation starts.
- `configs/`: `nuclei-web@1`, `nuclei-web-scan@1`, and scan instructions; these
  are new catalog identities (additive, CAT-R74).
- Shared fixtures: explicit binding projection (including omission/defaults),
  scanner input validation, installed-package manifests, UserScope rulepack
  provenance and pack validation exercised by Go and Python
  (`make test-wire-cross-language`), per AGENTS.md §3.1.
- UI and CLI: the scan settings widget, UserScope rulepack selection/upload,
  sweep launch/results views and the Phase 1 resumable manifest script.

## Verification

- Phase 1: `nuclei-web@1` runs with every optional parameter omitted, with a
  single rule as in the CLI example, and with each selection filter. Server and
  Runtime agree on typed/defaulted arguments while persisted Run parameters
  retain the original strings. Missing required inputs, empty/zero/out-of-range
  rates, signs, whitespace, leading zeroes, fractions, exponents and unknown
  severity fail deterministically. A policy-denied supplied target fails before
  scanner launch. The report records resolved arguments and input digest.
- Installed rules: different template bytes under the same ID, changed
  resources, a changed local package or replacement allocation with the wrong
  package fail before launch. Mutating the installed directory after private
  snapshot preparation cannot change scanned bytes. Reports identify the pinned
  package, actually selected templates and engine version.
- Sweep: a fan-out over a scope creates one grouped Project Run per Project,
  skips a Project without `httpTarget` with a reason, is idempotent per
  (sweep, Project) on replay, and respects the scope-size bound. Lose a child
  creation acknowledgement and restart the client/controller after partial
  admission; every member retains its captured request and returns the same
  Run. Update Project targets and source bindings between attempts; resumed
  work still uses the captured values. Changed requests with old keys conflict;
  an explicitly new sweep uses new keys. Exercise cancellation and populated
  purge, independent bounded ticks and paginated results with real PostgreSQL.
- Phase 2 (real Runtime with Nuclei installed): a custom single-template
  rulepack detects what the installed set does not; an oversized, escaping or
  non-template pack, including a pack mixing valid and invalid entries, fails
  before launch. An absent optional pack still permits an installed-rule scan;
  a supplied invalid ref fails without selecting installed rules. Any supplied
  pack is the sole source, with or without ID filters. The report names the
  exact RunScope fork and template/pack digests.
- UserScope rulepack admission: upload once and use the same exact source for
  standalone and several Project Runs without ProjectScope copies. Wrong-owner
  or inaccessible refs fail atomically. Source mutation/deletion after Run or
  sweep admission cannot affect retained snapshots, pending members or replay;
  Run/sweep cleanup never deletes the owner's source artifact. Verify input
  forks, lineage and retention against real PostgreSQL/ArtifactStore.
- Phase 3: the sweep resource and `target` defaulting; the settings widget
  authors valid parameters and never writes secrets or model-visible state;
  render at 390 px in light, dark and black themes.
- The mandatory scan process gate (`make test-openapi-audit-scan-e2e` analogue
  for the new Workflow) with disposable PostgreSQL and installed Nuclei.

## Proposed tasks

1. Server + Runtime + configs: Phase 1 `nuclei-web@1` and
   `nuclei-web-scan@1`, explicit integer/default parameter bindings, Server and
   Runtime projection/validation, installed-package digest pinning and private
   snapshots, report provenance, resumable client fan-out, spec/wire/OpenAPI and
   shared fixtures. The working caller and binding extension land together
   (AGENTS.md §3.7); publishing only the YAML does not complete Phase 1.
2. Server + Runtime + contract: Phase 2 `scan_nuclei` `rulepack_ref`, optional
   artifact binding omission, `sourceScope: user` admission and authorized exact
   forks for standalone/Project Runs, media types, complete-pack validation,
   exclusive source selection, retention and shared cross-language fixtures.
3. Server + API: Phase 3 sweep admission and retained input snapshots,
   controller recovery with existing claims, scope/tick/page bounds, existing
   create-idempotency reuse, cancellation/purge and captured `httpTarget.url`.
4. UI + CLI: owner-UserScope rulepack selection/upload, the scan settings
   widget and sweep launch/resume/results views, with no ProjectScope upload
   prerequisite.

## Implementation tracking

The user authorized task creation and implementation on 2026-10-23. Milestone
`web-nuclei-rule-sweeps-2026-10` is tracked by the V406 series. The branch-local
V403 task IDs were reassigned during main integration because V403 already
identifies the artifact-preview work on main:

- [V406-001](../../tasks/done/v406-001-nuclei-web-parameter-bindings.yml): Phase 1
  bindings, web Workflow and pinned installed-rule execution (completed).
- [V406-002](../../tasks/done/v406-002-resumable-project-scan-client.yml): resumable
  Phase 1 client fan-out and real-process gate.
- [V406-003](../../tasks/done/v406-003-user-rulepack-inputs.yml): optional UserScope
  rulepacks and exact authorized input forks.
- [V406-004](../../tasks/v406-004-durable-scope-sweeps.yml): durable bounded
  server sweep and recovery/cancellation/purge.
- [V406-005](../../tasks/v406-005-web-scan-settings-ui.yml): single-scan form
  and owner rulepack selection/upload.
- [V406-006](../../tasks/v406-006-scope-sweep-ui-cli.yml): sweep UI/CLI.
- [V406-007](../../tasks/v406-007-web-scan-release-gate.yml): milestone exit
  and complete process/browser/release evidence.

The implementation defaults are a 1,000-Project maximum scope, a resumable
client loop in Phase 1, and target capture from `httpTarget.url` only at sweep
admission. Ordinary Run creation continues to require its explicit target.
Task records state the independent package, snapshot, tick and page bounds;
the owning specification registries acquire them with each implementation.
