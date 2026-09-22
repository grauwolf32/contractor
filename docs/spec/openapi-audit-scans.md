# OpenAPI Audit scan adapter

Status: **Released and verified ([V62-009](../../tasks/v62/v62-009-audit-openapi-scan-profile.yml)). Prepare-role execution that would generate the OpenAPI is specified in [19 §4.4](19-audits.md#44-preparation-contract) but not yet delivered, so prepare profiles return `preparation_unsupported`.**

## Delivery order

The scan profile takes a supplied exact OpenAPI and explicit scan settings and
runs SQLMap request checks and Nuclei URL checks through the existing scanner
Workers and Scheduler. Source preparation (generating the OpenAPI through a
`prepare` role such as `openapi-from-workspace@7`) is a separate capability
governed by [19 §4.4](19-audits.md#44-preparation-contract) and is not a
prerequisite for this profile. Round result snapshots, retained dependencies,
proposed-check routing and standalone project-analysis preparation are deferred
and are not release prerequisites.

## Assigned inputs and authority

`inventory.implementation: openapi-scans@1` selects the built-in, versioned
OpenAPI-to-scan-task builder. It is not a Workflow or AuditProfile name.
The profile chooses its Workflow by an ordinary exact `ref`; profile, Workflow,
Worker, role and Stage names are operator-authored. Audit input names can differ
from Workflow input aliases through the declared input mappings.

The builder declares the `openapi-scan@1` task contract. A consuming Workflow
explicitly selects an executor Stage:

```yaml
spec:
  auditTask:
    contract: openapi-scan@1
    stage: check-request
  entryStage: prepare-context
  # Ordinary Stages may run before and after check-request.
```

The catalog validates producer/consumer contract compatibility, then validates
the selected executor's capabilities and input mappings. `auditTask.stage` need
not be the entry Stage. Other Workflow Stages and other Audit roles are allowed.
The scan executor accepts one assigned item and one scanner Worker; its
batch-size, exact-input and approval constraints belong to that executor.
It uniquely owns the canonical check-result output. Ordinary graph validation
rejects successful paths without that required output, and no other Stage can
replace it. A second scan executor requires a different execution/result contract.

Inventory construction does not inspect the executor's scanner or entry Stage.
After construction, both input preview and Audit start check actual task
requirements against the declared executor. SQLMap settings cannot be dispatched
to a Nuclei Worker. During dispatch the manifest uses Workflow input aliases;
the task retains original producer input identities. Additional exact inputs for
surrounding Stages do not change the scanner's assigned source or settings.

The complete `auditTask` declaration is cloned and retained in immutable Workflow
and profile snapshots. Recovery reads that declaration to select the scanner's
journal; surrounding Stage outcomes cannot be mistaken for scan attempts.
Absent `auditTask` fields remain omitted on ordinary Workflows, preserving their
existing serialized shape and completion behavior.

An accepted item pins one source operation by its canonical JSON Pointer,
the original OpenAPI ref and byte digest, explicit target and operation values,
scanner policy and exact Workflow/template/configuration versions. A model or
scanner report cannot select a different operation or substitute a newer source.
Dispatch validates the original-to-Run fork mapping under the existing Audit
artifact contract. Preparation helpers receive already-authorized bytes; they
do not perform ownership checks or grant network authority.

The Audit adapter requires an explicit absolute target server selected before
start. It passes this override into preparation instead of trusting generated
OpenAPI servers as authorization. Scope and existing active-check approval remain
authoritative. Scanner arguments and template selections are pinned by the
resolved profile/Worker configuration, not arbitrary strings returned by a model.

Concrete values are supplied with operation bindings using the existing
`scanplan.Options` contract (`path:name`, `query:name`, and for SQLMap also
header/cookie/body/authentication). Examples/defaults follow
[preparation policy 2](31-scan-request-preparation.md#input-and-selection-policy);
types alone never invent values. Unsupported or missing data produce gaps. The
settings artifact is strictly validated and pinned before inventory creation.
Missing values require a corrected new Audit; resume cannot rewrite
the accepted input snapshot.

### Settings and inventory contract

`scanplan.DecodeAuditScanSettings` accepts JSON with schema
`contractor.audit.openapi-scan-settings.v1`, up to 256 KiB. Required fields are
`schema`, `scanner` (`sqlmap` or `nuclei`), `server` and nonempty `operations`.
The server is a concrete absolute HTTP(S) URL without credentials, template
variables, query or fragment. `operations` maps canonical operation pointers to
objects with optional `parameters` and `body` using the preparation types above.
Only listed operations enter the inventory; there is no implicit whole-document
scan. At most 1,000 operations are accepted. Bodies contain both `mediaType` and
`value`; an explicit JSON-null value is data, not an absent body.

SQLMap requires a nonempty `testParameters` list (up to 64 distinct names) and
allows `authentication` keyed by declared security scheme. Nuclei forbids
nonempty test parameters/authentication and any body/header/cookie bindings.
Unknown fields, case aliases, duplicate keys, null structural fields and unsafe
numbers are rejected with fixed diagnostics. Scanner arguments, templates and
execution budgets are not accepted through this artifact.

`auditdomain.BuildOpenAPIScanInventory` creates `openapi-scans` inventory with
`openapi-scan` task items; it requires active-check approval. Each task pins exact
source and settings refs/byte digests, the selected operation, preparation digest,
SQLMap request digest or Nuclei URL, explicit test parameters, runnable state and
preparation gap codes. Settings are an additional exact execution-manifest input.
Credentials and request bodies are not copied into task documents. URL query
values are retained as part of a Nuclei target.

`PrepareOpenAPIScanTask` checks source/settings bytes and reproduces the accepted
preparation before dispatch. The caller still verifies artifact ownership,
trusted manifest membership and the Project-to-Run fork mapping. A different
operation, request, target, parameters, preparation identity or gap set fails
this check. Correctly formed settings with unavailable path/body/auth data or
SQLMap parameters retain a non-runnable item with gaps. Missing operations or
malformed settings reject the inventory atomically.

The checked-in [local input examples](../../configs/scan/examples/audit-openapi-scan/README.md)
cover path/query values and an authenticated POST. Library and executor tests
verify preparation and canonical results without scanner/network calls. The
registered profiles must also pass the mandatory production-process gate below.

## Operation preparation

`scanplan.PrepareOperation` accepts one canonical `#/paths/<escaped-path>/<method>`
selector and returns a bounded RequestSet with at most that operation's request.
It retains the **original full** source ref and byte digest. It does not filter a
whole-document RequestSet after unrelated operations have consumed its budget.
Bindings for any other operation fail with `unassigned_operation_binding`.
Unknown selectors fail explicitly. An unresolved selected path or unusable
request remains one skipped operation with diagnostics, never an empty success.

The preparation identity binds the policy-2 base digest, selected operation and
`request` mode. Whole-document `Prepare` keeps its existing canonical identity.
Source parsing and size/depth/node bounds still apply to the entire document;
reference expansion and request preparation only concern the assigned operation.

SQLMap consumes the resulting RequestSet through the existing
[scan planner](32-scan-planning.md#plan-v1-and-selection). Its
nonempty `testParameters` must be explicitly selected and available in the
prepared request. The planner preserves method, URL, headers and body, and pins
the exact scanner request artifact. Excluded parameters and unsupported request
representations remain gaps; no automatic broadening is allowed.

## Nuclei fixed URLs

`scanplan.PrepareOperationTarget` produces an in-process `OperationTarget`:
original exact source, operation pointer, preparation digest, concrete URL and
limitations. `url-target` is a separate preparation identity. The URL uses the
selected server, path substitutions and query values. No missing path value is
invented. An empty URL with gaps is an unavailable target and must not be passed
to the scanner. This is deliberately not serialized as a synthetic GET request.

The URL can come from a POST operation: Nuclei receives that fixed URL and runs
its pinned templates. It does not replay the OpenAPI method, body, headers,
cookies or authenticated session. Declared requirements are retained as
`http_method_not_replayed`, `request_body_not_replayed`,
`non_url_parameter_not_applied` and `authentication_not_applied` limitations.
All targets carry `url_template_scan_only`. Explicit authentication, body,
header or cookie bindings are rejected for this interface so supplied request
data cannot be silently discarded. An explicitly anonymous URL scan may still
target an operation declaring authentication; it does not count as testing that
authenticated operation.

The adapter must retain the target provenance/limitations beside the exact
target-list artifact passed to the ordinary Nuclei planner. Deduplicating equal
URLs must preserve every operation contribution. A bare target-list artifact is
insufficient Audit provenance. Pin template configuration; template scans may
make several requests and must not be presented as one exact HTTP replay.

## Scan coverage and semantic results

The separate `openapi-operations@1` inventory continues to produce
`operation-trace` / `operation-resolution`; its canonical bytes and
`traced-complete` semantics must not change. The new `openapi-scans` inventory has
separate coverage requirements: `sqlmap-request-scan` and
`nuclei-url-template-scan`. The builder is registered as `openapi-scans@1`;
its consumers declare `auditTask.contract: openapi-scan@1` independently.

Account independently for:

- Assigned operation and prepared URL/request, including preparation gaps.
- SQLMap selected test parameters or Nuclei URL and selected template policy.
- Scanner execution: completed, failed, unavailable, incomplete or unknown.
- Semantic evidence: reported observation, inconclusive or not tested.

A successful process exit is execution completion, not a clean security verdict.
SQLMap request mode reports `reported` or `unknown`; absence of a
technique cannot establish `refuted`/`satisfied`. Nuclei findings are scanner
observations. An empty result does not establish that the operation is secure.
URL-only limitations must reach the Audit report even if scanner execution
succeeds. The importer accepts `not-tested`, `blocked` or `inconclusive` for scan
tasks. Completed execution requires a retained `scanner-report` evidence item
and a runnable accepted task; it still yields inconclusive security coverage.
Scanner observations can be retained without declaring a verified violation.
Conclusive assessments belong to a separate verification, outside this slice.
Preparation gaps are always merged into coverage and scan provenance into the
retained task origin. Trusted assembly must bind accepted item/manifest identity
and exact evidence; raw scanner JSON is not itself an Audit result package.

## Retry and completion requirements

The existing scan session fence is keyed by Stage execution identity. It protects
recovery of that execution, not a new Stage or Audit item attempt. The Audit
importer treats a failed Run as retryable
([19 §10](19-audits.md#10-audit-and-round-lifecycle)). The adapter must address
both layers.

Do not impose a blanket one-attempt limit or disable ordinary Workflow retries.
Failures known to precede scanner invocation and known failed executions may use
the existing bounded retry policy, with each execution attempt recorded and
charged to its budgets. A committed successful scan result is reused during
recovery; collection/read/publication retries do not rerun its scanner.

Distinguish a known failure from a lost result after possible scanner execution.
That unknown outcome must not silently become a fresh scan solely because
collection sees a failed Run: retain the attempted execution and its gap. This
is an outcome/recovery rule, not a prohibition on running the same scan again or
a requirement for a new human approval flow. Automatic handling of unknown
outcomes must be explicit in the pinned execution policy before activation.

Use the ordinary scan planner journal for scanner execution. Do not apply
`audit-check-results@1` to a multi-stage/tool-only Worker. Trusted result assembly
must retain reports for failures as well as successful execution, and must be
tested at the interruption between scanner completion and package publication.

## Required release evidence

`make test-openapi-audit-scan-e2e` requires all seven cases of
`TestOpenAPIAuditScanAcrossProductionProcesses` to execute and pass; missing or
skipped cases fail. Real Server/Scheduler/Runtime/public API tests with local
controlled SQLMap and Nuclei fixtures prove exact dispatch, accepted
Audit packages, distinct scan coverage, bounded retries and recovery without
implicit redispatch of completed or unknown executions across Run/Stage attempts.
The matrix includes missing path/body/auth values, excluded SQLMap parameters,
Nuclei POST-derived URLs, credential rejection, unknown outcomes, cancelled
execution, lost acknowledgements and retained evidence after source Run deletion.
Scripted tests do not establish model-generated OpenAPI quality or scanner recall.
The recorded gate result is [`tasks/evidence/v62-009.json`](../../tasks/evidence/v62-009.json).
