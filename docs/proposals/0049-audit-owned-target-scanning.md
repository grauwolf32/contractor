# Target scanning through Audit

Date: 2026-10-31. Reviewed base before deletion: a6a90e292 and its working tree. Scope: direct-target Nuclei checks, standalone local and durable Project sweeps, their callers and the existing Audit replacement.

Status: the user decided on 2026-10-31 that Sweep should not exist as a separate feature. A target scan belongs to an Audit and executes through a selected Workflow. The removal plan below implements that direction; it does not authorize running a migration or changing demo data during this design review.

## Outcome

The owner opens a Project, starts a Web scan Audit for one explicit target, selects installed or custom rules, and follows the ordinary Audit progress, evidence, coverage and results. The Audit selects the existing scan Workflow. An ordinary Run and its Allocation perform execution.

There is no independent Scan or Sweep resource, controller, membership store, create/cancel/purge API, CLI manifest or portfolio coordinator. Installed rules and custom rules are choices within the same check.

This decision replaces the earlier research recommendation to retain the Server sweep and merge the local CLI implementation into it. That recommendation retained an unnecessary product abstraction.

## Why the second orchestration path survived

[V406-004](/Users/ruslan/src/contractor/tasks/done/v406-004-durable-scope-sweeps.yml) introduced a durable owner-bound batch across Projects. It deliberately added two tables, claims, admission/retry/cancellation, retained inputs and six public API operations. The reviewed `internal/httpapi/public/routes.go:25` exposed those operations.

[V406-018](/Users/ruslan/src/contractor/tasks/done/v406-018-project-scan-audits.yml) later moved the user-facing Web scan and custom-rule flow into Project Audits. Its scope explicitly excluded deleting the independent cross-Project CLI and API. The UI migration therefore left the old orchestration running beside the replacement.

The current handwritten UI has no sweep launch flow. The shipped [web-scan profile](/Users/ruslan/src/contractor/configs/audit-profiles/web_scan.yaml:1) already binds [audit-web-nuclei-scan](/Users/ruslan/src/contractor/configs/workflows/audit_web_nuclei_scan.yaml:1), with one target, one inventory item, one scan job per attempt and ordinary Audit retention. [ASC-R151 through ASC-R153](/Users/ruslan/src/contractor/docs/spec/audits/scans.md:857) describe that path.

The reviewed technical split was wider than the endpoints: `configureSweeps` in `internal/app/composition_scans.go:30` created a separate controller and service. `PublicCreateParams` and `normalizePublicCreate` carried sweep-specific target, RuntimeConfig and retained-input overrides into generic Run creation. These historical paths refer to the reviewed base above.

The earlier feature had production callers and tests. The issue is the product boundary those callers implement; keeping their tests green does not justify retaining the boundary after the user rejected it.

## Responsibility model

    Project
      -> Audit with exact target, settings and rule selection
        -> AuditProfile selects a check Workflow
          -> ordinary Run
            -> Stage and Allocation
              -> scan_nuclei tool
            <- scanner report and evidence Artifacts
        <- Audit collection, coverage, findings and report

| Concern | Owner |
| --- | --- |
| User-visible check lifecycle | Existing Audit |
| Target and selected settings | Exact baseline input Artifacts of that Audit |
| Workflow, template and installed-rule identity | Existing pinned Audit configuration |
| Custom rules | Optional exact Project rulepack input |
| Start, pause, cancellation and review policy | Existing Audit services and controller |
| Actual execution, placement and retries | Existing Run, Scheduler and Allocation |
| Scanner transport and local policy | Existing Runtime scanner and HTTP authority |
| Evidence and payload retention | Ordinary Artifact and Audit mechanisms |
| Result interpretation | Existing Audit scan collection and domain rules |

The target is fixed for the started Audit. Project target metadata may prefill a draft, but later Project edits must not retarget an accepted check. Rules and settings are pinned with the baseline. Target credentials remain subject to existing explicit selection and origin/scope restrictions; a rulepack cannot grant authority.

The application check is an Audit. A Workflow remains the reusable execution definition. Generic Run execution and deterministic scanner tooling stay available as platform mechanisms; they do not justify a second product-facing target-scan lifecycle.

## User and API flow

Reuse the existing Web scan form, Audit draft/start APIs and common check detail. Original selection and retries use the existing create/start keys. Results show the state reported by the Audit and child Run. A successful scanner exit is not a clean assessment; missing or ambiguous results retain their existing gaps and inconclusive outcomes.

A custom YAML rule or rulepack uses the same profile and check Workflow. Remove the separate Rule sweep concept from current product descriptions wherever it still suggests another resource or lifecycle.

Delete all six scan-sweep API operations:

- POST and GET /v1/scan-sweeps.
- GET and DELETE /v1/scan-sweeps/{sweepId}.
- GET /v1/scan-sweeps/{sweepId}/members.
- POST /v1/scan-sweeps/{sweepId}/cancel.

Remove their OpenAPI schemas, operation registration, generated client methods and operation-coverage expectations together. The obsolete routes disappear; no aliases, compatibility handlers or replacement scanner-specific endpoints are added.

Delete both local and remote scan-sweep CLI commands, manifest persistence and platform file locking. Current CLI audit support is findings export/download, not full Audit authoring. General Audit setup/start CLI work is already scoped in V411-017; deleting Sweep does not require inventing a scan-specific command or absorbing that later task.

There is no replacement bulk launch. If a later user need requires operating across Projects, review it as a separate product request against ordinary Audits, with measured need before adding another coordinator.

## What to remove and what to preserve

| Area | Removal or change | Preserved boundary |
| --- | --- | --- |
| Product orchestration | scanservice, scanstore, scancontroller, startup composition and maintenance loops | Audit controller and generic Scheduler |
| Persistence | scan_sweeps and scan_sweep_members through a new migration | Existing Projects, child Runs and their history |
| API | Six operations and exclusively owned DTOs/schemas | Ordinary Audit, Run and Artifact operations |
| CLI | Local manifest and remote sweep paths | Generic Run tooling and existing Audit export |
| Artifact support | Exclusively sweep-owned retain/purge helpers and snapshots | Shared immutable blobs, input forks and source Artifacts |
| Receipts and claims | ScanSweepReceiptKey and exclusively sweep-owned usage/rows | KeyedMutation, ControllerClaims and claimpass |
| Run creation | CreatePublicIn, CapturedTargetURL, PinnedRuntimeConfig and OriginalInputSources where sole production callers disappear | PinnedWorkflow and digest checks also used by Evals |
| Primary shipped catalog | Standalone nuclei-web Workflow and exclusively owned template/instructions/presentation | web-scan AuditProfile, audit-web-nuclei-scan and shared scanner adapters |
| UI | Obsolete sweep/standalone launcher references and generated types | Existing installed/custom Web scan Audit form and reports |
| Tests and gates | Sweep-specific fixtures and selectors; retarget scanner process journeys to Audit | Real process, retry, target policy, custom rules and report verification |
| Metadata | Ownership registry, persistence owner table, ERD, architecture and dependency maps | Current registries for remaining entities |

Perform a caller inventory before deleting each shared helper. PinnedWorkflow has a concrete Eval caller and remains. The generic Nuclei argument validation used outside sweep overrides is not automatically removed with the sweep-specific branch.

The optional configs/scan directory is explicitly a model-free tool fixture catalog. Its ordinary workflows and other scanner fixtures are not themselves the Sweep product. Preserve technical tool coverage where it has a real caller; stop documenting or using the primary standalone nuclei-web catalog entry as the application check journey. This removal does not introduce a scanner-name denylist into Scheduler or a blanket ban on arbitrary user-authored Workflows.

## Existing data and cutover

Use the existing single-active-Server upgrade model. After the old Server stops, the new revision has no sweep controller and cannot resume pending members. Previously admitted child Runs remain ordinary Runs with their existing states, inputs, reports and cancellation controls.

The preserved guarantee is readable Run history, exact inputs/reports and ordinary cancellation. The removed standalone Workflow is not kept executable or repeatable across a Runtime upgrade; the repository explicitly rejects backward compatibility. Complete needed standalone work before a later upgrade rather than rewrite historical snapshots or retain a compatibility adapter.

Do not invent Audits around historical Runs: that would fabricate a baseline, coverage or acceptance history. Do not restart or replay scans as part of migration. Retire exclusively sweep-owned auxiliary metadata and retained bindings through a migration and existing Artifact ownership rules, while preserving child inputs and shared blobs. The migration must be tested against populated active and terminal sweeps.

No compatibility reader, abandoned sweep maintenance controller or indefinite feature-specific retention path remains after removal. Apply the repository's backup and migration requirements before any later demo upgrade; implementation verification uses disposable databases.

## Verification against the user outcome

The first demonstration is the existing real UI/API/Server/Runtime path with a deterministic scanner fixture:

1. Create an ordinary Project and start its Web scan Audit against a local target.
2. Prove that the selected scan Workflow creates an ordinary child Run and Allocation.
3. Observe the scanner fixture, exact target/rule selection, retained report and Audit result.
4. Repeat with one custom rulepack, with no installed-rule fallback.
5. Exercise lost create/start responses, cancellation and source/Project edits; accepted inputs remain exact and no duplicate check is launched.
6. Verify that the removed API/CLI surfaces are absent and that startup runs no sweep work.
7. Upgrade a populated database and prove existing child Runs and input/report Artifacts survive without new target traffic.

Retarget the existing Nuclei process gate and its fixtures rather than remove scanner coverage with the old feature. Update every Make/CI selector when a test is renamed; a gate matching zero tests fails. A deterministic fixture proves composition and lifecycle, not live scanner quality.

## Replanning

Implementation belongs to V425-033. The previous sweep consolidation proposal is withdrawn. The web-nuclei milestone outcome must describe Audit-owned target scanning, and its current completion boundary must include removing the obsolete product.

V425-005 should no longer repair sweep ID bounds or cosmetically relocate sweep ownership entries. V425-033 removes that schema and metadata; V425-005 retains all other stored-size defects. V411-017 continues to extend ordinary Audit API/CLI consumers only and must not restore standalone scanner endpoints. Existing completed V406 records remain historical evidence of the superseded design.

## Delivery — 2026-10-31

Implemented by V425-033 in commit 5392479ea, on `codex/remove-sweeps-audit-scans`.
The real Audit demonstration also exposed the existing 16 MiB input-read budget
being rejected by the 4 MiB plan/output ceiling; V425-034 fixed that prerequisite
separately. The obsolete standalone template-manifest result advertisement was
removed; exact internal manifest provenance remains in the ordinary report.
Production sources decreased by 2,622 lines including the migration. Required
checks passed, with the full Runtime suite verified on a supported Linux host.
