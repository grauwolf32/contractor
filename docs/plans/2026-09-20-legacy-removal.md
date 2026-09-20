# Legacy compatibility removal — analysis and plan

Status: C01–C07 and D01–D05 implemented, verified and integrated into local `main`;
D06–D07 remain planned.
C01–C03 were integrated into local `main` as `1d661196`, C04 as `bc32a76b`,
C05 as `71f77e00`, C06 as `6acb157d`; C07 followed on
`refactor/catalog-legacy-removal`.
Reviewed working tree on 2026-09-20, HEAD
`e1ea6209713d1c8d637d1106ce92a4960a4a2c58`, including existing uncommitted
documentation. The user requested finding code retained only for backward
compatibility and planning its removal. This document records the findings,
consumers, removal order and acceptance conditions.

The user subsequently clarified that this is a fresh project and backward
compatibility is not required. Remove obsolete paths directly, updating current
callers and contracts in the same increment. A deprecation window, compatibility
adapter or migration of obsolete formats is not a prerequisite. This decision
supersedes the initial review's historical-data inventory gates.

## Recommendation

Start with six groups of runtime/API implementation shims, then retire the
**41 explicitly superseded catalog entries**. Seven further groups support
historical persisted data; replace them with strict current-format handling
in separate increments.

Recovery, history, repeat and idempotency replay must continue to work for
current records. Old-only records may become unsupported; missing authority
must produce an explicit failure instead of invented current values. Test
fixtures should implement current internal interfaces instead of keeping a
second production execution path alive.

No live database, operator-managed catalog, installed client or external
deployment was inspected. Their usage is unknown. Searches covered Go, Python,
UI, static serving, configuration, deployment examples, tests and specifications.
The initial inventory was a static review, not proof of whole-repository
dead-code elimination. Subsequent implementation checks are recorded below.

## Candidates without a persisted-data migration

The C identifiers below are local plan items, not task-registry entries.
Each row should be a separate implementation change, with its consumer changes
and checks in the same change.

| ID | Candidate and evidence | Removal and remaining dependency | Risk |
| --- | --- | --- | --- |
| C01 | Python compatibility import and test-only session property | Remove the unused import module; update four test accesses before deleting the property. | Low inside the repository; external Python imports are unknown. |
| C02 | Deprecated server flags, environment variables and duplicate config fields | Update launcher/test inputs to canonical settings, then remove parsing and assertions. | Old launchers may stop working or lose their configured root. |
| C03 | Four families of old UI redirects | Retire old URLs together in the router and static server; update links and route contracts. | Old bookmarks stop redirecting. |
| C04 | Scheduler compatibility with older in-process allocators/stores | Require contextual reservations, complete placement results and durable creation times; update fakes. | Execution-sensitive refactor; cancellation, policy and provenance need regression checks. |
| C05 | Pre-ModelAccess Streamline/Router factory path | Make configured factories the single model-backed construction path; inject fake models through the same path. | Tests must retain temperature, budgets and model-selection coverage. |
| C06 | Per-stage HTTP read fallbacks for older interfaces | Require batch methods on the relevant read interfaces; update test readers. | Preserve absent optional data and authorization/error behavior. |

### C01 — Python import and session shims (implemented)

- Removed `runtime/src/contractor_runtime/app.py`, which only re-exported
  `server.create_app` and explicitly called itself a compatibility import.
  Repository callers, including [cli.py](../../runtime/src/contractor_runtime/cli.py),
  already import from `contractor_runtime.server`; no consumer of
  `contractor_runtime.app` was found.
- [worker/runtime.py](../../runtime/src/contractor_runtime/worker/runtime.py),
  no longer exposes `AdkWorkerRuntime._session_id`, the compatibility-only test
  view. Updated three accesses in
  [test_adk_runtime.py](../../runtime/tests/test_adk_runtime.py) and one in
  [test_agent_skill_toolset.py](../../runtime/tests/test_agent_skill_toolset.py).
  The tests now obtain the explicit shared session ID from the session lifecycle
  and assert its presence before inspecting the session. Shared/isolated session
  behavior and cleanup assertions remain covered.

### C02 — process configuration aliases (implemented)

Removed the following compatibility surface together:

- `--config-root` and `CONTRACTOR_CONFIG_ROOT`, superseded by
  `--operator-config-root` and `CONTRACTOR_OPERATOR_CONFIG_ROOT`;
- `Config.ConfigRoot`, a duplicate of `Config.OperatorConfigRoot`;
- `--public-user-id`, `CONTRACTOR_PUBLIC_USER_ID`, the corresponding input/config
  fields and the comparison with local-auth's user ID. Local-auth already owns
  the authenticated principal; the compatibility value is only an assertion.

Evidence: [config_flags.go](../../internal/app/config_flags.go),
[config_inputs.go](../../internal/app/config_inputs.go),
[config_validate.go](../../internal/app/config_validate.go) and
[app.go](../../internal/app/app.go). Updated
[specification 06](../spec/06-server-ui-and-operations.md) to the current surface.

Updated [Makefile](../../Makefile)'s `run-local`, numerous
`tests/e2e` launchers, `tests/eval/project_workflows`, `tests/ui-stack`, and
[local-stack.md](../guides/local-stack.md). Cross-owner tests still create/change
the actual local-auth identity and check isolation. Removed only their duplicate
environment setting and the unit test of the obsolete equality assertion.

Old flags now receive the normal unknown-flag error. Old environment variables
are no longer read; set `CONTRACTOR_OPERATOR_CONFIG_ROOT` or
`--operator-config-root` for a non-default catalog root. Local-auth is the sole
principal source. No replacement compatibility parser or warning path was added.

`--server-config` is another spelling for `--config`, but was not shown to be a
historical-only path; it is outside this confirmed deprecated set.

### C03 — UI URL redirects (implemented)

Removed four compatibility handlers:

- `LegacyCatalogRedirect` in [catalog/layout.tsx](../../ui/src/routes/catalog/layout.tsx):
  `/workflows`, `/workflows/:name/:version`, `/skills` to `/catalog/...`;
- `LegacyQueueRedirect` in [runs/index.tsx](../../ui/src/routes/runs/index.tsx):
  `/queue` to `/runs`;
- `LegacyRuntimeConfigurationRedirect` in
  [runs/configuration.tsx](../../ui/src/routes/runs/configuration.tsx):
  `/operations/runtime-configs...` to `/runs/configuration...`;
- `ProjectLegacySectionRedirect` in
  [projects/navigation.tsx](../../ui/src/routes/projects/navigation.tsx):
  five old project-root hashes to section paths.

Updated [router.tsx](../../ui/src/app/router.tsx),
[projects/detail.tsx](../../ui/src/routes/projects/detail.tsx),
[static-server.mjs](../../ui/server/static-server.mjs) and specifications 06/18.
Removed obsolete static route patterns as well as React handlers. Canonical
direct loads, encoded identities, query strings and current section navigation
remain supported. Retired path routes produce the existing 404 experience.
Old project hashes can remain on the valid project root but no longer select
sections; HTTP never receives those hashes.

Updated Skills links in Artifacts and Project artifact links in Run details
and the Workflow launch form to explicit current paths. Existing route and
browser tests now open canonical addresses; retired URLs are covered by the
existing React recovery and static-server rejection tests.

Preserved the hash anchors used by evaluation workspace pages and
`/evals/legacy`: these render content. `/catalog` still opens its current
Workflows index.

### C04 — Scheduler interfaces and fallback execution settings (implemented)

Scheduler's [Allocator](../../internal/scheduler/types.go) now requires
`ReserveAllContext`. Removed the optional interface/type assertion and the
context-free `PlacementAllocator.ReserveAll` wrapper. Production composition
already uses `PlacementAllocator`; the independent `InMemoryRegistry.ReserveAll`
operation remains in use by the Registry and its consumers.

Removed the following fallback behavior:

- [allocation_reservations.go](../../internal/scheduler/allocation_reservations.go)
  rejects missing resolved RuntimeConfig, identity/label revision and invalid
  collection policy before persisting allocations or preparing Workers. Policy
  is no longer replaced with `disabled`. Recovery requires complete matching
  durable provenance.
- [workflow.go](../../internal/scheduler/workflow.go) rejects missing persisted
  Stage `CreatedAt` as `scheduler_state_invalid`. The deadline always derives
  from the immutable timestamp; observing a Stage cannot start a new budget.
- [execution_preparation.go](../../internal/scheduler/execution_preparation.go)
  requires one complete reservation set and rejects duplicate/missing Workers
  or resolved configuration. Removed the test-only wrapper and
  `fallbackResolvedWorkerConfig`; Scheduler cannot manufacture provenance.

Updated in-memory allocators and stores to provide contextual reservations,
complete provenance and creation times, including manual Resume/retry fixtures.
Test placement configuration uses the real `runtimeconfig.ResolveRuntimeConfig`
resolver with explicit fixture inputs. Model-free `tool@1` Workers remain valid
with no LLM route. Updated specification 05. V61 toolset pinning is separate
work on overlapping allocation/configuration paths and is not merged here.

### C05 — model-backed Planner constructors (implemented)

Removed the direct-LLM `NewFactory`/`NewFactoryWithMemory` constructors from
[Streamline](../../internal/planner/streamline/factory.go) and
[Router](../../internal/planner/router/factory.go), their unconfigured Router
delegates and `Factory.model`. The configured factories are now the only
model-backed construction path. Each invocation requires valid ModelAccess
before constructing its model client; execution budgets come from ModelPolicy.

Removed the missing-access branches in
[streamline/planner.go](../../internal/planner/streamline/planner.go), including
the implicit zero temperature and placeholder telemetry model alias. An omitted
policy temperature stays omitted; an explicit zero remains explicit.

Unit fixtures now supply valid ModelAccess and inject scripted models through
InvocationModelFactory. Budget tests set the invocation's ModelPolicy;
configuration rejection and per-invocation model selection cover both profiles.
The PostgreSQL/Gateway recovery test and opt-in live Router harness construct
clients from the supplied ModelAccess. Their deterministic zero temperature is
an explicit fixture setting.

[composition_execution.go](../../internal/app/composition_execution.go) already
used configured factories. `passthrough@1` and `scan-plan@1` remain model-free.

### C06 — HTTP batch reader fallbacks (implemented)

[run_detail_batch.go](../../internal/httpapi/public/run_detail_batch.go) now
calls the batch methods required by the corresponding interfaces in
[public/types.go](../../internal/httpapi/public/types.go). Removed the three
optional-interface assertions and individual Stage read loops. Production uses:

- `PostgresStore.ListStageAllocationsBatch`;
- `telemetry.Repository.GetStageMetricsBatch`;
- `planner/session.Service.LoadPlans`.

Also removed the nested per-session fallback in
[planner/session.LoadPlans](../../internal/planner/session/plans_batch.go),
requiring `GetPlannerSessions` on the session store. Test readers now implement
the current batch contracts. Individual methods used by other consumers remain
available on the production stores and Planner session service.

Metrics and PlannerPlans dependencies remain optional. Authorized-Run reads,
best-effort metrics, exact Planner session identity checks and bounded SQL
query counts retain their existing behavior. The PostgreSQL regression test
asserts response contents and query bounds directly instead of comparing with
the retired per-stage path. Updated specification 06.

## C07 — retire superseded configuration from the default catalog (implemented)

Removed the 41 superseded definitions inventoried in
[memory-catalog.json](../../configs/memory-catalog.json): **20 AgentTemplates,
16 Workflows and 5 AuditProfiles**. All 41 successor YAML files retain their
previous bytes. The default catalog now contains 21 AgentTemplates, 17 Workflows,
6 AuditProfiles and 38 instruction files; the separate Audit completion example
remains supported.

Removed 14 instruction files with no remaining default-catalog consumer. Other
pre-Memory instruction files are still selected by current Planner stages and
remain required. The four ordinary examples now use current templates under
version 2. Router/Streamline examples use their existing Memory example
bytes at the canonical filenames; the two duplicate `_memory.yaml` copies are
removed.

Consumers and tests:

- Updated repository-catalog assertions, process Workflow/Audit selectors,
  shared Skill assignment checks and ordinary project/Audit evaluation harnesses.
  Exact tool allowlists now include their existing Memory operations. Domain
  capabilities, outputs, topology, completion policy and snapshot-copy assertions
  remain covered.
- The isolated `configs/e2e`, `testdata/configs` and minimal loader catalogs keep
  their own behavioral fixtures. `configtest` now includes its artifact-copy
  Workflow and instructions alongside the existing test-only artifact builder
  and policies. Its escalation fixture uses the current validator template.
- Frozen V40 `catalog-baseline.json`, variants, candidate definitions and release
  manifests remain unchanged; their tests load the retained catalog directly.
- [production_memory_test.go](../../internal/config/production_memory_test.go)
  verifies current closure, all six Memory operations, retired-selector rejection
  and reading complete pinned Workflow/AuditProfile snapshots without a catalog
  lookup. [catalog_cleanup_test.go](../../internal/config/catalog_cleanup_test.go)
  verifies the exact current Workflow set and instruction/template reachability.
- New requests for absent exact selectors fail resolution. Repeat retains the
  original identity and reports blocking `workflow_unavailable`; it never picks
  a successor automatically. Independently published managed/operator resources
  are outside this source-catalog deletion.

[configs/README.md](../../configs/README.md) and
[configs/MEMORY.md](../../configs/MEMORY.md) describe the current inventory.
`memory-catalog.json` schema version 2 keeps `retired`, `active` and `active_file`
as an inventory and retirement record, with no paths to deleted definitions.

## Historical-data readers: strict-format cleanup

These are confirmed compatibility branches. Under the user's fresh-project
decision, old shapes can become unsupported. Their removal remains separate
from C01–C07 because it touches execution/replay semantics, not because old
data needs a compatibility window.

| ID | Code and historical shape | Current behavior to preserve when removing it |
| --- | --- | --- |
| D01 (implemented) | [config/persisted.go](../../internal/config/persisted.go): removed the missing Stage `session` → `shared` compatibility rule. | Both snapshot decoders require an explicit valid mode; authored omission still becomes `isolated` before persistence. |
| D02 (implemented) | [config/persisted.go](../../internal/config/persisted.go) and [audit_profile.go](../../internal/config/audit_profile.go): removed old Workflow role inference and the legacy digest algorithm. | Require explicit current role kinds and the current digest, including embedded closures and optional Worker completion. Old snapshots fail without reinterpretation or rewriting. |
| D03 (implemented) | [evalstore/receipts.go](../../internal/evalstore/receipts.go): removed conversion from old `{id, revision, state}` to typed receipts. | Require the current operation-specific shape, reject legacy/mixed receipts and preserve typed replay identity without new effects or byte rewriting. |
| D04 (implemented) | [auditservice/resume.go](../../internal/auditservice/resume.go): removed reopening terminal `deadline_exhausted` Audits, item reopening and report-link archival. | Resume accepts only paused Audits, including current deadline pauses. Terminal Resume fails without changing retained state; expired-review renewal, time-limit choices, replay and holds remain supported. |
| D05 (implemented) | [runstore/allocation_store.go](../../internal/runstore/allocation_store.go), [telemetry/allocation_resources.go](../../internal/telemetry/allocation_resources.go), [telemetry/repository.go](../../internal/telemetry/repository.go), [auditstore/validation.go](../../internal/auditstore/validation.go) and [auditstore/read.go](../../internal/auditstore/read.go): removed readers for absent allocation provenance/policy and historical `provenanceIncomplete`. | Require complete current provenance; keep current disabled/unsupported/missing-report states and model-free Workers valid. Reject incomplete old records instead of fabricating policy or origin. |
| D06 | [public/run_repeat_handlers.go](../../internal/httpapi/public/run_repeat_handlers.go): reconstruct inputs from lineage when repeat-request authority is absent. | Keep Repeat from a valid retained request. Missing/corrupt authority must block Repeat; audit-managed Runs remain excluded. |
| D07 | [auditstore/report_review.go](../../internal/auditstore/report_review.go) and [validation.go](../../internal/auditstore/validation.go): accept old `text/plain` report summaries; current publisher writes `text/markdown`. | Require the current summary media type; retain report acceptance and exact artifact validation without relabeling old artifacts. |

D04 concerns only the historical terminal-closure path. Current time-limit
expiry pauses an Audit, and continuing that paused Audit is core functionality.
The removed compatibility state is no longer written or consumed by the
controller, importer, store or UI. Existing terminal records remain final.

### D01 — explicit persisted Stage session mode (implemented)

Removed `normalizePersistedStageSession` and the extra JSON shape pass used to
infer a missing field. The Workflow and Stage snapshot decoders now validate
the already decoded `WorkerSessionMode` directly. Missing, null, empty, unknown
and non-string values fail decoding without a repaired or partially usable
snapshot. Existing AuditProfile closure validation applies the same explicit
mode requirement to its embedded Workflows.

Current writers already resolve authored omission to `isolated` and persist
`ResolvedStage.Session` without `omitempty`. Explicit `shared` retains its
intentional allocation-local conversation behavior. No writer, current snapshot
format or Runtime session lifecycle needed changing.

Updated [specification 00](../spec/00-workflow-and-planner.md). The
[configuration tests](../../internal/config/worker_session_test.go) cover both
valid modes through Workflow/Stage round trips, malformed and missing modes,
and a missing mode inside an AuditProfile. The
[Scheduler regression](../../internal/scheduler/worker_session_test.go) verifies
that missing Workflow or Stage session authority fails before allocation or
Planner/Worker execution, including recovery of preparing/running Stages, and
that no snapshot is rewritten. Existing retry and escalation tests retain
explicit `shared` across fresh attempts.

### D02 — explicit persisted Audit Workflow roles (implemented)

Removed the extra JSON shape pass, missing-kind inference and alternate digest
algorithm from AuditProfile decoding. Every persisted Workflow binding now
requires `check`, `discovery` or `assessment`; the single digest implementation
always includes that kind, the embedded Workflow and any explicit
`workerCompletion` contract. Current authors already provide explicit role kinds
and current writers retain them. Their serialized format and digest are unchanged.

Updated [specification 19](../spec/19-audits.md). The
[snapshot tests](../../internal/config/persisted_audit_profile_test.go) cover
current role round trips, legacy and mixed snapshots, missing/null/empty/unknown/
non-string kinds, a changed valid kind and a frozen digest from the removed
algorithm. Invalid snapshots return no usable profile and retain original bytes.
The [completion test](../../internal/config/audit_completion_test.go) also rejects
removing an explicit completion contract without changing its digest. The
[Controller regression](../../internal/auditcontroller/controller_test.go) closes
dispatch for the retired shape in accepted and assessing rounds before any
child Run or execution intent, preserving the stored snapshot and round/items.

### D03 — typed Eval mutation receipts (implemented)

Removed `legacyReceipt` and its four conversion callbacks. Dataset, experiment,
command and submission readers now decode their own current fields directly,
rejecting unknown fields and trailing JSON as well as missing/invalid identity
and revision. A rejected read returns no partially usable typed receipt.
Current writers already marshal these four types; no writer, stored current
format, public response schema or database migration needed changing.

Updated the [persistence contract](../../internal/evalstore/README.md). The
[receipt tests](../../internal/evalstore/receipts_test.go) cover current reads and
replays, cross-type rejection, legacy/mixed bodies and malformed/missing typed
fields without changing response bytes. The
[PostgreSQL HTTP regression](../../internal/httpapi/public/eval_receipts_postgres_test.go)
seeds historical receipt bytes for all four operations and verifies that
repeated requests fail at the response boundary without changing receipts,
datasets, experiment revisions/clocks, commands or submission intents. Current
receipts still replay their exact original response after those rejected reads.
Transaction ordering, owner checks and request-digest conflict handling remain
unchanged; an unsupported stored response never triggers a replacement mutation.

### D04 — terminal Audit deadline continuation (implemented)

Removed terminal-state admission and its dependency revalidation, item/round
reopening, report-link archival and continuation event flag from Resume.
The controller no longer retries cancelled roles through a continuation counter
or specially cancels children for a historical deadline finalization. Current
deadline pauses continue to let running children finish and collect results.
The importer publishes ordinary `report.json` and `report.md` artifacts.

Removed `ContinuationCount` from the store model/read projections.
[Migration 63](../../internal/persistence/migrations/000063_remove_audit_terminal_continuation.sql)
drops its obsolete column; checksum-verified migration 55 is unchanged.
The [forward migration test](../../internal/persistence/postgres/audit_continuation_removal_test.go)
preserves all other fields of paused/completed/failed records and exact old report
links, including historical artifact names. No report or accepted result is
rewritten or purged.

The UI offers Continue only for paused Audits. Terminal deadline records retain
their reason and results without a misleading prompt to resume. Start and paused
Resume still support default/remaining time, a new limit and no limit. Updated
the Audit specification, user stories, guide and public OpenAPI description;
regenerated Go and TypeScript clients. The current request/response shapes stay
the same; terminal Resume now returns the existing precondition failure.

The [service regression](../../internal/auditservice/resume_postgres_test.go)
compares retained authority before/after rejected terminal requests and covers
current deadline pauses, fresh decisions for expired reviews and exact replay.
The [HTTP regression](../../internal/httpapi/public/audit_postgres_integration_test.go)
checks successful paused Resume/replay and HTTP 412 for old terminal closures.
The [UI tests](../../ui/src/routes/projects/audits/audits.test.tsx) keep the
deadline-pause Continue flow and reject Continue controls on completed/failed
records. Existing controller tests cover retaining in-flight work at the limit.

### D05 — allocation and Audit provenance (implemented)

[RunStore](../../internal/runstore/allocation_store.go) requires complete persisted
Runtime identity, positive Agent-label revision, the current configuration schema,
valid resolved configuration and an explicit performance policy on both ordinary
and batch reads. The current writer already requires these fields. Model-free
Workers retain the current configuration without LLM routing.

[Telemetry](../../internal/telemetry/allocation_resources.go) no longer maps absent
or unrecognized policy to `legacy`. History and per-Stage projections fail on
missing/invalid policy; report ingestion requires a current envelope policy before
writing or replaying. Disabled, unsupported, pending, missing/expired reports,
partial observations and malformed optional resource blocks keep their existing
semantics. Removed the obsolete policy/reason constants and public enum values,
and updated the UI response parser and examples.

[AuditStore](../../internal/auditstore/read.go) requires complete item origin and
Workflow provenance for bound executions. Removed `ProvenanceIncomplete` and the
read-only incomplete-origin exception. Strict decoding rejects historical and
mixed marker shapes, including a marker added to an otherwise complete record.
Current tombstones still retain provenance after deleting a child Run. Unbound
executions still have no Run provenance.

Updated OpenAPI required fields, regenerated Go/TypeScript clients and moved
current finding-history/UI fixtures to full origin and Workflow identity. The
[allocation regression](../../internal/telemetry/allocation_legacy_postgres_test.go)
checks individual/batch reads, resource history, owner isolation and report replay.
The [Audit regression](../../internal/auditservice/provenance_postgres_test.go)
checks current, historical, mixed and missing-field records, including deleted
Runs. Repeated rejected operations preserve exact stored state. Existing SQL
migrations remain unchanged; no historical provenance is inferred, repaired or
purged.

### Removal rules for persisted formats

For each D item, verify that current writers supply the required shape, remove
the old reader, update the owning contract and test both current-format behavior
and explicit rejection of missing authority. Do not add automatic repair,
dual-read support or a migration tool solely for obsolete data.

Current Run recovery, paused Audit Resume, report acceptance and idempotent
replay remain product behavior. Keep their positive and negative tests. Existing
SQL migrations are checksum-verified execution inputs; use forward schema changes
when needed. Removing a compatibility decoder does not itself require purging
records or rewriting immutable artifact bytes.

## Items that should remain outside this cleanup

| Apparent legacy surface | Why it is not proven compatibility-only |
| --- | --- |
| `audit-results@1` and [audit_results/v1.py](../../runtime/src/contractor_runtime/toolsets/audit_results/v1.py) | Five active Memory templates still select it: ASVS, both OpenAPI tracers, risk checker and source checker. `source-checklist@3`/`audit-source-check@4` demonstrate v2 completion, but do not replace all these families. Migrating them is a product/configuration change before Runtime support can be retired. |
| [audit_results/packages.py](../../runtime/src/contractor_runtime/toolsets/audit_results/packages.py), despite its “legacy” docstring | Shared package codecs also support current trusted completion through `encoding.py`. Deleting the whole module with v1 would break current behavior. |
| `/evals/legacy`, old evaluation workspace pages and Project kind `evaluation` | These render/access actual retained work. Current managed Evals setup also creates `evaluation` Projects. Direct/portable evaluation remains an explicit contract in specs 26/30; it is not replaced automatically by a native experiment. |
| SQLMap's URL input mode in [scan/tools.py](../../runtime/src/contractor_runtime/toolsets/scan/tools.py) | It remains an executable input mode alongside `request_ref` under the same `scan@1` surface. Removing it reduces that tool API; establish consumer/version retirement separately. |
| Ordinary Worker finalizer | Current completion policy and provider limitations, not a reader for an obsolete format. The existing V57-003 decision is archived; this cleanup does not reopen or silently implement it. |
| Omitted optional fields, stable canonical digests, ciphertext, ordinary `@1` refs and SQL migration history | These can be current contracts/invariants. Age or absence of a field does not prove an obsolete branch; do not bulk-delete compatibility tests or every `@1` implementation. |

Adjacent dead code: `InventoryCompatibility` in
[auditservice/compatibility.go](../../internal/auditservice/compatibility.go)
always returns `nil`, while `start.go` and `preview.go` still check its result.
This can be considered in a small separate cleanup after checking V62's
inventory work. It is not evidence that all Audit compatibility validation can
be removed. Likewise `batching_unsupported` is retained as a public wire enum;
shrinking that contract is separate from removing an internal no-op.

## Execution order and acceptance

C01/C02 are implemented in the first increment, C03 in the second, C04 in
the third, C05 in the fourth, C06 in the fifth and C07 in the sixth;
D01 is implemented in the seventh increment, D02 in the eighth, D03 in the
ninth, D04 in the tenth and D05 in the eleventh; D06–D07 remain planned.
This document uses local IDs and does not mark task-registry entries complete.

| Order | Work | Exit condition |
| --- | --- | --- |
| 1 | C01 | No compatibility import/property or old test accesses; current server/session tests pass. |
| 2 | C02 and C03, each as its own change | Canonical launchers/routes work, old consumers are accounted for, specification and retirement notes match actual behavior. |
| 3 | C04, C05, C06, each as its own change | Tests use current interfaces; no parallel old implementation path remains; execution, auth, cancellation and SQL bounds hold. Coordinate overlapping V61 work. |
| 4 | C07 | All 41 entries are removed from the default catalog after resolving consumers; retained fixtures and historical execution remain valid. |
| 5 | D01–D07, as individually reviewable changes | Current-format writes, reads and recovery pass; old formats fail explicitly. No historical-data inventory or compatibility transition is required. |

Relevant verification for the future implementation:

| Scope | Checks |
| --- | --- |
| C01 | `runtime/.venv/bin/pytest -W error runtime/tests/test_app.py runtime/tests/test_adk_runtime.py runtime/tests/test_agent_skill_toolset.py runtime/tests/test_session_lifecycle.py` |
| C02 | `go test ./internal/app ./internal/cli`; affected process/UI-stack launchers with a disposable PostgreSQL database; owner-isolation scenarios. |
| C03 | `make ui-typecheck ui-test ui-build`; canonical and retired direct route loads, encoded identities and project navigation. |
| C04–C05 | `go test ./internal/scheduler ./internal/controlplane ./internal/planner/...`; corresponding PostgreSQL placement/provenance and scheduler cancellation/deadline gates with `CONTRACTOR_TEST_DATABASE_URL` set. |
| C06 | `go test ./internal/httpapi/public ./internal/runstore ./internal/telemetry ./internal/planner/session`; database-backed Run detail batching/query-budget tests. |
| C07 | `make test-config`; catalog closure/Memory tests and affected process/evaluation fixture preparation. No paid model run is implied by fixture validation. |
| D changes | Targeted upgrade/history/recovery tests for the affected format; `make verify-wire-contracts test-wire-cross-language verify-public-api ui-generate-check` when a wire/schema boundary changes. Regenerate affected clients first. |

Database tests that skip without a test URL are not upgrade evidence. Keep
the existing negative tests for corrupt/mixed snapshots, identity mismatches,
missing authority and response-loss replay. Change old-success expectations
only for the explicitly retired boundary, rather than deleting the tests
wholesale. At integration, run the normal `make verify` and affected release
gates once; record the actual results and exclusions.

## Initial static review

- Inspected the cited production branches, composition roots, consumers,
  existing regression tests and owning specifications.
- Parsed `memory-catalog.json`, checked the 20/16/5 counts and existence of both
  files for every mapping, and searched remaining configuration YAML consumers.
- Confirmed that five active Memory Audit templates still use `audit-results@1`.
- Checked all 52 local document links at the review baseline.
- That initial review changed no application code and ran no application tests
  or migrations; no production-data absence or performance improvement was claimed.

## First increment — C01/C02 results

Removed the Python compatibility module and test-only session property, the two
deprecated flags and environment inputs, the duplicate Config fields and the
obsolete user-ID equality assertion. Updated the four Python test accesses,
Go configuration tests, process launchers, cross-owner test setup, Makefile,
local guide and owning specification. Existing authentication and ownership
checks continue to use the local-auth principal.

Verification completed on 2026-09-20:

- `go test ./internal/app ./internal/cli ./internal/auth ./internal/httpapi/public`
  passed. Optional PostgreSQL tests in these packages were not enabled in this run.
- `runtime/.venv/bin/pytest -W error runtime/tests/test_app.py runtime/tests/test_adk_runtime.py runtime/tests/test_agent_skill_toolset.py runtime/tests/test_session_lifecycle.py`
  passed: **102 tests**.
- With `CONTRACTOR_TEST_DATABASE_URL` pointing at a disposable PostgreSQL 17
  container, `go test -tags=e2e -count=1 -timeout=6m ./tests/e2e -run '^(TestLocalGoToPythonArtifactCopy|TestRunMetadataLabelsAcrossProcesses)$'`
  passed. This exercises actual Go/Python startup and Workflow execution, plus
  the foreign-owner metadata/history access boundary using local-auth alone.
- `go test -tags=e2e -run '^$' ./tests/e2e ./tests/ui-stack ./tests/eval/project_workflows`
  passed as a compile check only; browser and live-model evaluations were not run.
- Ruff lint/format checks for the changed Python files, Go formatting and
  `git diff --check` passed. Searches found no remaining retired inputs or
  Python compatibility accesses in implementation, executable fixtures or
  current guides/specifications.

Initial sandboxed test attempts could not open local sockets; Go and Python
checks were rerun successfully after full access was enabled. No production
database was used.

## Second increment — C03 results

Removed the four UI compatibility handlers and their static-server route
patterns. Updated current links, route/browser fixtures and the owning
specifications. Retired path URLs now return 404; old Project root fragments
no longer redirect. Current Catalog, Runs, Runtime Configuration and Project
section URLs remain the supported entry points.

Verification completed on 2026-09-20 using Node 24.20.0 and pnpm 11.24.0:

- `pnpm typecheck` passed for application and browser-test TypeScript.
- `pnpm test --run` passed: **496 tests across 70 files**.
- `pnpm test:server` passed: **11 tests**, including retired path rejection,
  canonical direct GET/HEAD loads, encoded identities and 404 recovery.
- `pnpm build --outDir /tmp/contractor-c03-dist` passed. Vite reported its
  chunk-size warning; the build completed successfully.
- Chromium scenarios in `catalog.spec.ts`, `runs-navigation.spec.ts`,
  `project-workspace.spec.ts` and `operations-forms.spec.ts` passed:
  **11 scenarios total** against the built static UI with mocked API responses.
  Coverage includes direct routes, search/filter state, exact Agent versions,
  Project sections and uploads, Workflow launch, Run results, Operations
  capability checks and mobile layouts.
- ESLint and Prettier checks for the changed UI files, document-link validation
  and `git diff --check` passed.

The first browser attempt exposed a stale Catalog fixture assumption about the
default Agent version and a test-server CSP/API-origin mismatch. The fixture
now explicitly selects its required version; the local static server was
configured for the mocked cross-origin API. All eight initially failing
scenarios passed on recheck; the three Operations scenarios passed initially.
The real Go/Python browser stack was not run in this increment.

## Third increment — C04 results

Implemented in the isolated `refactor/scheduler-legacy-removal` worktree based
on local `main` commit `1d661196`. Existing unrelated working-tree changes on
`main` were preserved.

Verification completed on 2026-09-20:

- With `CONTRACTOR_TEST_DATABASE_URL` pointing to a disposable PostgreSQL 17
  container, `go test -race -count=1 -timeout=8m ./internal/scheduler ./internal/controlplane ./internal/planner/...`
  passed. Database tests were enabled, including placement/provenance,
  model-free Workers, credential connection use, scheduler claims/cancellation,
  recovery, result publication and Planner sessions.
- Added regression coverage for missing Stage creation times, incomplete live
  and durable allocation provenance, missing/duplicate Worker reservations and
  propagation of cancellation through the required contextual interface.
- `go test -run '^$' ./...` and
  `go test -tags=e2e -run '^$' ./tests/e2e ./tests/ui-stack ./tests/eval/project_workflows`
  passed as compile checks. Full process/browser/live-model suites were not run
  for this increment.
- Go formatting and `git diff --check` passed. The initial unit run identified
  one manual-Resume fixture family without creation times; it was updated and
  the complete database-backed race run passed afterward.

## Fourth increment — C05 results

Removed the direct-LLM constructors and missing-ModelAccess branches from
Streamline/Router. Updated unit and integration fixtures to use the configured
factories with explicit per-invocation policies. Implemented in the isolated
`refactor/planner-legacy-removal` worktree based on `bc32a76b` and integrated
into local `main`, preserving unrelated working-tree changes.

Verification completed on 2026-09-20:

- With `CONTRACTOR_TEST_DATABASE_URL` pointing to a disposable PostgreSQL 17
  container, `go test -race -count=1 -timeout=8m ./internal/planner/... ./internal/scheduler ./internal/controlplane ./tests/integration/streamline`
  passed. Database tests were enabled, including Gateway/Worker recovery
  without semantic replay.
- Both Planner profiles reject missing or invalid ModelAccess before model,
  session or Worker side effects. Existing per-invocation model selection,
  model/token/Worker budgets and omitted-versus-explicit-zero temperature
  checks pass through the current factory path.
- `go test -run '^$' ./...` and
  `go test -tags=e2e -run '^$' ./tests/e2e ./tests/ui-stack ./tests/eval/project_workflows`
  passed as compile checks. Full process/browser and opt-in live-model suites
  were not run for this increment.
- Go formatting, local plan-link validation and `git diff --check` passed.
  Searches found no remaining retired constructors or missing-access
  compatibility branches in the affected implementation and tests.

## Fifth increment — C06 results

Removed three optional batch-interface checks and per-stage loops from the Run
detail handler, plus the nested compatibility loop in Planner session loading.
The HTTP readers and Planner session store now require batch methods. Updated
test readers, direct response assertions and specification 06. Implemented in
`refactor/http-batch-legacy-removal` and integrated into local `main`, preserving
unrelated working-tree changes.

Verification completed on 2026-09-20:

- With `CONTRACTOR_TEST_DATABASE_URL` pointing to a disposable PostgreSQL 17
  container, `go test -race -count=1 -timeout=8m ./internal/httpapi/public ./internal/runstore ./internal/telemetry ./internal/planner/session`
  passed. All four packages completed with database tests enabled.
- The PostgreSQL Run detail regression test verifies a constant query count
  for 1, 5 and 30 Stages, one query per related collection, exact plans and
  outputs, absent/corrupt optional metrics, and owner rejection before related
  reads. Planner session tests retain missing/mismatched identity rejection.
- HTTP regression tests cover absent optional readers, unavailable metrics,
  required allocation/plan read failures, sanitized errors and an empty Run
  that needs no related reads.
- `go test -run '^$' ./...` and
  `go test -tags=e2e -run '^$' ./tests/e2e ./tests/ui-stack ./tests/eval/project_workflows`
  passed as compile checks. Full process/browser/live-model suites were not run
  for this increment.
- Go formatting, local plan-link validation and `git diff --check` passed.
  The four retired optional batch interfaces and their compatibility branches
  have no remaining implementation references.

## Sixth increment — C07 results

Retired the superseded default-catalog definitions and their orphan instructions,
updated the ordinary examples, and moved all default-catalog consumers to current
selectors. Implemented in `refactor/catalog-legacy-removal` and integrated into
local `main`, preserving unrelated working-tree changes. The 41 successor YAML files and frozen V40 catalog inputs
retain their exact bytes.

Verification completed on 2026-09-20:

- `make test-config` passed, including command-line validation of all 21 templates,
  17 Workflows, 6 AuditProfiles and 38 instructions.
- PostgreSQL 17 race tests passed for configuration, Agent Skills, Audit
  controller, public HTTP API, Run service, scheduler, ordinary and frozen eval
  harnesses, untagged E2E matrix tests and fault matrix tests.
- Seven process scenarios for Memory, project Workflows, Audit programs, Findings,
  Agent Skills, HTTP/Caido and taint annotations passed against real Go/Python processes and scripted model gateways.
  The Audit scenario includes removal of its current staged catalog and a Server
  restart before checking retained results and provenance.
- The real Chromium Operations/Project/Streamline browser scenario passed with
  Node 24.20.0 and pnpm 11.24.0; optional screenshot OCR was unavailable. Full
  managed-Evals browser journeys and paid-model/real-Podman-container gates were
  not run in this increment.
- Four offline Python Podman Workflow tests passed with warnings as errors.
- Browser fixture closure tests load both ordinary and managed-Evals catalogs,
  including both Workflow/Audit arms. E2E, UI-stack and project-eval packages
  compile with the `e2e` build tag.
- Initial runs exposed stale test assumptions: removed selectors/files, exact
  tool lists without Memory, old policy budgets, an omitted summarizer instruction
  dependency, a fault-matrix source-file pointer, and a Findings comparison that
  included the changing read timestamp. The Agent Skills retry fixture now uses
  a transient Gateway response with HTTP retries disabled, so it exercises the
  intended Stage retry. Updated those fixtures while retaining
  domain, authorization, snapshot, review-state and tool-boundary assertions.
- Local documentation links, formatting and `git diff --check` passed.

## Seventh increment — D01 results

Implemented in `refactor/persisted-session-legacy-removal`, based on C07 commit
`97240aa0`, and integrated into local `main`. Unrelated working-tree edits were
preserved.

Verification completed on 2026-09-20:

- `go test -count=1 -timeout=8m ./...` passed with optional database/live-model
  environment variables unset: **63 packages with tests**. No unrelated fixtures
  needed updating to retain the removed historical interpretation.
- With `CONTRACTOR_TEST_DATABASE_URL` pointing to disposable PostgreSQL 17,
  `go test -race -count=1 -timeout=8m` passed for `./internal/config/...`,
  `./internal/scheduler`, `./internal/runservice`, `./internal/app`,
  `./internal/auditservice`, `./internal/auditcontroller`, `./internal/auditimport`,
  `./internal/findingintake` and `./internal/httpapi/public`.
- Focused configuration tests cover explicit shared/isolated round trips,
  missing/null/empty/unknown/non-string persisted modes, the AuditProfile closure
  and the current authored default. The Scheduler rejects missing authority in
  queued and recovering Workflows and preparing/running Stages before reservation,
  Worker preparation or Planner creation/invocation; original bytes are retained.
- `TestWorkerSessionModesAcrossProductionProcesses` passed with a disposable
  PostgreSQL 17 database, real Go/Python processes and a scripted model gateway.
  It covers Streamline isolated sessions, Router shared sessions in separate
  logical Worker allocations, later isolated Stages and Runtime reuse.
- Go formatting, specification/plan link validation and `git diff --check`
  passed. Searches found no remaining implementation reference to the removed
  normalizer. No paid model call, browser run or data migration was needed.

## Eighth increment — D02 results

Implemented in `refactor/audit-role-legacy-removal`, based on D01 commit
`a4643be9`, and integrated into local `main`. Unrelated working-tree edits were
preserved.

Verification completed on 2026-09-20:

- Compared all six default AuditProfile digests before and after removing the
  alternate algorithm: every current digest is identical, including the profile
  with explicit Worker completion.
- `go test -count=1 -timeout=8m ./...` passed with optional database/live-model
  environment variables unset: **63 packages with tests**.
- With `CONTRACTOR_TEST_DATABASE_URL` pointing to disposable PostgreSQL 17,
  `go test -race -count=1 -timeout=8m` passed for `./internal/config/...`,
  `./internal/auditservice`, `./internal/auditcontroller`, `./internal/auditimport`,
  `./internal/runservice` and `./internal/httpapi/public`. Existing draft/start,
  paused Resume, replay and completion tests continue to pass.
- `TestAuditProgramsAcrossProductionProcesses` passed against disposable
  PostgreSQL 17, real Go/Python processes and a scripted model gateway. It covers
  the current Audit programs and retained results/provenance after removing the
  staged catalog and restarting the Server.
- `go vet ./...`, changed-file Go formatting, local specification/plan links
  and `git diff --check` passed. Searches found no remaining implementation of
  the removed role inference or legacy digest helpers.
- Full `make verify`, browser journeys and paid-model tests were not run in
  this Go-only increment. No public wire schema, UI or Runtime code changed,
  and no obsolete-data migration was introduced.

## Ninth increment — D03 results

Implemented in `refactor/eval-receipt-legacy-removal`, based on D02 commit
`3e29ed16`, and integrated into local `main`. Unrelated working-tree edits were
preserved.

Verification completed on 2026-09-20:

- `go test -count=1 -timeout=8m ./...` passed with optional database/live-model
  environment variables unset: **63 packages with tests**.
- With `CONTRACTOR_TEST_DATABASE_URL` pointing to disposable PostgreSQL 17,
  `go test -race -count=1 -timeout=8m` passed for `./internal/evaldomain`,
  `./internal/evalstore`, `./internal/evalservice`, `./internal/evalcoordinator`,
  `./internal/persistence/postgres` and `./internal/httpapi/public`. Existing
  owner/CAS/digest-conflict, response-loss recovery and replay-after-purge
  coverage passed with the current receipt format.
- `TestEvalPostgresRejectsLegacyReceiptReplayWithoutNewEffects` passed for
  dataset, experiment, command and submission receipts, including both legacy
  and mixed shapes. Each rejected request is retried; exact database snapshots
  prove retained receipt bytes, revisions, clocks and mutation effects stay
  unchanged. Current receipts then replay their original HTTP responses.
- `go vet ./...`, changed-file Go formatting, documentation links and
  `git diff --check` passed. Searches found no remaining legacy receipt type or
  conversion implementation.
- Full `make verify`, browser/process journeys and paid-model tests were not run
  in this Go-only increment. Public schemas and clients did not need regeneration.

## Tenth increment — D04 results

Implemented in `refactor/audit-continuation-legacy-removal`, rebased onto
`6b1a3e53` to retain the new Audit preset catalog, ASVS and WSTG definitions.
Integrated into local `main`, preserving unrelated working-tree changes.

Verification completed on 2026-09-20:

- Full `make verify` passed: formatting/static checks, Go tests and builds,
  **2468 Python tests** (39 optional tests skipped), generated TypeScript client
  consistency, UI lint/typecheck, **506 UI tests**, **11 static-server tests**
  and the production UI build. `make verify-public-api` also passed.
- With `CONTRACTOR_TEST_DATABASE_URL` pointing to disposable PostgreSQL 17,
  `go test -race -count=1 -timeout=10m` passed for `./internal/auditstore`,
  `./internal/auditservice`, `./internal/auditcontroller`, `./internal/auditimport`,
  `./internal/httpapi/public`, `./internal/persistence/postgres` and
  `./internal/app`.
- `TestAuditProgramsAcrossProductionProcesses` passed against PostgreSQL 17,
  real Go/Python processes and a scripted model gateway. Retained results and
  provenance remain readable after removing the staged catalog and restarting
  the Server.
- Terminal deadline Resume is rejected for both completed and failed records,
  including repeated requests with omitted, positive and zero time limits.
  Database snapshots verify unchanged authority, items, coverage, reviews,
  reports, events and receipts. Current paused Resume, remaining-time behavior,
  expired-review renewal, unlimited mode and idempotent replay pass.
- Migration 63 preserves all other Audit fields and historical report references
  and remains idempotent. Updated the migration 62 test to exclude the column
  subsequently removed by migration 63 from its retained-row comparison.
- Integration with the new catalog exposed an outdated Skill assignment list.
  Added its new `audit_standard_source_verifier` template to the expected
  `trace` users without changing the catalog or weakening package checks.
- Local documentation links, Go formatting and `git diff --check` passed.
  The obsolete counter has no remaining production reader or writer; its only
  references are migration history, the forward removal and migration tests.
- Browser journeys and paid-model tests were not run in this increment.

## Eleventh increment — D05 results

Implemented in `refactor/allocation-provenance-legacy-removal`, initially based on
`14905d0e` and rebased onto `1a1b8afd` to retain the independent Evals refresh fix.
Integrated into local `main`, preserving unrelated working-tree changes.

Verification completed on 2026-09-20:

- Full `make verify` passed: formatting/static checks, Go tests and builds,
  **2468 Python tests** (39 optional tests skipped), generated client consistency,
  UI lint/typecheck, **510 UI tests**, **11 static-server tests** and the UI build.
  After rebasing onto the Evals fix, `make ui-verify` passed again with **515 UI
  tests**, generated-client validation, static-server tests and the build.
- `make verify-wire-contracts test-wire-cross-language verify-public-api` passed,
  including shared Go/Python contract fixtures and public OpenAPI conformance.
- With disposable PostgreSQL 17, `go test -race -count=1` passed for
  `./internal/runstore`, `./internal/telemetry`, `./internal/controlplane`,
  `./internal/scheduler`, `./internal/auditstore`, `./internal/auditservice`,
  `./internal/auditcontroller`, `./internal/auditimport`, `./internal/httpapi/public`,
  `./internal/app` and `./internal/persistence/postgres`. This includes durable
  placement of model-free `tool@1` Workers and current Scheduler recovery.
- PostgreSQL regressions reject allocations without Runtime provenance or
  collection policy through individual and batch reads. Resource history and
  per-Stage views reject missing policy while retaining owner isolation.
  Empty/legacy/unknown report policies fail before replay or writes; current
  report replay preserves the exact retained row.
- Audit PostgreSQL regressions reject historical, mixed and missing provenance,
  preserve the stored bytes across repeated reads and accept complete current
  origin and deleted-Run Workflow tombstones.
- `TestAuditProgramsAcrossProductionProcesses` and
  `TestLabelDrivenRuntimeConfigurationAcrossProcesses` passed against real
  Go/Python processes, PostgreSQL 17 and scripted model gateways. Audit results
  and provenance survive removal of the staged catalog and a Server restart;
  label-driven configuration still reaches the intended Runtime allocations.
- Updated the RunStore report fixture to carry its allocation's explicit policy,
  and the public batch-query fixture to use the current allocation writer.
  The latter still checks constant query counts for 1, 5 and 30 Stages, exact
  plans/outputs, optional metrics, owner isolation and physical-identity redaction;
  it now also asserts that committed allocation provenance reaches the response.
- Documentation links, formatting and `git diff --check` passed. No historical
  data rewrite or new SQL migration was introduced. Browser journeys and paid
  model calls were not run in this increment.

Next increment: D06, remove Repeat input reconstruction when retained request
authority is absent.
