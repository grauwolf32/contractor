# Open questions

Unresolved design questions, and contradictions and gaps between the legacy
specifications found during the 2026-10 migration to the domain layout.
Resolve each one in the owning specification and delete it here.

IDs such as `19#69` name paragraphs (blocks) of the deleted legacy documents:
`19` is `19-audits.md`, `oas` is `openapi-audit-scans.md`, `uis` is
`ui-user-stories.md` and `readme` is the old `README.md`. They are only for
traceability: [spec-migration.md](spec-migration.md) maps each legacy document
to its new owners, and the `<!-- from: … -->` provenance comments in the git
history of each domain document locate a block in the new text.

## From `05-first-slice-and-open-decisions.md`

### 05 — First slice and open decisions › Deliberately deferred

The following cross-cutting decisions remain open; no legacy document defines
them implicitly. Decisions deferred inside one document's scope live in that
document's own deferred section:
[07](../spec/configuration/runtime-config.md),
[09](../spec/tools/skills.md),
[10](../spec/tools/workspace.md),
[11](../spec/tools/http-caido.md),
[14](../spec/execution/worker.md),
[15](../spec/execution/worker.md),
[19](spec-history.md),
[20](../spec/operations/operations.md)
and [25](../spec/audits/worker-completion.md).

- remaining Workflow YAML details outside the transition, Agent-binding,
  artifact-slot and output-mapping contracts;
- Planner authority for generic artifacts and domain Toolsets other than the
  explicitly specified MemoryTools mirror;
- optional incremental Worker metric delivery for retaining detail across a
  hard process crash;
- S3 blob backend (deferral recorded in [23](../spec/artifacts/blob-storage.md)),
  streaming uploads, and longer-term Artifact retention;
- explicit Run-output publication endpoint;
- shared Runtime Agent Registry and coordination for multiple active Control
  Plane replicas;
- Contractor credential-encryption master-key rotation/re-encryption and
  external Vault/KMS adapters;
- LiteLLM virtual-key TTL/automatic rotation and policy fields beyond the
  explicit first-slice model, spend/reset, TPM, RPM and concurrency allowlist;
- concrete CA bootstrap, certificate delivery, lifetime, rotation and
  revocation procedures;
- future OIDC authentication, multiple users, user/Operations RBAC and
  multi-tenant authorization and quota policy;
- Web UI component library/design system and whether the initial editor
  covers Workflow and AgentTemplate or only ModelPolicy, LLMGatewayConfig and
  credentials;
- artifact preview size limits, whether OpenAPI/LikeC4 receive specialized
  renderers, and the list retention window.

## Contradictions and gaps

Each entry names its kind, the question, the legacy sources and what the
specification says now.

### [api/public-api.md](../spec/api/public-api.md)

- **Contradiction.** Does the WebSocket upgrade check `Origin` against one
  configured UI origin or against the browser-origin allowlist? Sources:
  06#122, 06#124. API-R9 gives Server an exact allowlist (the repeatable
  `--browser-origin` setting, so possibly several origins); API-R33 says the
  upgrade rejects every `Origin` other than "the exact configured UI origin",
  in the singular. `internal/httpapi/public/events/hub.go` checks the upgrade
  against the allowlist, so API-R33 may need aligning.
- **Duplicate.** RUN-R87 in [execution/runs.md](../spec/execution/runs.md)
  restates API-R48 (a Run detail response includes the event cursor); reduce
  RUN-R87 to a link. Source: 06#128.
- **Duplicate.** SEC-R45 in
  [platform/security.md](../spec/platform/security.md) still says ownership
  checks keep the non-disclosing `404`, which API-R25 owns and registers;
  SEC-R45 should keep only `401` and `403` and link API-R25. Source: 06#146.
- **Tension.** 16#6 handled the incompatible split of `labels` into
  `runtimeLabels` plus the metadata map by upgrading Server, UI and bundled
  clients together, while 06#31 requires a new public API version for
  incompatible changes and 06#26 says only additive changes avoid a coordinated
  release. Sources: 16#6, 06#26, 06#31. Each statement is kept with its owner
  (RUN-R8, API-R3, UI-R7) and none is reconciled.

### [artifacts/projects.md](../spec/artifacts/projects.md)

- **Gap.** Which publication result does the Run that loses a publication race
  record? Sources: 17#34, 17#35, 17#50. PRJ-R54 says the loser "records a
  publication conflict"; PRJ-R56 limits results to `published`,
  `already_present` or `failed`, and PRJ-R58 maps an existing destination to
  `already_present`. The mapping is not stated.
- **Contradiction.** Is the diff namespace `diff` or `diffs`? Sources: 17#14,
  17#16. PRJ-R13 lists `diff` among the names Server does not enumerate; the
  upload-shortcut table uses `diffs`.

### [artifacts/retention-and-deletion.md](../spec/artifacts/retention-and-deletion.md)

- **Gap.** Where does the Audit cancellation and drain phase fall among the
  Project deletion phases? Sources: 18#27, 19#179, 30#69. DEL-R44 keeps the
  five phases of 18#27, which omit the Audit phase (DEL-R45: before the Run and
  ProjectScope purge) and the eval fence (DEL-R41: before draining); the
  flowchart orders the fence only before the drain steps, and the Audit phase's
  position relative to phases 1–2 is not specified.

### [audits/audits.md](../spec/audits/audits.md)

- **Contradiction.** AUX-R56 in
  [audits/execution.md](../spec/audits/execution.md) says "Start always creates
  the deterministic accepted first inventory" (19#130), but a prepare profile
  enters phase `preparing` with no `currentRoundId`, and its first inventory
  and Round are accepted only after all prepare roles are accepted (19#113
  here, 19#117 in scans.md). Sources: 19#130, 19#113, 19#117. AUX-R56 should be
  limited to direct-input profiles.
- **Gap.** How does deletion proceed from `waiting_review`? Sources: 19#103,
  19#107. The Audit state diagram has `deleting` transitions only from draft,
  completed, cancelled, failed and from cancelling with retained delete intent,
  while AUD-R222 keeps cancel and delete available during the report decision;
  19#179 in review-and-reports implies the path goes through `cancelling`.
- **Duplicate with different wording.** AUD-R17 restates both hold lifetimes
  that [audits/review-and-reports.md](../spec/audits/review-and-reports.md)
  owns. Sources: 19#13, 19#174. For dispatch holds AUD-R17 says they "end after
  dispatch is durably closed and all submission intents are resolved", while
  REV-R187 (LIFE-AUD-1) says they "may be released once dispatch is durably
  closed and no unresolved child submission intent can create another Run"; for
  evidence and artifact holds AUD-R17 says they "remain until Audit deletion",
  while 19#174 adds "or an explicit later retention policy transfers and
  releases them". Reconcile AUD-R17 with review-and-reports or reduce it to a
  link.
- **Duplicate.** ART-R25 in
  [artifacts/artifact-store.md](../spec/artifacts/artifact-store.md) still
  restates the access rule of AUD-R218 (generic Artifact APIs cannot write
  `audit-standards` or read or list raw package bytes; only the trusted catalog
  seed path creates a binding); it should keep only the namespace reservation
  and link AUD-R218. Source: 19#34.
- **Legacy compatibility.** AUD-R170 keeps terminal Audits final "including
  historical records whose stop reason is `deadline_exhausted`", which implies
  an earlier behaviour where the time limit ended the Audit; AUD-R164 now
  pauses admission with that stop reason. Sources: 19#104, 19#105, 19#118.
  Decide whether to remove the historical clause.
- **Inconsistency.** The Purpose says an AuditProfile selects Workflows "for
  check, discovery or assessment roles" (19#4), but AUD-R27 and AUD-R106 define
  four role kinds, including `prepare` (19#26, 19#59). Sources: 19#4, 19#26,
  19#59.
- **Legacy compatibility.** AUD-R91 keeps `batching_unsupported` and
  `preparation_unsupported` as reserved wire values "for compatible clients",
  and AUD-R118 keeps older retained task packages without `profile_mode`
  decodable as non-risk. Sources: 19#41, 19#62. Both were flagged for removal
  under the no-backward-compatibility rule; decide whether to remove them.

### [audits/execution.md](../spec/audits/execution.md)

- **Duplicate.** AUD-R131 in [audits/audits.md](../spec/audits/audits.md)
  restates the `check-results` package layout of AUX-R94 and owns its limit
  LIM-AUD-21; reduce AUD-R131 to a link and move the limit to an AUX entry.
  Source: 19#78.
- **Duplicate.** AUD-R130 in [audits/audits.md](../spec/audits/audits.md)
  restates the execution manifest content of AUX-R93 (ordered item keys and
  exact task and input refs); reduce it to a link. Source: 19#76.
- **Duplicate.** PRJ-R62 in
  [artifacts/projects.md](../spec/artifacts/projects.md) restates AUX-R22
  (trusted `publication_mode=audit-managed` suppresses the `outputs/<slot>`
  publication; frozen outputs remain ordinary accepted outputs); reduce PRJ-R62
  to a link and keep PRJ-R63. Source: 19#170.
- **Wrong cross-reference.** DEL-R57 in
  [artifacts/retention-and-deletion.md](../spec/artifacts/retention-and-deletion.md)
  says audits/execution.md specifies when the collector records each
  disposition, but the `collection-contract-invalid` conditions are REV-R154
  (FAIL-REV-9) in
  [audits/review-and-reports.md](../spec/audits/review-and-reports.md); repoint
  DEL-R57 to REV-R154. Sources: 19#171, 19#110.

### [audits/scans.md](../spec/audits/scans.md)

- **Gap.** When does the executor check run for an inventory that a prepare
  profile generates? Sources: oas#8, 19#49. ASC-R79 makes input preview and
  Audit start check actual task requirements against the declared executor
  after construction, but ASC-R8 says preview validates only the original
  inputs and generated documents are parsed only after accepted preparation
  outputs exist, so neither preview nor start can check a generated inventory.
- **Contradiction.** The scan evidence vocabularies do not map onto each other.
  Sources: oas#26, oas#27. ASC-R127 lists semantic evidence as "reported
  observation, inconclusive or not tested", ASC-R129 has SQLMap request mode
  report `reported` or `unknown`, and ASC-R132 has the importer accept
  `not-tested`, `blocked` or `inconclusive`; `blocked` appears only in the
  importer list.

### [audits/worker-completion.md](../spec/audits/worker-completion.md)

- **Wrong value.** The activation example binds `ref: audit-source-check@2`,
  while AWC-R121 says the bundled checklist profile is `source-checklist@1` →
  `audit-source-check@1` → `audit_source_checker@1`. Sources: 25#11, 25#52.
  Reconcile the Workflow version.
- **Duplicate.** CTL-R80 in
  [execution/control-plane.md](../spec/execution/control-plane.md) restates
  AWC-R24 to AWC-R26 (placement requires the advertised completion contract and
  Toolset capability, fails closed and never downgrades silently); reduce
  CTL-R80 to a link. Sources: 02#6, 25#20.
- **Duplicate.** [execution/worker.md](../spec/execution/worker.md) WRK-R64
  states the ordinary-strategy row of this document's completion table (25#7),
  and WRK-R53 repeats the last sentence of AWC-R104 (25#47). Keep one owner for
  each and link from the other. Sources: 25#7, 25#47.

### [configuration/catalog.md](../spec/configuration/catalog.md)

- **Duplicate.** RUN-R2 in [execution/runs.md](../spec/execution/runs.md)
  restates CAT-R28 (WorkflowCatalog resolves the selector to the exact Workflow
  document), and the first sentence of RUN-R24 restates CAT-R34 (the durable
  Run snapshot is the execution authority); both link the catalog but remain
  separate requirements. Sources: 00#9, 00#47, 00#80.
- **Gap.** Two catalog errors have no code. Sources: 06#11, 06#72. Reusing a
  catalog cursor with another query is only "a bounded 400" (CAT-R68), and a
  managed publication that reuses an identity with different content only
  "conflicts", with no HTTP status (CAT-R59);
  [platform/errors.md](../spec/platform/errors.md) lists them as
  `(unnamed) catalog cursor mismatch` and
  `(unnamed) configuration identity conflict`.

### [configuration/runtime-config.md](../spec/configuration/runtime-config.md)

- **Contradiction.** RCF-R51 says the completed credential delete operation is
  the audit tombstone "containing only the ID, actor and deletion time", but
  the `credential_operations` schema has no actor field and also holds
  `idempotency_key`, `request_hash`, `operation_kind` and `phase`. Sources:
  06#81, 06#83.
- **Gap.** RCF-R128 forbids deleting an immutable RuntimeConfig version while
  it is referenced, but the public API has no delete operation for
  RuntimeConfig versions and version garbage collection is deferred
  (LIFE-RCF-3: "no delete endpoint"). Sources: 07#23, 07#127, 07#132. Decide
  whether the blocking rule should stay without a deletion path.
- **Contradiction.** Can an ADK Worker call an unauthenticated Gateway?
  Sources: 29#12, 01#35. SCN-R21 in
  [tools/scanners.md](../spec/tools/scanners.md) says AllocationSpec requires
  `llmGatewayUrl` and token for ADK, while RCF-R14 says omitting the credential
  means an unauthenticated local Gateway request; RCF-R160 only says
  RuntimeSettings permits their absence.
- **Gap.** Is Worker telemetry export retry timing sent to the Runtime?
  Sources: 07#96, 07#37, 07#38. 07#37 says Server pins the export values and
  passes them through RuntimeSettings and 07#38 defines
  `worker.telemetry.export.retry`, but the RuntimeSettings wire sketch's
  `telemetry.export` lists only `batchSizeBytes`, `maxAttempts`,
  `maxPendingSpans` and `maxPendingBytes`.

### [configuration/workflows.md](../spec/configuration/workflows.md)

- **Contradiction.** Does `primary` affect publication? Sources: 00#104, 17#30,
  17#36. WFL-R67 calls it "presentation and publication intent" and WFL-R68
  says it identifies canonical results "for Project Workflow recommendations
  and create-only publication", but PRJ-R102 says `primary` does not select
  which outputs are published, because automatic publication covers every
  declared output. Decide whether to drop the publication part of WFL-R67 and
  WFL-R68.
- **Duplicate.** Rules whose blocks this document owns are restated elsewhere:
  SCH-R11 restates WFL-R42 and WFL-R47 (retry and escalation create a new
  StageExecution), SCH-R76 restates WFL-R39 (cancellation is not a graph edge)
  and ART-R89 restates WFL-R72 (no implicit output mapping). Sources: 00#37,
  00#59, 00#101. Reduce the copies to links.

### [evals/managed-evals.md](../spec/evals/managed-evals.md)

- **Gap.** Does `ExperimentReceipt` include `portableExperimentId`,
  `expectedMembers` and `controlMode`? Sources: 30#39, 30#44. EVM-R147 says the
  receipt carries the experiment ID, the accepted revision and the state; the
  Create example returns `experimentId`, `portableExperimentId`,
  `state: draft`, `revision: 1`, `expectedMembers: 8`, `controlMode: server`
  and the ETag.
- **Gap.** Which error code does an external producer receive after the Project
  deletion fence? Sources: 30#69, 30#70. DEL-R43 gives `409` and says the code
  is specified in managed-evals.md, but managed-evals.md only lists
  `eval_project_deleting` (EVM-R227), and the errors registry marks its status
  and that mapping "not specified". The code returns `eval_project_deleting`
  with `409` (`internal/evaldomain/errors.go:59`). Either state "409
  `eval_project_deleting` (FAIL-DEL-5)" here and set the registry row to `409`,
  not retryable, or delete the pointer sentence from DEL-R43.

### [execution/control-plane.md](../spec/execution/control-plane.md)

- **Gap.** A restarted process of the same principal registers while the old
  instance still holds a confirmed control lease: is it rejected, does it
  supersede immediately, or does it wait for the lease to expire? Sources:
  07#53, 02#22, 02#23, 06#58. CTL-R24 rejects "a second live `instance_id`",
  CTL-R20 lets a later process of the same principal replace an earlier
  registration that shares its endpoint, and CTL-R21 makes the old registration
  unavailable after its confirmed lease expires.

### [execution/planners.md](../spec/execution/planners.md)

- **Gap.** Does PLN-R77 (the strategy maps an expected Worker failure into a
  failed candidate) apply to model-backed strategies? Sources: 04#24, 14#21.
  For Streamline and Router a WorkerFailure becomes a closed tool error and the
  Planner may reason on or still finish succeeded (WRK-R39, PLN-R73); only
  Passthrough maps it directly to a failed candidate.
- **Duplicate.** Planner rules are stated twice: SCH-R37 and PLN-R112, SCH-R38
  and PLN-R85, SCH-R39 and PLN-R67, SCH-R49 and PLN-R80 (scheduler.md), TPL-R26
  and PLN-R82, TPL-R27 and PLN-R84 (agent-templates.md). Sources: 04#24, 04#25,
  00#157, 00#154. Keep one owner per pair and link from the other.
- **Stale figure.** The Planner overview image
  (`assets/contractor-planner.png`) labels Passthrough with "create subtask"
  and a "Simple subtask lifecycle: add → execute → finish", but the text says
  Passthrough has no reasoning step, no model-facing Planner tools and no plan
  operations or finish: it sends one deterministic Worker-task request with
  subtask ID 0. Sources: 00#3, 00#82, 14#78.

### [execution/runs.md](../spec/execution/runs.md)

- **Duplicate.** RUN-R35 restates ART-R92 (accepting the success StageResult,
  applying the output mapping, freezing outputs and `running -> succeeded` are
  one transaction) and ART-R95 (frozen outputs are never implicitly published),
  with links. Source: 00#128.
- **Duplicate.** SEC-R14 in
  [platform/security.md](../spec/platform/security.md) restates RUN-R15
  (secrets enter a Runtime Agent only through `RuntimeSettings`; parameters are
  not a secret channel). Source: 00#56.
- **Duplicate.** RCF-R133 in
  [configuration/runtime-config.md](../spec/configuration/runtime-config.md)
  restates RUN-R6 (immutable Run-selected labels; `default` pinned first; every
  binding resolved and pinned in the Run transaction); reduce RCF-R133 to a
  link. Source: 07#69.

### [execution/runtime-agent.md](../spec/execution/runtime-agent.md)

- **Contradiction.** WSP-R163 in
  [tools/workspace.md](../spec/tools/workspace.md) says release "makes the one
  Runtime slot idle" (10#85), but RTA-R90, RTA-R111 and RTA-R92 keep the
  allocation identity and cached report fenced after private release, and only
  a later heartbeat release action lets the Runtime become idle (02#62, 02#65,
  matching CTL-R124). Sources: 02#62, 10#85.
- **Gap.** Can an unremovable orphan directory remain while the slot reports
  idle? Sources: 01#75, 02#65, 10#85. RTA-R52 removes recognized orphan
  allocation directories on startup before the slot can report `idle`, while
  10#85 (tools/workspace.md) logs and retains a stale marked directory that
  cannot be removed, without blocking initialization or later allocations.
- **Wrong cross-reference.**
  [platform/failure-model.md](../spec/platform/failure-model.md) names RTA-R42
  and RTA-R43 as owners of the bounded preparation failure, but that
  classification is RTA-R106, the owner requirement of FAIL-RTA-3. Source:
  07#102.

### [execution/scheduler.md](../spec/execution/scheduler.md)

- **Contradiction.** Does a capacity-deferred Stage consume its execution
  deadline? Sources: 02#94, 02#92, 04#4. SCH-R19 retries with backoff "until
  capacity appears, the execution deadline expires or cancellation is
  requested", while RUN-R28 says `admitted_at` starts the Stage's wall-clock
  budget, "so a queued Stage can wait for capacity without consuming its
  execution deadline".
- **Duplicate.** CTL-R119 in
  [execution/control-plane.md](../spec/execution/control-plane.md) restates
  SCH-R56 (the abort command performs no semantic work, is idempotent for
  `abort_id` and returns the cached report on retry), each with its own failure
  row (FAIL-CTL-6, FAIL-SCH-7). Source: 04#82.

### [execution/worker.md](../spec/execution/worker.md)

- **Gap.** How does a secret-bearing summarizer result fail? Sources: 15#32,
  14#38, 15#31. WRK-R152 makes missing, invalid, mismatched or oversized
  summarizer output the retryable `worker_summarization_failed`; WRK-R59 makes
  secret-retention violations of ordinary results non-retryable, and 15#31
  applies the same secret scan to summarizer output. The sources do not say
  whether it fails as non-retryable `unsafe_worker_result` or as
  `worker_summarization_failed`.
- **Gap.** Which code and retryability does an unsupported provider structured
  response from the result finalizer get? Sources: 14#34, 14#38. WRK-R52 says
  it fails closed with a bounded stable Worker error; FAIL-WRK-3 treats an
  invalid response as the retryable `worker_result_invalid` and leaves the
  unsupported case unspecified.

### [findings/findings.md](../spec/findings/findings.md)

- **Duplicate.** CAN-R130 in
  [tools/code-analysis.md](../spec/tools/code-analysis.md) restates FND-R152
  (annotation identity versus participation; `changed=false`); reduce it to a
  link. Source: 13#9.
- **Duplicate.** AUD-R103 and AUD-R104 in
  [audits/audits.md](../spec/audits/audits.md) restate FND-R153 (known
  `CWE@4.20` references are accepted without pinning and never satisfy an
  assigned entry; unknown IDs and other versions are rejected). Source: 27#58.
- **Specification versus code.** DEL-R28 says source-Run deletion "marks its
  imported finding proposal receipts `audit-held`" (19#102), but the trigger
  `contractor_reconcile_finding_retention`
  (`internal/persistence/migrations/000093_fold_finding_proposal_retention.sql`)
  sets `audit-held` as soon as an Audit hold row exists, before any deletion,
  and reverts to `source-held` or `discarded` when the last hold is removed;
  the findings state diagram therefore shows no transition into `audit-held`.
  Sources: 19#175, 19#102, 27#15. DEL-R28 and DEL-R29 also restate FND-R154,
  and LIFE-DEL-6 and LIFE-FND-1 are two lifecycle rows for the same cascade:
  reconcile DEL-R28 with the implementation, link FND-R154 and merge the rows.
- **Duplicate.** ART-R80 in
  [artifacts/artifact-store.md](../spec/artifacts/artifact-store.md) and
  FND-R119 both state that Run creation does not follow the JSON refs of a
  findings collection ZIP and has no collection-specific recursive input
  handling. Source: 27#33.
- **Duplicate.** The `[runtime-target-credential]` capture rule is stated three
  times: FND-R52 here (27#66), HTP-R74 in
  [tools/http-caido.md](../spec/tools/http-caido.md) (11#36) and PRJ-R84 in
  [artifacts/projects.md](../spec/artifacts/projects.md) (17#42). Pick one
  owner and link from the others. Sources: 27#66, 11#36, 17#42, 33#96.
- **Gap.** Several findings error codes are unnamed: invalid or oversized
  collections report "a bounded `findings_*` code" during preparation (27#38),
  an "explicit limit error" has no code (27#43), and publication gives HTTP
  statuses `404`, `409`, `400` and `422` without codes (27#28). The Runtime
  uses further codes that the specification does not name, such as
  `findings_collection_invalid`, `findings_limit_exceeded` and
  `findings_cursor_invalid`. Sources: 27#38, 27#43, 27#47, 27#28.

### [operations/operations.md](../spec/operations/operations.md)

- **Incomplete registry row.** LIFE-CTL-4 (RuntimeAgentPrincipal) in
  [platform/lifecycle.md](../spec/platform/lifecycle.md) still says "not
  specified" for Deleted by, Blocked by and Cascades, although OPS-R35
  specifies the removal: deleted by the operator through
  `DELETE /v1/operations/runtime-agent-principals/{runtimeAgentId}`
  (`deleteRuntimeAgentPrincipal`), blocked by a non-empty label set or a live
  instance or allocation that refers to it, and a later registration seeds it
  anew. Sources: 07#123, 06#118.

### [overview.md](../spec/overview.md)

- **Possible duplicate limit.** LIM-SYS-2 (one active Server/Control Plane on
  the deployment host, SYS-R10) and LIM-PER-1 (one active Server/Scheduler
  process per PostgreSQL database, PER-R1) describe the same
  single-active-Server bound and are enforced by the same advisory lease
  (PER-R2). Decide whether they are one bound; if so, drop LIM-SYS-2 and change
  [platform/persistence.md](../spec/platform/persistence.md) in the same
  change.

### [platform/failure-model.md](../spec/platform/failure-model.md)

- **Contradiction.** Is a 4xx Gateway response that carries
  `x-should-retry: true` or matches a `modelUnavailable` signature retryable?
  Sources: 02#71, 04#7. WRK-R18 (02#71) makes every HTTP error other than 408,
  409, 429 and 5xx non-retryable, including 400, 401, 403, 404, 413 and 422;
  FAIL-R2 and FAIL-R5 (04#7) make an explicit hint transient and a signature
  match (an exact status such as 400) transient `model_unavailable`. 04#7 also
  does not say whether the hint overrides 401 and 403; the Go classifier
  returns permanent for 401/403 before it checks the hint.
- **Gap.** Are `insufficient_quota`, `budget_exceeded` and
  `context_length_exceeded` non-retryable without a Gateway declaration?
  Sources: 02#71, 04#7. WRK-R18 fixes them as non-retryable in the Runtime
  Worker; FAIL-R6 leaves provider codes to each Gateway's `permanentCodes` (the
  shared fixture declares `insufficient_quota` and `context_length_exceeded`
  there).
- **Duplicate registry rows.** The failure registry has two rows for several
  operations: FAIL-RCF-4 and FAIL-FAIL-2 (label resolution at Run creation),
  FAIL-RCF-7 and FAIL-FAIL-3 (Agent-label assignment), FAIL-RCF-9 and
  FAIL-FAIL-4 (temporary insufficient compatible capacity), FAIL-RTA-3 and
  FAIL-FAIL-5 (adapter preparation). Sources: 02#71, 04#6, 04#7, 07#102,
  07#104, 07#105. Since this document owns 07#104 and 07#105, RCF-R135,
  RCF-R141 and RCF-R181 should cite FAIL-FAIL-2, FAIL-FAIL-3 and FAIL-FAIL-4,
  FAIL-RCF-4, FAIL-RCF-7 and FAIL-RCF-9 should keep only the binding-pin race,
  the compare-and-set and the try-another-candidate step, FAIL-RTA-3 should be
  narrowed to preparation outside adapter preparation, and FAIL-RCF-10 should
  state how it relates to the schema mismatch of FAIL-R16.

### [platform/lifecycle.md](../spec/platform/lifecycle.md)

- **Incomplete registry row.** RCF-R54 applies the RCF-R128 deletion fence to
  both LLM and runtime credentials, but LIFE-RCF-1 (LLM credential) cites only
  RCF-R51 and LIFE-RCF-5 (runtime credential) only RCF-R176; under LIFE-R3,
  LIFE-RCF-1 should also name RCF-R128 and RCF-R54 and LIFE-RCF-5 should name
  RCF-R54. Sources: 07#23, 07#51.

### [platform/observability.md](../spec/platform/observability.md)

- **Inconsistent wording.** Invariant 5 of
  [execution/runs.md](../spec/execution/runs.md) says labels are never used "as
  unbounded metric dimensions", weaker than OBS-R108 "Metadata labels are not
  metric dimensions". Sources: 04#67, 16#30. Align the invariant with OBS-R108.
- **Gap.** Does the Worker `otlp-http@1` adapter also refuse redirects and
  ambient proxy variables? Sources: 07#120, 07#40, 07#41. OBS-R95 scopes "does
  not read ambient proxy variables, follow redirects or retry" to the Planner
  adapter, because Worker delivery retries up to `maxAttempts`; the source says
  nothing about redirects or proxies for the Worker.
- **Duplicate with different wording.** The Agent Skill telemetry projection is
  stated in OBS-R29 (keeps "result size"; drops instructions, resource and
  package bytes, extracted paths and generated context) and in SKL-R114 (keeps
  "result byte count" and also never retains owner refs and revisions).
  Sources: 04#68, 09#85. Merge them under one owner.
- **Duplicate.** The MemoryTools diagnostic projection is stated in OBS-R28
  (operation, `worker_name`, note name, byte sizes, outcome, duration) and in
  MEM-R78 (request and result byte sizes, success or error code), which
  [tools/memory.md](../spec/tools/memory.md) owns (08#56). Sources: 04#68,
  08#56. Reduce OBS-R28 to a link, keeping its 04#68 rule that content,
  description and tags are dropped rather than truncated.
- **Duplicate limits.** The 16 UTF-8 byte secret-matching threshold is one
  shared policy (04#68) but is registered three times: LIM-OBS-3, LIM-WRK-12
  and LIM-CTL-5. Sources: 04#68, 14#39, 02#82. Merge them into one entry.
- **Duplicate.** OBS-R17 and OBS-R18 (a Gateway quota rejection follows the
  ordinary bounded LLM error path; ModelPolicy ceilings are enforced
  independently) are also stated by TPL-R17 in
  [configuration/agent-templates.md](../spec/configuration/agent-templates.md)
  and RCF-R36 in
  [configuration/runtime-config.md](../spec/configuration/runtime-config.md).
  Source: 06#102.

### [platform/persistence.md](../spec/platform/persistence.md)

- **Duplicate.** CTL-R69 and CTL-R27 in
  [execution/control-plane.md](../spec/execution/control-plane.md) restate
  PER-R11 (multiple Control Plane replicas must not share an unfenced
  `runtime_agents` table) and PER-R12 (durable principals and their labels are
  configuration records only); replace them with links. Source: 02#58.

### [platform/security.md](../spec/platform/security.md)

- **Contradicting limits.** SEC-R49 and LIM-SEC-8 set the login raw JSON body
  limit to 8 KiB (justified by a 6,237-byte worst case), and the handler uses
  `maximumLoginBodyBytes = 8 * 1024`
  (`internal/httpapi/public/auth_handlers.go:22`), but the login operation in
  `api/openapi/contractor-public-v1.yaml` declares
  `x-max-request-body-bytes: 2048`. Source: 06#138. Reconcile the OpenAPI
  document with the specification.
- **Gap.** May the UI `apiBaseUrl` use HTTP on an RFC 1918 address? Sources:
  06#144, 06#27. SEC-R60 lets insecure loopback mode accept exact HTTP browser
  origins on RFC 1918 private IPv4 literals for a trusted local-network demo
  proxy, while UI-R10 in [ui/ui.md](../spec/ui/ui.md) requires HTTPS for
  `apiBaseUrl` except for a loopback origin in local development; a demo proxy
  that also serves the API over HTTP on an RFC 1918 address satisfies the first
  and violates the second.

### [proposals/0034-audit-check-prioritization.md](../proposals/0034-audit-check-prioritization.md)

- **Ambiguous limit.** 34#31 limits item and evidence IDs to at most 160 bytes,
  but the bounds row of 34#41 reads "16 evidence IDs and 16 missing-context
  strings of at most 256 bytes each", which can be read as allowing 256-byte
  evidence IDs. Sources: 34#31, 34#41. The 256-byte bound is probably meant for
  missing-context strings only; the source does not say so.
- **Possible inconsistency.** The separate example profile pins `maxRounds=3`
  for the multi-pass journey (34#56), while the verification case reads "Two
  passes of 10 from 25 leave five not-tested after `maxRounds=2`" (34#63).
  Sources: 34#56, 34#63. Confirm whether the case uses a separate test profile.

### [tools/code-analysis.md](../spec/tools/code-analysis.md)

- **Gap.** Should the trace-oriented template example, which selects only
  `search_def` and `list_symbols` from code-analysis@1 (13#47), mirror the
  checked-in `workspace_taint_analyst@1` template, which selects the complete
  local graph surface (13#49, CAN-R132)? Sources: 13#47, 13#49.

### [tools/code-execution.md](../spec/tools/code-execution.md)

- **Confirm reading.** EXE-R66 reads 21#41 ("the current StageContentRequest
  has no additional wall-clock deadline field") as: `StageContentRequest`
  carries no wall-clock deadline for a command beyond the supplied execution
  deadline of EXE-R63, and `StageContentRequest.deadline` is WRK-R6 (01#110).
  Sources: 21#41, 01#110. Confirm that the supplied execution deadline is the
  WRK-R6 deadline, with no separate per-command field.
- **Duplicate.** CTL-R78 in
  [execution/control-plane.md](../spec/execution/control-plane.md) restates
  EXE-R13 and EXE-R120 (`podman@1` placement requirements, which are
  binding-specific); reduce it to a link. Sources: 02#85, 21#11.

### [tools/http-caido.md](../spec/tools/http-caido.md)

- **Duplicate.** The last sentence of RCF-R105 in
  [configuration/runtime-config.md](../spec/configuration/runtime-config.md)
  restates HTP-R123 (a label can retarget the Caido client but cannot make
  `caido@1` model-visible). Source: 07#66.
- **Duplicate.** CTL-R77 in
  [execution/control-plane.md](../spec/execution/control-plane.md) restates
  HTP-R14 (placement rejects a selected Caido Toolset when labels do not
  resolve `caido-graphql@1` and chooses an Agent that advertises it). Source:
  11#14.
- **Duplicate.** ART-R24 in
  [artifacts/artifact-store.md](../spec/artifacts/artifact-store.md) restates
  HTP-R66 (the reserved `http.body.` prefix; generic model-visible Artifact
  tools neither list nor read or write the bindings; a Toolset authority
  boundary). ART-R24 should link HTP-R66; its reserved-prefix table row can
  stay. Source: 11#34.
- **Duplicate limits.** FND-R53 in
  [findings/findings.md](../spec/findings/findings.md) defines LIM-FND-7 (1 MiB
  outgoing body per attempt) and LIM-FND-9 (13 records: ten redirects plus
  three attempts), which mirror LIM-HTP-5, LIM-HTP-7 and LIM-HTP-9 ("Limits
  mirror existing HTTP bounds"). FND-R53 should cite those entries, stating the
  13-record total as a derived bound if it is kept. Sources: 11#20, 11#21,
  27#67.
- **Inconsistency.** HTP-R10 calls the `worker.caido` `endpoint` an absolute
  `http` or `https` origin, which by definition has no path, yet lets a
  deployment include a fixed path prefix in it. Source: 11#12.

### [tools/memory.md](../spec/tools/memory.md)

- **Duplicate.** ART-R19, ART-R21, the Rationale after ART-R21 and ART-R22 in
  [artifacts/artifact-store.md](../spec/artifacts/artifact-store.md) restate
  MEM-R89 to MEM-R92 (the `memory.` prefix reservation and filtering by every
  model-visible artifact-backed Toolset, the domain-neutral private Artifact
  API used by the trusted wrapper, and `memory.*` in reserved Namespaces not
  being a note). Sources: 08#15, 03#25. Reduce them to links, keeping only what
  03#25 and 03#106 add (filtering even when the same AgentTemplate selects
  MemoryTools, no other endpoint or grant kind, ART-R20).
- **Decision.** Should corrupt stored ordinal state get its own error code?
  Sources: 08#54, 08#33. MEM-R48 returns `memory_unavailable` for it, which the
  registry (ERR-R3) defines as a retryable "Bounded infrastructure failure",
  although retrying cannot clear corrupt state; memory.md and FAIL-MEM-3 say so
  explicitly. A separate code would change the exact Memory code set of
  MEM-R72.

### [tools/scanners.md](../spec/tools/scanners.md)

- **Overloaded code.** `scan_incomplete` means both an ffuf observation with
  missing or incomplete progress (SCN-R108) and the failure of a `scan-plan@1`
  Stage whose selected jobs did not all complete (SCN-R214). Sources: 03#58,
  32#23. The errors registry notes both meanings and leaves its retryability as
  the Stage failure unspecified.
- **Legacy compatibility.** 29#16 says "Single-report invocation digests and
  receipt shapes remain compatible with previously completed invocations";
  SCN-R40 states only the current rule (an invocation with no additional result
  bindings computes its digest and writes its receipt without
  additional-binding fields). Source: 29#16. Decide whether that rule is still
  wanted or SCN-R40 should be removed (its ID retired).

### [tools/skills.md](../spec/tools/skills.md)

- **Duplicate.** TPL-R43 in
  [configuration/agent-templates.md](../spec/configuration/agent-templates.md)
  restates SKL-R11 and SKL-R12 (rejecting Toolset operations named
  `list_skills`, `load_skill` or `load_skill_resource`; the tool-using
  ModelPolicy requirement); reduce it to a link. Sources: 01#46, 09#13.
- **Gap.** Who performs the Skill RunScope fork? Sources: 09#39, 09#53, 09#58.
  SKL-R46 says SkillCatalog owns "exact UserScope-to-RunScope skill forks",
  SKL-R72 has Scheduler fork each selected version, SKL-R78 says "Scheduler
  alone creates its bindings" and SKL-R71 names Run initialization; the sources
  do not say whether Scheduler forks through SkillCatalog.
- **Duplicate.** PRJ-R20 and PRJ-R30 in
  [artifacts/projects.md](../spec/artifacts/projects.md) restate SKL-R5 and
  SKL-R71 (09#6), and ART-R14, ART-R39 and ART-R108 to ART-R110 in
  [artifacts/artifact-store.md](../spec/artifacts/artifact-store.md) restate
  SKL-R78 to SKL-R80 (09#58). Sources: 09#6, 09#58, 03#26. Reduce the
  artifact-plane copies to links.

### [tools/toolsets.md](../spec/tools/toolsets.md)

- **Specification versus code.** TLS-R22 lists three infrastructure channels
  (`runtime-http-client`, `runtime-subprocess-launcher`,
  `caido-graphql-client`), but
  `api/testdata/toolset-descriptors.json` also declares
  `sandbox-execution` for `code-execution@1` `exec_command`, and the Server
  descriptors and Runtime code-execution Toolset use it; no specification
  document names that channel. Source: 01#55.
- **Duplicate.** HTP-R3 (a Caido mutation is permitted only when its exact tool
  name is selected) and HTP-R5 (the model never chooses an infrastructure
  endpoint, proxy credential or Caido bearer token) in
  [tools/http-caido.md](../spec/tools/http-caido.md) restate TLS-R5 and
  TLS-R10. Sources: 11#56, 11#4.
- **Duplicate.** TPL-R42 in
  [configuration/agent-templates.md](../spec/configuration/agent-templates.md)
  restates TLS-R19 (AgentTemplateCatalog rejects unknown Toolset refs, unknown
  tool names and cross-Toolset collisions independently of the connected
  fleet). Source: 01#57.
- **Duplicate.** TLS-R37 and ART-R33 state the same rule (a domain Toolset may
  use the allocation-bound private Artifact client internally without exposing
  `run-artifacts@1` operations). Sources: 01#53, 03#24. Keep ART-R33, whose
  domain owns 03#24, and reduce TLS-R37 to a link.

### [tools/workspace.md](../spec/tools/workspace.md)

- **Legacy compatibility.** WSP-R130 keeps "the legacy `source-analysis@1`
  compatibility path" below allocation-private `scratch/source` for `openapi@1`
  without a project workspace, and WSP-R131 gives the precedence rule. Source:
  10#71. Under the no-backward-compatibility rule the path should be removed
  and WSP-R130 and WSP-R131 retired.

### [ui/journeys.md](../spec/ui/journeys.md)

- **Possible duplicates.** The story completion criteria for uis#9, uis#12,
  uis#14, uis#15, uis#17, uis#21 and uis#27 resemble more detailed rules
  elsewhere: UI-R108, UI-R117, UI-R121, UI-R123, UI-R169, UI-R79, UI-R120,
  UI-R82, UI-R89, UI-R90 and UI-R50 to UI-R53 in [ui/ui.md](../spec/ui/ui.md),
  PRJ-R38, and RUN-R101, RUN-R104 and RUN-R105. Decide whether to fold them
  into links.
- **Incomplete registry row.** The custom duration of the time-limit dialog
  (UIJ-R77) is bounded by LIM-AUD-23 (365 days), but LIM-AUD-23 names only the
  Audit API as enforcement point; add the Web UI dialog. Sources: 19#191,
  uis#25.
- **Incomplete registry row.** UIJ-R38 to UIJ-R42 add lifecycle facts to the
  Run draft (restore for the same user, Workflow version and Project; explicit
  reset with confirmation; completion on successful launch; no persistence
  guarantee across a tab reload), but the LIFE-UI-2 row cites only UI-R79 and
  leaves Deleted by "not specified". Source: uis#14.

### [ui/ui.md](../spec/ui/ui.md)

- **Decision.** Where does the configuration workspace that lists RuntimeConfig
  versions and credentials live in the navigation? Sources: 06#109, 18#5, 18#6,
  06#5. 06#109 calls it the "Runs Configuration workspace", but the Runs
  destination has exactly two views (Queue and Completed) and 06#5 lists LLM
  configurations and Credentials under Operations; ui.md calls it "the
  configuration workspace", keeps an open-question paragraph after UI-R158, and
  the client route table has no entry. The UI routes it at
  `/operations/configuration` (`ui/src/app/router.tsx`,
  `RuntimeConfigurationLayout`), which matches 06#5.
- **Duplicate.** Rules that ui.md owns are restated elsewhere: PRJ-R34,
  PRJ-R38, PRJ-R45, PRJ-R50 and PRJ-R100 in
  [artifacts/projects.md](../spec/artifacts/projects.md), ART-R71 in
  [artifacts/artifact-store.md](../spec/artifacts/artifact-store.md) and
  OPS-R63 in [operations/operations.md](../spec/operations/operations.md)
  restate UI-R130, UI-R165 and UI-R181 to UI-R186. Sources: 06#16, 06#47,
  06#152, 06#155, 20#52. Link to ui.md; FAIL-PRJ-3 stays the one failure row
  for Project Workflow matching.

## Resolutions recorded during the migration

How apparent conflicts between legacy sources were settled. Nothing here needs
a decision; the entries are history.

### [api/public-api.md](../spec/api/public-api.md)

- 06#147, 07#128: the blocks agree; API-R28 states the RuntimeCredential
  deletion idempotency exception and links RCF-R176 for the replay response and
  key-format validation.

### [artifacts/artifact-store.md](../spec/artifacts/artifact-store.md)

- 03#27, 11#34: ART-R36 keeps the union of exceptions to the generic tools'
  read authority: `skills`, `memory.` and `http.body.`.
- 03#106, 03#3, 17#11: ART-R1 keeps three structural scopes (UserScope,
  ProjectScope, RunScope) in one ArtifactStore.
- 00#113, 03#61, 03#62, 17#21: ART-R74 keeps the endpoint-selected input source
  scope (UserScope for a standalone Run, owner-authorized ProjectScope for a
  Project Run).
- 01#94, 03#17: ART-R16 keeps the complete list of Namespaces that cannot be
  Agent Namespaces: `inputs`, `outputs` and `skills`.
- 03#62, 00#113, 03#103, 17#21: ART-R76 names the trusted Run-creation
  operation as the input-fork actor (Workflow Scheduler for a standalone Run,
  the Project endpoint for a Project Run).

### [artifacts/blob-storage.md](../spec/artifacts/blob-storage.md)

- 03#104, 23#11: BLB-R14 takes the 23#11 operation set (store, open, delete an
  exact object) as normative; the session-based 03#104 protocol is kept as an
  illustrative sketch under Interfaces.
- 23#20, 23#21: BLB-R48 and FAIL-BLB-6 say a re-upload creates a new revision
  and cannot silently repair an old exact reference, apart from the same-digest
  replacement of missing content in BLB-R41.

### [artifacts/git-import.md](../spec/artifacts/git-import.md)

- 24#23, 24#24: GIT-R61 requires shallow negotiation at depth one, and a remote
  that cannot provide a bounded selected-ref snapshot fails explicitly.

### [artifacts/projects.md](../spec/artifacts/projects.md)

- 00#104, 17#30, 17#36, 17#51: every declared output is published whether or
  not it is primary, and `primary` affects only recommendations (PRJ-R48,
  PRJ-R52, PRJ-R102).
- 06#152, 17#31: recommendations are suppressed only when all primary
  destinations exist (PRJ-R49), and a suppressed Workflow stays available
  through All workflows (PRJ-R50).
- 17#45, 17#4: PRJ-R89 renders the Queue scope as "`project` Projects,
  `evaluation` Projects and standalone execution".

### [artifacts/retention-and-deletion.md](../spec/artifacts/retention-and-deletion.md)

- 03#89, 18#21, 23#23: DEL-R9 uses the most specific referencer list of 18#21
  (binding, retained revision, pin, lineage edge or other scope outside the
  purge set); physical object selection is BLB-R51.
- 18#16, 19#171: both Run deletion gates are kept (DEL-R16, DEL-R17), and
  DEL-R20 takes the missing-receipt conflict (`run_not_deletable`, reason
  `audit_collection_pending`) from the public OpenAPI document.

### [audits/execution.md](../spec/audits/execution.md)

- 19#101, 19#178: AUX-R92 makes an unexpected proposal under
  `findingConfirmation: disabled` contract-invalid.

### [audits/worker-completion.md](../spec/audits/worker-completion.md)

- 25#2, 25#21: AWC-R19 makes retained `audit-results@1` snapshots unplaceable,
  and AWC-R5 states 25#2 with that snapshot as an explicit exception.

### [configuration/agent-templates.md](../spec/configuration/agent-templates.md)

- 09#10, 01#15: "existing versionless ArtifactRef" in `spec.skills` means the
  ArtifactRef shape; packages are resolved only at Run initialization (TPL-R11,
  TPL-R14).
- 01#11, 01#7, 29#2: TPL-R2 scopes `instructions`, `modelPolicy`, `skills` and
  `summarizer` to `runtime: adk@1`; the other fields and the closed-object rule
  apply to every runtime, and TPL-R5 links the `tool@1` contract.

### [configuration/catalog.md](../spec/configuration/catalog.md)

- 00#15, 00#17, 19#17: CAT-R7 names the six YAML subtrees explicitly;
  `instructions` is a non-YAML subtree.

### [configuration/runtime-config.md](../spec/configuration/runtime-config.md)

- 07#96, 29#12: the wire sketch is kept verbatim; RCF-R160 and a note under the
  wire contract say `llmGatewayUrl` may be absent.
- 01#108, 07#96: conceptual snake_case names are in RCF-R157, camelCase wire
  names in Interfaces.
- 06#87, 06#95: RCF-R48 keeps the 06#87 credential-deletion blockers
  (non-terminal Run, active label binding, live allocation snapshot), and
  RCF-R69 points to it.
- 00#24, 06#86: RCF-R47 makes a YAML default that names a missing credential
  runnable only when a higher-precedence Run or Agent layer replaces it before
  prepare.

### [configuration/workflows.md](../spec/configuration/workflows.md)

- 00#96, 00#106, 17#21: WFL-R53 and WFL-R83 follow ART-R76 and name the trusted
  Run-creation operation, not Scheduler, as validating and forking inputs;
  LIM-WFL-7 names the same enforcer.
- 00#104, 17#30: WFL-R69 keeps both scopes: handoff outputs are normally
  non-primary, and never primary in built-in generator Workflows.
- 00#13, 00#91: WFL-R11 states presentation absence as ordinary optional-field
  behaviour, without the "predate" history qualifier.

### [evals/managed-evals.md](../spec/evals/managed-evals.md)

- 30#39, 30#50: EVM-R130: the submission POST acknowledges the durable request,
  the member view shows only verified executions, and a successful response
  returns the verified association.

### [execution/control-plane.md](../spec/execution/control-plane.md)

- 10#28, 02#88, 02#85, 21#11: workspace storage constrains placement only for
  `podman@1` and is otherwise diagnostics (CTL-R78, CTL-R79).
- 10#17, 02#13: AllocationSpec carries no version label ("private v2 allocation
  manifest" is dropped), following 02#13.
- 02#101, 02#119, 04#84: allocations are released after the terminal Stage
  outcome, except under Queue Pause, where drain and release may finish while
  the StageExecution keeps its finalizing candidate or aborting termination
  (CTL-R114, CTL-R122, CTL-R123).
- 22#48, 02#13, 02#14: CTL-R138 states the performance-metrics registration
  field without old/new release framing: absence is valid, Servers are upgraded
  before Runtimes and no fallback hides a protocol error.

### [execution/planners.md](../spec/execution/planners.md)

- 04#8, 04#9: PLN-R89 drops "only": any other failed model request ends with
  the retryable `planner_gateway_unavailable` (FAIL-PLN-1).

### [execution/runs.md](../spec/execution/runs.md)

- 06#44, 04#4, 00#125: RUN-R99 retains the repeat-request fragment at Run
  creation in the same transaction as the input forks; "and Run admission" is
  dropped.

### [execution/runtime-agent.md](../spec/execution/runtime-agent.md)

- 02#30, 02#31: the factory probe timeout is twelve seconds (LIM-RTA-1,
  `DEFAULT_FACTORY_PROBE_TIMEOUT_SECONDS = 12.0`), and RTA-R19 refers to
  LIM-RTA-1 instead of the stale five seconds.

### [execution/scheduler.md](../spec/execution/scheduler.md)

- 02#119, 04#90, 04#84, 18#14, 00#172: SCH-R47, SCH-R67 and SCH-R68 keep the
  Queue Pause exception to releasing allocations only after the terminal Stage
  outcome.
- 03#32, 04#36, 08#45, 04#90: SCH-R60 fences all grants before the durable
  transition and links MEM-R97 for the barrier ordering and MEM-R63 for
  `memory_forbidden`; SCH-R46 and ART-R86 reject writes from the transition
  onward.
- 20#6, 04#4, 18#13: SCH-R129 says a running Run remains running while it lacks
  capacity; `pending` and admission are left to runs.md.
- 20#56, 12#16, 20#30: the fairness non-goal is narrowed to guarantees beyond
  the rotation of SCH-R121 and SCH-R123.

### [findings/findings.md](../spec/findings/findings.md)

- 19#95, 27#55, 27#56, 27#57: the finding tool follows 27 (hidden derived
  `client_key`, optional evidence refs, nullable subject; FND-R16, FND-R18,
  FND-R25, FND-R27), matching the Runtime facades.
- 19#72, 27#57, 27#18: the proposal example uses a `call-` SHA-256 client key
  and `evidence-1`, matching FND-R25 and FND-R93.
- 27#10: the Audit-domain identifier grammar (1–160 ASCII bytes,
  `[A-Za-z0-9][A-Za-z0-9._:-]*`) stays in FND-R77 and LIM-FND-11 because
  audits.md does not state it.
- 27#27, 27#28, 27#30: publication validates and normalizes before the replay
  lookup, and more than 256 selectors is rejected like an invalid selector
  (FND-R103, LIM-FND-25, FAIL-FND-3), matching `internal/findingintake`.

### [operations/operations.md](../spec/operations/operations.md)

- 06#118, 07#123: OPS-R35 requires an empty label set and no live instance or
  allocation referring to the principal before offline removal.

### [overview.md](../spec/overview.md)

- readme#11: SYS-R22 cites SCH-R52 to SCH-R58 so the abort deadlines are
  included, and the Stage completion diagram uses one "cancelled or
  interrupted" node because the source gives no mapping.

### [platform/observability.md](../spec/platform/observability.md)

- 04#68, 14#39: OBS-R31 keeps one secret-matching rule and applies `[REDACTED]`
  only to metrics; Worker results and Agent Card checks reject instead
  (WRK-R60, WRK-R61, CTL-R110).

### [proposals/0034-audit-check-prioritization.md](../proposals/0034-audit-check-prioritization.md)

- 34#4, 34#34: `fewer_candidates_remaining` applies when fewer than `topN`
  candidates remain.

### [tools/code-analysis.md](../spec/tools/code-analysis.md)

- 12#71, 12#73: `workspace_source_graph_analyst@1` selects all eleven
  code-analysis@1 operations.

### [tools/code-execution.md](../spec/tools/code-execution.md)

- 21#39, 21#40: EXE-R76 treats Podman CLI exit codes 125–127 as
  `sandbox_outcome_unknown`, and EXE-R72 (a nonzero exit is `completed`) is
  subject to it.

### [tools/http-caido.md](../spec/tools/http-caido.md)

- 11#22, 11#36: HTP-R36 keeps the 11#22 redacted session-view list plus the
  rule that cookie values are not model-visible after being set.

### [tools/memory.md](../spec/tools/memory.md)

- 08#19, 08#25, 08#26, 03#38: MEM-R24 and MEM-R37 use the ArtifactStore's
  authoritative database clock (ART-R55), matching the PostgreSQL
  `clock_timestamp()` defaults.

### [tools/scanners.md](../spec/tools/scanners.md)

- 32#9, oas#20: SCN-R181 records missing SQLMap parameters and unsupported
  serialization as plan skips; the oas#20 "remain gaps" wording stays in
  audits/scans.md.
- 32#11, 11#29: scanners.md owns `scan_target_denied` and
  `scan_target_unresolved` (SCN-R56, FAIL-SCN-6), registered as not retryable;
  SEC-R78 is cited for the policy decisions.

### [tools/skills.md](../spec/tools/skills.md)

- 06#74, 09#51, 09#87: SKL-R59 keeps the separate Skills UI over the ordinary
  Artifact API; no Skill endpoint, active flag or table is a non-goal, and
  SKL-R65 was removed.

### [tools/workspace.md](../spec/tools/workspace.md)

- 10#3, 10#9, 07#64: Scheduler pins the workspace artifacts and the Control
  Plane sends the manifest carrying their exact refs (WSP-R2).
