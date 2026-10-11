# Glossary

Terms used across the specification. Each entry names the document that owns the concept. · [Specification index](README.md)

## A

**Abort** — The bounded `aborting` path, identified by `abort_id`, that a Stage takes when Scheduler-owned cancellation or interruption wins before `finalizing`: StageTermination is stored, writes are fenced and a bounded best-effort drain ends in `cancelled` or `interrupted`. Owner: [Scheduler](execution/scheduler.md).

**Aborting path** — see **Abort**.

**Accepted preparation output** — A profile-mapped output of a successful prepare role, retained with its exact revision and accepted atomically with the role's other mapped outputs. Owner: [Audit preparation and OpenAPI scans](audits/scans.md).

**Active dispatch** — The in-flight Worker dispatch of the current subtask, holding a Server-generated call ID and the logical Worker name. Owner: [Planners](execution/planners.md).

**Active note** — A Memory note whose current `memory.` binding exists in its Memory Namespace; old immutable revisions do not count as active notes. Owner: [Memory](tools/memory.md).

**Active Server** — The Server process that holds the advisory lease of a PostgreSQL database and therefore runs credential recovery, the process-local Control Plane, the Scheduler and the private API. Owner: [Persistence](platform/persistence.md).

**Active workflow** — A Caido workflow that `caido_workflow_run` starts asynchronously for a stored `request_id`. Owner: [HTTP and Caido tools](tools/http-caido.md).

**Adapter** — The provider-specific implementation of the preflight, prepare, submit, reconcile, observe, cancel and collect operations through which a case runs on existing Server execution. Owner: [Portable evaluation format](evals/format.md).

**Advisory lease** — The session-level PostgreSQL advisory lease a Server holds on a dedicated connection to be the one active Server, and the one active Control Plane instance, for its database; it is distinct from the Runtime Agent control lease. Owner: [Persistence](platform/persistence.md).

**Agent label set** — The authoritative durable set of labels assigned to one stable Runtime Agent principal. Owner: [Runtime configuration, labels, credentials and Gateways](configuration/runtime-config.md).

**Agent Skill** — A directory with `SKILL.md` in the open Agent Skills package format, also called a skill package, stored as one ZIP artifact and progressively disclosed to a Worker as model context; unrelated to an A2A Agent Card `AgentSkill`. Owner: [Agent Skills](tools/skills.md).

**AgentTemplate** — The reusable, immutable definition of a Worker's intended behavior, a Contractor contract that is neither an ADK object nor a running Agent. Owner: [Agent templates](configuration/agent-templates.md).

**AgentTemplateCatalog** — The in-process Server configuration boundary that resolves AgentTemplate selectors to exact refs and validated bodies. Owner: [Configuration catalog](configuration/catalog.md).

**AgentTemplateRef** — The `template_id`, `version` and `digest` that identify one exact AgentTemplate body. Owner: [Agent templates](configuration/agent-templates.md).

**Aggregate report** — The Stage-level application/json report that references the plan and every per-job report. Owner: [Scanners](tools/scanners.md).

**All workflows** — The Project view that always permits an explicit Run of a Workflow, including one suppressed from the Recommended group. Owner: [Projects](artifacts/projects.md).

**Allocation** — The assignment of one resolved Stage Agent binding to one Runtime Agent slot, identified by a globally unique, never reused allocation_id. Owner: [Control Plane](execution/control-plane.md).

**Allocation directory** — The fresh directory named `allocation-<32 hex>` that `local-workdir@1` creates below the work root for one allocation. Owner: [Runtime Agent](execution/runtime-agent.md).

**Allocation grant** — The Server-side record that binds the certificate-derived Runtime Agent principal, process `instance_id` and `allocation_id` and authorizes private Artifact API access to one RunScope. Owner: [Artifact store](artifacts/artifact-store.md).

**Allocation history** — The durable list of terminal allocations with their resource summaries or the reason none is available. Owner: [Operations](operations/operations.md).

**Allocation resource summary** — The optional runtime.resources block of a RuntimeReport describing Runtime process consumption during one allocation. Owner: [Operations](operations/operations.md).

**Allocation-scoped A2A endpoint** — The Runtime Agent's one stable authenticated A2A address, configured from the active AgentTemplate while the slot is allocated. Owner: [Runtime Agent](execution/runtime-agent.md).

**AllocationAdapterHost** — The allocation-owned holder of the RuntimeAdapters present in the resolved settings, constructed before the sandbox, any Toolset or the Worker. Owner: [Runtime Agent](execution/runtime-agent.md).

**AllocationFinalReport** — The report that finalization or abort of an allocation produces, carrying the allocation's Worker execution report and Runtime report; the Runtime caches it with the allocation until release. Owner: [Observability](platform/observability.md).

**AllocationSpec** — The complete, Server-generated construction input for one allocation, including the resolved template, ModelPolicy, RuntimeSettings, lease and deadline. Owner: [Control Plane](execution/control-plane.md).

**Allowed target network** — An operator-configured strict CIDR network, set at Runtime Agent startup, that unlocks loopback and link-local destinations but never an always-denied one. Owner: [Security](platform/security.md).

**Ambiguous outcome** — A result that does not prove whether a mutation committed or rolled back. Owner: [Failure model](platform/failure-model.md).

**Analyst / researcher** — The scenario role that prepares materials, starts analyses, reads results and makes Audit decisions; it is not an authorization role. Owner: [UI journeys](ui/journeys.md).

**Analyst severity** — The owner-assigned severity of a true_positive finding, distinct from the Worker's severity_suggestion. Owner: [Review and reports](audits/review-and-reports.md).

**Analyst verdict** — An owner's true_positive or false_positive decision on a finding, recorded through the finding-triage review flow. Owner: [Review and reports](audits/review-and-reports.md).

**Annotation block** — The contiguous canonical taint annotations immediately adjacent to a declaration, to which new annotation lines append. Owner: [Code analysis](tools/code-analysis.md).

**Annotation session** — The per-allocation state of taint-annotations@1 through which annotation calls serialize. Owner: [Code analysis](tools/code-analysis.md).

**Applicability coverage** — The ratio (satisfied + violated) / selected applicable, with inconclusive counted separately and a zero denominator reported as N/A. Owner: [Review and reports](audits/review-and-reports.md).

**Argument binding** — One entry of execution.arguments that takes its value from exactly one source: parameter, artifact or literal. Owner: [Scanners](tools/scanners.md).

**Arm** — One variant of the A/B comparison (baseline or candidate), bound to one exact Workflow or AuditProfile. Owner: [Managed Evals](evals/managed-evals.md).

**ArtifactBlobStore** — The replaceable internal blob interface, beside the ArtifactRegistry under the ArtifactStore, through which every payload is stored, opened and deleted. Owner: [Artifact blob storage](artifacts/blob-storage.md).

**ArtifactRef** — The public artifact address `(namespace, name, revision?)`; scope is never encoded in it. Owner: [Artifact store](artifacts/artifact-store.md).

**ArtifactRegistry** — The PostgreSQL component of ArtifactStore that holds scopes, immutable versions, logical bindings, CAS, grants and lineage. Owner: [Artifact store](artifacts/artifact-store.md).

**ArtifactStore** — The single physical artifact store of Contractor; UserScope, ProjectScope and RunScope are scoped views of it. Owner: [Artifact store](artifacts/artifact-store.md).

**Assessment** — Versioned scorer judgments over one exact result (`playground.assessment/v1`). Owner: [Portable evaluation format](evals/format.md).

**Attempt disposition** — The outcome that an AuditExecutionItem records for one attempt during collection. Owner: [Audit execution](audits/execution.md).

**AttemptDiagnostics** — The optional public projection of normalized, bounded Planner and Worker report errors in Run status. Owner: [Runs](execution/runs.md).

**Audit** — A long-lived, Project-bound assessment against an immutable baseline, a selected scope and a versioned program that coordinates ordinary WorkflowRuns, deterministic inventory items, evidence, finding proposals, rounds and authenticated human decisions. Owner: [Audits](audits/audits.md).

**Audit Controller** — The Server component that owns coordination across Runs: rounds, items, child Run intents, collection, coverage and Audit budgets. Owner: [Audits](audits/audits.md).

**Audit hold** — An exact Audit evidence hold or accepted proposal hold that keeps retained revisions alive under Audit ownership. Owner: [Review and reports](audits/review-and-reports.md).

**Audit package** — A versioned application/zip package with a strict canonical manifest.json (contractor.audit.package.v1) that carries a related variable-sized set through the ArtifactStore. Owner: [Audits](audits/audits.md).

**Audit revision** — The counter that advances on every projection-visible Audit mutation (but not on Controller claim traffic) and backs the Audit's strong ETag. Owner: [Audits](audits/audits.md).

**Audit standard package** — A canonical ZIP with media type application/vnd.contractor.audit-standard+zip, built from one contractor.audit-standard.v1 document holding curated requirements or risks, evidence contracts and mappings. Owner: [Audits](audits/audits.md).

**Audit workspace snapshot** — The owner-scoped SQL snapshot returned by GET /v1/audits/{auditId}/workspace, carrying auditRevision and asOf with current-round and whole-Audit counts. Owner: [Web UI](ui/ui.md).

**Audit-check completion contract** — `audit-check-results@1`, the opt-in completion strategy a trusted Audit check binding pins to one logical Worker, replacing the ordinary LLM serializer with Runtime-side result collection, a completeness gate, deterministic publication and a Runtime-authored WorkerResult. Owner: [Audit Worker completion](audits/worker-completion.md).

**`audit-managed` publication mode** — The trusted-only `publication_mode` of Audit child Runs that suppresses ordinary `outputs/<slot>` publication and is not accepted from the public Run-create API. Owner: [Projects](artifacts/projects.md).

**AuditArtifactLink** — The Audit record that retains exact immutable content independently of source Run lifetime and makes an Audit-managed binding visible to owner-facing Audit projections. Owner: [Review and reports](audits/review-and-reports.md).

**AuditExecution** — One submission of a Workflow role for an Audit, with at most one Run and one immutable execution-input manifest. Owner: [Audits](audits/audits.md).

**AuditItem** — The logical unit of a check, attempt, result, evidence, proposal attribution, coverage and settlement, even when several items share one AuditExecution and WorkflowRun. Owner: [Audits](audits/audits.md).

**AuditProfile** — The versioned, operator-authored configuration document with identity (name, version) that selects an Audit's mode, standards, inputs, inventory, Workflow roles, execution limits and interaction policy. Owner: [Audits](audits/audits.md).

**Authority revision** — The experiment revision and ETag that advance only on authority mutations and guard lifecycle and selection CAS. Owner: [Managed Evals](evals/managed-evals.md).

**Auto export** — Runtime-owned persistence of overlay state and human diff into declared Stage result slots before a graceful terminal A2A response. Owner: [Run workspaces and filesystem tools](tools/workspace.md).

**Available agents section** — The deterministic roster of logical Workers and their AgentTemplate descriptions that Server appends to the Router system instruction. Owner: [Planners](execution/planners.md).

## B

**Backtrace** — The retained relations that lead from an Audit finding to its source proposals, receipts and originating Runs, its verifying items, attempts and Runs, their Workflow closures and its checklist origins. Owner: [Review and reports](audits/review-and-reports.md).

**Baseline** — Everything the Audit start transaction freezes: exact inputs, scope, credential identities and holds, the resolved profile and Workflow closures, the Run RuntimeConfig snapshot, Skill packages, the standard selection, limits and interaction policy. Owner: [Audits](audits/audits.md).

**Bin filter token** — An opaque, context-bound token that selects exactly the cohort of one displayed chart bin for drill-down. Owner: [Managed Evals](evals/managed-evals.md).

**Binding** — (1) The logical `(scope, namespace, name)` entry that points to one current immutable version; conflicting writes by authorized participants to the same binding use compare-and-swap. Owner: [Artifact store](artifacts/artifact-store.md). (2) The provider selection and provider-specific execution and output mapping of one variant (`playground.binding/v1`). Owner: [Portable evaluation format](evals/format.md).

**Bootstrap secret file** — A secret-bearing file the Server reads once at startup under one shared file check: the master key file, the LLM Gateway admin binding and key files, and the local-auth file. Owner: [Security](platform/security.md).

**Bound** — A numeric maximum, minimum or allowed range on one named quantity (a size, count, depth, duration or resource); limits, ceilings and budgets are bounds. Owner: [Limits](platform/limits.md).

**Boundary result** — The deterministic result that a Runtime label-resolution or adapter-preparation failure produces where it crosses a component boundary. Owner: [Failure model](platform/failure-model.md).

**Bounded subprocess launcher** — The Runtime-owned launcher through which a selected tool runs its bounded subprocesses, including the child-only proxy environment of tool-subprocess. Owner: [Runtime Agent](execution/runtime-agent.md).

**Browser-origin allowlist** — The operator-configured list of exact browser origins that Server allows, and the only origins for which credentialed CORS is enabled. Owner: [Public API](api/public-api.md).

**Budget** — A bound on a quantity consumed cumulatively, such as across calls over an allocation's lifetime, across the files of one scan, across the work of one extraction, or summed over the Workers of a selection. Owner: [Limits](platform/limits.md).

**Bundled skill source** — A reviewable package directory under `configs/skills/<name>/` that SkillCatalog uses only to create missing owner Skill artifacts at startup. Owner: [Agent Skills](tools/skills.md).

## C

**Caido client handle (`caido-graphql-client`)** — The one private infrastructure handle that `caido-graphql@1` advertises and every `caido@1` tool requires. Owner: [HTTP and Caido tools](tools/http-caido.md).

**Caido exchange artifact** — The single `application/vnd.contractor.caido-exchange+json` artifact that holds one Caido request together with its response. Owner: [HTTP and Caido tools](tools/http-caido.md).

**Caido GraphQL adapter (`caido-graphql@1`)** — The typed private Runtime adapter that creates the allocation's Caido GraphQL client and is not an HTTP forward proxy. Owner: [HTTP and Caido tools](tools/http-caido.md).

**Caido Toolset (`caido@1`)** — The Toolset whose model-visible tools call the trusted Caido control GraphQL API through static operations. Owner: [HTTP and Caido tools](tools/http-caido.md).

**Candidate** — (1) The StageResult a Planner returns before the Scheduler validates, normalizes and stores it. Owner: [Scheduler](execution/scheduler.md). (2) One possible (Worker, input) pairing in a scan plan, with its selection decision. Owner: [Scanners](tools/scanners.md).

**Candidate batch** — The ordered set of compatible ready items that selection forms before the submission builder applies the item-task-set byte envelope. Owner: [Audit execution](audits/execution.md).

**Candidate compatibility** — The Run-draft layer meaning only that an artifact's stored media type satisfies a Workflow input slot. Owner: [Projects](artifacts/projects.md).

**Canonical request digest** — The digest of the validated create-Run request that Server binds to the authenticated owner and the Idempotency-Key. Owner: [Runs](execution/runs.md).

**`canonical_inventory_digest`** — SHA-256 over a versioned JCS representation of the normalized inventory basis (enumerated paths/operations, their resolved local dependency closures and normalized gap descriptors). Owner: [Audits](audits/audits.md).

**Capability ID** — A versioned string, such as `source.archive@1`, naming a capability that a case requires or a binding claims. Owner: [Portable evaluation format](evals/format.md).

**Capability probe** — A bounded startup check by which one enabled factory proves what it can honor in the process environment. Owner: [Runtime Agent](execution/runtime-agent.md).

**CapabilitySnapshot** — The immutable, positive set of WorkerRuntime, Toolset/tool, SandboxProfile and RuntimeAdapter capabilities a Runtime Agent proves before its first registration. Owner: [Control Plane](execution/control-plane.md).

**Capacity deferral** — Keeping a StageExecution preparing, with no slot reserved, while compatible Runtime capacity is missing. Owner: [Scheduler](execution/scheduler.md).

**Cardinality target** — A bound on how many members a collection or selection may contain, such as active notes per Memory Namespace or skills per AgentTemplate. Owner: [Limits](platform/limits.md).

**Cascade** — The deletion of other entities or records that the deletion of an entity causes. Owner: [Entity lifecycle](platform/lifecycle.md).

**Case** — One evaluation task with its visible inputs, required capabilities and outputs, and private expected data (`playground.case/v2`). Owner: [Portable evaluation format](evals/format.md).

**Catalog of checks** — The extensible set of security checks, whose check-specific behavior lives in versioned Workflow, Planner and AgentTemplate definitions, instructions, Skills and toolsets. Owner: [System overview](overview.md).

**Catalog source fingerprint** — The SHA-256 fingerprint of the immutable catalog source a listing result was built from, to which a continuation cursor is bound. Owner: [Configuration catalog](configuration/catalog.md).

**Check type** — The user-facing name of an AuditProfile. Owner: [Web UI](ui/ui.md).

**Checkpoint** — The effective tree from which the current invocation's diff and rollback are calculated. Owner: [Run workspaces and filesystem tools](tools/workspace.md).

**Cleanup reserve** — The part of a command budget reserved for stopping: the stop grace plus one second, at most half the budget. Owner: [Code execution](tools/code-execution.md).

**Client key** — The invocation-scoped proposal key `client_key`, derived as `call-` plus the SHA-256 of the ADK function-call ID, which never becomes a global finding ID. Owner: [Findings](findings/findings.md).

**Closed instrumentation vocabulary** — The fixed set of Planner span kinds and metadata attributes, which content never enters. Owner: [Observability](platform/observability.md).

**Closed vocabulary** — A model-visible error vocabulary that lists every code its toolset may return. Owner: [Error and stop codes](platform/errors.md).

**Code-analysis session** — The per-allocation state of code-analysis@1, holding only the current snapshot digest and bounded derived state. Owner: [Code analysis](tools/code-analysis.md).

**`code-analysis@1`** — The read-only structural code-analysis Toolset that always analyzes the current effective workspace snapshot through a shallow Tree-sitter surface and a Trailmark graph surface. Owner: [Code analysis](tools/code-analysis.md).

**`code-execution@1`** — The explicitly selected Toolset that exports `exec_command`, which runs a shell program inside the allocation's Podman container. Owner: [Code execution](tools/code-execution.md).

**Collection** — The transition that validates an execution's exact frozen output, retains accepted revisions, commits one AuditCollectionReceipt and records each member's attempt outcome. Owner: [Audit execution](audits/execution.md).

**Collection document** — The exact retained bytes of one proposal or evidence artifact inside a finding collection, identified by `doc-` plus the SHA-256 of its scope, exact ref and content metadata. Owner: [Findings](findings/findings.md).

**Collection entry** — One selected receipt inside a finding collection, with its origin, retention state, proposal document, evidence links and review observations. Owner: [Findings](findings/findings.md).

**Command observation** — The bounded structured result of one completed `exec_command` call: status, exit code, previews, truncation flags, duration and error code. Owner: [Code execution](tools/code-execution.md).

**Command receipt** — The durable record of an accepted experiment command, replayed for the same idempotency key and body. Owner: [Managed Evals](evals/managed-evals.md).

**Comparison** — The report of expected denominators, pairing, completeness, quality and cost of the arms of an experiment (`playground.comparison/v1`). Owner: [Portable evaluation format](evals/format.md).

**Complete injective matching** — An assignment of every logical binding of a Stage to a distinct eligible Runtime Agent, found before anything is reserved. Owner: [Control Plane](execution/control-plane.md).

**Completion boundary** — The one Runtime-owned point, before WorkerCompletion is published, at which a Worker invocation resolves to continue, complete or fail. Owner: [Worker](execution/worker.md).

**Completion contract** — The optional, Server-pinned AllocationSpec.completionContract that carries the contract kind, the exact task and execution-manifest refs and the versionless result binding to Runtime. Owner: [Audit Worker completion](audits/worker-completion.md).

**Completion facts** — The bounded completion diagnostics object carrying contract kind, phase, accepted and total counts, reminder count and failure code. Owner: [Audit Worker completion](audits/worker-completion.md).

**Concrete value** — A value taken from an explicit binding or a document hint; preparation never invents one. Owner: [Scanners](tools/scanners.md).

**Conditional quality** — Quality among fully scored members, reported with its explicit denominator. Owner: [Portable evaluation format](evals/format.md).

**Configuration namespace** — The single logical namespace formed by the operator root and the managed root together, with no root precedence or override. Owner: [Configuration catalog](configuration/catalog.md).

**Configuration set** — The complete validated and resolved in-memory configuration that the Server serves. Owner: [Configuration catalog](configuration/catalog.md).

**Confirmed control lease** — The liveness lease that advances only through a new heartbeat request, acknowledgement and echo round trip. Owner: [Control Plane](execution/control-plane.md).

**Content capture** — The opt-in captureContent=true mode in which prompts, responses and tool arguments and results are exported, unredacted, to the telemetry sink. Owner: [Observability](platform/observability.md).

**Context artifact** — A Stage-local name bound in `context.artifacts` to a versionless RunScope `(namespace, name)` with a `required` flag. Owner: [Workflows](configuration/workflows.md).

**Context boundary** — The prompt-token boundary `min(floor(contextWindowTokens * contextWindowRatio), contextWindowTokens - maxOutputTokens)` at which an enabled summarizer is requested. Owner: [Worker](execution/worker.md).

**Context compaction** — Rewriting older conversation events so the same Worker can continue using tools, which Contractor does not enable. Owner: [Worker](execution/worker.md).

**Contractor v2** — An Application Security orchestration platform in which Workflows define reusable security checks and supporting analyses, and Audits coordinate WorkflowRuns into Project-bound assessments with shared scope, evidence, findings, coverage and review. Owner: [System overview](overview.md).

**Contractor-owned Worker State** — The reserved, versioned and bounded contractor subtree of a Worker's ADK State, the only State surface Runtime exports ("full state"). Owner: [Worker](execution/worker.md).

**`contractor-skill`** — The repository authoring helper that writes the canonical skill ZIP representation and validates a package before upload. Owner: [Agent Skills](tools/skills.md).

**Control mode** — The experiment's immutable controlMode, server for a native experiment or external for one driven by an external producer. Owner: [Managed Evals](evals/managed-evals.md).

**Control Plane** — The Server component that owns Runtime Agent registration, availability, placement and allocation lifecycle. Owner: [Control Plane](execution/control-plane.md).

**Control Plane URI SAN prefix** — The reserved prefix `urn:contractor:control-plane:` whose presence in a URI SAN, followed by a non-empty instance identifier, marks a certificate as a Control Plane certificate. Owner: [Security](platform/security.md).

**Controlled scope purge** — The trusted lifecycle cleanup that removes a complete RunScope or ProjectScope and the lineage edges owned by it, and the only exception to the create-only, immutable Artifact APIs. Owner: [Retention and deletion](artifacts/retention-and-deletion.md).

**Convert workflow** — A Caido workflow that `caido_workflow_run` runs on non-empty UTF-8 `input`. Owner: [HTTP and Caido tools](tools/http-caido.md).

**Coordination boundary** — The allocation-owned workspace operation lock that serializes Workspace API reads, mutations and snapshot acquisition. Owner: [Run workspaces and filesystem tools](tools/workspace.md).

**Coverage (code analysis)** — The bounded metadata in every successful code-analysis response that counts what was analyzed and states whether and why the answer may be incomplete. Owner: [Code analysis](tools/code-analysis.md).

**Creation facade** — One of the Toolsets security-findings@1, security-findings-code@1 and security-findings-http@1, each exporting the model-visible tool `finding` with its own required arguments. Owner: [Findings](findings/findings.md).

**Credential manager** — The optional Gateway declaration (credentialManager, e.g. litellm-virtual-keys@1) through which Server Operations creates and deletes LLM credentials. Owner: [Runtime configuration, labels, credentials and Gateways](configuration/runtime-config.md).

**Credential-encryption master key** — The 256-bit bootstrap key, kept outside PostgreSQL, that encrypts stored credential secrets. Owner: [Security](platform/security.md).

**Credential-lifecycle barrier** — The lock shared by Run initialization, binding and Agent-label mutation, allocation config pinning and credential deletion so that no Run pins a key being deleted. Owner: [Runtime configuration, labels, credentials and Gateways](configuration/runtime-config.md).

**CSRF token** — The session-bound 32-byte random token that a cookie-authenticated non-safe request presents in X-CSRF-Token. Owner: [Public API](api/public-api.md).

**Cumulative budget** — The optional soft boundary on the summed provider-reported total tokens of completed normal-loop model calls in one A2A invocation. Owner: [Worker](execution/worker.md).

**Current subtask** — The first unresolved subtask of the plan, named by currentSubtaskId. Owner: [Planners](execution/planners.md).

## D

**Dataset revision** — An immutable set of exact visible case definitions with a source identity, revision and digest. Owner: [Managed Evals](evals/managed-evals.md).

**Default** — The value a configurable bound takes when its setting is omitted, lying within the setting's allowed values. Owner: [Limits](platform/limits.md).

**Default binding** — The reserved default label binding used as the base layer for every Run, including a Run with no labels. Owner: [Runtime configuration, labels, credentials and Gateways](configuration/runtime-config.md).

**`deletable`** — The Server-derived boolean on Run list and detail responses that marks the safe deletion boundary at which the UI shows a trash action. Owner: [Retention and deletion](artifacts/retention-and-deletion.md).

**`deleting`** — The Project lifecycle state entered atomically when deletion is accepted, in which Project metadata and target changes, artifact writes and new Project Runs are rejected while cleanup proceeds. Owner: [Retention and deletion](artifacts/retention-and-deletion.md).

**Deletion blocker** — A reference, a state or a prerequisite that prevents an entity from being deleted while it holds. Owner: [Entity lifecycle](platform/lifecycle.md).

**Deletion fence** — The durable record that blocks new submissions and assessments for an experiment or Project that is being deleted. Owner: [Managed Evals](evals/managed-evals.md).

**Deletion intent** — The dedicated deletion_requested_at marker that records that owner or Project deletion requested an Audit's deletion. Owner: [Review and reports](audits/review-and-reports.md).

**Deployment setting** — A durable, explicitly editable Operations resource that applies to the whole deployment rather than to an owner, Project or Run. Owner: [Operations](operations/operations.md).

**Deployment trust domain** — The Server and every Runtime Agent, rooted in one deployment-owned CA and connected by mutual TLS on the private control protocol. Owner: [Security](platform/security.md).

**Deterministic inventory** — An inventory generator, such as standard-mappings@1 or openapi-operations@1, that produces a finite subject and item list without network side effects. Owner: [Audits](audits/audits.md).

**Deterministic publication** — The phase in which the common finalizer's Audit strategy builds the check-results package from the sealed collection and writes it to the Server-declared result binding without the model. Owner: [Audit Worker completion](audits/worker-completion.md).

**Diagnostic listener** — The separate loopback-only HTTP listener that serves an allowlist of Go profiles when --pprof is enabled. Owner: [Operations](operations/operations.md).

**Direct transport** — The Runtime-owned HTTP transport that `http-tools@1` uses when no `tool-http` route is resolved. Owner: [HTTP and Caido tools](tools/http-caido.md).

**Direct verification** — Discovery and verification of a finding inside the same Run, without consuming an earlier hypothesis. Owner: [Review and reports](audits/review-and-reports.md).

**Direct verification reference** — The retained record of a direct verification: source Run and exact Workflow closure, result artifact revision/digest, evidence refs, contract revision/digest, semantic assessment and acceptance record. Owner: [Review and reports](audits/review-and-reports.md).

**Disclosure budget** — The 16 MiB allocation-lifetime budget that `list_skills`, `load_skill` and `load_skill_resource` share and that Runtime reserves before native dispatch. Owner: [Agent Skills](tools/skills.md).

**Dispatch hold** — A credential usage hold that Audit start adds for future child Runs and that ends once dispatch is durably closed and all submission intents are resolved. Owner: [Audits](audits/audits.md).

**Dispatch look-ahead** — The Server-owned per-Audit window, equal to the current global maxConcurrentRuns, that bounds an Audit's outstanding count. Owner: [Audit execution](audits/execution.md).

**Displaced generation** — A previously attached filesystem object whose blob now points at a replacement upload candidate; it is not eagerly unlinked. Owner: [Artifact blob storage](artifacts/blob-storage.md).

**Draft** — (1) A clone of a published configuration whose typed fields are edited in the Operations authoring flow; it is not executable configuration. Owner: [Configuration catalog](configuration/catalog.md). (2) The mutable native setup of name, exact dataset and case selection, A/B bindings, input and parameter mapping, evaluation policy, repetitions, order and budgets. Owner: [Managed Evals](evals/managed-evals.md).

**`draining`** — The slot state in which the Runtime rejects new A2A work while its Worker stops and after it has exited; the allocation remains reserved until the private release after its terminal StageExecution, or local lease loss, moves the slot to fenced. Owner: [Runtime Agent](execution/runtime-agent.md).

**Duplicate link** — An acyclic, human-decided link inside one Audit from a finding to its canonical target. Owner: [Review and reports](audits/review-and-reports.md).

## E

**Effective deadline** — The minimum of a command's requested timeout, the operator policy and the supplied execution deadline. Owner: [Code execution](tools/code-execution.md).

**Eligibility** — The planned status of a member: `eligible`, `unsupported` or `blocked`. Owner: [Portable evaluation format](evals/format.md).

**End-to-end pass** — Members that are eligible, succeeded, have complete required outputs and a passing assessment, counted once per member and divided by expected members; conditional quality instead carries its own, explicitly named denominator. Owner: [Portable evaluation format](evals/format.md).

**Enforcement point** — A component that checks a bound, such as the Server before scheduling or the Runtime for one allocation. Owner: [Limits](platform/limits.md).

**Entry Stage** — The Stage named by `entryStage`, where the Workflow graph starts. Owner: [Workflows](configuration/workflows.md).

**Error code** — A stable, bounded identifier of one failure class that a component reports to a model, an API caller or the Scheduler. Owner: [Error and stop codes](platform/errors.md).

**Escalation variant** — The ResolvedExecutionConfig of an escalated attempt, resolved and pinned at Run initialization so the Scheduler only selects it. Owner: [Agent templates](configuration/agent-templates.md).

**EvalCoordinator** — The Go Server component that dispatches, pauses, resumes and cancels native experiments under a durable claim. Owner: [Managed Evals](evals/managed-evals.md).

**EvalStore** — The component that persists private eval records, expected membership, receipts and Project deletion fences in Contractor's PostgreSQL database. Owner: [Managed Evals](evals/managed-evals.md).

**Evaluation label convention** — The eval subsystem's use of the generic `purpose=eval` and `eval.*` WorkflowRun labels for correlation and discovery. Owner: [Managed Evals](evals/managed-evals.md).

**Event cursor** — An opaque generation combined with an unsigned decimal string sequence that identifies a position in one live event stream. Owner: [Public API](api/public-api.md).

**Evidence budget** — `maxEvidenceBytes`, the Audit's bound on retained evidence bytes, which charges each distinct retained revision once. Owner: [Audit execution](audits/execution.md).

**Evidence contract** — An independently versioned part of a standard package that defines allowed assessments, evidence kinds and counts, rationale and human-review policy. Owner: [Audits](audits/audits.md).

**Exact-subject review ledger** — The shared mechanism for human decisions on Audit items, findings and reports, in which a review request pins an immutable subject revision/digest and a decision is accepted only through the owner-authenticated Audit API with CAS and idempotency. Owner: [Review and reports](audits/review-and-reports.md).

**`execution`** — The AgentTemplate block of a tool Worker that names the tool, its argument bindings, its report binding and its deadline. Owner: [Scanners](tools/scanners.md).

**Execution core** — The generic machinery in which the Workflow Scheduler manages Run and Stage lifecycles without branching on security-check types. Owner: [System overview](overview.md).

**Execution function** — `execute_current_subtask`, the Planner-specific function that dispatches the current subtask to a Worker. Owner: [Planners](execution/planners.md).

**Execution handle** — The private, allocation-owned sandbox channel through which selected execution tools reach the container, distinct from the host `runtime-subprocess-launcher` channel. Owner: [Code execution](tools/code-execution.md).

**Execution key** — The SHA-256 of canonical JSON of runId, stageExecutionId, logicalAgentName and subtaskId that identifies one tool invocation. Owner: [Scanners](tools/scanners.md).

**Execution lane** — One Scheduler loop that holds at most one Run claim while performing one bounded progression attempt. Owner: [Scheduler](execution/scheduler.md).

**Execution manifest** — The Controller-generated, immutable, ordered list of item keys and exact task and input refs of one execution. Owner: [Audits](audits/audits.md).

**Execution projection** — The visible part of a case, `{id, task, inputs, requires, outputs}`, which is the only part of a case that crosses the adapter boundary. Owner: [Portable evaluation format](evals/format.md).

**Execution report** — The framework-neutral ExecutionReport that a Planner or an in-process Worker runtime produces, with aggregate metrics, tool-call records and errors. Owner: [Observability](platform/observability.md).

**Execution success** — Succeeded members divided by expected members, reported per arm. Owner: [Portable evaluation format](evals/format.md).

**Execution-selection leaf** — An object with the optional fields `modelPolicy`, `llmGateway` and `credential`, at least one of which must be present. Owner: [Agent templates](configuration/agent-templates.md).

**ExecutionConfig / `executionConfig`** — (1) **ExecutionConfig**: The configuration kind of an immutable, Stage-local escalation override profile with optional `planner` and `agents` selections. (2) **`executionConfig`**: Reference-only selections of ModelPolicy, LLMGatewayConfig and credential for Planner and Worker consumers, given as Workflow defaults, a Run override or an escalation override. Owner: [Agent templates](configuration/agent-templates.md).

**ExecutionConfigRef** — The exact `(name, version)` of an ExecutionConfig plus the SHA-256 digest of its normalized RFC 8785 JCS manifest. Owner: [Agent templates](configuration/agent-templates.md).

**ExecutionRef** — The common opaque reference `{provider, connection_id, kind, id}` to one provider execution. Owner: [Portable evaluation format](evals/format.md).

**Expected denominator** — The complete expected member set of a plan, including unsupported and blocked members. Owner: [Portable evaluation format](evals/format.md).

**Experiment** — One evaluation invocation with its requested suites, variants, repetitions, order, budgets, comparison and publication (`playground.experiment/v1`). Owner: [Portable evaluation format](evals/format.md).

**Experiment bundle** — The local evaluator authority of one experiment: its immutable plan and records and an fsynced journal under an exclusive process lock. Owner: [Portable evaluation format](evals/format.md).

**Exported tool** — A tool that one exact Toolset version exports and that an AgentTemplate may select. Owner: [Toolsets](tools/toolsets.md).

**External producer** — An authorized client that registers and drives an external experiment through the public API and submits attributed results and assessments. Owner: [Managed Evals](evals/managed-evals.md).

**External registration** — The frozen external setup of a schema-versioned safe plan projection, the exact source plan digest, explicit comparison policy and budgets, and exact executable bindings and mappings for managed submissions. Owner: [Managed Evals](evals/managed-evals.md).

## F

**FactoryRegistry** — The Runtime Agent's enabled local set of WorkerRuntime, Toolset, SandboxProfile and RuntimeAdapter factories. Owner: [Runtime Agent](execution/runtime-agent.md).

**Failure signature** — An exact provider-response match that an LLM Gateway declares in `failureSignatures`, either a `modelUnavailable` entry or a `permanentCodes` code; a Gateway's failure signatures are the optional provider-specific part of model recovery classification. Owner: [Failure model](platform/failure-model.md).

**`fenced`** — The slot state in which the Runtime rejects all work and keeps the old allocation identity until release is authoritatively acknowledged. Owner: [Runtime Agent](execution/runtime-agent.md).

**Final allocation report** — see **AllocationFinalReport**.

**Finalization** — The bounded finalizing path, identified by finalization_id, that drains Workers and collects reports after a durable candidate. Owner: [Scheduler](execution/scheduler.md).

**Finding collection** — A frozen selection of existing proposal receipts and available review observations, published as one immutable application/vnd.contractor.findings-collection+zip package and consumed as an ordinary input artifact. Owner: [Findings](findings/findings.md).

**Finding fingerprint** — A grouping hint combining scope snapshot, subject and normalized issue family that never merges finding identities. Owner: [Review and reports](audits/review-and-reports.md).

**FindingProposal** — The immutable candidate and evidence envelope, under schema contractor.audit.finding-proposal.v1, that a Worker submits through `finding` to describe an observation, a hypothesis or a finding with evidence, without confirming a vulnerability. Owner: [Findings](findings/findings.md).

**Fork** — A new independent binding with a lineage edge to an exact source version, which may reuse the same immutable blob. Owner: [Artifact store](artifacts/artifact-store.md).

**Format match** — see **Format-only suggestion**.

**Format-only suggestion** — An automatically prepopulated, MIME-based Artifact candidate for a Workflow input slot whose exact revision the user has not yet confirmed; it requires user review and does not prove semantic suitability. Owner: [Web UI](ui/ui.md).

**Frame** — One 15-second in-memory record of Server performance measurements. Owner: [Operations](operations/operations.md).

**Frozen binding** — A binding that can no longer advance: for example a Run `outputs` binding once the Run is terminal (except through manual continuation of an eligible failed Run), or an Audit-managed `audit-*` Project binding, which is created frozen. Owner: [Artifact store](artifacts/artifact-store.md).

**Frozen contract artifact** — The Audit artifact created on acceptance of a direct verification that freezes the ResolvedWorkflow snapshot, its closure digest, the selected output declaration, the result schema and the identity/evidence binding rule. Owner: [Review and reports](audits/review-and-reports.md).

**Frozen report candidate** — The exact machine/summary report candidate that finalization freezes, with its acceptance request, when a human report policy is selected. Owner: [Review and reports](audits/review-and-reports.md).

## G

**Gateway policy** — The immutable key policy (gatewayPolicy: ModelPolicyRefs, budget and rate limits) sent to LiteLLM when an LLM credential is created. Owner: [Runtime configuration, labels, credentials and Gateways](configuration/runtime-config.md).

**Generated inventory** — An inventory built in the inventory phase from accepted preparation outputs instead of from the original baseline. Owner: [Audit preparation and OpenAPI scans](audits/scans.md).

**Git admission** — The per-process slot an active Git import holds from before fetch through its metadata response. Owner: [Git import](artifacts/git-import.md).

**Git import** — Creating an ordinary exact source ZIP artifact from one resolved commit of a remote Git repository. Owner: [Git import](artifacts/git-import.md).

**Git remote allowlist** — The operator-configured list of exact host:port pairs that outbound Git imports may connect to; empty by default, which disables Git network import. Owner: [Git import](artifacts/git-import.md).

**Git-source record** — The immutable provenance record (normalized repository URL, requested ref, full resolved commit object ID, Server import timestamp) attached to an imported Artifact version. Owner: [Git import](artifacts/git-import.md).

**`gitSource`** — The optional field of authorized Artifact metadata and version responses that exposes a version's Git-source record; its absence means an ordinary upload. Owner: [Git import](artifacts/git-import.md).

**Graph surface** — The nine code-analysis@1 operations backed by Trailmark in an allocation-local child process and advertised only by local workspace Runtimes. Owner: [Code analysis](tools/code-analysis.md).

## H

**Host guardian** — The trusted host component, independent of the Runtime process, that holds the container's cgroup scope, enforces the liveness deadline and performs completion checks. Owner: [Code execution](tools/code-execution.md).

**HTTP session** — The allocation-memory cookies, default headers, authentication and history of `http-tools@1`. Owner: [HTTP and Caido tools](tools/http-caido.md).

**HTTP Toolset (`http-tools@1`)** — The Toolset that sends model-described HTTP requests to application targets and keeps an allocation-scoped HTTP session. Owner: [HTTP and Caido tools](tools/http-caido.md).

**Human diff** — The deterministic `text/x-diff` unified diff exported for analysis and review, not for authoritative reconstruction. Owner: [Run workspaces and filesystem tools](tools/workspace.md).

## I

**`in_sync`** — The bundled-initialization outcome for an existing owner Skill binding with the identical ZIP digest and canonical Skill media type; the binding is not advanced. Owner: [Agent Skills](tools/skills.md).

**Inbox** — The Web UI destination at / that lists decisions first, then blocked work, finished results and running checks. Owner: [Web UI](ui/ui.md).

**Incidental classification** — A known CWE@4.20 reference retained on an imported proposal without pinning CWE as an Audit standard, which never satisfies an assigned entry. Owner: [Audits](audits/audits.md).

**Incomplete placeholder** — The report with unknown fields that Scheduler records for an expected participant that started but produced no report. Owner: [Observability](platform/observability.md).

**Inert PID 1** — The separate subordinate-UID init process of the approved image that reaps orphans inside the container. Owner: [Code execution](tools/code-execution.md).

**Infrastructure channel** — A class of allocation-owned handle supplied by Runtime that an exported tool declares it consumes: `runtime-http-client`, `runtime-subprocess-launcher` or `caido-graphql-client`. Owner: [Toolsets](tools/toolsets.md).

**`initial-inventory` link** — The protected Artifact link that pins the derived inventory snapshot of a prepared Audit and commits with its first Round. Owner: [Audit preparation and OpenAPI scans](audits/scans.md).

**Input digest** — The digest of the template ref, resolved arguments and output bindings of one tool invocation. Owner: [Scanners](tools/scanners.md).

**Input slot / output slot** — A named public artifact slot of a Workflow, declared in `spec.inputs` or `spec.outputs` with a required flag and accepted media types. Owner: [Workflows](configuration/workflows.md).

**Insecure loopback mode** — The explicit loopback-development mode enabled by `--insecure-loopback-cookie` that uses the separately named `contractor_loopback_session` cookie on an IP-literal loopback listener. Owner: [Security](platform/security.md).

**`instance_id`** — The fresh random identity a Runtime Agent process generates at start, used for correlation and routing and never reused after the process restarts. Owner: [Control Plane](execution/control-plane.md).

**Instruction ref** — The normalized configuration-root-relative path of an instruction resource, such as instructions/openapi/planner.md. Owner: [Configuration catalog](configuration/catalog.md).

**Instruction resource** — A UTF-8 text resource below instructions/, identified by its normalized logical path rather than by a lookup key. Owner: [Configuration catalog](configuration/catalog.md).

**Insufficient compatible capacity** — The temporary placement result returned when no complete matching currently exists, leaving the StageExecution in preparing with no slot reserved. Owner: [Control Plane](execution/control-plane.md).

**Internal helper** — Code used by a selected tool that is not itself a model-visible tool and need not appear in the allowlist. Owner: [Toolsets](tools/toolsets.md).

**Invocation receipt** — The Runtime-owned `tool-invocation.<hex-key>` artifact that records the phase of one tool invocation. Owner: [Scanners](tools/scanners.md).

**Item origin** — The immutable record on an AuditItem of the exact source it was generated from: source ref and content digest, canonical inventory digest, entry key and optional entry version. Owner: [Audits](audits/audits.md).

**Item revision** — The invocation-local positive revision of one accepted item, starting at 1. Owner: [Audit Worker completion](audits/worker-completion.md).

**Item-action subject** — The subject of an item review, a hash over the Audit and item identities, the pinned profile digest, the Workflow role and the exact task artifact. Owner: [Review and reports](audits/review-and-reports.md).

## J

**Job** — A selected candidate, executed as one tool Worker invocation. Owner: [Scanners](tools/scanners.md).

## K

**Keep remaining time** — The time-limit option, default for ordinary paused Audits, that omits the API override so the Audit keeps its remaining allowance or its no-limit mode. Owner: [UI journeys](ui/journeys.md).

**Key generation** — The identity of one stored value of an owner Git key; an admitted import captures one and may finish with it. Owner: [Git import](artifacts/git-import.md).

**`key_id`** — see **Master-key fingerprint**.

## L

**Label binding** — The mutable, revisioned Control Plane pointer from one label to one exact RuntimeConfig version. Owner: [Runtime configuration, labels, credentials and Gateways](configuration/runtime-config.md).

**`lean@1`** — The fixed, Runtime-authored compact observation projection carried in WorkerResult.observations. Owner: [Worker](execution/worker.md).

**Lease session** — The dedicated PostgreSQL connection that holds the advisory lease and requires direct PostgreSQL or session pooling. Owner: [Persistence](platform/persistence.md).

**Lease-session loss** — A single liveness probe of the lease session that fails within the tolerance window, upon which the active Server closes its listeners and cancels the Scheduler. Owner: [Persistence](platform/persistence.md).

**Legacy labelled Run** — An ordinary WorkflowRun grouped only by evaluation labels, outside any managed experiment's verified membership. Owner: [Managed Evals](evals/managed-evals.md).

**Library** — The user-facing name of the Catalog, which groups reusable execution definitions. Owner: [Web UI](ui/ui.md).

**Lifecycle entry** — The one registry row `LIFE-<area>-<n>` for an entity, where `<area>` is the requirement prefix of the owning document and `<n>` numbers the entries that document owns. Owner: [Entity lifecycle](platform/lifecycle.md).

**Lineage edge** — A recorded provenance link from a binding to the exact source version it was forked from, for example of kind `output_bind` or `project_output_publish`. Owner: [Artifact store](artifacts/artifact-store.md).

**Live event WebSocket** — The read-only GET /v1/events/ws endpoint, negotiated with subprotocol contractor.events.v1, that multiplexes run and operations subscriptions. Owner: [Public API](api/public-api.md).

**LLM credential** — An immutable encrypted Gateway token bound to one exact LLMGatewayConfig and selected through LLMCredentialRef; a row that exists is active. Owner: [Runtime configuration, labels, credentials and Gateways](configuration/runtime-config.md).

**LLM Gateway** — The backend-neutral Contractor role of the endpoint that serves model calls; LiteLLM is the initial backend, not a required component. Owner: [Runtime configuration, labels, credentials and Gateways](configuration/runtime-config.md).

**LLMGatewayConfig** — An immutable, published description of one OpenAI-compatible Gateway endpoint (protocol, URL, optional credential manager and failure signatures), kept separate from ModelPolicy. Owner: [Runtime configuration, labels, credentials and Gateways](configuration/runtime-config.md).

**Local discard** — A span that the Worker exporter itself drops, counted once under one droppedSpans reason. Owner: [Observability](platform/observability.md).

**Local principal** — The single user principal the Server authenticates, owning both ordinary user and Operations capabilities. Owner: [Security](platform/security.md).

**Local watchdog** — The Runtime Agent's monotonic timer over the confirmed control lease. Owner: [Runtime Agent](execution/runtime-agent.md).

**Local-auth file** — The bootstrap YAML file named by `--local-auth-file` that defines the local principal's userId, username and Argon2id password hash. Owner: [Security](platform/security.md).

**Local/artifact-only tool** — An exported tool that declares no infrastructure channel. Owner: [Toolsets](tools/toolsets.md).

**Logical Agent name** — The key of a Stage `agents` mapping: the stable Worker name visible to Planner, not a physical Runtime Agent or process identity. Owner: [Workflows](configuration/workflows.md).

**Logical skill ref** — A versionless `ArtifactRef` in Namespace `skills` declared by an AgentTemplate, resolved in the Run owner's UserScope at Run initialization. Owner: [Agent Skills](tools/skills.md).

**Lookup key** — The identity (kind, metadata.name, metadata.version) of a manifest; file name and nesting are organizational only. Owner: [Configuration catalog](configuration/catalog.md).

**Lower and upper** — In overlay mode, the unchanged hydrated base (lower) and the layer of canonical text writes, explicit directories and tombstones (upper). Owner: [Run workspaces and filesystem tools](tools/workspace.md).

## M

**Main Worker model** — The tool-using ADK Worker Agent that runs AgentTemplate behavior over one task and ends with terminal Worker text. Owner: [Worker](execution/worker.md).

**Maintenance loop** — A bounded, dedicated loop for safety and recovery maintenance that consumes no execution lane. Owner: [Scheduler](execution/scheduler.md).

**Managed Eval experiment** — An explicit, owner-scoped comparison of a baseline and a candidate arm inside an evaluation Project, with one frozen plan or registration, a fixed set of expected members and an authority revision. Owner: [Managed Evals](evals/managed-evals.md).

**Managed execution supervisor** — The approved supervisor that runs commands through the managed execution protocol, accounts for command descendants and confirms their termination. Owner: [Code execution](tools/code-execution.md).

**Managed publication** — The atomic, create-only addition of a new immutable ModelPolicy or LLMGatewayConfig version to the managed root and the configuration set. Owner: [Configuration catalog](configuration/catalog.md).

**Managed root** — The Server-managed configuration root that receives UI publications, configured by --managed-config-root / CONTRACTOR_MANAGED_CONFIG_ROOT. Owner: [Configuration catalog](configuration/catalog.md).

**Managed text file** — A workspace file that is strict UTF-8 with no NUL; the managed-text projection is the set of such files. Owner: [Run workspaces and filesystem tools](tools/workspace.md).

**Manifest** — One YAML configuration document with the envelope apiVersion, kind, metadata and spec. Owner: [Configuration catalog](configuration/catalog.md).

**Manual continuation** — The owner's explicit continuation of an eligible failed Run from its latest failed or interrupted StageExecution through a new attempt of the same Stage. Owner: [Runs](execution/runs.md).

**Manual-continuation receipt** — The durable record of one manual continuation, unique per source execution. Owner: [Runs](execution/runs.md).

**Mapping source** — The kind of a mapping reference: audit-input, prepare-output, item-package, execution-manifest or retained-output. Owner: [Audit preparation and OpenAPI scans](audits/scans.md).

**Master-key fingerprint (`key_id`)** — The non-secret lowercase `sha256:<hex>` digest of the decoded master key stored beside each ciphertext. Owner: [Security](platform/security.md).

**`maxSubmittedRuns`** — The pinned cumulative Audit budget over every child Run attempt, including retries and later rounds. Owner: [Audit execution](audits/execution.md).

**Measured interval** — The window, from accepting allocation ownership to just before the cached finalize or abort response is built, during which the Runtime samples allocation resource metrics. Owner: [Runtime Agent](execution/runtime-agent.md).

**Member** — One expected execution of one case, sample and variant in a plan, identified by the SHA-256 of `[experiment_id, suite_id, case_id, sample, variant_id]`; managed Evals use the same identity. Owner: [Portable evaluation format](evals/format.md).

**Memory Namespace** — The logical view (run_id, resolved Agent Namespace) that holds the notes shared by a Planner and its logical Worker; it is not another ArtifactStore scope. Owner: [Memory](tools/memory.md).

**Memory note** — One named entry of a Memory Namespace, stored as one immutable-versioned RunScope artifact named `memory.<note name>`. Owner: [Memory](tools/memory.md).

**MemoryNote** — The full model-visible note projection with exactly name, content, description, tags, ordinal, created_at and updated_at. Owner: [Memory](tools/memory.md).

**MemoryTools (`memory-tools@1`)** — The built-in shared-note Toolset whose six operations form a purpose-specific, namespace-bound model-tool wrapper over RunScope artifacts. Owner: [Memory](tools/memory.md).

**Merge unit** — A leaf or atomic block of RuntimeConfig that a higher precedence layer replaces as a whole and on which same-layer labels conflict. Owner: [Runtime configuration, labels, credentials and Gateways](configuration/runtime-config.md).

**Metadata labels** — The immutable `labels` key/value string map that identifies, groups and queries one WorkflowRun without affecting its execution. Owner: [Runs](execution/runs.md).

**Metadata-only instrumentation** — Instrumentation that exports no content whatever the content-capture setting, used for a whole allocation whose Worker selects a sensitive-output tool. Owner: [Observability](platform/observability.md).

**Minute record** — One row of performance_minutes holding the minute aggregates of one Server generation. Owner: [Operations](operations/operations.md).

**Mirror** — The uniquely named copy of a snapshot's admitted managed UTF-8 files below the allocation's private scratch directory, from which the Trailmark child builds its graph. Owner: [Code analysis](tools/code-analysis.md).

**Missing-content error** — The explicit non-success Artifact error returned when referenced content is missing or corrupt, never an empty successful response. Owner: [Artifact blob storage](artifacts/blob-storage.md).

**Model authority boundary** — AgentTemplate tool selection, which alone decides which tools the model can see and call; labels, grants and Skills add none. Owner: [Toolsets](tools/toolsets.md).

**Model consumer** — A modeled user of a ModelPolicy (an `adk@1` Worker, a Worker summarizer, or a `streamline@1` or `router@1` Planner) that declares which ModelPolicy fields it requires. Owner: [Agent templates](configuration/agent-templates.md).

**Model-backed Planner** — A streamline@1 or router@1 Planner, which runs one Google ADK Go LlmAgent inside Server. Owner: [Planners](execution/planners.md).

**Model-visible error vocabulary** — The set of error codes a toolset returns to the model. Owner: [Error and stop codes](platform/errors.md).

**`model_unavailable`** — The transient classification of a provider response that exactly matches a modelUnavailable failure signature. Owner: [Failure model](platform/failure-model.md).

**ModelPolicy** — The shared, immutable model-loop policy of a Planner or Worker: an opaque Gateway model alias plus portable generation and cumulative execution-safety fields. Owner: [Agent templates](configuration/agent-templates.md).

## N

**Namespace** — The addressing component that also fixes the scope of domain tools; it is unique only within its scope and is not a security boundary against granted current-Run tools. Owner: [Artifact store](artifacts/artifact-store.md).

**Native experiment** — An experiment in server control mode, authored in Contractor, prepared into a frozen plan and dispatched by the EvalCoordinator. Owner: [Managed Evals](evals/managed-evals.md).

**Navigation rail** — The Web UI's top-level navigation, arranged in main, secondary and bottom groups. Owner: [Web UI](ui/ui.md).

**Next-Round builder** — The Controller logic that admits unscheduled proposal checks from held finding receipts into a next Round. Owner: [Audit execution](audits/execution.md).

**Node trust boundary** — The boundary the deployment trust domain draws around Contractor processes, which authenticates nodes and not users. Owner: [Security](platform/security.md).

**Normalizer** — An allowlisted, versioned local function with a pinned implementation digest that decodes or projects observed provider data into logical roles and evidence. Owner: [Portable evaluation format](evals/format.md).

**Note preview** — The model-visible note projection without content, returned by list_memories and search_memory. Owner: [Memory](tools/memory.md).

## O

**ObservationExtractor** — A narrow per-tool converter that turns one completed tool call into a typed, bounded, content-free observation fact. Owner: [Worker](execution/worker.md).

**Observed state** — What a Runtime Agent reports about its slot (idle, allocated, draining or fenced), as distinct from the Control Plane's authoritative allocation record. Owner: [Control Plane](execution/control-plane.md).

**Offline cleanup command** — The operator-only command, dry-run by default, that removes physical orphans and abandoned staging files while every writer using the registry and blob root is stopped. Owner: [Artifact blob storage](artifacts/blob-storage.md).

**Opaque leaf** — An unsupported on-disk entry of a local direct workspace that complete acquisition lists but never opens, reads, follows or interprets. Owner: [Run workspaces and filesystem tools](tools/workspace.md).

**OpenAPI scan inventory builder** — The built-in, versioned OpenAPI-to-scan-task builder registered as openapi-scans@1. Owner: [Audit preparation and OpenAPI scans](audits/scans.md).

**`openapi-scan@1` task contract** — The execution contract that the openapi-scans@1 builder declares and that a consuming Workflow's executor Stage implements. Owner: [Audit preparation and OpenAPI scans](audits/scans.md).

**`operation_key`** — SHA-256 over a domain-separated canonical tuple of canonical_inventory_digest, the exact path template and the lowercase HTTP method. Owner: [Audits](audits/audits.md).

**Operations invalidation** — A typed, process-local notification on the operations stream that a Runtime Agent, allocation, configuration, credential or scheduler-settings resource changed. Owner: [Public API](api/public-api.md).

**Operations snapshot** — The current-state REST view of the in-process Control Plane registry, identified by a process generation and a revision. Owner: [Operations](operations/operations.md).

**Operations surface** — The authenticated Operations area of the API and UI: a read-only observation of Runtime Agents and allocations plus explicitly editable deployment settings and infrastructure configuration resources. Owner: [Operations](operations/operations.md).

**OperationTarget** — The in-process fixed URL target that scanplan.PrepareOperationTarget produces for Nuclei. Owner: [Audit preparation and OpenAPI scans](audits/scans.md).

**Operator** — The scenario role that monitors execution and resources, changes settings and resolves infrastructure issues that cause waits or failures; it is not an authorization role. Owner: [UI journeys](ui/journeys.md).

**Operator root** — The operator/bootstrap configuration root, normally read-only, configured by --operator-config-root / CONTRACTOR_OPERATOR_CONFIG_ROOT. Owner: [Configuration catalog](configuration/catalog.md).

**Ordinal** — The wrapper-assigned, monotonically increasing creation order of a note within its Memory Namespace, starting at 0. Owner: [Memory](tools/memory.md).

**Outcome** — The `on` key a Stage attempt selects: `succeeded` or `failed` from a StageResult, `interrupted` from a StageTermination. Owner: [Workflows](configuration/workflows.md).

**Outcome class** — The kind of result a failed external call or mutation produced: transient, permanent, ambiguous, timeout or cancelled. Owner: [Failure model](platform/failure-model.md).

**Outgoing exchange store** — The allocation memory store of the latest outgoing exchanges from which a finding snapshots selected HTTP evidence. Owner: [HTTP and Caido tools](tools/http-caido.md).

**Outstanding count** — An Audit's reserved submission intents plus its associated nonterminal child Runs. Owner: [Audit execution](audits/execution.md).

**Owner Git key** — The single optional, owner-scoped Git SSH private key used for SSH imports. Owner: [Git import](artifacts/git-import.md).

**Owner marker** — The sibling file the Runtime writes before creating an allocation directory and by which it recognizes that directory as its own. Owner: [Runtime Agent](execution/runtime-agent.md).

**Owner queue pause** — The owner-scoped durable admission gate, stored as OwnerQueueControl, that stops new normal Stage admission for one owner. Owner: [Scheduler](execution/scheduler.md).

**Owning requirement** — The one requirement that states a bound's value and its rule. Owner: [Limits](platform/limits.md).

## P

**Package digest** — Lowercase `sha256:<64 hex>` computed over the exact ZIP payload bytes of a skill package. Owner: [Agent Skills](tools/skills.md).

**Package validator code** — One of the six Agent Skill package classification codes: skill_archive_invalid, skill_path_invalid, skill_member_forbidden, skill_manifest_invalid, skill_name_mismatch and skill_limit_exceeded. Owner: [Error and stop codes](platform/errors.md).

**Pair** — The baseline and candidate members with the same `(suite_id, case_id, sample)`; a pair ID is the canonical hash of `[experiment_id, suite_id, case_id, sample]`, not a Run ID. Owner: [Portable evaluation format](evals/format.md).

**Participant** — The Planner or one logical Worker of a StageExecution, each of which reports separately. Owner: [Observability](platform/observability.md).

**Passthrough Planner (`passthrough@1`, `PassthroughPlanner`)** — The deterministic, model-free baseline Planner that sends one Worker task to its sole Worker and maps the completion into a candidate. Owner: [Planners](execution/planners.md).

**Payload backend** — The storage that holds the immutable Artifact payload bytes of an installation, either `postgresql` (default) or `filesystem`, selected once at Server startup. Owner: [Artifact blob storage](artifacts/blob-storage.md).

**Performance metrics** — Contractor's own collection, storage and presentation of Server, HTTP, connection-pool, PostgreSQL and host GPU measurements and of completed-allocation resource summaries. Owner: [Operations](operations/operations.md).

**Permanent failure** — A failure classified as non-retryable within its retryability scope. Owner: [Failure model](platform/failure-model.md).

**Phase** — An Audit's progress through not-started, preparing, inventory and rounds, required in public projections and separate from its lifecycle state. Owner: [Audits](audits/audits.md).

**Physical generation** — One physical copy of a payload identified by its object key; several generations can share a content digest. Owner: [Artifact blob storage](artifacts/blob-storage.md).

**Physical object key** — The Server-generated name of one filesystem payload object, which includes a unique physical generation; logical artifact names and URLs never become filesystem paths. Owner: [Artifact blob storage](artifacts/blob-storage.md).

**Physical orphan** — A filesystem object or abandoned staging file that no registry record references, left for offline cleanup. Owner: [Artifact blob storage](artifacts/blob-storage.md).

**Pin** — The recorded exact identity of one dimension that affects an execution or its scoring, with its availability and evidence origin. Owner: [Portable evaluation format](evals/format.md).

**Placement requirement** — The exact runtime, sandbox, tools, adapters and workspace constraints of one resolved Stage Agent binding. Owner: [Control Plane](execution/control-plane.md).

**Plan** — The frozen, expanded experiment with its pins and expected member set (`playground.plan/v1`), whose exact JSON bytes are frozen before the first submit. Owner: [Portable evaluation format](evals/format.md).

**Plan revision** — The unsigned counter that every committed visible plan change increments. Owner: [Planners](execution/planners.md).

**Planner** — The Server component constructed for one prepared Stage that runs one invocation over the fixed prepared Worker set and returns a candidate StageResult. Owner: [Planners](execution/planners.md).

**Planner mirroring** — The Server-side derivation of a model-backed Planner's MemoryTools interface from the operations its prepared logical Workers select. Owner: [Memory](tools/memory.md).

**Planner Session** — The database-backed Contractor session of one Planner, created for one StageExecution. Owner: [Planners](execution/planners.md).

**Planner timeline** — The public sequence of fixed-kind reduced Planner facts. Owner: [Planners](execution/planners.md).

**PlannerFactory** — The replaceable Go abstraction that constructs a Planner for one prepared Stage. Owner: [Planners](execution/planners.md).

**`PlannerSubtask`** — see **Subtask**.

**PlannerTelemetryAdapter registry** — The Server-owned, versioned registry of telemetry adapters a Planner may use, against which publication validates planner.telemetry.adapter. Owner: [Observability](platform/observability.md).

**`podman@1`** — The SandboxProfile that gives an allocation one rootless Podman container sharing its local direct project directory. Owner: [Code execution](tools/code-execution.md).

**Preparation attempt** — One submission of a prepare role, identified by (audit_id, workflow_role, role_attempt), with no Round, no execution items and at most one Run. Owner: [Audit preparation and OpenAPI scans](audits/scans.md).

**Preparation digest** — The digest that binds preparation policy, source, options, selected operation and mode. Owner: [Scanners](tools/scanners.md).

**Preparation gap** — A pointer and fixed code recording source data that preparation omitted, ignored or could not use. Owner: [Scanners](tools/scanners.md).

**Preparation interruption** — A bounded failure of allocation preparation on the Runtime Agent in which no Planner starts and the whole reservation batch drains; its retryability depends on the cause. Owner: [Failure model](platform/failure-model.md).

**Prepare role** — A Workflow role binding with kind: prepare whose ordinary Workflow produces Audit inputs before any item exists. Owner: [Audit preparation and OpenAPI scans](audits/scans.md).

**Prepared native plan** — The exact portable plan/v1 and its reachable case, suite and binding resources, materialized by the Go implementation from a draft and then frozen. Owner: [Managed Evals](evals/managed-evals.md).

**Prepared reader** — The `list_findings` tool of one allocation once Runtime has materialized and verified every document of its single bound collection in the current Run. Owner: [Findings](findings/findings.md).

**PreparedExecution** — The execution resource that `AllocationService` prepares through `PodmanLifecycle` after hydration, separate from the allocation scratch. Owner: [Code execution](tools/code-execution.md).

**Presentation metadata** — Optional authored Catalog strings (`displayName`, `description`) on a Workflow with no execution semantics. Owner: [Workflows](configuration/workflows.md).

**Primary journey** — The user path Project → materials → Workflow → Run setup → execution → result → next Run or Audit review. Owner: [UI journeys](ui/journeys.md).

**Primary output** — A Workflow output slot marked `primary: true`, identifying a canonical user-facing result; it never affects execution and affects only Project Workflow recommendations. Owner: [Workflows](configuration/workflows.md).

**Probe container** — A container that Podman startup discovery creates under a separate ownership identity and must clean up before the profile is advertised. Owner: [Code execution](tools/code-execution.md).

**Process generation** — A random value carried by the Operations snapshot that changes whenever the Server restarts. Owner: [Operations](operations/operations.md).

**Process settings** — The typed, non-secret, restart-only settings in ServerConfig.spec. Owner: [Operations](operations/operations.md).

**Profile mode** — The program kind of an AuditProfile (risk-assessment, requirements-verification, custom-checklist, operation-tracing or finding-verification), which fixes the initial item source and the principal result. Owner: [Audits](audits/audits.md).

**Progressive disclosure** — The Skill access pattern in which the model first sees skill names and descriptions and loads instructions or resources only by calling `load_skill` or `load_skill_resource`. Owner: [Agent Skills](tools/skills.md).

**Project** — An owner-scoped, long-lived workspace that groups reusable artifacts and the ordinary WorkflowRuns created from them; it is not a Workflow, Scheduler, queue, execution state machine, authorization role or Runtime boundary. Owner: [Projects](artifacts/projects.md).

**Project HTTP target** — The optional application URL and RuntimeCredential reference a Project stores for HTTP tools, exposed as httpTarget and pinned into Runs as projectHttpTarget. Owner: [Projects](artifacts/projects.md).

**Project output binding** — The binding `ProjectScope(project_id): outputs/<slot>` that create-only automatic publication creates from a successful Project Run output. Owner: [Projects](artifacts/projects.md).

**Project Overview** — The bounded root page of a Project that reads fixed samples instead of every workspace collection. Owner: [Web UI](ui/ui.md).

**Project Run** — A WorkflowRun created through POST /v1/projects/{project_id}/runs whose artifact refs resolve in ProjectScope and which carries immutable project_id membership. Owner: [Projects](artifacts/projects.md).

**Project workspace** — The allocation-private project file tree that a Stage declares in `context.workspace`, a session separate from the `AllocationWorkspace` scratch directory. Owner: [Run workspaces and filesystem tools](tools/workspace.md).

**Projection tool** — A Server-side Planner tool that returns one fixed projection of a logical Worker's live State snapshot. Owner: [Planners](execution/planners.md).

**ProjectScope** — The ArtifactStore structural scope `ProjectScope(project_id)` of a Project owner's long-lived reusable artifact workspace; it holds one Project's artifact bindings, is bound from authentication and the Project resource, and is accessible only to the Project owner. Owner: [Projects](artifacts/projects.md).

**Proposal receipt** — The trusted intake record, identified by `receipt_id`, that the Server commits in the same transaction as the proposal Artifact and that a generic artifact write cannot forge. Owner: [Findings](findings/findings.md).

**Provider** — An execution system registered under a `VersionedId`, for example `contractor@1`. Owner: [Portable evaluation format](evals/format.md).

**Public API** — The versioned, authenticated /v1 Server API described by the committed OpenAPI 3.1 contract and used by browser and non-UI clients. Owner: [Public API](api/public-api.md).

**Publication** — A controlled fork of an exact Run output version into UserScope (explicit and revision-checked) or ProjectScope (create-only, on Project Run success). Owner: [Artifact store](artifacts/artifact-store.md).

**Publication projection** — The explicit, allowlisted safe projection of a private record that is written to a Project, carrying its own digest and a `source_record_sha256`. Owner: [Portable evaluation format](evals/format.md).

**Publication receipt** — The small JSON artifact in `finding-collection-receipts/<clientKey>` that records the normalized request digest and exact ZIP metadata of one published finding collection. Owner: [Findings](findings/findings.md).

**Publication result** — The bounded record of one automatic publication attempt for one present declared output: published, already_present or failed, with the exact source revision and, when published, the target revision. Owner: [Projects](artifacts/projects.md).

**Purge set** — The complete RunScope or ProjectScope that one controlled scope purge removes, together with the lineage edges it owns. Owner: [Retention and deletion](artifacts/retention-and-deletion.md).

## Q

**Queue** — The owner-scoped top-level UI and read API over ordinary nonterminal WorkflowRuns, which is not a durable queue table or Scheduler. Owner: [Projects](artifacts/projects.md).

## R

**Receipt** — (1) The recorded response to an accepted submission, which is not an ArtifactRef and claims neither publication nor Audit acceptance. Owner: [Audit Worker completion](audits/worker-completion.md). (2) An immutable record appended to a submission; authoritative receipts establish member association. Owner: [Portable evaluation format](evals/format.md).

**Recommended group** — The default group of Project Workflow cards for compatible Workflows (every required artifact slot has at least one candidate), minus any Workflow that has at least one primary output and whose primary destinations all already exist at `ProjectScope(project_id): outputs/<slot>`. A Workflow with no primary output, or with a partially complete set, stays in the group. Owner: [Projects](artifacts/projects.md).

**Reconcile step** — One claimed pass of the Audit Controller over one Audit. Owner: [Audit execution](audits/execution.md).

**Reconciliation** — The Control Plane's handling of a mismatch between observed and authoritative allocation state, which can trigger drain, reissue or release but never itself releases, adopts or replaces an allocation. Owner: [Control Plane](execution/control-plane.md).

**Recorded-output provider** — A provider that serves recorded fixture results through the same interface as a real provider, used in conformance tests. Owner: [Portable evaluation format](evals/format.md).

**Recoverable Project deletion** — The durable, restart-safe cancel/drain/purge operation that removes a Project after it enters the deleting state. Owner: [Retention and deletion](artifacts/retention-and-deletion.md).

**Recovery gate** — Persisted model-availability state for one owner, resolved gateway digest, model, credential identity and model HTTP proxy route. Owner: [Scheduler](execution/scheduler.md).

**Recovery staging** — An Audit-managed binding that exists before its receipt or report-link transaction commits and is not an acceptance signal. Owner: [Review and reports](audits/review-and-reports.md).

**Redacted session view** — The model-visible projection of the HTTP session that `http_session_get` returns. Owner: [HTTP and Caido tools](tools/http-caido.md).

**Reduced fact** — A durable, allowlisted record derived from a Planner event that never carries prompts, model text or generic tool payloads. Owner: [Planners](execution/planners.md).

**Reference-safe garbage collection** — Removal of an immutable artifact version or blob only when no binding, retained revision, pin, lineage edge or other scope outside the purge set still references it. Owner: [Retention and deletion](artifacts/retention-and-deletion.md).

**Registry entry (limit)** — The one row for a bound in the limits registry, identified as `LIM-<area>-<n>`, where `<area>` is the requirement prefix of the owning document and `<n>` numbers the bounds that document owns. Owner: [Limits](platform/limits.md).

**Related telemetry** — A report delivered after its StageExecution is terminal, retained without changing the frozen terminal facts. Owner: [Observability](platform/observability.md).

**Reminder** — The deterministic, Runtime-authored continuation that names the missing trusted item keys. Owner: [Audit Worker completion](audits/worker-completion.md).

**Repeat draft** — The owner-only, read-only projection of a terminal Run that the UI uses to configure another Run. Owner: [Runs](execution/runs.md).

**Repeat-request fragment** — The bounded non-secret part of the create-Run request that Run creation retains for the repeat draft. Owner: [Runs](execution/runs.md).

**Replay** — A repeated submission of identical canonical content, which returns the existing receipt without incrementing the item revision. Owner: [Audit Worker completion](audits/worker-completion.md).

**Repository access** — The personal Settings editor in which an owner replaces or removes their Git SSH key. Owner: [Git import](artifacts/git-import.md).

**Request ID** — (1) The bounded X-Request-ID value that correlates one HTTP request across hops and diagnostics. Owner: [Observability](platform/observability.md). (2) The allocation-monotonic identifier of one `http_request` call. Owner: [HTTP and Caido tools](tools/http-caido.md).

**Request tag** — The opaque allocation-derived `X-Request-Id` value that lets proxy history correlate HTTP tool and Caido Replay traffic. Owner: [HTTP and Caido tools](tools/http-caido.md).

**RequestSet** — The bounded, versioned application/vnd.contractor.http-requests+json set of neutral prepared HTTP requests with exact source provenance and preparation gaps. Owner: [Scanners](tools/scanners.md).

**Required Audit capabilities** — The closed, versioned set of Server mechanisms a profile needs, computed for presentation by profile reads and draft creation and enforced at start. Owner: [Audits](audits/audits.md).

**Required-equal dimension** — A pinned dimension that the experiment's `comparison.required_equal` declares must match between baseline and candidate. Owner: [Portable evaluation format](evals/format.md).

**Reservation** — Capacity set aside inside a shared ceiling or budget before the content that uses it is admitted. Owner: [Limits](platform/limits.md).

**Reserved Namespace** — A Namespace or Namespace prefix with its own contract: `inputs`, `outputs`, `skills` and the hidden `contractor-system` in RunScope, none of which can be a Stage Agent Namespace; exactly `audit-standards` in UserScope; the `audit-*` prefix in ProjectScope. Write rules differ per Namespace: the allocation baseline write grant permits CAS updates of existing `inputs` bindings (working input copies) but never creates one, while `outputs`, `skills`, `contractor-system`, `audit-standards` and `audit-*` bindings are written only by trusted Server operations. Owner: [Artifact store](artifacts/artifact-store.md).

**ResolvedExecutionConfig** — The effective per-consumer ModelPolicy, Gateway and credential selection that Run initialization stores in the immutable WorkflowRun snapshot. Owner: [Agent templates](configuration/agent-templates.md).

**ResolvedInstructions** — The normalized configuration-root-relative instruction ref, its content digest and the resolved UTF-8 text. Owner: [Agent templates](configuration/agent-templates.md).

**ResolvedRuntimeConfig** — The merged, non-secret, exact configuration and provenance for one consumer or allocation. Owner: [Runtime configuration, labels, credentials and Gateways](configuration/runtime-config.md).

**ResolvedRuntimeConfigProvenance** — The secret-free record of label pins, binding revisions, RuntimeConfig refs and digests, adapter refs and Gateway and credential refs stored for a StageExecution or allocation. Owner: [Runtime configuration, labels, credentials and Gateways](configuration/runtime-config.md).

**ResolvedSkill** — The AllocationSpec `resolvedSkills` entry that names one exact Run skill fork and its package digest. Owner: [Agent Skills](tools/skills.md).

**Resource admission** — The step that moves a pending Run to running and sets its first started_at. Owner: [Runs](execution/runs.md).

**Response body artifact** — The ordinary Run artifact of media type `application/vnd.contractor.http-body+json` that holds one complete response body. Owner: [HTTP and Caido tools](tools/http-caido.md).

**Response record** — The bounded metadata and preview that `http_request` returns for one completed request. Owner: [HTTP and Caido tools](tools/http-caido.md).

**Result** — The observed execution and collected outputs of one member, independent of quality (`playground.result/v2`). Owner: [Portable evaluation format](evals/format.md).

**Result artifact** — A Stage-local name in `result.artifacts` that may appear as a key of `StageResult.artifacts`, with a slot contract and a versionless `from` binding (omitted only by a Runtime-owned workspace export slot). Owner: [Workflows](configuration/workflows.md).

**Result collector** — The bounded, invocation-local collection of accepted check results that submit_check_result builds. Owner: [Audit Worker completion](audits/worker-completion.md).

**Result finalizer** — The ephemeral, tool-free ADK agent that copies the subtask ID and terminal Worker text exactly into a WorkerModelResult in one provider call. Owner: [Worker](execution/worker.md).

**Retention** — How long an entity, or a version of it, stays stored or readable, for example for a fixed period, without ever being purged, while something refers to it, or for the lifetime of another entity. Owner: [Entity lifecycle](platform/lifecycle.md).

**Retryability scope** — What a retryability classification is relative to: the same request, the same immutable configuration, or a fresh StageExecution. Owner: [Failure model](platform/failure-model.md).

**Retryable classification** — Whether another attempt may succeed after the failure a code reports: for Memory codes a retry without changing the input, for a Worker or Stage failure a fresh StageExecution. Owner: [Error and stop codes](platform/errors.md).

**Review kind** — The purpose of a review request: active-check-approval, requirement-applicability, finding-triage or report-acceptance. Owner: [Review and reports](audits/review-and-reports.md).

**Review observation** — The captured Audit review state of one finding at one revision, recorded in a collection entry as provenance rather than as a dereference capability. Owner: [Findings](findings/findings.md).

**Review renewal** — The single procedure that Resume and the Controller use to replace an expired item request with a fresh exact-subject request. Owner: [Review and reports](audits/review-and-reports.md).

**Reviewed form readiness** — The Run-draft layer meaning the user explicitly chose, uploaded or confirmed one exact ArtifactRef for every populated slot. Owner: [Projects](artifacts/projects.md).

**Revision** — An opaque selection token that identifies one exact retained binding revision and is not an authorization credential. Owner: [Artifact store](artifacts/artifact-store.md).

**Round** — An immutable, validated set of AuditItems with the lifecycle `proposed` → `accepted` → `executing` → `assessing` → `closed`. Owner: [Audits](audits/audits.md).

**Router Planner (`router@1`)** — The model-backed Planner that works through a bounded subtask plan and selects a logical Worker by worker_name for each dispatch. Owner: [Planners](execution/planners.md).

**Run** — see **WorkflowRun**.

**Run claim** — The durable PostgreSQL claim through which one execution lane owns one WorkflowRun. Owner: [Scheduler](execution/scheduler.md).

**Run draft** — The in-memory, unsubmitted Run request for one exact Workflow version and scope, including its per-slot confirmations. Owner: [Web UI](ui/ui.md).

**Run event** — A row of workflow_run_events, the sole Run replay authority for the run stream. Owner: [Public API](api/public-api.md).

**Run Runtime-label snapshot** — The exact label-binding and RuntimeConfig versions pinned when a WorkflowRun is created. Owner: [Runtime configuration, labels, credentials and Gateways](configuration/runtime-config.md).

**Run setup drawer** — The Configure Run surface, shown as a drawer on desktop and a full-width panel on mobile. Owner: [Web UI](ui/ui.md).

**Run skill fork** — The exact RunScope `skills/<name>` binding that Scheduler forks from the pinned owner source revision and that allocations load. Owner: [Agent Skills](tools/skills.md).

**Run snapshot** — The complete validated and dependency-resolved Workflow snapshot stored with a Run, which is the execution authority for that Run. Owner: [Runs](execution/runs.md).

**`run-artifacts@1`** — The built-in current-Run Artifact Toolset that exports exactly `list_artifacts`, `read_artifact` and `write_artifact`; each operation must be explicitly selected. Owner: [Artifact store](artifacts/artifact-store.md).

**`run_workdir`** — The single logical root of a project workspace, to which every workspace path is relative. Owner: [Run workspaces and filesystem tools](tools/workspace.md).

**RunArtifactSpace** — The RunScope of one WorkflowRun, viewed as the durable Stage-output store and data plane for concurrent Worker collaboration rather than as a separate storage component. Owner: [Artifact store](artifacts/artifact-store.md).

**Runs surface** — The single Runs destination of the Web UI, with its Queue and Completed views. Owner: [Web UI](ui/ui.md).

**RunScope** — The scope of one WorkflowRun's artifacts, shared by its Planner and every Worker across all Stages. Owner: [Artifact store](artifacts/artifact-store.md).

**Runtime adapter** — Versioned Runtime-owned code that consumes one typed settings block, for example otlp-http@1, http-proxy@1 or caido-graphql@1. Owner: [Runtime configuration, labels, credentials and Gateways](configuration/runtime-config.md).

**Runtime adapter metrics** — Per-adapter counters in a Runtime report, keyed by the exact RuntimeAdapter ref selected by trusted Server provenance. Owner: [Observability](platform/observability.md).

**Runtime Agent** — One long-running process with one exclusive slot whose code implements both the Runtime Agent and the in-process Worker behaviour. Owner: [Runtime Agent](execution/runtime-agent.md).

**Runtime Agent principal** — The stable runtime_agent_id, lowercase hex SHA-256 of the authenticated leaf certificate's DER SubjectPublicKeyInfo, together with its durable Agent labels. Owner: [Control Plane](execution/control-plane.md).

**Runtime Agent Registry** — The Control Plane's in-memory index of live Runtime Agent processes, their heartbeat state, control leases and slot availability. Owner: [Control Plane](execution/control-plane.md).

**Runtime config document** — The non-secret /runtime-config.json that Node serves, holding exactly the UI version, the supported API versions and one absolute apiBaseUrl. Owner: [Web UI](ui/ui.md).

**Runtime credential** — An encrypted secret of a versioned adapter-owned credential kind, such as otlp-headers@1, referenced by ID from a RuntimeConfig. Owner: [Runtime configuration, labels, credentials and Gateways](configuration/runtime-config.md).

**Runtime credential marker** — `[runtime-target-credential]` or `[runtime-credential]`, the text that replaces a header line carrying a credential the Runtime injected; HTTP finding evidence captures `[runtime-target-credential]` in place of the Project target credential. Owner: [HTTP and Caido tools](tools/http-caido.md).

**Runtime label** — A bounded name such as debug or caido selected through runtimeLabels on a Run or assigned to a Runtime Agent, acting as an alias for an exact RuntimeConfig. Owner: [Runtime configuration, labels, credentials and Gateways](configuration/runtime-config.md).

**Runtime report** — The RuntimeReport of allocation lifecycle facts, including Runtime adapter metrics, that the Runtime Agent around a Worker observes. Owner: [Observability](platform/observability.md).

**Runtime service endpoint** — An endpoint the Runtime itself depends on (LLM Gateway, Artifact API, telemetry collector, forward proxy, Caido control API, Control Plane, advertised control and A2A URLs, private listener), which the target policy always denies. Owner: [Security](platform/security.md).

**Runtime target credential marker** — see **Runtime credential marker**.

**Runtime target policy** — The shared Runtime rule, used by http-tools@1, scan@1 and request-bound caido@1 actions, that classifies destination IP addresses and decides which ones tools may reach. Owner: [Security](platform/security.md).

**RuntimeConfig** — One immutable, versioned, typed, non-secret infrastructure configuration stored by Server in PostgreSQL. Owner: [Runtime configuration, labels, credentials and Gateways](configuration/runtime-config.md).

**RuntimeSettings** — The secret-bearing wire snapshot created for one allocation after placement and never mutated while the allocation is active. Owner: [Runtime configuration, labels, credentials and Gateways](configuration/runtime-config.md).

**RuntimeSettings secret** — A secret value delivered to a Runtime Agent in an allocation's RuntimeSettings, such as the LLM Gateway token or an infrastructure credential. Owner: [Security](platform/security.md).

## S

**Sample** — The one-based repetition number of a case and variant in a plan. Owner: [Portable evaluation format](evals/format.md).

**Sample group** — A set of performance measurements collected together that carries its own observation time, last attempt and result, coverage and status. Owner: [Operations](operations/operations.md).

**SandboxProfile** — Registered Runtime Agent behavior that an AgentTemplate selects by exact `<id>@<version>` ref, never an arbitrary mapping of host settings. Owner: [Agent templates](configuration/agent-templates.md).

**Scan executor Stage** — The Workflow Stage named by auditTask.stage, which executes one assigned openapi-scan task with one scanner Worker. Owner: [Audit preparation and OpenAPI scans](audits/scans.md).

**Scan plan** — The immutable application/vnd.contractor.scan-plan+json artifact that scan-plan@1 builds from one input. Owner: [Scanners](tools/scanners.md).

**Scan session** — The durable journal of a scan plan's job records, fenced by the Scheduler claim. Owner: [Scanners](tools/scanners.md).

**Scan settings artifact** — The JSON document with schema contractor.audit.openapi-scan-settings.v1 that selects the scanner, target server, operations and their bindings. Owner: [Audit preparation and OpenAPI scans](audits/scans.md).

**Scan-budget stop** — A next-Round builder pass that reaches its receipt scan limit while another receipt remains. Owner: [Audit execution](audits/execution.md).

**Scanner wordlist** — A text/vnd.contractor.wordlist (or text/plain) artifact holding one scanner payload per line. Owner: [Scanners](tools/scanners.md).

**ScanToolset** — The scan@1 Toolset exporting scan_nuclei, scan_sqlmap, scan_naabu and scan_ffuf. Owner: [Scanners](tools/scanners.md).

**Scheduler** — see **Workflow Scheduler**.

**Scheduler settings** — The singleton deployment setting that holds maxConcurrentRuns, its revision and its update time. Owner: [Operations](operations/operations.md).

**Seal** — The atomic freeze of a full valid collection at a normal completion, after which submissions and updates are rejected. Owner: [Audit Worker completion](audits/worker-completion.md).

**`seed_drift`** — The bundled-initialization outcome for an existing owner Skill binding whose digest or media type differs from the bundled source; the binding stays current. Owner: [Agent Skills](tools/skills.md).

**Selected HTTP evidence** — The `http_exchange` snapshot of one captured outgoing request that the HTTP facade copies into a proposal when given a `request_id`. Owner: [Findings](findings/findings.md).

**Selection** — The explicit choice of exact result and assessment records that a view uses. Owner: [Managed Evals](evals/managed-evals.md).

**Selector** — The compact `<id>@<version>` authoring and API lookup value, never the complete durable execution identity. Owner: [Configuration catalog](configuration/catalog.md).

**Sensitive-output tool** — A tool that declares contractor_sensitive_output because its results carry command output, captured HTTP traffic or HTTP session material. Owner: [Observability](platform/observability.md).

**Server generation** — The identity of one Server boot in performance history; counters from different generations are never subtracted. Owner: [Operations](operations/operations.md).

**Session cookie** — The `__Host-contractor_session` cookie that carries the unpadded base64url form of a browser session's random 32-byte value. Owner: [Public API](api/public-api.md).

**Session mode** — The Stage's immutable `isolated` or `shared` Worker session policy, shared by all of its Agent bindings, which decides whether sequential invocations of one allocation get fresh or shared ADK conversations. Owner: [Worker](execution/worker.md).

**Session policy** — see **Session mode**.

**Settlement** — The final disposition of an AuditItem, reached by an accepted final result or by exhaustion, a non-retryable outcome, cancellation or explicit exclusion. Owner: [Audit execution](audits/execution.md).

**Shallow surface** — The search_def and list_symbols operations of code-analysis@1, implemented with Tree-sitter directly on the workspace snapshot and portable across both workspace providers. Owner: [Code analysis](tools/code-analysis.md).

**Shared classification fixture** — The test fixture api/testdata/v1alpha1/gateway-failure-classification-cases.json on which the Server-side Planner and Runtime Worker Gateway failure classifiers must agree. Owner: [Failure model](platform/failure-model.md).

**Shipped catalog** — The operator-root manifests and instruction resources that the repository ships under `configs/`; until the first release they may change in place without a new version (CAT-R74). Owner: [Configuration catalog](configuration/catalog.md).

**Shutdown grace** — The bound (CONTRACTOR_SHUTDOWN_GRACE_SECONDS, default 10 seconds) within which the Runtime stops and destroys its Worker after lease loss, before forced termination. Owner: [Runtime Agent](execution/runtime-agent.md).

**Single-VM baseline** — The baseline deployment layout in which one Contractor Server process, PostgreSQL and one or two single-slot Runtime Agent processes (the second only if concurrency is needed) run on one VM. Owner: [System overview](overview.md).

**Skill package** — see **Agent Skill**.

**SkillCatalog** — The Server's internal ArtifactStore-backed boundary that discovers bundled sources, validates packages, resolves owner refs and forks exact packages into RunScope, with no HTTP endpoint or database schema of its own. Owner: [Agent Skills](tools/skills.md).

**Slot** — The single unit of Runtime Agent capacity; one Runtime Agent process has exactly one. Owner: [Control Plane](execution/control-plane.md).

**Snapshot revision** — An unsigned decimal counter of the Operations snapshot that every observable registry change advances. Owner: [Operations](operations/operations.md).

**`source_content_digest`** — SHA-256 over the exact accepted OpenAPI source artifact bytes, identifying provenance and changing after reformatting or JSON/YAML conversion. Owner: [Audits](audits/audits.md).

**SQLMap request artifact** — The six-field JSON description of one HTTP request that scan_sqlmap request mode accepts. Owner: [Scanners](tools/scanners.md).

**Stable service owner** — The stable, unique local Runtime service identity that owns Podman containers, held under an exclusive local ownership lock. Owner: [Code execution](tools/code-execution.md).

**Stage** — A named node of the Workflow graph with one objective, one Planner, Worker bindings, context, result contract and transitions. Owner: [Workflows](configuration/workflows.md).

**Stage Agent binding** — The object a logical Agent name maps to: an exact AgentTemplate selector and an optional Namespace that defaults to the name. Owner: [Workflows](configuration/workflows.md).

**Stage Agent Namespace** — The Namespace that Scheduler resolves for a Stage Agent from its binding before allocation and fixes when Worker tools are constructed. Owner: [Artifact store](artifacts/artifact-store.md).

**StageContext** — The immutable snapshot of exact artifact versions that a StageExecution pins before requesting any allocation. Owner: [Scheduler](execution/scheduler.md).

**StageExecution** — One attempt to execute one Stage in one WorkflowRun; there is no separate StageAttempt entity. Owner: [Scheduler](execution/scheduler.md).

**StageOutcome** — The outcome of a StageResult: succeeded or failed. Owner: [Scheduler](execution/scheduler.md).

**StageResult** — The normalized semantic result of one Planner invocation, with StageOutcome succeeded or failed. Owner: [Scheduler](execution/scheduler.md).

**StageTermination** — The durable Scheduler decision that an execution ends without an accepted semantic result, producing cancelled or interrupted. Owner: [Scheduler](execution/scheduler.md).

**Staging file** — A file under a unique staging name inside the blob root that receives payload bytes before atomic publication and is never visible to readers. Owner: [Artifact blob storage](artifacts/blob-storage.md).

**Standalone Run** — A WorkflowRun created without a Project. Owner: [Runs](execution/runs.md).

**Standby Server** — A Server process on the same database that does not hold the advisory lease and has only public health and 503 readiness until it acquires it. Owner: [Persistence](platform/persistence.md).

**Static operation** — A GraphQL document that is a source-code constant selected by a Caido tool method. Owner: [HTTP and Caido tools](tools/http-caido.md).

**Status rules** — The fixed openai-compatible@1 classification of transport errors, HTTP statuses and the x-should-retry hint into transient and permanent failures. Owner: [Failure model](platform/failure-model.md).

**Stop code** — An error code with which a Planner invocation or a StageExecution ends without a semantic candidate and which is recorded in its interrupted StageTermination. Owner: [Error and stop codes](platform/errors.md).

**Streamline Planner (`streamline@1`)** — The model-backed Planner that works through a bounded subtask plan with exactly one logical Worker. Owner: [Planners](execution/planners.md).

**Submission** — (1) The durable per-member intent of one external mutation, with its receipts and recovery bookkeeping (`playground.submission/v1`). Owner: [Portable evaluation format](evals/format.md). (2) The immutable member operation, with a request digest, stable key and verified execution receipt, that creates the member's ordinary Run or Audit. Owner: [Managed Evals](evals/managed-evals.md).

**Submission identity** — The submission key derived from audit_id, round/role, the immutable ordered execution-item/attempt set and the execution manifest digest. Owner: [Audit execution](audits/execution.md).

**Submission intent** — The durable record of one planned child Run submission that reserves Audit budget in its own transaction and carries one stable idempotency key until the original Run is recovered. Owner: [Audit execution](audits/execution.md).

**Subtask (`PlannerSubtask`)** — One plan record with a Server-assigned decimal string ID, its objective, its instructions and an adapter-controlled status. Owner: [Planners](execution/planners.md).

**Subtask ID** — The opaque, mandatory subtask_id of a private Worker request; Streamline and Router copy the ID of the exact claimed PlannerSubtask and Passthrough uses the fixed ID 0. Owner: [Worker](execution/worker.md).

**Suite** — Exact cases with their scorer configuration, thresholds and required checks (`playground.suite/v2`). Owner: [Portable evaluation format](evals/format.md).

**`symbolId`** — The opaque, integrity-protected, allocation-local identity of one graph symbol, bound to one workspace digest and one complete upstream Trailmark node ID. Owner: [Code analysis](tools/code-analysis.md).

## T

**Taint annotation** — One machine-readable @trace, @validate or @sink comment line placed above a function-like declaration while tracing data and control flow. Owner: [Code analysis](tools/code-analysis.md).

**`taint-annotations@1`** — The structured source-mutation Toolset that inserts canonical @trace, @validate and @sink comments above function-like declarations in the allocation workspace. Owner: [Code analysis](tools/code-analysis.md).

**Target list** — A text/vnd.contractor.target-list artifact of nonempty target lines. Owner: [Scanners](tools/scanners.md).

**Terminal observation** — The authoritative terminal Run version and outcome that the Controller records on an AuditExecution. Owner: [Audit execution](audits/execution.md).

**Terminal release recovery** — Recovery that retries the incomplete allocation releases of terminal StageExecutions. Owner: [Scheduler](execution/scheduler.md).

**Terminal Run hard deletion** — Permanent removal of a terminal, fully released Run and all Run-owned durable state in one controlled transaction. Owner: [Retention and deletion](artifacts/retention-and-deletion.md).

**Terminal summarization** — Stopping the normal Worker loop and making one separate tool-free model call that returns a final WorkerResult with summarized true. Owner: [Worker](execution/worker.md).

**Terminal Worker text** — The bounded semantic text the main tool-using Worker model ends with, stating what one exact subtask produced and nothing about Contractor lifecycle. Owner: [Worker](execution/worker.md).

**Time-limit dialog** — The dialog that Start and Continue open on an Audit to choose its time limit before any mutation is sent. Owner: [UI journeys](ui/journeys.md).

**Tolerance window** — The bounded time of ten seconds that a lease-session liveness probe may run, long enough to absorb transient stalls and below the lease session's keepalive bound. Owner: [Persistence](platform/persistence.md).

**Tombstone provenance** — Bounded provenance that survives a deleted source Run when its content does not. Owner: [Retention and deletion](artifacts/retention-and-deletion.md).

**Tool description** — The English, model-facing description of one selected callable, kept in `description` and exposed as `__doc__` to ADK FunctionTool. Owner: [Toolsets](tools/toolsets.md).

**Tool Worker** — A Worker whose AgentTemplate selects the model-free runtime tool@1 and invokes one registered callable without a model. Owner: [Scanners](tools/scanners.md).

**Tool Worker report** — The bounded application/json artifact that carries a tool invocation's unchanged observation. Owner: [Scanners](tools/scanners.md).

**Toolset** — A versioned group of model-visible tools implemented by the selected Worker runtime. Owner: [Toolsets](tools/toolsets.md).

**Toolset capability** — An exact Toolset ref plus the subset of its exported tools whose prerequisites passed in one Runtime Agent's process environment, with the same granularity as ToolsetSelection. Owner: [Toolsets](tools/toolsets.md).

**Toolset descriptor registry** — The Server's lightweight registry of Toolset descriptors used for authoring-time validation. Owner: [Toolsets](tools/toolsets.md).

**Toolset ref** — The exact versioned identifier of registered Toolset code, for example `likec4@1`. Owner: [Toolsets](tools/toolsets.md).

**Toolset selection (`ToolsetSelection`)** — An AgentTemplate's choice of one exact Toolset ref together with an explicit, non-empty allowlist of its exported tool names. Owner: [Agent templates](configuration/agent-templates.md).

**ToolsetFactory registry** — A Runtime Agent's own registry of installed and enabled Toolset factories, from which its startup probes derive its capability snapshot. Owner: [Toolsets](tools/toolsets.md).

**Trailmark child** — The allocation-local, killable child process that builds and queries one Trailmark graph for the current workspace digest. Owner: [Code analysis](tools/code-analysis.md).

**Transfer slot** — One unit of the per-Server-process budget of four simultaneous full-payload transfers, held from before body buffering or blob reads. Owner: [Artifact blob storage](artifacts/blob-storage.md).

**Transient failure** — A failure classified as retryable because a later attempt may succeed. Owner: [Failure model](platform/failure-model.md).

**Transition action** — The one typed action an outcome selects: `next`, `retry`, `escalate`, `succeed` or `fail`. Owner: [Workflows](configuration/workflows.md).

**Transition decision** — The immutable record, keyed by the source stage_execution_id, of the next, retry, escalate, succeed or fail progression chosen for one Stage outcome. Owner: [Scheduler](execution/scheduler.md).

**Transport adapter** — The small handwritten adapter around the generated TypeScript API client that alone resolves the base URL, sends the session cookie and CSRF token, supplies idempotency and precondition headers and maps responses into one bounded client error type. Owner: [Web UI](ui/ui.md).

**Trusted proxy** — A reverse proxy listed in `trustedProxies` whose `X-Forwarded-For` entries the Server walks to find the client IP. Owner: [Security](platform/security.md).

**Trusted Server envelope** — The identities, attribution and correlation metadata that the Server binds to a report from its own records, never from the report payload. Owner: [Observability](platform/observability.md).

**Trusted sink** — A telemetry backend that content capture trusts with unredacted content. Owner: [Observability](platform/observability.md).

**Two-phase release** — Release in which the Runtime first confirms cleanup while staying fenced and a later heartbeat release action lets it become idle. Owner: [Control Plane](execution/control-plane.md).

**Typed location** — A closed Source (`file` with optional `line` or `range`) or Web (`url` with optional `method`) object recording where the producer claims a finding is. Owner: [Findings](findings/findings.md).

**Typed Planner-plan projection** — The validated durable projection of the subtask plan used for recovery and the Run UI. Owner: [Planners](execution/planners.md).

## U

**Unwrapped single-item path** — Dispatch of one valid task package on its own, without an item-task-set. Owner: [Audit execution](audits/execution.md).

**Upload candidate** — The complete, published immutable object written for one upload before the registry transaction attaches it. Owner: [Artifact blob storage](artifacts/blob-storage.md).

**URL-only limitation** — A limitation code that records which declared request data a fixed URL scan does not replay or apply. Owner: [Audit preparation and OpenAPI scans](audits/scans.md).

**User story** — One user goal with completion criteria, identified as US-01 to US-11, where US-nn corresponds to UC-nn of the usability study. Owner: [UI journeys](ui/journeys.md).

**UserScope** — The scope of a user's durable artifact library, bound from authentication. Owner: [Artifact store](artifacts/artifact-store.md).

## V

**Variant** — One `{id, binding}` entry of an experiment; in an A/B comparison the baseline and candidate variants are its two arms. Owner: [Portable evaluation format](evals/format.md).

**Versioned ref** — An ArtifactRef with a revision; it addresses one exact historical binding revision and never falls forward. Owner: [Artifact store](artifacts/artifact-store.md).

**Versionless ref** — An ArtifactRef without a revision; it addresses the current logical binding. Owner: [Artifact store](artifacts/artifact-store.md).

**View generation** — One complete read model of an experiment for one selection revision, identified to clients by its viewSnapshot. Owner: [Managed Evals](evals/managed-evals.md).

## W

**Wake hint** — A process-local notification that may trigger reconciliation early but carries no authority. Owner: [Audit execution](audits/execution.md).

**Web UI** — The separately deployable Node.js service and the client-rendered single-page application it serves, which uses the Server's public API and is not a second Scheduler, configuration resolver or source of lifecycle truth. Owner: [Web UI](ui/ui.md).

**Work root** — The operator-configured, dedicated directory below which the Runtime creates allocation directories and probe resources. Owner: [Runtime Agent](execution/runtime-agent.md).

**Worker** — The temporary, allocation-scoped role and runtime instance created in process from one AgentTemplate for one Runtime Agent allocation, not a separate service or process. Owner: [Worker](execution/worker.md).

**Worker runtime** — The generic code, selected by an AgentTemplate's `runtime` field (such as `adk@1`), that executes the template. Owner: [Agent templates](configuration/agent-templates.md).

**Worker runtime factory** — An in-process factory, known by an exact WorkerRuntime ref such as adk@1, that creates the Worker instance of an allocation. Owner: [Runtime Agent](execution/runtime-agent.md).

**Worker summarizer** — The optional one-shot terminal summarization policy of an `adk@1` AgentTemplate, with fixed `terminal@1` semantics and its own ModelPolicy. Owner: [Agent templates](configuration/agent-templates.md).

**WorkerCompletion / `workerCompletion`** — (1) **WorkerCompletion**: The versioned private A2A union that carries exactly one WorkerResult or WorkerFailure plus invocation and state-revision control metadata. Owner: [Worker](execution/worker.md). (2) **`workerCompletion`**: The field of an AuditProfile check-workflow binding that selects the contract kind and, through stage and agent, the one logical Worker it applies to. Owner: [Audit Worker completion](audits/worker-completion.md).

**WorkerFailure** — A Runtime-authored technical Worker failure with a stable code, a sanitized message and a retryable flag. Owner: [Worker](execution/worker.md).

**WorkerHandle** — The immutable value that lets the Workflow Scheduler and Planner address one prepared Worker. Owner: [Control Plane](execution/control-plane.md).

**WorkerInstrumentationPlugin** — The one ADK plugin of a Worker that observes run, model and tool callbacks and feeds independent typed metrics and observation reducers. Owner: [Worker](execution/worker.md).

**WorkerModelResult** — The internal two-field (subtask_id, result) structure produced by the result finalizer or terminal summarizer, which never crosses A2A. Owner: [Worker](execution/worker.md).

**WorkerResult** — The trusted Worker result Runtime assembles from the validated semantic result, Runtime-owned observations, exact trusted artifact refs and the Runtime-owned summarized flag. Owner: [Worker](execution/worker.md).

**WorkerStateStore** — The canonical allocation-local store of the Contractor-owned Worker State from which isolated sessions are rehydrated. Owner: [Worker](execution/worker.md).

**Workflow** — A reusable, versioned definition of an application security check or supporting analysis as a sequence or graph of Stages. Owner: [Workflows](configuration/workflows.md).

**Workflow / agent author** — The scenario role that selects and checks published definitions, reads prompts and compares behavior variants; it is not an authorization role. Owner: [UI journeys](ui/journeys.md).

**Workflow discovery card** — The card that Project and Catalog views share for one logical Workflow name, with its version selector. Owner: [Web UI](ui/ui.md).

**Workflow parameter** — A named string input declared in a Workflow's `spec.parameters` with a mandatory boolean `required`; a Run request supplies the values as a string-to-string mapping, separately from its artifact-slot bindings, and the Run stores them immutably. Owner: [Workflows](configuration/workflows.md).

**Workflow role** — A named binding in an AuditProfile from an operator-chosen role name to a pinned ResolvedWorkflow closure with an explicit role kind (prepare, check, discovery or assessment) and its input, parameter and output mappings. Owner: [Audits](audits/audits.md).

**Workflow Scheduler** — The Server component that claims WorkflowRuns and creates, progresses and is the only durable writer of their StageExecutions. Owner: [Scheduler](execution/scheduler.md).

**Workflow user surface** — The UI capabilities for discovering published Workflows and configuring, starting, following and inspecting their Runs. Owner: [UI journeys](ui/journeys.md).

**WorkflowCatalog** — The Server component that loads and validates Workflows and resolves their selectors and instruction refs into the Run snapshot. Owner: [Configuration catalog](configuration/catalog.md).

**`workflowOutputs`** — A Stage's explicit mapping from Workflow output slots to its result artifact names. Owner: [Workflows](configuration/workflows.md).

**WorkflowRun** — The durable execution record that one Workflow invocation creates ("Run" for short), to which concrete input and output bindings belong. Owner: [Runs](execution/runs.md).

**WorkflowRunCancellation** — The immutable cancellation record (code user_cancelled, requested_at, optional requested_by and reason) written by the winning cancel request. Owner: [Runs](execution/runs.md).

**WorkflowRunError** — The stable machine-readable error (code, message, optional stage_execution_id) that a failed Run records. Owner: [Runs](execution/runs.md).

**Workspace capability** — The positive, frozen snapshot of storage, supported modes and limits that a Runtime reports at registration. Owner: [Run workspaces and filesystem tools](tools/workspace.md).

**Workspace handles** — The narrow `WorkspaceReader`, `WorkspaceWriter` and `WorkspaceChanges` interfaces through which tools reach a workspace session. Owner: [Run workspaces and filesystem tools](tools/workspace.md).

**Workspace mode** — Workflow-selected workspace semantics: `direct` or `overlay`. Owner: [Run workspaces and filesystem tools](tools/workspace.md).

**Workspace observation coverage** — What a Worker actually exposed to the model during one A2A invocation, as opposed to code-analysis scan coverage. Owner: [Worker](execution/worker.md).

**Workspace session** — One allocation-private hydrated tree and its tool handles. Owner: [Run workspaces and filesystem tools](tools/workspace.md).

**Workspace source** — An exact RunScope ZIP artifact used to hydrate a subtree of `run_workdir`. Owner: [Run workspaces and filesystem tools](tools/workspace.md).

**Workspace state** — A cumulative text-only overlay artifact which, together with the exact sources, reconstructs the effective tree. Owner: [Run workspaces and filesystem tools](tools/workspace.md).

**Workspace storage** — Runtime startup choice of physical workspace storage: `local` or `memory`. Owner: [Run workspaces and filesystem tools](tools/workspace.md).

**WorkspaceSnapshot** — An immutable value of the current managed-text projection and its canonical digest, acquired for one operation. Owner: [Run workspaces and filesystem tools](tools/workspace.md).

**Write fence** — The Server-side rejection of an allocation's Artifact writes, established on its grant when its StageExecution durably enters `finalizing` or `aborting` and serialized with complete mutations. The Control Plane establishes it together with the authoritative terminal phase before any outbound finalize or abort call, and a write-fenced allocation never continues Worker execution. Owner: [Artifact store](artifacts/artifact-store.md).
