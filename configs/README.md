# Contractor configuration

The pre-production catalog was reset on 2026-11-23. Each supplied object has
one current definition at `@1`; exact selectors and content locks still apply.

This directory contains the executable default configuration for Contractor's
Application Security orchestration platform. Versioned Workflows define checks
and supporting analyses; AuditProfiles define how Audits coordinate selected
Workflows into assessment programs. Extend the catalog with new Workflow and
AgentTemplate definitions, instructions and Skills, using the available toolsets and the
[configuration contract](../docs/spec/configuration/workflows.md).

[`server.local.yaml`](server.local.yaml) is a separate, non-secret
`ServerConfig` for the loopback demo process. It is not a published catalog
resource and the Workflow configuration loader ignores root-level YAML files.
Start the process with `contractor server run --config
./configs/server.local.yaml`; relative paths in that document resolve from the
document's directory. Database URLs, public bearer tokens, development LLM
tokens and all key bytes remain environment/file secrets and are deliberately
not valid `ServerConfig` fields.

The current user-facing Workflow set is:

- `openapi-prepare-work@1` for passive deterministic Server preparation of an exact
  OpenAPI document into separately selectable operation work. Supply
  `checking_workflow`, `selection` (`operations` or `scan-requests`) and
  `max_items`; scan-request selection also requires the exact `settings` input.
  The Workflow creates ordinary JSON materials, without a model or Runtime.

- `audit-web-nuclei-scan@1` for the `web-scan@1` AuditProfile: one explicit
  target with installed or exact custom rules; see [scanner setup](../runtime/README.md#cli-scanners);
- `openapi-from-workspace@1` and `likec4-from-workspace@1` for four-Stage local
  graph-backed analysis and document generation;
- `openapi-from-workspace-streamline@1` and `likec4-from-workspace-streamline@1`
  for the same document contracts with a modeled single-Worker Planner;
- `openapi-from-analysis@1` and `likec4-from-analysis@1` when the caller already
  has exact dependency and project reports;
- `security-analysis@1` and `taint-trace-from-workspace@1` for their explicit
  focused analysis contracts, and `taint-trace-from-workspace-compact@1`, the
  same trace with opt-in Worker context compaction;
- `audit-openapi-operation-observe@1` for graph-backed operation analysis in
  the limited `openapi-operation-observe@1` comparison variant;
- `audit-openapi-operation-trace@1` for graph-backed analysis and finding proposals
  in `openapi-operation-trace@1`, plus `findings-review@1` for a supplied collection
  and `source-findings-review@1` for exact source-backed proposal review;
  `project-findings-review@1` discovers current Project findings and retains an explicit exact selection;
- `audit-openapi-nuclei-scan@1` and `audit-openapi-sqlmap-scan@1` for the
  `openapi-nuclei-scan@1` and `openapi-sqlmap-scan@1` AuditProfiles;
- `project-facts-extract@1` for optional observation selection from an exact Memory snapshot, using one passthrough Worker and common fact;
- `artifact-copy@1` as a deterministic Server copy over an exact source, without a Worker;
- `podman-python-check@1` as the explicitly selected offline Podman execution fixture.

The passthrough stages of the workspace and analysis-based OpenAPI/LikeC4
Workflows, security analysis, taint tracing and both findings-review Workflows
retain terminal summaries and continue once with `streamline@1`. Each
continuation has a 30-minute allowance capped by the Server's Planner timeout,
and retains the original task, Worker selection, namespace, recorded artifacts
and selected Memory. Normal completion follows the original Stage graph.
Existing Streamline variants already use that Planner. Bounded Audit Workers
have no terminal summarizer. The Podman smoke fixture retains direct passthrough
completion and failure checks; artifact-copy executes as a Server operation.

All shipped Streamline/Router Planner Stages and summary-continuation targets
explicitly select `compact@1` with the shared generated `compactor@1` ModelPolicy
and `instructions/planner-compaction.md`. The Planner keeps recent complete
steps and a rolling summary while exact earlier Worker results remain readable.
For a direct Planner Stage the selection is `plannerContextManagement` with
`modelPolicy: compactor@1`; for a summary continuation, set the target's
`plannerContextManagement` under `on.summarized.escalate` and select
`executionConfig.planner.compactorModelPolicy: compactor@1` there.
The source Worker configuration remains independent.

Within each multi-Stage Workflow, standard YAML anchors and aliases define an
identical Planner block once at its first use. Direct Stages reuse
`&planner_compaction` / `*planner_compaction`; identical summary-continuation
targets reuse `&planner_continuation` / `*planner_continuation`. They produce
ordinary fully resolved Stage snapshots; there is no separate template/profile
or cross-file `$ref` mechanism.

[`examples/streamline_review_workflow.yaml`](examples/streamline_review_workflow.yaml)
and [`examples/router_openapi_workflow.yaml`](examples/router_openapi_workflow.yaml)
can be copied under `workflows/` directly; each uses the existing `compactor@1`.
The API still allows an explicit uncompacted Planner policy. Deterministic
Passthrough, scan-plan and server-operation Stages do not invoke a Planner model.

Set optional `spec.maxSubtasks` on a Workflow to bound each Streamline/Router
plan, including summary continuation. For example, `maxSubtasks: 64` allows up
to 64 records per invocation. Omission selects 32; explicit values must be
integers from 1 to 1000. Model-call, Worker-call, token, byte and time budgets
remain independent. Existing Runs retain their pinned selection.

Historical source-analysis and earlier workspace/Skill versions are deleted,
not archived or republished. Existing Runs retain their complete resolved
Workflow snapshots; the Audit catalog now starts at `@1` with current completion semantics.
This deliberate demo-era reset does not preserve pre-reset selector meanings. Process-only
Router/Streamline Memory fixtures live under `configs/e2e` and do not enter the
default catalog. The working successors listed above select Memory explicitly.
Repeat is blocked with `workflow_unavailable` when
the original exact Workflow is absent. Removing a bundled resource does not
remove copies published into a separate managed or operator catalog. Frozen
instruction-evaluation inputs retain their own exact catalog under `tests/eval`.

The shipped Audit catalog contains six common programs:
`openapi-operation-observe@1`, `openapi-operation-trace@1`,
`openapi-sqlmap-scan@1`, `openapi-nuclei-scan@1`,
`source-openapi-sqlmap-scan@1` and `web-scan@1`.
Eight checklist/standard profiles execute fixed assertions through the common
Task/Run/result path: `source-checklist`, Top 10 2025 source risk, ASVS 5.0 L1
pilot/source review, and full/fast WSTG 4.2 source/active HTTP checks. Checklist
inputs use `contractor.audit.assertions.v1`; standard selections pin the immutable
curated package version. Start derives fixed seeds without a model and preserves
the original requirement denominator. HTTP checks require an engagement and
selected materials. Model completion emits scoped conclusions through
`audit-check-results@1`, with optional evidenced problem Findings.

A profile pins ordinary Workflows, exact input mappings, policy and budgets.
OpenAPI preparation runs the existing deterministic Server operation and emits
a bounded ordinary work list. Common intake accepts separate operation Tasks;
the coordinator dispatches them through RunService. Web-scan accepts one Task
for the selected target.

Model-backed observation/tracing uses `audit-results@2` with a trusted exact
Task context and `workerCompletion: audit-check-results@1`. Workers record
bounded rationale, tested conditions, limitations, evidence and an optional
verdict. Runtime assembles ordinary JSON output; Server derives Task/hypothesis
origin. Exploration needs no hypothesis or Finding. Trace can explicitly
publish evidenced Project problems through the existing Finding tool.

The Nuclei and SQLMap Workflows use model-free tool Workers and retain an
ordinary, self-contained JSON operation report. Preparation and checking
outputs remain exact Audit materials after producer Runs are removed. There
is no special preparation phase, execution role or result ZIP in this path.
See the [Audit guide](../docs/guides/audits.md) for inputs and API continuation.

`audit-standards/` remains the operator-owned source of immutable curated
standard packages. Changing published standard content requires a new version.
Exact sources, licensing and regeneration instructions remain in the
[standard source guide](../docs/guides/audit-standard-sources.md).

The document-generation overlay workspace Workflows hydrate the exact `inputs/source` ZIP below a
private project-workspace root for every Stage and uses bounded filesystem/code
tools. It exports exact cumulative `workspace_state` plus checkpoint
`workspace_diff`; a later Stage imports the exact prior state revision. With no
imported state, the cumulative overlay is the canonical empty delta over the
hydrated sources. An unchanged Stage therefore exports an identity state and
empty diff. These are internal lineage artifacts retained for future explicit
Workflow composition; no join behavior or implicit workspace sharing exists.

`podman-python-check@1` is a separate opt-in **direct** workspace example. It
selects `podman_python_fixer@1`, fixes a small offline Python fixture and publishes
only an explicitly written JSON report. It neither exports nor imports overlay
state, mounts Skills nor downloads packages. See the [local Podman setup and
source ZIP recipe](../runtime/README.md#local-podman-workflow). Existing agents
need no policy changes; only agents advertising verified local/direct Podman
execution can accept this workflow.

The discovery Stages use `workspace_source_graph_analyst@1` and select all
eleven bounded structural operations. They wait for a Runtime Agent whose
positive capability includes the complete local Trailmark graph surface; there
is no automatic downgrade to shallow analysis.

Model authoring is split into three deployment sources:

| Change | Source |
| --- | --- |
| Upstream model, endpoint, API-key environment name, context and output capacity | `deploy/models/catalog/<model>.yaml` |
| Initial model for all roles and optional role overrides | `deploy/models/selection.yaml` |
| Planner, Worker, summarizer, Audit completion and compactor spending/generation budgets | `deploy/models/roles.yaml` |
| Whether and when a Worker produces a terminal summary | `agent-templates/*.yaml`, `spec.summarizer` |
| Whether a Workflow's Worker compacts its context and continues | `workflows/*.yaml`, agent `contextManagement` |

For a changed locked policy, increment its `policyVersion` and update the
referring manifest versions before extending the catalog lock.
Run `make model-config` after editing these deployment sources, then
`make test-config`. The generated `model-policies/*.yaml`,
`model-policies/models/*.yaml`, `model-selection.json` and
`deploy/litellm/litellm_config.yaml` must not be edited by hand;
`make model-config-check` (also in `make lint`) detects drift. The model-choice
catalog contains all deployed model-specific policies, so a Run or Eval can
select an exact policy without replacing a shared route. The initial selection
is a deployment bootstrap; it is not a mutable setting of an active Run.
E2E and optional Eval fixtures have their own policies outside this generator.

The default catalog uses `planner@1` and `worker@1` for normal execution,
`summarizer@1` for terminal summaries, `audit_completion_worker@1` for the
separately selected Audit completion example and `compactor@1` for Workflows
that select `compact@1`. Each model also has up to five exact model-specific
policies under `model-policies/models/`. The compactor always uses the default
model: `deploy/models/selection.yaml` cannot override it, and it is not part of
the model-choice catalog, because a Workflow pins its compactor policy. Aliases carry the
upstream model name, a model-profile digest and the role; changing the selection
never retargets an old alias. A changed profile produces new aliases. Keep a
profile deployed for as long as its pinned Runs may still execute. A requested
model identifier is not proof of a provider's observed weights revision.

The Planner and Worker policies both permit 200 model calls at temperature 0.6.
`planner@1` permits 2,500,000 cumulative provider-reported tokens, 16,384 output
tokens per response and 200 Worker calls; `worker@1` permits 25,000,000 cumulative
tokens, 32,768 output tokens per response and 200 tool calls. Cumulative token
budgets are independent of model context; a larger context can exhaust the
spending budget before the call budget. Model changes never raise these budgets.

The passthrough workspace Workflows pin `worker@1`.
`openapi-from-workspace-streamline@1` and `likec4-from-workspace-streamline@1`
retain the exact graph-backed workspace, artifact handoffs, cumulative state/diff
and output contracts of their respective `*-from-workspace@1` variants, with
`streamline@1` in every Stage. Their modeled Planners use `planner@1`;
their Workers use `worker@1`. The three OpenAPI workflow families retry validation
with `maxAttempts: 2`: the initial attempt and at most one retry for a retryable
failure. The retry preserves the effective Run policy; no production escalation
profile is published. Escalation fixtures remain under testdata.
Choose these when each semantic Stage benefits from explicit
ordered subtask decomposition; the passthrough variant remains the simpler
default when one Worker invocation can own the complete Stage objective.

The GLM-5.3-Flash profile pins an operating context of 1,000,000 tokens and
an output capability ceiling of 128,000 tokens; individual role output budgets
remain lower. Ordinary model-backed Worker templates enable a one-shot,
tool-free terminal summarizer at `contextWindowRatio: 0.8`: the threshold is
`min(floor(worker context * ratio), worker context - worker max output)`.
With the current Worker policy this is 800,000 provider-reported prompt tokens.
The check occurs after a completed normal model response, not before the next
request; context metadata does not guarantee prompt admission at the provider.
The summarizer has its own selected model and an 8,192-token output budget.
No cumulative summary threshold is configured in the default templates.

The summarizer completes the invocation; it does not compact history and resume
the same tool loop. Required output artifacts must already exist, and an invalid
or incomplete summary still fails the task. Its transcript projection remains
bounded to 512 KiB, with individual projected text bounded to 64 KiB. These byte
limits do not grow with the model window and do not prove that the projection
fits a smaller summarizer model's token window. Planner request/response bounds,
tool read sizes and request timeouts are separate absolute limits; see
[deployment](../docs/deployment.md#llm-gateway) and
[timeout configuration](../docs/operations/timeout-configuration.md).

The model-backed Audit Worker templates, including the
`audit_source_checker@1` completion variant, retain mandatory
`audit-results@2` submission and omit terminal summarization;
`audit-check-results@1` completion prohibits it. The two tool-runtime scan
templates use `scan@1` instead.
Planner context metadata does not enable a Planner summarizer.

Terminal summarizer instructions live in `instructions/terminal-summarizer.md`.
Templates select the file through `spec.summarizer.instructions.ref`; resolution
pins its exact text and digest into the Run snapshot. The file may contain at
most 8,000 Unicode characters (including whitespace and newlines); blank text
is invalid. Roughly 2,000 tokens is a planning estimate, not a guaranteed bound
for arbitrary text or tokenizers. Literal braces are preserved. Legacy templates
that omit this field retain the built-in instruction for compatibility.

V47-003 context admission has been removed: summarizer no longer charges one
token per byte or trims its request against that estimate. Context errors come
from the gateway. The earlier 512 KiB document projection remains in place.

A Workflow can instead let an isolated, ordinary Worker continue past its
context boundary. `workflows/taint_trace_from_workspace_compact.yaml` is the
opt-in example: its analyst binding selects `contextManagement: compact@1` with
`compactor@1` and `instructions/worker-compaction.md`. That branch replaces the
template's terminal summarizer for this binding only; the AgentTemplate and its
digest stay unchanged. When the measured prompt reaches
`min(floor(worker context * contextWindowRatio), worker context - worker max
output)` (ratio 0.75 by default), Runtime asks the tool-free compactor for one
summary of the oldest complete call groups, keeps the most recent groups
(`keepRecentGroups`, default 4), and continues the same subtask; the Worker can
read compacted groups again with `read_context_history`. A Workflow that uses
`compact@1` cannot also declare `on.summarized`, and shared sessions and Audit
completion Workers are rejected for now. See
[Workflows](../docs/spec/configuration/workflows.md#WFL-R91).

The local LiteLLM model and request timeouts are 240 seconds; the Worker client
timeout is 300 seconds. Planner execution retains its 30-minute timeout;
Stage preparation uses the separate scheduler operation budget. See
[timeout configuration](../docs/operations/timeout-configuration.md). Model context capacity, individual request timeouts and Stage
deadlines remain separate bounds. No aggregate Workflow/Run/Audit token budget
is introduced. Process-only policies under `configs/e2e` are independent of
these defaults.

Both current LikeC4 template families select the versionless logical
`skills/likec4` artifact for detailed DSL guidance while retaining mandatory
artifact, evidence, validation and completion rules in always-on instructions.
The package source is `skills/likec4/SKILL.md` plus its on-demand references. It
is packaged and published into the owner-scoped `skills/likec4` artifact by the
Skill Catalog path; configuration refers to the logical binding, not a
checked-in digest.

`skills/security-audit/` bundles Cloudflare's MIT-licensed source-first security
review methodology with fourteen on-demand references, its JSON report schema
and two inert Node.js validator sources. Provenance and Contractor-specific
capability requirements are recorded in its `assets/provenance.md`. It is seeded
into the owner Skill catalog without changing existing Workflow selections.
See [the Skill guide](../docs/guides/agent-skills.md) for selection and execution.

`security-analysis@1` is an opt-in single-Stage workflow for authorized HTTP
and Caido analysis. It requires the caller to provide `objective`, `target` and
`authorization_scope` string parameters, accepts one optional text context
artifact, and freezes one Markdown report. Its `caido_analyst@1` template pins
the versionless `skills/caido-stored-evidence` artifact and explicitly selects every HTTP and
Caido operation named by that package. `http_explorer@1` provides a reusable
HTTP-only template without the Caido Skill or adapter requirement.

Infrastructure is not embedded in those manifests. Before starting the
workflow, publish and bind a RuntimeConfig such as the secret-free
[`caido-analysis@1` example](../deploy/runtime-labels/runtime-configs.example.yaml),
create its write-only credential separately, and add the resulting `caido`
label to the Run. Omitting the label leaves the Stage waiting for a compatible
allocation configuration; it does not silently fall back to an unconfigured or
direct Caido client.

All limits remain finite and are enforced per Planner or Worker invocation.
Select the passthrough variant when one Worker can follow the complete Stage
contract directly; select Streamline when the Planner should decompose that
Stage into ordered subtasks for its one fixed logical Worker. Validate the
complete set from the repository root with:

```sh
contractor server config validate --root ./configs
```

The command also checks the sibling `managed-configs/` root without creating
it. Use `--managed-root` when Server has a different managed root. Offline
validation checks AuditProfile standard selections against bundled standards;
it rejects references that exist only in the database because it cannot
verify them before startup.

Manifest identity comes from `kind`, `metadata.name`, and `metadata.version`;
file names and nesting are only organizational. Instruction references are
paths relative to this directory.

The committed `catalog.lock.json` freezes shipped identities and their semantic
content ([CAT-R74](../docs/spec/configuration/catalog.md)). Use a new manifest
version or instruction path for changed content, then run:

```sh
go run ./cmd/contractor-server config lock --root ./configs
go run ./cmd/contractor-server config validate --root ./configs --require-catalog-lock
```

The offline lock command adds new identities and retains retired entries; it
refuses to replace an existing digest. Commit the manifest changes and lock
together. Startup verifies a present lock, and the release gate requires it.
Custom operator roots may remain unlocked. Runs, started Audits and Eval pins
still retain exact content digests. `audit-standards/` and `skills/` keep their
own create-only seed rules.

`llm-gateways/` contains immutable, non-secret endpoint descriptions. The
shipped `local-litellm@1` config uses the OpenAI-compatible `/v1` inference
path and a loopback-only HTTP LiteLLM management origin. Tokens and LiteLLM
admin keys are never valid fields in these manifests. An optional
`spec.failureSignatures` block declares the provider's exact
model-unavailable responses and permanent error codes for model recovery;
when omitted, the `openai-compatible@1` default (LM Studio unloads observed
through LiteLLM) applies, and the block joins the Gateway digest only when
declared.

Managed credentials bind that exact digest-bearing Gateway ref to an
owner-only admin-key file through the separate process bootstrap document
passed with `--llm-gateway-admin-bindings-file`. The binding document is not a
configuration resource, is never served by the public API, and must be updated
explicitly when the immutable Gateway digest changes.

Workflow `spec.executionConfig` contains reference-only defaults. A modeled
consumer resolves an exact `modelPolicy`, `llmGateway`, and optional non-secret
credential ID; per-Stage and per-Agent leaves override Workflow-wide defaults.
Run and Eval callers select an authored Workflow. Inline escalation alone
accepts `credential: null` to clear an inherited credential. The resolved bodies,
digests, origins, and credential IDs are pinned in the Run snapshot, while
token bytes are resolved only when constructing a Planner client or Worker
allocation.

Escalation execution settings are inline in the owning Workflow. Standard YAML
anchors and aliases reuse a block within that document, for example
`executionConfig: &strong_execution` at its first use and
`executionConfig: *strong_execution` at another compatible transition. There is
no standalone ExecutionConfig manifest or cross-file profile lookup. The process
example is [`e2e/workflows/escalating_copy.yaml`](e2e/workflows/escalating_copy.yaml).

For local development only, `CONTRACTOR_LLM_GATEWAY_TOKEN` and
`CONTRACTOR_PLANNER_LLM_GATEWAY_TOKEN` bootstrap the fixed credential IDs
`development-worker` and `development-planner` against `local-litellm@1`.
The shipped `workflows/` manifests select these IDs explicitly and separately
for Workers and modeled Planners. They do not override the Gateway URL or model
selected by configuration. If the corresponding token is absent, Run creation
fails while resolving the selected credential, before a Stage or allocation is
started. Deployments using different credentials select those identities in their own
Workflow definitions.

`examples/` contains Memory-enabled Router and Streamline examples, plus
copyable multi-Stage, bounded-retry, single-Worker
`streamline@1`, and multi-Worker `router@1` Workflow manifests. They are
intentionally outside `workflows/`, so they document supported shapes without
changing the default end-to-end fixture.

### Changing the active model selection

After deploying profiles and their Gateway routes, use **Operations → Settings →
Models** to change the default or a role override for new executions. PostgreSQL
stores this selection with a revision; `selection.yaml` initializes it only on
the first startup. Restarting or regenerating files does not reset a saved
selection. Removing a selected model makes startup fail until a valid deployment
and saved selection agree. The UI does not install provider credentials or
provision remote routes.

Workflow definitions can select exact model-specific policies for Run and Eval experiments,
including `workers.summarizerModelPolicy` for a summary-enabled Worker. An Audit
pins defaults at draft creation; a prepared Eval retains its binding through
submission. Run history shows requested aliases and exact policy digests per
consumer; provider weight revisions remain unknown unless observed. Context
capacity comes from the profile, role spending/output budgets from `roles.yaml`,
and the terminal-summary trigger and instructions from the AgentTemplate.

Worker model selection belongs to Workflow `spec.executionConfig`. AgentTemplate
contains instructions, tools, Skills and optional summarizer behavior. Every
modeled binding needs an effective `modelPolicy`; terminal summarization also
needs `summarizerModelPolicy`. Use Workflow-wide defaults for shared selections
and Stage/agent selections for exceptions, for example a small model for a
bounded secret review Worker and a stronger model for a deeper source review.
YAML anchors can reuse these selections within the same Workflow document.

Planner compaction follows the same split: select its `compactorModelPolicy`
in Workflow `executionConfig.planner` or the Stage's Planner execution selection.
`plannerContextManagement` contains the strategy, instructions and thresholds.
It contains no model selection; summary continuations use the same grammar.

Investigation templates select common `security-findings@1` `fact` alongside their
source/HTTP/trace finding facade. Discovery and dependency analysis, source review,
Audit source/operation/WSTG investigation, taint analysis and callback investigation
can preserve reusable observations while reading primary evidence. Instructions
select facts useful for later checks, preserve scope and uncertainty, allow zero
facts, and keep normal reports and Audit evidence/results. Facts require an active
Project; a publication denial is a reported limitation. Audit publication precedes
final result submission. Builders, validators, scanners, hypothesis generation and
existing-Finding review retain their existing roles. See FND-R242.
