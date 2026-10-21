# Contractor configuration

For new Runs and Audits, use the versioned [working Memory catalog](MEMORY.md).
It maps all 20 roles and their Workflow/AuditProfile references to explicit
Memory selections. Superseded exact selectors are removed from this catalog;
new requests must select an available version explicitly.

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

- `openapi-from-workspace@7` and `likec4-from-workspace@7` for four-Stage local
  graph-backed analysis and document generation;
- `openapi-from-workspace-streamline@2` and `likec4-from-workspace-streamline@4`
  for the same document contracts with a modeled single-Worker Planner;
- `openapi-from-analysis@4` and `likec4-from-analysis@5` when the caller already
  has exact dependency and project reports;
- `security-analysis@4` and `taint-trace-from-workspace@4` for their explicit
  focused analysis contracts;
- `audit-source-check@1` for bounded task sets in `source-checklist@1`;
- `audit-openapi-operation-observe@1` for graph-backed operation analysis in
  the limited `openapi-operation-observe@1` comparison variant;
- `audit-openapi-operation-trace@1` for graph-backed analysis and finding proposals
  in `openapi-operation-trace@1`, plus `findings-review@2` for a supplied collection
  and `source-findings-review@1` for exact source-backed proposal review;
- `audit-openapi-nuclei-scan@1` and `audit-openapi-sqlmap-scan@1` for the
  `openapi-nuclei-scan@1` and `openapi-sqlmap-scan@1` AuditProfiles;
- `audit-top10-source-risk@1` and `audit-asvs-source-verification@1` for the
  standard-specific AuditProfiles, including `owasp-top10-2025-source-risk@1`;
- `artifact-copy@2` as the ordinary artifact smoke fixture;
- `podman-python-check@2` as the explicitly selected offline Podman execution fixture.

The passthrough stages of the workspace and analysis-based OpenAPI/LikeC4
Workflows, security analysis, taint tracing and both findings-review Workflows
retain terminal summaries and continue once with `streamline@1`. Each
continuation has a 30-minute allowance capped by the Server's Planner timeout,
and retains the original task, Worker selection, namespace, recorded artifacts
and selected Memory. Normal completion follows the original Stage graph.
Existing Streamline variants already use that Planner. Bounded Audit Workers
have no terminal summarizer; the short artifact-copy and Podman smoke fixtures
retain their direct passthrough completion and failure checks.

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
[MEMORY.md](MEMORY.md) lists the retired identities and their active successors;
it does not define aliases. Repeat is blocked with `workflow_unavailable` when
the original exact Workflow is absent. Removing a bundled resource does not
remove copies published into a separate managed or operator catalog. Frozen
instruction-evaluation inputs retain their own exact catalog under `tests/eval`.

The source-checklist and model-backed OpenAPI review profiles are bounded,
one-round and non-certifying. Source-backed profiles require an exact source
ZIP alongside their checklist or OpenAPI input. Their model-backed Audit
workers use `audit-results@2`: `read_audit_task` reads the pinned task set, and
`submit_check_result` records validated items incrementally. Those profiles
pin `workerCompletion: audit-check-results@1` to their Worker.
Runtime publishes the complete canonical ZIP after normal completion; tool
receipts report collection progress, not published artifacts. Missing results
or publication failures fail the child Run. The checklist batches up to two
compatible items; OpenAPI retains `batchSize: 1`.

The `openapi-nuclei-scan@1` and `openapi-sqlmap-scan@1` profiles instead select
`audit-openapi-nuclei-scan@1` and `audit-openapi-sqlmap-scan@1`. They take exact
`openapi` and `settings` inputs and bind the `audit-nuclei-scan@1` or
`audit-sqlmap-scan@1` template. These are model-free `tool@1` Workers using
`scan@1`; they do not select `audit-results@2` or `workerCompletion`. See the
[Audit guide](../docs/guides/audits.md) for the scan path.

The model-backed Audit Workflows require the trusted completion contract
supplied by an Audit. They cannot be launched as ordinary Runs with
user-supplied task artifacts.
All bundled Audit Workflow, AgentTemplate and AuditProfile identities start at
`@1`; distinct scenarios have distinct names. Toolset and wire format versions
keep their actual protocol versions, including `audit-results@2`.

`openapi-operation-observe@1` selects `audit_openapi_operation_observer@1`, with
all eleven `code-analysis@1` operations and bounded filesystem reads. Its
read-only tools inspect an overlay workspace hydrated from the exact Audit
`source` input. The existing task/execution-manifest and result ZIP contracts
are retained. Graph coverage and source evidence inform the trace; the current
coverage key remains `operation-resolution`. This limited comparison variant
does not export annotations or propose findings. Its separate name distinguishes observation-only tracing from the finding producer.
A Runtime must advertise local workspace and graph capabilities to run this profile.

`openapi-operation-trace@1` selects `audit_openapi_operation_tracer@1`, adding
`security-findings-code@1.finding` and `text-artifacts@1.write_text_artifact` with
`findingConfirmation: human-required`. It preserves the operation inventory and
prohibits active checks. Findings do not turn `operation-resolution` into full
trace coverage. `findings-review@2` independently reads a pinned collection with
`list_findings` and publishes an analysis report; it does not confirm findings.

`audit-standards/` is the operator-owned source for immutable curated standard
packages. Each immediate package directory contains only a strict
`standard.json`; Server startup validates and canonicalizes the complete set
before create-only seeding into the protected owner catalog. AuditProfiles name
only `(scheme, version)`. Audit start resolves and retains the exact package
revision and license provenance; changing content under an existing identity
is fatal drift, so changed content must use a new version.

`owasp-asvs-5-0-l1-source-review@1` selects all 70 ASVS 5.0.0 Level 1
requirements from package edition `5.0.0-l1-source.1`, with the five-requirement pilot available separately as
`owasp-asvs-5-0-l1-source-pilot@1`. `owasp-wstg-4-2-source-review@1` is a
separate source-review preset with 94 active WSTG 4.2 scenarios. Both use
`audit-standard-source-review@1`; live-dependent checks retain gaps instead of
claiming successful active tests. Exact sources, licensing and regeneration
instructions are in [the standard source guide](../docs/guides/audit-standard-sources.md).

`owasp-wstg-4-2-active-http@1` is the separate live HTTP preset, using edition
`4.2-http.1` and `audit-wstg-active-http@1`. It requires a context brief, target,
authorization scope and active-item approval; it does not require source code.
Each of its 94 items has one attempt, and unavailable capabilities remain gaps.

`owasp-wstg-4-2-fast-source-review@1` and
`owasp-wstg-4-2-fast-active-http@1` are separate first-pass presets with 16
selected scenarios, one attempt per check and a four-hour default deadline.
They preserve full-preset evidence and approval rules. Their `4.2-fast-source.1`
and `4.2-fast-http.1` package editions make the excluded 78 scenarios explicit.

The document-generation overlay workspace Workflows hydrate the exact `inputs/source` ZIP below a
private project-workspace root for every Stage and uses bounded filesystem/code
tools. It exports exact cumulative `workspace_state` plus checkpoint
`workspace_diff`; a later Stage imports the exact prior state revision. With no
imported state, the cumulative overlay is the canonical empty delta over the
hydrated sources. An unchanged Stage therefore exports an identity state and
empty diff. These are internal lineage artifacts retained for future explicit
Workflow composition; no join behavior or implicit workspace sharing exists.

`podman-python-check@2` is a separate opt-in **direct** workspace example. It
selects `podman_python_fixer@2`, fixes a small offline Python fixture and publishes
only an explicitly written JSON report. It neither exports nor imports overlay
state, mounts Skills nor downloads packages. See the [local Podman setup and
source ZIP recipe](../runtime/README.md#local-podman-workflow). Existing agents
need no policy changes; only agents advertising verified local/direct Podman
execution can accept this workflow.

The discovery Stages use `workspace_source_graph_analyst@3` and select all
eleven bounded structural operations. They wait for a Runtime Agent whose
positive capability includes the complete local Trailmark graph surface; there
is no automatic downgrade to shallow analysis.

Model authoring is split into three deployment sources:

| Change | Source |
| --- | --- |
| Upstream model, endpoint, API-key environment name, context and output capacity | `deploy/models/catalog/<model>.yaml` |
| Initial model for all roles and optional role overrides | `deploy/models/selection.yaml` |
| Planner, Worker, summarizer and Audit completion spending/generation budgets | `deploy/models/roles.yaml` |
| Whether and when a Worker produces a terminal summary | `agent-templates/*.yaml`, `spec.summarizer` |

Run `make model-config` after editing these deployment sources, then
`make test-config`. The generated `model-policies/*.yaml`,
`model-policies/models/*.yaml`, `model-selection.json` and
`deploy/litellm/litellm_config.yaml` must not be edited by hand;
`make model-config-check` (also in `make lint`) detects drift. The model-choice
catalog contains all deployed model-specific policies, so a Run or Eval can
select an exact policy without replacing a shared route. The initial selection
is a deployment bootstrap; it is not a mutable setting of an active Run.
E2E and optional Eval fixtures have their own policies outside this generator.

The default catalog uses `planner@2` and `worker@2` for normal execution,
`summarizer@1` for terminal summaries and `audit_completion_worker@1` for the
separately selected Audit completion example. Each model also has up to four exact
model-specific policies under `model-policies/models/`. Aliases carry the
upstream model name, a model-profile digest and the role; changing the selection
never retargets an old alias. A changed profile produces new aliases. Keep a
profile deployed for as long as its pinned Runs may still execute. A requested
model identifier is not proof of a provider's observed weights revision.

The Planner and Worker policies both permit 200 model calls at temperature 0.6.
`planner@2` permits 2,500,000 cumulative provider-reported tokens, 16,384 output
tokens per response and 200 Worker calls; `worker@2` permits 25,000,000 cumulative
tokens, 32,768 output tokens per response and 200 tool calls. Cumulative token
budgets are independent of model context; a larger context can exhaust the
spending budget before the call budget. Model changes never raise these budgets.

The passthrough workspace Workflows pin `worker@2`.
`openapi-from-workspace-streamline@2` and `likec4-from-workspace-streamline@4`
retain the exact graph-backed workspace, artifact handoffs, cumulative state/diff
and output contracts of their respective `*-from-workspace@7` variants, with
`streamline@1` in every Stage. Their modeled Planners use `planner@2`;
their Workers use `worker@2`. The three OpenAPI workflow families retry validation
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
from the gateway. The earlier 512 KiB document projection remains in place;
compactification is not an implemented strategy.

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

`security-analysis@4` is an opt-in single-Stage workflow for authorized HTTP
and Caido analysis. It requires the caller to provide `objective`, `target` and
`authorization_scope` string parameters, accepts one optional text context
artifact, and freezes one Markdown report. Its `caido_analyst@3` template pins
the versionless `skills/caido` artifact and explicitly selects every HTTP and
Caido operation named by that package. `http_explorer@3` provides a reusable
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

Until the first release this catalog is pre-production: a shipped manifest or
instruction may change in place without a new `metadata.version`, and Server
loads the new content under the same identity at its next start
([CAT-R74](../docs/spec/configuration/catalog.md)). Runs, started Audits and
Eval pins keep the digests of what they used, so compare digests rather than
`name@version`. `audit-standards/` and `skills/` are seeded create-only and keep
their own rules: changed standard content needs a new version, and a changed
Skill source never replaces an owner's existing Skill artifact (Server reports
`seed_drift`).

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
`POST /v1/runs` accepts the same shape as an override, with
`credential: null` as the only explicit clear operation. The resolved bodies,
digests, origins, and credential IDs are pinned in the Run snapshot, while
token bytes are resolved only when constructing a Planner client or Worker
allocation.

`execution-configs/` is the catalog location for immutable Stage-local
escalation patches; the default catalog currently ships none. A runnable
example is under [`e2e/execution-configs/`](e2e/execution-configs/). A
Workflow `failed` or `interrupted` action may select one exact profile ref, or
declare the same `planner`/`agents` patch inline. The loader resolves and
validates every possible variant, and Run creation reapplies it over that
Run's base overrides before pinning the complete effective configuration.
Credential `null` is an explicit clear in this patch shape.

For local development only, `CONTRACTOR_LLM_GATEWAY_TOKEN` and
`CONTRACTOR_PLANNER_LLM_GATEWAY_TOKEN` bootstrap the fixed credential IDs
`development-worker` and `development-planner` against `local-litellm@1`.
The shipped `workflows/` manifests select these IDs explicitly and separately
for Workers and modeled Planners. They do not override the Gateway URL or model
selected by configuration. If the corresponding token is absent, Run creation
fails while resolving the selected credential, before a Stage or allocation is
started. Deployments using different credentials must override these
reference-only selections in the Run `executionConfig` or publish their own
Workflow versions.

`examples/` contains Memory-enabled Router and Streamline examples, plus
copyable multi-Stage, bounded-retry, single-Worker
`streamline@1`, and multi-Worker `router@1` Workflow manifests. They are
intentionally outside `workflows/`, so they document supported shapes without
changing the default end-to-end fixture.
