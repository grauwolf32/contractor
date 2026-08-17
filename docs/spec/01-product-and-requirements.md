# Product and requirements

## 1. Product definition

Contractor is a local-first agent orchestration product for security-oriented
source analysis. A user selects a workflow, points it at a project directory,
and optionally supplies an OpenAPI document, a vulnerability seed, a natural
language prompt, or a live target URL. The product coordinates model-driven
agents over a confined view of the source tree and persists intermediate and
final artifacts for reuse by later tasks or later invocations.

The primary executable is a finite command-line run, not a long-running API
service. The secondary executable is a local explorer that reads prompts, task
templates, skills, workflow topology, evaluation results, and review comments.

## 2. Product goals

A conforming implementation MUST provide these capabilities:

1. Generate an OpenAPI description from source code and validate it.
2. Enrich an existing OpenAPI description from deeper source analysis.
3. Build and validate a LikeC4-style architecture model.
4. Trace OpenAPI operations into source, annotate data-flow points in an
   isolated overlay, and report supported vulnerabilities.
5. Run breadth-first vulnerability discovery directly over source code.
6. Verify findings statically and, when explicitly configured, assess them
   against a live target through HTTP and optional proxy/code-execution tools.
7. Compose the above primitives into resumable multi-stage assessment
   workflows.
8. Route a free-form user request to specialist agents.
9. Persist enough artifacts, metrics, and evaluation data to inspect,
   reproduce, compare, and resume runs.

## 3. Actors and trust zones

| Actor or system | Trust | Permitted interaction |
|---|---|---|
| CLI user | trusted operator | Chooses workflow, source root, output location, model alias, seeds, and destructive reset/resume flags. |
| Target source tree | untrusted content | Read through a rooted virtual filesystem. Writes occur only in an in-memory overlay unless a specifically read/write workflow is used. |
| Model backend | external processor | Receives prompts, tool schemas, selected source excerpts, memories, and artifacts. It MUST NOT receive unrestricted host filesystem access. |
| Live target | explicitly authorized external system | Contacted only by exploitability/HTTP tools when a target URL is configured. |
| Artifact store | trusted local persistence | Stores task outputs, memories, HTTP sessions/bodies, findings, verifications, overlays, and seeds by application/user key. |
| Code-execution sandbox | hostile-workload boundary | Ephemeral container with source mounted read-only, writable scratch space, explicit lifetime cleanup, and bounded output. |
| Observability backend | optional external processor | Receives spans when enabled. Failure MUST degrade to no-op telemetry. |
| Explorer browser | local user interface | Reads package metadata and eval results; only comment endpoints mutate a local review database. |

## 4. Functional requirements

### 4.1 Run setup

- The CLI MUST require an existing project directory and resolve it to a
  canonical absolute path.
- A requested folder scope MUST resolve to an existing directory inside that
  project. `/` denotes the whole project.
- Artifact persistence MUST be namespaced by canonical project path so two
  same-named projects in different directories cannot reuse one another's
  state.
- A supplied seed artifact MUST be UTF-8 text. Binary input seeds are rejected
  at the CLI boundary.
- `reset artifacts` and `resume` are mutually exclusive.
- A prompt-driven workflow MUST receive a non-empty prompt. Interactive mode
  may collect it; non-interactive mode must reject a missing prompt.

### 4.2 Workflow execution

- Each registered workflow MUST accept one immutable run context containing the
  project path, scoped folder name, model selection, timeout, application/user
  identity, artifact service, source filesystem, optional artifact/prompt, and
  optional checkpoint path.
- A workflow MUST emit exactly one `workflow_started` and one
  `workflow_finished` event around its implementation, even when it fails.
- Workflow cleanup MUST run before `workflow_finished`. Cleanup failures are
  logged and MUST NOT replace the workflow's original result or exception.
- Lifecycle event consumers are telemetry/UI integrations. Their ordinary
  failures MUST NOT abort work; cancellation MUST propagate.
- Fan-out jobs may explicitly opt into failure isolation. Verification stages
  whose persisted verdicts are authoritative MUST fail rather than silently
  report an empty result.

### 4.3 Task communication

- Tasks MUST communicate through named artifacts, not process globals or
  implicit shared variables.
- Every planner-driven task publishes `result`, `summary`, and `records` under
  an invocation-specific artifact key.
- A downstream task declares artifact names; the runtime loads their text and
  injects them into both template rendering and an inbox memory namespace.
- Missing declared inputs are represented as empty content rather than giving a
  task access to an arbitrary fallback path.

### 4.4 Model-driven work

- Workflows assemble agents but do not implement analysis logic themselves.
- Planner-driven tasks use a planner that decomposes work into strict-state
  subtasks and delegates each subtask to a worker agent.
- Direct-agent workflows may bypass the planner when one operation maps cleanly
  to one agent invocation.
- Prompts and task bodies MUST be versioned manifests with an active version and
  explicit file mapping. Evaluation or environment configuration may pin a
  non-active version.
- Context limits, tool-result retention, rate limits, required terminal tool
  calls, and structured-output validation MUST be enforced outside model prose.

### 4.5 Persistence and resume

- Artifact keys MUST reject traversal segments and fan-out-derived segments
  MUST use collision-resistant portable encodings.
- Checkpoints MUST record workflow ownership, task reference, task identity,
  template version, and published artifact names.
- A checkpoint entry is restorable only when workflow ownership and template
  key/version match and every artifact expected from the invocation's current
  output key is present. Compatibility restore is existence-only, so even an
  empty text part counts; authoritative downstream stages apply their own
  stronger content/freshness postconditions.
- Checkpoint updates MUST be atomic and merge-safe across threads and processes.
- A freshly executed verifier task MUST update its persisted verification; a
  checkpoint-restored verifier may reuse its already-persisted verification.

### 4.6 User-visible output

- After a successful run, every non-internal artifact MUST be exported to the
  output directory while preserving its logical key hierarchy and whether its
  payload is text or binary.
- Internal memory artifacts MUST be excluded from the exported deliverable
  listing.
- Metrics MUST be appended as one flat JSON object per line.
- The live UI MUST remain active until `workflow_finished`; task-run or finding
  completion events are not terminal for the overall workflow.

## 5. Security requirements

1. All source paths are virtual paths rooted at the selected project.
2. Traversal and symlink escapes MUST appear nonexistent. Directory enumeration
   MUST not disclose escaped symlink targets.
3. Overlay recursive moves MUST reject equal or descendant destinations.
4. Explorer resource identifiers and resolved paths MUST be confined to their
   package roots, including after percent-decoding and symlink resolution.
5. Explorer-rendered dynamic strings MUST use text nodes or equivalent escaping;
   Markdown links MUST allow only explicitly safe schemes.
6. HTTP request identities MUST never be reused after a body may have been
   persisted, including cancellation and final-save failure paths.
7. Code execution MUST occur in an ephemeral external sandbox, never in the
   host process, and source must be mounted read-only.
8. Vulnerability and verification record loading MUST reject ambiguous duplicate
   logical names rather than silently overwrite them.
9. Artifact/export paths MUST preserve the artifact store's text/binary type.

Credential export and telemetry redaction policy are deployment-sensitive. The
runtime must expose a single persistence boundary where a deployment can apply
its chosen policy consistently.

## 6. Reliability and performance requirements

- Model requests have a configurable timeout; task attempts may additionally
  have a wall-clock timeout.
- Retries use bounded attempt counts. Successful iterations accumulate across
  attempts; a failed attempt does not erase earlier successful iterations.
- Filesystem walks, read output, graph paths, graph results, HTTP history,
  response previews, summarizer records, and retained heavy tool outputs are
  bounded by configuration.
- Path-parallel trace work uses a semaphore and isolated overlays. Completed
  sibling work MUST be merged/persisted even when another sibling fails.
- Shared HTTP artifact namespaces MUST serialize state transitions across client
  instances within one event loop.
- Optional integrations (observability, dense retrieval, Caido, graph tooling)
  MUST have a defined disabled or unavailable behavior.

## 7. Non-goals

- Contractor is not a general remote code execution service.
- It does not guarantee that model-generated findings are correct; verification
  and evaluation contracts measure and improve that behavior.
- It is not a browser automation platform. HTTP probing and proxy automation are
  supported; DOM/visual browser behavior is outside the core contract.
- It does not prescribe one model provider or one source language.
- It does not mutate the target source tree during trace annotation; changes are
  overlay artifacts unless a workflow deliberately supplies read/write tools.
