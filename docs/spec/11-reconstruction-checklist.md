# Reconstruction checklist

## 1. Definition of a reconstruction

A reconstruction is conforming when it can execute the same public workflows
over the same project and seed inputs, expose the same durable artifacts and
interfaces, enforce the same safety/state invariants, and consume the same
versioned prompt/task/skill assets. It does not need to use the original source
language, agent SDK, web framework, or storage libraries.

Model output is nondeterministic, so byte-identical prose is not required.
Control flow, schemas, namespaces, validation, limits, and observable failure
behavior are required.

## 2. Recommended build order

### Phase 1 — contracts and data types

- Implement typed records for workflow context, task template, queued task,
  task/subtask state, task metrics, artifacts, tool envelopes, vulnerability
  findings/verifications, HTTP sessions, checkpoints, and events.
- Implement deterministic JSON/YAML serialization and structured validation.
- Freeze the public names in file 12 as persistence/API identifiers.
- Add fixture builders and schema round-trip tests before adding model calls.

Exit condition: every record can be created, validated, serialized, rejected on
invalid input, and round-tripped without losing optional-versus-empty meaning.

### Phase 2 — virtual filesystem and artifacts

- Implement the root-confined source filesystem and virtual path normalizer.
- Implement bounded read/list/walk/glob/grep operations.
- Implement the durable artifact service with application/user scoping and
  text/binary parts.
- Implement memory/inbox namespaces and artifact-key validation.
- Implement copy-on-write overlay operations, patch/diff serialization, fork,
  and merge.

Exit condition: the full traversal/symlink/type-change/merge test corpus passes,
and artifacts persist across a process restart.

### Phase 3 — event and model runtime adapters

- Implement the event taxonomy and best-effort asynchronous dispatcher.
- Implement a model-gateway adapter with streaming turns, tool calls, usage
  accounting, timeout, sampling, and structured output.
- Implement session state and one direct-agent runner.
- Implement callbacks/plugins for rate limits, token limits, tool policy,
  context compaction, heavy-result elision, metrics, and cleanup.

Exit condition: a deterministic fake model can drive a direct agent through
multiple tools to a terminal structured result while events and metrics remain
correlated.

### Phase 4 — planner-driven task engine

- Implement task manifest/version resolution and template rendering.
- Implement planner strict state and subtask tools.
- Build a fresh planner/worker/session for each attempt.
- Add iteration accumulation, retries, timeout, standardized publication,
  artifact injection, skill injection, observations, and checkpoints.
- Guarantee event ordering and checkpoint compatibility rules from file 03.

Exit condition: scripted fake planners cover success, retry, partial progress,
invalid transitions, restored tasks, and exhausted attempts with the exact
artifact/event outcomes.

### Phase 5 — deterministic tool layer

- Add source search, symbol/definition lookup, annotations, and call graph.
- Add memory, artifact-pool, task-control, OpenAPI, findings, verification, and
  LikeC4 tools.
- Add HTTP persistence/retries/body storage and optional Caido adapter.
- Add optional RAG and code-execution adapters with safe unavailable behavior.

Exit condition: every tool has a standard envelope, bounded result, explicit
agent binding, failure tests, and persistence ownership documented in files 06
and 07.

### Phase 6 — agent, task, and skill assets

- Recreate the version manifests and all versions in file 12.
- Implement the agent factories with their exact tool sets, namespace rules,
  callbacks, input/output schemas, and limits from file 04.
- Recreate task templates and declared artifact/skill inputs.
- Recreate skill indexes and reference documents while preserving their names.

Exit condition: every active/pinned asset resolves; static introspection can
enumerate it; and a fake-model smoke run proves each factory is constructible
without tool-name collisions.

### Phase 7 — workflows

- Implement primitive OpenAPI, LikeC4, trace, vulnerability, exploitability,
  verification, and router workflows first.
- Implement graph/parallel/post-diff variants after overlay and path-key tests.
- Implement composite workflows and sweeps last.
- Register exactly the public keys in file 12 and load sibling configuration.

Exit condition: workflow assembly tests match the stage/artifact graphs in file
05 and each workflow emits one start/finish pair under success and failure.

### Phase 8 — CLI, metrics, and explorer

- Implement CLI validation, store derivation, run context, reset/resume, export,
  plain rendering, and live rendering.
- Implement static explorer routing and safe static-file resolution.
- Implement metadata readers, static AST-equivalent introspection, eval reader,
  comments database, and safe browser UI.

Exit condition: file 08 interface/security tests pass and the explorer can
inspect the reconstructed assets without importing/running model code.

### Phase 9 — deployment and evaluations

- Package the non-root runtime image and separate sandbox image.
- Configure the model proxy, optional pgvector, observability, and local model
  helper.
- Port fixtures/scorers and validate `eval/v1` output and discovery.

Exit condition: a clean environment can start required services, run a small
workflow, restart and inspect its artifacts, and display an isolated evaluation
run in the explorer.

## 3. Component completion checklist

### 3.1 Core runtime

- [ ] One immutable workflow context is used throughout a run.
- [ ] Workflow start/cleanup/finish behavior survives exceptions.
- [ ] All event consumers are best-effort except cancellation propagation.
- [ ] Queue/task/attempt/session/agent/subtask/tool-call identities do not
      collapse into one identifier.
- [ ] Task publication always writes `result`, `summary`, and `records`.
- [ ] Retries retain completed iterations without reusing corrupt attempt state.
- [ ] Checkpoints are ownership/version/artifact compatible and atomic.
- [ ] Parallel workflows preserve completed sibling output on partial failure.

### 3.2 Safety boundaries

- [ ] Rooted paths reject lexical traversal and canonical/symlink escape.
- [ ] Overlay moves reject self/descendant destinations.
- [ ] Artifact and fan-out keys are portable and collision resistant.
- [ ] Live HTTP is impossible without explicit target configuration.
- [ ] HTTP IDs cannot be reused after a body could have been stored.
- [ ] Code executes only in ephemeral external isolation with read-only source.
- [ ] Duplicate logical vulnerability records fail loudly.
- [ ] Explorer IDs, static assets, references, and eval IDs are doubly confined.
- [ ] Browser-rendered dynamic values and links are safe.

### 3.3 Operational behavior

- [ ] Optional Langfuse, pgvector, Caido, graph, GitLab, and sandbox services have
      explicit disabled/unavailable results.
- [ ] All configured size/count/time/concurrency limits bind in tests.
- [ ] Text and binary artifacts preserve their type through export.
- [ ] Metrics are append-only, flat, serializable, and redacted at persistence.
- [ ] The live terminal remains active through the workflow terminal event.
- [ ] Runtime and sandbox images do not require root.

## 4. Compatibility rules

### 4.1 Stable identifiers

The following are compatibility-sensitive and MUST NOT be renamed without a
migration and reader aliases:

- public workflow keys;
- agent/task/skill names and version IDs;
- task refs stored in checkpoints;
- artifact namespaces and standard suffixes;
- event type strings;
- vulnerability/verification source namespaces;
- HTTP client namespace and request IDs;
- explorer route shapes and `eval/v1` fields.

Implementation class/function/module names are not stable unless surfaced in
the explorer inventory; the explorer may report the new implementation name
while retaining the stable public key.

### 4.2 Prompt and task evolution

- Never edit the semantics of a published version in place when reproducibility
  matters; add a version and move `active` deliberately.
- A pinned environment/evaluation version overrides `active` only for that
  asset.
- Checkpoints created for another task version do not restore.
- A new task input artifact or skill is a template-interface change and should
  normally receive a new task version.

### 4.3 Persistent schema evolution

New readers SHOULD accept old additive schemas. Writers emit only the current
schema. A breaking record change requires:

1. a schema/version discriminator;
2. a pure, tested migration or explicit incompatibility outcome;
3. atomic replacement that preserves the source on failure; and
4. explorer/export compatibility where the record is user-visible.

Unknown checkpoint formats are treated as absent, not partially restored.
Unsupported evaluation schemas are ignored by discovery. Corrupt authoritative
verification artifacts fail the owning workflow rather than becoming an empty
finding set.

## 5. End-to-end reconstruction walkthrough

Use this language-neutral scenario to validate the whole system:

1. Create a small project with two HTTP operations, a service/data layer, one
   intentional tainted-data sink, and at least one symlink pointing outside the
   project.
2. Run `oas_build`; confirm project/dependency information and a validated
   OpenAPI artifact are persisted and exported.
3. Run a trace graph variant with resume enabled; confirm path-derived keys,
   overlay patch/diff output, vulnerability records, and checkpoint entries.
4. Interrupt after one completed task, restart with resume, and prove only a
   compatible complete task is restored.
5. Run verification; prove every input finding receives an authoritative
   verification or the workflow fails.
6. Configure a disposable authorized target and run assessment; confirm HTTP
   bodies use unique persisted IDs and live-target evidence is namespaced.
7. Run a vulnerability sweep with one forced worker failure; confirm configured
   isolation and completed siblings.
8. Export artifacts, restart the explorer, inspect agents/tasks/skills/workflow
   graph, add/update/delete a source comment, and view a small `eval/v1` result.
9. Attempt project, artifact, explorer, overlay, and symlink escapes; all must be
   rejected without disclosing host content.
10. Disable every optional service and repeat a non-network static workflow;
    it must remain usable with documented fallbacks.

## 6. Release gate

A release is ready only when:

- all public inventory entries resolve and all workflow assembly tests pass;
- all deterministic safety and persistence suites pass on at least one
  case-sensitive and one case-insensitive-path simulation;
- static checks and the complete non-eval suite pass from a clean checkout;
- reference images build and run as non-root;
- one agent, one task, and one pipeline evaluation emit valid isolated
  `eval/v1` results; and
- the implementation, specifications, tests, manifests, and deployment defaults
  describe the same current behavior.

The final item is a hard gate: a code change that alters a stable contract
without updating this directory is incomplete.
