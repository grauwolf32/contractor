# System architecture

## 1. Logical layers

```text
Operator
  |
  v
CLI boundary -----> local live UI / JSONL metrics
  |
  v
Workflow registry and workflow context
  |
  +----> sequential task runtime ----> planner ----> worker agents
  |                                      |              |
  |                                      +------ tools -+
  |
  +----> direct agent runtime ----------> specialist agent + tools
  |
  +----> nested/sub-workflows
  |
  v
Artifact service <----> memory, inbox, task outputs, findings, HTTP state
  |
  +----> exported deliverables

Source filesystem ---> rooted sandbox ---> optional memory overlay/forks
Model gateway <------ agent runtime
Optional systems: observability, pgvector, Caido, call graph, LikeC4,
                  containerized code execution
```

The design has five strict separation rules:

1. The CLI constructs a context; it does not contain workflow logic.
2. Workflows assemble stages; they do not implement agent reasoning.
3. The task runtime owns retries, sessions, state, publication, events, and
   checkpoints; agents do not.
4. Agents can affect the world only through their declared tools.
5. Tasks exchange durable artifacts and memories, never hidden in-process data.

## 2. Top-level components

| Component | Responsibility | Stable boundary |
|---|---|---|
| CLI adapter | Validate options, select workflow, construct source/artifact services, drive run, export artifacts. | Options and exit/error behavior in §8. |
| Workflow registry | Map public workflow keys to constructors. | Exact key catalog in §5 and §12. |
| Workflow base | Emit lifecycle events, persist optional seed, provide artifact-exists/skip/failure-isolation helpers. | `WorkflowContext` and lifecycle contract. |
| Task runtime | Queue versioned templates, render inputs, inject skills/artifacts, build fresh planner/worker attempts, publish outputs, checkpoint. | Models and state machine in §3. |
| Direct-agent runtime | Create a session and stream one specialist agent to completion without the streamline planner. | Used by router and direct trace variants. |
| Agent factory layer | Bind model, prompt version, namespace, filesystem, tools, callbacks, schemas, and limits. | Agent catalog and factory contract in §4. |
| Callback layer | Enforce token/rate/tool/result policies and compact old context. | Ordered callback semantics in §4. |
| Tool layer | Filesystem, code search/graph, memory, artifacts, OpenAPI, findings, HTTP, proxy, code execution, LikeC4. | Tool contracts in §6–§7. |
| Artifact service | Versionless logical key/value storage scoped by application and user. | Text/binary part contract and key taxonomy in §7. |
| Explorer | Local static SPA and JSON API over package metadata/eval results plus review comments. | HTTP API in §8. |
| Evaluation harness | Execute agent/task/pipeline scenarios repeatedly, score, and persist `eval/v1`. | Envelope and acceptance behavior in §10. |

Section symbols above refer to the numbered specification files, not headings
inside this file.

## 3. Canonical run sequence

1. Load environment-backed settings once.
2. Validate and canonicalize `project-path` and its project-relative folder
   scope.
3. Derive a per-project artifact-store directory:
   `<artifact-base>/<sanitized-project-basename>-<first-8-hex-of-SHA1(canonical-path)>`.
4. Construct an artifact service rooted at that directory and a virtual
   filesystem rooted at the canonical project path.
5. Optionally delete all artifacts for the selected application/user.
6. Resolve the public workflow key and construct an immutable workflow context.
7. Start the event handler and optional live renderer.
8. Enter an optional top-level observability span tagged with workflow, model,
   user, project, and folder metadata.
9. Run the workflow. The workflow may seed artifacts, execute task runners,
   execute direct agents, or delegate to nested workflows.
10. On workflow exit, run cleanup and emit the terminal workflow event.
11. Export every non-internal artifact to the output directory, preserving
    hierarchy and payload type.
12. Render a grouped artifact summary. Metrics remain in
    `<output>/metrics.jsonl`; a resume checkpoint, when enabled, remains at
    `<output>/checkpoint.json`.

## 4. Context and identity model

Every workflow receives this logical record:

| Field | Type | Meaning |
|---|---|---|
| `project_path` | canonical directory | Host source root, never exposed directly to agents. |
| `folder_name` | virtual absolute path | Scoped project folder, `/` for the root. |
| `model` | string | Alias resolved by the configured model gateway. |
| `timeout` | integer seconds | Per-model-request timeout. The compatibility CLI does not range-check it; a hardened boundary should require a positive value. |
| `app_name` | string | Artifact/session application namespace; CLI uses `contractor`. |
| `user_id` | string | Artifact/session user namespace. |
| `artifact_service` | interface | Load/save/delete/list logical artifact keys. |
| `fs` | virtual filesystem interface | Root-confined source view. |
| `artifact` | optional text | User-provided seed. |
| `prompt` | optional text | Free-form prompt for prompt-driven workflows. |
| `checkpoint_path` | optional path | Enables task-level restoration and completion recording. |

Identity exists at several nested levels and MUST remain distinguishable:

- workflow key/class;
- task runner name (checkpoint ownership);
- queued task `ref` (stable resume identity);
- task invocation UUID (fresh queue identity);
- numeric task index;
- attempt/iteration number;
- session UUID;
- model-runtime invocation ID;
- agent name;
- subtask numeric ID;
- tool call ID.

Metrics and persisted records carry enough of these identifiers to correlate a
tool result with its call without assuming tool arguments are unique.

## 5. Storage topology

### 5.1 Logical artifact store

The artifact service is scoped by `(app_name, user_id, filename)`; session ID is
unused for workflow artifacts. Important filename families are:

```text
<artifact-key>/{result,summary,records}
user:memory/<namespace>/...
user:inbox/<namespace>/...
user:vulnerability-reports/<source-namespace>
user:vulnerability-verifications/<source-namespace>
oas-openapi-building
user:oas-openapi-building
trace-openapi-{fs,diff}
http/<client-name>/session.json
http/<client-name>/responses/<zero-padded-request-id>.json
```

The exact taxonomy and payload schemas are normative in files 03, 05, and 07.

### 5.2 Export directory

The output directory is a user-facing materialization, not the primary store.
It contains:

- all non-memory artifacts, mapped one logical key to one relative file;
- `metrics.jsonl` appended during the run;
- `checkpoint.json` only when resume mode is enabled;
- optional evaluation/report files written by scripts.

Deleting the exported copy does not necessarily delete the per-project artifact
store. The explicit artifact-reset option operates on the store.

### 5.3 Source and overlay storage

The base filesystem maps virtual `/x` to `<project>/x` after canonicalization
and containment checks. Trace workflows wrap it with a copy-on-write memory
overlay. Overlay state is serialized as a patch plus a human-readable diff;
parallel trace variants fork overlays and merge them with conflict reporting.

## 6. Runtime dependency contracts

A reimplementation needs behavioral equivalents of these services:

| Capability | Required behavior |
|---|---|
| Agent runtime | Streaming model turns, structured input/output schemas, function tools, nested agent-as-tool calls, sessions/state, callback/plugin hooks. |
| Model gateway | OpenAI-style chat/tool calling through user-selected aliases; configurable timeout and sampling. |
| Artifact service | Async text/binary parts, key listing, load/save/delete, durable local implementation and in-memory test implementation. |
| Filesystem abstraction | Virtual paths, stat/list/walk/glob/open/move/copy/remove, local and remote implementations. |
| Source parser | Cross-language symbol/definition search and optional call-graph extraction. |
| YAML/JSON/Markdown/XML support | Safe parsing and deterministic formatting for prompts, tasks, tool results, and OpenAPI. |
| Optional vector store | Namespace/document chunks with embeddings and similarity retrieval. |
| Optional interception proxy | Request history, replay, automation, sitemap, and workflow GraphQL operations. |
| Optional container runtime | Ephemeral named containers, read-only source mount, writable scratch, command execution, teardown. |

No specific library is normative.

## 7. Concurrency model

- A normal `TaskRunner` executes its queue sequentially.
- A worker may make nested calls through tools; task tool execution is guarded
  so duplicate concurrent claims cannot execute one subtask twice.
- Path-parallel trace creates one overlay fork per configured route group and
  runs up to `max_concurrency` groups simultaneously.
- Vulnerability sweep runs independent class nomination tasks concurrently up
  to `sweep_concurrency`, sharing only artifact/checkpoint storage.
- Checkpoint saves serialize read-merge-replace in-process and use a per-file
  cross-process advisory lock.
- HTTP clients addressing the same persisted namespace share an event-loop
  lock, even when represented by different client objects.
- The explorer uses a threaded local HTTP server; its comment database opens a
  connection per request and uses write-ahead logging.

Cancellation is never swallowed by generic failure isolation. Completed
parallel work should be merged/persisted in a `finally`-equivalent boundary.

## 8. Failure domains

| Failure | Required outcome |
|---|---|
| Invalid CLI path/seed/flag combination | Clean usage error before workflow work. |
| Workflow constructor validation | Clean usage error. |
| Model/tool failure inside a task attempt | Attempt failure event; retry while budget remains. |
| Attempts exhausted | Task failure event and task-not-completed exception. |
| Skippable fan-out unit fails | Log and emit task-skipped; continue only when the caller selected isolation. |
| Verification artifact missing/stale | Fail workflow; never reinterpret as zero findings. |
| Event handler failure | Log and continue; cancellation still propagates. |
| Workflow cleanup failure | Log; preserve original outcome. |
| Observability failure | Log at warning/debug and behave as disabled. |
| Corrupt/unsupported checkpoint | Warn and treat as absent. |
| Checkpoint write failure | Keep caller snapshot dirty and retryable; do not adopt merged sibling state. |
| Sandbox cleanup failure | Log; do not replace run result; retain process-exit/TTL backstops. |
| Explorer malformed resource ID | 404; never resolve it as a filesystem path. |

## 9. Extensibility rules

- A new workflow gets one registry key, an assembler, a sibling configuration
  file, and an exported constructor.
- A new planner worker gets a versioned prompt manifest and a factory that binds
  a declared tool set; planner protocol instrumentation remains generic.
- A new task gets a versioned manifest/body and declares artifact/skill inputs.
- A new tool is exposed only through explicit agent factory composition and
  returns a standard success/error envelope.
- New durable artifact families must document ownership, key construction,
  payload schema, update semantics, and collision behavior.
- New telemetry events join the shared event taxonomy so the sink, UI, analyzer,
  and eval metrics can agree on names.
