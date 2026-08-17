# Runtime Orchestration Specification

## 1. Purpose and reconstruction boundary

This document specifies the execution runtime that turns versioned task templates into agent runs, persists their outputs, emits telemetry, retries incomplete work, and resumes completed work from checkpoints. It is intentionally implementation-language neutral. A conforming rebuild may use any agent framework, storage implementation, or concurrency library, provided that the observable schemas, ordering, state transitions, and failure behavior below are preserved.

The runtime has two entry points:

- `TaskRunner`: a sequential queue of versioned, planner-mediated tasks with attempts, completion counting, artifact publication, checkpoint restore, and built-in tracing/metrics/cleanup plugins.
- `AgentRunner`: a thin single-agent runner with session state, final-text extraction, optional plugins, optional artifact publication, and no templates, queue, retries, or checkpoints.

The following are external ports that the runtime requires:

1. **Agent engine** — accepts an agent, app name, user ID, session ID, user message, session service, artifact service, and plugins; exposes an asynchronous event stream.
2. **Session service** — creates and retrieves sessions identified by `(app_name, user_id, session_id)`. A session contains a mutable string-keyed state map.
3. **Artifact service** — saves and loads content parts identified by `(app_name, user_id, optional_session_id, filename)`.
4. **Model descriptor** — an opaque model/client configuration that can be set per runner or per invocation.
5. **Event handler** — an asynchronous consumer of the event envelope defined below.
6. **Worker builder** — a callable that builds a domain worker from at least `namespace` and output `format`; workers that support source-coverage capture may additionally accept `capture_in_scope`.

The task/planner/worker protocol itself is specified in `04-agents-callbacks-skills.md`.

## 2. Canonical data contracts

### 2.1 Enumerations

`TaskStatus` has exactly these values:

| Value | Meaning |
|---|---|
| `running` | The overall queued task has not been finalized by the planner. |
| `done` | The planner called its finish operation with a successful completion. |

The task runner emits these core event types:

| Value |
|---|
| `run_started` |
| `run_finished` |
| `task_started` |
| `task_finished` |
| `task_failed` |
| `global_task_finished` |
| `iteration_started` |
| `iteration_finished` |
| `iteration_result` |
| `final_text` |

Plugins may emit additional string event types through the same envelope. Event consumers must therefore accept both the core values and arbitrary extension strings.

Artifact kind is one of `result`, `summary`, or `records`.

### 2.2 `TaskInvocation`

A queued invocation contains:

| Field | Type/default | Contract |
|---|---|---|
| `id` | opaque string | Random 32-hex invocation identity generated when queued. It is stable for the queued object, including checkpoint restores. |
| `ref` | string | Human/workflow identity. Must be unique within one runner queue. |
| `template_key` | string | Task template manifest key. |
| `template_version` | string | Concrete resolved version, never an unresolved `active` alias. |
| `worker_builder` | callable | Builds the domain worker. |
| `params` | map, `{}` | Invocation-level template values. |
| `artifacts` | list of filenames, `[]` | Input artifacts to load before rendering. |
| `skills` | list of skill names, `[]` | Skill directories to inject before the first attempt. |
| `artifact_key` | nullable string | Output namespace override. If null, use `template_key`. |
| `iterations` | integer, `1` | Number of successful `done` attempts required. This is not the total attempt count. |
| `max_attempts` | integer, `1` | Maximum number of attempts, successful or not. |
| `max_steps` | integer, `15` | Maximum total planner subtasks for each attempt. |
| `timeout_s` | nullable numeric duration | Wall-clock timeout applied independently to each entire attempt. Null means no runtime timeout. Queue construction does not require a positive value; zero or negative values flow to the timeout primitive and normally make the attempt time out immediately. |
| `namespace` | nullable string | Memory/tool namespace; null means runner name. |
| `model` | nullable model | Per-task model; null means runner default. |
| `observations` | nullable observation config | Per-task override; null means runner-level configuration. |

Derived values:

- `effective_artifact_key = artifact_key ?? template_key`.
- `effective_namespace = namespace ?? runner.name`.
- `effective_model = model ?? runner.default_model`.
- Template cache identity is the pair `(template_key, template_version)`, not the key alone.

### 2.3 `TaskTemplate` and `RenderedTask`

The immutable loaded template record is:

```text
TaskTemplate {
  key: string,
  version: string,
  title: string,
  objective: string,
  instructions: string,
  output_format: string,
  default_artifacts: string[] = [],
  default_skills: string[] = [],
  default_iterations: integer = 1,
  format: string = "json",
  default_params: map<string,string> = {}
}
```

Rendering produces a separate immutable record:

```text
RenderedTask {
  key: string,
  title: string,
  objective: rendered string,
  instructions: rendered string,
  output_format: rendered string,
  format: string,
  artifacts: map<artifact-filename,text> = {}
}
```

Title and format are copied rather than placeholder-rendered. The three task body strings are independently rendered from the same scope. The artifacts map preserves the caller's declared order.

### 2.4 `TaskResult`

Each normal attempt returns, and a restored task synthesizes, this record:

| Field | Contract |
|---|---|
| `invocation_id` | Queue-time `TaskInvocation.id`, not the agent engine's invocation ID. |
| `task_ref` | Invocation ref. |
| `task_key` | Rendered task key. |
| `task_title` | Rendered task title. |
| `template_key` | Template manifest key. |
| `task_id` | Zero-based queue index for the current run. |
| `session_id` | Per-attempt session ID; empty for checkpoint restore. |
| `final_response` | Last non-empty final-response text emitted by the engine; empty is valid. |
| `state` | Final session state snapshot. |
| `carry_state` | Deep copy of final state used to seed a later attempt after filtering. |
| `status` | Value of `task::{task_id}::status`, or null/missing. |
| `result` | Value of `task::{task_id}::result`, default empty string. |
| `summary` | Value of `task::{task_id}::summary`, default empty string. |
| `records` | Value of `task::{task_id}::pool`, default empty list. |
| `params` | Deep copy of invocation params. |
| `input_artifacts` | Deep copy of loaded artifact filename-to-text map. |
| `published_artifacts` | Deterministic kind-to-filename map for the effective output key, whether or not saving has yet succeeded. |
| `restored` | `true` only for validated checkpoint reuse; otherwise `false`. |

No schema coercion is performed on values read from final state beyond the defaults above. A conforming implementation should preserve unexpected values for diagnostic fidelity rather than silently normalize them.

### 2.5 Event envelope

Every runner event has:

```text
TaskRunnerEvent {
  type: core-enum-or-string,
  task_name: string,
  task_id: integer,
  payload: open map
}
```

The event object is immutable after construction. Payload values may contain rich in-process objects, including a full `TaskResult`; serialization is the event sink's responsibility.

### 2.6 Task-scoped state

For queue index `N`, fixed state keys are:

```text
_global_task_id              = N
task::N::objective           = rendered objective
task::N::status              = "running"
task::N::current             = null
task::N::result              = ""
task::N::summary             = ""
task::N::pool                = []
```

Planner-internal keys add at least one further segment:

```text
task::N::<agent-invocation-id>::<manager-name>::...
```

The fixed keys are the authoritative task outcome surface. Final model text by itself never marks a task complete.

## 3. Task templates and rendering

### 3.1 Filesystem layout and manifest schema

For template key `K`, load `tasks/K.yml`:

```yaml
active: v2
versions:
  v1:
    file: K/v1.yml
  v2:
    file: K/v2.yml
```

The body file must contain:

```yaml
task:
  name: optional display title
  objective: required string
  instructions: required string
  output_format: required string
  artifacts: optional list of artifact filenames
  skills: optional list of skill directory names
  iterations: optional integer, default 1
  format: optional string; supported values are json/markdown/yaml/xml, default json
  params: optional map of placeholder defaults
```

The concrete version is selected in this precedence order:

1. Explicit `add_task(version=...)` value.
2. Environment value `CONTRACTOR_TASK_VERSION_<UPPERCASE_TEMPLATE_KEY>`.
3. Manifest `active` value.

The selected version must exist in `versions`, its entry must be a map with a non-empty `file`, and the referenced body must exist. A missing manifest, malformed manifest, undeclared version, missing body, missing top-level `task`, or missing required body field is a queue-time error.

Body defaults are exact:

- title: `task.name`, falling back to template key when absent or empty;
- artifacts and skills: empty lists;
- iterations: `int(value or 1)`;
- format: `value or "json"`; the template loader does not validate the supported-value set, so an unknown value reaches downstream formatters/builders rather than failing here;
- params: keys converted to strings; null values converted to empty strings; all other values converted to strings.

The reconstruction's current manifest data selects these active versions:

| Task key | Active |
|---|---|
| `dependency_information` | `v1` |
| `exploitability_assessment` | `v4` |
| `knowledge_consolidation` | `v1` |
| `knowledge_discovery` | `v1` |
| `likec4_build` | `v1` |
| `likec4_validate` | `v2` |
| `oas_enrich` | `v2` |
| `oas_update` | `v2` |
| `oas_validate` | `v1` |
| `project_information` | `v1` |
| `project_information_short` | `v1` |
| `sink_nomination` | `v1` |
| `threat_analysis` | `v1` |
| `trace_annotation` | `v3` |
| `trace_verify` | `v1` |
| `vuln_analytics` | `v1` |
| `vuln_scan` | `v3` |
| `vuln_scan_fast` | `v1` |

All declared historical body files are runtime data, not dead migrations: explicit version arguments and the environment override can select them. A reconstruction must retain each manifest's complete declared version map and body text, not only the active body.

### 3.2 Rendering scope

Render the three fields `objective`, `instructions`, and `output_format` using ordinary named-placeholder substitution. Build the scope from low to high precedence:

1. Template default params.
2. Runner variables set through `add_variable`.
3. Invocation params.
4. Runtime-generated artifact variables described below.

Missing placeholders are errors. Extra scope values are ignored.

The runtime adds `artifacts`, whose value is a Unicode-preserving YAML serialization of the complete filename-to-text input map, retaining insertion order.

For every artifact ref, it also creates a variable:

```text
artifact__<normalized-segment>__<normalized-segment>...
```

Normalize each non-empty slash-separated segment by replacing each run of non-ASCII-alphanumeric/non-underscore characters with `_`, trimming leading/trailing `_`, lowercasing, and substituting `task` if the result is empty. Distinct refs that normalize to the same variable are a hard render error naming both refs; the later artifact must not silently overwrite the earlier one.

### 3.3 User message

The rendered task is sent to the planner as one user message:

```text
TASK:
<title>

OBJECTIVE:
<objective>

INSTRUCTIONS:
<instructions>

OUTPUT FORMAT:
<output_format>

INBOX:
artifacts from previous tasks, stored as memories:
* <artifact ref>
...
```

Omit the entire `INBOX` section when there are no declared artifacts.

## 4. Queue construction

`TaskRunner` configuration consists of runner `name`, artifact service, optional checkpoint path, template cache, queue, variables, default model, session service, and runner-level observation configuration.

`add_task` performs all deterministic validation before appending:

1. Load and cache the concrete template by `(key, version)`.
2. If an output artifact key is supplied, validate it as specified in section 8.
3. Choose `ref = supplied_ref ?? "<name>:<current_queue_length>"` and reject a duplicate ref.
4. Select invocation skills. A supplied empty list overrides template defaults; null uses defaults. Verify every named skill directory exists.
5. Select artifacts with the same empty-list-versus-null semantics.
6. Resolve retries:
   - `iterations = explicit ?? template.default_iterations`;
   - `max_attempts = explicit ?? max(1, iterations)`;
   - require `iterations >= 1`;
   - require `max_attempts >= iterations`.
7. Create a random invocation ID, append the invocation, and return its ID.

There is no queue-time range validation for `max_steps` or `timeout_s`; preserving downstream behavior for zero/negative values is part of compatibility rather than treating them as an early validation error.

The queue is not cleared after `run`. The runner is deliberately non-reentrant: simultaneous `run` calls on one instance are unsupported because they share the queue, session service, and a mutable current event-handler field.

## 5. `TaskRunner.run` lifecycle

### 5.1 Top-level algorithm

Tasks execute strictly in queue order; TaskRunner itself does not fan them out.

```text
set current event handler
results = []
load-or-create checkpoint
emit run_started

for each invocation at zero-based task_id:
    if checkpoint restore validates:
        synthesize restored result
        append it
        continue

    result = run_task_with_retries(...)
    append result
    checkpoint the completed task
    emit global_task_finished

emit run_finished(ok=true)
return results
```

On any thrown value, including cancellation:

1. Emit `run_finished(ok=false)` with `completed_tasks = len(results)`.
2. Re-throw the original failure unless failure-event delivery itself raises cancellation.

In a final cleanup block, clear the current event handler and sweep all code-execution sandboxes. Sandbox cleanup errors are logged and suppressed.

A failed task stops the run; later queued tasks do not execute.

### 5.2 `run_started` payload

Use `task_name="__runner__"`, `task_id=-1`, and include:

- `total_tasks`;
- `completed_tasks=0`;
- `user_id`;
- snapshot of every valid agent prompt's active version;
- runner observation configuration as a JSON-friendly tag;
- `task_invocations`, each containing `ref`, `template_key`, `template_version`, and the effective observation tag.

### 5.3 Per-task setup

For a non-restored invocation:

1. Load every declared input artifact before rendering.
2. A missing artifact becomes empty text and produces a warning; it does not fail the task.
3. Render the task.
4. Emit `task_started` with template identity, task title, retry counts, params, declared input refs, deterministic output filenames, total/completed counts, and effective observations.
5. Inject skills once into the effective namespace.
6. Inject all input artifacts once into that namespace as inbox memories.
7. Enter the attempt loop.

Load, render, event, or injection failures before the attempt loop propagate directly. They produce top-level `run_finished(ok=false)` but not `task_failed`, because no attempt loop was entered.

## 6. Attempts, sessions, completion counting, and timeouts

### 6.1 Attempt initialization

Each attempt receives a one-based `iteration` number and a fresh random session ID. It also gets a newly constructed planning agent and fresh plugin instances.

Build initial state by:

1. Starting from a deep copy of the previous normal attempt's final state.
2. Removing only keys that begin `task::<current-task-id>::` and have another `::` after that prefix. These are stale planner-invocation internals.
3. Preserving fixed task keys, keys for other task IDs, and unrelated namespaces.
4. Overwriting the fixed current-task keys with the fresh active-state values from section 2.6. This resets result, summary, and records for every attempt.

Pre-create the session with this state before running the agent. This is required even if an engine offers an initial-state argument on its run method; the compatible behavior is that session state already exists before event consumption.

Emit `iteration_started` with iteration, session ID, objective, complete initial state, and template identity.

### 6.2 Event consumption and final text

Consume the engine event stream to exhaustion. For each event:

1. Ignore it unless the engine marks it as a final response.
2. From a final response, concatenate all non-empty text parts with newline separators and trim outer whitespace.
3. Ignore a final response whose extracted text is empty.
4. Replace the remembered final text with this text; the last qualifying final response wins.
5. Read current session state and emit `final_text` with iteration, session ID, text, and the state snapshot.

After the stream ends, retrieve final session state. If the session is missing, use `{}`. Build `TaskResult`, emit `iteration_finished` carrying the full result, and return it.

Artifact text decoding accepts a direct text field first. Otherwise, inline string data is used as-is; inline bytes are UTF-8 decoded. Invalid UTF-8 is decoded with replacement characters and a warning. Unsupported or absent content becomes empty text.

### 6.3 Attempt loop

`iterations` means **required cumulative successful runs**, not retries and not necessarily consecutive successes.

```text
carry_state = {}
last_result = null
last_exception = null
successful_runs = 0

for attempt in 1..max_attempts:
    run one iteration, optionally under timeout_s
    if cancellation:
        rethrow immediately
    if any other exception, including timeout:
        remember exception
        emit iteration_result(completed=false, error details)
        continue

    remember result
    completed = (result.state[task::<id>::status] == "done")
    emit iteration_result with projected successful count

    if completed:
        successful_runs += 1
        publish result/summary/records immediately
        if successful_runs >= iterations:
            emit task_finished
            return this result

    carry_state = result.carry_state

emit task_failed
throw TaskNotCompletedError, chained from last exception if any
```

Compatibility implications:

- A sequence `done, running, done` satisfies `iterations=2`; the intervening non-completion does not reset the first success.
- Exceptions and timeouts consume attempts but do not alter carry state. A normal incomplete result does update carry state.
- Fixed result/summary/record keys are reset at the start of every attempt, so records from multiple successful attempts are not accumulated in the final TaskResult. Cross-attempt durable knowledge must live in preserved unrelated state or namespace artifacts/memories.
- Publication happens on every `done` attempt, including a `done` attempt before the required success count is reached. Later successful attempts overwrite the same filenames.
- Publication occurs after the guarded agent-attempt call. A publication failure therefore propagates immediately, does not consume-and-retry another attempt, does not emit `task_failed`, and does not produce a checkpoint. Earlier files from the three-write sequence may already exist.
- The per-attempt timeout covers the entire single-iteration operation. Cancellation caused by timeout is converted by the timeout mechanism into a timeout exception and consumes the attempt. External cancellation must propagate immediately, emit neither `iteration_result` nor `task_failed`, and trigger top-level cleanup.
- A normal final response is irrelevant to completion unless fixed state status equals `done`.

`iteration_result` for an exception contains `session_id=null`, `status/result/summary=null`, `completed=false`, required counts, current successful count, exception type, and message. A normal result includes its session/status/result/summary and the successful count that would apply if completed.

On exhaustion, `task_failed` carries `max_attempts`, last normal result, and last exception message. `TaskNotCompletedError` exposes `ref`, `iterations`, `max_attempts`, and nullable `last_error`; its message includes all counts and last error when present.

### 6.4 Successful task completion

`task_finished` is emitted inside the retry operation with final session ID, status, result, summary, records, published filenames, template identity, and progress fields. The outer run then:

1. Appends the result.
2. Saves its checkpoint entry.
3. Emits `global_task_finished` with status/result/summary and `completed_tasks=task_id+1`.

Checkpoint save failure occurs after artifacts were published and the result appended locally, but before `global_task_finished`; it fails the run.

## 7. Worker construction hook

For each attempt, call the invocation's worker builder as:

```text
worker_builder(
  namespace = effective_namespace,
  _format = rendered_task.format,
  capture_in_scope = true  // only under the condition below
)
```

The leading underscore in the `_format` keyword is part of the current
builder-call compatibility contract; a builder exposed only as `format` will
not receive this call successfully.

Pass `capture_in_scope=true` only when:

- the effective observation config enables coverage-gap tracking; and
- signature introspection succeeds; and
- the worker builder explicitly declares a parameter named `capture_in_scope`.

Do not pass the argument otherwise. Builders that cannot be introspected are treated as not supporting it.

Wrap the worker in the planning agent using XML for planner-facing memory/task-tool formatting, the effective model, `max_steps`, effective namespace, and effective observations. The worker's domain tools were already built using the rendered task format, but planner instrumentation uses XML for the Subtask request/result handshake. The result parser is deliberately cross-format, so a context-limit callback built with the domain format (normally JSON) can still return a parseable result.

## 8. Artifact contract

### 8.1 Artifact keys and filenames

Validate an artifact key by trimming surrounding whitespace, then surrounding `/` characters. Reject it if empty or if any slash-delimited segment is exactly `..`. Do not otherwise canonicalize internal characters or collapse slashes.

For validated key `K`, filenames are exactly:

```text
K/result
K/summary
K/records
```

All task result artifacts and input artifacts use `session_id=null`, making them user/application scoped rather than attempt-session scoped.

### 8.2 Publication encoding

Save all three artifacts in the order `result`, `summary`, `records`:

- `result`: supplied string or empty string;
- `summary`: supplied string or empty string;
- `records`: if already a string, preserve it exactly; otherwise serialize `records || []` as compact JSON with Unicode preserved.

The three writes are not transactional. If a later save fails, earlier artifacts may remain. Return the kind-to-filename map only after all writes succeed.

### 8.3 Stable single-segment slugging

Fan-out workflows may need to convert arbitrary identities into one portable artifact-key segment. The canonical slug algorithm is:

1. Reserve direct output for a non-empty raw value matching `[a-z0-9_-]+`, length at most 160, not beginning `h_`, and not equal to a lowercase Windows device name (`aux`, `con`, `nul`, `prn`, `com1`-`com9`, `lpt1`-`lpt9`). Return such a value unchanged.
2. Otherwise trim the raw value for readability, replace each run outside `[A-Za-z0-9_-]` with `_`, trim `_`, and use `item` if empty.
3. Compute the full lowercase hexadecimal SHA-256 of the original untrimmed raw value.
4. Return `h_<readable-prefix>_h<digest>`, truncating and right-trimming the readable prefix so total length is at most 160. Use `item` if no readable prefix remains.

The raw and encoded domains are disjoint because raw identifiers beginning `h_` are always encoded. Case, surrounding whitespace, and degenerate non-empty values remain identity-significant through the digest.

### 8.4 Inbox memory injection

For each loaded input artifact, upsert a memory note into the effective namespace:

- name: exact artifact filename;
- body: decoded artifact text;
- description: cached template title when the first filename segment matches a loaded template key, otherwise `result from previous task <name>`;
- tags: exact list `[name, "inbox", "previous-task-result"]`.

Injection is performed once per queued task, not once per attempt.

## 9. Checkpoints and concurrent writers

### 9.1 File schema

Checkpoint JSON is:

```json
{
  "version": 1,
  "workflow": "runner-name",
  "updated_at": "UTC ISO-8601 timestamp",
  "tasks": [
    {
      "task_id": 0,
      "ref": "stable-ref",
      "template_key": "template",
      "template_version": "v2",
      "published_artifacts": {
        "result": "key/result",
        "summary": "key/summary",
        "records": "key/records"
      }
    }
  ]
}
```

Entry identity is `ref`. `mark_done` removes any previous same-ref entry and appends the replacement, so updated refs move to the end.

Loading returns no checkpoint, with a warning where applicable, for:

- nonexistent file;
- I/O or JSON parse failure;
- unsupported version;
- a top-level value that cannot supply map-style fields;
- a `tasks` value that cannot be iterated as task maps;
- a task missing `task_id`, `ref`, `template_key`, or `template_version`;
- duplicate refs.

Compatibility parsing is intentionally permissive rather than full schema
validation. A missing `workflow` is accepted as the empty string, missing
`tasks` as an empty list, and omitted `published_artifacts` as `{}`. Field
types and the shape of `published_artifacts` are not independently validated
when construction accepts them. Unknown fields are ignored. Duplicate refs are
detected when the baseline snapshot is captured and make the whole checkpoint
invalid. The checkpoint parent directory must already exist before saving.

### 9.2 Runner ownership

If no checkpoint file exists or it is invalid, create an empty in-memory checkpoint owned by `runner.name`. If a valid checkpoint's `workflow` differs from `runner.name`, warn and start with a fresh checkpoint; never restore or merge the other workflow's entries.

### 9.3 Restore validation

For the current invocation ref, restore only if:

1. An entry exists.
2. Entry template key and concrete version exactly match the invocation.
3. Every deterministic current filename for `effective_artifact_key` exists in the artifact store.

Do not trust or use the entry's recorded artifact map to choose filenames. This prevents a sibling invocation with the same template from validating the wrong output namespace. Artifact contents are not parsed or hashed; existence is sufficient. Entry task ID is not required to match the current queue index.

On restore, emit `task_started`, then synthesize:

```text
session_id = ""
final_response = ""
state = {}
carry_state = {}
status = "done"
result = "(restored from checkpoint)"
summary = ""
records = []
input_artifacts = {}
restored = true
```

Emit `task_finished` and `global_task_finished`, both with `restored=true`; append the result; do not rewrite the checkpoint. Restored events use current queue task ID and current invocation metadata.

### 9.4 Atomic, merge-preserving save

Checkpoint save must tolerate multiple processes holding stale snapshots of the same workflow.

Maintain per in-memory checkpoint:

- a set of refs explicitly changed through `mark_done`;
- a baseline snapshot of each loaded ref: `(task_id, template_key, template_version, canonical-JSON(published_artifacts))`.

At save time:

1. Reject duplicate refs in the current in-memory list.
2. Compute dirty refs as explicit dirty refs union refs whose current snapshot differs from baseline.
3. Compute deleted refs as baseline refs absent from the current list. This supports direct public-list deletion.
4. Acquire a process-local mutex and a cross-process exclusive advisory lock on sibling file `.<checkpoint-name>.lock`.
5. Reload the latest checkpoint while holding both locks.
6. If latest exists and has the same workflow:
   - start from latest entries excluding locally deleted refs;
   - apply only locally dirty entries with replace-and-append semantics;
   - retain untouched sibling additions and updates.
7. Serialize version, workflow, current UTC timestamp, and merged tasks to a uniquely named temporary file in the destination directory.
8. Atomically replace the destination with the temporary file.
9. Delete any remaining temporary file in cleanup.
10. Only after successful replacement, adopt merged entries into memory, clear dirty refs, and capture the new baseline.

If writing or replacement fails, do not adopt merged sibling data and do not clear dirty/baseline state; a retry must still distinguish local changes from imported siblings. The advisory lock file may remain as a harmless coordination file.

## 10. Events and plugins

### 10.1 Event delivery semantics

If no handler is installed, emission is a no-op. Otherwise await the handler synchronously before proceeding.

- Handler cancellation is never swallowed; it cancels the run.
- Every other handler exception is logged and suppressed. Events are best-effort telemetry and cannot fail task execution.
- There is no event queue, retry, or delivery guarantee.

The canonical telemetry taxonomy extends core runner events with agent lifecycle, tool calls/results/errors, model usage, filesystem coverage, run summaries, callback summaries, and full engine events. A persisted telemetry record should add millisecond epoch time, UTC ISO time, and nullable session/invocation/run/task/iteration/agent identifiers, while allowing arbitrary event-specific fields.

### 10.2 Per-attempt plugin set

Each TaskRunner attempt installs, in order:

1. Full trace plugin.
2. Metrics plugin.
3. Sandbox cleanup plugin.

Each trace/metrics plugin is initialized with task ref, queue task ID, one-based iteration, session ID, and the runner emitter. Plugin names include this context so instances are unique.

### 10.3 Trace plugin

The trace plugin emits:

| Hook | Event | Required payload in addition to common context |
|---|---|---|
| before outer/inner run | `agent_run_start` | engine invocation ID |
| after outer/inner run | `agent_run_end` | engine invocation ID |
| before tool | `adk_tool_call` | tool name, normalized arguments, agent/invocation identity, state snapshot |
| after tool | `adk_tool_result` | same plus normalized result and post-call state |
| tool error | `adk_tool_error` | same plus represented error |
| every engine event | `adk_event` | author, full event object, invocation ID |

Tool engines may supply arguments as either `tool_args` or `args`; prefer `tool_args` when present. They may supply response as either `tool_response` or `result`; prefer `tool_response` when present. State snapshot conversion tries common map-export methods, then direct map copying, and otherwise returns `{}`; snapshot conversion failures are suppressed.

### 10.4 Metrics plugin

Metrics are bucketed by engine invocation ID and agent name, substituting `unknown_invocation` / `unknown_agent` when absent.

Before each tool call:

- increment `calls_total`;
- register a monotonically numbered call;
- fingerprint `(invocation, agent, tool, canonical arguments)`;
- compute first 16 hex characters of SHA-256 over canonical JSON arguments;
- capture UTC start time and monotonic start time;
- emit `tool_call` with arguments, approximate JSON byte size, hash, and `call_<number>` identity.

On error, count an exception once per logical call and emit `tool_exception`. On normal result, classify as result-error when response is a map whose lowercase `status` is `error`, `failed`, or `failure`, or whose `error`, `error_message`, or `errors` value is non-empty. Otherwise count success. If an error callback was already seen for the same call, the later after-tool callback must not double count it. Emit elapsed milliseconds when a matching before-call exists.

Identical in-flight calls are correlated FIFO, preferring a non-errored pending call over an errored fallback so a retry is not paired with the previous error. Starting another identical call closes older errored calls' optional after-tool window. Invocation cleanup removes call-tracking state.

After every model response with usage metadata, accumulate per-agent call and token counters (`input`, `output`, `total`, `thoughts`, `cached`) and emit `llm_usage`. Missing or malformed counts become zero.

After tools, detect filesystem-coverage state changes and emit `fs_coverage`. Also write a compact current-invocation worker usage snapshot into session state for planner observations:

```json
{
  "tools": {"tool_name": {"calls": 2, "errors": 1}},
  "fs_coverage": {"files": 10, "read": 4}
}
```

At invocation end, emit `run_summary` containing every agent's metrics and a list of accounting imbalances where calls do not equal success plus exception plus result-error. If session state contains a non-empty `callbacks` map, emit `callback_summary`. Then delete that invocation's metric and call-tracker state.

Before trace/metric payloads are forwarded, recursively redact values under credential-bearing keys in `arguments`, `result`, and `tool_response`. Matching is case-insensitive for explicit names such as authorization, cookies, auth, passwords, secrets, credentials, private keys, API keys, bearer values, and common access/refresh/session token forms; unambiguous password/secret/API-key substrings and singular `_token`/`-token` suffixes also match. Replace values with `***REDACTED***`, recurse through maps/lists to depth 12, and never mutate the original payload. Token-count fields such as `max_tokens` and `prompt_tokens` are not secret matches.

### 10.5 Sandbox cleanup

The cleanup plugin records the first run invocation ID it sees as the outer/root invocation. It ignores completion of nested worker runs and calls global sandbox teardown only when that root ends. Cleanup failures are logged and suppressed.

An engine event generator interrupted by failure or cancellation may never invoke its after-run hook. Therefore TaskRunner's unconditional final sweep is required as a backstop. The runtime assumes code-execution sandboxes are used sequentially when sweeping all containers.

## 11. `AgentRunner`

### 11.1 Contract

AgentRunner has only `name` (used as app name), artifact service, and session service. Its `run` accepts an agent, string or prebuilt content message, user ID, optional session ID, optional initial state, optional plugins, optional event handler, and optional event-name override.

Algorithm:

1. `emit_name = event_name ?? agent.name`.
2. Generate a random session ID if omitted.
3. Wrap a string as one user-role text part; pass prebuilt content unchanged.
4. Emit `agent_run_started` with logical task name, actual agent name, and session ID.
5. Treat initial state as present only when it is truthy. If truthy, explicitly create the session and disable engine auto-creation. Empty `{}` is treated as absent and auto-creation remains enabled.
6. Run the agent with supplied plugins (default empty list).
7. Apply the same final-text extraction rule as TaskRunner. Every non-empty final text replaces the previous one and emits `final_text`; last wins.
8. Read final session state, defaulting to `{}` if missing.
9. Emit `agent_run_finished`.
10. Return `{final_text, session_id, final_state}`.

Unlike TaskRunner, AgentRunner has no error/finally lifecycle event: engine errors propagate and `agent_run_finished` is not emitted. Event-handler errors are still best effort except cancellation. The handler is call-local, so concurrent runs on one AgentRunner do not overwrite each other's handlers.

### 11.2 Optional artifact publishing

Given an AgentRunner result, publication reads the fixed task-scoped keys for a caller-supplied task ID (default 0), applies empty defaults, and uses the same three-artifact contract as TaskRunner. It returns the kind-to-filename map. It does not infer completion or write a checkpoint.

## 12. Failure and cancellation matrix

| Condition | Attempt consumed? | Retry? | Task failure event? | Run continues? |
|---|---:|---:|---:|---:|
| Normal result with status not `done` | Yes | Until budget | Only on exhaustion | Yes, within task |
| Model/tool/network exception inside attempt | Yes | Until budget | Only on exhaustion | Yes, within task |
| Per-attempt timeout | Yes | Until budget | Only on exhaustion | Yes, within task |
| External cancellation | No further attempts | No | No | No; rethrow |
| Event handler ordinary exception | N/A | N/A | No | Yes |
| Event handler cancellation | No | No | No | No; rethrow |
| Missing input artifact | No failure | N/A | No | Yes, empty substitution |
| Render/skill/artifact-injection error before loop | No | No | No | No |
| Result artifact save error | Normal attempt already returned | No | No | No; propagate |
| Checkpoint save error | Task already succeeded | No | No | No |
| Sandbox cleanup error | N/A | No | No | Yes/suppressed |

## 13. Minimum conformance scenarios

A reconstruction should include tests proving at least:

1. Two required successes can be separated by an incomplete attempt and still complete.
2. Exceptions and timeouts consume attempts; external cancellation does not.
3. Each normal attempt gets a new session and resets fixed current-task state while filtering stale invocation-scoped keys.
4. A missing artifact warns and renders as empty text.
5. Output artifact keys isolate sibling fan-out invocations using one template.
6. Checkpoint restore requires current deterministic artifact names and exact template version.
7. Stale concurrent checkpoint snapshots merge additions, updates, direct mutation, and deletion without losing sibling changes.
8. Failed atomic checkpoint replacement leaves dirty/baseline state retryable.
9. Handler exceptions do not abort work, but handler cancellation does.
10. TaskRunner sweeps sandboxes after success, failure, and cancellation.
11. AgentRunner supports concurrent calls with independent event handlers.
12. Final text is extracted only from final responses and the last qualifying response wins.
