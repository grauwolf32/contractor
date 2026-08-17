# CLI and explorer interfaces

## 1. Executables

Contractor exposes two independent local executables:

1. The **workflow CLI** validates a project-scoped request, constructs all run
   services, executes exactly one workflow, records events, and exports its
   artifacts.
2. The **explorer** serves a static single-page application and a read-mostly
   JSON API for inspecting the installed agents, tasks, skills, workflows, and
   evaluation results. Its only mutable data is reviewer comments.

Neither executable is a remotely managed multi-user service. Authentication,
TLS termination, and remote tenancy are outside their contract. The explorer
MUST bind to loopback by default.

## 2. Workflow CLI

### 2.1 Invocation contract

The application name and default artifact namespace are `contractor`. The CLI
accepts the following options:

| Option | Required/default | Contract |
|---|---|---|
| `--workflow <key>` | default `oas_build` | One exact public registry key from the workflow catalog. |
| `--project-path <dir>` | required | Existing directory; canonicalized before any identity or containment calculation. |
| `--folder-name <path>` | default `/` | Existing directory inside the project; `/` means the entire project. |
| `--artifact <file>` | absent | Existing regular file read as UTF-8 text and supplied as the workflow seed. |
| `--user-id <id>` | default `cli-user` | Artifact and session user namespace. |
| `--model <alias>` | configured default | Model-gateway alias. |
| `--timeout <seconds>` | configured default, initially `300` | Model request timeout passed through the workflow context. |
| `--prompt <text>` | absent | Free-form input for workflows that consume one. |
| `--rm` | false | Delete all stored artifacts for this application/user/project before running. |
| `--resume` | false | Enable restoration from `<output>/checkpoint.json`. |
| `-o`, `--output <dir>` | `<project>/.contractor` | Export, metrics, and optional checkpoint directory. |
| `--no-ui` | false | Disable the live terminal display and render events as ordinary output. |

`--rm` and `--resume` are mutually exclusive. Invalid option values, an
unknown workflow, an invalid seed file, and a workflow-constructor validation
failure MUST be presented as usage errors rather than stack traces.

At present, `router` is the only workflow for which a non-empty prompt is a CLI
precondition. If it is omitted in interactive mode, the CLI repeatedly prompts
until it receives non-whitespace text. In `--no-ui` mode, omission is a usage
error. Other workflows may consume a prompt when supplied but MUST NOT be
silently added to this required set without updating this contract.

### 2.2 Path validation

The CLI performs path checks before constructing a workflow:

1. Resolve `project-path` to a canonical absolute path and require a directory.
2. Normalize an empty folder or `/` to virtual `/`; otherwise strip leading and
   trailing separators and resolve it relative to the project.
3. Require the resolved folder to be an existing directory whose canonical
   location is inside the canonical project root.
4. Present the workflow with the virtual folder name, never a second host root.
5. Read a seed only if its path is an existing regular file. Decode strictly as
   UTF-8; reject binary or otherwise undecodable data.

The default output directory is part of the selected project, but output may be
placed elsewhere when the user explicitly supplies it. Output placement does
not expand the source filesystem visible to tools.

### 2.3 Project artifact-store identity

Persistent workflow artifacts do not live directly in the export directory.
The store root is derived as:

```text
base = configured artifacts directory, otherwise <repository>/artifacts
safe_name = regex-replace every run outside [A-Za-z0-9._-] in the canonical
            path basename with "-", trim boundary "-", or use "project"
suffix = first 8 lowercase hexadecimal characters of SHA-1(UTF-8 canonical path)
store = <base>/<safe_name>-<suffix>
```

Sanitization MUST produce a portable directory component. The digest is over
the complete canonical path so two projects with the same basename have
different stores. All artifact operations within that store remain scoped by
`(app_name="contractor", user_id)`.

### 2.4 Startup and execution sequence

The CLI MUST execute these steps in order:

1. Load environment settings, configure logging, and initialize optional
   observability.
2. Validate flags, seed, workflow key, project/folder paths, and prompt
   requirements.
3. Create the output directory and, only for `--resume`, select its checkpoint
   path.
4. Initialize source-language parsers, then create the project-specific
   artifact service and rooted read-only-by-default filesystem.
5. If `--rm` is present, list and delete every artifact for the application and
   user before workflow construction.
6. Construct the immutable workflow context defined in file 02.
7. Resolve and construct the workflow, converting constructor validation
   failures into usage errors.
8. Install the metrics sink and either the live terminal renderer or plain
   renderer.
9. Enter a top-level observability context named `workflow.<workflow-key>`, with
   session label `<workflow-key>:<project-basename>`, workflow/model tags, and
   project, folder, and model metadata; attach the user through the trace's
   dedicated user field.
10. Run the selected workflow to a terminal outcome.
11. Export artifacts and render a grouped artifact summary.

A workflow error results in a failed command. Event-renderer and telemetry
failures are best-effort and MUST NOT mask the workflow outcome. Cancellation
MUST propagate.

### 2.5 Artifact reset and export

Reset lists every filename visible to the selected application/user and
deletes it from the persistent store. It does not delete the source project or
perform a recursive deletion of the export directory.

After a successful workflow run, export operates as follows:

- List all logical artifact filenames for the selected application/user.
- Skip every key beginning with `user:memory/`.
- Load the current value of every remaining key.
- Map the logical hierarchy to the same relative hierarchy below the output
  directory, creating parent directories as needed.
- Normalize both separator forms and require the resolved export target to
  remain below the output root; reject an unsafe logical key rather than
  materializing it.
- Write inline/binary artifact data byte-for-byte.
- Write textual artifact data as UTF-8.
- Do not infer binary status from a filename extension.

The final summary groups exported files by their first relative path component
and presents local file links where the terminal supports them. Metrics and the
checkpoint are operational files, not synthesized workflow artifacts.

### 2.6 Event-to-metrics contract

Every event in the shared Agio event taxonomy is offered to the
metrics sink before it is rendered. The metrics file is
`<output>/metrics.jsonl`; records are appended and existing records are not
truncated.

Each line is one valid JSON object with this flattened envelope:

| Field | Value |
|---|---|
| `type` | Stable event type string. |
| `timestamp` | Metrics-persistence time as Unix epoch milliseconds. |
| `ts_iso` | The same instant in UTC ISO-8601 form. |
| `task_name` | Event task name when available, otherwise null/absent according to the event. |
| `task_id` | Event task identity when available. |
| remaining fields | Event payload fields copied to the top level without replacing envelope fields. |

Writes MUST be serialized within the process and blocking append I/O MUST NOT
block the asynchronous event loop. Serialization recursively supports JSON
primitives, mappings, sequences/sets, filesystem paths as strings, objects with
a standard model/dictionary conversion, and finally a printable representation.
The implementation MUST apply its configured persistence redaction policy at
this single sink boundary.

### 2.7 Terminal presentation state

The live renderer is a projection of events, not a source of workflow state. It
tracks:

- workflow status and elapsed time;
- current and historical tasks/subtasks;
- attempts, progress, and completion/failure/skip state;
- cumulative input/output token counts;
- tool calls paired with results by call identity;
- recent messages, with a bounded history of 200 entries;
- final exported artifact groups.

High-volume internal events (`agent_run_start`, `agent_run_end`, `adk_event`,
`adk_tool_call`, `adk_tool_result`, `adk_tool_error`, filesystem-coverage
updates, and intermediate run summaries) may update internal counters without
being appended to the visible history. The ordinary `tool_call`, `tool_result`,
and `tool_exception` events remain user-visible. Only `workflow_finished` is
terminal for the live display. A task, subtask, agent, tool, finding, or
verification completion MUST NOT stop it.

With `--no-ui`, events use a non-live textual renderer. Metrics behavior is
identical in both modes.

## 3. Local explorer server

### 3.1 Process contract

The explorer accepts:

| Option | Default | Contract |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `8765` | Requested TCP port. |
| `--no-browser` | false | Do not launch the system browser. |
| `-v`, `--verbose` | false | Enable verbose request/process logging. |

If the requested port is unavailable, the server MUST request an operating-
system-selected free port and report the actual URL. Unless disabled, it opens
that URL in the default browser after a short delay. The server handles requests
concurrently and performs a graceful shutdown on interrupt.

### 3.2 HTTP response rules

- API success responses are JSON with a correct JSON content type and
  `Cache-Control: no-store`.
- Unknown API resources return a JSON `404`.
- Validation failures use a client-error status and explanatory JSON.
- Unexpected failures return JSON `500` and are logged server-side.
- Static assets use a content type selected from HTML, CSS, JavaScript, SVG,
  icon, and JSON; unknown extensions use binary content type.
- Static responses are also non-cacheable during local development.
- A non-API route that does not name a real confined static file falls back to
  `index.html`, enabling client-side routing.

### 3.3 Resource identifier safety

Agent, task, skill, and related path identifiers MUST, after URL decoding,
match:

```regex
[A-Za-z0-9][A-Za-z0-9_.-]{0,127}
```

Encoded separators, traversal components, nested percent-encoding intended to
recover a separator, backslashes, and identifiers outside this grammar are
rejected. Every candidate metadata/static/reference path is then canonicalized
and checked to remain within its specific package root. A symlink escape is
treated as missing. Validation and containment are both required; neither is a
replacement for the other.

### 3.4 Read API

The following routes are normative. Path placeholders are safe identifiers as
defined above.

| Method and path | Result |
|---|---|
| `GET /api/overview` | Counts and summary lists for agents, tasks, skills, workflows, and evaluation runs. |
| `GET /api/crossrefs` | Reverse indexes connecting agents and skills to tasks/workflows and tasks to workflows. |
| `GET /api/agents` | All agent manifests with active version and summaries. |
| `GET /api/agents/{name}` | One agent manifest, all available versions, and statically introspected tool bindings. |
| `GET /api/agents/{name}/{version}` | One resolved prompt version and its source content. |
| `GET /api/tasks` | All task manifests with active version and summaries. |
| `GET /api/tasks/{name}` | One task manifest and version list. |
| `GET /api/tasks/{name}/{version}` | Parsed task body, recognized fields, extras, and raw source. |
| `GET /api/skills` | Skill index summaries. |
| `GET /api/skills/{name}` | Skill index/front matter and reference list. |
| `GET /api/skills/{name}/ref/{ref}` | One confined Markdown reference. |
| `GET /api/workflows` | Public workflow keys, implementation class names, and summaries. |
| `GET /api/workflows/{key}` | Workflow details, configuration, and approximate execution graph. |
| `GET /api/evals` | Discovered `eval/v1` run summaries. |
| `GET /api/evals/{run_id}` | One evaluation run with derived fixture, tool, skill, CWE, and verdict summaries. |
| `GET /api/comments?kind=&id=&version=` | Reviewer comments, optionally filtered by target. |

Metadata is read from the installed source tree on request, except where a
pure read cache is used. Safe YAML parsing is required. Malformed optional
metadata may yield an empty/default representation, but it MUST NOT cause
arbitrary object construction or code execution.

Version ordering sorts conventional numeric `vN` identifiers by descending
number, then other identifiers deterministically. A summary is the first
meaningful prose line when no explicit description exists. Task parsing exposes
these standard fields when present: `name`, `objective`, `instructions`,
`output_format`, `context`, `artifacts`, `skills`, `iterations`, `format`, and
`max_steps`; unknown top-level fields remain visible as extras.

### 3.5 Static workflow introspection

The explorer MUST NOT execute workflow constructors or import agent factories
merely to draw a graph. It statically inspects workflow source to identify:

- task additions;
- explicit artifact dependencies;
- subworkflow invocations;
- conditional branches that can be recognized safely.

Nodes are task or subworkflow calls. An edge connects a consuming task to the
first compatible producer inferred from artifact name/namespace. Because this
is an approximation, unrecognized dynamic branches are returned as warnings,
not fabricated edges. Workflow configuration contributes declared agents and
budgets to the displayed detail.

Agent tool extraction is likewise static and cached: it parses builder/factory
source and returns recognizable tool bindings without importing the runtime or
contacting a model provider.

### 3.6 Evaluation discovery

The explorer discovers only files matching `eval_runs/**/eval_results.json`
whose parsed `schema` is exactly `eval/v1`. A run ID is its directory relative
to `eval_runs`, with path separators represented by `~`; the root form is
`root`. Detail lookup MUST resolve only IDs present in the discovered index. It
MUST NOT turn an arbitrary user-provided ID into a filesystem path.

Details derive presentation aggregates from the stored envelope: fixture
summaries, up to fourteen most-used tools, skills observed, per-CWE detection,
and a verdict matrix. The canonical result schema remains the one in file 10.

## 4. Reviewer comment API

### 4.1 Routes

| Method and path | Effect |
|---|---|
| `POST /api/comments` | Validate and create a comment. |
| `PUT /api/comments/{integer-id}` | Validate and replace only the body of an existing comment. |
| `DELETE /api/comments/{integer-id}` | Delete one comment. |

Comment targets are not arbitrary paths. `kind` is one of `agent`, `task`, or
`skill`; `target_id` and `version` are required non-empty identifiers. The body
is non-whitespace text; persistence silently truncates it to 20,000 characters.
Source locations satisfy `1 <= line_start <= line_end`.

### 4.2 Persistence schema

Comments are stored in the local SQLite database `.contractor/explorer.db`.
The service opens a connection per request, enables write-ahead logging, and
creates the schema idempotently:

| Column | Contract |
|---|---|
| `id` | Integer primary key, automatically incremented. |
| `kind` | Valid target kind. |
| `target_id` | Agent/task/skill identifier. |
| `version` | Displayed source version. |
| `line_start`, `line_end` | Inclusive one-based source range. |
| `body` | Validated comment text. |
| `created_at`, `updated_at` | UTC timestamps with second precision. |

An index covers `(kind, target_id, version)`. Lists are ordered by
`line_start`, then `id`. Creation sets both timestamps; update preserves
`created_at` and replaces `updated_at`.

## 5. Explorer browser behavior

The browser application is a build-free static client with hash routes for
overview, agents, tasks, skills, evaluations, and workflows. It caches fetched
API data for navigation; an explicit refresh clears that cache.

The source viewers support line-range comments and version comparison. Version
comparison uses a line-oriented longest-common-subsequence diff. Workflow pages
render the approximate directed graph, and evaluation pages render charts and
matrices from the derived API data.

All dynamic plain strings MUST be inserted as text, not HTML. Markdown rendering
MUST escape source text before applying its supported transformations. Link
destinations are permitted only for `http`, `https`, and `mailto` schemes;
unsafe or unparseable destinations render without an active dangerous link.
The renderer need only support the subset used by project documentation:
headings, paragraphs, emphasis, fenced/inline code, links, lists, and tables.

## 6. Interface acceptance criteria

A replacement satisfies this interface specification when automated tests show
that:

1. every CLI option, default, mutual exclusion, and prompt rule above is
   preserved;
2. same-basename projects receive different persistent stores;
3. folder, seed, static-file, metadata, reference, and eval-run traversal probes
   cannot escape their roots;
4. text and binary artifacts round-trip through export without type loss;
5. the live UI stops only at `workflow_finished` and metrics are equivalent in
   UI and non-UI modes;
6. every listed API route returns its documented resource or a bounded JSON
   error;
7. comment validation, ordering, timestamps, update, and deletion survive a
   process restart; and
8. malicious Markdown, comment text, identifiers, and verdict strings cannot
   create executable browser markup.
