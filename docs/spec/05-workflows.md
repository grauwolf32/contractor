# Workflow catalog and algorithms

## 1. Scope and terminology

This chapter is the normative reconstruction specification for the workflow
layer. It defines the public workflow registry, stage graphs, ordering,
fan-out, namespaces, artifact contracts, configuration, resume behavior, and
failure boundaries. Agent reasoning prompts and individual tool internals are
specified in the adjacent chapters; this chapter specifies how those pieces
are assembled.

The following stable terms are used throughout:

- **workflow key** is the value accepted by the public `--workflow` selector;
- **task template** is a versioned declarative objective such as
  `oas_update` or `trace_annotation`;
- **task runner** executes a queue sequentially through a planner and worker;
- **direct runner** invokes one already-built agent without the planner;
- **namespace** selects the durable memory and security-record stores visible
  to an agent;
- **artifact key** is a logical filename within one application/user artifact
  scope;
- **path key** or **group key** is the versioned encoding defined in §5;
- **finding store** is a YAML mapping at
  `user:vulnerability-reports/<namespace>`;
- **verification store** is a YAML mapping at
  `user:vulnerability-verifications/<namespace>`.

Names beginning with `user:` explicitly request user-scoped persistence. A
storage adapter may also treat a session-less save as user scoped, but a
conforming workflow MUST retain the logical filenames below because workflows
use both prefixed and unprefixed forms intentionally.

## 2. Public registry

The registry MUST expose exactly these 16 keys. Lookup returns a workflow
constructor; all constructors implement the lifecycle contract in §3.

| Public key | Workflow type | Execution style | Primary purpose |
|---|---|---|---|
| `oas_build` | `OasBuildingWorkflow` | one sequential task runner | Discover a project, build an OpenAPI document, and lint/repair it. |
| `oas_update` | `OasEnrichmentWorkflow` | one sequential task runner | Enrich and revalidate a pre-existing OpenAPI document. |
| `exploit` | `ExploitabilityWorkflow` | one task runner per finding | Probe a live target and persist one verdict per seed finding. |
| `likec4` | `LikeC4BuildingWorkflow` | one sequential task runner plus overlay cleanup | Build and validate a security-oriented architecture model. |
| `trace` | `TraceAnnotationWorkflow` | one task runner per OpenAPI path | Planner-driven source tracing and vulnerability reporting. |
| `trace-direct` | `TraceAnnotationDirectWorkflow` | one direct run per operation | Prompt-only trace baseline without the planning layer. |
| `trace-graph` | `TraceGraphWorkflow` | one direct run per operation | `trace-direct` with call-graph tools. |
| `trace-graph-pathpar` | `TraceGraphPathParWorkflow` | bounded concurrent route groups | Graph trace over isolated overlay forks, merged at the end. |
| `trace-postdiff` | `TracePostDiffWorkflow` | direct trace then direct analytics | Annotate first; judge only the resulting annotation diff. |
| `trace-verify` | `TraceVerifyWorkflow` | one task per discovered finding | Static, code-only verification of trace-produced findings. |
| `vuln-assess` | `VulnAssessWorkflow` | composed task/direct workflows | OpenAPI build, parallel trace, then optional live exploitability. |
| `vuln-scan` | `VulnScanWorkflow` | one sequential task runner | Single breadth-first source vulnerability scan. |
| `vuln-scan-fast` | `VulnScanFastWorkflow` | task runners plus direct per-finding trace | High-recall scan, deterministic deduplication, trace, optional exploit. |
| `vuln-scan-trace` | `VulnScanTraceWorkflow` | scan task then one trace task per finding | Breadth-first nomination followed by depth-first code tracing. |
| `vuln-sweep` | `VulnSweepWorkflow` | concurrent class sweeps then sequential traces | Recall-oriented class-specific nominations followed by DFS tracing. |
| `router` | `RouterWorkflow` | one direct planner/router run | Dispatch a free-form prompt among specialist agents. |

`build` and `enrich` appear in historical prose but are **not** registry keys.
The canonical keys are `oas_build` and `oas_update`. Similarly,
`trace-postdiff` and `vuln-sweep` are registered even though some older
workflow summaries omit them.

## 3. Shared workflow contract

### 3.1 Context

Every workflow is constructed with one immutable logical context containing:

| Field | Required behavior |
|---|---|
| `project_path` | Host project root, used for orchestration and nested contexts. |
| `folder_name` | Virtual project-relative folder inserted as `project_path` in task templates. |
| `model` | Model alias used to build every workflow agent. |
| `timeout` | Per-model-request timeout forwarded whenever the model client is built. |
| `app_name` | Application scope for direct runners and direct artifact operations. CLI value is `contractor`. |
| `user_id` | Default artifact/session user scope and seed-write identity. |
| `artifact_service` | Persistent text/binary artifact interface. |
| `fs` | Root-confined project filesystem. |
| `artifact` | Optional UTF-8 seed text. |
| `prompt` | Optional free-form router input. |
| `checkpoint_path` | Optional shared task checkpoint. |

The `user_id` passed to `run` is used by runners and workflow reads. Seed
persistence uses the identity stored in the context. Normal CLI construction
sets them equal; a non-CLI caller SHOULD do the same.

### 3.2 Lifecycle and events

The public workflow operation MUST behave as follows:

```text
emit workflow_started(workflow=<type-name>, phase="initializing")
ok = false
try:
    value = implementation(user_id, event_handler)
    ok = true
    return value
finally:
    try cleanup(user_id); log and suppress ordinary cleanup errors
    emit workflow_finished(workflow=<type-name>, ok=ok)
```

Both lifecycle events use task id `-1` and the workflow type name as the event
task name. An ordinary event-consumer error is logged and ignored. Cancellation
from the consumer propagates. In particular, cancellation while emitting the
start event occurs before the implementation/cleanup guard and aborts without
running either.

Nested composites in this chapter deliberately call child
**implementations**, not child public lifecycle wrappers. They therefore do
not emit nested `workflow_started`/`workflow_finished` events and do not run a
child cleanup hook unless the child implementation persists inline. This is
material for `vuln-assess` and `vuln-scan-fast`.

### 3.3 Seed, existence, and finding loaders

`persist_seed(filename)` MUST save `context.artifact` as text at `filename`
only when the seed is truthy. It MUST NOT delete or replace an existing
artifact when the supplied seed is absent or empty.

`artifact_exists(filename)` returns true when a loaded part has binary inline
data or non-empty text. A missing part or an empty text part returns false.
This predicate drives discovery-stage skips; it is stricter than checkpoint
restore, which validates artifact presence.

Finding consumers MUST load YAML safely. Missing, empty, invalid, or
non-mapping YAML yields no records. For each top-level `key: mapping` row, copy
the mapping and set `name=key` only if the row does not already declare a
name. Non-mapping rows are ignored. The more general YAML mapping loader keeps
all mapping keys/values without applying the finding reshape.

### 3.4 Planner-task execution and publication

Unless a workflow section says otherwise, a task runner:

1. Executes queued tasks in insertion order.
2. Loads each declared input artifact once before attempts. A missing input is
   warned about and represented as empty text.
3. Injects selected skills and input artifacts into the task namespace once.
4. Creates a fresh planner, worker, session, and attempt identity for each
   attempt.
5. Counts a run as successful only when the planner marks the task state
   `done`.
6. Requires `iterations` successful runs within `max_attempts`; failed or
   incomplete attempts consume the attempt budget. Successful runs accumulate.
7. Applies `timeout_s`, when configured, to one whole attempt.
8. Publishes successful state after every successful run as:

   ```text
   <effective-artifact-key>/result
   <effective-artifact-key>/summary
   <effective-artifact-key>/records
   ```

9. Raises a task-not-completed error after exhausting attempts. A sequential
   queue stops at that point unless the enclosing workflow explicitly isolates
   the task or job.

The default effective artifact key is the task-template key. Fan-out stages
normally override it with a stable identifier, never a queue index. Namespace
memory is stored under `user:memory/<namespace>`; vulnerability and
verification agents additionally use the stores named in §1.

Task-runner names are observable because they own sessions, artifact scope,
and checkpoint ownership. `oas_build`, `oas_update`, and `likec4` use
`context.app_name`; planner-driven security tasks use the literal
`contractor`. Under the CLI both values are `contractor`.

### 3.5 Resume semantics

When `checkpoint_path` is absent, every queued task runs. When present:

- the checkpoint workflow owner MUST equal the task-runner name; a mismatch
  starts a fresh checkpoint for that runner;
- lookup is by stable task `ref`;
- template key and resolved template version MUST match;
- the three artifacts derived from the invocation's **current effective
  artifact key** MUST all exist;
- on a match, return a task result marked `restored=true` without running the
  agent;
- otherwise rerun and atomically merge the new completion entry.

Conditional stages therefore use explicit name-based refs. Direct-runner
workflows do not use task checkpoints; trace overlays provide their only
incremental state.

### 3.6 Failure isolation

The standard skippable-job boundary catches ordinary errors, logs them, emits
`task_skipped` with reason `job_failed: <message>`, and returns a caller-supplied
default. Cancellation and keyboard interruption always propagate. Verification
can set this boundary to fail-fast.

| Workflow/stage | Isolation unit | Behavior after ordinary failure |
|---|---|---|
| `oas_build`, `oas_update`, OAS portion of `vuln-assess` | none | Abort on exhausted task or setup/persistence error. |
| `likec4` task chain | none | Abort chain, then attempt overlay persistence in workflow cleanup. |
| `trace` | one OpenAPI path | Skip failed path; continue later paths; cleanup persists completed overlay work. |
| `trace-direct`, `trace-graph` | one operation | Skip failed operation; continue later operations and paths. Artifact-save failure aborts. |
| `trace-graph-pathpar` | one route group | Stop that group, preserve sibling groups, then merge/persist all forks. |
| `trace-postdiff` | one route group including both stages | Stop that group, save accumulated overlay, continue later groups. |
| `trace-verify` | none | Any task or persisted-verdict postcondition failure aborts the workflow. |
| `exploit` | none per finding loop | The first failed assessment or non-isolated persistence failure aborts remaining findings. |
| `vuln-scan` | none | Abort on scan failure. |
| `vuln-scan-fast` | discovery and scan fail-fast; direct trace per finding caught | Continue after a trace-confirm error; exploit child is fail-fast. |
| `vuln-scan-trace` | one trace finding | Scan is fail-fast; failed DFS finding is skipped. |
| `vuln-sweep` | one nomination class and one DFS finding | Continue after failed class or finding. |
| `router` | none | Direct run failure aborts. |

### 3.7 Stable fan-out artifact segments

Finding names and arbitrary namespaces used as a single artifact path segment
MUST use this stable slug algorithm:

1. Preserve the raw value unchanged only if it consists solely of lowercase
   ASCII letters, digits, `_`, or `-`; is at most 160 characters; does not
   start with `h_`; and is not a lowercase Windows device basename (`aux`,
   `con`, `nul`, `prn`, `com1`…`com9`, `lpt1`…`lpt9`).
2. Otherwise trim only for the readable portion, replace runs of all other
   characters with `_`, strip boundary `_`, and use `item` when that result is
   empty.
3. Hash the **original, untrimmed** value with SHA-256.
4. Return a value in the reserved domain
   `h_<bounded-readable>_h<64-lowercase-hex-digest>`, with total length no more
   than 160.

The raw and encoded domains are disjoint. Case changes, whitespace, unsafe
characters, and long values therefore cannot alias one another.

## 4. Workflow configuration

### 4.1 Configuration model

Every workflow module loads a sibling `config.yaml` during module loading. A
missing or invalid file prevents that workflow module from loading. The
supported top-level sections are:

```yaml
budgets:
  <name>: <scalar used by that workflow>
tasks:
  <logical-stage>:
    iterations: 1
    max_attempts: 1
    max_steps: 15
    timeout_s: null
agents:
  <agent-name>:
    output_format: json       # json | xml | yaml | markdown
    with_graph_tools: false
    with_code_exec: false
observations:
  enabled: false
  track_tools: true
  tracked_tools: null
  include_tool_errors: false
  track_skills: true
  track_files: true
  malformed_only: false
  track_file_paths: false
  track_coverage_gap: false
  track_memories: false
  in_record: true
  in_result: true
```

Unknown task/agent fields are rejected by typed construction. An invalid
`output_format` is rejected explicitly. An omitted agent entry resolves to all
defaults. An omitted observations block disables deterministic observations.
Workflow token budgets are context-retention/summarization thresholds, not
model output-token caps. `iterations`, `max_attempts`, `max_steps`, and
`timeout_s` have the task-runner meanings in §3.4.

The `CONTRACTOR_EVAL_OBSERVATIONS` environment value may contain a JSON object
whose fields overlay the YAML observation block for evaluation. Unknown
observation keys and a non-list/non-null `tracked_tools` value are rejected.
Unrecognized top-level YAML sections and budget names are otherwise ignored by
the generic loader. Task body selection follows this precedence:

```text
explicit add-task version
  > CONTRACTOR_TASK_VERSION_<UPPERCASE_TEMPLATE_KEY>
  > task manifest active version
```

### 4.2 Active task-template versions

| Template | Active version | Used by |
|---|---:|---|
| `dependency_information` | `v1` | OAS build, LikeC4, `vuln-assess`, `vuln-scan-fast` |
| `project_information` | `v1` | same discovery chains |
| `oas_update` | `v2` | OAS build and `vuln-assess` |
| `oas_enrich` | `v2` | OAS enrichment |
| `oas_validate` | `v1` | OAS build/enrichment and `vuln-assess` |
| `likec4_build` | `v1` | LikeC4 build |
| `likec4_validate` | `v2` | LikeC4 repair/validation |
| `trace_annotation` | `v3` | all direct/planner trace and DFS-confirm stages |
| `trace_verify` | `v1` | static trace verification |
| `vuln_analytics` | `v1` | post-diff judgement |
| `vuln_scan` | `v3` | simple and BFS/DFS scans |
| `vuln_scan_fast` | `v1` | high-recall fast scan |
| `sink_nomination` | `v1` | vulnerability sweep |
| `exploitability_assessment` | `v4` | live exploitability |

### 4.3 Task-stage intent contracts

The workflow graph is not sufficient by itself: the active task body defines
the minimum work each planner must cause its worker to perform. A compatible
reimplementation MAY phrase the instructions differently, but MUST preserve
these stage outcomes and stop conditions.

| Template | Required stage behavior |
|---|---|
| `dependency_information@v1` | Locate manifest/lock files without descending into dependency/build output; report only runtime dependencies that interact with external systems; tag protocol/database/identity/secrets/crypto roles; cite manifest, import, or configuration evidence; return the prescribed Markdown inventory and explicit gaps. |
| `project_information@v1` | Breadth-first map the project at bounded depth, then inspect only relevant areas; inventory configuration, tests, domain logic, models, crypto, security controls, external integrations, documentation, and API specifications; attach risk notes to configuration/crypto/security paths; return one Markdown project map. |
| `oas_update@v2` | Incrementally build the tool-managed document in small evidence-backed batches; inspect coverage, prefer reusable components, create dependencies before paths, attach `x-path-files`/`x-component-files` containing source-code files, verify each mutation with a targeted read, and report changes/coverage/next work without pasting the schema. |
| `oas_enrich@v2` | Apply the same incremental/provenance/read-back discipline to an existing document, prioritizing unexplored code and richer operation/component/security detail rather than rebuilding blindly. |
| `oas_validate@v1` | Run the serious-issue linter once, treat findings as hypotheses, verify with minimal schema/code reads, apply only the smallest evidence-based tool mutation, relint exactly once, and stop; cosmetic/speculative work is out of scope. |
| `likec4_build@v1` | Consume discovery artifacts, build or extend one security-focused model at `/architecture.c4`, cover actors, entry points, deployables, stores/data classes, integrations, trust zones and security relationships, include landscape/container/security views, validate after meaningful phases, and return the full validated source. |
| `likec4_validate@v2` | Read the `likec4_source` memory note, validate once, make only localized repairs in that note, revalidate once, and return status rather than DSL. The current workflow additionally treats overlay `/architecture.c4` as the persisted canonical file; an implementation must bridge the task's memory-note contract and that overlay contract. |
| `trace_annotation@v3` | Identify the target entrypoint and request parameters; trace at least one concrete path to a sink/terminal; identify validation and auth/authz; add only structured trace/validation/sink annotations; produce fixed sections for inserted annotations, data flow, handler controls, findings, and material uncertainties; stop once the five completion criteria are met. |
| `vuln_analytics@v1` | Enumerate every annotated flow in the supplied diff, confirm it in source, judge it as finding/clean/uncertain, evaluate handler controls, call the finding reporter for each supported issue, and stop after every flow/entrypoint has a verdict. It MUST NOT edit or retrace broadly. |
| `trace_verify@v1` | Load one named upstream report, independently find the external entrypoint and trace data to the sink, record controls and at least two alternatives before a negative verdict, select one of the four verification verdicts, and persist exactly one verification call; task text output is only a handshake. |
| `vuln_scan@v3` | Perform both dangerous-pattern detection and absence-of-control detection over handlers, including missing auth/ownership/role/rate-limit/path-confinement controls; report every confirmed finding with the complete finding shape. |
| `vuln_scan_fast@v1` | Maximize recall in four passes: enumerate entrypoints, sweep dangerous patterns, triage surrounding context without suppressing uncertainty, then audit cross-cutting controls; report even low-confidence candidates, include CWE in details, and list reports at completion. |
| `sink_nomination@v1` | Sweep the entire project for one configured class, nominate rather than deep-confirm, treat missing-control handlers as candidates without requiring taint, report every plausible candidate, and verify nomination coverage before finishing. |
| `exploitability_assessment@v4` | Probe the configured target for exactly one finding, remain on-host and stop at safe proof, send at least one HTTP request, use the finding's endpoint/auth information, compare a credible oracle, and persist one verdict with concrete URL/evidence/request tags. Skipping or concluding without a probe is a process failure. |

Task output prose is a planner/runner handshake. Whenever a task also has a
tool-managed OpenAPI document, finding report, verification record, overlay, or
LikeC4 file, that side-effect artifact is authoritative.

### 4.4 Shipped tuning values

The following values are part of the reproducible default workflow shape.
`I/A/S/T` means iterations, maximum attempts, maximum planner steps, and
attempt timeout seconds.

| Workflow | Agent/context budgets | Task budgets | Other behavior |
|---|---|---|---|
| `oas_build` | SWE/builder/validator `100000` each | dependency `1/2/20/–`; project `1/2/20/–`; update `2/4/20/–`; validate `1/1/20/–` | observations enabled, tool errors off, file paths on |
| `oas_update` | builder/validator `120000` each | enrich `3/6/30/–`; validate `2/2/20/–` | same observations |
| `likec4` | SWE `100000`; builder `120000` | dependency `1/3/20/–`; project `1/3/20/–`; build `3/6/20/–`; validate `1/2/20/–` | same observations |
| `trace` | trace `80000` | annotate `1/3/20/–` | graph tools on; same observations |
| `trace-direct` | trace `100000` | direct single run | graph tools off |
| `trace-graph` | trace `100000` | direct single run | graph tools on |
| `trace-graph-pathpar` | trace `100000` | direct single run | concurrency `3`; group depth `0` |
| `trace-postdiff` | trace `100000`; analytics `100000` | direct single runs | max diff `60000` characters; group depth `1`; graph tools on for both |
| `trace-verify` | verifier `80000` | verify `1/2/20/–` | same observations |
| `exploit` | exploit `80000` | assess `1/2/25/–` | code execution on; same observations |
| `vuln-assess` | SWE/builder/validator `100000` each | same four task budgets as `oas_build` | trace/exploit read their own configs |
| `vuln-scan` | scan `80000` | scan `1/2/75/–` | graph tools on; same observations |
| `vuln-scan-fast` | scan `80000`; SWE `100000` | dependency `1/2/20/–`; project `1/2/20/–`; scan `1/2/50/–` | code-review and trace graph tools on; same observations |
| `vuln-scan-trace` | scan/trace `80000` each | scan `1/2/75/–`; trace `1/1/30/–` | both graph-equipped; same observations |
| `vuln-sweep` | scan `60000`; trace `80000` | sweep `1/2/50/2700`; trace `1/1/30/1200` | class concurrency `3`; trace cap `40`; same observations |
| `router` | every specialist `120000` | planner maximum `20` subtasks | trace graph tools on; same observations |

No positive-range validation is performed for workflow-only concurrency/group
scalars. Deployment configuration MUST supply sensible values; in particular,
a zero semaphore limit would prevent progress.

## 5. OpenAPI decomposition, path keys, and grouping

### 5.1 Operation extraction

All OpenAPI trace consumers share one extraction algorithm. Given a parsed
document:

1. If the document is not a mapping or `paths` is not a mapping, warn and
   return an empty list.
2. Read global `components.securitySchemes`, defaulting to an empty mapping.
3. Visit path entries in document order.
4. Remove and remember path-item extension `x-path-files`. This mutates the
   parsed path-item mapping; callers needing the original MUST copy first.
5. Visit remaining path-item entries in document order. Recognize only the
   lowercase standard operation methods:
   `get`, `put`, `post`, `delete`, `options`, `head`, `patch`, and `trace`.
   Ignore path-level `parameters` and all other keys.
6. If an operation value is not a mapping, warn and skip it.
7. Deep-copy the operation and recursively resolve local references against the
   full document:
   - JSON Pointer `~1` and `~0` escapes are decoded;
   - an in-progress reference cycle becomes `{"$circular_ref": "<ref>"}`;
   - sibling fields next to `$ref` override fields from a resolved mapping;
   - external references remain unresolved; malformed local references or a
     recursion depth above 100 fail resolution for that operation;
   - any resolution error is logged and that operation is skipped.
8. Choose `operation_id` from the resolved operation. Convert a non-null value
   to text and trim it. If absent, null, or empty, use
   `<UPPERCASE-METHOD> <path>`.
9. If path files were present, attach them to the resolved operation as
   `x-path-files`.
10. If global security schemes were present, ensure a resolved operation has a
    `components.securitySchemes` mapping and merge the global schemes into it.
11. Append the operation record `(operation_id, lowercase method, path,
    resolved schema)` to its path record.
12. Keep only paths containing at least one recognized, successfully resolved
    operation.

This preserves path order and method order. Every trace fan-out and verification
namespace derivation MUST use this same result. The defensive guards are
deliberately limited to the document/`paths` and individual operation values;
a malformed non-mapping `components` or path-item value may still abort
extraction rather than being silently ignored.

### 5.2 Per-path planner payload

The planner-driven `trace` workflow collapses every operation on one path into
one task. It MUST:

- create one method mapping keyed by lowercase method;
- remove per-operation `x-path-files` and `components` from method bodies;
- retain the first non-empty `x-path-files` value at path-item level;
- merge all discovered security schemes at document level;
- serialize a document shaped as `{path: methods, components?: ...}` without
  key sorting;
- pass a comma-and-space-separated operation-id string alongside that YAML.

Direct trace variants instead serialize one `{path: {method: schema}}` mapping
per operation.

### 5.3 Versioned path-key algorithm

Path and group keys MUST use version `v2` and be at most 160 ASCII characters,
including their scope prefix.

Define `encode_body(path)`:

1. `/` maps to `p-_root_`.
2. A path beginning with `/` drops that one leading slash and starts with
   `p-`; an invalid relative path retains all text and starts with
   `p-relative_`, keeping it distinct from its absolute counterpart.
3. Encode the remaining text as UTF-8 bytes. Lowercase ASCII letters, digits,
   and `-` remain literal. A slash becomes `__`. Every other byte becomes
   `_<two-uppercase-hex-digits>`.

Define `scoped(path, depth)`:

```text
normalized_depth = max(depth, 0)
scope = "v2/d" + decimal(normalized_depth) + "/"
body = encode_body(path)
if scope + body exceeds 160 characters:
    digest = SHA256(UTF8(path)) as 64 lowercase hex digits
    suffix = "_h" + digest
    truncate body to the available prefix length, strip trailing "_",
    then append suffix
return scope + body
```

If the scope itself leaves no room for at least one readable character plus
the digest suffix, reject the depth.

`openapi_path_key(path)` is `scoped(path, 0)`. Examples:

| Input | Key |
|---|---|
| `/` | `v2/d0/p-_root_` |
| `/users/{id}` | `v2/d0/p-users___7Bid_7D` |
| `/users/id` | `v2/d0/p-users__id` |
| `/a_b` | `v2/d0/p-a_5Fb` |
| `/a/b` | `v2/d0/p-a__b` |
| `/Users` | `v2/d0/p-_55sers` |

Historical unversioned, brace-removing keys MUST NOT be probed or reused.
Migrating to this scheme requires regenerating old trace/report artifacts.

### 5.4 Route groups

For grouping depth `D`:

- `D <= 0`: one group per full path, key `openapi_path_key(path)`;
- `D > 0`: strip outer slashes, split on `/`, discard empty segments, retain
  the first `D` segments, prepend `/`, and call `scoped(prefix, D)`;
- when the route has fewer than `D` segments, retain the requested depth in
  the key. Thus `v2/d1/...` and `v2/d3/...` never co-mingle;
- root at a positive depth is scoped separately from the depth-zero root.

Group construction preserves first-seen group order and original path order
within a group. A group's operation list is the concatenation of each member
path's operation list in that order.

### 5.5 Trace namespaces and common artifacts

The trace family uses logical API namespace `openapi`. Producer namespace
prefixes, in verifier probe order, are:

```text
trace-annotation
trace-graph
trace-graph-pathpar
trace-postdiff
```

The complete producer namespace is
`<prefix>:openapi:<path-or-group-key>`. Trace and trace-direct share
`trace-annotation`; graph, path-parallel graph, and post-diff each have their
own prefix.

All overlay trace variants consume:

```text
oas-openapi-building          OpenAPI seed/current source
trace-openapi-fs              optional serialized overlay patch
```

and persist:

```text
trace-openapi-fs              deterministic overlay patch as JSON
trace-openapi-diff            unified overlay diff with 4 context lines
user:memory/<producer-namespace>
user:vulnerability-reports/<producer-namespace>   when reporting is enabled
```

No trace overlay is applied to the host source tree. The two source-scan DFS
workflows called out later intentionally pass the base filesystem instead.

## 6. OpenAPI and architecture workflows

### 6.1 `oas_build`

**Inputs:** project filesystem; optionally reusable discovery artifacts.

**Outputs:** `user:oas-openapi-building`, task output triplets, and memories.

The task runner name MUST be `context.app_name`. Add variable
`project_path=context.folder_name`. Build one model client and these workers:

| Worker | Filesystem/tools | Context budget |
|---|---|---:|
| `swe_agent` | project analysis, read-only source | `100000` |
| `oas_builder` | read-only source plus code/OpenAPI mutation tools | `100000` |
| `oas_validator` | read-only source plus OpenAPI lint/mutation tools | `100000` |

Queue in this exact order:

| Ref/template | Condition | Namespace | Declared inputs |
|---|---|---|---|
| `dependency_information` | only if `dependency_information/result` is not non-empty | `dependency_information` | none |
| `project_information` | only if `project_information/result` is not non-empty | `project_information` | `dependency_information/result` |
| `oas_update` | always | `openapi-building` | dependency and project results |
| `oas_validate` | always | `openapi-building` | dependency, project, and `oas_update/result` |

Skipped discovery tasks emit `task_skipped` with reason
`artifact_already_exists`. All refs are explicit and name-based so either
conditional omission does not shift checkpoint identity. The builder and
linter mutate the same tool-managed OpenAPI artifact
`user:oas-openapi-building`. The linter runs after the configured two
successful builder iterations and is the final repair gate. Return the task
runner's result list.

### 6.2 `oas_update`

**Inputs:** optional seed text; existing dependency/project result artifacts;
existing tool-managed OpenAPI state when present.

**Outputs:** updated `user:oas-openapi-building`,
`oas_enrich/{result,summary,records}`, and
`oas_validate/{result,summary,records}`.

Before queue construction, persist a truthy context seed at
`oas-openapi-building`. Use a task runner named `context.app_name`, variable
`project_path=context.folder_name`, one `oas_builder` with a `120000` context
budget, and one `oas_validator` with a `120000` budget.

Queue:

1. `oas_enrich` in `openapi-building`, consuming
   `dependency_information/result` and `project_information/result`; default ref
   is `oas_enrich:0`.
2. `oas_validate` in the same namespace, consuming both discovery results and
   `oas_enrich/result`; default ref is `oas_validate:1`.

The workflow does not regenerate missing discovery artifacts; the runner
injects empty text for absent declared inputs. Enrichment requires three
successful iterations within six attempts, then validation requires two
successful iterations within two attempts. Return the task result list.

### 6.3 `likec4`

**Inputs:** project filesystem, optional discovery results, optional
`likec4-architecture.c4`.

**Outputs:** `likec4-architecture.c4` plus task output triplets and memories.

At construction, wrap the project filesystem in a memory overlay for the
architecture builder. Before running tasks, load
`likec4-architecture.c4`; when it has non-empty text, write it to the overlay
at the canonical path `/architecture.c4`.

Use a runner named `context.app_name`, a project-filesystem SWE worker, and an
overlay-filesystem `likec4_builder`. Queue:

| Ref/template | Condition | Namespace | Declared inputs |
|---|---|---|---|
| `dependency_information` | discovery result absent | `dependency_information` | none |
| `project_information` | discovery result absent | `project_information` | dependency result |
| `likec4_build` | always | `likec4-building` | both discovery results |
| `likec4_validate` | always | `likec4-building` | both discovery results and `likec4_build/result` |

All refs are explicit and stable. The build stage requires three successful
runs. The validate stage is a repair pass using the same builder/validator
toolset and requires one successful run.

Workflow cleanup MUST run on both success and failure. If `/architecture.c4`
is visible in the overlay, read it as UTF-8 and save it to
`likec4-architecture.c4`; if not, leave any prior artifact untouched. Cleanup
write errors are logged/suppressed by the shared lifecycle wrapper. The LikeC4
validator capability executes against overlay content and treats an empty issue
list as valid.

## 7. Trace producer workflows

### 7.1 Trace-family comparison

| Key | Unit sent to an agent | Planner | Graph | Concurrency/grouping | Vulnerability author |
|---|---|---|---|---|---|
| `trace` | all operations on one path | yes | on | paths sequential, depth 0 | trace agent |
| `trace-direct` | one operation | no | off | operations and paths sequential, depth 0 | trace agent |
| `trace-graph` | one operation | no | on | operations and paths sequential, depth 0 | trace agent |
| `trace-graph-pathpar` | one operation inside a group | no | shared graph tools | groups bounded-concurrent, configured depth 0 | trace agent |
| `trace-postdiff` | one operation, then one group diff | no | on in both stages | groups sequential, configured depth 1 | analytics agent only |

Each direct operation gets a fresh model session and trace/metrics plugins.
Operations within one path or group remain sequential so later operations see
earlier overlay edits and namespace memory.

### 7.2 `trace`

1. Persist a truthy context seed to `oas-openapi-building`.
2. Load that artifact; if no part exists, fail with `No OpenAPI artifact
   found`.
3. Parse YAML and extract paths using §5.1.
4. Load `trace-openapi-fs` if present, parse its JSON patch, and apply it to one
   shared overlay. Mark the overlay seeded only after successful load.
5. For each path in order, invoke one skippable path job.
6. For a non-empty path, construct the combined payload from §5.2 and namespace
   `trace-annotation:openapi:<path-key>`.
7. Create a fresh task runner named `contractor`, add
   `project_path`, `operation_id`, and `operation_schema`, then queue exactly
   one task:

   ```text
   template:      trace_annotation@v3
   ref:           trace_annotation:openapi:<path-key>
   artifact key:  trace_annotation/openapi/<path-key>
   namespace:     trace-annotation:openapi:<path-key>
   skills:        [trace]
   inputs:        none
   budget:        1 successful run / 3 attempts / 20 planner steps
   worker:        trace_agent over the shared overlay, graph tools on,
                  vulnerability reporting on
   ```

8. On public workflow cleanup, if overlay seeding completed, persist the patch
   and diff even if one or more path jobs failed.

The workflow implementation returns no aggregate value. Path task artifacts
contain the scoped `v2/d0/...` key hierarchy; findings are authoritative under
the producer namespace.

### 7.3 `trace-direct`

The setup and common artifacts match `trace`, but there is no task runner or
checkpoint.

For each path in order:

1. Reload `trace-openapi-fs` into the shared overlay when present. Loading
   resets the overlay before replaying the patch.
2. Inject the `trace` skill once into
   `trace-annotation:openapi:<path-key>`.
3. For each operation in order, run an isolated direct job:
   - render `trace_annotation@v3` with project folder, operation id, and the
     one-operation YAML;
   - build a fresh trace agent over the shared overlay in the path namespace;
   - derive the agent output format from the task template;
   - disable graph tools and enable vulnerability reporting;
   - create a random session and emit under
     `trace_annotation:openapi:<operation-id>`;
   - assign task id equal to the operation's zero-based index within the path.
4. Save the full overlay patch and diff after the path, even when an individual
   operation was skipped.

An overlay-save failure aborts later paths. The producer prefix is deliberately
the same as planner-driven `trace`.

### 7.4 `trace-graph`

Implement exactly the `trace-direct` algorithm with these differences:

- producer namespace prefix is `trace-graph`;
- graph tools are enabled on every freshly built trace agent;
- event name is `trace_graph:openapi:<operation-id>`.

The call-graph attachment may lazily build and cache an engine behind the
filesystem, allowing subsequent operations to reuse it. Overlay load/save and
per-operation failure isolation remain identical to `trace-direct`.

### 7.5 `trace-graph-pathpar`

This workflow is the reusable parallel trace stage in `vuln-assess`.

1. Seed/load/parse the OpenAPI artifact and load a previous overlay patch once.
2. Build graph-tool closures once against the main overlay. They are shared
   read-only by all operation agents.
3. Group paths with the configured `group_depth` (default 0).
4. Snapshot the main overlay patch and its in-memory file-content mapping.
5. Create one independent overlay fork per group by replaying the snapshot over
   the same read-only base filesystem.
6. Run group jobs in structured concurrency, limited by a semaphore of
   `max_concurrency` (default 3). Each group gets its own direct runner named
   `context.app_name`.
7. Within a group:
   - namespace is `trace-graph-pathpar:openapi:<group-key>`;
   - inject `trace` once;
   - flatten member operations and execute them sequentially;
   - build each trace agent over the group's fork, pass the shared graph-tool
     closures, enable vulnerability reporting, and use event name
     `trace_graph_pathpar:openapi:<operation-id>`;
   - operation task id is its zero-based index in the flattened group.
8. Wrap the whole group, not each operation, in a skippable boundary. Therefore
   one failed operation prevents later operations in that group but does not
   cancel sibling groups.
9. In a `finally`-equivalent boundary, merge every fork into the main overlay
   and persist patch/diff. Conflicting writes to the same file are logged; the
   merge selects the longest byte representation (first encountered wins an
   equal-length tie). New post-fork deletions propagate.

The merge/save boundary runs on normal completion, ordinary group failure, and
structured-concurrency unwinding. A merge or artifact-save error itself is not
isolated. Callers may override `max_concurrency` when constructing this
workflow; grouping depth is read from workflow configuration.

### 7.6 `trace-postdiff`

This is a two-stage navigation/judgement split. The trace agent cannot write
findings; the analytics agent is the sole report author.

1. Seed/load/parse the OpenAPI artifact.
2. Group paths at configured depth 1 by default.
3. Process groups sequentially. Before each group, replay a persisted overlay
   patch when present.
4. Run the entire group within one skippable boundary using namespace
   `trace-postdiff:openapi:<group-key>` and one injected `trace` skill.
5. Snapshot the overlay's current file-content mapping.
6. **Stage A — annotation:** for every flattened operation, render the trace
   template and run a graph-equipped trace agent with vulnerability reporting
   disabled. Event name is
   `trace_postdiff:openapi:<operation-id>`.
7. Compute files newly added or content-modified since the snapshot. Ignore
   deletions. If there are none, skip Stage B.
8. **Stage B — analytics:** filter the full four-context-line overlay diff to
   chunks whose `diff --overlay a<path> b<path>` header identifies a changed
   file. Truncate to `analytics_diff_max_chars` (default 60000) and append a
   marker instructing the agent to read files for the remainder:
   `... [diff truncated — read the annotated files directly for the rest]`.
9. Serialize all group operations as a YAML target summary and render
   `vuln_analytics@v1` with the target summary and filtered diff.
10. Run one graph-equipped `vuln_analytics_agent` over the same overlay and
    namespace. It MUST use `report_vulnerability` for supported findings.
    Event name is `trace_postdiff:openapi:<group-key>:analytics`, and its task
    id is the number of operations in the group.
11. After the group boundary, persist the overlay patch and full diff whether
    the group succeeded or was skipped. Then continue to the next group.

An operation failure stops the rest of Stage A and prevents analytics for that
group. No planner-task result triplets are created by this direct workflow.

## 8. Static trace verification

### 8.1 Discovery

`trace-verify` seeds and loads `oas-openapi-building`, failing when no artifact
part exists, then extracts paths. For each path, derive these candidates in
this exact order:

| Producer prefix | Key depth |
|---|---:|
| `trace-annotation` | `0` |
| `trace-graph` | `0` |
| `trace-graph-pathpar` | that producer's currently loaded `group_depth` |
| `trace-postdiff` | that producer's currently loaded `group_depth` |

The candidate is `<prefix>:openapi:<derived-key>`. Probe only the current
configured depth; never scan other depths or legacy keys. A grouped namespace
may correspond to several sibling paths, so keep a workflow-run set of already
processed source namespaces and verify each at most once. Probe all distinct
producer prefixes; the same logical finding under two prefixes is verified
twice in its respective namespace.

For every non-empty finding store, create one task runner named `contractor`.
Queue one task for each finding having a truthy name:

```text
template:      trace_verify@v1
ref:           trace_verify:openapi:<current-path-key>:<finding-name>
artifact key:  trace_verify/<slug(source-namespace)>/<slug(finding-name)>
namespace:     <source-namespace>
inputs:        none
budget:        1 successful run / 2 attempts / 20 planner steps
```

Pass `project_path` globally and pass finding name/title/place type/place,
severity, confidence, summary, and source namespace as task parameters. The
worker has read-only source/code access, read-only access to the upstream
finding store, and write access to the verification store for that same source
namespace. An after-model guard MUST require exactly the terminal
`report_verification` capability to have been used before completion, allowing
at most three reminder turns. The task contract instructs the worker to call it
once; the callback's hard invariant is presence rather than call-counting.

### 8.2 Authoritative persistence and freshness

The verification store, not the task result, is authoritative. Before running
tasks, load existing verification rows keyed by each row's explicit name or its
mapping key. After the runner returns:

1. Every named queued finding MUST exist at
   `user:vulnerability-verifications/<source-namespace>`.
2. If a queued task ran freshly, its entire parsed verification mapping MUST
   differ from the row loaded before the task runner.
3. If a queued task was restored from a validated checkpoint, existence is
   sufficient; unchanged content is permitted.
4. A missing or unchanged-required row raises `MissingVerificationError` and
   aborts the workflow.

The outer per-path call explicitly disables skippable failure swallowing. If no
findings are discovered under any producer, log a warning and complete as a
no-op. Raw nameless rows count toward the discovery total but do not create
tasks or postconditions; a faithful implementation should preserve this edge
behavior.

## 9. Source vulnerability workflows

### 9.1 `vuln-scan`

Create one runner named `contractor`, add
`project_path=context.folder_name`, and queue:

```text
template:      vuln_scan@v3
ref:           vuln-scan:full
namespace:     vuln-scan
skills:        [vuln_scan]
params:        project_path=<folder>
budget:        1 successful run / 2 attempts / 75 planner steps
worker:        graph-equipped codereview_agent, read-only source,
               vulnerability reporting enabled, context budget 80000
```

The report tool persists findings at
`user:vulnerability-reports/vuln-scan`. The task also publishes
`vuln_scan/{result,summary,records}`. There is no downstream verification.

### 9.2 `vuln-scan-trace`

This workflow performs BFS discovery then sequential DFS confirmation.

**Phase 1:** queue the same `vuln_scan@v3` shape under namespace
`vuln-scan-trace:scan`, ref `vuln-scan-trace:scan`, and budget
`1/2/75`. Load findings from
`user:vulnerability-reports/vuln-scan-trace:scan`. Sort them stably by severity
`critical`, `high`, `medium`, `low`, then unknown. If none, warn and return.

**Phase 2:** for each finding in sorted order, skip rows without both `name`
and `place`. Build a textual operation schema containing title, file, severity,
summary, and details. Create one runner named `contractor` and queue:

```text
template:      trace_annotation@v3
ref:           vuln-scan-trace:trace:<name>
artifact key:  trace_annotation/<slug(name)>
namespace:     vuln-scan-trace:trace:<name>
skills:        [trace]
params:        operation_id=<name>, operation_schema=<finding text>
budget:        1 successful run / 1 attempt / 30 planner steps
worker:        graph-equipped trace_agent, vulnerability reporting enabled
```

Each phase-2 runner is a skippable job. Unlike OpenAPI trace workflows, this
agent receives the base project filesystem, not a memory overlay; structured
trace annotations may therefore modify that filesystem implementation. The
authoritative phase-2 report is
`user:vulnerability-reports/vuln-scan-trace:trace:<name>`.

### 9.3 `vuln-sweep`

`vuln-sweep` inherits the DFS implementation above but substitutes its own
namespace and tuning. Its first pass comprises exactly these five ordered
classes:

| Class key | Nomination surface |
|---|---|
| `injection` | SQL/NoSQL/ORM, command, template, LDAP, and expression sinks. |
| `deserialization` | Unsafe object construction/deserialization. |
| `ssrf-fileio` | Input-derived outbound requests, file paths, and redirects. |
| `secrets-crypto` | Hardcoded secrets, debug settings, weak crypto/randomness. |
| `missing-access-control` | Per-handler absence of authentication, authorization, or ownership checks. |

Run one class job per entry with structured concurrency and a semaphore of
three. Each job creates a runner named `contractor` and queues:

```text
template:      sink_nomination@v1
ref:           vuln-sweep:sweep:<class-key>
artifact key:  sink_nomination                 # shared default
namespace:     vuln-sweep:sweep:<class-key>
skills:        [vuln_scan]
params:        project_path, sink_class, class_guidance
budget:        1 successful run / 2 attempts / 50 steps / 2700 s per attempt
worker:        graph-equipped codereview_agent, context budget 60000
```

Each class catches and logs ordinary failure. The class-specific finding store,
not the shared task output triplet, is authoritative. All class tasks currently
publish to the same `sink_nomination/{result,summary,records}` key; concurrent
last-writer behavior for those non-authoritative summaries is part of the
current implementation.

After all class jobs settle:

1. Load class report stores in the class order above.
2. Deduplicate by exact `(string(place), string(name))`, preserving the first
   row seen.
3. Stable-sort by severity (`critical`, `high`, `medium`, `low`, unknown) then
   confidence (`high`, `medium`, `low`, unknown).
4. If empty, warn and return.
5. Retain only the first 40 nominations when above the configured cap.
6. Trace each survivor sequentially using the inherited phase-2 algorithm,
   namespace/ref prefix `vuln-sweep:trace:`, context budget `80000`, and
   per-attempt timeout `1200` seconds.

The DFS trace uses the base project filesystem and may insert annotations there.
There is no exploit stage.

### 9.4 `vuln-scan-fast`

This five-stage pipeline intentionally carries high-recall candidates forward.

#### Stage 1 — discovery

Create one runner named `contractor`. Conditionally queue the same explicit-ref
dependency and project tasks used by `oas_build`; run the runner only if at
least one task was queued. Existing non-empty result artifacts emit skip
events.

#### Stage 2 — fast scan

Queue `vuln_scan_fast@v1`, ref `vuln-scan-fast:full`, namespace
`vuln-scan-fast`, skill `vuln_scan`, budget `1/2/50`, and a graph-equipped
code-review worker with context budget `80000`. The authoritative raw report is
`user:vulnerability-reports/vuln-scan-fast`.

#### Stage 3 — deterministic deduplication

For each loaded finding, derive:

```text
normalized_place = lowercase(string(place or "").strip leading/trailing "/")
cwe = digits from the first regex match of "CWE-(digits)" in string(details or "")
      or "" when absent
key = (normalized_place, cwe)
```

Keep the first finding per key unless a later finding has a strictly higher
confidence rank (`high=3`, `medium=2`, `low=1`, anything else `0`). Replacing a
value does not change its bucket order. This means unrelated findings in the
same file with no CWE collapse into one bucket; preserve this exact behavior.
Return early when no buckets remain.

#### Stage 4 — direct trace confirmation

Create one shared, non-cached memory overlay over the project and one direct
runner. For every deduplicated finding in order:

- derive namespace `trace-confirm:<name>`;
- inject `trace`;
- build a graph-equipped trace agent over the shared overlay, reporting
  vulnerabilities;
- send a manually composed message containing title, file, and the first 500
  characters of details, asking the agent to confirm or deny;
- emit under `trace_confirm:<name>`;
- catch/log ordinary run errors and continue.

The overlay is neither persisted nor applied to the base source. Confirmed
reports may be written to
`user:vulnerability-reports/trace-confirm:<name>`, but the next stage does
**not** filter against those reports: every original deduplicated scan finding
continues to exploitation.

#### Stage 5 — optional exploitability

If no configured target URL exists, emit a skipped `exploit` event and return.
Otherwise serialize named deduplicated findings as a YAML mapping keyed by
name, construct a child context with that text as its seed, and invoke the
`exploit` implementation directly. Duplicate names use ordinary mapping
last-writer behavior. This child produces no nested workflow lifecycle events.

## 10. Live exploitability

### 10.1 Preconditions and finding fan-out

Construction of `exploit` MUST read runtime settings and reject an absent
target URL with a usage-level error. It also captures optional HTTP proxy,
Caido URL, and Caido auth token settings. When Caido is configured but no
explicit proxy is supplied, the exploit agent routes HTTP through the Caido URL
and disables TLS verification for that proxy route.

On run:

1. Persist a truthy context seed to `vulnerability-reports-seed`.
2. Load that seed with the finding-loader contract. If empty, warn and return.
3. Process findings sequentially. Skip a row with no truthy name.
4. For finding `F`, derive:
   - source namespace `exploitability:F`;
   - opaque request tag prefix `r` plus the first 10 lowercase hex digits of
     SHA-1 over UTF-8 `F`;
   - task ref `exploitability:F`;
   - artifact key `exploitability_assessment/<slug(F)>`.
5. Build one runner named `contractor`, add project folder and target URL, and
   queue `exploitability_assessment@v4` with budget `1/2/25`.
6. Inject skills `[exploit, code-exec, auth]` and append `caido` when configured.
7. Build an exploitability agent with read-only source/code access, HTTP tools,
   optional Caido tools, container code-execution tools, read-only access to
   the seed finding under the source namespace, and verification-write tools.
8. Require at least one of `submit_verdict` or `report_verification` before the
   agent may finish, with at most three reminder turns.
9. Record the run-start epoch in milliseconds immediately before executing the
   runner, then collect optional HTTP evidence after runner success.

Verdicts are stored at
`user:vulnerability-verifications/exploitability:F`. There is no separate
workflow postcondition equivalent to `trace-verify`; the mandatory terminal
tool callback is the enforcement boundary.

### 10.2 Request-chain collection

When Caido is absent, evidence materialization is a no-op. Otherwise:

1. Read the finding's verification row and coerce non-empty
   `evidence_request_ids` to strings.
2. Query Caido history for the full current-finding sequence using a raw-header
   filter for `X-Request-Id: <tag-prefix>-`, maximum 50 rows. History arrives
   newest-first and is reversed to oldest-first.
3. Drop rows whose parsed creation epoch is below the current run start.
   Missing/unparseable epochs are zero and are dropped when a run boundary is
   supplied.
4. Compute the upper median of non-zero response lengths.
5. If cited ids exist, query each exact tag and add cited requests first,
   deduplicated by Caido request id; then add uncited anomalous full-sequence
   rows. A row is anomalous when status is at least 500 or a positive body
   length is at least twice, or at most half, the positive median.
6. If no cited ids exist, use the full tagged sequence as fallback evidence.
7. Fetch raw detail for each candidate in order, skip detail errors, and stop at
   50 exchanges.
8. Render a text document containing finding name, count/source breakdown, and
   for each exchange the tag, method, reconstructed scheme/host/path, raw
   request, and raw response.
9. Save it at `user:exploit-http-chains/F`.

History/detail failures are logged and treated as no chain. The Caido client is
closed in a finalization boundary. The final artifact save is outside the
backend-fetch catch and therefore may still abort the workflow. A failure for
one finding stops the sequential finding loop.

## 11. Composite full assessment

### 11.1 `vuln-assess`

This pipeline composes OAS build, path-parallel trace, and live exploitation.

#### Steps 1–3 — OAS stage

Persist a truthy context seed to `oas-openapi-building`. If
`user:oas-openapi-building` is already non-empty, emit one skipped `oas_stage`
event and skip the whole OAS runner. Otherwise assemble the exact four-task OAS
chain from §6.1 with runner name `contractor` and the `vuln-assess` tuning.
Dependency/project tasks remain individually skippable.

#### Bridge

If `oas-openapi-building` already exists and is non-empty, leave it unchanged.
Otherwise load `user:oas-openapi-building`; if missing, warn and continue. If
present, save that part as `oas-openapi-building`. This bridge exists because
the OpenAPI tools write the explicit user-scoped key while trace workflows
read the bare seed/current key.

#### Step 4 — trace

If the bare OpenAPI artifact is not non-empty, log an error and skip trace
without raising. Otherwise construct `TraceGraphPathParWorkflow` with the same
context and invoke its implementation directly. Its inline finally-merge/save
contract is why bypassing the public lifecycle does not lose overlay output.

#### Step 5 — exploit

If no target URL is configured, emit skipped `exploit` and return. Otherwise
collect findings:

1. Start with all rows from `vulnerability-reports-seed`.
2. Load `oas-openapi-building`; invalid YAML is ignored.
3. Extract paths and derive exactly the current path-parallel trace group key
   using that producer's configured `group_depth`.
4. Deduplicate derived namespaces across sibling paths.
5. Load each
   `user:vulnerability-reports/trace-graph-pathpar:openapi:<group-key>` mapping
   and merge it into the seed mapping in path/group order. Later rows overwrite
   an earlier row of the same name.
6. If the merged mapping is empty, emit skipped `exploit` with reason
   `no_findings`.
7. Otherwise serialize it without sorting, construct an exploit child context,
   and invoke the exploit implementation directly.

Only the `openapi` trace namespace is collected. Neither legacy keys nor other
trace producer prefixes are considered. The composite does not run
`trace-verify`; its verification stages are OAS lint/repair and optional live
exploitability.

## 12. Prompt router

`router` requires `trim(context.prompt)` to be non-empty and rejects an absent
prompt. It does not use a task checkpoint or publish standard task result
triplets.

Build one model and five specialist agents over the shared namespace `router`:

| Specialist | Key behavior |
|---|---|
| `swe_agent` | General project analysis over the source filesystem. |
| `oas_builder` | Source analysis plus mutation of `user:oas-router`. |
| `oas_linter` | Lint/repair of the same router OpenAPI namespace. |
| `trace_agent` | Base-filesystem trace, graph tools and vulnerability reporting enabled. |
| `http_agent` | HTTP interaction under the router namespace. |

Give each specialist the common `120000` context budget. Wrap them in a router
agent whose input schema is `Subtask` and output schema is
`SubtaskExecutionResult`. The router already implements its own dispatch
protocol, so planner worker instrumentation MUST be disabled.

Build a planning agent named `router`, namespace `router`, with the router as
its worker, a 20-subtask maximum, and configured observations. Inject the
`trace` skill before running. Initialize task id 0 state as:

```text
global task id = 0
objective      = original trimmed prompt
status         = running
current        = null
result         = ""
summary        = ""
pool           = []
```

Create a random session id, attach trace and metrics plugins scoped to task
`router`, iteration 1, and invoke one direct runner named
`context.app_name`. Send the prompt both as the model message and planner
objective. Return the direct runner result. Specialist side effects are the
router's outputs; notably the trace specialist receives the base filesystem,
so any permitted annotations are not protected by the OpenAPI trace overlay.

## 13. Artifact and namespace matrix

The matrix lists authoritative or user-visible workflow products. Standard
task output triplets and `user:memory/<namespace>` exist in addition where a
task runner is used.

| Workflow | Primary inputs | Authoritative/user-visible outputs |
|---|---|---|
| `oas_build` | source, optional discovery results | `user:oas-openapi-building`; four task triplets |
| `oas_update` | optional `oas-openapi-building` seed; discovery results | updated `user:oas-openapi-building`; enrich/validate triplets |
| `likec4` | source; optional prior C4 artifact | `likec4-architecture.c4`; four task triplets |
| `trace` | `oas-openapi-building`; optional overlay patch | patch/diff; per-path `trace-annotation` findings and task triplets |
| `trace-direct` | same | patch/diff; per-path `trace-annotation` findings |
| `trace-graph` | same | patch/diff; per-path `trace-graph` findings |
| `trace-graph-pathpar` | same | merged patch/diff; per-group `trace-graph-pathpar` findings |
| `trace-postdiff` | same | patch/diff; per-group `trace-postdiff` analytics findings |
| `trace-verify` | OpenAPI plus trace finding stores | paired per-source verification stores; per-finding task triplets |
| `vuln-scan` | source | `user:vulnerability-reports/vuln-scan`; scan triplet |
| `vuln-scan-trace` | source | scan report; per-finding trace reports/triplets; possible source annotations |
| `vuln-sweep` | source | five class reports; capped per-finding trace reports; shared nomination triplet |
| `vuln-scan-fast` | source; optional live target | raw fast-scan report; trace-confirm reports; optional verdicts/chains |
| `exploit` | `vulnerability-reports-seed`; live target | per-finding verification, HTTP session/body state, optional raw chains, task triplets |
| `vuln-assess` | source, optional OAS seed, optional live target | OAS, trace patch/diff/reports, optional verdicts/chains |
| `router` | prompt and source | specialist-dependent OpenAPI, findings, memories, HTTP state, or source annotations |

## 14. Reconstruction acceptance criteria

A workflow-layer reimplementation is conforming when automated tests prove at
least the following:

1. The registry key set and key-to-workflow mapping exactly match §2.
2. Conditional discovery skips do not change stable refs or downstream
   artifact declarations.
3. Every standard HTTP method is extracted; missing/null operation ids receive
   deterministic fallbacks; a malformed operation is skipped.
4. The path-key examples in §5.3 match byte-for-byte, long Unicode paths remain
   at most 160 ASCII bytes, and legacy collisions (`/` vs `/root`,
   `/users/{id}` vs `/users/id`, `/a_b` vs `/a/b`) remain distinct.
5. Group depth is encoded into keys and group ordering is first-seen stable.
6. Every trace producer writes the same prefix/depth that `trace-verify` and
   `vuln-assess` probe.
7. Direct trace operations are sequential within a path/group; path-parallel
   groups obey the semaphore and merge completed sibling overlays after a
   failure.
8. Post-diff analytics runs only after changed annotations, receives only
   changed-file diff chunks, and is the only report author in that workflow.
9. A successful fresh verifier run without a new/changed persisted verdict
   fails; a checkpoint-restored verifier may reuse its existing verdict.
10. Fast-scan deduplication, sweep class order/dedup/sort/cap, and severity
    ordering reproduce the algorithms above exactly.
11. Exploit request tags are deterministic and opaque; chain collection is
    current-run scoped, oldest-first, cited-first, anomaly-augmented, and capped
    at 50 exchanges.
12. Missing target URL behavior differs correctly: `exploit` constructor
    rejects it, while `vuln-assess` and `vuln-scan-fast` skip only their final
    exploit stage.
13. Public workflow cleanup and event failure behavior matches §3.2, including
    persistence of LikeC4/trace overlays after ordinary workflow failure.
14. Task and workflow configuration values, active task versions, and direct
    versus planner execution match §4 and the per-workflow algorithms.
