# Agents, Callbacks, Task Tools, Observations, and Skills Specification

## 1. Purpose and terminology

This document specifies how agents are assembled, how a planner delegates work to a domain worker, how callback middleware constrains model/tool behavior, how subtask state changes, and how prompt manifests and skills become runtime context. It is language and framework neutral: names identify compatibility contracts, not required implementation technologies.

Terms:

- **LLM agent** — named object with instruction text, model, tools, optional input/output schemas, and lifecycle callbacks.
- **Agent tool** — an LLM agent exposed to another agent as a callable tool. Calling it creates a child agent invocation and propagates state deltas back to the caller.
- **Worker** — domain agent that can read/search/edit/probe/report through a specialized tool palette.
- **Planner** — coordination-only agent that owns a bounded subtask plan and calls the worker through `execute_current_subtask`.
- **Queued task** — TaskRunner-level unit described in `03-runtime-orchestration.md`. One attempt at a queued task creates one planner agent and session.
- **Subtask** — planner-level unit executed by one worker child invocation.
- **Invocation ID** — engine-provided identity for one agent invocation. This differs from TaskRunner's queue-time invocation ID.

## 2. Base agent and model contract

Every LLM agent must support:

```text
Agent {
  name: stable identifier,
  description: short capability description,
  instruction: prompt text,
  model: model descriptor,
  tools: ordered list of functions or agent tools,
  input_schema: optional schema,
  output_schema: optional schema,
  output_key: optional session-state destination,
  callbacks: callback-type-to-chain map
}
```

The common model factory defaults to model alias `lm-studio-qwen3.6` with a 300-second client timeout. Optional global temperature and top-p values are forwarded only when configured; leaving them null preserves backend defaults. The default model object is a shared singleton, so per-request enforcement must never mutate model/client shared state.

The model client performs two outbound compatibility adaptations:

1. In tool parameter schemas, rename a property literally named `$ref` to `ref` and update the corresponding `required` entry. Preserve genuine string-valued JSON-Schema `$ref` references. Remove every `examples` key recursively. This is idempotent and is applied in place to the outbound tool schema.
2. Read task-local forced `tool_choice` and `response_format` values immediately before each completion call and merge them into that call's arguments. A forced value wins over ordinary configuration. A forced response format is supplied only when non-null and the call does not already carry a truthy response format.

Back-end data models that expose OpenAPI `$ref` fields must accept both external `$ref` and sanitized `ref` names.

## 3. Prompt manifests

### 3.1 Standard agent prompt layout

Each standard agent directory has:

```text
agents/<agent-name>/prompt.yml
agents/<agent-name>/prompts/<version-file>.md
```

Manifest schema:

```yaml
active: v3
versions:
  v2:
    file: prompts/v2.md
  v3:
    file: prompts/v3.md
```

Loading behavior:

1. Manifest must exist and parse as a map with string `active` and a version map whose entries contain `file`.
2. Select explicit requested version when supplied; otherwise select `active`.
3. Reject an undeclared version or missing referenced file.
4. Return UTF-8 prompt text and concrete version.

Most agent modules load their default prompt into a constant at module initialization. Changing a manifest during a live process therefore does not change an already imported default. Builders that expose a direct `prompt` argument let workflows override that constant.

For run provenance, scan agent subdirectories in sorted order; for each directory with `prompt.yml`, attempt to load its active version. Return `{agent_name: resolved_version}` and silently skip broken manifests.

The current prompt manifests select:

| Agent | Active prompt |
|---|---|
| `codereview_agent` | `v3` |
| `exploitability_agent` | `shannon` |
| `http_agent` | `v1` |
| `librarian_agent` | `v1` |
| `likec4_builder_agent` | `v3` |
| `oas_builder_agent` | `v4` |
| `oas_linter_agent` | `v1` |
| `planning_agent` | `v5` |
| `router_agent` | `v2` |
| `swe_agent` | `v2` |
| `swe_edit_agent` | `v2` |
| `threat_model_agent` | `v1` |
| `trace_agent` | `converge` |
| `trace_verifier_agent` | `v1` |
| `triage_agent` | `v1` |
| `vuln_analytics_agent` | `v1` |
| `web_exploitability_agent` | `v4` |

Every version declared in each manifest remains selectable behavior and must be retained as reconstruction data, even when inactive. Domain prompt Markdown is authoritative behavioral configuration: factories load it rather than synthesizing an equivalent prompt from code.

### 3.2 Section-based analyzer prompts

The legacy OpenAPI analyzer uses a separate YAML prompt system. Each section file contains optional `format`, optional `role` (default `Professional Software Engineer`), and one or more named tasks with `objective`, optional `instructions`, and optional `examples`. A task formats as:

```text
ROLE:
<role>

OBJECTIVE:
<objective>

[INSTRUCTIONS and EXAMPLES when non-empty]
OUTPUT FORMAT:
<format>
```

Missing section files are errors.

## 4. Standard worker factory

### 4.1 Inputs and defaults

The common worker factory accepts name, instruction, description, ordered tools, output format, agent-specific context-limit summary bullets, optional model, and these defaults:

| Setting | Default |
|---|---:|
| maximum worker tokens before summary enforcement | 80,000 |
| function-result elision enabled | true |
| default heavy tools | `read_file`, `grep`, `glob`, `list_symbols` |
| recent eligible results retained | 15 |
| retained eligible result character budget | global setting, default 0/disabled |
| repeated identical-call threshold | 5 |

At the common `build_worker` layer, an explicit elision target list replaces
the default heavy list. If elision is disabled or the resolved target list is
empty, do not register the elision callback. A positive global recent-result
override replaces the factory's count value; zero means use the factory value.
Three specialized wrappers preserve a current truthiness quirk: code review,
exploitability, and web exploitability resolve their argument as
`explicit_list OR hard_coded_default`, so passing `[]` to those wrappers
restores their hard-coded list rather than disabling elision. Use their
separate enable/disable switch to disable it. Librarian passes an explicit
empty list through normally.

### 4.2 Callback registration order

Construct a fresh callback adapter named after the worker and register exactly:

1. `TokenUsageCallback` — after model.
2. `SummarizationLimitCallback` — before model; depends on token usage.
3. Optional `FunctionResultsRemovalCallback` — before model.
4. `InvalidToolCallGuardrailCallback` — after model, with `default_tool` as fallback and fallback argument `meta`.
5. `RepeatedToolCallCallback` — before tool.

The summarization message is the agent-specific bullets followed by the complete generated worker result contract from section 8.2. Its limit callback uses total tokens, the configured forced tool choice (default `none`; blank disables forcing), and a forced JSON response schema when worker format is JSON.

The JSON response schema is the `SubtaskExecutionResult` schema after outbound-schema sanitization, with `task_id` replaced by `{type: string, const: ""}`. An empty ID allows text parsing to backfill the actual current subtask ID instead of accepting a hallucinated ID when termination is forced.

Return a new LLM agent with the supplied tool order and callback chains. The common factory itself does not set worker input/output schemas; planner instrumentation does that later.

Callback instances are mutable and belong to the constructed agent instance. Standard TaskRunner usage constructs a fresh worker for each queued-task attempt. Reusing one worker agent concurrently would share counters/latches and is outside the supported concurrency contract.

## 5. Specialized agent factories

Every standard worker begins its tool order with `default_tool`, then appends the palette below. Memory tools are scoped by `namespace` and use the worker output format. File tools use the same format. Graph tools are attached only when explicitly enabled and the filesystem supports local graph construction, unless an explicit graph-tool list is supplied.

| Factory | Tool palette and special behavior |
|---|---|
| Software engineering | Read-only files, memory, code-navigation, optional graph. Coverage-capable file setup may capture the full in-scope set. |
| Software editing | Read/write files with interaction tools, memory, code-navigation, optional graph. |
| Code review | Read-only files with interaction tools, memory, code, optional graph, vulnerability report read/write. Default elision is the four source-heavy tools. Coverage capture is optional. |
| Trace annotation | Read/write filesystem is created, but generic source mutators `insert_line`, `edit`, `replace_range`, `write_file`, `append_file`, `mkdir`, `rm`, `cp`, and `mv` are removed. Memory, code, structured annotation, optional graph, and optional vulnerability reporting remain. Read-side diff/changed-path/restore tools remain available. |
| Trace verifier | Read-only files, memory, code, optional graph, read-only upstream vulnerability tools, verification tools. It cannot author upstream vulnerabilities. Requires `report_verification` before finish through mandatory-tool enforcement. |
| Post-trace vulnerability analytics | Read-only annotated-overlay files with interaction tools, memory, code, optional/explicit graph, vulnerability reporting. |
| OpenAPI builder | Read-only files, memory, mutable OpenAPI tools, code, optional graph. Memory namespace and OpenAPI artifact name may differ so a new reasoning wave can refine the same schema without inheriting old memory. |
| OpenAPI linter | Read-only files without interaction tools, memory, mutable OpenAPI and lint tools. Heavy-result elision disabled. |
| LikeC4 builder | Read/write files with interaction tools, memory, code, optional graph, LikeC4 validation/manipulation tools. |
| HTTP worker | HTTP request/history/body tools plus memory. Heavy-result elision disabled. |
| Exploitability | Read-only files, memory, code, optional graph, optional sandboxed code execution, HTTP, optional Caido tools, read-only upstream vulnerabilities, and verification tools. If Caido URL exists and no explicit proxy is supplied, route HTTP through Caido and disable TLS verification. Requires either verdict alias. |
| Web exploitability | Memory, optional sandboxed code execution without filesystem, HTTP, optional Caido, read-only upstream vulnerabilities, verification. No source filesystem/code/graph tools. Requires either verdict alias. The lite variant delegates here with Caido disabled. |
| Librarian | Read-only files with interaction tools, memory, code, and cross-namespace artifact-pool read/search. Its only write surface is its own memory namespace. By default only `pool_read` and `pool_search` are elided as heavy results. |
| Threat model | Read-only files, memory, code, optional graph, vulnerability reporting, and optionally only the read operations from the OpenAPI tool set. Currently not wired into production workflows. |
| Triage | Read-only files, memory, code, optional graph, vulnerability report tools. Currently not wired into production workflows. |

The exact read-only OpenAPI subset for threat modeling is `list_paths`, `list_components`, `list_servers`, `get_info`, `get_path`, `get_component`, and `get_full_openapi_schema`.

The vulnerability read-only subset is exactly `get_vulnerability` and `list_vulnerabilities`. Verdict aliases are exactly `submit_verdict` and `report_verification`.

Agent-specific heavy-result defaults are compatibility-significant:

- code review: `read_file`, `grep`, `glob`, `list_symbols`;
- exploitability: those four plus `http_request`, `http_read_body`, `caido_history`, `caido_request_detail`, `caido_automate_results`, and `caido_workflow_findings`;
- web exploitability: the same network/Caido set without the four source tools;
- librarian: only `pool_read` and `pool_search`;
- HTTP and OpenAPI linter: elision disabled;
- other standard workers: the common four-tool set when those tools exist, unless explicitly overridden.

Exploitability and web-exploitability HTTP tools use a 512-character response-body preview and accept a request-tag prefix. Their upstream-read and verification stores use `source_namespace ?? namespace`; ordinary memory and HTTP history use `namespace`.

### 5.1 Router agent

The router is not built with the standard worker factory. It requires at least one sub-agent, wraps each as an agent tool, adds namespace memory tools, and registers token usage, invalid-tool fallback, and repeated-call protection at threshold 3. Its instruction tells it to choose a specialized sub-agent and return the delegated result.

### 5.2 Legacy OpenAPI analyzer composition

The standalone OpenAPI analyzer is a sequential composite:

1. `AnalyticAgent` runs one review LLM agent whose structured output is stored at `oas_analyzer::service_information`.
2. It then runs every task-specific agent from `appsec`, `datasec`, and `ddos` section prompts. Each receives `save_vulnerability`.
3. The review child runs with the original invocation context. Each
   task-specific child from `appsec`, `datasec`, and `ddos` gets a copied
   context with isolated branch identifier `<analytic-agent>.<child-agent>`
   appended to any existing branch; all children share session state.
4. `save_vulnerability` lowercases method/severity/confidence, derives tag from the child's name prefix before `_`, validates the endpoint-vulnerability schema, appends it to `oas_analyzer::vulnerabilities`, and explicitly writes the list back to state.
5. `ReportAgent` requires service information, groups vulnerabilities by sorted tag, sorts each group by severity `critical`, `high`, `medium`, `low` (unknown last), renders service information plus HTML vulnerability tables, saves `oas_vulnerabilities.md`, and yields it as the final event. Missing service info or artifact service is fatal.

The review output schema is:

```text
ServiceBasicInfo {
  name: string,
  description: string,
  summary: string,
  diagram: string,
  criticality: low | medium | high,
  criticality_reason: string
}
```

The saved vulnerability schema is:

```text
EndpointVulnerability {
  tag: string,
  path: string,
  method: string,
  parameters: string[],
  vulnerability: string,
  description: string,
  severity: low | medium | high | critical,
  confidence: low | medium | high
}
```

The report extracts only the content of the first fenced Mermaid block from `diagram`; when none exists, it emits an empty Mermaid block. It always renders service name, description, summary, diagram, criticality, and criticality reason before optional vulnerabilities.

This analyzer does not use TaskRunner's standard planner/worker callback stack.

## 6. Callback middleware model

### 6.1 Callback types and exact signatures

Supported custom callback types are:

| Type | Required positional/keyword parameter names in order |
|---|---|
| `before_model_callback` | `callback_context`, `llm_request` |
| `after_model_callback` | `callback_context`, `llm_response` |
| `before_agent_callback` | `callback_context` |
| `after_agent_callback` | `callback_context` |
| `before_tool_callback` | `tool`, `args`, `tool_context` |
| `after_tool_callback` | `tool`, `args`, `tool_context`, `tool_response` |

Validation compares parameter names and parameter kinds only. Type annotations and return annotations are irrelevant. A mismatched signature is an assembly-time type error.

Each callback exposes:

- callback type;
- dependency names;
- assigned agent name;
- logical name, defaulting to class name;
- serializable `to_state` result;
- callable behavior.

### 6.2 Adapter, dependencies, and chain semantics

The adapter owns a global callback-name registry and one ordered chain per callback type.

Registration algorithm:

1. Reject an already registered logical name.
2. Require every declared dependency name to already exist in the adapter registry, regardless of callback type. There is no automatic topological sort.
3. Assign the adapter's agent name to the callback.
4. Validate its signature.
5. Append it to the relevant chain and registry.

A chain invokes callbacks synchronously in registration order and returns immediately on the first **truthy** result. Null and other false values continue the chain. This truthiness rule is a compatibility detail: an empty map does not short-circuit, while a non-empty tool response or model response does. The chain neither awaits callback results nor catches callback exceptions; a coroutine callback would itself be a truthy returned object, and an exception propagates into the agent engine.

The adapter exports only chains that have at least one callback, keyed by callback type string.

### 6.3 Callback state

Callbacks publish observability state into session state:

```text
callbacks["<agent-name>::<callback-logical-name>"] = callback.to_state()
```

The whole `callbacks` map must be explicitly written back after modification so state-delta tracking sees the change. Dependency reads use the same agent-scoped key. Some token aggregates additionally use global keys described below.

### 6.4 Appending an enforcement callback

To add an after-model enforcement callback to an already built agent:

1. Require that the new callback type is after-model.
2. Build and validate a one-callback chain scoped to the agent name.
3. If it exposes `blocks_forced_tool_choice_none`, register that predicate with every existing before-model callback that exposes `add_force_none_blocker`.
4. If there is no existing after-model callback, install the new chain.
5. Otherwise install a wrapper that invokes the original callback first. Invoke the new chain only when the original result is exactly null; propagate any non-null original result.

This is stricter than ordinary truthy-chain behavior and ensures response-rewriting callbacks complete their rewrite before a mandatory persistence callback sees a later model turn.

## 7. Callback algorithms

### 7.1 Token usage

`TokenUsageCallback` runs after each model response with usage metadata. It tracks input, output, and total tokens.

State surfaces:

- `::<TokenUsageCallback>` — cumulative token counts across all invocations sharing the session state;
- `::<TokenUsageCallback>::history` — map from invocation ID to that invocation's latest cumulative counts;
- agent-scoped callback state — current invocation ID and current invocation counter.

For every response:

1. Null token fields become zero.
2. Add the response counts to the global counter.
3. If this is the first observed invocation, adopt its ID.
4. If invocation ID matches current, accumulate; otherwise replace the current counter with this response's counts and adopt the new ID.
5. Write the current invocation counter to history on every response, overwriting the same key. This guarantees the last invocation is represented even though no later invocation-change event may occur.
6. Save agent-scoped state.

A response without usage metadata is a complete no-op.

### 7.2 Context-limit summarization and forced termination

`SummarizationLimitCallback` runs before model and depends on token usage. It reads a configured counter key, usually `total`, from agent-scoped token callback state.

Before-model occurs before token usage can reset on the first child response. Therefore, if token callback state's invocation ID differs from current invocation ID, treat the count as zero. A null ID on either side is treated as compatible.

At each request:

1. `over_limit = count >= max_tokens`.
2. Candidate forced choice is configured choice when over limit, otherwise null.
3. If candidate is `none`, degrade it to null when either:
   - request tool declarations include `set_model_response`, because that tool is the only finish path; or
   - any registered mandatory-action blocker reports pending work.
4. If force configuration is enabled, publish the resulting task-local tool choice on every call, including null while under limit. Publish forced response format only when effective choice is `none`; otherwise clear it.
5. Save `last_forced` for telemetry.
6. When under limit, save state and return.
7. When over limit, append the configured user-role summary instruction only once per invocation ID. Record a current epoch-second history entry and save state.
8. On later over-limit requests in the same invocation, refresh forcing but do not append the message again.

The task-local values isolate concurrent invocations sharing the default model client. They remain in that asynchronous task's context until refreshed; the before-model callback must therefore clear them on under-limit calls. Disabling force configuration means this callback does not touch the task-local values.

### 7.3 Function-result elision

`FunctionResultsRemovalCallback` runs before model. Construction requires at least one positive limit among `keep_last_n` and `keep_budget_chars`, rejects negative limits, and rejects specifying both target and exempt tool sets.

It scans the full conversation in reverse, considering only function responses eligible under the tool filter and not already marked `elided`.

Before scanning, when deduplication is enabled:

1. Collect function calls in forward order as `(tool name, canonical sorted-JSON args)`.
2. Collect function responses in forward order.
3. Pair the Nth response with the Nth call only if names match.
4. Give an unmatched response a unique sentinel signature so unmatched responses never deduplicate against one another or a real argumentless call.

During reverse scan:

1. If a signature has already been seen, replace response with `{elided: true, tool: <name>, reason: "stale"}`.
2. Otherwise mark the signature seen and measure response as serialized character length.
3. Elide with `{elided: true, tool: <name>}` when retaining it would exceed the positive character budget or positive count budget.
4. Character budget never removes the first eligible retained response, even when that response alone exceeds the budget.

The callback mutates request history in place, increments a cumulative elision counter, saves state, and returns null.

### 7.4 Invalid tool calls and fallback tool

The invalid-tool callback runs after model. Its valid names are every supplied tool name plus reserved `transfer_to_agent`. Construction requires the configured fallback name to be in that set.

For every response function-call part:

- preserve it if its name is valid and args are a map;
- otherwise rewrite its name to fallback and args to `{<fallback-arg>: metadata}`;
- for unknown names, metadata contains original `func_name` and `func_args`;
- for non-map args, metadata contains a malformed-format error;
- record metadata in callback history.

Text parts are preserved. Save callback state whenever content exists. Return null when no part changed so downstream after-model callbacks execute. Return the modified response when anything changed, short-circuiting the current after-model chain.

`default_tool(meta)` returns `{error: "tool <original-name> is not available!"}`. If `meta` is a map it reads `func_name`; otherwise it uses `meta` directly.

### 7.5 Repeated identical tool calls

`RepeatedToolCallCallback` runs before tool and requires threshold greater than 1.

- Calls with empty args pass through, do not advance a streak, and do not break an existing streak.
- Non-empty signature is `<tool-name>::<canonical sorted JSON args>`, falling back to a representation when serialization fails.
- Same signature increments run length; a different signature starts length 1.
- Below threshold, save state and allow execution.
- At threshold and every identical call thereafter, return a non-empty warning response instead of executing the tool.
- Add one history record only at the first threshold crossing.

The standard worker threshold is 5, planner threshold 2, and router threshold 3.

### 7.6 Thinking budget

`ThinkingBudgetGuardrailCallback` runs before model and depends on token usage. Its key must be `input`, `output`, or `total`. It blocks only when token count is strictly greater than its budget, returning a synthetic system-role response with finish reason `MAX_TOKENS`; equality is allowed. It always saves its state.

### 7.7 Per-tool maximum calls

`ToolMaxCallsGuardrailCallback` runs before tool. Its logical name includes the target tool name, allowing one instance per tool. Matching calls increment the count and save state. Calls through `max_calls` execute; call `max_calls + 1` and later return the configured response, default `{result: "Tool call limit reached."}`.

### 7.8 Mandatory tool/verdict enforcement

`MandatoryToolCallback` runs after model and requires a non-empty tool-name set. Its default nudge allowance is 2; verdict-producing factories override it to 3.

State is held for the lifetime of the worker agent instance, not reset by invocation ID:

- `called`: required tool names observed in model-proposed function calls;
- `step_count`: number of model responses containing any function call;
- `nudge_count`: text-only finishes redirected so far.

When a response has function calls, add any required names, increment step count once, save, and return null. It observes proposal, not successful tool execution.

Requirement is satisfied when all configured names were observed, unless `require_any=true`, in which case one intersection is enough. When a text-only response arrives while unsatisfied:

- if nudges remain, return a synthetic user-role message telling the model to call the lexically first missing tool and increment nudge count;
- once `max_nudges` is reached, save state and permit the text finish.

The pending predicate blocks context-limit `tool_choice=none`, preventing termination enforcement from making the mandatory tool unavailable.

Trace verification requires `report_verification`, with three nudges. Exploitability workers configure `{submit_verdict, report_verification}`, three nudges, and `require_any=true` because these are alternate persistence surfaces. Workflow postconditions must still verify the persisted artifact; the callback alone proves only that the model proposed the call.

### 7.9 Request/token rate limits

Both rate-limit callbacks run synchronously before model and use blocking sleep. They are suitable only for single-agent execution because sleeping stalls the event loop and every concurrent agent on it.

Tokens-per-minute:

1. Key must be input/output/total.
2. On first call, record current epoch second and current global token count baseline.
3. Within a window, `diff = cumulative_count - baseline`.
4. Throttle only when `diff > limit`; equality is allowed.
5. Sleep `60 - elapsed + 1` seconds when positive, record history, then start a new window at current cumulative count.
6. If 60 seconds elapsed under budget, roll the window without sleeping.

Requests-per-minute is analogous: first call starts at count 1; each subsequent request increments; throttle only when count exceeds limit; after sleep or a naturally expired window, restart at count 1 for the current request.

## 8. Planner construction and worker protocol

### 8.1 Planning agent assembly

The planning factory receives queued-task ref/name, effective namespace, worker, planner-facing format (TaskRunner uses XML), maximum steps, model, worker-instrumentation flag, output-schema flag (TaskRunner uses false), and observation config.

It builds:

1. Namespace memory tools.
2. Task tools with manager name equal to queued-task ref and `max_tasks=max_steps`.
3. Ordered tool list: `default_tool`, task tools, memory tools.
4. Agent name `task_planner_<safe-ref>`, where unsafe runs become `_`, edges are trimmed, result lowercased, and empty becomes `task`.
5. Callback stack: token usage; invalid-tool fallback; repeated-call threshold 2.
6. Planning prompt with every `<<MAX_SUBTASKS>>` replaced by decimal max steps.

The planner does not receive domain file/HTTP/report tools. It can reach the worker only through `execute_current_subtask`.

### 8.2 Worker instrumentation and schemas

Before wrapping the worker as an agent tool:

1. If input schema is enabled, or output format is JSON, assign `Subtask` as worker input schema.
2. If output schema is enabled, assign `SubtaskExecutionResult` as worker output schema.
3. Snapshot the worker's original instruction on first instrumentation.
4. Set instruction to original plus a generated contract. Repeated instrumentation replaces the generated suffix instead of concatenating it.
5. Wrap as an agent tool.

TaskRunner planning deliberately sets worker output schema **off**. Workers return free text parsed by the task tools. This avoids output-schema/tool incompatibilities while still using a structured input schema.

The formatter supplied to planner task tools controls this generated Subtask handshake. In TaskRunner it is XML, even though the worker factory and the worker's domain tools were configured with the task template's format (normally JSON). Consequently the worker carries two compatible pieces of guidance: the worker factory's context-limit message uses the domain format, while instrumentation's ordinary final-result contract uses XML. Parsing accepts all supported formats, so either path is valid.

The generated worker contract requires:

- fully execute the assigned subtask before stopping;
- copy exact task ID;
- choose only `done` or `incomplete`;
- use `done` only when the deliverable is complete;
- output concrete evidence/results, never a plan or invented facts;
- identify unresolved work and blocker when incomplete;
- return only the structured result in the selected format;
- includes field descriptions and complete done/incomplete examples.

When instrumentation is disabled while structured input is enabled, the underlying worker must already declare an input schema; otherwise assembly fails. A worker already wrapped as an agent tool must wrap an LLM agent, not an arbitrary composite.

### 8.3 Planner prompt policy versus runtime enforcement

The active planner prompt imposes these policies:

- planner performs no domain work itself;
- initial plan uses at most 70% of subtask budget and reserves at least 30%;
- every subtask description ends with exactly `Acceptance: <observable evidence>`;
- one subtask has one outcome;
- prompt-level decomposition depth is one; a failed child should be skipped with a structural blocker rather than decomposed again;
- bootstrap reads memories once, reuses relevant facts, and adds the fewest useful subtasks;
- fresh worker evidence overrides stale memory;
- after a done subtask, execute remaining work, finish if objective met, or add one evidence-backed corrective subtask;
- decompose incomplete/malformed work only to address a named blocker, into 1-3 children;
- approved skip prefixes are `out_of_scope:`, `duplicate:`, `budget_exhausted:`, and `structural_blocker:`;
- `finish` is the final tool call; done requires no open new subtasks and a standalone result matching task output format;
- durable facts and blockers belong in memory; raw logs and execution metadata do not.

These are model instructions unless explicitly enforced by task tools below. In particular, acceptance-line syntax, 70/30 reservation, approved skip prefixes, and one-level decomposition are not validated mechanically. A compatible reconstruction should retain both layers rather than silently upgrading prompt policy into different hard failures.

## 9. Subtask schemas and state machine

### 9.1 Schemas

`SubtaskSpec`:

```text
{ title: required string, description: required string }
```

Descriptions recommend an imperative title under 80 characters and detailed scope, but those length/style rules are prompt metadata rather than validators.

`Subtask`:

```text
{
  task_id: required string matching ^\d+(\.\d+)*$,
  title: non-empty string,
  description: non-empty string,
  status: new | done | incomplete | malformed | skipped | decomposed
}
```

`SubtaskExecutionResult`:

```text
{
  task_id: required string,
  status: done | incomplete,
  output: required string,
  summary: required string
}
```

`SubtaskDecomposition` is `{subtasks: [SubtaskSpec...]}` with one to three children.

### 9.2 Hard transition graph

| From | Allowed destinations |
|---|---|
| `new` | `done`, `incomplete`, `malformed`, `skipped` |
| `incomplete` | `decomposed`, `skipped` |
| `malformed` | `decomposed`, `skipped` |
| `done` | none |
| `decomposed` | none |
| `skipped` | none |

An invalid transition is an execution error describing current, requested, and allowed states. Mutation sessions persist their copied subtask list only on clean exit, so a thrown transition leaves the previous persisted list unchanged.

### 9.3 Planner-state keys

For global queued-task ID `G`, engine planner invocation `I`, and task-tool manager name `M`:

```text
base                       = task::G::I::M
base::tasks                = ordered serialized Subtask list
base::idx                  = current zero-based list index
base::execution-claim      = null or {task_id, claim_id}
task::G::pool              = ordered execution records
```

The planner session state is contractually single-threaded for synchronous map mutations. The execution claim protects same-turn asynchronous duplicate worker calls.

### 9.4 Canonical Subtask formatting

JSON uses the schema maps directly. The string formats are exact in shape:

```text
Markdown Subtask:
### <title> [ID: <task_id>]
**Description**: <description>
**Status**: <status>

Markdown Result:
### RESULT [ID: <task_id>]
**Status**: <status>
**Output**: <output>
**Summary**: <summary>
---
```

YAML wraps a subtask under key `task_<task_id>` and a result under `result_<task_id>`. XML uses:

```xml
<task id="<task_id>">
    <title>...</title>
    <description>...</description>
    <status>...</status>
</task>

<result task_id="<task_id>">
    <status>...</status>
    <output>...</output>
    <summary>...</summary>
</result>
```

Escape XML text/attribute content. XML lists wrap entries in `<subtasks>` or `<results>` with one indentation level. Markdown/YAML lists concatenate item renderings with newlines. JSON lists are arrays of maps. When type hints are requested, wrap only string formats in a fenced block named for the active format; JSON maps/arrays remain native values.

A JSON task record starts with the Subtask map, overlays result fields except duplicate task ID, and adds `usage` when truthy. Other formats concatenate task plus result and append a format-native observation block.

## 10. Task tool surface and algorithms

Task-tool factory options and defaults are:

| Option | Default/behavior |
|---|---|
| `max_tasks` | required planner budget; no range validation |
| `use_skip` | true |
| `use_type_hint` | false |
| `use_input_schema` | true |
| `use_output_schema` | true at the generic factory; planner factory overrides false |
| `use_summarization` | true |
| `worker_instrumentation` | true |
| `max_records` | 20 |
| `n_retries` | 3 total worker calls |
| `observations` | disabled default config |

`max_records` and `n_retries` have no range validation. A non-positive retry budget performs no worker call and falls into malformed-result handling. Compatibility note: a zero record cap behaves as an unbounded slice in the current runtime and therefore returns/summarizes all records, rather than none.

The task tool factory returns this order:

```text
add_subtask
get_current_subtask
list_subtasks
get_records
execute_current_subtask
decompose_subtask
finish
[skip when enabled]
```

Tool responses generally use `{result: ...}` or `{error: ...}`. Domain backend exceptions are normally normalized by tool frontends into error envelopes; task-manager logic returns explicit guidance errors.

### 10.1 Add and inspect

`add_subtask`:

1. Refuse when total stored subtask count is already `max_tasks`.
2. Generate root ID `0` when empty; otherwise one greater than the maximum integer root segment among all existing IDs.
3. Create status `new`.
4. Make it current when current index is absent/invalid or current status is done, skipped, or decomposed. Otherwise append without disturbing current.

`get_current_subtask` returns indexed task or a no-subtasks error. It does not require current status `new`.

`list_subtasks(view="all")` returns complete ordered history. Default `remaining` returns current and all later list entries. It returns an explicit no-remaining-work message when current is absent/invalid, or when the current is the final list item and is no longer new.

`get_records` returns only the most recent `max_records` records, default 20, preserving chronological order.

### 10.2 Skip

The public tool requires non-whitespace reason and exact current task ID. Hard transition rules allow skipping only new, incomplete, or malformed.

Additional tool policy: an incomplete subtask cannot be skipped while it is not the last list entry and total capacity remains; it must be decomposed. It may be skipped when last or when task capacity is exhausted. Malformed may be skipped without that restriction.

On success, change status, append a record whose output is the reason and summary is `Skipped: <reason>`, and advance index when a later item exists. At the end, index remains on the resolved task and response reports no active subtask.

### 10.3 Decompose

The public tool accepts several model-produced input shapes:

- a JSON string is parsed;
- a bare list is wrapped as `{subtasks: list}`;
- a map is schema-validated;
- invalid values return a schema-bearing error rather than throwing.

Require exact current task ID and current status incomplete or malformed. Require `current_total + child_count <= max_tasks`; when some capacity remains, report how many children can fit so the planner can retry smaller.

On success:

1. Transition parent to decomposed.
2. Create children `<parent-id>.1`, `.2`, ... in supplied order, all new.
3. Insert them immediately after parent, ahead of later siblings/root tasks.
4. Set current index to first child.
5. Append a decomposed parent record listing child IDs.

The manager itself permits deeper dotted descendants if a child later becomes incomplete/malformed; the one-level depth limit is planner prompt policy only.

### 10.4 Execution claim

Before the first asynchronous worker operation:

1. Read current subtask and require status new.
2. Choose claim ID from engine function-call ID, falling back to random identity.
3. Synchronously reject if a truthy claim exists.
4. Re-read current and require exact expected ID and new status.
5. Write `{task_id: expected, claim_id}` with no scheduling point.
6. Store claim ownership in task-local execution context so a rejected concurrent duplicate cannot release the winning call's claim.

After the worker returns, completion again requires:

- current subtask ID equals result task ID;
- claim exists and both expected ID and claim ID match;
- current ID still equals expected ID.

Thus a concurrent planner action may skip/advance the plan while a worker awaits, but the stale worker result cannot mutate or append a record to the new current task.

Release changes claim to null only when both IDs match. Release occurs on worker failure/cancellation, normal completion, malformed completion, post-processing exceptions, and the outer execution wrapper's final cleanup. There is no delete operation requirement on the session-state map.

### 10.5 Worker call and response retries

If format is JSON or structured input is enabled, call worker agent tool with the current Subtask map. Otherwise call it with `{request: formatted-subtask}`.

Before worker execution, when observations are enabled, reset per-subtask raw observation accumulators: worker usage, skills read, memories written/read, and file paths.

`n_retries` (default 3) is the total worker-call budget, not additional retries. Retry on:

- null or blank-string response;
- unparseable response;
- parsed result whose task ID does not exactly equal current task ID.

Any thrown value, including cancellation, releases the claim and propagates; worker exceptions are not converted into malformed results by this layer.

Accepted worker response shapes are an already typed result, a map validating directly as the result schema, or a string parsed as follows:

1. Trim; remove case-insensitive `<think>` tags; remove matching outer single/double quotes.
2. Try fenced code blocks with recognized language using that language's parser.
3. Try every fenced body using expected format.
4. Try full text in expected format, then each other supported format.
5. JSON accepts strict JSON, then a bounded safe literal form only when text is at most 50,000 characters and begins `{` or `[`.
6. YAML accepts direct schema map or one-key wrapper whose value is the schema map.
7. Markdown recognizes a `RESULT [ID: ...]` heading and status/output/summary fields, including multiline and bullet forms.
8. XML extracts the first nested `<result task_id="...">` containing status, output, and summary.

For parsed string results only, an empty task ID may be replaced with expected ID. Direct map/typed results retain their ID and therefore fail the mismatch check when empty.

If all calls fail validation, transition current from new to malformed and record:

- task ID;
- status malformed;
- raw last output, or a mismatch explanation;
- standard malformed summary/guidance.

Serialize raw values when possible and cap stored raw text to 20,000 characters plus a truncation marker. Return the record together with an error and action requiring decompose or skip. If plan/claim validation fails during application, surface that stale error and do not mutate the plan.

For a valid result:

1. Transition new to done or incomplete.
2. Advance to next list item only for done and only if one exists. Incomplete remains current.
3. If transition/claim validation succeeded, format and append one execution record.
4. Return a formatted record even when application was rejected, plus optional observations, optional application error, and action guidance. A rejected application does not append that local response record to persistent history. Incomplete always asks for decomposition; a completed last task reports no active subtask.

### 10.6 Finish and summarizer

`finish(status, result)` accepts only done or failed.

Hard validation for done requires:

- at least one subtask exists;
- no subtask has status new;
- at least one subtask has status done.

It does **not** mechanically reject remaining incomplete/malformed tasks, check the acceptance-line policy, or prove the global objective. Failed has no subtask-state prerequisite.

When summarization is enabled, pre-create a dedicated `task_summarizer` agent using the worker's model, fixed summary instruction, and no tools. At finish:

1. Read fixed overall objective.
2. Take most recent `max_records`.
3. Truncate each direct string record, or each direct string field of a map record, at 20,000 characters plus marker. Nested strings are not recursively truncated.
4. Send JSON `{objective, records, result, status}` in a `request` argument.
5. Use raw string response as summary; serialize non-string response to JSON. There is no summarizer retry or validation.

Write fixed overall result, summary, and status state keys. Then set the engine invocation's end flag and return instructions to stop immediately. TaskRunner recognizes only exact status done as a successful attempt.

## 11. Deterministic worker observations

### 11.1 Configuration

Observation configuration defaults disabled and contains:

```text
enabled=false
track_tools=true
tracked_tools=null       // null = all
include_tool_errors=false
track_skills=true
track_files=true
track_file_paths=false
track_coverage_gap=false
track_memories=false
malformed_only=false
in_record=true
in_result=true
```

Resolve from a workflow mapping, then overlay JSON object from `CONTRACTOR_EVAL_OBSERVATIONS`. Reject malformed JSON, non-object overlay, unknown fields, and a `tracked_tools` value other than null or list of strings. Store tracked tools immutably. Emit a JSON-friendly copy of all fields as run/task tags.

Raw capture is cheap and generally always on; this configuration controls projection into model-visible records/results. The execution tool resets raw accumulators only when enabled.

### 11.2 Raw state and projection

Raw keys are:

```text
worker_usage
skills_read
memories_written
memories_read
file_paths
```

The metrics plugin writes tool calls/errors and latest filesystem coverage under worker usage. Skill/memory/file tools append canonical successful operations, deduplicated in first-seen order.

Projection returns null when disabled. Otherwise it may include:

- `tools`: tool-to-call-count, or `{calls, errors}` when errors enabled, optionally allowlisted;
- `files`: filesystem coverage snapshot;
- `files_read_paths`: first 25 read paths plus `... (+N more)`;
- `unvisited_in_scope_paths`: original in-scope paths sorted after comparing normalized forms (strip `./` and leading `/`) against normalized read paths, first 25 plus marker;
- `skills_read`;
- `memories_written` and `memories_read`.

On a valid worker result, include observations only when not `malformed_only` and at least one projected section is non-empty. Attach independently to persisted record and immediate result according to `in_record`/`in_result`.

On malformed result, bypass the non-empty and `malformed_only` gates; lack of deterministic work is diagnostic. A non-null projection is returned immediately when `in_result` is enabled. A literally empty projection map may be omitted from the persisted record by the formatter's truthiness check, while a map containing enabled-but-empty sections is retained.

Records keep observations structurally separate from worker-claimed output: a `usage` map in JSON, `observations` block in YAML/Markdown, or `<observations>` element in XML.

## 12. Skills and injected knowledge

### 12.1 Skill package layout

A skill is a directory under the skills root. Every Markdown file below it is loaded recursively in sorted path order; non-Markdown files are ignored.

```text
skills/<skill>/index.md
skills/<skill>/references/<topic>.md
```

Queue-time validation checks directory existence only. Unknown names fail before any task runs and the error lists all unknown and available directories. File content remains lazy until task setup.

### 12.2 Front matter and memory naming

A Markdown file may begin with YAML front matter delimited by `---`. Parse only when both delimiters exist, YAML is valid, and metadata is a map. Otherwise treat the entire source as body with empty metadata.

Memory names:

- any loaded file whose basename is `index.md` -> `<skill>` and `is_index=true` (skill packages should therefore contain only one such file, normally at the root);
- every other file -> `<skill>/<relative-path-without-.md>`.

Description is front-matter `description` when truthy. Otherwise use `<skill> skill` for index and `<skill> skill / <relative-path-without-extension>` for a reference.

Convert every file to a memory note with exact tags `["skill", <skill>]` and its body excluding valid front matter.

### 12.3 Injection timing and persistence

TaskRunner injects all selected skill notes once per non-restored queued task, after `task_started` and before input-artifact inbox notes and attempts. No selected skills or an empty skill directory is a no-op.

Skill notes are upserted into user-scoped namespace artifact `user:memory/<namespace>`. Existing notes preserve ordinal and creation timestamp and receive a new update timestamp; new notes receive increasing ordinals and current creation/update times. Injection is not repeated across retries.

Skills are not concatenated directly into the worker system prompt. Planner and worker share namespace memory tools, and the worker must discover/read them through dedicated tools.

### 12.4 Reserved memory access

Memory notes tagged `skill` or `inbox` are hidden from generic `list_memories`, `search_memory`, and `read_memory`. Generic writes may not overwrite an existing reserved note unless the write itself carries a reserved tag.

Dedicated skill tools:

- `skills_list`: insertion-ordered previews of all skill-tagged notes;
- `skills_read(name)`: full skill body.

Skill reference lookup tolerates a trailing `.md` and resolves in tiers:

1. exact normalized name;
2. unique slash-suffix match;
3. unique basename match.

Any ambiguous tier returns no match. A successful read records the canonical resolved name once in `skills_read`; a miss returns available names and records nothing.

Inbox notes use analogous `inbox_list`/`inbox_read` tools. This separation prevents domain memories from being polluted by large skill bodies and ensures the observation system can distinguish skill consultation from ordinary memory reads.

## 13. Verdict persistence aliases

Verification tools share a verified-findings store namespace and expose:

- `submit_verdict`: simplified surface accepting name, verdict, summary, entry point, evidence, optional sink/control facts and request IDs;
- `report_verification`: full surface accepting source namespace, attacker control, sink reachability, ordered data flow, broken-path location, impact, notes, and request IDs;
- read/list verification tools.

Verdict values are `exploitable`, `exploitable_unverified`, `not_exploitable`, or `inconclusive`. Attacker control is `full`, `partial`, or `none`. Both write aliases upsert by finding name and persist immediately before returning their formatted record.

Mandatory enforcement treats the two writer names as alternatives only in exploitability agents. Static trace verifier requires the full report alias specifically.

## 14. Minimum conformance scenarios

A reconstruction should test at least:

1. Callback dependency order, duplicate names, signature rejection, and truthy short-circuiting.
2. Invalid calls are rewritten only when needed, allowing downstream enforcement on untouched responses.
3. Context-limit message fires once per invocation and forced values do not leak between concurrent asynchronous tasks.
4. Forced `none` degrades while a finish tool or mandatory verdict is pending.
5. Mandatory `require_any` accepts either verdict alias and nudges text-only finish when neither was proposed.
6. Token history includes the final invocation without requiring an invocation change.
7. Result elision deduplicates stale identical calls, respects count/character limits, and always retains at least the newest eligible result.
8. Repeated-call protection ignores argumentless tools and blocks the threshold-th identical non-empty call.
9. Parallel duplicate `execute_current_subtask` calls run the worker once; a result made stale by plan advancement is discarded.
10. Worker output retries empty, unparseable, and mismatched-ID responses, then records malformed raw output after exhaustion.
11. Every legal/illegal subtask transition, decomposition insertion order, and root/child ID generation is covered.
12. Done finish is rejected with no tasks, any new task, or no done task; failed finish remains available.
13. Observation projections honor all gates/caps and preserve empty evidence on malformed output.
14. Skill loading, front-matter fallback, name aliases, reserved-note hiding, and canonical successful-read tracking are covered.
