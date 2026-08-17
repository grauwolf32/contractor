# Implementation inventory

## 1. Purpose

This file is the completeness manifest for a reconstruction. It lists every
public workflow, versioned agent/task/skill asset, callable tool family, runtime
subsystem, event identifier, and operator-facing component present in the
specified product. Behavioral details live in files 03–10.

Names and active versions below are current normative defaults. Historical
versions remain loadable because evaluations and environment overrides may pin
them.

## 2. Public workflows

The workflow registry contains exactly these keys:

| Public key | Assembler identity | Primary purpose |
|---|---|---|
| `oas_build` | `OasBuildingWorkflow` | Discover project/dependencies, build, and validate OpenAPI. |
| `oas_update` | `OasEnrichmentWorkflow` | Seed and enrich/update an existing OpenAPI schema. |
| `exploit` | `ExploitabilityWorkflow` | Assess persisted vulnerability findings against an authorized live target. |
| `likec4` | `LikeC4BuildingWorkflow` | Build and validate a security-focused LikeC4 architecture model. |
| `trace` | `TraceAnnotationWorkflow` | Planner-driven per-operation trace annotation over an overlay. |
| `trace-direct` | `TraceAnnotationDirectWorkflow` | Direct trace-agent per-operation variant. |
| `trace-graph` | `TraceGraphWorkflow` | Call-graph-assisted trace grouping/annotation. |
| `trace-graph-pathpar` | `TraceGraphPathParWorkflow` | Path-group-parallel graph trace with overlay forks/merge. |
| `trace-postdiff` | `TracePostDiffWorkflow` | Analyze an existing trace overlay/diff for vulnerabilities. |
| `trace-verify` | `TraceVerifyWorkflow` | Statically verify trace-derived findings. |
| `vuln-assess` | `VulnAssessWorkflow` | Composite OpenAPI build/lint, parallel trace discovery, and optional live assessment. |
| `vuln-scan` | `VulnScanWorkflow` | Broad planner-driven source vulnerability scan. |
| `vuln-scan-fast` | `VulnScanFastWorkflow` | Faster staged/partitioned vulnerability scan. |
| `vuln-scan-trace` | `VulnScanTraceWorkflow` | Compose trace-based discovery with vulnerability analysis. |
| `vuln-sweep` | `VulnSweepWorkflow` | Concurrent vulnerability-class nomination and scanning sweep. |
| `router` | `RouterWorkflow` | Route a free-form prompt to specialist agents. |

Registry matching is case-insensitive at the CLI choice boundary but the keys
above are the canonical persisted/displayed spellings.

## 3. Versioned agent assets

Each row corresponds to one prompt manifest and one factory. `Active` is used
unless an allowed override pins another value.

| Asset | Active | Loadable versions | Role |
|---|---|---|---|
| `codereview_agent` | `v3` | `v1`, `v2`, `v3` | Source vulnerability/code review. |
| `exploitability_agent` | `shannon` | `v1`–`v7`, `shannon` | Finding-specific source/live exploitability assessment with proof discipline. |
| `http_agent` | `v1` | `v1` | General stateful HTTP probing. |
| `librarian_agent` | `v1` | `v1` | Artifact-pool discovery and consolidation. |
| `likec4_builder_agent` | `v3` | `v1`, `v2`, `v3` | Security architecture model creation/validation. |
| `oas_builder_agent` | `v4` | `v1`, `v2`, `v3`, `v4` | Incremental OpenAPI construction and enrichment. |
| `oas_linter_agent` | `v1` | `v1` | OpenAPI lint interpretation and verified repair. |
| `planning_agent` | `v5` | `v1`–`v5`, `pentestgpt` | Strict-state task decomposition/delegation. |
| `router_agent` | `v2` | `v1`, `v2` | Prompt routing to specialist agent-tools. |
| `swe_agent` | `v2` | `v1`, `v2` | Read-oriented software/source investigation. |
| `swe_edit_agent` | `v2` | `v1`, `v2` | Overlay-backed source editing. |
| `threat_model_agent` | `v1` | `v1` | STRIDE/security threat analysis using code and OpenAPI. |
| `trace_agent` | `converge` | `v0`–`v7`, `shannon`, `converge` | Direct request-path trace, annotations, controls, and findings. |
| `trace_verifier_agent` | `v1` | `v1`, `shannon` | Static code-backed finding verification. |
| `triage_agent` | `v1` | `v1` | Finding/analysis triage. |
| `vuln_analytics_agent` | `v1` | `v1` | Convert trace diffs/annotations into vulnerability conclusions. |
| `web_exploitability_agent` | `v4` | `v1`, `v2`, `v3`, `v4` | Web exploitation using HTTP/proxy and optional code execution. |

The `web_exploitability_agent` factory also exposes a reduced/lite variant that
uses the same versioned asset with a smaller tool/configuration surface.

### 3.1 Legacy analyzer

`oas_analyzer` is an unversioned legacy multi-agent evaluator composed of
analytic sub-agents plus a `report_generator`. It remains present for its
standalone evaluation harness but is not a public workflow dependency and does
not participate in the prompt-manifest explorer contract. A compatible rebuild
either retains this evaluation-only component or explicitly migrates its eval
to a versioned agent; it MUST NOT invent an active manifest entry for it.

## 4. Versioned task assets

| Task asset | Active | Loadable versions | Declared skill in active version | Purpose |
|---|---|---|---|---|
| `dependency_information` | `v1` | `v1` | — | Inventory runtime-relevant dependencies. |
| `exploitability_assessment` | `v4` | `v1`–`v4` | — | Probe and persist a verdict for one finding. |
| `knowledge_consolidation` | `v1` | `v1` | — | Consolidate discovered artifact knowledge. |
| `knowledge_discovery` | `v1` | `v1` | — | Search artifact-pool knowledge. |
| `likec4_build` | `v1` | `v1` | `likec4` | Build `/architecture.c4` in an overlay. |
| `likec4_validate` | `v2` | `v1`, `v2` | `likec4` | Validate and repair the architecture model. |
| `oas_enrich` | `v2` | `v1`, `v2` | — | Enrich schema using project code and existing artifacts. |
| `oas_update` | `v2` | `v1`, `v2` | — | Incrementally update paths/components with provenance. |
| `oas_validate` | `v1` | `v1` | — | Lint, verify, and repair serious OpenAPI issues. |
| `project_information` | `v1` | `v1` | — | Detailed source-project map. |
| `project_information_short` | `v1` | `v1` | — | Bounded shallow project map. |
| `sink_nomination` | `v1` | `v1` | `vuln_scan` | Nominate likely sink/vulnerability focus areas. |
| `threat_analysis` | `v1` | `v1` | `stride` | Produce a STRIDE-oriented threat analysis. |
| `trace_annotation` | `v3` | `v1`, `v2`, `v3`, `shannon` | injected by workflow/worker (`trace`) | Trace one operation and annotate an overlay. |
| `trace_verify` | `v1` | `v1` | — | Verify one trace-derived finding statically. |
| `vuln_analytics` | `v1` | `v1` | — | Analyze annotated flows and emit findings. |
| `vuln_scan` | `v3` | `v1`, `v2`, `v3` | `vuln_scan` | Broad pattern and missing-control scan. |
| `vuln_scan_fast` | `v1` | `v1` | `vuln_scan` | Focused fast scan stage. |

The trace workflow supplies its domain skill at worker construction even though
the active lean task body describes rather than declares that injection. That
wiring is part of file 04/05, not a manifest parser default.

## 5. Skill assets

Every skill has an `index.md`; reference names are stable lookup IDs without the
`.md` suffix.

| Skill | References |
|---|---|
| `auth` | none |
| `caido` | none |
| `code-exec` | none |
| `exploit` | `auth-bypass`, `auth-discovery`, `broken-auth`, `cmdi`, `idor`, `info-disclosure`, `mass-assignment`, `nosqli`, `path-traversal`, `rate-limiting`, `sqli`, `ssrf`, `ssti`, `xss`, `xxe` |
| `likec4` | `cli`, `configuration`, `deployment`, `dynamic-views`, `examples`, `identifier-validity`, `include-predicates-wildcards`, `model`, `predicates`, `relationships-bidirectional`, `specification`, `style-tokens-colors`, `troubleshooting`, `views` |
| `stride` | none |
| `trace` | `annotations`, `controls`, `cwe-mapping`, `finding-shapes`, `frameworks`, `sinks`, `sources` |
| `vuln_scan` | `absence-detection`, `business-logic`, `checklist`, `grep-patterns`, `miss-patterns`, `php-wordpress`, `secrets`, `sink-patterns` |
| `vulns` | `idor`, `ssrf`, `ssti`, `xxe` |

Skill loading, reference confinement, injection, and read metrics are specified
in file 04.

## 6. Public tool-call inventory

Factories expose only the tools selected by an agent. The complete built-in
callable inventory is grouped below; internal helper methods are not model
tools.

Every standard agent also exposes `default_tool(meta)`. Models are instructed
not to call it directly; the invalid-call callback rewrites an unknown tool or
malformed argument object to this fallback, which returns
`{"error":"tool <reported-name-or-null> is not available!"}` and includes the
failed-call metadata in `meta`.

### 6.1 Read and write filesystem

Read surface:

```text
ls, glob, read_file, grep,
interaction_stats, list_touched_files, list_untouched_files,
list_match_only_files, reset_interaction_tracking
```

Write-capable surface adds:

```text
write_file, append_file, mkdir, rm, cp, mv,
insert_line, replace_range, edit, restore, changed_paths, diff
```

### 6.2 Source and call graph

```text
search_def, list_symbols,
annotate_trace, annotate_validate, annotate_sink,
graph_summary, find_symbol, find_callers, find_callees,
paths_between, entrypoint_paths_to, attack_surface,
complexity_hotspots, functions_that_raise
```

Graph tools attach only when the filesystem has a resolvable confined local
root and graph construction is available.

### 6.3 Memory, skills, and inbox

```text
write_memory, append_memory, link_memories, read_memory, search_memory,
list_tags, list_memories,
skills_list, skills_read,
inbox_list, inbox_read
```

### 6.4 Planner/subtasks

```text
add_subtask, get_current_subtask, list_subtasks, get_records,
decompose_subtask, skip, execute_current_subtask, finish
```

The summarizer is an agent-as-tool internal to the planner toolset, not a
separately registered public workflow.

### 6.5 Artifact pool and dense retrieval

```text
pool_namespaces, pool_list, pool_read, pool_read_memory, pool_search
```

`pool_search` selects pgvector embeddings when configured and keyword ranking
otherwise.

### 6.6 OpenAPI and LikeC4

```text
upsert_path, remove_path, list_paths, get_path,
upsert_component, remove_component, list_components, get_component,
set_info, get_info, add_server, remove_server, list_servers,
get_full_openapi_schema, lint_openapi, validate_likec4
```

### 6.7 Vulnerability reports and verifications

```text
report_vulnerability, get_vulnerability, list_vulnerabilities,
report_verification, submit_verdict, get_verification, list_verifications
```

The two verdict-writing names share the verification persistence domain but
serve factories/prompts with different terminology.

### 6.8 Stateful HTTP

```text
http_request, http_read_body, http_history,
http_session_set, http_session_get, http_session_clear
```

### 6.9 Caido

```text
caido_scope, caido_history, caido_request_detail, caido_replay,
caido_automate_run, caido_automate_results, caido_sitemap,
caido_workflow_list, caido_workflow_run, caido_workflow_findings
```

### 6.10 Sandbox execution

```text
run_python, execute_bash
```

All tools return the standard envelope in file 06 unless a documented runtime
adapter requires a framework-native wrapper around the same logical result.

## 7. Runtime and callback components

| Component | Required responsibility |
|---|---|
| `TaskRunner` | Sequential queue, attempts/iterations, planner/worker assembly, artifact publication, events, checkpointing. |
| `AgentRunner` | Direct specialist session/stream lifecycle and events. |
| artifact helpers | Safe invocation key construction and result/summary/records publication/loading. |
| skill loader | Index/reference validation, template injection, metrics. |
| Agio adapter | Shared event taxonomy and flat envelopes. |
| callback adapter/base | Normalize framework callback signatures and execute callback chains safely. |
| context callback | Compact/elide old messages and large function results. |
| guardrails | Enforce required tool calls, terminal actions, valid verdicts, and output structure. |
| rate-limit callback | Enforce call/time budgets. |
| token callback | Track usage, request summarization, and enforce context ceilings. |
| metrics plugin | Correlate model/tool usage, timing, exceptions, and filesystem coverage. |
| trace plugin | Persist detailed framework events/tool snapshots for analysis. |
| sandbox cleanup plugin | Tear down invocation-owned execution containers. |

## 8. Event identifiers

The shared taxonomy consists of:

```text
agent_initialized
agent_run_start, agent_run_end
agent_ttft, agent_tps
agent_loop_start, agent_loop_end
tool_call, tool_result
user_feedback
tool_exception, llm_usage, fs_coverage, run_summary, callback_summary
adk_tool_call, adk_tool_result, adk_tool_error, adk_event
workflow_started, workflow_finished
run_started, run_finished
agent_run_started, agent_run_finished
task_started, task_finished, task_failed, task_skipped
global_task_finished
iteration_started, iteration_finished, iteration_result
final_text
```

`agent_run_start/end` are framework/plugin events; the similarly named
`agent_run_started/finished` delimit the direct `AgentRunner`. They MUST remain
distinct.

The canonical event base carries type, epoch-millisecond timestamp, UTC ISO
time, session ID, invocation ID, run ID, task name/ID, iteration, and agent name;
event-specific fields are additive.

## 9. Operator-facing components

| Component | Inventory |
|---|---|
| Workflow CLI | validation/context driver, rooted local FS adapter, JSONL metrics sink, plain renderer, live renderer, artifact export utilities |
| Explorer server | threaded static/API server, metadata reader, workflow registry/graph extractor, tool introspector, eval reader, comments store |
| Browser client | build-free hash-routed SPA, source/diff/comment views, workflow DAG, evaluation charts, vendored chart library |
| Runtime deployment | multi-stage non-root application image with LikeC4 CLI |
| Model deployment | LiteLLM-compatible proxy config/launcher and llama.cpp local serving helper |
| Optional data service | pgvector compose definition, vector initialization, lifecycle/DSN helper |
| Sandbox deployment | Kali-derived tool image and Podman build helper |

## 10. Evaluation and maintenance utilities

The supported utility categories are:

- run agent, task, trace-pipeline, vulnerability, exploitability, threat, and
  librarian evaluations;
- rebuild/migrate standardized evaluation envelopes;
- score trace vulnerabilities and threat reports;
- compare two or more agent/task/exploit variants;
- sweep prompt/task configuration axes;
- analyze metrics, vulnerability results, and observability traces;
- prepare external vulnerability benchmarks; and
- visualize planner subtasks.

Utilities may evolve faster than public workflows. Their stable boundary is
the artifacts and `eval/v1` schema they consume/produce, not each script's
command-line spelling.

## 11. Inventory audit procedure

At release time, automatically compare this inventory with the implementation:

1. enumerate registry keys;
2. enumerate prompt/task manifests, active values, and mapped files;
3. enumerate skill indexes and confined references;
4. statically enumerate model-exposed function names from every factory;
5. enumerate the event-type definition; and
6. enumerate explorer routes.

Fail the audit for an undocumented addition, a missing documented item, an
active version whose file is absent, duplicate public tool names within one
factory, or a reference that escapes its skill directory.
