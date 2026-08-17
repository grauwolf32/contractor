# Testing and acceptance

## 1. Verification strategy

Contractor is accepted at four distinct levels:

1. **Static quality gates** check syntax, imports, type contracts, and common
   defect patterns without contacting a model.
2. **Unit tests** exercise deterministic subsystem contracts with in-memory or
   temporary services.
3. **Integration tests** verify real composition boundaries such as the task
   runner plus artifact/session services and optional external adapters.
4. **Evaluations** measure nondeterministic model behavior against versioned
   source fixtures and emit a stable analytics envelope.

No model-dependent score is a substitute for deterministic safety tests. A
reimplementation MUST first pass path confinement, persistence, state-machine,
and record-schema tests; evaluations then judge the quality of the behavior
inside those boundaries.

The normal test command MUST NOT contact a model or live target. Evaluation
tests are opt-in and distinctly marked.

## 2. Deterministic test suites

### 2.1 Runtime and orchestration

Tests MUST cover:

- task queue ordering, unique refs/IDs, template rendering, artifact inbox
  injection, skill loading, and standard publication;
- planner/subtask state transitions, duplicate tool claims, invalid terminal
  transitions, structured result validation, and required terminal calls;
- retries, per-attempt timeout, iteration accumulation, and correct failure
  events after exhaustion;
- event ordering and best-effort event-handler behavior, including propagation
  of cancellation;
- plugin/callback ordering, context compaction, token and rate limits, heavy
  tool-result retention, and metrics folding;
- checkpoint ownership, template-version compatibility, artifact completeness,
  atomic replacement, process/thread merging, and dirty-state behavior after a
  failed save;
- direct-agent sessions and cleanup behavior; and
- workflow lifecycle events, seed persistence, fan-out isolation, and mandatory
  verification postconditions.

### 2.2 Storage, filesystems, and tools

Tests MUST exercise the contracts in files 06 and 07, including:

- logical artifact-key traversal rejection, portable path-derived keys, and
  case-variant collision behavior;
- local-root confinement for absolute/relative paths and symlink escapes;
- bounded list, read, glob, grep, and code-walk output with visible truncation;
- overlay create/modify/delete/type-change patches, recursive move guards,
  fork/merge conflicts, and deterministic unified diffs;
- function/symbol search and graph result/depth ceilings;
- standard tool success/error envelopes and schema-reference compatibility;
- OpenAPI parse/normalize/validate/write behavior;
- vulnerability report and verification deduplication, ambiguous duplicate
  rejection, and source-namespace ownership;
- HTTP request-ID lifecycle across success, retry, cancellation, and failed
  final persistence; body indirection; history limits; and shared-namespace
  locking;
- proxy, RAG, and code-execution unavailable/fallback behavior; and
- sandbox timeout, output bound, read-only source, artifact capture, and cleanup.

### 2.3 CLI and explorer

Tests MUST cover:

- project/folder canonicalization, default output, per-project store identity,
  UTF-8 seed rejection, reset/resume exclusion, and prompt requirements;
- text/binary artifact export and exclusion of internal memory;
- flat JSONL metrics, sensitive-field redaction, renderer state, and the rule
  that only the workflow terminal event stops the live UI;
- every explorer route and JSON error path;
- percent-encoded, doubly encoded, separator, traversal, and symlink attacks on
  resource and static paths;
- static workflow/tool introspection without runtime imports;
- evaluation discovery limited to exact discovered IDs;
- comment CRUD/validation/order/timestamps; and
- browser escaping and safe Markdown-link schemes for all dynamic values.

### 2.4 Workflow assembly tests

Every public workflow key in file 05 MUST have an assembly test that constructs
it with fake services and asserts:

- its stages are in the specified order;
- each queued task uses the correct template and artifact dependencies;
- nested workflows receive the intended context/seed/namespace;
- configured parallelism and failure-isolation flags are wired correctly;
- final artifact existence and verification gates are enforced; and
- optional configuration changes the intended budget without changing unrelated
  defaults.

Assembly tests should inspect the orchestration structure, not duplicate the
agent-quality evaluation.

## 3. Evaluation opt-in and prerequisites

Evaluation tests are marked `eval` and run only when either:

- the test marker expression explicitly mentions `eval`; or
- `CONTRACTOR_RUN_EVAL` is one of `1`, `true`, `yes`, or `on`, ignoring case
  and surrounding whitespace.

An unrelated marker expression does not opt in. `not eval` is still handled by
the test runner's own deselection. Without either opt-in, collected eval items
are skipped with an explanation.

An evaluation environment requires:

1. a reachable model gateway whose aliases match configuration;
2. the source-fixture submodule initialized under `tests/playground`;
3. the selected model alias, defaulting to the production default and
   overridable with `CONTRACTOR_EVAL_MODEL`; and
4. writable `eval_runs` and artifact directories.

Evaluation harness requests allow up to 600 seconds when they construct their
own model client. Slow behavior should be addressed by a deliberate timeout
change, not an unnoticed model substitution.

## 4. Evaluation scenarios and isolation

Exactly three scenario identifiers are valid:

| Scenario | One attempt executes |
|---|---|
| `agent` | One specialist model agent directly. |
| `task` | One planner-driven task, including planner and worker chain. |
| `pipeline` | One complete workflow/CLI-equivalent pipeline. |

Each case runs `pass_at = X` attempts. It passes if at least one attempt
passes. The representative metrics/detail are taken from the first passing
attempt, or from the first attempt if none pass. `pass_count` records all
passing attempts. A per-attempt `runs` array is present only when `X > 1`.

One harness call is one pass. Retries and iterations within that pass share its
artifact tree so planner memory remains useful. Separate pass@X attempts MUST
use separate artifact roots and sessions; otherwise a later attempt could read
the prior attempt's `user:memory/...` state and invalidate the measurement.
Task harnesses use `<artifact-root>/pass-<attempt-id>` for this purpose.

## 5. Fixture contract

Each fixture directory contains `meta.yaml` with at least:

```yaml
slug: stable-id
source_root: tests/playground/<relative-project>
```

Optional descriptive fields include language, framework, benchmark, and
description; they aid selection but do not change the source root. The root is
resolved relative to the repository and MUST exist before the fixture runs.

Ground truth is loaded lazily from the files required by a test:

| File | Purpose |
|---|---|
| `vuln-cases.json` | Per-vulnerability source truth: ID, vulnerable flag, CWE, file/function, and related facts. |
| `vulnerabilities.expected.json` | Expected operation-level vulnerability class/method/path/severity. |
| `trace-cases.json` | Entrypoint plus expected annotated file/function locations. |
| `oas.expected.yaml` | Expected OpenAPI structure. |
| `exploitability-cases.json` | Expected exploitability verdicts and evidence conditions. |
| `swe-cases.json`, `planner-cases.json`, `task-cases.json` | Scenario-specific inputs and expected outputs. |

Case fixtures are automatically parameterized as independent test items named
`<slug>/<case-id>`. This provides independent timeouts, filtering, parallelism,
and CI visibility. Scoring logic stays in the test/scorer; harnesses standardize
execution, isolation, event metrics, artifact capture, and serialization.

## 6. `eval/v1` result schema

Every evaluation producer MUST write UTF-8 JSON to
`<run-directory>/eval_results.json` with this envelope:

```json
{
  "schema": "eval/v1",
  "scenario": "agent|task|pipeline",
  "unit": "stable unit name",
  "metric_kind": "detection|verdict|capture|diff|generic",
  "pass_at": 1,
  "model": "model alias or null",
  "prompt_version": "version or null",
  "timestamp": "UTC ISO-8601",
  "meta": {},
  "fixtures": [],
  "headline": {},
  "totals": {}
}
```

### 6.1 Fixture and case records

A fixture record is:

```json
{
  "slug": "fixture-id",
  "cases_total": 1,
  "cases_passed": 1,
  "cases": [],
  "tokens": 123,
  "latency_ms": 456.7
}
```

`tokens` and `latency_ms` are optional and are omitted when not measured. A
case is:

```json
{
  "id": "case-id",
  "passed": true,
  "pass_count": 1,
  "attempts": 1,
  "metrics": {},
  "detail": {}
}
```

For repeated cases, `runs` contains each attempt's `{passed, metrics, detail}`.
The redundant counts and derived fields make a result self-describing and MUST
agree with the contained arrays.

### 6.2 Metric kinds

| Kind | Required/standard `detail` data | Derived headline |
|---|---|---|
| `detection` | `tp`, `fp`, `fn`, optional `tn`, precision/recall/F1, per-CWE data, findings, matches | Micro precision, recall, F1 from summed TP/FP/FN. |
| `verdict` | expected verdict, actual verdict, `has_evidence` | Fraction of cases with evidence. |
| `capture` | `captured`, exploit chain, tags | Fraction with a non-empty/true chain. |
| `diff` | precision, recall, F1, matched, missing, extra | Arithmetic mean of case F1 values. |
| `generic` | Domain-specific data | No extra domain scalar. |

Every headline also contains `pass_rate`, `passed`, and `total`. Rates and
precision/recall/F1 values are rounded to three decimal places. Zero-denominator
values are zero.

### 6.3 Totals

Totals are derived from each case's representative metrics, not by summing all
pass@X attempts. The record contains:

```text
fixtures, cases,
input_tokens, output_tokens, total_tokens,
total_tool_calls, tool_errors, llm_calls, http_requests, skill_reads,
duration_s, tool_counts
```

`total_tokens` falls back to input plus output tokens when explicit totals sum
to zero. `duration_s` is rounded to one decimal. `skill_reads` is derived from
the `skills_read` tool count. When at least one fixture reports cost fields,
the totals also include summed `tokens` and/or `latency_ms`.

Event-based metric folding counts `tool_call`, `tool_result`,
`tool_exception`, and `llm_usage` records. Task-based folding sums each task's
token, call, error, timing, and tool-count metrics.

## 7. Result persistence and archives

The default results root is `eval_runs/`. A per-process UTC run stamp is
`MMDD-HHMMSS`; `CONTRACTOR_EVAL_RUN_STAMP` may override it after replacing
characters outside alphanumeric, `-`, and `_` with `_`.

The session sink groups cases by `(scenario, unit, metric_kind)`. This complete
key is mandatory because different metric kinds have incompatible detail
shapes. The first non-null model and prompt version seed the bucket; later
conflicts are warned and the first value wins. The bucket's `pass_at` is the
maximum recorded value.

Persistence has two forms:

1. A stable latest result at
   `eval_runs/<scenario>-<safe-unit>[-<metric-kind>]/eval_results.json`, where
   the suffix is omitted only for `generic`. It may be replaced by the next
   run of the same bucket.
2. A never-overwritten per-run/per-fixture archive at
   `eval_runs/<stamp>/<scenario>-<safe-unit>-eval-<fixture>/`.

Each case is persisted immediately, before session flush, under
`cases/<safe-case>/metrics.json`, with optional named text artifacts beside it.
The harness's live artifact tree is `cases/<safe-case>/artifacts`. This makes a
timeout or process crash diagnosable even if the aggregate envelope is never
flushed.

## 8. Domain scoring invariants

Vulnerability detection scoring MUST use a general application-security
taxonomy rather than fixture-specific sink names. Trace-task scoring combines
structured vulnerability-report artifacts with recognized result blocks,
normalizes them to general families (for example SQL injection, SSRF, IDOR,
CSRF, path traversal, sensitive-data exposure, auth/crypto, rate-limit abuse,
and business logic), attributes them to operations, then performs greedy
one-to-one matching by family and path with exact path preferred.

Expected benchmark details may inform ground truth but MUST NOT be embedded as
special cases in the detector or scorer. A scoring change requires tests proving
that unmatched expected items become false negatives and unmatched reported
items become false positives.

## 9. Evaluation inventory

The maintained evaluation suite covers at least these units:

| Scenario | Units | Primary kind |
|---|---|---|
| agent | trace, code review/vulnerability scan, exploitability, OpenAPI builder/analyzer, SWE | diff, detection, verdict, generic |
| task | OpenAPI build/enrich, project information, LikeC4, exploitability assessment, threat analysis, planner, web exploit | diff, verdict, capture |
| pipeline | trace graph/parallel variants | diff |

Standalone drivers may run vulnerability scans, trace tasks, exploitability,
A/B comparisons, Cartesian prompt/task sweeps, metric analysis, and external
benchmark import. They MUST emit the same `eval/v1` envelope if their results
are to be discoverable by the explorer.

## 10. Reconstruction acceptance matrix

Before declaring a reconstructed implementation complete, demonstrate:

| Capability | Minimum acceptance evidence |
|---|---|
| Registry and CLI | Every key constructs; option/path/error/export tests pass. |
| Runtime | State-machine, retry, callback, event, checkpoint, and concurrency tests pass under forced failures. |
| Agent/task assets | Every manifest resolves its active and pinned versions; all declared skills and artifacts load. |
| Source confinement | Traversal and symlink escape suites pass for local, overlay, explorer, and any remote adapter. |
| Persistence | Text/binary artifacts, records, HTTP state, checkpoints, and comments survive process restart. |
| Workflows | Assembly tests match file 05 and smoke runs with deterministic fake agents produce the specified artifact graph. |
| Optional services | Each has both an available integration test and an unavailable/disabled behavior test. |
| Security UI | Malicious identifier/Markdown/dynamic-value corpus produces no scriptable markup. |
| Evaluation | A small fixture per applicable scenario writes a valid, discoverable `eval/v1` envelope with isolated attempts. |
| Quality | Static checks and the complete non-eval test suite pass with no reliance on test order. |

Exact test counts are intentionally not normative because the suite grows. The
normative condition is coverage of every contract and failure boundary above.
