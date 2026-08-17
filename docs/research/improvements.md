# Project review — bugs and improvements

Full-codebase review (2026-06-10, branch `eval/gt-completeness`). Five parallel review passes
covered: runners, agents + subtask state machine, tools + sandbox, workflows + CLI + settings,
callbacks + utils + scripts + eval infra. Every finding below was verified against the actual
code path (the sandbox findings were confirmed with throwaway repro scripts).

Severity: **high** = wrong behavior in a production path right now; **medium** = real bug,
latent or recoverable, or silently degrades results; **low** = footgun, inconsistency, or
robustness gap.

---

## 1. High-impact bugs

> **Status: all six fixed (2026-06-10)** — `777c636` (§1.1), `e1841b2` (§1.3),
> `3e4f0c9` (§1.4, §1.5, §1.6 runner side + the `artifacts=[]` footgun from §3.1),
> `fcf7e96` (§1.2, §1.6 workflow side). Note from the fix work: the
> `trace_annotation`/`trace_graph*` workflows also fan out per OpenAPI path under a shared
> `trace_annotation` publish key — same latent clobbering as §1.6, can adopt `artifact_key`
> the same way (their refs already carry `path_key`).

### 1.1 Exploitability verdict enforcement is completely inert (guardrail short-circuits the chain)

**Severity:** high
**Where:** `contractor/callbacks/guardrails.py:173-222`, `contractor/callbacks/adapter.py:30-35`,
`contractor/agents/exploitability_agent/agent.py:162-175`, `contractor/agents/web_exploitability_agent/agent.py:143+`

`InvalidToolCallGuardrailCallback.__call__` returns `llm_response` unconditionally whenever
`content.parts` is non-empty — even when it modified nothing. `CallbackChain.__call__` stops at
the first truthy return, and this guardrail is registered **last** in the worker chain
(`worker_factory.py:141-147`). The exploitability agents append `MandatoryToolCallback` *after*
that chain via `_chain`, guarded by `if result is not None` — which is always not-None. So the
mandatory-verdict nudge never fires and its `called`/`step_count` tracking never runs. The
verdict-tool enforcement these agents were built around is dead code.

**Fix:** track a `modified` flag in `InvalidToolCallGuardrailCallback` and return `None` when no
part was rewritten (state can still be saved); or invert the exploitability chaining so
`MandatoryToolCallback` runs first.

**Related gap:** no unit tests exist for `InvalidToolCallGuardrailCallback`,
`MandatoryToolCallback`, `ThinkingBudgetGuardrailCallback`, or `ToolMaxCallsGuardrailCallback`
(only `RepeatedToolCallCallback` is covered in `test_guardrails.py`) — exactly the gap that let
this ship. Add chain-interaction tests when fixing.

### 1.2 `trace-verify` after `trace-graph` silently verifies nothing

**Severity:** high
**Where:** `contractor/workflows/trace_verify/workflow.py:81`

Trace-verify reads findings only under the `trace-annotation:{ns}:{path_key}` memory prefix,
which pairs with `trace` and `trace-direct`. But `trace-graph` writes under `trace-graph:...`
(`trace_graph/workflow.py:131`) and pathpar under `trace-graph-pathpar:...`
(`trace_graph_pathpar/workflow.py:51`). Since v7+graph is the production default, the standard
pipeline `trace-graph → trace-verify` finds zero findings on every path, logs it only at DEBUG,
and skips everything — the run "succeeds" having verified nothing.

**Fix:** probe all known namespace prefixes (share them as constants, the way
`PATH_NAMESPACE_PREFIX` already does for pathpar), and fail loudly / warn when zero findings
exist across all paths.

### 1.3 `MemoryOverlayFileSystem.glob` has broken `**` and multi-level semantics

**Severity:** high
**Where:** `contractor/tools/fs/overlayfs.py:1232-1264`

Confirmed by repro:

- `ov.glob("/**/*.py")` omits root-level files (`/top.py` missing); `RootedLocalFileSystem`
  returns all matches.
- `ov.glob("/src/**/*.py")` returns `[]` even when `/src/deep/deeper/mod.py` exists.
- `ov.glob("a/*/b.py")` returns `[]` — non-`**` patterns deeper than one wildcard level fail,
  because the non-`**` branch only `ls`-es the static prefix one level.

Root cause: the overlay relies on `PurePosixPath.match`, which on Python 3.12 (project floor)
does **not** treat `**` as a recursive glob and matches right-anchored with a fixed segment
count. This is reachable in production: trace/likec4/vuln-scan workflows wrap the overlay, and
`rw_file_tools` delegates `glob → FsspecInteractionFileTools.glob → self.fs.glob`, so agents
editing in the overlay get wrong or empty glob results.

**Fix:** reuse the path-aware `_glob_to_regex` matcher from `RootedLocalFileSystem.glob` (or
walk + regex) instead of `PurePosixPath.match`.

### 1.4 Exceptions during an iteration bypass `max_attempts` entirely

**Severity:** high
**Where:** `contractor/runners/task_runner.py:836-893` (also `runners/skills.py:59-62`, loaded
per-iteration at `task_runner.py:643`)

The retry loop only treats "ran to completion but status != DONE" as a failure. Any exception
inside `_run_single_iteration` — transient LiteLLM/network errors, an ADK `KeyError` from prompt
brace interpolation, a `FileNotFoundError` from a typo'd skill name — propagates straight out of
`run()`, aborting the whole multi-task workflow with no retry and no checkpoint entry for the
current task. This contradicts the documented invariant ("failures keep retrying until
`max_attempts` is exhausted") for the most common real-world failure mode, and is expensive in
hours-long runs.

**Fix:** catch a designated set of transient exceptions around `_run_single_iteration`, count
them as failed attempts (emit `ITERATION_RESULT` with `completed=False` + error payload), and
still raise `TaskNotCompletedError` on exhaustion. Let programming errors (template `KeyError`)
propagate.

### 1.5 Missing input artifacts silently render as empty strings

**Severity:** high
**Where:** `contractor/runners/task_runner.py:514-529`, `contractor/runners/_helpers.py:24-28`

`_load_artifact_text` passes the `Part | None` from `artifact_service.load_artifact` to
`_decode_part_text`, which returns `""` for `None`. A typo'd ref in `add_task(artifacts=[...])`,
a task YAML `artifacts:` entry, or an upstream task that never published yields an empty inbox
memory and an empty `{artifact__*}` substitution with zero diagnostics — the downstream task
runs against nothing and "succeeds". This already bites in practice: `OasEnrichmentWorkflow`
declares `dependency_information/result` / `project_information/result`
(`oas_enrichment/workflow.py:56-59`) which only a prior `oas_build` run publishes.

**Fix:** in `_load_artifacts`, raise (or at minimum `logger.warning` + emit an event) when a
declared input ref loads as `None`/empty. Keep a lenient path only behind an explicit
`optional` marker.

### 1.6 Fan-out tasks overwrite each other's published artifacts

**Severity:** high (latent contract break + wrong checkpoint validation)
**Where:** `contractor/runners/task_runner.py:531-545`, `contractor/runners/artifacts.py:30-35`;
fan-out callers: `trace_verify/workflow.py:112-137`, `exploitability/workflow.py:389`,
vuln-scan-trace per-finding tasks

Published artifacts are keyed `{template_key}/{result|summary|records}`. Workflows that queue
one task per finding/path from the same template overwrite each other's `result`/`summary`/
`records` — the exported artifact reflects only the last finding. Today the real verdicts
survive in `user:vulnerability-*` memory namespaces, but this silently breaks the "tasks
communicate only via artifacts" contract the moment someone adds a downstream
`artifacts: ["trace_verify/result"]`. It also hollows out checkpoint validation:
`_try_restore_from_checkpoint` (`task_runner.py:301-314`) only checks the artifact *exists*,
and any sibling finding's run satisfies that, so a restore can "validate" against another
task's content.

**Fix:** include the invocation `ref` (sanitized) in the artifact key for fan-out tasks
(e.g. `{template_key}/{ref}/{kind}`), or support an explicit per-invocation artifact key on
`add_task`.

---

## 2. Medium bugs

> **Status: all 20 fixed (2026-06-10)** — `de95fce` (§2.2–2.4, plus the `Checkpoint.load`
> hardening from §3.1), `287afa3` (§2.1, 2.5, 2.6, 2.8, 2.19, 2.20), `3b259fb` (§2.9–2.12,
> plus the decompose-message, 1–3-children schema, and unbounded-raw-output items from §3.2),
> `2a95169` (§2.13–2.16 + the dump_langfuse env path), `44c731c` (§2.7, 2.17, 2.18).
> Deliberate scope cuts: §2.11 got the once-per-invocation latch only — registering
> `ThinkingBudgetGuardrailCallback` as a hard stop is a production behavior change that should
> go through an eval first (the callback is kept for that). §2.10's prompt fix was a minimal
> in-place wording correction to v5 bootstrap step 2, not a new prompt version. §2.18 chose a
> per-request `httpx.AsyncClient` (no teardown seam reaches the agent factories); keep-alive
> pooling is forfeited — if that ever matters, thread a closer through `build_worker`.

### 2.1 `vuln_scan_fast._dedup` crash paths after the expensive scan stage

**Severity:** medium
**Where:** `contractor/workflows/vuln_scan_fast/workflow.py:211-215` (also `:280`)

Two crash paths in the programmatic dedup step:

- `f.get("details", "").split("CWE-")[1].split()[0]` raises `IndexError` when `details` ends
  with `"CWE-"` (the split tail is `""`, and `"".split()` is `[]`).
- A finding YAML with an explicit-null `details:` or `place:` makes `f.get("details", "")`
  return `None` → `"CWE-" in None` is a `TypeError`; `None.strip("/")` is an `AttributeError`.
  The same null-`details` issue hits `details[:500]` at line 280.

This runs *after* the expensive scan stage, so a single malformed finding aborts the whole
workflow before steps 4–5.

**Fix:** coerce `details = str(f.get("details") or "")`, same for `place`, and extract the CWE
with `re.search(r"CWE-(\d+)", details)`.

### 2.2 An exception in the event handler aborts the whole run

**Severity:** medium
**Where:** `contractor/runners/task_runner.py:472-486`, `contractor/runners/agent_runner.py:187-199`,
`cli/metrics.py:67-89`

`_emit` awaits the user-supplied handler unguarded; the CLI handler chains `MetricsSink.write`
(raw `open`/`json.dump` — raises `OSError` on disk-full/permissions) and `ui.on_event`. Any
observability failure therefore kills a potentially hours-long workflow mid-task — at odds with
the project's stated "observability is best-effort, don't gate it" philosophy.

**Fix:** wrap the handler call in `try/except Exception: logger.exception(...)` inside `_emit`
(both runners), since event delivery is telemetry, not control flow.

### 2.3 Metrics plugin: errored tool calls are never finished, corrupting retry accounting

**Severity:** medium
**Where:** `contractor/runners/plugins/metrics_plugin.py:198-213, 390-425, 427-477`

`on_tool_error_callback` sets `call.exception_seen = True` but never calls
`self._tracker.finish(call)`. Per the plugin's own comment, ADK does not fire `after_tool` after
an error unless a plugin returns a non-None error response (none do), so the errored call stays
unfinished in `_pending_by_fp`. When the agent retries the same tool with identical args (the
canonical retry-streak case), `before_tool` registers call₂, but the retry's `after_tool`
resolves the *first* unfinished call — the stale errored call₁. Consequences: the retry's
success is never recorded; `TOOL_RESULT` is emitted with `successful=False` for a successful
call; `execution_time_ms` is measured from call₁'s start; call₂ leaks; and the off-by-one
cascades to every subsequent identical call in the invocation.

**Fix:** finish the call in `on_tool_error_callback` (keep `exception_seen` only for the
documented after-error `after_tool` pairing case), or make `resolve()` prefer the most recent
unfinished call without `exception_seen`.

### 2.4 Checkpoint restore never verifies the entry still matches the invocation

**Severity:** medium
**Where:** `contractor/runners/task_runner.py:287-321`, `:262-265`

`CheckpointEntry` stores `template_key`/`template_version`, but `_try_restore_from_checkpoint`
matches solely on `ref` + artifact existence — it never compares template key/version against
the invocation, and params aren't fingerprinted. After editing a workflow (template version
change via `CONTRACTOR_TASK_VERSION_*`, changed params, or a different task reusing a ref), a
stale checkpoint silently skips the task and feeds old artifacts downstream. Also
`Checkpoint.workflow` is stored but never validated on load, and all runners share
`name="contractor"`-style naming, so cross-workflow checkpoint files are distinguishable only
by ref discipline.

**Fix:** skip restore (and log) when `entry.template_key/template_version` differ from the
invocation; consider adding a params hash to `CheckpointEntry`.

### 2.5 `--resume` misses checkpoints when refs are positional and tasks are conditional

**Severity:** medium
**Where:** `contractor/runners/task_runner.py:121`; affected workflows: `oas_building`,
`likec4_building`, `vuln_assess` (conditional discovery tasks)

Default task ref is `f"{name}:{len(self.queue)}"` and checkpoint entries are keyed by ref. In
workflows that add discovery tasks conditionally (skipped when artifacts already exist), the
same task gets a different ref between runs (`oas_update:2` → `oas_update:0`), so `--resume`
silently misses entries and re-runs completed work.

**Fix:** pass stable explicit `ref=`s in workflows with conditional task addition.

### 2.6 `app_name` inconsistency across the artifact contract

**Severity:** medium (latent — works only by accident)
**Where:** `contractor/workflows/oas_building/workflow.py:24,59`,
`oas_enrichment/workflow.py:23`, `likec4_building/workflow.py:57`, `cli/utils.py:27`,
`cli/main.py:220`, `contractor/runners/task_runner.py:516,539`

`OasBuildingWorkflow`/`OasEnrichmentWorkflow` build `TaskRunner(name="oas_builder")` and likec4
uses `name="likec4_builder"`, so artifacts publish/load with that `app_name` — while the same
workflows' `artifact_exists` skip checks and the CLI's final export use `app_name="contractor"`.
This only works because the current ADK `FileArtifactService` ignores `app_name` entirely; under
any app-partitioned service (`InMemoryArtifactService`, GCS) skip checks never match and
exports miss artifacts.

**Fix:** pass `ctx.app_name` as the TaskRunner name everywhere (the other 11 workflows already
use `"contractor"`).

### 2.7 `task_failed` permanently stops the live UI even when the workflow continues

**Severity:** medium
**Where:** `cli/main.py:70, 148-152`; `vuln_scan_trace/workflow.py:166-169`

`task_failed` is in `_UI_STOP_EVENTS`, but it's not always terminal: `vuln_scan_trace` catches
per-finding failures and continues, as does vuln_scan_fast's trace-confirm. After `ui.stop()`
the handler still takes the `if ui is not None:` branch and returns, so all subsequent events
are neither live-rendered nor print-fallback-rendered — the rest of the run is invisible.

**Fix:** only stop the UI on `run_finished`/`workflow_finished`, or set `ui = None` after stop
so the print fallback kicks in.

### 2.8 `trace_graph_pathpar` loses all completed paths if any path raises

**Severity:** medium
**Where:** `contractor/workflows/trace_graph_pathpar/workflow.py:140-152`

If any path's task raises, `asyncio.TaskGroup` cancels siblings and propagates, so
`merge_overlay_forks` and `_save_overlay_artifacts` never run — **all** completed paths'
annotations/diff are lost. `trace_annotation` solves exactly this with a `_cleanup` override
(`trace_annotation/workflow.py:149`); pathpar has none.

**Fix:** move the merge+save into `_cleanup` or wrap the TaskGroup so partial forks are still
merged and saved.

### 2.9 `finish`-time summarizer inherits the worker's full toolset and an uncapped records pool

**Severity:** medium
**Where:** `contractor/tools/tasks/tools.py:246-255, 739-748`

`summarizer_agent = LlmAgent(..., tools=agent_ref.tools, ...)` gives a pure-summarization agent
the worker's full domain toolset (fs/code/http/vuln) — wasted context on tool schemas and a
wandering risk for small models. The payload also uses `mgr.get_records(...)` unsliced (not the
`max_records` cap applied elsewhere at line 344), and the summarizer has none of the
`build_worker` callbacks (no `SummarizationLimitCallback`) — a long run can blow the context
window *inside* `finish`, the exception propagates, and the planner retries `finish` into the
same wall.

**Fix:** `tools=[]`, cap records (`[-max_records:]`), and consider truncating each record.

### 2.10 Planner prompt v5 instructs a `finish` call the tool is guaranteed to refuse

**Severity:** medium
**Where:** `contractor/agents/planning_agent/prompts/v5.md:71` vs
`contractor/tools/tasks/tools.py:724-733`

BOOTSTRAP step 2 says: "If memory already shows the objective is met → `finish(status='done',
...)` with no subtasks." But `finish` rejects `status="done"` unless at least one subtask exists
and is `done`. Following the prompt yields `DO_NOT_FINISH_WITH_NO_TASKS_DONE`, whose guidance
("set status='failed'") then pushes the planner to mark an already-met objective as failed,
burning a TaskRunner attempt.

**Fix:** either relax the gate for the zero-subtask case, or change the prompt to "add one
verification subtask, execute it, then finish". Related contradiction:
`TASK_LIMIT_REACHED_MSG` (`tools/tasks/models.py:20-24`) tells the planner to "call `finish`
immediately", but `finish(done)` is refused while `new` subtasks remain — reword to "execute or
skip the remaining subtasks, then finish".

### 2.11 Summarization-on-context-limit has no latch and no hard stop behind it

**Severity:** medium
**Where:** `contractor/callbacks/context.py:40-59`; `contractor/callbacks/guardrails.py:51-136`

Once the per-invocation counter exceeds `max_tokens` (it only grows; nothing resets it within an
invocation), the summarize message is appended to *every* subsequent LLM request. If the model
doesn't comply, nothing terminates the loop: `ThinkingBudgetGuardrailCallback` — the intended
hard stop — is registered nowhere in production, so the only backstops are planner
`max_steps`/task budgets.

**Fix:** inject the message once (or every Nth call), and register the budget guardrail (with
budget > summarization threshold) in `build_worker`.

### 2.12 `FunctionResultsRemovalCallback` can elide non-duplicate results after call/response misalignment

**Severity:** medium
**Where:** `contractor/callbacks/context.py:135-160, 201-204`

Function calls are paired to responses by global index order; on any mismatch (parallel calls
reordered, a call without a response, a guardrail-rewritten call name) the fallback signature is
`(name, "")`. Two responses falling back to `(name, "")` for the same tool — or legitimately
argument-less calls to a stateful tool — compare equal, and all but the newest are elided as
"stale", losing live context.

**Fix:** use a sentinel non-equal signature (e.g. include the response index) for unmatched
responses so they never dedup against each other.

### 2.13 Eval auto-skip gate defeated by any `-m` expression; env var parsed as truthy string

**Severity:** medium
**Where:** `tests/eval/conftest.py:190-192`

`if config.getoption("-m"): return` disables the eval skip for *any* marker expression, not just
`eval` — `pytest -m "not foo"` silently runs the full LLM-bound eval suite. Also
`CONTRACTOR_RUN_EVAL=0` *enables* evals (any non-empty string is truthy).

**Fix:** bypass only when `"eval"` appears in the markexpr; parse the env var as a boolean
(`in {"1", "true", "yes"}`).

### 2.14 `rebuild_eval_envelope.py` is stale relative to the current `EvalSink` layout

**Severity:** medium
**Where:** `scripts/rebuild_eval_envelope.py:44-50, 78-80` vs `tests/eval/results.py:506-519`

The script scans `eval_runs/<unit>/cases/*/metrics.json`, but `EvalSink._persist_case` now
writes `eval_runs/<RUN_STAMP>/<scenario>-<unit>-eval-<fixture>/cases/<case>/metrics.json`.
Confirmed on disk: new runs (e.g. `eval_runs/0607-converge-small/...`) are unreachable — the
script prints "skipped" and only works on legacy flat dirs.

**Fix:** also glob `eval_runs/*/{scenario}-{unit}-eval-*/cases/*/metrics.json`.

### 2.15 `EvalSink.flush` "latest pointer" collision between buckets sharing a unit

**Severity:** medium
**Where:** `tests/eval/results.py:493-501, 536`

Runs are bucketed by `(scenario, unit, metric_kind)` precisely so they don't merge, but
`run_name` defaults to `_safe_name(unit)` only — two buckets sharing `unit` (different scenario
or metric_kind) both write `eval_runs/<unit>/eval_results.json`, the second silently overwriting
the first in the same flush. Also `model`/`prompt_version`/`meta` from the first `record()` win
silently for the whole bucket.

**Fix:** include scenario/metric_kind in the default run_name; warn on conflicting
model/prompt_version within a bucket.

### 2.16 `prepare_vuln_benchmarks.py` ignores fetch/checkout failures — fixtures silently built from HEAD

**Severity:** medium (invalidates ground truth)
**Where:** `scripts/prepare_vuln_benchmarks.py:241-250`

In `clone_realvuln_repo`, the `git fetch --depth=1 origin <sha>` and `git checkout <sha>`
returncodes are discarded; on failure it still prints `OK (pinned to <sha>)` and the fixture is
built from HEAD instead of the pinned commit.

**Fix:** check `returncode` and raise, like the clone path already does.

### 2.17 `oas_analyzer` prompt factory silently drops objective/examples

**Severity:** medium (eval-only consumer)
**Where:** `contractor/agents/oas_analyzer/prompts/factory.py:20-27`

`TaskDescription.format()`'s chained conditional returns objective+instructions only
`if self.instructions`; with no instructions it returns *only* examples; with neither it returns
`""`. Verified impact: the `ddos/general`, `datasec/general`, `appsec/general` sub-agents run
with ROLE + OUTPUT FORMAT and **no objective at all**; `appsec/idor` and `appsec/ssrf` lose
their defined examples.

**Fix:** compose sections additively — objective always, then instructions/examples if present.

### 2.18 `http_tools()` never closes its `httpx.AsyncClient`

**Severity:** medium
**Where:** `contractor/tools/http.py:654-685`

`http_tools()` constructs `HTTPClient(...)` (which opens an `httpx.AsyncClient` in `__init__`)
and returns only the tool closures; nothing ever calls `aclose()` (grep confirms it's only
invoked by `caido.py` and the client's own `__aexit__`). Each agent build (`http_agent`,
`exploitability_agent`, `web_exploitability_agent`) leaks a client/connection pool for the
process lifetime — sockets leak over many runs, with "unclosed client" warnings.

**Fix:** expose the client for lifecycle management (return it / register async teardown on the
agent or runner), or lazily create-and-close per request.

### 2.19 Settings hygiene: `CONTRACTOR_TARGET_URL` / `CONTRACTOR_PROXY` read ad hoc

**Severity:** medium (convention violation)
**Where:** `contractor/workflows/exploitability/workflow.py:316,321`,
`vuln_assess/workflow.py:219`, `vuln_scan_fast/workflow.py:307`

All read `os.environ` directly, violating the "all config knobs live in `Settings`" convention
(`caido_url`/auth were already migrated correctly at `exploitability/workflow.py:323-326`).
Related: `ExploitabilityWorkflow.__init__` raises a bare `ValueError` when
`CONTRACTOR_TARGET_URL` is unset (`:318`) → uncaught traceback from the CLI instead of a clean
click error, since instantiation happens inside `async_main` (`cli/main.py:204`).

**Fix:** add `target_url: str | None` and `proxy: str | None` to `Settings`, route through
`get_settings()`, and convert the missing-value error to a `click.UsageError` in the CLI.

### 2.20 `.env` discovery only works from the CLI entrypoint

**Severity:** medium
**Where:** `contractor/utils/settings.py:24-26`; `cli/main.py:14`;
`scripts/dump_langfuse_trace.py:367`

`Settings` declares `env_file=".env"` (CWD-relative) and settings.py's own `load_dotenv()` walks
up from `contractor/utils/` — neither finds the documented `cli/.env`. It is only picked up
because `cli/main.py` calls `load_dotenv()` from inside `cli/` before importing contractor. Any
non-CLI entrypoint (scripts, tests importing Settings first) silently gets no env file. Concrete
casualty: `dump_langfuse_trace.py:_load_env` looks for `contractor/cli/.env`, a path that does
not exist — its advertised Langfuse-cred auto-load never works and the script exits 2 unless
creds are exported manually.

**Fix:** anchor the env file path explicitly (e.g. resolve `cli/.env` relative to the repo root
via `Path(__file__).parents[...]`) in one place; fix the script's candidate list.

---

## 3. Low-severity bugs and footguns

> **Status: done (2026-06-10)** — `08c30f9` (§3.1), `aa22fc7` (§3.2 + likec4 from §3.3),
> `38390f9` (§3.3 fs/sandbox + the walk-ceiling knob `FS_MAX_FILES_PER_WALK`), `d104874`
> (§3.4), `55c56d4` (§3.5 + the per-path `artifact_key` follow-up), `d971e28` (§3.6).
> Two review findings were WRONG and are corrected here rather than "fixed":
> `AgioEventType.TASK_SKIPPED` is not dead — `Workflow.emit_task_skipped` emits it from four
> workflows (kept, now pinned by a mirroring test); and `after_run_callback` DOES fire in the
> TaskRunner nesting (verified empirically with a probe plugin) — it only skips on mid-stream
> errors/cancellation, so `SandboxCleanupPlugin` is the live primary teardown and the `run()`
> sweep is the backstop; the task_runner comment was the wrong one. Also: the per-path
> clobbering noted under §1 applied only to `trace_annotation` — the other three trace
> workflows drive `AgentRunner` directly and publish no task artifacts. The TOCTOU item got a
> real fix (resolved-path), not just a comment. Rate-limit callbacks stay sync (`CallbackChain`
> composes synchronously) with a prominent blocking warning instead of async conversion.
> Deliberately kept: `ThinkingBudgetGuardrailCallback`, `ToolMaxCallsGuardrailCallback`,
> `Tpm/RpmRatelimitCallback` (now production-ready for later wiring).

### 3.1 Runners

- **`Checkpoint.load` crashes on structurally malformed entries** —
  `contractor/runners/models.py:456-468`. Only `json.JSONDecodeError`/`OSError` are caught; a
  valid-JSON checkpoint with an entry missing `task_id`/`ref`/`template_key` raises an uncaught
  `KeyError`. And `_load_checkpoint` is called *outside* the `try` in `run()`
  (`task_runner.py:154`), so the `finally` never runs and the `self._on_event` set just before
  leaks on the instance. Fix: catch `KeyError`/`TypeError` in `Checkpoint.load` with the same
  "ignoring corrupt checkpoint" warning; move `_load_checkpoint()` inside the `try`.
- **Contradictory claims about `after_run_callback`** — `task_runner.py:236-237` says it "does
  not fire in the TaskRunner + AgentTool nesting" while
  `runners/plugins/sandbox_cleanup.py:12-15` asserts it "reliably fires" and relies on it for
  teardown. One is wrong; if the task_runner comment is right, `SandboxCleanupPlugin` never
  tears down and sandboxes survive until workflow end. Resolve with one instrumented run.
- **Duplicated event-type enums; unmirrored types silently dropped** —
  `runners/models.py:27-39` vs `runners/agio.py:25-83`; `cli/metrics.py:82-84` filters on
  `ALL_AGIO_EVENT_TYPES`. Adding an `EventType` member without mirroring it in `agio.py`
  silently loses those events from `metrics.jsonl`. Also `AgioEventType.TASK_SKIPPED` is emitted
  nowhere (dead member). Fix: add a one-line unit test `set(EventType) <= ALL_AGIO_EVENT_TYPES`;
  emit or delete `TASK_SKIPPED`.
- **`AgentRunner._on_event` PrivateAttr races on concurrent `run()`** —
  `agent_runner.py:53,73,141-142`. Two concurrent runs on one instance clobber each other's
  handler; the first to finish silences the second's events. Fix: pass the handler down the call
  chain, or document non-reentrancy like `TaskRunner` does (`task_runner.py:87-89`).
- **`_artifact_var_name` normalization can collide** — `runners/models.py:202-210, 355-356`.
  `oas-build/result` and `oas_build/result` both map to `artifact__oas_build__result`; the later
  artifact silently wins. Fix: detect collisions and raise.
- **`TaskTemplate.load` raises bare `KeyError` for missing required fields** —
  `runners/models.py:255-257`. `raw["objective"]` etc. crash with an opaque `KeyError` while the
  surrounding code raises descriptive `ValueError`s. Fix: validate the three fields with the
  same pattern.
- **carry_state accumulates dead invocation-scoped planner keys across attempts** —
  `tools/tasks/manager.py:38-46` + `task_runner.py:498-507`. Each retry drags every previous
  attempt's `task::{gid}::{invocation_id}::…` keys along — unreachable but deep-copied twice per
  iteration and re-emitted in every trace-plugin `snapshot_state`, inflating `metrics.jsonl`.
  Fix: strip stale invocation-scoped keys when building carry/initial state.
- **Artifact bytes decoded with `errors="ignore"`** — `runners/_helpers.py:36-41`. Non-UTF-8
  inline data is silently truncated/corrupted on load. Use `errors="replace"` + warning log.
- **Skill names validated lazily; skills/artifacts re-injected every attempt** —
  `task_runner.py:120, 643-652`. A typo'd skill only surfaces as `FileNotFoundError` when the
  task's first iteration starts, possibly hours in; `add_task` could existence-check eagerly.
  And `_run_single_iteration` re-reads all skill files and rewrites the memory artifact on every
  attempt — idempotent but wasted I/O; inject once per task before the retry loop.
- **`artifacts=[]` falls back to template defaults** — `task_runner.py:135`.
  `list(artifacts or template.default_artifacts)` means an explicit empty list (passed by
  `trace_annotation/workflow.py:215` to mean "none") resurrects defaults — asymmetric with
  `skills` one line below which correctly tests `is not None`. Fix: mirror the `is not None`
  test.

### 3.2 State machine / planner tools

- **`skip` conflates three outcomes into one non-error result** —
  `tools/tasks/tools.py:462-465` + `manager.py:183-230`. `mgr.skip()` returns `None` both when
  the skip succeeded with no next subtask and when it was silently rejected
  (`InvalidStatusTransitionError` swallowed at `manager.py:200-209`); the planner can't tell
  whether the skip happened. Fix: return `(skipped, next_subtask)` or surface the
  invalid-transition case as `{"error": ...}`.
- **`decompose_subtask` reports "task limit reached" for unrelated failures** —
  `tools/tasks/tools.py:404-410`. Every `None` from `mgr.decompose_current_subtask` maps to
  `TASK_LIMIT_REACHED_MSG`, which can mislead the planner into skipping when retrying with 1
  child would fit. Also `len(insertion) == 0` at `:409` is unreachable (`min_length=1`).
- **"1–3 children" contract not enforced in schema** — `tools/tasks/models.py:135-149`.
  `SubtaskDecomposition.subtasks` declares `min_length=1` only; a model can decompose into 10
  children with only the global `max_tasks` budget pushing back. Fix: add `max_length=3`.
  (The prompt's "depth limit 1" rule is likewise unenforced — acceptable, but noted.)
- **Malformed-path raw output stored unbounded** — `tools/tasks/tools.py:604-653`. When worker
  output can't be parsed, the entire raw response is stored verbatim in the record and echoed to
  the planner; `_MAX_LITERAL_EVAL_LEN` caps parsing only. One 200KB garbage response inflates
  every later `get_records`/finish-summarizer call. Fix: truncate with a marker.
- **`task_tools(worker_instrumentation=False)` silently assumes `input_schema` is set** —
  `tools/tasks/tools.py:530-533`. RouterWorkflow remembers
  (`workflows/router/workflow.py:113-114`) but nothing validates it; a future caller gets an
  obscure ADK `KeyError('request')`. Fix: raise early in `task_tools` when instrumentation is
  off, `use_input_schema=True`, and the worker has no `input_schema`.
- **Wedge analysis: clean.** All transition paths in `SUBTASK_STATUS_TRANSITIONS` were traced
  against the tool surface — `incomplete`/`malformed` are always resolvable, `finish(failed)` is
  never gated; no permanent wedge exists with the default toolset.

### 3.3 Sandbox / filesystem

- **`walk()` leaks symlinked-outside file *names*** — `cli/fs.py:149-167`. `walk()` prunes
  symlinked directories but not symlinked files from the yielded list (confirmed by repro);
  content remains blocked on read, so this is name-disclosure only and an inconsistency with
  `ls`. Fix: filter `files` the way `ls`/`glob` do.
- **Check-then-use TOCTOU in `_strip_protocol`** — `cli/fs.py:129-135`. Validates
  `realpath(candidate)` but returns the unresolved candidate; a component could be swapped to an
  escaping symlink between check and open. Very low practical risk for a local single-user
  sandbox; a robust fix opens via the resolved path / `O_NOFOLLOW`.
- **`merge_overlay_forks` compares the delete-set against the wrong set** —
  `tools/fs/merge.py:105-108`. `fork._deleted - set(pre_fork_files)` subtracts tombstones from
  *file-content* keys (essentially always disjoint), so the filter is a no-op and every fork
  tombstone propagates — including deletes that predated the fork. Fix: capture
  `pre_fork_deleted` and subtract that.
- **`likec4_tools()` raises at tool-build time when the binary is missing** —
  `tools/likec4.py:71-72, 261`. `Likec4Linter.__post_init__` eagerly resolves the command, so a
  missing `likec4`/`bunx`/`npx` crashes workflow *assembly* instead of surfacing as a tool
  result, contradicting the docstring and the per-call `validate()`. Fix: resolve lazily inside
  `validate()`/`_impl`.
- **Unbounded tree walks in overlay glob/grep** — `FsspecInteractionFileTools.glob`/`grep` walk
  the full tree with no `max_files` bound, unlike `code/tools.py:_iter_all_files` (hard-bounded
  by `code_max_files_per_walk`). On very large repos fs grep/glob can run away. Consider a
  shared walk ceiling.

### 3.4 Callbacks / observability

- **Dead code:** `ThinkingBudgetGuardrailCallback` + `ToolMaxCallsGuardrailCallback`
  (`guardrails.py:51-136`) registered nowhere in production; `Tpm/RpmRatelimitCallback`
  (`ratelimits.py`) used only by test dummies; `FINISH_MAX_CALLS_RVALUE`
  (`planning_agent/agent.py:24-26`) orphaned; `TokenUsageCallbackException` (`tokens.py:14`)
  never raised; `verify_signature` (`callbacks/base.py:53-58`) commented out and always `True`,
  making `BaseCallback.validate()` a no-op that pretends to validate. Wire up or delete.
- **`time.sleep` in rate-limit callbacks blocks the event loop** — `ratelimits.py:70,126`.
  Would stall all concurrently running agents (trace-parallel workflows) if ever enabled. Use
  `await asyncio.sleep` before wiring them up.
- **`RpmRatelimitCallback` never rolls the window when under the limit** —
  `ratelimits.py:109-138`. Unlike TPM there's no stale-window reset branch; the count only
  resets when the limit is crossed. Approximate rather than wrong (no false sleeps), but
  inconsistent with the TPM implementation.
- **`TokenUsageCallback` never flushes the final invocation to history** —
  `tokens.py:116-121`. History is written only on invocation-id change, so per-invocation
  history consumers undercount by the last invocation. Also Russian inline comments at
  `:106-116` in an otherwise English codebase.
- **`observability.run_context` can raise when Langfuse is enabled but broken** —
  `utils/observability.py:122`. `start_as_current_span(...).__enter__` failures propagate,
  contradicting the module's never-raises contract. (The disabled-path no-op invariant itself is
  correctly upheld — verified.) Also `init()` sets `_initialized=True` after a *failed* init so
  it never retries — probably intentional, deserves a comment.
- **Module-level `logger.setLevel(...)` in library code** — `ratelimits.py:12`,
  `guardrails.py:16` (DEBUG at import), plus `trace_annotation/workflow.py:21`,
  `trace_annotation_direct/workflow.py:28`, `trace_graph/workflow.py:41`,
  `trace_graph_pathpar/workflow.py:44` — overrides the CLI's `_QUIET_LOGGERS` policy. Remove.

### 3.5 Agents / prompts

- **Three agents share the description `"software engineering agent"`** —
  `swe_agent/agent.py:54`, `oas_builder_agent/agent.py:56`, `oas_linter_agent/agent.py:51`.
  Descriptions become AgentTool declaration text; RouterWorkflow exposes all three
  simultaneously, so the dispatch model sees three identically described tools. Give
  builder/linter accurate descriptions.
- **http_agent prompt references a nonexistent "report tool"** — `http_agent/agent.py:16-18`.
  At context limit the prompt instructs an impossible action; reword to "persist findings to
  memory and return the structured result".
- **Planner v5 Rule 6 example is format-specific** — `planning_agent/prompts/v5.md:31` shows a
  `<task><result>` XML shape, but the default planner `_format` is `json`. Harmless for the
  rule's intent, wrong for the default config.
- **oas_analyzer nits** — severity sort is alphabetical so `low` sorts before `medium`
  (`sub_agents/report_agent.py:75` — use a rank map); sub-agent build order iterates a set
  literal so it varies per process (`sub_agents/analytic_agents.py:135` — use a tuple);
  `BotFactory.build`'s `output_schema: BaseModel | None` should be `type[BaseModel] | None`
  (`:32`).
- **Leftover dir** — `contractor/agents/code_graph_agent/` contains only a gitignored
  `__pycache__` for a deleted module; remove.

### 3.6 Eval harness / scripts

- **`harness.run_agent` loses all captured partial results on timeout** —
  `tests/eval/harness.py:176`. `asyncio.wait_for` timeout discards the partial `AgentRun`;
  returning it would aid debugging.
- **`select_fixture` return annotation lies** — `tests/eval/conftest.py:340`:
  annotated `-> EvalFixture | None` but can never return None (`_load_fixture` raises).
- **`analyze_metrics.py` pricing table is stale** — lines 72-77 carry Gemini pricing in an
  lm-studio-default project, so cost columns are fiction. Informational only; drop or label.
- **`MetricsSink` nit** — payload keys shadowing envelope keys (`type`, `task_name`, …) are
  silently dropped by `setdefault` (`cli/metrics.py`) — by design, but worth a comment.

---

## 4. Structural improvements (duplication / dead config / doc drift)

- **trace-direct and trace-graph are now functionally identical.**
  `trace_annotation_direct/config.yaml:4` sets `with_graph_tools: true` — same template, same
  loop, same overlay contract as trace_graph. The trace_graph docstring
  (`trace_graph/workflow.py:1-10`, "vs the prompt-only baseline") and the README A/B framing are
  stale. Decide which workflow survives, or restore `false` in trace-direct to keep the
  baseline.
- **Dead `output_format` knobs** — `trace_annotation_direct/config.yaml:4` and
  `trace_graph/config.yaml:4` declare `output_format: json` for `trace_agent`, but both
  workflows pass `_format=cast(TraceFormat, self._template.format)` instead — the YAML knob is
  ignored. Drop it or read it.
- **`contractor/workflows/shannon/`** contains only `DESIGN.md` — a placeholder folder inside
  the workflows package violating the one-folder-per-workflow structure (no workflow.py /
  config.yaml / `__init__.py`, not registered). Move to `docs/` or `docs/research/`.
- **Extraction candidates (verbatim duplication):**
  - the ~35-line `_chain` after_model-callback merge in `exploitability_agent/agent.py:147-175`
    and `web_exploitability_agent/agent.py:137-165`, along with duplicated
    `_READ_ONLY_VULN_TOOL_NAMES` / `_VERDICT_TOOL_NAMES` (a third copy of the read-only set
    lives in `trace_verifier_agent/agent.py:28-30`) — extract a
    `chain_after_model_callback(agent, cb)` helper (natural home: `worker_factory.py` or
    `callbacks/adapter.py`) plus a shared vuln-tool-names constant;
  - the conditional discovery block (`dependency_information` + `project_information` add_task
    with skip-emit) copy-pasted in 4 workflows (oas_building, likec4_building, vuln_assess,
    vuln_scan_fast);
  - the per-operation trace loop (~70 lines incl. plugin wiring) triplicated in
    trace_annotation_direct / trace_graph / trace_graph_pathpar;
  - the YAML findings loader (name→fields dict to list-of-dicts with `setdefault("name")`)
    appearing 5 times (exploitability:445, trace_verify:140, vuln_scan_fast:167,
    vuln_scan_trace:171, vuln_assess:262).
- **fs tool registry boilerplate** — read_tools.py vs write_tools.py have ~5 near-duplicate
  tool-registry closures differing only in docstring; delegation is already deduped via
  `_reader`, the closure boilerplate could be factored.

---

## 5. Verified clean

For the record, areas that were explicitly checked and found sound:

- **Sandbox escapes**: symlinked files/dirs, absolute paths, and `..` traversal all blocked by
  `RootedLocalFileSystem` (confirmed with repro scripts); content never leaks (only the
  `walk()` name-disclosure nit in §3.3).
- **Retry/iteration accounting** in `_run_task_with_retries` correctly implements the documented
  invariant (cumulative `successful_runs`, no reset on failure, `TaskNotCompletedError` on
  exhaustion; `_resolve_retry_params` enforces `max_attempts >= iterations`).
- **Workflow registry**: all 14 name→class→folder mappings load (`get_workflows()` executed);
  every `CFG.budgets.* / CFG.tasks.* / CFG.agent(...)` read exists in its sibling YAML; no
  missing or dead config keys beyond the `output_format` knobs noted above.
- **Task templates**: all brace tokens in all 24 version bodies are covered by caller-supplied
  variables/params; no ADK bare-`{id}` hazards anywhere (agent prompts machine-checked with
  ADK's exact regex); literal-brace `{{...}}` escapes are correct.
- **Prompt manifests**: all 16 `prompt.yml` files consistent — every `active` declared, every
  declared version file exists, no dead prompt files on disk.
- **Memory namespaces**: `user:memory/{name}`, `http/{name}/...`, `user:oas-{name}` keyspaces
  disjoint; reserved-tag (`skill`/`inbox`) isolation and write-collision guard correct.
- **`format_output` byte/line caps** (fs/format.py): dual truncation with footer-fit re-trim and
  resume-offset recomputation correct; no off-by-one.
- **openapi/ref_resolver.py**: cycle detection, `max_depth`, and pointer parsing correct.
- **Checkpoint and metrics persistence**: atomic save (tmp + `replace`); `MetricsSink` has no
  buffer/handle leaks, crash-safe JSONL.
- **Token accounting**: no double counting; `TokenUsageCallback` updates the global counter
  exactly once per response.
- **Observability disabled-path**: every public function checks `_enabled()` first — true no-op
  when Langfuse is off.
- **Conventions**: zero `assert` statements in production code across the entire reviewed
  surface; test collection healthy (1048 unit+integration tests collect cleanly;
  `tests/units/contractor_tests/tools/test_tasks.py` passes 55/55 against current code).

---

## Suggested first batch

Small, independently testable, highest payoff:

1. §1.1 guardrail short-circuit (+ chain unit tests)
2. §1.2 trace-verify namespace prefixes
3. §1.3 overlay glob matcher
4. §1.4 transient-exception retry accounting
5. §1.5 loud failure on missing declared artifacts
6. §1.6 fan-out artifact keys (include invocation ref)
