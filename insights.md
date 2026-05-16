# Insights — open-source feature borrowings for contractor

Working notes from auditing four OSS repos for ideas applicable to contractor.
Clones live under `.research/` (untracked).

## Repos audited

| Repo | License | Verdict |
|---|---|---|
| `bytedance/UI-TARS-desktop` (`.research/UI-TARS-desktop`) | Apache-2.0 | Not really — one borrow (Agio event taxonomy) |
| `rohitg00/agentmemory` (`.research/agentmemory`) | MIT (TS) | Partially — 2–3 primitives |
| `Hmbown/DeepSeek-TUI` (`.research/DeepSeek-TUI`) | MIT (Rust) | Partially — 3 UX patterns |
| `statewright/statewright` (https://github.com/statewright/statewright) | Apache-2.0 (Rust) | Skip wholesale, borrow one idea |

## Punch list

### ✅ Done

- **Agio event taxonomy** (UI-TARS `multimodal/tarko/agio/src/index.ts`).
  Implementation: `contractor/runners/agio.py`, refactored emitters in
  `contractor/runners/plugins/{metrics,trace}_plugin.py`, sink in
  `cli/metrics.py`. Commit `c489aaf`.
  - Renames: `metrics_tool_call → tool_call`, `metrics_tool_result → tool_result`,
    `metrics_tool_exception_error → tool_exception`, `metrics_llm_usage → llm_usage`,
    `metrics_summary → run_summary`, `metrics_fs_coverage → fs_coverage`,
    `adk_before_run → agent_run_start`, `adk_after_run → agent_run_end`.
    Trace plugin's `tool_call/result/error` moved to `adk_tool_*` namespace.
  - Field renames: `tool_args → arguments`, `duration_ms → execution_time_ms`,
    usage keys `prompt/completion → input/output`.
  - New fields on tool events: `tool_call_id`, `arguments_size`, `result_size`,
    `successful`.
  - Records on disk are flat — no nested `payload`. Adds `timestamp` (ms epoch)
    + `ts_iso`. Full `arguments` + `result` are preserved (analysis needs them;
    upstream Agio sanitises to sizes only — we override).
  - UI split: `tool_exception` (Python raise) vs `tool_result` with
    `successful=False` (LLM-facing error). Distinct render paths in
    `cli/render.py` + `cli/ui.py`.
- **`scripts/analyze_metrics.py`** updated to read the new Agio shape.
  Commit `ce81eab`.

### ⬜ Pending — task-to-task `handoff_memory` primitive

Borrowed concept from agentmemory's cross-run namespace, re-framed as an
**explicit programmer-controlled primitive** that workers call when they want
to publish a memory note for a downstream task.

**Gap it closes**: today task-to-task communication is artifacts-only (declared
in YAML, single result/summary/records blob per task). Workers can't decide at
runtime "pass X to the trace task".

**Design**:
- New tool in `contractor/tools/memory.py`:
  `handoff_memory(name, memory, description, tags=None, to_task=None)`.
  `to_task=None` = broadcast to all downstream tasks.
- Storage: piggyback `BaseArtifactService` with filename convention
  `handoff/<run_id>/<to_task or "*">/<note_name>`.
- Wire-up in `contractor/runners/task_runner.py`: before spawning worker for
  task `T`, scan `handoff/<run_id>/{T,*}/*` and call existing
  `MemoryTools.inject(..., tag="inbox")`. Receiver side is already wired —
  workers have `inbox_list` / `inbox_read` tools (`memory.py:681-719`).
- ~120 LOC total.

**Open decisions**:
- Scope: only-already-scheduled downstream tasks (cleanest) vs queue for tasks
  added later in the run.
- Persistence: per-run only (default), or also persist across runs keyed on
  `(project_path, pipeline)` — opens cross-run handoff but increases blast
  radius.

### ⬜ Pending — versioned `MemoryNote` + Jaccard-dedup

Borrowed shape from agentmemory `src/types.ts:67-87` + `jaccardSimilarity` from
`src/state/schema.ts:68-78` (~10 LOC, no deps).

**Gap it closes**:
- `MemoryNote` (`contractor/tools/memory.py:17-25`) has no history; `write_memory`
  is overwrite-on-name. When the planner re-derives "the same fact" under a
  slightly different name, both are stored forever. Unbounded YAML growth in
  `MemoryTools.save` (`:387-390`).

**Design**:
- Extend `MemoryNote` with: `kind` (taxonomy: `pattern/preference/architecture/bug/workflow/fact`),
  `version`, `supersedes: list[str]`, `is_latest: bool`, `forget_after: Optional[str]`.
- On write: tokenize-and-Jaccard against every existing note sharing any tag.
  If similarity > threshold and name differs, treat new as a version of closest
  match (bump version, set `supersedes`, flip old `is_latest=False`).
- On `list_memories` / `search_memory`: filter `is_latest=True` by default;
  expose `include_history` flag.
- Add `prune_expired(now)` called from `save`.
- ~80 LOC. No new deps. Old YAML loads with `version=1, is_latest=True`.

**Open decision**: Jaccard threshold — start at 0.75?

### ⬜ Pending — `analyze_metrics.py` Agio update

Done (`ce81eab`). Listed here for completeness.

## Reject outright

- All Electron / screenshot / browser / mouse-keyboard bits of UI-TARS-desktop.
- DeepSeek-TUI's ratatui rendering, OS sandbox crates (we already have
  `RootedLocalFileSystem`), DeepSeek prefix-cache cost tracker, MCP/ACP servers.
- agentmemory's MCP server, viewer, embedding/circuit-breaker stack, `iii-sdk`
  daemon, multi-agent governance modules (`graph/team/routines/viewer/sentinels/
  crystals/leases/governance/mesh`).
- statewright's Rust runtime + MCP gateway + cloud — large blast radius for one
  FSM contractor already enforces in `contractor/tools/tasks.py:288`.

## Honorable mentions (worth porting when touched)

- **statewright** per-state tool gating — filter `agent.tools` by current
  `SubtaskStatus` in `contractor/tools/tasks.py:107` `instrument_worker`.
  Their SWE-bench delta on small local models (2→10 of 10) maps to our
  `lm-studio-qwen3.6` case.
- **DeepSeek-TUI approval handshake** (`core/engine/approval.rs`) — wire as a
  `before_tool_callback` for `contractor/tools/http/*` and `RootedLocalFileSystem`
  writes.
- **UI-TARS snapshot replay harness** (`multimodal/tarko/agent-snapshot/`) —
  record LLM req/resp per loop iteration; normalize non-determinism; replay
  against a stubbed model. Would unblock offline eval (no LiteLLM proxy needed).
  Target: `tests/eval/harness.py`.
- **agentmemory BM25 index** (`src/state/search-index.ts`) ~150 LOC, no deps —
  term-weighted recall in `MemoryTools.find` without pulling embeddings.
- **DeepSeek-TUI session save/resume** (`session_manager.rs`) — `--resume <run-id>`
  skipping finished `TaskInvocation`s after crash. Target:
  `contractor/runners/task_runner.py` + `artifacts.py`.

## Domain insight worth keeping

Tool failure has two distinct signals in contractor — keep them separate:

| Signal | Source | Event type | LLM sees it? |
|---|---|---|---|
| Python exception in tool | ADK `on_tool_error_callback` | `tool_exception` | No (infra) |
| Tool returned an error payload | `after_tool_callback` with `result_error=True` | `tool_result` with `successful=False` | Yes (expected to react) |

Conflating these in event types, callbacks, or UI obscures retry-vs-fix logic.

## Outstanding state

- Branch `main` is 8 commits ahead of `origin/main` (not pushed).
- `.research/` clones sit in the working tree, untracked. Keep for further
  drilling on pending items, or remove — user's call.
