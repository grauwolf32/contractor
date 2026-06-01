# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Contractor is a CLI that drives Google-ADK + LiteLLM agents to generate, enrich, and trace OpenAPI specs from a project's source code. It is **not** an HTTP service — every "run" is a CLI invocation that spins up a workflow of agents over a target codebase and writes artifacts to a local directory.

See [docs/README.md](docs/README.md) for the deep dive on planner/worker internals, the streamline subtask state machine, and the memory / artifact contract. Read it before changing anything in `contractor/runners/` or `contractor/agents/planning_agent/` — it documents non-obvious invariants (e.g. terminal subtask states, `finish` semantics).

## Common commands

```bash
poetry install                              # install runtime + dev deps
poetry run contractor --help                # main CLI (entrypoint: cli.main:main)

# Workflows (see contractor/workflows/__init__.py for the full registry):
#   oas_build | oas_update | likec4 | trace | trace-direct | trace-graph | trace-verify | router
poetry run contractor --workflow oas_build --project-path ./target --folder-name src --model lm-studio-qwen3.6
poetry run contractor --workflow oas_update  --project-path ./target --artifact openapi.yaml --model ...
poetry run contractor --workflow router  --project-path ./target --prompt "..."   # prompt-driven; interactive prompt opens when --prompt omitted

# Tests
poetry run pytest                           # unit + integrational; eval tests are auto-skipped
poetry run pytest tests/units/              # unit tests only
poetry run pytest -m eval                   # run LLM-bound eval suite (slow, needs LiteLLM proxy up)
CONTRACTOR_RUN_EVAL=1 poetry run pytest tests/eval/test_trace_agent_eval.py
poetry run pytest tests/eval/test_trace_agent_eval.py -k vulnyapi  # single fixture
# Eval-only env overrides: CONTRACTOR_EVAL_MODEL, CONTRACTOR_EVAL_TRACE_PROMPT_VERSION

# Lint / format / type-check (dev deps only — no pre-commit hook, no CI config in repo)
poetry run ruff check .
poetry run isort .
poetry run mypy            # scoped to contractor/ + cli/ via [tool.mypy] files; ~180 pre-existing errors
poetry run pytest --cov    # coverage configured in [tool.coverage]; not gated/forced by default

# Inspect a run after it finishes
python scripts/analyze_metrics.py <output_dir>/metrics.jsonl    # charts + tables
python scripts/dump_langfuse_trace.py ...                       # pull a Langfuse trace
```

### Submodule (required for eval fixtures)

`tests/playground/` is a git submodule (`security-playground`). The eval fixtures under `tests/eval/fixtures/<slug>/meta.yaml` resolve `source_root` into this tree, so evals fail with a clear error if it is missing.

```bash
git submodule update --init --recursive
```

### LiteLLM proxy + .env

`--model` is **not** a model name — it is an alias from `deploy/litellm/litellm_config.yaml` resolved by the LiteLLM proxy. Start the proxy (`cd deploy/litellm && bash run.sh`, requires Podman) before running any workflow. Configuration is read from `cli/.env` via `pydantic-settings`; the full schema lives in `contractor/utils/settings.py` (`Settings`). Don't add config knobs anywhere else.

## Architecture cheatsheet

```
cli/main.py                       — Click entrypoint; wires flags into WorkflowContext
contractor/workflows/__init__.py  — Workflow base + WorkflowContext + get_workflows() registry
contractor/workflows/config.py    — WorkflowConfig.load(__file__): typed loader for per-folder config.yaml
contractor/workflows/<mode>/      — one folder per workflow: workflow.py (thin assembler: build
                                    TaskInvocations, hand them to TaskRunner) + config.yaml
                                    (budgets / tasks / per-agent tool options)
contractor/runners/task_runner.py — per-task state machine: render → spawn planner → iterate → publish
contractor/runners/agent_runner.py— bare single-agent runner used by RouterWorkflow
contractor/runners/models.py      — TaskTemplate / TaskInvocation / RenderedTask / TaskScopedKeys
contractor/runners/artifacts.py   — `{template_key}/{result|summary|records}` artifact naming + save
contractor/runners/skills.py      — load `contractor/skills/<name>/*.md` and inject as memories
contractor/runners/plugins/       — ADK plugins (trace + metrics) attached to every Runner
contractor/agents/planning_agent/ — streamline planner: tools = add_subtask / execute_current_subtask
                                    / decompose_subtask / skip / finish + memory tools
contractor/agents/<worker>/       — workers (swe, oas_builder, oas_linter, trace, http, router, …)
contractor/tasks/*.yml            — task templates loaded by name (`TaskTemplate.load`)
contractor/skills/<name>/         — markdown reference bundles pre-injected per task
contractor/tools/tasks.py         — streamline subtask schemas + status transitions
contractor/tools/{fs,code,memory,openapi,vuln,http,likec4}/ — tool surface for workers
contractor/callbacks/             — token usage, guardrails, summarization-on-context-limit, rate limits
```

Key invariants worth knowing before editing:

- **Workflows live in `contractor/`, one folder each.** Each workflow is `contractor/workflows/<mode>/` with `workflow.py` (the assembler), `config.yaml` (its tunable budgets/tasks/agents), and `__init__.py` re-exporting the public `*Workflow` class. `cli/` only imports `contractor.workflows`, never the reverse. To add a workflow: create the folder, write `workflow.py` + `config.yaml`, register the class in `get_workflows()`.
- **Per-workflow tunables are data, not code.** `workflow.py` reads `CFG = WorkflowConfig.load(__file__)` and uses `CFG.budgets.<name>` (token budgets / scalars), `CFG.tasks.<name>.as_kwargs()` (retry/iteration/step budgets), and `CFG.agent("<agent_name>")` (per-agent tool options: `.output_format` → the `_format` knob, `.with_graph_tools`) — all from the sibling `config.yaml`. `CFG.agent()` returns an all-default `AgentToolConfig` for agents not listed, so YAML need only declare the ones it tunes. To tune, edit the YAML. This is *not* global `Settings` (that's env/tool defaults + LLM sampling in `contractor/utils/settings.py`).
- **Workflows never call agents directly.** They build `TaskInvocation`s with a `worker_builder` partial and let `TaskRunner` spawn a fresh planner+worker pair per attempt. Use this pattern when adding a new workflow.
- **Tasks communicate only via artifacts.** Each task declares `artifacts: ["<upstream_template_key>/result", ...]` in `add_task()`. The runner loads those artifact texts and re-injects them into the next task's memory namespace tagged `inbox` / `previous-task-result`. Don't pass data via shared variables or globals.
- **`iterations` vs `max_attempts`.** A task is only marked finished after `iterations` *successful* runs in a row; failures keep retrying until `max_attempts` is exhausted (then `TaskNotCompletedError`). Defaults come from the template's `iterations:` field.
- **Workers are uniform.** `instrument_worker` (in `contractor/tools/tasks.py`) attaches `input_schema = Subtask` and `output_schema = SubtaskExecutionResult` plus a worker-instructions trailer to *any* LlmAgent, so adding a new worker means writing one prompt + one `build_<agent>` factory — no planner glue.
- **Subtask state machine is strict.** See `SUBTASK_STATUS_TRANSITIONS` in `contractor/tools/tasks.py`. `incomplete` / `malformed` cannot be re-executed — only decomposed (1-3 children) or skipped. `finish` is the only way to mark a task `done`, and it refuses if any subtask is still `new`.
- **Filesystem is sandboxed.** Workers see a `RootedLocalFileSystem` (`cli/fs.py`) rooted at `--project-path`; all tool paths are virtual, rooted at `/`. Symlinks are never followed; anything escaping the sandbox returns "non-existent". The trace workflow additionally wraps it in `MemoryOverlayFileSystem` so worker writes are persisted as an artifact diff, not applied to the host filesystem.

## Conventions and gotchas

### Prompt brace interpolation (ADK pitfall)

Agent prompts under `contractor/agents/<agent>/prompts/*.md` are passed through Google ADK's `inject_session_state` before every LLM call. ADK scans for `{...}` blocks via `r'{+[^{}]*}+'` and, if the inner text is a valid Python identifier (or `app:|user:|temp:` prefixed), looks it up in session state — **missing keys raise `KeyError` at runtime**. Use `{var?}` for optional substitution, or a non-identifier form like `{user-id}` (hyphen) for OpenAPI path-parameter examples. Bare `{id}` will crash.

Planner-level prompts use the `<<TOKEN>>` convention instead (e.g. `<<MAX_SUBTASKS>>`) and are substituted with plain `str.replace` to sidestep ADK's interpolator entirely. Follow this pattern when authoring new templated placeholders that aren't session-state-driven.

### Prompt versioning

Each agent dir contains `prompt.yml` (manifest) + `prompts/v*.md`. `active:` selects the default; `load_prompt(agent_name)` returns its text, `load_prompt_with_version(name, version)` returns a specific version. The trace eval pipes `CONTRACTOR_EVAL_TRACE_PROMPT_VERSION` to compare prompt variants — same pattern works for any agent.

### Task templates

YAML under `contractor/tasks/*.yml` has a single top-level `task:` mapping (`name`, `objective`, `instructions`, `output_format`, optional `artifacts`, `skills`, `iterations`, `format`). Rendering uses `str.format(**scope)` where `scope = variables ∪ params ∪ {artifact__<safe_name>: content}` plus a `{artifacts}` YAML dump. So the same brace rules apply inside `.yml` instructions too — keep literal braces non-identifier.

### Metrics + observability

Every `TaskRunnerEvent` plus ADK plugin event (`metrics_tool_call`, `metrics_llm_usage`, …) lands in `<output_dir>/metrics.jsonl` via `cli/metrics.MetricsSink`. The output dir defaults to `<project>/.contractor`. When `USE_LANGFUSE=true`, `contractor/utils/observability.run_context(...)` wraps the workflow so spans inherit workflow-level tags. Initialisation is idempotent; if Langfuse is disabled all observability calls are no-ops, so don't gate them in callers.
