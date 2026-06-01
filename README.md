<p align="center">
  <img src="docs/logo.jpg" alt="Contractor" width="280" />
</p>

<h1 align="center">Contractor</h1>

<p align="center">
  <strong>AI-powered OpenAPI generation, enrichment, and vulnerability tracing from your codebase</strong>
</p>

<p align="center">
  <code>oas_build</code> &nbsp;·&nbsp; <code>oas_update</code> &nbsp;·&nbsp; <code>trace</code> &nbsp;·&nbsp; <code>vuln-scan</code> &nbsp;·&nbsp; <code>exploit</code> &nbsp;·&nbsp; <code>router</code>
</p>

---

Contractor is a CLI tool that drives a pipeline of LLM agents over your project's source code to **generate**, **improve**, and **trace** OpenAPI specifications — no manual writing required.

## Quick Start

```bash
poetry install
poetry run contractor --help
```

```bash
poetry run contractor \
  --workflow oas_build \
  --project-path ./your-project \
  --folder-name src
```

> Place the target project in a separate isolated folder so the agent does not accidentally wander into neighboring projects.

## Workflows

`--workflow` selects the agent chain to run. The full registry lives in
[`contractor/workflows/__init__.py`](contractor/workflows/__init__.py).

**OpenAPI / architecture**

| Workflow | Purpose | Key flags |
|----------|---------|-----------|
| **`oas_build`** | Generate an OpenAPI spec from code | `--project-path`, `--folder-name` |
| **`oas_update`** | Improve/enrich an existing spec using code context | `--artifact openapi.yaml` |
| **`likec4`** | Generate a LikeC4 architecture model | `--project-path` |

**Trace & annotate**

| Workflow | Purpose | Key flags |
|----------|---------|-----------|
| **`trace`** | Planner-driven vulnerability trace, per-operation overlay FS | `--artifact openapi.yaml` |
| **`trace-direct`** | Single-agent trace, one pass per operation (no planner) | `--artifact openapi.yaml` |
| **`trace-graph`** | `trace-direct` + call-graph (trailmark) tools | `--artifact openapi.yaml` |
| **`trace-graph-pathpar`** | Path-parallel variant of `trace-graph` | `--artifact openapi.yaml` |
| **`trace-verify`** | Per-finding static verifier over trace results | `--artifact openapi.yaml` |

**Vulnerability detection**

| Workflow | Purpose | Key flags |
|----------|---------|-----------|
| **`vuln-scan`** | Breadth-first vulnerability scan over source | `--project-path` |
| **`vuln-scan-fast`** | High-recall scan → dedup → trace-confirm → exploit | `--project-path` |
| **`vuln-scan-trace`** | BFS discovery → DFS confirmation | `--project-path` |
| **`vuln-assess`** | Discovery → OAS → trace → exploit | `--project-path` |
| **`exploit`** | Per-finding exploitability assessment vs. a live target | `--artifact` |

**Prompt-driven**

| Workflow | Purpose | Key flags |
|----------|---------|-----------|
| **`router`** | Dispatches a prompt to a specialised sub-agent (interactive if `--prompt` omitted) | `--prompt "..."` |

> `--artifact` is optional for `oas_update` / `trace*` / `exploit` / `vuln-*`: if omitted,
> the workflow reuses the seed already in the per-project artifact store from a prior run.

### Typical workflow

```bash
# 1. Generate spec from code
contractor --workflow oas_build --project-path . --folder-name src --model lm-studio-qwen3.6

# 2. Enrich it with deeper analysis
contractor --workflow oas_update --project-path . --artifact .contractor/openapi.yaml --model lm-studio-qwen3.6

# 3. Trace execution paths & find vulnerabilities
contractor --workflow trace --project-path . --artifact .contractor/openapi.yaml --model lm-studio-qwen3.6
```

## How It Works

Contractor runs a **pipeline of AI agents** coordinated by a planning agent:

```
Source Code  ──►  Planner Agent  ──►  Worker Agents  ──►  OpenAPI Artifacts
                     │                    │
                     ▼                    ▼
               Subtask decomposition   Code analysis, API inference,
               Progress tracking       Spec generation, Vuln tracing
```

1. **Scan** — reads your codebase through a sandboxed filesystem
2. **Plan** — a planner agent decomposes the work into subtasks
3. **Execute** — specialized worker agents process each subtask
4. **Produce** — results are assembled into OpenAPI specs and saved as artifacts

Each run records metrics and intermediate results to `<project>/.contractor/metrics.jsonl`.

## Configuration

### Environment

Configuration is read from `cli/.env` via `pydantic-settings`; the full schema is
[`contractor/utils/settings.py`](contractor/utils/settings.py) (`Settings`).

```env
# LiteLLM proxy (required)
LITELLM_PROXY_API_BASE=http://localhost:4000
LITELLM_PROXY_API_KEY=sk-litellm-changeme
USE_LITELLM_PROXY=True

# Langfuse observability (optional)
USE_LANGFUSE=False
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=http://localhost:3000

# Caido proxy (optional; needed for exploit proof chains)
CAIDO_URL=http://127.0.0.1:8080
CAIDO_AUTH_TOKEN=...
```

### Model Setup (LiteLLM)

`--model` is an alias from `deploy/litellm/litellm_config.yaml`, resolved by a LiteLLM proxy. Start the proxy before running any workflow:

```bash
cd deploy/litellm && bash run.sh   # requires Podman
```

Example alias config:

```yaml
model_list:
  - model_name: lm-studio-qwen3.6
    litellm_params:
      model: openai/qwen/qwen3.6-35b-a3b
      api_base: http://localhost:1234/v1
      api_key: lm-studio
```

## CLI Reference

| Option | Description |
|--------|-------------|
| `--workflow` | Workflow mode (see tables above; default `oas_build`) |
| `--project-path` | Path to target codebase (**required**) |
| `--folder-name` | Project-relative folder scope used inside task templates (default `/`) |
| `--artifact` | Input artifact (optional seed for `oas_update` / `trace*` / `exploit` / `vuln-*`) |
| `--prompt` | Prompt for `router` (interactive if omitted; required with `--no-ui`) |
| `--model` | LiteLLM model alias (default from `Settings.default_model_name`) |
| `--timeout` | Per-request model timeout in seconds (default `300`) |
| `--user-id` | User id for the ADK session/artifact namespace (default `cli-user`) |
| `-o, --output` | Output directory (default `<project>/.contractor`) |
| `--rm` | Remove previous artifacts before the run (mutually exclusive with `--resume`) |
| `--resume` | Resume from `checkpoint.json`, skipping completed tasks |
| `--no-ui` | Disable the live UI (CI mode) |

## Project Structure

```
cli/                          CLI entrypoint, sandboxed FS, live UI, metrics sink
contractor/
  workflows/                  One folder per workflow (assembler + config.yaml)
  agents/                     LLM agents (planner, swe, trace, router, ...)
  runners/                    Task-runner execution engine
  tools/                      Sandboxed tools (fs, code, memory, openapi, http, vuln)
  tasks/                      YAML task templates
  skills/                     Markdown reference bundles injected per task
  callbacks/                  Token usage, guardrails, rate limits
  utils/                      Settings, observability, helpers
deploy/litellm/               LiteLLM proxy config
docs/                         Architecture docs & diagrams
```

## Documentation

- [docs/README.md](docs/README.md) — deep dive on planner/worker internals, the streamline subtask state machine, and the memory/artifact contract.
- [docs/tuning.md](docs/tuning.md) — full inventory of tunable knobs (CLI flags, settings, per-workflow budgets, callbacks, tool caps) + a tuning playbook.
- [docs/eval-tuning.md](docs/eval-tuning.md) — parameter-sweep configs runnable through the eval suite.
- [docs/insights-parallel-vuln-pipelines.md](docs/insights-parallel-vuln-pipelines.md) — notes on path-level parallelism and vulnerability detection.
