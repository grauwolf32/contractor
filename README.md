<p align="center">
  <img src="docs/logo.jpg" alt="Contractor" width="280" />
</p>

<h1 align="center">Contractor</h1>

<p align="center">
  <strong>AI-powered OpenAPI generation, enrichment, and vulnerability tracing from your codebase</strong>
</p>

<p align="center">
  <code>build</code> &nbsp;·&nbsp; <code>enrich</code> &nbsp;·&nbsp; <code>trace</code> &nbsp;·&nbsp; <code>router</code>
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
  --pipeline build \
  --project-path ./your-project \
  --folder-name src
```

> Place the target project in a separate isolated folder so the agent does not accidentally wander into neighboring projects.

## Pipelines

| Pipeline | Purpose | Key flags |
|----------|---------|-----------|
| **`build`** | Generate an OpenAPI spec from code | `--project-path`, `--folder-name` |
| **`enrich`** | Improve an existing spec using code context | `--artifact openapi.yaml` |
| **`trace`** | Trace execution paths & find vulnerabilities | `--artifact openapi.yaml` |
| **`trace-graph`** | Graph-based trace with cross-layer analysis | `--artifact openapi.yaml` |
| **`trace-direct`** | Single-pass trace (no planner) | `--artifact openapi.yaml` |
| **`trace-verify`** | Verify trace results | `--artifact openapi.yaml` |
| **`likec4`** | Generate LikeC4 architecture model | `--project-path` |
| **`router`** | Prompt-driven agent (interactive if `--prompt` omitted) | `--prompt "..."` |

### Typical workflow

```bash
# 1. Generate spec from code
contractor --pipeline build --project-path . --folder-name src --model lm-studio-qwen3.6

# 2. Enrich it with deeper analysis
contractor --pipeline enrich --project-path . --artifact .contractor/openapi.yaml --model lm-studio-qwen3.6

# 3. Trace execution paths & find vulnerabilities
contractor --pipeline trace --project-path . --artifact .contractor/openapi.yaml --model lm-studio-qwen3.6
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

```bash
cp cli/.env.example cli/.env
```

```env
# LiteLLM proxy (required)
LITELLM_API_BASE=http://localhost:4000
LITELLM_API_KEY=sk-litellm-changeme

# Langfuse observability (optional)
USE_LANGFUSE=true
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=http://localhost:3000
```

### Model Setup (LiteLLM)

`--model` is an alias from `deploy/litellm/litellm_config.yaml`, resolved by a LiteLLM proxy. Start the proxy before running any pipeline:

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
| `--pipeline` | Pipeline mode (see table above) |
| `--project-path` | Path to target codebase |
| `--folder-name` | Subfolder scope within project |
| `--artifact` | Input artifact (required for `enrich`, `trace*`) |
| `--prompt` | Prompt for `router` (interactive if omitted) |
| `--model` | LiteLLM model alias |
| `--output` | Output directory (default: `<project>/.contractor`) |
| `--rm` | Clean temp files after run |

## Project Structure

```
cli/                          CLI entrypoint & pipeline assemblers
contractor/
  agents/                     LLM agents (planner, swe, trace, router, ...)
  runners/                    Pipeline execution engine
  tools/                      Sandboxed tools (fs, code, memory, openapi, http, vuln)
  tasks/                      YAML task templates
  skills/                     Markdown reference bundles injected per task
  callbacks/                  Token usage, guardrails, rate limits
  utils/                      Settings, observability, helpers
deploy/litellm/               LiteLLM proxy config
docs/                         Architecture docs & diagrams
```

## Documentation

See [docs/README.md](docs/README.md) for the deep dive on planner/worker internals, the streamline subtask state machine, and the memory/artifact contract.
