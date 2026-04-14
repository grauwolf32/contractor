
---

# Contractor

**Contractor** is a CLI tool that uses LLM agents to **generate and improve OpenAPI specs from your codebase**.

Instead of manually writing or updating API docs, Contractor analyzes your project and produces structured OpenAPI artifacts for you.

---

## ✨ What you can do with it

* Generate an OpenAPI spec from an existing codebase
* Improve an existing OpenAPI file using real code context
* Trace how the system thinks and behaves (for debugging and tuning)

---

## 🚀 Quick start

```bash
poetry install
poetry run contractor --help
```

Run your first pipeline:

```bash
poetry run contractor \
  --pipeline build \
  --project-path ./your-project
  --folder-name src-folder
```

It is better to place the project in a separate isolated folder so the agent does not accidentally wander into neighboring projects.

---

## 🧠 How it works (in plain terms)

Contractor runs a **pipeline of AI agents** over your project:

1. It scans your codebase
2. Extracts structure and dependencies
3. Uses LLMs to infer API behavior
4. Produces or updates an OpenAPI spec
5. Saves everything in artifact storage

Each run also records metrics and intermediate results so you can inspect what happened.

---

## 🔌 Pipelines

Think of pipelines as “modes” of operation.

### `build` — generate OpenAPI from code

Use this when you **don’t have an OpenAPI spec yet**.

```bash
contractor \
  --pipeline build \
  --project-path ./project \
  --folder-name src \
  --model lm-studio-qwen3.5
```

---

### `enrich` — improve an existing spec

Use this when you **already have an OpenAPI file** and want to refine it.

```bash
contractor \
  --pipeline enrich \
  --project-path ./project \
  --artifact openapi.yaml \
  --model lm-studio-qwen3.5
```

---

### `trace` — understand what’s happening internally

The trace pipeline inspects how your API behaves based on the OpenAPI schema and project code, then reconstructs execution flows.

```bash
contractor \
  --pipeline trace \
  --project-path ./project \
  --artifact openapi.yaml \
  --model lm-studio-qwen3.5
```

What it’s useful for
🔍 Tracing execution paths from OpenAPI endpoints into your code
🧩 Understanding system behavior across layers (handlers → services → dependencies)
⚠️ Finding potential vulnerabilities, such as:
  - missing validation
  - unsafe input handling
  - unexpected code paths
🛠 Debugging agent output when build/enrich results look incorrect

---

## ⚙️ Configuration

### `.env` setup

Contractor reads configuration from:

```
contractor/cli/.env
```

Start from the example:

```bash
cp contractor/cli/.env.example contractor/cli/.env
```

---

### Typical `.env` values

```env
# Enable tracing (optional)
USE_LANGFUSE=true

# Langfuse (optional)
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=http://localhost:3000

# LiteLLM
LITELLM_API_BASE=http://localhost:4000
LITELLM_API_KEY=sk-litellm-changeme
```

---

## 🤖 Model setup (LiteLLM)

Contractor uses **LiteLLM** to talk to models.

Your `--model` must match a configured alias:

```yaml
model_list:
  - model_name: lm-studio-qwen3.5
    litellm_params:
      model: openai/qwen/qwen3.5-35b-a3b
      api_base: http://localhost:1234/v1
      api_key: lm-studio
```


## 🧾 CLI options (simple version)

| Option           | What it does             |
| ---------------- | ------------------------ |
| `--pipeline`     | build / enrich / trace   |
| `--project-path` | your codebase            |
| `--folder-name`  | optional subfolder scope |
| `--artifact`     | required for enrich      |
| `--model`        | model alias              |
| `--output`       | where results go         |
| `--rm`           | clean temp files         |

---

## 📁 Output

By default, everything goes to:

```
<project>/.contractor
```

You’ll find:

* generated OpenAPI files
* intermediate artifacts
* `metrics.jsonl` (run logs)

---

## 🧪 Example workflow

Generate → improve → trace:

```bash
# 1. Generate spec
contractor --pipeline build --project-path . ...

# 2. Improve it
contractor --pipeline enrich \
  --project-path . \
  --artifact .contractor/openapi.yaml
  ...

# 3. Trace execution
contractor --pipeline trace --project-path . ...
```

---

## 🏗 Project structure (simplified)

```
cli/                 # CLI interface
contractor/
  agents/            # LLM agents
  runners/           # pipeline execution
  tools/             # code + filesystem tools
  utils/             # helpers
deploy/litellm/      # model setup
```
