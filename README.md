# Contractor

`contractor` is a CLI tool for generating and enriching OpenAPI specifications from source code using an LLM.

## Capabilities

The following pipelines are currently available:

* `build` — build or update an OpenAPI spec from the project code
* `enrich` — enrich an existing OpenAPI artifact based on the project structure and implementation

The CLI is installed as the `contractor` command.

---

## Requirements

* Python `>=3.10,<3.15`
* Poetry
* a running LiteLLM proxy
* an LLM backend with an OpenAI-compatible API

---

## Before You Start

Before running `contractor`, you need to:

1. Start the LiteLLM proxy
2. Configure it
3. Make sure the required models are available through the backend
4. Install the project dependencies with Poetry

In this repository, LiteLLM is usually started via:

```bash
deploy/litellm/run.sh
```

This is not mandatory. You can start the proxy in any convenient way, as long as:

* LiteLLM is reachable by the application
* the model names in the config match what is passed to `--model`

---

## Example LiteLLM Config

File `litellm_config.yaml`:

```yaml
model_list:
  - model_name: lm-studio-nemotron
    litellm_params:
      model: openai/nvidia/nemotron-3-nano
      api_key: lm-studio
      api_base: http://localhost:1234/v1
      tpm: 100000
      rpm: 10

  - model_name: lm-studio-openai
    litellm_params:
      model: openai/openai/gpt-oss-20b
      api_key: lm-studio
      api_base: http://localhost:1234/v1
      tpm: 100000
      rpm: 10

  - model_name: lm-studio-qwen3.5
    litellm_params:
      model: openai/qwen/qwen3.5-35b-a3b
      api_key: lm-studio
      api_base: http://localhost:1234/v1
      tpm: 100000
      rpm: 10

litellm_settings:
  num_retries: 3
  request_timeout: 300
```

### What matters here

* `model_name` — the name later used in the CLI via `--model`
* `api_base` — the OpenAI-compatible API endpoint
* `request_timeout: 300` — useful for long-running tasks
* `num_retries: 3` — the number of retry attempts on errors

---

## Example of Running LiteLLM Proxy via Podman

```bash
podman run --rm -d \
  -v $(pwd)/litellm_config.yaml:/app/config.yaml \
  -e LITELLM_MASTER_KEY="sk-litellm-changeme" \
  -e LITELLM_SALT_KEY="sk-random-hash-changeme" \
  --network="host" \
  "ghcr.io/berriai/litellm:main-stable" \
  --config /app/config.yaml
```

If the repository already contains a ready-made script, you can use it:

```bash
deploy/litellm/run.sh
```

---

## Installation

```bash
poetry install
python -c "from tree_sitter_language_pack import download_all;download_all();"
```

After installation, the CLI is available as:

```bash
poetry run contractor --help
```

If the Poetry environment is activated, you can simply run:

```bash
contractor --help
```

---

## Usage

### General format

```bash
contractor \
  --pipeline <pipeline-name> \
  --project-path <path-to-project> \
  --folder-name <project-relative-folder> \
  --user-id <user-id> \
  --model <model-name>
```

### Arguments

* `--pipeline` — the pipeline name
  Currently available: `build`, `enrich`

* `--project-path` — path to the project directory

* `--folder-name` — path inside the project that will be used in task templates
  Default: `/`

* `--artifact` — path to an existing OpenAPI file
  Used by pipelines that require an input artifact, such as `enrich`

* `--user-id` — user identifier for the runner
  Default: `cli-user`

* `--model` — model name from the LiteLLM config
  Default: `lm-studio-qwen3.5`

---

## Examples

### Building OpenAPI from project code

```bash
contractor \
  --pipeline build \
  --project-path /path/to/project \
  --folder-name /src \
  --model lm-studio-qwen3.5
```

It is better to place the project in a separate isolated folder so the agent does not accidentally wander into neighboring projects.

### Enriching an existing OpenAPI file

```bash
contractor \
  --pipeline enrich \
  --project-path /path/to/project \
  --folder-name /src \
  --artifact /path/to/openapi.yaml \
  --model lm-studio-qwen3.5
```

---

## Input Parameter Validation

The CLI validates:

* that `--project-path` exists and is a directory
* that `--folder-name` exists inside `--project-path`
* that `--artifact` exists and is a file
* that `--artifact` is only provided for pipelines that require it

---

After that, the new pipeline will automatically appear in `--pipeline`.

---

## Main Project Parts

* `contractor/main.py` — CLI entrypoint
* `contractor/agents/` — agents
* `contractor/runners/` — pipeline runners
* `contractor/tasks/` — YAML task definitions
* `contractor/tools/` — tools for agents
* `contractor/utils/` — utilities

It is important that the corresponding model is actually available through the backend.

---

## Typical Run Scenario

1. Start the backend with the model

2. Start the LiteLLM proxy

3. Install dependencies:

   ```bash
   poetry install
   ```

4. Check the CLI:

   ```bash
   poetry run contractor --help
   ```

5. Run the required pipeline:

   ```bash
   poetry run contractor \
     --pipeline build \
     --project-path /path/to/project \
     --model lm-studio-qwen3.5
   ```

---

## Troubleshooting

### `contractor: command not found`

Use:

```bash
poetry run contractor --help
```

or activate the Poetry virtual environment.

### Model error

Check that:

* the LiteLLM proxy is running
* the name passed in `--model` matches `model_name` in `litellm_config.yaml`
* the backend is actually reachable via `api_base`

### `--artifact` error

For `enrich`, you need to provide an existing file:

```bash
--artifact /path/to/openapi.yaml
```

### Long responses or timeouts

You can:

* increase `request_timeout` in the LiteLLM config
* choose a different model
* limit the analysis scope via `--folder-name`

---

## Development

Install dependencies:

```bash
poetry install
```

Run tests:

```bash
poetry run pytest
```

Checks:

```bash
poetry run ruff check .
poetry run mypy .
```

## Adding a New Pipeline

The list of pipelines is centralized in the pipeline registry in `contractor/main.py`.

To add a new pipeline, it is enough to:

1. implement a function that returns a `TaskRunner`
2. add it to `get_pipelines()`

Example:

```python
def get_pipelines() -> dict[str, PipelineSpec]:
    return {
        "build": PipelineSpec(builder=oas_building_pipeline),
        "enrich": PipelineSpec(builder=oas_enrichment_pipeline, requires_artifact=True),
        "my-new-pipeline": PipelineSpec(builder=my_new_pipeline),
    }
}
```
