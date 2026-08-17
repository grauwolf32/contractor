# Configuration, observability, and deployment

## 1. Configuration model

Configuration is a typed settings object loaded once and cached for the
process. Production code treats the cached object as a run-wide snapshot even
though the compatibility object is not mechanically frozen. All executable
entry points, agent builders, and tools
MUST resolve shared defaults through this object so a CLI run, evaluation run,
and imported workflow observe the same values.

### 1.1 Source precedence

Settings are case-insensitive and ignore unknown fields. From highest to lowest
precedence, values come from:

1. the process environment;
2. `.env` in the process working directory;
3. the repository-anchored `cli/.env` file;
4. the defaults in this specification.

Both environment files are optional. Loading either file MUST NOT overwrite an
already-set process variable. Programmatic construction may use the logical
field name even for fields with a legacy environment alias.

Because the settings object is cached, changing an environment variable after
its first resolution has no defined effect. Tests or embedding applications
that need a different configuration MUST start a new process or explicitly
clear the settings cache through a test-only mechanism.

### 1.2 Complete settings catalog

The environment spelling for an unaliased field is its uppercase field name.
An em dash in the default column means absent/null.

#### Model gateway and sampling

| Logical field / environment | Default | Meaning |
|---|---:|---|
| `default_model_name` / `DEFAULT_MODEL_NAME` | `lm-studio-qwen3.6` | Default model alias. |
| `default_model_timeout` / `DEFAULT_MODEL_TIMEOUT` | `300` | Request timeout in seconds. |
| `litellm_api_base` / `LITELLM_API_BASE` | — | OpenAI-compatible proxy base URL. |
| `litellm_api_key` / `LITELLM_API_KEY` | — | Proxy credential. |
| `model_temperature` / `MODEL_TEMPERATURE` | — | Global sampling temperature; omission preserves backend default. |
| `model_top_p` / `MODEL_TOP_P` | — | Global nucleus sampling value; omission preserves backend default. |
| `summarization_force_tool_choice` / `SUMMARIZATION_FORCE_TOOL_CHOICE` | `none` | Tool-choice sent after a worker crosses its summarization limit. Empty/null means message-only enforcement. |

The recognized portable values for forced tool choice are `none`, `auto`, and
`required`; an incompatible backend may ignore them. Model construction passes
the selected alias, timeout, and only non-null sampling fields to the gateway.
Outbound function schemas rename a property literally named `$ref` to `ref` to
avoid incompatible model-server tool parsers. Input validation for affected
records accepts the compatible spelling.

#### Tool resource limits

| Logical field | Default | Meaning |
|---|---:|---|
| `http_timeout` | `30.0` | HTTP request timeout in seconds. |
| `http_body_preview_chars` | `2048` | Inline response-body preview limit. |
| `http_history_size` | `20` | Recent response summaries returned to an agent. |
| `http_retry_attempts` | `3` | Bounded HTTP attempt count. |
| `http_retry_base_delay` | `0.5` | Initial retry delay in seconds. |
| `http_retry_max_delay` | `8.0` | Maximum retry delay in seconds. |
| `fs_max_items` | `100` | Default number of filesystem result entries. |
| `fs_max_output` | `50000` | Maximum characters/bytes represented by a filesystem read result. |
| `fs_max_read_lines` | `2000` | Default line ceiling per file read; null disables only this axis. |
| `fs_max_files_per_walk` | `100000` | Hard file-visitation ceiling for filesystem glob/grep walks. |
| `fs_heavy_keep_budget_chars` | `0` | Cumulative retained heavy-result character budget; zero disables budget-based elision. |
| `fs_heavy_keep_last_n` | `0` | Override for retained heavy-result count; zero uses the agent-specific default. |
| `code_max_walk_depth` | `50` | Source-analysis walker depth. |
| `code_max_files_per_walk` | `100000` | Source-analysis walker file ceiling. |
| `graph_max_results` | `200` | Maximum call-graph result count. |
| `graph_max_paths` | `25` | Maximum enumerated graph paths. |
| `graph_max_path_depth` | `30` | Maximum graph path depth. |
| `likec4_validate_timeout` | `120.0` | Architecture-model validation timeout in seconds. |

Where both a line/count and size budget apply, processing stops at the first
limit reached and reports truncation. An individual agent or workflow may use a
stricter explicit value, but omission means the global baseline above.

#### Optional integrations and storage

| Logical field / environment | Default | Meaning |
|---|---:|---|
| `use_langfuse` / `USE_LANGFUSE` | `false` | Enable Langfuse/OpenInference instrumentation. |
| `langfuse_host` / `LANGFUSE_HOST` | — | Observability endpoint. |
| `langfuse_public_key` / `LANGFUSE_PUBLIC_KEY` | — | Observability public key. |
| `langfuse_secret_key` / `LANGFUSE_SECRET_KEY` | — | Observability secret key. |
| `target_url` / `CONTRACTOR_TARGET_URL` | — | Authorized live target base URL. |
| `proxy` / `CONTRACTOR_PROXY` | — | Optional outbound HTTP proxy. |
| `caido_url` / `CAIDO_URL` | — | Caido GraphQL/API endpoint. |
| `caido_auth_token` / `CAIDO_AUTH_TOKEN` | — | Caido credential. |
| `gitlab_private_token` / `GITLAB_PRIVATE_TOKEN` | — | GitLab private-token authentication. |
| `gitlab_oauth_token` / `GITLAB_OAUTH_TOKEN` | — | GitLab OAuth authentication. |
| `ci_job_token` / `CI_JOB_TOKEN` | — | GitLab CI job-token authentication. |
| `artifacts_dir` / `CONTRACTOR_ARTIFACTS_DIR` | — | Base directory below which the CLI creates per-project stores. |
| `rag_db_dsn` / `RAG_DB_DSN` | — | Optional pgvector database connection string. |
| `rag_embedding_model` / `RAG_EMBEDDING_MODEL` | `lm-studio-embed` | Embedding model gateway alias. |
| `rag_embedding_dim` / `RAG_EMBEDDING_DIM` | `1024` | Embedding/vector column dimension. |

An absent live target disables network exploitability rather than inventing a
target. An absent RAG DSN selects the dependency-free keyword ranker. An absent
proxy integration makes proxy-specific tools report unavailable in their
standard tool envelope.

### 1.3 Workflow configuration files

Every configurable workflow has a sibling YAML document read by its assembler.
The common schema is:

```yaml
budgets:
  <name>: <scalar>
tasks:
  <task-key>:
    <override-name>: <scalar>
agents:
  <agent-key>:
    <override-name>: <scalar>
observations:
  enabled: <boolean>
  track_tools: <boolean>
  tracked_tools: <null-or-list-of-tool-names>
  include_tool_errors: <boolean>
  track_skills: <boolean>
  track_files: <boolean>
  malformed_only: <boolean>
  track_file_paths: <boolean>
  track_coverage_gap: <boolean>
  track_memories: <boolean>
  in_record: <boolean>
  in_result: <boolean>
```

Task entries accept exactly `iterations`, `max_attempts`, `max_steps`, and
`timeout_s`. Agent entries accept exactly `output_format` (`json`, `xml`,
`yaml`, or `markdown`), `with_graph_tools`, and `with_code_exec`.

Only fields consumed by the selected workflow have effect. Missing mappings
use code-defined defaults. The observations block is optional and an
all-default block is disabled, preserving behavior for old configurations.
Workflow-specific keys and effective defaults are enumerated with their
algorithms in file 05; a reimplementation MUST reject structurally invalid
values with a configuration error that names the file and field.

## 2. Observability

### 2.1 Initialization

Observability is optional. Initialization is idempotent for the lifetime of the
process:

- When disabled, initialization records that decision and returns without
  importing the optional client.
- When enabled, it installs OpenInference instrumentation for the agent runtime
  and obtains a Langfuse client.
- Import, instrumentation, or client failure is logged once and degrades to
  no-op behavior.
- A failed initialization is still considered complete; the system does not
  repeatedly retry and re-log it during the run.

No public observability operation may raise into workflow execution.

### 2.2 Run context

The top-level `run_context` accepts a span name plus optional user ID, session
ID, tags, and metadata. When enabled and healthy it:

1. opens a current top-level span;
2. attaches those values to the current trace;
3. lets automatically instrumented model, agent, and tool operations become
   children;
4. closes the span with the currently propagating exception state; and
5. flushes the client before returning control to a short-lived CLI process.

Client acquisition, span open/close, trace update, and flush failures are each
caught and logged at an appropriate warning/debug level. The context always
yields exactly once, with either the span or null.

### 2.3 Data handling

The JSONL metrics sink applies the shared sensitive-value redactor before local
persistence. It recursively redacts credential-bearing keys and authentication
headers/cookies while preserving benign counters such as token usage.
Automatically instrumented third-party spans do not necessarily traverse that
sink. Operators MUST therefore treat the observability backend as a trusted
processor, configure its capture/redaction controls separately, and avoid
enabling it for source or targets whose data policy forbids external telemetry.

## 3. Reference runtime image

A deployable Contractor image has these behavioral properties, independent of
the build technology used:

- Application runtime dependencies are resolved in a build stage, excluding
  development-only dependencies.
- The final image contains the Contractor package and CLI but not a colocated
  model proxy.
- It includes a current Node-compatible runtime and the `likec4` command because
  architecture validation invokes it as an external program.
- Its artifact base is `/data/artifacts` by default.
- It runs as non-root UID `1001` in group `0`; `/data` is owned by that identity
  and group permissions mirror owner permissions, supporting arbitrary-UID/
  OpenShift-style policies.
- The source/application directory is not used for mutable artifact data.
- The reference entry point runs the workflow CLI and defaults to help output;
  a job/deployment supplies the actual workflow arguments.

The model gateway is a separately deployed OpenAI-compatible service selected
through `LITELLM_API_BASE` and credentials. The image MUST NOT assume that a
model server is reachable on its own loopback interface in an orchestrated
deployment.

## 4. Model-proxy reference deployment

The supplied proxy configuration maps stable Contractor aliases to external
OpenAI-compatible backends. The stable aliases in the reference file are:

```text
lm-studio-nemotron
lm-studio-openai
lm-studio-qwen3.5
lm-studio-glm
lm-studio-qwen3.5-opus
lm-studio-qwen3.5-hauhau
lm-studio-qwen3.6
lm-studio-qwen3.6-mtp
lm-studio-qwen3.6-27b-mtp
llamacpp-qwen3.6-27b-mtp
llamacpp-qwen3.6-35b-a3b-mtp
llamacpp-qwen3.6-35b-a3b
lm-studio-embed
```

LM Studio aliases target `localhost:1234/v1`; the llama.cpp aliases target
ports `8081`, `8082`, and `8083`. The proxy applies three retries and a
300-second request timeout. Its reference launcher uses a rootless Podman
container, host networking, a configuration bind mount, and proxy master/salt
keys injected into the container environment. The reference script
contains literal `sk-litellm-changeme` and `sk-random-hash-changeme`
placeholders rather than reading caller-provided values; a production launcher
MUST replace that behavior. Its configuration bind mount is not read-only in
the compatibility script and SHOULD be mounted read-only in a hardened
deployment.

These aliases are deployment inventory rather than a guarantee all servers are
simultaneously available. Callers select one alias, and ordinary gateway error
handling applies when its backing server is offline.

## 5. Local llama.cpp serving helper

The local model helper locates GGUF files below `MODELS_DIR` (default
`/var/ai-models`), excluding multimodal projection files. It supports:

- `--list` to list relative model paths;
- `--print <substring>` to resolve and display a shell-safe command only;
- `<substring> [-- extra-args...]` to run the selected model;
- `stop` to terminate the listener on the configured port; and
- `stop --all` to terminate all matching model-server processes.

Selection is case-insensitive substring matching. For a sharded model the
first `00001-of-...` shard is selected; otherwise the largest matching file is
used. No match or a missing executable is a hard, explanatory error. The API
alias defaults to the model directory name.

Reference tunables are:

| Variable | Default |
|---|---:|
| `HOST`, `PORT` | `127.0.0.1`, `8081` |
| `CTX`, `NP`, `NGL` | `128k`, `2`, `99` |
| `CTXCP`, `CACHE_RAM` | `2`, `0` |
| `BATCH`, `UBATCH` | `512`, `512` |
| `TEMP`, `TOP_P`, `TOP_K`, `MIN_P` | `0.4`, `0.9`, `40`, `0.05` |
| `REPEAT`, `PRESENCE` | `1.15`, `0.0` |
| `SPEC` | `auto` |

The launch keeps model layers on the GPU, disables memory mapping, enables
flash attention and Jinja chat templates, and disables the prompt cache to
avoid unbounded host-memory snapshots at long context. `k` and `m` context
suffixes mean binary multiples. `SPEC=auto` enables MTP self-speculative
decoding when the selected path contains `mtp`; `on` and `off` override it.

## 6. Optional pgvector service

The RAG database reference deployment runs `pgvector/pgvector:pg16` with:

| Variable | Default |
|---|---|
| `PGVECTOR_USER` | `contractor` |
| `PGVECTOR_PASSWORD` | `contractor` |
| `PGVECTOR_DB` | `contractor_rag` |
| `PGVECTOR_PORT` | host `5433` mapped to container `5432` |

It persists PostgreSQL data in a named volume, installs the `vector` extension
at first initialization, restarts unless stopped, and declares readiness using
`pg_isready` every five seconds with a five-second timeout and ten attempts.

The management helper selects Docker Compose, `podman-compose`, or Podman
Compose in that order. Commands are:

- `up` (default): start detached and print the `RAG_DB_DSN` value;
- `down`: stop while retaining the volume;
- `nuke`: stop and delete the volume; and
- `dsn`: print only the local PostgreSQL DSN.

`nuke` is destructive and MUST require an explicit command. The configured
embedding dimension must equal both the embedding backend output and the
database vector column dimension.

## 7. Code-execution sandbox image

Exploit agents use a separate image, default tag
`contractor-sandbox:latest`, based on a rolling penetration-testing
distribution. The reference toolset includes:

```text
python, curl, wget, jq, nmap, sqlmap, netcat, gobuster,
requests, httpx, BeautifulSoup, PyJWT, pwntools
```

At execution time the source project is bind-mounted read-only at `/project`,
and an isolated writable scratch directory is available at `/work`. The
container uses host networking so a deliberately configured local target is
reachable. It is kept alive only for the owning agent run and commands are
executed through the sandbox tool protocol in file 06. Cleanup, naming,
timeouts, and output bounds are mandatory even if an alternative image is
used.

The reference build helper requires Podman and accepts an optional image tag.
Building an image does not authorize running exploit workflows against an
unapproved target.

## 8. Deployment acceptance criteria

A replacement deployment is conforming when:

1. all fields above resolve with the documented precedence and defaults;
2. optional integrations fail closed or degrade to their documented fallback;
3. observability failures cannot change a workflow result and successful
   short-lived runs flush spans;
4. the runtime writes persistent state only to configured writable locations
   and can run without root privileges;
5. model aliases, proxy timeout/retry behavior, and embedding dimensions remain
   internally consistent;
6. the pgvector `down` path retains data and only the explicit destructive path
   removes it; and
7. sandbox commands cannot write through the `/project` mount and containers
   are removed after their owning run.
