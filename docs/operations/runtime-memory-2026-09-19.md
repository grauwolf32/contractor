# Runtime memory investigation, 2026-09-19

The largest avoidable allocation found in this investigation was workspace
digest serialization. Hashing the complete canonical JSON document created
large temporary copies of every source file. Hashing the same document one
file at a time preserved its digest and substantially reduced peak RSS in a
local replay of an existing Run's source archive.

The follow-up implementation also removes the OpenAI Python SDK from the Runtime
gateway path and dependency set, and shares fixed node-type strings in the symbol
cache. See [Implemented changes](#implemented-changes-and-validation) for the
measurements on the actual updated source and locked environment.

## What the UI measures

The UI's **Observed RSS peak** is the largest sampled resident size of the
entire Runtime process during an allocation. Sampling runs every 15 seconds,
with additional start/end observations. Child processes, including the
Trailmark graph worker, are excluded. Libraries and memory retained from
earlier allocations are included. This is neither the allocation's incremental
memory cost nor a guaranteed capture of brief peaks.

The latest 100 available allocation summaries were from September 7–8, before
the September 19 restarts. In that sample, 92 `check` allocations had:

| Measurement | Median | Range |
| --- | ---: | ---: |
| RSS at allocation start | 213.8 MiB | 105.8–265.1 MiB |
| Observed RSS peak | 265.7 MiB | 192.4–309.1 MiB |
| RSS at allocation end | 225.0 MiB | 147.2–265.1 MiB |

These historical values are not a measurement of the current build under load.
See [Operations performance](performance.md) for the collection contract.

## Baseline and library costs

The initial live agents from release `a98073a1` each used approximately 140 MiB
RSS, 121 MiB PSS and 108 MiB private memory while idle. During the investigation,
another deployment restarted them on `dc4d6f3a`; their subsequent idle RSS was
approximately 148 MiB each. This investigation did not restart those services
or submit live allocations.

Fresh-process import profiling on Python 3.13.14 found the following inclusive
RSS increases. They depend on import order; nested rows must not be added:

| Import | Increase | Includes |
| --- | ---: | --- |
| Runtime CLI | 107.9 MiB | All startup imports below |
| LLM factory | 46.4 MiB | Google model interfaces and OpenAI SDK |
| Google GenAI | 23.6 MiB | Nested within the LLM factory |
| OpenAI SDK | 19.8 MiB | Nested within the LLM factory |
| ADK worker factory | 24.7 MiB | Additional worker, skills and A2A imports |
| Runtime wire contracts | 14.9 MiB | Contracts and Pydantic infrastructure |

Parser capability probing added about 8.2 MiB, mostly mapped native libraries.
The full standalone startup/probe replay ended near 140 MiB RSS. It excludes
the running server's connections and uses an isolated temporary workspace.
The dependency versions inspected were ADK 2.8.0, Pydantic 2.13.5, OpenAI 2.54.0
and Tree-sitter language pack 1.14.3.

An additional test constructed, invoked and finalized 30 small ADK workers
using the real gateway adapter with an in-memory HTTP transport. First use
loaded more modules; RSS then stayed around 171.4–171.6 MiB. This test includes
test-helper imports and excludes source workspaces. It did not reproduce
ongoing growth between allocations, but cannot rule out leaks in other tools
or large real workloads.

## Workspace optimization

The archived source used by a recent `audit-openapi-operation-trace@4` Run
contained 5,906 files, totaling 29,019,349 expanded bytes (27.7 MiB). Its ZIP was
8,071,292 bytes. The replay used the same local storage and overlay mode,
hydrated the archive in a temporary directory, took three snapshots, ran a
bounded `list_symbols` scan, then closed and removed the workspace. No model or external
tool endpoint was called, and source files were never executed.

For the final comparison both processes used the current checkout and virtual
environment. The baseline restored the old digest function inside that process;
the optimized process used the local implementation. RSS figures below are
from runs without `tracemalloc`:

| Measurement | Previous digest | Incremental digest |
| --- | ---: | ---: |
| RSS after hydration and dropping the archive reader | 173.5 MiB | 173.4 MiB |
| RSS after three snapshots | 207.7 MiB | 179.9 MiB |
| RSS after `list_symbols` | 227.7 MiB | 198.7 MiB |
| Process RSS high-water mark across the replay | 361.1 MiB | 205.0 MiB |
| RSS after workspace cleanup and GC | 221.9 MiB | 190.8 MiB |

The reduction was about 156 MiB in the replay's process peak and 29 MiB in RSS
after symbol analysis. Snapshot-only replays showed approximately 361 to
181 MiB process peaks. The parser scan took approximately five seconds in both
cases; this was not a controlled latency benchmark.

Separate allocation tracing confirmed the large transient JSON allocations:
the peak traced Python memory after tracing began before hydration was about
276 MiB with the old digest and 44 MiB with the incremental digest. After
workspace cleanup, only about 0.2 MiB of those traced allocations remained in
either case. RSS did not fall by the same amount, so retained RSS alone is not
evidence that workspace objects remain reachable. Tracing adds its own memory
overhead and its process RSS was excluded from the comparison table.

The implementation retains JCS encoding for each file and the sorted directory
list, and streams their canonical JSON delimiters into SHA-256. Temporary
content serialization now scales with the largest file rather than the entire
workspace, alongside path metadata. Every replay produced the same digest:

```text
sha256:ba47b3b4c0840de944c36e2683f1f67f1e8fe9224269f1ab54d36c9c4b30739d
```

Compatibility tests compare against the original complete JCS document,
including empty input, insertion order, escaping, combining Unicode and
non-BMP characters. Existing workspace, overlay import/export and security
checks also passed: 135 workspace tests and 51 code-analysis tests, with
warnings treated as errors. Ruff lint and formatting checks passed.

## Further opportunities and limits

The concrete fix is in `runtime/src/contractor_runtime/projectfs/storage.py`.
It is local and was not deployed by this investigation. A fresh full allocation
is still needed to measure its effect on the UI's sampled peak. The replay does
not justify promising that every 260 MiB allocation will fall to 205 MiB.

SDK imports and schema metadata dominate the remaining idle baseline. Delaying
the OpenAI import could move roughly 20 MiB out of cold startup, but first model
use would pay that cost again. Google GenAI and OpenAI already enable deferred
Pydantic model building; enabling that option again is not an optimization.
Replacing an SDK or isolating more worker code would require a separate change
and behavior verification.

Further workspace work could avoid whole-tree snapshots for operations that
only need paths or one file. Such a change must retain the existing consistency,
digest and filesystem validation guarantees; it was not included here.

## Follow-up: accounting for the 199 MiB measurement

The 198.7 MiB figure belongs to the isolated workspace/symbol-analysis replay,
with no model invocation or running HTTP server. Its measured RSS increments
were:

| Phase | RSS contribution |
| --- | ---: |
| Python, startup standard library and Runtime imports | 129.6 MiB |
| Test-helper imports used by the replay | 5.2 MiB |
| Hydrated workspace after dropping the compressed ZIP reader | 38.6 MiB |
| Resident pages retained after three temporary snapshots | 6.5 MiB |
| Symbol analysis, parser loading and cache | 18.7 MiB |
| Total, subject to rounding | approximately 199 MiB |

This is accounting by sequential RSS increases, not independent package heap
sizes. Shared dependencies belong to the phase where they were first loaded.
The test helpers are not part of production Runtime. Model first-use imports,
conversation history and real transport connections can add memory beyond this
replay.

An untraced repeat walked the retained workspace and symbol-cache objects,
deduplicating object identities across the source, current and checkpoint trees:

| Retained Python objects | Size |
| --- | ---: |
| Source strings | 30.9 MiB |
| Workspace paths, dictionaries, sets and tree metadata | 1.4 MiB |
| Cache for 36,848 symbols in 4,897 parsed files | 10.7 MiB |

The three overlay trees share source strings; they do not hold three copies of
every file. These object sizes are components of the RSS phase totals, not
additional memory. Allocation rounding, retained free pages, native libraries
and parser buffers explain the remaining portions of those phase increases.
RSS was recorded before object traversal so traversal's own temporary sets and
dictionaries were not included in the replay's reported endpoint.

### Additional optimization candidates

**Share fixed node-type strings.** The cache contained 36,848 distinct string
objects for just 13 node types, using 2,177,868 bytes. An isolated prototype
stored the already-existing `NodeSpec.node_type` string instead of asking the
native node for another string. Node-type storage became 768 bytes and the
whole symbol cache decreased from 10.7 to 8.6 MiB. Both scans produced 36,848
symbols for the same 4,897 files; RSS before object inspection was approximately
197.9 versus 195.5 MiB. This first prototype was applied only inside the profiling
process; sharing node-type strings was subsequently included in the implementation
below. Symbol names contain another approximately
1.1 MiB of duplicate strings, but a deduplication dictionary would consume some
of that potential saving.

**Read immutable source files on demand.** A disk-backed overlay could retain
only edits and a bounded file cache in RAM, and stream files for hashing and
analysis. The directly measured opportunity is the 30.9 MiB of source strings
for this archive; actual RSS savings still require a prototype. This needs an
interface and storage change that preserves snapshots, overlay rollback/export,
digest compatibility and filesystem validation. Merely copying overlay
dictionaries less often would save little because their strings are already
shared.

**Return free allocator pages after allocation cleanup.** In an isolated glibc
experiment, `gc.collect()` did not change the active workspace's RSS. Calling
`malloc_trim(0)` while the workspace and symbol cache remained live reduced RSS
from 197.5 to 194.3 MiB, taking about 7 ms. After closing tools and the workspace,
another trim reduced RSS from 185.4 to 159.3 MiB. This demonstrates approximately
26 MiB of reclaimable pages at that cleanup boundary, not 26 MiB that can be
removed from a live workspace. Such a policy would primarily lower idle RSS
and later allocation baselines; it would not rewrite the completed allocation's
peak. It needs bounded scheduling and repeated-allocation latency checks before
being introduced. It was not enabled on the services. See the GNU-specific
[`malloc_trim` contract](https://www.man7.org/linux/man-pages/man3/malloc_trim.3.html)
and [Python allocator documentation](https://docs.python.org/3.13/c-api/memory.html).

**Reduce SDK residency.** The OpenAI import's approximately 20 MiB is already in
the startup baseline. Lazy import helps idle or tool-only processes but returns
on first model use. Removing that dependency from the gateway path, for example
with a narrower HTTP adapter, is a separate refactor requiring request,
response, retry, timeout and error compatibility tests. No post-invocation
memory saving for that refactor was measured here.

The new profiling script is `breakdown-profile.py`, with `--tools --objects`,
`--tools --trim`, and `--tools --objects --share-node-types` modes. Results are in
`breakdown-symbol-strings.json`, `breakdown-trim.json` and
`breakdown-shared-node-types.json` under the same local evidence directory.
The separate traced scan hit its 10-second work deadline because tracing slowed
it down; its partial symbol inventory was not used for the cache comparison.

## Dependency audit: reducing memory during model execution

The follow-up inspected the installed packages and actual import paths, then
compared isolated implementations. The target is resident memory during work;
merely moving imports until the first model request is insufficient.

### What is in the approximately 130 MiB baseline

One fresh CLI process ended at 129.9 MiB RSS. Assigning each import's exclusive
RSS increment to the importing package gives this approximate accounting:

| Component | MiB |
| --- | ---: |
| OpenAI SDK | 19.4 |
| Google ADK | 12.7 |
| Google GenAI | 11.9 |
| aiohttp | 6.9 |
| FastAPI | 4.9 |
| A2A and Protobuf | 8.4 |
| Initial Pydantic/Pydantic-core infrastructure | 6.8 |
| Contractor Runtime modules and contracts | 13.4 |
| Python, standard library and remaining libraries | 45.6 |

These rows are disjoint import-time increments, subject to rounding and allocator
behavior, not retained heap ownership. In particular, schema objects created
while importing SDK modules are charged to those modules, not to the Pydantic
infrastructure row. Inclusive import costs differ: OpenAI is about 20 MiB and
Google GenAI about 24 MiB when their newly loaded dependencies are included.
The latter includes approximately 9 MiB for aiohttp and its dependencies.

The SDK imports are much broader than Contractor's gateway use:

- OpenAI loads 781 modules at startup, including 734 modules under `openai.types`
  containing 1,807 top-level class definitions. These include image, video,
  batch, evaluation, vector-store and other API schemas. Contractor's client
  calls only non-streaming `chat.completions.create`.
- Google GenAI loads 31 modules; its generated `types.py` has 1,002 top-level
  class definitions. ADK genuinely uses its `Content`, `Part`, function and
  usage/configuration types. Package initialization also loads the Google API
  client. Importing its HTTP-options types eagerly imports aiohttp even though
  Contractor's gateway transport uses HTTPX.
- FastAPI loads through ADK's `auth.auth_schemes`, which imports
  `fastapi.openapi.models`. A2A's route package is another import path. Removing
  an A2A route re-export alone therefore does not eliminate FastAPI. ADK uses
  these schema classes even though Contractor serves Starlette/A2A JSON-RPC.
- A2A includes legacy protocol-conversion code and route helpers. Some conversion
  code also participates in current agent-card serialization, so it cannot be
  classified as wholly unused solely from the advertised protocol version.
- NumPy is installed but not imported into the parent Runtime. Pandas, LiteLLM,
  Anthropic, gRPC, SQLAlchemy, MCP and Google Cloud client modules are also absent
  from the startup module inventory. Removing unloaded packages would not reduce
  this parent process's RSS. Trailmark runs separately when used.

### Cold-start savings are not active-run savings

In fresh processes with cached bytecode, postponing Contractor's OpenAI imports
reduced CLI RSS from 129.5 to 111.4 MiB. Simulating the absence of aiohttp reduced
it to approximately 120.4 MiB; combining both reached 102.3 MiB. Initial runs of
newly copied source had a few MiB of additional compilation/allocator overhead.

The deferred OpenAI import loads again when constructing a model client. An
additional path exists inside ADK: on the first model invocation,
`flows.llm_flows.contents._id_pairing_model_types()` tries importing
`google.adk.labs.openai.OpenAIResponsesLlm` to check optional provider types. That
imports the OpenAI SDK even when Contractor supplies its own model implementation.
Removing the SDK from Contractor's client alone does not prevent this import if
the package remains installed. ADK catches `ImportError` for this optional
provider, so a distribution without the OpenAI package follows a supported
absence path; no ADK monkey patch was needed for this experiment.

The first ADK invocation also imports workflow execution and OAuth support via
`Runner._find_agent_to_run()`. In the traced import order these added approximately
15 MiB, including Authlib and cryptography. Workflow execution is actually used
by the installed ADK runner, so this is not an immediately removable library group.

### Measured active-work prototypes

An isolated copy replaces the OpenAI client with a narrow HTTPX chat-completion
transport, preserving the existing ADK request/response adapter. It implements
bounded retries, status classification, request deadlines, cancellation and
client ownership for the tested path. An import finder simulates package absence
inside that process; no installed package or service was changed.

The same in-memory gateway returned the same result to 30 successive complete
ADK workers. The small-worker experiment excludes source workspaces and live
network connections:

| Configuration | RSS after 30 workers |
| --- | ---: |
| Current SDK client | 169.9 MiB |
| HTTPX prototype, OpenAI still installed and imported by ADK | 154.7 MiB |
| HTTPX prototype, OpenAI package unavailable | 135.1 MiB |
| HTTPX prototype, OpenAI and aiohttp unavailable | 126.4 MiB |

Thus eliminating the OpenAI dependency from both the client and its environment
saved approximately 35 MiB after model execution. Excluding aiohttp added
approximately 9 MiB. Unlike lazy imports, these savings remained after repeated
model use. aiohttp is currently a required dependency of ADK 2.8.0, despite being
optional in GenAI's HTTP implementation; shipping its exclusion requires a
supported dependency/import change rather than blindly uninstalling it.

A second comparison kept the real archived source workspace and symbol cache
alive while creating and invoking an ADK worker through the mock gateway. Both
processes used the incremental workspace digest and identical archive:

| Phase | Current SDK client | HTTPX prototype without OpenAI |
| --- | ---: | ---: |
| Workspace loaded and symbol scan completed | 197.3 MiB | 181.0 MiB |
| ADK worker invoked, workspace/cache still resident | 221.4 MiB | 188.7 MiB |

This demonstrates approximately 33 MiB less resident memory with source analysis
and a model invocation in one process. The 199 MiB measurement from the earlier
section stopped before model execution, so it should not be compared directly
with the second row as if both were the same phase. These remain controlled
replays, not a live Run with its full conversation history and external tools.

The prototype passed 139 existing gateway, ADK runtime, capability, A2A and skill
tests with warnings treated as errors. For the package-absence run, only the
gateway test's client constructor import and retry-sleep patch target were
redirected to the prototype; its behavioral assertions were retained. The test
suite covers tool-call conversion, structured output, token accounting, errors,
cancellation and cleanup. It does not establish complete SDK compatibility or
replace verification of real gateway/proxy behavior before shipping the refactor.

The most useful dependency optimization for active runs is therefore the narrow
gateway client plus removal of OpenAI from the deployed dependency set. GenAI's
unused aiohttp import is a smaller second opportunity. Removing ADK/GenAI wholesale
would require replacing core model/event/tool interfaces and is a much larger
architectural change. The measurements in this subsection used diagnostic source
copies. The subsequent implementation is described below.

The evidence files are `dependencies-baseline.json`, `dependency-probe-*.json`,
`dependency-workload-*.json`, `dependency-active-imports.json`,
`dependency-invoke-stacks.json`, and `dependency-workspace-active-*.json` in the
local evidence directory. `dependency-variant.py` selects the isolated source
copies and optional-package absence simulation. For example:

```sh
runtime/.venv/bin/python .local/runtime-memory-20260919/dependency-variant.py workload.py
runtime/.venv/bin/python .local/runtime-memory-20260919/dependency-variant.py --httpx-gateway --no-openai workload.py
runtime/.venv/bin/python .local/runtime-memory-20260919/dependency-variant.py --httpx-gateway --no-openai dependency-workspace-active.py --tools
```

## Implemented changes and validation

The Runtime source now contains all three changes:

1. Incremental JCS workspace hashing with the same snapshot digests.
2. An allocation-owned HTTPX Chat Completions client and direct JSON-to-ADK
   response projection. `openai`, `jiter` and `tqdm` were removed from the lockfile
   and local environment with `uv sync --locked --offline`.
3. Shared `NodeSpec.node_type` strings in the symbol cache.

The HTTP client retains one initial attempt plus three transport retries,
per-attempt timeouts and the outer request-timeout-plus-60-seconds deadline.
It honors retry headers, including HTTP dates, and cancellation during retry
waits. Final status/error-code classification remains separate from transport
retry decisions. Credentials are cleared on close, and borrowed proxy HTTP
clients remain owned by their adapters. Failed-response buffers are released
before retry waits; errors retain only the classified failure metadata.

The latest measurements use the actual checkout and its synchronized environment,
with no prototype source copies or blocked-import simulations:

| Phase / scenario | Before gateway/cache changes | Implemented |
| --- | ---: | ---: |
| After 30 small ADK workers | 169.9 MiB | 136.9 MiB |
| Real archive and symbol cache, before model use | 197.3 MiB | 178.8 MiB |
| Same archive/cache held during an ADK model invocation | 221.4 MiB | 188.5 MiB |

This is approximately 33 MiB less RSS after model execution in both measured
scenarios. All three archived-workspace snapshots retained the original digest
and 5,906-file count. The active-work replay had zero loaded OpenAI modules after
model use. aiohttp remains installed and loaded: removing an ADK-required
dependency is not part of this implementation. These are local replay results
with an in-memory gateway, not measurements of a deployed live allocation.

Validation completed:

- Full Runtime suite with warnings treated as errors: 1,953 passed, 31 skipped
  because optional external environments/gates were not enabled.
- Ruff checks and formatting checks across all Runtime source and tests passed.
- Worker summarizer matrix checks passed after updating the renamed retry test.
- Existing local-socket HTTP proxy integration checks passed, including model
  routing, authentication, TLS failures and lack of direct fallback.
- New checks cover retry exhaustion, server retry delays, malformed responses,
  cancellation/deadlines during backoff, allocation-client isolation and an
  actual ADK invocation without loading optional provider SDKs.

The source and local environment are updated; no deployed service was restarted
as part of this implementation. Reports are `implemented-workload.json` and
`implemented-workspace-active.json` in the local evidence directory. Reproduce
them from the repository root after synchronizing the Runtime environment:

```sh
runtime/.venv/bin/python .local/runtime-memory-20260919/workload.py
runtime/.venv/bin/python .local/runtime-memory-20260919/dependency-workspace-active.py --tools
```

## Evidence and reproduction

Local scripts, the private source ZIP and JSON measurements are under
`.local/runtime-memory-20260919/` and are intentionally not repository fixtures.
Earlier SDK baseline reports were captured before its removal; reproducing those
exact dependency comparisons requires the pre-change source and locked environment.
The current checkout replays the implemented HTTPX path.
From the repository root, the key comparison can be repeated with:

```sh
runtime/.venv/bin/python .local/runtime-memory-20260919/workspace-profile.py --tools --legacy
runtime/.venv/bin/python .local/runtime-memory-20260919/workspace-profile.py --tools
```

`main-workspace-tools-before.json` and `workspace-tools-after.json` hold the
final comparison. `allocation-history.json` holds the 100 historical summaries;
`main-imports.json`, `main-baseline.json` and `workload-sdk.json` cover startup
and small repeated workers. `workspace-digest-trace-before.json` and
`workspace-digest-trace-after.json` contain the separate allocation traces.

Current RSS/PSS/private pages were read from `/proc/PID/smaps_rollup`; isolated
process high-water marks came from `/proc/PID/status` (`VmHWM`). These differ
from the UI's 15-second observations. Definitions are documented in the
[Linux proc documentation](https://docs.kernel.org/filesystems/proc.html).
Allocation tracing used Python's
[tracemalloc](https://docs.python.org/3.13/library/tracemalloc.html).
