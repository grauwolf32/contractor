# ScanTools and deterministic Workers

Plan dated 2026-09-19. Series V55; V54 is reserved for the separate pentest Audit plan.
Current statuses, dependencies and verification are recorded in `tasks/v55-*.yml`.

## First working set

The extensible `scan@1` provides typed operations for nuclei, sqlmap, naabu and,
later, ffuf. At Runtime startup, each binary is independently probed with a
bounded version-command invocation. Capabilities include only available
operations. A missing scanner does not prevent other scanners or Runtime from
starting. Installing binaries and nuclei templates remains an environment concern.

The shared layer manages the process, timeout, output limit, cancellation,
temporary-file cleanup and metrics. Each scanner descriptor owns its probe
command, argument schema, file preparation and result parsing. Adding a scanner
requires registering an adapter and a matching Server descriptor; name-based
branches in the common executor are unnecessary.

The pseudo-agent is an ordinary Worker with `runtime: tool@1`. AgentTemplate pins
the operation from the selected toolset, input bindings, output slot and deadline.
Input data does not select an executable, shell command or tool. The Worker
validates arguments, calls ToolInstance and stores the report through the
existing Artifact API. A2A, Allocation, lease, queue and WorkerCompletion retain
their current responsibilities. This Worker needs no LLM, ModelPolicy, gateway,
model credentials, ADK finalizer or summarizer.

The exact YAML/wire contract for input bindings is established in code and the
specification in V55-002 before implementing V55-003. It must cover the nuclei
`target` parameter without requiring users to create a JSON file, template
constants and exact input artifact refs. Objective or instruction text must not
be implicitly mixed into command arguments. No separate Server dispatcher is
needed: the first scenario uses the existing `passthrough@1`.

Simple scenarios:

1. nuclei: target URL, optional template filters, JSON report.
2. naabu: one host, a bounded set of TCP ports, JSON report.
3. sqlmap: one prepared HTTP request with method, URL, headers, body and selected
   test parameters; the adapter creates a private `-r` file.
4. ffuf: URL containing `FUZZ`, one wordlist from an exact artifact revision,
   rate/time limits and response filters; a report with explicit truncation indicators.

Process completion, detected matches and a conclusion that no vulnerabilities
exist are separate facts. Timeout, an unknown outcome, invalid output or truncated
output cannot become a clean-target conclusion. Redelivery of a completed call
returns its previous result. An unknown outcome does not trigger an automatic
repeat of an active scan.

## Wordlists and requests as artifacts

Use existing UserScope/ProjectScope and exact RunScope forks.
No new storage service or local paths in tool inputs are needed.

- `text/vnd.contractor.wordlist`: UTF-8, one payload per line.
- `text/plain`: accepted for user `.txt` files in the wordlist slot.
- `text/vnd.contractor.target-list`: a target list for a later stage.
- `application/vnd.contractor.http-requests+json`: the future RequestSet.

V55-005 defines limits on size, line count and line length, along with LF/CRLF,
empty-line and final-newline rules. The validator does not trim whitespace,
remove duplicates or interpret payloads as comments. The artifact is materialized
privately for the duration of the call; a large list is not inserted into model
context. The semantic type in the UI helps users select a file, while Runtime
validates its actual contents.

## Sequence

| Task | Outcome | Dependencies |
| --- | --- | --- |
| V55-001 | Extensible ScanToolset: nuclei/sqlmap/naabu, capabilities, lifecycle | — |
| V55-002 | Normative tool Worker, input-binding and report contracts | 001 |
| V55-003 | Server and Runtime without an LLM; nuclei/naabu through passthrough | 002 |
| V55-004 | SQLMap using a complete prepared HTTP request | 003 |
| V55-005 | Wordlist artifacts and ffuf | 003 |
| V55-006 | User-facing Workflows/UI and end-to-end checks for the first set | 004, 005 |
| V55-007 | RequestSet and deterministic OpenAPI conversion | 006 |
| V55-008 | `scan-plan@1`: candidates, budgets, deduplication, persisted plan | 007 |
| V55-009 | Optional LLM ranking of prepared candidates | 008 |
| V55-010 | Katana as a source of targets/RequestSet | 007 |

V55-001–006 form the first set. V55-007–010 are P2, after verifying the simple
scenarios. Later, the LLM selects IDs of already prepared candidates and supplies
a rationale; the Server validates membership, budget bounds and coverage.
OpenAPI must not silently turn missing auth, path parameters or body examples
into arbitrary runnable requests: gaps appear in the preparation report.
Katana integrates through the same scanner registry.

## Integration points

- `internal/config`, `internal/contracts`, `api/v1alpha1`: description and pinning.
- `internal/runtimeconfig`, `internal/controlplane`, `internal/scheduler`:
  Worker placement and settings without a route to a model.
- `runtime/src/contractor_runtime/worker`, `allocation`, `a2a_server.py`:
  Worker lifecycle, deterministic execution and state.
- `runtime/src/contractor_runtime/toolsets/scan`: scanner adapters.
- `internal/planner/passthrough.go`: the first single-step invocation.
- `ui/src/api/artifacts.ts`, artifact and Run-creation forms: list upload.

The changes do not include tool installation, production rollout, bulk scan
execution or a new Audit program. A configured subprocess proxy must not be
silently bypassed: until an adapter supports that routing, it returns an explicit
error. Examples must make proxy compatibility clear.
