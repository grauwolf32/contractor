# Deterministic scan planning

`scan-plan@1` is the model-free scan Planner. It consumes one exact Run
Artifact containing prepared HTTP requests or a target list, creates a bounded
immutable plan, and invokes existing fixed logical `tool@1` Workers. It does
not discover targets, invent scanner arguments, resolve remote schemas or ask a
model to rank candidates. Request preparation remains the separate
[preparation library](31-scan-request-preparation.md).

## Inputs and fixed Workers

A Stage selects `planner: scan-plan@1` and supplies `scanPlan` with:

- `inputArtifact`: a required Stage context Artifact slot;
- `maxInputs`: 1–1,000 prepared requests or nonempty target lines;
- `maxJobs`: 1–100 selected jobs across all Workers;
- `maxTotalSeconds`: 1–86,400 summed configured Worker timeout seconds;
- `tools`: 1–4 entries, each naming a fixed `worker`, its `maxJobs` and
  `maxTotalSeconds`, and any scanner-specific selection inputs.

Every Stage Worker must appear exactly once in `tools`, and each scanner may
have only one Worker. Workers use `tool@1`, `scan@1`, and the `report` result
slot. No Planner model configuration is accepted. All extra tool arguments are
pinned scalar literals in the resolved AgentTemplate, so policy cannot override
template-owned scanner flags. Worker allocation, readiness, leases and placement
remain Scheduler responsibilities. All configured Workers must be prepared before
the Planner runs; a missing scanner binary can therefore fail Stage preparation
before a scan plan exists.

The source media type is either
`application/vnd.contractor.http-requests+json` (RequestSet v1) or
`text/vnd.contractor.target-list`. A target list is bounded UTF-8 text, at most
2 MiB and 1,000 nonempty lines. LF and CRLF are accepted; bare CR and a UTF-8 BOM
are rejected. Empty lines are ignored; other lines retain their exact bytes.
Provenance uses original line numbers, including preceding blank lines.

RequestSet entries retain their neutral HTTP method, URL, headers and body.
Request IDs refer back to all original OpenAPI operation pointers through the
exact source Artifact. Preparation gaps and coverage are copied into the plan.
The planner neither discards those gaps nor treats preparation coverage as scan
coverage.

Tool eligibility is explicit:

| Scanner | Dynamic Worker input | Selection rule |
| --- | --- | --- |
| nuclei | `url` from parameter `target` | Absolute HTTP(S) target; a RequestSet entry must be GET without a body or headers. |
| naabu | `host` from parameter `target` | Valid DNS name or IP address; URL input contributes its hostname. Host names are normalized to lowercase. |
| sqlmap | `request_ref` from Artifact `request` | RequestSet only; nonempty `testParameters` in policy must name available query, body, cookie or header inputs. |
| ffuf | `url` from parameter `target`, `wordlist_ref` from Artifact `wordlist` | Target-only HTTP(S) input with a `FUZZ` marker in path/query; policy names a required exact `wordlistArtifact` context slot. |

SQLMap uses the existing single-request artifact format, with explicit selected
parameter names. The planner checks its request-file representation before job
selection; missing parameters and unsupported serialization are recorded as
skips. JSON selection covers top-level object keys, and form bodies use ordinary
form fields. It does not infer nested schema paths or broaden the set of tested
parameters. The generated single-request Artifact revision belongs to execution
state and is absent from immutable job identity.

For nuclei and ffuf, authenticated requests or requests with other headers are
skipped because their target-only Worker interfaces cannot preserve those inputs.
FFUF never invents a marker or a wordlist. An exact wordlist ArtifactRef becomes
part of the semantic job input.

## Plan v1 and selection

The plan media type is `application/vnd.contractor.scan-plan+json`; the structural
schema is [scan-plan.schema.json](../../api/scan/v1/scan-plan.schema.json).
`BuildPlan(input, policy, bindings, wordlists)` is a pure Go function in
`internal/scanplan`. The caller supplies exact input refs, actual source bytes
and resolved templates from the Run snapshot. `MarshalPlan` emits canonical JSON;
`DecodePlan` rejects unknown, duplicate, missing or null required fields, invalid
identities, inconsistent accounting and oversized input. Cross-field identities,
ordering, budgets and scanner input semantics are enforced by the Go codec in
addition to the structural JSON schema.

The plan contains `schemaVersion: 1`, `id`, `source`, normalized `policy`,
`preparationGaps`, `candidates` and `jobs`. RequestSet sources also contain
`preparationCoverage`. Arrays are present even when empty. Source metadata binds
the exact ArtifactRef, media type and SHA-256 digest of the supplied bytes.

Each candidate records its stable ID, logical Worker, scanner, sorted unique
`sourceIds`, `selection` and diagnostic `code`. A selected candidate has an empty
code and exactly one job. Unselected candidates retain a fixed reason such as
unsupported input, unavailable selected parameter, or an exceeded budget. The
codec also reserves `unavailable` for candidates with an explicit diagnostic;
the pure builder classifies unsupported candidates as `skipped`.

Each selected job contains semantic parameters or SQLMap request data, exact
external input refs, resolved execution configuration, template selector and
digest, logical Worker namespace, and timeout. Physical allocation IDs, Run IDs,
Stage execution IDs and generated request/report Artifact revisions are not job
inputs. Candidate IDs derive from semantic job content; duplicate jobs combine
source provenance. Policy tools and SQLMap parameter names are normalized before
hashing. Candidates and jobs sort by candidate ID.

The input bound applies before deduplication. Candidates outside `maxInputs`
remain visible with `input_budget_exceeded`. Within that bound, deduplicated
candidates are considered in stable candidate-ID order. Selection checks global
job count, per-tool job count, global summed timeout, then per-tool summed timeout.
Rejected candidates remain in the plan with the corresponding reason. A later
candidate may fit a remaining time budget even when an earlier one did not.

The plan ID is SHA-256 of canonical plan content with its own ID and job IDs
blanked. A job ID binds the plan ID and candidate ID. Source revision, template,
wordlist, policy or concrete input changes therefore change the plan/job identity.
There are at most 4,000 candidates, 100 selected jobs and 4 MiB of serialized plan.
These bounds are not guarantees about the number of network requests generated
inside a scanner. Time budgets count configured Worker deadlines; each invocation
is additionally bounded by the remaining Stage deadline.

## Persistence, execution and recovery

The Planner reads through the authorized Run Artifact service. Before dispatch,
it persists canonical plan bytes and any derived SQLMap request artifacts, then
initializes a durable scan session containing the exact plan ref, digest and job
records. Create-only Artifact writes use deterministic bindings. A lost write
acknowledgement can be recovered only when the existing immutable bytes match;
conflicting content is not overwritten.

The Planner invokes selected jobs sequentially through the
ordinary Worker invoker. Each job has a stable subtask ID and its own result
Artifact binding. The journal is fenced by the current durable Scheduler claim.
A job must transition from `pending` to `started` durably before its Worker call.
Only the owner of that claim may mutate the session or complete it.

A new Scheduler claim preserves completed records and converts any previously
`started` job to `unknown`. Only `pending` jobs remain eligible for execution.
Completed, failed, unavailable, incomplete and unknown jobs are not silently
restarted. An ambiguous invocation outcome stops further dispatch in that attempt;
remaining jobs receive an incomplete outcome. This deliberately records a
possible scan without claiming that it finished or automatically repeating it.

Recovery rechecks the exact source snapshot, canonical persisted plan and derived
input Artifact refs. Changed bytes, revisions or ownership fail explicitly.
A durably completed session returns its previously validated aggregate result.
This provides conservative recovery accounting; it does not promise exactly-once
remote scanner execution across an acknowledgement failure.

## Results and coverage

The Stage publishes one required `application/json` aggregate `report`, in a
namespace separate from Workers. It references the exact plan and per-job reports
and accounts for selected, skipped and terminal job outcomes. Aggregate and
per-job report bindings include the Stage execution identity to avoid collisions
between Stages or retries with different semantic work.

The configured `report.from.name` is a prefix of at most 100 bytes. Its actual
binding name is `<prefix>.<first 16 hex characters of SHA-256(stageExecutionId)>`;
the returned result and Workflow output retain that exact reference. Aggregate
JSON contains `schemaVersion: 1`, `plan`, `planId`, `jobs` and `coverage`.
Coverage counts candidates, selected jobs, skipped candidates, unavailable
candidates/jobs, and completed, failed, incomplete and unknown jobs.
`coverage.complete` requires every selected job to complete with no selection or
preparation gaps. The Stage succeeds when at least one job was selected and all
selected jobs completed; it can therefore succeed with partial overall coverage.
Otherwise it fails with `scan_incomplete` while retaining the aggregate report.

Job outcomes distinguish `completed`, `failed`, `unavailable`, `incomplete` and
`unknown`. Scanner absence is `unavailable`; timeouts, truncation, incomplete ffuf
scans and malformed outputs are `incomplete`, while an ordinary nonzero scanner
exit remains `failed`. When `tool_execution_failed` omits the report reference,
the Planner resolves only that job's own deterministic report binding for
classification; a failure whose report binding cannot be resolved remains failed.
A completed invocation is not a claim that a target is secure. Scanner
error reports and interrupted results remain separate from completed coverage;
missing or invalid reports do not become successful scans. The aggregate preserves
preparation gaps and candidate selection reasons through its exact plan reference.
Inspect both selection accounting and execution outcomes before interpreting a
security report.

Prepared requests, plan artifacts and scanner reports may contain credentials,
private examples or target data. They remain ordinary scoped Artifacts; diagnostic
codes and journal records do not copy supplied HTTP bodies or scanner arguments.

Verification uses fabricated local inputs and fake Worker/Artifact adapters:
`go test ./internal/scanplan/... ./internal/planner/... ./internal/scheduler/...`.
These checks do not launch scanners, model calls or active network probes.
