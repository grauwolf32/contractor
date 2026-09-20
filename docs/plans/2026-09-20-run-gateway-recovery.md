# Run admission and gateway recovery

Status: implemented and verified, V65-001 through V65-003. Live demo rollout has not been performed.

The first crAPI campaign exposed two independent problems: Run initialization
publishes running before Stage admission, and dependency failure immediately
allows the next queued Run to fail against the same unavailable model.
LiteLLM logs establish HTTP 400 with `Model unloaded by user or API request.`
and `Model is unloaded.` from LM Studio, wrapped as BadRequestError.

## Contract

After initialization, a Run is pending until it is actually admitted. A started
Run waiting for model recovery is waiting; tool execution remains running.
Terminal outcomes and cancellation keep their existing meanings. Queue residence
must not consume the Stage execution deadline. The configured Stage wall-clock
deadline remains authoritative after admission, including recovery time; token,
model-call and tool budgets remain separate. A request timeout is not that deadline.

Recovery is coordinated by Server across Runtime processes for an exact resolved
model route. A failure closes admission to that route. One probe may test recovery;
queued work is not used up as a series of failing semantic attempts. Recovery
retries a model request inside its existing invocation and does not replay tools.
After the configured automatic recovery allowance, waiting requires an explicit
retry. Cancellation and the overall admitted Stage deadline still take effect.
The Runtime session is retained in memory; durable restoration after Runtime
process loss is outside this task and must never be claimed.

A single normalized classification drives HTTP retries, recovery, Worker failures
and diagnostics. Model unload signatures are narrowly recognized provider
responses backed by captured incident fixtures, not a rule that retries every 400.
Provider bodies/headers and secrets are discarded after classification.

Public API and UI expose pending/waiting, a safe reason and recovery timing.
The contract changes in place during development; no old-schema compatibility
layer is introduced. Tests include real server/runtime processes with a scripted
gateway outage, permanent input errors, independent routes and cancellation.

## Verification (2026-09-21)

- PostgreSQL regression suites passed for RunStore, RunService, Scheduler, App,
  public/private API, Audit controller, eval service, memory, project lifecycle
  and credentials. Recovery arbitration also passed under the Go race detector.
- Runtime contract, allocation, A2A, ADK, gateway, telemetry, completion and
  summarizer tests passed. UI Run/Queue regression group: 63 tests passed;
  The Home active-Run projection also covers pending/waiting and its test passes.
  TypeScript checks and production build passed.
- Real Server/Python Runtime process checks passed for three queued Runs, an
  HTTP 400 unload followed by HTTP 504, exhausted automatic window/manual retry,
  unchanged Stage identity and exactly one finding per Run. A separate process
  check covered cancellation while waiting and permanent HTTP 400.
- The repository-wide Go run is not fully green: instruction-evaluation frozen
  config/digest checks and the Top 10 Audit profile fixture fail against the
  separate Audit configuration changes already present in this working tree.
  Those tests/configuration files were not modified for V65. The model-selection
  probe fixture was updated to supply the new optional recovery context and passes.
- Migrations 64 and 65 were tested in isolated PostgreSQL schemas. The live demo
  database and services remain on their prior release; no real-model campaign
  was submitted and crAPI truth/scoring was not changed.

## Commit isolation (2026-09-21)

The V65 implementation is committed independently of pending Finding facade and
Audit composition changes. Its Go and TypeScript clients are generated from the
Run recovery API change alone. The process gate uses the existing ordinary
Finding fixture with optional skills removed from that temporary test catalog:
skill preparation requires a Runtime allocation and otherwise leaves queued
Runs initializing before the admission scenario can be exercised.

V65-001 through V65-003 land together because Runtime recovery, persisted Run
states, public contracts and their consumers must agree in a buildable commit.
The isolated tree passes the repository-wide Go suite; the earlier failures
above describe the combined working tree, not this commit candidate.

Final commit verification also passed after incorporating the neighboring main
commits: `go test ./...`, both recovery process tests (94.0 s), 64 focused UI
tests, both TypeScript checks and the production UI build. PostgreSQL suites,
413 Runtime tests and recovery race checks passed on the isolated V65 tree.
