# Independent code review — 2026-09-20

Found **11 reproducible findings: one P1, nine P2 and one P3**.
Each is based on reading the current implementation and a separate check.
Previous reports are not evidence of either the presence or absence of defects.
During discovery, source code was not fixed; existing documentation and task changes were preserved.

Initial commit: `ea1bab91a8c3a12f972fbb2188597334e4652ce3`.
During the review, main advanced to `5856beb5c13429f4be54856015ad7e86bd8b573d`,
adding ScanPlan and related contracts. Files containing findings did not change
between those commits. The new module also received a selective code review and
a separate test run. Descriptions and line numbers below refer to the code before corrections.

## Correction status

All CR-01–CR-11 findings are corrected by tasks V60-013–V60-023 with permanent
regression tests. See the [fix results](2026-09-20-independent-review-fix-results.md)
for implementation commits and verification. The original reproductions below
remain historical evidence. Broader readability work remains in V60-025.

## Confirmed defects

### CR-01 · P1 · Runtime confirms release while a filesystem operation continues

Location: [source_analysis/tools.py:213](../../runtime/src/contractor_runtime/toolsets/source_analysis/tools.py#L213),
cancellation handling at line 218, file writes at line 654.

`open_source_archive` launches extraction through `asyncio.to_thread`.
Cancelling the awaiting coroutine does not stop the thread. The handler removes
staging, releases the session lock, and subsequent allocation cleanup considers
the work complete. The still-running extractor calls `mkdir(parents=True)` and
recreates the directory.

Reproduction uses real `AllocationService`, `AdkWorkerRuntime`,
`SourceAnalysisToolsetFactory` and `LocalWorkdirFactory`. The model and Artifact
storage use fixtures; ZIP reading is paused at a specific point. After
`abort → release → confirm_release`, the result is:

```text
slot=idle; workspaceExists=False; extractorFinished=False
forcedExit=[]; reportComplete=True
```

After the real extractor resumes, the directory exists again and contains
`.source-staging-*/src/private.py` with the previous source text. This was checked
with ordinary `local-workdir@1`, which creates no supervisor subprocess. The check
uses Runtime lifecycle directly, without HTTP/mTLS.

Another manifestation: [likec4/tools.py:295](../../runtime/src/contractor_runtime/toolsets/likec4/tools.py#L295)
launches a synchronous validator in the same way. After cancelling validation,
calling `tools.close()` and deleting the workspace, a real test CLI child remains
alive and the thread has not finished. The test terminates and reaps the process
itself. This variant was checked at the tool/session level, separately from the
full AllocationService.

Fix: explicitly track unfinished filesystem operations and child processes.
Cancellation must prevent new operations; cleanup must wait for or terminate
existing ones before confirming release. Filesystem operations can use the
existing `WorkspaceOperationGuard` approach; CLI execution needs a managed
subprocess with terminate/kill/wait and bounded output.

### CR-02 · P2 · CloneWorkerHandle shares its map with the original and creates a race

Location: [passthrough.go:505](../../internal/planner/passthrough.go#L505).

After `result := handle`, `AgentCard` points to the same map.
`json.Unmarshal(encoded, &result.AgentCard)` reuses the non-nil map, so cloning
itself writes into the original object. The helper is called at A2A invocation
and Worker State read boundaries.

Verified: changing `name` and the nested A2A URL in the clone changes the original.
Two concurrent `CloneWorkerHandle(original)` calls without external writes
produce `WARNING: DATA RACE` between Marshal and Unmarshal.
The State reader calls the helper after releasing its mutex.

Fix: decode into a new map, then assign it. A temporary overlay setting
`result.AgentCard = nil` eliminates both reproductions, including under `-race`.
A copy error must not silently fall back to returning the shared map.

### CR-03 · P2 · Placement and Rebind wait for a second connection while holding the first

Locations:

- [placement.go:440](../../internal/controlplane/placement.go#L440), also line 452;
- [binding_service.go:82](../../internal/runtimeconfig/binding_service.go#L82),
  then `validateTargetWith → validateSpecRuntimeCredentials`;
- [runtime_service.go:169](../../internal/credentials/runtime_service.go#L169).

Both operations open a transaction, but nested credential readers use services
created on the shared `pgxpool.Pool`. Production composition uses that same pool.
Placement affects managed LLM and Runtime credentials; Rebind affects
configuration with proxy/telemetry/Caido credentials.

Two independent checks on real PostgreSQL:

| Operation | MaxConns=2 | MaxConns=1 |
| --- | --- | --- |
| Placement with real EncryptedProvider | Allocation created, about 19 ms | Deadline after 1 s, `lookup encrypted credential metadata` |
| Rebind with real RuntimeCredentialService | Success, about 20 ms | `context deadline exceeded` after 1.2 s |

Direct credential reads pass. Connections are released on exit: this is nested
waiting, not a leak. The same resource conflict is possible when a larger pool
is saturated; the concurrent variant was not measured separately.

Fix: pass credential readers bound to the current transaction. Run creation and
Eval preflight already use this approach. Preserve the credential lifecycle
barrier; increasing pool size is not a substitute for the fix.

### CR-04 · P2 · Advertised Run resume after escalation does not work

Location: [resume_store.go:124](../../internal/runstore/resume_store.go#L124).

`ResumableStage` permits resuming an unsuccessful escalated attempt.
`ResumeFailedRun` creates a new attempt with the same `ExecutionConfigVariant`
and `EscalationOrdinal`. The [unique index](../../internal/persistence/migrations/000008_scheduler_escalation.sql#L15)
forbids this combination for the same Run/Stage.

On PostgreSQL, the `base` control resumed successfully. Both
`failed_escalation` and `interrupted_escalation` were initially advertised as
resumable, then failed with `runstore optimistic state conflict`.

Fix: distinguish automatic escalation identity from manual resume with inherited
effective configuration. Coordinate the index and attempt counting:
[stage_finalization.go:333](../../internal/scheduler/stage_finalization.go#L333)
also rejects repeated ordinals. Simply removing the unique index is insufficient.

### CR-05 · P2 · HTTP tool changes the original query even without new parameters

Location: [http/tools.py:1109](../../runtime/src/contractor_runtime/toolsets/http/tools.py#L1109),
re-encoding at line 1116.

`parse_qsl` and `urlencode` unconditionally decode and rebuild the original query.
The real `http_request` callable with `httpx.MockTransport` and no `query`
argument sends:

```text
?q=%FF      → ?q=%EF%BF%BD
?q=%20&flag → ?q=+&flag=
?q=a%2fb    → ?q=a%2Fb
```

The first example changes the value's actual bytes. This corrupts security-test
payloads and URLs whose signatures depend on the original request target.

Fix: preserve the existing raw query. Validate its limits separately;
when adding parameters, encode only the new pairs.

### CR-06 · P2 · A late session response clears CSRF after a new login

Location: [client.ts:293](../../ui/src/api/client.ts#L293), especially line 298.

Single-user scenario: a repeated `/auth/session` request is still pending;
another request receives 401 and `SessionProvider` cancels queries; the user logs
in again; the old session request returns 401 and calls `csrf.clear()`.
React Query discards the cancelled result, but not the API's internal side effect.

A check with real `PublicAPI` and `SessionProvider` confirms that the UI has the
new session while `mutationHeaders()` fails with
`An authenticated CSRF token is required`. This is not a multi-owner hypothesis.

Fix: check the authentication generation before `csrf.replace/clear` in
`getSession`. Similar protection for ordinary 401 responses already exists in
the transport. Passing `AbortSignal` is additionally useful but does not replace
state protection.

### CR-07 · P2 · Coverage remains stale after Audit completion

Location: [coverage-data.ts:20](../../ui/src/routes/projects/audits/coverage-data.ts#L20),
polling disabled at line 47.

The parent fetches Audit roughly once per second; Coverage refreshes every five
seconds. When the Audit becomes terminal in the same round, polling is disabled,
the query key stays the same, and there is no final refetch.

Verified: after active revision 1, completed revision 2 of the same round was
provided; after 5200 ms, `listAuditCoverage` had still been called exactly once
and displayed the old data. Final results do not appear until another refetch.

Fix: guarantee a final projection refresh on terminal transition or bind fetching
to the accepted revision/state. Preserve cross-page consistency checks.
Items/report show a similar pattern, but separate reproductions were not performed.

### CR-08 · P2 · HTTP target editor does not show credentials beyond the first page

Location: [http-target-editor.tsx:133](../../ui/src/routes/projects/http-target-editor.tsx#L133).

`listRuntimeCredentials(api)` requests 50 records. The editor uses only `items`,
ignores `hasMore/nextCursor`, and offers neither pagination nor ID entry.
In a large catalog, the required credential is inaccessible.

Component/API probe: page one contains 50 records and `nextCursor`; the required
credential is on page two. One GET with `limit=50` was made; neither the desired
option nor pagination controls were present.

Fix: use a dedicated paginated picker or gather all pages for this bounded
selector. Use a separate query key for the aggregated list and keep the
currently selected credential visible.

### CR-09 · P2 · Static server rejects valid links generated by the UI

Location: [static-server.mjs:211](../../ui/server/static-server.mjs#L211), route patterns at lines 40–41.

RuntimeConfig permits version `1.0.0+local`. The UI builds the URL using
`encodeURIComponent`, producing `%2B`. The server checks the decoded path for
unsafe components but returns the encoded pathname for route matching.

The real `createStaticServer` handler returns:

```text
/runs/configuration/default/1.0.0        → 200
/runs/configuration/default/1.0.0+local  → 200
/runs/configuration/default/1.0.0%2Blocal → 404
```

Navigation within the SPA may work while direct opening and reload fail.
The same failure was reproduced for `%3A` in a Project ID.

Fix: align safe route-segment decoding with the URL builder. Preserve rejection
of encoded separators and traversal. A shared table of valid identifiers should
be checked against both browser routing and the static server.

### CR-10 · P2 · CLI --force broadens permissions on a private downloaded file

Location: [artifact_commands.go:595](../../internal/cli/artifact_commands.go#L595).

Ordinary file creation respects umask. With `--force`, the temporary file receives
`Chmod(0644)` and replaces the existing file through rename. Reproduction using
the real function under `umask 0077`: an ordinary download creates `0600`;
repeating the download with `--force` changes permissions to `0644`.

Artifact download and Run output are affected. If the parent directory is
accessible to other OS users, previously private data becomes readable to them.

Fix: preserve existing file permissions on replacement; for a new file, do not
broaden the safe permissions from `CreateTemp`. Preserve atomic rename.

### CR-11 · P3 · CLI loses the meaning of the -- separator

Location: [root.go:263](../../internal/cli/root.go#L263), argument return at line 288.

`interspersedFlags` removes `--` and collects positional arguments separately,
but does not separate them from options before the subsequent `flag.Parse`.
Verified: `source push -- -source` is rejected with
`flag provided but not defined: -source` instead of accepting the directory name.

Fix: insert `--` between collected options and positionals; also test ordinary
flags after the resource name and literal `-` for stdin.

## Simplification and readability

These recommendations are separate from confirmed defects; file sizes alone
are not defects.

1. **Make background-operation ownership consistent.** Source/validators use bare
   `to_thread`, while projectfs/code-analysis already have operation-wait mechanisms.
   A small shared operation/subprocess layer would remove the cancellation-semantics
   mismatch demonstrated by CR-01.
2. **Make transaction-bound credential readers explicit in types/APIs.** Currently,
   `resolveCandidate(ctx, db, …)` looks transaction-bound but accesses allocator
   fields with their own pool. Reuse Run creation's transaction-lookup approach
   to prevent the CR-03 class of defects.
3. **Consolidate verified deep-copy helpers for contract structures.** They are
   currently distributed across planner/controlplane/scheduler. Start with
   WorkerHandle; test independence of nested values and copy-error handling.
   Do not replace deep copy with `maps.Clone` for nested JSON structures.
4. **Split Audit detail by its existing sections.**
   `ui/src/routes/projects/audits/detail.tsx` is about 2100 lines and mixes checks,
   reviews, findings, report and polling. Extract projections into hooks with a
   shared terminal-refresh policy, and sections into separate components.
5. **Unify repeated paginated-selector handling.** Credentials, Project inventory
   and Audit collections should all handle `hasMore` explicitly. This prevents
   omissions such as CR-08. Shared fixtures would also help align UI route
   builders with Node route patterns.
6. **Separate HTTP transport from batch orchestration.** `runtime_client.go`
   combines wire validation, endpoint handling, fan-out and cleanup; extracting
   `RuntimeBatchController` would reduce its responsibilities. In Python,
   `control_client.py` and `artifacts.py` duplicate HTTP framing; a shared parser
   with configurable limits would reduce validation drift.
7. **Simplify RuntimeConfig merge without reflection-based assignment.** In
   `internal/runtimeconfig/merge.go`, `any` and `reflect.Value.Set` hide the
   destination/value relationship. A typed generic helper would preserve the
   existing conflict policy while moving type errors to compile time.

## Checks and limits

| Check | Result |
| --- | --- |
| `go test -json ./...` | 2826 pass, 220 skip; no package failures. Counts include subtests. |
| Go + real PostgreSQL: auditservice, auditstore, evalservice, evalstore, runtimeconfig, credentials, persistence/postgres | 326 pass, no skips or failures |
| `runtime/.venv/bin/python -m pytest -W error tests` | 2352 passed, 34 skipped; 61.30 s |
| Full UI Vitest | 62 files, 435 tests passed |
| App and E2E TypeScript; Node static-server tests | Typecheck passed; server tests passed after allowing local sockets |
| ScanPlan and updated contracts after HEAD changed | Both packages pass |
| Focused probes | All described defects reproduced; principal checks repeated independently |

Go regression probes asserting correct behavior predictably FAIL on unfixed
code. Python/UI/CLI reproduction probes assert the presence of the defect and
pass. Their PASS does not mean the defect is fixed.

The first sandboxed Go run could not write build cache or open loopback
listeners. A repeat with access enabled passed. PostgreSQL ran in a separate
temporary container with isolated test schemas. The container was stopped and
removed afterward. No real models or external security targets were called.

Reviewed execution/recovery, placement/planner, Run storage/resume, blobs and
transactions, parts of Audit/Evals, auth/credentials/configuration, Runtime
lifecycle and several toolsets, API/UI/CLI, CI and deployment. This is a broad
subsystem pass, not a claim of line-by-line review of every file.
The full real-browser stack, Podman release gates, live-model evals and production
backup recovery scenarios were not run. No separate production-load performance
measurement was performed.

Probes, raw logs and file hashes are retained locally in
[.local/code-review-2026-09-20](../../.local/code-review-2026-09-20/); summary:
[evidence.json](../../.local/code-review-2026-09-20/evidence.json).
The `.local` directory is outside Git; this report is stored separately from
existing dirty plans/tasks. Overlay files do not modify production source.

Recommended order: CR-01; then CR-02/03/04/10; then CR-05–09 and CR-11.
Perform refactors separately from fixes, preserving behavioral checks.
