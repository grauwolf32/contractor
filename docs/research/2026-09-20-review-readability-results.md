# Review readability follow-up — V60-025

This follow-up separates existing responsibilities and removes an unsafe assignment
mechanism while preserving behavior. It follows the eleven corrective tasks and
[V60-024 verification](2026-09-20-independent-review-fix-results.md). The baseline is
`70b4277f`. During verification, the independent Scan Planner commit `e6f6c3a2`
landed on the shared branch. Affected Go tests, all-package compilation and the
complete UI checks were repeated on the combined state. Those feature changes
remain outside this implementation commit.

## Changes

| Area | Result | Reason |
| --- | --- | --- |
| Audit detail | `detail.tsx` is now a 229-line route shell, down from 2132 lines. Overview, executions, findings, reviews and report have dedicated modules. | Navigation and parent loading are readable together; each section keeps its existing query and mutation logic nearby. |
| Finding card | `finding-card.tsx` contains the shared card, provenance and review controls. Project findings imports it directly. | A reusable card no longer depends on a route module. Existing CSS imports and the compatibility export remain intact. |
| Runtime lifecycle | `runtime_client.go` contains HTTP/TLS transport and validation (787 lines); `runtime_batch.go` contains orchestration and cleanup (383 lines). Corresponding batch tests move together. | Protocol handling and allocation ownership can be reviewed independently. Public interfaces and call ordering are unchanged. |
| RuntimeConfig merge | A generic selector and destination replace `any` values and `reflect.Value.Set`; temporary untyped merge collections are removed. | The compiler checks assignment types. Deep atomic equality, explicit clears and deterministic conflict selection stay intact. |

The Audit extraction retains component boundaries, hook order, query keys, final
projection refresh, pagination, explicit revision pins, keyring/draft lifetimes and
module-level lazy preview components. Shared artifact links retain exact revision
encoding. It does not introduce a generic component or state-management framework.

## Verification

Moved declarations were inspected directly, independently of prior review reports:

- Go token comparison: all 58 production declarations and 39 test declarations
  retain their signatures and bodies; imports, file location and explanatory
  comments account for the transport/controller split.
- TypeScript comparison: all 20 function bodies are identical. A separate parameter
  and initializer comparison covers 24 functions/module constants; the only type
  spelling change in the shared artifact component replaces an equivalent alias.
- RuntimeConfig differential probe: 18,000 deterministic input/permutation cases
  match the baseline implementation's returned spec and structured error under
  the race detector. Inputs include omitted/cleared values, repeated refs, nested
  telemetry pointers, proxy target slices and multiple simultaneous conflicts.

Final checks passed:

| Check | Result |
| --- | --- |
| Go with disposable PostgreSQL and `-race` | Control Plane, RuntimeConfig, Scheduler and app packages passed; RuntimeConfig was repeated after the final helper cleanup |
| Go/Python mTLS lifecycle | Both finalize and abort cases passed |
| All Go packages | Compiled successfully; affected packages passed `go vet` |
| Full UI suite | 452 tests in 65 files passed on the combined Scan Planner/refactor state |
| UI TypeScript, ESLint/Prettier and production build | Passed; Vite still reports large generated chunks |
| Whitespace and moved declarations | Passed |

The commands, completed test results and implementation hash are recorded in
`tasks/evidence/v60-025.json`. Raw local logs and replayable comparison probes are
retained in `.local/review-readability-2026-09-20/`.

## Deliberate scope boundaries

Python ownership helpers were reviewed and kept separate. `WorkspaceOperationGuard`
fences a workspace and retains its owner after returning cancellation to its caller;
LikeC4 waits for its syscall before releasing the outer session lock; subprocess
cleanup joins a task that kills and reaps the process group. Their similar shield
loops express different cancellation and exception contracts. The shared bounded
subprocess runner already provides the useful common behavior.

Control and Artifact HTTP framing also have different contracts: strict versus
permissive numeric/header parsing, handling of simultaneous Transfer-Encoding and
Content-Length, Latin-1 versus ASCII header values, whole-exchange versus per-phase
timeouts, body limits and cancellation during connection close. Consolidating them
requires a separate contract decision and transport regressions. It is outside this
behavior-preserving refactor.
