# V57-004: WorkerResult assembly and the origins of size limits

Date: 2026-09-19. The user approved separating result decoding from assembly,
then explicitly chose to remove only the redundant Audit limit and discuss
the shared size contract separately.

## Change

`_build_runtime_result` remains the common entry point for JSON from the ordinary
finalizer and terminal summarizer. `_decode_model_result` validates input size and
JSON; `_validate_result_fields` validates text, schema and the expected subtask ID.
`_assemble_runtime_result` applies Runtime policy and builds WorkerResult with
request-owned identity and permitted artifact refs.

After the collector/publisher gate, trusted completion passes fresh fields
directly into validation and assembly. It no longer creates a `json.dumps`
wrapper only to decode it with `json.loads`. The ID comes from the request;
the supplied completion object does not assign observations, summarized or
output slot names. Its verified artifact values supplement refs observed in
the current invocation, preserving the published revision's precedence. Shared
assembly preserves reserved bindings, exporter-owned slots, known-secret
checking and the final encoded bound.

Both paths pass freshly validated results into synchronous assembly without an
`await` between them. An existing mutable `WorkerResult` is not treated as
permanent proof that its fields are valid. No unchecked model construction
was introduced.

The ordinary tool-free LLM finalizer, exact-copy comparison, summarizer, budgets,
workspace export, cancellation and terminal State retain their positions in the
pipeline. Exact-copy mismatch is still checked after successful assembly;
decoder errors still return as WorkerFailure, including the existing translation
to summarizer failure codes.

## The only selected change to accepted data

Previously, `json.dumps` used `ensure_ascii=True` and the temporary string's
length was compared against 256 KiB. Trusted completion now checks the size of
the text and the actual result. The previous artificial limit is not reproduced.

Verified example: a string of 44,000 U+007F characters is 44,000 UTF-8 bytes,
but the synthetic wrapper with `subtaskId="0"` occupies 264,032 bytes. The final
WorkerResult without artifacts occupies 44,145 bytes. Previously, the wrapper
caused this typed result to be rejected; it now passes. If the same data arrives
as oversized JSON from the model, the decoder still rejects it.

The actual Audit publisher returns a short, fixed ASCII summary, for example
`Recorded and published results for 1 assigned Audit items.`
The report contents are stored in the published ZIP. This change does not alter
its ordinary result, publication or number of model calls. The task makes no
claim of a measured Run speedup.

## Where 256 KiB came from

Repository history shows several separate uses of this value. The reviewed
tasks, specifications and commit messages contain no calculation or benchmark
justifying exactly 256 KiB. This is a Contractor budget, not an ADK or A2A limit.

| Limit | Introduction / migration |
| --- | --- |
| Python result JSON | `a565c242`, MVP-012, 2026-08-29: MAX_STAGE_RESULT_JSON_BYTES |
| Go Planner result payload | `aacda4b3`, V1-006, 2026-08-30: maxStagePayloadBytes; V17-003 retained the limit when switching to WorkerResult |
| Workspace-exported result | `4d39a692`, V11-010, 2026-09-01: MAX_EXPORTED_RESULT_JSON_BYTES; Runtime began using the shared exporter constant |
| WorkerCompletion in Go/Python | `b939133f`, V17-001, 2026-09-02: a separate bound on the entire completion |
| LLM finalizer input | `6631c606`, V21-001, 2026-09-04: the limit was explicitly included in the task scope and spec14 |
| Synthetic Audit wrapper | `bd36ef6b`, V39-005, 2026-09-07: reuse of the existing decoder; no separate decision to limit this temporary wrapper was found |
| Named Go result limits | `79d30099`, V45-012: separate names for existing constants without changing values |

## Remaining limits

| Data | Current limit | Owner |
| --- | --- | --- |
| Worker result text | 64 KiB UTF-8 | Python/Go contracts |
| Model result JSON | 256 KiB | Runtime model decoder |
| Ordinary finalizer input | 256 KiB JSON | V21 / finalizer |
| Assembled WorkerResult, including after workspace export | 256 KiB JSON | Runtime / exporter; separate Planner check |
| Full WorkerCompletion | 256 KiB JSON | Python and Go wire contract |
| StageResult | 256 KiB | Go Stage contract |
| Outer A2A response/DataPart | 1 MiB | Planner A2A client; this is not the permitted size of the inner completion |

Raising the shared limit requires coordinating Runtime, exporter, Python/Go
WorkerCompletion, Planner and StageResult, including envelope overhead.
V57-004 did not change these values. Large Audit data travels through artifacts:
Audit ZIP has a separate 16 MiB limit, and the general Artifact transport limit
is 64 MiB. These are different contracts and do not increase the permitted
WorkerResult text size.

## Verification

Named new/strengthened regressions, executed commands and results are recorded
in the [evidence](../../tasks/evidence/v57-004.json).
Tests separately cover existing model errors, typed handoff, fresh field
validation, removal of only the synthetic bound, artifact authority, actual
output bounds and the real Audit runner with continuation and verified publication.

Sources: [Runtime](../../runtime/src/contractor_runtime/worker/runtime.py),
[Audit completion](../../runtime/src/contractor_runtime/toolsets/audit_results/completion.py),
[Python result contracts](../../runtime/src/contractor_runtime/contracts/worker.py),
[Python limits](../../runtime/src/contractor_runtime/contracts/base.py),
[Go completion](../../internal/contracts/worker_completion.go),
[Go result limits](../../internal/contracts/result_limits.go),
[exporter](../../runtime/src/contractor_runtime/projectfs/exporter.py).
