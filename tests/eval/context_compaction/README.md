# Context compaction evaluation

This is an offline fixture/scorer gate and an opt-in consumer of existing
externally controlled Managed Eval experiments. It does not publish configurations,
plan members, choose models, create a charge ledger or compute comparison conclusions.
Actual Gateway measurements belong to V417-010; its development trial and infrastructure status are recorded in `docs/research/context-compaction-evaluation.md`.

The 2026-11-28 R5 study stopped after 63 of 120 members and remains retained
with its original assessments. V501-004 corrected API scoring on refreshed main.
The current observer reads the successful `operation.kind=context_compaction`
boundary that the actual OTLP adapter retains, and checks each allocation's
published adapter provenance. A Planner executor with no Worker adapter requires
Planner telemetry provenance and a retained closed Planner invocation span for
that exact Stage. This confirms trace presence; complete Server exporter/drop
metrics are not exposed by that allocation's Runtime report.

Run the deterministic gate from the repository root:

```sh
make test-context-compaction-eval-fixtures
```

The gate checks seven declared variants, three case classes and five repeats
(105 recorded scored samples), deliberate corruptions, private-data boundaries,
public result/assessment shapes, bounded replay, preflight, submission receipts,
unknown usage, cancellation and dispatch limits. These samples verify the scorer
and harness. They do not demonstrate live quality or a Runtime checkpoint.
The Go test resolves all seven authored Workflows through the actual catalog loader.
Network connections are forbidden in the Python suite. OpenAPI scoring runs the
repository's pinned `kin-openapi` validator before checking route constraints.
It rejects malformed structure and unresolved/external references. The validator
is built once in a task-owned temporary directory; each document is limited to
1 MiB and a five-second subprocess allowance. Processes are joined and the binary
is removed on exit. Go and the already downloaded module dependencies are required;
validator builds disable module/toolchain downloads.

The 2026-11-28 contract correction requires fresh evaluator registrations (for
example `context@2`), case uploads, scorer pins and plans. The API case now explicitly
requires `likec4` to be a JSON string with newline-separated dependency edges.
Scoring accepts documented non-200 responses, inherited root security and path-item
parameters. An explicit operation security override still wins, including an empty
anonymous override, which fails the tenant requirement. Optional query fields are
checked against their own route rather than a document-wide list.
Independent checks survive malformed API fields; `dependency_model_text` identifies
an incorrect dependency field type without retaining supplied or private values.
Existing frozen studies retain their original scorer, case and plan digests. Do
not replace their files, post revised scores to their registrations, or combine
old and corrected quality rates. These changes are evaluator corrections and do
not demonstrate an improvement in the configured model.

## Cases and variants

`visible/*.json` are source payloads for a recorded provider or a JSON Artifact.
They are **not** public `EvalCase` recipes: their `inputs` contain source data rather
than Artifact references. The public recipe wraps one exact visible payload as its
`source` input Artifact, maps `source -> source` and `report -> result`, and uses an
ordinary task `{kind, objective, parameters}`. Never upload `private/`.

- Source review: derive the concatenation sink from the supplied Python AST,
  preserve exact location and distinguish an observed defect from an unproven
  exploitability hypothesis.
- API/dependency: preserve operation IDs, tenant authorization, optionality and
  dependency edges. The LikeC4 field is a dependency projection (one edge per
  line); its scorer does not claim to compile a full LikeC4 project.
- Negative memory: preserve denied/allowed observations and their IDs, retrieve
  an earlier receipt when needed, and avoid repeating a settled probe or
  generalizing it to untested inputs.

Visible tasks declare output fields and constraint/hypothesis IDs. Private truth
contains expected answers and later questions; the execution projection allowlists
only `id`, `task`, `inputs`, `requires`, `outputs`. Scorers reject plausible wrong
reasoning, dropped constraints, changed locations/IDs and repeated effects.

`variants.json` freezes payload, truth, scorer/consumer and authored-catalog hashes.
Its `scorerSha256` hashes, in order with NUL separators, `harness.py`,
`openapivalidator/main.go`, repository `go.mod` and `go.sum`. Validator or dependency
changes therefore require new check pins and registrations, just like scorer changes.
Its production comparisons are terminal reasoning off/on, compact reasoning off/on,
and terminal off/terminal handoff, each with exactly two arms. A fourth comparison uses controlled native compaction and observation masking
through the test-only Runtime adapter. Its later question becomes visible only
after a real checkpoint/mask and is scored separately. The complete registered
matrix has 120 members: four comparisons, three cases, five repeats, two arms.
The 105 offline samples cover seven distinct variants and are not live members.
Live Workflows are independent end-to-end samples. Identical-checkpoint replay
is unsupported; the former local file-copy simulation has been removed.

## Preparing a live manifest

Prepare the locked Runtime environment with `cd runtime && uv sync --locked`.
Run the consumer on macOS or Linux. Setup/publication is a separate, explicit
operator action through the existing catalog and Eval public APIs:

1. Author and preload the seven fixture Workflows against the intended configured
   ModelPolicy/Gateway/credential selection. `catalog/` is an authored overlay on
   the e2e catalog, not a production model recommendation. Keep generic production
   instructions free of benchmark answers. Use a new identity if publication
   detects drift. The harness never publishes or changes a model.
2. Upload only the three exact visible JSON payloads as immutable Artifact
   revisions. Construct valid public `EvalCase` recipes with these BlobRefs and
   the mappings above. Register four externally controlled two-arm experiments
   through `POST /v1/projects/{projectId}/eval-experiments`, using existing portable
   plans/member identities and `ExternalRegistration` receipts. Keep at least five
   repeats per case and arm; use the fixed seed/order from the plan. Freeze actual
   registered case hashes, not hashes of the raw source payloads.
3. Register one required external check with evaluator such as `context@1` and
   `implementationSha256` equal to `variants.json`'s `scorerSha256` closure hash. Freeze the
   existing comparison gates: minimum candidate end-to-end pass 1, maximum quality
   drop 0, maximum total-token ratio 1.0. Keep all failed, unsupported and missing
   expected members in the existing server comparison.
4. Freeze the manifest below from the published configuration and registration
   receipts. Use one exclusive producer per frozen experiment; other writers make
   the run confounded. No seed is injected into a model that does not support it.
   An unavailable upstream weights revision is `null`, distinct from an exact
   configured alias and policy/Gateway digests.
5. Supply a pinned evaluator-private, read-only Python observer outside visible
   inputs/catalog. It reads one stdin JSON `{runId, case}` and returns only
   `{mutations, effects, usefulAfterCheckpoint, retrieved, newProbes}`: nonnegative
   counts and lists of opaque IDs (at most 1024 IDs, each at most 128 characters).
   Establish these observations independently from execution evidence/environment,
   not the model's claims. Do not return prompts, reasoning, truth or credentials.
   The subprocess has a separate at-most-30-second timeout and 1-MiB output bound;
   its process group is stopped and joined. Useful post-checkpoint work is required
   for compact/handoff, not for a terminal-only answer. If a tiny pilot completes
   without a checkpoint, record that limitation; do not fabricate continuation.

The closed `context-compaction-live/v1` manifest contains:

| Field | Frozen value |
| --- | --- |
| `id`, `repetitions`, `seed`, `thresholds` | Experiment label, 5–100 repeats, seed 417, gates above |
| `fixtureSha256`, `observerSha256` | Exact `variants.json` and observer bytes |
| `runtime` | `{adk, lockSha256, build}`; `adk` equals the exact `google-adk` version in the current Runtime lock; exact lock bytes and Runtime build digest |
| `budgets` | Positive integer `wallMs`, `memberMs`, `maxCalls`, `maxTokens`, `maxToolCalls`, `memberCalls`, `memberTokens`, `memberToolCalls` |
| `variants` | Seven entries keyed by variant ID, each with `workflow`, `closureSha256`, `models`, `capabilitiesSha256`, `instructionsSha256`, `skillsSha256`, `reasoningAvailable`, `runtime` |
| `models` per variant | Configured provenance entries `{policyRef, alias, gatewaySha256, policySha256, weightsRevision}`; include the same policy set for main/auxiliary/Planner/retry use, without inventing another role enum |
| `experiments` | Four entries `{id, portableId, planSha256, setupSha256, variants, casePins, memberOrder}`; `casePins` maps the three case IDs to registered digests; `memberOrder` copies frozen existing IDs |
| `resources` | At most 2048 exact GET response pins `{path, sha256}` for `/v1/workflows/...` and `/v1/configurations/...`; include every Workflow and effective closure resource |

Hashes have the `sha256:` prefix. `closureSha256` is the registered member binding digest. File pins hash exact bytes. GET/setup pins hash
UTF-8 JSON with sorted keys, compact separators, no nonfinite numbers. Freeze
capabilities, generic instructions, Skills, models, reasoning availability and
Runtime identically across arms; intended differences are the authored context
strategy/reasoning switch/handoff. Record unavailable reasoning explicitly.
Capability/instruction/Skill equality is an operator attestation backed by the
immutable execution binding. Runtime build identity comes from the operator deployment record; it cannot be
discovered through the current Eval API. Verify all eligible Runtime processes
against that pinned build before a live trial. Model aliases/policy and Gateway digests are checked
against the exact published resources and every effective Worker/Planner selection
in each Workflow, including summarizer/compactor policies, failure/interruption
escalation and terminal handoff. Every actual route must belong to the common
declared policy set; an arm may use a subset. Worker routes must name an explicit
Gateway in the Workflow, since a deployment-default route cannot be verified through
that public projection. Unsupported missing route evidence fails before dispatch.
This is not a claim
that the harness can discover upstream weights from an alias.

Use conservative per-member ceilings from the frozen execution closure, including
all Worker/Planner calls, terminal summarization, compaction, finalization and retry
limits. Global budgets are shared across all arms. Before each sequential member
dispatch the consumer reserves a complete bounded task. Missing cumulative usage
stops further dispatch. Existing cooperative token limits can observe an in-flight
member's overshoot; this consumer is not a new hard per-token limiter. Increasing
a timeout requires a new frozen equal-budget experiment, not changing the model.

`test_harness.py:complete_manifest` is a complete fictional offline example;
its fake IDs/digests are not suitable for a live server. Invalid, missing, duplicate,
oversized or confounded manifests fail before submission. Every read-only resource,
registered matrix, scorer and Workflow selection is checked before the first write.

The preflight also reads the native `/pairs` projection before any submission.
`pins-unverified` blocks the entire live matrix, including when a local manifest
contains matching attributed hashes. Configuration hashes do not manufacture
observed source/task equality. The 2026-11-16 study exposed missing evidence
in external registrations. Completed
[V466-021](../../../tasks/done/v466-021-observe-external-eval-input-task-pins.yml)
observes those dimensions for newly frozen bindings after local validation.
Retained plans stay unchanged; use a new frozen identity, preserve required
equality and consume the native comparison. Nonterminal/quality-incomplete
exclusions are expected before dispatch; provenance must already be known.

## Invocation and retained evidence

Use absolute local paths, a new output filename, and an existing API session token:

```sh
CONTRACTOR_CONTEXT_EVAL_MANIFEST=/absolute/private/manifest.json \
CONTRACTOR_CONTEXT_EVAL_OBSERVER=/absolute/private/observer.py \
CONTRACTOR_CONTEXT_EVAL_API=http://127.0.0.1:8080 \
CONTRACTOR_CONTEXT_EVAL_OUTPUT=/absolute/private/result.json \
  make eval-context-compaction-live
```

Provide `CONTRACTOR_CONTEXT_EVAL_TOKEN` through the environment; do not put it in
the manifest or retained evidence. No redirect or model substitution is performed.
The live target uses locked installed dependencies and is excluded from ordinary
CI, lint, verify and release-verify; the release-graph guard checks that boundary.

Submission uses the existing `eval-{memberId}` key; lost acknowledgement replays
that exact request. Settlement waits for the authoritative execution and released
Run. Result and assessment receipts are explicitly selected through the ordinary
`/selections` endpoint with the current revision. Ambiguous result/assessment/selection
responses repeat the identical key, body and CAS at most three times within the
current transport deadline. A stale CAS fails without overwriting another choice;
resume skips an already selected exact result/assessment pair. There is no new receipt
store. Comparison publication is asynchronous: the consumer observes all four
views within a shared separate ten-second deadline. A view still stale or unavailable
at that deadline remains inconclusive. Cancellation uses the existing Eval command with revision/plan preconditions
and a separate 60-second cleanup deadline. Unknown or failed cleanup is retained.
Resume with the same frozen registrations/keys and a new local output path; do not
replace members or delete an uncertain execution to improve the denominator.

The sibling `.members.jsonl` streams safe per-member records as they settle;
the final JSON retains every expected ID/status, manifest digest and the **server's**
comparison summaries/freshness. Exit 0 means the frozen matrix is fully submitted,
terminal and scored in fresh Server comparisons, with no missing or conflicting
members. Negative and inconclusive conclusions and failed/cancelled executions stay
unchanged: completing a measurement does not assert improvement. Exit 1 means the
evidence is incomplete; invalid preflight/consumer errors exit 2. Missing or stale
comparisons remain inconclusive and incomplete.
Usage comes from common cumulative Eval snapshots, including terminal summarizer
usage once. Run monetary activity and Runtime phase diagnostics are breakdowns,
never added to usage again. Missing tokens or prices remain `null`; there is no
local savings estimate or second comparison engine. Diagnostic-read failure marks
coverage unavailable without replacing an execution result. Keep the private
manifest, observer and truth under evaluator access controls; delete generated
experiments/Projects only through their existing lifecycle after evidence review.

## Controlled Runtime preparation

`controlled_runtime.py` composes the ordinary Runtime factory for the later
live study. Its offline tests exercise the actual ADK Worker, Gateway-handle
cleanup and error callbacks with network access denied. The consumer includes its separate fourth comparison. These offline tests are
not live masking evidence.

Launch it with the ordinary Runtime arguments plus `--questions`,
`--questions-sha256` and `--private-records`. The private, byte-pinned question
document is at most 64 KiB and has only `schemaVersion`
(`context-compaction-probes/v1`) and `questions`, keyed by the three case IDs.
Each question is at most 2048 characters. It contains questions, never answers.
Include the launcher and that exact document in the operator Runtime build
attestation before dispatch.

Only the explicitly named `context-eval-controlled-compact` and
`context-eval-masking` templates activate controls. They require isolated
sessions, no terminal summarizer or Skills, and only the text-artifact tools.
Each probe belongs to one isolated session; a different session is rejected.
The former uses native `compact@1`; the latter uses no native compactor.
Masking reuses `ContextArchive` to identify complete older groups, replaces
only their text observation bodies, and retains the latest group, tool IDs,
artifact receipts and a digest of the replaced response. SDK session events
are untouched. This is a model-request experiment, not Runtime memory savings.

A later question becomes visible only after an actual native checkpoint or
nonempty mask. It is restated on subsequent controlled requests because a
request projection is not a persisted user event. Ordinary templates keep their
native callbacks. This adds no logical task, model loop, usage owner, budget
reset or production resume mechanism. Identical-checkpoint pairing remains
unsupported until demonstrated separately.

Safe per-Run JSONL records contain IDs, counts, timestamps and digests, with a
2 MiB ceiling. The request digest covers the native `model_request_content`
SDK projection **after** the agent callback, at most 4 MiB; it is not raw
provider transport bytes. Native OTLP input capture precedes that callback
and therefore cannot attest the later masking projection. Neither record
contains question text, observations, reasoning or credentials. Optional
recording failures preserve native execution and cleanup; the evaluator must
report missing or oversized evidence as unavailable, never as a zero count.

`observer.py` is an evaluator-side reader of existing OTLP evidence and the
controlled SDK projection records. Set `CONTRACTOR_CONTEXT_EVAL_EVIDENCE` to an
owner-only directory outside Runtime workspaces, with `traces/{runId}/*.json`
(OTLP protobuf JSON projection) and `projections/{runId}.jsonl`. Each Run is
bounded to 4096 spans and 16 MiB of trace evidence. Missing, foreign or truncated
evidence fails collection instead of becoming zero activity. Successful tool
spans must start after an observed checkpoint to count as useful continuation.
Native compaction is identified by its retained successful
`operation.kind=context_compaction` boundary, independently of the configured
alias and the adapter's span-name projection. A delivered `model.phase=compaction`
also identifies the auxiliary call; the current adapter filters that attribute. The terminal summarizer currently emits
no phase, so its separately pinned study alias identifies that checkpoint;
an explicit main/finalization phase never becomes a checkpoint through an alias.
Source receipt observation and native archive retrieval are separate facts; no
HTTP probe capability is available, so absence of repeated network probes is
enforced by capabilities and does not demonstrate a model decision.

Artifact pagination may set the tool response's `truncated` flag while returning
a complete observed page. That is valid evidence. Only the native telemetry
encoder's `{truncated, preview}` envelope makes captured arguments/output
unavailable; regression tests exercise the actual encoder and protobuf form.
