# 29 — Deterministic tool Workers

Status: implemented by V55-003; V55-002 defines the contract.

## Template and arguments

`tool@1` invokes one registered callable without a model. It uses ordinary
Allocation, A2A, WorkerState, Artifact API and WorkerCompletion contracts.
The initial version accepts exactly one selected tool, no Agent Skills,
summarizer, instructions, ModelPolicy, custom completion contract or project
workspace. `local-workdir@1` owns disposable scanner files. ADK requirements
remain unchanged; `execution` is forbidden for other runtimes.

```yaml
apiVersion: contractor/v1alpha1
kind: AgentTemplate
metadata:
  name: nuclei-scan
  version: "1"
spec:
  description: Scan one supplied target with installed nuclei templates.
  runtime: tool@1
  sandboxProfile: local-workdir@1
  toolsets:
    - ref: scan@1
      tools: [scan_nuclei]
  execution:
    tool: scan_nuclei
    arguments:
      url: {source: parameter, name: target}
      rate_limit: {source: literal, value: 10}
    resultArtifact: report
    timeoutSeconds: 300
```

`execution` has exactly these four required fields. `arguments` is an object
with at most 32 keys. Argument names match `[A-Za-z_][A-Za-z0-9_]{0,63}`.
Each binding selects exactly one source:

| source | Required payload | Resolved value |
| --- | --- | --- |
| parameter | `name`, 1–128 non-whitespace characters | unchanged StageContentRequest.parameters string |
| artifact | `name`, same bounds | exact ArtifactRef object from StageContentRequest.artifacts |
| literal | `value` | non-null JSON string, boolean or finite number |

`name` is forbidden for literal; `value` is forbidden for other sources.
Literal strings are at most 8192 UTF-8 bytes. Numbers must be finite and within
the exactly representable integer range when integral (±9007199254740991).
There is no string interpolation or implicit JSON/type coercion. Numeric limits
can be pinned as literals; omitted arguments use callable defaults. Artifacts
remain refs for adapters to resolve, never local paths or model context.

`tool` must equal the one selected exported tool name. `resultArtifact` names
a versionless binding in StageContentRequest.resultArtifacts and is required
for every call; the target must be in the Worker's own namespace. Binding names
with the reserved prefix `tool-invocation.` cannot be output targets.
`timeoutSeconds` is an integer in 1–3600. The Worker enforces it around the tool
call, independent of the tool's own deadline. Bound arguments plus artifact
refs and output binding are limited to 64 KiB of canonical JSON.

Unknown arguments, missing sources, wrong types or required arguments fail
before invocation. Runtime builds a strict input schema from the selected
callable's annotations/defaults at allocation preparation. The callable returns
a JSON object with `status: completed|failed` and optional fixed `errorCode`;
invalid/non-finite or oversized responses fail explicitly. Runtime never
interprets objective/instructions as arguments or chooses a different tool.

Invalid examples include:

```yaml
# Missing parameter name; a literal cannot also name a parameter.
badBindings:
  url: {source: parameter}
  rate_limit: {source: literal, name: rate, value: 10}
# String-to-integer coercion is not permitted by the callable schema.
badTypedInput:
  rate_limit: {source: literal, value: "10"}
# Contradictory runtime selection.
badTemplate:
  runtime: tool@1
  modelPolicy: worker@1
```

## Resolution, placement and wire shape

ResolvedAgentTemplate embeds `execution` unchanged. Its canonical digest uses
the existing sorted toolsets/runtime/sandbox/description projection, omits
instructions/modelPolicy and adds execution. Existing ADK digests are unchanged.
The same absence is preserved in resolved consumer execution config and
AllocationSpec; null or placeholder model objects are not a model-free route.

RuntimeSettings permits absent llmGatewayUrl/token, but AllocationSpec requires
them for ADK and forbids them for tool@1. Tool allocation provenance has no LLM
gateway/credential refs. Template and effective model absence are independently
validated; omission does not relax model validation for ADK.

Workflow/Run/escalation execution patches targeting a tool Worker cannot set a
model policy, gateway or model credential (including explicit clear). Shared
Runtime labels may contain model routes for other Workers; those leaves are
irrelevant to tool@1 and must be projected out before merge, catalog/credential
lookup and adapter materialization. Telemetry and applicable HTTP/proxy adapters
retain their normal precedence and provenance. Merely selecting tool@1 never
requires a model catalog or key. Existing tool/runtime capability matching and
active-check classification still apply; no fallback to ADK is allowed.

Stage-local parameter/artifact/output bindings are checked against the resolved
execution contract when loading a Workflow. The ordinary `passthrough@1`
Planner supplies the first single-call scenarios and needs no model.

## Receipt, report and replay

The execution key is SHA-256 of canonical JSON containing runId,
stageExecutionId, logicalAgentName and subtaskId. The owned namespace stores a
`tool-invocation.<hex-key>` application/json receipt. It is created with
If-None-Match before tool launch. The input digest covers template ref,
resolved arguments and output binding; changes under the same key fail as
`tool_input_conflict`. Objective/instructions do not affect execution identity.
This is delivery deduplication within one StageExecution; an explicitly new
Run/StageExecution is new work.

The receipt starts with schemaVersion 1, inputDigest and phase `started`.
Only its creator may launch. A terminal CAS update records phase `completed`
or `failed`, optional exact report ref and a fixed error code. An existing
started receipt means `tool_outcome_unknown`; Runtime does not rescan. A
write with an ambiguous result also prevents launch. An in-memory completed
delivery returns the same WorkerCompletion; receipt replay on another allocation
publishes fresh local WorkerState with zero new tool/model calls and reuses the
exact report. Receipts contain no target, credentials or raw arguments.

The report is application/json, at most 512 KiB, with schemaVersion 1,
tool, inputDigest, exact inputArtifacts and the unchanged bounded tool response
under `observation`. Publication is create-only to the supplied output binding.
Runtime must not overwrite an existing report or infer its ownership by name.
There is no generic finding publication and no raw unbounded log artifact.

| Event | Completion / recovery |
| --- | --- |
| Tool completes, report and receipt persist | result with report ref, summarized=false |
| Tool reports failed | publish diagnostic report; nonretryable WorkerFailure |
| Invalid input | tool_input_invalid, no process launch |
| Tool exception / invalid output | tool_execution_failed / tool_output_invalid |
| Tool deadline | tool_timeout; await tool cancellation and cleanup |
| User cancel, lease loss, abort | cancel task, join child cleanup; started receipt prevents rescan if final persistence is fenced |
| Report publication fails | tool_report_failed; no rescan, receipt retains failure if writable |
| Terminal receipt persistence is ambiguous | tool_outcome_unknown, no success declaration or rescan |
| Same key, same completed input | replay existing result/ref, no new tool call |
| Same key, different input | tool_input_conflict, no launch |
| Existing started or malformed receipt | tool_outcome_unknown, no launch |

Failures use bounded fixed messages and retryable=false. Reports may contain
scanner evidence including sensitive target data under normal artifact access
rules; metrics/receipts do not contain argument values or output. Worker state
revisions and tool counts are updated through WorkerStateStore. Model counters
remain zero; no ADK finalizer/summarizer is constructed. Empty matches and
truncation never imply an absence of vulnerabilities.

## Verification matrix

V55-003 must exercise valid/invalid bindings and Go/Python/schema/digest parity;
no-gateway config resolution and placement with broken irrelevant model routes;
real A2A invocation with an artifact transport and controlled tool; successful
publication; missing capabilities; timeout/cancel/close; report/receipt failures;
in-memory and recreated-allocation replay; and changed-input rejection. Existing
ADK configuration, allocation and runtime suites must pass unchanged.
