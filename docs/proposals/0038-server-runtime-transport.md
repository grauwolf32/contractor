# Server–Runtime transport and one wire IDL

Proposal 0038 · Status: **Draft; no decision taken.**

On 2026-10-23 the user asked whether the Server ↔ Runtime Agent interaction
should move to gRPC. This proposal records the current boundary, what gRPC
would and would not change, the alternatives, a recommendation and the
conditions under which the transport question should be reopened. It creates
no implementation commitment. Figures were measured on `main @ af9fd522`.

The public API (`/v1`, [OpenAPI](../../api/openapi/contractor-public-v1.yaml)),
the Runtime's calls to the LLM Gateway and OTLP export are out of scope.

## Current boundary

Every channel uses mTLS ([security](../spec/platform/security.md)).

| Direction | Operations | Transport | Owner |
| --- | --- | --- | --- |
| Runtime → Server, control | `POST /private/v1/agents/register`, `POST /private/v1/agents/{instanceID}/heartbeat` | HTTP + JSON | `internal/controlplane/http.go` |
| Runtime → Server, allocation services | artifacts list, get, metadata and put (`If-Match`/`If-None-Match`, bodies up to 64 MiB), finding proposals, gateway recovery | HTTP + JSON, binary bodies | `internal/httpapi/privateartifacts` |
| Server → Runtime, allocation lifecycle | prepare, finalize, abort, release, `GET …/agent-state` | HTTP + JSON | `internal/controlplane/runtime_client.go`, `runtime/src/contractor_runtime/server.py` |
| Planner → Worker | `StageContentRequest` in, `WorkerCompletion` out, as A2A DataParts; the Planner polls `GetTask` with an adaptive interval of at most 2 s ([PLN-R114](../spec/execution/planners.md)) | A2A JSON-RPC (`a2a-go` v2.5.0, `a2a-sdk` 1.1.2) | `internal/planner/a2a`, `runtime/src/contractor_runtime/a2a_server.py` |

The private contract ([CTL-R6–R10, CTL-R149, CTL-R152,
CTL-R153](../spec/execution/control-plane.md)):

- 16 JSON Schemas in [`api/v1alpha1`](../../api/v1alpha1/), 3,539 lines, the
  canonical shapes (CTL-R8);
- hand-written strict models on both sides: `internal/contracts` and its six
  subpackages, 5,797 lines of Go, and `contractor_runtime/contracts`, 3,088
  lines of Python;
- shared valid, invalid and generated cases in
  [`api/testdata/v1alpha1`](../../api/testdata/v1alpha1/) (required members,
  forbidden nulls, mutations, blank text), run on both sides by
  `make test-wire-cross-language`.

Message rate is low: registration, heartbeats and a handful of lifecycle calls
per Stage. Performance does not motivate a transport change.

## Why the question comes up

The boundary's defect history has two kinds of drift between the Go and Python
models:

- **Shape:** Go read missing required members as zero values (V368-002),
  duplicate keys and case aliases were accepted (V368-007), explicit nulls of
  optional members were accepted (V371-003), the Worker State endpoint served
  bodies its schema rejects (V371-002).
- **Value:** blank-text and byte counting differed (V368-008), the capability
  grammar had three copies (V367-007), diagnostic byte bounds differed
  (V372-027), integers above 2^53 cannot cross the A2A DataPart exactly
  (V371-004, open; [WRK-R164](../spec/execution/worker.md)).

Generating both models from one description removes the first kind by
construction. The second kind needs value rules that both languages evaluate
the same way, whatever the transport.

## What gRPC would change

| Concern | gRPC with binary protobuf | Today |
| --- | --- | --- |
| Model duplication | Go and Python types generated from one `.proto` | Schemas plus two hand-written models |
| Required members | proto3 has no required members; presence exists only for `optional` fields, so every requirement stays a hand-written check | Strict decoder rejects missing members (CTL-R149) |
| Unknown members | Kept as unknown fields, not rejected | Rejected (CTL-R7) |
| Enums | Open: an unknown numeric value is preserved | Closed |
| Value rules (UTF-8 byte bounds, blank text, ID grammars, ranges) | Not expressible in the IDL; needs protovalidate (CEL rules) in Go and Python, or hand-written code | JSON Schema with `x-maxBytes`, per-language code, shared cases |
| 64-bit integers | Exact | Exact on the HTTP channels; not across the A2A DataPart |
| Deadlines and cancellation | Propagated with the call | Per-call HTTP timeouts on each side |
| Streaming | Server and bidirectional streams | None; heartbeat and Worker State are request/response, A2A results are polled |
| Artifact transfer | Chunked streams, precondition fields of our own, 4 MiB default message limit | Native HTTP: binary bodies, ETags and conditional requests |
| Debugging and tests | Needs grpcurl or reflection; raw-HTTP tests rewritten | curl; tests drive raw HTTP |
| Runtime dependencies | `grpcio` with native wheels, which is not in the Runtime lock today; the Runtime's resident memory is tracked ([runtime memory](../operations/runtime-memory-2026-09-19.md)) | None new |

The Planner ↔ Worker channel stays A2A in every option. The A2A specification
defines a gRPC binding and both SDKs in use ship one (`a2a-go` package
`a2agrpc`, `a2a-sdk` extra `grpc`), but a DataPart remains a protobuf
`Struct` or `Value`, whose numbers are doubles. Moving A2A to gRPC
therefore does not lift the 2^53 limit of V371-004; carrying the
`WorkerCompletion` as bytes, or bounding its integers, does.

## Options

**A. gRPC for control and allocation services, A2A unchanged.** Two RPC
stacks on one boundary. Removes model duplication for the control messages
only, rewrites both servers, clients, artifact transfer and the process tests,
and still needs value validation and an unknown-field policy.

**B. gRPC everywhere, including the A2A gRPC binding.** One transport, but the
largest rewrite, and it depends on both A2A SDKs' gRPC bindings for the
Planner path. The A2A payload limits above remain.

**C. One IDL, transport unchanged.** Keep HTTP + JSON and the routes; generate
the Go and Python models from one source.

- **C1: JSON Schema as the source.** The schemas are already canonical (CTL-R8).
  Generate the Python and Go types and their structural validators; `maxLength`,
  `pattern` and `enum` map to generated checks, `x-maxBytes` and the blank-text
  rule need a generator extension or remain shared hand-written functions held
  by the existing cases.
- **C2: protobuf as the source, ProtoJSON on the wire.** The protobuf runtime
  is already loaded in both processes (`google.golang.org/protobuf` is a direct
  Go dependency, `protobuf` comes with `a2a-sdk`), so C2 adds no runtime
  dependency. Python `json_format` (protobuf 6.33.6 in the Runtime lock)
  rejects unknown members and duplicate keys by default, but it accepts the
  original proto field name as an alias of the JSON name and treats `null` as
  an absent member; the ProtoJSON mapping specifies the same for Go. Both break
  CTL-R149, so C2 still needs the strict JSON pre-pass (`strictjson`, the
  Runtime codec) in front of the generated decoder. ProtoJSON writes 64-bit
  integers as JSON strings, which changes the wire format of every such member.

**D. Status quo.** Keep the hand-written models, guarded by the shared cases
and the cross-language gate that V368–V372 completed.

## Recommendation

Keep the transport (D) and run C as a bounded experiment. Do not adopt A or B
now:

- most open defects at this boundary are value rules, which gRPC does not
  check, and proto3's defaults (no required members, open enums, kept unknown
  fields) are looser than the project's strictness rules;
- the A2A channel and its payload limits stay as they are under A, and B only
  moves them to another binding;
- artifact transfer relies on HTTP semantics that gRPC would have to rebuild;
- the strict JSON decoders and the cross-language gate were completed in
  V368–V372; replacing them now discards that work for a boundary with no load
  problem.

## Experiment for option C

Create a task before starting (`tasks/templates/chore.yml`, epic
`api-contracts`) and compare C1 and C2 on one schema that has required,
optional and nullable members, an enum, a map, byte-bounded strings and
integers, for example [`allocation.schema.json`](../../api/v1alpha1/allocation.schema.json).

Measure, for each variant:

- hand-written model lines removed on each side;
- whether every case of `required-member-cases.json`, `mutation-cases.json`
  and `blank-text-cases.json` for that schema still gets the same verdict from
  the schema, the Server and the Runtime;
- files touched to add one optional member end to end, before and after;
- Runtime resident memory at startup, measured as in the runtime memory report.

Adopt the better variant for all 16 schemas only if it removes the
hand-written models without weakening any shared case. Otherwise record the
result here and keep D.

## Decision record

| Item | Content |
| --- | --- |
| Guarantee kept | One strict private contract: Server and Runtime reach the same verdict on every shared case (CTL-R149, CTL-R152, CTL-R153), and a change of the contract needs compatible builds (CTL-R10). |
| Alternatives | A, B, C1, C2, D above. |
| Evidence | Channel inventory and line counts on `af9fd522`; defect tasks V367-007, V368-002, V368-007, V368-008, V371-002, V371-003, V371-004, V372-027; `json_format` behaviour read from the locked protobuf 6.33.6; `grpcio` absent from the Runtime lock. |
| Reversal condition | Any trigger below. |

## Reopening the transport question

Owner: `execution-core` (Control Plane), with `planner-worker` for A2A and
`api-contracts` for the wire contract. Reopen A or B when one of these holds:

1. **Runtime Agents must run where the Server cannot connect to them**: behind
   NAT, in another network or at a customer site. Today the Server calls each
   Runtime for the allocation lifecycle and Worker State, and the Planner calls
   each Worker over A2A, so every Runtime must accept inbound connections. A
   Runtime-initiated bidirectional stream would carry the Server's commands
   over the connection the Runtime opened. This is the strongest trigger.
2. **Scale:** tens to hundreds of Runtime Agents, or live Worker state that must
   be pushed instead of polled.
3. **A third language** implements the private boundary.
4. **Both A2A SDKs in use make the gRPC binding their primary transport**, so
   that staying on JSON-RPC becomes the costly path.
