# Deterministic scan request preparation

V55-007 defines a preparation library, `internal/scanplan`, for ordinary Artifact
inputs. It performs no network requests, scanner execution, credential lookup or
model calls. The caller supplies bytes read from an authorized exact ArtifactRef;
the library validates the ref and records a SHA-256 digest of the supplied bytes.
The caller remains responsible for matching those bytes to the selected revision.
Workflow dispatch and scanner selection belong to V55-008.

## RequestSet v1

The media type is `application/vnd.contractor.http-requests+json`. Its normative
JSON schema and fixture live in [api/scan/v1](../../api/scan/v1/README.md).
The object contains `schemaVersion: 1`, `source` (exact `artifact` and
`contentDigest`), `preparationDigest`, `requests`, `gaps` and `coverage`.
Arrays are always present, including when empty.

Each entry contains `id`, `contentDigest`, a `request` and sorted unique `origins`.
A request contains `method`, `url`, `headers` and UTF-8 text `body`. Headers use
lowercase names, sorted by name; duplicates are forbidden. An origin contains
the operation's JSON Pointer, such as `#/paths/~1pets~1{id}/get`.
The request digest is SHA-256 of RFC 8785 canonical JSON of the request alone,
formatted `sha256:<64 lowercase hex digits>`. Its ID is `request-<same hex>`.
Identical requests share one entry with all contributing origins. Entries sort
by ID. Provenance and gaps do not change the identity of the HTTP request.

These are neutral prepared HTTP inputs. They do not choose SQLMap test parameters
or declare scanner eligibility. A later planner must select parameters explicitly
and construct the six-field single-request artifact defined in
[01](01-agent-template.md#sqlmap-http-request-artifacts), including its additional
scanner-specific validation. Passing a RequestSet entry directly to SQLMap is
not supported.

The preparation digest binds policy version 1, the exact source ref and bytes,
the input media type, and normalized preparation options. RequestSet artifacts
can contain credentials and private example data. Neither their canonical bytes
nor option values belong in logs. Diagnostic codes contain no supplied values;
JSON Pointers identify source locations, not resolved values.

`coverage.operations` counts discovered path operations; `prepared` counts
origins attached to retained requests, and `skipped` is their difference.
`complete` requires zero skipped operations and zero gaps. This describes the
documented one-sample preparation policy, not exhaustive API coverage or the
security of a target. Gaps contain `pointer` and a fixed `code`, sorted by pointer
then code and deduplicated. Invalid whole documents/options return a fixed
preparation error; they never return a successful empty RequestSet.

## Input and selection policy

Version 1 accepts OpenAPI **3.0.0 through 3.0.4**, as JSON or YAML
(`application/json`, `application/yaml`, `application/x-yaml`, `text/yaml`).
Other OpenAPI versions are rejected explicitly. JSON/YAML must contain one
UTF-8 mapping with string keys, no duplicate keys, YAML anchors/aliases,
nonstandard tags, nonfinite numbers or numbers outside the exact integer range
of IEEE 754 doubles. This prevents silent changes to large numeric examples.
This preparer validates the subset it consumes; it is not a replacement for
a complete OpenAPI document validator.
Selection and serialization follow the
[OpenAPI 3.0.4 specification](https://spec.openapis.org/oas/v3.0.4.html).

Only in-document `#/...` JSON Pointer references are resolved. The resolver
does not fetch files, URLs or external example values. Cyclic, missing or
external references needed by an operation produce a gap and skip that
operation. Unused response schemas are not traversed. Reference siblings and
unsupported request schema constructs are reported instead of silently changing
their meaning. Callbacks are reported and never dispatched.

Operations are visited by path lexicographically, then by method lexicographically.
Supported methods are DELETE, GET, HEAD, OPTIONS, PATCH, POST and PUT. TRACE is
an explicit unsupported operation. There is at most one sample per operation.

`Options` supplies a global `Server`, `ServerVariables`, `Authentication`,
per-operation `Operations` and `MaxRequests`. Operation keys are the pointers
above. An `OperationInput` supplies `Parameters` keyed by `location:name` and
an optional `Body` containing `MediaType` and a JSON-compatible `Value`.
Unknown operation bindings fail validation. Bound values take precedence over
document examples; no values are read from environment variables.

Server precedence is explicit override, operation servers, path servers, root
servers. The first declared server is selected. Relative URLs need an explicit
absolute override. Server variables use supplied values or declared defaults,
with declared enum checks. Variable values cannot introduce new `{...}`
placeholders. The final server must be absolute ASCII HTTP(S), without
credentials, a fragment or query. Base paths are retained; the OpenAPI
path is appended without URL resolution or path cleaning.

Path-level parameters are overridden by operation parameters with the same
`in` and `name`. Header names and `header:name` bindings are compared
case-insensitively; duplicate bindings are rejected. Reserved `Accept`, `Content-Type`
and `Authorization` parameter definitions are ignored with an explicit gap.
The first version supports scalar string/number/integer/boolean parameters using simple style for
path/header and form style for query/cookie. Arrays, objects, parameter content,
allowReserved and other styles produce explicit gaps. Path and query values are
percent-encoded; cookie and header values are validated for their respective
syntax. An explicit binding wins over `example`, then the lexicographically
first named `examples` entry, then `schema.example`. Schema defaults and enum
members are never used as invented values. Required missing values skip the
operation; optional missing values are omitted with a coverage gap. Examples
are checked against their supported schema before serialization.
Bounded `allOf`, `anyOf`, `oneOf` and `not` validation is supported; compositions
never invent or merge examples. Other JSON Schema dialects are unsupported.

Bodies use explicit supplied data or media-type examples with the same named
example ordering, then schema examples. Supported body media types are
`application/json` and `text/plain`; unsupported media types produce a gap.
An explicit media type must be declared; otherwise the first supported declared
type in lexicographic order is selected. JSON is canonicalized deterministically;
plain text must be a string. The matching content-type header is added. Missing
required bodies skip the operation; missing optional bodies are omitted with
a gap. Binary and multipart bodies are not synthesized.
Operations declaring bodies on GET, HEAD or DELETE are skipped explicitly:
OpenAPI 3.0 consumers cannot rely on those body definitions.

Operation `security` overrides root security, including an empty array that
disables authentication. Requirement objects are OR alternatives in document
order; scheme names within each alternative are AND requirements. The first
fully satisfiable alternative is selected. An empty object allows anonymous
access. Version 1 supports apiKey in header/query/cookie and HTTP bearer. Values
must come from the explicit authentication binding for that scheme; declared
examples are never credentials. HTTP bearer bindings contain the raw token,
without the `Bearer ` prefix. Unsupported schemes, missing values or collisions
with ordinary parameters cannot silently create an unauthenticated request.

## Bounds

Source bytes are limited to 2 MiB, options to 256 KiB, parsed document depth to
64 and nodes to 100,000. There are at most 1,000 path operations and 4,096 local
reference resolutions across preparation, with reference depth at most 32.
Schema expansion additionally shares a 100,000-node work budget so repeated
references cannot expand into an unbounded tree.
`MaxRequests` defaults to 1,000 and can only lower that bound. Additional distinct
requests receive a limit gap, while duplicates can still contribute provenance.
The RequestSet is at most 4 MiB with at most 4,096 gaps. An entry has a URL of at
most 8,192 ASCII bytes, a body of at most 64 KiB, and at most 64 headers; names
are at most 128 bytes, values 8,192 bytes and total header bytes 32 KiB.
Whole-document, reference-work and serialized-output bounds fail explicitly
rather than returning unaccounted partial data.
