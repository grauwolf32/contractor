# Deterministic scan request preparation

The preparation library `internal/scanplan` prepares scan requests from ordinary
Artifact inputs. It performs no network requests, scanner execution, credential
lookup or model calls. The caller supplies bytes read from an authorized exact
ArtifactRef; the library validates the ref and records a SHA-256 digest of the
supplied bytes. The caller remains responsible for matching those bytes to the
selected revision. Workflow dispatch and scanner selection belong to
[scan planning](32-scan-planning.md).

The opt-in `PrepareOperation` and `PrepareOperationTarget` helpers for one
assigned operation are described in [OpenAPI Audit scans](openapi-audit-scans.md#operation-preparation).
The whole-document policy and RequestSet v1 below are unaffected by them.

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

The preparation digest binds policy version 2, the exact source ref and bytes,
the input media type, and normalized preparation options. RequestSet artifacts
can contain credentials and private example data. Neither their canonical bytes
nor option values belong in logs. Diagnostic codes contain no supplied values;
JSON Pointers identify source locations, not resolved values.

`coverage.operations` counts discovered path operations; `prepared` counts
origins attached to retained requests, and `skipped` is their difference.
`complete` requires zero skipped operations and zero gaps. It describes extraction
of the documented one-sample policy. It does not certify schema validity, exhaustive
API coverage or the security of a target. Gaps contain `pointer` and a fixed `code`,
sorted by pointer then code and deduplicated. Invalid whole documents/options return
a fixed preparation error; they never return a successful empty RequestSet.

## Input and selection policy

Preparation policy 2 accepts OpenAPI **3.0.x and 3.1.x**, as JSON or YAML
(`application/json`, `application/yaml`, `application/x-yaml`, `text/yaml`).
Other OpenAPI versions are rejected explicitly. JSON/YAML must contain one
UTF-8 mapping with string keys, no duplicate keys, YAML anchors/aliases,
nonstandard tags, nonfinite numbers or numbers outside the exact integer range
of IEEE 754 doubles. This prevents silent changes to large numeric examples.

Preparation extracts concrete HTTP data on a best-effort basis. Schema type,
format, dialect and validation constraints do not reject supplied values or
examples. Ignoring those validation keywords does not create a coverage gap.
Strict checks apply to source syntax, resource bounds and the resulting HTTP
representation. RequestSet keeps `schemaVersion: 1`; the changed extraction
policy is recorded by `preparationDigest`.

Only in-document `#/...` JSON Pointer references are resolved. The resolver
does not fetch files, URLs or external example values. Schema references are
followed when concrete hints are needed; an unused or broken schema reference
does not invalidate an explicit value. Unavailable hints can fall through to
other concrete values. If required data remains unavailable, only the affected
operation is skipped; unavailable optional inputs are omitted with a gap.
Unresolvable parameter definitions still skip the operation because their name,
location and requiredness cannot be established. Unused response schemas are
not traversed. Resolved reference siblings override referenced fields. Callbacks
and unknown operation metadata other than `x-` extensions are reported as gaps;
recognized fields can still produce a request.

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
without checking schema enum restrictions. Variable values cannot introduce new
`{...}` placeholders. The final server must be absolute ASCII HTTP(S), without
credentials, a fragment or query. Base paths are retained; the OpenAPI
path is appended without URL resolution or path cleaning.

Path-level parameters are overridden by operation parameters with the same
`in` and `name`. Header names and `header:name` bindings are compared
case-insensitively; duplicate bindings are rejected. Reserved `Accept`, `Content-Type`
and `Authorization` parameter definitions are ignored with an explicit gap.
Scalar strings, numbers and booleans use simple style for path/header and form
style for query/cookie, regardless of their declared schema type. Path parameters
are required even when their `required` flag is absent or false. Arrays, objects,
parameter content, `allowReserved: true` and other styles remain unsupported.
Optional unsupported parameters without explicit bindings are omitted with a gap. A
required input or explicitly bound value that cannot be serialized skips the
affected operation with a diagnostic. Path and query values are percent-encoded;
cookie and header values must satisfy their respective HTTP syntax. Invalid HTTP
syntax skips the operation, including when the offending input was optional.

Concrete value precedence is:

1. Explicit operation binding.
2. Parameter or media-type `example`.
3. First named `examples` entry with a concrete `value`, in lexicographic order.
4. Schema `example`.
5. Schema `examples[0]`.
6. Schema `default`.
7. Schema `const`.
8. Schema `enum[0]`.

Broken named example references and `externalValue` entries can fall through to
later entries or schema hints. Explicit `null` is a concrete value; its usability
depends on HTTP serialization, so it is accepted in a JSON body but not as a
scalar parameter or text body. Selecting a concrete hint does not retry lower
priority hints when that value cannot be serialized.

Without a direct hint, schema properties can supply an object. Synthesis omits
`readOnly` properties and fails when a required property has no concrete value.
Available object samples in `allOf` are merged in array order, later fields win,
and local property hints override the merged fields. Missing required data in an
`allOf` branch prevents that synthesis. If no object can be synthesized, `oneOf`
then `anyOf` branches are tried in array order for the first concrete sample;
branch uniqueness and schema intersection are not validated. A supplied object
retains its fields. No placeholder strings, random values or arbitrary values
inferred from types are generated.

Bodies use the same concrete-value precedence. Supported output media types are
`application/json` and `text/plain`. An explicit body binding can supply either
type even when its definition is absent, broken or does not declare that media
type; the deviation receives a gap. Without a binding, supported declared media
types are tried in lexicographic order until usable body data is found. JSON is
canonicalized deterministically; plain text must be a string. The matching
content-type header is added. Missing or unsupported optional bodies are omitted
with a gap, while unavailable required bodies skip the affected operation.
An unusable explicit body binding also skips that operation. Multipart and
binary output formats are not synthesized. A schema's `format` annotation does
not prevent use of a concrete JSON or text value. Bodies can be prepared for all
supported methods, including GET, HEAD and DELETE.

Operation `security` overrides root security, including an empty array that
disables authentication. Requirement objects are OR alternatives in document
order; scheme names within each alternative are AND requirements. The first
fully satisfiable alternative is selected. An empty object allows anonymous
access. Preparation supports apiKey in header/query/cookie and HTTP bearer. Values
must come from the explicit authentication binding for that scheme; declared
examples are never credentials. HTTP bearer bindings contain the raw token,
without the `Bearer ` prefix. Unsupported schemes, missing values or collisions
with ordinary parameters cannot silently create an unauthenticated request.

## Bounds

Source bytes are limited to 2 MiB, options to 256 KiB, parsed document depth to
64 and nodes to 100,000. There are at most 1,000 path operations and 4,096 local
reference resolutions across preparation, with reference depth at most 32.
Concrete-hint extraction additionally shares a 100,000-node work budget and
depth limit of 64 so repeated references cannot expand into an unbounded tree.
Expanded concrete values are measured before JSON encoding to bound amplification
from repeated references; serialized bodies must still fit the 64 KiB wire limit.
`MaxRequests` defaults to 1,000 and can only lower that bound. Additional distinct
requests receive a limit gap, while duplicates can still contribute provenance.
The RequestSet is at most 4 MiB with at most 4,096 gaps. An entry has a URL of at
most 8,192 ASCII bytes, a body of at most 64 KiB, and at most 64 headers; names
are at most 128 bytes, values 8,192 bytes and total header bytes 32 KiB.
Whole-document, reference-work and serialized-output bounds fail explicitly
rather than returning unaccounted partial data.
